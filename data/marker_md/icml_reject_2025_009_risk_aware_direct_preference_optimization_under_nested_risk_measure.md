011

014 015 016

018

024

026

034

036

038

# Risk-aware Direct Preference Optimization under Nested Risk Measure

Anonymous Authors<sup>1</sup>

# Abstract

When fine-tuning pre-trained Large Language Models (LLMs) to align with human values and intentions, the pursuit of maximizing the estimated reward can lead to superior performance, but it also introduces potential risks due to deviations from the original (reference) model's intended behavior. Most existing methods for aligning LLMs typically introduce KL divergence to constrain deviations between the training model and the reference model; however, this may not be sufficient in certain applications that require tight risk control. In this paper, we introduce Riskaware Direct Preference Optimization (Ra-DPO), a novel approach that incorporates risk-awareness by employing a token-level objective function under nested risk measure. This method formulates a constrained risk-aware advantage function maximization problem and then converts the Bradley-Terry model into a token-level representation. The ultimate objective function maximizes the likelihood of the policy while suppressing the deviation between a training model and the reference model using a sequential risk ratio, thereby enhancing the model's risk-awareness during the process of aligning LLMs. The proposed method's effectiveness is verified via three open-source datasets: IMDb Dataset, Anthropic HH Dataset, and AlpacaEval, and the results demonstrate superior performance of our method in balancing alignment performance and model drift.

# 1. Introduction

With the advanced and rapid developments of large language models (LLMs) technology, learning from human feedback, serving as a bridge in aligning LLMs with human values and intentions, has become increasingly crucial [\(Ouyang](#page-9-0) [et al.,](#page-9-0) [2022;](#page-9-0) [Bai et al.,](#page-8-0) [2022;](#page-8-0) [Touvron et al.,](#page-9-1) [2023;](#page-9-1) [Bider-](#page-8-1) [man et al.,](#page-8-1) [2023\)](#page-8-1). Reinforcement Learning from Human Feedback (RLHF), which typically involves supervised finetuning, reward model training, and further fine-tuning of policy models via reinforcement learning (RL) algorithms, demonstrates impressive capabilities across diverse tasks and has emerged as a concrete research agenda [\(Christiano](#page-8-2) [et al.,](#page-8-2) [2017;](#page-8-2) [Ouyang et al.,](#page-9-0) [2022;](#page-9-0) [Yuan et al.,](#page-9-2) [2023\)](#page-9-2). A criticized downside is that RLHF has a complex process that requires considerable memory and careful hyperparameter tuning to maintain the stability of RL training.

Direct Preference Optimization (DPO) [\(Rafailov et al.,](#page-9-3) [2023\)](#page-9-3), featuring a simple and straightforward training process, directly uses the likelihood of the policy to define an implicit reward fitted to the preference data, which has emerged as a popular alternative since it bypasses key challenges in explicit reward modeling and achieves notable efficiency and competitive performance. Nevertheless, some studies [\(Xiao et al.,](#page-9-4) [2024;](#page-9-4) [Wang et al.,](#page-9-5) [2024b\)](#page-9-5) have reported that DPO still suffers from issues such as excessively long generative responses and the significant KL divergence of the dispreferred response subset. To tackle these issues, numerous variants of DPO have been successively proposed, including f-DPO [\(Wang et al.,](#page-9-6) [2024a\)](#page-9-6), IPO [\(Azar et al.,](#page-8-3) [2024\)](#page-8-3), RDPO [\(Fisch et al.,](#page-8-4) [2024\)](#page-8-4), and SimPO [\(Meng et al.,](#page-9-7) [2024\)](#page-9-7), which introduce length control mechanisms or enhance KL divergence constraints. However, a key limitation is that these methods only consider evaluation at the sentence level, ignoring the fact that the generation of these responses occurs sequentially, following an auto-regressive approach.

Recently, a fresh perspective on LLMs alignment has been introduced, specifically the sequential and token-level direct preference optimization, known as TDPO [\(Zeng et al.,](#page-9-8) [2024\)](#page-9-8), which allows for examining divergence in relation to a reference LLM on a more granular, token-by-token basis. Specifically, inspired by Trust Region Policy Optimization (TRPO) [\(Schulman et al.,](#page-9-9) [2015\)](#page-9-9) in RL field, TDPO redefines the objective of maximizing restricted rewards in a sequential manner and establishes the connection between sentence-level reward and token-level generation through using the Bellman equation. However, since the objective at each step is to maximize the expected return, a risk-neutral criterion, which neglects the characteristics of the reward distribution beyond the mean, TDPO encounters the same

<sup>1</sup>[Anonymous Institution, Anonymous City, Anonymous Region,](#page-8-1) [Anonymous Country.](#page-8-1)

[Preliminary work. Under review by the International Conference](#page-8-1) [on Machine Learning \(ICML\). Do not distribute.](#page-8-1)

058

071

074

076

078

087 088

090 091

093 094

096

098

100

104

106

108 109 challenges as classic RL algorithms [\(Schulman et al.,](#page-9-9) [2015;](#page-9-9) [2017;](#page-9-10) [Bisi et al.,](#page-8-5) [2022\)](#page-8-5).

Fortunately, in the field of RL, a series of risk-sensitive methods [\(Bisi et al.,](#page-8-5) [2022;](#page-8-5) [Candela et al.,](#page-8-6) [2023\)](#page-8-6) have been proposed, which achieve superior performance by introducing various risk measure functions. Recently, some researchers have attempted to introduce this technology in order to align LLMs with human preferences. For instance, RA-RLHF [\(Chaudhary et al.,](#page-8-7) [2024\)](#page-8-7) introduces Conditional Value at Risk (CVaR) [\(Artzner,](#page-8-8) [1997\)](#page-8-8), a static risk measure function, into the fine-tuning of RL, while KTO [\(Ethayarajh](#page-8-9) [et al.,](#page-8-9) [2024\)](#page-8-9) introduces prospect theory [\(Tversky & Kahne](#page-9-11)[man,](#page-9-11) [1992\)](#page-9-11) to fit human choice behavior when faced with uncertain events. However, these methods only analyze the risk of the whole prompt-response at the sentence level by considering the distribution characteristics of the preference data, which neglects the fact that the generation of these responses occurs sequentially, following an auto-regressive approach.

In this paper, we focus on the risk in the value iteration at each step by introducing nested risk measures. Specifically, we investigate a novel direct preference optimization method for the problem of aligning with human preferences from a risk-sensitive perspective and provide corresponding theoretical and empirical results. Our main contributions are summarized as follows.

- We propose a novel Risk-aware Direct Preference Optimization (Ra-DPO) method. This method maximizes the likelihood of the policy while effectively suppressing the deviation between the training model and the reference model by means of a sequential risk ratio, thereby enhancing the model's risk-awareness during the process of balancing alignment performance and model drift.
- We design a new risk-aware token-level objective function by reformulating the constrained reward maximization problem into a token-level form, and then prove that maximizing the objective function will result in policy improvements. Furthermore, by establishing equivalence between the Bradley-Terry model and the Regret Preference Model and deriving the mapping between the risk-aware state-action value function and the optimal policy, we obtain the optimization objective that is solely related to the risk-sensitive policy.
- Experimentally, we provide the results across various text generation tasks to evaluate the effectiveness of our proposed method and the sensitivity to the risk control parameter. The experimental results demonstrate that our method can effectively suppress the risk of model drift while enhancing its performance.

#### 2. Preliminaries

#### 2.1. Preference-based Policy Optimization

Considering a preference-based language model fine-tuning task, let x denote an input prompt (question), and y denote the generated response (answer). The notation y<sup>w</sup> ≻ y<sup>l</sup> | x symbolizes the human preference data, where y<sup>w</sup> (win) represents a response that is more preferred by humans compared to y<sup>l</sup> (lose). Both x and yw/y<sup>l</sup> consist of a sequence of tokens.

Bradley-Terry Model. In the preference-based fine-tuning process, to align with human preferences, a preference predictor adhering to the Bradley-Terry (BT) [\(Bradley & Terry,](#page-8-10) [1952\)](#page-8-10) model has been widely employed for pairwise comparisons. The likelihood of a preference pair is commonly expressed using a latent reward model:

$$P_{\text{BT}}(y_w \succ y_l \mid x) = \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))}, \quad (1)$$

where r(x, yw) and r(x, yl) stand for the reward function at the sentence level from the preferred and dispreferred answers, respectively.

Directly Preference Optimization. Direct Preference Optimization (DPO) [\(Rafailov et al.,](#page-9-3) [2023\)](#page-9-3) commences with the following RL objective:

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot | x)} [r(x, y) - \beta D_{\text{KL}}(\pi_\theta(\cdot | x) || \pi_{\text{ref}}(\cdot | x))], \quad (2)$$

where D represents the human preference dataset, β is the coefficient of the reverse KL divergence penalty, πref (· | x) is the policy of fixed reference model (typically selected to be the model that has undergone post-supervised finetuning), and π<sup>θ</sup> (· | x) represents the policy of the trained model, initialized with π<sup>θ</sup> = πref.

By reparameterizing the reward function in Eq. [2](#page-1-0) using the policy in a supervised manner, DPO establishes a direct functional mapping between the reward model and the optimal policy.

$$r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x), \quad (3)$$

where Z(x) is the partition function or the normalizing constant.

Then, by plugging the reward from Eq. [3](#page-1-1) into the BT model in Eq. [1,](#page-1-2) DPO derives the objective function:

$$\mathcal{L}_{\text{DPO}} (\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} [\log \sigma (u (x, y_w, y_l))], \quad (4)$$

where

$$u(x, y_w, y_l) = \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)}.$$

114 115 116

118

124

126

128

131

134

136

138

151

154

158

160

164

#### 2.2. Preference-based Markov Decision Process

A Preference-based Markov Decision Process (Pb-MDP) can be formulated as a modification of the classical MDP: M⟨S, A, r, P, γ, T⟩, where S and A represent the finite state and action spaces, respectively; P : S × A → S is the probabilistic transition function; r represents the reward function of the entire prompt-response, which is defined as (S × A) <sup>T</sup> → R; γ is the discount factor, and T denotes the length of a trajectory or episode.

Specifically, for language generation, the state s<sup>t</sup> = [x, y<t] ∈ S is a combination of the prompt and the generated response up to the current step, action a<sup>t</sup> = y <sup>t</sup> ∈ A corresponds to the next generated token, and the token-wise reward is defined as R<sup>t</sup> := R (st, at) = R ([x, y<t] , y<sup>t</sup> ). Additionally, note that y <sup>&</sup>lt;<sup>1</sup> = [ ] is an empty sequence. Therefore, we denote [x] = [x, [ ]] = x, y<sup>&</sup>lt;<sup>1</sup> . For a given prompt x and the first t − 1 tokens y <t of the response y, we define the probability distribution of the next token conditioned on [x, y<t] as π<sup>θ</sup> (· | [x, y<t]).

#### 2.3. Risk Measure

It is more desirable to keep risk under control for language generation tasks instead of only considering a risk-neutral criterion, which overlooks the distribution characteristics of rewards, especially on certain safety-critical tasks that may have potential broad societal impact. Therefore, we introduce the risk-sensitive criterion [\(Bauerle & Rieder](#page-8-11) ¨ , [2014;](#page-8-11) [Wang & Chapman,](#page-9-12) [2022\)](#page-9-12) to quantify the hidden risk. More specifically, the definition of the quantile function and risk measure objective are as follows.

The quantile function is the coherent risk-measure [\(Artzner](#page-8-12) [et al.,](#page-8-12) [1999;](#page-8-12) [Bonetti et al.,](#page-8-13) [2023\)](#page-8-13) of random variable Z,

$$F_Z^{-1}(\xi) = \inf \{z \in \mathbb{R} \mid F_Z(z) \geq \xi\} ,$$

which satisfies the following properties for all Z, Z′ ∈ Z: *Concavity:* ∀ λ ∈ [0, 1] : η (λZ + (1 − λ)Z ′ ) ≥ λη (Z)+ (1 − λ) η (Z ′ ); *Monotonicity:* If Z ≥ Z ′ , then η(Z) ≥ η (Z ′ ); *Translation Equivariance:* ∀ ϵ ∈ <sup>R</sup> : η (Z + ϵ) = η (Z) + ϵ; *Positive Homogeneity:* ∀ λ > 0 : η (λZ) = λη (Z). Then, we introduce the nested risk-measures that are built upon Pb-MDP in Subsection [2.3.](#page-2-0)

Nested risk-measures. In the context of standard Pb-MDP, the nested quantile risk measures [\(Fei et al.,](#page-8-14) [2020;](#page-8-14) [Chen](#page-8-15) [et al.,](#page-8-15) [2024;](#page-8-15) [Zhao et al.,](#page-9-13) [2024\)](#page-9-13) can be elucidated in Bellman equation type as follows:

$$\begin{cases} Q_\pi ([x, y^{$$

where Q<sup>π</sup> ([x, y<t] , y<sup>t</sup> ) and V<sup>π</sup> ([x, y<t]) represent the state-action value and state value under the nested risk measures at timestep t ∈ [1, · · · , T], respectively. Φ(·) is a nested risk measure function with a risk control parameter µ. For any random variable Z, we have

$$\Phi^\mu(Z) = \int_0^1 F_Z^{-1}(\xi) dG(\xi),$$

where G is a weighting function over the quantiles.

This class captures a broad range of useful objectives, including the popular CVaR [\(Artzner,](#page-8-8) [1997\)](#page-8-8) objective. Due to space constraints, we provide a detailed survey about risk measure in Appendix [A.1](#page-10-0) and the expanded version of value function definition in Appendix [A.2.](#page-11-0)

# 3. Methodology

This section proposes a novel language model alignment method called Risk-aware Direct Preference Optimization (Ra-DPO). Specifically, we first conduct an analysis of the characteristics of nested risk measures and design a new risk-aware token-level objective function by reformulating the constrained reward maximization problem into a tokenlevel form. Subsequently, we prove that maximizing the objective function will result in policy improvements. Then, the optimization objective solely related to the risk-sensitive policy is obtained by deriving the mapping between the risk-aware state-action function and the optimal policy; and establishing BT model equivalence with the Regret Preference Model. Finally, we conduct a formalized analysis of this optimization objective in terms of derivatives and derive the loss function for Ra-DPO.

#### 3.1. Risk-aware Objective Function

In this subsection, we aim to design a new risk-aware objective function for preference-based language model finetuning. Unfortunately, although the recursive Bellman equation under nested risk measures was introduced in Subsection [2.3,](#page-2-0) it cannot be directly applied, mainly due to the following reasons:

(1) For the Pb-MDP setting, the algorithm can only obtain the reward (an implicit reward fitted to the preference data) at an entire prompt-response until the end and thus cannot compute the target value at each step.

(2) The nested risk-measures incorporate a Bellman-type recursion and are not law-invariant [\(Hau et al.,](#page-8-16) [2023\)](#page-8-16), which are complex and difficult to compute.

To surmount these obstacles, a straightforward approach is to introduce the state augmentation method, i.e., reconstructing an augmented Pb-MDP as described in [\(Zhao et al.,](#page-9-13) [2024\)](#page-9-13), where the state at each timestep includes historical trajectories. This method can reformulate the recursive

*174*

*181*

*183 184*

*190 191*

*200*

*204*

*206*

Bellman equation into a classical Bellman equation with augmented states. However, it is noteworthy that, in this paper, we directly define the state as a combination of the prompt and the generated response up to the current step to model the sequential and auto-regressive generation. It possesses a characteristic in that the state at the previous timestep is a subset of the state at the current timestep, i.e., x, y<t−<sup>1</sup> ⊂ [x, y<t]. Therefore, we can rewrite the nested quantile objective's Bellman equation in Eq. [5](#page-2-1) as follows:

$$\begin{cases} \tilde{Q}_\pi ([x, y^{$$

where Q˜ <sup>π</sup> ([x, y<t] , y<sup>t</sup> ) and V˜ <sup>π</sup> ([x, y<t]) represent the riskaware state value and state-action value under the policy π, respectively.

It is noteworthy that there is a significant difference in the calculation of the risk-aware state value function between Eq. [5](#page-2-1) and Eq. [6.](#page-3-0) And, according to the Lemma 3.6 in (Zhao et al., 2024), we can obtain the following lemma.

Lemma 3.1. *For a given Pb-MDP, the reward on the entire prompt-response can be decomposed as* P r = T <sup>t</sup>=1 γ <sup>t</sup>−<sup>1</sup>R ([x, y<t] , y<sup>t</sup> )*, the relationship between the state value function Eq. [5](#page-2-1) and Eq. [6](#page-3-0) is as follows:*

$$\tilde{V}_\pi \left( [x, y^{$$

*where* R1:t−<sup>1</sup> = P<sup>t</sup>−<sup>1</sup> <sup>h</sup>=1 γ <sup>h</sup>−<sup>1</sup>R x, y<h , y<sup>h</sup> *denotes the reward of the* 1 ∼ t − 1 *steps of a prompt-response, and* Vπ[x] *and* V˜ <sup>π</sup>[x] *are equivalent.*

The proof is detailed in Appendix [B.1.](#page-11-1)

Subsequently, based on the new risk-aware state value and state-action value in Eq. [6,](#page-3-0) we define the risk-aware advantage function as follows.

Definition 3.2. For a risk-sensitive Pb-MDP that satisfies the Bellman equation in Eq. [6,](#page-3-0) the risk-aware advantage function can be defined as

$$\tilde{A}_\pi \left( [x, y^{< t}], z \right) = \tilde{Q}_\pi \left( [x, y^{< t}], z \right) - \Phi^\mu (\tilde{V}_\pi \left( [x, y^{< t}] \right)), \quad (8)$$

where z subject to π<sup>θ</sup> (· | [x, y<t]).

The definition is reasonable, and the derivation provided in Appendix [B.2.](#page-12-0)

Furthermore, based on the definition of risk-aware advantage function in Definition [3.2,](#page-3-1) we propose a new risk-aware objective function:

$$\max_{\pi_{\theta}} \mathbb{E}_{x, y < t \sim \mathcal{D}, z \sim \pi_{\theta}(\cdot | x, y < t)} \left[ \tilde{A}_{\pi_{\text{ref}}} \left( [x, y^{< t}], z \right) \right. \\ \left. - \beta D_{\text{KL}} \left( \pi_{\theta} \left( \cdot \mid [x, y^{< t}] \right) \middle| \pi_{\text{ref}} \left( \cdot \mid [x, y^{< t}] \right) \right) \right]. \quad (9)$$

The objective function maximizes a risk-sensitive advantage function subject to a KL divergence constraint, which takes into account the risk when selecting the optimal policy, thereby achieving a better balance between alignment performance and model drift. Next, we prove that maximizing the risk-aware objective function in Eq. [9](#page-3-2) will result in policy improvements, as stated in the following lemma.

Lemma 3.3. *Given two policies* π *and* π ′ *, if for any state* s<sup>t</sup> = [x, y<t] , <sup>E</sup>z∼π′ h A˜ <sup>π</sup> ([x, y<t] , z) i ≥ 0*, then we can conclude:*

$$\mathbb{E}_{x \sim \mathcal{D}} \left[ \tilde{V}_{\pi'}([x]) \right] \geq \mathbb{E}_{x \sim \mathcal{D}} \left[ \tilde{V}_{\pi}([x]) \right]. \quad (10)$$

The proof is provided in Appendix [B.3.](#page-13-0)

## 3.2. Risk-aware Preference Optimization

In this subsection, we focus on how to convert the BT model into risk-sensitive token-level representation to obtain the optimization objective that is solely related to the risksensitive policy, which is divided into two steps: (1) derive the mapping between the risk-aware state-action function and the optimal policy; (2) establish BT model equivalence with the Regret Preference Model.

Specifically, starting from the risk-aware token-level objective function in Eq. [9,](#page-3-2) we first derive the mapping between the risk-aware state-action function Q˜ <sup>π</sup> and the optimal policy π ∗ θ , as stated in the following lemma.

Lemma 3.4. *The constrained problem in Eq. [9](#page-3-2) has the closed-form solution:*

$$\begin{aligned} & \pi_{\theta}^* (z \mid [x, y^{$$

*where*

$$Z \left( [x, y^{< t}] ; \beta \right) = \mathbb{E}_{z \sim \pi_{\text{ref}}(\cdot | [x, y^{< t}])} e^{\frac{1}{\beta} \tilde{Q}_{\pi_{\text{ref}}}([x, y^{< t}], z)},$$

*which is the partition function.*

The proof is provided in Appendix [B.4.](#page-13-1) Then, by rearranging Eq. [11,](#page-3-3) we can obtain the expression of the risk-aware state-action function in terms of the policy

$$\begin{aligned} & \hat{Q}_{\pi_{\text{ref}}} \left( [x, y^{< t}], z \right) \\ &= \beta \log \frac{\pi_{\theta}^*(z \mid [x, y^{< t}])}}{\pi_{\text{ref}}(z \mid [x, y^{< t}])} + \beta \log Z \left( [x, y^{< t}]; \beta \right). \end{aligned} \quad (12)$$

Subsequently, by utilizing the reward decomposition formula r = P<sup>T</sup> <sup>t</sup>=1 γ <sup>t</sup>−<sup>1</sup>R ([x, y<t] , y<sup>t</sup> ) from Lemma [3.1,](#page-3-4) we establish BT model equivalence with the Regret Preference Model, as shown in the following lemma.

226

228

231

234

236

238

254

256

258

260

264

266

268

271

274

Lemma 3.5. *Given a reward function* r *of the entire prompt-response, based on a relationship between tokenwise rewards and the reward function represented by* P r = T <sup>t</sup>=1 γ <sup>t</sup>−<sup>1</sup>R ([x, y<t] , y<sup>t</sup> )*, we can establish the equivalence between the Bradley-Terry model and the Regret Preference Model, i.e.,*

$$P_{\text{BT}}(y_1 \succ y_2 \mid x) = \sigma \left( \sum_{t=1}^{T_1} \gamma^{t-1} \tilde{A}_\pi([x, y_1^{$$

*where* σ(z) = 1/ (1 + exp(−z)) *is the logistic sigmoid function for any random variable* z*.*

The proof is provided in Appendix [B.5.](#page-14-0)

According to the definition of the risk-aware advantage function in Definition [3.2,](#page-3-1) we can directly establish the relationship between the optimal solution in Eq. [12](#page-3-5) and preference optimization objective in Eq. [13.](#page-4-0) In this way, we ultimately reformulate the BT model to be directly tied to the risk-aware optimal policy π ∗ θ and the reference policy πref, which is summarized in the following theorem.

Theorem 3.6. *Given prompts* x *and pairwise responses* (y1, y2)*, and the risk-aware objective function in Eq. [9,](#page-3-2) the Bradley-Terry model expresses the human preference probability in terms of the risk-aware optimal policy* π ∗ θ *and reference policy* πref*:*

$$P_{\text{BT}}^*(y_1 \succ y_2 \mid x) = \sigma(u^*(x, y_1, y_2) - \delta^*(x, y_1, y_2)), \quad (14)$$

*where* u (x, y1, y2) *represents the difference in implicit rewards defined by the risk-aware policy* π ∗ θ *and the reference policy* πref*, weighted by* β*, represented as*

$$u(x, y_1, y_2) = \beta \log \frac{\pi_\theta(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} - \beta \log \frac{\pi_\theta(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)}, \quad (15)$$

*and* δ (x, y1, y2) *represents the difference in sequential risk ratio between two pairs* (x, y1) *and* (x, y2)*, expressed as*

$$\delta(x, y_1, y_2) = \beta D_{\text{SeqRR}}(x, y_2; \pi_{\text{ref}} \mid \pi_\theta) - \beta D_{\text{SeqRR}}(x, y_1; \pi_{\text{ref}} \mid \pi_\theta), \quad (16)$$

*where*

$$D_{\text{SeqRR}}(x, y; \pi_{\text{ref}} \mid \pi_{\theta}) = \sum_{t=1}^T \Phi_{z \sim \pi_{\text{ref}}}^{\mu} \left( \log \frac{\pi_{\text{ref}}(z \mid x)}{\pi_{\theta}(z \mid x)} \right).$$

The proof is provided in the Appendix [B.6.](#page-15-0)

# 3.3. Loss Function and Formal Analysis

Drawing on Theorem [3.6,](#page-4-1) we reformulate the BT model into a structure solely relevant to the risk-sensitive policy, which enables us to formulate a likelihood maximization objective for a parametrized policy πθ, and then our loss function becomes:

$$\begin{aligned}\mathcal{L}_{\text{Ra-DPO}_1} & (\pi_\theta; \pi_{\text{ref}}) \\ &= -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} [\log \sigma(u(x, y_w, y_l) - \delta(x, y_w, y_l))].\end{aligned}\tag{17}$$

Through this approach, we explicitly introduce sequential risk ratio into the loss function, which incorporates riskawareness during the process of balancing alignment performance and model drift. To elucidate the benefit of the proposed method, we give further interpretation by analyzing the loss function and its gradient. Specifically, we conduct a derivative analysis of our method. For convenience, we use u to denote u (x, yw, yl), and δ to represent δ (x, yw, yl). By simple calculations, we can derive the gradient of the loss function in Eq. [17](#page-4-2) with respect to the parameters θ :

$$\begin{aligned}\nabla_{\theta}\mathcal{L}_{\text{Ra-DPO}_1}(\pi_{\theta}; \pi_{\text{ref}}) \\ = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} [(-u + \delta) [\nabla_{\theta} u - \nabla_{\theta} \delta]],\end{aligned}\tag{18}$$

where (−u + δ) serves as the weighting factor for the gradient.

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right) \right]$$
$$\mathcal{L}_{\text{TDPO}_2}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E} \left[ \log \sigma \left( \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right) - \alpha (\beta D_{\text{SeqKL}}(x, y_l; \pi_{\text{ref}} \| \pi_\theta) - \text{sg}(\beta D_{\text{SeqKL}}(x, y_w; \pi_{\text{ref}} \| \pi_\theta))) \right) \right]$$
$$\mathcal{L}_{\text{RA-DPO}_2}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E} \left[ \log \sigma \left( \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right) - \alpha (\beta D_{\text{SeqRR}}(x, y_l; \pi_{\text{ref}} \| \pi_\theta) - \text{sg}(\beta D_{\text{SeqRR}}(x, y_w; \pi_{\text{ref}} \| \pi_\theta))) \right) \right) \right]$$

Figure 1. Comparison of loss functions for DPO, TDPO<sup>2</sup> and Ra-DPO<sup>2</sup> methods. The sg denotes the stop-gradient operator.

From Eq. [18,](#page-4-3) we can observe that the first part (−u) corresponds to the weight factor in the first part of loss function of TDPO. Its value will increase when the language model makes prediction errors relative to human preferences, i.e., log <sup>π</sup>θ(yl|x) <sup>π</sup>ref (yl|x) <sup>&</sup>gt; log <sup>π</sup>θ(yw|x) πref (yw|x) . The second part δ consists of the difference between the sequential risk ratio of the dispreferred response subset and the preferred response subset, which is a distinctive component of our method. When selecting a convex function (risk-averse), such as CVaR, as the risk measure function, our method automatically balances the risk ratio.

Furthermore, based on a common starting point shared by our method and TDPO [\(Zeng et al.,](#page-9-8) [2024\)](#page-9-8), i.e., reducing risks stemming from model drift and ensuring training stability, we also provide the second version of our method,

278

289 290

294

296

298

300

304

306

308 309

311

314 315 316

318

324

326

![](_page_5_Figure_1.jpeg)

Figure 2. The experiment on the IMDb dataset with GPT-2 Large serving as the base model. Figure [2\(a\)](#page-5-0) and Figure [2\(b\)](#page-5-1) present the progression of sequential KL divergence (the lower the better) of both preferred response and dispreferred responses. Additionally, Figure [2\(c\)](#page-5-2) illustrates the reward accuracy curves (the higher the better).

Ra-DPO2. The loss function of Ra-DPO<sup>2</sup> is given by:

$$\begin{aligned}\mathcal{L}_{\text{Ra-DPO}_2} & (\pi_\theta; \pi_{\text{ref}}) \\ &= -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} [\log \sigma(u(x, y_w, y_l) - \alpha \delta_2(x, y_w, y_l))],\end{aligned}\tag{19}$$

where α is a parameter, and

$$\begin{aligned} \delta_2(x, y_1, y_2) &= \beta D_{\text{SeqRR}}(x, y_2; \pi_{\text{ref}} \mid \pi_{\theta}) \\ &\quad - \text{sg}(\beta D_{\text{SeqRR}}(x, y_1; \pi_{\text{ref}} \mid \pi_{\theta})). \end{aligned}$$

The sg represents the stop-gradient operator, which blocks the propagation of gradients. Ra-DPO<sup>2</sup> modifies the loss function of Ra-DPO<sup>1</sup> by discontinuing the gradient propagation of DSeqRR(x, yw; πref | πθ) and treating it as a baseline term for alignment of DSeqRR(x, y<sup>l</sup> ; πref | πθ). The aim of the modification is to ensure training stability, rather than accelerating the training speed.

To summarize, the comparison of the loss functions for DPO, TDPO2, and Ra-DPO<sup>2</sup> is shown in Figure [1.](#page-4-4) In addition, we give a procedure of our method, and provide its pseudocode (Algorithm [1\)](#page-16-0) in Appendix [B.7.](#page-16-1)

# 4. Experiments

We empirically evaluate our method via several open-source datasets and pre-trained models. Our experiments aim to answer the following questions: First, how does the performance of our method compare with existing methods, and is our method more sensitive to risks when tackling challenging text generation tasks? Second, how does the risk control parameter µ affect the performance of our method?

To answer these questions, we conduct experiments on IMDb Dataset [\(Maas et al.,](#page-9-14) [2011\)](#page-9-14), Anthropic HH Dataset [\(Bai et al.,](#page-8-0) [2022\)](#page-8-0) and AlpacaEval [\(Dubois et al.,](#page-8-17) [2024\)](#page-8-17) for three different text generation tasks. Based on the original *KTO implementation*[<sup>1</sup>](#page-5-3) , we trained Ra-DPO and the baseline

models using the same hyperparameters. Specifically, for Ra-DPO, we employed the popular CVaR [\(Artzner,](#page-8-8) [1997\)](#page-8-8) as the risk measure function. We compare our method against the following algorithms: (1) DPO [\(Rafailov et al.,](#page-9-3) [2023\)](#page-9-3), which only considers evaluation at the sentence level; (2) PPO [\(Schulman et al.,](#page-9-10) [2017\)](#page-9-10), which is an offline PPO variant provided by the original KTO implementation; (3) TDPO<sup>1</sup> and TDPO<sup>2</sup> [\(Zeng et al.,](#page-9-8) [2024\)](#page-9-8), which convert the BT model into token-level representation to obtain the optimization objective; (4) KTO [\(Ethayarajh et al.,](#page-8-9) [2024\)](#page-8-9), which considers humans make decisions that do not maximize their expected value when faced with uncertain events. All reported results of our algorithm and baseline algorithms are trained using 4 × A100 GPUs, each with 40GB of memory.

## 4.1. Experiments on IMDb Dataset

Experimental setup: The IMDb dataset is a controlled semantic generation dataset within the context of movie reviews, serving as a valuable resource for training and evaluating sentiment analysis models. We employ GPT-2 Large [\(Radford et al.,](#page-9-15) [2019\)](#page-9-15) as our base model and use the model checkpoint: *insub/gpt2-large-IMDb-fine-tuned*[<sup>2</sup>](#page-5-4) as the SFT model. In this setup, the model is presented with prompts consisting of prefixes from movie reviews, and is required to generate responses with positive sentiment. Specifically, we implement the versions of Ra-DPO<sup>1</sup> with risk control parameter µ ∈ {0.99, 0.98, 0.97, 0.95}. Moreover, in order to achieve a fair comparison, we calculate the sequential KL divergence for our method. Note that the risk ratio value is slightly larger than the KL divergence value when selecting CVaR (a convex function) as the risk measure function. The results are shown in Figure [2.](#page-5-5)

Evaluation: Figure [2](#page-5-5) shows that Ra-DPO<sup>1</sup> can outperform or achieve reward accuracy similar to the advanced TDPO algorithm while also maintaining a slight model drift (indicated by the lower KL divergence), demonstrating the

<sup>1</sup>Available at [https://github.com/ContextualAI/](https://github.com/ContextualAI/HALOs) [HALOs](https://github.com/ContextualAI/HALOs)

<sup>2</sup> https://huggingface.co/insub/gpt2-large-IMDb-fine-tuned

334

336

338

351

354

356

358

360 361

364

366

368

371

374

378

![](_page_6_Figure_1.jpeg)

Figure 3. The experiment on the Anthropic HH dataset with Pythia-1.4B serving as the base model. We implemented TDPO2, and different versions of Ra-DPO<sup>2</sup> with respect to the risk control parameter µ while keeping coefficient α constant at 0.5. Figure [3\(a\)](#page-6-0) and Figure [3\(b\)](#page-6-1) present the progression of sequential KL divergence (the lower the better) of both preferred response and dispreferred responses. Additionally, Figure [3\(c\)](#page-6-2) illustrates the reward accuracy curves (the higher the better).

![](_page_6_Figure_4.jpeg)

Figure 4. The reward accuracy of each algorithm on the Anthropic HH dataset, using Pythia-1.4B as the base model.

risk-awareness of Ra-DPO<sup>1</sup> during the process of balancing alignment performance and model drift.

## 4.2. Experiments on Anthropic HH Dataset

Experimental setup: Anthropic HH dataset contains 170k dialogues between a human and an automated assistant, where each transcript ends with a pair of responses generated by an LLM along with a preference label denoting the human-preferred response. We use Pythia-1.4B and Pythia-2.8B [\(Biderman et al.,](#page-8-1) [2023\)](#page-8-1) as the base models to test our method on Anthropic HH dataset, respectively. Here, the reference models are trained by fine-tuning the base models on chosen completions. Specifically, we implement TDPO<sup>2</sup> and different versions of Ra-DPO<sup>2</sup> with respect to the parameters µ and α The results are depicted in Figure [3,](#page-6-3) Figure [4,](#page-6-4) and Appendix [C.1.](#page-16-2)

Evaluation: Figure [3](#page-6-3) shows the performance of TDPO2, and different versions of Ra-DPO<sup>2</sup> with respect to the risk control parameter µ while keeping coefficient α constant at

Table 1. AlpacaEval compares the responses generated by Algorithms DPO, PPO, KTO, TDPO1, TDPO<sup>2</sup> (α = 0.5), Ra-DPO<sup>1</sup> (µ = 0.97), and Ra-DPO<sup>2</sup> (α = 0.5, µ = 0.97) with those generated by *gpt4 1106 preview*. The winrate and length-controlled winrate (Lc winrate) are evaluated based on *oasst pythia 12b*.

| M ETHOD  | W      | INRATE | L C  | WINRATE |
|----------|--------|--------|------|---------|
| DPO      | 51.1   | ± 1.9  | 44.7 | ± 0.4   |
| PPO      | 52.1   | ± 1.8  | 51.9 | ± 0.5   |
| KTO      | 51.5   | ± 1.8  | 50.2 | ± 0.6   |
| TDPO 1   | 51.9   | ± 1.8  | 53.0 | ± 0.6   |
| TDPO 2   | 52.2   | ± 1.6  | 52.2 | ± 0.5   |
| R A -DPO | 1 53.5 | ± 1.8  | 53.9 | ± 0.5   |
| R A -DPO | 2 52.1 | ± 1.8  | 55.7 | ± 0.5   |

0.5. From the figure, we notice that Ra-DPO<sup>2</sup> achieves superior performance (the higher reward accuracy) and maintains a slight model drift (the lower KL divergence). Figure [4](#page-6-4) shows the reward accuracy of responses generated by models trained with different algorithms. The results demonstrate that when the coefficient α > 0.1, the reward accuracy of Ra-DPO<sup>2</sup> exceeds that of TDPO<sup>2</sup> across all risk control parameter µ. These results demonstrate that Ra-DPO<sup>2</sup> possesses a strong capability to align with human preferences.

#### 4.3. Experiments on AlpacaEval

Experimental setup: To comprehensively evaluate the performance of Ra-DPO2, we conducted pairwise comparisons on AlpacaEval using models trained on Anthropic HH dataset. Following the official *AlpacaEval implementation*[<sup>3</sup>](#page-6-5) , we sampled responses with a temperature coefficient of 0.7. The comparisons about winrate based on *oasst pythia 12b*[<sup>4</sup>](#page-6-6) are summarized in Table [1](#page-6-7) and Figure [5.](#page-7-0)

https://github.com/tatsu-lab/alpaca eval

<sup>4</sup> https://huggingface.co/OpenAssistant/oasst-sft-4-pythia-12bepoch-3.5

![](_page_7_Figure_1.jpeg)

Figure 5. AlpacaEval comparison between DPO, PPO, TDPO1, TDPO<sup>2</sup> and Ra-DPO<sup>2</sup> methods. The win, tie, and lose rates are evaluated based on *oasst-pythia-12b*.

Evaluation: Table [1](#page-6-7) reveals that under the two indicators of winrate and length-controlled winrate, most of the implemented algorithms can outperform the common default baseline *gpt4 1106 preview* (DPO is more prone to generating long responses). Among them, Ra-DPO<sup>1</sup> and Ra-DPO<sup>2</sup> demonstrate the highest level of performance, especially when it comes to the length-controlled winrate indicator. Figure [5](#page-7-0) presents a straightforward result: Compared to the baseline algorithms, Ra-DPO<sup>2</sup> achieves a high winrate, demonstrating superior performance in assisting LLMs to generate high-quality responses.

# 5. Related Work

# 5.1. LLMs Alignment

During the development and implementation of LLMs, numerous researchers have encountered challenges in balancing adherence to human instructions (explicit objective) with the pursuit of being helpful, honest, and harmless (implicit objectives), challenges that stem from the misaligned next token prediction task used in the pre-training stage [\(Bai et al.,](#page-8-0) [2022;](#page-8-0) [Bhardwaj & Poria,](#page-8-18) [2023;](#page-8-18) [Dai et al.,](#page-8-19) [2024;](#page-8-19) [Yeh et al.,](#page-9-16) [2024\)](#page-9-16). Therefore, a typical post-training stage, referred to as preference optimization (e.g., RLHF and DPO), is additionally performed to align pre-trained language models with human intentions, and it has become a crucial aspect in the fine-tuning of large models, often indispensable. Currently, most approaches [\(Wu et al.,](#page-9-17) [2023;](#page-9-17) [Wang et al.,](#page-9-6) [2024a;](#page-9-6) [Meng](#page-9-7) [et al.,](#page-9-7) [2024\)](#page-9-7) utilize KL divergence at the sentence level to ensure that the training model remains closely aligned with a reference model, preventing significant deviations. However, the generation of these responses occurs sequentially, following an auto-regressive approach. Recent works [\(Zeng et al.,](#page-9-8) [2024;](#page-9-8) [Ouyang et al.,](#page-9-18) [2024\)](#page-9-18) introduce a fresh perspective, specifically the sequential and token-level direct preference optimization, which allows for examining

KL divergence in relation to a reference LLM on a more granular, token-by-token basis. However, due to the neglect of the characteristics of a reward distribution other than the mean, these methods still suffer from the trouble of being insensitive to risk.

## 5.2. Risk-aware Reinforcement Learning

Reinforcement learning has made groundbreaking achievements through approaches such as Q-learning [\(Mnih et al.,](#page-9-19) [2015\)](#page-9-19) and policy gradients [\(Schulman et al.,](#page-9-9) [2015;](#page-9-9) [2017\)](#page-9-10) in sequential decision tasks, but it also faces challenges when considering application in the real world [\(Mnih et al.,](#page-9-19) [2015;](#page-9-19) [Wang & Chapman,](#page-9-12) [2022\)](#page-9-12). A primary reason is that the riskneutral criterion (maximizing the expectation) ignores the characteristics of a reward distribution other than the mean, which may be important for systems with safety concerns, especially in certain applications requiring tight risk control [\(Fei et al.,](#page-8-14) [2020;](#page-8-14) [Bisi et al.,](#page-8-5) [2022\)](#page-8-5). In order to tackle this challenge, two types of risk-sensitive measures have been introduced: nested and static quantile risk-aware measures. Static risk measures [\(Fei et al.,](#page-8-20) [2021;](#page-8-20) [Wang et al.,](#page-9-20) [2023\)](#page-9-20) are straightforward to interpret, but the resulting optimal policy may not remain Markovian and may become historydependent. On the other hand, nested risk measures [\(Chen](#page-8-15) [et al.,](#page-8-15) [2024;](#page-8-15) [Zhao et al.,](#page-9-13) [2024\)](#page-9-13) utilize MDPs to ensure risk sensitivity of the value iteration at each step under the current state, resulting in a more conservative approach. In this paper, we prefer nested risk measures because they recursively adhere to the Bellman equation and allow the MDPs to be reconstructed through state augmentation, enabling them to remain Markovian and ensuring that policy choices depend solely on the current state.

# 6. Conclusion

A pressing challenge arises for language generation tasks in the area of risk control, as the models, once trained, are often required to interact directly with humans. In this paper, we propose a novel direct preference optimization method that incorporates risk awareness by introducing nested risk measures into the Bellman equation, to align pre-trained LLMs with human preferences. Specifically, we design a new riskaware token-level objective function by reformulating the constrained reward maximization problem into a token-level form and then prove that maximizing this objective function leads to improvements in policy performance. Then, an optimization objective solely related to the risk-sensitive policy is obtained by deriving the mapping between the risk-aware state-action function and the optimal policy and establishing BT model equivalence with the Regret Preference Model. Finally, we conduct a formal analysis of this optimization objective and derive the loss function of Ra-DPO, which has practical implications for language generation tasks.

- 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 This paper presents work that aims to make LLMs more helpful and safer. Our work has many positive societal impacts, such as providing a theoretical foundation for riskaware language generation task, none of which we feel must be specifically highlighted. There are no negative societal impacts on our work. References Artzner, P. Thinking coherently. *Risk*, 10:68–71, 1997. Artzner, P., Delbaen, F., Eber, J.-M., and Heath, D. Coherent measures of risk. *Mathematical finance*, 9(3):203–228, 1999. Azar, M. G., Guo, Z. D., Piot, B., Munos, R., Rowland, M., Valko, M., and Calandriello, D. A general theoretical paradigm to understand learning from human preferences. In *AISTATS*, 2024. Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., Das-Sarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. *arXiv preprint arXiv:2204.05862*, 2022. Bauerle, N. and Rieder, U. More risk-sensitive markov ¨ decision processes. *Mathematics of Operations Research*, 39(1):105–120, 2014. Bhardwaj, R. and Poria, S. Red-teaming large language models using chain of utterances for safety-alignment. *arXiv preprint arXiv:2308.09662*, 2023. Biderman, S., Schoelkopf, H., Anthony, Q. G., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., et al. Pythia: A suite for analyzing large language models across training and scaling. In *ICML*, 2023. Bisi, L., Santambrogio, D., Sandrelli, F., Tirinzoni, A., Ziebart, B. D., and Restelli, M. Risk-averse policy optimization via risk-neutral policy optimization. *Artificial Intelligence*, 311:103765, 2022. Bonetti, M., Bisi, L., and Restelli, M. Risk-averse optimization of reward-based coherent risk measures. *Artificial Intelligence*, 316:103845, 2023. Bradley, R. A. and Terry, M. E. Rank analysis of incomplete block designs: I. the method of paired comparisons. *Biometrika*, 39(3/4):324–345, 1952. Candela, E., Doustaly, O., Parada, L., Feng, F., Demiris, Y., and Angeloudis, P. Risk-aware controller for autonomous vehicles using model-based collision prediction and reinforcement learning. *Artificial Intelligence*, 320:103923, 2023. Chaudhary, S., Dinesha, U., Kalathil, D., and Shakkottai,
  - S. Risk-averse fine-tuning of large language models. In *NeurIPS*, 2024. Chen, Y., Du, Y., Hu, P., Wang, S., Wu, D., and Huang, L. Provably efficient iterated cvar reinforcement learning with function approximation and human feedback. In *ICLR*, 2024. Christiano, P. F., Leike, J., Brown, T. B., Martic, M., Legg, S., and Amodei, D. Deep reinforcement learning from human preferences. In *NeurIPS*, 2017. Dai, J., Pan, X., Sun, R., Ji, J., Xu, X., Liu, M., Wang, Y., and Yang, Y. Safe rlhf: Safe reinforcement learning from human feedback. In *ICLR*, 2024. Dubois, Y., Galambosi, B., Liang, P., and Hashimoto, T. B. Length-controlled alpacaeval: A simple way to debias automatic evaluators. *arXiv preprint arXiv:2404.04475*, 2024. Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D., and Kiela, D. Model alignment as prospect theoretic optimization. In *ICML*, 2024. Fei, Y., Yang, Z., Chen, Y., Wang, Z., and Xie, Q. Risksensitive reinforcement learning: Near-optimal risksample tradeoff in regret. In *NeurIPS*, 2020. Fei, Y., Yang, Z., and Wang, Z. Risk-sensitive reinforcement learning with function approximation: A debiasing approach. In *ICML*, 2021. Fisch, A., Eisenstein, J., Zayats, V., Agarwal, A., Beirami, A., Nagpal, C., Shaw, P., and Berant, J. Robust preference optimization through reward model distillation. *arXiv preprint arXiv:2405.19316*, 2024. Givan, R., Dean, T., and Greig, M. Equivalence notions and model minimization in markov decision processes. *Artificial intelligence*, 147(1-2):163–223, 2003. Hau, J. L., Petrik, M., and Ghavamzadeh, M. Entropic risk optimization in discounted mdps. In *AISTATS*, pp. 47–76, 2023. Huber, J., Payne, J. W., and Puto, C. Adding asymmetrically dominated alternatives: Violations of regularity and the similarity hypothesis. *Journal of consumer research*, 9 (1):90–98, 1982. Lowd, D. and Davis, J. Learning markov network structure with decision trees. In *ICDM*, 2010.

# Impact Statement

504

506

508 509

511

514 515 516

518

524

526

528

531

534

536

538

- Maas, A., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., and Potts, C. Learning word vectors for sentiment analysis. In *ACL*, pp. 142–150, 2011. Meng, Y., Xia, M., and Chen, D. Simpo: Simple preference optimization with a reference-free reward. *arXiv preprint arXiv:2405.14734*, 2024. Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M. A., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., and Hassabis, D. Human-level control through deep reinforcement learning. *Nature*, 518:529–533, 2015. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright,
- C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. In *NeurIPS*, 2022. Ouyang, Y., Wang, L., Yang, F., Zhao, P., Huang, C., Liu, J., Pang, B., Yang, Y., Zhan, Y., Sun, H., et al. Token-level proximal policy optimization for query generation. *arXiv preprint arXiv:2411.00722*, 2024. Peuter, S. D., Zhu, S., Guo, Y., Howes, A., and Kaski,
- S. Preference learning of latent decision utilities with a human-like model of preferential choice. In *NeurIPS*, 2024. Pichler, A. and Schlotter, R. Entropy based risk measures. *European Journal of Operational Research*, 285(1):223– 236, 2020. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. Language models are unsupervised multitask learners. *OpenAI blog*, 1(8):9, 2019. Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning,
- C. D., and Finn, C. Direct preference optimization: your language model is secretly a reward model. In *NeurIPS*, 2023. Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz,
- P. Trust region policy optimization. In *ICML*, 2015. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*, 2017. Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and finetuned chat models. *arXiv preprint arXiv:2307.09288*, 2023. Tversky, A. and Kahneman, D. Advances in prospect theory: Cumulative representation of uncertainty. *Journal of Risk and uncertainty*, 5:297–323, 1992. Wang, C., Jiang, Y., Yang, C., Liu, H., and Chen, Y. Beyond reverse kl: Generalizing direct preference optimization with diverse divergence constraints. In *ICLR*, 2024a. Wang, K., Kallus, N., and Sun, W. Near-minimax-optimal risk-sensitive reinforcement learning with cvar. In *ICML*, 2023. Wang, Y. and Chapman, M. P. Risk-averse autonomous systems: A brief history and recent developments from the perspective of optimal control. *Artificial Intelligence*, 311:103743, 2022. Wang, Z., Bi, B., Pentyala, S. K., Ramnath, K., Chaudhuri, S., Mehrotra, S., Mao, X.-B., Asur, S., et al. A comprehensive survey of llm alignment techniques: Rlhf, rlaif, ppo, dpo and more. *arXiv preprint arXiv:2407.16216*, 2024b. Wu, Z., Hu, Y., Shi, W., Dziri, N., Suhr, A., Ammanabrolu, P., Smith, N. A., Ostendorf, M., and Hajishirzi, H. Finegrained human feedback gives better rewards for language model training. In *NeurIPS*, 2023. Xiao, W., Wang, Z., Gan, L., Zhao, S., He, W., Tuan, L. A., Chen, L., Jiang, H., Zhao, Z., and Wu, F. A comprehensive survey of datasets, theories, variants, and applications in direct preference optimization. *arXiv preprint arXiv:2410.15595*, 2024. Yeh, M.-H., Tao, L., Wang, J., Du, X., and Li, Y. How reliable is human feedback for aligning large language models? *arXiv preprint arXiv:2410.01957*, 2024. Yuan, Z., Yuan, H., Tan, C., Wang, W., Huang, S., and Huang, F. Rrhf: Rank responses to align language models with human feedback without tears. *arXiv preprint arXiv:2304.05302*, 2023. Zeng, Y., Liu, G., Ma, W., Yang, N., Zhang, H., and Wang,
  - J. Token-level direct preference optimization. In *ICML*, 2024. Zhang, L., Li, L., Wei, W., Song, H., Yang, Y., and Liang, J. Scalable constrained policy optimization for safe multiagent reinforcement learning. In *NeurIPS*, 2024. Zhao, W., He, T., and Liu, C. Model-free safe control for zero-violation reinforcement learning. In *CoRL*, 2021. Zhao, Y., Escamilla, J. E. A., Lu, W., and Wang, H. Ra-pbrl: Provably efficient risk-aware preference-based reinforcement learning. In *NeurIPS*, 2024.

554

556

558

560

564

566

568

571

574

576

578

594

596

598

## A. Supplementary Materials for Section [2](#page-1-3)

#### A.1. Risk Measure: A Brief Overview

Risk-aware Reinforcement Learning. Reinforcement learning has made groundbreaking achievements through approaches such as Q-learning [\(Mnih et al.,](#page-9-19) [2015\)](#page-9-19) and policy gradients [\(Schulman et al.,](#page-9-9) [2015;](#page-9-9) [2017\)](#page-9-10) in sequence decision tasks and has been gradually maturing in laboratory-level applications. In recent years, many researchers have gradually shifted their attention to real-world cyber-physical applications and found that focusing only on the mean of reward-to-go and corresponding Bellman equation is impractical, especially in some safety-critical scenarios requiring tight risk control, such as autonomous vehicle navigation [\(Candela et al.,](#page-8-6) [2023\)](#page-8-6) and robot control [\(Zhao et al.,](#page-9-21) [2021;](#page-9-21) [Zhang et al.,](#page-9-22) [2024\)](#page-9-22). A primary reason is that the risk-neutral criterion (maximizing the expectation) ignores the characteristics of a reward distribution other than the mean, which may be important for systems with safety concerns. For example, a system may be required to operate in a manner that alleviates harmful consequences, even in rare situations that are difficult to predict.

To handle this kind of issue, some works [\(Wang & Chapman,](#page-9-12) [2022\)](#page-9-12) introduce the worst-case criterion for autonomous systems with safety concerns to achieve zero-constraint violations by finding a policy that satisfies the constraints of a specific cost function, which generally assumes the maximum cost can quantify how bounded adversarial disturbances can inhibit the satisfactory operation of a system. However, due to the reliance on the typical assumption of bounded adversarial disturbances, the worst-case criterion may not be suitable for some applications that possess certain characteristics, such as the difficulty in characterizing the bounds of disturbances with a sufficient degree of certainty. Recently, risk-averse criterion [\(Bauerle & Rieder](#page-8-11) ¨ , [2014;](#page-8-11) [Bisi et al.,](#page-8-5) [2022\)](#page-8-5), an intermediary criterion between the risk-neutral and worst-case criteria, has garnered extensive attention, which describes people or algorithms that prefer outcomes with reduced uncertainty by seeking to optimize risk metrics, such as entropy risk measures (ERM) [\(Pichler & Schlotter,](#page-9-23) [2020\)](#page-9-23) or conditional value-at-risk (CVaR) [\(Artzner,](#page-8-8) [1997;](#page-8-8) [Chen et al.,](#page-8-15) [2024\)](#page-8-15), of the possible cumulative reward which emphasizes its distributional characteristics.

In general, there are mainly two types of risk-sensitive measures: nested and static quantile risk-aware measures, each possessing distinct advantages and limitations. Static risk measures [\(Fei et al.,](#page-8-20) [2021;](#page-8-20) [Wang et al.,](#page-9-20) [2023\)](#page-9-20) are straightforward to interpret, but the resulting optimal policy may not remain Markovian and may become history-dependent. On the other hand, nested risk measures [\(Chen et al.,](#page-8-15) [2024;](#page-8-15) [Zhao et al.,](#page-9-13) [2024\)](#page-9-13) utilize MDPs to ensure risk sensitivity of the value iteration at each step under the current state, resulting in a more conservative approach. In this paper, we prefer nested risk measures because they recursively adhere to the Bellman equation and allow the MDPs to be reconstructed through state augmentation, enabling them to remain Markovian and ensuring that policy choices depend solely on the current state.

Specifically, we introduce the popular CVaR [\(Artzner,](#page-8-8) [1997\)](#page-8-8) objective as follows:

$$G(\xi) = \begin{cases} \frac{1}{\mu}\xi & \text{if } \xi < \mu, \\ 1 & \text{if } \xi \geq \mu, \end{cases} \quad (20)$$

and Φ µ (Z) becomes

$$\Phi^\mu(Z) = \frac{1}{\mu} \int_0^\mu F_Z^{-1}(\xi) d\xi, \quad (21)$$

where G is LG-Lipschitz continuous for some L<sup>G</sup> ∈ <sup>R</sup>>0, and G(0) = 0, G(1) = 1.

Risks in LLMs Alignment. When aligning large language models with human preferences, there are many factors that may pose risks, primarily encompassing the following three types:

(1) There exist conflicts and contradictions among human preferences (or choices), thus introducing uncertainty in the objectives when aligning models with human preferences. In addition, human choice behavior has contextual choice effects [\(Peuter et al.,](#page-9-24) [2024\)](#page-9-24), i.e., a decision maker's choice between two options is influenced by adding more options to the choice set [\(Huber et al.,](#page-8-21) [1982\)](#page-8-21).

(2) Humans do not make decisions by maximizing their expected value for uncertain events; instead, they perceive random variables in a biased but well-defined manner [\(Ethayarajh et al.,](#page-8-9) [2024\)](#page-8-9). For example, relative to some reference point, humans are more sensitive to losses than gains, a phenomenon known as loss aversion.

(3) Many popular methods, such as DPO [\(Rafailov et al.,](#page-9-3) [2023\)](#page-9-3), RDPO [\(Fisch et al.,](#page-8-4) [2024\)](#page-8-4), and simPO [\(Meng et al.,](#page-9-7) [2024\)](#page-9-7), utilize KL divergence to ensure that the training model remains closely aligned with a reference model during the training process, preventing significant deviations. These methods still face the issue of being insensitive to strategic risks because

they only consider the mean of reward or utility and the corresponding Bellman equation, which is risk-neutral and does not capture the distribution characteristics of rewards efficiently.

Since the first two types of risks stem from the distribution of preference data itself, in this article, we focus on the third type of risk, which comes from the process during model alignment. Specifically, we investigate a novel direct preference optimization method for the problem of aligning with human preferences from a risk-sensitive perspective and provide theoretical and empirical results on its performance and risk-awareness.

## A.2. The Expanded Version of Value Function Definition

The definition of value function for nested risk measure, i.e., Eq. [5](#page-2-1) in Subsection [2.3,](#page-2-0) can be expanded as

$$Q_{\pi} ([x, y^{$$

$$V_\pi([x, y^{< t}]) = R([x, y^{< t}], \pi(\cdot | [x, y^{< t}])) + \Phi^\mu(R([x, y^{< t+1}], \pi(\cdot | [x, y^{< t+1}]))) + \Phi^\mu(\dots \Phi^\mu(R([x, y^{< T}], \pi(\cdot | [x, y^{< T}])))). \quad (23)$$

Similarly, the definition of the optimal value function, can be expanded as

$$Q_{\pi}^*([x, y^{$$

$$V_\pi^* \left( [x, y^{< t}] \right) = \max \left\{ R \left( [x, y^{< t}], \pi \left( \cdot \mid [x, y^{< t}]) \right) + \Phi^\mu \left( R \left( [x, y^{< t+1}], \pi \left( \cdot \mid [x, y^{< t+1}]) \right) \right) \right. \right. \\ \left. \left. + \Phi^\mu \left( \cdots \Phi^\mu \left( R \left( [x, y^{< T}], \pi \left( \cdot \mid [x, y^{< T}]) \right) \right) \right) \right) \right\}. \quad (25)$$

# B. Supplementary Materials for Section [3](#page-2-2)

# B.1. The Proof of Lamma [3.1](#page-3-4)

P Lemma 3.2 Restated. For a given Pb-MDP, the reward on the entire prompt-response can be decomposed as r = T <sup>t</sup>=1 γ <sup>t</sup>−<sup>1</sup>R ([x, y<t] , y<sup>t</sup> ), Vπ[x] in Eq. [5](#page-2-1) and V˜ <sup>π</sup>[x] in Eq. [6](#page-3-0) are equivalent, which implies the following characteristics:

*Proof.* Firstly, according to [\(Givan et al.,](#page-8-22) [2003;](#page-8-22) [Lowd & Davis,](#page-8-23) [2010;](#page-8-23) [Zhao et al.,](#page-9-13) [2024\)](#page-9-13), we can reformulate the Pb-MDP as a decision tree-like MDP.

- (1) The state transition graph of the Pb-MDP is connected and acyclic;
- (2) Each state in the Pb-MDP corresponds to a unique node in the tree;
- (3) There is a single root node from which every other node is reachable via a unique path;
- (4) The transition probabilities between states follow the Markov property, i.e., the probability of transitioning to any future state depends only on the current state and not on the sequence of events that preceded it.

Formally, let S be the set of states and pij be the transition probabilities between states s<sup>i</sup> and s<sup>j</sup> . For an Pb-MDP with a tree-like structure, the probabilistic transition matrix P is defined such that:

$$p_{ij} > 0 \text{ if there is an edge between } s_i \text{ and } s_j \text{ in the tree, and } p_{ij} = 0 \text{ otherwise.} \quad (26)$$

Moreover, for each non-root node s<sup>j</sup> , there exists exactly one s<sup>i</sup> such that pij > 0, and s<sup>i</sup> is the unique parent of s<sup>j</sup> in the tree structure.

To classify the two value iteration in Eq. [5](#page-2-1) and Eq. [6,](#page-3-0) we denote the value given by Eq. [6](#page-3-0) as V˜ <sup>π</sup> ([x, y<t]) and the value given by Eq. [5](#page-2-1) as V<sup>π</sup> ([x, y<t]), thus, in tree-like Pb-MDP with the reward of the entire prompt-response, which can be decomposed as r = P<sup>T</sup> <sup>t</sup>=1 γ <sup>t</sup>−<sup>1</sup>R ([x, y<t] , y<sup>t</sup> ), we have the following relationship:

$$\tilde{V}_\pi \left( [x, y^{$$

689 690

694

696

698

700

704

706

708 709

711

where R1:t−<sup>1</sup> = P<sup>t</sup>−<sup>1</sup> <sup>h</sup>=1 γ <sup>h</sup>−<sup>1</sup>R x, y<h , y<sup>h</sup> denotes the reward of the 1 ∼ t − 1 steps of a prompt-response. We prove this relationship by mathematical induction.

Initial Case. Using the tree-like Pb-MDP and the initial conditions of the Bellman equation, at the final step t = T, we have

$$\begin{aligned} \tilde{V}_\pi ([x, y^{$$

Induction Step. We now proved that if V˜ π x, y<t+1 = V<sup>π</sup> x, y<t+1 <sup>+</sup> <sup>R</sup>1:<sup>t</sup> holds, then <sup>V</sup>˜ <sup>π</sup> ([x, y<t]) = V<sup>π</sup> ([x, y<t]) + R1:t−<sup>1</sup> also holds. Since this policy π on tree-like Pb-MDP is fixed, it has only one path to arrive t-th state (s<sup>t</sup> = [x, y<t]), denoted as:

$$\Xi_t(s_{T,1}) = \Xi_h(s_{T,2}) \quad \forall s_{T,1}, s_{T,2} \in \{s_T \mid S_t(s_T) = [x, y^{$$

Therefore, R1:t−<sup>1</sup> is unique.

$$\begin{aligned}\tilde{V}_\pi([x, y^{$$

where the third equality holds because the risk measure function Φ satisfies translation invariance. Then, by applying conclusion, we observe that when t = 1, V˜ <sup>π</sup>[x] = Vπ[x] hold on. Thus, we have proven that for the Pb-MDP, the reward of the entire trajectory can be decomposed as r = P<sup>T</sup> <sup>t</sup>=1 γ <sup>t</sup>−<sup>1</sup>R ([x, y<t] , y<sup>t</sup> ), and Vπ[x] in Eq. [5](#page-2-1) and V˜ <sup>π</sup>[x] in Eq. [6](#page-3-0) are equivalent.

## B.2. The derivation of Definition [3.2](#page-3-1)

Definition 3.3 Restated. For a risk-sensitive Pb-MDP that satisfies the Bellman equation in Eq. [6,](#page-3-0) the risk-aware advantage function can be defined as

$$\tilde{A}_\pi \left( [x, y^{$$

where z subject to π<sup>θ</sup> (· | [x, y<t]).

In terms of designing the objective function at the token level, [\(Zeng et al.,](#page-9-8) [2024\)](#page-9-8) provides us with a valuable insight by introducing the advantage function from the TRPO algorithm in reinforcement learning as the target for each step. In this paper, building upon TDPO, we consider the risk associated with language generation at each step and devise a novel risk-sensitive advantage function. First, based on assumption that r = P<sup>T</sup> <sup>t</sup>=1 γ <sup>t</sup>−<sup>1</sup>R ([x, y<t] , y<sup>t</sup> ), we can get:

$$\begin{aligned}
r &= \sum_{t=1}^T \gamma^{t-1} R([x, y^{$$

Next, note that y <sup>T</sup> = EOS denotes the end of the text sequence. Therefore,

$$V_\pi([x, y^{$$

718

724

726

728

731

734

736

738

751

754

756

758

760

764

766

Furthermore, we have

$$r = \Phi^\mu \left( \tilde{V}_\pi \left( [x] \right) \right) + \sum_{t=1}^T \gamma^{t-1} \left( \tilde{Q}_\pi \left( [x, y^{$$

So, we definite the risk-aware advantage function as A˜ <sup>π</sup> ([x, y<t] , z) = Q˜ <sup>π</sup> ([x, y<t] , z) − Φ µ V˜ <sup>π</sup> ([x, y<t]) , where z ∼ π<sup>θ</sup> (· | [x, y<t]).

# B.3. The Proof of Lemma [3.3](#page-3-6)

Lemma 3.4 Restated. Given two policies π and π ′ , if for any state s<sup>t</sup> = [x, y<t] , <sup>E</sup>z∼π′ h A˜ <sup>π</sup> ([x, y<t] , z) i ≥ 0 holds, then we can conclude:

$$\mathbb{E}_{x \sim \mathcal{D}} \left[ \tilde{V}_{\pi'}([x]) \right] \geq \mathbb{E}_{x \sim \mathcal{D}} \left[ \tilde{V}_{\pi}([x]) \right].$$

*Proof.* Let trajectory τ := x, y<sup>1</sup> , y<sup>2</sup> , . . . , and the notation <sup>E</sup>τ|π′ [·] indicates that actions are sampled from π ′ to generate τ . So we can get

$$\begin{aligned} & \mathbb{E}_{x \sim \mathcal{D}} \left[ \tilde{V}_{\pi'}([x]) \right] - \mathbb{E}_{x \sim \mathcal{D}} \left[ \tilde{V}_{\pi}([x]) \right] \\ &= \mathbb{E}_{\tau|\pi'} \left[ \sum_{t=1}^{\infty} \gamma^{t-1} \left( R([x, y^{$$

Since for any state s<sup>t</sup> = [x, y<t] , <sup>E</sup>z∼π′ h A˜ <sup>π</sup> ([x, y<t] , z) i ≥ 0, so we can obtain

$$\mathbb{E}_{x \sim \mathcal{D}} \left[ \tilde{V}_{\pi'}([x]) \right] - \mathbb{E}_{x \sim \mathcal{D}} \left[ \tilde{V}_{\pi}([x]) \right] \geq 0.$$

## B.4. The Proof of Lemma [3.4](#page-3-7)

Lemma 3.5 Restated. The constrained problem in Eq. [9](#page-3-2) has the closed-form solution:

$$\pi_\theta^*(z \mid [x, y^{< t}]) = \frac{\pi_{\text{ref}}(z \mid [x, y^{< t}]) \exp\left(\frac{1}{\beta} \tilde{Q}_{\text{ref}}([x, y^{< t}], z)\right)}{Z([x, y^{< t}]; \beta)},$$

774

776

778

794

796

800

804

806

808

*Proof.*

$$\begin{aligned} & \max_{\pi_{\theta}} \mathbb{E}_{z \sim \pi_{\theta}(\cdot | [x, y^{< t]})} \tilde{A}_{\pi_{\text{ref}}} ([x, y^{< t}], z) - \beta D_{\text{KL}} (\pi_{\theta} (\cdot | [x, y^{< t}]) \| \pi_{\text{ref}} (\cdot | [x, y^{< t}])) \\ &= \max_{\pi_{\theta}} \mathbb{E}_{z \sim \pi_{\theta}(\cdot | [x, y^{< t]})} \left( \left( (\tilde{Q}_{\pi_{\text{ref}}} ([x, y^{< t}], z) - \tilde{V}_{\pi_{\text{ref}}} ([x, y^{< t}])) + \beta \log \left( \frac{\pi_{\text{ref}} (z | [x, y^{< t}])}{\pi_{\theta} (z | [x, y^{< t}])} \right) \right) \right. \\ &= \max_{\pi_{\theta}} \beta \mathbb{E}_{z \sim \pi_{\theta}(\cdot | [x, y^{< t]})} \log \left( \frac{\pi_{\text{ref}} (z | [x, y^{< t}]) e^{\frac{1}{\beta} \tilde{Q}_{\pi_{\text{ref}}} ([x, y^{< t}], z)}}{\pi_{\theta} (z | [x, y^{< t}])} \right) - \tilde{V}_{\pi_{\text{ref}}} ([x, y^{< t}]) \\ &= \max_{\pi_{\theta}} \beta \mathbb{E}_{z \sim \pi_{\theta}(\cdot | [x, y^{< t]})} \log \left( \frac{\pi_{\text{ref}} (z | [x, y^{< t}]) e^{\frac{1}{\beta} \tilde{Q}_{\pi_{\text{ref}}} ([x, y^{< t}], z)}}{Z ([x, y^{< t}]; \beta) \pi_{\theta} (z | [x, y^{< t}])} \right) \\ &\quad - \tilde{V}_{\pi_{\text{ref}}} ([x, y^{< t}]) + \beta \log Z ([x, y^{< t}]; \beta) \\ &= \max_{\pi_{\theta}} -\beta D_{\text{KL}} \left( \pi_{\theta} (z | [x, y^{< t}]) \| \frac{\pi_{\text{ref}} (z | [x, y^{< t}]) e^{\frac{1}{\beta} \tilde{Q}_{\pi_{\text{ref}}} ([x, y^{< t}], z)}}{Z ([x, y^{< t}]; \beta)} \right) \\ &\quad - \tilde{V}_{\pi_{\text{ref}}} ([x, y^{< t}]) + \beta \log Z ([x, y^{< t}]; \beta), \end{aligned} \tag{33}$$

where Z ([x, y<t] ; β) is the partition function:

$$Z \left( [x, y^{< t}] ; \beta \right) = \mathbb{E}_{z \sim \pi_{\text{ref}}(\cdot | [x, y^{< t}])} \exp \left( \frac{1}{\beta} \tilde{Q}_{\pi_{\text{ref}}} \left( [x, y^{< t}], z \right) \right). \quad (34)$$

Then, we can derive the relationship between the optimal policy and the state-action function:

$$\pi_{\theta}^*(z \mid [x, y^{< t}]) = \frac{\pi_{\text{ref}}(z \mid [x, y^{< t}]) \exp\left(\frac{1}{\beta} \hat{Q}_{\pi_{\text{ref}}}([x, y^{< t}], z)\right)}{Z([x, y^{< t}]; \beta)}. \quad (35)$$

## B.5. The Proof of Lemma [3.5](#page-3-8)

Lemma 3.6 Restated. Given a reward function r, based on a relationship between token-wise rewards and the reward function represented by r = P<sup>T</sup> <sup>t</sup>=1 γ <sup>t</sup>−<sup>1</sup>R ([x, y<t] , y<sup>t</sup> ), we can establish the equivalence between the Bradley-Terry model and the Regret Preference Model in the language generation task, i.e.,

$$P_{\text{BT}}(y_1 \succ y_2 \mid x) = \sigma \left( \sum_{t=1}^{T_1} \gamma^{t-1} \tilde{A}_{\pi} \left( [x, y_1^{$$

where σ(z) = 1/ (1 + exp(−z)) is the logistic sigmoid function for any random variable z.

*Proof.* Recalling to the BT model in Eq. [40](#page-15-1)

$$P_{\text{BT}}(y_1 \succ y_2 \mid x) = \frac{\exp(r(x, y_1))}{\exp(r(x, y_1)) + \exp(r(x, y_2))}, \quad (37)$$

and the equivalence between prompt-response reward and the risk-aware advantage function:

$$\begin{aligned} r &= \Phi^\mu \left( \tilde{V}_\pi ([x]) \right) + \sum_{t=1}^T \gamma^{t-1} \left( \tilde{Q}_\pi ([x, y^{$$

Then, we have

$$P_{\text{BT}}(y_1 \succ y_2 \mid x) = \sigma \left( \sum_{t=1}^{T_1} \gamma^{t-1} \tilde{A}_\pi \left( [x, y_1^{$$

828

831

834

836

838

854

856

858

860

864

866

868

874

876

#### B.6. The Proof of Theorem [3.6](#page-4-1)

Theorem 3.7 Restated. Given prompts x and pairwise responses (y1, y2), and the risk-aware objective function in Eq. [9,](#page-3-2) the Bradley-Terry model expresses the human preference probability in terms of the risk-aware optimal policy π ∗ θ and reference policy πref:

$$P_{\text{BT}}^*(y_1 \succ y_2 \mid x) = \sigma(u^*(x, y_1, y_2) - \delta^*(x, y_1, y_2)),$$

where u (x, y1, y2) represents the difference in implicit rewards defined by the risk-aware policy π ∗ θ and the reference policy πref, weighted by β, represented as

$$u(x, y_1, y_2) = \beta \log \frac{\pi_\theta(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} - \beta \log \frac{\pi_\theta(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)},$$

and δ (x, y1, y2) represents the difference in sequential risk ratio between two pairs (x, y1) and (x, y2), expressed as

$$\delta(x, y_1, y_2) = \beta D_{\text{SeqRR}}(x, y_2; \pi_{\text{ref}} \mid \pi_{\theta}) - \beta D_{\text{SeqRR}}(x, y_1; \pi_{\text{ref}} \mid \pi_{\theta}).$$

*Proof.* According to the Lemma [3.4,](#page-3-7) we have

$$\pi_\theta^*(z \mid [x, y^{$$

where Z ([x, y<t] ; β) = <sup>E</sup>z∼πref (·|[x,y<t])e 1 <sup>β</sup> <sup>Q</sup>˜πref ([x,y<t],z) is the partition function. Rearrange Eq. [38,](#page-15-2) we obtain

$$\tilde{Q}_{\pi_{\text{ref}}} \left( [x, y^{< t}], z \right) = \beta \log \frac{\pi_{\theta}^* \left( z \mid [x, y^{< t}] \right)}{\pi_{\text{ref}} \left( z \mid [x, y^{< t}] \right)} + \beta \log Z \left( [x, y^{< t}]; \beta \right). \quad (39)$$

From Lemma [3.5,](#page-3-8) we can get

$$P_{\text{BT}}(y_1 \succ y_2 \mid x) = \sigma \left( \sum_{t=1}^{T_1} \left( \gamma^{t-1} \tilde{A}_\pi \left( [x, y_1^{$$

By leveraging Eq. [39,](#page-15-3) we can derive

$$\begin{aligned} & \sum_{t=1}^T \gamma^{t-1} \tilde{A}_{\pi_{\text{ref}}} ([x, y^{$$

Note that

$$\mathbb{E}_{z \sim \pi_{\text{ref}}} [\beta \log Z([x, y^{$$

Therefore,

$$\begin{aligned} & \sum_{t=1}^T \gamma^{t-1} \tilde{A}_{\pi_{\text{ref}}} ([x, y^{$$

887 888

890

894

896

898

911

914 915 916

918

924

928

When substituting γ = 1 into the expression, we obtain a more concise form:

$$\begin{aligned} \sum_{t=1}^T \tilde{A}_{\pi_{\text{ref}}} \left( [x, y^{$$

where <sup>D</sup>SeqRR (x, y; <sup>π</sup>ref | <sup>π</sup>θ) = P<sup>T</sup> <sup>t</sup>=1 Φ µ <sup>z</sup>∼πref log <sup>π</sup>ref (z|x) πθ(z|x) 

.

Then, we let

$$u(x, y_1, y_2) = \beta \log \frac{\pi_\theta(y_1 \mid x)}{\pi_{\text{ref}}(y_1 \mid x)} - \beta \log \frac{\pi_\theta(y_2 \mid x)}{\pi_{\text{ref}}(y_2 \mid x)}, \quad (44)$$

$$\delta(x, y_1, y_2) = \beta D_{\text{SeqRR}}(x, y_2; \text{ref}_{\text{ref}} \mid \theta_{\text{ref}}) - \beta D_{\text{SeqRR}}(x, y_1; \text{ref}_{\text{ref}} \mid \theta_{\text{ref}}). \quad (45)$$

Substituting Eq. [43](#page-16-3) into Eq. [40,](#page-15-1) we arrive at P ∗ BT (y<sup>1</sup> ≻ y<sup>2</sup> | x) = σ (u ∗ (x, y1, y2) − δ ∗ (x, y1, y2)).

## B.7. Algorithm

In this subsection, we provide the main pseudocode for Risk-aware Direct Preference Optimization (Ra-DPO), as outlined in Algorithm [1.](#page-16-0)

| Algorithm 1 Risk-aware Direct Preference | Optimization (Ra-DPO)                                                      |
|------------------------------------------|----------------------------------------------------------------------------|
| Input: Reference model π ref , Policy    | model π θ , Coefficient α , β , Risk control parameter µ , Learning rate η |
| Input: Dataset D =                       |                                                                            |
| ( x, y w , y l )                         |                                                                            |
| o                                        | N                                                                          |
|                                          | i =1                                                                       |
|                                          | of size N , Method M                                                       |
| Initialize: π θ ← π ref                  |                                                                            |
| for each epoch do                        |                                                                            |
| Sample mini-batch D m = { ( x, y         | w , y l )                                                                  |
|                                          | m }                                                                        |
|                                          | m =1 from D                                                                |
| Predict the probabilities π θ ( y w      | x ) and π θ ( y l                                                          |
|                                          | x ) for ( x, y w , y l ) in the mini-batch D m using the policy model      |
| Predict the probabilities π ref ( y w    | x ) and π ref ( y l                                                        |
|                                          | x ) for ( x, y w , y l ) in the mini-batch D m using the reference model   |
| Calculate the function u ( x, y w , y    | l ) = β log π θ ( y w   x )                                                |
|                                          | π ref ( y w   x ) − β log π θ ( y l   x )                                  |
|                                          | π ref ( y l   x )                                                          |
| Compute the sequential risk ratio        | D SeqRR ( x, y w ; π ref   π θ ) for ( x, y w ) in the mini-batch D m      |
| Compute the sequential risk ratio        | D SeqRR ( x, y l                                                           |
|                                          | ; π ref   π θ ) for ( x, y l ) in the mini-batch D m                       |
| if Method M is Ra-DPO 1 then             |                                                                            |
| Calculate the function δ ( x, y w        | , y l ) = βD SeqRR ( x, y l                                                |
|                                          | ; π ref   π θ ) − βD SeqRR ( x, y w ; π ref   π θ )                        |
| θ ← θ + η ∇ θ E ( x,y w ,y l ) ∼D m [log | σ ( u ( x, y w , y l ) − δ ( x, y w , y l ))]                              |
| else { Method M is Ra-DPO 2 }            |                                                                            |
| Calculate the function δ 2 ( x, y        | w , y l ) = βD SeqRR ( x, y l                                              |
|                                          | ; π ref   π θ ) − sg ( βD SeqRR ( x, y w ; π ref   π θ ))                  |
| θ ← θ + η ∇ θ E ( x,y w ,y l ) ∼D m [log | σ ( u ( x, y w , y l ) − αδ 2 ( x, y w , y l ))]                           |
| end if                                   |                                                                            |
| end for                                  |                                                                            |

# C. Supplementary Materials for Section [4](#page-5-6)

# C.1. Additional experimental results

In this paper, we evaluate the performance of our proposed algorithm, Ra-DPO (Algorithm [1](#page-16-0) in the Appendix [B.7\)](#page-16-1), against baseline algorithms on several text tasks. Here, we provide some additional experimental results, which are illustrated in Figures [6-](#page-17-0)[7.](#page-18-0)

![](_page_17_Figure_2.jpeg)

Figure 6. The experiment on the Anthropic HH dataset with Pythia-1.4B serving as the base model. We implemented TDPO2, and different versions of Ra-DPO<sup>2</sup> with respect to the parameters α and µ. The progression of sequential KL divergence (the lower the better) of both preferred response and dispreferred responses are presented on the left and in the middle. Additionally, the reward accuracy curves (the higher the better) are illustrated on the right.

![](_page_18_Figure_1.jpeg)

Figure 7. The experiment on the Anthropic HH dataset with Pythia-2.8B serving as the base model. We implemented TDPO2, and different versions of Ra-DPO<sup>2</sup> with respect to the parameters α and µ. The progression of sequential KL divergence (the lower the better) of both preferred response and dispreferred responses are presented on the left and in the middle. Additionally, the reward accuracy curves (the higher the better) are illustrated on the right.