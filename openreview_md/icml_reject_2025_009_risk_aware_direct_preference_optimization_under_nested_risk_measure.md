# Risk-Aware Direct Preference Optimization Under Nested Risk Measure

## Anonymous Authors1 Abstract

When fine-tuning pre-trained Large Language Models (LLMs) to align with human values and intentions, the pursuit of maximizing the estimated reward can lead to superior performance, but it also introduces potential risks due to deviations from the original (reference) model's intended behavior. Most existing methods for aligning LLMs typically introduce KL divergence to constrain deviations between the training model and the reference model; however, this may not be sufficient in certain applications that require tight risk control. In this paper, we introduce Riskaware Direct Preference Optimization (Ra-DPO), a novel approach that incorporates risk-awareness by employing a token-level objective function under nested risk measure. This method formulates a constrained risk-aware advantage function maximization problem and then converts the Bradley- Terry model into a token-level representation. The ultimate objective function maximizes the likelihood of the policy while suppressing the deviation between a training model and the reference model using a sequential risk ratio, thereby enhancing the model's risk-awareness during the process of aligning LLMs. The proposed method's effectiveness is verified via three open-source datasets: IMDb Dataset, Anthropic HH Dataset, and AlpacaEval, and the results demonstrate superior performance of our method in balancing alignment performance and model drift.

## 1. Introduction

With the advanced and rapid developments of large language models (LLMs) technology, learning from human feedback, serving as a bridge in aligning LLMs with human values and intentions, has become increasingly crucial (Ouyang et al., 2022; Bai et al., 2022; Touvron et al., 2023; Bider1Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country.

1 000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054 man et al., 2023). Reinforcement Learning from Human Feedback (RLHF), which typically involves supervised finetuning, reward model training, and further fine-tuning of policy models via reinforcement learning (RL) algorithms, demonstrates impressive capabilities across diverse tasks and has emerged as a concrete research agenda (Christiano et al., 2017; Ouyang et al., 2022; Yuan et al., 2023). A criticized downside is that RLHF has a complex process that requires considerable memory and careful hyperparameter tuning to maintain the stability of RL training. Direct Preference Optimization (DPO) (Rafailov et al., 2023), featuring a simple and straightforward training process, directly uses the likelihood of the policy to define an implicit reward fitted to the preference data, which has emerged as a popular alternative since it bypasses key challenges in explicit reward modeling and achieves notable efficiency and competitive performance. Nevertheless, some studies (Xiao et al., 2024; Wang et al., 2024b) have reported that DPO still suffers from issues such as excessively long generative responses and the significant KL divergence of the dispreferred response subset. To tackle these issues, numerous variants of DPO have been successively proposed, including f-DPO (Wang et al., 2024a), IPO (Azar et al., 2024), RDPO (Fisch et al., 2024), and SimPO (Meng et al.,
2024), which introduce length control mechanisms or enhance KL divergence constraints. However, a key limitation is that these methods only consider evaluation at the sentence level, ignoring the fact that the generation of these responses occurs sequentially, following an auto-regressive approach. Recently, a fresh perspective on LLMs alignment has been introduced, specifically the sequential and token-level direct preference optimization, known as TDPO (Zeng et al., 2024), which allows for examining divergence in relation to a reference LLM on a more granular, token-by-token basis. Specifically, inspired by Trust Region Policy Optimization (TRPO) (Schulman et al., 2015) in RL field, TDPO redefines the objective of maximizing restricted rewards in a sequential manner and establishes the connection between sentence-level reward and token-level generation through using the Bellman equation. However, since the objective at each step is to maximize the expected return, a risk-neutral criterion, which neglects the characteristics of the reward distribution beyond the mean, TDPO encounters the same challenges as classic RL algorithms (Schulman et al., 2015; 2017; Bisi et al., 2022). Fortunately, in the field of RL, a series of risk-sensitive methods (Bisi et al., 2022; Candela et al., 2023) have been proposed, which achieve superior performance by introducing various risk measure functions. Recently, some researchers have attempted to introduce this technology in order to align LLMs with human preferences. For instance, RA-RLHF (Chaudhary et al., 2024) introduces Conditional Value at Risk (CVaR) (Artzner, 1997), a static risk measure function, into the fine-tuning of RL, while KTO (Ethayarajh et al., 2024) introduces prospect theory (Tversky & Kahneman, 1992) to fit human choice behavior when faced with uncertain events. However, these methods only analyze the risk of the whole prompt-response at the sentence level by considering the distribution characteristics of the preference data, which neglects the fact that the generation of these responses occurs sequentially, following an auto-regressive approach.

In this paper, we focus on the risk in the value iteration at each step by introducing nested risk measures. Specifically, we investigate a novel direct preference optimization method for the problem of aligning with human preferences from a risk-sensitive perspective and provide corresponding theoretical and empirical results. Our main contributions are summarized as follows.

055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109
- We propose a novel Risk-aware Direct Preference Optimization (Ra-DPO) method. This method maximizes the likelihood of the policy while effectively suppressing the deviation between the training model and the reference model by means of a sequential risk ratio, thereby enhancing the model's risk-awareness during the process of balancing alignment performance and model drift.

- We design a new risk-aware token-level objective function by reformulating the constrained reward maximization problem into a token-level form, and then prove that maximizing the objective function will result in policy improvements. Furthermore, by establishing equivalence between the Bradley-Terry model and the Regret Preference Model and deriving the mapping between the risk-aware state-action value function and the optimal policy, we obtain the optimization objective that is solely related to the risk-sensitive policy.

- Experimentally, we provide the results across various text generation tasks to evaluate the effectiveness of our proposed method and the sensitivity to the risk control parameter. The experimental results demonstrate that our method can effectively suppress the risk of model drift while enhancing its performance.

## 2. Preliminaries 2.1. Preference-Based Policy Optimization

Considering a preference-based language model fine-tuning task, let x denote an input prompt (question), and y denote the generated response (answer). The notation yw ≻ yl| x symbolizes the human preference data, where yw (win) represents a response that is more preferred by humans compared to yl (lose). Both x and yw/yl consist of a sequence of tokens. Bradley-Terry Model. In the preference-based fine-tuning process, to align with human preferences, a preference predictor adhering to the Bradley-Terry (BT) (Bradley & Terry, 1952) model has been widely employed for pairwise comparisons. The likelihood of a preference pair is commonly expressed using a latent reward model:

$$P_{\rm BT}\left(y_{w}\succ y_{l}\mid x\right)=\frac{\exp\left(r\left(x,y_{w}\right)\right)}{\exp\left(r\left(x,y_{w}\right)\right)+\exp\left(r\left(x,y_{l}\right)\right)},\tag{1}$$

where r(*x, y*w) and r(x, yl) stand for the reward function at the sentence level from the preferred and dispreferred answers, respectively. Directly Preference Optimization. Direct Preference Optimization (DPO) (Rafailov et al., 2023) commences with the following RL objective:

$$\begin{array}{c}{{\operatorname*{max}_{\pi_{\theta}}\mathbb{E}_{x\sim\mathcal{D},y\sim\pi_{\theta}(\cdot\mid x)}\left[r\left(x,y\right)\right.}}\\ {{\left.-\beta D_{\mathrm{KL}}\left(\pi_{\theta}(\cdot\mid x)\right)\mid\!\!\pi_{\mathrm{ref}}(\cdot\mid x)\right)\right],}}\end{array}$$
$$(2)$$

where D represents the human preference dataset, β is the coefficient of the reverse KL divergence penalty, πref (· | x)
is the policy of fixed reference model (typically selected to be the model that has undergone post-supervised finetuning), and πθ (· | x) represents the policy of the trained model, initialized with πθ = πref.

By reparameterizing the reward function in Eq. 2 using the policy in a supervised manner, DPO establishes a direct functional mapping between the reward model and the optimal policy.

$$r(x,y)=\beta\log\frac{\pi_{\theta}(y\mid x)}{\pi_{\rm ref}(y\mid x)}+\beta\log Z(x),\tag{3}$$

where Z(x) is the partition function or the normalizing constant. Then, by plugging the reward from Eq. 3 into the BT model in Eq. 1, DPO derives the objective function:

$$\mathcal{L}_{\text{DPO}}\left(\pi_{\theta};\pi_{\text{ref}}\right)=-\mathbb{E}_{(x,y_{w},y_{l})\sim\mathcal{D}}\left[\log\sigma\left(u\left(x,y_{w},y_{l}\right)\right)\right],\tag{4}$$
where
$u\left(x,y_{w},y_{l}\right)=\beta\log\frac{\pi_{\theta}\left(y_{w}\mid x\right)}{\pi_{\rm ref}\left(y_{w}\mid x\right)}-\beta\log\frac{\pi_{\theta}\left(y_{l}\mid x\right)}{\pi_{\rm ref}\left(y_{l}\mid x\right)}$.  
2.2. Preference-based Markov Decision Process 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 A Preference-based Markov Decision Process (Pb-MDP) can be formulated as a modification of the classical MDP: M⟨S, A, r, P*, γ, T*⟩, where S and A represent the finite state and action spaces, respectively; P : *S × A → S* is the probabilistic transition function; r represents the reward function of the entire prompt-response, which is defined as
(S × A)
T → R; γ is the discount factor, and T denotes the length of a trajectory or episode. Specifically, for language generation, the state st =
[x, y<t] ∈ S is a combination of the prompt and the generated response up to the current step, action at = y t ∈ A
corresponds to the next generated token, and the token-wise reward is defined as Rt := R (st, at) = R ([x, y<t] , yt).

Additionally, note that y
<1 = [ ] is an empty sequence.

Therefore, we denote [x] = [x, [ ]] = -*x, y*<1. For a given prompt x and the first t − 1 tokens y
<t of the response y, we define the probability distribution of the next token conditioned on [x, y<t] as πθ (· | [*x, y*<t]).

## 2.3. Risk Measure

It is more desirable to keep risk under control for language generation tasks instead of only considering a risk-neutral criterion, which overlooks the distribution characteristics of rewards, especially on certain safety-critical tasks that may have potential broad societal impact. Therefore, we introduce the risk-sensitive criterion (Bauerle & Rieder ¨ , 2014; Wang & Chapman, 2022) to quantify the hidden risk. More specifically, the definition of the quantile function and risk measure objective are as follows. The quantile function is the coherent risk-measure (Artzner et al., 1999; Bonetti et al., 2023) of random variable Z,

$$F_{Z}^{-1}(\xi)=\operatorname*{inf}\left\{z\in\mathbb{R}\mid F_{Z}\left(z\right)\geq\xi\right\},$$

which satisfies the following properties for all Z, Z′ ∈ Z:
Concavity: ∀ λ ∈ [0, 1] : η (λZ + (1 − λ)Z
′) ≥ λη (Z)+
(1 − λ) η (Z
′); *Monotonicity:* If Z ≥ Z
′, then η(Z) ≥
η (Z
′); *Translation Equivariance:* ∀ ϵ ∈ R : η (Z + ϵ) =
η (Z) + ϵ; Positive Homogeneity: ∀ λ > 0 : η (λZ) = λη (Z). Then, we introduce the nested risk-measures that are built upon Pb-MDP in Subsection 2.3. Nested risk-measures. In the context of standard Pb-MDP, the nested quantile risk measures (Fei et al., 2020; Chen et al., 2024; Zhao et al., 2024) can be elucidated in Bellman equation type as follows:

$$\begin{cases}Q_{\pi}\left(\left[x,y^{<t}\right],y^{t}\right)=R\left(\left[x,y^{<t}\right],y^{t}\right)+\Phi^{\mu}\left(V_{\pi}\left(\left[x,y^{<t}\right]\right)\right),\\ V_{\pi}\left(\left[x,y^{<t}\right]\right)=Q_{\pi}\left(\left[x,y^{<t}\right],\pi\left(\cdot\mid\left[x,y^{<t}\right]\right)\right),\\ V_{\pi}\left(\left[x,y^{<T}\right]\right)=R\left(\left[x,y^{<T}\right]\right),\end{cases}\tag{5}$$
(5)
where $Q_{\pi}\left(\left[x,y^{<t}\right],y^{t}\right)$ and $V_{\pi}\left(\left[x,y^{<t}\right]\right)$ represent the 
state-action value and state value under the nested risk measures at timestep t ∈ [1, · · · , T], respectively. Φ(·) is a nested risk measure function with a risk control parameter µ. For any random variable Z, we have

$$\Phi^{\mu}(Z)=\int_{0}^{1}F_{Z}^{-1}(\xi)\mathrm{d}G(\xi),$$

where G is a weighting function over the quantiles. This class captures a broad range of useful objectives, including the popular CVaR (Artzner, 1997) objective. Due to space constraints, we provide a detailed survey about risk measure in Appendix A.1 and the expanded version of value function definition in Appendix A.2.

## 3. Methodology

This section proposes a novel language model alignment method called Risk-aware Direct Preference Optimization (Ra-DPO). Specifically, we first conduct an analysis of the characteristics of nested risk measures and design a new risk-aware token-level objective function by reformulating the constrained reward maximization problem into a tokenlevel form. Subsequently, we prove that maximizing the objective function will result in policy improvements. Then, the optimization objective solely related to the risk-sensitive policy is obtained by deriving the mapping between the risk-aware state-action function and the optimal policy; and establishing BT model equivalence with the Regret Preference Model. Finally, we conduct a formalized analysis of this optimization objective in terms of derivatives and derive the loss function for Ra-DPO.

## 3.1. Risk-Aware Objective Function

In this subsection, we aim to design a new risk-aware objective function for preference-based language model finetuning. Unfortunately, although the recursive Bellman equation under nested risk measures was introduced in Subsection 2.3, it cannot be directly applied, mainly due to the following reasons: (1) For the Pb-MDP setting, the algorithm can only obtain the reward (an implicit reward fitted to the preference data) at an entire prompt-response until the end and thus cannot compute the target value at each step. (2) The nested risk-measures incorporate a Bellman-type recursion and are not law-invariant (Hau et al., 2023), which are complex and difficult to compute. To surmount these obstacles, a straightforward approach is to introduce the state augmentation method, i.e., reconstructing an augmented Pb-MDP as described in (Zhao et al., 2024), where the state at each timestep includes historical trajectories. This method can reformulate the recursive Bellman equation into a classical Bellman equation with augmented states. However, it is noteworthy that, in this paper, we directly define the state as a combination of the prompt and the generated response up to the current step to model the sequential and auto-regressive generation. It possesses a characteristic in that the state at the previous timestep is a subset of the state at the current timestep, i.e.,
-x, y<t−1⊂ [*x, y*<t]. Therefore, we can rewrite the nested quantile objective's Bellman equation in Eq. 5 as follows:

$$\begin{cases}\tilde{Q}_{\pi}\left(\left[x,y^{<t}\right],y^{t}\right)=\Phi^{\mu}\left(\tilde{V}_{\pi}\left(y^{t+1}\circ\left(\left[x,y^{<t}\right],y^{t}\right)\right)\right),\\ \tilde{V}_{\pi}\left(\left[x,y^{<t}\right]\right)=\tilde{Q}_{\pi}\left(\left[x,y^{<t}\right],\pi\left(\cdot\mid\left[x,y^{<t}\right]\right)\right),\\ \tilde{V}_{\pi}\left(\left[x,y^{<T}\right]\right)=R\left(\left[x,y^{<T}\right]\right),\end{cases}\tag{6}$$

(6)
where Q˜π ([x, y<t] , yt) and V˜π ([*x, y*<t]) represent the riskaware state value and state-action value under the policy π, respectively. It is noteworthy that there is a significant difference in the calculation of the risk-aware state value function between Eq. 5 and Eq. 6. And, according to the Lemma 3.6 in (Zhao et al., 2024), we can obtain the following lemma. Lemma 3.1. For a given Pb-MDP, the reward on the entire prompt-response can be decomposed as P
r =
T
t=1 γ t−1R ([x, y<t] , yt), the relationship between the state value function Eq. 5 and Eq. 6 *is as follows:*

$${\tilde{V}}_{\pi}\left(\left[x,y^{<t}\right]\right)=V_{\pi}\left(\left[x,y^{<t}\right]\right)+R_{1:t-1},$$

where R1:t−1 =Pt−1 h=1 γ h−1R-x, y<h, yhdenotes the reward of the 1 ∼ t − 1 *steps of a prompt-response, and* Vπ[x] and V˜π[x] *are equivalent.*
The proof is detailed in Appendix B.1. Subsequently, based on the new risk-aware state value and state-action value in Eq. 6, we define the risk-aware advantage function as follows.

Definition 3.2. For a risk-sensitive Pb-MDP that satisfies the Bellman equation in Eq. 6, the risk-aware advantage function can be defined as

$$\tilde{A}_{\pi}\left(\left[x,y^{<t}\right],z\right)=\tilde{Q}_{\pi}\left(\left[x,y^{<t}\right],z\right)-\Phi^{\mu}(\tilde{V}_{\pi}\left(\left[x,y^{<t}\right]\right)),\tag{8}$$
where z subject to πθ (· | [*x, y*<t]).
The definition is reasonable, and the derivation provided in Appendix B.2. Furthermore, based on the definition of risk-aware advantage function in Definition 3.2, we propose a new risk-aware objective function:

objective function.  $$\max_{\pi_{\theta}}\mathbb{E}_{x,y<^{t}\sim\mathcal{D},z\sim\pi_{\theta}(\cdot\mid[x,y<^{t}])}\left[\tilde{A}_{\pi_{\rm ref}}\left(\left[x,y^{<t}\right],z\right)\right.$$ $$\left.-\beta D_{\rm KL}\left(\pi_{\theta}\left(\cdot\mid\left[x,y^{<t}\right]\right)\parallel\pi_{\rm ref}\left(\cdot\mid\left[x,y^{<t}\right]\right)\right)\right].\tag{9}$$
The objective function maximizes a risk-sensitive advantage function subject to a KL divergence constraint, which takes into account the risk when selecting the optimal policy, thereby achieving a better balance between alignment performance and model drift. Next, we prove that maximizing the risk-aware objective function in Eq. 9 will result in policy improvements, as stated in the following lemma. Lemma 3.3. Given two policies π and π
′*, if for any state* st = [*x, y*<t] , Ez∼π′
hA˜π ([x, y<t] , z)
i≥ 0, then we can conclude:

$$\mathbb{E}_{x\sim\mathcal{D}}\left[\tilde{V}_{\pi^{\prime}}([x])\right]\geq\mathbb{E}_{x\sim\mathcal{D}}\left[\tilde{V}_{\pi}([x])\right].\tag{10}$$

The proof is provided in Appendix B.3.

## 3.2. Risk-Aware Preference Optimization

In this subsection, we focus on how to convert the BT
model into risk-sensitive token-level representation to obtain the optimization objective that is solely related to the risksensitive policy, which is divided into two steps: (1) derive the mapping between the risk-aware state-action function and the optimal policy; (2) establish BT model equivalence with the Regret Preference Model.

$$\left(7\right)$$

Specifically, starting from the risk-aware token-level objective function in Eq. 9, we first derive the mapping between the risk-aware state-action function Q˜π and the optimal policy π
∗
θ, as stated in the following lemma.

Lemma 3.4. The constrained problem in Eq. 9 *has the* closed-form solution:

$$\pi_{\theta}^{*}\left(z\;|\;\left[x,y^{<t}\right]\right)$$
$$\pi_{\rm ref}\left(z\mid[x,y^{<t}]\right)\exp\left(\frac{1}{\beta}\tilde{Q}_{\pi_{\rm ref}}\left([x,y^{<t}]\,,z\right)\right)$$ $$=\frac{\pi_{\rm ref}\left(z\mid[x,y^{<t}]\right)\exp\left(\frac{1}{\beta}\tilde{Q}_{\pi_{\rm ref}}\left([x,y^{<t}]\,,z\right)\right)}{Z\left([x,y^{<t}]\,;\beta\right)},\tag{1}$$
where
$$(11)$$
$$Z\left(\left[x,y^{<t}\right];\beta\right)=\mathbb{E}_{z\sim\pi_{\mathrm{ref}}\left(\cdot|\left[x,y^{<t}\right]\right)}e^{\frac{1}{\beta}\hat{Q}_{\pi_{\mathrm{ref}}}\left(\left[x,y^{<t}\right],z\right)},$$

which is the partition function. The proof is provided in Appendix B.4. Then, by rearranging Eq. 11, we can obtain the expression of the risk-aware state-action function in terms of the policy

$$\bar{Q}_{\pi_{\mathrm{ref}}}\left(\left[x,y^{<t}\right],z\right)$$
= β log  π ∗ θ(z | [x, y<t]) πref (z | [x, y<t])  + β log Z-x, y<t; β.
$\left(12\right)^{2}$
Subsequently, by utilizing the reward decomposition formula r =PT
t=1 γ t−1R ([x, y<t] , yt) from Lemma 3.1, we establish BT model equivalence with the Regret Preference Model, as shown in the following lemma.

4 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 Lemma 3.5. Given a reward function r of the entire prompt-response, based on a relationship between tokenwise rewards and the reward function represented by P
r =
T
t=1 γ t−1R ([x, y<t] , yt), we can establish the equivalence between the Bradley-Terry model and the Regret Preference Model, i.e.,

_erene Model, i.e._,  $$P_{\rm BT}\left(y_{1}\succ y_{2}\mid x\right)=\sigma\left(\sum_{t=1}^{T_{1}}\gamma^{t-1}\bar{A}_{\pi}\left(\left[x,y_{1}^{<t}\right],y_{1}^{t}\right)\right.$$ $$\left.-\sum_{t=1}^{T_{2}}\gamma^{t-1}\bar{A}_{\pi}\left(\left[x,y_{2}^{<t}\right],y_{2}^{t}\right)\right),\tag{13}$$  $I_{\rm BT}\left(y_{1}\succ y_{2}\mid x\right)=\sigma\left(\sum_{t=1}^{T_{1}}\gamma^{t-1}\bar{A}_{\pi}\left(\left[x,y_{2}^{<t}\right],y_{2}^{t}\right)\right),$
where σ(z) = 1/ (1 + exp(−z)) is the logistic sigmoid function for any random variable z. The proof is provided in Appendix B.5.

According to the definition of the risk-aware advantage function in Definition 3.2, we can directly establish the relationship between the optimal solution in Eq. 12 and preference optimization objective in Eq. 13. In this way, we ultimately reformulate the BT model to be directly tied to the risk-aware optimal policy π
∗ θ and the reference policy πref, which is summarized in the following theorem.

Theorem 3.6. Given prompts x *and pairwise responses*
(y1, y2), and the risk-aware objective function in Eq. 9, the Bradley-Terry model expresses the human preference probability in terms of the risk-aware optimal policy π
∗
θand reference policy πref:

$$P_{\rm BT}^{*}\left(y_{1}\succ y_{2}\mid x\right)=\sigma\left(u^{*}\left(x,y_{1},y_{2}\right)-\delta^{*}\left(x,y_{1},y_{2}\right)\right),\tag{14}$$

(14)
where u (x, y1, y2) represents the difference in implicit rewards defined by the risk-aware policy π
∗
θ and the reference policy πref, weighted by β*, represented as*

$$u\left(x,y_{1},y_{2}\right)=\beta\log\frac{\pi_{\theta}\left(y_{1}\mid x\right)}{\pi_{\rm ref}\left(y_{1}\mid x\right)}-\beta\log\frac{\pi_{\theta}\left(y_{2}\mid x\right)}{\pi_{\rm ref}\left(y_{2}\mid x\right)},\tag{15}$$

and δ (x, y1, y2) *represents the difference in sequential risk* ratio between two pairs (x, y1) and (x, y2)*, expressed as*

$$\delta\left(x,y_{1},y_{2}\right)=\beta D_{\rm SeqRR}\left(x,y_{2};\pi_{\rm ref}\mid\pi_{\theta}\right)\tag{16}$$ $$-\beta D_{\rm SeqRR}\left(x,y_{1};\pi_{\rm ref}\mid\pi_{\theta}\right),$$

where

$$D_{\mathrm{SeqRR}}\left(x,y;\pi_{\mathrm{ref}}\mid\pi_{\theta}\right)=\sum_{t=1}^{T}\Phi_{z\sim\pi_{\mathrm{ref}}}^{\mu}\left(\log{\frac{\pi_{\mathrm{ref}}\left(z\mid x\right)}{\pi_{\theta}\left(z\mid x\right)}}\right)$$

The proof is provided in the Appendix B.6.

## 3.3. Loss Function And Formal Analysis

Drawing on Theorem 3.6, we reformulate the BT model into a structure solely relevant to the risk-sensitive policy, 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 which enables us to formulate a likelihood maximization objective for a parametrized policy πθ, and then our loss function becomes:
LRa-DPO1
(πθ; πref)

= −E(x,yw,yl)∼D [log σ (u (x, yw, yl) − δ (x, yw, yl))] . (17)
$\mathrm{dPO}_1\;(\pi_\theta;\pi_\mathrm{ref})$
Through this approach, we explicitly introduce sequential risk ratio into the loss function, which incorporates riskawareness during the process of balancing alignment performance and model drift. To elucidate the benefit of the proposed method, we give further interpretation by analyzing the loss function and its gradient. Specifically, we conduct a derivative analysis of our method. For convenience, we use u to denote u (x, yw, yl), and δ to represent δ (x, yw, yl).

By simple calculations, we can derive the gradient of the loss function in Eq. 17 with respect to the parameters θ :

$$\begin{array}{l}\nabla_{\theta}\mathcal{L}_{\text{Ra-DPO}_{1}}\left(\pi_{\theta};\pi_{\text{ref}}\right)\\ \qquad=-\mathbb{E}_{(x,y_{w},y_{l})\sim\mathcal{D}}\left[\left(-u+\delta\right)\left[\nabla_{\theta}u-\nabla_{\theta}\delta\right]\right],\end{array}\tag{18}$$  where $(-u+\delta)$ denotes the activation factor for the $\alpha$-th
$$\mathcal{L}_{\text{QPO}}\left(\pi_{\theta};\pi_{\text{ref}}\right)=-\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_{\theta}\left(y_{w}\mid x\right)}{\pi_{\text{ref}}\left(y_{w}\mid x\right)}-\beta\log\frac{\pi_{\theta}\left(y_{l}\mid x\right)}{\pi_{\text{ref}}\left(y_{l}\mid x\right)}\right)\right].$$
$$\mathcal{L}_{\text{TDEO}_{2}}\left(\pi_{\theta};\pi_{\text{ref}}\right)=-\mathbb{E}\left[\log\sigma\left(\left(\beta\log\frac{\pi_{\theta}\left(y_{w}\mid x\right)}{\pi_{\text{ref}}\left(y_{w}\mid x\right)}-\beta\log\frac{\pi_{\theta}\left(y_{l}\mid x\right)}{\pi_{\text{ref}}\left(y_{l}\mid x\right)}\right)\right.\right.$$ $$\left.\left.-\alpha\left(\beta D_{\text{SqKL}}\left(x,y_{l};\pi_{\text{ref}}\|\pi_{\theta}\right)-\text{sg}\left(\beta D_{\text{SqKL}}\left(x,y_{w};\pi_{\text{ref}}\|\pi_{\theta}\right)\right)\right)\right]\right.$$

where (−u + δ) serves as the weighting factor for the gradient.

Figure 1. Comparison of loss functions for DPO, TDPO2 and Ra-DPO2 methods. The sg denotes the stop-gradient operator.

From Eq. 18, we can observe that the first part (−u) corresponds to the weight factor in the first part of loss function of TDPO. Its value will increase when the language model makes prediction errors relative to human preferences, i.e.,
log πθ(yl|x)
πref (yl|x) > log πθ(yw|x)
πref (yw|x)
. The second part δ consists of the difference between the sequential risk ratio of the dispreferred response subset and the preferred response subset, which is a distinctive component of our method. When selecting a convex function (risk-averse), such as CVaR, as the risk measure function, our method automatically balances the risk ratio. Furthermore, based on a common starting point shared by our method and TDPO (Zeng et al., 2024), i.e., reducing risks stemming from model drift and ensuring training stability, we also provide the second version of our method, Ra-DPO2. The loss function of Ra-DPO2 is given by:

$$\pi_{\mathrm{ref}}\,]$$

LRa-DPO2(πθ; πref)
$\mathbb{E}\sigma_{2}\left(\sigma_{3},\sigma_{4}\right)$  $-\mathbb{E}_{\left(x,y_{w},y_{l}\right)\sim\mathcal{D}}\left[\log\sigma\left(u\left(x,y_{w},y_{l}\right)-\alpha\delta_{2}\left(x,y_{w},y_{l}\right)\right)\right],$
(19)
where α is a parameter, and
$\delta_{2}\left(x,y_{1},y_{2}\right)=\beta D_{\rm SeqRR}\left(x,y_{2};\pi_{\rm ref}\mid\pi_{\theta}\right)$  $-\,{\rm sg}\left(\beta D_{\rm SeqRR}\left(x,y_{1};\pi_{\rm ref}\mid\pi_{\theta}\right)\right).$
The sg represents the stop-gradient operator, which blocks the propagation of gradients. Ra-DPO2 modifies the loss function of Ra-DPO1 by discontinuing the gradient propagation of DSeqRR(*x, y*w; πref | πθ) and treating it as a baseline term for alignment of DSeqRR(*x, y*l; πref | πθ). The aim of the modification is to ensure training stability, rather than accelerating the training speed. To summarize, the comparison of the loss functions for DPO,
TDPO2, and Ra-DPO2 is shown in Figure 1. In addition, we give a procedure of our method, and provide its pseudocode (Algorithm 1) in Appendix B.7.

## 4. Experiments

We empirically evaluate our method via several open-source datasets and pre-trained models. Our experiments aim to answer the following questions: First, how does the performance of our method compare with existing methods, and is our method more sensitive to risks when tackling challenging text generation tasks? Second, how does the risk control parameter µ affect the performance of our method? To answer these questions, we conduct experiments on IMDb Dataset (Maas et al., 2011), Anthropic HH Dataset (Bai et al., 2022) and AlpacaEval (Dubois et al., 2024) for three different text generation tasks. Based on the original KTO implementation1, we trained Ra-DPO and the baseline 1Available at https://github.com/ContextualAI/
HALOs 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 models using the same hyperparameters. Specifically, for Ra-DPO, we employed the popular CVaR (Artzner, 1997) as the risk measure function. We compare our method against the following algorithms: (1) DPO (Rafailov et al., 2023), which only considers evaluation at the sentence level; (2) PPO (Schulman et al., 2017), which is an offline PPO variant provided by the original KTO implementation; (3) TDPO1 and TDPO2 (Zeng et al., 2024), which convert the BT model into token-level representation to obtain the optimization objective; (4) KTO (Ethayarajh et al., 2024), which considers humans make decisions that do not maximize their expected value when faced with uncertain events. All reported results of our algorithm and baseline algorithms are trained using 4 × A100 GPUs, each with 40GB of memory.

## 4.1. Experiments On Imdb Dataset

Experimental setup: The IMDb dataset is a controlled semantic generation dataset within the context of movie reviews, serving as a valuable resource for training and evaluating sentiment analysis models. We employ GPT-2 Large
(Radford et al., 2019) as our base model and use the model checkpoint: *insub/gpt2-large-IMDb-fine-tuned*2as the SFT
model. In this setup, the model is presented with prompts consisting of prefixes from movie reviews, and is required to generate responses with positive sentiment. Specifically, we implement the versions of Ra-DPO1 with risk control parameter µ ∈ {0.99, 0.98, 0.97, 0.95}. Moreover, in order to achieve a fair comparison, we calculate the sequential KL divergence for our method. Note that the risk ratio value is slightly larger than the KL divergence value when selecting CVaR (a convex function) as the risk measure function. The results are shown in Figure 2.

Evaluation: Figure 2 shows that Ra-DPO1 can outperform or achieve reward accuracy similar to the advanced TDPO algorithm while also maintaining a slight model drift (indicated by the lower KL divergence), demonstrating the 2https://huggingface.co/insub/gpt2-large-IMDb-fine-tuned

DSeq KL(x, y w; ref 
)

TDPO1 Ra-DPO1 = 0.99 Ra-DPO1 = 0.98 Ra-DPO1 = 0.97 Ra-DPO1 = 0.95 0 5k 10k 15k 20k 25k step 0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 0 5k 10k 15k 20k 25k step 0.5 0.6 0.7 0.8 0.9 0 5k 10k 15k 20k 25k step 0 2 4 6 8 10 12 DSeqKL(x, yl; ref 
)

Re ward Ac cura cy TDPO1 Ra-DPO1 = 0.99 Ra-DPO1 = 0.98 Ra-DPO1 = 0.97 Ra-DPO1 = 0.95 TDPO1 Ra-DPO1 = 0.99 Ra-DPO1 = 0.98 Ra-DPO1 = 0.97 Ra-DPO1 = 0.95
(a)
(b)
(c)
0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 coefficient 0.53 0.54 0.55 0.56 0.57 0.58 0.59 Re wa rd A
ccu ra cy TDPO2 Ra-DPO2 = 0.99 Ra-DPO2 = 0.98 Ra-DPO2 = 0.97 Ra-DPO2 = 0.95

| METHOD   | WINRATE   | LC WINRATE   |
|----------|-----------|--------------|
| DPO      | 51.1± 1.9 | 44.7± 0.4    |
| PPO      | 52.1± 1.8 | 51.9± 0.5    |
| KTO      | 51.5± 1.8 | 50.2± 0.6    |
| TDPO1    | 51.9± 1.8 | 53.0± 0.6    |
| TDPO2    | 52.2± 1.6 | 52.2± 0.5    |
| RA-DPO1  | 53.5± 1.8 | 53.9± 0.5    |
| RA-DPO2  | 52.1± 1.8 | 55.7± 0.5    |

0.5. From the figure, we notice that Ra-DPO2 achieves superior performance (the higher reward accuracy) and maintains a slight model drift (the lower KL divergence). Figure 4 shows the reward accuracy of responses generated by models trained with different algorithms. The results demonstrate that when the coefficient α > 0.1, the reward accuracy of Ra-DPO2 exceeds that of TDPO2 across all risk control parameter µ. These results demonstrate that Ra-DPO2 possesses a strong capability to align with human preferences.

## 4.3. Experiments On Alpacaeval

Experimental setup: To comprehensively evaluate the performance of Ra-DPO2, we conducted pairwise comparisons on AlpacaEval using models trained on Anthropic HH
dataset. Following the official *AlpacaEval implementation*3, we sampled responses with a temperature coefficient of 0.7.

The comparisons about winrate based on oasst pythia 12b4 are summarized in Table 1 and Figure 5.

3https://github.com/tatsu-lab/alpaca eval 4https://huggingface.co/OpenAssistant/oasst-sft-4-pythia-12bepoch-3.5

0 20k 40k 60k 80k 100k 120k 140k 160k step 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0 20k 40k 60k 80k 100k 120k 140k 160k step 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0 20k 40k 60k 80k 100k 120k 140k 160k step 0.475 0.500 0.525 0.550 0.575 0.600 0.625 DSeq KL(x, y w; ref 
)

DSeqKL(x, yl; ref 
)

Rew ard Ac cura cy TDPO2 = 0.5 Ra-DPO2 = 0.5 = 0.99 Ra-DPO2 = 0.5 = 0.98 Ra-DPO2 = 0.5 = 0.97 Ra-DPO2 = 0.5 = 0.95 TDPO2 = 0.5 Ra-DPO2 = 0.5 = 0.99 Ra-DPO2 = 0.5 = 0.98 Ra-DPO2 = 0.5 = 0.97 Ra-DPO2 = 0.5 = 0.95 TDPO2 = 0.5 Ra-DPO2 = 0.5 = 0.99 Ra-DPO2 = 0.5 = 0.98 Ra-DPO2 = 0.5 = 0.97 Ra-DPO2 = 0.5 = 0.95
(a)
(b)
(c)
risk-awareness of Ra-DPO1 during the process of balancing alignment performance and model drift.

## 4.2. Experiments On Anthropic Hh Dataset

330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 Experimental setup: Anthropic HH dataset contains 170k dialogues between a human and an automated assistant, where each transcript ends with a pair of responses generated by an LLM along with a preference label denoting the human-preferred response. We use Pythia-1.4B and Pythia2.8B (Biderman et al., 2023) as the base models to test our method on Anthropic HH dataset, respectively. Here, the reference models are trained by fine-tuning the base models on chosen completions. Specifically, we implement TDPO2 and different versions of Ra-DPO2 with respect to the parameters µ and α The results are depicted in Figure 3, Figure 4, and Appendix C.1.

Evaluation: Figure 3 shows the performance of TDPO2, and different versions of Ra-DPO2 with respect to the risk control parameter µ while keeping coefficient α constant at 7 Evaluation: Table 1 reveals that under the two indicators of winrate and length-controlled winrate, most of the implemented algorithms can outperform the common default baseline gpt4 1106 *preview* (DPO is more prone to generating long responses). Among them, Ra-DPO1 and Ra-DPO2 demonstrate the highest level of performance, especially when it comes to the length-controlled winrate indicator. Figure 5 presents a straightforward result: Compared to the baseline algorithms, Ra-DPO2 achieves a high winrate, demonstrating superior performance in assisting LLMs to generate high-quality responses.

## 5. Related Work 5.1. Llms Alignment

385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 During the development and implementation of LLMs, numerous researchers have encountered challenges in balancing adherence to human instructions (explicit objective) with the pursuit of being helpful, honest, and harmless (implicit objectives), challenges that stem from the misaligned next token prediction task used in the pre-training stage (Bai et al., 2022; Bhardwaj & Poria, 2023; Dai et al., 2024; Yeh et al., 2024). Therefore, a typical post-training stage, referred to as preference optimization (e.g., RLHF and DPO), is additionally performed to align pre-trained language models with human intentions, and it has become a crucial aspect in the fine-tuning of large models, often indispensable. Currently, most approaches (Wu et al., 2023; Wang et al., 2024a; Meng et al., 2024) utilize KL divergence at the sentence level to ensure that the training model remains closely aligned with a reference model, preventing significant deviations. However, the generation of these responses occurs sequentially, following an auto-regressive approach. Recent works
(Zeng et al., 2024; Ouyang et al., 2024) introduce a fresh perspective, specifically the sequential and token-level direct preference optimization, which allows for examining KL divergence in relation to a reference LLM on a more granular, token-by-token basis. However, due to the neglect of the characteristics of a reward distribution other than the mean, these methods still suffer from the trouble of being insensitive to risk.

## 5.2. Risk-Aware Reinforcement Learning

Reinforcement learning has made groundbreaking achievements through approaches such as Q-learning (Mnih et al., 2015) and policy gradients (Schulman et al., 2015; 2017) in sequential decision tasks, but it also faces challenges when considering application in the real world (Mnih et al., 2015; Wang & Chapman, 2022). A primary reason is that the riskneutral criterion (maximizing the expectation) ignores the characteristics of a reward distribution other than the mean, which may be important for systems with safety concerns, especially in certain applications requiring tight risk control
(Fei et al., 2020; Bisi et al., 2022). In order to tackle this challenge, two types of risk-sensitive measures have been introduced: nested and static quantile risk-aware measures.

Static risk measures (Fei et al., 2021; Wang et al., 2023)
are straightforward to interpret, but the resulting optimal policy may not remain Markovian and may become historydependent. On the other hand, nested risk measures (Chen et al., 2024; Zhao et al., 2024) utilize MDPs to ensure risk sensitivity of the value iteration at each step under the current state, resulting in a more conservative approach. In this paper, we prefer nested risk measures because they recursively adhere to the Bellman equation and allow the MDPs to be reconstructed through state augmentation, enabling them to remain Markovian and ensuring that policy choices depend solely on the current state.

## 6. Conclusion

A pressing challenge arises for language generation tasks in the area of risk control, as the models, once trained, are often required to interact directly with humans. In this paper, we propose a novel direct preference optimization method that incorporates risk awareness by introducing nested risk measures into the Bellman equation, to align pre-trained LLMs with human preferences. Specifically, we design a new riskaware token-level objective function by reformulating the constrained reward maximization problem into a token-level form and then prove that maximizing this objective function leads to improvements in policy performance. Then, an optimization objective solely related to the risk-sensitive policy is obtained by deriving the mapping between the risk-aware state-action function and the optimal policy and establishing BT model equivalence with the Regret Preference Model. Finally, we conduct a formal analysis of this optimization objective and derive the loss function of Ra-DPO, which has practical implications for language generation tasks.

## References

Artzner, P. Thinking coherently. *Risk*, 10:68–71, 1997.

## Impact Statement

This paper presents work that aims to make LLMs more helpful and safer. Our work has many positive societal impacts, such as providing a theoretical foundation for riskaware language generation task, none of which we feel must be specifically highlighted. There are no negative societal impacts on our work. Artzner, P., Delbaen, F., Eber, J.-M., and Heath, D. Coherent measures of risk. *Mathematical finance*, 9(3):203–228, 1999.

Azar, M. G., Guo, Z. D., Piot, B., Munos, R., Rowland, M.,
Valko, M., and Calandriello, D. A general theoretical paradigm to understand learning from human preferences.

In *AISTATS*, 2024.

Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., Das-
Sarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. *arXiv preprint* arXiv:2204.05862, 2022.

Bauerle, N. and Rieder, U. More risk-sensitive markov ¨
decision processes. *Mathematics of Operations Research*, 39(1):105–120, 2014.

Bhardwaj, R. and Poria, S. Red-teaming large language models using chain of utterances for safety-alignment. arXiv preprint arXiv:2308.09662, 2023.

Biderman, S., Schoelkopf, H., Anthony, Q. G., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., et al. Pythia: A suite for analyzing large language models across training and scaling. In ICML, 2023.

Bisi, L., Santambrogio, D., Sandrelli, F., Tirinzoni, A.,
Ziebart, B. D., and Restelli, M. Risk-averse policy optimization via risk-neutral policy optimization. Artificial Intelligence, 311:103765, 2022.

440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 Bonetti, M., Bisi, L., and Restelli, M. Risk-averse optimization of reward-based coherent risk measures. Artificial Intelligence, 316:103845, 2023.

Bradley, R. A. and Terry, M. E. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

Candela, E., Doustaly, O., Parada, L., Feng, F., Demiris, Y.,
and Angeloudis, P. Risk-aware controller for autonomous vehicles using model-based collision prediction and reinforcement learning. *Artificial Intelligence*, 320:103923, 2023.

Chaudhary, S., Dinesha, U., Kalathil, D., and Shakkottai, S. Risk-averse fine-tuning of large language models. In NeurIPS, 2024.

Chen, Y., Du, Y., Hu, P., Wang, S., Wu, D., and Huang, L.

Provably efficient iterated cvar reinforcement learning with function approximation and human feedback. In ICLR, 2024.

Christiano, P. F., Leike, J., Brown, T. B., Martic, M., Legg, S., and Amodei, D. Deep reinforcement learning from human preferences. In *NeurIPS*, 2017.

Dai, J., Pan, X., Sun, R., Ji, J., Xu, X., Liu, M., Wang, Y.,
and Yang, Y. Safe rlhf: Safe reinforcement learning from human feedback. In *ICLR*, 2024.

Dubois, Y., Galambosi, B., Liang, P., and Hashimoto, T. B.

Length-controlled alpacaeval: A simple way to debias automatic evaluators. *arXiv preprint arXiv:2404.04475*,
2024.

Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D., and Kiela, D. Model alignment as prospect theoretic optimization. In *ICML*, 2024.

Fei, Y., Yang, Z., Chen, Y., Wang, Z., and Xie, Q. Risksensitive reinforcement learning: Near-optimal risksample tradeoff in regret. In *NeurIPS*, 2020.

Fei, Y., Yang, Z., and Wang, Z. Risk-sensitive reinforcement learning with function approximation: A debiasing approach. In ICML, 2021.

Fisch, A., Eisenstein, J., Zayats, V., Agarwal, A., Beirami, A., Nagpal, C., Shaw, P., and Berant, J. Robust preference optimization through reward model distillation. arXiv preprint arXiv:2405.19316, 2024.

Givan, R., Dean, T., and Greig, M. Equivalence notions and model minimization in markov decision processes. Artificial intelligence, 147(1-2):163–223, 2003.

Hau, J. L., Petrik, M., and Ghavamzadeh, M. Entropic risk optimization in discounted mdps. In *AISTATS*, pp. 47–76, 2023.

Huber, J., Payne, J. W., and Puto, C. Adding asymmetrically dominated alternatives: Violations of regularity and the similarity hypothesis. *Journal of consumer research*, 9 (1):90–98, 1982.

Lowd, D. and Davis, J. Learning markov network structure with decision trees. In *ICDM*, 2010.

Maas, A., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., and Potts, C. Learning word vectors for sentiment analysis. In ACL, pp. 142–150, 2011.

Meng, Y., Xia, M., and Chen, D. Simpo: Simple preference optimization with a reference-free reward. arXiv preprint arXiv:2405.14734, 2024.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M. A., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., and Hassabis, D. Human-level control through deep reinforcement learning. *Nature*, 518:529–533, 2015.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. In *NeurIPS*, 2022.

Ouyang, Y., Wang, L., Yang, F., Zhao, P., Huang, C., Liu, J.,
Pang, B., Yang, Y., Zhan, Y., Sun, H., et al. Token-level proximal policy optimization for query generation. arXiv preprint arXiv:2411.00722, 2024.

Peuter, S. D., Zhu, S., Guo, Y., Howes, A., and Kaski, S. Preference learning of latent decision utilities with a human-like model of preferential choice. In *NeurIPS*, 2024.

Pichler, A. and Schlotter, R. Entropy based risk measures.

European Journal of Operational Research, 285(1):223– 236, 2020.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D.,
Sutskever, I., et al. Language models are unsupervised multitask learners. *OpenAI blog*, 1(8):9, 2019.

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., and Finn, C. Direct preference optimization: your language model is secretly a reward model. In *NeurIPS*, 2023.

Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz, P. Trust region policy optimization. In *ICML*, 2015.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and finetuned chat models. *arXiv preprint arXiv:2307.09288*, 2023.

Tversky, A. and Kahneman, D. Advances in prospect theory:
Cumulative representation of uncertainty. *Journal of Risk* and uncertainty, 5:297–323, 1992.

Wang, C., Jiang, Y., Yang, C., Liu, H., and Chen, Y. Beyond reverse kl: Generalizing direct preference optimization with diverse divergence constraints. In *ICLR*, 2024a.

Wang, K., Kallus, N., and Sun, W. Near-minimax-optimal risk-sensitive reinforcement learning with cvar. In *ICML*, 2023.

Wang, Y. and Chapman, M. P. Risk-averse autonomous systems: A brief history and recent developments from the perspective of optimal control. *Artificial Intelligence*, 311:103743, 2022.

Wang, Z., Bi, B., Pentyala, S. K., Ramnath, K., Chaudhuri, S., Mehrotra, S., Mao, X.-B., Asur, S., et al. A comprehensive survey of llm alignment techniques: Rlhf, rlaif, ppo, dpo and more. *arXiv preprint arXiv:2407.16216*,
2024b.

Wu, Z., Hu, Y., Shi, W., Dziri, N., Suhr, A., Ammanabrolu, P., Smith, N. A., Ostendorf, M., and Hajishirzi, H. Finegrained human feedback gives better rewards for language model training. In *NeurIPS*, 2023.

Xiao, W., Wang, Z., Gan, L., Zhao, S., He, W., Tuan, L. A.,
Chen, L., Jiang, H., Zhao, Z., and Wu, F. A comprehensive survey of datasets, theories, variants, and applications in direct preference optimization. arXiv preprint arXiv:2410.15595, 2024.

Yeh, M.-H., Tao, L., Wang, J., Du, X., and Li, Y. How reliable is human feedback for aligning large language models? *arXiv preprint arXiv:2410.01957*, 2024.

Yuan, Z., Yuan, H., Tan, C., Wang, W., Huang, S., and Huang, F. Rrhf: Rank responses to align language models with human feedback without tears. arXiv preprint arXiv:2304.05302, 2023.

Zeng, Y., Liu, G., Ma, W., Yang, N., Zhang, H., and Wang, J. Token-level direct preference optimization. In *ICML*, 2024.

Zhang, L., Li, L., Wei, W., Song, H., Yang, Y., and Liang, J.

Scalable constrained policy optimization for safe multiagent reinforcement learning. In *NeurIPS*, 2024.

Zhao, W., He, T., and Liu, C. Model-free safe control for zero-violation reinforcement learning. In *CoRL*, 2021.

Zhao, Y., Escamilla, J. E. A., Lu, W., and Wang, H. Ra-pbrl:
Provably efficient risk-aware preference-based reinforcement learning. In *NeurIPS*, 2024.

495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549

## A. Supplementary Materials For Section 2

A.1. Risk Measure: A Brief Overview 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 Risk-aware Reinforcement Learning. Reinforcement learning has made groundbreaking achievements through approaches such as Q-learning (Mnih et al., 2015) and policy gradients (Schulman et al., 2015; 2017) in sequence decision tasks and has been gradually maturing in laboratory-level applications. In recent years, many researchers have gradually shifted their attention to real-world cyber-physical applications and found that focusing only on the mean of reward-to-go and corresponding Bellman equation is impractical, especially in some safety-critical scenarios requiring tight risk control, such as autonomous vehicle navigation (Candela et al., 2023) and robot control (Zhao et al., 2021; Zhang et al., 2024). A primary reason is that the risk-neutral criterion (maximizing the expectation) ignores the characteristics of a reward distribution other than the mean, which may be important for systems with safety concerns. For example, a system may be required to operate in a manner that alleviates harmful consequences, even in rare situations that are difficult to predict. To handle this kind of issue, some works (Wang & Chapman, 2022) introduce the worst-case criterion for autonomous systems with safety concerns to achieve zero-constraint violations by finding a policy that satisfies the constraints of a specific cost function, which generally assumes the maximum cost can quantify how bounded adversarial disturbances can inhibit the satisfactory operation of a system. However, due to the reliance on the typical assumption of bounded adversarial disturbances, the worst-case criterion may not be suitable for some applications that possess certain characteristics, such as the difficulty in characterizing the bounds of disturbances with a sufficient degree of certainty. Recently, risk-averse criterion (Bauerle & Rieder ¨ , 2014; Bisi et al., 2022), an intermediary criterion between the risk-neutral and worst-case criteria, has garnered extensive attention, which describes people or algorithms that prefer outcomes with reduced uncertainty by seeking to optimize risk metrics, such as entropy risk measures (ERM) (Pichler & Schlotter, 2020) or conditional value-at-risk (CVaR) (Artzner, 1997; Chen et al., 2024), of the possible cumulative reward which emphasizes its distributional characteristics.

In general, there are mainly two types of risk-sensitive measures: nested and static quantile risk-aware measures, each possessing distinct advantages and limitations. Static risk measures (Fei et al., 2021; Wang et al., 2023) are straightforward to interpret, but the resulting optimal policy may not remain Markovian and may become history-dependent. On the other hand, nested risk measures (Chen et al., 2024; Zhao et al., 2024) utilize MDPs to ensure risk sensitivity of the value iteration at each step under the current state, resulting in a more conservative approach. In this paper, we prefer nested risk measures because they recursively adhere to the Bellman equation and allow the MDPs to be reconstructed through state augmentation, enabling them to remain Markovian and ensuring that policy choices depend solely on the current state. Specifically, we introduce the popular CVaR (Artzner, 1997) objective as follows: and Φ
µ(Z) becomes

$$G(\xi)=\begin{cases}\frac{1}{\mu}\xi&\mathrm{if}\;\xi<\mu,\\ 1&\mathrm{if}\;\xi\geq\mu,\end{cases}$$
$$(21)^{\frac{1}{2}}$$
$$(20)^{\frac{1}{2}}$$
$$\Phi^{\mu}(Z)=\frac{1}{\mu}\int_{0}^{\mu}F_{Z}^{-1}(\xi)\mathrm{d}\xi,$$
Z(ξ)dξ, (21)
where G is LG-Lipschitz continuous for some LG ∈ R>0, and G(0) = 0, G(1) = 1.

Risks in LLMs Alignment. When aligning large language models with human preferences, there are many factors that may pose risks, primarily encompassing the following three types: (1) There exist conflicts and contradictions among human preferences (or choices), thus introducing uncertainty in the objectives when aligning models with human preferences. In addition, human choice behavior has contextual choice effects (Peuter et al., 2024), i.e., a decision maker's choice between two options is influenced by adding more options to the choice set (Huber et al., 1982). (2) Humans do not make decisions by maximizing their expected value for uncertain events; instead, they perceive random variables in a biased but well-defined manner (Ethayarajh et al., 2024). For example, relative to some reference point, humans are more sensitive to losses than gains, a phenomenon known as loss aversion. (3) Many popular methods, such as DPO (Rafailov et al., 2023), RDPO (Fisch et al., 2024), and simPO (Meng et al., 2024), utilize KL divergence to ensure that the training model remains closely aligned with a reference model during the training process, preventing significant deviations. These methods still face the issue of being insensitive to strategic risks because they only consider the mean of reward or utility and the corresponding Bellman equation, which is risk-neutral and does not capture the distribution characteristics of rewards efficiently. Since the first two types of risks stem from the distribution of preference data itself, in this article, we focus on the third type of risk, which comes from the process during model alignment. Specifically, we investigate a novel direct preference optimization method for the problem of aligning with human preferences from a risk-sensitive perspective and provide theoretical and empirical results on its performance and risk-awareness.

## A.2. The Expanded Version Of Value Function Definition

The definition of value function for nested risk measure, i.e., Eq. 5 in Subsection 2.3, can be expanded as

$$\begin{array}{c}{{Q_{\pi}\left(\left[x,y^{<t}\right],y^{\prime}\right)=R\left(\left[x,y^{<t}\right],y^{\prime}\right)+\Phi^{\mu}\left(R\left(\left[x,y^{<t+1}\right],\pi\left(\cdot\mid\left[x,y^{<t+1}\right]\right)\right)\right.}}\\ {{\left.\phantom{\left.+\Phi^{\mu}\left(\cdots\Phi^{\mu}\left(R\left(\left[x,y^{<T}\right],\pi\left(\cdot\mid\left[x,y^{<T}\right]\right)\right)\right)\right)\right),}}\end{array}$$
$$(222)$$
$$\begin{array}{c}{{V_{\pi}\left(\left[x,y^{<t}\right]\right)=R\left(\left[x,y^{<t}\right],\pi\left(\cdot\mid\left[x,y^{<t}\right]\right)\right)+\Phi^{\mu}\left(R\left(\left[x,y^{<t+1}\right],\pi\left(\cdot\mid\left[x,y^{<t+1}\right]\right)\right)\right.}}\\ {{\left.+\Phi^{\mu}\left(\cdots\Phi^{\mu}\left(R\left(\left[x,y^{<T}\right],\pi\left(\cdot\mid\left[x,y^{<T}\right]\right)\right)\right)\right)\right).}}\end{array}$$
$$(23)$$

Similarly, the definition of the optimal value function, can be expanded as

$$\begin{array}{c}{{Q_{\pi}^{\star}\left(\left[x,y^{<t}\right],y^{t}\right)=\max\left\{R\left(\left[x,y^{<t}\right],y^{t}\right)+\Phi^{\mu}\left(R\left(\left[x,y^{<t+1}\right],\pi\left(\cdot\mid\left[x,y^{<t+1}\right]\right)\right)\right)\right.}}\\ {{\left.\qquad\qquad\left.+\Phi^{\mu}\left(\cdots\Phi^{\mu}\left(R\left(\left[x,y^{<T}\right],\pi\left(\cdot\mid\left[x,y^{<T}\right]\right)\right)\right)\right)\right\},}}\end{array}$$
$$(24)^{\frac{1}{2}}$$
$$(25)$$

$$V_{\pi}^{\star}\left(\left[x,y^{<t}\right]\right)=\max\left\{R\left(\left[x,y^{<t}\right],\pi\left(\cdot\mid\left[x,y^{<t}\right]\right)\right)+\Phi^{\mu}\left(R\left(\left[x,y^{<t+1}\right],\pi\left(\cdot\mid\left[x,y^{<t+1}\right]\right)\right)\right.\right.$$ $$\left.\left.+\Phi^{\mu}\left(\cdots\Phi^{\mu}\left(R\left(\left[x,y^{<T}\right],\pi\left(\cdot\mid\left[x,y^{<T}\right]\right)\right)\right)\right)\right\}.$$

## B. Supplementary Materials For Section 3

B.1. The Proof of Lamma 3.1 P Lemma 3.2 Restated. For a given Pb-MDP, the reward on the entire prompt-response can be decomposed as r =
T
t=1 γ t−1R ([x, y<t] , yt), Vπ[x] in Eq. 5 and V˜π[x] in Eq. 6 are equivalent, which implies the following characteristics:
Proof. Firstly, according to (Givan et al., 2003; Lowd & Davis, 2010; Zhao et al., 2024), we can reformulate the Pb-MDP as a decision tree-like MDP. (1) The state transition graph of the Pb-MDP is connected and acyclic; (2) Each state in the Pb-MDP corresponds to a unique node in the tree; (3) There is a single root node from which every other node is reachable via a unique path; (4) The transition probabilities between states follow the Markov property, i.e., the probability of transitioning to any future state depends only on the current state and not on the sequence of events that preceded it.

605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659 Formally, let S be the set of states and pij be the transition probabilities between states si and sj . For an Pb-MDP with a tree-like structure, the probabilistic transition matrix P is defined such that:

$$p_{i j}{\textsf{7}}$$

pij > 0 if there is an edge between si and sj in the tree, and pij = 0 otherwise. (26)
Moreover, for each non-root node sj , there exists exactly one si such that pij > 0, and siis the unique parent of sj in the tree structure.

To classify the two value iteration in Eq. 5 and Eq. 6, we denote the value given by Eq. 6 as V˜π ([*x, y*<t]) and the value given by Eq. 5 as Vπ ([*x, y*<t]), thus, in tree-like Pb-MDP with the reward of the entire prompt-response, which can be decomposed as r =PT
t=1 γ t−1R ([x, y<t] , yt), we have the following relationship:

$$\tilde{V}_{\pi}\left(\left[x,y^{<t}\right]\right)=V_{\pi}\left(\left[x,y^{<t}\right]\right)+R_{1:t-1},$$
$\eqref{eq:walpha}$. 
12 where R1:t−1 =Pt−1 h=1 γ h−1R-x, y<h, yhdenotes the reward of the 1 ∼ t − 1 steps of a prompt-response. We prove this relationship by mathematical induction. Initial Case. Using the tree-like Pb-MDP and the initial conditions of the Bellman equation, at the final step t = T, we have

$$\tilde{V}_{\pi}\left(\left[x,y^{<T}\right]\right)=V_{\pi}\left(\left[x,y^{<T}\right],\pi\left(\cdot\mid\left[x,y^{<t}\right]\right)\right)+R_{1:T-1}$$ $$=V_{\pi}\left(\left[x,y^{<T}\right]\right)+R_{1:T-1}.$$
$$(27)$$
$$\Xi_{t}\left(s_{T,1}\right)=\Xi_{h}\left(s_{T,2}\right)\quad\forall\,s_{T,1},s_{T,2}\in\left\{s_{T}\mid S_{t}\left(s_{T}\right)=\left[x,y^{<t}\right]\right\}.$$

Therefore, R1:t−1 is unique.

## B.2. The Derivation Of Definition 3.2

Definition 3.3 Restated. For a risk-sensitive Pb-MDP that satisfies the Bellman equation in Eq. 6, the risk-aware advantage function can be defined as

$$\bar{A}_{\pi}\left(\left[x,y^{<t}\right],z\right)=\bar{Q}_{\pi}\left(\left[x,y^{<t}\right],z\right)-\Phi^{\mu}(\bar{V}_{\pi}\left(\left[x,y^{<t}\right]\right)),$$

where z subject to πθ (· | [*x, y*<t]).

660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714 In terms of designing the objective function at the token level, (Zeng et al., 2024) provides us with a valuable insight by introducing the advantage function from the TRPO algorithm in reinforcement learning as the target for each step. In this paper, building upon TDPO, we consider the risk associated with language generation at each step and devise a novel risk-sensitive advantage function. First, based on assumption that r =PT
t=1 γ t−1R ([x, y<t] , yt), we can get:
Next, note that y T = EOS denotes the end of the text sequence. Therefore,

$$(29)$$
$$(30)^{\frac{1}{2}}$$
$$V_{\pi}\left(\left[x,y^{<T+1}\right]\right)=\mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty}\gamma^{k}R\left(\left[x,y^{<T+1+k}\right],y^{T+1+k}\right)\mid s_{t}=\left[x,y^{<T+1}\right]\right]=0.$$

where the third equality holds because the risk measure function Φ satisfies translation invariance. Then, by applying conclusion, we observe that when t = 1, V˜π[x] = Vπ[x] hold on. Thus, we have proven that for the Pb-MDP, the reward of the entire trajectory can be decomposed as r =PT
t=1 γ t−1R ([x, y<t] , yt), and Vπ[x] in Eq. 5 and V˜π[x] in Eq. 6 are equivalent.

= ΦµV˜π ([x])+X T t=1 γ t−1R-x, y<t, yt+ γ Φ µV˜π-x, y<t+1− Φ µV˜π-x, y<t − γ T Φ µV˜π-x, y<T +1 = ΦµV˜π ([x])+ X T t=1 γ t−1Q˜π-x, y<t, yt− Φ µV˜π-x, y<t − γ T Φ µV˜π-x, y<T +1. t=1
$$\tilde{V}_{\pi}\left(\left[x,y^{<t}\right]\right)=\Phi^{\mu}\left(V_{\pi}\left(\left[x,y^{<t+1}\right]\right)+R_{1:t}\right),$$ $$=\Phi^{\mu}\left(V_{\pi}\left(\left[x,y^{<t+1}\right]\right)+R\left(\left[x,y^{<t}\right],\pi\left(\cdot\mid\left[x,y^{<t}\right]\right)\right)+R_{1:t-1}\right),$$ $$=\Phi^{\mu}\left(V_{\pi}\left(\left[x,y^{<t+1}\right]\right)+R\left(\left[x,y^{<t}\right],\pi\left(\cdot\mid\left[x,y^{<t}\right]\right)\right)\right)+R_{1:t-1},$$ $$=V_{\pi}\left(\left[x,y^{<t+1}\right]\right)+R_{1:t-1},$$
$$(28)$$
13

$$r=\sum_{t=1}^{T}\gamma^{t-1}R\left(\left[x,y^{<t}\right],y^{t}\right)$$ $$=\sum_{t=1}^{T}\gamma^{t-1}\left(R\left(\left[x,y^{<t}\right],y^{t}\right)+\gamma\,\Phi^{\mu}\left(\tilde{V}_{\pi}\left(\left[x,y^{<t+1}\right]\right)\right)-\gamma\,\Phi^{\mu}\left(\tilde{V}_{\pi}\left(\left[x,y^{<t+1}\right]\right)\right)\right).$$

Induction Step. We now proved that if V˜π-x, y<t+1 = Vπ-*x, y*<t+1 + R1:t holds, then V˜π ([*x, y*<t]) =
Vπ ([*x, y*<t]) + R1:t−1 also holds. Since this policy π on tree-like Pb-MDP is fixed, it has only one path to arrive t-th state (st = [*x, y*<t]), denoted as:
Furthermore, we have

$$r=\Phi^{\mu}\left(\tilde{V}_{\pi}\left([x]\right)\right)+\sum_{t=1}^{T}\gamma^{t-1}\left(\tilde{Q}_{\pi}\left(\left[x,y^{<t}\right],y^{t}\right)-\Phi^{\mu}\left(\tilde{V}_{\pi}\left(\left[x,y^{<t}\right]\right)\right)\right).\tag{31}$$

So, we definite the risk-aware advantage function as A˜π ([*x, y*<t] , z) = Q˜π ([x, y<t] , z) − Φ
µV˜π ([*x, y*<t]), where z ∼ πθ (· | [*x, y*<t]).

B.3. The Proof of Lemma 3.3 715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769 Lemma 3.4 Restated. Given two policies π and π
′, if for any state st = [*x, y*<t] , Ez∼π′
hA˜π ([x, y<t] , z)
i≥ 0 holds, then we can conclude:
Proof. Let trajectory τ := x, y1, y2*, . . .*, and the notation Eτ|π′ [·] indicates that actions are sampled from π
′to generate τ .

So we can get B.4. The Proof of Lemma 3.4 Lemma 3.5 Restated. The constrained problem in Eq. 9 has the closed-form solution:

$$\pi_{\theta}^{*}\left(z\mid[x,y^{<t}]\right)=\frac{\pi_{\mathrm{ref}}\left(z\mid[x,y^{<t}]\right)\exp\left(\frac{1}{\beta}\tilde{Q}_{\pi_{\mathrm{ref}}}\left([x,y^{<t}]\,,z\right)\right)}{Z\left([x,y^{<t}]\,;\beta\right)},$$

where Z ([*x, y*<t] ; β) = Ez∼πref (·|[x,y<t])e 1 β Q˜πref ([x,y<t],z)is the partition function.

14

$$\mathbb{E}_{x\sim{\mathcal{D}}}\left[{\tilde{V}}_{\pi^{\prime}}([x])\right]-\mathbb{E}_{x\sim{\mathcal{D}}}\left[{\tilde{V}}_{\pi}([x])\right]\geq0.$$
Ex∼D hV˜π′ ([x])i− Ex∼D hV˜π([x])i =Eτ|π′ "X ∞ t=1 γ t−1R-x, y<t, yt+ γ Φ µV˜π-x, y<t+1 − V˜π([x])# =Eτ|π′ "X ∞ t=1 γ t−1R-x, y<t, yt+ γ Φ µV˜π-x, y<t+1− Φ µV˜π-x, y<t# (32) =Eτ|π′ "X∞ t=1 γ t−1A˜π-x, y<t, yt# =Eτ|π′ "X∞ t=1 γ t−1Eyt∼π′ hA˜π-x, y<t, yti#.
Since for any state st = [*x, y*<t] , Ez∼π′
hA˜π ([x, y<t] , z)
i≥ 0, so we can obtain

$$\mathbb{E}_{x\sim{\mathcal{D}}}\left[\tilde{V}_{\pi^{\prime}}([x])\right]\geq\mathbb{E}_{x\sim{\mathcal{D}}}\left[\tilde{V}_{\pi}([x])\right].$$

Proof.

$$Z\left(\left[x,y^{<t}\right];\beta\right)=\mathbb{E}_{z\sim\pi_{\mathrm{ref}}\left(\cdot\left[x,y^{<t}\right]\right)}\exp\left(\frac{1}{\beta}\hat{Q}_{\pi_{\mathrm{ref}}}\left(\left[x,y^{<t}\right],z\right)\right).$$

Then, we can derive the relationship between the optimal policy and the state-action function:

$$\pi_{\theta}^{*}\left(z\mid\left[x,y^{<\epsilon}\right]\right)=\frac{\pi_{\mathrm{{ref}}}\left(z\mid\left[x,y^{<\epsilon}\right]\right)\exp\left(\frac{1}{\beta}\tilde{Q}_{\pi_{\mathrm{{ref}}}}\left(\left[x,y^{<\epsilon}\right],z\right)\right)}{Z\left(\left[x,y^{<\epsilon}\right];\beta\right)}.$$

B.5. The Proof of Lemma 3.5 Lemma 3.6 Restated. Given a reward function r, based on a relationship between token-wise rewards and the reward function represented by r =PT
t=1 γ t−1R ([x, y<t] , yt), we can establish the equivalence between the Bradley-Terry model and the Regret Preference Model in the language generation task, i.e., where σ(z) = 1/ (1 + exp(−z)) is the logistic sigmoid function for any random variable z. Proof. Recalling to the BT model in Eq. 40

$$P_{\mathrm{BT}}\left(y_{1}\succ y_{2}\mid x\right)={\frac{\exp\left(r\left(x,y_{1}\right)\right)}{\exp\left(r\left(x,y_{1}\right)\right)+\exp\left(r\left(x,y_{2}\right)\right)}},$$
and the equivalence between prompt-response reward and the risk-aware advantage function:
Then, we have

$$r=\Phi^{\mu}\left(\tilde{V}_{\pi}\left(\left[x\right]\right)\right)+\sum_{t=1}^{T}\gamma^{t-1}\left(\tilde{Q}_{\pi}\left(\left[x,y^{<t}\right],y^{t}\right)-\Phi^{\mu}\left(\tilde{V}_{\pi}\left(\left[x,y^{<t}\right]\right)\right)\right)$$ $$=\Phi^{\mu}\left(\tilde{V}_{\pi}\left(\left[x\right]\right)\right)+\sum_{t=1}^{T}\gamma^{t-1}\tilde{A}_{\pi}\left(\left[x,y^{<t}\right],y^{t}\right).$$
$$P_{\mathrm{BT}}\left(y_{1}\succ y_{2}\mid x\right)=\sigma\left(\sum_{t=1}^{T_{1}}\gamma^{t-1}\bar{A}_{\pi}\left(\left[x,y_{1}^{<t}\right],y_{1}^{t}\right)-\sum_{t=1}^{T_{2}}\gamma^{t-1}\bar{A}_{\pi}\left(\left[x,y_{2}^{<t}\right],y_{2}^{t}\right)\right).$$
$$\square$$

770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 810 811 812 813 814 815 816 817 818 819 820 821 822 823 824

$$P_{\mathrm{BT}}\left(y_{1}\succ y_{2}\mid x\right)=\sigma\left(\sum_{t=1}^{T_{1}}\gamma^{t-1}\bar{A}_{\pi}\left(\left[x,y_{1}^{<t}\right],y_{1}^{t}\right)-\sum_{t=1}^{T_{2}}\gamma^{t-1}\bar{A}_{\pi}\left(\left[x,y_{2}^{<t}\right],y_{2}^{t}\right)\right),$$
$$(36)$$
$$(37)$$
!, (36)
max
πθ
Ez∼πθ(·|[x,y<t])A˜πref -x, y<t, z− βDKL πθ· | -*x, y*<t ∥πref · | -*x, y*<t
= max
πθ
Ez∼πθ(·|[x,y<t]) 
Q˜πref -x, y<t, z− V˜πref -*x, y*<t+ β log πref (z | [*x, y*<t])
πθ (z | [*x, y*<t]) 

= max
πθβEz∼πθ(·|[x,y<t]) log πref (z | [*x, y*<t]) e
1
β Q˜πref ([x,y<t],z)
πθ (z | [*x, y*<t]) 
!
− V˜πref -*x, y*<t
= max
πθβEz∼πθ(·|[x,y<t]) log πref (z | [*x, y*<t]) e
1
β Q˜πref ([x,y<t],z)
Z ([*x, y*<t] ; β) πθ (z | [*x, y*<t]) 
!
− V˜πref -*x, y*<t + β log Z-*x, y*<t; β
= max
πθ
−βDKL
 
πθz |-*x, y*<t ∥
πref (z | [*x, y*<t]) e
1
β Q˜πref ([x,y<t],z)
Z ([*x, y*<t] ; β)
!
− V˜πref -*x, y*<t + β log Z-*x, y*<t; β,
where Z ([*x, y*<t] ; β) is the partition function:
15

$$(33)$$
$$(34)$$
$$(35)$$

B.6. The Proof of Theorem 3.6 Theorem 3.7 Restated. Given prompts x and pairwise responses (y1, y2), and the risk-aware objective function in Eq.

9, the Bradley-Terry model expresses the human preference probability in terms of the risk-aware optimal policy π
∗
θand reference policy πref:
P
∗
BT (y1 ≻ y2 | x) = σ (u
∗(*x, y*1, y2) − δ
∗(x, y1, y2)),
where u (x, y1, y2) represents the difference in implicit rewards defined by the risk-aware policy π
∗
θ and the reference policy πref, weighted by β, represented as

$$=\sigma\left(u^{*}\right)$$
u (x, y1, y2) = β log  πθ (y1 | x) πref (y1 | x) − β log  πθ (y2 | x) πref (y2 | x) ,
and δ (x, y1, y2) represents the difference in sequential risk ratio between two pairs (*x, y*1) and (*x, y*2), expressed as δ (x, y1, y2) = βDSeqRR (*x, y*2; πref | πθ) − βDSeqRR (*x, y*1; πref | πθ).

Proof. According to the Lemma 3.4, we have

$$\pi_{\theta}^{*}\left(z\mid\left[x,y^{<t}\right]\right)=\frac{\pi_{\mathrm{ref}}\left(z\mid\left[x,y^{<t}\right]\right)\exp\left(\frac{1}{\beta}\tilde{Q}_{\pi_{\mathrm{ref}}}\left(\left[x,y^{<t}\right],z\right)\right)}{Z\left(\left[x,y^{<t}\right];\beta\right)},$$
Z ([*x, y*<t] ; β), (38)
where $Z\left(\left[x,y^{<c}\right];\beta\right)=\mathbb{E}_{z\sim\mathbb{E}_{\text{ref}}\left(\left[x,y^{<c}\right]\right)}e^{\frac{1}{\beta}\hat{Q}_{\text{ref}}\left(\left[x,y^{<c}\right];\beta\right)}$ is the partition function. Rearrange Eq. 38, we obtain 
$$\begin{array}{l l}{{}}&{{\left[\left[x,y^{<t}\right],z\right)=\beta\log\frac{\pi_{\theta}^{*}\left(z\mid\left[x,y^{<t}\right]\right)}{\pi_{\mathrm{ref}}\left(z\mid\left[x,y^{<t}\right]\right)}+\beta\log Z\left(\left[x,y^{<t}\right];\beta\right).}}\end{array}$$
+ β log Z-*x, y*<t; β. (39)
From Lemma 3.5, we can get

$$P_{\mathrm{BT}}\left(y_{1}\succ y_{2}\mid x\right)=\sigma\left(\sum_{t=1}^{T_{1}}\left(\gamma^{t-1}\bar{A}_{\pi}\left(\left[x,y_{1}^{c\,t}\right],y_{1}^{t}\right)\right)-\sum_{t=1}^{T_{2}}\left(\gamma^{t-1}\bar{A}_{\pi}\left(\left[x,y_{2}^{c\,t}\right],y_{2}^{t}\right)\right)\right).$$
$$(38)$$
$$(39)$$

$$(40)$$
!. (40)
825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 By leveraging Eq. 39, we can derive

Note that
$$\mathbb{E}_{z\sim\pi_{\mathrm{ref}}}\left[\beta\log Z\left(\left[x,y^{<t}\right];\beta\right)\right]=\beta\log Z\left(\left[x,y^{<t}\right];\beta\right).$$
Therefore,

$$\sum_{t=1}^{T}\gamma^{t-1}\bar{A}_{\pi_{\mathrm{ref}}}\left(\left[x,y^{<t}\right],y^{t}\right)$$
t=1 =β X T t=1 γ t−1log π ∗ θ (y t| [x, y<t]) πref (y t| [x, y<t])  − Φ µ z∼πref  log π ∗θ (z | [x, y<t]) πref (z | [x, y<t])  =βX T t=1 γ t−1log  π ∗ θ(y t| [x, y<t]) πref (y t| [x, y<t])  + βX T t=1 γ t−1 Φ µ z∼πref  log π ∗ θ(z | [x, y<t]) πref (z | [x, y<t]) .

$$(42)$$
=X T t=1 γ t−1Qπref -x, y<t, yt− Φ µV˜πref -x, y<t =X T t=1 γ t−1Q˜πref -x, y<t, yt− Φ µQ˜πref -x, y<t, z =X T t=1 γ t−1 β log π ∗ θ (y t| [x, y<t]) π ref (y t| [x, y<t])  + β log Z-x, y<t; β− Φ µβ log π ∗ θ (z | [x, y<t]) π ref (z | [x, y<t])  + β log Z-x, y<t; β . (41)
$$(41)$$
$$\sum_{t=1}^{T}\gamma^{t-1}\bar{A}_{\pi_{\mathrm{ref}}}\left(\left[x,y^{<t}\right],y^{t}\right)$$

16 When substituting γ = 1 into the expression, we obtain a more concise form:

X T t=1 A˜πref -x, y<t, yt=βX T t=1 log  π ∗ θ(y t| [x, y<t]) πref (y t| [x, y<t])  + βX T t=1 Φ µ z∼πref  log π ∗ θ(z | [x, y<t]) πref (z | [x, y<t])  =β log π ∗ θ(y | x) πref (y | x) + DSeqRR (x, y; πref | π ∗ θ) ,
$$(44)^{\frac{1}{2}}$$

$$\alpha\left(x,y_{1};\pi_{\mathrm{ref}}\mid\pi_{\theta}\right).$$
$$(45)^{\frac{1}{2}}$$
$$(43)^{\frac{1}{2}}$$
where DSeqRR (*x, y*; πref | πθ) = PT
$$=\sum_{t=1}^{T}\Phi_{z\sim\pi_{\mathrm{ref}}}^{\mu}\left(\log\frac{\pi_{\mathrm{ref}}(z|x)}{\pi_{\theta}(z|x)}\right).$$
Then, we let

$$\begin{array}{c}{{u\left(x,y_{1},y_{2}\right)=\beta\log\frac{\pi_{\theta}\left(y_{1}\mid x\right)}{\pi_{\mathrm{ref}}\left(y_{1}\mid x\right)}-\beta\log\frac{\pi_{\theta}\left(y_{2}\mid x\right)}{\pi_{\mathrm{ref}}\left(y_{2}\mid x\right)},}}\\ {{\delta\left(x,y_{1},y_{2}\right)=\beta D_{\mathrm{SeqRR}}\left(x,y_{2};\pi_{\mathrm{ref}}\mid\pi_{\theta}\right)-\beta D_{\mathrm{SeqRR}}\left(x,y_{1};\pi_{\mathrm{ref}}\right),}}\end{array}$$
, (44)
δ (*x, y*1, y2) = βDSeqRR (*x, y*2; πref | πθ) − βDSeqRR (*x, y*1; πref | πθ). (45)
Substituting Eq. 43 into Eq. 40, we arrive at P
∗BT (y1 ≻ y2 | x) = σ (u
∗(x, y1, y2) − δ
∗(x, y1, y2)).

B.7. Algorithm In this subsection, we provide the main pseudocode for Risk-aware Direct Preference Optimization (Ra-DPO), as outlined in Algorithm 1.

880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934

## C. Supplementary Materials For Section 4

C.1. Additional experimental results In this paper, we evaluate the performance of our proposed algorithm, Ra-DPO (Algorithm 1 in the Appendix B.7), against baseline algorithms on several text tasks. Here, we provide some additional experimental results, which are illustrated in Figures 6-7. Algorithm 1 Risk-aware Direct Preference Optimization (Ra-DPO)
Input: Reference model πref, Policy model πθ, Coefficient α, β, Risk control parameter µ, Learning rate η Input: Dataset D =
n(x, yw, yl)
ioN
i=1 of size N, Method M
Initialize: πθ ← πref for each epoch do Sample mini-batch Dm = {(x, yw, yl)
m}
M
m=1 from D
Predict the probabilities πθ (yw | x) and πθ (yl| x) for (x, yw, yl) in the mini-batch Dm using the policy model Predict the probabilities πref (yw | x) and πref (yl| x) for (x, yw, yl) in the mini-batch Dm using the reference model Calculate the function u (x, yw, yl) = β log πθ(yw|x)
πref (yw|x) 
− β log πθ(yl|x)
πref (yl|x)
Compute the sequential risk ratio DSeqRR (x, yw; πref | πθ) for (*x, y*w) in the mini-batch Dm Compute the sequential risk ratio DSeqRR (*x, y*l; πref | πθ) for (*x, y*l) in the mini-batch Dm if Method M is Ra-DPO1 **then**
Calculate the function δ (x, yw, yl) = βDSeqRR (*x, y*l; πref | πθ) − βDSeqRR (*x, y*w; πref | πθ)
θ ← θ + η∇θE(x,yw,yl)∼Dm [log σ (u (x, yw, yl) − δ (x, yw, yl))]
else {Method M is Ra-DPO2}
Calculate the function δ2 (x, yw, yl) = βDSeqRR (*x, y*l; πref | πθ) − sg (βDSeqRR (*x, y*w; πref | πθ))
θ ← θ + η∇θE(x,yw,yl)∼Dm [log σ (u (x, yw, yl) − αδ2 (x, yw, yl))]
end if end for

935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989 0 20k 40k 60k 80k 100k 120k 140k 160k step 0.475 0.500 0.525 0.550 0.575 0.600 0.625 0.650 TDPO2 = 0.3 Ra-DPO2 = 0.3 = 0.99 Ra-DPO2 = 0.3 = 0.98 Ra-DPO2 = 0.3 = 0.97 Ra-DPO2 = 0.3 = 0.95 0 20k 40k 60k 80k 100k 120k 140k 160k step 0.0 0.2 0.4 0.6 0.8 1.0 0 20k 40k 60k 80k 100k 120k 140k 160k step 0.0 0.2 0.4 0.6 0.8 1.0 DS
eqKL
(x, y w; ref 
)

DS
eqKL(
x, yl; ref 
)

Re wa rd A
cc ura cy TDPO2 = 0.5 Ra-DPO2 = 0.5 = 0.99 Ra-DPO2 = 0.5 = 0.98 Ra-DPO2 = 0.5 = 0.97 Ra-DPO2 = 0.5 = 0.95 0 20k 40k 60k 80k 100k 120k 140k 160k step 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0 20k 40k 60k 80k 100k 120k 140k 160k step 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0 20k 40k 60k 80k 100k 120k 140k 160k step 0.475 0.500 0.525 0.550 0.575 0.600 0.625 DSeq KL(x
, y w; ref 
)

DSeqK
L(x, yl; ref 
)

Reward Ac curac y 0 20k 40k 60k 80k 100k 120k 140k 160k step 0.0 0.1 0.2 0.3 0.4 0.5 0.6 TDPO2 = 0.7 Ra-DPO2 = 0.7 = 0.99 Ra-DPO2 = 0.7 = 0.98 Ra-DPO2 = 0.7 = 0.97 Ra-DPO2 = 0.7 = 0.95 0 20k 40k 60k 80k 100k 120k 140k 160k step 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0 20k 40k 60k 80k 100k 120k 140k 160k step 0.45 0.50 0.55 0.60 0.65 DS
eqKL(x, y w; ref 
)

DSe qKL(
x, yl; ref 
)

Re wa rd Acc ura cy 0 20k 40k 60k 80k 100k 120k 140k 160k step 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0 20k 40k 60k 80k 100k 120k 140k 160k step 0.45 0.50 0.55 0.60 0.65 TDPO2 = 0.9 Ra-DPO2 = 0.9 = 0.99 Ra-DPO2 = 0.9 = 0.98 Ra-DPO2 = 0.9 = 0.97 Ra-DPO2 = 0.9 = 0.95 0 20k 40k 60k 80k 100k 120k 140k 160k step 0.0 0.1 0.2 0.3 0.4 0.5 DS
eqKL(x, y w; ref 
)

DS
eqKL(
x, yl; ref 
)

Re wa rd Acc ur acy 990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 0 20k 40k 60k 80k 100k 120k 140k 160k step 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 0 20k 40k 60k 80k 100k 120k 140k 160k step 0.45 0.50 0.55 0.60 0.65 TDPO2 = 0.3 Ra-DPO2 = 0.3 = 0.99 Ra-DPO2 = 0.3 = 0.98 Ra-DPO2 = 0.3 = 0.97 Ra-DPO2 = 0.3 = 0.95 0 20k 40k 60k 80k 100k 120k 140k 160k step 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 DSeqK
L(x, y w; ref 
)

DS
eqKL(
x, yl; ref 
)

Rew ard Acc ura cy 0 20k 40k 60k 80k 100k 120k 140k 160k step 0.425 0.450 0.475 0.500 0.525 0.550 0.575 0.600 0.625 TDPO2 = 0.5 Ra-DPO2 = 0.5 = 0.99 Ra-DPO2 = 0.5 = 0.98 Ra-DPO2 = 0.5 = 0.97 Ra-DPO2 = 0.5 = 0.95 0 20k 40k 60k 80k 100k 120k 140k 160k step 0.0 0.2 0.4 0.6 0.8 1.0 1.2 0 20k 40k 60k 80k 100k 120k 140k 160k step 0.0 0.2 0.4 0.6 0.8 1.0 1.2 DS
eqKL(x, y w; ref 
)

DSe qKL(
x, yl; ref 
)

Re wa rd Acc ura cy TDPO2 = 0.7 Ra-DPO2 = 0.7 = 0.99 Ra-DPO2 = 0.7 = 0.98 Ra-DPO2 = 0.7 = 0.97 Ra-DPO2 = 0.7 = 0.95 0 20k 40k 60k 80k 100k 120k 140k 160k step 0.0 0.2 0.4 0.6 0.8 1.0 0 20k 40k 60k 80k 100k 120k 140k 160k step 0.0 0.2 0.4 0.6 0.8 1.0 0 20k 40k 60k 80k 100k 120k 140k 160k step 0.45 0.50 0.55 0.60 DSeq KL(x
, 
y w; ref 
)

DSeqK
L(x, yl; ref 
)

Reward Ac curac y 0 20k 40k 60k 80k 100k 120k 140k 160k step 0.0 0.2 0.4 0.6 0.8 1.0 0 20k 40k 60k 80k 100k 120k 140k 160k step 0.425 0.450 0.475 0.500 0.525 0.550 0.575 0.600 0.625 TDPO2 = 0.9 Ra-DPO2 = 0.9 = 0.99 Ra-DPO2 = 0.9 = 0.98 Ra-DPO2 = 0.9 = 0.97 Ra-DPO2 = 0.9 = 0.95 0 20k 40k 60k 80k 100k 120k 140k 160k step 0.0 0.2 0.4 0.6 0.8 DSeqKL
(x, y w; ref 
)

DS
eqKL(
x, yl; ref 
)

Re wa rd A
cc ura cy