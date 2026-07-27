# Provably Mitigating Corruption, Overoptimization, And Verbosity Simultaneously In Offline And Online Rlhf/Dpo Alignment

## Anonymous Authors1 Abstract

Reinforcement learning from human feedback
(RLHF) and direct preference optimization (DPO) are important techniques to align large language models (LLM) with human preference. However, the quality of RLHF and DPO training is seriously compromised by C*orrupted* preference, reward O*veroptimization*, and bias towards V*erbosity*. To our knowledge, most existing works tackle only one of these important issues, and the few other works require much computation to estimate multiple reward models and lack theoretical guarantee of generalization ability. In this work, we propose RLHF-COV and DPO-COV algorithms that can simultaneously mitigate these three issues, in both offline and online settings. This ability is theoretically demonstrated by obtaining length-regularized generalization error rates for our DPO-COV algorithms trained on corrupted data, which match the best-known rates for simpler cases with clean data and without length regularization. Moreover, our DPO-COV algorithm is simple to implement without reward estimation, and is proved to be equivalent to our RLHF-COV
algorithm, which directly implies the equivalence between the vanilla RLHF and DPO algorithms.

## 1. Introduction

Reinforcement learning from human feedback (RLHF) has been widely used in robotics (Christiano et al., 2017; Bukharin et al., 2024), autonomous driving (Wang et al., 2024; Cao et al., 2024), large language models (LLM) (Ouyang et al., 2022; Bai et al., 2022b; Rafailov et al., 2023), image and video generation (Wallace et al., 2023; Liang et al., 2024; Liu et al., 2024b), etc. This work will focus on the application of RLHF to LLM alignment which makes 1Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author
<anon.email@domain.com>.

1 000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054 LLM more helpful, honest, and harmless (Ouyang et al., 2022; Bai et al., 2022b). LLM alignment has two critical steps. The first step is reward modeling, which estimates the reward model that measures the quality of LLM responses, based on human preference data. The second step is reinforcement learning (RL), which fine-tunes the LLM policy to generate responses with an improved expected value of the learned reward (Ouyang et al., 2022). Direct preference optimization (DPO) (Rafailov et al., 2023) further simplifies the standard RLHF process by directly fine-tuning the optimal policy without reward estimation. However, the LLM aligned by RLHF and DPO sometimes yields undesirable responses, due to the corruption, overoptimization, and **verbosity** issues, as introduced below. Corruption. The quality of preference data is essential in RLHF and DPO. However, preference labels given by human may be corrupted due to inexperience, inattention, personal bias, unclear context, and even malicious falsification (Bukharin et al., 2024). For instance, when fine-tuning LLM for automated content moderation on social media, malicious annotators may mislabel harmful contents like misinformation and hate speech as preferable, which misleads the LLM to generate such harmful contents. Therefore, robustness of RLHF and DPO to such corruption is critical, but is tackled by only a few recent works to our knowledge. For example, (Cheng et al., 2024; Mandal et al., 2024; Gao et al., 2024b) use confidence-based data filtering. (Ethayarajh et al., 2024) maximizes the utility function defined based on the prospect theory of human decision making (Tversky and Kahneman, 1992) to filter out noisy data. (Coste et al., 2024; Rame et al., 2024) estimate an ensemble of rewards. The recently proposed robust RLHF and robust DPO approaches in (Bukharin et al., 2024) use noise modeling to automatically select the outliers and the estimated reward provably converges to the true reward. Overoptimization. RLHF and DPO may overoptimize the reward model, yielding LLM responses of high estimated reward but low actual quality (Gao et al., 2023; Casper et al., 2023). Various methods have been proposed to tackle such overoptimization issue (a.k.a. reward hacking). For example, (Gao et al., 2023) uses larger reward model which significantly increases the computational cost of pretraining.

(Moskovitz et al., 2024) applies constraints to RLHF. The ΦPo method (Azar et al., 2024) optimizes a general preference function. (Eisenstein et al., 2024; Coste et al., 2024; Rame et al., 2024; Fisch et al., 2024; Zhai et al., 2023) use an ensemble of estimated rewards.

055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109 An emerging and popular strategy with provable generalization ability to solve overoptimization is to adopt a pessimistic (resp. an optimistic) approach for RLHF and DPO with offline (resp. online) data. Specifically, in the offline setting where only precollected offline preference data is available for training, there are many out-of-distribution samples about which we cannot obtain any information. Therefore, (Zhu et al., 2023; 2024; Liu et al., 2024c; Cen et al., 2024; Ji et al., 2024; Yang et al., 2024; Huang et al., 2024; Xiong et al., 2024; Ye et al., 2024; Fisch et al., 2024) apply pessimistic principle to RLHF or DPO which penalizes LLM from generating such unknown out-of-distribution responses and thus to mitigate overoptimization. Such pessimism principle has also been used in conventional offline RL (Xie et al., 2021; Jin et al., 2021; Rashidinejad et al., 2021; Bai et al., 2022a; Cheng et al., 2022). In contrast, in the online setting where online data can be collected from the up-to-date policy during the training process, optimistic approaches have been used to encourage the collection of unexplored samples to enrich data diversity in RLHF and DPO (Cen et al., 2024; Xie et al., 2024; Zhang et al., 2024; Ye et al., 2024; Xiong et al., 2024) as well as conventional RL (Wei et al., 2017; Zhong and Zhang, 2023; Liu et al., 2023a;b). Verbosity. LLM aligned by vanilla RLHF and DPO is likely to prefer verbose but possibly low-quality responses (Singhal et al., 2023; Chen et al., 2024; Liu et al., 2024a; Dong et al., 2024; Fisch et al., 2024). Multiple methods have been used to tackle verbosity. For example, (Shen et al., 2023; Chen et al., 2024) disentangle length-related reward component. (Guo et al., 2024) instructs the LLM to prefer concise response. (Eisenstein et al., 2024; Fisch et al., 2024; Chakraborty et al., 2024) estimate an ensemble of reward models. (Singhal et al., 2023; Liu et al., 2024a; Dong et al., 2024; Park et al., 2024) use length penalty and similarly (Meng et al., 2024) uses length normalization.

Our Motivation. However, to our knowledge, most existing works primarily tackle only one of these three issues (corruption, overoptimization and verbosity). The only method to our knowledge that has been used to tackle all these issues is to estimate an ensemble of reward models (Coste et al., 2024; Fisch et al., 2024; Eisenstein et al., 2024; Rame et al., 2024), which, however, requires much computation and lacks theoretical guarantee of generalization ability. Therefore, we are motivated to ask the following research question.

Q: Can we design RLHF and DPO algorithms that solve corruption, **overoptimization** and *verbosity* simultaneously with simple implementation and theoretical guarantee of generalization ability?

## 1.1. Our Contributions

We answer the above question affirmatively, by proposing RLHF-COV and DPO-COV algorithms that simultaneously mitigate Corruption, O*veroptimization* and V*erbosity* issues, in both offline and online settings. Specifically, we tackle C*orruption* by noise modeling, tackle O*veroptimization* by pessimistic and optimistic regularizers in the offline and online settings respectively, and tackle V*erbosity* by length regularizer. Our DPO-COV algorithms are almost as simple to implement as the vanilla DPO algorithm without reward model estimation. We prove that our RLHF-COV and DPO- COV are equivalent in the reward-induced policy space in both the offline and online settings. Since our RLHF-COV and DPO-COV algorithms generalize the vanilla RLHF and DPO algorithms respectively, our equivalence result implies that the vanilla RLHF and DPO algorithms are also equivalent. Moreover, we obtain the length-regularized generalization error rates of our DPO-COV algorithms on both offline and online datasets obtained from corrupted preference, and the rates match the existing results in the simple special case with clean dataset and without verbosity regularization. This theoretically demonstrates that our algorithms can simultaneously mitigate the Corruption, O*veroptimization* and V*erbosity* issues. In particular, the effect of noise modeling on the generalization error of learned policy for corrupted data has not been studied to our knowledge, which requires novel proof techniques. The true and estimated noise terms have very different effects on the generalization error, and thus have to be analyzed at different stages. To elaborate, the estimated noise has to be bounded before applying concentration inequality, such that this unbounded estimated noise term can be canceled out by the noise regularizer. In contrast, the true noise has to be bounded after applying the concentration inequality, since the concentration inequality bounds the distance between the true data distribution (with the true noise term) and the estimated data distribution.

## 2. Preliminaries

Reinforcement learning from human feedback (RLHF). A large language model (LLM) provides a random language response a ∈ X to any given language prompt x ∈ X
(for example, instruction or question) following the LLM's policy π(·|x). Fine-tuning LLM by reinforcement learning from human feedback (RLHF) consists of two critical steps: training reward model and reinforcement learning 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164

$$\min_{r\in\mathcal{R}}-\frac{1}{N}\sum_{i=1}^{N}\log\sigma[r(x_{i},a_{i}^{w})-r(x_{i},a_{i}^{\ell})].\tag{2}$$

Finally, given the estimated reward model r ∈ R, the optimal policy is obtained by the following optimization problem over the whole policy space Π
def 
=
{π|π(·|x) is a distribution over A for any x}. where ρ is the prompt distribution, πref is the reference policy obtained by supervised fine-tuning, and KL(p∥q) =
Pa∈A p(a) log p(a) q(a)
denotes the KL divergence between any pair of response distributions p, q and β > 0 is the regularizer coefficient which controls the trade-off between generating responses with high expected reward and bounded distance from the reference policy πref.

Direct preference optimization (DPO). As introduced above, classical RLHF requires two large-scale optimization problems to learn the reward model r and the optimal policy π respectively. DPO (Rafailov et al., 2023) is introduced to remove the reward learning step and thus reducing computation. To elaborate, note that the optimization problem (3) has the following analytical solution.

$$\pi(a|x)={\frac{\pi_{\mathrm{ref}}(a|x)}{Z(x)}}\exp\Big[{\frac{r(x,a)}{\beta}}\Big],$$
i, (4)
where Z(x) := Pa′∈A πref(a
′|x) exp[r(*x, a*′)/β] is the normalization factor. Conversely, given the optimal policy π, r(*x, a*) = β log π(a|x)
πref (a|x)
is a solution to Eq. (1).

Substituting this reward model into the MLE objective (3),
(Rafailov et al., 2023) develops the following simple DPO
objective which only requires policy training.

$$\operatorname*{min}_{\pi\in\Pi}-{\frac{1}{N}}\sum_{i=1}^{N}\log\sigma\left[\beta\log{\frac{\pi(a_{i}^{w}|x_{i})}{\pi_{\mathrm{ref}}(a_{i}^{w}|x_{i})}}\right.$$ $$\left.\qquad\qquad\qquad\qquad\qquad-\beta\log{\frac{\pi(a_{i}^{\ell}|x_{i})}{\pi_{\mathrm{ref}}(a_{i}^{\ell}|x_{i})}}\right].$$
$$(S)^{\frac{1}{2}}$$
i. (5)
However, this DPO objective and the aforementioned vanilla RLHF process are prone to suffer from *corrupted* preference, reward *overoptimization*, and bias towards *verbose* response. We will propose our novel variants of RLHF and DPO to solve the three issues simultaneously, for both offline and online settings, in Sections 3 and 4 respectively.

$$(1)$$

## 3. Our Offline Dpo-Cov Algorithm

In this section, we will derive our proposed offline RLHF- COV objective and offline DPO-COV algorithm (Algorithm 1) which simultaneously solve the Corruption, O*veroptimization* and V*erbosity* issues, and then obtain the generalization error rates of our offline DPO-COV algorithm.

## 3.1. Our Offline Rlhf-Cov Objective

Offline Data from *Corrupted* **Preference.**
Assumption 1. *The offline data* Ddef =
{xi, a
(1)
i, a
(−1)
i, yi}
N
i=1 = {xi, aw i
, aℓ i
, yi}
N
i=1 is generated from the following model with corrupted preference.

$$(3)$$
$$x_{i}\sim\rho,\quad a_{i}^{(-1)},a_{i}^{(1)}\sim\pi_{b}(\cdot|x_{i}),\tag{6}$$  $$\mathbb{P}(a_{i}^{(1)}\succ a_{i}^{(-1)})=\sigma[r^{*}(x_{i},a_{i}^{(1)})-r^{*}(x_{i},a_{i}^{(-1)})+\xi_{i}^{*}],\tag{7}$$

where πb *denotes the behavior policy and* ξ
∗
i ∈ R *denotes* the true preference noise for the i*-th sample. If* a
(1)
i ≻ a
(−1)
i, assign the label yi = 1 *and denote* a w i = a
(1)
ias the more preferable response and a ℓ i = a
(−1)
i*as the less preferable* response; Otherwise, let yi = −1, a w i = a
(−1)
i, a ℓ i = a
(1)
i.

The above assumption is very similar to that of offline vanilla RLHF and DPO, except that we add noise ξ
∗
ito the Bradley-Terry model (1) for each possibly corrupted sample i (Bukharin et al., 2024).

Based on Assumption 1, P(yi|a
(1)
i, a
$\begin{array}{cc}.\,a_i^{(-1)})&=\\ \in&\{-1,1\}^{\color{red}\dagger}.\\ \text{loglikelihood}\end{array}$
σ[r
∗(xi, aw
i
) − r
∗(xi, aℓi
) + yiξ
∗
i
], yi *∈ {−*1, 1}
Hence, we define a penalized negative log-likelihood
function of the labels {yi}
N
i=1 as follows.
$$\quad(4)$$

$${\mathcal{L}}_{N,\lambda}(r,\xi)\stackrel{\mathrm{def}}{=}-\frac{1}{N}\sum_{i=1}^{N}\log\sigma[r(x_{i},a_{i}^{w})-r(x_{i},a_{i}^{\ell})+y_{i}\xi_{i}]$$
1We corrected the mistake in (Bukharin et al., 2024) which uses P(yi|a
(1)
i, a
(−1)
i) = σ[r
∗(xi, aw i ) − r
∗(xi, aℓi ) + ξ
∗
i], yi ∈
{−1, 1} that yields Pyi∈{−1,1} P(yi|a
(1)
i, a
(−1)
i) ̸= 1.

$$\begin{array}{c}{{\operatorname*{max}_{\pi\in\Pi}\mathbb{E}_{x\sim\rho,a\sim\pi(\cdot|x)}[r(x,a)]}}\\ {{\qquad-\beta\mathbb{E}_{x\sim\rho}\mathrm{KL}\left[\pi(\cdot|x)\left|\left|\pi_{\mathrm{ref}}(\cdot|x)\right.\right|,\right.}}\end{array}$$

where σ(x)
def 
= 1/(1 + e
−x) and r
∗is the unknown true reward model. r
∗can be estimated by maximum likelihood estimation (MLE), that is, to minimize the following negative log-likelihood function over a certain reward model family R.

$$\mathbb{P}(a^{\prime}\succ a|x)=\sigma[r^{*}(x,a^{\prime})-r^{*}(x,a)]$$

(RL) (Ouyang et al., 2022). The reward model is denoted by a function r(x, a) ∈ R which measures the quality of the response a given the prompt x. To train the reward model, preference data D = {xi, aw i, aℓi}
N
i=1 of size N is collected where a pair of responses a w i, aℓ iare generated given each i-th prompt xi, and the response a w iis more preferable than a ℓi(i.e. a w i ≻ a ℓi). Such a pairwise preference is widely assumed to follow the Bradley-Terry model (Bradley and Terry, 1952), that is, given prompt x, the generated response a
′is more desirable than a with the following probability.

$$+\;\frac{\lambda}{N}\|\xi\|_{1},$$
∥ξ∥1, (8)
which, compared with the standard non-corrupted negative log-likelihood function (2), adds the estimated preference noise ξ = [ξ1*, . . . , ξ*N ] ∈ R
N and the noise regularizer
∥ξ∥1 =PN
i=1 |ξi| with coefficient λ > 0 to encourage the sparsity of the noise. Reward Estimation via Pessimistic MLE to Solve Overoptimization. After collecting offline data, the next step is to learn the reward model r. One may consider corrupted MLE objective minr∈R,ξ∈RN LN,λ(*r, ξ*) (Bukharin et al.,
2024) which generalizes the non-corrupted MLE objective (2). However, this corrupted MLE objective tend to overfit limited offline data (Gao et al., 2023; Zhu et al., 2024; Liu et al., 2024c; Cen et al., 2024; Xiong et al., 2024), producing an inaccurately estimated reward that leads to overoptimization. Therefore, we consider the following pessimistic MLE inspired by (Liu et al., 2024c; Cen et al., 2024; Ji et al., 2024; Yang et al., 2024).

$$\min_{r\in\mathcal{R},\xi\in\mathbb{R}^{N}}\Big\{\mathcal{L}_{N,\lambda}(r,\xi)+\eta\max_{\pi\in\Pi}V_{\beta}(\pi,r)\Big\},\tag{9}$$

where the pessimistic hyperparameter η ≥ 0 and

$$V_{\beta}(\pi,r)\stackrel{{\rm def}}{{=}}\mathbb{E}_{x\sim\rho,a\sim\pi(\cdot|x),a^{\prime}\sim\pi_{\rm base}(\cdot|x)}\left[r(x,a)-r(x,a^{\prime})\right]$$ $$\qquad-\beta\mathbb{E}_{x\sim\rho}{\rm KL}\left[\pi(\cdot|x)\big{|}\big{|}\pi_{\rm ref}(\cdot|x)\right]\tag{10}$$

165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 denotes the relative value of the policy π to a certain baseline policy πbase given the reward r. The regularizer maxπ∈Π Vβ(*π, r*) in Eq. (9) can be seen as the relative value of the optimal policy, and will help reduce the reward value r(*x, a*) of any sample *x, a* with small π*base*(a|x), so that the optimal policy π(a|x) given by Eq. (4) will also be reduced. In other words, such samples *x, a* are considered pessimistic and are thus discouraged from being generated by the learned policy π. Hence, the regularizer maxπ∈Π Vβ(*π, r*) is called the pessimistic regularizer. Furthermore, if we select πbase to represent the offline data distribution (see the end of Section 3.2 for the choice of πbase), then these samples *x, a* with small πbase(a|x) can be seen as out-of-distribution, so that such pessimism on the out-of-distribution samples mitigates the overoptimization issue which often results from overestimation of the reward on low-quality out-of-distribution samples (Liu et al., 2024c). Policy Training with Penalized *Verbosity*. The vanilla RLHF usually yields reward model r(x, a) that has bias towards long and detailed responses. To suppress verbose responses in the policy optimization step maxπ∈Π Vβ(*π, r*), we can replace the reward model r(*x, a*) with the proxy reward model rω(*x, a*) = r(*x, a*) − ω|a| where |a| is the length (i.e., number of tokens) of the response a and the hyperparameter ω ≥ 0 controls the length penalty strength

$$({\boldsymbol{\delta}})$$

(Singhal et al., 2023; Liu et al., 2024a; Dong et al., 2024; Park et al., 2024). In this way, the policy training objective Vβ(*π, r*) (defined by Eq. (10)) is generalized to the following length-regularized relative value function.

$$V_{\beta,\omega}(\pi,r)$$
$$\begin{array}{l}\mbox{def}\mathbb{E}_{x\sim\rho,a\sim\pi(\cdot|x),a^{\prime}\sim\pi_{\rm base}(\cdot|x)}\left[r(x,a)-\omega|a|-r(x,a^{\prime})\right.\\ \left.+\left.\omega|a^{\prime}|\right]-\beta\mathbb{E}_{x\sim\rho}{\rm KL}\left[\pi(\cdot|x)\right|\left|\pi_{\rm ref}(\cdot|x)\right|.\end{array}\right.\tag{11}$$

Replacing Vβ(*π, r*) with Vβ,ω(*π, r*) in the pessimistic MLE
objective (9), we propose offline RLHF-COV objective below.

## (Offline Rlhf-Cov):

min
r∈R,ξ∈RN
$$\operatorname*{max}_{\mathbb{R}^{N}}\left\{{\mathcal{L}}_{N,\lambda}(r,\xi)+\eta V_{\beta,\omega}(\pi,r)\stackrel{(\otimes),()}{=}}\right.$$
$+$$\eta$$\mathbb{E}_{x\sim\rho,a\sim\pi(\cdot|x),a^{\prime}\sim\pi_{\rm base}}(\cdot|x)$$[r(x,a)-\omega|a|-r(x,a^{\prime})+\omega|a^{\prime}|]$$\cdot$$N$
$$+\frac{1}{N}\sum_{i=1}^{N}\Bigl{\{}\lambda|\xi_{i}|-\log\sigma[r(x_{i},a_{i}^{w})-r(x_{i},a_{i}^{\ell})+y_{i}\xi_{i}]\Bigr{\}}\Bigr{\}}$$ $$-\beta\eta\mathbb{E}_{x\sim\rho}\mathrm{KL}\bigl{[}\pi(\cdot|x)\bigr{\|}\pi_{\mathrm{ref}}(\cdot|x)\bigr{]}.\tag{12}$$
Remark: Our offline RLHF-COV objective above simultaneously tackles the Corruption, O*veroptimization* and V*erbosity* issues, via noise modeling, pessimism and length penalty with controllable hyperparameters λ, η, ω respectively. Specifically, the length penalty is only added to Vβ,ω not LN,λ, because in the pessimistic MLE we still want to obtain a reward r possibly with length bias, and then verbosity is only suppressed in the policy optimization part maxπ∈Π Vβ,ω(π, r). When λ ≥ 1 and η = ω = 0, our offline RLHF-COV objective above reduces to the reward estimation (2) and policy optimization (3) in the vanilla RLHF.

## 3.2. Our Offline Dpo-Cov Algorithm

The offline RLHF-COV objective (12) involves minimax optimization over three high-dimensional variables *r, ξ, π*. As the first step to simplify this objective, we obtain the following proposition. Proposition 1. (π, r, ξ) is the solution to the offline RLHF- COV objective (12) *if and only if* π = πrdef
= arg maxπ′∈ΠVβ,ω(π
′, r), ξ = ξrdef =
arg minξ∈RN LN,λ(r, ξ) and r is the solution to the following optimization problem.

$$\min_{r\in\mathcal{R}}[\mathcal{L}_{N,\lambda}(r,\xi_{r})+\eta V_{\beta,\omega}(\pi_{r},r)].\tag{13}$$

In addition, πr and ξr,i (the i-th entry of ξr) have the following analytical solutions.

$$\pi_{r}(a|x){=}{\frac{\pi_{\mathrm{ref}}(a|x)}{Z_{r}(x)}}\exp\Big[{\frac{r(x,a)-\omega|a|}{\beta}}\Big],$$
$$(14)^{\frac{1}{2}}$$
$$\xi_{r,i}=y_{i}I\{\lambda<1\}$$
$$\begin{array}{l}{{=y_{i}T\{\lambda<1\}}}\\ {{\left[\log\left(\frac{1}{\lambda}-1\right)-r(x_{i},a_{i}^{w})+r(x_{i},a_{i}^{\ell})\right]_{+},}}\end{array}$$
, (15)
where Zr(x)
def 
=Pa′∈A πref(a
′|x) exp -r(x,a′)−ω|a
′| βis the normalization factor, I{λ < 1} equals 1 if λ < 1 and 0 otherwise, and [u]+ = max(u, 0) *for any* u ∈ R.

The above proposition simplifies the offline RLHF-COV objective (12) into the reward estimation problem (13). Next, we will transform it into our DPO-COV objective of the policy π. In Eq. (14), given π = πr, a solution to the reward model r is

$$r^{\pi}(x,a)\ {\stackrel{\mathrm{def}}{=}}\ \omega|a|+\beta\log\Big[{\frac{\pi(a|x)}{\pi_{\mathrm{ref}}(a|x)}}\Big].$$
i. (16)
With the above reward r π, the corresponding noise can also be parameterized by π as ξ π def = ξrπ , whose i-th entry has the following analytical solution based on Eqs. (15) and (16).

$$\xi_{i}^{\pi}\stackrel{{\rm def}}{{=}}\xi_{r^{\pi},i}=y_{i}I\{\lambda<1\}\Big{[}\log(\frac{1}{\lambda}-1)-\omega(|a_{i}^{w}|-|a_{i}^{\ell}|)$$ $$-\beta\log\big{(}\frac{\pi(a_{i}^{w}|x_{i})\pi_{\rm ref}(a_{i}^{\ell}|x_{i})}{\pi(a_{i}^{\ell}|x_{i})\pi_{\rm ref}(a_{i}^{w}|x_{i})}\big{)}\Big{]}_{+},\tag{17}$$

Substituting the above r πand ξ π iinto Eq. (13), we propose our DPO-COV objective as follows.

$$({\mathrm{Offline~DPO-COV}})\colon$$
$$\min_{\in\Pi_{\mathcal{R}}}\,\left\{\mathcal{L}_{N,\lambda}(r^{\pi},\xi^{\pi})+\eta V_{\beta,\omega}(\pi_{r^{\pi}},r^{\pi})=0\right\}$$
$-\beta\eta\mathbb{E}_x$. 
− βηEx∼ρ,a∼πbase(·|x)
$$\omega_{\rho,a\sim\pi_{\mathrm{base}}}(\cdot|x)\left[\log\pi(a|x)\right]$$
$$+\frac{1}{N}\sum_{i=1}^{N}\Big{[}\lambda|\xi_{i}^{\pi}|-\log\sigma\Big{(}\omega(|a_{i}^{w}|-|a_{i}^{\ell}|)$$ $$+\beta\log\frac{\pi(a_{i}^{w}|x_{i})\pi_{\rm ref}(a_{i}^{\ell}|x_{i})}{\pi(a_{i}^{\ell}|x_{i})\pi_{\rm ref}(a_{i}^{w}|x_{i})}\Big{)}+y_{i}\xi_{i}^{\pi}\Big{]}+C_{\rm off}\Big{\}},\tag{18}$$

where Coff def 
= βηEx∼ρ,a∼πbase(·|x)
-log πref(a|x)is a constant independent of π, and we use the reward-induced policy space ΠR
def 
= {πr : r *∈ R}* since the optimal policy is πr for some reward r based on Proposition 1. Note that such ΠR is sufficiently general to admit any parameterized policy πθ since by defining R = {r πθ: θ ∈ Θ}, we have ΠR = {πθ : θ ∈ Θ} based on Lemma 3.

Remark: Our proposed offline DPO-COV objective (18)
simultaneously tackles Corruption, O*veroptimization* and V*erbosity* issues. C*orruption* is modeled by the noise term ξ π = [ξ π 1
, . . . , ξπN ] which becomes sparser as the hyperparameter λ ≥ 0 increases, and ξ π = 0 when λ ≥ 1.

O*veroptimization* is tackled by the pessimistic regularizer
−βηEx∼ρ,a∼πbase(·|x)
-log π(a|x)which helps to increase

$$(15)$$

Algorithm 1 Offline DPO-COV Algorithm 1: **Inputs:** Hyperparameters *β, η, ω, λ* ≥ 0, offline data
{xi, aw i
, aℓ i
}
N
i=1, reference policy πref.

2: **Output:** Obtain policy πb via the following practical offline DPO-COV objective.

$$\begin{array}{l}{{\operatorname*{min}_{\pi\in\Pi_{\mathcal{R}}}\psi_{N}(\pi)\stackrel{\mathrm{def}}{=}\frac{1}{N}\sum_{i=1}^{N}\Big\{\lambda|\xi_{i}^{\pi}|-\beta\eta\log\pi(a_{i}^{w}|x)\Big\}}}\\ {{\quad\quad-\log\sigma\Big[\omega(|a_{i}^{w}|-|a_{i}^{\ell}|)}}\end{array}$$
$$(19)$$
$$+\;\beta\log\Big(\frac{\pi(a_{i}^{w}|x_{i})\pi_{\mathrm{ref}}(a_{i}^{\ell}|x_{i})}{\pi(a_{i}^{\ell}|x_{i})\pi_{\mathrm{ref}}(a_{i}^{w}|x_{i})}\Big)+y_{i}\xi_{i}^{\pi}\Big]\Big\},$$
ed by Eq. (17).  
io, (19)
$$(16)^{\frac{1}{2}}$$

where ξ π i is defined by Eq. (17).

π(a|x) for in-distribution samples (x, a) well covered by πbase. V*erbosity* is penalized by the length regularizers ω|a w i |, ω|a ℓ i |. When λ ≥ 1 and η = ω = 0, our above offline DPO-COV objective (18) reduces to the vanilla DPO objective (5).

We formally establish the equivalence between our offline RLHF-COV objective (12) and offline DPO-COV objective (18) in the following Proposition 2, which implies the equivalence between the vanilla RLHF and DPO algorithms as a special case when λ ≥ 1 and η = ω = 0. Proposition 2. A policy π ∈ Π is optimal for the offline DPO-COV objective (18) *if and only if there exist* r ∈ R, ξ ∈ R
N such that (π, r, ξ) is optimal for the offline RLHF-COV objective (12)*. In this case,* ξ = ξ π, and for any x ∈ X , there exists Uπ(x) ∈ R *such that* r(x, ·) = r π(x, ·) + Uπ(x).

As suggested by (Liu et al., 2024c; Yang et al., 2024) and discussed in Section 3.3, in the DPO-COV objective (18), we can take πbase(·|x) as the distribution of the preferable responses a w igiven xi = x under Assumption 1, and then adopt the simple stochastic approximation Ex∼ρ,a∼πbase(·|x)-log π(a|x)≈1N
PN
i=1 log π(a w i|xi).

This yields our fully stochastic offline DPO-COV algorithm as Algorithm 1, which only requires to solve the policy optimization problem that is almost as simple as the vanilla DPO objective (5).

## 3.3. Generalization Analysis Of Offline Dpo-Cov

While the policy π is trained from the offline data D, the ultimate goal is to make π generalize well to all possible prompts x ∼ ρ. Specifically, we define the following lengthregularized value function which characterizes the generalization ability of the policy π as a trade-off among the true reward value r
∗(response quality), the length of the 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 generated response a, and the policy's distance to πref.

$$J_{\beta,\omega}(\pi):=\mathbb{E}_{x\sim\rho,a\sim\pi(\cdot|x)}\bigg[r^{*}(x,a)-\omega|a|\bigg]$$
$$-\beta{\rm KL}\left[\pi(\cdot|x)\|\pi_{\rm ref}(\cdot|x)\right]\right].\tag{20}$$

To analyze the generalization error of the policy πb obtained from Algorithm 1, we make the standard assumptions below. Assumption 2 (Realizable and Bounded Reward (Zhu et al., 2023; Zhan et al., 2024; Cen et al., 2024; Ji et al., 2024; Liu et al., 2024c)). The reward model set R *includes the* true reward model r
∗*, that is,* r
∗ ∈ R*. Also, there exists a* constant R ∈ (0, +∞) such that for any x ∈ X , a ∈ A and r ∈ R, we have r(*x, a*) ∈ [0, R]. Assumption 3 (Offline Data Coverage (Zhan et al., 2024; Ji et al., 2024; Liu et al., 2024c)). *There exists a constant* GD ∈ (0, +∞) called offline coverage coefficient, such that the choice of the baseline policy πbase satisfies the following coverage property for all r ∈ R.

$$\mathbb{E}_{x\sim\rho,a\sim\pi_{r^{*}}(\cdot|x),a^{\prime}\sim\pi_{\rm{bann}}(\cdot|x)}$$ $$\left[r^{*}(x,a)-r^{*}(x,a^{\prime})-r(x,a)+r(x,a^{\prime})\right]\leq G_{\cal{D}}E_{r},\tag{21}$$

where Er def 
=-ED
r
∗(x1, aw 1
)−r
∗(x1, aℓ1
)−r(x1, aw 1
) +
r(x1, aℓ1)
21/2with the offline data sample x1, aw 1, aℓ1 generated via Assumption 1.

The offline coverage coefficient GD above describes how well the offline data D covers the responses from πbase and the true optimal policy πr
∗ ∈ arg maxπ∈ΠJβ,ω(π). Algorithm 1 takes πbase(·|x) as the distribution of the preferable responses a w igiven xi = x, which is well covered by D.

Theorem 1. Suppose Assumptions 1-3 hold and R is a convex set. For any δ ∈ (0, 1)*, select hyperparameters* λ ∈ [σ(R), 1], η =
2
√∥ξ
∗∥1+5 log[|N1/N (R)|/δ]
√N(3+eR). Then, the policy πe *from the offline DPO-COV objective* (18) has the following generalization error rate with probability at least 1 − δ.

$$\operatorname*{max}_{\pi\in\Pi}J_{\beta,\omega}(\pi)-J_{\beta,\omega}(\widetilde{\pi})$$
$$\leq\frac{(G_{\mathcal{D}}^{2}+1)(3+e^{R})}{\sqrt{N}}\sqrt{\|\xi^{*}\|_{1}+5\log[|{\mathcal{N}}_{1/N}({\mathcal{R}})|/\delta]},\tag{22}$$

where N1/N (R) is a (1/N)-cover of R*, that is, for any* r ∈ R*, there exists* r
† ∈ N1/N (R) *satisfying* ∥r
† − r∥∞ ≤
1/N.

Comparison with Existing Works. Note that |N1/N (R)*| ≤ O*[(RN)
|X ||A|] since R ⊂ [0, R]
|X ||A| by Assumption 2. Hence, as long as ∥ξ
∗∥1 ≤ O[log(N)]
(much weaker than Assumption 4.2 of (Bukharin et al.,
275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 2024) that there exist constants c0, c∞ > 0 such that ξ
∗ has at most c0 nonzero entries and they range in
[−c∞, c∞]), the generalization error rate (22) has the order of O[log(N)/
√N]. This rate matches the existing error rates of the offline pessimistic DPO-type algorithms (Liu et al., 2024c; Cen et al., 2024; Ji et al., 2024) up to logarithm, in the simple case with clean data (λ ≥ 1) and without length regularization (ω = 0). This implies that our offline DPO-COV algorithm provably mitigates O*veroptimization*. In addition, Theorem 1 also for the first time extends to the corrupted data and the length-regularized generalization error, which shows that our Algorithm 1 also mitigates C*orruption* and V*erbosity*. In particular, to mitigate C*orruption*, we use novel techniques below to bound the noise terms in the generalization error of the learned policy, whereas (Bukharin et al., 2024) only analyzes the estimation error of the reward and noise, but not that of the policy.

Technical Novelty. The proof logic of Theorem 1 is inspired from that of (Liu et al., 2024c), but our proof requires novel techniques to bound the effects of the true noise ξ
∗
and estimated noise ξ π. To elaborate, the ξ πis analyzed by our proposed Lemma 4, such that the error bound σ(R)|ξr,i| can later be canceled out by the regularizer −λ|ξr,i| when bounding the MLE error in Lemma 8. Next, we bound the distance between the true data distribution under (r
∗, ξ∗)
and the noiseless data distribution under the estimated r and ξ = 0 (see (c) of Eq. (43)) by concentration inequality. Then we bound ξ
∗ by our proposed Lemma 5 which has a different form from Lemma 4 used for bounding ξ π.

## 4. Our Online Dpo-Cov Algorithm

Compared with offline RLHF and DPO-type algorithms which use precollected offline data, the online algorithms improve the data coverage and the quality of the trained policy (Cen et al., 2024; Dong et al., 2024; Xu et al., 2024; Ye et al., 2024; Guo et al., 2024) at the computation cost of collecting the online preference data in the training process (Zhan et al., 2024; Ji et al., 2024; Huang et al., 2024; Mandal et al., 2024). Therefore, online and offline algorithms have different advantages, so both are important. In this section, we will derive our online RLHF-COV objective and online DPO-COV algorithm, and provide the generalization analysis result of our DPO-COV algorithm.

At each t-th iteration of our online algorithm, we use the current policy πt to obtain the t-th sample by xt ∼ ρ, a
(−1)
t ∼ πref(·|xt), a
(1)
t ∼ πt(·|xt), and the label yt is obtained from a stochastic oracle (such as GPT-4) assumed to follow the corrupted preference model (7). We propose the following online RLHF-COV objective to train the next policy πt+1 on the online data {xi, a
(−1)
i, a
(1)
i, yi}
t i=1.

(Online RLHF-COV):
πt+1∈arg minπ∈Π min
r∈R,ξ(t)∈Rt
nLt,λ(r, ξ(t))−ηVβ,ω(π, r)
(8),(11)
= βηEx∼ρKL-π(·|x)πref(·|x)
$$+\frac{1}{t}\sum_{i=1}^{t}\left\{\lambda|\xi_{i}|-\log\sigma[r(x_{i},a_{i}^{w})-r(x_{i},a_{i}^{\ell})+y_{i}\xi_{i}]\right\}\Biggr]$$
$$\L\left[\pi(\cdot|x)\|\pi_{\mathrm{ref}}(\cdot|x)\right]$$
$$\begin{array}{l}{{\mathrm{-}\,\eta\mathbb{E}_{x\sim\rho,a\sim\pi(\cdot|x),a^{\prime}\sim\pi_{\mathrm{base}}(\cdot|x)}}}\\ {{\left[r(x,a)+\omega|a|-r(x,a^{\prime})-\omega|a^{\prime}|\right],}}\end{array}$$
′|, (23)
330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 where ξ
(t) = [ξ1*, . . . , ξ*t] denotes the noise. The above online RLHF-COV objective is similar to the offline RLHF-COV objective (12) with the major difference that they tackle overoptimization in seemingly opposite ways. The offline RLHF-COV objective (12) (i.e.,
minr∈R,ξ∈RN [LN,λ(r, ξ) + η maxπ∈Π Vβ,ω(*π, r*)]) uses the pessimistic term +η maxπ∈Π Vβ,ω(π, r) to discourage LLM from generating out-of-distribution samples. In contrast, inspired by (Cen et al., 2024), our above online RLHF-COV objective (i.e., minr∈R,ξ∈RN [Lt,λ(*r, ξ*) −
η maxπ∈Π Vβ,ω(*π, r*)]) uses the sign-flipped optimistic term
−η maxπ∈Π Vβ,ω(*π, r*) to encourage LLM to collect out-ofdistribution samples to enrich the diversity of the online data to improve policy optimization. Similar to the offline DPO-COV objective (18), we obtain our online DPO-COV objective as follows.

## (Online Dpo-Cov):

πt+1 ∈ arg minπ∈ΠR
$$\L_{\mathrm{{I\!R}}}\left\{{\mathcal{L}}_{t,\lambda}(r^{\pi},\xi^{\pi,(t)})-\eta V_{\beta,\omega}(\pi_{r^{\pi}},r^{\pi})\right\}$$
$$-\log\sigma\Big{(}\omega(|a_{i}^{w}|-|a_{i}^{\ell}|)+\beta\log\frac{\pi(a_{i}^{w}|x_{i})\pi_{\rm ref}(a_{i}^{\ell}|x_{i})}{\pi(a_{i}^{\ell}|x_{i})\pi_{\rm ref}(a_{i}^{w}|x_{i})}\Big{)}$$ $$+y_{i}\xi_{i}^{\pi}\Big{]}+C_{\rm on}\Big{\}},\tag{2}$$
$$=\beta\eta\mathbb{E}_{x\sim\rho,a\sim\pi_{\mathrm{base}}(\cdot|x)}\left[\log\pi(a|x)\right]+\frac{1}{t}\sum_{i=1}^{t}\left[\lambda|\xi_{i}^{\pi}|x\right]\,.$$
$\phi$
Inspired by (Xie et al., 2024), we select πbase = πref and use its generated samples {a
(−1)
i}
t i=1 to approximate the Algorithm 2 Online DPO-COV Algorithm 1: **Inputs:** *β, η, ω, λ >* 0, reference policy πref, inital policy π0.

2: for Iterations t = 1*, . . . , T* do 3: Generate the t-th sample by xt ∼ ρ, a
(−1)
t ∼
πref(·|xt), a
(1)
t ∼ πt(·|xt), and label yt from a certain stochastic oracle assumed to follow the corrupted preference model (7).

4: Obtain πt+1 by solving the following stochastic online DPO-COV objective (25).

$$(23)$$
$$\min_{\pi\in\Pi_{\mathcal{R}}}\phi_{t}(\pi)=\frac{1}{t}\sum_{i=1}^{t}\left\{\lambda|\xi_{i}^{\pi}|+\beta\eta\log\pi(a_{i}^{(-1)}|x_{i})\right\}$$ $$-\log\sigma\Big{[}\omega(|a_{i}^{w}|-|a_{i}^{\ell}|)$$
$$+\beta\log\left(\frac{\pi(a_{i}^{w}|x_{i})\pi_{\rm ref}(a_{i}^{\ell}|x_{i})}{\pi(a_{i}^{\ell}|x_{i})\pi_{\rm ref}(a_{i}^{w}|x_{i})}\right)+y_{i}\xi_{i}^{\pi}\left]\right\},\tag{25}$$

5: **end for**
6: **Output:** πTb where Tb ∼ Uniform({2, 3*, . . . , T, T* +
1}).

expectation in the above online DPO-COV objective. This yields our fully stochastic online DPO-COV algorithm (Algorithm 2), which is also almost as simple to implement as the online vanilla DPO algorithm (Guo et al., 2024) (also Algorithm 2 with η = ω = 0 and λ = 1). To analyze the generalization error of Algorithm 2, define the following coverability coefficient (Xie et al., 2024),
which ensures that there exists at least one policy ν ∈ ΠR
with good coverage over the responses generated by any policy π ∈ ΠR.

$$G_{\rm on}=\inf_{\nu\in\Pi_{\cal R}}\sup_{x\in{\cal X},a\in{\cal A},\pi\in\Pi_{\cal R}}\frac{\pi(a|x)}{\nu(a|x)}.\tag{26}$$
$$(24)$$

Theorem 2. Under Assumption 2 *and for any* δ ∈
(0, 1)*, select hyperparameters* λ ∈ [σ(R), 1], η = 
√log[4TN1/T (R)/δ]+∥ξ
∗∥1
(3+eR)
√T Gon*where* ξ
∗ = [ξ
∗1, . . . , ξ∗T]. Then the output policy πTb of Algorithm 2 satisfies the following generalization error rate with probability at least 1 − δ.

$$\operatorname*{max}_{\pi\in\Pi}J_{\beta,\omega}(\pi)-\mathbb{E}\big[J_{\beta,\omega}(\pi_{\hat{T}})\big]\leq37(3+e^{R})(\log T)$$
$$\sqrt{\frac{G_{\rm on}}{T}\left[\log\left(\frac{4T|{\cal N}_{1/T}({\cal R})|}{\delta}\right)+\|\xi^{*}\|_{1}\right]}.\tag{27}$$

Remark: Theorem 2 above demonstrates that our online DPO-COV algorithm can simultaneously mitigate the Corruption, O*veroptimization* and V*erbosity* issues. When ∥ξ
∗∥1 ≤ O(log T), the above generalization error rate is Oe(1/
√T), which also matches the existing results of the 7 where ξ π,(t) def = [ξ π 1
, . . . , ξπ t
] is given by Eq. (17) and Con = −βηEx∼ρ,a∼πbase(·|x)[log πref(a|x)] is a constant independent of π. Similar to Proposition 2, we can show that the online RLHF-COV objective (23) and the online DPO-COV objective (24) are equivalent as follows.

Proposition 3. A policy π ∈ Π is optimal for the online DPO-COV objective (24) *if and only if there exist* r ∈ R, ξ ∈ R
N such that (π, r, ξ) is optimal for the offline RLHF-COV objective (23)*. In this case,* ξ = ξ π and for any x ∈ X , there exists Uπ(x) ∈ R *such that* r(x, ·) = r π(x, ·) + Uπ(x).

## 5.1. Experiment On The Argilla Data

| Algorithms                               | λ   | η      | ω      | LC-win rates   |
|------------------------------------------|-----|--------|--------|----------------|
| Our DPO-COV (all 3 components activated) | 0.7 | 0.0005 | 0.0005 | 7.61%          |
| Robust DPO (Corruption only)             | 0.1 | 0      | 0      | 7.04%          |
| Pessimistic DPO (Overoptimization only)  | 1   | 0.005  | 0      | 5.50%          |
| Length-regularized DPO (Verbosity only)  | 1   | 0      | 0.0005 | 7.30%          |
| Vanilla DPO                              | 1   | 0      | 0      | 6.29%          |
| Reference model πref                     | -   | -      | -      | 4.92%          |

Model GSM8K ARC ARC

(Easy) (Challenge)

Our DPO-COV 46.78 72.52 **49.32** Robust DPO 46.25 72.14 47.35 Pessimistic DPO 45.19 72.14 46.16 Length-reg DPO 44.50 72.31 46.16 Vanilla DPO 45.26 71.89 46.50 Reference Model 42.38 71.72 45.14

online optimistic DPO-type algorithms (Xie et al., 2024; Cen et al., 2024) up to logarithm. Technical Novelty. Similar to the proof of Theorem 1, we also use the novel bounds on the effect of the estimated and true noise terms, which are obtained in Lemmas 5 and 4 respectively.

## 5. Experiments On Offline Data

In this section, we will compare the following offline DPO-
type algorithms on offline datasets. The experiments to compare online DPO-type algorithms on online datasets are shown in Appendix A.

1. Our offline DPO-COV algorithm with three modules activated (Corruption, Overoptimization, V*erbosity*): This is Algorithm 1 with *η, ω >* 0 and λ ∈ (0, 1).

2. Offline robust DPO algorithm (Bukharin et al., 2024):
This is a special case of Algorithm 1 with η = ω = 0 and λ ∈ (0, 1), which only tackles C*orruption*.

5. Offline vanilla DPO (Rafailov et al., 2023): Algorithm 1 with η = ω = 0 and λ = 1.

We select the preference dataset D to be Argilla-DPO-Mix7K (Argill, 2024), and πref to be zephyr-7b-gemma-sftv0.1 (HuggingFaceH4, 2024), which is a fine-tuned version of gemma-7b on the Deita dataset (Wang et al., 2023). Then we apply LoRA (Hu et al., 2021) and two epochs of the AdamW optimizer (Loshchilov and Hutter, 2017) with learning rate 5 × 10−7to the objective (19). For each algorithm, we fix β = 0.05 and perform grid search on the other hyperparameters over a holdout validation set of the preference dataset. We compare the Length-Control win rates (a.k.a. LC-win rates, defined in AlpacaEval 2.0 (Dubois et al., 2024)) of πref and that of the models obtained by the above algorithms against the model GPT-4 Preview (11/06) (OpenAI, 2024). We summarize the LC-win rates and the hyperparameter values in Table 1, which indicates that our offline DPO-COV algorithm with all three components activated achieves the highest LC win rates. Therefore, it is important to tackle the Corruption, O*veroptimization* and V*erbosity* issues simultaneously.

## 5.2. Experiment On Math And Reasoning

We also compare our Algorithm 1 with other offline DPO variants over math and reasoning tasks: Grade School Math 8K (GSM8K) and AI2 Reasoning Challenge (ARC) tasks. We run the benchmark test with (Gao et al., 2024a) and report the accuracies in Table 2. The model hyper-parameters are the same as in Table 1. The results shown in Table 2 indicate that our DPO-COV algorithm outperforms the other variants also on the math and reasoning tasks.

## 6. Conclusion

We proposed RLHF-COV and DPO-COV algorithms that simultaneously mitigate the Corruption, O*veroptimization* and V*erbosity* issues, in both offline and online settings. This ability is theoretically proved by length-regularized generalization analysis on corrupted data. In addition, we proved the equivalence of our proposed RLHF-COV and DPO-COV algorithms. A future direction is to extend this work to account for various preferences among diverse human groups (Ramesh et al., 2024; Chakraborty et al., 2024).

3. Offline pessimistic DPO algorithm (Liu et al., 2024c):
This is a special case of Algorithm 1 with η > 0, ω = 0 and λ = 1, which only tackles O*veroptimization*.

4. Offline length regularized DPO algorithm (Park et al.,
2024): This is a special case of Algorithm 1 with η = 0, ω > 0 and λ = 1, which only tackles V*erbosity*.

385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

Argill (2024). Argilla-dpo-mix-7k. https://huggin gface.co/datasets/argilla/dpo-mix-7k. Accessed: 2024-09-30.

440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 Azar, M. G., Guo, Z. D., Piot, B., Munos, R., Rowland, M., Valko, M., and Calandriello, D. (2024). A general theoretical paradigm to understand learning from human preferences. In International Conference on Artificial Intelligence and Statistics (AISTATS), pages 4447–4455.

Bai, C., Wang, L., Yang, Z., Deng, Z.-H., Garg, A., Liu, P., and Wang, Z. (2022a). Pessimistic bootstrapping for uncertainty-driven offline reinforcement learning. In International Conference on Learning Representations (ICLR).

Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., Das-
Sarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al. (2022b). Training a helpful and harmless assistant with reinforcement learning from human feedback. ArXiv:2204.05862.

Bradley, R. A. and Terry, M. E. (1952). Rank analysis of incomplete block designs: I. the method of paired comparisons. *Biometrika*, 39(3/4):324–345.

Bukharin, A., Hong, I., Jiang, H., Zhang, Q., Zhang, Z.,
and Zhao, T. (2024). Robust reinforcement learning from corrupted human feedback. *ArXiv:2406.15568*.

Casper, S., Davies, X., Shi, C., Gilbert, T. K., Scheurer, J.,
Rando, J., Freedman, R., Korbak, T., Lindner, D., Freire, P., et al. (2023). Open problems and fundamental limitations of reinforcement learning from human feedback. Transactions on Machine Learning Research.

Chakraborty, S., Qiu, J., Yuan, H., Koppel, A., Huang, F.,
Manocha, D., Bedi, A. S., and Wang, M. (2024). Maxminrlhf: Towards equitable alignment of large language models with diverse human preferences. *ArXiv:2402.08925*.

Chen, L., Zhu, C., Chen, J., Soselia, D., Zhou, T., Goldstein, T., Huang, H., Shoeybi, M., and Catanzaro, B. (2024). Odin: Disentangled reward mitigates hacking in rlhf. In International Conference on Machine Learning (ICML).

Cheng, C.-A., Xie, T., Jiang, N., and Agarwal, A. (2022).

Adversarially trained actor critic for offline reinforcement learning. In International Conference on Machine Learning (ICML), pages 3852–3878. PMLR.

Cheng, J., Xiong, G., Dai, X., Miao, Q., Lv, Y., and Wang, F.-
Y. (2024). Rime: Robust preference-based reinforcement learning with noisy preferences. *ArXiv:2402.17257*.

Christiano, P. F., Leike, J., Brown, T. B., Martic, M., Legg, S., and Amodei, D. (2017). Deep reinforcement learning from human preferences. In *International Conference on* Neural Information Processing Systems (Neurips), pages 4302–4310.

Coste, T., Anwar, U., Kirk, R., and Krueger, D. (2024).

Reward model ensembles help mitigate overoptimization. In International Conference on Learning Representations (ICLR).

Dong, H., Xiong, W., Pang, B., Wang, H., Zhao, H., Zhou, Y., Jiang, N., Sahoo, D., Xiong, C., and Zhang, T. (2024). Rlhf workflow: From reward modeling to online rlhf. ArXiv:2405.07863.

Dubois, Y., Galambosi, B., Liang, P., and Hashimoto, T. B.

(2024). Length-controlled alpacaeval: A simple way to debias automatic evaluators.

Eisenstein, J., Nagpal, C., Agarwal, A., Beirami, A.,
D'Amour, A. N., Dvijotham, K. D., Fisch, A., Heller, K. A., Pfohl, S. R., Ramachandran, D., Shaw, P., and Berant, J. (2024). Helping or herding? reward model ensembles mitigate but do not eliminate reward hacking.

In *First Conference on Language Modeling (COLM)*.

Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D., and Kiela, D. (2024). Kto: Model alignment as prospect theoretic optimization. *ArXiv:2402.01306*.

Fan, K. (1953). Minimax theorems. Proceedings of the National Academy of Sciences, 39(1):42–47.

Fisch, A., Eisenstein, J., Zayats, V., Agarwal, A., Beirami, A., Nagpal, C., Shaw, P., and Berant, J. (2024). Robust preference optimization through reward model distillation. ArXiv:2405.19316.

Cen, S., Mei, J., Goshvadi, K., Dai, H., Yang, T., Yang, S., Schuurmans, D., Chi, Y., and Dai, B. (2024). Valueincentivized preference optimization: A unified approach to online and offline rlhf. *ArXiv:2405.19320*.

Cao, Y., Ivanovic, B., Xiao, C., and Pavone, M. (2024).

Reinforcement learning with human feedback for realistic traffic simulation. In IEEE International Conference on Robotics and Automation (ICRA), pages 14428–14434. IEEE.

495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 Gao, L., Schulman, J., and Hilton, J. (2023). Scaling laws for reward model overoptimization. In International Conference on Machine Learning (ICML), pages 10835– 10866.

Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPofi, A., Foster, C., Golding, L., Hsu, J., Le Noac'h, A., Li, H., McDonell, K., Muennighoff, N., Ociepa, C., Phang, J., Reynolds, L., Schoelkopf, H., Skowron, A., Sutawika, L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou, A. (2024a). A framework for few-shot language model evaluation. https://zenodo.org/records/1 2608602.

Gao, Y., Alon, D., and Metzler, D. (2024b). Impact of preference noise on the alignment performance of generative language models. In First Conference on Language Modeling (CoLM).

Guo, S., Zhang, B., Liu, T., Liu, T., Khalman, M., Llinares, F., Rame, A., Mesnard, T., Zhao, Y., Piot, B., et al. (2024). Direct language model alignment from online ai feedback.

ArXiv:2402.04792.

Harsha, P. (2011). Lecture note 12 on communication complexity. url: https://www.tcs.tifr.res.in/
~prahladh/teaching/2011-12/comm/lectu res/l12.pdf.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. (2021). Lora: Low-rank adaptation of large language models.

Huang, A., Zhan, W., Xie, T., Lee, J. D., Sun, W., Krishnamurthy, A., and Foster, D. J. (2024). Correcting the mythos of kl-regularization: Direct alignment without overparameterization via chi-squared preference optimization. *ArXiv:2407.13399*.

HuggingFaceH4 (2024). zephyr-7b-gemma-sft-v0.1. http s://huggingface.co/HuggingFaceH4/zep hyr-7b-gemma-sft-v0.1. Accessed: 2024-09-30.

Ji, X., Kulkarni, S., Wang, M., and Xie, T. (2024). Selfplay with adversarial critic: Provable and scalable offline alignment for language models. *ArXiv:2406.04274*.

Jin, Y., Yang, Z., and Wang, Z. (2021). Is pessimism provably efficient for offline rl? In *International Conference* on Machine Learning (ICML), pages 5084–5096.

Liang, Z., Yuan, Y., Gu, S., Chen, B., Hang, T., Li, J., and Zheng, L. (2024). Step-aware preference optimization: Aligning preference with denoising performance at each step. *ArXiv:2406.04314*.

Liu, J., Zhou, Z., Liu, J., Bu, X., Yang, C., Zhong, H.-S., and Ouyang, W. (2024a). Iterative length-regularized direct preference optimization: A case study on improving 7b language models to gpt-4 level. *ArXiv:2406.11817*.

Liu, Q., Weisz, G., György, A., Jin, C., and Szepesvári, C. (2023a). Optimistic natural policy gradient: a simple efficient policy optimization framework for online rl. In *International Conference on Neural Information* Processing Systems (Neurips), pages 3560–3577.

Liu, Y., Zhang, K., Li, Y., Yan, Z., Gao, C., Chen, R., Yuan, Z., Huang, Y., Sun, H., Gao, J., et al. (2024b). Sora: A review on background, technology, limitations, and opportunities of large vision models. *ArXiv:2402.17177*.

Liu, Z., Lu, M., Xiong, W., Zhong, H., Hu, H., Zhang, S.,
Zheng, S., Yang, Z., and Wang, Z. (2023b). Maximize to explore: one objective function fusing estimation, planning, and exploration. In *International Conference on* Neural Information Processing Systems (Neurips), pages 22151–22165.

Liu, Z., Lu, M., Zhang, S., Liu, B., Guo, H., Yang, Y.,
Blanchet, J., and Wang, Z. (2024c). Provably mitigating overoptimization in rlhf: Your sft loss is implicitly an adversarial regularizer. *ArXiv:2405.16436*.

Loshchilov, I. and Hutter, F. (2017). Decoupled weight decay regularization.

Mandal, D., Nika, A., Kamalaruban, P., Singla, A., and Radanovic, G. (2024). Corruption robust offline reinforce- ´ ment learning with human feedback. *ArXiv:2402.06734*.

Meng, Y., Xia, M., and Chen, D. (2024). Simpo: Simple preference optimization with a reference-free reward.

ArXiv:2405.14734.

Moskovitz, T., Singh, A. K., Strouse, D., Sandholm, T.,
Salakhutdinov, R., Dragan, A., and McAleer, S. M. (2024). Confronting reward model overoptimization with constrained rlhf. In *International Conference on Learning* Representations (ICLR).

OpenAI (2024). Gpt-4 technical report. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C.,
Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Gray, A., et al. (2022). Training language models to follow instructions with human feedback. In *International Conference* on Neural Information Processing Systems (Neurips).

Park, R., Rafailov, R., Ermon, S., and Finn, C. (2024). Disentangling length from quality in direct preference optimization. *ArXiv:2403.19159*.

Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D.,
Ermon, S., and Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. In International Conference on Neural Information Processing Systems (Neurips), volume 36.

Rame, A., Vieillard, N., Hussenot, L., Dadashi, R., Cideron, G., Bachem, O., and Ferret, J. (2024). Warm: On the benefits of weight averaged reward models. In International Conference on Machine Learning (ICML).

Ramesh, S. S., Hu, Y., Chaimalas, I., Mehta, V., Sessa, P. G., Ammar, H. B., and Bogunovic, I. (2024). Group robust preference optimization in reward-free rlhf. ArXiv:2405.20304.

Rashidinejad, P., Zhu, B., Ma, C., Jiao, J., and Russell, S. (2021). Bridging offline reinforcement learning and imitation learning: a tale of pessimism. In *International* Conference on Neural Information Processing Systems (Neurips), pages 11702–11716.

RLHFlow (2024). pair-preference-model-llama3-8b. http s://huggingface.co/RLHFlow/pair-prefe rence-model-LLaMA3-8B. Accessed: 2024-09-30.

Shen, W., Zheng, R., Zhan, W., Zhao, J., Dou, S., Gui, T.,
Zhang, Q., and Huang, X.-J. (2023). Loose lips sink ships: Mitigating length bias in reinforcement learning from human feedback. In *Findings of the Association for* Computational Linguistics: EMNLP 2023, pages 2859– 2873.

Singhal, P., Goyal, T., Xu, J., and Durrett, G. (2023). A
long way to go: Investigating length correlations in rlhf. ArXiv:2310.03716.

Tversky, A. and Kahneman, D. (1992). Advances in prospect theory: Cumulative representation of uncertainty.

Journal of Risk and uncertainty, 5:297–323.

Wallace, B., Dang, M., Rafailov, R., Zhou, L., Lou, A.,
Purushwalkam, S., Ermon, S., Xiong, C., Joty, S., and Naik, N. (2023). Diffusion model alignment using direct preference optimization. *AXiv:2311.12908*.

Wang, Y., Liu, L., Wang, M., and Xiong, X. (2024). Reinforcement learning from human feedback for lane changing of autonomous vehicles in mixed traffic. ArXiv:2408.04447.

Wang, Y., Liu, Q., and Jin, C. (2023). Is rlhf more difficult than standard rl? *ArXiv*, abs/2306.14111.

Wei, C.-Y., Hong, Y.-T., and Lu, C.-J. (2017). Online reinforcement learning in stochastic games. In International Conference on Neural Information Processing Systems (Neurips), pages 4994–5004.

Xie, T., Cheng, C. A., Jiang, N., Mineiro, P., and Agarwal, A. (2021). Bellman-consistent pessimism for offline reinforcement learning. In International Conference on 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 Neural Information Processing Systems (Neurips), pages 6683–6694.

Xie, T., Foster, D. J., Krishnamurthy, A., Rosset, C., Awadallah, A., and Rakhlin, A. (2024). Exploratory preference optimization: Harnessing implicit q*-approximation for sample-efficient rlhf. *ArXiv:2405.21046*.

Xiong, W., Dong, H., Ye, C., Wang, Z., Zhong, H., Ji, H.,
Jiang, N., and Zhang, T. (2024). Iterative preference learning from human feedback: Bridging theory and practice for rlhf under kl-constraint. In International Conference on Machine Learning (ICML).

Xu, W., Li, J., Wang, W. Y., and Li, L. (2024). Bpo: Supercharging online preference learning by adhering to the proximity of behavior llm. *ArXiv:2406.12168*.

Yang, R., Ding, R., Lin, Y., Zhang, H., and Zhang, T. (2024).

Regularizing hidden states enables learning generalizable reward model for llms. *ArXiv:2406.10216*.

Ye, C., Xiong, W., Zhang, Y., Jiang, N., and Zhang, T. (2024). Online iterative reinforcement learning from human feedback with general preference model. ArXiv:2402.07314.

Zhai, Y., Zhang, H., Lei, Y., Yu, Y., Xu, K., Feng, D., Ding, B., and Wang, H. (2023). Uncertainty-penalized reinforcement learning from human feedback with diverse reward lora ensembles. *ArXiv:2401.00243*.

Zhan, W., Uehara, M., Kallus, N., Lee, J. D., and Sun, W.

(2024). Provable offline preference-based reinforcement learning. In International Conference on Learning Representations (ICLR).

Zhang, S., Yu, D., Sharma, H., Yang, Z., Wang, S., Hassan, H., and Wang, Z. (2024). Self-exploring language models: Active preference elicitation for online alignment. ArXiv:2405.19332.

Zhang, T. (2023). Mathematical analysis of machine learning algorithms. Cambridge University Press.

Zhong, H. and Zhang, T. (2023). A theoretical analysis of optimistic proximal policy optimization in linear markov decision processes. In *International Conference on Neural* Information Processing Systems (Neurips), pages 73666– 73690.

Zhu, B., Jordan, M., and Jiao, J. (2023). Principled reinforcement learning with human feedback from pairwise or k-wise comparisons. In International Conference on Machine Learning (ICML), pages 43037–43067.

Zhu, B., Jordan, M., and Jiao, J. (2024). Iterative data smoothing: Mitigating reward overfitting and overoptimization in rlhf. In International Conference on Machine Learning (ICML).

# Appendix

## Table Of Contents

| Algorithms                               | λ   | η      | ω      | LC-win rates   |
|------------------------------------------|-----|--------|--------|----------------|
| Our DPO-COV (all 3 components activated) | 0.7 | 0.0005 | 0.0005 | 7.87%          |
| Robust DPO (Corruption only)             | 0.1 | 0      | 0      | 7.03%          |
| Optimistic DPO (Overoptimization only)   | 1   | 0.005  | 0      | 6.23%          |
| Length-regularized DPO (Verbosity only)  | 1   | 0      | 0.0005 | 6.19%          |
| Vanilla DPO                              | 1   | 0      | 0      | 6.58%          |
| Reference model πref                     | -   | -      | -      | 4.92%          |

605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659 Table 3: Hyperparameter Values and LC-win Rates of Online DPO-type Algorithms

## B. Supporting Lemmas

Lemma 1. For any A ∈ (0, ∞) and z1, z2 ∈ [−R, R]*, the following inequality holds.*

$${\frac{|z_{1}-z_{2}|}{3+e^{R}}}\leq|\sigma(z_{1})-\sigma(z_{2})|\leq{\frac{1}{4}}|z_{1}-z_{2}|.$$
|z1 − z2|. (28)
Similar to the offline experiments in Section 5, we compare important special cases of Algorithm 2, including our online DPO-COV with all 3 components activated, the online variant of the robust DPO algorithm (Bukharin et al., 2024),
online optimistic DPO algorithm (named XPO in (Xie et al., 2024)), online length regularized DPO algorithm (Liu et al., 2024a) and online vanilla DPO algorithm (using DPO objective in (Guo et al., 2024)). We use zephyr-7b-gemma-sftv0.1 (HuggingFaceH4, 2024) as the reference model πref and the initial model π0. Each algorithm is trained with β = 0.05 and T = 3 iterations. In each iteration, we generate the online labels yt from pair-preference-model-LLaMA3-8B (RLHFlow, 2024), and combine the online data with 50% of the preference dataset of Argilla-DPO-Mix-7K (Argill, 2024). Then we apply LoRA (Hu et al., 2021) and two epochs of the AdamW optimizer (Loshchilov and Hutter, 2017) with stepsize 5 × 10−7to the objective (25). On AlpacaEval 2.0 (Dubois et al., 2024), we compare the LC-win rates of πref and that of the models obtained by the above algorithms against the model GPT-4 Preview (11/06) (OpenAI, 2024). Again, the results in Table 3 indicate that our online DPO-COV algorithm with all three components activated achieves the highest length-control win rates. Therefore, it is important to tackle the Corruption, O*veroptimization* and V*erbosity* issues simultaneously.

12

## A. Experiment On Online Data

| A   | Experiment on Online Data   | 12   |
|-----|-----------------------------|------|
| B   | Supporting Lemmas           | 12   |
| C   | Proof of Proposition 1      | 23   |
| D   | Proof of Proposition 2      | 24   |
| E   | Proof of Proposition 3      | 25   |
| F   | Proof of Theorem 1          | 25   |
| G   | Proof of Theorem 2          | 26   |

Remark: Our bound (28) is strictly tighter than |z1−z2|
(1+eR)
2 ≤ |σ(z1) − σ(z2)*| ≤ |*z1 − z2| obtained in Lemma A.2 of (Liu et al., 2024c). Proof. Denote zmin = min(z1, z2) and zmax = max(z1, z2). Then we have

$$|z_{1}-z_{2}|=z_{\operatorname*{max}}-z_{\operatorname*{min}},$$
$\int_{\text{}}^{z_{\text{max}}}$. 
$\sigma(z_{1})-\sigma(z_{2})|=\sigma(z_{\max})-\sigma(z_{\min})=\int_{z_{\min}}^{z_{\max}}$
$$\stackrel{\mathrm{x}}{\sigma^{\prime}}(z)d z.$$

Hence, it suffices to prove that σ
′(v) ∈-1 3+eR ,
1 4 for any v ∈ [zmin, zmax] ⊂ [−*R, R*]. Note that for any v ∈ [zmin, zmax] ⊂
[−R, R], σ(v) ∈ [σ(−R), σ(R)] = [1 − σ(R), σ(R)]. Hence, we conclude the proof by the following two bounds.

r πr(*x, a*1) − r πr(*x, a*0) = r(x, a1) − r(*x, a*0), (29)
where πr and r π *are defined by Eqs.* (14) and (16) respectively. Furthermore, under Assumption 2, both sides of the above Eq. (29) *range in* [−R, R]. Proof.

$$r^{\pi r}(x,a_{1})-r^{\pi r}(x,a_{0})$$
$\frac{a}{\pi_{r}(a_{0}|x)\pi_{\rm ref}(a_{0}|x)}$  $\frac{a}{\pi_{r}(a_{0}|x)\pi_{\rm ref}(a_{1}|x)}$
$$\stackrel{(b)}{=}r(x,a_{1})-r(x,a_{0}),$$

where (a) uses Eq. (16) and (b) uses Eq. (14).

Furthermore, under Assumption 2, r(x, a0), r(x, a1) ∈ [0, R], so

$$r^{\pi r}(x,a_{1})-r^{\pi r}(x,a_{0})=r(x,a_{1})-r(x,a_{0})\in[-R,R].$$

Lemma 3. Any policy π ∈ Π satisfies π = πrπ where πr and r π *are defined by Eqs.* (14) and (16) respectively. Furthermore, under Assumption *2, any* π ∈ ΠR
def = {πr : r ∈ R} *satisfies* |r π(*x, a*1) − r π(x, a0)| ≤ R for any x ∈ X , a0, a1 ∈ A.

$$\sigma^{\prime}(v)=\frac{1}{4}-\left[\sigma(v)-\frac{1}{2}\right]^{2}$$ $$\geq\frac{1}{4}-\left[\sigma(R)-\frac{1}{2}\right]^{2}$$ $$=\sigma(R)[1-\sigma(R)]$$ $$=\frac{1}{1+e^{R}}\frac{e^{R}}{1+e^{R}}$$ $$=\frac{1}{(1+e^{R})(1+e^{-R})}$$ $$=\frac{1}{2+e^{R}+e^{-R}}\geq\frac{1}{3+e^{R}}.$$
$\square$
$$(29)^{\frac{1}{2}}$$
$\uparrow$). 
$$\sigma^{\prime}(v)=\sigma(v)[1-\sigma(v)]=\frac{1}{4}-\left[\sigma(v)-\frac{1}{2}\right]^{2}\leq\frac{1}{4}.$$
$\square$
Lemma 2. For any x ∈ X , a0, a1 ∈ A and r ∈ R*, the following equality holds* 660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714 Proof. Eq. (16) implies that for any x ∈ X and a ∈ A, we have

$$\pi_{\mathrm{ref}}(a|x)\exp\left[{\frac{r^{\pi}(x,a)-\omega|a|}{\beta}}\right]=\pi(a|x).$$
$$(30)$$
$$(31)$$

Hence,

$$Z_{r^{*}}(x)=\sum_{a\in\mathcal{A}}\pi_{\mathrm{ref}}(a|x)\exp\left[\frac{r^{\pi}(x,a)-\omega|a|}{\beta}\right]=\sum_{a\in\mathcal{A}}\pi(a|x)=1.$$

Therefore, π = πrπ can be proved as follows.

where (a) uses Eq. (14) and (b) uses Eqs. (30) and (31).

When π ∈ ΠR
def = {πr : r *∈ R}*, there exists r ∈ R such that π = πr. Hence,

$$|r^{\pi}(x,a_{1})-r^{\pi}(x,a_{0})|\stackrel{(a)}{=}|r^{\pi_{r}}(x,a_{1})-r^{\pi_{r}}(x,a_{0})|\stackrel{(b)}{=}|r(x,a_{1})-r(x,a_{0})|\stackrel{(c)}{\leq}R,$$

where (a) uses π = πr, (b) uses Eq. (29) and (c) uses Assumption 2.

Lemma 4. Under Assumption 2, for any r ∈ R and ξr,i *defined by Eq.* (15)*, the following inequality holds.*

$$\log\sigma[r(x_{i},a_{i}^{w})-r(x_{i},a_{i}^{\ell})+y_{i}\xi_{r,i}]\leq0$$

i) + yiξr,i] ≤ log σ[r(xi, aw i) − r(xi, aℓi)] + σ(R)|ξr,i|. (32)
For any π ∈ ΠR
def = {πr : r ∈ R} and ξ π i*defined by Eq.* (17)*, the following inequality holds.*
715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769

$$\log\sigma[r^{\pi}(x_{i},a_{i}^{w})-r^{\pi}(x_{i},a_{i}^{\ell})+y_{i}\xi_{i}^{\pi}]\leq\log\sigma[r^{\pi}(x_{i},a_{i}^{w})-r^{\pi}(x_{i},a_{i}^{\ell})]+\sigma(R)|\xi_{i}^{\pi}|.$$
i|. (33)
Proof. yiξr,i ≥ 0 by Eq. (15) since yi *∈ {−*1, 1}. Then Eq. (32) follows from d dv [log σ(v)] = σ(−v) ≤ σ(R) for any v ∈ [r(xi, aw i) − r(xi, aℓi), r(xi, aw i) − r(xi, aℓi) + yiξr,i] ⊆ [−R, +∞) where ⊂ is implied by Assumption 2.

Similarly, yiξ π i ≥ 0 by Eq. (17) since yi *∈ {−*1, 1}. Then Eq. (33) follows from d dv [log σ(v)] = σ(−v) ≤ σ(R) for any v ∈ [r π(xi, aw i
) − r π(xi, aℓ i
), rπ(xi, aw i
) − r π(xi, aℓi
) + yiξ π i
] ⊆ [−R, +∞) where ⊂ is implied by Lemma 3.

Lemma 5. For any ξi ∈ R and reward models r, r′: X × A → R*, we have*

$$\left\{\sigma[r^{\prime}(x_{i},a_{i}^{w})-r^{\prime}(x_{i},a_{i}^{\ell})+y_{i}\xi_{i}]-\sigma[r(x_{i},a_{i}^{w})-r(x_{i},a_{i}^{\ell})]\right\}^{2}$$ $$\geq\left\{\sigma[r^{\prime}(x_{i},a_{i}^{w})-r^{\prime}(x_{i},a_{i}^{\ell})]-\sigma[r(x_{i},a_{i}^{w})-r(x_{i},a_{i}^{\ell})]\right\}^{2}-\frac{1}{2}|\xi_{i}^{*}|.$$
$$(33)^{\frac{1}{2}}$$
$$(34)$$
$$f(u)=\left[\sigma(A_{i}^{\prime}+u)-\sigma(A_{i})\right]^{2}.$$
i + u) − σ(Ai)2. (35)
Note that the range of the sigmoid function σ is (0, 1). Hence, for any u ∈ R,

$${\frac{d}{d u}}f(u)=2\sigma(A_{i}^{\prime}+u)\big[1-\sigma(A_{i}^{\prime}+u)\big]\big[\sigma(A_{i}^{\prime}+u)-\sigma(A_{i})\big]\in\Big(-{\frac{1}{2}},{\frac{1}{2}}\Big).$$
. (36)
Therefore,

$$f(0)-f(y_{i}\xi_{i})\leq|f(y_{i}\xi_{i})-f(0)|\leq{\frac{1}{2}}|y_{i}\xi_{i}|={\frac{1}{2}}|\xi_{i}|,$$

which implies Eq. (34).

$$(35)^{\frac{1}{2}}$$
$$(36)$$

14 Proof. Denote A′i = r
′(xi, aw i) − r
′(xi, aℓ i) and Ai = r(xi, aw i) − r(xi, aℓ i). Define the following function.

$$\pi_{r^{\pi}}(a|x)\stackrel{(a)}{=}\frac{\pi_{\mathrm{ref}}(a|x)}{Z_{r^{\pi}}(x)}\exp\left[\frac{r^{\pi}(x,a)-\omega|a|}{\beta}\right]\stackrel{(b)}{=}\pi(a|x),$$
$\square$
$$(32)^{\frac{1}{2}}$$
$${\vec{x}})|\xi_{r,i}|.$$

Lemma 6. For any x ∈ X , a ∈ A and r, r′ ∈ R*, the policies* πr, πr
′ *defined by the analytical solution* (14) *satisfy*

$$\left|\log{\frac{\pi_{r^{\prime}}(a|x)}{\pi_{r}(a|x)}}\right|\leq{\frac{2\|r^{\prime}-r\|_{\infty}}{\beta}},$$

β, (37)
770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 where ∥r
′ − r∥∞ = supx∈X ,a∈A |r
′(x, a) − r(*x, a*)|.

Proof. Note that for any x ∈ X , a
′ ∈ A and r, r′ ∈ R, we have Therefore, As a result, which directly implies Eq. (37).

We slightly adjust Theorem 13.2 of (Zhang, 2023) as follows, by using filtration Ft = ∅ (so the conditional expectation becomes the total expectation), replacing −ξi with Zi, and negating the small probability event. Lemma 7. *Consider random variables* {Zi}
N
i=0. For any δ ∈ (0, 1) and λ
′ > 0, the following inequality holds simultaneously for all n = 1, 2, . . . , N *with probability at least* 1 − δ.

Lemma 8. Fix ϵ > 0, λ ∈ [σ(R), 1] and δ ∈ (0, 1). Under Assumption 1, the following bound holds for any r ∈ R and ξr = [ξr,1, . . . , ξr,N ] ∈ R
N *(given by Eq.* (15)) simultaneously with probability at least 1 − δ.

$${\mathcal{L}}_{N,\lambda}(r^{*},\xi^{*})-{\mathcal{L}}_{N,\lambda}(r,\xi_{r})\leq{\frac{2}{N}}\Big[\|\xi^{*}\|_{1}+\log\Big({\frac{|{\mathcal{N}}_{\epsilon}({\mathcal{R}})|}{\delta}}\Big)\Big]-{\frac{E_{\epsilon}^{2}}{2(3+e^{R})^{2}}}+7\epsilon,$$

where Er := 
qED
r
∗(x1, aw 1) − r
∗(x1, aℓ1) − r(x1, aw 1) + r(x1, aℓ1)
2and Nϵ(R) is a finite ϵ-cover of R*, that is, for any* r ∈ R*, there exists* r
† ∈ Nϵ(R) *satisfying* ∥r
† − r∥∞ ≤ ϵ.

Proof. Based on Assumption 1, given (xi, a
(1)
i, a
(−1)
i), the target label y *∈ {−*1, 1} as well as the underlying reward r and noise ξi, the event yi = y occurs with the following probability.

$$p_{r,\xi_{i}}(y|x_{i},a_{i}^{(1)},a_{i}^{(-1)})=\begin{cases}\sigma[r(x_{i},a_{i}^{(1)})-r(x_{i},a_{i}^{(-1)})+\xi_{i}],&y=1\\ \sigma[r(x_{i},a_{i}^{(-1)})-r(x_{i},a_{i}^{(1)})-\xi_{i}],&y=-1.\end{cases}$$
$$(40)$$

By merging the two cases above, we have

$$p_{r,\xi_{i}}(y_{i}|x_{i},a_{i}^{(1)},a_{i}^{(-1)})=\sigma[r(x_{i},a_{i}^{w})-r(x_{i},a_{i}^{\ell})+y_{i}\xi_{i}].$$
) + yiξi]. (41)
$$\begin{array}{c}{{\frac{\pi_{r^{\prime}}(a|x)}{\pi_{r}(a|x)}=\!\!\left(\frac{Z_{r^{\prime}}(x)}{Z_{r}(x)}\right)^{-1}\frac{\pi_{\mathrm{ref}}(a^{\prime}|x)\exp\left[\frac{r^{\prime}(x,a^{\prime})-\omega|a^{\prime}|}{\beta}\right]}{\pi_{\mathrm{ref}}(a^{\prime}|x)\exp\left[\frac{r(x,a^{\prime})-\omega|a^{\prime}|}{\beta}\right]}}}\\ {{\in\!\!\left[\exp(-2\|r^{\prime}-r\|_{\infty}/\beta),\exp(2\|r^{\prime}-r\|_{\infty}/\beta)\right]}}\end{array}$$
′ − r∥∞/β)(38)
$$\sum_{i=1}^{n}Z_{i}\leq{\frac{\log(1/\delta)}{\lambda^{\prime}}}+{\frac{1}{\lambda^{\prime}}}\sum_{i=1}^{n}\log\mathbb{E}[\exp(\lambda^{\prime}Z_{i})].$$

$$(38)$$
$$(39)$$
$$\begin{array}{c}{{\frac{Z_{r^{\prime}}(x)}{Z_{r}(x)}=\frac{\sum_{a^{\prime}\in{\mathcal{A}}}\pi_{\mathrm{ref}}(a^{\prime}|x)\exp\left[\frac{r^{\prime}(x,a^{\prime})-\omega|a^{\prime}|}{\beta}\right]}{\sum_{a^{\prime}\in{\mathcal{A}}}\pi_{\mathrm{ref}}(a^{\prime}|x)\exp\left[\frac{r(x,a^{\prime})-\omega|a^{\prime}|}{\beta}\right]}}}\\ {{\in\left[\exp(-\|r^{\prime}-r\|_{\infty}/\beta),\exp(\|r^{\prime}-r\|_{\infty}/\beta)\right].}}\end{array}$$
$$(41)^{\frac{1}{2}}$$

15

$${\frac{\pi_{\mathrm{ref}}\big(a^{\prime}|x\big)\exp\big[{\frac{r^{\prime}(x,a^{\prime})-\omega|a^{\prime}|}{\beta}}\big]}{\pi_{\mathrm{ref}}\big(a^{\prime}|x\big)\exp\big[{\frac{r(x,a^{\prime})-\omega|a^{\prime}|}{\beta}}\big]}}=\exp\Big[{\frac{r^{\prime}(x,a^{\prime})-r(x,a^{\prime})}{\beta}}\Big]$$ $$\in\big[\exp(-\|r^{\prime}-r\|_{\infty}/\beta),\exp(\|r^{\prime}-r\|_{\infty}/\beta)\big].$$
$$(37)$$

Define the following random variables for r ∈ R and i = 1, . . . , N.

Then the following inequality holds for finitely many r ∈ Nϵ(R) simultaneously with probability at least 1 − δ.

825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879

(b)
≤
1
N
X
N
i=1
-|ξ
∗
i
| + 2Zi(r)
(c)
≤
1
N
X
N
i=1
|ξ
∗
i
| + 2 log ED-exp[Zi(r)]	 +
2
N
log |Nϵ(R)|
δ

i=1
log ED
(Eyi∼pr∗,ξ∗
i
(·|xi,a
(1)
i,a
(−1)
i)
"vuut
pr,0(yi|xi, a
(1)
i, a
(−1)
i)
pr
∗,ξ∗
i
(yi|xi, a
(1)
i, a
(−1)
i)

xi, a
(1)
i, a
(−1)
i
#)
(d)
=
2
N
X
N
+
1
N
h∥ξ
∗∥1 + 2 log |Nϵ(R)|
δ
i
(e)
≤
2
N
X
N
i=1
ED
" X
y∈{−1,1}
qpr,0(y|xi, a
(1)
i, a
(−1)
i)pr
∗,ξ∗
i
(y|xi, a
(1)
i, a
(−1)
i) − 1
#
+
1
N
h∥ξ
∗∥1 + 2 log |Nϵ(R)|
δ
i
= −
1
N
X
N
i=1
ED
" X
y∈{−1,1}

qpr,0(y|xi, a
(1)
i, a
(−1)
i) −
qpr
∗,ξ∗
i
(y|xi, a
(1)
i, a
(−1)
i)

2#
+
1
N
h∥ξ
∗∥1 + 2 log |Nϵ(R)|
δ
i
(f)
≤ −1
4N
X
N
i=1
ED
" X
y∈{−1,1}
pr,0(y|xi, a
(1)
i, a
(−1)
i) − pr
∗,ξ∗
i
(y|xi, a
(1)
i, a
(−1)
i)
2
#
+
1
N
h∥ξ
∗∥1 + 2 log |Nϵ(R)|
δ
i
(g)
= −1
2N
X
N
i=1
EDσ[r
∗(xi, aw
i) − r
∗(xi, aℓi) + yiξ
∗
i] − σ[r(xi, aw
i) − r(xi, aℓ
i)]	2
+
1
N
h∥ξ
∗∥1 + 2 log |Nϵ(R)|
δ
i
(h)
≤ −1
2N
X
N
i=1
nEDσ[r
∗(xi, aw
i) − r
∗(xi, aℓ
i)] − σ[r(xi, aw
i) − r(xi, aℓi)]	2−
1
2
|ξ
∗
i|
o
+
1
N
h∥ξ
∗∥1 + 2 log |Nϵ(R)|
δ
i
(i)
≤ −1
2(3 + eR)
2
EDr
∗(x1, aw
1) − r
∗(x1, aℓ1) − r(x1, aw
1) + r(x1, aℓ1)
2
16
LN,λ(r ∗, ξ∗) − LN,λ(r, ξr) = 1 N X N i=1 logσ[r(xi, aw i)−r(xi, aℓ i)+yiξr,i]−logσ[r ∗(xi, aw i)−r ∗(xi, aℓi)+yiξ ∗ i]+λ(|ξ ∗ i |−|ξr,i|)	 (a) ≤ 1 N X N i=1 log σ[r(xi, aw i) − r(xi, aℓi)] + σ(R)|ξr,i| − log σ[r ∗(xi, aw i) − r ∗(xi, aℓ i) + yiξ ∗ i] + λ(|ξ ∗ i| − |ξr,i|)	
$$Z_{i}(r)=\frac{1}{2}\log\frac{\sigma[r(x_{i},a_{i}^{w})-r(x_{i},a_{i}^{\ell})]}{\sigma[r^{*}(x_{i},a_{i}^{w})-r^{*}(x_{i},a_{i}^{\ell})+y_{i}\xi_{i}^{*}]}=\frac{1}{2}\log\frac{p_{r,0}(y_{i}|x_{i},a_{i}^{(1)},a_{i}^{(-1)})}{p_{r^{*},\xi_{i}^{*}}(y_{i}|x_{i},a_{i}^{(1)},a_{i}^{(-1)})}.$$
$$\quad(42)$$
$$(43)$$
$$\begin{array}{c}{{+\,\frac{2}{N}\Big[\|\xi^{*}\|_{1}+\log\Big(\frac{|{\mathcal{N}}_{\epsilon}({\mathcal{R}})|}{\delta}\Big)\Big]}}}\\ {{\stackrel{(j)}{=}\frac{2}{N}\Big[\|\xi^{*}\|_{1}+\log\Big(\frac{|{\mathcal{N}}_{\epsilon}({\mathcal{R}})|}{\delta}\Big)\Big]-\frac{E_{r}^{2}}{2(3+e^{R})^{2}},}}\end{array}$$
, (43)
880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 where (a) uses Eq. (32) from Lemma 4, (b) uses Eq. (42) and σ(R) ≤ λ ≤ 1, (c) denotes ED as the expectation under Assumption 1 and (c) holds for finitely many r ∈ Nϵ(R) simultaneously with probability at least 1 − δ (by Lemma 7 with λ
′ = 1), (d) uses Eq. (42) and Assumption 1, (e) uses log v ≤ v − 1 for any v > 0, (f) uses Lemma 12.2 of (Harsha, 2011),
(g) uses Eq. (41), (h) uses Lemma 5, (i) uses Lemma 1 as well as the fact that the N samples {xi, aw i, aℓi}
N
i=1 are i.i.d., (j)
denotes Er := 
qED
r
∗(x1, aw 1) − r
∗(x1, aℓ1) − r(x1, aw 1) + r(x1, aℓ1)
2.

We have proved that with probability at least 1 − δ, the event E := {Eq. (43) holds for all r ∈ Nϵ(R) simultaneously} occurs. We will extend the range to any r ∈ R. By the definition of the ϵ cover Nϵ(R), there exists at least one r
† ∈ Nϵ(R)
such that ∥r
† − r∥∞ ≤ ϵ. Therefore,

LN,λ(r, ξr) − LN,λ(r †, ξr † ) (a) =  1 N X N i=1 log σ[r †(xi, aw i) − r †(xi, aℓ i) + ξr †,i] − log σ[r(xi, aw i) − r(xi, aℓi) + ξr,i]	 + λ N (∥ξr∥1 − ∥ξr † ∥1)  (b) ≤ 1 N X N i=1 h[r †(xi, aw i ) − r †(xi, aℓi ) + ξr †,i] − [r(xi, aw i ) − r(xi, aℓ i ) + ξr,i] + λ(|ξr,i| − |ξr †,i|) i ≤ 1 N X N i=1 hr †(xi, aw i) − r(xi, aw i) +r(xi, aℓ i) − r †(xi, aℓi) +ξr †,i − ξr,i + λ(|ξr,i − ξr †,i|) i (c) ≤ 1 N X N i=1 hr †(xi, aw i) − r(xi, aw i) +r(xi, aℓ i) − r †(xi, aℓi)
$$+\,(\lambda+1)\big|r(x_{i},a_{i}^{\ell})-r\big(x_{i},a_{i}^{w}\big)-[r^{\dagger}(x_{i},a_{i}^{\ell})-r^{\dagger}\big(x_{i},a_{i}^{w}\big)]\big|\big]\stackrel{(d)}{\leq}6\epsilon,$$
≤ 6ϵ, (44)
where (a) uses the definition of LN,λ given by Eq. (8), (b) uses triangle inequality and d dv [log σ(v)] = σ(−v) ∈ [0, 1] for any v ∈ R, (c) uses the property that ξr,i defined by Eq. (15) is a 1-Lipschitz continuous function of r(xi, aℓi
) − r(xi, aw i
)
(since max(·, 0) is 1-Lipschitz continuous), (d) uses ∥r
† − r∥∞ ≤ ϵ and λ ≤ 1. Under the event E, Eq. (43) holds with r replaced by r
+, which along with Eq. (44) implies the following inequality.

$$(44)$$
LN,λ(r ∗, ξ∗) − LN,λ(r, ξr) ≤[LN,λ(r †, ξr † ) − LN,λ(r, ξr)] + [LN,λ(r ∗, ξ∗) − LN,λ(r †, ξr † )] ≤6ϵ + 2 N h∥ξ ∗∥1 + log |Nϵ(R)| δ i −E2 r † 2(3 + eR) 2 =6ϵ + 2 N h∥ξ ∗∥1 + log |Nϵ(R)| δ i − E2 r † − E2 r 2(3 + eR) 2 −E2 r 2(3 + eR) 2 (a) ≤6ϵ + 2 N h∥ξ ∗∥1 + log |Nϵ(R)| δ i +4Rϵ (3 + eR) 2 −E2 r 2(3 + eR) 2 (b) ≤7ϵ + 2 N h∥ξ ∗∥1 + log |Nϵ(R)| δ i −E2 r 2(3 + eR) 2
, (45)
which proves Eq. (39). Here, (a) uses the following inequality and (b) uses (3 + e R)
2 > 6e R + e 2R > 6R + 2R = 8R.

$$|E_{r^{\dag}}^{2}-E_{r}^{2}|$$
$$|E_{r\uparrow}^{\varepsilon}-E_{r}^{\varepsilon}|$$ $$=\left|\mathbb{E}_{\mathcal{D}}\big{\{}\left[r^{*}(x_{1},a_{1}^{w})-r^{*}(x_{1},a_{1}^{\ell})-r^{\dagger}(x_{1},a_{1}^{w})+r^{\dagger}(x_{1},a_{1}^{\ell})\right]^{2}\big{\}}\right|$$
$$(45)$$

17

$$-\mathbb{E}_{\mathcal{D}}\big\{\big[r^{*}(x_{1},a_{1}^{w})-r^{*}(x_{1},a_{1}^{\ell})-r(x_{1},a_{1}^{w})+r(x_{1},a_{1}^{\ell})\big]^{2}\big\}\big|$$
$$=\left|\mathbb{E}_{\mathcal{D}}\big{\{}\big{[}r(x_{1},a_{1}^{w})-r(x_{1},a_{1}^{\ell})-r^{\dagger}(x_{1},a_{1}^{w})+r^{\dagger}(x_{1},a_{1}^{\ell})\big{]}\right.$$ $$\left.\left[2r^{*}(x_{1},a_{1}^{w})-2r^{*}(x_{1},a_{1}^{\ell})-r^{\dagger}(x_{1},a_{1}^{w})+r^{\dagger}(x_{1},a_{1}^{\ell})-r(x_{1},a_{1}^{w})+r(x_{1},a_{1}^{\ell})\right]\right\}\right|$$
$$\stackrel{(a)}{\leq}(2\epsilon)(4R)=8R\epsilon,$$

where (a) uses Assumption 2 and ∥r
† − r∥∞ ≤ ϵ.

Lemma 9. Fixing any ϵ > 0, δ ∈ (0, 1), the online dataset {xi, aw i, aℓi, yi}
T
i=1 generated from Algorithm 2 satisfies the following bound for all t = 1, . . . , T and π ∈ ΠR
def = {πr : r ∈ R} *simultaneously with probability at least* 1 − δ.

Proof. Define the following function.

935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989 where pr,ξi(yi|xi, a
(1)
i, a
(−1)
i) is defined by Eq. (41).

For any r ∈ R, there exists r
† ∈ Nϵ(R) satisfying ∥r
† − r∥∞ ≤ ϵ, and thus we can temporarily denote ru = urπr† + (1 −
u)r π(u ∈ [0, 1]). Then we obtain that

 d du  log σ-ru(xi, aw i ) − ru(xi, aℓ i ) =σ-ru(xi, aℓi) − ru(xi, aw i)r πr†(xi, aw i) − r πr†(xi, aℓi) − r π(xi, aw i) + r π(xi, aℓi) (a) ≤r †(xi, aw i) − r †(xi, aℓ i) − r π(xi, aw i) + r π(xi, aℓi) ≤r †(xi, aw i) − r π(xi, aw i) +r π(xi, aℓ i) − r †(xi, aℓi) ≤ 2ϵ, (49)
$$f_{\pi}(x,a^{(1)},a^{(-1)})\stackrel{\mathrm{def}}{=}r^{*}(x,a^{(1)})-r^{*}(x,a^{(-1)})-r^{\pi}(x,a^{(1)})+r^{\pi}(x,a^{(-1)}),$$
qπ,ξi(yi|xi, a (1) i, a (−1) i) def =   σ β log π(a (1) i |xi) πref (a (1) i|xi) − β log π(a (−1) i|xi) πref (a (−1) i|xi) + ω(|a (1) i| − |a (−1) i|) + ξi , yi = 1 σ β log π(a (−1) i|xi) πref (a (−1) i|xi) − β log π(a (1) i |xi) πref (a (1) i |xi) + ω(|a (−1) i| − |a (1) i|) − ξi , yi = −1. =σ-r π(xi, aw i) − r π(xi, aℓ i) + yiξi , (47)
$$W_{i}(\pi)=\frac{1}{2}\log\frac{\sigma\left[r^{\pi}(x_{i},a_{i}^{w})-r^{\pi}(x_{i},a_{i}^{\prime})\right]}{\sigma\left[r^{\star}(x_{i},a_{i}^{w})-r^{\star}(x_{i},a_{i}^{\prime})+y_{i}\xi_{i}^{\star}\right]}=\frac{1}{2}\log\frac{q_{\pi,0}(y_{i}|x_{i},a_{i}^{(1)},a_{i}^{(-1)})}{p_{r^{\star},\xi_{i}^{\star}}(y_{i}|x_{i},a_{i}^{(1)},a_{i}^{(-1)})},$$
, (48)
where the second = uses Eq. (16) and merges the above two cases. The above qπ,ξi(yi|xi, a
(1)
i, a
(−1)
i) can be seen as a conditional probability of yi *∈ {−*1, 1} since qπ,ξi(1|xi, a
(1)
i, a
(−1)
i) + qπ,ξi(−1|xi, a
(1)
i, a
(−1)
i) = 1.

Then define the following random variables for i = 1*, . . . , T*.

Xt i=1 log σ-r π(xi, aw i ) − r π(xi, aℓ i ) + yiξ π i  σ-r ∗(xi, aw i ) − r ∗(xi, aℓi ) + yiξ ∗ i  ≤2 log T|Nϵ(R)| δ + 4tϵ +X t i=1 n1 4 |ξ ∗ i| + σ(R)|ξ π i| −1 2(3 + eR) 2 Ex∼ρ,a(1)∼πi(·|x),a(−1)∼πref (·|x) -f 2 π(x, a(1), a(−1))o,
$$\square$$
$$(46)$$
$$(47)$$
where the function fπ is defined below and Nϵ(R) is a finite ϵ-cover of R, that is, for any r ∈ R*, there exists* r
† ∈ Nϵ(R)
satisfying ∥r
† − r∥∞ ≤ ϵ.

$$(48)$$

$$(49)$$

18 where (a) uses Eq. (29) and σ(x) ∈ (0, 1) for any x ∈ R. Therefore,

(a) = 1 2 hlog qπr† ,0(yi|xi, a (1) i, a (−1) i) − log qπ,0(yi|xi, a (1) i, a (−1) i) i (b) = 1 2  log σ-r πr†(xi, aw i ) − r πr†(xi, aℓ i )− log σ-r π(xi, aw i ) − r π(xi, aℓ i ) (c) = 1 2  log σ-r1(xi, aw i) − r1(xi, aℓi)− log σ-r0(xi, aw i) − r0(xi, aℓ i) (d)
≤ ϵ, (50)
where (a) and (b) use Eq. (48), (c) uses the above notation that ru = urπr† + (1 − u)r π(u ∈ [0, 1]), and (d) uses Eq. (49).

Then based on Algorithm 2 and Assumption 1, given (xi, a
(1)
i, a
(−1)
i), the label yiis generated with probability distribution pr
∗,ξi(yi|xi, a
(1)
i, a
(−1)
i) defined by Eq. (41). Therefore, given any δ ∈ (0, 1) and ϵ > 0, by Lemma 7 with λ
′ = 1, the following inequality holds for t = 1*, . . . , T* and finitely many π
′ ∈ Nϵ(R) simultaneously with probability at least 1 − δ.

990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 where µi denotes the distribution of the i-th online data sample (xi, a
(−1)
i, a
(1)
i, yi) generated by Algorithm 2. We further upper bound the above inequality as follows.

$$\sum_{i=1}^{t}W_{i}(\pi^{\prime})-\log\left({\frac{T|{\mathcal{N}}_{\epsilon}({\mathcal{R}})|}{\delta}}\right)$$

where (a) uses log v ≤ v − 1 for any v > 0, (b) uses Lemma 12.2 of (Harsha, 2011), (c) uses Eqs. (41) and (47), (d) uses Eq. (29) and Lemma 5, and (e) uses Assumption 2 and Lemma 1. Combining Eqs. (50) and (51), we obtain the following inequality which holds for all t = 1*, . . . , T* and π ∈ Π simultaneously with probability at least 1 − δ.

$$\stackrel{\mathrm{\perp}}{\sum}W_{i}(\pi)$$
t
i=1
$$(51)$$
$$\leq\sum_{i=1}^{t}\log\mathbb{E}_{\mu_{i}}[e^{W_{i}(\pi^{\prime})}]$$
i=1
log Eµi
(Eyi∼pr∗,ξ∗
i
(·|xi,a
(1)
i
,a
(−1)
i)
"vuut
qπ′,0(yi|xi, a
(1)
i, a
(−1)
i)
pr
∗,ξ∗
i
(yi|xi, a
(1)
i, a
(−1)
i)

xi, a
(1)
i, a
(−1)
i
#)
(48)
=X
t
(a)
≤X
t
i=1
Eµi
" X
y∈{−1,1}
qqπ′,0(y|xi, a
(1)
i, a
(−1)
i)pr
∗,ξ∗
i
(y|xi, a
(1)
i, a
(−1)
i) − 1
#
= −
1
2
X
t
i=1
Eµi
" X
y∈{−1,1}

qqπ′,0(y|xi, a
(1)
i, a
(−1)
i) −
qpr
∗,ξ∗
i
(y|xi, a
(1)
i, a
(−1)
i)

2#
(b)
≤ −
1
8
X
t
i=1
Eµi
" X
y∈{−1,1}
qπ′,0(y|xi, a
(1)
i, a
(−1)
i) − pr∗,ξ∗
i
(y|xi, a
(1)
i, a
(−1)
i)
2
#
(c)
= −
1
4
X
t
i=1
Eµi
nσ[r
π
′(xi, aw
i) − r
π
′(xi, aℓ
i)] − σ[r
∗(xi, aw
i) − r
∗(xi, aℓi) + yiξ
∗
i]
o2
(d)
≤ −
1
4
X
t
i=1
nhEµi
-σ[r
πr∗(xi, aw
i
) − r
πr∗(xi, aℓ
i
)] − σ[r
π
′(xi, aw
i
) − r
π
′(xi, aℓ
i
)]2i−
1
2
|ξ
∗
i
|
o
(e)
≤
1
8
X
t
i=1
n|ξ
∗
i|− 
2
(3 + eR)
2
Eµi
hr
πr∗(xi, aw
i)−r
πr∗(xi, aℓ
i)−r
π
′(xi, aw
i)+r
π
′(xi, aℓi)
2io, (51)
$$\sum_{i=1}^{t}W_{i}(\pi^{\prime})\leq\log\left(\frac{T|{\mathcal{N}}_{\epsilon}({\mathcal{R}})|}{\delta}\right)+\sum_{i=1}^{t}\log\mathbb{E}_{\mu_{i}}[e^{W_{i}(\pi^{\prime})}].$$
(50)  $$\begin{array}{l}\mbox{\rm(50)}\end{array}$$
$$|W_{i}(\pi_{r^{\dag}})-W_{i}(\pi)|$$
≤X t i=1 [Wi(π) − Wi(πr † )] + Wi(πr † ) (a) ≤ 1 8 X t i=1 n|ξ ∗ i | − 2 (3 + eR) 2 Eµi h-r πr∗(xi, aw i ) − r πr∗(xi, aℓ i ) − r πr†(xi, aw i ) + r πr†(xi, aℓ i )2io + log T|Nϵ(R)| δ + tϵ (b) ≤ 1 X t n|ξ ∗ i| −  2 2 Eµi h-r πr∗(xi, aw i) − r πr∗(xi, aℓ i) − r π(xi, aw i) + r π(xi, aℓi)2io
$$\stackrel{{(b)}}{{\leq}}\frac{1}{8}\sum_{i=1}^{t}\left\{[\xi_{i}^{*}]-\frac{2}{(3+e^{\hbar})^{2}}\mathbb{E}_{\mu_{i}}\left[\left[r^{\pi_{r^{*}}}(x_{i},a_{i}^{w})-r^{\pi_{r^{*}}}(x_{i},a_{i}^{t})-r^{\pi}(x_{i},a_{i}^{w})+r^{\pi}(x_{i},a_{i}^{t})\right]^{2}\right]\right\}$$ $$+\log\left(\frac{T|\mathcal{N}_{\epsilon}(\mathcal{R})|}{\delta}\right)+2t\epsilon,$$
$$(52)$$

where (a) uses Eq. (51) (with π
′replaced by πr
† ) and Eq. (50), (b) uses the following inequality and (3 + e R)
2 >
6e R + e 2R > 6R + 2R = 8R.

where (a) uses Eq. (29), and (b) uses ∥r
† − r∥∞ ≤ ϵ and Lemma 3.

Finally, we conclude the proof as follows.

1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079 1080 1081 1082 1083 1084 1085 1086 1087 1088 1089 1090 1091 1092 1093 1094 1095 1096 1097 1098 1099

Xt i=1 log σ-r π(xi, aw i ) − r π(xi, aℓi ) + yiξ π i  σ-r ∗(xi, aw i ) − r ∗(xi, aℓ i ) + yiξ ∗ i  (a) ≤X t i=1 hlogσ-r π(xi, aw i) − r π(xi, aℓ i) σ-r ∗(xi, aw i) − r ∗(xi, aℓi) + yiξ ∗ i  + σ(R)|ξ π i| i (b) =Xt -2Wi(π) + σ(R)|ξ π i|
$${\overset{(b)}{=}}\sum_{i=1}^{t}\left[2W_{i}(\pi)+\sigma(R)|\xi_{i}^{\pi}|\right]$$
i=1 (c) ≤2 log T|Nϵ(R)| δ + 4tϵ +X t i=1 n1 4 |ξ ∗ i| + σ(R)|ξ π i| −1 2(3 + eR) 2 Eµi h-r πr∗(xi, aw i) − r πr∗(xi, aℓ i) − r π(xi, aw i) + r π(xi, aℓ i)2io (d) = 2 log T|Nϵ(R)| δ + 4tϵ +X t i=1 n1 4 |ξ ∗ i| + σ(R)|ξ π i| −1 2(3 + eR) 2 Eµi h-r ∗(xi, a (1) i) − r ∗(xi, a (−1) i) − r π(xi, a (1) i) + r π(xi, a (−1) i)2io (e) = 2 log T|Nϵ(R)| δ + 4tϵ +X t i=1 n1 4 |ξ ∗ i| + σ(R)|ξ π i| −1 2(3 + eR) 2 Ex∼ρ,a(1)∼πi(·|x),a(−1)∼πref (·|x) -f 2 π(x, a(1), a(−1))o,
20

-r πr∗(xi, aw i) − r πr∗(xi, aℓ i) − r π(xi, aw i) + r π(xi, aℓ i)2 −-r πr∗(xi, aw i) − r πr∗(xi, aℓ i) − r πr†(xi, aw i) + r πr†(xi, aℓ i)2 =-r πr†(xi, aw i) − r πr†(xi, aℓ i) − r π(xi, aw i) + r π(xi, aℓ i) -2r πr∗(xi, aw i) − 2r πr∗(xi, aℓ i) − r π(xi, aw i) + r π(xi, aℓ i) − r πr†(xi, aw i) + r πr†(xi, aℓ i) (a) =-r †(xi, aw i ) − r †(xi, aℓi ) − r π(xi, aw i ) + r π(xi, aℓi ) -2r πr∗(xi, aw i) − 2r πr∗(xi, aℓ i) − r π(xi, aw i) + r π(xi, aℓ i) − r πr†(xi, aw i) + r πr†(xi, aℓ i) (b) ≤(2ϵ)(4R) = 8Rϵ,