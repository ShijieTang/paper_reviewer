011

014 015 016

018

024

026

034

036

038

# Provably Mitigating Corruption, Overoptimization, and Verbosity Simultaneously in Offline and Online RLHF/DPO Alignment

Anonymous Authors<sup>1</sup>

# Abstract

Reinforcement learning from human feedback (RLHF) and direct preference optimization (DPO) are important techniques to align large language models (LLM) with human preference. However, the quality of RLHF and DPO training is seriously compromised by *Corrupted* preference, reward *Overoptimization*, and bias towards *Verbosity*. To our knowledge, most existing works tackle only one of these important issues, and the few other works require much computation to estimate multiple reward models and lack theoretical guarantee of generalization ability. In this work, we propose RLHF-COV and DPO-COV algorithms that can simultaneously mitigate these three issues, in both offline and online settings. This ability is theoretically demonstrated by obtaining length-regularized generalization error rates for our DPO-COV algorithms trained on corrupted data, which match the best-known rates for simpler cases with clean data and without length regularization. Moreover, our DPO-COV algorithm is simple to implement without reward estimation, and is proved to be equivalent to our RLHF-COV algorithm, which directly implies the equivalence between the vanilla RLHF and DPO algorithms.

# 1. Introduction

Reinforcement learning from human feedback (RLHF) has been widely used in robotics [\(Christiano et al.,](#page-8-0) [2017;](#page-8-0) [Bukharin et al.,](#page-8-1) [2024\)](#page-8-1), autonomous driving [\(Wang et al.,](#page-10-0) [2024;](#page-10-0) [Cao et al.,](#page-8-2) [2024\)](#page-8-2), large language models (LLM) [\(Ouyang et al.,](#page-9-0) [2022;](#page-9-0) [Bai et al.,](#page-8-3) [2022b;](#page-8-3) [Rafailov et al.,](#page-9-1) [2023\)](#page-9-1), image and video generation [\(Wallace et al.,](#page-10-1) [2023;](#page-10-1) [Liang](#page-9-2) [et al.,](#page-9-2) [2024;](#page-9-2) [Liu et al.,](#page-9-3) [2024b\)](#page-9-3), etc. This work will focus on the application of RLHF to LLM alignment which makes

LLM more helpful, honest, and harmless [\(Ouyang et al.,](#page-9-0) [2022;](#page-9-0) [Bai et al.,](#page-8-3) [2022b\)](#page-8-3). LLM alignment has two critical steps. The first step is reward modeling, which estimates the reward model that measures the quality of LLM responses, based on human preference data. The second step is reinforcement learning (RL), which fine-tunes the LLM policy to generate responses with an improved expected value of the learned reward [\(Ouyang et al.,](#page-9-0) [2022\)](#page-9-0). Direct preference optimization (DPO) [\(Rafailov et al.,](#page-9-1) [2023\)](#page-9-1) further simplifies the standard RLHF process by directly fine-tuning the optimal policy without reward estimation.

However, the LLM aligned by RLHF and DPO sometimes yields undesirable responses, due to the corruption, overoptimization, and verbosity issues, as introduced below.

Corruption. The quality of preference data is essential in RLHF and DPO. However, preference labels given by human may be corrupted due to inexperience, inattention, personal bias, unclear context, and even malicious falsification [\(Bukharin et al.,](#page-8-1) [2024\)](#page-8-1). For instance, when fine-tuning LLM for automated content moderation on social media, malicious annotators may mislabel harmful contents like misinformation and hate speech as preferable, which misleads the LLM to generate such harmful contents. Therefore, robustness of RLHF and DPO to such corruption is critical, but is tackled by only a few recent works to our knowledge. For example, [\(Cheng et al.,](#page-8-4) [2024;](#page-8-4) [Mandal et al.,](#page-9-4) [2024;](#page-9-4) [Gao et al.,](#page-9-5) [2024b\)](#page-9-5) use confidence-based data filtering. [\(Ethayarajh et al.,](#page-8-5) [2024\)](#page-8-5) maximizes the utility function defined based on the prospect theory of human decision making [\(Tversky and Kahneman,](#page-10-2) [1992\)](#page-10-2) to filter out noisy data. [\(Coste et al.,](#page-8-6) [2024;](#page-8-6) [Rame et al.,](#page-10-3) [2024\)](#page-10-3) estimate an ensemble of rewards. The recently proposed robust RLHF and robust DPO approaches in [\(Bukharin et al.,](#page-8-1) [2024\)](#page-8-1) use noise modeling to automatically select the outliers and the estimated reward provably converges to the true reward.

Overoptimization. RLHF and DPO may overoptimize the reward model, yielding LLM responses of high estimated reward but low actual quality [\(Gao et al.,](#page-9-6) [2023;](#page-9-6) [Casper et al.,](#page-8-7) [2023\)](#page-8-7). Various methods have been proposed to tackle such overoptimization issue (a.k.a. reward hacking). For example, [\(Gao et al.,](#page-9-6) [2023\)](#page-9-6) uses larger reward model which significantly increases the computational cost of pretraining.

<sup>1</sup>Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>.

Preliminary work. Under review by the International Conference on Machine Learning (ICML). Do not distribute.

056 058 [\(Moskovitz et al.,](#page-9-7) [2024\)](#page-9-7) applies constraints to RLHF. The ΦPo method [\(Azar et al.,](#page-8-8) [2024\)](#page-8-8) optimizes a general preference function. [\(Eisenstein et al.,](#page-8-9) [2024;](#page-8-9) [Coste et al.,](#page-8-6) [2024;](#page-8-6) [Rame et al.,](#page-10-3) [2024;](#page-10-3) [Fisch et al.,](#page-8-10) [2024;](#page-8-10) [Zhai et al.,](#page-10-4) [2023\)](#page-10-4) use an ensemble of estimated rewards.

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

108 109

An emerging and popular strategy with provable generalization ability to solve overoptimization is to adopt a pessimistic (resp. an optimistic) approach for RLHF and DPO with offline (resp. online) data. Specifically, in the offline setting where only precollected offline preference data is available for training, there are many out-of-distribution samples about which we cannot obtain any information. Therefore, [\(Zhu et al.,](#page-10-5) [2023;](#page-10-5) [2024;](#page-10-6) [Liu et al.,](#page-9-8) [2024c;](#page-9-8) [Cen](#page-8-11) [et al.,](#page-8-11) [2024;](#page-8-11) [Ji et al.,](#page-9-9) [2024;](#page-9-9) [Yang et al.,](#page-10-7) [2024;](#page-10-7) [Huang et al.,](#page-9-10) [2024;](#page-9-10) [Xiong et al.,](#page-10-8) [2024;](#page-10-8) [Ye et al.,](#page-10-9) [2024;](#page-10-9) [Fisch et al.,](#page-8-10) [2024\)](#page-8-10) apply pessimistic principle to RLHF or DPO which penalizes LLM from generating such unknown out-of-distribution responses and thus to mitigate overoptimization. Such pessimism principle has also been used in conventional offline RL [\(Xie et al.,](#page-10-10) [2021;](#page-10-10) [Jin et al.,](#page-9-11) [2021;](#page-9-11) [Rashidinejad et al.,](#page-10-11) [2021;](#page-10-11) [Bai et al.,](#page-8-12) [2022a;](#page-8-12) [Cheng et al.,](#page-8-13) [2022\)](#page-8-13). In contrast, in the online setting where online data can be collected from the up-to-date policy during the training process, optimistic approaches have been used to encourage the collection of unexplored samples to enrich data diversity in RLHF and DPO [\(Cen et al.,](#page-8-11) [2024;](#page-8-11) [Xie et al.,](#page-10-12) [2024;](#page-10-12) [Zhang et al.,](#page-10-13) [2024;](#page-10-13) [Ye et al.,](#page-10-9) [2024;](#page-10-9) [Xiong et al.,](#page-10-8) [2024\)](#page-10-8) as well as conventional RL [\(Wei et al.,](#page-10-14) [2017;](#page-10-14) [Zhong and Zhang,](#page-10-15) [2023;](#page-10-15) [Liu et al.,](#page-9-12) [2023a;](#page-9-12)[b\)](#page-9-13).

Verbosity. LLM aligned by vanilla RLHF and DPO is likely to prefer verbose but possibly low-quality responses [\(Singhal et al.,](#page-10-16) [2023;](#page-10-16) [Chen et al.,](#page-8-14) [2024;](#page-8-14) [Liu et al.,](#page-9-14) [2024a;](#page-9-14) [Dong et al.,](#page-8-15) [2024;](#page-8-15) [Fisch et al.,](#page-8-10) [2024\)](#page-8-10). Multiple methods have been used to tackle verbosity. For example, [\(Shen](#page-10-17) [et al.,](#page-10-17) [2023;](#page-10-17) [Chen et al.,](#page-8-14) [2024\)](#page-8-14) disentangle length-related reward component. [\(Guo et al.,](#page-9-15) [2024\)](#page-9-15) instructs the LLM to prefer concise response. [\(Eisenstein et al.,](#page-8-9) [2024;](#page-8-9) [Fisch](#page-8-10) [et al.,](#page-8-10) [2024;](#page-8-10) [Chakraborty et al.,](#page-8-16) [2024\)](#page-8-16) estimate an ensemble of reward models. [\(Singhal et al.,](#page-10-16) [2023;](#page-10-16) [Liu et al.,](#page-9-14) [2024a;](#page-9-14) [Dong et al.,](#page-8-15) [2024;](#page-8-15) [Park et al.,](#page-9-16) [2024\)](#page-9-16) use length penalty and similarly [\(Meng et al.,](#page-9-17) [2024\)](#page-9-17) uses length normalization.

Our Motivation. However, to our knowledge, most existing works primarily tackle only one of these three issues (corruption, overoptimization and verbosity). The only method to our knowledge that has been used to tackle all these issues is to estimate an ensemble of reward models [\(Coste et al.,](#page-8-6) [2024;](#page-8-6) [Fisch et al.,](#page-8-10) [2024;](#page-8-10) [Eisenstein et al.,](#page-8-9) [2024;](#page-8-9) [Rame et al.,](#page-10-3) [2024\)](#page-10-3), which, however, requires much computation and lacks theoretical guarantee of generalization ability. Therefore, we are motivated to ask the following research question.

*Q: Can we design RLHF and DPO algorithms that solve corruption, overoptimization and verbosity simultaneously with simple implementation and theoretical guarantee of generalization ability?*

#### 1.1. Our Contributions

We answer the above question affirmatively, by proposing RLHF-COV and DPO-COV algorithms that simultaneously mitigate *Corruption*, *Overoptimization* and *Verbosity* issues, in both offline and online settings. Specifically, we tackle *Corruption* by noise modeling, tackle *Overoptimization* by pessimistic and optimistic regularizers in the offline and online settings respectively, and tackle *Verbosity* by length regularizer. Our DPO-COV algorithms are almost as simple to implement as the vanilla DPO algorithm without reward model estimation. We prove that our RLHF-COV and DPO-COV are equivalent in the reward-induced policy space in both the offline and online settings. Since our RLHF-COV and DPO-COV algorithms generalize the vanilla RLHF and DPO algorithms respectively, our equivalence result implies that the vanilla RLHF and DPO algorithms are also equivalent. Moreover, we obtain the length-regularized generalization error rates of our DPO-COV algorithms on both offline and online datasets obtained from corrupted preference, and the rates match the existing results in the simple special case with clean dataset and without verbosity regularization. This theoretically demonstrates that our algorithms can simultaneously mitigate the *Corruption*, *Overoptimization* and *Verbosity* issues.

In particular, the effect of noise modeling on the generalization error of learned policy for corrupted data has not been studied to our knowledge, which requires novel proof techniques. The true and estimated noise terms have very different effects on the generalization error, and thus have to be analyzed at different stages. To elaborate, the estimated noise has to be bounded before applying concentration inequality, such that this unbounded estimated noise term can be canceled out by the noise regularizer. In contrast, the true noise has to be bounded after applying the concentration inequality, since the concentration inequality bounds the distance between the true data distribution (with the true noise term) and the estimated data distribution.

# 2. Preliminaries

Reinforcement learning from human feedback (RLHF). A large language model (LLM) provides a random language response a ∈ X to any given language prompt x ∈ X (for example, instruction or question) following the LLM's policy π(·|x). Fine-tuning LLM by reinforcement learning from human feedback (RLHF) consists of two critical steps: training reward model and reinforcement learning

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

(RL) [\(Ouyang et al.,](#page-9-0) [2022\)](#page-9-0). The reward model is denoted by a function r(x, a) ∈ <sup>R</sup> which measures the quality of the response a given the prompt x. To train the reward model, preference data D = {x<sup>i</sup> , a<sup>w</sup> i , a<sup>ℓ</sup> i } N <sup>i</sup>=1 of size N is collected where a pair of responses a w i , a<sup>ℓ</sup> i are generated given each i-th prompt x<sup>i</sup> , and the response a w i is more preferable than a ℓ i (i.e. a w <sup>i</sup> ≻ a ℓ i ). Such a pairwise preference is widely assumed to follow the Bradley-Terry model [\(Bradley and](#page-8-17) [Terry,](#page-8-17) [1952\)](#page-8-17), that is, given prompt x, the generated response a ′ is more desirable than a with the following probability.

$$\mathbb{P}(a' \succ a | x) = \sigma[r^*(x, a') - r^*(x, a)] \quad (1)$$

where σ(x) def = 1/(1 + <sup>e</sup> −x ) and r ∗ is the unknown true reward model. r ∗ can be estimated by maximum likelihood estimation (MLE), that is, to minimize the following negative log-likelihood function over a certain reward model family R.

$$\min_{r \in \mathcal{R}} -\frac{1}{N} \sum_{i=1}^N \log \sigma[r(x_i, a_i^w) - r(x_i, a_i^\ell)]. \quad (2)$$

Finally, given the estimated reward model r ∈ R, the optimal policy is obtained by the following optimization problem over the whole policy space Π def = {π|π(·|x) is a distribution over A for any x}.

$$\begin{aligned} & \max_{\pi \in \Pi} \mathbb{E}_{x \sim \rho, a \sim \pi(\cdot|x)} [r(x, a)] \\ & \quad - \beta \mathbb{E}_{x \sim \rho} \text{KL} [\pi(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)], \end{aligned} \quad (3)$$

where ρ is the prompt distribution, πref is the reference policy obtained by supervised fine-tuning, and KL(p∥q) = P <sup>a</sup>∈A <sup>p</sup>(a) log <sup>p</sup>(a) q(a) denotes the KL divergence between any pair of response distributions p, q and β > 0 is the regularizer coefficient which controls the trade-off between generating responses with high expected reward and bounded distance from the reference policy πref.

Direct preference optimization (DPO). As introduced above, classical RLHF requires two large-scale optimization problems to learn the reward model r and the optimal policy π respectively. DPO [\(Rafailov et al.,](#page-9-1) [2023\)](#page-9-1) is introduced to remove the reward learning step and thus reducing computation. To elaborate, note that the optimization problem [\(3\)](#page-2-0) has the following analytical solution.

$$\pi(a|x) = \frac{\pi_{\text{ref}}(a|x)}{Z(x)} \exp \left[ \frac{r(x, a)}{\beta} \right], \quad (4)$$

where Z(x) := P <sup>a</sup>′∈A πref(a ′ |x) exp[r(x, a′ )/β] is the normalization factor. Conversely, given the optimal policy π, r(x, a) = β log <sup>π</sup>(a|x) πref (a|x) is a solution to Eq. [\(1\)](#page-2-1). Substituting this reward model into the MLE objective [\(3\)](#page-2-0), [\(Rafailov et al.,](#page-9-1) [2023\)](#page-9-1) develops the following simple DPO

objective which only requires policy training.

$$\min_{\pi \in \Pi} -\frac{1}{N} \sum_{i=1}^N \log \sigma \left[ \beta \log \frac{\pi(a_i^w | x_i)}{\pi_{\text{ref}}(a_i^w | x_i)} - \beta \log \frac{\pi(a_i^\ell | x_i)}{\pi_{\text{ref}}(a_i^\ell | x_i)} \right]. \quad (5)$$

However, this DPO objective and the aforementioned vanilla RLHF process are prone to suffer from *corrupted* preference, reward *overoptimization*, and bias towards *verbose* response. We will propose our novel variants of RLHF and DPO to solve the three issues simultaneously, for both offline and online settings, in Sections [3](#page-2-2) and [4](#page-5-0) respectively.

#### 3. Our Offline DPO-COV Algorithm

In this section, we will derive our proposed offline RLHF-COV objective and offline DPO-COV algorithm (Algorithm [1\)](#page-4-0) which simultaneously solve the *Corruption*, *Overoptimization* and *Verbosity* issues, and then obtain the generalization error rates of our offline DPO-COV algorithm.

#### 3.1. Our Offline RLHF-COV Objective

#### Offline Data from *Corrupted* Preference.

Assumption 1. *The offline data* D def = {x<sup>i</sup> , a (1) i , a (−1) i , yi} N <sup>i</sup>=1 = {x<sup>i</sup> , a<sup>w</sup> i , a<sup>ℓ</sup> i , yi} N <sup>i</sup>=1 *is generated from the following model with corrupted preference.*

$$x_i \sim \rho, \quad a_i^{(-1)}, a_i^{(1)} \sim \pi_b(\cdot | x_i), \quad (6)$$

$$\mathbb{P}(a_i^{(1)} \succ a_i^{(-1)}) = \sigma[r^*(x_i, a_i^{(1)}) - r^*(x_i, a_i^{(-1)}) + \xi_i^*], \quad (7)$$

i i i *where* π<sup>b</sup> *denotes the behavior policy and* ξ ∗ <sup>i</sup> ∈ <sup>R</sup> *denotes the true preference noise for the* i*-th sample. If* a (1) <sup>i</sup> ≻ a (−1) i *, assign the label* y<sup>i</sup> = 1 *and denote* a w <sup>i</sup> = a (1) i *as the more preferable response and* a ℓ <sup>i</sup> = a (−1) i *as the less preferable response; Otherwise, let* y<sup>i</sup> = −1*,* a w <sup>i</sup> = a (−1) i *,* a ℓ <sup>i</sup> = a (1) i *.*

The above assumption is very similar to that of offline vanilla RLHF and DPO, except that we add noise ξ ∗ i to the Bradley-Terry model [\(1\)](#page-2-1) for each possibly corrupted sample i [\(Bukharin et al.,](#page-8-1) [2024\)](#page-8-1).

Based on Assumption [1,](#page-2-3) <sup>P</sup>(y<sup>i</sup> |a (1) i , a (−1) i ) = σ[r ∗ (x<sup>i</sup> , a<sup>w</sup> i ) − r ∗ (x<sup>i</sup> , a<sup>ℓ</sup> i ) + yiξ ∗ i ], y<sup>i</sup> ∈ {−1, 1} [1](#page-2-4) . Hence, we define a penalized negative log-likelihood function of the labels {yi} N <sup>i</sup>=1 as follows.

$$\mathcal{L}_{N,\lambda}(r, \xi) \stackrel{\text{def}}{=} -\frac{1}{N} \sum_{i=1}^N \log \sigma[r(x_i, a_i^w) - r(x_i, a_i^\ell) + y_i \xi_i]$$

<sup>1</sup>We corrected the mistake in [\(Bukharin et al.,](#page-8-1) [2024\)](#page-8-1) which uses <sup>P</sup>(yi|a (1) i , a (−1) i ) = σ[r ∗ (xi, a<sup>w</sup> <sup>i</sup> ) − r ∗ (xi, a<sup>ℓ</sup> <sup>i</sup> ) + ξ ∗ i ], y<sup>i</sup> ∈ {−1, 1} that yields P <sup>y</sup>i∈{−1,1} <sup>P</sup>(yi|<sup>a</sup> (1) i , a (−1) i ) ̸= 1.

*174*

*181*

*183 184*

*190 191*

*200*

*204*

*206*

$$+ \frac{\lambda}{N} \|\xi\|_1, \quad (8)$$

which, compared with the standard non-corrupted negative log-likelihood function [\(2\)](#page-2-5), adds the estimated preference noise ξ = [ξ1, . . . , ξ<sup>N</sup> ] ∈ <sup>R</sup> <sup>N</sup> and the noise regularizer ∥ξ∥<sup>1</sup> = P<sup>N</sup> <sup>i</sup>=1 |ξ<sup>i</sup> | with coefficient λ > 0 to encourage the sparsity of the noise.

Reward Estimation via Pessimistic MLE to Solve *Overoptimization*. After collecting offline data, the next step is to learn the reward model r. One may consider corrupted MLE objective minr∈R,ξ∈R<sup>N</sup> LN,λ(r, ξ) [\(Bukharin et al.,](#page-8-1) [2024\)](#page-8-1) which generalizes the non-corrupted MLE objective [\(2\)](#page-2-5). However, this corrupted MLE objective tend to overfit limited offline data [\(Gao et al.,](#page-9-6) [2023;](#page-9-6) [Zhu et al.,](#page-10-6) [2024;](#page-10-6) [Liu](#page-9-8) [et al.,](#page-9-8) [2024c;](#page-9-8) [Cen et al.,](#page-8-11) [2024;](#page-8-11) [Xiong et al.,](#page-10-8) [2024\)](#page-10-8), producing an inaccurately estimated reward that leads to overoptimization. Therefore, we consider the following pessimistic MLE inspired by [\(Liu et al.,](#page-9-8) [2024c;](#page-9-8) [Cen et al.,](#page-8-11) [2024;](#page-8-11) [Ji et al.,](#page-9-9) [2024;](#page-9-9) [Yang et al.,](#page-10-7) [2024\)](#page-10-7).

$$\min_{r \in \mathcal{R}, \xi \in \mathbb{R}^N} \left\{ \mathcal{L}_{N,\lambda}(r, \xi) + \eta \max_{\pi \in \Pi} V_{\beta}(\pi, r) \right\}, \quad (9)$$

where the pessimistic hyperparameter η ≥ 0 and

$$V_\beta(\pi, r) \stackrel{\text{def}}{=} \mathbb{E}_{x \sim \rho, a \sim \pi(\cdot | x), a' \sim \pi_{\text{base}}(\cdot | x)} [r(x, a) - r(x, a')] \\ - \beta \mathbb{E}_{x \sim \rho} \text{KL}[\pi(\cdot | x) \| \pi_{\text{Ref}}(\cdot | x)] \quad (10)$$

denotes the relative value of the policy π to a certain baseline policy πbase given the reward r. The regularizer maxπ∈<sup>Π</sup> Vβ(π, r) in Eq. [\(9\)](#page-3-0) can be seen as the relative value of the optimal policy, and will help reduce the reward value r(x, a) of any sample x, a with small πbase(a|x), so that the optimal policy π(a|x) given by Eq. [\(4\)](#page-2-6) will also be reduced. In other words, such samples x, a are considered pessimistic and are thus discouraged from being generated by the learned policy π. Hence, the regularizer maxπ∈<sup>Π</sup> Vβ(π, r) is called the pessimistic regularizer. Furthermore, if we select πbase to represent the offline data distribution (see the end of Section [3.2](#page-3-1) for the choice of πbase), then these samples x, a with small πbase(a|x) can be seen as out-of-distribution, so that such pessimism on the out-of-distribution samples mitigates the overoptimization issue which often results from overestimation of the reward on low-quality out-of-distribution samples [\(Liu et al.,](#page-9-8) [2024c\)](#page-9-8).

Policy Training with Penalized *Verbosity*. The vanilla RLHF usually yields reward model r(x, a) that has bias towards long and detailed responses. To suppress verbose responses in the policy optimization step maxπ∈<sup>Π</sup> Vβ(π, r), we can replace the reward model r(x, a) with the proxy reward model rω(x, a) = r(x, a) − ω|a| where |a| is the length (i.e., number of tokens) of the response a and the hyperparameter ω ≥ 0 controls the length penalty strength

[\(Singhal et al.,](#page-10-16) [2023;](#page-10-16) [Liu et al.,](#page-9-14) [2024a;](#page-9-14) [Dong et al.,](#page-8-15) [2024;](#page-8-15) [Park et al.,](#page-9-16) [2024\)](#page-9-16). In this way, the policy training objective Vβ(π, r) (defined by Eq. [\(10\)](#page-3-2)) is generalized to the following length-regularized relative value function.

$$\begin{aligned} & V_{\beta,\omega}(\pi, r) \\ & \stackrel{\text{def}}{=} \mathbb{E}_{x \sim \rho, a \sim \pi(\cdot|x), a' \sim \pi_{\text{base}}(\cdot|x)} [r(x, a) - \omega|a] - r(x, a') \\ & \quad + \omega |a'| - \beta \mathbb{E}_{x \sim \rho} \text{KL}[\pi(\cdot|x) || \pi_{\text{ref}}(\cdot|x)]. \end{aligned} \quad (11)$$

Replacing Vβ(π, r) with Vβ,ω(π, r) in the pessimistic MLE objective [\(9\)](#page-3-0), we propose offline RLHF-COV objective below.

(Offline RLHF-COV):

$$\begin{aligned} \min_{r \in \mathcal{R}, \xi \in \mathbb{R}^N} \max_{\pi \in \Pi} \left\{ \mathcal{L}_{N, \lambda}(r, \xi) + \eta V_{\beta, \omega}(\pi, r) \right\} &\stackrel{(8),(11)}{=} \\ &+ \eta \mathbb{E}_{x \sim \rho, \alpha \sim \pi(\cdot | x), \alpha' \sim \pi_{\text{base}}(\cdot | x)} \\ &\quad [r(x, a) - \omega | a] - r(x, a') + \omega | a'] \\ &+ \frac{1}{N} \sum_{i=1}^N \{ |\lambda| \xi_i| - \log \sigma[r(x_i, a_i^w) - r(x_i, a_i^\ell) + y_i \xi_i] \} \\ &- \beta \eta \mathbb{E}_{x \sim \rho} \text{KL}[\pi(\cdot | x) || \pi_{\text{ref}}(\cdot | x)]. \end{aligned} \quad (12)$$

Remark: Our offline RLHF-COV objective above simultaneously tackles the *Corruption*, *Overoptimization* and *Verbosity* issues, via noise modeling, pessimism and length penalty with controllable hyperparameters λ, η, ω respectively. Specifically, the length penalty is only added to Vβ,ω not LN,λ, because in the pessimistic MLE we still want to obtain a reward r possibly with length bias, and then verbosity is only suppressed in the policy optimization part maxπ∈<sup>Π</sup> Vβ,ω(π, r). When λ ≥ 1 and η = ω = 0, our offline RLHF-COV objective above reduces to the reward estimation [\(2\)](#page-2-5) and policy optimization [\(3\)](#page-2-0) in the vanilla RLHF.

#### 3.2. Our Offline DPO-COV Algorithm

The offline RLHF-COV objective [\(12\)](#page-3-5) involves minimax optimization over three high-dimensional variables r, ξ, π. As the first step to simplify this objective, we obtain the following proposition.

Proposition 1. (π, r, ξ) *is the solution to the offline RLHF-COV objective* [\(12\)](#page-3-5) *if and only if*

π = π<sup>r</sup> def = arg maxπ′∈ΠVβ,ω(π ′ , r)*,* ξ = ξ<sup>r</sup> def = arg minξ∈R<sup>N</sup> LN,λ(r, ξ) *and* r *is the solution to the following optimization problem.*

$$\min_{r \in \mathcal{R}} [\mathcal{L}_{N,\lambda}(r, \xi_r) + \eta V_{\beta,\omega}(\pi_r, r)]. \quad (13)$$

*In addition,* π<sup>r</sup> *and* ξr,i *(the* i*-th entry of* ξr*) have the following analytical solutions.*

$$\pi_r(a|x) = \frac{\pi_{\text{ref}}(a|x)}{Z_r(x)} \exp \left[ \frac{r(x, a) - \omega|a|}{\beta} \right], \quad (14)$$

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

$$\begin{aligned} \xi_{r,i} &= y_i I\{\lambda < 1\} \\ &\left[ \log\left(\frac{1}{\lambda} - 1\right) - r(x_i, a_i^w) + r(x_i, a_i^\ell) \right]_+, \quad (15) \end{aligned}$$

*where* Zr(x) def = P <sup>a</sup>′∈A πref(a ′ |x) exp r(x,a′ )−ω|a β *is the normalization factor,* I{λ < 1} *equals 1 if* λ < 1 *and 0 otherwise, and* [u]<sup>+</sup> = max(u, 0) *for any* u ∈ <sup>R</sup>*.*

The above proposition simplifies the offline RLHF-COV objective [\(12\)](#page-3-5) into the reward estimation problem [\(13\)](#page-3-6). Next, we will transform it into our DPO-COV objective of the policy π. In Eq. [\(14\)](#page-3-7), given π = πr, a solution to the reward model r is

$$r^\pi(x, a) \stackrel{\text{def}}{=} \omega|a| + \beta \log \left[ \frac{\pi(a|x)}{\pi_{\text{ref}}(a|x)} \right]. \quad (16)$$

With the above reward r π , the corresponding noise can also be parameterized by π as ξ <sup>π</sup> def <sup>=</sup> <sup>ξ</sup>r<sup>π</sup> , whose <sup>i</sup>-th entry has the following analytical solution based on Eqs. [\(15\)](#page-4-1) and [\(16\)](#page-4-2).

$$\xi_i^{\pi} \stackrel{\text{def}}{=} \xi_{r,\pi}, i = y_i I \{ \lambda < 1 \} \left[ \log\left(\frac{1}{\lambda} - 1\right) - \omega(|a_i^w| - |a_i^\ell|) - \beta \log\left(\frac{\pi(a_i^w |x_i|) \pi_{\text{ref}}(a_i^\ell |x_i|)}{\pi(a_i^\ell |x_i|) \pi_{\text{ref}}(a_i^w |x_i|)}\right) \right]_+, \quad (17)$$

Substituting the above r π and ξ π i into Eq. [\(13\)](#page-3-6), we propose our DPO-COV objective as follows.

### (Offline DPO-COV):

$$\begin{aligned} \min_{\pi \in \Pi_R} \left\{ \mathcal{L}_{\mathcal{N}, \lambda}(r^\pi, \xi^\pi) + \eta V_{\beta, \omega}(\pi_{r^\pi}, r^\pi) \right. \\ \left. - \beta \eta \mathbb{E}_{x \sim \rho, a \sim \pi_{\text{base}}(\cdot | x)} [\log \pi(a | x)] \right. \\ \left. + \frac{1}{N} \sum_{i=1}^N \left[ \lambda |\xi_i^\pi| - \log \sigma \left( \omega(|a_i^w| - |a_i^\ell|) \right) \right. \right. \\ \left. + \beta \log \frac{\pi(a_i^w | x_i) \pi_{\text{ref}}(a_i^\ell | x_i)}{\pi(a_i^\ell | x_i) \pi_{\text{ref}}(a_i^w | x_i)} \right) + y_i \xi_i^\pi ] + C_{\text{off}} \}, \quad (18) \end{aligned}$$

where Coff def <sup>=</sup> βηEx∼ρ,a∼πbase(·|x) log πref(a|x) is a constant independent of π, and we use the reward-induced policy space Π<sup>R</sup> def <sup>=</sup> {π<sup>r</sup> : <sup>r</sup> ∈ R} since the optimal policy is π<sup>r</sup> for some reward r based on Proposition [1.](#page-3-8) Note that such Π<sup>R</sup> is sufficiently general to admit any parameterized policy π<sup>θ</sup> since by defining R = {r π<sup>θ</sup> : θ ∈ Θ}, we have Π<sup>R</sup> = {π<sup>θ</sup> : θ ∈ Θ} based on Lemma [3.](#page-12-0)

Remark: Our proposed offline DPO-COV objective [\(18\)](#page-4-3) simultaneously tackles *Corruption*, *Overoptimization* and *Verbosity* issues. *Corruption* is modeled by the noise term ξ <sup>π</sup> = [ξ π 1 , . . . , ξ<sup>π</sup> <sup>N</sup> ] which becomes sparser as the hyperparameter λ ≥ 0 increases, and ξ <sup>π</sup> = 0 when λ ≥ 1. *Overoptimization* is tackled by the pessimistic regularizer −βηEx∼ρ,a∼πbase(·|x) log π(a|x) which helps to increase

Algorithm 1 Offline DPO-COV Algorithm

1: Inputs: Hyperparameters β, η, ω, λ ≥ 0, offline data {x<sup>i</sup> , a<sup>w</sup> i , a<sup>ℓ</sup> i } N <sup>i</sup>=1, reference policy πref. 2: Output: Obtain policy <sup>π</sup>b via the following practical offline DPO-COV objective.

$$\begin{aligned} \min_{\pi \in \Pi_{\pi}} \psi_N(\pi) &\stackrel{\text{def}}{=} \frac{1}{N} \sum_{i=1}^N \left\{ \lambda |\xi_i^{\pi}| - \beta \eta \log \pi(a_i^w |x) \right\} \\ &- \log \sigma \left[ \omega(|a_i^w| - |a_i^\ell|) \right. \\ &\left. + \beta \log \left( \frac{\pi(a_i^w |x_i) \pi_{\text{ref}}(a_i^\ell |x_i)}{\pi(a_i^\ell |x_i) \pi_{\text{ref}}(a_i^w |x_i)} \right) + y_i \xi_i^{\pi} \right] \right\}, \quad (19) \end{aligned}$$

where ξ π i is defined by Eq. [\(17\)](#page-4-4).

π(a|x) for in-distribution samples (x, a) well covered by πbase. *Verbosity* is penalized by the length regularizers ω|a w i |, ω|a ℓ i |. When λ ≥ 1 and η = ω = 0, our above offline DPO-COV objective [\(18\)](#page-4-3) reduces to the vanilla DPO objective [\(5\)](#page-2-7).

We formally establish the equivalence between our offline RLHF-COV objective [\(12\)](#page-3-5) and offline DPO-COV objective [\(18\)](#page-4-3) in the following Proposition [2,](#page-4-5) which implies the equivalence between the vanilla RLHF and DPO algorithms as a special case when λ ≥ 1 and η = ω = 0.

Proposition 2. *A policy* π ∈ Π *is optimal for the offline DPO-COV objective* [\(18\)](#page-4-3) *if and only if there exist* r ∈ R, ξ ∈ R <sup>N</sup> *such that* (π, r, ξ) *is optimal for the offline RLHF-COV objective* [\(12\)](#page-3-5)*. In this case,* ξ = ξ π *, and for any* x ∈ X *, there exists* Uπ(x) ∈ <sup>R</sup> *such that* r(x, ·) = r π (x, ·) + Uπ(x)*.*

As suggested by [\(Liu et al.,](#page-9-8) [2024c;](#page-9-8) [Yang et al.,](#page-10-7) [2024\)](#page-10-7) and discussed in Section [3.3,](#page-4-6) in the DPO-COV objective [\(18\)](#page-4-3), we can take πbase(·|x) as the distribution of the preferable responses a w i given x<sup>i</sup> = x under Assumption [1,](#page-2-3) and then adopt the simple stochastic approximation <sup>E</sup>x∼ρ,a∼πbase(·|x) log π(a|x) ≈ 1 N P<sup>N</sup> <sup>i</sup>=1 log π(a w i |xi). This yields our fully stochastic offline DPO-COV algorithm as Algorithm [1,](#page-4-0) which only requires to solve the policy optimization problem that is almost as simple as the vanilla DPO objective [\(5\)](#page-2-7).

#### 3.3. Generalization Analysis of Offline DPO-COV

While the policy π is trained from the offline data D, the ultimate goal is to make π generalize well to all possible prompts x ∼ ρ. Specifically, we define the following lengthregularized value function which characterizes the generalization ability of the policy π as a trade-off among the true reward value r ∗ (response quality), the length of the

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

328

generated response a, and the policy's distance to πref.

$$J_{\beta, \omega}(\pi) := \mathbb{E}_{x \sim \rho, a \sim \pi(\cdot | x)} \left[ r^*(x, a) - \omega | a \right] - \beta \text{KL}[\pi(\cdot | x) \| \pi_{\text{ref}}(\cdot | x)]. \quad (20)$$

To analyze the generalization error of the policy <sup>π</sup>b obtained from Algorithm [1,](#page-4-0) we make the standard assumptions below.

Assumption 2 (Realizable and Bounded Reward [\(Zhu et al.,](#page-10-5) [2023;](#page-10-5) [Zhan et al.,](#page-10-18) [2024;](#page-10-18) [Cen et al.,](#page-8-11) [2024;](#page-8-11) [Ji et al.,](#page-9-9) [2024;](#page-9-9) [Liu et al.,](#page-9-8) [2024c\)](#page-9-8)). *The reward model set* R *includes the true reward model* r ∗ *, that is,* r <sup>∗</sup> ∈ R*. Also, there exists a constant* R ∈ (0, +∞) *such that for any* x ∈ X *,* a ∈ A *and* r ∈ R*, we have* r(x, a) ∈ [0, R]*.*

Assumption 3 (Offline Data Coverage [\(Zhan et al.,](#page-10-18) [2024;](#page-10-18) [Ji et al.,](#page-9-9) [2024;](#page-9-9) [Liu et al.,](#page-9-8) [2024c\)](#page-9-8)). *There exists a constant* G<sup>D</sup> ∈ (0, +∞) *called offline coverage coefficient, such that the choice of the baseline policy* πbase *satisfies the following coverage property for all* r ∈ R*.*

$$\mathbb{E}_{x \sim \rho, a \sim \pi_{r,*}(\cdot | x), a' \sim \pi_{\text{base}}(\cdot | x)} [r^*(x, a) - r^*(x, a') - r(x, a) + r(x, a')] \leq G_{\mathcal{D}} E_r, \quad (21)$$

*where* E<sup>r</sup> def = -E<sup>D</sup> r ∗ (x1, a<sup>w</sup> 1 )−r ∗ (x1, a<sup>ℓ</sup> 1 )−r(x1, a<sup>w</sup> 1 ) + r(x1, a<sup>ℓ</sup> 1 ) <sup>2</sup><sup>1</sup>/<sup>2</sup> *with the offline data sample* x1, a<sup>w</sup> 1 , a<sup>ℓ</sup> <sup>1</sup> *generated via Assumption [1.](#page-2-3)*

The offline coverage coefficient G<sup>D</sup> above describes how well the offline data D covers the responses from πbase and the true optimal policy π<sup>r</sup> <sup>∗</sup> ∈ arg maxπ∈ΠJβ,ω(π). Algorithm [1](#page-4-0) takes πbase(·|x) as the distribution of the preferable responses a w i given x<sup>i</sup> = x, which is well covered by D.

Theorem 1. *Suppose Assumptions [1](#page-2-3)[-3](#page-5-1) hold and* R *is a convex set. For any* δ ∈ (0, 1)*, select hyperparameters* λ ∈ [σ(R), 1]*,* η = 2 √ <sup>∗</sup>∥1+5 log[|N1/N (R)|/δ] √ N(3+e<sup>R</sup>) *. Then, the policy* <sup>π</sup>e *from the offline DPO-COV objective* [\(18\)](#page-4-3) *has the following generalization error rate with probability at least* 1 − δ*.*

$$\begin{aligned} & \max_{\pi \in \Pi} J_{\beta, \omega}(\pi) - J_{\beta, \omega}(\tilde{\pi}) \\ & \leq \frac{(G_{\mathcal{D}}^2 + 1)(3 + e^{R_1})}{\sqrt{N}} \sqrt{\|\xi^*\|_1 + 5 \log[|\mathcal{N}_{1/N}(\mathcal{R})|/\delta]}, \end{aligned} \quad (22)$$

*where* N1/N (R) *is a* (1/N)*-cover of* R*, that is, for any* r ∈ R*, there exists* r † ∈ N1/N (R) *satisfying* ∥r † − r∥<sup>∞</sup> ≤ 1/N*.*

Comparison with Existing Works. Note that |N1/N (R)| ≤ O[(RN) |X ||A|] since R ⊂ [0, R] |X ||A| by Assumption [2.](#page-5-2) Hence, as long as ∥ξ <sup>∗</sup>∥<sup>1</sup> ≤ O[log(N)] (much weaker than Assumption 4.2 of [\(Bukharin et al.,](#page-8-1) [2024\)](#page-8-1) that there exist constants c0, c<sup>∞</sup> > 0 such that ξ <sup>∗</sup> has at most c<sup>0</sup> nonzero entries and they range in [−c∞, c∞]), the generalization error rate [\(22\)](#page-5-3) has the order of O[log(N)/ √ N]. This rate matches the existing error rates of the offline pessimistic DPO-type algorithms [\(Liu](#page-9-8) [et al.,](#page-9-8) [2024c;](#page-9-8) [Cen et al.,](#page-8-11) [2024;](#page-8-11) [Ji et al.,](#page-9-9) [2024\)](#page-9-9) up to logarithm, in the simple case with clean data (λ ≥ 1) and without length regularization (ω = 0). This implies that our offline DPO-COV algorithm provably mitigates *Overoptimization*. In addition, Theorem [1](#page-5-4) also for the first time extends to the corrupted data and the length-regularized generalization error, which shows that our Algorithm [1](#page-4-0) also mitigates *Corruption* and *Verbosity*. In particular, to mitigate *Corruption*, we use novel techniques below to bound the noise terms in the generalization error of the learned policy, whereas [\(Bukharin et al.,](#page-8-1) [2024\)](#page-8-1) only analyzes the estimation error of the reward and noise, but not that of the policy.

Technical Novelty. The proof logic of Theorem [1](#page-5-4) is inspired from that of [\(Liu et al.,](#page-9-8) [2024c\)](#page-9-8), but our proof requires novel techniques to bound the effects of the true noise ξ ∗ and estimated noise ξ π . To elaborate, the ξ π is analyzed by our proposed Lemma [4,](#page-13-0) such that the error bound σ(R)|ξr,i| can later be canceled out by the regularizer −λ|ξr,i| when bounding the MLE error in Lemma [8.](#page-14-0) Next, we bound the distance between the true data distribution under (r ∗ , ξ<sup>∗</sup> ) and the noiseless data distribution under the estimated r and ξ = 0 (see (c) of Eq. [\(43\)](#page-16-0)) by concentration inequality. Then we bound ξ <sup>∗</sup> by our proposed Lemma [5](#page-13-1) which has a different form from Lemma [4](#page-13-0) used for bounding ξ π .

# 4. Our Online DPO-COV Algorithm

Compared with offline RLHF and DPO-type algorithms which use precollected offline data, the online algorithms improve the data coverage and the quality of the trained policy [\(Cen et al.,](#page-8-11) [2024;](#page-8-11) [Dong et al.,](#page-8-15) [2024;](#page-8-15) [Xu et al.,](#page-10-19) [2024;](#page-10-19) [Ye et al.,](#page-10-9) [2024;](#page-10-9) [Guo et al.,](#page-9-15) [2024\)](#page-9-15) at the computation cost of collecting the online preference data in the training process [\(Zhan et al.,](#page-10-18) [2024;](#page-10-18) [Ji et al.,](#page-9-9) [2024;](#page-9-9) [Huang et al.,](#page-9-10) [2024;](#page-9-10) [Man](#page-9-4)[dal et al.,](#page-9-4) [2024\)](#page-9-4). Therefore, online and offline algorithms have different advantages, so both are important. In this section, we will derive our online RLHF-COV objective and online DPO-COV algorithm, and provide the generalization analysis result of our DPO-COV algorithm.

At each t-th iteration of our online algorithm, we use the current policy π<sup>t</sup> to obtain the t-th sample by x<sup>t</sup> ∼ ρ, a (−1) <sup>t</sup> ∼ πref(·|xt), a (1) <sup>t</sup> ∼ πt(·|xt), and the label y<sup>t</sup> is obtained from a stochastic oracle (such as GPT-4) assumed to follow the corrupted preference model [\(7\)](#page-2-8). We propose the following online RLHF-COV objective to train the next policy πt+1 on the online data {x<sup>i</sup> , a (−1) i , a (1) i , yi} t <sup>i</sup>=1.

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

#### (Online RLHF-COV):

$$\begin{aligned} \pi_{t+1} \in \arg \min_{\pi \in \Pi} \min_{r \in \mathcal{R}, \xi^{(t)} \in \mathbb{R}^t} \left\{ \mathcal{L}_{t,\lambda}(r, \xi^{(t)}) - \eta V_{\beta, \omega}(\pi, r) \right\} \\ \stackrel{(8), (11)}{=} \beta \eta \mathbb{E}_{x \sim \rho} \text{KL}[\pi(\cdot | x) \parallel \pi_{\text{ref}}(\cdot | x)] \\ + \frac{1}{t} \sum_{i=1}^t \{ \lambda |\xi_i| - \log \sigma[r(x_i, a_i^w) - r(x_i, a_i^\ell) + y_i \xi_i] \} \\ - \eta \mathbb{E}_{x \sim \rho, a \sim \pi(\cdot | x), a' \sim \pi_{\text{base}}(\cdot | x)} \\ [r(x, a) + \omega |a| - r(x, a') - \omega |a'|], \end{aligned} \tag{23}$$

$$\begin{aligned} \min_{\pi \in \Pi_{\mathcal{R}}} \phi_t(\pi) &= \frac{1}{t} \sum_{i=1}^t \left\{ \lambda |\xi_i^\pi| + \beta \eta \log \pi (a_i^{(-1)} |x_i|) \right. \\ &\quad \left. - \log \sigma \left[ \omega(|a_i^w| - |a_i^\ell|) \right. \right. \\ &\quad \left. + \beta \log \left( \frac{\pi(a_i^w |x_i|) \pi_{\text{ref}}(a_i^\ell |x_i|)}{\pi(a_i^\ell |x_i|) \pi_{\text{ref}}(a_i^w |x_i|)} \right) + y_t \xi_i^\pi \right] \right\}, \quad (25) \end{aligned}$$

where ξ (t) = [ξ1, . . . , ξt] denotes the noise. The above online RLHF-COV objective is similar to the offline RLHF-COV objective [\(12\)](#page-3-5) with the major difference that they tackle overoptimization in seemingly opposite ways. The offline RLHF-COV objective [\(12\)](#page-3-5) (i.e., minr∈R,ξ∈R<sup>N</sup> [LN,λ(r, ξ) + η maxπ∈<sup>Π</sup> Vβ,ω(π, r)]) uses the pessimistic term +η maxπ∈<sup>Π</sup> Vβ,ω(π, r) to discourage LLM from generating out-of-distribution samples. In contrast, inspired by [\(Cen et al.,](#page-8-11) [2024\)](#page-8-11), our above online RLHF-COV objective (i.e., minr∈R,ξ∈R<sup>N</sup> [Lt,λ(r, ξ) − η maxπ∈<sup>Π</sup> Vβ,ω(π, r)]) uses the sign-flipped optimistic term −η maxπ∈<sup>Π</sup> Vβ,ω(π, r) to encourage LLM to collect out-ofdistribution samples to enrich the diversity of the online data to improve policy optimization.

Similar to the offline DPO-COV objective [\(18\)](#page-4-3), we obtain our online DPO-COV objective as follows.

#### (Online DPO-COV):

$$\begin{aligned} \pi_{t+1} \in \arg \min_{\pi \in \Pi_R} \left\{ \mathcal{L}_{t,\lambda}(r^\pi, \xi^{\pi,(t)}) - \eta V_{\beta,\omega}(r^\pi, r^\pi) \right. \\ = \beta \eta \mathbb{E}_{x \sim \rho, a \sim \pi_{\text{base}}(\cdot | x)} [\log \pi(a | x)] + \frac{1}{t} \sum_{i=1}^t [\lambda | \xi_i^\pi | \\ - \log \sigma \left( \omega(|a_i^w| - |a_i^\ell|) + \beta \log \frac{\pi(a_i^w | x_i) \pi_{\text{ref}}(a_i^\ell | x_i)}{\pi(a_i^\ell | x_i) \pi_{\text{ref}}(a_i^w | x_i)} \right) \\ \left. + y_i \xi_i^\pi \right] + C_{\text{on}} \}, \end{aligned} \quad (24)$$

where ξ π,(t) def = [<sup>ξ</sup> π 1 , . . . , ξ<sup>π</sup> t ] is given by Eq. [\(17\)](#page-4-4) and Con = −βηEx∼ρ,a∼πbase(·|x) [log πref(a|x)] is a constant independent of π. Similar to Proposition [2,](#page-4-5) we can show that the online RLHF-COV objective [\(23\)](#page-6-0) and the online DPO-COV objective [\(24\)](#page-6-1) are equivalent as follows.

Proposition 3. *A policy* π ∈ Π *is optimal for the online DPO-COV objective* [\(24\)](#page-6-1) *if and only if there exist* r ∈ R, ξ ∈ R <sup>N</sup> *such that* (π, r, ξ) *is optimal for the offline RLHF-COV objective* [\(23\)](#page-6-0)*. In this case,* ξ = ξ π *and for any* x ∈ X *, there exists* Uπ(x) ∈ <sup>R</sup> *such that* r(x, ·) = r π (x, ·) + Uπ(x)*.*

Inspired by [\(Xie et al.,](#page-10-12) [2024\)](#page-10-12), we select πbase = πref and use its generated samples {a (−1) i } t <sup>i</sup>=1 to approximate the

Algorithm 2 Online DPO-COV Algorithm

- 1: Inputs: β, η, ω, λ > 0, reference policy πref, inital policy π0. 2: for Iterations t = 1, . . . , T do 3: Generate the t-th sample by x<sup>t</sup> ∼ ρ, a (−1) <sup>t</sup> ∼ πref(·|xt), a
- (1) <sup>t</sup> ∼ πt(·|xt), and label y<sup>t</sup> from a certain stochastic oracle assumed to follow the corrupted preference model [\(7\)](#page-2-8). 4: Obtain πt+1 by solving the following stochastic online DPO-COV objective [\(25\)](#page-6-2). min π∈Π<sup>R</sup> <sup>ϕ</sup>t(π) = <sup>1</sup> t Xt i=1 n λ|ξ π i | + βη log π(a (−1) i |xi) − log σ h ω(|a w i | − |a ℓ i |) <sup>+</sup> <sup>β</sup> log <sup>π</sup>(<sup>a</sup> w i |xi)πref(a ℓ i |xi) π(a ℓ i |xi)πref(a w i |xi) + yiξ π i io, (25) 5: end for 6: Output: <sup>π</sup>T<sup>b</sup> where <sup>T</sup>b <sup>∼</sup> Uniform({2, <sup>3</sup>, . . . , T, T <sup>+</sup> 1}).

expectation in the above online DPO-COV objective. This yields our fully stochastic online DPO-COV algorithm (Algorithm [2\)](#page-6-3), which is also almost as simple to implement as the online vanilla DPO algorithm [\(Guo et al.,](#page-9-15) [2024\)](#page-9-15) (also Algorithm [2](#page-6-3) with η = ω = 0 and λ = 1).

To analyze the generalization error of Algorithm [2,](#page-6-3) define the following coverability coefficient [\(Xie et al.,](#page-10-12) [2024\)](#page-10-12), which ensures that there exists at least one policy ν ∈ Π<sup>R</sup> with good coverage over the responses generated by any policy π ∈ ΠR.

$$G_{\text{on}} \stackrel{\text{def}}{=} \inf_{\nu \in \Pi_{\mathcal{R}}} \sup_{x \in \mathcal{X}, a \in \mathcal{A}, \pi \in \Pi_{\mathcal{R}}} \frac{\pi(a|x)}{\nu(a|x)}. \quad (26)$$

Theorem 2. *Under Assumption [2](#page-5-2) and for any* δ ∈ (0, 1)*, select hyperparameters* <sup>λ</sup> <sup>∈</sup> [σ(R), 1]*,* <sup>η</sup> <sup>=</sup> √ log[4TN1/T (R)/δ]+∥ξ <sup>∗</sup>∥<sup>1</sup> (3+e<sup>R</sup>) √ T Gon *where* ξ <sup>∗</sup> = [ξ ∗ 1 , . . . , ξ<sup>∗</sup> T ]*. Then the output policy* <sup>π</sup>Tb *of Algorithm [<sup>2</sup>](#page-6-3) satisfies the following generalization error rate with probability at least* 1 − δ*.*

$$\max_{\pi \in \Pi} J_{\beta, \omega}(\pi) - \mathbb{E}[J_{\beta, \omega}(\pi_{\widehat{T}})] \leq 37(3 + e^R)(\log T)$$

$$\sqrt{\frac{G_{\text{on}}}{T}} \left[ \log \left( \frac{4T|\mathcal{N}_{1/T}(\mathcal{R})|}{\delta} \right) + \|\xi^*\|_1 \right]. \quad (27)$$

Remark: Theorem [2](#page-6-4) above demonstrates that our online DPO-COV algorithm can simultaneously mitigate the *Corruption*, *Overoptimization* and *Verbosity* issues. When ∥ξ <sup>∗</sup>∥<sup>1</sup> ≤ O(log T), the above generalization error rate is <sup>O</sup>e(1/ √ T), which also matches the existing results of the

394

396

Table 1: Hyperparameter Values and LC-win Rates of Offline DPO-type Algorithms

| Algorithms                                       | $\lambda$ | $\eta$ | $\omega$ | LC-win rates |
|--------------------------------------------------|-----------|--------|----------|--------------|
| <b>Our DPO-COV (all 3 components activated)</b>  | 0.7       | 0.0005 | 0.0005   | <b>7.61%</b> |
| Robust DPO ( <i>Corruption only</i> )            | 0.1       | 0      | 0        | 7.04%        |
| Pessimistic DPO ( <i>Overoptimization only</i> ) | 1         | 0.005  | 0        | 5.50%        |
| Length-regularized DPO ( <i>Verbosity only</i> ) | 1         | 0      | 0.0005   | 7.30%        |
| Vanilla DPO                                      | 1         | 0      | 0        | 6.29%        |
| Reference model $\pi_{\text{ref}}$               | -         | -      | -        | 4.92%        |

Table 2: Experimental Results on Math and Reasoning

| Model       |         | GSM8K | ARC (Easy) | ARC (Challenge) |
|-------------|---------|-------|------------|-----------------|
| Our         | DPO-COV | 46.78 | 72.52      | 49.32           |
| Robust      | DPO     | 46.25 | 72.14      | 47.35           |
| Pessimistic | DPO     | 45.19 | 72.14      | 46.16           |
| Length-reg  | DPO     | 44.50 | 72.31      | 46.16           |
| Vanilla     | DPO     | 45.26 | 71.89      | 46.50           |
| Reference   | Model   | 42.38 | 71.72      | 45.14           |

online optimistic DPO-type algorithms [\(Xie et al.,](#page-10-12) [2024;](#page-10-12) [Cen et al.,](#page-8-11) [2024\)](#page-8-11) up to logarithm.

Technical Novelty. Similar to the proof of Theorem [1,](#page-5-4) we also use the novel bounds on the effect of the estimated and true noise terms, which are obtained in Lemmas [5](#page-13-1) and [4](#page-13-0) respectively.

# 5. Experiments on Offline Data

In this section, we will compare the following offline DPOtype algorithms on offline datasets. The experiments to compare online DPO-type algorithms on online datasets are shown in Appendix [A.](#page-11-0)

- 1. Our offline DPO-COV algorithm with three modules activated (*Corruption*, *Overoptimization*, *Verbosity*): This is Algorithm [1](#page-4-0) with η, ω > 0 and λ ∈ (0, 1).
- 2. Offline robust DPO algorithm [\(Bukharin et al.,](#page-8-1) [2024\)](#page-8-1): This is a special case of Algorithm [1](#page-4-0) with η = ω = 0 and λ ∈ (0, 1), which only tackles *Corruption*.
- 3. Offline pessimistic DPO algorithm [\(Liu et al.,](#page-9-8) [2024c\)](#page-9-8): This is a special case of Algorithm [1](#page-4-0) with η > 0, ω = 0 and λ = 1, which only tackles *Overoptimization*.
- 4. Offline length regularized DPO algorithm [\(Park et al.,](#page-9-16) [2024\)](#page-9-16): This is a special case of Algorithm [1](#page-4-0) with η = 0, ω > 0 and λ = 1, which only tackles *Verbosity*.
- 5. Offline vanilla DPO [\(Rafailov et al.,](#page-9-1) [2023\)](#page-9-1): Algorithm [1](#page-4-0) with η = ω = 0 and λ = 1.

#### 5.1. Experiment on the Argilla Data

We select the preference dataset D to be Argilla-DPO-Mix-7K [\(Argill,](#page-8-18) [2024\)](#page-8-18), and πref to be zephyr-7b-gemma-sftv0.1 [\(HuggingFaceH4,](#page-9-18) [2024\)](#page-9-18), which is a fine-tuned version of gemma-7b on the Deita dataset [\(Wang et al.,](#page-10-20) [2023\)](#page-10-20). Then we apply LoRA [\(Hu et al.,](#page-9-19) [2021\)](#page-9-19) and two epochs of the AdamW optimizer [\(Loshchilov and Hutter,](#page-9-20) [2017\)](#page-9-20) with learning rate 5 × 10−<sup>7</sup> to the objective [\(19\)](#page-4-7). For each algorithm, we fix β = 0.05 and perform grid search on the other hyperparameters over a holdout validation set of the preference dataset. We compare the Length-Control win rates (a.k.a. LC-win rates, defined in AlpacaEval 2.0 [\(Dubois](#page-8-19) [et al.,](#page-8-19) [2024\)](#page-8-19)) of πref and that of the models obtained by the above algorithms against the model GPT-4 Preview (11/06) [\(OpenAI,](#page-9-21) [2024\)](#page-9-21). We summarize the LC-win rates and the hyperparameter values in Table [1,](#page-7-0) which indicates that our offline DPO-COV algorithm with all three components activated achieves the highest LC win rates. Therefore, it is important to tackle the *Corruption*, *Overoptimization* and *Verbosity* issues simultaneously.

#### 5.2. Experiment on Math and Reasoning

We also compare our Algorithm [1](#page-4-0) with other offline DPO variants over math and reasoning tasks: Grade School Math 8K (GSM8K) and AI2 Reasoning Challenge (ARC) tasks. We run the benchmark test with [\(Gao et al.,](#page-9-22) [2024a\)](#page-9-22) and report the accuracies in Table [2.](#page-7-1) The model hyper-parameters are the same as in Table [1.](#page-7-0) The results shown in Table [2](#page-7-1) indicate that our DPO-COV algorithm outperforms the other variants also on the math and reasoning tasks.

### 6. Conclusion

We proposed RLHF-COV and DPO-COV algorithms that simultaneously mitigate the *Corruption*, *Overoptimization* and *Verbosity* issues, in both offline and online settings. This ability is theoretically proved by length-regularized generalization analysis on corrupted data. In addition, we proved the equivalence of our proposed RLHF-COV and DPO-COV algorithms. A future direction is to extend this work to account for various preferences among diverse human groups [\(Ramesh et al.,](#page-10-21) [2024;](#page-10-21) [Chakraborty et al.,](#page-8-16) [2024\)](#page-8-16).

- 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 Impact Statement This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. References Argill (2024). Argilla-dpo-mix-7k. [https://huggin](https://huggingface.co/datasets/argilla/dpo-mix-7k) [gface.co/datasets/argilla/dpo-mix-7k](https://huggingface.co/datasets/argilla/dpo-mix-7k). Accessed: 2024-09-30. Azar, M. G., Guo, Z. D., Piot, B., Munos, R., Rowland, M., Valko, M., and Calandriello, D. (2024). A general theoretical paradigm to understand learning from human preferences. In *International Conference on Artificial Intelligence and Statistics (AISTATS)*, pages 4447–4455. Bai, C., Wang, L., Yang, Z., Deng, Z.-H., Garg, A., Liu, P., and Wang, Z. (2022a). Pessimistic bootstrapping for uncertainty-driven offline reinforcement learning. In *International Conference on Learning Representations (ICLR)*. Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., Das-Sarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al. (2022b). Training a helpful and harmless assistant with reinforcement learning from human feedback. *ArXiv:2204.05862*. Bradley, R. A. and Terry, M. E. (1952). Rank analysis of incomplete block designs: I. the method of paired comparisons. *Biometrika*, 39(3/4):324–345. Bukharin, A., Hong, I., Jiang, H., Zhang, Q., Zhang, Z., and Zhao, T. (2024). Robust reinforcement learning from corrupted human feedback. *ArXiv:2406.15568*. Cao, Y., Ivanovic, B., Xiao, C., and Pavone, M. (2024). Reinforcement learning with human feedback for realistic traffic simulation. In *IEEE International Conference on Robotics and Automation (ICRA)*, pages 14428–14434. IEEE. Casper, S., Davies, X., Shi, C., Gilbert, T. K., Scheurer, J., Rando, J., Freedman, R., Korbak, T., Lindner, D., Freire, P., et al. (2023). Open problems and fundamental limitations of reinforcement learning from human feedback. *Transactions on Machine Learning Research*. Cen, S., Mei, J., Goshvadi, K., Dai, H., Yang, T., Yang, S., Schuurmans, D., Chi, Y., and Dai, B. (2024). Valueincentivized preference optimization: A unified approach to online and offline rlhf. *ArXiv:2405.19320*. Chakraborty, S., Qiu, J., Yuan, H., Koppel, A., Huang, F., Manocha, D., Bedi, A. S., and Wang, M. (2024). Maxminrlhf: Towards equitable alignment of large language models with diverse human preferences. *ArXiv:2402.08925*. Chen, L., Zhu, C., Chen, J., Soselia, D., Zhou, T., Goldstein, T., Huang, H., Shoeybi, M., and Catanzaro, B. (2024). Odin: Disentangled reward mitigates hacking in rlhf. In *International Conference on Machine Learning (ICML)*. Cheng, C.-A., Xie, T., Jiang, N., and Agarwal, A. (2022). Adversarially trained actor critic for offline reinforcement learning. In *International Conference on Machine Learning (ICML)*, pages 3852–3878. PMLR. Cheng, J., Xiong, G., Dai, X., Miao, Q., Lv, Y., and Wang, F.-
  - Y. (2024). Rime: Robust preference-based reinforcement learning with noisy preferences. *ArXiv:2402.17257*. Christiano, P. F., Leike, J., Brown, T. B., Martic, M., Legg, S., and Amodei, D. (2017). Deep reinforcement learning from human preferences. In *International Conference on Neural Information Processing Systems (Neurips)*, pages 4302–4310. Coste, T., Anwar, U., Kirk, R., and Krueger, D. (2024). Reward model ensembles help mitigate overoptimization. In *International Conference on Learning Representations (ICLR)*. Dong, H., Xiong, W., Pang, B., Wang, H., Zhao, H., Zhou, Y., Jiang, N., Sahoo, D., Xiong, C., and Zhang, T. (2024). Rlhf workflow: From reward modeling to online rlhf. *ArXiv:2405.07863*. Dubois, Y., Galambosi, B., Liang, P., and Hashimoto, T. B. (2024). Length-controlled alpacaeval: A simple way to debias automatic evaluators. Eisenstein, J., Nagpal, C., Agarwal, A., Beirami, A., D'Amour, A. N., Dvijotham, K. D., Fisch, A., Heller,
  - K. A., Pfohl, S. R., Ramachandran, D., Shaw, P., and Berant, J. (2024). Helping or herding? reward model ensembles mitigate but do not eliminate reward hacking. In *First Conference on Language Modeling (COLM)*. Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D., and Kiela, D. (2024). Kto: Model alignment as prospect theoretic optimization. *ArXiv:2402.01306*. Fan, K. (1953). Minimax theorems. *Proceedings of the National Academy of Sciences*, 39(1):42–47. Fisch, A., Eisenstein, J., Zayats, V., Agarwal, A., Beirami, A., Nagpal, C., Shaw, P., and Berant, J. (2024). Robust preference optimization through reward model distillation. *ArXiv:2405.19316*.

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

- Gao, L., Schulman, J., and Hilton, J. (2023). Scaling laws for reward model overoptimization. In *International Conference on Machine Learning (ICML)*, pages 10835– 10866. Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPofi, A., Foster, C., Golding, L., Hsu, J., Le Noac'h, A., Li, H., McDonell, K., Muennighoff, N., Ociepa, C., Phang, J., Reynolds, L., Schoelkopf, H., Skowron, A., Sutawika, L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou,
- A. (2024a). A framework for few-shot language model evaluation. [https://zenodo.org/records/1](https://zenodo.org/records/12608602) [2608602](https://zenodo.org/records/12608602). Gao, Y., Alon, D., and Metzler, D. (2024b). Impact of preference noise on the alignment performance of generative language models. In *First Conference on Language Modeling (CoLM)*. Guo, S., Zhang, B., Liu, T., Liu, T., Khalman, M., Llinares, F., Rame, A., Mesnard, T., Zhao, Y., Piot, B., et al. (2024). Direct language model alignment from online ai feedback. *ArXiv:2402.04792*. Harsha, P. (2011). Lecture note 12 on communication complexity. url: [https://www.tcs.tifr.res.in/](https://www.tcs.tifr.res.in/~prahladh/teaching/2011-12/comm/lectures/l12.pdf) [~prahladh/teaching/2011-12/comm/lectu](https://www.tcs.tifr.res.in/~prahladh/teaching/2011-12/comm/lectures/l12.pdf) [res/l12.pdf](https://www.tcs.tifr.res.in/~prahladh/teaching/2011-12/comm/lectures/l12.pdf). Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. (2021). Lora: Low-rank adaptation of large language models. Huang, A., Zhan, W., Xie, T., Lee, J. D., Sun, W., Krishnamurthy, A., and Foster, D. J. (2024). Correcting the mythos of kl-regularization: Direct alignment without overparameterization via chi-squared preference optimization. *ArXiv:2407.13399*. HuggingFaceH4 (2024). zephyr-7b-gemma-sft-v0.1. [http](https://huggingface.co/HuggingFaceH4/zephyr-7b-gemma-sft-v0.1) [s://huggingface.co/HuggingFaceH4/zep](https://huggingface.co/HuggingFaceH4/zephyr-7b-gemma-sft-v0.1) [hyr-7b-gemma-sft-v0.1](https://huggingface.co/HuggingFaceH4/zephyr-7b-gemma-sft-v0.1). Accessed: 2024-09-30. Ji, X., Kulkarni, S., Wang, M., and Xie, T. (2024). Selfplay with adversarial critic: Provable and scalable offline alignment for language models. *ArXiv:2406.04274*. Jin, Y., Yang, Z., and Wang, Z. (2021). Is pessimism provably efficient for offline rl? In *International Conference on Machine Learning (ICML)*, pages 5084–5096. Liang, Z., Yuan, Y., Gu, S., Chen, B., Hang, T., Li, J., and Zheng, L. (2024). Step-aware preference optimization: Aligning preference with denoising performance at each step. *ArXiv:2406.04314*. Liu, J., Zhou, Z., Liu, J., Bu, X., Yang, C., Zhong, H.-S., and Ouyang, W. (2024a). Iterative length-regularized direct preference optimization: A case study on improving 7b language models to gpt-4 level. *ArXiv:2406.11817*. Liu, Q., Weisz, G., György, A., Jin, C., and Szepesvári,
  - C. (2023a). Optimistic natural policy gradient: a simple efficient policy optimization framework for online rl. In *International Conference on Neural Information Processing Systems (Neurips)*, pages 3560–3577. Liu, Y., Zhang, K., Li, Y., Yan, Z., Gao, C., Chen, R., Yuan, Z., Huang, Y., Sun, H., Gao, J., et al. (2024b). Sora: A review on background, technology, limitations, and opportunities of large vision models. *ArXiv:2402.17177*. Liu, Z., Lu, M., Xiong, W., Zhong, H., Hu, H., Zhang, S., Zheng, S., Yang, Z., and Wang, Z. (2023b). Maximize to explore: one objective function fusing estimation, planning, and exploration. In *International Conference on Neural Information Processing Systems (Neurips)*, pages 22151–22165. Liu, Z., Lu, M., Zhang, S., Liu, B., Guo, H., Yang, Y., Blanchet, J., and Wang, Z. (2024c). Provably mitigating overoptimization in rlhf: Your sft loss is implicitly an adversarial regularizer. *ArXiv:2405.16436*. Loshchilov, I. and Hutter, F. (2017). Decoupled weight decay regularization. Mandal, D., Nika, A., Kamalaruban, P., Singla, A., and Radanovic, G. (2024). Corruption robust offline reinforce- ´ ment learning with human feedback. *ArXiv:2402.06734*. Meng, Y., Xia, M., and Chen, D. (2024). Simpo: Simple preference optimization with a reference-free reward. *ArXiv:2405.14734*. Moskovitz, T., Singh, A. K., Strouse, D., Sandholm, T., Salakhutdinov, R., Dragan, A., and McAleer, S. M. (2024). Confronting reward model overoptimization with constrained rlhf. In *International Conference on Learning Representations (ICLR)*. OpenAI (2024). Gpt-4 technical report. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Gray, A., et al. (2022). Training language models to follow instructions with human feedback. In *International Conference on Neural Information Processing Systems (Neurips)*. Park, R., Rafailov, R., Ermon, S., and Finn, C. (2024). Disentangling length from quality in direct preference optimization. *ArXiv:2403.19159*. Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward

- 551 554 556 558 560 564 566 568 571 574 576 578 580 581 582 583 584 585 586 587 588 589 590 594 596 598 600 601 602 603 model. In *International Conference on Neural Information Processing Systems (Neurips)*, volume 36. Rame, A., Vieillard, N., Hussenot, L., Dadashi, R., Cideron, G., Bachem, O., and Ferret, J. (2024). Warm: On the benefits of weight averaged reward models. In *International Conference on Machine Learning (ICML)*. Ramesh, S. S., Hu, Y., Chaimalas, I., Mehta, V., Sessa,
  - P. G., Ammar, H. B., and Bogunovic, I. (2024). Group robust preference optimization in reward-free rlhf. *ArXiv:2405.20304*. Rashidinejad, P., Zhu, B., Ma, C., Jiao, J., and Russell,
  - S. (2021). Bridging offline reinforcement learning and imitation learning: a tale of pessimism. In *International Conference on Neural Information Processing Systems (Neurips)*, pages 11702–11716. RLHFlow (2024). pair-preference-model-llama3-8b. [http](https://huggingface.co/RLHFlow/pair-preference-model-LLaMA3-8B) [s://huggingface.co/RLHFlow/pair-prefe](https://huggingface.co/RLHFlow/pair-preference-model-LLaMA3-8B) [rence-model-LLaMA3-8B](https://huggingface.co/RLHFlow/pair-preference-model-LLaMA3-8B). Accessed: 2024-09-30. Shen, W., Zheng, R., Zhan, W., Zhao, J., Dou, S., Gui, T., Zhang, Q., and Huang, X.-J. (2023). Loose lips sink ships: Mitigating length bias in reinforcement learning from human feedback. In *Findings of the Association for Computational Linguistics: EMNLP 2023*, pages 2859– 2873. Singhal, P., Goyal, T., Xu, J., and Durrett, G. (2023). A long way to go: Investigating length correlations in rlhf. *ArXiv:2310.03716*. Tversky, A. and Kahneman, D. (1992). Advances in prospect theory: Cumulative representation of uncertainty. *Journal of Risk and uncertainty*, 5:297–323. Wallace, B., Dang, M., Rafailov, R., Zhou, L., Lou, A., Purushwalkam, S., Ermon, S., Xiong, C., Joty, S., and Naik, N. (2023). Diffusion model alignment using direct preference optimization. *AXiv:2311.12908*. Wang, Y., Liu, L., Wang, M., and Xiong, X. (2024). Reinforcement learning from human feedback for lane changing of autonomous vehicles in mixed traffic. *ArXiv:2408.04447*. Wang, Y., Liu, Q., and Jin, C. (2023). Is rlhf more difficult than standard rl? *ArXiv*, abs/2306.14111. Wei, C.-Y., Hong, Y.-T., and Lu, C.-J. (2017). Online reinforcement learning in stochastic games. In *International Conference on Neural Information Processing Systems (Neurips)*, pages 4994–5004. Xie, T., Cheng, C. A., Jiang, N., Mineiro, P., and Agarwal,
- A. (2021). Bellman-consistent pessimism for offline reinforcement learning. In *International Conference on Neural Information Processing Systems (Neurips)*, pages 6683–6694. Xie, T., Foster, D. J., Krishnamurthy, A., Rosset, C., Awadallah, A., and Rakhlin, A. (2024). Exploratory preference optimization: Harnessing implicit q\*-approximation for sample-efficient rlhf. *ArXiv:2405.21046*. Xiong, W., Dong, H., Ye, C., Wang, Z., Zhong, H., Ji, H., Jiang, N., and Zhang, T. (2024). Iterative preference learning from human feedback: Bridging theory and practice for rlhf under kl-constraint. In *International Conference on Machine Learning (ICML)*. Xu, W., Li, J., Wang, W. Y., and Li, L. (2024). Bpo: Supercharging online preference learning by adhering to the proximity of behavior llm. *ArXiv:2406.12168*. Yang, R., Ding, R., Lin, Y., Zhang, H., and Zhang, T. (2024). Regularizing hidden states enables learning generalizable reward model for llms. *ArXiv:2406.10216*. Ye, C., Xiong, W., Zhang, Y., Jiang, N., and Zhang,
  - T. (2024). Online iterative reinforcement learning from human feedback with general preference model. *ArXiv:2402.07314*. Zhai, Y., Zhang, H., Lei, Y., Yu, Y., Xu, K., Feng, D., Ding, B., and Wang, H. (2023). Uncertainty-penalized reinforcement learning from human feedback with diverse reward lora ensembles. *ArXiv:2401.00243*. Zhan, W., Uehara, M., Kallus, N., Lee, J. D., and Sun, W. (2024). Provable offline preference-based reinforcement learning. In *International Conference on Learning Representations (ICLR)*. Zhang, S., Yu, D., Sharma, H., Yang, Z., Wang, S., Hassan, H., and Wang, Z. (2024). Self-exploring language models: Active preference elicitation for online alignment. *ArXiv:2405.19332*. Zhang, T. (2023). *Mathematical analysis of machine learning algorithms*. Cambridge University Press. Zhong, H. and Zhang, T. (2023). A theoretical analysis of optimistic proximal policy optimization in linear markov decision processes. In *International Conference on Neural Information Processing Systems (Neurips)*, pages 73666– 73690. Zhu, B., Jordan, M., and Jiao, J. (2023). Principled reinforcement learning with human feedback from pairwise or k-wise comparisons. In *International Conference on Machine Learning (ICML)*, pages 43037–43067. Zhu, B., Jordan, M., and Jiao, J. (2024). Iterative data smoothing: Mitigating reward overfitting and overoptimization in rlhf. In *International Conference on Machine Learning (ICML)*.

| 611 612 |            |    |             |             |    |
|---------|------------|----|-------------|-------------|----|
| A       | Experiment |    | on          | Online Data | 12 |
| B       | Supporting |    | Lemmas      |             | 12 |
| C       | Proof      | of | Proposition | 1           | 23 |
| D       | Proof      | of | Proposition | 2           | 24 |
| E       | Proof      | of | Proposition | 3           | 25 |
| F       | Proof      | of | Theorem     | 1           | 25 |
| G       | Proof      | of | Theorem     | 2           | 26 |

# Appendix

# Table of Contents

# A. Experiment on Online Data

Similar to the offline experiments in Section [5,](#page-7-2) we compare important special cases of Algorithm [2,](#page-6-3) including our online DPO-COV with all 3 components activated, the online variant of the robust DPO algorithm [\(Bukharin et al.,](#page-8-1) [2024\)](#page-8-1), online optimistic DPO algorithm (named XPO in [\(Xie et al.,](#page-10-12) [2024\)](#page-10-12)), online length regularized DPO algorithm [\(Liu et al.,](#page-9-14) [2024a\)](#page-9-14) and online vanilla DPO algorithm (using DPO objective in [\(Guo et al.,](#page-9-15) [2024\)](#page-9-15)). We use zephyr-7b-gemma-sftv0.1 [\(HuggingFaceH4,](#page-9-18) [2024\)](#page-9-18) as the reference model πref and the initial model π0. Each algorithm is trained with β = 0.05 and T = 3 iterations. In each iteration, we generate the online labels y<sup>t</sup> from pair-preference-model-LLaMA3-8B [\(RLHFlow,](#page-10-22) [2024\)](#page-10-22), and combine the online data with 50% of the preference dataset of Argilla-DPO-Mix-7K [\(Argill,](#page-8-18) [2024\)](#page-8-18). Then we apply LoRA [\(Hu et al.,](#page-9-19) [2021\)](#page-9-19) and two epochs of the AdamW optimizer [\(Loshchilov and Hutter,](#page-9-20) [2017\)](#page-9-20) with stepsize × 10−<sup>7</sup> to the objective [\(25\)](#page-6-2). On AlpacaEval 2.0 [\(Dubois et al.,](#page-8-19) [2024\)](#page-8-19), we compare the LC-win rates of πref and that of the models obtained by the above algorithms against the model GPT-4 Preview (11/06) [\(OpenAI,](#page-9-21) [2024\)](#page-9-21). Again, the results in Table [3](#page-11-2) indicate that our online DPO-COV algorithm with all three components activated achieves the highest length-control win rates. Therefore, it is important to tackle the *Corruption*, *Overoptimization* and *Verbosity* issues simultaneously.

Table 3: Hyperparameter Values and LC-win Rates of Online DPO-type Algorithms

| Algorithms                                       | $\lambda$ | $\eta$ | $\omega$ | LC-win rates |
|--------------------------------------------------|-----------|--------|----------|--------------|
| <b>Our DPO-COV (all 3 components activated)</b>  | 0.7       | 0.0005 | 0.0005   | <b>7.87%</b> |
| Robust DPO ( <i>Corruption only</i> )            | 0.1       | 0      | 0        | 7.03%        |
| Optimistic DPO ( <i>Overoptimization only</i> )  | 1         | 0.005  | 0        | 6.23%        |
| Length-regularized DPO ( <i>Verbosity only</i> ) | 1         | 0      | 0.0005   | 6.19%        |
| Vanilla DPO                                      | 1         | 0      | 0        | 6.58%        |
| Reference model $\pi_{\text{ref}}$               | -         | -      | -        | 4.92%        |

# B. Supporting Lemmas

Lemma 1. *For any* A ∈ (0, ∞) *and* z1, z<sup>2</sup> ∈ [−R, R]*, the following inequality holds.*

$$\frac{|z_1 - z_2|}{3 + e^R} \leq |\sigma(z_1) - \sigma(z_2)| \leq \frac{1}{4}|z_1 - z_2|. \quad (28)$$

689 690

694

696

698

700

704

706

708 709

711

714

Remark: Our bound [\(28\)](#page-11-3) is strictly tighter than <sup>|</sup>z1−z2<sup>|</sup> (1+e<sup>R</sup>) <sup>2</sup> ≤ |σ(z1) − σ(z2)| ≤ |z<sup>1</sup> − z2| obtained in Lemma A.2 of [\(Liu](#page-9-8) [et al.,](#page-9-8) [2024c\)](#page-9-8).

*Proof.* Denote zmin = min(z1, z2) and zmax = max(z1, z2). Then we have

$$\begin{aligned} |z_1 - z_2| &= z_{\max} - z_{\min}, \\ |\sigma(z_1) - \sigma(z_2)| &= \sigma(z_{\max}) - \sigma(z_{\min}) = \int_{z_{\min}}^{z_{\max}} \sigma'(z) dz. \end{aligned}$$

Hence, it suffices to prove that σ ′ (v) ∈ -1 3+e<sup>R</sup> , 1 4 for any v ∈ [zmin, zmax] ⊂ [−R, R]. Note that for any v ∈ [zmin, zmax] ⊂ [−R, R], σ(v) ∈ [σ(−R), σ(R)] = [1 − σ(R), σ(R)]. Hence, we conclude the proof by the following two bounds.

$$\begin{aligned}\sigma'(v) &= \sigma(v)[1 - \sigma(v)] = \frac{1}{4} - \left[ \sigma(v) - \frac{1}{2} \right]^2 \leq \frac{1}{4}. \\ \sigma'(v) &= \frac{1}{4} - \left[ \sigma(v) - \frac{1}{2} \right]^2 \\ &\geq \frac{1}{4} - \left[ \sigma(R) - \frac{1}{2} \right]^2 \\ &= \sigma(R)[1 - \sigma(R)] \\ &= \frac{1}{1 + e^R} \frac{e^R}{1 + e^R} \\ &= \frac{1}{(1 + e^R)(1 + e^{-R})} \\ &= \frac{1}{2 + e^R + e^{-R}} \geq \frac{1}{3 + e^R}.\end{aligned}$$

Lemma 2. *For any* x ∈ X *,* a0, a<sup>1</sup> ∈ A *and* r ∈ R*, the following equality holds*

$$r^{\pi_r}(x, a_1) - r^{\pi_r}(x, a_0) = r(x, a_1) - r(x, a_0), \quad (29)$$

*where* π<sup>r</sup> *and* r <sup>π</sup> *are defined by Eqs.* [\(14\)](#page-3-7) *and* [\(16\)](#page-4-2) *respectively. Furthermore, under Assumption [2,](#page-5-2) both sides of the above Eq.* [\(29\)](#page-12-1) *range in* [−R, R]*.*

*Proof.*

$$\begin{aligned} & r^{\pi_r}(x, a_1) - r^{\pi_r}(x, a_0) \\ & \stackrel{(a)}{=} \omega(|a_1| - |a_0|) + \beta \log \left( \frac{\pi_r(a_1|x)\pi_{\text{ref}}(a_0|x)}{\pi_r(a_0|x)\pi_{\text{ref}}(a_1|x)} \right) \\ & \stackrel{(b)}{=} r(x, a_1) - r(x, a_0), \end{aligned}$$

where (a) uses Eq. [\(16\)](#page-4-2) and (b) uses Eq. [\(14\)](#page-3-7).

Furthermore, under Assumption [2,](#page-5-2) r(x, a0), r(x, a1) ∈ [0, R], so

$$r^{\pi_r}(x, a_1) - r^{\pi_r}(x, a_0) = r(x, a_1) - r(x, a_0) \in [-R, R].$$

Lemma 3. *Any policy* π ∈ Π *satisfies* π = πr<sup>π</sup> *where* π<sup>r</sup> *and* r <sup>π</sup> *are defined by Eqs.* [\(14\)](#page-3-7) *and* [\(16\)](#page-4-2) *respectively. Furthermore, under Assumption [2,](#page-5-2) any* π ∈ Π<sup>R</sup> def <sup>=</sup> {π<sup>r</sup> : <sup>r</sup> ∈ R} *satisfies* |<sup>r</sup> π (x, a1) − r π (x, a0)| ≤ R *for any* x ∈ X *,* a0, a<sup>1</sup> ∈ A*.*

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

*Proof.* Eq. [\(16\)](#page-4-2) implies that for any x ∈ X and a ∈ A, we have

$$\pi_{\text{ref}}(a|x) \exp \left[ \frac{r^\pi(x, a) - \omega|a|}{\beta} \right] = \pi(a|x). \quad (30)$$

Hence,

$$Z_{r,\pi}(x) = \sum_{a \in \mathcal{A}} \pi_{\text{ref}}(a|x) \exp \left[ \frac{r^{\pi}(x, a) - \omega|a|}{\beta} \right] = \sum_{a \in \mathcal{A}} \pi(a|x) = 1. \quad (31)$$

Therefore, π = πr<sup>π</sup> can be proved as follows.

$$\pi_{r^\pi}(a|x) \stackrel{(a)}{=} \frac{\pi_{\text{ref}}(a|x)}{Z_{r^\pi}(x)} \exp \left[ \frac{r^\pi(x, a) - \omega|a|}{\beta} \right] \stackrel{(b)}{=} \pi(a|x),$$

where (a) uses Eq. [\(14\)](#page-3-7) and (b) uses Eqs. [\(30\)](#page-13-2) and [\(31\)](#page-13-3).

When π ∈ Π<sup>R</sup> def <sup>=</sup> {π<sup>r</sup> : <sup>r</sup> ∈ R}, there exists <sup>r</sup> ∈ R such that <sup>π</sup> <sup>=</sup> <sup>π</sup>r. Hence,

$$|r^\pi(x, a_1) - r^\pi(x, a_0)| \stackrel{(a)}{=} |r^{\pi_r}(x, a_1) - r^{\pi_r}(x, a_0)| \stackrel{(b)}{=} |r(x, a_1) - r(x, a_0)| \stackrel{(c)}{\leq} R,$$

where (a) uses π = πr, (b) uses Eq. [\(29\)](#page-12-1) and (c) uses Assumption [2.](#page-5-2)

Lemma 4. *Under Assumption [2,](#page-5-2) for any* r ∈ R *and* ξr,i *defined by Eq.* [\(15\)](#page-4-1)*, the following inequality holds.*

$$\log \sigma[r(x_i, a_i^w) - r(x_i, a_i^\ell) + y_i\xi_{r,i}] \leq \log \sigma[r(x_i, a_i^w) - r(x_i, a_i^\ell)] + \sigma(R)|\xi_{r,i}|. \quad (32)$$

*For any* π ∈ Π<sup>R</sup> def <sup>=</sup> {π<sup>r</sup> : <sup>r</sup> ∈ R} *and* <sup>ξ</sup> π i *defined by Eq.* [\(17\)](#page-4-4)*, the following inequality holds.*

$$\log \sigma[r^\pi(x_i, a_i^w) - r^\pi(x_i, a_i^w) + y_i\xi_i^\pi] \leq \log \sigma[r^\pi(x_i, a_i^w) - r^\pi(x_i, a_i^w)] + \sigma(R)|\xi_i^\pi|. \quad (33)$$

*Proof.* yiξr,i ≥ 0 by Eq. [\(15\)](#page-4-1) since y<sup>i</sup> ∈ {−1, 1}. Then Eq. [\(32\)](#page-13-4) follows from <sup>d</sup> dv [log σ(v)] = σ(−v) ≤ σ(R) for any v ∈ [r(x<sup>i</sup> , a<sup>w</sup> i ) − r(x<sup>i</sup> , a<sup>ℓ</sup> i ), r(x<sup>i</sup> , a<sup>w</sup> i ) − r(x<sup>i</sup> , a<sup>ℓ</sup> i ) + yiξr,i] ⊆ [−R, +∞) where ⊂ is implied by Assumption [2.](#page-5-2)

Similarly, yiξ π <sup>i</sup> ≥ <sup>0</sup> by Eq. [\(17\)](#page-4-4) since y<sup>i</sup> ∈ {−1, <sup>1</sup>}. Then Eq. [\(33\)](#page-13-5) follows from <sup>d</sup> dv [log σ(v)] = σ(−v) ≤ σ(R) for any v ∈ [r π (x<sup>i</sup> , a<sup>w</sup> i ) − r π (x<sup>i</sup> , a<sup>ℓ</sup> i ), r<sup>π</sup> (x<sup>i</sup> , a<sup>w</sup> i ) − r π (x<sup>i</sup> , a<sup>ℓ</sup> i ) + yiξ π i ] ⊆ [−R, +∞) where ⊂ is implied by Lemma [3.](#page-12-0)

Lemma 5. *For any* ξ<sup>i</sup> ∈ <sup>R</sup> *and reward models* r, r′ : X × A → R*, we have*

$$\begin{aligned} & \{\sigma[r'(x_i, a_i^w) - r'(x_i, a_i^\ell) + y_i\xi_i] - \sigma[r(x_i, a_i^w) - r(x_i, a_i^\ell)]\}^2 \\ & \geq \{\sigma[r'(x_i, a_i^w) - r'(x_i, a_i^\ell)] - \sigma[r(x_i, a_i^w) - r(x_i, a_i^\ell)]\}^2 - \frac{1}{2}|\xi_i^*|. \end{aligned} \quad (34)$$

*Proof.* Denote A′ <sup>i</sup> = r ′ (x<sup>i</sup> , a<sup>w</sup> i ) − r ′ (x<sup>i</sup> , a<sup>ℓ</sup> i ) and A<sup>i</sup> = r(x<sup>i</sup> , a<sup>w</sup> i ) − r(x<sup>i</sup> , a<sup>ℓ</sup> i ). Define the following function.

$$f(u) = [\sigma(A'_i + u) - \sigma(A_i)]^2. \quad (35)$$

Note that the range of the sigmoid function σ is (0, 1). Hence, for any u ∈ <sup>R</sup>,

$$\frac{d}{du} f(u) = 2\sigma(A'_i + u)[1 - \sigma(A'_i + u)][\sigma(A'_i + u) - \sigma(A_i)] \in \left(-\frac{1}{2}, \frac{1}{2}\right). \quad (36)$$

Therefore,

$$f(0) - f(y_i\xi_i) \leq |f(y_i\xi_i) - f(0)| \leq \frac{1}{2}|y_i\xi_i| = \frac{1}{2}|\xi_i|,$$

774

776

778

794

796

800

804

806

808

Lemma 6. *For any* x ∈ X *,* a ∈ A *and* r, r′ ∈ R*, the policies* πr*,* π<sup>r</sup> ′ *defined by the analytical solution* [\(14\)](#page-3-7) *satisfy*

$$\left| \log \frac{\pi_{r'}(a|x)}{\pi_r(a|x)} \right| \leq \frac{2\|r' - r\|_\infty}{\beta}, \quad (37)$$

*where* ∥r ′ − r∥<sup>∞</sup> = supx∈X ,a∈A |r ′ (x, a) − r(x, a)|*.*

*Proof.* Note that for any x ∈ X , a ′ ∈ A and r, r′ ∈ R, we have

$$\begin{aligned} \frac{\pi_{\text{ref}}(a'|x) \exp \left[ \frac{r'(x,a') - \omega|a'|}{\beta} \right]}{\pi_{\text{ref}}(a'|x) \exp \left[ \frac{r(x,a') - \omega|a'|}{\beta} \right]} &= \exp \left[ \frac{r'(x,a') - r(x,a')}{\beta} \right] \\ &\in \left[ \exp(-\|r' - r\|_\infty/\beta), \exp(\|r' - r\|_\infty/\beta) \right]. \end{aligned}$$

Therefore,

$$\begin{aligned} \frac{Z_{r'}(x)}{Z_r(x)} &= \frac{\sum_{a' \in \mathcal{A}} \pi_{\text{ref}}(a'|x) \exp \left[ \frac{r'(x,a') - \omega|a'|}{\beta} \right]}{\sum_{a' \in \mathcal{A}} \pi_{\text{ref}}(a'|x) \exp \left[ \frac{r(x,a') - \omega|a'|}{\beta} \right]} \\ &\in \left[ \exp(-\|r' - r\|_\infty / \beta), \exp(\|r' - r\|_\infty / \beta) \right]. \end{aligned}$$

As a result,

$$\begin{aligned} \frac{\pi_{r'}(a|x)}{\pi_r(a|x)} &= \left( \frac{Z_{r'}(x)}{Z_r(x)} \right)^{-1} \frac{\pi_{\text{ref}}(a'|x) \exp \left[ \frac{r'(x,a') - \omega|a'|}{\beta} \right]}{\pi_{\text{ref}}(a'|x) \exp \left[ \frac{r(x,a') - \omega|a'|}{\beta} \right]} \\ &\in [\exp(-2\|r' - r\|_{\infty}/\beta), \exp(2\|r' - r\|_{\infty}/\beta)] \end{aligned} \quad (38)$$

which directly implies Eq. [\(37\)](#page-14-1).

We slightly adjust Theorem 13.2 of [\(Zhang,](#page-10-23) [2023\)](#page-10-23) as follows, by using filtration F<sup>t</sup> = ∅ (so the conditional expectation becomes the total expectation), replacing −ξ<sup>i</sup> with Z<sup>i</sup> , and negating the small probability event.

Lemma 7. *Consider random variables* {Zi} N <sup>i</sup>=0*. For any* δ ∈ (0, 1) *and* λ ′ > 0*, the following inequality holds simultaneously for all* n = 1, 2, . . . , N *with probability at least* 1 − δ*.*

$$\sum_{i=1}^n Z_i \leq \frac{\log(1/\delta)}{\lambda'} + \frac{1}{\lambda'} \sum_{i=1}^n \log \mathbb{E}[\exp(\lambda' Z_i)].$$

Lemma 8. *Fix* ϵ > 0*,* λ ∈ [σ(R), 1] *and* δ ∈ (0, 1)*. Under Assumption [1,](#page-2-3) the following bound holds for any* r ∈ R *and* ξ<sup>r</sup> = [ξr,1, . . . , ξr,N ] ∈ <sup>R</sup> <sup>N</sup> *(given by Eq.* [\(15\)](#page-4-1)*) simultaneously with probability at least* 1 − δ*.*

$$\mathcal{L}_{N,\lambda}(r^*, \xi^*) - \mathcal{L}_{N,\lambda}(r, \xi_r) \leq \frac{2}{N} [\|\xi^*\|_1 + \log\left(\frac{|\mathcal{N}_\epsilon(\mathcal{R})|}{\delta}\right)] - \frac{E_r^2}{2(3+e^R)^2} + 7\epsilon, \quad (39)$$

*where* <sup>E</sup><sup>r</sup> := q E<sup>D</sup> r <sup>∗</sup>(x1, a<sup>w</sup> 1 ) − r <sup>∗</sup>(x1, a<sup>ℓ</sup> 1 ) − r(x1, a<sup>w</sup> ) + r(x1, a<sup>ℓ</sup> ) 2 *and* Nϵ(R) *is a finite* ϵ*-cover of* R*, that is, for any* r ∈ R*, there exists* r † ∈ Nϵ(R) *satisfying* ∥r † − r∥<sup>∞</sup> ≤ ϵ*.*

*Proof.* Based on Assumption [1,](#page-2-3) given (x<sup>i</sup> , a (1) i , a (−1) i ), the target label y ∈ {−1, 1} as well as the underlying reward r and noise ξ<sup>i</sup> , the event y<sup>i</sup> = y occurs with the following probability.

$$p_{r,\xi_i}(y|x_i, a_i^{(1)}, a_i^{(-1)}) = \begin{cases} \sigma[r(x_i, a_i^{(1)}) - r(x_i, a_i^{(-1)}) + \xi_i], & y = 1 \\ \sigma[r(x_i, a_i^{(-1)}) - r(x_i, a_i^{(1)}) - \xi_i], & y = -1. \end{cases} \quad (40)$$

By merging the two cases above, we have

$$p_{r,\xi_i}(y_i|x_i, a_i^{(1)}, a_i^{(-1)}) = \sigma[r(x_i, a_i^w) - r(x_i, a_i^\ell) + y_i\xi_i]. \quad (41)$$

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

Define the following random variables for r ∈ R and i = 1, . . . , N.

$$Z_i(r) = \frac{1}{2} \log \frac{\sigma[r(x_i, a_i^w) - r(x_i, a_i^\ell)]}{\sigma[r^*(x_i, a_i^w) - r^*(x_i, a_i^\ell) + y_i\xi_i^*]} = \frac{1}{2} \log \frac{p_{r,0}(y_i|x_i, a_i^{(1)}, a_i^{(-1)})}{p_{r^*, \xi_i^*}(y_i|x_i, a_i^{(1)}, a_i^{(-1)})}. \quad (42)$$

Then the following inequality holds for finitely many r ∈ Nϵ(R) simultaneously with probability at least 1 − δ.

$$\begin{aligned} & \mathcal{L}_{N,\lambda}(r^*, \xi^*) - \mathcal{L}_{N,\lambda}(r, \xi_r) \\ &= \frac{1}{N} \sum_{i=1}^N \{ \log \sigma[r(x_i, a_i^w) - r(x_i, a_i^\ell) + y_i \xi_{r,i}] - \log \sigma[r^*(x_i, a_i^w) - r^*(x_i, a_i^\ell) + y_i \xi_i^*] + \lambda(|\xi_i^*| - |\xi_{r,i}|) \} \\ &\stackrel{(a)}{\leq} \frac{1}{N} \sum_{i=1}^N \{ \log \sigma[r(x_i, a_i^w) - r(x_i, a_i^\ell)] + \sigma(R) |\xi_{r,i}| - \log \sigma[r^*(x_i, a_i^w) - r^*(x_i, a_i^\ell) + y_i \xi_i^*] \\ &\quad + \lambda(|\xi_i^*| - |\xi_{r,i}|) \} \\ &\stackrel{(b)}{\leq} \frac{1}{N} \sum_{i=1}^N [|\xi_i^*| + 2Z_i(r)] \\ &\stackrel{(c)}{\leq} \frac{1}{N} \sum_{i=1}^N \{ |\xi_i^*| + 2 \log \mathbb{E}_{\mathcal{D}} [\exp[Z_i(r)]] \} + \frac{2}{N} \log \left( \frac{|\mathcal{N}_\epsilon(\mathcal{R})|}{\delta} \right) \\ &\stackrel{(d)}{=} \frac{2}{N} \sum_{i=1}^N \log \mathbb{E}_{\mathcal{D}} \left\{ \mathbb{E}_{y_i \sim p_{r^*, \xi_i^*}(\cdot | x_i, a_i^{(1)}, a_i^{(-1)})} \left[ \sqrt{\frac{p_{r,0}(y_i | x_i, a_i^{(1)}, a_i^{(-1)})}{p_{r^*, \xi_i^*}(y_i | x_i, a_i^{(1)}, a_i^{(-1)})}} \right] x_i, a_i^{(1)}, a_i^{(-1)} \right] \} \\ &\quad + \frac{1}{N} [\|\xi^*\|_1 + 2 \log \left( \frac{|\mathcal{N}_\epsilon(\mathcal{R})|}{\delta} \right)] \\ &\stackrel{(e)}{\leq} \frac{2}{N} \sum_{i=1}^N \mathbb{E}_{\mathcal{D}} \left[ \sum_{y \in \{-1,1\}} \sqrt{p_{r,0}(y | x_i, a_i^{(1)}, a_i^{(-1)})} p_{r^*, \xi_i^*}(y | x_i, a_i^{(1)}, a_i^{(-1)}) - 1 \right] \\ &\quad + \frac{1}{N} [\|\xi^*\|_1 + 2 \log \left( \frac{|\mathcal{N}_\epsilon(\mathcal{R})|}{\delta} \right)] \\ &= - \frac{1}{N} \sum_{i=1}^N \mathbb{E}_{\mathcal{D}} \left[ \sum_{y \in \{-1,1\}} \left| \sqrt{p_{r,0}(y | x_i, a_i^{(1)}, a_i^{(-1)})} - \sqrt{p_{r^*, \xi_i^*}(y | x_i, a_i^{(1)}, a_i^{(-1)})} \right|^2 \right] \\ &\quad + \frac{1}{N} [\|\xi^*\|_1 + 2 \log \left( \frac{|\mathcal{N}_\epsilon(\mathcal{R})|}{\delta} \right)] \\ &\stackrel{(f)}{\leq} - \frac{1}{4N} \sum_{i=1}^N \mathbb{E}_{\mathcal{D}} \left[ \sum_{y \in \{-1,1\}} |p_{r,0}(y | x_i, a_i^{(1)}, a_i^{(-1)}) - p_{r^*, \xi_i^*}(y | x_i, a_i^{(1)}, a_i^{(-1)})|^2 \right] \\ &\quad + \frac{1}{N} [\|\xi^*\|_1 + 2 \log \left( \frac{|\mathcal{N}_\epsilon(\mathcal{R})|}{\delta} \right)] \\ &\stackrel{(g)}{=} - \frac{1}{2N} \sum_{i=1}^N \mathbb{E}_{\mathcal{D}} \{ \sigma[r^*(x_i, a_i^w) - r^*(x_i, a_i^\ell) + y_i \xi_i^*] - \sigma[r(x_i, a_i^w) - r(x_i, a_i^\ell)] \}^2 \\ &\quad + \frac{1}{N} [\|\xi^*\|_1 + 2 \log \left( \frac{|\mathcal{N}_\epsilon(\mathcal{R})|}{\delta} \right)] \\ &\stackrel{(h)}{\leq} - \frac{1}{2N} \sum_{i=1}^N \{ \mathbb{E}_{\mathcal{D}} \{ \sigma[r^*(x_i, a_i^w) - r^*(x_i, a_i^\ell)] - \sigma[r(x_i, a_i^w) - r(x_i, a_i^\ell)] \}^2 - \frac{1}{2} |\xi_i^*| \} \\ &\quad + \frac{1}{N} [\|\xi^*\|_1 + 2 \log \left( \frac{|\mathcal{N}_\epsilon(\mathcal{R})|}{\delta} \right)] \\ &\stackrel{(i)}{\leq} - \frac{1}{2(3+eR)^2} \mathbb{E}_{\mathcal{D}} |r^*(x_1, a_1^w) - r^*(x_1, a_1^\ell) - r(x_1, a_1^w) + r(x_1, a_1^\ell)|^2 \end{aligned}$$

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

$$\begin{aligned} & + \frac{2}{N} \left[ \|\xi^*\|_1 + \log \left( \frac{|\mathcal{N}_\epsilon(\mathcal{R})|}{\delta} \right) \right] \\ & \stackrel{\underline{=}}{=} \frac{2}{N} \left[ \|\xi^*\|_1 + \log \left( \frac{|\mathcal{N}_\epsilon(\mathcal{R})|}{\delta} \right) \right] - \frac{E_r^2}{2(3 + e^{R r})^2}, \end{aligned} \quad (43)$$

where (a) uses Eq. [\(32\)](#page-13-4) from Lemma [4,](#page-13-0) (b) uses Eq. [\(42\)](#page-15-0) and σ(R) ≤ λ ≤ 1, (c) denotes <sup>E</sup><sup>D</sup> as the expectation under Assumption [1](#page-2-3) and (c) holds for finitely many r ∈ Nϵ(R) simultaneously with probability at least 1 − δ (by Lemma [7](#page-14-2) with λ ′ = 1), (d) uses Eq. [\(42\)](#page-15-0) and Assumption [1,](#page-2-3) (e) uses log v ≤ v − 1 for any v > 0, (f) uses Lemma 12.2 of [\(Harsha,](#page-9-23) [2011\)](#page-9-23), (g) uses Eq. [\(41\)](#page-14-3), (h) uses Lemma [5,](#page-13-1) (i) uses Lemma [1](#page-11-4) as well as the fact that the N samples {x<sup>i</sup> , a<sup>w</sup> i , a<sup>ℓ</sup> i } N <sup>i</sup>=1 are i.i.d., (j) denotes <sup>E</sup><sup>r</sup> := q E<sup>D</sup> r <sup>∗</sup>(x1, a<sup>w</sup> 1 ) − r <sup>∗</sup>(x1, a<sup>ℓ</sup> 1 ) − r(x1, a<sup>w</sup> 1 ) + r(x1, a<sup>ℓ</sup> 1 ) .

We have proved that with probability at least 1 − δ, the event E := {Eq. [\(43\)](#page-16-0) holds for all r ∈ Nϵ(R) simultaneously} occurs. We will extend the range to any r ∈ R. By the definition of the ϵ cover Nϵ(R), there exists at least one r † ∈ Nϵ(R) such that ∥r † − r∥<sup>∞</sup> ≤ ϵ. Therefore,

$$\begin{aligned}
& |\mathcal{L}_{N,\lambda}(r, \xi_r) - \mathcal{L}_{N,\lambda}(r^\dagger, \xi_{r^\dagger})| \\
& \stackrel{(a)}{=} \left| \frac{1}{N} \sum_{i=1}^N \{ \log \sigma[r^\dagger(x_i, a_i^w) - r^\dagger(x_i, a_i^\ell) + \xi_{r^\dagger,i}] - \log \sigma[r(x_i, a_i^w) - r(x_i, a_i^\ell) + \xi_{r,i}] \} \right. \\
& \quad \left. + \frac{\lambda}{N} (\|\xi_r\|_1 - \|\xi_{r^\dagger}\|_1) \right| \\
& \stackrel{(b)}{\leq} \frac{1}{N} \sum_{i=1}^N \left[ |[r^\dagger(x_i, a_i^w) - r^\dagger(x_i, a_i^\ell) + \xi_{r^\dagger,i}] - [r(x_i, a_i^w) - r(x_i, a_i^\ell) + \xi_{r,i}]| + \lambda(|\xi_{r,i}| - |\xi_{r^\dagger,i}|) \right] \\
& \leq \frac{1}{N} \sum_{i=1}^N \left[ |r^\dagger(x_i, a_i^w) - r(x_i, a_i^w)| + |r(x_i, a_i^\ell) - r^\dagger(x_i, a_i^\ell)| + |\xi_{r^\dagger,i} - \xi_{r,i}| + \lambda(|\xi_{r,i} - \xi_{r^\dagger,i}|) \right] \\
& \stackrel{(c)}{\leq} \frac{1}{N} \sum_{i=1}^N \left[ |r^\dagger(x_i, a_i^w) - r(x_i, a_i^w)| + |r(x_i, a_i^\ell) - r^\dagger(x_i, a_i^\ell)| \right. \\
& \quad \left. + (\lambda + 1) |r(x_i, a_i^\ell) - r(x_i, a_i^w)| - [r^\dagger(x_i, a_i^\ell) - r^\dagger(x_i, a_i^w)] \right] \stackrel{(d)}{\leq} 6\epsilon,
\end{aligned} \tag{44}$$

where (a) uses the definition of LN,λ given by Eq. [\(8\)](#page-3-3), (b) uses triangle inequality and <sup>d</sup> dv [log σ(v)] = σ(−v) ∈ [0, 1] for any v ∈ R, (c) uses the property that ξr,i defined by Eq. [\(15\)](#page-4-1) is a 1-Lipschitz continuous function of r(x<sup>i</sup> , a<sup>ℓ</sup> i ) − r(x<sup>i</sup> , a<sup>w</sup> i ) (since max(·, 0) is 1-Lipschitz continuous), (d) uses ∥r † − r∥<sup>∞</sup> ≤ ϵ and λ ≤ 1. Under the event E, Eq. [\(43\)](#page-16-0) holds with r replaced by r <sup>+</sup>, which along with Eq. [\(44\)](#page-16-1) implies the following inequality.

$$\begin{aligned}
& \mathcal{L}_{N,\lambda}(r^*) - \mathcal{L}_{N,\lambda}(r, \xi_r) \\
& \leq [\mathcal{L}_{N,\lambda}(r^\dagger, \xi_{r^\dagger}) - \mathcal{L}_{N,\lambda}(r, \xi_r)] + [\mathcal{L}_{N,\lambda}(r^*, \xi^*) - \mathcal{L}_{N,\lambda}(r^\dagger, \xi_{r^\dagger})] \\
& \leq 6\epsilon + \frac{2}{N} \left[ \|\xi^*\|_1 + \log \left( \frac{|\mathcal{N}_\epsilon(\mathcal{R})|}{\delta} \right) \right] - \frac{E_{r^\dagger}^2}{2(3 + e^R)^2} \\
& = 6\epsilon + \frac{2}{N} \left[ \|\xi^*\|_1 + \log \left( \frac{|\mathcal{N}_\epsilon(\mathcal{R})|}{\delta} \right) \right] - \frac{E_{r^\dagger}^2 - E_r^2}{2(3 + e^R)^2} - \frac{E_r^2}{2(3 + e^R)^2} \\
& \stackrel{(a)}{\leq} 6\epsilon + \frac{2}{N} \left[ \|\xi^*\|_1 + \log \left( \frac{|\mathcal{N}_\epsilon(\mathcal{R})|}{\delta} \right) \right] + \frac{4R\epsilon}{(3 + e^R)^2} - \frac{E_r^2}{2(3 + e^R)^2} \\
& \stackrel{(b)}{\leq} 7\epsilon + \frac{2}{N} \left[ \|\xi^*\|_1 + \log \left( \frac{|\mathcal{N}_\epsilon(\mathcal{R})|}{\delta} \right) \right] - \frac{E_r^2}{2(3 + e^R)^2}, \tag{45}
\end{aligned}$$

which proves Eq. [\(39\)](#page-14-4). Here, (a) uses the following inequality and (b) uses (3 + e R) <sup>2</sup> > 6e <sup>R</sup> + e <sup>2</sup><sup>R</sup> > 6R + 2R = 8R.

$$\begin{aligned} & |E_{r^\dagger}^2 - E_r^2| \\ &= |\mathbb{E}_{\mathcal{D}} \{ [r^*(x_1, a_1^w) - r^*(x_1, a_1^\ell) - r^\dagger(x_1, a_1^w) + r^\dagger(x_1, a_1^\ell)]^2 \} \end{aligned}$$

938

954

956

958

971

974

976

978

$$\begin{aligned} & -\mathbb{E}_{\mathcal{D}} \left\{ \left[ r^*(x_1, a_1^w) - r^*(x_1, a_1^\ell) - r(x_1, a_1^w) + r(x_1, a_1^\ell) \right]^2 \right\} \Big| \\ &= \left| \mathbb{E}_{\mathcal{D}} \left\{ \left[ r(x_1, a_1^w) - r(x_1, a_1^\ell) - r^\dagger(x_1, a_1^w) + r^\dagger(x_1, a_1^\ell) \right] \right. \right. \\ & \left. \left. \left[ 2r^*(x_1, a_1^w) - 2r^*(x_1, a_1^\ell) - r^\dagger(x_1, a_1^w) + r^\dagger(x_1, a_1^\ell) - r(x_1, a_1^w) + r(x_1, a_1^\ell) \right] \right] \right\} \\ & \stackrel{(a)}{\leq} (2\epsilon)(4R) = 8R\epsilon, \end{aligned}$$

where (a) uses Assumption [2](#page-5-2) and ∥r † − r∥<sup>∞</sup> ≤ ϵ.

Lemma 9. *Fixing any* ϵ > 0*,* δ ∈ (0, 1)*, the online dataset* {x<sup>i</sup> , a<sup>w</sup> i , a<sup>ℓ</sup> i , yi} T <sup>i</sup>=1 *generated from Algorithm [2](#page-6-3) satisfies the following bound for all* t = 1, . . . , T *and* π ∈ Π<sup>R</sup> def <sup>=</sup> {π<sup>r</sup> : <sup>r</sup> ∈ R} *simultaneously with probability at least* <sup>1</sup> − <sup>δ</sup>*.*

$$\begin{aligned} & \sum_{i=1}^t \log \frac{\sigma [r^\pi(x_i, a_i^w) - r^\pi(x_i, a_i^\ell) + y_i \xi_i^\pi]}{\sigma [r^*(x_i, a_i^w) - r^*(x_i, a_i^\ell) + y_i \xi_i^*]} \\ & \leq 2 \log \left( \frac{T |\mathcal{N}_\epsilon(\mathcal{R})|}{\delta} \right) + 4t\epsilon + \sum_{i=1}^t \left\{ \frac{1}{4} |\xi_i^*| + \sigma(R) |\xi_i^\pi| \right. \\ & \left. - \frac{1}{2(3 + e^{R/2})^2} \mathbb{E}_{x \sim \rho, a^{(1)} \sim \pi_i(\cdot | x), a^{(-1)} \sim \pi_{\text{ref}}(\cdot | x)} [f_\pi^2(x, a^{(1)}, a^{(-1)}]] \right\}, \end{aligned}$$

*where the function* f<sup>π</sup> *is defined below and* Nϵ(R) *is a finite* ϵ*-cover of* R*, that is, for any* r ∈ R*, there exists* r † ∈ Nϵ(R) *satisfying* ∥r † − r∥<sup>∞</sup> ≤ ϵ*.*

$$f_\pi(x, a^{(1)}, a^{(-1)}) \stackrel{\text{def}}{=} r^*(x, a^{(1)}) - r^*(x, a^{(-1)}) - r^\pi(x, a^{(1)}) + r^\pi(x, a^{(-1)}), \quad (46)$$

*Proof.* Define the following function.

$$\begin{aligned} & q_{\pi, \xi_i}(y_i | x_i, a_i^{(1)}, a_i^{(-1)}) \\ & \stackrel{\text{def}}{=} \begin{cases} \sigma\left(\beta \log \frac{\pi(a_i^{(1)} | x_i)}{\pi_{\text{ref}}(a_i^{(1)} | x_i)} - \beta \log \frac{\pi(a_i^{(-1)} | x_i)}{\pi_{\text{ref}}(a_i^{(-1)} | x_i)} + \omega(|a_i^{(1)}| - |a_i^{(-1)}|) + \xi_i\right), & y_i = 1 \\ \sigma\left(\beta \log \frac{\pi(a_i^{(-1)} | x_i)}{\pi_{\text{ref}}(a_i^{(-1)} | x_i)} - \beta \log \frac{\pi(a_i^{(1)} | x_i)}{\pi_{\text{ref}}(a_i^{(1)} | x_i)} + \omega(|a_i^{(-1)}| - |a_i^{(1)}|) - \xi_i\right), & y_i = -1. \end{cases} \\ & = \sigma[r^{\pi}(x_i, a_i^w) - r^{\pi}(x_i, a_i^{\ell}) + y_i \xi_i], \end{aligned} \tag{47}$$

where the second = uses Eq. [\(16\)](#page-4-2) and merges the above two cases. The above qπ,ξ<sup>i</sup> (y<sup>i</sup> |x<sup>i</sup> , a (1) i , a (−1) i ) can be seen as a conditional probability of y<sup>i</sup> ∈ {−1, 1} since qπ,ξ<sup>i</sup> (1|x<sup>i</sup> , a (1) i , a (−1) i ) + qπ,ξ<sup>i</sup> (−1|x<sup>i</sup> , a (1) i , a (−1) i ) = 1.

Then define the following random variables for i = 1, . . . , T.

$$W_i(\pi) = \frac{1}{2} \log \frac{\sigma [r^\pi(x_i, a_i^{(1)}) - r^\pi(x_i, a_i^{(\ell)})]}{\sigma [r^*(x_i, a_i^{(1)}) - r^*(x_i, a_i^{(\ell)}) + y_i \xi_i^*]} = \frac{1}{2} \log \frac{q_{\pi,0}(y_i|x_i, a_i^{(1)}, a_i^{(-1)})}{p_{r^*, \xi_i^*}(y_i|x_i, a_i^{(1)}, a_i^{(-1)})}, \quad (48)$$

where pr,ξ<sup>i</sup> (y<sup>i</sup> |x<sup>i</sup> , a (1) i , a (−1) i ) is defined by Eq. [\(41\)](#page-14-3).

For any r ∈ R, there exists r † ∈ Nϵ(R) satisfying ∥r † − r∥<sup>∞</sup> ≤ ϵ, and thus we can temporarily denote r<sup>u</sup> = ur<sup>π</sup>r† + (1 − u)r π (u ∈ [0, 1]). Then we obtain that

$$\begin{aligned} & \left| \frac{d}{du} \log \sigma [r_u(x_i, a_i^w) - r_u(x_i, a_i^\ell)] \right| \\ &= \sigma [r_u(x_i, a_i^\ell) - r_u(x_i, a_i^w)] |r^{\pi_{r^\dagger}}(x_i, a_i^w) - r^{\pi_{r^\dagger}}(x_i, a_i^\ell) - r^\pi(x_i, a_i^w) + r^\pi(x_i, a_i^\ell)| \\ &\stackrel{(a)}{\leq} |r^\dagger(x_i, a_i^w) - r^\dagger(x_i, a_i^\ell) - r^\pi(x_i, a_i^w) + r^\pi(x_i, a_i^\ell)| \\ &\leq |r^\dagger(x_i, a_i^w) - r^\pi(x_i, a_i^w)| + |r^\pi(x_i, a_i^\ell) - r^\dagger(x_i, a_i^\ell)| \leq 2\epsilon, \end{aligned} \tag{49}$$

994

996

998

1000 1001 1002 1003 where (a) and (b) use Eq. [\(48\)](#page-17-0), (c) uses the above notation that r<sup>u</sup> = ur<sup>π</sup>r† + (1 − u)r π (u ∈ [0, 1]), and (d) uses Eq. [\(49\)](#page-17-1). Then based on Algorithm [2](#page-6-3) and Assumption [1,](#page-2-3) given (x<sup>i</sup> , a (1) i , a (−1) i ), the label y<sup>i</sup> is generated with probability distribution pr <sup>∗</sup>,ξ<sup>i</sup> (y<sup>i</sup> |x<sup>i</sup> , a (1) i , a (−1) i ) defined by Eq. [\(41\)](#page-14-3). Therefore, given any δ ∈ (0, 1) and ϵ > 0, by Lemma [7](#page-14-2) with λ ′ = 1, the following inequality holds for t = 1, . . . , T and finitely many π ′ ∈ Nϵ(R) simultaneously with probability at least 1 − δ.

1004 1005 1006

1007 1008 1009 where µ<sup>i</sup> denotes the distribution of the i-th online data sample (x<sup>i</sup> , a (−1) i , a (1) i , yi) generated by Algorithm [2.](#page-6-3) We further upper bound the above inequality as follows.

1014

1016

1019

1024

1026

1029

1034

1036

1039 1040 where (a) uses log v ≤ v − 1 for any v > 0, (b) uses Lemma 12.2 of [\(Harsha,](#page-9-23) [2011\)](#page-9-23), (c) uses Eqs. [\(41\)](#page-14-3) and [\(47\)](#page-17-2), (d) uses Eq. [\(29\)](#page-12-1) and Lemma [5,](#page-13-1) and (e) uses Assumption [2](#page-5-2) and Lemma [1.](#page-11-4) Combining Eqs. [\(50\)](#page-18-0) and [\(51\)](#page-18-1), we obtain the following inequality which holds for all t = 1, . . . , T and π ∈ Π simultaneously with probability at least 1 − δ.

where (a) uses Eq. [\(29\)](#page-12-1) and σ(x) ∈ (0, 1) for any x ∈ <sup>R</sup>. Therefore,

$$\begin{aligned} & |W_i(\pi_{r^\dagger}) - W_i(\pi)| \\ & \stackrel{(a)}{=} \frac{1}{2} \left[ \log q_{\pi_{r^\dagger},0}(y_i | x_i, a_i^{(1)}, a_i^{(-1)}) - \log q_{\pi,0}(y_i | x_i, a_i^{(1)}, a_i^{(-1)}) \right] \\ & \stackrel{(b)}{=} \frac{1}{2} \left| \log \sigma [r^{\pi_{r^\dagger}}(x_i, a_i^w) - r^{\pi_{r^\dagger}}(x_i, a_i^\ell)] - \log \sigma [r^\pi(x_i, a_i^w) - r^\pi(x_i, a_i^\ell)] \right| \\ & \stackrel{(c)}{=} \frac{1}{2} \left| \log \sigma [r_1(x_i, a_i^w) - r_1(x_i, a_i^\ell)] - \log \sigma [r_0(x_i, a_i^w) - r_0(x_i, a_i^\ell)] \right| \stackrel{(d)}{\leq} \epsilon, \end{aligned} \tag{50}$$

$$\sum_{i=1}^t W_i(\pi') \leq \log \left( \frac{|T| \mathcal{N}_\epsilon(\mathcal{R})||}{\delta} \right) + \sum_{i=1}^t \log \mathbb{E}_{\mu_i} [e^{W_i(\pi')}].$$

$$\begin{aligned} & \sum_{i=1}^t W_i(\pi') - \log \left( \frac{|\mathcal{N}_\epsilon(\mathcal{R})|}{\delta} \right) \\ & \leq \sum_{i=1}^t \log \mathbb{E}_{\mu_i} [e^{W_i(\pi')}] \\ & \stackrel{(48)}{=} \sum_{i=1}^t \log \mathbb{E}_{\mu_i} \left\{ \mathbb{E}_{y_i \sim p_{r^*, \xi_i^*}(\cdot | x_i, a_i^{(1)}, a_i^{(-1)})} \left[ \sqrt{\frac{q_{\pi', 0}(y_i | x_i, a_i^{(1)}, a_i^{(-1)})}{p_{r^*, \xi_i^*}(y_i | x_i, a_i^{(1)}, a_i^{(-1)})}} \right] x_i, a_i^{(1)}, a_i^{(-1)} \right] \right\} \\ & \stackrel{(a)}{\leq} \sum_{i=1}^t \mathbb{E}_{\mu_i} \left[ \sum_{y \in \{-1,1\}} \sqrt{q_{\pi', 0}(y | x_i, a_i^{(1)}, a_i^{(-1)}) p_{r^*, \xi_i^*}(y | x_i, a_i^{(1)}, a_i^{(-1)})} - 1 \right] \\ & = -\frac{1}{2} \sum_{i=1}^t \mathbb{E}_{\mu_i} \left[ \sum_{y \in \{-1,1\}} \left| \sqrt{q_{\pi', 0}(y | x_i, a_i^{(1)}, a_i^{(-1)})} - \sqrt{p_{r^*, \xi_i^*}(y | x_i, a_i^{(1)}, a_i^{(-1)})} \right|^2 \right] \\ & \stackrel{(b)}{\leq} -\frac{1}{8} \sum_{i=1}^t \mathbb{E}_{\mu_i} \left[ \sum_{y \in \{-1,1\}} |q_{\pi', 0}(y | x_i, a_i^{(1)}, a_i^{(-1)}) - p_{r^*, \xi_i^*}(y | x_i, a_i^{(1)}, a_i^{(-1)})|^2 \right] \\ & \stackrel{(c)}{=} -\frac{1}{4} \sum_{i=1}^t \mathbb{E}_{\mu_i} \left\{ \sigma[r^{\pi'}(x_i, a_i^w) - r^{\pi'}(x_i, a_i^\ell)] - \sigma[r^*(x_i, a_i^w) - r^*(x_i, a_i^\ell) + y_i \xi_i^*] \right\}^2 \\ & \stackrel{(d)}{\leq} -\frac{1}{4} \sum_{i=1}^t \left\{ \left[ \mathbb{E}_{\mu_i} [\sigma[r^{\pi_{r^*}}(x_i, a_i^w) - r^{\pi_{r^*}}(x_i, a_i^\ell)] - \sigma[r^{\pi'}(x_i, a_i^w) - r^{\pi'}(x_i, a_i^\ell)]]^2 \right] - \frac{1}{2} |\xi_i^*| \right\} \\ & \stackrel{(e)}{\leq} \frac{1}{8} \sum_{i=1}^t \left\{ |\xi_i^*| - \frac{2}{(3 + eR)^2} \mathbb{E}_{\mu_i} \left[ |r^{\pi_{r^*}}(x_i, a_i^w) - r^{\pi_{r^*}}(x_i, a_i^\ell) - r^{\pi'}(x_i, a_i^w) + r^{\pi'}(x_i, a_i^\ell)|^2 \right] \right\}, \end{aligned} \tag{51}$$

$$\sum_{i=1}^t W_i(\pi)$$

1054

1056

1059 where (a) uses Eq. [\(51\)](#page-18-1) (with π ′ replaced by π<sup>r</sup> † ) and Eq. [\(50\)](#page-18-0), (b) uses the following inequality and (3 + e R) <sup>2</sup> > 6e <sup>R</sup> + e <sup>2</sup><sup>R</sup> > 6R + 2R = 8R.

1074

1076

1079

$$\begin{aligned} &\leq \sum_{i=1}^t [W_i(\pi) - W_i(\pi_{r^\dagger})] + W_i(\pi_{r^\dagger}) \\ &\stackrel{(a)}{\leq} \frac{1}{8} \sum_{i=1}^t \left\{ |\zeta_i^*| - \frac{2}{(3 + e^R)^2} \mathbb{E}_{\mu_i} \left[ [r^{\pi_{r^*}}(x_i, a_i^w) - r^{\pi_{r^*}}(x_i, a_i^\ell) - r^{\pi_{r^\dagger}}(x_i, a_i^w) + r^{\pi_{r^\dagger}}(x_i, a_i^\ell)]^2 \right] \right\} \\ &\quad + \log \left( \frac{T|\mathcal{N}_\epsilon(\mathcal{R})|}{\delta} \right) + t\epsilon \\ &\stackrel{(b)}{\leq} \frac{1}{8} \sum_{i=1}^t \left\{ |\zeta_i^*| - \frac{2}{(3 + e^R)^2} \mathbb{E}_{\mu_i} \left[ [r^{\pi_{r^*}}(x_i, a_i^w) - r^{\pi_{r^*}}(x_i, a_i^\ell) - r^\pi(x_i, a_i^w) + r^\pi(x_i, a_i^\ell)]^2 \right] \right\} \\ &\quad + \log \left( \frac{T|\mathcal{N}_\epsilon(\mathcal{R})|}{\delta} \right) + 2t\epsilon, \end{aligned} \tag{52}$$

$$\begin{aligned}
& \left[ r^{\pi_{r^*}} (x_i, a_i^w) - r^{\pi_{r^*}} (x_i, a_i^\ell) - r^\pi (x_i, a_i^w) + r^\pi (x_i, a_i^\ell) \right]^2 \\
& - \left[ r^{\pi_{r^*}} (x_i, a_i^w) - r^{\pi_{r^*}} (x_i, a_i^\ell) - r^{\pi_{r^\dagger}} (x_i, a_i^w) + r^{\pi_{r^\dagger}} (x_i, a_i^\ell) \right]^2 \\
& = \left[ r^{\pi_{r^\dagger}} (x_i, a_i^w) - r^{\pi_{r^\dagger}} (x_i, a_i^\ell) - r^\pi (x_i, a_i^w) + r^\pi (x_i, a_i^\ell) \right] \\
& \left[ 2r^{\pi_{r^*}} (x_i, a_i^w) - 2r^{\pi_{r^*}} (x_i, a_i^\ell) - r^\pi (x_i, a_i^w) + r^\pi (x_i, a_i^\ell) - r^{\pi_{r^\dagger}} (x_i, a_i^w) + r^{\pi_{r^\dagger}} (x_i, a_i^\ell) \right] \\
& \stackrel{(a)}{=} \left[ r^\dagger (x_i, a_i^w) - r^\dagger (x_i, a_i^\ell) - r^\pi (x_i, a_i^w) + r^\pi (x_i, a_i^\ell) \right] \\
& \left[ 2r^{\pi_{r^*}} (x_i, a_i^w) - 2r^{\pi_{r^*}} (x_i, a_i^\ell) - r^\pi (x_i, a_i^w) + r^\pi (x_i, a_i^\ell) - r^{\pi_{r^\dagger}} (x_i, a_i^w) + r^{\pi_{r^\dagger}} (x_i, a_i^\ell) \right]
\end{aligned}$$

where (a) uses Eq. [\(29\)](#page-12-1), and (b) uses ∥r † − r∥<sup>∞</sup> ≤ ϵ and Lemma [3.](#page-12-0)

Finally, we conclude the proof as follows.

$$\begin{aligned} & \sum_{i=1}^t \log \frac{\sigma [r^\pi(x_i, a_i^w) - r^\pi(x_i, a_i^\ell) + y_i \xi_i^\pi]}{\sigma [r^*(x_i, a_i^w) - r^*(x_i, a_i^\ell) + y_i \xi_i^*]} \\ & \stackrel{(a)}{\leq} \sum_{i=1}^t \left[ \log \frac{\sigma [r^\pi(x_i, a_i^w) - r^\pi(x_i, a_i^\ell)]}{\sigma [r^*(x_i, a_i^w) - r^*(x_i, a_i^\ell) + y_i \xi_i^*]} + \sigma(R) |\xi_i^\pi| \right] \\ & \stackrel{(b)}{=} \sum_{i=1}^t [2W_i(\pi) + \sigma(R) |\xi_i^\pi|] \\ & \stackrel{(c)}{\leq} 2 \log \left( \frac{T|\mathcal{N}_\epsilon(\mathcal{R})|}{\delta} \right) + 4t\epsilon + \sum_{i=1}^t \left\{ \frac{1}{4} |\xi_i^*| + \sigma(R) |\xi_i^\pi| \right. \\ & \quad - \frac{1}{2(3+e^{R^*})^2} \mathbb{E}_{\mu_i} \left[ [r^{\pi_{r^*}}(x_i, a_i^w) - r^{\pi_{r^*}}(x_i, a_i^\ell) - r^\pi(x_i, a_i^w) + r^\pi(x_i, a_i^\ell)]^2 \right] \left. \right\} \\ & \stackrel{(d)}{=} 2 \log \left( \frac{T|\mathcal{N}_\epsilon(\mathcal{R})|}{\delta} \right) + 4t\epsilon + \sum_{i=1}^t \left\{ \frac{1}{4} |\xi_i^*| + \sigma(R) |\xi_i^\pi| \right. \\ & \quad - \frac{1}{2(3+e^{R^*})^2} \mathbb{E}_{\mu_i} \left[ [r^*(x_i, a_i^{(1)}) - r^*(x_i, a_i^{(-1)}) - r^\pi(x_i, a_i^{(1)}) + r^\pi(x_i, a_i^{(-1)})]^2 \right] \left. \right\} \\ & \stackrel{(e)}{=} 2 \log \left( \frac{T|\mathcal{N}_\epsilon(\mathcal{R})|}{\delta} \right) + 4t\epsilon + \sum_{i=1}^t \left\{ \frac{1}{4} |\xi_i^*| + \sigma(R) |\xi_i^\pi| \right. \\ & \quad - \frac{1}{2(3+e^{R^*})^2} \mathbb{E}_{x \sim \rho, a^{(1)} \sim \pi_i(\cdot|x), a^{(-1)} \sim \pi_{\text{ref}}(\cdot|x)} [f_\pi^2(x, a^{(1)}, a^{(-1)})] \left. \right\}, \end{aligned}$$

1104

1106

1109

1111

1114

1116

1118 1119

1124

1126

1129

1134

1136

1151

where (a) uses Eq. [\(33\)](#page-13-5) from Lemma [4,](#page-13-0) (b) uses Wi(π) defined by Eq. [\(48\)](#page-17-0), (c) uses Eq. [\(52\)](#page-19-0), (d) uses Eq. [\(29\)](#page-12-1) and {a w i , a<sup>ℓ</sup> i } = {a (1) i , a (−1) i } (based on Assumption [1\)](#page-2-3), and (e) uses Eq. [\(46\)](#page-17-3).

Lemma 10 (Azuma-Hoeffding Inequality [\(Xie et al.,](#page-10-12) [2024\)](#page-10-12)). *The random variables* {Xt} T <sup>t</sup>=1 *satisfy* |Xt| ≤ C *almost surely. Then with probability at least* 1 − δ*, we have*

$$\left| \sum_{t=1}^T [X_t - \mathbb{E}(X_t|X_1, \dots, X_{t-1})] \right| \leq C \sqrt{8T \log(2/\delta)}. \quad (53)$$

Lemma 11. *Fixing any* ϵ > 0*,* δ ∈ (0, 1)*, the online dataset* {x<sup>i</sup> , a (1) i , a (−1) i , yi} T <sup>i</sup>=1 *generated from Algorithm [2](#page-6-3) satisfies the following inequality for all* t = 1, . . . , T *and* π ∈ Π<sup>R</sup> def <sup>=</sup> {π<sup>r</sup> : <sup>r</sup> ∈ R} *simultaneously with probability at least* <sup>1</sup> − <sup>δ</sup>*.*

$$\left| \left[ \sum_{i=1}^t \log \frac{\pi(a_i^{(-1)} |x_i|)}{\pi_{r^*}(a_i^{(-1)} |x_i|)} \right] - t \mathbb{E}_{x \sim \rho, a \sim \pi_{\text{ref}}(\cdot | x)} \left[ \log \frac{\pi(a | x)}{\pi_{r^*}(a | x)} \right] \right| \leq \frac{4R}{\beta} \sqrt{2t \log \left[ \frac{2T \mathcal{N}_\epsilon(\mathcal{R})}{\delta} \right]} + \frac{4t\epsilon}{\beta}.$$

*Proof.* For any <sup>r</sup> ∈ R, denote <sup>X</sup>i(r) = log <sup>π</sup>r(<sup>a</sup> (−1) i |xi) πr<sup>∗</sup> (a (−1) i |xi) which satisfies |Xi(r)| ≤ <sup>2</sup><sup>R</sup> β based on Lemma [6](#page-13-7) and Assumption [2.](#page-5-2)

Then by applying Lemma [10](#page-20-0) to Xi(r) with union bound, we obtain the following inequality which holds for all t = 0, 1, . . . , T − 1 and r ′ ∈ Nϵ(R) simultaneously with probability at least 1 − δ.

$$\left| \sum_{i=1}^t [X_i(r') - \mathbb{E}_{\mu_i} X_i(r')] \right| \leq \frac{2R}{\beta} \sqrt{8t \log \left[ \frac{2T\mathcal{N}_\epsilon(\mathcal{R})}{\delta} \right]}. \quad (54)$$

where µ<sup>i</sup> denotes the distribution of the i-th online data sample (x<sup>i</sup> , a (−1) i , a (1) i , yi) generated by Algorithm [2.](#page-6-3)

For any r ∈ R, there exists r † ∈ Nϵ(R) satisfying ∥r † − r∥<sup>∞</sup> ≤ ϵ, so Lemma [6](#page-13-7) implies that

$$|X_i(r^\dagger) - X_i(r)| = \left| \log \frac{\pi_{r^\dagger}(a_i^{(-1)}|x_i|)}{\pi_r(a_i^{(-1)}|x_i|)} \right| \leq \frac{2\epsilon}{\beta}.$$

Therefore, if the above high probability event E := {Eq. [\(54\)](#page-20-1) holds for all r ′ ∈ Nϵ(R)} occurs, then the following inequality holds for any r ∈ R.

$$\left| \sum_{i=1}^t [X_i(r) - \mathbb{E}_{\mu_i} X_i(r)] \right| \leq \frac{2R}{\beta} \sqrt{8t \log \left[ \frac{2T\mathcal{N}_\epsilon(\mathcal{R})}{\delta} \right]} + \frac{4t\epsilon}{\beta}. \quad (55)$$

For any π ∈ Π<sup>R</sup> def <sup>=</sup> {π<sup>r</sup> : <sup>r</sup> ∈ R}, there exists <sup>r</sup> ∈ R satisfying <sup>π</sup> <sup>=</sup> <sup>π</sup>r. Then we have

$$X_i(r) = \log \frac{\pi(a_i^{(-1)} | x_i)}{\pi_{r^*}(a_i^{(-1)} | x_i)}.$$

and thus

$$\mathbb{E}_{\mu_i} X_i(r) = \mathbb{E}_{x_i \sim \rho, a_i^{(-1)} \sim \pi_{\text{ref}}(\cdot | x)} \left[ \log \frac{\pi(a_i^{(-1)} | x_i)}{\pi_{r^*}(a_i^{(-1)} | x_i)} \right] = \mathbb{E}_{x \sim \rho, a \sim \pi_{\text{ref}}(\cdot | x)} \left[ \log \frac{\pi(a | x)}{\pi_{r^*}(a | x)} \right].$$

Substituting the above two equalities into Eq. [\(55\)](#page-20-2) concludes the proof.

Lemma 12. *Suppose that the offline dataset* {x<sup>i</sup> , a<sup>w</sup> i , a<sup>ℓ</sup> i , yi} N <sup>i</sup>=1 *is generated from Assumption [1,](#page-2-3) and select the baseline policy* πbase *to be the distribution of* a w i *given* x<sup>i</sup> *. Then fixing any* ϵ > 0*,* δ ∈ (0, 1)*, the following inequality holds for all* π ∈ Π<sup>R</sup> def <sup>=</sup> {π<sup>r</sup> : <sup>r</sup> ∈ R} *simultaneously with probability at least* <sup>1</sup> − <sup>δ</sup>*.*

$$\left| \left[ \sum_{i=1}^N \log \frac{\pi(a_i^w | x_i)}{\pi_{r^*}(a_i^w | x_i)} \right] - N\mathbb{E}_{x \sim \rho, a \sim \pi_{\text{base}}(\cdot | x)} \left[ \log \frac{\pi(a | x)}{\pi_{r^*}(a | x)} \right] \right| \leq \frac{4R}{\beta} \sqrt{2N \log \left[ \frac{2N_\epsilon(\mathcal{R})}{\delta} \right]} + \frac{4N\epsilon}{\beta}.$$

1159 1160 1161

1164

1174

1176

1194

1196

1199 1200

1204

1206

*Proof.* The proof logic is the same as that of Lemma [11.](#page-20-3) The major difference is that the inequality here only has to hold for any π ∈ Π<sup>R</sup> while Lemma [11](#page-20-3) requires to hold also for t = 1, . . . , T. As a result, when applying Lemma [10](#page-20-0) with union bound, <sup>2</sup>TNϵ(R) δ in the proof of Lemma [11](#page-20-3) is replaced with <sup>2</sup>Nϵ(R) δ .

Lemma 13. *Define the following quantity.*

$$I_t \stackrel{\text{def}}{=} \frac{\left[ \mathbb{E}_{x \sim \rho, a(1) \sim \pi_{t+1}(\cdot | x), a(-1) \sim \pi_{\text{ref}}(\cdot | x)} f_{\pi_{t+1}}(x, a^{(1)}, a^{(-1)}) \right]^2}{R^2 + \sum_{i=1}^t \mathbb{E}_{x \sim \rho, a(1) \sim \pi_i(\cdot | x), a(-1) \sim \pi_{\text{ref}}(\cdot | x)} \left[ f_{\pi_{t+1}}^2(x, a^{(1)}, a^{(-1)}) \right], \quad (56)$$

*where the function* f<sup>π</sup> *is defined by Eq.* [\(46\)](#page-17-3)*. Then we have*

$$\sum_{t=1}^T I_t \leq 12G_{\text{on}} \log(T + 2), \quad (57)$$

*where* Gon *is defined by Eq.* [\(26\)](#page-6-6)*.*

*Proof.* Applying Assumption [2](#page-5-2) and Lemma [3](#page-12-0) to the function f<sup>π</sup> defined by Eq. [\(46\)](#page-17-3), we have

$$f_\pi(x, a^{(1)}, a^{(-1)}) = r^*(x, a^{(1)}) - r^*(x, a^{(-1)}) - r^\pi(x, a^{(1)}) + r^\pi(x, a^{(-1)}) \in [-2R, 2R]. \quad (58)$$

Denote ν <sup>∗</sup> ∈ argminν∈Π<sup>R</sup> supx∈X ,a∈A,π∈Π<sup>R</sup> π(a|x) ν(a|x) as the policy used in the coverability coefficient [\(26\)](#page-6-6). Then we have

$$\pi(a^{(1)}|x) \leq G_{\text{On}}\nu^*(a^{(1)}|x), \quad \forall x \in \mathcal{X}, a^{(1)} \in \mathcal{A}, \pi \in \Pi_{\mathcal{R}}. \quad (59)$$

Then for each (x, a(1)) ∈ X × A, define the following quantity (min ∅ = +∞ by default)

$$\tau(x, a^{(1)}) = \min \left\{ t \geq 1 \mid \sum_{i=1}^t \pi_{i+1}(a^{(1)}|x) \geq G_{\text{on}} \nu^*(a^{(1)}|x) \right\}. \quad (60)$$

Hence,

$$\sum_{t=1}^T \pi_{t+1}(a^{(1)}|x) \mathbb{I}\{t \leq \tau(x, a^{(1)}) - 1\} < G_{\text{on}} \nu^*(a^{(1)}|x), \quad (61)$$

$$\sum_{i=1}^t \pi_i(a^{(1)}|x) \geq G_{\text{on}} \nu^*(a^{(1)}|x), \quad \forall t \geq \tau(x, a^{(1)}) + 1. \quad (62)$$

Then we conclude the proof as follows.

$$\begin{aligned} & \sum_{t=1}^T I_t \\ &= \sum_{t=1}^T \frac{\left[ \mathbb{E}_{x \sim \rho, a^{(1)} \sim \pi_{t+1}(\cdot|x), a^{(-1)} \sim \pi_{\text{ref}}(\cdot|x)} f_{\pi_{t+1}}(x, a^{(1)}, a^{(-1)}) \mathbb{I}\{t \leq \tau(x, a^{(1)})\} \right]^2}{R^2 + \sum_{i=1}^t \mathbb{E}_{x \sim \rho, a^{(1)} \sim \pi_i(\cdot|x), a^{(-1)} \sim \pi_{\text{ref}}(\cdot|x)} [f_{\pi_{t+1}}^2(x, a^{(1)}, a^{(-1)})]} \\ &+ \sum_{t=1}^T \frac{\left[ \mathbb{E}_{x \sim \rho, a^{(1)} \sim \pi_{t+1}(\cdot|x), a^{(-1)} \sim \pi_{\text{ref}}(\cdot|x)} f_{\pi_{t+1}}(x, a^{(1)}, a^{(-1)}) \mathbb{I}\{t \geq \tau(x, a^{(1)}) + 1\} \right]^2}{R^2 + \sum_{i=1}^t \mathbb{E}_{x \sim \rho, a^{(1)} \sim \pi_i(\cdot|x), a^{(-1)} \sim \pi_{\text{ref}}(\cdot|x)} [f_{\pi_{t+1}}^2(x, a^{(1)}, a^{(-1)})]} \\ &\stackrel{(a)}{\leq} \frac{1}{R^2} \sum_{t=1}^T (2R \mathbb{E}_{x \sim \rho, a^{(1)} \sim \pi_{t+1}(\cdot|x)} \mathbb{I}\{t \leq \tau(x, a^{(1)})\})^2 \\ &+ \sum_{t=1}^T \frac{\left[ \mathbb{E}_{x \sim \rho, a^{(1)} \sim \pi_t(\cdot|x), a^{(-1)} \sim \pi_{\text{ref}}(\cdot|x)} f_{\pi_{t+1}}(x, a^{(1)}, a^{(-1)}) \cdot \frac{\pi_{t+1}(a^{(1)}|x)}{\pi_t(a^{(1)}|x)} \mathbb{I}\{t \geq \tau(x, a^{(1)}) + 1\} \right]^2}{t \mathbb{E}_{x \sim \rho, a^{(1)} \sim \pi_t(\cdot|x), a^{(-1)} \sim \pi_{\text{ref}}(\cdot|x)} [f_{\pi_{t+1}}^2(x, a^{(1)}, a^{(-1)})]} \end{aligned}$$

1216

1218 1219

1224

1226

1229

1234

1236

1254

1256

1259

1260 This prove the first part of the theorem.

$$\begin{aligned} & \stackrel{(b)}{\leq} 4 \sum_{t=1}^T \mathbb{E}_{x \sim \rho, a^{(1)} \sim \pi_{t+1}(\cdot | x)} \mathbb{I}\{t \leq \tau(x, a^{(1)})\} \\ & + \sum_{t=1}^T \frac{1}{t} \mathbb{E}_{x \sim \rho, a^{(1)} \sim \bar{\pi}_t(\cdot | x)} \left[ \frac{\pi_{t+1}(a^{(1)}|x)}{\bar{\pi}_t(a^{(1)}|x)} \right]^2 \mathbb{I}\{t \geq \tau(x, a^{(1)}) + 1\} \\ & = 4 \sum_{x, a^{(1)}} \rho(x) \left[ \sum_{t=1}^T [\pi_{t+1}(a^{(1)}|x) \mathbb{I}\{t \leq \tau(x, a^{(1)}) - 1\}] + \sum_{t=1}^T [\pi_{t+1}(a^{(1)}|x) \mathbb{I}\{t = \tau(x, a^{(1)})\}] \right] \\ & + 2 \sum_{x, a^{(1)}} \rho(x) \sum_{t=1}^T \frac{\pi_{t+1}(a^{(1)}|x)}{t \bar{\pi}_t(a^{(1)}|x) + t \bar{\pi}_t(a^{(1)}|x)} [\pi_{t+1}(a^{(1)}|x) \mathbb{I}\{t \geq \tau(x, a^{(1)}) + 1\}] \\ & \stackrel{(c)}{\leq} 4 \sum_{x, a^{(1)}} \rho(x) [G_{\text{on}} \nu^*(a^{(1)}|x) + G_{\text{on}} \nu^*(a^{(1)}|x)] \\ & + 2 \sum_{x, a^{(1)}} \rho(x) \sum_{t=1}^T \frac{\pi_{t+1}(a^{(1)}|x)}{t \bar{\pi}_t(a^{(1)}|x) + G_{\text{on}} \nu^*(a^{(1)}|x)} [\pi_{t+1}(a^{(1)}|x) \mathbb{I}\{t \geq \tau(x, a^{(1)}) + 1\}] \\ & \stackrel{(d)}{\leq} 8 G_{\text{on}} \sum_{x, a^{(1)}} \rho(x) \nu^*(a^{(1)}|x) \\ & + 4 \sum_{x, a^{(1)}} \rho(x) \sum_{t=1}^T \log \left[ \frac{(t+1) \bar{\pi}_{t+1}(a^{(1)}|x) + G_{\text{on}} \nu^*(a^{(1)}|x)}{t \bar{\pi}_t(a^{(1)}|x) + G_{\text{on}} \nu^*(a^{(1)}|x)} \right] [G_{\text{on}} \nu^*(a^{(1)}|x)] \\ & = 8 G_{\text{on}} + 4 G_{\text{on}} \sum_{x, a^{(1)}} \rho(x) \nu^*(a^{(1)}|x) \log \left[ \frac{(T+1) \bar{\pi}_{T+1}(a^{(1)}|x) + G_{\text{on}} \nu^*(a^{(1)}|x)}{\bar{\pi}_1(a^{(1)}|x) + G_{\text{on}} \nu^*(a^{(1)}|x)} \right] \\ & \stackrel{(e)}{\leq} 8 G_{\text{on}} + 4 G_{\text{on}} \sum_{x, a^{(1)}} \rho(x) \nu^*(a^{(1)}|x) \log \left[ \frac{(T+1) G_{\text{on}} \nu^*(a^{(1)}|x) + G_{\text{on}} \nu^*(a^{(1)}|x)}{G_{\text{on}} \nu^*(a^{(1)}|x)} \right] \\ & \leq 12 G_{\text{on}} \log(T+2), \end{aligned}$$

where (a) denotes π<sup>t</sup> = 1 t P<sup>t</sup> <sup>i</sup>=1 π<sup>i</sup> and uses Eq. [\(58\)](#page-21-0) and (EX) <sup>2</sup> ≤ <sup>E</sup>(X<sup>2</sup> ) for any random variable X ∈ <sup>R</sup>, (b) uses Cauchy-Schwartz inequality, (c) uses Eqs. [\(59\)](#page-21-1), [\(61\)](#page-21-2) and Eq. [\(62\)](#page-21-3), (d) uses Eq. [\(59\)](#page-21-1) and the inequality that u ≤ 2 log(1 + u) for u = πt+1(a (1)|x) tπt(a(1)|x)+Gonν<sup>∗</sup>(a(1)|x) ∈ [0, 1] (u ∈ [0, 1] due to Eq. [\(59\)](#page-21-1)), (e) uses Eq. [\(59\)](#page-21-1).

# C. Proof of Proposition [1](#page-3-8)

(π, r, ξ) is the solution to the offline RLHF-COV objective [\(12\)](#page-3-5) means the following two conditions hold

$$\begin{aligned} \pi &\in \arg \max_{\pi' \in \Pi} \mathcal{L}_{N,\lambda}(r, \xi) + \eta V_{\beta, \omega}(\pi', r), \\ (r, \xi) &\in \arg \min_{r' \in \mathcal{R}, \xi' \in \mathbb{R}^N} \max_{\pi' \in \Pi} \mathcal{L}_{N,\lambda}(r', \xi') + \eta V_{\beta, \omega}(\pi', r'). \end{aligned}$$

Based on the notation that π<sup>r</sup> def = arg maxπ′∈ΠVβ,ω(π ′ , r), the above two conditions are equivalent to

$$\pi = \pi_r, \quad (r, \xi) \in \arg \min_{r' \in \mathcal{R}, \xi' \in \mathbb{R}^N} \mathcal{L}_{N, \lambda}(r', \xi') + \eta V_{\beta, \omega}(\pi_{r'}, r')$$

Furthermore, based on the notation that ξ<sup>r</sup> def = arg minξ∈R<sup>N</sup> <sup>L</sup>N,λ(r, ξ), the above two conditions are equivalent to

$$\pi = \pi_r, \quad \xi = \xi_r, \quad r = \arg \min_{r' \in \mathcal{R}} \mathcal{L}_{N,\lambda}(r', \xi_{r'}) + \eta V_{\beta,\omega}(\pi_{r'}, r'). \quad (63)$$

Next, we will obtain the analytical solutions of π<sup>r</sup> and ξr,i. We rewrite the function [\(11\)](#page-3-4) as follows.

$$V_{\beta,\omega}(\pi,r)$$

1269

1274

1276

1279

1289 1290

1294

1296

1306 1307

1309

1314

1316

$$\begin{aligned}
&= \mathbb{E}_{x \sim \rho, a \sim \pi(\cdot|x), a' \sim \pi_{\text{base}}(\cdot|x)} \left[ r(x, a) + \omega|a| - r(x, a') - \omega|a'| \right] - \beta \mathbb{E}_{x \sim \rho} \text{KL} \left[ \pi(\cdot|x) \parallel \pi_{\text{ref}}(\cdot|x) \right] \\
&= \mathbb{E}_{x \sim \rho, a \sim \pi(\cdot|x)} \left[ r(x, a) + \omega|a| - \beta \log \frac{\pi(a|x)}{\pi_{\text{ref}}(a|x)} \right] - \mathbb{E}_{x \sim \rho, a' \sim \pi_{\text{base}}(\cdot|x)} \left[ r(x, a') + \omega|a'| \right] \\
&= -\beta \mathbb{E}_{x \sim \rho, a \sim \pi(\cdot|x)} \left[ \log \frac{\pi(a|x)/Z_r(x)}{\pi_{\text{ref}}(a|x) \exp \left[ [r(x, a) + \omega|a|]/\beta \right] / Z_r(x) \right] \\
&\quad - \mathbb{E}_{x \sim \rho, a' \sim \pi_{\text{base}}(\cdot|x)} \left[ r(x, a') + \omega|a'| \right] \\
&= C - \beta \mathbb{E}_{x \sim \rho} \text{KL} \left[ \pi(\cdot|x) \parallel \pi_{\text{ref}}(\cdot|x) \exp \left[ [r(x, \cdot) + \omega|\cdot|]/\beta \right] / Z_r(x) \right],
\end{aligned}$$

where Zr(x) def = P <sup>a</sup>′∈A πref(a ′ |x) exp r(x,a′ )−ω|a ′ β and the constant C = βEx∼<sup>ρ</sup> log Zr(x) − <sup>E</sup>x∼ρ,a′∼πbase(·|x) r(x, a′ ) + ω|a ′ is independent of π. Therefore, π<sup>r</sup> def = arg maxπ′∈ΠVβ,ω(π ′ , r) should minimize the above KL term, which gives the analytical solution [\(14\)](#page-3-7).

Note that the log-likelihood function [\(8\)](#page-3-3) can be rewritten as follows.

$$\mathcal{L}_{N,\lambda}(r, \xi) \stackrel{\text{def}}{=} \frac{1}{N} \sum_{i=1}^N f_i(\xi_i),$$

where fi(v) := λ|v| − log σ[r(x<sup>i</sup> , a<sup>w</sup> i ) − r(x<sup>i</sup> , a<sup>ℓ</sup> i ) + yiv]. Hence, ξ<sup>r</sup> ∈ arg minξLN,λ(r, ξ) is equivalent to the following condition:

$$\xi_{r,i} \in \arg \min_{v \in \mathbb{R}} f_i(v); i = 1, 2, \dots, N.$$

As f<sup>i</sup> is a convex function for λ > 0, the above optimality condition is equivalent to the following stationary condition.

$$0 \in \partial f_i(\xi_{r,i}) = \lambda \partial |\xi_{r,i}| + y_i \{ \sigma[r(x_i, a_i^w) - r(x_i, a_i^\ell) + y_i \xi_{r,i}] - 1 \}, \quad (64)$$

where ∂ denotes partial differential. Noticing that y<sup>i</sup> ∈ {−1, 1}, it can be easily verified that the above equation has unique solution ξr,i defined by Eq. [\(15\)](#page-4-1).

# D. Proof of Proposition [2](#page-4-5)

Note that

$$\xi_{\pi_r} \stackrel{(a)}{=} \xi_{r\pi_r} \stackrel{(b)}{=} \xi_r, \quad (65)$$

where (a) uses Eq. [\(17\)](#page-4-4) and (b) substitutes Eq. [\(29\)](#page-12-1) into Eq. [\(15\)](#page-4-1). Therefore, by using Lemma [3,](#page-12-0) Eq. [\(65\)](#page-23-1), and substituting Eq. [\(29\)](#page-12-1) into Eqs. [\(8\)](#page-3-3) and [\(11\)](#page-3-4), we obtain that

$$\mathcal{L}_{N,\lambda}(r^{\pi_r}, \xi^{\pi_r}) + \eta V_{\beta,\omega}(\pi_{r^\pi}, r^{\pi_r}) = \mathcal{L}_{N,\lambda}(r, \xi_r) + \eta V_{\beta,\omega}(\pi, r), \quad (66)$$

Since Π<sup>R</sup> def <sup>=</sup> {π<sup>r</sup> : <sup>r</sup> ∈ R}, the following two statements are equivalent.

(P1): π is optimal for the offline DPO-COV objective [\(18\)](#page-4-3), i.e.,

$$\pi \in \arg \min_{\pi' \in \Pi_{\mathcal{R}}} [\mathcal{L}_{N,\lambda}(r^{\pi'}, \xi^{\pi'}) + \eta V_{\beta,\omega}(\pi_{r^{\pi'}}, r^{\pi'})].$$

(P2): There exists r ∈ arg min<sup>r</sup> ′∈R[LN,λ(r πr′ , ξ<sup>π</sup>r′ ) + ηVβ,ω(π<sup>r</sup> πr′ , r<sup>π</sup>r′ )] such that π = πr.

This along with Eq. [\(66\)](#page-23-2) implies that (P2) is equivalent to the following statement.

(P3): There exists r ∈ arg min<sup>r</sup> ′∈R[LN,λ(r ′ , ξ<sup>r</sup> ′ ) + ηVβ,ω(π<sup>r</sup> ′ , r′ )] such that π = πr.

By Proposition [1,](#page-3-8) (P3) is equivalent to the following statement.

1326

1329

1334

1336

1344 1345 Then denote <sup>π</sup>e<sup>2</sup> <sup>∈</sup> arg maxπ′∈Πmin<sup>r</sup> ′∈R,ξ′∈R<sup>N</sup> -LN,λ(r ′ , ξ′ ) + ηVβ,ω(π ′ , r′ ) and we have

1354

1356

1369

So far, we have proved the equivalence among (P1)-(P4), so the first part of this proposition is correct which states that (P1) and (P4) are equivalent.

It remains to prove the second part of this proposition, i.e., to figure out ξ and r given π under the assumption that (P1)-(P4) hold. Note that based on the analytical solution [\(14\)](#page-3-7) of πr, π = π<sup>r</sup> required by (P2)-(P4) holds if and only if for any x ∈ X there exists Uπ(x) ∈ <sup>R</sup> such that r(x, ·) = r π (x, ·) + Uπ(x). In this case, we have

$$\xi \stackrel{(a)}{=} \xi_r \stackrel{(b)}{=} \xi_{r^\pi} \stackrel{(c)}{=} \xi^\pi,$$

where (a) uses (P4), (b) substitutes r(x, ·) = r π (x, ·) + Uπ(x) into Eq. [\(16\)](#page-4-2), (c) uses ξ <sup>π</sup> def <sup>=</sup> <sup>ξ</sup>r<sup>π</sup> .

# E. Proof of Proposition [3](#page-6-5)

The proof logic is exactly the same as that of Proposition [2,](#page-4-5) with η replaced by −η.

### F. Proof of Theorem [1](#page-5-4)

Obtain <sup>π</sup>e <sup>∈</sup> arg minπ∈Π<sup>R</sup> -LN,λ(r π , ξ<sup>π</sup> ) + ηVβ,ω(πr<sup>π</sup> , r<sup>π</sup> ) by minimizing the offline DPO-COV objective [\(18\)](#page-4-3). Then based on Proposition [\(2\)](#page-4-5), there exists <sup>r</sup>e ∈ R such that (π, e r, ξ e πe ) (ξ πe is defined by Eq. [\(17\)](#page-4-4)) is the optimal solution to the offline RLHF-COV objective [\(12\)](#page-3-5), that is,

$$(\tilde{r}, \tilde{\xi}^{\tilde{\pi}}) \in \arg \min_{r' \in \mathcal{R}, \xi' \in \mathbb{R}^N} \max_{\pi' \in \Pi} [\mathcal{L}_{N,\lambda}(r', \xi') + \eta V_{\beta, \omega}(\pi', r')], \quad (67)$$

$$\tilde{\pi} = \pi_{\tilde{\tau}} \in \arg \max_{\pi' \in \Pi} V_{\beta, \omega}(\pi', \tilde{\tau}). \quad (68)$$

$$\begin{aligned} & \mathcal{L}_{N,\lambda}(\tilde{r}, \xi^{\tilde{\pi}}) + \eta V_{\beta,\omega}(\tilde{\pi}_2, \tilde{r}) \\ & \geq \min_{r' \in \mathcal{R}, \xi' \in \mathbb{R}^N} [\mathcal{L}_{N,\lambda}(r', \xi') + \eta V_{\beta,\omega}(\tilde{\pi}_2, r')] \\ & \stackrel{(a)}{=} \max_{\pi' \in \Pi} \min_{r' \in \mathcal{R}, \xi' \in \mathbb{R}^N} [\mathcal{L}_{N,\lambda}(r', \xi') + \eta V_{\beta,\omega}(\pi', r')] \\ & \stackrel{(b)}{=} \min_{r' \in \mathcal{R}, \xi' \in \mathbb{R}^N} \max_{\pi' \in \Pi} [\mathcal{L}_{N,\lambda}(r', \xi') + \eta V_{\beta,\omega}(\pi', r')] \\ & \stackrel{(c)}{=} \max_{\pi' \in \Pi} [\mathcal{L}_{N,\lambda}(\tilde{r}, \xi^{\tilde{\pi}}) + \eta V_{\beta,\omega}(\pi', \tilde{r})], \end{aligned} \tag{69}$$

where (a) uses <sup>π</sup>e<sup>2</sup> <sup>∈</sup> arg maxπ′∈Πmin<sup>r</sup> ′∈R,ξ′∈R<sup>N</sup> -LN,λ(r ′ , ξ′ )+ηVβ,ω(π ′ , r′ ) , (b) applies the minimax theorem (Theorem 1 of [\(Fan,](#page-8-20) [1953\)](#page-8-20)) to the function LN,λ(r ′ , ξ′ ) + ηVβ,ω(π ′ , r′ ) (defined by Eqs. [\(8\)](#page-3-3) and [\(11\)](#page-3-4)) which is a concave function of π ′ ∈ Π and a convex function of (r ′ , ξ′ ) ∈ R × <sup>R</sup> d , and (c) uses Eq. [\(67\)](#page-24-2). The above inequality implies that <sup>π</sup>e<sup>2</sup> <sup>∈</sup> maxπ′∈<sup>Π</sup> <sup>V</sup>β,ω(<sup>π</sup> ′ , <sup>r</sup>e) and thus <sup>π</sup>e<sup>2</sup> <sup>=</sup> <sup>π</sup>r<sup>e</sup> [\(68\)](#page-24-3) <sup>=</sup> <sup>π</sup>e. This means

$$\tilde{\pi} = \tilde{\pi}_2 \in \arg \max_{\pi' \in \Pi_{r''}} \min_{r'' \in \mathcal{R}, \xi' \in \mathbb{R}^N} [\mathcal{L}_{N, \lambda}(r', \xi') + \eta V_{\beta, \omega}(\pi', r')]. \quad (71)$$

Note that for any π ∈ Π, Eqs. [\(11\)](#page-3-4), [\(20\)](#page-5-5) imply that

$$J_{\beta,\omega}(\pi) - J_{\beta,\omega}(\tilde{\pi}) = V_{\beta,\omega}(\pi) - V_{\beta,\omega}(\tilde{\pi}). \quad (72)$$

Hence, π<sup>r</sup> <sup>∗</sup> ∈ arg maxπ∈ΠVβ,ω(π) also satisfies

$$\pi_{r^*} \in \arg \max_{\pi \in \Pi} J_{\beta, \omega}(\pi). \quad (73)$$

Finally, we prove the generalization error rate [\(22\)](#page-5-3) as follows.

$$\max_{\pi \in \Pi} J_{\beta, \omega}(\pi) - J_{\beta, \omega}(\tilde{\pi})$$

1379

1389 1390

1394

1396

1399 1400 1401

1402 The update rule [\(25\)](#page-6-2) implies that

$$\begin{aligned}
&\stackrel{(a)}{=} V_{\beta,\omega}(\pi_{r^*}, r^*) - \eta^{-1} \max_{\pi \in \Pi} \min_{r \in \mathcal{R}, \xi \in \mathbb{R}^N} [\mathcal{L}_{N,\lambda}(r, \xi) + \eta V_{\beta,\omega}(\pi, r)] \\
&\quad + \eta^{-1} \min_{r \in \mathcal{R}, \xi \in \mathbb{R}^N} [\mathcal{L}_{N,\lambda}(r, \xi) + \eta V_{\beta,\omega}(\tilde{\pi}, r)] - V_{\beta,\omega}(\tilde{\pi}, r^*) \\
&\stackrel{(b)}{\leq} V_{\beta,\omega}(\pi_{r^*}, r^*) - \eta^{-1} \min_{r \in \mathcal{R}} [\mathcal{L}_{N,\lambda}(r, \xi_r) + \eta V_{\beta,\omega}(\pi_{r^*}, r)] \\
&\quad + \eta^{-1} [\mathcal{L}_{N,\lambda}(r^*, \xi^*) + \eta V_{\beta,\omega}(\tilde{\pi}, r^*)] - V_{\beta,\omega}(\tilde{\pi}, r^*) \\
&\stackrel{(c)}{=} \max_{r \in \mathcal{R}} \left\{ \mathbb{E}_{x \sim \rho, a \sim \pi_{r^*}(\cdot|x), a' \sim \pi_{\text{base}}(\cdot|x)} [r^*(x, a) - r^*(x, a') - r(x, a) + r(x, a')] \right. \\
&\quad \left. + \eta^{-1} [\mathcal{L}_{N,\lambda}(r^*, \xi^*) - \mathcal{L}_{N,\lambda}(r, \xi_r)] \right\} \\
&\stackrel{(d)}{\leq} \max_{r \in \mathcal{R}} \left\{ G_{\mathcal{D}} E_r + \frac{2}{N\eta} \left[ \|\xi^*\|_1 + \log \left( \frac{|\mathcal{N}_{1/N}(\mathcal{R})|}{\delta} \right) \right] - \frac{E_r^2}{2\eta(3+e^R)^2} + \frac{7}{N\eta} \right\} \\
&\stackrel{(e)}{\leq} \frac{2}{N\eta} \left[ \|\xi^*\|_1 + 5 \log \left( \frac{|\mathcal{N}_{1/N}(\mathcal{R})|}{\delta} \right) \right] + \frac{\eta G_{\mathcal{D}}^2}{2} (3 + e^R)^2 \\
&\stackrel{(f)}{\leq} \frac{(G_{\mathcal{D}}^2 + 1)(3 + e^R)}{\sqrt{N}} \sqrt{\|\xi^*\|_1 + 5 \log[|\mathcal{N}_{1/N}(\mathcal{R})|/\delta]}, \tag{74}
\end{aligned}$$

where (a) uses Eqs. [\(71\)](#page-24-4), [\(72\)](#page-24-5) and [\(73\)](#page-24-6), (b) uses ξ<sup>r</sup> ∈ arg minξ∈R<sup>N</sup> LN,λ(r, ξ) as well as r <sup>∗</sup> ∈ R in Assumption q [2,](#page-5-2) (c) uses Eq. [\(11\)](#page-3-4), (d) uses Assumption [3](#page-5-1) and Lemma [8](#page-14-0) with ϵ = 1/N and E<sup>r</sup> := E<sup>D</sup> r <sup>∗</sup>(x1, a<sup>w</sup> 1 ) − r <sup>∗</sup>(x1, a<sup>ℓ</sup> 1 ) − r(x1, a<sup>w</sup> 1 ) + r(x1, a<sup>ℓ</sup> 1 ) 2 , (e) uses 1 ≤ log[|N1/N (R)|/δ] as well as bE − aE<sup>2</sup> ≤ b 4a for any a > 0 and b, E ∈ R, (f) uses η = √ <sup>∗</sup>∥1+5 log[|N1/N (R)|/δ] √ N(3+e<sup>R</sup>) .

# G. Proof of Theorem [2](#page-6-4)

$$\begin{aligned} 0 &\leq t\phi_t(\pi_{r^*}) - t\phi_t(\pi_{t+1}) \\ &\stackrel{(a)}{=} \sum_{i=1}^t \left\{ \lambda(|\xi_{r^*,i}| - |\xi_i^{\pi_{t+1}}|) + \beta\eta \log \frac{\pi_{r^*}(a_i^{(-1)}|x_i)}{\pi_{t+1}(a_i^{(-1)}|x_i)} \right. \\ &\quad \left. + \log \frac{\sigma[r^{\pi_{t+1}}(x_i, a_i^w) - r^{\pi_{t+1}}(x_i, a_i^\ell) + y_i\xi_i^{\pi_{t+1}}]}{\sigma[r^*(x_i, a_i^w) - r^*(x_i, a_i^\ell) + y_i\xi_{r^*,i}]} \right\} \\ &\stackrel{(b)}{\leq} \sum_{i=1}^t \left\{ \lambda(|\xi_i^*| - |\xi_i^{\pi_{t+1}}|) + \beta\eta \log \frac{\pi_{r^*}(a_i^{(-1)}|x_i)}{\pi_{t+1}(a_i^{(-1)}|x_i)} \right. \\ &\quad \left. + \log \frac{\sigma[r^{\pi_{t+1}}(x_i, a_i^w) - r^{\pi_{t+1}}(x_i, a_i^\ell) + y_i\xi_i^{\pi_{t+1}}]}{\sigma[r^*(x_i, a_i^w) - r^*(x_i, a_i^\ell) + y_i\xi_i^*]} \right\}, \end{aligned} \tag{75}$$

where (a) uses Eq. [\(16\)](#page-4-2), ξ πr<sup>∗</sup> <sup>i</sup> = ξ<sup>r</sup> <sup>∗</sup>,i (by Eq. [\(65\)](#page-23-1)), and Lemma [2](#page-12-2) (with r replaced by r ∗ ) and (b) uses the fact that ξr <sup>∗</sup>,i ∈ arg minξi∈<sup>R</sup> λ|ξ<sup>i</sup> | − log σ[r ∗ (x<sup>i</sup> , a<sup>w</sup> i ) − r ∗ (x<sup>i</sup> , a<sup>ℓ</sup> i ) + yiξ<sup>i</sup> , the i-th component of Lt,λ(r ∗ , ξ) defined in Eq. [\(8\)](#page-3-3). Based on Lemmas [9](#page-17-4) and [11](#page-20-3) (both with δ replaced by δ/2 and π replaced by πt+1), the following two inequalities hold for t = 1, . . . , T simultaneously with probability at least 1 − δ.

$$\begin{aligned} & \sum_{i=1}^t \log \frac{\sigma[r^{\pi_{t+1}}(x_i, a_i^w) - r^{\pi_{t+1}}(x_i, a_i^\ell) + y_i \xi_i^{\pi_{t+1}}]}{\sigma[r^*(x_i, a_i^w) - r^*(x_i, a_i^\ell) + y_i \xi_i^*]} \\ & \leq 2 \log \left( \frac{2T|\mathcal{N}_\epsilon(\mathcal{R})|}{\delta} \right) + 4t\epsilon + \sum_{i=1}^t \left\{ \frac{1}{4} |\xi_i^*| + \sigma(R) |\xi_i^{\pi_{t+1}}| \right. \\ & \quad \left. - \frac{1}{2(3+eR)^2} \mathbb{E}_{x \sim \rho, a^{(1)} \sim \pi_i(\cdot | x), a^{(-1)} \sim \pi_{\text{ref}}(\cdot | x)} [f_{\pi_{t+1}}^2(x, a^{(1)}, a^{(-1)})] \right\}, \end{aligned} \quad (76)$$

$$\sum_{i=1}^t \log \frac{\pi_{r^*}(a_i^{(-1)}|x_i)}{\pi_{t+1}(a_i^{(-1)}|x_i)} \leq \frac{4R}{\beta} \sqrt{2t \log \left[ \frac{2TN_\epsilon(\mathcal{R})}{\delta} \right]} + \frac{4t\epsilon}{\beta} + t\mathbb{E}_{x \sim \rho, a \sim \pi_{\text{ref}}(\cdot|x)} \left[ \log \frac{\pi_{r^*}(a|x)}{\pi_{t+1}(a|x)} \right]. \quad (77)$$

Substituting Eqs. [\(76\)](#page-25-1) and [\(77\)](#page-26-0) into Eq. [\(75\)](#page-25-2), we obtain that

$$\begin{aligned} 0 \leq 4\eta R \sqrt{2t \log \left[ \frac{4T\mathcal{N}_\epsilon(\mathcal{R})}{\delta} \right]} + 4\eta\epsilon t + \beta\eta t \mathbb{E}_{x \sim \rho, a \sim \pi_{\text{ref}}(\cdot | x)} \left[ \log \frac{\pi_{r^*}(a|x)}{\pi_{t+1}(a|x)} \right] \\ + \lambda \sum_{i=1}^t (|\xi_i^*| - |\xi_i^{\pi_{t+1}}|) + 2 \log \left( \frac{2T|\mathcal{N}_\epsilon(\mathcal{R})|}{\delta} \right) + 4t\epsilon + \sum_{i=1}^t \left\{ \frac{1}{4} |\xi_i^*| + \sigma(R) |\xi_i^{\pi_{t+1}}| \right. \\ \left. - \frac{1}{2(3+e^R)^2} \mathbb{E}_{x \sim \rho, a^{(1)} \sim \pi_i(\cdot | x), a^{(-1)} \sim \pi_{\text{ref}}(\cdot | x)} [f_{\pi_{t+1}}^2(x, a^{(1)}, a^{(-1)})] \right\} \\ \stackrel{(a)}{\leq} 4\eta R \sqrt{2t \log \left[ \frac{4T\mathcal{N}_\epsilon(\mathcal{R})}{\delta} \right]} + 2 \log \left( \frac{2T|\mathcal{N}_\epsilon(\mathcal{R})|}{\delta} \right) + 4\eta\epsilon t + 4\epsilon t \\ - \beta\eta t \mathbb{E}_{x \sim \rho, a \sim \pi_{\text{ref}}(\cdot | x)} \left[ \log \frac{\pi_{t+1}(a|x)}{\pi_{r^*}(a|x)} \right] \\ + \sum_{i=1}^t \left\{ \frac{5}{4} |\xi_i^*| - \frac{1}{2(3+e^R)^2} \mathbb{E}_{x \sim \rho, a^{(1)} \sim \pi_i(\cdot | x), a^{(-1)} \sim \pi_{\text{ref}}(\cdot | x)} [f_{\pi_{t+1}}^2(x, a^{(1)}, a^{(-1)})] \right\}, \end{aligned} \quad (78)$$

where (a) uses λ ∈ [σ(R), 1]. Then, we have

$$\begin{aligned} & J_{\beta,\omega}(\pi_{r^*}) - J_{\beta,\omega}(\pi_{t+1}) \\ & \stackrel{(a)}{=} \mathbb{E}_{x \sim \rho, a \sim \pi_{r^*}(\cdot|x)} \left[ r^*(x, a) - \omega|a| - \beta \log \frac{\pi_{r^*}(a|x)}{\pi_{\text{ref}}(a|x)} \right] \\ & \quad - \mathbb{E}_{x \sim \rho, a \sim \pi_{t+1}(\cdot|x)} \left[ r^*(x, a) - \omega|a| - \beta \log \frac{\pi_{t+1}(a|x)}{\pi_{\text{ref}}(a|x)} \right] \\ & \stackrel{(b)}{=} \mathbb{E}_{x \sim \rho, a \sim \pi_{\text{ref}}(\cdot|x)} \left[ r^*(x, a) - \omega|a| - \beta \log \frac{\pi_{r^*}(a|x)}{\pi_{\text{ref}}(a|x)} \right] \\ & \quad - \mathbb{E}_{x \sim \rho, a \sim \pi_{t+1}(\cdot|x)} \left[ r^*(x, a) - \omega|a| - \beta \log \frac{\pi_{t+1}(a|x)}{\pi_{\text{ref}}(a|x)} \right] \\ & = \beta \mathbb{E}_{x \sim \rho, a \sim \pi_{\text{ref}}(\cdot|x)} \left[ \log \frac{\pi_{t+1}(a|x)}{\pi_{r^*}(a|x)} \right] + \mathbb{E}_{x \sim \rho, a \sim \pi_{t+1}(\cdot|x)} \left[ \omega|a| + \beta \log \frac{\pi_{t+1}(a|x)}{\pi_{\text{ref}}(a|x)} - r^*(x, a) \right] \\ & \quad - \mathbb{E}_{x \sim \rho, a \sim \pi_{\text{ref}}(\cdot|x)} \left[ \omega|a| + \beta \log \frac{\pi_{t+1}(a|x)}{\pi_{\text{ref}}(a|x)} - r^*(x, a) \right] \\ & \stackrel{(c)}{=} \beta \mathbb{E}_{x \sim \rho, a \sim \pi_{\text{ref}}(\cdot|x)} \left[ \log \frac{\pi_{t+1}(a|x)}{\pi_{r^*}(a|x)} \right] + \mathbb{E}_{x \sim \rho, a \sim \pi_{t+1}(\cdot|x)} \left[ r^{\pi_{t+1}}(x, a) - r^*(x, a) \right] \\ & \quad - \mathbb{E}_{x \sim \rho, a \sim \pi_{\text{ref}}(\cdot|x)} \left[ r^{\pi_{t+1}}(x, a) - r^*(x, a) \right] \\ & \stackrel{(d)}{=} \beta \mathbb{E}_{x \sim \rho, a \sim \pi_{\text{ref}}(\cdot|x)} \left[ \log \frac{\pi_{t+1}(a|x)}{\pi_{r^*}(a|x)} \right] - \mathbb{E}_{x \sim \rho, a^{(1)} \sim \pi_{t+1}(\cdot|x), a^{(-1)} \sim \pi_{\text{ref}}(\cdot|x)} [f_{\pi_{t+1}}(x, a^{(1)}, a^{(-1)})] \\ & \stackrel{(e)}{\leq} \beta \mathbb{E}_{x \sim \rho, a \sim \pi_{\text{ref}}(\cdot|x)} \left[ \log \frac{\pi_{t+1}(a|x)}{\pi_{r^*}(a|x)} \right] + \frac{\eta t}{2} (3 + e^R)^2 I_t \\ & \quad + \frac{1}{2\eta t (3 + e^R)^2} \left\{ R^2 + \sum_{i=1}^t \mathbb{E}_{x \sim \rho, a^{(1)} \sim \pi_i(\cdot|x), a^{(-1)} \sim \pi_{\text{ref}}(\cdot|x)} [f_{\pi_{t+1}}^2(x, a^{(1)}, a^{(-1)})] \right\} \\ & \stackrel{(f)}{\leq} \frac{\eta t}{2} (3 + e^R)^2 I_t + \frac{1}{2\eta t} + 4R \sqrt{\frac{2}{t} \log \left[ \frac{4T\mathcal{N}_\epsilon(\mathcal{R})}{\delta} \right]} + \frac{2}{\eta t} \log \left( \frac{2T|\mathcal{N}_\epsilon(\mathcal{R})|}{\delta} \right) \\ & \quad + 4\epsilon + \frac{4\epsilon}{\eta} + \frac{5}{4\eta t} \sum_{i=1}^t |\xi_i^*|, \end{aligned}$$

1485 1486 1487 where (a) uses Eq. [\(20\)](#page-5-5), (b) uses Eq. [\(14\)](#page-3-7) which implies that r ∗ (x, a) − ω|a| − β log <sup>π</sup>r<sup>∗</sup> (a|x) <sup>π</sup>ref (a|x) = β log Z<sup>r</sup> <sup>∗</sup> (x) does not rely on a, (c) uses Eqs. [\(16\)](#page-4-2), (d) uses Eq. [\(46\)](#page-17-3), (e) applies Cauchy-Schwartz inequality to Eq. [\(56\)](#page-21-4), (f) uses Eq. [\(78\)](#page-26-1) and 3 + e <sup>R</sup> > R > 0. Finally, we conclude the proof by averaging the above inequality over t ∈ {1, 2, . . . , T} as follows.

1504

1506

1509

1518 1519

1524

1526

1529

1534

1536

$$\begin{aligned} \mathbb{E}[J_{\beta,\omega}(\pi_{r^*}) - J_{\beta,\omega}(\pi_{\widehat{T}})] &= \frac{1}{T} \sum_{t=1}^T [J_{\beta,\omega}(\pi_{r^*}) - J_{\beta,\omega}(\pi_{t+1})] \\ &\stackrel{(a)}{\leq} 6\eta G_{\text{on}}(3 + e^R)^2 \log(T + 2) + \frac{3 \log T}{2\eta T} + 8R \sqrt{\frac{2}{T}} \log \left[ \frac{4T\mathcal{N}_\epsilon(\mathcal{R})}{\delta} \right] \\ &\quad + \frac{6 \log T}{T\eta} \log \left( \frac{2T|\mathcal{N}_\epsilon(\mathcal{R})|}{\delta} \right) + 4\epsilon + \frac{4\epsilon}{\eta} + \frac{15 \log T}{4T\eta} \sum_{i=1}^T |\xi_i^*| \\ &\stackrel{(b)}{\leq} 6(3 + e^R) \log(T + 2) \sqrt{\frac{G_{\text{on}}}{T}} \left[ \log \left( \frac{4T|\mathcal{N}_{1/T}(\mathcal{R})|}{\delta} \right) + \|\xi^*\|_1 \right] + \frac{3(3 + e^R)(\log T)\sqrt{G_{\text{on}}}}{2\sqrt{T} \log[2T\mathcal{N}_{1/T}(\mathcal{R})/\delta]} \\ &\quad + 8R \sqrt{\frac{2}{T}} \log \left[ \frac{4T\mathcal{N}_{1/T}(\mathcal{R})}{\delta} \right] + 6(3 + e^R)(\log T) \sqrt{\frac{G_{\text{on}}}{T}} \log \left( \frac{4T|\mathcal{N}_{1/T}(\mathcal{R})|}{\delta} \right) \\ &\quad + \frac{4}{T} + 4(3 + e^R) \sqrt{\frac{G_{\text{on}}}{T \log[2T\mathcal{N}_{1/T}(\mathcal{R})/\delta]}} + \frac{15(3 + e^R)(\log T)\sqrt{G_{\text{on}}}}{4\sqrt{T} \log[4T\mathcal{N}_{1/T}(\mathcal{R})/\delta] + T\|\xi^*\|_1} \|\xi^*\|_1 \\ &\stackrel{(c)}{\leq} (6 + 1.5 + 8\sqrt{2} + 6 + 4 + 4)(3 + e^R)(\log T) \sqrt{\frac{G_{\text{on}}}{T}} \left[ \log \left( \frac{4T|\mathcal{N}_{1/T}(\mathcal{R})|}{\delta} \right) + \|\xi^*\|_1 \right] \\ &\quad + \frac{15(3 + e^R)(\log T)\sqrt{G_{\text{on}}}}{4\sqrt{T} \log[4T\mathcal{N}_{1/T}(\mathcal{R})/\delta] + T\|\xi^*\|_1} \{ \log[4T\mathcal{N}_{1/T}(\mathcal{R})/\delta] + \|\xi^*\|_1 \} \\ &\leq 37(3 + e^R)(\log T) \sqrt{\frac{G_{\text{on}}}{T}} \left[ \log \left( \frac{4T|\mathcal{N}_{1/T}(\mathcal{R})|}{\delta} \right) + \|\xi^*\|_1 \right], \end{aligned}$$

where (a) uses P<sup>T</sup> t=1 <sup>t</sup> ≤ 1 + log T ≤ 3 log T, P<sup>T</sup> <sup>t</sup>=1 √ 1 t ≤ 2 √ T and Eq. [\(57\)](#page-21-5), (b) uses η = √ log[4TN1/T (R)/δ]+∥ξ <sup>∗</sup>∥<sup>1</sup> (3+e<sup>R</sup>) √ T Gon , ϵ = T , and (c) uses Gon ≥ 1 (by Eq. [\(26\)](#page-6-6)), R < 3 + e <sup>R</sup>, log(<sup>T</sup> + 2) <sup>≤</sup> 2 log <sup>T</sup> and log <sup>4</sup>T|N1/T (R)<sup>|</sup> δ ≥ log T ≥ 1.