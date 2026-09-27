# Temporal-Difference Variational Continual Learning

## Abstract

Anonymous Authors1 000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054 Machine Learning models in real-world applications must continuously learn new tasks to adapt to shifts in the data-generating distribution. Yet, for Continual Learning (CL), models often struggle to balance learning new tasks (plasticity) with retaining previous knowledge (memory stability). Consequently, they are susceptible to Catastrophic Forgetting, which degrades performance and undermines the reliability of deployed systems. In the Bayesian CL literature, variational methods tackle this challenge by employing a learning objective that recursively updates the posterior distribution while constraining it to stay close to its previous estimate. Nonetheless, we argue that these methods may be ineffective due to compounding approximation errors over successive recursions. To mitigate this, we propose new learning objectives that integrate the regularization effects of multiple previous posterior estimations, preventing individual errors from dominating future posterior updates and compounding over time. We reveal insightful connections between these objectives and Temporal-Difference methods, a popular learning mechanism in Reinforcement Learning and Neuroscience. Experiments on challenging CL benchmarks show that our approach effectively mitigates Catastrophic Forgetting, outperforming strong Variational CL methods.

## 1. Introduction

A fundamental aspect of robust Machine Learning (ML) models is to learn from non-stationary sequential data. In this scenario, two main properties are necessary: first, models must learn from new incoming data - potentially from a different task -– with satisfactory asymptotic performance and sample complexity. This capability is called plasticity.

1Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>.

PermutedMNIST-Hard Method Online MLE
Batch MLEVCL
VCL CoreSet N-Step TD-VCL
TD( **)-VCL**
1 2 3 4 5 6 7 8 9 10 Number of Observed Tasks 0.75 0.80 0.85 0.90 0.95 1.00 A
c c u r a c y

Figure 1. **Average accuracy across observed tasks in the** PermutedMNIST-Hard benchmark. The TD-VCL approach, proposed in this work, leads to a substantial improvement against standard VCL and non-variational approaches.

Second, they must retain the knowledge from previously learned tasks, known as memory stability. When this does not happen, and the performance of previous tasks degrades, the model suffers from Catastrophic Forgetting (Goodfellow et al., 2015; McCloskey & Cohen, 1989). These two properties are the central core of Continual Learning (CL) (Schlimmer & Fisher, 1986; Abraham & Robins, 2005), being strongly relevant for ML systems susceptible to test-time distributional shifts. Given the critical importance of this topic, extensive literature addresses the challenges of CL in traditional ML methods (Schlimmer & Fisher, 1986; Sutton & Whitehead, 1993; McCloskey & Cohen, 1989; French, 1999) and, more recently, for overparameterized models (Hadsell et al., 2020; Goodfellow et al., 2015; Serra et al., 2018). In this work, we focus on Bayesian CL methods, for two reasons. First, it provides a principled, self-consistent framework for learning in online or low-data regimes (Rainforth et al., 2024).

Second, Bayesian models express their own uncertainty over predictions, which is crucial for safety-critical applications
(Kendall & Gal, 2017) and for enabling principled data selection (Gal et al., 2017; Melo et al., 2024). Particularly, we investigate Variational Continual Learning (VCL) approaches (Nguyen et al., 2018). As detailed in 1 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109 Figure 2. **An intuitive illustration of how TD-VCL functions in comparison to vanilla VCL**. At each timestep t, a new task dataset Dt arrives. Both methods aim to learn variational parameters qt(θ) over a family of distributions Q that approximates the true posterior p(θ | D1:t) via minimizing the KL divergence DKL(qt(θ) || p(θ | D1:t)). VCL optimization (left) is only constrained by the most recent posterior, which compounds approximation errors from previous estimations and potentially deviates far from the true posterior. TD-VCL (right) is regularized by a sequence of past estimations, alleviating the impact of compounded errors.

Section 3, VCL identifies a recursive relationship between subsequent posterior distributions over tasks. A variational optimization objective then leverages this recursion, which regularizes the updated posterior to stay close to the very latest posterior approximation. Nevertheless, we argue that solely relying on a single previous posterior estimate for building up the next optimization target may be ineffective, as the approximation error propagates to the next update and compounds after successive recursions. If a particular estimation is especially poor, the error will be carried over to the next step entirely, which can dramatically degrade model's performance. In this work, we show that the same optimization objective can be represented as a function of a sequence of previous posterior estimates and task likelihoods. We thus propose a new Continual Learning objective, n-Step KL VCL, that explicitly regularizes the posterior update considering several past posterior approximations. By considering multiple previous estimates, the objective dilutes individual errors, allows correct posterior approximates to exert a corrective influence, and leverages a broader global context to the learning target, reducing the impact of compounding errors over time. Figure 2 illustrates the underlying mechanism. We further generalize this unbiased optimization target to a broader family of CL objectives, namely Temporal- Difference VCL, which constructs the learning target by prioritizing the most recent approximated posteriors. We reveal a link between the proposed objective and Temporal- Difference (TD) methods, a popular learning mechanism in Reinforcement Learning (Sutton, 1988) and Neuroscience
(Schultz et al., 1997). Furthermore, we show that TD-VCL represents a spectrum of learning objectives that range from vanilla VCL to n-Step KL VCL. Finally, we present experiments on several challenging and popular CL benchmarks, demonstrating that they outperform standard VCL (as shown in Figure 1), other VCL-based methods, and non-variational baselines, effectively alleviating Catastrophic Forgetting.

## 2. Related Work

Continual Learning has been studied throughout the past decades, both in Artificial Intelligence (Schlimmer & Fisher, 1986; Sutton & Whitehead, 1993; Ring, 1997) and in Neuroand Cognitive Sciences (Flesch et al., 2023; French, 1999; McCloskey & Cohen, 1989). More recently, the focus has shifted towards overparameterized models, such as deep neural networks (Hadsell et al., 2020; Goodfellow et al., 2015; Serra et al., 2018; Adel et al., 2020). Given their powerful predictive capabilities, recent literature approaches CL from a wide range of perspectives. For instance, by regularizing the optimization objective to account for old tasks (Kirkpatrick et al., 2016; Zenke et al., 2017; Chaudhry et al., 2018); by replaying an external memory composed by a set of previous tasks (Lopez-Paz & Ranzato, 2017; Bang et al., 2021; Rebuffi et al., 2016); or by modifying the optimization procedure or manipulating the estimated gradients (Zeng et al., 2018; Javed & White, 2019; Liu & Liu, 2022). We refer to Wang et al. for an extensive review of recent approaches. Our proposed method is placed between regularization-based and replay-based methods. Bayesian CL. In the Bayesian framework, prior methods 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 exploit the recursive relationship between subsequent posteriors that emerge from the Bayes' rule in the CL setting (Section 3). Since Bayesian inference is often intractable, they fundamentally differ in the design of approximated inference. We highlight works that learn posteriors via Laplace approximation (Ritter et al., 2018; Schwarz et al., 2018), sequential Bayesian Inference (Titsias et al., 2020; Pan et al., 2020), and Variational Inference (VI) (Nguyen et al., 2018; Loo et al., 2021). Our work and proposed method lies in the latter category. Variational Inference for CL. Variational Continual Learning (VCL) (Nguyen et al., 2018) introduced the idea of online VI for the Continual Learning setting. It leverages the Bayesian recursion of posteriors to build an optimization target for the next step's posterior based on the current one. Similarly, our work also optimizes a target based on previous approximated posteriors. On the other hand, rather than relying on a single past posterior estimation, it bootstraps on several previous estimations to prevent compounded errors. Nguyen et al. (2018) further incorporate an heuristic external replay buffer to prevent forgetting, requiring a twostep optimization. In contrast, our work only requires a single-step optimization as the replay mechanism naturally emerges from the learning objective. Other derivative works usually blend VCL with architectural and optimization improvements (Loo et al., 2020; 2021; Guimeng et al., 2022; Tseran, 2018; Ebrahimi et al., 2020; Thapa & Li, 2025) or different posterior modeling assumptions (Auddy et al., 2020; Yang et al., 2019; Ahn et al., 2019). We specifically highlight UCB (Ebrahimi et al., 2020), which adapts the learning rate according to the uncertainty of the Bayesian model, and UCL (Ahn et al., 2019), which introduces a different implementation for the VCL objective by proposing the notion of node-wise uncertainty. While their contribution are orthogonal to ours, we adopt UCB and UCL as comparison methods to further show that our proposed objective can also be combined with other variational methods and enhance their performance.

## 3. Preliminaries

Problem Statement. In the Continual Learning setting, a model learns from a streaming of tasks, which forms a nonstationary data distribution throughout time. More formally, we consider a task distribution T and represent each task t ∼ T as a set of pairs {(xt, yt)}
Nt, where Nt is the dataset size. At every timestep t1, the model receives a batch of data Dt for training. We evaluate the model in held-out test sets, considering all previously observed tasks.

1We represent each task with the index t, which also denotes the timestep in the sequence of tasks.

In the **Bayesian framework** for CL, we assume a prior distribution over parameters p(θ), and the goal is to learn a posterior distribution p(θ | D1:T ) after observing T tasks.

Crucially, given the sequential nature of tasks, we identify a recursive property of posteriors:

$$p(\boldsymbol{\theta}\mid\mathcal{D}_{1:T})\propto p(\boldsymbol{\theta})p(\mathcal{D}_{1:T}\mid\boldsymbol{\theta})\stackrel{{\text{i.i.d}}}{{=}}$$ $$p(\boldsymbol{\theta})\prod_{t=1}^{T}p(\mathcal{D}_{t}\mid\boldsymbol{\theta})\propto p(\boldsymbol{\theta}\mid\mathcal{D}_{1:T-1})p(\mathcal{D}_{T}\mid\boldsymbol{\theta}),\tag{1}$$

where we assume that tasks are i.i.d. Equation 1 shows that we may update the posterior estimation online, given the likelihood of the subsequent task.

Variational Continual Learning. Despite the elegant recursion, computing the posterior p(θ | D1:T ) exactly is often intractable, especially for large parameter spaces. Hence, we rely on an approximation. VCL achieves this by employing online variational inference (Ghahramani & Attias, 2000). It assumes the existence of variational parameters q(θ) whose goal is to approximate the posterior by minimizing the following KL divergence over a space of variational approximations Q:

$$q_{t}(\mathbf{\theta})=\operatorname*{arg\,min}_{q\in\mathcal{Q}}\mathcal{D}_{KL}(q(\mathbf{\theta})\mid\mid\frac{1}{Z_{t}}q_{t-1}(\mathbf{\theta})p(\mathcal{D}_{t}\mid\mathbf{\theta})),\tag{2}$$

where Zt represents a normalization constant. The objective in Equation 2 is equivalent to maximizing the variational lower bound of the online marginal likelihood:

$$\mathcal{L}_{VCL}^{t}(\mathbf{\theta})=\mathbb{E}_{\mathbf{\theta}\sim q_{t}(\mathbf{\theta})}[\log p(\mathcal{D}_{t}\mid\mathbf{\theta})]$$ $$-\mathscr{D}_{KL}(q_{t}(\mathbf{\theta})\mid\mid q_{t-1}(\mathbf{\theta})).\tag{3}$$

We can interpret the loss in Equation 3 through the lens of the stability-plasticity dilemma (Abraham & Robins, 2005). The first term maximizes the likelihood of the new task (encouraging plasticity), whereas the KL term penalizes parametrizations that deviate too far from the previous posterior estimation, which supposedly contains the knowledge from past tasks (encouraging memory stability).

## 4. Temporal-Difference Variational Continual Learning

Maximizing the objective in Equation 3 is equivalent to the optimization in Equation 2, but its computation relies on two main approximations. First, computing the expected log-likelihood term analytically is not tractable, which requires a Monte-Carlo (MC) approximation. Second, the KL term relies on a previous posterior estimate, which may be biased from previous approximation errors. While updating the posterior to account for the next task, these biases deviate the learning target from the true objective. Crucially, as Equation 3 solely relies on the very latest posterior estimation, the error compounds with successive recursive updates. Alternatively, we may represent the same objective as a function of several previous posterior estimations and alleviate the effect of the approximation error from any particular one. By considering several past estimates, the objective dilutes individual errors, allows correct posterior approximates to exert a corrective influence, and leverages a broader global context to the learning target, reducing the impact of compounding errors over time.

## 4.1. Variational Continual Learning With N-Step Kl Regularization

We start by presenting a new objective that is equivalent to Equation 2 while also meeting the aforementioned desiderata:
Proposition 4.1. The standard KL minimization objective in Variational Continual Learning (Equation 2) is equivalently represented as the following objective, where n ∈ N0 is a hyperparameter:
165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219

$$q_{t}(\mathbf{\theta})=\arg\max_{q\in\mathcal{Q}}\mathbb{E}_{\mathbf{\theta}\sim q_{t}(\mathbf{\theta})}\Big{[}\sum_{i=0}^{n-1}\frac{(n-i)}{n}\log p(\mathcal{D}_{t-i}\mid\mathbf{\theta})\Big{]}$$ $$-\sum_{i=0}^{n-1}\frac{1}{n}\mathcal{D}_{KL}(q_{t}(\mathbf{\theta})\mid\mid q_{t-i-1}(\mathbf{\theta})).\tag{4}$$

We present the proof of Proposition 4.1 in **Appendix** A. We name Equation 4 as the n-Step KL regularization objective. It represents the same learning target of Equation 2 as a sum of weighted likelihoods and KL terms that consider different posterior estimations, which can be interpreted as
"distributing" the role of regularization among them. For instance, if an estimate qt−i deviates too far from the true posterior, it only affects 1/n of the KL regularization term. The hyperparameter n assumes integer values up to t and defines how far in the past the learning target goes. If n is set to 1, we recover vanilla VCL. An interesting insight comes from the likelihood term. It contains the likelihood of different tasks, weighted by their recency. Hence, the idea of re-estimating old task likelihoods, commonly leveraged as a heuristic in CL methods, fundamentally emerges in the proposed objective. We may estimate these likelihood terms by replaying data from different tasks simultaneously, alleviating the violation of the i.i.d assumption that happens given the online, sequential nature of CL (Hadsell et al., 2020).

## 4.2. From N-Step Kl To Temporal-Difference Targets

The learning objective in Equation 4 relies on several different posterior estimates, alleviating the compounding error problem. A caveat is that all estimates have the same weight in the final objective. One may want to have more flexibility by giving different weights for them - for instance, amplifying the effect from the most recent estimate while drastically reducing the impact of previous ones. It is possible to accomplish that, as shown in the following proposition: Proposition 4.2. The standard KL minimization objective in VCL (Equation 2) is equivalently represented as the following objective, with n ∈ N0, and λ ∈ [0, 1) hyperparameters:

$$q_{t}(\theta)=$$
$$\arg\max_{q\in\mathcal{Q}}\mathbb{E}_{\boldsymbol{\theta}\sim q_{t}(\boldsymbol{\theta})}\Big{[}\sum_{i=0}^{n-1}\frac{\lambda^{i}(1-\lambda^{n-i})}{1-\lambda^{n}}\log p(\mathcal{D}_{t-i}\mid\boldsymbol{\theta})\Big{]}$$ $$-\sum_{i=0}^{n-1}\frac{\lambda^{i}(1-\lambda)}{1-\lambda^{n}}\mathcal{D}_{KL}(q_{t}(\boldsymbol{\theta})\mid\mid q_{t-i-1}(\boldsymbol{\theta})).\tag{5}$$

The proof is available in **Appendix** B. We call Equation 5 the TD(λ)-VCL objective2. It augments the n-Step KL Regularization to weight the regularization effect of different estimates in a way that geometrically decays - via the λ i term - as far as it goes in the past. Other λ-related terms serve as normalization constants. Equation 5 provides a more granular level of target control.

Interestingly, this objective relates intrinsically to the λreturns for Temporal-Difference (TD) learning in valuedbased reinforcement learning (Sutton & Barto, 2018). More broadly, both objectives of Equations 4 and 5 are compound updates that combine n-step Temporal-Difference targets, as shown below. First, we formally define a TD target in the CL context: Definition 4.3. For a timestep t, the n-Step Temporal- Difference target for Variational Continual Learning is defined as, ∀n ∈ N0, n ≤ t:

$$\mathrm{TD}_{t}(n)=\mathbb{E}_{\mathbf{\theta}\sim q_{t}(\mathbf{\theta})}\left[\sum_{i=0}^{n-1}\log p(\mathcal{D}_{t-i}\mid\mathbf{\theta})]\right]\\ -\mathcal{D}_{KL}(q_{t}(\mathbf{\theta})\mid\mid q_{t-n}(\mathbf{\theta})).\tag{6}$$

In **Appendix** C, we reveal the connection between Equation 6 and the TD targets employed in Reinforcement Learning, justifying the adopted terminology. From this definition, it follows that:
2We refer to both n-Step KL Regularization and TD(λ)-VCL
as TD-VCL objectives.

Proposition 4.4. ∀n ∈ N0, n ≤ t , the objective in Equation 2 *can be equivalently represented as:*

$$q_{t}(\mathbf{\theta})=\operatorname*{arg\,max}_{q\in\mathcal{Q}}\mathrm{TD}_{t}(n),$$
TDt(n), (7)
with TDt(n) as in Definition *4.3. Furthermore, the objective* in Equation 5 *can also be represented as:*

$$q_{t}(\mathbf{\theta})=\operatorname{arg\,max}_{q\in\mathcal{Q}}{\frac{1-\lambda}{1-\lambda^{n}}}\underbrace{\left[\sum_{k=0}^{n-1}\lambda^{k}\mathrm{TD}_{t}(k+1))\right]}_{\mathrm{Discounted~sum~of~TD~targets}}.$$
$$(8)$$

The proof is in **Appendix** D. Proposition 4.4 states that the TD(λ)-VCL objective is a sum of discounted TD targets (up to a normalization constant), effectively representing λ-returns. In parallel, one can show that the n-Step KL Regularization objective, as a particular case, is a simple average of n-Step TD targets. Fundamentally, the key idea behind these objectives is *bootstrapping*: they build a learning target estimate based on other estimates. Ultimately, the
"λ-target" in Equation 5 provides flexibility for bootstrapping by allowing multiple previous estimates to influence the objective. The TD-VCL objectives generalize a spectrum of Continual Learning algorithms. As a final remark, in **Appendix** E, we show that, based on the choice of hyperparameters, the TD(λ)-VCL objective forms a family of learning algorithms that span from Vanilla VCL to n-Step KL Regularization. Fundamentally, it mixes different targets of MC approximations for expected log-likelihood and KL regularization. This process is similar to how TD(λ) and n-step TD mix MC updates and TD predictions in Reinforcement Learning, effectively providing a mechanism to strike a balance between the variance from MC estimations and the bias from bootstrapping (Sutton & Barto, 2018).

## 5. Experiments And Discussion

Our central hypothesis is that for Bayesian CL, leveraging multiple past posterior estimates mitigates the impact of compounded errors inherent to the VCL objective, thus alleviating the problem of Catastrophic Forgetting. We now provide an experimental setup for validation. Specifically, we evaluate this hypothesis by analyzing the questions highlighted in Section 5.1.

Implementation. We use a Gaussian mean-field approximate posterior and assume a Gaussian prior N (0, σ2I),
and parameterize all distributions as deep networks. For all variational objectives, we compute the KL term analytically and employ Monte Carlo approximations for the expected 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274

$$(7)$$

log-likelihood terms, leveraging the reparametrization trick (Kingma & Welling, 2014) for computing gradients. We employed likelihood-tempering (Loo et al., 2021) to prevent variational over-pruning (Trippe & Turner, 2018). Lastly, for test-time evaluation, we compute the posterior predictive distribution by marginalizing out the approximated posterior via Monte-Carlo sampling. We provide further detail about architecture and training in Appendix F and our code3.

Comparison Methods. We compare TD-VCL and n-Step KL VCL against several methods. We first evaluate nonvariational naive methods for CL: **Online MLE** naively applies maximum likelihood estimation in the current task data. It serves as a lower bound for other methods, as well as a way to evaluate how challenging the benchmark is. **Batch** MLE applies maximum likelihood estimation considering a buffer of current and old task data. Next, we adopt the following variational methods for direct comparison in the Bayesian CL setting: VCL, introduced by Nguyen et al. (2018), optimizes the objective in Equation 3. VCL Core-
Set is a VCL variant that incorporates a replay set to mitigate any residual forgetting (Nguyen et al., 2018). UCL (Ahn et al., 2019) is another variational method that implements adaptive regularization based on the notion of node-wise uncertainty. Finally, UCB (Ebrahimi et al., 2020) also optimizes the objective of Equation 3 but adapts the learning rate for each parameter based on their uncertainty. Particularly for UCL and UCB, we compare them with the proposed TD-UCL and **TD-UCB**, which incorporate the introduced objective into UCL and UCB, respectively. Benchmarks. We evaluate five benchmarks for Continual Learning (CL). First, we introduce three new benchmarks: PermutedMNIST-Hard, **SplitMNIST-Hard**, and SplitNotMNIST-Hard. These are more challenging versions of traditional CL benchmarks with similar names. They are significantly harder due to two key restrictions. First, the amount of replay memory that any method can use is limited in both dataset size and the number of tasks. As empirically shown in Appendix H, this creates a much more acute scenario of Catastrophic Forgetting. Second, they enforce the adoption of single-head classifiers. As also shown in Appendix H, this requires the model to account for the potential negative transfer learning among tasks, which makes MNIST/NotMNIST-based benchmarks non-trivial for current research. Next, we also evaluate on two other popular CL benchmarks: **CIFAR100-10** and TinyImageNet-10. Both benchmarks are very challenging classification problems, particularly in our setting where no pre-trained representations are used. In Appendix I, we detail all benchmark tasks and specific constraints adopted for robust evaluation.

3https://anonymous.4open.science/r/
vcl-nstepkl-5707

| especially when the number of observed tasks increase.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | PermutedMNIST-Hard                                                                        |       |       |       |       |       |       |        |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|-------|-------|-------|-------|-------|-------|--------|
| t = 2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | t = 3                                                                                     | t = 4 | t = 5 | t = 6 | t = 7 | t = 8 | t = 9 | t = 10 |
| Online MLE                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 0.87±0.07 0.77±0.06 0.73±0.08 0.69±0.08 0.65±0.13 0.57±0.16 0.51±0.14 0.46±0.11 0.40±0.08 |       |       |       |       |       |       |        |
| Batch MLE                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | 0.95±0.01 0.93±0.01 0.88±0.04 0.83±0.04 0.77±0.10 0.71±0.13 0.64±0.12 0.57±0.11 0.51±0.06 |       |       |       |       |       |       |        |
| VCL                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 0.95±0.00 0.94±0.01 0.93±0.02 0.91±0.02 0.89±0.03 0.86±0.03 0.83±0.04 0.80±0.06 0.78±0.04 |       |       |       |       |       |       |        |
| VCL CoreSet                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 0.96±0.00 0.95±0.00 0.94±0.00 0.93±0.02 0.91±0.01 0.89±0.02 0.86±0.03 0.84±0.04 0.81±0.03 |       |       |       |       |       |       |        |
| n-Step TD-VCL 0.95±0.01 0.94±0.00 0.94±0.00 0.93±0.01 0.92±0.01 0.91±0.01 0.90±0.02 0.89±0.01 0.88±0.02 TD(λ)-VCL 0.97±0.00 0.96±0.00 0.95±0.00 0.94±0.01 0.93±0.01 0.92±0.01 0.91±0.01 0.90±0.01 0.89±0.02 SplitMNIST-Hard SplitNotMNIST-Hard t = 2 t = 3 t = 4 t = 5 t = 2 t = 3 t = 4 t = 5 Online MLE 0.86±0.02 0.61±0.03 0.75±0.04 0.57±0.06 0.72±0.02 0.61±0.05 0.61±0.00 0.51±0.04 Batch MLE 0.95±0.04 0.65±0.04 0.82±0.04 0.59±0.03 0.71±0.02 0.65±0.03 0.61±0.00 0.50±0.06 VCL 0.87±0.02 0.66±0.04 0.82±0.03 0.64±0.11 0.69±0.04 0.63±0.03 0.60±0.00 0.51±0.06 VCL CoreSet 0.93±0.04 0.68±0.07 0.84±0.04 0.62±0.03 0.69±0.04 0.65±0.02 0.60±0.01 0.51±0.07 n-Step TD-VCL 0.98±0.01 0.79±0.08 0.88±0.04 0.67±0.04 0.72±0.04 0.73±0.05 0.70±0.04 0.58±0.08 TD(λ)-VCL 0.98±0.01 0.81±0.07 0.89±0.03 0.66±0.02 0.74±0.02 0.73±0.03 0.69±0.03 0.58±0.09 |                                                                                           |       |       |       |       |       |       |        |

275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329

| consistently outperform the baselines in harder benchmarks with more complex architectures, such as Bayesian CNNs. CIFAR100-10 TinyImageNet-10 t = 2 t = 4 t = 6 t = 8 t = 10 t = 2 t = 4 t = 6 t = 8   | t = 10                                            |                                                   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|---------------------------------------------------|
| Online MLE                                                                                                                                                                                              | 0.56±0.05 0.57±0.06 0.56±0.03 0.53±0.06 0.52±0.04 | 0.48±0.03 0.45±0.02 0.44±0.01 0.45±0.02 0.44±0.03 |
| Batch MLE                                                                                                                                                                                               | 0.57±0.03 0.58±0.04 0.58±0.05 0.56±0.06 0.54±0.07 | 0.50±0.02 0.48±0.02 0.48±0.02 0.50±0.02 0.51±0.03 |
| VCL                                                                                                                                                                                                     | 0.64±0.02 0.63±0.02 0.60±0.02 0.61±0.05 0.66±0.01 | 0.53±0.06 0.51±0.03 0.51±0.03 0.51±0.02 0.51±0.02 |
| VCL CoreSet                                                                                                                                                                                             | 0.64±0.05 0.63±0.03 0.63±0.02 0.61±0.02 0.65±0.02 | 0.52±0.03 0.51±0.02 0.51±0.02 0.54±0.02 0.54±0.02 |
| n-Step TD-VCL 0.67±0.01 0.67±0.02 0.65±0.01 0.68±0.04 0.69±0.02                                                                                                                                         | 0.56±0.02 0.55±0.02 0.54±0.02 0.56±0.02 0.56±0.02 |                                                   |
| TD(λ)-VCL                                                                                                                                                                                               | 0.66±0.02 0.66±0.04 0.66±0.02 0.67±0.01 0.71±0.01 | 0.57±0.03 0.56±0.02 0.55±0.03 0.56±0.02 0.56±0.02 |

## 5.1. Experiments

We highlight and analyze the following questions to evaluate our hypothesis and proposed method:
Do the TD-VCL objectives effectively alleviate Catastrophic Forgetting in challenging CL benchmarks? Tables 1 and 2 present the results for all benchmarks. Each column presents the average accuracy across the past t observed tasks, and we show the results starting from t = 2 as t = 1 is simply single-task learning. For **PermutedMNIST-Hard**, all methods present high accuracy for t = 2, suggesting that they could fit the data successfully. As the number of tasks increases, they start manifesting Catastrophic Forgetting at different levels. While Online and Batch MLE drastically suffer, variational approaches considerably retain old tasks' performance. The Core Set slightly helps VCL, and both n-Step KL and TD-VCL outperform them by a considerable margin, attaining approximately 90% average accuracy after all tasks. For completeness, Figure 1 graphically shows the results. We emphasize the discrepancy between variational approaches and naive baselines and highlight the performance boost by adopting TD-VCL objectives. For **SplitMNIST-Hard**, we highlight that the TD-VCL objectives also surpass baselines in all configurations, but with a decrease in performance for t = 5, suggesting a more challenging setup for addressing Catastrophic Forgetting that opens a venue for future research. We discuss SplitMNIST-Hard results in more detail in Appendix J. Next, SplitNotMNIST-Hard is a harder benchmark, as the letters come from a diverse set of font styles. Furthermore, we purposely decided to employ a modest network architecture (as for previous benchmarks). Facing hard tasks with less expressive parametrizations will result in higher posterior approximation error. Our goal is to evaluate how the variational methods behave in this setting. Once again, n-step KL and TD-VCL surpassed the baselines after observing more than three tasks. The effect is more pronounced after increasing the number of observed tasks. These objectives are

Temporal-Difference Variational Continual Learning PermutedMNIST-Hard: Per Task Performance Task 1 1 2 3 4 5 6 7 8 9 10 0.2 0.4 0.6 0.8 1.0 Task 2 1 2 3 4 5 6 7 8 9 10 Number of Observed Tasks 0.2 0.4 0.6 0.8 1.0 Task 3 1 2 3 4 5 6 7 8 9 10 0.2 0.4 0.6 0.8 1.0 Task 4 1 2 3 4 5 6 7 8 9 10 0.2 0.4 0.6 0.8 1.0 Task 5 1 2 3 4 5 6 7 8 9 10 0.2 0.4 0.6 0.8 1.0 Ac cu ra cy Task 6 1 2 3 4 5 6 7 8 9 10 0.2 0.4 0.6 0.8 1.0 Task 7 1 2 3 4 5 6 7 8 9 10 Number of Observed Tasks 0.2 0.4 0.6 0.8 1.0 Task 8 1 2 3 4 5 6 7 8 9 10 0.2 0.4 0.6 0.8 1.0 Task 9 1 2 3 4 5 6 7 8 9 10 0.2 0.4 0.6 0.8 1.0 Task 10 1 2 3 4 5 6 7 8 9 10 0.2 0.4 0.6 0.8 1.0 Ac cu ra cy Method Online MLE
Batch MLE
VCL VCL CoreSet N-Step TD-VCL TD( )-VCL
330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 the only ones whose resultant models achieved non-trivial average accuracy after observing all tasks.

Lastly, we analyze the results on **CIFAR100-10** and TinyImageNet-10 in Table 2. These are considerably harder benchmarks, as the distribution of images and classes is much richer than the previous benchmarks. Furthermore, they necessarily require better architectures to attain nontrivial performance. Following previous work (Serra et al., 2018; Kumar et al., 2021; Konishi et al., 2023), we adopt an AlexNet architecture (Krizhevsky, 2009). This setup is ideal for evaluating how the learning objective functions at a larger scale with more complex, deep architectures such as (Bayesian) convolutional networks. Once again, TD- VCL objectives attain superior performance, particularly for later timesteps, where Catastrophic Forgetting is more pronounced in the baselines. This suggests that leveraging multiple posterior estimates for learning is better than only the latest one, even when the approximation error is high.

How do the TD-VCL objectives affect per-task performance? While the previous question analyze the performance averaged across different tasks, we now investigate the accuracy of each task separately in the course of online learning. This setup is relevant since solely considering the averaged accuracy may hide a stronger Catastrophic Forgetting effect from earlier tasks by "compensating" with higher accuracy from later tasks. We show the results for PermutedMNIST-Hard in Figure 3 (we defer additional pertask results for Appendix J). It presents a sequence of plots, where each figure represents the accuracy of one task while the number of observed tasks increases. Naturally, the tasks that appear at later stages present fewer data points: for instance, "Task 10" has a single data point as it does not have test data for earlier timesteps. As observed, per-task performance explicitly shows a stronger effect of Catastrophic Forgetting for earlier tasks in the adopted baselines. We particularly highlight how non-variational approaches fail for them. In this direction, TD-VCL objectives presented a more robust performance against others. For instance, we highlight the results for Task 1. After observing all tasks, the proposed methods demonstrated accuracy of around 80% and 85%. The VCL baselines dropped to 50% and 60%, and MLE-based methods failed with only 20% of accuracy.

## How Does Td-Vcl (And Variants) Perform Against Other Bayesian Cl Methods?

In this work, we focus on Continual Learning with a Bayesian lens. As highlighted in Section 1, it provides a formal, uncertainty-aware framework crucial for safetycritical applications and data-efficient learning. Thus, we analyze the TD objective (Equation 5) on other Bayesian CL methods. UCL and UCB are variational methods that optimize the objective in Equation 2 but propose new mechanisms for regularization and learning rate adaptation. Since these enhancements are orthogonal to the objective, we in-

| 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406   | all benchmarks, their TD-enhanced versions consistently achieve better performance, particularly at later timesteps. PermutedMNIST-Hard SplitMNIST-Hard t = 2 t = 4 t = 6 t = 8 t = 10 t = 2 t = 3 t = 4   | t = 5                                             |       |        |       |       |       |       |        |
|-----------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|-------|--------|-------|-------|-------|-------|--------|
| VCL                                                                               | 0.95±0.00 0.93±0.02 0.89±0.03 0.83±0.04 0.78±0.04                                                                                                                                                          | 0.87±0.02 0.66±0.04 0.82±0.03 0.64±0.11           |       |        |       |       |       |       |        |
| TD(λ)-VCL 0.97±0.00 0.95±0.00 0.93±0.01 0.91±0.01 0.89±0.02                       | 0.98±0.01 0.79±0.08 0.88±0.04 0.67±0.04                                                                                                                                                                    |                                                   |       |        |       |       |       |       |        |
| UCL                                                                               | 0.97±0.00 0.94±0.00 0.89±0.02 0.83±0.06 0.73±0.12                                                                                                                                                          | 0.88±0.04 0.68±0.03 0.83±0.03 0.66±0.06           |       |        |       |       |       |       |        |
| TD(λ)-UCL 0.97±0.00 0.95±0.00 0.92±0.02 0.88±0.04 0.84±0.04                       | 0.97±0.01 0.85±0.06 0.90±0.02 0.70±0.04                                                                                                                                                                    |                                                   |       |        |       |       |       |       |        |
| UCB                                                                               | 0.93±0.01 0.92±0.01 0.89±0.02 0.86±0.02 0.83±0.02                                                                                                                                                          | 0.85±0.16 0.79±0.12 0.83±0.06 0.75±0.10           |       |        |       |       |       |       |        |
| TD(λ)-UCB 0.94±0.00 0.93±0.00 0.91±0.01 0.90±0.01 0.88±0.02                       | 0.93±0.02 0.89±0.03 0.87±0.03 0.80±0.03                                                                                                                                                                    |                                                   |       |        |       |       |       |       |        |
| CIFAR100-10                                                                       | TinyImageNet-10                                                                                                                                                                                            |                                                   |       |        |       |       |       |       |        |
| t = 2                                                                             | t = 4                                                                                                                                                                                                      | t = 6                                             | t = 8 | t = 10 | t = 2 | t = 4 | t = 6 | t = 8 | t = 10 |
| VCL                                                                               | 0.64±0.02 0.63±0.02 0.60±0.02 0.61±0.05 0.66±0.01                                                                                                                                                          | 0.53±0.06 0.51±0.03 0.51±0.03 0.51±0.02 0.51±0.02 |       |        |       |       |       |       |        |
| TD(λ)-VCL 0.66±0.02 0.66±0.04 0.66±0.02 0.67±0.01 0.71±0.01                       | 0.57±0.03 0.56±0.02 0.55±0.03 0.56±0.02 0.56±0.06                                                                                                                                                          |                                                   |       |        |       |       |       |       |        |
| UCL                                                                               | 0.65±0.03 0.64±0.05 0.60±0.05 0.58±0.02 0.62±0.02                                                                                                                                                          | 0.55±0.02 0.52±0.03 0.51±0.02 0.52±0.02 0.50±0.03 |       |        |       |       |       |       |        |
| TD(λ)-UCL 0.68±0.02 0.64±0.01 0.70±0.02 0.66±0.03 0.67±0.03                       | 0.55±0.03 0.54±0.01 0.54±0.01 0.55±0.01 0.56±0.01                                                                                                                                                          |                                                   |       |        |       |       |       |       |        |
| UCB                                                                               | 0.65±0.01 0.66±0.02 0.66±0.03 0.65±0.01 0.66±0.01                                                                                                                                                          | 0.52±0.06 0.51±0.02 0.48±0.04 0.45±0.02 0.42±0.03 |       |        |       |       |       |       |        |
| TD(λ)-UCB 0.64±0.02 0.66±0.01 0.67±0.01 0.68±0.01 0.70±0.01                       | 0.54±0.04 0.52±0.01 0.51±0.02 0.50±0.03 0.47±0.02                                                                                                                                                          |                                                   |       |        |       |       |       |       |        |

corporate the proposed TD objective with these methods, resulting in TD-UCL and TD-UCB, respectively. We aim to show that the TD objectives for CL work across different base methods and promote a performance boost on them.

## 6. Closing Remarks

385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 Table 3 compares the base methods (VCL, UCL, and UCB) with their TD-enhanced counterparts (complete results in Appendix L). While there is no dominant base method across the benchmarks, the TD counterparts consistently improve upon their respective base methods, especially at later timesteps. These results indicate that the TD objective is robust among different Bayesian CL algorithms and may be incorporated effectively into methods that rely on the variational objective in Equation 2. How do the TD-VCL objectives behave with the choice of the hyperparameters n, λ**, and the likelihood-tempering** parameter β? The proposed learning objectives introduce two new hyperparameters: n (the number of considered previous posterior estimates in the learning target) and λ for TD(λ)-VCL (which controls the level of influence for each past posterior estimate). Furthermore, it also inherits the β parameter from VCL. Hence, we evaluate the sensitivity of the proposed objectives concerning these hyperparameters, presenting results and detailed discussion in Appendix K. We highlight three main findings. First, similarly to VCL,
TD-VCL objectives are sensitive to the likelihood-tempering hyperparameter. Second, increasing n is beneficial up to a certain point, from which it becomes detrimental, suggesting the existence of an optimal range for leveraging posterior estimates. Lastly, TD-VCL objectives present robustness over the choice of λ, with a more pronounced effect when the number of observed tasks increases.

In this work, we presented a new family of variational objectives for Continual Learning, namely Temporal-Difference VCL. TD-VCL is an unbiased proxy of the standard VCL objective but leverages several previous posterior estimates to alleviate the compounding error caused by recursive approximations. We showed that TD-VCL represents a spectrum of Continual Learning algorithms and is equivalent to a discounted sum of n-step Temporal-Difference targets. Lastly, we empirically presented that it helps address Catastrophic Forgetting, surpassing Bayesian CL baselines in several challenging benchmarks.

Limitations. Despite being theoretically principled and attaining superior performance, TD-VCL presents limitations. First, the hyperparameters n and λ depend on the evaluated setting, which may require certain tuning. Second, the objectives rely on past posterior estimates, which may increase memory requirements. Still, we believe this is not a major limitation as TD-VCL suits well modern deep Bayesian architectures that target smaller parameter subspaces for posterior approximation (Yang et al., 2024; Dwaracherla et al., 2024; Melo et al., 2024). Future Work. While presenting connections with Temporal-Difference methods, TD-VCL is not an RL algorithm. Further mathematical connections with Markov Decision/Reward Processes formalism are left as future work. Another interesting direction is to apply TD-VCL objectives for other problems that involve sequential variational inference, such as probabilistic meta-learning (Finn et al., 2018; Zintgraf et al., 2020).

## Impact Statement

This work develops a novel learning objective for Bayesian Continual Learning. As such, we believe our work has a positive impact on fundamental research for Machine Learning for three reasons. First, we argue that advancing Continual Learning research is crucial for ensuring the long-term quality of ML models in production systems, as they are vulnerable to potential distributional shifts in the data generation distribution. We also argue that CL is crucial for developing safe autonomous learning agents, as Catastrophic Forgetting may be a dangerous challenge while interacting with the physical or digital world. Second, our particular focus on the Bayesian framework is relevant for designing uncertaintyaware models, which, as argued in Section 1, is crucial for robust Machine Learning and general AI safety. Lastly, we provide a solid theoretical connection between Variational Continual Learning methods and Temporal-Difference methods, effectively bridging two seemingly distant disciplines into a unified family of algorithms. We believe this will inspire further research in the intersection of both areas.

## References

440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 Abraham, W. C. and Robins, A. Memory retention - the synaptic stability versus plasticity dilemma. Trends in Neurosciences, 28(2):73–78, 2005. ISSN 0166-2236. doi: https://doi.org/10.1016/j.tins.2004.12. 003. URL https://www.sciencedirect.com/ science/article/pii/S0166223604003704.

Adel, T., Zhao, H., and Turner, R. E. Continual learning with adaptive weights (CLAW). In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. URL https://openreview.net/forum? id=Hklso24Kwr.

Ahn, H., Cha, S., Lee, D., and Moon, T. *Uncertainty-based* continual learning with adaptive regularization. Curran Associates Inc., Red Hook, NY, USA, 2019.

Auddy, S., Hollenstein, J., and Saveriano, M. Can expressive posterior approximations improve variational continual learning? Workshop on Lifelong Learning for Long-term Human-Robot Interaction of the 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2020.

Bang, J., Kim, H., Yoo, Y., Ha, J.-W., and Choi, J. Rainbow memory: Continual learning with a memory of diverse samples. In *2021 IEEE/CVF Conference on Computer* Vision and Pattern Recognition (CVPR), pp. 8214–8223, 2021. doi: 10.1109/CVPR46437.2021.00812.

Chaudhry, A., Dokania, P. K., Ajanthan, T., and Torr, P. H. S.

Riemannian walk for incremental learning: Understanding forgetting and intransigence. In Ferrari, V., Hebert, M., Sminchisescu, C., and Weiss, Y. (eds.), Computer Vision - ECCV 2018, pp. 556–572, Cham, 2018. Springer International Publishing. ISBN 978-3-030-01252-6.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pp. 248–255, 2009. doi: 10.1109/CVPR.2009.5206848.

Dwaracherla, V., Asghari, S. M., Hao, B., and Roy, B. V. Efficient exploration for LLMs. In Forty-first International Conference on Machine Learning, 2024. URL https:
//openreview.net/forum?id=PpPZ6W7rxy.

Ebrahimi, S., Elhoseiny, M., Darrell, T., and Rohrbach, M.

Uncertainty-guided continual learning with bayesian neural networks. In *International Conference on Learning* Representations, 2020. URL https://openreview. net/forum?id=HklUCCVKDB.

Finn, C., Xu, K., and Levine, S. Probabilistic modelagnostic meta-learning. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, NIPS'18, pp. 9537–9548, Red Hook, NY, USA, 2018. Curran Associates Inc.

Flesch, T., Saxe, A., and Summerfield, C. Continual task learning in natural and artificial agents. Trends in Neurosciences, 46(3):199–210, 2023. ISSN 0166-2236. doi: https://doi.org/10.1016/j.tins.2022.12. 006. URL https://www.sciencedirect.com/ science/article/pii/S0166223622002600.

French, R. M. Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences*, 3(4):128–135, 1999. ISSN 1364-6613. doi: https://doi.org/10.1016/S1364-6613(99)01294-2.

URL https://www.sciencedirect.com/ science/article/pii/S1364661399012942.

Gal, Y., Islam, R., and Ghahramani, Z. Deep bayesian active learning with image data. In Proceedings of the 34th International Conference on Machine Learning - Volume 70, ICML'17, pp. 1183–1192. JMLR.org, 2017.

Ghahramani, Z. and Attias, H. Online variational bayesian learning. In *NeurIPS Workshop on Online Learning*, NeurIPS, 2000.

Goodfellow, I. J., Mirza, M., Xiao, D., Courville, A., and Bengio, Y. An empirical investigation of catastrophic forgetting in gradient-based neural networks. In International Conference on Learning Representations, pp. 1–10, 2015.

Guimeng, L., Yang, G., Sze Yin, C. W., Nagartnam Suganathan, P., and Savitha, R. Unsupervised generative variational continual learning. In *2022 IEEE International* Conference on Image Processing (ICIP), pp. 4028–4032, 2022. doi: 10.1109/ICIP46576.2022.9897538.

Kumar, A., Chatterjee, S., and Rai, P. Bayesian structural adaptation for continual learning. In Meila, M. and Zhang, T. (eds.), Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 5850–5860. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr. press/v139/kumar21a.html.

495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 Hadsell, R., Rao, D., Rusu, A. A., and Pascanu, R. Embracing change: Continual learning in deep neural networks. Trends in Cognitive Sciences, 24(12):1028–1040, 2020. ISSN 1364-6613. doi: https://doi.org/10.1016/j.tics.2020.09. 004. URL https://www.sciencedirect.com/ science/article/pii/S1364661320302199.

Liu, H. and Liu, H. Continual learning with recursive gradient optimization. In *International Conference* on Learning Representations, 2022. URL https:// openreview.net/forum?id=7YDLgf9_zgm.

Loo, N., Swaroop, S., and Turner, R. E. Combining variational continual learning with fiLM layers. In 4th Lifelong Machine Learning Workshop at ICML 2020, 2020. URL https://openreview.net/forum? id=fZBEGA1d-4Y.

Javed, K. and White, M. Meta-learning representations for continual learning. Curran Associates Inc., Red Hook, NY, USA, 2019.

Kendall, A. and Gal, Y. What uncertainties do we need in bayesian deep learning for computer vision? In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS'17, pp. 5580–5590, Red Hook, NY, USA, 2017. Curran Associates Inc. ISBN 9781510860964.

Loo, N., Swaroop, S., and Turner, R. E. Generalized variational continual learning. In International Conference on Learning Representations, 2021. URL https: //openreview.net/forum?id=_IM-AfFhna9.

Lopez-Paz, D. and Ranzato, M. A. Gradient episodic memory for continual learning. In Guyon, I., Luxburg, U. V.,
Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., and Garnett, R. (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips. cc/paper_files/paper/2017/file/ f87522788a2be2d171666752f97ddebb-Paper. pdf.

Kingma, D. and Ba, J. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), San Diego, CA, USA, 2015.

Kingma, D. P. and Welling, M. Auto-Encoding Variational Bayes. In *2nd International Conference on Learning* Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings, 2014.

Kirkpatrick, J., Pascanu, R., Rabinowitz, N. C., Veness, J.,
Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., and Hadsell, R. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114:3521 - 3526, 2016. URL https://api.semanticscholar. org/CorpusID:4704285.

McCloskey, M. and Cohen, N. J. Catastrophic interference in connectionist networks: The sequential learning problem. Psychology of Learning and Motivation, 24:109–165, 1989. URL https://api. semanticscholar.org/CorpusID:61019113.

Melo, L. C., Tigas, P., Abate, A., and Gal, Y. Deep bayesian active learning for preference modeling in large language models, 2024. URL https://arxiv.org/abs/ 2406.10023.

Konishi, T., Kurokawa, M., Ono, C., Ke, Z., Kim, G., and Liu, B. Parameter-level soft-masking for continual learning. In Proceedings of the 40th International Conference on Machine Learning, ICML'23. JMLR.org, 2023.

Nguyen, C. V., Li, Y., Bui, T. D., and Turner, R. E.

Variational continual learning. In International Conference on Learning Representations, 2018. URL https: //openreview.net/forum?id=BkQqq0gRb.

Krizhevsky, A. Learning multiple layers of features from tiny images. In *Technical Report, University of Toronto*, 2009. URL http://www.cs.toronto.edu/
˜kriz/learning-features-2009-TR.pdf.

Pan, P., Swaroop, S., Immer, A., Eschenhagen, R., Turner, R. E., and Khan, M. E. Continual deep learning by functional regularisation of memorable past. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS '20, Red Hook, NY, USA,
2020. Curran Associates Inc. ISBN 9781713829546.

Krizhevsky, A., Sutskever, I., and Hinton, G. E. Imagenet classification with deep convolutional neural networks. Commun. ACM, 60(6):84–90, May 2017. ISSN 00010782. doi: 10.1145/3065386. URL https://doi.

org/10.1145/3065386.

550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 Rainforth, T., Foster, A., Ivanova, D. R., and Smith, F. B.

Modern Bayesian Experimental Design. Statistical Science, 39(1):100 - 114, 2024. doi: 10.1214/23-STS915. URL https://doi.org/10.1214/23-STS915.

Rebuffi, S.-A., Kolesnikov, A., Sperl, G., and Lampert, C. H. icarl: Incremental classifier and representation learning. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5533–5542, 2016. URL https://api.semanticscholar. org/CorpusID:206596260.

Ring, M. B. Child: A first step towards continual learning.

Mach. Learn., 28(1):77–104, jul 1997. ISSN 0885-6125.

doi: 10.1023/A:1007331723572. URL https://doi. org/10.1023/A:1007331723572.

Ritter, H., Botev, A., and Barber, D. Online structured laplace approximations for overcoming catastrophic forgetting. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, NIPS'18, pp. 3742–3752, Red Hook, NY, USA, 2018. Curran Associates Inc.

Schlimmer, J. C. and Fisher, D. A case study of incremental concept induction. In *Proceedings of the Fifth AAAI* National Conference on Artificial Intelligence, AAAI'86, pp. 496–501. AAAI Press, 1986.

Schultz, W., Dayan, P., and Montague, P. R. A neural substrate of prediction and reward. *Science*, 275 (5306):1593–1599, 1997. doi: 10.1126/science.275.5306. 1593. URL https://www.science.org/doi/ abs/10.1126/science.275.5306.1593.

Schwarz, J., Czarnecki, W., Luketina, J., Grabska-
Barwinska, A., Teh, Y. W., Pascanu, R., and Hadsell, R. Progress & compress: A scalable framework for continual learning. In Dy, J. and Krause, A. (eds.), Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 4528–4537. PMLR, 10–15 Jul 2018.

URL https://proceedings.mlr.press/v80/ schwarz18a.html.

Serra, J., Suris, D., Miron, M., and Karatzoglou, A. Overcoming catastrophic forgetting with hard attention to the task. In Dy, J. and Krause, A. (eds.), Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 4548–4557. PMLR, 10–15 Jul 2018. URL https://proceedings.mlr.press/v80/ serra18a.html.

Sutton, R. S. Learning to predict by the methods of temporal differences. *Mach. Learn.*, 3(1):9–44, August 1988. ISSN 0885-6125. doi: 10.1023/
A:1022633531479. URL https://doi.org/10. 1023/A:1022633531479.

Sutton, R. S. and Barto, A. G. *Reinforcement Learning: An* Introduction. A Bradford Book, Cambridge, MA, USA, 2018. ISBN 0262039249.

Sutton, R. S. and Whitehead, S. D. Online learning with random representations. In Proceedings of the Tenth International Conference on International Conference on Machine Learning, ICML'93, pp. 314–321, San Francisco, CA, USA, 1993. Morgan Kaufmann Publishers Inc. ISBN 1558603077.

Thapa, J. and Li, R. Bayesian adaptation of network depth and width for continual learning. In Proceedings of the 41st International Conference on Machine Learning, ICML'24. JMLR.org, 2025.

Titsias, M. K., Schwarz, J., de G. Matthews, A. G., Pascanu, R., and Teh, Y. W. Functional regularisation for continual learning with gaussian processes. In International Conference on Learning Representations, 2020. URL https: //openreview.net/forum?id=HkxCzeHFDB.

Trippe, B. and Turner, R. Overpruning in variational bayesian neural networks, 2018.

Tseran, H. Natural variational continual learning.

2018. URL https://api.semanticscholar.

org/CorpusID:155098533.

Wang, L., Zhang, X., Su, H., and Zhu, J. A comprehensive survey of continual learning: Theory, method and application. IEEE transactions on pattern analysis and machine intelligence, PP, February 2024. ISSN 0162-8828. doi: 10.1109/tpami.2024.3367329. URL https://arxiv.org/pdf/2302.00487.

Yang, A. X., Robeyns, M., Wang, X., and Aitchison, L.

Bayesian low-rank adaptation for large language models. In *The Twelfth International Conference on Learning* Representations, 2024. URL https://openreview.

net/forum?id=FJiUyzOF1m.

Yang, Y., Chen, B., and Liu, H. Memorized variational continual learning for dirichlet process mixtures. *IEEE* Access, 7:150851–150862, 2019. doi: 10.1109/ACCESS. 2019.2947722.

Zeng, G., Chen, Y., Cui, B., and Yu, S. Continual learning of context-dependent processing in neural networks. *Nature Machine Intelligence*, 1:364 - 372, 2018. URL https://api.semanticscholar. org/CorpusID:52908642.

Zenke, F., Poole, B., and Ganguli, S. Continual learning through synaptic intelligence. In Proceedings of the 34th International Conference on Machine Learning - Volume 70, ICML'17, pp. 3987–3995. JMLR.org, 2017.

605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659 Zintgraf, L., Shiarlis, K., Igl, M., Schulze, S., Gal, Y.,
Hofmann, K., and Whiteson, S. Varibad: A very good method for bayes-adaptive deep rl via meta-learning. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum? id=Hkl9JlBYvr.

## A. Derivation Of The N-Step Kl Regularization Objective

In this Section, we prove Proposition 4.1: Proposition 4.1. The standard KL minimization objective in Variational Continual Learning (Equation *2) is equivalently* represented as the following objective, where n ∈ N0 *is a hyperparameter:*
Proof. Starting from Equation 2, we can expand it as a sum of equal terms and utilize the recursive property (Equation 1) to expand these terms:
660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714

$$=\operatorname*{arg\,max}_{q\in\mathbb{Q}}\mathbb{E}_{\mathbf{\theta}\sim q_{t}(\mathbf{\theta})}\Big[\sum_{i=0}^{n-1}\frac{(n-i)}{n}\log p(\mathcal{D}_{t-i}\mid\mathbf{\theta})\Big]-\sum_{i=0}^{n-1}\frac{1}{n}\mathcal{D}_{K L}(q_{t}(\mathbf{\theta})\mid\mid q_{t-i-1}(\mathbf{\theta})).$$
−1 n −1 = arg min q∈Q 1 n " DKL(qt(θ) || qt−1(θ)) − Eθ∼qt(θ)[log p(Dt | θ)] + DKL(qt(θ) || qt−2(θ)) − Eθ∼qt(θ)[log p(Dt | θ) + log p(Dt−1 | θ)] + . . . + DKL(qt(θ) || qt−n(θ)) − Eθ∼qt(θ)[ n X −1 i=0 log p(Dt−i| θ)]# = arg min q∈Q 1 n "nX −1 i=0 DKL(qt(θ) || qt−i(θ)) − Eθ∼qt(θ) hn log p(Dt | θ) + (n − 1) log p(Dt−1 | θ) + · · · + log p(Dt−n+1 | θ) i# Qn−1 i=0 Zt−i i=0
" qt(θ) = arg min q∈Q DKL(q(θ) ||  1 Zt qt−1(θ)p(Dt | θ)) = arg min q∈Q n n DKL(q(θ) || 1 Zt qt−1(θ)p(Dt | θ)) = arg min q∈Q 1 n " DKL(q(θ) ||  1 Zt qt−1(θ)p(Dt | θ)) + DKL(q(θ) || 1 ZtZt−1 qt−2(θ)p(Dt | θ)p(Dt−1 | θ)) + . . . + DKL(q(θ) || 1 Qn−1 i=0 Zt−i qt−n(θ) n Y −1 i=0 p(Dt−i| θ))#
$$({\mathfrak{g}})$$

$$q_{t}(\mathbf{\theta})=\arg\max_{q\in\mathcal{Q}}\mathbb{E}_{\mathbf{\theta}\sim q_{t}(\mathbf{\theta})}\Big{[}\sum_{i=0}^{n-1}\frac{(n-i)}{n}\log p(\mathcal{D}_{t-i}\mid\mathbf{\theta})\Big{]}$$ $$\qquad\qquad-\sum_{i=0}^{n-1}\frac{1}{n}\mathcal{D}_{KL}(q_{t}(\mathbf{\theta})\mid\mid q_{t-i-1}(\mathbf{\theta})).$$
$$\quad(4)$$

## B. Derivation Of The Temporal-Difference Vcl Objective

Before proving Proposition 4.2, we start by presenting a well known result for the sum of geometric series: Lemma B.1. The finite sum of a geometric series with n terms, common ratio λ and initial term a *is given by:*

$$\sum_{k=0}^{n-1}\lambda^{k}a={\frac{a(1-\lambda^{n})}{(1-\lambda)}}$$

(1 − λ)(10)
Proof. Let sn =Pn k=0 λ ka. Hence,

$$s_{n}-\lambda s_{n}=\sum_{k=0}^{n-1}\lambda^{k}a-\lambda\sum_{k=0}^{n-1}\lambda^{k}a=a-a\lambda^{n}$$ $$\Longleftrightarrow s_{n}(1-\lambda)=a(1-\lambda^{n})$$ $$\Longleftrightarrow s_{n}=\frac{a(1-\lambda^{n})}{(1-\lambda)}.$$
$$(10)$$
$$(11)$$
$\square$
715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769 Now, we prove Proposition 4.2. Proposition 4.2. The standard KL minimization objective in VCL (Equation 2) is equivalently represented as the following objective, with n ∈ N0, and λ ∈ [0, 1) *hyperparameters:*

$$q_{t}(\theta)=$$

Proof. We can use Lemma B.1 to expand the sum of KL terms:

arg max q∈Q Eθ∼qt(θ) h nX −1 i=0 λ i(1 − λ n−i) 1 − λnlog p(Dt−i| θ) i − n X −1 i=0 λ i(1 − λ)
1 − λnDKL(qt(θ) || qt−i−1(θ)). (5)
$$({\boldsymbol{5}})$$
770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 810 811 812 813 814 815 816 817 818 819 820 821 822 823 824

qt(θ) = arg min
q∈Q
DKL(q(θ) || 
1
Zt
qt−1(θ)p(Dt | θ))
= arg min
q∈Q
1 − λ
1 − λn
1 − λ
n
1 − λ
DKL(q(θ) || 
1
Zt
qt−1(θ)p(Dt | θ))
= arg min
q∈Q
1 − λ
1 − λn
"
DKL(q(θ) || 
1
Zt
qt−1(θ)p(Dt | θ))
+ λDKL(q(θ) || 1
ZtZt−1
qt−2(θ)p(Dt | θ)p(Dt−1 | θ)) + . . .
+ λ
n−1DKL(q(θ) || 1
Qn−1
i=0 Zt−i
qt−i(θ)
n
Y
−1
i=0
p(Dt−i| θ))#
= arg min
q∈Q
1 − λ
1 − λn
"
DKL(qt(θ) || qt−1(θ)) − Eθ∼qt(θ)[log p(Dt | θ)]
+ λDKL(qt(θ) || qt−2(θ)) − λEθ∼qt(θ)[log p(Dt | θ) + log p(Dt−1 | θ)] + . . .
+ λ
n−1DKL(qt(θ) || qt−n(θ)) − λ
n−1Eθ∼qt(θ)[
n
X
−1
i=0
log p(Dt−i| θ)]#
= arg min
q∈Q
1 − λ
1 − λn
"nX
−1
i=0
λ
iDKL(qt(θ) || qt−i−1(θ)) − Eθ∼qt(θ)
h nX
−1
i=0
λ
ilog p(Dt | θ)
+
n
X
−1
i=1
λ
ilog p(Dt−1 | θ) + *· · ·* + λ
n−1log p(Dt−n+1 | θ)
i#
= arg min
q∈Q
1 − λ
1 − λn
"nX
−1
i=0
λ
iDKL(qt(θ) || qt−i−1(θ)) − Eθ∼qt(θ)
h1 − λ
n
1 − λ
log p(Dt | θ)
+
λ(1 − λ
n−1)
1 − λlog p(Dt−1 | θ) + *· · ·* + λ
n−1log p(Dt−n+1 | θ)
i#
= arg max
q∈Q
Eθ∼qt(θ)
h nX
−1
i=0
λ
i(1 − λ
n−i)
1 − λnlog p(Dt−i| θ)
i−
n
X
−1
i=0
λ
i(1 − λ)
1 − λnDKL(qt(θ) || qt−i−1(θ)).
(12)  $\text{}$  . 

## C. The Connection Of Td Targets In Td-Vcl And Reinforcement Learning

825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 In the Section 4, we formalize the concept of n-Step Temporal-Difference for the Variational CL objective (Definition 4.3). In this Section, we reveal the connections between this definition and the widely used Temporal-Difference methods in Reinforcement Learning. Our aim is to clarify why Equation 6 indeed represents a temporal-difference target, both in a broad and strict senses.

In a **broad** sense, *bootstrapping* characterizes a Temporal-Difference target: building a learning target estimate based on previous estimates. Crucially, the leveraged estimates are functions of different timesteps. TD-VCL objectives applies bootstrapping in the KL regularization term, by considering one or more of posteriors estimates from previous timesteps. In a **strict** sense, we can show that Equation 6 deeply resembles TD targets in Reinforcement Learning. RL assumes the formalism of a Markov Decision Process (MDP), defined by a tuple M = (S, A,P, R,P0*, γ, H*), where S is a state space, A is an action space, P : *S × A × S →* [0, ∞) is a transition dynamics, R : S × A → [−Rmax, Rmax] is a bounded reward function, P0 : S → [0, ∞) is an initial state distribution, γ ∈ [0, 1] is a discount factor, and H is the horizon.

The standard RL objective is to find a policy that maximizes the cumulative reward:

$$\pi_{\mathbf{\theta}}^{*}=\operatorname*{arg\,max}_{\pi}\mathbb{E}_{\pi}[\sum_{k=0}^{H}\gamma^{k}\mathcal{R}(s_{t+k},a_{t+k})],\tag{1}$$
$$(13)$$
$$(14)$$

with at ∼ πθ(at | st), st ∼ P(st | st−1, at−1), and s0 ∼ P0(s), where πθ : *S × A →* [0, ∞) is a policy parameterized by θ. Hence, we can define the following learning target, which represents a "value" function at each state st:

$$v_{\pi}(s_{t}):=\mathbb{E}_{\pi}[\sum_{k=0}^{H}\gamma^{k}\mathcal{R}(s_{t+k},a_{t+k})\mid s=s_{t}],\forall s_{t}\in\mathcal{S}.\tag{1}$$

Naturally, it follows that π
∗
θ = arg maxπvπ(s), ∀s ∈ S. Crucially, we can expand Equation 14 as follows:
Temporal-Difference methods estimates a learning target directly from Equation 15: Now, we turn our attention back to our Variational Continual Learning setting. The standard VCL objective is given by Equation 2:

$$q_{t}(\mathbf{\theta})=\operatorname*{arg\,min}_{q\in\mathbb{Q}}{\mathcal{D}}_{K L}(q(\mathbf{\theta})\mid\mid{\frac{1}{Z_{t}}}q_{t-1}(\mathbf{\theta})p({\mathcal{D}}_{t}\mid\mathbf{\theta})).$$
$$(16)$$

16

$${\hat{v}}_{\pi}(s):=\mathrm{TD}_{\mathrm{BL}}(n)=\underbrace{\mathbb{E}_{\pi}[\sum_{k=0}^{n-1}\gamma^{k}\mathcal{R}(s_{t},a_{t})]}_{\mathrm{Estimated~via~MC~sampling}}+\underbrace{\gamma^{n}{\hat{v}}_{\pi}(s_{t+n})}_{\mathrm{Bootstrapped~via~point~equations}},\forall s_{t}\in\mathcal{S},n\leq H.$$
$$=\mathbb{E}_{\pi}[\mathcal{R}(s_{t},a_{t})+\sum_{k=1}^{H}\gamma^{k}\mathcal{R}(s_{t+k},a_{t+k})\mid s=s_{t}]$$ $$=\mathbb{E}_{\pi}[\mathcal{R}(s_{t},a_{t})+\gamma v_{\pi}(s_{t+1})],$$ $$=\mathbb{E}_{\pi}[\mathcal{R}(s_{t},a_{t})+\gamma\mathcal{R}(s_{t+1},a_{t+1})+\gamma^{2}v_{\pi}(s_{t+2})],$$ $$=\mathbb{E}_{\pi}[\sum_{k=0}^{n-1}\gamma^{k}\mathcal{R}(s_{t},a_{t})+\gamma^{n}v_{\pi}(s_{t+n})],\forall s_{t}\in\mathcal{S},n\leq H.$$
$$(15)$$
nvπ(st+n)], ∀st ∈ S, n ≤ H. (15)
$v_{\pi}(s_{t}):=\mathbb{E}_{\pi}[\sum_{k=0}^{H}\gamma^{k}\mathcal{R}(s_{t+k},a_{t+k})\mid s=s_{t}]$
We can similarly define a learning target as a "value" function which we aim to maximize:
Similarly to the RL case, it follows that qt(θ) = arg maxq∈Q uq(θ)(t). Lastly, we assume the following estimation of the
"value" function defined in Equation 17:
880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934

qt(θ) = arg max q∈Q uˆq(θ)(t) = arg max q∈Q Eθ∼qt(θ) "nX −1 i=0 log p(Dt−i| θ)]#− DKL(qt(θ) || qt−n(θ)) +  "nX −1 i=0 log Zt−i # = arg max q∈Q Eθ∼qt(θ) "nX−1 i=0 log p(Dt−i| θ)]#− DKL(qt(θ) || qt−n(θ)) . (19) | {z } TDCL(n) Equation 19 is exactly n-Step Temporal-Difference target in Definition 4.3 from Section 4. The main differences from the CL
$$(19)$$
recursion in Equation 17 and the RL one in Equation 15 are two-fold. First, the CL setup is not discounted (or, equivalently, assumes the discount factor γ = 1). Second, the RL recursion looks over future timesteps, while the CL one looks over past timesteps. Besides these two differences, both scenarios are strongly connected. Particularly, they share the same purpose for leveraging TD targets: to strike a balance between MC estimation (which incurs variance) and bootstrapping (which incurs bias) while estimating the learning objective.

uq(θ)(t) := −DKL(q(θ) || 1 Zt qt−1(θ)p(Dt | θ)) = Eθ∼qt(θ) " log p(Dt | θ)] + log Zt # − DKL(qt(θ) || qt−1(θ)) = Eθ∼qt(θ) " log p(Dt | θ)] + log Zt # − DKL(qt(θ) ||  1 Zt−1 qt−2(θ)p(Dt−1 | θ)) = Eθ∼qt(θ) " log p(Dt | θ)] + log Zt # + uq(θ)(t − 1) = Eθ∼qt(θ) "nX −2 i=0 log p(Dt−i| θ)] + n X −2 i=0 log Zt−i # + uq(θ)(t − n + 1), n ∈ N0, n ≤ t. (17)
$$\hat{u}_{q(\mathbf{\theta})}(t)=\mathbb{E}_{\mathbf{\theta}\sim\mathbf{u}_{t}(\mathbf{\theta})}\left[\sum_{i=0}^{n-2}\log p(\mathcal{D}_{t-i}\mid\mathbf{\theta})\right]+\sum_{i=0}^{n-2}\log Z_{t-i}\right]+\hat{u}_{q(\mathbf{\theta})}(t-n+1)$$ $$=\underbrace{\mathbb{E}_{\mathbf{\theta}\sim\mathbf{u}_{t}(\mathbf{\theta})}\left[\sum_{i=0}^{n-1}\log p(\mathcal{D}_{t-i}\mid\mathbf{\theta})\right]}_{\text{Estimated via MC Sampling}}-\underbrace{\mathcal{D}_{KL}(q_{t}(\mathbf{\theta})\mid\mid q_{t-n}(\mathbf{\theta}))}_{\text{Backward to a past posterior estimations}}+\underbrace{\left[\sum_{i=0}^{n-1}\log Z_{t-i}\right]}_{\text{Combined task}\in\mathbf{I}\ \mathbf{\theta}}.$$
$$(18)$$
. (18)
We notice that Zt is constant with respect to θ, hence we can disregard it and still have the same learning target. Thus, we have:

## D. Td(Λ**)-Vcl Is A Discounted Sum Of N-Step Td Targets**

In Section 4, we mention that the TD-VCL learning target is a compound update that averages n-step temporal-difference targets, as per Proposition 4.4, which we prove below.

Proposition 4.4. ∀n ∈ N0, n ≤ t , the objective in Equation 2 *can be equivalently represented as:*

$$q_{t}(\mathbf{\theta})=\operatorname*{arg\,max}_{q\in\mathbb{Q}}\mathrm{TD}_{t}(n),$$
TDt(n), (7)
with TDt(n) as in Definition 4.3. Furthermore, the objective in Equation 5 *can also be represented as:*
Proof. We start by proving the equivalence between Equation 2 and Equation 7: Now, we show that Equation 5 is a discounted sum of n-Step targets:
935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989 In Equation 7, if we set n = 1, the n-Step TD target recovers the VCL objective. Furthermore, it is worth highlighting that an n-Step TD target is not the same as n-Step KL Regularization. The latter leverages several previous posterior estimates, while the former only relies on a single estimate. Lastly, we can follow a similar idea to prove that the n-Step KL Regularization objective is a simple average of n-step TD targets, by leveraging the expansion in Equation 9 and identifying the sum of TD targets.

qt(θ) = arg min q∈Q DKL(q(θ) ||  1 Zt qt−1(θ)p(Dt | θ)) = arg min q∈Q DKL(q(θ) || 1 Qn−1 i=0 Zt−i qt−n(θ) n Y −1 i=0 p(Dt−i| θ)) (20) = arg max q∈Q Eθ∼qt(θ) "nX −1 i=0 log p(Dt−i| θ)]#− DKL(qt(θ) || qt−n(θ)) = arg max q∈Q TDt(n).
qt(θ) = arg max q∈Q 1 − λ 1 − λn "Eθ∼qt(θ)[log p(Dt | θ) − DKL(qt(θ) || qt−1(θ))] + λEθ∼qt(θ)[log p(Dt | θ) + log p(Dt−1 | θ)] − λDKL(qt(θ) || qt−2(θ)) + . . . + λ n−1Eθ∼qt(θ)[ n X −1 i=0 log p(Dt−i| θ)] − λ n−1DKL(qt(θ) || qt−n(θ))# = arg max q∈Q 1 − λ 1 − λn " TDt(1) + λTDt(2) + . . . λn−1TDt(n) # = arg max q∈Q 1 − λ 1 − λn "nX −1 k=0 λ kTDt(k + 1))#. | {z } Disconted sum of TD targets
$$(21)$$
$$\left(7\right)$$
$$(8)$$
$$q_{t}(\mathbf{\theta})=\operatorname*{arg\,max}_{q\in\mathcal{Q}}{\frac{1-\lambda}{1-\lambda^{n}}}\underbrace{\left[\sum_{k=0}^{n-1}\lambda^{k}\mathrm{TD}_{t}(k+1))\right]}_{\mathrm{Discounted~sum~of~TD~targets}}.$$

## E. Td-Vcl: A Spectrum Of Continual Learning Algorithms

In this Section, we describe how TD-VCL spans a spectrum of algorithms that mix different levels of Monte Carlo approximation for expected log-likelihood and KL regularization. Our goal is to show that by choosing specific hyperparameters for Equation 5, one may recover vanilla VCL in one extreme and n-Step KL regularization in the opposite. Let us consider the TD-VCL objective in Equation 5:

$$\operatorname*{arg\,max}_{\boldsymbol{\theta}\in\mathcal{Q}}\mathbb{E}_{\boldsymbol{\theta}\sim\boldsymbol{q}_{t}(\boldsymbol{\theta})}\Big{[}\sum_{i=0}^{n-1}\frac{\lambda^{i}(1-\lambda^{n-i})}{1-\lambda^{n}}\log p(\mathcal{D}_{t-i}\mid\boldsymbol{\theta})\Big{]}-\sum_{i=0}^{n-1}\frac{\lambda^{i}(1-\lambda)}{1-\lambda^{n}}\mathcal{D}_{KL}(q_{t}(\boldsymbol{\theta})\mid\mid q_{t-i-1}(\boldsymbol{\theta})).$$

Trivially, if we set λ = 0, assuming 0 0 = 1, it recovers the Vanilla VCL objective, as stated in Equation 3, regardless of the choice of n. More interestingly, we investigate the learning target as λ → 1:

lim λ→1 (Eθ∼qt(θ) h nX −1 i=0 λ i(1 − λ n−i) 1 − λnlog p(Dt−i| θ) i− n X −1 i=0 λ i(1 − λ) 1 − λnDKL(qt(θ) || qt−i−1(θ))) = Eθ∼qt(θ) h nX −1 log p(Dt−i| θ) i− n X −1 i=0 lim λ→1 nλ i(1 − λ n−i) 1 − λn o i=0 lim λ→1 nλ i(1 − λ) 1 − λn o DKL(qt(θ) || qt−i−1(θ)) | {z } (I) | {z } (II)
990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 Let us develop (I) and (II) separately by applying the L'Hopital's rule. First, for ˆ (I): Now, for (II): Applying Equations 22 and 23 to TD-VCL objective, we obtain:

$$(22)$$

$$(23)^{\frac{1}{2}}$$
$$\operatorname*{lim}_{\lambda\to1}\left\{{\frac{\lambda^{i}(1-\lambda)}{1-\lambda^{n}}}\right\}=\operatorname*{lim}_{\lambda\to1}\left\{{\frac{i\lambda^{i-1}(1-\lambda)-\lambda^{i}}{-n\lambda^{n-1}}}\right\}={\frac{1}{n}}.$$
. (23)
$$\operatorname*{arg\,max}_{q\in\mathbb{Q}}\mathbb{E}_{\mathbf{\theta}\sim q_{t}(\mathbf{\theta})}\Big[\sum_{i=0}^{n-1}{\frac{(n-i)}{n}}\log p(\mathcal{D}_{t-i}\mid\mathbf{\theta})\Big]-\sum_{i=0}^{n-1}{\frac{1}{n}}\mathcal{D}_{K L}(q_{t}(\mathbf{\theta})\mid\mid q_{t-i-1}(\mathbf{\theta})),$$

which is exactly the N-Step KL Regularization objective.

$$\operatorname*{lim}_{\lambda\to1}\Big\{\frac{\lambda^{i}(1-\lambda^{n-i})}{1-\lambda^{n}}\Big\}=\operatorname*{lim}_{\lambda\to1}\Big\{\frac{i\lambda^{i-1}(1-\lambda^{n-i})-\lambda^{i}(n-i)\lambda^{n-i-1}}{-n\lambda^{n-1}}\Big\}$$ $$=\operatorname*{lim}_{\lambda\to1}\Big\{\frac{i\lambda^{i-1}-i\lambda^{n-1}-(n-i)\lambda^{n-1}}{-n\lambda^{n-1}}\Big\}=\frac{n-i}{n}.$$

## F. Implementation Details And Reproducibility

1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079 1080 1081 1082 1083 1084 1085 1086 1087 1088 1089 1090 1091 1092 1093 1094 1095 1096 1097 1098 1099 Operationalization. For all experiments, we use a Gaussian mean-field approximate posterior and assume a Gaussian prior N (0, σ2I) for the variational methods. We parameterize all distributions as deep networks. For all considered objectives, we compute the KL term analytically and employ the Monte Carlo approximations for the expected loglikelihood terms, leveraging the reparametrization trick (Kingma & Welling, 2014) for computing gradients. Lastly, we employ likelihood-tempering (Loo et al., 2021) to prevent variational over-pruning (Trippe & Turner, 2018).

Model Architecture and Hyperpatameters. We adopt fully connected neural networks for PermutedMNIST-Hard, SplitMNIST-Hard and SplitNotMNIST-Hard. We choose different depths and sizes depending on the benchmark, and we provide a full list of hyperparameters in Appendix G. For CIFAR100-10 and TinyImageNet-10, we implement a Bayesian version of the AlexNet (Krizhevsky et al., 2017), a traditional convolutional neural network architecture, as in prior Bayesian CL literature (Thapa & Li, 2025). Crucially, also following prior literature (Ebrahimi et al., 2020), we do not use pre-trained representations, as our goal is to evaluate how the proposed objectives perform in the CL setting, which also requires learning their own robust representations. Finally, for training, we adopt the Adam optimizer (Kingma & Ba, 2015) and employ early stopping with a patience parameter of five epochs, which drastically reduces the number of epochs needed for each new task in comparison to previous work (Nguyen et al., 2018). Hyperparamter Tuning Protocol. We conduct hyperparameter tuning for all methods in the paper, including the baselines (VCL, UCL, UCB). We follow a random search for each evaluated benchmark. For a fair comparison, we ensure that all methods use approximately the same compute of 1 GPU day. We provide the search space for each method in our released code. For the proposed methods, we mainly tuned three hyperparameters: n (as in n-Step KL), λ (as in TD-VCL), and β (the likelihood tempering parameter). We conducted a grid search for each evaluated benchmark, with n ∈ {1, 2, 3, 5, 8, 10}, λ ∈ {0.0, 0.1, 0.5, 0.8, 0.9, 0.99}, and β ∈ {1e − 5, 1e − 4, 1e − 3, 5e − 3, 1e − 2, 5e − 2, 1e − 1, 1.0}. Reproducibility. Reported results are averaged across ten different seeds for PermutedMNIST-Hard, SplitMNIST-Hard, and SplitNotMNIST-Hard, and five seeds for CIFAR100-10 and TinyImageNet-10. Error bars represent 95% confidence intervals, while tables show 2-sigma errors up to two decimal places. We execute all experiments using a single GPU RTX 4090. We provide our implementation code for the proposed methods (TD-VCL, TD-UCB, TD-UCL, and n-
Step), as well as considered baselines (Batch MLE, Online MLE, VCL, VCL CoreSet, UCB, and UCL) in https: //anonymous.4open.science/r/vcl-nstepkl-5707.