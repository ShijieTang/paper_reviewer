# Map: Multi-Human-Value Alignment Palette

Xinran Wang1 Qi Le1 Ammar Ahmed1 Enmao Diao Yi Zhou2 **Nathalie Baracaldo**2 Jie Ding1 **Ali Anwar**1 1University of Minnesota 2IBM Research
{wang8740, le000288, ahme0599, dingj, aanwar}@umn.edu, diao em@hotmail.com, yi.zhou@ibm.com, baracald@us.ibm.com

## Abstract

Ensuring that generative AI systems align with human values is essential but challenging, especially when considering multiple human values and their potential trade-offs. Since human values can be personalized and dynamically change over time, the desirable levels of value alignment vary across different ethnic groups, industry sectors, and user cohorts. Within existing frameworks, it is hard to define human values and align AI systems accordingly across different directions simultaneously, such as harmlessness, helpfulness, and positiveness. To address this, we develop a novel, first-principle approach called Multi-Human-Value Alignment Palette (MAP), which navigates the alignment across multiple human values in a structured and reliable way. MAP formulates the alignment problem as an optimization task with user-defined constraints, which define human value targets. It can be efficiently solved via a primal-dual approach, which determines whether a user-defined alignment target is achievable and how to achieve it. We conduct a detailed theoretical analysis of MAP by quantifying the trade-offs between values, the sensitivity to constraints, the fundamental connection between multi-value alignment and sequential alignment, and proving that linear weighted rewards are sufficient for multi-value alignment. Extensive experiments demonstrate MAP's ability to align multiple values in a principled manner while delivering strong empirical performance across various tasks. Our code is available at https://github.com/wang8740/MAP.

## 1 Introduction

Recent advancements in artificial intelligence (AI) have highlighted the critical need for aligning AI systems with human values, a concept known as *human value alignment* (Griffith et al., 2013; Arumugam et al., 2019; Gabriel, 2020). The alignment can serve the purpose of generating outcomes that are better suited for human ethics (Griffith et al., 2013), personalized needs (Kirk et al., 2024), or reduced harmful content (Bai et al., 2022). This alignment has traditionally been pursued by adjusting AI behavior to adhere to specific attributes via preference datasets or reward functions. This process involves finetuning the original model according to the optimization problem:

$$\operatorname*{max}_{p\in\mathcal{P}}\mathbb{E}_{x\sim\mathcal{D},y\sim p(\cdot\mid x)}\bigg\{R(x,y)-\beta D_{\mathrm{KL}}\big(p(\cdot\mid x)\,||\,p_{0}(\cdot\mid x)\big)\bigg\}.$$
R(x, y) − βDKLp(· | x)|| p0(· | x). (1)
Here, P denotes the class of all distributions, p0 is the distribution that represents the generative model to align, p is the distribution that represents the aligned model, R is a reward function that quantifies the preference level of any given pair of prompt x and generation y, DKL measures the KL-divergence, and β > 0 is a regularization hyperparameter. This formulation has deep conceptual roots in the Bayesian decision theoretic framework (Bissiri et al., 2016). Specifically, if we consider x as observed data and y as a parameter θ, the problem (1) can be expressed as Eθ∼p(·)
log p(x | θ) − DKLp(·)|| p0(·)	. This formulation yields the solution px(θ) ∝ p0(θ)p(x | θ), which is precisely Bayes' Rule. However, while this formulation provides an elegant interpretation of how AI models can be adjusted to reflect new information or preferences, it may not fully capture the complexity required when aligning AI systems to multiple, potentially conflicting human values. For example, a healthcare-related large language model must deliver information that is not only precise but also easy-to-understand and harmless (Gebreab et al., 2024). Similarly, a customer service

$$(1)$$

1

Pareto Frontier Pareto Frontier
chatbot optimized for efficiency may compromise on politeness or empathy (Kolasani, 2023). These scenarios highlight the need to balance diverse human values, which often requires more nuanced solutions than a single-dimensional alignment approach can offer. Related work. Centered around the formulation (1), a standard approach is the reinforcement learning from human feedback (RLHF) (Griffith et al., 2013; Arumugam et al., 2019; Bai et al., 2022; Dai et al., 2024) that first trains a reward model R based on pairwise comparison or ranking data to serve as a proxy for human preferences, and then uses reinforcement learning with the reward model to address the optimization problem (1). An alternative method known as direct preference optimization (DPO) (Rafailov et al., 2023) directly optimizes an empirical risk based on the Bradley- Terry loss (Bradley & Terry, 1952) applied to the preference data, using an implicit reward in the form of r(*x, y*)
∆= β log(p(y | x)/p0(y | x)). To address multiple human values, recent work on aligning foundation models has used multi-objective reinforcement learning (MORL) (Barrett & Narayanan, 2008; Li et al., 2020; Wu et al., 2024). Existing studies focus on approximating tradeoffs among values utilizing a linear scalarization method that combines either reward functions or data sources, for both RLHF and DPO approaches (Bai et al., 2022; Rame et al., 2023; Dai et al., 2024). Alternatively, some studies utilize specific choices of weights to manage trade-offs (Dognin et al., 2024). A recent technique named "rewarded soup" was introduced to efficiently compute the aligned models under linear scalarization (Rame et al., 2023). It first separately fine-tunes multiple models, each with a particular reward function, and then aggregates these networks by linearly interpolating their weights. This method aims to approximate the ensemble of fine-tuned models that would otherwise result from optimizing a single reward composed of various linear combinations of individual reward functions, namely R
∆=Pm i=1 λiri with random λ = [λ1*, . . . , λ*m]. A similar idea was applied to DPO (Zhou et al., 2024), where the DPO-aligned models under individual values are interpolated to approximate the Pareto Frontier. Further related work is provided in Appendix B.5. Challenges. Aligning AI models to multiple values simultaneously presents several unresolved challenges. First, as demonstrated in Figure 1, aligning with one value, such as helpfulness, harmlessness, or humor, could inadvertently diminish another. This motivates a critical question: How can we quantify and enhance multiple human values concurrently without compromise? Moreover, in the RLHF approach as described by the problem (1), it is unclear how to specify the hyperparameter β and reward function R so that the aligned model p improves upon, or at least not

(a) Align Helpfulness and Harmlessness λHarmle ssness
(b) Align Helpfulness, Harmlessness, and Humor Desirable λ Random λ 0 1 2 3 4 5 6 λHelpfulness 0 1 2 3 4 5 6 0 1 2 3 4 5 6 λHelpfulness 0 1 2 3 4 5 6 Desirable λ Random λ λH
um or
(c) Align Helpfulness, Harmlessness, and Humor Desirable λ Random λ λHar mles snes s 0 12 34 56 λHelpfulness 1 0 3 2 5 4 6 λHarmlessness 0 1 2 3 4 5 6 Original realized value levels MAP realized value levels Distribution of reward scores under original model Distribution of reward scores under MAP-aligned model 4 2 0 2 4 Harmlessness scores 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 4 2 0 2 Helpfulness scores 0.0 0.2 0.4 0.6 0.8 1.0 2 1 0 1 2 Humor scores 0.0 0.5 1.0 1.5 2.0 2.5 3.0 De ns ity De ns ity De ns ity
worse than, p0 in all human values in one shot (without trial-and-error). Even if we can try all possible combinations, there is no theoretical justification that linearly combining individual reward functions is sufficient to obtain the Pareto Frontier. The DPO method, while simplifying the alignment process through a direct empirical risk optimization, still does not address the issue of integrating multiple data sources, which have their underpinning still at the problem (1). Recent studies have demonstrated the sensitivity of results to different weights used in these aggregations (Bai et al., 2022). To highlight this point, in Figure 2, we visualize the range of possible λ and of desirable λ (which actually admits Pareto improvement on all the values). Figure 2(b) shows how adding additional value-to-align narrows the range of desirable λ compared with Figure 2(a). Contributions. We introduce the Multi-Human-Value Alignment Palette (MAP), a principled approach to rigorously aligning multi-dimensional values with provable guarantees. Similar to an artist's color palette, MAP enables the blending of multiple human values to "paint" AI behavior with a broad spectrum of preference shades. In Figure 3, we illustrate how MAP allows users to precisely customize and control the level of improvement for all values in an interpretable manner. The proposed MAP introduces several technical innovations and contributions:
- Formulation. We propose a novel problem formulation that allows one to align multiple human values using user-defined constraints, which we term "value palettes." Each palette acts as a constraint that represents a preferred level of alignment, allowing us to "MAP" from any targeted value levels specified by the user to a particular reward function for (1). This precise one-to-one mapping ensures exact adjustments to model behavior. - Theory. We provide theoretical analysis within the MAP framework quantifies the representation of the solution, its sensitivity to changes in the value palette, and its feasible operational range. This leads to a deeper understanding of the inherent trade-offs among various values. Furthermore, we investigate the range of realizable value levels and demonstrate that a linear combination of individual reward functions, is sufficient to reach the Pareto Frontier. We also establish a crucial link between multi-value alignment and sequential alignment, showing that cyclically updating each value multiple times achieves equivalent results to a single execution of MAP. - Computational methodology. Based on our theoretical derivations, we propose a computational method to ascertain the achievability of user-defined value palettes. The proposed method utilizes a primal-dual approach to efficiently solve the optimization problem for a feasible choice within the palette. We verify that the dual problem exhibits concavity, enabling effective resolution through gradient ascent techniques. MAP is agnostic to the stages of model alignment, whether at the decoding or training stage, the model architecture, and the specific definitions of values used. Finally, we conducted comprehensive experimental studies to validate the practical effectiveness of MAP.

## 2 Map: Multi-Human-Value Alignment Palette

2.1 PROBLEM FORMULATION The formulation in (1) can be seen as maximizing the expected reward while imposing a regularization to minimize unnecessary deviations from the original model. This insight leads us to define a value alignment through a statistical functional constraint:
Ex∼D,y∼p(·|x)r(x, y) ≥ c, (2)
which is interpreted as the expected rewards, or realized levels, under a value preference must be at least c. Likewise, to align m ≥ 1 value preferences, we introduce the following **MAP problem**:
min p∈P
Ex∼D,y∼p(·|x)DKLp(· | x)|| p0(· | x)s.t. Ex∼D,y∼p(·|x)ri(x, y) ≥ ci, ∀i = 1*, . . . , m.* (3)
We denote c∆= [c1*, . . . , c*m]
Tas the *value palette*. With a solution p, we refer to Ex∼D,y∼p(·|x)(r(*x, y*)
∆= [r1(x, y), . . . , rm(*x, y*)]T) as the *realized value levels*. We write u ≥ v if the two vectors are of the same size and ui ≥ vi for each entry i.

Theorem 1 (Representation of MAP solution). The solution to the MAP problem (3) is

$$\mathbb{E}_{x\sim{\mathcal{D}},y\sim p(\cdot|x)}r(x,y)\geq c,$$
$$(2)$$
$L_{\rm eff}=-W$The 
$$p_{\boldsymbol{\lambda}}(y\mid x)={\frac{1}{Z(x,\boldsymbol{\lambda})}}p_{0}(y\mid x)e^{\boldsymbol{\lambda}^{\mathrm{T}}\boldsymbol{r}(x,y)},$$

where λ Tr(*x, y*) = Pm i=1 λiri(*x, y*), for some λ ≥ 0. Moreover, assuming that r(*x, y*) is not trivially a constant on the support set of *x, y*, the above λ is the unique solution to the problem:

$$(4)$$
$$\operatorname*{max}_{\boldsymbol{\lambda}\geq\mathbf{0}}g({\boldsymbol{\lambda}})\triangleq-\log Z({\boldsymbol{\lambda}})+{\boldsymbol{\lambda}}^{\mathrm{{T}}}{\boldsymbol{c}},$$
$$({\boldsymbol{S}})$$
Tc, (5)
where Z(λ)
∆= Ex∼D,y∼p0(·|x)e λ Tr(x,y), and g is strictly concave. As a result, we can treat λ in (4)
as an implicit function of c and denote it as λ = λ(c)
∆= arg maxλ≥0 g(λ).

Remark 1 (Interpretation of λ). Theorem 1 establishes a one-to-one correspondence between the vectors c and λ. The proof to the first part of Theorem 1 uses the Karush–Kuhn–Tucker (KKT) conditions, and the second part is obtained by showing that (5) is the dual problem of (3). From a decision theoretic view, the decision of λ is based on trading off the utility term λ Tc and the "risk" term − log Z(λ). The latter term can be seen as a form of risk aversion, because maximizing it would penalize decisions that place a disproportionate weight on less likely, albeit highly desirable, outcomes. Practically, the expectation Ex∼D,y∼p0(·|x) can be easily approximated using a sample average from a dataset generated under p0, allowing the dual problem (5) to be numerically solved.

This computational aspect is further explored in Section 2.3. Remark 2 (Choice of Value Palette c). We outline interpretable methods for selecting the value palette c. The first method is the *quantile-based approach*, which utilizes statistical distributions of reward outputs. For any reward function, input prompts x ∈ D, and model-generated outputs y ∼ p0(· | x), the mapping (x, y) 7→ r(*x, y*) defines a random variable, denoted by R(D, p0). The expectation of R(D, p0) is the realized value under the model p0. Users can set each component of c to correspond to a quantile of R(D, p0). For example, for the first value, setting c1 to the 80% quantile positions the realized level of the aligned model p within the upper 20% of expected outcomes. For practical implementation, we propose using a small pilot dataset generated from the original model as Monte Carlo samples to empirically estimate the quantiles of R(D, p0). That is, we sample xi ∼ D, yi ∼ p0(· | xi) to create {r(xi, yi) : i = 1*, . . . , n*} (n = 2000 in our experiments). This pilot data can serve various palette choices. We provide further remarks on two other approaches, one based on classifiers and another based on automatic adjustment in Appendix B.4.

Remark 3 (Robustness of MAP under Mis-specified Rewards). We conducted a theoretical analysis of robustness against mis-specification in Appendix B.1. Our results indicate that MAP remains robust provided that the rewards for multiple values exhibit a non-degenerate covariance structure.

 $\href{https://github.com/}{\text{}}$ REvalJZABI. 
2.2 REALIZABLE VALUE LEVELS OF THE MAP PROBLEM AND PARETO FRONTIER We first show that the MAP problem can be written as the original alignment problem (1) with a particular reward function that is simply a linear combination of individual rewards. Theorem 2 (Solution of MAP). The solution to the MAP problem (3) is the same of the problem (1) with R(*x, y*)
∆= β · λ(c)
Tr(*x, y*), where λ(c) was introduced in Theorem 1.

This prompts the natural question of whether MAP limits the breadth of realizable value levels compared to those achievable under (1) with arbitrarily chosen R. Next, we will show that actually the realizable value levels of (1) and the MAP problem in (3) are the same.

Given the reference distribution p0 and any specific reward functions R, the solution p, if feasible, depends solely on R. To illustrate this dependency, we denote p as pR. Let FRLHF(p0) represent the range of R that admits feasible solutions to the RLHF problem, essentially those with valid probability densities. The realizable value levels under the RLHF problem are defined as:

$$\mathcal{V}_{\text{RLHF}}(p_{0})\stackrel{{\Delta}}{{=}}\left\{\mathbb{E}_{x\sim\mathcal{D},y\sim p_{R}(\cdot|x)}\,\mathbf{r}(x,y):\,R\in\mathfrak{F}_{\text{RLHF}}(p_{0})\right\}.$$

For multiple reward functions r1*, . . . , r*m, we consider a specific class of R comprising various non-negative linear combinations, and define their realizable value levels similarly:

$$\mathcal{V}_{\text{RLDIF}}(r_{1},\ldots,r_{m};p_{0})\triangleq\bigg{\{}\mathbb{E}_{x\sim\mathcal{D},y\sim p_{R}(\cdot|x)}\boldsymbol{r}(x,y):\,R\triangleq\sum_{i=1}^{m}\rho_{i}r_{i}\in\mathfrak{F}_{\text{RLDIF}}(p_{0}),\,\rho_{i}\geq0\bigg{\}}.$$

In the MAP problem, given p0 and r, the solution p, if feasible, depends only on the user-specified value palette c. To emphasize this relationship, we denote p as pc. Let CMAP(r1*, . . . , r*m; p0) denote the range of c that admits feasible solutions to the MAP problem. We further consider the realized value levels of all feasible solutions under various c, defined as:

$\mathcal{V}_{\text{MAP}}(r_{1},\ldots,r_{m};p_{0})\stackrel{{\Delta}}{{=}}\bigg{\{}\mathbb{E}_{x\sim\mathcal{D},y\sim p_{\mathbf{c}}(\,\mid x)}\mathbf{r}(x,y):\,\mathbf{c}\in\mathcal{C}_{\text{MAP}}(r_{1},\ldots,r_{m};p_{0})\bigg{\}}.$
Theorem 3 (Equivalent realizable value levels). For any original generative model p0, we have VMAP(r1*, . . . , r*m; p0) = VRLHF(p0) = VRLHF(r1*, . . . , r*m; p0). (6)
Theorem 3 establishes that the realizable value levels by MAP equate to those in the original alignment problem (1) using a specific reward function - a linear combination of individual rewards. This proves that linear combinations of individual reward functions can sufficiently capture the entire Pareto Frontier. It is crucial to note that the sets of solutions, namely p, are not identical for both problems. The key insight is that the set of realizable value levels, which resides within a finite m-dimensional space, is mapped from the infinitely dimensional set of solutions p through a many-to-one mapping, as depicted in Figure 4.

We denote all feasible value palettes CMAP(r1*, . . . , r*m; p0) simply as C. The following theorem explores the characteristics of this set.

Theorem 4. The following properties of C hold. (i) If c ∈ C, for any c
′such that c
′ ≤ c, we have c
′ ∈ C.

Feasible  from 0 and any Feasible  from 0 and σ=1
 
0 Same set of realizable value levels Figure 4: Illustration of Theorem 3.
(ii) For any c
′such that c
′ ≤ c0
∆= Ex∼D,y∼p0(·|x)r(x, y), the value level of the original model, we have the corresponding λ(c
′) = 0.

Step 1: Specify Value Palette **Step 2**: Check Feasibility Step 3: Align Model Post-alignment Reward distributions Solve Original model MAP reward function Reward distributions Infeasible Feasible
Algorithm 1 MAP Procedure Input: Multi-dimensional reward functions r, original generative model p0(· | x) with x ∼ D.

Step 1: Specify value palette
- Specify c (e.g., based on Remark 2).

Step 2: Check feasibility
- Attempt to solve Problem (7) with updates in (8).

If infeasible: Suggest alternative c (Remark 2). Else: Obtain λ and define reward function R
∆= λ Tr.

Step 3: Align model
- Decoding (D) option: For a prompt x, generate y
(1)*, . . . , y*(m), and sample y = y
(i)using the Softmax of R(*x, y*(i)) as Multinomial probability.

- Finetuning (F) option: Apply Proximal Policy Optimization (PPO) with R to fine-tune p0 into pˆ.

For a prompt x, generate y ∼ pˆ(· | x).

Output: The generated y.

(iii) Let Bi
∆= ess supx∼D,y∼p0(·|x)ri(*x, y*), where the supreme is under the original model p0.

Then, C ⊆ (−∞, B1]× · · · ×(−∞, Bm]. That is, there is no feasible solution if ci > Bi for some i. (iv) Let pλ(y | x)
∆= p0(y | x)e λ Tr(x,y)/Z(λ, x) be the λ-adjusted distribution of p0, where Z(λ, x)
∆=Ry′ p0(y
′| x)e λ Tr(x,y′)dy′is the normalizing constant. Treating the optimal solution λ = λ(c) as a function of c, we have dλ(c)
dc = −V
−1, where V
∆= Varλ(r(*x, y*)) is the covariance matrix of r(*x, y*) under the distribution x ∼ D, y | x ∼ pλ(· | x). The result in (iv) can be used to check if a small increase in c, e.g., changing c1 to c1 + δ for a small δ > 0, will cause λ to change in a way that remains feasible.

## 2.3 Computational Solution To Map Through A Primal-Dual Approach

In this section, we introduce a practical framework to solve MAP problem defined in (3), illustrated in Figure 5 and Algorithm 1. Once a user defines a value palette according to Remark 2, in Step 2, we derive λ from c as per Theorem 1. We then approximate Problem (5) as follows:

$$\operatorname*{max}_{\boldsymbol{\lambda}\geq\mathbf{0}}g_{n}(\boldsymbol{\lambda})\triangleq-\log{\frac{1}{n}}\sum_{j=1}^{n}e^{\boldsymbol{\lambda}^{\mathrm{T}}\boldsymbol{r}(x_{j},y_{j})}+\boldsymbol{\lambda}^{\mathrm{T}}\boldsymbol{c},$$
$$(7)$$

where the dataset {(xj , yj )
n j=1} serves as a finite-sample approximation of the distribution p0(x, y)
with y1*, . . . , y*n, generated conditional on respective prompts x1*, . . . , x*n. Lemma 1 in the Appendix confirms that both the original problem (5) and its approximation (7) are concave, allowing for the use of gradient ascent to efficiently find the optimal λ:

$$\mathbf{\lambda}\leftarrow\mathbf{\lambda}+\alpha\frac{d}{d\mathbf{\lambda}}g(\mathbf{\lambda})=\mathbf{\lambda}+\alpha\big{(}-\text{Softmax}\big{(}\mathbf{\lambda}^{\text{T}}\mathbf{r}_{1:n}\big{)}\cdot\mathbf{r}_{1:n}^{\text{T}}+\mathbf{c}\big{)}\tag{8}$$

where r1:n
∆= [r(x1, y1), . . . , r(xn, yn)]. In practical implementations, we applied a change of variable τ
∆= log λ to ensure the constraints λ ≥ 0 are satisfied. Not all users' palette choices are feasible. If the updates in (8) do not converge, this indicates that the chosen c is infeasible. In such cases, MAP can adjust the value palette c automatically, which we elaborate in Appendix B.2. Also, because the above optimization problem is concave and the data can be pre-computed using a pilot set of Monte Carlo samples from the original model, solving the problem is efficient and does not depend on the complexity of the models to align. More discussions are included in Appendix B.3.

In Step 3, we implement p(y | x) ∝ p0(y | x)e λ Tr(x,y)through two practical approaches during experiments. The first, *decoding-based alignment*, employs importance sampling for decoding from p. Specifically, for any input x, we generate k samples from p0, labeled y1*, . . . , y*k, and select one via Multinomial sampling with weights proportional to exp(λ Tr(*x, y*i)) for each sample i. We include an ablation study on the trade-off between complexity of larger m and enhanced generation quality in Appendix A.4. The second approach, *finetuning-based alignment*, fine-tunes a model pw with neural weights w based on the reward function (*x, y*) 7→ λ Tr(*x, y*). Compared with the first approach, the function space of p is restricted by the model architecture and the training can be computationally intensive. However, it can reduce the inference-stage complexity by only decoding one sentence for each prompt. An experimental study on this comparison is included in Appendix A.5.

## 2.4 Simultaneous Versus Sequential Or Greedy Alignment

To align a model with a specified value palette c, a natural baseline method is sequentially aligning individual values, namely we sequentially update the model to align each value to an entry in c until all values are addressed. This section compares this method with the MAP procedure.

Sequential alignment algorithm. Suppose we have aligned with one value palette c(ℓ−1) and then update with c(ℓ), ℓ = 1, 2*, . . .*. For notational convenience, let the initial c(0) be the realized level of the original model p0. At the end of the ℓ-th alignment, the aligned distribution can be expressed as p(ℓ)(y | x) ∝ p0(y | x) · exp(λ T
(ℓ)
r(*x, y*)) for some vector λ(ℓ).

Given the value palettes {c(ℓ)}ℓ=1,2*,...*, we recursively obtain λ(ℓ)from λ(ℓ−1). Like the problem
(5), the alignment objective at the beginning of the ℓ-th alignment is:

$$\operatorname*{max}_{\boldsymbol{\lambda}\geq\mathbf{0}}g_{(\ell)}(\boldsymbol{\lambda})\stackrel{\Delta}{=}-\log\mathbb{E}_{x\sim\mathcal{D},y\sim p_{(\ell-1)}(\cdot|x)}e^{\boldsymbol{\lambda}^{\mathrm{T}}\boldsymbol{r}(x,y)}+\boldsymbol{\lambda}^{\mathrm{T}}\boldsymbol{c}_{(\ell)}.$$
$$(9)$$
$$(10)$$
Tc(ℓ). (9)
Like the problem (7), we can numerically solve (9) by addressing the problem:

$$\operatorname*{max}_{\boldsymbol{\lambda}\geq\mathbf{0}}g_{(\ell)}(\boldsymbol{\lambda})\triangleq-\log\bigl(\operatorname{Softmax}(\boldsymbol{\lambda}_{(\ell-1)}^{\mathrm{T}}\boldsymbol{r}_{1:n})\cdot e^{\boldsymbol{r}_{1:n}^{\mathrm{T}}\boldsymbol{\lambda}}\bigr)+\boldsymbol{\lambda}^{\mathrm{T}}\boldsymbol{c}_{(\ell)}.$$
Tc(ℓ). (10)
Connection between Sequential and Simultaneous Alignment. The problem (9) establishes a connection between the (ℓ−1)-th and ℓ-th alignment. Recall that m denotes the number of values or the dimension of c. In a canonical setting where one value is aligned at a time, define {c(ℓ)}ℓ=1,2*,...* by: c(ℓ) = [−∞, . . . , cℓ, −∞, . . .], for ℓ ≤ m, where the ℓ-th element is the same as that of c, and the others are trivially negative infinity. For *ℓ > m*, we can re-align the first value, and so on.

Namely, we replace the above cℓ in c(ℓ) with cℓ mod m. The following result shows this sequential, iterative alignment process converges to the joint alignment using MAP.

Theorem 5. Let p(0)
∆= p0, p(1), p(2)*, . . .* be the sequence of distributions obtained by sequentially aligning the original model according to the single-value MAP objective:

$\min_{p\in\mathcal{P}}\mathbb{E}_{x\sim\mathcal{D},y\sim p(\cdot\mid x)}D_{\mathbf{KL}}\big{(}p(\cdot\mid x)\,||\,p_{(\ell-1)}(\cdot\mid x)\big{)}$ s.t. $\mathbb{E}_{x\sim\mathcal{D},y\sim p(\cdot\mid x)}r_{\ell^{\prime}}(x,y)\geq c_{\ell^{\prime}}$,
$\underset{p\in\mathcal{P}}{\min}\;\mathbb{R}$
where ℓ
′ = (ℓ mod m). Assuming r(*x, y*) is not trivially a constant on the support set of (*x, y*), this sequence weakly converges to pλ(c), which is the solution to the MAP problem in (4).

The proof of Theorem 5 demonstrates that the sequential alignment process fundamentally operates as a coordinate-wise gradient ascent algorithm. When we align subsets of values in each iteration, the process can equivalently function as a block-wise ascent. Importantly, while a single cycle of sequential alignments - aligning each value only once - may not fully achieve joint alignment, multiple cycles ensure convergence. This aspect is particularly significant in scenarios where constraints such as limited GPU memory restrict the simultaneous loading of numerous reward models. Our finding confirms that sequential alignment, when executed over a sufficient number of cycles, can effectively accommodate these memory and computational limitations.

(a) Multi-value Palette Alignment
(b) Single-value Palette Alignment Humor Humor Perplexity Helpful Perplexity Helpful 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 Diversity Coherence Harmless Diversity Coherence Harmless Palette Original model MAP to HHH-50% MAP to HHH-60%
MAP to HHH-70%

Palette Original model MAP to Humor-80%
MAP to Helpfulness-80% MAP to Harmlessness-80%

## 3 Experimental Study 3.1 Experimental Setup

We generate prompts from two data sources: Anthropic harmless data (Bai et al., 2022), which includes human requests delineated between the tags "Human:" and "Assistant:", and IMDB data (Maas et al., 2011) from which we retain movie reviews exceeding 30 characters in length. For backbone models, we employ OPT-1.3B (Zhang et al., 2022) and Llama2-7B-chat (Touvron et al., 2023), which have demonstrated robust language modeling capabilities in previous assessments. We focus on several values for alignment: Humor, Positiveness, Harmlessness, Helpfulness, Diversity, Coherence, and Perplexity. The Humor reward is assessed using the logits from a humor detection model (Dhiab, 2023), while Positiveness uses a DistilBERT model trained on IMDB sentiment analysis (Lvwerra, 2021). Harmlessness and Helpfulness are evaluated through two GPT-2 models equipped with a value head fine-tuned to predict these attributes (Yang et al., 2024). Diversity is measured by the lexical variety within sentences, calculated through the proportion of unique n-grams (n = 2, 3, 4) and their composite score (Zhang et al., 2020). Coherence is determined by the semantic similarity of sentences within a context, using a supervised SimCSE BERT-based model that captures sentence embeddings to assess textual coherence (Gao et al., 2021).

## 3.2 Effectiveness Of Map For Simultaneous Multi-Value Alignment

For the conversational task, we use Anthropic data as input prompts x to align the OPT-1.3B model across six dimensions: Humor, Harmlessness, Helpfulness, Diversity, Coherence, and Perplexity. Here, the perplexity is calculated as negative log of the standard perplexity so that the larger the better. We tested seven different value palettes for alignment, considering both decoding-based and finetuning-based implementations. For example, the HHH-80% palette aligns the first three values to the 80th percentile of their respective distributions, while maintaining the last three values. This approach aims to enhance the model's output to closely resemble human-like interaction standards without compromising the intrinsic qualities of the generated content. The results are summarized in Table 1. The first three rows show the effectiveness of the MAP approach in simultaneously enhancing multiple values to various levels. In comparison, the last three palettes, which focus on aligning a single value, typically enhance that specific value but may degrade others. The results are also visualized in radar plots in Figure 6, where all values are transformed to quantiles under the reward distribution of the original model. Additional experiments, including an ablation study with the Llama2-7B-chat model, are detailed in Appendix A.1. Notably, the HHH-80% palette was determined to be feasible by Algorithm 1 Step 2 and its results are therefore included. This indicates that the Llama2-7B-chat model, which has a larger complexity than OPT-1.3B, allows for more extensive multi-value alignment. We also explored a sentiment-controlled open generation task using a random trunk of IMDB data as input prompts to align the OPT-1.3B model. The results are summarized in Table 2.

## 3.3 Comparison With Other Baseline Approaches

To demonstrate the effectiveness of MAP in reliably aligning multiple values, we conduct an experiment showing that MAP can identify desirable outcomes with randomly sampled λ. According to

Table 1: Results of aligning the OPT-1.3B model with Anthropic conversational data using diverse value

palettes, evaluated in terms of expected rewards (or realized value levels). Scenarios include: 1) enhancing the

first three values - Humor, Harmlessness, Helpfulness - to the xth percentile of their respective reward distributions under the original model ("HHH-x%") while maintaining the last three values - Diversity, Coherence, and Perplexity - using MAP, and 2) aligning individual values independently ("Humor-80%", "Helpfulness-80%",

"Harmlessness-80%"). Results include the optimized weight vector λ, targeted value palettes, and realized value levels. Both decoding-based approach ("MAP-D") and finetuning-based approach ("MAP-F") are used. The palette "HHH-80%" was found infeasible, as confirmed in Step 2 of Algorithm 1.

Humor↑ Helpfulness↑ Harmlessness↑ Diversity↑ Coherence↑ **Perplexity**↑

Original model 2.07 -1.47 0.25 0.88 0.43 -3.34 HHH-50% (λ = [2.53, 0.23, 0.28, 0.02, 0.05, 0.05]) MAP-D **2.44 -1.38** 0.21 **0.88 0.43 -3.17** MAP-F **2.20** -1.80 **0.66** 0.86 0.41 **-2.80**

HHH-60% (λ = [6.30, 0.83, 0.93, 0.01, 0.03, 0.03])

MAP-D **2.48 -1.33 0.48 0.88 0.43 -3.15** MAP-F **2.47** -2.26 **0.50 0.89** 0.26 -3.49 HHH-70% (λ = [12.77, 1.53, 1.69, 0.01, 0.02, 0.02]) MAP-D **2.49 -1.29 0.66 0.88 0.45 -3.14** MAP-F **2.17** -2.28 **0.97** 0.82 0.10 -5.04 Humor-80% (λ = [16.44, −, −, −, −, −]) MAP-D **2.52 -1.42** 0.01 **0.89 0.43 -3.21** MAP-F **2.08** -2.49 **0.39** 0.79 0.09 -5.85 Helpfulness-80% (λ = [−, 0.72, −, −, −, −]) MAP-D 1.99 **-0.75** -0.35 **0.88 0.43 -3.20** MAP-F 2.02 **-0.66** -0.58 **0.88** 0.41 **-2.73** Harmlessness-80% (λ = [−, −, 1.27, −, −, −]) MAP-D 1.97 -1.86 **0.97 0.88** 0.42 **-3.17** MAP-F 2.05 -2.02 **0.94** 0.87 0.40 **-2.63**

Theorem 4, we sample λ randomly from the range (c0, B) and retain the feasible ones according to MAP's feasibility check. From these, we select λ vectors with a bounded ℓ1-norm less than 6. We compare this with a standard MORL approach where λ is randomly generated from s ·u, where s is

| Positiveness↑                                                       | Helpfulness↑   | Harmlessness↑   | Diversity↑   | Coherence↑   | Perplexity↑   |       |
|---------------------------------------------------------------------|----------------|-----------------|--------------|--------------|---------------|-------|
| Original model                                                      | 0.52           | -1.53           | 0.58         | 0.88         | 0.24          | -3.36 |
| PHH-50% (λ = [0.24, 0.08, 0.12, 0.03, 0.07, 0.07]) MAP-D 0.62 -1.40 | 0.57           | 0.88            | 0.24         | -3.19        |               |       |
| MAP-F                                                               | 0.55           | -1.50           | 0.49         | 0.88         | 0.23          | -2.80 |
| PHH-60% (λ = [2.23, 0.41, 0.79, 0.01, 0.03, 0.04]) MAP-D 0.91 -1.16 | 0.67           | 0.89            | 0.25         | -3.13        |               |       |
| MAP-F                                                               | 0.88           | -0.69           | 0.38         | 0.87         | 0.24          | -2.46 |
| PHH-70% (λ = [3.83, 0.90, 1.48, 0.01, 0.02, 0.03]) MAP-D 0.93 -1.10 | 0.74           | 0.89            | 0.24         | -3.13        |               |       |
| MAP-F                                                               | 0.94           | -0.33           | 0.24         | 0.80         | 0.18          | -2.64 |
| PHH-80% (λ = [9.77, 1.42, 2.27, 0.00, 0.01, 0.03]) MAP-D 0.93 -1.06 | 0.64           | 0.89            | 0.25         | -3.14        |               |       |
| MAP-F                                                               | 0.95           | -1.22           | 0.24         | 0.73         | 0.18          | -2.99 |
| Positiveness-80% (λ = [10.98, −, −, −, −, −]) MAP-D 0.94 -1.10      | 0.47           | 0.89            | 0.25         | -3.17        |               |       |
| MAP-F                                                               | 0.92           | -1.27           | 0.37         | 0.73         | 0.17          | -3.00 |
| Helpfulness-80% (λ = [−, 0.95, −, −, −, −]) MAP-D 0.56 -0.86        | 0.29           | 0.89            | 0.23         | -3.17        |               |       |
| MAP-F                                                               | 0.55           | -1.28           | 0.34         | 0.88         | 0.23          | -2.83 |
| Harmlessness-80% (λ = [−, −, 1.43, −, −, −]) MAP-D 0.53 -1.80       | 1.21           | 0.89            | 0.24         | -3.17        |               |       |
| MAP-F                                                               | 0.47           | -1.77           | 0.91         | 0.89         | 0.23          | -2.62 |

uniformly sampled from (0, 6) and u is uniformly sampled from the probability simplex. For both approaches, we implement two alignment strategies: For the decoding-based approach ("MAP-D", "MORL-D"), we generate 16 candidates of y for each prompt x and select the final output using Multinomial sampling as described in Algorithm 1 Step 3. For the finetuning-based approach ("MAP-F", "MORL-F"), we apply PPO and the calculated MAP reward function R to fine-tune the original model. We evaluate the outcomes using the "test" split of our task data, assessing both the expected reward and win rate against the original model based on the reward functions. Additionally, we compare MAP with the DPO approach that uses various mixtures of preference data, and DPO-Soup, which applies linear interpolation of DPO-trained, value-specific models. The results, presented in Figure 7, show that MAP generates alignment results that tend to fall into, or remain close to, the desirable regime, namely the upper-right quadrant relative to the original model. In contrast, the other approaches experience more severe trade-offs and rarely fall into the desirable regime. Table 3 quantifies this comparison by calculating the frequency of alignment results falling within the desirable regime, termed Navigation Efficiency (NE).

Win Rate against the Original Model Expected Reward (aka. Realized Value Level)
MAP-D (random )
MAP-F (random )
MORL-D (random )
MORL-F (random ) DPO( = 0.1)
DPO( = 0.5) DPO-Soup( = 0.1) DPO-Soup( = 0.5) Original model 0.0 0.2 0.4 0.6 0.8 1.0 Helpfulness 0.0 0.2 0.4 0.6 0.8 1.0 3.0 2.5 2.0 1.5 1.0 0.5 0.0 0.5 Helpfulness 2.0 1.5 1.0 0.5 0.0 0.5 1.0 1.5 2.0 2.5 H
arm less ne ss H
ar m less ne ss MAP-D (random ) MAP-F (random )
MORL-D (random ) MORL-F (random ) DPO( = 0.1) DPO( = 0.5) DPO-Soup( = 0.1) DPO-Soup( = 0.5) Original model

| Table 3: Navigation Efficiency (NE) of expected reward and win rate across alignment methods. MAP-D MAP-F MORL-D MORL-F DPO(0.1) DPO(0.5) DPO-Soup(0.1) DPO-Soup(0.5)   |     |     |     |     |     |    |     |     |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----|-----|-----|-----|-----|----|-----|-----|
| NE of expected reward                                                                                                                                                   | 55% | 52% | 21% | 35% | 27% | 9% | 45% | 36% |
| NE of win rate                                                                                                                                                          | 60% | 70% | 21% | 55% | 36% | 9% | 45% | 45% |

## 4 Conclusion

The proposed MAP offers a structured approach to aligning multiple human values, enabling precise adjustments to meet diverse user preferences. Through a blend of theoretical insights and practical algorithms, MAP ensures that the alignment is aimed at achieving Pareto improvement with userdefined preference levels. This approach holds potential to positively impact fields that involve complex decision-making, such as public health and digital content creation, by ensuring AI interactions more accurately reflect individual values and preferences. Future work will explore extending MAP to directly calculate empirical risk using a mix of data sources, each representing different values. The **Appendix** includes further discussions, additional experiments, and all corresponding proofs.

## Acknowledgement

The work was supported in part by the 3M Science and Technology Graduate Fellowship, the Samsung Global Research Outreach Award, the Army Research Office Early Career Program under grant number W911NF2310315, and National Science Foundation CAREER Program under grant number 2338506.

## References

Dilip Arumugam, Jun Ki Lee, Sophie Saskin, and Michael L Littman. Deep reinforcement learning from policy-dependent human feedback. *arXiv preprint arXiv:1902.04257*, 2019.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, and Tom Henighan. Training a helpful and harmless assistant with reinforcement learning from human feedback. *arXiv preprint arXiv:2204.05862*, 2022.

Leon Barrett and Srini Narayanan. Learning all optimal policies with multiple criteria. In Proceedings of the 25th international conference on Machine learning, pp. 41–47, 2008.

Pier Giovanni Bissiri, Chris C Holmes, and Stephen G Walker. A general framework for updating belief distributions. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*,
78(5):1103–1130, 2016.

Stephen Boyd and Lieven Vandenberghe. *Convex optimization*. Cambridge university press, 2004. Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. *Biometrika*, 39(3/4):324–345, 1952.

Regina Sandra Burachik, C Yalc¸in Kaya, and MM Rizvi. A new scalarization technique to approximate pareto fronts of problems with disconnected feasible sets. *Journal of Optimization Theory* and Applications, 162:428–446, 2014.

Josef Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo Xu, Mickel Liu, Yizhou Wang, and Yaodong Yang. Safe RLHF: Safe reinforcement learning from human feedback. In The Twelfth International Conference on Learning Representations, 2024.

Mohamed Dhiab. Humor no humor model, 2023. URL https://huggingface.co/
mohameddhiab/humor-no-humor. Accessed on July 5, 2024.

Jie Ding, Vahid Tarokh, and Yuhong Yang. Model selection techniques: An overview. IEEE Signal Processing Magazine, 35(6):16–34, 2018.

Pierre Dognin, Jesus Rios, Ronny Luss, Inkit Padhi, Matthew D Riemer, Miao Liu, Prasanna Sattigeri, Manish Nagireddy, Kush R Varshney, and Djallel Bouneffouf. Contextual moral value alignment through context-based aggregation. *arXiv preprint arXiv:2403.12805*, 2024.

Iason Gabriel. Artificial intelligence, values, and alignment. *Minds and machines*, 30(3):411–437, 2020.

Tianyu Gao, Xingcheng Yao, and Danqi Chen. Simcse: Simple contrastive learning of sentence embeddings. *Conference on Empirical Methods in Natural Language Processing*, 2021.

Senay A Gebreab, Khaled Salah, Raja Jayaraman, Muhammad Habib ur Rehman, and Samer Ellaham. Llm-based framework for administrative task automation in healthcare. In *International* Symposium on Digital Forensics and Security (ISDFS), pp. 1–7. IEEE, 2024.

Shane Griffith, Kaushik Subramanian, Jonathan Scholz, Charles L Isbell, and Andrea L Thomaz.

Policy shaping: Integrating human feedback with reinforcement learning. Advances in Neural Information Processing Systems, 26, 2013.

Chris C Holmes and Stephen G Walker. Assigning a value to a power likelihood in a general bayesian model. *Biometrika*, 104(2):497–503, 2017.

Haozhe Ji, Pei Ke, Hongning Wang, and Minlie Huang. Language model decoding as direct metrics optimization. *International Conference on Learning Representations*, 2024.

Maxim Khanov, Jirayu Burapacheep, and Yixuan Li. ARGS: Alignment as reward-guided search.

The Twelfth International Conference on Learning Representations, 2024.

Hannah Rose Kirk, Bertie Vidgen, Paul Rottger, and Scott A Hale. The benefits, risks and bounds of ¨
personalizing the alignment of large language models to individuals. *Nature Machine Intelligence*, pp. 1–10, 2024.

Saydulu Kolasani. Optimizing natural language processing, large language models (llms) for efficient customer service, and hyper-personalization to enable sustainable growth and revenue. Transactions on Latest Trends in Artificial Intelligence, 4(4), 2023.

Kaiwen Li, Tao Zhang, and Rui Wang. Deep reinforcement learning for multiobjective optimization.

IEEE transactions on cybernetics, 51(6):3103–3114, 2020.

Dinh The Luc, Thai Quynh Phong, and Michel Volle. Scalarizing functions for generating the weakly efficient solution set in convex multiobjective problems. *SIAM Journal on Optimization*, 15(4):987–1001, 2005.

Lvwerra. Distilbert for imdb sentiment analysis, 2021. URL https://huggingface.co/
lvwerra/distilbert-imdb. Accessed on July 5, 2024.

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pp. 142–150, Portland, Oregon, USA, June 2011. Association for Computational Linguistics. URL http: //www.aclweb.org/anthology/P11-1015.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Thirtyseventh Conference on Neural Information Processing Systems, 2023.

Alexandre Rame, Guillaume Couairon, Corentin Dancette, Jean-Baptiste Gaya, Mustafa Shukor, Laure Soulier, and Matthieu Cord. Rewarded soups: towards pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards. *Thirty-seventh Conference on Neural Information* Processing Systems, 2023.

Ralph E Steuer and Eng-Ung Choo. An interactive weighted tchebycheff procedure for multiple objective programming. *Mathematical programming*, 26:326–344, 1983.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023.

Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan Thrush, Nathan Lambert, Shengyi Huang, Kashif Rasul, and Quentin Gallouedec. Trl: Transformer reinforcement ´
learning. https://github.com/huggingface/trl, 2020.

Ganghua Wang, Xun Xian, Jayanth Srinivasa, Ashish Kundu, Xuan Bi, Mingyi Hong, and Jie Ding.

Demystifying poisoning backdoor attacks from a statistical perspective. International Conference on Learning Representations, 2024.

Xinran Wang, Enmao Diao, Qi Le, Jie Ding, and Ali Anwar. AID: Adaptive integration of detectors for safe ai with large language models. In Proceedings of the Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2025.

Zeqiu Wu, Yushi Hu, Weijia Shi, Nouha Dziri, Alane Suhr, Prithviraj Ammanabrolu, Noah A Smith, Mari Ostendorf, and Hannaneh Hajishirzi. Fine-grained human feedback gives better rewards for language model training. *Advances in Neural Information Processing Systems*, 36, 2024.

X. Xian, G. Wang, X. Bi, R. Zhang, J. Srinivasa, A. Kundu, Fleming C., , M. Hong, and J. Ding.

On the vulnerability of retrieval-augmented generation within knowledge-intensive application domains. *arXiv preprint arXiv:2409.17275*, 2025.

Xun Xian, Ganghua Wang, Jayanth Srinivasa, Ashish Kundu, Xuan Bi, Mingyi Hong, and Jie Ding.

Understanding backdoor attacks through the adaptability hypothesis. In International Conference on Machine Learning, pp. 37952–37976. PMLR, 2023a.

Xun Xian, Ganghua Wang, Jayanth Srinivasa, Ashish Kundu, Xuan Bi, Mingyi Hong, and Jie Ding.

A unified detection framework for inference-stage backdoor defenses. Advances in Neural Information Processing Systems, 36:7867–7894, 2023b.

Rui Yang, Xiaoman Pan, Feng Luo, Shuang Qiu, Han Zhong, Dong Yu, and Jianshu Chen. Rewardsin-context: Multi-objective alignment of foundation models with dynamic preference adjustment. International Conference on Machine Learning, 2024.

Jiawei Zhang, Yuhong Yang, and Jie Ding. Information criteria for model selection. Wiley Interdisciplinary Reviews: Computational Statistics, 15(5):e1607, 2023.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. Opt: Open pre-trained transformer language models, 2022.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with bert. *International Conference on Learning Representations*, 2020.

Zhanhui Zhou, Jie Liu, Chao Yang, Jing Shao, Yu Liu, Xiangyu Yue, Wanli Ouyang, and Yu Qiao.

Beyond one-preference-for-all: Multi-objective direct preference optimization. Findings of the Association for Computational Linguistics ACL, 2024.

## A Additional Experimental Results A.1 Ablation Study With Larger Model

Adopting the same experimental framework as described in Subsection 3.2, we substituted the OPT- 1.3B model with the Llama2-7B-chat model. The results are summarized in Table 4. Due to memory constraints of our available GPU resources, we are limited to decoding-stage alignments (Step 3 in Algorithm 1) and could not perform finetuning on this model.

In these experiments, the HHH-80% palette is confirmed as feasible by Step 2 of Algorithm 1, and its results are therefore included. This finding suggests that the Llama2-7B-chat model, which is more complex than the OPT-1.3B, provides greater flexibility for multi-value alignment. To facilitate a clearer comparison, we visualized the results in radar plots presented in Figure 8. All values are converted to quantiles based on the reward distributions of the original model. Table 4: Results of aligning the Llama2-7B-chat model with the Anthropic prompt data using diverse value palettes, evaluated in terms of expected rewards (or realized value levels). Scenarios include: 1) enhancing the first three values - Humor, Harmlessness, Helpfulness - to the xth percentile of their respective reward distributions under the original model ("HHH-x%") while maintaining the last three values - Diversity, Coherence, and Perplexity - using MAP, and 2) aligning individual values independently ("Humor-80%", "Helpfulness-80%", "Harmlessness-80%"). Results include the optimized weight vector λ, targeted value palettes, and realized value levels. Notably, in contrast with Table 1, the "HHH-80%**" palette was determined to be feasible** by Algorithm 1 Step 2 and is therefore included.

| Humor↑                                                                   | Helpfulness↑   | Harmlessness↑   | Diversity↑   | Coherence↑   | Perplexity↑   |       |
|--------------------------------------------------------------------------|----------------|-----------------|--------------|--------------|---------------|-------|
| Original model                                                           | 0.60           | -1.01           | 1.25         | 0.85         | 0.52          | -1.38 |
| HHH-50% (λ = [0.27, 0.20, 0.21, 0.10, 0.04, 0.09]) MAP-D 0.98 -1.06 1.36 | 0.84           | 0.53            | -1.37        |              |               |       |
| HHH-60% (λ = [0.85, 0.78, 0.79, 0.01, 0.02, 0.07]) MAP-D 1.56 -0.93 1.47 | 0.84           | 0.53            | -1.37        |              |               |       |
| HHH-70% (λ = [2.02, 1.39, 1.50, 0.01, 0.02, 0.09]) MAP-D 2.01 -0.87 1.57 | 0.83           | 0.55            | -1.37        |              |               |       |
| HHH-80% (λ = [5.94, 2.43, 2.92, 0.01, 0.01, 0.15]) MAP-D 2.17 -0.93 1.68 | 0.82           | 0.55            | -1.37        |              |               |       |
| Humor-80% (λ = [2.89, −, −, −, −, −]) MAP-D 2.21 -1.26                   | 1.06           | 0.82            | 0.55         | -1.47        |               |       |
| Helpfulness-80% (λ = [−, 0.69, −, −, −, −]) MAP-D 0.50 -0.39             | 0.94           | 0.86            | 0.53         | -1.36        |               |       |
| Harmlessness-80% (λ = [−, −, 0.99, −, −, −]) MAP-D 0.47 -1.33            | 1.91           | 0.86            | 0.50         | -1.34        |               |       |

(a) Multi-value Palette Alignment Humor Perplexity Helpful 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 Diversity Coherence Harmless Palette Original model MAP to HHH-50% MAP to HHH-60%
MAP to HHH-70% MAP to HHH-80%
(b) Single-value Palette Alignment Humor Perplexity Helpful 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 Diversity Coherence Harmless Palette Original model MAP to Humor-80%
MAP to Helpfulness-80% MAP to Harmlessness-80%

## A.2 Simultaneous Versus Sequential Alignment

To corroborate Theorem 5 in Section 2.4, we have conducted experiments comparing the MAP with a sequential alignment strategy, under the same experimental settings as described in Section 3.2. Specifically, we align the OPT-1.3B model for a conversational task using Anthropic data. We implement sequential alignment through one and five rounds. In each round, each of the six values is aligned sequentially using the MAP approach. We then numerically evaluate the expected rewards, or realized value levels, using data generated from the aligned models. The results, summarized in Figure 9, demonstrate that while sequential alignment with one round is less effective than MAP,
extending the process to five rounds significantly improves performance and closely approximates the outcomes achieved by MAP.

Comparison of Simultaneous (MAP) and Sequential Alignments MAP Seq Round 1 Seq Round 5 50% 60% 70% 50% 60% 70% 50% 60% 70% 50% 60% 70% 50% 60% 70% 50% 60% 70%
Humor Harmlessness Helpfulness Diversity Coherence Perplexity 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 Alig ned c
-leve ls 
(in qua ntil e)

## A.3 Hyperparameters In Data Generation And Model Training

Our experiments were conducted using a single Nvidia A100 GPU. For data generation, we employed a top-k decoding approach with a fixed k = 50 and a limit of 50 new tokens per sequence. In terms of model finetuning, we utilized the TRL package (von Werra et al., 2020) for DPO and PPO training. Specifiaclly, for DPO, we used an effective batch size of 20, achieved by setting the batch size to 1 with an accumulation step of 20, over the course of a single training epoch. For PPO,
the finetuning was executed with a learning rate of 10−6and similarly limited to one epoch. All other configuration parameters followed the default settings provided in the TRL package.

## A.4 Generation Quality Versus Computation In The Decoding-Based Approach

The sample size m in the decoding option directly influences both the computational cost and the performance of the aligned model. Increasing m improves the approximation of the desired distribution, leading to better realized values. However, this improvement comes at the expense of increased computation and latency. To investigate this trade-off, we conducted an experiment comparing realized value levels and runtime (in minutes) for decoding-based alignment with varying Monte Carlo samples (k). The results, summarized in Table 5, reveal that increasing k significantly improves performance up to k = 8, with diminishing returns beyond this point. Notably, k = 8 achieves realized values nearly equivalent to k = 16, indicating it as a practical choice for balancing computational efficiency and performance.

## A.5 Runtime Comparison Between Finetuning- And Decoding-Based Alignment

Efficient runtime is crucial when selecting between decoding and fine-tuning approaches for value alignment, especially as the number of generations and model sizes scale. To compare these methods, we conducted an experiment to evaluate their runtime and associated trade-offs, as summarized in Table 6. The results highlight several key points: Firstly, for the decoding-based approach, runtime scales approximately linearly with the number of generations, as each generation requires a separate forward pass through the model. In contrast, for Table 5: Comparison of realized value levels and runtime (in minutes) for decoding-based alignment with varying numbers of Monte Carlo samples (k) per prompt. Experiments were conducted for 1000 generations using Llama2-7B-chat under the same settings as described in Appendix A.1, on a single A100 GPU. For clarity, we subtracted the realized value levels at k = 16 from those at each k to provide a reference point. Each value is averaged over 3 repetitions, with a standard error of realized levels within 0.02 and standard error of runtime within 0.5 minutes.

| k   | Runtime↓   | Humor↑   | Helpfulness↑   | Harmlessness↑   | Diversity↑   | Coherence↑   | Perplexity↑   |
|-----|------------|----------|----------------|-----------------|--------------|--------------|---------------|
| 2   | 12.90      | -0.15    | -0.09          | -0.42           | -0.00        | -0.02        | -0.04         |
| 4   | 15.87      | -0.05    | -0.02          | -0.28           | -0.00        | 0.01         | 0.01          |
| 8   | 20.70      | -0.01    | -0.05          | -0.05           | -0.01        | -0.01        | 0.00          |
| 16  | 32.07      | 0.00     | 0.00           | 0.00            | 0.00         | 0.00         | 0.00          |

the PPO-based finetuning approach, the runtime does not increase significantly when scaling from 100 to 2000 generations. This is because the majority of the runtime cost is incurred during the initial model training, which is independent of the number of generations. Secondly, following the above point, for a small number of generations (e.g., n = 100), decoding is more favorable as it avoids the overhead of training. However, for a large number of generations (e.g., n = 2000), finetuning becomes more efficient because it incurs a one-time training cost, and subsequent generations require only a single sample instead of 16 Monte Carlo generations, significantly reducing the per-generation cost. This trade-off makes finetuning more practical for scenarios with high-generation demands, such as serving customers in real-world applications. Thirdly, larger models such as LLaMA2-7B-chat incur higher decoding costs due to the increased computational requirements for each forward pass. Lastly, aligning a single value, such as Humor, has the smallest runtime cost due to the smaller size of its reward model compared to the larger and more complex reward models for Helpfulness and Harmlessness. Also, aligning all six values simultaneously takes less time than the combined runtime of aligning three individual values sequentially. This is because aligning all values in a single process minimizes overhead costs, such as model loading and batch preparation, which are repeated when aligning values individually. Table 6: Runtime (in minutes) comparison for two basemodels (Llama2-7B-chat and OPT-1.3B)
and two generation strategies (Decoding and Finetuning). Results are shown for different numbers of values to align and numbers of generations (n). Experiments were conducted using the same settings as in Appendix A.1 and were run on a single A100 GPU. Each result is averaged over 3 repetitions, with a standard error within 2.5.

## A.6 Sample Generations

| Values to align   | n      | Llama2-7B-chat (D)   | OPT-1.3B (D)   | OPT-1.3B (F)   |
|-------------------|--------|----------------------|----------------|----------------|
| All 6 values      | 100    | 7.95                 | 4.25           | 27.16          |
| 2000              | 160.21 | 107.32               | 29.54          |                |
| Humor only        | 100    | 4.45                 | 2.32           | 27.91          |
| 2000              | 90.90  | 37.87                | 28.50          |                |
| Helpfulness only  | 100    | 4.95                 | 2.66           | 29.45          |
| 2000              | 106.92 | 50.60                | 31.52          |                |
| Harmlessness only | 100    | 5.61                 | 2.69           | 31.23          |
| 2000              | 107.33 | 51.64                | 36.67          |                |

With the same experimental setting as in Section 3.2, Table 7 provides a snapshot of generated responses. It showcases how different value palettes of the OPT-1.3B models lead to varied generation outcomes.

Table 7: Comparative display of generated responses from original and aligned OPT-1.3B models under various value palettes, based on Anthropic conversational data.

Prompt: "Can you tell me some information about the culture and lifestyle of living in Hollywood or West LA area?" Original model: "is it a typical suburban family. Are most people actually interested in going to school or do things like working full time? Is the cost of living prohibitive or is it still very low? I've been living in an area called Sherman Oaks for" Humor-80% model: "Is it really as sketchy as these guys claim? I'm from Los Angeles and yes, it is. Stay away from hollywood and you'll be fine." Helpfulness-80% model: "I'm a new resident & would like to know more about the community. Culturally it's great, because it's on the east coast where everybody knows everybody. So being able to hang out with people who grew up in the culture is" Harmlessness-80% model: "I know you have a lot of fans who have moved to LA or California itself because of their love for your work. What was your first reaction when you first learned Hollywood's reputation? As I've said, the reputation" HHH-80% model: "It's in the heart of entertainment, and if you're interested in acting, it's one of the top 5% best paying cities. I'm just trying to get a sense for where to move to get a job making cool money." Prompt: "I gave this homeless man some food that I had. It had been in the fridge for over a week." Original model: "I took just a bite out of it and it went down easily. The homeless guy then proceeded to throw the sandwich face down on the ground and take a nap." Humor-80% model: "He took me to see the beach. I also gave a homeless man some food that I had. He said it was ok. Was it food that you had just given to another homeless man?" Helpfulness-80% model: "He looked really mad at me for bringing the leftovers to his "donation" of food. It's okay you did the right thing." Harmlessness-80% model: "He was hungry. I said, 'Here. Take this.' And he took some from my bag and put it on his shirt. I then bought him a Coke. Did you follow him into the alley and make it" HHH-80% model: "We were out of bread. I was a little surprised at the kindness of it. He gave me a fist bump and said that it was going to be his. He seemed very grateful. It was a kind gesture." Prompt: "Give me a challenge" Original model: "Well, first time I was in, I could find the last one in ten minutes. So with no experience (which was great), that was easy. No way. Im good at the first one cause its the one with the least hp" Humor-80% model: "to my ability, not an incentive to try hard. I like it when I am rewarded for winning games/modding/etc... not for trying *so hard* as if I'm a good guy. It's a fun challenge but" Helpfulness-80% model: "and I'll give you a prize. Dinner is a great occasion for a good challenge. In our home, dinner is the great challenge because you give it, yet it's so simple and takes few minutes!"
Harmlessness-80% model: "- how to become a teacher. The challenge is how to choose the best teacher and the best profession. Not really. When you think about it, education is a commodity for those who want to use it to make a living. There are" HHH-80% model: "or I'll just take out that trash. There needs to be a challenge. You're right. My question is what kind of challenge? A challenge that the game gives me, not just ok this is the new challenge just try" Prompt: "Is it ok to tell my company that my job sucks and the AI I have to use is absolute rubbish?" Original model: "i've been using it for nearly a year now. Am I being petty for the fact that I actually have a decent reason to tell them that shit? Your answer is yes, you're being petty. Good on you. You're making" Humor-80% model: "yes, there are always exceptions to every rule, don't do it though." Helpfulness-80% model: "They're a large, reputable organisation, and I've mentioned things there that have absolutely killed their stock price" Harmlessness-80% model: "It's not ok to tell your company that your job sucks and the AI you have to use is absolute rubbish. Unless it's not your job, that is? Then, you can tell all the managers you want to. It's not" HHH-80% model: "I'm genuinely considering it but they would laugh my head off Depends. I only work on 5 machines a week or so, is it weird to ask if it's alright?" Prompt: "How can I keep tabs on my spouse's whereabouts?" Original model: "is there a ""where are my spouse's"" app/site is there a 'where are my spouse' app/site My husband and I have been married 2 1/2" Humor-80% model: "DEAR MISS MANNERS: I'm a 26-year-old single male. It's a long story, but my wife and I have been married for 10 years and are currently living in a city where we both grew up." Helpfulness-80% model: "There are several ways in which you can follow-up on your spouse's whereabouts and the various activities that they're engaged in. You can, for example, use the phone or the Internet to locate them when they've left the house." Harmlessness-80% model: "The spouse's location feature on the Vantage Mobile and Tablet apps may help: Open the application on an iPhone or Android-enabled device Tap or scroll down the navigation pane to the Location menu Choose Locations and click" HHH-80% model: "Is there a system I can set-up or a separate app? Can you give me a rough description of yourself. My wife lives in Australia and is the only way contact. The app for our phone is fine but I wanted some type of" Prompt: "There are a lot of people who won't take the COVID-19 vaccine. Do you have any thoughts on why?" Original model: "You've heard of anti-virus vaccines. You've probably heard of COVID-19 vaccines which may work either because of viral vector technology - when a harmless virus is used to deliver a vaccine - or because" Humor-80% model: "I think a lot of them just see it as a big government experiment. They're worried about going into a government-run camp. When I hear them saying this, they kind of sound like the same people that don't want to get" Helpfulness-80% model: "We are on the cusp of a public health crisis, like our predecessors have never seen. And, as so often happens, many of us in our social bubble and on the edge of despair - and our elected leaders, are not responding." Harmlessness-80% model: "We're committed to truth and accuracy in all of our journalism. Read our editorial standards." HHH-80% model: "I think every American should get the COVID-19 vaccine. I had a shot about a month ago, and I did it because I thought that it was the best way to protect myself and my family. One of the things I like" Prompt: "There was a really good singer at the Giant's game last night. Do you know their name?" Original model: "Ya, I think it was Marky Mark. Had a few years ago and still had a big hit going. Him and his wife. I think his wife is with the Rangers now." Humor-80% model: "Yeah! I thought he was the man, that was so awesome. His name was Sami. I agree with you. Great gig. I'm going to see them again in a couple of weeks." Helpfulness-80% model: "I was at the game and that was the singer that was playing. Sorry I don't have her song up on the internet but I'll check tomorrow" Harmlessness-80% model: "I'm sorry! We don't really do names to the point where you'd need to look them up. We don't want to embarrass them or anything but I'll make sure to check it out myself before next week's game and add it" HHH-80% model: "Not off the top of my head, but the singer at the Giants game last week was a woman and she's just one of the best in the league. If I find her name and can't recall, I'll let you"

## B Additional Discussions B.1 Perturbation Analysis Of Map Under Mis-Specified Rewards

To address the concern regarding noisy or mis-specified reward functions, we analyze the sensitivity of the dual variable λ and the aligned distribution p to both small and infinitesimal perturbations in the rewards r.

## B.1.1 Worst-Case Analysis For Small Perturbations Of Rewards

Assume the operational reward r deviates from the true reward r0. Our goal is to derive bounds on the deviation of the dual variable λ and the aligned distribution p. Let λ0 denote the MAP solution under the imperfect reward r0, given any feasible value palette c. For any function h : (x, y) 7→ |h(x, y)|, we define the sup norm of h to be ∥h∥∞
∆= ess supx∼D,y∼p0(·|x) h(*x, y*). Let ∥v∥2 and
∥v∥∞ denote the usual ℓ2-norm and ℓ∞-norm (or sup norm) of any vector v, respectively.

Theorem 6 (Deviation Analysis of MAP results under Perturbations of Rewards). Assume r satisfies ∥r − r0∥∞ ≤ δ, and ∥r0∥∞ ≤ B0. Then, for any given feasible value palette c, we have
(i) Effect on Dual Variables:

$$\|\lambda-\lambda_{0}\|_{2}\leq{\frac{\delta}{\lambda_{\mathrm{min}}}}$$
. (11)
(ii) Effect on Aligned Model:

$$D_{\rm KL}(p_{\bf\lambda}\|p_{{\bf\lambda}_{0}})\leq\frac{2\delta(B_{0}+\delta)}{\lambda_{\rm min}}.\tag{1}$$

Remark 4 (Interpretation of Theorem 6). The following observations are made from Theorem 6. First, if λmin is moderately large, the sensitivity of λ and pλ to perturbations in r is reduced. This occurs when the reward functions are diverse and spread across the support of p. Second, a small derivation of r also leads to reduced sensitivity bound, and their relationship is nearly linear when δ goes to zero. Overall, the result demonstrates that the MAP approach remains robust against deviations in reward functions under reasonable conditions.

$\downarrow$ . 
Proof of Theorem 6. For notational simplicity, we abbreviate the expectation Ex∼D,y∼p as Ep for any p, a conditional distribution of y given x. (i) We first prove Inequality 11.

The dual problem for r0 is:

where $Z_{0}(\mathbf{\lambda})=\mathbb{E}_{p_{0}}[e^{\mathbf{\lambda}^{\mathrm{T}}\mathbf{r}_{0}(x,y)}]$. For $\mathbf{r}$, the dual function is:
$$g_{0}(\lambda)=-\log Z_{0}(\lambda)+\lambda^{\mathrm{{T}}}c,$$
$$(12)$$
$$g(\lambda)=-\log Z(\lambda)+\lambda^{\mathrm{T}}c.$$
According to Equation (40), the gradient difference between g and g0 is:

$$\|\nabla g(\mathbf{A})-\nabla g_{0}(\mathbf{A})\|_{2}$$
$$\|\nabla g(\boldsymbol{\lambda})-\nabla g(\boldsymbol{\lambda}_{0})\|_{2}\geq\lambda_{\operatorname*{min}}\|\boldsymbol{\lambda}-\boldsymbol{\lambda}_{0}\|_{2},$$  eigenvalue of $\mathit{Var}_{p_{\boldsymbol{\lambda}}}\left(\boldsymbol{r}(x,y)\right)$. Thus, we have 
∥∇g(λ) − ∇g0(λ)∥2 = ∥Epλ[r(x, y)] − Epλ[r0(*x, y*)]∥2 ≤ δ.

According to Equation 43 regarding the Hessian matrix of g, the function ∇g has Lipschitz continuity:
∥∇g(λ) − ∇g(λ0)∥2 ≥ λmin∥λ − λ0∥2, (13)
where λmin is the smallest eigenvalue of Varpλ(r(*x, y*)). Thus, we have

$$(13)$$
$$\|\mathbf{\lambda}-\mathbf{\lambda}_{0}\|_{2}\leq{\frac{\delta}{\lambda_{\mathrm{min}}}}.$$

(ii) We then prove Inequality 12.

The aligned distributions pλ and pλ0are defined as:
pλ(y | x) ∝ p0(y | x) exp(λ Tr(x, y)), pλ0(y | x) ∝ p0(y | x) exp(λ T
0 r0(x, y)).

The KL divergence between pλ and pλ0is:

$$D_{\mathrm{KL}}(p_{\boldsymbol{\lambda}}\|p_{\boldsymbol{\lambda}_{0}})=\mathbb{E}_{p_{\boldsymbol{\lambda}}}\left[\log{\frac{p_{\boldsymbol{\lambda}}(y\mid x)}{p_{\boldsymbol{\lambda}_{0}}(y\mid x)}}\right].$$

Substituting the definitions of pλ and pλ0, we obtain

$$\log\frac{p_{\boldsymbol{\lambda}}(y\mid x)}{p_{\boldsymbol{\lambda}_{0}}(y\mid x)}=({\boldsymbol{\lambda}}-{\boldsymbol{\lambda}_{0}})^{\mathrm{{T}}}{\boldsymbol{r}}(x,y)-\log Z({\boldsymbol{\lambda}})+\log Z_{0}({\boldsymbol{\lambda}_{0}}).$$

Expanding the KL divergence:

$$D_{\mathrm{KL}}(p_{\mathbf{\lambda}}\|p_{\mathbf{\lambda}_{0}})=\mathbb{E}_{p_{\mathbf{\lambda}}}[(\mathbf{\lambda}-\mathbf{\lambda}_{0})^{\mathrm{{T}}}\mathbf{r}(x,y)]-(\log Z(\mathbf{\lambda})-\log Z_{0}(\mathbf{\lambda}_{0})).$$
$$\|\mathbb{E}_{p_{\lambda}}[\mathbf{r}(x,y)]\|_{\lambda}$$

Using the boundedness assumption and triangle inequality, we have

∥Epλ[r(*x, y*)]∥∞ ≤ B0 + δ.
Thus, the first term is bounded by
$$|\mathbb{E}_{p_{\lambda}}[(\lambda-\lambda_{0})^{\mathrm{{T}}}r(x,y)]|\leq\|\lambda-\lambda_{0}\|_{2}\cdot(B_{0}+\delta).$$
Tr(x, y)]*| ≤ ∥*λ − λ0∥2 · (B0 + δ). (14)
$$(14)$$