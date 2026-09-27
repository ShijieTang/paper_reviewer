# Advancing Expert Specialization For Better Moe

Hongcan Guo1∗ Haolang Lu1∗ Guoshun Nan1† Bolun Chu1 **Jialin Zhuang**1 Yuan Yang1 Wenhao Che1 Xinye Cao1 Sicong Leng2 Qimei Cui1 **Xudong Jiang**2 1Beijing University of Posts and Telecommunications, China 2Nanyang Technological University, Singapore
{ai.guohc,lhl_2507,nanguo2021}@bupt.edu.cn

## Abstract

Mixture-of-Experts (MoE) models enable efficient scaling of large language models (LLMs) by activating only a subset of experts per input. However, we observe that the commonly used auxiliary load balancing loss often leads to expert overlap and overly uniform routing, which hinders expert specialization and degrades overall performance during post-training. To address this, we propose a simple yet effective solution that introduces two complementary objectives: (1) an orthogonality loss to encourage experts to process distinct types of tokens, and (2) a variance loss to encourage more discriminative routing decisions. Gradient-level analysis demonstrates that these objectives are compatible with the existing auxiliary loss and contribute to optimizing the training process. Experimental results over various model architectures and across multiple benchmarks show that our method significantly enhances expert specialization. Notably, our method improves classic MoE baselines with auxiliary loss by up to 23.79%, while also maintaining load balancing in downstream tasks, without any architectural modifications or additional components. Our code is available at this link.

## 1 Introduction

Large language models (LLMs) [67, 65, 62, 6] have demonstrated remarkable generalization capabilities [52, 69, 74, 73] across a wide range of tasks [53, 24], but their inference cost [15, 57] grows rapidly with scale, hindering practical deployment and efficiency. Mixture-of-Experts (MoE) [9, 3, 37] architectures alleviate this problem by activating only a subset of experts per input [19], thus enabling greater model capacity without a commensurate increase in computational overhead [22, 49, 33]. To maximize parameter utilization, MoE systems typically introduce load balancing [56, 20] objectives that encourage a more uniform routing of tokens across experts during pre-training. While load balancing is effective in avoiding idle experts during large-scale pre-training, it often hinders model adaptation in the **post-training stage** for downstream tasks, where data distributions are narrower and more domain-specific. In such settings, token occurrences are typically concentrated within particular subspaces (e.g., numeric or symbolic tokens in math tasks), intensifying the tension between balanced routing and expert specialization. A widely observed phenomenon is that load balancing encourages uniform expert routing across inputs, resulting in highly overlapping token distributions [14, 79]. This overlap leads to convergence in expert representations [46], ultimately compromising the development of specialized functionalities. The lack of specialization [14] becomes particularly problematic during fine-tuning [17, 60, 2, 80] on downstream tasks with strong domain preferences, where the model struggles to adapt and exhibits degraded performance [34].

∗Equal contribution. †Corresponding author.

Routing Diversification( ): **Expert Specialization(** ):
OVERLAP
Expert Index Token CountVariance Experts Token Balance Experts Load Variance Decrease Training

(Ours)
Routi ng Sc ore Me tric Top 3 Experts Token Embedding Visualization Experts After **Train**

Specialize Token Assignment for Experts Experts Before Train Routing Output Diverse **Routing Output**
Diverse (Training)
Discriminative Routing VarianceRouting Output Variance Growth
This highlights a core challenge in MoE post-training: the inherent conflict between encouraging expert specialization [50, 38, 36]and *enforcing routing uniformity* [83] via auxiliary losses. From the expert perspective, load-balanced routing causes overlapping training intentions across experts [14, 45, 46, 7], suppressing the development of distinct expert behaviors. From the **router** perspective, as experts become less specialized, the router receives less variation across experts, leading to increasingly uniform and less informed token-to-expert assignments [82]. These dynamics form a self-reinforcing loop: diminished specialization and uniform routing exacerbate each other over time, progressively degrading both expert expressiveness and routing quality [20]. This compounding effect reveals a deeper limitation of existing training objectives, which lack mechanisms to decouple expert specialization from the uniformity constraints imposed by auxiliary losses.

To address this challenge, we propose a gradient-based multi-objective optimization framework that promotes expert specialization and routing diversification, while preserving load balance from auxiliary loss. We introduce two complementary objectives, as shown in Figure 1: 1) **Expert** Specialization, which fosters distinct expert representations by ensuring that each expert specializes in processing different tokens. 2) **Routing Diversification**, which drives differentiated routing decisions, enabling more precise token-to-expert assignments by enhancing the variance in routing. By jointly optimizing these objectives, our method mitigates the trade-off between model performance and routing efficiency in MoE training. We demonstrate that our approach successfully achieves:
- **Enhanced expert–routing synergy**. Our joint objectives reduce expert overlap by up to 45%
and increase routing score variance by over 150%, leading to clearer specialization and more discriminative expert assignment.

- **Stable load balancing**. Despite introducing new objectives, our method matches the baseline's MaxVioglobal across all models, with RMSE under 8.63 in each case.

- **Improved downstream performance**. We achieve 23.79% relative gains across 11 benchmarks and outperform all baselines on 92.42% of tasks ,all without modifying the MoE architecture.

## 2 Motivation 2.1 Preliminaries Of Moe

In a typical MoE layer, let there be n experts, and a sequence of input tokens represented by X = {x1, x2, · · · , xN }, where N is the total number of tokens in the sequence. The routing score matrix after applying the top-k mechanism is denoted as:

$$\mathcal{S}=\begin{pmatrix}s_{11}&s_{12}&\cdots&s_{1n}\\ s_{21}&s_{22}&\cdots&s_{2n}\\ \vdots&\vdots&\ddots&\vdots\\ s_{N1}&s_{N2}&\cdots&s_{Nn}\end{pmatrix},\quad\sum_{j=1}^{n}s_{ij}=1,\quad i=1,2,\cdots,N\tag{1}$$

where sij represents the routing weight assigned to the i-th token for the j-th expert. Let F = {f1, f2, · · · , fn} represent the proportion of tokens assigned to each expert, where fj is the number of tokens assigned to the j-th expert. For any given MoE layer, the total loss function L consists of two parts, the main loss Lh and the auxiliary loss Laux:

$$\mathcal{L}=\mathcal{L}_{h}+\alpha\cdot\mathcal{L}_{aux}=\mathcal{L}_{h}+\alpha\sum_{j=1}^{n}f_{j}\cdot p_{j},p_{j}=\sum_{i=1}^{N}s_{ij},\tag{2}$$  computed from the output of the MoF layer and $\mathcal{L}_{aux}$ is the auxiliary loss term.  
where Lh is the loss computed from the output of the MoE layer, and Laux is the auxiliary loss term, α denotes the weighting coefficient for the auxiliary loss. Here, pj represents the total routing score for the j-th expert, which is the sum of the routing weights for all tokens assigned to that expert.

## 2.2 Observations

Obs I (Expert Overlap): Introduction of the auxiliary loss function leads to a more homogenized distribution of tokens across experts, which may reduce the distinctiveness of each expert. It has been observed that the auxiliary loss function is independent of the expert parameter matrices θEj. Therefore, for the j-th expert, its gradient can be written as:

$$\frac{\partial\mathcal{L}}{\partial\theta_{E_{j}}}=\frac{\partial\mathcal{L}_{h}}{\partial\theta_{E_{j}}}+\alpha\cdot\frac{\partial\mathcal{L}_{aux}}{\partial\theta_{E_{j}}}=\frac{\partial\mathcal{L}}{\partial y_{h}}\cdot\frac{\partial y_{h}}{\partial\theta_{E_{j}}}=\sum_{i=1}^{N}x_{i}\cdot s_{ij},j=1,2,\cdots,n.\tag{3}$$  $\bullet$\(\
where θEjis the parameter matrix of the j-th expert, and yh is the output of the MoE layer. During gradient descent, the addition of the auxiliary loss Laux forces the routing mechanism to evenly distribute the tokens across experts as much as possible.

This results in input token xi being assigned to an expert that may not be semantically aligned with it, causing an unintended gradient flow to expert j. Mathematically, after applying the top-k mechanism, the routing score sij transitions from 0 to a non-zero value, introducing gradients from tokens that originally had no affinity with expert j.

Obs II (Routing Uniformity)*: As training progresses, the routing output tends to become more* uniform, with the expert weight distribution gradually converging towards an equal allocation. To understand this phenomenon, we first examine the source of gradients with respect to the routing parameters θR. Since the routing mechanism produces only the score matrix S = sij , the gradient
∂L/∂θR can be written as:

$$\frac{\partial L}{\partial\theta_{R}}=\frac{\partial{\cal L}_{h}}{\partial\theta_{R}}+\alpha\cdot\frac{\partial{\cal L}_{aux}}{\partial\theta_{R}}=\sum_{i=1}^{N}x_{i}\sum_{j=1}^{n}\theta_{E_{j}}\cdot\frac{\partial s_{ij}}{\partial\theta_{R}}+\alpha\cdot\sum_{j=1}^{n}f_{j}\sum_{i=1}^{N}\frac{\partial s_{ij}}{\partial\theta_{R}},\tag{4}$$

where xi· θEjrepresents the output of expert j for token xi, and fj denotes the frequency with which expert j is selected. This formulation reveals that the routing gradient is primarily influenced by the expert outputs and the token distribution across experts.

The auxiliary loss Laux is introduced to encourage balanced token assignment by optimizing the uniformity of fj . However, since fj is non-differentiable, direct optimization is not feasible. Instead, a surrogate variable pj , which is differentiable and positively correlated with fj , is employed to approximate the objective and enable gradient flow back to the routing network. As training proceeds, the optimization objective increasingly favors the uniformity of pj , which drives fj toward an even distribution. Moreover, as discussed in Observation I, incorrect token assignments caused by auxiliary regularization introduce overlapping gradients among experts, increasing the similarity of xi· θEj across different j.

Obs III (Expert–Routing Interaction): While *Obs I* concerns expert specialization, while *Obs II* reflects the uniformity of routing. These two effects interact during training, jointly driving the model toward degraded performance.

- *Expert-side interference caused by Obs I leads to blurred specialization.* Tokens are assigned to mismatched experts, and the resulting gradient interference reduces expert distinctiveness. As the routing weights become more uniform, different experts receive similar gradients from the same tokens, increasing their functional overlap.

- *This expert similarity feeds back into the routing mechanism.* As expert outputs become less distinguishable, the routing network finds fewer cues to differentiate among experts, leading to even more uniform weight distributions. This promotes random top-k selection and further misalignment between tokens and their optimal experts.

Together, this loop gradually steers the model toward more uniform token allocation and reduced expert specialization, highlighting potential opportunities for improving the routing strategy and expert assignment.

## 3 Method

Based on the observations above, we propose the following design to mitigate *expert overlap* and routing uniformity, the overall loss function L is defined as follows:

$${\mathcal{L}}={\mathcal{L}}_{h}+{\mathcal{L}}_{b a l a n c e},\quad{\mathcal{L}}_{b a l a n c e}=\alpha\cdot{\mathcal{L}}_{a u x}+\beta\cdot{\mathcal{L}}_{o}+\gamma\cdot{\mathcal{L}}_{v},$$

where Laux represents the existing auxiliary loss, with coefficient α, and the newly introduced orthogonality loss Lo and variance loss Lv (see Subsec 3.1), with coefficients β and γ respectively. It is worth noting that the theoretical complementarity of these optimization objectives, rather than any inherent conflict, is formally analyzed and demonstrated in Subsection 3.2.

## 3.1 Implementations Of Losses Lo And Lv

In this section, we introduce two critical loss functions Lo and Lv that act on the expert and router components, respectively. Expert Specialization. We introduce an orthogonalization objective that encourages independent expert representations. Specifically, we design the following orthogonality loss:

$${\mathcal{L}}_{o}=\sum_{i=1}^{N}\sum_{j=1}^{n}\sum_{\begin{subarray}{c}k=1\\ k\neq j\end{subarray}}^{n}\left\|{\frac{\langle{\tilde{x}}_{i j},{\tilde{x}}_{i k}\rangle}{\langle{\tilde{x}}_{i k},{\tilde{x}}_{i k}\rangle+\epsilon}}{\tilde{x}}_{i k}\right\|^{2},\quad{\tilde{x}}_{i j}=x_{i}\cdot\theta_{E_{j}}\cdot\mathbb{I}_{\{s_{i j}>0\}},$$
$$(S)$$

where ⟨·⟩ denotes the inner product between two vectors, and I*sij >* 0 is an indicator function that evaluates to 1 when sij > 0 and 0 otherwise. Here, x˜ij represents the output of expert j for token xi after the top-k routing selection.

The orthogonality loss Lo reduces the overlap between different expert outputs within the same top-k group by minimizing their projections onto each other. This encourages experts to develop more distinct representations, promoting specialization in processing different token types.

Routing Diversification. We introduce a variance-based loss to encourage more diverse routing decisions and promote expert specialization. Specifically, we define the variance loss as:

$${\mathcal{L}}_{v}=-\sum_{i=1}^{N}\sum_{j=1}^{n}{\frac{1}{n}}\cdot(s_{i j}-{\bar{s}}_{j})^{2},{\bar{s}}_{j}={\frac{1}{N}}\cdot\sum_{i=1}^{N}s_{i j},$$
$$(6)$$
$$(7)$$

where s¯j denotes the average routing score for expert j across the batch. By maximizing the variance of routing scores, Lv discourages uniform token-to-expert assignments and encourages more deterministic and distinct routing patterns, thereby facilitating expert specialization.

## 3.2 Compatibility Of Multi-Objective Optimization

In this section, we analyze how each component influences the optimization dynamics of expert parameters θEjand routing parameters θR during training. Meanwhile, we will focus on the optimization and compatibility of the two losses Lo and Lv with respect to load balancing and expert specificity. The following two key questions guide our analysis.

Balancing Expert and Routing. How can expert (Lo) and routing (Lv) optimizations be designed to complement each other without compromising their respective objectives?

We first demonstrate that Lo and Lv are compatible in their optimization directions within MoE, then show that they mutually reinforce each other.

Mutually Compatible. We elaborate on the compatibility of Lo and Lv from the perspectives of expert and Routing.

From the **expert perspective**, we observe that the auxiliary loss Laux and the variance loss Lv do not directly contribute gradients to the expert parameter matrix θEj. Consequently, the gradient of the total loss with respect to θEjis derived solely from the primary task loss Lh and the orthogonality loss Lo:

$${\frac{\partial{\mathcal{L}}}{\partial\theta_{E_{j}}}}=\sum_{i=1}^{N}\left(s_{i j}\cdot g_{y i}+\beta\cdot\sum_{\stackrel{k=1}{k\neq j}}^{n}{\frac{{\tilde{x}}_{i k}{\tilde{x}}_{i k}^{\top}}{\langle{\tilde{x}}_{i k},{\tilde{x}}_{i k}\rangle+\epsilon}}\cdot{\tilde{x}}_{i j}\right)\cdot x_{i}^{\top}$$

$$({\boldsymbol{8}})$$
i(8)
Here, gyi = ∇yiLh denotes the gradient of the primary task loss with respect to the model output. This gradient is influenced by both the routing score sij and the expert representation x˜ij . As training progress, the variance of expert weights increases, and the gradient encourages stronger preferences in different directions for each token.

From the **routing perspective**, we notice that Lo does not affect the gradient with respect to routing parameters θR. The gradient of the total loss with respect to θR is:

$$\frac{\partial\mathcal{L}}{\partial\theta_{R}}=\frac{\partial\mathcal{L}}{\partial s_{ij}}\cdot\frac{\partial s_{ij}}{\partial\theta_{R}}=\sum_{i=1}^{N}\sum_{j=1}^{n}\left(\tilde{x}_{ij}+\alpha\cdot f_{j}-\gamma\cdot\frac{2(N-1)}{nN}\cdot(s_{ij}-\tilde{s}_{j})\right)\cdot\frac{\partial s_{ij}}{\partial\theta_{R}}.\tag{9}$$  This gradient is influenced by expert representations $\tilde{x}_{ij}$, expert load $f_{j}$, and routing weights $s_{ij}$. As 
the model converges, the expert load fj becomes more balanced, and the variance of routing weights sij increases. Orthogonalizing expert representations causes the routing gradients to flow in more orthogonal directions, making the weight allocation more biased towards the representations and increasing the weight variance.

Summary. Expert parameters θEj are solely influenced by the gradients of Lo without conflict. While routing parameters θR are affected by both Lo and Lv, the objectives of these two losses (orthogonalityfriendliness vs. score diversification) remain non-conflicting.

Mutually Reinforcing. Lo aims to encourage the effective output vectors of different selected experts
j and k to tend to be orthogonal for the same input token xi, i.e., ⟨x˜ij , x˜ik⟩ ≈ 0. The learning signal
for the routing mechanism partially originates from the gradient of the primary task loss Lh with
the routing mechanism partially originates from the gradient of the primary task loss $\mathcal{L}_{h}$ with set to the routing score $s_{ij}$:  $$\frac{\partial\mathcal{L}}{\partial s_{ij}}=\underbrace{g_{u}^{T}\tilde{x}_{ij}}_{\text{from}\mathcal{L}_{h}}+\underbrace{\partial\frac{\partial\mathcal{L}_{\text{min}}}{\partial s_{ij}}}_{\text{from}\mathcal{L}_{u}}-\underbrace{\gamma\frac{2(N-1)}{nN}(s_{ij}-\tilde{s}_{j})}_{\text{from}\mathcal{L}_{u}},\quad y_{i}=\sum_{j}s_{ij}\tilde{x}_{ij},\quad g_{u}=\frac{\partial\mathcal{L}_{h}}{\partial y_{i}}\tag{10}$$  $\mathcal{L}_{u}$\(\mathcal{L}_
Assuming pij = g T
yi x˜ij , when the expert outputs tend to be orthogonal, for any given task gradient gyi
, the projections pij onto these approximately orthogonal expert outputs are more likely to exhibit significant differences. The increased variance of the primary task-related signals pij implies that the routing mechanism receives more discriminative and stronger learning signals, which creates more favorable conditions for Lv to achieve diversification of routing scores. Lv enhances the diversity of routing scores sij by optimizing routing parameters θR. Meanwhile, due to the influence of Lo's gradient β
∂Lo
∂sij on θR, routing tends to assign more specialized token subsets Tj to each expert j. Expert parameters θEjlearn the unique features of tokens within Tj ,
leading to gradual functional divergence among experts, thereby promoting expert orthogonality. Multi-Objective Optimization. *How do expert and routing maintain their balance while enhancing* Laux and Lh *independently, ensuring mutually beneficial performance improvements?*
Lemma 1 Let S ∈ RN×n be a matrix that satisfies following conditions: each row sums to 1, each row contains k non-zero elements and n − k zero elements. Then, there always exists a state in which the following two objectives are simultaneously optimized: 1. The sum of the elements in each column tends to the average value Nn
; 2. The variance of the non-zero elements in each row increases.

Lemma 2 For two sets of points A and B of equal size, it is always possible to partition A ∪ B *such* that A ∩ B = ∅ and |A| = |B|.

The overall objective function L optimizes four key dimensions: accurate data fitting(Lh), expert orthogonalization(Lo), balanced expert routing weights(Laux), and increased variance in routing outputs(Lv). Our core objective is to achieve an **optimal balance by jointly optimizing these**
multiple objectives, ensuring they complement each other for enhanced model performance.

As shown by Lemma 1, expert load fj and routing weights sij can be optimized together. As demonstrated in Lemma 2, the objectives of orthogonalization and load balancing are not in conflict and can be jointly optimized. Thus, both expert and routing modifications can be optimized alongside load balancing (balanced expert routing weights). Moreover, orthogonalization enhances routing weight variance, in turn, improves expert specialization (as discussed in Section 2.2). This leads to more distinctive expert representations, aligning with performance (accurate data fitting) improvements when optimized together.

## 4 Experiments

In this section, we conduct experiments to address the following research questions:
- RQ1: Does introducing the orthogonality loss (Lo) and variance loss (Lv) lead to better overall performance in downstream tasks compared to baseline approaches?

- RQ2: To what extent does our method maintain expert load balancing during training?

- RQ3: How do the orthogonality loss (Lo) and variance loss (Lv) interact with each other, and what are their respective and joint impacts on expert specialization and routing behavior?

- RQ4: What are the individual and combined contributions of Lo, Lv, and the auxiliary loss Laux to the final model performance?

## 4.1 Experimental Setup

Environment. All experiments are performed on a CentOS Linux 7 server with PyTorch 2.3. The hardware specifications consist of 240GB of RAM, a 16-core Intel Xeon CPU, and two NVIDIA A800 GPUs, each having 80GB of memory. Implementation details are provided in the Appendix F. Datasets. We evaluate our method on a total of **11 benchmarks**. Specifically, we use the training sets from Numina [41], GLUE [66], and the FLAN collection [72] to train our models. Our benchmarks include: ❶ **Mathematics**: GSM8K [12], MATH500 [44], and Numina [41]; ❷ **Multi-Domain**
Tasks: MMLU [31, 30], MMLU-pro [70], BBH [63], GLUE [66]; LiveBench [76] and GPQA [59]. ❸
 **Code generation**: HumanEval [10] and MBPP [4]. We group training and test sets by language, reasoning, science, math, and code to match downstream evaluation needs. Detail in Appendix D. Baselines. We compare our method with **4 existing MoE training strategies**. With Aux Loss [46] applies auxiliary load-balancing losses during routing to encourage expert utilization diversity. GShard [39] introduces a foundational sparse expert framework with automatic sharding and routing; ST-MoE [85] enhances training stability via router dropout and auxiliary losses; Loss-Free Balancing [68] achieves balanced expert routing without auxiliary objectives. Detail in Appendix G.

Metrics. We employ **6 evaluation metrics** to test our method in terms of accuracy, expert load balancing (MaxVioglobal [68]), clustering quality (Silhouette Coefficient), expert specialization (Expert Overlap), routing stability (Routing Variance), and prediction error (RMSE). Detail in Appendix E.

Table 1: **Performance on different downstream tasks.** The table shows accuracies of methods across

models and downstream tasks. Notably, **we categorize sub-downstream tasks in Multi-Domain and ensure** training/evaluation sets are domain-aligned, following downstream task requirements.

Method Model Multi-Domain (Avg.) Code **Math**

MMLU MMLU-pro BBH GLUE Livebench GPQA HumanEval MBPP **GSM8K MATH500 NuminaTest**

With Aux Loss

| training/evaluation sets are domain-aligned, following downstream task requirements. Method Model Multi-Domain (Avg.) Code   | Math       |                                 |                                             |                                             |                                  |                                  |            |            |
|------------------------------------------------------------------------------------------------------------------------------|------------|---------------------------------|---------------------------------------------|---------------------------------------------|----------------------------------|----------------------------------|------------|------------|
| MMLU                                                                                                                         | MMLU-pro   | BBH                             | GLUE                                        | Livebench                                   | GPQA HumanEval                   | MBPP GSM8K MATH500 NuminaTest    |            |            |
| With Aux Loss Loss-Free Balancing                                                                                            | 30.71±2.10 | 16.81±0.70                      | 32.99±1.00 49.60±1.30 9.79±0.20             | 20.63±1.60                                  | 53.16±2.40                       | 32.80±1.40 21.28±0.40            | 5.83±1.30  | 17.23±1.60 |
| 29.27±0.10                                                                                                                   | 19.47±2.50 | 26.92±2.30 49.26±0.40 7.43±0.10 | 21.15±0.40                                  | 51.52±1.50                                  | 31.36±1.10 15.70±2.40            | 5.47±1.50                        | 14.99±2.40 |            |
| 16B                                                                                                                          |            |                                 |                                             |                                             |                                  |                                  |            |            |
| GShard                                                                                                                       | 27.05±2.00 | 20.48±0.60                      | 29.83±1.80 53.83±0.70 8.69±1.20             | 24.28±2.30                                  | 57.75±2.20                       | 34.50±1.70 27.12±1.30            | 8.20±1.50  | 16.99±0.70 |
| MoE-                                                                                                                         |            |                                 |                                             |                                             |                                  |                                  |            |            |
| ST-MOE                                                                                                                       | 34.23±2.20 | 19.71±0.80                      | 36.91±1.90 54.56±2.30 6.48±0.70             | 20.35±0.90                                  | 53.28±1.60                       | 36.34±1.50 30.10±2.00            | 7.08±0.40  | 15.48±1.20 |
| Ours                                                                                                                         | 33.35±2.20 | 24.87±1.20                      | 37.52±1.40 60.01±1.00 11.00±1.70 25.15±0.40 | 63.30±0.70                                  | 40.03±0.40 35.00±1.00 10.82±0.30 | 20.41±0.10                       |            |            |
| With Aux Loss                                                                                                                | eekepSDe            | 33.23±2.10                      | 28.40±0.20                                  | 34.80±1.40 35.97±0.20 11.70±0.50 24.92±0.80 | 40.24±0.80                       | 41.23±0.20 44.79±2.10 42.03±1.40 | 42.01±1.90 |            |
| Loss-Free Balancing                                                                                                          | 30.23±0.80 | 30.75±2.10                      | 34.21±1.10 39.83±1.80 10.15±1.10 26.33±0.60 | 41.28±1.40                                  | 36.02±2.30 43.35±0.70 39.76±1.10 | 43.90±1.10                       |            |            |
| ek                                                                                                                              |            |                                 |                                             |                                             |                                  |                                  |            |            |
| GShard                                                                                                                       | 30.86±1.10 | 29.13±0.80                      | 37.67±0.30 38.89±1.00 13.17±1.80 24.34±2.10 | 45.36±1.60                                  | 37.00±2.10 45.39±1.50 43.61±2.10 | 43.25±0.70                       |            |            |
| Lite DeepSe                                                                                                                  |            |                                 |                                             |                                             |                                  |                                  |            |            |
| ST-MOE                                                                                                                       | 32.68±2.10 | 30.28±2.10                      | 38.78±0.90 38.27±1.00 10.60±2.30 22.33±0.40 | 44.10±0.20                                  | 39.72±2.30 47.78±1.80 46.74±0.50 | 48.65±0.70                       |            |            |
| V2-                                                                                                                          |            |                                 |                                             |                                             |                                  |                                  |            |            |
| Ours                                                                                                                         | 35.59±0.50 | 37.37±0.20                      | 38.84±1.70 41.20±2.00 14.60±2.50 28.76±0.10 | 43.58±0.30                                  | 43.53±2.40 50.94±2.40 49.33±2.40 | 50.67±1.10                       |            |            |
| With Aux Loss                                                                                                                | 35.82±1.40 | 36.10±1.50                      | 47.17±0.70 26.16±1.20 15.84±1.70 30.72±1.90 | 63.61±1.90                                  | 47.34±1.50 82.32±1.50 57.03±1.60 | 45.41±0.40                       |            |            |
| Loss-Free Balancing                                                                                                          | 27.40±0.10 | 31.91±2.10                      | 42.45±0.50 32.97±1.60 20.05±2.40 29.27±1.80 | 62.93±2.50                                  | 44.92±1.30 79.34±0.70 57.77±0.50 | 42.82±0.10                       |            |            |
| 3B                                                                                                                           |            |                                 |                                             |                                             |                                  |                                  |            |            |
| GShard                                                                                                                       | 36.06±0.90 | 30.65±0.50                      | 49.20±1.70 34.46±2.40 13.97±2.30 31.13±1.10 | 64.50±1.50                                  | 49.85±0.50 84.62±0.80 56.09±2.20 | 47.18±2.30                       |            |            |
| -A                                                                                                                           |            |                                 |                                             |                                             |                                  |                                  |            |            |
| ST-MOE                                                                                                                       | 33.03±0.90 | 26.83±1.70                      | 46.78±0.30 30.18±1.50 16.99±1.70 30.93±1.50 | 66.04±1.60                                  | 47.97±2.20 84.45±0.90 57.61±1.60 | 49.42±2.10                       |            |            |
| 16B                                                                                                                          |            |                                 |                                             |                                             |                                  |                                  |            |            |
| Ours                                                                                                                         | 40.36±2.20 | 34.90±0.30                      | 52.42±1.80 37.01±1.10 20.85±1.10 32.01±0.90 | 70.64±0.20                                  | 47.77±1.00 87.62±2.20 59.64±0.20 | 52.88±1.70                       |            |            |
| ghtnliMoo                                                                                                                              |            |                                 |                                             |                                             |                                  |                                  |            |            |

Setup. Each benchmark is fine-tuned separately on 6,000 high-quality examples, primarily from the official training split and supplemented when necessary. Answers are generated using strong teacher models (OpenAI o3-mini and DeepSeek R1) and manually verified for correctness. Fine-tuning is limited to three epochs (∼550 steps) to prevent overfitting. All experiments adopt LoRA-based fine-tuning, with LoRA modules inserted into both router and expert layers to enable joint optimization. A rank of 32 is used to approximate full-model updates. Detailed configurations, including optimizer, batch size, and learning rate, are provided in Appendix H.2.

## 4.2 Performance In Downstream Tasks (Rq1)

To verify that our Lbalance enhances model performance in downstream task scenarios through expert orthogonality and routing output diversification, as shown in Table 1, we design downstream task scenarios on 11 well-known benchmarks and validate our method against four baseline methods with distinct loss designs on three widely used MoE models. We make the following observations:
Obs.❶ **Baseline methods without guidance for expert specialization exhibit varied performance**
and fail to effectively improve downstream task performance. As shown in Table 1, the four baseline methods show no clear overall performance ranking across the 11 tasks, with performance variations within 2% in many tasks. Their overall performance is significantly lower than our method, demonstrating no potential to improve downstream task performance.

Obs.❷ **Our method guiding expert specialization effectively enhances model performance in**
downstream tasks. As shown in Table 1, we achieve state-of-the-art (SOTA) results in over 85% of the 33 tasks across the three models. In some tasks, the average across multiple measurements even outperforms the next-best method by nearly 7%. Extensive experiments indicate that our method significantly improves model performance in downstream task scenarios by enhancing expert specialization. More results on additional baselines and MoE architectures are provided in Appendix I.

## 4.3 Load Balancing (Rq2)

To verify that our newly added losses Lv and Lo do not affect the load balancing effect, we conduct statistical measurements on the load balancing of all combinations of Laux, Lv, and Lo across various models during training.

Figure 2 shows the variation of *M axV io*global ↓ across training steps for different loss combinations, as well as the RMSE of differences between our method and other combinations. We make the following observations:
Obs.❸ Loss combinations without Laux **exhibit significantly worse load balancing performance**
than those with Laux. As shown in Figure 2, across three distinct models, the *M axV io*global of the w/o all method (with no losses added) is significantly higher than that of other methods, indicating

M

a x Vi o g l o b a l RMSE (vs ours):

 only aux: 0.023 w/o lv: 0.019 w/o lo: 0.023 w/o all: 0.650 DeepSeek-Moe-16B
w/o all w/o lo w/o lv only aux ours M

a x Vi o g l o b a l RMSE (vs ours):

 only aux: 0.012 w/o lv: 0.012 w/o lo: 0.012 w/o all: 0.291 Moonlight-16B-A3B
w/o all w/o lo w/o lv only aux ours M

a x Vi o g l o b a l RMSE (vs ours):

 only aux: 0.026 w/o lv: 0.026 w/o lo: 0.022 w/o all: 1.044 DeepSeek-V2-Lite w/o all w/o lo w/o lv only aux ours Method 6.75 7.00 7.25 7.50 7.75 8.00 8.25 8.50 8.75 Method 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 Method 3 4 5 6 7 0 100 200 300 400 500 Step ours only aux w/o lv w/o lo w/o all 0 100 200 300 400 500 Step ours only aux w/o lv w/o lo w/o all 0 100 200 300 400 500 Step ours only aux w/o lv w/o lo w/o all
notably poorer load balancing. In particular, for the DeepSeek-V2-Lite model, the method without Laux converges to 6.14, whereas methods with Laux converge to 2.48, demonstrating that loss combinations containing Laux achieve significantly better load balancing.

Obs.❹ Incorporating any combination of Lv and Lo into Laux **does not affect load balancing.**
As shown in Figure 2, for methods with Laux, the trends of "only aux" (no additional losses), "w/o lv" (only Lo), "w/o lo" (only Lv), and "ours" (both Lv and Lo) are nearly identical. Additionally, the RMSE (root mean squared error) of our method relative to other baselines does not exceed 0.03, further corroborating the conclusion that the combination of Lv and Lo does not impact load balancing.

## 4.4 Behaviors Of Experts And Routing (Rq3)

To verify that Lv and Lo can jointly promote expert orthogonality and routing score diversification, following the method setup in Section 4.3, we will conduct evaluations of expert orthogonality and measurements of routing score diversification for different loss combinations.

Silhouette Coefficient ( ) Comparison ours w/o lo w/o lv w/o all only aux Expert Overlap ( ) Comparison ours w/o lo w/o lv w/o all only aux Routing Variance ( ) Comparison ours w/o lo w/o lv w/o all only aux 0.15 0.10 0.05 0.00 0.05 0.10 silhouette coefficient DeepSe ek
-V2-Lite Moonlight
-16B-A3B
DeepSeek
-Moe-1 6B
0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 expert overlap DeepSe ek
-V2-Lite Moonlight
-16B-A3B
DeepSeek
-Moe-1 6B
0.000 0.002 0.004 0.006 0.008 0.010 variance DeepSe ek
-V2-Lite Moonlight
-16B-A3B
DeepSeek
-Moe-1 6B
Mo del Mo del Mo del
As shown in Figure 3, the first two subplots demonstrate the orthogonality of experts, while the last subplot illustrates the diversification of routing outputs. We make the following observations:
Obs.❺ Lo directly promotes expert orthogonality, and Lv **also aids in expert orthogonality.** As shown in the first two panels of Figure 3, our method with both Lo and Lv achieves state-of-the-art (SOTA) results across three models, with Expert Overlap even dropping below 0.3. The method with only Lo and Laux (w/o lv) consistently ranks second-best, indicating that Lo has a more significant impact on expert orthogonality. Notably, the method with only Lv and Laux (w/o lo) significantly outperforms the method with only Laux across all three models, confirming that Lv also contributes to expert orthogonality.

Obs.❻ Lv directly enhances routing output diversification, and Lo also supports this diversification. Similarly, our method exhibits the highest routing score variance (exceeding 0.010), followed by the method with only Lv and Laux, while the method with only Laux performs worst. This strongly supports the conclusion.

Obs.❼ Laux **leads to higher expert overlap and more homogeneous routing outputs.** Compared to the w/o all method (no losses), the aux only method (with only Laux) shows a Silhouette Coefficient that is over 0.05 higher and a routing output variance that is 0.0045 higher. This indicates that w/o all exhibits significantly greater expert orthogonality and routing output diversification than aux only.

## 4.5 Ablation Among Losses (Rq4)

To demonstrate that both Lo and Lv have positive effects on the model's performance in downstream task scenarios, and their combination synergistically enhances each other's efficacy, we design ablation experiments for these two losses on three models.

Ablation Experiment on Moonlight-16B-A3B
Ablation Experiment on DeepSeek-Moe-16B
Ablation Experiment on DeepSeek-V2-Lite MATH500 MMLU NuminaTest MATH500 MMLU NuminaTest MATH500 MMLU NuminaTest MMLU-pro MMLU-pro MMLU-pro 0 13 26 39 52 65 0 12 24 36 48 60 0 20 40 60 80 100 BBH
GSM8K
BBH
GSM8K
BBH
GSM8K
GLUE
GPQA HumanEval MBPP
GLUE
GPQA HumanEval MBPP
GLUE
GPQA HumanEval MBPP
w/o all w/o low/o lv only auxOurs w/o all w/o low/o lv only auxOurs w/o all w/o low/o lv only auxOurs
Figure 4 illustrates the performance of different ablation method combinations across various downstream tasks. We make the following observations:
Obs.❽ The combination of Lo and Lv **significantly enhances model performance in downstream**
tasks, and each loss individually also improves performance. Our method (combining Lo and Lv)
exhibits the largest coverage area across all three models, nearly encompassing other methods. When either Lo or Lv is ablated (i.e., w/o lv or w/o lo), the coverage areas of these methods are larger than that of the only aux method (with only Laux), indicating performance improvements over the baseline.

Obs.❾ Laux **impacts model performance on downstream tasks.** Figure 4 clearly shows that the only aux method (with only Laux) is nearly entirely enclosed by other methods across all three models, consistently exhibiting the smallest coverage area. Notably, the w/o all method (with no losses)
achieves performance improvements and a larger coverage area than the only aux method when Laux is removed, supporting this conclusion. Beyond the ablation results in Fig. 4, we further conduct a sensitivity analysis on the loss-weight coefficients α, β, and γ. The detailed results and discussions are provided in Appendix H.1.

## 5 Related Work

Auxiliary Losses in MoE Training. Auxiliary losses [39, 85] are commonly used to prevent expert collapse by encouraging balanced expert utilization [14]. Early approaches focus on suppressing routing imbalance, while later works [81] introduce capacity constraints or multi-level objectives to separate routing stability from load balancing [65, 39, 20]. Recent methods [75] further reduce manual tuning by dynamically adjusting auxiliary weights or replacing them with entropy-based routing [42]. However, fixed-rule strategies may underutilize expert capacity, and dynamic schemes can introduce instability or overhead, making robust balancing still a challenge [32, 68]. Orthogonality in MoE. Orthogonalization [47, 28] improves expert diversity by encouraging independent representations [29]. Some methods [54, 84, 51] regularize expert weights directly, while others [14, 29] assign experts to disentangled subspaces based on task semantics. Recent routingbased approaches [47, 58] also impose orthogonality on token-to-expert assignments to reduce redundancy. Nonetheless, static constraints [11] often fail to adapt to dynamic inputs, and dynamic ones [78, 35, 25, 64] may conflict with balancing, complicating expert allocation [32, 82, 27, 68]. Our work addresses these tensions by integrating orthogonalization and balance into a unified, gradient-consistent optimization framework.

## 6 Limitation & Future Discussion

While L*balance* balances load and enhances performance in downstream tasks, its potential in other domains remains unexplored. Specifically, it could be extended to visual models, as suggested in recent work [26], and multimodal or full-modal settings [8], offering opportunities for crossdomain applications. Additionally, investigating L*balance* within lightweight MoE fine-tuning, such as LoRA-MoE [21], could make our approach viable for resource-constrained environments [43]. Furthermore, there is considerable potential in exploring expert-distributed deployment, where L*balance* can optimize both parameter inference efficiency and model performance. This avenue could significantly enhance the scalability and practicality of MoE models in real-world applications, providing new opportunities for distributed expert architectures.

## 7 Conclusion

In this work, we present a theoretically grounded framework that resolves the inherent conflict between expert specialization and routing uniformity in MoE training. By introducing orthogonality and variance-based objectives, our method significantly improves downstream performance without any architectural changes. This demonstrates that MoE efficiency and specialization can be simultaneously optimized through loss-level innovations alone. Experiments show the effectiveness of our method.

## 8 Acknowledgements

This work was supported in part by the National Key Research and Development Program of China under Grant 2022YFB2902200; in part by the Guangxi Key Research and Development Program under Grant FN2504240005; in part by the National Natural Science Foundation of China under Grant 62471064; in part by the Fundamental Research Funds for the Beijing University of Posts and Telecommunications under Grant 2025AI4S02.

## References

[1] Eneko Agirre, Llu'is M'arquez, and Richard Wicentowski, editors. Proceedings of the Fourth International Workshop on Semantic Evaluations (SemEval-2007). Association for Computational Linguistics, Prague, Czech Republic, June 2007.

[2] Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi, Alessandro Cappelli, Ruxandra Cojocaru, Merouane Debbah, Etienne Goffinet, Daniel Heslow, Julien Launay, Quentin Malartic, et al. Falcon-40b: an open large language model with state-of-the-art performance, 2023.

[3] Mikel Artetxe, Shruti Bhosale, Naman Goyal, Todor Mihaylov, Myle Ott, Sam Shleifer, Xi Victoria Lin, Jingfei Du, Srinivasan Iyer, Ramakanth Pasunuru, et al. Efficient large scale language modeling with mixtures of experts. *arXiv preprint arXiv:2112.10684*, 2021.

[4] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. *arXiv preprint arXiv:2108.07732*, 2021.

[5] Baidu-ERNIE-Team. Ernie 4.5 technical report, 2025. [6] Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, et al. Deepseek llm: Scaling open-source language models with longtermism. *arXiv preprint arXiv:2401.02954*, 2024.

[7] Weilin Cai, Juyong Jiang, Le Qin, Junwei Cui, Sunghun Kim, and Jiayi Huang.

Shortcut-connected expert parallelism for accelerating mixture-of-experts. *arXiv preprint* arXiv:2404.05019, 2024.

[8] Weilin Cai, Juyong Jiang, Fan Wang, Jing Tang, Sunghun Kim, and Jiayi Huang. A survey on mixture of experts. *arXiv preprint arXiv:2407.06204*, 2024.

[9] Weilin Cai, Juyong Jiang, Fan Wang, Jing Tang, Sunghun Kim, and Jiayi Huang. A survey on mixture of experts in large language models. IEEE Transactions on Knowledge and Data Engineering, 2025.

[10] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. *arXiv preprint arXiv:2107.03374*, 2021.

[11] Tianlong Chen, Zhenyu Zhang, Ajay Kumar Jaiswal, Shiwei Liu, and Zhangyang Wang. Sparse moe as the new dropout: Scaling dense and self-slimmable transformers. In *ICLR*, 2023.

[12] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*, 2021.

[13] Ido Dagan, Oren Glickman, and Bernardo Magnini. The PASCAL recognising textual entailment challenge. In Machine learning challenges. evaluating predictive uncertainty, visual object classification, and recognising tectual entailment, pages 177–190. Springer, 2006.

[14] Damai Dai, Chengqi Deng, Chenggang Zhao, Rx Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y Wu, et al. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1280–1297, 2024.

[15] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in neural information processing systems, 35:16344–16359, 2022.

[16] DeepSeek-AI. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model, 2024.

[17] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. *Advances in neural information processing systems*, 36:10088–
10115, 2023.

[18] William B Dolan and Chris Brockett. Automatically constructing a corpus of sentential paraphrases. In *Proceedings of the International Workshop on Paraphrasing*, 2005.

[19] William Fedus, Jeff Dean, and Barret Zoph. A review of sparse expert models in deep learning.

arXiv preprint arXiv:2209.01667, 2022.

[20] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *Journal of Machine Learning Research*, 23
(120):1–39, 2022.

[21] Wenfeng Feng, Chuzhan Hao, Yuewei Zhang, Yu Han, and Hao Wang. Mixture-of-loras: An efficient multitask tuning for large language models. *arXiv preprint arXiv:2403.03432*, 2024.

[22] Chongyang Gao, Kezhen Chen, Jinmeng Rao, Ruibo Liu, Baochen Sun, Yawen Zhang, Daiyi Peng, Xiaoyuan Guo, and VS Subrahmanian. Mola: Moe lora with layer-wise expert allocation. In *Findings of the Association for Computational Linguistics: NAACL 2025*, pages 5097–5112, 2025.

[23] Danilo Giampiccolo, Bernardo Magnini, Ido Dagan, and Bill Dolan. The third PASCAL
recognizing textual entailment challenge. In *Proceedings of the ACL-PASCAL workshop on* textual entailment and paraphrasing, pages 1–9. Association for Computational Linguistics, 2007.

[24] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024.

[25] Yongxin Guo, Zhenglin Cheng, Xiaoying Tang, and Tao Lin. Dynamic mixture of experts: An auto-tuning approach for efficient transformer models. *CoRR*, abs/2405.14297, 2024.

[26] Xumeng Han, Longhui Wei, Zhiyang Dou, Zipeng Wang, Chenhui Qiang, Xin He, Yingfei Sun, Zhenjun Han, and Qi Tian. Vimoe: An empirical study of designing vision mixture-of-experts. arXiv preprint arXiv:2410.15732, 2024.

[27] Xin He, Shunkang Zhang, Yuxin Wang, Haiyan Yin, Zihao Zeng, Shaohuai Shi, Zhenheng Tang, Xiaowen Chu, Ivor Tsang, and Ong Yew Soon. Expertflow: Optimized expert activation and token allocation for efficient mixture-of-experts inference. *arXiv preprint arXiv:2410.17954*,
2024.

[28] Ahmed Hendawy, Jan Peters, and Carlo D'Eramo. Multi-task reinforcement learning with mixture of orthogonal experts. *arXiv preprint arXiv:2311.11385*, 2023.

[29] Ahmed Hendawy, Jan Peters, and Carlo D'Eramo. Multi-task reinforcement learning with mixture of orthogonal experts. In The Twelfth International Conference on Learning Representations, 2024.

[30] Dan Hendrycks, Collin Burns, Steven Basart, Andrew Critch, Jerry Li, Dawn Song, and Jacob Steinhardt. Aligning ai with shared human values. Proceedings of the International Conference on Learning Representations (ICLR), 2021.

[31] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR), 2021.

[32] Quzhe Huang, Zhenwei An, Nan Zhuang, Mingxu Tao, Chen Zhang, Yang Jin, Kun Xu, Liwei Chen, Songfang Huang, and Yansong Feng. Harder task needs more experts: Dynamic routing in moe models. In *Proceedings of the 62nd Annual Meeting of the Association for Computational* Linguistics (Volume 1: Long Papers), pages 12883–12895, 2024.

[33] Yongqi Huang, Peng Ye, Chenyu Huang, Jianjian Cao, Lin Zhang, Baopu Li, Gang Yu, and Tao Chen. Ders: Towards extremely efficient upcycled mixture-of-experts models. arXiv preprint arXiv:2503.01359, 2025.

[34] Ranggi Hwang, Jianyu Wei, Shijie Cao, Changho Hwang, Xiaohu Tang, Ting Cao, and Mao Yang. Pre-gated moe: An algorithm-system co-design for fast and scalable mixture-of-expert inference. In *2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture*
(ISCA), pages 1018–1031. IEEE, 2024.

[35] Gagan Jain, Nidhi Hegde, Aditya Kusupati, Arsha Nagrani, Shyamal Buch, Prateek Jain, Anurag Arnab, and Sujoy Paul. Mixture of nested experts: Adaptive processing of visual tokens. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[36] Ganesh Jawahar, Subhabrata Mukherjee, Xiaodong Liu, Young Jin Kim, Muhammad Abdul-
Mageed, Laks VS Lakshmanan, Ahmed Hassan Awadallah, Sébastien Bubeck, and Jianfeng Gao. Automoe: Heterogeneous mixture-of-experts with adaptive computation for efficient neural machine translation. In *ACL (Findings)*, 2023.

[37] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. *arXiv preprint arXiv:2401.04088*, 2024.

[38] Junmo Kang, Leonid Karlinsky, Hongyin Luo, Zhen Wang, Jacob Hansen, James Glass, David Cox, Rameswar Panda, Rogerio Feris, and Alan Ritter. Self-moe: Towards compositional large language models with self-specialized experts. *arXiv preprint arXiv:2406.12034*, 2024.

[39] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. *arXiv preprint arXiv:2006.16668*, 2020.

[40] Hector J Levesque, Ernest Davis, and Leora Morgenstern. The Winograd schema challenge.

In *AAAI Spring Symposium: Logical Formalizations of Commonsense Reasoning*, volume 46, page 47, 2011.

[41] Jia LI, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. Numinamath. [https://huggingface.co/AI-MO/NuminaMath-CoT](https://github.com/
project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf),
2024.

[42] Jing Li, Zhijie Sun, Xuan He, Li Zeng, Yi Lin, Entong Li, Binfan Zheng, Rongqian Zhao, and Xin Chen. Locmoe: A low-overhead moe for large language model training. arXiv preprint arXiv:2401.13920, 2024.

[43] Jing Li, Zhijie Sun, Dachao Lin, Xuan He, Yi Lin, Binfan Zheng, Li Zeng, Rongqian Zhao, and Xin Chen. Expert-token resonance: Redefining moe routing through affinity-driven active selection. *arXiv preprint arXiv:2406.00023*, 2024.

[44] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step. arXiv preprint arXiv:2305.20050, 2023.

[45] Bin Lin, Zhenyu Tang, Yang Ye, Jiaxi Cui, Bin Zhu, Peng Jin, Jinfa Huang, Junwu Zhang, Yatian Pang, Munan Ning, et al. Moe-llava: Mixture of experts for large vision-language models.

arXiv preprint arXiv:2401.15947, 2024.

[46] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

[47] Boan Liu, Liang Ding, Li Shen, Keqin Peng, Yu Cao, Dazhao Cheng, and Dacheng Tao. Diversifying the mixture-of-experts representation for language models with orthogonal optimizer. arXiv preprint arXiv:2310.09762, 2023.

[48] Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, Yanru Chen, Huabin Zheng, Yibo Liu, Shaowei Liu, Bohong Yin, Weiran He, Han Zhu, Yuzhi Wang, Jianzhou Wang, Mengnan Dong, Zheng Zhang, Yongsheng Kang, Hao Zhang, Xinran Xu, Yutao Zhang, Yuxin Wu, Xinyu Zhou, and Zhilin Yang. Muon is scalable for llm training, 2025. URL https://arxiv.org/abs/2502.16982.

[49] Xinyi Liu, Yujie Wang, Fangcheng Fu, Xupeng Miao, Shenhan Zhu, Xiaonan Nie, and Bin CUI.

Netmoe: Accelerating moe training through dynamic sample placement. In The Thirteenth International Conference on Learning Representations, 2025.

[50] Xudong Lu, Qi Liu, Yuhui Xu, Aojun Zhou, Siyuan Huang, Bo Zhang, Junchi Yan, and Hongsheng Li. Not all experts are equal: Efficient expert pruning and skipping for mixture-ofexperts large language models. *arXiv preprint arXiv:2402.14800*, 2024.

[51] Tongxu Luo, Jiahe Lei, Fangyu Lei, Weihao Liu, Shizhu He, Jun Zhao, and Kang Liu. Moelora:
Contrastive learning guided mixture of experts on parameter-efficient fine-tuning for large language models. *arXiv preprint arXiv:2402.12851*, 2024.

[52] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. *Advances in Neural Information Processing Systems*, 36:46534–46594, 2023.

[53] Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, Pouya Tafti, et al. Gemma: Open models based on gemini research and technology. *CoRR*, 2024.

[54] Basil Mustafa, Carlos Riquelme, Joan Puigcerver, Rodolphe Jenatton, and Neil Houlsby. Multimodal contrastive learning with limoe: the language-image mixture of experts. *Advances in* Neural Information Processing Systems, 35:9564–9576, 2022.

[55] Nabil Omi, Siddhartha Sen, and Ali Farhadi. Load balancing mixture of experts with similarity preserving routers. *arXiv preprint arXiv:2506.14038*, 2025.

[56] Bowen Pan, Yikang Shen, Haokun Liu, Mayank Mishra, Gaoyuan Zhang, Aude Oliva, Colin Raffel, and Rameswar Panda. Dense training, sparse inference: Rethinking training of mixtureof-experts language models. *CoRR*, 2024.

[57] Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. Efficiently scaling transformer inference.

Proceedings of Machine Learning and Systems, 5:606–624, 2023.

[58] Peijun Qing, Chongyang Gao, Yefan Zhou, Xingjian Diao, Yaoqing Yang, and Soroush Vosoughi.

Alphalora: Assigning lora experts based on layer training quality. In *EMNLP*, 2024.

[59] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In *First Conference on Language Modeling*, 2024.

[60] Sheng Shen, Le Hou, Yanqi Zhou, Nan Du, Shayne Longpre, Jason Wei, Hyung Won Chung, Barret Zoph, William Fedus, Xinyun Chen, et al. Mixture-of-experts meets instruction tuning:
A winning combination for large language models. *arXiv preprint arXiv:2305.14705*, 2023.

[61] Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In *Proceedings of EMNLP*, pages 1631–1642, 2013.

[62] Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al.

Beyond the imitation game: Quantifying and extrapolating the capabilities of language models.

TRANSACTIONS ON MACHINE LEARNING RESEARCH, 2022.

[63] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, , and Jason Wei. Challenging big-bench tasks and whether chain-of-thought can solve them. *arXiv preprint* arXiv:2210.09261, 2022.

[64] Peng Tang, Jiacheng Liu, Xiaofeng Hou, Yifei Pu, Jing Wang, Pheng-Ann Heng, Chao Li, and Minyi Guo. Hobbit: A mixed precision expert offloading system for fast moe inference. arXiv preprint arXiv:2411.01433, 2024.

[65] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[66] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman.

Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461, 2018.

[67] Kun Wang, Guibin Zhang, Zhenhong Zhou, Jiahao Wu, Miao Yu, Shiqian Zhao, Chenlong Yin, Jinhu Fu, Yibo Yan, Hanjun Luo, et al. A comprehensive survey in llm (-agent) full stack safety:
Data, training and deployment. *arXiv preprint arXiv:2504.15585*, 2025.

[68] Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, and Damai Dai. Auxiliary-loss-free load balancing strategy for mixture-of-experts. *arXiv preprint arXiv:2408.15664*, 2024.

[69] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13484–13508, 2023.

[70] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. *arXiv preprint arXiv:2406.01574*, 2024.

[71] Alex Warstadt, Amanpreet Singh, and Samuel R. Bowman. Neural network acceptability judgments. *arXiv preprint 1805.12471*, 2018.

[72] Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. Finetuned language models are zero-shot learners, 2022. URL https://arxiv.org/abs/2109.01652.

[73] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models.

Advances in neural information processing systems, 35:24824–24837, 2022.

[74] Jerry Wei, Le Hou, Andrew Lampinen, Xiangning Chen, Da Huang, Yi Tay, Xinyun Chen, Yifeng Lu, Denny Zhou, Tengyu Ma, et al. Symbol tuning improves in-context learning in language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 968–979, 2023.

[75] Tianwen Wei, Bo Zhu, Liang Zhao, Cheng Cheng, Biye Li, Weiwei Lü, Peng Cheng, Jianhao Zhang, Xiaoyu Zhang, Liang Zeng, et al. Skywork-moe: A deep dive into training techniques for mixture-of-experts language models. *arXiv preprint arXiv:2406.06563*, 2024.

[76] Colin White, Samuel Dooley, Manley Roberts, Arka Pal, Benjamin Feuer, Siddhartha Jain, Ravid Shwartz-Ziv, Neel Jain, Khalid Saifullah, Sreemanti Dey, Shubh-Agrawal, Sandeep Singh Sandha, Siddartha Venkat Naidu, Chinmay Hegde, Yann LeCun, Tom Goldstein, Willie Neiswanger, and Micah Goldblum. Livebench: A challenging, contamination-free LLM benchmark. In *The Thirteenth International Conference on Learning Representations*, 2025.

[77] Adina Williams, Nikita Nangia, and Samuel R. Bowman. A broad-coverage challenge corpus for sentence understanding through inference. In *Proceedings of NAACL-HLT*, 2018.

[78] Qiong Wu, Zhaoxi Ke, Yiyi Zhou, Xiaoshuai Sun, and Rongrong Ji. Routing experts: Learning to route dynamic experts in existing multi-modal large language models. In *The Thirteenth* International Conference on Learning Representations, 2025.

[79] Fuzhao Xue, Zian Zheng, Yao Fu, Jinjie Ni, Zangwei Zheng, Wangchunshu Zhou, and Yang You. Openmoe: An early effort on open mixture-of-experts language models. arXiv preprint arXiv:2402.01739, 2024.

[80] Shu Yang, Muhammad Asif Ali, Cheng-Long Wang, Lijie Hu, and Di Wang. Moral: Moe augmented lora for llms' lifelong learning. *arXiv preprint arXiv:2402.11260*, 2024.

[81] Zihao Zeng, Yibo Miao, Hongcheng Gao, Hao Zhang, and Zhijie Deng. Adamoe: Tokenadaptive routing with null experts for mixture-of-experts language models. In *Findings of the* Association for Computational Linguistics: EMNLP 2024, pages 6223–6235, 2024.

[82] Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew M Dai, Quoc V Le, James Laudon, et al. Mixture-of-experts with expert choice routing. Advances in Neural Information Processing Systems, 35:7103–7114, 2022.

[83] Tong Zhu, Xiaoye Qu, Daize Dong, Jiacheng Ruan, Jingqi Tong, Conghui He, and Yu Cheng.

Llama-moe: Building mixture-of-experts from llama with continual pre-training. In *Proceedings* of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 15913–
15923, 2024.

[84] Yun Zhu, Nevan Wichers, Chu-Cheng Lin, Xinyi Wang, Tianlong Chen, Lei Shu, Han Lu, Canoee Liu, Liangchen Luo, Jindong Chen, et al. Sira: Sparse mixture of low rank adaptation. arXiv preprint arXiv:2311.09179, 2023.

[85] Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William Fedus. St-moe: Designing stable and transferable sparse expert models. arXiv preprint arXiv:2202.08906, 2022.

## Neurips Paper Checklist 1. **Claims**

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: In both the abstract and the introduction, we clearly present the key contributions of our paper, including our optimization method based on expert specialization. Guidelines:
- The answer NA means that the abstract and introduction do not include the claims made in the paper.

- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. **Limitations**

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: We provide a thorough discussion of the limitations of our work and suggest potential directions for future research. Guidelines:
- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.

- The authors are encouraged to create a separate "Limitations" section in their paper. - The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

- The authors should reflect on the factors that influence the performance of the approach.

For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. **Theory Assumptions And Proofs**

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

## Answer: [Yes]

Justification: In this paper, we provide the full set of assumption and a complete proof in the main paper and appendix. Guidelines:
- The answer NA means that the paper does not include theoretical results. - All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

- All assumptions should be clearly stated or referenced in the statement of any theorems. - The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. **Experimental Result Reproducibility**

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: In this paper, the experimental code and datasets will be publicly available in the future. The details necessary for reproducing all reported results are thoroughly described in Section 4.2 (Implementation Details). Guidelines:
- The answer NA means that the paper does not include experiments. - If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

- Depending on the contribution, reproducibility can be accomplished in various ways.

For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. **Open Access To Data And Code**

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: The code and datasets will be publicly released in the future, all reported results are fully reproducible based on the provided data and the detailed implementation described in Section 4.2. Further experimental procedures are documented in the appendix. Guidelines:
- The answer NA means that paper does not include experiments requiring code. - Please see the NeurIPS code and data submission guidelines (https://nips.cc/
public/guides/CodeSubmissionPolicy) for more details.

- While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:
//nips.cc/public/guides/CodeSubmissionPolicy) for more details.

- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. **Experimental Setting/Details**

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: We present the dataset construction process and all experimental details, including hyperparameter settings and other implementation specifics, in the Appendix and in Section 4.2 (Implementation Details). Guidelines:
- The answer NA means that the paper does not include experiments.

- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. **Experiment Statistical Significance**

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: The vast majority Of experiments in this article report variance measurements. Guidelines:
- The answer NA means that the paper does not include experiments.

- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors). - It should be clear whether the error bar is the standard deviation or the standard error of the mean.

- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).

- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. **Experiments Compute Resources**

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? [Yes] Justification: We report the resource consumption metrics for all experimental procedures conducted in this study. Guidelines:
- The answer NA means that the paper does not include experiments. - The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. **Code Of Ethics**

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes] Justification: All aspects of this work are in full compliance with the NeurIPS Code of Ethics. Guidelines:
- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. - If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. **Broader Impacts**

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [Yes] Justification: We discuss it in the limitation&discussion section. Guidelines:
- The answer NA means that there is no societal impact of the work performed.