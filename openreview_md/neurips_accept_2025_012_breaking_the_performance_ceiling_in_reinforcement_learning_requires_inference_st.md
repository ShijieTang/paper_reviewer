# Breaking The Performance Ceiling In Reinforcement Learning Requires Inference Strategies

Felix Chalumeau∗1 Daniel Rajaonarivonivelomanantsoa∗1,2 **Ruan de Kock**∗1 Claude Formanek1 Sasha Abramowitz1 Omayma Mahjoub1 **Wiem Khlifi**1 Simon Du Toit1 Louay Ben Nessir1 Refiloe Shabe1 **Noah De Nicola**1 Arnol Fokam1 Siddarth Singh1 Ulrich Mbou Sob1 **Arnu Pretorius**1,2 1InstaDeep 2Stellenbosch University

## Abstract

Reinforcement learning (RL) systems have countless applications, from energygrid management to protein design. However, such real-world scenarios are often extremely difficult, combinatorial in nature, and require complex coordination between multiple agents. This level of complexity can cause even state-of-theart RL systems, trained until convergence, to hit a performance ceiling which they are unable to break out of with zero-shot inference. Meanwhile, many digital or simulation-based applications allow for an inference phase that utilises a specific time and compute budget to explore multiple attempts before outputting a final solution. In this work, we show that such an inference phase employed at execution time, and the choice of a corresponding inference strategy, are key to breaking the performance ceiling observed in complex multi-agent RL problems. Our main result is striking: **we can obtain up to a 126% and,** on average, a 45% improvement over the previous state-of-the-art across 17 tasks, using only a couple seconds of extra wall-clock time during execution. We also demonstrate promising compute scaling properties, supported by over 60k experiments, making it the largest study on inference strategies for complex RL to date. **Our experimental data and code are available at** https://sites.google.com/view/inference-strategies-rl.

Tasks 9.5% 10.4% **14.0%**
20.7% 21.8% **22.3%**
32.3% 35.0% 35.7% 36.5%
42.3%
47.9%
53.3%
85.5% **86.8%**
92.2%
126.4%
Figure 1: **Improvement from using inference-time search over zero-shot state-of-the-art**. Across 17 complex reinforcement learning tasks, we obtain consistent and significant performance gains using only a 30 second search budget during execution.

## 1 Introduction

Learning to solve sequential decision-making tasks is a central challenge in artificial intelligence (AI), with far-reaching applications ranging from energy-grid optimisation (Ahmad et al., 2021) and autonomous logistics (Laterre et al., 2018) to molecular discovery (Olivecrona et al., 2017) and drug design (Popova et al., 2018). Complex sequential real-world problems that cannot be solved by traditional optimisation techniques are inherently complex and require navigating high-dimensional solution spaces. Reinforcement Learning (RL) presents a promising avenue to improve our capacity to find efficient solutions to these problems, but despite stunning progress over the past decade, such as human-level performance in Atari games (Mnih et al., 2015), defeating the world champion in the game of Go (Silver et al., 2016) or aligning AI systems with human preferences (Stiennon et al.,
2020), current approaches are facing challenges that prevent their common deployment in most real-world systems (Dulac-Arnold et al., 2020). A major source of this difficulty lies in the combinatorial nature of many decision-making tasks. As the problem size increases, the space of possible solutions grows exponentially (Karp, 1975). In multi-agent systems, the challenge compounds: agents must coordinate in environments where only partial information is available, the joint action space is combinatorial, and optimal behaviour depends on precise interaction with other agents (Bernstein et al., 2000; Canese et al., 2021). These properties make it fundamentally difficult to rely on the zero-shot performance of a trained policy, even if that policy was optimised to convergence on a representative training distribution. This causes the gap between zero-shot performance and optimality to grow substantially with increasing complexity (see Fig. 1).

However, numerous practical applications are not restricted to producing a single zero-shot solution.

Instead, inference is often permitted to take place over a few seconds, minutes or hours, with a given computational resource. Furthermore, models and simulators are often accessible and very efficient (e.g., energy grid management, train scheduling, package delivery, routing, printed circuit board design) and provide either an exact score or a very accurate approximation. In other applications where the gap to reality may be larger (e.g., protein design, robotics), improving the solution under the simulated score can still arguably provide significant improvement towards the real objective (Hayes et al., 2025; Dona et al., 2024; Hundt et al., 2019; Rao et al., 2020). This opens up an opportunity: rather than relying on a single attempt of the trained policy, the time budget and compute capacity can be leveraged to actively search for better solutions using multiple attempts, following an *inference-time* strategy. For instance, progressively building a tree of possible solutions, or adapting the policy using outcomes of past attempts. Even straightforward strategies can provide significant performance improvement with low time cost. For instance, generating a large batch of diversified solutions in parallel, using stochastic sampling, rather than a single greedy solution: given a modern GPU, this enables to produce hundreds of solutions, for the same wall-clock time, enabling massive exploration at no time cost. These strategies are rarely emphasized in existing benchmarks (Papoudakis et al., 2021; Mahjoub et al., 2025), and many practitioners invest months of research trying to improve the zero-shot performance of their models on scenarios where only marginal improvement may still be achieved. Whereas they could unlock performance gains from inference-time search, at negligible wall-clock time cost with moderate compute capacity (Fig. 1). Research in RL for Combinatorial Optimisation (CO) has produced efficient inference strategies (Bello* et al., 2017; Hottung et al., 2022; Choo et al., 2022; Chalumeau et al., 2023b), often referred to as active search, or online adaptation methods. However, their empirical study is still limited to a few problems, a narrow range of budget settings (Chalumeau et al., 2025), and barely no insight on their scaling properties. In the multi-agent case, there is no study on inference strategies for collaborative teams: most work on team adaptation focus on Ad Hoc Teamwork (Mirsky et al., 2022; Wang et al., 2024a; Ruhdorfer et al., 2025), which is adjacent to our objective. Interestingly, most recent studies about the impact of inference strategies come from the Large Language Model (LLM) literature (Snell et al., 2025; Muennighoff et al., 2025; Wu et al., 2025), where the adequate combination of efficient models with inference strategies is currently state-of-the-art (SOTA). In this work, we formalise and investigate the role of inference strategies in complex decision-making tasks. To capture the full complexity of tasks described above, we formulate our problem setting as a decentralised partially observable Markov decision process (Dec-POMDP) (Kaelbling et al., 1998). This is instead of the typical single-agent MDP used in many RL studies. We make this choice for several reasons: (1) it more accurately maps onto many complex real-world problems of interest, (2) Dec-POMDPs subsume MDPs by being strictly more complex (Bernstein et al., 2000), and (3) because of this, we expect our findings to translate to all simpler problem formulations. Within this setting, we provide a unifying view of popular inference-strategy paradigms, including policy sampling, tree search (Choo et al., 2022), online fine-tuning (Bello* et al., 2017; Hottung et al., 2022), and diversity-based search (Chalumeau et al., 2023b). Strikingly, we show that across a wide range of specifically selected difficult RL problems, inference strategies boost performance on average by 45%, over zero-shot SOTA. Furthermore, in the best of cases, this boost can be as large as 126%. All of this, using only a couple of seconds of additional execution time. Our results call for a shift in how RL systems are evaluated and deployed: inference strategies are not a minor post-processing step, but a key performance driver in realistic conditions. This work sets the foundation for a more nuanced view of inference in sequential decision-making and provides the tools to build systems that can scale with compute. All our code and experimental data can be accessed at: https://sites.google.com/view/inference-strategies-rl.

## 2 Related Work

Inference Strategies from RL for CO Beyond naive stochastic sampling, several paradigms have been explored to generate the best possible solution using a trained policy checkpoint during inference. Online fine-tuning (Bello* et al., 2017) retrains all policy parameters with RL using past attempts. Hottung et al. (2022) re-trains only a subset of the policy's parameters to reduce memory and compute overheads, enabling more attempts for a given inference budget, and adds an imitation learning term to the RL term, to force exploration close to the best solution found so far. Macfarlane et al. (2024) also investigates inference-time policy improvement via sequential updates. Choo et al. (2022) uses tree search, with simulation guided node estimates under budget constraints, which outperforms Beam Search and Monte Carlo Tree Search (Coulom, 2006). Diversity-based methods: inspired by previous diversity-seeking approaches, like unsupervised skill discovery (Eysenbach et al., 2019; Sharma et al., 2019; Kumar et al., 2020) and quality-diversity (Chalumeau et al., 2023a; Cully and Demiris, 2017). Grinsztajn et al. (2023) introduces an RL objective that trains a population of diverse and specialized policies, efficient for few-shot performance. Chalumeau et al. (2023b) uses this objective and encodes the diversity in a continuous latent space that can be searched at inference-time, introducing the SOTA method COMPASS; meanwhile Hottung et al. (2024) uses a similar approach but with a discrete encoding space.

These works have introduced most of the inference strategies we consider in this paper, but they fall short on three important aspects that we aim to improve: (i) they evaluate inference strategies on benchmarks where over 95% zero-shot optimality is already achieved, leaving little room for meaningful gains, (ii) these benchmarks rely on domain-specific tricks such as starting points or instance augmentations; and (iii) methods are compared under a unique budget setting, overlooking the fact that relative performance depends on the available compute and time budget. In addition, their ability to scale with compute remains unexplored, despite being a critical property. Policy adaptation in Meta-RL and Offline-to-Online RL Meta-RL and offline-to-online RL both design mechanisms for policy adaptation, thereby sharing close conceptual links with several popular inference strategies. COMPASS (Chalumeau et al., 2023b) and VariBAD (Zintgraf et al., 2021) both condition a policy on a latent space and search it to adapt. MEMENTO (Chalumeau et al., 2025)
and RL2 (Duan et al., 2016) similarly rely on memory and a learned rule to search the policy space.

DIMES (Qiu et al., 2022) explicitly performs meta-RL following MAML's methodology (Finn et al.,
2017). While we are not aware of inference strategies directly inspired by the offline-to-online RL
literature, Nakamoto et al. (2023) and Mark et al. (2025) introduce mechanisms which could be beneficial at inference time. It is worth highlighting fundamental distinctions between these fields. Inference strategies focus on (i) the solutions found rather than the learned policy, (ii) maximum rather than average performance, and (iii) single instances at test time, whereas most meta-RL and offline-to-online RL methods adapt from a distribution to another distribution to improve generalisation.

1. TRAINING PHASE **2. INFERENCE PHASE**
Compute & Time budget Training distribution New instance Inference strategies Stochastic sampling **Tree search**
Decision-making policy Trained policy Online fine-tuning **COMPASS' search**
Search and adaptation in Multi-Agent RL There is only limited work on inference strategies for MARL. Most work about search and adaptation within MARL focus on the challenge of Ad Hoc Teamwork (Yourdshahi et al., 2018; Hu et al., 2020; Mirsky et al., 2022; Wang et al., 2024a; Ruhdorfer et al., 2025; Hammond et al., 2025), often in the form of zero-shot coordination, where agents must generalize to new partners at execution time. While these lines of work share some methodological similarities, for instance using diversity-seeking training (Long et al., 2024; Lupu et al., 2021) or adapting through tree search (Yourdshahi et al., 2018), they pursue fundamentally different goals and remain orthogonal. In our work, our focus is primarily on solving difficult and complex industrial optimisation tasks.

Inference-time compute for LLMs Recent advances in LLMs are closely intertwined with the use of inference strategies (Snell et al., 2025; Wei et al., 2022; Wang et al., 2024b), and a growing effort has gone into studying their scaling properties (Muennighoff et al., 2025; Wu et al., 2025). However, the typical inference-time setting is usually different from ours. LLMs have very costly forward passes, and cannot access the exact score of their answers, but can approximate them using a reward model (Ouyang et al., 2022). Most popular strategies for LLMs are designed for few shots, namely sampling and ensembling (e.g., majority voting). Overall, numerous efficient inference strategies have been proposed in the literature, yet their efficiency under various settings remains unexamined. Multi-agent RL, despite its inherent complexity, rarely considers inference-time search beyond Ad Hoc Teamwork. Moreover, the broader field of decision-making has not systematically studied how inference strategies scale with compute. Our work aims at filling these gaps by extending the evaluation of inference strategies in RL, demonstrating major performance gains over a wide range of budget settings with impressive scaling properties.

## 3 Finding The Best Solution For A Given Time And Compute Budget 3.1 Preliminaries

We focus on RL approaches, and use a neural network (policy) that can construct a solution by taking a sequence of actions. This policy is optimised during a training phase and then used during an inference phase, along with an *inference strategy*, to construct the best possible solution to a new problem instance under a given time and compute budget. These two phases, illustrated in Fig. 2, have different assumptions, objectives and constraints, detailed in the following paragraphs.

Problem instances We assume that each problem instance can be formulated as a Dec-POMDP (Kaelbling et al., 1998), defined by the tuple M =N, S, O, Ω, A*, R, P, γ, H*. Here N is the number of agents, S the environment state space, O =QN
i=1 Oithe joint agent observation space, Ω : *S 7→* O the observation function, A =QN
i=1 Aithe joint action space, R : S × A 7→ R the shared reward function, P : S × A 7→ ∆S the environment transition function, the scalar γ ∈ [0, 1] is the discount factor and H the finite episode horizon. At each timestep, the environment occupies a state st which is mapped to a joint partial observation ot via Ω. The joint action is then sampled following the joint policy, at ∼ π(· | ot) and is executed; after which the environment transitions to the next state st+1, following P(· | st, at), and the team receives a reward rt = R(st, at).

Training Phase We assume a distribution of problem instances D, that can be sampled from during training. The joint policy πθ, parameterised by θ, is used to construct a solution sequentially by taking joint actions conditioned on the joint observation at each timestep. We use RL to train this policy to maximise the expected return obtained when building solutions to instances drawn from the distribution D over a horizon H: J(πθ) = ED
hPH
t=0 γ tR(st, at)
i.

This training objective corresponds to a single attempt (zero-shot). Ideally, this objective should anticipate the multiple attempts allowed at inference, but this is hard to scale. Recent works incorporate such few-shot objectives (Grinsztajn et al., 2023; Chalumeau et al., 2023b, 2025), but none can yet scale beyond 200 attempts. The training phase is usually loosely constrained in terms of time and compute capacity, as typically industrial stakeholders are willing to invest days, weeks or even months of training to obtain a high-performing policy that can generate accurate solutions when deployed in production. Hence, in our experiments, we train all policies until convergence.

Inference Phase At inference time, a new problem instance ρ is drawn from a distribution D′
(possibly different than D). Here, there are typically hard constraints to outputting a final solution: a fixed time limit Tmax constrains wall-clock execution, and a compute capacity Bmax constrains the number of operations that can be done in parallel. The trained policy πθ can be used within these constraints to generate solutions to the problem, and the best solution is ultimately used. The reward function R can still be used to score attempted solutions and inform subsequent attempts. Inference strategies can be defined as a function I : (ρ,πθ, Bmax, Tmax) 7→ (a
∗1*, ...,* a
∗H) that uses the base policy πθ and any additional inference-time search, adaptation, storage, or optimisation methods under the budget Tmax and Bmax to produce the best possible solution to the problem instance ρ, defined by the sequence of actions (a
∗
i)1≤i≤H. The objective can hence be written as:

$I(\mathcal{I})=\sum_{t=0}^{H}R(s_{t},\mathbf{a}_{t}^{*})$ s.t. $\mathbf{C}(\mathcal{I})\leq\mathbf{B}_{\max}$, $\mathbf{T}(\mathcal{I})\leq\mathbf{T}_{\max}$.  
where C(I) and T(I) represent the compute and time cost of the inference strategy. This formulation highlights that, unlike traditional RL, where zero-shot performance is the primary measure, we focus on strategies which enable further improvement under given constraints. We provide three real-world examples in Appendix H that correspond to this problem setting, illustrating practical scenarios where inference-time search is both natural and feasible.

Inference strategies differ in how they explore the solution space and how they incorporate the outcomes of previous attempts to influence future sampling. Their effectiveness depends on several contextual factors, including the parameter count of the pre-trained policy, the problem's underlying structure, the episode horizon, the nature of the reward function, but most critically, on the available time and compute budget.

## 3.2 One Budget, Many Possibilities: Inference-Time Search And Adaptation

In this section, we detail four types of inference strategies and how we adapt them to work in the multi-agent setting. We implement and release all of these methods in JAX (Bradbury et al., 2018). Stochastic policy sampling The first natural lever to improve solution quality is to re-sample from a stochastic policy. In other words, beyond the creation of a unique greedy solution (i.e., using a = arg maxa′ πθ(a
′|o) over a trajectory of observations), one can sample stochastically (as in a ∼ πθ(·|o)) in order to create diverse solutions. **Multi-agent policy sampling** generalises easily to the multi-agent case by sampling from the joint action distribution, a ∼ πθ(·|o).

Tree search These methods store information about partial solutions using past attempts to preferentially search promising regions of the solution space without updating the pre-trained policy. Simulation guided beam search (SGBS) (Choo et al., 2022) provides the best time to performance balance in the literature, outperforming Monte Carlo Tree Search (Coulom, 2006). Like most tree searches, SGBS has three steps: expansion, simulation, pruning. Expansion uses the policy to decide on the most promising next actions (i.e., a = top-K(πθ(·|o)) ) from the current node (partial solution). A simulated rollout of an episode is produced greedily using πθ and the return is collected for each node. Pruning keeps only the best nodes found so far based on the return. Solely the expansion step needs to be adapted for **Multi-agent SGBS**. This is trivial when the explicit joint actions are accessible (de Witt et al., 2020; Yu et al., 2022), since we can still select the top ones (i.e.,
a = top-K(πθ(·|o))). For methods using auto-regressive action selection (Mahjoub et al., 2025),
having access to the top joint actions is intractable, hence we sample K times stochastically from the same node (i.e., a[1]*, ...,* a[K] ∼ πθ(·|o)).

Online fine-tuning These methods keep updating policy parameters at inference time. Given a base policy πθ, online fine-tuning optimises θ using inference-time rollouts and policy gradient updates:
θ
′ = θ + α∇θJ(πθ), where J(πθ) represents an adaptation objective. In line with (Bello* et al.,
2017), we keep maximising expected returns (Bello* et al., 2017) over past attempts on the fixed instance (instead of over a training distribution). **Multi-agent online fine-tuning** re-trains πθ on the new instance using the MARL algorithm that was used during pre-training (Bello* et al., 2017; Mahjoub et al., 2025; de Witt et al., 2020; Yu et al., 2022).

Diversity-based approaches These methods pre-train a collection of diverse specialised policies which can be used to search for the most appropriate solution at inference-time. COMPASS (Chalumeau et al., 2023b) encodes specialised policies in a continuous latent space L by augmenting a pretrained policy to condition on both the observation and a latent vector sampled from L (i.e., a ∼
πθ(· | o, z), z ∼ L): effectively creating a continuous collection of policies. COMPASS achieves SOTA in single-agent RL for CO. To avoid having the latent space of **Multi-agent COMPASS** growing exponentially with the number of agents, we keep one latent space L for all agents (i.e.,
a ∼ πθ(· | o, z), z ∼ L). This allows for tractable training, and for efficient inference search with the covariance matrix adaptation evolution strategy CMA-ES (Hansen and Ostermeier, 2001). Aside from being multi-agent, we keep the training and inference phases close to the original method described in Chalumeau et al. (2023b), and provide further details and explanations in Appendix F.2. Unlike other inference strategies, COMPASS includes an additional training phase, which remains accessible since the training phase is unconstrained (all policies are trained until convergence). Creating a COMPASS checkpoint from a pre-trained base policy involves adding parameters to process the latent vectors (Appendix F.2), resulting in a modest increase in model size. This increase never exceeds 2% of the total policy size, and has negligible impact on the overall computational or memory footprint. We report all parameter counts in Appendix I.

## 4 Experiments

In our experimental study, we combine popular MARL algorithms and inference strategies and benchmark them on a set of complex RL tasks from the literature. Each task was specifically selected for its difficulty. We evaluate all base policies with and without inference-time search across a wide range of budget settings. Our experiments constitute the largest-ever study of inference strategies for decision-making. Baselines We use three MARL approaches to obtain our base policies: Independent PPO (de Witt et al., 2020) (IPPO) and Multi-Agent PPO (Yu et al., 2022) (MAPPO), which are widely used and well-known MARL methods, and the recent SOTA sequence modelling approach SABLE (Mahjoub et al., 2025).

Each of these, referred to as *base policies*, is evaluated with all four inference strategies introduced in Section 3.2, namely stochastic sampling, SGBS, online fine-tuning and COMPASS.

Set of 17 complex RL tasks Connector StarCraft (SMAC) RWARE
Wide range of settings time compute **complexity**
Evaluation of inference strategies
Tasks Mahjoub et al. (2025) established SOTA over the most comprehensive MARL benchmark published in the field to date. Interestingly, their results demonstrate that there remain certain tasks for which no existing method (including SABLE) is able to achieve good performance. Specifically, these are tiny-2ag-hard, tiny-4ag-hard, small-4ag, small-4ag-hard, medium-4ag, medium-4ag-hard, medium-6ag, large-4ag, large-4ag-hard, large-8ag, large-8ag-hard, xlarge-4ag and xlarge-4ag-hard from Multi-Robot Warehouse (Papoudakis et al., 2021) (RWARE), smacv2_10_units and smacv2_20_units from the StarCraft Multi-Agent challenge (Samvelyan et al., 2019; Ellis et al., 2023) (SMAC), and con-10x10x10a and con-15x15x23a from Connector (Bonnet et al., 2023). Each environment (illustrated on Fig. 3) introduces distinct challenges that contribute to its complexity. RWARE requires agents to coordinate in order to pick up and deliver packages without collision and has a very sparse reward signal. In SMACv2 tasks, a team cooperates in real-time combat against enemies across diverse scenarios with randomised generation. Connector models the routing of a printed circuit board where agents must connect to designated targets without crossing paths. All three environments feature combinatorial and high-dimensional action spaces, partial observability and the need for tightly coordinated behaviours, making these 17 tasks a compelling test-bed for complex RL with desirable properties modelling aspects of real-world tasks. We use JAX-based implementations of Connector and RWARE from Jumanji (Bonnet et al., 2023) and for SMAC from JaxMARL (Rutherford et al., 2023).

20 M steps Converged COMPASS
xlarge-4ag-hard 0.0 0.2 0.4 0.6 0.8 1.0 N
o r ma li s ed pe rfo r ma n c e large-4ag-hard large-8ag large-8ag-hard medium-4ag medium-4ag-hard medium-6ag smacv2_
10_
units smacv2_
20_
units small-4ag small-4ag-hard tiny-2ag-hard tiny-4ag-hard xlarge-4ag con-10x10x10a con-15x15x23a large-4ag
Training base policies To obtain clear performance ceilings for each algorithm and best isolate the effects of inference strategies, we train all base policies until convergence. For the sake of continuity with previous work with truncated training budgets, typically of 20M steps, we report the zero-shot results for each converged checkpoint compared to its previous corresponding reported performance on Fig. 4. We observe that in most tasks (14 out of 17), the converged policy stays below 70% normalised performance, demonstrating that the benchmark is still far from saturated. COMPASS requires an additional training phase, which reincarnates the existing base policies to create the latent space specialisation. For each base policy and task, we also train the COMPASS checkpoint until convergence. This leads to 102 trained policy checkpoints. Evaluating performance during inference Evaluating inference strategies in a way that is unbiased and aligned with real-world settings is challenging. Most papers report results where the budget is based on a number of attempts, hence not directly incorporating the time cost of the inference strategy. The time costs are reported, but it is tough to analyse due to the plurality of hardware used to obtain them. Having re-implemented all of the baselines in the same code base and setting the budget in terms of time (in seconds), we can avoid this bias in our study. We use the same fixed hardware for all our experiments, namely a NVIDIA-A100-SXM4-80GB GPU. For statistical robustness, we always run 128 independent seeds. In all cases, we control for Bmax by varying the permitted number of batched parallel attempts instead of altering hardware between experiments. For aggregation across multiple tasks we follow the recommendations made by Agarwal et al. (2021) and use the rliable library to compute and report the inter-quartile mean (IQM) and 95% stratified bootstrap confidence intervals.

Hyperparameters To train the base policies, we re-use the hyperparameters reported in Mahjoub et al. (2025), which have been optimised for our tasks. For the inference strategies, we follow recommendations from the literature (Choo et al., 2022; Chalumeau et al., 2023b). All hyperparameters choices are reported in Appendix D.

## 4.1 A Couple Of Seconds Is All You Need

In this section, we demonstrate that inference-time search can help reach close to maximum task performance, using base policies for which zero-shot performance stagnates around 60%. Experiments To demonstrate that inference-time strategies are accessible, we use a small budget: 30 seconds, and a compute capacity enabling to generate 64 solutions in parallel. Each base policy is evaluated greedily for a single attempt, and then evaluated with the search budget, using each inference strategy. We report the performance distribution over the 17 tasks in Fig. 5, and the performance gains offered by the best inference-time search over the best zero-shot on Fig. 1.

Greedy (zero-shot) Stochastic Fine-tuning SGBS COMPASS
IPPO MAPPO **Sable**
0.0 0.2 0.4 0.6 0.8 1.0 No rma lise d pe rfo r m a nce s co re
 (
I
Q
M
)

Zero-shot SOTA
Discussion We can draw four main conclusions. First, inference-time search does provide a massive performance boost over zero-shot, which stands for every base policy. For the SOTA zero-shot method SABLE, this translates to pushing the best-ever achieved aggregated performance by more than 45% and creating a system (SABLE
+COMPASS) that achieves close to 100% win-rate in all tasks where this metric is available. Second, the improvement enabled over zero-shot performance increases significantly (almost exponentially) with respect to the complexity of the task (see Fig. 1). This suggests substantial gains are still ahead as the field moves toward increasingly realistic scenarios. Third, we observe that COMPASS is the leading strategy across tasks and base algorithms, and that SABLE remains the SOTA base policy even when using inference-time search. Interestingly, under a small time budget, stochastic sampling outperforms online fine-tuning. We nevertheless show in the following section that, given more budget, this result can be nuanced, and we share our interpretation of these findings.

Figure 5: Performance obtained by inference strategies over the benchmark. Each base policy is evaluated with each possible inference strategy. We report the inter-quartile mean over tasks with 95% stratified bootstrap confidence intervals.

## 4.2 Mapping Performance With Compute And Time Budget

A recurrent limitation in previous work on inference strategies is the use of a fixed budget during evaluation, creating a narrow view on methods, and often creating a bias towards certain types of methods. In this section, we aim at providing a much broader perspective over inference-time search by reporting performance over a grid of time and compute budgets.

Experiments We choose a maximum time of 300 seconds, and evaluate all inference strategies using the leading base policy (SABLE), with a compute budget of {4, 8, 16, 32, 64, 128, 256}. All in all, we have 4 inference strategies, 7 possible compute budgets, 17 tasks, and 128 seeds per task, leading to 60 928 evaluated episodes. This constitutes the largest study released on inference strategies. We report these results using contour plots, where the x-axis is time, y-axis the number of parallel attempts allowed (our proxy for compute) and colour corresponds to the performance achieved (win-rate when accessible or min-max normalised return) going from dark purple (min) to yellow (max). We keep the 8 hardest tasks, the lower half based on the zero-shot performance of the converged SABLE checkpoints (see Fig. 4), on Fig. 6 and defer remaining tasks to Appendix A.

Discussion As a sanity check, we remark that performance always increases (colours become lighter) when time or compute increases (going towards the upper right corner). We now highlight three main observations. First, COMPASS demonstrates impressive versatility and achieves significant gains over other inference strategies, dominating all maps, except for con-10x10x10a, where it gets

COMPASS
con-10x10x10a con-15x15x23a 100 200 300 0.6 0.8 0.9 large-4ag-hard large-4ag 100 200 300 0.6 0.7 0.8 large-8ag 100 200 300 0.6 0.7 0.8 0.9 large-8ag-hard 100 200 300 0.7 0.8 0.9 xlarge-4ag xlarge-4ag-hard 1.00 10 1 10 2 0.50.7 0.8 0.9 0.5 0.7 0.8 0.9 0.7 0.8 0.9 0.5 0.7 0.8 0.9 0.9 0.9 0.5 0.7 0.9 0.5 0.7 0.8 0.9 0.8 0.9 Compute ca pacity (#paral lel attempts
)

0.40.5 0.6 0.6 Stochastic 0.6 0.7 0.8 0.5 0.6 0.7 0.4 0.5 0.6 0.6 10 1 10 2 0.3 0.4 0.5 0.3 0.4 0.5 0.75 Normalised Return 0.2 0.3 0.2 0.3 0.3 0.4 0.5 0.6 0.2 0.3 0.4 0.5 0.4 0.5 0.6 0.7 0.3 0.4 0.5 0.6 10 1 10 2 Fine-tuning 0.7 0.8 0.9 0.4 0.5 0.7 0.8 0.2 0.3 0.3 0.4 0.5 0.6 0.6 0.3 0.50 100 200 300 10 1 10 2 SGBS
0.3 0.4 0.5 0.6 100 200 300 0.6 0.7 0.8 100 200 300 0.2 0.3 0.4 0.5 100 200 300 0.5 0.6 0.7 0.25 Time in seconds
slightly outperformed by online fine-tuning. Second, we observe high variance for online fine-tuning: getting close to COMPASS for large budgets on con-10x10x10a, yet struggling to match stochastic sampling on others (e.g., RWARE's large-8ag). This shows that fine-tuning can be detrimental by reducing the number of attempts made within the time budget. Plus, policy gradients can be unstable (small batch size) or converge to local optima. This observation disproves the common belief that inference-time search is as trivial as over-fitting to the problem instance. Finally, we observe that SGBS, despite failing on the connector tasks, achieves competitive performance overall and even leads in the low-budget regime (i.e., below 100 seconds and fewer than 10 parallel attempts) in some scenarios (large-4ag and large-4ag-hard).

## 4.3 Scaling With Increasing Budget

In practical applications, time is often more restricted than compute: a couple of seconds or minutes can be allowed, sometimes a few hours (train scheduling for the next day), but rarely more. Being able to improve solutions' quality under a fixed time budget by increasing compute is desirable. In this section, we analyse how different methods scale with additional compute. Experiments We keep the time budget fixed, 300 seconds, and we plot the final performance for each possible compute budget
(still using the number of parallel attempts as a proxy). We use SABLE as the base policy and evaluate all the inference strategies across the 8 hardest tasks. We report results per strategy, and aggregated over the tasks on Fig. 7.

Discussion As expected, stochastic sampling has the lowest scaling coefficient: its search solely relies on chance, no adaptation or additional search is happening. On the other hand, we can see that online fine-tuning benefits from more compute, probably due to a better estimation of the policy gradient.

It nevertheless requires a budget of 64 parallel attempts to clearly outperform stochastic

Normalised perform ance score (IQM
)

Stochastic Fine-tuning SGBS COMPASS
4 8 16 32 64 128 256 Compute capacity (#parallel attempts)
0.0 0.2 0.4 0.6 0.8 1.0
sampling. On the other hand, COMPASS consistently provides a significant advantage. Its scaling trend seems linear at first, with a high coefficient, and only seems to decline for higher budgets, as performance limits are reached (over 95% win-rate when accessible, best-ever observed performance elsewhere). We have a two-fold explanation for these particularly impressive scaling properties: (i) the diversity contained in the latent space can be exploited by more parallelism, leading to a massive exploration of the solution space, even when the initial policy is far from optimal, and (ii) the higher the batch size, the better the CMA-ES search, enabling COMPASS to exploit even more information from any additional searching step allowed within the given time budget. We discuss this further in Appendix K.

## 5 Conclusion

In this work, we demonstrate that inference-time strategies are a critical and underutilized lever for boosting the performance of RL systems in complex tasks, using multi-agent RL as a representative test-bed. While training-time improvements have long dominated the field, our results show that inference-time search may offer significant performance gains using only a few seconds to minutes of additional wall-clock time during execution. We introduce a unified view of inference strategies, extend it to the multi-agent setting, and empirically validate its effectiveness under varying compute and time budgets. Our large-scale evaluation, the most comprehensive to date on inference-time methods, reveals three key takeaways: (i) inference-time search with a relevant strategy yields significant improvements, even under tight time constraints; (ii) the gains depend on the inference budget, and our contour maps provide practitioners with practical guidance based on their constraints; and (iii) SABLE +COMPASS not only dominates the benchmark but also exhibits the most favourable scaling trends, making it particularly effective for increasingly complex decision-making problems. Altogether, our findings call for a shift in how decision-making models are evaluated and deployed: inference-time strategies should be treated as core components of the solution pipeline, not as optional refinements. We hope our results and open-source tools will encourage broader adoption and inspire further innovation in the design of scalable inference-time algorithms. Limitations and future work We focus on multi-agent RL as it better captures the complexity of real-world decision-making systems, where successful operation often relies on coordination among multiple agents. Nonetheless, we acknowledge that naturally single-agent tasks can also be highly complex. In the spirit of seeding a wider investigation for future work beyond the multi-agent setting, we provide additional results on the single-agent Craftax benchmark in Appendix C, showing that a simple 30-second inference-time search using stochastic sampling can provide a 37% performance boost. This provides initial (but limited) evidence that our claims hold more generally and we leave a more thorough investigation for future work. Beyond the broader RL setting, we intend to investigate two main future research directions. First, studying how to best combine existing inference paradigms to leverage their complementary strengths. Second, investigating how inference strategies compare when evaluated out-of-distribution.

## Acknowledgements

We would like to thank Guillaume Toujas-Bernate, Jake Lourie and Thomas Lecat for useful discussions on the use of inference strategies in real-world applications. We thank our MLOps team for developing our model training and experiment orchestration platform AIchor. We thank the Python and JAX communities for developing tools that made this research possible. We thank the anonymous reviewers for their constructive feedback and valuable suggestions. Finally, we thank Google's TPU Research Cloud (TRC) for supporting our research with Cloud TPUs.

## References

R. Agarwal, M. Schwarzer, P. S. Castro, A. Courville, and M. G. Bellemare. Deep reinforcement learning at the edge of the statistical precipice. In Advances in Neural Information Processing Systems, 2021. URL https://arxiv.org/abs/2108.13264.

T. Ahmad, D. Zhang, C. Huang, H. Zhang, N. Dai, Y. Song, and H. Chen. Artificial intelligence in sustainable energy industry: Status quo, challenges and opportunities. Journal of Cleaner Production, 289:125834, 2021. ISSN 0959-6526. doi: https://doi.org/10.1016/j.jclepro.2021.125834. URL https://www.sciencedirect.com/science/article/pii/S0959652621000548.

I. Bello*, H. Pham*, Q. V. Le, M. Norouzi, and S. Bengio. Neural combinatorial optimization with reinforcement learning, 2017. URL https://openreview.net/forum?id=rJY3vK9eg.

D. S. Bernstein, S. Zilberstein, and N. Immerman. The complexity of decentralized control of markov decision processes. In Proceedings of the Sixteenth Conference on Uncertainty in Artificial Intelligence, UAI'00, page 32–37, San Francisco, CA, USA, 2000. Morgan Kaufmann Publishers Inc. ISBN 1558607099.

C. Bonnet, D. Luo, D. Byrne, S. Abramowitz, V. Coyette, P. Duckworth, D. Furelos-Blanco, N. Grinsztajn, T. Kalloniatis, V. Le, O. Mahjoub, L. Midgley, S. Surana, C. Waters, and A. Laterre. Jumanji: a suite of diverse and challenging reinforcement learning environments in jax, 2023. URL https://github.com/instadeepai/jumanji.

J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson, C. Leary, D. Maclaurin, G. Necula, A. Paszke, J. VanderPlas, S. Wanderman-Milne, and Q. Zhang. JAX: composable transformations of Python+NumPy programs, 2018. URL http://github.com/jax-ml/jax.

L. Canese, G. C. Cardarilli, L. Di Nunzio, R. Fazzolari, D. Giardino, M. Re, and S. Spanò. Multi-agent reinforcement learning: A review of challenges and applications. *Applied Sciences*, 11(11), 2021. ISSN 2076-3417. doi: 10.3390/app11114948. URL https://www.mdpi.com/2076-3417/11/ 11/4948.

F. Chalumeau, R. Boige, B. Lim, V. Macé, M. Allard, A. Flajolet, A. Cully, and T. Pierrot. Neuroevolution is a competitive alternative to reinforcement learning for skill discovery. In *International* Conference on Learning Representations, 2023a. URL https://openreview.net/forum?id= 6BHlZgyPOZY.

F. Chalumeau, S. Surana, C. Bonnet, N. Grinsztajn, A. Pretorius, A. Laterre, and T. D. Barrett.

Combinatorial optimization with policy adaptation using latent space search. In Thirty-seventh Conference on Neural Information Processing Systems, 2023b.

F. Chalumeau, B. Lim, R. Boige, M. Allard, L. Grillotti, M. Flageat, V. Macé, G. Richard, A. Flajolet, T. Pierrot, et al. Qdax: A library for quality-diversity and population-based algorithms with hardware acceleration. *Journal of Machine Learning Research*, 25(108):1–16, 2024.

F. Chalumeau, R. Shabe, N. D. Nicola, A. Pretorius, T. D. Barrett, and N. Grinsztajn. Memoryenhanced neural solvers for routing problems. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=
p7WHZy8TCG.

J. Choo, Y.-D. Kwon, J. Kim, J. Jae, A. Hottung, K. Tierney, and Y. Gwon. Simulation-guided beam search for neural combinatorial optimization. In Advances in Neural Information Processing Systems (NeurIPS), 2022. URL https://arxiv.org/abs/2207.06190.

R. Coulom. Efficient selectivity and backup operators in monte-carlo tree search. In International conference on computers and games, pages 72–83. Springer, 2006.

A. Cully and Y. Demiris. Quality and diversity optimization: A unifying modular framework. IEEE
Transactions on Evolutionary Computation, 22(2):245–259, 2017.

R. de Kock, O. Mahjoub, S. Abramowitz, W. Khlifi, C. R. Tilbury, C. Formanek, A. Smit, and A. Pretorius. Mava: a research library for distributed multi-agent reinforcement learning in jax.

arXiv preprint arXiv:2107.01460, 2021.

C. S. de Witt, T. Gupta, D. Makoviichuk, V. Makoviychuk, P. H. S. Torr, M. Sun, and S. Whiteson.

Is independent learning all you need in the starcraft multi-agent challenge?, 2020. URL https:
//arxiv.org/abs/2011.09533.

DeepMind, I. Babuschkin, K. Baumli, A. Bell, S. Bhupatiraju, J. Bruce, P. Buchlovsky, D. Budden, T. Cai, A. Clark, I. Danihelka, A. Dedieu, C. Fantacci, J. Godwin, C. Jones, R. Hemsley, T. Hennigan, M. Hessel, S. Hou, S. Kapturowski, T. Keck, I. Kemaev, M. King, M. Kunesch, L. Martens, H. Merzic, V. Mikulik, T. Norman, G. Papamakarios, J. Quan, R. Ring, F. Ruiz, A. Sanchez, L. Sartran, R. Schneider, E. Sezener, S. Spencer, S. Srinivasan, M. Stanojevic,´
W. Stokowiec, L. Wang, G. Zhou, and F. Viola. The DeepMind JAX Ecosystem, 2020. URL
http://github.com/google-deepmind.

J. Dona, A. Flajolet, A. Marginean, A. Cully, and T. Pierrot. Quality-diversity for one-shot biological sequence design. In ICML'24 Workshop ML for Life and Material Science: From Theory to Industry Applications, 2024. URL https://openreview.net/forum?id=ZZPwFG5W7o.

Y. Duan, J. Schulman, X. Chen, P. L. Bartlett, I. Sutskever, and P. Abbeel. Rl2: Fast reinforcement learning via slow reinforcement learning, 2016. URL https://arxiv.org/abs/1611.02779.

G. Dulac-Arnold, N. Levine, D. J. Mankowitz, J. Li, C. Paduraru, S. Gowal, and T. Hester. An empirical investigation of the challenges of real-world reinforcement learning. *CoRR*, abs/2003.11881, 2020. URL https://arxiv.org/abs/2003.11881.

B. Ellis, J. Cook, S. Moalla, M. Samvelyan, M. Sun, A. Mahajan, J. Foerster, and S. Whiteson.

Smacv2: An improved benchmark for cooperative multi-agent reinforcement learning. Advances in Neural Information Processing Systems, 36:37567–37593, 2023.

B. Eysenbach, A. Gupta, J. Ibarz, and S. Levine. Diversity is all you need: Learning skills without a reward function. In *International Conference on Learning Representations*, 2019.

C. Finn, P. Abbeel, and S. Levine. Model-agnostic meta-learning for fast adaptation of deep networks.

In *Proceedings of the 34th International Conference on Machine Learning - Volume 70*, ICML'17, page 1126–1135. JMLR.org, 2017.

N. Grinsztajn, D. Furelos-Blanco, S. Surana, C. Bonnet, and T. D. Barrett. Winner takes it all: Training performant rl populations for combinatorial optimization. In Advances in Neural Information Processing Systems, 2023.

R. Hammond, D. Craggs, M. Guo, J. N. Foerster, and I. Reid. Symmetry-breaking augmentations for ad hoc teamwork. In *ICLR 2025 Workshop on Bidirectional Human-AI Alignment*, 2025. URL https://openreview.net/forum?id=pEQwTKmcks.

N. Hansen and A. Ostermeier. Completely derandomized self-adaptation in evolution strategies.

Evolutionary Computation, 9(2):159–195, 2001. doi: 10.1162/106365601750190398.

T. Hayes, R. Rao, H. Akin, N. J. Sofroniew, D. Oktay, Z. Lin, R. Verkuil, V. Q. Tran, J. Deaton, M. Wiggert, R. Badkundri, I. Shafkat, J. Gong, A. Derry, R. S. Molina, N. Thomas, Y. A. Khan, C. Mishra, C. Kim, L. J. Bartie, M. Nemeth, P. D. Hsu, T. Sercu, S. Candido, and A. Rives. Simulating 500 million years of evolution with a language model. *Science*, 387(6736):850–858, 2025. doi: 10.1126/science.ads0018. URL https://www.science.org/doi/abs/10.1126/ science.ads0018.

J. Heek, A. Levskaya, A. Oliver, M. Ritter, B. Rondepierre, A. Steiner, and M. van Zee. Flax: A
neural network library and ecosystem for JAX, 2024. URL http://github.com/google/flax.

A. Hottung, Y.-D. Kwon, and K. Tierney. Efficient active search for combinatorial optimization problems. In *International Conference on Learning Representations*, 2022.

A. Hottung, M. Mahajan, and K. Tierney. Polynet: Learning diverse solution strategies for neural combinatorial optimization. *arXiv preprint arXiv:2402.14048*, Feb 2024.

H. Hu, A. Lerer, A. Peysakhovich, and J. Foerster. "other-play " for zero-shot coordination. In Proceedings of the 37th International Conference on Machine Learning, ICML'20. JMLR.org, 2020.

A. Hundt, B. Killeen, H. Kwon, C. Paxton, and G. Hager. "good robot!": Efficient reinforcement learning for multi-step visual tasks via reward shaping, 09 2019.

L. P. Kaelbling, M. L. Littman, and A. R. Cassandra. Planning and acting in partially observable stochastic domains. *Artificial intelligence*, 101(1-2):99–134, 1998.

R. M. Karp. On the computational complexity of combinatorial problems. *Networks*, 5(1):45–68, 1975.

S. Kumar, A. Kumar, S. Levine, and C. Finn. One solution is not all you need: Few-shot extrapolation via structured maxent rl. *Advances in Neural Information Processing Systems*, 33:8198–8210, 2020.

A. Laterre, Y. Fu, M. K. Jabri, A.-S. Cohen, D. Kas, K. Hajjar, T. S. Dahl, A. Kerkeni, and K. Beguir.

Ranked reward: Enabling self-play reinforcement learning for combinatorial optimization, 2018. URL https://arxiv.org/abs/1807.01672.

W. Long, W. Wen, P. Zhai, and L. Zhang. Role play: Learning adaptive role-specific strategies in multi-agent interactions, 2024. URL https://arxiv.org/abs/2411.01166.

A. Lupu, B. Cui, H. Hu, and J. Foerster. Trajectory diversity for zero-shot coordination. In M. Meila and T. Zhang, editors, *Proceedings of the 38th International Conference on Machine Learning*, volume 139 of *Proceedings of Machine Learning Research*, pages 7204–7213. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/v139/lupu21a.html.

M. Macfarlane, E. Toledo, D. J. Byrne, P. Duckworth, and A. Laterre. SPO: Sequential monte carlo policy optimisation. In *The Thirty-eighth Annual Conference on Neural Information Processing* Systems, 2024. URL https://openreview.net/forum?id=XKvYcPPH5G.

O. Mahjoub, S. Abramowitz, R. de Kock, W. Khlifi, S. du Toit, J. Daniel, L. B. Nessir, L. Beyers, C. Formanek, L. Clark, and A. Pretorius. Sable: a performant, efficient and scalable sequence model for marl. In *Proceedings of the 42th International Conference on Machine Learning*, ICML'25, 2025.

M. S. Mark, T. Gao, G. G. Sampaio, M. K. Srirama, A. Sharma, C. Finn, and A. Kumar.

Policy-agnostic RL: Offline RL and online RL fine-tuning of any class and backbone. In 7th Robot Learning Workshop: Towards Robots with Human-Level Abilities, 2025. URL
https://openreview.net/forum?id=VGOFxdc6AD.

M. Matthews, M. Beukman, B. Ellis, M. Samvelyan, M. Jackson, S. Coward, and J. Foerster. Craftax:
a lightning-fast benchmark for open-ended reinforcement learning. In *Proceedings of the 41st* International Conference on Machine Learning, ICML'24. JMLR.org, 2024.

R. Mirsky, I. Carlucho, A. Rahman, E. Fosong, W. Macke, M. Sridharan, P. Stone, and S. V. Albrecht.

A survey of ad hoc teamwork research. In Multi-Agent Systems: 19th European Conference, EUMAS 2022, Düsseldorf, Germany, September 14–16, 2022, Proceedings, page 275–293, Berlin, Heidelberg, 2022. Springer-Verlag. ISBN 978-3-031-20613-9. doi: 10.1007/978-3-031-20614-6_
16. URL https://doi.org/10.1007/978-3-031-20614-6_16.

V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. A.

Riedmiller, A. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis. Human-level control through deep reinforcement learning. *Nature*, 518:529–533, 2015.

N. Muennighoff, Z. Yang, W. Shi, X. L. Li, L. Fei-Fei, H. Hajishirzi, L. Zettlemoyer, P. Liang, E. Candès, and T. Hashimoto. s1: Simple test-time scaling, 2025. URL https://arxiv.org/ abs/2501.19393.

M. Nakamoto, Y. Zhai, A. Singh, M. S. Mark, Y. Ma, C. Finn, A. Kumar, and S. Levine. Cal-QL:
Calibrated offline RL pre-training for efficient online fine-tuning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id= GcEIvidYSw.

M. Olivecrona, T. Blaschke, O. Engkvist, and H. Chen. Molecular de novo design through deep reinforcement learning. *Journal of Cheminformatics*, 9, 09 2017. doi: 10.1186/s13321-017-0235-x.

L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, and R. Lowe. Training language models to follow instructions with human feedback. In *Proceedings of the 36th International Conference on Neural Information Processing* Systems, NIPS '22, Red Hook, NY, USA, 2022. Curran Associates Inc. ISBN 9781713871088.

G. Papoudakis, F. Christianos, L. Schäfer, and S. V. Albrecht. Benchmarking multi-agent deep reinforcement learning algorithms in cooperative tasks. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS), 2021. URL http://arxiv. org/abs/2006.07869.

M. Popova, O. Isayev, and A. Tropsha. Deep reinforcement learning for de novo drug design. Science Advances, 4(7):eaap7885, 2018. doi: 10.1126/sciadv.aap7885. URL https://www.science.

org/doi/abs/10.1126/sciadv.aap7885.

R. Qiu, Z. Sun, and Y. Yang. DIMES: A differentiable meta solver for combinatorial optimization problems. In A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, editors, Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=9u05zr0nhx.

K. Rao, C. Harris, A. Irpan, S. Levine, J. Ibarz, and M. Khansari. Rl-cyclegan: Reinforcement learning aware simulation-to-real. pages 11154–11163, 06 2020. doi: 10.1109/CVPR42600.2020.01117.

C. Ruhdorfer, M. Bortoletto, A. Penzkofer, and A. Bulling. The overcooked generalisation challenge, 2025. URL https://arxiv.org/abs/2406.17949.

A. Rutherford, B. Ellis, M. Gallici, J. Cook, A. Lupu, G. Ingvarsson, T. Willi, A. Khan, C. S. de Witt, A. Souly, et al. Jaxmarl: Multi-agent rl environments in jax. *arXiv preprint arXiv:2311.10090*,
2023.

M. Samvelyan, T. Rashid, C. S. De Witt, G. Farquhar, N. Nardelli, T. G. Rudner, C.-M. Hung, P. H. Torr, J. Foerster, and S. Whiteson. The starcraft multi-agent challenge. *arXiv preprint* arXiv:1902.04043, 2019.

A. Sharma, S. Gu, S. Levine, V. Kumar, and K. Hausman. Dynamics-aware unsupervised discovery of skills. *arXiv preprint arXiv:1907.01657*, 2019.

D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. Van Den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, et al. Mastering the game of go with deep neural networks and tree search. *Nature*, 529(7587):484–489, 2016.

C. V. Snell, J. Lee, K. Xu, and A. Kumar. Scaling LLM test-time compute optimally can be more effective than scaling parameters for reasoning. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=4FWAwZtd2n.

N. Stiennon, L. Ouyang, J. Wu, D. M. Ziegler, R. Lowe, C. Voss, A. Radford, D. Amodei, and P. Christiano. Learning to summarize from human feedback. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS '20, Red Hook, NY, USA,
2020. Curran Associates Inc. ISBN 9781713829546.

C. Wang, A. Rahman, I. Durugkar, E. Liebman, and P. Stone. N-agent ad hoc teamwork. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024a. URL
https://openreview.net/forum?id=q7TxGUWlhD.

R. Wang, E. Zelikman, G. Poesia, Y. Pu, N. Haber, and N. Goodman. Hypothesis search: Inductive reasoning with language models. In The Twelfth International Conference on Learning Representations, 2024b. URL https://openreview.net/forum?id=G7UtIGQmjm.

J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. H. Chi, Q. V. Le, and D. Zhou.

Chain-of-thought prompting elicits reasoning in large language models. In *Proceedings of the 36th* International Conference on Neural Information Processing Systems, NIPS '22, Red Hook, NY,
USA, 2022. Curran Associates Inc. ISBN 9781713871088.

Y. Wu, Z. Sun, S. Li, S. Welleck, and Y. Yang. Inference scaling laws: An empirical analysis of compute-optimal inference for problem-solving with language models, 2025. URL https:
//arxiv.org/abs/2408.00724.

E. S. Yourdshahi, T. Pinder, G. Dhawan, L. S. Marcolino, and P. Angelov. Towards large scale ad-hoc teamwork. In *2018 IEEE International Conference on Agents (ICA)*, pages 44–49, 2018. doi:
10.1109/AGENTS.2018.8460136.

C. Yu, A. Velu, E. Vinitsky, J. Gao, Y. Wang, A. Bayen, and Y. Wu. The surprising effectiveness of PPO in cooperative multi-agent games. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022. URL https://openreview.net/
forum?id=YVXaxB6L2Pl.

L. Zintgraf, S. Schulze, C. Lu, L. Feng, M. Igl, K. Shiarlis, Y. Gal, K. Hofmann, and S. Whiteson.

Varibad: variational bayes-adaptive deep rl via meta-learning. *J. Mach. Learn. Res.*, 22(1), Jan. 2021. ISSN 1532-4435.

## Neurips Paper Checklist

1. **Claims**
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: / Guidelines:
- The answer NA means that the abstract and introduction do not include the claims made in the paper.

- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. **Limitations**

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: Guidelines:
- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.

- The authors are encouraged to create a separate "Limitations" section in their paper. - The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

- The authors should reflect on the factors that influence the performance of the approach.

For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. **Theory Assumptions And Proofs**

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [NA] Justification: No theoretical results. Guidelines:
- The answer NA means that the paper does not include theoretical results. - All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

- All assumptions should be clearly stated or referenced in the statement of any theorems. - The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. **Experimental Result Reproducibility**

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]
Justification: Code and model checkpoints are provided. Guidelines:
- The answer NA means that the paper does not include experiments.

- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

- Depending on the contribution, reproducibility can be accomplished in various ways.

For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. **Open Access To Data And Code**

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: Guidelines:
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

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes] Justification: Yes, values are given in the appendix. Guidelines:
- The answer NA means that the paper does not include experiments. - The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. **Experiment Statistical Significance**

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: Guidelines:
- The answer NA means that the paper does not include experiments.

- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).

- It should be clear whether the error bar is the standard deviation or the standard error of the mean.

- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).

- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. **Experiments Compute Resources**

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: Guidelines:
- The answer NA means that the paper does not include experiments. - The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. **Code Of Ethics**

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes] Justification: Guidelines:
- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. - If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. **Broader Impacts**

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]
Justification: Not relevant. Guidelines:
- The answer NA means that there is no societal impact of the work performed. - If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

- Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. **Safeguards**

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)? Answer: [NA] Justification: Guidelines:
- The answer NA means that the paper poses no such risks. - Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. **Licenses For Existing Assets**

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]
Justification: Guidelines:
- The answer NA means that the paper does not use existing assets. - The authors should cite the original paper that produced the code package or dataset. - The authors should state which version of the asset is used and, if possible, include a URL.

- The name of the license (e.g., CC-BY 4.0) should be included for each asset. - For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. **New Assets**

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?