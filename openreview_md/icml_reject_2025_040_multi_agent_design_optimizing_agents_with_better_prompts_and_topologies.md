# Multi-Agent Design: Optimizing Agents With Better Prompts And Topologies

Anonymous Authors1

## Abstract

000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054

## 1. Introduction

Large language models, employed as multiple agents that interact and collaborate with each other, have excelled at solving complex tasks. The agents are programmed with *prompts* that declare their functionality, along with the *topologies* that orchestrate interactions across agents. Designing prompts and topologies for multi-agent systems (MAS) is inherently complex. To automate the entire design process, we first conduct an in-depth analysis of the design space aiming to understand the factors behind building effective MAS. We reveal that prompts together with topologies play critical roles in enabling more effective MAS design. Based on the insights, we propose Multi-Agent System Search (MASS), a MAS optimization framework that efficiently exploits the complex MAS design space by interleaving its optimization stages, from local to global, from prompts to topologies, over three stages: 1) block-level (*local*) prompt optimization; 2) workflow topology optimization; 3) workflow-level (*global*) prompt optimization, where each stage is conditioned on the iteratively optimized prompts/topologies from former stages. We show that MASS-optimized multi-agent systems outperform a spectrum of existing alternatives by a substantial margin. Based on the MASS-found systems, we finally propose design principles behind building effective multi-agent systems.

Large language models (LLMs) have showcased extraordinary capabilities in understanding, reasoning, and generating coherent responses based on user prompts, revolutionizing a wide range of applications (Ouyang et al., 2022; Kojima et al., 2022). LLM-based agents enhance usability 1Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>.

1

MASS 
Optimized MAS design Prompt Optimization Space Optimized topology Instruction Exemplar Multi-Agent Design Space aggregate summarize

<ex_1>
<ex_1>
<ex_1>
<ex_2>
<ex_2>
<ex_2>
re ect tool-use

...

...

...

Topology Optimizer debate custom

<ins>
<ins>
<ins>
Optimized prompt for each agent type
by autonomously handling complex tasks across diverse domains, including code generation and debugging (Jimenez et al., 2023), retrieval-augmented generation (Singh et al., 2025; Wang et al., 2024a), data analysis (Hu et al., 2024b; Guo et al., 2024), and interactive decision-making (Su et al., 2025; Li et al., 2025). These agents are typically programmed with prompts that reinforce them to interact with the environment, utilizing available tools, and approach their objectives over multiple turns (Yao et al., 2023). Beyond individual agents, LLMs can be orchestrated within complex topologies that coordinate multiple agents toward a shared objective. This type of multi-agent system (MAS) typically outperforms its single-agent counterpart by involving more diverse agentic perspectives or role profiles, such as agents as verifiers (Shinn et al., 2024) and multi-agent debate (Wang et al., 2024b; Qian et al., 2024). However, designing effective MAS for new domains often proves to be challenging. First, the single agent might suffer from prompt sensitivity (Verma et al., 2024), where simple modifications in the prompt can already exert significant but unexpected degradation of performance (Zhou et al., 2024b; Liu et al., 2024a). In MAS, when sensitive agents are cascaded, the compounding effect due to prompt sensitivity may be amplified. Together with the prompt design, crafting an effective topology might demand a substantial amount of manual experimentation, based on trial and error. The problem complexity is exacerbated by the overall combinatorial search space, over not only the unbounded space of prompt design but also the design decisions of what agent to integrate into the topology. Although recent research has explored automating various aspects of agentic designs, there is still a gap in understanding of what matters most regarding improved MAS performance. For example, DSPy (Khattab et al., 2024) automates the process of designing exemplars for improved prompt programming. Li et al. (2024a) proposes to optimize MAS by scaling up the number of agents in majority voting. ADAS (Hu et al., 2024a) programs new topologies expressed in code via an LLM-based meta-agent. AFlow (Zhang et al., 2024b) searches better topologies using Monte Carlo Tree Search within a set of predefined operators. However, the interplay between multiple design spaces, including prompts and topologies, remains unclear. In this paper, we first conduct in-depth analyses of common design spaces in MAS, examining the influence of various aspects such as optimizing the prompts, scaling the number of agents, and involving different types of topologies. Our analyses reveal that prompts frequently form an influential design component that yields strong-performing MAS, and influential topologies only represent a small fraction of the full search space. Based on these insights, we aim to distill the essence of influential MAS components into a pruned search space, thereby lowering the complexity of the overall search process. We propose Multi-Agent System Search (MASS), a novel multi-stage optimization framework that automates the optimization for MAS over an efficient search space. MASS integrates a plug-and-play prompt optimizer and workflow optimizer over a configurable topology space. It overcomes the complexity of joint optimization on MAS by interleaving the optimization stages, from local to global, from prompts to topologies, over three stages: 1) blocklevel (*local*) prompt 'warm-up' for each topology block; 2) workflow topology optimization in a *pruned* set of topology space; 3) workflow-level (*global*) prompt optimization given the best-found topology. By optimizing over the identified influential components, MASS yields optimized MAS that achieves state-of-theart performance, outperforming existing manually-crafted MAS baselines and automatically-generated alternatives, by a substantial margin, demonstrated across an extensive selection of tasks, including reasoning, multi-hop understanding, and code generation. Based on the strongest MAS found by MASS, we provide further insights and guidelines behind building effective MAS. Overall, our contributions can be summarized as follows: 1) we provide an in-depth analysis of the design factors that influence the performance of LLM-based MAS, highlighting the importance of prompts and identifying the influential topologies; 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109 2) we propose MASS, a novel multi-stage optimizer that automates the MAS design by interleaving the optimization of prompts and topologies in an influential search space; 3) MASS shows significant performance improvement on various evaluation benchmarks, delivering guidelines for building effective multi-agent systems for the future.

## 2. Designing Multi-Agent Systems

In this section, we provide a formulation for designing MAS, followed by analyzing the influence of prompt and topology designs. We refer to the structural arrangements of agents (or equivalently, building blocks) as the topology of agents and define workflow W as the logical sequence across different topologies that builds the MAS. The design of a MAS can thus be broadly divided into two levels: block-level design and workflow-level orchestration. At the block level, we aim to design effective individual agents that best perform their intended role with better *prompt* design. On the other hand, at the workflow level, the optimization involves determining the *types* and *quantities* of agents to include and how to arrange them in the most effective way, referred to as the topology optimization. Formally, given a search space A that defines all valid configurations a over the blocks (see Fig. 4), *workflow topology optimization* can be expressed as the following optimization problem with an objective function f(·, ·) on a target input and output set (x, y) ∼ D:

$${\mathcal{W}}^{*}(a)=\arg\max_{a\sim{\mathcal{A}}}\mathbb{E}_{(x,y)\sim{\mathcal{D}}}[f({\mathcal{W}}(a(x)),y)].\tag{1}$$

In the rest of this section, we provide an in-depth analysis of each component of MAS design.

## 2.1. Block-Level Analysis: Prompt Design For Agents

At the block level, the primary "optimizable component" that significantly influences downstream performance is the prompt, which defines the role of the agent (e.g., "You are an expert in reflecting on errors..."), provides additional instructions to shape its behavior (e.g., "You should think step by step...") and optionally, contains *few-shot demonstrations* (in-context examples) to guide the agent's responses (Wan et al., 2024). For instance, a state-of-the-art prompt optimizer searches both instructions and few-shot demonstrations, where demonstrations are bootstrapped from the model's own, correct predictions on the validation set based on a validation metric. Conditioned on the demonstrations, the prompt optimizer then proposes a few candidates for the instruction with a dataset summary or various hints to improve candidate diversity (Opsahl-Ong et al., 2024). The instructions and demonstrations are then jointly optimized.

Although it is well known that LLMs are sensitive to prompts (Zhou et al., 2024a; Verma et al., 2024), applying automatic prompt optimization (APO) techniques to

74 76 78 80 82 84 SC Reflect Debate Prompting Prompt->SC
Acc ura c y (
%
)

10 3 Tokens
110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 MAS is rather non-trivial. Unlike single-turn tasks where APO can be easily performed by treating prompts as optimizable variables and performance over a validation set as the target. In MAS, APO becomes more complex due to the interdependence across agents (e.g., the output of one agent may be the input of another agent in a cascade with groundtruth responses for intermediate outputs not being available) and exponentially increasing complexity for combinatorial optimization with more number of agents n involved; The reward signals also become more sparse when n increases, preventing us for implementing APO directly on MAS in any manageable budget; as such, many prior works (Zhang et al., 2024f; Xia et al., 2024) in MAS still primarily use handcrafted prompts instead of including the prompts as optimizable components in the MAS design. To systematically understand the influence of prompt design in MAS, we specifically and quantitatively analyze the effect of prompt optimization and compare its effectiveness to other operations common in MAS literature, such as scaling with more agents but with default prompts. We conduct APO on a chain-of-thought (Kojima et al., 2022) agent with both instruction optimization and 1-shot exemplar optimization via MIPRO (Opsahl-Ong et al., 2024), and fairly compare the total inference token cost with selfconsistency (Kojima et al., 2022), self-refine (Madaan et al., 2024), and multi-agent debate (Du et al., 2024), where the specifications are provided in App. §B. In Fig. 2, prompting, which equips agents with more informative instructions and exemplars, demonstrates significant advantages in its tokeneffectiveness over other building blocks. Furthermore, by applying self-consistency on top of the prompt-optimized agent, we observe an improved scaling performance on the token cost, whereas standard approaches in scaling the num-

HotpotQA
-15%
+6%
+10%
LiveCodeBench CoT SC Ref. Deb. Sum.

Method 62 64 66 68 70 Base
+1%
-0%
+3%
CoT SC Ref. Deb. Exe.

Method 60 70 80 Base
+7%
Pe rfo rm a nc e 
(
%
)

-2%
ber of agents (e.g. SC, or Reflect) saturate much earlier. This empirical observation sheds light on the importance of prompting while providing early evidence for designing effective MAS - optimize agents locally before scaling their topology.

## 2.2. Workflow-Level Search Space Design

At the workflow level, the primary focus is on orchestrating agents to achieve the best performance effectively. As a relatively new concept specific to MAS, topology optimization has recently garnered significant attention (Li et al., 2024c; Zhang et al., 2024b). However, while much of the existing research emphasizes *search methods*—such as discovering the most efficient and effective way to identify the optimal configuration—there has been less focus on the design of *search spaces*, which determines the perimeter and the scope of any search algorithm. This imbalance draws a parallel to the historical development of neural architecture search (NAS) (White et al., 2023). Initially, the field concentrated on sophisticated search methods, such as Bayesian optimization (Kandasamy et al., 2018; Ru et al., 2021) and differentiable search (Liu et al., 2018). Follow-up works have highlighted the often-overlooked importance of search space design, arguing that it can be equally, if not more, critical (Wan et al., 2022; Zhou et al., 2023). Inspired by this insight, we hypothesize that manually crafted topologies might be sub-optimal, and automatic topology optimization (potentially framed as a rigorous optimization problem) can play a similarly pivotal role via judiciously designing search space for MAS. To achieve so, we first define an expressive search space, similar to prior works, that consists of the connections between the following *building blocks*: - *Aggregate*: Agents can collaborate in parallel with diversified predictions, which is then followed by an aggregation operator that obtains the most consistent prediction. The aggregate block can be parameterized by Na agents

ect 1 Block-level Prompt Optimization 2 Work 3 Work ow Topology Optimization ow-level Prompt Optimization Predictor Aggregate P
 R 
</> </>
Self-re Proposed candidate

</> </> </> </> </>
</> </> </> </> </>
Best topology from Step 2

</>
P 
</>
</> </> </> </> </>
P 
P 
A

S P 
Summarize

</> </>
</>
Evaluate on validation task / split Evaluate on validation task / split Validation metric Long inputs

</> </>
</> </>
Multi-agent debate ow Validation metric

</>: Optimizable prompts Tool-use D

A

</>
P 
P 
Optimizer: Store evaluations and propose new work
</>
Instruction optimization Let's think step by step → (Example new prompt) <example_1> <example_2> ...

P 
</>
D

(
 ,
 75%) 
,
 63%)
( 
T 
Demo optimization P 
D

.

 .

 . 

</>: Optimizable prompts

## Optimizable Binary 'Insertion' Decision Nt ∈ {0, 1}.

acting in parallel. Majority vote (Li et al., 2024a) and selfconsistency (Chen et al., 2024c) sits within this topology.

165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219
- *Reflect*: Agents can act as verifiers, providing critics and improvement suggestions based on former predictions. The feedback is then fed into the predictor or the reflector itself for an iterative improvement. Similarly, reflect can be parameterized by Nr that defines the number of rounds for self-reflection. The self-refine (Madaan et al., 2024) and Reflexion (Shinn et al., 2024) represent this block. - *Debate*: Agents in debate can elicit more truthful predictions than single-agent prediction (Du et al., 2024; Liang et al., 2024), where each debating agent would collect opinions from all other agents and provides an updated response.

This topology would involve a mixture of agents, and Nd defines the number of rounds for debating.

- *Custom Agents*: While the former three forms of agents represent the vast majority of agent topologies constructed as multiple parallel, serial, and mixture of agents, more versatile definitions of agents can be inserted into the MAS design space. For example, for task-specific use cases, we introduce an agent as summarize to improve the longcontext capability in the customizable design space.

fl fl
- *Tool-use*: Building towards an effective MAS, enabling agents to leverage tools to access external information is critical for system performance, such as using retriever for RAG (Lewis et al., 2020) and executor with test cases in coding (Chen et al., 2024d). We introduce tool-use as an To understand the influence of individual topology, we report the performance of various topologies in Fig. 3. It is noticeable that not all topologies are beneficial to MAS design, whereas positively influenced topologies only represent a small fraction of the overall set, such that, in HotpotQA
(Yang et al., 2018), only debate brings 3% gain while others fail to improve or even degrade systematic performance. We again observe similar trends in the test-output-prediction subtask of LiveCodeBench (Jain et al., 2024). It highlights the importance of searching in the influential set of search space, whereas including decremental building blocks may not only result in higher search complexity but also degrade the performance.

## 3. Mass**: Multi-Agent System Search**

Our analyses in Sec. 2 underscore the importance of welldesigned prompts for individual agents and the careful definition of the search space to achieve effective MAS performance. Building on these, we propose a multistage optimization algorithm, **Multi-Agent System Search** (MASS), that surpasses prior arts that focused solely on optimizing workflow topology without appropriate prompt designs. Instead, our approach demonstrates the greater effectiveness of MAS design with properly optimized prompts and thoughtfully designed search spaces. MASS framework is illustrated in Algorithm 1 and Fig. 4, following an intuition from local to fl 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 global, from block-level to workflow-level, that conquers the complexity of combinatorial optimization with effective per-stage optimization detailed below.

1) Block-level prompt optimization. Before composing agents, we first ensure that individual agents are thoroughly optimized at the block level, as highlighted in Sec. 2.1 and Fig. 2 - this step ensures that each agent is primed for its role with the most effective instructions in the most manageable computation budget. To further overcome the complexity of joint optimization on a large MAS space, we first warm up the initial predictor with single-agent APO, a
∗0 ← OD(a0), where both instruction and exemplars are jointly optimized with the modular prompt optimizer O. Followed by conditioning on the warmed predictor, we continue optimizing each topology with a minimum number of agents, a
∗
i ← OD(ai|a
∗0), such that, 2 predictors paired with 1 debator form the minimum building block as the debate topology, thereby lowering the complexity for optimization, and this topology can be scaled up later with more predictors and debators but all equipped with optimized prompts. To measure the influence of each building block, we store the validation performance once the optimization is completed. It is important that though Stage (1) serves as the warmup stage per building block, it is still a critical stage that guarantees the follow-up topology optimization is searching in an effective space, composing well-performing agents instead of suffering from the compounding impact from any ill-formed agents with manual prompts. 2) Workflow topology optimization. In this stage, we focus on optimizing the overall MAS structure, determining the most effective arrangement and connectivity between agents. The analysis in Fig. 3 shows that beneficial topologies only represent a small fraction of the full design space. Therefore, we aim to distill the essence of strong-performing topologies into a pruned space, thereby making the workflow-level topology search more efficient. Here, we propose to measure the incremental influence Iai = E(a
∗ i
)/E(a
∗0
) that quantifies the relative gain for integrating the topology ai over the initial agent a0. Following the intuition that influential dimension comes with higher selection probability, we activate the corresponding topology dimension a if *u > p*a, given u ∼ U(0, 1) and pa = Softmax(Ia, t). To compose diverse topologies into a unified space, we constrain the workflow with a rule-based order to reduce the optimization complexity, following a predefined sequence, such that [summarize, reflect, debate, aggregate]. We integrate rejection sampling over the pre-defined design space that rejects any deactivated dimension, or invalid topology compositions exceeding a maximum budget B on the number of agents. We refer to App. §B for the detailed search space per task. 3) Workflow-level prompt optimization. As a final step, Algorithm 1 MASS: Multi-Agent System Search 1: **Input**: Agentic modules in the search space ai ∈ A, workflow of agents W(a), prompt optimizer O, evaluator E, validation set D, temperature t, number of candidates N, budget B.

2: **Output**: Optimized multi-agent system W∗.

3: [*Block*-level **Prompt** Optimization] 4: Prompt optimization for the initial agent a
∗0 ← OD(a0).

5: for ai in *A \ {*a0} do 6: Local prompt optimization for each building block in the design space: a
∗
i ← OD(ai|a
∗0)
7: Obtain incremental Influence Iai ← E(a
∗
i )/E(a
∗
0).

8: **end for** 9: [Workflow **Topology** Optimization]
10: Obtain the selection probability pa ← Softmax(Ia, t) 11: **while** *n < N* do 12: Reject invalid configurations c and cap a budget B. The design space is pruned by the selection probability pa, Wc ← (a
∗
i (·), a∗
i+1(·)*, . . .*) with optimized prompts.

13: Store evaluations ED(Wc) and propose new workflows. 14: **end while**
15: Obtain the best-performing W∗
c ← arg maxc∈C ED(Wc).

16: [*Workflow*-level **Prompt** Optimization] 17: Workflow-level prompt optimization for the best-performing topology: W∗ ← OD(W∗
c ).

18: **Return** optimized multi-agent system W∗.

we treat the entire MAS design as an integrated entity and run an additional round of prompt optimization, conditioned on the best topology discovered in Stage (2),
W∗ = OD(W∗
c
). It is worth noting that although prompts were optimized at the individual level in Stage (1), this stage acts as an adaptation or fine-tuning process, ensuring that prompts are tailored for orchestration within the MAS and that the interdependence between agents is optimized appropriately. Our experiments (Fig. 5 & 6) demonstrate that this stage often yields practical benefits.

## 4. Related Work

Forms of LLM-based agentic systems. The simplest form of an LLM-based agentic system involves a single agent that can dynamically interact and respond to the environment (Yao et al., 2023). Recent advances endow agents with diverse roles and tools (Wu et al., 2023), orchestrating multiple agents to cooperate with each other (Chen et al., 2024b). Standard forms of agent cooperation (i.e., topology) often involve parallel and serial flows of information. The parallel form usually diversifies the exploration among many agents in parallel (Li et al., 2024a), and self-consistency (SC) (Wang et al., 2023) is a representative way for scaling agents in parallel. The serial form aims to advance the exploitation of a task via a chain of agents, where LLMs can serve as reflective agents to self-justify and refine former predictions (Madaan et al., 2024; Shinn et al., 2024).

Later, the opinions from multiple agents can be summarized to retrieve the most consistent answer by an aggregation agent (Chen et al., 2024c; Lin et al., 2024). Moreover, 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 multi-agent debate consists of a more complex flow of information (Chen et al., 2024a; Wang et al., 2024c; Zhang et al., 2024c), and recent research shows that debating can elicit more truthful predictions (Khan et al., 2024; Du et al., 2024). Recent agent topology extends beyond the above connections (Wang et al., 2024b; Qian et al., 2024), and MASS can automatically search the best topology among the aforementioned spaces. Automatic optimization for MAS. Recent research starts automating agent design by interpreting agent functions as learnable policies (Zhang et al., 2024d;e) and synthesizing trajectories for agent fine-tuning (Qiao et al., 2024). Going further from a single agent, automatic multi-agent optimization faces a higher level of complexity, thereby requiring a more sophisticated design of search space and algorithms. Among all recent advances in multi-agent optimization, the optimization space has spanned prompts (Khattab et al., 2024), tools (Zhou et al., 2024c), workflows (Li et al., 2024c), and thinking strategies (Shang et al., 2024). Aligning closer to our topology search space, DyLAN (Liu et al., 2024b) dynamically activates the composition of agents, and Archon (Saad-Falcon et al., 2024) frames MAS as a hyperparameter optimization problem. Neither of them has taken the important prompt space into account, where we demonstrated the importance of prompt optimization in Sec. 2.1. In addition, GPTSwarm (Zhuge et al., 2024) optimizes the connections between agentic nodes using a policy gradient algorithm. State-of-the-art automatic agent design methods, ADAS (Hu et al., 2024a) and AFlow (Zhang et al., 2024b),
also attempt to optimize agentic workflows with advanced search algorithms and LLM as optimizers. However, we observe that the importance of proper prompt designs has been relatively under-studied in these prior works.

## 5. Experiments

Models and evaluation data. Aside from the common benchmarks used for automating MAS (Hu et al., 2024a; Zhang et al., 2024b), we conduct experiments on an extensive collection of tasks: 1) Hendryck's MATH (Hendrycks et al., 2021) and DROP (Dua et al., 2019) for reasoning; HotpotQA (Yang et al., 2018), MuSiQue (Trivedi et al., 2022), 2WikiMultiHopQA (Ho et al., 2020) from Long- Bench (Bai et al., 2024) for long-context understanding; 3) MBPP (Austin et al., 2021), HumanEval (Chen et al., 2021), and LiveCodeBench (LCB) 'test output prediction' (Jain et al., 2024) for coding. We refer to App. §B & §D for details on data splits and prompt templates. We run all experiments primarily on two Gemini 1.5 model sizes (Reid et al., 2024) (gemini-1.5-{pro,flash}-002) and further validate key findings on Claude 3.5 Sonnet (claude-3-5-sonnet@20240620) (Anthropic, 2024).

Baselines. We consider the following baselines: 1) CoT (Kojima et al., 2022): direct chain-of-thought reasoning via zero-shot prompting; 2) CoT-SC (Wang et al., 2023): with self-consistency to find the most consistent answers from diversified reasoning traces; 3) Self-Refine (Madaan et al., 2024; Shinn et al., 2024): reflective agents to verify and self-refine predictions; 4) Multi-Agent Debate (Du et al., 2024; Liang et al., 2024): with agent justifying answers and aggregating information from other agents; 5) ADAS (Hu et al., 2024a): an automatic agent design framework, where an LLM-based meta-agent iteratively proposes new agents based on former evaluations; 6) AFlow (Zhang et al., 2024b): automatic workflow design via Monte-Carto Tree Search over a set of pre-defined operators. We fairly compare all baselines by limiting the maximum number of agents to 10. We refer to App. §B for all specifications.

Setup. MASS integrates the state-of-the-art prompt optimizer, MIPRO (Opsahl-Ong et al., 2024), which optimizes both instructions and demonstrations for each agent via a Bayesian surrogate model. We limit the number of bootstrapped demonstrations to 3 and instruction candidates to 10, per agent in 10 rounds. In topology optimization for all tasks, we search for 10 different topologies via rejection sampling. Along with topology optimization, each topology is evaluated on the validation set 3 times to stabilize the prediction. The optimized MAS is then reported on the heldout test set over three runs. We set model temperature T at 0.7, maximum output tokens at 4096, and the t in Softmax at 0.05 for sharpening the selection probability pa for each search dimension. We implement the same LLM backbone as both evaluator and optimizer in all phases. Main results. We present the main results of MASS compared to the baselines on the evaluation set in Table 1.

MASS yields substantial gains over common forms of multiagent systems, (e.g. self-consistency, self-refine, and multiagent debate), that scale up without optimizing prompts for agents in collaboration. MASS leads to high-performing MAS: 78.8% and 74.3% on average on Gemini 1.5 Pro and Flash, respectively, where we observe consistent improvements on Claude 3.5 Sonnet as reported in Table 4. By comparing MASS with state-of-the-art automatic agent design baselines, ADAS and AFlow, we first notice that ADAS only brings subtle gains even by already conditioning its metaagent generation based on the common forms of agents.

The meta-agent keeps proposing complex topologies but without optimizing the prompt design. AFlow, on the other hand, demonstrates a competitive performance to MASS, especially on 2WikiMQA and HumanEval. We attribute the performance of AFlow to: 1) its 'expansion' phase that generates new nodes based on an error log that contrasts the predictions with the ground truth, which provides implicit textual gradients (Pryzant et al., 2023) to reflect on any formatting errors in prompt design; 2) a more refined search Table 1. Results on the evaluation set with Gemini 1.5 Pro and Gemini 1.5 Flash. We report the mean and standard deviation for all results with 3 runs of evaluations. We report the accuracy (%) for MATH and the test-output-prediction subtask of LiveCodeBench (LCB), F1 score for DROP, HotpotQA, MuSiQue, and 2WikiMQA, and pass@1 for MBPP and HumanEval. We note that the meta-prompt of AFlow*only works properly with Claude 3.5 Sonnet. Therefore, we reproduce AFlow with Gemini 1.5 Pro as the executor and Claude 3.5 Sonnet as the optimizer, where *indicates the results are only for reference. Number of agents in inference for all methods are below 10.

330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 space within a pre-defined set of operators. Though AFlow draws similar inspirations on the importance of search space design as MASS, it still lacks a phase of prompt optimization to *optimize* its pre-defined operators properly, resulting in under-performance for MAS search results at MATH and MuSiQue. Different from these baselines, the consistent improvements brought by MASS highlight the importance

| Gemini-1.5-pro-002   |           |                        |           |           |           |           |           |           |       |
|----------------------|-----------|------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-------|
| Task                 | Reasoning | Multi-hop Long-context | Coding    |           |           |           |           |           |       |
| Method               | MATH      | DROP                   | HotpotQA  | MuSiQue   | 2WikiMQA  | MBPP      | HumanEval | LCB       | Avg.  |
| CoT                  | 71.673.30 | 70.591.67              | 57.430.52 | 37.811.43 | 63.391.12 | 68.330.47 | 86.670.94 | 66.330.62 | 65.28 |
| Self-Consistency     | 77.331.25 | 74.060.90              | 58.602.19 | 41.811.00 | 67.791.19 | 69.500.71 | 86.000.82 | 70.330.94 | 68.18 |
| Self-Refine          | 79.672.36 | 71.031.31              | 60.623.33 | 42.151.34 | 66.742.43 | 63.670.24 | 84.001.63 | 67.331.31 | 66.90 |
| Multi-Agent Debate   | 78.670.94 | 71.780.71              | 64.870.23 | 46.000.80 | 71.780.63 | 68.670.85 | 86.671.25 | 73.671.65 | 70.26 |
| ADAS                 | 80.000.82 | 72.960.90              | 65.881.29 | 41.951.24 | 71.140.66 | 73.001.08 | 87.671.70 | 65.171.25 | 69.72 |
| AFlow*               | 76.000.82 | 88.920.63              | 68.620.47 | 32.051.29 | 76.511.05 | -         | 88.000.00 | -         | -     |
| MASS (Ours)          | 84.670.47 | 90.520.64              | 69.911.11 | 51.400.42 | 73.340.67 | 86.500.41 | 91.670.47 | 82.330.85 | 78.79 |
| Gemini-1.5-flash-002 |           |                        |           |           |           |           |           |           |       |
| CoT                  | 66.672.36 | 71.790.69              | 57.821.10 | 37.101.35 | 63.400.68 | 63.331.25 | 75.671.89 | 51.170.24 | 60.87 |
| Self-Consistency     | 69.331.25 | 73.420.19              | 60.191.01 | 41.940.93 | 67.980.72 | 63.670.62 | 77.671.89 | 53.831.18 | 63.50 |
| Self-Refine          | 71.330.94 | 73.711.09              | 58.843.04 | 41.211.99 | 65.561.57 | 63.331.25 | 81.671.89 | 52.001.41 | 63.46 |
| Multi-Agent Debate   | 71.670.94 | 74.790.87              | 64.171.69 | 46.271.33 | 72.190.54 | 63.000.71 | 79.671.25 | 55.500.41 | 65.91 |
| ADAS                 | 68.001.41 | 75.951.18              | 61.362.89 | 48.811.03 | 66.901.00 | 65.830.24 | 80.672.49 | 50.501.63 | 64.75 |
| MASS (Ours)          | 81.002.45 | 91.680.14              | 66.530.38 | 43.671.21 | 76.690.50 | 78.000.82 | 84.670.47 | 72.170.85 | 74.30 |

CoT APO 1PO 2TO 3PO
Stage 55 60 65 70 75 80 63.5 68.0 74.2 77.3 78.8 w/o PO w/o Prune PO TO
Ablation 62 64 66 68 P
erfo rm a nc e (
%
)

0 10 20 30 40 50 60 70 80 Round 65 70 75 80 85 90 95 F
1 
(
%
)
Better Prompt 1PO: Aggregate 1PO: Debate 2TO: More parallel agents involved ADAS 3PO: Workflow-level PO
AFlow MASS (Ours) 

## Of Searching In Both Prompt And Topology Design Space.

Ablating optimization stages. To understand the incremental gain per MASS optimization stage, we provide a stage-by-stage ablation study in Fig. 5. We list the aver-

1 Block-level Prompt Optimization ( 62% → 79% )

Debator: You are a seasoned math professor specializing in clear and concise explanations. You are reviewing student solutions to math problems. Below, you will find the problem, followed by solutions from several students. Carefully examine each student's solution, identifying any errors in their logic or calculations. Provide a comprehensive rationale explaining your analysis of each student's work, clearly stating whether their final answer is correct or incorrect and why.

Finally, provide your own definitive and simplified solution to the problem, ensuring its accuracy and clarity. Present your final answer bracketed between <answer> and </answer> at the end. Question: Compute $17^{-1}\\pmod{83}$.

Solutions: Agent 0: 44\nAgent 1: 74 Rationale: <Rationale>
Answer: 44 <Task Demo: Exemplar_2> <Task Demo: Examplar_3>
2 Work ow Topology Optimization ( 79% → 83% )

P

D

D

( P ) ( )

P

A

. . 

. 

P

( )

P

( D ) A 
P

P

D

P

3 Work ow-level Prompt Optimization ( 83% → 85% )

Predictor: Let's think step by step to solve the given problem. Clearly explain your reasoning process, showing all intermediate calculations and justifications. Express your final answer as a single numerical value or simplified expression enclosed within
<answer></answer> tags. Avoid extraneous text or explanations outside of the core reasoning and final answer. <Task Demo: Exemplar_1>
Figure 7. A demonstration of the optimization trajectory of MASS on MATH. In (1) block-level optimization: multi-agent debate serves as the best-performing topology. In (2) workflow **topology** optimization, aggregating with more parallel agents outweighs the performance of agents in debate. Lastly, (3) workflow-level optimization discovers the optimal prompt conditioned on the best topology.

385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 age performance of MASS from block-level to workflowlevel optimization and compare it with a single agent APO baseline, where the block-level optimization performance indicates the best-performing building block a ∈ A after APO. First, we notice that there is a large gain, 6% on average, between block-level optimization and singleagent optimization, showing that MAS benefits substantially from having its agents optimized inside the building block. In addition, going from Stage (1) to (2), another 3% gain can be achieved by composing influential topologies while searching the optimal configurations. Here, we provide an additional ablation on conducting Stage (2) without prompt optimization beforehand or without search space pruning. Fig. 5 (right) shows that both of them are critical for effective search space exploration. Lastly, MASS obtains further gains (∼2%) by conducting workflow-level prompt optimization on the best-found topology, which indicates that optimizing the prompts towards modeling the interdependence of agents is beneficial in the MAS design. Cost-effectiveness of MASS. We conduct analysis on the cost-effectiveness of MASS. In particular, we visualize the optimization trajectory of MASS as shown in Fig. 6.

MASS's trajectory demonstrates a steady trend of optimization that gradually improves the validation performance via interleaving the search towards better prompts and topologies. However, when it comes to automatic design baselines without explicit prompt optimization stages, AFlow is exposed to a larger variance in its optimization due to the nature of MCTS, whereas ADAS gets trapped in discovering over-complex topologies that appear to be less effective than the prompt design space. Overall, the optimization trajectory of MASS highlights the importance of optimizing in an effective design space, where interleaved optimization further resolves the complexity with more consecutive rewards. Following Sec. 2.1, MASS also demonstrated advanced token-effectiveness, which we refer to Fig. 9. Best-found MAS architectures & Design principles. We further inspect an example of optimized prompts and the trajectory of MASS in discovering more effective topologies in Fig. 7. The optimization starts from a zero-shot CoT agent, and soon MASS in Stage (1) identifies the high-performing topology in debate with its optimized prompt. However, as found in Stage (2), aggregating with more parallel agents actually outweighs the multi-agent debate. Workflow-level prompt optimization then leads to the best-performing predictor for aggregation. The overall optimization flow sheds light on our guidelines for building effective MAS: 1) optimizing individual agents properly is important before composing them into an MAS; 2) more effective MAS can be built by composing influential topologies; and 3) modeling the interdependence between agents is beneficial, and can be achieved via workflow-level joint optimization.

## 6. Conclusion

fl We approach designing effective MAS by first conducting a thorough analysis of the massive design space, revealing the crucial role of prompts, and identifying an influential subset of search space. Building on these findings, we introduce MASS, a novel multi-stage optimization framework that searches within a pruned design space, interleaving prompt and topology optimization to efficiently generate high-performing MAS. Our experiments demonstrate that MASS-optimized MAS significantly outperforms existing manual and automated approaches across an extensive set of tasks. Finally, based on the optimized systems discovered by MASS, we extract valuable design principles to guide the development of future effective LLM-based MAS.

fl

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

Anthropic. The claude 3 model family: Opus, sonnet, haiku.

2024.

Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., et al.

Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

Bai, Y., Lv, X., Zhang, J., Lyu, H., Tang, J., Huang, Z., Du, Z., Liu, X., Zeng, A., Hou, L., Dong, Y., Tang, J., and Li, J. LongBench: A bilingual, multitask benchmark for long context understanding. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3119–3137, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.172. URL https: //aclanthology.org/2024.acl-long.172/.

Chen, J., Saha, S., and Bansal, M. ReConcile: Round-table conference improves reasoning via consensus among diverse LLMs. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), *Proceedings of the 62nd Annual Meeting of* the Association for Computational Linguistics (Volume 1: Long Papers), pp. 7066–7085, Bangkok, Thailand, August 2024a. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.381. URL https: //aclanthology.org/2024.acl-long.381/.

Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. D. O.,
Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al. Evaluating large language models trained on code. *arXiv preprint arXiv:2107.03374*, 2021.

Chen, W., Su, Y., Zuo, J., Yang, C., Yuan, C., Chan, C.-M.,
Yu, H., Lu, Y., Hung, Y.-H., Qian, C., Qin, Y., Cong, X., Xie, R., Liu, Z., Sun, M., and Zhou, J. Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors. In The Twelfth International Conference on Learning Representations, 2024b. URL https: //openreview.net/forum?id=EHg5GDnyq1.

Chen, X., Aksitov, R., Alon, U., Ren, J., Xiao, K., Yin, P.,
Prakash, S., Sutton, C., Wang, X., and Zhou, D. Universal self-consistency for large language models. In ICML 2024 Workshop on In-Context Learning, 2024c. URL https:
//openreview.net/forum?id=LjsjHF7nAN.

440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 Chen, X., Lin, M., Scharli, N., and Zhou, D. Teaching ¨
large language models to self-debug. In *The Twelfth* International Conference on Learning Representations, 2024d. URL https://openreview.net/forum? id=KuPixIqPiq.

Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., and Mordatch, I. Improving factuality and reasoning in language models through multiagent debate. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024. URL https://openreview.net/forum? id=zj7YuTE4t8.

Dua, D., Wang, Y., Dasigi, P., Stanovsky, G., Singh, S.,
and Gardner, M. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In Burstein, J., Doran, C., and Solorio, T. (eds.), Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 2368–2378, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1246. URL https: //aclanthology.org/N19-1246/.

Guo, S., Deng, C., Wen, Y., Chen, H., Chang, Y., and Wang, J. Ds-agent: Automated data science by empowering large language models with case-based reasoning, 2024.

URL https://arxiv.org/abs/2402.17453.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. *NeurIPS*,
2021. URL https://openreview.net/forum? id=7Bywt2mQsCe.

Ho, X., Duong Nguyen, A.-K., Sugawara, S., and Aizawa, A. Constructing a multi-hop QA dataset for comprehensive evaluation of reasoning steps. In Scott, D.,
Bel, N., and Zong, C. (eds.), Proceedings of the 28th International Conference on Computational Linguistics, pp. 6609–6625, Barcelona, Spain (Online), December 2020. International Committee on Computational Linguistics. doi: 10.18653/v1/2020.coling-main. 580. URL https://aclanthology.org/2020.

coling-main.580/.

Hu, S., Lu, C., and Clune, J. Automated design of agentic systems. *arXiv preprint arXiv:2408.08435*, 2024a.

Hu, X., Zhao, Z., Wei, S., Chai, Z., Ma, Q., Wang, G., Wang, X., Su, J., Xu, J., Zhu, M., Cheng, Y., Yuan, J., Li, J., Kuang, K., Yang, Y., Yang, H., and Wu, F. Infiagentdabench: Evaluating agents on data analysis tasks, 2024b. URL https://arxiv.org/abs/2401.05507.

Jain, N., Han, K., Gu, A., Li, W.-D., Yan, F., Zhang, T.,
Wang, S., Solar-Lezama, A., Sen, K., and Stoica, I. Livecodebench: Holistic and contamination free evaluation of large language models for code. *arXiv preprint* arXiv:2403.07974, 2024.

Li, Y., Du, Y., Zhang, J., Hou, L., Grabowski, P.,
Li, Y., and Ie, E. Improving multi-agent debate with sparse communication topology. In Al- Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.), Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 7281–7294, Miami, Florida, USA, November 2024b. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp. 427. URL https://aclanthology.org/2024. findings-emnlp.427/.

495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., and Narasimhan, K. Swe-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770, 2023.

Li, Z., Xu, S., Mei, K., Hua, W., Rama, B., Raheja, O.,
Wang, H., Zhu, H., and Zhang, Y. Autoflow: Automated workflow generation for large language model agents. arXiv preprint arXiv:2407.12821, 2024c.

Kandasamy, K., Neiswanger, W., Schneider, J., Poczos, B.,
and Xing, E. P. Neural architecture search with bayesian optimisation and optimal transport. Advances in neural information processing systems, 31, 2018.

Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y., Wang, R.,
Yang, Y., Shi, S., and Tu, Z. Encouraging divergent thinking in large language models through multi-agent debate.

In Al-Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.),
Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 17889–17904, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024. emnlp-main.992. URL https://aclanthology. org/2024.emnlp-main.992/.

Khan, A., Hughes, J., Valentine, D., Ruis, L., Sachan, K.,
Radhakrishnan, A., Grefenstette, E., Bowman, S. R., Rocktaschel, T., and Perez, E. Debating with more per- ¨ suasive LLMs leads to more truthful answers. In Fortyfirst International Conference on Machine Learning, 2024. URL https://openreview.net/forum? id=iLCZtl7FTa.

Khattab, O., Singhvi, A., Maheshwari, P., Zhang, Z., Santhanam, K., A, S. V., Haq, S., Sharma, A., Joshi, T. T.,
Moazam, H., Miller, H., Zaharia, M., and Potts, C. DSPy:
Compiling declarative language model calls into state-ofthe-art pipelines. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=sY5N0zY5Od.

Lin, L., Fu, J., Liu, P., Li, Q., Gong, Y., Wan, J., Zhang, F., Wang, Z., Zhang, D., and Gai, K. Just ask one more time! self-agreement improves reasoning of language models in (almost) all scenarios. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), *Findings of the Association* for Computational Linguistics: ACL 2024, pp. 3829– 3852, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024. findings-acl.230. URL https://aclanthology. org/2024.findings-acl.230/.

Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., and Iwasawa, Y. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:
22199–22213, 2022.

Liu, F., AlDahoul, N., Eady, G., Zaki, Y., AlShebli, B., and Rahwan, T. Self-reflection outcome is sensitive to prompt construction. *arXiv preprint arXiv:2406.10400*, 2024a.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V.,
Goyal, N., Kuttler, H., Lewis, M., Yih, W.-t., Rockt ¨ aschel, ¨ T., et al. Retrieval-augmented generation for knowledgeintensive nlp tasks. Advances in Neural Information Processing Systems, 33:9459–9474, 2020.

Liu, H., Simonyan, K., and Yang, Y. Darts: Differentiable architecture search. *arXiv preprint arXiv:1806.09055*, 2018.

Li, J., Zhang, Q., Yu, Y., FU, Q., and Ye, D. More agents is all you need. Transactions on Machine Learning Research, 2024a. ISSN 2835-8856. URL https: //openreview.net/forum?id=bgzUSZ8aeg.

Liu, Z., Zhang, Y., Li, P., Liu, Y., and Yang, D. A dynamic LLM-powered agent network for task-oriented agent collaboration. In *First Conference on Language Modeling*, 2024b. URL https://openreview.net/forum? id=XII0Wp1XA9.

Li, M., Zhao, S., Wang, Q., Wang, K., Zhou, Y., Srivastava, S., Gokmen, C., Lee, T., Li, L. E., Zhang, R., Liu, W.,
Liang, P., Fei-Fei, L., Mao, J., and Wu, J. Embodied agent interface: Benchmarking llms for embodied decision making, 2025. URL https://arxiv.org/
abs/2410.07166.

Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S.,
Yang, Y., et al. Self-refine: Iterative refinement with selffeedback. *Advances in Neural Information Processing* Systems, 36, 2024.

Opsahl-Ong, K., Ryan, M. J., Purtell, J., Broman, D., Potts, C., Zaharia, M., and Khattab, O. Optimizing instructions and demonstrations for multi-stage language model programs. In Al-Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 9340– 9366, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024. emnlp-main.525. URL https://aclanthology. org/2024.emnlp-main.525/.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C.,
Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. *Advances in neural information* processing systems, 35:27730–27744, 2022.

Pryzant, R., Iter, D., Li, J., Lee, Y., Zhu, C., and Zeng, M. Automatic prompt optimization with "gradient descent" and beam search. In Bouamor, H., Pino, J., and Bali, K. (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 7957–7968, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023. emnlp-main.494. URL https://aclanthology. org/2023.emnlp-main.494/.

Qian, C., Xie, Z., Wang, Y., Liu, W., Dang, Y., Du, Z.,
Chen, W., Yang, C., Liu, Z., and Sun, M. Scaling largelanguage-model-based multi-agent collaboration. arXiv preprint arXiv:2406.07155, 2024.

Qiao, S., Zhang, N., Fang, R., Luo, Y., Zhou, W., Jiang, Y.,
Lv, C., and Chen, H. AutoAct: Automatic agent learning from scratch for QA via self-planning. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3003–3021, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long. 165. URL https://aclanthology.org/2024. acl-long.165/.

Reid, M., Savinov, N., Teplyashin, D., Lepikhin, D., Lillicrap, T. P., Alayrac, J., Soricut, R., Lazaridou, A., Firat, O., Schrittwieser, J., Antonoglou, I., Anil, R., Borgeaud, S., Dai, A. M., Millican, K., Dyer, E., Glaese, M., Sottiaux, T., Lee, B., Viola, F., Reynolds, M., Xu, Y., Molloy, J., Chen, J., Isard, M., Barham, P., Hennigan, T., McIlroy, R., Johnson, M., Schalkwyk, J., Collins, E., Rutherford, E., Moreira, E., Ayoub, K., Goel, M., Meyer, C., Thornton, G., Yang, Z., Michalewski, H., Abbas, Z., Schucher, N., Anand, A., Ives, R., Keeling, J., Lenc, K., Haykal, S., Shakeri, S., Shyam, P., Chowdhery, A., Ring, R., Spencer, S., Sezener, E., and et al.

Gemini 1.5: Unlocking multimodal understanding across 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 millions of tokens of context. *CoRR*, abs/2403.05530, 2024. doi: 10.48550/ARXIV.2403.05530. URL https: //doi.org/10.48550/arXiv.2403.05530.

Ru, B., Wan, X., Dong, X., and Osborne, M. Interpretable neural architecture search via bayesian optimisation with weisfeiler-lehman kernels. International Conference on Learning Representations (ICLR), 2021.

Saad-Falcon, J., Lafuente, A. G., Natarajan, S., Maru, N.,
Todorov, H., Guha, E., Buchanan, E. K., Chen, M., Guha, N., Re, C., et al. Archon: An architecture search ´ framework for inference-time techniques. *arXiv preprint* arXiv:2409.15254, 2024.

Shang, Y., Li, Y., Zhao, K., Ma, L., Liu, J., Xu, F., and Li, Y. Agentsquare: Automatic llm agent search in modular design space. *arXiv preprint arXiv:2410.06153*, 2024.

Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., and Yao, S. Reflexion: Language agents with verbal reinforcement learning. *Advances in Neural Information* Processing Systems, 36, 2024.

Singh, A., Ehtesham, A., Kumar, S., and Khoei, T. T. Agentic retrieval-augmented generation: A survey on agentic rag. *arXiv preprint arXiv:2501.09136*, 2025.

Su, H., Sun, R., Yoon, J., Yin, P., Yu, T., and Arık, S. O. ¨
Learn-by-interact: A data-centric framework for selfadaptive agents in realistic environments. *arXiv preprint* arXiv:2501.10893, 2025.

Trivedi, H., Balasubramanian, N., Khot, T., and Sabharwal, A. MuSiQue: Multihop questions via single-hop question composition. *Transactions of the Association for* Computational Linguistics, 10:539–554, 2022. doi: 10. 1162/tacl a 00475. URL https://aclanthology. org/2022.tacl-1.31/.

Verma, M., Bhambri, S., and Kambhampati, S. On the brittle foundations of react prompting for agentic large language models. *arXiv preprint arXiv:2405.13966*, 2024.

Wan, X., Ru, B., Esperanc¸a, P. M., and Li, Z. On redundancy and diversity in cell-based neural architecture search. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum? id=rFJWoYoxrDB.

Wan, X., Sun, R., Nakhost, H., and Arik, S. O. Teach better or show smarter? on instructions and exemplars in automatic prompt optimization. In *The Thirty-eighth* Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/
forum?id=IdtoJVWVnX.

Wang, F., Wan, X., Sun, R., Chen, J., and Arık, S. O. Astute ¨
rag: Overcoming imperfect retrieval augmentation and knowledge conflicts for large language models. arXiv preprint arXiv:2410.07176, 2024a.

Wang, J., Wang, J., Athiwaratkun, B., Zhang, C., and Zou, J. Mixture-of-agents enhances large language model capabilities. *arXiv preprint arXiv:2406.04692*, 2024b.

Wang, Q., Wang, Z., Su, Y., Tong, H., and Song, Y. Rethinking the bounds of LLM reasoning: Are multi-agent discussions the key? In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 6106–6131, Bangkok, Thailand, August 2024c. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.331. URL https: //aclanthology.org/2024.acl-long.331/.

Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi, E. H., Narang, S., Chowdhery, A., and Zhou, D. Selfconsistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=1PL1NIMMrw.

White, C., Safari, M., Sukthanker, R., Ru, B., Elsken, T., Zela, A., Dey, D., and Hutter, F. Neural architecture search: Insights from 1000 papers. arXiv preprint arXiv:2301.08727, 2023.

Wu, Q., Bansal, G., Zhang, J., Wu, Y., Zhang, S., Zhu, E., Li, B., Jiang, L., Zhang, X., and Wang, C. Autogen: Enabling next-gen llm applications via multi-agent conversation framework. *arXiv preprint arXiv:2308.08155*, 2023.

Xia, C. S., Deng, Y., Dunn, S., and Zhang, L. Agentless: Demystifying llm-based software engineering agents. arXiv preprint arXiv:2407.01489, 2024.

Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W.,
Salakhutdinov, R., and Manning, C. D. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Riloff, E., Chiang, D., Hockenmaier, J., and Tsujii, J. (eds.), Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 2369–2380, Brussels, Belgium, October- November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1259. URL https: //aclanthology.org/D18-1259/.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. R., and Cao, Y. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?

id=WE_vluYUL-X.

605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659 Zhang, G., Yue, Y., Li, Z., Yun, S., Wan, G., Wang, K.,
Cheng, D., Yu, J. X., and Chen, T. Cut the crap: An economical communication pipeline for llm-based multiagent systems. *arXiv preprint arXiv:2410.02506*, 2024a.

Zhang, J., Xiang, J., Yu, Z., Teng, F., Chen, X., Chen, J.,
Zhuge, M., Cheng, X., Hong, S., Wang, J., et al. Aflow: Automating agentic workflow generation. *arXiv preprint* arXiv:2410.10762, 2024b.

Zhang, J., Xu, X., Zhang, N., Liu, R., Hooi, B., and Deng, S. Exploring collaboration mechanisms for LLM agents: A social psychology view. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 14544–14607, Bangkok, Thailand, August 2024c. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long. 782. URL https://aclanthology.org/2024. acl-long.782/.

Zhang, S., Zhang, J., Liu, J., Song, L., Wang, C., Krishna, R., and Wu, Q. Offline training of language model agents with functions as learnable weights. In Fortyfirst International Conference on Machine Learning, 2024d. URL https://openreview.net/forum? id=2xbkWiEuR1.

Zhang, W., Tang, K., Wu, H., Wang, M., Shen, Y., Hou, G., Tan, Z., Li, P., Zhuang, Y., and Lu, W. Agent-pro: Learning to evolve via policy-level reflection and optimization. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), *Proceedings of the 62nd Annual Meeting of* the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5348–5375, Bangkok, Thailand, August 2024e. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.292. URL https: //aclanthology.org/2024.acl-long.292/.

Zhang, Y., Sun, R., Chen, Y., Pfister, T., Zhang, R., and Arik, S. O. Chain of agents: Large language models collaborating on long-context tasks. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024f. URL https://openreview.net/ forum?id=LuCLf4BJsr.

Zhou, H., Wan, X., Vulic, I., and Korhonen, A. ´
Survival of the most influential prompts: Efficient black-box prompt search via clustering and pruning. In Bouamor, H., Pino, J., and Bali, K. (eds.),
Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 13064–13077, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp. 870. URL https://aclanthology.org/2023.

findings-emnlp.870/.

Zhou, H., Wan, X., Proleev, L., Mincu, D., Chen, J., Heller, K. A., and Roy, S. Batch calibration: Rethinking calibration for in-context learning and prompt engineering. In The Twelfth International Conference on Learning Representations, 2024b. URL https://openreview. net/forum?id=L3FHMoKZcS.

Zhou, W., Ou, Y., Ding, S., Li, L., Wu, J., Wang, T.,
Chen, J., Wang, S., Xu, X., Zhang, N., et al. Symbolic learning enables self-evolving agents. arXiv preprint arXiv:2406.18532, 2024c.

Zhuge, M., Wang, W., Kirsch, L., Faccio, F., Khizbullin, D., and Schmidhuber, J. GPTSwarm: Language agents as optimizable graphs. In Forty-first International Conference on Machine Learning, 2024. URL https: //openreview.net/forum?id=uTC9AFXIhg.

660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714 Zhou, H., Wan, X., Liu, Y., Collier, N., Vulic, I., and Ko- ´
rhonen, A. Fairer preferences elicit improved humanaligned large language model judgments. In Al-Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 1241–1252, Miami, Florida, USA, November 2024a. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main. 72. URL https://aclanthology.org/2024. emnlp-main.72/.

## A. Limitations And Future Work

MASS is a multi-agent design meta-framework also orthogonal to prompt and topology optimizers. MASS has brought substantial improvements over a single agent design by searching in a customizable topology space. Though our proposed topology space has covered the vast majority of effective MAS designs, including serial, parallel, and mixture of connections, it is still likely that incorporating other topologies may further improve the final performance of MASS, which is complementary to the development of MASS. For instance, the debate topology proposed in MASS involves a fully-connected topology across agents. Recent work has been identifying the sparsity of agent communications (Li et al., 2024b; Zhang et al., 2024a),
and pruning redundant communications may further enhance the overall efficiency of the strongest MASS-found design. Though the topology optimizer in MASS already traverses efficiently in the proposed topology space, incorporating more advanced search algorithms, such as the Bayes optimizer (Kandasamy et al., 2018; Ru et al., 2021), may further improve the sample efficiency of MASS when faces a more complex design space. Similarly, the sample efficiency of the prompt optimizer may be further enhanced by conditioning on textual feedback from error logs (Pryzant et al., 2023; Wan et al., 2024), which we will endeavor to explore in future work.

## B. Implementation Details B.1. Datasets 715

716 717 718 719 720 721 722 723 724 725 726 727 728 729

## 730 731

732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769 In this work, we included the following dataset: 1) Hendryck's MATH (Hendrycks et al., 2021) consisting challenging competition-level mathematics problems, and DROP (Dua et al., 2019) requires discrete and symbolic reasoning over paragraphs; 2) HotpotQA (Yang et al., 2018), MuSiQue (Trivedi et al., 2022), and 2WikiMultiHopQA (Ho et al., 2020) to evaluate on information seeking from long-context with agentic systems, which we report from standardized versions in LongBench (Bai et al., 2024); 3) MBPP (Austin et al., 2021), HumanEval (Chen et al., 2021), and LiveCodeBench (Jain et al., 2024) as well-established coding benchmarks. Regarding LiveCodeBench, we use the 'test output prediction' task as an agent cooperative task. In line with AFlow (Zhang et al., 2024b), we use the public test cases of MBPP and HumanEval for the executor to retrieve reliable external feedback signals.

To save computation resources, we randomly sample a subset of the original validation and test splits to conduct all the experiments, where the specifications are reported in Table 2.

| Task          | Type                           | |Val|   | |Test|   | Topology Search Space                   | MASS         |
|---------------|--------------------------------|---------|----------|-----------------------------------------|--------------|
| MATH          | Mathematical Reasoning         | 60      | 100      | {Aggregate, Reflect, Debate}            | {9, 0, 0}    |
| DROP          | Discrete Reasoning             | 60      | 200      | {Aggregate, Reflect, Debate}            | {5, 0, 0}    |
| HotpotQA      | Long-context Understanding     | 50      | 100      | {Summarize, Aggregate, Reflect, Debate} | {0, 5, 0, 1} |
| MuSiQue       | Long-context Understanding     | 50      | 100      | {Summarize, Aggregate, Reflect, Debate} | {0, 3, 0, 2} |
| 2WikiMQA      | Long-context Understanding     | 50      | 100      | {Summarize, Aggregate, Reflect, Debate} | {0, 3, 0, 1} |
| MBPP          | Coding                         | 60      | 200      | {Aggregate, Reflect, Debate, Executor}  | {1, 4, 0, 1} |
| HumanEval     | Coding                         | 50      | 100      | {Aggregate, Reflect, Debate, Executor}  | {1, 3, 0, 1} |
| LiveCodeBench | Coding: test output prediction | 100     | 200      | {Aggregate, Reflect, Debate, Executor}  | {3, 1, 1, 1} |

Table 3. The search dimension for each topology. The minimum topology defines the building block that MASS Stage (1) optimized.

| Topology   | Search Space    | Minimum Topology Building Block   | Specification   |
|------------|-----------------|-----------------------------------|-----------------|
| Summarize  | {0, 1, 2, 3, 4} | {Summarizer, Predictor}           | {1, 1}          |
| Aggregate  | {1, 3, 5, 7, 9} | {Predictor, Aggregator}           | {3, 1}          |
| Reflect    | {0, 1, 2, 3, 4} | {Predictor, Reflector}            | {1, 1}          |
| Debate     | {0, 1, 2, 3, 4} | {Predictor, Debator}              | {2, 1}          |
| Execute    | {0, 1}          | {Predictor, Executor, Reflector}  | {1, 1, 1}       |

## B.2. Baselines

770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 810 811 812 813 814 815 816 817 818 819 820 821 822 823 824

Topology Building Blocks Summarize Aggregate Re ect Debate Executor P

A

P

S P P P R

P E R

P

P

D
MASS-optimized Topology MATH DROP HotpotQA MuSiQue MBPP HumanEval LiveCodeBench P

A

. . 

. 

P

2WikiMQA

E

E

E

R

R

R

P

P

P

P

P

D D

D D
D

D

D
P

P

D

D

D
D

A

. . 

. 

P

A

. . 

.

. . 

.

P

P

R

E

P

P

P

P

R

E

A

A

A
P

D

D D

In this section, we report the specifications of all our baselines. We note that for the baselines: CoT, SC, Self-Refine, and Multi-Agent Debate, we follow the prompts given in ADAS (Hu et al., 2024a). 1) Chain-of-Thought (CoT) (Kojima et al., 2022). Direct chain-of-thought reasoning via zero-shot prompting: "Please think step by step and then solve the task." 2) Self-Consistency (SC) (Wang et al., 2023). In self-consistency, we generate diverse chain-of-thought reasoning traces with a temperature of 0.8, followed by a rule-based majority vote that collects the most consistent answer. In Table 1, we report SC@9 to provide a fair comparison across baselines. 3) Self-Refine (Madaan et al., 2024): This baseline consists of one predictor that constantly takes feedback and a self-reflector that provides criticism. It involves a stop criterion whenever the self-reflector outputs "correct" in its prediction. We set the maximum number of rounds of reflections to 5, such that the worst case will involve 11 (1 + 2 × 5) calls. 4) Multi-Agent Debate (Du et al., 2024; Liang et al., 2024). In this baseline, it involves 3 agents that conduct reasoning and debating for 3 rounds. The opinions along the rounds of debating are finally judged by an aggregator that makes the final prediction. Hence, it contains 10 (3 × 3 + 1) agents.

5) Automated Design of Agentic Systems (ADAS) (Hu et al., 2024a). Consistent with our main experimental setups. We use Gemini 1.5 as both LLM optimizer and evaluator for reproducing all ADAS results. The generation of ADAS is conditioned on former evaluations of baselines, including CoT, SC, Self-Refine, and Multi-Agent Debate. We report ADAS with 30 rounds of search, and each round is evaluated on the validation set 3 times to stablize the prediction. 6) AFlow (Zhang et al., 2024b). Automatic workflow design via Monte-Carto Tree Search over a set of pre-defined operators.

Similar to ADAS, AFlow also relies on an LLM optimizer to generate new nodes and topologies expressed in codes.

However, we find the meta-prompt of AFlow does not generalize to other LLM backbones. Consequently, we report AFlow with its original LLM optimizer by Claude 3.5 Sonnet, and reproduce experiments with Gemini 1.5 Pro as the LLM executor. Therefore, the comparison is not completely fair, and we treat the results from AFlow as a good reference. We note that the '-' in Table 1 refers to out-of-time errors, where the LLM executor has been trapped in executing accidental scripts with infinite loops. We still endeavored to report most results from AFlow as shown in Table 1 & Fig. 6 with the default experimental setup from AFlow: 20 rounds, 5 runs of validation per round, and k at 3.

fl

## B.3. Mass**: Multi-Agent System Search**

In this section, we provide additional details for MASS. The topology search space for each task is defined in Table 2. In addition, for Stage (1) block-level prompt optimization, the specification of the building block is defined in Table 3. We provide the visualization of both the minimum building blocks and the optimized topology in Fig. 8. We refer the reader to App. §D & §E for the prompt templates we used to define each type of agent and the best prompts discovered.

## C. Additional Experiments

Table 4. Results on the evaluation set with Claude 3.5 Sonnet. We keep the same experimental setup as Table 1. Since Claude 3.5 Sonnet does not support the same context window as Gemini, we report the standard HotpotQA instead of the LongBench. As we transfer the prompt template for each agent from Gemini to Claude, it is noticeable that the basic topology on some tasks may result in severe degradation of performance, and MASS successfully recovers the performance and brings significant improvements over the initial agent. Table 5. The detailed ablation results per optimization stage of MASS. Practical gains can be obtained by further conducting workflow-level prompt optimization (3PO) on the best-found topology.

825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861

862

863 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879

MATH (gemini-1.5-pro-002)
0 1000 2000 3000 4000 5000 6000 7000 8000 Total Tokens 70 72 74 76 78 80 82 84 MASSMASS 
MASS MASS
Refine@5 A
c c u ra c y
 (
%
)

Debate 1R@2A
Debate 2R@3A
Step-Back Quality-Diverse ADAS-Tool ADAS-T&S
CoT-SC@3 CoT-SC@5 CoT
Role Assign

| Claude-3.5-Sonnet   |           |           |           |           |           |           |       |
|---------------------|-----------|-----------|-----------|-----------|-----------|-----------|-------|
| Task                | Reasoning | Multi-hop | Coding    |           |           |           |       |
| Method              | MATH      | DROP      | HotpotQA  | MBPP      | HumanEval | LCB       | Avg.  |
| CoT                 | 57.330.94 | 55.520.42 | 23.561.52 | 67.501.47 | 88.671.70 | 72.672.39 | 60.21 |
| Self-Consistency    | 61.671.89 | 57.860.45 | 25.690.44 | 69.170.62 | 90.000.82 | 72.672.39 | 62.84 |
| Self-Refine         | 57.001.63 | 56.260.56 | 23.572.56 | 68.000.82 | 87.001.41 | 49.331.65 | 56.86 |
| Multi-Agent Debate  | 45.003.74 | 26.620.11 | 31.413.30 | 00.000.00 | 84.333.30 | 72.821.84 | 43.36 |
| MASS                | 63.000.00 | 68.930.38 | 66.980.99 | 68.830.62 | 93.000.82 | 73.731.43 | 72.43 |

| Gemini-1.5-pro-002   |           |                        |           |           |           |           |           |           |       |
|----------------------|-----------|------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-------|
| Task                 | Reasoning | Multi-hop Long-context | Coding    |           |           |           |           |           |       |
| Method               | MATH      | DROP                   | HotpotQA  | MuSiQue   | 2WikiMQA  | MBPP      | HumanEval | LCB       | Avg.  |
| Base Agent           | 62.330.94 | 71.650.61              | 56.961.26 | 43.320.13 | 49.200.61 | 68.830.85 | 89.331.70 | 66.332.09 | 63.54 |
| + APO                | 79.331.89 | 77.510.38              | 59.720.00 | 43.970.00 | 61.490.24 | 67.001.08 | 86.331.25 | 68.501.22 | 67.44 |
| + 1PO                | 80.000.00 | 86.450.90              | 62.521.86 | 48.860.61 | 67.400.58 | 80.331.25 | 91.671.25 | 76.000.00 | 74.56 |
| + 2TO                | 83.001.63 | 86.751.32              | 65.221.34 | 52.610.52 | 72.820.86 | 85.001.08 | 92.000.82 | 81.330.00 | 77.55 |
| + 3PO                | 84.670.47 | 90.520.64              | 69.911.11 | 51.400.42 | 73.340.67 | 86.500.41 | 91.670.47 | 82.330.85 | 78.40 |

Figure 9. The Pareto-front of MASS-optimized designs compared to multi-agent baselines. Total tokens include both inference input tokens and output tokens. Additional multi-agent baselines from ADAS (Hu et al., 2024a) and two best-found ADAS designs are included.

## D. Prompt Template

We provide all prompt templates we used for defining the MASS search space. We use <> to enclose texts that have been skipped for presentation purposes. We follow the DSPy (Khattab et al., 2024) in constructing these agentic templates. The general template for instruction, exemplar, and input/output fields:
<Instruction> Follow the following format. Input: ${Input} ... Output: ${output} --- <example_1> --- Input: <Input> ... Output: <output>
MATH:
Predictor: Let's think step by step. --- Question: ${question} Reasoning: Let's think step by step in order to ${produce the answer}. We ... Answer: ${answer} ------------ Reflector: Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output 'True' in 'correctness'.

---
Question: ${question} Text: ${text} Reasoning: Let's think step by step in order to ${produce the correctness}. We ... Feedback: ${feedback} Correctness: True/False indicating if answer is correct given the question. ------------ Refiner: Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better. Show your final answer bracketed between <answer > and </answer> at the end.

--- Question: ${question} Previous answer: ${previous_answer} Reflection: ${reflection} Correctness: ${correctness} Thinking: ${thinking} Answer: ${answer} ------------ Debator: These are the solutions to the question from other agents. Examine the solutions from other agents in your rationale
, finish by giving an updated answer. Show your final answer bracketed between <answer> and </answer> at the end.

--- Question: ${question} Solutions: the solutions to the question from other agents Reasoning: Let's think step by step in order to ${Examine the solutions from other agents}. We ... Answer: The updated answer for the question. Do not repeat Answer:
880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934

# Multi-Agent Design: Optimizing Agents With Better Prompts And Topologies

935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 DROP:
Predictor: Please think step by step and then solve the task. \# Your Task: Please answer the following question based on the given context. --- Question: ${question} Context: ${context} Thinking: ${thinking} Answer: Directly answer the question. Keep it very concise. ------------ Reflector: Verify that the answer is based on the provided context. Give your reflection in the rationale. --- Question: ${question} Context: ${context} Text: ${text} Reasoning: Let's think step by step in order to ${produce the correctness}. We ... Correctness: True/False indicating if answer is correct given the observations and question. ------------ Refiner: Please think step by step and then solve the task. \# Your Task: Based on the reflection, correctness of the previous answer, and the context again, give an updated answer.

---
Question: ${question} Context: ${context} Previous answer: ${previous_answer} Reflection: ${reflection} Correctness: ${correctness} Thinking: ${thinking} Answer: Directly answer the question. Keep it very concise. ------------ Debator: These are the solutions to the question from other agents. Based on the context, examine the solutions from other agents in your rationale, finish by giving an updated answer.

--- Question: ${question} Context: ${context} Solutions: the solutions to the question from other agents Reasoning: Let's think step by step in order to ${Examine the solutions from other agents}. We ... Answer: The updated answer for the question. Do not repeat Answer:
969 970 971 972 973 974 975 976 978 979 980 981 982 983 984 985 986 987 988 989 HotpotQA, MuSiQue, and 2WikiMQA:
Predictor: Answer the question with information based on the context. Only return the answer as your output. --- Question: ${question} Context: ${context} Answer: Only give me the answer. Do not output any other words. ------------
Summarizer:
Based on the question, retrieve relevant information from context that is ONLY helpful in answering the question.

Include all key information. Do not repeat context.

--- Question: ${question} Context: ${context} Summary: Only generate the summary. Start with Summary: ------------ Reflector: Verify that the answer is based on the provided context. --- Question: ${question} Context: ${context} Text: ${text}

# Multi-Agent Design: Optimizing Agents With Better Prompts And Topologies

990 991 992 993 994 996 997 998 Reasoning: Let's think step by step in order to ${produce the correctness}. We ... Correctness: True/False indicating if answer is correct given the observations and question. ------------ Debator: These are the solutions to the question from other agents. Based on the context, examine the solutions from other agents in your rationale, finish by giving an updated answer.

--- Question: ${question} Context: ${context} Solutions: the solutions to the question from other agents Reasoning: Let's think step by step in order to ${Examine the solutions from other agents}. We ... Answer: The updated answer for the question. Do not repeat Answer:
999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 MBPP:
Predictor: Let's think step by step. Provide a complete and correct code implementation in python. --- Question: ${question} Thinking: ${thinking} Answer: Only the code implementation. Do not include example usage or explainations. ------------ Reflector: Please determine the correctness of the solution in passing all test cases. If it fails, based on the error message and trackback, think step by step, carefully propose an updated solution in the answer output with a correct code implementation in python.

--- Question: ${question} Previous solution: ${previous_solution} Traceback: It contains the test cases, execution results, and ground truth. If there is an error, the relevant traceback is given.

Correctness: 'True/False' based on the correctness of executive feedback. If there is an error message, output '
False' Thinking: ${thinking} Answer: ${answer} ------------ Debator: These are the solutions to the question from other agents. Examine the solutions from other agents in your rationale
, finish by giving an updated answer. Let's think step by step. Provide a complete and correct code implementation in python.

--- Question: ${question} Solutions: the solutions to the question from other agents Reasoning: Let's think step by step in order to ${Examine the solutions from other agents}. We ... Answer: ${answer}
1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 HumanEval:
Predictor: Let's think step by step. Provide a complete and correct code implementation in python.

---
Question: ${question} Thinking: ${thinking} Answer: ${answer} ------------ Reflector: Please determine the correctness of the solution in passing all test cases. If it fails, based on the error message and trackback, think step by step, carefully propose an updated solution in the answer output with a correct code implementation in python.

--- Question: ${question} Previous solution: ${previous_solution} Traceback: ${traceback} Thinking: ${thinking} Answer: ${answer}
1039 1040 1041 1042 1043 1044 995 1045 1046 1047 Debator:
1048 1049 1050 1051 These are the solutions to the question from other agents. Examine the solutions from other agents in your rationale
, finish by giving an updated answer. Let's think step by step. Provide a complete and correct code implementation in python.

--- Question: ${question} Solutions: the solutions to the question from other agents Reasoning: Let's think step by step in order to ${Examine the solutions from other agents}. We ... Answer: ${answer}
1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 LiveCodeBench:
Predictor: You are a helpful programming assistant and an expert Python programmer. The user has written a input for the testcase. Think step by step. You will generate the code based on the problem requirepement. You will calculate the output of the testcase and write the whole assertion statement in the markdown code block with the correct output.

--- Question: ${question} Thinking: ${thinking} Code: ${code} Answer: complete the testcase with assertion. ------------
Reflector:
If there is an executive output in the traceback, parse the output into an assertion in the answer given the executive output.

--- Question: ${question} Previous solution: ${previous_solution} Traceback: It contains the test cases, execution results, and ground truth. If there is an error, the relevant traceback is given.

Correctness: 'True/False' based on the correctness of executive feedback. If there is an error message, output '
False' Thinking: ${thinking} Answer: ${answer} ------------ Debator: These are the solutions to the question from other agents. Examine the solutions from other agents in your rationale
, finish by giving an updated answer.

--- Question: ${question} Solutions: the solutions to the question from other agents Reasoning: Let's think step by step in order to ${Examine the solutions from other agents}. We ... Answer: assert ${function(input)} == {executive_output}
1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079 1080 1081 1082 1085 1086 1087 In this section, we show a few typical prompts that have been discovered by MASS. Similarly, we indicate <> that skips some comments.

1088 1089 1090 MATH:
Predictor: Let's think step by step to solve the given problem. Clearly explain your reasoning process, showing all intermediate calculations and justifications. Express your final answer as a single numerical value or simplified expression enclosed within <answer></answer> tags. Avoid extraneous text or explanations outside of the core reasoning and final answer.

--- Follow the following format. Question: ${question} Reasoning: Let's think step by step in order to ${produce the answer}. We ... Answer: ${answer}
1091 1092 1093 1094 1095 1096 1097 1098 1099

## E. Best Prompts Discovered 1084

1083