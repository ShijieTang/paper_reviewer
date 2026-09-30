# Ebench**: Comprehensive Benchmarking Multi-Modal Large** Language Models For Vision-Driven Embodied Agents

Rui Yang 1 * **Hanyang Chen** 1 * **Junyu Zhang** 1 * **Mark Zhao** 3‡ *
Cheng Qian 1 Kangrui Wang 2 Qineng Wang 2 Teja Venkat Koripella 1 **Marziyeh Movahedi** 4‡
Manling Li 2 Heng Ji 1 Huan Zhang 1 † **Tong Zhang** 1 †

## Abstract

Leveraging Multi-modal Large Language Models (MLLMs) to create embodied agents offers a promising avenue for tackling real-world tasks. While language-centric embodied agents have garnered substantial attention, MLLM-based embodied agents remain underexplored due to the lack of comprehensive evaluation frameworks. To bridge this gap, we introduce EMBODIEDBENCH, an extensive benchmark designed to evaluate visiondriven embodied agents. EMBODIEDBENCH features: (1) a diverse set of 1,128 testing tasks across four environments, ranging from high-level semantic tasks (e.g., household) to low-level tasks involving atomic actions (e.g., navigation and manipulation); and (2) six meticulously curated subsets evaluating essential agent capabilities like commonsense reasoning, complex instruction understanding, spatial awareness, visual perception, and long-term planning. Through extensive experiments, we evaluated 24 leading proprietary and open-source MLLMs within EMBODIEDBENCH. Our findings reveal that: MLLMs excel at highlevel tasks but struggle with low-level manipulation, with the best model, GPT-4o, scoring only 28.9% on average. EMBODIEDBENCH provides a multifaceted standardized evaluation platform that not only highlights existing challenges but also offers valuable insights to advance MLLM- based embodied agents. Our code and dataset are available at https://embodiedbench.github.io.

## 1. Introduction

Developing embodied agents capable of solving complex tasks in real world remains a significant challenge (Durante et al., 2024). Recent advancements in foundation models, including Large Language Models (LLMs) (Brown et al., 2020; Achiam et al., 2023; Touvron et al., 2023; Yang et al., 2024a) and Multimodal Large Language Models (MLLMs) (OpenAI, 2024a; Reid et al., 2024; Liu et al., 2024a; Wang et al., 2024; Chen et al., 2023c; 2025), have unlocked unprecedented potential toward this goal. These models, trained on extensive internet-scale datasets, demonstrate exceptional proficiency in understanding human knowledge and performing human-like reasoning. Based on these capabilities, researchers can now design intelligent agents that use off-the-shelf foundation models to solve complex tasks through interaction with environments (Huang et al., 2022a;b; 2023c; Ahn et al., 2022; Song et al., 2023; Singh et al., 2023; Liang et al., 2023; Qian et al., 2024). Given the multitude of proposed algorithms, there is a pressing need for standardized and automated evaluation frameworks to enable comprehensive assessment and comparison. To address this need, several initiatives have been exploring LLM-based embodied agent evaluation (Liu et al., 2023b; Choi et al., 2024; Li et al., 2024b). While these efforts significantly contribute to understanding LLM-based agent design, the evaluation of MLLM embodied agents remains underexplored, posing a challenge for creating more versatile agents. VisualAgentBench (Liu et al., 2024e) represents the first benchmark for evaluating MLLM agents, covering embodied tasks such as household and Minecraft. However, its limited scope, focusing exclusively on high-level planning, leaves critical questions unanswered, such as the role of vision in embodied tasks and the performance of MLLM agents in low-level tasks like navigation and manipulation. To address these questions, we introduce EMBODIED-
BENCH, a comprehensive benchmark comprising 1,128 testing instances across four environments. EMBODIEDBENCH is designed with two key features that set it apart from existing benchmarks: **1. Diverse tasks with hierarchical action** levels. Among the four environments, EB-ALFRED and EB-
1

Hierarchical Action Representation Vision-Driven Embodied Agents High-level:
Low-level: [X, Y, Z, Roll, Pitch, Yaw, Gripper]
[{"action": "find a HandTowel"}, {"action": "pick up the HandTowel"},…]
Tasks with various action levels Capability-oriented fine-grained evaluation Instruction: Put the books on the desk.

High-level Household EB-ALFRED EB-Habitat Base Capability Spatial Awareness Stack the right cylinder on top of the left moon.

Place a plate with a spoon on a counter.

Common Sense Visual Appearance Instruction: Find an orange on the TV stand and move it to the sink.

I'm feeling thirsty and need a small container to hold water ... Please navigate to that object and stay near it. (cup)
Obtain a round red fruit and put it in the right counter.

Low-level Navigation EB-Navigation Low-level Manipulation EB-Manipulation Place a cold apple slice on the table.

7 Complex Instruction Long Horizon While you're tidying up, find a spot on the back of the toilet to put two rolls of toilet paper. *It's* a great place for convenient access ...

5 8 17 Instruction: Navigate to the laptop and stay close.

Instruction: Pick up the star and place it into the silver container.
Habitat focus on high-level task decomposition and planning (e.g., "put a book on the desk"), while EB-Navigation and EB-Manipulation demand planning with low-level actions (e.g., translational/rotational control) and require precise perception and spatial reasoning. **2. Capability-oriented** evaluation. Unlike previous benchmarks that primarily emphasize overall accuracy (Liu et al., 2023b; Choi et al.,
2024; Liu et al., 2024e) or module-specific performance (Li et al., 2024b), EMBODIEDBENCH introduces a fine-grained evaluation framework that assesses six critical capabilities of embodied agents, including basic task solving, commonsense reasoning, complex instruction understanding, spatial awareness, visual perception, and long-horizon planning. To facilitate the evaluation of MLLMs as embodied agents, we design a unified agent framework that integrates egocentric visual perception, few-shot in-context examples, interaction history, and environment feedback for decisionmaking. This powerful framework can unlock the full potential of current off-the-shelf MLLMs and tackle both highlevel and low-level tasks effectively. Based on EMBOD-
IEDBENCH and our agent pipeline, we evaluate 24 leading closed-source MLLMs (e.g., GPT-4o, Gemini, Claude3.7, and Qwen-VL-Max) and 7B–90B open-source models (e.g., Llama-3.2 Vision (Meta, 2024), InternVL3 (Zhu et al., 2025), Qwen2.5-VL (Bai et al., 2025), and Gemma-3 (Team et al., 2025)). Our evaluation yields three key findings:
(1) While MLLMs excel at high-level tasks, they struggle with low-level manipulation. (2) Long-horizon planning emerges as the most challenging subset. (3) Vision input is crucial for low-level tasks, with performance degrading by 40%–70% when removed, whereas its impact on high-level tasks is minimal. Additionally, our ablation studies provide practical insights into MLLM agent design, particularly regarding image resolution, multi-step image input, and visual in-context learning.

Our contributions are threefold: (1) proposing a comprehensive benchmark suite for evaluating MLLM-based embodied agents with different action levels and fine-grained capability-oriented subsets, (2) the development of an efficient MLLM agent framework, (3) conducting extensive evaluations and ablation studies of leading MLLMs, providing valuable insights for vision-driven agent design.

## 2. Related Work

In embodied agent research, LLMs are primarily used to support high-level planning (Ahn et al., 2022; Huang et al., 2022a;b; Yao et al., 2023; Huang et al., 2023d; Rana et al., 2023; Chen et al., 2023a; Gao et al., 2024b). MLLMs are then integrated for perception-related tasks (Chen et al., 2023b; Wang et al., 2023d; Gao et al., 2024b). Beyond perception, MLLMs also contribute to decision-making, either by directly generating actions in an end-to-end manner (Shridhar et al., 2022; Driess et al., 2023; Du et al., 2023; Mu et al., 2024) or by producing code to develop policy or value functions (Liang et al., 2023; Huang et al., 2023c). As this field rapidly evolves, a variety of simulators (Kolve et al., 2017; Shridhar et al., 2020a; Xiang et al., 2020; Li et al., 2021; 2023) and evaluation benchmarks (Shridhar et al., 2020b;a; James et al., 2020; Zheng et al., 2022; Szot

| VisualAgentBench include domains such as household, games, and Web. 2VLABench is originally used for evaluating VLA models. Benchmark Category Action Level #Env. #Test Tasks Multimodal Fine-grained LLM/VLM Support ALFWorld (Shridhar et al., 2020b) Household High 1 274 × × × Alfred (Shridhar et al., 2020a) Household High 1 3062 ✓ × × VLMbench (Zheng et al., 2022) Manipulation Low 1 4760 ✓ × × Behavior-1K (Li et al., 2023) Household High 1 1000 ✓ × × Language Rearrangement (Szot et al., 2023) Household High 1 1000 ✓ ✓ × GOAT-bench (Khanna et al., 2024) Navigation Low 1 3919 ✓ × × AgentBench (Liu et al., 2023b) Multi-domain1 High 8 1091 × × ✓ Lota-bench (Choi et al., 2024) Household High 2 308 × × ✓ VisualAgentBench (Liu et al., 2024e) Multi-domain1 High 5 746 ✓ × ✓ Embodied Agent Interface (Li et al., 2024b) Household High 2 438 × ✓ ✓ VLABench (Zhang et al., 2024a) Manipulation Low2 1 100 ✓ ✓ ✓ EMBODIEDBENCH (ours) Multi-domain High & Low 4 1128 ✓ ✓ ✓   |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

et al., 2023; Liu et al., 2023b; 2024e; Choi et al., 2024; Li et al., 2024b; Zhang et al., 2024a; Cheng et al., 2025) have emerged. Table 1 provides a comprehensive comparison with existing works, highlighting how EMBODIEDBENCH sets itself apart from prior works in several aspects. More related works are listed in Appendix A.

## 3. Problem Formulation

Definition of Action Levels. In embodied agent research, actions can be systematically classified into hierarchical levels based on their executability in robotic systems (Ma et al., 2024b; Belkhale et al., 2024). Lowlevel actions correspond to atomic commands directly executable by robots, defined as operations that specify translational or rotational displacements. For instance, a robotic arm's action is often parameterized as a 7-dimensional vector: a = [*X, Y, Z,* Roll,Pitch, Yaw, Gripper], where
(*X, Y, Z*) denote incremental translational displacements, (Roll,Pitch, Yaw) represent rotational deltas in Euler angles, and Gripper encodes the binary open/closed state of the end-effector. Similarly, commands like "move forward 0.1 m" qualify as low-level actions, as they map unambiguously to kinematic transformations. In contrast, high-level actions can be decomposed into sequences of low-level primitives. Formally, a high-level action is defined as a h = [a1, a2*, . . . , a*n], where each aiis a low-level executable primitive. For example, executing
"find a HandTowel" might involve iterating through lowlevel behaviors: rotating certain degrees, scanning for the target, and moving towards it.

Vision-driven Agents. Vision-driven agents are autonomous systems that make sequential decisions based on visual perception and language instructions. This problem can be formally modeled as a Partially Observable Markov Decision Process (POMDP) augmented with language instructions, defined by the tuple (S, A, Ω, T , O*, L,* R). Here, S is the complete state space unobservable to the agent; A is the space of high-level or low-level actions for the agents; Ω is the visual perception space, where each observation It ∈ Ω corresponds to an image frame at time t; T is the transition dynamics; O relates the underlying states to the agent's visual observations; L is the language instruction that specifies the desired goal; R evaluates task completion given the language instruction (
L: rt =
1 if st |= L (instruction achieved)
0 otherwise
. At timestep t, the agent maintains a history ht = (I0, a0, ..., It−1, at−1, It) and selects actions through a policy π(at|*L, h*t). The objective is to maximize the probability of task success:
maxπ E [rτ ], where τ is the terminal timestep—either when the task is successfully completed (sτ |= L) or when the maximum horizon is reached.

## 4. Embodiedbench

To thoroughly assess MLLMs as embodied agents across various action levels and capabilities, we introduce EM-
BODIEDBENCH, a benchmark comprising four environments: EB-ALFRED, EB-Habitat, EB-Navigation, and EB- Manipulation. To evaluate six core embodied agents' capabilities, we developed new datasets and enhanced existing simulators to support comprehensive assessments. Below is an overview of the four benchmark tasks, with further details available in Appendix C.

## 4.1. High-Level And Low-Level Tasks

EB-ALFRED. We develop EB-ALFRED based on the AL- FRED dataset (Shridhar et al., 2020a) and the AI2-THOR simulator (Kolve et al., 2017). Our simulator is based on Lota-Bench's implementation (Choi et al., 2024) for 8 highlevel skill types: "pick up", "open", "close", "turn on",
"turn off", "slice", "put down", and "find", each customizable with specific objects, for example, "find an apple". The simulator provides an egocentric view as observation, along with textual feedback on action validity and possible failure reasons. Despite its strengths, Lota-Bench's simulator has several limitations, which we outline in Appendix C.1. To

Visual Perception
"Stack the lime moon and the gray moon in sequence." Human Instruction In-context Demonstrations with Reasoning Example 1: Human Instruction: Stack the red star and the olive star in sequence.

Input: {'object 1': [42, 24, 19], 'object 2': [54, 53, 19],...}
Output: {"visual_state_description": "From left to right, I can see a purple star at [42, 24, 19], an olive star at [54, 53, 19]...",
"reasoning_and_reflection": "I need to understand the instruction first. To stack the objects in sequence, the red star should be at the bottom and the olive star should be placed on top. The plan involves moving the gripper to...",
"executable_plan": "[[54, 51, 28, 3, 66, 37, 1], [54, 54, 20, 3, 66, 37, 0]...]"
Example 2: ...

Environment Feedback **Interaction History** Output Action History General Information Skill Sets
["Move forward by 0.25",
"Move leftward by 0.25", "Rotate to right by 90 degrees","Rotate to left by 90 degrees..."]
Optional Information Task-
Specific Information Detection Box Object Position MLLM-based Task Planner 1 2 3 4 5 Visual State Description Reflection Reasoning Language Plan Executable Plan From left to right, I can see a lime moon at [82, 17, 19], 
a blue moon at [57, 61, 27], and a green moon at [52, 65, 19].

The previous actions indicate some invalid paths, so I need to ensure the gripper moves correctly ...

The task is to ... The lime moon at [82, 17, 19] should be at the bottom, and the gray moon at [57, 61, 27] should be placed on top. 

1. Move to the gray moon's position. 

...

5. Open the gripper to release the gray moon onto the lime moon.

[{"action": "[57, 61, 30, 0, 60, 25, 0]"}, {"action": "[82, 17, 30, 0, 60, 25, 0]"},
{"action": "[82, 17, 26, 0, 60, 25, 1]"}]
EB-ALFRED EB-Habitat **Embodied Environments** EB-Navigation EB-Manipulation Updated Visual Perception Environment Feedback **Interaction History** Output Action History 1. "Last action executed successfully." 2. "At this moment, you have completed executing 2 steps. Last action is invalid. The current reward obtained is 0.0 ..." 
{"action": "[57, 61, 30, 0, 60, 25, 1]"}, {"action": "[57, 61, 27, 0, 60, 25, 0]"}, {"action": ...}, ...]
enhance the simulation, we introduced key improvements, such as support for multiple instances of the same object type, allowing us to cover all task types in ALFRED. Additionally, we streamlined the action space by merging "put down" actions into a single action, since only one object can be held at a time. Due to the varying number of objects in ALFRED, the action space of EB-ALFRED is dynamic, ranging from 171 to 298 actions. Furthermore, we manually corrected simulator errors and refined instruction quality, ensuring more accurate action execution and improved task solvability. These enhancements make EB-ALFRED a highquality benchmark for evaluating embodied agents.

EB-Habitat. EB-Habitat is built upon the Language Rearrangement benchmark (Szot et al., 2023), featuring 282 diverse language instruction templates. It leverages the Habitat 2.0 simulator (Szot et al., 2021) and focuses on planning and executing 70 high-level skills to achieve user-defined goals. These skills fall into five categories: "navigation",
"pick", "place", "open", and "close", with each skill parameterized by a set of objects. Unlike ALFRED, which permits navigation to any object, EB-Habitat restricts navigation to receptacle-type objects, requiring robots to visit multiple locations to find desired items. With its wide variety of language instructions and unique navigation constraints, EB- Habitat serves as a valuable complement to EB-ALFRED. EB-Navigation. EB-Navigation is an evaluation suite based on AI2-THOR (Kolve et al., 2017), designed to assess embodied agents' navigation abilities. Each unique navigation task is primarily defined by: (1) *initial Robot Pose*, (2) target object information, and (3) *language instruction* that specifies which target object to locate, such as "navigate to the laptop". The robot can only rely on visual observations and textual feedback, without direct positioning data, to navigate to the target object. Success is defined as reaching within a specified distance of the target. The action space includes 8 low-level actions: (1) Move forward/backward/left/right by ∆x. (2) Rotate to the right/left by ∆θ degrees. (3) Tilt the camera upward/downward by ∆φ degrees. The environment provides textual feedback on action validity, such as collision detection. Additionally, we offer a script for automatic task generation, allowing users to create custom task datasets by specifying the configuration. EB-Manipulation. EB-Manipulation extends VLMBench (Zheng et al., 2022) to evaluate MLLM-based embodied agents in low-level object manipulation. The agent controls a robotic arm using a 7-dimensional action vector, specifying movement parameters. Direct low-level manipulation is challenging for MLLMs. To overcome this challenge, we implemented enhancements, as illustrated in Figure 2: (1) action space discretization (Yin et al., 2024), which divides the position components (*x, y, z*) into 100 bins and the orientation components (*roll, pitch, yaw*) into 120 bins, enabling valid actions to take forms like [*x, y, z, roll, pitch, yaw, gripper*] =

Replanning **Success**
Step 1 Step 7 Step 11 Step 14 Step 16 Step 17 Step 20 High-level Planning Trajectory Planning **Fail**
Instruction: Place a sauce pan with a spatula on the kitchen counter.

Step 1 Step 2 Step 3 Step 4 Step 5 Step 6 Low-level Planning Trajectory Instruction: Stack the right moon on top of the left star.

Planning **Success**
[57, 61, 20, 10, 60, 25, 1]; and (2) additional information like YOLO (Redmon, 2016) detection boxes with index markers (Yang et al., 2023a) and object pose estimation for indexed objects, reducing the need for precise 3D location.

## 4.2. Capability-Oriented Data Collection

We aim to collect capability-oriented data for our four environments. To accomplish this, we have identified six capability categories, as outlined in Table 5: (1) The **Base** subset evaluates basic task-solving skills necessary for planning action sequences across tasks of low to medium difficulty. (2) The **Common Sense** subset focuses on the use of common sense knowledge to indirectly refer to objects, such as describing a refrigerator as "a receptacle that can keep food fresh for several days." This subset evaluates the ability of embodied agents to reason using common sense. (3)
The **Complex Instruction** subset includes relatively longer contexts, which can be relevant or irrelevant, to obscure the instruction. This measures an agent's ability to discern user intent from a long context. (4) The **Spatial Awareness** subset refers to objects by their location relative to other objects. (5) The **Visual Appearance** subset involves referring to objects based on their visual attributes, such as color or shape. (6) The **Long Horizon** subset comprises tasks requiring extended action sequences, typically more than 15 steps in EB-ALFRED. These subsets cover a broad range of scenarios, enabling a fine-grained evaluation of embodied agents' capabilities. To construct a diverse dataset, we employ different data collection strategies. For EB-ALFRED and EB-Manipulation, data was gathered through a combination of manual annotation and instruction augmentation using GPT-4o (OpenAI, 2024a). For EB-Habitat, we reorganized and adapted an existing dataset from (Szot et al., 2023), aligning it with our specific objectives. Differently, data for EB-Navigation was generated entirely through automated Python programs. In summary, EB-ALFRED and EB-Habitat each include 300 test instances, with 50 instances for 6 subsets. Due to design challenges, EB-Navigation omits the spatial awareness subset and EB-Manipulation excludes the long-horizon subset.

EB-Navigation consists of 300 test cases distributed across 5 subsets (60 instances each), while EB-Manipulation contains a total of 228 instances, with 48 instances for each subset except visual appearance, which includes 36 instances. Detailed data collection is provided in Appendix C.

## 4.3. Vision-Driven Agent Design

To evaluate MLLMs as agents in EMBODIEDBENCH, we design a unified embodied agent pipeline, illustrated in Figure 2. This pipeline provides a robust framework for processing multimodal inputs, reasoning through interactions, and generating structured, executable plans composed of sequential actions. Two planning examples are provided in Figure 3, with additional examples available in Appendix J. Below, we outline the key components of our agent design.

Agent Input: The agent processes a variety of inputs, including language instructions, visual perceptions, in-context demonstrations, interaction history, and task-specific information. For visual perception, the agent can utilize either the current step image or a sequence of historical images within a sliding window. However, we observe that current MLLMs struggle to understand multiple historical images effectively, so we primarily rely on the current step image for efficiency. An exception is made for EB-Navigation, which is discussed in more detail in Appendix G. Task-specific information varies by task type. For high-level tasks and EB-Navigation, the agent requires valid skill sets, while EB-Manipulation includes descriptions of the action format. Additionally, EB-Manipulation incorporates detection boxes with visual markers and object positions to help MLLMs accurately identify 3D locations. More examples of input prompts are provided in Appendix I. Task Planner: At each planning step, the agent: (1) generates a textual description of the current visual input; (2) reflects on past actions and environmental feedback; (3)
reasons about how to achieve the goal using available information; (4) formulates a language-based plan; and (5) converts it into an executable plan in the required format.

All outputs are structured in JSON. Unlike prior work planning one action per timestep (Liu et al., 2024e), we support multi-step planning, allowing the agent to dynamically decide the number of actions needed. It offers two advantages: (1) better alignment with in-context examples for sequential decision-making, and (2) reduced plan redundancy, especially in low-level tasks where single action causes limited changes in images, thereby minimizing MLLM API calls. If a plan fails or triggers an invalid action, the agent restarts planning from the latest state.

## 5. Experiments

In this section, we conduct comprehensive experiments to evaluate the performance of various MLLMs in EMBODIED-
BENCH, followed by ablation studies in Sections 5.3 and 5.4 and error analysis in Section 5.5.

## 5.1. Experimental Setups

We benchmark 24 models, including 8 leading proprietary models and 16 SOTA open-source models. The proprietary models include GPT-4o and GPT-4o-mini (OpenAI, 2024a;b), Claude-3.5-Sonnet and Claude-3.7-Sonnet (Anthropic, 2024), Gemini Pro and Gemini Flash (Team et al.,
2024a; DeepMind, 2024), and Qwen-VL-Max (Bai et al., 2023). The open-source models include InternVL2.5 and InternVL3 (8B / 38B / 78B) (Chen et al., 2025; Zhu et al., 2025), Qwen2-VL and Qwen2.5-VL (7B / 72B) (Wang et al., 2024; Bai et al., 2025), Gemma-3 (12B / 27B) (Team et al., 2025), Ovis2 (16B / 34B) (Lu et al., 2024), and LLaMA3.2 Vision Instruct (11B / 90B) (Meta, 2024). For consistency, all models are set with a temperature of 0 and a maximum completion token length of 2048. All images are standardized to a resolution of 500×500 pixels. The maximum number of environment steps is 30 for high-level tasks, 20 for EB-Navigation, and 15 for EB-Manipulation. We use the task success rate as the primary metric in our experiments. More results and ablations are deferred to Appendix F.

## 5.2. Benchmark Results

Overall Results. Tables 2 and 3 summarize the results for high-level and low-level tasks, respectively. Overall, **current MLLMs demonstrate strong performance on**
high-level tasks but struggle with low-level tasks, especially EB-Manipulation. Among **proprietary models**, we observe that different models excel at different task levels: Claude-3.5-Sonnet achieves the highest average accuracy on high-level tasks, with 64.0% on EB-ALFRED and 68.0% on EB-Habitat, while GPT-4o leads in low-level tasks, scoring 57.7% on EB-Navigation and 28.9% on EB- Manipulation. For **open-source models**, InternVL3-78B delivers the strongest overall performance, surpassing several proprietary models and closely matching GPT-4o on low-level tasks with 53.7% on EB-Navigation and 26.3% on EB-Manipulation. Additionally, open-source models exhibit a clear scaling trend, with performance improving as model size increases. Nevertheless, a substantial performance gap remains between the top proprietary and open-source models, particularly on high-level tasks that demand advanced reasoning capabilities. The Role of Vision in Embodied Agent. By comparing the performance of embodied agents with and without visual information (marked as "Lang") in Tables 2 and 3, we observe a clear distinction between low-level and high-level tasks. *Low-level tasks show a much stronger reliance on* vision compared to high-level tasks. For example, disabling vision causes GPT-4o's EB-Navigation performance to drop sharply from 57.7% to 17.4%, with long-horizon planning completely collapsing to 0%. This sharp decline highlights the critical importance of visual signals for lowlevel control tasks. Conversely, high-level tasks show much less dependence on visual input. GPT-4o (Lang) and GPT- 4o-mini (Lang) perform on par with or even outperform their vision-enabled counterparts in EB-ALFRED and EB- Habitat, suggesting that these tasks may rely more heavily on textual information rather than visual input. We will further investigate the impact of language-centric factors in Section 5.3. These findings emphasize two key insights: (1)
when designing MLLM-based embodied AI benchmarks, it is essential to consider action-level taxonomy, with greater attention to low-level action tasks, and (2) more advanced methods are needed to effectively leverage visual input for high-level embodied tasks. Fine-grained Results across Subsets. We have the following findings based on our evaluation across 6 subsets. (1) Performance Varies across Different Subsets. We observe that models perform differently across various subsets. For instance, while Claude-3.5-Sonnet is the best model on EB-Habitat overall, GPT-4o surpasses it on long-horizon subsets (64% vs. 58%). This divergence is even more evident in low-level tasks. In EB-Manipulation, for example, Claude-3.5-Sonnet scores 14.6 and 5.6 points higher than GPT-4o on the complex instruction and visual appearance subsets, respectively, but falls significantly behind on other capabilities. These results highlight the importance of fine-grained evaluations to uncover nuanced limitations in current models. (2) Long-Horizon Planning Is the Most Challenging Task. The long-horizon subset consistently proves to be the most difficult, showing the largest performance gap compared to base scores. For instance, in EB-Habitat, Claude-3.5-Sonnet achieves 96% on the base subset but drops to 58% on the long-horizon subset. Similarly, GPT-4o falls from 86% to 64%. This trend holds true across both high-level and lowlevel tasks, suggesting that long-horizon planning remains a significant bottleneck for current MLLM-based agents.

Table 2. Task success rates on 6 subsets of EB-ALFRED and EB-Habitat, with the best proprietary model in bold and open-source model underlines per column. Success rates for subsets are integers since each subset consists of 50 test instances.

Model EB-ALFRED **EB-Habitat**

Avg Base Common Complex Visual Spatial Long Avg Base Common Complex **Visual Spatial Long**

Proprietary MLLMs

GPT-4o 56.3 64 54 68 46 52 54 59.0 86 44 56 68 36 64

GPT-4o-mini 24.0 34 28 36 24 22 0 32.7 74 22 32 22 32 14

Claude-3.7-Sonnet 67.7 68 68 70 **68 62 70** 58.7 90 58 58 62 38 46 Claude-3.5-Sonnet 64.0 72 66 76 60 58 52 **68.0 96 68 78** 70 38 58

Gemini-1.5-Pro 62.3 70 64 72 58 52 58 56.3 92 52 48 56 38 52

Gemini-2.0-flash 52.3 62 48 54 46 46 58 42.3 82 38 38 36 34 26 Gemini-1.5-flash 39.3 44 40 56 42 26 28 39.3 76 32 48 36 32 12

Qwen-VL-Max 41.3 44 48 44 42 38 32 45.3 74 40 50 42 30 36 GPT-4o (Lang) 58.0 62 64 70 52 46 54 56.0 82 52 58 74 34 36

GPT-4o-mini (Lang) 31.3 42 36 46 30 20 14 36.7 82 30 34 30 30 14

Open-Source MLLMs

Llama-3.2-90B-Vision-Ins 32.0 38 34 44 28 32 16 40.3 94 24 50 32 28 14 Llama-3.2-11B-Vision-Ins 13.7 24 8 16 22 6 6 25.0 70 16 28 10 20 6

InternVL2 5-78B 37.7 38 34 42 34 36 42 49.0 80 42 56 58 30 28 InternVL2 5-38B 23.3 36 30 36 22 14 26 38.3 60 28 48 34 32 28

InternVL2 5-8B 2.0 4 6 2 0 0 0 11.3 36 4 0 10 16 2

InternVL3-78B 39.0 38 34 46 42 38 36 55.0 84 58 60 56 32 40 InternVL3-38B 38.0 42 34 48 30 30 44 43.3 80 26 52 40 30 32

InternVL3-8B 10.3 20 14 14 12 0 2 24.3 60 14 24 18 20 10

Qwen2-VL-72B-Ins 33.7 40 30 40 30 32 30 35.7 70 30 36 32 28 18

Qwen2-VL-7B-Ins 1.7 6 0 2 0 0 2 18.3 48 6 16 20 18 2

Qwen2.5-VL-72B-Ins 39.7 50 42 42 36 34 34 37.7 74 28 42 40 24 18

Qwen2.5-VL-7B-Ins 4.7 10 8 6 2 0 2 14.3 32 2 26 10 14 2

Ovis2-34B 28.7 34 30 38 28 18 24 37.0 68 34 38 38 30 14 Ovis2-16B 16.3 26 16 24 12 16 4 32.0 66 26 42 28 22 8

gemma-3-27b-it 37.0 42 40 48 30 36 26 35.7 68 26 30 40 28 22 gemma-3-12b-it 25.7 32 26 38 26 20 12 23.0 58 10 24 18 24 4

(a) Different Camera Resolutions (b) Detection Boxes
(c) Multi-step Images (d) Visual In-context Learning

## 5.3. Language-Centric Ablation

We explore the role of the language-centric components, specifically focusing on **environment feedback** and the number of in-context examples. Comparisons are conducted using the base subset of EB-ALFRED. Our findings in Figure 4 reveal that removing environment feedback—which provides critical information during interaction—causes a 10% drop in success rate for GPT-4o and an 8% drop for Claude-3.5-Sonnet. Furthermore, while our experiments use 10 in-context examples by default, reducing this number significantly affects performance. In a 0-shot setting, the success rate drops to around 40%. When compared with results in Table 2, where removing vision can even lead to performance gains, these findings highlight that high-level tasks rely more heavily on textual information than on visual input.

## 5.4. Visual-Centric Ablation

Visual information is critical for the performance of lowlevel tasks. In this section, we thoroughly analyze the impact of four factors or potential enhancements: camera resolution, detection boxes, multi-step images, and visual in-context

| model underlines per column. Model   | EB-Navigation   | EB-Manipulation       |      |      |      |                       |         |      |      |      |      |      |
|--------------------------------------|-----------------|-----------------------|------|------|------|-----------------------|---------|------|------|------|------|------|
| Avg                                  | Base            | Common Complex Visual | Long | Avg  | Base | Common Complex Visual | Spatial |      |      |      |      |      |
| Proprietary MLLMs                    |                 |                       |      |      |      |                       |         |      |      |      |      |      |
| GPT-4o                               | 57.7            | 55.0                  | 60.0 | 58.3 | 60.0 | 55.0                  | 28.9    | 39.6 | 29.2 | 29.2 | 19.4 | 25.0 |
| GPT-4o-mini                          | 32.8            | 31.7                  | 33.3 | 35.0 | 28.3 | 33.3                  | 4.8     | 4.2  | 6.3  | 2.1  | 0.0  | 10.4 |
| Claude-3.7-Sonnet                    | 45.0            | 50.0                  | 61.7 | 50.0 | 36.7 | 26.7                  | 28.5    | 31.3 | 20.8 | 43.8 | 25.0 | 20.8 |
| Claude-3.5-Sonnet                    | 44.7            | 66.7                  | 51.7 | 41.7 | 36.7 | 26.7                  | 25.4    | 37.5 | 16.7 | 29.2 | 19.4 | 22.9 |
| Gemini-1.5-Pro                       | 24.3            | 23.3                  | 25.0 | 25.0 | 28.3 | 20.0                  | 21.1    | 14.6 | 14.6 | 22.9 | 16.7 | 35.4 |
| Gemini-2.0-flash                     | 48.7            | 63.3                  | 65.0 | 50.0 | 51.7 | 13.3                  | 16.7    | 14.6 | 8.3  | 14.6 | 13.9 | 31.3 |
| Gemini-1.5-flash                     | 41.7            | 56.7                  | 50.0 | 46.7 | 50.0 | 5.0                   | 9.6     | 14.6 | 10.4 | 4.2  | 8.3  | 10.4 |
| Qwen-VL-Max                          | 39.7            | 50.0                  | 46.7 | 41.7 | 35.0 | 25.0                  | 18.0    | 25.0 | 10.4 | 18.8 | 2.8  | 29.2 |
| GPT-4o (Lang)                        | 17.4            | 21.7                  | 21.7 | 26.7 | 16.7 | 0.0                   | 16.2    | 16.7 | 16.7 | 14.6 | 19.4 | 14.6 |
| GPT-4o-mini (Lang)                   | 8.3             | 3.3                   | 13.3 | 10.0 | 15.0 | 0.0                   | 6.6     | 12.5 | 0.0  | 2.1  | 2.8  | 14.6 |
| Open-Source MLLMs                    |                 |                       |      |      |      |                       |         |      |      |      |      |      |
| Llama-3.2-90B-Vision-Ins             | 30.0            | 48.3                  | 23.3 | 38.3 | 33.3 | 6.7                   | 14.9    | 10.4 | 12.5 | 16.7 | 10.4 | 20.8 |
| Llama-3.2-11B-Vision-Ins             | 21.4            | 23.3                  | 21.7 | 26.7 | 18.3 | 17.0                  | 0.9     | 0.0  | 0.0  | 2.1  | 0.0  | 2.1  |
| InternVL2 5-78B                      | 30.7            | 36.7                  | 38.3 | 33.3 | 21.7 | 23.3                  | 18.0    | 16.7 | 16.7 | 14.6 | 22.2 | 20.8 |
| InternVL2 5-38B                      | 30.3            | 35.0                  | 28.3 | 38.3 | 26.7 | 23.3                  | 15.8    | 22.9 | 16.7 | 8.3  | 13.9 | 16.7 |
| InternVL2 5-8B                       | 21.3            | 35.0                  | 23.3 | 21.7 | 26.7 | 0.0                   | 7.0     | 8.3  | 2.1  | 6.3  | 8.3  | 10.4 |
| InternVL3-78B                        | 53.7            | 66.7                  | 63.3 | 61.7 | 45.0 | 31.7                  | 26.3    | 29.2 | 22.9 | 22.9 | 25.0 | 31.3 |
| InternVL3-38B                        | 50.7            | 55.0                  | 61.7 | 55.0 | 56.7 | 25.0                  | 22.6    | 20.8 | 14.6 | 20.8 | 19.4 | 37.5 |
| InternVL3-8B                         | 29.3            | 38.3                  | 30.0 | 40.0 | 33.3 | 5.0                   | 11.5    | 10.4 | 10.4 | 12.5 | 13.9 | 10.4 |
| Qwen2-VL-72B-Ins                     | 21.2            | 26.7                  | 30.0 | 28.3 | 16.0 | 5.0                   | 13.6    | 18.8 | 20.8 | 4.2  | 8.3  | 14.6 |
| Qwen2-VL-7B-Ins                      | 14.0            | 26.7                  | 10.0 | 15.0 | 15.0 | 3.3                   | 0.0     | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |
| Qwen2.5-VL-72B-Ins                   | 40.0            | 46.7                  | 46.7 | 46.7 | 26.7 | 33.3                  | 16.2    | 12.5 | 12.5 | 16.7 | 22.2 | 18.8 |
| Qwen2.5-VL-7B-Ins                    | 20.3            | 20.0                  | 26.7 | 38.3 | 16.7 | 0.0                   | 9.6     | 8.3  | 8.3  | 8.3  | 5.6  | 16.7 |
| Ovis2-34B                            | 45.7            | 63.3                  | 50.0 | 56.7 | 46.7 | 11.7                  | 26.8    | 31.3 | 25.0 | 18.8 | 27.8 | 31.3 |
| Ovis2-16B                            | 47.7            | 60.0                  | 46.7 | 58.3 | 48.3 | 25.0                  | 11.3    | 10.4 | 4.2  | 16.7 | 16.7 | 8.3  |
| gemma-3-27b-it                       | 45.4            | 53.3                  | 45.0 | 61.7 | 50.0 | 16.7                  | 17.5    | 25.0 | 16.7 | 16.7 | 8.3  | 20.8 |
| gemma-3-12b-it                       | 34.0            | 38.3                  | 36.7 | 48.3 | 40.0 | 6.7                   | 20.6    | 20.8 | 22.9 | 20.8 | 19.4 | 18.8 |

their important role in object localization for low-level tasks.

learning. All comparisons are based on the base subset of EB-Manipulation. Additional ablation results can be found in Appendix F.

Camera Resolutions. We investigate the effect of three camera resolutions on task performance. Our results, shown in Figure 5 (a), indicate that mid-range resolutions
(500 × 500) achieve better results compared to both lower (300 × 300) and higher (700 × 700) resolutions. While lowresolution images may lack fine-grained details necessary for task execution, excessively high resolutions can introduce unnecessary complexity, making it harder for MLLMs to focus on relevant information for decision-making. These results highlight the importance of selecting an appropriate resolution when deploying MLLM-based embodied agents. Detection Boxes. In EB-Manipulation, detection boxes and visual markers are used to align language instructions with visual information, helping to localize key objects in the scene. Figure 5 (b) shows that removing detection boxes reduces success rates from 39.6% to 27.1% for GPT-4o and from 37.5% to 29.2% for Claude-3.5-Sonnet, emphasizing Multi-step Image Input. We also explore whether incorporating multi-step historical observations can enhance performance in our agent framework, as they may help address partial observability. For EB-Manipulation, we include observations from the past two steps in addition to the current step. Two multi-step image examples are shown in Figure 10 and 11. Figure 5 (c) presents the quantitative results. Our experiments reveal that current MLLMs struggle to effectively utilize multiple image inputs, often leading to confusion about their current state. Future work could focus on developing methods to better leverage multiple images for enhanced understanding and reasoning. Visual In-context Learning (ICL). Previous work has primarily relied on text-based ICL demonstrations. In this study, we investigate the impact of visual ICL for embodied agents by including image observations as part of the in-context examples for EB-Manipulation. This approach helps the model better understand the relationship between successful low-level actions and the object positions in the

hallucination 4%
wrong recog.

22%
reasoning error 41%
planning error 55%
wrong recog.

3% insufficient exploration 12%
wrong termination decision 13%
invalid action 1%
action id mismatch 1%
invalid action 22%
perception error 4%
perception error 33%
reasoning error 23%
inaccurate action 42%
planning error 44%
spatial understanding 8%
spatial reasoning 10%
reflection error 13%
reflection error 17% inaccurate action 9%
missing step 23%
(a) EB-ALFRED
(b) EB-Manipulation
image. Visual ICL examples are demonstrated in Figure 15. We limit the number of examples to two to avoid overwhelming the model with excessive visual input. This may slightly lower the baseline performance, as the main results use more than two text-based examples. As shown in Figure 5 (d), the results demonstrate that visual ICL significantly outperforms language-only ICL. For instance, Claude-3.5- Sonnet achieves a 16.7% performance boost. These findings underscore the potential of visual ICL as a promising avenue for future research in embodied agents.

## 5.5. Error Analysis

We conducted an error analysis on GPT-4o to identify potential failure modes in EB-ALFRED and EB-Manipulation.

For each environment, we sample 10 failure episodes from each subset, resulting in a total of 110 failed episodes to be analyzed. We found three main types of errors: perception errors, reasoning errors, and planning errors. Each error category corresponds to a specific stage in our agent pipeline, with definitions of sub-errors provided in Appendix H. Overall, planning errors are the most common issue in both environments, while perception errors are more prevalent in low-level tasks. In EB-ALFRED, planning errors (55%) and reasoning errors (41%) dominate, while only 4% of errors are perception errors. Among planning errors, missing steps (23%) and invalid actions (22%) are the most common issues, highlighting challenges in generating complete and valid plans. Reflection errors (17%) suggest the model often fails to recognize planning mistakes in its action history. Another common failure is wrong termination errors (13%), where the model prematurely assumes the task is complete and stops too early. For EB-Manipulation, planning errors remain the primary cause of failure (44%), due to inaccurate actions, indicating difficulties in estimating precise gripper poses. Perception errors make up 33% of failures, with wrong recognition errors (22%) being the most frequent.

These errors show that even with detection boxes annotated in the visual input, the model still fails to recognize object attributes correctly. This highlights considerable room for improvement in the visual capabilities of GPT-4o.

## 6. Conclusion

We introduce EMBODIEDBENCH, a comprehensive evaluation framework designed to assess MLLM-based embodied agents across tasks with varying action levels and capabilityoriented subsets. Through extensive experiments, we identified key challenges, including difficulties in low-level manipulation and long-horizon planning, and the varying significance of vision input across tasks. By highlighting these areas for improvement, we hope EMBODIEDBENCH will inspire and guide future research toward building more capable and versatile vision-driven embodied agents.

## Limitations

A key limitation of this work is that our evaluation is conducted solely in simulated environments, without real-world experiments. This reflects a common trade-off between reproducibility, cost, safety, and real-world applicability. While real-world testing is essential for practical deployment, simulated benchmarks offer a standardized and reproducible setting, significantly reducing time, financial costs, and safety risks (Li et al., 2024c; Liu et al., 2024e).

EMBODIEDBENCH represents a step forward in evaluating MLLM agents across diverse simulated embodied tasks. Future work could explore more realistic and complex simulations (Li et al., 2023) or develop standardized, cost-effective real-world test suites (Zhao et al., 2023; Fu et al., 2024) to bridge the gap toward practical deployment.

## Impact Statement

This work aims to advance the development of vision-driven embodied agents. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here. In addition, **a discussion of**
possible future directions is provided in Appendix B.

## Acknowledgement

Tong Zhang acknowledges the support of NSF grant No.

2416897. Huan Zhang acknowledges the support of the University Research Program (URP) from Toyota Research Institute (TRI). This research reflects solely the opinions and conclusions of its authors, not those of TRI or any other Toyota entity. This research is also based upon work supported by U.S. DARPA ITM Program No. FA8650-23-C-7316 and DARPA ECOLE Program No. \#HR00112390060. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of DARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.

## References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I.,
Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. *arXiv preprint* arXiv:2303.08774, 2023.

Agarwal, N., Ali, A., Bala, M., Balaji, Y., Barker, E., Cai, T., Chattopadhyay, P., Chen, Y., Cui, Y., Ding, Y., et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.

Ahn, M., Brohan, A., Brown, N., Chebotar, Y., Cortes, O.,
David, B., Finn, C., Fu, C., Gopalakrishnan, K., Hausman, K., et al. Do as i can, not as i say: Grounding language in robotic affordances. *arXiv preprint arXiv:2204.01691*, 2022.

Ajay, A., Han, S., Du, Y., Li, S., Gupta, A., Jaakkola, T., Tenenbaum, J., Kaelbling, L., Srivastava, A., and Agrawal, P. Compositional foundation models for hierarchical planning. *Advances in Neural Information* Processing Systems, 36:22304–22325, 2023.

Anthropic. Claude 3.5 sonnet, 2024. URL
https://www.anthropic.com/news/claude3-5-sonnet.

Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J.,
Zhou, C., and Zhou, J. Qwen-vl: A frontier large visionlanguage model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023.

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y.,
Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., and Lin, J. Qwen2.5-vl technical report, 2025.

Belkhale, S., Ding, T., Xiao, T., Sermanet, P., Vuong, Q.,
Tompson, J., Chebotar, Y., Dwibedi, D., and Sadigh, D.

Rt-h: Action hierarchies using language. arXiv preprint arXiv:2403.01823, 2024.

Bommasani, R., Hudson, D. A., Adeli, E., Altman, R.,
Arora, S., von Arx, S., Bernstein, M. S., Bohg, J., Bosselut, A., Brunskill, E., et al. On the opportunities and risks of foundation models. *arXiv preprint arXiv:2108.07258*, 2021.

Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Dabis, J.,
Finn, C., Gopalakrishnan, K., Hausman, K., Herzog, A., Hsu, J., et al. Rt-1: Robotics transformer for real-world control at scale. *arXiv preprint arXiv:2212.06817*, 2022.

Brohan, A., Brown, N., Carbajal, J., Chebotar, Y., Chen, X., Choromanski, K., Ding, T., Driess, D., Dubey, A., Finn, C., et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. *arXiv preprint* arXiv:2307.15818, 2023.

Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M.,
Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language models are few-shot learners. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS'20, Red Hook, NY, USA, 2020. Curran Associates Inc. ISBN 9781713829546.

Calli, B., Singh, A., Walsman, A., Srinivasa, S., Abbeel, P.,
and Dollar, A. M. The ycb object and model set: Towards common benchmarks for manipulation research. In 2015 international conference on advanced robotics (ICAR),
pp. 510–517. IEEE, 2015.

Chang, M., Chhablani, G., Clegg, A., Cote, M. D., Desai, R., Hlavac, M., Karashchuk, V., Krantz, J., Mottaghi, R., Parashar, P., et al. Partnr: A benchmark for planning and reasoning in embodied multi-agent tasks. arXiv preprint arXiv:2411.00081, 2024.

Chattopadhyay, P., Hoffman, J., Mottaghi, R., and Kembhavi, A. Robustnav: Towards benchmarking robustness in embodied navigation. In *Proceedings of the IEEE/CVF* International Conference on Computer Vision, pp. 15691– 15700, 2021.

Chen, B., Xu, Z., Kirmani, S., Ichter, B., Sadigh, D., Guibas, L., and Xia, F. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14455–14465, 2024a.

Chen, Y., Cui, W., Chen, Y., Tan, M., Zhang, X., Zhao, D.,
and Wang, H. Robogpt: an intelligent agent of making embodied long-term decisions for daily instruction tasks.

arXiv preprint arXiv:2311.15649, 2023a.

Chen, Y., Wang, X., Li, M., Hoiem, D., and Ji, H. Vistruct:
Visual structural knowledge extraction via curriculum guided code-vision representation. In Proc. The 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP2023), 2023b.

Chen, Y., Wang, X., Peng, H., and Ji, H. Solo: A single transformer for scalable vision-language modeling. In Transactions on Machine Learning Research, 2024b.

Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S.,
Zhong, M., Zhang, Q., Zhu, X., Lu, L., Li, B., Luo, P., Lu, T., Qiao, Y., and Dai, J. Internvl: Scaling up vision foundation models and aligning for generic visuallinguistic tasks. *arXiv preprint arXiv:2312.14238*, 2023c.

Chen, Z., Wang, W., Cao, Y., Liu, Y., Gao, Z., Cui, E., Zhu, J., Ye, S., Tian, H., Liu, Z., Gu, L., Wang, X., Li, Q., Ren, Y., Chen, Z., Luo, J., Wang, J., Jiang, T., Wang, B., He, C., Shi, B., Zhang, X., Lv, H., Wang, Y., Shao, W., Chu, P., Tu, Z., He, T., Wu, Z., Deng, H., Ge, J., Chen, K., Zhang, K., Wang, L., Dou, M., Lu, L., Zhu, X., Lu, T., Lin, D., Qiao, Y., Dai, J., and Wang, W. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling, 2025. URL https://arxiv.org/abs/2412.05271.

Cheng, A.-C., Yin, H., Fu, Y., Guo, Q., Yang, R., Kautz, J., Wang, X., and Liu, S. Spatialrgpt: Grounded spatial reasoning in vision language model. arXiv preprint arXiv:2406.01584, 2024.

Cheng, Z., Tu, Y., Li, R., Dai, S., Hu, J., Hu, S., Li, J.,
Shi, Y., Yu, T., Chen, W., et al. Embodiedeval: Evaluate multimodal llms as embodied agents. *arXiv preprint* arXiv:2501.11858, 2025.

Chi, C., Xu, Z., Feng, S., Cousineau, E., Du, Y., Burchfiel, B., Tedrake, R., and Song, S. Diffusion policy: Visuomotor policy learning via action diffusion. *The International* Journal of Robotics Research, pp. 02783649241273668, 2023.

Choi, J.-W., Yoon, Y., Ong, H., Kim, J., and Jang, M. Lotabench: Benchmarking language-oriented task planners for embodied agents. *arXiv preprint arXiv:2402.08178*, 2024.

Contributors, L. Lmdeploy: A toolkit for compressing, deploying, and serving llm. https://github.com/ InternLM/lmdeploy, 2023.

DeepMind, G. Introducing gemini 2.0: our new ai model for the agentic era, 2024. URL https://blog.google/
technology/google-deepmind/googlegemini-ai-update-december-2024/.

Driess, D., Xia, F., Sajjadi, M. S., Lynch, C., Chowdhery, A., Ichter, B., Wahid, A., Tompson, J., Vuong, Q., Yu, T., et al. Palm-e: an embodied multimodal language model.

In Proceedings of the 40th International Conference on Machine Learning, pp. 8469–8488, 2023.

Du, A., Gao, B., Xing, B., Jiang, C., Chen, C., Li, C., Xiao, C., Du, C., Liao, C., Tang, C., Wang, C., Zhang, D., Yuan, E., Lu, E., Tang, F., Sung, F., Wei, G., Lai, G., Guo, H.,
Zhu, H., et al. Kimi k1.5: Scaling reinforcement learning with llms. *arXiv preprint arXiv:2501.12599*, 2025.

Du, Y., Yang, M., Florence, P., Xia, F., Wahid, A.,
Ichter, B., Sermanet, P., Yu, T., Abbeel, P., Tenenbaum, J. B., et al. Video language planning. arXiv preprint arXiv:2310.10625, 2023.

Durante, Z., Huang, Q., Wake, N., Gong, R., Park, J. S.,
Sarkar, B., Taori, R., Noda, Y., Terzopoulos, D., Choi, Y., et al. Agent ai: Surveying the horizons of multimodal interaction. *arXiv preprint arXiv:2401.03568*, 2024.

Fu, Z., Zhao, T. Z., and Finn, C. Mobile aloha: Learning bimanual mobile manipulation with low-cost whole-body teleoperation. *arXiv preprint arXiv:2401.02117*, 2024.

Gao, C., Zhao, B., Zhang, W., Mao, J., Zhang, J., Zheng, Z.,
Man, F., Fang, J., Zhou, Z., Cui, J., et al. Embodiedcity: A benchmark platform for embodied agent in real-world city environment. *arXiv preprint arXiv:2410.09604*, 2024a.

Gao, J., Sarkar, B., Xia, F., Xiao, T., Wu, J., Ichter, B., Majumdar, A., and Sadigh, D. Physically grounded visionlanguage models for robotic manipulation. In *2024 IEEE* International Conference on Robotics and Automation (ICRA), pp. 12462–12469. IEEE, 2024b.

Gu, Q., Kuwajerwala, A., Morin, S., Jatavallabhula, K. M.,
Sen, B., Agarwal, A., Rivera, C., Paul, W., Ellis, K.,
Chellappa, R., et al. Conceptgraphs: Open-vocabulary 3d scene graphs for perception and planning. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 5021–5028. IEEE, 2024.

Gulino, C., Fu, J., Luo, W., Tucker, G., Bronstein, E., Lu, Y., Harb, J., Pan, X., Wang, Y., Chen, X., et al. Waymax: An accelerated, data-driven simulator for large-scale autonomous driving research. Advances in Neural Information Processing Systems, 36, 2024.

Huang, H., Lin, F., Hu, Y., Wang, S., and Gao, Y.

Copa: General robotic manipulation through spatial constraints of parts with foundation models. *arXiv preprint* arXiv:2403.08248, 2024a.

Huang, J., Yong, S., Ma, X., Linghu, X., Li, P., Wang, Y., Li, Q., Zhu, S.-C., Jia, B., and Huang, S. An embodied generalist agent in 3d world. *arXiv preprint arXiv:2311.12871*, 2023a.

Huang, S., Jiang, Z., Dong, H., Qiao, Y., Gao, P., and Li, H. Instruct2act: Mapping multi-modality instructions to robotic actions with large language model. *arXiv preprint* arXiv:2305.11176, 2023b.

Huang, W., Abbeel, P., Pathak, D., and Mordatch, I. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. In International Conference on Machine Learning, pp. 9118–9147.

PMLR, 2022a.

Huang, W., Xia, F., Xiao, T., Chan, H., Liang, J., Florence, P., Zeng, A., Tompson, J., Mordatch, I., Chebotar, Y., et al. Inner monologue: Embodied reasoning through planning with language models. *arXiv preprint arXiv:2207.05608*, 2022b.

Huang, W., Wang, C., Zhang, R., Li, Y., Wu, J., and Fei-Fei, L. Voxposer: Composable 3d value maps for robotic manipulation with language models. *arXiv preprint* arXiv:2307.05973, 2023c.

Huang, W., Xia, F., Shah, D., Driess, D., Zeng, A., Lu, Y., Florence, P., Mordatch, I., Levine, S., Hausman, K., et al. Grounded decoding: Guiding text generation with grounded models for robot control. arXiv preprint arXiv:2303.00855, 2023d.

Huang, W., Wang, C., Li, Y., Zhang, R., and Fei-Fei, L.

Rekep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation. arXiv preprint arXiv:2409.01652, 2024b.

James, S., Ma, Z., Arrojo, D. R., and Davison, A. J. Rlbench:
The robot learning benchmark & learning environment. IEEE Robotics and Automation Letters, 5(2):3019–3026, 2020.

Jiang, H., Huang, B., Wu, R., Li, Z., Garg, S., Nayyeri, H.,
Wang, S., and Li, Y. Roboexp: Action-conditioned scene graph via interactive exploration for robotic manipulation. arXiv preprint arXiv:2402.15487, 2024.

Khanna, M., Ramrakhya, R., Chhablani, G., Yenamandra, S., Gervet, T., Chang, M., Kira, Z., Chaplot, D. S., Batra, D., and Mottaghi, R. Goat-bench: A benchmark for multi-modal lifelong navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16373–16383, 2024.

Kim, M. J., Pertsch, K., Karamcheti, S., Xiao, T., Balakrishna, A., Nair, S., Rafailov, R., Foster, E., Lam, G., Sanketi, P., et al. Openvla: An open-source vision-languageaction model. *arXiv preprint arXiv:2406.09246*, 2024.

Koh, J. Y., Lo, R., Jang, L., Duvvur, V., Lim, M. C., Huang, P.-Y., Neubig, G., Zhou, S., Salakhutdinov, R., and Fried, D. Visualwebarena: Evaluating multimodal agents on realistic visual web tasks. *arXiv preprint arXiv:2401.13649*,
2024.

Kolve, E., Mottaghi, R., Han, W., VanderBilt, E., Weihs, L.,
Herrasti, A., Deitke, M., Ehsani, K., Gordon, D., Zhu, Y., et al. Ai2-thor: An interactive 3d environment for visual ai. *arXiv preprint arXiv:1712.05474*, 2017.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with pagedattention. In *Proceedings of the ACM SIGOPS* 29th Symposium on Operating Systems Principles, 2023.

Li, C., Xia, F., Mart´ın-Mart´ın, R., Lingelbach, M., Srivastava, S., Shen, B., Vainio, K., Gokmen, C., Dharan, G., Jain, T., et al. igibson 2.0: Object-centric simulation for robot learning of everyday household tasks. arXiv preprint arXiv:2108.03272, 2021.

Li, C., Zhang, R., Wong, J., Gokmen, C., Srivastava, S.,
Mart´ın-Mart´ın, R., Wang, C., Levine, G., Lingelbach, M., Sun, J., et al. Behavior-1k: A benchmark for embodied ai with 1,000 everyday activities and realistic simulation. In Conference on Robot Learning, pp. 80–93. PMLR, 2023.

Li, K., Yu, B., Zheng, Q., Zhan, Y., Zhang, Y., Zhang, T., Yang, Y., Chen, Y., Sun, L., Cao, Q., Shen, L., Li, L., Tao, D., and He, X. Muep: A multimodal benchmark for embodied planning with foundation models. In Larson, K. (ed.), Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24, pp. 129–138. International Joint Conferences on Artificial Intelligence Organization, 8 2024a. doi:
10.24963/ijcai.2024/15. URL https://doi.org/ 10.24963/ijcai.2024/15. Main Track.

Li, M., Zhao, S., Wang, Q., Wang, K., Zhou, Y., Srivastava, S., Gokmen, C., Lee, T., Li, L. E., Zhang, R., et al. Embodied agent interface: Benchmarking llms for embodied decision making. *arXiv preprint arXiv:2410.07166*, 2024b.

Li, X., Hsu, K., Gu, J., Pertsch, K., Mees, O., Walke, H. R.,
Fu, C., Lunawat, I., Sieh, I., Kirmani, S., et al. Evaluating real-world robot manipulation policies in simulation. arXiv preprint arXiv:2405.05941, 2024c.

Liang, J., Huang, W., Xia, F., Xu, P., Hausman, K., Ichter, B.,
Florence, P., and Zeng, A. Code as policies: Language model programs for embodied control. In *2023 IEEE* International Conference on Robotics and Automation (ICRA), pp. 9493–9500. IEEE, 2023.

Liu, H., Li, C., Li, Y., Li, B., Zhang, Y., Shen, S.,
and Lee, Y. J. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024a. URL
https://llava-vl.github.io/blog/202401-30-llava-next/.

Liu, J., Li, S., Wang, Z., Li, M., and Ji, H. A language first approach for procedural planning. In *Proc. The 61st* Annual Meeting of the Association for Computational Linguistics (ACL2023) Findings, 2023a.

Liu, S., Chen, J., Ruan, S., Su, H., and Yin, Z. Exploring the robustness of decision-level through adversarial attacks on llm-based embodied models. In *Proceedings of the* 32nd ACM International Conference on Multimedia, pp. 8120–8128, 2024b.

Liu, S., Wu, L., Li, B., Tan, H., Chen, H., Wang, Z., Xu, K., Su, H., and Zhu, J. Rdt-1b: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864, 2024c.

Liu, S., Ren, Z., Gupta, S., and Wang, S. Physgen: Rigidbody physics-grounded image-to-video generation. In European Conference on Computer Vision, pp. 360–378. Springer, 2025.

Liu, X., Yu, H., Zhang, H., Xu, Y., Lei, X., Lai, H., Gu, Y.,
Ding, H., Men, K., Yang, K., et al. Agentbench: Evaluating llms as agents. *arXiv preprint arXiv:2308.03688*, 2023b.

Liu, X., Guo, D., Zhang, X., and Liu, H. Heterogeneous embodied multi-agent collaboration. *IEEE Robotics and* Automation Letters, 2024d.

Liu, X., Zhang, T., Gu, Y., Iong, I. L., Xu, Y., Song, X.,
Zhang, S., Lai, H., Liu, X., Zhao, H., et al. Visualagentbench: Towards large multimodal models as visual foundation agents. *arXiv preprint arXiv:2408.06327*, 2024e.

Lu, S., Li, Y., Chen, Q.-G., Xu, Z., Luo, W., Zhang, K.,
and Ye, H.-J. Ovis: Structural embedding alignment for multimodal large language model. arXiv preprint arXiv:2405.20797, 2024.

Luo, J., Xu, C., Liu, F., Tan, L., Lin, Z., Wu, J., Abbeel, P.,
and Levine, S. Fmb: a functional manipulation benchmark for generalizable robotic learning. The International Journal of Robotics Research, pp. 02783649241276017, 2023.

Ma, Y., Cui, C., Cao, X., Ye, W., Liu, P., Lu, J., Abdelraouf, A., Gupta, R., Han, K., Bera, A., et al. Lampilot: An open benchmark dataset for autonomous driving with language model programs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15141–15151, 2024a.

Ma, Y., Song, Z., Zhuang, Y., Hao, J., and King, I. A survey on vision-language-action models for embodied ai. *arXiv* preprint arXiv:2405.14093, 2024b.

Madan, N., Møgelmose, A., Modi, R., Rawat, Y. S., and Moeslund, T. B. Foundation models for video understanding: A survey. *arXiv preprint arXiv:2405.03770*,
2024.

Mao, J., Qian, Y., Zhao, H., and Wang, Y. Gpt-driver: Learning to drive with gpt. *arXiv preprint arXiv:2310.01415*,
2023.

Mazzaglia, P., Verbelen, T., Dhoedt, B., Courville, A., and Rajeswar, S. Genrl: Multimodal-foundation world models for generalization in embodied agents. arXiv preprint arXiv:2406.18043, 2024.

Meta. Llama 3.2: Revolutionizing edge ai and vision with open, customizable models, 2024. URL https: //ai.*meta*.com/blog/llama-3-2-connect2024-vision-edge-mobile-devices/.

Mu, Y., Zhang, Q., Hu, M., Wang, W., Ding, M., Jin, J., Wang, B., Dai, J., Qiao, Y., and Luo, P. Embodiedgpt: Vision-language pre-training via embodied chain of thought. *Advances in Neural Information Processing* Systems, 36, 2024.

Nasiriany, S., Maddukuri, A., Zhang, L., Parikh, A., Lo, A.,
Joshi, A., Mandlekar, A., and Zhu, Y. Robocasa: Largescale simulation of everyday tasks for generalist robots.

arXiv preprint arXiv:2406.02523, 2024a.

Nasiriany, S., Xia, F., Yu, W., Xiao, T., Liang, J., Dasgupta, I., Xie, A., Driess, D., Wahid, A., Xu, Z., et al. Pivot: Iterative visual prompting elicits actionable knowledge for vlms. *arXiv preprint arXiv:2402.07872*, 2024b.

OpenAI. Hello gpt-4o, 2024a. URL https://
openai.com/index/hello-gpt-4o/.

OpenAI. Gpt-4o mini: advancing cost-efficient intelligence, 2024b. URL https://openai.com/index/
gpt-4o-mini-advancing-cost-efficientintelligence/.

Puig, X., Ra, K., Boben, M., Li, J., Wang, T., Fidler, S.,
and Torralba, A. Virtualhome: Simulating household activities via programs. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp.

8494–8502, 2018.

Qian, C., Han, P., Luo, Q., He, B., Chen, X., Zhang, Y.,
Du, H., Yao, J., Yang, X., Zhang, D., Li, Y., and Ji, H. Escapebench: Pushing language models to think outside the box. In *arxiv*, 2024.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G.,
Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In *International conference on* machine learning, pp. 8748–8763. PMLR, 2021.

Rana, K., Haviland, J., Garg, S., Abou-Chakra, J., Reid, I. D., and Suenderhauf, N. Sayplan: Grounding large language models using 3d scene graphs for scalable task planning. *CoRR*, 2023.

Redmon, J. You only look once: Unified, real-time object detection. In *Proceedings of the IEEE conference on* computer vision and pattern recognition, 2016.

Reid, M., Savinov, N., Teplyashin, D., Lepikhin, D., Lillicrap, T., Alayrac, J.-b., Soricut, R., Lazaridou, A., Firat, O., Schrittwieser, J., et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

Song, C. H., Wu, J., Washington, C., Sadler, B. M., Chao, W.-L., and Su, Y. Llm-planner: Few-shot grounded planning for embodied agents with large language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 2998–3009, 2023.

Rohmer, E., Singh, S. P., and Freese, M. V-rep: A versatile and scalable robot simulation framework. In *2013* IEEE/RSJ international conference on intelligent robots and systems, pp. 1321–1326. IEEE, 2013.

Song, X., Chen, W., Liu, Y., Chen, W., Li, G., and Lin, L. Towards long-horizon vision-language navigation: Platform, benchmark and method. *arXiv preprint* arXiv:2412.09082, 2024.

Sarch, G., Somani, S., Kapoor, R., Tarr, M. J., and Fragkiadaki, K. Helper-x: A unified instructable embodied agent to tackle four interactive vision-language domains with memory-augmented language models. arXiv preprint arXiv:2404.19065, 2024a.

Stone, A., Xiao, T., Lu, Y., Gopalakrishnan, K., Lee, K.-
H., Vuong, Q., Wohlhart, P., Kirmani, S., Zitkovich, B., Xia, F., et al. Open-world object manipulation using pre-trained vision-language models. arXiv preprint arXiv:2303.00905, 2023.

Sarch, G. H., Jang, L., Tarr, M. J., Cohen, W. W., Marino, K., and Fragkiadaki, K. Vlm agents generate their own memories: Distilling experience into embodied programs of thought. In *The Thirty-eighth Annual Conference on* Neural Information Processing Systems, 2024b.

Sun, H. Reinforcement learning in the era of llms:
What is essential? what is needed? an rl perspective on rlhf, prompting, and beyond. *arXiv preprint* arXiv:2310.06147, 2023.

Szot, A., Clegg, A., Undersander, E., Wijmans, E., Zhao, Y., Turner, J., Maestre, N., Mukadam, M., Chaplot, D. S., Maksymets, O., et al. Habitat 2.0: Training home assistants to rearrange their habitat. *Advances in neural* information processing systems, 34:251–266, 2021.

Sharma, S., Huang, H., Shivakumar, K., Chen, L. Y., Hoque, R., Ichter, B., and Goldberg, K. Semantic mechanical search with large vision and language models. arXiv preprint arXiv:2302.12915, 2023.

Shen, B., Xia, F., Li, C., Mart´ın-Mart´ın, R., Fan, L.,
Wang, G., Perez-D'Arpino, C., Buch, S., Srivastava, S., ´
Tchapmi, L., et al. igibson 1.0: A simulation environment for interactive tasks in large realistic scenes. In 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 7520–7527. IEEE, 2021.

Szot, A., Schwarzer, M., Agrawal, H., Mazoure, B., Metcalf, R., Talbott, W., Mackraz, N., Hjelm, R. D., and Toshev, A. T. Large language models as generalizable policies for embodied tasks. In The Twelfth International Conference on Learning Representations, 2023.

Szot, A., Mazoure, B., Attia, O., Timofeev, A., Agrawal, H., Hjelm, D., Gan, Z., Kira, Z., and Toshev, A. From multimodal llms to generalist embodied agents: Methods and lessons. *arXiv preprint arXiv:2412.08442*, 2024.

Shridhar, M., Thomason, J., Gordon, D., Bisk, Y., Han, W., Mottaghi, R., Zettlemoyer, L., and Fox, D. Alfred: A benchmark for interpreting grounded instructions for everyday tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp.

10740–10749, 2020a.

Team, G., Georgiev, P., Lei, V. I., Burnell, R., Bai, L.,
Gulati, A., Tanzer, G., Vincent, D., Pan, Z., Wang, S.,
et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. *arXiv preprint* arXiv:2403.05530, 2024a.

Shridhar, M., Yuan, X., Cote, M.-A., Bisk, Y., Trischler, A.,
and Hausknecht, M. Alfworld: Aligning text and embodied environments for interactive learning. In International Conference on Learning Representations, 2020b.

Team, G., Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., Perrin, S., Matejovicova, T., Rame, A., ´ Riviere, M., et al. Gemma 3 technical report. ` *arXiv* preprint arXiv:2503.19786, 2025.

Shridhar, M., Manuelli, L., and Fox, D. Cliport: What and where pathways for robotic manipulation. In *Conference* on robot learning, pp. 894–906. PMLR, 2022.

Team, O. M., Ghosh, D., Walke, H., Pertsch, K., Black, K., Mees, O., Dasari, S., Hejna, J., Kreiman, T., Xu, C., et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024b.

Singh, I., Blukis, V., Mousavian, A., Goyal, A., Xu, D.,
Tremblay, J., Fox, D., Thomason, J., and Garg, A. Progprompt: Generating situated robot task plans using large language models. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pp. 11523–11530.

IEEE, 2023.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P.,
Bhosale, S., et al. Llama 2: Open foundation and finetuned chat models. *arXiv preprint arXiv:2307.09288*, 2023.

Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., and Anandkumar, A. Voyager: An openended embodied agent with large language models. *arXiv* preprint arXiv:2305.16291, 2023a.

Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., et al. Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution. *arXiv preprint arXiv:2409.12191*, 2024.

Wang, Q., Li, M., Chan, H. P., Huang, L., Hockenmaier, J., Girish, C., and Ji, H. Multimedia generative script learning for task planning. In *Proc. The 61st Annual* Meeting of the Association for Computational Linguistics
(ACL2023) Findings, 2023b.

Wang, Y., Xian, Z., Chen, F., Wang, T.-H., Wang, Y., Fragkiadaki, K., Erickson, Z., Held, D., and Gan, C. Robogen: Towards unleashing infinite data for automated robot learning via generative simulation. arXiv preprint arXiv:2311.01455, 2023c.

Wang, Z., Blume, A., Li, S., Liu, G., Cho, J., Tang, Z.,
Bansal, M., and Ji, H. Paxion: Patching video-language foundation models with action knowledge. In *Proc. 2023* Conference on Neural Information Processing Systems (NeurIPS2023) [Spotlight Paper], 2023d.

Wang, Z., Cai, S., Chen, G., Liu, A., Ma, X., and Liang, Y.

Describe, explain, plan and select: Interactive planning with large language models enables open-world multitask agents. *arXiv preprint arXiv:2302.01560*, 2023e.

Wu, C. H., Shah, R. R., Koh, J. Y., Salakhutdinov, R., Fried, D., and Raghunathan, A. Dissecting adversarial robustness of multimodal lm agents. In *NeurIPS 2024 Workshop* on Open-World Agents, 2024a.

Wu, Z., Chen, X., Pan, Z., Liu, X., Liu, W., Dai, D., Gao, H.,
Ma, Y., Wu, C., Wang, B., et al. Deepseek-vl2: Mixtureof-experts vision-language models for advanced multimodal understanding. *arXiv preprint arXiv:2412.10302*, 2024b.

Xiang, F., Qin, Y., Mo, K., Xia, Y., Zhu, H., Liu, F., Liu, M.,
Jiang, H., Yuan, Y., Wang, H., et al. Sapien: A simulated part-based interactive environment. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11097–11107, 2020.

Xiang, J., Liu, G., Gu, Y., Gao, Q., Ning, Y., Zha, Y., Feng, Z., Tao, T., Hao, S., Shi, Y., et al. Pandora: Towards general world model with natural language actions and video states. *arXiv preprint arXiv:2406.09455*, 2024.

Xiao, T., Chan, H., Sermanet, P., Wahid, A., Brohan, A.,
Hausman, K., Levine, S., and Tompson, J. Robotic skill acquisition via instruction augmentation with visionlanguage models. *arXiv preprint arXiv:2211.11736*, 2022.

Xie, J., Chen, Z., Zhang, R., Wan, X., and Li, G.

Large multimodal agents: A survey. *arXiv preprint* arXiv:2402.15116, 2024.

Xu, J., Yang, R., Luo, F., Fang, M., Wang, B., and Han, L. Robust decision transformer: Tackling data corruption in offline rl via sequence modeling. arXiv preprint arXiv:2407.04285, 2024.

Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., et al. Qwen2. 5 technical report. *arXiv preprint arXiv:2412.15115*, 2024a.

Yang, J., Zhang, H., Li, F., Zou, X., Li, C., and Gao, J.

Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. *arXiv preprint arXiv:2310.11441*, 2023a.

Yang, R., Zhong, H., Xu, J., Zhang, A., Zhang, C., Han, L., and Zhang, T. Towards robust offline reinforcement learning under diverse data corruption. arXiv preprint arXiv:2310.12955, 2023b.

Yang, R., Ding, R., Lin, Y., Zhang, H., and Zhang, T. Regularizing hidden states enables learning generalizable reward model for llms. *arXiv preprint arXiv:2406.10216*, 2024b.

Yang, R., Pan, X., Luo, F., Qiu, S., Zhong, H., Yu, D., and Chen, J. Rewards-in-context: Multi-objective alignment of foundation models with dynamic preference adjustment. *arXiv preprint arXiv:2402.10207*, 2024c.

Yang, Y., Zhou, T., Li, K., Tao, D., Li, L., Shen, L., He, X.,
Jiang, J., and Shi, Y. Embodied multi-modal agent trained by an llm from a parallel textworld. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26275–26285, 2024d.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. R., and Cao, Y. React: Synergizing reasoning and acting in language models. In *The Eleventh International* Conference on Learning Representations, 2023.

Yin, Y., Wang, Z., Sharma, Y., Niu, D., Darrell, T., and Herzig, R. In-context learning enables robot action prediction in llms. *arXiv preprint arXiv:2410.12782*, 2024.

Zawalski, M., Chen, W., Pertsch, K., Mees, O., Finn, C., and Levine, S. Robotic control via embodied chain-of-thought reasoning. *arXiv preprint arXiv:2407.08693*, 2024.

Zhai, S., Bai, H., Lin, Z., Pan, J., Tong, P., Zhou, Y., Suhr, A., Xie, S., LeCun, Y., Ma, Y., et al. Fine-tuning large vision-language models as decision-making agents via reinforcement learning. *Advances in Neural Information* Processing Systems, 37:110935–110971, 2025.

Zhang, S., Xu, Z., Liu, P., Yu, X., Li, Y., Gao, Q., Fei, Z., Yin, Z., Wu, Z., Jiang, Y.-G., et al. Vlabench: A large-scale benchmark for language-conditioned robotics manipulation with long-horizon reasoning tasks. *arXiv* preprint arXiv:2412.18194, 2024a.

Zhang, X., Li, J., Chu, W., Hai, J., Xu, R., Yang, Y., Guan, S., Xu, J., and Cui, P. On the out-of-distribution generalization of multimodal large language models. arXiv preprint arXiv:2402.06599, 2024b.

Zhao, T. Z., Kumar, V., Levine, S., and Finn, C. Learning fine-grained bimanual manipulation with low-cost hardware. *arXiv preprint arXiv:2304.13705*, 2023.

Zheng, K., Chen, X., Jenkins, O. C., and Wang, X.

Vlmbench: A compositional benchmark for vision-andlanguage manipulation. Advances in Neural Information Processing Systems, 35:665–678, 2022.

Zhou, G., Hong, Y., and Wu, Q. Navgpt: Explicit reasoning in vision-and-language navigation with large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 7641–7649, 2024a.

Zhou, Y., Li, X., Wang, Q., and Shen, J. Visual in-context learning for large vision-language models. arXiv preprint arXiv:2402.11574, 2024b.

Zhu, J., Wang, W., Chen, Z., Liu, Z., Ye, S., Gu, L., Duan, Y., Tian, H., Su, W., Shao, J., et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. *arXiv preprint arXiv:2504.10479*, 2025.

Zou, C., Guo, X., Yang, R., Zhang, J., Hu, B., and Zhang, H.

Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models. *arXiv preprint arXiv:2411.00836*, 2024.

## A. Additional Related Works

Foundation models (Bommasani et al., 2021), particularly Large Language Models (LLMs) (Brown et al., 2020; Achiam et al., 2023; Touvron et al., 2023; Yang et al., 2024a;c) and Multi-Modal Large Language Models (MLLMs) (Radford et al., 2021; Team et al., 2024a; Wang et al., 2024; Wu et al., 2024b; Du et al., 2025; Chen et al., 2024b; Xie et al., 2024), fundamentally transform how embodied agents perceive, make decisions, and act in physical and simulated environments. The integration of these models into embodied agents evolves through several key approaches. Initially, Large Language Models (LLMs) are introduced to assist with high-level planning (Ahn et al., 2022; Huang et al., 2022a;b; Rana et al., 2023; Gao et al., 2024b; Huang et al., 2023d; Wang et al., 2023a; Huang et al., 2023b; Liu et al., 2023a; Wang et al., 2023b; Chen et al., 2023a; Huang et al., 2023a; Zhou et al., 2024a). They are also adopted for low-level controls (Mao et al., 2023; Yin et al., 2024). MLLMs are then incorporated for perception tasks such as object attribute identification, visual relation extraction, and action recognition (Xiao et al., 2022; Chen et al., 2023b; Wang et al., 2023d;e; Gao et al., 2024b; Gu et al., 2024). Subsequently, the role of MLLMs extends into policy-making through various approaches. Some works implement MLLMs in an end-to-end manner for direct action generation (Shridhar et al., 2022; Driess et al., 2023; Du et al., 2023; Yang et al., 2024d; Mu et al., 2024). Others enhance policy generation by using MLLMs to create visual markers or generate constraints or guidance with visual masks (Sharma et al., 2023; Stone et al., 2023; Nasiriany et al., 2024b; Huang et al., 2024a; Jiang et al., 2024). A different approach involves prompting MLLMs to generate code for creating policy or value functions (Liang et al., 2023; Huang et al., 2023c; 2024b). Most recently, Vision Language Action Models (VLAs) (Brohan et al., 2022; 2023; Chi et al., 2023; Belkhale et al., 2024; Team et al., 2024b; Liu et al., 2024c; Kim et al., 2024) have emerged as a promising direction. These models typically utilize MLLMs or language-conditioned diffusion models as their foundation and are trained on low-level robotics action data. Another promising direction leverages world models as action simulators (Xiang et al., 2024; Agarwal et al., 2025; Liu et al., 2025). These approaches employ diffusion models conditioned on language inputs to predict future states given actions or task descriptions. In response to the rapid advancements in this field, various simulators (Kolve et al., 2017; Puig et al., 2018; Shridhar et al., 2020a; Xiang et al., 2020; Shen et al., 2021; Li et al., 2021; 2023; Nasiriany et al., 2024a) and evaluation benchmarks (Shridhar et al., 2020b;a; Zheng et al., 2022; Li et al., 2023; Szot et al., 2023; Luo et al., 2023; Li et al., 2024a; Koh et al., 2024; Choi et al., 2024; Khanna et al., 2024; Liu et al., 2024e; Li et al., 2024b; Zhang et al., 2024a; Song et al., 2024) have been developed. However, existing benchmarks exhibit notable limitations. For instance, ALFWorld (Shridhar et al., 2020b), AgentBench (Liu et al., 2023b), Lota-bench (Choi et al., 2024), and Embodied Agent Interface (Li et al., 2024b) lack support for multimodal input evaluation. Furthermore, most benchmarks are narrowly focused on specific domains, particularly high-level household tasks (Shridhar et al., 2020a; Li et al., 2023; Szot et al., 2023), while others, such as VLMbench (Zheng et al., 2022) and GOAT-bench (Khanna et al., 2024), concentrate on low-level control for manipulation and navigation, respectively. Although VisualAgentBench (Liu et al., 2024e) pioneers the evaluation of MLLMs across multiple domains, it is limited to high-level tasks like household and Minecraft, and does not support fine-grained capability assessment. Embodied Agent Interface (Li et al., 2024b) and VLABench (Zhang et al., 2024a) introduce fine-grained evaluation metrics with language model support, but their focus remains primarily on LLMs and VLAs rather than MLLMs. Concurrently, EmbodiedEval (Cheng et al., 2025) proposes a multi-domain benchmark for evaluating MLLMs across navigation, object interaction, social interaction, attribute question answering, and spatial question answering. While it overlaps with our work in navigation and object interaction, it does not include low-level manipulation tasks or capability-oriented evaluation. Moreover, the benchmark is limited in scale, containing only 328 test instances.

## B. Future Research Directions

While EMBODIEDBENCH represents a significant step forward in evaluating MLLM-based embodied agents, several challenges remain, offering rich opportunities for future research. Below, we outline potential research directions:
- *Expanding Task Diversity.* Current benchmarks for MLLM-based embodied agents are still limited in task diversity.

Future research could explore more realistic and complex environments with different action levels, such as autonomous driving (Gulino et al., 2024; Ma et al., 2024a; Gao et al., 2024a), multi-agent collaboration (Liu et al., 2024d), and human-agent interaction (Chang et al., 2024). These scenarios would better assess the agents' adaptability and generalization capabilities in real-world settings.

- *Low-Level Tasks and Spatial Reasoning.* Our findings show that current MLLM-based agents struggle with spatial reasoning and low-level control. Future research could improve these capabilities by better integrating spatial reasoning with low-level action planning, including 3D visual grounding (Chen et al., 2024a; Cheng et al., 2024) and alignment (Ahn et al., 2022; Yang et al., 2024d).

- *Long-Horizon Planning.* Long-horizon planning is still challenging for embodied agents. Future research can study techniques like hierarchical planning (Song et al., 2023; Ajay et al., 2023), memory-augmented methods (Sarch et al.,
2024a), and world models (Mazzaglia et al., 2024) to enhance their ability to plan and execute complex, multi-step tasks more effectively.

- *Multi-step/Multi-view Image Understanding.* Our experiments show that current MLLMs struggle with multi-step and multi-view image inputs. Future research could improve multi-frame and multi-view comprehension, temporal reasoning, and spatial awareness to enhance MLLM agents' visual perception and reasoning. One promising direction is leveraging video pretraining (Madan et al., 2024; Wang et al., 2024) to better equip embodied agents for these challenges.

- *Visual In-context Learning (ICL).* Our experiments confirm the effectiveness of visual ICL (Zhou et al., 2024b; Sarch et al., 2024b) in embodied decision-making. This approach is promising because it enables adaptability and versatility without fine-tuning, allowing better use of off-the-shelf MLLMs. However, designing more effective visual ICL methods for embodied tasks remains an open problem for future research.

- *Training Multimodal Embodied Agents.* While our work focuses on evaluation, fine-tuning MLLMs for embodied tasks could significantly enhance their performance (Mu et al., 2024; Szot et al., 2024; Zawalski et al., 2024; Zhai et al., 2025). Future research can explore embodied pretraining, imitation learning, and both offline and online reinforcement learning (Sun, 2023) to better optimize MLLMs for embodied decision-making. Additionally, developing end-to-end learning approaches that seamlessly integrate perception, reasoning, and action could reduce the need for designing complex agent frameworks, leading to more adaptive and generalizable agents.

- *Robustness and Generalization of MLLM Agents.* Ensuring real-world applicability requires a thorough study of MLLM agents' robustness and generalization capabilities. While related studies are emerging in other domains (Zou et al., 2024; Xu et al., 2024; Yang et al., 2023b; 2024b; Zhang et al., 2024b), research on MLLM agents remains limited. Potential methods involve incorporating adversarial settings (Liu et al., 2024b; Wu et al., 2024a), dynamically generated environments (Wang et al., 2023c), or domain shifts (Chattopadhyay et al., 2021) to assess and enhance the ability of embodied agents to perform reliably in varying conditions.

By exploring these directions, the field can move closer to realizing the full potential of MLLM-based embodied agents in real-world applications.

## C. Details About Embodiedbench **Environments And Datasets**

Below, we provide detailed descriptions of four environments and their corresponding datasets. Please note that the maximum number of environment steps varies by task: 30 steps for high-level tasks (EB-ALFRED and EB-Navigation), 20 steps for EB-Navigation, and 15 steps for EB-Manipulation. In addition to task completion and exceeding the maximum step limit, we introduce two additional stopping conditions: (1) *Invalid Action Limit:* If the model generates more than 10 invalid actions in a single trajectory, indicating a lack of understanding and difficulty in producing valid actions. (2) *Empty Plan* Generation: If the model generates an empty plan because it incorrectly assumes the task is complete. This issue mainly occurs in high-level tasks, and once it happens, the model tends to keep generating empty plans without making progress. These additional stopping conditions help reduce unnecessary computational costs and improve evaluation efficiency.

## C.1. Eb-Alfred

Task Description. We develop the EB-ALFRED tasks based on the ALFRED dataset and the AI2-THOR simulator, which are well-regarded within the embodied AI community for their diverse household tasks and scenes. These tasks aim to evaluate an agent's ability to organize and execute sequences of high-level actions in household scenarios, such as "Put washed lettuce in the refrigerator." Each task in ALFRED can be described using the Planning Domain Definition Language (PDDL), which helps assess the agent's success in completing the task or subgoals. The ALFRED dataset includes 7 task types, *Pick & Place, Stack & Place, Pick Two & Place, Clean & Place, Heat & Place, Cool& Place, and Examine in Light*. Our simulator is based on Lota-Bench's implementation for 8 high-level action types: "pick up", "open", "close", "turn on", "turn off", "slice", "put down", and "find". Each action can be parameterized with a specific object to form an action, e.g.,"find an apple" or "pick up an apple". The simulation offers an egocentric view and text feedback on the validity of action execution and potential reasons for any invalid actions. For example, it may indicate "failure to pick up an object because another object is already being held." Despite its strengths, Lota-Bench's simulator has **three notable limitations**: (1) it does not support the *Pick Two & Place* task type due to the inability to handle multiple instances of one object type. (2) Some actions lead to incorrect task execution, such as the "put down" action erroneously placing an object on top of the sink instead of inside it, causing a correct action but unsuccessful outcome. (3) Additionally, some instructions in the original ALFRED dataset suffer from low quality. We observe the erroneous use of "potato" in task related to "tomato", which prevents agents from successfully completing the tasks due to these incorrect instructions. To enhance the simulation, we implemented several improvements. Firstly, we introduced **support for multi-instance** settings in ALFRED by appending index suffixes to objects, such as "find a cabinet 2," to accommodate multiple instances of the same object type. Therefore, we can support all 7 task types in ALFRED. Given the dynamic number of objects in the ALFRED dataset, we made the action space of EB-ALFRED dynamic, ranging from 171 to 298 actions. To minimize redundancy in the action space, we merge all "put down" actions into a single action, since only one object can be held at a time. Additionally, we manually **corrected bugs in the original simulation and improved the quality of**
language instructions to ensure tasks are solvable and actions can be executed more accurately. These enhancements make EB-ALFRED a high-quality benchmark for evaluating embodied agents. Dataset Collection. Following Lota-Bench (Choi et al., 2024), we use the valid seen set from the ALFRED dataset. We first partition the dataset based on the number of steps in the oracle policy. Specifically, we select 50 samples from the subset with fewer than 15 steps, carefully refining their instructions to minimize ambiguity and improve task solvability.

The commonsense and complex instruction subsets are primarily derived from this base subset, with GPT-4o augmentation tailored to specific capabilities. Additionally, we select 50 tasks with more than 15 steps to form the long-horizon subset. The visual appearance and spatial awareness subsets are chosen directly from the original dataset based on language descriptions of color/shape, or relative positions. In total, EB-ALFRED comprises 300 testing instances, evenly distributed across six subsets (50 instances each).

## C.2. Eb-Habitat

Task Description. EB-Habitat is developed based on the Language Rearrangement benchmark (Szot et al., 2023), featuring 282 diverse language instruction templates designed for robotic rearrangement tasks. It leverages the Habitat 2.0 simulator (Szot et al., 2021) and includes object data from the YCB dataset (Calli et al., 2015) and ReplicaCAD (Szot et al., 2021). The benchmark focuses on planning and executing 70 high-level skills to achieve user-defined goals, such as "Find a toy airplane and move it to the right counter." These skills are categorized into five action types: "navigation", "pick", "place",
"open", and "close", each parameterized by specific objects.

Unlike ALFRED, which permits navigation to any object, EB-Habitat constrains navigation to receptacle-type objects, requiring robots to visit multiple locations to locate target items. Task and subgoal completion are evaluated using PDDL, with agents receiving visual input and textual feedback similar to ALFRED. Given its broad range of language instructions and distinct navigation constraints, EB-Habitat serves as a complementary counterpart to EB-ALFRED, expanding the scope of our high-level embodied tasks. Dataset Collection. Habitat already provides fine-grained evaluation datasets with multiple subsets. We reorganize the subsets to formulate our dataset. Specifically, we merge "new scenes", "novel objects", and "instruction rephrasing" to form our base subset; we use the "context" set as our commonsense subset; we merge the "conditional instructions" and
"irrelevant instruction text" as our complex instruction subset; we use the "referring expressions" as our visual appearance subset; we use the "spatial relationship" as our spatial awareness subset; we merge the "multiple rearrangements" and
"multiple objects" as our long-horizon subset. Then, we sample 50 instances from each subset to form our EB-Habitat dataset, resulting in a total of 300 testing instances.

## C.3. Eb-Navigation

Task Description. EB-Navigation is an evaluation suite built on AI2-THOR, designed to assess the navigation capabilities of embodied agents. In each task, the agent is placed at a starting position and must use visual observations and behavior feedback to execute low-level actions. The goal is to locate a target object and navigate to its vicinity. The agent's action space consists of seven actions that are executable by physical robots: (1) Move forward/backward by ∆x. (2) Move rightward/leftward by ∆y. (3) Rotate to the right/left by ∆θ degrees. (4) Tilt the camera upward/downward by ∆φ degrees. At the start of each task, the agent is provided with a textual description of the action space, where each action is mapped to a unique index. Then, the agent selects an action by outputting the corresponding index, which the environment then executes. At the beginning of each step, the environment provides the agent with a first-person visual observation. Using this visual input, the agent performs planning and decision-making to choose its next action. After executing an action, the environment evaluates its validity. For example, it checks for collisions or obstacles that might cause the action to fail. The environment then provides this valid or invalid signal as feedback to the agent. This signal is the only feedback the agent receives, as it is feasible to obtain in real-world scenarios. Together with the visual observations, this feedback equips the agent with sufficient information to perform navigation tasks effectively. Dataset Collection. We constructed the dataset based on the original dataset provided by AI2-THOR. In AI2-THOR (Kolve et al., 2017), there are diverse scenes including environments such as kitchens, living rooms, and bedrooms, we designed a total of 90 navigation tasks, one for each scene. Each task dataset includes the following information: (1) Initial Robot Pose: Including its (*x, y, z*) coordinates and initial orientation. (2) *Target object information*: Specifying the object type, ID and the 3D coordinates of the object's center. (4) *Language navigation instruction*: A human-readable instruction specifying the target object the agent needs to navigate to. We ensure the validness of the task dataset through the implementation of the following characteristics: (1) Initial distance: The agent's starting position is carefully constrained to be at least a certain adjustable distance (denoted as α) from the target object. This adjustable α allows users to customize the number of navigation steps required for each task. (2) Target object accessibility: All target objects are exposed in the environment, reachable without requiring the agent to leave the ground. (3) Task completion conditions: A task is considered complete if the agent reaches a position within a specified distance threshold from the target object or if the maximum number of steps is exceeded. Additionally, the dataset includes an automated task-generation script. This script allows users to create custom task datasets by specifying parameters such as the target object type, initial distance threshold, and random seed for each scene. This flexibility ensures the dataset can be adapted to various research needs and scenarios.

For the capability-oriented subsets, we begin by sampling 60 instances from the original 90 tasks to form the base subset.

We then use GPT-4 to perform instruction augmentation, generating more complex instructions and incorporating common sense knowledge to create the complex instruction and common sense subsets. The visual appearance subset is manually curated to include detailed descriptions of the target object's color and shape. Finally, the long horizon subset is constructed by ensuring the target object is not visible in the agent's initial view, requiring extended navigation to locate it. In total, we collect 300 testing instances across these 5 subsets (excluding the spatial awareness subset).

## C.4. Eb-Manipulation

Task description. EB-Manipulation is an extension of VLMBench (Zheng et al., 2022) using the CoppeliaSim simulator (Rohmer et al., 2013) to control a 7-DoF Franka Emika Panda robotic arm. EB-Manipulation includes four task categories: (1) *Pick & Place Objects*, (2) *Stack Objects*, (3) *Shape Sorter Placement*, and (4) *Table Wiping*, each with randomly varied instances in color, position, shape, and orientation for diverse evaluation. The action space is a 7-dimensional vector. The simulator processes these actions and performs automatic motion planning to achieve the desired position. To facilitate motion planning, the environment operates in ABS EE POSE PLAN WORLD FRAME mode, ensuring automatic trajectory execution from the current pose to the target pose. This simplifies the agent's role in predicting keypoints necessary for task completion. Direct low-level manipulation is challenging for MLLMs due to insufficient domain-specific training. To overcome this, we implemented enhancements. **(1) Action space discretization**(Yin et al., 2024), which divides the position component into 100 bins and the orientation component into 120 bins, enabling valid actions to take forms like [*x, y, z, roll, pitch, yaw, gripper*] = [57, 61, 20, 10, 60, 25, 1]. Here, the first three dimensions (X, Y, Z) range from 0 to 100, while the next three (Pitch, Yaw, Gripper) range from 0 to 120. The gripper state remains binary (0.0 or 1.0). By discretizing the originally continuous action space, the model can predict actions using integer values, reducing complexity