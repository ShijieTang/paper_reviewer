000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

## 1 Introduction

Large language models (LLMs) have achieved notable success in tasks such as logical reasoning, commonsense understanding, and question answering (Achiam et al., 2023; Touvron et al., 2023; Brown et al., 2020; Wei et al., 2022; Yao et al., 2023). Building agents with LLM as backbones to handle tasks in interactive scenarios has been identified as a feasible and promising approach but applying these agents in the real world remains highly challenging (Wang et al., 2023b; Kim et al., 2024; Wang et al., 2024). Web automation is one such scenario where LLM agents autonomously control web browsers to perform online activities such as shopping and posting by executing actions like clicking or typing. When handling web automation tasks, two predominant agent workflow paradigms are empirically employed by current LLM agents. The first involves sequentially generating a series of actions (Koh et al., 2024; Deng et al., 2024; Zheng et al., 2023), as illustrated in Figure 1(a). The second commences with a planning phase, breaking down the task into a series of subtasks which are to be tackled one by one(Kim et al., 2024; Sodhi et al., 2024). For complex tasks that require multi-step reasoning, the second workflow is beneficial for that it breaks actions down into easier subtasks like chain-of-thought instead of a hard final task. However, such workflows are often constrained by the lack of guarantees regarding subtask feasibility (Wang et al., 2023b). An effective and commonlyused remedy is to rely on expertly designed subtask candidates with manually crafted prompts to decompose the entire task as illustrated in Figure 1(b). However, this method requires human annotation for new tasks, which limits the ability of LLM agents to autonomously adapt to more advanced and diverse web environments Sodhi et al. (2024); Zhou et al. (2022). Additionally, there is a significant shortage of data available for learning within these workflows, which collectively restricts the capabilities of LLMs during the planning and execution phases Lai et al. (2024). To overcome these limitations, we propose VeSX, a framework for LLM agents in web automation that seeks to improve subtask feasibility and empower LLM both in planning and execution through Verification, Self-correction, and in-conteXt learning.

Anonymous authors Paper under double-blind review

# Vesx: A Framework Featured By Verification, Self-Correction And In-Context Learning For Web Automation Tasks

While large language models have achieved remarkable success in tasks such as reasoning and question answering, applying LLMs to interactive tasks like web automation remains challenging. In web automation, existing planning-execution workflow often faces limitations due to the infeasible subtasks. We propose VeSX, a framework designed to enhance subtask feasibility through verification, selfcorrection, and in-context learning. VeSX introduces three key improvements: (1) subgoal-guided verification, which verifies the execution results of subtasks based on the preset subgoals; (2) hierarchical self-correction, which combines reflection and replanning, targeting to self-correct mistakes in both planning and execution phases; (3) exemplar bank, which improves in-context learning by partitioning execution trajectories and heuristically generating metadata for exemplars. We evaluate VeSX on WebArena benchmark and achieve the state-of-the-art average success rate of 0.34, which significantly outperforms existing methods without human guidance on all five scenarios.

## Abstract

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107

: the i-th subtask : the j-th action to address the i-th subtask : the i-th subgoal : verification for the i-th subgoal
(a) Only Execution Workflow Query: Tell me the status of my latest order and when will it arrive.

(b) Planning-Execution Workflow
(with human guidance)
Decompose into a series of subtasks
... ...

: find_order
("my latest order")
: get_information
("its status and arrival time") : **Answer**
planning:
+ +
click [status] **click [view details]**

... ... ...

Generate actions one by one predict action and execute:
... ...

predict action and execute:
... ...

click [my account] answer [Delivery. Tomorrow]

click [Sales]

click [my account] finish_subtask **answer [Delivery. Tomorrow]**

(c) VeSX Workflow (Ours):
Decompose into a series of subtask and subgoal pairs
: find the **latest order** ... ...

planning:
: Answer : check_url
("my_account/orders")
: check_webpage

("The order is latest")
click [my_account] scroll [down] click [orders] answer [Delivery. Tomorrow] click [sorted by] **click [order_123]**
... ... ...

: find my order list predict action, execute and verify:
pass fail pass click [time]

finish_subtask **finish_subtask**

fail selfcorrectionselfcorrection
VeSX introduces two significant modules to the planning-execution workflow for web automation tasks, as illustrated in Figure 1(c). The first involves adopting **subgoal-guided verification** to ascertain the successful completion of subtasks. VeSX generates subgoals in the planning phase and conducts verification to check whether the execution result satisfies the subgoal once execution is finished. We devised two modes for verification: self-verification which makes LLMs do verification by themselves, and external verification which calls for external functions to conduct verification. Verification ensures the completion of the subtasks and avoids the issue of subsequent subtasks becoming infeasible due to errors in the execution of previous subtasks. The second improvement is **hierarchical self-correction**, which aims to self-correct both execution and planning. When verification fails, VeSX first tries to identify and correct errors in the current subtask's execution trajectory, a process referred to as "reflection". If reflection is unsuccessful, the subtask is likely infeasible, prompting VeSX to regenerate the remaining incomplete subtasks in a process called "replanning". This hierarchical self-correction mechanism facilitates error correction during both the execution and planning phases, enabling the framework to effectively assess the feasibility of subtasks and generate new plans as replacements. Furthermore, VeSX develops an **exemplar bank** for in-context learning within the planning and execution phase. Existing research on in-context learning related to web automation primarily focuses on the execution-only workflow and is limited to simple simulated tasks, leading to a dependence on complete execution trajectories as exemplars (Kim et al., 2024; Zheng et al., 2023). However, complex web automation tasks in the real world involve long action sequences, and using entire trajectories as exemplars includes a lot of irrelevant information, which will consume excessive content space. To address these issues, VeSX samples actions and planning strategy from the entire trajectory as exemplars for execution and planning, and heuristically generates metadata for retrieval. This method creates a larger and more diverse dataset for the LLM to learn from, and therefore improve its performance in both the planning and execution phases. We evaluate VeSX on 5 scenarios under the web automation benchmark, WebArena(Zhou et al., 2023b). VeSX outperforms all publicly reported methods without human guidance by a huge margin and pushes the state-of-the-art success rate to 34.0, which even exceeds the performance of the method using human guidance.

## 2 Vesx 2.1 Overview

Planning LLM for Planning task query

: the i-th subtask : the i-th subgoal
: the j-th action to address the i-th subtask

: the expected observation after action
: verification for the i-th subgoal

: the real observation after action
: **check_url**
("/my_account/order/history")

: **Check my orders**
II. Hierarchical Self-Correction Execution LLM for Action Prediction Replanning Environment Input Current plan is infeasible, please generate a new plan:
I. Subgoal-Guided Verification Output

: Check **my express list**
: check_url

("/my_account/express")
: check_trajectory

("express sorted by time")
: check_trajectory

("This is the latest order")
External Verification
: click
[my_account]
: Sort by time Fail func check_url func check_page
: Find the order infomation of the latest
②

func check_traj Pass Fail Times Reflection Input When address the current subtask, something wrong happened. Try to find the step that is most in execution trajectory :

①
: scroll
[down]

Execution of next Subtask
... ... ... ...

Output Step 2 is wrong, I will retry from this step finish_subtask

The VeSX involves innovative design in both process and data aspects. In terms of process, VeSX implements a highly accurate verification scheme based on subgoals generated in the planning phase, which we refer to as subgoal-guided verification (Section 2.2). Additionally, when verification fails, we design a hierarchical self-correction scheme that involves local reflection to tackle the problems within one certain subtask and global replanning to address the errors happening in the planning (Section 2.3). In terms of data, to address the low efficiency of in-context-learning in the previous web automation methods, VeSX uses a single step instead of the entire trajectory for in-contextlearning, and improves the metadata construction method to efficiently and accurately retrieve effective exemplars (Section 2.4).

## 2.2 Subgoal-Guided Verification

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 Generate subgoals alongside with subtasks In order to verify whether execution results align with the intentions, VeSX instructs the LLM to generate not only subtasks but also corresponding subgoal pairs during the planning phase, as illustrated in Figure 2. The requirements for tasks in real-world scenarios are often highly diverse, leading to a wide array of corresponding solutions. However, the methods for assessing whether these requirements can be met are typically simplistic and fixed. For example, in web automation, although there may be hundreds of distinct task intents to handle, the evaluation methods are relatively limited. For navigation tasks, the evaluation may be based on whether a specific webpage has been reached. For information-gathering tasks, the presence of the required information on the final webpage is often the criterion. We observe that evaluating the execution result of a subtask is much simpler and more standardized compared to addressing the subtask itself. From this perspective, we propose the subgoal-guided verification, which evaluates the execution results of the subtasks according to the subgoals. Specifically, in VeSX, we generate pairs of subtasks and subgoals during the planning phase. Subtasks are articulated in natural language to capture the diverse task requirements, which in turn guide the generation of actions during the execution phase. Meanwhile, subgoals, which are formulated in either natural 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215

: Find my orders : Last URL should contains "my_account/orders" and the last page should be the list of my orders.

(a) Self-Verification LLM for verification url web page content action history pass / fail
: Find my orders **: check_url (endwith="my_account/orders") OR**
check_webpage_content (contain="my order list")
(b) External Verification url check_url Logical Expression pass / fail web page content **check_webpage_content**
action history **check_action_history**
: the i-th subtask **: the i-th subgoal**
or structured language—corresponding to the two verification methods we discuss later—are used to invoke external models or pre-designed tools to verify whether the subtasks have been successfully completed. Verification Method We employed two methods to implement verification as illustrated in Figure 3. The first method is self-verification, which involves using natural language for subgoals. Selfverification incorporates all external environmental information and execution processes, making the LLM itself verify whether the execution results align with the previous intent. Self-verification has the advantage of flexibility, which means it can give the LLM to make decisions according to the current environment rather than totally depending on the predefined scheme. The second method is external verification, which involves presenting subgoals in a structured language akin to function calls. In this approach, subgoals parameterize a set of predefined verification functions, creating a logical expression that defines the pass/fail criteria. During the verification process, external functions will be invoked according to the specifications of the subgoals. The results of these invocations will undergo logical operations to produce the final verification outcome, as illustrated in Figure 3(b). The advantage of this method is that it relies on external functions for verification, which is more stable, controllable, allows for swift adjustments, and reduces dependence on the LLM's capabilities.

## 2.3 Hierarchical Self-Correction

Due to the inherent unpredictability in web environments, errors during an agent's execution are often unavoidable, making self-correction essential. However, since planning can lead to infeasible subtasks, merely reflecting on and retrying execution actions is insufficient for effective error correction. From this perspective, we propose a hierarchical self-correction approach. Specifically,the hierarchical self-correction phase will be engaged when the verification fails. First, VeSX will em216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269

: check_webpage + **("The order is latest")** ... ...

: check_url
("my_account/orders") ... ...

: find the **latest**
order
... ...

... ...

click [sort by] **click [order 123]**
click [time]
Abstract Action Context Abstract Subtask input Execution exemplar bank Metadata Exemplar as key output as value environment observations
: the i-th subtask : the j-th action to address the i-th subtask
: click [order 123]
: the i-th subgoal **: verification for the i-th subgoal**
predicted action
ploy reflection to identify mistakes in the execution trajectories associated with the current subtask. If reflection fails, it indicates that the generated subtasks may be infeasible or too challenging to solve. In that case, replanning will adjust the subtasks and subgoals dynamically. Reflection In reflection, the LLM takes the execution trajectory and verification results as inputs to identify the step where the error occurred and subsequently re-execute from that point. Previous work on making LLMs reflect primarily focused on errors that occur when actions are generated by the environment and task queries. However, in real interactive scenarios, errors are likely to occur when changes in the environment after action execution do not align with the expected outcomes of those actions. Adjustments to previous actions are also needed in such cases. To address this issue, we enhanced the outputs during the execution phase by having the LLM additionally output the expected results e ijfollowing an action. This allows the model to reflect on whether there is a discrepancy between expectations and environment changes during reflection thereby adjusting the action generation accordingly. Replanning To enable dynamic adjustment capabilities in planning, VeSX employs replanning when reflection fails to correct errors. Specifically, when execution can not pass verification and reflection cannot rectify the issue, we consider the current subtask to potentially be infeasible. The previous error information and environment feedback are then fed into the LLM, prompting the LLM to regenerate subsequent subtasks and subgoals. Compared to the common framework that relies solely on reflection, the hybrid self-correction approach allows for adaptive adjustment, avoiding infeasible subtasks and thereby enhancing the long-term planning capabilities within the framework.

## 2.4 Exemplar Bank

Previous research has typically relied on recalling the entire correct execution trajectories to perform in-context learning for execution. However, real-world tasks often require multi-step reasoning and operations, with the underlying logic of these steps varying significantly. This makes it challenging to use the full trajectory as an exemplar to guide the prediction of single-step actions, as it incorporates a significant amount of extraneous information and consumes substantial context. We decompose trajectories into individual actions as exemplars and have meticulously designed a heuristic approach to generate their metadata, which will serve as the key for recalling these exemplars later there is a lack of research on collecting and recalling planning exemplars.

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 Metadata of execution exemplars VeSX utilizes single-step action instead of the entire trajectory as an exemplar to avoid the redundancy of the whole trajectory and enables LLM to learn from more diverse action examples. This involves designing metadata to describe the exemplars effectively. Specifically, for each action, VeSX will abstract the tackling subtask and concatenate with the action context, which means the sequence of prior actions within that subtask to get the metadata for the exemplar, as shown in Figure 4. For instance, in the subtask "Find the latest order" the correct execution trajectory constitutes three actions click [sort by], click [time], and click [order 123]. We treat each action as an individual exemplar and generate its metadata.

The metadata for the third action would be "Abstract Subtask: find conditional order; Action Context: click sort by, click time". For each piece of exemplar, we store its embedding of metadata as the key and the exemplar as the value to construct the execution exemplar bank. Subsequently, during execution, VeSX recalls the most relevant examples from the exemplar bank through similaritymatching embeddings for in-context learning. Collecting planning exemplars Due to the planning phase playing a significant role in the planning-execution framework, VeSX also collects the planning exemplars from the correct trajectory except for the execution exemplars. The metadata of planning exemplars is generated from its corresponding task query, abstracting the specific concepts in it, and then the planning exemplar bank is constructed with the key-value pairs of embedding of metadata and exemplars. Construct planning exemplar bank enhance models' reasoning ability in the planning phase and facilitate the generation of more feasible subtasks.

## 3 Experiments 3.1 Experimental Setup

We evaluate VeSX on 5 different scenarios within the benchmark WebArena. WebArena is a benchmark featured with complex web tasks across multiple scenarios including Shop (Amazon Shopping site), CMS (Amazon Commercial Management System), Red (Reddit), Git (Gitlab), and Map (Open Street Map). It builds several virtual websites from the snapshots of real-world websites like Amazon Shopping, GitLab, and Reddit, and allows agents to take actions freely like humans do in browsers, making it well-suited for evaluating what LLM web agents will do in the real world. To ensure fair comparisons, we choose gpt-4o the same as previous works without finetuning the LLMs. We use text-embedding-3-small as the embedding model and retrieve the top 3 relevant exemplars for in-context learning. We follow the browser system and action space built by Zhou et al. (2023b). We also set the temperature of the LLM to zero to minimize the uncertainty to generate reproducible results. For the data collection for the exemplar, we sampled 60 tasks randomly for each scenario and let the LLM try to solve them but not evaluate them with the correct answer. We heuristically filter out the bad trajectories like repeating actions or reaching the max actions, then make LLM self-evaluate the result by itself. If LLM considers the whole execution trajectory to be reasonable and the final result may be correct, the actions in this trajectory will be collected as exemplars. We use external verification in default unless a specific verification method is explicitly specified.

## 3.2 Main Results

Baselines We selected five works from different perspectives as the baseline for the experiment. Zhou et al. (2023b) and Lai et al. (2024) focus on improving the single model's reasoning ability by prompt or reinforcement learning: Zhou et al. (2023b) tries to improve the model performance using various combinations of prompt strategies. Lai et al. (2024) uses reinforcement learning to fine-tune an LLM with more knowledge of the web environment. Drouin et al. (2024) tries to refine this problem from the perspective of the environment, which builds a new browser system with more versatile actions and tries to resolve the tasks from the perspective of the environment. Koh et al. (2024) and Sodhi et al. (2024) improve from the perspective of the LLM agnet workflow, using the strategy like reflection and planning: Koh et al. (2024) enhances the workflow of generating actions sequentially by tree search method. And Sodhi et al. (2024) improves the planning-execution workflow with the planning in a stack of subtasks and expert-designed examples.

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

Method Model Human-Feedback **Shop. CMS Red. Git. Map Avg** Zhou et al. (2023b) gpt-4o × - - - - - 0.131 Lai et al. (2024) AutoWebGLM × - - - - - 0.182 Drouin et al. (2024) gpt-4o × 0.172 0.148 0.202 0.190 0.255 0.235 Koh et al. (2024) gpt-4o × 0.281 0.165 0.105 0.133 0.255 0.192

Sodhi et al. (2024) gpt-4o ✓ *0.369 0.242* **0.594 0.317 0.303** *0.335*

VeSX gpt-4o × **0.412 0.249** *0.545 0.283 0.291* **0.340**

Table 1: Task success rate on five scenarios of WebArena. The best performance is displayed in bold, and the second performance is displayed in italics. It constitutes of five scenarios: Shop (Amazon Shopping site), CMS (Amazon Commercial Management System), Red (Reddit), Git (Gitlab), and Map (Open Street Map) Table 2: Ablation study on subgoal-guided verification. SR represents the success rate.

Table 3: Ablation study on hierarchical selfcorrection. SR represents the success rate.

Table 4: Ablation study on exemplar bank. SR represents the success rate. \#P refers to the number of planning exemplars and \#E refers to the number of planning exemplars.

Table 5: Ablation study on building metadata.

SR represents the success rate.

As shown in Table 1, VeSX consistently outperforms the publicly known best-reported methods without human guidance in all five different scenarios. The improvements are particularly Significant in the Shop, CMS, and Red scenarios, with relative enhancements of 47%, 51%, and 169%, respectively, benefiting from our robust error detection and correction capabilities. Additionally, compared to the method in Sodhi et al. (2024), which uses expert-written examples for each scenario, our framework—entirely reliant on autonomous planning and execution by the LLM—demonstrates competitive performance, achieving superior results in two out of five scenarios, which are Shop and CMS. The hard tasks in Shop and CMS often require multi-step reasoning, creating a great opportunity for VeSX to showcase its error correction capabilities.

## 3.3 Ablation Studies

| Reflection       | Replan   | SR    | ∆SR    |    |    |       |     |
|------------------|----------|-------|--------|----|----|-------|-----|
| Do Verification? | Methods  | SR    | ∆SR    | ×  | ✓  | 0.261 | 0.0 |
| ✓(w/o expect)    | ×        | 0.314 | +0.052 |    |    |       |     |
| ✓                | ×        | 0.340 | +0.078 |    |    |       |     |
| ✓                | ✓        | 0.412 | +0.150 |    |    |       |     |
| ×                | -        | 0.288 | 0.0    |    |    |       |     |
| ✓                | self     | 0.346 | +0.058 |    |    |       |     |
| ✓                | external | 0.412 | +0.134 |    |    |       |     |

To evaluate the impact of each part of VeSX, we conduct a comprehensive ablation study in this section. Our experiments are mainly conducted in the Shop scenario, whose tasks often involves multistep reasoning and operations, well-suited for evaluating the effectiveness of our various methods. SR means the success rate of all tasks in this scenario, which corresponds to the column of Shop in Table 1. These findings validate the effectiveness of the subgoal-guided verification, hierarchical self-correction, and exemplar bank.

| correction. SR represents the success rate. Methods SR ∆SR   |       |        |    |     |                              |       |        |
|--------------------------------------------------------------|-------|--------|----|-----|------------------------------|-------|--------|
| Methods                                                      | SR    | ∆SR    | #P | #E  |                              |       |        |
| only execution exemplars                                     | 0.340 | 0.0    | 0  | 171 | w/o abstraction              | 0.275 | 0.0    |
| only planning exemplars                                      | 0.373 | +0.033 | 35 | 0   | w/o action context           | 0.333 | +0.058 |
| VeSX                                                         | 0.412 | +0.072 | 35 | 171 | abstraction & action context | 0.412 | +0.137 |

## 3.3.1 Ablating Subgoal-Guided Verification

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 We conducted a comparative experiment to assess the implementation and methodologies of verification. verification will not be performed for the implementation of no verification because it does not have the ability to detect the error. As shown in Table 2, with the implementation of both verification schemes, the success rate has significantly improved. It is because the two verification methods can both enable the framework the error-detection ability. Moreover, using external verification provides additional support compared to self-verification, resulting in even more pronounced enhancements.

## 3.3.2 Ablating Hierarchical Self-Correction

We conducted ablation experiments to assess the impact of various components of self-correction on overall performance. We tested three implementations: using only reflection, using only replanning, and employing reflection without generating expectations during execution. As illustrated in Table 3, it is evident that removing any component significantly affects the final performance. In comparison to replanning, the removal of reflection has a more substantial impact, indicating that errors generated during the execution phase are more frequent or potentially easier to rectify. Nonetheless, the absence of replanning also leads to a rapid decline in the success rate, underscoring its necessity. Moreover, the success rate decreases when expectations are not utilized, demonstrating that expectations indeed enhance VeSu's ability to detect discrepancies between environmental changes and action predictions.

## 3.3.3 Ablating Exemplar Bank

We performed ablation experiments on the components of the exemplar bank and evaluated various strategies for generating metadata to determine each element's contribution to overall system performance. The construction of the exemplar bank follows the same format as the main results. We sampled 60 tasks from the shopping experiment and employed a heuristic approach, allowing the LLM to evaluate its own execution results. Ultimately, we extracted 35 trajectories that were deemed correctly completed, thereby forming the exemplar bank. As demonstrated in Table 5, excluding either the planning or execution exemplars led to a decrease in the success rate. Notably, utilizing only planning exemplars, without the inclusion of execution exemplars, achieved a higher success rate despite relying on fewer data samples. Based on the observations found in Section 3.3.2, this phenomenon suggests that for each subtask, generating accurate execution sequences requires dynamic adjustments. Even with a limited number of samples, the LLM can accomplish tasks through self-correction, indicating a higher dependency on an effective error-correction mechanism rather than on the volume of data. Conversely, errors in the planning phase seem to pose a more complex challenge, requiring data as a reference for effective reasoning. This also implies that, compared to execution exemplars, planning exemplars are more efficient in improving performance. We conducted an ablation study on two techniques: abstracting subtasks and using action context. Our results indicate that both techniques are essential for enhancing the effectiveness of in-context learning. Moreover, omitting abstraction results in a more significant loss in accuracy compared to omitting action context. This may be attributed to the system's challenges in retrieving the most relevant examples from the exemplar bank and learning from them without abstraction.

## 4 Related Work

Web automation tasks. Web automation is a sequential decision process, where agents are given a task that can be carried out on a series of websites, such as buying products, booking a ticket, and summarizing opinions on forums. The earliest research in this area often formulated the task as a Markov Decision Process and employed reinforcement learning methods Liu et al. (2018); Gur et al. (2018); Jia et al. (2018); Humphreys et al. (2022). However, with the increase in the scale of websites and tasks, these methods suffer for they are hard to comprehend and process the long web documents and in lack of generalization to different tasks (Gur et al., 2023).

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 Large language model for web automation. Language models have been used as text decoders since the era reinforcement learning is the prevalent solution to web automation tasks (Humphreys et al., 2022). As the ability to reasoning and acting of language models soared, using language models to make decisions in a web environment shows better performance, particularly on the realworld scale in recent years. Most of these methods use in-context learning paradigm (Kim et al., 2024; Zheng et al., 2023) or deal with other difficulties like long document (Zheng et al., 2023; Deng et al., 2024). However, applying this paradigm to multi-step reasoning tasks often leads to suboptimal performance. Other approaches decompose these tasks into predefined subtasks (Sodhi et al., 2024) or sequences of actions (Kim et al., 2024), which require manual design and may introduce inaccuracies. There also be works using finetuning to improve language models' ability of reasoning or understanding of web pages (Gur et al., 2023; Lai et al., 2024), which is orthogonal to our method. Large language model agent workflow. LLM agents are artificial entities that can sense their environment, make decisions, and take actions. They leverage the capabilities of large language models (LLMs) to perform tasks autonomously (Xi et al., 2023). The general conceptual framework for LLM-based agents includes three main components: brain (primarily composed of an LLM), perception (multimodal inputs), and action (expanding the action space of LLM-based agents). There are many development in the research of how to construct LLM agent to interact with wild outer world, such as (Li et al., 2023; Wang et al., 2023a;b) or construct LLM agent to follow human's instructions to complete a given task with external tools, such as (Lightman et al., 2023; Hong et al., 2023; Lai et al., 2024; Significant Gravitas; Agashe et al., 2024). Web automation is a task that necessitates LLM the ability to interact with the web environment with the target given by human. Planning with large language model. How to augment language models with reasoning skills and ability to act is of great significance when using language models in real world. Recent researches find that providing a series of intermediate reasoning steps can help LLM perform complex reasoning (Wei et al., 2022). Previous works augment the reasoning steps with the observations of the external tools (Yao et al., 2023; 2024). These prompt strategies try to decomposing the task into a sequence of actions. However, when the tasks become complex and need more steps to solve, it will be hard for LLM to generate correct action each step and the reasoning steps are hard to learn through in-context learning. Some other methods try to decompose the tasks into predefined subtasks, which can be guided to finish through in-context learning (Khot et al., 2022; Zhou et al., 2022; Sodhi et al., 2024). These methods need to manually define and write prompts for subtasks, and when the tasks become more and more various, it will be time-consuming and costly. Researches in other areas demonstrated that decomposing the task into the subtasks and using the verification module to ensure the effectiveness of the subtasks perform well in complex reasoning tasks (Wang et al., 2023b; Gou et al., 2023; Zhou et al., 2023a). Due to the strong interactivity and complexity of web automation tasks, simply transplanting these solutions is not applicable.

## 5 Limitations And Discussion

VeSX is an innovative LLM workflow designed for handling web automation tasks, characterized by subgoal-guided verification, hierarchical self-correction, and an exemplar bank. During our research process, we identified the following limitations of VeSX. First, its performance depends on the accuracy of verification and the success rate of reflection and replanning corrections. Although VeSX has achieved significant improvements by relying on these modules, there is still considerable room for enhancement to fully solve web automation tasks. Second, data in the web automation field is relatively scarce. While VeSX has implemented sophisticated designs for the exemplar bank to increase data utilization, generating high-quality data automatically without human annotation remains a significant challenge.

Additionally, the workflow concepts in VeSX may lead to opportunities in other fields. The novel approaches to verification and self-correction could be beneficial for tasks that require long reasoning steps and facilitate the verification of intermediate results. For instance, this could apply to complex mathematical logic reasoning or coding tasks, as well as long-term tasks in simulation environments like Minecraft. VeSX also represents a technological solution for building LLM agents with error-correction capabilities. Our future work will first focus on making new improvements 486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 In this paper, we introduce VeSX, a framework of LLM agent for web automation tasks, featured with subgoal-guided verification, hierarchical self-correction, and exemplar bank to deal with the problem about infeasible subtasks and lack of data. Our experiments on the five different scenarios show VeSX have achieved state-of-the-art success rate and significantly outperform existing methods without human guidance. Our work shows the great potential of the workflow paradigm that armed with the planning-execution workflow with verification and self-correction. Moreover, VeSX proposes a new method to generate exemplars' descriptive metadata and constructs a memory system named exemplar bank for in-context learning in planning and execution phase. The ablation studies on the three contributions validate their effectiveness.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023.

Saaket Agashe, Jiuzhou Han, Shuyu Gan, Jiachen Yang, Ang Li, and Xin Eric Wang. Agent s: An open agentic framework that uses computers like a human. *arXiv preprint arXiv:2410.08164*, 2024.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020.

Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su.

Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36, 2024.

Alexandre Drouin, Maxime Gasse, Massimo Caccia, Issam H Laradji, Manuel Del Verme, Tom Marty, David Vazquez, Nicolas Chapados, and Alexandre Lacoste. Workarena: How capable are web agents at solving common knowledge work tasks? In Forty-first International Conference on Machine Learning, 2024.

Zhibin Gou, Zhihong Shao, Yeyun Gong, Yujiu Yang, Minlie Huang, Nan Duan, Weizhu Chen, et al. Tora: A tool-integrated reasoning agent for mathematical problem solving. arXiv preprint arXiv:2309.17452, 2023.

Izzeddin Gur, Ulrich Rueckert, Aleksandra Faust, and Dilek Hakkani-Tur. Learning to navigate the web. In *International Conference on Learning Representations*, 2018.

Izzeddin Gur, Hiroki Furuta, Austin V Huang, Mustafa Safdari, Yutaka Matsuo, Douglas Eck, and Aleksandra Faust. A real-world webagent with planning, long context understanding, and program synthesis. In *The Twelfth International Conference on Learning Representations*, 2023.

Sirui Hong, Xiawu Zheng, Jonathan Chen, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, et al. Metagpt: Meta programming for multiagent collaborative framework. *arXiv preprint arXiv:2308.00352*, 2023.

Peter C Humphreys, David Raposo, Tobias Pohlen, Gregory Thornton, Rachita Chhaparia, Alistair Muldal, Josh Abramson, Petko Georgiev, Adam Santoro, and Timothy Lillicrap. A data-driven approach for learning to control computers. In *International Conference on Machine Learning*,
pp. 9466–9482. PMLR, 2022.

Sheng Jia, Jamie Ryan Kiros, and Jimmy Ba. Dom-q-net: Grounded rl on structured language. In International Conference on Learning Representations, 2018.

and contributions in data generation for web automation tasks, and then explore how the VeSX framework can be implemented and utilized in other tasks.

## 6 Conclusion

Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, and Ashish Sabharwal. Decomposed prompting: A modular approach for solving complex tasks. In The Eleventh International Conference on Learning Representations, 2022.

Geunwoo Kim, Pierre Baldi, and Stephen McAleer. Language models can solve computer tasks.

Advances in Neural Information Processing Systems, 36, 2024.

Jing Yu Koh, Stephen McAleer, Daniel Fried, and Ruslan Salakhutdinov. Tree search for language model agents. *arXiv preprint arXiv:2407.01476*, 2024.

Hanyu Lai, Xiao Liu, Iat Long Iong, Shuntian Yao, Yuxuan Chen, Pengbo Shen, Hao Yu, Hanchen Zhang, Xiaohan Zhang, Yuxiao Dong, et al. Autowebglm: Bootstrap and reinforce a large language model-based web navigating agent. *arXiv preprint arXiv:2404.03648*, 2024.

Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem.

Camel: Communicative agents for "mind" exploration of large language model society, 2023.

URL https://arxiv.org/abs/2303.17760.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step. *arXiv preprint* arXiv:2305.20050, 2023.

Evan Zheran Liu, Kelvin Guu, Panupong Pasupat, Tianlin Shi, and Percy Liang. Reinforcement learning on web interfaces using workflow-guided exploration. In *International Conference on* Learning Representations, 2018.

Significant Gravitas. AutoGPT. URL https://github.com/Significant-Gravitas/
AutoGPT.

Paloma Sodhi, S. R. K. Branavan, Yoav Artzi, and Ryan McDonald. Step: Stacked llm policies for web actions, 2024.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023a.

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):186345, 2024.

Zihao Wang, Shaofei Cai, Guanzhou Chen, Anji Liu, Xiaojian Ma, and Yitao Liang. Describe, explain, plan and select: Interactive planning with large language models enables open-world multi-task agents. *arXiv preprint arXiv:2302.01560*, 2023b.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, Rui Zheng, Xiaoran Fan, Xiao Wang, Limao Xiong, Yuhao Zhou, Weiran Wang, Changhao Jiang, Yicheng Zou, Xiangyang Liu, Zhangyue Yin, Shihan Dou, Rongxiang Weng, Wensen Cheng, Qi Zhang, Wenjuan Qin, Yongyan Zheng, Xipeng Qiu, Xuanjing Huang, and Tao Gui. The rise and potential of large language model based agents: A survey, 2023. URL https://arxiv.org/abs/2309.07864.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao.

React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in Neural Information Processing Systems, 36, 2024.

Longtao Zheng, Rundong Wang, Xinrun Wang, and Bo An. Synapse: Trajectory-as-exemplar prompting with memory for computer control. In The Twelfth International Conference on Learning Representations, 2023.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Aojun Zhou, Ke Wang, Zimu Lu, Weikang Shi, Sichun Luo, Zipeng Qin, Shaoqing Lu, Anya Jia, Linqi Song, Mingjie Zhan, et al. Solving challenging math word problems using gpt-4 code interpreter with code-based self-verification. In The Twelfth International Conference on Learning Representations, 2023a.

Denny Zhou, Nathanael Scharli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuur- ¨
mans, Claire Cui, Olivier Bousquet, Quoc V Le, et al. Least-to-most prompting enables complex reasoning in large language models. In The Eleventh International Conference on Learning Representations, 2022.

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Yonatan Bisk, Daniel Fried, Uri Alon, et al. Webarena: A realistic web environment for building autonomous agents. *arXiv preprint arXiv:2307.13854*, 2023b.

## A Vesx

648 649 650 651 652 653 654 655 656 657 658 659 660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 To help readers better understand VeSX, we have provided a simplified algorithm flowchart of the VeSX workflow. We have emphasized only the most critical modules of the algorithm. Some variables and dependencies are not depicted in the flowchart, as they would make the overall presentation overly lengthy, such as the transmission of error messages after verification, and the control for the environment by reflection and replanning, among others. The construction of the Exemplar Bank is also not represented in this inference workflow.

## A.1.2 One Example

Here is a complete execution example of the workflow to clearly illustrate the VeSX process. The section for input prompts is presented in a simplified format to save space and enhance readability. For detailed information on the inputs and outputs of each module, please refer to Appendix A.2.

## Algorithm 1: Simplified Vesx Workflow

Input: Web Automation Task Query Q, initial state s0, LLM agent for planning LLMPLANNING, LLM agent for action prediction LLMACTION, LLM agent for reflection LLMREFLECTION, LLM agent for replanning LLMREPLANNING, verification module fVERIFICATION, exemplar-bank retrieval module R
Output: Answer for the query A
1 S := [(t1, g1),(t2, g2), ...,(tn, gn)] ← fPLANNING(*Q, s*0) ▷ Initial Planning 2 s ← s0, U ← [ ]
3 **foreach** (t i, gi) in S do 4 C ← [], m ← F alse, v ← *F alse* 5 **while** not v do 6 r ← R(t, c), a ← LLMACTION(*t, s, r*) ▷ Retrieval and Action Prediction 7 c ← A + a ▷ Action Chain ("+" indicates concatenate)
8 if a = FINISH *SUBTASK* **then**
9 v ← fVERIFICATION(*g, s, c*) ▷ Verification 10 if not v **then**
11 if not m **then**
12 c, s ← LLMREFLECTION(*t, c, s*) ▷ Reflection 13 m ← *T rue* 14 end 15 **else** 16 [(t i+1, gi+1), ...] ← LLMREPLANNING(S) ▷ Replanning 17 S.update([(t i+1, gi+1)*, ...*])
18 break 19 end 20 end 21 end 22 **else** 23 s ← ENV (*s, a*) 24 end 25 end 26 u ← LLMSUMMARY(*t, c, s*) ▷ Subtask Summary 27 U ← U + u 28 end 29 A ← LLMANSWER(*U, s*) ▷ Get Final Answer 30 **return** A
A.1.1 FLOWCHART A.1 WORKFLOW
702 703 704 705 706 707 708 709 710 711 712 713 714 715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 Subtask 1 1. Navigate to the "Search" function on the One Stop Market website. Figure of Observation (only for visualization, not used in input prompt): Analyze and Predict Action: Thought: Let's think step-by-step. The current observation shows the One Stop Market homepage with a search combobox and a disabled search button. The first step is to type "white desk" into the search combobox to enable the search button. Next Action: According to the thoughts, the next action I will perform is "'type [1697] [white desk]"' Target: After typing "white desk" into the search combobox, the search button should be enabled. Summary: The current observation shows the One Stop Market homepage with a search combobox and a disabled search button. The next step is to type "white desk" into the search combobox to enable the search button. The expected outcome is to enable the search button. Query Intent: What is the price range of Canon photo printer in the One Stop Market? Ground Truth Final Answer: 'must include': ['2.56', '649.99'] Planning 1. Navigate to the "Search" function on the One Stop Market website. - Objective: check in webpage("You have located the 'Search' function") 2. Enter "Canon photo printer" into the search bar and initiate the search. - Objective: check in url("search") |OR|check in webpage("Search results for 'Canon photo printer' displayed") 3. Filter the search results to display only products related to "Canon photo printer". - Objective: check in webpage("Search results filtered to display only 'Canon photo printer' products") 4. Identify the prices of all products listed under the "Canon photo printer" category. - Objective: check in webpage("Prices of all 'Canon photo printer' products identified") 5. Determine and display the price range (minimum and maximum prices) for the "Canon photo printer" products. - Objective: check in history("Price range for 'Canon photo printer' products determined and displayed")

## One Complete Example

756 757 758 759 760 761 762 763 764 765 766 767 768 769 770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 Action: type [1697] [white desk ] where [1697] is combobox 'ue615 Search' autocomplete: both hasPopup: listbox required: False expanded: False Figure of Observation (only for visualization, not used in input prompt):

## Analyze And Predict Action:

Thought: Let's think step-by-step. The current observation shows the search results page for "white desk" on the One Stop Market website. This indicates that the search function has already been navigated to and the search has been performed. Therefore, the subtask to navigate to the "Search" function on the One Stop Market website has been completed.

Next Action: According to the thoughts, the next action I will perform is "'finish subtask"'.

Target: Confirm that the search function has been navigated to and the search results for "white desk" are displayed. Summary: The current observation shows the search results page for "white desk" on the One Stop Market website. This indicates that the search function has already been navigated to and the search has been performed. Therefore, the subtask to navigate to the "Search" function on the One Stop Market website has been completed. Action: finish **subtask []** Verification: check in **webpage("You have located the 'Search' function")** 1.0 To verify if the 'Search' function has been located, I need to check if there is an element related to searching on the webpage. This could include a search bar, search button, or any other search-related elements. Verification Pass Subtask 2 Enter "Canon photo printer" into the search bar and initiate the search. Figure of Observation (only for visualization, not used in input prompt):
810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 Analyze and Predict Action: Thought: Let's think step-by-step. The current observation shows the search results page for "white desk" on the One Stop Market website. The search combobox is present, and the search button is currently disabled. The first step is to type "Canon photo printer" into the search combobox to enable the search button. Next Action: According to the thoughts, the next action I will perform is "'type [4312] [Canon photo printer] [clear previous content=1] [press enter after=1]"' Target: After typing "Canon photo printer" into the search combobox and pressing enter, the search results for "Canon photo printer" should be displayed. Summary: The current observation shows the search results page for "white desk" on the One Stop Market website. The next step is to type "Canon photo printer" into the search combobox and press enter to initiate the search. The expected outcome is to display the search results for "Canon photo printer". Action: type [4312] [Canon photo printer ] where [4312] is combobox 'ue615 Search' autocomplete: both hasPopup: listbox required: False expanded: False Figure of Observation (only for visualization, not used in input prompt):
864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 Verification: Objective: check in url("search") |OR|check in webpage("Search results for 'Canon photo printer' displayed" check in **url("search")** 1.0 URL: http://127.0.0.1:7770/catalogsearch/result/?q=Canon+photo+printer, "search" is in URL check in **webpage("Search results for 'Canon photo printer' displayed"** 1.0 The content of the webpage includes multiple instances of the phrase "Search results for: 'Canon photo printer'" and related elements such as headings and static text that confirm the search results are displayed Verification Pass Subtask 3 Filter the search results to display only products related to "Canon photo printer". Figure of Observation (only for visualization, not used in input prompt): Analyze and Predict Action: Thought: Let's think step-by-step. The current observation shows that I am on the search results page for the query "Canon photo printer." To filter the search results to display only products related to "Canon photo printer," I need to ensure that the search results are already filtered correctly. The page title and the search results indicate that the products displayed are related to "Canon photo printer." Next Action: According to the thoughts, the next action I will perform is "'finish subtask"' Analyze and Predict Action: Thought: Let's think step-by-step. The current observation shows the search results for "Canon photo printer" on the One Stop Market website. This indicates that the search function has already been navigated to and the search has been performed. Therefore, the subtask to enter "Canon photo printer" into the search bar and initiate the search has been completed. Next Action: According to the thoughts, the next action I will perform is "'finish subtask"'. Target: Confirm that the search results for "Canon photo printer" are displayed. Summary: The current observation shows the search results for "Canon photo printer" on the One Stop Market website. This indicates that the search function has already been navigated to and the search has been performed. Therefore, the subtask to enter "Canon photo printer" into the search bar and initiate the search has been completed. Action: finish **subtask []**
918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 Target: Confirm that the search results are already filtered to display only products related to "Canon photo printer." Summary: The current observation shows that the search results page is already displaying products related to "Canon photo printer." Therefore, the subtask has been completed. Action: finish **subtask []** Verification: check in webpage("Search results filtered to display only 'Canon photo printer' products") 0.0 I need to verify if the search results are specifically filtered to display only 'Canon photo printer' products. This can be determined by checking the search results and ensuring that they are relevant to 'Canon photo printer'.

Verification Fail Reflection Reflection: The execution process correctly identified that the search results page is displaying products related to "Canon photo printer." However, the verification result indicates that there needs to be a check to ensure that the search results are specifically filtered to display only "Canon photo printer" products. The mistake lies in not explicitly verifying the relevance of each product in the search results to "Canon photo printer." To solve the task, I should include a step to verify that all the products listed in the search results are indeed related to "Canon photo printer." This can be done by checking the titles and descriptions of the products to ensure they match the search criteria. Since the mistake occurred after identifying the search results, I should reload from the point just before finishing the subtask to include the verification step. Mistakes happened during verification. There is no need to retry. Verification (changed to) Pass Subtask 4 Identify the prices of all products listed under the "Canon photo printer" category. Figure of Observation (only for visualization, not used in input prompt): Analyze and Predict Action: Thought: Let's think step-by-step. The current subtask is to identify the prices of all products listed under the "Canon photo printer" category. The current observation shows the search 972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989 990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 results for "Canon photo printer" on the One Stop Market website. I need to locate the prices of the products listed under this category. Next Action: According to the thoughts, the next action I will perform is "'scroll [direction=down]"' Target: Scroll down the page to view more products and their prices. Summary: I will scroll down the page to view more products listed under the "Canon photo printer" category and their prices. Action: scroll [down] Analyze and Predict Action: Thought: Let's think step-by-step. I have identified several products under the "Canon photo printer" category along with their prices. Here are the products and their prices: 1. Canon Pixma iP3500 Photo Printer - $184.99 2. Canon PIXMA iP4920 Premium Inkjet Photo Printer - $649.99 3. Canon MP620 Wireless All-in-One Photo Printer - $599.99 4. Canon PIXMA MG2120 Color Photo Printer with Scanner and Copier - $2.56 5. Canon PIXMA MG5420 Wireless Color Photo Printer - $500.42 6. Canon PIXMA MP560 Wireless Inkjet All-In-One Photo Printer - $56.78 7. Canon SELPHY CP1300 Compact Photo Printer -
$208.99 8. Canon TS202 Inkjet Photo Printer - $49.54 I have identified the prices of all products listed under the "Canon photo printer" category. Next Action: According to the thoughts, the next action I will perform is "'finish subtask [I have identified the prices of all products listed under the "Canon photo printer" category: Canon Pixma iP3500 Photo Printer - $184.99, Canon PIXMA iP4920 Premium Inkjet Photo Printer - $649.99, Canon MP620 Wireless All-in-One Photo Printer - $599.99, Canon PIXMA MG2120 Color Photo Printer with Scanner and Copier - $2.56, Canon PIXMA MG5420 Wireless Color Photo Printer - $500.42, Canon PIXMA MP560 Wireless Inkjet All-In-One Photo Printer - $56.78, Canon SELPHY CP1300 Compact Photo Printer - $208.99, Canon TS202 Inkjet Photo Printer - $49.54]"' Target: I expect to complete the subtask of identifying the prices of all products Action: finish **subtask [I have identified the prices of all products listed under the** "Canon photo printer" category: Canon Pixma iP3500 Photo Printer - $184.99, Canon PIXMA iP4920 Premium Inkjet Photo Printer - $649.99, Canon MP620 Wireless All-in-One Photo Printer - $599.99, Canon PIXMA MG2120 Color Photo Printer with Scanner and Copier - $2.56, Canon PIXMA MG5420 Wireless Color Photo Printer - $500.42, Canon PIXMA MP560 Wireless Inkjet All-In-One Photo Printer - $56.78, Canon SELPHY CP1300 Compact Photo Printer - $208.99, Canon TS202 Inkjet Photo Printer - $49.54]
1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079

Figure of Observation (only for visualization, not used in input prompt):
Analyze and Predict Action: Thought: Let's think step-by-step. The current subtask is to determine and display the price range (minimum and maximum prices) for the "Canon photo printer" products. The current observation shows multiple Canon photo printers with their respective prices. I need to identify the minimum and maximum prices from the listed products. Here are the prices of the listed Canon photo printers: - $184.99 - $649.99 - $599.99 - $2.56 - $500.42 - $56.78 - $208.99 - $49.54 The minimum price is $2.56, and the maximum price is $649.99. Next Action: According to the thoughts, the next action I will perform is "'finish subtask [2.56, 649.99]"' Target: Confirm the price range (minimum and maximum prices) for the "Canon photo printer" products. Summary: The prices of the listed Canon photo printers have been identified, with the minimum price being $2.56 and the maximum price being $649.99. Therefore, the subtask has been completed. Action: finish **subtask [2.56, 649.99]** Verification: check in history("Price range for 'Canon photo printer' products determined and displayed") 1.0 The subgoal target is to determine and display the price range for 'Canon photo printer' products. The summary indicates that the prices have been identified, with a minimum price of $2.56 and a maximum price of $649.99. This information aligns with the subgoal target. Verification Pass

## Subtask 5

Determine and display the price range (minimum and maximum prices) for the "Canon photo printer" products. Verification: check in **webpage("Prices of all 'Canon photo printer' products identified")** 1.0 Prices of all 'Canon photo printer' products have been listed and identified. Verification Pass