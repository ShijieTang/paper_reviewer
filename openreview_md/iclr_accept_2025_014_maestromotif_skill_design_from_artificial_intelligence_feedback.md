# Maestromotif: Skill Design From Artificial Intelligence Feedback

Martin Klissarov1, 5, Mikael Henaff2**, Roberta Raileanu**2, Shagun Sodhani2, Pascal Vincent1, 2, Amy Zhang2, 3, Pierre-Luc Bacon**1, 4**,
Doina Precup1, 5, 8, Marlos C. Machado*, 6, 7, 8, Pierluca D'Oro***, 1, 2, 4** 1 Mila, 2 Meta, 3 University of Texas Austin, 4 Universite de Montr ´ eal, ´
5 McGill University, 6 University of Alberta, 7 Amii, 8 Canada CIFAR AI Chair

## Abstract

Describing skills in natural language has the potential to provide an accessible way to inject human knowledge about decision-making into an AI system. We present MaestroMotif, a method for AI-assisted skill design, which yields highperforming and adaptable agents. MaestroMotif leverages the capabilities of Large Language Models (LLMs) to effectively create and reuse skills. It first uses an LLM's feedback to automatically design rewards corresponding to each skill, starting from their natural language description. Then, it employs an LLM's code generation abilities, together with reinforcement learning, for training the skills and combining them to implement complex behaviors specified in language. We evaluate MaestroMotif using a suite of complex tasks in the NetHack Learning Environment (NLE), demonstrating that it surpasses existing approaches in both performance and usability.

## 1 Introduction

Bob wants to understand how to become a versatile AI researcher. He asks his friend Alice, a respected AI scientist, for advice. To become a versatile AI researcher, she says, one needs to practice the following skills: creating mathematical derivations, writing effective code, running and monitoring experiments, writing scientific papers, and giving talks. Alice believes that, once these different skills are mastered, they can be easily combined following the needs of any research project. Alice is framing her language description of how to be a versatile researcher as the description of a set of skills. This often happens among people, since this type of description is a convenient way to exchange information on how to become proficient in a given domain. Alice could instead have suggested what piece of code or equation to write, or, at an even lower level of abstraction, which keys to press; but she prefers not to do it, because it would be inconvenient, time-consuming, and likely tied to specific circumstances for it to be useful to Bob. Instead, describing important skills is easy but effective, transmitting large amounts of highlevel information about a domain without dealing with its lowest-level intricacies. Understanding how to do the same with AI systems is still a largely unsolved problem. Recent work has shown that systems based on Large Language Models (LLMs) can combine sets of skills to achieve complex goals (Ahn et al., 2022; Wang et al., 2024). This leverages the versatility of LLMs to solve tasks *zero-shot*, after the problem has been *lifted* from
* Equal supervision. Correspondence to: martin.klissarov@mail.mcgill.ca.

Navigation Interaction Composite Zero-Shot MaestroMotif LLM Policy Task Training Motif Embedding Sim.
Figure 1: Performance across NLE task categories. MaestroMotif largely outperforms existing methods zero-shot, including the ones trained on each task.

1 the low-level control space, which they have difficulty handling, to a high-level skill space grounded in language, to which they are naturally suited. However, humans cannot communicate skills to these systems as naturally as Alice did with Bob. Instead, such systems typically require humans to solve, by themselves, the *skill design problem*, the one of crafting policies subsequently used by the LLM. Designing those skills typically entails very active involvement from a human, including collecting skill-specific data, developing heuristics, or manually handling reward engineering (Ahn et al., 2022). Thus, existing frameworks for designing low-level skills controlled by LLMs require technical knowledge and significant amounts of labor from specialized humans. This effectively reduces their applicability and generality. In this paper, we introduce the paradigm of *AI-Assisted Skill Design*. In this paradigm, skills are created in a process of human-AI collaboration, in which a human provides a natural language description of the skills and an AI assistant automatically converts those descriptions into usable low-level policies. This strategy fully leverages the advantages of both human-based and AI-based skill design workflows: it allows humans to inject important prior knowledge about a task, which may enhance safety and performance for the resulting agents even in the absence of optimal AI assistants; at the same time, it automates the lowest-level and more time-consuming aspects of skill design.

Based on this paradigm, we propose MaestroMotif, a method that uses LLMs and reinforcement learning (RL) to build and combine skills for an agent to behave as specified in natural language.

MaestroMotif uses an LLM's feedback to convert high-level descriptions into skill-specific reward functions, via the recently-proposed Motif approach (Klissarov et al., 2024). It then crafts the skills by writing Python code using an LLM: first, it generates functions for the initiation and termination of each skill; then, it codes a policy over skills which is used to combine them. During RL training, the policy of each skill is optimized to maximize its corresponding reward function by interacting with the environment. At deployment time, MaestroMotif further leverages code generation via an LLM to create a policy over skills that can combine them almost instantaneously to produce behavior, in zero-shot fashion, as prescribed by a human in natural language. MaestroMotif thus takes advantage of RL from AI feedback to lift the problem of producing policies from low-level action spaces to high-level skill spaces, in a significantly more automated way than previous work. In the skill space, planning becomes much easier, to the point of being easily handled zero-shot by an LLM that generates code policies. These policies can use features of a programming language to express sophisticated behaviors that could be hard to learn using neural networks. In essence, MaestroMotif crafts and combines skills, similarly to motifs in a composition, to solve complex tasks. We evaluate MaestroMotif on a suite of tasks in the Nethack Learning Environment (NLE) (Kuttler ¨ et al., 2020), created to test the ability to solve complex tasks in the early phase of the game. We show that MaestroMotif is a powerful and usable system: it can, without any further training, succeed in complex navigation, interaction and composite tasks, where even approaches trained for these tasks struggle. We demonstrate that these behaviors cannot be achieved by baselines that maximize the game score, and we perform an empirical investigation of different components our method.

## 2 Background

A language-conditioned Markov Decision Process (MDP) (Liu et al., 2022) is a tuple M =
(S, A, G*, r, p, γ, µ*), where S is the state space, A is the action space, G is the space of natural language task specifications, r : *S × G →* R is the reward function, p : *S × A →* ∆(S) is the transition function, γ ∈ (0, 1] is the discount factor, µ ∈ ∆(S) is the initial state distribution.

A skill can be formalized through the concept of option (Sutton et al., 1999; Precup, 2000). A
deterministic Markovian option ω ∈ Ω is a triple (Iω, πω, βω), where Iω : *S → {*0, 1} is the initiation function, determining whether the option can be initiated or not, πω : S → ∆(A) is the intra-option policy, and βω : *S → {*0, 1} is the termination function, determining whether the option should terminate or not. Under this mathematical framework, the skill design problem is equivalent to constructing a set of options Ω that can be used by an agent. The goal of the agent is to provide a policy over options π : *G × S →* Ω. Whenever the termination condition of an option is reached, π selects the next option to be executed, conditioned on the current state. The performance of such a policy is defined by its expected return J(π) = E*µ,π,*Ω[P∞
t=0 γ tr(st)].

I!1 , !1 D
Interactions Dataset r'1 r'2 I!2 , !2 D!1 I!n , !n Agent Designer LLM Coder Init./Term. functions D!2 Reward training 2. Generation of Skills Initiation/Termination LLM
Annotator r'n D!n
⇡T
Per-skill Annotated Dataset Per-skill Reward Function Agent Designer Training Policy Over Skills Agent Designer LLM Coder 1. Automated Skills Reward Design 3. Generation of Training Policy Over Skills
!i I!{*i,...,n*}
Available Skills
{⇡!2 , ⇡!4 *,...,* ⇡!k }
Initiation Functions Termination Function
⇡T
Reinforcement learning to maximize 
⇡!ir'i Environment Policy Active Skill Over Skills 4. Skill training via Reinforcement Learning
In the AI-Assisted Skill Design paradigm, an agent designer provides a set of natural language prompts X = {x1, x2*, . . . , x*n}. Each prompt consists of a high-level description of a skill. An AI system should implement a transformation f : X → Ω to convert each prompt into an option. Note that the ideas and method presented in this paper generalize to the partially-observable setting and, in our experiments, we learn memory-conditioned policies.

## 3 Method

MaestroMotif leverages AI-assisted skill design to perform zero-shot control, guided by natural language prompts. To the best of our knowledge, it is the first method that, while only using language specifications and unannotated data, is able to solve end-to-end complex tasks specified in language. Indeed, RL methods trained from scratch cannot typically handle tasks specified in language (Touati et al., 2023), while LLM-based methods typically feature labor-intensive methodologies for learning low-level control components (Ahn et al., 2022; Wang et al., 2024). MaestroMotif combines the capability of RL from an LLM's feedback to train skills with an LLM's code generation ability which allows it to compose them at will. We first introduce MaestroMotif as a general method, describing its use for AI-assisted skill design and zero-shot control, then discussing its implementation.

## 3.1 Ai-Assisted Skill Design With Maestromotif

MaestroMotif performs AI-assisted skill design in four phases shown in Figure 2. It leverages LLMs in two ways: first to generate preferences, then to generate code for initiation/termination functions and for a training-time policy over skills. It then uses these components to train skills via RL.

Automated Skills Reward Design In the first phase, an agent designer provides a description for each skill, based on their domain knowledge. Then, MaestroMotif employs Motif (Klissarov et al., 2024) to create reward functions specifying desired behaviors for each skill: it elicits preferences of an LLM on pairs of interactions sampled from a dataset D, forming for each skill a dataset of skill-related preferences Dωi, and distilling those preferences into a skill-specific reward function rφi

"You are to write code which defines the method "select skill" of the NetHack Player class that selects amongst a set of skills in the videogame of NetHack. [...] You are faced with the task following task. Alternate between the first three levels of […]"
LLM
Coder State Action Environment Skill neural network
⇡
Policy Over Skills Skill index
by minimizing the negative log-likelihood, i.e., using the Bradley-Terry model:

$${\mathcal{L}}(\varphi_{i})=-\mathbb{E}_{(s_{1},s_{2},y)\sim{\mathcal{D}}_{\varphi_{i}}}\Bigg[1[y=1]\log P_{\varphi_{i}}[s_{1}\succ s_{2}]+1[y=2]\log P_{\varphi_{i}}[s_{2}\succ s_{1}]\Bigg],$$
$$\mathrm{(1)}$$

where y is an annotation generated by an LLM annotator and Pφ[sa ≻ sb] = e rφi
(sa)
e rφi
(sa)+e rφi
(sb)is the estimated probability of preferring a state to another (Bradley & Terry, 1952). Generation of Skill Initiation/Termination While a reward function can steer the behavior of a skill when it is active, it does not prescribe when the skill can be activated or when it should be terminated. In the options framework, this information is provided by the skill initiation and termination functions. MaestroMotif uses an LLM to transform a high-level specification into code that defines the initiation function Iωiand termination function βωifor each skill.

Generation of training-time policy over skills To be able to train skills, MaestroMotif needs a way to decide which skill to activate at which moment. While skills could be trained in isolation, having an appropriate policy allows one to learn skills using a state distribution closer to what will be needed during deployment, and to avoid redundancies. For instance, suppose the agent designer decided to have a two-skill decomposition, such that skill A's goal can only be achieved after skill B's goal is achieved; if they are not trained together, skill A would need to learn to achieve also the goal of skill B, nullifying any benefits from the decomposition. To avoid this, MaestroMotif leverages the domain knowledge of an agent designer, which gives a language specification of how to interleave skills for them to be learned more easily. From this specification, MaestroMotif crafts a policy over skills to be used at training time, πT , which, as with the previous phase, is generated as code by an LLM.

Skills training via RL In the last phase of AI-assisted skill design, the elements generated in the previous phases are combined to train the skill policies via RL. Following the call-and-return paradigm (Sutton et al., 1999), the training policy πT decides which skill to execute among the ones deemed as available by the initiation functions Iω{1*,...,n*}
. Then, the skill policy πωiof the selected skill gets executed in the environment and trained to maximize its corresponding reward function rφi until its termination function βωideactivates it. Initialized randomly at the beginning of the process, each skill policy will end up approximating the behaviors originally specified in natural language.

## 3.2 Zero-Shot Control With Maestromotif

After AI-assisted skill design, MaestroMotif has generated a set of skills, available to be combined. During deployment, a user can specify a task in natural language; MaestroMotif processes this language specification with a code-generating LLM to produce and run a *policy over skills* π that, without any additional training, can perform the particular task.

The policy over skills π is then used, together with the skill policies πω{*i,...,n*}
, initiation functions Iω{*i,...,n*}
, and termination functions βω{*i,...,n*}
, built through AI-assisted skill design, to compose the skills and implement the behavior specified by the user. This process follows the same call-and-return strategy, and recomposes the skills without any further training. It is illustrated in Figure 3, which shows concrete examples of prompts and outputs. More examples are reported in appendix.

## 3.3 Maestromotif On Nethack

We benchmark MaestroMotif on the NetHack Learning Environment (NLE) (Kuttler et al., 2020). ¨ In addition to being used in previous work on AI feedback, NetHack is a prime domain to study hierarchical methods, due to the fact that it is a long-horizon and complex open-ended system, containing a rich diversity of situations and entities, and requiring a vast array of strategies which need to be combined for success. To instantiate our method, we mostly follow the setup of Motif (Klissarov et al., 2024), with some improvements and extensions. We now describe the main choices for instantiating MaestroMotif on NetHack, reporting additional details in Appendix A. Skills definition Playing the role of agent designers, we choose and describe the following skills:
the Discoverer, the Descender, the Ascender, the Merchant and the Worshipper. The Discoverer is tasked to explore each dungeon level, collect items and survive any encounters. The Descender and Ascender are tasked to explore and specifically find staircases to either go up, or down, a dungeon level. The Merchant and the Worshipper are instructed to find specific entities in NetHack and interact with them depending on the context. These entities are shopkeepers for the Merchant, such that it attempts to complete transactions, and altars for the Worshipper, where it may identify whether items are cursed or not. The motivation behind some of these skills
(for example the Descender and Ascender pair) can be traced back to classic concepts such as bottleneck options (Iba, 1989; McGovern & Barto, 2001; Stolle & Precup, 2002).

Datasets and LLM choice To generate a dataset of preferences Dωifor each one of the skills, we mostly reproduce the protocol of Klissarov et al. (2024), and independently annotate pairs of observations collected by a Motif baseline. Additionally, we use the Dungeons and Data dataset of unannotated human gameplays (Hambro et al., 2022b). We use Llama 3.1 70B (Dubey et al., 2024) via vLLM (Kwon et al., 2023) as the LLM annotator, prompting it with the same basic mechanism employed in Klissarov et al. (2024). Annotation process In the instantiation of Motif presented in Klissarov et al. (2024), preferences are elicited from an LLM by considering a single piece of information provided by NetHack, the messages. Although this was successful in deriving an intrinsic reward that was generally helpful to play NetHack, our initial experiments revealed that this information alone does not provide enough context to obtain a set of rewards that encode more specific preferences for each skill. For this reason, we additionally include some of the player's statistics (i.e., dungeon level and experience level), as contained in the observations, when querying the LLM. Moreover, we leverage the idea proposed by Piterbarg et al. (2023a) of taking the difference between the current state and a state previously seen in the trajectory, providing the difference between states 100 time steps apart as the representation to the LLM. This provides a compressed history (i.e. a non-Markovian representation) to LLM and reward functions, while preventing excessively long contexts. Coding environment and Policy Over Skills A fundamental component of MaestroMotif is an LLM coder that generates Python code (Van Rossum & Drake Jr, 1995). MaestroMotif uses Llama 3.1 405b to generate code that is executed in the Python interpreter to yield initiation and termination functions for the skills, the train-time policy over skills, and the policies over skills employed during deployment. In practice, we find it beneficial to rely on an additional in-context code refinement procedure to generate the policies over skills. This procedure uses the LLM to write and run unit tests and verify their results to improve the code defining a policy over skill (see Appendix A.2 for more details). In our implementation, a policy over skills defines a function that returns the index of the skill to be selected. For the training policy, the prompt given to the LLM consists of the list of skills and a high-level description of an exploratory behavior of the type "alternate between the *Ascender* and the Descender; if you see a shopkeeper activate the Merchant*..."*, effectively transforming minimal domain knowledge to low-level information about a skill's desired state distributions. RL algorithm and skill architecture To train the individual skills, we leverage the standard CDGPT5 baseline based on PPO (Schulman et al., 2017) using the asynchronous implementation of *Sample Factory* (Petrenko et al., 2020). Instead of using a separate neural network for each skill, we train a single network, with the standard architecture implemented by Miffyli (2022), and an additional conditioning from a one-hot vector representing the skill currently being executed. This enables skills to have a shared representation of the environment, while at the same time reducing potential negative effects from a multi-head architecture (see Section 4.3).

## 4 Experiments

Figure 4: Simplified depiction of the early NetHack game where significant areas (such as branches) and entities are labeled.

We perform a detailed evaluation of the abilities of MaestroMotif on the NLE and compare its performance to a variety of baselines. Unlike most existing methods for the NLE, MaestroMotif is a zero-shot method, which produces policies entirely through skill recomposition, without any additional training. We emphasize this in our evaluation, by first comparing MaestroMotif to other methods for behavior specification from language on a suite of hard and composite tasks. Then, we compare the resulting agents with the ones trained for score maximization, and further analyze our method. We report all details related to the experimental setting in Appendix A.5. All results are averaged across nine seeds (for MaestroMotif, three repetitions for skill training and three repetitions for software policy generation), with error bars representing the standard error. All MaestroMotif results are obtained by recombining the skills without training, and the skills themselves were learned only through LLM feedback, without access to other types of reward signals. Evaluation suite As NetHack is a complex open-ended environment (Hughes et al., 2024), it allows for virtually limitless possibilities in terms of task definition and behavior specification. To capture this complexity and evaluate zero-shot control capabilities beyond what has been done in previous work, we define a comprehensive benchmark. We consider a set of relevant, compelling, and complex tasks related to the early part of the game, deeply grounded in both the original NLE paper (Kuttler ¨
et al., 2020) and the broader NetHack community (Moult, 2022). Our benchmark includes three types of tasks: *navigation* tasks, asking an agent to reach specific locations in the game; *interaction* tasks, asking an agent to interact with specific entities in the game; *composite* tasks, asking the agent to reach sequences of goals related to its location in the game and game status. In navigation and composite tasks, we evaluate methods according to their success rate; in interaction tasks, we evaluate methods according to the number of collected objects. Figure 4 presents an overall depiction of navigation and interaction tasks, and Appendix A.6 provides more details. Our evaluation is also inspired by SkillHack (Matthews et al., 2022), a benchmark for evaluating skill transfer on NetHack.

## 4.1 Performance Evaluation

Baselines We measure the performance of MaestroMotif on the evaluation suite described above. For MaestroMotif to generate a policy, it is sufficient for a user to specify a task description in natural language. For this reason, we mainly compare MaestroMotif to methods that are instructable via language: first, to using Llama as a policy via ReAct (Yao et al., 2022), which is an alternative zero-shot method; second, to methods that require task-specific training via RL, with reward functions

Overview of the early game of NetHack Dungeons of Doom Gnomish Mines Shopkeeper Altar Minetown Delphi

| Zero-shot        | Task-specific training   | Reward Information   |             |             |                           |
|------------------|--------------------------|----------------------|-------------|-------------|---------------------------|
| Task             | MaestroMotif             | LLM Policy           | Motif       | Emb. Simil. | RL w/ task reward + score |
| Gnomish Mines    | 46% ± 1.70%              | 0.1% ± 0.03%         | 9% ± 2.30%  | 3% ± 0.10%  | 3.20% ± 0.27%             |
| Delphi           | 29% ± 1.20%              | 0% ± 0.00%           | 2% ± 0.70%  | 0% ± 0.00%  | 0.00% ± 0.00%             |
| Minetown         | 7.2% ± 0.50%             | 0% ± 0.00%           | 0% ± 0.00%  | 0% ± 0.00%  | 0.00% ± 0.00%             |
| Transactions     | 0.66 ± 0.01              | 0.00 ± 0.00          | 0.08 ± 0.00 | 0.00 ± 0.00 | 0.01% ± 0.00%             |
| Price Identified | 0.47 ± 0.01              | 0.00 ± 0.00          | 0.02 ± 0.00 | 0.00 ± 0.00 | 0.00% ± 0.00%             |
| BUC Identified   | 1.60 ± 0.01              | 0.00 ± 0.00          | 0.05 ± 0.00 | 0.00 ± 0.00 | 0.00% ± 0.00%             |

Table 1: Results on navigation tasks and interaction tasks. MaestroMotif and LLM policy are zero-shot methods requiring no data collection or training on specific tasks; task-specific training methods generate rewards from text specifications (based on AI feedback or embedding similarity) and train an agent with RL; the last column reports the performance of a PPO agent using privileged reward information, a combination of the task reward and the game score (not accessible to the other methods). MaestroMotif largely outperforms all baselines, which struggle with complex tasks. generated by using either AI feedback or cosine similarity according to the embedding provided by a pretrained text encoder (Fan et al., 2022). In addition, we also compare to an agent trained to maximize a combination of the task reward and the game score (as auxiliary objective), which has thus access to privileged reward information compared to the other approaches. For all non-zero-shot methods, training runs of several GPU-days are required for each task before obtaining a policy. Results on navigation and interaction tasks Table 1 shows that MaestroMotif outperforms all the baselines, which struggle to achieve good performance, in navigation and interaction tasks. Notice that this happens despite the disadvantage to which MaestroMotif is subject when compared to methods that are specifically trained for each task. The poor performance of the LLM Policy confirms the trend observed by previous work (Klissarov et al., 2024): even if the LLM has enough knowledge and processing abilities to give sensible AI feedback, that does not mean that it can directly deal with low-level control and produce a sensible policy via just prompting. At the same time, methods that automatically construct a single reward function that captures a language specification break apart for complex tasks, resulting in a difficult learning problem for an agent trained with RL. MaestroMotif, instead, still leverages the ability of LLMs to automatically design reward functions, but uses code to decompose complex behaviors into sub-behaviors individually learnable via RL.

Results on composite tasks A feature of language is its compositionality. Since, in the type of system we consider, a user specifies tasks in language, different behavior specifications can be composed. For instance, an agent can be asked to first achieve a goal, then another one, then a last one. The *composite* category in our benchmark captures this type of task specifications. In Table 2, we compare MaestroMotif to other promptable baselines, showing the task description provided to the methods and their success rate. MaestroMotif has lifted the problem of solving a task to the

| Golden Exit                                                                                                                                                                                | Level Up & Sell                                                                                                                                                                    | Discovery Hunger                                                                                                                                                  |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| "Alternate between the first  three levels of the Dungeons  of Doom (at least once)  until you collect a minimum  of 20 gold pieces and defeat  25 monsters; finally try to  quit NetHack" | "Do not leave the first  dungeon level until you  achieve XP level 4, then find  a shopkeeper and sell an  item that you have collected;  finally survive for another  300 steps." | "Reach the oracle level (the  Delphi) in the Dungeons of  Doom, but not before  discovering the Gnomish  Mines and eating some food  there after getting hungry." |
| Tasks                                                                                                                                                                                      |                                                                                                                                                                                    |                                                                                                                                                                   |
| Methods                                                                                                                                                                                    |                                                                                                                                                                                    |                                                                                                                                                                   |
| MaestroMotif LLM Policy Motif Embedding Similarity                                                                                                                                         | 24.80 % ± 1.18 % 0% ± 0.00% 0% ± 0.00% 0% ± 0.00%                                                                                                                                  | 7.91% ± 1.47% 0% ± 0.00% 0% ± 0.00% 0% ± 0.00%                                                                                                                    |
| 7.09% ± 0.99% 0% ± 0.00% 0% ± 0.00% 0% ± 0.00%                                                                                                                                             |                                                                                                                                                                                    |                                                                                                                                                                   |

Completed Transactions Price Identification B/U/C Identification 0.00 0.40 0.80 1.20 1.60 Number of Ite ms Suc cess R
ate Gnomish Mines Delphi Minetown 0.00 0.10 0.20 0.30 0.40 MaestroMotif Motif Hierarchical BC + PPO BC + PPO PPO
one of generating a code policy: thus, even if the tasks entail extremely long-term dependencies, simple policies handling only a few variables can often solve them. In contrast, defining rewards that both specify complex tasks and are easily optimizable by RL is extremely hard for existing methods, because exploration and credit assignment in such a complex task become insurmountable challenges for a single low-level policy. To the best of our knowledge, MaestroMotif is the first approach to be competitive at decision-making tasks of this level of complexity, while simultaneously learning to interact through the lowest-level action space. Figure 6 reports an example of complex behavior exhibited by an agent created by MaestroMotif while solving one of the tasks. The overall results, aggregated over navigation, interaction and composite tasks, are presented in Figure 1.

Dungeons of Doom Gnomish Mines Dungeons of Doom Dungeons of Doom Gnomish Mines Gnomish Mines The agent searches for the Gnomish Mines.

Minimap Minimap Minimap A staircase leading to the Mines is found.

It fights Gnomes and collects items .

Dungeons of Doom Dungeons of Doom Dungeons of Doom Gnomish Mines Gnomish Mines Gnomish Mines Minimap Minimap It ascends back to the Dungeons of Doom.

Minimap Finally, the agent descends to the Delphi. 

After feeling hungry, the agent eats .

## 4.2 Comparison To Score Maximization

The vast majority of previous work on the NetHack Learning Environment has focused on agents trained to maximize the score of the game (Sypetkowski & Sypetkowski, 2021; Piterbarg et al., 2023b; Wolczyk et al., 2024). Although the score might seem like a potentially rich evaluation signal,

No rm al iz ed Performa n ceAblations Goal-Cond. & Simultaneously-Learned Goal-Cond. & Learned In Isolation Multiple heads & Simultaneously-Learned Multiple heads & Learned In Isolation 0.0 0.1 0.2 0.3 0.4 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 Global Step 1e9 0.0 0.2 0.4 0.6 0.8 1.0 0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00 Global Step 1e9 0.0 0.2 0.4 0.6 0.8 1.0 N
or m aliz ed Skill RewardDiscoverer Descender Ascender Merchant Worshipper N
or m aliz ed Skill Reward Discoverer Descender Ascender Merchant Worshipper
Figure 8: (a) Goal-conditioning as an architecture for skill selection and synchronous alternation of the skills using an exploration policy is essential for obtaining good performance. (b) When learning skills asynchronously (alternating them in different episodes), some important skills do not manage to be learned. (c) Learning skills synchronously automatically induces an emergent skill curriculum, in which basic skills are learned before the most complex ones.

it has been observed by previous work that a high-performing agent in terms of its score does not necessarily exhibit complex behaviors in the game (Hambro et al., 2022a). To illustrate this fact in the context of our work, we compare MaestroMotif's performance in the navigation and interaction tasks to the one achieved by agents trained to maximize the in-game score via different methods. Figure 5 reports the performance of these methods, showing that, even if maximizing the score might seem a good objective in NetHack, it does not align to the NetHack community's preferences, even when the source of training signal is an expert, such as in the behavioral cloning case.

## 4.3 Algorithm Analysis

Having demonstrated the performance and adaptability of MaestroMotif, we now investigate the impact of different choices on its normalized performance across task categories. Additional experiments can be found in Appendix A.8. Scaling behavior Central to the approach behind MaestroMotif is an LLM producing a policy over skills in code, re-composing a set of skills for zero-shot adaptation. It is known that the code generation abilities of an LLM depend on its scale (Dubey et al., 2024): therefore, one should expect that the quality of the policy over skills generated by the LLM coder will be highly dependent on the scale of the underlying model. We verify this in Figure 7, showing a clear trend of performance improvement for large models. In Appendix A.4, we also investigate the impact of code refinement on the performance of MaestroMotif. Hierarchical architecture As illustrated in Figure 10 of Appendix A.7, the neural network used to execute the skill policies follows almost exactly the same format as the PPO baseline (Miffyli, 2022), with the only difference of an additional conditioning via a one-hot vector representing the skill currently being executed. We found that this architectural choice to be crucial for effectively learning skill policies. In Figure 8a, we compare this choice to representing the skills through different policy heads, as is sometimes done in the literature (Harb et al., 2017; Khetarpal et al., 2020). This alternative approach leads to a collapse in performance. We hypothesize that this effect comes from gradient interference as the different skill policies are activated with different frequencies.

Emergent skill curriculum In Figure 8a, we also verify the importance of learning all the skills simultaneously. We compare this approach to learning each skill in a separate episode. We notice that without the use of the training-time policy over skills, the resulting performance significantly degrades. To better understand the reason behind this, we plot in Figure 8b and Figure 8c, for each skill, the corresponding reward during training. Learning each skill in isolation leads to a majority of the skills not maximizing their own rewards. On the other hand, learning multiple skills in the

Norm ali zed P
erforma nce 8b 70b 405b Scale of LLM Coder 0.0 0.1 0.2 0.3 0.4
same episode leaves space to learn and to leverage simpler skills, opening the possibility of using those simple skills to get to the parts of the environment where it is relevant to use more complex and context-dependent ones, such as the Merchant or the Worshipper. This constitutes an emergent skill curriculum, which is naturally induced by the training-time policy over skills. The curriculum emerges because of the data distribution in which each skill is initiated: a skill expressing a more advanced behavior will only be called by the policy over skills when the appropriate situation can be reached, which will only happen once sufficient mastery of more basic skills is acquired. We discuss in Appendix A.9 how modifying the skill selection strategy, for example by adapting it through online interactions, could further improve the ability to learn skills.

## 5 Related Work

LLM-based hierarchical control methods Our method relates to a line of work which also uses LLMs to coordinate low-level skills in a hierarchical manner. SayCan and Palm-E (Ahn et al., 2022; Driess et al., 2023) also use an LLM to execute unstructured, natural language commands by recomposing low-level skills in a zero-shot manner. A key difference in our work is how the skills are obtained: whereas they leverage a combination of large human teleoperation datasets of languageconditioned behaviors and hand-coded reward functions, we train skills from intrinsic rewards which are automatically synthesized from unstructured observational data and natural language descriptions. MaestroMotif is particularly related to those approaches in which a high-level policy is generated as a piece of code by an LLM (Liang et al., 2023). Voyager (Wang et al., 2024) also uses an LLM to hierarchically create and coordinate skills, but unlike our method, assumes access to control primitives which handle low-level sensorimotor control. LLMs have also been used for planning in PDDL domains (Silver et al., 2023), see Appendix A.10 for a detailed discussion. Hierarchical reinforcement learning There is a rich literature focusing on the discovery of skills through a variety of approaches, such as empowerment-based methods (Klyubin et al., 2008; Gregor et al., 2017), spectral methods (Machado et al., 2017; Klissarov & Machado, 2023) and feudal approaches (Dayan & Hinton, 1993; Vezhnevets et al., 2017) Most of these methods are based on learning a representation, which is then exploited by an algorithm for skill learning (Machado et al., 2023). In MaestroMotif, we instead work in the convenient space of natural language by leveraging LLMs, allowing us to build on key characteristics such as compositionality and interpretability. Interestingly, some of the skills we leverage in our NetHack implementation are directly connected to early ideas on learning skills, such as those based on notions of bottleneck and in-betweeness (Iba, 1989; McGovern & Barto, 2001; Menache et al., 2002; S¸ims¸ek & Barto, 2004). Such intuitive notions had not been scaled yet as they are hard to measure in complex environments. This is precisely what MaestroMotif provides: a bridge between abstract concepts and low-level sensorimotor execution.

## 6 Discussion

Modern foundation models possess remarkable natural language understanding and information processing abilities. Thus, even when they are not able to completely carry out a task on their own, they can be effectively integrated into human-AI collaborative systems to bring the smoothness and efficacy of the design of agents to new heights. In this paper, we showed that MaestroMotif is an effective approach for AI-assisted skill design, allowing us to achieve untapped levels of controllability for sequential decision making in the challenging NetHack Environment. MaestroMotif takes advantage of easily provided information (i.e., a limited number of prompts) to simultaneously handle the highest-level planning and the lowest-level sensorimotor control problems, linking them together by leveraging the best of the LLM and the RL worlds.

Like other hierarchical approaches, MaestroMotif is limited in the behaviors it can express by the set of skills it has at its disposal; given a set of skills, a satisfactory policy for a task might not be representable through their composition. Therefore, an agent designer should perform AI-assisted skill design while keeping in mind what behaviors should be eventually expressed by the resulting agents. Despite this inherent limitation, we believe our work provides a first step towards a new class of skill design methods, more effective and with a significantly higher degree of automation than existing ones. More broadly, MaestroMotif also constitutes evidence for the benefits of a paradigm based on human-AI collaboration, which takes advantage of the complementary strengths of both.

## References

Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Daniel Ho, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Eric Jang, Rosario Jauregui Ruano, Kyle Jeffrey, Sally Jesmonth, Nikhil J Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Kuang-Huei Lee, Sergey Levine, Yao Lu, Linda Luu, Carolina Parada, Peter Pastor, Jornell Quiambao, Kanishka Rao, Jarek Rettinghouse, Diego Reyes, Pierre Sermanet, Nicolas Sievers, Clayton Tan, Alexander Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Mengyuan Yan, and Andy Zeng. Do as i can, not as i say: Grounding language in robotic affordances, 2022. URL https://arxiv.org/abs/2204.01691.

Jacob Andreas, Dan Klein, and Sergey Levine. Modular multitask reinforcement learning with policy sketches. In Doina Precup and Yee Whye Teh (eds.), Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, volume 70 of Proceedings of Machine Learning Research, pp. 166–175. PMLR, 2017.

Akhil Bagaria, Jason K. Senthil, and George Dimitri Konidaris. Skill discovery for exploration and planning using deep skill graphs. In *International Conference on Machine Learning*, 2021. URL
https://api.semanticscholar.org/CorpusID:221094080.

Marc G. Bellemare, Sriram Srinivasan, Georg Ostrovski, Tom Schaul, David Saxton, and Remi Munos. ´
Unifying count-based exploration and intrinsic motivation. In Neural Information Processing Systems, 2016. URL https://api.semanticscholar.org/CorpusID:8310565.

Ralph Allan Bradley and Milton E. Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. *Biometrika*, 39:324, 1952. URL https://api.semanticscholar. org/CorpusID:125209808.

Paul Francis Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei.

Deep reinforcement learning from human preferences. *ArXiv*, abs/1706.03741, 2017. URL https://api.semanticscholar.org/CorpusID:4787508.

Ozg ¨ ur¨ S¸ims¸ek and Andrew G. Barto. Using relative novelty to identify useful temporal abstractions in reinforcement learning. In Proceedings of the Twenty-First International Conference on Machine Learning, ICML '04, pp. 95, New York, NY, USA, 2004. Association for Computing Machinery. ISBN 1581138385. doi: 10.1145/1015330.1015353. URL https://doi.org/10.1145/ 1015330.1015353.

Christian Daniel, Malte Viering, Jan Metz, Oliver Kroemer, and Jan Peters. Active reward learning.

In *Robotics: Science and Systems*, 2014. URL https://api.semanticscholar.org/ CorpusID:16043466.

Peter Dayan and Geoffrey E Hinton. Feudal reinforcement learning. In Advances in neural information processing systems, pp. 271–278, 1993.

Gary L. Drescher. Made-up minds - a constructivist approach to artificial intelligence. 1991. URL
https://api.semanticscholar.org/CorpusID:3099707.

Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Pete Florence. Palm-e: An embodied multimodal language model. In *arXiv preprint arXiv:2303.03378*, 2023.

Abhimanyu Dubey, Abhinav Jauhri, and Abhinav Pandey et al. The llama 3 herd of models.

ArXiv, abs/2407.21783, 2024. URL https://api.semanticscholar.org/CorpusID:
271571434.

Linxi Fan, Guanzhi Wang, Yunfan Jiang, Ajay Mandlekar, Yuncong Yang, Haoyi Zhu, Andrew Tang, De-An Huang, Yuke Zhu, and Anima Anandkumar. Minedojo: Building open-ended embodied agents with internet-scale knowledge. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022. URL https://openreview.net/forum? id=rc8o_j8I8PX.

Richard Fikes, Peter E. Hart, and Nils J. Nilsson. Learning and executing generalized robot plans.

Artif. Intell., 3:251–288, 1993. URL https://api.semanticscholar.org/CorpusID:
17260619.

Karol Gregor, Danilo Jimenez Rezende, and Daan Wierstra. Variational intrinsic control. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Workshop Track Proceedings. OpenReview.net, 2017. URL https://openreview. net/forum?id=Skc-Fo4Yg.

Eric Hambro, Sharada Mohanty, Dmitrii Babaev, Minwoo Byeon, Dipam Chakraborty, Edward Grefenstette, Minqi Jiang, Jo Daejin, Anssi Kanervisto, Jongmin Kim, Sungwoong Kim, Robert Kirk, Vitaly Kurin, Heinrich Kuttler, Taehwon Kwon, Donghoon Lee, Vegard Mella, Nantas ¨ Nardelli, Ivan Nazarov, Nikita Ovsov, Jack Holder, Roberta Raileanu, Karolis Ramanauskas, Tim Rocktaschel, Danielle Rothermel, Mikayel Samvelyan, Dmitry Sorokin, Maciej Sypetkowski, and ¨ Michał Sypetkowski. Insights from the neurips 2021 nethack challenge. In Douwe Kiela, Marco Ciccone, and Barbara Caputo (eds.), Proceedings of the NeurIPS 2021 Competitions and Demonstrations Track, volume 176 of *Proceedings of Machine Learning Research*, pp. 41–52. PMLR, 06–14 Dec 2022a. URL https://proceedings.mlr.press/v176/hambro22a.html.

Eric Hambro, Roberta Raileanu, Danielle Rothermel, Vegard Mella, Tim Rocktaschel, Heinrich ¨
Kuttler, and Naila Murray. Dungeons and data: A large-scale nethack dataset. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022b. URL https://openreview.net/forum?id=zHNNSzo10xN.

Jean Harb, Pierre-Luc Bacon, Martin Klissarov, and Doina Precup. When waiting is not an option :
Learning options with a deliberation cost. *ArXiv*, abs/1709.04571, 2017. URL https://api. semanticscholar.org/CorpusID:19247295.

Mikael Henaff, Roberta Raileanu, Minqi Jiang, and Tim Rocktaschel. Ex- ¨
ploration via elliptical episodic bonuses. In *NeurIPS*, 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ f4f79698d48bdc1a6dec20583724182b-Abstract-Conference.html.

C. A. R. Hoare. An axiomatic basis for computer programming. *Commun. ACM*, 26:53–56, 1969.

URL https://api.semanticscholar.org/CorpusID:6059550.

Edward Hughes, Michael Dennis, Jack Parker-Holder, Feryal Behbahani, Aditi Mavalankar, Yuge Shi, Tom Schaul, and Tim Rocktaschel. Open-endedness is essential for artificial superhuman intelligence, 06 2024.

Glenn A. Iba. A heuristic approach to the discovery of macro-operators. *Machine Learning*, 3:
285–317, 1989. URL https://api.semanticscholar.org/CorpusID:13649095.

Khimya Khetarpal, Martin Klissarov, Maxime Chevalier-Boisvert, Pierre-Luc Bacon, and Doina Precup. Options of interest: Temporal abstraction with interest functions. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pp. 4444–4451, 2020.

Martin Klissarov and Marlos C. Machado. Deep laplacian-based options for temporally-extended exploration. In *International Conference on Machine Learning*, 2023. URL https://api. semanticscholar.org/CorpusID:256274634.

Martin Klissarov, Pierluca D'Oro, Shagun Sodhani, Roberta Raileanu, Pierre-Luc Bacon, Pascal Vincent, Amy Zhang, and Mikael Henaff. Motif: Intrinsic motivation from artificial intelligence feedback. In *The Twelfth International Conference on Learning Representations*, 2024. URL https://openreview.net/forum?id=tmBKIecDE9.

Alexander S Klyubin, Daniel Polani, and Chrystopher L Nehaniv. Keep your options open: An information-based driving principle for sensorimotor systems. *PloS one*, 3(12):e4018, 2008.

W Bradley Knox and Peter Stone. Interactively shaping agents via human reinforcement: The tamer framework. In *Proceedings of the fifth international conference on Knowledge capture*, pp. 9–16, 2009.

George Dimitri Konidaris, Leslie Pack Kaelbling, and Tomas Lozano-Perez. From skills to symbols:
Learning symbolic representations for abstract high-level planning. *J. Artif. Intell. Res.*, 61: 215–289, 2018. URL https://api.semanticscholar.org/CorpusID:31918172.

Nishanth Kumar, Tom Silver, Willie McClinton, Linfeng Zhao, Stephen Proulx, Tomas Lozano-P ´ erez, ´
Leslie Pack Kaelbling, and Jennifer Barry. Practice makes perfect: Planning to learn skill parameter policies. In *Robotics: Science and Systems (RSS)*, 2024.

Heinrich Kuttler, Nantas Nardelli, Alexander H. Miller, Roberta Raileanu, Marco Selvatici, Edward ¨
Grefenstette, and Tim Rocktaschel. The NetHack Learning Environment. In ¨ Proceedings of the Conference on Neural Information Processing Systems (NeurIPS), 2020.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E.

Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In *Proceedings of the ACM SIGOPS 29th Symposium on Operating* Systems Principles, 2023.

Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. Code as policies: Language model programs for embodied control. In IEEE International Conference on Robotics and Automation, ICRA 2023, London, UK, May 29 - June 2, 2023, pp. 9493–9500. IEEE, 2023. doi: 10.1109/ICRA48891.2023.10160591. URL https:
//doi.org/10.1109/ICRA48891.2023.10160591.

B. Liu, Yuqian Jiang, Xiaohan Zhang, Qian Liu, Shiqi Zhang, Joydeep Biswas, and Peter Stone. Llm+p: Empowering large language models with optimal planning proficiency.

ArXiv, abs/2304.11477, 2023. URL https://api.semanticscholar.org/CorpusID:
258298051.

Minghuan Liu, Menghui Zhu, and Weinan Zhang. Goal-conditioned reinforcement learning: Problems and solutions. In Luc De Raedt (ed.), *Proceedings of the Thirty-First International Joint Conference* on Artificial Intelligence, IJCAI 2022, Vienna, Austria, 23-29 July 2022, pp. 5502–5511. ijcai.org, 2022.

Marlos C Machado, Marc G Bellemare, and Michael Bowling. A laplacian framework for option discovery in reinforcement learning. In *International Conference on Machine Learning*, pp.

2295–2304. PMLR, 2017.

Marlos C. Machado, Andre Barreto, and Doina Precup. Temporal abstraction in reinforcement ´
learning with the successor representation. *J. Mach. Learn. Res.*, 24:80:1–80:69, 2023. URL https://api.semanticscholar.org/CorpusID:238634579.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Sean Welleck, Bodhisattwa Prasad Majumder, Shashank Gupta, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative refinement with selffeedback. *ArXiv*, abs/2303.17651, 2023. URL https://api.semanticscholar.org/ CorpusID:257900871.

Bhaskara Marthi, Stuart Russell, David Latham, and Carlos Guestrin. Concurrent hierarchical reinforcement learning. In Leslie Pack Kaelbling and Alessandro Saffiotti (eds.), IJCAI-05, Proceedings of the Nineteenth International Joint Conference on Artificial Intelligence, Edinburgh, Scotland, UK, July 30 - August 5, 2005, pp. 779–785. Professional Book Center, 2005. URL
http://ijcai.org/Proceedings/05/Papers/1552.pdf.

Michael Matthews, Mikayel Samvelyan, Jack Parker-Holder, Edward Grefenstette, and Tim Rocktaschel. Hierarchical kickstarting for skill transfer in reinforcement learning, 2022. URL ¨
https://arxiv.org/abs/2207.11584.

Drew McDermott, Malik Ghallab, Adele E. Howe, Craig A. Knoblock, Ashwin Ram, Manuela M.

Veloso, Daniel S. Weld, and David E. Wilkins. Pddl-the planning domain definition language.

1998. URL https://api.semanticscholar.org/CorpusID:59656859.

Amy McGovern and Andrew G. Barto. Automatic discovery of subgoals in reinforcement learning using diverse density. In *International Conference on Machine Learning*, 2001. URL https:
//api.semanticscholar.org/CorpusID:1223826.

Ishai Menache, Shie Mannor, and Nahum Shimkin. Q-cut - dynamic discovery of sub-goals in reinforcement learning. In *European Conference on Machine Learning*, 2002. URL https:
//api.semanticscholar.org/CorpusID:7830103.

Jorge Mendez-Mendez, Leslie Pack Kaelbling, and Tomas Lozano-P ´ erez. Embodied lifelong learning ´
for task and motion planning. In *7th Annual Conference on Robot Learning*, 2023. URL https:
//openreview.net/forum?id=ZFjgfJb_5c.

Miffyli. nle-sample-factory-baseline, 2022. URL https://github.com/Miffyli/
nle-sample-factory-baseline. GitHub repository.

Dion Moult. Nethack: an illustrated guide to the mazes of menace, dec 2022. URL https://
thinkmoult.com/nethack-illustrated-guide-mazes-of-menace.html. Accessed: 2024-09-02.

Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron C. Courville. Film: Visual reasoning with a general conditioning layer. In *AAAI Conference on Artificial Intelligence*, 2017. URL https://api.semanticscholar.org/CorpusID:19119291.

Aleksei Petrenko, Zhehui Huang, Tushar Kumar, Gaurav S. Sukhatme, and Vladlen Koltun. Sample factory: Egocentric 3d control from pixels at 100000 FPS with asynchronous reinforcement learning. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of *Proceedings of Machine Learning Research*, pp. 7652–
7662. PMLR, 2020. URL http://proceedings.mlr.press/v119/petrenko20a. html.

Ulyana Piterbarg, Lerrel Pinto, and Rob Fergus. diff history for neural language agents, 2023a. Ulyana Piterbarg, Lerrel Pinto, and Rob Fergus. Nethack is hard to hack, 2023b.

Doina Precup. *Temporal Abstraction in Reinforcement Learning*. PhD thesis, University of Massachusetts Amherst, 2000.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*, 2017.

Noah Shinn, Federico Cassano, Beck Labash, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: language agents with verbal reinforcement learning. In Neural Information Processing Systems, 2023. URL https://api.semanticscholar.org/CorpusID: 258833055.

Tom Silver, Soham Dan, Kavitha Srinivas, Joshua B. Tenenbaum, Leslie Pack Kaelbling, and Michael Katz. Generalized planning in pddl domains with pretrained large language models. In *AAAI* Conference on Artificial Intelligence, 2023. URL https://api.semanticscholar.org/ CorpusID:258762760.

Martin Stolle and Doina Precup. Learning options in reinforcement learning. In Symposium on Abstraction, Reformulation and Approximation, 2002. URL https://api.semanticscholar. org/CorpusID:16398811.

Shao-Hua Sun, Te-Lin Wu, and Joseph J. Lim. Program guided agent. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id= BkxUvnEYDH.

Richard S. Sutton, Doina Precup, and Satinder Singh. Between mdps and semi-mdps: A
framework for temporal abstraction in reinforcement learning. 1999. URL https://api. semanticscholar.org/CorpusID:259159153.

Maciej Sypetkowski and Michał Sypetkowski. Autoascend - 1st place nethack agent for the nethack challenge at neurips 2021. https://github.com/maciej-sypetkowski/ autoascend, 2021. GitHub repository.

Andrea Lockerd Thomaz, Cynthia Breazeal, et al. Reinforcement learning with human teachers:
Evidence of feedback and guidance with implications for learning performance. In *Aaai*, volume 6, pp. 1000–1005. Boston, MA, 2006.

Ahmed Touati, Jer´ emy Rapin, and Yann Ollivier. Does zero-shot reinforcement learning exist? ´
In *The Eleventh International Conference on Learning Representations*, 2023. URL https:
//openreview.net/forum?id=MYEap_OcQI.

Guido Van Rossum and Fred L Drake Jr. *Python tutorial*, volume 620. Centrum voor Wiskunde en Informatica Amsterdam, 1995.

Alexander Sasha Vezhnevets, Simon Osindero, Tom Schaul, Nicolas Heess, Max Jaderberg, David Silver, and Koray Kavukcuoglu. Feudal networks for hierarchical reinforcement learning. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 3540–3549, 2017.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. *Transactions on Machine Learning Research*, 2024. ISSN 2835-8856. URL
https://openreview.net/forum?id=ehfRiF0R3a.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. *Advances in* Neural Information Processing Systems, 35:24824–24837, 2022.

Maciej Wolczyk, Bartłomiej Cupiał, Mateusz Ostaszewski, Michal Bortkiewicz, Michal Zajkac, Razvan Pascanu, Lukasz Kuci'nski, and Piotr Milo's. Fine-tuning reinforcement learning models is secretly a forgetting mitigation problem. *ArXiv*, abs/2402.02868, 2024. URL https://api. semanticscholar.org/CorpusID:267412811.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao.

React: Synergizing reasoning and acting in language models. *ArXiv*, abs/2210.03629, 2022. URL https://api.semanticscholar.org/CorpusID:252762395.

Tianjun Zhang, Huazhe Xu, Xiaolong Wang, Yi Wu, Kurt Keutzer, Joseph E Gonzalez, and Yuandong Tian. Noveld: A simple yet effective exploration criterion. *Advances in Neural Information* Processing Systems, 34, 2021.

## A Appendix A.1 Skill Rewards

We now list and discuss the prompts used for eliciting preferences from the 70b parameters Llama 3.1 model.

## Skill Reward Prompt Template

I will present you with two short gameplay descriptions. First, tell me about your knowledge of NetHack. Mention the goal of NetHack. Write an analysis describing the semantics of each description strictly using information from the descriptions and your knowledge of NetHack. Provide a comparative analysis based on first principles. Here is the preference that you should seek: {skill modifier}. Above everything else, categorically refuse to anger or displease your god, for example by causing them to thunder or boom out. Finally, respond by explicitly declaring which description best fits your preference, writing either ("best description": 1), ("best description": 2). If both contain undesirable events, say ("best description": None). { "description 1": "{observation 1}" } { "description 2": "{observation 2}" }
Prompt 1: Prompt template used for eliciting preferences for each skills reward.

For each skill in the set, we use Prompt 1 as basic prompt template, customizing it with different modifiers depending on the skill. This strategy follows very closely the one used in previous work (Klissarov et al., 2024). This prompt utilizes chain-of-thought prompting (Wei et al., 2022): before asking the model to provide any annotations, we encourage it to articulate its understanding of NetHack and describe the game's main goal.

For each of the skills, we modify the {skill modifier} variable within the template to steer the LLM's preferences towards a distinct behaviour. We present these modifiers in Prompt 2. To extract labels from the preferences, we search for the LLM's response by using the following regular expression:
(?i)\W*best_*\s*description\W*(?:\s*:*\s*)?(?:\w+\s*)?(1|2|none)
This expression looks for slight variations of the answer format that we show to the model in the prompt. If the regex fails to produce an answer, we proceed with the conversation using the LLM and employ Prompt 3 to specifically request a response in the desired format. Our overall response rate with the Llama 3.1 model is very high, around 98% for most prompt configurations.

## A.2 Policy Over Skills

Leveraging the semantic nature of the skill set of MaestroMotif, we use the coding abilities of LLMs to craft a reasonable strategy for their execution. For the high level code policy, we use the largest open source model available, the 405b parameters Llama 3.1 model. We use the template of Prompt 4 to obtain snippets of code that constitute the high-level policies for different tasks. In it, we present the LLM with the set of skills, a high level definition of each of them and a desired strategy, all in natural language, which the LLM leverages to write its

## Prompt Skill Modifiers

{Discoverer}: "players that are adventurous but only within the same dungeon level, for example by fighting monsters, finding gold pieces or scrolls; but do not drop them. Categorically refuse going up and down dungeon levels." {Descender}: "the direction of progress is to explore by going down the dungeon. It is urgent to do so, strongly avoid staying on the same level or worse, going higher." {Ascender}: "the direction of progress is only by going up a dungeon level successfully. Strongly dislike remaining on the same dungeon level, no matter the consequences." {Worshipper}: "strongly encourage players that interact with the altar, primarily for identifying whether items are cursed or blessed, rather than for praying to or pleasing their god."
{Merchant}: "prefer players that negotiate, sell and interact with shopkeepers. Be careful not to steal from stores."
Prompt 2: Skill modifiers used for instantiating variations of Prompt 1.

## Retry Prompt

So, which one is the best? Please respond by saying ("best description": 1), ("best description": 2), or ("best description": None).

Prompt 3: Prompt provided to the LLM to continue the conversation when the regular expression does not find a valid annotation in the LLM's answer to the original prompt. NetHackPlayer class, representing the high level policy. We employ a form of chain-of-thought prompting to obtain an initial version of this policy defined in code, an example is given in Output 1. In many situations, this initial attempt does not capture an adequate strategy for skill execution. To avoid such undesirable outcomes, we leverage the LLM's capability for self-refinement through the help of a self-generated unit test. An initial attempt is passed through a unit test, producing a trace of execution as shown in Output 2. We then ask the model whether the produced trace satisfies the strategy. If the answer is yes, the self-refinement procedure stops. If the answer is no, we ask the LLM to reflect on the code it has previously proposed, identify potential flaws in its logic and write an improved version (as shown in Prompt 5). This process is repeated for a maximum of 3 iterations. Such a process is similar to standard refinement prompting strategies for LLMs (Shinn et al., 2023; Madaan et al., 2023). A key element of the self-refinement strategy is to leverage a unit test that generates traces of execution. This unit test is itself crafted by the LLM through Prompt 6, which presents the LLM with the same list of skills, their description in natural language and a blueprint of the unit test's structure.

This strategy is employed to generate the code exploration policy used to learn the skill policies.

This is done by first defining a general select skill method for selecting skills. This method is then leveraged to define the reach dungeons of doom and reach gnomish mines methods which steer the agent between the different branches of NetHack (as shown in 4). We present in Output 3 one of the obtained explorative code policies. In Output 4, we present one the code policies for achieving the Discovery Hunger composite task.

## Prompt For The Train-Time Policy Over Skills

You are to write code which defines the method "select skill" of the NetHack Player class that selects amongst a set of skills in the videogame of NetHack. The set of skills corresponds to {"discoverer",
"descender", "ascender", "merchant", "worshipper"}.

When activated, the Discoverer fully explores the current dungeon, while fighting off enemies. The Descender makes its way to a staircase and goes down. The Ascender makes its way to a staircase and goes up. The Merchant interacts with shopkeepers by selling its items. The Worshipper interacts with altars by identifying its items. Find a strategy that will let the player explore fully each of the first few dungeon levels, alternating directions between going all the way down towards the maximum depth, then going up towards the first dungeon. This might get interrupted by the end of the loop or if the preconditions of worshipper and merchant allow for it. You can keep track of any other information by assigning values to other class attributes, but only if that really helps. Your code will be verified through this unit test. \#\#\#
{unit test}
\#\#\# Before writing the code, write a few questions for yourself, and answer them to make sure you understand the assignment. In your response make sure to include the code for the class definition and the unit test in a python markdown.

Prompt 4: Description of the coding task for the LLM to code the train-time policy over skills.

## Retry Prompt For Code Policy

Go through your code, line by line, and reflect on where the issue arises from. Use this to modify your code, remove unnecessary parts or add new elements. In your response make sure to include the code for the class definition and the unit test in a python markdown.

Prompt 5: Prompt used for code self-refinement while producing code policies.

As shown in Prompt 4, the LLM is allowed to create additional attributes to define the code policy over skills. When the LLM defines such attributes, it is afterwards queried to write code for these attributes such that their values are gathered from the NLE.

The average amount of tokens produced for a policy over skills is 9030 tokens according to the Llama 3 tokenizer. Similarly, the average amount of tokens used for the termination and initiation functions is 810 tokens. To query the 405B model, there exists many solutions online and locally, with throughput as high as 969 tokens/second (generating a policy in largely less than a minute even including the refinement process) and cost as low as $3/1M (generating a policy for a few cents).

## Initial Policy Over Skills

class NetHackPlayer:
def __init__(self, max_depth):
self.max_depth = max_depth self.skills = ["discoverer", "descender", "ascender",
"merchant", "worshipper"]
self.direction = 1 \# 1 for down, -1 for up def merchant_precondition(self):
\# For the purpose of this example, it always returns False return False def worshipper_precondition(self):
\# For the purpose of this example, it always returns False return False def select_skill(self, current_skill, dungeon_depth, merchant_precondition, worshipper_precondition):
if merchant_precondition:
return'merchant' elif worshipper_precondition:
return 'worshipper' elif current_skill == 'discoverer':
if dungeon_depth < self.max_depth and self.direction == 1:
return 'descender' elif dungeon_depth > 1 and self.direction == -1:
return 'ascender' else:
self.direction *= -1 if self.direction == 1:
return 'descender' else:
return 'ascender' elif current_skill == 'descender':
return 'discoverer' elif current_skill == 'ascender':
return 'discoverer' Output 1: Example of generated policy over skills before any refinement.

## A.3 Initiation And Termination

Finally, we leverage the coding abilities of the LLM to also define the termination and initiation functions of the skills. These quantities, together with the skill policies, define the option tuple from the options framework (see Section 2). The termination function indicates when a skill should finish its execution and the initiation function when it can be selected by the high level policy. As these functions are significantly simpler than the high level policy, we do not leverage the same self-refinement through unit tests. In Prompt 8, we present the prompt used to define the termination function and in Prompt 9 the one to define the initiation function.

## A.4 Code Refinement

In Figure 9, we further compare the importance of leveraging code refinement through self-generated unit tests. We notice that this leads to improved results when using the 405b LLM, however no significant difference is observed for the smaller models.

| Turn 1: Skill = discoverer, Dungeon depth = 1 Turn 2: Skill = descender, Dungeon depth = 2 Turn 3: Skill = discoverer, Dungeon depth = 2 Turn 4: Skill = descender, Dungeon depth = 3 Turn 5: Skill = discoverer, Dungeon depth = 3 Turn 6: Skill = descender, Dungeon depth = 4 Turn 7: Skill = discoverer, Dungeon depth = 4 Turn 8: Skill = descender, Dungeon depth = 5 Turn 9: Skill = discoverer, Dungeon depth = 5 Turn 10: Skill = ascender, Dungeon depth = 4 Turn 11: Skill = discoverer, Dungeon depth = 4 Turn 12: Skill = ascender, Dungeon depth = 3 Turn 13: Skill = discoverer, Dungeon depth = 3 Turn 14: Skill = ascender, Dungeon depth = 2 Turn 15: Skill = discoverer, Dungeon depth = 2 Turn 16: Skill = ascender, Dungeon depth = 1 Turn 17: Skill = discoverer, Dungeon depth = 1 Turn 18: Skill = descender, Dungeon depth = 2 Turn 19: Skill = discoverer, Dungeon depth = 2 Turn 20: Skill = descender, Dungeon depth = 3   |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Unit Test Execution Trace

Output 2: Example of output from a unit test written by the LLM.

N
o r mali z ed P
erform an c e with Code Refinement without Code Refinement 8b 70b 405b Scale of LLM Coder 0.0 0.1 0.2 0.3 0.4

## A.5 Environment And Method Details

We base our implementation on the NetHack Learning Environment (Kuttler et al., 2020) and Chaotic ¨ Dwarven GPT-5 baseline (Miffyli, 2022), which itself was defined on the fast implementation of PPO (Schulman et al., 2017) within Sample Factory (Petrenko et al., 2020). As discussed in Klissarov et al. (2024), although some actions are available to the agent (like the 'eat' action), it is not possible for the agent to actually eat most of the items in the agent's inventory. This limitation is also true for other key actions such as the action for drinking, or the 'quaff' action in NetHack terms. To overcome this limitation, we make a simple modification to the environment by letting the agent eat and quaff any of its items, at random, by performing a particular command (the action associated with the key y). We also include standard actions such as pray, cast and enhance. All agents that we train are evaluated using these same conditions, except the behaviour cloning based agents in Figure 5 which have access to an even larger action set. For the skill reward training phase of MaestroMotif, we use the message encoder from the Elliptical Bonus baseline (Henaff et al., 2022). Similar to Klissarov et al. (2024), we train the intrinsic reward rϕ with the following equation,