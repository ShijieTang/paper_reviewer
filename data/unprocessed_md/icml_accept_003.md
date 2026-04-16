# **Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination**

**Kunal Jha** [1] **Wilka Carvalho** [2] **Yancheng Liang** [1] **Simon S. Du** [1] **Max Kleiman-Weiner** [* 1] **Natasha Jaques** [* 1]



**Abstract**


Zero-shot coordination (ZSC), the ability to adapt
to a new partner in a cooperative task, is a critical component of human-compatible AI. While
prior work has focused on training agents to cooperate on a single task, these specialized models do not generalize to new tasks, even if they
are highly similar. Here, we study how reinforcement learning on a **distribution of environ-**
**ments** **with** **a** **single** **partner** enables learning
general cooperative skills that support ZSC with
**many** **new** **partners** **on** **many** **new** **problems** .
We introduce _two_ Jax-based, procedural generators that create billions of solvable coordination
challenges. We develop a new paradigm called
**Cross-Environment** **Cooperation** **(CEC)**, and
show that it outperforms competitive baselines
quantitatively and qualitatively when collaborating with real people. Our findings suggest that
learning to collaborate across many unique scenarios encourages agents to develop general norms,
which prove effective for collaboration with different partners. Together, our results suggest a
new route toward designing generalist cooperative agents capable of interacting with humans
without requiring human data. Code for environment, training, and testing scripts and more can
[be found at https://kjha02.github.io/](https://kjha02.github.io/publication/cross-env-coop)
[publication/cross-env-coop.](https://kjha02.github.io/publication/cross-env-coop)


**1. Introduction**


Humans excel at ad-hoc cooperation, readily adapting to
new partners and environments by attending jointly to relevant objects, reasoning about shared intentions, and playing their role within an implicit collective plan (Tomasello,


*Equal contribution 1Department of Computer Science, University of Washington, Seattle, WA [2] Kempner Institute for the
Study of Natural and Artificial Intelligence, Harvard University, Cambridge, Massachusetts. Correspondence to: Kunal Jha
_<_ kjha@uw.edu _>_ .


_Proceedings_ _of_ _the_ _42_ _[nd]_ _International_ _Conference_ _on_ _Machine_
_Learning_, Vancouver, Canada. PMLR 267, 2025. Copyright 2025
by the author(s).



1999; Kleiman-Weiner et al., 2016; Shum* et al., 2019; Wu*
et al., 2021). This ability to effectively represent collective
tasks allows cooperation to transfer across both partners and
environments. For instance, after mastering a family recipe
with their parents, a novice chef can easily cook the same
dish (and much more) at home with their spouse. These
cognitive mechanisms may be important for building AI
that coordinates in novel scenarios. However, current reinforcement learning (RL) methods have yet to address this
challenge (Wang et al., 2024). Cooperation across partners
on a single problem has been studied, but agents that can
zero-shot coordinate (ZSC) with new partners in unfamiliar
environments will unlock flexible, human-compatible AI
agents in a range of applications: household robots, adaptive
educational assistants, or autonomous vehicles (Ma et al.,
2023; Stone et al., 2010b; Atchley et al., 2024; Dinneweth
et al., 2022; Ribeiro et al., 2021).


Prior work on ZSC has mainly focused on the challenge of
adapting to novel partners, including novel human partners.
Typical RL approaches use methods like population-based
training (PBT) and variations on self-play (SP), approaches
which helped algorithms such as AlphaStar achieve superhuman performance in zero-sum games (Vinyals et al., 2019).
These approaches work by leveraging “partner diversity,”
during training time. They simulate diverse training partners with the idea that these varied experiences will be
sufficient for agents to generalize to human partners (Yan
et al., 2023; Carroll et al., 2020; Strouse et al., 2018; Sarkar
et al., 2023). Although agents trained under this paradigm
often do adapt to new partners, this adaptation is limited
to the single environment they are trained on. PBT, as it is
often deployed (and similar approaches), fails to generalize
when faced with even a slight variation of the same problem.
Thus, current PBT-based approaches must be retrained every
time there is a slight variation in the environment or task.
This is not a recipe which can scale well to the real world
(Yan et al., 2023). If agents can only successfully cooperate
with others on the specific environment they were trained
on, such as a household robot which can only interact with
people in one bedroom rather than the entire house, they
lack a more general notion of cooperation — the ability to
flexibly interact in a broad range of scenarios with many
people.


In this work, we investigate the following question: how can



1


**Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination**










|Col1|✅|✅|
|---|---|---|
||❌|<br>✅|
||❌|✅|



contrasting with prior work which suggests self-play is insufficient for learning general norms for cooperation.



we train AI agents capable of zero-shot collaboration with
_novel partners_ in _novel tasks_ ? To investigate this, we focus
on two main sources of variation during training: partner
diversity and environment diversity. While PBT methods
focus on partner diversity to encourage agents to adapt to
different strategies on a single-task, we hypothesize that
training across diverse environments will instead foster a
richer understanding of the task structure itself, enabling
agents to generalize to novel partners and settings.


To systematically test our hypothesis and isolate the impact
of environment variation compared to partner variation, we
present a new paradigm, **Cross-environment Cooperation**
**(CEC)**, in which agents are trained via self-play (with a
single partner) to cooperate across a wide variety of procedurally generated tasks. To evaluate CEC, we first test on
both a toy environment to illustrate key principles, and then
scale up to the human-AI coordination benchmark, Overcooked. Overcooked is a 2-player, cooperative 2D cooking
game where an AI agent collaborates with an AI or human
partner to prepare a recipe (Carroll et al., 2020; Wu* et al.,
2021; Strouse et al., 2022; Zhao et al., 2022; Sarkar et al.,
2023). We design a procedurally generated Overcooked environment implemented in Jax, enabling high performance
and efficient training speeds of up to _10 million steps per_
_minute on a single GPU_ . Unlike previous work, which studies at most five handcrafted Overcooked levels (Carroll et al.,
2020; Strouse et al., 2018; Zhao et al., 2021; Myers et al.,
2025), our generator creates up to 1 _._ 16 _×_ 10 [17] diverse and
solvable kitchen configurations.


Our experiments intriguingly reveal that by training on diverse _environments_, agents learn to consistently improve
generalization to new _partners_ (see Figure 1). We conduct
extensive simulated and human experiments to evaluate the
performance of CEC agents against state-of-the-art (SOTA)
baselines. Our human study reveals that CEC agents outperform PBT on performance and outperform all methods
on subjective measures of cooperation, suggesting that the
generalized cooperative skills learned through diverse envi


ronmental exposure translate well to human-AI interactions.


The main contributions of this work are:


1. **Cross-Environment** **Cooperation:** A paradigm for
ZSC that replaces partner diversity with environment
diversity. Via procedural environment generation, we
eliminate the need for partner populations while training a single policy to cooperate across diverse tasks.


2. **Algorithmic infrastructure:** (1) Fast Jax-based procedural generation for Overcooked (1 _._ 16 _×_ 10 [17] layouts,
10M steps/min); (2) A toy benchmark isolating environmental vs. partner diversity, demonstrating CEC’s
partner and environment generalization over PBT.


3. We find that **environment diversity outperforms part-**
**ner diversity in cross-play performance** . Through
analyzing generalization to different agents in crossplay, and via Empirical Game-Theoretic analysis, we
show CEC outperforms PBT and other baselines.


4. **Human-AI cooperation insights:** Human studies reveal CEC outperforms PBT in cooperation score and
state-of-the-art methods in qualitative evaluations by
participants, demonstrating its success in realistic cooperative scenarios.


**2. Related Work**


Building agents that can coordinate with novel people
quickly has the potential for broad impacts across robotics
(Breazeal et al., 2005; Sheridan, 2016), digital assistants
(Guo et al., 2024; Poddar et al., 2024; Ying et al., 2024), and
other scientific domains (Ghazimirsaeid et al., 2023; Roche
et al., 2008; Castelfranchi, 2001).


**Self-Play** **(SP)** has been highly successful in zero-sum
games because there is only a single mixed strategy equilibrium (Xia et al., 2018; Silver et al., 2017; Vinyals et al.,
2019; Zhou et al., 2020). However, when applied to cooperative games self-play often converges to a brittle and
inflexible policy that struggles with unfamiliar partners that



2


**Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination**





_Figure 2._ The Dual Destination Problem. In the fixed task (a),
players start in opposite squares and must enter different green
squares from each other to receive a reward. In the procedurally
generated variation (b), the initial positions of the green goal cells
and agents are randomized.


play in novel ways (Strouse et al., 2022; Ma et al., 2023). Intuitively, when training in self-play in a zero-sum game, the
training procedure provides a continuous curriculum of diverse experience as the model is continuously rewarded for
exploiting their opponent. In cooperative games with multiple equilibria, the situation is reversed. Once an equilibrium
is found, neither player has an incentive to unilaterally explore other equilibria. Thus, diverse experiences are not
generated through self-play in this context.


**Population-based training (PBT)** has been explored as a
way to combat the shortcomings of self-play in cooperative games. PBT trains a cooperator agent to learn a best
response to a diverse pool of partners and thus experience
multiple equilibria of the game (Vinyals et al., 2019). PBTbased methods generally outperform self-play in zero-shot
Human-AI coordination. In Overcooked, previous works
(Carroll et al., 2020; Strouse et al., 2022; Zhao et al., 2022;
Sarkar et al., 2023) induced partner diversity at training
time through variations in neural network initializations or
auxiliary loss functions. While PBT allows agents to train
against different strategies for a single task, it is computationally expensive since each task requires first training
a sizable population of agents. In contrast, our approach
systematically varies the environment instead of the partner
distribution, eliminating the need to train multiple policies.


**Procedural Environment Generation.** Recent work has
demonstrated that procedurally generated environments can
improve the generalization of reinforcement learning (RL)
methods in single and multi-agent settings (Cobbe et al.,
2020; 2019; Fontaine et al., 2021; Carion et al., 2019; Chen
et al., 2023; Samvelyan et al., 2023b). These studies show
that exposure to a large and diverse set of samples enhances
generalization (Cobbe et al., 2020). However, they typically
evaluate agents with the same team seen during training,
which doesn’t address the core challenge of zero-shot coordination (ZSC). There has been work in the zero-sum
setting showing environment diversity and partner diversity
are not completely orthogonal axes and prioritizing either



can be suboptimal, and that regret-based sampling enables
agents to efficiently learn to compete in novel scenarios with
novel partners (Samvelyan et al., 2023a; 2024). McKee et al.
(2022) find environment diversity improves agents’ ability
to collaborate on novel levels in the ZSC setting, but they do
not demonstrate the benefits of this approach with ad-hoc
partners such as humans, the effects of combining partner
with environment diversity, the relative benefits of environment diversity compared to partner diversity, or a deeper
qualitative understanding of the effect of environment diversity on learned cooperation strategies. Related to our work,
Ruhdorfer et al. (2024) study unsupervised environment
design (UED) in the context of Overcooked. However, their
work does not prevent the generation of impossible coordination challenges (Dennis et al., 2021; Mediratta et al.,
2023; Li et al., 2023) and their results reveal poor generalization performance to new partners on held-out levels. In
contrast, we show training across many coordination tasks
with the same partner still can improve generalization to
novel partners.


**3. Technical Preliminaries**


A two-player cooperative Markov Game is defined as a
tuple _⟨S, A, T, R, H⟩_, where _S_ are states, _A_ are actions
(shared by both agents), _T_ is the deterministic function
_T_ : _S_ _× A × A_ _→_ _S_, rewards are _R_ : _S_ _× A × A_ _→_ R,
and the game horizon is _H_ . Both the transition and reward functions are assumed to be unknown. We investigate
the problem of training a cooperator policy _πC_ which can
achieve high reward when paired with many different partner policies _πp_ _∼P_, where _P_ represents a distribution of
possible partners, analogous to the distribution of humans
an agent might need to assist. We assume that cooperation
also takes place across many different environments. Here
we define each environment, or cooperation task, as drawn
from a set of possible tasks _m ∼M_ . The task _m_ defines the
initial state distribution, _p_ ( _s_ 0 _|m_ ), but tasks share transition
dynamics _T_ and reward function _R_ . Each different task _m_
is analogous to a new environment layout that may significantly alter the space of effective coordination strategies
that will lead to high reward. We define the score obtained
in the cooperative game obtained by summing the joint pertimestep rewards when the cooperator _πC_ plays partner _πp_
in task _m_ as:



**Cross-Play (XP) Evaluation:** The objective of cross-play
(XP) evaluation is to test how well a cooperator policy _πC_
performs when paired with a novel partner policy _πp_ in an
environment _m_, i.e. evaluate _S_ ( _πp, πC, m_ ). In line with
prior work (Strouse et al., 2022; Zhao et al., 2021; Yan



_S_ ( _πp, πC, m_ ) = E _s_ 0 _∼m,s∼T_
_a_ _[p]_ _∼πp,a_ _[C]_ _∼πC_




- - _[H]_ _R_ ( _st, a_ _[p]_ _t_ _[, a]_ _t_ _[C]_ [)] 
_t_ =0



3


**Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination**



et al., 2023), we simulate novel partners for XP evaluation
by training multiple agents using the same algorithm with
different initial random seeds. Doing so results in different
network initializations and exploration patterns, causing
agents to learn arbitrarily different, yet successful, ways to
coordinate on a problem (Carroll et al., 2020). Collaborating
with novel initializations of the same learning algorithm is
referred to as the Zero-shot Coordination (ZSC) setting (Hu
et al., 2020). We evaluate all agents in ZSC, but also study
cooperation in settings where agents must collaborate with
novel partners that trained with different learning algorithms,
also referred to as “Ad-hoc Teamplay” (Stone et al., 2010a).


**Population-based** **training** **(PBT):** (e.g. (Strouse et al.,
2022; Zhao et al., 2021; Liang et al., 2024)) aim to construct a diverse set of simulated partner policies _P_ =
_{π_ 1 _, π_ 2 _, . . ., πn}_ to simulate various human behaviors, but
focus training on a single task _m_ . Thus, they optimize:


_J_ ( _πC_ ) = E _πi∼P_ [ _S_ ( _πi, πC, m_ )] (1)


For example, Fictitious Co-Play (FCP) is a popular PBTbased method, where a single cooperator agent tries to learn
a best-response policy to a population of self-play agents
at different points of their learning progress (Strouse et al.,
2022).


**4. Cross-Environment Cooperation Improves**
**Partner Generalization**


The core hypothesis in our work is that enhancing the diversity of training tasks will improve agents’ ability to generalize to both new tasks, and new cooperation partners. Our
approach, _Cross-Environment_ _Cooperation_ _(CEC)_, trains
a cooperator policy, _πC_, by sampling diverse tasks from a
broad distribution of tasks, _m_ _∼M_ . To isolate the effect
of task diversity from partner diversity, we use self-play to
train the CEC cooperator policy. The objective of CEC is
thus defined as:


_J_ ( _πC_ ) = E _mi∼M_ [ _S_ ( _πC, πC, mi_ )] (2)


By sampling diverse tasks, CEC encourages the cooperator
policy to generalize its behavior to handle a variety of cooperative scenarios. Unlike PBT, which requires maintaining
a large pool of partner policies, CEC simplifies the training
process because it only requires training a single policy. We
investigate whether using CEC to optimize Eq. 2 can actually provide ZSC performance gains over using PBT to
optimize Eq. 1.


**Dual Destination Game.** To gain intuition for the existing
gap in the ability of multi-agent reinforcement learning
methods to collaborate with partners they have not seen
during training (Lowe et al., 2017), we design a simple
gridworld environment as follows: two agents (red and



_Figure 3._ Evaluation of IPPO and FCP baselines on the Fixed and
Procedurally generated versions of the Dual Destination problem
(error bars show the standard error of the mean). CEC generalizes
better in both cases ( _p <_ 0 _._ 001 for t-tests comparing CEC to both
FCP and IPPO).


blue) begin on the opposing sides of a grid without any
walls. They are both given a +3 reward for moving to
opposing green grid cells. In the basic setup, these grid
cells are equidistant from each other and the agents’ original
starting position, as illustrated in Figure 2(a). The agents
receive a _−_ 1 step cost and can move up, down, left, right,
or stay in place. The state is fully observable.


We hypothesize that self-play (SP) agents will have poor
cross-play (XP) performance in this game, because they
will pick a convention (red agent goes up, blue goes down)
that will not generalize to different SP agents which found
a different convention. PBT methods like FCP may learn
a more general strategy, such as “go up if someone goes
down, and down if they go up.” However, these policies
are still brittle: if the goal location is shifted by even a few
squares or the agents begin in a novel state, the FCP-trained
agent will not have experience training in this environment,
and thus, we hypothesize that it will fail.


To test whether cross-environment training can ameliorate
these shortcomings, we train a CEC model on the Dual
Destination game by randomizing the agent starting and
goal positions at the beginning of a new episode. We show
examples of potential initial states in Figure 2(b). We ensure
that the train task used for the other methods (‘Fixed Task’)
is _held_ _out_ from the CEC train distribution. We then use
IPPO, a SOTA multi-agent SP algorithm (de Witt et al.,
2020), to train agents to optimize the CEC objective in Eq.
2. The CEC model was trained without a population of other
agents or any additional auxiliary losses.


**AI-AI ZSC Evaluation.** We compare the ZSC performance
between SP (IPPO), FCP, and CEC in the Dual Destination tasks shown in Figure 2, and include the task used to
train the SP and FCP methods (‘Fixed Task’), as well as
100 novel procedurally generated tasks. We first measure
how well models generalize cooperation to novel partners
(cross-play performance with novel partners). Second, we
measure how well they generalize to novel partners in novel



100 Procedurally
Generated Tasks


Algorithm



0.8


0.4


0.0



Fixed Task


Algorithm



4


environments).



for 97 steps, equating to a **0.955 normalized reward.** **CEC**
**scored 0.931 normalized reward** with a standard error of
0.013, indicating it underperforms the oracle’s cross play
performance by about 2.5%.Moreover, Figure 3 indicates
that CEC models also generalize to 100 new initial state
configurations _and_ new partners _using_ _the_ _same_ _amount_
_of_ _compute_ as the vanilla single-task IPPO baseline. We
extended our analysis in the Appendix to include partially
observable and multi-task variations of the Dual Destination
problem (Appendices A.2 and A.3, respectively). Our findings consistently show that even with imperfect information
and task uncertainty, **CEC agents outperform population-**
**based and naive self-play methods in cooperating with**
**novel partners on novel problems**, mirroring the trends
observed in the fully observable ZSC case.


These findings suggest that investing in training cooperative
agents on a large distribution of environments is potentially
much more effective than training across a large distribution
of partners, just as prior work has suggested this kind of procedural generalization (or domain randomization) improves
generalization in the single-agent setting in robotics (Jakobi,
1997; Sadeghi & Levine, 2016; Tobin et al., 2017).


**Procedurally** **Generated** **Overcooked.** To test whether
these findings replicate in a more scaled-up scenario, we
extend the Overcooked environment from the JaxMARL
project to support a wider variety of levels (Rutherford et al.,
2023).


Previous work on Overcooked has focused on five layouts
that possess diverse coordination challenges (Carroll et al.,
2020; Strouse et al., 2022; Zhao et al., 2021; Yan et al.,
2023). For our new environment, we uniformly sample
the wall structure from each of these five layouts, then randomly generate features like goals, plates, pots, and onions
within the grid. Doing so structures the generation process
and introduces variable complexity depending on where the



objects are sampled. We ensure tasks remain solvable by
placing essential objects in reachable areas from the layouts
in Figure 4 and detailed in A.1. We introduce additional
environment diversity by sampling additional goal locations,
plate piles, pots, and onion piles on unoccupied walls. The
grids are also rotated randomly to encourage better representation learning. For example, if an agent learned to complete
a task on a very wide layout, it should be able to do that
same task if it is rotated vertically by performing a simple
mapping from the left/right actions to up/down, and vice
versa.


Our procedural generator creates new coordination challenges in Overcooked, shown in Figure 5. Moreover, Jax
allows us to run the entire training and evaluation pipeline,
from the environment generation to the neural network updating of agents, at _10 million steps per minute on a single_
_GPU_ . We leverage this speed to train CEC agents, and all
other baselines, for 3 billion steps. By standardizing the
training duration for all algorithms, we can directly compare
whether it is better to invest compute in training on more
environments or in methods like PBT.


**5. Experiments**


From this point on, we will refer to agents trained to cooperate across multiple coordination challenges by optimizing
Eq. 2 as CEC, agents trained on a single task as ST, the
self-play performance of the agents as SP, and the cross-play
performance of the agents as XP. Our primary focus is on
the XP setting, which measures how well agents collaborate
with novel partners on tasks they have never seen before.


When evaluating AI-AI and Human-AI coordination, we
aim to answer the following research questions:


1. Is increasing environment diversity more effective than
increasing partner diversity for ZSC?


2. Compared to ST methods, how well do CECs general


5


**Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination**



has to generalize to new partners and environments. Towards mitigating the bias against our method and answering
Research Question 3, we study the effects of trading off
generalization for specialization in CEC models. After training a single CEC agent on a distribution of environments
created by the procedural generator, we create five copies
of the model, one corresponding to each of the original five
layouts in Figure 4 which were held out during training. For
each copy of the CEC agent, we perform an additional 100
million steps of training on a single layout with a reduced
learning rate, again in self-play using IPPO. We call this
approach “CEC-Finetune (CEC-FT).”


**Human** **Studies.** Due to the expense of human evaluations, our Human-AI evaluations only looked at the two
most challenging grids of the original five, _Counter Circuit_
and _Coordination Ring_ . We recruit 80 human participants
for our study using Prolific, where 40 participants complete each layout. Our study follows a protocol approved
by our university’s IRB. During the study, each user plays
multiple rounds of Overcooked with a partner via a web
interface, where in each round the partner is controlled
by one of the models, in randomized order. We sample a
new model trained with a different random seed for each
of the game rounds. Subjects played with agents for 200
timesteps and the entire experiment took approximately 30
minutes. After each round, the user answered questions
about their subjective experience playing with the agent
using a Likert scale (Likert, 1932). This survey allows us to
quantify the factors that most heavily influence overall performance and users’ preferences for coordination partners.
All of our Human-AI experiments and surveys utilized the
[NiceWebRL Python package (https://github.com/](https://github.com/wcarvalho/nicewebrl)
[wcarvalho/nicewebrl), which leverages Jax’s paral-](https://github.com/wcarvalho/nicewebrl)
lelizability to efficiently crowd-source participant data on
reinforcement learning environments.


**6. Results**


**Q1:** **Is increasing environment diversity more effective**
**than** **increasing** **partner** **diversity** **of** **ZSC?** Figure 6
shows the XP performance of the baseline models averaged across the five original Overcooked layouts. Although
ST approaches are capable of adapting to partner strategies during training time, they fall short in XP compared
to CEC models. Fine-tuning CEC agents on a single grid
layout resulted in a substantial improvement in cross-play
generalization performance. While it is unsurprising that
training in the test environment helps performance, even
without training in the test layout or using any strategies to
enhance partner diversity, CEC outperforms all baselines
in terms of XP. The success of CEC and CEC-Finetune in
XP indicates the benefits of both broad pre-training with
task-specific adaptation. More importantly, we obtain evidence to support an affirmative answer to research question



100 Procedurally

Generated Grids


Algorithm



160


80


0



5 Heldout Grids


Algorithm



_Figure 6._ Evaluation of baselines on (left) 5 original Overcooked
layouts vs. (right) 100 procedurally generated held-out layouts
(standard error bars). Single-task methods and PBT struggle in
both settings, while CEC agents generalize effectively. Finetuning
CEC on a single grid improves XP performance on original layouts
(outperforming FCP and IPPO; _p_ _<_ 0 _._ 01, t-test) but reduces
generalization on novel layouts. CEC significantly surpasses all
baselines in procedural generalization tasks ( _p <_ 0 _._ 0001, t-test).


ize cooperative strategies to novel environments?


3. How close can CEC come to state-of-the-art performance with novel humans for a single task it has never
seen before through environment diversity alone?


**Baselines.** For single-task baselines, we train six seeds for
three different agents for each of the original five layouts in
Figure 4, one with vanilla IPPO (de Witt et al., 2020), one
with FCP (Strouse et al., 2022), and one with Efficient Endto-End Training (E3T) (Yan et al., 2023). E3T is the state-ofthe-art algorithm for single-task ZSC in Overcooked. It is
a SP method that achieves high XP performance by adding
randomness to one of the partner’s policies during training,
and training an auxiliary network which tries to predict
the actions of other agents in the world. We test CEC’s
cross-play performance with novel partners on each of the
single-tasks FCP and E3T were trained on.


**Evaluation** **Protocol.** Following prior work (Zhao et al.,
2021; Carroll et al., 2020; Yan et al., 2023; Strouse et al.,
2022), we first evaluate the ability of all models to generalize
to new partners on the five original Overcooked Layouts
4. Note that we hold out those five layouts from the CEC
generator, so that when we evaluate CEC on these layouts
we are able to test generalization across both partners and
tasks. Second, we introduce an additional evaluation setting
where we have the Overcooked procedural environment
generator create 100 coordination challenges that neither
the ST baselines nor any of the CEC agents have seen during
training and assess how well the different approaches can
generalize to both novel partners and novel environments.
We train six seeds for each type of agent. The architectural
details are included in A.7.


**Cross-Environment Cooperation as a Pre-training Step.**
Holding out the original five layouts from the CEC generator puts CEC at a disadvantage compared to ST baselines,
which only have to generalize to novel partners, while CEC



6


**Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination**


1: increasing environment diversity can actually be more
effective at improving cross- _partner_ generalization, than
training on a single environment with many partners.



One challenge with assessing coordination using the singletask levels is that it does not control for the possibility that
all seeds just happen to play the same strategy (e.g., “always
rotate clockwise.”). This would result in high cross play
scores without the ability to generalize to novel partners.
Therefore, we compute cross-algorithm cooperation scores,
shown in Figures 21 and 22, to control for this possibility
and evaluate agents in the Ad-hoc Teamplay setting. Using this cross-algorithm performance matrix to represent a
meta-game, we conduct an **empirical game-theoretic anal-**
**ysis** where players choose model types rather than actions.
Essentially, players in the meta-game choose a model (like
CEC vs. FCP) in order to maximize cross-play score. This
can also be see as treating Figures 21 and 22 as a payoff
matrices, and analyzing the equilibria of these games. This
approach allows us to interpret how a simulated population of agents would adopt different strategies over time,
revealing attractors, equilibria, and cyclic behaviors that
may not be apparent from simple win rates or Elo ratings
(Wellman et al., 2024). Following the approach outlined in
(Tuyls et al., 2018; Serrino* et al., 2019), Figure 8 shows
the gradient of the replicator dynamic in these meta-games
as a way of assessing the dynamics and equilibrium of the
meta-game. For both the five original tasks and the 100
procedurally generated tasks, the direction of the gradient
is towards either CEC or CEC-Finetune. This shows that
CEC trained models generalize robustly across different
algorithms.


**Q2:** **Compared** **to** **single-task** **methods,** **how** **well** **do**
**CECs** **generalize** **their** **cooperative** **strategies** **to** **novel**
**environments?** Similar to the Toy environment, the evaluation on the held-out Overcooked layouts created by the
procedural environment generator, as depicted in Figure 6,


100



0



Training Progress (100M steps)



SP Training
XP Asymm Advantages
XP Forced Coord



XP Coord Ring
XP Cramped Room
XP Counter Circuit



_Figure 7._ CEC SP Training Performance compared to XP Performance on 5 held-out levels. Despite the distribution sampling each
layout predicate at uniform, CEC gets better at different layouts at
different rates as it consistently improves across all tasks.



_Figure 8._ Empirical game-theoretic evaluation of cross-algorithm
play on the (a) five original and (b) 100 procedurally generated
Overcooked tasks. Arrows show the gradient of the replicator
dynamic on the cross-algorithm meta-game. Vectors flow towards
CEC and CEC-Finetune indicate they are likely equilibria.


reveals that FCP and E3T completely fail to generalize to
novel tasks, receiving 0 reward on any of the 100 test tasks.
While both methods try to increase partner strategy coverage
in self-play by adding entropy to the partner policy, they fail
to generalize to similar, procedurally generated layouts due
to insufficient _environment coverage_ . Without encountering
the same task in diverse scenarios, agents learn low-level
sequences (e.g., ”move to the third cell from the left and interact”) rather than higher-level task structures (e.g., ”cook
onions and deliver them”), hindering environment generalization.


We also assess how well humans can collaborate with ST
single-task agents playing a layout they have never seen
during training. We call the IPPO and FCP versions of these
agents IPPO _[−]_ and FCP _[−]_ respectively. Just as in simulation,
we find that models only optimized to cooperate on one
task cannot generalize to new tasks with ad-hoc partners
(Figure 9), and are consistently rated the most frustrating to
play with by humans (Figure 27). Taken together with the
FCP cross-play performance, this provides strong evidence
that ST methods cannot generalize at all to novel, similar
cooperative settings whereas CEC agents can.


Interestingly, Figure 6 shows CEC-Finetune did not perform
as well as vanilla CEC on the novel procedurally generated
layouts. This highlights a tension between generality and
specialization in agent training. While CEC fine-tuning
demonstrates that pre-training on diverse problems provides
a solid foundation for learning to collaborate on a specific
problem with new partners (Figure 6), it also results in a
loss of the ability to generalize to novel levels.


In Appendix A.4, we explored whether combining partner
and environmental diversity enhances agents’ ZSC abilities.
We trained E3T agents within the CEC framework and evaluated their XP performance on unseen tasks with unfamiliar



7


**Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination**



2


0



2


0



Algorithm



_Figure 9._ (Top) Average success rates of algorithms cooperating
with ad-hoc human partners on _Counter Circuit_ and _Coordination_
_Ring_, with standard error bars. CEC outperforms PBT methods and
approaches E3T’s performance, despite only training on diverse
layouts. Using a 2-sided t-test, CEC significantly outperforms
FCP ( _p_ _<_ 0 _._ 001, t-test). (Bottom) Human ratings of algorithms’
cooperative ability across 7 metrics, averaged over _Counter Circuit_
and _Coordination Ring_ evaluations, with standard error bars. CEC
and CEC-Finetune are preferred partners despite lower rewards.
CEC-Finetune significantly outperforms FCP ( _p_ _<_ 0 _._ 01, t-test)
and E3T ( _p <_ 0 _._ 01, t-test).


partners. As Figure 16 illustrates, introducing novel partners
while learning in a constantly changing environment _did not_
_improve_ _agents’_ _ZSC_ _capabilities_ . This suggests further
research is needed to determine how to best integrate these
two forms of diversity to realize their combined benefits.


Last, we analyze XP performance at different points in the
learning trajectory of CEC agents. Figure 7 shows the CEC
SP training reward and the corresponding XP evaluation
reward on each of the five layouts at the same point in
training. We find that CEC XP performance improves at
different rates across layouts. This can partly be explained
by the diversity in optimal coordination strategies across
the original five layouts (Figure 4). In both _Cramped Room_
and _Coordination Ring_, the spatial constraints lead agents
to discover cooperative strategies more quickly. However,
for layouts such as _Counter_ _Circuit_, where agents have
many options for where to pass items along the middle
border, whether to rotate around the middle wall clockwise
or counter clockwise, etc., CEC’s XP performance improves
slower but increases steadily. It could potentially be driven
even higher with more training.


**Q3:** **How close can CEC come to state-of-the-art perfor-**
**mance with novel humans for a single task it has never**
**seen before through environment diversity alone?** To
answer this question, we analyzed how well CEC agents
can collaborate with new people on novel tasks under the
Ad-hoc Teamplay setting. As Figure 9 demonstrates, when
evaluated in terms of score in the cooperation task, on levels that ST methods have seen during training, CEC and
CEC-finetune are able to collaborate better than IPPO and



FCP models, but fall short of E3T, which is the state-ofthe-art method for single level ad-hoc single level ad-hoc
collaboration performance with humans. While reward is
an important metric to optimize, past work (Carroll et al.,
2020) has shown that it can obfuscate frustrating behaviors,
such as forcing the human to adapt to the agent’s strategy, or completing the task effectively but independently,
while not truly cooperating with the human. Therefore we
must also consider human subjective preferences as equally
important in assessing an agents’ human-AI cooperation
abilities. In Figure 9, we show that across 7 different metrics of human enjoyment, CEC consistently outperforms
all baselines, with CEC-FT significantly surpassing E3T
( _p <_ 0 _._ 01 _, t_ = 3 _._ 1233 _)._


So how is it that CEC obtains such high ratings, but less
than SOTA score in the tasks? A closer look at users’ qualitative assessments of playing with the different models in
Figures 27 gives us a partial explanation for why this may
be the case. We see that CEC models were ranked highest
by users in terms of their ability to adapt to the participant’s
behaviors, while also being the least frustrating to work with
and most consistent in their actions. Moreover, CEC was
rated most enjoyable to cooperate with and best in terms of
ability to coordinate. We synthesize the correlation between
different users’ qualitative ratings to establish a hierarchy
of factors most relevant to human’s perception of effective
cooperation in Overcooked, and show our findings in Figure 10. We report a Cronbach’s alpha score of _≈_ 0 _._ 874, and





_Figure 10._ Heatmap depicting the Pearson’s Correlation Coefficient between users’ qualitative rating of different AI collaborators
in Overcooked. By clustering based on correlation degree, we
can categorize the factors most indicative of what makes a good
collaborator. We report a Cronbach’s alpha score of _≈_ 0 _._ 874

.



8


50

25

0



Algorithm



**Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination**


**Acknowledgments**



_Figure 11._ Average number of collisions between humans and AI
partners on _Counter Circuit_ and _Coordination Ring_, with standard
error bars shown. CEC achieves the lowest average collision rate.


rics, in addition to reward, as an indicator of CEC’s success
compared to other baselines.


We also note that CEC’s increased proclivity to adapt its
behaviors to people can lead to lower rewards if participants
are playing a suboptimal strategy. On the other hand, E3T,
which has seen the environment during training it is evaluated on, is rated as less adaptive (see Figures 28 and 29),
and instead may focus too much on optimizing score. In
playing with the agents, we observed that CEC is much
better at avoiding collisions and getting out of the player’s
way, even though this may lead to less reward. Figure 11
quantifies the frequency of collisions between humans and
different AI collaborators. The results indicate that, despite
achieving lower rewards than the SOTA method E3T, CEC
has fewer collisions with humans on average. This aligns
with participants’ subjective assessments of model behavior
(Figure 27). We hypothesize that this **collision-avoidance**
**behavior reflects a general norm learned by CEC**, which
enhances performance across a wide range of cooperative
tasks. This supports our hypothesis that cross-environment
training helps agents learn more generalized cooperative
strategies that work not only in a range of environments, but
with a range of partners.


**7. Discussion**


We introduce Cross-environment Cooperation, a novel approach to ZSC in multi-agent reinforcement learning. Crossenvironment training helps agents learn more generalized
cooperative strategies that work not only in a range of environments but with a range of partners. Although trained in
self-play, they zero-shot coordinate with new partners _and_
in new environments never seen during training. We show
that CEC succeeds in cooperating with current state-of-theart baselines and with human users. These results challenge
the prevailing wisdom that self-play is insufficient for ZSC
and Ad-hoc Teamplay in cooperative games. Future work
will explore composing CEC with other training algorithms.
For cooperative AI agents to work effectively with us in
our offices and homes, agents must rapidly adapt to a wide
range of diverse environments and partners.



We would like to thank the Cooperative AI Foundation, the
Foresight Institute, the Amazon+UW Science Gift Hub, the
UW Tsukuba NVIDIA Amazon Cross-Pacific AI Initiative
and the Sony Resarch Award Program for their generous
support of our research. This work was additionally supported by NSF CCF 2212261, NSF IIS 2229881, the Alfred
P. Sloan Research Fellowship, and the Schmidt Sciences
AI 2050 Fellowship. We are also grateful for this work
being supported by a gift from the Chan Zuckerberg Initiative Foundation to establish the Kempner Institute for
the Study of Natural and Artificial Intelligence. Lastly, we
would like to express our gratitude to our colleagues at the
Social Reinforcement Lab and the Computational Minds
and Machines Lab for inspiring conversations and insights
during the course of this project.


**Impact Statement**


This paper presents a novel approach to training agents to
collaborate with people it has not seen before. As such, it
has the ability to improve the lives of many applications
as household robotics, autonomous driving, or language
assistants. To provide evidence for the merits of our approach, we conducted studies with human participants. We
made sure to follow all ethical practices described under the
instructions of our university’s Institutional Review Board
(IRB), including but not limited to compensating participants at minimum wage and receiving informed consent. As
a final point, while the work in this paper described how to
build a general cooperative AI, it is not clear or well studied
if a similar approach could be used to create a machine
capable of harming humans it has not seen before in a wide
range of cases. Future works should try to establish whether
this is the case and how to mitigate these potential risks.


**References**


Nicegui — nicegui.io. https://nicegui.io. [Accessed 26-092024].


Atchley, P., Pannell, H., Wofford, K., Hopkins, M., and
Atchley, R. A. Human and AI collaboration in the
higher education environment: opportunities and concerns. _Cognitive_ _Research:_ _Principles_ _and_ _Implica-_
_tions_, 9(1):20, April 2024. ISSN 2365-7464. doi:
10.1186/s41235-024-00547-9. URL [https://doi.](https://doi.org/10.1186/s41235-024-00547-9)
[org/10.1186/s41235-024-00547-9.](https://doi.org/10.1186/s41235-024-00547-9)


Bonnet, C., Luo, D., Byrne, D., Surana, S., Abramowitz, S.,
Duckworth, P., Coyette, V., Midgley, L. I., Tegegn, E.,
Kalloniatis, T., Mahjoub, O., Macfarlane, M., Smit, A. P.,
Grinsztajn, N., Boige, R., Waters, C. N., Mimouni, M. A.,
Sob, U. A. M., de Kock, R., Singh, S., Furelos-Blanco, D.,



9


**Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination**



Le, V., Pretorius, A., and Laterre, A. Jumanji: a diverse
suite of scalable reinforcement learning environments in
jax, 2024. [URL https://arxiv.org/abs/2306.](https://arxiv.org/abs/2306.09884)
[09884.](https://arxiv.org/abs/2306.09884)


Breazeal, C., Kidd, C., Thomaz, A., Hoffman, G., and
Berlin, M. Effects of nonverbal communication on
efficiency and robustness in human-robot teamwork.
pp. 708  - 713, 09 2005. ISBN 0-7803-8912-3. doi:
10.1109/IROS.2005.1545011.


Carion, N., Synnaeve, G., Lazaric, A., and Usunier, N. A
structured prediction approach for generalization in cooperative multi-agent reinforcement learning, 2019. URL
[https://arxiv.org/abs/1910.08809.](https://arxiv.org/abs/1910.08809)


Carroll, M., Shah, R., Ho, M. K., Griffiths, T. L., Seshia,
S. A., Abbeel, P., and Dragan, A. On the utility of learning
about humans for human-ai coordination, 2020. URL
[https://arxiv.org/abs/1910.05789.](https://arxiv.org/abs/1910.05789)


Carvalho, w. Nicewebrl: a framework for comparing humans and ai across many domains, 2025. [URL https:](https://github.com/wcarvalho/nicewebrl)
[//github.com/wcarvalho/nicewebrl.](https://github.com/wcarvalho/nicewebrl)


Castelfranchi, C. The theory of social functions:
challenges for computational social science and
multi-agent learning. _Cognitive_ _Systems_ _Re-_
_search_, 2(1):5–38, 2001. ISSN 1389-0417. doi:
https://doi.org/10.1016/S1389-0417(01)00013-4.
URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/S1389041701000134)
[science/article/pii/S1389041701000134.](https://www.sciencedirect.com/science/article/pii/S1389041701000134)


Chen, Y., Tang, C., Tian, R., Li, C., Li, J., Tomizuka, M.,
and Zhan, W. Quantifying agent interaction in multiagent reinforcement learning for cost-efficient generalization, 2023. [URL https://arxiv.org/abs/2310.](https://arxiv.org/abs/2310.07218)
[07218.](https://arxiv.org/abs/2310.07218)


Cobbe, K., Klimov, O., Hesse, C., Kim, T., and Schulman,
J. Quantifying generalization in reinforcement learning.
In Chaudhuri, K. and Salakhutdinov, R. (eds.), _Proceed-_
_ings_ _of_ _the_ _36th_ _International_ _Conference_ _on_ _Machine_
_Learning_, volume 97 of _Proceedings of Machine Learn-_
_ing Research_, pp. 1282–1289. PMLR, 09–15 Jun 2019.
[URL https://proceedings.mlr.press/v97/](https://proceedings.mlr.press/v97/cobbe19a.html)
[cobbe19a.html.](https://proceedings.mlr.press/v97/cobbe19a.html)


Cobbe, K., Hesse, C., Hilton, J., and Schulman, J. Leveraging procedural generation to benchmark reinforcement
learning, 2020. URL [https://arxiv.org/abs/](https://arxiv.org/abs/1912.01588)
[1912.01588.](https://arxiv.org/abs/1912.01588)


de Witt, C. S., Gupta, T., Makoviichuk, D., Makoviychuk,
V., Torr, P. H. S., Sun, M., and Whiteson, S. Is independent learning all you need in the starcraft multiagent challenge? _CoRR_, abs/2011.09533, 2020. URL
[https://arxiv.org/abs/2011.09533.](https://arxiv.org/abs/2011.09533)


10



Dennis, M., Jaques, N., Vinitsky, E., Bayen, A., Russell, S.,
Critch, A., and Levine, S. Emergent complexity and zeroshot transfer via unsupervised environment design, 2021.
[URL https://arxiv.org/abs/2012.02096.](https://arxiv.org/abs/2012.02096)


Dinneweth, J., Boubezoul, A., Mandiau, R., and Espie,´
S. Multi-agent reinforcement learning for autonomous
vehicles: a survey. _Autonomous Intelligent Systems_, 2(1):
27, November 2022. ISSN 2730-616X. doi: 10.1007/
s43684-022-00045-z. [URL https://doi.org/10.](https://doi.org/10.1007/s43684-022-00045-z)
[1007/s43684-022-00045-z.](https://doi.org/10.1007/s43684-022-00045-z)


Fontaine, M. C., Hsu, Y.-C., Zhang, Y., Tjanaka, B., and
Nikolaidis, S. On the importance of environments
in human-robot coordination, 2021. URL [https://](https://arxiv.org/abs/2106.10853)
[arxiv.org/abs/2106.10853.](https://arxiv.org/abs/2106.10853)


Ghazimirsaeid, S. S., Jonban, M. S., Mudiyanselage,
M. W., Marzband, M., Martinez, J. L. R., and Abusorrah, A. Multi-agent-based energy management of
multiple grid-connected green buildings. _Journal_ _of_
_Building_ _Engineering_, 74:106866, 2023. ISSN 23527102. doi: https://doi.org/10.1016/j.jobe.2023.106866.
URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/S2352710223010458)
[science/article/pii/S2352710223010458.](https://www.sciencedirect.com/science/article/pii/S2352710223010458)


Guo, T., Chen, X., Wang, Y., Chang, R., Pei, S., Chawla,
N. V., Wiest, O., and Zhang, X. Large language model
based multi-agents: A survey of progress and challenges, 2024. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2402.01680)
[2402.01680.](https://arxiv.org/abs/2402.01680)


Henrich, J., Heine, S. J., and Norenzayan, A. The weirdest
people in the world? _Behavioral and Brain Sciences_, 33
(2–3):61–83, 2010. doi: 10.1017/S0140525X0999152X.


Hu, H., Lerer, A., Peysakhovich, A., and Foerster, J. “Otherplay” for zero-shot coordination. In III, H. D. and Singh,
A. (eds.), _Proceedings of the 37th International Confer-_
_ence on Machine Learning_, volume 119 of _Proceedings_
_of Machine Learning Research_, pp. 4399–4410. PMLR,
[13–18 Jul 2020. URL https://proceedings.mlr.](https://proceedings.mlr.press/v119/hu20a.html)
[press/v119/hu20a.html.](https://proceedings.mlr.press/v119/hu20a.html)


Jakobi, N. Evolutionary robotics and the radical envelopeof-noise hypothesis. _Adaptive Behavior_, 6(2):325–368,
[1997. doi: 10.1177/105971239700600205. URL https:](https://doi.org/10.1177/105971239700600205)
[//doi.org/10.1177/105971239700600205.](https://doi.org/10.1177/105971239700600205)


Kleiman-Weiner, M., Ho, M. K., Austerweil, J. L.,
Michael L, L., and Tenenbaum, J. B. Coordinate to cooperate or compete: abstract goals and joint intentions
in social interaction. In _Proceedings of the 38th Annual_
_Conference of the Cognitive Science Society_, 2016.


Li, W., Varakantham, P., and Li, D. Generalization through
diversity: Improving unsupervised environment design.


**Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination**



In Elkind, E. (ed.), _Proceedings of the Thirty-Second In-_
_ternational_ _Joint_ _Conference_ _on_ _Artificial_ _Intelligence,_
_IJCAI-23_, pp. 5411–5419. International Joint Conferences on Artificial Intelligence Organization, 8 2023. doi:
10.24963/ijcai.2023/601. [URL https://doi.org/](https://doi.org/10.24963/ijcai.2023/601)
[10.24963/ijcai.2023/601.](https://doi.org/10.24963/ijcai.2023/601) Main Track.


Liang, Y., Chen, D., Gupta, A., Du, S. S., and Jaques,
N. Learning to cooperate with humans using generative
agents. _arXiv preprint arXiv:2411.13934_, 2024.


Likert, R. A technique for the measurement of attitudes.
_Archives of Psychology_, 140:1–55, 1932.


Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., and Mordatch, I. Multi-agent actor-critic for mixed cooperativecompetitive environments. _CoRR_, abs/1706.02275, 2017.
[URL http://arxiv.org/abs/1706.02275.](http://arxiv.org/abs/1706.02275)


Lu, C., Beukman, M., Matthews, M., and Foerster, J.
_JaxLife:_ _An_ _Open-Ended_ _Agentic_ _Simulator_, volume
ALIFE 2024: Proceedings of the 2024 Artificial Life
Conference of _Artificial_ _Life_ _Conference_ _Proceed-_
_ings_ . July 2024. doi: 10.1162/isal ~~a~~ ~~0~~ 0770. URL
[https://doi.org/10.1162/isal_a_00770.](https://doi.org/10.1162/isal_a_00770)

~~e~~ print: https://direct.mit.edu/isal/proceedingspdf/isal2024/36/47/2461075/isal ~~a~~ ~~0~~ 0770.pdf.


Ma, M., Liu, J., Sokota, S., Kleiman-Weiner, M., and Foerster, J. N. Learning intuitive policies using action features.
In _International_ _Conference_ _on_ _Machine_ _Learning_, pp.
23358–23372. PMLR, 2023.


Matthews, M., Beukman, M., Ellis, B., Samvelyan, M.,
Jackson, M., Coward, S., and Foerster, J. Craftax: A
lightning-fast benchmark for open-ended reinforcement
learning, 2024. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2402.16801)
[2402.16801.](https://arxiv.org/abs/2402.16801)


McKee, K. R., Leibo, J. Z., Beattie, C., and Everett,
R. Quantifying the effects of environment and population diversity in multi-agent reinforcement learning.
_Autonomous_ _Agents_ _and_ _Multi-Agent_ _Systems_, 36(1):
21, March 2022. ISSN 1573-7454. doi: 10.1007/
s10458-022-09548-8. [URL https://doi.org/10.](https://doi.org/10.1007/s10458-022-09548-8)
[1007/s10458-022-09548-8.](https://doi.org/10.1007/s10458-022-09548-8)


Mediratta, I., Jiang, M., Parker-Holder, J., Dennis, M.,
Vinitsky, E., and Rocktaschel,¨ T. Stabilizing unsupervised environment design with a learned adversary. In
Chandar, S., Pascanu, R., Sedghi, H., and Precup, D.
(eds.), _Proceedings of The 2nd Conference on Lifelong_
_Learning Agents_, volume 232 of _Proceedings of Machine_
_Learning_ _Research_, pp. 270–291. PMLR, 22–25 Aug
[2023. URL https://proceedings.mlr.press/](https://proceedings.mlr.press/v232/mediratta23a.html)
[v232/mediratta23a.html.](https://proceedings.mlr.press/v232/mediratta23a.html)


11



Myers, V., Ellis, E., Levine, S., Eysenbach, B., and Dragan, A. Learning to assist humans without inferring
rewards, 2025. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2411.02623)
[2411.02623.](https://arxiv.org/abs/2411.02623)


Nikulin, A., Kurenkov, V., Zisman, I., Agarkov, A., Sinii,
V., and Kolesnikov, S. Xland-minigrid: Scalable metareinforcement learning environments in jax, 2024. URL
[https://arxiv.org/abs/2312.12044.](https://arxiv.org/abs/2312.12044)


Poddar, S., Wan, Y., Ivison, H., Gupta, A., and Jaques,
N. Personalizing reinforcement learning from human
feedback with variational preference learning, 2024. URL
[https://arxiv.org/abs/2408.10075.](https://arxiv.org/abs/2408.10075)


Rabinowitz, N., Perbet, F., Song, F., Zhang, C., Eslami,
S. A., and Botvinick, M. Machine theory of mind. In
_International conference on machine learning_, pp. 4218–
4227. PMLR, 2018.


Ribeiro, T., Gonc¸alves, F., Garcia, I. S., Lopes, G., and
Ribeiro, A. F. Charmie: A collaborative healthcare and
home service and assistant robot for elderly care. _Applied_
_Sciences_, 11(16):7248, 2021.


Roche, B., Guegan,´ J.-F., and Bousquet, F. Multiagent systems in epidemiology: a first step for computational biology in the study of vector-borne disease transmission. _BMC_ _Bioinformatics_, 9(1):435,
October 2008. ISSN 1471-2105. doi: 10.1186/
1471-2105-9-435. URL [https://doi.org/10.](https://doi.org/10.1186/1471-2105-9-435)
[1186/1471-2105-9-435.](https://doi.org/10.1186/1471-2105-9-435)


Ruhdorfer, C., Bortoletto, M., Penzkofer, A., and Bulling,
A. The overcooked generalisation challenge. 2024. URL
[https://arxiv.org/abs/2406.17949.](https://arxiv.org/abs/2406.17949)


Rutherford, A., Ellis, B., Gallici, M., Cook, J., Lupu, A.,
Ingvarsson, G., Willi, T., Khan, A., de Witt, C. S., Souly,
A., Bandyopadhyay, S., Samvelyan, M., Jiang, M., Lange,
R. T., Whiteson, S., Lacerda, B., Hawes, N., Rocktaschel,
T., Lu, C., and Foerster, J. N. Jaxmarl: Multi-agent rl
environments in jax. _arXiv preprint arXiv:2311.10090_,
2023.


Sadeghi, F. and Levine, S. (cad)$ˆ2$rl: Real singleimage flight without a single real image. _CoRR_,
abs/1611.04201, 2016. URL [http://arxiv.org/](http://arxiv.org/abs/1611.04201)
[abs/1611.04201.](http://arxiv.org/abs/1611.04201)


Samvelyan, M., Khan, A., Dennis, M., Jiang, M., ParkerHolder, J., Foerster, J., Raileanu, R., and Rocktaschel, T.¨
Maestro: Open-ended environment design for multi-agent
reinforcement learning. _arXiv preprint arXiv:2303.03376_,
2023a.


**Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination**



Samvelyan, M., Khan, A., Dennis, M., Jiang, M., ParkerHolder, J., Foerster, J., Raileanu, R., and Rocktaschel,¨
T. Maestro: Open-ended environment design for multiagent reinforcement learning, 2023b. [URL https://](https://arxiv.org/abs/2303.03376)
[arxiv.org/abs/2303.03376.](https://arxiv.org/abs/2303.03376)


Samvelyan, M., Paglieri, D., Jiang, M., Parker-Holder,
J., and Rocktaschel,¨ T. Multi-agent diagnostics for
robustness via illuminated diversity. _arXiv_ _preprint_
_arXiv:2401.13460_, 2024.


Sarkar, B., Shih, A., and Sadigh, D. Diverse conventions for
human-ai collaboration, 2023. [URL https://arxiv.](https://arxiv.org/abs/2310.15414)
[org/abs/2310.15414.](https://arxiv.org/abs/2310.15414)


Serrino*, J., Kleiman-Weiner*, M., Parkes, D. C., and
Tenenbaum, J. B. Finding friend and foe in multi-agent
games. In _Advances in Neural Information Processing_
_Systems_, volume 32, 2019.


Sheridan, T. B. Human–robot interaction: Status and challenges. _Human Factors_, 58(4):525–532, 2016. doi: 10.
1177/0018720816644364. [URL https://doi.org/](https://doi.org/10.1177/0018720816644364)
[10.1177/0018720816644364.](https://doi.org/10.1177/0018720816644364) PMID: 27098262.


Shum*, M., Kleiman-Weiner*, M., Littman, M. L., and
Tenenbaum, J. B. Theory of minds: Understanding behavior in groups through inverse planning. In _Proceedings of_
_the AAAI conference on artificial intelligence_, volume 33,
pp. 6163–6170, 2019.


Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou,
I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M.,
Bolton, A., Chen, Y., Lillicrap, T., Hui, F., Sifre, L.,
van den Driessche, G., Graepel, T., and Hassabis, D.
Mastering the game of Go without human knowledge.
_Nature_, 550(7676):354–359, October 2017. ISSN 14764687. doi: 10.1038/nature24270. [URL https://doi.](https://doi.org/10.1038/nature24270)
[org/10.1038/nature24270.](https://doi.org/10.1038/nature24270)


Stone, P., Kaminka, G., Kraus, S., and Rosenschein, J. Ad
hoc autonomous agent teams: Collaboration without precoordination. _Proceedings_ _of_ _the_ _AAAI_ _Conference_ _on_
_Artificial Intelligence_, 24(1):1504–1509, Jul. 2010a. doi:
10.1609/aaai.v24i1.7529. [URL https://ojs.aaai.](https://ojs.aaai.org/index.php/AAAI/article/view/7529)
[org/index.php/AAAI/article/view/7529.](https://ojs.aaai.org/index.php/AAAI/article/view/7529)


Stone, P., Kaminka, G. A., Kraus, S., and Rosenschein, J. S.
Ad hoc autonomous agent teams: collaboration without
pre-coordination. In _Proceedings of the Twenty-Fourth_
_AAAI Conference on Artificial Intelligence_, AAAI’10, pp.
1504–1509. AAAI Press, 2010b.


Strouse, D., Kleiman-Weiner, M., Tenenbaum, J., Botvinick,
M., and Schwab, D. Learning to share and hide intentions
using information regularization. In _Advances in Neural_
_Information Processing Systems_, volume 31, 2018.


12



Strouse, D., McKee, K. R., Botvinick, M., Hughes, E., and
Everett, R. Collaborating with humans without human
[data, 2022. URL https://arxiv.org/abs/2110.](https://arxiv.org/abs/2110.08176)
[08176.](https://arxiv.org/abs/2110.08176)


Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., and
Abbeel, P. Domain randomization for transferring deep
neural networks from simulation to the real world, 2017.
[URL https://arxiv.org/abs/1703.06907.](https://arxiv.org/abs/1703.06907)


Tomasello, M. _The cultural origins of human cognition._ The
cultural origins of human cognition. Harvard University
Press, Cambridge, MA, US, 1999. ISBN 0-674-00070-6
(Hardcover). Pages: vi, 248.


Tuyls, K., Perolat, J., Lanctot, M., Leibo, J. Z., and Graepel,´
T. A generalised method for empirical game theoretic
analysis. _CoRR_, abs/1803.06376, 2018. URL [http:](http://arxiv.org/abs/1803.06376)
[//arxiv.org/abs/1803.06376.](http://arxiv.org/abs/1803.06376)


Vinyals, O., Babuschkin, I., Czarnecki, W. M., Mathieu, M.,
Dudzik, A., Chung, J., Choi, D. H., Powell, R., Ewalds, T.,
Georgiev, P., Oh, J., Horgan, D., Kroiss, M., Danihelka, I.,
Huang, A., Sifre, L., Cai, T., Agapiou, J. P., Jaderberg, M.,
Vezhnevets, A. S., Leblond, R., Pohlen, T., Dalibard, V.,
Budden, D., Sulsky, Y., Molloy, J., Paine, T. L., Gulcehre,
C., Wang, Z., Pfaff, T., Wu, Y., Ring, R., Yogatama,
D., Wunsch,¨ D., McKinney, K., Smith, O., Schaul, T.,
Lillicrap, T., Kavukcuoglu, K., Hassabis, D., Apps, C.,
and Silver, D. Grandmaster level in StarCraft II using
multi-agent reinforcement learning. _Nature_, 575(7782):
350–354, November 2019. ISSN 1476-4687. doi: 10.
1038/s41586-019-1724-z. [URL https://doi.org/](https://doi.org/10.1038/s41586-019-1724-z)
[10.1038/s41586-019-1724-z.](https://doi.org/10.1038/s41586-019-1724-z)


Wang, J. X., Kurth-Nelson, Z., Kumaran, D., Tirumala, D., Soyer, H., Leibo, J. Z., Hassabis, D., and
Botvinick, M. Prefrontal cortex as a meta-reinforcement
learning system. _Nature_ _Neuroscience_, 21(6):860–
868, June 2018. ISSN 1546-1726. doi: 10.1038/
s41593-018-0147-8. URL [https://doi.org/10.](https://doi.org/10.1038/s41593-018-0147-8)
[1038/s41593-018-0147-8.](https://doi.org/10.1038/s41593-018-0147-8)


Wang, X., Zhang, S., Zhang, W., Dong, W., Chen, J.,
Wen, Y., and Zhang, W. Quantifying zero-shot coordination capability with behavior preferring partners,
2024. [URL https://openreview.net/forum?](https://openreview.net/forum?id=wTRpjTO3F7)
[id=wTRpjTO3F7.](https://openreview.net/forum?id=wTRpjTO3F7)


Wellman, M. P., Tuyls, K., and Greenwald, A. Empirical game-theoretic analysis: A survey. _arXiv_ _preprint_
_arXiv:2403.04018_, 2024.


Wu*, S. A., Wang*, R. E., Evans, J. A., Tenenbaum, J. B.,
Parkes, D. C., and Kleiman-Weiner, M. Too many cooks:
Bayesian inference for coordinating multi-agent collaboration. _Topics in Cognitive Science_, 13(2):414–432, 2021.


**Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination**


Xia, G.-S., Bai, X., Ding, J., Zhu, Z., Belongie, S., Luo, J.,
Datcu, M., Pelillo, M., and Zhang, L. Dota: A large-scale
dataset for object detection in aerial images. In _2018_
_IEEE/CVF Conference on Computer Vision and Pattern_
_Recognition_, pp. 3974–3983, 2018. doi: 10.1109/CVPR.
2018.00418.


Yan, X., Guo, J., Lou, X., Wang, J., Zhang, H., and
Du, Y. An efficient end-to-end training approach for
zero-shot human-AI coordination. In _Thirty-seventh_
_Conference on Neural Information Processing Systems_,
2023. [URL https://openreview.net/forum?](https://openreview.net/forum?id=6ePsuwXUwf)
[id=6ePsuwXUwf.](https://openreview.net/forum?id=6ePsuwXUwf)


Ying, L., Jha, K., Aarya, S., Tenenbaum, J. B., Torralba,
A., and Shu, T. Goma: Proactive embodied cooperative
communication via goal-oriented mental alignment, 2024.
[URL https://arxiv.org/abs/2403.11075.](https://arxiv.org/abs/2403.11075)


Zhao, R., Song, J., Hu, H., Gao, Y., Wu, Y., Sun,
Z., and Wei, Y. Maximum entropy population based
training for zero-shot human-ai coordination. _CoRR_,
abs/2112.11701, 2021. [URL https://arxiv.org/](https://arxiv.org/abs/2112.11701)
[abs/2112.11701.](https://arxiv.org/abs/2112.11701)


Zhao, R., Song, J., Yuan, Y., Haifeng, H., Gao, Y., Wu, Y.,
Sun, Z., and Wei, Y. Maximum entropy population-based
training for zero-shot human-ai coordination, 2022. URL
[https://arxiv.org/abs/2112.11701.](https://arxiv.org/abs/2112.11701)


Zhou, Y., Li, J., and Zhu, J. Posterior sampling for multiagent reinforcement learning: solving extensive games
with imperfect information. In _International Conference_
_on_ _Learning_ _Representations_, 2020. URL [https://](https://openreview.net/forum?id=Syg-ET4FPS)
[openreview.net/forum?id=Syg-ET4FPS.](https://openreview.net/forum?id=Syg-ET4FPS)


13


**A.1. Procedurally Generated Overcooked**


In this section, we provide additional details for how we create many solvable coordination challenges in Overcooked. We
first sample which of the five layouts we would like to use as a base predicate. Next, we remove all items and agents from
the layout so that we are left with just walls and free spaces. We begin by first sampling a set of plate piles (three white
circles in triangle shape in Figure 12), onion piles (three yellow circles in triangle shape in Figure 12), pots (Darker black
rectangle with lid embedded within a gray wall in Figure 12), and goals (green squares in Figure 12) to be on a wall not in
the regions marked with a red cross in Figure 12. We then sample an additional set of each of the aforementioned items on
any of the remaining walls that are not occupied by other objects.


Next, we sample initial agent locations. In the _Coordination Ring, Counter Circuit,_ and _Cramped Room_ layouts, we can
sample both of the agents in any of the available free spaces. However, in _Asymmetric Advantages_ and _Forced Coordination_
this can be problematic since a divider prevents agents from moving between two halves of the grid. As such, if all
task-relevant items are sampled on inaccessible walls, the agents will not be able to complete the tasks. To avoid this issue,
for these layouts only we make sure to sample initial agent locations in free spaces on separate halves of the grid. That is,
the red agent’s initial location will be sampled on a free cell in the left half of the grid and the blue agent’s initial location
will be sampled on the right half of the grid, or vice versa.


By guaranteeing at least one set of items relevant to the task are reachable and accessible by at least one agent, we can
create a new, solvable coordination challenge. We randomly sample whether we will rotate the sampled grid 90 degrees
clockwise, and embed the resulting state in a larger 9x9x26 observation space by padding additional cells as just being
walls. After making this 9x9x26 grid, we compare it to held-out levels to see whether the goal positions, pot positions, plate
pile positions, and onion pile positions are the same. If they are, we resample a new grid. We provide pseudocode for our
approach in Algorithm 1.


**A.2. CEC in Partially Observable Environments**


We modified the Dual Destination problem to give agents a 3x3 visibility window instead of the fully observable 5x5 window
from the results in Figure 3. Then, we trained CEC, FCP, and IPPO using the same architectures as in the fully observable
setting and 300 million steps of training. The challenge of breaking handshakes when learning multi-agent policies is even
more pronounced in the partially observable case, as agents may form arbitrary conventions to handle high uncertainty about
the state of the world. **From our results in Figure 13, we find the same conclusions in the partially observable setting**
**as we did in the fully observable results described in the paper:** **CEC has high cross-play performance in ZSC with**
**other agents on novel environments (0.74), outperforming population based methods (0.61) and naive self-play (0.03).**


**A.3. CEC in Multi-task Environments**


We explored whether CEC would be beneficial for cross-partner and cross-environment generalization when there are
multiple solutions to a task. The intuition here is that with multiple possible optimal responses a team of agents could
have for collaboration, PBT or self-play methods with sufficient exploration might be able to form robust, object-oriented


14


**Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination**


**Algorithm 1** Solvable Overcooked Coordination Challenge Generation


**Input:** Layout set _L_ = _{_ Coordination Ring, _. . ._, Forced Coordination _}_, Held-out set of evaluation levels _Gh_
Sample _L_ base _∼U_ ( _L_ ) _{_ Discrete uniform distribution _}_
Initialize grid _G ←_ LoadWalls( _L_ base)
_G ←_ _G \ {_ objects _∪_ agents _}_
**Phase 1:** **Mandatory Object Placement**
**for** _∀ot_ _∈O_ = _{_ PlatePile _,_ OnionPile _,_ Pot _,_ Goal _}_ **do**

Let _V_ = _{w_ _∈_ _G_ walls _| w_ _∈/_ _R_ red _}_
_G ←_ _G ∪{ot_ (rand( _V_ )) _}_
**end for**
**Phase 2:** **Supplemental Object Placement**
**for** _∀ot_ _∈O_ **do**

_V_ remaining = _{w_ _∈_ _V_ _| w_ _∈/_ _G}_
_G ←_ _G ∪{ot_ (rand( _V_ remaining)) _}_
**end for**
**Agent Positioning**
**if** _L_ base _∈{_ Asymmetric Advantages _,_ Forced Coordination _}_ **then**

_pos_ 1 _∼U_ ( _{c ∈_ FreeSpacesleft _}_ )
_pos_ 2 _∼U_ ( _{c ∈_ FreeSpacesright _}_ )
**else**

_pos_ 1 _∼U_ (FreeSpaces)
_pos_ 2 _∼U_ (FreeSpaces _\ {pos_ 1 _}_ )
**end if**
**Post-Processing**
**if** _u ∼U_ ([0 _,_ 1]) _>_ 0 _._ 5 **then**

_G ←_ Rotate( _G,_ 90 _[◦]_ )
**end if**
_G ←_ Pad( _G,_ walls _,_ 9 _×_ 9)
**if** _G_ _∈/_ _Gh_ **then**

**Return** Valid configuration challenge _G_
**else**

Repeat generation process
**end if**


representations without the need for CEC.


To test this, we extended the Dual Destination environment to have two possible valid solutions to reward agents (Figure 14.)
Now, agents are rewarded if they are on opposite green or opposite pink squares. As shown in Figure 15, in the multi-task
variant both valid squares remain equidistant from the agents so that there are now 4 strategies which could be rewarded.
For the procedural generator, just as in the original Dual Destination problem we randomly shuffle agent and goal locations
so that they all lie on unique grid cells. We show **even in the multi-task setting, CEC agents (0.404) outperform PBT**
**methods (0.251) and naive self-play (0.083) when collaborating with novel partners on tasks PBT and naive self-play**
**method swerve trained on.** **Just as in the single-task setting, PBT (0.005) and naive self-play (0.004) cannot generalize**
**to novel partners on novel environments, whereas CEC can (0.446)**, albeit with a slight performance reduction from
the single-task setting in Figure 3 of our paper (CEC=0.931 and 0.966 on fixed and procedurally generated single-task
problems respectively). This finding illustrates that additional work is needed to understand the impacts of task complexity
and procedural environment generation.


**A.4. Combining Partner and Environment Diversity**


We tested the impact of combining partner diversity with environment diversity by training E3T agents under the CEC
paradigm. We set the partner policy randomness to 0.5, consistent with human experiments and E3T’s original design.
Results in the ZSC setting in simulation (Figure 16) show CEC-E3T performs worse than other models on Overcooked’s five
original layouts (CEC=130.51, CEC-E3T=28.21) but outperforms CEC-Finetune on 100 held-out grids (CEC-FT=41.73,


15


**Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination**


_Figure 13._ In the partially observable setting, CEC ZSC performance
on the Dual Destination problem replicates findings from the fully
observable case, suggesting it has promise for other games with
imperfect information and the need for dynamic conventions, such as
Hanabi.



_Figure 14._ Overview of the Multi-Task Dual Destination problem.
Agents are rewarded for going to either opposite pink squares or
opposite green squares.



_Figure 15._ Even in this multi-task setting, CEC outperforms
population-based methods when tasked with collaborating with
novel partners on novel problems. Just as in the single-task setting,
naive self-play and PBT methods fail to achieve substantial rewards on environments they have not seen during training, whereas
CEC can.



CEC-E3T=58.13). The learning curves (Figures 17 and 18) from a checkpoint 2 billions steps into training reveal that noisy
partners introduced additional training noise, which, combined with dynamic environments, likely requires larger networks
and longer training times for convergence compared to vanilla CEC.


**A.5. Necessity of Recurrent Networks for CEC**


We include recurrent networks with CEC to enable a basic meta-learning algorithm (Wang et al., 2018; Rabinowitz et al.,
2018). We tested whether it is possible for CEC to retain reasonable performance without using recurrent policies. In
Overcooked and the Dual Destination problem, agents without recurrence failed to converge or adapt effectively. As shown
in the learning curves for the Dual Destination problem, CEC with LSTMs successfully converged (Figure 19), while the
non-recurrent version couldn’t even achieve positive rewards (Figure 20).


**A.6. NiceWebRL for Jax-based Human Experiments**


Reinforcement Learning has experienced accelerated progress recently due to the adoption of Jax-based environments that
enable tackling single-agent (Bonnet et al., 2024; Nikulin et al., 2024; Matthews et al., 2024) and multi-agent (Rutherford
et al., 2023; Lu et al., 2024) problems at millions of steps per second on academic-scale compute. From. To that effect, an
open question within the fields of cognitive and computer science is how can these advances in environment design help us
progress our understanding of individual and collective agency in humans and machines. For our human experiments we


16


**Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination**



100 Procedurally

Generated Grids


Algorithm



160


80


0



5 Heldout Grids


Algorithm



_Figure 16._ CEC with E3T exhibits lower cross play performance in the ZSC setting than any other method on the original 5 Overcooked
layouts, but exhibits better generality to the 100 held out procedurally generated layouts than CEC fine-tuned on one of the original 5



_Figure 17._ CEC on its own is able to achieve very high rewards
during training across novel environments



_Figure 18._ Combining CEC with a partner diversity method (E3T)
leads to a training curve which achieves half of the reward as CEC
on its own in the same amount of training time



used NiceWebRL (Carvalho, 2025), a unified tool for evaluating single-Human, Human-AI, and Human-Human performance
in turn-based or simultaneous-action games.


NiceWebRL is built on top of the NiceGUI nic Python package, and leverages server-side environment parallelization
to make a seamless client-side study experience for any single or multi-agent game based in Jax. It precompiles the
environment reset and step functions, as well as the code to render the state as a pixel-based image. To ensure minimal
latency, NiceWebRL then simulates future possible states and stores an image representing each possible future-world
client-side. Then, when the active Human actually takes an action in the game, it renders the corresponding state and
simulates forward in time during the few milliseconds before the human can select another action.


Not only does NiceWebRL provide an intuitive way to load in environments and AI models to conduct human experiments
on, it also gives researchers an accessible method for collecting and saving participant data thanks to NiceGUI’s large variety
of survey options. Moreover, it provides a seamless way to deploy experiments on the web from a laptop: even running
Human-AI experiments with complex models on a laptop with a few cpu cores, we were able to host several simultaneous
user studies across the globe with low latency. The authors of the package originally included just single-agent support, and
we extended that package to evaluate human-AI and human-human gameplay.


17


**Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination**



_Figure 19._ In 300 million timesteps, CEC with an LSTM converged to the maximum reward on the Dual Destination problem.


**A.7. Agent Training Details**


A.7.1. NETWORK ARCHITECTURE



_Figure 20._ In 300 million timesteps, CEC without an LSTM cannot
achieve a positive reward on the Dual Destination problem.



For all IPPO, FCP, and CEC agents, we use the architecture in Table 1 consisting of three main components: an observation
encoder, a recurrent core, and separate actor-critic heads.


















|Component|Layer|Details<br>√|
|---|---|---|
|Observation Encoder|Conv1<br>Conv2<br>FC1<br>FC2|2×2 kernel, 64 flters, orth(<br><br>2), ReLU activation<br>2×2 kernel, 32 flters, orth(<br>_√_<br>2), ReLU activation<br>Fully-connected, 512 units, orth(<br>_√_<br>2), ReLU activation<br>Fully-connected, 512 units, orth(<br>_√_<br>2), ReLU activation|
|Recurrent Core|LSTM|Feature size: 256, state resets at episode boundaries|
|Actor Head|FC1<br>FC2<br>FC3<br>FC4<br>Output|Fully-connected, 256 units, orth(2), ReLU activation<br>Fully-connected, 192 units, orth(2), ReLU activation<br>Fully-connected, 128 units, orth(2), ReLU activation<br>Fully-connected, 64 units, orth(2) [Overcooked only], ReLU activation<br>Fully-connected, [6 for Overcooked, 5 for Dual Destination Problem]<br>units (logits for discrete actions), orth(0.01)|
|Critic Head|FC1<br>FC2<br>FC3<br>FC4<br>Output|Fully-connected, 512 units, orth(2), ReLU activation<br>Fully-connected, 256 units, orth(2), ReLU activation<br>Fully-connected, 192 units, orth(2) [Overcooked only], ReLU activation<br>Fully-connected, 128 units, orth(2) [Overcooked only], ReLU activation<br>Fully-connected, 1 unit (value prediction), orth(1.0)|



_Table 1._ Agent Architecture for IPPO, FCP, and CEC agents. All layers use orthogonal weight initialization with layer-specific scaling
factors (orth(scale)) and zero bias initialization.


For E3T, we use the architecture described in (Yan et al., 2023), with the randomness parameter for the partner policy = 0 _._ 55


A.7.2. PPO PARAMETERS


We use the parameters in Table 2 for training all PPO agents:


**A.8. Cross-Algorithm Analysis**


Agents trained with each of the different algorithms in the single-task setting (IPPO, FCP, E3T) play each other and the two
CEC models on both the five original and 100 procedurally generated held-out Overcooked grids. We show the results in


18


**Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination**

|Parameter|Value|Description|
|---|---|---|
|LR|3_ ×_ 10_−_4|Initial learning rate for policy optimization|
|NUM ~~S~~TEPS|256|Number of steps to collect per environment<br>before updating|
|TOTAL ~~T~~IMESTEPS|3_ ×_ 109|Total number of environment steps for train-<br>ing|
|UPDATE ~~E~~POCHS|4|Number of epochs to update policy per col-<br>lected batch|
|NUM ~~M~~INIBATCHES|2|Number of minibatches to split collected<br>data into|
|GAMMA|0.99|Discount factor for future rewards|
|GAE ~~L~~AMBDA|0.95|Lambda parameter for Generalized Advan-<br>tage Estimation|
|CLIP ~~E~~PS|0.2|PPO clipping parameter for policy loss|
|ENT ~~C~~OEF|0.005|Entropy coeffcient for encouraging explo-<br>ration|
|VF ~~C~~OEF|1.0|Value function loss coeffcient|
|MAX ~~G~~RAD ~~N~~ORM|0.5|Maximum gradient norm for gradient clip-<br>ping|
|ANNEAL ~~L~~R|True|Whether to use learning rate annealing|



_Table 2._ PPO Hyperparameters


Figures 21 and 22. From these heatmaps, we see that CEC can collaborate with agents trained with different algorithms on
par with PBT methods, indicating that it is truly learning how to adapt rather than follow a single strategy (Question 1). We
use the values in Figures 21 and 22 as the payoff matrices for our Empirical Game Theory analyses in Figure 8.


We additionally include the AI-AI XP performance for agents across each of the original 5 layouts in Figure 23, which
shows that CEC or CEC-Finetune perform better than all other models in 4 out of 5 layouts and competitively with other
methods in the layout it did not do as well in.


**A.9. Limitations**


The learning curves in Figure 7 show that CEC has not yet converged from SP training or plateaued in terms of XP
performance on any level. Due to cost and time constraints, we were not able to train for longer than 3 billion steps per
model, leaving 2 open questions for future research: 1. What is upper-ceiling on XP performance for IPPO CEC agents,
and 2. How can we balance the bias towards adaptation introduced in CEC with greedier policies which maximize reward
beyond CEC-Finetune.


Moreover, our human experiments filtered for participants capable of fluently speaking English. This can bias our results, as
speaking English typically comes with its own set of cultural practices which impact participants’ abilities to play the game
and also their assessments of agents’ cooperative abilities (Henrich et al., 2010).


**A.10. Qualitative Analysis of Learned Norms**


In this section, we provide additional intuition for the success of CEC in comparison to single-task methods. We examine
_Counter Circuit_ (see Figure 4), a layout with a large amount of strategy diversity. In Figures 24, 25, and 26, we compare the
frequency of different locations visited by IPPO and CEC agents in _Counter Circuit_ . The prevalence of darker regions for
the IPPO agents indicates that they visit certain areas less frequently and tend to follow a fixed route when completing the
task. Such strategies can be brittle if people try moving in the opposite direction around the center block, potentially forcing
agents into unfamiliar locations where they lack experience acting. In contrast, CECs have a more uniform distribution over
the cells they visit. This does not inherently mean CECs are capable of better adaptation, as agents uniformly sampling
actions would theoretically also have uniform state coverage. However, we observe that the cells closest to both pots, both
onion piles, the plate pile, and the goal location receive the highest concentration of visitations by CEC agents. In contrast,
the IPPO agents seem to visit at most one pot. This leads us to infer that CEC agents have developed a richer representation


19


**Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination**



Normalized Reward


Algorithm



0.7


0.6


0.5


0.4


0.3



Normalized Reward


Algorithm



0.5


0.4


0.3


0.2


0.1



_Figure 21._ Heatmap comparing different algorithms playing each
other in the single-task setting, averaged across the original 5
layouts. Brighter yellow regions indicate better XP performance.



_Figure 22._ Heatmap comparing different algorithms playing each
other on 100 held-out procedurally generated Overcooked layouts.
Brighter yellow regions indicate better XP performance.



of the different subtasks involved in cooking, and may be more capable of adapting to users’ actions if forced by the humans
to play a different role within the collective plan (i.e. cook items in a different pot, pickup cooked items instead of pass
onions, etc.).


**A.11. User Assessments of Partners from Human-AI Experiments**


At the end of every episode a participant plays with a different RL agent, we provide them with a survey to assess the agent
they just played with. We ask 7 questions that participants respond to with a rating on the Likert scale (Likert, 1932). The
questions, and corresponding participant ratings of different models, averaged across the two human-AI experiments we ran
( _Counter Circuit_ and _Coordination Ring_ ) are included in Figure 27.


We also include the responses averaged across participants for each individual experiment in Figures 28 and 29.



240


160


80


0



Layout



_Figure 23._ Comparison of model performance across each of the original 5 Overcooked layouts, with standard error bars shown. CEC or
CEC-Finetune achieve the highest mean reward in 4 out of 5 layouts.


20


4


3


2







tively cooperate with novel partners on
novel problems.



1


0



Question



_Figure 27._ Participant assessments of different models across 7 different metrics averaged across _Counter Circuit_ and _Coordination Ring_ .


**A.12. Human-AI Collaboration Success Per Layout**


We plot the success of different models playing with humans in our user study in Figures 30 and 31. We show the results of
_Counter Circuit_ and _Coordination Ring_ respectively.


21


4


3


2


1


0


4


3


2


1


0



**Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination**


Question


_Figure 28._ Participant assessments of different models across 7 different metrics for the experiment _Counter Circuit_ .


Question


_Figure 29._ Participant assessments of different models across 7 different metrics for the experiment _Coordination Ring_ .


22


**Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination**



Algorithm



2


1


0



Algorithm



3


2


1


0



_Figure 30._ Success rates of different algorithms playing _Counter_
_Circuit_ with humans.



_Figure 31._ Success rates of different algorithms playing _Coordina-_
_tion Ring_ with humans.



23


