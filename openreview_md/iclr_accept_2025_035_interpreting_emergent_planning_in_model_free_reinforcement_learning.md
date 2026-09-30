# Interpreting Emergent Planning In Model- Free Reinforcement Learning

Thomas Bush1, Stephen Chung1†, Usman Anwar1†**, Adria Garriga-Alonso** `
2**, David Krueger**3 1University of Cambridge, 2FAR AI, 3Mila, University of Montreal 28tbush@gmail.com, {mhc48,ua237,dsk30}@cam.ac.uk, adria@far.ai
†Equal contribution.

## Abstract

We present the first mechanistic evidence that model-free reinforcement learning agents can learn to plan. This is achieved by applying a methodology based on concept-based interpretability to a model-free agent in Sokoban - a commonly used benchmark for studying planning. Specifically, we demonstrate that DRC, a generic model-free agent introduced by Guez et al. (2019), uses learned concept representations to internally formulate plans that both predict the long-term effects of actions on the environment and influence action selection. Our methodology involves: (1) probing for planning-relevant concepts, (2) investigating plan formation within the agent's representations, and (3) verifying that discovered plans (in the agent's representations) have a causal effect on the agent's behavior through interventions. We also show that the emergence of these plans coincides with the emergence of a planning-like property: the ability to benefit from additional test-time compute. Finally, we perform a qualitative analysis of the planning algorithm learned by the agent and discover a strong resemblance to parallelized bidirectional search. Our findings advance understanding of the internal mechanisms underlying planning behavior in agents, which is important given the recent trend of emergent planning and reasoning capabilities in LLMs through RL.

## 1 Introduction

In reinforcement learning (RL), decision-time planning - that is, the capacity of selecting immediate actions to perform by predicting and evaluating the consequences of future actions - is conventionally associated with agents that possess explicit world models, like MuZero (Schrittwieser et al., 2020). This naturally raises the question: can model-free reinforcement learning agents - that is, agents which lack explicit world models - also learn to perform decision-time planning? In prior work, Guez et al. (2019) introduced Deep Repeated ConvLSTM (DRC) agents. Despite lacking an explicit world model, DRC agents behave like they perform decision-time planning. For example, they excel at strategic domains like Sokoban, and perform better if given extra testtime compute (Guez et al., 2019; Taufeeque et al., 2024). However, this only partially answers the above question as these behaviors may not be due to internal planning but, rather, other mechanisms that generate planning-like behavior in the environments studied. In this paper, we mechanistically analyze a Sokoban-playing DRC agent and show that it is indeed internally planning. In doing so, we provide the first non-behavioral evidence that model-free RL agents can learn to internally plan. Using concept-based interpretability (Kim et al., 2018), we provide three types of convergent evidence showing that the DRC agent has learned, and is making use of, concepts that are instrumentally useful for planning. First, we use linear probes (Alain & Bengio, 2016) to show that the agent represents specific concepts that predict the long-term effects of its actions on the environment.

Then, we demonstrate that these concept representations are associated with a learned planning process by analyzing how the agent uses them to iteratively construct 'plans' at test-time. Finally, we demonstrate that these concept representations causally influence the agent's behavior as would be expected if these representations were being used for planning To summarize, this paper makes the following contributions:
1
- We design a procedure, based on concept-based interpretability, for determining if a modelfree agent performs planning using a hypothesized set of concepts. This procedure involves (1) probing for planning-relevant concepts, (2) investigating plan formation in the agent's internal representations, and (3) verifying the causal effect of plans on the agent's behavior.

- Using this procedure, we show that, in Sokoban, a DRC agent (Guez et al., 2019) internally forms plans, and that these plans can be altered to steer the agent. We find this agent learns a planning algorithm resembling parallelized bidirectional search, which differs from commonly-used planning algorithms in RL.

This work aligns with the growing body of research demonstrating that model-free RL agents can learn to plan and even reason. For example, in this study, we show that DRC agents can learn to evaluate and revise plans. Recently, DeepSeek-R1, an LLM with reasoning capabilities primarily trained via RL, has demonstrated similar self-correction behavior in its reasoning, referred to as 'aha moments' (Guo et al., 2025). As such, we believe that understanding the mechanisms behind these emergent capabilities in RL agents is highly important.

## 2 Background 2.1 Planning In Reinforcement Learning

Planning has many meanings in RL, encompassing algorithms utilizing environment models during training (Sutton, 1991) or at decision time (Silver et al., 2016; Chung et al., 2024a). In this work, we study whether an RL agent is specifically performing *decision-time* planning. Henceforth, we use
'planning' and 'decision-time planning' interchangeably. In past work, an agent is considered to be planning in this sense if it engages with an (explicit) world model to select actions associated with the best predicted long-term consequences (Hamrick et al., 2020; Chung et al., 2024a). An example is MuZero (Schrittwieser et al., 2020), which applies a planning algorithm called Monte Carlo Tree Search (Coulom, 2006) to a model of its environment to select actions associated with the best longrun consequences. Other similar agents are VPN (Oh et al., 2017), IBP (Pascanu et al., 2017), I2A (Racaniere et al., 2017), MCTSNet (Guez et al., 2018b), and Thinker (Chung et al., 2024a) agents. ` By definition, model-free RL agents lack an *explicit* world model. This makes it difficult to reuse past definitions of planning that presume that an explicit world model is available. Thus, for the purposes of this work, we provide a pragmatic characterization of planning that we use as a foundation for investigating whether the model-free agent studied in this paper performs planning. We consider plans to be sequences of potential future actions. We characterize an agent as planning if it selects actions to perform by considering plans that it formulates and evaluates based on predicted future consequences. This is similar to how planning is understood in neuroscience (Mattar & Lengyel, 2022). It also mirrors model-based definitions of planning but relaxes the requirement for an explicit world model to the requirement that an agent predict consequences of future actions, regardless of the method used. We discuss our characterization further in Appendix E.1. For an agent to plan under our characterization, it must: (i) form plans, (ii) evaluate plans by predicting their consequences, and (iii) be influenced by these plans when acting.

## 2.2 Sokoban

Sokoban is an episodic, fully-observable, deterministic environment in which an agent moves around walls in an 8x8 grid to push four boxes onto four targets. When an agent moves up/down/left/right into a square containing a box, the box is pushed up/down/left/right. Sokoban levels let agents perform actions with irreversible, negative, long-run consequences (moving boxes so the puzzle is unsolvable). Sokoban is thus difficult - it is PSPACE-
complete (Culberson, 1997) - and a common benchmark for studying planning (Racaniere et al., 2017; Guez et al., 2019; Hamrick ` et al., 2020). We study a version of Sokoban where the agent observes a symbolic representation xt ∈ R
8×8×7 of the environment.

For ease of inspection, all figures are presented as pixel representations. Figure 2 compares these two representations. Appendix E.2 further explains this environment.

(a) Pixel (b) Symbolic Figure 2: Pixel and symbolic representations of a Sokoban board.

## 2.3 Deep Repeated Convlstm (Drc) Agents

Deep Repeated ConvLSTM (DRC) agents (Guez et al., 2019) are model-free agents based on ConvLSTMs that perform multiple computational ticks per time step. ConvLSTMs (Shi et al., 2015) are LSTMs (Hochreiter & Schmidhuber, 1997) that utilize 3D hidden states and convolutional connections. At each time step t, a DRC agent passes an observation xt through a convolutional encoder to generate an encoding it ∈ R
H0×W0×G0. This is then processed by D ConvLSTM layers. At time t, the d-th ConvLSTM has a cell state g d t ∈ R
Hd×Wd×Gd . Unlike standard recurrent networks which perform a single tick of recurrent computation per time step, DRC agents perform N ticks of recurrent computation per step. Guez et al. (2019) show these internal ticks improve the performance and generalization of DRC agents. Appendix E.3 provides further architectural details. DRC agents behave in a manner that suggests they internally engage in decision-time planning. For instance, DRC agents outperform model-based agents like MuZero (Schrittwieser et al., 2020) in Sokoban (Chung et al., 2024b), and exhibit improved performance when given extra test-time compute (Taufeeque et al., 2024). This raises a question: do DRC agents genuinely learn to internally perform planning, or is their planning-like behavior merely a result of complex learned heuristics? In this paper, we investigate whether a Sokoban-playing DRC agent internally plans. The agent we study has D = 3 ConvLSTM layers and performs N = 3 internal ticks per step. The agent's encoder and ConvLSTMs have 32 channels (Gd = 32) and utilize kernels of size 3 with a single layer of input zero padding. Thus, all cell states share Sokoban's spatial dimensions (Hd = Wd = 8).

The agent is trained for 250 million transitions on the unfiltered Boxoban training set (Guez et al., 2018a) using a similar training setup as Guez et al. (2019) as explained in Appendix E.4. Appendix E.5 shows that, consistent with Guez et al. (2019), this agent exhibits planning-like behavior.

## 2.4 Concept-Based Interpretability

Concept-based interpretability is an approach to explaining neural network behavior that involves identifying which concepts a network internally represents (Kim et al., 2018). A concept is generally understood as a unit of knowledge (Schut et al., 2023). In this paper, we specifically consider 'multi-class' concepts, which can formally be defined as mappings from input states (or parts of input states) to some fixed classes. That is, multi-class concepts correspond to interpretable, discrete features, and map inputs to classes of that concept. For instance, a multi-class Sokoban concept might be 'the number of empty targets'. This concept would map any observed Sokoban board xt to a class in {ONE, TWO, THREE, FOUR} depending on the number of remaining empty targets in xt.

We focus on concepts networks represent *linearly* (Mikolov et al., 2013). To check if a network linearly represents concepts, we use *linear probes*. These are linear classifiers trained to predict concept classes assigned to inputs using the associated network activations (Alain & Bengio, 2016). As linear classifiers, linear probes compute logits lk = w T
kg for each class k by projecting network activations g ∈ R
dalong a class-specific vector wk ∈ R
d. Belinkov (2022) explains probes further.

## 3 Methodology 3.1 A Procedure For Investigating Model-Free Planning

In Section 2.1, we characterized planning as requiring that an agent (i) formulate plans, (ii) evaluate the consequences of these plans, and (iii) be guided by these plans when selecting actions. If an agent learns to plan, we expect planning-relevant concepts to emerge in its internal representations to meet the first condition. These concepts ought to reflect the agent's plan, and so should correspond to potential future actions, or to their likely environmental effects. Additionally, evidence of plan evaluation - such as avoiding or improving bad plans - should exist to satisfy the second condition. Lastly, to fulfill the third condition, the plan must causally influence the agent's behavior. To determine if an agent exhibits these three properties, we follow the procedure outlined below:
1. **Probe for Concept Representations.** First, we identify a group of environment-specific concepts that could be instrumentally useful for planning. We then use linear probes to establish whether these concepts are being (linearly) represented by the agent (Section 4).

2. **Investigate Plan Formation.** Next, we focus on gathering qualitative evidence of the agent forming plans based on the planning-relevant concepts probed for in the previous step, and evidence of the agent evaluating and refining these plans (Section 5).

3. **Confirm Behavioral Dependence.** Finally, we confirm that these internal plans influence the agent's behavior. For instance, we show that the agent can be steered to form and execute desired plans by intervening on plan representations within the network (Section 6).

## 3.2 Planning-Relevant Concepts In Sokoban

To apply this procedure, we must specify concepts we expect the agent to plan with. Sokoban has a grid-based structure with localized transition dynamics, i.e., the future state of a square is determined by the current state of its neighbors. This makes spatially local concepts (i.e., concepts related to individual or connected squares) more natural for planning than spatially global concepts (i.e., representations of the whole board). We thus claim that an agent that learns to plan in Sokoban may do so by encoding concepts localized to individual squares. We call these 'square-level' concepts. Such concepts seem natural for DRC agents as the 3D structure of ConvLSTMs allows for spatial correspondence between the Sokoban grid and agent hidden states. We focus on multi-class squarelevel concepts which, as explained further in Appendix E.6, map grid squares to concept classes. We hypothesize that the agent will plan using the following square-level, multi-class concepts:
- **Agent Approach Direction** (CA): For a given square, this concept encodes whether the agent will move onto the square in the future. If so, it also encodes the direction from which the agent will move onto the square the next time the agent moves onto it.

- **Box Push Direction** (CB): For a given square, this concept encodes whether a box will be pushed off the square in the future. If so, it also encodes the direction in which the next box pushed off this square will be pushed.

Figure 3 illustrates the classes assigned to each square of a Sokoban board by these concepts over six transitions near the end of an episode. Both concepts map each grid square of the agent's observed

(a) Agent Approach Direction CA
Figure 3: Examples of the classes assigned to the squares of a Sokoban board over 6 transitions
(from left to right) by the concepts 'Agent Approach Direction' (CA) and 'Box Push Direction' (CB). An arrow corresponds to the assignment of the associated directional class. The lack of an arrow in a square indicates the assignment of the class NEVER.

Sokoban board to the classes {UP, DOWN, LEFT, RIGHT, NEVER}. The directional classes correspond to the agent's movement directions. If the next time the agent *steps onto a specific square*, the agent steps onto that square from the left, the concept CA would map this square to the class LEFT. If the next time the agent *pushes a box off of specific square*, the box is pushed to the left, the concept CB would map this square to the class LEFT. Finally, the class NEVER corresponds to the agent not stepping onto or pushing a box off of a square again for the remainder of the episode.

Both concepts depend on the agent's behavior: we can only determine the classes these concepts map grid squares to *after* observing the agent's behavior over the entire episode. Furthermore, as shown in Figure 3, the classes squares are mapped to will change at every transition. Once an agent steps onto a square, the classes assigned to that square will update to represent the agent's *future* interactions with that square. We investigate alternate concepts in Appendices D.4 and D.5.

## 4 Probing For Concept Representations

We now perform the first step of our analysis: determining whether the agent internally represents the concepts that we hypothesize it uses to internally form and evaluate plans.

## 4.1 Experiment Details

Specifically, we use linear probes to determine if the agent represents (a) CA, the agent's future movement onto squares, and (b) CB, the future directions boxes are pushed off of squares. We train linear probes that take as input the agent's cell state activations after the final of the three computational ticks performed each step. We train separate probes for the agent's three layers. We hypothesize the agent will learn a spatial bijection between its cell state and the Sokoban grid.

Thus, when predicting CA and CB at each location (*x, y*), our probes receive as input cell state activations centered on (*x, y*). We train both 1x1 probes (which take as input just the activations at (*x, y*)) and 3x3 probes (which take as input the 3x3 patch of activations around (x, y)). These probes have 160 and 1440 parameters, so are unlikely to overfit. We consider larger probes in Appendix D.3. Each probe is trained using logistic regression with the AdamW optimizer, and five unique initialization seeds. The training dataset is generated by running the agent for 3000 episodes on levels from the Boxoban unfiltered training dataset (Guez et al., 2018a). We test probes on a test set of transitions generated by running the agent for 1000 episodes on levels from the Boxoban unfiltered validation dataset. Further probe training details are given in Appendix D.1. We compare the performance of all probes to baseline probes that receive the raw observation xt as input. This comparison aims to assess the extent to which probes' abilities to predict concept classes are due to these concepts being internally represented by the agent rather than the probes learning how to do so themselves.

(a) CA (b) CB
(a) (b) (c)

## 4.2 Results

In many Sokoban boards, the agent will never move onto, nor push a box off, a large number of squares. As a result, many squares are assigned the label NEVER for both concepts in our probing datasets, leading to class imbalance. We therefore evaluate probe performance using macro F1 scores in place of accuracy. Figure 4 shows the macro F1 scores achieved by probes trained to predict the classes assigned to Sokoban squares by CA and CB. The probes that predict these concepts using the agent's cell state activations vastly outperform the baseline, implying the agent linearly represents CA and CB. This aligns with past work finding linear concept representations in many different networks (Nanda et al., 2023; McGrath et al., 2022; Zou et al., 2023).

Figure 4 confirms that the agent represents square-level concepts at localized positions of its ConvLSTM cells as opposed to distributing representations across adjacent positions. This is evidenced by the minimal improvement in performance when moving from a 1x1 probe to a 3x3 probe, compared to the significant improvement in baseline performance. We thus focus on 1x1 probes for the remainder of this paper. Interestingly, Figure 4 also shows that while probes at layer 2 generally perform slightly better than probes at layer 1, there is little variation in performance across layers. This indicates that the concepts are represented across all layers. We thus hypothesize that the agent is engaged in iterative computation (Jastrzebski et al., 2018), whereby it refines plans across layers.

## 5 Investigating Plan Formation

In this section, we now provide qualitative evidence that the agent forms plans by searching forward from the boxes and backward from the targets, and that the agent develops, evaluates, and adapts plans in parallel. In this section, we primarily focus on descriptive explanations of how the agent forms plans and the general shape of the plans. We defer more conclusive evidence - in the form of intervening on the agent's plan formation process to steer the agent's behavior - to the next section.

Previously, we demonstrated that the agent encodes (at least) two planning-relevant concepts: CA and CB. These concepts represent predictions regarding how the agent will act when moving onto a given square in the future, and how the environment - specifically, the locations of boxes - will be affected by these actions. We thus posit that the agent's representations of these concepts - when looked at holistically, over the entire board - will collectively constitute a plan that the agent forms and adapts. For example, in Figure 5 we visualize the agent's representations of CB and CA
over entire Sokoban boards, as decoded from the agent's cell state by a 1x1 probe in different levels. Three observations can be made from Figure 5: (a) the arrows, which indicate the direction the agent expects to move or push boxes, tend to be connected and trace a path; (b) the arrows tend to connect boxes to specific targets; (c) the arrows collectively form a plan which corresponds to solving the level. In Appendix A.1 we visualize the agent's plan across layers, and show that, while the agent's plans often contains flaws (like the lack of one necessary arrow in Figure 5c), they usually consist of connected paths for the agent to follow and connected routes linking boxes and targets. A natural question then arises: how does the agent form plans? To answer this, we direct attention to Figure 1. Figure 1 visualizes the agent's plans in terms of CB (e.g. the routes the agent plans to push boxes) over the initial steps (A-C) and internal ticks (D-E) of episodes. As can be seen in Figure 1, the agent forms plans *iteratively*. Interestingly, the agent appears to form plans iteratively by searching *forward* from boxes - as illustrated in Figure 1(C) - and *backward* from targets - as illustrated in Figure 1(D). That the agent seems to plan via bidirectional search - which is known to be especially efficient when it is applicable (Russell & Norvig, 2010) - may explain why Guez et al. (2019) found DRC agents to rival specialized planning architectures reliant on forward search. Indeed, as shown in Figure 1(E), the agent seems to utilize a form of *parallelized* bidirectional search whereby it extends multiple plans simultaneously. Appendices A.2.3, A.2.4 and A.2.5 respectively contain further instances of the agent appearing to utilize forward, backward, and parallel search. However, recall that, in Section 2.1, we characterized planning as requiring an agent to evaluate the plans it considers. Evidence suggestive of the agent evaluating plans can be seen in Figure 1(A)- (B). Figures 1(A)-(B), show examples in which the agent appears to (1) formulate a naive plan, (2) evaluate it, and then, upon realizing that it is infeasible or could be improved, (3) adapt its plan accordingly. For instance, in Figure 1(B), the agent changes the targets it plans to push different boxes towards. This is suggestive of the agent using an *evaluative* search algorithm when forming plans. Appendices A.2.1 and A.2.2 contain further examples of the agent seeming to evaluate plans and either plan to push a box a longer route, or change which boxes it plans to push to which targets. Further evidence of the agent planning via an iterative search algorithm can be seen in Figure 6. For Figure 6, we forced the agent to remain stationary for 5 steps prior to acting in 1000 episodes. These 5 'thinking steps' give the agent 15 internal ticks of extra test-time compute. Figure 6 reports the macro F1 when using 1x1 probes to decode CA and CB from the agent's final layer cell state at each of the 15 extra internal ticks, averaged over 1000 episodes. Clearly, the macro F1 improves with the number of ticks. Since the concepts are predictions of future behavior, we can see the predictions of our probes at any tick as being the agent's internal plan *at that* tick. We can then see the corresponding macro F1 as reflecting the quality of the agent's plan at that tick. Figure 6 shows that, as would be expected if the agent planned via an iterative search, the agent's plans iteratively improve when given extra compute. Appendix A.3.1 shows test-time plan refinement occurs at all layers. Appendix A.3.2 provides evidence that it is a consequence of the agent searching deeper. Appendix C.2 shows that this 'test-time plan refinement capability' arises early in training. When considered with the agent's planning-like behavior, the above evidence indicates the agent uses its representations of CA and CB for search-based planning. Further evidence of this is given in Appendices A.2.6-A.2.9 which show examples of the agent planning in out-of-distribution levels, such as levels in which the agent itself is not present (Appendix A.2.6), levels with additional boxes and targets (Appendix A.2.7), and levels in which walls appear and disappear (Appendices A.2.8- A.2.9). These examples suggest the agent's ability to adapt and generalize - benefits of model-based planning Guez et al. (2019) show DRC agents possess - relate to its representations of CA and CB.

Figure 6: Macro F1 when using 1x1 probes to decode CA and CB from the agent's final layer cell state at each of the additional 15 internal ticks performed by the agent when the agent is given 5 'thinking steps', averaged over 1000 episodes.

| Layer 1     | Layer 2     | Layer 3      |             |              |             |              |
|-------------|-------------|--------------|-------------|--------------|-------------|--------------|
| Trained (%) | Random (%)  | Trained (%)  | Random (%)  | Trained (%)  | Random (%)  |              |
| AS          | 94.6 (±0.5) | 33.7 (±32.7) | 90.1 (±1.9) | 29.8 (±36.8) | 98.8 (±0.0) | 27.8 (±37.9) |
| BS          | 56.2 (±1.4) | 31.5 (±13.9) | 72.7 (±1.1) | 30.9 (±25.8) | 80.6 (±2.4) | 4.1 (±5.4)   |

Table 1: Success rates (%) when intervening on each layer using representations from trained and randomly initialized probes. AS and BS refer to 'Agent-Shortcut' and 'Box-Shortcut' interventions. Success rates are averaged over 5 interventions performed. We report ±1 standard deviations.

## 6 Investigating The Role Of Plans

So far, we have shown that the DRC agent represents CA and CB (Section 4), and that it uses these representations to form internal plans (Section 5). We now conclude our analysis by showing that these representations are causally responsible for the agent's behavior. Specifically, we: (1) use these representations to intervene on the agent to force it to form and execute specific plans, and (2) show that these representations emerge concurrently with planning-like behavior during training.

## 6.1 Intervening On Agent Plans

First, we show we can intervene on the agent's activations to alter its behavior over entire episodes. Our interventions involve adding concept vectors learned by probes to the agent's activations to force it to represent concepts in specific ways. We then observe the causal effect of our interventions on the agent's behavior. Recall that a 1x1 probe projects activations along a vector wk ∈ R
32 to compute a logit for class k of some multi-class concept C. We thus encourage the agent to represent square (*x, y*) as class k for concept C by adding wk to position (x, y) of the agent's cell state gx,y:
g
′
x,y ← gx,y + wk (1)
If the agent indeed uses CA and CB for planning, altering the agent's square-level representations of these concepts ought to modify its internal plan and, subsequently, its long-term behavior. We intervene in two sets of handcrafted levels: 'Agent-Shortcut' and 'Box-Shortcut' levels. These sets of levels are characterized by, in each level, there existing two plans: a short plan and a long plan. The plans are similar, but differ in lengths. The agent by default follows the optimal (short) plan. We show our interventions cause it to instead form and execute the suboptimal (long) plan. In 'Agent-Shortcut' levels all boxes and targets are in one region of the board, and the agent can follow either a long or short path to this region. In these levels, we intervene using vectors learned by probes trained to predict CA to steer the agent to plan to move along the long path. Our intervention consists of two parts. We add the vector for NEVER to cell state positions on the short path. We call this the 'short-route' intervention. We also add the vector for the direction which would lead the agent to move onto the first square of the long path to the appropriate cell state position. We call this the 'directional' intervention. An Agent-Shortcut intervention is illustrated in Figure 7b. 'Box-Shortcut' levels are specially-designed levels in which three boxes are adjacent to targets and a fourth box is not. The final box can be pushed a long or short route to a target. In these levels, we intervene using vectors learned by probes trained to predict CB to steer the agent to push this box the long route. Our intervention again consists of two parts. We add the vector for NEVER to cell positions on the short route We also add the directional representation which would encourage the agent to push the box the longer route to the box's initial position. We again call these the 'short-route' and 'directional' interventions. A Box-Shortcut intervention is illustrated in Figure 8b. We intervene on 200 levels of each type. We created 25 levels of each type and then generated 8 versions of each level by applying vertical reflection and 90°, 180°, and 270° rotations. In all levels, we repeat the 'short-route' intervention every step but repeat the 'directional' intervention only until the agent moves onto, or pushes the box off, the corresponding square. We perform our interventions on the agent's cell state at each layer. An intervention is considered successful if it causes the agent to solve the level in the desired suboptimal way. As a baseline, we intervene using representations from randomly initialized probes. For comparability, we scale random
(a) Plan without intervention (b) Intervention (c) Plan with intervention Figure 7: An Agent-Shortcut intervention and its effect on the agent's plan as formulated in terms of CA: (a) the agent's plan after 4 steps *without* the intervention, (b) the initial state of the level and the intervention, and (c) the agent's plan after 4 steps *with* the intervention. The 'short-route' intervention adds the representation of NEVER for CA to positions with white crosses. The 'directional' intervention adds the representation of DOWN for CA to the position with the white arrow.

Figure 8: A Box-Shortcut intervention and its effect on the agent's plan as formulated in terms of CB: (a) the agent's plan after 4 steps *without* the intervention, (b) the initial state of the level and the intervention, and (c) the agent's plan after 4 steps *with* the intervention. The 'short-route' intervention adds the representation of NEVER for CB to positions with white crosses. The 'directional' intervention adds the representation of RIGHT for CB to the position with the white arrow.

probe representations so that the norms of both the random and trained probes are similar. Success rates are averaged over interventions performed with five independently trained or initialized probes. Table 1 shows intervention success rates. At all layers, Agent-Shortcut interventions are successful. While the success rate of Box-Shortcut interventions is lower, it remains high relative to the baseline of interventions using random probes. These results indicate that the agent's representations of CA and CB
influence its behavior in the way that would be expected if it used them for planning. Figures 7 and 8 provide examples of the effect of interventions on the agent's internal plans. These examples suggest the agent not only behaves differently following the interventions, but does so due to forming a different plan. We show more examples of interventions altering the agent's internal plans in Appendix B.1. Appendix B.2 reports success rates when using an intervention scaling factor and varying the squares intervened on. Appendix B.3 reports success rates when intervening to encourage optimal behavior in levels which the agent by default cannot solve. These extra experiments further indicate that the agent's representations of CA and CB influence its behavior as expected.

Figure 9: The relationship between the percentage of extra levels, of medium difficulty, solved when an agent is given 5 steps to 'think', and macro F1 score of probes when predicting CA (blue) and CB (orange) from the agent's final layer cell state. Each point corresponds to these quantities calculated for a single checkpoint.

## 6.2 Investigating The Emergence Of Planning During Training

Finally, we show that the emergence of the agent's representations of CA and CB during training coincides with the agent beginning to exhibit planning-like behavior. This indicates that the agent indeed uses its representations of CA and CB for planning. Specifically, we show the emergence of these representations coincides with the emergence of the agent's ability to benefit from extra testtime compute (Guez et al., 2019; Taufeeque et al., 2024). In particular, we collect checkpoints every 1 million transitions for the first 50 million transitions of training. For every checkpoint, we measure two quantities: (i) the macro F1 score of 1x1 probes trained to decode the concepts CA and CB
given the agent's cell state (following the procedure described in Section 4.1), and (ii) the number of additional levels out of 1000 medium-difficulty levels from the Boxoban dataset (Guez et al., 2018a) the agent can solve when given extra test-time compute by forcing the agent to remain stationary for the first 5 steps of an episode. Figure 9 plots these quantities against each other and shows a strong correlation between them. This implies the agent only reliably begins to exhibit planninglike behavior - benefiting from extra test-time compute - once its final layer representations of CA and CB are sufficiently formed. Appendix C.3 shows that this holds for its representations of CA and CB at all layers. Appendix C.4 shows the agent begins to perform better with extra compute at a similar point in training as to when it can use this compute to refine its plans.

## 7 Additional Results

In the Appendix, we include interesting results that we lacked space to include in the main text. Appendices F provides evidence of DRC agents planning both without internal ticks, and with additional internal ticks. Appendix H provides evidence of a DRC agents planning in a different environment: Mini PacMan. Finally, Appendix G provides evidence of a ResNet (He et al., 2016) agent planning in Sokoban. However, the question of whether a generic agent can learn to plan in a generic environment remains unanswered.

## 8 Related Work

Past work has investigated concept representations learned by game-playing agents (Schut et al., 2023; McGrath et al., 2022; Hammersborg & Strumke, 2022; 2023; Lovering et al., 2022; Mini ¨ et al., 2023) and language models (Li et al., 2023; Nanda et al., 2023; Karvonen, 2024; Ivanitskiy et al., 2024). While past work has focused primarily on whether networks internally represent specific concepts, we study concept representations for the broader purpose of determining if an agent possesses a capability - planning. An exception is work by Jenner et al. (2024), which finds evidence of look-ahead in a chess-playing agent, but does not investigate a wider capacity to 'plan'. Concept-based interpretability is not the only approach to interpreting agents. An alternative is attribution-based interpretability. This involves determining - usually via saliency maps - which features in an agent's observation influence its behavior (Weitkamp et al., 2019; Iyer et al., 2018; Puri et al., 2020; Greydanus et al., 2018; Hilton et al., 2020). Attribution-based methods were not used here as they can depend on subjective interpretation (Atrey et al., 2020). Another approach, examplebased interpretability, explains agent behavior by providing examples of illustrative trajectories or transitions (Rupprecht et al., 2020; Sequeira & Gervasio, 2020; Deshmukh et al., 2023; Zahavy et al., 2016). Due to not studying model internals, example-based methods were ill-suited for this paper. Finally, this paper contributes to recent work investigating the emergence of reasoning capabilities in neural networks (Wei et al., 2022; Kojima et al., 2022; Lehnert et al., 2024; Nye et al., 2021; Wang et al., 2024). However, unlike this paper in which we provide evidence of an agent *internally* performing planning, most work thus far has focused on providing *behavioral* evidence of reasoning.

An exception to this is work by Brinkmann et al. (2024) in which an algorithm learned by a transformer trained on a simple symbolic reasoning task is reverse-engineered. However, Brinkmann et al. (2024) focus on a much simpler form of reasoning than planning as considered in this paper.

## 9 Future Work

In this paper, we proposed a methodology for investigating model-free planning and used it to provide the first non-behavioral evidence of learned planning in a model-free agent. Future work may extend our investigation to other RL agents, and other environments. In particular, it would be helpful to better understand the role of different training factors, e.g., model architecture, environment dynamics in the emergence of planning.

## Acknowledgments

We are thankful to Erik Jenner and Joschka Braun for providing thoughtful feedback on the draft. For much of the duration of this work, TB was supported by the Cambridge Trust and Good Ventures Foundation. UA was supported by OpenPhil AI Fellowship and Vitalik Buterin Fellowship in AI Existential Safety. This work was performed using resources provided by the Cambridge Service for Data Driven Discovery (CSD3) operated by the University of Cambridge Research Computing Service (www.csd3.cam.ac.uk), provided by Dell EMC and Intel using Tier-2 funding from the Engineering and Physical Sciences Research Council (capital grant EP/T022159/1), and DiRAC funding from the Science and Technology Facilities Council (www.dirac.ac.uk).

## References

Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes. *arXiv preprint arXiv:1610.01644*, 2016.

Jacob Andreas. Language models, world models, and human model-building, 2024. URL https:
//lingo.csail.mit.edu/blog/world_models/.

Akanksha Atrey, Kaleigh Clary, and David Jensen. Exploratory not explanatory: Counterfactual analysis of saliency maps for deep reinforcement learning. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=rkl3m1BFDB.

Yonatan Belinkov. Probing classifiers: Promises, shortcomings, and advances. *Computational* Linguistics, 48(1):207–219, 2022.

Jannik Brinkmann, Abhay Sheshadri, Victor Levoso, Paul Swoboda, and Christian Bartelt. A mechanistic analysis of a transformer trained on a symbolic multi-step reasoning task. *arXiv preprint* arXiv:2402.11917, 2024.

Stephen Chung, Ivan Anokhin, and David Krueger. Thinker: learning to plan and act. Advances in Neural Information Processing Systems, 36, 2024a.

Stephen Chung, Scott Niekum, and David Krueger. Predicting future actions of reinforcement learning agents. In *First Reinforcement Learning Safety Workshop*, 2024b. URL https: //openreview.net/forum?id=SohRnh7M8Q.

Remi Coulom. Efficient selectivity and backup operators in monte-carlo tree search. In ´ *International* conference on computers and games, pp. 72–83. Springer, 2006.

Joseph C. Culberson. Sokoban is pspace-complete, 1997. URL https://api.

semanticscholar.org/CorpusID:61114368.

Shripad Vilasrao Deshmukh, Arpan Dasgupta, Balaji Krishnamurthy, Nan Jiang, Chirag Agarwal, Georgios Theocharous, and Jayakumar Subramanian. Explaining RL decisions with trajectories.

In *The Eleventh International Conference on Learning Representations*, 2023. URL https: //openreview.net/forum?id=5Egggz1q575.

Ashley D Edwards, Laura Downs, and James C Davidson. Forward-backward reinforcement learning. *arXiv preprint arXiv:1803.10227*, 2018.

Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Vlad Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, et al. Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures. In *International conference on machine learning*, pp. 1407–1416. PMLR, 2018.

Daniel Freeman, David Ha, and Luke Metz. Learning to predict without looking ahead: World models without forward prediction. *Advances in Neural Information Processing Systems*, 32, 2019.

Anirudh Goyal, Philemon Brakel, William Fedus, Soumye Singhal, Timothy Lillicrap, Sergey Levine, Hugo Larochelle, and Yoshua Bengio. Recall traces: Backtracking models for efficient reinforcement learning. *arXiv preprint arXiv:1804.00379*, 2018.

Samuel Greydanus, Anurag Koul, Jonathan Dodge, and Alan Fern. Visualizing and understanding atari agents. In *International conference on machine learning*, pp. 1792–1801. PMLR, 2018.

Arthur Guez, Mehdi Mirza, Karol Gregor, Rishabh Kabra, Sebastien Racaniere, Theophane Weber, David Raposo, Adam Santoro, Laurent Orseau, Tom Eccles, Greg Wayne, David Silver, Timothy Lillicrap, and Victor Valdes. An investigation of model-free planning: boxoban levels. https://github.com/deepmind/boxoban-levels/, 2018a.

Arthur Guez, Theophane Weber, Ioannis Antonoglou, Karen Simonyan, Oriol Vinyals, Daan Wier- ´
stra, Remi Munos, and David Silver. Learning to search with mctsnets. In ´ International conference on machine learning, pp. 1822–1831. PMLR, 2018b.

Arthur Guez, Mehdi Mirza, Karol Gregor, Rishabh Kabra, Sebastien Racani ´ ere, Th ` eophane Weber, ´
David Raposo, Adam Santoro, Laurent Orseau, Tom Eccles, et al. An investigation of model-free planning. In *International Conference on Machine Learning*, pp. 2464–2473. PMLR, 2019.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025.

David Ha and Jurgen Schmidhuber. Recurrent world models facilitate policy evolution. ¨ *Advances* in Neural Information Processing Systems, 31, 2018.

Patrik Hammersborg and Inga Strumke. Reinforcement learning in an adaptable chess environment ¨
for detecting human-understandable concepts. *arXiv preprint arXiv:2211.05500*, 2022.

Patrik Hammersborg and Inga Strumke. Information based explanation methods for deep learning ¨
agents–with applications on large open-source chess models. *arXiv preprint arXiv:2309.09702*, 2023.

Jessica B Hamrick, Abram L Friesen, Feryal Behbahani, Arthur Guez, Fabio Viola, Sims Witherspoon, Thomas Anthony, Lars Buesing, Petar Velickovi ˇ c, and Th ´ eophane Weber. On the role of ´ planning in model-based deep reinforcement learning. *arXiv preprint arXiv:2011.04021*, 2020.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770–778, 2016.

James A Hendler, Austin Tate, and Mark Drummond. Ai planning: Systems and techniques. AI
magazine, 11(2):61–61, 1990.

Jacob Hilton, Nick Cammarata, Shan Carter, Gabriel Goh, and Chris Olah. Understanding rl vision.

Distill, 2020. doi: 10.23915/distill.00029. https://distill.pub/2020/understanding-rl-vision.

Sepp Hochreiter and Jurgen Schmidhuber. Long short-term memory. ¨ *Neural computation*, 9(8):
1735–1780, 1997.

Michael Ivanitskiy, Alexander F Spies, Tilman Rauker, Guillaume Corlouer, Christopher Mathwin, ¨
Lucia Quirke, Can Rager, Rusheb Shah, Dan Valentine, Cecilia Diniz Behn, et al. Linearly structured world representations in maze-solving transformers. In Proceedings of UniReps: the First Workshop on Unifying Representations in Neural Models, pp. 133–143. PMLR, 2024.

Rahul Iyer, Yuezhang Li, Huao Li, Michael Lewis, Ramitha Sundar, and Katia Sycara. Transparency and explanation in deep reinforcement learning neural networks. In Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society, pp. 144–150, 2018.

Stanisław Jastrzebski, Devansh Arpit, Nicolas Ballas, Vikas Verma, Tong Che, and Yoshua Bengio.

Residual connections encourage iterative inference. In International Conference on Learning Representations, 2018.

Erik Jenner, Shreyas Kapur, Vasil Georgiev, Cameron Allen, Scott Emmons, and Stuart Russell. Evidence of learned look-ahead in a chess-playing neural network. *arXiv preprint arXiv:2406.00877*, 2024.

Kristopher T. Jensen, Guillaume Hennequin, and Marcelo G. Mattar. A recurrent network model of planning explains hippocampal replay and human behavior. *Nature Neuroscience*, 27(7):1340– 1348, Jul 2024. ISSN 1546-1726. doi: 10.1038/s41593-024-01675-7. URL https://doi. org/10.1038/s41593-024-01675-7.

Hermann Kaindl and Gerhard Kainz. Bidirectional heuristic search reconsidered. Journal of Artificial Intelligence Research, 7:283–317, 1997.

Adam Karvonen. Emergent world models and latent variable estimation in chess-playing language models. *arXiv preprint arXiv:2403.15498*, 2024.

Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, et al.

Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav). In *International conference on machine learning*, pp. 2668–2677. PMLR, 2018.

Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), San Diega, CA, USA, 2015.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. *Advances in neural information processing systems*, 35:22199–22213, 2022.

Richard E. Korf. Planning as search: A quantitative approach. *Artificial Intelligence*, 33(1):65–88, 1987. ISSN 0004-3702. doi: https://doi.org/10.1016/0004-3702(87)90051-8.

Hang Lai, Jian Shen, Weinan Zhang, and Yong Yu. Bidirectional model-based policy optimization.

In *International Conference on Machine Learning*, pp. 5618–5627. PMLR, 2020.

Su Young Lee, Choi Sungik, and Sae-Young Chung. Sample-efficient deep reinforcement learning via episodic backward update. *Advances in neural information processing systems*, 32, 2019.

Lucas Lehnert, Sainbayar Sukhbaatar, Paul Mcvay, Michael Rabbat, and Yuandong Tian. Beyond a*: Better planning with transformers via search dynamics bootstrapping. arXiv preprint arXiv:2402.14083, 2024.

Kenneth Li, Aspen K Hopkins, David Bau, Fernanda Viegas, Hanspeter Pfister, and Martin Watten- ´
berg. Emergent world representations: Exploring a sequence model trained on a synthetic task. ICLR, 2023.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id= Bkg6RiCqY7.

Charles Lovering, Jessica Forde, George Konidaris, Ellie Pavlick, and Michael Littman. Evaluation beyond task performance: analyzing concepts in alphazero in hex. Advances in Neural Information Processing Systems, 35:25992–26006, 2022.

Marcelo G. Mattar and Mat´ e Lengyel. Planning in the brain. ´ *Neuron*, 110(6):914–934, 2022.

ISSN 0896-6273. doi: https://doi.org/10.1016/j.neuron.2021.12.018. URL https://www. sciencedirect.com/science/article/pii/S0896627321010357.

Thomas McGrath, Andrei Kapishnikov, Nenad Tomasev, Adam Pearce, Martin Wattenberg, Demis ˇ
Hassabis, Been Kim, Ulrich Paquet, and Vladimir Kramnik. Acquisition of chess knowledge in alphazero. *Proceedings of the National Academy of Sciences*, 119(47):e2206625119, 2022.

Tomas Mikolov, Wen-tau Yih, and Geoffrey Zweig. Linguistic regularities in continuous space word representations. In Lucy Vanderwende, Hal Daume III, and Katrin Kirchhoff (eds.), ´ Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 746–751, Atlanta, Georgia, June 2013. Association for Computational Linguistics. URL https://aclanthology.org/N13-1090.

Kevin J. Miller, Matthew M. Botvinick, and Carlos D. Brody. Dorsal hippocampus contributes to model-based planning. *Nature Neuroscience*, 20(9):1269–1276, Sep 2017. ISSN 1546-1726. doi: 10.1038/nn.4613. URL https://doi.org/10.1038/nn.4613.

Ulisse Mini, Peli Grietzer, Mrinank Sharma, Austin Meek, Monte MacDiarmid, and Alexander Matt Turner. Understanding and controlling a maze-solving policy network. arXiv preprint arXiv:2310.08043, 2023.

Andrew W. Moore and Christopher G. Atkeson. Prioritized sweeping: Reinforcement learning with less data and less time. *Machine Learning.*, 13(1):103–130, 1993. ISSN 0885-6125. doi:
10.1023/A:1022635613229. URL https://doi.org/10.1023/A:1022635613229.

Neel Nanda, Andrew Lee, and Martin Wattenberg. Emergent linear representations in world models of self-supervised sequence models. In Proceedings of the 6th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, pp. 16–30, 2023.

Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. Show your work: Scratchpads for intermediate computation with language models. *arXiv preprint* arXiv:2112.00114, 2021.

Junhyuk Oh, Satinder Singh, and Honglak Lee. Value prediction network. Advances in neural information processing systems, 30, 2017.

Razvan Pascanu, Yujia Li, Oriol Vinyals, Nicolas Heess, Lars Buesing, Sebastien Racaniere, David `
Reichert, Theophane Weber, Daan Wierstra, and Peter Battaglia. Learning model-based planning ´ from scratch. *arXiv preprint arXiv:1707.06170*, 2017.

Nikaash Puri, Sukriti Verma, Piyush Gupta, Dhruv Kayastha, Shripad Deshmukh, Balaji Krishnamurthy, and Sameer Singh. Explain your move: Understanding agent actions using specific and relevant feature attribution. In *International Conference on Learning Representations*, 2020.

Sebastien Racani ´ ere, Th ` eophane Weber, David Reichert, Lars Buesing, Arthur Guez, Danilo ´
Jimenez Rezende, Adria Puigdom ` enech Badia, Oriol Vinyals, Nicolas Heess, Yujia Li, et al. ` Imagination-augmented agents for deep reinforcement learning. Advances in neural information processing systems, 30, 2017.

Jonathan Richens and Tom Everitt. Robust agents learn causal world models. arXiv preprint arXiv:2402.10877, 2024.

Christian Rupprecht, Cyril Ibrahim, and Christopher J Pal. Finding and visualizing weaknesses of deep reinforcement learning agents. In *International Conference on Learning Representations*, 2020.

Stuart Russell and Peter Norvig. *Artificial Intelligence: A Modern Approach*. Prentice Hall, 3 edition, 2010.

Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, Timothy Lillicrap, and David Silver. Mastering atari, go, chess and shogi by planning with a learned model. *Nature*, 588 (7839):604–609, December 2020. ISSN 1476-4687. doi: 10.1038/s41586-020-03051-4. URL
http://dx.doi.org/10.1038/s41586-020-03051-4.

Lisa Schut, Nenad Tomasev, Tom McGrath, Demis Hassabis, Ulrich Paquet, and Been Kim. Bridging the human-ai knowledge gap: Concept discovery and transfer in alphazero. *arXiv preprint* arXiv:2310.16410, 2023.

Pedro Sequeira and Melinda Gervasio. Interestingness elements for explainable reinforcement learning: Understanding agents' capabilities and limitations. *Artificial Intelligence*, 288:103367, 2020.

Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-Kin Wong, and Wang-chun Woo.

Convolutional lstm network: A machine learning approach for precipitation nowcasting. Advances in neural information processing systems, 28, 2015.

Yaron Shoham and Gal Elidan. Solving sokoban with forward-backward reinforcement learning, 2021.

David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. *nature*, 529(7587):484–489, 2016.

David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, Timothy Lillicrap, Karen Simonyan, and Demis Hassabis. A general reinforcement learning algorithm that masters chess, shogi, and go through self-play. *Science*, 362(6419):1140–1144, 2018. doi: 10.1126/
science.aar6404. URL https://www.science.org/doi/abs/10.1126/science. aar6404.

Nathan R Sturtevant, Shahaf Shperberg, Ariel Felner, and Jingwei Chen. Predicting the effectiveness of bidirectional heuristic search. In *Proceedings of the International Conference on Automated* Planning and Scheduling, volume 30, pp. 281–290, 2020.

Richard S Sutton. Dyna, an integrated architecture for learning, planning, and reacting. *ACM Sigart* Bulletin, 2(4):160–163, 1991.

Richard S. Sutton and Andrew G. Barto. *Reinforcement Learning: An Introduction*. A Bradford Book, Cambridge, MA, USA, 2018. ISBN 0262039249.

Mohammad Taufeeque, Philip Quirke, Maximilian Li, Chris Cundy, Aaron David Tucker, Adam Gleave, and Adria Garriga-Alonso. Planning in a recurrent neural network that plays sokoban. ` arXiv preprint arXiv:2407.15421, 2024.

Hado P Van Hasselt, Matteo Hessel, and John Aslanides. When to use parametric models in reinforcement learning? *Advances in Neural Information Processing Systems*, 32, 2019.

Boshi Wang, Xiang Yue, Yu Su, and Huan Sun. Grokked transformers are implicit reasoners: A
mechanistic journey to the edge of generalization. *arXiv preprint arXiv:2405.15071*, 2024.

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. *arXiv preprint arXiv:2206.07682*, 2022.

Laurens Weitkamp, Elise van der Pol, and Zeynep Akata. Visual rationalizations in deep reinforcement learning for atari games. In Artificial Intelligence: 30th Benelux Conference, BNAIC 2018,'s-Hertogenbosch, The Netherlands, November 8–9, 2018, Revised Selected Papers 30, pp. 151–165. Springer, 2019.

Tom Zahavy, Nir Ben-Zrihem, and Shie Mannor. Graying the black box: Understanding dqns. In International conference on machine learning, pp. 1899–1908. PMLR, 2016.

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. Representation engineering: A
top-down approach to ai transparency. *arXiv preprint arXiv:2310.01405*, 2023.

# Appendix

| Table of Contents A Additional Investigations of Internal Planning   | 17                                                                                                                     |    |    |
|----------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|----|----|
| A.1                                                                  | Further Examples of Internal Plans                                                                                     | 17 |    |
| A.2                                                                  | Further Examples of Internal Plan Formation                                                                            |    | 21 |
| A.3                                                                  | Further Results Regarding Iterative Plan Refinement                                                                    | 35 |    |
| B                                                                    | Additional Intervention Results                                                                                        | 39 |    |
| B.1                                                                  | Additional Examples of Interventions                                                                                   |    | 39 |
| B.2                                                                  | Additional Intervention Experiments: Further Agent-Shortcut and Box-Shortcut Intervention Experiments                  | 42 |    |
| B.3                                                                  | Additional Intervention Experiments: Intervening in a New Set of Levels To Encourage Optimal Behavior                  | 48 |    |
| C                                                                    | Additional Training-Time Interpretability Results                                                                      | 49 |    |
| C.1                                                                  | Investigating the Emergence of Planning-Relevant Concept Representations During Training                                                                                                                        | 51 |    |
| C.2                                                                  | Investigating The Emergence of Test-Time Plan Refinement During Training . .                                           | 51 |    |
| C.3                                                                  | Investigating the Co-Emergence of Planning-Relevant Concept Representations and Planning-Like Behavior During Training | 51 |    |
| C.4                                                                  | Investigating the Co-Emergence of Test-Time Plan Refinement and PlanningLike Behavior During Training                                                                                                                        | 53 |    |
| D                                                                    | Additional Probing Results                                                                                             | 54 |    |
| D.1                                                                  | Probe Training Details                                                                                                 | 54 |    |
| D.2                                                                  | Additional Probing Metrics                                                                                             | 55 |    |
| D.3                                                                  | Probing Using Larger Probes                                                                                            | 58 |    |
| D.4                                                                  | Probing For Alternative Square-Level Concepts                                                                          | 58 |    |
| D.5                                                                  | Probing For Future Actions                                                                                             | 60 |    |
| E                                                                    | Additional Background Material                                                                                         | 61 |    |
| E.1                                                                  | Decision-Time Planning                                                                                                 |    | 62 |
| E.2                                                                  | Sokoban                                                                                                                |    | 63 |
| E.3                                                                  | Deep Repeated ConvLSTM (DRC) Agent Architecture                                                                        |    | 63 |
| E.4                                                                  | DRC Agent Training Details                                                                                             | 65 |    |
| E.5                                                                  | Behavioral Evidence of Planning Exhibited By The DRC Agent                                                             |    | 65 |
| E.6                                                                  | Operationalizing Concepts                                                                                              | 66 |    |
| E.7                                                                  | Application of Methodology to Other Model-Free Architectures                                                           |    | 66 |
| F                                                                    | Investigating Planning in DRC Agents of Different Sizes                                                                | 67 |    |
| F.1                                                                  | Investigating Planning in a DRC(1,9) Agent                                                                             | 67 |    |
| F.2                                                                  | Investigating Planning in a DRC(9,1) Agent                                                                             | 69 |    |
| G                                                                    | Investigating Planning in a Different Architecture: ResNet                                                             | 70 |    |
| H                                                                    | Investigating Planning in a Different Environment: Mini Pacman                                                         | 76 |    |
| H.1                                                                  | Mini PacMan                                                                                                            | 76 |    |
| H.2                                                                  | Preliminary Probing Results                                                                                            |    | 77 |

## A Additional Investigations Of Internal Planning

In Section 5, we provide evidence suggestive of the agent possessing a search-based internal planning mechanism. In this section, we now provide further complementary evidence regarding the agent's internal planning procedure. This section proceeds as follows:
- Appendix A.1 provides further examples of the agent's internal plan at all layers. - Appendix A.2 provides additional examples of the agent forming plans in a manner suggestive of a search-based planning algorithm.

- Appendix A.3 provides additional investigations of the agent's ability to use extra test-time compute to improve its plans.

## A.1 Further Examples Of Internal Plans

In Figure 5 we provided examples of 'internal plans' formulated by the agent. We understood the agent's internal plans to consist of its internal representations, for each square of its observed Sokoban board, of CA and CB. In Figure 5, all internal plans were decoded from the agent's final layer cell state. In this section we now provide additional examples of internal plans formulated by the agent as decoded from its cell state at each layer. Figures 10, 11 and 12 show internal plans decoded from the agent's first, second, and third layer cell states in many different levels. Specifically, Figure 10 shows the agent's internal plans at each layer at six transitions where the agent's internal plan as decoded from its first-layer cell state corresponds to a complete plan to solve the respective level. Similarly, Figures 11 and 12 show the agent's internal plan at each layer at six transitions where the agent's internal plan as respectively decoded from its second- and third-layer cell state correspond to a complete plan to solve the respective levels. We note that the observations we made regarding Figure 5 likewise hold here. That is, (1) the arrows tend to form connected paths, (2) the agent's plans tend to connect specific boxes to specific targets, and (3) the agent often forms complete plans to solve levels very early on in episodes. Note, however, that the agent's plans in Figures 10, 11 and 12 often contain mistakes. This is despite the illustrated transitions being selected such that the agent's plan is correct in at least one layer. A few things can be noted about these mistakes. First, the agent's plans for box movements contain, on average, far fewer mistakes than the agent's plans for its own movements. Second, the mistakes in the agent's plan for its own movements are usually minor and consist of e.g. a few arrows being wrong, but the overall 'shape' of the plan being correct. Third, the agent's mistakes when planning its own movements in the examples tend to be mistakes regarding how it can move when not pushing boxes. We think these observations suggest that the agent is primarily planning by constructing plans in terms of CB connecting boxes and targets, and then augmenting these plans with planned agent movements where needed. At a high-level, we suspect that mistakes in the agent's plan are best seen as relating to intermediate steps of the agent's internal planning process. First, this is because many mistakes seems to be plans that the agent considers on its way to arriving at its final plan. This is because mistakes are almost always fixed in future transitions. Second, some mistakes seem to be temporarily added to the agent's otherwise-correct plan at specific layers. We believe these mistakes potentially relate to the fact that, as part of its planning process, the agent sometimes considers variations on its plan.

(a) Example 1 - Layer 1 (b) Example 1 - Layer 2 (c) Example 1 - Layer 3 (d) Example 2 - Layer 1 (e) Example 2 - Layer 2 (f) Example 2 - Layer 3 (g) Example 3 - Layer 1 (h) Example 3 - Layer 2 (i) Example 3 - Layer 3
(j) Example 4 - Layer 1 (k) Example 4 - Layer 2 (l) Example 4 - Layer 3
(m) Example 5 - Layer 1 (n) Example 5 - Layer 2 (o) Example 5 - Layer 3
(p) Example 6 - Layer 1 (q) Example 6 - Layer 2 (r) Example 6 - Layer 3 (a) Example 1 - Layer 1 (b) Example 1 - Layer 2 (c) Example 1 - Layer 3 (d) Example 2 - Layer 1 (e) Example 2 - Layer 2 (f) Example 2 - Layer 3 (g) Example 3 - Layer 1 (h) Example 3 - Layer 2 (i) Example 3 - Layer 3
(j) Example 4 - Layer 1 (k) Example 4 - Layer 2 (l) Example 4 - Layer 3
(m) Example 5 - Layer 1 (n) Example 5 - Layer 2 (o) Example 5 - Layer 3
(p) Example 6 - Layer 1 (q) Example 6 - Layer 2 (r) Example 6 - Layer 3 (a) Example 1 - Layer 1 (b) Example 1 - Layer 2 (c) Example 1 - Layer 3 (d) Example 2 - Layer 1 (e) Example 2 - Layer 2 (f) Example 2 - Layer 3 (g) Example 3 - Layer 1 (h) Example 3 - Layer 2 (i) Example 3 - Layer 3
(j) Example 4 - Layer 1 (k) Example 4 - Layer 2 (l) Example 4 - Layer 3
(m) Example 5 - Layer 1 (n) Example 5 - Layer 2 (o) Example 5 - Layer 3
(p) Example 6 - Layer 1 (q) Example 6 - Layer 2 (r) Example 6 - Layer 3