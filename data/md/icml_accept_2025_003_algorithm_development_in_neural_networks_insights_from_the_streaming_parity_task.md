# Algorithm Development In Neural Networks: Insights From The Streaming Parity Task

Loek van Rossem 1 **Andrew M. Saxe** 1 2

## Abstract

Even when massively overparameterized, deep neural networks show a remarkable ability to generalize. Research on this phenomenon has focused on generalization within distribution, via smooth interpolation. Yet in some settings neural networks also learn to extrapolate to data far beyond the bounds of the original training set, sometimes even allowing for infinite generalization, implying that an algorithm capable of solving the task has been learned. Here we undertake a case study of the learning dynamics of recurrent neural networks (RNNs) trained on the streaming parity task in order to develop an effective theory of algorithm development. The streaming parity task is a simple but nonlinear task defined on sequences up to arbitrary length. We show that, with sufficient finite training experience, RNNs exhibit a phase transition to perfect infinite generalization. Using an effective theory for the representational dynamics, we find an implicit representational merger effect which can be interpreted as the construction of a finite automaton that reproduces the task. Overall, our results disclose one mechanism by which neural networks can generalize infinitely from finite training experience.

## 1. Introduction

Examples of computational algorithms appearing in deep networks are numerous (Olah et al., 2020; Goh et al., 2021; Wang et al., 2022; Power et al., 2022; Nanda et al., 2023; Zhong et al., 2023). Furthermore, recurrent neural networks and transformers have shown a noteworthy ability to generalize (Loula et al., 2018; Lake & Baroni, 2018; Brown et al.,
1Gatsby Computational Neuroscience Unit, University College London 2Sainsbury Wellcome Centre, University College London. Correspondence to: Loek van Rossem <loek.rossem.22@ucl.ac.uk>.

Training data 0101 0 001 1 11 0 Loss Validation data 1010111001101 0 1 0001001100 0.50 0.25 Epochs 0 500 1000 0.00
Figure 1. Sequence length generalization of a recurrent network on a simple task. **Left**: Example sequences of the streaming parity task, from the training dataset (short sequences) and the validation dataset (long sequences). **Right**: The mean squared loss of a recurrent neural network during training on the streaming parity task. Note that the loss also goes to zero for sequences much longer than found in the training data. This was tested for sequences up to length 10000.

2020) in particular to sequences with lengths not seen in the training data (Dai et al., 2022; Abbe et al., 2023; Cohen- Karlik et al., 2023). These results are surprising as gradient descent provides no clear incentive to generalize beyond the training domain. Why do deep learning systems sometimes develop proper computational algorithms, instead of simply interpolating the data? Understanding this is crucial for the safe application of machine learning models, as for instance, models can appear to generalize initially, but break after moving too far away from the training domain (Anil et al., 2022; Zhou et al., 2024). Similarly, in neuroscience, an important goal is to try to figure out the types of algorithms the brain employs. This is particularly difficult since the brain is so complex, interconnected, and poorly understood (Thompson & Best, 1989; Olshausen & Field, 2006) that it is unclear how to even recognize an algorithm when we see it. Instead, studying the dynamics of algorithm development in the brain might be more tractable (Richards et al., 2019), as learning rules may not be as complex as the learned algorithm itself. Despite attempts to understand these dynamics (Zhou et al., 2021; El-Gaby et al., 2024), it is still a highly challenging problem by itself. Here we hope to provide a better sense of what to look for in the brain, by answering analogous questions in a simpler setting.

1 Let us consider a relatively simple computational problem for which algorithm development can still be studied: the streaming parity task (Figure 1). Given a sequence of zeros and ones with varying length, the aim of the task is to output a zero when the number of ones is even and output a one when the number of ones is odd. A recurrent neural network trained only on sequences up to some short finite length will sometimes generalize infinitely. It is able to solve the task accurately for any sequence length no matter how large, even for sequences thousands of times longer than shown in the training data. As longer sequences are not within the same domain as shorter sequences, the generalization cannot simply be explained by interpolation. One can continue feeding in symbols and the network will continue predicting correctly, suggesting it has somehow learned a computational algorithm. It is unclear how such an algorithm can develop during training from gradient descent optimization. The network is trained to reduce its loss only on the shorter sequence dataset; it is not penalized for breaking after a certain length.

The main goal of this paper is to find simple mathematical models able to explain this seemingly surprising behavior. Our main contributions are as follows:
- In Section 3, we provide a local interaction theory for representational learning dynamics in recurrent neural networks.

- In Section 4, we explain how these interactions can result in the development of an algorithm capable of out-of-distribution generalization.

- In Section 5, we find algorithm development occurs in two phases: an initial tree fitting phase, and a secondary generalization phase.

## 2. Automata And Recurrent Neural Networks 2.1. Interpreting Recurrent Neural Networks

The first step to understanding the development of computational algorithms in recurrent neural networks, is to choose the right way of representing the information that defines the network. The model's parameters are a complete but poor representation. They also contain a large amount of redundant information, e.g. swapping the order of neurons does not affect the encoded algorithm in any way. For analyzing the inner structure of neural networks a better approach is to consider the geometry of the representational structure, i.e. how are the hidden activations corresponding to the data structured in the network (Lin et al., 2019; Williams et al., 2022; Lin & Kriegeskorte, 2023). This is however, still not ideal in the context of computational algorithms. Although understanding how data is represented in the network may be helpful, it does not directly say much about the nature of the computations being performed on that representational space. Moreover, the representational geometry can vary greatly across RNN architectures trained on the same task (Maheswaranathan et al., 2019). One common approach to interpret RNNs is with dynamical systems (Sussillo & Barak, 2013; Laurent & Brecht, 2016; Can et al., 2020; Driscoll et al., 2024). Here, due to our focus on computational algorithms, we will instead be interpreting the recurrent neural network using deterministic finite automata (DFA), an approach also well known in the RNN literature (Servan-Schreiber et al., 1989; Giles et al., 1992; Omlin & Giles, 1996; Tino et al., 1999; Weiss et al., 2020; Merrill & Tsilivis, 2022; Michaud et al., 2024), which has also seen some applications in a neuroscience context (Turner et al., 2021; Brennan et al., 2023). A DFA consists of a set of states including an initial state, transitions between the states given input symbols, and an output symbol for each state. An example of such a DFA
solving the streaming parity task can be seen in Figure 2.

1 0 0 output 1 initial output 0 1
Figure 2. Example of a two state DFA solving the streaming parity task. Every time it receives a 1 as an input it alternates between the state that will output 0 and the state that will output 1. When it receives a 0 it remains in its current state.

## 2.2. Automaton Extraction

We will use a relatively simple method for constructing an automaton from the representational space of an RNN, as this will be enough for our purposes. Consider an abstract recurrent neural network:

(1)  $\frac{1}{2}$ .............................. 
$$\begin{array}{l}{{h_{t}=f_{h}(h_{t-1},x_{t})}}\\ {{y_{t}=f_{y}(h_{t})}}\end{array},$$
where the exact forms of the recurrent map fh and output map fy will depend on the architectural details of the network. From the hidden representations of this network we can extract a deterministic finite automaton. States are defined to be the hidden representations after receiving each sequence, transitions are determined by following where states go after the network received an input symbol, and outputs are simply the network's output map evaluated at each state. This procedure is illustrated in Figure 3. When two different sequences of input symbols are assigned the same internal activation vector, an additional application

h 1 1 1 0output y initial 1 10 0 0 0 0 1
of the recurrent map will send them to the same internal activation vector for any possible input symbol received. Their activations will remain the same in the future, and the output map will always assign them the same output from that point on. They can no longer be distinguished, so we will consider them to be the same state in the automaton. If enough representations overlap, it may be the case that all transitions go to already existing states. At this point, we have a finite set of states capable of representing all possible internal states of the network. These states, together with the transitions between them and their outputs, form a discrete computational algorithm capable of producing outputs on input sequences, which match the outputs of the RNN. For more details on automaton extraction, see Appendix A.

## 2.3. Automata During Training

In order to visualize the development of an algorithm, we extract automata at each epoch from an RNN during training on the streaming parity task (Figure 4). We can see that, due to random small weights, the automaton initially has few states and random outputs. As the model trains, the automaton expands into a complete tree fitting the training data. Then, right when the training loss becomes zero, states in the automaton appear to merge until it becomes finite, and we see generalization on all data. To understand this better, we will first try to model the representational dynamics, and then study the induced dynamics on the automaton.

## 3. Implicit State Merger 3.1. Intuition

Why would states merge during training? The representational space is typically high-dimensional, so it seems statistically unlikely that many different representations end up at the same vector by chance. The key insight here is that, due to continuity, sometimes the fastest way for gradient descent to minimize the loss is to merge nearby representations. As an example for illustration, suppose that at some point during training, two sequences in the dataset agree on target outputs, and one already has the correct predicted output.

Then, if the recurrent map fh adjusts to move the other representation closer, its target output will also move towards the correct prediction, as the output map fy is continuous.

Continuity thus gives rise to an interaction effect between nearby representations, potentially resulting in merging. Implicit bias from gradient descent is a well-studied topic in the deep learning literature (Neyshabur et al., 2015; Gunasekar et al., 2018; Chizat & Bach, 2020; Soudry et al., 2022). However, the relatively simple effect we are considering here will turn out to be particularly interesting in the context of algorithm development in RNNs.

## 3.2. Interaction Model

Let us attempt to formalize this intuition by modeling the interaction between two nearby states, using the modeling approach from (van Rossem & Saxe, 2024), adapted for recurrent networks. Suppose that we have two input sequences x
(1)
1*· · ·* x
(m1)
1and x
(1)
2*· · ·* x
(m2)
2in our dataset, with corresponding hidden representations h(x
(1)
1*· · ·* x
(m1)
1), h(x
(1)
2*· · ·* x
(m2)
2). We would like to understand the behavior of these representations when they get near each other during training, in an arbitrary neural network. Arbitrary architecture with high expressivity. Instead of analyzing this interaction for a specific recurrent architecture, we model it for an arbitrary network with high expressivity, meaning any network large and complex enough to have the freedom to behave like a smooth map. Under this assumption, we replace the effect of the assignment of hidden states by the network's parameters with two arbitrarily optimizable vectors h1, h2, and replace the effect of the network assigning output prediction to those hidden states with arbitrarily optimizable smooth maps y1*, . . . , y*N (schematized in Figure 5). Note that because we are dealing with a recurrent architecture, there may be multiple data points which have h(x
(1)
1*· · ·* x
(m1)
1) or h(x
(1)
2*· · ·* x
(m2)
2) as intermediate hidden states. Thus, we have N output maps, one for each possible sequence after x
(1)
1*· · ·* x
(m1)
1and x
(1)
2*· · ·* x
(m2)
2.

Training loss Validation loss Number of states Lo ss Initial state State with all outgoing edges shown Epochs State with unshown outgoing edges
To simplify the theoretical analysis, we assume here that any sequence x
(1)
1*· · ·* x
(m1)
1 x˜
(1)*· · ·* x˜
(ni)in the dataset has a matching sequence x
(1)
2*· · ·* x
(m2)
2 x˜
(1)*· · ·* x˜
(ni)and vice versa, as we do not expect pairs for which this is not the case to contribute to the interaction.

h y
We will take gradients with respect to these arbitrary smooth maps. While a neural network may not optimize with smooth map dynamics and the gradient will depend on the architectural details, universal approximation theorems (Hornik et al., 1989; Csaji ´ , 2001) tell us that an expressive one is at least able to freely model smooth maps. Therefore, we are modeling a simple behavior that, at least in principle, any expressive network can exhibit. The exact details of the architecture are not required to understand the key mechanisms by which the network learns to solve the task. Abstracting them away may help to find simpler and more intuitive explanations.

Note also that the maps h1, h2, y1*, . . . y*N may share parameters. They are not independent smooth maps. However, if the network is expressive enough, it can still have enough freedom to effectively optimize each map independently at specific points, so we will choose to ignore potential interaction effects arising from parameter sharing. Local linear approximation. As we are trying to model the interaction between two representations, we will consider the case where the distance dh := h2 − h1 between them is small. We can thus take linear approximations of the maps y1*, . . . , y*N around the representational mean 1 2
(h2 + h1):

$$y(x_{\alpha}^{(1)}\cdots x_{\alpha}^{(n_{\alpha})}\bar{x}_{i}^{(1)}\cdots\bar{x}_{i}^{(m_{i})})\approx\bar{y}_{i}+\frac{1}{2}D_{y_{i}}(h_{\alpha}-h_{\rightarrow\alpha}),\tag{2}$$
where $\alpha\in1,2$ and $i\in1,\cdots,N$.  
In general, the dynamics are undefined without specifying the parameterization. However, for the local linearized system, there is a unique choice of parameterindependent dynamics, namely by optimizing with respect to the effective linear parameters of the network h1, h2, y¯1, . . . , y¯N , Dy1
, . . . , DyN . For the mean squared loss

$$L=\frac{1}{2}\langle||\bar{y}_{i}+\frac{1}{2}D_{y_{i}}(h_{\alpha}-h_{\rightarrow\alpha})-y_{\alpha,i}^{*}||^{2}\rangle_{\alpha=1,2,i=1...,N},\tag{3}$$

after taking the continuous-time limit and considering solutions where representations either move closer or further away form each other, we find the self-contained 3-scalar system

d dt ||dh||2 = − 1 2 1 τh ⟨wi⟩i d dt ⟨||dyi||2⟩i = − 1 2 (1 Nτy ||dh||2 + 1 τh ⟨||dyi||2⟩i ||dh||2)⟨wi⟩i d dt ⟨wi⟩i = − 1 4 1 Nτy (3⟨wi⟩i − ⟨||dyi||2⟩i + ⟨||y ∗ 2,i − y ∗ 1,i||2⟩i)||dh||2 − 1 4 1 τh ⟨wi⟩i ||dh||2 (⟨||dyi||2⟩i + ⟨wi⟩i), (4)
where ||dh||2is the squared representational distance,
⟨||dyi||2⟩ is the average squared prediction distance, and ⟨wi⟩i:= ⟨||dyi||2 − dy⊤
i(y
∗
2,i − y
∗
1,i)⟩ithe average of an output alignment metric. The constants 1/τh and 1/τy are the *effective* learning rates of the representational map and output map respectively, as we are now optimizing with respect to these smooth maps as opposed to the model's parameters. The derivation and more details can be found in Appendix B.1. Note that because we are considering the continuous-time limit, no noise has been introduced, and we also did not include any form of regularization. Any results in terms of generalization are from inductive biases in gradient descent alone, e.g. the intuition discussed in Section 3.1. Regularization and noise may also play a crucial role in generalization in some settings (Zhou et al., 2019; Ziyin et al., 2025).

## 3.3. State Merger Condition

The final representational distance is exactly solved in Appendix B.2 to find:

$$||d h||^{2}=\frac{1}{2}A_{\mathrm{high}}+\sqrt{\frac{1}{4}A_{\mathrm{high}}^{2}+A_{\mathrm{low}}^{2}},$$
$$(S)$$

where

$$\begin{array}{l}{{A_{\mathrm{high}}=||d h(0)||^{2}-\frac{N\tau_{y}}{\tau_{h}}\langle\frac{||d y_{i}(0)||^{2}}{||d h(0)||^{2}}\rangle_{i}}}\\ {{A_{\mathrm{low}}=\sqrt{\frac{N\tau_{y}}{\tau_{h}}}\cdot\sqrt{\langle||y_{2,i}^{*}-y_{1,i}^{*}||^{2}\rangle_{i}}.}}\end{array}$$
$$(6)$$

Note that these results are not dependent on the details of the neural network architecture used here. They are the results of an interaction effect locally present in any smooth, expressive machine learning system with hidden representations. A merger occurs when the final representational distance is zero, which is when

$$A_{\mathrm{low}}=0\;\mathrm{and}\;A_{\mathrm{high}}<0.$$
$$(7)$$

Suppose the network's parameters are initialized at some scale G < 1, where G is the average decrease of representational distances when applying the recurrent map. We roughly have

$$\begin{array}{c}{{\vert\vert d h(0)\vert\vert^{2}\propto G^{m}}}\\ {{\langle\frac{\vert\vert d y_{i}(0)\vert\vert^{2}}{\vert\vert d h(0)\vert\vert^{2}}\rangle_{i}\propto G^{n}\ ,}}\end{array}$$
$$(8)$$
$$(9)$$

where m = min (m1, m2) and n = min (n1*, . . . , n*N ) are the minimal sequence lengths of the sequences corresponding to the representations and their potential subsequences respectively. Equation (7) then reduces to the condition

$$\forall_{i}\;y_{1,i}^{*}=y_{2,i}^{*}\;\mathrm{and}\;C<N\cdot G^{n-m},$$

where C is an unknown constant depending on the network architecture. We can make three observations from this:
1. The only states that can merge are those which always agree on outputs after receiving both the same sequence.

2. States only merge if the sequences they correspond to are long enough.

3. Mergers only start to occur given enough data and small enough initial weights.

We will study these observations and their implications in more detail in the next section.

## 4. Automaton Development 4.1. System Of Interacting Particles

The interaction model requires the states to be close to each other. It does not model the global behavior of the dataset during training, only the local interaction between any two states. Many other things may occur during training, which are potentially more complex and exist on a global scale. We will ignore these effects here and treat the representational learning dynamics as a system of locally interacting particles (Liu et al., 2022; Geshkovski et al., 2023). The aim is to investigate how much of algorithm learning in recurrent neural networks can already be explained from this simple interaction alone.

## 4.2. Developing An Algorithm

We can see from Equation (9) that two states will only merge if they agree on target outputs for any possible subsequence in the data, i.e. they need not be distinguished in order to solve the task1. Indeed, out of all 103460 pairs of representations that ended up merging, all agreed on all possible future outputs.

Once enough of such pairs merge, the automaton will become finite. Because the merging of agreeing pairs does not affect the output of the automaton, it will still predict correctly on the training data. If, such as for the streaming parity task, the task can be expressed with a finite automaton, and the training dataset is large enough, the learned automaton and task automaton must agree on all possible sequences. In particular, as long as the training dataset contains all sequences up to the length of the task automaton's size, it is guaranteed that the automata are equivalent, as all its reachable states can be reached with the training data, on which the two automata agree. When the learned automaton becomes finite, the behavior of the RNN becomes fixed for any sequence length, and we should see instant generalization on all lengths. This sudden complete generalization can be observed in Figure 6.

## 4.3. Redundant States

Since weights are initialized small, i.e. G < 1, it follows from Equation (9) that the first pairs to merge when decreasing the weight initialization are the ones for which n − m is minimal. Therefore, representations corresponding to shorter sequence pairs may not merge, even when enough longer sequences do. For the training data used here, the smallest possible n is 0, so we should start to see mergers once m reaches a certain threshold, which can be observed in Figure 7.

1In the language of automata theory, this is equivalent to the absence of a distinguishing extension on the pair of input sequences.

Train 25 100 250 1000 2500 10000 Epochs Seque nce lengt h Sequence length
Because not all pairs agreeing on all future outputs merge, redundant states will be present in the learned automaton. In fact, from Figure 7 it can be seen that not even all of the agreeing pairs that reached the minimum length threshold ended up merging. A possible explanation for this is that some pairs that never end up getting close during training, and therefore the effects from the interaction model do not apply.

It is not necessary for all agreeing pairs to merge to fully generalize, only enough for the final automaton to become finite. Redundant states are consequently expected to be learned. The final automaton in Figure 4 has far more states than the minimal two required to solve the task. However, it is still equivalent in computational function to the two-state automaton shown in Figure 2, which can be seen by merging all its redundant states (Figure 8).

Redundant states in the brain There is some evidence of redundant states in the brain (Morcos & Harvey, 2016; Marmor et al., 2023). From a functional perspective, it is not so clear why these states may exist. However, as can Figure 8. Learned automaton in the RNN from Figure 4 after redundant state reduction using Hopcroft's Algorithm (Appendix A.3). It is equivalent to the automaton representing the streaming parity task Figure 2.

be seen from the above argument, purely from a learning dynamics perspective, they are expected in the final learned automaton. Similar representations containing seemingly identical information have also been observed in other machine learning settings such as wide feed-forward networks (Doimo et al., 2021).

## 4.4. Full Generalization Phase Transition

Finally, we can see from Equation (9) that the merger condition only holds given a small enough initial weight scale G and a large enough number of datapoints N. As either the weight initialization is decreased or the training set size is increased, agreeing states will start to merge. This can be seen on the left side of Figure 9. At some point, enough states will have merged such that the automaton becomes finite, and the RNN will generalize to all sequences. Since no state mergers can occur before the merger condition is reached, we see a sharp boundary in number of states and in particular in the validation accuracy on the right side of Figure 9, splitting the training setting landscape into two regimes: one where it fits the training data with a complete tree, and one where it learns a finite, generalizing representation of the task. Such a separation resembles the phenomenon of rich

Init ial weight s cale Number of states Validation accuracy Training data fraction Training data fraction Initial weight sca le
and lazy learning found in RNNs (Schuessler et al., 2024) and other settings (Chizat et al., 2020; Flesch et al., 2021; Atanasov et al., 2022).

## 5. Merger Dynamics 5.1. Two Development Phases

Something that remains unclear about the dynamics in Figure 4 is the presence of an initial phase where the automaton expands into a complete tree of all possible sequences, before a phase of mergers resulting in a finite algorithm. Why would the RNN learn to memorize individual outputs per input sequence in the first place, only to collapse into a finite automaton later?

## 5.2. Fixed Expansion Point Interaction Model

A relatively simple dynamical explanation for this behavior can be found in the representational drift of each interaction pair. If the two representations during an interaction start to drift, and their effective learning rates 1/τh1, 1/τh2 differ, they will drift at different speeds. In this case, their distance may initially start to increase as one outpaces the other. The local interaction model we have considered does not exhibit this behavior, as the linear expansion point is chosen to be at the moving representational mean, enforcing a representational movement symmetry. In Appendix B.3 the analytical learning trajectory for an agreeing pair is solved for in this model, and their representational distance can be seen to decay exponentially. To allow for enough freedom in the interaction model for both representations to drift freely, we can instead keep the expansion point fixed:

$$y(x_{\alpha}^{(1)}\cdots x_{\alpha}^{(n_{\alpha})}\bar{x}_{i}^{(1)}\cdots\bar{x}_{i}^{(m_{i})})\approx\bar{y}_{i}+D_{y_{i}}h_{\alpha}.\tag{10}$$

The downside of this choice is that as the pair drifts far away from the expansion point, the interaction model may lose accuracy. In Appendix B.4 we use a similar approach as before to reduce these dynamics to a self-contained system of 9 variables.

## 5.3. Diverging Mergers

We can see from numerical solutions to this system (Figure 10) that agreeing pairs initially diverge before they end up merging. This divergence is only present when the effective learning rates 1/τh1, 1/τh2 differ. Experimentally, we find qualitatively similar divergence behavior in the RNN during training. We also see that the divergence occurs more often in pairs with a higher effective learning rate difference (Appendix D.3). Such an initial divergence of many agreeing pairs, may explain the tree fitting phase. A division in two phases with some similarities has been studied in feed-forward networks in (Shwartz-Ziv & Tishby, 2017).

Theory Experiment Training loss Distance agreeing pair Distance disagreeing pair Los s Epochs Epochs Epochs

## 6. Other Settings 6.1. Random Regular Tasks

The ideas considered in this paper should be applicable to any task that can be described by an automaton. Therefore, all experiments were also performed on a set of tasks defined by randomly generated automata. Similar results were found as with the streaming parity task (Appendix D.5).

## 6.2. Architecture Independence

The local interaction models discussed here are universal with respect to the neural network architecture. They represent intuitions that apply to any model with a smooth, expressive recurrent map on some hidden space and a similarly smooth and expressive output map on this space. To illustrate this, we replaced the ReLU activation function with a hyperbolic tangent and found similar qualitative results (Appendix D.6).

## 6.3. Transformers

Transformers have been shown to exhibit a similar ability to learn computational algorithms. One may wonder to what extent the ideas considered here for recurrent networks still apply to the transformer architecture. The interpretation of the internal structure via an automaton is not as clearly applicable to a transformer. The recurrent map was an essential ingredient in the understanding of the formation of a finite automaton, as it allows for mergers to result in automaton transitions going to previous states. Interestingly enough, despite some generalization to larger sequence lengths, transformers fail to fully generalize to sequences of arbitrary length on parity computation (Anil et al., 2022). However, the intuition behind the local interactions from continuity still applies, and so we may still find similar merging dynamics. To investigate this, we compute the number of states in a transformer during training on the modular subtraction task from (Power et al., 2022). As can be seen from Figure 11, we do not find a clear state merger pattern in the hidden representations of the transformer. However, we do see a state merger pattern in the attention matrix that is reminiscent of the two phases found in recurrent networks. A similar pattern was found for a local complexity measure in (Humayun et al., 2024). The merging of attention patterns may possibly play an important role in out-of-distribution generalization, similar to representation merger in recurrent neural networks. Other phenomena observed here in RNNs also resemble behaviors in transformers, such as the sudden transition to full generalization (Hoffmann et al., 2024). The exact way in which a transformer can learn an algorithm from mergers is not as clear and requires further study.

Training loss Validation loss Number of states (attention)
Number of states (hidden)
Lo ss Epochs

## 6.4. Discontinuity

must be specifically highlighted here.

One of the assumptions in the interaction model is continuity of the recurrent and output maps. This may not necessarily be a reasonable assumption in the context of neuroscience, as the spiking coupling between neurons is not typically viewed as continuous. For the intuitions to work, however, we only really need predictions of nearby representations to move closer when the representations do *on average*. Continuity gives us this, but may not be a necessary condition. To explore this, we add a step-continuity in the output map of the recurrent network. We find qualitatively similar results, albeit with noisier dynamics Appendix D.9.

## 7. Conclusion

Despite the surprising nature of infinite generalization from finite data, there exists a setting in which it can be understood through relatively simple intuitions about inductive bias in gradient descent. In this setting, we found that algorithm development occurs in two phases, an initial tree fitting phase and a secondary merging phase that results in generalization. The merging phase occurs via a phase transition, when the right training conditions are met. Therefore, algorithm learning and infinite generalization can occur in deep networks, but not consistently. We also saw that from a dynamical perspective, redundant states are expected in the final learned algorithm. This is of particular interest to neuroscience, as it suggests that different animals may learn different but equivalent versions of an algorithm, which is something that should be taken into account when comparing representations. Finally, we found that intuitions about automaton formation do not apply as well to transformers, and that at least for some specific tasks, recurrent networks have an advantage in terms of infinite generalization. Limitations The theoretical approach used here is relatively simple and not necessarily a realistic model of the complete learning dynamics. Higher-order local interactions, global interactions, inductive biases from architectural choices, regularization, and noise were ignored, but may have additional effects on algorithm development worth studying. Additionally, the interpretation of an RNN as an automaton may provide incomplete information in more complex or continuous data settings. Other mathematical objects may be necessary to properly represent the internal structure of an RNN in such settings.

## Impact Statement

This paper presents work whose goal is to increase understanding of deep learning, which may lead to advancements in the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel

## Acknowledgements

We thank Stefano Sarao Mannelli and Chenxiao Ma for useful feedback. This work was supported by a Sir Henry Dale Fellowship from the Wellcome Trust and Royal Society (216386/Z/19/Z) to A.S., and the Sainsbury Wellcome Centre Core Grant from Wellcome (219627/Z/19/Z) and the Gatsby Charitable Foundation (GAT3755).

## References

Abbe, E., Bengio, S., Lotfi, A., and Rizk, K. Generalization on the Unseen, Logic Reasoning and Degree Curriculum. In Proceedings of the 40th International Conference on Machine Learning, pp. 31–60. PMLR, July 2023. URL https://proceedings.mlr.press/ v202/abbe23a.html. ISSN: 2640-3498.

Adriaensen, R. and Maene, J. Extracting Finite State Machines from Transformers, October 2024. URL http://arxiv.org/abs/2410. 06045. arXiv:2410.06045 [cs].

Anil, C., Wu, Y., Andreassen, A., Lewkowycz, A., Misra, V., Ramasesh, V., Slone, A., Gur-Ari, G., Dyer, E., and Neyshabur, B. Exploring Length Generalization in Large Language Models, November 2022. URL http:// arxiv.org/abs/2207.04901. arXiv:2207.04901
[cs].

Ashby, W. R. *Automata Studies: Annals of Mathematics* Studies. Number 34. Princeton University Press, April 1956. ISBN 978-0-691-07916-5. Google-Books-ID:
oL57iECEeEwC.

Atanasov, A., Bordelon, B., Sainathan, S., and Pehlevan, C.

The Onset of Variance-Limited Behavior for Networks in the Lazy and Rich Regimes, December 2022. URL https://arxiv.org/abs/2212.12147v1.

Brennan, C., Aggarwal, A., Pei, R., Sussillo, D., and Proekt, A. One dimensional approximations of neuronal dynamics reveal computational strategy. PLOS Computational Biology, 19(1):e1010784, January 2023. ISSN 1553-7358. doi: 10.1371/journal.pcbi.1010784.

URL https://journals.plos.org/
ploscompbiol/article?id=10.1371/
journal.pcbi.1010784. Publisher: Public Library of Science.

Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D. M.,
Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D. Language Models are Few-Shot Learners, July 2020. URL http://arxiv.org/abs/2005. 14165. arXiv:2005.14165 [cs].

Can, T., Krishnamurthy, K., and Schwab, D. J. Gating creates slow modes and controls phase-space complexity in GRUs and LSTMs. In Proceedings of The First Mathematical and Scientific Machine Learning Conference, pp. 476–511. PMLR, August 2020. URL https://proceedings.mlr.press/ v107/can20a.html. ISSN: 2640-3498.

Chizat, L. and Bach, F. Implicit Bias of Gradient Descent for Wide Two-layer Neural Networks Trained with the Logistic Loss. In Proceedings of Thirty Third Conference on Learning Theory, pp. 1305–1338. PMLR, July 2020. URL https://proceedings.mlr.press/ v125/chizat20a.html. ISSN: 2640-3498.

Chizat, L., Oyallon, E., and Bach, F. On Lazy Training in Differentiable Programming, January 2020. URL http://arxiv.org/abs/1812. 07956. arXiv:1812.07956 [cs, math].

Cohen-Karlik, E., Menuhin-Gruman, I., Giryes, R., Cohen, N., and Globerson, A. Learning Low Dimensional State Spaces with Overparameterized Recurrent Neural Nets, March 2023. URL http://arxiv.org/abs/ 2210.14064. arXiv:2210.14064 [cs].

Csaji, B. ´ *Approximation with Artificial Neural Networks*.

PhD thesis, June 2001.

Dai, X., Chalkidis, I., Darkner, S., and Elliott, D. Revisiting Transformer-based Models for Long Document Classification, October 2022. URL http://arxiv.org/ abs/2204.06683. arXiv:2204.06683 [cs] version: 2.

Doimo, D., Glielmo, A., Goldt, S., and Laio, A. Representation mitosis in wide neural networks. October 2021. URL https://openreview.net/forum? id=pVU7Gp7Nq4k.

Driscoll, L. N., Shenoy, K., and Sussillo, D. Flexible multitask computation in recurrent networks utilizes shared dynamical motifs. *Nature Neuroscience*, 27(7):1349– 1363, July 2024. ISSN 1546-1726. doi: 10.1038/ s41593-024-01668-6. URL https://www.nature. com/articles/s41593-024-01668-6. Publisher: Nature Publishing Group.

El-Gaby, M., Harris, A. L., Whittington, J. C. R.,
Dorrell, W., Bhomick, A., Walton, M. E., Akam, T., and Behrens, T. E. J. A cellular basis for mapping behavioural structure. *Nature*, 636(8043): 671–680, 2024. ISSN 0028-0836. doi: 10.1038/ s41586-024-08145-x. URL https://www.ncbi. nlm.nih.gov/pmc/articles/PMC11655361/.

Flesch, T., Juechems, K., Dumbalska, T., Saxe, A.,
and Summerfield, C. Rich and lazy learning of task representations in brains and neural networks, April 2021. URL https://www.biorxiv.org/ content/10.1101/2021.04.23.441128v1.

Pages: 2021.04.23.441128 Section: New Results.

Geshkovski, B., Letrouit, C., Polyanskiy, Y., and Rigollet, P. The emergence of clusters in self-attention dynamics. Advances in Neural Information Processing Systems, 36:57026–57037, December 2023.

URL https://people.lids.mit.edu/yp/ homepage/data/2023_transformers1.pdf.

Giles, C. L., Miller, C. B., Chen, D., Chen, H. H., Sun, G. Z., and Lee, Y. C. Learning and Extracting Finite State Automata with Second-Order Recurrent Neural Networks. *Neural Computation*, 4(3):393–405, May 1992. ISSN 0899-7667. doi: 10.1162/neco.1992.4. 3.393. URL https://ieeexplore.ieee.org/ document/6796344. Conference Name: Neural Computation.

Goh, G., †, N. C., †, C. V., Carter, S., Petrov, M.,
Schubert, L., Radford, A., and Olah, C. Multimodal Neurons in Artificial Neural Networks. *Distill*, 6(3): e30, March 2021. ISSN 2476-0757. doi: 10.23915/ distill.00030. URL https://distill.pub/2021/ multimodal-neurons.

Gunasekar, S., Lee, J. D., Soudry, D., and Srebro, N. Implicit Bias of Gradient Descent on Linear Convolutional Networks. In Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018. URL
https://arxiv.org/abs/1806.00468.

Hoffmann, D. T., Schrodi, S., Bratulic, J., Behrmann, N., ´
Fischer, V., and Brox, T. Eureka-Moments in Transformers: Multi-Step Tasks Reveal Softmax Induced Optimization Problems, June 2024. URL http://arxiv. org/abs/2310.12956. arXiv:2310.12956.

Hopcroft, J. An n log n algorithm for minimizing states in a finite automaton. In Kohavi, Z. and Paz, A. (eds.), *Theory* of Machines and Computations, pp. 189–196. Academic Press, January 1971. ISBN 978-0-12-417750-5. doi: 10.1016/B978-0-12-417750-5.50022-1. URL https: //www.sciencedirect.com/science/ article/pii/B9780124177505500221.

Hornik, K., Stinchcombe, M., and White, H. Multilayer feedforward networks are universal approximators. *Neural Networks*, 2(5):359–366, January 1989. ISSN 08936080. doi: 10.1016/0893-6080(89) 90020-8. URL https://linkinghub.elsevier. com/retrieve/pii/0893608089900208.

Humayun, A. I., Balestriero, R., and Baraniuk, R.

Deep Networks Always Grok and Here is Why. In Proceedings of the 41st International Conference on Machine Learning, pp. 20722–20745. PMLR, July 2024. URL https://proceedings.mlr.press/ v235/humayun24a.html. ISSN: 2640-3498.

Lake, B. and Baroni, M. Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks. In Proceedings of the 35th International Conference on Machine Learning, pp. 2873–2882. PMLR, July 2018. URL https://proceedings.mlr.press/v80/ lake18a.html. ISSN: 2640-3498.

Laurent, T. and Brecht, J. v. A recurrent neural network without chaos, December 2016. URL http://arxiv. org/abs/1612.06212. arXiv:1612.06212 [cs].

Lin, B. and Kriegeskorte, N. The Topology and Geometry of Neural Representations, September 2023. URL http://arxiv.org/abs/2309. 11028. arXiv:2309.11028 [cs, q-bio, stat].

Lin, B., Mur, M., Kietzmann, T., and Kriegeskorte, N. Visualizing Representational Dynamics with Multidimensional Scaling Alignment, July 2019. URL http:// arxiv.org/abs/1906.09264. arXiv:1906.09264
[cs, q-bio, stat].

Liu, Z., Kitouni, O., Nolte, N., Michaud, E. J., Tegmark, M., and Williams, M. Towards Understanding Grokking: An Effective Theory of Representation Learning, October 2022. URL http://arxiv.org/abs/2205. 10343. arXiv:2205.10343 [cs].

Loula, J., Baroni, M., and Lake, B. M. Rearranging the Familiar: Testing Compositional Generalization in Recurrent Networks, July 2018. URL http://arxiv. org/abs/1807.07545. arXiv:1807.07545 [cs].

Maheswaranathan, N., Williams, A. H., Golub, M. D., Ganguli, S., and Sussillo, D. Universality and individuality in neural dynamics across large populations of recurrent networks, December 2019. URL http://arxiv.org/ abs/1907.08549. arXiv:1907.08549 [cs, q-bio].

Marmor, O., Pollak, Y., Doron, C., Helmchen, F., and Gilad, A. History information emerges in the cortex during learning. *eLife*, 12:e83702, November 2023. ISSN 2050-084X. doi: 10.7554/eLife.83702. URL https: //doi.org/10.7554/eLife.83702. Publisher:
eLife Sciences Publications, Ltd.

Merrill, W. and Tsilivis, N. Extracting Finite Automata from RNNs Using State Merging, April 2022. URL http:// arxiv.org/abs/2201.12451. arXiv:2201.12451 [cs].

Michaud, E. J., Liao, I., Lad, V., Liu, Z., Mudide, A.,
Loughridge, C., Guo, Z. C., Kheirkhah, T. R., Vukelic,´ M., and Tegmark, M. Opening the AI black box: program synthesis via mechanistic interpretability, February 2024. URL http://arxiv.org/abs/2402. 05110. arXiv:2402.05110 [cs].

Morcos, A. S. and Harvey, C. D. History-dependent variability in population dynamics during evidence accumulation in cortex. *Nature Neuroscience*, 19(12):1672– 1681, December 2016. ISSN 1546-1726. doi: 10.

1038/nn.4403. URL https://www.nature.com/ articles/nn.4403. Publisher: Nature Publishing Group.

Nanda, N., Chan, L., Lieberum, T., Smith, J., and Steinhardt, J. Progress measures for grokking via mechanistic interpretability, October 2023. URL http://arxiv. org/abs/2301.05217. arXiv:2301.05217 [cs].

Neyshabur, B., Tomioka, R., and Srebro, N. In Search of the Real Inductive Bias: On the Role of Implicit Regularization in Deep Learning, April 2015. URL http:
//arxiv.org/abs/1412.6614. arXiv:1412.6614 [cs, stat].

Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M.,
and Carter, S. Zoom In: An Introduction to Circuits, 2020.

URL https://distill.pub/2020/circuits/ zoom-in/.

Olshausen, B. A. and Field, D. J. What Is the Other 85 Percent of V1 Doing? In van Hemmen, J. L. and Sejnowski, T. J. (eds.), *23 Problems in Systems Neuroscience*, pp. 0. Oxford University Press, January 2006. ISBN 978-019-514822-0. doi: 10.1093/acprof:oso/9780195148220.

003.0010. URL https://doi.org/10.1093/ acprof:oso/9780195148220.003.0010.

Omlin, C. W. and Giles, C. L. Constructing deterministic finite-state automata in recurrent neural networks. *Journal of the ACM*, 43(6):937–972, November 1996. ISSN 0004-5411, 1557-735X. doi: 10.1145/ 235809.235811. URL https://dl.acm.org/doi/ 10.1145/235809.235811.

Power, A., Burda, Y., Edwards, H., Babuschkin, I.,
and Misra, V. Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets, January 2022. URL http://arxiv.org/abs/2201.

02177. arXiv:2201.02177 [cs].

Richards, B. A., Lillicrap, T. P., Beaudoin, P., Bengio, Y., Bogacz, R., Christensen, A., Clopath, C., Costa, R. P., de Berker, A., Ganguli, S., Gillon, C. J., Hafner, D., Kepecs, A., Kriegeskorte, N., Latham, P., Lindsay, G. W., Miller, K. D., Naud, R., Pack, C. C., Poirazi, P., Roelfsema, P., Sacramento, J., Saxe, A., Scellier, B., Schapiro, A. C., Senn, W., Wayne, G., Yamins, D., Zenke, F., Zylberberg, J., Therien, D., and Kording, K. P. A deep learning framework for neuroscience. *Nature Neuroscience*, 22(11):1761–1770, November 2019. ISSN 1546-1726. doi: 10.1038/
s41593-019-0520-2. URL https://www.nature. com/articles/s41593-019-0520-2. Number:
11 Publisher: Nature Publishing Group.

Schuessler, F., Mastrogiuseppe, F., Ostojic, S., and Barak, O. Aligned and oblique dynamics in recurrent neural networks, August 2024. URL http://arxiv.org/ abs/2307.07654. arXiv:2307.07654 [q-bio].

Servan-Schreiber, D., Cleeremans, A., and Mcclelland, J.

Encoding Sequential Structure in Simple Recurrent Networks. pp. 40, July 1989.

Shwartz-Ziv, R. and Tishby, N. Opening the Black Box of Deep Neural Networks via Information, April 2017. URL http://arxiv.org/abs/1703. 00810. arXiv:1703.00810 [cs].

Soudry, D., Hoffer, E., Nacson, M. S., Gunasekar, S., and Srebro, N. The Implicit Bias of Gradient Descent on Separable Data, July 2022. URL http://arxiv.org/ abs/1710.10345. arXiv:1710.10345 [cs, stat].

Sussillo, D. and Barak, O. Opening the black box:
low-dimensional dynamics in high-dimensional recurrent neural networks. *Neural Computation*, 25(3):626– 649, March 2013. ISSN 1530-888X. doi: 10.1162/
NECO a 00409.

Thompson, L. T. and Best, P. J. Place cells and silent cells in the hippocampus of freely-behaving rats. The Journal of Neuroscience: The Official Journal of the Society for Neuroscience, 9(7):2382–2390, July 1989. ISSN 02706474. doi: 10.1523/JNEUROSCI.09-07-02382.1989.

Tino, P., Horne, W., Giles, C., and Collingwood, P. Finite State Machines and Recurrent Neural Networks - Automata and Dynamical Systems Approaches. March 1999. ISSN 9780125264204. doi: 10.1016/B978-012526420-4/ 50007-0.

Turner, E., Dabholkar, K., and Barak, O. Charting and navigating the space of solutions for recurrent neural networks, November 2021. URL http://arxiv.org/
abs/2111.09356. arXiv:2111.09356 [q-bio].

van Rossem, L. and Saxe, A. M. When Representations Align: Universality in Representation Learning Dynamics. In Proceedings of the 41st International Conference on Machine Learning, pp. 49098–49121. PMLR, July 2024. URL https://proceedings.mlr.press/ v235/van-rossem24a.html. ISSN: 2640-3498.

Wang, K., Variengien, A., Conmy, A., Shlegeris, B., and Steinhardt, J. Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small, November 2022. URL http://arxiv.org/abs/2211.

00593. arXiv:2211.00593 [cs].

Weiss, G., Goldberg, Y., and Yahav, E. Extracting Automata from Recurrent Neural Networks Using Queries and Counterexamples, February 2020. URL http:// arxiv.org/abs/1711.09576. arXiv:1711.09576
[cs].

Williams, A. H., Kunz, E., Kornblith, S., and Linderman, S. W. Generalized Shape Metrics on Neural Representations, January 2022. URL http://arxiv.org/
abs/2110.14739. arXiv:2110.14739 [cs, stat].

Zhong, Z., Liu, Z., Tegmark, M., and Andreas, J. The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks, November 2023. URL http:// arxiv.org/abs/2306.17844. arXiv:2306.17844 [cs].

Zhou, J., Jia, C., Montesinos-Cartagena, M., Gardner, M. P. H., Zong, W., and Schoenbaum, G. Evolving schema representations in orbitofrontal ensembles during learning. *Nature*, 590(7847):606–611, February 2021. ISSN 1476-4687. doi: 10.1038/
s41586-020-03061-2. URL https://www.nature. com/articles/s41586-020-03061-2. Publisher: Nature Publishing Group.

Zhou, M., Liu, T., Li, Y., Lin, D., Zhou, E., and Zhao, T. Toward Understanding the Importance of Noise in Training Neural Networks. In Proceedings of the 36th International Conference on Machine Learning, pp. 7594–7602.

PMLR, May 2019. URL https://proceedings. mlr.press/v97/zhou19d.html. ISSN: 26403498.

Zhou, Y., Alon, U., Chen, X., Wang, X., Agarwal, R., and Zhou, D. Transformers Can Achieve Length Generalization But Not Robustly, February 2024. URL http:// arxiv.org/abs/2402.09371. arXiv:2402.09371 [cs].

Ziyin, L., Chuang, I., Galanti, T., and Poggio, T. Formation of Representations in Neural Networks, February 2025. URL http://arxiv.org/abs/2410. 03006. arXiv:2410.03006 [cs].

## A. Automaton Extraction

A.1. Definition Deterministic Finite Automaton Formally a deterministic finite automaton (Ashby, 1956) is a tuple (Q, Σ, δ, q0, F), consisting of 1. A finite set of states Q 2. A finite set of possible input symbols Σ, called the alphabet 3. A transition function δ : Q × Σ → Q 4. An initial state q0 5. A subset of accepting states F
Given some string of input symbols x = x
(1)x
(2) *. . . x*(n)the automaton is said to accept the string x when there exists a sequence of states r
(0)*, . . . , r*(n) ∈ Q such that 1. r
(0) = q0 2. ∀i r
(i+1) = δ(r
(i), x(i+1))
3. r
(n) ∈ F
In the context of the streaming parity task we can take the subset of accepting states to be precisely those for which the model predicts an output 1.

## A.2. Extraction Algorithm

In order to extract one from a recurrent neural network, we define the state corresponding to an input string x as the hidden representation in the network after it received that string, i.e.

$$\begin{array}{c}{{q_{0}:=h_{0}}}\\ {{q_{x_{n}\ldots x_{1}}:=f_{h}(q_{x_{n-1}\ldots x_{1}},x_{n})^{2}}}\end{array}$$
$$(11)$$
. (11)
When two states are on top of each other, i.e. the representations are within some small distance ϵ, we will consider them as the same state. Given an evaluation dataset X of input sequences, the set of states can be defined as

$$Q:=\{q_{x}|x\in X\}/(q\sim q^{\prime}\iff||q-q^{\prime}||<\epsilon).$$
′|| < ϵ). (12)
The alphabet Σ is the set of all possible input symbols in the data. The transition function δ(*q, σ*) is given by the state corresponding to fh(q, σ). The set of accepting states F are all states for which fy(q) is closest to the output 1. If we are considering a task with more than two possible output symbols, we can generalize this definition using a Moore machine, which replaces the set of accepting states with an output function.

## A.3. State Reduction Algorithm

To help interpret the final learning automaton within the recurrent neural network, we can reduce it to a minimal state automaton with equivalent outputs. For this we use Hopcroft's Algorithm (Hopcroft, 1971) (see Algorithm 1), which returns the unique smallest DFA, equivalent to the provided DFA. Essentially what this algorithm does is it merges all pairs of states which are indistinguishable for any possible input string.

$$(12)$$

Algorithm 1 Hopcroft's Algorithm Input: set of states Q with output 0, set of states F with output 1 Output: minimal state partition P P := {F, Q \ F} W := {*F, Q* \ F} while W is not empty do choose and remove a set A from W for c in Σ do let X be the set of states for which a transition on c leads to a state in A for set Y in P for which X ∩ Y is nonempty and Y \ X is nonempty do replace Y in P by the two sets X ∩ Y and Y \ X if Y is in W **then**
replace Y in W by the same two sets else if |X ∩ Y *| ≤ |*Y \ X| **then**
add X ∩ Y to W
else add Y \ X to W
end if end if end for end for end while return P

## B. Details Theoretical Analysis B.1. Reduction To A 3-Dimensional System

To model the two-point interaction we consider two sequences x
(1)
1*. . . x*
(m1)
1and x
(1)
2*. . . x*
(m2)
2 with nearby representations h1 = h(x
(1)
1*. . . x*
(m1)
1) respectively h2 = h(x
(1)
2*. . . x*
(m2)
2). Let D = {(x1,i, y∗1,i),(x2,i, y∗
2,i)}
N
i=1 be the set of all datapoints contained within the training dataset which have x
(1)
1*. . . x*
(m1)
1or x
(1)
2*. . . x*
(m2)
2as a subsequence. Assuming high expressivity, we model h1 and h2 as freely optimizable vectors and the output predictions of the network for each subsequent sequence in D as given by smooth, freely optimizable maps yi: H → Y .

In contrast to (van Rossem & Saxe, 2024), we cannot use an optimizable linearized hidden map as here we cannot smoothly vary the inputs for the hidden map. Since our input symbols are discrete, we can instead consider arbitrarily optimizable hidden vectors, as no two differing input symbols can ever get arbitrarily close to each other in the input space.

As h1 and h2 are close, we take a linear approximation of each output prediction map around the representational mean:

$$y(x_{\alpha,i})=\bar{y}_{i}+\frac{1}{2}D_{y_{i}}(h_{\alpha}-h_{\lnot\alpha}).$$
Dyi(hα − h¬α). (13)
The mean squared loss in this approximation takes the form:

$${\cal L}=\frac{1}{2}\langle||\bar{y}_{i}+\frac{1}{2}D_{y_{i}}(h_{\alpha}-h_{\to\alpha})-y_{\alpha,i}^{*}||^{2}\rangle_{\cal D}.$$
∗α,i||2⟩D. (14)
$\left(13\right)^2$
$\eqref{eq:walpha}$. 
We apply gradient decent optimization directly with respect to Dyi
, hα and y¯i, resulting in the dynamics:

$$\begin{array}{c}{{\frac{\mathrm{d}}{\mathrm{d}t}\bar{y}_{i}=-\frac{1}{\tau_{\bar{y}_{i}}}\frac{\partial L}{\partial\bar{y}_{i}}}}\\ {{=-\frac{1}{\tau_{\bar{y}_{i}}}\frac{1}{N}\langle\bar{y}_{i}+\frac{1}{2}D_{y_{i}}(h_{\alpha}-h_{-\alpha})-y_{\alpha,i}^{*}\rangle_{\alpha=1,2}}}\\ {{=-\frac{1}{\tau_{\bar{y}_{i}}}\frac{1}{N}(\bar{y}_{i}-\frac{y_{2,i}^{*}+y_{1,i}^{*}}{2})}}\end{array}$$
$$\frac{d}{dt}h(\alpha)=-\frac{1}{r_{h_{a}}}\frac{\partial L}{\partial h_{\alpha}}$$ $$=-\frac{1}{r_{h_{a}}}\frac{1}{4}(D_{h}^{\top}(\tilde{y}_{h}+\frac{1}{2}D_{y}\left(h_{\alpha}-h_{-\alpha}\right)-y_{\alpha,i}^{*})-D_{h}^{\top}(\tilde{y}_{i}+\frac{1}{2}D_{y_{i}}\left(h_{-\alpha}-h_{\alpha}\right)-y_{-\alpha,i}^{*}))_{i=1,\ldots,N}\tag{15}$$ $$=-\frac{1}{r_{h_{a}}}\frac{1}{4}(D_{h}^{\top}(D_{y_{i}}(h_{\alpha}-h_{-\alpha})-(y_{\alpha,i}^{*}-y_{-\alpha,i}^{*})))_{i=1,\ldots,N}$$
$$(16)$$
$$\frac{\mathrm{d}}{\mathrm{d}t}D_{y_{i}}=-\frac{1}{\tau_{y_{i}}}\,\frac{\partial L}{\partial D_{y_{i}}}$$ $$=-\frac{1}{\tau_{y_{i}}}\,\frac{1}{N}\langle(\frac{1}{2}D_{y_{i}}(h_{\alpha}-h_{-\alpha})(h_{\alpha}-h_{-\alpha})^{\top}+(\bar{y}_{i}-y_{\alpha,i}^{*})(h_{\alpha}-h_{-\alpha})^{\top})\rangle_{\alpha=1,2}$$ $$=-\frac{1}{\tau_{y_{i}}}\,\frac{1}{N}\frac{1}{4}(D_{y_{i}}(h_{2}-h_{1})-(y_{2,i}^{*}-y_{1,i}^{*}))(h_{2}-h_{1})^{\top},$$

where we used the matrix differentiation identities ∂a⊤Xb
∂X = ab⊤,
∂a⊤X⊤CXa
∂X = (C + C
$C^{\top})Xaa^{\top}$ and $\frac{\partial\left\|\right.Ax+b\left\|\right\|^2}{\partial x}$ =. 
2A⊤(Ax + b).

The y¯i dynamics are decoupled and can be solved directly:

$$\bar{y}_{i}(t)=\frac{y_{2,i}^{*}+y_{1,i}^{*}}{2}+(y_{i}(0)-\frac{y_{2,i}^{*}+y_{1,i}^{*}}{2})e^{-\frac{1}{\gamma y_{i}}\frac{1}{N}t},$$

the solution of which takes the form of exponential decay towards each pairs target output mean.

Define dh := h2 − h1, dyi:= Dyi(h2 − h1), wi:= ||dyi||2 − dy⊤
i
(y
∗
2,i − y
∗1,i). We take as an Anzats representational movement of the two points towards or away from each other, i.e.

$${\frac{\mathrm{d}}{\mathrm{d}t}}d h\propto d h\implies{\frac{{\frac{d}{d t}}d h}{||{\frac{d}{d t}}d h||}}={\frac{d h}{||d h||}}\implies{\frac{d}{d t}}d h={\frac{||{\frac{d}{d t}}d h||}{||d h||}}d h$$

Applying this twice allows us to write

$$D_{v_{0}}\frac{d}{dt}dh=\frac{||\frac{d}{dt}dh||}{||dh||}D_{p_{0}}dh=\frac{||\frac{d}{dt}dh||}{||dh||^{2}}D_{p_{0}}dh=\frac{dh^{\top}(||\frac{d}{dt}dh||}{||dh||^{2}}D_{p_{0}}dh=\frac{dh^{\top}\frac{d}{dt}dh||}{||dh||^{2}}D_{p_{0}}dh=\frac{1}{2}\frac{d}{||dh||^{2}}D_{p_{0}}dh=\frac{1}{2}\frac{d}{||dh||^{2}}D_{p_{0}}dh\tag{18}$$
$$(17)$$

which we can use to find a self-contained scalar system:

$$\frac{\mathrm{d}}{\mathrm{d}t}||dh||^{2}=2dh^{\top}\frac{\mathrm{d}}{\mathrm{d}t}dh$$ $$=dh^{\top}(-\frac{1}{2}\frac{1}{\tau_{h}}\langle D_{y_{i}}^{\top}(D_{y_{i}}dh-(y_{2,i}^{*}-y_{1,i}^{*}))\rangle_{i=1,...,N})$$ $$=-\frac{1}{2}\frac{1}{\tau_{h}}\langle||dy_{i}||^{2}-dy_{i}^{\top}(y_{2,i}^{*}-y_{1,i}^{*})\rangle_{i=1,...,N}$$ $$=-\frac{1}{2}\frac{1}{\tau_{h}}\langle w_{i}\rangle_{i=1,...,N}$$
d dt ||dyi||2 = 2dy⊤ i d dt dyi = 2dy⊤ i(D˙yi dh + dh⊤ d dt dh ||dh||2 Dyi dh) , (19) = 2dy⊤ i(D˙yi dh + 1 2 d dt ||dh||2 ||dh||2  Dyi dh) = 2dy⊤ i(− 1 τyi 1 N 1 4 (dyi − (y ∗ 2,i − y ∗ 1,i))||dh||2 − 1 4 1 τh ⟨wi⟩i=1,...,N ||dh||2dyi) = − 1 τyi 1 N 1 2 (||dyi||2 − dy⊤ i(y ∗ 2,i − y ∗ 1,i))||dh||2 − 1 2 1 τh ⟨wi⟩i=1,...,N ||dyi||2 ||dh||2
= − 1 τyi 1 N 1 4 (3wi − ||dyi||2 + ||y ∗ 2,i − y ∗ 1,i||2)||dh||2 − 1 4 1 τh ⟨wi⟩i=1,...,N ||dh||2(||dyi||2 + wi) d dt wi = (2dyi − (y ∗ 2,i − y ∗ 1,i))⊤ d dt dyi = (2dyi − (y ∗ 2,i − y ∗ 1,i))⊤(− 1 τyi 1 N 1 4 (dyi − (y ∗ 2,i − y ∗ 1,i))||dh||2 − 1 4 1 τh ⟨wi⟩i=1,...,N ||dh||2dyi)
where 1
τh
=1
τh1
$${\frac{1}{\tau_{1}}}+{\frac{1}{\tau_{h_{2}}}}.$$

In the case that the output effective learning rates are all equal, i.e. ∀iτyi = τy, this system can be reduced to a 3-dimensional scalar system:

d dt ||dh||2 = − 1 2 1 τh ⟨wi⟩i d dt ⟨||dyi||2⟩i = − 1 2 (1 Nτy ||dh||2 + 1 τh ⟨||dyi||2⟩i ||dh||2)⟨wi⟩i d dt ⟨wi⟩i = − 1 4 1 Nτy (3⟨wi⟩i − ⟨||dyi||2⟩i + ⟨||y ∗ 2,i − y ∗ 1,i||2⟩i)||dh||2 − 1 4 1 τh ⟨wi⟩i ||dh||2 (⟨||dyi||2⟩i + ⟨wi⟩i).
$$(20)$$

## B.2. Final Representational Structure

In order to study the final representational structure learned by the network, we solve the final representational distance for the pair. Using the relationship

$$\frac{\mathrm{d}}{\mathrm{d}t}\frac{\langle||d\mu_{i}||^{2}\rangle_{i}}{||d\mu_{i}||^{2}}=\frac{||d\mu_{i}||^{2}\frac{\mathrm{d}}{\mathrm{d}t}\langle||d\mu_{i}||^{2}\rangle_{i}-\langle||d\mu_{i}||^{2}\rangle_{i}\frac{\mathrm{d}}{\mathrm{d}t}||d\mu_{i}||^{2}}{||d\mu_{i}||^{4}}=-\frac{1}{2N\tau_{y}}\langle w_{i}\rangle_{i}=\frac{\tau_{b}}{N\tau_{y}}\frac{\mathrm{d}}{\mathrm{d}t}||d\mu_{i}||^{2},\tag{21}$$

we can solve ⟨||dyi||2⟩i(t) as a function of ||dh||2(t):

$$\langle||d y_{i}||^{2}\rangle_{i}(t)=\frac{\tau_{h}}{N\tau_{y}}||d h||^{4}(t)+\left(\frac{\langle||d y_{i}(0)||^{2}\rangle_{i}}{||d h(0)||^{2}}-\frac{\tau_{h}}{N\tau_{y}}||d h(0)||^{2}\right)||d h||^{2}(t),$$  in turn 2-dimensional case.  
reducing the dynamics to a 2-dimensional system:

$$\frac{\mathrm{d}}{\mathrm{d}t}||dh||^{2}=-\frac{1}{2}\frac{1}{\tau_{h}}\langle w_{i}\rangle_{i}$$ $$\frac{\mathrm{d}}{\mathrm{d}t}\langle w_{i}\rangle_{i}=-\frac{1}{4}(-\frac{\tau_{h}}{N\tau_{y}\tau}||dh_{1}|^{6}+\frac{1}{N\tau_{y}}||y_{2}-y_{1}||^{2}||dh||^{2}+\frac{4}{N\tau_{y}}||dh||^{2}\langle w_{i}\rangle_{i}+\frac{1}{\tau_{h}}\frac{\langle w_{i}\rangle_{i}^{2}}{||dh||^{2}}$$ $$+\left(\frac{||dy(0)||^{2}}{||dh(0)||^{2}}-\frac{\tau_{h}}{N\tau_{y}}||dh(0)||^{2}\right)(\frac{1}{\tau_{h}}\langle w_{i}\rangle_{i}-\frac{1}{N\tau_{y}}||dh||^{4})).$$
$$(22)$$
$$(24)$$

This system has three fixed points

$$||dh||^{2}=\frac{1}{2}A_{\text{high}}-\sqrt{\frac{1}{4}A_{\text{high}}^{2}+A_{\text{low}}^{2}},\langle w_{i}\rangle_{i}=0$$ $$||dh||^{2}=\frac{1}{2}A_{\text{high}}+\sqrt{\frac{1}{4}A_{\text{high}}^{2}+A_{\text{low}}^{2}},\langle w_{i}\rangle_{i}=0^{{}^{\circ}}$$ $$||dh||^{2}=0,\langle w_{i}\rangle_{i}=0$$
$$(26)$$
$$(27)$$
$$(28)$$

where

 ${A_\text{high}=||dh(0)||^2-\frac{N\tau_y}{\tau_h}\frac{\langle||dy_i(0)||^2\rangle_i}{||dh(0)||^2}}$  ${A_\text{low}=\sqrt{\frac{N\tau_y}{\tau_h}}\cdot\sqrt{\langle||y_{2,i}^*-y_{i,i}^*||^2\rangle_i}.}$  A tight solution and is the most likely solution. 
$$(25)$$
The first fixed point has negative representational distance and is thus not a valid solution. The second fixed point has Jacobian with negative trace

$$\begin{array}{c c c}{{}}&{{}}&{{}}\\ {{}}&{{}}&{{0}}&{{}}\\ {{}}&{{}}&{{1}}\,\frac{\gamma_{\mathrm{m}}}{4\,\pi^{2}\tau_{\mathrm{w}}^{2}}(2A_{\mathrm{low}}^{2}+\frac{1}{2}(A_{\mathrm{high}}+\sqrt{A_{\mathrm{high}}^{2}+4A_{\mathrm{low}}^{2}})^{2})}&{{-\frac{1}{\tau_{\mathrm{w}}}(\frac{1}{2}A_{\mathrm{high}}+\sqrt{A_{\mathrm{high}}^{2}+4A_{\mathrm{low}}^{2}})}}\end{array}\Bigg]\,,$$ reduce $$\mathrm{Tr}(J)=-\frac{1}{N\tau_{\mathrm{w}}}(\frac{1}{2}A_{\mathrm{high}}+\sqrt{A_{\mathrm{high}}^{2}+4A_{\mathrm{low}}^{2}}),$$ determinant
, (26)
and positive determinant

$$\operatorname*{det}(J)=\frac{1}{4}\frac{1}{N^{2}\tau_{y}^{2}}(2A_{\mathrm{low}}^{2}+\frac{1}{2}(A_{\mathrm{high}}+\sqrt{A_{\mathrm{high}}^{2}+4A_{\mathrm{low}}^{2}})^{2}),$$

and is therefore always stable. The final fixed point has Jacobian

$$J=\left[\begin{array}{c c c}{{0}}&{{-\frac{1}{\tau_{h}}}}\\ {{\frac{1}{2}(\frac{1}{\tau_{h}}\frac{\left(w_{i}\right)^{2}}{||d h||^{2}}-\frac{1}{N\tau_{y}}(||y_{2,i}^{*}-y_{1,i}^{*}||^{2})_{i})}}&{{(\frac{1}{2}\frac{1}{\tau_{y}}A_{\mathrm{high}}-\frac{1}{\tau_{h}}\frac{\left(w_{i}\right)_{i}}{||d h||^{2}})}}\end{array}\right],$$

which cannot be directly evaluated at ⟨wi⟩i = 0, ||dh||2 = 0 because of the undetermined term ⟨wi⟩i ||dh||2 . By replacing 
⟨wi⟩i ||dh||2 with the direction of approach ba we can solve for eigenvectors

$$\left[\begin{array}{c c}{{0}}&{{-\frac{1}{\tau_{n}}}}\\ {{\frac{1}{2}(\frac{1}{\tau_{n}}\frac{b^{2}}{a^{2}}-\frac{1}{N\tau_{y}}\langle||y_{2,i}^{*}-y_{1,i}^{*}||^{2}\rangle_{i})}}&{{(\frac{1}{2}\frac{1}{N\tau_{y}}A_{\mathrm{high}}-\frac{1}{\tau_{n}}\frac{b}{a})}}\end{array}\right]\left[\begin{array}{c}{{a}}\\ {{b}}\end{array}\right]=\lambda\left[\begin{array}{c}{{a}}\\ {{b}}\end{array}\right]$$
$$(29)$$
$$(30)$$

17 to find

$$v_{\pm}=\left[\begin{array}{c}{{1}}\\ {{-\frac{\tau_{h}}{N\tau_{y}}\frac{A_{\mathrm{high}}\pm\sqrt{A_{\mathrm{high}}^{2}+4A_{\mathrm{low}}^{2}}}{2}}}\end{array}\right]\tag{1}$$
$$(32)$$
$$(31)$$

with one positive and one negative eigenvalue

$$\lambda_{\pm}=\frac{1}{N\tau_{y}}\frac{A_{\mathrm{high}}\pm\sqrt{A_{\mathrm{high}}^{2}+4A_{\mathrm{low}}^{2}}}{2}.$$

There is always one direction along which perturbations increase, so this fixed point is not stable. Only the second fixed point is valid and stable, hence we expect the final representational distance to reach it.

## B.3. Agreeing Pair Dynamics

When the pair agrees on all possible future outputs, i.e. ∀i y
∗
2,i = y
∗1,i we have that ⟨wi⟩i = ⟨||dyi||2⟩i, allowing us to reduce the system Equation (20) to

$$\frac{\mathrm{d}}{\mathrm{d}t}||dh||^{2}=-\frac{1}{2}\frac{1}{\tau_{h}}\langle||dy_{i}||^{2}\rangle_{i}\tag{33}$$ $$\frac{\mathrm{d}}{\mathrm{d}t}\langle||dy_{i}||^{2}\rangle_{i}=-\frac{1}{2}\langle||dy_{i}||^{2}\rangle_{i}(\frac{1}{N\tau_{y}}||dh||^{2}+\frac{1}{\tau_{h}}\frac{\langle||dy_{i}||^{2}\rangle_{i}}{||dh||^{2}}).$$

Using Equation (22) we can write a self-contained equation for ||dh||(t):

$$\frac{\mathrm{d}}{\mathrm{d}t}||dh||^{2}=-\frac{1}{2}\frac{1}{N\tau_{y}}||dh||^{4}+\frac{1}{2}\frac{1}{N\tau_{y}}A_{\text{high}}||dh||^{2},\tag{34}$$

which is Bernoulli and has solution

$$||d h(t)||^{2}={\frac{A_{\mathrm{high}}}{1+{\big(}{\frac{A_{\mathrm{high}}}{||d h(0)||^{2}}}-1{\big)}e^{-{\frac{1}{2}}{\frac{1}{N+{\frac{1}{\nu}}}}A_{\mathrm{high}}t}}},$$

which, as Ahigh *≤ ||*dh(0)||2exponentially decays towards the final representational distance ||dh(∞)||2 = Ahigh.

B.4. Fixed Expansion Point Interaction Model We take the same approach before but instead keep the linear expansion point fixed during training:

$$(35)$$
$$y(x_{\alpha,i})=b_{i}+D_{y_{i}}h_{\alpha}.$$
y(xα,i) = bi + Dyihα. (36)
The mean squared loss in this approximation has the form:

$$L=\frac{1}{2}\langle||b_{i}+D_{y_{i}}h_{\alpha}-y_{\alpha,i}^{*}||^{2}\rangle_{\mathcal{D}}.\tag{14}$$

Motivated by the assumption of high model expressivity, we apply gradient decent optimization directly with respect to Dyi
,

$$(36)^{\frac{1}{2}}$$
$$(37)$$

hα and bi, resulting in the dynamics:

 -  $\dfrac{\mathrm{d}}{\mathrm{d}t}b_i=-\dfrac{1}{\tau_{b_i}}\dfrac{\partial L}{\partial b_i}$  $\qquad=-\dfrac{1}{\tau_{b_i}}\dfrac{1}{N}\langle b_i+D_{y_i}h_{\alpha}-y_{\alpha,i}^*\rangle_{\alpha=1,2}$  $\qquad=-\dfrac{1}{\tau_{\bar{y}_i}}\dfrac{1}{N}(b_i+D_{y_i}\langle h_{\alpha}\rangle_{\alpha}-\langle y_{\alpha,i}^*\rangle_{\alpha})$
$$\frac{d}{dt}h_{\alpha}=-\frac{1}{\tau_{h_{\alpha}}}\frac{\partial L}{\partial h_{\alpha}}$$ $$=-\frac{1}{\tau_{h_{\alpha}}}\frac{1}{2}\langle D_{y_{i}}^{\top}(b_{i}+D_{y_{i}}h_{\alpha}-y_{\alpha,i}^{*})\rangle_{i=1,...,N}$$
$$(38)$$
$$\begin{array}{l}{{\frac{\mathrm{d}}{\mathrm{d}t}D_{y_{i}}=-\frac{1}{\tau_{y_{i}}}\frac{\partial L}{\partial D_{y_{i}}}}}\\ {{=-\frac{1}{\tau_{y_{i}}}\frac{1}{N}\langle D_{y_{i}}h_{\alpha}h_{\alpha}^{\top}+(b_{i}-y_{\alpha,i}^{*})h_{\alpha}^{\top}\rangle_{\alpha=1,2}}}\\ {{=-\frac{1}{\tau_{y_{i}}}\frac{1}{N}(D_{y_{i}}\langle h_{\alpha}h_{\alpha}^{\top}\rangle_{\alpha=1,2}+b_{i}\langle h_{\alpha}^{\top}\rangle_{\alpha=1,2}-\langle y_{\alpha,i}^{*}h_{\alpha}^{\top}\rangle_{\alpha=1,2}).}}\end{array}$$
We again try the Ansatz where the representations only move towards or away from each other, which, since we take the expansion point to be the representational mean at t = 0, can be written by shifting coordinates without loss of generality as

$$\mathbf{h}_{\alpha}\propto h_{\alpha}\mathbf{v},$$
hα ∝ hαv, (39)
for some vector v with ||v|| = 1. We define di:= v
⊤Dyiv, bi:= v
⊤D⊤
yi bi, allowing us write using the derivatives

$$\frac{d}{dt}h_{\alpha}=-\frac{1}{\tau_{h_{\alpha}}}\frac{1}{2}\langle(D_{y_{i}}v)^{\top}(b_{i}+h_{\alpha}D_{y_{i}}v-y^{*}_{\alpha,i})\rangle_{i}$$ $$\frac{\mathrm{d}}{\mathrm{d}t}b_{i}=-\frac{1}{\tau_{b_{i}}}\frac{1}{N}(b_{i}+\langle h_{\alpha}\rangle_{\alpha}D_{y_{i}}v-\langle y^{*}_{\alpha,i}\rangle_{\alpha})$$ $$\frac{\mathrm{d}}{\mathrm{d}t}D_{y_{i}}v=-\frac{1}{\tau_{y_{i}}}\frac{1}{N}\langle\langle h^{2}_{\alpha}\rangle_{\alpha}D_{y_{i}}v+\langle h_{\alpha}\rangle_{\alpha}b_{i}-\langle h_{\alpha}y^{*}_{\alpha,i}\rangle_{\alpha}\rangle,$$
$$(40)$$

a scalar system which takes the form

d
dthα = −
1
τhα
1
2
⟨b
⊤
i Dyiv + hα||Dyiv||2 − y
⊤
α,iDyiv⟩i
d
dt
b
⊤
i Dyiv = −
1
τbi
1
N
(b
⊤
i Dyiv + ⟨hα⟩α||Dyiv||2 − ⟨y
⊤
α,iDyiv⟩α) −
1
τyi
1
N
(⟨h
2
α⟩αb
⊤
i Dyiv + ⟨hα⟩α||bi||2 − ⟨hαb
⊤
iy
∗
α,i⟩α)
d
dt
b
⊤
i
y
∗
β,i = −
1
τbi
1
N
(b
⊤
i
y
∗
β,i + ⟨hα⟩αy
⊤
β,iDyi
v − y
⊤
β,i⟨y
∗
α,i⟩α)
.
d
dt
||bi||2 = −2
1
τbi
1
N
(||bi||2 + ⟨hα⟩αb
⊤
i Dyi
v − ⟨b
⊤
i
y
∗
α,i⟩α)
d
dt
||Dyi
v||2 = −2
1
τyi
1
N
(⟨h
2
α⟩α||Dyi
v||2 + ⟨hα⟩αb
⊤
i Dyi
v − ⟨hαy
⊤
α,iDyi
v⟩α)
d
dt
y
∗⊤
β,iDyiv = −
1
τyi
1
N
(⟨h
2α⟩αy
∗⊤
β,iDyiv + ⟨hα⟩αy
∗⊤
β,ibi − ⟨hαy
∗⊤
β,iy
∗α,i⟩α)
(41)
If the output effective learning rates are again all equal, i.e. ∀ τbi = τb, ∀ τyi = τy, this system can be reduced to 9 scalars:

d
dthα = −
1
τhα
1
2
(⟨b
⊤
i Dyiv⟩i + hα⟨||Dyiv||2⟩i − ⟨y
⊤
α,iDyiv⟩i)
d
dt
⟨b
⊤
i Dyiv⟩i = −
1
τb
1
N
(⟨b
⊤
i Dyiv⟩i + ⟨hα⟩α⟨||Dyiv||2⟩i − ⟨y
⊤
α,iDyiv⟩α,i)
−
1
τy
1
N
(⟨h
2α⟩α⟨b
⊤
i Dyi
v⟩i + ⟨hα⟩α⟨||bi||2⟩i − ⟨hα⟨b
⊤
i
y
∗α,i⟩i⟩α)
d
dt
⟨b
⊤
iy
∗
β,i⟩i = −
1
τb
1
N
(⟨b
⊤
iy
∗
β,i⟩i + ⟨hα⟩α⟨y
⊤
β,iDyiv⟩i − ⟨y
⊤
β,iy
∗
α,i⟩α,i)
d
dt
⟨y
∗⊤
β,iDyiv⟩i = −
1
τy
1
N
(⟨h
2
α⟩α⟨y
∗⊤
β,iDyiv⟩i + ⟨hα⟩α⟨y
∗⊤
β,ibi⟩i − ⟨hα⟨y
∗⊤
β,iy
∗
α,i⟩i⟩α)
. (42)
d
dt
⟨||bi||2⟩i = −2
1
τb
1
N
(⟨||bi||2⟩i + ⟨hα⟩α⟨b
⊤
i Dyiv⟩i − ⟨b
⊤
iy
∗
α,i⟩α,i)
d
dt
⟨||Dyiv||2⟩i = −2
1
τy
1
N
(⟨h
2
α⟩α⟨||Dyiv||2⟩i + ⟨hα⟩α⟨b
⊤
i Dyiv⟩i − ⟨hα⟨y
⊤
α,iDyiv⟩i⟩α)
Training loss The loss can be written expressed using the variables from this system

L =
$$\frac{1}{2}(\langle||b_{i}||^{2}\rangle_{i}+\langle h^{2}_{\alpha}\rangle_{\alpha}\langle||D_{b_{i}}v||^{2}\rangle_{i}+\langle||y^{*}_{\alpha,i}||^{2}\rangle_{\alpha,i}+2\langle h_{\alpha}\rangle_{\alpha}\langle b^{\top}_{i}D_{b_{i}}v\rangle_{i}-2\langle h_{\alpha}\langle y^{*\top}_{\alpha,i}D_{b_{i}}v\rangle_{i}\rangle_{\alpha}-2\langle\langle y^{*\top}_{\alpha,b_{i}}b_{i}\rangle_{i}\rangle_{\alpha}).\tag{43}$$
Equal hidden effective learning rates When the pairs agree on all outputs,s we have

$$\langle y^{*\top}{}_{1,i}y_{\alpha,i}^{*}\rangle_{\alpha,i}=\langle y^{*\top}{}_{2,i}y_{\alpha,i}^{*}\rangle_{\alpha,i}$$
α,i⟩α,i (44)
Assuming no correlation at initialization due to random weights

$$\langle b_{i}^{\top}D_{y_{i}}v\rangle_{i}(0)=0$$ $$\langle b_{i}^{\top}y_{\beta,i}^{*}\rangle_{i}(0)=0,$$ $$\langle y^{*}{}_{\beta,i}^{\top}D_{y_{i}}v\rangle_{i}(0)=0$$
$$(444)$$
$$(45)$$
$$(46)$$

we find the relations such that

$$\begin{array}{l}{{\frac{\mathrm{d}}{\mathrm{d}t}\langle b_{i}^{\top}y_{1,i}^{*}\rangle_{i}=\frac{\mathrm{d}}{\mathrm{d}t}\langle b_{i}^{\top}y_{2,i}^{*}\rangle_{i}}}\\ {{\frac{\mathrm{d}}{\mathrm{d}t}\langle y_{1,i}^{*\top}D_{y,i}v\rangle_{i}=\frac{\mathrm{d}}{\mathrm{d}t}\langle y_{2,i}^{*\top}D_{y,i}v\rangle_{i}}}\end{array},$$
$$\langle b_{i}^{\top}y_{1,i}^{*}\rangle_{i}=\langle b_{i}^{\top}y_{2,i}^{*}\rangle_{i}$$ $$\langle y^{*}{}_{1,i}^{\top}D_{y_{i}}v\rangle_{i}=\langle y^{*}{}_{2,i}^{\top}D_{y_{i}}v\rangle_{i}\tag{1}$$

From this it follows that

$$\frac{d}{dt}(\tau_{2}h_{2}-\tau_{1}h_{1})=-\frac{1}{2}((h_{2}-h_{1})\langle||D_{y_{i}}v||^{2}\rangle_{i}).\tag{1}$$

In the case of equal effective hidden learning rates τh1 = τh2, the representational distance can only increase

$${\frac{d}{d t}}\vert\vert h_{2}-h_{1}\vert\vert^{2}=-\langle\vert\vert D_{y_{i}}v\vert\vert^{2}\rangle_{i},$$
, (49)
so no initial divergence occurs for agreeing pairs in this case.

$$(47)$$
$$(48)^{\frac{1}{2}}$$
$$(49)$$