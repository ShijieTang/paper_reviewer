# The Emergence Of Sparse Attention: Impact Of Data Distribution And Benefits Of Repetition

Nicolas Zucchet∗
ETH Zürich Francesco D'Angelo EPFL
Andrew Lampinen†
Google DeepMind Stephanie Chan†
Google DeepMind

## Abstract

Emergence is a fascinating property of large language models and neural networks more broadly: as models scale and train for longer, they sometimes develop new abilities in sudden ways. Despite initial studies, we still lack a comprehensive understanding of how and when these abilities emerge. To address this gap, we study the emergence over training of sparse attention, a critical and frequently observed attention pattern in Transformers. By combining theoretical analysis of a toy model with empirical observations on small Transformers trained on a linear regression variant, we uncover the mechanics driving sparse attention emergence and reveal that emergence timing follows power laws based on task structure, architecture, and optimizer choice. We additionally find that repetition can greatly speed up emergence. Finally, we confirm these results on a well-studied in-context associative recall task. Our findings provide a simple, theoretically grounded framework for understanding how data distributions and model design influence the learning dynamics behind one form of emergence.

Scaling has been central to the recent success of large language models [1–5], with scaling laws [6, 7] describing how increased model size, data, and training consistently improve average performance. However, beneath this macroscopic predictability, model performance on specific tasks often reveals capabilities that appear suddenly beyond critical scaling thresholds - a phenomenon known as emergence [8–11]. While recent studies have begun to characterize how emergence can appear after a critical training time [12–18], a comprehensive scientific understanding remains elusive. This work explores sparse attention as a lens to understand emergence during training. The formation of sparse attention - where Transformers' attention layers focus on a small subset of critical tokens - coincides with sudden performance improvements in many emergent behaviors, including in-context learning abilities [12, 15] and factual recall [18]. We investigate why the development of sparse attention can lead to abrupt performance improvements and reveal how characteristics of the training data influence the speed of emergence. We make two key contributions:
- We design a variant of linear regression that specifically requires Transformers to learn to focus on a few tokens within the context (Section 2). This analytically tractable task allows us to mathematically characterize, in a toy model, the mechanics behind the emergence of sparse attention and precisely quantify how shorter sequences and repetition accelerate emergence.

- We apply our sparse attention framework to explain the emergence of in-context learning in an associative recall task (Section 3). Our theoretical predictions successfully account for how data influences the emergence speed of the induction head that solves this task.

Overall, our results suggest that sparse attention may provide a unifying perspective for understanding seemingly diverse emergence phenomena in large language models. They additionally highlight the potential practical benefits of repetition for accelerating the formation of specific neural circuits.

∗Correspondance to nzucchet@ethz.ch.

†Advisory capacity only.

## 1 Motivation

Can emergence be predicted? Emergence in large language models not only challenges our scientific understanding of how they acquire new skills but also poses AI safety issues [19] due to its unpredictable nature, while additionally complicating frontier model development. To address this unpredictability, researchers have proposed several progress metrics under which scaling has more predictable consequences, including validation loss [20], metrics that reward partial progress [21, 22], and those that employ mechanistic interpretability-motivated measures [23]. However, a significant limitation is that these metrics typically can only be derived post-emergence [24]. An alternative approach demonstrates that fine-tuned models' performance on certain tasks can predict whether the ability to solve the task will emerge in larger models [17]. Our work explores predictability from a different angle: understanding how the data distribution influences the learning time at which emergence occurs. Is there a link between emergence and sparse attention? The driving hypothesis underlying this work is that learning of sparse attention patterns is particularly prone to produce sharp transitions in behavior during training - i.e., the sudden emergence of new capabilities. There is a statistical argument behind this intuition: when the target a Transformer has to predict depends only on a few tokens within the context, these tokens initially have very low weight in the prediction of the model, as attention typically starts uniform. Initial progress is therefore slow and the more targeted (thus sparse) attention is, the faster learning becomes. Multiple empirical observations strengthen our hypothesis. The induction head [12], whose formation coincides with the sudden emergence of certain in-context learning abilities, fundamentally relies on the combination of two attention layers with sparse patterns. Similarly, the mechanisms underlying factual recall in large language models [23, 25] demonstrate sparse attention properties and show emergent learning dynamics [18]. These specific emergent phenomena appear to be linked to the development of sparse attention mechanisms, suggesting the existence of a potential causal relationship between the two.

The intriguing interplay between repetition and emergence. While data diversity is generally considered a gold standard in machine learning, a growing body of evidence suggests that repetition can actually accelerate emergence over training. Chan et al. [13] demonstrated that showing some tasks more often favors the emergence of in-context learning. Charton and Kempe [16] revealed that repeating a subset of examples more frequently than others dramatically accelerates Transformers' ability to solve certain arithmetic tasks. This pattern extends beyond mathematical reasoning, as Zucchet et al. [18] (and to some extent Allen-Zhu and Li [26]) showed that repeating biographies of specific individuals speeds up the development of circuits critical for factual recall from model parameters. Collectively, these results establish repetition as a fundamental property of data distributions that can systematically influence emergence timing, thus justifying integrating it in our model to better understand its role.

## 2 Theoretical Insights On An Attention-Based Linear Regression Task

To illustrate how emergence can arise from sparse attention learning, we introduce a variant of linear regression that additionally requires selecting a relevant token from the input sequence. We introduce this task alongside a minimal attention-based toy model for solving it (Section 2.1), theoretically analyze its learning dynamics both without (Section 2.2) and with repetition (Section 2.3), and empirically show that our findings extend to more realistic Transformers (Section 2.4).

## 2.1 The Single-Location Linear Regression Task

We consider the following supervised learning task. We are given an input sequence (xt)
T
t=1 of length T in which each token xt ∈ R
dis drawn i.i.d. from a zero-mean normal distribution with variance 1/d and aim to predict the target y
∗ given by y
∗ = W∗xT (1)
Here, the target weight matrix W∗ ∈ R
d×dis a predetermined matrix and y ∈ R
d. To successfully solve this task, an attention-based model must learn to attend to the last token only and learn the ground-truth target weights W∗. We deliberately incorporate a sparse attention target mechanism to study the relationship between sparse attention and emergence. This task shares similarities with the

Task. Single-location linear regression a.

b.

10 3 10 2 10 1 100 101 d 64 32 16 0 5k 10k time 10 6 10 5 10 4 10 3 10 2 10 1 va lu e lo ss tokens in context In-context repetition w 0 5k 10k time the relevant token appears times c.

103 104 105 Cross-sample repetition pl at ea u le ng th 256 512 1024 2048 4096 data points scaling law 101 102 d the same appears with probability
Figure 1: **A simple task to study the emergence of sparse attention.** (left) We introduce a variant of linear regression task that is analytically tractable and in which Transformers-like models need to learn sparse attention. The model must identify which token (here the last one, xT ) is relevant for the target output y
∗. We incorporate two realistic forms of repetition in the data: in-context repetition, where the relevant token appears multiple times within the context, and cross-sample repetition, where an input sequence contains a special token x˜ (here colored in green) at the relevant position with probability p. See Section 2.1 for details. (right) a. As desired, the reduced learning dynamics of our simplified Transformer (Eq. 2) exhibit a multi-phase behavior including an initial plateau, on the task without repetition (T = 512). b. Mechanistically, the weights w begin learning before attention to the relevant token α (T = 512, d = 64). Dashed lines represent optimal values. c. The duration of the initial plateau increases as a function of sequence length T and input/output dimension d, closely following a power law scaling relationship (R2 = 0.999) that can be accurately predicted by linearizing the dynamics around initialization (Equation 6). See Section 2.2 for details. single-location regression task of Marion et al. [27], although in our case, the relevant token location is always the same. The left panel of Figure 1 summarizes this task. We model two forms of repetition that are ubiquitous in natural language:
- **In-context repetition.** This occurs when specific token groups (e.g., a person's name) repeatedly appear within a single context window. In our framework, we model this property by repeating the relevant token xT multiple times in the sequence (xt)
T
t=1, always at the same positions for our simplified model to be able to use it. Following Chan et al. [13], who termed this property
"burstiness", we denote B as the number of times the task-relevant token appears in the context.

- **Cross-sample repetition.** This form of repetition comes from having certain information (such as biographical details of specific individuals) overrepresented in the overall training data. We implement this by first sampling the input sequence normally, but then occasionally (with probability p) replacing the relevant token xT with a special predefined token x˜.

For the purpose of our theoretical analysis, we introduce a simplified attention layer, defined by

$$y=W\sum_{t=1}^{T}\mathrm{softmax}(a)_{t}\,x_{t}$$
softmax(a)t xt (2)
with a ∈ R
Tthe attention scores vector and W ∈ R
d×dthe weight matrix. Unlike in standard Transformer attention [28], our model does not use any semantic information to determine where to attend. This simplification facilitates theoretical analysis and implicitly assumes that the attention

$$(2)$$

layer has already learned to filter irrelevant contextual information. The model's parameters (*a, W*) are learned by minimizing the expected mean square error between predictions y and targets y
∗.

## 2.2 The Learning Dynamics Of The Simplified Transformer Exhibit Emerging Abilities

The performance of our simplified model on this task exhibits a characteristic learning pattern: an initial plateau where the loss minimally decreases, followed by a sharp phase transition towards significantly lower loss values. The analysis we detail below reveals that this emergent behavior arises from the interaction between feedforward weights and attention during learning, and that the duration of the initial plateau increases with the sequence length T and data dimensionality d. Reduced learning dynamics. We analyze the gradient flow dynamics of the simplified model. The assumptions and mathematical details of this analysis can be found in Appendix A. Under these mild assumptions, the learning dynamics of the entire model reduces to two key scalar variables:
∆a := aT − at for any *t < T* (all attention scores to non-relevant tokens stay the same under our assumptions) and w the scalar projection1 of W on W∗. These variables are initially both equal to 0 and they evolve according to the following system of ordinary differential equations:

$$\dot{w}=\frac{\alpha(\sqrt{d}-\alpha w)}{d}-\frac{(1-\alpha)^{2}w}{d(T-1)}\tag{3}$$  $$\dot{\Delta}a=\alpha(1-\alpha)\left(\frac{w(\sqrt{d}-\alpha w)}{d}+\frac{(1-\alpha)w^{2}}{d(T-1)}\right),\tag{4}$$

with α := (1 + (T − 1) exp(−∆a))−1the attention given to the final token. In these two equations, the first term is the signal coming from the relevant token at position T, and the second term is the noise coming from the T − 1 non-relevant tokens. This decomposition follows from the derivation in Appendix A.3, which groups token contributions by their relevance to the target prediction.

These equations enable us to elucidate the roots of emergence in this task. Initial learning is slow, as ∆a does not receive any teaching signal as w = 0 and w slowly increases as attention is initially uniform (α = 1/T). As a consequence, the feedforward weight W must first align with the target weights W∗ before attention can learn. A positive feedback loop is then progressively established:
increased attention improves the learning signal for w, and a better-learned w further reinforces the correct attention pattern. This dynamic, similar to the one found in deep linear networks [29], leads to a sharp decrease in loss and the sudden emergence we observe in Figure 1.a.

Predicting when emergence arise. To estimate how long it takes to escape the initial loss plateau,
we linearize the dynamics around the initial conditions and obtain
$$\left(\begin{array}{c}{{\frac{1}{w}}}\\ {{\Delta a}}\end{array}\right)=\left(\begin{array}{c}{{\frac{1}{\sqrt{d T}}}}\\ {{0}}\end{array}\right)+\left(\begin{array}{c c}{{0}}&{{\frac{1}{\sqrt{d T}}}}\\ {{\frac{1}{\sqrt{d T}}}}&{{0}}\end{array}\right)\left(\begin{array}{c}{{w}}\\ {{\Delta a}}\end{array}\right).$$
$$(S)$$
$$(6)$$
This linearization provides two key insights: First, it confirms that feedforward weight learning precedes and drives attention learning, as evidenced by the initial gradient and the top-right entry of the matrix in the equation above, and corroborated by the simulations of Figure 1.b. Second, it enables us to estimate the escape time from initial conditions, defined as the time required to reach
(1 − ε) of the initial loss value. It is equal to

$$T_{\varepsilon}=\frac{\sqrt{dT}}{2}\,\ln\left(\varepsilon\sqrt{dT}\right).$$  In that both known and higher dimensional 
This formula succinctly demonstrates that both longer sequences and higher-dimensional inputs increase the time spent on the plateau and delay emergence. This theoretically derived scaling closely matches the one obtained in simulations, as depicted in Figure 1.c (ε = 0.8 in these simulations).

## 2.3 Repetition Speeds Up Emergence

Now that we have thoroughly examined the vanilla case, we focus on understanding the effects of repetition. Our analysis reveals that in-context repetition makes the attention pattern to be learned less 1w is formally defined as w := ⟨W∗, W⟩F /∥W∗∥F ∈ R with ⟨ · , · ⟩F the Froebenius inner product.

In-context repetition Cross-sample repetition 103 104 105 103 104 105 B
1 2 test los s p 0 0.2 0 5k 10k time 10 5 10 3 10 1 0 5k 10k time 10 5 10 3 10 1 pl ate a u l e n gt h pl ate a u l e n gt h test los s p 0.0 0.1 0.2 0.4 B
1 2 4 8 16 101 102 101 102 d d
sparse, thereby simplifying the task, while cross-sample repetition accelerates feedforward weight learning in specific directions, which subsequently increases overall learning speed by increasing the attention of the model to relevant tokens earlier. We present the key findings below. In-context repetition. The role of in-context repetition is rather simple. When relevant information repeats multiple times within the context, it becomes more correlated with the token to predict. This can be understood as increasing the signal to noise ratio or as effectively reducing the sequence length from T to *T /B*. Since learning time scales with sequence length, this makes the task fundamentally easier. Theoretically, we can show, cf. Appendix A.3, that this intuition precisely holds and that the escape time from the initial loss plateau becomes proportional to T
√d/B. Replacing T by *T /B*
in the scaling law we obtained in Figure 1.c yields an almost perfect empirical fit, cf. Figure 2.b. This result highlights that B reduces the sparsity of the target attention pattern and thus accelerates emergence. This analysis ignores, by design, any ability of the attention mechanism to flexibly use semantic or positional information, as we directly parameterized the attention scores. We argue that our findings will extend to the more general case. Indeed, both the attention scores, which would now be the output of some function, and the feedforward weights receive larger gradients with in-context repetition, and thus learning will overall be faster. We will confirm that the same conclusion holds empirically for more realistic architectures and optimizers in the next sections. Cross-sample repetition. The role of cross-sample repetition is more intricate. Repeating one token more frequently, that is increasing p, causes the input covariance matrix of the relevant token, E[xT x
⊤
T], to become anisotropic. The different components of the weight matrix W are then learned at different speeds. While this difference in learning speed traditionally leads to slower convergence rates in vanilla linear regression [30], it turns out to be beneficial in our attention-based version of the task. Indeed, following similar principles to the ones detailed in the previous section, learning weights in the repeated direction will lead to the attention to the relevant tokens increasing faster, and this then speeds up the learning of the non-repeated dimensions. As a consequence, the model escapes the initial learning plateau faster. Importantly, this also holds when measuring the model's performance on non-repeated data, as seen in Figure 2. The effect of cross-sample repetition can also be understood as increasing the signal-to-noise ratio. Repeating the same token increases the amount of signal coming from the relevant token while keeping the amount of noise received from other tokens fixed, at the price of losing some information about the relevant signal. Theoretical analysis requires the introduction of an additional variable, so that the learning speed in both the repeated dimension and the other dimensions can be tracked independently. This model, which we introduce in Appendix A.4, enables us to justify the mechanistic insights mentioned above, as well as to derive that the plateau length scales as 
√*dT /*pp 2d + (1 − p)
2, which accurately describes empirical behavior (cf. Figure 2). The repetition probability p thus implicitly interpolates

10 4 10 4 10 4 pl a te a u le n g t h pl a te a u le n g t h pl a te a u le n g t h 10 3 10 3 10 3 T
32 64 128 256 16 32 64 128 d p 0.0 0.1 0.2 0.4 B
1 2 4 8 16 32 64 128 256 d 16 32 64 128 d
between the d-dimensional case and the 1-dimensional case, highlighting that the learning of the feedforward mapping becomes less of a bottleneck. However, cross-sample repetition is not a free lunch: it only provides a temporary advantage. First, these dynamics only have a smaller test loss in the medium term. In the long run, learning is slower for similar reasons as for the standard linear regression. Second, there is an additional overfitting problem. While it does not appear in this simplified setting as anisotropic input covariance does not bias the final solution, it starts appearing for more realistic architectures. This phenomenon has been observed in practice: examples include [31, 16, 18] for synthetic tasks that have properties similar to the one we focus on here, and [32, 33] for the pretraining of large language models.

## 2.4 The Theory Qualitatively Captures Learning Dynamics Of Full-Fledged Transformers

We conclude our linear regression analysis by examining how our theoretical predictions extend to more realistic task versions, optimizers, and models - particularly those with standard attention that varies with inputs. Our findings, detailed in the remainder of this section, indicate that learning dynamics behave qualitatively similarly, showing sharp phase transitions, with emergence time maintaining similar dependencies on data properties. However, the precise dependencies differ, which we investigate in more depth. For this investigation, we train a standard 2-layer 4-heads Transformer with the Adam optimizer [34],
using a constant learning rate of 10−4. Our only deviation from standard architectures is removing layer normalization, which makes solving the task more challenging. To enhance task realism, we randomly sample the positions of relevant tokens and incorporate an additional feature into the input to indicate whether a token is task-relevant. Further experimental details are provided in the appendix. The learning dynamics of Transformers align qualitatively with our theoretical predictions. First, the loss evolution exhibits a sharp phase transition (see Figure 11 in the appendix for an example). Second, emergence time depends similarly on the data properties identified in our theory, with repetition accelerating emergence. However, this result comes with several nuances. For scenarios with no repetition or with in-context repetition, power laws still accurately capture the dependency of emergence time on sequence length T, dimension d and burstiness B, though with significantly different exponents. Ablations (see Appendix B.3) reveal that optimizer, architecture, and task specifics all influence these exponents. Notably, replacing Adam with SGD substantially slows emergence, both in absolute terms and by increasing dependency on task difficulty. This finding illustrates a broader observation: Adam is crucial for efficient Transformer learning [e.g., 35]. Regarding cross-sample repetition, power laws no longer accurately describe the empirical relationship, partly because measuring plateau length becomes more difficult (see the learning dynamics

32 64 128 256 vocab size 10 2 10 3 10 4 10 5 10 6 0 50k 100k 150k 200k steps 0.0 0.2 0.4 0.6 0.8 1.0 number pairs 8 16 32 pl at ea u l en gt h te st acc ura cy number pairs 8 16 32 0 50k 100k 150k 200k steps 0 2 4 te st l os s number pairs 8 16 32 64
under cross-sample repetition in Figure 12 in the appendix for an example). Nevertheless, the trends identified in our theory still hold: this form of repetition accelerates emergence, with the effect becoming more pronounced as d increases.

## 3 Emergence Of In-Context Learning, Through The Lens Of Sparse Attention

We conclude by investigating to what extent the insights developed on our simple linear regression task extend to more realistic learning problems, in particular one in which in-context learning emerges.

To this end, we examine an in-context associative recall task, which can be solved by learning an induction head - a well-studied circuit strongly implicated in in-context learning [12, 15], which necessitates at least two attention layers with sparse attention patterns. Our theory qualitatively captures both the learning dynamics and the impact of the training distribution on learning speed.

## 3.1 The In-Context Associative Recall Task

The in-context associative recall task serves as a standard benchmark for testing language models' ability to access information in their context and to perform a simple form of one-shot learning. This task requires abilities that correlate with models' capacity for in-context learning [12], and variations of it have been extensively studied to better understand how in-context learning emerges [13, 15, 36– 40]. It has also been used as a testbed for recently proposed linear recurrent architectures [e.g. 41–43], as it acts as an important differentiator between attention-based and recurrent architectures [44].

We implement this task as follows: each sequence consists of Npairs key-value token pairs (5 such pairs in the example below) followed by a query token (Z below), as shown:
Y I A X U R Z Y C A Z ?

The query corresponds to one of the keys in the context, and the model must output the corresponding value - Y in the example above (since the query Z matches the key in the pair Z Y). In practice, we work with a total of Ntokens unique tokens provided to the network via one-hot encodings. We train the model using a cross-entropy loss comparing the model output to the target value.

The number of pairs Npairs controls the sequence length T as T = 2Npairs + 1 (accounting for the query token), and the vocabulary size Ntokens plays a role comparable to the input dimension d in the linear regression task. To model in-context repetition, we ensure that the query token appears on average B times in the context (as a key). To model cross-sample repetition, we select a subset of 2 tokens that will appear more often2and vary p, the probability to sample the query from this subset.

22 is an arbitrary choice that we have not ablated.

The precise description of the task is provided in the appendix. The results we report in the main text are testing the learned models on data without any repetition, thereby testing the generalization abilities of models learned on repeated data.

Transformers typically solve this type of task by implementing an induction head - a circuit that combines an attention layer responsible for concatenating the representation of the current token with the previous token, and a selection attention layer responsible for retrieving the relevant information from the context. Both attention layers focus on a sparse subset of tokens (usually just one), making this task particularly well-suited to test our theory. Unlike the linear regression task where sparse attention was hard-coded in the target input-output mapping, there is no inherent constraint forcing models to develop sparse attention here. We note that our toy model (Equation 2) cannot directly implement an induction head, as it lacks the relative positional information for copying and semantic similarity mechanisms for token matching. This task thus serves as the perfect testbed to show that our theoretical insights extend to more realistic, yet still finely controlled, learning scenarios in which their simplifying assumptions do not hold.

## 3.2 The Emergence Of In-Context Learning Depends On Data As In The Theory

As with the linear regression task, we investigate the learning dynamics of Transformers on this task and compare the qualitative findings with those of our developed theory. The experimental setup being closely related to the one in Section 2.4, we defer its thorough description to the appendix and note that we use layer normalization and MLPs for this set of experiments. Overall, we find that our theory accurately describes both the learning dynamics (in-context learning emerges in this task) and the dependency of emergence timing on data distribution properties.

Longer sequences and larger vocabulary size delay emergence. We begin by verifying that the intuition described above applies and that the learning curve exhibits the sharp phase transitions characteristic of emergence. This is indeed the case for sufficiently long sequences and/or large vocabularies, as illustrated in Figure 4. Importantly, there is only one phase transition despite two attention layers being needed to learn sparse patterns; Figure 17 shows that they are learned simultaneously. This observation is consistent with the results of Reddy [36] and Singh et al. [15]. In Figure 4 (right), we report how the emergence time, arbitrarily defined as the time needed to reach 5% accuracy, evolves as a function of sequence length and data dimension. It accurately follows a power law, as in all our other experiments, albeit with exponents that are relatively low for Ntokens (0.79) and extremely high for Npairs (2.25). We hypothesize that the lower exponent for Ntokens stems from the fact that most of the feedforward processing can be handled by residual connections, and given that the dimension of the data primarily influences the speed of feedforward learning, data dimension does not have such a large effect. For the higher exponent for Npairs, our hypothesis is that this occurs as two sparse attention patterns must be learned jointly. In the toy model we analyze theoretically, a similar coupling exists between the attention and the feedforward weights, resulting in a multiplicative interaction. We posit that a similar mechanism may be at work here, with each attention layer contributing additively to the Npairs exponent. We leave a more thorough investigation of how different circuits interact to influence emergence time to future work. Repetition speeds up emergence but comes with overfitting risks. We next investigate the benefits of in-context and cross-sample repetition. From the high sensitivity of the emergence time on sequence length revealed by previous analysis, we expect in-context repetition to have a stronger effect than the cross-sample one. This is what we observe. The results reported in Figure 5, which evaluate the model performance on test data that has no repetition, demonstrate that repetition can significantly speed up emergence, by a factor of 4 for in-context repetition and a factor of 2 for cross-sample repetition. The attention sparsity perspective thus explains why in-context repetition has been found to favor in-context learning vs. in-weight learning [13, 36, 45]: the induction head needed to solve this kind of task is formed earlier with such a form of repetition.

However, this benefit comes with an overfitting risk: while repetition consistently accelerates learning (repetition always speeds up learning on the train loss, cf. Figure 16 in the appendix), too much repetition leads to learning strategies that do not generalize well. For example, selecting the most frequent value is a valid strategy for this task whenever the query appears two or more times in the context. Interestingly, we also observe some grokking-like patterns [46] as the test accuracy eventually starts to increase late in training for large amounts of repetition. We argue that this occurs

In-context repetition Cross-sample repetition B 1 2 4 8 0 50k 100k 150k steps 0.00 0.25 0.50 0.75 1.00 0 50k 100k 150k steps 0.00 0.25 0.50 0.75 1.00 p 0.0 0.2 0.4 0.8 0 50k 100k 150k steps 0 2 4 6 0 50k 100k 150k steps 0 2 4 6 test ac cu ra cy test ac cu ra cy te st lo ss te st lo ss
because the no-repetition case is still represented in the training data, albeit with very little weight. This trade-off between learning speed and generalization is consistent with what Park et al. [31] reported on another in-context learning task. The overfitting we observe here may appear inconsistent with our theory but is not: our theory addresses learning speed (performance on the training loss) rather than generalization ability.

## 4 Discussion

Connection with other theoretical work. Through the lens of sparse attention, we provide a unifying perspective on a set of empirical results highlighted in the motivation (Section 1). While our focus on sparse attention and the effect of repetition is unique (to the best of our knowledge),
there are multiple closely related works. First, the interaction between the feedforward weights and attention we study is closely related to the one between two consecutive feedforward linear layers [29, 47, 48]. Linear networks also display incremental learning with abrupt transitions characterizing each phase change [49–52]. This line of research on linear networks has since then been extended to linear [45, 53] and softmax [54] Transformers, in particular to study the emergence of in-context learning. On the more conceptual side, there exist other theories of how emergence could arise from longer training, such as grokking [46, 14] or singular learning theory [55, 56]. While rarer, other theories focus on emergence from increased model size [57]. Compared to these, our theory focuses on learning dynamics (emergence over the course of training) and is directly tied to the internal mechanisms of the Transformers. How much emergence comes from sparse attention? We find that the learning of sparse attention is prone to producing emergence over the course of training. Given that sparse or concentrated attention is a common emergent feature of Transformers (e.g., induction heads [12], task/function vectors [58, 59], or high-norm tokens in vision transformers [60]), one may ask how much currently known emergent behaviors can be attributed to the learning of sparse attention patterns, and whether there are many more emergent behaviors than what is currently reported. However, these points must be weighed. As noted above, even simpler MLPs can give rise to abrupt emergence over training [e.g. 29]; thus, learning in other circuits within Transformers might contribute to sudden transitions in their performance. Furthermore, emergence results at large scale mostly show a sharp phase transition of the model as the number of FLOPs [e.g., 8], which roughly corresponds to the model size times the number of training steps, reaches a certain threshold. It is therefore unclear whether these examples of emergence are rooted in longer training, models with higher capacity, or a complex interplay between the two. More empirical and theoretical work is needed to better understand the causes of emergence in large models and the possible complex relationship between training-based and size-based emergence. When does repetition help? Overall, we expect repetition to be beneficial when training on tasks prone to performance plateaus, as there is a clear benefit in trading off learning speed for generalization in these cases. Cross-sample repetition should be particularly helpful in tasks where learning feedforward transformations is challenging (high d in our analysis). This includes learning many factual associations as in Zucchet et al. [18] and the reasoning-heavy arithmetic tasks of Charton and Kempe [16], which require learning complex non-linear transformations. In-context repetition should be most beneficial when learning from very long sequences, as it effectively reduces the sequence length the model must process. Repetition, both in-context and cross-sample, is a natural feature of language that may accelerate some parts of language learning. The principles we touch upon could already be more directly at play in the current training pipeline of large language models, for instance when code, which typically has lower syntactic diversity than other types of text [61], is included at specific moments during training. Data diversity as a path towards active learning? A key result of our work is that low data diversity can actually improve performance. This contrasts with classic machine learning principles, which state that low diversity hurts generalization. These two apparently conflicting statements are actually compatible. As our results highlight, low diversity initially accelerates the learning of sparse attention, but high data diversity is better as training time goes to infinity. We hypothesize that data diversity might be a powerful lever towards enabling active learning [62]. Specifically, our findings suggest a simple active learning algorithm: whenever an agent detects it is stuck on a task, it can decrease data diversity to accelerate learning, then increase diversity after the critical transition occurs to improve generalization. This dynamic adjustment of diversity provides a practical mechanism for agents to actively control their own learning trajectories, with the ultimate goal of letting the learner decide what it wants to learn on and reaching human-level sample efficiency that current deep learning systems crucially lack [63–65]. Humans also encounter varying levels of data diversity over development. For example, infants receive progressively increasing data diversity during their early years [66]. Our theory suggests that it may be an important factor in accelerating our development. While promising, this thread requires more extensive analysis, both at larger scale and on more realistic data distributions.

## Acknowledgments

The authors thank Jay McClelland for discussions that inspired this work. F.D. thanks Aditya Varre for the useful discussions.

## References

[1] Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Jauvin. A neural probabilistic language model. *Journal of machine learning research*, 3, 2003.

[2] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, and others. Language models are unsupervised multitask learners. *OpenAI blog*, 1(8):9, 2019.

[3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, and others. Language models are few-shot learners. *Advances in neural information processing systems*, 2020.

[4] OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, and others. GPT-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023.

[5] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, and others. Gemini: a family of highly capable multimodal models. *arXiv preprint arXiv:2312.11805*, 2023.

[6] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

[7] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, and others. Training computeoptimal large language models. *arXiv preprint arXiv:2203.15556*, 2022.

[8] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, and others. Emergent abilities of large language models.

Transactions on machine learning research, 2022.

[9] Deep Ganguli, Danny Hernandez, Liane Lovitt, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova Dassarma, Dawn Drain, Nelson Elhage, and others. Predictability and surprise in large generative models. In *Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency*, pages 1747–1764, 2022.

[10] Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, and others. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. *arXiv preprint arXiv:2206.04615*, 2022.

[11] Philip W Anderson. More is different: Broken symmetry and the nature of the hierarchical structure of science. *Science*, 177(4047), 1972.

[12] Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, and others. In-context learning and induction heads.

Transformer circuits thread, 2022.

[13] Stephanie Chan, Adam Santoro, Andrew Lampinen, Jane Wang, Aaditya Singh, Pierre Richemond, James McClelland, and Felix Hill. Data distributional properties drive emergent in-context learning in transformers. *Advances in neural information processing systems*, 2022.

[14] Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability. *International conference on learning representations*, 2023.

[15] Aaditya K Singh, Ted Moskovitz, Felix Hill, Stephanie CY Chan, and Andrew M Saxe. What needs to go right for an induction head? a mechanistic study of in-context learning circuits and their formation. International Conference on Machine Learning, 2024.

[16] François Charton and Julia Kempe. Emergent properties with repeated examples. *arXiv preprint* arXiv:2410.07041, 2024.

[17] Charlie Snell, Eric Wallace, Dan Klein, and Sergey Levine. Predicting emergent capabilities by finetuning.

In *Conference on Language Modelling*, 2024.

[18] Nicolas Zucchet, Jörg Bornschein, Stephanie Chan, Andrew Lampinen, Razvan Pascanu, and Soham De. How do language models learn facts? Dynamics, curricula and hallucinations. arXiv preprint arXiv:2503.21676, 2025.

[19] Usman Anwar, Abulhair Saparov, Javier Rando, Daniel Paleka, Miles Turpin, Peter Hase, Ekdeep Singh Lubana, Erik Jenner, Stephen Casper, Oliver Sourbut, and others. Foundational challenges in assuring alignment and safety of large language models. *arXiv preprint arXiv:2404.09932*, 2024.

[20] Zhengxiao Du, Aohan Zeng, Yuxiao Dong, and Jie Tang. Understanding emergent abilities of language models from the loss perspective. *Advances in neural information processing systems*, 2024.

[21] Rylan Schaeffer, Brando Miranda, and Sanmi Koyejo. Are emergent abilities of large language models a mirage? *Advances in neural information processing systems*, 2023.

[22] Rosie Zhao, Tian Qin, David Alvarez-Melis, Sham Kakade, and Naomi Saphra. Distributional scaling laws for emergent capabilities. *arXiv preprint arXiv:2502.17356*, 2025.

[23] Neel Nanda, S Rajamanoharan, J Kramár, and R Shah. Fact finding: Attempting to reverse-engineer factual recall on the neuron level. In *AI alignment forum, 2023*, 2023.

[24] Boaz Barak. Emergent abilities and grokking: Fundamental, Mirage, or both?, 2023. URL https://windowsontheory.org/2023/12/22/ emergent-abilities-and-grokking-fundamental-mirage-or-both/.

[25] Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. Dissecting recall of factual associations in auto-regressive language models. *Conference on empirical methods in natural language processing*, 2023.

[26] Zeyuan Allen-Zhu and Yuanzhi Li. Physics of language models: Part 3.1, knowledge storage and extraction.

arXiv preprint arXiv:2309.14316, 2023.

[27] Pierre Marion, Raphaël Berthier, Gérard Biau, and Claire Boyer. Attention layers provably solve singlelocation regression. In *International conference on learning representations*, 2024.

[28] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *Advances in Neural Information Processing* Systems, 2017.

[29] Andrew M Saxe, James L McClelland, and Surya Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. *International conference on learning representations*, 2014.

[30] Francis Bach. *Learning theory from first principles*. 2024. [31] Core Francisco Park, Ekdeep Singh Lubana, Itamar Pres, and Hidenori Tanaka. Competition dynamics shape algorithmic phases of in-context learning. *arXiv preprint arXiv:2412.01003*, 2024.

[32] Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang, Douglas Eck, Chris Callison-Burch, and Nicholas Carlini. Deduplicating training data makes language models better. In *Annual Meeting of the* Association for Computational Linguistics, 2021.

[33] Danny Hernandez, Tom Brown, Tom Conerly, Nova DasSarma, Dawn Drain, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Tom Henighan, Tristan Hume, and others. Scaling laws and interpretability of learning from repeated data. *arXiv preprint arXiv:2205.10487*, 2022.

[34] Diederik P. Kingma and Jimmy Ba. Adam: a method for stochastic optimization. In International Conference on Learning Representations, 2015.

[35] Yushun Zhang, Congliang Chen, Tian Ding, Ziniu Li, Ruoyu Sun, and Zhiquan Luo. Why transformers need Adam: A Hessian perspective. *Advances in neural information processing systems*, 37, 2024.

[36] Gautam Reddy. The mechanistic basis of data dependence and abrupt learning in an in-context classification task. In *International conference on learning representations*, 2024.

[37] Alberto Bietti, Vivien Cabannes, Diane Bouchacourt, Herve Jegou, and Leon Bottou. Birth of a transformer:
A memory viewpoint. *Advances in neural information processing systems*, 2023.

[38] Ezra Edelman, Nikolaos Tsilivis, Benjamin Edelman, Eran Malach, and Surbhi Goel. The evolution of statistical induction heads: In-context learning markov chains. Advances in neural information processing systems, 2024.

[39] Eshaan Nichani, Alex Damian, and Jason D Lee. How transformers learn causal structure with gradient descent. *International conference on machine learning*, 2024.

[40] Francesco D'Angelo, Francesco Croce, and Nicolas Flammarion. Selective induction heads: How transformers select causal structures in context. In *International conference on learning representations*, 2025.

[41] Soham De, Samuel L Smith, Anushan Fernando, Aleksandar Botev, George Cristian-Muraru, Albert Gu, Ruba Haroun, Leonard Berrada, Yutian Chen, Srivatsan Srinivasan, and others. Griffin: Mixing gated linear recurrences with local attention for efficient language models. *arXiv preprint arXiv:2402.19427*, 2024.

[42] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. Conference on language modeling, 2024.

[43] Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transformers with the delta rule over sequence length. In *Advances in neural information processing systems*, 2024.

[44] Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra, and Christopher Ré. Zoology: Measuring and improving recall in efficient language models. In International conference on learning representations, 2023.

[45] Aaditya Singh, Stephanie Chan, Ted Moskovitz, Erin Grant, Andrew Saxe, and Felix Hill. The transient nature of emergent in-context learning in transformers. *Advances in neural information processing systems*, 2024.

[46] Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra. Grokking: Generalization beyond overfitting on small algorithmic datasets. *arXiv preprint arXiv:2201.02177*, 2022.

[47] Andrew M Saxe, James L McClelland, and Surya Ganguli. A mathematical theory of semantic development in deep neural networks. *Proceedings of the National Academy of Sciences*, 116(23), 2019.

[48] Aditya Vardhan Varre, Maria-Luiza Vladarean, Loucas Pillaud-Vivien, and Nicolas Flammarion. On the spectral bias of two-layer linear networks. *Advances in neural information processing systems*, 2023.

[49] Arthur Jacot, François Ged, Berfin ¸Sim¸sek, Clément Hongler, and Franck Gabriel. Saddle-to-saddle dynamics in deep linear networks: Small initialization training, symmetry, and sparsity. arXiv preprint arXiv:2106.15933, 2021.

[50] Emmanuel Abbe, Enric Boix Adsera, and Theodor Misiakiewicz. The merged-staircase property: a necessary and nearly sufficient condition for sgd learning of sparse functions on two-layer neural networks. In *Conference on learning theory*, 2022.

[51] Scott Pesme and Nicolas Flammarion. Saddle-to-saddle dynamics in diagonal linear networks. Advances in neural information processing systems, 2023.

[52] Emmanuel Abbe, Enric Boix Adsera, and Theodor Misiakiewicz. SGD learning on neural networks: Leap complexity and saddle-to-saddle dynamics. In *Conference on learning theory*, 2023.

[53] Yedi Zhang, Aaditya K Singh, Peter E Latham, and Andrew Saxe. Training dynamics of in-context learning in linear attention. *arXiv preprint arXiv:2501.16265*, 2025.

[54] Siyu Chen, Heejune Sheen, Tianhao Wang, and Zhuoran Yang. Training dynamics of multi-head softmax attention for in-context learning: emergence, convergence, and optimality. In Conference on learning theory, 2024.

[55] Sumio Watanabe. *Algebraic geometry and statistical learning theory*. Cambridge University Press, 2009. [56] Jesse Hoogland, George Wang, Matthew Farrugia-Roberts, Liam Carroll, Susan Wei, and Daniel Murfet.

The developmental landscape of in-context learning. *arXiv preprint arXiv:2402.02364*, 2024.

[57] Sanjeev Arora and Anirudh Goyal. A theory for emergence of complex skills in language models. arXiv preprint arXiv:2307.15936, 2023.

[58] Roee Hendel, Mor Geva, and Amir Globerson. In-context learning creates task vectors. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings of the association for computational linguistics: EMNLP 2023, 2023.

[59] Eric Todd, Millicent L Li, Arnab Sen Sharma, Aaron Mueller, Byron C Wallace, and David Bau. Function vectors in large language models. *International conference on learning representations*, 2024.

[60] Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers.

In *International conference on learning representations*, 2024.

[61] Abram Hindle, Earl T Barr, Mark Gabel, Zhendong Su, and Premkumar Devanbu. On the naturalness of software. *Communications of the ACM*, 59(5), 2016.

[62] Burr Settles. Active learning literature survey. University of Wisconsin-Madison, Department of Computer Sciences, 2009.

[63] Brenden M Lake, Ruslan Salakhutdinov, and Joshua B Tenenbaum. Human-level concept learning through probabilistic program induction. *Science*, 350(6266), 2015.

[64] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves, Martin Riedmiller, Andreas K. Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, and Demis Hassabis. Human-level control through deep reinforcement learning. *Nature*, 518(7540), 2015.

[65] Alex Warstadt, Aaron Mueller, Leshem Choshen, Ethan Wilcox, Chengxu Zhuang, Juan Ciro, Rafael Mosquera, Bhargavi Paranjape, Adina Williams, Tal Linzen, and others. Findings of the BabyLM challenge:
Sample-efficient pretraining on developmentally plausible corpora. *arXiv preprint arXiv:2504.08165*,
2025.

[66] Linda B Smith, Swapnaa Jayaraman, Elizabeth Clerkin, and Chen Yu. The developing infant creates a curriculum for statistical learning. *Trends in cognitive sciences*, 22(4), 2018.

[67] Jeremy M Cohen, Simran Kaur, Yuanzhi Li, J Zico Kolter, and Ameet Talwalkar. Gradient descent on neural networks typically occurs at the edge of stability. In International Conference on Learning Representations, 2021.

[68] James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, and Skye Wanderman-Milne. JAX: composable transformations of Python+ NumPy programs, 2018. URL http://github.com/google/jax.

[69] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: an imperative style, high-performance deep learning library. In Advances in neural information processing systems, 2019.

## Neurips Paper Checklist 1. **Claims**

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes] Justification: The abstract enumerates the two principal contributions (the toy linear-regression analysis and the associative-recall study) and these are precisely the contributions developed in Sections 2 and 3 of the paper, with no additional claims beyond that scope. Guidelines:
- The answer NA means that the abstract and introduction do not include the claims made in the paper.

- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. **Limitations**

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes] Justification: The assumption of the theory are thoroughly discussed in Appendix A.1. The limitations of repetition are mentioned throughout the paper. Guidelines:
- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.

- The authors are encouraged to create a separate "Limitations" section in their paper. - The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. **Theory Assumptions And Proofs**

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

## Answer: [Yes]

Justification: Appendix A outlines the modeling assumptions, derives the ODEs and supplies full proofs for both the no-repetition and repetition cases. Guidelines:
- The answer NA means that the paper does not include theoretical results.

- All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced. - All assumptions should be clearly stated or referenced in the statement of any theorems. - The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. **Experimental Result Reproducibility**

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results?

Answer: [Yes] Justification: The Appendix specifies training details, and the code used to perform experiments is provided as supplementary material. Guidelines:
- The answer NA means that the paper does not include experiments. - If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model
(e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. **Open Access To Data And Code**

Question: Does the paper provide open access to the data and code?

Answer: [Yes] Justification: Code is provided as supplementary material.

## Guidelines:

- The answer NA means that paper does not include experiments requiring code. - Please see the NeurIPS code and data submission guidelines (https://nips.cc/public/
guides/CodeSubmissionPolicy) for more details.

- While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https://nips.

cc/public/guides/CodeSubmissionPolicy) for more details.

- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

- At submission time, to preserve anonymity, the authors should release anonymized versions
(if applicable).

- Providing as much information as possible in supplemental material (appended to the paper)
is recommended, but including URLs to data and code is permitted.

## 6. **Experimental Setting/Details**

Question: Does the paper specify all the training and test details necessary to understand the results?

Answer: [Yes]
Justification: All experimental details are specified in the appendix, as well as in the code base.

Guidelines:
- The answer NA means that the paper does not include experiments. - The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. **Experiment Statistical Significance**

Question: Does the paper report error bars or other statistical-significance information?

Answer: [Yes]
Justification: Figures report point curves and R2 fits to power-laws, but do not include confidence intervals, standard deviations or hypothesis tests. Guidelines:
- The answer NA means that the paper does not include experiments. - The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors). - It should be clear whether the error bar is the standard deviation or the standard error of the mean.

- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).

- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. **Experiments Compute Resources**

Question: For each experiment, does the paper provide sufficient information on the computer resources used?

Answer: [Yes] Justification: These details are provided in the appendix.

Guidelines:
- The answer NA means that the paper does not include experiments. - The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. **Code Of Ethics**

Question: Does the research conform with the NeurIPS Code of Ethics?

Answer: [Yes] Justification: The work is purely theoretical/simulation based, uses only synthetic data, and raises no privacy or human-subject concerns; no deviations from the Code of Ethics are indicated. Guidelines:
- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. - If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. **Broader Impacts**

Question: Does the paper discuss both potential positive and negative societal impacts?

Answer: [NA] Justification: This work aims at gaining fundamental insights on the training of large language models and has therefore no direct societal impact. Guidelines:
- The answer NA means that there is no societal impact of the work performed. - If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

- Examples of negative societal impacts include potential malicious or unintended uses (e.g.,
disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. **Safeguards**

Question: Does the paper describe safeguards for responsible release of high-risk data or models?

Answer: [NA] Justification: No models or datasets with foreseeable misuse potential are released.

Guidelines:
- The answer NA means that the paper poses no such risks.

- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. **Licenses For Existing Assets**

Question: Are licences for external assets properly cited?

Answer: [Yes] Justification: The paper reports the frameworks used to train neural networks.

Guidelines:
- The answer NA means that the paper does not use existing assets. - The authors should cite the original paper that produced the code package or dataset. - The authors should state which version of the asset is used and, if possible, include a URL. - The name of the license (e.g., CC-BY 4.0) should be included for each asset. - For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. **New Assets**

Question: Are new assets introduced in the paper well documented?

Answer: [Yes] Justification: The code is well documented, in particular, on how to reproduce experiments.

Guidelines:
- The answer NA means that the paper does not release new assets. - Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

- The paper should discuss whether and how consent was obtained from people whose asset is used.

- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

14. **Crowdsourcing and research with human subjects**
Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]
Guidelines:
- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

15. **Institutional review board (IRB) approvals or equivalent for research with human subjects**
Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]
Guidelines:
- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

16. **Declaration of LLM usage**
Question: Does the paper describe the usage of LLMs if it is an important, original, or nonstandard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]
- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

- Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.