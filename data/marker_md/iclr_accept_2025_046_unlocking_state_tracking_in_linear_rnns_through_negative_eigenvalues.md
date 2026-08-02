# UNLOCKING STATE-TRACKING IN LINEAR RNNS THROUGH NEGATIVE EIGENVALUES

Riccardo Grazzi∗♡, Julien Siems∗♢, Arber Zela♢,

Jorg K.H. Franke ¨ ♢, Frank Hutter♢♣, Massimiliano Pontil♡♠

Equal contribution<sup>∗</sup> , CSML, Istituto Italiano di Tecnologia♡, University of Freiburg♢,

ELLIS Institute Tubingen ¨ ♣, AI Centre, University College London♠

riccardograzzi4@gmail.com juliensiems@gmail.com

# ABSTRACT

Linear Recurrent Neural Networks (LRNNs) such as Mamba, RWKV, GLA, mL-STM, and DeltaNet have emerged as efficient alternatives to Transformers for long sequences. However, both Transformers and LRNNs struggle to perform state-tracking, which may impair performance in tasks such as code evaluation. In one forward pass, current architectures are unable to solve even parity, the simplest state-tracking task, which non-linear RNNs can handle effectively. Recently, [Sarrof et al.](#page-13-0) [\(2024\)](#page-13-0) demonstrated that the failure of LRNNs like Mamba to solve parity stems from restricting the value range of their diagonal state-transition matrices to [0, 1] and that incorporating negative values can resolve this issue. We extend this result to non-diagonal LRNNs such as DeltaNet. We prove that finite precision LRNNs with state-transition matrices having only positive eigenvalues cannot solve parity, while non-triangular matrices are needed to count modulo 3. Notably, we also prove that LRNNs can learn any regular language when their state-transition matrices are products of identity minus vector outer product matrices, each with eigenvalues in the range [−1, 1]. Our experiments confirm that extending the eigenvalue range of Mamba and DeltaNet to include negative values not only enables them to solve parity but consistently improves their performance on state-tracking tasks. We also show that state-tracking enabled LRNNs can be pretrained stably and efficiently at scale (1.3B parameters), achieving competitive performance on language modeling and showing promise on code and math tasks.

# 1 INTRODUCTION

0 10000 20000 Training Steps 0.00 0.25 0.50 0.75 1.00 Scaled AccuracyEigenvalue Range [0, 1] [ 1, 1] Figure 1: Extending the eigenvalue range of the state transition matrices of diagonal LRNNs improves performance from random guessing (range [0, 1]) to perfect score (range [−1, 1]) on learning parity. Trained on sequences up to length 40; Tested on lengths 40–256 (3 seeds). Transformer architectures [\(Vaswani et al., 2017\)](#page-13-1) have revolutionized NLP but scale quadratically in sequence length, posing computational challenges for long sequences. To address this, Linear Recurrent Neural Networks (LRNNs) have emerged as promising alternatives that offer linear scaling while maintaining competitive performance [\(Gu & Dao,](#page-11-0) [2024;](#page-11-0) [Dao & Gu, 2024;](#page-11-1) [Yang et al., 2024a;](#page-13-2) [Peng et al., 2023;](#page-13-3) [Deletang et al., 2023;](#page-11-2) [Sun et al., 2024;](#page-13-4) [Beck et al., 2024\)](#page-10-0). LRNNs update their state via matrix-vector products with structured and often input-dependent state-transition matrices. The structure of the state-transition matrices largely determines the expressivity of LRNNs. While successful models like Mamba [\(Gu & Dao, 2024\)](#page-11-0) and GLA [\(Yang et al.,](#page-13-2) [2024a\)](#page-13-2) use diagonal matrices (diagonal LRNN) which only mix tokens along the sequence dimension, recent work explores more complex forms. Notably, non-diagonal matrices using generalized Householder (GH) transformations, defined as I − uu<sup>⊤</sup> where u is a learnable vector and I is the identity, enable models like DeltaNet [\(Schlag et al., 2021;](#page-13-5) [Yang et al., 2024b\)](#page-13-6) and TTT-Linear [\(Sun et al., 2024\)](#page-13-4) to achieve richer expressiveness through simultaneous token-channel mixing while maintaining efficiency.

![](_page_0_Figure_11.jpeg)

Surprisingly, both Transformers and current LRNNs face a fundamental limitation: they struggle to learn how to track the state of even simple finite-state machines from sequences of state-transitions [\(Deletang et al., 2023\)](#page-11-2). This limitation may impair performance on tasks such as entity tracking in narratives, handling nested structures in code, and reasoning tasks that can benefit from maintaining and updating an internal state over time [\(Merrill et al., 2024\)](#page-12-0). Even the simplest state-tracking task, computing the parity of a sequence of bits, cannot be solved by modern architectures, while non-linear RNNs like LSTM [\(Hochreiter & Schmidhuber, 1997\)](#page-11-3) and sLSTM [\(Beck et al., 2024\)](#page-10-0) can effectively track the state of any finite state machine. However, parallelizing non-linear RNNs across the sequence length presents significant challenges [\(Lim et al., 2024;](#page-12-1) [Gonzalez et al., 2024\)](#page-11-4).

Recently, [Sarrof et al.](#page-13-0) [\(2024\)](#page-13-0) demonstrated that the inability of diagonal LRNNs to solve the *parity* problem stems from the fact that the eigenvalues of their state-transition matrices are constrained to be positive. Specifically, they proved that finite precision diagonal LRNNs with exclusively positive real eigenvalues, cannot solve the parity problem in one forward pass for sequences of arbitrary length. However, their work did not provide empirical evidence showing that diagonal LRNNs with negative eigenvalues can be successfully trained to overcome this limitation. We prove that the same limitation also affects LRNNs with non-diagonal state-transition matrices, and further prove that additionally, non-triangular matrices are necessary to solve the more challenging task of modular counting (when the modulus is not a power of two). Our findings also apply to the GH matrices used by DeltaNet, as they share the same eigenvalue limitations. To overcome this, we propose a simple yet powerful solution: extend the range of possible eigenvalues from [0, 1] to [−1, 1]. This change enables state-tracking and significantly improves the expressivity of LRNNs without compromising their efficiency and training stability. As illustrated in Figure [1,](#page-0-0) it allows diagonal LRNNs to learn parity successfully. The code for part of our experiments is available at [https://github.com/automl/unlocking](https://github.com/automl/unlocking_state_tracking) state tracking

### In summary, we make the following *contributions:*

- 1. We prove that any finite precision LRNN with only positive real eigenvalues in the state-transition matrices (most LRNNs used in practice) cannot solve parity at arbitrary sequence lengths (Theorem [1\)](#page-4-0), while non-triangular matrices are also required to learn counting modulo 3 (Theorem [2\)](#page-4-1).
- 2. By extending the eigenvalue range, we significantly improve the state-tracking capabilities of LRNNs. We prove that LRNNs with state-transition matrices formed by products of generalized Householder (GH) matrices, each with eigenvalues in the range [−1, 1], can learn any regular language (Theorem [4\)](#page-6-0), in some cases with just one layer (Theorem [3\)](#page-5-0). Notably, this range extension allows LRNNs using just one GH matrix (like DeltaNet), to learn substantially harder tasks, as the repeated composition of permutations of two (over n) elements, compared to diagonal LRNNs.
- 3. We show that the eigenvalue range of Mamba and DeltaNet can be extended to [−1, 1] without compromising efficiency or training stability. We test the modified methods on parity, modular arithmetic, and permutation composition, demonstrating improved state-tracking performance.
- 4. We pre-train modified versions of DeltaNet and Mamba (up to 1.3B parameters) and show that they reach performance comparable to the original models on generative language modeling tasks, while DeltaNet shows improved perplexity on code and math datasets.

# 2 RELATED WORK

Linear RNNs. Linear RNNs encompass state-space models and causal, linear attention mechanisms. State-space models, originally used for continuous dynamical systems, inspired LRNN variants like S4 [\(Gu et al., 2022\)](#page-11-5) and H4 [\(Fu et al., 2021\)](#page-11-6) (see [Tiezzi et al.](#page-13-7) [\(2024\)](#page-13-7) for a survey). Recent advancements, such as Mamba [\(Gu & Dao, 2024;](#page-11-0) [Dao & Gu, 2024\)](#page-11-1), introduced input-dependent gating of the hidden state, significantly improving language modeling performance. Concurrently, linear attention has emerged as an alternative to classical softmax attention, with [Katharopoulos](#page-12-2) [et al.](#page-12-2) [\(2020\)](#page-12-2) demonstrating that causal linear attention Transformers can be reformulated as RNNs with linear scaling in sequence length. Building on this, [Yang et al.](#page-13-2) [\(2024a\)](#page-13-2) proposed Gated Linear Attention (GLA), adding a gating mechanism similar to Mamba, while DeltaNet [\(Schlag et al., 2021;](#page-13-5) [Yang et al., 2024b\)](#page-13-6) and TTT-Linear [\(Sun et al., 2024\)](#page-13-4) explored more expressive recurrences with non-diagonal state-transition matrices. [Beck et al.](#page-10-0) [\(2024\)](#page-10-0) recently proposed xLSTM, a successor to LSTM [\(Hochreiter & Schmidhuber, 1997\)](#page-11-3) which combines non-linear and linear RNNs.

Expressivity Results. Several studies have explored the expressive power of Transformers and RNNs (see e.g. [\(Merrill et al., 2020;](#page-12-3) [Strobl et al., 2024;](#page-13-8) [Bhattamishra et al., 2024\)](#page-10-1)). Here, we focus on the ones most relevant to our work. While [Hahn](#page-11-7) [\(2020\)](#page-11-7) proved that Transformers cannot model periodic languages such as parity, see also [\(Bhattamishra et al., 2020,](#page-10-2) Lemma C.4), and some context-free languages at arbitrary sequence lengths, [Liu et al.](#page-12-4) [\(2023\)](#page-12-4) demonstrated that Transformers can learn shortcut solutions for *solvable* finite state automata, though these solutions lack generalizability to arbitrary sequence lengths and perform poorly out-of-distribution. Unlike RNNs, the high parallelizability of Transformers prevents them from learning *unsolvable* finite state automata [\(Merrill & Sabharwal, 2023\)](#page-12-5). These findings typically use techniques from algebraic formal language theory (we refer to [Liu et al.](#page-12-4) [\(2023\)](#page-12-4) for a short tutorial) and circuit complexity, using the *log-precision assumption* and a number of layers scaling linearly or logarithmically with sequence length. While earlier research established Transformers' Turing completeness, it relied on either arbitrary precision [\(Perez et al., 2021\)](#page-13-9) or arbitrary depth and weight sharing [\(Giannou et al., 2023\)](#page-11-8). ´ Diagonal LRNNs can simulate any RNN with infinite depth [\(Gu et al., 2021\)](#page-11-9) and approximate regular enough functions when the state dimension grows linearly with sequence length [\(Orvieto et al.,](#page-12-6) [2024\)](#page-12-6). However, things change when depth and state size are fixed. [Merrill et al.](#page-12-0) [\(2024\)](#page-12-0) proved that finite-depth diagonal LRNNs, like Transformers, struggle to learn unsolvable finite state automata when restricted to log-precision arithmetic. The work by [Fan et al.](#page-11-10) [\(2024\)](#page-11-10) highlights a similar limitation, while in a finite precision setting, [Sarrof et al.](#page-13-0) [\(2024\)](#page-13-0) showed that diagonal LRNNs with positive values in the state-transition matrix, while capable of learning all star-free languages, cannot solve even the simple *parity* problem, a non-star-free language recognizable by an automaton with two states. However, their analysis was limited to the diagonal case and they did not test the benefit of negative eigenvalues in practice. Using a continuous time framework, also [Cirone et al.](#page-10-3) [\(2025\)](#page-10-3) pointed out the limitations of diagonal state transition matrices. [Irie et al.](#page-12-7) [\(2021;](#page-12-7) [2023\)](#page-12-8) empirically showed how state-tracking can be enabled by modifying DeltaNet as a fast weight programmer [\(Schmidhuber, 1992\)](#page-13-10), but this makes its recurrence non-linear, hence hard to parallelize. Unlike previous work, we demonstrate that non-diagonal LRNNs like DeltaNet can achieve robust state-tracking through a minimal modification while maintaining efficient large-scale training.

# 3 BACKGROUND

#### 3.1 LINEAR RECURRENT NEURAL NETWORKS (LRNNS)

We describe LRNNs using notation inspired by [Sarrof et al.](#page-13-0) [\(2024\)](#page-13-0), focusing on the core linear recurrences while abstracting away the non-linear computations for each token. LRNNs are stacks of layers that share a common structure but have distinct learnable parameters. Each layer takes input vectors x1, . . . , x<sup>t</sup> ∈ <sup>R</sup> l (outputs of the previous layer) and outputs yˆ1, . . . , yˆ<sup>t</sup> ∈ <sup>R</sup> p as:

$$\begin{aligned} H_i &= \mathbf{A}(\mathbf{x}_i) H_{i-1} + \mathbf{B}(\mathbf{x}_i), & \hat{\mathbf{y}}_i &= \text{dec}(H_i, \mathbf{x}_i), & \text{for all } i \in \{1, \dots, t\}, \\ H_0 &\in \mathbb{C}^{n \times d}, & \mathbf{A} : \mathbb{R}^l \rightarrow \mathbb{C}^{n \times n}, & \mathbf{B} : \mathbb{R}^l \rightarrow \mathbb{C}^{n \times d}, & \text{dec} : \mathbb{C}^{n \times d} \times \mathbb{R}^l \rightarrow \mathbb{R}^p \end{aligned} \quad (1)$$

Here, A, B and dec are learnable, generally non-linear functions, with dec usually containing a feed-forward neural network. This definition encompasses most LRNN variants, which differ in the form of A, B and dec. Table [1](#page-2-0) illustrates how three popular LRNNs fit this framework. For other architectures see [\(Yang et al., 2024b,](#page-13-6) Table 4). Additional details on the notation are in Appendix [A.1.](#page-15-0)

Table 1: Instances of LRNN layers in [\(1\)](#page-2-1), where αt=sigmoid(Wαxt), ∆t=softplus(W∆xt), βt=sigmoid(w<sup>⊤</sup> <sup>β</sup> xt), while qt, k<sup>t</sup> ∈ <sup>R</sup> <sup>n</sup>, v<sup>t</sup> ∈ <sup>R</sup> d are output of learnable functions of xt. Also, ψ : R <sup>d</sup> → <sup>R</sup> d is another learnable function usually containing an MLP and a normalization, while W<sup>1</sup> ∈ <sup>R</sup> n×d , W<sup>∆</sup> ∈ <sup>R</sup> d×l , W<sup>α</sup> ∈ <sup>R</sup> n×l , w<sup>β</sup> ∈ <sup>R</sup> l and w<sup>2</sup> ∈ <sup>R</sup> d are learnable parameters. For simplicity, we omitted 1D convolutions. For Mamba, the matrices in the first two columns represent the recurrence for the i-th row of H<sup>t</sup> and we set kt=(kt,1, . . . , kt,n) <sup>⊤</sup>, W1=(w1,1, . . . , w1,n) ⊤, and l = d.

|          | A ( x t )                           | B ( x t )       | dec( H t , x t )    |
|----------|-------------------------------------|-----------------|---------------------|
| Mamba    | Diag (exp ( − ∆ t ⊙ exp( w 1 ,i ))) | k t,i ∆ t ⊙ x t | ψ ( H ⊤             |
|          |                                     |                 | t q t + w 2 ⊙ x t ) |
| GLA      | Diag ( α t )                        | k t v           |                     |
|          |                                     | t               | ψ ( H ⊤             |
|          |                                     |                 | t q t )             |
| DeltaNet | I − β t k t k                       |                 |                     |
|          | t                                   | β t k t v       |                     |
|          |                                     | t               | ψ ( H ⊤             |
|          |                                     |                 | t q t )             |

The *state-transition matrices* A(xt) are typically diagonal or generalized Householder (GH), i.e., identity minus vector outer product, as shown in Table [1,](#page-2-0) to enable efficient matrix-vector products on modern hardware. These matrices consistently have eigenvalues (and norm) in the range [0, 1].

#### 3.2 FORMAL LANGUAGE THEORY

Finite State Automata and Regular Languages. A (deterministic) finite state automaton (FSA) is a tuple A = (Σ, Q, q0, δ) where Σ is a finite set of letters called alphabet, Q is a finite set of states, q<sup>0</sup> ∈ Q is the starting state and δ : Q × Σ →Q is the state-transition function (see [Hopcroft &](#page-11-11) [Ullman, 2001,](#page-11-11) for an introduction). We define the set Σ ∗ , whose elements are sequences called words, as the smallest superset of Σ that contains the empty word ε and is closed under word concatenation. We extend the state-transition function to δ : Q × Σ <sup>∗</sup> →Q by defining δ(q, ε) = q and δ(q, w) = δ(δ(q, w<sup>1</sup> . . . wi−1), wi) for any w = w<sup>1</sup> . . . w<sup>i</sup> ∈ Σ <sup>∗</sup> with i ≥ 2. We say that δ(q0, w) is the state that A reaches after reading the word w ∈ Σ ∗ . A *language* L ⊆ Σ ∗ is said to be recognized by A if there exists a recognizing set R ⊆ Q such that L = {w ∈ Σ ∗ : δ(q0, w) ∈ R}. Regular languages are the ones that can be recognized by an FSA. Given an FSA A, the set T (A) = {δ(·, w) : w ∈ Σ <sup>∗</sup>} of functions ρ : Q →Q, together with the function composition operation forms a *monoid* called *transition monoid*, i.e. it is associative, closed and contains the identity δ(·, ε). This monoid has a finite number of elements, since |Q| < ∞. Moreover, if δ(·, w) is bijective for every w ∈ Σ, then T (A) forms a *group*, i.e. it contains the inverse of each element.

State-Tracking and Monoid Word Problems. State-tracking is the problem of determining the state of a system only by observing a sequence of updates applied to it. Formally, it can be expressed as a *monoid word problem* [\(Merrill et al., 2024\)](#page-12-0), where given a monoid (M, ·) (M is the set and · is the associative operation), we want to send words m<sup>1</sup> . . . m<sup>t</sup> ∈ M<sup>∗</sup> , describing the sequence of updates, to their product m<sup>1</sup> · m<sup>2</sup> · · · m<sup>t</sup> ∈ M, representing the state of the system after the updates. If M is finite there is a corresponding FSA (M, M, e, δ) that solves the word problem, where the starting state is e (the identity element), and the transition function is δ(m1, m2) = m<sup>2</sup> · m<sup>1</sup> for m1, m<sup>2</sup> ∈ M. In this work, we focus on group word problems, i.e. problems where the monoid is also a group. In particular, on the cyclic group <sup>Z</sup>m, i.e. addition modulo m, and the symmetric group Sm, i.e. the group of permutations on m elements. Parity is equivalent to the S<sup>2</sup> word problem, while many state-tracking problems such as tracking chess moves or code evaluation, can be shown to be harder than the S<sup>5</sup> word problem, which cannot be solved by Transformers and diagonal LRNNs even in log-precision for arbitrary word lengths [\(Merrill et al., 2024;](#page-12-0) [Merrill & Sabharwal, 2023\)](#page-12-5).

One LRNN Layer is an automaton. Given an alphabet Σ ⊂ N, we can view one layer of an LRNN in [\(1\)](#page-2-1) as the automaton Alin = (Σ, H, H0, δlin), where δlin(H, w) = A(w)H + B(w), which is extended as we saw previously[<sup>1</sup>](#page-3-0) , and H = {δlin(H0, w) : w ∈ Σ <sup>∗</sup>} ⊆ <sup>R</sup> n×d . We say that an LRNN layer in [\(1\)](#page-2-1) *implements* the FSA A = (Σ, Q, q0, δ) if Alin can mimic the state transitions of A[<sup>2</sup>](#page-3-1) . Formally, if there exists a surjective function g : H → Q, such that for any H ∈ H, w ∈ Σ δ(g(H), w) = g(δlin(H, w)) = g(A(w)H + B(w)). Every language L recognized by A can also be recognized by this LRNN layer with a sufficiently powerful dec. In particular if R ⊆ Q is the recognizing set for L and q<sup>0</sup> = g(H0), then the decoder dec(Ht, wt) = 1{g(Ht) ∈ R}, will correctly determine if w ∈ L. Therefore, implementing A is at least as hard as recognizing L. A principal goal of this work is to show that current LRNNs cannot recognize simple languages such as parity (negative results) while appropriate modifications to the state-transition matrices, enable LRNNs to implement broader classes of FSA (positive results), with certain classes of FSA requiring a single layer. Note, that while LRNNs with one layer can recognize any regular language, the state transition matrices might not fit into the structure imposed by current LRNNs, such as those in Table [1](#page-2-0) (see Appendix [A.3](#page-16-0) for more details).

# 4 THEORETICAL ANALYSIS

### 4.1 LIMITATIONS OF CURRENT LRNNS

In this section, we describe how positive eigenvalues and non-triangular state transition matrices limit LRNNs state-tracking capabitlies. In particular, we focus on parity and modular addition. The parity y<sup>t</sup> ∈ {0, 1} of a sequence of ones and zeros x<sup>1</sup> . . . x<sup>t</sup> ∈ {0, 1} t is 1 if the total number of ones in the sequence is odd, and 0 if it's even. Equivalent to addition modulo 2, it can be computed by summing the values in the input sequence and then applying the modulo 2 function: y<sup>t</sup> = (P<sup>t</sup> <sup>i</sup>=1 xi) mod 2. This solution can be implemented by an LRNN with one layer and scalar

<sup>1</sup>We let δlin : <sup>R</sup> <sup>n</sup>×<sup>d</sup> × Σ → <sup>R</sup> n×d and extend it to δlin : R <sup>n</sup>×<sup>d</sup> × Σ <sup>∗</sup> → <sup>R</sup> n×d , then we define H.

<sup>2</sup>This definition is equivalent to that of FSA homomorphism, see [\(Maler & Pnueli, 1994,](#page-12-9) Definition 3).

![](_page_4_Figure_1.jpeg)

![](_page_4_Diagram_2.jpeg)

Figure 2: *Parity requires negative eigenvalues.* States of one-layer LRNNs with the sequence 1111 . . . as input. If the eigenvalues of A(1) are nonnegative, the states either diverge or converge monotonically, and so, for large enough t and in finite precision, cannot be distinguished. In contrast, the LRNN with a(1) = −1 alternates between two states like the parity automaton.

states by setting A(xt) = 1, B(xt) = xt, H<sup>0</sup> = 0, and dec(Ht, xt) = H<sup>t</sup> mod 2 in [\(1\)](#page-2-1). However, implementing such a solution with finite precision presents an issue: the state h<sup>t</sup> can grow indefinitely with t, eventually reaching the limit of our precision range. Indeed, h<sup>t</sup> ∈ {0, . . . , t}, requiring log<sup>2</sup> (t + 1) bits for storage. Moreover, in practice dec must approximate the modulus 2 function, which is challenging to learn due to its discontinuous and periodic nature.

A more efficient solution, which implements the two-state FSA solving this problem, can still be realized by a finite precision LRNN with one layer and scalar states (and consequently also with vector states and diagonal state-transition matrices) using the recurrence h<sup>t</sup> = a(xt)ht−<sup>1</sup> + b(xt), h<sup>0</sup> = b(0) = 0, b(1) = a(0) = 1, a(1) = −1, y<sup>t</sup> = ht. Note that the state-transition scalar a(1) is negative, while current diagonal LRNNs do not allow negative values. [\(Sarrof et al., 2024,](#page-13-0) Theorem 2) states that this fact makes real-valued diagonal LRNNs unable to solve parity, which raises the question: *can non-diagonal LRNNs which allow only positive eigenvalues, such as DeltaNet, solve parity?* The following result answers this question negatively by generalizing [Sarrof et al.](#page-13-0) [\(2024,](#page-13-0) Theorem 2) to non-diagonal matrices. To solve parity, the state transition matrices must allow at least one eigenvalue to be neither real nor positive. For non-diagonal matrices, this eigenvalue could simply have nonzero imaginary part. The main idea of the theorem is illustrated in Figure [2.](#page-4-2)

Theorem 1 (Parity). *A finite precision LRNN with finitely many layers as in [\(1\)](#page-2-1) can solve parity for arbitrary input lengths, in particular, it can recognize the language* (11)<sup>∗</sup> *, only if in at least one layer, there exist* x *such that* A(x) *has at least one eigenvalue* λ /∈ {x ∈ <sup>R</sup> : x ≥ 0}*.*

The proof in Appendix [B.1](#page-19-0) uses the same core idea as the one in [\(Sarrof et al., 2024,](#page-13-0) Theorem 2). For one layer, we show that when x = 1<sup>k</sup> and the conditions for the eigenvalues of A(1) are not met, the mapping k 7→ H<sup>k</sup> and consequently also the one k 7→ yˆ<sup>k</sup> will be constant (in finite precision and for large enough k), while k 7→ yk, with y<sup>k</sup> being the parity of x, alternates between 0 and 1. To show this, we use the expression for the powers of the Jordan canonical form of A(1).

We now study the problem of counting modulo m, an easier version of addition modulo m where the input of length k never changes and is x = 1<sup>k</sup> , while the correct output is y<sup>k</sup> = (P<sup>k</sup> <sup>i</sup>=1 xi) mod m. The following theorem shows that to solve this problem, products of state-transition matrices must have at least one eigenvalue with nonzero imaginary part.

Theorem 2 (Modular Counting). *A finite precision LRNN with* L *layers, each as in [\(1\)](#page-2-1), can count modulo* m*, i.e. it can recognize the language* (1<sup>m</sup>) ∗ *, with* m *not a power of two, only if there exist* i ∈ {1, . . . , L} *and* x1, . . . , x<sup>2</sup> <sup>i</sup>−<sup>1</sup> *such that for the* i*-th layer the product* A(x1)A(x2)· · · A(x<sup>2</sup> <sup>i</sup>−<sup>1</sup> ) *has at least one eigenvalue* λ *with nonzero imaginary part, i.e.* λ /∈ <sup>R</sup>*.*

The proof is in Appendix [B.2.](#page-20-0) When L = 1 a key step is to show that if A(1) has real (even negative) eigenvalues, the map k → H<sup>k</sup> will alternate between two values (in finite precision and for large enough k), not enough to count modulo m > 2. For L > 1, we proceed by induction using our assumption on the eigenvalues of the product of state-transition matrices.

Discussion Theorems [1](#page-4-0) and [2](#page-4-1) identify a fundamental limitation of current design choices on the structure of the state-transition matrices of LRNNs. Specifically, current LRNNs, as the ones outlined in Table [1,](#page-2-0) are incapable of solving parity, as the eigenvalues of their state-transition matrices are confined to the interval [0, 1]. Further, even if we allow negative eigenvalues, LRNNs using common structures for the state transition matrices, such as diagonal or triangular with real entries, cannot solve counting modulo m. In contrast, as we will show, LRNNs with state-transition matrices that are (products of) generalized Householder matrices, each with eigenvalues in the range [−1, 1], are much more expressive.

### 4.2 ALLOWING NEGATIVE EIGENVALUES

We focus on two classes of LRNNs determined by the structure of their state-transition matrices: diagonal (such as Mamba, Mamba2, and GLA) and generalized Householder (GH, as in DeltaNet). In particular, if we let s : R <sup>l</sup> → [0, 1]<sup>n</sup>, ϕ : <sup>R</sup> <sup>l</sup> → [0, 1] and v : <sup>R</sup> <sup>l</sup> → <sup>R</sup> <sup>n</sup>, being learnable functions such that ∥v(x)∥ = 1 for every x ∈ <sup>R</sup> , then the state transition matrices of each layer of many LRNNs, such as those in Table [1,](#page-2-0) can be written as either

$$\mathbf{A}_{\text{diag}}(\mathbf{x}) := \text{Diag}(\mathbf{s}(\mathbf{x})), \quad \text{or} \quad \mathbf{A}_{\text{GH}}(\mathbf{x}) := \mathbf{I} - \phi(\mathbf{x})\mathbf{v}(\mathbf{x})\mathbf{v}(\mathbf{x})^\top,$$

where Adiag(x) is diagonal with eigenvalues s(x)<sup>i</sup> ∈ [0, 1], while AGH(x) is GH with all eigenvalues equal to one except for the one associated to the eigenvector v(x), which is equal to 1 − ϕ(x) ∈ [0, 1]. To address the limitations discussed in the previous section, we propose the following modification that can be easily applied to LRNNs belonging to either class.

$$\mathbf{A}_{\text{diag}}^-(\mathbf{x}) := \text{Diag}(2\mathbf{s}(\mathbf{x}) - \mathbf{1}), \quad \mathbf{A}_{\text{GH}}^-(\mathbf{x}) := \mathbf{I} - 2\phi(\mathbf{x})\mathbf{v}(\mathbf{x})\mathbf{v}(\mathbf{x})^\top. \quad (2)$$

Hence, A<sup>−</sup> diag(x) has eigenvalues <sup>2</sup>s(x)<sup>i</sup> <sup>−</sup> <sup>1</sup> <sup>∈</sup> [−1, 1] and <sup>A</sup><sup>−</sup> GH(x) has one eigenvalue equal to 1−2ϕ(x) ∈ [−1, 1]. Thus, we have extended the eigenvalues range from [0, 1] to [−1, 1]. The norm of the matrix is still less than or equal to one, keeping the recurrence stable at long sequence lengths.

LRNNs with the modified state transition matrices can implement the solution to parity in [\(2\)](#page-5-1) by setting s(1) = 0 and ϕ(1) = 1 so that if we consider a scalar recursion, then A<sup>−</sup> diag(1) = −1. However, Theorem [2](#page-4-1) shows that we cannot count modulo 3 with triangular state transition matrices, even when allowing negative eigenvalues. Therefore, in the next section, we examine the impact of our change to the eigenvalue range on non-triangular state-transition matrices.

### 4.3 EXPRESSIVITY OF PRODUCTS OF GENERALIZED HOUSEHOLDER MATRICES

We focus on state-transition matrices that are products of k GH matrices. For DeltaNet k = 1. For any n, k ∈ N, we define the set of all matrices in R <sup>n</sup>×<sup>n</sup> that can be expressed as a product of k GH matrices, each having the only interesting eigenvalue in the range Ω ⊆ R, as

$$\mathcal{M}_k^n(\Omega) := \{C_1 C_2 \cdots C_k : C_i = I - \beta_i \mathbf{v}_i \mathbf{v}_i^\top, \quad (1 - \beta_i) \in \Omega, \quad \mathbf{v}_i \in \mathbb{R}^n, \|\mathbf{v}_i\| = 1\}. \quad (3)$$

Intuitively, higher k means higher expressivity but also higher cost for matrix-vector products. Furthermore, as long as Ω ⊆ [−1, 1], the norm of the matrices is bounded by one, which guarantees that repeated matrix product do not diverge. We observe that if M ∈ M<sup>n</sup> 1 ({−1}), then M is a reflection (or Householder) matrix, and that for any x ∈ R l , AGH(x) ∈ M<sup>n</sup> 1 ([0, 1]) and A<sup>−</sup> GH(x) ∈ M<sup>n</sup> 1 ([−1, 1]) so that with our change we also include reflections. Moreover, M<sup>n</sup> k (Ω) ⊆ M<sup>n</sup> <sup>k</sup>′ (Ω′ ) if Ω ⊆ Ω ′ and either k ′ = k or k ′ ≥ k, 1 ∈ Ω.

Our next result shows that products of GH matrices can represent any matrix with Euclidean norm less than or equal to 1, but only when [−1, 1] ⊆ Ω. In contrast, repeated products of (e.g. upper) triangular matrices with eigenvalues in [−1, 1] remain triangular, with eigenvalues in the same range.

Proposition 1 (Expressivity of products of GH matrices). *The following hold for* M<sup>n</sup> k *in [\(3\)](#page-5-2):*

- *1. For any* N ∈ M<sup>n</sup> k ([−1, 1])*,* ∥N∥ ≤ 1*.*
- *2. For any* M ∈ R <sup>n</sup>×<sup>n</sup> *with* ∥M∥≤ 1*, then* M ∈ M<sup>n</sup> 3n ([−1, 1]) *and if* M *is orthogonal then* M ∈ M<sup>n</sup> n ({−1, 1})*, while* M ∈ M<sup>n</sup> n−1 ({−1, 1}) *when* M *is a permutation matrix.*
- *3. Any eigenvalue* λ *of any matrix* N ∈ M<sup>n</sup> k ((−1, 1]) *is either* 1 *or satisfies* |λ| < 1 *and if in addition* N ∈ M<sup>n</sup> k ([0, 1]) *and* k ≤ 2*, then* λ ∈ [0, 1] ⊂ <sup>R</sup>*.*

The proof in Appendix [C.2](#page-22-0) uses mainly linear algebra arguments such as the SVD decomposition and the fact that every n × n orthogonal matrix can be written as a product of n reflections, due to the Cartan–Dieudonne Theorem [\(Gallier & Gallier, 2011\)](#page-11-12). ´

A consequence of Proposition [1.](#page-5-3)3 is that LRNNs with layers of the form [\(1\)](#page-2-1), where A : R <sup>l</sup> → M<sup>n</sup> k ([0, 1]), have state transition matrices that are either the identity or not orthogonal, and hence cannot be reflections or rotations. Also, if k ≤ 2 the eigenvalues are positive and hence the LRNN cannot learn parity due to Theorem [1.](#page-4-0) In contrast, if we allow A : R <sup>l</sup> → M<sup>n</sup> k ([−1, 1]) and k is large enough, the following theorem shows that an LRNN with one layer can implement any FSA whose transition monoid is a group, and that n = k = 2 is enough for cyclic groups (modular addition).

2 1 3 1 3 2 2 3 1

swap swap

0 1 0 1 0 0 0 0 1

I − 2v1v ⊤ 1

1 0 0 0 0 1 0 1 0

I − 2v2v ⊤ 2

0 1 0 0 0 1 1 0 0

<sup>×</sup> <sup>=</sup>

Figure 3: A permutation of k elements is also a composition of at most k−1 swaps. This maps to a product of k−1 Hoseholders, each representing a swap. Illustrated for k = 3. v ⊤ <sup>1</sup> = √ 1 , − √ 2 , 0 , v ⊤ <sup>2</sup> = 0, √ 2 , − √ 1 .

Theorem 3. *Every FSA* A = (Σ, Q, q0, δ) *whose transition monoid* T (A) *is a group, can be implemented by a finite precision LRNN with one layer and* A : Σ → M<sup>n</sup> k−1 ({−1, 1})*, where* n *is the smallest natural number such that* T (A) *is isomorphic to a subgroup of* Sn*, and* k = maxw∈<sup>Σ</sup> P <sup>q</sup>∈<sup>Q</sup> 1{δ(q, w) ̸= q} *is the maximum number of changed states after applying a single transition. Moreover, if* T (A) *is isomorphic to the cyclic group* <sup>Z</sup>m*, then we can set* A : Σ → M<sup>2</sup> 2 ([−1, 1]) *and if* m = 2 *(parity) we can set* A : Σ → {−1, 1}*.*

In the proof in Appendix [C.3,](#page-23-0) we map each state-transition function to a matrix representation. This can always be done using permutation matrices, but for cyclic groups, we can also use rotation matrices (Appendix [C.1\)](#page-21-0). For permutations, if every state-transition permutes at most k states then the corresponding permutation matrix will be in M<sup>n</sup> k−1 ({−1, 1}), since it is either the identity or can be written as a product of at most k − 1 permutations of two elements (swaps), each in M<sup>n</sup> 1 ({−1}) (see Figure [3\)](#page-6-1). A consequence of Theorem [3](#page-5-0) is that if every transition function of the FSA has a permutation representation corresponding to a swap or the identity, then an LRNN layer with A = A<sup>−</sup> GH, can implement it. This is useful in practice because the time complexity of an LRNN having a product of k GH matrices as one state-transition matrix increases linearly with k. Also, for natural language tasks, the state-transitions for the FSA might be either simple or encoded using multiple letters. For example, for addition modulo 5, a word may look like "3+2+4=4" (two letters per addition). This allows an LRNN with state-transition matrices inM<sup>n</sup> 1 ([−1, 1]) to model complex transitions. Indeed, if each transition uses k letters and we set B ≡ 0 and A : R <sup>l</sup> → M<sup>n</sup> 1 ([−1, 1]) in [\(1\)](#page-2-1), then the LRNN layer can model permutations that change up to k + 1 elements since

$$H_t = C(x_t, \dots, x_{t-k})H_{t-k}, \quad C(x_t, \dots, x_{t-k}) := A(x_t)A(x_{t-1}) \cdots A(x_{t-k}) \in \mathcal{M}_k^n([-1, 1]).$$

In Appendix [D](#page-25-0) we also show that, interestingly, an LRNN with two layers (instead of just one), each having only reflections (instead of rotations) as state-transition matrices, can solve addition modulo m. We now present an important result on the expressivity of LRNNs with multiple layers.

Theorem 4. *LRNNs with state transition matrices that are repeated products of GH matrices, each with eigenvalues in the range* [−1, 1]*, can recognize any regular language. In particular, every FSA* A = (Σ, Q, q0, δ) *can be implemented by a finite precision LRNN with* s ≤ 2 |Q| *layers, each of the form [1,](#page-2-1) where* n ≤ |Q|*,* p ≤ s*,* d = 1*,* A : <sup>R</sup> <sup>l</sup> → M<sup>n</sup> n ([−1, 1]) *and* B : <sup>R</sup> <sup>l</sup> → <sup>N</sup> n*.*

The proof in Appendix [C.5](#page-24-0) exploits the landmark Theorem by [Krohn & Rhodes](#page-12-10) [\(1965\)](#page-12-10), which states that every FSA can be decomposed as a *cascade* of simpler FSAs whose state-transition functions are either one-to-one or constant. Each layer of the LRNN will implement one FSA (with n states) of the cascade using n × n permutation matrices, which are in M<sup>n</sup> n−1 ({−1, 1}), for the one-to-one transitions, while for constant (state-independent) transitions it will set the corresponding statetransition matrix to 0 ∈ M<sup>n</sup> n ({0}) and the function B appropriately. Note that we can obtain the zero matrix only inefficiently as a product of n GH matrices, while it could also be obtained with a single diagonal matrix. This points towards LRNNs using a mix of GH and diagonal matrices, as recently explored by Gated DeltaNet [\(Yang et al., 2025\)](#page-14-0) and [RWKV-7.](https://github.com/BlinkDL/modded-nanogpt-rwkv)

Discussion The results in Theorems [3](#page-5-0) and [4](#page-6-0) for LRNNs are in sharp contrast with the ones for Transformers [\(Liu et al., 2023;](#page-12-4) [Merrill & Sabharwal, 2023\)](#page-12-5) and diagonal LRNNs [\(Merrill et al., 2024\)](#page-12-0), which require either the number of layers or the precision growing with the input sequence length, and can only implement an FSA if all groups in its transition monoid are *solvable*, i.e. excluding groups isomorphic to S<sup>n</sup> with n ≥ 5. However, compared to LRNNs without any restriction to the norm of the state-transition matrices, which need only one layer to recognize any regular language, our result requires both the number of layers and the width of the LRNN to be (in the worst case) exponential in the number of states of the FSA, although we conjecture that the number of layers might be reduced to at most linear using a more refined decomposition.

### 5 EXPERIMENTS

Table 2: Summary of modifications to the state-transition matrices A(xt) to extend the eigenvalue range from [0, 1] (Table [1\)](#page-2-0) to [−1, 1]. We set s(xt) = exp (−∆<sup>t</sup> exp(w1,i)).

|          | [0 , 1]          | [ − 1 , 1]            |
|----------|------------------|-----------------------|
| Mamba    | Diag( s ( x t )) | Diag(2 s ( x t ) − 1) |
| DeltaNet | I − β t k t k    |                       |
|          | t                | I − 2 β t k t k       |

We investigate the effects of expanding the eigenvalue range of state-transition matrices from [0, 1] to [−1, 1], as explained in Section [4.2,](#page-5-4) on both synthetic tasks and language modeling. Our experiments involve Mamba, and DeltaNet, with variants trained using both the original and extended eigenvalue ranges, as shown in Table [2.](#page-7-0) We label these variants accordingly. Note that the changes increase the expressivity of Mamba and DeltaNet while coming at no additional computational cost. Detailed information on the implementation can be found in Appendix [E.4.](#page-30-0)

#### 5.1 CHOMSKY HIERARCHY

Table 3: Performance comparison of various recurrent models on formal language tasks. We report the best of 3 runs (Table [5](#page-27-0) in the Appendix reports the median). Scores are scaled accuracy, with 1.0 indicating perfect performance and 0.0 random guessing. The positive impact of allowing negative eigenvalues ([−1, 1] range) versus restricting to positive eigenvalues ([0, 1] range) is evident for both Mamba and DeltaNet. Results in parenthesis are as reported in [Beck et al.](#page-10-0) [\(2024\)](#page-10-0).

|             |            |       | Parity | Mod. (w/o | Arithm. brackets) | Mod. Arithm. (w/ brackets) |
|-------------|------------|-------|--------|-----------|-------------------|----------------------------|
| Transformer |            |       | 0.022  |           | 0.031             | 0.067                      |
| mLSTM       |            | 0.087 | (0.04) | 0.040     | (0.04)            | 0.114 (0.03)               |
| sLSTM       |            | 1.000 | (1.00) | 0.787     | (1.00)            | 0.178 (0.57)               |
| Mamba       | [0 , 1]    |       | 0.000  |           | 0.095             | 0.123                      |
| Mamba       | [ − 1 , 1] |       | 1.000  |           | 0.241             | 0.116                      |
| DeltaNet    | [0 , 1]    |       | 0.017  |           | 0.314             | 0.194                      |
| DeltaNet    | [ − 1 , 1] |       | 1.000  |           | 0.971             | 0.260                      |

We conducted experiments with some of the formal language tasks proposed by [Deletang et al.](#page-11-2) [\(2023\)](#page-11-2) and similarly used to benchmark xLSTM [\(Beck et al.,](#page-10-0) [2024\)](#page-10-0). Our focus was on tasks where mLSTM (an LRNN) previously underperformed while sLSTM (a non-linear RNN) succeeded, specifically parity, modular arithmetic without brackets (both regular languages) and modular arithmetic with brackets (context-free language). As in [Beck et al.](#page-10-0) [\(2024\)](#page-10-0), we trained each model with sequence lengths ranging from 3 to 40 and evaluated on lengths from 40 to 256, to assess length generalization. Note that our theoretical results cover just regular languages, excluding modular arithmetic with brackets. We compared a Transformer, mLSTM and sLSTM against

two variants each of Mamba and DeltaNet - with and without eigenvalue range extension.

Results Our findings, presented in Table [3,](#page-7-1) demonstrate that expanding the range of eigenvalues from [0, 1] to [−1, 1] enables all examined models to fully solve the parity task, confirming Theorem [1.](#page-4-0) For both modular arithmetic tasks, this expansion led to substantial performance improvements for Mamba and especially DeltaNet, since the latter has non-diagonal state-transition matrices that are more suited for these tasks (see Theorem [3\)](#page-5-0). In Figure [6](#page-28-0) in the Appendix, we visualize the length extrapolation performance of each model on all considered tasks. Note that we were unable to reproduce the sLSTM results reported by [Beck et al.](#page-10-0) [\(2024\)](#page-10-0) for the modular arithmetic tasks. Additional experiments and details on the tasks in Appendix [E.1.](#page-26-0)

### 5.2 STATE-TRACKING

We perform experiments on group word problems, relying on the code provided by [Merrill et al.,](#page-12-0) [2024.](#page-12-0) We focus on the S<sup>5</sup> group—the first *unsolvable* symmetric group where current LRNNs and Transformers are known to underperform. We also report results for addition modulo 60 (i.e., the cyclic group <sup>Z</sup>60) in Appendix [E.2.2,](#page-29-0) and note that parity corresponds to S2. In these experiments, the model receives a sequence of group elements as input, and the supervision is another sequence of group elements, each representing the product of the preceding input elements. Since solving S<sup>5</sup> might need LRNNs with state-transition matrices formed by repeated products of four GH matrices (see Theorem [3\)](#page-5-0), each with eigenvalues in [−1, 1], we also consider three simplified setups: (i) allowing only permutations of up to 2 elements (identity and swaps), (ii) allowing only permutations of up to 3 elements, and (iii) using 4 tokens for each permutation. Additional details are in Ap-

![](_page_8_Figure_1.jpeg)

Figure 4: Sequence accuracy for varying sequence lengths on S<sup>5</sup> after 100 epochs of training. We report the best of 3 seeds for each method (in Figure [7](#page-29-1) we report all seeds). The dashed vertical line indicates the sequence length used during training (32 except for the third plot from the left where it is 64). Each method is labeled with name, eigenvalue range, and number of layers. The dashed vertical line indicates the sequence length used during training. "Full matrix simple" is a one-layer baseline where the state update matrices are full and we have no control over the eigenvalue range.

![](_page_8_Figure_3.jpeg)

Figure 5: Performance vs sequence length of DeltaNet variants (340M (top) and 1.3B (bottom) parameters) on four datasets. DeltaNet with eigenvalue range [−1, 1] improves perplexity in coding and math compared to the [0, 1] baseline. Dashed vertical line at training context length (2048).

pendix [E.2.](#page-27-1) We stress that, even when restricting the inputs to only identity and swaps, the group elements for the supervision still cover the entire group, because swaps are generators of the group.

Results Figure [4](#page-8-0) shows that, as predicted by Theorem [3,](#page-5-0) restricting the inputs to only swap permutations allows DeltaNet [−1, 1] with even one layer to fully learn the task (since its state-transition matrices can model swaps), while DeltaNet [0, 1] with 5 layers generalizes just slightly beyond the training length. In contrast, by including also permutations of 3 elements, we notice a substantial decrease in the performance of all models. Interestingly, extending the range is still advantageous in this case and DeltaNet [−1, 1] with 5 layers reaches a good length generalization. Moreover, using 4 tokens per group element seems also beneficial compared to standard S5, since DeltaNet [−1, 1] with 5 layers manages to extrapolate very well until around length 200, which corresponds to 50 group elements, while on standard S<sup>5</sup> all models have 0 sequence accuracy prior to sequence length 30. We also report that Mamba, a diagonal LRNN, performs poorly on all setups, with and without increased eigenvalue range.

# 5.3 LANGUAGE MODELING

Experimental Setup We train DeltaNet models with 340M and 1.3B parameters and Mamba models with 370M parameters, each using both original and extended eigenvalue ranges. Training is done on the full FineWeb-100B dataset [\(Penedo et al., 2024\)](#page-12-11). We chose FineWeb rather than FineWeb-Edu since it contains more code. We aligned our training pipeline with [Yang et al.](#page-13-6) [\(2024b\)](#page-13-6); see Appendix [E.3.1](#page-30-1) for details. Given our previous theoretical and experimental findings, we hypothesize that models (especially DeltaNet) with extended eigenvalue range will perform better on language modeling tasks linked to state-tracking such as coding or mathematics, compared to unmodified models. To test this hypothesis, we evaluate the perplexity of these models in a length extrapolation setup using various datasets: CodeParrot [\(Tunstall et al., 2022\)](#page-13-11) for coding, Math-Hard [\(Hendrycks](#page-11-13) [et al., 2021\)](#page-11-13) for mathematics, TriviaQA [\(Joshi et al., 2017\)](#page-12-12), and SlimPajama [\(Soboleva et al., 2023\)](#page-13-12).

Results All models trained stably with our modification and without changing the learning rate. The validation perplexity of the proposed variants was comparable, albeit slightly worse than that of the original models throughout training (see Figure [9](#page-31-0) in the Appendix). The experiments in Fig-

Table 4: Performance comparison using lm-harness benchmark [\(Gao et al., 2024\)](#page-11-14) (SlimPajama (SPJ) reproduced from [Yang et al.](#page-13-6) [\(2024b\)](#page-13-6), Fine-Web (FW) ours). Results are shown for the original and extended eigenvalue range. Our models show comparable performance across tasks.

| Model 340M params 15B       | Wiki. ppl ↓ | LMB. ppl ↓ | LMB. acc ↑ | PIQA acc ↑ | Hella. acc n ↑ | Wino. acc ↑ | ARC-e acc ↑ | ARC-c acc n ↑ | Avg. ↑ | SWDE cont. ↑ | SQUAD cont. ↑ | FDA cont. ↑ |
|-----------------------------|-------------|------------|------------|------------|----------------|-------------|-------------|---------------|--------|--------------|---------------|-------------|
| Transformer++ SlimPajama    | 28.39       | 42.69      | 31.0       | 63.3       | 34.0           | 50.4        | 44.5        | 24.2          | 41.2   | 42.2         | 22.1          | 21.4        |
| Mamba [0 , 1]               | 28.39       | 39.66      | 30.6       | 65.0       | 35.4           | 50.1        | 46.3        | 23.6          | 41.8   | 12.4         | 23.0          | 2.1         |
| GLA [0 , 1]                 | 29.47       | 45.53      | 31.3       | 65.1       | 33.8           | 51.6        | 44.4        | 24.6          | 41.8   | 24.0         | 24.7          | 7.3         |
| DeltaNet [0 , 1]            | 28.24       | 37.37      | 32.1       | 64.8       | 34.3           | 52.2        | 45.8        | 23.5          | 42.1   | 26.4         | 28.9          | 12.8        |
| 340M params                 |             |            |            |            |                |             |             |               |        |              |               |             |
| DeltaNet [0 , 1] 100B       | 24.68       | 31.49      | 33.7       | 70.3       | 45.1           | 51.3        | 50.0        | 26.1          | 46.1   | 35.2         | 28.7          | 11.8        |
| DeltaNet [ − 1 , 1] FineWeb | 24.54       | 31.15      | 34.0       | 69.9       | 44.6           | 51.9        | 50.0        | 24.4          | 45.8   | 37.2         | 33.1          | 6.6         |
| 370M params                 |             |            |            |            |                |             |             |               |        |              |               |             |
| Mamba [0 , 1]               | 24.84       | 24.69      | 35.6       | 70.6       | 48.4           | 51.2        | 53.4        | 24.8          | 47.3   | 21.6         | 27.7          | 2.8         |
| Mamba [ − 1 , 1]            | 25.02       | 24.71      | 36.2       | 70.5       | 47.8           | 53.3        | 54.7        | 26.7          | 48.2   | 20.9         | 24.8          | 2.5         |
| 1.3B params 100B            |             |            |            |            |                |             |             |               |        |              |               |             |
| Transformer++ SlimPajama    | 16.85       | 13.44      | 48.9       | 70.8       | 49.6           | 53.6        | 56.0        | 26.5          | 50.9   | 66.6         | 31.5          | 27.4        |
| Mamba [0 , 1]               | 17.06       | 13.89      | 46.2       | 72.2       | 40.1           | 54.1        | 59.0        | 28.2          | 50.0   | 41.4         | 35.2          | 6.2         |
| GLA [0 , 1]                 | 17.22       | 14.47      | 46.9       | 71.8       | 49.8           | 53.9        | 57.2        | 26.6          | 51.0   | 50.6         | 42.6          | 19.9        |
| DeltaNet [0 , 1]            | 16.87       | 12.21      | 48.9       | 71.2       | 50.2           | 53.6        | 57.2        | 28.3          | 51.6   | 49.5         | 37.4          | 17.2        |
| 1.3B params 100B            |             |            |            |            |                |             |             |               |        |              |               |             |
| DeltaNet [0 , 1] FW         | 18.54       | 14.32      | 43.5       | 73.7       | 56.2           | 56.9        | 58.2        | 29.9          | 53.1   | 49.1         | 35.1          | 8.6         |
| DeltaNet [ − 1 , 1]         | 18.57       | 12.73      | 43.7       | 73.3       | 55.8           | 56.8        | 56.9        | 27.9          | 52.4   | 48.8         | 33.9          | 12.3        |

ure [5](#page-8-1) demonstrate that on coding and math datasets, DeltaNet with an eigenvalue range of [−1, 1] achieves lower perplexity than the baseline with range [0, 1] for both model sizes. For TriviaQA, the perplexity of DeltaNet [−1, 1] is slightly higher. Note, that this is a task relying on memorization, not linked to state-tracking, and hence we do not expect an improvement. On SlimPajama, we also observe slight improvement with our modification. For Mamba instead, our modifications consistently degrades the performance on these tasks (Figure [10](#page-31-1) in the Appendix).

To ensure that our models are comparable with those obtained by [Yang et al.](#page-13-6) [\(2024b\)](#page-13-6), we evaluate them on the same benchmark tasks from lm-harness [\(Gao et al., 2024\)](#page-11-14) in Table [4.](#page-9-0) Note, that we trained on 100B tokens of FineWeb, while [Yang et al.](#page-13-6) [\(2024b\)](#page-13-6) reported results from training on 15B and 100B tokens of SlimPajama. At 340-370M parameters, with the extended range both architectures show enhanced performance in some of the tasks: Mamba in the second subset of tasks (+2.1% average accuracy) and DeltaNet in retrieval tasks (+2% SWDE, +4.4% SQUAD). At 1.3B parameters, extending the eigenvalue range of DeltaNet shows mixed results, suggesting that the increased expressivity may need training beyond 100B tokens to fully unlock the model's capacity.

# 6 CONCLUSION

In this work, we showed the substantial impact of extending the eigenvalue range of state-transition matrices in LRNNs from [0, 1] to [−1, 1]. This modification provably enhances LRNN expressivity in state-tracking tasks, without adding overhead in training or inference. While Mamba successfully solves the parity problem, its diagonal matrix structure limits further gains. In contrast, DeltaNet, thanks to its non-diagonal state transition matrices which enable simultaneous token and channel mixing, excels across a broader spectrum of tasks. Our results underscore the critical role of nondiagonal state-transition matrices in augmenting state-tracking capabilities, highlighting a promising direction for future LRNN advancements.

Limitations and Future work Our modification is not directly compatible with a numerical technique used by some diagonal LRNNs such as Mamba2, GLA and mLSTM. In particular, these models rely on positive state-transition matrices to compute cumulative products in log space, which improves numerical accuracy and potentially training stability (see Appendix [E.4](#page-30-0) for details). Further research is needed to assess the impact of training large-scale language models with state-tracking capabilities. To this end, we aim to understand the potential downsides of increased expressivity. For example, we hypothesize a fundamental trade-off between state-tracking and associative recall, which is also of theoretical interest and could guide hybrid model design. Moreover, the theoretical expressivity of DeltaNet [−1, 1] with multiple layers is still unclear. We showed that it can solve addition modulo m (in Appendix [D\)](#page-25-0) which is equivalent to the <sup>Z</sup><sup>3</sup> group word problem, but we do not know if it can also solve other word problems, such as the ones for the symmetric groups S<sup>n</sup> with n ≥ 3.

# ACKNOWLEDGMENTS

We would like to thank David Salinas, Herilalaina Rakotoarison, Eric Alcaide, Arya Akhavan, Matia Bojovic, Erfan Mirzaei and the active members of the Flash Linear Attention discord channel for their constructive discussions and feedback. We acknowledge the support and assistance of the Data Science and Computation Facility and its Support Team, in particular Mattia Pini, in utilizing the IIT High-Performance Computing Infrastructure, on which we run our largest experiments. This research was partially supported by the following sources: PNRR MUR Project PE000013 CUP J53C22003010006 "Future Artificial Intelligence Research (FAIR)", funded by the European Union – NextGenerationEU, and EU Project ELSA under grant agreement No. 101070617. TAILOR, a project funded by EU Horizon 2020 research and innovation programme under GA No 952215; the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under grant number 417962828; the European Research Council (ERC) Consolidator Grant "Deep Learning 2.0" (grant no. 101045765). Frank Hutter acknowledges financial support by the Hector Foundation. The authors acknowledge support from ELLIS and ELIZA. Funded by the European Union. The authors gratefully acknowledge the Gauss Center for Supercomputing eV (<www.gauss-centre.eu>) for funding this project by providing computing time on the GCS supercomputer JUWELS at Julich ¨ Supercomputing Center (JSC). The MATH-HARD dataset which we use in one of our experiments was compiled from AoPS & the AoPS Community, MATHCOUNTS, the MAA, the Centre for Education in Mathematics and Computing, the Harvard-MIT Math Tournament, the Math Prize for Girls, MOEMS, the Mandelbrot Competition, and the Institute of Mathematics and Applications. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the ERC. Neither the European Union nor the ERC can be held responsible for them.

![](_page_10_Picture_3.jpeg)

# REFERENCES


[1] Simran Arora, Brandon Yang, Sabri Eyuboglu, Avanika Narayan, Andrew Hojel, Immanuel Trummer, and Christopher Re. Language Models Enable Simple Systems for Generating Structured ´ Views of Heterogeneous Data Lakes. *Proceedings of the VLDB Endowment*, 17(2):92–105, 2023. Maximilian Beck, Korbinian Poppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, ¨ Michael Kopp, Gunter Klambauer, Johannes Brandstetter, and Sepp Hochreiter. xLSTM: Ex- ¨ tended Long Short-Term Memory. In *Advances in Neural Information Processing Systems*. Curran Associates, Inc., 2024. Satwik Bhattamishra, Kabir Ahuja, and Navin Goyal. On the ability and limitations of transformers to recognize formal languages. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 7096–7116, 2020. Satwik Bhattamishra, Michael Hahn, Phil Blunsom, and Varun Kanade. Separations in the representational capabilities of transformers and recurrent architectures. *Advances in Neural Information Processing Systems*, 36, 2024. Yonatan Bisk, Rowan Zellers, Ronan Le bras, Jianfeng Gao, and Yejin Choi. PIQA: Reasoning about physical commonsense in natural language. *Proceedings of the AAAI Conference on Artificial Intelligence*, 34(05):7432–7439, Apr. 2020. Nicola Muca Cirone, Antonio Orvieto, Benjamin Walker, Cristopher Salvi, and Terry Lyons. Theoretical foundations of deep selective state-space models. *Advances in Neural Information Processing Systems*, 37:127226–127272, 2025. Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? Try arc, the ai2 reasoning challenge. *arXiv preprint arXiv:1803.05457*, 2018.

[2] Tri Dao and Albert Gu. Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. In *International Conference on Machine Learning*. PMLR, 2024. Gregoire Deletang, Anian Ruoss, Jordi Grau-Moya, Tim Genewein, Li Kevin Wenliang, Elliot Catt, Chris Cundy, Marcus Hutter, Shane Legg, Joel Veness, et al. Neural Networks and the Chomsky Hierarchy. In *The Eleventh International Conference on Learning Representations*, 2023. Ting-Han Fan, Ta-Chung Chi, and Alexander Rudnicky. Advancing Regular Language Reasoning in Linear Recurrent Neural Networks. In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers)*, pp. 45–53, 2024. Daniel Y Fu, Tri Dao, Khaled Kamal Saab, Armin W Thomas, Atri Rudra, and Christopher Re. Hungry Hungry Hippos: Towards Language Modeling with State Space Models. In *The Eleventh International Conference on Learning Representations*, 2021. Jean Gallier and Jean Gallier. The Cartan–Dieudonne Theorem. ´ *Geometric Methods and Applications: For Computer Science and Engineering*, pp. 231–280, 2011. Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 07 2024. Angeliki Giannou, Shashank Rajput, Jy-yong Sohn, Kangwook Lee, Jason D Lee, and Dimitris Papailiopoulos. Looped transformers as programmable computers. In *International Conference on Machine Learning*, pp. 11398–11442. PMLR, 2023. Xavier Gonzalez, Andrew Warrington, Jimmy T.H. Smith, and Scott Linderman. Towards Scalable and Stable Parallelization of Nonlinear RNNs. In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024. Albert Gu and Tri Dao. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. In *First Conference on Language Modeling*, 2024. Albert Gu, Isys Johnson, Karan Goel, Khaled Saab, Tri Dao, Atri Rudra, and Christopher Re. Com- ´ bining recurrent, convolutional, and continuous-time models with linear state space layers. *Advances in neural information processing systems*, 34:572–585, 2021. Albert Gu, Karan Goel, and Christopher Re. Efficiently Modeling Long Sequences with Structured State Spaces. In *International Conference on Learning Representations*, 2022. Sylvain Gugger, Lysandre Debut, Thomas Wolf, Philipp Schmid, Zachary Mueller, Sourab Mangrulkar, Marc Sun, and Benjamin Bossan. Accelerate: Training and inference at scale made simple, efficient and adaptable. <https://github.com/huggingface/accelerate>, 2022. Michael Hahn. Theoretical limitations of self-attention in neural sequence models. *Transactions of the Association for Computational Linguistics*, 8:156–171, 2020. Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. In *Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)*, 2021. Sepp Hochreiter and Jurgen Schmidhuber. Long Short-Term Memory. ¨ *Neural Computation*, 9(8): 1735–1780, 1997. John Hopcroft and Jeffrey Ullman. *Introduction to Automata Theory, Languages, and Computation*. Addison-Wesley, 2001. Roger A Horn and Charles R Johnson. *Matrix Analysis*. Cambridge University Press, 2012.

[3] Kazuki Irie, Imanol Schlag, Robert Csord ´ as, and J ´ urgen Schmidhuber. Going beyond linear trans- ¨ formers with recurrent fast weight programmers. *Advances in neural information processing systems*, 34:7703–7717, 2021. Kazuki Irie, Robert Csord ´ as, and J ´ urgen Schmidhuber. Practical computational power of linear ¨ transformers and their recurrent and self-referential extensions. *arXiv preprint arXiv:2310.16076*, 2023. Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 1601–1611, 2017. Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franc¸ois Fleuret. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. In *International Conference on Machine Learning*, pp. 5156–5165. PMLR, 2020. Kenneth Krohn and John Rhodes. Algebraic theory of machines. i. prime decomposition theorem for finite semigroups and machines. *Transactions of the American Mathematical Society*, 116: 450–464, 1965. Yi Heng Lim, Qi Zhu, Joshua Selfridge, and Muhammad Firmansyah Kasim. Parallelizing nonlinear sequential models over the sequence length. In *The Twelfth International Conference on Learning Representations*, 2024. Bingbin Liu, Jordan T Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang. Transformers Learn Shortcuts to Automata. In *The Eleventh International Conference on Learning Representations*, 2023. Colin Lockard, Prashant Shiralkar, and Xin Luna Dong. When open information extraction meets the semi-structured web. *NAACL-HLT. Association for Computational Linguistics*, 2019. Ilya Loshchilov and Frank Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts. In *International Conference on Learning Representations*, 2017. Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. In *International Conference on Learning Representations*, 2019. Oded Maler and Amir Pnueli. On the cascaded decomposition of automata, its complexity and its application to logic. *ACTS Mobile Communication*, 48, 1994. William Merrill and Ashish Sabharwal. The parallelism tradeoff: Limitations of log-precision transformers. *Transactions of the Association for Computational Linguistics*, 11:531–545, 2023. William Merrill, Gail Weiss, Yoav Goldberg, Roy Schwartz, Noah A Smith, and Eran Yahav. A Formal Hierarchy of RNN Architectures. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pp. 443–459, 2020. William Merrill, Jackson Petty, and Ashish Sabharwal. The Illusion of State in State-Space Models. In *Forty-first International Conference on Machine Learning*, 2024. Antonio Orvieto, Soham De, Caglar Gulcehre, Razvan Pascanu, and Samuel L Smith. Universality of Linear Recurrences Followed by Non-linear Projections: Finite-Width Guarantees and Benefits of Complex Eigenvalues. In *Forty-first International Conference on Machine Learning*, 2024. Denis Paperno, German Kruszewski, Angeliki Lazaridou, Ngoc-Quan Pham, Raffaella Bernardi, ´ Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernandez. The LAMBADA dataset: ´ Word prediction requiring a broad discourse context. In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 1525–1534, 2016. Guilherme Penedo, Hynek Kydl´ıcek, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin ˇ Raffel, Leandro Von Werra, and Thomas Wolf. The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale, 2024.

[4] Bo Peng, Eric Alcaide, Quentin Gregory Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Nguyen Chung, Leon Derczynski, et al. RWKV: Reinventing RNNs for the Transformer Era. In *The 2023 Conference on Empirical Methods in Natural Language Processing*, 2023. Jorge Perez, Pablo Barcel ´ o, and Javier Marinkovic. Attention is turing-complete. ´ *Journal of Machine Learning Research*, 22(75):1–35, 2021. Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In *SC20: International Conference for High Performance Computing, Networking, Storage and Analysis*, pp. 1–16. IEEE, 2020. Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don't know: Unanswerable questions for squad. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, pp. 784–789, 2018. Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. *Communications of the ACM*, 64(9):99–106, 2021. Yash Sarrof, Yana Veitsman, and Michael Hahn. The Expressive Capacity of State Space Models: A Formal Language Perspective. *Advances in Neural Information Processing Systems*, 2024. Imanol Schlag, Kazuki Irie, and Jurgen Schmidhuber. Linear transformers are secretly fast weight ¨ programmers. In *International Conference on Machine Learning*, pp. 9355–9366. PMLR, 2021. Jurgen Schmidhuber. Learning to control fast-weight memories: An alternative to dynamic recurrent ¨ networks. *Neural Computation*, 4(1):131–139, 1992. Daria Soboleva, Faisal Al-Khateeb, Robert Myers, Jacob R Steeves, Joel Hestness, and Nolan Dey. SlimPajama: A 627B token cleaned and deduplicated version of RedPajama, June 2023. Lena Strobl, William Merrill, Gail Weiss, David Chiang, and Dana Angluin. What Formal Languages can Transformers express? A Survey. *Transactions of the Association for Computational Linguistics*, 12:543–561, 2024. Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, et al. Learning to (learn at test time): RNNs with expressive hidden states. *arXiv preprint arXiv:2407.04620*, 2024. Matteo Tiezzi, Michele Casoni, Alessandro Betti, Marco Gori, and Stefano Melacci. State-Space Modeling in Long Sequence Processing: A Survey on Recurrence in the Transformer Era, 2024. Alexandre Torres. mamba.py: A simple, hackable and efficient Mamba implementation in pure PyTorch and MLX., 2024. URL <https://github.com/alxndrTL/mamba.py>. Lewis Tunstall, Leandro Von Werra, and Thomas Wolf. *Natural Language Processing with Transformers*. O'Reilly Media, Inc., 2022. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is All you Need. In *Advances in Neural Information Processing Systems*, volume 30. Curran Associates, Inc., 2017. Songlin Yang and Yu Zhang. FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanism, January 2024. URL [https://github.com/](https://github.com/sustcsonglin/flash-linear-attention) [sustcsonglin/flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention). Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated Linear Attention Transformers with Hardware-Efficient Training. In *Forty-first International Conference on Machine Learning*, 2024a. Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing Linear Transformers with the Delta Rule over Sequence Length. *Advances in Neural Information Processing Systems*, 36, 2024b.

[5] Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated Delta Networks: Improving Mamba2 with Delta Rule. In *The Thirteenth International Conference on Learning Representations*, 2025.

[6] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a Machine Really Finish Your Sentence? In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pp. 4791–4800, 2019.
# SUPPLEMENTARY MATERIAL

The supplementary material is structured as follows.

- Appendix [A](#page-15-1) contains additional details on the notation used, on Table [1,](#page-2-0) on the relationship between RNNs and regular languages, on the assumption of finite precision, on the states, and on the function dec.
- Appendices [B](#page-17-0) and [C](#page-21-1) contain the proofs for the theoretical results in Sections [4.1](#page-3-2) and [4.3.](#page-5-5)
- Appendix [D](#page-25-0) contains a theorem showing that a 2 Layer LRNN having reflections as statetransition matrices can solve addition modulo m.
- Appendix [E](#page-26-1) contains additional details on the experiments and additonal results.

### A ADDITIONAL BACKGROUND

### A.1 NOTATION

We denote with C, R, N the sets of complex, real, and natural numbers, respectively. We use lowercase letters for scalar quantities (e.g. x ∈ R), bold lowercase letters for (column) vectors (e.g. v ∈ R <sup>n</sup>), and bold uppercase letters for matrices (e.g. M ∈ <sup>R</sup> n×d ). Some functions with matrix (vector) outputs, such as A and B in [\(1\)](#page-2-1), are also bold upper (lower) case letters to emphasize the fact that they output matrices (vectors). We use ⊙ to indicate the element-wise (Hadamard) product between two vectors or matrices. We denote with ∥v∥ the Euclidean norm of the vector v ∈ <sup>R</sup> n. When M ∈ R n×d , ∥M∥ also refers to the Euclidean norm, corresponding to the largest singular value. The vector e<sup>i</sup> ∈ <sup>R</sup> <sup>n</sup> is the i-th vector of the canonical bases in <sup>R</sup> <sup>n</sup>, i.e. the one-hot vector with 1 only in the i-th component and 0 in the others. We define the binomial coefficient for every k, j ∈ N with j ≤ k as

$$\binom{k}{0} := 1, \quad \binom{k}{j} := \frac{k(k-1) \dots (k-j+1)}{j!}.$$

We also define for a Boolean s and x ∈ R

$$\mathbf{1}\{s\} := \begin{cases} 1 & \text{if } s \text{ is true} \\ 0 & \text{if } s \text{ is false} \end{cases} , \quad \text{sign}(x) := \begin{cases} 1 & \text{if } x \geq 0 \\ -1 & \text{if } x < 0 \end{cases}.$$

We define sigmoid(x) := 1/(1 + e −x ) and softplus(x) := ln(1 + e x ).

We sometimes use regular expressions (see e.g. [Hopcroft & Ullman, 2001\)](#page-11-11), to represent their corresponding regular language. So that e.g. (11)<sup>∗</sup> = {11} ∗ , where {11} is the set containing the word 11 and ∗ is the *Kleene star* operation, is the language containing the empty word ϵ and all the words with an even number of ones, while (1<sup>m</sup>) <sup>∗</sup> = {1 <sup>m</sup>} ∗ is the language containing the words with a number of ones divisible by m since 1 <sup>m</sup> indicates the word containing 1 repeated m times. A language is *star-free* if it can be expressed with a regular expression that does not contain the Kleene star.

### A.2 DETAILS OF TABLE [1](#page-2-0)

The Mamba recurrence in Equations 3 and 4 in [\(Gu & Dao, 2024\)](#page-11-0) is applied independently to each channel of the input sequence. Expressing the full recurrence in the matrix-form of [\(1\)](#page-2-1) is challenging, as it would require concatenating the rows of the matrix Ht. For simplicity, in Table [1](#page-2-0) we write instead the recurrence for each row of Ht. In particular, Let x<sup>t</sup> ∈ <sup>R</sup> <sup>d</sup> be the input of the layer, W<sup>∆</sup> ∈ <sup>R</sup> d×d , w<sup>2</sup> ∈ <sup>R</sup> d , W<sup>1</sup> = (w1, . . . , wn) <sup>⊤</sup> ∈ <sup>R</sup> <sup>n</sup>×<sup>d</sup> be learnable parameters, q<sup>t</sup> ∈ R <sup>n</sup>, k<sup>t</sup> = (kt,1, . . . , kt,n) <sup>⊤</sup> ∈ <sup>R</sup> <sup>n</sup> be learnable functions of the input and ∆<sup>t</sup> = softplus(W∆xt). Then, if we set H<sup>t</sup> = (ht,1, . . . , ht,n) <sup>⊤</sup> ∈ <sup>R</sup> n×d and H<sup>0</sup> = 0, we can write the recurrence for the i-th row of H<sup>t</sup> and the output as

$$h_{t,i} = \mathbf{A}_i(\mathbf{x}_t)\mathbf{h}_{t-1,i} + \mathbf{B}_i(\mathbf{x}_t), \quad \hat{\mathbf{y}}_t = \psi(\mathbf{H}_t^\top \mathbf{q}_t + \mathbf{w}_2 \odot \mathbf{x}_t))$$

where Ai(xt) and Bi(xt) are the matrices stated in Table [1,](#page-2-0) i.e.

$$\mathbf{A}_i(\mathbf{x}_t) := \text{Diag}(\exp(-\mathbf{\Delta}_t \odot \exp(\mathbf{w}_{1,i}))) \in \mathbb{R}^{d \times d}, \quad \mathbf{B}_i(\mathbf{x}_t) := k_{t,i} \mathbf{\Delta}_t \odot \mathbf{x}_t \in \mathbb{R}^d.$$

Alternatively, as done in [\(Yang et al., 2024b,](#page-13-6) Table 4), one could write the full matrix recurrence as:

$$H_t = \underbrace{\exp\left(-1\Delta_t^\top \odot \exp(W_1)\right)}_{A(\mathbf{x}_t)} \odot H_{t-1} + \underbrace{\mathbf{k}_t(\Delta_t \odot \mathbf{x}_t)^\top}_{B(\mathbf{x}_t)}.$$

where 1 is the vector of n ones. However, such a recurrence is not in the form [\(1\)](#page-2-1), since we have replaced the matrix-matrix product A(xt)H<sup>t</sup> with the element-wise product A(xt)⊙Ht. Note that we follow the implementation of B(xt) used in the official Mamba codebase, which simplifies the expression originally presented in Equation 4 of [\(Gu & Dao, 2024\)](#page-11-0) as described by the authors in a GitHub Issue[<sup>3</sup>](#page-16-1) .

### A.3 REGULAR LANGUAGES AND RECURRENT NEURAL NETWORKS

RNNs Can Recognize Any Regular Language A layer of a general RNN can be formulated similarly to [\(1\)](#page-2-1) just by replacing the linear state update with a generic state-transition function g as:

$$\mathbf{h}_t = g(\mathbf{h}_{t-1}, \mathbf{x}_t), \quad \mathbf{h}_0 \in \mathbb{R}^n.$$

Clearly, any FSA can be implemented by an RNN layer if g is sufficiently expressive to model its state transition function.

LRNNs Can Recognize Any Regular Language As explained in [\(Liu et al., 2023,](#page-12-4) Appendix A.2) and in the proof of [\(Merrill et al., 2024,](#page-12-0) Theorem 5), we can implement any FSA A = (Σ, Q, q0, δ), and thus recognize any regular language, using matrix-vector multiplication. As a result, a single-layer LRNN by using one-hot vectors as the LRNN states and having boolean state transition matrices can recognize any language. More specifically, in [\(1\)](#page-2-1), we can set n = |Q|, H<sup>0</sup> = (1, 0 . . . , 0)<sup>⊤</sup> and for any letter w ∈ Σ, B(w) = 0 and A(w) ∈ <sup>R</sup> <sup>n</sup>×<sup>n</sup> being the matrix with entries A(w)<sup>q</sup> ′ ,q = 1{δ(w, q) = q ′}. Note that in such a construction, the matrix A(w) can have norm greater than one, and enabling the state-transition matrix of LRNNs to have norm greater than one can make the recurrence unstable and is therefore never done in language models (see e.g. Table [1\)](#page-2-0).

### A.4 FINITE PRECISION

For our positive results on LRNNs expressivity (Theorems [3](#page-5-0) and [4\)](#page-6-0), by finite precision we mean that since we have a finite number of quantities involved in the computations, then there exists a finite set D ⊂ R that contains them and thus we do not require computations to be done in the reals but we can use D as datatype. In particular, D does not depend on the length of the input sequence. In practice, such data type is chosen beforehand, e.g. floating point numbers requiring a given number of bits of precision, which may not capture all quantities in our constructions.

In our negative results of Theorems [1](#page-4-0) and [2](#page-4-1) instead, we can pick the finite set D ⊂ R arbitrarily, e.g. floating point numbers, and we also make the use of the function cast : R → D, defined in [\(4\)](#page-17-1). that we extend to C by applying it separately to the real and imaginary part and to vector and matrices by applying it element-wise. The cast function is used because some computations of the state of the LRNN will be allowed to be in infinite precision and then transformed to finite precision using cast as specified in the proofs. This function provides a simplification of the actual conversion that happens in practice.

We believe that the finite precision setup is not only realistic but also allows a better focus on the drawbacks of modern LRNN. Note that for Transformers, results usually rely instead on the weaker notion of log-precision [\(Liu et al., 2023\)](#page-12-4), meaning that the size of D grows logarithmically with the sequence length. This is mainly due to their limited expressivity compared to LRNNs. We also note that concerning the state-transition matrices of modern LRNNs (see Table [1\)](#page-2-0), the values at the extremes of the eigenvalue range are technically not included (because of the use of the sigmoid and softplus functions). However, since we are working with finite precision, we can still include them by choosing the appropriate datatype D, which in practice includes key values such as 0, 1, and −1.

<sup>3</sup><https://github.com/state-spaces/mamba/issues/19>

#### A.4.1 INITIAL STATE, MATRIX-VALUED STATES, AND THE DECODER FUNCTION

When introducing the LRNN layer in [\(1\)](#page-2-1), we mention that A, B and dec are learnable functions. However, to learn the constructions in our theoretical results, we need also H<sup>0</sup> ⊆ <sup>C</sup> n×d to be learnable. We do this only to simplify the results, since the same effect can also be achieved by using a special token \$ at the beginning of each sequence input to the model, called the beginning of sequence token and setting, H<sup>0</sup> = 0 for each LRNN layer so that B(x1) will have the same role as the learnable H<sup>0</sup> in our constructions. This practice is standard and used in all our experiments.

While we mention that the states H<sup>t</sup> are generally matrices of dimension n × d, for our theoretical constructions (excluding the first two theorems), we set d = 1, so that states are vector-valued. Hence, for the problems that we consider, we find that having a matrix-valued state (d > 1) brings no theoretical advantage, while it is very important for associative recall.

To compute the output yˆ<sup>t</sup> from the state H<sup>t</sup> and the vector x<sup>t</sup> of an LRNN layer in [\(1\)](#page-2-1), we use the function dec, to abstract away the computations that are done on H<sup>t</sup> and xt, since they are not part of the recurrence. In this work, we do not consider the internal structure of dec, but it usually contains a normalization and a feed-forward neural network and it can approximate any continuous function.

In our negative results on LRNNs expressivity in Theorems [1](#page-4-0) and [2,](#page-4-1) our choice of an arbitrary decoder guarantees the stronger results. For our positive results instead, we either do not consider the decoder (Theorem [3\)](#page-5-0) or we make use of a linear decoder (Theorem [4\)](#page-6-0). We point out that to recognize regular languages efficiently and with a smaller LRNN state it is beneficial to have a more powerful (non-linear) decoder, as in the case of word problems for cyclic or permutation groups. However, such a decoder may be hard to learn.

# B PARITY AND MODULAR COUNTING – PROOFS

We report the proofs for the theorems in Section [4.1.](#page-3-2) We start by defining the function cast : R → D, for a finite set D ⊂ R, which provides a simple model for the conversion of real numbers into a finite precision representation.

$$\text{cast}(x) = \min_{z \in \mathcal{D}_{\min}} z, \quad \mathcal{D}_{\min} := \arg \min_{z \in \mathbb{D}} |z - x|. \quad (4)$$

Note that Dmin might not be a singleton. We naturally extend this function on complex numbers by applying it separately to the real and imaginary part, and then to complex-valued matrices by applying it element-wise. The following lemma is a key element of the proofs of Theorems [1](#page-4-0) and [2.](#page-4-1) There, the sequence a<sup>k</sup> in the lemma takes the form of the imaginary or real part of the elements of the k-th power of a matrix with real eigenvalues (λ<sup>i</sup> will be one eigenvalue), expressed using the Jordan canonical form. See Appendix [B.1](#page-19-0) for more details on the Jordan Canonical Form. Intuitively, the lemma shows that if some of the λi-s are negative then for k large enough, a<sup>k</sup> in finite precision will alternate between two values. Instead, if the λi-s are only nonnegative, a<sup>k</sup> in finite precision becomes constant for large enough k.

Lemma 1. *Let* n, m¯ ∈ N *and for every* k > m¯ *let*

$$a_k := \sum_{i=1}^n c_i \binom{k}{m_i} \lambda_i^{k-m_i}, \quad \text{with } c_i, \lambda_i \in \mathbb{R}, m_i \in \mathbb{N}, m_i \leq \bar{m}, \quad \forall i \in \{1, \dots, n\},$$

*then there exist* ¯k ∈ <sup>N</sup> *such that for every* k ≥ ¯k *there exist* a¯1, a¯<sup>2</sup> ∈ <sup>D</sup> *such that*

$$\text{cast}(a_{2k}) = \bar{a}_1, \quad \text{cast}(a_{2k+1}) = \bar{a}_2.$$

*Furthermore, if* λ<sup>i</sup> ≥ 0 *for every* i ∈ {1, . . . , n}*, then* cast(ak) = ¯a<sup>1</sup> = ¯a<sup>2</sup> *for* k ≥ ¯k*.*

*Proof.* If c<sup>i</sup> = 0 for every i, or λ<sup>i</sup> = 0 for every i, then a<sup>k</sup> = 0 for all k and the statement is trivially satisfied. Without loss of generality we can assume that that c<sup>i</sup> ̸= 0 and λ<sup>i</sup> ̸= 0 for every i ∈ {1, . . . , n}, since for each i where this is not true we can remove the corresponding term in the sum (since it will be 0) and use smaller value for n. We divide the proof into two parts.

Positive powers: Assume that λ<sup>i</sup> > 0 for all i ∈ {1, . . . , n}. This yields that for every i and every k>m¯ , k m<sup>i</sup> λ k−m<sup>i</sup> <sup>i</sup> > 0. Since the cast function is piecewise constant with a finite number of pieces,

we can divide the real line into a finite number of intervals where cast is constant. We now show that for k large enough, the interval where a<sup>k</sup> belongs, and hence cast(ak), does not vary with k.

Without loss of generality we assume that for every i, j ∈ {1, . . . , n} we have that (m<sup>i</sup> , λi) ̸= (m<sup>j</sup> , λ<sup>j</sup> ), since otherwise we can factor out k m<sup>i</sup> λ k−m<sup>i</sup> i and use a smaller n. Note that k m<sup>i</sup> λ k−m<sup>i</sup> <sup>i</sup> = k(k−1)···(k−mi+1) mi! λ k−m<sup>i</sup> i and hence gi(k) = k m<sup>i</sup> λ k−m<sup>i</sup> i for large k behaves like the function k <sup>m</sup>iλ k , i.e. the product of a polynomial and an exponential function of k. Without loss of generality, we therefore take the order of the indices of the terms in the sum such that the functions g<sup>i</sup> are in decreasing order of growth:

$$\lambda_i > \lambda_j \text{ or } \lambda_i = \lambda_j, m_i > m_j \quad \forall i, j : i > j.$$

By factoring out g1(k), i.e. the fastest growing term, from a<sup>k</sup> we get

$$a_k = \binom{k}{m_1} \lambda_1^{k-m_1} (c_1 + b_k) \quad b_k := \sum_{i=2}^n c_i \frac{\binom{k}{m_i} \lambda_i^{k-m_i}}{\binom{k}{m_1} \lambda_1^{k-m_1}},$$

with limk→∞ b<sup>k</sup> = 0 and therefore, since for every i and every k > m¯ , k m<sup>i</sup> λ k−m<sup>i</sup> <sup>i</sup> > 0 and c<sup>1</sup> ̸= 0, there exist ˆk ∈ <sup>N</sup> such that for every k ≥ ˆk, sign(ak) = sign(c<sup>1</sup> + bk) = sign(c1). Now let <sup>D</sup> = {z1, . . . , zd} with z<sup>1</sup> < z<sup>2</sup> < · · · < z<sup>d</sup> and let y<sup>1</sup> = −∞, yd+1 = ∞ and y<sup>i</sup> = (zi−<sup>1</sup> + zi)/2 for i ∈ {2, . . . , d}. From its definition, cast is a piecewise constant function such that cast(x) = z<sup>i</sup> for every x ∈ (y<sup>i</sup> , yi+1). We now consider three cases according to the values of λ<sup>1</sup> and m1.

1) If λ<sup>1</sup> > 1 or λ<sup>1</sup> = 1, m<sup>1</sup> > 0, then limk→∞ k m<sup>1</sup> λ k−m<sup>i</sup> <sup>1</sup> <sup>=</sup> <sup>∞</sup> and there exists ¯<sup>k</sup> <sup>≥</sup> <sup>ˆ</sup><sup>k</sup> such that for every k ≥ ¯k, either a<sup>k</sup> > y<sup>d</sup> (if sign(c1) = 1) or a<sup>k</sup> < y<sup>2</sup> (if sign(c1) = −1) and hence cast(ak) = ¯a ∈ {z1, zd}.

2) If λ<sup>1</sup> < 1 then limk→∞ k m<sup>1</sup> λ k−m<sup>i</sup> <sup>1</sup> = 0 and hence there exist ϵ > 0, j ∈ {1, . . . , d}, ¯k > ˆk such that for every k ≥ ¯k, a<sup>k</sup> ∈ Ω ⊆ (y<sup>j</sup> , yj+1), where Ω = (0, ϵ) if sign(c1) = 1 and Ω = (−ϵ, 0) if sign(c1) = −1. Therefore, cast(ak) = z<sup>j</sup> for every k ≥ ¯k.

3) If λ<sup>1</sup> = 1, m<sup>1</sup> = 0, then k m<sup>1</sup> λ k−m<sup>i</sup> <sup>1</sup> = 1 for every k and hence

$$a_k = c_1 + b_k, \quad b_k = \sum_{i=2}^n c_i \binom{k}{m_i} \lambda_i^{k-m_i} \quad \text{with } \lambda_i < 1 \forall i \in \{2, \dots, n\}$$

Note that b<sup>k</sup> has now the same structure as ak, just with one less term in the sum, therefore we can factor out the term λ<sup>2</sup> m<sup>2</sup> λ <sup>k</sup>−m<sup>2</sup> and, since λ<sup>2</sup> < 1, apply the same reasoning as for the second case (λ<sup>1</sup> < 1) to c<sup>1</sup> + b<sup>k</sup> and prove that there exist ϵ > 0, j ∈ {1, . . . , d}, ¯k > ˆk such that for every k ≥ ¯k, we have that sign(bk) = sign(c2), a<sup>k</sup> ∈ Ω ⊆ (y<sup>j</sup> , yj+1), where Ω = (c1, ϵ) if sign(c2) = 1 and Ω = (−ϵ, c1) if sign(c2) = −1. Therefore cast(ak) = z<sup>j</sup> for every k ≥ ¯k.

In summary, we proved that when λ<sup>i</sup> ≥ 0 for every i, there exist a¯ ∈ <sup>D</sup>, ¯k ∈ <sup>N</sup> such that for every k ≥ ¯k a<sup>k</sup> = ¯a, which concludes the first part of the proof.

Some powers can be negative: Consider the general case where λ<sup>i</sup> ∈ <sup>R</sup> can be negative. We can write

$$a_k = \sum_{i=1}^n c_i \binom{k}{m_i} \text{sign}(\lambda_i)^{k-m_i} |\lambda_i|^{k-m_i}.$$

Since sign(x) <sup>2</sup>k−m<sup>i</sup> and sign(x) <sup>2</sup>k+1−m<sup>i</sup> do not vary with k we consider the two subsequences

$$\begin{aligned} a_{2k} &= \sum_{i=1}^n \hat{c}_i \binom{2k}{m_i} |\lambda_i|^{2k-m_i}, \quad \hat{c}_i = c_i \text{sign}(\lambda_i)^{2k-m_i} \\ a_{2k+1} &= \sum_{i=1}^n \tilde{c}_i \binom{2k+1}{m_i} |\lambda_i|^{2k+1-m_i}, \quad \tilde{c}_i = c_i \text{sign}(\lambda_i)^{2k+1-m_i}, \end{aligned}$$

and we can apply the same proof as for the case when λ<sup>i</sup> > 0 for every i to each of the subsequences above, which gives the final result in the case λ<sup>i</sup> ∈ <sup>R</sup> for every i.

# B.1 PROOF OF THEOREM [1](#page-4-0)

The language (11)<sup>∗</sup> contains all sequences with an even number of ones. An FSA recognizing the language, for the sequence 1 <sup>k</sup> will output y<sup>k</sup> = 1 if k is even and y<sup>k</sup> = 0 if k is odd. Consider an LRNN with one layer as in [\(1\)](#page-2-1). We will prove that if A(1) has only nonnegative eigenvalues, then there exists a k > 0 such that for every k ≥ k, the finite precision version of the state H<sup>k</sup> corresponding to the sequence 1 <sup>k</sup> does not depend on k and is equal to H. Hence, no matter the choice of dec, also the finite precision version of yˆ<sup>k</sup> will not vary with k and thus for some k ′ ≥ ¯k, yˆk′ ̸= k ′ mod 2 = yk′ . An inductive argument can then be used for the case of LRNNs with multiple (finitely many) layers, using the fact that the input of the next layer will be constant for k large enough, as the input of the first layers.

By unrolling the recursion in [1](#page-2-1) we obtain a closed-form expression for the state

$$H_k = \sum_{i=1}^{k-1} \left( \prod_{j=i+1}^{k-1} A(\mathbf{x}_j) \right) B(\mathbf{x}_i) + \left( \prod_{i=1}^k A(\mathbf{x}_i) \right) H_0,$$

where we set Q<sup>k</sup>−<sup>1</sup> <sup>j</sup>=<sup>k</sup> A(x<sup>j</sup> ) = I to avoid clutter. We follow [Merrill et al.](#page-12-0) [\(2024\)](#page-12-0) and make the simplifying assumption that in finite precision the state at time k is computed by first evaluating all products involving the matrices A(x<sup>j</sup> ) separately and in infinite precision, followed by casting them into finite precision, and finally executing the sum also in infinite precision and casting the result in finite precision. This avoids having to deal with the individual matrix sums and products in finite precision, which would break associativity and be harder to analyze. Hence, if we set x<sup>1</sup> . . . x<sup>k</sup> = 1<sup>k</sup> , we get the following exact and finite precision expressions for the state at time k.

$$H_k = \sum_{i=0}^{k-1} A(1)^i B(1) + A(1)^k H_0, \quad \widehat{H}_k = \text{cast} \left( \sum_{i=0}^{k-1} \text{cast} \left( A(1)^i B(1) \right) + \text{cast} \left( A(1)^k H_0 \right) \right),$$

where cast, defined in [\(4\)](#page-17-1), is an operation that converts matrices with complex values element-wise into finite precision by e.g. separately converting real and imaginary parts.

Using the Jordan canonical form theorem (see e.g. [Horn & Johnson, 2012,](#page-11-15) Chap. 3.1), we can write A(1) = P JP <sup>−</sup><sup>1</sup> , where J is block diagonal made of the Jordan blocks J1, . . . , J<sup>s</sup> with s ≤ n, J<sup>i</sup> ∈ <sup>R</sup> <sup>k</sup>i×k<sup>i</sup> and with corresponding complex eigenvalues λ<sup>1</sup> . . . λ<sup>s</sup> (with multiplicity taken into account). Such decomposition is useful because it allows, for k ≥ max<sup>i</sup> k<sup>i</sup> − 1, to write

$$A(1)^k = PJ^k P^{-1}, \quad J_i^k = \begin{bmatrix} \lambda_i^k & (k)\lambda_i^{k-1} & (k)\lambda_i^{k-2} & \cdots & \cdots & (k)\lambda_{i-1}^{k-1}\lambda_i^k & (k)\lambda_{i-2}^{k-1}\lambda_i^k \\ & \lambda_i^k & (k)\lambda_i^{k-1} & \cdots & \cdots & (k)\lambda_{i-2}^{k-1}\lambda_i^k & (k)\lambda_{i-2}^{k-1}\lambda_i^k \\ & & & \ddots & \ddots & \vdots & \vdots \\ & & & & \ddots & \ddots & \vdots \\ & & & & & \lambda_i^k & (k)\lambda_i^{k-1} \\ & & & & & & \lambda_i^k \end{bmatrix}.$$

Then, from the structure of the Jordan decomposition, the imaginary and real part of each element of the matrices A(1)<sup>k</sup>B(1) and A(1)<sup>k</sup>H<sup>0</sup> will be a linear combination of elements of the Jordan blocks taking the same form of a<sup>k</sup> in Lemma [1.](#page-17-2) Therefore since λ<sup>i</sup> ≥ 0 for every i, we can apply Lemma [1](#page-17-2) component-wise and conclude that there exists <sup>τ</sup> ∈ <sup>N</sup>, <sup>C</sup>b ∈ <sup>C</sup> n×d and <sup>D</sup>b ∈ <sup>C</sup> n×d such that for every <sup>k</sup> ≥ <sup>τ</sup> , <sup>C</sup>b<sup>k</sup> = cast(A(1)<sup>k</sup>B(1)) = <sup>C</sup>b and <sup>D</sup>b <sup>k</sup> = cast(A(1)<sup>k</sup>H0) = <sup>D</sup>b and hence

$$\widehat{H}_k = \text{cast} \left( \sum_{i=0}^{\tau-1} \widehat{C}_i + \widehat{D} + (1 - \tau) \widehat{C} + k \widehat{C} \right).$$

Note that only the matrix <sup>k</sup><sup>C</sup>b varies with <sup>k</sup> and for large enough <sup>k</sup>, the real and imaginary parts of each element of <sup>k</sup><sup>C</sup>b will be either <sup>0</sup>, smaller than minx∈<sup>R</sup> cast(x) or larger than maxx∈<sup>R</sup> cast(x). Therefore, we obtain that there exists H ∈ C n×d and ¯k ≥ τ such that for every k ≥ ¯k we have <sup>H</sup>c<sup>k</sup> <sup>=</sup> <sup>H</sup>, which concludes the proof.

### B.2 PROOF OF THEOREM [2](#page-4-1)

One Layer Let <sup>H</sup>c<sup>k</sup> and <sup>y</sup>ˆ<sup>k</sup> := cast(dec(Hck, xk)) be the finite precision versions of the state <sup>H</sup><sup>k</sup> and (scalar) output of a one-layer LRNN on the input x = x<sup>1</sup> . . . x<sup>k</sup> = 1<sup>k</sup> . Let also y<sup>k</sup> = 1{k mod m = 0} be the correct output recognizing the word x. We will show that if the assumptions on the eigenvalues are not satisfied, i.e. if for any x, every eigenvalue λ of A(x) is real, then there exist H1, H<sup>2</sup> ∈ <sup>C</sup> <sup>n</sup>×<sup>n</sup>, y¯1, y¯<sup>2</sup> ∈ <sup>R</sup> p and τ ∈ N such that for all k ≥ τ

$$\widehat{H}_k := \begin{cases} \overline{H}_1 & \text{if } k \bmod 2 = 0 \\ \overline{H}_2 & \text{otherwise} \end{cases}, \quad \widehat{y}_k = \begin{cases} \bar{y}_1 & \text{if } k \bmod 2 = 0 \\ \bar{y}_2 & \text{otherwise} \end{cases} \quad (5)$$

where without loss of generality we take y¯1, y¯<sup>2</sup> ∈ {0, 1}. If y¯<sup>1</sup> = ¯y2, then, similarly to parity, yˆ<sup>k</sup> = yˆk+1 for all k > τ , while since m > 2, if k mod m = m −1, then 1 = yk+1 ̸= y<sup>k</sup> = 0. Otherwise if y¯<sup>1</sup> ̸= ¯y<sup>2</sup> then if we assume that k mod d = 1 and yˆ<sup>k</sup> = y<sup>k</sup> = 0, then 1 = ˆyk+1 ̸= yk+1 = 0 since m > 2. This will prove the result for a one-layer LRNN. Then, we will proceed with the proof of finitely many layers.

To prove [\(5\)](#page-20-1), we set

$$\widehat{H}_k = \text{cast} \left( \sum_{i=0}^{k-1} \text{cast} \left( \mathbf{A}(1)^i \mathbf{B}(1) \right) + \text{cast} \left( \mathbf{A}(1)^k \mathbf{H}_0 \right) \right),$$

and proceed similarly to Theorem [1.](#page-4-0) Indeed, using the k-th power formula for the Jordan Decomposition of the matrix A(1) with eigenvalues λ1, . . . , λs, the imaginary and real part of each element of the matrices A(1)<sup>k</sup>B(1) and A(1)<sup>k</sup>H<sup>0</sup> will be a linear combination of elements of the Jordan blocks taking the same form of a<sup>k</sup> in Lemma [1.](#page-17-2) Therefore since our assumptions with L = 1 imply that λ<sup>i</sup> ∈ <sup>R</sup> for every i, we can apply Lemma [1](#page-17-2) to show that there exist τ¯ ∈ <sup>N</sup>, C1, C2, D1, D<sup>2</sup> ∈ <sup>C</sup> n×d such that for every k ≥ τ we have

$$\widehat{C}_k := \text{cast}(\mathbf{A}(1)^k \mathbf{B}) = \begin{cases} \overline{C}_1 \text{ if } k \bmod 2 = 1 & \widehat{D}_k := \text{cast}(\mathbf{A}(1)^k \mathbf{H}_0) = \begin{cases} \overline{D}_1 \text{ if } k \bmod 2 = 1 \\ \overline{D}_2 \text{ if } k \bmod 2 = 0 \end{cases} \end{cases}$$

Finally, if for simplicity we consider τ mod 2 = 0, we have that for 2k ≥ τ

$$\begin{aligned}\widehat{H}_{2k} &= \text{cast} \left( \sum_{i=1}^{\tau-1} \widehat{C}_i + \left(k - \frac{\tau}{2} + 1\right) \overline{C}_2 + \left(k - \frac{\tau}{2}\right) \overline{C}_1 + k \overline{D}_2 \right) \\ \widehat{H}_{2k+1} &= \text{cast} \left( \sum_{i=1}^{\tau-1} \widehat{C}_i + \left(k - \frac{\tau}{2} + 1\right) (\overline{C}_2 + \overline{C}_1) + k \overline{D}_1 \right)\end{aligned}$$

where by factoring out k inside cast, we note that for large enough k, the real and imaginary parts of each element of the matrices inside cast will be either constant, smaller than minx∈<sup>R</sup> cast(x) or larger than maxx∈<sup>R</sup> cast(x). Thus there exist H1, H<sup>2</sup> ∈ <sup>C</sup> n×d and ¯k ≥ τ such that [\(5\)](#page-20-1) is satisfied, concluding the proof for the case of a single layer.

Multiple Layers Note that for one layer we have two subsequences (one of even and one of odd elements) of the output sequence yˆ1, yˆ2, . . . converging after a finite number of elements. This means that there exist a, b ∈ R p such that for all k ≥ ¯k we have

$$\hat{y}_{2k} = a, \quad \hat{y}_{2k+1} = b.$$

Now, consider an additional layer that takes as input x (2) 1 , . . . , x (2) k , with x (2) <sup>i</sup> = yˆ<sup>i</sup> and outputs yˆ (2) 1 , . . . , yˆ (2) k as

$$H_k^{(2)} = A^{(2)}(\mathbf{x}_k^{(2)})H_{k-1}^{(2)} + B^{(2)}(\mathbf{x}_k^{(2)}), \quad \hat{y}_k^{(2)} = \text{dec}^{(2)}(H_k^{(2)}, \mathbf{x}_k^{(2)}).$$

Without loss of generality, assume for simplicity that ¯k = 1 and that xˆ (2) <sup>2</sup><sup>k</sup> = a and xˆ (2) <sup>2</sup>k+1 = b for all k. If we set

$$\begin{aligned} A_1 &:= A^{(2)}(a), & A_2 &:= A^{(2)}(b), \\ B_1 &:= B^{(2)}(a), & B_2 &:= B^{(2)}(b), \\ C_1 &:= A_1 A_2, & C_2 &:= A_1 B_2 + B_1, \end{aligned}$$

then we can write the states of the second layer at even indices as

$$\begin{aligned} H_{2k}^{(2)} &= A_1 H_{2k-1}^{(2)} + B_1 = A_1 A_2 H_{2k-2}^{(2)} + A_1 B_2 + B_1 \\ &= C_1 H_{2(k-1)}^{(2)} + C_2 = \sum_{i=0}^{k-1} C_1^i C_2 + C_1^k H_0 \end{aligned}$$

Furthermore, for the states at odd indices, we have

$$H_{2k+1}^{(2)} = A_2 H_{2k}^{(2)} + B_2 = \sum_{i=0}^{k-1} A_2 C_1^i C_2 + A_2 C_1^k H_0 + B_2.$$

We notice that the sequences H (2) 2k and H (2) <sup>2</sup>k+1 are in a form similar to H<sup>k</sup> of the first layer. If the assumption on the eigenvalues of the state-transition matrices of the second layer does not hold, this means that for all x, y each eigenvalue of A(2)(x)A(2)(y), including C1, is real (but possibly negative). Therefore, we can proceed similarly to the case of one layer, i.e. using the powers of the Jordan canonical form of <sup>C</sup>1, to show that if we let <sup>H</sup>c(2) 2k and <sup>H</sup>c(2) <sup>2</sup>k+1 being the finite precision counterparts of H (2) 2k and H (2) <sup>2</sup>k+1, then there exist H (2) 1 , H (2) 2 , H (2) 3 , H (2) <sup>4</sup> ∈ <sup>C</sup> n×d , ¯k<sup>2</sup> ≥ 0 such that for every k ≥ ¯k

$$\widehat{H}_{2k}^{(2)} = \begin{cases} \overline{H}_1^{(2)} & \text{if } k \bmod 2 = 0 \\ \overline{H}_2^{(2)} & \text{if } k \bmod 2 = 1 \end{cases}, \quad \widehat{H}_{2k+1}^{(2)} = \begin{cases} \overline{H}_3^{(2)} & \text{if } k \bmod 2 = 0 \\ \overline{H}_4^{(2)} & \text{if } k \bmod 2 = 1 \end{cases}$$

.

Therefore, for k ≥ ¯k2, the function k 7→ H (2) <sup>k</sup> will be periodic with period a divisor of four and hence no matter the choice of dec(2), also the function k 7→ yˆ (2) <sup>k</sup> will be periodic with period a divisor of 4. Consequently, with two layers one can recognize the language (1<sup>m</sup>) <sup>∗</sup> only when m = 1, m = 2, or m = 4, since those are the only cases where k 7→ y<sup>k</sup> has a period which is a divisor of 4. Thanks to the assumption on the eigenvalues of the products of state-transition matrices, we can extend this argument inductively to the case of an LRNN with L layers. In particular, for the i-th layer, the induction hypothesis is that we assume k 7→ x (i) k , mapping k to the k-th input to the layer, to be periodic with period a divisor of 2 i−1 for k large enough. Hence, there will be 2 i−1 subsequences of states, each containing powers of the product of 2 i−1 state-transition matrices. From our hypothesis on the eigenvalues of products of state-transition matrices, such product will have only real eigenvalues and hence each subsequence will have 2 converging subsequences resulting in k 7→ H (i) k and consequently k 7→ yˆ (i) k and hence k 7→ x (i+1) k , for k large enough, being periodic with period a divisor of 2 i . Therefore, for the L-th layer, there exists ¯k<sup>L</sup> ≥ 0 such that for every k ≥ ¯kL, the function k 7→ yˆ (L) k is periodic with a period which is a divisor of 2 <sup>L</sup> and thus it can recognize the language (1<sup>m</sup>) <sup>∗</sup> only when 2 <sup>L</sup> mod m = 0, which happens only when there exists p ≤ L such that m = 2<sup>p</sup> and hence m is a power of two, ending the proof.

# C PRODUCTS OF GENERALIZED HOUSEHOLDER MATRICES – PROOFS

We provide proofs for the results stated in Section [4.3.](#page-5-5) Before that, we illustrate how a linear RNN with one layer and state transition matrices that are products of 2 Householder matrices can count modulo m.

### C.1 PRODUCTS OF TWO HOUSEHOLDERS AND MODULAR COUNTING

Counting modulo m can be achieved by rotating a vector in R <sup>2</sup> by an angle of 2π/m radians, and we can express a rotation matrix as a product of two reflection matrices, which are GH matrices with eigenvalues in {−1, 1} (see Appendix [C.1\)](#page-21-0). Inded, for any m ∈ N there exist unit norm vectors v1, v<sup>2</sup> ∈ <sup>R</sup> 2 such that

$$R(\theta) := \begin{bmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{bmatrix} = (I - 2v_1v_1^\top) (I - 2v_2v_2^\top), \quad \theta = \frac{2\pi}{m}.$$

If we set the state-transition matrix in [\(1\)](#page-2-1) to A(1) = R(θ), an LRNN with one layer can count modulo m, since if we also set H<sup>0</sup> = (1, 0)<sup>⊤</sup> and dec(H, x) = arg max<sup>i</sup> D<sup>⊤</sup> <sup>i</sup> H, with D<sup>i</sup> = R(iθ)H<sup>0</sup> for all i ∈ {0, . . . , m − 1}, then for the input x = 1<sup>t</sup> and since R has period 2π, we get

$$\hat{y}_t = \text{dec}(\mathbf{H}_t, 1) = \text{dec}(\mathbf{A}(1)^t \mathbf{H}_0, 1) = \text{dec}(\mathbf{R}(t\theta) \mathbf{H}_0, 1) = t \bmod m.$$

#### C.2 PROOF OF PROPOSITION [1](#page-5-3)

First item It can be shown by noting that if C ∈ M<sup>n</sup> 1 ([−1, 1]), then ∥C∥ ≤ 1 and using the sub-multiplicative property of the Euclidean norm, i.e the fact that ∥AB∥ ≤ ∥A∥∥B∥.

Second item Note that any real matrix has a singular value decomposition. Hence we can write

$$M = USV^T$$

with U,V ∈ R <sup>n</sup>×<sup>n</sup> orthogonal and S = Diag(σ1, . . . , σn) with σ<sup>i</sup> ∈ [0, 1], since ∥M∥ ≤ 1. It follows from the n-reflections theorem[<sup>4</sup>](#page-22-1) that we can write U and V as either the identity I ∈ M<sup>n</sup> 1 ({1}) or the product of at most n reflections, each of which is in M<sup>n</sup> 1 ({−1}). Hence U,V ∈ M<sup>n</sup> n ({−1, 1}). We can also write the matrix S as the product of n GH matrices as

$$S = S_1 S_2 \dots S_n, \quad S_i = I - (1 - \sigma_i) e_i e_i^\top$$

where e<sup>i</sup> is the i-th element of the canonical basis of R <sup>n</sup>. Hence, S ∈ M<sup>n</sup> n ([0, 1]). The proof of the first part is concluded since we wrote each of U,S,V as a product of at most n GH matrices. If M is orthogonal, we apply the n-reflections theorem directly. We also note that if M = P ∈ {0, 1} n×n with P being a permutation matrix different from the identity, it can be written as products of at most n − 1 *swaps*, i.e. permutation matrices permuting only two elements. Therefore we have that there exists an integer k ≤ n − 1 and indices i1, . . . , i<sup>k</sup> and j1, . . . , j<sup>k</sup> such that i<sup>l</sup> ̸= j<sup>l</sup> and

$$P = \prod_{l=1}^{k-1} P_{i_l j_l}, \quad , \mathbf{P}_{ij} = (I - 2\mathbf{v}_{ij}\mathbf{v}_{ij}^\top) \quad v_{ijl} = \begin{cases} 1/\sqrt{2} & \text{if } l = i \\ -1/\sqrt{2} & \text{if } l = j \\ 0 & \text{otherwise} \end{cases},$$

where we set vij = (vij1, . . . , vijn). Note that since ∥vij∥ = 1, Pij ∈ M<sup>n</sup> k ({−1}) with k ≤ n. For the the case where M = I we can use the fact that I ∈ M<sup>n</sup> 1 ({1}).

Third item Let N = C1C<sup>2</sup> · · · C<sup>k</sup> ∈ M<sup>n</sup> k ((−1, 1]), with C<sup>i</sup> = I − βiziz ⊤ <sup>i</sup> with ∥zi∥ = 1 and β<sup>i</sup> ∈ [0, 2). If N = I the statement is satisfied, otherwise, let V = span{z<sup>i</sup> : i ∈ {1, . . . , k}, β<sup>i</sup> > 0}. Any unit vector v ∈ <sup>R</sup> <sup>n</sup> can then be written as v = v<sup>1</sup> + v<sup>2</sup> with v<sup>1</sup> ∈ V, v<sup>2</sup> ∈ V<sup>⊤</sup> and ∥v1∥ , ∥v2∥ ≤ 1. Now, if v<sup>1</sup> = 0, then N v = v, and hence v is an eigenvector with eigenvalue 1. Instead, if v<sup>1</sup> ̸= 0, then there exists i ′ ∈ {1, . . . , k} (we take the largest one) such that β<sup>i</sup> ′ ∈ (0, 2) and (v <sup>⊤</sup>z<sup>i</sup> ′ ) <sup>2</sup> = (v ⊤ <sup>1</sup> z<sup>i</sup> ′ ) <sup>2</sup> ∈ (0, 1]. Therefore, if i ′ < k, then either β<sup>j</sup> = 0 or z ⊤ <sup>j</sup> v = 0 so that Cjv = v for all j ∈ {i ′ + 1, . . . , k}. Moreover, we have that

$$\|\mathbf{C}_{i'} \mathbf{v}\|^2 = \|\mathbf{v} - \beta_{i'} \mathbf{z}_{i'} \mathbf{z}_{i'}^\top \mathbf{v}\|^2 = 1 - \beta_{i'} (2 - \beta_{i'}) (\mathbf{v}^\top \mathbf{z}_{i'})^2 < 1,$$

where the last line comes from the fact that minx∈[0,2] x(2 − x) = 0 and is only reached at x = 0 and x = 2, while β<sup>i</sup> ′ ∈ (0, 2). Therefore, since for every i, ∥Ci∥ ≤ 1 and the Euclidean norm is sub-multiplicative we have

$$\| \mathbf{N} \mathbf{v} \| = \| \mathbf{C}_1 \mathbf{C}_2 \dots \mathbf{C}_k \mathbf{v} \| = \| \mathbf{C}_1 \mathbf{C}_2 \dots \mathbf{C}_{i' \mathbf{v}} \| \leq \| \mathbf{C}_1 \| \dots \| \mathbf{C}_{i' \mathbf{v}} \| < 1.$$

Therefore, if v is also an eigenvector with eigenvalue λ ∈ C, then ∥N v∥ = |λ| < 1. Hence, we proved that for every eigenvector with eigenvalue λ either λ = 1 or |λ| < 1.

It remains to show that all eigenvalues of N ∈ M<sup>n</sup> ([0, 1]) are in [0, 1]. From the assumptions N = C1C<sup>2</sup> with C1, C<sup>2</sup> symmetric and positive semi-definite, therefore C<sup>1</sup> has a unique symmetric and positive semi-definite square root C 1/2 1 such that C 1/2 <sup>1</sup> C 1/2 <sup>1</sup> = C1. If C<sup>1</sup> is non-singular (invertible) then

$$\mathbf{C}_1\mathbf{C}_2 = \mathbf{C}_1^{1/2}\mathbf{C}_1^{1/2}\mathbf{C}_2\mathbf{C}_1^{1/2}\mathbf{C}_1^{-1/2}.$$

<sup>4</sup>This is a specialization of the Cartan–Dieudonne Theorem to ´ <sup>R</sup> n , see Theorem 3 in [https://faculty.](https://faculty.uml.edu/dklain/orthogonal.pdf) [uml.edu/dklain/orthogonal.pdf](https://faculty.uml.edu/dklain/orthogonal.pdf) for a proof.

Thus, C1C<sup>2</sup> is similar to C 1/2 <sup>1</sup> C2C 1/2 and shares its eigenvalues. Moreover C 1/2 <sup>1</sup> C2C 1/2 1 is symmetric positive semi-definite (having real nonnegative eigenvalues) because C 1/2 1 and C<sup>2</sup> are symmetric and v <sup>⊤</sup>C 1/2 <sup>1</sup> C2C 1/2 <sup>1</sup> v = z <sup>⊤</sup>C2z ≥ 0 with z = C 1/2 <sup>1</sup> v since C<sup>2</sup> is positive semi-definite. Instead, if C<sup>1</sup> is singular, for t > 0 the matrix C<sup>1</sup> + tI is positive definite and non-singular. Hence (C<sup>1</sup> + tI)C<sup>2</sup> has real and nonnegative eigenvalues. Since C1C<sup>2</sup> = limt→0(C<sup>1</sup> + tI)(C2) and the eigenvalues are a continuous function of the entry of the matrix, C1C<sup>2</sup> has positive real eigenvalues. Since the modulus of any eigenvalue is smaller or equal than the euclidean norm of the matrix, which is smaller than one from the first point of the theorem, the statement follows.

### C.3 PROOF OF THEOREM [3](#page-5-0)

We first recall the notion of group isomorphism. Two groups (G, ∗) and (H, ·) where G, H are the sets and ⋆ and · are the associative operations, are isomorphic, if there exists a bijective map f : G → H such that for every g ∈ G, h ∈ H

$$f(g * h) = f(g) \cdot f(h).$$

We view the LRNN layer in [\(1\)](#page-2-1) as the automaton Alin = (Σ, H, H0, δlin), where δlin(H, w) = A(w)H + B(w), which is extended in the usual way, and H = {δlin(H0, w) : w ∈ Σ <sup>∗</sup>}. Since we assumed that T (A) is a group, from Cayley's theorem we have that it is isomorphic to a subgroup of Sn, which is the set of permutations on a set of n elements. Furthermore, each element in S<sup>n</sup> can be represented as an n × n permutation matrix. Since in general n ̸= |Q|, we cannot let H to be a set of one hot vectors each corresponding to states in Q. Instead, we let H<sup>0</sup> = (1, . . . , n) ⊤, P ⊂ {0, 1} <sup>n</sup>×<sup>n</sup> be the set of permutation matrices and set B ≡ 0 and A : Σ → P to be the function mapping each letter w ∈ Σ to the permutation matrix corresponding to δ(·, w). With this choice we can see that the function f : T (Alin) → T (A) such that f(δlin(·, w)) = δ(·, w) for every w ∈ Σ ∗ is one-to-one (bijective), and from our choice of H0, the map h : T (Alin) → H such that for every w ∈ Σ ∗ , h(δlin(·, w)) = δlin(H0, w) is also bijective. Moreover, the map ϕ : T (A) → Q such that ϕ(δ(·, w)) = δ(q0, w) is surjective because without loss of generality we can consider states that are only reachable from the initial state q0, i.e. Q = {δ(q0, w) : w ∈ Σ <sup>∗</sup>}. Hence if we set g = ϕ ◦ f ◦ h −1 , then g : H → Q is surjective and for every w ∈ Σ and H ∈ H we have that

$$g(\delta_{\text{lin}}(\mathbf{H}, w)) = \delta(g(\mathbf{H}), w)$$

Thus, we have shown that such an LRNN implements A and it does so with finite precision because the entries of all vectors and matrices are bounded integers. Moreover, Let k = maxw∈<sup>Σ</sup> P <sup>q</sup>∈<sup>Q</sup> 1{δ(q, w) ̸= q} = maxw∈<sup>Σ</sup> P<sup>n</sup> <sup>i</sup>=1 1{(A(w)H0)<sup>i</sup> = H0,i} be the maximum number of displaced element of the permutation associated with the alphabet Σ. Then, this means that each permutation can be written as a product of at most k − 1 permutations of two elements. Hence, for every w ∈ Σ, A(w) ∈ M<sup>n</sup> k−1 ({−1, 1}).

If in addition there exists m ∈ <sup>N</sup> such that T (A) is isomorphic to a subgroup of the cyclic group <sup>Z</sup><sup>m</sup> with elements {0, . . . , m − 1}, we can modify the construction above to use a smaller dimension. If m = 2, then <sup>Z</sup><sup>2</sup> has elements {0, 1}, and A implements the parity automaton. Thus, we can set H<sup>0</sup> = −1, A(0) = 1, A(1) = −1 and g(1) = 1 while g(−1) = 0, which means that we can use a scalar recursion. Otherwise, if m ≥ 3, we can modify the construction above by setting H<sup>0</sup> = (1, 0)<sup>⊤</sup> and, if for simplicity we assume Σ ∈ {0, . . . , m − 1}, for every w ∈ Σ we let A(w) be the 2 × 2 rotation matrix corresponding to δ(·, w):

$$\mathbf{A}(w) = \mathbf{R}(\theta_w) = \begin{bmatrix} \cos \theta_w & -\sin \theta_w \\ \sin \theta_w & \cos \theta_w \end{bmatrix}, \quad \theta_w = \frac{2\pi w}{m},$$

such that R(θw) ∈ M<sup>2</sup> 2 ({−1}) (from Proposition [1\)](#page-5-3). This concludes the proof.

# C.4 KROHN-RHODES THEOREM

Before presenting the proof for Theorem [4,](#page-6-0) we provide the statement for the landmark result of Krohn-Rhodes [\(Krohn & Rhodes, 1965\)](#page-12-10), after giving the definition of the cascade product of two FSA.

Definition 1 (Cascade product). *Given two FSA* A = (Σ, Q, q0, δ) *and* B = (Q × Σ, Q′ , q′ 0 , δ′ )*, we define the cascade product FSA as* C = B ◦ A = (Σ, Q × Q′ ,(q0, q′ 0 ), δ′′) *where for any* w ∈ Σ

$$\delta''((q, q'), w) := (\delta(q, w), \delta(q', (q, w)))$$

Theorem 5 (Krohn-Rhodes, Theorem 4 in [Maler & Pnueli](#page-12-9) [\(1994\)](#page-12-9)). *For every FSA* A = (Σ, Q, q0, δ) *there exists* s ≤ 2 <sup>|</sup>Q<sup>|</sup> *and a cascade product FSA* C = A(s) ◦ · · · ◦ A(1) = (Σ, Q×, q<sup>×</sup> 0 , δ<sup>×</sup>)*, with* A(i) = Σ (i) , Q(i) , q (i) 0 , δ(i) *, with* |Q(i) | ≤ |Q|*, and a function* W : Q<sup>×</sup> → Q *such that for any* w ∈ Σ ∗ *,* δ(q0, w) = W(δ <sup>×</sup>(q × 0 , w)) *and each* A(i) *is permutation-reset automaton, which means that for every* w (i) ∈ Σ (i) *,* δ (i) (·, w(i) ) *is either a bijection (i.e. a permutation over* Q*) or constant, ie.* δ(·, w(i) ) = q(w (i) ) ∈ Q(i) *.*

#### C.5 PROOF OF THEOREM [4](#page-6-0)

We apply the Krohn-Rhodes theorem (Theorem [5\)](#page-24-1) to write A as the cascade product FSA C = A(s) ◦ · · · ◦ A(1) with each FSA A(i) = Σ (i) , Q(i) , q (i) 0 , δ(i) being permutation-reset and we show how the LRNN can implement C by first showing how its i-th layer, with the structure in [\(1\)](#page-2-1), can implement A(i) .

Let n = |Q(i) | and without loss of generality assume that Σ = {1, 2, . . . , |Σ|} and Q(i) = {1, 2, . . . , n} with q (i) <sup>0</sup> = 1. For every w ∈ Σ (i) we set A(i) (w) ∈ {0, 1} <sup>n</sup>×<sup>n</sup>, B(i) (w) ∈ {0, 1} n such that for every q, q′ ∈ Q(i)

$$\begin{aligned} \mathbf{A}^{(i)}(w)_{q',q} &= \mathbf{1}\{\delta(q, w) = q'\}, & \mathbf{B}^{(i)}(w)_{q'} &= 0, & \text{if } \delta^{(i)}(\cdot, w) \text{ is bijective, or} \\ \mathbf{A}^{(i)}(w)_{q',q} &= 0, & \mathbf{B}^{(i)}(w)_{q'} &= \mathbf{1}\{q' = q(w)\}, & \text{if } \delta^{(i)}(\cdot, w) \equiv q(w). \end{aligned}$$

Then, for every word w(i) = w (i) 1 . . . w (i) <sup>t</sup> ∈ Σ (i)∗ , we set g : R <sup>n</sup> → <sup>R</sup>, such that g(x) = (1, . . . , n) <sup>⊤</sup>x and

$$\begin{aligned} \mathbf{H}_t^{(i)} &= \mathbf{A}^{(i)}(w_t^{(i)})\mathbf{H}_{t-1}^{(i)} + \mathbf{B}^{(i)}(w_t^{(i)}), & \mathbf{H}_0^{(i)} &= (1, 0, \dots, 0)^\top \in \mathbb{R}^d \\ \mathbf{y}^{(i)} &= \text{dec}^{(i)}(\mathbf{H}_t^{(i)}, w_t^{(i)}) = (g(\mathbf{H}_t^{(i)}), w_t^{(i)}) = (\delta^{(i)}(q_0^{(i)}, \mathbf{w}^{(i)}), w^{(i)}) \end{aligned}$$

So that such construction implements A(i) . In addition, by letting w = w<sup>1</sup> . . . w<sup>t</sup> ∈ Σ <sup>∗</sup> be the input to the LRNN, i.e. w (1) <sup>j</sup> = w<sup>j</sup> , and setting the output of each layer as the input to the next, i.e. w (i) <sup>j</sup> = y (i−1) j for i ≥ 2, for the output of the last layer we get

$$\begin{aligned} y_t^{(s)} &= \deg^{(s)}(\mathbf{H}_t, w_t^{(s)}) \\ &= (\delta^{(s)}(q_0^{(s)}, \mathbf{w}^{(s)}), y_t^{(s-1)}) \\ &= (\delta^{(s)}(q_0^{(s)}, \mathbf{w}^{(s)}), \delta^{(s-1)}(q_0^{(s-1)}, \mathbf{w}^{(s-1)}), y_t^{(s-2)}) \\ &= (\delta^{(s)}(q_0^{(s)}, \mathbf{w}^{(s)}), \dots, \delta^{(1)}(q_0^{(1)}, \mathbf{w}), w_t) \in \mathbb{N}^{s+1}, \end{aligned}$$

where we removed the nested parenthesis for simplicity. Hence, the first s elements of y (s) t are exactly the output of the cascade FSA C. Note that our construction can be implemented in finite precision since we only used matrices/vectors with entries either in {0, 1}, requiring only one bit, or in Q(i) ⊂ <sup>N</sup>, that can also be implemented using finite precision with |Q(i) | integers, requiring log<sup>2</sup> (|Q(i) |) bits. Also note that we can exclude w<sup>t</sup> from the output y (s) <sup>t</sup> by changing dec(s) , to bring the dimension of the output, end hence the width of the LRNN, to N s .

It is also the case that A(i) (w) ≤ 1 for every w ∈ Σ (i) since A(i) (w) is either a permutation matrix ( A(i) (w) = 1 ) or the zero matrix ( A(i) (w) = 0). Also, for every permutation matrix P ∈ {0, 1} <sup>n</sup>×<sup>n</sup> which permutes only k ≤ n elements we have that P ∈ M<sup>n</sup> k−1 ({−1, 1}).

Furthermore, for the zero matrix, we have

$$0 = \prod_{i=1}^n (I - \mathbf{e}_i \mathbf{e}_i^\top) \in \mathcal{M}_n^n(\{0\})$$

# D LRNNS CAN DO MODULAR ADDITION USING ONLY REFLECTIONS

In this section, we explain how an LRNN with two layers and using only Householder state transition matrices (reflections) can compute addition modulo m ∈ <sup>N</sup>, i.e it can map words x1, . . . , x<sup>t</sup> with <sup>x</sup><sup>i</sup> ∈ {0, . . . , m − <sup>1</sup>} into y<sup>t</sup> = (P<sup>m</sup> <sup>i</sup>=1 xi) mod m for arbitrary t ∈ <sup>N</sup>. This corresponds to solving the group word problem associated with the cyclic group Zm. We note that our modification of DeltaNet, namely DeltaNet [−1, 1] can therefore solve addition modulo m with 2 layers.

If the state transition matrices can be generic rotation matrices, then an LRNN can perform addition modulo m using just one layer by mapping each element of <sup>Z</sup><sup>m</sup> to the corresponding 2 × 2 rotation matrix as shown in Appendix [C.3.](#page-23-0) Such construction requires a number of states for the LRNN equal to m, i.e. the number of elements of the group <sup>Z</sup>m. However, since here we assume that state transition matrices are reflections, we cannot map each element of the group to a rotation (since those are a product of 2 reflections) and our construction for the LRNN will require two layers. Specifically, the first layer will count modulo 2, i.e. it will output the sequence y (1) 1 , . . . , y (1) <sup>t</sup> where y (1) <sup>i</sup> = (x<sup>i</sup> , i mod 2), while the second layer will have 2m states and will use two different reflection matrices for each group element, depending on the value of y (1) i,<sup>2</sup> = i mod 2. Formally, we have the following result.

Theorem 6 (Modular addition with reflections). *An LRNN with two layers in the form [\(1\)](#page-2-1), where* A : N → {−1} *for the first layer and* A : <sup>R</sup> <sup>2</sup> → M<sup>2</sup> 1 ({−1}) *for the second layer, with* M<sup>2</sup> <sup>1</sup> *defined in [\(3\)](#page-5-2), can perform addition modulo* m *for any* m ∈ N*. In particular, the LRNN will have 2 scalar states in the first layer and* 2m *states, each being a vector in* R 2 *, in the second layer.*

*Proof.* The first layer of the LRNN will implement counting modulo 2 as follows.

$$h_0^{(1)} = 0, \quad h_t^{(1)} = -h_{t-1}^{(1)} + 1, \quad \mathbf{y}_t^{(1)} = \text{dec}^{(1)}(h_t, x_t) = (x_t, h_t).$$

We note that the state-transition matrix (the scalar −1) is a reflection since {−1} = M<sup>1</sup> 1 ({−1}). For the second layer, we have instead

$$\begin{aligned} h_0^{(2)} &= (1, 0)^\top, \quad h_t^{(2)} = \mathbf{A}^{(2)}(\mathbf{y}_t^{(1)}) \mathbf{h}_{t-1}^{(2)}, \quad \mathbf{y}_t^{(2)} = \text{dec}^{(2)}(\mathbf{h}_t^{(2)}, \mathbf{y}_t^{(1)}) \\ \mathbf{A}^{(2)}(\mathbf{y}) &= \mathbf{H}(\theta(y_1, y_2)) = \begin{bmatrix} \cos \theta(y_1, y_2) & \sin \theta(y_1, y_2) \\ \sin \theta(y_1, y_2) & -\cos \theta(y_1, y_2) \end{bmatrix} \\ \text{dec}^{(2)}(\mathbf{h}, \mathbf{y}) &= \arg \max_{i \in \{0, \dots, m-1\}} \max(\mathbf{c}_i^\top \mathbf{h}, \mathbf{d}_i^\top \mathbf{h}) \end{aligned}$$

where y = (y1, y2) <sup>⊤</sup> ∈ {0, . . . , m−1} × {0, 1}, H(α) is the 2×2 reflection matrix that reflects all vectors by a line having an angle of α/2 with the line passing from the origin and the vector (1, 0)<sup>⊤</sup> and θ : {0, . . . , m − 1} × {0, 1} → <sup>R</sup> determines the angle of the reflection and is defined as

$$\theta(i, 1) = \frac{(1-2i)\pi}{m}, \quad \theta(i, 0) = \frac{(1+2i)\pi}{m}, \quad \text{for all } i \in \{0, \dots, m-1\}.$$

Moreover C = {c0, . . . , cm−1} and D = {d0, . . . , dm−1} are the two sets of states corresponding to reflections and rotations respectively and are defined as

$$\begin{aligned} d_0 &= h_0^{(2)} = (1, 0)^\top, \quad c_0 = H(\pi/m)d_0, \\ d_i &= R(2i\pi/m)d_0, \quad c_i = R(-2i\pi/m)c_0 \quad \text{for all } i \in \{0, \dots, m-1\}, \end{aligned}$$

where R(β) is a rotation matrix with angle β ∈ <sup>R</sup>.

Let α, γ ∈ R, the following are standard identities of products of 2D rotations and reflections.

$$\begin{aligned} R(\alpha)R(\gamma) &= R(\alpha + \gamma), & H(\alpha)H(\gamma) &= R(\alpha - \gamma), \\ R(\alpha)H(\gamma) &= H(\alpha + \gamma) & H(\gamma)R(\alpha) &= H(\gamma - \alpha). \end{aligned}$$

From our choice of θ, d<sup>i</sup> and c<sup>i</sup> , using the identities above and the the fact that R is a periodic function with period 2π we have that

$$\begin{aligned} H(\theta(j, 1))d_i &= H(\theta(j, 1))R(2i\pi/m)d_0 \\ &= H(\theta(j, 1))R(2i\pi/m)H(\pi/m)c_0 \\ &= H(\theta(j, 1))H(\theta(i, 0))c_0 \\ &= R(\theta(j, 1) - \theta(i, 0))c_0 \\ &= R(-(2(i + j)\pi/m)c_0 = c_{i+j \bmod m}, \end{aligned} \tag{6}$$

and similarly

$$\begin{aligned} H(\theta(j, 0))c_i &= H(\theta(j, 1))R(-2i\pi/m)c_0 \\ &= H(\theta(j, 0))R(-2i\pi/m)H(\pi/m)d_0 \\ &= H(\theta(j, 0))H(\theta(i, 1))d_0 \\ &= R(\theta(j, 0) - \theta(i, 1))d_0 \\ &= R(2(i + j)\pi/m)d_0 = d_{i+j \bmod m}, \end{aligned} \tag{7}$$

for every i, j ∈ {0, . . . , m − 1}. We will now prove by induction that

$$h_t^{(2)} = \begin{cases} c_{y_t} & \text{if } t \bmod 2 = 1 \\ d_{y_t} & \text{if } t \bmod 2 = 0 \end{cases} \quad (8)$$

where we recall that y<sup>i</sup> := (P<sup>i</sup> <sup>j</sup>=1 x<sup>j</sup> ) mod m and that, by definition, h (2) <sup>0</sup> = d<sup>0</sup> and h (2) <sup>i</sup> = H(θ(x<sup>i</sup> , i mod 2))h (2) i−1 , since y (1) <sup>i</sup> = (x<sup>i</sup> , i mod 2). For the base case we have that

$$\begin{aligned} h_1^{(2)} &= H(\theta(x_1, 1))h_0^{(2)} = H(\theta(x_1, 1))d_0 = c_{x_1 \bmod m} = c_{y_1} \\ h_2^{(2)} &= H(\theta(x_2, 0))h_1^{(2)} = H(\theta(x_2, 0))c_{x_1 \bmod m} = d_{x_1+x_2 \bmod m} = d_{y_2}, \end{aligned}$$

where we have used [\(6\)](#page-25-1) and [\(7\)](#page-26-2). As induction hypothesis, suppose that for i ≥ 2

$$h_i^{(2)} = \begin{cases} c_{y_i} & \text{if } i \bmod 2 = 1 \\ d_{y_i} & \text{if } i \bmod 2 = 0 \end{cases}$$

then, using again [\(6\)](#page-25-1) and [\(7\)](#page-26-2), we obtain

$$h_{i+1}^{(2)} = \begin{cases} \mathbf{H}(\theta(x_{i+1}, 1))\mathbf{h}_i^{(2)} = \mathbf{H}(\theta(x_{i+1}, 1))\mathbf{c}_{y_i} = \mathbf{c}_{x_{i+1}+y_i} \text{ mod } m = \mathbf{c}_{y_{i+1}} & \text{if } i \text{ mod } 2 = 1 \\ \mathbf{H}(\theta(x_{i+1}, 0))\mathbf{h}_i^{(2)} = \mathbf{H}(\theta(x_{i+1}, 0))\mathbf{d}_{s_i} = \mathbf{d}_{x_{i+1}+y_i} \text{ mod } m = \mathbf{d}_{y_{i+1}} & \text{if } i \text{ mod } 2 = 0 \end{cases}$$

.

which completes our proof by induction yielding [\(8\)](#page-26-3). Finally, using the definition of dec(2), [\(8\)](#page-26-3) and as long as d<sup>i</sup> ̸= c<sup>j</sup> , d<sup>i</sup> ̸= d<sup>j</sup> and c<sup>i</sup> ̸= c<sup>j</sup> for every i, j with i ̸= j, which is guaranteed by our choice of θ, we have that dec(2)(h (2) t , y (1) t ) = (P<sup>i</sup> <sup>j</sup>=1 x<sup>j</sup> ) mod m = yt, ending the proof.

# E EXPERIMENTS

### E.1 CHOMSKY HIERARCHY

Here, we provide details on the formal language tasks and experimental protocol of Section [5.1.](#page-7-2)

#### E.1.1 DETAILS ON THE EXPERIMENTAL SETUP

Like [Beck et al.](#page-10-0) [\(2024\)](#page-10-0), we trained each model with sequence lengths ranging from 3 to 40 and evaluated on lengths from 40 to 256, to understand the length generalization capabilities. We compared mLSTM and sLSTM with two models: Mamba [\(Gu & Dao, 2024\)](#page-11-0) and DeltaNet [\(Yang et al., 2024b\)](#page-13-6). Moreover, we also include a Transformer [\(Vaswani et al., 2017\)](#page-13-1) baseline. For parity, all models contain 2 blocks (layers), with 4 heads for the xLSTM and DeltaNet models. We set the embedding and heads' dimensions to 128. For Mamba and DeltaNet, we also enable the 1-D depthwise-separable convolution layer with kernel size equal to 4 after the query/key/value projection. For modular arithmetic, we increase the number of layers to 3 and use a gradient clipping norm of 1.0 for Transformer, Mamba, and DeltaNet, while for mLSTM and sLSTM we decrease the embedding size and number of heads to 64 and 1, respectively, as well as use a standard initialization for the bias parameters. We train each model using AdamW [\(Loshchilov & Hutter, 2019\)](#page-12-13) without gradient clipping, using 3 different learning rates (1e-2, 1e-3, 5e-4 1e-4), with 3 different seeds each. We pick the best based on the median of the 3 seeds for every learning rate value. We use a batch size of 1024 (except for mLSTM, where we use 512 due to OOM error) and a cosine annealing learning rate schedule [\(Loshchilov & Hutter, 2017\)](#page-12-14) (minimum learning rate: 1e-6) after 10% warm-up steps. The weight decay is set to 0.1 during training. We train on every task for 100k steps in total. At each training step, we make sure to generate a valid random sample from the task at hand (see below).

Table 5: Performance comparison of various recurrent models on regular and context-free language tasks. recurrent models on formal language tasks. We report the median ± median absolute deviation (*left table*) and best score (*right table*) of 3 independent runs with different random seeds. Scores represent scaled accuracy, with 1.0 indicating perfect performance and 0.0 random guessing. The positive impact of allowing negative eigenvalues ([−1, 1] range) versus restricting to positive eigenvalues ([0, 1] range) is evident across different model architectures.

|             |                  | Parity  | Mod. (w/o | brackets) Arithmetic | Mod. (w/ | brackets) Arithmetic | Mod. (w/ no | Arithm. brackets, mult) |
|-------------|------------------|---------|-----------|----------------------|----------|----------------------|-------------|-------------------------|
| Transformer | 0.003            | ± 0.013 | 0.018     | ± 0.009              | 0.064    | ± 0.003              | 0.025       | ± 0.000                 |
| mLSTM       | 0.018            | ± 0.035 | 0.027     | ± 0.013              | 0.114    | ± 0.000              | 0.034       | ± 0.001                 |
| sLSTM       | 1.000            | ± 0.000 | 0.124     | ± 0.000              | 0.163    | ± 0.015              | 0.153       | ± 0.020                 |
| Mamba [0    | , 1] 0.000       | ± 0.000 | 0.066     | ± 0.029              | 0.116    | ± 0.007              | 0.072       | ± 0.008                 |
| Mamba [     | − 1 , 1] 1.000   | ± 0.000 | 0.214     | ± 0.027              | 0.098    | ± 0.009              | 0.126       | ± 0.010                 |
| DeltaNet    | [0 , 1] 0.010    | ± 0.005 | 0.214     | ± 0.056              | 0.162    | ± 0.018              | 0.113       | ± 0.009                 |
| DeltaNet    | [ − 1 , 1] 0.999 | ± 0.006 | 0.826     | ± 0.146              | 0.227    | ± 0.011              | 0.129       | ± 0.016                 |

#### E.1.2 DETAILS ON THE EVALUATED TASKS

In Section [5.1](#page-7-2) we conducted empirical evaluations on 3 tasks –parity, modular arithmetic without brackets and with brackets – from various levels of the Chomsky Hierarchy, as proposed by [Deletang](#page-11-2) [et al.](#page-11-2) [\(2023\)](#page-11-2) and similarly used in xLSTM [\(Beck et al., 2024\)](#page-10-0). Details for each task are given below, where |Σ| is the vocabulary size and Accrand is the accuracy of random guessing:

- Parity (|Σ| = 2, Accrand = 0.5). The parity y<sup>t</sup> ∈ {0, 1} of a sequence of ones and zeros x = x<sup>1</sup> . . . x<sup>t</sup> ∈ {0, 1} t is equal to 1 (resp. 0) if the total number of ones in the sequence is odd (resp. even). It is equivalent to addition modulo 2, it can be computed by summing all previous values and then using the modulo 2 function as y<sup>t</sup> = (P<sup>t</sup> <sup>i</sup>=1 xi) mod 2.
- Modular Arithmetic w/o Brackets (|Σ| = 10, Accrand = 1/5). Given a set of special tokens Σ<sup>s</sup> = {+, −, ∗, =, [PAD]} and a modulus m ≥ 1, we set Σ = Σ<sup>s</sup> ∪ {0, . . . , m − 1} and y<sup>t</sup> is equal to the result of the operations modulo m in the sequence x = x1, . . . , x<sup>t</sup> with x<sup>i</sup> ∈ Σ. In our experiments m = 5. An example sequence is as follows:

$$2 - 3 - 3 * 2 = 3$$
 [PAD]

- Modular Arithmetic w/ Brackets, (|Σ| = 12, Accrand = 1/5). Same definition as the modular arithmetic without brackets with a set of special tokens Σ<sup>s</sup> = {+, −, ∗, =,),(, [PAD]}. In our experiments m = 5. An example sequence is as follows:

$$(((3 + 3) + -1) + -2) - ((3 - (-3)) + ((1 + 4))) = 2 \text{ [PAD]}$$

### E.2 STATE-TRACKING

### E.2.1 DETAILS OF THE EXPERIMENTS

For the experiments in Section [5.2,](#page-7-3) we map each element of the group S<sup>5</sup> to an integer from 0 to 119, where 0 corresponds to the identity permutation, and then construct inputs and output sequences of integers x1, . . . x<sup>t</sup> and y1, . . . , y<sup>t</sup> as follows

- S<sup>5</sup> We sample x<sup>i</sup> uniformly at random from {0, . . . , 119}. y<sup>i</sup> is computed as the product of the permutations corresponding to x1, . . . , x<sup>i</sup> applied in order from 1 to i.
- S<sup>5</sup> only swaps As S<sup>5</sup> but x<sup>i</sup> is sampled from the permutations that permute up to two elements (swaps and identity).
- S<sup>5</sup> swaps, 3-permutations As S<sup>5</sup> but x<sup>i</sup> is sampled from the permutations that permute up to three elements.
- S<sup>5</sup> 4 tokens per transition If i mod 4 = 0, then x<sup>i</sup> is sampled uniformly at random from {0, . . . , 119}, otherwise x<sup>i</sup> = 120 (special token). For i > 3, yi+3 is the product of the permutations corresponding to x1, . . . , x<sup>i</sup> , where 120 is treated as the identity permutation. y<sup>i</sup> = 0 for i ∈ {1, 2, 3}.

![](_page_28_Figure_1.jpeg)

Figure 6: Performance (scaled accuracy) vs sequence length of *Transformer*, *mLSTM*, *sLSTM*, *Mamba* and *DeltaNet* variants on different formal language tasks. Trained on sequences up to length 40 (dashed vertical red line). At test time, we sample uniformly at random 8192 sequences with lengths between 40 and 256. The curves show the mean and 95% CI. Note, that the Transformer model fails to length extrapolate, but performs nearly perfectly within the training context length.

For each setup, we randomly sample 1.6M examples for and 40K examples of length 500 to construct the train and test dataset. We note that we are using a substantially larger training set compared to [\(Merrill & Sabharwal, 2023\)](#page-12-5), to reduce the chances of overfitting. We run 3 seeds for each method, changing the network initialization and sampling of the minibatches. The train and validation datasets are kept the same across runs.

![](_page_29_Figure_1.jpeg)

Figure 7: Validation sequence accuracy across different lengths on S<sup>5</sup> after 100 epochs of training (3 seeds). The dashed vertical line indicates the sequence length used during training. Each method is labeled with name, eigenvalue range, and number of layers. The dashed vertical line indicates the sequence length used during training.

We train all models using AdamW with weight decay 0.01, learning rate 0.0001, gradient clipping to 1.0, and a batch size of 512. Both DeltaNet and Mamba models use an embedding dimension of 128 and 4 heads for DeltaNet. In the case of DeltaNet, we do not use 1-D convolutions for these experiments. Other parameters are kept as default.

Full Matrix Baseline. For the full matrix baseline we use a single layer and map directly each token xi to a learnable full state-transition matrix A(xi) ∈ <sup>R</sup> <sup>n</sup>×<sup>n</sup> via one-hot encoding. We then compute, for i ∈ {1, . . . , t} the recursion

$$H_i = A(x_i)H_{i-1}, \quad H_0 = I \in \mathbb{R}^{n \times n}$$

where n is set to 32 for efficiency reasons (memory and compute time grow quickly with n). After that, we flatten each H<sup>i</sup> into a vector and apply first a projection on the unit ball and then a linear decoder to get the final outputs. The projection was added to increase stability since we do not bound the norm of A(xi). Since this model uses a full matrix, with n ≥ 5 it should be fully able to learn S<sup>5</sup> without restricting the transitions in input or using more tokens per transition. However, in some situations, the performance degrades quickly after some input sequence length, probably because the norm of the learned A(xi) is not close enough to one and hence part of the state either vanish or explode for long sequences.

Plots with all runs. We report the plots with all 3 runs per method in Figure [7](#page-29-1) (In Figure [4](#page-8-0) we reported only the best one for each method). Despite our efforts to decrease the variance of the results by increasing training time and dataset size, we report that there is still some variability. For example, one of the runs of DeltaNet [−1, 1] (5L) on S<sup>5</sup> with 4 tokens per transition did not achieve a good accuracy.

# E.2.2 CYCLIC GROUPS

We report in Figure [8](#page-30-2) some experiments on group word problems with the group Z60. For this experiment, we also consider the simplified version where each transition is encoded using 2 tokens. This is done as in the experiments of S<sup>5</sup> with 4 tokens, but using 2 tokens instead of 4. Extending the

![](_page_30_Figure_1.jpeg)

Figure 8: Validation sequence accuracy at different sequence lengths on the cyclic group Z<sup>60</sup> (1 seed). Dashed vertical lines indicate the sequence length used for training (left 32, right 64). Using 2 tokens per transition seems to help only marginally in this case. Mamba [−1, 1] is the bestperforming model. The variants with eigenvalues in [0,1] performed worse.

eigenvalue range seems to help in both settings, although surprisingly, Mamba [−1, 1], even though it has a diagonal state-transition matrix, seems to perform best. We conjecture that in this case, the models might learn the shortcut solutions, also because they do not generalize very well to longer sequences.

### E.3 LANGUAGE MODELING

### E.3.1 DETAILS ON THE EXPERIMENTAL SETUP

We use the training pipeline which is part of the flash-linear-attention library (flame) [\(Yang & Zhang,](#page-13-13) [2024\)](#page-13-13) and which in turn is based on HuggingFace accelerate [\(Gugger et al., 2022\)](#page-11-16). We use stage-2 of the ZeRO optimizer [\(Rajbhandari et al., 2020\)](#page-13-14) with gradient clipping set to auto. The 1.3B parameter DeltaNet models are trained on 32 Nvidia A100s using a per-device batch size of 6 and 5 gradient accumulation steps for 50,000 steps. The 340M parameter DeltaNet models and the 370M parameter Mamba models are trained using a training batch size of 16 and 200,000 steps on 16 Nvidia A100s. All models are trained using a context length of 2048, learning rate of 3e-4. For optimization, we use AdamW [\(Loshchilov & Hutter, 2019\)](#page-12-13), the learning rate was adjusted using cosine annealing [\(Loshchilov & Hutter, 2017\)](#page-12-14) following a linear warm-up period of 250/500 steps for the 340/370M and 1.3B parameter models respectively. We applied a weight decay of 0.01 throughout the training process.

### E.3.2 DETAILS ON THE EVALUATED TASKS

To produce the results in Table [4,](#page-9-0) we use the lm-harness benchmark [\(Gao et al., 2024\)](#page-11-14), focusing on the same tasks as [Yang et al.](#page-13-6) [\(2024b\)](#page-13-6): LAMBADA (LMB) [\(Paperno et al., 2016\)](#page-12-15), PIQA [\(Bisk et al.,](#page-10-4) [2020\)](#page-10-4), HellaSwag (Hella.) [\(Zellers et al., 2019\)](#page-14-1), Winogrande (Wino.) [\(Sakaguchi et al., 2021\)](#page-13-15), and ARC-easy (ARC-e) and ARC-challenge (ARC-c) [\(Clark et al., 2018\)](#page-10-5). Additionally, we evaluate the performance on recall-intensive tasks (like [Yang et al.](#page-13-6) [\(2024b\)](#page-13-6)), including FDA [\(Arora et al., 2023\)](#page-10-6), SWDE [\(Lockard et al., 2019\)](#page-12-16), and SQUAD [\(Rajpurkar et al., 2018\)](#page-13-16), to provide a comprehensive evaluation of our models' capabilities.

### E.4 IMPLEMENTATION

We build on the original code for Mamba[<sup>5</sup>](#page-30-3) and DeltaNet[<sup>6</sup>](#page-30-4) . For DeltaNet, implementing the extended eigenvalue range is straightforward, since there is no need to modify the Triton kernel. However, Mamba requires modifications to the CUDA code of the associative scan for both forward and backward passes which however had no impact on computational cost. We ensured the accuracy of the modifications by comparing the results with a naive implementation using a for-loop. For initial testing of the extended eigenvalue range, we used the pure PyTorch implementation of Mamba by [Torres](#page-13-17) [\(2024\)](#page-13-17). We provide listings of the necessary code changes in Mamba and DeltaNet in Appendix [E.4.1.](#page-32-0) For DeltaNet, this changes also B(xt) in Table [1,](#page-2-0) multiplying it by 2.

<sup>5</sup><https://github.com/state-spaces/mamba>

<sup>6</sup><https://github.com/sustcsonglin/flash-linear-attention>

![](_page_31_Figure_1.jpeg)

Figure 9: Learning curves of DeltaNet 340M (top left), Mamba 370M (top right) and DeltaNet 1.3B (bottom), training on 100B tokens of Fine-Web 100B. 1.3B runs required only 50k optimizer steps versus the 200k of the 340M runs due to the 4x larger batch size. All models trained stably with the same hyperparameters. Training curves were smoothed with a rolling window of 500 steps.

![](_page_31_Figure_3.jpeg)

Figure 10: Length extrapolation performance of Mamba variants on different datasets. Mamba with eigenvalue range [−1, 1] shows worse perplexity on coding and math tasks compared to the [0, 1] baseline. The dashed, vertical line indicates the training context length of 2048 tokens.

Products in Log-space We note that some diagonal models such as Mamba2 [\(Dao & Gu, 2024\)](#page-11-1), GLA [\(Yang et al., 2024a\)](#page-13-2), and mLSTM [\(Beck et al., 2024\)](#page-10-0) take advantage of the fact that all values of the state-transition matrices are positive to compute their repeated products in log-space. Our change would not allow us to do this directly, and early tests on the chunkwise parallel form of GLA showed degraded performance. Therefore, for this work, we decided to focus on Mamba and DeltaNet since they do not compute the products in log space. We mention however, that at the cost of increased computation time, it would be possible to do products in log space by converting each value in the diagonal state-transition matrix to the product of its absolute value and sign. This way, absolute values can be multiplied in log space, while products of signs are coincidentally equivalent to addition modulo 2, i.e. parity, and hence can be done stably. We leave the investigation of this approach to future work. Furthermore, we also believe that our change may be less suited for methods that use a normalized RNN state, such as mLSTM, since it might happen that the normalization term can be very close to zero due to the negative values.

# E.4.1 IMPLEMENTATION OF EXTENDED EIGENVALUE RANGE

 if constexpr (!kIsComplex) { - thread data[i] = make float2(exp2f(delta vals[r][i] \* A val[r]), + thread data[i] = make float2(2.0f \* exp2f(delta vals[r][i] \* A val[r]) - 1.0f, !kIsVariableB ? delta\_u\_vals[r][i] : B\_vals[i] \* delta\_u\_vals[r][i]); if constexpr (!Ktraits::kIsEvenLen) { if (threadIdx.x \* kNItems + i >= params.seqlen - chunk \* kChunkSize) { thread\_data[i] = make\_float2(1.f, 0.f); } } }

Figure 11: Modifications to the forward pass of the Mamba associative scan. These changes extend the eigenvalue range from [0, 1] to [−1, 1], enhancing the model's expressive capacity. Adapted from selective scan fwd [kernel.cuh.](https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan_bwd_kernel.cuh) The original implementation (in red) is replaced with an adjusted version (in green).

 - const float delta a exp = exp2f(delta vals[i] \* A scaled) + const float delta a exp = 2.0f \* exp2f(delta vals[i] \* A scaled) - 1.0f - typename Ktraits::BlockScanT(smem scan).InclusiveScan( + typename Ktraits::BlockScanT(smem scan).ExclusiveScan( thread\_data, thread\_data, SSMScanOp<weight\_t>(), prefix\_op ); - const float a = thread data[i].y - (!kIsVariableB ? delta vals[i] \* float(u vals[i]) : - delta vals[i] \* float(u vals[i]) \* B vals[i]); + float delta a exp = 2.0f \* exp2f(delta vals[i] \* A scaled) - 1.0f; + const float ddelta a exp = delta a exp + 1; + const float a = ddelta a exp \* thread data[i].y; + const float hi = delta a exp \* thread data[i].y + (!kIsVariableB ? delta vals[i] \* + float(u vals[i]) : delta vals[i] \* float(u vals[i]) \* B vals[i]); if constexpr (!kIsVariableB || !kIsVariableC) { if constexpr (!kIsVariableB) { // dBC\\_val is dB\\_val - dBC val += dout vals[i] \* (!kIsVariableC ? thread data[i].y : thread data[i].y \* C vals[i]); + dBC val += dout vals[i] \* (!kIsVariableC ? hi : hi \* C vals[i]); } else { // dBC\\_val is dC\\_val - dBC val += dout vals[i] \* thread data[i].y; + dBC val += dout vals[i] \* thread data[i].y; } } if constexpr (kIsVariableB) { dB\_vals[i] = dx \* delta\_vals[i] \* float(u\_vals[i]); } if constexpr (kIsVariableC) { - dC vals[i] = dout vals[i] \* (!kIsVariableB ? thread data[i].y \* B val : thread data[i].y); + dC vals[i] = dout vals[i] \* (!kIsVariableB ? hi \* B val : hi); }

Figure 12: Necessary changes to selective scan bwd [kernel.cuh.](https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan_bwd_kernel.cuh) The original implementation (in red) is replaced with an adjusted version (in green).

 if self.use\_beta: - beta = rearrange(self.b proj(hidden states), 'b l h -> b h l').sigmoid() + beta = 2 \* rearrange(self.b proj(hidden states), 'b l h -> b h l').sigmoid() else: beta = q.new\_ones(q.shape[0], q.shape[1], q.shape[2])

Figure 13: Simple modification to the beta calculation in DeltaNet [\(Source\)](https://github.com/sustcsonglin/flash-linear-attention/blob/3bafa4fcb505391d19cb7c47aa9bc9fa8e598b15/fla/layers/delta_net.py#L196) allowing the extension of the eigenvalues to the range [−1, 1] . The original implementation (in red) is replaced with an adjusted version (in green).