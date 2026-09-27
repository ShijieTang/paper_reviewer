# Unlocking State-Tracking In Linear Rnns Through Negative Eigenvalues

Riccardo Grazzi∗♡, Julien Siems∗♢**, Arber Zela**♢,
Jorg K.H. Franke ¨
♢, Frank Hutter♢♣**, Massimiliano Pontil**♡♠
Equal contribution∗, CSML, Istituto Italiano di Tecnologia♡, University of Freiburg♢,
ELLIS Institute Tubingen ¨
♣, AI Centre, University College London♠
riccardograzzi4@gmail.com juliensiems@gmail.com

## Abstract

Linear Recurrent Neural Networks (LRNNs) such as Mamba, RWKV, GLA, mL-
STM, and DeltaNet have emerged as efficient alternatives to Transformers for long sequences. However, both Transformers and LRNNs struggle to perform state-tracking, which may impair performance in tasks such as code evaluation. In one forward pass, current architectures are unable to solve even parity, the simplest state-tracking task, which non-linear RNNs can handle effectively. Recently, Sarrof et al. (2024) demonstrated that the failure of LRNNs like Mamba to solve parity stems from restricting the value range of their diagonal state-transition matrices to [0, 1] and that incorporating negative values can resolve this issue. We extend this result to non-diagonal LRNNs such as DeltaNet. We prove that finite precision LRNNs with state-transition matrices having only positive eigenvalues cannot solve parity, while non-triangular matrices are needed to count modulo 3. Notably, we also prove that LRNNs can learn any regular language when their state-transition matrices are products of identity minus vector outer product matrices, each with eigenvalues in the range [−1, 1]. Our experiments confirm that extending the eigenvalue range of Mamba and DeltaNet to include negative values not only enables them to solve parity but consistently improves their performance on state-tracking tasks. We also show that state-tracking enabled LRNNs can be pretrained stably and efficiently at scale (1.3B parameters), achieving competitive performance on language modeling and showing promise on code and math tasks.

## 1 Introduction

Transformer architectures (Vaswani et al., 2017) have revolutionized NLP but scale quadratically in sequence length, posing computational challenges for long sequences. To address this, Linear Recurrent Neural Networks (LRNNs) have emerged as promising alternatives that offer linear scaling while maintaining competitive performance (Gu & Dao, 2024; Dao & Gu, 2024; Yang et al., 2024a; Peng et al., 2023; Deletang et al., 2023; Sun et al., 2024; Beck et al., 2024). LRNNs update their state via matrix-vector products with structured and often input-dependent state-transition matrices. The structure of the state-transition matrices largely determines the expressivity of LRNNs. While successful models like Mamba (Gu & Dao, 2024) and GLA (Yang et al., 2024a) use diagonal matrices (diagonal LRNN) which only mix tokens along the sequence dimension, recent work explores more complex forms. Notably, non-diagonal matrices using generalized Householder (GH) transformations, defined as I − uu⊤ where u is a learnable vector and I is the identity, enable models like DeltaNet (Schlag et al., 2021; Yang et al., 2024b) and TTT-Linear (Sun et al., 2024) to achieve richer expressiveness through simultaneous token-channel mixing while maintaining efficiency.

S
caled A
cc urac y Eigenvalue Range
[0, 1] [ 1, 1]
0 10000 20000 Training Steps 0.00 0.25 0.50 0.75 1.00 Figure 1: Extending the eigenvalue range of the state transition matrices of diagonal LRNNs improves performance from random guessing (range [0, 1]) to perfect score (range
[−1, 1]) on learning parity. Trained on sequences up to length 40; Tested on lengths 40–256 (3 seeds).

Surprisingly, both Transformers and current LRNNs face a fundamental limitation: they struggle to learn how to track the state of even simple finite-state machines from sequences of state-transitions (Deletang et al., 2023). This limitation may impair performance on tasks such as entity tracking in narratives, handling nested structures in code, and reasoning tasks that can benefit from maintaining and updating an internal state over time (Merrill et al., 2024). Even the simplest state-tracking task, computing the parity of a sequence of bits, cannot be solved by modern architectures, while non-linear RNNs like LSTM (Hochreiter & Schmidhuber, 1997) and sLSTM (Beck et al., 2024) can effectively track the state of any finite state machine. However, parallelizing non-linear RNNs across the sequence length presents significant challenges (Lim et al., 2024; Gonzalez et al., 2024). Recently, Sarrof et al. (2024) demonstrated that the inability of diagonal LRNNs to solve the *parity* problem stems from the fact that the eigenvalues of their state-transition matrices are constrained to be positive. Specifically, they proved that finite precision diagonal LRNNs with exclusively positive real eigenvalues, cannot solve the parity problem in one forward pass for sequences of arbitrary length. However, their work did not provide empirical evidence showing that diagonal LRNNs with negative eigenvalues can be successfully trained to overcome this limitation. We prove that the same limitation also affects LRNNs with non-diagonal state-transition matrices, and further prove that additionally, non-triangular matrices are necessary to solve the more challenging task of modular counting (when the modulus is not a power of two). Our findings also apply to the GH matrices used by DeltaNet, as they share the same eigenvalue limitations. To overcome this, we propose a simple yet powerful solution: extend the range of possible eigenvalues from [0, 1] to [−1, 1]. This change enables state-tracking and significantly improves the expressivity of LRNNs without compromising their efficiency and training stability. As illustrated in Figure 1, it allows diagonal LRNNs to learn parity successfully. The code for part of our experiments is available at https://github.com/automl/unlocking state tracking In summary, we make the following *contributions:* 1. We prove that any finite precision LRNN with only positive real eigenvalues in the state-transition matrices (most LRNNs used in practice) cannot solve parity at arbitrary sequence lengths (Theorem 1), while non-triangular matrices are also required to learn counting modulo 3 (Theorem 2).

2. By extending the eigenvalue range, we significantly improve the state-tracking capabilities of LRNNs. We prove that LRNNs with state-transition matrices formed by products of generalized Householder (GH) matrices, each with eigenvalues in the range [−1, 1], can learn any regular language (Theorem 4), in some cases with just one layer (Theorem 3). Notably, this range extension allows LRNNs using just one GH matrix (like DeltaNet), to learn substantially harder tasks, as the repeated composition of permutations of two (over n) elements, compared to diagonal LRNNs.

3. We show that the eigenvalue range of Mamba and DeltaNet can be extended to [−1, 1] without compromising efficiency or training stability. We test the modified methods on parity, modular arithmetic, and permutation composition, demonstrating improved state-tracking performance.

4. We pre-train modified versions of DeltaNet and Mamba (up to 1.3B parameters) and show that they reach performance comparable to the original models on generative language modeling tasks, while DeltaNet shows improved perplexity on code and math datasets.

## 2 Related Work

Linear RNNs. Linear RNNs encompass state-space models and causal, linear attention mechanisms. State-space models, originally used for continuous dynamical systems, inspired LRNN variants like S4 (Gu et al., 2022) and H4 (Fu et al., 2021) (see Tiezzi et al. (2024) for a survey). Recent advancements, such as Mamba (Gu & Dao, 2024; Dao & Gu, 2024), introduced input-dependent gating of the hidden state, significantly improving language modeling performance. Concurrently, linear attention has emerged as an alternative to classical softmax attention, with Katharopoulos et al. (2020) demonstrating that causal linear attention Transformers can be reformulated as RNNs with linear scaling in sequence length. Building on this, Yang et al. (2024a) proposed Gated Linear Attention (GLA), adding a gating mechanism similar to Mamba, while DeltaNet (Schlag et al., 2021; Yang et al., 2024b) and TTT-Linear (Sun et al., 2024) explored more expressive recurrences with non-diagonal state-transition matrices. Beck et al. (2024) recently proposed xLSTM, a successor to LSTM (Hochreiter & Schmidhuber, 1997) which combines non-linear and linear RNNs. Expressivity Results. Several studies have explored the expressive power of Transformers and RNNs (see e.g. (Merrill et al., 2020; Strobl et al., 2024; Bhattamishra et al., 2024)). Here, we focus on the ones most relevant to our work. While Hahn (2020) proved that Transformers cannot model periodic languages such as parity, see also (Bhattamishra et al., 2020, Lemma C.4), and some context-free languages at arbitrary sequence lengths, Liu et al. (2023) demonstrated that Transformers can learn shortcut solutions for *solvable* finite state automata, though these solutions lack generalizability to arbitrary sequence lengths and perform poorly out-of-distribution. Unlike RNNs, the high parallelizability of Transformers prevents them from learning *unsolvable* finite state automata (Merrill & Sabharwal, 2023). These findings typically use techniques from algebraic formal language theory (we refer to Liu et al. (2023) for a short tutorial) and circuit complexity, using the log-precision assumption and a number of layers scaling linearly or logarithmically with sequence length. While earlier research established Transformers' Turing completeness, it relied on either arbitrary precision (Perez et al., 2021) or arbitrary depth and weight sharing (Giannou et al., 2023). ´ Diagonal LRNNs can simulate any RNN with infinite depth (Gu et al., 2021) and approximate regular enough functions when the state dimension grows linearly with sequence length (Orvieto et al., 2024). However, things change when depth and state size are fixed. Merrill et al. (2024) proved that finite-depth diagonal LRNNs, like Transformers, struggle to learn unsolvable finite state automata when restricted to log-precision arithmetic. The work by Fan et al. (2024) highlights a similar limitation, while in a finite precision setting, Sarrof et al. (2024) showed that diagonal LRNNs with positive values in the state-transition matrix, while capable of learning all star-free languages, cannot solve even the simple *parity* problem, a non-star-free language recognizable by an automaton with two states. However, their analysis was limited to the diagonal case and they did not test the benefit of negative eigenvalues in practice. Using a continuous time framework, also Cirone et al. (2025) pointed out the limitations of diagonal state transition matrices. Irie et al. (2021; 2023) empirically showed how state-tracking can be enabled by modifying DeltaNet as a fast weight programmer (Schmidhuber, 1992), but this makes its recurrence non-linear, hence hard to parallelize. Unlike previous work, we demonstrate that non-diagonal LRNNs like DeltaNet can achieve robust state-tracking through a minimal modification while maintaining efficient large-scale training.

## 3 Background 3.1 Linear Recurrent Neural Networks (Lrnns)

We describe LRNNs using notation inspired by Sarrof et al. (2024), focusing on the core linear recurrences while abstracting away the non-linear computations for each token. LRNNs are stacks of layers that share a common structure but have distinct learnable parameters. Each layer takes input vectors x1*, . . . ,* xt ∈ R
l(outputs of the previous layer) and outputs yˆ1*, . . . ,* yˆt ∈ R
pas:
Hi = A(xi)Hi−1 + B(xi), yˆi = dec(Hi, xi), for all i ∈ {1*, . . . , t*},
H0 ∈ C
n×d, A : R
l → C
n×n, B : R
l → C
n×d, dec : C
n×d × R
l → R
p(1)
Here, A, B and dec are learnable, generally non-linear functions, with dec usually containing a feed-forward neural network. This definition encompasses most LRNN variants, which differ in the form of A, B and dec. Table 1 illustrates how three popular LRNNs fit this framework. For other architectures see (Yang et al., 2024b, Table 4). Additional details on the notation are in Appendix A.1. Table 1: Instances of LRNN layers in (1), where αt=sigmoid(Wαxt), ∆t=softplus(W∆xt),
βt=sigmoid(w⊤
β xt), while qt, kt ∈ R
n, vt ∈ R
dare output of learnable functions of xt. Also, ψ : R
d → R
dis another learnable function usually containing an MLP and a normalization, while W1 ∈ R
n×d, W∆ ∈ R
d×l, Wα ∈ R
n×l, wβ ∈ R
land w2 ∈ R
dare learnable parameters. For simplicity, we omitted 1D convolutions. For Mamba, the matrices in the first two columns represent the recurrence for the i-th row of Ht and we set kt=(kt,1, . . . , kt,n)
⊤, W1=(w1,1*, . . . ,* w1,n)
⊤,
and l = d.

A(xt) B(xt) dec(Ht, xt)
Mamba Diag (exp (−∆t ⊙ exp(w1,i))) kt,i∆t ⊙ xt ψ(H⊤
t qt + w2 ⊙ xt)
GLA Diag (αt) ktv
⊤
t ψ(H⊤
t qt)
DeltaNet I − βtktk
⊤
t βtktv
⊤
t ψ(H⊤
t qt)
The *state-transition matrices* A(xt) are typically diagonal or generalized Householder (GH), i.e., identity minus vector outer product, as shown in Table 1, to enable efficient matrix-vector products on modern hardware. These matrices consistently have eigenvalues (and norm) in the range [0, 1].

## 3.2 Formal Language Theory

Finite State Automata and Regular Languages. A (deterministic) finite state automaton (FSA) is a tuple A = (Σ, Q, q0, δ) where Σ is a finite set of letters called alphabet, Q is a finite set of states, q0 ∈ Q is the starting state and δ : Q × Σ →Q is the state-transition function (see Hopcroft & Ullman, 2001, for an introduction). We define the set Σ
∗, whose elements are sequences called words, as the smallest superset of Σ that contains the empty word ε and is closed under word concatenation. We extend the state-transition function to δ : Q × Σ
∗ →Q by defining δ(q, ε) = q and δ(q, w) = δ(δ(q, w1 . . . wi−1), wi) for any w = w1 *. . . w*i ∈ Σ
∗ with i ≥ 2. We say that δ(q0, w) is the state that A reaches after reading the word w ∈ Σ
∗. A *language* L ⊆ Σ
∗is said to be recognized by A if there exists a recognizing set R ⊆ Q such that L = {w ∈ Σ
∗: δ(q0, w) ∈ R}. Regular languages are the ones that can be recognized by an FSA. Given an FSA A, the set T (A) = {δ(·, w) :
w ∈ Σ
∗} of functions ρ : Q →Q, together with the function composition operation forms a *monoid* called *transition monoid*, i.e. it is associative, closed and contains the identity δ(·, ε). This monoid has a finite number of elements, since |Q| < ∞. Moreover, if δ(·, w) is bijective for every w ∈ Σ, then T (A) forms a *group*, i.e. it contains the inverse of each element. State-Tracking and Monoid Word Problems. State-tracking is the problem of determining the state of a system only by observing a sequence of updates applied to it. Formally, it can be expressed as a *monoid word problem* (Merrill et al., 2024), where given a monoid (M, ·) (M is the set and ·
is the associative operation), we want to send words m1 *. . . m*t ∈ M∗, describing the sequence of updates, to their product m1 · m2 *· · ·* mt ∈ M, representing the state of the system after the updates.

If M is finite there is a corresponding FSA (*M, M, e, δ*) that solves the word problem, where the starting state is e (the identity element), and the transition function is δ(m1, m2) = m2 · m1 for m1, m2 ∈ M. In this work, we focus on group word problems, i.e. problems where the monoid is also a group. In particular, on the cyclic group Zm, i.e. addition modulo m, and the symmetric group Sm, i.e. the group of permutations on m elements. Parity is equivalent to the S2 word problem, while many state-tracking problems such as tracking chess moves or code evaluation, can be shown to be harder than the S5 word problem, which cannot be solved by Transformers and diagonal LRNNs even in log-precision for arbitrary word lengths (Merrill et al., 2024; Merrill & Sabharwal, 2023).

One LRNN Layer is an automaton. Given an alphabet Σ ⊂ N, we can view one layer of an LRNN in (1) as the automaton Alin = (Σ, H, H0, δlin), where δlin(H, w) = A(w)H + B(w),
which is extended as we saw previously1, and H = {δlin(H0, w) : w ∈ Σ
∗} ⊆ R
n×d. We say that an LRNN layer in (1) *implements* the FSA A = (Σ, Q, q0, δ) if Alin can mimic the state transitions of A2. Formally, if there exists a surjective function g : H → Q, such that for any H ∈ H, w ∈ Σ
δ(g(H), w) = g(δlin(H, w)) = g(A(w)H + B(w)). Every language L recognized by A can also be recognized by this LRNN layer with a sufficiently powerful dec. In particular if R ⊆ Q is the recognizing set for L and q0 = g(H0), then the decoder dec(Ht, wt) = 1{g(Ht) ∈ R}, will correctly determine if w ∈ L. Therefore, implementing A is at least as hard as recognizing L. A principal goal of this work is to show that current LRNNs cannot recognize simple languages such as parity (negative results) while appropriate modifications to the state-transition matrices, enable LRNNs to implement broader classes of FSA (positive results), with certain classes of FSA requiring a single layer. Note, that while LRNNs with one layer can recognize any regular language, the state transition matrices might not fit into the structure imposed by current LRNNs, such as those in Table 1 (see Appendix A.3 for more details).

## 4 Theoretical Analysis 4.1 Limitations Of Current Lrnns

In this section, we describe how positive eigenvalues and non-triangular state transition matrices limit LRNNs state-tracking capabitlies. In particular, we focus on parity and modular addition.

The parity yt ∈ {0, 1} of a sequence of ones and zeros x1 . . . xt ∈ {0, 1}
tis 1 if the total number of ones in the sequence is odd, and 0 if it's even. Equivalent to addition modulo 2, it can be computed by summing the values in the input sequence and then applying the modulo 2 function:
yt = (Pti=1 xi) mod 2. This solution can be implemented by an LRNN with one layer and scalar 1We let δlin : R
n×d × Σ → R
n×dand extend it to δlin : R
n×d × Σ
∗ → R
n×d, then we define H.

2This definition is equivalent to that of FSA homomorphism, see (Maler & Pnueli, 1994, Definition 3).

ht = A(1)ht−1 + B(1), eigs(A(1)) ≥ 0 t States converge or diverge ht = a(1)ht+1 + b(1), a(1) = −1 Parity automaton t Odd Even Odd 1 Even Clear state separation 1
states by setting A(xt) = 1, B(xt) = xt, H0 = 0, and dec(Ht, xt) = Ht mod 2 in (1). However, implementing such a solution with finite precision presents an issue: the state ht can grow indefinitely with t, eventually reaching the limit of our precision range. Indeed, ht ∈ {0*, . . . , t*}, requiring log2
(t + 1) bits for storage. Moreover, in practice dec must approximate the modulus 2 function, which is challenging to learn due to its discontinuous and periodic nature.

A more efficient solution, which implements the two-state FSA solving this problem, can still be realized by a finite precision LRNN with one layer and scalar states (and consequently also with vector states and diagonal state-transition matrices) using the recurrence ht = a(xt)ht−1 + b(xt), h0 = b(0) = 0, b(1) = a(0) = 1, a(1) = −1, yt = ht. Note that the state-transition scalar a(1) is negative, while current diagonal LRNNs do not allow negative values. (Sarrof et al., 2024, Theorem 2) states that this fact makes real-valued diagonal LRNNs unable to solve parity, which raises the question: can non-diagonal LRNNs which allow only positive eigenvalues, such as DeltaNet, solve parity? The following result answers this question negatively by generalizing Sarrof et al. (2024, Theorem 2) to non-diagonal matrices. To solve parity, the state transition matrices must allow at least one eigenvalue to be neither real nor positive. For non-diagonal matrices, this eigenvalue could simply have nonzero imaginary part. The main idea of the theorem is illustrated in Figure 2. Theorem 1 (Parity). A finite precision LRNN with finitely many layers as in (1) can solve parity for arbitrary input lengths, in particular, it can recognize the language (11)∗, only if in at least one layer, there exist x such that A(x) has at least one eigenvalue λ /∈ {x ∈ R : x ≥ 0}.

The proof in Appendix B.1 uses the same core idea as the one in (Sarrof et al., 2024, Theorem 2).

For one layer, we show that when x = 1kand the conditions for the eigenvalues of A(1) are not met, the mapping k 7→ Hk and consequently also the one k 7→ yˆk will be constant (in finite precision and for large enough k), while k 7→ yk, with yk being the parity of x, alternates between 0 and 1.

To show this, we use the expression for the powers of the Jordan canonical form of A(1). We now study the problem of counting modulo m, an easier version of addition modulo m where the input of length k never changes and is x = 1k, while the correct output is yk = (Pk i=1 xi) mod m.

The following theorem shows that to solve this problem, products of state-transition matrices must have at least one eigenvalue with nonzero imaginary part. Theorem 2 (Modular Counting). A finite precision LRNN with L layers, each as in (1), can count modulo m*, i.e. it can recognize the language* (1m)
∗, with m *not a power of two, only if there exist* i ∈ {1, . . . , L} and x1, . . . , x2i−1 such that for the i-th layer the product A(x1)A(x2)*· · ·* A(x2i−1 )
has at least one eigenvalue λ with nonzero imaginary part, i.e. λ /∈ R.

The proof is in Appendix B.2. When L = 1 a key step is to show that if A(1) has real (even negative) eigenvalues, the map k → Hk will alternate between two values (in finite precision and for large enough k), not enough to count modulo m > 2. For L > 1, we proceed by induction using our assumption on the eigenvalues of the product of state-transition matrices. Discussion Theorems 1 and 2 identify a fundamental limitation of current design choices on the structure of the state-transition matrices of LRNNs. Specifically, current LRNNs, as the ones outlined in Table 1, are incapable of solving parity, as the eigenvalues of their state-transition matrices are confined to the interval [0, 1]. Further, even if we allow negative eigenvalues, LRNNs using common structures for the state transition matrices, such as diagonal or triangular with real entries, cannot solve counting modulo m. In contrast, as we will show, LRNNs with state-transition matrices that are (products of) generalized Householder matrices, each with eigenvalues in the range [−1, 1], are much more expressive.

## 4.2 Allowing Negative Eigenvalues

We focus on two classes of LRNNs determined by the structure of their state-transition matrices: diagonal (such as Mamba, Mamba2, and GLA) and generalized Householder (GH, as in DeltaNet).

In particular, if we let s : R
l → [0, 1]n, ϕ : R
l → [0, 1] and v : R
l → R
n, being learnable functions such that ∥v(x)∥ = 1 for every x ∈ R
l, then the state transition matrices of each layer of many LRNNs, such as those in Table 1, can be written as either Adiag(x) := Diag(s(x)), or AGH(x) := I − ϕ(x)v(x)v(x)
⊤,
where Adiag(x) is diagonal with eigenvalues s(x)i ∈ [0, 1], while AGH(x) is GH with all eigenvalues equal to one except for the one associated to the eigenvector v(x), which is equal to 1 − ϕ(x) ∈ [0, 1]. To address the limitations discussed in the previous section, we propose the following modification that can be easily applied to LRNNs belonging to either class.

A−
diag(x) := Diag(2s(x)−1), A−GH(x) := I − 2ϕ(x)v(x)v(x)
⊤. (2)
Hence, A−
diag(x) has eigenvalues 2s(x)i − 1 ∈ [−1, 1] and A−GH(x) has one eigenvalue equal to 1−2ϕ(x) ∈ [−1, 1]. Thus, we have extended the eigenvalues range from [0, 1] to [−1, 1]. The norm of the matrix is still less than or equal to one, keeping the recurrence stable at long sequence lengths.

LRNNs with the modified state transition matrices can implement the solution to parity in (2) by setting s(1) = 0 and ϕ(1) = 1 so that if we consider a scalar recursion, then A−
diag(1) = −1.

However, Theorem 2 shows that we cannot count modulo 3 with triangular state transition matrices, even when allowing negative eigenvalues. Therefore, in the next section, we examine the impact of our change to the eigenvalue range on non-triangular state-transition matrices.

## 4.3 Expressivity Of Products Of Generalized Householder Matrices

We focus on state-transition matrices that are products of k GH matrices. For DeltaNet k = 1. For any *n, k* ∈ N, we define the set of all matrices in R
n×n that can be expressed as a product of k GH
matrices, each having the only interesting eigenvalue in the range Ω ⊆ R, as Mnk
(Ω) := C1C2 *· · ·* Ck : Ci = I − βiviv
⊤ i
, (1 − βi) ∈ Ω, vi ∈ R
n, ∥vi∥ = 1	. (3)
Intuitively, higher k means higher expressivity but also higher cost for matrix-vector products. Furthermore, as long as Ω ⊆ [−1, 1], the norm of the matrices is bounded by one, which guarantees that repeated matrix product do not diverge. We observe that if M ∈ Mn 1({−1}), then M is a reflection (or Householder) matrix, and that for any x ∈ R
l, AGH(x) ∈ Mn 1([0, 1])
and A−GH(x) ∈ Mn 1([−1, 1]) so that with our change we also include reflections. Moreover, Mn k
(Ω) ⊆ Mn k′ (Ω′) if Ω ⊆ Ω
′and either k
′ = k or k
′ ≥ k, 1 ∈ Ω.

Our next result shows that products of GH matrices can represent any matrix with Euclidean norm less than or equal to 1, but only when [−1, 1] ⊆ Ω. In contrast, repeated products of (e.g. upper) triangular matrices with eigenvalues in [−1, 1] remain triangular, with eigenvalues in the same range.

Proposition 1 (Expressivity of products of GH matrices). *The following hold for* Mn k in (3):
1. For any N ∈ Mn k([−1, 1]), ∥N∥ ≤ 1.

2. *For any* M ∈ R
n×n with ∥M∥≤ 1, then M ∈ Mn 3n([−1, 1]) and if M is orthogonal then M ∈ Mnn({−1, 1}), while M ∈ Mnn−1({−1, 1}) when M *is a permutation matrix.*
3. Any eigenvalue λ of any matrix N ∈ Mn k((−1, 1]) is either 1 or satisfies |λ| < 1 and if in addition N ∈ Mnk([0, 1]) and k ≤ 2*, then* λ ∈ [0, 1] ⊂ R.

The proof in Appendix C.2 uses mainly linear algebra arguments such as the SVD decomposition and the fact that every n × n orthogonal matrix can be written as a product of n reflections, due to the Cartan–Dieudonne Theorem (Gallier & Gallier, 2011). ´
A consequence of Proposition 1.3 is that LRNNs with layers of the form (1), where A : R
l →
Mn k
([0, 1]), have state transition matrices that are either the identity or not orthogonal, and hence cannot be reflections or rotations. Also, if k ≤ 2 the eigenvalues are positive and hence the LRNN
cannot learn parity due to Theorem 1. In contrast, if we allow A : R
l → Mn k
([−1, 1]) and k is large enough, the following theorem shows that an LRNN with one layer can implement any FSA whose transition monoid is a group, and that n = k = 2 is enough for cyclic groups (modular addition).

2 1 3 1 3 2 2 3 1 swap swap 0 1 0 1 0 0 0 0 1 0 1 0 0 0 1 1 0 0

× = 

1 0 0 0 0 1 0 1 0 

I − 2v1v
⊤ 1 I − 2v2v
⊤ 2
Figure 3: A permutation of k elements is also a composition of at most k−1 swaps. This maps to a product of k−1 Hoseholders, each representing a swap. Illustrated for k = 3. v
⊤
1 =
2
, − √
1
1
2
$\mathbf{\frac{1}{2},0}$), $\mathbf{v}_2^\top=\Big(0,\,\cdot\,\mathbf{\frac{1}{2}}$
$\mathbf{r}_i=-i$
, − √
1
2
.
$=\left(\frac{1}{\sqrt{2}}\right)$
Theorem 3. *Every FSA* A = (Σ, Q, q0, δ) whose transition monoid T (A) is a group, can be implemented by a finite precision LRNN with one layer and A : Σ → Mnk−1
({−1, 1}),
where n is the smallest natural number such that T (A) is isomorphic to a subgroup of Sn*, and* k = maxw∈ΣPq∈Q 1{δ(q, w) ̸= q} is the maximum number of changed states after applying a single transition. Moreover, if T (A) is isomorphic to the cyclic group Zm*, then we can set* A : Σ → M22([−1, 1]) *and if* m = 2 *(parity) we can set* A : Σ *→ {−*1, 1}.

In the proof in Appendix C.3, we map each state-transition function to a matrix representation. This can always be done using permutation matrices, but for cyclic groups, we can also use rotation matrices (Appendix C.1). For permutations, if every state-transition permutes at most k states then the corresponding permutation matrix will be in Mn k−1({−1, 1}), since it is either the identity or can be written as a product of at most k − 1 permutations of two elements (swaps), each in Mn 1({−1})
(see Figure 3). A consequence of Theorem 3 is that if every transition function of the FSA has a permutation representation corresponding to a swap or the identity, then an LRNN layer with A = A−GH, can implement it. This is useful in practice because the time complexity of an LRNN
having a product of k GH matrices as one state-transition matrix increases linearly with k. Also, for natural language tasks, the state-transitions for the FSA might be either simple or encoded using multiple letters. For example, for addition modulo 5, a word may look like "3+2+4=4" (two letters per addition). This allows an LRNN with state-transition matrices inMn 1([−1, 1]) to model complex transitions. Indeed, if each transition uses k letters and we set B ≡ 0 and A : R
l → Mn 1([−1, 1])
in (1), then the LRNN layer can model permutations that change up to k + 1 elements since Ht = C(xt, . . . , xt−k)Ht−k, C(xt*, . . . , x*t−k) := A(xt)A(xt−1)· · · A(xt−k) ∈ Mn k([−1, 1]).

In Appendix D we also show that, interestingly, an LRNN with two layers (instead of just one), each having only reflections (instead of rotations) as state-transition matrices, can solve addition modulo m. We now present an important result on the expressivity of LRNNs with multiple layers. Theorem 4. LRNNs with state transition matrices that are repeated products of GH matrices, each with eigenvalues in the range [−1, 1]*, can recognize any regular language. In particular, every FSA* A = (Σ, Q, q0, δ) *can be implemented by a finite precision LRNN with* s ≤ 2 |Q|layers, each of the form 1, where n ≤ |Q|, p ≤ s, d = 1, A : R
l → Mnn([−1, 1]) and B : R
l → N
n.

The proof in Appendix C.5 exploits the landmark Theorem by Krohn & Rhodes (1965), which states that every FSA can be decomposed as a *cascade* of simpler FSAs whose state-transition functions are either one-to-one or constant. Each layer of the LRNN will implement one FSA (with n states)
of the cascade using n × n permutation matrices, which are in Mnn−1({−1, 1}), for the one-to-one transitions, while for constant (state-independent) transitions it will set the corresponding statetransition matrix to 0 ∈ Mnn({0}) and the function B appropriately. Note that we can obtain the zero matrix only inefficiently as a product of n GH matrices, while it could also be obtained with a single diagonal matrix. This points towards LRNNs using a mix of GH and diagonal matrices, as recently explored by Gated DeltaNet (Yang et al., 2025) and RWKV-7. Discussion The results in Theorems 3 and 4 for LRNNs are in sharp contrast with the ones for Transformers (Liu et al., 2023; Merrill & Sabharwal, 2023) and diagonal LRNNs (Merrill et al., 2024), which require either the number of layers or the precision growing with the input sequence length, and can only implement an FSA if all groups in its transition monoid are *solvable*, i.e. excluding groups isomorphic to Sn with n ≥ 5. However, compared to LRNNs without any restriction to the norm of the state-transition matrices, which need only one layer to recognize any regular language, our result requires both the number of layers and the width of the LRNN to be (in the worst case) exponential in the number of states of the FSA, although we conjecture that the number of layers might be reduced to at most linear using a more refined decomposition.

## 5 Experiments

We investigate the effects of expanding the eigenvalue range of state-transition matrices from [0, 1] to [−1, 1], as explained in Section 4.2, on both synthetic tasks and language modeling. Our experiments involve Mamba, and DeltaNet, with variants trained using both the original and extended eigenvalue ranges, as shown in Table 2. We label these variants accordingly. Note that the changes increase the expressivity of Mamba and DeltaNet while coming at no additional computational cost. Detailed information on the implementation can be found in Appendix E.4.

Table 2: Summary of modifications to the state-transition matrices A(xt) to extend the eigenvalue range from [0, 1] (Table 1) to [−1, 1]. We set s(xt) = exp (−∆t exp(w1,i)).

[0, 1] [−1, 1]
Mamba Diag(s(xt)) Diag(2s(xt)−1) DeltaNet I − βtktk
⊤
t I − 2βtktk
⊤ t

## 5.1 Chomsky Hierarchy

We conducted experiments with some of the formal language tasks proposed by Deletang et al. (2023) and similarly used to benchmark xLSTM (Beck et al., 2024). Our focus was on tasks where mLSTM (an LRNN) previously underperformed while sLSTM (a non-linear RNN) succeeded, specifically parity, modular arithmetic without brackets (both regular languages) and modular arithmetic with brackets (context-free language). As in Beck et al. (2024), we trained each model with sequence lengths ranging from 3 to 40 and evaluated on lengths from 40 to 256, to assess length generalization. Note that our theoretical results cover just regular languages, excluding modular arithmetic with brackets. We compared a Transformer, mLSTM and sLSTM against two variants each of Mamba and DeltaNet - with and without eigenvalue range extension. Results Our findings, presented in Table 3, demonstrate that expanding the range of eigenvalues from [0, 1] to [−1, 1] enables all examined models to fully solve the parity task, confirming Theorem 1. For both modular arithmetic tasks, this expansion led to substantial performance improvements for Mamba and especially DeltaNet, since the latter has non-diagonal state-transition matrices that are more suited for these tasks (see Theorem 3). In Figure 6 in the Appendix, we visualize the length extrapolation performance of each model on all considered tasks. Note that we were unable to reproduce the sLSTM results reported by Beck et al. (2024) for the modular arithmetic tasks. Additional experiments and details on the tasks in Appendix E.1.

Table 3: Performance comparison of various recurrent models on formal language tasks. We report the best of 3 runs (Table 5 in the Appendix reports the median). Scores are scaled accuracy, with 1.0 indicating perfect performance and 0.0 random guessing. The positive impact of allowing negative eigenvalues ([−1, 1] range) versus restricting to positive eigenvalues ([0, 1] range) is evident for both Mamba and DeltaNet. Results in parenthesis are as reported in Beck et al. (2024).

Parity **Mod. Arithm.**
(w/o brackets)
Mod. Arithm.

(w/ brackets)
Transformer 0.022 0.031 0.067 mLSTM 0.087 (0.04) 0.040 (0.04) 0.114 (0.03)
sLSTM **1.000** (1.00) **0.787** (1.00) **0.178** (0.57)
Mamba [0, 1] 0.000 0.095 **0.123** Mamba [−1, 1] **1.000 0.241** 0.116 DeltaNet [0, 1] 0.017 0.314 0.194 DeltaNet [−1, 1] **1.000 0.971 0.260**

## 5.2 State-Tracking

We perform experiments on group word problems, relying on the code provided by Merrill et al.,
2024. We focus on the S5 group—the first *unsolvable* symmetric group where current LRNNs and Transformers are known to underperform. We also report results for addition modulo 60 (i.e., the cyclic group Z60) in Appendix E.2.2, and note that parity corresponds to S2. In these experiments, the model receives a sequence of group elements as input, and the supervision is another sequence of group elements, each representing the product of the preceding input elements. Since solving S5 might need LRNNs with state-transition matrices formed by repeated products of four GH matrices (see Theorem 3), each with eigenvalues in [−1, 1], we also consider three simplified setups: (i) allowing only permutations of up to 2 elements (identity and swaps), (ii) allowing only permutations of up to 3 elements, and (iii) using 4 tokens for each permutation. Additional details are in Ap-

10 1 10 2 Sequence Length 0 50 100 S5 only swaps DeltaNet [0,1] (1L)
DeltaNet [-1,1] (1L) DeltaNet [0,1] (5L)
DeltaNet [-1,1] (5L) Mamba [0,1] (5L) Mamba [-1,1] (5L) Full matrix simple 10 1 10 2 Sequence Length 0 50 100 S5 swaps, 3-perm.

10 1 10 2 Sequence Length (# tokens)
0 50 100 S5 4 tokens per trans.

10 1 10 2 Sequence Length 0 50 100 S5
Figure 4: Sequence accuracy for varying sequence lengths on S5 after 100 epochs of training. We report the best of 3 seeds for each method (in Figure 7 we report all seeds). The dashed vertical line indicates the sequence length used during training (32 except for the third plot from the left where it is 64). Each method is labeled with name, eigenvalue range, and number of layers. The dashed vertical line indicates the sequence length used during training. "Full matrix simple" is a one-layer baseline where the state update matrices are full and we have no control over the eigenvalue range.

CodeParrot 24 26 Math-Hard 15 16Trivia QA
16 18SlimPajama 22 24 26 Perple xity DeltaNet 340M Eigenvalue Range:
[0, 1] [-1, 1]
DeltaNet 1.3B Eigenvalue Range:
[0, 1] [-1, 1]
0 2048 Sequence Length 20 22 24 12 13 0 2048 Sequence Length 12 13 14 0 2048 Sequence Length 13 14 15 Perp lexity 0 2048 Sequence Length
pendix E.2. We stress that, even when restricting the inputs to only identity and swaps, the group elements for the supervision still cover the entire group, because swaps are generators of the group. Results Figure 4 shows that, as predicted by Theorem 3, restricting the inputs to only swap permutations allows DeltaNet [−1, 1] with even one layer to fully learn the task (since its state-transition matrices can model swaps), while DeltaNet [0, 1] with 5 layers generalizes just slightly beyond the training length. In contrast, by including also permutations of 3 elements, we notice a substantial decrease in the performance of all models. Interestingly, extending the range is still advantageous in this case and DeltaNet [−1, 1] with 5 layers reaches a good length generalization. Moreover, using 4 tokens per group element seems also beneficial compared to standard S5, since DeltaNet [−1, 1]
with 5 layers manages to extrapolate very well until around length 200, which corresponds to 50 group elements, while on standard S5 all models have 0 sequence accuracy prior to sequence length 30. We also report that Mamba, a diagonal LRNN, performs poorly on all setups, with and without increased eigenvalue range.

## 5.3 Language Modeling

Experimental Setup We train DeltaNet models with 340M and 1.3B parameters and Mamba models with 370M parameters, each using both original and extended eigenvalue ranges. Training is done on the full FineWeb-100B dataset (Penedo et al., 2024). We chose FineWeb rather than FineWeb-Edu since it contains more code. We aligned our training pipeline with Yang et al. (2024b); see Appendix E.3.1 for details. Given our previous theoretical and experimental findings, we hypothesize that models (especially DeltaNet) with extended eigenvalue range will perform better on language modeling tasks linked to state-tracking such as coding or mathematics, compared to unmodified models. To test this hypothesis, we evaluate the perplexity of these models in a length extrapolation setup using various datasets: CodeParrot (Tunstall et al., 2022) for coding, Math-Hard (Hendrycks et al., 2021) for mathematics, TriviaQA (Joshi et al., 2017), and SlimPajama (Soboleva et al., 2023). Results All models trained stably with our modification and without changing the learning rate. The validation perplexity of the proposed variants was comparable, albeit slightly worse than that of the original models throughout training (see Figure 9 in the Appendix). The experiments in Fig-

Sl

imPajama 15B*340M params*

Transformer++ 28.39 42.69 31.0 63.3 34.0 50.4 44.5 24.2 41.2 **42.2** 22.1 **21.4**

Mamba [0, 1] 28.39 39.66 30.6 65.0 **35.4** 50.1 **46.3** 23.6 41.8 12.4 23.0 2.1

GLA [0, 1] 29.47 45.53 31.3 **65.1** 33.8 51.6 44.4 **24.6** 41.8 24.0 24.7 7.3 DeltaNet [0, 1] 28.24 37.37 **32.1** 64.8 34.3 **52.2** 45.8 23.5 **42.1** 26.4 **28.9** 12.8

Fi

neWe

b 100B

340M params

DeltaNet [0, 1] 24.68 31.49 33.7 70.3 45.1 51.3 50.0 26.1 46.1 35.2 28.7 **11.8** DeltaNet [−1, 1] **24.54** 31.15 34.0 69.9 44.6 51.9 50.0 24.4 45.8 **37.2 33.1** 6.6

370M params

Mamba [0, 1] 24.84 **24.69** 35.6 **70.6 48.4** 51.2 53.4 24.8 47.3 21.6 27.7 2.8

Mamba [−1, 1] 25.02 24.71 **36.2** 70.5 47.8 **53.3 54.7 26.7 48.2** 20.9 24.8 2.5

SlimPajama 1

00B*1.3B params*

Transformer++ **16.85** 13.44 **48.9** 70.8 49.6 53.6 56.0 26.5 50.9 **66.6** 31.5 **27.4** Mamba [0, 1] 17.06 13.89 46.2 **72.2** 40.1 **54.1 59.0** 28.2 50.0 41.4 35.2 6.2 GLA [0, 1] 17.22 14.47 46.9 71.8 49.8 53.9 57.2 26.6 51.0 50.6 **42.6** 19.9 DeltaNet [0, 1] 16.87 12.21 **48.9** 71.2 **50.2** 53.6 57.2 **28.3 51.6** 49.5 37.4 17.2

FW 1

00B*1.3B params*

DeltaNet [0, 1] **18.54** 14.32 43.5 73.7 56.2 56.9 58.2 29.9 53.1 **49.1 35.1** 8.6

DeltaNet [−1, 1] 18.57 12.73 **43.7** 73.3 55.8 56.8 56.9 27.9 52.4 48.8 33.9 **12.3**

ure 5 demonstrate that on coding and math datasets, DeltaNet with an eigenvalue range of [−1, 1] achieves lower perplexity than the baseline with range [0, 1] for both model sizes. For TriviaQA, the perplexity of DeltaNet [−1, 1] is slightly higher. Note, that this is a task relying on memorization, not linked to state-tracking, and hence we do not expect an improvement. On SlimPajama, we also observe slight improvement with our modification. For Mamba instead, our modifications consistently degrades the performance on these tasks (Figure 10 in the Appendix). To ensure that our models are comparable with those obtained by Yang et al. (2024b), we evaluate them on the same benchmark tasks from lm-harness (Gao et al., 2024) in Table 4. Note, that we trained on 100B tokens of FineWeb, while Yang et al. (2024b) reported results from training on 15B and 100B tokens of SlimPajama. At 340-370M parameters, with the extended range both architectures show enhanced performance in some of the tasks: Mamba in the second subset of tasks (+2.1% average accuracy) and DeltaNet in retrieval tasks (+2% SWDE, +4.4% SQUAD). At 1.3B parameters, extending the eigenvalue range of DeltaNet shows mixed results, suggesting that the increased expressivity may need training beyond 100B tokens to fully unlock the model's capacity.

## 6 Conclusion

In this work, we showed the substantial impact of extending the eigenvalue range of state-transition matrices in LRNNs from [0, 1] to [−1, 1]. This modification provably enhances LRNN expressivity in state-tracking tasks, without adding overhead in training or inference. While Mamba successfully solves the parity problem, its diagonal matrix structure limits further gains. In contrast, DeltaNet, thanks to its non-diagonal state transition matrices which enable simultaneous token and channel mixing, excels across a broader spectrum of tasks. Our results underscore the critical role of nondiagonal state-transition matrices in augmenting state-tracking capabilities, highlighting a promising direction for future LRNN advancements. Limitations and Future work Our modification is not directly compatible with a numerical technique used by some diagonal LRNNs such as Mamba2, GLA and mLSTM. In particular, these models rely on positive state-transition matrices to compute cumulative products in log space, which improves numerical accuracy and potentially training stability (see Appendix E.4 for details). Further research is needed to assess the impact of training large-scale language models with state-tracking capabilities. To this end, we aim to understand the potential downsides of increased expressivity.

For example, we hypothesize a fundamental trade-off between state-tracking and associative recall, which is also of theoretical interest and could guide hybrid model design. Moreover, the theoretical expressivity of DeltaNet [−1, 1] with multiple layers is still unclear. We showed that it can solve addition modulo m (in Appendix D) which is equivalent to the Z3 group word problem, but we do not know if it can also solve other word problems, such as the ones for the symmetric groups Sn with n ≥ 3.

## Acknowledgments

We would like to thank David Salinas, Herilalaina Rakotoarison, Eric Alcaide, Arya Akhavan, Matia Bojovic, Erfan Mirzaei and the active members of the Flash Linear Attention discord channel for their constructive discussions and feedback. We acknowledge the support and assistance of the Data Science and Computation Facility and its Support Team, in particular Mattia Pini, in utilizing the IIT High-Performance Computing Infrastructure, on which we run our largest experiments. This research was partially supported by the following sources: PNRR MUR Project PE000013 CUP J53C22003010006 "Future Artificial Intelligence Research (FAIR)", funded by the European Union - NextGenerationEU, and EU Project ELSA under grant agreement No. 101070617. TAILOR, a project funded by EU Horizon 2020 research and innovation programme under GA No 952215; the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under grant number 417962828; the European Research Council (ERC) Consolidator Grant "Deep Learning 2.0" (grant no. 101045765). Frank Hutter acknowledges financial support by the Hector Foundation. The authors acknowledge support from ELLIS and ELIZA. Funded by the European Union. The authors gratefully acknowledge the Gauss Center for Supercomputing eV (www.gauss-centre.eu) for funding this project by providing computing time on the GCS supercomputer JUWELS at Julich ¨ Supercomputing Center (JSC). The MATH-HARD dataset which we use in one of our experiments was compiled from AoPS & the AoPS Community, MATHCOUNTS, the MAA, the Centre for Education in Mathematics and Computing, the Harvard-MIT Math Tournament, the Math Prize for Girls, MOEMS, the Mandelbrot Competition, and the Institute of Mathematics and Applications. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the ERC. Neither the European Union nor the ERC can be held responsible for them.

## References

Simran Arora, Brandon Yang, Sabri Eyuboglu, Avanika Narayan, Andrew Hojel, Immanuel Trummer, and Christopher Re. Language Models Enable Simple Systems for Generating Structured ´ Views of Heterogeneous Data Lakes. *Proceedings of the VLDB Endowment*, 17(2):92–105, 2023.

Maximilian Beck, Korbinian Poppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, ¨
Michael Kopp, Gunter Klambauer, Johannes Brandstetter, and Sepp Hochreiter. xLSTM: Ex- ¨ tended Long Short-Term Memory. In *Advances in Neural Information Processing Systems*. Curran Associates, Inc., 2024.

Satwik Bhattamishra, Kabir Ahuja, and Navin Goyal. On the ability and limitations of transformers to recognize formal languages. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 7096–7116, 2020.

Satwik Bhattamishra, Michael Hahn, Phil Blunsom, and Varun Kanade. Separations in the representational capabilities of transformers and recurrent architectures. *Advances in Neural Information* Processing Systems, 36, 2024.

Yonatan Bisk, Rowan Zellers, Ronan Le bras, Jianfeng Gao, and Yejin Choi. PIQA: Reasoning about physical commonsense in natural language. Proceedings of the AAAI Conference on Artificial Intelligence, 34(05):7432–7439, Apr. 2020.

Nicola Muca Cirone, Antonio Orvieto, Benjamin Walker, Cristopher Salvi, and Terry Lyons. Theoretical foundations of deep selective state-space models. Advances in Neural Information Processing Systems, 37:127226–127272, 2025.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? Try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Tri Dao and Albert Gu. Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. In *International Conference on Machine Learning*. PMLR, 2024.

Gregoire Deletang, Anian Ruoss, Jordi Grau-Moya, Tim Genewein, Li Kevin Wenliang, Elliot Catt, Chris Cundy, Marcus Hutter, Shane Legg, Joel Veness, et al. Neural Networks and the Chomsky Hierarchy. In *The Eleventh International Conference on Learning Representations*, 2023.

Ting-Han Fan, Ta-Chung Chi, and Alexander Rudnicky. Advancing Regular Language Reasoning in Linear Recurrent Neural Networks. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies
(Volume 2: Short Papers), pp. 45–53, 2024.

Daniel Y Fu, Tri Dao, Khaled Kamal Saab, Armin W Thomas, Atri Rudra, and Christopher Re.

Hungry Hungry Hippos: Towards Language Modeling with State Space Models. In *The Eleventh* International Conference on Learning Representations, 2021.

Jean Gallier and Jean Gallier. The Cartan–Dieudonne Theorem. ´ Geometric Methods and Applications: For Computer Science and Engineering, pp. 231–280, 2011.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 07 2024.

Angeliki Giannou, Shashank Rajput, Jy-yong Sohn, Kangwook Lee, Jason D Lee, and Dimitris Papailiopoulos. Looped transformers as programmable computers. In International Conference on Machine Learning, pp. 11398–11442. PMLR, 2023.

Xavier Gonzalez, Andrew Warrington, Jimmy T.H. Smith, and Scott Linderman. Towards Scalable and Stable Parallelization of Nonlinear RNNs. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

Albert Gu and Tri Dao. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. In First Conference on Language Modeling, 2024.

Albert Gu, Isys Johnson, Karan Goel, Khaled Saab, Tri Dao, Atri Rudra, and Christopher Re. Com- ´
bining recurrent, convolutional, and continuous-time models with linear state space layers. Advances in neural information processing systems, 34:572–585, 2021.

Albert Gu, Karan Goel, and Christopher Re. Efficiently Modeling Long Sequences with Structured State Spaces. In *International Conference on Learning Representations*, 2022.

Sylvain Gugger, Lysandre Debut, Thomas Wolf, Philipp Schmid, Zachary Mueller, Sourab Mangrulkar, Marc Sun, and Benjamin Bossan. Accelerate: Training and inference at scale made simple, efficient and adaptable. https://github.com/huggingface/accelerate, 2022.

Michael Hahn. Theoretical limitations of self-attention in neural sequence models. Transactions of the Association for Computational Linguistics, 8:156–171, 2020.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.

Sepp Hochreiter and Jurgen Schmidhuber. Long Short-Term Memory. ¨ *Neural Computation*, 9(8):
1735–1780, 1997.

John Hopcroft and Jeffrey Ullman. *Introduction to Automata Theory, Languages, and Computation*.

Addison-Wesley, 2001.

Roger A Horn and Charles R Johnson. *Matrix Analysis*. Cambridge University Press, 2012. Kazuki Irie, Imanol Schlag, Robert Csord ´ as, and J ´ urgen Schmidhuber. Going beyond linear trans- ¨
formers with recurrent fast weight programmers. *Advances in neural information processing* systems, 34:7703–7717, 2021.

Kazuki Irie, Robert Csord ´ as, and J ´ urgen Schmidhuber. Practical computational power of linear ¨
transformers and their recurrent and self-referential extensions. *arXiv preprint arXiv:2310.16076*, 2023.

Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1601–1611, 2017.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franc¸ois Fleuret. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. In International Conference on Machine Learning, pp. 5156–5165. PMLR, 2020.

Kenneth Krohn and John Rhodes. Algebraic theory of machines. i. prime decomposition theorem for finite semigroups and machines. *Transactions of the American Mathematical Society*, 116: 450–464, 1965.

Yi Heng Lim, Qi Zhu, Joshua Selfridge, and Muhammad Firmansyah Kasim. Parallelizing nonlinear sequential models over the sequence length. In *The Twelfth International Conference on* Learning Representations, 2024.

Bingbin Liu, Jordan T Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang. Transformers Learn Shortcuts to Automata. In The Eleventh International Conference on Learning Representations, 2023.

Colin Lockard, Prashant Shiralkar, and Xin Luna Dong. When open information extraction meets the semi-structured web. *NAACL-HLT. Association for Computational Linguistics*, 2019.

Ilya Loshchilov and Frank Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts. In International Conference on Learning Representations, 2017.

Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. In International Conference on Learning Representations, 2019.

Oded Maler and Amir Pnueli. On the cascaded decomposition of automata, its complexity and its application to logic. *ACTS Mobile Communication*, 48, 1994.

William Merrill and Ashish Sabharwal. The parallelism tradeoff: Limitations of log-precision transformers. *Transactions of the Association for Computational Linguistics*, 11:531–545, 2023.

William Merrill, Gail Weiss, Yoav Goldberg, Roy Schwartz, Noah A Smith, and Eran Yahav. A
Formal Hierarchy of RNN Architectures. In *Proceedings of the 58th Annual Meeting of the* Association for Computational Linguistics, pp. 443–459, 2020.

William Merrill, Jackson Petty, and Ashish Sabharwal. The Illusion of State in State-Space Models.

In *Forty-first International Conference on Machine Learning*, 2024.

Antonio Orvieto, Soham De, Caglar Gulcehre, Razvan Pascanu, and Samuel L Smith. Universality of Linear Recurrences Followed by Non-linear Projections: Finite-Width Guarantees and Benefits of Complex Eigenvalues. In *Forty-first International Conference on Machine Learning*, 2024.

Denis Paperno, German Kruszewski, Angeliki Lazaridou, Ngoc-Quan Pham, Raffaella Bernardi, ´
Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernandez. The LAMBADA dataset: ´ Word prediction requiring a broad discourse context. In *Proceedings of the 54th Annual Meeting* of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1525–1534, 2016.

Guilherme Penedo, Hynek Kydl´ıcek, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin ˇ
Raffel, Leandro Von Werra, and Thomas Wolf. The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale, 2024.

Bo Peng, Eric Alcaide, Quentin Gregory Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Nguyen Chung, Leon Derczynski, et al. RWKV:
Reinventing RNNs for the Transformer Era. In The 2023 Conference on Empirical Methods in Natural Language Processing, 2023.

Jorge Perez, Pablo Barcel ´ o, and Javier Marinkovic. Attention is turing-complete. ´ *Journal of Machine Learning Research*, 22(75):1–35, 2021.

Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In *SC20: International Conference for High Performance Computing, Networking, Storage and Analysis*, pp. 1–16. IEEE, 2020.

Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don't know: Unanswerable questions for squad. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 784–789, 2018.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. *Communications of the ACM*, 64(9):99–106, 2021.

Yash Sarrof, Yana Veitsman, and Michael Hahn. The Expressive Capacity of State Space Models:
A Formal Language Perspective. *Advances in Neural Information Processing Systems*, 2024.

Imanol Schlag, Kazuki Irie, and Jurgen Schmidhuber. Linear transformers are secretly fast weight ¨
programmers. In *International Conference on Machine Learning*, pp. 9355–9366. PMLR, 2021.

Jurgen Schmidhuber. Learning to control fast-weight memories: An alternative to dynamic recurrent ¨
networks. *Neural Computation*, 4(1):131–139, 1992.

Daria Soboleva, Faisal Al-Khateeb, Robert Myers, Jacob R Steeves, Joel Hestness, and Nolan Dey.

SlimPajama: A 627B token cleaned and deduplicated version of RedPajama, June 2023.

Lena Strobl, William Merrill, Gail Weiss, David Chiang, and Dana Angluin. What Formal Languages can Transformers express? A Survey. Transactions of the Association for Computational Linguistics, 12:543–561, 2024.

Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, et al. Learning to (learn at test time): RNNs with expressive hidden states. *arXiv preprint arXiv:2407.04620*, 2024.

Matteo Tiezzi, Michele Casoni, Alessandro Betti, Marco Gori, and Stefano Melacci. State-Space Modeling in Long Sequence Processing: A Survey on Recurrence in the Transformer Era, 2024.

Alexandre Torres. mamba.py: A simple, hackable and efficient Mamba implementation in pure PyTorch and MLX., 2024. URL https://github.com/alxndrTL/mamba.py.

Lewis Tunstall, Leandro Von Werra, and Thomas Wolf. *Natural Language Processing with Transformers*. O'Reilly Media, Inc., 2022.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is All you Need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

Songlin Yang and Yu Zhang. FLA: A Triton-Based Library for Hardware-Efficient Implementations of Linear Attention Mechanism, January 2024. URL https://github.com/
sustcsonglin/flash-linear-attention.

Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated Linear Attention Transformers with Hardware-Efficient Training. In Forty-first International Conference on Machine Learning, 2024a.

Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing Linear Transformers with the Delta Rule over Sequence Length. Advances in Neural Information Processing Systems, 36, 2024b.

Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated Delta Networks: Improving Mamba2 with Delta Rule. In *The Thirteenth International Conference on Learning Representations*, 2025.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a Machine Really Finish Your Sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 4791–4800, 2019.

## Supplementary M**Aterial**

The supplementary material is structured as follows.

- Appendix A contains additional details on the notation used, on Table 1, on the relationship between RNNs and regular languages, on the assumption of finite precision, on the states, and on the function dec.

- Appendices B and C contain the proofs for the theoretical results in Sections 4.1 and 4.3. - Appendix D contains a theorem showing that a 2 Layer LRNN having reflections as statetransition matrices can solve addition modulo m.

- Appendix E contains additional details on the experiments and additonal results.

## A Additional Background A.1 Notation

We denote with C, R, N the sets of complex, real, and natural numbers, respectively. We use lowercase letters for scalar quantities (e.g. x ∈ R), bold lowercase letters for (column) vectors (e.g.

v ∈ R
n), and bold uppercase letters for matrices (e.g. M ∈ R
n×d). Some functions with matrix
(vector) outputs, such as A and B in (1), are also bold upper (lower) case letters to emphasize the fact that they output matrices (vectors). We use ⊙ to indicate the element-wise (Hadamard) product between two vectors or matrices. We denote with ∥v∥ the Euclidean norm of the vector v ∈ R
n.

When M ∈ R
n×d, ∥M∥ also refers to the Euclidean norm, corresponding to the largest singular value. The vector ei ∈ R
n is the i-th vector of the canonical bases in R
n, i.e. the one-hot vector with 1 only in the i-th component and 0 in the others. We define the binomial coefficient for every k, j ∈ N with j ≤ k as

$${\binom{k}{0}}:=1,\quad{\binom{k}{j}}:={\frac{k(k-1)\ldots(k-j+1)}{j!}}$$
$\overline{\Sigma}$ 4. 
We also define for a Boolean s and x ∈ R

$$\mathbf{1}\{s\}:={\begin{cases}1{\mathrm{~if~}}s{\mathrm{~is~true}}\\ 0{\mathrm{~if~}}s{\mathrm{~is~false}}\end{cases}},\qquad{\mathrm{sign}}(x):={\begin{cases}1\quad&{\mathrm{if~}}x\geq0\\ -1\quad&{\mathrm{if~}}x<0\end{cases}}.$$

We define sigmoid(x) := 1/(1 + e
−x) and softplus(x) := ln(1 + e x).

We sometimes use regular expressions (see e.g. Hopcroft & Ullman, 2001), to represent their corresponding regular language. So that e.g. (11)∗ = {11}
∗, where {11} is the set containing the word 11 and ∗ is the *Kleene star* operation, is the language containing the empty word ϵ and all the words with an even number of ones, while (1m)
∗ = {1 m}
∗is the language containing the words with a number of ones divisible by m since 1 m indicates the word containing 1 repeated m times. A
language is *star-free* if it can be expressed with a regular expression that does not contain the Kleene star.

## A.2 Details Of Table 1

The Mamba recurrence in Equations 3 and 4 in (Gu & Dao, 2024) is applied independently to each channel of the input sequence. Expressing the full recurrence in the matrix-form of (1) is challenging, as it would require concatenating the rows of the matrix Ht. For simplicity, in Table 1 we write instead the recurrence for each row of Ht. In particular, Let xt ∈ R
d be the input of the layer, W∆ ∈ R
d×d, w2 ∈ R
d, W1 = (w1*, . . . ,* wn)
⊤ ∈ R
n×d be learnable parameters, qt ∈
R

n, kt = (kt,1, . . . , kt,n)
⊤ ∈ R
n be learnable functions of the input and ∆t = softplus(W∆xt).

Then, if we set Ht = (ht,1, . . . , ht,n)
⊤ ∈ R
n×dand H0 = 0, we can write the recurrence for the i-th row of Ht and the output as

$\mathbf{h}_{t,i}=\mathbf{A}_{i}(\mathbf{x}_{t})\mathbf{h}_{t-1,i}+\mathbf{B}_{i}(\mathbf{x}_{t}),\qquad\hat{\mathbf{y}}_{t}=\psi(\mathbf{H}_{t}^{\top}\mathbf{q}_{t}+\mathbf{w}_{2}\odot\mathbf{x}_{t}))$,
where Ai(xt) and Bi(xt) are the matrices stated in Table 1, i.e.

Ai(xt) := Diag (exp (−∆t ⊙ exp(w1,i))) ∈ R
d×d, Bi(xt) := kt,i∆t ⊙ xt ∈ R
d.

Alternatively, as done in (Yang et al., 2024b, Table 4), one could write the full matrix recurrence as:

One in (Tang et al., 20246, Table 4), one could write the full map  $$\boldsymbol{H}_{t}=\underbrace{\exp\left(-1\boldsymbol{\Delta}_{t}^{\top}\odot\exp(\boldsymbol{W}_{1})\right)}_{\boldsymbol{A}(\boldsymbol{x}_{t})}\odot\boldsymbol{H}_{t-1}+\underbrace{\boldsymbol{k}_{t}(\boldsymbol{\Delta}_{t}\odot\boldsymbol{x}_{t})^{\top}}_{\boldsymbol{B}(\boldsymbol{x}_{t})}.$$
A(xt)
where 1 is the vector of n ones. However, such a recurrence is not in the form (1), since we have replaced the matrix-matrix product A(xt)Ht with the element-wise product A(xt)⊙Ht. Note that we follow the implementation of B(xt) used in the official Mamba codebase, which simplifies the expression originally presented in Equation 4 of (Gu & Dao, 2024) as described by the authors in a GitHub Issue3.

A.3 REGULAR LANGUAGES AND RECURRENT NEURAL NETWORKS RNNs Can Recognize Any Regular Language A layer of a general RNN can be formulated similarly to (1) just by replacing the linear state update with a generic state-transition function g as:
ht = g(ht−1, xt), h0 ∈ R
n.

Clearly, any FSA can be implemented by an RNN layer if g is sufficiently expressive to model its state transition function. LRNNs Can Recognize Any Regular Language As explained in (Liu et al., 2023, Appendix A.2) and in the proof of (Merrill et al., 2024, Theorem 5), we can implement any FSA A =
(Σ, Q, q0, δ), and thus recognize any regular language, using matrix-vector multiplication. As a result, a single-layer LRNN by using one-hot vectors as the LRNN states and having boolean state transition matrices can recognize any language. More specifically, in (1), we can set n = |Q|, H0 = (1, 0 *. . . ,* 0)⊤ and for any letter w ∈ Σ, B(w) = 0 and A(w) ∈ R
n×n being the matrix with entries A(w)q
′,q = 1{δ(*w, q*) = q
′}. Note that in such a construction, the matrix A(w) can have norm greater than one, and enabling the state-transition matrix of LRNNs to have norm greater than one can make the recurrence unstable and is therefore never done in language models (see e.g. Table 1).

## A.4 Finite Precision

For our positive results on LRNNs expressivity (Theorems 3 and 4), by finite precision we mean that since we have a finite number of quantities involved in the computations, then there exists a finite set D ⊂ R that contains them and thus we do not require computations to be done in the reals but we can use D as datatype. In particular, D does not depend on the length of the input sequence. In practice, such data type is chosen beforehand, e.g. floating point numbers requiring a given number of bits of precision, which may not capture all quantities in our constructions.

In our negative results of Theorems 1 and 2 instead, we can pick the finite set D ⊂ R arbitrarily, e.g. floating point numbers, and we also make the use of the function cast : R → D, defined in (4). that we extend to C by applying it separately to the real and imaginary part and to vector and matrices by applying it element-wise. The cast function is used because some computations of the state of the LRNN will be allowed to be in infinite precision and then transformed to finite precision using cast as specified in the proofs. This function provides a simplification of the actual conversion that happens in practice. We believe that the finite precision setup is not only realistic but also allows a better focus on the drawbacks of modern LRNN. Note that for Transformers, results usually rely instead on the weaker notion of log-precision (Liu et al., 2023), meaning that the size of D grows logarithmically with the sequence length. This is mainly due to their limited expressivity compared to LRNNs. We also note that concerning the state-transition matrices of modern LRNNs (see Table 1), the values at the extremes of the eigenvalue range are technically not included (because of the use of the sigmoid and softplus functions). However, since we are working with finite precision, we can still include them by choosing the appropriate datatype D, which in practice includes key values such as 0, 1, and −1.

3https://github.com/state-spaces/mamba/issues/19

## A.4.1 Initial State, Matrix-Valued States, And The Decoder Function

When introducing the LRNN layer in (1), we mention that A, B and dec are learnable functions.

However, to learn the constructions in our theoretical results, we need also H0 ⊆ C
n×dto be learnable. We do this only to simplify the results, since the same effect can also be achieved by using a special token $ at the beginning of each sequence input to the model, called the beginning of sequence token and setting, H0 = 0 for each LRNN layer so that B(x1) will have the same role as the learnable H0 in our constructions. This practice is standard and used in all our experiments. While we mention that the states Ht are generally matrices of dimension n × d, for our theoretical constructions (excluding the first two theorems), we set d = 1, so that states are vector-valued. Hence, for the problems that we consider, we find that having a matrix-valued state (d > 1) brings no theoretical advantage, while it is very important for associative recall.

To compute the output yˆt from the state Ht and the vector xt of an LRNN layer in (1), we use the function dec, to abstract away the computations that are done on Ht and xt, since they are not part of the recurrence. In this work, we do not consider the internal structure of dec, but it usually contains a normalization and a feed-forward neural network and it can approximate any continuous function. In our negative results on LRNNs expressivity in Theorems 1 and 2, our choice of an arbitrary decoder guarantees the stronger results. For our positive results instead, we either do not consider the decoder (Theorem 3) or we make use of a linear decoder (Theorem 4). We point out that to recognize regular languages efficiently and with a smaller LRNN state it is beneficial to have a more powerful (non-linear) decoder, as in the case of word problems for cyclic or permutation groups. However, such a decoder may be hard to learn.

## B Parity And Modular Counting - Proofs

We report the proofs for the theorems in Section 4.1. We start by defining the function cast : R → D, for a finite set D ⊂ R, which provides a simple model for the conversion of real numbers into a finite
precision representation.
$$\operatorname{cast}(x)=\operatorname*{min}_{z\in{\mathcal{D}}_{\operatorname*{min}}}z,\quad{\mathcal{D}}_{\operatorname*{min}}:=\operatorname*{arg\,min}_{z\in\mathbb{D}}|z-x|.$$
|z − x|. (4)
Note that Dmin might not be a singleton. We naturally extend this function on complex numbers by applying it separately to the real and imaginary part, and then to complex-valued matrices by applying it element-wise. The following lemma is a key element of the proofs of Theorems 1 and 2.

There, the sequence ak in the lemma takes the form of the imaginary or real part of the elements of the k-th power of a matrix with real eigenvalues (λi will be one eigenvalue), expressed using the Jordan canonical form. See Appendix B.1 for more details on the Jordan Canonical Form.

Intuitively, the lemma shows that if some of the λi-s are negative then for k large enough, ak in finite precision will alternate between two values. Instead, if the λi-s are only nonnegative, ak in finite precision becomes constant for large enough k.

Lemma 1. Let n, m¯ ∈ N and for every k > m¯ let

  **2.1.** Let $m,m\in\mathbb{N}$ and $m\neq m$ be a sequence of integers $n$ and $m$ be a sequence of integers $n$ and $m$. Then $a_{k}:=\sum_{i=1}^{n}c_{i}\binom{k}{m_{i}}\lambda_{i}^{k-m_{i}},\quad\text{with}c_{i},\lambda_{i}\in\mathbb{R},m_{i}\in\mathbb{N},m_{i}\leq\bar{m},\quad\forall i\in\{1,\ldots,n\},$
then there exist ¯k ∈ N such that for every k ≥ ¯k there exist a¯1, a¯2 ∈ D *such that*

$$\quad(4)$$

 *for every $k\geq\bar{k}$ there exist $\bar{a}_1,\bar{a}_2\in\mathbb{D}$*. 
$$\operatorname{ast}(a_{2k+1})={\bar{a}}_{2}.$$

cast(a2k) = ¯a1, cast(a2k+1) = ¯a2.

Furthermore, if λi ≥ 0 for every i ∈ {1, . . . , n}*, then* cast(ak) = ¯a1 = ¯a2 for k ≥ ¯k.

Proof. If ci = 0 for every i, or λi = 0 for every i, then ak = 0 for all k and the statement is trivially satisfied. Without loss of generality we can assume that that ci ̸= 0 and λi ̸= 0 for every i ∈ {1*, . . . , n*}, since for each i where this is not true we can remove the corresponding term in the sum (since it will be 0) and use smaller value for n. We divide the proof into two parts. Positive powers: Assume that λi > 0 for all i ∈ {1*, . . . , n*}. This yields that for every i and every k>m¯ ,kmi λ k−mi i > 0. Since the cast function is piecewise constant with a finite number of pieces,

$$={\bar{a}}_{2}\,f o r\,k\geq{\bar{k}}.$$

we can divide the real line into a finite number of intervals where cast is constant. We now show that for k large enough, the interval where ak belongs, and hence cast(ak), does not vary with k. Without loss of generality we assume that for every i, j ∈ {1*, . . . , n*} we have that (mi, λi) ̸=
(mj , λj ), since otherwise we can factor out k mi λ k−mi iand use a smaller n. Note that kmi λ k−mi i =
k(k−1)···(k−mi+1)
mi!λ k−mi iand hence gi(k) = kmi λ k−mi ifor large k behaves like the function k miλ k, i.e. the product of a polynomial and an exponential function of k. Without loss of generality, we therefore take the order of the indices of the terms in the sum such that the functions gi are in decreasing order of growth:
λi > λj or λi = λj , mi > mj ∀i, j : *i > j.*
By factoring out g1(k), i.e. the fastest growing term, from ak we get

$$a_{k}={\binom{k}{m_{1}}}\lambda_{1}^{k-m_{1}}\left(c_{1}+b_{k}\right)\qquad b_{k}:=\sum_{i=2}^{n}c_{i}{\frac{\left({\frac{k}{m_{i}}}\right)\lambda_{i}^{k-m_{i}}}{\left({\frac{k}{m_{1}}}\right)\lambda_{1}^{k-m_{1}}}},$$

with limk→∞ bk = 0 and therefore, since for every i and every k > m¯ ,kmi λ k−mi i > 0 and c1 ̸= 0, there exist ˆk ∈ N such that for every k ≥ ˆk, sign(ak) = sign(c1 + bk) = sign(c1). Now let D = {z1*, . . . , z*d} with z1 < z2 < · · · < zd and let y1 = −∞, yd+1 = ∞ and yi = (zi−1 + zi)/2 for i ∈ {2*, . . . , d*}. From its definition, cast is a piecewise constant function such that cast(x) = zi for every x ∈ (yi, yi+1). We now consider three cases according to the values of λ1 and m1.

1) If λ1 > 1 or λ1 = 1, m1 > 0, then limk→∞ k m1 λ k−mi 1 = ∞ and there exists ¯k ≥ ˆk such that for every k ≥ ¯k, either ak > yd (if sign(c1) = 1) or ak < y2 (if sign(c1) = −1) and hence cast(ak) = ¯a ∈ {z1, zd}.

2) If λ1 < 1 then limk→∞ km1 λ k−mi 1 = 0 and hence there exist ϵ > 0, j ∈ {1*, . . . , d*},
¯k > ˆk such that for every k ≥ ¯k, ak ∈ Ω ⊆ (yj , yj+1), where Ω = (0, ϵ) if sign(c1) = 1 and Ω = (−ϵ, 0) if sign(c1) = −1. Therefore, cast(ak) = zj for every k ≥ ¯k.

3) If λ1 = 1, m1 = 0, then km1 λ k−mi 1 = 1 for every k and hence

$$a_{k}=c_{1}+b_{k},\quad b_{k}=\sum_{i=2}^{n}c_{i}{\binom{k}{m_{i}}}\lambda_{i}^{k-m_{i}}\qquad\text{with}\lambda_{i}<1\ \forall i\in\{2,\ldots,n\}\.$$

Note that bk has now the same structure as ak, just with one less term in the sum, therefore we can factor out the term λ2 m2 λ k−m2 and, since λ2 < 1, apply the same reasoning as for the second case
(λ1 < 1) to c1 + bk and prove that there exist ϵ > 0, j ∈ {1, . . . , d},¯k > ˆk such that for every k ≥ ¯k, we have that sign(bk) = sign(c2), ak ∈ Ω ⊆ (yj , yj+1), where Ω = (c1, ϵ) if sign(c2) = 1 and Ω = (−*ϵ, c*1) if sign(c2) = −1. Therefore cast(ak) = zj for every k ≥ ¯k. In summary, we proved that when λi ≥ 0 for every i, there exist a¯ ∈ D,¯k ∈ N such that for every k ≥ ¯k ak = ¯a, which concludes the first part of the proof. Some powers can be negative: Consider the general case where λi ∈ R can be negative. We can write

 $ a_k=\sum_{i=1}^n c_i\binom{k}{m_i}\text{sign}(\lambda_i)^{k-m_i}\left|\lambda_i\right|^{k-m_i}.$  $ m(x)^{2k+1-m_i}$ do not vary with $ k$ we consider. 
Since sign(x)
2k−mi and sign(x)
2k+1−mi do not vary with k we consider the two subsequences

$$a_{2k}=\sum_{i=1}^{n}\hat{c}_{i}\binom{2k}{m_{i}}|\lambda_{i}|^{2k-m_{i}},\quad\hat{c}_{i}=c_{i}\text{sign}(\lambda_{i})^{2k-m_{i}}$$ $$a_{2k+1}=\sum_{i=1}^{n}\tilde{c}_{i}\binom{2k+1}{m_{i}}|\lambda_{i}|^{2k+1-m_{i}},\quad\tilde{c}_{i}=c_{i}\text{sign}(\lambda_{i})^{2k+1-m_{i}},$$

and we can apply the same proof as for the case when λi > 0 for every i to each of the subsequences above, which gives the final result in the case λi ∈ R for every i.

## B.1 Proof Of Theorem 1

The language (11)∗contains all sequences with an even number of ones. An FSA recognizing the language, for the sequence 1 k will output yk = 1 if k is even and yk = 0 if k is odd. Consider an LRNN with one layer as in (1). We will prove that if A(1) has only nonnegative eigenvalues, then there exists a k > 0 such that for every k ≥ k, the finite precision version of the state Hk corresponding to the sequence 1 k does not depend on k and is equal to H. Hence, no matter the choice of dec, also the finite precision version of yˆk will not vary with k and thus for some k
′ ≥ ¯k, yˆk′ ̸= k
′ mod 2 = yk′ . An inductive argument can then be used for the case of LRNNs with multiple (finitely many) layers, using the fact that the input of the next layer will be constant for k large enough, as the input of the first layers. By unrolling the recursion in 1 we obtain a closed-form expression for the state

$$\mathbf{H}_{k}=\sum_{i=1}^{k-1}\left(\prod_{j=i+1}^{k-1}\mathbf{A}(\mathbf{x}_{j})\right)\mathbf{B}(\mathbf{x}_{i})+\left(\prod_{i=1}^{k}\mathbf{A}(\mathbf{x}_{i})\right)\mathbf{H}_{0},$$

where we set Qk−1 j=k A(xj ) = I to avoid clutter. We follow Merrill et al. (2024) and make the simplifying assumption that in finite precision the state at time k is computed by first evaluating all products involving the matrices A(xj ) separately and in infinite precision, followed by casting them into finite precision, and finally executing the sum also in infinite precision and casting the result in finite precision. This avoids having to deal with the individual matrix sums and products in finite precision, which would break associativity and be harder to analyze. Hence, if we set x1 *. . .* xk = 1k, we get the following exact and finite precision expressions for the state at time k.

$$\mathbf{H}_{k}=\sum_{i=0}^{k-1}\mathbf{A}(1)^{i}\mathbf{B}(1)+\mathbf{A}(1)^{k}\mathbf{H}_{0},\quad\widehat{\mathbf{H}}_{k}=\text{cast}\left(\sum_{i=0}^{k-1}\text{cost}\left(\mathbf{A}(1)^{i}\mathbf{B}(1)\right)+\text{cast}\left(\mathbf{A}(1)^{k}\mathbf{H}_{0}\right)\right),$$

where cast, defined in (4), is an operation that converts matrices with complex values element-wise into finite precision by e.g. separately converting real and imaginary parts. Using the Jordan canonical form theorem (see e.g. Horn & Johnson, 2012, Chap. 3.1), we can write A(1) = *P JP* −1, where J is block diagonal made of the Jordan blocks J1*, . . . ,* Js with s ≤ n, Ji ∈ R
ki×ki and with corresponding complex eigenvalues λ1 *. . . λ*s (with multiplicity taken into account). Such decomposition is useful because it allows, for k ≥ maxi ki − 1, to write

$$\mathbf{A}(1)^{k}=\mathbf{P}\mathbf{J}^{k}\mathbf{P}^{-1},\quad\mathbf{J}_{i}^{k}=\left[\begin{array}{ccccc}\lambda_{1}^{k}&\binom{k}{1}\lambda_{1}^{k-1}&\binom{k}{2}\lambda_{2}^{k-2}&\ldots&\binom{k}{k-1}\lambda_{i}^{k-k_{i}+1}\\ \lambda_{i}^{k}&\binom{k}{1}\lambda_{i}^{k-1}&\ldots&\binom{k}{k-2}\lambda_{i}^{k-k_{i}+2}\\ &\ddots&\ddots&\vdots&\vdots\\ &&&\ddots&\ddots&\\ &&&\lambda_{i}^{k}&\binom{k}{1}\lambda_{i}^{k-1}\\ &&&\lambda_{i}^{k}\end{array}\right].$$
.
Then, from the structure of the Jordan decomposition, the imaginary and real part of each element of the matrices A(1)kB(1) and A(1)kH0 will be a linear combination of elements of the Jordan blocks taking the same form of ak in Lemma 1. Therefore since λi ≥ 0 for every i, we can apply Lemma 1 component-wise and conclude that there exists τ ∈ N, Cb ∈ C
n×dand Db ∈ C
n×dsuch that for every k ≥ τ , Cbk = cast(A(1)kB(1)) = Cb and Db k = cast(A(1)kH0) = Db and hence

$${\widehat{\mathbf{H}}}_{k}=\operatorname{cast}\left(\sum_{i=0}^{\tau-1}{\widehat{\mathbf{C}}}_{i}+{\widehat{\mathbf{D}}}+(1-\tau){\widehat{\mathbf{C}}}+k{\widehat{\mathbf{C}}}\right)$$
.
Note that only the matrix kCb varies with k and for large enough k, the real and imaginary parts of each element of kCb will be either 0, smaller than minx∈R cast(x) or larger than maxx∈R cast(x).

Therefore, we obtain that there exists H ∈ C
n×dand ¯k ≥ τ such that for every k ≥ ¯k we have Hck = H, which concludes the proof.