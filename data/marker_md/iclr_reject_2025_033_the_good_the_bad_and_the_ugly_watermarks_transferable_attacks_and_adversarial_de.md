**006**

**009**

**019**

**024**

**029 030**

**032**

**034**

**036**

# THE GOOD, THE BAD AND THE UGLY: WATERMARKS, TRANSFERABLE ATTACKS AND ADVER-SARIAL DEFENSES

Anonymous authors Paper under double-blind review

#### ABSTRACT

We formalize and extend existing definitions of backdoor-based watermarks and adversarial defenses as *interactive protocols* between two players. The existence of these schemes is inherently tied to the learning tasks for which they are designed. Our main result shows that for *almost every* learning task, at least one of the two – a watermark or an adversarial defense – exists. The term "almost every" indicates that we also identify a third, counterintuitive but necessary option, i.e., a scheme we call a *transferable attack*. By transferable attack, we refer to an efficient algorithm computing queries that look indistinguishable from the data distribution and fool *all* efficient defenders. To this end, we prove the necessity of a transferable attack via a construction that uses a cryptographic tool called homomorphic encryption. Furthermore, we show that any task that satisfies our notion of a transferable attack implies a *cryptographic primitive*, thus requiring the underlying task to be computationally complex. These two facts imply an "*equivalence*" between the existence of transferable attacks and cryptography. Finally, we show that the class of tasks of bounded VC-dimension has an adversarial defense, and a subclass of them has a watermark.

# 1 INTRODUCTION

A company invested considerable resources to train a new classifier f. They want to open-source f but also ensure that if someone uses f, it can be detected in a black-box manner. In other words, they want to embed a *watermark* into f. [<sup>1</sup>](#page-0-0) Alice, an employee, is in charge of this project. Bob, a member of an AI Safety team, has a different task. His goal is to make f *adversarially robust*, i.e., to ensure it is hard to find queries that appear unsuspicious but cause f to make mistakes. Alice, after many unsuccessful approaches, reports to her boss that it might be inherently impossible to create a black-box watermark in f that cannot be removed. After a similar experience, Bob reports to his boss that, due to the sheer number of possible modes of attack, he was only able to produce an ever-growing, 'ugly' defense.

One day, after discussing their respective projects, Alice and Bob realized that their projects are intimately connected. Alice said that her idea was to plant a backdoor in f, creating fA, so she could later craft queries with a *hidden trigger* that activates the backdoor, causing f<sup>A</sup> to misclassify, while remaining *indistinguishable* from standard queries. By sending these tailored queries in a black-box manner to a party suspected of using fA, she can detect whether f<sup>A</sup> is being used based on the responses triggered by her backdoor. But Bob realized that his defenses were trying to render such a situation impossible. One of his ideas for defense was to take f and then "smooth" its outputs to obtain fB, aiming for robustness against attacks. Bob noticed that this procedure removes some of the backdoor-based watermarks that Alice came up with. Conversely, Alice noticed that any f with a watermark that is difficult to remove implies that some models are inherently difficult to make robust. Alice and Bob realized that their challenges are two sides of the same coin: the impossibility of one task guarantees the success of the other.

<sup>1</sup>Note that they want to watermark the model itself, not its outputs.

**059**

**061**

**064**

**067**

**069 070 071**

**074**

**079**

**089 090 091**

**094**

**104 105 106**

#### 1.1 CONTRIBUTIONS

This paper initiates a formal study of the above observation that backdoor-based watermarks and adversarial defenses span all possible scenarios. By scenarios, we refer to learning tasks that f is supposed to solve.

*Our main contribution is:*

*We prove that almost every learning task has at least one of the two: A Watermark or an Adversarial Defense.*

To do that, we formalize and extend existing definitions of watermarks and adversarial defenses, frame Alice and Bob's dynamic as a formal game, and show that this game is guaranteed to have at least one winner. Along the way to proving the main result, we identify a potential reason why this fact was not discovered earlier. There is also a third, counterintuitive but necessary option, i.e., *there are tasks with neither a Watermark nor an Adversarial Defense*.

Imagine that Alice plays the following game. The game is played with respect to a specific learning task L = (D, h), where D is the data distribution and h is the ground truth. Alice sends queries to a player and receives their responses. She wins if the responses have a lot of errors and if the player cannot distinguish them from the queries from D. Importantly, whether she wins the game depends on how much compute and data Alice and the player have. If Alice wins the game against any player having the same amount of resources as her, then we call Alice's queries a *Transferable Attack*. Intuitively, the harder a query becomes, the easier it is to distinguish it from queries from D. But this seems to indicate that it is hard to design Transferable Attacks.

However, we provably show:

- An example of a Transferable Attack defined as above. Interestingly, the example uses tools from the field of cryptography, namely Fully Homomorphic Encryption (FHE) [\(Gentry,](#page-11-0) [2009\)](#page-11-0). Notably, a Transferable Attack rules out Watermarks and Adversarial Defenses, thus constituting the third necessary option.
- That every Transferable Attack implies a certain *cryptographic primitive*, i.e., access to samples from the underlying task is enough to build essential parts of encryption systems. Thus, every task with a Transferable Attack has to be complex in the computational complexity theory sense.

Finally, we complement the above results with instantiations of Watermarks and Adversarial Defenses:

- We show the existence of an Adversarial Defense for all learning tasks with bounded Vapnik–Chervonenkis (VC) dimension, thereby ruling out Transferable Attacks in this regime.
- We give an example of a black-box Watermark for a class of learning tasks with bounded VC-dimension. Notably, in this case, both a Watermark and an Adversarial Defense exist.

# 2 RELATED WORK

This paper lies at the intersection of machine learning theory, interactive proof systems, and cryptography. We review recent advances and related contributions from these areas that closely align with our research.

Interactive Proof Systems in Machine Learning. *Interactive Proof Systems* [\(Goldwasser & Sipser,](#page-11-1) [1986\)](#page-11-1) have recently gained considerable attention in machine learning for their ability to formalize and verify complex interactions between agents, models, or even human participants. A key advancement in this area is the introduction of *Prover-Verifier Games* (PVGs) [\(Anil et al., 2021\)](#page-10-0), which employ a game-theoretic approach to guide learning agents towards decision-making with verifiable outcomes. Building on PVGs, [Kirchner et al.](#page-12-0) [\(2024\)](#page-12-0) enhance this framework to improve the legibility of Large Language Models (LLMs) outputs, making them more accessible for human evaluation. Similarly, [Wäldchen et al.](#page-14-0) [\(2024\)](#page-14-0) apply the prover-verifier setup to offer interpretability guarantees for classifiers.

**114 115**

**117**

**119**

**127**

**129 130**

**134**

**136**

![](_page_2_Diagram_1.jpeg)

Figure 1: Schematic overview of the interaction structure, along with short, informal versions of our definitions of (a) Watermark (Definition [1\)](#page-4-0), (b) Adversarial Defense (Definition [2\)](#page-4-1), and (c) Transferable Attack (Definition [3\)](#page-5-0), with (c) tied to cryptography (see Section [5\)](#page-6-0).

Extending these concepts, self-proving models [Amit et al.](#page-10-1) [\(2024\)](#page-10-1) introduce generative models that not only produce outputs but also generate proof transcripts to validate their correctness. In the context of AI safety, scalable *debate protocols* [\(Condon et al., 1993;](#page-10-2) [Irving et al., 2018;](#page-12-1) [Brown-Cohen](#page-10-3) [et al., 2023\)](#page-10-3) leverage interactive proof systems to enable complex decision processes to be broken down into verifiable components, ensuring reliability even under adversarial conditions.

Overall, these developments highlight the emerging role of interactive proof systems in addressing key aspects of AI Safety, such as interpretability, verifiability, and alignment. While current research predominantly focuses on applying this framework to improve these safety attributes, our approach takes an orthogonal direction by examining the *feasibility* of properties related to *adversarial robustness* and *backdoor-based watermarks*.

Planting Undetectable Backdoors. A key related work is presented by [Goldwasser et al.](#page-12-2) [\(2022\)](#page-12-2), which demonstrates how a learner can plant undetectable backdoors in any classifier, allowing hidden manipulation of the model's output with minimal perturbation of the input. These backdoors are activated by specific *"triggers"*, which are subtle changes to the input that cause the model to misclassify *any* input with the trigger applied, while maintaining its expected behavior on regular inputs. The authors propose two frameworks. The first utilizes digital signature schemes [\(Gold](#page-11-2)[wasser et al., 1985\)](#page-11-2) that make backdoored models indistinguishable from the original model to any computationally-bounded observer. The second involves Random Fourier Features (RFF) [\(Rahimi &](#page-13-0) [Recht, 2007\)](#page-13-0), which ensures undetectability even with full transparency of the model's weights and training data.

In a concurrent and independent work, [Christiano et al.](#page-10-4) [\(2024\)](#page-10-4) introduce a defendability framework that formalizes the interaction between an attacker planting a backdoor and a defender tasked with detecting it. The attacker modifies a classifier to alter its behavior on a trigger input while leaving other inputs unaffected. The defender then attempts to identify this trigger during evaluation, and if successful with high probability, the function class is considered defendable. The authors show an equivalence between their notion of defendability (in a computationally unbounded setting) and Probably Approximately Correct (PAC) learnability, and thus the boundedness of the VC-dimension of a class. In computationally bounded cases, they propose that *efficient defendability* serves as an important intermediate concept between efficient learnability and obfuscation. A major difference between our work and that of [Christiano et al.](#page-10-4) [\(2024\)](#page-10-4), is that in their approach, the attacker chooses the distribution, whereas we keep the distribution fixed. This makes defendability in their model harder since the attacker has more control. However, in their framework, the backdoor trigger x ∗

**166 167**

**169**

**171**

**204**

**206**

is sampled ∼ D, so the attacker does not influence it. In contrast, our model allows the attacker to choose specific x's, making defendability easier in this regard. Thus, the definitions are a priori incomparable. A second major difference is that our main result holds for *all* learning tasks, while their contributions hold only for restricted classes. This makes defendability in their model harder since the attacker has more control. However, in their framework, the backdoor trigger x ∗ is sampled ∼ D, so the attacker does not influence it. In contrast, our model allows the attacker to choose specific x's, making defendability easier in this regard. Thus, the definitions are a priori incomparable. However, there are many interesting connections. Computationally unbounded defendability is shown to be equivalent to PAC learnability, while we, in a similar spirit, show an Adversarial Defense for all tasks with bounded VC-dimension. They show that efficient PAC learnability implies efficient defendability, and we show that the same fact implies an efficient Adversarial Defense. Using cryptographic tools, they show that the class of polynomial-size circuits is not efficiently defendable, while we use different cryptographic tools to give a Transferable Attack, which rules out a Defense.

Backdoor-Based Watermarks. In black-box settings, where model auditors lack access to internal parameters, watermarking methods often involve embedding backdoors during training. Techniques by [Adi et al.](#page-10-5) [\(2018\)](#page-10-5) and [Zhang et al.](#page-14-1) [\(2018\)](#page-14-1) use crafted input patterns as triggers linked to specific outputs, enabling ownership verification by querying the model with these specific inputs. Advanced methods by [Merrer et al.](#page-13-1) [\(2017\)](#page-13-1) utilize adversarial examples, which are perturbed inputs that yield predefined outputs. Further enhancements by [Namba & Sakuma](#page-13-2) [\(2019\)](#page-13-2) focus on the robustness of watermarks, ensuring the watermark remains detectable despite model alterations or attacks.

In the domain of Natural Language Processing (NLP), backdoor-based watermarks have been studied for Pre-trained Language Models (PLMs), as exemplified by works such as [\(Gu et al., 2022;](#page-12-3) [Peng](#page-13-3) [et al., 2023\)](#page-13-3) and [\(Li et al., 2023\)](#page-12-4). These approaches embed backdoors using rare or common word triggers, ensuring watermark robustness across downstream tasks and resistance to removal techniques like fine-tuning or pruning. However, it is important to note that these lines of research are predominantly empirical, with limited theoretical exploration.

Adversarial Robustness. As we emphasize, the study of backdoors is closely related to adversarial robustness, which focuses on improving model resilience to adversarial inputs. The extensive literature in this field includes key contributions such as *adversarial training* [\(Madry et al., 2018\)](#page-12-5), which improves robustness by training on adversarial examples, and certified defenses [\(Raghunathan](#page-13-4) [et al., 2018\)](#page-13-4), which offer *provable guarantees* against adversarial attacks by ensuring prediction stability within specified perturbation bounds. Techniques like *randomized smoothing* [\(Cohen](#page-10-6) [et al., 2019\)](#page-10-6) extend these robustness guarantees. Notably, [Goldwasser et al.](#page-12-2) [\(2022\)](#page-12-2) show that some undetectable backdoors can, in fact, be removed by randomized smoothing, highlighting the intersection of adversarial robustness and backdoor methods.

# 3 WATERMARKS, ADVERSARIAL DEFENSES AND TRANSFERABLE ATTACKS

In this section, we outline interactive protocols between a verifier and a prover. Each protocol is designed to address specific tasks such as watermarking, adversarial defense, and transferable attacks. We first introduce the preliminaries before detailing the properties that each protocol must satisfy.

#### 3.1 PRELIMINARIES

Discriminative Learning Task. For n ∈ <sup>N</sup>, we define [n] := 0, 1, . . . , n − 1 . A *learning task* L is a pair (D, h) of a distribution D, supp(D) ⊆ X (the input space), and a ground truth map h: X → Y ∪ {⊥}, where Y is a finite space of labels and ⊥ represents a situation where h is not defined. To every f : X → Y, we associate err(f) := <sup>E</sup>x∼D[f(x) ̸= h(x)]. We implicitly assume h does not map to ⊥ on supp(D). This definition of ⊥ is introduced for generality, as it becomes relevant in adversarial scenarios where samples may lie outside supp(D).

For q ∈ <sup>N</sup>, x ∈ X <sup>q</sup> , y ∈ Y<sup>q</sup> , we define

$$\text{err}(\mathbf{x}, \mathbf{y}) := \frac{1}{q} \sum_{i \in [q]} \mathbb{1}_{\{h(x_i) \neq y_i, h(x_i) \neq \perp\}},$$

**224**

**236 237**

**254**

**256**

**259**

**269**

which means that we count (x, y) ∈ X × Y as an error if h is well-defined on x and h(x) ̸= y.

Advantage and Indistinguishability: For an algorithm A (also known as the distinguisher) and two distributions D0, D1, consider the following game between a sender and the distinguisher:

- 1. The sender samples a bit b ∼ U({0, 1}) and then draws a random sample x ∼ Db.
- 2. A receives x and outputs ˆb := A(x) ∈ {0, 1}. A wins if ˆb = b.

We say that δ ∈ (0, ) is the *advantage* of A for *distinguishing* D<sup>0</sup> from D<sup>1</sup> if: Pb∼U({0,1}),x∼D<sup>b</sup> [A(x) = b] = <sup>1</sup> <sup>2</sup> + δ. For a class of algorithms, we say that the two distributions D<sup>0</sup> and D<sup>1</sup> are δ-*indistinguishable* if for any algorithm in the class, its advantage is at most δ.

#### 3.2 DEFINITIONS

In our protocols, Alice (A, verifier) and Bob (B, prover) engage in interactive communication, with distinct roles depending on the specific task. Each protocol is defined with respect to a learning task L = (D, h), an error parameter ε ∈ 0, 2 , and time bounds T<sup>A</sup> and TB. A scheme is successful if the corresponding algorithm satisfies the desired properties with high probability, and we denote the set of such algorithms by SCHEME(L, ε, TA, TB), where SCHEME refers to WATERMARK, DEFENSE, or TRANSFATTACK.

### Definition 1 (*Watermark, informal*).

An algorithm AWATERMARK, running in time TA, implements a *watermarking scheme* for the learning task L with error parameter ϵ > 0, if an interactive protocol in which AWATERMARK computes a classifier f : X → Y and a sequence of queries x ∈ X <sup>q</sup> , and a prover B outputs y = B(f, x) ∈ Y q , satisfies the following properties: Alice

![](_page_4_Diagram_10.jpeg)

- Figure 2: Schematic overview of the interaction between Alice and Bob in *Watermark* (Definition [1\)](#page-4-0).
- 1. Correctness: f has low error, i.e., err(f) ≤ ϵ.
- 2. Uniqueness: There exists a prover B, running in time bounded by TA, which provides low-error answers, such that err(x, y) ≤ 2ϵ.
- 3. Unremovability: For every prover B running in time TB, it holds that err(x, y) > 2ϵ.
- 4. Undetectability: For every prover B running in time TB, the advantage of B in distinguishing the queries x generated by AWATERMARK from random queries sampled from D<sup>q</sup> is small.

Note that, due to *uniqueness*, we require that any defender, who *did not use* f and trained a model fScratch, must be accepted as a distinct model. This requirement is essential, as it mirrors real-world scenarios where independent models could have been trained within the given time constraint TA. Additionally, the property enforces that any successful Watermark must satisfy the condition that Bob's time is strictly less than TA, i.e., T<sup>B</sup> < TA.

# Definition 2 (*Adversarial Defense, informal*).

An algorithm BDEFENSE, running in time TB, implements an *adversarial defense* for the learning task L with error parameter ϵ > 0, if an interactive protocol in which BDEFENSE computes a classifier f : X → Y, a verifier A replies with x = A(f), where x ∈ X <sup>q</sup> , and BDEFENSE outputs b = BDEFENSE(f, x) ∈ {0, 1}, satisfies the following properties:

![](_page_4_Diagram_15.jpeg)

Figure 3: Schematic overview of the interaction between Alice and Bob in *Adversarial Defense* (Definition [2\)](#page-4-1).

- 1. Correctness: f has low error, i.e., err(f) ≤ ϵ.
- 2. Completeness: When x ∼ D<sup>q</sup> , then b = 0.
- 3. Soundness: For every A running in time TA, we have err(x, f(x)) ≤ 7ϵ or b = 1.

**289 290 291**

**294**

**301**

**304**

**306**

**309**

**314 315**

**318 319**

**321**

The key requirement for a successful defense is the ability to *detect when it is being tested*. To bypass the defense, an attacker must provide samples that are both *adversarial*, causing the classifier to make mistakes, and *indistinguishable* from samples drawn from the data distribution D.

#### Definition 3 (*Transferable Attack, informal*).

An algorithm ATRANSFATTACK, running in time TA, implements a *transferable attack* for the learning task L with error parameter ϵ > 0, if an interactive protocol in which ATRANSFATTACK computes x ∈ X <sup>q</sup> and B outputs y = B(x) ∈ Y<sup>q</sup> satisfies the following properties:

![](_page_5_Diagram_4.jpeg)

Figure 4: Schematic overview of the interaction between Alice and Bob in *Adversarial Defense* (Definition [3\)](#page-5-0).

- 1. Transferability: For every prover B running in time TA, we have err(x, y) > 2ϵ.
- 2. Undetectability: For every prover B running in time TB, the advantage of B in distinguishing the queries x generated by ATRANSFATTACK from random queries sampled from D<sup>q</sup> is small.

Verifiability of Watermarks. For a watermarking scheme AWATERMARK, if the *unremovability* property holds with a stronger guarantee, i.e., much larger than 2ϵ, then AWATERMARK could determine whether B had stolen f. To achieve this, AWATERMARK runs, after completing its interaction with B, the procedure guaranteed by *uniqueness* to obtain y ′ . It then verifies whether y and y ′ differ for many queries. If this condition is met, AWATERMARK concludes that B had stolen f. [<sup>2</sup>](#page-5-1) Alternatively, if *unremovability* holds with 2ϵ, as originally defined, the test described above may fail. In this scenario, we consider an external party overseeing the interaction, potentially with knowledge of the distribution and h, who can directly compute the necessary errors to make a final decision. This setup is similar to the use of human judgment oracles in [\(Brown-Cohen et al., 2023\)](#page-10-3). An interesting direction for future work would be to explore cases where the parties have access to *restricted* versions of error oracles. While this is beyond the scope of this work, we outline potential avenues for addressing this in Appendix [E.](#page-24-0)

# 4 MAIN RESULT

We are ready to state an informal version of our main theorem. Please refer to Theorem [5](#page-21-0) for the details and full proof. The key idea is to define a *zero-sum game* between A and B, where the actions of each player are the possible algorithms or circuits that can be implemented in the given time bound. Zero-sum games are not a modeling choice but a proof strategy, as they allow us to analyze the complementary nature of attacks on watermarks and adversarial defenses with clean mathematical guarantees. Specifically, the unique value of a zero-sum game eliminates concerns about equilibrium selection. Notably, this game is finite, but there are exponentially many such actions for each player. We rely on some key properties of such large zero-sum games [\(Lipton & Young, 1994b;](#page-12-6) [Lipton et al.,](#page-12-7) [2003\)](#page-12-7) to argue about our main result. The formal statement and proof is deferred to Appendix [D.](#page-20-0)

Theorem 1 (*Main Theorem, informal*). *For every learning task* L *and* ϵ ∈ 0, 2 , T ∈ N*, where a learner exists that runs in time* T *and, with high probability, learns* f *satisfying err*(f) ≤ ϵ*, at least one of these three exists:*

WATERMARK 
$$\left(\mathcal{L}, \epsilon, T, T^{1/\sqrt{\log(T)}}\right)$$
,  
 DEFENSE  $\left(\mathcal{L}, \epsilon, T^{1/\sqrt{\log(T)}}, O(T)\right)$ ,  
 TRANSFATTACK  $\left(\mathcal{L}, \epsilon, T, T\right)$ .

*Proof (Sketch).* The intuition of the proof relies on the complementary nature of Definitions [1](#page-4-0) and [2.](#page-4-1) Specifically, every attempt to remove a fixed Watermark can be transformed to a potential Adversarial

<sup>2</sup>Observe that this test *would not work*, if there were many valid labels for a given input, i.e., a situation often encountered in large language models.

**329**

**334**

**354 355 356**

**358 359**

**361**

**364**

**369**

Defense, and vice versa. We define a zero-sum game G between watermarking algorithms A and algorithms attempting to remove a watermark B. The use of a zero-sum game ensures that the value of the game is unique, allowing us to focus on the interplay between watermarking and adversarial defenses without ambiguity about equilibrium selection. The actions of each player are the class of algorithms that they can run in their respective time bounds, and the payoff is determined by the probability that the errors and rejections meet specific requirements. According to Nash's theorem, there exists a Nash equilibrium for this game, characterized by strategies ANASH and BNASH. This equilibrium framework simplifies the analysis since Nash equilibria are well-studied and provide tractable guarantees for two-player zero-sum games.

A careful analysis shows that depending on the value of the game, we have a Watermark, an Adversarial Defense, or a Transferable Attack. In the first case, where the expected payoff at the Nash equilibrium is greater than a threshold, we show there is an Adversarial Defense. We define BDEFENSE as follows. BDEFENSE first learns a low-error classifier f, then sends f to the party that is attacking the Defense, then receives queries x, and simulates (y, b) = BNASH(f, x). The bit b = 1 if BNASH thinks it is attacked. Finally, BDEFENSE replies with b ′ = 1 if b = 1, and if b = 0 it replies with b ′ = 1 if the fraction of queries on which f(x) and y differ is high. Careful analysis shows BDEFENSE is an Adversarial Defense. In the second case, where the expected payoff at the Nash equilibrium is below the threshold, we have either a Watermark or a Transferable Attack. The reason that there are two cases is due to the details of the definition of G. Full proof can be found in Appendix [D.](#page-20-0)

Our Definitions [1,](#page-4-0) [2,](#page-4-1) [3](#page-5-0) and Theorem [1](#page-5-2) are phrased with respect to a *fixed* learning task, while VC-theory takes an alternate viewpoint that tries to show guarantees on the risk (mostly sample complexity-based) for any distribution. However, for DNNs and other modern architectures, moving beyond classical VC-theory is necessary [\(Zhang et al., 2021;](#page-14-2) [Nagarajan & Kolter, 2019\)](#page-13-5). In our case, due to the requirements of our schemes (e.g., *unremovability* and *undetectability*), it may not be feasible to achieve a formalization that applies to all distributions, as in classical VC-theory. We end this section with the following observation.

Fact 1 (*Transferable Attacks are disjoint from Watermarks and Adversarial Defenses*). For every learning task L and ϵ ∈ 0, 1 2 , T <sup>∈</sup> <sup>N</sup>, if <sup>T</sup>RANSFATTACK L, ϵ, T, T exists, then neither WATERMARK (L, ϵ, T, o(T)) nor DEFENSE (L, ϵ, T, T) exists.

This result follows straightforwardly from rephrasing the Definitions [1](#page-4-0) to [3.](#page-5-0) Indeed, a Transferable Attack is a strong notion of an attack, so it rules out a Defense. Secondly, a Transferable Attack against defenders running in time T rules out a Watermark, since it is in conflict with *uniqueness*.

# 5 TRANSFERABLE ATTACKS ARE "EQUIVALENT" TO CRYPTOGRAPHY

In this section, we show that tasks with Transferable Attacks exist. To construct such examples, we use cryptographic tools. But importantly, the fact that we use cryptography is not coincidental. As a second result of this section, we show that every learning task with a Transferable Attack *implies* a certain cryptographic primitive. One can interpret this as showing that Transferable Attacks exist only for *complex learning tasks*, in the sense of computational complexity theory. The two results together justify, why we can view Transferable Attacks and the existence of cryptography as "equivalent".

# 5.1 A CRYPTOGRAPHY-BASED TASK WITH A TRANSFERABLE ATTACK

Next, we give an example of a cryptography-based learning task with a Transferable Attack. The following is an informal statement of the first theorem of this section. The formal version (Theorem [7\)](#page-26-0) is given in Appendix [G.](#page-25-0)

Theorem 2 (*Transferable Attack for a Cryptography-based Learning Task, informal*). *There exists a learning task* L *crypto with a distribution* D *and hypothesis class* H*, and* A *such that for all* ϵ *if* h *is sampled from* H *then*

$$\mathbf{A} \in \text{TRANSFATTACK} \left( (\mathcal{D}, h), \epsilon, T_{\mathbf{A}} \approx \frac{1}{\epsilon}, T_{\mathbf{B}} = \frac{1}{\epsilon^2} \right).$$

*Moreover, the learning task is such that for every* ϵ*,* ≈ ϵ *time (and* ≈ 1 ϵ *samples) is enough, and* ≈ ϵ *samples (and in particular time) is necessary to learn a classifier of error* ϵ*.*

**381**

**384**

**386**

Notably, the parameters are set so that A (the party computing x) has *less* time than B (the party computing y), specifically ≈ 1/ϵ compared to 1/ϵ<sup>2</sup> . Furthermore, because of the encryption scheme, this is a setting where a single input maps to multiple outputs, which deviates away from the setting of classification learning tasks considered in Theorem [1.](#page-5-2)

*Proof (Sketch).* We start with a definition of a learning task that will be later augmented with a cryptographic tool to produce L crypto .

Lines on Circle Learning Task L ◦ (Figure [5\)](#page-7-0). Consider a binary classification task L ◦ , where the input space is defined as X = {x ∈ <sup>R</sup> 2 | ∥x∥<sup>2</sup> = 1}, representing points on the unit circle. The hypothesis class is given by H = {h<sup>w</sup> | w ∈ <sup>R</sup> 2 , ∥w∥<sup>2</sup> = 1}, where each hypothesis is defined as hw(x) := sgn(⟨w, x⟩). The data distribution D is uniform on X , i.e., D = U(X ). Additionally, let Bw(α) := {x ∈ X | |∡(x, w)| ≤ α} denote the set of points within an angular distance up to α to w.

Fully Homomorphic Encryption (FHE) (Appendix [F\)](#page-24-1). FHE [\(Gentry, 2009\)](#page-11-0) allows for computation on encrypted data *without* decrypting it. An FHE scheme allows to encrypt x via an efficient procedure e<sup>x</sup> = FHE.ENC(x), so that later, for any algorithm C, it is possible to run C on x *homomorphically*. More concretely, it is possible to produce an encryption of the result of running C on x, i.e., eC,x := FHE.EVAL(C, ex). Finally, there is a procedure FHE.DEC that, when given a *secret key* sk, can decrypt eC,x, i.e., y := FHE.DEC(sk, eC,x), where y is the result of running C on x. Crucially, encryptions of any two messages are indistinguishable for all efficient adversaries.

Cryptography-based Learning Task L crypto (Figure [5\)](#page-7-0). L crypto is derived from *Lines on Circle Learning Task* L ◦ . Let w ∈ X . We define the distribution as an equal mixture of two parts D = <sup>2</sup>D<sup>C</sup>LEAR + <sup>2</sup>D<sup>E</sup>NC. The first part, i.e.,D<sup>C</sup>LEAR, is equal to x ∼ U(X ) with label y = hw(x). The second part, i.e.,D<sup>E</sup>NC, is equal to x ′ ∼ U(X ), y′ = hw(x ′ ),(x, y) = (FHE.ENC(x ′ ), FHE.ENC(y ′ )), which can be thought of as D<sup>C</sup>LEAR under an encryption. See Figure [5](#page-7-0) for a visual representation.

![](_page_7_Diagram_6.jpeg)

Figure 5: The left part of the figure represents a *Lines on Circle Learning Task* L ◦ with a ground truth function denoted by h. On the right, we define a *cryptography-augmented* learning task derived from L ◦ . In its distribution, a "clear" or an "encrypted" sample is observed with equal probability. Given their respective times, both A and B are able to learn a low-error classifier h <sup>A</sup>, h <sup>B</sup> respectively, by learning only on the *clear samples*. A is able to compute a Transferable Attack by computing an encryption of a point close to the decision boundary of her classifier h A.

Transferable Attack (Figure [5\)](#page-7-0). Consider the following attack strategy A. First, A collects O(1/ϵ) samples from the distribution D<sup>C</sup>LEAR and learns a classifier h A <sup>w</sup>′ ∈ H that is consistent with these samples. Since the VC-dimension of H is 2, the hypothesis h A <sup>w</sup>′ has error at most ϵ with high probability.[<sup>3</sup>](#page-7-1) Next, A samples a point xBND uniformly at random from a region close to the decision

<sup>3</sup>A can also evaluate h A <sup>w</sup>′ homomorphically (i.e., run FHE.EVAL) on FHE.ENC(x) to obtain FHE.ENC(y) of error ϵ on D<sup>E</sup>NC also. This means that A is able to learn a low-error classifier on D.

boundary of h A <sup>w</sup>′ , i.e., xBND ∼ U(Bw′ (ϵ)). Finally, with equal probability, A sets as an attack x either FHE.ENC(xBND) or a uniformly random point D<sup>C</sup>LEAR = U(X ). We claim that x [4](#page-8-0) satisfies the properties of a Transferable Attack.

Since h A <sup>w</sup>′ has low error with high probability, xBND is a uniformly random point from an arc containing the boundary of h<sup>w</sup> (see Figure [5\)](#page-7-0). The running time of B is upper-bounded by 1/ϵ<sup>2</sup> , meaning it can only learn a classifier with error ⪆ 10ϵ 2 (see Lemma [3](#page-26-1) for details). B's can only learn (Lemma [3\)](#page-26-1) a classifier of error, ⪆ 10ϵ 2 . Taking these two facts together, we expect B to misclassify x ′ with probability ≈ 2 · 10ϵ <sup>ϵ</sup> = 5ϵ > <sup>2</sup>ϵ, where the factor <sup>1</sup> 2 takes into account that we send an encrypted sample only half of the time. This implies *transferability*.

Note that x is encrypted with the same probability as in the original distribution because we send FHE.ENC(xBND) and a uniformly random x ∼ D<sup>C</sup>LEAR = U(X ) with equal probability. Crucially, FHE.ENC(xBND) is indistinguishable, for efficient adversaries, from FHE.ENC(x) for any other x ∈ X . This follows from the security of the FHE scheme. Consequently, *undetectability* holds.

Note 1. *We want to emphasize that it is crucial (for our construction) that the distribution has both an encrypted (*D<sup>E</sup>NC*) and an unencrypted part (*D<sup>C</sup>LEAR*). If there was no* D<sup>C</sup>LEAR*, then* A *would not be able to generate* FHE.ENC(xBND)*. The properties of the FHE would allow* A *to learn a low-error classifier* h A <sup>w</sup>′ *but only* under *the FHE encryption. Although* A *can produce encryptions of points of her choice, she knows* w ′ *only under encryption, so she does not know which point to encrypt! If there was no* D<sup>E</sup>NC*, then everything would happen in the clear and so* B *would be able to distinguish* x*'s that appear too close to the boundary.*

#### 5.2 TASKS WITH TRANSFERABLE ATTACKS IMPLY CRYPTOGRAPHY

In this section, we show that a Transferable Attack for any task implies a *cryptographic primitive*.

#### 5.2.1 EFID PAIRS

In cryptography, an *EFID pair* [\(Goldreich, 1990\)](#page-11-3) is a pair of distributions D0, D1, that are Efficiently samplable, statistically Far, and computationally Indistinguishable. By a seminal result [\(Goldreich,](#page-11-3) [1990\)](#page-11-3), we know that the existence of EFID pairs is equivalent to the existence of *Pseudorandom Generators* (PRG). A PRG is an efficient algorithm which stretches short seeds into longer output sequences such that the output distribution on a uniformly chosen seed is computationally indistinguishable from a uniform distribution. Together with what is known about PRGs, this implies that EFID pairs can be used for tasks in cryptography, including encryption and key generation [\(Goldreich,](#page-11-3) [1990\)](#page-11-3).

For two time bounds T, T′ we call a pair of distributions (D0, D1) a (T, T′ ) EFID pair if (i) D0, D<sup>1</sup> are samplable in time T, (ii) D0, D<sup>1</sup> are statistically far, (iii) D0, D<sup>1</sup> are indistinguishable for algorithms running in time T ′ .

#### 5.2.2 TASKS WITH TRANSFERABLE ATTACKS IMPLY EFID PAIRS

The second result of this section shows that any task with a Transferable Attack implies the existence of a type of EFID pair. The proof is deferred to Appendix [H.](#page-30-0)

Theorem 3 (*Tasks with Transferable Attacks imply EFID pairs, informal*). *For every* ϵ, T, T′ ∈ N, T ≤ T ′ *, every learning task* <sup>L</sup> *if there exists* <sup>A</sup> <sup>∈</sup> <sup>T</sup>RANSFATTACK L, ϵ, T, T′ *and there exists a learner running in time* T *that, with high probability, learns* f *such that err*(f) ≤ ϵ*, then there exists a* (T, T′ ) *EFID pair.*

# 6 TASKS WITH WATERMARKS AND ADVERSARIAL DEFENSES

In this section, we give examples of tasks with Watermarks and Adversarial Defenses. In the first example, we show that hypothesis classes of bounded VC-dimension have Adversarial Defenses against all attackers. The second example is a learning task of bounded VC-dimension that has

<sup>4</sup> In this proof sketch, we have q = 1, i.e., A sends only one x to B. This is not true for the formal scheme.

**504**

**506**

**509**

**514 515 516**

**518 519**

**524**

**529**

**539**

![](_page_9_Figure_1.jpeg)

Figure 6: Overview of the taxonomy of learning tasks, illustrating the presence of Watermarks, Adversarial Defenses, and Transferable Attacks for learning tasks of bounded VC dimension. The axes represent the time bound for the parties in the corresponding schemes. The blue regions depict positive results, the red negative, and the gray regimes of parameters which are not of interest. See Lemma [1](#page-9-0) and [2](#page-9-1) for details about blue regions. The curved line represents a potential application of Theorem [1,](#page-5-2) which says that at least one of the three points should be blue.

a Watermark, which is secure against fast adversaries. These lemmas demonstrate why the upper bounds on the running time of A and B are crucial parameters. Lemmas are proven in the appendix. The first lemma relies heavily on a result from [Goldwasser et al.](#page-12-8) [\(2020\)](#page-12-8). The authors give a defense against *arbitrary examples* in a transductive model with rejections. In contrast, our model does not allow rejections, but we do require indistinguishability. Careful analysis leads to the following result. Lemma 1 (*Adversarial Defense for bounded VC-Dimension, informal*). *Let* d ∈ N *and* H *be a binary hypothesis class on input space* X *of VC-dimension bounded by* d*. There exists an algorithm* B *such that for every* ϵ ∈ 0, 8 *,* D *over* X *and* h ∈ H *we have*

$$\mathbf{B} \in \text{DEFENSE} \left( (\mathcal{D}, h), \epsilon, T_{\mathbf{A}} = \infty, T_{\mathbf{B}} = \text{poly} \left( \frac{d}{\epsilon} \right) \right).$$

Note that, by the PAC learning bound, this is a setting of parameters, where B has enough time to learn a classifier of error ϵ. By slightly abusing the notation, we write T<sup>A</sup> = ∞, meaning that the defense is secure against *all* adversaries regardless of their running time.

Lemma 2 (*Watermark for bounded VC-Dimension against fast Adversaries, informal*). *For every* d ∈ N *there exists a distribution* D *and a binary hypothesis class* H *of VC-dimension* d *there exists* A *such that for any* ϵ ∈ 10000 <sup>d</sup><sup>2</sup> , 8 *if* h ∈ H *is taken uniformly at random from* H *then*

$$\mathbf{A} \in \text{WATERMARK} \left( (\mathcal{D}, h), \epsilon, T_{\mathbf{A}} = O \left( \frac{d}{\epsilon} \right), T_{\mathbf{B}} = \frac{d}{100} \right).$$

Note that the setting of parameters is such that A can learn (with high probability) a classifier of error ϵ, but B is *not* able to learn a low-error classifier in its allotted time t. This contrasts with Lemma [5,](#page-32-0) where B has enough time to learn. This is the regime of interest for Watermarks, where the scheme is expected to be secure against fast B's.

# 7 IMPLICATIONS FOR AI SAFETY

In contrast to years of adversarial robustness research [\(Carlini, 2024\)](#page-10-7), we conjecture that for discriminative learning tasks encountered in safety-critical regimes, an Adversarial Defense *will* exist in the future. Three pieces of evidence support this contrarian belief. (i) Theorem [1,](#page-5-2) (ii) in the securitycritical scenarios for Watermarks, the security should hold even against strong defenders, i.e., T<sup>B</sup> approaching TA. In this regime, we believe an analog of Theorem [8](#page-30-1) can be shown for Watermarks, given the similarity between the *unremovability* (Definition [1\)](#page-4-0) and *transferability* (Definition [3\)](#page-5-0) property. (iii) Transferable Attacks imply cryptography (Theorem [8\)](#page-30-1), which we suspect is rare in practical scenarios.

**554 555 556**

**559**

**561**

**564**

**569**

**579**

**584**

# REFERENCES


[1] Yossi Adi, Carsten Baum, Moustapha Cisse, Benny Pinkas, and Joseph Keshet. Turning your weakness into a strength: Watermarking deep neural networks by backdooring. In *27th USENIX Security Symposium (USENIX Security 18)*, pp. 1615–1631, 2018. Noga Amit, Shafi Goldwasser, Orr Paradise, and Guy Rothblum. Models that prove their own correctness. *arXiv preprint arXiv:2405.15722*, 2024. Maksym Andriushchenko, Francesco Croce, and Nicolas Flammarion. Jailbreaking leading safetyaligned llms with simple adaptive attacks, 2024. Cem Anil, Guodong Zhang, Yuhuai Wu, and Roger Grosse. Learning to give checkable answers with prover-verifier games. *arXiv preprint arXiv:2108.12099*, 2021. Zvika Brakerski, Craig Gentry, and Vinod Vaikuntanathan. (leveled) fully homomorphic encryption without bootstrapping. In *Proceedings of the 3rd Innovations in Theoretical Computer Science Conference*, ITCS '12, pp. 309–325, New York, NY, USA, 2012. Association for Computing Machinery. ISBN 9781450311520. doi: 10.1145/2090236.2090262. URL [https://doi.org/](https://doi.org/10.1145/2090236.2090262) [10.1145/2090236.2090262](https://doi.org/10.1145/2090236.2090262). Jonah Brown-Cohen, Geoffrey Irving, and Georgios Piliouras. Scalable ai safety via doubly-efficient debate. *arXiv preprint arXiv:2311.14125*, 2023. Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, Ilya Sutskever, and Jeffrey Wu. Weak-to-strong generalization: Eliciting strong capabilities with weak supervision. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), *Proceedings of the 41st International Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning Research*, pp. 4971–5012. PMLR, 21–27 Jul 2024. URL <https://proceedings.mlr.press/v235/burns24b.html>. Nicholas Carlini. Yet another broken defense: How AI security continues to fail, 2024. URL [https:](https://nicholas.carlini.com/writing/2024/yet-another-broken-defense.html) [//nicholas.carlini.com/writing/2024/yet-another-broken-defense.](https://nicholas.carlini.com/writing/2024/yet-another-broken-defense.html) [html](https://nicholas.carlini.com/writing/2024/yet-another-broken-defense.html). Accessed: 2024-10-02. Nicholas Carlini, Milad Nasr, Christopher A. Choquette-Choo, Matthew Jagielski, Irena Gao, Anas Awadalla, Pang Wei Koh, Daphne Ippolito, Katherine Lee, Florian Tramèr, and Ludwig Schmidt. Are aligned neural networks adversarially aligned? *ArXiv*, abs/2306.15447, 2023. URL [https:](https://api.semanticscholar.org/CorpusID:259262181) [//api.semanticscholar.org/CorpusID:259262181](https://api.semanticscholar.org/CorpusID:259262181). Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, and Eric Wong. Jailbreaking black box large language models in twenty queries, 2023. Jiefeng Chen, Yang Guo, Xi Wu, Tianqi Li, Qicheng Lao, Yingyu Liang, and Somesh Jha. Towards adversarial robustness via transductive learning. *arXiv preprint arXiv:2106.08387*, 2021. Miranda Christ, Sam Gunn, and Or Zamir. Undetectable watermarks for language models. *arXiv preprint arXiv:2306.09194*, 2023. Paul Christiano, Jacob Hilton, Victor Lecomte, and Mark Xu. Backdoor defense, learnability and obfuscation. *arXiv preprint arXiv:2409.03077*, 2024. Jeremy Cohen, Elan Rosenfeld, and Zico Kolter. Certified adversarial robustness via randomized smoothing. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), *Proceedings of the 36th International Conference on Machine Learning*, volume 97 of *Proceedings of Machine Learning Research*, pp. 1310–1320. PMLR, 09–15 Jun 2019. URL [https://proceedings.mlr.](https://proceedings.mlr.press/v97/cohen19c.html) [press/v97/cohen19c.html](https://proceedings.mlr.press/v97/cohen19c.html). Anne Condon, Joan Feigenbaum, Carsten Lund, and Peter Shor. Probabilistically checkable debate systems and approximation algorithms for pspace-hard functions. In *Proceedings of the twenty-fifth annual ACM symposium on Theory of Computing*, pp. 305–314, 1993.

[2] **604**

[3] **606**

[4] **614 615**

[5] **617**

[6] **619**

[7] **629**

[8] **634**

[9] **636**

[10] Bita Darvish Rouhani, Huili Chen, and Farinaz Koushanfar. Deepsigns: An end-to-end watermarking framework for ownership protection of deep neural networks. In *Proceedings of the twenty-fourth international conference on architectural support for programming languages and operating systems*, pp. 485–497, 2019. Yinpeng Dong, Tianyu Pang, Hang Su, and Jun Zhu. Evading defenses to transferable adversarial examples by translation-invariant attacks. *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 4307–4316, 2019. URL [https://api.semanticscholar.](https://api.semanticscholar.org/CorpusID:102350868) [org/CorpusID:102350868](https://api.semanticscholar.org/CorpusID:102350868). Logan Engstrom, Dimitris Tsipras, Ludwig Schmidt, and Aleksander Madry. A rotation and a translation suffice: Fooling cnns with simple transformations. *ArXiv*, abs/1712.02779, 2017. URL <https://api.semanticscholar.org/CorpusID:21929206>. Yousof Erfani, Ramin Pichevar, and Jean Rouat. Audio watermarking using spikegram and a twodictionary approach. *IEEE Transactions on Information Forensics and Security*, 12(4):840–852, 2017. doi: 10.1109/TIFS.2016.2636094. Pierre Fernandez, Guillaume Couairon, Hervé Jégou, Matthijs Douze, and Teddy Furon. The stable signature: Rooting watermarks in latent diffusion models. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 22466–22477, 2023.

[11] A. Gammerman, V. Vovk, and V. Vapnik. Learning by transduction. In *Proceedings of the Fourteenth Conference on Uncertainty in Artificial Intelligence*, UAI'98, pp. 148–155, San Francisco, CA, USA, 1998. Morgan Kaufmann Publishers Inc. ISBN 155860555X. Craig Gentry. Fully homomorphic encryption using ideal lattices. In *Proceedings of the Forty-First Annual ACM Symposium on Theory of Computing*, STOC '09, pp. 169–178, New York, NY, USA, 2009. Association for Computing Machinery. ISBN 9781605585062. doi: 10.1145/1536414. 1536440. URL <https://doi.org/10.1145/1536414.1536440>. Justin Gilmer, Luke Metz, Fartash Faghri, Samuel S. Schoenholz, Maithra Raghu, Martin Wattenberg, and Ian J. Goodfellow. Adversarial spheres. In *6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Workshop Track Proceedings*, 2018. URL <https://openreview.net/forum?id=SkthlLkPf>. Surbhi Goel, Sham Kakade, Adam Kalai, and Cyril Zhang. Recurrent convolutional neural networks learn succinct learning algorithms. *Advances in Neural Information Processing Systems*, 35: 7328–7341, 2022. Oded Goldreich. A note on computational indistinguishability. *Information Processing Letters*, 34(6):277–281, 1990. ISSN 0020-0190. doi: https://doi.org/10.1016/0020-0190(90) 90010-U. URL [https://www.sciencedirect.com/science/article/pii/](https://www.sciencedirect.com/science/article/pii/002001909090010U) [002001909090010U](https://www.sciencedirect.com/science/article/pii/002001909090010U). S Goldwasser and M Sipser. Private coins versus public coins in interactive proof systems. In *Proceedings of the Eighteenth Annual ACM Symposium on Theory of Computing*, STOC '86, pp. 59–68, New York, NY, USA, 1986. Association for Computing Machinery. ISBN 0897911938. doi: 10.1145/12130.12137. URL <https://doi.org/10.1145/12130.12137>. S Goldwasser, S Micali, and C Rackoff. The knowledge complexity of interactive proof-systems. In *Proceedings of the Seventeenth Annual ACM Symposium on Theory of Computing*, STOC '85, pp. 291–304, New York, NY, USA, 1985. Association for Computing Machinery. ISBN 0897911512. doi: 10.1145/22145.22178. URL <https://doi.org/10.1145/22145.22178>. Shafi Goldwasser, Yael Kalai, Raluca Ada Popa, Vinod Vaikuntanathan, and Nickolai Zeldovich. Reusable garbled circuits and succinct functional encryption. In *Proceedings of the Forty-Fifth Annual ACM Symposium on Theory of Computing*, STOC '13, pp. 555–564, New York, NY, USA, 2013. Association for Computing Machinery. ISBN 9781450320290. doi: 10.1145/2488608. 2488678. URL <https://doi.org/10.1145/2488608.2488678>.

[12] **654**

[13] **656**

[14] **659**

[15] **661**

[16] **664 665**

[17] **669**

[18] **674**

[19] **684**

[20] **686**

[21] **689 690 691**

[22] Shafi Goldwasser, Adam Tauman Kalai, Yael Tauman Kalai, and Omar Montasser. Beyond perturbations: Learning guarantees with arbitrary adversarial test examples. In *Proceedings of the 34th International Conference on Neural Information Processing Systems*, NIPS'20, Red Hook, NY, USA, 2020. Curran Associates Inc. ISBN 9781713829546. Shafi Goldwasser, Michael P. Kim, Vinod Vaikuntanathan, and Or Zamir. Planting undetectable backdoors in machine learning models. *ArXiv*, abs/2204.06974, 2022. URL [https://api.](https://api.semanticscholar.org/CorpusID:248177888) [semanticscholar.org/CorpusID:248177888](https://api.semanticscholar.org/CorpusID:248177888). Chenxi Gu, Chengsong Huang, Xiaoqing Zheng, Kai-Wei Chang, and Cho-Jui Hsieh. Watermarking pre-trained language models with backdooring. *arXiv preprint arXiv:2210.07543*, 2022. Geoffrey Irving, Paul Christiano, and Dario Amodei. Ai safety via debate, 2018. URL [https:](https://arxiv.org/abs/1805.00899) [//arxiv.org/abs/1805.00899](https://arxiv.org/abs/1805.00899). Zhengyuan Jiang, Jinghuai Zhang, and Neil Zhenqiang Gong. Evading watermark based detection of ai-generated content. *Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security*, 2023. URL [https://api.semanticscholar.org/CorpusID:](https://api.semanticscholar.org/CorpusID:258557682) [258557682](https://api.semanticscholar.org/CorpusID:258557682). John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein. A watermark for large language models. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pp. 17061–17084. PMLR, 23–29 Jul 2023. URL [https://proceedings.mlr.](https://proceedings.mlr.press/v202/kirchenbauer23a.html) [press/v202/kirchenbauer23a.html](https://proceedings.mlr.press/v202/kirchenbauer23a.html). Jan Hendrik Kirchner, Yining Chen, Harri Edwards, Jan Leike, Nat McAleese, and Yuri Burda. Prover-Verifier Games improve legibility of LLM outputs, 2024. URL [https://arxiv.org/](https://arxiv.org/abs/2407.13692) [abs/2407.13692](https://arxiv.org/abs/2407.13692). Rohith Kuditipudi, John Thickstun, Tatsunori Hashimoto, and Percy Liang. Robust distortion-free watermarks for language models. *CoRR*, abs/2307.15593, 2023. doi: 10.48550/ARXIV.2307.15593. URL <https://doi.org/10.48550/arXiv.2307.15593>. Peixuan Li, Pengzhou Cheng, Fangqi Li, Wei Du, Haodong Zhao, and Gongshen Liu. Plmmark: a secure and robust black-box watermarking framework for pre-trained language models. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 37, pp. 14991–14999, 2023. Richard J. Lipton and Neal E. Young. Simple strategies for large zero-sum games with applications to complexity theory. In *Proceedings of the Twenty-Sixth Annual ACM Symposium on Theory of Computing*, STOC '94, pp. 734–740, New York, NY, USA, 1994a. Association for Computing Machinery. ISBN 0897916638. doi: 10.1145/195058.195447. URL [https://doi.org/10.](https://doi.org/10.1145/195058.195447) [1145/195058.195447](https://doi.org/10.1145/195058.195447). Richard J Lipton and Neal E Young. Simple strategies for large zero-sum games with applications to complexity theory. In *Proceedings of the twenty-sixth annual ACM symposium on Theory of computing*, pp. 734–740, 1994b. Richard J Lipton, Evangelos Markakis, and Aranyak Mehta. Playing large games using simple strategies. In *Proceedings of the 4th ACM Conference on Electronic Commerce*, pp. 36–41, 2003. Chang Liu, Jie Zhang, Han Fang, Zehua Ma, Weiming Zhang, and Nenghai Yu. Dear: A deeplearning-based audio re-recording resilient watermarking. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 37, pp. 13201–13209, 2023. Yanpei Liu, Xinyun Chen, Chang Liu, and Dawn Song. Delving into transferable adversarial examples and black-box attacks. *arXiv preprint arXiv:1611.02770*, 2016. Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In *6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings*. OpenReview.net, 2018. URL [https://openreview.net/forum?id=](https://openreview.net/forum?id=rJzIBfZAb) [rJzIBfZAb](https://openreview.net/forum?id=rJzIBfZAb).

[23] **704**

[24] **706**

[25] **709**

[26] **721**

[27] **724**

[28] **729 730**

[29] **754**

[30] Anay Mehrotra, Manolis Zampetakis, Paul Kassianik, Blaine Nelson, Hyrum Anderson, Yaron Singer, and Amin Karbasi. Tree of attacks: Jailbreaking black-box llms automatically, 2024. Erwan Le Merrer, Patrick Pérez, and Gilles Trédan. Adversarial frontier stitching for remote neural network watermarking. *Neural Computing and Applications*, 32:9233 – 9244, 2017. URL <https://api.semanticscholar.org/CorpusID:11008755>. Omar Montasser, Steve Hanneke, and Nathan Srebro. Transductive robust learning guarantees. In *International Conference on Artificial Intelligence and Statistics*, pp. 11461–11471. PMLR, 2022. Yuki Nagai, Yusuke Uchida, Shigeyuki Sakazawa, and Shin'ichi Satoh. Digital watermarking for deep neural networks. *International Journal of Multimedia Information Retrieval*, 7:3–16, 2018. Vaishnavh Nagarajan and J Zico Kolter. Uniform convergence may be unable to explain generalization in deep learning. *Advances in Neural Information Processing Systems*, 32, 2019. Ryota Namba and Jun Sakuma. Robust watermarking of neural network with exponential weighting. *Proceedings of the 2019 ACM Asia Conference on Computer and Communications Security*, 2019. URL <https://api.semanticscholar.org/CorpusID:58028915>. Noam Nisan. Pseudorandom generators for space-bounded computations. In *Proceedings of the twenty-second annual ACM symposium on Theory of computing*, pp. 204–212, 1990. Wenjun Peng, Jingwei Yi, Fangzhao Wu, Shangxi Wu, Bin Zhu, Lingjuan Lyu, Binxing Jiao, Tong Xu, Guangzhong Sun, and Xing Xie. Are you copying my model? protecting the copyright of large language models for eaas via backdoor watermark. *arXiv preprint arXiv:2305.10036*, 2023. Aditi Raghunathan, Jacob Steinhardt, and Percy Liang. Certified defenses against adversarial examples. In *6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings*. OpenReview.net, 2018. URL <https://openreview.net/forum?id=Bys4ob-Rb>. Ali Rahimi and Benjamin Recht. Random features for large-scale kernel machines. In J. Platt, D. Koller, Y. Singer, and S. Roweis (eds.), *Advances in Neural Information Processing Systems*, volume 20. Curran Associates, Inc., 2007. URL [https://proceedings.neurips.cc/paper\\_files/paper/2007/file/](https://proceedings.neurips.cc/paper_files/paper/2007/file/013a006f03dbc5392effeb8f18fda755-Paper.pdf) [013a006f03dbc5392effeb8f18fda755-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2007/file/013a006f03dbc5392effeb8f18fda755-Paper.pdf). Oded Regev. On lattices, learning with errors, random linear codes, and cryptography. In *Proceedings of the thirty-seventh annual ACM symposium on Theory of computing*, pp. 84–93. ACM, 2005.

[31] R. Rivest, L. Adleman, and M. Dertouzos. On data banks and privacy homomorphisms. In *Foundations of Secure Computation*, pp. 169–179, New York, NY, USA, 1978. Academic Press. Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian J. Goodfellow, and Rob Fergus. Intriguing properties of neural networks. In Yoshua Bengio and Yann LeCun (eds.), *2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings*, 2014. URL [http://arxiv.org/abs/](http://arxiv.org/abs/1312.6199) [1312.6199](http://arxiv.org/abs/1312.6199). Stuart A. Thompson Tiffany Hsu. Disinformation researchers raise alarms about a.i. chatbots. <https://scottaaronson.blog/?p=6823>, 2023. Accessed: March 2024. Florian Tramer, Nicholas Carlini, Wieland Brendel, and Aleksander Madry. On adaptive attacks to adversarial example defenses. *Advances in neural information processing systems*, 33:1633–1645, 2020. Yusuke Uchida, Yuki Nagai, Shigeyuki Sakazawa, and Shin'ichi Satoh. Embedding watermarks into deep neural networks. In *Proceedings of the 2017 ACM on international conference on multimedia retrieval*, pp. 269–277, 2017. Vinod Vaikuntanathan. Computing blindfolded: New developments in fully homomorphic encryption. In *Proceedings of the 2011 IEEE 52nd Annual Symposium on Foundations of Computer Science*, FOCS '11, pp. 5–16, Washington, DC, USA, 2011. IEEE Computer Society. ISBN 9780769543001. doi: 10.1109/FOCS.2011.98. URL <https://doi.org/10.1109/FOCS.2011.98>.

[32] **756 757 759 761 764 766 769 771 772 773 774 779 780 781 784 786 787 788 789 790 791 793 794 795 796 797 798 799 804 805 806 809** Stephan Wäldchen, Kartikey Sharma, Berkant Turan, Max Zimmer, and Sebastian Pokutta. Interpretability Guarantees with Merlin-Arthur Classifiers. In *International Conference on Artificial Intelligence and Statistics*, pp. 1963–1971. PMLR, 2024. Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. Jailbroken: How does llm safety training fail? *ArXiv*, abs/2307.02483, 2023. URL [https://api.semanticscholar.org/CorpusID:](https://api.semanticscholar.org/CorpusID:259342528) [259342528](https://api.semanticscholar.org/CorpusID:259342528). Yuxin Wen, Neel Jain, John Kirchenbauer, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Hard prompts made easy: Gradient-based discrete optimization for prompt tuning and discovery. In A. Oh, T. Neumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), *Advances in Neural Information Processing Systems*, volume 36, pp. 51008–51025. Curran Associates, Inc., 2023a. URL [https://proceedings.neurips.cc/paper\\_files/paper/2023/](https://proceedings.neurips.cc/paper_files/paper/2023/file/a00548031e4647b13042c97c922fadf1-Paper-Conference.pdf) [file/a00548031e4647b13042c97c922fadf1-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/a00548031e4647b13042c97c922fadf1-Paper-Conference.pdf). Yuxin Wen, John Kirchenbauer, Jonas Geiping, and Tom Goldstein. Tree-ring watermarks: Fingerprints for diffusion images that are invisible and robust. *ArXiv*, abs/2305.20030, 2023b. URL <https://api.semanticscholar.org/CorpusID:258987524>. Eric Wong and J. Zico Kolter. Provable defenses against adversarial examples via the convex outer adversarial polytope. In Jennifer G. Dy and Andreas Krause (eds.), *Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018*, volume 80 of *Proceedings of Machine Learning Research*, pp. 5283– 5292. PMLR, 2018. URL <http://proceedings.mlr.press/v80/wong18a.html>. Yi-Hsuan Wu, Chia-Hung Yuan, and Shan-Hung Wu. Adversarial robustness via runtime masking and cleansing. In Hal Daumé III and Aarti Singh (eds.), *Proceedings of the 37th International Conference on Machine Learning*, volume 119 of *Proceedings of Machine Learning Research*, pp. 10399–10409. PMLR, 13–18 Jul 2020. URL [https://proceedings.mlr.press/v119/](https://proceedings.mlr.press/v119/wu20f.html) [wu20f.html](https://proceedings.mlr.press/v119/wu20f.html). Cihang Xie, Zhishuai Zhang, Jianyu Wang, Yuyin Zhou, Zhou Ren, and Alan Loddon Yuille. Improving transferability of adversarial examples with input diversity. *2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2725–2734, 2018. URL [https:](https://api.semanticscholar.org/CorpusID:3972825) [//api.semanticscholar.org/CorpusID:3972825](https://api.semanticscholar.org/CorpusID:3972825). Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning (still) requires rethinking generalization. *Communications of the ACM*, 64(3):107–115, 2021. Hanlin Zhang, Benjamin L. Edelman, Danilo Francati, Daniele Venturi, Giuseppe Ateniese, and Boaz Barak. Watermarks in the sand: Impossibility of strong watermarking for generative models. *arXiV*, abs/2311.04378, 2023. doi: 10.48550/ARXIV.2311.04378. URL [https:](https://doi.org/10.48550/arXiv.2311.04378) [//doi.org/10.48550/arXiv.2311.04378](https://doi.org/10.48550/arXiv.2311.04378). Jialong Zhang, Zhongshu Gu, Jiyong Jang, Hui Wu, Marc Ph. Stoecklin, Heqing Huang, and Ian Molloy. Protecting intellectual property of deep neural networks with watermarking. In *Proceedings of the 2018 on Asia Conference on Computer and Communications Security*, ASI-ACCS '18, pp. 159–172, New York, NY, USA, 2018. Association for Computing Machinery. ISBN 9781450355766. doi: 10.1145/3196494.3196550. URL [https://doi.org/10.1145/](https://doi.org/10.1145/3196494.3196550) [3196494.3196550](https://doi.org/10.1145/3196494.3196550). Xuandong Zhao, Prabhanjan Ananth, Lei Li, and Yu-Xiang Wang. Provable robust watermarking for ai-generated text. *CoRR*, abs/2306.17439, 2023a. doi: 10.48550/ARXIV.2306.17439. URL <https://doi.org/10.48550/arXiv.2306.17439>. Xuandong Zhao, Kexun Zhang, Yu-Xiang Wang, and Lei Li. Invisible image watermarks are provably removable using generative ai. 2023b. URL [https://api.semanticscholar.](https://api.semanticscholar.org/CorpusID:259075167) [org/CorpusID:259075167](https://api.semanticscholar.org/CorpusID:259075167). Yunqing Zhao, Tianyu Pang, Chao Du, Xiao Yang, Ngai-Man Cheung, and Min Lin. A recipe for watermarking diffusion models. *ArXiv*, abs/2303.10137, 2023c. URL [https://api.](https://api.semanticscholar.org/CorpusID:257622907) [semanticscholar.org/CorpusID:257622907](https://api.semanticscholar.org/CorpusID:257622907).

[33] **814 815**

[34] **817**

[35] **819**

[36] **829**

[37] **834**

[38] **836**

[39] **854**

[40] **856**

[41] Andy Zou, Zifan Wang, J. Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models. *ArXiv*, abs/2307.15043, 2023. URL [https://api.](https://api.semanticscholar.org/CorpusID:260202961) [semanticscholar.org/CorpusID:260202961](https://api.semanticscholar.org/CorpusID:260202961).
# A ADDITIONAL METHODS IN RELATED WORK

This section provides an overview of the main areas relevant to our work: Watermarking techniques, adversarial defenses, and transferable attacks on Deep Neural Networks (DNNs). Each subsection outlines important contributions and the current state of research in these areas, offering additional context and details beyond those covered in the main body

#### A.1 WATERMARKING

Watermarking techniques are crucial for protecting the intellectual property of machine learning models. These techniques can be broadly categorized based on the type of model they target. We review watermarking schemes for both discriminative and generative models, with a primary focus on discriminative models, as our work builds upon these methods.

#### A.1.1 WATERMARKING SCHEMES FOR DISCRIMINATIVE MODELS

Discriminative models, which are designed to categorize input data into predefined classes, have been a major focus of watermarking research. The key approaches in this domain can be divided into black-box and white-box approaches.

Black-Box Setting. In the black-box setting, the model owner does not have access to the internal parameters or architecture of the model, but can query the model to observe its outputs. This setting has seen the development of several watermarking techniques, primarily through backdoor-like methods.

[Adi et al.](#page-10-5) [\(2018\)](#page-10-5) and [Zhang et al.](#page-14-1) [\(2018\)](#page-14-1) proposed frameworks that embed watermarks using specifically crafted input data (e.g., unique patterns) with predefined outcomes. These watermarks can be verified by feeding these special inputs into the model and checking for the expected outputs, thereby confirming ownership.

Another significant contribution in this domain is by [Merrer et al.](#page-13-1) [\(2017\)](#page-13-1), who introduced a method that employs adversarial examples to embed the backdoor. Adversarial examples are perturbed inputs that cause the model to produce specific outputs, thus serving as a watermark.

[Namba & Sakuma](#page-13-2) [\(2019\)](#page-13-2) further enhanced the robustness of black-box watermarking schemes by developing techniques that withstand various model modifications and attacks. These methods ensure that the watermark remains intact and detectable even when the model undergoes transformations.

Provable undetectability of backdoors was achieved in the context of classification tasks by [Gold](#page-12-2)[wasser et al.](#page-12-2) [\(2022\)](#page-12-2). Unfortunately, it is known ([\(Goldwasser et al., 2022\)](#page-12-2)) that some undetectable watermarks are easily removed by simple mechanisms similar to randomized smoothing [\(Cohen et al.,](#page-10-6) [2019\)](#page-10-6).

The popularity of black-box watermarking is due to its practical applicability, as it does not require access to the model's internal workings. This makes it suitable for scenarios where models are deployed as APIs or services. Our framework builds upon these black-box watermarking techniques.

White-Box Setting. In contrast, the white-box setting assumes that the model owner has full access to the model's parameters and architecture, allowing for direct examination to confirm ownership. The initial methodologies for embedding watermarks into the weights of DNNs were introduced by [Uchida et al.](#page-13-6) [\(2017\)](#page-13-6) and [Nagai et al.](#page-13-7) [\(2018\)](#page-13-7). [Uchida et al.](#page-13-6) [\(2017\)](#page-13-6) presented a framework for embedding watermarks into the model weights, which can be examined to confirm ownership.

An advancement in white-box watermarking is provided by [Darvish Rouhani et al.](#page-11-4) [\(2019\)](#page-11-4), who developed a technique to embed an N-bit (N ≥ 1) watermark in DNNs. This technique is both *data*and *model-dependent*, meaning the watermark is activated only when specific data inputs are fed into

**869**

**874**

**884**

**886**

**889 890 891**

**904**

**906**

**909**

the model. For revealing the watermark, activations from intermediate layers are necessary in the case of white-box access, whereas only the final layer's output is needed for black-box scenarios.

Our work does not focus on white-box watermarking techniques. Instead, we concentrate on exploring the interaction between backdoor-like watermarking techniques, adversarial defenses, and transferable attacks. Overall, watermarking through backdooring has become more popular due to its applicability in the black-box setting.

#### A.1.2 WATERMARKING SCHEMES FOR GENERATIVE MODELS

Watermarking techniques for generative models have attracted considerable attention with the advent of Large Language Models (LLMs) and other advanced generative models. This increased interest has led to a surge in research and diverse contributions in this area.

Backdoor-Based Watermarking for Pre-trained Language Models. In the domain of Natural Language Processing (NLP), backdoor-based watermarks have been increasingly studied for Pretrained Language Models (PLMs), as exemplified by works such as [\(Gu et al., 2022\)](#page-12-3) and [\(Li et al.,](#page-12-4) [2023\)](#page-12-4). These methods leverage rare or common word triggers to embed watermarks, ensuring that they remain robust across downstream tasks and resilient to removal techniques like fine-tuning or pruning. While these approaches have demonstrated promising results in practical applications, they are primarily empirical, with theoretical aspects of watermarking and robustness requiring further exploration.

Watermarking the Output of LLMs. Watermarking the generated text of LLMs is critical for mitigating potential harms. Significant contributions in this domain include [\(Kirchenbauer et al.,](#page-12-9) [2023\)](#page-12-9), who proposed a watermarking framework that embeds signals into generated text that are invisible to humans but detectable algorithmically. This method promotes the use of a randomized set of "green" tokens during text generation, and detects the watermark without access to the language model API or parameters.

[Kuditipudi et al.](#page-12-10) [\(2023\)](#page-12-10) introduced robust distortion-free watermarks for language models. Their method ensures that the watermark does not distort the generated text, providing robustness against various text manipulations while maintaining the quality of the output.

[Zhao et al.](#page-14-3) [\(2023a\)](#page-14-3) presented a provable, robust watermarking technique for AI-generated text. This approach offers strong theoretical guarantees for the robustness of the watermark, making it resilient against attempts to remove or alter it without significantly changing the generated text.

However, [Zhang et al.](#page-14-4) [\(2023\)](#page-14-4) highlighted vulnerabilities in these watermarking schemes. Their work demonstrates that current watermarking techniques can be effectively broken, raising important considerations for the future development of robust and secure watermarking methods for LLMs.

Image Generation Models. Various watermarking techniques have been developed for image generation models to address ethical and legal concerns. [Fernandez et al.](#page-11-5) [\(2023\)](#page-11-5) introduced a method combining image watermarking with Latent Diffusion Models, embedding invisible watermarks in generated images for future detection. This approach is robust against modifications such as cropping. [Wen et al.](#page-14-5) [\(2023b\)](#page-14-5) proposed Tree-Ring Watermarking, which embeds a pattern into the initial noise vector during sampling, making the watermark robust to transformations like convolutions and rotations. [Jiang et al.](#page-12-11) [\(2023\)](#page-12-11) highlighted vulnerabilities in watermarking schemes, showing that human-imperceptible perturbations can evade watermark detection while maintaining visual quality. [Zhao et al.](#page-14-6) [\(2023c\)](#page-14-6) provided a comprehensive analysis of watermarking techniques for Diffusion Models, offering a recipe for efficiently watermarking models like Stable Diffusion, either through training from scratch or fine-tuning. Additionally, [Zhao et al.](#page-14-7) [\(2023b\)](#page-14-7) demonstrated that invisible watermarks are vulnerable to regeneration attacks that remove watermarks by adding random noise and reconstructing the image, suggesting a shift towards using semantically similar watermarks for better resilience.

Audio Generation Models. Watermarking techniques for audio generators have been developed for robustness against various attacks. [Erfani et al.](#page-11-6) [\(2017\)](#page-11-6) introduced a spikegram-based method, embedding watermarks in high-amplitude kernels, robust against MP3 compression and other attacks

**924**

**929**

**954**

**956**

**959**

**961**

while preserving quality. [Liu et al.](#page-12-12) [\(2023\)](#page-12-12) proposed DeAR, a deep-learning-based approach resistant to audio re-recording (AR) distortions.

#### A.2 ADVERSARIAL DEFENSE

The field of adversarial robustness has a rich and extensive literature [\(Szegedy et al., 2014;](#page-13-8) [Gilmer](#page-11-7) [et al., 2018;](#page-11-7) [Raghunathan et al., 2018;](#page-13-4) [Wong & Kolter, 2018;](#page-14-8) [Engstrom et al., 2017\)](#page-11-8). Adversarial defenses are essential for ensuring the security and reliability of machine learning models against adversarial attacks that aim to deceive them with carefully crafted inputs.

For discriminative models, there has been significant progress in developing adversarial defenses. Techniques such as adversarial training [\(Madry et al., 2018\)](#page-12-5), which involves training the model on adversarial examples, have shown promise in improving robustness. Certified defenses [\(Raghunathan](#page-13-4) [et al., 2018\)](#page-13-4) provide provable guarantees against adversarial attacks, ensuring that the model's predictions remain unchanged within a specified perturbation bound. Additionally, methods like *randomized smoothing* [\(Cohen et al., 2019\)](#page-10-6) offer robustness guarantees.

A particularly relevant work for our study is [\(Goldwasser et al., 2020\)](#page-12-8), which considers a different model for generating adversarial examples. This approach has significant implications for the robustness of watermarking techniques in the face of adversarial attacks.

In the context of Large Language Models (LLMs), there is a rapidly growing body of research focused on identifying adversarial examples [\(Zou et al., 2023;](#page-15-0) [Carlini et al., 2023;](#page-10-8) [Wen et al., 2023a\)](#page-14-9). This research is closely related to the notion of *jailbreaking* [\(Andriushchenko et al., 2024;](#page-10-9) [Chao et al.,](#page-10-10) [2023;](#page-10-10) [Mehrotra et al., 2024;](#page-13-9) [Wei et al., 2023\)](#page-14-10), which involves manipulating models to bypass their intended constraints and protections.

#### A.3 TRANSFERABLE ATTACKS AND TRANSDUCTIVE LEARNING

Transferable attacks refer to adversarial examples that are effective across multiple models. Moreover, *transductive learning* has been explored as a means to enhance adversarial robustness, and since our Definition [3](#page-5-0) captures some notion of transductive learning in the context of Transferable Attacks, we highlight significant contributions in these areas.

Adversarial Robustness via Transductive Learning. Transductive learning [\(Gammerman et al.,](#page-11-9) [1998\)](#page-11-9) has shown promise in improving the robustness of models by utilizing both training and test data during the learning process. This approach aims to make models more resilient to adversarial perturbations encountered at test time.

One significant contribution is by [Goldwasser et al.](#page-12-8) [\(2020\)](#page-12-8), which explores learning guarantees in the presence of arbitrary adversarial test examples, providing a foundational framework for transductive robustness. Another notable study by [Chen et al.](#page-10-11) [\(2021\)](#page-10-11) formalizes transductive robustness and proposes a bilevel attack objective to challenge transductive defenses, presenting both theoretical and empirical support for transductive learning's utility.

Additionally, [Montasser et al.](#page-13-10) [\(2022\)](#page-13-10) introduce a transductive learning model that adapts to perturbation complexity, achieving a robust error rate proportional to the VC dimension. The method by [Wu et al.](#page-14-11) [\(2020\)](#page-14-11) improves robustness by dynamically adjusting the network during runtime to mask gradients and cleanse non-robust features, validated through experimental results. Lastly, [Tramer](#page-13-11) [et al.](#page-13-11) [\(2020\)](#page-13-11) critique the standard of adaptive attacks, demonstrating the need for specific tuning to effectively evaluate and enhance adversarial defenses.

Transferable Attacks on DNNs. Transferable attacks exploit the vulnerability of models to adversarial examples that generalize across different models. For discriminative models, significant works include [Liu et al.](#page-12-13) [\(2016\)](#page-12-13), which investigates the transferability of adversarial examples and their effectiveness in black-box attack scenarios, [\(Xie et al., 2018\)](#page-14-12), who propose input diversity techniques to enhance the transferability of adversarial examples across different models, and [\(Dong et al.,](#page-11-10) [2019\)](#page-11-10), which presents translation-invariant attacks to evade defenses and improve the effectiveness of transferable adversarial examples.

**979**

**989 990 991**

**994**

**1011**

**1014 1015**

**1017**

In the context of generative models, including large language models (LLMs) and other advanced generative architectures, relevant research is rapidly emerging, focusing on the transferability of adversarial attacks. This area is crucial as it aims to understand and mitigate the risks associated with adversarial examples in these powerful models. Notably, [Zou et al.](#page-15-0) [\(2023\)](#page-15-0) explored universal and transferable adversarial attacks on aligned language models, highlighting the potential vulnerabilities and the need for robust defenses in these systems.

|                                                 | Undetectability | Unremovability         | Uniqueness |
|-------------------------------------------------|-----------------|------------------------|------------|
| Goldwasser et al. (2022)                        | "               | robust to some         |            |
|                                                 |                 | smoothing attacks      | " (E)      |
| Adi et al. (2018) ; Zhang et al. (2018)         | " (E)           | %                      | " (E)      |
| Merrer et al. (2017)                            | " (E)           | robust to fine tunning |            |
|                                                 |                 | attacks                | " (E)      |
| Christ et al. (2023) ; Kuditipudi et al. (2023) | "               | %                      | "          |
| Zhao et al. (2023a)                             | %               | robust to edit         |            |
|                                                 |                 | distance attacks only  | "          |
| Tiffany Hsu (2023)                              | " (E)           | %                      | "          |
| Kirchenbauer et al. (2023)                      | %               | %                      | "          |

Table 1: Overview of properties across various watermarking schemes. The symbol " denotes properties with formal guarantees or where proof is plausible, whereas % indicates the absence of such guarantees. Entries marked with "(E) represent properties observed empirically; these lack formal proof in the corresponding literature, suggesting that deriving such proof may present substantial challenges. The LLM watermarking schemes refer to those applied to text generated by these models.

# B PRELIMINARIES

Learning. For a set Ω, we write ∆(Ω) to denote the set of all probability measures defined on the measurable space (Ω, F), where F is some fixed σ-algebra that is implicitly understood. We denote by X the domain and by Y the label space. A *model* is a function f : X → Y.

Definition 4 (*Learning task*). For a fixed X , Y a *learning task* is an element of ∆ ∆(X ) × Y<sup>X</sup> . We will often use L to denote a learning task.

For a *distribution* D ∈ ∆(X ) and a *ground truth* h : X → Y, we define an *error* of f as errD,h(f) := <sup>E</sup>x∼D[f(x) ̸= h(x)], where the index of err will often be understood implicitly and omitted in notation. For D ∈ ∆(X ), h : X → Y we define an *example oracle* Ex(D, h) as an oracle that samples x ∼ D and returns (x, h(x)).

Communication. When Ex(D, h) generates (x, h(x)) it is encoded as a bit-string of some length. For a *message space* M a *representation class* over (X , Y) is a mapping R : M → Y<sup>X</sup> .

Computation. Let U be a universal Turing Machine.

### B.1 DISCUSSION

Definition [4](#page-18-0) models a learner's prior knowledge of the learning task as a distribution over pairs (D, h), i.e. over pairs of distributions over the domain X and ground truths h : X → Y. It can be viewed as a generalization of, for instance, PAC-Bayes, where priors are distributions over hypothesis spaces. For us prior knowledge (what we call a learning task) is a distribution over not only hypotheses but also distributions themselves. Note that we consider a realizable scenario as there is a fixed ground truth. We could have considered a more general case, i.e. agnostic learning, where a learning task

**1029**

**1034**

**1054**

**1056**

**1071**

would be an element of ∆ (∆(X × Y)). We chose the former for simplicity and we believe most of the results would generalize to the agnostic case.

When Ex(D, h) generates (x, h(x)) it is encoded is some form, e.g. x ∈ {0, 1} <sup>n</sup>, but importantly n *is not* a parameter that the learner can control, i.e. the encoding is fixed. This precludes thinking of n as a security parameter that the watermarking party can increase to boost the security.

# C FORMAL DEFINITIONS

Definition 5 (*Succinct Circuits*). Let C be a circuit of width w and depth d. We will denote size(C) := w · d. We say that C is *succinctly representable* if there exists a circuit of size 100 log(size(C))[<sup>5</sup>](#page-19-0) that accepts as input i ∈ [w], j, j1, j<sup>2</sup> ∈ [d], g ∈ [O(1)], where g represents a gate from a universal constant-sized gate set, and returns 0 or 1, depending if g appears in location (i, j) in C and if it is connected to gates in locations (i − 1, j1) and (i − 1, j2).

We are ready to state formal versions of our main definitions.

Definition 6 (*Watermark*). Let L = (D, h) be a learning task. Let T, t, q ∈ <sup>N</sup>, ϵ ∈ 0, 1 2 , l, c, s ∈ (0, 1), s < c, where t bounds the running time of B, and T the running time of A, q the number of queries, ϵ the risk level, c probability that *uniqueness* holds, s probability that *unremovability* and *undetectability* holds, l the learning probability.

We say that a succinctly representable circuit AWATERMARK *of size* T implements a watermarking scheme for L, denoted by AWATERMARK ∈ WATERMARK(L, ϵ, q, T, t, l, c, s), if an interactive protocol in which AWATERMARK computes (f, x), f : X → Y, x ∈ X <sup>q</sup> , and B outputs y = B(f, x), y ∈ Y<sup>q</sup> satisfies the following

- Correctness (f has low error). With probability at least l

$$\text{err}(f) \leq \epsilon.$$

- Uniqueness (models trained from scratch give low-error answers). There exists a succinctly representable circuit B of size T such that with probability at least c

$$\text{err}(\mathbf{x}, \mathbf{y}) \leq 2\epsilon.$$

- Unremovability (fast B gives high-error answers). For every succinctly representable circuit B *of size at most* t we have that with probability at most s

$$\text{err}(\mathbf{x}, \mathbf{y}) \leq 2\epsilon.$$

- Undetectability (fast B cannot detect that they are tested). Distributions D<sup>q</sup> and x ∼ AWATERMARK are <sup>s</sup> 2 -indistinguishable for a class of succinctly representable circuits B *of size at most* t.

Definition 7 (*Adversarial Defense*). Let L = (D, h) be a learning task. Let T, t, q ∈ <sup>N</sup>, ϵ ∈ 0, 1 2 , l, c, s ∈ (0, 1), s < c, where t bounds the running time of A, and T the running time of B, q the number of queries, ϵ the error parameter, c the completeness, s the soundness, l the learning probability.

We say that a succinctly representable circuit BDEFENSE of size T implements an adversarial defense for L, denoted by BDEFENSE ∈ DEFENSE(L, ϵ, q, t, T, l, c, s), if an interactive protocol in which BDEFENSE computes f : X → Y, A replies with x = A(f), x ∈ X <sup>q</sup> , and BDEFENSE outputs b = BDEFENSE(f, x), b ∈ {0, 1} satisfies the following.

- Correctness (f has low error). With probability at least l

$$\text{err}(f) \leq \epsilon.$$

<sup>5</sup>Constant 100 is chosen arbitrarily. One often considers circuits representable by polylog-sized circuits. But for us, the constants play a role and this is why we formulate Definition [5.](#page-19-1)

**1099**

**1104**

**1106**

**1109**

**1119**

- Completeness (if x came from the right distribution BDEFENSE does not signal it is attacked). When x ∼ D<sup>q</sup> then with probability at least c

$$b = 0.$$

- Soundness (fast attacks creating x on which f makes mistakes are detected). For every succinctly representable circuit A of size at most t we have that with probability at most s,

$$\text{err}(\mathbf{x}, f(\mathbf{x})) > 7\epsilon$$
 and  $b = 0$ .

Definition 8 (*Transferable Attack*). Let L = (D, h) be a learning task. Let T, t, q ∈ <sup>N</sup>, ϵ ∈ 0, 1 2 , c, s ∈ (0, 1), where T bounds the running time of A and B, q the number of queries, ϵ the error parameter, c the *transferability* probability, s the *undetectability* probability.

We say that a succinctly representable circuit A *running in time* T is a transferable adversarial attack, denoted by ATRANSFATTACK ∈ TRANSFATTACK(L, ϵ, q, T, t, c, s), if an interactive protocol in which ATRANSFATTACK computes x ∈ X <sup>q</sup> , and B outputs y = B(x), y ∈ Y<sup>q</sup> satisfies the following.

- Transferability (fast provers return high error answers). For every succinctly representable circuit B of size at most t we have that with probability at least c

$$\text{err}(\mathbf{x}, \mathbf{y}) > 2\epsilon.$$

- Undetectability (fast provers cannot detect that they are tested). Distributions x ∼ D<sup>q</sup> and x := ATRANSFATTACK are <sup>s</sup> 2 -indistinguishable for a class of succinctly representable circuits B of size at most t.

# D MAIN THEOREM

Before proving our main theorem we recall a result from [Lipton & Young](#page-12-14) [\(1994a\)](#page-12-14) about simple strategies for large zero-sum games.

Game theory. A *two-player zero-sum game* is specified by a payoff matrix G. G is an r × c matrix. MIN, the row player, chooses a probability distribution p<sup>1</sup> over the rows. MAX, the column player, chooses a probability distribution p<sup>2</sup> over the columns. A row i and a column j are drawn from p<sup>1</sup> and p<sup>2</sup> and MIN pays Gij to MAX. MIN tries to minimize the expected payment; MAX tries to maximize it.

By the Min-Max Theorem, there exist optimal strategies for both MIN and MAX. Optimal means that playing first and revealing one's mixed strategy is not a disadvantage. Such a pair of strategies is also known as a Nash equilibrium. The expected payoff when both players play optimally is known as the value of the game and is denoted by V(G).

We will use the following theorem from [Lipton & Young](#page-12-14) [\(1994a\)](#page-12-14), which says that optimal strategies can be approximated by uniform distributions over sets of pure strategies of size O(log(c)).

Theorem 4 [\(Lipton & Young](#page-12-14) [\(1994a\)](#page-12-14)). *Let* G *be an* r × c *payoff matrix for a two-player zero-sum game. For any* η ∈ (0, 1) *and* k ≥ log(c) <sup>2</sup>η<sup>2</sup> *there exists a multiset of pure strategies for the* MIN *(row player) of size* k *such that a mixed strategy* p<sup>1</sup> *that samples uniformly from this multiset satisfies*

$$\max_j \sum_i p_1(i) \mathcal{G}_{ij} \leq \mathcal{V}(\mathcal{G}) + \eta(\mathcal{G}_{max} - \mathcal{G}_{min}),$$

*where* G*max*, G*min denote the maximum and minimum entry of* G *respectively. The symmetric result holds for the* MAX *player.*

Succinct Representations. Before we prove the main theorem we give a short discussion about why we consider succinctly representable circuits. Additionally, we require that the algorithms A and B in all the schemes to be *succinctly* representable, meaning their code should be much smaller than their running time. This requirement forbids a trivial way to circumvent learning by *hard-coding* ground-truth classifier in the description of the Watermark or Adversarial Defense algorithms.[<sup>6</sup>](#page-20-1)

<sup>6</sup> It is known in certain prover-verifier games to verify classification, described by [Anil et al.](#page-10-0) [\(2021\)](#page-10-0), this situation leads to undesirable equilibria, which is dubbed as the "trivial verifier" failure mode.

**1154**

**1159**

**1171**

**1174 1175**

**1177**

Additionally, the succinct representation of algorithms is also in accordance with how learning takes place in practice, for instance, consider DNNs and learning algorithms for those DNNs. The code representing gradient descent algorithms is almost always much shorter than the time required for the optimization of weights. For instance, a provable neural network model that learns succinct algorithms is described by [Goel et al.](#page-11-11) [\(2022\)](#page-11-11).

We are ready to prove our main theorem.

Theorem 5. *For every learning task* L = (D, h)*; and* ϵ ∈ (0, 1 2 )*,* T, q ∈ <sup>N</sup>*, such that there exists a succinctly representable circuit of size* T <sup>2</sup><sup>10</sup>√log(<sup>T</sup> ) *that learns* L *up to error* ϵ *with probability* 1 − 1 <sup>48</sup> *, at least one of*

$$\begin{aligned} \text{WATERMARK} \left( \mathcal{L}, \epsilon, q, T, T^{\frac{1}{2^{10}\sqrt{\log(T)}}}, l = \frac{21}{24}, c = \frac{21}{24}, s = \frac{19}{24} \right), \\ \text{DEFENSE} \left( \mathcal{L}, \epsilon, q, T^{\frac{1}{2^{10}\sqrt{\log(T)}}}, 2T, l = 1 - \frac{1}{48}, c = \frac{13}{24}, s = \frac{11}{24} \right), \\ \text{TRANSFATTACK} \left( \mathcal{L}, \epsilon, q, T, T, c = \frac{3}{24}, s = \frac{19}{24} \right) \end{aligned}$$

*exists.*

*Proof of Theorem [5.](#page-21-0)* Let L = D, h be a learning task. Let T, q, C ∈ N, ϵ ∈ 0, 2 .

*Proof of Theorem 5.* Let 
$$\mathcal{L} = (\mathcal{D}, h)$$
 be a learning task. Let  $T, q, C \in \mathbb{N}, \epsilon \in (0, \frac{1}{2})$ .

Let Candidate<sup>W</sup> be a set of T 1 <sup>2</sup><sup>10</sup>√log(<sup>T</sup> ) -sized succinctly representable circuits computing (f, x), where f : X → Y. Similarly, let Candidate<sup>D</sup> be a set of T 1 <sup>2</sup><sup>10</sup>√log(<sup>T</sup> ) -sized succinctly representable circuits accepting as input (f, x) and outputting (y, b), where y ∈ Y<sup>q</sup> , b ∈ {0, 1}. We interpret Candidate<sup>W</sup> as candidate algorithms for a watermark, and Candidate<sup>D</sup> as candidate algorithms for attacks on watermarks.

Define a zero-sum game G between (A, B) ∈ Candidate<sup>W</sup> × CandidateD. The payoff is given by

$$\begin{aligned} \mathcal{G}(\mathbf{A}, \mathbf{B}) &= \frac{1}{2} \mathbb{P}_{(\mathbf{x}, \mathbf{y}):=\mathbf{A}, (\mathbf{y}, \mathbf{b}):=\mathbf{B}} \left[ \text{err}(f) > \epsilon \text{ or } \text{err}(\mathbf{x}, \mathbf{y}) \leq 2\epsilon \text{ or } b = 1 \right] \\ &+ \frac{1}{2} \mathbb{P}_{f:=\mathbf{A}, \mathbf{x}\sim\mathcal{D}^q, (\mathbf{y}, \mathbf{b}):=\mathbf{B}} \left[ \text{err}(f) > \epsilon \text{ or } \left( \text{err}(\mathbf{x}, \mathbf{y}) \leq 2\epsilon \text{ and } b = 0 \right) \right], \end{aligned}$$

where A tries to minimize and B maximize the payoff.

Applying Theorem [4](#page-20-2) to G with η = 2−<sup>5</sup> we get two probability distributions, p over a multiset of pure strategies in Candidate<sup>W</sup> and r over a multiset of pure strategies in Candidate<sup>D</sup> that lead to a 2 −5 -approximate Nash equilibrium.

The size k of the multisets is bounded

$$\begin{aligned} k &\leq 2^6 \log(|\mathcal{C}\text{andidate}_{\mathfrak{M}}|) \\ &\leq 2^6 \log \left( 2 T^{100 \log \left( T^{\frac{1}{2^{10} \sqrt{\log(T)}} \right)} \right) \\ &\leq 2^{13} \log \left( T^{\frac{1}{2^{10} \sqrt{\log(T)}}} \right) \\ &\leq 2^3 \sqrt{\log(T)}. \end{aligned}$$

Next, observe that the mixed strategy corresponding to the distribution p can be represented by a succinct circuit of size

$$k \cdot 100 \log \left( T^{\frac{1}{2^{10}\sqrt{\log(T)}}} \right) \leq \frac{k}{2^3} \sqrt{\log(T)}, \quad (2)$$

because we can create a circuit that is a collection of k circuits corresponding to the multiset of p, where each one is of size 100 log T <sup>2</sup><sup>10</sup>√log(<sup>T</sup> ) . Combining equation [1](#page-21-1) and equation [2](#page-21-2) we get that

**1224**

**1227**

**1229**

the size of the circuit succinctly representing the strategy p is bounded by

$$\begin{aligned} & \frac{k}{2^3} \sqrt{\log(T)} \\ & \leq 2^3 \sqrt{\log(T)} \cdot \frac{1}{2^3} \sqrt{\log(T)} \\ & \leq \log(T). \end{aligned}$$

This implies that p can be implemented by a T-sized succinctly representable circuit. The same hold for r. Let's call the strategy corresponding to p, ANash, and the strategy corresponding to r, BNash. Consider cases:

Case G(ANASH, BNASH) ≥ 19 <sup>24</sup> . Define BDEFENSE to work as follows:

- 1. Simulate f := L, where L is a succinctly representable circuit of size T 1 <sup>2</sup><sup>10</sup>√log(<sup>T</sup> ) , such that

$$\mathbb{P}\left[\text{err}(f) \leq \epsilon\right] \geq 1 - \frac{1}{48}.$$

- 2. Send f to A.
- 3. Receive x from A.
- 4. Simulate (y, b) := BNASH(f, x).
- 5. Return b ′ = 1 if b = 1 or d(f(x), y) > 3ϵq and b ′ = 0 otherwise,

where d(·, ·) is the Hamming distance. Note that BDEFENSE runs in time 2T and not T because it first simulates L and then simulates BNASH.

We claim that

$$\mathbf{B}_{\text{DEFENSE}} \in \text{DEFENSE} \left( \mathcal{L}, \epsilon, q, T^{\frac{1}{2^{10}\sqrt{\log(T)}}}, 2T, l = 1 - \frac{1}{48}, c = \frac{13}{24}, s = \frac{11}{24} \right). \quad (3)$$

Assume towards contradiction that completeness or soundness of BDEFENSE as defined in Definition [7](#page-19-2) does not hold.

If completeness of BDEFENSE does not hold, then

$$\mathbb{P}_{\mathbf{x} \sim \mathcal{D}^q} [b' = 0] < \frac{13}{24}. \quad (4)$$

Let us compute the payoff of A, which first runs f := L and sets x ∼ D<sup>q</sup> , in the game G, when playing against BNASH

$$\begin{aligned} \mathcal{G}(\mathbf{A}, \mathbf{B}_{\text{NASH}}) \\ &= \frac{1}{2} \mathbb{P}_{(f, \mathbf{x}) := \mathbf{A}} \left[ \text{err}(f) > \epsilon \text{ or } \text{err}(\mathbf{x}, \mathbf{y}) \leq 2\epsilon \text{ or } b' = 1 \right] \\ &+ \frac{1}{2} \mathbb{P}_{f := \mathbf{A}, \mathbf{x} \sim \mathcal{D}^q} \left[ \text{err}(f) > \epsilon \text{ or } (\text{err}(\mathbf{x}, \mathbf{y}) \leq 2\epsilon \text{ and } b' = 0) \right] \\ &\leq \delta + \frac{1}{2} \mathbb{P}_{f := \mathbf{L}, \mathbf{x} \sim \mathcal{D}^q} \left[ \text{err}(\mathbf{x}, \mathbf{y}) \leq 2\epsilon \text{ or } b' = 1 \right] \\ &+ \frac{1}{2} \mathbb{P}_{f := \mathbf{L}, \mathbf{x} \sim \mathcal{D}^q} \left[ \text{err}(\mathbf{x}, \mathbf{y}) \leq 2\epsilon \text{ and } b' = 0 \right] \\ &&\text{Def. of } \mathbf{A}, \mathbf{B}_{\text{DEFENSE}}, \mathbb{P} \left[ \text{err}(f) \leq \epsilon \right] \geq \frac{47}{48} \\ &< \frac{1}{48} + \frac{1}{2} + \frac{\frac{13}{24}}{2} \\ &= \frac{38}{48} \\ &\leq \mathcal{G}(\mathbf{A}_{\text{NASH}}, \mathbf{B}_{\text{NASH}}), \quad \forall \end{aligned}$$

**1267**

**1281**

**1284**

**1287**

where the contradiction is with the properties of Nash equilibria.

Assume that A breaks the soundness of BDEFENSE, which translates to

$$\mathbb{P}_{\mathbf{x}:=\mathbf{A}(f)} \left[ \text{err}(\mathbf{x}, f(\mathbf{x})) > 7\epsilon \text{ and } b = 0 \text{ and } d(f(\mathbf{x}), \mathbf{y})) > 3\epsilon q \right] > \frac{11}{24}. \quad (5)$$

Let A′ first simulate f := L, then runs x := A(f), and returns (f, x). We have

$$\begin{aligned} & \mathcal{G}(\mathbf{A}', \mathbf{B}_{\text{NASH}}) \\ &= \frac{1}{2} \mathbb{P}_{(f, \mathbf{x}) := \mathbf{A}'} \left[ \text{err}(f) > \epsilon \text{ or } \text{err}(\mathbf{x}, \mathbf{y}) \leq 2\epsilon \text{ or } b' = 1 \right] \\ &+ \frac{1}{2} \mathbb{P}_{f := \mathbf{A}', \mathbf{x} \sim \mathcal{D}^q} \left[ \text{err}(f) > \epsilon \text{ or } (\text{err}(\mathbf{x}, \mathbf{y}) \leq 2\epsilon \text{ and } b' = 0) \right] \\ &= \frac{1}{2} \mathbb{P}_{f := \mathbf{L}, \mathbf{x} = \mathbf{A}(f)} \left[ \text{err}(f) > \epsilon \text{ or } \text{err}(\mathbf{x}, \mathbf{y}) \leq 2\epsilon \text{ or } b' = 1 \right] \\ &+ \frac{1}{2} \mathbb{P}_{f := \mathbf{L}, \mathbf{x} \sim \mathcal{D}^q} \left[ \text{err}(f) > \epsilon \text{ or } (\text{err}(\mathbf{x}, \mathbf{y}) \leq 2\epsilon \text{ and } b' = 0) \right] \quad \text{By def. of } \mathbf{A}' \\ &< \frac{1}{2} + \frac{1 - \frac{11}{24}}{2} \quad \text{By equation 5} \\ &= \frac{37}{48} \\ &\leq \mathcal{G}(\mathbf{A}_{\text{NASH}}, \mathbf{B}_{\text{NASH}}), \quad \forall \end{aligned}$$

≤ G(ANASH, <sup>B</sup>NASH), where the contradiction is with the properties of Nash equilibria. Thus equation [3](#page-22-1) holds.

Case G(ANASH, BNASH) < 19 <sup>24</sup> . Consider B that returns (f(x), b) for a uniformly random b. We have

$$\mathcal{G}(\mathbf{A}_{\text{NASH}}, \mathbf{B}) \geq \left(1 - \mathbb{P}_{f:=\mathbf{A}_{\text{NASH}}} \left[ \text{err}(f) \leq \epsilon \right] \right) + \mathbb{P}_{f:=\mathbf{A}_{\text{Nash}}} \left[ \text{err}(f) \leq \epsilon \right] \cdot \frac{1}{2},$$

because when x ∼ D<sup>q</sup> and err(f) ≤ ϵ the probability that err(x, y) ≤ 2ϵ and b = 0 is <sup>1</sup> 2 , and similarly when x := ANASH then the probability that b = 1 is equal <sup>1</sup> 2 . The assumption that G(ANash, B) < 19 <sup>24</sup> and properties of Nash equilibria imply that <sup>P</sup>f:=ANash [err(f) ≤ ϵ] ≥ <sup>24</sup> . This implies that *correctness* holds for ANash with l = <sup>24</sup> .

Next, assume towards contradiction that *unremovability* of ANASH does not hold, i.e., there is B running in time t such that P err(x, y) ≤ 2ϵ > 19 <sup>24</sup> . Consider <sup>B</sup>′ that on input (f, x) returns (B(f, x), 0). Then by definition of G, G(ANASH, B′ ) > 19 <sup>24</sup> , which is a contradiction .

Next, assume towards contradiction that *undetectability* of ANASH does not hold, i.e., there exists B such that it distinguishes x ∼ D<sup>q</sup> from x := ANASH with probability higher than <sup>19</sup> <sup>24</sup> . Consider <sup>B</sup>′ that on input (f, x) returns (f(x), B(f, x)). [<sup>7</sup>](#page-23-1) Then by definition of G, G(ANASH, B′ ) > 19 <sup>24</sup> , which is a contradiction

There are two further subcases. If ANASH satisfies *uniqueness* then

$$\mathbf{A}_{\text{NASH}} \in \text{WATERMARK} \left( \mathcal{L}, \epsilon, q, T, T^{\frac{1}{2^{10}\sqrt{\log(T)}}}, l = \frac{10}{24}, c = \frac{21}{24}, s = \frac{19}{24} \right)$$

.

If ANASH does not satisfy *uniqueness*, then, by definition, every succinctly representable circuit B of size T satisfies err(x, y) ≤ 2ϵ with probability at most <sup>21</sup> <sup>24</sup> . Consider the following A. It computes (f, x) := ANash , ignores f and sends x to B. By the assumption that *uniqueness* is not satisfied for ANASH we have that *transferability* of Definition [3](#page-5-0) holds for A with c = <sup>24</sup> . Note that B in the transferable attack does not receive f but it makes it no easier for it to satisfy the properties. Note that *undetectability* still holds with the same parameter. Thus

$$\mathbf{A}_{\text{NASH}} \in \text{TRANSFATTACK} \left( \mathcal{L}, \epsilon, q, T, T, c = \frac{3}{24}, s = \frac{19}{24} \right).$$

<sup>7</sup> Formally B receives as input (f, x) and not only x.

**1317**

**1319**

**1321**

**1324**

**1334**

### E BEYOND CLASSIFICATION

Inspired by Theorem [2,](#page-6-1) we conjecture a possibility of generalizing our results to generative learning tasks. Instead of a ground truth function, one could consider a ground truth quality oracle Q, which measures the quality of every input and output pair. This model introduces new phenomena *not* present in the case of classification. For example, the task of *generation*, i.e., producing a high-quality output y on input x, is decoupled from the task of *verification*, i.e., evaluating the quality of y as output for x. By decoupled, we mean that there is no clear formal reduction from one task to the other. Conversely, for classification, where the space of possible outputs is small, the two tasks are equivalent. Without going into details, this decoupling is the reason why the proof of Theorem [1](#page-5-2) does not automatically transfer to the generative case.

This decoupling introduces new complexities, but it also suggests that considering new definitions may be beneficial. For example, because generation and verification are equivalent for classification tasks, we allowed neither A nor B access to h, as it would trivialize the definitions. However, a modification of the Definition [6](#page-19-3) (Watermark), where access to Q is given to B could be investigated in the generative case. Interestingly, such a setting was considered in [\(Zhang et al., 2023\)](#page-14-4), where access to Q was crucial for mounting a provable attack on "all" strong watermarks. As we alluded to earlier, Theorem [2](#page-6-1) can be seen as an example of a task, where generation is easy but verification is hard – the opposite to what [Zhang et al.](#page-14-4) [\(2023\)](#page-14-4) posits.

We hope that careful formalizations of the interaction and capabilities of all parties might give insights into not only the schemes considered in this work, but also problems like weak-to-strong generalization [\(Burns et al., 2024\)](#page-10-13) or scalable oversight [\(Brown-Cohen et al., 2023\)](#page-10-3).

# F FULLY HOMOMORPHIC ENCRYPTION (FHE)

We include a definition of fully homomorphic encryption based on the definition from [Goldwasser](#page-11-12) [et al.](#page-11-12) [\(2013\)](#page-11-12). The notion of fully homomorphic encryption was first proposed by Rivest, Adleman and Dertouzos [Rivest et al.](#page-13-13) [\(1978\)](#page-13-13) in 1978. The first fully homomorphic encryption scheme was proposed in a breakthrough work by Gentry in 2009 [Gentry](#page-11-0) [\(2009\)](#page-11-0). A history and recent developments on fully homomorphic encryption is surveyed in [\(Vaikuntanathan, 2011\)](#page-13-14).

#### F.1 PRELIMINARIES

We say that a function f is *negligible* in an input parameter λ, if for all d > 0, there exists K such that for all λ > K, f(λ) < λ−<sup>d</sup> . For brevity, we write: for all sufficiently large λ, f(λ) = negl(λ). We say that a function f is *polynomial* in an input parameter λ, if there exists a polynomial p such that for all λ, f(λ) ≤ p(λ). We write f(λ) = poly(λ). A similar definition holds for polylog(λ). For two polynomials p, q, we say p ≤ q if for every λ ∈ <sup>N</sup>, p(λ) ≤ q(λ).

When saying that a Turing machine A is p.p.t. we mean that A is a non-uniform probabilistic polynomial-time machine.

#### F.2 DEFINITIONS

Definition 9 [\(Goldwasser et al.](#page-11-12) [\(2013\)](#page-11-12)). A homomorphic (public-key) encryption scheme FHE is a quadruple of polynomial time algorithms (FHE.KEYGEN, FHE.ENC, FHE.DEC, FHE.EVAL) as follows:

- FHE.KEYGEN(1<sup>λ</sup> ) is a probabilistic algorithm that takes as input the security parameter 1 λ and outputs a public key pk and a secret key sk.
- FHE.ENC(pk, x ∈ {0, 1}) is a probabilistic algorithm that takes as input the public key pk and an input bit x and outputs a ciphertext ψ.
- FHE.DEC(sk, ψ) is a deterministic algorithm that takes as input the secret key sk and a ciphertext ψ and outputs a message x <sup>∗</sup> ∈ {0, 1}.

**1371**

**1374**

- FHE.EVAL(pk, C, ψ1, ψ2, . . . , ψn) is a deterministic algorithm that takes as input the public key pk, some circuit C that takes n bits as input and outputs one bit, as well as n ciphertexts ψ1, . . . , ψn. It outputs a ciphertext ψ<sup>C</sup> .

Compactness: For all security parameters λ, there exists a polynomial p(·) such that for all input sizes n, for all x1, . . . , xn, for all C, the output length of FHE.EVAL is at most p(n) bits long.

Definition 10 (C*-homomorphism, [Goldwasser et al.](#page-11-12) [\(2013\)](#page-11-12)*). Let C = {Cn}n∈<sup>N</sup> be a class of boolean circuits, where C<sup>n</sup> is a set of boolean circuits taking n bits as input. A scheme FHE is C-homomorphic if for every polynomial n(·), for every sufficiently large security parameter λ, for every circuit C ∈ Cn, and for every input bit sequence x1, . . . , xn, where n = n(λ),

$$\mathbb{P} \left[ \begin{array}{l} (pk, sk) \leftarrow \text{FHE.KEYGEN}(1^\lambda); \\ \psi_i \leftarrow \text{FHE.ENC}(pk, x_i) \text{ for } i = 1 \dots n; \\ \psi \leftarrow \text{FHE.EVAL}(pk, C, \psi_1, \dots, \psi_n) : \\ \text{FHE.DEC}(sk, \psi) \neq C(x_1, \dots, x_n) \end{array} \right] = \text{negl}(\lambda),$$

where the probability is over the coin tosses of FHE.KEYGEN and FHE.ENC.

Definition 11 (*Fully homomorphic encryption*). A scheme FHE is fully homomorphic if it is homomorphic for the class of all arithmetic circuits over GF(2).

Definition 12 (*Leveled fully homomorphic encryption*). A leveled fully homomorphic encryption scheme is a homomorphic scheme where FHE.KEYGEN receives an additional input 1 d and the resulting scheme is homomorphic for all depth-d arithmetic circuits over GF(2).

Definition 13 (*IND-CPA security*). A scheme FHE is IND-CPA secure if for any p.p.t. adversary A,

$$\left| \mathbb{P} \left[ (pk, sk) \leftarrow \text{FHE.KEYGEN}(1^\lambda) : \mathcal{A}(pk, \text{FHE.ENC}(pk, 0)) = 1 \right] + -\mathbb{P} \left[ (pk, sk) \leftarrow \text{FHE.KEYGEN}(1^\lambda) : \mathcal{A}(pk, \text{FHE.ENC}(pk, 1)) = 1 \right] \right| = \text{negl}(\lambda).$$

We now state the result of Brakerski, Gentry, and Vaikuntanathan [\(Brakerski et al., 2012\)](#page-10-14) that shows a leveled fully homomorphic encryption scheme based on a standard assumption in cryptography called Learning with Errors [\(Regev, 2005\)](#page-13-15):

Theorem 6 (*Fully Homomorphic Encryption, definition from [Goldwasser et al.](#page-11-12) [\(2013\)](#page-11-12)*). *Assume that there is a constant* 0 < ϵ < 1 *such that for every sufficiently large* ℓ*, the approximate shortest vector problem gapSVP in* ℓ *dimensions is hard to approximate to within a* 2 O(ℓ ) *factor in the worst case. Then, for every* n *and every polynomial* d = d(n)*, there is an IND-CPA secure* d*leveled fully homomorphic encryption scheme where encrypting* n *bits produces ciphertexts of length poly*(n, λ, d<sup>1</sup>/ϵ)*, the size of the circuit for homomorphic evaluation of a function* f *is size*(C<sup>f</sup> ) · *poly*(n, λ, d<sup>1</sup>/ϵ) *and its depth is depth*(C<sup>f</sup> ) · *poly*(log n, log d)*.*

# G TRANSFERABLE ATTACKS EXIST

Learning Theory Preliminaries. For the next lemma, we will consider a slight generalization of learning tasks to the case where there are many valid outputs for a given input. This can be understood as the case of generative tasks. We call a function h : X × Y → {0, 1} an error oracle for a learning task (D, h) if the error of f : X → Y is defined as

$$\text{err}(f) := \mathbb{E}_{x \sim \mathcal{D}}[h(x, f(x))],$$

where the randomness of expectation includes the potential randomness of f. We assume that all parties have access to samples (x, y) ∈ X × Y, where x ∼ D and y ∈ Y is some y such that h(x, y) = 0.

The following learning task will be crucial for our construction.

Definition 14 (*Lines on a Circle Learning Task* L ◦ ). The input space is X = {x ∈ <sup>R</sup> 2 | ∥x∥<sup>2</sup> = 1}, and the output space Y = {−1, +1}. The hypothesis class is H = {h<sup>w</sup> | w ∈ <sup>R</sup> , ∥w∥<sup>2</sup> = 1}, where hw(x) := sgn(⟨w, x⟩). Let D = U(X ) and L = (D, H). Note that H has VC-dimension equal to 2 so L is learnable to error ϵ with O( 1 ϵ ) samples.

Lemma 3 (*Learning lower bound for* L ◦ ). *Let* L *be a learning algorithm for* L ◦ *(Definition [14\)](#page-25-1) that uses* K *samples and returns a classifier* f*. Then*

$$\mathbb{P}_{w \sim U(\mathcal{X}), f \leftarrow \mathbf{L}} \left[ \mathbb{P}_{x \sim U(\mathcal{X})}[f(x) \neq h_w(x)] \leq \frac{1}{2K} \right] \leq \frac{3}{100}.$$

*Proof.* Consider the following algorithm A. It first simulates L on K samples to compute f. Next, it performs a smoothing of f, i.e., computes

$$f_\eta(x) := \begin{cases} +1, & \text{if } \mathbb{P}_{x' \sim U(B_x(2\pi\eta))}[f(x') = +1] > \mathbb{P}_{x' \sim U(B_x(2\pi\eta))}[f(x') = -1] \\ -1, & \text{otherwise.} \end{cases}$$

Note that if err(f) ≤ η for a ground truth h<sup>w</sup> then for every x ∈ X \Bx(2πη) we have fη(x) = hw(x). This implies that A can be adapted to an algorithm that with probability 1 finds w ′ such that |∡(w, w′ )| ≤ err(f).

Assuming towards contradiction that the statement of the lemma does not hold it means that there is an algorithm using K samples that with probability <sup>3</sup> <sup>100</sup> locates <sup>w</sup> up to angle <sup>1</sup> 2K .

Consider any algorithm A using K samples. Probability that A does not see any sample in Bw(2πη) is at least

$$(1 - 4\eta)^K \geq \left( (1 - 4\eta)^{\frac{1}{4\eta}} \right)^{4\eta K} \geq \left( \frac{1}{2e} \right)^{4\eta K},$$

which is bigger than 1 − <sup>100</sup> if we set η = 1 2K . But note that if there is no sample in Bw(2πη) then A cannot locate w up to η with certainty. This proves the lemma.

Lemma 4 (*Boosting for* L ◦ ). *Let* η, ν ∈ (0, 4 )*,* L *be a learning algorithm for* (D, H) *that uses* K *samples and outputs* f : X → {−1, +1} *such that with probability* δ

$$\mathbb{P}_w \sim U(\mathcal{X}), x \sim U(B_w(2\pi\eta)) [f(x) \neq h_w(x)] \leq \nu. \quad (6)$$

*Then there exists a learning algorithm* L ′ *that uses* max K, <sup>9</sup> η *samples such that with probability* δ − <sup>1000</sup> *returns* f ′ *such that*

$$\mathbb{P}_{w \sim U(\mathcal{X}), x \sim U(\mathcal{X})}[f'(x) \neq h_w(x)] \leq 4\eta\nu.$$

*Proof.* Let L ′ first draws max K, <sup>9</sup> η samples Q and defines g : X → {−1, +1, ⊥} as, g maps to −1 the smallest continuous interval containing all samples from Q with label −1. Similarly g maps to +1 the smallest continuous interval containing all samples from Q with label +1. The intervals are disjoined by construction. Unmapped points are mapped to ⊥. Next, L ′ simulates L with K samples and gets a classifier f that with probability δ satisfies the assumption of the lemma. Finally, it returns

$$f'(x) := \begin{cases} g(x), & \text{if } g(x) \neq \perp \\ f(x), & \text{otherwise.} \end{cases}$$

Consider 4 arcs defined as the 2 arcs constituting Bw(2πη) divided into 2 parts each by the line {x ∈ <sup>R</sup> 2 | ⟨w, x⟩ = 0}. Let E be the event that some of these intervals do not contain a sample from Q. Observe that

$$\mathbb{P}[E] \leq 4(1 - \eta)^{\frac{9}{\eta}} \leq \frac{1}{1000}.$$

By the union bound with probability δ − <sup>1000</sup> , f satisfies equation [6](#page-26-2) and E does not happen. By definition of f ′ this gives the statement of the lemma.

Theorem 7 (*Transferable Attack for a Cryptography based Learning Task*). *There exists a polynomial* p *such that for every polynomial* r ≥ p [<sup>8</sup>](#page-26-3) *and for every sufficiently large security parameter* λ ∈ <sup>N</sup> *there exists a family of distributions* <sup>D</sup><sup>λ</sup> = {D<sup>k</sup> λ }k*, hypothesis class of error oracles* H<sup>λ</sup> = {h k λ }k*, distribution* D<sup>L</sup> *over* k *such that the following conditions are satisfied.*

<sup>8</sup>This is only a formal requirement so that the interval (1/r(λ), 1/p(λ)) is non-empty.

- *1. There exists* A *such that for all* ϵ ∈ 1 r(λ) , 1 p(λ) *if* k ∼ D<sup>L</sup> *then* <sup>A</sup> <sup>∈</sup> <sup>T</sup>RANSFATTACK D k λ , h<sup>k</sup> λ , ϵ, q = 16 ϵ , T = 10<sup>3</sup> ϵ 1.3 , t = 1 ϵ , c = 1 − 1 10 , s = *negl*(λ)
- *2. There exists a learner* L *such that for every* ϵ ∈ r(λ) , p(λ) *, with probability* 1 − <sup>10</sup> *over the choice of* k *and the internal randomness of* L*,* L *returns a classifier of error at most* ϵ*. Additionally,* L *runs in time* <sup>10</sup><sup>3</sup> ϵ <sup>1</sup>.<sup>3</sup> *and uses* <sup>900</sup> ϵ *samples.*
- *3. For every* ϵ ∈ r(λ) , p(λ) *, every learner* L *using at most* <sup>1</sup> ϵ *samples (and in particular time) the probability over the choice of* k *and the internal randomness of* L *that it returns a classifier of error at most* ϵ *is smaller than* <sup>1</sup> <sup>10</sup> *.*

.

Next, we give a formal proof.

*Proof.* The learning task is based on L ◦ from Definition [14.](#page-25-1)

Setting of Parameters for FHE. Let FHE be a fully homomorphic encryption scheme from Theorem [6.](#page-25-2) We will use the scheme for constant leveled circuits d = O(1). Let s(n, λ) be the polynomial bounding the size of the encryption of inputs of length n with λ security as well as bounding size of the circuit for holomorphic evaluation, which is guaranteed to exist by Theorem [6.](#page-25-2) Let β ∈ (0, 1) and p be a polynomial such that

$$s(n^\beta, \lambda, d) \leq (n \cdot p(\lambda))^{0.1}, \quad (7)$$

which exist because s is a polynomial. Let λ ∈ N and n := p <sup>1</sup>/β(λ) [9](#page-27-0) for the length of inputs in the FHE scheme. Observe

$$\begin{aligned} s(n, \lambda, d) &\leq (p(\lambda) \cdot p(\lambda))^{0.1} && \text{By equation 7} \\ &\leq \frac{1}{\epsilon^{0.2}} && \text{By } \epsilon \in \left( \frac{1}{r(\lambda)}, \frac{1}{p(\lambda)} \right). \end{aligned} \quad (8)$$

Learning Task. We will omit λ from indexes of D, D and h for simplicity of notation. Let <sup>D</sup> = {D(pk,sk)}(pk,sk), H = {h (pk,sk,w)}(pk,sk,w) indexed by valid public/secret key pairs of FHE and w ∈ X , with X as in Definition [14.](#page-25-1) Let D<sup>L</sup> over (pk,sk, w) be equal to FHE.KEYGEN(1<sup>λ</sup> ) × U(X ).

For a valid (pk,sk) pair we define D(pk,sk) as the result of the following process: x ∼ D = U(X ), with probability <sup>1</sup> 2 return (0, x, pk) and with probability <sup>1</sup> 2 return (1, FHE.ENC(pk, x), pk), where the first element of the triple describes if the x is encrypted or not. x is represented as a number ∈ (0, 1) using n bits.[<sup>10</sup>](#page-27-2)

For a valid (pk,sk) pair and w ∈ X we define h (pk,sk,w)((b, x, pk), y) as a result of the following process: if b = 0 return <sup>1</sup>hw(x)=y, otherwise let xDEC ← FHE.DEC(sk, x), y<sup>D</sup>EC ← FHE.DEC(sk, y) and if xDEC, y<sup>D</sup>EC ̸=⊥ (decryption is succesful) return <sup>1</sup>hw(xDEC)=yDEC and return 1 otherwise.

Note 2 (Ω( <sup>1</sup> ϵ )*-sample learning lower bound.*). *Note, that by construction any learner using* K *samples for learning task* {D*(pk,sk)*}*(pk,sk)*, {h *(pk,sk,w)*}*(pk,sk,w) can be transformed (potentially computationally inefficiently) into a learner using* K *samples for the task from Defnition [14](#page-25-1) that returns a classifier of at most the same error. This together with a lower bound for learning from Lemma [3](#page-26-1) proves point 3 of the lemma.*

<sup>9</sup>Note that this setting allows to represent points on X up to 2 −p <sup>1</sup>/β(λ) precision and this precision is better than <sup>1</sup> r(λ) for every polynomial r for sufficiently large λ. This implies that this precision is enough to allow for learning up to error ϵ, because of the setting ϵ ≥ .

q(λ) <sup>10</sup>Note that the space over which D (pk,sk) is defined on is *not* X .

**1517**

**1519**

**1521**

**1534**

**1554**

Algorithm 1 TRANSFATTACK(D<sup>k</sup> λ , Hλ, ϵ, λ)

1: Input: Oracle access to a distribution D<sup>k</sup> λ for some D<sup>k</sup> <sup>λ</sup> ∈ <sup>D</sup>λ, the hypothesis class H<sup>λ</sup> = {h k λ }k,

error level ϵ ∈ (0, 1), and the security parameter λ. 2: N := 900/ϵ, q := 16/ϵ 3: Q = {((b<sup>i</sup> , x<sup>i</sup> , pk), yi)}i∈[N] ∼ (D<sup>k</sup> λ ) <sup>N</sup> ▷ N i.i.d. samples from D<sup>k</sup> λ 4: Q<sup>C</sup>LEAR = {((b, x, pk), y) ∈ Q : b = 0} ▷ Q<sup>C</sup>LEAR ⊆ Q of unencrypted x's 5: fw′ (·) := sgn(⟨w ′ , ·⟩) ← a line consistent with samples from Q<sup>C</sup>LEAR ▷ fw′ : X → {−1, +1} 6: {x ′ i }i∈[q] ∼ U(X q ) 7: S ∼ U(2[q] ) ▷ S ⊆ [q] a uniformly random subset 8: EBND; = ∅ 9: for i ∈ [q − |S|] do 10: xBND ∼ U(Bw′ (2π(ϵ + ϵ <sup>100</sup> ))) ▷ xBND is close to the decision boundary of fw′ 11: EBND := EBND ∪ {FHE.ENC(pk, xBND)} 12: end for 13: x := {(0, x′ i , pk) | i ∈ [q] \ S} ∪ {(1, x′ , pk) | x ′ ∈ EBND} 14: Return x

Definition of A (Algorithm [1\)](#page-28-0). A draws N samples Q = {((b<sup>i</sup> , x<sup>i</sup> , pk), yi)}i∈[N] for N := <sup>900</sup> ϵ .

Next, A chooses a subset Q<sup>C</sup>LEAR ⊆ Q of samples for which b<sup>i</sup> = 0. It trains a classifier fw′ (·) := sgn(⟨w ′ , ·⟩) on Q<sup>C</sup>LEAR by returning any fw′ consistent with Q<sup>C</sup>LEAR. This can be done in time

$$N \cdot n \leq \frac{900}{\epsilon} \cdot p^{1/\beta}(\lambda) \leq \frac{900}{\epsilon^{1.1}} \quad (9)$$

by keeping track of the smallest interval containing all samples in Q<sup>C</sup>LEAR labeled with +1 and then returning any fw′ consistent with this interval.

Note 3 (O( 1 ϵ <sup>1</sup>.<sup>3</sup> )*-time learning upper bound.*). *First note that* A *learns well, i.e., with probability at least* 1 − 2 1 − ϵ <sup>100</sup> <sup>900</sup> <sup>ϵ</sup> ≥ 1 − <sup>1000</sup> *we have that*

$$|\angle(w, w')| \leq \frac{2\pi\epsilon}{100} \quad (10)$$

*Moreover,* fw′ (x) *can be implemented by a circuit* Cfw′ *that compares* x *with the endpoints of the interval. This can be done by a constant leveled circuit. Moreover* Cfw′ *can be evaluated with* FHE.EVAL *in time*

$$\text{size}(C_{f_w, \cdot})s(n, \lambda, d) \leq 10n \cdot s(n, \lambda, d) \leq 10p^{1/\beta}(\lambda)s(n, \lambda, d) \leq \frac{10}{\epsilon^{0.3}},$$

*where the last inequality follows from equation [8.](#page-27-3) This implies that* A *can, in time* T*, return a classifier of error* ≤ ϵ *for* (D*(pk,sk)*, h*(pk,sk,w)*)*. This proves point 2. of the lemma.*

Next, A prepares x as follows. It samples q = 16 ϵ points {x ′ i }i∈[q] from X uniformly at random. It chooses a uniformly random subset S ⊆ [q]. Next, A generates q − |S| inputs using the following process: xBND ∼ U(Bw′ (2π(ϵ + ϵ <sup>100</sup> ))) (xBND is close to the decision boundary of fw′ ), return FHE.ENC(pk, xBND). Call the set of q − |S| points EBND. A defines:

$$\mathbf{x} := \{(0, x'_i, \mathbf{pk}) \mid i \in [q] \setminus S\} \cup \{(1, x', \mathbf{pk}) \mid x' \in E_{\text{BND}}\}.$$

The running time of this phase is dominated by evaluations of FHE.EVAL, which takes

$$q \cdot s(n, \lambda, d) \leq \frac{16}{\epsilon} \cdot \frac{1}{\epsilon^{0.2}} \leq \frac{16}{\epsilon^{1.2}}, \quad (11)$$

where the first inequality follows from equation [8.](#page-27-3) Taking the sum of equation [9](#page-28-1) and equation [11](#page-28-2) we get that the running time of A is smaller than the required T = <sup>10</sup><sup>3</sup> /ϵ 1.3 .

**1571**

**1574**

A constitutes a Transferable Attack. Now, consider B that runs in time t = ϵ <sup>2</sup> . By the assumption t ≤ r(λ), which implies that the security guarantees of FHE hold for B.

We first claim that x is indistinguishable from D(pk,sk) for B. Observe that by construction the distribution of ratio of encrypted and not encrypted x's in x is identical to that of D(pk,sk). Moreover, the distribution of unencrypted x's is identical to that of D(pk,sk) by construction. Finally, by the IND-CPA security of FHE and the fact that the running time of B is bounded by q(λ) for some polynomial q we have that FHE.ENC(pk, xBND) is distinguishable from x ∼ X , FHE.ENC(pk, x) with advantage at most negl(λ). Thus *undetectability* holds with near perfect soundness s = 1 <sup>2</sup> + negl(λ).

Next, we claim that B can't return low-error answers on x.

Assume towards contradiction that with probability <sup>5</sup> 100

$$\mathbb{P}_{w \sim U(\mathcal{X}), x \sim U(B_w(2\pi\epsilon))}[f(x) \neq h_w(x)] \leq 10\epsilon. \quad (12)$$

We can apply Lemma [4](#page-26-4) to get that there exists a learner using t + 9 ϵ samples that with probability <sup>4</sup> 100 returns f ′ such that

$$\mathbb{P}_{w \sim U(\mathcal{X}), x \sim U(\mathcal{X})}[f'(x) \neq h_w(x)] \leq 40\epsilon^2. \quad (13)$$

Applying Lemma [3](#page-26-1) to equation [13](#page-29-0) we know that

$$40\epsilon^2 \geq \frac{1}{2(t + \frac{9}{\epsilon})},$$

which implies

$$t \geq \frac{10}{\epsilon^2},$$

which is a contradiction with the assumed running time of B. Thus equation [12](#page-29-1) does not hold and in consequence using equation [10](#page-28-3) we have that with probability 1 − 6 100

$$\mathbb{P}_{w \sim U(\mathcal{X}), x \sim U(B_{w'}(2\pi(\epsilon + \frac{\epsilon}{10}))}[f(x) \neq h_w(x)] \geq \frac{10}{14} \cdot 10\epsilon \geq 7\epsilon, \quad (14)$$

where crucially x is sampled from U(Bw′ ) and not U(Bw). By Fact [2](#page-29-2) we know that |S| ≥ <sup>q</sup> <sup>3</sup> with probability at least

$$1 - 2e^{-\frac{q}{72}} = 1 - 2e^{-\frac{1}{8\epsilon}} \geq 1 - \frac{1}{1000}.$$

Another application of the Chernoff bound and the union bound we get from equation [14](#page-29-3) that with probability at least 1 − 1 <sup>10</sup> we have that err(x, y) is larger than 2ϵ by the setting of q = 16 ϵ .

Note 4. *We want to emphasize that it is crucial (for our construction) that the distribution has both an encrypted and an unencrypted part.*

*As mentioned before, if there was no* D<sup>C</sup>LEAR *then* A *would see only samples of the form*

(FHE.ENC(
$$x$$
), FHE.ENC( $y$ ))

*and would not know which of them lie close to the boundary of* hw*, and so it would not be able to choose tricky samples.* A *would be able to learn a low-error classifier, but* only *under the encryption. More concretely,* A *would be able to homomorphically evaluate a circuit that, given a training set and a test point, learns a good classifier and classifies the test point with it. However, it would* not *be able to, with high probability, generate* FHE.ENC(x)*, for* x *close to the boundary as it would not know (in the clear) where the decision boundary is.*

*If there was no* D<sup>E</sup>NC *then everything would happen in the clear and so* B *would be able to distinguish* x*'s that appear too close to the boundary.*

Fact 2 (*Chernoff-Hoeffding*). Let X1, . . . , X<sup>k</sup> be independent Bernoulli variables with parameter p. Then for every 0 < ϵ < 1

$$\mathbb{P} \left[ \left| \frac{1}{k} \sum_{i=1}^k X_i - p \right| > \epsilon \right] \leq 2e^{-\frac{\epsilon^2 k}{2}}$$

**1624**

**1627**

**1629**

**1657**

and

$$\mathbb{P} \left[ \frac{1}{k} \sum_{i=1}^k X_i \leq (1 - \epsilon)p \right] \leq e^{-\frac{\epsilon^2 k p}{2}}.$$

$$\mathbb{P} \left[ \frac{1}{k} \sum_{i=1}^k X_i > (1 + \delta)p \right] \leq e^{-\frac{\delta^2 k p}{2+\delta}}.$$

Also for every δ > 0

# H TRANSFERABLE ATTACKS IMPLY CRYPTOGRAPHY

#### H.1 EFID PAIRS

The typical way in which security of EFID pairs is defined, e.g., in [\(Goldreich, 1990\)](#page-11-3), is that they should be secure against all polynomial-time algorithms. However, for the case of pseudorandom generators (PRGs), which are known are equivalent to EFIDs pairs, more granular notions of security were considered. For instance, in [\(Nisan, 1990\)](#page-13-16) the existence of PRGs secure against time and space bounded adversaries was considered. In a similar spirit we consider EFID pairs that are secure against adversaries with a fixed time bound.

Definition 15 (*Total Variation*). For two distrbutions D0, D<sup>1</sup> over a finite domain X we define their *total variation distance* as

$$\triangle(\mathcal{D}_0, \mathcal{D}_1) := \sum_{x \in \mathcal{X}} \frac{1}{2} |\mathcal{D}_0(x) - \mathcal{D}_1(x)|.$$

Definition 16 (*EFID pairs*). For parameters η, δ ∈ (0, 1) we call a pair of distributions (D0, D1) a (T, T′ , η, δ) EFID pair if

- 1. D0, D<sup>1</sup> are samplable in time T,
- 2. △(D0, D1) ≥ η,
- 3. D0, D<sup>1</sup> are δ-indistinguishable for adversaries running in time T ′ .

#### H.2 TRANSFERABLE ATTACKS IMPLY EFID PAIRS

Theorem 8 (*Tasks with Transferable Attacks imply EFID pairs*). *For every* ϵ, T, T′ ∈ <sup>N</sup>, T ≤ T ′ *, every learning task* <sup>L</sup> *if there exists* <sup>A</sup> <sup>∈</sup> <sup>T</sup>RANSFATTACK L, ϵ, q, T, T′ , c, s *and there exists a learner running in time* T *that, with probability* p*, learns* f *such that err*(f) ≤ ϵ*, then there exists a* (T, T′ , 1 2 (p + c − 1 − e − <sup>3</sup> ), s 2 ) *EFID pair.*

*Proof.* Let ϵ, T, T′ , q, c, s,L = (D, h) and A be as in the assumption of the theorem. Firstly, define D<sup>0</sup> := D<sup>q</sup> , where q is the number of samples A sends in the attack. Secondly, define D<sup>1</sup> to be the distribution of x := A. Note that x ∈ X <sup>q</sup> .

Observe that D0, D<sup>1</sup> are samplable in time T as A runs in time T. Secondly, D0, D<sup>1</sup> are <sup>s</sup> 2 indistinguishable for T ′ -bounded adversaries by *undetectability* of A. Finally, the fact that D0, D<sup>1</sup> are statistically far follows from *transferability*. Indeed, the following procedure accepting input x ∈ X <sup>q</sup> is a distinguisher:

- 1. Run the learner (the existence of which is guaranteed by the assumption of the theorem) to obtain f.
- 2. y := f(x).
- 3. If err(x, y) ≤ 2ϵ return 0, otherwise return 1.

If x ∼ D<sup>0</sup> = D<sup>q</sup> then err(f) ≤ ϵ with probability p. By Fact [2](#page-29-2) and the union bound we also know that err(x, y) ≤ 2ϵ with probability p − e − ϵq <sup>3</sup> and so, the distinguisher will return 0 with

**1693 1694**

**1696 1697**

**1699**

probability p − e − ϵq <sup>3</sup> . On the other hand, if x ∼ D<sup>1</sup> = A we know from *transferability* of A that every algorithm running in time T ′ will return y such that err(x, y) > 2ϵ with probability at least c. By the assumption that T ′ ≥ T we know that err(x, f(x)) > 2ϵ with probability at least c also. Consequently, the distinguisher will return 1 with probability at least c in this case. By the properties of total variation this implies that △(D0, D1) ≥ 2 (p + c − 1 − e − ϵq <sup>3</sup> ) Summarizing, (D0, D1) is a (T, T′ , 1 2 (p + c − 1 − e − <sup>3</sup> ), s 2 ) EFID pair.

Note 5 (*Setting of parameters*). *Observe that if* p ≈ 1*, i.e., it is possible to almost surely learn* f *in time* T *such that err*(f) ≤ ϵ*,* c *is a constant,* q = Ω( <sup>1</sup> ϵ ) *then* △(D0, D1) *is a constant.*

Note 6. *We want to emphasize that our distinguisher crucially uses the error oracle in its last step. So it is possible that it is not implementable for all time bounds!*

# I ADVERSARIAL DEFENSES EXIST

Our result is based on [\(Goldwasser et al., 2020\)](#page-12-8). Before we state and prove our result we give an overview of the learning model considered in [\(Goldwasser et al., 2020\)](#page-12-8).

#### I.1 TRANSDUCTIVE LEARNING WITH REJECTIONS.

In [\(Goldwasser et al., 2020\)](#page-12-8) the authors consider a model, where a learner L receives a training set of labeled samples from the original distribution (xD, y<sup>D</sup> = h(xD)), x ∼ D<sup>N</sup> , y<sup>D</sup> ∈ {−1, +1} N , where h is the ground truth, together with a test set x<sup>T</sup> ∈ X <sup>q</sup> . Next, L uses (xD, yD, x<sup>T</sup> ) to compute y<sup>T</sup> ∈ {−1, +1, ⊔⊓} q , where ⊔⊓ represents that L abstains (rejects) from classifying the corresponding x.

Before we define when learning is successful, we will need some notation. For q ∈ <sup>N</sup>, x ∈ X <sup>q</sup> , y ∈ {−1, +1, ⊔⊓} <sup>q</sup> we define

$$\text{err}(\mathbf{x}, \mathbf{y}) := \frac{1}{q} \sum_{i \in [q]} \mathbb{1} \left\{ h(x_i) \neq y_i, y_i \neq \square, h(x_i) \neq \perp \right\}, \quad \square(\mathbf{y}) := \frac{1}{q} \left| \left\{ i \in [q] : y_i = \square \right\} \right|,$$

which means that we count (x, y) ∈ X × {−1, +1, ⊔⊓} as an error if h is well defined on x, y is not an abstantion and h(x) ̸= y.

Learning is successful if it satisfies two properties.

- If x<sup>T</sup> ∼ D<sup>q</sup> then with high probability err(x<sup>T</sup> , y<sup>T</sup> ) and ⊔⊓(y<sup>T</sup> ) are small.
- For *every* x<sup>T</sup> ∈ X <sup>q</sup> with high probability err(x<sup>T</sup> , y<sup>T</sup> ) is small.[<sup>11</sup>](#page-31-0)

The formal guarantee of a result from [Goldwasser et al.](#page-12-8) [\(2020\)](#page-12-8) are given in Theorem [9.](#page-32-1) Let's call this model Transductive Learning with Rejections (TLR).

Note the differences between TLR and our definition of Adversarial Defenses. To compare the two models we associate the learner L from TLR with B in our setup, and the party producing x<sup>T</sup> with A in our definition. First, in TLR, B does not send f to A. Secondly, and most importantly, we do not allow B to reply with rejections (⊔⊓) but instead require that B can "distinguish" that it is being tested (see soundness of Definition [7\)](#page-19-2). Finally, there are no apriori time bounds on either A or B in TLR. The models are similar but a priori incomparable and any result for TLR needs to be carefully analyzed before being used to prove that it is an Adversarial Defense.

#### I.2 FORMAL GUARANTEE FOR TRANSDUCTIVE LEARNING WITH REJECTIONS (TLR)

<sup>11</sup>Note that, crucially, in this case ⊔⊓(y<sup>T</sup> ) might be very high, e.g., equal to 1.

**1731**

**1734**

**1737**

**1751**

**1754**

**1764**

**1767**

Theorem 9 (*TLR guarantee [\(Goldwasser et al.](#page-12-8) [\(2020\)](#page-12-8))*). *For any* N ∈ <sup>N</sup>, ϵ ∈ (0, 1), h ∈ H *and distribution* D *over* X *:*

$$\mathbb{P}_{\mathbf{x}_D, \mathbf{x}'_D \sim \mathcal{D}^N} \left[ \forall \mathbf{x}_T \in \mathcal{X}^N : err(\mathbf{x}_T, f(\mathbf{x}_T)) \leq \epsilon^* \wedge \Box(f(\mathbf{x}'_D)) \leq \epsilon^* \right] \geq 1 - \epsilon,$$

*where* ϵ <sup>∗</sup> = q 2d N log (2N) + <sup>1</sup> N log 1 ϵ *and* f = REJECTRON(xD, h(xD), x<sup>T</sup> , ϵ<sup>∗</sup> )*, where* f : X → {−1, +1, ⊔⊓} *and* d *denotes the VC-dimension on* H*.* REJECTRON *is defined in Figure 2. in [\(Gold](#page-12-8)[wasser et al., 2020\)](#page-12-8).*

REJECTRON is an algorithm that accepts a labeled training set (xD, h(xD)) and a test set x<sup>T</sup> and returns a classifier f, which might reject some inputs. The learning is successful if with a high probability f rejects a small fraction of D<sup>N</sup> and for every x<sup>T</sup> ∈ X <sup>N</sup> the error on labeled x's in x<sup>T</sup> is small.

I.3 ADVERSARIAL DEFENSE FOR BOUNDED VC-DIMENSION

We are ready to state the main result of this section.

Lemma 5 (*Adversarial Defense for bounded VC-dimension*). *Let* d ∈ N *and* H *be a binary hypothesis class on input space* X *of VC-dimension bounded by* d*. There exists an algorithm* B *such that for every* ϵ ∈ 0, 8 *,* D *over* X *and* h ∈ H *we have*

$$\mathbf{B} \in \text{DEFENSE} \left( (\mathcal{D}, h), \epsilon, q = \frac{d \log^2(d)}{\epsilon^3}, t = \infty, T = \text{poly} \left( \frac{d}{\epsilon} \right), l = 1 - \epsilon, c = 1 - \epsilon, s = \epsilon \right).$$

*Proof.* The proof is based on an algorithm from [Goldwasser et al.](#page-12-8) [\(2020\)](#page-12-8).

Construction of B. Let ϵ ∈ (0, 1) and

$$N := \frac{d \log^2(d)}{\epsilon^3}.$$

Let q := N. First, B, draws N labeled samples (xFRESH, h(xFRESH)). Next, it finds f ∈ H consistent with them and sends f to A. Importantly this computation is the same as the first step of REJECTRON.

Next, B receives as input x ∈ X <sup>q</sup> from A. B. Let ϵ ∗ := q 2d N log (2N) + <sup>1</sup> N log 1 ϵ . Next B runs f ′ = REJECTRON(xFRESH, h(xFRESH), x, ϵ<sup>∗</sup> ), where REJECTRON is starting from the second step of the algorithm (Figure 2 [\(Goldwasser et al., 2020\)](#page-12-8)). Importantly, for every x ∈ X , if f ′ (x) ̸= ⊔⊓ then f(x) = f ′ (x). In words, f ′ is equal to f everywhere where f ′ does not reject.

Finally B returns 1 if ⊔⊓(f ′ (x)) > 2 3 ϵ, and returns 0 otherwise.

B is a Defense. First, by the standard PAC theorem we have that with probability at least 1 − ϵ, err(f) ≤ ϵ 2 . This means that *correctness* holds with probability l = 1 − ϵ.

Note that with our setting of N, we have that

$$\epsilon^* \leq \frac{\epsilon}{2}.$$

Theorem [9](#page-32-1) guarantees that

- if x ∈ D<sup>q</sup> then with probability at least 1 − ϵ we have that

$$\Box(f'(\mathbf{x})) \leq \frac{\epsilon}{2}.$$

**1787**

**1821**

**1823 1824**

**1827**

- for every x ∈ X <sup>q</sup> with probability at least 1 − ϵ we have that

$$\text{err}(\mathbf{x}, f'(\mathbf{x})) \leq \frac{\epsilon}{2}.$$

To compute soundness we want to upper bound the probability that err(x, f(x)) > 2ϵ [12](#page-33-0) and b = 0. By construction of B if b = 0 then ⊔⊓(f ′ (x)) ≤ 2ϵ 3 , which means that with probability at least 1 − ϵ

$$\text{err}(\mathbf{x}, \mathbf{y}) \leq \frac{2\epsilon}{3} + \frac{\epsilon}{2} < 2\epsilon \text{ or } b = 1.$$

This translates to *soundness* holding with s = ϵ.

REJECTRON runs in polynomial time in the number of samples and makes O( 1 ϵ ) calls to an Empirical Risk Minimizer on H (that we assume runs in time polynomial in d), which implies the promised running time.

# J WATERMARKS EXIST

Lemma 6 (*Watermark for bounded VC-dimension against fast adversaries*). *For every* d ∈ N *there exists a distribution* D *and a binary hypothesis class* H *of VC-dimension* d *there exists* A *such that for any* ϵ ∈ 10000 d , 8 *if* h ∈ H *is taken uniformly at random from* H *then*

$$\mathbf{A} \in \text{WATERMARK} \left( (\mathcal{D}, h), \epsilon, q = O \left( \frac{1}{\epsilon} \right), T = O \left( \frac{d}{\epsilon} \right), t = \frac{d}{100}, l = 1 - \frac{1}{100}, c = 1 - \frac{2}{100}, s = \frac{56}{100} \right).$$

*Proof.* Let X = <sup>N</sup>. Let D be the uniform distribution over [N] for N = 100d 2 . Let H be the concept class of functions that have exactly d +1's in [N]. Note H has VC-dimension d. Let h ∈ H be the ground truth.

Construction of A. A works as follows. It draws n = O d ϵ samples from D labeled with h. Let's call them xTRAIN. Let

$$A := \{x \in [N] : \mathbf{x}_{\text{TRAIN}}, h(x) = +1\}, B := \{x \in [N] : x \in \mathbf{x}_{\text{TRAIN}}, h(x) = -1\}.$$

A takes a uniformly random subset A<sup>w</sup> ⊆ A of size q. It defines sets

$$A' := A \setminus A_w, \quad B' := B \cup A_w.$$

A computes f consistent with the training set {(x, +1) : x ∈ A′} ∪ {(x, −1) : x ∈ B′}. A samples S ∼ D<sup>q</sup> . It defines the watermark to be x := A<sup>w</sup> with probability <sup>1</sup> 2 and x := S with probability <sup>1</sup> 2 . A sends (f, x) to B. A can be implemented in time O d ϵ .

A is a Watermark. We claim that (f, x) constitutes a watermark.

It is possible to construct a watermark of prescribed size, i.e., find a subset A<sup>w</sup> of a given size, only if |A| ≥ q. The probability that a single sample from D is labeled +1 is <sup>d</sup> N , so by the Chernoff bound (Fact [2\)](#page-29-2) |A|, |B| > dn <sup>2</sup><sup>N</sup> ≥ q with probability 1 − <sup>100</sup> , where we used that n = O d ϵ , N = 100d 2 , q = O( 1 ϵ ).

Correctness. Let h ′ (x) := h(x) if x ∈ [N] \ A<sup>w</sup> and h ′ (x) := −h(x) otherwise. Note that h ′ has exactly d − q +1's in [N]. By construction, f is a classifier consistent with h ′ . By the PAC theorem we know that with probability 1 − <sup>100</sup> , f has an error at most ϵ wrt to h ′ (because the hypothesis class of functions with *at most* d +1's has a VC dimension of O(d)). h ′ differs from h on q points, so

$$\text{err}(f) \leq \epsilon + q/N = O\left(\epsilon + \frac{1}{\epsilon d^2}\right) = O(\epsilon). \quad (15)$$

<sup>12</sup>Note that we measure the error of f not f ′ .

Distinguishing of x and D<sup>q</sup> . Note that the distribution of A<sup>w</sup> is the same as the distribution of a uniformly random subset of [N] of size q (when taking into account the randomness of the choice of h ∼ U(H)). Observe that the probability that drawing q i.i.d. samples from U([N]) we encounter repetitions is at most

$$\frac{1}{N} + \frac{2}{N} + \dots + \frac{q}{N} \leq \frac{3q^2}{N} \leq \frac{1}{100},$$

because q < <sup>d</sup> <sup>100</sup> < √ N <sup>10</sup> . This means that <sup>1</sup> <sup>100</sup> is an information-theoretic upper bound on the distinguishing advantage between x = A<sup>w</sup> and D<sup>q</sup> .

Moreover, B has access to at most t samples and the probability that the set of samples B draws from D<sup>t</sup> and A<sup>w</sup> have empty intersection is at least 1 − <sup>100</sup> . It is because it is at least (1 − t N ) <sup>t</sup> ≥ (1 − √ 1 N ) √ N/<sup>10</sup> ≥ 1 − <sup>100</sup> , where we used that t < √ N <sup>10</sup> . [13](#page-34-0)

Note that by construction f maps all elements of A<sup>w</sup> to −1. The probability over the choice of F ∼ D<sup>q</sup> that F ⊆ h −1 ({−1}), i.e., all elements of F have true label −1, is at least

$$\left(1 - \frac{d}{N}\right)^q \geq 1 - \frac{1}{100}.$$

The three above observations and the union bound imply that the distinguishing advantage for distinguishing x from D<sup>q</sup> of B is at most <sup>4</sup> <sup>100</sup> and so the *undetectability* holds with s = 8 <sup>100</sup> .

Unremovability. Assume, towards contradiction with *unremovability*, that B can find y that with probability s ′ = 1 <sup>2</sup> + <sup>100</sup> satisfies err(x, y) ≤ 2ϵ. Notice, that err(Aw, f(Aw)) = 1 by construction. Consider an algorithm A for distinguishing A<sup>w</sup> from D<sup>q</sup> . Upon receiving (f, x) it first runs y = B(f, x) and returns 1 iff d(y, f(x)) ≥ q 2 . We know that the distinguishing advantage is at most 1 <sup>2</sup> + 4 <sup>100</sup> , so

$$\frac{1}{2}\mathbb{P}_{\mathbf{x}:=A_w}[\mathcal{A}(f, \mathbf{x}) = 1] + \frac{1}{2}\mathbb{P}_{\mathbf{x}\sim\mathcal{D}^q}[\mathcal{A}(f, \mathbf{x}) = 0] \leq \frac{1}{2} + \frac{4}{100}.$$

But also note that

$$\begin{aligned} s' &\leq \mathbb{P}_{\mathbf{x} \sim \mathbf{A}}[\text{err}(\mathbf{x}, \mathbf{y}) \leq 2\epsilon] \\ &\leq \frac{1}{2}\mathbb{P}_{\mathbf{x}:=A_w}[d(\mathbf{y}, f(\mathbf{x})) \geq (1 - 2\epsilon)q] + \frac{1}{2}\mathbb{P}_{\mathbf{x} \sim \mathcal{D}^q}[d(\mathbf{y}, f(\mathbf{x})) \leq (2\epsilon + \text{err}(f))q] \\ &\leq \frac{1}{2}\mathbb{P}_{\mathbf{x}:=A_w}[d(\mathbf{y}, f(\mathbf{x})) \geq q/2] + \frac{1}{2}\mathbb{P}_{\mathbf{x} \sim \mathcal{D}^q}[d(\mathbf{y}, f(\mathbf{x})) \leq q/2] + \frac{1}{100} \\ &\leq \frac{1}{2}\mathbb{P}_{\mathbf{x}:=A_w}[\mathcal{A}(f, \mathbf{x}) = 1] + \frac{1}{2}\mathbb{P}_{\mathbf{x} \sim \mathcal{D}^q}[\mathcal{A}(f, \mathbf{x}) = 0] + \frac{1}{100}. \end{aligned}$$

Combining the two above equations we get a contradiction and thus the *unremovability* holds with s ′ = 1 <sup>2</sup> + 6 <sup>100</sup> .

Uniqueness. The following B certifies *uniqueness*. It draws O d ϵ samples from D, let's call them x ′ <sup>T</sup>RAIN and trains f ′ consistent with it. By the PAC theorem err(f ′ ) ≤ ϵ with probability at least 1 − 1 <sup>100</sup> . Next upon receiving <sup>x</sup> ∈ X <sup>q</sup> = [N] q it returns y = f ′ (x). By the fact that x is a random subset of [N] of size q by the Chernoff bound, the union bound we know that err(x, y) = err(x, f′ (x)) ≤ 2ϵ with probability at least 1 − 2 <sup>100</sup> over the choice of h. This proves *uniqueness*.

<sup>13</sup>If the sets were not disjoint then B could see it as suspicious because f makes mistakes on all of Aw.