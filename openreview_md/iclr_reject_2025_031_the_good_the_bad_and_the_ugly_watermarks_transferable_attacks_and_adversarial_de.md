000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# The Good, The Bad And The Ugly: Watermarks, Transferable Attacks And Adver- Sarial Defenses

We formalize and extend existing definitions of backdoor-based watermarks and adversarial defenses as *interactive protocols* between two players. The existence of these schemes is inherently tied to the learning tasks for which they are designed. Our main result shows that for *almost every* learning task, at least one of the two - a watermark or an adversarial defense - exists. The term "almost every" indicates that we also identify a third, counterintuitive but necessary option, i.e., a scheme we call a *transferable attack*. By transferable attack, we refer to an efficient algorithm computing queries that look indistinguishable from the data distribution and fool all efficient defenders. To this end, we prove the necessity of a transferable attack via a construction that uses a cryptographic tool called homomorphic encryption. Furthermore, we show that any task that satisfies our notion of a transferable attack implies a *cryptographic primitive*, thus requiring the underlying task to be computationally complex. These two facts imply an "*equivalence*" between the existence of transferable attacks and cryptography. Finally, we show that the class of tasks of bounded VC-dimension has an adversarial defense, and a subclass of them has a watermark.

## 1 Introduction

A company invested considerable resources to train a new classifier f. They want to open-source f but also ensure that if someone uses f, it can be detected in a black-box manner. In other words, they want to embed a *watermark* into f.

1 Alice, an employee, is in charge of this project. Bob, a member of an AI Safety team, has a different task. His goal is to make f *adversarially robust*, i.e., to ensure it is hard to find queries that appear unsuspicious but cause f to make mistakes. Alice, after many unsuccessful approaches, reports to her boss that it might be inherently impossible to create a black-box watermark in f that cannot be removed. After a similar experience, Bob reports to his boss that, due to the sheer number of possible modes of attack, he was only able to produce an ever-growing, 'ugly' defense. One day, after discussing their respective projects, Alice and Bob realized that their projects are intimately connected. Alice said that her idea was to plant a backdoor in f, creating fA, so she could later craft queries with a *hidden trigger* that activates the backdoor, causing fA to misclassify, while remaining *indistinguishable* from standard queries. By sending these tailored queries in a black-box manner to a party suspected of using fA, she can detect whether fA is being used based on the responses triggered by her backdoor. But Bob realized that his defenses were trying to render such a situation impossible. One of his ideas for defense was to take f and then "smooth" its outputs to obtain fB, aiming for robustness against attacks. Bob noticed that this procedure removes some of the backdoor-based watermarks that Alice came up with. Conversely, Alice noticed that any f with a watermark that is difficult to remove implies that some models are inherently difficult to make robust. Alice and Bob realized that their challenges are two sides of the same coin: the impossibility of one task guarantees the success of the other.

Anonymous authors Paper under double-blind review

## Abstract

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 To do that, we formalize and extend existing definitions of watermarks and adversarial defenses, frame Alice and Bob's dynamic as a formal game, and show that this game is guaranteed to have at least one winner. Along the way to proving the main result, we identify a potential reason why this fact was not discovered earlier. There is also a third, counterintuitive but necessary option, i.e., *there* are tasks with neither a Watermark nor an Adversarial Defense. Imagine that Alice plays the following game. The game is played with respect to a specific learning task L = (D, h), where D is the data distribution and h is the ground truth. Alice sends queries to a player and receives their responses. She wins if the responses have a lot of errors and if the player cannot distinguish them from the queries from D. Importantly, whether she wins the game depends on how much compute and data Alice and the player have. If Alice wins the game against any player having the same amount of resources as her, then we call Alice's queries a Transferable Attack. Intuitively, the harder a query becomes, the easier it is to distinguish it from queries from D.

But this seems to indicate that it is hard to design Transferable Attacks. However, we provably show:
- An example of a Transferable Attack defined as above. Interestingly, the example uses tools from the field of cryptography, namely Fully Homomorphic Encryption (FHE) (Gentry, 2009). Notably, a Transferable Attack rules out Watermarks and Adversarial Defenses, thus constituting the third necessary option.

- That every Transferable Attack implies a certain *cryptographic primitive*, i.e., access to samples from the underlying task is enough to build essential parts of encryption systems. Thus, every task with a Transferable Attack has to be complex in the computational complexity theory sense.

Finally, we complement the above results with instantiations of Watermarks and Adversarial Defenses:
- We show the existence of an Adversarial Defense for all learning tasks with bounded Vapnik–Chervonenkis (VC) dimension, thereby ruling out Transferable Attacks in this regime.

- We give an example of a black-box Watermark for a class of learning tasks with bounded VC-dimension. Notably, in this case, both a Watermark and an Adversarial Defense exist.

## 2 Related Work

This paper lies at the intersection of machine learning theory, interactive proof systems, and cryptography. We review recent advances and related contributions from these areas that closely align with our research.

Interactive Proof Systems in Machine Learning. *Interactive Proof Systems* (Goldwasser & Sipser, 1986) have recently gained considerable attention in machine learning for their ability to formalize and verify complex interactions between agents, models, or even human participants. A key advancement in this area is the introduction of *Prover-Verifier Games* (PVGs) (Anil et al., 2021), which employ a game-theoretic approach to guide learning agents towards decision-making with verifiable outcomes. Building on PVGs, Kirchner et al. (2024) enhance this framework to improve the legibility of Large Language Models (LLMs) outputs, making them more accessible for human evaluation. Similarly, Wäldchen et al. (2024) apply the prover-verifier setup to offer interpretability guarantees for classifiers. This paper initiates a formal study of the above observation that backdoor-based watermarks and adversarial defenses span all possible scenarios. By scenarios, we refer to learning tasks that f is supposed to solve. Our main contribution is:
We prove that almost every learning task has at least one of the two:
A Watermark or an Adversarial Defense.

## 1.1 Contributions

x f x f y b Alice verifies robustness Bob proves defense Alice verifies if f **was stolen**
Bob proves innocence
(b)
(a)
x y Alice verifies transferability Bob proves defendability
(c)
Figure 1: Schematic overview of the interaction structure, along with short, informal versions of our definitions of (a) Watermark (Definition 1), (b) Adversarial Defense (Definition 2), and (c) Transferable Attack (Definition 3), with (c) tied to cryptography (see Section 5). Extending these concepts, self-proving models Amit et al. (2024) introduce generative models that not only produce outputs but also generate proof transcripts to validate their correctness. In the context of AI safety, scalable *debate protocols* (Condon et al., 1993; Irving et al., 2018; Brown-Cohen et al., 2023) leverage interactive proof systems to enable complex decision processes to be broken down into verifiable components, ensuring reliability even under adversarial conditions. Overall, these developments highlight the emerging role of interactive proof systems in addressing key aspects of AI Safety, such as interpretability, verifiability, and alignment. While current research predominantly focuses on applying this framework to improve these safety attributes, our approach takes an orthogonal direction by examining the *feasibility* of properties related to *adversarial robustness* and *backdoor-based watermarks*.

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 Planting Undetectable Backdoors. A key related work is presented by Goldwasser et al. (2022), which demonstrates how a learner can plant undetectable backdoors in any classifier, allowing hidden manipulation of the model's output with minimal perturbation of the input. These backdoors are activated by specific *"triggers"*, which are subtle changes to the input that cause the model to misclassify any input with the trigger applied, while maintaining its expected behavior on regular inputs. The authors propose two frameworks. The first utilizes digital signature schemes (Goldwasser et al., 1985) that make backdoored models indistinguishable from the original model to any computationally-bounded observer. The second involves Random Fourier Features (RFF) (Rahimi & Recht, 2007), which ensures undetectability even with full transparency of the model's weights and training data. In a concurrent and independent work, Christiano et al. (2024) introduce a defendability framework that formalizes the interaction between an attacker planting a backdoor and a defender tasked with detecting it. The attacker modifies a classifier to alter its behavior on a trigger input while leaving other inputs unaffected. The defender then attempts to identify this trigger during evaluation, and if successful with high probability, the function class is considered defendable. The authors show an equivalence between their notion of defendability (in a computationally unbounded setting) and Probably Approximately Correct (PAC) learnability, and thus the boundedness of the VC-dimension of a class. In computationally bounded cases, they propose that *efficient defendability* serves as an important intermediate concept between efficient learnability and obfuscation. A major difference between our work and that of Christiano et al. (2024), is that in their approach, the attacker chooses the distribution, whereas we keep the distribution fixed. This makes defendability in their model harder since the attacker has more control. However, in their framework, the backdoor trigger x
∗
162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 is sampled ∼ D, so the attacker does not influence it. In contrast, our model allows the attacker to choose specific x's, making defendability easier in this regard. Thus, the definitions are a priori incomparable. A second major difference is that our main result holds for all learning tasks, while their contributions hold only for restricted classes. This makes defendability in their model harder since the attacker has more control. However, in their framework, the backdoor trigger x
∗is sampled
∼ D, so the attacker does not influence it. In contrast, our model allows the attacker to choose specific x's, making defendability easier in this regard. Thus, the definitions are a priori incomparable. However, there are many interesting connections. Computationally unbounded defendability is shown to be equivalent to PAC learnability, while we, in a similar spirit, show an Adversarial Defense for all tasks with bounded VC-dimension. They show that efficient PAC learnability implies efficient defendability, and we show that the same fact implies an efficient Adversarial Defense. Using cryptographic tools, they show that the class of polynomial-size circuits is not efficiently defendable, while we use different cryptographic tools to give a Transferable Attack, which rules out a Defense. Backdoor-Based Watermarks. In black-box settings, where model auditors lack access to internal parameters, watermarking methods often involve embedding backdoors during training. Techniques by Adi et al. (2018) and Zhang et al. (2018) use crafted input patterns as triggers linked to specific outputs, enabling ownership verification by querying the model with these specific inputs. Advanced methods by Merrer et al. (2017) utilize adversarial examples, which are perturbed inputs that yield predefined outputs. Further enhancements by Namba & Sakuma (2019) focus on the robustness of watermarks, ensuring the watermark remains detectable despite model alterations or attacks.

In the domain of Natural Language Processing (NLP), backdoor-based watermarks have been studied for Pre-trained Language Models (PLMs), as exemplified by works such as (Gu et al., 2022; Peng et al., 2023) and (Li et al., 2023). These approaches embed backdoors using rare or common word triggers, ensuring watermark robustness across downstream tasks and resistance to removal techniques like fine-tuning or pruning. However, it is important to note that these lines of research are predominantly empirical, with limited theoretical exploration. Adversarial Robustness. As we emphasize, the study of backdoors is closely related to adversarial robustness, which focuses on improving model resilience to adversarial inputs. The extensive literature in this field includes key contributions such as *adversarial training* (Madry et al., 2018), which improves robustness by training on adversarial examples, and certified defenses (Raghunathan et al., 2018), which offer *provable guarantees* against adversarial attacks by ensuring prediction stability within specified perturbation bounds. Techniques like *randomized smoothing* (Cohen et al., 2019) extend these robustness guarantees. Notably, Goldwasser et al. (2022) show that some undetectable backdoors can, in fact, be removed by randomized smoothing, highlighting the intersection of adversarial robustness and backdoor methods.

## 3 Watermarks, Adversarial Defenses And Transferable Attacks

In this section, we outline interactive protocols between a verifier and a prover. Each protocol is designed to address specific tasks such as watermarking, adversarial defense, and transferable attacks. We first introduce the preliminaries before detailing the properties that each protocol must satisfy.

## 3.1 Preliminaries

Discriminative Learning Task. For n ∈ N, we define [n] := 0, 1*, . . . , n* − 1	. A learning task L is a pair (D, h) of a distribution D, supp(D) ⊆ X (the input space), and a ground truth map h: *X → Y ∪ {⊥}*, where Y is a finite space of labels and ⊥ represents a situation where h is not defined. To every f : *X → Y*, we associate err(f) := Ex∼D[f(x) ̸= h(x)]. We implicitly assume h does not map to ⊥ on supp(D). This definition of ⊥ is introduced for generality, as it becomes relevant in adversarial scenarios where samples may lie outside supp(D).

For q ∈ N, x ∈ X q, y ∈ Yq, we define

$$\operatorname{err}(\mathbf{x},\mathbf{y}):={\frac{1}{q}}\sum_{i\in[q]}\mathbb{1}\left\{h(x_{i})\neq y_{i},h(x_{i})\neq\perp\right\},$$

which means that we count (x, y) *∈ X × Y* as an error if h is well-defined on x and h(x) ̸= y.

## 3.2 Definitions

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 In our protocols, Alice (A, verifier) and Bob (B, prover) engage in interactive communication, with distinct roles depending on the specific task. Each protocol is defined with respect to a learning task L = (D, h), an error parameter ε ∈0, 1 2
, and time bounds TA and TB. A scheme is successful if the corresponding algorithm satisfies the desired properties with high probability, and we denote the set of such algorithms by SCHEME(L, ε, TA, TB), where SCHEME refers to WATERMARK, DEFENSE,
or TRANSFATTACK.

## Definition 1 (Watermark, Informal).

An algorithm AWATERMARK, running in time TA, implements a *watermarking scheme* for the learning task L with error parameter ϵ > 0, if an interactive protocol in which AWATERMARK computes a classifier f : *X → Y* and a sequence of queries x ∈ X q, and a prover B outputs y = B(f, x) ∈
Y
q, satisfies the following properties: **Alice**
(runs in TA)
Bob
(runs in TB)

x f y
1. **Correctness:** f has low error, i.e., err(f) ≤ ϵ.

Figure 2: Schematic overview of the interaction between Alice and Bob in *Watermark* (Definition 1).

2. **Uniqueness:** There exists a prover B, running in time bounded by TA, which provides low-error answers, such that err(x, y) ≤ 2ϵ.

3. **Unremovability:** For every prover B running in time TB, it holds that err(x, y) > 2ϵ.

4. **Undetectability:** For every prover B running in time TB, the advantage of B in distinguishing the queries x generated by AWATERMARK from random queries sampled from Dqis small.

Note that, due to *uniqueness*, we require that any defender, who *did not use* f and trained a model fScratch, must be accepted as a distinct model. This requirement is essential, as it mirrors real-world scenarios where independent models could have been trained within the given time constraint TA.

Additionally, the property enforces that any successful Watermark must satisfy the condition that Bob's time is strictly less than TA, i.e., TB < TA.

Definition 2 (*Adversarial Defense, informal*).

An algorithm BDEFENSE, running in time TB, implements an adversarial defense for the learning task L with error parameter ϵ > 0, if an interactive protocol in which BDEFENSE
computes a classifier f : *X → Y*, a verifier A replies with x = A(f), where x ∈ X q, and BDEFENSE outputs b = BDEFENSE(f, x) ∈ {0, 1}, satisfies the following properties:
Alice
(runs in TA)
Bob
(runs in TB)
1. **Correctness:** f has low error, i.e., err(f) ≤ ϵ.

2. **Completeness:** When x ∼ Dq, then b = 0.

3. **Soundness:** For every A running in time TA,
we have err(x, f(x)) ≤ 7ϵ or b = 1.

x f b
Figure 3: Schematic overview of the interaction between Alice and Bob in *Adversarial Defense* (Definition 2).

1. The sender samples a bit b ∼ U({0, 1}) and then draws a random sample x ∼ Db.

2. A receives x and outputs ˆb := A(x) ∈ {0, 1}. A wins if ˆb = b.

We say that δ ∈ (0, 1 2
) is the *advantage* of A for *distinguishing* D0 from D1 if:
Pb∼U({0,1}),x∼Db[A(x) = b] = 12 + δ. For a class of algorithms, we say that the two distributions D0 and D1 are δ-*indistinguishable* if for any algorithm in the class, its advantage is at most δ.

Advantage and Indistinguishability: For an algorithm A (also known as the distinguisher) and two distributions D0, D1, consider the following game between a sender and the distinguisher:
270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 The key requirement for a successful defense is the ability to *detect when it is being tested*. To bypass the defense, an attacker must provide samples that are both *adversarial*, causing the classifier to make mistakes, and *indistinguishable* from samples drawn from the data distribution D. Definition 3 (*Transferable Attack, informal*).

An algorithm ATRANSFATTACK, running in time TA, implements a *transferable attack* for the learning task L with error parameter ϵ > 0, if an interactive protocol in which ATRANSFATTACK computes x ∈ X qand B outputs y =
B(x) ∈ Yqsatisfies the following properties:**Alice**
(runs in TA)
Bob
(runs in TB)
1. **Transferability:** For every prover B running in time TA, we have err(x, y) > 2ϵ.

2. **Undetectability:** For every prover B running in time TB, the advantage of B in distinguishing the queries x generated by ATRANSFATTACK from random queries sampled from Dqis small.

x y
Figure 4: Schematic overview of the interaction between Alice and Bob in *Adversarial Defense* (Definition 3).

Verifiability of Watermarks. For a watermarking scheme AWATERMARK, if the *unremovability* property holds with a stronger guarantee, i.e., much larger than 2ϵ, then AWATERMARK could determine whether B had stolen f. To achieve this, AWATERMARK runs, after completing its interaction with B,
the procedure guaranteed by *uniqueness* to obtain y
′. It then verifies whether y and y
′ differ for many queries. If this condition is met, AWATERMARK concludes that B had stolen f.

2 Alternatively, if *unremovability* holds with 2ϵ, as originally defined, the test described above may fail. In this scenario, we consider an external party overseeing the interaction, potentially with knowledge of the distribution and h, who can directly compute the necessary errors to make a final decision. This setup is similar to the use of human judgment oracles in (Brown-Cohen et al., 2023). An interesting direction for future work would be to explore cases where the parties have access to *restricted* versions of error oracles. While this is beyond the scope of this work, we outline potential avenues for addressing this in Appendix E.

## 4 Main Result

We are ready to state an informal version of our main theorem. Please refer to Theorem 5 for the details and full proof. The key idea is to define a *zero-sum game* between A and B, where the actions of each player are the possible algorithms or circuits that can be implemented in the given time bound. Zero-sum games are not a modeling choice but a proof strategy, as they allow us to analyze the complementary nature of attacks on watermarks and adversarial defenses with clean mathematical guarantees. Specifically, the unique value of a zero-sum game eliminates concerns about equilibrium selection. Notably, this game is finite, but there are exponentially many such actions for each player. We rely on some key properties of such large zero-sum games (Lipton & Young, 1994b; Lipton et al., 2003) to argue about our main result. The formal statement and proof is deferred to Appendix D.

Theorem 1 (Main Theorem, informal). For every learning task L and ϵ ∈0, 1 2
, T ∈ N*, where a* learner exists that runs in time T and, with high probability, learns f satisfying err(f) ≤ ϵ, at least one of these three exists:

$$\begin{array}{c}\mbox{WATEMark}\left({\cal L},\epsilon,T,T^{1/\sqrt{\log(T)}}\right),\\ \mbox{DefENSE}\left({\cal L},\epsilon,T^{1/\sqrt{\log(T)}},O(T)\right),\end{array}$$
$${\mathrm{TransFATTACK}}{\stackrel{.}{\left(}{\mathcal{L}},\epsilon,T,T\right)}.$$

Proof (Sketch). The intuition of the proof relies on the complementary nature of Definitions 1 and 2. Specifically, every attempt to remove a fixed Watermark can be transformed to a potential Adversarial 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 Defense, and vice versa. We define a zero-sum game G between watermarking algorithms A and algorithms attempting to remove a watermark B. The use of a zero-sum game ensures that the value of the game is unique, allowing us to focus on the interplay between watermarking and adversarial defenses without ambiguity about equilibrium selection. The actions of each player are the class of algorithms that they can run in their respective time bounds, and the payoff is determined by the probability that the errors and rejections meet specific requirements. According to Nash's theorem, there exists a Nash equilibrium for this game, characterized by strategies ANASH and BNASH. This equilibrium framework simplifies the analysis since Nash equilibria are well-studied and provide tractable guarantees for two-player zero-sum games. A careful analysis shows that depending on the value of the game, we have a Watermark, an Adversarial Defense, or a Transferable Attack. In the first case, where the expected payoff at the Nash equilibrium is greater than a threshold, we show there is an Adversarial Defense. We define BDEFENSE as follows. BDEFENSE first learns a low-error classifier f, then sends f to the party that is attacking the Defense, then receives queries x, and simulates (y, b) = BNASH(f, x). The bit b = 1 if BNASH thinks it is attacked. Finally, BDEFENSE replies with b
′ = 1 if b = 1, and if b = 0 it replies with b
′ = 1 if the fraction of queries on which f(x) and y differ is high. Careful analysis shows BDEFENSE
is an Adversarial Defense. In the second case, where the expected payoff at the Nash equilibrium is below the threshold, we have either a Watermark or a Transferable Attack. The reason that there are two cases is due to the details of the definition of G. Full proof can be found in Appendix D. Our Definitions 1, 2, 3 and Theorem 1 are phrased with respect to a *fixed* learning task, while VC-theory takes an alternate viewpoint that tries to show guarantees on the risk (mostly sample complexity-based) for any distribution. However, for DNNs and other modern architectures, moving beyond classical VC-theory is necessary (Zhang et al., 2021; Nagarajan & Kolter, 2019). In our case, due to the requirements of our schemes (e.g., *unremovability* and *undetectability*), it may not be feasible to achieve a formalization that applies to all distributions, as in classical VC-theory. We end this section with the following observation. Fact 1 (*Transferable Attacks are disjoint from Watermarks and Adversarial Defenses*). For every learning task L and ϵ ∈0, 1 2
, T ∈ N, if TRANSFATTACK L*, ϵ, T, T*exists, then neither WATERMARK (L*, ϵ, T, o*(T)) nor DEFENSE (L*, ϵ, T, T*) exists.

This result follows straightforwardly from rephrasing the Definitions 1 to 3. Indeed, a Transferable Attack is a strong notion of an attack, so it rules out a Defense. Secondly, a Transferable Attack against defenders running in time T rules out a Watermark, since it is in conflict with *uniqueness*.

## 5 Transferable Attacks Are "Equivalent" To Cryptography

In this section, we show that tasks with Transferable Attacks exist. To construct such examples, we use cryptographic tools. But importantly, the fact that we use cryptography is not coincidental. As a second result of this section, we show that every learning task with a Transferable Attack *implies* a certain cryptographic primitive. One can interpret this as showing that Transferable Attacks exist only for *complex learning tasks*, in the sense of computational complexity theory. The two results together justify, why we can view Transferable Attacks and the existence of cryptography as "equivalent".

## 5.1 A Cryptography-Based Task With A Transferable Attack

Next, we give an example of a cryptography-based learning task with a Transferable Attack. The following is an informal statement of the first theorem of this section. The formal version (Theorem 7) is given in Appendix G. Theorem 2 (Transferable Attack for a Cryptography-based Learning Task, informal). There exists a learning task L
crypto with a distribution D and hypothesis class H, and A such that for all ϵ if h is sampled from H *then*

$$\mathbf{A}\in\mathrm{TransFATTACK}\left(\left({\mathcal{D}},h\right),\epsilon,T_{\mathbf{A}}\approx{\frac{1}{\epsilon}},T_{\mathbf{B}}={\frac{1}{\epsilon^{2}}}\right)$$

Moreover, the learning task is such that for every ϵ, ≈
1 ϵ time (and ≈
1 ϵ samples) is enough, and ≈
1 ϵ samples (and in particular time) is necessary to learn a classifier of error ϵ.

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 Notably, the parameters are set so that A (the party computing x) has *less* time than B (the party computing y), specifically ≈ 1/ϵ compared to 1/ϵ2. Furthermore, because of the encryption scheme, this is a setting where a single input maps to multiple outputs, which deviates away from the setting of classification learning tasks considered in Theorem 1. Proof (Sketch). We start with a definition of a learning task that will be later augmented with a cryptographic tool to produce L
crypto.

Lines on Circle Learning Task L
◦**(Figure 5).** Consider a binary classification task L
◦, where the input space is defined as X = {x ∈ R
2| ∥x∥2 = 1}, representing points on the unit circle. The hypothesis class is given by H = {hw | w ∈ R
2, ∥w∥2 = 1}, where each hypothesis is defined as hw(x) := sgn(⟨*w, x*⟩). The data distribution D is uniform on X , i.e., D = U(X ). Additionally, let Bw(α) := {x ∈ X | |∡(x, w)| ≤ α} denote the set of points within an angular distance up to α to w.

Fully Homomorphic Encryption (FHE) (Appendix F). FHE (Gentry, 2009) allows for computation on encrypted data *without* decrypting it. An FHE scheme allows to encrypt x via an efficient procedure ex = FHE.ENC(x), so that later, for any algorithm C, it is possible to run C on x homomorphically. More concretely, it is possible to produce an encryption of the result of running C on x, i.e., eC,x := FHE.EVAL(*C, e*x). Finally, there is a procedure FHE.DEC that, when given a secret key sk, can decrypt eC,x, i.e., y := FHE.DEC(sk, eC,x), where y is the result of running C on x. Crucially, encryptions of any two messages are indistinguishable for all efficient adversaries.

Cryptography-based Learning Task L
crypto **(Figure 5).** L
crypto is derived from *Lines on Circle* Learning Task L
◦. Let w ∈ X . We define the distribution as an equal mixture of two parts D =
1 2DCLEAR +
1 2DENC. The first part, i.e.,DCLEAR, is equal to x ∼ U(X ) with label y = hw(x). The second part, i.e.,DENC, is equal to x
′ ∼ U(X ), y′ = hw(x
′),(*x, y*) = (FHE.ENC(x
′), FHE.ENC(y
′)),
which can be thought of as DCLEAR under an encryption. See Figure 5 for a visual representation.

≈ ϵ Learning Task L
crypto **with distribution** D =
1 2DC**LEAR** +
1 2DENC :
1. x ∼ U(X *), b* ∼ Ber(1/2), where U(X ) is the uniform distribution on the circle 2. If b = 0:
Return (*x, h(x*))
3. Else:
Return (FHE.ENC(x), FHE.ENC(h(x)))
h x hA
y Alice
(runs in TA ≈ 1/ϵ)
Bob
(runs in TB = 1/ϵ2)
Figure 5: The left part of the figure represents a *Lines on Circle Learning Task* L
◦ with a ground truth function denoted by h. On the right, we define a *cryptography-augmented* learning task derived from L
◦. In its distribution, a "clear" or an "encrypted" sample is observed with equal probability. Given their respective times, both A and B are able to learn a low-error classifier h A, h B respectively, by learning only on the *clear samples*. A is able to compute a Transferable Attack by computing an encryption of a point close to the decision boundary of her classifier h A.

Transferable Attack (Figure 5). Consider the following attack strategy A. First, A collects O(1/ϵ) samples from the distribution DCLEAR and learns a classifier h A
w′ ∈ H that is consistent with these samples. Since the VC-dimension of H is 2, the hypothesis h A
w′ has error at most ϵ with high probability.3 Next, A samples a point xBND uniformly at random from a region close to the decision 3A can also evaluate h A
w′ homomorphically (i.e., run FHE.EVAL) on FHE.ENC(x) to obtain FHE.ENC(y)
of error ϵ on DENC also. This means that A is able to learn a low-error classifier on D.

boundary of h Aw′ , i.e., xBND ∼ U(Bw′ (ϵ)). Finally, with equal probability, A sets as an attack x either FHE.ENC(xBND) or a uniformly random point DCLEAR = U(X ). We claim that x 4satisfies the properties of a Transferable Attack. Since h A
w′ has low error with high probability, xBND is a uniformly random point from an arc containing the boundary of hw (see Figure 5). The running time of B is upper-bounded by 1/ϵ2, meaning it can only learn a classifier with error ⪆ 10ϵ 2(see Lemma 3 for details). B's can only learn
(Lemma 3) a classifier of error, ⪆ 10ϵ 2. Taking these two facts together, we expect B to misclassify x
′ with probability ≈
1 2
·
10ϵ 2 ϵ = 5ϵ > 2ϵ, where the factor 12 takes into account that we send an encrypted sample only half of the time. This implies *transferability*.

Note that x is encrypted with the same probability as in the original distribution because we send FHE.ENC(xBND) and a uniformly random x ∼ DCLEAR = U(X ) with equal probability. Crucially, FHE.ENC(xBND) is indistinguishable, for efficient adversaries, from FHE.ENC(x) for any other x ∈ X . This follows from the security of the FHE scheme. Consequently, *undetectability* holds. Note 1. *We want to emphasize that it is crucial (for our construction) that the distribution has both* an encrypted (DENC*) and an unencrypted part (*DCLEAR*). If there was no* DCLEAR, then A *would not* be able to generate FHE.ENC(xBND). The properties of the FHE would allow A *to learn a low-error* classifier h A
w′ *but only* under the FHE encryption. Although A *can produce encryptions of points* of her choice, she knows w
′ *only under encryption, so she does not know which point to encrypt! If* there was no DENC, then everything would happen in the clear and so B would be able to distinguish x*'s that appear too close to the boundary.*

## 5.2 Tasks With Transferable Attacks Imply Cryptography

In this section, we show that a Transferable Attack for any task implies a *cryptographic primitive*.

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

## 5.2.1 Efid Pairs

In cryptography, an *EFID pair* (Goldreich, 1990) is a pair of distributions D0, D1, that are Efficiently samplable, statistically Far, and computationally Indistinguishable. By a seminal result (Goldreich, 1990), we know that the existence of EFID pairs is equivalent to the existence of Pseudorandom Generators (PRG). A PRG is an efficient algorithm which stretches short seeds into longer output sequences such that the output distribution on a uniformly chosen seed is computationally indistinguishable from a uniform distribution. Together with what is known about PRGs, this implies that EFID pairs can be used for tasks in cryptography, including encryption and key generation (Goldreich, 1990).

For two time bounds *T, T*′ we call a pair of distributions (D0, D1) a (*T, T*′) EFID pair if (i) D0, D1 are samplable in time T, (ii) D0, D1 are statistically far, (iii) D0, D1 are indistinguishable for algorithms running in time T
′.

## 5.2.2 Tasks With Transferable Attacks Imply Efid Pairs

The second result of this section shows that any task with a Transferable Attack implies the existence of a type of EFID pair. The proof is deferred to Appendix H.

Theorem 3 (Tasks with Transferable Attacks imply EFID pairs, informal). For every *ϵ, T, T*′ ∈
N, T ≤ T
′, every learning task L *if there exists* A ∈ TRANSFATTACK L, ϵ, T, T′and there exists a learner running in time T that, with high probability, learns f such that err(f) ≤ ϵ, then there exists a (T, T′) *EFID pair.*

## 6 Tasks With Watermarks And Adversarial Defenses

In this section, we give examples of tasks with Watermarks and Adversarial Defenses. In the first example, we show that hypothesis classes of bounded VC-dimension have Adversarial Defenses against all attackers. The second example is a learning task of bounded VC-dimension that has 4In this proof sketch, we have q = 1, i.e., A sends only one x to B. This is not true for the formal scheme.

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 a Watermark, which is secure against fast adversaries. These lemmas demonstrate why the upper bounds on the running time of A and B are crucial parameters. Lemmas are proven in the appendix. The first lemma relies heavily on a result from Goldwasser et al. (2020). The authors give a defense against *arbitrary examples* in a transductive model with rejections. In contrast, our model does not allow rejections, but we do require indistinguishability. Careful analysis leads to the following result.

Lemma 1 (Adversarial Defense for bounded VC-Dimension, informal). Let d ∈ N and H *be a* binary hypothesis class on input space X of VC-dimension bounded by d. There exists an algorithm B *such that for every* ϵ ∈0, 1 8
, D over X and h ∈ H *we have*

$$\mathbf{B}\in\mathrm{{Defense}}\left((D,h),\epsilon,T_{\mathbf{A}}=\infty,T_{\mathbf{B}}=\mathrm{{poly}}\left({\frac{d}{\epsilon}}\right)\right)$$
 .
Note that, by the PAC learning bound, this is a setting of parameters, where B has enough time to learn a classifier of error ϵ. By slightly abusing the notation, we write TA = ∞, meaning that the defense is secure against all adversaries regardless of their running time. Lemma 2 (Watermark for bounded VC-Dimension against fast Adversaries, informal). *For every* d ∈ N there exists a distribution D and a binary hypothesis class H of VC-dimension d there exists A *such that for any* ϵ ∈10000 d2 ,
1 8 if h ∈ H is taken uniformly at random from H *then*

$$\mathbf{A}\in\mathrm{Watermark}\left(({\mathcal{D}},h),\epsilon,T_{\mathbf{A}}=O\left({\frac{d}{\epsilon}}\right),T_{\mathbf{B}}={\frac{d}{100}}\right)$$
.
Note that the setting of parameters is such that A can learn (with high probability) a classifier of error ϵ, but B is not able to learn a low-error classifier in its allotted time t. This contrasts with Lemma 5, where B has enough time to learn. This is the regime of interest for Watermarks, where the scheme is expected to be secure against fast B's.

## 7 Implications For Ai Safety

In contrast to years of adversarial robustness research (Carlini, 2024), we conjecture that for discriminative learning tasks encountered in safety-critical regimes, an Adversarial Defense *will* exist in the future. Three pieces of evidence support this contrarian belief. (i) Theorem 1, (ii) in the securitycritical scenarios for Watermarks, the security should hold even against strong defenders, i.e., TB approaching TA. In this regime, we believe an analog of Theorem 8 can be shown for Watermarks, given the similarity between the *unremovability* (Definition 1) and *transferability* (Definition 3) property. (iii) Transferable Attacks imply cryptography (Theorem 8), which we suspect is rare in practical scenarios.

Ali ce Bo b Bo b Alice Bob Alice

## References

Yossi Adi, Carsten Baum, Moustapha Cisse, Benny Pinkas, and Joseph Keshet. Turning your weakness into a strength: Watermarking deep neural networks by backdooring. In *27th USENIX* Security Symposium (USENIX Security 18), pp. 1615–1631, 2018.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Maksym Andriushchenko, Francesco Croce, and Nicolas Flammarion. Jailbreaking leading safetyaligned llms with simple adaptive attacks, 2024.

Cem Anil, Guodong Zhang, Yuhuai Wu, and Roger Grosse. Learning to give checkable answers with prover-verifier games. *arXiv preprint arXiv:2108.12099*, 2021.

Zvika Brakerski, Craig Gentry, and Vinod Vaikuntanathan. (leveled) fully homomorphic encryption without bootstrapping. In Proceedings of the 3rd Innovations in Theoretical Computer Science Conference, ITCS '12, pp. 309–325, New York, NY, USA, 2012. Association for Computing Machinery. ISBN 9781450311520. doi: 10.1145/2090236.2090262. URL https://doi.org/ 10.1145/2090236.2090262.

Jonah Brown-Cohen, Geoffrey Irving, and Georgios Piliouras. Scalable ai safety via doubly-efficient debate. *arXiv preprint arXiv:2311.14125*, 2023.

Collin Burns, Pavel Izmailov, Jan Hendrik Kirchner, Bowen Baker, Leo Gao, Leopold Aschenbrenner, Yining Chen, Adrien Ecoffet, Manas Joglekar, Jan Leike, Ilya Sutskever, and Jeffrey Wu.

Weak-to-strong generalization: Eliciting strong capabilities with weak supervision. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), *Proceedings of the 41st International Conference on Machine Learning*,
volume 235 of *Proceedings of Machine Learning Research*, pp. 4971–5012. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/burns24b.html.

Nicholas Carlini. Yet another broken defense: How AI security continues to fail, 2024. URL https:
//nicholas.carlini.com/writing/2024/yet-another-broken-defense.

html. Accessed: 2024-10-02.

Nicholas Carlini, Milad Nasr, Christopher A. Choquette-Choo, Matthew Jagielski, Irena Gao, Anas Awadalla, Pang Wei Koh, Daphne Ippolito, Katherine Lee, Florian Tramèr, and Ludwig Schmidt.

Are aligned neural networks adversarially aligned? *ArXiv*, abs/2306.15447, 2023. URL https:
//api.semanticscholar.org/CorpusID:259262181.

Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, and Eric Wong.

Jailbreaking black box large language models in twenty queries, 2023.

Jiefeng Chen, Yang Guo, Xi Wu, Tianqi Li, Qicheng Lao, Yingyu Liang, and Somesh Jha. Towards adversarial robustness via transductive learning. *arXiv preprint arXiv:2106.08387*, 2021.

Miranda Christ, Sam Gunn, and Or Zamir. Undetectable watermarks for language models. *arXiv* preprint arXiv:2306.09194, 2023.

Paul Christiano, Jacob Hilton, Victor Lecomte, and Mark Xu. Backdoor defense, learnability and obfuscation. *arXiv preprint arXiv:2409.03077*, 2024.

Jeremy Cohen, Elan Rosenfeld, and Zico Kolter. Certified adversarial robustness via randomized smoothing. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of *Proceedings of Machine Learning* Research, pp. 1310–1320. PMLR, 09–15 Jun 2019. URL https://proceedings.mlr. press/v97/cohen19c.html.

Anne Condon, Joan Feigenbaum, Carsten Lund, and Peter Shor. Probabilistically checkable debate systems and approximation algorithms for pspace-hard functions. In Proceedings of the twenty-fifth annual ACM symposium on Theory of Computing, pp. 305–314, 1993.

Noga Amit, Shafi Goldwasser, Orr Paradise, and Guy Rothblum. Models that prove their own correctness. *arXiv preprint arXiv:2405.15722*, 2024.

Bita Darvish Rouhani, Huili Chen, and Farinaz Koushanfar. Deepsigns: An end-to-end watermarking framework for ownership protection of deep neural networks. In Proceedings of the twenty-fourth international conference on architectural support for programming languages and operating systems, pp. 485–497, 2019.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Logan Engstrom, Dimitris Tsipras, Ludwig Schmidt, and Aleksander Madry. A rotation and a translation suffice: Fooling cnns with simple transformations. *ArXiv*, abs/1712.02779, 2017. URL https://api.semanticscholar.org/CorpusID:21929206.

Yousof Erfani, Ramin Pichevar, and Jean Rouat. Audio watermarking using spikegram and a twodictionary approach. *IEEE Transactions on Information Forensics and Security*, 12(4):840–852, 2017. doi: 10.1109/TIFS.2016.2636094.

Pierre Fernandez, Guillaume Couairon, Hervé Jégou, Matthijs Douze, and Teddy Furon. The stable signature: Rooting watermarks in latent diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 22466–22477, 2023.

A. Gammerman, V. Vovk, and V. Vapnik. Learning by transduction. In Proceedings of the Fourteenth Conference on Uncertainty in Artificial Intelligence, UAI'98, pp. 148–155, San Francisco, CA,
USA, 1998. Morgan Kaufmann Publishers Inc. ISBN 155860555X.

Craig Gentry. Fully homomorphic encryption using ideal lattices. In Proceedings of the Forty-First Annual ACM Symposium on Theory of Computing, STOC '09, pp. 169–178, New York, NY, USA,
2009. Association for Computing Machinery. ISBN 9781605585062. doi: 10.1145/1536414.

1536440. URL https://doi.org/10.1145/1536414.1536440.

Justin Gilmer, Luke Metz, Fartash Faghri, Samuel S. Schoenholz, Maithra Raghu, Martin Wattenberg, and Ian J. Goodfellow. Adversarial spheres. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Workshop Track Proceedings, 2018. URL https://openreview.net/forum?id=SkthlLkPf.

Surbhi Goel, Sham Kakade, Adam Kalai, and Cyril Zhang. Recurrent convolutional neural networks learn succinct learning algorithms. *Advances in Neural Information Processing Systems*, 35: 7328–7341, 2022.

Oded Goldreich. A note on computational indistinguishability. Information Processing Letters, 34(6):277–281, 1990. ISSN 0020-0190. doi: https://doi.org/10.1016/0020-0190(90) 90010-U. URL https://www.sciencedirect.com/science/article/pii/ 002001909090010U.

S Goldwasser and M Sipser. Private coins versus public coins in interactive proof systems. In Proceedings of the Eighteenth Annual ACM Symposium on Theory of Computing, STOC '86, pp. 59–68, New York, NY, USA, 1986. Association for Computing Machinery. ISBN 0897911938. doi: 10.1145/12130.12137. URL https://doi.org/10.1145/12130.12137.

S Goldwasser, S Micali, and C Rackoff. The knowledge complexity of interactive proof-systems. In Proceedings of the Seventeenth Annual ACM Symposium on Theory of Computing, STOC '85, pp. 291–304, New York, NY, USA, 1985. Association for Computing Machinery. ISBN 0897911512. doi: 10.1145/22145.22178. URL https://doi.org/10.1145/22145.22178.

Shafi Goldwasser, Yael Kalai, Raluca Ada Popa, Vinod Vaikuntanathan, and Nickolai Zeldovich.

Reusable garbled circuits and succinct functional encryption. In *Proceedings of the Forty-Fifth* Annual ACM Symposium on Theory of Computing, STOC '13, pp. 555–564, New York, NY, USA,
2013. Association for Computing Machinery. ISBN 9781450320290. doi: 10.1145/2488608. 2488678. URL https://doi.org/10.1145/2488608.2488678.

Yinpeng Dong, Tianyu Pang, Hang Su, and Jun Zhu. Evading defenses to transferable adversarial examples by translation-invariant attacks. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4307–4316, 2019. URL https://api.semanticscholar. org/CorpusID:102350868.

Shafi Goldwasser, Adam Tauman Kalai, Yael Tauman Kalai, and Omar Montasser. Beyond perturbations: Learning guarantees with arbitrary adversarial test examples. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS'20, Red Hook, NY, USA, 2020. Curran Associates Inc. ISBN 9781713829546.

Shafi Goldwasser, Michael P. Kim, Vinod Vaikuntanathan, and Or Zamir. Planting undetectable backdoors in machine learning models. *ArXiv*, abs/2204.06974, 2022. URL https://api. semanticscholar.org/CorpusID:248177888.

Chenxi Gu, Chengsong Huang, Xiaoqing Zheng, Kai-Wei Chang, and Cho-Jui Hsieh. Watermarking pre-trained language models with backdooring. *arXiv preprint arXiv:2210.07543*, 2022.

Geoffrey Irving, Paul Christiano, and Dario Amodei. Ai safety via debate, 2018. URL https:
//arxiv.org/abs/1805.00899.

Zhengyuan Jiang, Jinghuai Zhang, and Neil Zhenqiang Gong. Evading watermark based detection of ai-generated content. Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security, 2023. URL https://api.semanticscholar.org/CorpusID: 258557682.

648 649 650 651 652 653 654 655 656 657 658 659 660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein.

A watermark for large language models. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 17061–17084. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.

press/v202/kirchenbauer23a.html.

Jan Hendrik Kirchner, Yining Chen, Harri Edwards, Jan Leike, Nat McAleese, and Yuri Burda.

Prover-Verifier Games improve legibility of LLM outputs, 2024. URL https://arxiv.org/ abs/2407.13692.

Rohith Kuditipudi, John Thickstun, Tatsunori Hashimoto, and Percy Liang. Robust distortion-free watermarks for language models. *CoRR*, abs/2307.15593, 2023. doi: 10.48550/ARXIV.2307.15593.

URL https://doi.org/10.48550/arXiv.2307.15593.

Peixuan Li, Pengzhou Cheng, Fangqi Li, Wei Du, Haodong Zhao, and Gongshen Liu. Plmmark:
a secure and robust black-box watermarking framework for pre-trained language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pp. 14991–14999, 2023.

Richard J. Lipton and Neal E. Young. Simple strategies for large zero-sum games with applications to complexity theory. In *Proceedings of the Twenty-Sixth Annual ACM Symposium on Theory of* Computing, STOC '94, pp. 734–740, New York, NY, USA, 1994a. Association for Computing Machinery. ISBN 0897916638. doi: 10.1145/195058.195447. URL https://doi.org/10. 1145/195058.195447.

Richard J Lipton and Neal E Young. Simple strategies for large zero-sum games with applications to complexity theory. In *Proceedings of the twenty-sixth annual ACM symposium on Theory of* computing, pp. 734–740, 1994b.

Richard J Lipton, Evangelos Markakis, and Aranyak Mehta. Playing large games using simple strategies. In *Proceedings of the 4th ACM Conference on Electronic Commerce*, pp. 36–41, 2003.

Chang Liu, Jie Zhang, Han Fang, Zehua Ma, Weiming Zhang, and Nenghai Yu. Dear: A deeplearning-based audio re-recording resilient watermarking. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pp. 13201–13209, 2023.

Yanpei Liu, Xinyun Chen, Chang Liu, and Dawn Song. Delving into transferable adversarial examples and black-box attacks. *arXiv preprint arXiv:1611.02770*, 2016.

Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu.

Towards deep learning models resistant to adversarial attacks. In *6th International Conference on* Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net, 2018. URL https://openreview.net/forum?id=
rJzIBfZAb.

Anay Mehrotra, Manolis Zampetakis, Paul Kassianik, Blaine Nelson, Hyrum Anderson, Yaron Singer, and Amin Karbasi. Tree of attacks: Jailbreaking black-box llms automatically, 2024.

Erwan Le Merrer, Patrick Pérez, and Gilles Trédan. Adversarial frontier stitching for remote neural network watermarking. *Neural Computing and Applications*, 32:9233 - 9244, 2017. URL https://api.semanticscholar.org/CorpusID:11008755.

Omar Montasser, Steve Hanneke, and Nathan Srebro. Transductive robust learning guarantees. In International Conference on Artificial Intelligence and Statistics, pp. 11461–11471. PMLR, 2022.

Yuki Nagai, Yusuke Uchida, Shigeyuki Sakazawa, and Shin'ichi Satoh. Digital watermarking for deep neural networks. *International Journal of Multimedia Information Retrieval*, 7:3–16, 2018.

Vaishnavh Nagarajan and J Zico Kolter. Uniform convergence may be unable to explain generalization in deep learning. *Advances in Neural Information Processing Systems*, 32, 2019.

Ryota Namba and Jun Sakuma. Robust watermarking of neural network with exponential weighting.

Proceedings of the 2019 ACM Asia Conference on Computer and Communications Security, 2019.

URL https://api.semanticscholar.org/CorpusID:58028915.

Noam Nisan. Pseudorandom generators for space-bounded computations. In Proceedings of the twenty-second annual ACM symposium on Theory of computing, pp. 204–212, 1990.

Wenjun Peng, Jingwei Yi, Fangzhao Wu, Shangxi Wu, Bin Zhu, Lingjuan Lyu, Binxing Jiao, Tong Xu, Guangzhong Sun, and Xing Xie. Are you copying my model? protecting the copyright of large language models for eaas via backdoor watermark. *arXiv preprint arXiv:2305.10036*, 2023.

Aditi Raghunathan, Jacob Steinhardt, and Percy Liang. Certified defenses against adversarial examples. In *6th International Conference on Learning Representations, ICLR 2018, Vancouver,* BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net, 2018. URL https://openreview.net/forum?id=Bys4ob-Rb.

Ali Rahimi and Benjamin Recht. Random features for large-scale kernel machines.

In J. Platt, D. Koller, Y. Singer, and S. Roweis (eds.), Advances in Neural Information Processing Systems, volume 20. Curran Associates, Inc., 2007. URL https://proceedings.neurips.cc/paper_files/paper/2007/file/ 013a006f03dbc5392effeb8f18fda755-Paper.pdf.

Oded Regev. On lattices, learning with errors, random linear codes, and cryptography. In *Proceedings* of the thirty-seventh annual ACM symposium on Theory of computing, pp. 84–93. ACM, 2005.

R. Rivest, L. Adleman, and M. Dertouzos. On data banks and privacy homomorphisms. In Foundations of Secure Computation, pp. 169–179, New York, NY, USA, 1978. Academic Press.

Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian J. Goodfellow, and Rob Fergus. Intriguing properties of neural networks. In Yoshua Bengio and Yann LeCun (eds.), 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings, 2014. URL http://arxiv.org/abs/
1312.6199.

702 703 704 705 706 707 708 709 710 711 712 713 714 715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 Stuart A. Thompson Tiffany Hsu. Disinformation researchers raise alarms about a.i. chatbots.

https://scottaaronson.blog/?p=6823, 2023. Accessed: March 2024.

Florian Tramer, Nicholas Carlini, Wieland Brendel, and Aleksander Madry. On adaptive attacks to adversarial example defenses. *Advances in neural information processing systems*, 33:1633–1645, 2020.

Yusuke Uchida, Yuki Nagai, Shigeyuki Sakazawa, and Shin'ichi Satoh. Embedding watermarks into deep neural networks. In Proceedings of the 2017 ACM on international conference on multimedia retrieval, pp. 269–277, 2017.

Vinod Vaikuntanathan. Computing blindfolded: New developments in fully homomorphic encryption.

In *Proceedings of the 2011 IEEE 52nd Annual Symposium on Foundations of Computer Science*, FOCS '11, pp. 5–16, Washington, DC, USA, 2011. IEEE Computer Society. ISBN 9780769543001. doi: 10.1109/FOCS.2011.98. URL https://doi.org/10.1109/FOCS.2011.98.

Stephan Wäldchen, Kartikey Sharma, Berkant Turan, Max Zimmer, and Sebastian Pokutta. Interpretability Guarantees with Merlin-Arthur Classifiers. In International Conference on Artificial Intelligence and Statistics, pp. 1963–1971. PMLR, 2024.

Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. Jailbroken: How does llm safety training fail?

ArXiv, abs/2307.02483, 2023. URL https://api.semanticscholar.org/CorpusID:
259342528.

756 757 758 759 760 761 762 763 764 765 766 767 768 769 770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 Yuxin Wen, Neel Jain, John Kirchenbauer, Micah Goldblum, Jonas Geiping, and Tom Goldstein.

Hard prompts made easy: Gradient-based discrete optimization for prompt tuning and discovery. In A. Oh, T. Neumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), *Advances in* Neural Information Processing Systems, volume 36, pp. 51008–51025. Curran Associates, Inc.,
2023a. URL https://proceedings.neurips.cc/paper_files/paper/2023/ file/a00548031e4647b13042c97c922fadf1-Paper-Conference.pdf.

Yuxin Wen, John Kirchenbauer, Jonas Geiping, and Tom Goldstein. Tree-ring watermarks: Fingerprints for diffusion images that are invisible and robust. *ArXiv*, abs/2305.20030, 2023b. URL https://api.semanticscholar.org/CorpusID:258987524.

Eric Wong and J. Zico Kolter. Provable defenses against adversarial examples via the convex outer adversarial polytope. In Jennifer G. Dy and Andreas Krause (eds.), Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmässan, Stockholm, Sweden, July 10-15, 2018, volume 80 of *Proceedings of Machine Learning Research*, pp. 5283–
5292. PMLR, 2018. URL http://proceedings.mlr.press/v80/wong18a.html.

Yi-Hsuan Wu, Chia-Hung Yuan, and Shan-Hung Wu. Adversarial robustness via runtime masking and cleansing. In Hal Daumé III and Aarti Singh (eds.), *Proceedings of the 37th International* Conference on Machine Learning, volume 119 of *Proceedings of Machine Learning Research*, pp.

10399–10409. PMLR, 13–18 Jul 2020. URL https://proceedings.mlr.press/v119/
wu20f.html.

Cihang Xie, Zhishuai Zhang, Jianyu Wang, Yuyin Zhou, Zhou Ren, and Alan Loddon Yuille.

Improving transferability of adversarial examples with input diversity. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 2725–2734, 2018. URL https:
//api.semanticscholar.org/CorpusID:3972825.

Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning (still) requires rethinking generalization. *Communications of the ACM*, 64(3):107–115, 2021.

Hanlin Zhang, Benjamin L. Edelman, Danilo Francati, Daniele Venturi, Giuseppe Ateniese, and Boaz Barak. Watermarks in the sand: Impossibility of strong watermarking for generative models. *arXiV*, abs/2311.04378, 2023. doi: 10.48550/ARXIV.2311.04378. URL https:
//doi.org/10.48550/arXiv.2311.04378.

Jialong Zhang, Zhongshu Gu, Jiyong Jang, Hui Wu, Marc Ph. Stoecklin, Heqing Huang, and Ian Molloy. Protecting intellectual property of deep neural networks with watermarking. In Proceedings of the 2018 on Asia Conference on Computer and Communications Security, ASI- ACCS '18, pp. 159–172, New York, NY, USA, 2018. Association for Computing Machinery. ISBN 9781450355766. doi: 10.1145/3196494.3196550. URL https://doi.org/10.1145/ 3196494.3196550.

Xuandong Zhao, Prabhanjan Ananth, Lei Li, and Yu-Xiang Wang. Provable robust watermarking for ai-generated text. *CoRR*, abs/2306.17439, 2023a. doi: 10.48550/ARXIV.2306.17439. URL https://doi.org/10.48550/arXiv.2306.17439.

Xuandong Zhao, Kexun Zhang, Yu-Xiang Wang, and Lei Li. Invisible image watermarks are provably removable using generative ai. 2023b. URL https://api.semanticscholar. org/CorpusID:259075167.

Yunqing Zhao, Tianyu Pang, Chao Du, Xiao Yang, Ngai-Man Cheung, and Min Lin. A recipe for watermarking diffusion models. *ArXiv*, abs/2303.10137, 2023c. URL https://api. semanticscholar.org/CorpusID:257622907.

## A Additional Methods In Related Work

810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 Andy Zou, Zifan Wang, J. Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models. *ArXiv*, abs/2307.15043, 2023. URL https://api. semanticscholar.org/CorpusID:260202961.

This section provides an overview of the main areas relevant to our work: Watermarking techniques, adversarial defenses, and transferable attacks on Deep Neural Networks (DNNs). Each subsection outlines important contributions and the current state of research in these areas, offering additional context and details beyond those covered in the main body

## A.1 Watermarking

Watermarking techniques are crucial for protecting the intellectual property of machine learning models. These techniques can be broadly categorized based on the type of model they target. We review watermarking schemes for both discriminative and generative models, with a primary focus on discriminative models, as our work builds upon these methods.

## A.1.1 Watermarking Schemes For Discriminative Models

Discriminative models, which are designed to categorize input data into predefined classes, have been a major focus of watermarking research. The key approaches in this domain can be divided into black-box and white-box approaches. Black-Box Setting. In the black-box setting, the model owner does not have access to the internal parameters or architecture of the model, but can query the model to observe its outputs. This setting has seen the development of several watermarking techniques, primarily through backdoor-like methods. Adi et al. (2018) and Zhang et al. (2018) proposed frameworks that embed watermarks using specifically crafted input data (e.g., unique patterns) with predefined outcomes. These watermarks can be verified by feeding these special inputs into the model and checking for the expected outputs, thereby confirming ownership. Another significant contribution in this domain is by Merrer et al. (2017), who introduced a method that employs adversarial examples to embed the backdoor. Adversarial examples are perturbed inputs that cause the model to produce specific outputs, thus serving as a watermark. Namba & Sakuma (2019) further enhanced the robustness of black-box watermarking schemes by developing techniques that withstand various model modifications and attacks. These methods ensure that the watermark remains intact and detectable even when the model undergoes transformations. Provable undetectability of backdoors was achieved in the context of classification tasks by Goldwasser et al. (2022). Unfortunately, it is known ((Goldwasser et al., 2022)) that some undetectable watermarks are easily removed by simple mechanisms similar to randomized smoothing (Cohen et al., 2019). The popularity of black-box watermarking is due to its practical applicability, as it does not require access to the model's internal workings. This makes it suitable for scenarios where models are deployed as APIs or services. Our framework builds upon these black-box watermarking techniques. White-Box Setting. In contrast, the white-box setting assumes that the model owner has full access to the model's parameters and architecture, allowing for direct examination to confirm ownership. The initial methodologies for embedding watermarks into the weights of DNNs were introduced by Uchida et al. (2017) and Nagai et al. (2018). Uchida et al. (2017) presented a framework for embedding watermarks into the model weights, which can be examined to confirm ownership. An advancement in white-box watermarking is provided by Darvish Rouhani et al. (2019), who developed a technique to embed an N-bit (N ≥ 1) watermark in DNNs. This technique is both dataand *model-dependent*, meaning the watermark is activated only when specific data inputs are fed into 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 the model. For revealing the watermark, activations from intermediate layers are necessary in the case of white-box access, whereas only the final layer's output is needed for black-box scenarios. Our work does not focus on white-box watermarking techniques. Instead, we concentrate on exploring the interaction between backdoor-like watermarking techniques, adversarial defenses, and transferable attacks. Overall, watermarking through backdooring has become more popular due to its applicability in the black-box setting. A.1.2 WATERMARKING SCHEMES FOR GENERATIVE MODELS Watermarking techniques for generative models have attracted considerable attention with the advent of Large Language Models (LLMs) and other advanced generative models. This increased interest has led to a surge in research and diverse contributions in this area. Backdoor-Based Watermarking for Pre-trained Language Models. In the domain of Natural Language Processing (NLP), backdoor-based watermarks have been increasingly studied for Pretrained Language Models (PLMs), as exemplified by works such as (Gu et al., 2022) and (Li et al., 2023). These methods leverage rare or common word triggers to embed watermarks, ensuring that they remain robust across downstream tasks and resilient to removal techniques like fine-tuning or pruning. While these approaches have demonstrated promising results in practical applications, they are primarily empirical, with theoretical aspects of watermarking and robustness requiring further exploration. Watermarking the Output of LLMs. Watermarking the generated text of LLMs is critical for mitigating potential harms. Significant contributions in this domain include (Kirchenbauer et al., 2023), who proposed a watermarking framework that embeds signals into generated text that are invisible to humans but detectable algorithmically. This method promotes the use of a randomized set of "green" tokens during text generation, and detects the watermark without access to the language model API or parameters. Kuditipudi et al. (2023) introduced robust distortion-free watermarks for language models. Their method ensures that the watermark does not distort the generated text, providing robustness against various text manipulations while maintaining the quality of the output. Zhao et al. (2023a) presented a provable, robust watermarking technique for AI-generated text. This approach offers strong theoretical guarantees for the robustness of the watermark, making it resilient against attempts to remove or alter it without significantly changing the generated text. However, Zhang et al. (2023) highlighted vulnerabilities in these watermarking schemes. Their work demonstrates that current watermarking techniques can be effectively broken, raising important considerations for the future development of robust and secure watermarking methods for LLMs. Image Generation Models. Various watermarking techniques have been developed for image generation models to address ethical and legal concerns. Fernandez et al. (2023) introduced a method combining image watermarking with Latent Diffusion Models, embedding invisible watermarks in generated images for future detection. This approach is robust against modifications such as cropping. Wen et al. (2023b) proposed Tree-Ring Watermarking, which embeds a pattern into the initial noise vector during sampling, making the watermark robust to transformations like convolutions and rotations. Jiang et al. (2023) highlighted vulnerabilities in watermarking schemes, showing that human-imperceptible perturbations can evade watermark detection while maintaining visual quality. Zhao et al. (2023c) provided a comprehensive analysis of watermarking techniques for Diffusion Models, offering a recipe for efficiently watermarking models like Stable Diffusion, either through training from scratch or fine-tuning. Additionally, Zhao et al. (2023b) demonstrated that invisible watermarks are vulnerable to regeneration attacks that remove watermarks by adding random noise and reconstructing the image, suggesting a shift towards using semantically similar watermarks for better resilience. Audio Generation Models. Watermarking techniques for audio generators have been developed for robustness against various attacks. Erfani et al. (2017) introduced a spikegram-based method, embedding watermarks in high-amplitude kernels, robust against MP3 compression and other attacks while preserving quality. Liu et al. (2023) proposed DeAR, a deep-learning-based approach resistant to audio re-recording (AR) distortions.

## A.2 Adversarial Defense

918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 The field of adversarial robustness has a rich and extensive literature (Szegedy et al., 2014; Gilmer et al., 2018; Raghunathan et al., 2018; Wong & Kolter, 2018; Engstrom et al., 2017). Adversarial defenses are essential for ensuring the security and reliability of machine learning models against adversarial attacks that aim to deceive them with carefully crafted inputs.

For discriminative models, there has been significant progress in developing adversarial defenses.

Techniques such as adversarial training (Madry et al., 2018), which involves training the model on adversarial examples, have shown promise in improving robustness. Certified defenses (Raghunathan et al., 2018) provide provable guarantees against adversarial attacks, ensuring that the model's predictions remain unchanged within a specified perturbation bound. Additionally, methods like randomized smoothing (Cohen et al., 2019) offer robustness guarantees. A particularly relevant work for our study is (Goldwasser et al., 2020), which considers a different model for generating adversarial examples. This approach has significant implications for the robustness of watermarking techniques in the face of adversarial attacks. In the context of Large Language Models (LLMs), there is a rapidly growing body of research focused on identifying adversarial examples (Zou et al., 2023; Carlini et al., 2023; Wen et al., 2023a). This research is closely related to the notion of *jailbreaking* (Andriushchenko et al., 2024; Chao et al., 2023; Mehrotra et al., 2024; Wei et al., 2023), which involves manipulating models to bypass their intended constraints and protections.

## A.3 Transferable Attacks And Transductive Learning

Transferable attacks refer to adversarial examples that are effective across multiple models. Moreover, transductive learning has been explored as a means to enhance adversarial robustness, and since our Definition 3 captures some notion of transductive learning in the context of Transferable Attacks, we highlight significant contributions in these areas. Adversarial Robustness via Transductive Learning. Transductive learning (Gammerman et al.,
1998) has shown promise in improving the robustness of models by utilizing both training and test data during the learning process. This approach aims to make models more resilient to adversarial perturbations encountered at test time. One significant contribution is by Goldwasser et al. (2020), which explores learning guarantees in the presence of arbitrary adversarial test examples, providing a foundational framework for transductive robustness. Another notable study by Chen et al. (2021) formalizes transductive robustness and proposes a bilevel attack objective to challenge transductive defenses, presenting both theoretical and empirical support for transductive learning's utility. Additionally, Montasser et al. (2022) introduce a transductive learning model that adapts to perturbation complexity, achieving a robust error rate proportional to the VC dimension. The method by Wu et al. (2020) improves robustness by dynamically adjusting the network during runtime to mask gradients and cleanse non-robust features, validated through experimental results. Lastly, Tramer et al. (2020) critique the standard of adaptive attacks, demonstrating the need for specific tuning to effectively evaluate and enhance adversarial defenses.

Transferable Attacks on DNNs. Transferable attacks exploit the vulnerability of models to adversarial examples that generalize across different models. For discriminative models, significant works include Liu et al. (2016), which investigates the transferability of adversarial examples and their effectiveness in black-box attack scenarios, (Xie et al., 2018), who propose input diversity techniques to enhance the transferability of adversarial examples across different models, and (Dong et al., 2019), which presents translation-invariant attacks to evade defenses and improve the effectiveness of transferable adversarial examples.

972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989 990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 In the context of generative models, including large language models (LLMs) and other advanced generative architectures, relevant research is rapidly emerging, focusing on the transferability of adversarial attacks. This area is crucial as it aims to understand and mitigate the risks associated with adversarial examples in these powerful models. Notably, Zou et al. (2023) explored universal and transferable adversarial attacks on aligned language models, highlighting the potential vulnerabilities and the need for robust defenses in these systems.

| Undetectability                                | Unremovability                         | Uniqueness                       |      |      |
|------------------------------------------------|----------------------------------------|----------------------------------|------|------|
| Goldwasser et al. (2022)                       | "                                      | robust to some smoothing attacks | "(E) |      |
| tion casifiClas                                | Adi et al. (2018); Zhang et al. (2018) | "(E)                             | %    | "(E) |
| Merrer et al. (2017)                           | "(E)                                   | robust to fine tunning attacks   | "(E) |      |
| Christ et al. (2023); Kuditipudi et al. (2023) | "                                      | %                                | "    |      |
| Zhao et al. (2023a)                            | %                                      | robust to edit                   |      |      |
| distance attacks only                          | "                                      |                                  |      |      |
| s M                                            | Tiffany Hsu (2023)                     | "(E)                             | %    | "    |
| LL                                             | Kirchenbauer et al. (2023)             | %                                | %    | "    |

Table 1: Overview of properties across various watermarking schemes. The symbol " denotes properties with formal guarantees or where proof is plausible, whereas % indicates the absence of such guarantees. Entries marked with "(E) represent properties observed empirically; these lack formal proof in the corresponding literature, suggesting that deriving such proof may present substantial challenges. The LLM watermarking schemes refer to those applied to text generated by these models.

## B Preliminaries

Learning. For a set Ω, we write ∆(Ω) to denote the set of all probability measures defined on the measurable space (Ω, F), where F is some fixed σ-algebra that is implicitly understood. We denote by X the domain and by Y the label space. A *model* is a function f : *X → Y*.

Definition 4 (*Learning task*). For a fixed X , Y a *learning task* is an element of ∆∆(X ) × YX.

We will often use L to denote a learning task.

For a distribution D ∈ ∆(X ) and a ground truth h : *X → Y*, we define an *error* of f as errD,h(f) :=
Ex∼D[f(x) ̸= h(x)], where the index of err will often be understood implicitly and omitted in notation. For D ∈ ∆(X ), h : *X → Y* we define an example oracle Ex(D, h) as an oracle that samples x ∼ D and returns (*x, h*(x)).

## B.1 Discussion

Communication. When Ex(D, h) generates (x, h(x)) it is encoded as a bit-string of some length.

For a message space M a *representation class* over (X , Y) is a mapping R : *M → Y*X .

Computation. Let U be a universal Turing Machine. Definition 4 models a learner's prior knowledge of the learning task as a distribution over pairs (D, h), i.e. over pairs of distributions over the domain X and ground truths h : *X → Y*. It can be viewed as a generalization of, for instance, PAC-Bayes, where priors are distributions over hypothesis spaces. For us prior knowledge (what we call a learning task) is a distribution over not only hypotheses but also distributions themselves. Note that we consider a realizable scenario as there is a fixed ground truth. We could have considered a more general case, i.e. agnostic learning, where a learning task 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079 would be an element of ∆ (∆(*X × Y*)). We chose the former for simplicity and we believe most of the results would generalize to the agnostic case. When Ex(D, h) generates (*x, h*(x)) it is encoded is some form, e.g. x ∈ {0, 1}
n, but importantly n is not a parameter that the learner can control, i.e. the encoding is fixed. This precludes thinking of n as a security parameter that the watermarking party can increase to boost the security.

## C Formal Definitions

Definition 5 (*Succinct Circuits*). Let C be a circuit of width w and depth d. We will denote size(C) :=
w · d. We say that C is *succinctly representable* if there exists a circuit of size 100 log(size(C))5 that accepts as input i ∈ [w], j, j1, j2 ∈ [d], g ∈ [O(1)], where g represents a gate from a universal constant-sized gate set, and returns 0 or 1, depending if g appears in location (i, j) in C and if it is connected to gates in locations (i − 1, j1) and (i − 1, j2).

We are ready to state formal versions of our main definitions.

Definition 6 (*Watermark*). Let L = (D, h) be a learning task. Let T, t, q ∈ N, ϵ ∈0, 1 2
*, l, c, s* ∈
(0, 1)*, s < c*, where t bounds the running time of B, and T the running time of A, q the number of queries, ϵ the risk level, c probability that *uniqueness* holds, s probability that *unremovability* and undetectability holds, l the learning probability. We say that a succinctly representable circuit AWATERMARK *of size* T implements a watermarking scheme for L, denoted by AWATERMARK ∈ WATERMARK(L*, ϵ, q, T, t, l, c, s*), if an interactive protocol in which AWATERMARK computes (f, x), f : X → Y, x ∈ X q, and B outputs y = B(f, x), y ∈ Yq satisfies the following
- **Correctness** (f has low error). With probability at least l err(f) ≤ ϵ.

- **Uniqueness** (models trained from scratch give low-error answers). There exists a succinctly representable circuit B of size T such that with probability at least c err(x, y) ≤ 2ϵ.

- **Unremovability** (fast B gives high-error answers). For every succinctly representable circuit B *of size at most* t we have that with probability at most s err(x, y) ≤ 2ϵ.

- **Undetectability** (fast B cannot detect that they are tested). Distributions Dqand x ∼
AWATERMARK are s2
-indistinguishable for a class of succinctly representable circuits B of size at most t.

Definition 7 (*Adversarial Defense*). Let L = (D, h) be a learning task. Let T, t, q ∈ N, ϵ ∈
0, 1 2
, l, c, s ∈ (0, 1)*, s < c*, where t bounds the running time of A, and T the running time of B,
q the number of queries, ϵ the error parameter, c the completeness, s the soundness, l the learning probability.

We say that a succinctly representable circuit BDEFENSE of size T implements an adversarial defense for L, denoted by BDEFENSE ∈ DEFENSE(L*, ϵ, q, t, T, l, c, s*), if an interactive protocol in which BDEFENSE computes f : *X → Y*, A replies with x = A(f), x ∈ X q, and BDEFENSE outputs b = BDEFENSE(f, x), b ∈ {0, 1} satisfies the following.

- **Correctness** (f has low error). With probability at least l err(f) ≤ ϵ.