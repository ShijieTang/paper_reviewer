# Auditing f-Differential Privacy in One Run

Saeed Mahloujifar <sup>1</sup> Luca Melis <sup>1</sup> Kamalika Chaudhuri <sup>1</sup>

Empirical auditing has emerged as a means of catching some of the flaws in the implementation of privacy-preserving algorithms. Existing auditing mechanisms, however, are either computationally inefficient – requiring multiple runs of the machine learning algorithms —- or suboptimal in calculating an empirical privacy. In this work, we present a tight and efficient auditing procedure and analysis that can effectively assess the privacy of mechanisms. Our approach is efficient; similar to the recent work of Steinke, Nasr, and Jagielski (2023), our auditing procedure leverages the randomness of examples in the input dataset and requires only a single (training) run of the target mechanism. And it is more accurate; we provide a novel analysis that enables us to achieve tight empirical privacy estimates by using the hypothesized f-DP curve of the mechanism, which provides a more accurate measure of privacy than the traditional ϵ, δ differential privacy parameters. We use our auditing procure and analysis to obtain empirical privacy, demonstrating that our auditing procedure delivers tighter privacy estimates.

# 1. Introduction

Differentially private machine learning [\(Chaudhuri et al.,](#page-9-0) [2011;](#page-9-0) [Abadi et al.,](#page-9-1) [2016\)](#page-9-1) has emerged as a principled solution to learning models from private data while still preserving privacy. Differential privacy [\(Dwork,](#page-9-2) [2006\)](#page-9-2) is a cryptographically motivated definition, which requires an algorithm to possess certain properties: specifically, a randomized mechanism is differentially private if it guarantees that the participation of any single person in the dataset does not impact the probability of any outcome by much.

Enforcing this guarantee requires the algorithm to be carefully designed and analyzed. The process of designing and analyzing such algorithms is prone to errors and imperfections as has been noted in the literature [\(Tramer et al.,](#page-11-0) [2022\)](#page-11-0). A result of this is that differentially private mechanisms may not perform as intended, either offering less privacy than expected due to flaws in mathematical analysis or implementation, or potentially providing stronger privacy guarantees that are not evident through a loose analysis.

Empirical privacy auditing [\(Ding et al.,](#page-9-3) [2018;](#page-9-3) [Nasr et al.,](#page-10-0) [2023;](#page-10-0) [Jagielski et al.,](#page-10-1) [2020\)](#page-10-1) has emerged as a critical tool to bridge this gap. By experimentally assessing the privacy of mechanisms, empirical auditing allows for the verification of privacy parameters. Specifically, an audit procedure is a randomized algorithm that takes an implementation of a mechanism M, runs it in a black-box manner, and attempts to test a privacy hypothesis (such as, a differential privacy parameter). The procedure outputs 0 if there is sufficient evidence that the mechanism does not satisfy the hypothesized guarantees and 1 otherwise. The audit mechanism must possess two essential properties: 1) it must have a *provably* small false-negative rate, ensuring that it would not erroneously reject a truly differentially private mechanism, with high probability; 2) it needs to *empirically* exhibit a "reasonable" false positive rate, meaning that when applied to a non-differentially private mechanism, it would frequently reject the privacy hypothesis. The theoretical proof of the false positive rate is essentially equivalent to privacy accounting [\(Abadi et al.,](#page-9-1) [2016;](#page-9-1) [Dong et al.,](#page-9-4) [2019;](#page-9-4) [Mironov,](#page-10-2) [2017\)](#page-10-2), which is generally thought to be impossible in a black-box manner [\(Zhu et al.,](#page-11-1) [2022\)](#page-11-1).

The prior literature on empirical audits of privacy consists of two lines of work, each with its own set of limitations. The first line of work [\(Ding et al.,](#page-9-3) [2018;](#page-9-3) [Jagielski et al.,](#page-10-1) [2020;](#page-10-1) [Tramer et al.,](#page-11-0) [2022;](#page-11-0) [Nasr et al.,](#page-10-0) [2023\)](#page-10-0) runs a differentially private algorithm multiple times to determine if the privacy guarantees are violated. This is highly computationally inefficient for most private machine learning use-cases, where running the algorithm involves training a large model.

Recent work [\(Steinke et al.,](#page-10-3) [2023\)](#page-10-3) remove this limitation by proposing an elegant auditing method that runs a differentially private training algorithm a single time. In particular, they rely on the randomness of training data to obtain bounds on the false negative rates of the audit procedure. A key limitation of the approach in [\(Steinke et al.,](#page-10-3) [2023\)](#page-10-3) is that their audit procedure is sub-optimal in the sense that there is a relatively large gap between the true privacy parameters of mainstream privacy-preserving algorithms (e.g., Gaussian mechanism) and those reported by their auditing algorithm.

<sup>1</sup>Meta. Correspondence to: Saeed Mahloujifar <saeedm@meta.com>.

*Proceedings of the* 42 nd *International Conference on Machine Learning*, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

computationally efficient and accurate. Our method requires only a single run of the privacy mechanism [<sup>1</sup>](#page-1-0) and leverages the f-DP curve [\(Dong et al.,](#page-9-4) [2019\)](#page-9-4), which allows for a more fine-grained accounting of privacy than the traditional reliance on ϵ, δ parameters. By doing so, we provide a tighter empirical assessment of privacy.

We experiment with our approach on both simple Gaussian mechanisms as well as a model trained on real data witth DP-SGD. Our experiments show that our auditing procedure can significantly outperform that of [\(Steinke et al.,](#page-10-3) [2023\)](#page-10-3) (see Figure [1\)](#page-7-0). This implies that better analysis may enable relatively tight auditing of differentially privacy guarantees in a computationally efficient manner in the context of large model training.

Technical overview: We briefly summarize the key technical components of our work and compare it with that of Steinke et al. (2023). Their auditing procedure employed a game similar to a membership inference process: the auditor selects a set of canaries and, for each canary, decides whether to inject it into the training set with independent probability 0.5. Once model training is completed, the auditor performs a membership inference attack to determine whether each canary was included. The number of correct guesses made by the adversary in this setting forms a random variable. The key technical contribution of Steinke et al. was to establish a tail bound on this random variable for mechanisms satisfying (ϵ)-DP. Specifically, they demonstrated that the tail of this random variable is bounded by that of a binomial distribution, binomial(n, p), where n is the number of canaries and p = e ϵ e <sup>ϵ</sup>+1 . To extend this analysis to approximate DP mechanisms, they further showed that the probability of the adversary's success exceeding this tail bound is at most O(n · δ).

Steinke et al. highlighted a limitation in their approach in auditing specific mechanisms, such as the Gaussian mechanism. They correctly argue that simplifying the mechanism's behavior to just two parameters, (ϵ, δ) , results in suboptimal auditing of specific mechanisms. In other words, the effectiveness of membership inference attacks against the Gaussian mechanism differs significantly from predictions based solely on the (ϵ, δ) parameters. To overcome this limitation, we propose auditing the entire privacy curve of a mechanism, rather than focusing solely on (ϵ, δ). Our solution involves three key technical steps:

- 1. We derive an upper bound on the adversary's success in correctly guessing a specific canary for mechanisms

satisfying f-DP. This bound is an improved version of the result by [\(Hayes et al.,](#page-9-5) [2023\)](#page-9-5) for bounding training data reconstruction in DP mechanisms. However, this is insufficient, as the adversary's guesses could be dependent, potentially leading to correlated successes (e.g., correctly or incorrectly guessing all samples).

- 2. To address the issue of dependency, we refine our analysis by defining p<sup>i</sup> as the probability of the adversary making exactly i correct guesses. We derive a recursive relation that bounds p<sup>i</sup> based on p1, . . . , pi−1. This recursive bound is the main technical novelty of our work. To derive this bound, we consider two conditions: the adversary correctly guesses the first canary or not. In the first case, we use our analysis from Step 1 to bound the probability of making i − 1 correct guesses given that the first guess was correct. For the incorrect guess case, we perform a combinatorial analysis to eliminate the condition. This analysis uses the fact that shuffling of the canaries does not change the probabilities of making i correct guesses. We note that it is crucial not to use the analysis of Step 1 for both cases. This is because the analysis of Step 1 cannot be tight for both cases at the same time. Finally, leveraging the convexity of trade-off functions and applying Jensen's inequality, we derive our final recursive relation. To the best of our knowledge, This combination of trade-off function with shuffling is a new technique and could have broader applications.
- 3. Finally, we design an algorithm that takes advantage of the recursive relation to numerically calculate an upper bound on the tail of the distribution. The algorithm is designed carefully so that we do not need to invoke the result of step 2 for very small events.

We also generalize our analysis to a broader notion of canary injection and membership inference. Specifically, we utilize a reconstruction game where the auditor can choose among k options for each canary point, introducing greater entropy for each choice. This generalization allows for auditing mechanisms with fewer canaries.

In the rest of the paper, we first introduce the notions of f-DP and explain what auditing based on f-DP entails. We then present our two auditing procedures based on membership inference and reconstruction attacks (Section 2). In Section 3, we provide a tight analysis of our audit's accuracy based on f-DP curves. Finally, in Section 4, we describe the experimental setup used to compare the bounds.

# 2. Auditing f- differential privacy

<sup>1</sup> In the context of privacy-preserving training of machine learning models, the privacy mechanism refers to the training algorithm. Therefore, when we mention a single run, we are specifically referring to a single execution of the training algorithm, not the inference algorithm.

used for a "privacy hypothesis," but they all share the common characteristic of being about an algorithm/mechanism M. For example, one possible hypothesis is that applying SGD with specific hyperparameters satisfies some notion of privacy. With this in mind, the privacy hypothesis are often mathematical constraints on the sensitivity of the algorithm's output to small changes in its input. The most well-known definition among these is differential privacy.

Definition 2.1. A mechanism M is (ϵ, δ)-DP if for all neighboring datasets S, S ′ with |S∆S ′ | = 1 and all measurable sets T, we have Pr[M(S) ∈ T] ≤ e <sup>ϵ</sup> Pr[M(S ′ ) ∈ T] + δ.

In essence, differential privacy ensures that the output distribution of the algorithm does not heavily depend on a single data point. Based on this definition, one can hypothesize that a particular algorithm satisfies differential privacy with certain ϵ and δ parameters. Consequently, auditing differential privacy involves designing a test for this hypothesis. We will later explore the desired properties of such an auditing procedure. However, at present, we recall a stronger notion of privacy known as f-differential privacy.

Notation For a function f : [0, 1] → [0, 1] we use ¯f to denote the function ¯f(x) = 1 − f(x).

Definition 2.2. A mechanism M is f-DP if for all neighboring datasets S, S ′ and all |S∆S ′ | = 1 measurable sets T we have

$$\Pr[M(\mathcal{S}) \in T] \leq \bar{f}\big(\Pr[M(\mathcal{S}')] \in T\big).$$

Note that this definition generalizes the notion of approximate differential privacy by allowing a more complex relation between the probability distributions of M(S) and M(S ′ ). The following proposition shows how one can express approximate DP as an instantiation of f-DP.

Proposition 2.3. *A mechanism is* (ϵ, δ)*-DP if it is* f*-DP with respect to* ¯f(x) = e ϵ · x + δ*.*

Although the function f could be an arbitrary function, without loss of generality, we only consider a specific class of functions in this notion.

*Remark* 2.4*.* Whenever we say that a mechanism satisfies f-DP, we implicitly imply that f is a valid trade-off function . That is, f is defined on domain [0, 1] and has a range of [0, 1]. Moreover, f is a decreasing and convex with f(x) ≤ 1 − x for all x ∈ [0, 1]. We emphasize that this is without loss of generality. That is, if a mechanism is f-DP for a an arbitrary function f : [0, 1] → [0, 1], then it is also f ′ -DP for valid trade-off function f ′ with f ′ (x) ≤ f(x) for all x ∈ [0, 1] (See Proposition 2.2 in [\(Dong et al.,](#page-9-4) [2019\)](#page-9-4)).

Definition 2.5 (Order of f-DP curves). For two trade-off functions f<sup>1</sup> and f2, we say f<sup>1</sup> is more private than f<sup>2</sup> and denote it by f<sup>1</sup> ≥ f<sup>2</sup> iff f1(x) ≥ f2(x) for all x ∈ [0, 1]. Also, for a family of trade-off functions F, we use maximal(F) to denote the set of maximal elements w.r.t to the privacy relation. Note that F could be a partial ordered set, and maximal(F) may have multiple elements.

Now that we have defined our privacy hypothesis, we can turn our attention to auditing these notions.

Definition 2.6 (Auditing f-DP). An audit procedure takes the description of a mechanism M, a trade-off function f, and outputs a bit that determines whether the mechanism satisfies f-DP or not. We define it as a two-step procedure.

- game: M → O, In this step, the auditor runs a potentially randomized experiment/game using the description of mechanism M ∈ M and obtains some observation o ∈ O.
- evaluate : O × F → {0, 1}, In this step, the auditor will output a bit b based on an observation o and a trade-off function f. This audit operation tries to infer whether the observation o is "likely" for a mechanism that satisfies f-DP.

The audit procedure is ψ-accurate if for all mechanism M that satisfy f-DP, we have

$$\Pr_{o \leftarrow \text{game}(\mathcal{M})} [\text{evaluate}(o, f) = 1] \geq \psi.$$

Note that we are defining the accuracy only for positive cases. This is the only guarantee we can get from attacks. For guarantees in negative cases, we need to perform privacy accounting for the mechanism [\(Wang et al.,](#page-11-2) [2023\)](#page-11-2).

Next, we formally define the notion of empirical privacy [\(Nasr et al.,](#page-10-4) [2021\)](#page-10-4) based on an auditing procedure. This notion provides the best privacy guarantee that is not violated by auditors' observation from a game setup.

Definition 2.7 (Empirical Privacy). Let (game, evaluate) be an audit procedure. We define the empirical privacy random variable for a mechanism M, w.r.t a family F of trade-off functions, to be the output of the following process. We first run the game to obtain observation o = game(M). We then construct

$$F_o = \max \text{imal}(\{f \in F; \text{evaluate}(o, f) = 1\})$$

where the maximal set is constructed according to Definition [2](#page-2-0).5. Then, the empirical privacy of the mechanism at a particular δ is defined as

$$\epsilon(\delta) = \min_{f \in F_o} \max_{x \in [0,1]} \log\left(\frac{1 - f(x) - \delta}{x}\right).$$

Note that the empirical privacy ϵ(δ) is a function of the observation o. Since, o itself is a random variable, then ϵ(δ) is also a random variable.

How to choose the family of trade-off functions? The family of trade-off functions should be chosen based on the expectations of the true privacy curve. For example, if one expects the privacy curve of a mechanism to be similar to that of a Gaussian mechanism, then they would choose the set of all trade-off functions imposed by a Gaussian mechanism as the family. For example, many believe that in the hidden state model of privacy [\(Ye & Shokri,](#page-11-3) [2022\)](#page-11-3), the final model would behave like a Gaussian mechanism with higher noise than what is expected from the accounting in the white-box model (where we assume we release all the intermediate models). Although we may not be able to prove this hypothesis , we can use our framework to calculate the empirical privacy, while assuming that the behavior of the final model would be similar to that of a Gaussian mechanism.

Auditing f-DP vs DP: f-DP can be viewed as a collection of DP parameters, where instead of considering (ϵ, δ) as fixed scalars, we treat ϵ as a function of δ. For any δ ∈ [0, 1], there exists an ϵ(δ) such that the mechanism satisfies (ϵ(δ), δ)-DP. The f-DP curve effectively represents the entire privacy curve rather than a single (ϵ, δ) pair. Thus, auditing f-DP can be expected to be more effective, as there are more constraints that need to be satisfied. A naive approach for auditing f-DP is to perform an audit for approximate DP at each (ϵ, δ) value along the privacy curve, rejecting if any of the audits fail. However, this leads to suboptimal auditing performance. First, the auditing analysis involves several inequalities that bound the probabilities of various events using differential privacy guarantees. The probability of these events could take any number between [0, 1]. Using a single (ϵ, δ) value to bound the probability of all these events cannot be tight because the linear approximation of privacy curve is tight in at most a single point. Hence, the guarantees of (ϵ, δ)-DP cannot be simultaneously tight for all events. However, with f-DP, we can obtain tight bounds on the probabilities of all events simultaneously. Second, For each (ϵ, δ) we have a small possibility of incorrectly rejecting the privacy hypothesis. So if we audit privacy for (ϵ(δ), δ) independently, we will reject any privacy hypothesis with probability 1.0. This challenge can be potentially resolved by using correlated randomness.

To demonstrate this key difference, we try a baseline for d auditing f-DP based on the work of [\(Steinke et al.,](#page-10-5) [2024b\)](#page-10-5) [2](#page-3-0) . In this baseline, we consider a gaussian mechanism with noise σ. Then, we audit the privacy curve at various values of δ. For this, we need to make sure that we run the attack once (the correlated randomness mentioned above), so we fix the number of guesses to be the optimal choice for δ = 10−<sup>5</sup> . Then we observe the attack's performance and apply

the method of [\(Steinke et al.,](#page-10-5) [2024b\)](#page-10-5). We observe that this improves the performance over the plain method but there it still has large gap with direct f-DP auditing. The details and results of this experiment are reported in Section [4.2.](#page-8-0)

#### 2.1. Guessing games

Here, we introduce the notion of guessing games which is a generalization of membership inference attacks [\(Nasr et al.,](#page-10-0) [2023\)](#page-10-0), and closely resembles the reconstruction setting introduced in [\(Hayes et al.,](#page-9-5) [2023\)](#page-9-5).

Definition 2.8. Consider a mechanism M : [k] <sup>m</sup> → Θ. In a guessing game we first sample an input dataset u ∈ [k] m from an arbitrary distribution. We run the mechanism to get θ ∼ M(u). Then a guessing adversary A : Θ → ([k]∪ {⊥}) <sup>m</sup> tries to guess the input to the mechanism from the output. We define

- the number of guesses by c ′ = P<sup>m</sup> <sup>i</sup>=1 I A(θ)<sup>i</sup> ̸= ⊥
- P and the number of correct guesses by c = m <sup>i</sup>=1 I A(θ)<sup>i</sup> = u<sup>i</sup> .

Then we output (c, c′ ) as the output of the game.

These guessing games are integral to our auditing strategies. We outline two specific ways to instantiate the guessing game. The first procedure is identical to that described in the work of [\(Steinke et al.,](#page-10-3) [2023\)](#page-10-3) and resembles membership inference attacks. The second auditing algorithm is based on the reconstruction approach introduced by [\(Hayes et al.,](#page-9-5) [2023\)](#page-9-5). In Section 3, we present all of our results in the context of the general notion of guessing games, ensuring that our findings extend to both the membership inference and reconstruction settings.

Auditing by membership inference: Algorithm 1 describes a game setup based on membership inference attacks. In this setup, we have a fixed training set T and a set of canaries C. We first sample a subset S of the canaries using poisson sampling. Then we run the mechanism M on T ∪ S to get a model θ ∼ M(T ∪ S). Then the adversary A inspects θ and tries to find examples that were present in S. Observe that this procedure is a guessing game with k = 2 and m = |C|. This is simply because the adversary is guessing between two choices for each canary, it is either included or not included. Note that this procedure is modular, we can use any T and C for the training set and canary set. We can also use any attack algorithm A.

We note that membership inference attacks have received a lot of attention recently [\(Homer et al.,](#page-9-6) [2008;](#page-9-6) [Shokri](#page-10-6) [et al.,](#page-10-6) [2017;](#page-10-6) [Leino & Fredrikson,](#page-10-7) [2020;](#page-10-7) [Bertran et al.,](#page-9-7) [2024;](#page-9-7) [Hu et al.,](#page-9-8) [2022;](#page-9-8) [Matthew et al.,](#page-10-8) [2023;](#page-10-8) [Duan et al.,](#page-9-9) [2024;](#page-9-9) [Zarifzadeh et al.,](#page-11-4) [2023\)](#page-11-4). These attack had a key difference

<sup>2</sup>This experiment was suggested by our anonymous reviewer. We than the reviewer for their suggestion.

from our attack setup and that is the fact that there is only a single example that the adversary is trying to make the inference for. Starting from the work of [\(Shokri et al.,](#page-10-6) [2017\)](#page-10-6), researchers have tried to improve attacks in various settings [\(Ye et al.,](#page-11-5) [2022;](#page-11-5) [Zarifzadeh et al.,](#page-11-4) [2023\)](#page-11-4). For example, using calibration techniques has been an effective way to improve membership inference attacks [\(Watson et al.,](#page-11-6) [2021;](#page-11-6) [Carlini](#page-9-10) [et al.,](#page-9-10) [2022\)](#page-9-10). Researchers have also changed their focus from average case performance of the attack to the tails of the distribution and measured the precision at low recall values [\(Ye et al.,](#page-11-5) [2022;](#page-11-5) [Nasr et al.,](#page-10-4) [2021\)](#page-10-4).

A substantial body of research has also explored the relationship between membership inference attacks and differential privacy [\(Sablayrolles et al.,](#page-10-9) [2019;](#page-10-9) [Mahloujifar et al.,](#page-10-10) [2022;](#page-10-10) [Balle et al.,](#page-9-11) [2022;](#page-9-11) [Bhowmick et al.,](#page-9-12) [2018;](#page-9-12) [Stock et al.,](#page-10-11) [2022;](#page-10-11) [Balle et al.,](#page-9-11) [2022;](#page-9-11) [Guo et al.,](#page-9-13) [2022;](#page-9-13) [Kaissis et al.,](#page-10-12) [2023;](#page-10-12) [2024\)](#page-10-13), using this connection to audit differential privacy [\(Steinke et al.,](#page-10-14) [2024a;](#page-10-14) [Pillutla et al.,](#page-10-15) [2024;](#page-10-15) [Jagielski et al.,](#page-10-1) [2020;](#page-10-1) [Ding et al.,](#page-9-3) [2018;](#page-9-3) [Bichsel et al.,](#page-9-14) [2018;](#page-9-14) [Nasr et al.,](#page-10-4) [2021;](#page-10-4) [2023;](#page-10-0) [Steinke et al.,](#page-10-5) [2024b;](#page-10-5) [Tramer et al.,](#page-11-0) [2022;](#page-11-0) [Bich](#page-9-15)[sel et al.,](#page-9-15) [2021;](#page-9-15) [Lu et al.,](#page-10-16) [2022;](#page-10-16) [Andrew et al.,](#page-9-16) [2023;](#page-9-16) [Cebere](#page-9-17) [et al.,](#page-9-17) [2024;](#page-9-17) [Annamalai & De Cristofaro,](#page-9-18) [2024;](#page-9-18) [Chadha et al.,](#page-9-19) [2024\)](#page-9-19). Some studies have investigated empirical methods to prevent membership inference attacks that do not rely on differential privacy [\(Hyland & Tople,](#page-10-17) [2019;](#page-10-17) [Jia et al.,](#page-10-18) [2019;](#page-10-18) [Chen & Pattabiraman,](#page-9-20) [2023;](#page-9-20) [Li et al.,](#page-10-19) [2024;](#page-10-19) [Tang](#page-10-20) [et al.,](#page-10-20) [2022;](#page-10-20) [Nasr et al.,](#page-10-21) [2018\)](#page-10-21). An intriguing avenue for future research is to use the concept of empirical privacy to compare the performance of these empirical methods with provable methods, such as DP-SGD.

Algorithm 1 Membership inference in one run game

input Oracle access to a mechanism M(·), A training dataset T , An indexed canary set C = {x<sup>i</sup> ;i ∈ [m]}, An attack algorithm A.

1: Set m = |C| 2: Sample u = (u1, . . . , um) ∼ Bernoulli(0.5)<sup>m</sup>, a binary vector where u<sup>i</sup> = 1 with probability 0.5. 3: Let S = {C[u<sup>i</sup> ]; u<sup>i</sup> = 1}i∈[m] , the subset of selected elements in C. 4: Run mechanism M on T ∪ S to get output θ. 5: Run membership inference attack A on θ to get set of membership predictions v = (v1, . . . , vm) which is supported on {0, 1, ⊥}<sup>m</sup>. 6: Count c, the number of correct guesses where u<sup>i</sup> = v<sup>i</sup> and c ′ the total number of guesses where v<sup>i</sup> ̸= ⊥. return (c, c′ ).

Auditing by reconstruction: We also propose an alternative way to perform auditing by reconstruction attacks. This setup starts with a training set St, similar to the membership inference setting. Then, we have a family of m

canary sets {S<sup>i</sup> c ;i ∈ [m]} where each S i c contains k distinct examples. Before training, we construct a set S<sup>s</sup> of size m by uniformly sampling an example from each S i c . Then, the adversary tries to find out which examples were sampled from each canary set S i <sup>c</sup> by inspecting the model. We recognize that this might be different from what one may consider a true "reconstruction attack", because the adversary is only performing a selection. However, if you consider the set size to be arbitrary large, and the distribution on the set to be arbitrary, then this will be general enough to cover various notions of reconstruction. We also note that [\(Hayes et al.,](#page-9-5) [2023\)](#page-9-5) use the same setup to measure the performance of the reconstruction attacks.

Algorithm 2 Reconstruction in one run game

input Oracle access to a mechanism M(·), A training dataset T , number of canaries m, number of options for each canary k, a matrix of canaries C = {x i j }i∈[m],j∈[k] , an attack algorthm A.

- 1: Let u = (u1, . . . , um) be a vector uniformly sampled from [k]
- m. 2: Let S = {x i u<sup>i</sup> }i∈[m] . 3: Run mechanism M on S ∪ T to get output θ. 4: Run a reconstruction attack A on θ to get a vector v = (v1, . . . , vm) which is a vector in ([k] ∪ {⊥})
- m. 5: Count c the number of coordinates where u<sup>i</sup> = v<sup>i</sup> and c ′ the number of coordinates where v<sup>i</sup> ̸= ⊥. return (c, c′ ).

# 3. Implications of f-DP for guessing games

In this section, we explore the implications of f-DP for guessing games. Specifically, we focus on bounding the probability of making more than c correct guesses for adversaries that make at most c ′ guesses. We begin by stating our main theorem, followed by an explanation of how it can be applied to audit the privacy of a mechanism.

Theorem 3.1. *[Bounds for adversary with bounded guesses] Let* M : [k] <sup>m</sup> → Θ *be a* f*-DP mechanism. Let* u *be a random variable uniformly distributed on* [k] <sup>m</sup>*. Let* A: Θ → ([k] ∪ {⊥}) <sup>m</sup> *be a guessing adversary which always makes at most* c ′ *guesses, that is*

$$\forall \theta \in \Theta, \Pr \left[ \left( \sum_{i=1}^m I(A(\theta)_i \neq \perp) \right) > c' \right] = 0,$$

*and let* <sup>v</sup> <sup>≡</sup> <sup>A</sup>(M(u))*. Define* <sup>p</sup><sup>i</sup> = Pr hP j∈[m] I(u<sup>j</sup> = v<sup>j</sup> ) = i i *. For all subset of indices* T ⊆ [c ′ ]*, we have*

$$\sum_{i \in T} \frac{i}{m} p_i \leq \bar{f} \left( \frac{1}{k-1} \sum_{i \in T} \frac{c' - i + 1}{m} p_{i-1} \right).$$

This Theorem, which we consider to be our main technical contribution, provides a nice invariant that bounds the probability p<sup>i</sup> (probability of making exactly i correct guesses) based on the value of other p<sup>j</sup> s. Imagine P<sup>f</sup> to be a set of vectors p = (p1, . . . , p<sup>c</sup> ′ ) that could be realized for an attack on a f-DP mechanism. Theorem [3.1](#page-4-0) significantly confines this set. However, this still does not resolve the auditing task. We are interested in bounding maxp∈P<sup>f</sup> P<sup>c</sup> ′ i=c pi , the maximum probability that an adversary can make more than c correct guesses for an f-DP mechanism. Next, we show how we can algorithmically leverage the limitations imposed by Theorem [3.1](#page-4-0) and calculate an upper bound on maxp∈P<sup>f</sup> P<sup>c</sup> i=c pi .

### 3.1. Numerically bounding the tail

In this subsection, we specify our procedure for bounding the tail of the distribution and hence the accuracy of our auditing procedure. Our algorithm needs oracle access to f and ¯f and decides an upper bound on the probability of an adversary making c correct guesses in a guessing game with alphabet size k and a mechanism that satisfies f-DP. This algorithm relies on the confinement imposed by Theorem [3](#page-4-0).1. Note that Algorithm [3](#page-5-0) is a decision algorithm, it takes a value τ and decide if the probability of making more than c correct guesses is less than or equal to τ . We can turn this algorithm to a estimation algorithm by performing a binary search on the value of τ . However, for our use cases, we are interested in a fixed τ . This is because we (similar to [\(Steinke et al.,](#page-10-3) [2023\)](#page-10-3)) want to set the accuracy of our audit to be a fixed value such as 0.95.

Algorithm 3 Numerically deciding an upper bound probability of making more than c correct guesses

input Oracle access to ¯f and ¯f −1 , number of guesses c ′ , number of correct guesses c, number of samples m, alphabet size k, probability threshold τ (default is τ = 0.05).

1: ∀0 ≤ i ≤ c set h[i] = 0, and r[i] = 0.

2: set r[c] = τ ·

c m .

3: set h[c] = τ ·

′−c <sup>m</sup> .

4: for i ∈ [c − 1, . . . , 0] do 5: h[i] = (k − 1) ¯f

−1 

r[i + 1]

6: r[i] = r[i + 1] + <sup>i</sup>

c ′−i · 

h[i] − h[i + 1]

.

7: end for

8: if r[0] + h[0] ≥

c <sup>m</sup> then

9: Return True; (Probability of c correct guesses (out of

c ′

) is less than τ ).

10: else

11: Return False; (Probability of having c correct guesses

(out of c ′

) could be more than τ ).

12: end if

Theorem 3.2. *If Algorithm [3](#page-5-0) returns True on inputs*

¯f, k, m, c, c′ *and* τ *, then for any* f*-DP mechanism* M : [k] <sup>m</sup> → Θ*, any guessing adversary* A: Θ → ([k] ∪ {⊥}) <sup>m</sup> *with at most* c ′ *guesses, defining* u *to be uniform over* [k] <sup>m</sup>*, and setting* v ≡ A M(u) *, we have* Pr[P<sup>m</sup> <sup>i</sup>=1 I(u<sup>i</sup> = vi) ≥ c] ≤ τ.

In a nutshell, this algorithm tries to obtain an upper bound on the sum p<sup>c</sup> +pc+1 +. . . , p<sup>c</sup> ′ . We assume this probability is greater than τ , and we obtain lower bound on pc−<sup>1</sup> + p<sup>c</sup> + · · · + p<sup>c</sup> ′ based on this assumption. We keep doing this recursively until we have a lower bound on p<sup>0</sup> + · · · + pc ′ . If this lower bound is greater than 1, then we have a contradiction and we return true. The detailed proof of this Theorem is involved and requires careful analysis. We defer the full proof of Theorem to appendix.

Auditing f-DP with Algorithm [3:](#page-5-0) When auditing the f-DP for a mechanism, we assume we have injected m canaries, and ran an adversary that is allowed to make c ′ guesses and recorded that the adversary have made c correct guesses. In such scenario, we will reject the hypothesized privacy of the mechanism if the probability of this observation is less than a threshold τ , which we by default set to 0.05. To this end, we just call Algorithm [3](#page-5-0) with parameters c, c ′ , m, τ = 0.05 and f. Then if the algorithm returns *True*, we will reject the privacy hypothesis and approve it otherwise.

Empirical privacy: Although auditing in essence is a hypothesis testing, previous work has used auditing algorithms to calculate empirical privacy as defined in definition [2.7.](#page-2-1) In this work, we follow the same route. For simplicity, we only consider an ordered set of privacy hypotheses h1, . . . , h<sup>w</sup> as our family of f-DP curves. These sets are ordered in their strength, meaning that any mechanism that satisfies h<sup>i</sup> , would also satisfy h<sup>j</sup> for all j < i. Then, we would report the strongest privacy hypothesis that passes the test as the empirical privacy of the mechanism.

# 4. Experiments

Most of our experiments are conducted in an *idealized setting*, similar to that used in [\(Steinke et al.,](#page-10-3) [2023\)](#page-10-3), unless otherwise stated. In this setting, the attack success rate is automatically calculated to simulate the expected number of correct guesses by an optimal adversary (details of the idealized setting are provided in Algorithm [4](#page-23-0) in Appendix). We then use this expected number as the default value for the number of correct guesses to derive the empirical ϵ. More specifically, as specified in Definition [2.6,](#page-2-2) we instantiate our auditing with a game and evaluation setup. We use Algorithm 4 in Appendix as our game setup. This algorithm returns the number of guesses and the number of correct guesses as the observations from the game. Then, we use Algorithm [3](#page-5-0) as our evaluation setup to audit an f-DP curve based on the observation from Algorithm 4. Note that in our

comparison with the auditing of Steinke et al., we always use the same membership inference game setup (k = 2) as defined in their work. This ensures that our comparison is only on the evaluation part of the audit procedure.

In all experiments, we use empirical ϵ as the primary metric for evaluating our bounds.

f-DP candidates: As described in Section [3.1](#page-5-1) , we need an ordered set of f-DP curves to obtain empirical privacy. In our experiments, we use f-DP curves for Gaussian mechanisms with varying standard deviations (this forms an ordered set because the f-DP curve of a Gaussian mechanism with a higher standard deviation dominates that of a lower standard deviation). For sub-sampled Gaussian mechanisms, the ordered set consists of f-DP curves for sub-sampled Gaussian mechanisms with fixed sub-sampling rates and number of steps, and various noise stds.

## 4.1. Comparison with [\(Steinke et al.,](#page-10-3) [2023\)](#page-10-3)

In this section, we evaluate our auditing method for membership inference in an idealized setting, using the work of [\(Steinke et al.,](#page-10-3) [2023\)](#page-10-3) as our main baseline. We compare our approach directly to their work, which operates in the same setting as ours.

Simple Gaussian Mechanism: In the first experiment (Figure [1\)](#page-7-0), we audit a simple Gaussian mechanism, varying the standard deviations from [0.5, 1.0, 2.0, 4.0], resulting in different theoretical ϵ values. We vary the number of canaries (m) from 10<sup>2</sup> to 10<sup>7</sup> for auditing, set the bucket size to k = 2, and adjust the number of guesses (c ′ ) for each number of canaries. For each combination of m, c ′ , and each standard deviation, we calculate (c) using Algorithm [4](#page-23-0) (the idealized setting in appendix). This algorithm calculates the expected number of correct guesses for an adversary who observes the output of an m-dimensional gaussian mechanism, V + N (0<sup>m</sup>, σ), with V being a uniform sample from {0, 1} <sup>m</sup>. The adversary's goal is to guess c ′ coordinates in V . c is calculated to be the expected number of correct guesses by the optimal adversary. Note that this setup is designed as the worst-case scenario for the gaussian mechanism. After obtaining c, we then audit all tuples of (m, c, c′ ) using the f-DP curves of the Gaussian mechanism. Then we find the c that achieves the highest empirical ϵ and then report that as the empirical ϵ. We audit the exact same setup with the auditing method of [\(Steinke et al.,](#page-10-5) [2024b\)](#page-10-5). Figure [1](#page-7-0) demonstrates that our approach outperforms the empirical privacy results from Steinke et al. Interestingly, while the bound in Steinke et al. (2023) degrades as the number of canaries increases, our bounds continue to improve.

4 [\(Zagoruyko & Komodakis,](#page-11-7) [2016\)](#page-11-7) architecture, which substitutes batch normalization with group normalization. We follow the setting proposed by [\(Sander et al.,](#page-10-22) [2023\)](#page-10-22), which use custom augmentation multiplicity (i.e., random crop around the center with 20 pixels padding with reflect, random horizontal flip and jitter) and apply an exponential moving average of the model weights with a decay parameter of 0.9999. We run white-box membership inference attacks by following the strongest attack used in the work of [\(Steinke et al.,](#page-10-3) [2023\)](#page-10-3), where the auditor injects multiple canaries in the training set with crafted gradients. More precisely, each canary gradient is set to zero except at a single random index ("Dirac canary" [\(Nasr et al.,](#page-10-0) [2023\)](#page-10-0)). Note that in the white-box attack, the auditor has access to all intermediate iterations of DP-SGD. The attack scores are computed as the dot product between the gradient update during consecutive model iterates and the aggregated gradients from dp-sgd. As done in the work of [\(Steinke et al.,](#page-10-3) [2023\)](#page-10-3), we audit CIFAR-10 model with m = 5, 000 canaries and all training points from CIFAR-10 n = 50, 000 for the attack. We set the batch size to 4, 096, using augumented multiplicity of K = 16 and training for 2, 500 DP-SGD steps. For ε = 8.0, δ = 10−<sup>5</sup> , we achieved 77% accuracy when auditing, compared to 80% without injected canaries. Figure [2](#page-7-1) shows the comparison between the auditing scheme by [\(Steinke et al.,](#page-10-3) [2023\)](#page-10-3) with ours for different values of theoretical ε. We are able to achieve tighter empirical lower bounds. We also report the performance of the black-box attack, where the auditor does not control the training pipeline and can only compute membership scores (losses) from the final model. Figure [3](#page-7-2) shows how we achieve tighter lower bounds compared to [Steinke et al.](#page-10-3) [\(2023\)](#page-10-3) where we set m = 1, 000 and all training samples are used for auditing (m = n). This corresponds to the stronger setup for the black-box auditor in [Steinke et al.](#page-10-3) [\(2023\)](#page-10-3).

Finally, we report the results of auditing the robust membership inference attack [\(Zarifzadeh et al.,](#page-11-4) [2023\)](#page-11-4) (RMIA), which to the best of our knowledge, represents the Stateof-The-Art (SoTA) black-box membership inference attack on CIFAR-10 from the literature. We reproduce the results in [\(Zarifzadeh et al.,](#page-11-4) [2023\)](#page-11-4) with a non-private WideResNet model (with depth 28 and width 2) for 100 training epochs on half of the dataset chosen at random resulting on a test accuracy of 92.2%. We run the low-cost black-box membership inference attack using 2 reference models in the offline setting [\(Zarifzadeh et al.,](#page-11-4) [2023\)](#page-11-4). We audit with m = 5, 000 canaries and report in Figure [4](#page-7-3) the comparison between our scheme and [\(Steinke et al.,](#page-10-3) [2023\)](#page-10-3) with different abstention values. Our auditing method clearly outperforms Steinke et al. for all bounded guesses settings, with higher empirical epsilon for larger abstention values (i.e., fewer guesses).

![](_page_7_Figure_1.jpeg)

Figure 1. Comparison between our empirical privacy lower bounds and that of [\(Steinke et al.,](#page-10-3) [2023\)](#page-10-3) at δ = 10−<sup>5</sup> .

![](_page_7_Figure_3.jpeg)

![](_page_7_Figure_4.jpeg)

Figure 2. Comparison with auditing procedure of [\(Steinke et al.,](#page-10-3) [2023\)](#page-10-3) on CIFAR-10 in white-box setting using gradient-based membership inference. Empirical ϵ is reported at δ = 10−<sup>5</sup> .

Figure 3. Comparison with [Steinke et al.](#page-10-3) [\(2023\)](#page-10-3) for CIFAR-10 in black-box setting. Empirical ϵ is reported at δ = 10−<sup>5</sup> .

Why is our bound better better than [\(Steinke et al.,](#page-10-3) [2023\)](#page-10-3)? The bounds in Steinke et al. audit approximate DP. That is, they take DP parameters (ϵ, δ) and prove an upper bound on the probability of any adversary obtaining c ′ correct guesses out of c total guesses, given m canaries

Figure 4. Comparison with auditing procedure of [\(Steinke et al.,](#page-10-3) [2023\)](#page-10-3) on non-private model trained on CIFAR-10 against blackbox RMIA method [\(Zarifzadeh et al.,](#page-11-4) [2023\)](#page-11-4). Empirical ϵ is reported at δ = 10−<sup>5</sup> .

![](_page_7_Figure_7.jpeg)

available. For the case of δ = 0, their bound is tight. For the case of δ > 0, however, they need to define a set of undesirable events and bound their collective probability. This incurs an additional O(m · δ) in the probability. The reason why their bounds start to degrade when we increase m is this very fact. The m · δ term starts to dominate and causes the empirical epsilon estimation to become worse. The reason we do not observe this behavior is that we do not use (ϵ, δ) to approximate the privacy curve, we use the exact curve as is. As we know, the linear approximation of privacy curve is optimal only in a single point for mechanisms that we are interested in (e.g. the Gaussian mechanism). Namely, there is only a single probability p ′ ∈ [0, 1] where we have

$$p = \Pr[M(D) \in E] \quad \text{and} \quad e^\epsilon \cdot p + \delta = \Pr[M(D') \in E].$$

Our bound is designed to avoid this issue. We derive a bound that uses the exact f-DP curve, which ensures that for all probabilities p ∈ [0, 1] the upper bound on the blow-up of events of size p is tight. Moreover, the way we invoke our Theorem [3.1](#page-4-0) in our numerical estimation [3](#page-5-0) is designed to

|   | Noise |   | #  | Canaries | Theoretical | Steinke et al. | Steinke et al. (pointwise) | Ours |
|---|-------|---|----|----------|-------------|----------------|----------------------------|------|
| σ | = 0   | 5 | 10 | 5        | 9.99        | 4.99           | 5.01                       | 8.16 |
| σ | = 1   | 0 | 10 | 5        | 4.37        | 2.61           | 2.71                       | 3.61 |
| σ | = 2   | 0 | 10 | 5        | 1.99        | 1.33           | 1.35                       | 1.59 |
| σ | = 4   | 0 | 10 | 6        | 0.92        | 0.61           | 0.67                       | 0.82 |

Table 1. Comparison of empirical privacy Gaussian noise levels. The reported numbers of are empirical ϵ at δ = 10−<sup>5</sup> .

apply the bound on events that can be simultaneously tight. This way, our bound does not have the problem of getting worse as the number of samples increases.

Note that this does not mean that there is no way to improve our bound. We still see some gap between the empirical epsilon and the true epsilon. The reason for this, we believe, is in the way numerical tail bound in Algorithm [3.2](#page-5-2) is designed. In this algorithm, we make some relaxations that can be a source of sub-optimality. Specifically, our analysis benefits from the fact that the expectation of correct guesses, conditioned on the correct guesses being greater than c divided by the expectation incorrect guesses conditioned on the same event is greater than c/c′ . This step is not tight as we cannot have a mechanism where the adversary makes exactly c correct guesses with probability greater than 0, while making more than c correct guesses with probability exactly 0. For a more interested reader, Equations [6](#page-18-0) and [7](#page-18-1) in the proof of Theorem [3.2](#page-5-2) is a source of sub-optimality that future work can resolve.

### 4.2. Improving [\(Steinke et al.,](#page-10-5) [2024b\)](#page-10-5) by testing multiple hypothesis.

In this section, we describe a method that uses the method of [\(Steinke et al.,](#page-10-5) [2024b\)](#page-10-5) to audit f-DP. We use the idea that if a mechanism satisfies f-DP, then for all δ ∈ [0, 1] it should pass the DP audit for (ϵδ, δ), where ϵ<sup>δ</sup> is the optimal ϵ obtained from f for δ. A key issue here is that auditing in one run will always suffer from probabilistic error. There is a small chance τ that the audit mechanism rejects the privacy hypothesis incorrectly. When doing the test multiple times, then we have to multiply the the failure probability by the number of trials.

However, we can avoid this by using shared randomness between trials. Specifically, if we only run the privacy game once and use the output of the game to audit privacy for different values of (ϵ, δ), we can potentially avoid this multiplication. Here, we design an experiment that shows even with this this approach, the bounds of previous work cannot match ours. We try to auditing Gaussian DP. First we instantiate a membership inference game with a fixed number of canaries (m) and a fixed number of guesses (c ′ ). This is optimized to achieve the best ϵ at δ = 10−<sup>5</sup> . We collect the number of correct guesses (c) in the membership inference

game. Using (m, c, c′ ) we can now auditing (ϵδ, δ)-DP for a large range of values of δ (δ = 10−<sup>x</sup> for 60 different values of x linearly spread between 3 to 9), where ϵ<sup>δ</sup> is the privacy of a gaussian mechanism with a given noise at δ. Then, we reject the privacy hypothesis for gaussian-DP if any of the individual tests are rejected. Using this auditing procedure, we obtain empirical epsilon values.

Table [1](#page-8-1) shows the results of our experiments. We can see that there is still a large gap between our auditing and the multiple run of the approach of previous work as described above. As discussed in Section [2,](#page-1-1) the reason for the multiple testing method being inferior to our direct f-DP auditing is that in the multiple DP-auditing approach, each auditing procedure is oblivious to other points on the f-DP curve and can only observe a single point on the curve. Whereas for our method, the audit procedure observes the entire curve. This point has also been discussed by the authors of [\(Steinke](#page-10-5) [et al.,](#page-10-5) [2024b\)](#page-10-5) as a limitation on of their approach.

## 5. Conclusions and limitations

We introduce a new approach for auditing the privacy of algorithms in a single run using f-DP curves. This method enables more accurate approximations of the true privacy guarantees, addressing the risk of a "false sense of privacy" that may arise from previous approximation techniques. By leveraging the entire f-DP curve, rather than relying solely on point estimates, our approach provides a more nuanced understanding of privacy trade-offs. This allows practitioners to make more informed decisions regarding privacyutility trade-offs in real-world applications. However, our approach does not provide a strict upper bound on privacy guarantees but instead offers an estimate of the privacy parameters that can be expected in practical scenarios. We also recognize that, despite the improvements over prior work, we still observe a gap between the empirical and theoretical privacy reported in the "one run" setting. Future work could focus on closing this gap to further enhance the reliability of empirical privacy estimations.

- Impact Statement This paper aims to advance the empirical measurement of algorithmic privacy. By improving our ability to evaluate the privacy risks associated with machine learning and data processing systems, this work contributes to the development of more trustworthy and accountable AI technologies. The main societal benefit is positive: practitioners and policymakers will be better equipped to assess and mitigate potential privacy harms, leading to safer deployment of data-driven systems. References Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., and Zhang, L. Deep learning with differential privacy. In *Proceedings of the 2016 ACM SIGSAC conference on computer and communications security*, pp. 308–318, 2016. Andrew, G., Kairouz, P., Oh, S., Oprea, A., McMahan,
- H. B., and Suriyakumar, V. One-shot empirical privacy estimation for federated learning. *arXiv preprint arXiv:2302.03098*, 2023. Annamalai, M. S. M. S. and De Cristofaro, E. Nearly tight black-box auditing of differentially private machine learning. *arXiv preprint arXiv:2405.14106*, 2024. Balle, B., Cherubin, G., and Hayes, J. Reconstructing training data with informed adversaries. In *2022 IEEE Symposium on Security and Privacy (SP)*, pp. 1138–1156. IEEE, 2022. Bertran, M., Tang, S., Roth, A., Kearns, M., Morgenstern,
- J. H., and Wu, S. Z. Scalable membership inference attacks via quantile regression. *Advances in Neural Information Processing Systems*, 36, 2024. Bhowmick, A., Duchi, J., Freudiger, J., Kapoor, G., and Rogers, R. Protection against reconstruction and its applications in private federated learning. *arXiv preprint arXiv:1812.00984*, 2018. Bichsel, B., Gehr, T., Drachsler-Cohen, D., Tsankov, P., and Vechev, M. Dp-finder: Finding differential privacy violations by sampling and optimization. In *Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security*, pp. 508–524, 2018. Bichsel, B., Steffen, S., Bogunovic, I., and Vechev, M. Dpsniper: Black-box discovery of differential privacy violations using classifiers. In *2021 IEEE Symposium on Security and Privacy (SP)*, pp. 391–409. IEEE, 2021. Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A., and Tramer, F. Membership inference attacks from first principles. In *2022 IEEE Symposium on Security and Privacy (SP)*, pp. 1897–1914. IEEE, 2022. Cebere, T., Bellet, A., and Papernot, N. Tighter privacy auditing of dp-sgd in the hidden state threat model. *arXiv preprint arXiv:2405.14457*, 2024. Chadha, K., Jagielski, M., Papernot, N., Choquette-Choo, C., and Nasr, M. Auditing private prediction. *arXiv preprint arXiv:2402.09403*, 2024. Chaudhuri, K., Monteleoni, C., and Sarwate, A. D. Differentially private empirical risk minimization. *Journal of Machine Learning Research*, 12(3), 2011. Chen, Z. and Pattabiraman, K. Overconfidence is a dangerous thing: Mitigating membership inference attacks by enforcing less confident prediction. *arXiv preprint arXiv:2307.01610*, 2023. Ding, Z., Wang, Y., Wang, G., Zhang, D., and Kifer, D. Detecting violations of differential privacy. In *Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security*, pp. 475–489, 2018. Dong, J., Roth, A., and Su, W. J. Gaussian differential privacy. *arXiv preprint arXiv:1905.02383*, 2019. Duan, M., Suri, A., Mireshghallah, N., Min, S., Shi, W., Zettlemoyer, L., Tsvetkov, Y., Choi, Y., Evans, D., and Hajishirzi, H. Do membership inference attacks work on large language models? *arXiv preprint arXiv:2402.07841*, 2024. Dwork, C. Differential privacy. In *International colloquium on automata, languages, and programming*, pp. 1–12. Springer, 2006. Guo, C., Karrer, B., Chaudhuri, K., and van der Maaten, L. Bounding training data reconstruction in private (deep) learning. In *International Conference on Machine Learning*, pp. 8056–8071. PMLR, 2022. Hayes, J., Mahloujifar, S., and Balle, B. Bounding training data reconstruction in dp-sgd. *arXiv preprint arXiv:2302.07225*, 2023. Homer, N., Szelinger, S., Redman, M., Duggan, D., Tembe, W., Muehling, J., Pearson, J. V., Stephan, D. A., Nelson,
  - S. F., and Craig, D. W. Resolving individuals contributing trace amounts of dna to highly complex mixtures using high-density snp genotyping microarrays. *PLoS genetics*, 4(8):e1000167, 2008. Hu, H., Salcic, Z., Sun, L., Dobbie, G., Yu, P. S., and Zhang,
    - X. Membership inference attacks on machine learning: A survey. *ACM Computing Surveys (CSUR)*, 54(11s):1–37, 2022.

- Hyland, S. L. and Tople, S. On the intrinsic privacy of stochastic gradient descent. *Preprint at https://arxiv. org/pdf/1912.02919. pdf*, 2019. Jagielski, M., Ullman, J., and Oprea, A. Auditing differentially private machine learning: How private is private sgd? *Advances in Neural Information Processing Systems*, 33:22205–22216, 2020. Jia, J., Salem, A., Backes, M., Zhang, Y., and Gong, N. Z. Memguard: Defending against black-box membership inference attacks via adversarial examples. In *Proceedings of the 2019 ACM SIGSAC conference on computer and communications security*, pp. 259–274, 2019. Kaissis, G., Hayes, J., Ziller, A., and Rueckert, D. Bounding data reconstruction attacks with the hypothesis testing interpretation of differential privacy. *arXiv preprint arXiv:2307.03928*, 2023. Kaissis, G., Ziller, A., Kolek, S., Riess, A., and Rueckert,
- D. Optimal privacy guarantees for a relaxed threat model: Addressing sub-optimal adversaries in differentially private machine learning. *Advances in Neural Information Processing Systems*, 36, 2024. Leino, K. and Fredrikson, M. Stolen memories: Leveraging model memorization for calibrated {White-Box} membership inference. In *29th USENIX security symposium (USENIX Security 20)*, pp. 1605–1622, 2020. Li, J., Li, N., and Ribeiro, B. {MIST}: Defending against membership inference attacks through {Membership-Invariant} subspace training. In *33rd USENIX Security Symposium (USENIX Security 24)*, pp. 2387–2404, 2024. Lu, F., Munoz, J., Fuchs, M., LeBlond, T., Zaresky-Williams, E., Raff, E., Ferraro, F., and Testa, B. A general framework for auditing differentially private machine learning. *Advances in Neural Information Processing Systems*, 35:4165–4176, 2022. Mahloujifar, S., Sablayrolles, A., Cormode, G., and Jha,
- S. Optimal membership inference bounds for adaptive composition of sampled gaussian mechanisms. *arXiv preprint arXiv:2204.06106*, 2022. Matthew, J., Milad, N., Christopher, C.-C., Katherine, L., and Nicholas, C. Students parrot their teachers: Membership inference on model distillation. *arXiv preprint arXiv: 2303.03446*, 2023. Mironov, I. Renyi differential privacy. In ´ *2017 IEEE 30th computer security foundations symposium (CSF)*, pp. 263–
- 275. IEEE, 2017. Nasr, M., Shokri, R., and Houmansadr, A. Machine learning with membership privacy using adversarial regularization. In *Proceedings of the 2018 ACM SIGSAC conference on computer and communications security*, pp. 634–646, 2018. Nasr, M., Songi, S., Thakurta, A., Papernot, N., and Carlin,
  - N. Adversary instantiation: Lower bounds for differentially private machine learning. In *2021 IEEE Symposium on security and privacy (SP)*, pp. 866–882. IEEE, 2021. Nasr, M., Hayes, J., Steinke, T., Balle, B., Tramer, F., Jagiel- ` ski, M., Carlini, N., and Terzis, A. Tight auditing of differentially private machine learning. *arXiv preprint arXiv:2302.07956*, 2023. Pillutla, K., Andrew, G., Kairouz, P., McMahan, H. B., Oprea, A., and Oh, S. Unleashing the power of randomization in auditing differentially private ml. *Advances in Neural Information Processing Systems*, 2024. Sablayrolles, A., Douze, M., Schmid, C., Ollivier, Y., and Jegou, H. White-box vs black-box: Bayes optimal strate- ´ gies for membership inference. In *International Conference on Machine Learning*, pp. 5558–5567. PMLR, 2019. Sander, T., Stock, P., and Sablayrolles, A. Tan without a burn: Scaling laws of dp-sgd. In *International Conference on Machine Learning*. PMLR, 2023. Shokri, R., Stronati, M., Song, C., and Shmatikov, V. Membership inference attacks against machine learning models. In *2017 IEEE symposium on security and privacy (SP)*, pp. 3–18. IEEE, 2017. Steinke, T., Nasr, M., and Jagielski, M. Privacy auditing with one (1) training run. *arXiv preprint arXiv:2305.08846*, 2023. Steinke, T., Nasr, M., Ganesh, A., Balle, B., Choquette-Choo, C. A., Jagielski, M., Hayes, J., Thakurta, A. G., Smith, A., and Terzis, A. The last iterate advantage: Empirical auditing and principled heuristic analysis of differentially private sgd. *arXiv preprint arXiv:2410.06186*, 2024a. Steinke, T., Nasr, M., and Jagielski, M. Privacy auditing with one (1) training run. *Advances in Neural Information Processing Systems*, 36, 2024b. Stock, P., Shilov, I., Mironov, I., and Sablayrolles, A. Defending against reconstruction attacks with r\'enyi differential privacy. *arXiv preprint arXiv:2202.07623*, 2022. Tang, X., Mahloujifar, S., Song, L., Shejwalkar, V., Nasr, M., Houmansadr, A., and Mittal, P. Mitigating membership inference attacks by {Self-Distillation} through a novel ensemble architecture. In *31st USENIX Security Symposium (USENIX Security 22)*, pp. 1433–1450, 2022.

Tramer, F., Terzis, A., Steinke, T., Song, S., Jagielski, M., and Carlini, N. Debugging differential privacy: A case study for privacy auditing. *arXiv preprint arXiv:2202.12219*, 2022. Wang, J. T., Mahloujifar, S., Wu, T., Jia, R., and Mittal, P. A randomized approach for tight privacy accounting. *arXiv preprint arXiv:2304.07927*, 2023. Watson, L., Guo, C., Cormode, G., and Sablayrolles, A. On the importance of difficulty calibration in membership inference attacks. *arXiv preprint arXiv:2111.08440*, 2021. Ye, J. and Shokri, R. Differentially private learning needs hidden state (or much faster convergence). *Advances in Neural Information Processing Systems*, 35:703–715, 2022. Ye, J., Maddi, A., Murakonda, S. K., Bindschaedler, V., and Shokri, R. Enhanced membership inference attacks against machine learning models. In *Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security*, pp. 3093–3106, 2022. Zagoruyko, S. and Komodakis, N. Wide residual networks. *arXiv preprint arXiv:1605.07146*, 2016. Zarifzadeh, S., Liu, P. C.-J. M., and Shokri, R. Low-cost high-power membership inference by boosting relativity. 2023. Zhu, Y., Dong, J., and Wang, Y.-X. Optimal accounting of differential privacy via characteristic function. In *International Conference on Artificial Intelligence and Statistics*, pp. 4782–4817. PMLR, 2022.

## A. Proofs

## A.1. Proof outline for Theorem [3.1](#page-4-0)

In this subsection, we outline the main ingredients we need to prove our Theorem [3.1.](#page-4-0) We also provide a warm up proof for a simplified version of Theorem [3.1](#page-4-0) without abstentions and then we focus on the proof of the main theorem. First, we have a Lemma that bounds the probability of any event conditioned on correctly guessing a single canary.

Lemma A.1. *Let* M : [k] <sup>m</sup> → Θ *be a mechanism that satisfies* f*-DP. Also let* A: Θ → ([k] ∪ {⊥}) <sup>m</sup> *be a guessing attack. Let* u *be a random variable uniformly distributed over* [k] <sup>m</sup> *and let* v ≡ A M(u) *. Then for any subset* E ⊆ Θ *we have*

$$f''_{\mathbf{k}} \left( \Pr [M(\mathbf{u}) \in E] \right) \leq \Pr [M(\mathbf{u}) \in E \text{ and } u_1 = v_1] \leq f'_{\mathbf{k}} \left( \Pr [M(\mathbf{u}) \in E] \right)$$

*where*

$$f'_k(x) = \sup\{\alpha; \alpha + f(\frac{x-\alpha}{k-1}) \leq 1\} \quad \text{and} \quad f''_k(x) = \inf\{\alpha; (k-1)f(\alpha) + x - \alpha \leq 1\}.$$

This Lemma which is a generalization and an improvement over the main Theorem of [\(Hayes et al.,](#page-9-5) [2023\)](#page-9-5), shows that the probability of an event cannot change too much if we condition on the success of adversary on one of the canaries. Note that this Lemma immediately implies a bound on the expected number of correct guesses by any guessing adversary (by just using linearity of expectation). However, here we are not interested in expectations. Rather, we need to derive tail bounds. The proof of Theorem [3.1](#page-4-0) relies on some key properties of the f ′ and f ′′ functions defined in the statement of Lemma [A.1.](#page-12-0) These properties are specified in the following Proposition and proved in the Appendix.

Proposition A.2. *The functions* f ′ k *as defined in Lemma [A.1](#page-12-0) is increasing and concave. The function* f ′′ k *as defined in Lemma [A.1](#page-12-0) is increasing and convex.*

Now, we are ready to outline the proof of a simplified variant of our Theorem [3.1](#page-4-0) for adversaries that make a guess on all canaries. This makes the proof much simpler and enables us to focus more on the key steps in the proof.

Theorem A.3 (Special case of [3.1\)](#page-4-0). *Let* M : [k] <sup>m</sup> → Θ *be a* f*-DP mechanism. Let* u *be a random variable uniformly distributed on* [k] <sup>m</sup>*. Let* A: Θ → [k] <sup>m</sup> *be a guessing adversary and let* <sup>v</sup> <sup>≡</sup> <sup>A</sup>(M(u))*. Define* <sup>p</sup><sup>i</sup> = Pr h ( P j∈[m] I u<sup>j</sup> = v<sup>j</sup> ) = i i *. For all subset of indices* T ⊆ [m]*, we have*

$$\sum_{i \in T} \frac{i}{m} p_i \leq \bar{f} \left( \frac{1}{k-1} \sum_{i \in T} \frac{m-i+1}{m} p_{i-1} \right)$$

*Proof.* Let us define a random variable t = (t1, . . . , tm) which is defined as t<sup>i</sup> = I(u<sup>i</sup> = vi) We have

$$p_c = \Pr[\sum_{i=1}^m \mathbf{t}_i = c] = \Pr[\sum_{i=2}^m \mathbf{t}_i = c - 1 \text{ and } \mathbf{t}_1 = 1] + \Pr[\sum_{i=2}^m \mathbf{t}_i = c \text{ and } \mathbf{t}_1 = 0]$$

Now by Lemma [A.1](#page-12-0) we have Pr[P<sup>m</sup> <sup>i</sup>=2 t<sup>i</sup> = c − 1 and t<sup>1</sup> = 1] ≤ f ′ k ( P<sup>m</sup> <sup>i</sup>=2 t<sup>i</sup> = c − 1). This is a nice invariant that we can use but P<sup>m</sup> <sup>i</sup>=2 t<sup>i</sup> = c − 1 could be really small depending on how large m is. To strengthen the bound we sum all pc's for c ∈ T, and then apply the lemma on the aggregate. That is

$$\begin{aligned} \sum_{j \in T} p_j &= \sum_{j \in T} \Pr[\sum_{i=1}^m \mathbf{t}_i = j] = \sum_{j \in T} \Pr[\sum_{i=2}^m \mathbf{t}_i = j \text{ and } \mathbf{t}_1 = 0] + \sum_{j \in T} \Pr[\sum_{i=2}^m \mathbf{t}_i = j - 1 \text{ and } \mathbf{t}_1 = 1] \\ &= \Pr[\sum_{i=2}^m \mathbf{t}_i \in T \text{ and } \mathbf{t}_1 = 0] + \Pr[1 + \sum_{i=2}^m \mathbf{t}_i \in T \text{ and } \mathbf{t}_1 = 1] \end{aligned}$$

Now we only use the inequality from Lemma [A.1](#page-12-0) for the second quantity above. Using the inequality for both probabilities is not ideal because they cannot be tight at the same time. So we have,

$$\sum_{j \in T} p_j \leq \Pr\left[\sum_{i=2}^m \in T \text{ and } \mathbf{t}_1 = 0\right] + f'_k(\Pr[1 + \sum_{i=2}^m \mathbf{t}_i \in T]).$$

Now we use a trick to make this cleaner. We use the fact that this inequality is invariant to the order of indices. So we can permute ti's and the inequality still holds. We have,

$$\begin{aligned} \sum_{j \in T} p_j &\leq \frac{E}{\pi \sim \Pi[m]} \left[ \Pr \left[ \sum_{i=2}^m \mathbf{t}_{\pi(i)} \in T \text{ and } \mathbf{t}_{\pi(1)} = 0 \right] \right] + \frac{E}{\pi \sim \Pi[m]} [f'_k(\Pr[1 + \sum_{i=2}^m \mathbf{t}_{\pi(i)} \in T])] \\ &\leq \frac{E}{\pi \sim \Pi[m]} \left[ \Pr \left[ \sum_{i=2}^m \mathbf{t}_{\pi(i)} \in T \text{ and } \mathbf{t}_{\pi(1)} = 0 \right] \right] + f'_k \left( \frac{E}{\pi \sim \Pi[m]} [\Pr[1 + \sum_{i=2}^m \mathbf{t}_{\pi(i)} \in T]] \right). \end{aligned}$$

Now we perform a double counting argument. Note that when we permute the order P<sup>m</sup> <sup>i</sup>=2 tπ(i) = j and tπ(1) = 0 counts each instance t1, . . . , t<sup>m</sup> with exactly j non-zero locations, for exactly (m − j) × (m − 1)! times. Therefore, we have

$$\mathbb{E}_{\pi \sim \Pi[m]} [\Pr[\sum_{i=2}^m \mathbf{t}_{\pi(i)} \in T \text{ and } \mathbf{t}_{\pi(1)} = 0]] = \sum_{j \in T} \frac{m-j}{m} p_j.$$

With a similar argument we have,

$$\mathbf{E}_{\pi \sim \Pi[m]} [\Pr[1 + \sum_{i=2}^m \mathbf{t}_{\pi(i)} \in T]] = \sum_{j \in T} \frac{m - j + 1}{m} p_{j-1} + \frac{j}{m} p_j.$$

Then, we have

$$\sum_{j \in T} p_j \leq \sum_{j \in T} \frac{m-j}{m} p_j + f'_k\left(\sum_{j \in T} \frac{j}{m} p_j + \frac{m-j+1}{m} p_{j-1}\right).$$

And this implies

$$\sum_{j \in T} \frac{j}{m} p_j \leq f'_k \left( \sum_{j \in T} \frac{j}{m} p_j + \frac{m-j+1}{m} p_{j-1} \right).$$

And this, by definition of f ′ k implies

$$\sum_{j \in T} \frac{j}{m} p_j \leq \bar{f}\left(\frac{1}{k-1} \sum_{j \in T} \frac{m-j+1}{m} p_{j-1}\right).$$

### A.2. Proof of Main Lemmas and Theorems

$$\begin{aligned}
p &= \sum_{i \in [k]} \Pr[M(\mathbf{u}) \in E \text{ and } u_1 = v_1 = i] \\
&= \frac{1}{k} \sum_{i \in [k]} \Pr[M(\mathbf{u}) \in E \text{ and } v_1 = i \mid u_1 = i] \\
&= \frac{1}{k} \sum_{i \in [k]} \frac{1}{k-1} \left( \sum_{j \in [k] \setminus \{i\}} \Pr[M(\mathbf{u}) \in E \text{ and } v_1 = i \mid u_1 = i] \right) \\
(\text{By definition of } f\text{-DP}) &\leq \frac{1}{k} \sum_{i \in [k]} \frac{1}{k-1} \left( \sum_{j \in [k] \setminus \{i\}} 1 - f(\Pr[M(\mathbf{u}) \in E \text{ and } v_1 = i \mid u_1 = j]) \right) \\
(\text{By convexity of } f) &\leq 1 - f \left( \frac{1}{k} \sum_{i \in [k]} \frac{1}{k-1} \left( \sum_{j \in [k] \setminus \{i\}} \Pr[M(\mathbf{u}) \in E \text{ and } v_1 = i \mid u_1 = j] \right) \right) \\
&= 1 - f \left( \frac{1}{k-1} \sum_{i \in [k]} \left( \sum_{j \in [k] \setminus \{i\}} \frac{1}{k} \Pr[M(\mathbf{u}) \in E \text{ and } v_1 = i \mid u_1 = j] \right) \right) \\
&= 1 - f \left( \frac{1}{k-1} \sum_{i \in [k]} \left( \sum_{j \in [k] \setminus \{i\}} \Pr[M(\mathbf{u}) \in E \text{ and } v_1 = i \text{ and } u_1 = j] \right) \right) \\
&= 1 - f\left(\frac{1}{k-1} \Pr[M(\mathbf{u}) \in E \text{ and } u_1 \neq v_1]\right) \\
&= 1 - f\left(\frac{q-p}{k-1}\right).
\end{aligned}$$

Similarly we have,

$$\begin{aligned}
 p &= \sum_{i \in [k]} \Pr[M(\mathbf{u}) \in E \text{ and } u_1 = v_1 = i] \\
 &= \frac{1}{k} \sum_{i \in [k]} \Pr[M(\mathbf{u}) \in E \text{ and } v_1 = i \mid u_1 = i] \\
 &= \frac{1}{k} \sum_{i \in [k]} \frac{1}{k-1} \left( \sum_{j \in [k] \setminus \{i\}} \Pr[M(\mathbf{u}) \in E \text{ and } v_1 = i \mid u_1 = i] \right) \\
 (\text{By definition of } f\text{-DP}) &\geq \frac{1}{k} \sum_{i \in [k]} \frac{1}{k-1} \left( \sum_{j \in [k] \setminus \{i\}} f^{-1}(1 - \Pr[M(\mathbf{u}) \in E \text{ and } v_1 = i \mid u_1 = j]) \right) \\
 (\text{By convexity of } f) &\geq f^{-1} \left( \frac{1}{k} \sum_{i \in [k]} \frac{1}{k-1} \left( \sum_{j \in [k] \setminus \{i\}} 1 - \Pr[M(\mathbf{u}) \in E \text{ and } v_1 = i \mid u_1 = j] \right) \right) \\
 &= f^{-1} \left( \frac{1}{k-1} \sum_{i \in [k]} \left( \sum_{j \in [k] \setminus \{i\}} \frac{1}{k} (1 - \Pr[M(\mathbf{u}) \in E \text{ and } v_1 = i \mid u_1 = j]) \right) \right) \\
 &= f^{-1} \left( \frac{1}{k-1} \sum_{i \in [k]} \left( \sum_{j \in [k] \setminus \{i\}} \Pr[M(\mathbf{u}) \in E \text{ and } v_1 = i \text{ and } u_1 = j] \right) \right) \\
 &= f^{-1} \left( \frac{1}{k-1} (1 - \Pr[M(\mathbf{u}) \in E \text{ and } u_1 \neq v_1]) \right) \\
 &= f^{-1} \left( \frac{1 - q + p}{k-1} \right).
 \end{aligned}$$

This implies that,

$$f(p) \cdot (k-1) + q - p \leq 1$$

*Proof of Proposition [A.2.](#page-12-1)* The function is increasing simply because f is decreasing. We now prove concavity. Let α<sup>1</sup> = fk(x1) and α<sup>2</sup> = fk(x2). By definition of f<sup>k</sup> we have

$$\alpha_1 + f\left(\frac{x_1 - \alpha_1}{k - 1}\right) \leq 1$$

and

$$\alpha_2 + f\left(\frac{x_2 - \alpha_2}{k-1}\right) \leq 1.$$

Averaging these two we get,

$$\frac{\alpha_1 + \alpha_2}{2} + \frac{f(\frac{x_1-\alpha_1}{k-1}) + f(\frac{x_2-\alpha_2}{k-1})}{2} \leq 1$$

By convexity of f we have

$$\frac{\alpha_1 + \alpha_2}{2} + f\left(\frac{\frac{x_1+x_2}{2} - \frac{\alpha_1+\alpha_2}{2}}{k-1}\right) \leq 1$$

Therefore, by definition of f ′ k , we have f ′ k ( x1+x<sup>2</sup> 2 ) ≥ α1+α<sup>2</sup> . Similarly, f ′′ k in increasing just because f is decreasing. And assuming α<sup>1</sup> = fk(x1) and α<sup>2</sup> = fk(x2) we have

$$f_k''\left(\frac{x_1 + x_2}{2}\right) \leq \frac{\alpha_1 + \alpha_2}{2}$$

which implies f ′′ k is convex.

*Proof of Theorem [3.1.](#page-4-0)* Instead of working with an adversary with c ′ guesses, we assume we have an adversary that makes a guess on all m inputs, however, it also submits a vector q ∈ {0, 1} <sup>m</sup>, with exactly c ′ 1s and m − c ′ 0s. So the output of this adversary is a vector v ∈ [k] <sup>m</sup> and a vector q ∈ {0, 1} <sup>m</sup>. Then, only correct guesses that are in locations that q is non-zero is counted. That is, if we define a random variable t = (t1, . . . , tm) as t<sup>i</sup> = I(u<sup>i</sup> = vi) then we have

$$\begin{aligned} p_c &= \Pr\left[\sum_{i=1}^m \mathbf{t}_i \cdot \mathbf{q}_i = c\right] \\ &= \Pr\left[\sum_{i=2}^m \mathbf{t}_i = c - 1 \text{ and } \mathbf{t}_1 = 1 \text{ and } \mathbf{q}_1 = 1\right] + \Pr\left[\sum_{i=2}^m \mathbf{t}_i = c \text{ and } \mathbf{t}_1 \cdot \mathbf{q}_1 = 0\right] \end{aligned}$$

Now by Lemma [A.1](#page-12-0) we have

$$\Pr\left[\sum_{i=2}^m \mathbf{t}_i = c - 1 \text{ and } \mathbf{t}_1 = 1 \text{ and } \mathbf{q}_1 = 1\right] \leq f'_k\left(\sum_{i=2}^m \mathbf{t}_i = c - 1 \text{ and } \mathbf{q}_1 = 1\right).$$

This is a nice invariant that we can use but P<sup>m</sup> <sup>i</sup>=2 t<sup>i</sup> = c − 1 could be really small depending on how large m is. To strengthen the bound we sum all pc's for c ∈ T, and then apply the lemma on the aggregate. That is

$$\begin{aligned} \sum_{j \in T} p_j &= \sum_{j \in T} \Pr\left[\sum_{i=1}^m \mathbf{t}_i = j\right] \\ &= \sum_{j \in T} \Pr\left[\sum_{i=2}^m \mathbf{t}_i = j \text{ and } \mathbf{t}_1 \cdot \mathbf{q}_1 = 0\right] + \sum_{j \in T} \Pr\left[\sum_{i=2}^m \mathbf{t}_i = j - 1 \text{ and } \mathbf{t}_1 = 1 \text{ and } \mathbf{q}_1 = 1\right] \\ &= \Pr\left[\sum_{i=2}^m \mathbf{t}_i \in T \text{ and } \mathbf{t}_1 \cdot \mathbf{q}_1 = 0\right] + \Pr\left[1 + \sum_{i=2}^m \mathbf{t}_i \in T \text{ and } \mathbf{t}_1 = 1 \text{ and } \mathbf{q}_1 = 1\right] \end{aligned}$$

Now we only use the inequality from Lemma [A.1](#page-12-0) for the second quantity above. Using the inequality for both probabilities is not ideal because they cannot be tight at the same time. So we have,

$$\sum_{j \in T} p_j \leq \Pr\left[\sum_{i=2}^m \in T \text{ and } \mathbf{t}_1 \cdot \mathbf{q}_1 = 0\right] + f'_k(\Pr[1 + \sum_{i=2}^m \mathbf{t}_i \in T \text{ and } \mathbf{q}_1 = 1]).$$

Now we use a trick to make this cleaner. We use the fact that this inequality is invariant to the order of indices. So we can permute ti's and the inequality still holds. We have,

$$\begin{aligned} \sum_{j \in T} p_j &\leq \mathbb{E}_{\pi \sim \Pi[m]} \left[ \Pr \left[ \sum_{i=2}^m \mathbf{t}_{\pi(i)} \in T \text{ and } \mathbf{t}_{\pi(1)} \cdot \mathbf{q}_{\pi(1)} = 0 \right] \right] + \mathbb{E}_{\pi \sim \Pi[m]} \left[ f'_k(\Pr \left[ 1 + \sum_{i=2}^m \mathbf{t}_{\pi(i)} \in T \right]) \right] \\ &\leq \mathbb{E}_{\pi \sim \Pi[m]} \left[ \Pr \left[ \sum_{i=2}^m \mathbf{t}_{\pi(i)} \in T \text{ and } \mathbf{t}_{\pi(1)} = 0 \right] \right] + f'_k \left( \mathbb{E}_{\pi \sim \Pi[m]} \left[ \Pr \left[ 1 + \sum_{i=2}^m \mathbf{t}_{\pi(i)} \in T \text{ and } \mathbf{q}_{\pi(1)} = 1 \right] \right] \right). \end{aligned}$$

Now we perform a double counting argument. Note that when we permute the order P<sup>m</sup> <sup>i</sup>=2 tπ(i) = j and tπ(1) = 0 counts each instance t1, . . . , t<sup>m</sup> with exactly j non-zero locations, for exactly (m − j) × (m − 1)! times. Therefore, we have

$$\mathbb{E}_{\pi \sim \Pi[m]} [\Pr[\sum_{i=2}^m \mathbf{t}_{\pi(i)} \cdot \mathbf{q}_{\pi(i)} \in T \text{ and } \mathbf{t}_{\pi(1)} \cdot \mathbf{q}_{\pi(i)} = 0]] = \sum_{j \in T} \frac{m-j}{m} p_j.$$

With a similar argument we have,

$$\mathbf{E}_{\pi \sim \Pi[m]} [\Pr[1 + \sum_{i=2}^m \mathbf{t}_{\pi(i)} \cdot \mathbf{q}_{\pi(i)} \in T \text{ and } \mathbf{q}_{\pi(1)} = 1]] = \sum_{j \in T} \frac{c' - j + 1}{m} p_{j-1} + \frac{j}{m} p_j.$$

Then, we have

$$\begin{aligned} \sum_{j \in T} p_j &\leq \sum_{j \in T} \frac{m-j}{m} p_j + f'_k(\sum_{j \in T} \frac{j}{m} p_j + \frac{c'-j+1}{m} p_{j-1}) \\ &= \sum_{j \in T} \frac{m-j}{m} p_j + f'_k(\sum_{j \in T} \frac{j}{m} p_j + \frac{c'-j+1}{m} p_{j-1}). \end{aligned}$$

And this implies

$$\sum_{j \in T} \frac{j}{m} p_j \leq f'_k \left( \sum_{j \in T} \frac{j}{m} p_j + \frac{c' - j + 1}{m} p_{j-1} \right).$$

And this, by definition of f ′ k implies

$$\sum_{j \in T} \frac{j}{m} p_j \leq \bar{f}\left(\frac{1}{k-1} \sum_{j \in T} \frac{c' - j + 1}{m} p_{j-1}\right).$$

*Proof of Theorem [3.2.](#page-5-2)* To prove Theorem [3.2,](#page-5-2) we first state and prove a lemma which is consequence of Theorem [3.1.](#page-4-0)

Lemma A.4. *For all* c ≤ c ′ ∈ [m] *let us define*

$$\alpha_c = \sum_{i=c}^{c'} \frac{i}{m} p_i \quad \text{and} \quad \beta_c = \sum_{i=c}^{c'} \frac{c' - i}{m} p_i$$

*We also define a family of functions* r = {ri,j : [0, 1] × [0, 1] → [0, 1]}i≤j∈[m] *and* h = {hi,j : [0, 1] → [0, 1]} *that are defined recursively as follows.*

∀i ∈ [m] : ri,i(α, β) = α *and* hi,i(α, β) = β *and for all* i < j *we have*

$$h_{i,j}(\alpha, \beta) = (k-1)\bar{f}^{-1}\left(r_{i+1,j}(\alpha, \beta)\right)$$

*Then for all* i ≤ j *we have*

$$\alpha_i \geq r_{i,j}(\alpha_j, \beta_j) \quad \text{and} \quad \beta_i \geq h_{i,j}(\alpha_j, \beta_j)$$

*Moreover, for* i < j*,* ri,j *and* hi,j *are increasing with respect to their first argument and decreasing with respect to their second argument.*

*Proof of Lemma [A.4.](#page-16-0)* We prove this by induction on j − i. For j − i = 0, the statement is trivially correct. We have

$$h_{i,j}(\alpha_j, \beta_j) = (k-1)\bar{f}^{-1}(r_{i+1,j}(\alpha_j, \beta_j)).$$

By induction hypothesis, we have ri+1,j (α<sup>j</sup> , β<sup>j</sup> ) ≤ αi+1. Therefore we have

$$h_{i,j}(\alpha_j, \beta_j) \leq (k-1)\bar{f}^{-1}(\alpha_{i+1}). \quad (1)$$

Now by invoking Theorem [3.1,](#page-4-0) we have

$$\alpha_{i+1} \leq \bar{f}\left(\frac{\beta_i}{k-1}\right).$$

Now since ¯f is increasing, this implies

$$(k-1)\bar{f}^{-1}(\alpha_{i+1}) \leq \beta_i \quad (2)$$

Now putting, inequalities [1](#page-17-0) and [2](#page-17-1) together we have hi,j (α<sup>j</sup> , β<sup>j</sup> ) ≤ β<sup>i</sup> . This proves the first part of the induction hypothesis for the function h. Also note that hi,j is increasing in its first component and decreasing in the second component by invoking induction hypothesis and the fact that ¯f −1 is increasing. Now we focus on function ri,j . First note that there is an alternative form for ri,j by opening up the recursive relation. Let γ<sup>z</sup> = z c ′−<sup>z</sup> − z−1 c ′−z+1 . We have ,

$$\begin{aligned} r_{i,j}(\alpha, \beta) &= r_{j,j}(\alpha, \beta) + \frac{i}{c' - i} h_{i,j}(\alpha, \beta) - \frac{j-1}{c' - j + 1} h_{j,j}(\alpha, \beta) + \sum_{z=i+1}^{j-1} \gamma_z h_{z,j}(\alpha, \beta) \\ &= r_{j,j}(\alpha, \beta) + \frac{i}{c' - i} h_{i,j}(\alpha, \beta) - \frac{j}{c' - j} h_{j,j}(\alpha, \beta) + \sum_{z=i+1}^j \gamma_z h_{z,j}(\alpha, \beta) \\ &= \alpha - \frac{j}{c' - j} \beta + \frac{i}{c' - i} h_{i,j}(\alpha, \beta) + \sum_{z=i+1}^j \gamma_z h_{z,j}(\alpha, \beta). \end{aligned} \quad (3)$$

Now we show that for all i we have

$$\alpha_i = \frac{i}{c' - i} \beta_i + \sum_{z=i+1}^m \gamma_z \beta_z. \quad (4)$$

This is because we have

$$\alpha_i - \frac{i}{c' - i}\beta_i = \sum_{z=i+1}^{c'} \left( \frac{z}{m} - \frac{i(c' - z)}{(c' - i)m} \right) p_z.$$

On the other hand we have

$$\begin{aligned} \sum_{z=i+1}^m \gamma_z \beta_z &= \sum_{z=i+1}^m \left( \sum_{z'=i+1}^z \gamma_{z'} \right) \frac{c' - z}{m} p_z \\ &= \sum_{z=i+1}^m \left( \frac{z}{c' - z} - \frac{i}{c' - i} \right) \frac{c' - z}{m} p_z \\ &= \sum_{z=i+1}^m \left( \frac{z}{m} - \frac{i(c' - z)}{(c' - i)m} \right) p_z \end{aligned}$$

and this shows that Equation [4](#page-17-2) is correct. Therefore for all i < j we have

$$\alpha_i - \alpha_j = \frac{i}{c' - i}\beta_i - \frac{j}{c' - j}\beta_j + \sum_{z=i+1}^j \gamma_z \beta_z$$

Now, using the induction hypothesis for h we have,

$$\alpha_i \geq \alpha_j + \frac{i}{c' - i} h_{i,j}(\alpha_j, \beta_j) - \frac{j}{c' - j} \beta_j + \sum_{z=i+1}^j \gamma_z h_{z,j}(\alpha_j, \beta_j). \quad (5)$$

Now verify that the right hand side of Equation [5](#page-18-2) is equal to ri,j (α<sup>j</sup> , β<sup>j</sup> ) by the formulation of Equation [3](#page-17-3)

Also, using the induction hypothesis, we can observe that the right hand side of [3](#page-17-3) is increasing in α<sup>j</sup> and decreasing in β<sup>j</sup> because all terms there are increasing in α<sup>j</sup> and decreasing in β<sup>j</sup> .

This lemma enables us to prove that algorithm [3](#page-5-0) is deciding a valid upper bound on the probability correctly guessing c examples out of c ′ guesses. To prove this, assume that the probability of such event is equal to τ ′ , Note that this means α<sup>c</sup> + β<sup>c</sup> = c ′ <sup>m</sup> τ ′ . Also note that

$$\frac{\alpha_c}{\beta_c} \geq \frac{c}{c' - c} \quad (6)$$

therefore, we have

$$\alpha_c \geq \frac{c}{m} \tau' \quad (7)$$

and β<sup>c</sup> ≤ c ′−c <sup>m</sup> τ ′ . Therefore, using Lemma [A.1](#page-12-0) we have α<sup>0</sup> ≥ r0,c( c <sup>m</sup> τ ′ , c ′−c <sup>m</sup> τ ′ ) and β<sup>0</sup> ≥ h0,c( c <sup>m</sup> τ ′ , c ′−c <sup>m</sup> τ ′ ).

Now we prove a lemma about the function si,j (τ ) = hi,j ( c <sup>m</sup> τ, <sup>c</sup> ′−c <sup>m</sup> τ ) + ri,j ( c <sup>m</sup> τ, <sup>c</sup> ′−c <sup>m</sup> τ ).

Lemma A.5. *the function* si,j (τ ) = hi,j ( c <sup>m</sup> τ, <sup>c</sup> ′−c <sup>m</sup> τ ) + ri,j ( c <sup>m</sup> τ, <sup>c</sup> ′−c <sup>m</sup> τ ) *is increasing in* τ *for* i < j ≤ c*.*

*Proof.* To prove this, we show that for all i < j ≤ c both ri,j ( c <sup>m</sup> τ, <sup>c</sup> ′−c <sup>m</sup> τ ) and hi,j ( c <sup>m</sup> τ, <sup>c</sup> ′−c <sup>m</sup> τ ) are increasing in τ . We prove this by induction on j − i. For j − i = 1, we have

$$h_{i,i+1}(\frac{c}{m}\tau, \frac{c'-c}{m}\tau) = (k-1)\bar{f}^{-1}(\frac{c}{m}\tau).$$

We know that ¯f −1 is increasing, therefore hi,i+1( c <sup>m</sup> τ, <sup>c</sup> ′−c <sup>m</sup> τ ) is increasing in τ as well. For ri,i+1 we have

$$r_{i,i+1}\left(\frac{c}{m}\tau, \frac{c'-c}{m}\tau\right) = \frac{c}{m}\tau + \frac{i}{c'-i}(h_{i,i+1}\left(\frac{c}{m}\tau, \frac{c'-c}{m}\tau\right) - \frac{c'-c}{m}\tau)$$

$$\begin{aligned} r_{i,i+1}\left(\frac{c}{m}\tau, \frac{c'-c}{m}\tau\right) &= \frac{c(c'-i)-i(c'-c)}{m(c'-i)}\tau + \frac{i}{c'-i}h_{i,i+1}\left(\frac{c}{m}\tau, \frac{c'-c}{m}\tau\right) \\ &= \frac{(c-i)c'}{m(c'-i)}\tau + \frac{i}{c'-i}h_{i,i+1}\left(\frac{c}{m}\tau, \frac{c'-c}{m}\tau\right). \end{aligned}$$

We already proved that hi,i+1( c <sup>m</sup> τ, <sup>c</sup> ′−c <sup>m</sup> <sup>τ</sup> ) is increasing in <sup>τ</sup> . We also have (c−i)<sup>c</sup> ′ m(c ′−i) > 0, since i < c. Therefore

$$r_{i,i+1}\left(\frac{c}{m}\tau, \frac{c' - c}{m}\tau\right)$$

is increasing in τ . So the base of induction is proved. Now we focus on j − i > 1. For hi,j we have

$$h_{i,j}\left(\frac{c}{m}\tau, \frac{c'-c}{m}\tau\right) = (k-1)\bar{f}^{-1}(r_{i+1,j}\left(\frac{c}{m}\tau, \frac{c'-c}{m}\tau\right).$$

By the induction hypothesis, we know that ri+1,j ( c <sup>m</sup> τ, <sup>c</sup> ′−c <sup>m</sup> <sup>τ</sup> ) is increasing in <sup>τ</sup> , and we know that ¯<sup>f</sup> −1 is increasing, therefore, hi,j ( c <sup>m</sup> τ, <sup>c</sup> ′−c <sup>m</sup> τ ) is increasing in τ .

For ri,j , note that we rewrite it as follows

$$r_{i,j}(\alpha, \beta) = \alpha - \frac{j}{c' - j} \beta + \sum_{z=i}^{j-1} \lambda_z \cdot h_{z,j}(\alpha, \beta)$$

where λ<sup>z</sup> = ( <sup>z</sup>+1 c ′−z−<sup>1</sup> − c ′−z ) ≥ 0. Therefore, we have

$$\begin{aligned} r_{i,j}\left(\frac{c}{m}\tau, \frac{c'-c}{m}\tau\right) &= \tau\left(\frac{c}{m} - \frac{(c'-c)j}{m(c'-j)}\right) + \sum_{z=i}^{j-1} \lambda_z \cdot h_{z,j}\left(\frac{c}{m}\tau, \frac{c'-c}{m}\tau\right) \\ &= \tau \frac{c'(c-j)}{m(c'-j)} + \sum_{z=i}^{j-1} \lambda_z \cdot h_{z,j}\left(\frac{c}{m}\tau, \frac{c'-c}{m}\tau\right). \end{aligned}$$

Now we can verify that all terms in this equation are increasing in τ , following the induction hypothesis and the fact that λ<sup>z</sup> > 0 and also j ≤ c.

Now using this Lemma, we finish the proof. Note that we have α<sup>0</sup> + β<sup>0</sup> = c m .

So assuming that τ ′ ≥ τ , then we have

$$\frac{c'}{m} = \alpha_0 + \beta_0 \geq s_{0,c}(\tau') \geq s_{0,c}(\tau).$$

The last step of algorithm checks if s0,c ≥ c ′ <sup>m</sup> and it concludes that τ ′ ≤ τ if that's the case, because s0,c is increasing in τ . This means that the probability of having more than c guesses cannot be more than τ .

# B. Ablation Experiments

Reconstruction attacks: To show the effect of the bucket size (k) on the auditing performance, in Figure [5,](#page-20-0) we change the number of examples in the two different setups. In first setup we use 10,000 canaries and change the bucket size from 50 to 5000. In the other setup we only use 100 canaries and change the bucket-size from 3 to 50. Note that in these experiments, we do not use abstention and only consider adversaries that guess all examples.

![](_page_20_Figure_1.jpeg)

Figure 5. Effect of bucket size on the empirical lower bounds for reconstruction attack (Gaussian mechanism with standard deviation 0.6). Left: 10,000 canaries with bucket size up-to 5000. Right: 100 canaries with bucket-size up-to 50. Empirical ϵ is reported at δ = 10−<sup>5</sup> .

Effect of number of guesses In Figures [6–](#page-21-0)[9,](#page-21-1) we compare the theoretical upper bound, our lower bound, and the lower bound of Steinke et al. with varying number of guesses. In total, we have m = 10<sup>7</sup> canaries. The number of correct guesses is determined using Algorithm [4](#page-23-0) (the idealized setting). Then we use our and [\(Steinke et al.,](#page-10-3) [2023\)](#page-10-3)'s auditing with the resulting numbers and report the empirical ϵ. As we can see, both our and Steinke et al.'s auditing procedure achieve the best auditing performance for small number of guesses. This shows the importance of abstention in auditing.

A curious reader might wonder why the number of guesses has such a big impact on empirical privacy. Essentially, our analysis involves estimating how many correct guesses an adversary can make when given a certain number of attempts. We focus on specific percentiles of these distributions. The accuracy of our empirical privacy estimates can vary significantly based on how much the number of correct guesses fluctuates, which is influenced by how many guesses we allow the adversary to make. To explain further, consider a random variable representing the ratio of correct guesses (c) to total guesses (c ′ ). If we reduce the number of guesses, the variance of this ratio tends to decrease because the ratio approaches 1 (the adversary can make more correct guesses when we decrease c ′ ). Conversely, if we increase the number of guesses, the variance can also decrease because having more guesses generally leads to a more stable average, owing to the law of large numbers. This balance makes the number of guesses a crucial factor in optimizing for the best estimate of empirical privacy.

Varying δ and confidence levels: We also examine the effect of δ on the obtained empirical ϵ. We fix the number of canaries to 10<sup>5</sup> and the number of guesses to 1, 500 and the number of correct guesses are set to 1, 429, suggested by the idealized setting. We use a Gaussian mechanism with standard deviation 1.0, we vary the value of δ and the confidence level to observe how they affect the results. Figures [10](#page-22-0) and [11](#page-22-1) shows the bound of [\(Steinke et al.,](#page-10-3) [2023\)](#page-10-3) and our bound, respectively. Note that our lower bounds represent the true behavior of δ independent of the confidence level, in contrast to the bound of [\(Steinke et al.,](#page-10-3) [2023\)](#page-10-3).

# C. Other datasets

We also report in Figure [12](#page-21-2) our privacy analysis method in the black-box attack setting on the tabular dataset of shopping records Purchase [\(Shokri et al.,](#page-10-6) [2017\)](#page-10-6). We replicate the same setup in [\(Zarifzadeh et al.,](#page-11-4) [2023\)](#page-11-4), on a non-private MLP model trained on 25000 samples for 50 epochs. We outperform Steinke et al. method for all numbers of guesses

![](_page_21_Figure_1.jpeg)

![](_page_21_Figure_2.jpeg)

Figure 6. Effect of number of guesses (Gaussian mechanism with standard deviation 1.0). Empirical ϵ is reported at δ = 10−<sup>5</sup> .

![](_page_21_Figure_5.jpeg)

![](_page_21_Figure_6.jpeg)

Figure 7. Effect of number of guesses (Gaussian mechanism with standard deviation 2.0). Empirical ϵ is reported at δ = 10−<sup>5</sup> .

Figure 8. Effect of number of guesses (Gaussian mechanism with standard deviation 4.0). Empirical ϵ is reported at δ = 10−<sup>5</sup> .

Figure 9. Effect of number of guesses (Gaussian mechanism with standard deviation 0.5). Empirical ϵ is reported at δ = 10−<sup>5</sup> .

# D. Experimental details

Figure 12. Comparison with auditing procedure of [\(Steinke et al.,](#page-10-3) [2023\)](#page-10-3) on non-private model trained on Purchase against black-box RMIA method [\(Zarifzadeh et al.,](#page-11-4) [2023\)](#page-11-4). Empirical ϵ is reported at δ = 10−<sup>5</sup> .

Idealized setting: In the idealized setting, we work with a toy version of the mechanism to calculate the *expected* number of correct guesses for the ideal adversary. For Gaussian mechanism, the ideal setting for an adversary is when we have a Gaussian mechanism that is used to calculate the sum of vectors. In this setting, each canary represents a unit vector that is orthogonal to all other canary vectors. Then, given the noisy sum, the adversary will calculate the likelihood of the canary being used in the sum, and then decides on the guesses based on these likelihoods. For the setting that the adversary has more than 2 guesses (k > 2), we use a slightly different idealized setting. In all settings, we run the attack 100 times and average the result to get the expected number of correct guesses. Algorithm [4](#page-23-0) shows how we calculate the number of correct guesses in the idealized setting.

![](_page_21_Figure_11.jpeg)

![](_page_22_Figure_1.jpeg)

![](_page_22_Figure_2.jpeg)

Figure 10. Idealized setting for different values of δ and confidence levels for bounds of [\(Steinke et al.,](#page-10-3) [2023\)](#page-10-3).

Figure 11. Idealized setting for different values of δ and confidence levels for our bounds.

Algorithm 4 Simulate the Number of Correct Guesses

**import numpy as np from scipy.special import** softmax **from numpy.random import** normal, binomial **def** idealized\_setting(target\_noise, n\_guesses, n\_canaries, k): n\_correct\_vec = [] **if** k==2: **for** \_ **in** range(100): s\_vector = binomial(1, 0.5, size=n\_canaries) \* 2 - 1 noise = normal(0, 2\*target\_noise, n\_canaries) noisy\_s = s\_vector + noise sorted\_noisy\_s = np.sort(noisy\_s) threshold\_c = sorted\_noisy\_s[-int(n\_guesses)//2-1] n\_correct = np.ceil(n\_guesses\*(s\_vector[noisy\_s > threshold\_c] == ,→ 1).mean()) n\_correct\_vec.append(n\_correct) **else**: **for** \_ **in** range(100): s\_recon\_vec = np.random.randint(0, k, n\_canaries) s\_vec\_recn\_ohe = np.eye(k)[s\_recon\_vec] s\_recon\_noisy\_vec\_ohe = s\_vec\_recn\_ohe + normal(0, ,<sup>→</sup> np.sqrt(2)\*target\_noise, s\_vec\_recn\_ohe.shape) idx\_max = np.argmax(s\_recon\_noisy\_vec\_ohe, axis=1) buckets = softmax(s\_recon\_noisy\_vec\_ohe/(2\*target\_noise\*\*2), ,<sup>→</sup> axis=1)[np.arange(s\_recon\_noisy\_vec\_ohe.shape[0]), idx\_max] sorted\_buckets = np.sort(buckets) bucket\_c\_thr = sorted\_buckets[-int(n\_guesses)] n\_correct\_rec = np.ceil( n\_guesses\*(s\_recon\_vec[buckets > bucket\_c\_thr] == s\_recon\_noisy\_vec\_ohe[buckets > bucket\_c\_thr].argmax(1)).mean() ,→ ,→ ) n\_correct\_vec.append(n\_correct\_rec) **return** int(np.array(n\_correct\_vec).mean(0))

## Auditing code

Here we include the code to compute empirical epsilon.

**from scipy.stats import** norm **import numpy as np** # Calculate h and r recursively (no abstentions) **def** rh(inverse\_blow\_up\_function, alpha, beta, j, m, k=2): # Initialize lists to store h and r values h = [0 **for** \_ **in** range(j + 1)] r = [0 **for** \_ **in** range(j + 1)] # Set initial values for h and r h[j] = beta r[j] = alpha # Iterate from j-1 to 0 **for** i **in** range(j - 1, -1, -1): # Calculate h[i] using the maximum of h[i+1] and a scaled inverse ,<sup>→</sup> blow-up function h[i] = max(h[i + 1], (k - 1) \* inverse\_blow\_up\_function(r[i + 1])) # Update r[i] based on the difference between h[i] and h[i+1] r[i] = r[i + 1] + (i / (m - i)) \* (h[i] - h[i + 1]) # Return the lists of h and r values **return** (r, h) # Audit function without abstention **def** audit\_rh(inverse\_blow\_up\_function, m, c, threshold=0.05, k=2): # Calculate alpha and beta values alpha = threshold \* c / m beta = threshold \* (m - c) / m # Call the rh function to get the lists of h and r values r, h = rh(inverse\_blow\_up\_function, alpha, beta, c, m, k) # Check if the differential privacy condition is satisfied **if** r[0] + h[0] > 1.0: **return False else**: **return True** # Calculate h and r recursively (with abstentions) **def** rh\_with\_cap(inverse\_blow\_up\_function, alpha, beta, j, m,c\_cap, k=2): h=[0 **for** i **in** range(j+1)] r=[0 **for** i **in** range(j+1)] h[j]= beta r[j]= alpha **for** i **in** range(j-1,-1,-1): h[i]=max(h[i+1],(k-1)\*inverse\_blow\_up\_function(r[i+1])) r[i]= r[i+1] + (i/(c\_cap-i))\*(h[i] - h[i+1]) **return** (r,h) # Audit function with abstentions **def** audit\_rh\_with\_cap(inverse\_blow\_up\_function, m, c,c\_cap, threshold=0.05, ,→ k=2): threshold=threshold\*c\_cap/m

alpha=(threshold\*c/c\_cap) beta=threshold\*(c\_cap-c)/c\_cap r,h=rh\_with\_cap(inverse\_blow\_up\_function, alpha, beta, c, m, c\_cap, k) **if** r[0]+h[0]>c\_cap/m: **return False else**: **return True** # Calculate the blow-up function for Gaussian noise **def** gaussianDP\_blow\_up\_function(noise): **def** blow\_up\_function(x): # Calculate the threshold value threshold = norm.ppf(x) # Calculate the blown-up threshold value blown\_up\_threshold = threshold + 1 / noise # Return the CDF of the blown-up threshold value **return** norm.cdf(blown\_up\_threshold) **return** blow\_up\_function # Calculate the inverse blow-up function for Gaussian noise **def** gaussianDP\_blow\_up\_inverse(noise): **def** blow\_up\_inverse\_function(x): # Calculate the threshold value threshold = norm.ppf(x) # Calculate the blown-up threshold value blown\_up\_threshold = threshold - 1 / noise # Return the CDF of the blown-up threshold value **return** norm.cdf(blown\_up\_threshold) **return** blow\_up\_inverse\_function # Define a function to calculate delta for Gaussian noise **def** calculate\_delta\_gaussian(noise, epsilon): # Calculate delta using the formula delta = norm.cdf(-epsilon \* noise + 1 / (2 \* noise)) - np.exp(epsilon) \* ,<sup>→</sup> norm.cdf(-epsilon \* noise - 1 / (2 \* noise)) **return** delta # Define a function to calculate epsilon for Gaussian noise **def** calculate\_epsilon\_gaussian(noise, delta): # Set initial bounds for epsilon epsilon\_upper = 100 epsilon\_lower = 0 # Perform binary search to find epsilon **while** epsilon\_upper - epsilon\_lower > 0.001: epsilon\_middle = (epsilon\_upper + epsilon\_lower) / 2 **if** calculate\_delta\_gaussian(noise, epsilon\_middle) > delta: epsilon\_lower = epsilon\_middle **else**: epsilon\_upper = epsilon\_middle # Return the upper bound of epsilon **return** epsilon\_upper # Get the empirical epsilon value

**def** get\_gaussian\_emp\_eps\_ours(candidate\_noises, inverse\_blow\_up\_functions, m, ,→ c, threshold, delta, k=2): # Initialize the empirical privacy index empirical\_privacy\_index = 0 # Iterate through candidate noises until the privacy condition fails **while** audit\_rh(inverse\_blow\_up\_functions[empirical\_privacy\_index], m, c, ,→ threshold=0.05, k=k): empirical\_privacy\_index += 1 # Get the empirical noise and calculate the empirical epsilon empirical\_noise = candidate\_noises[empirical\_privacy\_index] empirical\_eps = calculate\_epsilon\_gaussian(empirical\_noise, delta=delta) # Return the empirical epsilon **return** empirical\_eps # Set target noise and generate candidate noises target\_noise = 0.6 candidate\_noises=[target\_noise+ i\*0.01 **for** i **in** range(1000)] inverse\_blow\_up\_functions=[gaussianDP\_blow\_up\_inverse(noise) **for** noise **in** ,→ candidate\_noises]