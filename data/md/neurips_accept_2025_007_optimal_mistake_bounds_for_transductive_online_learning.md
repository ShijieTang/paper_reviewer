# Optimal Mistake Bounds For Transductive Online Learning

Zachary Chase Kent State University zchase2@kent.edu Steve Hanneke Purdue University steve.hanneke@gmail.com Shay Moran Departments of Mathematics, Computer Science, and Data and Decision Sciences Technion - Israel Institute of Technology; Google Research smoran@technion.ac.il Jonathan Shafer MIT
shaferjo@mit.edu

## Abstract

We resolve a 30-year-old open problem concerning the power of unlabeled data in online learning by tightly quantifying the gap between transductive and standard online learning. In the standard setting, the optimal mistake bound is characterized by the Littlestone dimension d of the concept class H (Littlestone, 1987). We prove that in the transductive setting, the mistake bound is at least Ω
√d
. This constitutes an exponential improvement over previous lower bounds of Ω(log log(d)), Ω
plog(d)
, and Ω(log(d)), due respectively to Ben-David, Kushilevitz, and Mansour (1995, 1997), and Hanneke, Moran, and Shafer (2023). We also show that this lower bound is tight: for every d, there exists a class of Littlestone dimension d with transductive mistake bound O
√d
. Our upper bound also improves upon the best known upper bound of (2/3) · d from Ben-David et al. (1997). These results establish a quadratic gap between transductive and standard online learning, thereby highlighting the benefit of advance access to the unlabeled instance sequence. This contrasts with the PAC setting, where transductive and standard learning exhibit similar sample complexities.

## 1 **Introduction**

The transductive model is a basic and well-studied framework in learning theory, dating back to the early works of Vapnik. It has been investigated both in statistical and online settings, and is motivated by the principle that to make good predictions on a specific set of test instances, one need not construct a fully general classifier that performs well on the entire domain - including points that may never actually appear. Rather, it may be sufficient to tailor predictions for a fixed, known set of instances. This perspective naturally connects to a broader question in learning theory: what is the value of unlabeled data? In the transductive setting, the learner is given the sequence of unlabeled test instances in advance and is then required to predict their labels one by one. Thus, the transductive model can be viewed as a natural formalization of learning with unlabeled data: the test instances are known in advance, but their labels are not. The central question is whether such prior access to the unlabeled sequence can help reduce the number of prediction mistakes - compared to the standard online model, where the instances arrive and are labeled one at a time.

Recall for instance that in the standard PAC1 model of supervised learning, there are cases where access to unlabeled data is not helpful. Indeed, the "hard population distributions" used to prove the standard VC2lower bound are constructed by taking a fixed and known marginal distribution over a VC-shattered set. Namely, the cases that are hardest to learn in the PAC setting include ones where the learner knows the marginal distribution over the domain, and can therefore generate as much unlabeled data as it wishes. And yet, in those cases, access to unlabeled data provides no acceleration compared to an algorithm (like ERM3) that does not use unlabeled data.

Seeing as unlabeled data is often a lot easier to obtain than labeled data, there have been considerable efforts to understand when and to what extent can access to unlabeled data accelerate learning.4 In particular, it is natural to ask, for which plausible models of learning is access to unlabeled data beneficial? Online learning (Littlestone, 1987) is perhaps the model of learning that is mostextensively studied in learning theory after the PAC model and its variants. Therefore, the general question considered in this paper is: Question 1. Quantitatively, how much (if at all) is access to unlabeled data beneficial for learning in the online learning setting?

This question is naturally instantiated by comparing *transductive* online learning - where the learner has advance access to the full sequence x1, x2*, . . . , x*n of unlabeled instances - with *standard* online learning, where no such access is given. This perspective has also been adopted in prior work: for example, Kakade and Kalai (2005), Cesa-Bianchi and Shamir (2013), and Hoi, Sahoo, Lu, and Zhao (2021) (Section 7.3) all describe transductive online learning as a setting in which the learner has access to "unlabeled data". We thus refine the question above as follows: Question 2. *Quantitatively, how much (if at all) is learning in the transductive online learning setting* easier than learning in the standard online learning setting? Specifically, how much is the optimal number of mistakes in the transductive setting smaller than in the standard setting? Addressing this question, our main result (Theorem 1.1) states that the optimal number of mistakes in the transductive setting (with access to unlabeled data) is at most quadratically smaller than in the standard setting (without unlabeled data). Furthermore, there are hypothesis classes for which a quadratic gap is achieved.

## 1.1 Setting: Standard Vs. Transductive **Online Learning**

Standard online learning (Littlestone, 1987) is a zero-sum, perfect- and complete-information game played over n rounds between two players, a *learner* and an *adversary*. The game is played with respect to a *domain* set X and a hypothesis class *H ⊆ {*0, 1}
X (consisting of functions *X → {*0, 1}),
where n, X and H are fixed and known to both players. The game proceeds as in Game 1. The number of mistakes for a learner L and an adversary A is Mstd(H*, n, L, A*) = |{t ∈ [n] : ˆyt ̸= yt}|.

We are interested in understanding the *optimal number of mistakes*, which is Mstd(H) = sup n∈N
inf L∈L
sup A∈A
Mstd(H*, n, L, A*),
where A and L are the set of all deterministic adversaries and learners, respectively.5 For each round t = 1, 2*, . . . , n*:
a. The adversary selects an instance xt ∈ X and sends it to the learner. b. The learner selects a prediction yˆt ∈ {0, 1} and sends it to the adversary.

c. The adversary selects a label yt ∈ {0, 1} and sends it to the learner. The selected label must be *realizable*, meaning that ∃h *∈ H ∀*i ∈ [t]: h(xi) = yi.

Game 1: The standard online learning setting.

The adversary selects a *sequence* x1, x2, . . . , xn ∈ X and sends it to the learner.

For each round t = 1, 2*, . . . , n*:
a. The learner selects a *prediction* yˆt ∈ {0, 1} and sends it to the adversary. b. The adversary selects a label yt ∈ {0, 1} and sends it to the learner. The selected label must be *realizable*, meaning that ∃h *∈ H ∀*i ∈ [t]: h(xi) = yi.

Game 2: The transductive online learning setting.

It is well known that Mstd(H) is characterized by the the Littlestone dimension, namely, Mstd(H) =
LD(H) (see Theorem A.7 and Definition A.6).

The *transductive* online learning setting (Ben-David et al., 1995, 1997) is similar, except that the learner has access to the full sequence of unlabeled instances in advance. Namely, as in Game 2. The optimal number of mistakes for the transductive setting is defined exactly as before, Mtr(H*, n, L, A*) = |{t ∈ [n*] : ˆ*yt ̸= yt}|, and Mtr(H) = sup n∈N
inf L∈L
sup A∈A
Mtr(H*, n, L, A*),
with the only difference between the standard quantity Mstd(H) and the transductive quantity Mtr(H)
being in how the game is defined.

## 1.2 **Main Result**

Notice that for every hypothesis class H, Mtr(H) ≤ Mstd(H). Indeed, in the transductive setting the adversary declares the sequence x at the start of the game. This reduces the number of mistakes because the transductive adversary is less powerful (it cannot adaptively alter the sequence mid-game),
and also because the transductive learner is more powerful (it has more information).6 While for some classes Mtr(H) = Mstd(H), we study the largest possible separation. The best previous lower bound on Mtr, due to Hanneke, Moran, and Shafer (2023), states that for every class H,Mtr(H) ≥ Ω(log(d)),
where d = Mstd(H). In the other direction, Ben-David et al. (1997) constructed7a class H such that Mstd(H) = d and Mtr(H) ≤
2 3 d. This left an exponential gap between the best known lower and upper bounds on Mtr, namely Ω(log d) versus 23 d. Our main result closes this gap:
Theorem 1.1 (Main result).

- For every hypothesis class *H ⊆ {*0, 1}
X ,
Mtr(H) = Ω√d
,
- On the other hand, for every d there exists a hypothesis class H *with* Mstd(H) = d and

$$M_{\mathrm{tr}}({\mathcal{H}})=O{\Big(}{\sqrt{d}}{\Big)}.$$

This result is stated in considerably greater detail in Theorems B.1 and D.1.

## 1.3 **Related Works**

The notion of *transductive inference* as a more efficient alternative to *inductive inference* in statistical learning theory was introduced by Vapnik (1979, 2006); Gammerman, Vovk, and Vapnik (1998); Chapelle, Vapnik, and Weston (1999). The *online learning* setting is due to Littlestone (1987), who also proved that the optimal number of mistakes is characterized by the Littlestone dimension (see Theorem A.7). The *transductive online learning* setting studied in the current paper, was first defined by Ben-David, Kushilevitz, and Mansour (1995), who used the name *worst sequence off-line model*. Among other results, they showed a lower bound of Ω(log log(d)) on the number of mistakes required to learn a class with Littlestone dimension d. The authors subsequently presented an exponentially stronger lower bound of Ω
plog(d)
in Ben-David, Kushilevitz, and Mansour (1997). However, understanding where the optimal number of mistakes is situated within the range hΩ
plog(d)
, 2d/3 iremained an open question. Kakade and Kalai (2005) presented an oracle-efficient algorithm for the transductive online learning setting, and may have been the first to use that name. Their result was subsequently improved upon by Cesa-Bianchi and Shamir (2013). The present work is most similar to that of Hanneke, Moran, and Shafer (2023) which, among other results, gave a quadratically-stronger mistake lower bound of Ω(log(d)) for classes with Littlestone dimension d in the transductive online setting. The proof of our lower bound utilizes some of their ideas, but yields a quantitative improvement by combining it with some new ideas.

Hanneke, Raman, Shaeiri, and Subedi (2024) studied a setting of *multi-class* transductive online learning where the number of possible labels is unbounded.

## 2 **Technical Overview**

In this section we explain some of the main ideas in our proofs. Formal definition appear in Section A. Full formal statements of the results, as well as detailed rigorous proofs, appear in Sections B to D.

## 2.1 **Paths In Trees**

We make extensive use of the following notion. Given a perfect binary tree Td of depth d, every function f : Td → {0, 1} defines a unique *path* in the tree. The path is a sequence of nodes path(f) = (xi0, xi1*, . . . , x*id), as explained in Figure 1c. See Section A for formal definitions.

## 2.2 **Proof Ideas For The Lower Bound**

We start with an elementary observation about the adversary's dilemma in the transductive online learning setting. Before round t of the game, the adversary selected a full sequence of instances x1, x2, . . . , xn ∈ X , and assigned some initial labels y1, y2, . . . , yt−1 ∈ {0, 1}. At the start of round t, the adversary must consider the *version space*,
Ht =h ∈ H : (∀i ∈ [t − 1] : h(xi) = yi)	.

If all h ∈ Ht assign h(xt) = b for some b ∈ {0, 1}, then the adversary has no choice but to assign the label yt = b. Otherwise, the adversary can *force a mistake* at time t. Namely, after seeing the learner's prediction yˆt, the adversary can assign yt = 1 − yˆt, incrementing the number of learner mistakes by 1.

But "just because you can, doesn't mean you should". If the adversary is greedy and forces a mistake at time t, they may pay dearly for that later. As an extreme example, consider the case where there

x0 x1 x2 x3 x4 x5 x6 x0 0 1 x1 x2 0 1 0 1 x3 x4 x5 x6 0 1 0 1 0 1 0 1
- •
- •
- •
- •
x0 0 1 x1 x2 0 1 0 1 x3 x4 x5 x6 0 1 0 1 0 1 0 1
- •
- •
- •
- •
λ 0 1 0 1 0 1 0 1 00 01 10 11 0 1 0 1 0 1 0 1
- •
- •
- •
- •
is a single h1 ∈ Ht that assigns h1(xt) = 1, and all other functions h ∈ Ht assign h(xt) = 0.

If the learner selects yˆt = 1 and the adversary forces a mistake at time t, the version space at all subsequent times *s > t* will be Hs = {h1}, and the adversary will be prevented from forcing any further mistakes. A natural strategy for the adversary is therefore to be greedy up to a certain limit. Namely, at each time t the adversary computes the ratio8

$$r_{t}={\frac{|\{h\in{\mathcal{H}}_{t}:\,h(x_{t})=1\}|}{|{\mathcal{H}}_{t}|}}.$$

If rt ∈ [ε, 1 − ε] for some parameter ε > 0 ("the version space is not too unbalanced"), then the adversary forces a mistake. Otherwise, the adversary assigns the majority label, i.e., yt =
1(rt ≥ 1/2). This ensures that the version space does not shrink too fast:
- If no mistake is forced, then |Ht+1| ≥ (1 − ε) *· |H*t|, and - If a mistake is forced, |Ht+1| ≥ ε *· |H*t|.

$$|-\varepsilon)\cdot|{\mathcal{H}}_{t}|,{\mathrm{and}}$$

In particular, at the end of the game, the version space Hn+1 is of size |Hn+1| ≥ ε M · (1 − ε)
n−M *· |H| ≥* ε M · (1 − ε)
n· 2 d, (1)
where M is the number of mistakes that the adversary forces and n is the length of the sequence. The class has size *|H| ≥* 2 d because LD(H) = d, and by removing functions from the class if necessary
(which can only make learning easier), we may assume without loss of generality that |H| = 2d.

Namely, the class precisely shatters a Littlestone tree of depth d − 1 such that for every assignment of labels to a root-to-leaf path in the tree, the class contains exactly one function that agrees with that assignment (see Definition A.6 for detail). Notice that we have not yet specified how the adversary selects the sequence x. While the adversary's labeling strategy is extremely simple (determined by the ratio rt and the prediction yˆt), constructing of the sequence x requires some care, to ensure that it has the following two properties:
- **Property I:** The length n of the sequence satisfies n = 2Θ(
√d), and
- **Property II:** For every sequence of predictions yˆ1*, . . . ,* yˆn selected by the learner, the resulting sequence of labels y1*, . . . , y*n selected by the adversary are consistent with some function h ∈ H such that x contains all the nodes in path(h).

9 These properties can be achieved by carefully simulating all possible execution paths of the adversary.

Observe that if path(h) = (u1*, . . . , u*d) then the sequence of labels h(u1), h(u2)*, . . . , h*(ud)
uniquely identifies the function h within the class H. Hence, Property II and the assumption |H| = 2dimply that at the end of the game, the version space Hn+1 has cardinality |Hn+1| = 1. (2)
Combining Property I (n = 2Θ(
√d)), Eqs. (1) and (2), and choosing ε = 2−Θ(
√d) gives

$\downarrow$ . 
1 ≥ ε M · (1 − ε)
n· 2 d ≥ 2
−Θ(M·
√d)· 2 d, which implies M = Ω√d
, as desired.

## 2.3 **Proof Ideas For The Upper Bound**

In this section we explain the main ideas in the proof of Theorem D.1, which states that for every d ∈ N, there exists a class of Littlestone dimension d that is learnable in the transductive online setting with a mistake bound of O
√d
.

Of course, not every Littlestone class satisfies this property. For instance, the set of all functions
[d] → {0, 1} has Littlestone dimension d, but the adversary can force the learner to make d mistakes when learning this class in the transductive setting.10 So our task in this proof is to construct a class that is especially easy to learn in the transductive setting (i.e., learnable with O
√d mistakes), while still being hard (requiring d mistakes) in the standard setting.

## 2.3.1 **Sparse Encodings Are Easy To Guess**

We start with an elementary observation. Consider the following two bit strings:
Binary: 110101 One-hot: 0000000000000000000000000000000000000000000000000000100000000000 Both of these strings encode the number 53. However, one of the encodings is much easier to guess than the other: suppose we are tasked with guessing the bits in an encoding of an integer between 0 and 2 6 − 1. We guess the bits one at a time, and after each guess, an adaptive adversary tells us whether our guess was correct.

9Recall that the *path* of a function h is depicted in Figure 1c, and defined in Definition A.5.

10The adversary simply selects the sequence x = (1, 2, *3, . . . , d*), and for each xi, the adversary forces a mistake by selecting yi = 1 − yˆi. The adversary's choice of labels is realizable because we are working with the class of all function [d] → {0, 1}.

Now, if the bit string is a binary encoding, the task is hard. Each bit can either be 0 or 1, regardless of the values of the previous bits, and so the adversary can force a mistake on every bit. On the other hand, if we know that the string is a one-hot encoding, there exists an attractive strategy - always guess 0. This ensures that we will make at most 1 mistake.

Note that at the end of the guessing game we have learned the same amount of *information* (for a number between 0 and 2 n − 1, we learned n bits of information), but the number of *mistakes* is very different (n mistakes vs. 1 mistake).

## 2.3.2 **Construction Of The Hypothesis Class**

We now describe a construction of a hypothesis class that is easy to learn in the transductive setting, using the idea of a sparse encoding. Recall that a class H has Littlestone dimension at least d (Definition A.6 in Section A) if there exists a Littlestone tree of depth d − 1 such that for every b ∈ {0, 1}
dthere exists h = hb ∈ H such that the values on the path of h agree with b. More formally, ∀i ∈ [d] : h(b<i) = bi, and in particular path(h) = (λ, b≤1, b≤2, b≤3*, . . . , b*≤d−1). Thus, when constructing a class that shatters a specific Littlestone tree of depth d − 1, we need to define 2 dfunctions hb : b ∈ {0, 1}
d	. For each function hb, the on-path values of the function are fixed
(fully determined by b), while for the remaining values there is complete freedom (for the nodes u that are off-path we may assign any values hb(u) ∈ {0, 1}).

Perhaps the simplest way to construct a class of Littlestone dimension d is simply to assign all on-path values as required, and assign 0 to all other values. Namely, if u is a prefix of b then hb(u) = b|u|+1, and otherwise hb(u) = 0. In a sense, this is the 'minimal' class of Littlestone dimension d for a specific Littlestone tree.11 Observe that the 'minimal' class does not have the desired property of being easy to learn in the transductive setting.12 However, a certain variation of the 'minimal' class that embeds a sparse encoding does satisfy the requirement. In this variation, on-path value of the function hb are assigned as they must (as determined by b), while the off-path values are sampled independently using a biased coin, such that each of them is 0 with high probability, but has a small probability of being 1. The probability is chosen carefully so that the class satisfies some simple combinatorial properties, as described further in Section 2.3.6 and Lemma D.2.

## 2.3.3 **Naïve Learning Strategy**

information about the true labeling function. Additionally, when the adversary selects an off-path label of 1, that reveals a lot of information about the true labeling function (such labels are rare in the hypothesis class), and therefore the adversary cannot force many off-path mistakes. Overall, the information about the true labeling function is 'smeared' throughout all labels of the tree (0s and 1s, on-path and off-path).14 Thus, the naïve general strategy for the learner when using the probabilistically-constructed class is to learn most of the information about the true labeling function by observing off-path labels. By the time the learner reaches an on-path node, it hopefully has already learned enough about the true labeling function in order to make a good prediction on that node. However, making this general strategy work requires overcoming some very substantial obstacles:
1. Recall that in the transductive setting, the adversary can present the nodes of the tree in any order of its choosing - it does not have to present the tree in breadth-first order. The naïve strategy works only if the learner sees many off-path nodes before it sees most on-path nodes. But what happens if the adversary decides to present many on-path nodes near the beginning of the sequence? To handle this, the learner incorporates a strategy we call 'danger zone minimization', as described in Section 2.3.4.

2. Another, equally problematic, issue also arises from the fact that the sequence presented by the adversary might not be in breadth-first order. Recall that breadth-first order15 has the property that for every node u in the sequence, all the ancestors of u appear *before* u in the sequence. This means that by the time the learner needs to predict a label for u, the learner knows whether u is on-path or off-path for the true labeling function. But what happens if the adversary presents u before some of u's ancestors? Or omits some of u's ancestors from the sequence altogether? In this case the learner doesn't know if u is on-path or off-path, and this presents a double hazard. One hazard is that the leaner doesn't know what label to predict for u - if u is off-path, the learner can simply predict 0, but if it is on-path it must do something more elaborate. The second hazard is that, after seeing the correct label for u, it is not clear what the learner can infer from it. If u is off-path, its label should be interpreted as part of a sparse encoding of the labeling function. But if u is on-path, the interpretation must be entirely different. To overcome this challenge, the learner incorporates a strategy we call 'splitting experts', described in Section 2.3.5.

3. Limiting off-path mistakes. Thanks to the coin's bias, most off-path nodes have a true label of 0. Nonetheless, each function in the hypothesis class still has an expected number of 2 Ω(d)
off-path nodes labeled 1, so the learner can afford to misclassify only a vanishing fraction of them! To limit the number of mistakes, the learner extracts information from the sparse encoding and executes a 'transition to Halving' strategy, as described in Section 2.3.6.

## 2.3.4 **Danger Zone Minimization**

Concretely, at the beginning of the game the learner initializes a set S = {x1, x2*, . . . , x*tmax} consisting of the first tmax = 2Ω(
√d)instances in the sequence x selected by the adversary. This set represents the 'danger zone' - nodes in the beginning of the sequence that have not been labeled yet, that *might* be on-path, and that are not ancestors of a previously-labeled on-path node.16 To predict a label for an instance xi, the learner selects a label yˆi such that if yˆiis wrong, the danger zone will shrink by at least 1/3. That is, for b ∈ {0, 1}, if the set Sb of b-descendants of xi has cardinality |Sb*| ≥ |*S|/3, the learner predicts yˆi = b. Then, if the adversary selects yi = 1 − b, that implies that all b-descendants of xi are off-path for the true labeling functions. Therefore, the learner removes all b-descendants of xi from the danger zone, and the new cardinality is |S \ Sb| ≤ (2/3) · |S|. This guarantees that the learner can make at most O*(log(*tmax)) = O
√d such mistakes before the danger zone is empty.17 If neither S0 nor S1 have cardinality at least |S|/3, the learner predicts yˆi = 0. If yi = 1 and xiis on-path for the true labeling function, then the learner updates the danger zone to be S0 ∪ S1, 18 again shrinking the danger zone by a factor of at most 2/3. Otherwise, if yi = 1 and xiis off-path, then it was an off-path node labeled 1 (which is rare), and the learner can afford to misclassify it (see Section 2.3.6).

## 2.3.5 **Splitting Experts**

The danger zone minimization strategy requires that the learner know whether the node u being classified is on-path or off-path for the true labeling function. However, if u appears in the sequence before some of its ancestors, the learner does not know this. To overcome this difficulty, the learner implements a variant of the standard *multiplicative weights algorithm* using *splitting experts*. This means that initially there is a single expert executing danger zone minimization. When a node u is reached for which danger zone minimization requires knowing whether u is on-path or off-path and that information is not yet evident, each expert is split into two experts, one of which continues the execution of danger zone minimization under the assumption that u is on-path, and the other under the opposite assumption. Thus, at each point in time, there exists precisely one expert for which all path-related assumptions are correct, and therefore that expert will make at most O
√d mistakes.

The multiplicative weights algorithm guarantees that the overall number of mistakes will be linear in the the number of mistakes of the best expert, i.e., O
√d
.

## 2.3.6 **Transition To Halving**

The hypothesis class is engineered such that it satisfies the following property: there are at most 2 O(
√d)functions in the hypothesis class that agree with any set of tmax = 2Ω(
√d)labels, or that agree that a set of Θ
√d nodes are all off-path and labeled 1 (this follows from Lemma D.2).

Therefore, once the true labels for the first tmax instances x1, x2*, . . . , x*tmax have been revealed, or once Θ
√d off-path labels of 1 have been revealed (whichever happens first), the learner can *transition* to halving: stop doing danger zone minimization, and instead predict the labels for the remaining nodes using the standard Halving algorithm (Algorithm 7) on the subset of the hypothesis class that survived. Halving on 2 O(
√d)functions is guaranteed to make at most O
√d mistakes (Fact E.1).

However, seeing as the learner lacks information on which nodes are off-path, it uses experts, and each expert maintains different path-related assumptions. Thus, each expert decides separately at which point to transition to Halving. The unique expert that makes only correct assumptions will transition 'at the right time'. That expert will make at most O
√d mistakes during danger zone minimization, and then at most O
√d additional mistakes during halving.

## 2.4 **Some Intuition For The Quantity** √D

We briefly sketch where the quantity √d arises from. This is a back-of-the-envelope calculation without proof, intended purely as an aid for intuition. Suppose we assigned off-path labels of 1 with probability 2
−kinstead of 2
−
√d. Consider a sequence x1*, . . . , x*n of n = d/2k leaves. For any sequence of labels y1, . . . , yn ∈ {0, 1}, taking s =Pi∈[n]
yi, there exist roughly

$$2^{d}\cdot\left(2^{-k}\right)^{s}\cdot\left(1-2^{-k}\right)^{n-s}\geq2^{d}\cdot\left(2^{-k}\right)^{n}\gg0$$

functions in the class for which these leaves are off-path and which agree with the labels y1*, . . . , y*n.

Therefore, the adversary can force at least n = Ω(d/k) mistakes.

Similarly, for the sequence x1*, . . . , x*n consisting of all the nodes in the tree of depth at most k/2 in breadth-first order, the adversary can force a mistake on every on-path node while assigning a label of 0 to all off-path nodes, for a total of k/2 mistakes. This is true because for any assignment of on-path labels, the fraction of functions which agree with the on-path labels that assign a label of 0 to all off-path nodes is roughly 1 − 2
−k2 k/2
≈ 1, so in particular for any labeling of the on-path nodes there exists a function in the class that agrees with that labeling and assigns 0 to all off-path nodes.

Therefore, for any k, we obtain a *lower bound* of Ωdk + kon the number of mistakes. For any k, d k + k ≥
√d, giving a lower bound of Ω
√d
. Choosing k =
√d to minimize the lower bound will in fact yield a matching upper bound of O
√d
, as we show in this paper. This completes our overview of the upper bound.

## 3 **Directions For Future Work**

Following are some interesting open questions:
1. Does there exist an efficient learning algorithm that achieves the O
√d upper bound of Theorem D.1? One needs to be careful about the definition of efficiency here, but one possible formalization is as follows. Does there exist a learning algorithm A and a sequence of classes H1, H2*, . . .* , such that for every d ∈ N:
- LD(Hd) = d, and
- Given as input the index d and a sequence x1*, . . . , x*n, the algorithm A runs in time poly(*d, n*) and makes at most O
√d mistakes assuming the labels are realizable by Hd.

2. Is there a tradeoff between the cardinality of the domain X and the upper bound on the number of mistakes? We used a domain of size roughly 2 din order to obtain our upper bound of O
√d
. Is it possible to get the same bound with a domain of size poly(d)?

3. Obtaining more precise asymptotics; for example, is there (an explicit) constant α > 0 such that the optimal transductive mistake bound is α + o(1)√d?

## 4 **Organization**

Complete rigorous mathematical details are deferred to the appendices. Formal definitions appear in Section A. Formal statements and proofs for the lower bound and upper bound appear in Section B
and Section D, respectively. Optimal sequence length is discussed in Section C.

## Acknowledgments And Disclosure Of Funding

ZC is supported in part by NSF EnCORE inst (award \#2217058) and by Shachar Lovett's Simons Investigator Award (\#929894). SM is a Robert J. Shillman Fellow; he acknowledges support by ISF grant 1225/20, by BSF grant 2018385, by Israel PBC-VATAT, by the Technion Center for Machine Learning and Intelligent Systems (MLIS), and by the the European Union (ERC, GENERALIZATION, 101039692). JS is supported in part by NSF CNS-2154149, an Amazon Research Award, and by Vinod Vaikuntanathan's Simons Investigator Award. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

## References

Maria-Florina Balcan and Avrim Blum. A discriminative model for semi-supervised learning. J.

ACM, 57(3):19:1–19:46, 2010. doi:10.1145/1706591.1706599. URL https://doi.org/10.

1145/1706591.1706599.

Shai Ben-David, Eyal Kushilevitz, and Yishay Mansour. Online learning versus offline learning.

In Paul M. B. Vitányi, editor, Computational Learning Theory, Second European Conference, EuroCOLT '95, Barcelona, Spain, March 13-15, 1995, Proceedings, volume 904 of Lecture Notes in Computer Science, pages 38–52. Springer, 1995. doi:10.1007/3-540-59119-2_167. URL https://doi.org/10.1007/3-540-59119-2_167.

Shai Ben-David, Eyal Kushilevitz, and Yishay Mansour. Online learning versus offline learning.

Mach. Learn., 29(1):45–63, 1997. doi:10.1023/A:1007465907571. URL https://doi.org/10. 1023/A:1007465907571.

Shai Ben-David, Tyler Lu, Dávid Pál, and Miroslava Sotáková. Learning low-density separators.

CoRR, abs/0805.2891, 2008. URL http://arxiv.org/abs/0805.2891.

Gyora M. Benedek and Alon Itai. Learnability with respect to fixed distributions. *Theor. Comput.*
Sci., 86(2):377–390, 1991. doi:10.1016/0304-3975(91)90026-X. URL https://doi.org/10. 1016/0304-3975(91)90026-X.

Avrim Blum and Tom M. Mitchell. Combining labeled and unlabeled data with co-training. In Peter L. Bartlett and Yishay Mansour, editors, Proceedings of the Eleventh Annual Conference on Computational Learning Theory, COLT 1998, Madison, Wisconsin, USA, July 24-26, 1998, pages 92–100. ACM, 1998. doi:10.1145/279943.279962. URL https://doi.org/10.1145/279943. 279962.

Olivier Bousquet, Steve Hanneke, Shay Moran, Ramon van Handel, and Amir Yehudayoff. A
theory of universal learning. In Samir Khuller and Virginia Vassilevska Williams, editors, STOC 2021: 53rd Annual ACM SIGACT Symposium on Theory of Computing, Virtual Event, Italy, June 21-25, 2021, pages 532–541. ACM, 2021. doi:10.1145/3406325.3451087. URL https:
//doi.org/10.1145/3406325.3451087.

Nicolò Cesa-Bianchi and Ohad Shamir. Efficient transductive online learning via randomized rounding. In Bernhard Schölkopf, Zhiyuan Luo, and Vladimir Vovk, editors, Empirical Inference -
Festschrift in Honor of Vladimir N. Vapnik, pages 177–194. Springer, 2013. doi:10.1007/978-3642-41136-6_16. URL https://doi.org/10.1007/978-3-642-41136-6_16.

Olivier Chapelle, Vladimir N. Vapnik, and Jason Weston. Transductive inference for estimating values of functions. In Sara A. Solla, Todd K. Leen, and Klaus-Robert Müller, editors, Advances in Neural Information Processing Systems 12, [NIPS Conference, Denver, Colorado, USA, November 29 - December 4, 1999], pages 421–427. The MIT Press, 1999. URL http://papers.nips.cc/ paper/1699-transductive-inference-for-estimating-values-of-functions.

Olivier Chapelle, Bernhard Schölkopf, and Alexander Zien, editors. *Semi-Supervised Learning*. The MIT Press, 2006. ISBN 9780262033589. doi:10.7551/MITPRESS/9780262033589.001.0001.

URL https://doi.org/10.7551/mitpress/9780262033589.001.0001.

Malte Darnstädt, Hans Ulrich Simon, and Balázs Szörényi. Unlabeled data does provably help.

In Natacha Portier and Thomas Wilke, editors, 30th International Symposium on Theoretical Aspects of Computer Science, STACS 2013, February 27 - March 2, 2013, Kiel, Germany, volume 20 of *LIPIcs*, pages 185–196. Schloss Dagstuhl - Leibniz-Zentrum für Informatik, 2013. doi:10.4230/LIPICS.STACS.2013.185. URL https://doi.org/10.4230/LIPIcs. STACS.2013.185.

Alexander Gammerman, Volodya Vovk, and Vladimir N. Vapnik. Learning by transduction.

In Gregory F. Cooper and Serafín Moral, editors, UAI 1998: Proceedings of the Fourteenth Conference on Uncertainty in Artificial Intelligence, University of Wisconsin Business School, Madison, Wisconsin, USA, July 24-26, 1998, pages 148–155. Morgan Kaufmann, 1998. URL https://dslpitt.org/uai/displayArticleDetails.jsp?mmnu=1&smnu=2&
article_id=243&proceeding_id=14.

Christina Göpfert, Shai Ben-David, Olivier Bousquet, Sylvain Gelly, Ilya O. Tolstikhin, and Ruth Urner. When can unlabeled data improve the learning rate? In Alina Beygelzimer and Daniel Hsu, editors, *Conference on Learning Theory, COLT 2019, 25-28 June 2019, Phoenix, AZ, USA*, volume 99 of *Proceedings of Machine Learning Research*, pages 1500–1518. PMLR, 2019. URL http://proceedings.mlr.press/v99/gopfert19a.html.

Steve Hanneke, Shay Moran, and Jonathan Shafer. A trichotomy for transductive online learning. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/
3e32af2df2cd13dfbcbe6e8d38111068-Abstract-Conference.html.

Steve Hanneke, Vinod Raman, Amirreza Shaeiri, and Unique Subedi. Multiclass transductive online learning. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. URL http://papers.nips.cc/paper_files/ paper/2024/hash/6f244818d72b2a4be9b1225d1344e950-Abstract-Conference.html.

Steven C. H. Hoi, Doyen Sahoo, Jing Lu, and Peilin Zhao. Online learning: A comprehensive survey. *Neurocomputing*, 459:249–289, 2021. doi:10.1016/J.NEUCOM.2021.04.112. URL
https://doi.org/10.1016/j.neucom.2021.04.112.

Thorsten Joachims. Transductive inference for text classification using support vector machines. In Ivan Bratko and Saso Dzeroski, editors, *Proceedings of the Sixteenth International Conference* on Machine Learning (ICML 1999), Bled, Slovenia, June 27 - 30, 1999, pages 200–209. Morgan Kaufmann, 1999.

Sham M. Kakade and Adam Kalai. From batch to transductive online learning. In Advances in Neural Information Processing Systems 18 [Neural Information Processing Systems, NIPS 2005, December 5-8, 2005, Vancouver, British Columbia, Canada], pages 611–618, 2005. URL https://proceedings.neurips.cc/paper/2005/hash/ 17693c91d9204b7a7646284bb3adb603-Abstract.html.

Nick Littlestone. Learning quickly when irrelevant attributes abound: A new linear-threshold algorithm. *Mach. Learn.*, 2(4):285–318, 1987. doi:10.1007/BF00116827. URL https://doi.

org/10.1007/BF00116827.

Shai Shalev-Shwartz and Shai Ben-David. Understanding Machine Learning: From Theory to Algorithms. Cambridge University Press, 2014.

ISBN 978-1-10-705713-5. URL http://www.cambridge.org/de/academic/
subjects/computer-science/pattern-recognition-and-machine-learning/ understanding-machine-learning-theory-algorithms.

Vladimir N. Vapnik. *Estimation of Dependencies Based on Empirical Data*. Nauka, Moscow, 1979.

URL https://www.ipu.ru/node/63854/publications. In Russian.

Vladimir N. Vapnik. *Estimation of Dependences Based on Empirical Data*. Springer, 2nd edition, 2006. ISBN 978-0-387-30865-4. doi:10.1007/0-387-34239-7. URL https://doi.org/10. 1007/0-387-34239-7.

Xiaojin Zhu. Semi-supervised learning literature survey. Technical report, Department of Computer Sciences, University of Wisconsin–Madison, 2005.

Xiaojin Zhu. Semi-supervised learning. In Claude Sammut and Geoffrey I. Webb, editors, Encyclopedia of Machine Learning, pages 892–897. Springer, 2010. doi:10.1007/978-0-387-30164-8_749. URL https://doi.org/10.1007/978-0-387-30164-8_749.

Xiaojin Zhu and Andrew B. Goldberg. *Introduction to Semi-Supervised Learning*. Synthesis Lectures on Artificial Intelligence and Machine Learning. Morgan & Claypool Publishers, 2009. ISBN 978-3-031-00420-9. doi:10.2200/S00196ED1V01Y200906AIM006. URL https://doi.org/ 10.2200/S00196ED1V01Y200906AIM006.

## Technical Appendices And Supplementary Material A **Preliminaries**

A.1 **Basic Notation**
Notation A.1. N = {1, 2, 3, . . .}*, i.e.,* 0 ∈/ N. log(·) and ln(·) denote logarithm to base 2 and e, respectively.

Notation A.2 (Sequences). Let X be a set and n, k ∈ N*. For a sequence* x = (x1, . . . , xn) ∈ X n, we write x≤k to denote the subsequence (x1, . . . , xk). If k ≤ 0 then x≤k denotes the empty sequence, which is also denoted by λ = X
0*. We use the notation* X
≤n = ∪
n i=0X
i.

## A.2 **Standard Online Learning**

Let X be a set, and let *H ⊆ {*0, 1}
X be a collection of functions called a *hypothesis class*. A learner strategy or simply *learner* for the standard online learning game (Game 1) is a function

$$L:\bigcup_{i=0}^{n-1}\left({\mathcal{X}}\times\{0,1\}\right)^{i}\times{\mathcal{X}}\rightarrow\{0,1\},$$

where n ∈ N is the number of rounds in the game. The set of all such learner strategies is denoted Ln.

An *adversary strategy* or simply *adversary* for the standard online learning game is a pair of functions

$A_{\text{instance}}:\bigcup\limits_{i=0}^{n-1}\left(\mathcal{X}\times\{0,1\}\times\{0,1\}\right)^{i}\rightarrow\mathcal{X},\text{and}$  $A_{\text{label}}:\bigcup\limits_{i=1}^{n-1}\left(\mathcal{X}\times\{0,1\}\times\{0,1\}\right)^{i}\times\{0,1\}\rightarrow\{0,1\}.$
The set of all such adversary strategies is denoted An.

Semantically, the interpretation of these strategies is that in each round t ∈ [n] of Game 1, the adversary selects an instance xt = Ainstance(x1, yˆ1, y1, . . . , xt−1, yˆt−1, yt−1) ∈ X ,
then the learner makes a prediction yˆt = L(x1, y1, . . . , xt−1, yt−1, xt) ∈ {0, 1},
and finally, the adversary assigns a label yt = Alabel(x1, yˆ1, y1*, . . . , x*t−1, yˆt−1, yt−1, yˆt) ∈ {0, 1}.

The adversary's function Alabel must satisfy *realizability*, meaning that there exists h ∈ H such that
∀t ∈ [n] : yt = h(xt).

The number of mistakes in a game with n rounds and hypothesis class H between learner L and adversary A is Mstd(H*, n, L, A*) = |{t ∈ [n] : ˆyt ̸= yt}|.

## A.3 **Transductive Online Learning**

Given X and H as in Section A.2, a learner strategy for the *transductive online learning setting*
(Game 2) is a function

$$L,A)=|\{t\in[n]$$
$$L:\,{\mathcal{X}}^{n}\times\bigcup_{i=0}^{n-1}\,\{0,1\}^{i}\to\{0,1\},$$
$\downarrow$ . 
i=0
where n ∈ N is the number of rounds in the game. An adversary strategy consists of a sequence x ∈ X n and an *adversary labeling strategy*, which is a function

$$A:$$
$$\left(\bigcup_{i=0}^{n-1}\left\{0,1\right\}^{2i}\right)\times\{0,1\}\rightarrow\{0,1\}.$$

The sets of all such learner and adversary strategies are denoted Ln and An respectively.

Semantically, the interpretation of these strategies is that at the start of Game 2, the adversary selects the sequence x. Then, in each round t ∈ [n], the learner makes a prediction yˆt = L(x, y1, . . . , yt−1) ∈ {0, 1},
and then the adversary assigns a label yt = A(ˆy1, y1*, . . . ,* yˆt−1, yt−1, yˆt) ∈ {0, 1}.

Exactly as in Section A.2, the adversary's function A must satisfy realizability, namely,
∃h *∈ H ∀*t ∈ [n] : yt = h(xt),
and the number of mistakes in a game with sequence length n and hypothesis class H between learner L and adversary A is Mtr(H*, n, L, A*) = |{t ∈ [n*] : ˆ*yt ̸= yt}|.

## A.4 **Mistake Bounds**

In this paper, we study *optimal mistake bounds*, or the *optimal number of mistakes*, which is the value of Games 1 and 2. For M ∈ {Mstd, Mtr}, the optimal number of mistakes in a game with hypothesis class H and sequence length n is,

$$M({\mathcal{H}},n)=\operatorname*{inf}_{L\in{\mathcal{L}}_{n}}\operatorname*{sup}_{A\in{\mathcal{A}}_{n}}M({\mathcal{H}},n,L,A).$$

The optimal number of mistakes for hypothesis class H is

$$M({\mathcal{H}})=\operatorname*{sup}_{n\in\mathbb{N}}\,M({\mathcal{H}},n).$$

Remark A.3. As is common in learning theory literature, in both Game 1 and Game 2, we take the sets Ln and An to be the sets of all (deterministic) functions. In this paper, we do not consider randomized strategies. By allowing arbitrary functions, we ignore issues relating to computability.

## A.5 **Trees**

Definition A.4 (Notation for binary trees). Let d ∈ N ∪ {0}. A perfect binary tree of depth d is a collection of 2 d+1 − 1 *nodes, which we identify with the collection of binary strings*

$T_d=\left\{\left\{0\right\}\right\}$
Td ={0, 1}
k: k ∈ {0, 1, 2*, . . . , d*}	.

The empty string, denoted λ = {0, 1}
0, is a member of Td and is called the root of the tree. Every string u ∈ {0, 1}
dis called a leaf. The depth of a node u ∈ Td, denoted |u|, is the length of u as a string, namely, the integer k such that u ∈ {0, 1}
k.

For two nodes u, v ∈ Td, we say that u is a parent of v, and that v is a child of u*, if* v = u ◦ 0 or v = u ◦ 1, where ◦ denotes string concatenation. More fully, for b ∈ {0, 1}, we say that v is a b-child of u if v = u ◦ b.

Recursively, we define that u is an ancestor of v and that v is a descendant of u, and write u ≼ v*, if* one of the following holds:
- u = v*, or*
- ∃w ∈ Td ∃b ∈ {0, 1} : (u ≼ w) ∧ (w ◦ b = v).

For b ∈ {0, 1}, we say that v is a b-descendant of u, denoted u ≼b v, if v is a descendant of the b*-child of* u.

A function f : Td → {0, 1} specifies a particular root-to-leaf path in the tree Td (see Figure 1). The on-path nodes for f are the set of d + 1 nodes on that root-to-leaf path, as in the following definition.

$=\,w$). 
Definition A.5 (Paths in a binary tree). Let d, k ∈ N, k ≤ d. Let u ∈ {0, 1}
k *be a node in* Td.

The path to u *is the unique sequence* path(u) = (u0, u1, u2, . . . , uk) such that u0 = λ *is the root,*
uk = u, and uiis a child of ui−1 *for all* i ∈ [k].

Let f : Td → {0, 1} be a function. The path of f *is the unique sequence* path(f) =
(u0, u1, u2, . . . , ud) such that u0 = λ *is the root, and for each* i ∈ [d], ui = ui−1 ◦ f(ui−1).

Namely, uiis the f(ui−1)*-child of* ui−1.

For a node v ∈ Td and a function f : Td → {0, 1}*, we write* v ∈ path(f) if path(f) = (u0*, . . . , u*d)
and there exists i ∈ {0, . . . , d} such that ui = v. Otherwise, we write v /∈ path(f).

For a node v ∈ Td and a set of functions *F ⊆ {*0, 1}
Td *, we write* v ∈ path(F) if
∀f ∈ F : v ∈ path(f).

Otherwise, we write u /∈ path(F).

A.6 **Littlestone Dimension**
Definition A.6 (Littlestone, 1987). Let X be a set, let *H ⊆ {*0, 1}
X , and let d ∈ N ∪ {0}. We say that H shatters the binary tree Td if there exists a mapping Td → X given by u 7→ xu *such that for* every u ∈ {0, 1}
d+1 there exists hu ∈ H *such that*
∀i ∈ [d + 1] : h(xu≤i−1
) = ui.

The Littlestone dimension of H, denoted LD(H), is the supremum over all d ∈ N such that there exists a Littlestone tree of depth d − 1 *that is shattered by* H.

Note that by defining the Littlestone dimension this way, every class with Littlestone dimension d ∈ N contains at least 2 dfunctions.

0 x00 0 1 x0 0 0
∃ h ∈ H :
h(xλ) = 1 h(x1) = 0 h(x10) = 1 x01 1 1 1 0 1 0 xλ x10 0 1 x1 x11 1
Theorem A.7 (Littlestone, 1987). Let X be a set and let *H ⊆ {*0, 1}
X *such that* d = LD(H) < ∞.

Then there exists a strategy for the learner that guarantees that the learner will make at most d mistakes in the standard (non-transductive) online learning setting, regardless of the adversary's strategy and of the number n *of instances to be labeled. Furthermore, there exists an adversary that* forces every learner to make at least min {n, d} *mistakes.*

## B **Lower Bound** B.1 **Statement**

Our Ω
√d lower bound states the following.

Theorem B.1 (Lower bound). There exists a constant d0 ≥ 0 as follows. Let d ∈ N, d ≥ d0, let X
be a set, and let *H ⊆ {*0, 1}
X *be a hypothesis class with* LD(H) = d. Then there exist a sequence x ∈ X n *of length* n = O
d · 2
√
dand an adversary A that always selects the sequence x and uses a simple adaptive labeling strategy (as in Algorithm 1), such that for every learning rule L,

$$({\mathfrak{I}})$$
$$\cdot,n,L,A)$$
$$(4)$$

Mtr(H*, n, L, A*) ≥
√d/10. (3)
Furthermore, for every integer n ∈ N,

$$M_{\mathrm{tr}}({\mathcal{H}},n)\geq\operatorname*{min}\,\left\{{\sqrt{d}}/10,\lfloor\log(n+1)\rfloor\right\}.$$
o. (4)
Remark B.2. The assumption LD(H) = d implies that for all k ∈ [d], H shatters a Littlestone tree of depth k. Thus, the lower bound of Eq. (3) in Theorem B.1 immediately implies that for every k ∈ [d] there exists a sequence x
(k) ∈ X nk of length nk = O
k · 2
√ksuch that the adversary Ak that presents the sequence x
(k)and assigns labels using the simple labeling strategy of Algorithm 1 ensures that for every learner L,

$$M_{\mathrm{tr}}({\mathcal{H}},n_{k},L,A_{k})\geq{\sqrt{k}}/10.$$

See Section 2.2 for a general overview of Theorem B.1 and the main proof ideas. In the following subsections we prove Theorem B.1. Algorithm 1 gives an explicit construction of the adversary that witnesses the lower bound, using Algorithm 2 as a subroutine. We start with presenting some initial observations about the behavior of these algorithms in Section B.2.

Assumptions:
- d ∈ N, ε = 2−
√
d/2.

- T = Td is a perfect binary tree of depth d.

- *H ⊆ {*0, 1}
Tis a class that shatters T.

TRANSDUCTIVEADVERSARY (H):
(x1, x2*, . . . , x*n) ← CONSTRUCTSEQUENCE (H) ▷ See Algorithm 2.

send (x1, x2*, . . . , x*n) to learner H0 ← H
for t ∈ [n]:
receive yˆt from learner rt ←
|{h ∈ Ht−1 : h(xt) = 1}| |Ht−1| ymaj ← 1(rt ≥ 1/2)
yt ←
ymaj rt ∈/ [ε, 1 − ε]
1 − yˆt otherwise send yt to learner Ht ← {h ∈ Ht−1 : h(xt) = yt}
Algorithm 1: The strategy for the adversary that achieves the lower bound in Theorem B.1. Note that while the construction of the sequence x is not entirely trivial, the adversary's strategy for labeling this sequence is very simple.

## B.2 **Analysis Of The Adversary**

Claim B.3. Let d ∈ N*, let* M =
√d/10, and let *H ⊆ {*0, 1}
Td *be a hypothesis class. Consider an* execution of CONSTRUCTSEQUENCE (H) as in Algorithm 2 that produces a sequence x1, x2*, . . . , x*n.

Then:
Assumptions:
- d ∈ N, M =
√d/10, ε = 2−
√d/2.

- T = Td is a perfect binary tree of depth d.

- λ, the empty string, is the root of T.

- *H ⊆ {*0, 1}
Tis a class that shatters T.

CONSTRUCTSEQUENCE (H):
r ←
|{h ∈ Hb : h(xt) = 1}| |Hb| return (x1, x2*, . . . , x*t)
Algorithm 2: A subroutine of Algorithm 1 for selecting the sequence x.

(a) *For all* i ∈ [n], path(xi) is a subsequence of x0, x1, . . . , xi.

(b) The length n of the sequence satisfies n < nd*, where* nd = (d + 1) · 2M+1.

Proof.

(a) Fix i ∈ [n]. It suffices to show that for all u ∈ Td, if u ≼ xithen u ∈ (x1, x2*, . . . , x*i).

Proceed by induction on i. For the base case i = 1, the claim holds because x1 = λ.

For the induction step, assume the claim holds for i ∈ [n − 1]. Let u ≼ xi+1, we prove that u ∈ (x1, x2*, . . . , x*i+1). Assume xi+1 ̸= λ (otherwise, there is nothing to prove).

Hλ ← H
H0 *← {H*λ} ▷ A set of classes indexed by bit strings.

Q ← {λ} ▷ A set of nodes to be processed. t ← 0 while |Q| > 0:
t ← t + 1 xt ← arbitrary element from Q ▷ Pop an arbitrary element from Q
Q ← Q \ {x and add it to the output sequence. t}
Ht ← ∅ for Hb ∈ Ht−1:
Y ← 
{0, 1}r ∈ [ε, 1 − ε]∧|b| < M
{1(r ≥ 1/2)} otherwise ▷ Adversary will force mistakes on the first M balanced nodes.

for y ∈ Y:
b
′ ←
b |Y| = 1 b ◦ y |Y| = 2 ▷ Restrict class to agree with y. If splitting the class in two to force a mis-
Hb take then create new indices. ′ ← {h ∈ Hb : h(xt) = y}
Ht ← Ht *∪ {H*b
′}
if xt ∈ path(Hb
′ ) ∧ |xt| < d: ▷ If xt is on-path for Hb
′ and it has a Q ← Q ∪ {x y-child, add that child to Q. t ◦ y}
Because xi+1 appears in the sequence x, it must have been added to Q before it was added to x. The only place where items that are not λ are added to Q is in the line *Q ← Q ∪ {*xt ◦ y}. Namely, there exist an index j ∈ [i] and a bit y ∈ {0, 1} such that xi+1 = xj ◦ y (note that *j < i* + 1 because xj was added to the sequence before xi+1). If xj = u we are done. Otherwise, note that xj is the parent of xi+1, and therefore u ≼ xj . By the induction hypothesis, u ∈ (x1, x2*, . . . , x*j ). This concludes the proof.

(b) Items are added to the sequence x only if they were previously added to Q. By induction on i ∈ [n], for each xiin the sequence, there is at most one iteration of the "while |Q| > 0" loop in which xiis added to Q. The base case i = 1 holds because x1 = λ is the root, which is added to Q before the while loop, and λ is never added to Q within that loop because the line "*Q ← Q ∪ {*xt ◦ y}" can only add non-empty bit strings. For the induction step, if the claim holds for all natural numbers j such that 1 ≤ *j < i* ≤ n then it holds for i. Indeed, for i ≥ 2, xi can be added to Q only via the line "*Q ← Q ∪ {*xt ◦ y}", and only in the iteration of the while loop where xt is the parent of xiin the tree Td. In that iteration, the parent xt of xiis popped from Q, which implies that xt was added to Q in some previous iteration of the while loop (*t < i*), and is no longer in Q after being popped. By the induction hypothesis, xt will never be added to Q again, and therefore in all subsequent iterations of the while loop xt will not be the parent of xi, so xi cannot be added to Q in subsequent iterations via the line "*Q ← Q ∪ {*xt ◦ y}".

Furthermore, if a node xiis added to Q in some iteration of the while loop, then it remains in Q for the duration of that iteration. So for all i ∈ {2, 3*, . . . , n*}, there is precisely one execution of the line "*Q ← Q ∪ {*xt ◦ y}" that adds xito Q. Namely, there is precisely one point in time during the execution of Algorithm 2 in which xi = xt ◦ y, xi ∈/ Q, and the line "*Q ← Q ∪ {*xt ◦ y}" is executed resulting in xi ∈ Q.

Consider a function f that maps i ∈ {2, 3*, . . . , n*} to the value of the index b
′ during the unique execution of the line "*Q ← Q ∪ {*xt ◦ y}" that adds xito Q. Namely, if b
′ had some value β when xi was added to Q, then f(i) = β.

Notice that "*Q ← Q∪{*xt ◦y}" is executed only if the condition xt ∈ path(Hb
′ ) is satisfied in the previous line. Furthermore, the line "Hb
′ ← {h ∈ Hb : h(xt) = y}" ensures that the node xi = xt ◦ y being added to Q satisfies xt ◦ y ∈ path(Hb
′ ), namely
∀h ∈ Hb
′ : xi ∈ path(h).

Consequently, xi ∈ path(G) for any class G that is a subset of Hb
′ ; in particular, because the only way that Hb
′ might be modified later during the execution of Algorithm 2 is by removing elements, it follows that xi ∈ path(Hb
′ ) when the line "*Q ← Q ∪ {*xt ◦ y}" is executed and in all subsequent times.

However, |path(G)| = d + 1 for any class *G ⊆ {*0, 1}
Td . This implies that f maps at most
(d + 1) nodes to each bit string. In other words, for any bit string b, the size of the preimage satisfies |f
−1(b)| ≤ d + 1.

The condition "|b| < M" in Algorithm 2 ensures that |b
′| ≤ M, namely, b
′ ∈ {0, 1}
kfor k ∈ {0, 1, 2*, . . . , M*}. Thus,

$n=1+|\{2,3,\ldots,n\}|$  $=1+\sum\limits_{\begin{subarray}{c}b\in\{0,1\}^{k}\\ k\in\{0,\ldots,M\}\end{subarray}}|\{i\in\{2,3,\ldots,n\}:\ f(i)=b\}|$
$$=1+\sum_{b\in\{0,1\}^{k}}^{-}|f^{-1}(b)|$$
k∈{0*,...,M*}
≤ 1 + X b∈{0,1} k k∈{0,...,M} (d + 1) ≤ 1 + (d + 1) · (2M+1 − 1). < (d + 1) · 2 M+1,
Claim B.4. Let d ∈ N*, let* M =
√d/10, and let *H ⊆ {*0, 1}
Td be a hypothesis class. Consider an execution of TRANSDUCTIVEADVERSARY (H) as in Algorithm *1. Let* H0, H1*, . . . ,* Hn be the sequence of hypothesis classes created by TRANSDUCTIVEADVERSARY*, let* S =t ∈ [n] : rt ∈ [ε, 1 − ε]	
be the set of indices where TRANSDUCTIVEADVERSARY *forces a mistake, and let* H0, H1*, . . . ,* Hn be the sequence of collections created by the subroutine CONSTRUCTSEQUENCE (Algorithm *2). If* |S| ≤ M *then*
∀t ∈ {0, 1*, . . . , n*} : Ht ∈ Ht.

Proof. Proceed by induction on t ∈ {0, 1*, . . . , n*}. The base case t = 0 is satisfied, because H0 = *H ∈ {H}* = H0. For the induction step, assume that Hi−1 ∈ Hi−1 for some i ∈ [n]. We prove that Hi ∈ Hi.

Let yi be the label assigned to xi by TRANSDUCTIVEADVERSARY. Then Hi = {h ∈ Hi−1 : h(xi) = yi}.

Consider the iteration of the while loop in CONSTRUCTSEQUENCE that starts with t ← i. By the induction hypothesis, Hi−1 ∈ Hi−1. Therefore, in this iteration of the while loop, there will be an iteration of the "for Hb ∈ Ht−1" loop where Hb = Hi−1. In that iteration, yi ∈ Y by construction of yi and Y. Therefore, in the iteration of the "for y ∈ Y" loop in which y = yi, Hb
′ = {h ∈ Hb : h(xt) = y} = {h ∈ Hi−1 : h(xi) = yi} = Hi.

The class Hb
′ is then added to Hi = Ht in the line "Ht ← Ht *∪ {H*b
′}". Furthermore, no class is ever removed from Ht. So Hi ∈ Hi, as desired.

Claim B.5. Let d ∈ N*, let* M =
√d/10, and let *H ⊆ {*0, 1}
Td be a hypothesis class. Consider an execution of TRANSDUCTIVEADVERSARY (H) as in Algorithm 1 where the adversary constructs a sequence of nodes x1, x2, . . . , xn ∈ Td and a sequence of classes H0, H1, . . . , Hn ⊆ {0, 1}
Td *. Let* S =t ∈ [n] : rt ∈ [ε, 1 − ε]	
be the set of indices where TRANSDUCTIVEADVERSARY forces a mistake, and assume that |S| ≤ M.

Then for all k ∈ {0, 1, . . . , d} there exists i ∈ [n] such that 1. |xi| = k*, and* 2. xi ∈ path(Hi−1),
Proof. Proceed by induction on k. For the base case k = 0, notice that x1 = λ, |λ| = 0, and λ ∈ path(H−1). For the induction step, assume the claim holds for some k ∈ {0, 1*, . . . , d* − 1}, and take ik ∈ [n] such that |xik| = k and xik ∈ path(Hik−1); we prove that the claim holds for k + 1 as well.

Consider the iteration of the while loop in CONSTRUCTSEQUENCE in which xikis added to the sequence (i.e., the iteration starting with t ← ik). By Claim B.4 and the assumption |S| ≤ M,
Hik−1 ∈ Hik−1. Hence, within this iteration of the while loop, there is an iteration of the "for Hb ∈ Ht−1" loop such that Hb = Hik−1. By construction, the set Y always contains the label predicted by the adversary, so yik ∈ Y. Consider the iteration of the "for y ∈ Y" loop such that y = yik. By the induction hypothesis, xi ∈ path(Hik−1), and since Hb
′ ⊆ Hb = Hik−1, it follows that xik ∈ path(Hb
′ ). Seeing as |xik| < d, in the last line of this iteration of the "for y ∈ Y" loop, the node xik+1 := xik◦ yik is added to Q. This guarantees that xik+1 will eventually be popped from