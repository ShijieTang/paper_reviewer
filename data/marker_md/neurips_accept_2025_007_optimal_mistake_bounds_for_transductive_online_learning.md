# Optimal Mistake Bounds for Transductive Online Learning

Zachary Chase Kent State University zchase2@kent.edu

Steve Hanneke Purdue University steve.hanneke@gmail.com

Shay Moran Departments of Mathematics, Computer Science, and Data and Decision Sciences Technion – Israel Institute of Technology; Google Research smoran@technion.ac.il

Jonathan Shafer MIT shaferjo@mit.edu

## Abstract

We resolve a 30-year-old open problem concerning the power of unlabeled data in online learning by tightly quantifying the gap between transductive and standard online learning. In the standard setting, the optimal mistake bound is characterized by the Littlestone dimension <sup>d</sup> of the concept class H [\(Littlestone,](#page-11-0) [1987\)](#page-11-0). We prove that in the transductive setting, the mistake bound is at least Ω √ d . This constitutes an exponential improvement over previous lower bounds of Ω(log log(d)), Ω p log(d) , and Ω(log(d)), due respectively to [Ben-David, Kushilevitz, and](#page-10-0) [Mansour](#page-10-0) [\(1995,](#page-10-0) [1997\)](#page-10-1), and [Hanneke, Moran, and Shafer](#page-11-1) [\(2023\)](#page-11-1). We also show that this lower bound is tight: for every d, there exists a class of Littlestone dimension d with transductive mistake bound O √ d . Our upper bound also improves upon the best known upper bound of (2/3) · <sup>d</sup> from [Ben-David et al.](#page-10-1) [\(1997\)](#page-10-1). These results establish a quadratic gap between transductive and standard online learning, thereby highlighting the benefit of advance access to the unlabeled instance sequence. This contrasts with the PAC setting, where transductive and standard learning exhibit similar sample complexities.

## 1 Introduction

The transductive model is a basic and well-studied framework in learning theory, dating back to the early works of Vapnik. It has been investigated both in statistical and online settings, and is motivated by the principle that to make good predictions on a specific set of test instances, one need not construct a fully general classifier that performs well on the entire domain — including points that may never actually appear. Rather, it may be sufficient to tailor predictions for a fixed, known set of instances.

This perspective naturally connects to a broader question in learning theory: what is the value of unlabeled data? In the transductive setting, the learner is given the sequence of unlabeled test instances in advance and is then required to predict their labels one by one. Thus, the transductive model can be viewed as a natural formalization of learning with unlabeled data: the test instances are known in advance, but their labels are not. The central question is whether such prior access to the

unlabeled sequence can help reduce the number of prediction mistakes — compared to the standard online model, where the instances arrive and are labeled one at a time.

Recall for instance that in the standard PAC[<sup>1</sup>](#page-1-0) model of supervised learning, there are cases where access to unlabeled data is not helpful. Indeed, the "hard population distributions" used to prove the standard VC[<sup>2</sup>](#page-1-1) lower bound are constructed by taking a fixed and known marginal distribution over a VC-shattered set. Namely, the cases that are hardest to learn in the PAC setting include ones where the learner knows the marginal distribution over the domain, and can therefore generate as much unlabeled data as it wishes. And yet, in those cases, access to unlabeled data provides no acceleration compared to an algorithm (like ERM[<sup>3</sup>](#page-1-2) ) that does not use unlabeled data.

Seeing as unlabeled data is often a lot easier to obtain than labeled data, there have been considerable efforts to understand when and to what extent can access to unlabeled data accelerate learning.[<sup>4</sup>](#page-1-3)

In particular, it is natural to ask, for which plausible models of learning is access to unlabeled data beneficial? Online learning [\(Littlestone,](#page-11-0) [1987\)](#page-11-0) is perhaps the model of learning that is mostextensively studied in learning theory after the PAC model and its variants. Therefore, the general question considered in this paper is:

Question 1. *Quantitatively, how much (if at all) is access to unlabeled data beneficial for learning in the online learning setting?*

This question is naturally instantiated by comparing *transductive* online learning — where the learner has advance access to the full sequence x1, x2, . . . , x<sup>n</sup> of unlabeled instances — with *standard* online learning, where no such access is given. This perspective has also been adopted in prior work: for example, [Kakade and Kalai](#page-11-2) [\(2005\)](#page-11-2), [Cesa-Bianchi and Shamir](#page-10-2) [\(2013\)](#page-10-2), and [Hoi, Sahoo, Lu, and Zhao](#page-11-3) [\(2021\)](#page-11-3) (Section 7.3) all describe transductive online learning as a setting in which the learner has access to "unlabeled data". We thus refine the question above as follows:

Question 2. *Quantitatively, how much (if at all) is learning in the transductive online learning setting easier than learning in the standard online learning setting? Specifically, how much is the optimal number of mistakes in the transductive setting smaller than in the standard setting?*

Addressing this question, our main result (Theorem [1.1\)](#page-2-0) states that the optimal number of mistakes in the transductive setting (with access to unlabeled data) is at most quadratically smaller than in the standard setting (without unlabeled data). Furthermore, there are hypothesis classes for which a quadratic gap is achieved.

#### 1.1 Setting: *Standard* vs. *Transductive* Online Learning

*Standard online learning* [\(Littlestone,](#page-11-0) [1987\)](#page-11-0) is a zero-sum, perfect- and complete-information game played over n rounds between two players, a *learner* and an *adversary*. The game is played with respect to a *domain* set X and a *hypothesis class* H ⊆ {0, <sup>1</sup>} <sup>X</sup> (consisting of functions X → {0, <sup>1</sup>}), where <sup>n</sup>, X and H are fixed and known to both players. The game proceeds as in Game [1.](#page-2-1) The *number of mistakes* for a learner <sup>L</sup> and an adversary <sup>A</sup> is <sup>M</sup>std(H, n, L, A) = |{<sup>t</sup> ∈ [n] : ˆy<sup>t</sup> ̸<sup>=</sup> <sup>y</sup>t}|. We are interested in understanding the *optimal number of mistakes*, which is

$$M_{\text{std}}(\mathcal{H}) = \sup_{n \in \mathbb{N}} \inf_{L \in \mathcal{L}} \sup_{A \in \mathcal{A}} M_{\text{std}}(\mathcal{H}, n, L, A),$$

where A and L are the set of all deterministic adversaries and learners, respectively.[<sup>5</sup>](#page-1-4)

<sup>1</sup> Probably Approximately Correct. For an exposition of the standard terminology and results mentioned in this paragraph see, e.g., [Shalev-Shwartz and Ben-David](#page-11-4) [\(2014\)](#page-11-4).

<sup>2</sup>Vapnik–Chervonenkis.

<sup>3</sup>Empirical Risk Minimization.

<sup>4</sup>The literature on semi-supervised learning is surveyed in [Joachims](#page-11-5) [\(1999\)](#page-11-5); [Zhu](#page-11-6) [\(2005\)](#page-11-6); [Zhu and Goldberg](#page-12-0) [\(2009\)](#page-12-0); [Zhu](#page-12-1) [\(2010\)](#page-12-1); [Chapelle, Schölkopf, and Zien](#page-10-3) [\(2006\)](#page-10-3). Theoretical works on the topic include [Benedek](#page-10-4) [and Itai](#page-10-4) [\(1991\)](#page-10-4); [Blum and Mitchell](#page-10-5) [\(1998\)](#page-10-5); [Ben-David, Lu, Pál, and Sotáková](#page-10-6) [\(2008\)](#page-10-6); [Balcan and Blum](#page-10-7) [\(2010\)](#page-10-7); [Darnstädt, Simon, and Szörényi](#page-10-8) [\(2013\)](#page-10-8); [Göpfert, Ben-David, Bousquet, Gelly, Tolstikhin, and Urner](#page-11-7) [\(2019\)](#page-11-7).

<sup>5</sup>Because the adversary selects y<sup>t</sup> *after* seeing yˆt, randomness is not beneficial for either party, and we assume without loss of generality that both the learner and the adversary are deterministic. As is common in learning theory, we avoid questions of computability and allow the learner and adversary to be any function. See Section [A](#page-13-0) for formal definitions of A and L.

For each round t = 1, 2, . . . , n:

- *<sup>a</sup>*. The adversary selects an *instance* <sup>x</sup><sup>t</sup> ∈ X and sends it to the learner.
- *<sup>b</sup>*. The learner selects a *prediction* <sup>y</sup>ˆ<sup>t</sup> ∈ {0, <sup>1</sup>} and sends it to the adversary.
- *<sup>c</sup>*. The adversary selects a *label* <sup>y</sup><sup>t</sup> ∈ {0, <sup>1</sup>} and sends it to the learner. The selected label must be *realizable*, meaning that ∃<sup>h</sup> ∈ H ∀<sup>i</sup> ∈ [t]: <sup>h</sup>(xi) = <sup>y</sup><sup>i</sup> .

Game 1: The standard online learning setting.

The adversary selects a *sequence* <sup>x</sup>1, x2, . . . , x<sup>n</sup> ∈ X and sends it to the learner.

For each round t = 1, 2, . . . , n:

- *<sup>a</sup>*. The learner selects a *prediction* <sup>y</sup>ˆ<sup>t</sup> ∈ {0, <sup>1</sup>} and sends it to the adversary.
- *<sup>b</sup>*. The adversary selects a *label* <sup>y</sup><sup>t</sup> ∈ {0, <sup>1</sup>} and sends it to the learner. The selected label must be *realizable*, meaning that ∃<sup>h</sup> ∈ H ∀<sup>i</sup> ∈ [t]: <sup>h</sup>(xi) = <sup>y</sup><sup>i</sup> .

Game 2: The transductive online learning setting.

It is well known that <sup>M</sup>std(H) is characterized by the the Littlestone dimension, namely, <sup>M</sup>std(H) = LD(H) (see Theorem [A.7](#page-15-0) and Definition [A.6\)](#page-15-1).

The *transductive* online learning setting [\(Ben-David et al.,](#page-10-0) [1995,](#page-10-0) [1997\)](#page-10-1) is similar, except that the learner has access to the full sequence of unlabeled instances in advance. Namely, as in Game [2.](#page-2-2) The optimal number of mistakes for the transductive setting is defined exactly as before,

$$M_{\text{tr}}(\mathcal{H}, n, L, A) = |\{t \in [n] : \hat{y}_t \neq y_t\}|, \quad \text{and} \quad M_{\text{tr}}(\mathcal{H}) = \sup_{n \in \mathbb{N}} \inf_{L \in \mathcal{L}} \sup_{A \in \mathcal{A}} M_{\text{tr}}(\mathcal{H}, n, L, A),$$

with the only difference between the standard quantity <sup>M</sup>std(H) and the transductive quantity <sup>M</sup>tr(H) being in how the game is defined.

#### 1.2 Main Result

Notice that for every hypothesis class H, <sup>M</sup>tr(H) ≤ <sup>M</sup>std(H). Indeed, in the transductive setting the adversary declares the sequence x at the start of the game. This reduces the number of mistakes because the transductive adversary is less powerful (it cannot adaptively alter the sequence mid-game), and also because the transductive learner is more powerful (it has more information).[<sup>6</sup>](#page-2-3)

While for some classes <sup>M</sup>tr(H) = <sup>M</sup>std(H), we study the largest possible separation. The best previous lower bound on Mtr, due to [Hanneke, Moran, and Shafer](#page-11-1) [\(2023\)](#page-11-1), states that for every class H,

$$M_{\text{tr}}(\mathcal{H}) \geq \Omega(\log(d)),$$

where <sup>d</sup> <sup>=</sup> <sup>M</sup>std(H). In the other direction, [Ben-David et al.](#page-10-1) [\(1997\)](#page-10-1) constructed[<sup>7</sup>](#page-2-4) a class H such that <sup>M</sup>std(H) = <sup>d</sup> and <sup>M</sup>tr(H) ≤ 3 d. This left an exponential gap between the best known lower and upper bounds on Mtr, namely Ω(log d) versus <sup>2</sup> 3 d. Our main result closes this gap:

Theorem 1.1 (Main result).

- *For every hypothesis class* H ⊆ {0, <sup>1</sup>} X *,*

$$M_{\text{tr}}(\mathcal{H}) = \Omega\left(\sqrt{d}\right),$$

*where* <sup>d</sup> <sup>=</sup> <sup>M</sup>std(H).

<sup>6</sup>One could also define an intermediate setting, where the adversary is less powerful because it must select the sequence at the start of the game and cannot change it during the gameplay, but the learner does not have more information because the adversary only reveals the instances in the sequence one at a time as in the standard setting. However, this intermediate setting would not model the learner having *access* to unlabeled data.

<sup>7</sup>Their class consists of all disjoint unions of Θ(d) functions from a specific constant-sized class.

- *On the other hand, for every* <sup>d</sup> *there exists a hypothesis class* H *with* <sup>M</sup>std(H) = <sup>d</sup> *and*

$$M_{\text{tr}}(\mathcal{H}) = O\left(\sqrt{d}\right).$$

This result is stated in considerably greater detail in Theorems [B.1](#page-2-0) and [D.1.](#page-2-0)

#### 1.3 Related Works

The notion of *transductive inference* as a more efficient alternative to *inductive inference* in statistical learning theory was introduced by [Vapnik](#page-11-8) [\(1979,](#page-11-8) [2006\)](#page-11-9); [Gammerman, Vovk, and Vapnik](#page-11-10) [\(1998\)](#page-11-10); [Chapelle, Vapnik, and Weston](#page-10-9) [\(1999\)](#page-10-9). The *online learning* setting is due to [Littlestone](#page-11-0) [\(1987\)](#page-11-0), who also proved that the optimal number of mistakes is characterized by the Littlestone dimension (see Theorem [A.7\)](#page-15-0).

The *transductive online learning* setting studied in the current paper, was first defined by [Ben-David,](#page-10-0) [Kushilevitz, and Mansour](#page-10-0) [\(1995\)](#page-10-0), who used the name *worst sequence off-line model*. Among other results, they showed a lower bound of Ω(log log(d)) on the number of mistakes required to learn a class with Littlestone dimension d. The authors subsequently presented an exponentially stronger lower bound of Ω p log(d) in [Ben-David, Kushilevitz, and Mansour](#page-10-1) [\(1997\)](#page-10-1). However, understanding where the optimal number of mistakes is situated within the range h Ω p log(d) , 2d/3 i remained an open question.

[Kakade and Kalai](#page-11-2) [\(2005\)](#page-11-2) presented an oracle-efficient algorithm for the transductive online learning setting, and may have been the first to use that name. Their result was subsequently improved upon by [Cesa-Bianchi and Shamir](#page-10-2) [\(2013\)](#page-10-2).

The present work is most similar to that of [Hanneke, Moran, and Shafer](#page-11-1) [\(2023\)](#page-11-1) which, among other results, gave a quadratically-stronger mistake lower bound of Ω(log(d)) for classes with Littlestone dimension d in the transductive online setting. The proof of our lower bound utilizes some of their ideas, but yields a quantitative improvement by combining it with some new ideas.

[Hanneke, Raman, Shaeiri, and Subedi](#page-11-11) [\(2024\)](#page-11-11) studied a setting of *multi-class* transductive online learning where the number of possible labels is unbounded.

## 2 Technical Overview

In this section we explain some of the main ideas in our proofs. Formal definition appear in Section [A.](#page-13-0) Full formal statements of the results, as well as detailed rigorous proofs, appear in Sections [B](#page-15-2) to [D.](#page-24-0)

#### 2.1 Paths in Trees

We make extensive use of the following notion. Given a perfect binary tree T<sup>d</sup> of depth d, every function <sup>f</sup> : <sup>T</sup><sup>d</sup> → {0, <sup>1</sup>} defines a unique *path* in the tree. The path is a sequence of nodes path(f) = (x<sup>i</sup><sup>0</sup> , x<sup>i</sup><sup>1</sup> , . . . , x<sup>i</sup><sup>d</sup> ), as explained in Figure [1c.](#page-4-0) See Section [A](#page-13-0) for formal definitions.

#### 2.2 Proof Ideas for the Lower Bound

We start with an elementary observation about the adversary's dilemma in the transductive online learning setting. Before round t of the game, the adversary selected a full sequence of instances <sup>x</sup>1, x2, . . . , x<sup>n</sup> ∈ X , and assigned some initial labels <sup>y</sup>1, y2, . . . , yt−<sup>1</sup> ∈ {0, <sup>1</sup>}. At the start of round t, the adversary must consider the *version space*,

$$\mathcal{H}_t = \{h \in \mathcal{H} : (\forall i \in [t-1] : h(x_i) = y_i)\}.$$

If all <sup>h</sup> ∈ H<sup>t</sup> assign <sup>h</sup>(xt) = <sup>b</sup> for some <sup>b</sup> ∈ {0, <sup>1</sup>}, then the adversary has no choice but to assign the label y<sup>t</sup> = b. Otherwise, the adversary can *force a mistake* at time t. Namely, after seeing the learner's prediction <sup>y</sup>ˆt, the adversary can assign <sup>y</sup><sup>t</sup> = 1 − <sup>y</sup>ˆt, incrementing the number of learner mistakes by 1.

But "just because you can, doesn't mean you should". If the adversary is greedy and forces a mistake at time t, they may pay dearly for that later. As an extreme example, consider the case where there

![](_page_4_Diagram_0.jpeg)

(a) A *perfect binary tree* of depth 2. Each *node* is labeled by an element of the domain X . These labels need not be distinct (e.g., it is possible that x<sup>1</sup> = x6). x<sup>0</sup> is the *root* of the tree, x0, x<sup>1</sup> and x<sup>2</sup> are *internal nodes*, and x3, . . . , x<sup>6</sup> are *leaves*.

![](_page_4_Diagram_1.jpeg)

(b) A function f : X → {0, 1} assigns a binary label to each node in the tree, represented here by edges with arrowhead tips. This figure depicts the function f(xi) = <sup>1</sup>(i /∈ {2, 3}). (Note that the gray dots (•) in the figure are purely a pictorial detail. In this paper they are not considered nodes or leaves of the tree.)

![](_page_4_Diagram_4.jpeg)

(c) Every function f : X → {0, 1} defines a *path* in the tree, which is a sequence u0, u1, u2, . . . , ud−1, where u<sup>0</sup> is the root, d is the depth of the tree, and for each i ∈ [d − 1], u<sup>i</sup> is the b-child of ui−<sup>1</sup> with b = f(ui−1) ∈ {0, 1}. This figure shows that the function f from Figure [1b](#page-4-0) has path(f) = (x0, x2, x5), depicted in red. In particular, x<sup>2</sup> is 'on-path' for f, but x<sup>6</sup> is 'off-path' for f.

![](_page_4_Diagram_5.jpeg)

(d) In this paper we use a naming convention where, without loss of generality, we identify the domain elements x<sup>i</sup> that are assigned to nodes with bit strings. The root is identified with the empty string λ, and for each pair of nodes u, v such that u is the b-child of v (for b ∈ {0, 1}), we have u = v ◦ b, where '◦' denotes string concatenation. (Because the xi's may not be distinct, a domain element may be identified with more than one bit string.)

Figure 1: Paths in trees.

is a single <sup>h</sup><sup>1</sup> ∈ H<sup>t</sup> that assigns <sup>h</sup>1(xt) = 1, and all other functions <sup>h</sup> ∈ H<sup>t</sup> assign <sup>h</sup>(xt) = 0. If the learner selects yˆ<sup>t</sup> = 1 and the adversary forces a mistake at time t, the version space at all subsequent times s > t will be H<sup>s</sup> <sup>=</sup> {<sup>h</sup>1}, and the adversary will be prevented from forcing any further mistakes.

A natural strategy for the adversary is therefore to be greedy up to a certain limit. Namely, at each time t the adversary computes the ratio[<sup>8</sup>](#page-4-1)

$$r_t = \frac{|\{h \in \mathcal{H}_t : h(x_t) = 1\}|}{|\mathcal{H}_t|}.$$

If <sup>r</sup><sup>t</sup> ∈ [ε, <sup>1</sup> − <sup>ε</sup>] for some parameter ε > <sup>0</sup> ("the version space is not too unbalanced"), then the adversary forces a mistake. Otherwise, the adversary assigns the majority label, i.e., y<sup>t</sup> = <sup>1</sup>(r<sup>t</sup> ≥ <sup>1</sup>/2). This ensures that the version space does not shrink too fast:

- If no mistake is forced, then |Ht+1| ≥ (1 − <sup>ε</sup>) · |Ht|, and
- If a mistake is forced, |Ht+1| ≥ <sup>ε</sup> · |Ht|.

<sup>8</sup> For a class H of Littlestone dimension d, the adversary will use only a subset of H of cardinality 2 d that shatters a Littlestone tree of depth d − 1. So without loss of generality, we assume that H has cardinality 2 d (in particular, H is finite), and the ratio is well-defined.

In particular, at the end of the game, the version space Hn+1 is of size

$$|\mathcal{H}_{n+1}| \geq \varepsilon^M \cdot (1-\varepsilon)^{n-M} \cdot |\mathcal{H}| \geq \varepsilon^M \cdot (1-\varepsilon)^n \cdot 2^d, \quad (1)$$

where M is the number of mistakes that the adversary forces and n is the length of the sequence. The class has size |H| ≥ <sup>2</sup> <sup>d</sup> because LD(H) = <sup>d</sup>, and by removing functions from the class if necessary (which can only make learning easier), we may assume without loss of generality that |H| = 2<sup>d</sup> . Namely, the class precisely shatters a Littlestone tree of depth <sup>d</sup> − <sup>1</sup> such that for every assignment of labels to a root-to-leaf path in the tree, the class contains exactly one function that agrees with that assignment (see Definition [A.6](#page-15-1) for detail).

Notice that we have not yet specified how the adversary selects the sequence x. While the adversary's labeling strategy is extremely simple (determined by the ratio r<sup>t</sup> and the prediction yˆt), constructing of the sequence x requires some care, to ensure that it has the following two properties:

- Property I: The length n of the sequence satisfies n = 2Θ( √
  - d) , and
- Property II: For every sequence of predictions yˆ1, . . . , yˆ<sup>n</sup> selected by the learner, the resulting sequence of labels y1, . . . , y<sup>n</sup> selected by the adversary are consistent with some function <sup>h</sup> ∈ H such that <sup>x</sup> contains all the nodes in path(h). [9](#page-5-0)

These properties can be achieved by carefully simulating all possible execution paths of the adversary.

Observe that if path(h) = (u1, . . . , ud) then the sequence of labels h(u1), h(u2), . . . , h(ud) uniquely identifies the function <sup>h</sup> within the class H. Hence, Property II and the assumption |H| = 2<sup>d</sup> imply that at the end of the game, the version space Hn+1 has cardinality

$$|\mathcal{H}_{n+1}| = 1. \quad (2)$$

Combining Property I (n = 2<sup>Θ</sup>( √ d) ), Eqs. [\(1\)](#page-5-1) and [\(2\)](#page-5-2), and choosing ε = 2<sup>−</sup>Θ( √ <sup>d</sup>) gives

$$1 \geq \varepsilon^M \cdot (1 - \varepsilon)^n \cdot 2^d \geq 2^{-\Theta(M \cdot \sqrt{d})} \cdot 2^d,$$

which implies <sup>M</sup> = Ω√ d , as desired.

#### 2.3 Proof Ideas for the Upper Bound

In this section we explain the main ideas in the proof of Theorem [D.1,](#page-2-0) which states that for every <sup>d</sup> ∈ <sup>N</sup>, there exists a class of Littlestone dimension <sup>d</sup> that is learnable in the transductive online setting with a mistake bound of O √ d .

Of course, not every Littlestone class satisfies this property. For instance, the set of all functions [d] → {0, <sup>1</sup>} has Littlestone dimension <sup>d</sup>, but the adversary can force the learner to make <sup>d</sup> mistakes when learning this class in the transductive setting.[<sup>10</sup>](#page-5-3) So our task in this proof is to construct a class that is especially easy to learn in the transductive setting (i.e., learnable with O √ d mistakes), while still being hard (requiring d mistakes) in the standard setting.

#### 2.3.1 Sparse Encodings are Easy to Guess

We start with an elementary observation. Consider the following two bit strings:

Binary: 110101

One-hot: 0000000000000000000000000000000000000000000000000000100000000000

Both of these strings encode the number 53. However, one of the encodings is much easier to guess than the other: suppose we are tasked with guessing the bits in an encoding of an integer between 0 and 2 <sup>6</sup> − <sup>1</sup>. We guess the bits one at a time, and after each guess, an adaptive adversary tells us whether our guess was correct.

<sup>9</sup>Recall that the *path* of a function h is depicted in Figure [1c,](#page-4-0) and defined in Definition [A.5.](#page-14-0)

<sup>10</sup>The adversary simply selects the sequence x = (1, 2, 3, . . . , d), and for each xi, the adversary forces a mistake by selecting y<sup>i</sup> = 1 − yˆi. The adversary's choice of labels is realizable because we are working with the class of all function [d] → {0, 1}.

Now, if the bit string is a binary encoding, the task is hard. Each bit can either be 0 or 1, regardless of the values of the previous bits, and so the adversary can force a mistake on every bit. On the other hand, if we know that the string is a one-hot encoding, there exists an attractive strategy — always guess 0. This ensures that we will make at most 1 mistake.

Note that at the end of the guessing game we have learned the same amount of *information* (for a number between 0 and 2 <sup>n</sup> − <sup>1</sup>, we learned <sup>n</sup> bits of information), but the number of *mistakes* is very different (n mistakes vs. 1 mistake).

#### 2.3.2 Construction of the Hypothesis Class

We now describe a construction of a hypothesis class that is easy to learn in the transductive setting, using the idea of a sparse encoding. Recall that a class H has Littlestone dimension at least <sup>d</sup> (Definition [A.6](#page-15-1) in Section [A\)](#page-13-0) if there exists a Littlestone tree of depth <sup>d</sup> − <sup>1</sup> such that for every <sup>b</sup> ∈ {0, <sup>1</sup>} d there exists <sup>h</sup> <sup>=</sup> <sup>h</sup><sup>b</sup> ∈ H such that the values on the path of <sup>h</sup> agree with <sup>b</sup>. More formally, ∀<sup>i</sup> ∈ [d] : <sup>h</sup>(b<i) = <sup>b</sup><sup>i</sup> , and in particular path(h) = (λ, b≤1, b≤2, b≤3, . . . , b≤d−1). Thus, when constructing a class that shatters a specific Littlestone tree of depth <sup>d</sup> − <sup>1</sup>, we need to define 2 d functions <sup>h</sup><sup>b</sup> : <sup>b</sup> ∈ {0, <sup>1</sup>} d . For each function hb, the on-path values of the function are fixed (fully determined by b), while for the remaining values there is complete freedom (for the nodes u that are off-path we may assign any values <sup>h</sup>b(u) ∈ {0, <sup>1</sup>}).

Perhaps the simplest way to construct a class of Littlestone dimension d is simply to assign all on-path values as required, and assign 0 to all other values. Namely, if u is a prefix of b then hb(u) = b|u|+1, and otherwise hb(u) = 0. In a sense, this is the 'minimal' class of Littlestone dimension d for a specific Littlestone tree.[<sup>11</sup>](#page-6-0)

Observe that the 'minimal' class does not have the desired property of being easy to learn in the transductive setting.[<sup>12</sup>](#page-6-1) However, a certain variation of the 'minimal' class that embeds a sparse encoding does satisfy the requirement. In this variation, on-path value of the function h<sup>b</sup> are assigned as they must (as determined by b), while the off-path values are sampled independently using a biased coin, such that each of them is 0 with high probability, but has a small probability of being 1. The probability is chosen carefully so that the class satisfies some simple combinatorial properties, as described further in Section [2.3.6](#page-8-0) and Lemma [D.2.](#page-25-0)

#### 2.3.3 Naïve Learning Strategy

We now explain in broad strokes how the probabilistic construction of the hypothesis class in Section [2.3.2](#page-6-2) is useful for learning with few mistakes in the transductive setting.

Notice that when predicting labels for the 'minimal' class with nodes in breadth-first order, the learner knows at each step whether they are labeling an on-path or off-path node, because the learner has already seen the correct labels for all ancestors of the current node. For off-path nodes, the learner knows that the true label is 0, so it never makes mistakes on off-path nodes, but it also gains no new information when the true labels for off-path nodes are revealed. No risk, but no reward either. Instead, all the information about the true labeling function is revealed only at on-path nodes, where the adversary has complete freedom to assign labels and force mistakes. That's why the adversary can force d mistakes.

For the randomly-chosen class, when predicting labels for off-path nodes, the learner may still safely predict a label of 0. But the reasoning for this is quite different. Conceptually, every off-path label is part of a sparse codeword that identifies the correct labeling function.[<sup>13</sup>](#page-6-3) Because the coin is biased, each bit of the codeword is easy to guess (it is likely to be 0), but every time that the adversary reveals that the true label for an off-path node is indeed 0, the learner gains a small (nonzero) amount of

<sup>11</sup>More formally, this is a class with a minimal number of nodes labeled 1.

<sup>12</sup>The adversary can declare a sequence x consisting of all the nodes in the tree in breadth-first order, and then force d mistakes — one mistake in each layer (depth) of the tree. Specifically, regardless of how the adversary selects the labels, for each i ∈ [d] there exists a node u<sup>i</sup> at depth i that is on-path. When it is time for the learner to predict a label for this ui, the learner knows that u<sup>i</sup> is on-path because it has seen the correct labels for all the ancestors of ui. However, the adversary has the freedom to extend the path arbitrarily to the left or to the right, and can therefore force a mistake on ui.

<sup>13</sup>The coin-flips for off-path labels are all independent. For example, if X is a set of nodes all of which are off-path for a subset H of the hypothesis class, then the random variables {h(x) : h ∈ H, x ∈ X} are i.i.d.

information about the true labeling function. Additionally, when the adversary selects an off-path label of 1, that reveals a lot of information about the true labeling function (such labels are rare in the hypothesis class), and therefore the adversary cannot force many off-path mistakes. Overall, the information about the true labeling function is 'smeared' throughout all labels of the tree (0s and 1s, on-path and off-path).[<sup>14</sup>](#page-7-0)

Thus, the naïve general strategy for the learner when using the probabilistically-constructed class is to learn most of the information about the true labeling function by observing off-path labels. By the time the learner reaches an on-path node, it hopefully has already learned enough about the true labeling function in order to make a good prediction on that node.

However, making this general strategy work requires overcoming some very substantial obstacles:

- 1. Recall that in the transductive setting, the adversary can present the nodes of the tree in any order of its choosing — it does not have to present the tree in breadth-first order. The naïve strategy works only if the learner sees many off-path nodes before it sees most on-path nodes. But what happens if the adversary decides to present many on-path nodes near the beginning of the sequence? To handle this, the learner incorporates a strategy we call 'danger zone minimization', as described in Section [2.3.4.](#page-7-1)
- 2. Another, equally problematic, issue also arises from the fact that the sequence presented by the adversary might not be in breadth-first order. Recall that breadth-first order[<sup>15</sup>](#page-7-2) has the property that for every node u in the sequence, all the ancestors of u appear *before* u in the sequence. This means that by the time the learner needs to predict a label for u, the learner knows whether u is on-path or off-path for the true labeling function. But what happens if the adversary presents u before some of u's ancestors? Or omits some of u's ancestors from the sequence altogether? In this case the learner doesn't know if u is on-path or off-path, and this presents a double hazard. One hazard is that the leaner doesn't know what label to predict for u — if u is off-path, the learner can simply predict 0, but if it is on-path it must do something more elaborate. The second hazard is that, after seeing the correct label for u, it is not clear what the learner can infer from it. If u is off-path, its label should be interpreted as part of a sparse encoding of the labeling function. But if u is on-path, the interpretation must be entirely different. To overcome this challenge, the learner incorporates a strategy we call 'splitting experts', described in Section [2.3.5.](#page-8-1)
- 3. Limiting off-path mistakes. Thanks to the coin's bias, most off-path nodes have a true label of 0. Nonetheless, each function in the hypothesis class still has an expected number of 2 Ω(d) off-path nodes labeled 1, so the learner can afford to misclassify only a vanishing fraction of them! To limit the number of mistakes, the learner extracts information from the sparse encoding and executes a 'transition to Halving' strategy, as described in Section [2.3.6.](#page-8-0)

#### 2.3.4 Danger Zone Minimization

Utilizing information from the 'sparse encoding' of the off-path nodes to make good predictions for on-path nodes requires that the learner first see the true labels for many off-path nodes. Until that happens, the learner expects to make many mistakes on on-path nodes. However, whether a node is on-path or off-path is not fixed in advanced — the adversary may decide this adaptively, in response to the learners predictions.

*Danger zone minimization* is a strategy used by the learner, to force the adversary to assign few nodes in the beginning of the sequence as on-path (otherwise, if initial nodes are assigned to be on-path by the adversary, then the learner will make few mistakes on those nodes). This is analogous to the standard Halving algorithm (Algorithm [7\)](#page-32-0), but instead of minimizing the cardinality of the set of consistent hypotheses (the 'version space'), the learner minimizes a subset of the domain (the 'danger zone').

<sup>14</sup>Furthermore, the labels for most not-too-small subsets of the nodes reveal a lot of information about the correct labeling function — not just for a particular subset of nodes. These properties led us to code-name this construction while working on the paper as 'everything everywhere all at once' (in reference to a 2022 film of that name). This is in contrast to the 'minimal' function, where the information is concentrated entirely on the function path. The asymmetry between the 'minimal' class and the probabilistic class is similar to that between the binary and one-hot encodings in Section [2.3.1](#page-5-4) above.

<sup>15</sup>As well as depth-first order.

Concretely, at the beginning of the game the learner initializes a set <sup>S</sup> <sup>=</sup> {<sup>x</sup>1, x2, . . . , xtmax} consisting of the first tmax = 2Ω( √ d) instances in the sequence x selected by the adversary. This set represents the 'danger zone' — nodes in the beginning of the sequence that have not been labeled yet, that *might* be on-path, and that are not ancestors of a previously-labeled on-path node.[<sup>16</sup>](#page-8-2) To predict a label for an instance x<sup>i</sup> , the learner selects a label yˆ<sup>i</sup> such that if yˆ<sup>i</sup> is wrong, the danger zone will shrink by at least <sup>1</sup>/3. That is, for <sup>b</sup> ∈ {0, <sup>1</sup>}, if the set <sup>S</sup><sup>b</sup> of <sup>b</sup>-descendants of <sup>x</sup><sup>i</sup> has cardinality |<sup>S</sup>b| ≥ |S|/3, the learner predicts <sup>y</sup>ˆ<sup>i</sup> <sup>=</sup> <sup>b</sup>. Then, if the adversary selects <sup>y</sup><sup>i</sup> = 1 − <sup>b</sup>, that implies that all <sup>b</sup>-descendants of x<sup>i</sup> are off-path for the true labeling functions. Therefore, the learner removes all b-descendants of <sup>x</sup><sup>i</sup> from the danger zone, and the new cardinality is |<sup>S</sup> \ <sup>S</sup>b| ≤ (2/3) · |S|. This guarantees that the learner can make at most O(log(tmax)) = O √ d such mistakes before the danger zone is empty.[<sup>17</sup>](#page-8-3)

If neither <sup>S</sup><sup>0</sup> nor <sup>S</sup><sup>1</sup> have cardinality at least |S|/3, the learner predicts <sup>y</sup>ˆ<sup>i</sup> = 0. If <sup>y</sup><sup>i</sup> = 1 and <sup>x</sup><sup>i</sup> is on-path for the true labeling function, then the learner updates the danger zone to be <sup>S</sup><sup>0</sup> ∪ <sup>S</sup>1, [<sup>18</sup>](#page-8-4) again shrinking the danger zone by a factor of at most 2/3. Otherwise, if y<sup>i</sup> = 1 and x<sup>i</sup> is off-path, then it was an off-path node labeled 1 (which is rare), and the learner can afford to misclassify it (see Section [2.3.6\)](#page-8-0).

#### 2.3.5 Splitting Experts

The danger zone minimization strategy requires that the learner know whether the node u being classified is on-path or off-path for the true labeling function. However, if u appears in the sequence before some of its ancestors, the learner does not know this. To overcome this difficulty, the learner implements a variant of the standard *multiplicative weights algorithm* using *splitting experts*. This means that initially there is a single expert executing danger zone minimization. When a node u is reached for which danger zone minimization requires knowing whether u is on-path or off-path and that information is not yet evident, each expert is split into two experts, one of which continues the execution of danger zone minimization under the assumption that u is on-path, and the other under the opposite assumption. Thus, at each point in time, there exists precisely one expert for which all path-related assumptions are correct, and therefore that expert will make at most O √ d mistakes. The multiplicative weights algorithm guarantees that the overall number of mistakes will be linear in the the number of mistakes of the best expert, i.e., O √ d .

#### 2.3.6 Transition to Halving

The hypothesis class is engineered such that it satisfies the following property: there are at most 2 <sup>O</sup>( √ d) functions in the hypothesis class that agree with any set of tmax = 2<sup>Ω</sup>( √ d) labels, or that agree that a set of Θ √ d nodes are all off-path and labeled 1 (this follows from Lemma [D.2\)](#page-25-0).

Therefore, once the true labels for the first tmax instances x1, x2, . . . , x<sup>t</sup>max have been revealed, or once Θ √ d off-path labels of 1 have been revealed (whichever happens first), the learner can *transition to halving*: stop doing danger zone minimization, and instead predict the labels for the remaining nodes using the standard Halving algorithm (Algorithm [7\)](#page-32-0) on the subset of the hypothesis class that survived. Halving on 2 <sup>O</sup>( √ d) functions is guaranteed to make at most O √ d mistakes (Fact [E.1\)](#page-37-0).

However, seeing as the learner lacks information on which nodes are off-path, it uses experts, and each expert maintains different path-related assumptions. Thus, each expert decides separately at which point to transition to Halving. The unique expert that makes only correct assumptions will

<sup>16</sup>If u is an ancestor of some on-path node v, and v is a b-descendant of u for b ∈ {0, 1}, then the true label for u must be b.

<sup>17</sup>Once the danger zone is empty, the learner cannot make any further on-path mistakes within the prefix x1, x2, . . . , x<sup>t</sup>max . And it will make at most O √ d mistakes on the remaining nodes x<sup>t</sup>max+1, x<sup>t</sup>max+2, . . . , as explained in Section [2.3.6.](#page-8-0)

<sup>18</sup>Because on-path nodes must be either be descendants or ancestors of xi, and the definition of the danger zone does not require that it contain ancestors of nodes that have been labeled.

transition 'at the right time'. That expert will make at most O √ d mistakes during danger zone minimization, and then at most O √ d additional mistakes during halving.

#### 2.4 Some Intuition for the Quantity √ d

We briefly sketch where the quantity √ d arises from. This is a back-of-the-envelope calculation without proof, intended purely as an aid for intuition. Suppose we assigned off-path labels of 1 with probability 2 −k instead of 2 − √ d . Consider a sequence x1, . . . , x<sup>n</sup> of n = d/2k leaves. For any sequence of labels <sup>y</sup>1, . . . , y<sup>n</sup> ∈ {0, <sup>1</sup>}, taking <sup>s</sup> <sup>=</sup> P i∈[n] yi , there exist roughly

$$2^d \cdot (2^{-k})^s \cdot (1 - 2^{-k})^{n-s} \geq 2^d \cdot (2^{-k})^n \gg 0$$

functions in the class for which these leaves are off-path and which agree with the labels y1, . . . , yn. Therefore, the adversary can force at least n = Ω(d/k) mistakes.

Similarly, for the sequence x1, . . . , x<sup>n</sup> consisting of all the nodes in the tree of depth at most k/2 in breadth-first order, the adversary can force a mistake on every on-path node while assigning a label of 0 to all off-path nodes, for a total of k/2 mistakes. This is true because for any assignment of on-path labels, the fraction of functions which agree with the on-path labels that assign a label of 0 to all off-path nodes is roughly <sup>1</sup> − <sup>2</sup> −k 2 k/2 ≈ <sup>1</sup>, so in particular for any labeling of the on-path nodes there exists a function in the class that agrees with that labeling and assigns 0 to all off-path nodes. Therefore, for any k, we obtain a *lower bound* of Ω d <sup>k</sup> + k on the number of mistakes. For any k, d <sup>k</sup> <sup>+</sup> <sup>k</sup> ≥ √ d, giving a lower bound of Ω √ d . Choosing k = √ d to minimize the lower bound will in fact yield a matching upper bound of O √ d , as we show in this paper. This completes our overview of the upper bound.

## 3 Directions for Future Work

Following are some interesting open questions:

- 1. Does there exist an efficient learning algorithm that achieves the O √ d upper bound of Theorem [D.1?](#page-2-0) One needs to be careful about the definition of efficiency here, but one possible formalization is as follows. Does there exist a learning algorithm A and a sequence of classes H1, H2, . . . , such that for every <sup>d</sup> ∈ <sup>N</sup>:
  - LD(Hd) = <sup>d</sup>, and
  - Given as input the index d and a sequence x1, . . . , xn, the algorithm A runs in time poly(d, n) and makes at most O √ d mistakes assuming the labels are realizable by Hd.
- 2. Is there a tradeoff between the cardinality of the domain X and the upper bound on the number of mistakes? We used a domain of size roughly 2 d in order to obtain our upper bound of O √ d . Is it possible to get the same bound with a domain of size poly(d)?
- 3. Obtaining more precise asymptotics; for example, is there (an explicit) constant α > 0 such that the optimal transductive mistake bound is <sup>α</sup> <sup>+</sup> <sup>o</sup>(1)√ d?

### 4 Organization

Complete rigorous mathematical details are deferred to the appendices. Formal definitions appear in Section [A.](#page-13-0) Formal statements and proofs for the lower bound and upper bound appear in Section [B](#page-15-2) and Section [D,](#page-24-0) respectively. Optimal sequence length is discussed in Section [C.](#page-21-0)

## Acknowledgments and Disclosure of Funding

ZC is supported in part by NSF EnCORE inst (award #2217058) and by Shachar Lovett's Simons Investigator Award (#929894). SM is a Robert J. Shillman Fellow; he acknowledges support by ISF grant 1225/20, by BSF grant 2018385, by Israel PBC-VATAT, by the Technion Center for Machine Learning and Intelligent Systems (MLIS), and by the the European Union (ERC, GENERALIZATION, 101039692). JS is supported in part by NSF CNS-2154149, an Amazon Research Award, and by Vinod Vaikuntanathan's Simons Investigator Award.

Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

## References


[1] Maria-Florina Balcan and Avrim Blum. A discriminative model for semi-supervised learning. *J. ACM*, 57(3):19:1–19:46, 2010. doi[:10.1145/1706591.1706599.](https://doi.org/10.1145/1706591.1706599) URL [https://doi.org/10.](https://doi.org/10.1145/1706591.1706599) [1145/1706591.1706599](https://doi.org/10.1145/1706591.1706599). Shai Ben-David, Eyal Kushilevitz, and Yishay Mansour. Online learning versus offline learning. In Paul M. B. Vitányi, editor, *Computational Learning Theory, Second European Conference, EuroCOLT '95, Barcelona, Spain, March 13-15, 1995, Proceedings*, volume 904 of *Lecture Notes in Computer Science*, pages 38–52. Springer, 1995. doi[:10.1007/3-540-59119-2\\_167.](https://doi.org/10.1007/3-540-59119-2_167) URL [https://doi.org/10.1007/3-540-59119-2\\_167](https://doi.org/10.1007/3-540-59119-2_167). Shai Ben-David, Eyal Kushilevitz, and Yishay Mansour. Online learning versus offline learning. *Mach. Learn.*, 29(1):45–63, 1997. doi[:10.1023/A:1007465907571.](https://doi.org/10.1023/A:1007465907571) URL [https://doi.org/10.](https://doi.org/10.1023/A:1007465907571) [1023/A:1007465907571](https://doi.org/10.1023/A:1007465907571). Shai Ben-David, Tyler Lu, Dávid Pál, and Miroslava Sotáková. Learning low-density separators. *CoRR*, abs/0805.2891, 2008. URL <http://arxiv.org/abs/0805.2891>. Gyora M. Benedek and Alon Itai. Learnability with respect to fixed distributions. *Theor. Comput. Sci.*, 86(2):377–390, 1991. doi[:10.1016/0304-3975\(91\)90026-X.](https://doi.org/10.1016/0304-3975(91)90026-X) URL [https://doi.org/10.](https://doi.org/10.1016/0304-3975(91)90026-X) [1016/0304-3975\(91\)90026-X](https://doi.org/10.1016/0304-3975(91)90026-X). Avrim Blum and Tom M. Mitchell. Combining labeled and unlabeled data with co-training. In Peter L. Bartlett and Yishay Mansour, editors, *Proceedings of the Eleventh Annual Conference on Computational Learning Theory, COLT 1998, Madison, Wisconsin, USA, July 24-26, 1998*, pages 92–100. ACM, 1998. doi[:10.1145/279943.279962.](https://doi.org/10.1145/279943.279962) URL [https://doi.org/10.1145/279943.](https://doi.org/10.1145/279943.279962) [279962](https://doi.org/10.1145/279943.279962). Olivier Bousquet, Steve Hanneke, Shay Moran, Ramon van Handel, and Amir Yehudayoff. A theory of universal learning. In Samir Khuller and Virginia Vassilevska Williams, editors, *STOC 2021: 53rd Annual ACM SIGACT Symposium on Theory of Computing, Virtual Event, Italy, June 21-25, 2021*, pages 532–541. ACM, 2021. doi[:10.1145/3406325.3451087.](https://doi.org/10.1145/3406325.3451087) URL [https:](https://doi.org/10.1145/3406325.3451087) [//doi.org/10.1145/3406325.3451087](https://doi.org/10.1145/3406325.3451087). Nicolò Cesa-Bianchi and Ohad Shamir. Efficient transductive online learning via randomized rounding. In Bernhard Schölkopf, Zhiyuan Luo, and Vladimir Vovk, editors, *Empirical Inference - Festschrift in Honor of Vladimir N. Vapnik*, pages 177–194. Springer, 2013. doi[:10.1007/978-3-](https://doi.org/10.1007/978-3-642-41136-6_16) [642-41136-6\\_16.](https://doi.org/10.1007/978-3-642-41136-6_16) URL [https://doi.org/10.1007/978-3-642-41136-6\\_16](https://doi.org/10.1007/978-3-642-41136-6_16). Olivier Chapelle, Vladimir N. Vapnik, and Jason Weston. Transductive inference for estimating values of functions. In Sara A. Solla, Todd K. Leen, and Klaus-Robert Müller, editors, *Advances in Neural Information Processing Systems 12, [NIPS Conference, Denver, Colorado, USA, November 29 - December 4, 1999]*, pages 421–427. The MIT Press, 1999. URL [http://papers.nips.cc/](http://papers.nips.cc/paper/1699-transductive-inference-for-estimating-values-of-functions) [paper/1699-transductive-inference-for-estimating-values-of-functions](http://papers.nips.cc/paper/1699-transductive-inference-for-estimating-values-of-functions). Olivier Chapelle, Bernhard Schölkopf, and Alexander Zien, editors. *Semi-Supervised Learning*. The MIT Press, 2006. ISBN 9780262033589. doi[:10.7551/MITPRESS/9780262033589.001.0001.](https://doi.org/10.7551/MITPRESS/9780262033589.001.0001) URL <https://doi.org/10.7551/mitpress/9780262033589.001.0001>. Malte Darnstädt, Hans Ulrich Simon, and Balázs Szörényi. Unlabeled data does provably help. In Natacha Portier and Thomas Wilke, editors, *30th International Symposium on Theoreti-*

[2] *cal Aspects of Computer Science, STACS 2013, February 27 - March 2, 2013, Kiel, Germany*, volume 20 of *LIPIcs*, pages 185–196. Schloss Dagstuhl - Leibniz-Zentrum für Informatik, 2013. doi[:10.4230/LIPICS.STACS.2013.185.](https://doi.org/10.4230/LIPICS.STACS.2013.185) URL [https://doi.org/10.4230/LIPIcs.](https://doi.org/10.4230/LIPIcs.STACS.2013.185) [STACS.2013.185](https://doi.org/10.4230/LIPIcs.STACS.2013.185). Alexander Gammerman, Volodya Vovk, and Vladimir N. Vapnik. Learning by transduction. In Gregory F. Cooper and Serafín Moral, editors, *UAI 1998: Proceedings of the Fourteenth Conference on Uncertainty in Artificial Intelligence, University of Wisconsin Business School, Madison, Wisconsin, USA, July 24-26, 1998*, pages 148–155. Morgan Kaufmann, 1998. URL [https://dslpitt.org/uai/displayArticleDetails.jsp?mmnu=1&smnu=2&](https://dslpitt.org/uai/displayArticleDetails.jsp?mmnu=1&smnu=2&article_id=243&proceeding_id=14) [article\\_id=243&proceeding\\_id=14](https://dslpitt.org/uai/displayArticleDetails.jsp?mmnu=1&smnu=2&article_id=243&proceeding_id=14). Christina Göpfert, Shai Ben-David, Olivier Bousquet, Sylvain Gelly, Ilya O. Tolstikhin, and Ruth Urner. When can unlabeled data improve the learning rate? In Alina Beygelzimer and Daniel Hsu, editors, *Conference on Learning Theory, COLT 2019, 25-28 June 2019, Phoenix, AZ, USA*, volume 99 of *Proceedings of Machine Learning Research*, pages 1500–1518. PMLR, 2019. URL <http://proceedings.mlr.press/v99/gopfert19a.html>. Steve Hanneke, Shay Moran, and Jonathan Shafer. A trichotomy for transductive online learning. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, *Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023*, 2023. URL [http://papers.nips.cc/paper\\_files/paper/2023/hash/](http://papers.nips.cc/paper_files/paper/2023/hash/3e32af2df2cd13dfbcbe6e8d38111068-Abstract-Conference.html) [3e32af2df2cd13dfbcbe6e8d38111068-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3e32af2df2cd13dfbcbe6e8d38111068-Abstract-Conference.html). Steve Hanneke, Vinod Raman, Amirreza Shaeiri, and Unique Subedi. Multiclass transductive online learning. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, *Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024*, 2024. URL [http://papers.nips.cc/paper\\_files/](http://papers.nips.cc/paper_files/paper/2024/hash/6f244818d72b2a4be9b1225d1344e950-Abstract-Conference.html) [paper/2024/hash/6f244818d72b2a4be9b1225d1344e950-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2024/hash/6f244818d72b2a4be9b1225d1344e950-Abstract-Conference.html). Steven C. H. Hoi, Doyen Sahoo, Jing Lu, and Peilin Zhao. Online learning: A comprehensive survey. *Neurocomputing*, 459:249–289, 2021. doi[:10.1016/J.NEUCOM.2021.04.112.](https://doi.org/10.1016/J.NEUCOM.2021.04.112) URL <https://doi.org/10.1016/j.neucom.2021.04.112>. Thorsten Joachims. Transductive inference for text classification using support vector machines. In Ivan Bratko and Saso Dzeroski, editors, *Proceedings of the Sixteenth International Conference on Machine Learning (ICML 1999), Bled, Slovenia, June 27 - 30, 1999*, pages 200–209. Morgan Kaufmann, 1999. Sham M. Kakade and Adam Kalai. From batch to transductive online learning. In *Advances in Neural Information Processing Systems 18 [Neural Information Processing Systems, NIPS 2005, December 5-8, 2005, Vancouver, British Columbia, Canada]*, pages 611–618, 2005. URL [https://proceedings.neurips.cc/paper/2005/hash/](https://proceedings.neurips.cc/paper/2005/hash/17693c91d9204b7a7646284bb3adb603-Abstract.html) [17693c91d9204b7a7646284bb3adb603-Abstract.html](https://proceedings.neurips.cc/paper/2005/hash/17693c91d9204b7a7646284bb3adb603-Abstract.html). Nick Littlestone. Learning quickly when irrelevant attributes abound: A new linear-threshold algorithm. *Mach. Learn.*, 2(4):285–318, 1987. doi[:10.1007/BF00116827.](https://doi.org/10.1007/BF00116827) URL [https://doi.](https://doi.org/10.1007/BF00116827) [org/10.1007/BF00116827](https://doi.org/10.1007/BF00116827). Shai Shalev-Shwartz and Shai Ben-David. *Understanding Machine Learning: From Theory to Algorithms*. Cambridge University Press, 2014. ISBN 978-1-10-705713-5. URL [http://www.cambridge.org/de/academic/](http://www.cambridge.org/de/academic/subjects/computer-science/pattern-recognition-and-machine-learning/understanding-machine-learning-theory-algorithms) [subjects/computer-science/pattern-recognition-and-machine-learning/](http://www.cambridge.org/de/academic/subjects/computer-science/pattern-recognition-and-machine-learning/understanding-machine-learning-theory-algorithms) [understanding-machine-learning-theory-algorithms](http://www.cambridge.org/de/academic/subjects/computer-science/pattern-recognition-and-machine-learning/understanding-machine-learning-theory-algorithms). Vladimir N. Vapnik. *Estimation of Dependencies Based on Empirical Data*. Nauka, Moscow, 1979. URL <https://www.ipu.ru/node/63854/publications>. In Russian. Vladimir N. Vapnik. *Estimation of Dependences Based on Empirical Data*. Springer, 2nd edition, 2006. ISBN 978-0-387-30865-4. doi[:10.1007/0-387-34239-7.](https://doi.org/10.1007/0-387-34239-7) URL [https://doi.org/10.](https://doi.org/10.1007/0-387-34239-7) [1007/0-387-34239-7](https://doi.org/10.1007/0-387-34239-7). Xiaojin Zhu. Semi-supervised learning literature survey. Technical report, Department of Computer Sciences, University of Wisconsin–Madison, 2005.

[3] Xiaojin Zhu. Semi-supervised learning. In Claude Sammut and Geoffrey I. Webb, editors, *Encyclopedia of Machine Learning*, pages 892–897. Springer, 2010. doi[:10.1007/978-0-387-30164-8\\_749.](https://doi.org/10.1007/978-0-387-30164-8_749)

[4] URL [https://doi.org/10.1007/978-0-387-30164-8\\_749](https://doi.org/10.1007/978-0-387-30164-8_749). Xiaojin Zhu and Andrew B. Goldberg. *Introduction to Semi-Supervised Learning*. Synthesis Lectures on Artificial Intelligence and Machine Learning. Morgan & Claypool Publishers, 2009. ISBN 978-3-031-00420-9. doi[:10.2200/S00196ED1V01Y200906AIM006.](https://doi.org/10.2200/S00196ED1V01Y200906AIM006) URL [https://doi.org/](https://doi.org/10.2200/S00196ED1V01Y200906AIM006)

[10.2200/S00196ED1V01Y200906AIM006](https://doi.org/10.2200/S00196ED1V01Y200906AIM006).
## Technical Appendices and Supplementary Material

## A Preliminaries

#### A.1 Basic Notation

Notation A.1. <sup>N</sup> <sup>=</sup> {1, <sup>2</sup>, <sup>3</sup>, . . .}*, i.e.,* <sup>0</sup> ∈/ <sup>N</sup>*.* log(·) *and* ln(·) *denote logarithm to base* <sup>2</sup> *and* <sup>e</sup>*, respectively.*

Notation A.2 (Sequences). *Let* X *be a set and* n, k ∈ <sup>N</sup>*. For a sequence* <sup>x</sup> = (x1, . . . , xn) ∈ X <sup>n</sup>*, we write* <sup>x</sup>≤<sup>k</sup> *to denote the subsequence* (x1, . . . , xk)*. If* <sup>k</sup> ≤ <sup>0</sup> *then* <sup>x</sup>≤<sup>k</sup> *denotes the empty sequence, which is also denoted by* <sup>λ</sup> <sup>=</sup> X 0 *. We use the notation* X <sup>≤</sup><sup>n</sup> <sup>=</sup> ∪ n <sup>i</sup>=0X i *.*

#### A.2 Standard Online Learning

Let X be a set, and let H ⊆ {0, <sup>1</sup>} <sup>X</sup> be a collection of functions called a *hypothesis class*. A *learner strategy* or simply *learner* for the standard online learning game (Game [1\)](#page-2-1) is a function

$$L : \bigcup_{i=0}^{n-1} (\mathcal{X} \times \{0, 1\})^i \times \mathcal{X} \rightarrow \{0, 1\},$$

where <sup>n</sup> ∈ <sup>N</sup> is the number of rounds in the game. The set of all such learner strategies is denoted Ln. An *adversary strategy* or simply *adversary* for the standard online learning game is a pair of functions

$$\begin{aligned} A_{\text{instance}} : & \bigcup_{i=0}^{n-1} (\mathcal{X} \times \{0, 1\} \times \{0, 1\})^i \rightarrow \mathcal{X}, \text{ and} \\ A_{\text{label}} : & \bigcup_{i=1}^{n-1} (\mathcal{X} \times \{0, 1\} \times \{0, 1\})^i \times \{0, 1\} \rightarrow \{0, 1\}. \end{aligned}$$

The set of all such adversary strategies is denoted An.

Semantically, the interpretation of these strategies is that in each round <sup>t</sup> ∈ [n] of Game [1,](#page-2-1) the adversary selects an instance

$$x_t = A_{\text{instance}}(x_1, \hat{y}_1, y_1, \dots, x_{t-1}, \hat{y}_{t-1}, y_{t-1}) \in \mathcal{X},$$

then the learner makes a prediction

$$\hat{y}_t = L(x_1, y_1, \dots, x_{t-1}, y_{t-1}, x_t) \in \{0, 1\},$$

and finally, the adversary assigns a label

$$y_t = A_{\text{label}}(x_1, \hat{y}_1, y_1, \dots, x_{t-1}, \hat{y}_{t-1}, y_{t-1}, \hat{y}_t) \in \{0, 1\}.$$

The adversary's function <sup>A</sup>label must satisfy *realizability*, meaning that there exists <sup>h</sup> ∈ H such that

$$\forall t \in [n] : y_t = h(x_t).$$

The number of mistakes in a game with <sup>n</sup> rounds and hypothesis class H between learner <sup>L</sup> and adversary A is

$$M_{\text{std}}(\mathcal{H}, n, L, A) = |\{t \in [n] : \hat{y}_t \neq y_t\}|.$$

#### A.3 Transductive Online Learning

Given X and H as in Section [A.2,](#page-13-1) a learner strategy for the *transductive online learning setting* (Game [2\)](#page-2-2) is a function

$$L : \mathcal{X}^n \times \bigcup_{i=0}^{n-1} \{0, 1\}^i \rightarrow \{0, 1\},$$

where <sup>n</sup> ∈ <sup>N</sup> is the number of rounds in the game. An adversary strategy consists of a sequence <sup>x</sup> ∈ X <sup>n</sup> and an *adversary labeling strategy*, which is a function

$$A : \left( \bigcup_{i=0}^{n-1} \{0, 1\}^{2^i} \right) \times \{0, 1\} \rightarrow \{0, 1\}.$$

The sets of all such learner and adversary strategies are denoted L<sup>n</sup> and A<sup>n</sup> respectively.

Semantically, the interpretation of these strategies is that at the start of Game [2,](#page-2-2) the adversary selects the sequence <sup>x</sup>. Then, in each round <sup>t</sup> ∈ [n], the learner makes a prediction

$$\hat{y}_t = L(x, y_1, \dots, y_{t-1}) \in \{0, 1\},$$

and then the adversary assigns a label

$$y_t = A(\hat{y}_1, y_1, \dots, \hat{y}_{t-1}, y_{t-1}, \hat{y}_t) \in \{0, 1\}.$$

Exactly as in Section [A.2,](#page-13-1) the adversary's function A must satisfy realizability, namely,

$$\exists h \in \mathcal{H} \forall t \in [n] : y_t = h(x_t),$$

and the number of mistakes in a game with sequence length <sup>n</sup> and hypothesis class H between learner L and adversary A is

$$M_{\text{tr}}(\mathcal{H}, n, L, A) = |\{t \in [n] : \hat{y}_t \neq y_t\}|.$$

#### A.4 Mistake Bounds

In this paper, we study *optimal mistake bounds*, or the *optimal number of mistakes*, which is the value of Games [<sup>1</sup>](#page-2-1) and [2.](#page-2-2) For <sup>M</sup> ∈ {<sup>M</sup>std, Mtr}, the optimal number of mistakes in a game with hypothesis class H and sequence length <sup>n</sup> is,

$$M(\mathcal{H}, n) = \inf_{L \in \mathcal{L}_n} \sup_{A \in \mathcal{A}_n} M(\mathcal{H}, n, L, A).$$

The optimal number of mistakes for hypothesis class H is

$$M(\mathcal{H}) = \sup_{n \in \mathbb{N}} M(\mathcal{H}, n).$$

Remark A.3. As is common in learning theory literature, in both Game [1](#page-2-1) and Game [2,](#page-2-2) we take the sets L<sup>n</sup> and A<sup>n</sup> to be the sets of all (deterministic) functions. In this paper, we do not consider randomized strategies. By allowing arbitrary functions, we ignore issues relating to computability.

#### A.5 Trees

Definition A.4 (Notation for binary trees). *Let* <sup>d</sup> ∈ <sup>N</sup> ∪ {0}*. A perfect binary tree of depth* <sup>d</sup> *is a collection of* 2 <sup>d</sup>+1 − <sup>1</sup> *nodes, which we identify with the collection of binary strings*

$$T_d = \{\{0,1\}^k : k \in \{0,1,2,\dots,d\}\}.$$

*The empty string, denoted* <sup>λ</sup> <sup>=</sup> {0, <sup>1</sup>} 0 *, is a member of* T<sup>d</sup> *and is called the root of the tree. Every string* <sup>u</sup> ∈ {0, <sup>1</sup>} d *is called a leaf. The depth of a node* <sup>u</sup> ∈ <sup>T</sup>d*, denoted* |u|*, is the length of* <sup>u</sup> *as a string, namely, the integer* <sup>k</sup> *such that* <sup>u</sup> ∈ {0, <sup>1</sup>} k

*For two nodes* u, v ∈ <sup>T</sup>d*, we say that* <sup>u</sup> *is a parent of* <sup>v</sup>*, and that* <sup>v</sup> *is a child of* <sup>u</sup>*, if* <sup>v</sup> <sup>=</sup> <sup>u</sup> ◦ <sup>0</sup> *or* <sup>v</sup> <sup>=</sup> <sup>u</sup> ◦ <sup>1</sup>*, where* ◦ *denotes string concatenation. More fully, for* <sup>b</sup> ∈ {0, <sup>1</sup>}*, we say that* <sup>v</sup> *is a* <sup>b</sup>*-child of* <sup>u</sup> *if* <sup>v</sup> <sup>=</sup> <sup>u</sup> ◦ <sup>b</sup>*.*

*Recursively, we define that* u *is an ancestor of* v *and that* v *is a descendant of* u*, and write* u ≼ v*, if one of the following holds:*

- u = v*, or*
- ∃<sup>w</sup> ∈ <sup>T</sup><sup>d</sup> ∃<sup>b</sup> ∈ {0, <sup>1</sup>} : (<sup>u</sup> <sup>≼</sup> <sup>w</sup>) ∧ (<sup>w</sup> <sup>b</sup> <sup>=</sup> <sup>v</sup>)*.*

*For* <sup>b</sup> ∈ {0, <sup>1</sup>}*, we say that* <sup>v</sup> *is a* <sup>b</sup>*-descendant of* <sup>u</sup>*, denoted* <sup>u</sup> <sup>≼</sup><sup>b</sup> <sup>v</sup>*, if* <sup>v</sup> *is a descendant of the* b*-child of* u*.*

A function <sup>f</sup> : <sup>T</sup><sup>d</sup> → {0, <sup>1</sup>} specifies a particular root-to-leaf path in the tree <sup>T</sup><sup>d</sup> (see Figure [1\)](#page-4-0). The *on-path* nodes for f are the set of d + 1 nodes on that root-to-leaf path, as in the following definition. Definition A.5 (Paths in a binary tree). *Let* d, k ∈ <sup>N</sup>*,* <sup>k</sup> ≤ <sup>d</sup>*. Let* <sup>u</sup> ∈ {0, <sup>1</sup>} <sup>k</sup> *be a node in* Td*. The path to* u *is the unique sequence* path(u) = (u0, u1, u2, . . . , uk) *such that* u<sup>0</sup> = λ *is the root,* u<sup>k</sup> = u*, and* u<sup>i</sup> *is a child of* <sup>u</sup>i−<sup>1</sup> *for all* <sup>i</sup> ∈ [k]*.*

*Let* <sup>f</sup> : <sup>T</sup><sup>d</sup> → {0, <sup>1</sup>} *be a function. The path of* <sup>f</sup> *is the unique sequence* path(f) = (u0, u1, u2, . . . , ud) *such that* <sup>u</sup><sup>0</sup> <sup>=</sup> <sup>λ</sup> *is the root, and for each* <sup>i</sup> ∈ [d]*,* <sup>u</sup><sup>i</sup> <sup>=</sup> <sup>u</sup>i−<sup>1</sup> ◦ <sup>f</sup>(ui−1)*. Namely,* u<sup>i</sup> *is the* f(ui−1)*-child of* ui−1*.*

*For a node* <sup>v</sup> ∈ <sup>T</sup><sup>d</sup> *and a function* <sup>f</sup> : <sup>T</sup><sup>d</sup> → {0, <sup>1</sup>}*, we write* <sup>v</sup> ∈ path(f) *if* path(f) = (u0, . . . , ud) *and there exists* <sup>i</sup> ∈ {0, . . . , d} *such that* <sup>u</sup><sup>i</sup> <sup>=</sup> <sup>v</sup>*. Otherwise, we write* v /∈ path(f)*.*

*For a node* <sup>v</sup> ∈ <sup>T</sup><sup>d</sup> *and a set of functions* F ⊆ {0, <sup>1</sup>} <sup>T</sup><sup>d</sup> *, we write* <sup>v</sup> ∈ path(F) *if*

$$\forall f \in \mathcal{F} : v \in \text{path}(f).$$

*Otherwise, we write* u /∈ path(F)*.*

#### A.6 Littlestone Dimension

Definition A.6 [\(Littlestone,](#page-11-0) [1987\)](#page-11-0). *Let* X *be a set, let* H ⊆ {0, <sup>1</sup>} <sup>X</sup> *, and let* <sup>d</sup> ∈ <sup>N</sup> ∪ {0}*. We say that* H *shatters the binary tree* <sup>T</sup><sup>d</sup> *if there exists a mapping* <sup>T</sup><sup>d</sup> → X *given by* <sup>u</sup> 7→ <sup>x</sup><sup>u</sup> *such that for every* <sup>u</sup> ∈ {0, <sup>1</sup>} <sup>d</sup>+1 *there exists* <sup>h</sup><sup>u</sup> ∈ H *such that*

$$\forall i \in [d+1] : h(x_{u_{\leq i-1}}) = u_i.$$

*The Littlestone dimension of* H*, denoted* LD(H)*, is the supremum over all* <sup>d</sup> ∈ <sup>N</sup> *such that there exists a Littlestone tree of depth* <sup>d</sup> − <sup>1</sup> *that is shattered by* H*.*

Note that by defining the Littlestone dimension this way, every class with Littlestone dimension <sup>d</sup> ∈ <sup>N</sup> contains at least <sup>2</sup> d functions.

![](_page_15_Diagram_11.jpeg)

Figure 2: A shattered Littlestone tree of depth 2. The empty sequence is denoted by λ.

(Source: [Bousquet et al.,](#page-10-10) [2021\)](#page-10-10)

Theorem A.7 [\(Littlestone,](#page-11-0) [1987\)](#page-11-0). *Let* X *be a set and let* H ⊆ {0, <sup>1</sup>} <sup>X</sup> *such that* <sup>d</sup> <sup>=</sup> LD(H) <sup>&</sup>lt; ∞*. Then there exists a strategy for the learner that guarantees that the learner will make at most* d *mistakes in the standard (non-transductive) online learning setting, regardless of the adversary's strategy and of the number* n *of instances to be labeled. Furthermore, there exists an adversary that forces every learner to make at least* min {n, d} *mistakes.*

## B Lower Bound

#### B.1 Statement

Our Ω √ d lower bound states the following.

Theorem B.1 (Lower bound). *There exists a constant* <sup>d</sup><sup>0</sup> ≥ <sup>0</sup> *as follows. Let* <sup>d</sup> ∈ <sup>N</sup>*,* <sup>d</sup> ≥ <sup>d</sup>0*, let* X *be a set, and let* H ⊆ {0, <sup>1</sup>} <sup>X</sup> *be a hypothesis class with* LD(H) = <sup>d</sup>*. Then there exist a sequence* <sup>x</sup> ∈ X <sup>n</sup> *of length* <sup>n</sup> <sup>=</sup> <sup>O</sup> <sup>d</sup> · <sup>2</sup> √ d *and an adversary* A *that always selects the sequence* x *and uses a simple adaptive labeling strategy (as in Algorithm [1\)](#page-16-0), such that for every learning rule* L*,*

$$M_{\text{tr}}(\mathcal{H}, n, L, A) \geq \sqrt{d}/10. \quad (3)$$

*Furthermore, for every integer* <sup>n</sup> ∈ <sup>N</sup>*,*

$$M_{\text{tr}}(\mathcal{H}, n) \geq \min \left\{ \sqrt{d}/10, \lfloor \log(n+1) \rfloor \right\}. \quad (4)$$

Remark B.2. The assumption LD(H) = <sup>d</sup> implies that for all <sup>k</sup> ∈ [d], H shatters a Littlestone tree of depth k. Thus, the lower bound of Eq. [\(3\)](#page-16-1) in Theorem [B.1](#page-2-0) immediately implies that for every <sup>k</sup> ∈ [d] there exists a sequence <sup>x</sup> (k) ∈ X <sup>n</sup><sup>k</sup> of length <sup>n</sup><sup>k</sup> <sup>=</sup> <sup>O</sup> <sup>k</sup> · <sup>2</sup> √ k such that the adversary A<sup>k</sup> that presents the sequence x (k) and assigns labels using the simple labeling strategy of Algorithm [1](#page-16-0) ensures that for every learner L,

$$M_{\text{tr}}(\mathcal{H}, n_k, L, A_k) \geq \sqrt{k}/10.$$

See Section [2.2](#page-3-0) for a general overview of Theorem [B.1](#page-2-0) and the main proof ideas. In the following subsections we prove Theorem [B.1.](#page-2-0) Algorithm [1](#page-16-0) gives an explicit construction of the adversary that witnesses the lower bound, using Algorithm [2](#page-17-0) as a subroutine. We start with presenting some initial observations about the behavior of these algorithms in Section [B.2.](#page-16-2)

#### Assumptions:

- <sup>d</sup> ∈ <sup>N</sup>, <sup>ε</sup> = 2<sup>−</sup> √ d/2 .
- T = T<sup>d</sup> is a perfect binary tree of depth d.
- H ⊆ {0, <sup>1</sup>} T is a class that shatters T.

<sup>T</sup>RANSDUCTIVEADVERSARY (H):

(x1, x2, . . . , xn) ← <sup>C</sup>ONSTRUCTSEQUENCE (H) <sup>▷</sup> See Algorithm [2.](#page-17-0)

send (x1, x2, . . . , xn) to learner

H<sup>0</sup> ← H for <sup>t</sup> ∈ [n]:

receive yˆ<sup>t</sup> from learner

<sup>r</sup><sup>t</sup> ←

|{<sup>h</sup> ∈ Ht−<sup>1</sup> : <sup>h</sup>(xt) = 1}| |Ht−1|

$$y_{\text{maj}} \leftarrow \mathbb{1}(r_t \geq 1/2)$$

$$y_t \leftarrow \begin{cases} y_{\text{maj}} & r_t \notin [\varepsilon, 1 - \varepsilon] \\ 1 - \hat{y}_t & \text{otherwise} \end{cases}$$

send y<sup>t</sup> to learner

$$\mathcal{H}_t \leftarrow \{h \in \mathcal{H}_{t-1} : h(x_t) = y_t\}$$

Algorithm 1: The strategy for the adversary that achieves the lower bound in Theorem [B.1.](#page-2-0) Note that while the construction of the sequence x is not entirely trivial, the adversary's strategy for labeling this sequence is very simple.

#### B.2 Analysis of the Adversary

Claim B.3. *Let* <sup>d</sup> ∈ <sup>N</sup>*, let* <sup>M</sup> <sup>=</sup> √ d/10*, and let* H ⊆ {0, <sup>1</sup>} <sup>T</sup><sup>d</sup> *be a hypothesis class. Consider an execution of* <sup>C</sup>ONSTRUCTSEQUENCE (H) *as in Algorithm [<sup>2</sup>](#page-17-0) that produces a sequence* <sup>x</sup>1, x2, . . . , xn*. Then:*

- • <sup>d</sup> ∈ <sup>N</sup>, <sup>M</sup> <sup>=</sup> √ d/10, ε = 2<sup>−</sup> √ d/2 .
- T = T<sup>d</sup> is a perfect binary tree of depth d.
- λ, the empty string, is the root of T.
- H ⊆ {0, <sup>1</sup>} T is a class that shatters T.

<sup>C</sup>ONSTRUCTSEQUENCE (H):

$$\mathcal{H}_\lambda \leftarrow \mathcal{H}_{\pi\pi}$$

$$\mathbb{H}_0 \leftarrow \{\mathcal{H}_\lambda\}$$

<sup>H</sup><sup>0</sup> ← {Hλ} <sup>▷</sup> A set of classes indexed by bit strings. Q ← {λ} <sup>▷</sup> A set of nodes to be processed.

$$Q \leftarrow \{\lambda\}$$

$$t \leftarrow 0$$

**while** 
$$|\mathcal{Q}| > 0$$
:

<sup>x</sup><sup>t</sup> ← arbitrary element from Q <sup>▷</sup> Pop an arbitrary element from Q Q ← Q \ {<sup>x</sup> and add it to the output sequence. <sup>t</sup>}

$$\mathbb{H}_t \leftarrow \emptyset$$

for 
$$\mathcal{H}_b \in \mathbb{H}_{t-1}$$
:

$$r \leftarrow \frac{|\{h \in \mathcal{H}_b : h(x_t) = 1\}|}{|\mathcal{H}_b|}$$

$$\mathcal{V} \leftarrow \left\{ \begin{array}{l} \{0, 1\} \\ \{1 (r \geq 1/2)\} \end{array} \right. \quad (r \in [\varepsilon, 1 - \varepsilon]) \wedge (|b| < M) \quad \triangleright \text{Adversary will force mistakes on the first } M \text{ balanced nodes.}$$

$$\text{for } y \in \mathcal{Y}: \quad b' \leftarrow \begin{cases} b & |\mathcal{Y}| = 1 \\ b \circ y & |\mathcal{Y}| = 2 \end{cases}$$

<sup>b</sup> ◦ <sup>y</sup> |Y| = 2 <sup>▷</sup> Restrict class to agree with <sup>y</sup>. If splitting the class in two to force a mis-H<sup>b</sup> take then create new indices. ′ ← {<sup>h</sup> ∈ H<sup>b</sup> : <sup>h</sup>(xt) = <sup>y</sup>}

$$\mathbb{H}_t \leftarrow \mathbb{H}_t \cup \{\mathcal{H}_{b'}\}$$

**if** 
$$x_t \in \text{path}(\mathcal{H}_{b'}) \wedge |x_t| < d: \triangleright$$
 If  $x_t$  is on-path for  $\mathcal{H}_{b'}$  and it has a  $\mathcal{Q} \leftarrow \mathcal{Q} \cup \{x_t \circ y\}$   $y$ -child, add that child to  $\mathcal{Q}$ .

return (x1, x2, . . . , xt)

Algorithm 2: A subroutine of Algorithm [1](#page-16-0) for selecting the sequence x.

- *(a) For all* <sup>i</sup> ∈ [n]*,* path(xi) *is a subsequence of* <sup>x</sup>0, x1, . . . , x<sup>i</sup> *.*
- *(b) The length* <sup>n</sup> *of the sequence satisfies* n < nd*, where* <sup>n</sup><sup>d</sup> = (<sup>d</sup> + 1) · <sup>2</sup><sup>M</sup>+1 *.*

#### *Proof.*

- (a) Fix <sup>i</sup> ∈ [n]. It suffices to show that for all <sup>u</sup> ∈ <sup>T</sup>d, if <sup>u</sup> <sup>≼</sup> <sup>x</sup><sup>i</sup> then <sup>u</sup> ∈ (x1, x2, . . . , xi). Proceed by induction on i. For the base case i = 1, the claim holds because x<sup>1</sup> = λ. For the induction step, assume the claim holds for <sup>i</sup> ∈ [<sup>n</sup> − 1]. Let <sup>u</sup> <sup>≼</sup> <sup>x</sup>i+1, we prove that <sup>u</sup> ∈ (x1, x2, . . . , xi+1). Assume <sup>x</sup>i+1 ̸<sup>=</sup> <sup>λ</sup> (otherwise, there is nothing to prove).

Because <sup>x</sup>i+1 appears in the sequence <sup>x</sup>, it must have been added to Q before it was added to <sup>x</sup>. The only place where items that are not <sup>λ</sup> are added to Q is in the line Q ← Q ∪ {<sup>x</sup><sup>t</sup> ◦ <sup>y</sup>}. Namely, there exist an index <sup>j</sup> ∈ [i] and a bit <sup>y</sup> ∈ {0, <sup>1</sup>} such that <sup>x</sup>i+1 <sup>=</sup> <sup>x</sup><sup>j</sup> ◦ <sup>y</sup> (note that j < i + 1 because x<sup>j</sup> was added to the sequence before xi+1). If x<sup>j</sup> = u we are done. Otherwise, note that x<sup>j</sup> is the parent of xi+1, and therefore u ≼ x<sup>j</sup> . By the induction hypothesis, <sup>u</sup> ∈ (x1, x2, . . . , x<sup>j</sup> ). This concludes the proof.

- (b) Items are added to the sequence <sup>x</sup> only if they were previously added to Q. By induction on <sup>i</sup> ∈ [n], for each <sup>x</sup><sup>i</sup> in the sequence, there is at most one iteration of the "while |Q| <sup>&</sup>gt; <sup>0</sup>" loop in which x<sup>i</sup> is added to Q. The base case i = 1 holds because x<sup>1</sup> = λ is the root, which is added to Q before the while loop, and λ is never added to Q within that loop because the line "Q ← Q ∪ {<sup>x</sup><sup>t</sup> ◦ <sup>y</sup>}" can only add non-empty bit strings. For the induction step, if the claim holds for all natural numbers <sup>j</sup> such that <sup>1</sup> ≤ j < i ≤ <sup>n</sup> then it holds for <sup>i</sup>. Indeed, for <sup>i</sup> ≥ <sup>2</sup>, <sup>x</sup><sup>i</sup> can be added to <sup>Q</sup> only via the line "Q ← Q ∪ {<sup>x</sup><sup>t</sup> ◦ <sup>y</sup>}", and only in the iteration of the while loop where x<sup>t</sup> is the parent of x<sup>i</sup> in the tree Td. In that iteration, the parent x<sup>t</sup> of xi is popped from Q, which implies that x<sup>t</sup> was added to Q in some previous iteration of the while loop (t < i), and is no longer in Q after being popped. By the induction hypothesis, x<sup>t</sup> will never be added to Q again, and therefore in all subsequent iterations of the while loop x<sup>t</sup> will not be the parent of x<sup>i</sup> , so x<sup>i</sup> cannot be added to Q in subsequent iterations via the line "Q ← Q ∪ {<sup>x</sup><sup>t</sup> ◦ <sup>y</sup>}".

Furthermore, if a node x<sup>i</sup> is added to Q in some iteration of the while loop, then it remains in <sup>Q</sup> for the duration of that iteration. So for all <sup>i</sup> ∈ {2, <sup>3</sup>, . . . , n}, there is precisely one execution of the line "Q ← Q ∪ {<sup>x</sup><sup>t</sup> ◦ <sup>y</sup>}" that adds <sup>x</sup><sup>i</sup> to Q. Namely, there is precisely one point in time during the execution of Algorithm [<sup>2</sup>](#page-17-0) in which <sup>x</sup><sup>i</sup> <sup>=</sup> <sup>x</sup><sup>t</sup> ◦ <sup>y</sup>, <sup>x</sup><sup>i</sup> ∈/ <sup>Q</sup>, and the line "Q ← Q ∪ {<sup>x</sup><sup>t</sup> ◦ <sup>y</sup>}" is executed resulting in <sup>x</sup><sup>i</sup> ∈ <sup>Q</sup>.

Consider a function <sup>f</sup> that maps <sup>i</sup> ∈ {2, <sup>3</sup>, . . . , n} to the value of the index <sup>b</sup> ′ during the unique execution of the line "Q ← Q ∪ {<sup>x</sup><sup>t</sup> ◦ <sup>y</sup>}" that adds <sup>x</sup><sup>i</sup> to Q. Namely, if b ′ had some value β when x<sup>i</sup> was added to Q, then f(i) = β.

Notice that "Q ← Q∪{<sup>x</sup><sup>t</sup> ◦y}" is executed only if the condition <sup>x</sup><sup>t</sup> ∈ path(H<sup>b</sup> ′ ) is satisfied in the previous line. Furthermore, the line "H<sup>b</sup> ′ ← {<sup>h</sup> ∈ H<sup>b</sup> : <sup>h</sup>(xt) = <sup>y</sup>}" ensures that the node <sup>x</sup><sup>i</sup> <sup>=</sup> <sup>x</sup><sup>t</sup> ◦ <sup>y</sup> being added to <sup>Q</sup> satisfies <sup>x</sup><sup>t</sup> ◦ <sup>y</sup> ∈ path(H<sup>b</sup> ′ ), namely

$$\forall h \in \mathcal{H}_{b'} : x_i \in \text{path}(h).$$

Consequently, <sup>x</sup><sup>i</sup> ∈ path(G) for any class G that is a subset of H<sup>b</sup> ′ ; in particular, because the only way that H<sup>b</sup> ′ might be modified later during the execution of Algorithm [2](#page-17-0) is by removing elements, it follows that <sup>x</sup><sup>i</sup> ∈ path(H<sup>b</sup> ′ ) when the line "Q ← Q ∪ {<sup>x</sup><sup>t</sup> ◦ <sup>y</sup>}" is executed and in all subsequent times.

However, |path(G)| <sup>=</sup> <sup>d</sup> + 1 for any class G ⊆ {0, <sup>1</sup>} <sup>T</sup><sup>d</sup> . This implies that f maps at most (d + 1) nodes to each bit string. In other words, for any bit string b, the size of the preimage satisfies |<sup>f</sup> −1 (b)| ≤ <sup>d</sup> + 1.

The condition "|b| < M" in Algorithm [<sup>2</sup>](#page-17-0) ensures that |<sup>b</sup> ′ | ≤ <sup>M</sup>, namely, <sup>b</sup> ′ ∈ {0, <sup>1</sup>} k for <sup>k</sup> ∈ {0, <sup>1</sup>, <sup>2</sup>, . . . , M}. Thus,

$$\begin{aligned}
n &= 1 + |\{2, 3, \dots, n\}| \\
&= 1 + \sum_{\substack{b \in \{0,1\}^k \\ k \in \{0,\dots,M\}}} |\{i \in \{2, 3, \dots, n\} : f(i) = b\}| \\
&= 1 + \sum_{\substack{b \in \{0,1\}^k \\ k \in \{0,\dots,M\}}} |f^{-1}(b)| \\
&\leq 1 + \sum_{\substack{b \in \{0,1\}^k \\ k \in \{0,\dots,M\}}} (d + 1) \\
&\leq 1 + (d + 1) \cdot (2^{M+1} - 1). \\
&< (d + 1) \cdot 2^{M+1},
\end{aligned}$$

Claim B.4. *Let* <sup>d</sup> ∈ <sup>N</sup>*, let* <sup>M</sup> <sup>=</sup> √ d/10*, and let* H ⊆ {0, <sup>1</sup>} <sup>T</sup><sup>d</sup> *be a hypothesis class. Consider an execution of* <sup>T</sup>RANSDUCTIVEADVERSARY (H) *as in Algorithm [1.](#page-16-0) Let*

$$\mathcal{H}_0, \mathcal{H}_1, \dots, \mathcal{H}_n$$

*be the sequence of hypothesis classes created by* TRANSDUCTIVEADVERSARY*, let*

$$S = \{t \in [n] : r_t \in [\varepsilon, 1 - \varepsilon]\}$$

*be the set of indices where* TRANSDUCTIVEADVERSARY *forces a mistake, and let*

$$\mathbb{H}_0, \mathbb{H}_1, \dots, \mathbb{H}_n$$

*be the sequence of collections created by the subroutine* CONSTRUCTSEQUENCE *(Algorithm [2\)](#page-17-0). If* |S| ≤ <sup>M</sup> *then*

$$\forall t \in \{0, 1, \dots, n\} : \mathcal{H}_t \in \mathbb{H}_t.$$

*Proof.* Proceed by induction on <sup>t</sup> ∈ {0, <sup>1</sup>, . . . , n}. The base case <sup>t</sup> = 0 is satisfied, because H<sup>0</sup> <sup>=</sup> H ∈ {H} <sup>=</sup> <sup>H</sup>0. For the induction step, assume that Hi−<sup>1</sup> ∈ <sup>H</sup>i−<sup>1</sup> for some <sup>i</sup> ∈ [n]. We prove that H<sup>i</sup> ∈ <sup>H</sup><sup>i</sup> .

Let y<sup>i</sup> be the label assigned to x<sup>i</sup> by TRANSDUCTIVEADVERSARY. Then

$$\mathcal{H}_i = \{h \in \mathcal{H}_{i-1} : h(x_i) = y_i\}.$$

Consider the iteration of the while loop in <sup>C</sup>ONSTRUCTSEQUENCE that starts with <sup>t</sup> ← <sup>i</sup>. By the induction hypothesis, Hi−<sup>1</sup> ∈ <sup>H</sup>i−1. Therefore, in this iteration of the while loop, there will be an iteration of the "for H<sup>b</sup> ∈ <sup>H</sup>t−1" loop where H<sup>b</sup> <sup>=</sup> Hi−1. In that iteration, <sup>y</sup><sup>i</sup> ∈ Y by construction of <sup>y</sup><sup>i</sup> and Y. Therefore, in the iteration of the "for <sup>y</sup> ∈ Y" loop in which <sup>y</sup> <sup>=</sup> <sup>y</sup><sup>i</sup> ,

$$\mathcal{H}_{b'} = \{h \in \mathcal{H}_b : h(x_t) = y\} = \{h \in \mathcal{H}_{i-1} : h(x_i) = y_i\} = \mathcal{H}_i.$$

The class H<sup>b</sup> ′ is then added to <sup>H</sup><sup>i</sup> <sup>=</sup> <sup>H</sup><sup>t</sup> in the line "H<sup>t</sup> ← <sup>H</sup><sup>t</sup> ∪ {H<sup>b</sup> ′}". Furthermore, no class is ever removed from <sup>H</sup>t. So H<sup>i</sup> ∈ <sup>H</sup><sup>i</sup> , as desired.

Claim B.5. *Let* <sup>d</sup> ∈ <sup>N</sup>*, let* <sup>M</sup> <sup>=</sup> √ d/10*, and let* H ⊆ {0, <sup>1</sup>} <sup>T</sup><sup>d</sup> *be a hypothesis class. Consider an execution of* <sup>T</sup>RANSDUCTIVEADVERSARY (H) *as in Algorithm [<sup>1</sup>](#page-16-0) where the adversary constructs a sequence of nodes* <sup>x</sup>1, x2, . . . , x<sup>n</sup> ∈ <sup>T</sup><sup>d</sup> *and a sequence of classes* H0, H1, . . . , H<sup>n</sup> ⊆ {0, <sup>1</sup>} <sup>T</sup><sup>d</sup> *. Let*

$$S = \{t \in [n] : r_t \in [\varepsilon, 1 - \varepsilon]\}$$

*be the set of indices where* <sup>T</sup>RANSDUCTIVEADVERSARY *forces a mistake, and assume that* |S| ≤ <sup>M</sup>*. Then for all* <sup>k</sup> ∈ {0, <sup>1</sup>, . . . , d} *there exists* <sup>i</sup> ∈ [n] *such that*

1. 1. 
   $$|x_i| = k$$
   , and
2. 2.  $x_i \in \text{path}(\mathcal{H}_{i-1})$ ,

*Proof.* Proceed by induction on <sup>k</sup>. For the base case <sup>k</sup> = 0, notice that <sup>x</sup><sup>1</sup> <sup>=</sup> <sup>λ</sup>, |λ| = 0, and <sup>λ</sup> ∈ path(H−1).

For the induction step, assume the claim holds for some <sup>k</sup> ∈ {0, <sup>1</sup>, . . . , d − <sup>1</sup>}, and take <sup>i</sup><sup>k</sup> ∈ [n] such that |<sup>x</sup><sup>i</sup><sup>k</sup> | <sup>=</sup> <sup>k</sup> and <sup>x</sup><sup>i</sup><sup>k</sup> ∈ path(H<sup>i</sup>k−1); we prove that the claim holds for <sup>k</sup> + 1 as well.

Consider the iteration of the while loop in CONSTRUCTSEQUENCE in which x<sup>i</sup><sup>k</sup> is added to the sequence (i.e., the iteration starting with <sup>t</sup> ← <sup>i</sup>k). By Claim [B.4](#page-19-0) and the assumption |S| ≤ <sup>M</sup>, H<sup>i</sup>k−<sup>1</sup> ∈ <sup>H</sup>ik−1. Hence, within this iteration of the while loop, there is an iteration of the "for H<sup>b</sup> ∈ <sup>H</sup>t−1" loop such that H<sup>b</sup> <sup>=</sup> H<sup>i</sup>k−1. By construction, the set Y always contains the label predicted by the adversary, so <sup>y</sup><sup>i</sup><sup>k</sup> ∈ Y. Consider the iteration of the "for <sup>y</sup> ∈ Y" loop such that y = y<sup>i</sup><sup>k</sup> . By the induction hypothesis, <sup>x</sup><sup>i</sup> ∈ path(H<sup>i</sup>k−1), and since H<sup>b</sup> ′ ⊆ H<sup>b</sup> <sup>=</sup> H<sup>i</sup>k−1, it follows that <sup>x</sup><sup>i</sup><sup>k</sup> ∈ path(H<sup>b</sup> ′ ). Seeing as |<sup>x</sup><sup>i</sup><sup>k</sup> | < d, in the last line of this iteration of the "for <sup>y</sup> ∈ Y" loop, the node x<sup>i</sup>k+1 := x<sup>i</sup><sup>k</sup> ◦ <sup>y</sup><sup>i</sup><sup>k</sup> is added to Q. This guarantees that <sup>x</sup><sup>i</sup>k+1 will eventually be popped from

Q and added to the sequence returned by <sup>C</sup>ONSTRUCTSEQUENCE. Once a node has been added to the sequence, it is never removed.

Notice that |<sup>x</sup>ik+1 | <sup>=</sup> |<sup>x</sup>i<sup>k</sup> | + 1 = <sup>k</sup> + 1, satisfying Item [1.](#page-19-1) Therefore, it remains to show Item [2,](#page-19-2) namely, to show that <sup>x</sup>ik+1 ∈ path H<sup>i</sup>k+1−<sup>1</sup> .

Indeed, by the induction hypothesis, <sup>x</sup><sup>i</sup> ∈ path(H<sup>i</sup>k−1), and in the iteration of the "for <sup>y</sup> ∈ Y" discussed above, H<sup>b</sup> <sup>=</sup> H<sup>i</sup>k−1, H<sup>b</sup> ′ <sup>=</sup> H<sup>i</sup><sup>k</sup> , and H<sup>b</sup> ′ <sup>=</sup> {<sup>h</sup> ∈ H<sup>b</sup> : <sup>h</sup>(xi<sup>k</sup> ) = yi<sup>k</sup> }. Hence,

$$\forall h \in \mathcal{H}_{i_k} : x_{i_k} \in \text{path}(h) \wedge h(x_{i_k}) = y_{i_k}.$$

Seeing as xik+1 = xi<sup>k</sup> ◦ <sup>y</sup>i<sup>k</sup> This implies that

$$\forall h \in \mathcal{H}_{i_k} : x_{i_k+1} \in \text{path}(h).$$

Item [<sup>2</sup>](#page-19-2) follows from the inclusion H<sup>i</sup>k+1−<sup>1</sup> ⊆ H<sup>i</sup><sup>k</sup> .

#### B.3 Proof

Finally, we complete the proof of the lower bound.

*Proof of Theorem [B.1.](#page-2-0)* Fix <sup>d</sup><sup>0</sup> = 800 and assume <sup>d</sup> ≥ <sup>d</sup>0. Seeing as LD(H) = <sup>d</sup>, H shatters the tree <sup>T</sup>d. By replacing H with a suitable subset of H of cardinality <sup>2</sup> <sup>d</sup>+1, renaming the elements in the domain of H to nodes of <sup>T</sup>d, and restricting the domain of each function in H to <sup>T</sup>d, assume without loss of generality that H ⊆ {0, <sup>1</sup>} <sup>T</sup><sup>d</sup> , |H| = 2<sup>d</sup>+1, and H shatters <sup>T</sup>d.

Consider the loop "for <sup>t</sup> ∈ [n]" in Algorithm [1,](#page-16-0) and let

$$S = \{s_1, s_2, \dots, s_m\} = \{t \in [n] : r_t \in [\varepsilon, 1 - \varepsilon]\}$$

be the set of indices where the adversary forces a mistake, such that the learner makes at least <sup>m</sup> <sup>=</sup> |S| mistakes. Let M = √ d/10, and assume for contradiction that <sup>m</sup> ≤ <sup>M</sup>.

By Claim [B.5,](#page-19-3) there exists <sup>t</sup> ∈ [n] such that |<sup>x</sup>t| <sup>=</sup> <sup>d</sup> (i.e., <sup>x</sup><sup>t</sup> is a leaf in <sup>T</sup>d) and <sup>x</sup><sup>t</sup> ∈ path(Ht−1), namely,

∀<sup>h</sup> ∈ Ht−<sup>1</sup> : <sup>x</sup><sup>t</sup> ∈ path(h).

Seeing as x<sup>t</sup> is a leaf,

$$\forall h \in \mathcal{H}_{t-1} : \text{path}(x_t) = \text{path}(h). \quad (5)$$

By construction,

$$\mathcal{H}_t \subseteq \left\{ h \in \mathcal{H} : (\forall i \in [t] : h(x_i) = y_i) \right\},$$

and H<sup>t</sup> is not empty. Fix some <sup>h</sup> <sup>∗</sup> ∈ H<sup>t</sup> ⊆ Ht−1. By Item [\(a\)](#page-17-1) in Claim [B.3,](#page-16-3) path(xt) = path(<sup>h</sup> ∗ ) is a subsequence of x1, x2, . . . , xt, so

$$\forall h \in \mathcal{H}_t \quad \forall x \in \text{path}(h^*) : h(x) = h^*(x).$$

Seeing as H shatters <sup>T</sup><sup>d</sup> and |H| = 2<sup>k</sup>+1, if two functions h, h<sup>∗</sup> ∈ H agree on the labels for all nodes in path(h ∗ ), then h = h ∗ . We conclude that H<sup>t</sup> <sup>=</sup> {<sup>h</sup> <sup>∗</sup>} and |Ht| = 1.

Consider the loop "for <sup>t</sup> ∈ [n]" in Algorithm [1.](#page-16-0) For each <sup>t</sup> ∈ [n],

$$|\mathcal{H}_t| \geq \begin{cases} \varepsilon \cdot |\mathcal{H}_{t-1}| & t \in S \\ (1 - \varepsilon) \cdot |\mathcal{H}_{t-1}| & t \notin S. \end{cases}$$

Hence,

$$\begin{aligned} 1 &= |\mathcal{H}_t| \\ &\geq \varepsilon^m \cdot (1 - \varepsilon)^{n-m} \cdot |\mathcal{H}_0| \\ &= \varepsilon^m \cdot (1 - \varepsilon)^{n-m} \cdot 2^{d+1} \\ &\geq \varepsilon^m \cdot (1 - \varepsilon)^n \cdot 2^{d+1} \\ &\geq \varepsilon^m \cdot (1 - \varepsilon)^{n_d} \cdot 2^{d+1} \\ &\geq \varepsilon^m \cdot 2^d = 2^{-m\sqrt{d}/2+d}, \end{aligned} \quad (\text{by Item (b) in Claim B.3.}) \quad (6)$$

where the final line holds because ε = 2<sup>−</sup> √ d/2 , <sup>n</sup><sup>d</sup> = (<sup>d</sup> + 1) · <sup>2</sup> √ d/10+1, and

$$(1 - \varepsilon)^{n_d} = \left(1 - 2^{-\sqrt{d}/2}\right)^{(d+1) \cdot 2^{\sqrt{d}/10+1}} \geq \frac{1}{2}$$

for our choice of <sup>d</sup> ≥ <sup>800</sup>. Rearranging Eq. [\(6\)](#page-20-0) yields

$$2\sqrt{d} \leq m.$$

This is a contradiction to the assumption <sup>m</sup> ≤ <sup>M</sup> <sup>=</sup> √ d/10. We conclude that an adversary A following Algorithm [1](#page-16-0) satisfies

$$\inf_{L \in \mathcal{L}_n} M_{\text{tr}}(\mathcal{H}, n, L, A) \geq m > M = \sqrt{d}/10, \quad (7)$$

as desired.

To establish the "furthermore" part of the theorem, fix a length <sup>n</sup> ∈ <sup>N</sup>. Let <sup>k</sup> be the largest integer such that 2 ⌈ √ k/10⌉ ≤ <sup>n</sup>+ 1 and <sup>k</sup> ≤ <sup>d</sup>. By Eq. [\(7\)](#page-21-1), there exists some sequence on which the adversary can force every learning rule to make at least l√ k/10m mistakes. By Theorem [C.2,](#page-21-2) this implies that there exists a sequence of length 2 ⌈ √ k/10⌉ − <sup>1</sup> ≤ <sup>n</sup> on which the adversary can force every learning rule to make at least l√ k/10m = min nl√ d/10m , ⌊log(<sup>n</sup> + 1)⌋ o mistakes. Namely,

$$M_{\text{tr}}(\mathcal{H}, n) \geq \min \left\{ \left\lceil \sqrt{d}/10 \right\rceil, \lfloor \log(n+1) \rfloor \right\},$$

as in Eq. [\(4\)](#page-16-4).

## C Sequence Length

In this section, we show that if there exists a sequence on which the adversary can force M mistakes, then a sequence of length <sup>2</sup><sup>M</sup> − <sup>1</sup> is sufficient, and this upper bound is tight for some classes.[<sup>19</sup>](#page-21-3)

Definition C.1 (Minimal sequence). *Let* X *be a set, let* H ⊆ {0, <sup>1</sup>} <sup>X</sup> *be a class, and let* <sup>M</sup> ∈ <sup>N</sup>*.*

*The minimal sequence length for forcing* <sup>M</sup> *mistakes for the class* H*, denoted* MinLen(H, M) *is*

$$\text{MinLen}(\mathcal{H}, M) = \inf \{ n \in \mathbb{N} : (\exists x \in \mathcal{X}^n : M_{\text{tr}}(\mathcal{H}, x) \geq M) \}.$$

*In words,* MinLen(H, M) *is the smallest integer* <sup>n</sup> *for which there exists a sequence of length* n *on which the adversary can force at least* n *mistakes; if no such sequence exists, then* MinLen(H, M) = ∞*.*

Theorem C.2 (Minimal sequence bound). *Let* X *be a set, and fix* <sup>M</sup> ∈ <sup>N</sup>*. Then for any class* H ⊆ {0, <sup>1</sup>} <sup>X</sup> *, if* MinLen(H, M) <sup>&</sup>lt; ∞ *then*

$$\text{MinLen}(\mathcal{H}, M) \leq 2^M - 1.$$

*Furthermore, there exists a class* H ⊆ {0, <sup>1</sup>} <sup>X</sup> *for which* MinLen(H, M) = 2<sup>M</sup> − <sup>1</sup>*.*

Theorem [C.2](#page-21-2) is a corollary of the tree rank characterization of Mtr from [Ben-David et al.](#page-10-1) [\(1997\)](#page-10-1). For completeness, we present a direct proof of Theorem [C.2](#page-21-2) that does not directly invoke that characterization. Roughly, given an adversary A<sup>0</sup> that forces every learner to make at least M mistakes on a (possibly long) sequence x, we apply two modifications to obtain new adversaries

$$A_0 \rightsquigarrow A_1 \rightsquigarrow A_2.$$

A<sup>1</sup> forces M mistakes and has a specific structure that we call 'rigidity', but it still uses the same (possibly long) sequence x. Capitalizing on the rigid structure, A<sup>2</sup> selects a subsequence of x of length at most <sup>2</sup><sup>M</sup> − <sup>1</sup>, and forces <sup>M</sup> mistakes on that subsequence.

<sup>19</sup>Of course, there also exist classes for which a shorter sequence is sufficient. For instance, if the class shatters (in the VC sense) a subset of the domain of cardinality M, then a sequence of length M suffices.

#### C.1 Rigid Adversary

Definition C.3 (Rigid adversary). *Let* <sup>n</sup> ∈ <sup>N</sup>*, let* X *be a set, and let*

$$A : \left( \bigcup_{k=0}^{n-1} \{0, 1\}^{2k} \right) \times \{0, 1\} \rightarrow \{0, 1\}$$

*be an adversary strategy for some fixed sequence* <sup>x</sup> ∈ X <sup>n</sup>*. We say that* <sup>A</sup> *is rigid if there exists a function*

$$f : \bigcup_{k=0}^{n-1} \{0, 1\}^k \rightarrow \{0, 1, \star\}$$

*such that for all* <sup>k</sup> ∈ {0, <sup>1</sup>, . . . , n − <sup>1</sup>} *and all* y, <sup>y</sup><sup>ˆ</sup> ∈ {0, <sup>1</sup>} k *,*

$$A(\hat{y}_1, y_1, \dots, \hat{y}_k, y_k, \hat{y}_{k+1}) = \begin{cases} f(y_1, \dots, y_k) & f(y_1, \dots, y_k) \in \{0, 1\} \\ 1 - \hat{y}_{k+1} & f(y_1, \dots, y_k) = \star \end{cases}$$

Note that if an adversary is rigid, then the function f that witnesses this is uniquely determined.

Claim C.4 (Rigid adversary exists). *Let* n, M ∈ <sup>N</sup>*, let* X *be a set, let* <sup>x</sup> ∈ X <sup>n</sup>*, and let* H ⊆ {0, <sup>1</sup>} X *be a class. Let* A *be an adversary strategy that forces every learner to make at least* M *mistakes on* x*. Then there exists an adversary strategy* A<sup>∗</sup> *such that:*

- *1.* A<sup>∗</sup> *forces every learner to make at least* M *mistakes on* x *and* A<sup>∗</sup> *is rigid.*
- *2. Let* f *be the function that witnesses the rigidity of* A<sup>∗</sup> *. Then for every* <sup>y</sup> ∈ {0, <sup>1</sup>} <sup>n</sup>*, the sequence*

$$f(y_{\leq 0}), f(y_{\leq 1}), f(y_{\leq 2}), \dots, f(y),$$

*has at least* M *members equal to* ⋆*.*

*Proof of Claim [C.4.](#page-19-0)* For Item [1,](#page-22-0) consider the adversary strategy A<sup>∗</sup> that simulates an execution of A, as in Algorithm [3.](#page-23-0) In broad strokes, A<sup>∗</sup> functions as a middle-man between the learner and A. As the learner makes a sequence of predictions <sup>y</sup><sup>ˆ</sup> ∈ {0, <sup>1</sup>} <sup>n</sup>, the adversary A<sup>∗</sup> generates a sequence of (possibly different) predictions <sup>y</sup>˜ ∈ {0, <sup>1</sup>} <sup>n</sup>, and sends those to the adversary A. Adversary A sees only the predictions <sup>y</sup>˜, and assigns labels <sup>y</sup> ∈ {0, <sup>1</sup>} <sup>n</sup>, which are relayed back to the learner by A<sup>∗</sup> with no modifications.

First, observe that A<sup>∗</sup> satisfies the realizability requirement. Indeed, A<sup>∗</sup> simulates an execution of A such that the sequence of labels y1, . . . , y<sup>n</sup> sent by A<sup>∗</sup> to the learner is exactly the sequence of labels selected by A. Seeing as A is realizable, every sequence of labels selected by A is realizable, and therefore every sequence of labels selected by A<sup>∗</sup> must be realizable as well.

Second, observe that A<sup>∗</sup> forces every leaner to make at least M mistakes. To see this, notice that in Algorithm [3,](#page-23-0)

$$\sum_{t \in [n]} \mathbb{1}(\tilde{y}_t \neq y_t) \geq M. \quad (8)$$

Indeed, A forces every learner to make at least M mistakes, and in particular this applies to a learner that makes predictions y˜ as in the simulation. Furthermore, observe that A<sup>∗</sup> only alters the predictions it receives from the learner in cases when it selects a label that is accepted by A, namely,

$$\forall t \in [n] : \tilde{y}_t \neq \hat{y}_t \implies \tilde{y}_t = y_t. \quad (9)$$

Therefore, if <sup>E</sup> <sup>=</sup> {<sup>t</sup> ∈ [n] : ˜y<sup>t</sup> = ˆyt}, then

$$\begin{aligned} \sum_{t \in [n]} \mathbf{1}(\tilde{y}_t \neq y_t) &= \sum_{t \in E} \mathbf{1}(\tilde{y}_t \neq y_t) + \sum_{t \in [n] \setminus E} \mathbf{1}(\tilde{y}_t \neq y_t) \\ &= \sum_{t \in E} \mathbf{1}(\tilde{y}_t \neq y_t) + 0 \quad (\text{By Eq. (9)}) \\ &= \sum_{t \in E} \mathbf{1}(\hat{y}_t \neq y_t) \quad (\text{Definition of } E) \end{aligned}$$

- • <sup>n</sup> ∈ <sup>N</sup>, X is a set, <sup>x</sup> ∈ X <sup>n</sup> is a fixed sequence of instances.
- A : Sn−<sup>1</sup> <sup>k</sup>=0 {0, <sup>1</sup>} 2k × {0, <sup>1</sup>} → {0, <sup>1</sup>} is an adversary labeling strategy for <sup>x</sup>.

#### RIGIDADVERSARY:

send x1, . . . , x<sup>n</sup> to the learner

for t = 1, 2, . . . , n:

receive prediction yˆ<sup>t</sup> from learner

if A(˜y1, y1, . . . , y˜t−1, yt−1, 0) = 0:

<sup>y</sup>˜<sup>t</sup> ← <sup>0</sup>

else if A(˜y1, y1, . . . , y˜t−1, yt−1, 1) = 1:

<sup>y</sup>˜<sup>t</sup> ← <sup>1</sup>

else:

<sup>y</sup>˜<sup>t</sup> ← <sup>y</sup>ˆ<sup>t</sup> send prediction y˜<sup>t</sup> to A

receive label y<sup>t</sup> from A

send label y<sup>t</sup> to learner

Algorithm 3: Construction of a rigid adversary, by simulating a given adversary A.

$$\leq \sum_{t \in [n]} \mathbf{1}(\hat{y}_t \neq y_t). \quad (10)$$

Combining Eqs. [\(8\)](#page-22-2) and [\(10\)](#page-23-1) implies that A forces at least M mistakes.

Third, we show that A<sup>∗</sup> is rigid. We claim that there exists a function <sup>g</sup> : {0, <sup>1</sup>} <sup>≤</sup>n−<sup>1</sup> → {0, <sup>1</sup>} ≤n−1 such that for every <sup>t</sup> ∈ {0, <sup>1</sup>, <sup>2</sup>, . . . , n − <sup>1</sup>},

$$(\tilde{y}_1, \dots, \tilde{y}_t) = g(y_1, \dots, y_t).$$

Proceed by induction on t. For the base case t = 0 there is nothing to prove. For the induction step, we assume the claim holds for some <sup>t</sup> <sup>=</sup> k < n − <sup>1</sup>, and show that it holds for <sup>t</sup> <sup>=</sup> <sup>k</sup> + 1. From Algorithm [3,](#page-23-0) y˜k+1 satisfies

$$\tilde{y}_{k+1} = \begin{cases} 0 & A(\tilde{y}_1, y_1, \dots, \tilde{y}_k, y_k, 0) = 0 \\ 1 & A(\tilde{y}_1, y_1, \dots, \tilde{y}_k, y_k, 0) = A(\tilde{y}_1, y_1, \dots, \tilde{y}_k, y_k, 1) = 1 \\ 1 - y_{k+1} & \text{otherwise} \end{cases} \quad (11)$$

The first two cases in Eq. [\(11\)](#page-23-2) are immediate from Algorithm [3,](#page-23-0) and the remaining case occurs when <sup>A</sup> forces a mistake at time <sup>k</sup> + 1, namely, when <sup>A</sup> selects <sup>y</sup>k+1 = 1 − <sup>y</sup>˜k+1. Thus, <sup>y</sup>˜k+1 is a function of y≤k+1 and y˜≤k. By the induction hypothesis, y˜≤<sup>k</sup> = g(y≤k), so y˜k+1 is simply a function of y≤k+1. This establishes the existence of the desired function g.

Hence, A<sup>∗</sup> is rigid, as witnessed by the function

$$f(y_1, \dots, y_k) = \begin{cases} 0 & A(\tilde{y}_1, y_1, \dots, \tilde{y}_k, y_k, 0) = 0 \\ 1 & A(\tilde{y}_1, y_1, \dots, \tilde{y}_k, y_k, 0) = A(\tilde{y}_1, y_1, \dots, \tilde{y}_k, y_k, 1) = 1 \\ \star & \text{otherwise} \end{cases}$$

where f is a well-defined function because y˜≤<sup>k</sup> = g(y≤k).

We have seen that A<sup>∗</sup> is a valid (realizable) adversary that forces every learner to make at least M mistakes, and it is rigid. This concludes the proof of Item [1.](#page-22-0)

Finally, For Item [2,](#page-22-3) note that <sup>y</sup>˜<sup>t</sup> ̸<sup>=</sup> <sup>y</sup><sup>t</sup> only if <sup>A</sup> forces a mistake at time <sup>t</sup> in the sense that <sup>A</sup> selects <sup>y</sup><sup>t</sup> = 1 − <sup>b</sup> for any prediction <sup>b</sup> ∈ {0, <sup>1</sup>} provided at time <sup>t</sup>. If <sup>A</sup> forces a mistake at time <sup>t</sup>, then <sup>A</sup><sup>∗</sup> forces a mistake at time <sup>t</sup> as well. Therefore, if <sup>y</sup>˜<sup>t</sup> ̸<sup>=</sup> <sup>y</sup>t, then <sup>f</sup>(y<t) = <sup>⋆</sup>, namely, <sup>y</sup>˜<sup>t</sup> makes mistakes only when the value of f is ⋆. By Eq. [\(8\)](#page-22-2), y˜<sup>t</sup> makes at least M mistakes throughout the game, so there must be at least M rounds where f outputs ⋆, as desired.

#### C.2 Essential Indices

Definition C.5. *Let* n, M ∈ <sup>N</sup>*, let* X *be a set, let* <sup>x</sup> ∈ X <sup>n</sup>*, and let* H ⊆ {0, <sup>1</sup>} <sup>X</sup> *be a class. Let* A *be a rigid adversary strategy witnessed by function* <sup>f</sup>*. We say that an index* <sup>t</sup> ∈ [n] *is essential for* <sup>A</sup> *for forcing* <sup>M</sup> *mistakes on* <sup>x</sup> *if there exists a sequence* <sup>y</sup> ∈ {0, <sup>1</sup>} t−1 *such that* f(y) = ⋆ *and the sequence*

$$f(y_{\leq 0}), f(y_{\leq 1}), f(y_{\leq 2}), \dots, f(y_{\leq t-1})$$

*contains at most* <sup>M</sup> − <sup>1</sup> *members equal to* <sup>⋆</sup>*.*

Claim C.6. *Let* n, M ∈ <sup>N</sup>*, let* X *be a set, let* <sup>x</sup> ∈ X <sup>n</sup>*, and let* H ⊆ {0, <sup>1</sup>} <sup>X</sup> *be a class. Let* A *be a rigid adversary strategy. Then* [n] *contains at most* <sup>2</sup><sup>M</sup> − <sup>1</sup> *indices that are essential for* <sup>A</sup> *for forcing* M *mistakes on* x*.*

*Proof.* For each essential index <sup>t</sup> ∈ [n], there exists a label sequence <sup>y</sup> ∈ {0, <sup>1</sup>} t−1 that witnesses that t is essential, as in Definition [C.5.](#page-14-0) Each label sequence y is a witness for at most one index (the index |y| + 1), so it suffices to show that the set <sup>Y</sup> ⊆ {0, <sup>1</sup>} <sup>≤</sup>n−<sup>1</sup> of all witness label sequences is of cardinality at most <sup>2</sup><sup>M</sup> − <sup>1</sup>.

Think of Y as a collection of nodes in the binary tree Tn−<sup>1</sup> (Definition [A.4\)](#page-14-1). By Definition [C.5,](#page-14-0) if <sup>y</sup> ∈ <sup>Y</sup> , then the collection of all ancestors of <sup>y</sup> in <sup>Y</sup> has cardinality

$$|\{y_{\leq i} : i \in \{0, 1, 2, \dots, |y| - 1\}\} \cap Y| \leq M - 1.$$

Namely, <sup>Y</sup> is a subtree of depth at most <sup>d</sup> <sup>=</sup> <sup>M</sup> − <sup>1</sup> in the binary tree <sup>T</sup>n−1. [<sup>20</sup>](#page-24-1) Hence, the number of nodes in Y is at most

$$2^{d+1} - 1 = 2^M - 1,$$

as desired.

#### C.3 Proof

*Proof of Theorem [C.2.](#page-21-2)* If MinLen(H, M) <sup>&</sup>lt; ∞, then there exist a sequence <sup>x</sup> ∈ X <sup>n</sup>, and an adversary A<sup>0</sup> that forces every learner to make at least M mistakes on x. By Claim [C.4,](#page-19-0) there exists a rigid adversary A<sup>1</sup> that causes every learner to make at least M mistakes on x, [<sup>21</sup>](#page-24-2) and also satisfies Item [2](#page-22-3) in Claim [C.4.](#page-19-0) Let <sup>f</sup> be the function that witnesses the rigidity of <sup>A</sup>1. By Claim [C.6,](#page-24-3) the set <sup>I</sup> ⊆ [n] of indices that are essential for <sup>A</sup><sup>1</sup> for forcing <sup>M</sup> mistakes on <sup>x</sup> has cardinality <sup>k</sup> <sup>=</sup> |I| ≤ <sup>2</sup><sup>M</sup> − <sup>1</sup>.

Algorithm [4](#page-25-1) defines a new adversary, A2, which forces every learner to make at least M mistakes on a sequence of length k. A<sup>2</sup> is realizable, because A<sup>1</sup> is realizable.[<sup>22</sup>](#page-24-4)

To see that adversary A<sup>2</sup> forces every learner to make at least M mistakes, let y1, . . . , y<sup>n</sup> be the sequence of labels assigned by A2. Seeing as A<sup>2</sup> assigns the same labels as A1, and A<sup>1</sup> satisfies Item [<sup>2</sup>](#page-22-3) in Claim [C.4,](#page-19-0) it follows that there are at least <sup>M</sup> indices <sup>j</sup> ∈ [n] such that <sup>f</sup>(y≤j−1) = <sup>⋆</sup>. Fix <sup>J</sup> ⊆ [n] to be the first <sup>M</sup> such indices. Then <sup>J</sup> ⊆ <sup>I</sup>, namely, all the indices in <sup>J</sup> are essential for <sup>A</sup><sup>1</sup> for forcing M mistakes on x (Definition [C.5\)](#page-14-0).

Therefore, for each <sup>j</sup> ∈ <sup>J</sup>, <sup>A</sup><sup>2</sup> includes the instance <sup>x</sup><sup>j</sup> in the sequence of length <sup>k</sup> sent to the learner. Then, in round j of the n rounds simulated by A2:

- The leaner makes a prediction <sup>y</sup>ˆ<sup>j</sup> ∈ {0, <sup>1</sup>} corresponding to instance <sup>x</sup><sup>j</sup> .
- Adversary A<sup>2</sup> sends prediction yˆ<sup>j</sup> to adversary A1. Because f(y≤j−1) = ⋆, adversary A<sup>1</sup> assigns the label <sup>y</sup><sup>j</sup> = 1 − <sup>y</sup>ˆ<sup>j</sup> . Adversary <sup>A</sup><sup>2</sup> then sends that label <sup>y</sup><sup>j</sup> to the learner. So the learner makes a mistake on x<sup>j</sup> .

Hence, the learner makes at least |J| <sup>=</sup> <sup>M</sup> mistakes, as desired.

<sup>20</sup>The depth of a subtree is s if the longest root-to-node path contains s + 1 nodes from the subtree.

<sup>21</sup>This is Item [1](#page-22-0) in Claim [C.4.](#page-19-0)

<sup>22</sup>The argument for realizability is the same as in the proof of Claim [C.4.](#page-19-0)

- • n, M ∈ <sup>N</sup>, X is a set, <sup>x</sup> ∈ X <sup>n</sup> is a fixed sequence of instances.
- A<sup>1</sup> : Sn−<sup>1</sup> <sup>k</sup>=0 {0, <sup>1</sup>} 2k × {0, <sup>1</sup>} → {0, <sup>1</sup>} is a rigid adversary labeling strategy for x that forces every learner to make at least M mistakes on the sequence x, and satisfies Items [1](#page-22-0) and [2](#page-22-3) in Claim [C.4.](#page-19-0)
- <sup>I</sup> <sup>=</sup> {<sup>i</sup>1, i2, . . . , ik} ⊆ [n] is the set of indices that are essential for <sup>A</sup> for forcing <sup>M</sup> mistakes on <sup>x</sup>, and <sup>i</sup><sup>1</sup> ≤ <sup>i</sup><sup>2</sup> ≤ · · · ≤ <sup>i</sup>k. By Claim [C.6,](#page-24-3) <sup>k</sup> ≤ <sup>2</sup><sup>M</sup> − <sup>1</sup>.

MINIMALADVERSARY:

send xi<sup>1</sup> , xi<sup>2</sup> , . . . , xi<sup>k</sup> to the learner for t = 1, 2, . . . , n: if <sup>t</sup> ∈ <sup>I</sup>: receive prediction yˆ<sup>t</sup> from learner send prediction yˆ<sup>t</sup> to A<sup>1</sup> receive label y<sup>t</sup> from A<sup>1</sup> send label y<sup>t</sup> to learner else: send prediction yˆ<sup>t</sup> = 0 to A<sup>1</sup> receive label y<sup>t</sup> from A<sup>1</sup>

Algorithm 4: Construction of an adversary that forces M mistakes using a sequence x of length at most <sup>2</sup><sup>M</sup> − <sup>1</sup>. In the proof of Theorem [C.2,](#page-21-2) this adversary is <sup>A</sup>2. Internally, it simulates a rigid adversary A1.

## D Upper Bound

#### D.1 Statement

The following result states that the lower bound of Theorem [B.1](#page-2-0) is tight for some classes.

Theorem D.1 (Upper bound, and separation between standard and transductive online learning). *For every integer* <sup>d</sup> ≥ <sup>43</sup>*, there exists a hypothesis class* H ⊆ {0, <sup>1</sup>} <sup>X</sup> *with a domain* X *of size* |X | = 2<sup>d</sup> − <sup>1</sup> *such that* LD(H) = <sup>d</sup> *and the following two conditions hold for all* <sup>n</sup> ∈ <sup>N</sup>*:*

- *1.* <sup>M</sup>tr(H, n) ≤ <sup>48</sup> · √
  - d*.*
- *2.* <sup>M</sup>std(H, n) = min {n, d}*.*

#### D.2 Hypothesis Class

In this section we construct the hypothesis class for Theorem [D.1.](#page-2-0)

Lemma D.2. *Let* <sup>d</sup> ∈ <sup>N</sup>*,* <sup>d</sup> ≥ <sup>42</sup>*. Let* <sup>T</sup><sup>d</sup> *be a perfect binary tree of depth* <sup>d</sup>*, as in Definition [A.4.](#page-14-1) Then there exists a collection of functions* H ⊆ {0, <sup>1</sup>} <sup>T</sup><sup>d</sup> *such that* LD(H) = <sup>d</sup> + 1 *and the following two conditions hold for all* <sup>H</sup> ⊆ H *and all* <sup>X</sup> ⊆ <sup>T</sup>d*:*

- *1. If* ∀<sup>h</sup> ∈ <sup>H</sup> ∀<sup>x</sup> ∈ <sup>X</sup> : x /∈ path(h) ∧ <sup>h</sup>(x) = 0*, then* min {|H|, |X|} <sup>&</sup>lt; <sup>2</sup> 2 √ d *.*
- *2. If* ∀<sup>h</sup> ∈ <sup>H</sup> ∀<sup>x</sup> ∈ <sup>X</sup> : x /∈ path(h) ∧ <sup>h</sup>(x) = 1*, then* |H| <sup>&</sup>lt; <sup>2</sup> 2 √ <sup>d</sup> *or* |X| <sup>&</sup>lt; <sup>3</sup> √
  - d*.*

The proof employs the probabilistic method, showing that a hypothesis class sampled randomly from a suitable distribution has the desired properties with very high probability.

*Proof.* Let P be a probability distribution over hypothesis classes. Formally, P ∈ ∆ ({0, <sup>1</sup>} <sup>T</sup><sup>d</sup> ) 2 <sup>d</sup>+1 is a distribution over vectors of hypotheses. Each vector H ∈ supp(P) consists of 2 <sup>d</sup>+1 hypotheses,

$$\mathcal{H} = (h_b)_{b \in \{0,1\}^{d+1}},$$

where for each <sup>b</sup> ∈ {0, <sup>1</sup>} <sup>d</sup>+1, hypothesis <sup>h</sup><sup>b</sup> is a function <sup>h</sup><sup>b</sup> : <sup>T</sup><sup>d</sup> → {0, <sup>1</sup>} sampled independently as follows:

- For each <sup>i</sup> ∈ [d] ∪ {0}: <sup>h</sup>b(b≤i) = <sup>b</sup>i+1. (In particular, with probability <sup>1</sup>, path(hb) = (b≤0, b≤1, . . . , b≤d), each entry in the vector H is unique, and H shatters <sup>T</sup>d.)
- For each <sup>x</sup> ∈ <sup>T</sup>d\path(hb), the bit <sup>h</sup>b(x) ∈ {0, <sup>1</sup>} is sampled Ber 2 − √ d independently of all other bits in H, i.e., <sup>P</sup>[hb(x) = 1] = <sup>P</sup>[hb(x) = 1 | {<sup>h</sup><sup>b</sup> ′}b ′̸=b, {<sup>h</sup>b(<sup>x</sup> ′ )}x′̸=x] = 2<sup>−</sup> √ d .

In words, for all nodes on the path in the tree corresponding to b, the function h<sup>b</sup> assigns a label according to b, and for all other nodes, h<sup>b</sup> assigns a label of 1 with probability 2 − √ d , and a label of 0 otherwise. In particular, the collection H Littlestone-shatters the tree <sup>T</sup>d.

Fix <sup>B</sup> ⊆ {0, <sup>1</sup>} <sup>d</sup>+1 and <sup>X</sup> ⊆ <sup>T</sup>d, and let <sup>E</sup>(B, X, y) denote the event

$$\{\forall b \in B \forall x \in X : x \notin \text{path}(h_b) \wedge h_b(x) = y\}. \quad (12)$$

Seeing as each off-path label <sup>h</sup>b(x) ∈ {0, <sup>1</sup>} is sampled independently,

$$\begin{aligned}\mathbb{P}_{\mathcal{H} \sim \mathcal{P}}[E(B, X, 0)] &= \prod_{(b,x) \in B \times X} \mathbb{P}_{\mathcal{H} \sim \mathcal{P}}[x \notin \text{path}(h_b) \wedge h_b(x) = 0] \\ &\leq (1 - 2^{-\sqrt{d}})^{|B \times X|}.\end{aligned}\tag{13}$$

Hence,

$$\begin{aligned} & \mathbb{P}_{\mathcal{H} \sim \mathcal{P}} \left[ \exists B \subseteq \{0, 1\}^{d+1} \exists X \subseteq T_d : E(B, X, 0) \wedge \min \{|B|, |X|\} \geq 2^{2\sqrt{d}} \right] \\ & = \mathbb{P}_{\mathcal{H} \sim \mathcal{P}} \left[ \exists B \subseteq \{0, 1\}^{d+1} \exists X \subseteq T_d : E(B, X, 0) \wedge |B| = |X| = \left\lceil 2^{2\sqrt{d}} \right\rceil \right] \\ & \leq \binom{\left\lfloor \{0, 1\}^{d+1} \right\rfloor}{\left\lceil 2^{2\sqrt{d}} \right\rceil} \binom{\left\lfloor T_d \right\rfloor}{\left\lceil 2^{2\sqrt{d}} \right\rceil} (1 - 2^{-\sqrt{d}})^{2^{4\sqrt{d}}} \quad (\text{union bound, Eq. (13)}) \\ & < \binom{2^{d+1}}{2^{2\sqrt{d}} + 1}^2 \cdot (1 - 2^{-\sqrt{d}})^{2^{4\sqrt{d}}} \\ & < 2^{2 \cdot (d+1) \cdot (2^{2\sqrt{d}} + 1)} \cdot e^{-2^{-\sqrt{d}} \cdot 2^{4\sqrt{d}}} \\ & < 2^{2 \cdot (d+2) \cdot 2^{2\sqrt{d}}} \cdot 2^{-2^{-\sqrt{d}} \cdot 2^{4\sqrt{d}}} \\ & = 2^{2^{2\sqrt{d}} \cdot (2d+4-2^{2\sqrt{d}})} \\ & < 2^{-2^{2\sqrt{d}}}. \\ & < (2d+4 - 2^{\sqrt{d}}) < -1 \text{ for } d \geq 42 \\ & (14) \end{aligned}$$

Similarly,

$$\mathbb{P}_{\mathcal{H} \sim \mathcal{P}}[\forall b \in B \ \forall x \in X : x \notin \text{path}(h_b) \ \wedge \ h_b(x) = 1] \leq 2^{-\sqrt{d} |B \times X|}, \quad (15)$$

$$\mathbb{P}_{\mathcal{H} \sim \mathcal{P}} \left[ \exists B \subseteq \{0,1\}^{d+1} \exists X \subseteq T_d : E(B, X, 0) \wedge |H| \geq 2^{2\sqrt{d}} \wedge |X| \geq 3\sqrt{d} \right]$$

$$\begin{aligned} &\leq \left( \binom{|\{0,1\}^{d+1}|}{\lceil 2\sqrt{d} \rceil} \binom{|T_d|}{\lceil 3\sqrt{d} \rceil} \cdot 2^{-\sqrt{d} \cdot 2^{2\sqrt{d}} \cdot 3\sqrt{d}} \right) \quad (\text{union bound, Eq. (15)}) \\ &\leq \left( \frac{2^{d+1}}{2^{2\sqrt{d}} + 1} \right) \left( \frac{2^{d+1}}{3\sqrt{d} + 1} \right) \cdot 2^{-3d \cdot 2^{2\sqrt{d}}} \\ &< 2^{(d+1) \cdot (2^{2\sqrt{d}} + 1)} \cdot 2^{(d+1) \cdot (3\sqrt{d} + 1)} \cdot 2^{-3d \cdot 2^{2\sqrt{d}}} \\ &< 2^{(d+1) \cdot (2^{2\sqrt{d}} + 3\sqrt{d} + 2)} \cdot 2^{-3d \cdot 2^{2\sqrt{d}}} \\ &< 2^{2d \cdot 2^{2\sqrt{d}}} \cdot 2^{-3d \cdot 2^{2\sqrt{d}}} \quad (\text{for } d \geq 4) \\ &< 2^{-d2^{\sqrt{d}}}. \quad (16) \end{aligned}$$

Applying a union bound to Eqs. [\(14\)](#page-26-2) and [\(16\)](#page-27-0) gives

$$\mathbb{P}_{\mathcal{H} \sim \mathcal{P}}[\mathcal{H} \text{ satisfies Items 1 and 2}] \geq 1 - 2^{-2^{2\sqrt{d}}} - 2^{-d2^{\sqrt{d}}} \geq 1 - 10^{-100}.$$

In particular, there exists a collection H that satisfies Items [<sup>1</sup>](#page-25-2) and [2.](#page-25-3) Furthermore, this collection has LD(H) = <sup>d</sup> + 1 (namely, LD(H) ≥ <sup>d</sup> + 1 because it shatters <sup>T</sup>d; and LD(H) ≤ <sup>d</sup> + 1 because |H| = 2<sup>d</sup>+1).

#### D.3 Algorithm

In this section we describe Algorithms [5,](#page-29-0) [6a,](#page-30-0) and [6c,](#page-31-0) which together constitute the learning algorithm that achieves the O √ d mistake upper bound in the transductive setting, as in Theorem [D.1.](#page-2-0) See Section [2.3](#page-5-5) for a general overview of these algorithms.

#### D.3.1 How Experts Work

We start with some preliminary remarks about experts in Algorithms [5,](#page-29-0) [6a,](#page-30-0) and [6c.](#page-31-0)

Experts. A tuple e = (S, u, H) defines an expert that can make predictions using the procedure <sup>E</sup>XPERT.PREDICT(e, ·). The tuple <sup>e</sup> reflects two kinds of information:

- 1. *Knowledge.* Information that the expert *knows* with certainty. Specifically, this reflects the labels y1, y2, . . . sent by the adversary so far. All experts see the labels sent by the adversary, so this knowledge is the same for all experts.
- 2. *Assumptions.* At certain times, experts make *assumptions* about things that are not known for certain. Specifically, experts assume that certain nodes <sup>x</sup> are on-path (<sup>x</sup> ∈ path(h)) or off-path (x /∈ path(h)) with respect to the correct labeling function <sup>h</sup> : <sup>T</sup><sup>d</sup> → {0, <sup>1</sup>}. Assumptions are simply guesses that may be wrong, and therefore when an expert needs to make such an assumption, it splits into two experts (as described below), with one expert assuming <sup>x</sup> ∈ path(h), and the other expert assuming x /∈ path(h). This ensures that there always exists an expert for which all assumptions are correct.

In greater detail, the contents of the state tuple e = (S, u, H) represents the knowledge and assumptions of the expert as follows:

- <sup>u</sup> ∈ <sup>T</sup><sup>d</sup> This single node encodes everything the expert knows and assumes about which of the nodes labeled so far are on-path. Observe that if <sup>v</sup>1, v2, . . . , v<sup>k</sup> ∈ <sup>T</sup><sup>d</sup> are nodes that are assumed to be on-path (and all these assumptions are consistent), then these k assumptions can be represented succinctly by assigning u = v<sup>i</sup> <sup>∗</sup> where v<sup>i</sup> <sup>∗</sup> is the deepest node among v1, v2, . . . , vk. Therefore, u simply holds the deepest node in the tree that is known or assumed to be on-path. At the start of the algorithm, this value is initialized to be u = λ, because the root is known to be on-path regardless of the target function.
- <sup>S</sup> ⊆ <sup>T</sup><sup>d</sup> the 'danger zone', as described in Section [2.3.4.](#page-7-1) This is a collection that contains all nodes in the prefix x≤tmax = (x1, x2, . . . , x<sup>t</sup>max ) of the sequence to be classified that have not been labeled yet and *might* be on-path for the true labeling function h given what

the expert knows and assumes so far. However, S is not required to contain ancestors of nodes that are assumed to be on-path. Initially, S equals the prefix x≤tmax . As information accumulates, nodes that cannot be on-path are removed from <sup>S</sup>. For instance, if <sup>x</sup><sup>i</sup> ∈ <sup>T</sup><sup>d</sup> is assigned label <sup>y</sup><sup>i</sup> ∈ {0, <sup>1</sup>} by the adversary, then any (1 − <sup>y</sup>i)-descendant of <sup>x</sup><sup>i</sup> (including xi itself) may safely be removed from S.

- <sup>H</sup> ⊆ {0, <sup>1</sup>} <sup>T</sup><sup>d</sup> – the version space of the experts, i.e., the collection of all functions that could be the correct labeling function given everything that the expert knows and assumes. Initially, <sup>H</sup> contains all functions in H. As information accumulates, some functions are ruled out. Specifically, a function h can be removed from H for two reasons: (i) the adversary assigns a label <sup>y</sup> ̸<sup>=</sup> <sup>h</sup>(x) to some node <sup>x</sup> ∈ <sup>T</sup>d; (ii) the expert makes an assumption that some <sup>x</sup> ∈ <sup>T</sup><sup>d</sup> is on-path for the correct labeling function but x /∈ path(h), or vice versa, the expert assumes that <sup>x</sup> is off-path for the correct labeling function but <sup>x</sup> ∈ path(h).

Updates and splits. An expert can be modified using the procedure <sup>E</sup>XPERT.EXTENDEDUPDATE(e, ·, ·). This procedure either returns a single modified tuple (S, u, H) (in the first two return statements in the procedure), in which case we think of the expert as being *updated*; or alternatively, the procedure returns two tuples e<sup>∈</sup> = (S∈, u∈, H∈) and e∈/ = (S∈/, u∈/, H∈/) (in the third return [statement\)](#page-31-1), in which case we think of the expert as being *split* into two experts. The expert e<sup>∈</sup> corresponds to adding an assumption that the most recently presented node x<sup>t</sup> is on-path for the correct labeling function, and e∈/ corresponds to adding the opposite assumption.

Ancestry. At the end of each iteration of the outer 'for' loop in Algorithm [5,](#page-29-0) for each expert <sup>e</sup> ∈ <sup>E</sup>t+1 there exists a unique *ancestry* sequence ancestry(e) = (e1, e2, . . . , et+1) such that <sup>e</sup><sup>1</sup> = ({<sup>x</sup>1, . . . , x<sup>t</sup>max}, λ, H) is the initial single expert that was created before the start of the outer 'for' loop, <sup>e</sup>t+1 <sup>=</sup> <sup>e</sup> is the latest version of the expert, and for each <sup>i</sup> ∈ [t], the expert <sup>e</sup>i+1 was created by an execution of EXPERT.BASICUPDATE(e<sup>i</sup> , ·, ·) possibly followed by an execution of EXPERT.EXTENDEDUPDATE. [23](#page-28-0)

#### D.4 Analysis

In this section we prove our main result, Theorem [D.1.](#page-2-0)

#### D.4.1 Assumption-Consistent Expert

Occasionally, when an expert is updated, it makes an assumption about whether the most-recently presented node x<sup>t</sup> is on-path or off-path with respect to the true labeling function h. In these updates, the expert is split into two: one expert assumes that <sup>x</sup><sup>t</sup> ∈ path(h), and the other assumes <sup>x</sup><sup>t</sup> ∈/ path(h). Clearly, by splitting into two in this manner, we preserve the invariant that the set of experts always contains a 'vindicated' expert e ∗ such that all the assumptions made by e ∗ are correct. This simple observation is made formal in the following definition and claim.

Definition D.3 (Assumption consistency). *For an expert* <sup>e</sup> ∈ <sup>E</sup>t+1 *with* ancestry(e) = (e1, e2, . . . , et+1)*, and an index* <sup>i</sup> ∈ [t]*, we say that the* <sup>i</sup> → (<sup>i</sup> + 1) *update of* <sup>e</sup> *was assumption-consistent with a function* <sup>h</sup> : <sup>T</sup><sup>d</sup> → {0, <sup>1</sup>} *if one of the following conditions holds:*

- ei+1 = EXPERT.BASICUPDATE(e<sup>i</sup> , x<sup>i</sup> , yi)*; or*

<sup>23</sup>Note that in this paper, we use genealogical metaphors in two distinct contexts that should not be confused. First, as is customary, we use "child", "parent", "ancestor" and "descendant" to describe relations between nodes in the binary tree Td, which constitutes the domain of our hypothesis class. Separately from that, we use "ancestor" and "descendant" to describe relations between experts.

This overlap in terminology can partially be excused by the fact that the history of experts also forms a binary tree. Indeed, initially there is a single expert (the root of the tree), and experts can split into two, corresponding to a node having two children as in a binary tree. Seeing as experts cannot merge, the expert history corresponds precisely to a binary tree. (However, the domain T<sup>d</sup> is a *perfect* binary tree, whereas the binary tree corresponding to expert genealogy need not be balanced).

To reduce confusion, we use path(·) only for nodes in Td, and ancestry(·) only for experts, even though these operators are mathematically equivalent (however, path(·) is defined not only for nodes in T<sup>d</sup> but also for functions T<sup>d</sup> → {0, 1}).

- • d, n ∈ <sup>N</sup>, <sup>λ</sup> is the empty string.
- H ⊆ {0, <sup>1</sup>} <sup>T</sup><sup>d</sup> is the class that exists by Lemma [D.2.](#page-25-0)
- <sup>x</sup>1, x2, . . . , x<sup>n</sup> ∈ <sup>T</sup><sup>d</sup> are points to be classified.

<sup>T</sup>RANSDUCTIVELEARNER(H, <sup>d</sup>, (x1, x2, . . . , xn)):

| $t \leftarrow 0, t_{\max} \leftarrow 2^{4\sqrt{d}}$               |                                                                 |
|-------------------------------------------------------------------|-----------------------------------------------------------------|
| $e \leftarrow \{x_1, \dots, x_{t_{\max}}\}, \lambda, \mathcal{H}$ | ▷ The initial expert. An expert is defined by a 3-tuple.        |
| $w(e) \leftarrow 1$                                               | ▷ Assign the initial expert a weight of 1.                      |
| $e \in \{e\}$                                                     | ▷ $E_t$ is the set of experts used for predicting $\hat{y}_t$ . |
| $E_{2^{2^{1/2}}}, E_{2^{1/2}}, E_{2^{1/4}} \leftarrow \emptyset$  |                                                                 |

$$E_1 \leftarrow \{e_j \mid E_2, \dots, E_n, E_{n+1} \leftarrow \emptyset \}$$

**for** 
$$t \leftarrow 1, 2, \dots, n$$
:

$$\hat{y}_t \leftarrow \mathbf{1} \left( \sum_{e \in E_t} w(S) \cdot \text{EXPERT.PREDICT}(e, x_t) \geq \frac{1}{2} \right) \quad \triangleright$$

A weighted majority, using

Algorithm [6a.](#page-30-0)

send prediction yˆ<sup>t</sup> to adversary

receive correct label <sup>y</sup><sup>t</sup> ∈ {0, <sup>1</sup>} from adversary

for <sup>e</sup> ∈ <sup>E</sup>t: <sup>▷</sup> Update the experts.

<sup>e</sup> ← <sup>E</sup>XPERT.BASICUPDATE(e, xt, yt) <sup>▷</sup> Remove functions that disagree with

the label y<sup>t</sup> from the version space.

**if** EXPERT.PREDICT(
$$e, x_t$$
) =  $y_t$ :      ▷ If expert  $e$  made a correct prediction, no further update is needed.

else:

<sup>U</sup> ← <sup>E</sup>XPERT.EXTENDEDUPDATE(e, xt, yt) <sup>▷</sup> If <sup>e</sup> made a mistake,

update e using Algorithm [6c.](#page-31-0) This might cause e to be split into

two experts.

for e

′ ∈ <sup>U</sup>:

<sup>E</sup>t+1 ← <sup>E</sup>t+1 ∪ {<sup>e</sup>

′} <sup>▷</sup> Add updated expert(s) to <sup>E</sup>t+1.

w(e ′ ) ← <sup>w</sup>(e)/(2 · |U|) <sup>▷</sup> When <sup>e</sup> makes a mistake, its weight

is decreased by a factor of 2 and then split equally between its descendants.

Algorithm 5: A transductive online learning algorithm that makes at most O √ d mistakes. It is a variant of the multiplicative weights algorithm that employs splitting experts. Namely, we start with a single expert, and when an expert makes a mistake it may split into two experts. The behavior of the experts is defined in Algorithms [6a](#page-30-0) and [6c.](#page-31-0)

- ei+1 *was the single expert returned when executing* EXPERT.EXTENDEDUPDATE(e ′ i , x<sup>i</sup> , yi) *for* e ′ <sup>i</sup> = EXPERT.BASICUPDATE(e<sup>i</sup> , x<sup>i</sup> , yi)*; or*
- *Executing* EXPERT.EXTENDEDUPDATE(e ′ i , x<sup>i</sup> , yi) *with* e ′ <sup>i</sup> = EXPERT.BASICUPDATE(e<sup>i</sup> , x<sup>i</sup> , yi) *returned two experts* (S∈, u∈, H∈) *and* (S∈/, u∈/, H∈/) *(as in the third return [statement\)](#page-31-1), and furthermore,*

$$e_{i+1} = \begin{cases} (S_{\in}, u_{\in}, H_{\in}) & x_i \in \text{path}(h) \\ (S_{\notin}, u_{\notin}, H_{\notin}) & x_i \notin \text{path}(h). \end{cases} \quad (17)$$

- • <sup>d</sup> ∈ <sup>N</sup>, <sup>x</sup> ∈ <sup>T</sup>d.
- e = (S, u, H) is a tuple that defines an expert:
  - <sup>S</sup> ⊆ <sup>T</sup><sup>d</sup> a collection of nodes that could be on-path for the true labeling function given what the expert knows and assumes.
  - <sup>u</sup> ∈ <sup>T</sup><sup>d</sup> the deepest node known or assumed to be on-path by the expert.
  - <sup>H</sup> ⊆ {0, <sup>1</sup>} <sup>T</sup><sup>d</sup> – the collection of all functions that could be the correct labeling function given what the expert knows and assumes.

EXPERT.PREDICT(e, x):

(S, u, H) ← e ▷ Unpack the state that defines the expert.

if |H| ≤ <sup>2</sup>

2 √ d :

return HALVING.PREDICT(H, x) ▷ Once H becomes small enough, simu-

late the Halving algorithm (Algorithm [7\)](#page-32-0).

[Case [I\]](#page-35-0)

if x ≼ u:

return <sup>b</sup> ∈ {0, <sup>1</sup>} such that <sup>x</sup> <sup>≼</sup><sup>b</sup> u ▷ u is assumed to be on-path. If <sup>u</sup> is a <sup>b</sup>-

decendant of x, then the correct label for x

must be b. [Case [II\]](#page-35-1)

return <sup>1</sup>(|{<sup>x</sup>

′ ∈ <sup>S</sup> : <sup>x</sup> <sup>≼</sup><sup>1</sup> <sup>x</sup>

′

}| <sup>&</sup>gt; |S|/3) <sup>▷</sup> Output some <sup>b</sup> ∈ {0, <sup>1</sup>} such that more

than 1/3 of suspected on-path nodes are b-decendants of x, if such a b exists. Otherwise (when at least 1/3 of S are nondescendants of x), output 0. [Cases [III](#page-35-2)

to [VI\]](#page-36-0)

Algorithm 6a: A subroutine of Algorithm [5](#page-29-0) that defines how an expert makes predictions.

### Assumptions:

- x, e, S, u, H as in Algorithm [6a.](#page-30-0)
- y the correct label for x, as selected by the adversary.

EXPERT.BASICUPDATE(e, x, y):

(S, u, H) ← e ▷ Unpack the state that defines the expert.

<sup>H</sup> ← <sup>H</sup>ALVING.UPDATE(H, x, y) <sup>▷</sup> Update the version space, as in the Halving

algorithm (Algorithm [7\)](#page-32-0).

return (S, u, H)

Algorithm 6b: A subroutine of Algorithm [5](#page-29-0) that defines how an expert is updated each time that a label is selected by the adversary.

*We say that an expert* <sup>e</sup> ∈ <sup>E</sup>t+1 *is assumption-consistent with* <sup>h</sup> *if for all* <sup>i</sup> ∈ [t]*, the* <sup>i</sup> → (<sup>i</sup> + 1) *update of* e *was assumption-consistent with* h*.*

Claim D.4 (Existence of assumption-consistent expert). *Let* d, n, t ∈ <sup>N</sup>*,* <sup>t</sup> ≤ <sup>n</sup>*, let* H ⊆ {0, <sup>1</sup>} <sup>T</sup><sup>d</sup> *, let* <sup>x</sup>1, . . . , x<sup>n</sup> ∈ <sup>T</sup>d*, and let* <sup>h</sup> : <sup>T</sup><sup>d</sup> → {0, <sup>1</sup>}*. Consider an execution of*

TRANSDUCTIVELEARNER(
$$\mathcal{H}, d, (x_1, x_2, \dots, x_n)$$
)

*as in Algorithm [5.](#page-29-0) Then, at the end of the* t*-th iteration of the outer 'for' loop in* TRANSDUCTIVE-LEARNER*, there exists a unique expert* e ∗ <sup>t</sup>+1 ∈ <sup>E</sup>t+1 *that is assumption-consistent with* <sup>h</sup>*.*

- • d, x, e, S, u, H – as in Algorithm [6a.](#page-30-0)
- y the correct label for x, as selected by the adversary.

EXPERT.EXTENDEDUPDATE(e, x, y):

(S, u, H) ← e ▷ Unpack the state that defines the expert.

if |H| ≤ <sup>2</sup>

2 √ d

: ▷ If the version space is small, we just simu-

late the Halving algorithm, so the update is

complete. [Case [III\]](#page-35-2)

return {(S, u, H)}

for <sup>b</sup> ∈ {0, <sup>1</sup>}: <sup>S</sup><sup>b</sup> ← {<sup>x</sup>

′ ∈ <sup>S</sup> : <sup>x</sup> <sup>≼</sup><sup>b</sup> <sup>x</sup>

′} <sup>▷</sup> Set of suspected on-path nodes that are <sup>b</sup>-

descendant of x.

if |<sup>S</sup>(1−y)

| <sup>&</sup>gt; |S|/3:

S

′ ← <sup>S</sup> \ <sup>S</sup>(1−y) <sup>▷</sup> At least <sup>1</sup>/<sup>3</sup> of suspected on-path nodes were <sup>b</sup>-

decendants of x, and therefore the expert predicted label <sup>y</sup><sup>ˆ</sup> <sup>=</sup> <sup>b</sup>. But the correct label was <sup>y</sup> = 1 − <sup>b</sup>. Remove all b-descendants of x from S. [Case [IV\]](#page-35-3)

return {(<sup>S</sup>

′ , u, H)}

#### else:

<sup>S</sup>∈/ ← <sup>S</sup>; <sup>u</sup>∈/ ← u ▷ Split <sup>e</sup> in two. First, construct <sup>e</sup>∈/ to be an

updated version of e after adding the assumption that x /∈ path(h) for the correct label-

ing function h.

<sup>H</sup>∈/ <sup>=</sup> {<sup>h</sup> ∈ <sup>H</sup> : x /∈ path(h)}

<sup>e</sup>∈/ ← (S∈/, u∈/, H∈/)

<sup>S</sup><sup>∈</sup> ← <sup>S</sup><sup>0</sup> ∪ <sup>S</sup><sup>1</sup> <sup>▷</sup> Next, construct <sup>e</sup><sup>∈</sup> to be an updated version

of <sup>e</sup> adding the assumption <sup>x</sup> ∈ path(h). S<sup>∈</sup> contains only nodes that are descendants

of x. <sup>u</sup><sup>∈</sup> ← u ▷ u<sup>∈</sup> represents updating the prior assumption

that u is on path by adding that x is also on

path.

if u<sup>∈</sup> ≼ x: <sup>u</sup><sup>∈</sup> ← <sup>x</sup>

<sup>H</sup><sup>∈</sup> <sup>=</sup> {<sup>h</sup> ∈ <sup>H</sup> : <sup>x</sup> ∈ path(h)} ▷ H<sup>∈</sup> is obtained by updating the version

space to include only function where x is

on path.

<sup>e</sup><sup>∈</sup> ← (S∈, u∈, H∈)

return {<sup>e</sup>∈/, e∈} <sup>▷</sup> [Cases [V](#page-35-4) and [VI\]](#page-36-0)

Algorithm 6c: A subroutine of Algorithm [5](#page-29-0) that defines how an expert is updated (and possibly split into two) when it makes a mistake.

*Proof.* We prove by induction that, for all <sup>s</sup> ∈ [<sup>t</sup> + 1], <sup>E</sup><sup>s</sup> contains a unique expert that is assumptionconsistent with h. The base case s = 1 is clear, because E<sup>1</sup> contains only a single expert that was never modified. For the induction step, let e ∗ <sup>s</sup> be the unique assumption-consistent expert in Es, and consider the <sup>s</sup> → (<sup>s</sup> + 1) update. Notice that by Definition [D.3,](#page-22-4)

- For all <sup>e</sup> ∈ <sup>E</sup><sup>s</sup> \ {<sup>e</sup> ∗ <sup>s</sup>}, every expert <sup>e</sup> ′ ∈ <sup>E</sup>s+1 such that <sup>e</sup> ′ was created from e by executing EXPERT.BASICUPDATE(es, xs, ys) possibly followed by an execution of EXPERT.EXTENDEDUPDATE is not assumption-consistent with h; and
- Either EXPERT.BASICUPDATE(e ∗ s , xs, ys) ∈ <sup>E</sup>s+1 and EXPERT.EXTENDEDUPDATE(e ∗ s , xs, ys) is not executed (e ∗ s is added to Es+1 with just a basic update), or precisely one of the experts that were created from e ∗ <sup>s</sup> by executing EXPERT.EXTENDEDUPDATE and added to Es+1 is assumption-consistent with h.

- • X a set, <sup>k</sup> ∈ <sup>N</sup>.
- H ⊆ {0, <sup>1</sup>} <sup>X</sup> is a finite hypothesis class.
- x, x1, . . . , x<sup>k</sup> ∈ X , <sup>y</sup> ∈ {0, <sup>1</sup>}.

<sup>H</sup>ALVING(H, (x1, x2, . . . , xk)):

H<sup>1</sup> ← H for <sup>i</sup> ∈ [k]:

<sup>y</sup>ˆ<sup>i</sup> ← <sup>H</sup>ALVING.PREDICT(H, xi)

send prediction yˆ<sup>i</sup>

to adversary receive correct label <sup>y</sup><sup>i</sup> ∈ {0, <sup>1</sup>} from adversary

Hi+1 ← <sup>H</sup>ALVING.UPDATE(H<sup>i</sup>

, x<sup>i</sup> , yi)

<sup>H</sup>ALVING.PREDICT(H, <sup>x</sup>):

return 1

 |H| P

<sup>h</sup>∈H <sup>h</sup>(x) ≥

2 

<sup>H</sup>ALVING.UPDATE(H, <sup>x</sup>, <sup>y</sup>):

return

<sup>h</sup> ∈ H : <sup>h</sup>(x) = <sup>y</sup>

Algorithm 7: This is the well-known halving algorithm. The experts in Algorithms [6a](#page-30-0) and [6c](#page-31-0) simulate this algorithm once their version space becomes small enough.

Seeing as the <sup>s</sup> → (<sup>s</sup> + 1) update executes <sup>E</sup>XPERT.BASICUPDATE and <sup>E</sup>XPERT.EXTENDEDUPDATE at most once for each <sup>e</sup> ∈ <sup>E</sup>s, it follows that <sup>E</sup>s+1 contains precisely one expert that is assumption-consistent with h.

An expert e = (S, u, H) that is assumption-consistent with the correct labeling function enjoys two simple properties. The first property is that the node u in the expert encodes correct information about which previously seen nodes are on-path for the correct labeling function.

The second property is that the set S contains all future nodes that are on-path for the correct labeling function and are also deeper in the tree than all nodes assumed to be on-path so far. These two properties are formalized in the following claim.

Claim D.5 (Properties of assumption-consistent expert). *Let* d, n, t ∈ <sup>N</sup>*,* <sup>t</sup> ≤ <sup>n</sup>+ 1*, let* H ⊆ {0, <sup>1</sup>} <sup>T</sup><sup>d</sup> *, let* <sup>x</sup>1, . . . , x<sup>n</sup> ∈ <sup>T</sup>d*. Consider an execution of*

TRANSDUCTIVELEARNER(
$$\mathcal{H}, d, (x_1, x_2, \dots, x_n)$$
)

*as in Algorithm [5.](#page-29-0) Assume that the adversary selects labels* <sup>y</sup>1, y2, . . . , y<sup>n</sup> ∈ {0, <sup>1</sup>} *that are consistent with some function* <sup>h</sup> : <sup>T</sup><sup>d</sup> → [{](#page-32-1)0, <sup>1</sup>}*. Let* <sup>e</sup> ∗ <sup>t</sup> = (S ∗ t , u<sup>∗</sup> t , H<sup>∗</sup> t ) ∈ <sup>E</sup><sup>t</sup> *be the unique expert in* <sup>E</sup><sup>t</sup> *that is assumption-consistent with* h*.* <sup>24</sup> *Then the following two properties hold:*

- *1.* u ∗ <sup>t</sup> ∈ path(h)*.*
- *2.* {<sup>x</sup> ∈ {<sup>x</sup>t, xt+1, . . . , x<sup>t</sup>max } : <sup>x</sup> ∈ path(h) ∧ <sup>x</sup> ̸<sup>≼</sup> <sup>u</sup> ∗ <sup>t</sup> } ⊆ <sup>S</sup> ∗ t *.*

*Proof of Claim [D.5.](#page-19-3)* The proof proceeds by induction on t. For the base case t = 1, E<sup>1</sup> contains a single expert e ∗ <sup>1</sup> = (S ∗ 1 , u<sup>∗</sup> 1 , H<sup>∗</sup> 1 ) where u ∗ <sup>1</sup> <sup>=</sup> <sup>λ</sup> is the root of <sup>T</sup>d. Indeed, <sup>λ</sup> ∈ path(h) for

<sup>24</sup>Recall that e ∗ <sup>t</sup> exists by Claim [D.4.](#page-19-0)

any function  $h : T_d \rightarrow \{0, 1\}$ . This establishes the base case for Item 1. Additionally,  $S_1^* = \{x_1, x_2, \dots, x_{t_{\max}}\}$ , satisfying the base case for Item 2.

For the induction step, we assume that the claim holds for some integer  $t = i$ , and show that it holds for  $t = i + 1$  as well. First, we establish Item 1. If  $e_{i+1}^* = \text{EXPERT.BASICUPDATE}(e_i^*, x_i, y_i)$ , then the claim is immediate because  $u_{i+1}^* = u_i^* \in \text{path}(h)$ . Otherwise, by Definition D.3 and the first first two return statements in EXPERT.EXTENDEDUPDATE, either  $e_{i+1}^* = (S_{i+1}^*, u_{i+1}^*, H_{i+1}^*)$  has  $u_{i+1}^* = u_i^* \in \text{path}(h)$ , in which case the claim is immediate, or else  $e_{i+1}^*$  satisfies Eq. (17), namely,

$$e_{i+1}^* = \begin{cases} (S_{i+1}^*, u_{i+1}^*, H_{i+1}^*) & x_i \in \text{path}(h) \\ (S_{i+1}^*, u_{i+1}^*, H_{i+1}^*) & x_i \notin \text{path}(h). \end{cases}$$

As defined in EXPERT.EXTENDEDUPDATE,  $u_{i+1}^*$  is equal either to  $u_i^*$  or to  $x_i$ , so if  $x_i \in \text{path}(h)$  then

$$u_{i+1}^* = u_i \in \{u_i^*, x_i\} \subseteq \text{path}(h).$$

On the other hand, if  $x_i \notin \text{path}(h)$  then we get  $u_{i+1}^* = u_{i+1}^* = u_i^* \in \text{path}(h)$ . We see that in all cases,  $u_{i+1}^* \in \text{path}(h)$  as desired. This concludes the proof of Item 1.

For Item 2, again, if  $e_{i+1}^* = \text{EXPERT.BASICUPDATE}(e_i^*, x_i, y_i)$ , then the claim is immediate because  $S_{i+1}^* = S_i^*$  and  $u_{i+1}^* = u_i^*$ . Otherwise, consider the various ways in which  $u_{i+1}^*$  and  $S_{i+1}^*$  can be assigned by EXPERT.EXTENDEDUPDATE. In the first return statement,  $u_{i+1}^* = u_i^*$  and  $S_{i+1}^* = S_i^*$ , and the claim is immediate.

The second return statement assigns  $u_{i+1}^* = u_i^*$  and  $S_{i+1}^* = S_i^* \setminus S_{1-y_i}$ , where  $S_{1-y_i}$  is the set of  $(1-y_i)$ -descendants of  $x_i$  (including  $x_i$  itself). Notice that regardless of whether  $x_i$  is on-path for the correct labeling function  $h$  or not, none of the  $(1-y_i)$ -descendants of  $x_i$  (except possibly  $x_i$  itself) can be on-path for  $h$ , because  $h$  assigns a label  $y_i$  to  $x_i$ . And seeing as Item 2 only requires that  $S_{i+1}^*$  contain nodes from  $\{x_{i+1}, x_{i+2}, \dots, x_{t_{\max}}\}$ , it is also safe to remove  $x_i$ . Therefore, removing  $S_{1-y_i}$  preserves Item 2.

For the third return statement, there are two possibilities. The first possibility is that  $u_{i+1}^* = u_{i+1}^* = u_{i+1}^*$  and  $S_{i+1}^* = S_{i+1}^* = S_{i+1}^*$ , in which case the claim is immediate. The second possibility assigns  $u_{i+1}^* = u_{i+1}^*$ , and  $S_{i+1}^* = S_{i+1}^* = S_{i+1}^* = S_{i+1}^*$ , namely,  $S_{i+1}^*$  is constructed by removing the non-descendants of  $x_i$  from  $S_{i+1}^*$ . By Eq. (17), this happens when  $x_i \in \text{path}(h)$ , so all non-descendants of  $x_i$  or either off-path for  $h$ , or they are ancestors of  $x_i$ . Seeing as  $x_i \in \text{path}(h)$  and  $u_i^* \in \text{path}(h)$ , and  $u_{i+1}^*$  is the deeper node between these two, any node that is an ancestor of  $x_i$  is also an ancestor of  $u_{i+1}^* = u_{i+1}^*$ . Thus, all the nodes removed or either off-path for  $h$ , or they are ancestors of  $u_{i+1}^*$ , satisfying Item 2. (Similarly, any node that is an ancestor of  $u_i^*$  is also an ancestor of  $u_{i+1}^*$ , so we do not need to add any new nodes to  $S_{i+1}^*$  that are not included in  $S_{i+1}^*$ .)

We see that in all cases, Item 2 is preserved, as desired.  $\square$ 

#### D.4.2 Transition to Halving

**Claim D.6.** Let  $d, n, t \in \mathbb{N}$ ,  $d \geq 16$ , let  $\mathcal{H} \subseteq \{0, 1\}^{T_d}$ , and let  $x_1, \dots, x_n \in T_d$ . Consider an execution of

$$\text{TRANSDUCTIVELEARNER}(\mathcal{H}, (x_1, x_2, \dots, x_n))$$

as in Algorithm 5. Let  $t > t_{\max} = 2^{4\sqrt{d}}$  and let  $e = (S, u, H) \in E_t$  be an expert. Then

$$|H| \leq 2^{2\sqrt{d}}.$$

*Proof of Claim D.6.* Assume for contradiction that  $|H| > 2^{2\sqrt{d}}$ . Let  $H' \subseteq H$  be an arbitrary subset of size  $2^{2\sqrt{d}} + 1$ . Let

$$P = \cup_{h \in H'} \text{path}(h).$$

Seeing as each root-to-leaf path contains  $d + 1$  nodes,

$$|P| \leq |H'| \cdot (d + 1) \leq (2^{2\sqrt{d}} + 1) \cdot (d + 1) \leq d2^{2\sqrt{d}+1}. \quad (18)$$

Let  $y_1, y_2, \dots, y_t$  be the labels provided by the adversary in the first  $t_{\max}$  iterations. The line in EXPERT.BASICUPDATE constructing  $H$  using HALVING.UPDATE( $H, x, y$ ) ensures that

$$\forall h \in H \quad \forall i \in [t_{\max}] : h(x_i) = y_i. \quad (19)$$

- Case I. P<sup>t</sup>max <sup>i</sup>=1 <sup>y</sup><sup>i</sup> ≤ <sup>t</sup>max/2. Then the set

$$X_0 = \{x_i : i \in [t_{\max}] \wedge y_i = 0\}$$

has cardinality |<sup>X</sup>0| ≥ <sup>t</sup>max/2. Let <sup>X</sup>′ <sup>0</sup> <sup>=</sup> <sup>X</sup><sup>0</sup> \ <sup>P</sup>. By Eq. [\(18\)](#page-33-0),

$$|X'_0| \geq \frac{t_{\max}}{2} - d2^{2\sqrt{d}+1} = 2^{4\sqrt{d}} - d2^{2\sqrt{d}+1}. \quad (20)$$

From the choice of X′ 0 , the inclusion <sup>H</sup>′ ⊆ <sup>H</sup>, and Eq. [\(19\)](#page-33-1),

$$\forall h \in H' \quad \forall x \in X'_0 : x \notin \text{path}(h) \wedge h(x) = 0. \quad (21)$$

Seeing as |H′ | <sup>&</sup>gt; <sup>2</sup> 2 √ d , Eq. [\(21\)](#page-34-0) and Item [1](#page-25-2) from Lemma [D.2](#page-25-0) imply that

$$|X'_0| \leq 2^{2\sqrt{d}}. \quad (22)$$

Combining Eqs. [\(20\)](#page-34-1) and [\(22\)](#page-34-2) yields

$$2^{2\sqrt{d}} \geq |X'_0| \geq 2^{4\sqrt{d}} - d2^{2\sqrt{d}+1} \\ \geq 2^{4\sqrt{d}-1} \qquad (d \geq 16),$$

which is a contradiction.

- Case II. P<sup>t</sup>max <sup>i</sup>=1 y<sup>i</sup> > tmax/2. A similar argument gives a contradiction by defining

$$X_1 = \{x_i : i \in [t_{\max}] \wedge y_i = 1\}, \text{ and } X'_1 = X_1 \setminus P.$$

As before,

$$|X'_1| \geq \frac{t_{\max}}{2} - d2^{2\sqrt{d}+1} \geq 2^{4\sqrt{d}} - d2^{2\sqrt{d}+1}. \quad (23)$$

for all <sup>d</sup> ∈ <sup>N</sup>. However, |H′ | <sup>&</sup>gt; <sup>2</sup> 2 √ d and Item [2](#page-25-3) imply that

$$|X'_1| < 3\sqrt{d}, \quad (24)$$

which is a contradiction.

#### D.4.3 Performance of Best Expert

Claim D.7 (Existence of expert with large weight). *Let* d, n ∈ <sup>N</sup>*,* <sup>d</sup> ≥ <sup>16</sup>*, let* H ⊆ {0, <sup>1</sup>} <sup>T</sup><sup>d</sup> *, and let* <sup>x</sup>1, . . . , x<sup>n</sup> ∈ <sup>T</sup>d*. Consider an execution of*

TRANSDUCTIVELEARNER(
$$\mathcal{H}, (x_1, x_2, \dots, x_n)$$
)

*as in Algorithm [5.](#page-29-0) Then, at the end of the execution, there exists* <sup>e</sup> ∈ <sup>E</sup>n+1 *such that*

$$w(e) \geq 2^{-48\sqrt{d}}. \quad (25)$$

Note that the lower bound in Eq. [\(25\)](#page-34-3) does not depend on n.

*Proof.* Fix a hypothesis <sup>h</sup> ∈ H such that <sup>h</sup>(xt) = <sup>y</sup><sup>t</sup> for all <sup>t</sup> ∈ [n] (such an <sup>h</sup> exists because the adversary must always select a realizable label).

By Claim [D.4,](#page-19-0) there exists e ∗ <sup>n</sup>+1 ∈ <sup>E</sup>n+1 that is assumption-consistent with <sup>h</sup>. Let ancestry e ∗ n+1 = (e ∗ 1 , e<sup>∗</sup> 2 , . . . , e<sup>∗</sup> <sup>n</sup>+1). We argue that this ancestry sequence makes few mistakes. Specifically, for each <sup>t</sup> ∈ [n], let <sup>y</sup><sup>ˆ</sup> ∗ <sup>t</sup> = EXPERT.PREDICT(e ∗ t , xt). We claim that

$$m := \sum_{t=1}^n \mathbb{1}(\hat{y}_t^* \neq y_t) \leq 24\sqrt{d}.$$

Indeed, let <sup>B</sup> <sup>=</sup> {<sup>t</sup> ∈ [n] : ˆ<sup>y</sup> ∗ <sup>t</sup> ̸<sup>=</sup> <sup>y</sup>t} be the set of <sup>m</sup> indices where a mistake was made. For each <sup>t</sup> ∈ <sup>B</sup>, let <sup>e</sup> ∗ <sup>t</sup> = (S, u, H), and note that each <sup>t</sup> ∈ <sup>B</sup> has a corresponding execution of EXPERT.PREDICT(e ∗ t , xt), and an execution of e ′ <sup>t</sup> = EXPERT.BASICUPDATE(e ∗ t , xt, yt) followed by EXPERT.EXTENDEDUPDATE(e ′ t , xt, yt) that produces e ∗ <sup>t</sup>+1 (EXPERT.EXTENDEDUPDATE is executed because <sup>t</sup> ∈ <sup>B</sup>, i.e., a mistake was made). We partition the indices in <sup>B</sup> into six cases (six disjoint sets), and bound the number of indices that fall in each.

- • Case I. *The execution of* EXPERT.PREDICT(e ∗ t , xt) *exited via the first return [statement](#page-30-2) in that procedure.* This happens once |H| ≤ <sup>2</sup> 2 √ d , and from that point on, the expert and all subsequent experts in the ancestry are exactly simulating the HALVING algorithm (Algorithm [7\)](#page-32-0) in both predictions and updates. Hence, by Fact [E.1,](#page-37-0) B contains at most <sup>m</sup><sup>I</sup> = 2√ d such indices.
- • Case II. *The execution of* EXPERT.PREDICT(e ∗ t , xt) *exited via the second return [statement](#page-30-3) in that procedure.* In particular x ≼ u, and the predicted label was yˆ ∗ <sup>t</sup> <sup>=</sup> <sup>b</sup> ∈ {0, <sup>1</sup>} such that x<sup>t</sup> ≼<sup>b</sup> u. Because e ∗ t is assumption-consistent with h, Item [1](#page-32-2) in Claim [D.5](#page-19-3) implies that <sup>u</sup> ∈ path(h). Namely, we see that <sup>u</sup> is a <sup>b</sup>-descendant of <sup>x</sup><sup>t</sup> and <sup>u</sup> ∈ path(h). It follows that yˆ ∗ <sup>t</sup> = b = h(xt) = yt. So no mistakes are made in Case [II,](#page-35-1) and the number of indices <sup>t</sup> ∈ <sup>B</sup> that belong to Case [II](#page-35-1) is simply <sup>m</sup>II = 0.

In the remaining cases, we assume that EXPERT.PREDICT(e ∗ t , xt) exited via the third return [statement](#page-30-4) in that procedure, so the prediction was

$$\hat{y}_t^* = \mathbf{1}(|S_1| > |S|/3), \quad (26)$$

where <sup>S</sup><sup>1</sup> <sup>=</sup> {<sup>x</sup> ′ ∈ <sup>S</sup> : <sup>x</sup><sup>t</sup> <sup>≼</sup><sup>1</sup> <sup>x</sup> ′}. These cases are as follows.

- • Case III. *The execution of* EXPERT.EXTENDEDUPDATE(e ′ t , xt, yt) *exited via the first [return](#page-31-2) [statement](#page-31-2) in that procedure.* Namely, after the update, the resulting expert e ∗ <sup>t</sup>+1 has |H| ≤ <sup>2</sup> 2 √ d . However, because we are not in Case [I,](#page-35-0) at the beginning of the iteration expert e ∗ <sup>t</sup> had |H| <sup>&</sup>gt; <sup>2</sup> 2 √ d . Seeing as the cardinality of H decreases monotonically throughout the ancestry e ∗ 1 , . . . , e<sup>∗</sup> <sup>n</sup>+1, this type of mistake can happen at most mIII = 1 times.
- • Case IV. *The execution of* EXPERT.EXTENDEDUPDATE(e ′ t , xt, yt) *exited via the [second](#page-31-3) return [statement](#page-31-3) in that procedure.* In this case, |<sup>S</sup>(1−yt) | <sup>&</sup>gt; |S|/3, and <sup>e</sup> ∗ <sup>t</sup>+1 = (S ′ , u, H) with S ′ <sup>=</sup> <sup>S</sup> \ <sup>S</sup>1−y<sup>t</sup> . So |<sup>S</sup> ′ | <sup>&</sup>lt; <sup>2</sup>|S|/3. Namely, the update causes the cardinality of the set S to be multiplied by a factor of at most 2/3 and it strictly decreases. Seeing as the initial cardinality is tmax, and cardinalities are integers, the number of times this can happen is at most

$$m_{\text{IV}} = \frac{\log(t_{\text{max}})}{\log(3/2)} + 1 = \frac{4\sqrt{d}}{\log(3/2)} + 1. \quad (27)$$

In the remaining cases, we assume that the execution of EXPERT.EXTENDEDUPDATE(e ∗ t , xt, yt) exited via the third return [statement](#page-31-1) in that procedure. This implies that

$$|S_{\hat{y}_t^*}| \leq |S|/3 \quad (28)$$

Combining this with Eq. [\(26\)](#page-35-5), it follows yˆ ∗ <sup>t</sup> = 0 and therefore y<sup>t</sup> = 1. The remaining cases are as follows.

- • Case V. <sup>x</sup><sup>t</sup> ∈ path(h). Let <sup>e</sup> ∗ <sup>t</sup> = (S, u, H). Seeing as |H| <sup>&</sup>gt; <sup>2</sup> 2 √ d (because we are not in Case [I\)](#page-35-0), Claim [D.6](#page-24-3) (with the assumption <sup>d</sup> ≥ <sup>16</sup>) implies that <sup>t</sup> ≤ <sup>t</sup>max. By Item [2](#page-32-3) of Claim [D.5,](#page-19-3) the facts <sup>x</sup><sup>t</sup> ̸<sup>≼</sup> <sup>u</sup> (we are not in Case [II\)](#page-35-1) and <sup>x</sup><sup>t</sup> ∈ path(h) imply that <sup>x</sup><sup>t</sup> ∈ <sup>S</sup>. In particular, <sup>S</sup> is not empty. Because the <sup>t</sup> → (<sup>t</sup> + 1) update of <sup>e</sup> ∗ <sup>t</sup>+1 was assumption-consistent with h, Eq. [\(17\)](#page-29-1) implies that e ∗ <sup>t</sup>+1 = (S∈, u∈, H∈), with <sup>S</sup><sup>∈</sup> <sup>=</sup> <sup>S</sup><sup>0</sup> ∪ <sup>S</sup>1. Observe that
  - |<sup>S</sup>0| ≤ |S|/<sup>3</sup> (plugging <sup>y</sup><sup>ˆ</sup> ∗ <sup>t</sup> = 0 into Eq. [\(28\)](#page-35-6)); and
  - |<sup>S</sup>1| ≤ |S|/<sup>3</sup> (because otherwise, by Eq. [\(26\)](#page-35-5), the prediction would have been yˆ ∗ <sup>t</sup> = 1).

Therefore,

$$|S_{\epsilon}| \leq |S_0| + |S_1| \leq 2|S|/3. \quad (29)$$

As in Case [IV,](#page-35-3) combining Eq. [\(29\)](#page-35-7) and the fact that S is not empty imply an upper bound m<sup>V</sup> on the number of times Case [V](#page-35-4) can happen, with the bound being the same number m<sup>V</sup> = mIV as in Eq. [\(27\)](#page-35-8).

- • Case VI. <sup>x</sup><sup>t</sup> ∈/ path(h). So (xt, yt) is a pair such that <sup>x</sup><sup>t</sup> ∈/ path(h) and <sup>y</sup><sup>t</sup> = 1. Assume for contradiction that this type of mistake can happen strictly more than

$$m_{\text{VI}} = 3\sqrt{d}$$

times. Let t1, t2, . . . , tmVI be the indices of the first mVI iterations of the outer 'for' loop of TRANSDUCTIVELEARNER in which this type of mistake happened. Note that if at the end of iteration tmVI , we had expert e ∗ <sup>t</sup><sup>m</sup>VI+1 = (St<sup>m</sup>VI+1, ut<sup>m</sup>VI+1, Ht<sup>m</sup>VI+1) such that |<sup>H</sup>t<sup>m</sup>VI+1| ≤ <sup>2</sup> 2 √ d , then from that point onwards, the expert would be simulating the halving algorithm, and in particular, it would not make any further mistake of the type in Case [VI](#page-36-0) (all subsequent mistakes would belong to Case [I\)](#page-35-0). Hence, by the assumption that strictly more than mVI mistakes were made, it follows that |<sup>H</sup>t<sup>m</sup>VI+1| <sup>&</sup>gt; <sup>2</sup> 2 √ d . Let

$$H^* = \{h' \in \mathcal{H} : (\forall t \in [m_{\text{vl}}] : h'(x_t) = 1 \wedge x_t \notin \text{path}(h'))\}.$$

Because e ∗ <sup>t</sup><sup>m</sup>VI+1 is assumption-consistent with h, and from the construction of Ht<sup>m</sup>VI+1 using <sup>H</sup><sup>∈</sup> and <sup>H</sup>∈/ in <sup>E</sup>XPERT.EXTENDEDUPDATE, it follows that <sup>H</sup>t<sup>m</sup>VI+1 ⊆ <sup>H</sup><sup>∗</sup> . So there exist collections <sup>H</sup><sup>∗</sup> ⊆ H and <sup>X</sup> <sup>=</sup> {<sup>x</sup>t<sup>t</sup> : <sup>t</sup> ∈ [mVI]} ⊆ <sup>T</sup><sup>d</sup> such that

- |H<sup>∗</sup> | ≥ |<sup>H</sup><sup>t</sup><sup>m</sup>VI+1| <sup>&</sup>gt; <sup>2</sup> 2 √ d ,
- <sup>|</sup>X<sup>|</sup> <sup>=</sup> <sup>m</sup>VI = 3<sup>√</sup> d,
- ∀<sup>h</sup> ′ ∈ <sup>H</sup><sup>∗</sup> ∀<sup>x</sup> ∈ <sup>X</sup> : <sup>h</sup> ′
- (x) = 1.
- ∀<sup>h</sup> ′ ∈ <sup>H</sup><sup>∗</sup> ∀<sup>x</sup> ∈ <sup>X</sup> : x /∈ path(<sup>h</sup> ′ ).

This is a contradiction to the choice of H, specifically, to Item [2](#page-25-3) in Lemma [D.2.](#page-25-0)

Thus, combining the analyses of all cases, we see that the number of mistakes made by the ancestry e ∗ n+1 is at most

$$\begin{aligned} m &\leq m_{\text{I}} + m_{\text{II}} + m_{\text{III}} + m_{\text{IV}} + m_{\text{V}} + m_{\text{VI}} \\ &\leq 2\sqrt{d} + 0 + 1 + \left( \frac{4\sqrt{d}}{\log(3/2)} + 1 \right) + \left( \frac{4\sqrt{d}}{\log(3/2)} + 1 \right) + 3\sqrt{d} \\ &\leq 24\sqrt{d}. \end{aligned}$$

The weights satisfy

$$w(e_{t+1}^*) \begin{cases} = w(e_t^*) & \hat{y}_t^* = y_t \\ \geq \frac{1}{4} \cdot w(e_t^*) & \hat{y}_t^* \neq y_t. \end{cases}$$

This implies that w(e ∗ <sup>n</sup>+1) ≥ <sup>w</sup>(<sup>e</sup> ∗ 1 ) · Q<sup>n</sup> <sup>t</sup>=1 4 <sup>−</sup>1(ˆyi̸=yi) = w(e ∗ 1 ) · 4 <sup>−</sup><sup>m</sup> ≥ <sup>4</sup> −24√ <sup>d</sup> = 2<sup>−</sup>48<sup>√</sup> d , as desired.

### D.4.4 Multiplicative Weights Mistake Bound

Claim D.8 (Mistake bound for multiplicative weights). *Let* d, n ∈ <sup>N</sup>*, let* α > <sup>0</sup>*, let* H ⊆ {0, <sup>1</sup>} <sup>T</sup><sup>d</sup> *, and let* <sup>x</sup>1, . . . , x<sup>n</sup> ∈ <sup>T</sup>d*. Consider an execution of*

TRANSDUCTIVELEARNER(
$$\mathcal{H}, (x_1, x_2, \dots, x_n)$$
)

*as in Algorithm [5.](#page-29-0) Assume that at the end of the execution, there exists* e <sup>∗</sup> ∈ <sup>E</sup>n+1 *such that*

$$w(e^*) \geq 2^{-\alpha}.$$

*Then* TRANSDUCTIVELEARNER *makes at most* α *mistakes.*

*Proof of Claim [D.8.](#page-36-1)* For all <sup>i</sup> ∈ [<sup>n</sup> + 1], let <sup>w</sup>(Ei) = <sup>P</sup> e∈E<sup>i</sup> <sup>w</sup>(e). For each <sup>i</sup> ∈ [n], if <sup>y</sup>ˆ<sup>i</sup> ̸<sup>=</sup> <sup>y</sup><sup>i</sup> , then <sup>w</sup>(Ei+1) ≤ <sup>w</sup>(Ei)/2. Hence, if TRANSDUCTIVELEARNER makes <sup>m</sup> mistakes, then by induction

$$w(E_{n+1}) \leq w(E_1) \cdot \prod_{t=1}^n 2^{-1(\hat{y}_i \neq y_i)} = 2^{-m} \cdot w(E_1).$$

So

$$2^{-\alpha} \leq w(e^*) \leq \sum_{e \in E_{n+1}} w(e) = w(E_{n+1}) \leq 2^{-m} \cdot w(E_1) = 2^{-m}.$$

We conclude that

$$m \leq \alpha,$$

as desired.

### D.5 Proof

*Proof of Theorem [D.1.](#page-2-0)* Fix an integer <sup>d</sup> ≥ <sup>43</sup>. Let H ⊆ {0, <sup>1</sup>} <sup>T</sup>d−<sup>1</sup> be the class constructed by invoking Lemma [D.2](#page-25-0) for the integer <sup>d</sup> − <sup>1</sup> ≥ <sup>42</sup>. We argue that this class satisfies the requirements of Theorem [D.1.](#page-2-0)

By construction, H is a class of Littlestone dimension precisely <sup>d</sup>. By Theorem [A.7,](#page-15-0) this implies the equality in Item [2.](#page-25-4)

We now show the upper bound in Item [1.](#page-25-5) We argue that TRANSDUCTIVELEARNER (Algorithm [5\)](#page-29-0) satisfies this upper bound. By Claim [D.7,](#page-34-4) at the end of the execution of TRANSDUCTIVELEARNER there exists an expert <sup>e</sup> ∈ <sup>E</sup>n+1 such that <sup>w</sup>(e) ≥ <sup>2</sup> −48√ d . By Claim [D.8,](#page-36-1) this implies that the number of mistakes made by TRANSDUCTIVELEARNER is at most <sup>48</sup>√ d, as desired.

## E Halving

Fact E.1. *Let* X *be a set, and let* H ⊆ {0, <sup>1</sup>} <sup>X</sup> *be a hypothesis class. Then for all* <sup>n</sup> ∈ <sup>N</sup>*, all sequences* <sup>x</sup> ∈ X <sup>n</sup>*, and all realizable adversaries,* <sup>H</sup>ALVING *(Algorithm [7\)](#page-32-0) makes at most* log(|H|) *mistakes in the transductive online learning (Game [2\)](#page-2-2).*[<sup>25</sup>](#page-37-1) *Namely,*

$$\sup_{n \in \mathbb{N}} \sup_{A \in \mathcal{A}_n} M_{\text{tr}}(\mathcal{H}, n, \text{HALVING}, A) \leq \log(|\mathcal{H}|).$$

<sup>25</sup>With the suitable syntactic modification, it also makes at most log(|H|) mistakes in the standard online learning (Game [1\)](#page-2-1).

## NeurIPS Paper Checklist

#### 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The main claims made in the abstract and introduction accurately reflect the paper's contributions and scope.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

#### 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [NA]

Justification: Purely rigorous mathematical results. We explain precisely what our proofs imply (and therefore also what they do not imply).

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

#### 3. Theory assumptions and proofs

#### Answer: [Yes]

Justification: For each theoretical result, the paper provides the full set of assumptions and a complete (and correct) proof.

#### Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

#### 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

#### Answer: [NA]

Justification: The paper has no experiments.

### Guidelines:

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
  - (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
  - (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
  - (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
  - (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [NA]

Justification: The paper does not include experiments requiring code.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ([https://nips.cc/](https://nips.cc/public/guides/CodeSubmissionPolicy) [public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ([https:](https://nips.cc/public/guides/CodeSubmissionPolicy) [//nips.cc/public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

#### 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [NA]

Justification: The paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

### 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: The paper does not include experiments.

- The answer NA means that the paper does not include experiments.
- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

- The assumptions made should be given (e.g., Normally distributed errors).
- It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

#### 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [NA]

Justification: The paper does not include experiments.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

#### 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?

Answer: [Yes]

Justification: The research conducted in the paper conforms, in every respect, with the NeurIPS Code of Ethics.

Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

#### 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [No]

Justification: The work is purely theoretical with no immediate direct societal impacts forseeable.

- The answer NA means that there is no societal impact of the work performed.
- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

#### 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: The paper poses no such risks.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

#### 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: The paper does not use existing assets.

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, <paperswithcode.com/datasets> has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

#### 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: The paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

#### 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

#### 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development in this research does not involve LLMS as any important, original, or non-standard components.

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy (<https://neurips.cc/Conferences/2025/LLM>) for what should or should not be described.