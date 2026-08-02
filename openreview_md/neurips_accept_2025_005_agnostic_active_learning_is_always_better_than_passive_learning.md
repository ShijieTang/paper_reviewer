# Agnostic Active Learning Is Always Better Than Passive Learning

Steve Hanneke Department of Computer Science Purdue University steve.hanneke@gmail.com

## Abstract

This work resolves a long-standing open question of central importance to the theory of active learning, closing a qualitative and quantitative gap in our understanding of active learning in the non-realizable case. We provide the first sharp characterization of the optimal first-order query complexity of agnostic active learning, and propose a new general active learning algorithm which achieves it. Remarkably, the optimal query complexity admits a leading term which is *always* strictly smaller than the sample complexity of passive supervised learning (by a factor proportional to the best-in-class error rate). This was not previously known to be possible. For comparison, in all previous general analyses, the leading term exhibits an additional factor, such as the disagreement coefficient or related complexity measures, and therefore only provides improvements over passive learning in restricted cases. The present work completely removes such factors from the leading term, implying that every concept class benefits from active learning in the non-realizable case. Whether such benefits are possible has been the driving question underlying the past two decades of research on the theory of agnostic active learning. This work finally settles this fundamental question.

## 1 Introduction

Active learning is a well-known powerful variant of supervised learning, in which the learning algorithm interactively participates in the process of labeling the training examples. In this setting, there is a pool (or stream) of unlabeled examples, and the learning algorithm selects individual examples and queries an oracle (typically a human labeler) to observe their labels. This happens sequentially, so that the learner has observed previously-queried labels before deciding which example to query next. The intended purpose of active learning is to reduce the overall number of labels necessary for learning to a given accuracy, called the *query complexity*. We are therefore particularly interested in using active learning in scenarios where its query complexity is significantly smaller than the number of randomly-sampled training examples which would be needed to achieve the same accuracy, called the sample complexity of *passive* supervised learning. Active learning has not only been incredibly useful for many practical machine learning problems (e.g., Cohn et al., 1996; Tong and Koller, 2001; Zhu et al., 2003; Olsson, 2009; Settles, 2012; Ren et al., 2021; Mosqueira-Rey et al., 2023) but has also given rise to a rich and nuanced theoretical literature (e.g., Dasgupta, 2005, 2011; Balcan et al., 2009; Hanneke, 2007b, 2014; Zhang and Chaudhuri, 2014; Hanneke and Yang, 2015; see Appendix A for a detailed survey). Moreover, the insights and techniques discovered in this literature have had tremendous influence on other branches of the learning theory literature (e.g., Awasthi et al., 2014; Foster et al., 2021; Hanneke, 2009b, 2016a,b, 2024; Zhivotovskiy and Hanneke, 2018; Simon, 2015; Balcan and Long, 2013; El-Yaniv and Wiener, 2010; Balcan et al., 2022). Within the literature on the theory of active learning, a central topic which has garnered by-far the most interest is that of *agnostic* active learning: that is, the study of active learning algorithms capable of providing performance guarantees even in noisy or otherwise non-realizable learning problems, without assumptions on the form of the noise. This line of work was initiated by the groundbreaking A2algorithm (Agnostic Active) of Balcan, Beygelzimer, and Langford (2005, 2006, 2009) (with its general analysis later given by Hanneke, 2007b) and concurrently a lower bound analysis of Kääriäinen (2005, 2006) (later strengthened by Beygelzimer, Dasgupta, and Langford, 2009). These results were later refined and extended in numerous ways. However, throughout this two-decades long history, there has persisted a significant gap between the sharpest known upper and lower bounds on the optimal query complexity. Moreover, this gap represents an important *qualitative* distinction:
while the lower bound is always smaller than the sample complexity of passive learning, the existing upper bounds only reflect such improvements under further restrictive conditions (e.g., bounded disagreement coefficient). Thus, the issue of resolving this gap is of central importance to this subject, since it has implications for answering the question:
Does every *concept class admit benefits from using active learning instead of passive learning?*
The main contribution of the present work is to establish that this is indeed true, and in fact the known lower bound is always attainable. To achieve this, we introduce new algorithmic principles for active learning (the AVID principle), improving concentration of error estimates via adaptively isolating regions where the error estimates have high variance and allocating more queries to such regions.

## 2 Background And Summary Of The Main Result

Let C be any concept class1(a set of functions *X → {*0, 1} on a set X called the *instance space*) and denote by d = VC(C) the VC dimension of C (Vapnik and Chervonenkis, 1971; see Definition 4).

Let P be an (unknown) joint distribution on *X × {*0, 1}, and define the *error rate* of any *classifier* h : *X → {*0, 1} as erP (h) := P((*x, y*) : h(x) ̸= y). In the *active learning* problem, there is a sequence (X1, Y1), . . . ,(Xm, Ym) of i.i.d. samples from P, but the learner initially only observes the Xi values (the *unlabeled* examples). It then has the capability to *query* any example Xi, which reveals the corresponding true label Yi, in a *sequential* manner (i.e., it chooses its next query Xi
′ after observing the label Yi of its previous query point Xi). After a number of such queries, the learner returns a classifier hˆ. The goal is to achieve a small *excess* error rate erP (hˆ) ≤ infh∈C erP (h) + ε while making as few queries as possible. We are particularly interested in quantifying the number of queries sufficient to achieve this, as a function of ε and the value of the *best-in-class* error rate infh∈C erP (h), known as a *first-order* query complexity bound.

Specifically, for any *ε, δ, β* ∈ (0, 1), the optimal query complexity, QCa(*ε, δ*; β, C), is defined as the minimal Q ∈ N for which there exists an active learner Aa such that (for a sufficiently large number m of unlabeled examples), for every P with infh∈C erP (h) ≤ β, with probability at least 1 − δ, Aa makes at most Q queries and returns hˆ satisfying erP (hˆ) ≤ infh∈C erP (h) + ε. The main quantity for comparison is the *sample complexity* of supervised *passive learning*. A passive learner Ap simply trains on n *labeled* training examples (X1, Y1), . . . ,(Xn, Yn) sampled i.i.d. from P to produce a classifier hˆ. For *ε, δ, β* ∈ (0, 1), the *optimal sample complexity* of passive learning, Mp(ε, δ; β, C), is defined as the minimal size n ∈ N of such a training sample for which there exists a passive learner Ap such that, for every P with infh∈C erP (h) ≤ β, with probability at least 1 − δ, Ap returns hˆ satisfying erP (hˆ) ≤ infh∈C erP (h) + ε. We remark that, in both the active and passive cases, these definitions place no restrictions on the computational efficiency of the learning algorithms, but rather focus on the *data efficiency*, which is our primary interest in this work (see Section G).

Since both the query complexity and sample complexity concern the number of *labels* sufficient for learning, it is natural to compare QCa(*ε, δ*; β, C) with Mp(ε, δ; β, C) to quantify the benefits of active learning. Thus, the primary interest in the theory of agnostic active learning is quantifying how much smaller QCa
(*ε, δ*; β, C) is compared to Mp(*ε, δ*; β, C). Since our interest is *agnostic* learning, it is most interesting to focus on the regime where P is far-from-realizable: that is, where β is much larger than ε. In this regime, it is well known from the works of Vapnik and Chervonenkis (1974);
Devroye and Lugosi (1995); Hanneke, Larsen, and Zhivotovskiy (2024b) that the optimal sample 1To focus on non-trivial cases, we suppose |C| ≥ 3. We also suppose X is equipped with a σ-algebra specifying its measurable subsets, and we adopt the standard mild measure-theoretic restrictions on the σ-algebra and the class C from empirical process theory: namely, the image-admissible Suslin property (Dudley, 1999).

complexity of passive learning satisfies Mp(*ε, δ*; β, C) = Θβ ε 2d + log1δ
. In comparison, the known lower bound for active learning is QCa(ε, δ; β, C) = Ωβ 2 ε 2d + log1 δ
(Kääriäinen, 2006; Beygelzimer, Dasgupta, and Langford, 2009). Thus, the strongest improvement we might hope from active learning is a factor of β (representing the *best-in-class* error rate). However, in the prior literature, this β-factor improvement has only been demonstrated in upper bounds under *restrictions* to C or P. Specifically, every general upper bound on QCa(ε, δ; β, C)
in the literature has the form c(β)d β 2 ε 2 (ignoring logs), where c(β) is a (C, P)-dependent quantity.

For instance, one commonly appearing such quantity c(β) is the *disagreement coefficient* θ(β) of Hanneke (2007b). We refer the reader to Appendix A for a detailed survey of such quantities c(β) which have appeared in the literature. Importantly, for all such upper bounds in the literature, the corresponding factor c(β) has the property that there exist simple classes C and distributions P for which c(β) ≥
1 β
(see Hanneke and Yang, 2015; Hanneke, 2016b, 2024): for instance, even for linear classifiers on R
2 or singletons on N. Note that when c(β) ≥
1 β
, a query complexity c(β)d β 2 ε 2 becomes no smaller than d β ε2 , the sample complexity of *passive* learning. Moreover, one can show that avoiding such d β ε 2 query complexities would require new algorithmic techniques (see Appendix A).

Naturally, the question of refining such c(β) factors has been a subject of much interest for many years. In particular, it has remained open whether such factors might even be *avoided entirely*, so that the β-factor improvement might *always* be achievable. In a series of talks, I conjectured that the lower bound Ω
β 2 ε 2d + log1δ is *always* sharp (in the far-from-realizable regime), and even offered a sizable prize for a solution (along with lower-order terms) (e.g., Hanneke and Nowak, 2019). Contributions of this Work: In the present work, we completely resolve this question. We prove that (in the above regime) QCa(ε, δ; β, C) = Θβ 2 ε 2d + log1 δ
. In other words, the β-factor improvement is *always* achievable, the known lower bound is *sharp*, and there is *no need* for restrictions on (C, P) or additional factors c(β) as appear in all prior works.

Extending to the *full range* of β, the more-general form of the bound we prove also includes an additive *lower-order* term to account for the small-β regime. In the simplest such bound (Theorem 1),
this lower-order term is simply O˜dε
, so that the general form is QCa(ε, δ; β, C) = O˜d β 2 ε 2 +
d ε

(Theorem 3 and Appendix F refine this lower-order term for some classes). For comparison, the general form of the passive sample complexity is Mp(*ε, δ*; β, C) = Θ˜d β ε 2 +
d ε
. We note that, even in the *nearly-realizable* regime (β = O˜(ε)), it is known that dε is a lower bound on the query complexity for many classes C (Dasgupta, 2005; Hanneke, 2014; see Appendix D of Hanneke and Yang, 2015), so that this term is sometimes unavoidable, and hence the benefits of active learning can wane in the nearly-realizable regime. Likewise, the lower bound d β 2 ε2 implies the benefits can also diminish in the very-high-noise regime (β = Ω(1)). In contrast, as discussed above, in the *far-from-realizable* regime (√ε ≤ β ≪ 1), the bound is of order d β 2 ε2 , reflecting a β-factor improvement over the sample complexity of passive learning d β ε 2 . Additionally, the intermediate regime of *moderate-size* β (i.e., ε ≪ β < 
√ε) *also* exhibits improvements over passive learning for all C: in this regime, Mp(*ε, δ*; β, C) = Ωd β ε2
, whereas QCa(*ε, δ*; β, C) = O˜dε
≪ d β ε2 ,
reflecting an improvement by a factor O˜(
ε β
). Altogether, this result reveals a previously-unknown and truly remarkable fact: QCa(ε, δ; β, C) ≪ Mp(ε, δ; β, C) in all regimes ε ≪ β ≪ 1, or in other words, in all regimes outside the nearly-realizable and very-high-noise cases, the following is true:
For every concept class C*, the optimal query complexity of agnostic active learning is strictly smaller* than the optimal sample complexity of agnostic passive learning.

This result resolves an important long-standing open question central to the past two decades of research on the theory of agnostic active learning.

## 3 Main Results

Formally, the following theorem expresses the new upper bound, together with known lower bounds for comparison (Kääriäinen, 2006; Beygelzimer, Dasgupta, and Langford, 2009; Hanneke, 2014; Hanneke and Yang, 2015). A more-detailed version of the result appears in Theorem 5 (Appendix C).

Theorem 1. For every concept class C*, letting* d = VC(C), ∀*ε, δ* ∈ (0, 1/8), ∀β ∈ [0, 1],

$$\mathrm{QC}_{a}(\varepsilon,\delta;\beta,\mathbb{C})=O\bigg({\frac{\beta^{2}}{\varepsilon^{2}}}\left(\mathbf{d}+\log\!\left({\frac{1}{\delta}}\right)\right)\bigg)+\tilde{O}\bigg({\frac{\mathbf{d}}{\varepsilon}}\bigg)$$

and QCa(*ε, δ*; β, C) = Ωβ
$=\;\Omega\bigg(\frac{\beta^2}{\varepsilon^2}\left(\phi\right)\bigg)$  . 
2d + log1δ
. Moreover, for every d ∈ N there exists C *with* VC(C) = d such that QCa(*ε, δ*; β, C) = Ωβ 2 ε 2d + log1δ
 +
d ε
.

We provide a new general active learning algorithm Aavid achieving this upper bound in Section 4.

Importantly, the algorithm *does not need to know* β (or anything else about P) to achieve this guarantee: i.e., it is completely *adaptive* to the value β. Moreover, the number of *unlabeled* examples the algorithm requires is only Θ˜d β ε 2 +
d ε
, of the same order as the sample complexity of passive learning; it can also adaptively determine how many unlabeled examples to use without knowing β. The AVID Principle: The main innovation underlying the algorithm, which enables it to achieve this query complexity, represents a new principle for the design of active learning learning algorithms, which we call *Adaptive Variance Isolation by Disagreements* (AVID). The algorithm adaptively partitions the instance space X into *regions*, with the aim of *isolating* a region ∆ ⊆ X where it is most challenging to learn, due to exceptionally high *variance* in the error estimation problem in the ∆ region (where ∆ will be defined as a union of pairwise *disagreement* regions witnessing the high variance, carefully selected to ensure PX (∆) = O(β)). It then allocates disproportionately more queries to this challenging region ∆ compared to the (considerably-easier) remaining region X \ ∆. This idea has interesting connections to techniques explored in other branches of the literature (e.g., Hanneke, Larsen, and Zhivotovskiy, 2024b; Bousquet and Zhivotovskiy, 2021; Puchkin and Zhivotovskiy, 2022), discussed in Appendix A.

## 3.1 Refinement Of The Lower-Order Term For Some Classes

The AVID principle already suffices to achieve the query complexity bound in Theorem 1. Moreover, for *most* concept classes of interest, the query complexity bound in Theorem 1 is already *optimal*,
matching a lower bound (up to log factors in the lower-order term): e.g., linear classifiers in R
k, k ≥ 2 (Dasgupta, 2005; Hanneke, 2014; Hanneke and Yang, 2015). However, while the lead term β 2 ε 2d + log1δ is already optimal for every concept class C, there do exist some special classes C
for which a further refinement of the *lower-order* term dε is possible (e.g., threshold classifiers 1[a,∞)
on R). As our second main result, we provide a refinement of the upper bound in Theorem 1 to capture such special classes, thereby establishing a query complexity bound which is nearly optimal for *every* concept class. Since such refinements are only possible for some concept classes, the expression of this refinement necessarily depends on an additional complexity measure of the class C. We prove that the *optimal* lower-order term in the query complexity is well-captured by a quantity known as the *star number* of C, introduced by Hanneke and Yang (2015). In particular, Hanneke and Yang (2015) showed that the star number precisely characterizes the optimal query complexity in the *realizable case* (β = 0); since this is a limiting case of agnostic learning, it is natural that this quantity plays a crucial role in characterizing the optimal lower-order term. The formal definition is as follows.

Definition 2. For any concept class C*, the* star number s = s(C) is the supremum n ∈ N *for which*
∃x1, . . . , xn ∈ X and h0, h1, . . . , hn ∈ C such that ∀i, j ∈ {1*, . . . , n*}, hi(xj ) ̸= h0(xj ) ⇔ i = j.

The star number essentially describes a scenario which is intuitively challenging for active learners in the realizable case, wherein there is a set of instances xj and a *default* labeling h0(xj ), but the *target* concept is some hi which differs from h0 at just one instance xi, unknown to the learner (which must therefore query nearly all of these xj instances, searching for the special point xi, in order to identify the target concept hi). Hanneke and Yang (2015) provide numerous examples calculating s for various concept classes. For instance, thresholds on R have s = 2 and decision stumps on R 
k have s = 2k. However, it is worth noting that s is typically large (or infinite) for most concept classes of interest in learning theory (e.g., s = ∞ for linear classifiers on R
k, k ≥ 2). This fact is important to the present work, since Hanneke and Yang (2015); Hanneke (2016b, 2024) have shown that the c(β) factors (discussed in Section 2 above) appearing in all previous general upper bounds all become no smaller than s ∧
1 β in the worst case over distributions (subject to the β constraint). Thus, all general upper bounds c(β)d β 2 ε 2 from the prior literature become no smaller than d β ε 2 in the worst case when s = ∞. In a sense, this means Theorem 1 is actually *most* interesting in the (typical) case of s = ∞, since no previously known upper bounds offer any improvements over passive learning in this case (without further restrictions to P), in stark contrast to Theorem 1 which has *no dependence* on s and provides improvements over passive learning in the lead term for *every* concept class. Nonetheless, the special structure of classes with s < ∞ turns out to provide some additional advantages for active learning, so that in order to state a general query complexity bound which is optimal for *every* concept class C, we need to account for this structure, via a dependence on s in the lower-order term. Specifically, by combining the AVID principle with existing principles for active learning (namely, *disagreement-based* queries), we can take further advantage of the power of active learning, thereby enabling a refinement of the lower-order term for classes with s < ∞.

The following result presents a new general query complexity bound reflecting such refinements, together with a known lower bound for comparison (due to Kääriäinen, 2006; Beygelzimer, Dasgupta, and Langford, 2009; Hanneke and Yang, 2015). The implication is that this new upper bound is nearly optimal for *every* concept class C (including the lower-order term, up to a factor of d, which we discuss below). A more-detailed version of the result appears in Theorem 5 of Appendix C (and distribution-dependent variants are presented in Appendix F, replacing s with variants of the disagreement coefficient).

Theorem 3. For every C*, letting* d = VC(C) and s = s(C), ∀*ε, δ* ∈ (0, 1/8), ∀β ∈ [0, 1],

), $\forall\beta\in[0,1]$,
$$\mathrm{QC}_{a}(\varepsilon,\delta;\beta,\mathbb{C})=O\bigg{(}\frac{\beta^{2}}{\varepsilon^{2}}\left(\mathrm{d}+\log\!\left(\frac{1}{\delta}\right)\right)\bigg{)}+\tilde{O}\bigg{(}\left(\mathsf{s}\wedge\frac{1}{\varepsilon}\right)\mathrm{d}\bigg{)}\,,$$  _and $\mathrm{QC}_{a}(\varepsilon,\delta;\beta,\mathbb{C})=\Omega\bigg{(}\frac{\beta^{2}}{\varepsilon^{2}}\left(\mathrm{d}+\log\!\left(\frac{1}{\delta}\right)\right)+\mathsf{s}\wedge\frac{1}{\varepsilon}\bigg{)}\,.$_
We may note that the upper bound in Theorem 1 is an immediate implication of Theorem 3 (we have stated Theorem 1 separately merely to emphasize that the improvements over passive learning are available without any special properties of C such as finite star number). Theorem 3 provides a refinement in the lower-order term compared to Theorem 1 when s <
1 ε
. In particular, for s < ∞,
the asymptotic dependence on ε in the lower-order term is log21ε
. We leave open the question of whether this can be further refined to log1 ε
, which would match a known lower bound on this dependence for all infinite classes (Kulkarni, Mitter, and Tsitsiklis, 1993; Hanneke and Yang, 2015).

The only significant difference between the upper and lower bounds in Theorem 3 is the factor of d in the lower-order term. I conjecture this term can be further refined to O˜s ∧
d ε
, which is known to be sharp for some classes (Hanneke and Yang, 2015), and would fully answer a question posed by Hanneke and Nowak (2019). Beyond this, it is known that a gap between such lower-order terms in general upper and lower bounds is unavoidable if the only dependence on C is via d and s. Specifically, it follows from arguments in Appendix D of Hanneke and Yang (2015) that for some classes C this term should be Θ˜s ∧
d ε while for other classes C the term should be Θ˜s ∧
1 ε 
+ d. Thus, obtaining matching (big-Θ) upper and lower bounds would require introducing a new complexity measure reflecting the distinctions between these types of classes, which we leave as an open question.

## 4 Algorithm And Outline Of The Analysis

We next present the algorithm achieving Theorems 1 and 3 and a sketch of its analysis (the complete formal proof is given in Appendix E). Before stating the algorithm, we first introduce a few additional definitions and convenient notational conventions. Error and disagreement regions: For any function h : *X → {*0, 1}, define its *error region* ER(h) := {(x, y) *∈ X × {*0, 1} : h(x) ̸= y}.

In particular, note that erP (h) = P(ER(h)). For any set V ⊆ C define the *region of disagreement*:
DIS(V ) := {x ∈ X : ∃f, g ∈ *V, f*(x) ̸= g(x)}. For any two functions f, g : *X → {*0, 1}, abbreviate by {f ̸= g} := {x ∈ X : f(x) ̸= g(x)} their *pairwise disagreement region*. Overloaded set notation: For convenience, we adopt a convention of treating sets A ⊆ X as notationally interchangeable with their labeled extension A × {0, 1*} ⊆ X × {*0, 1}. For instance, for functions f, g, h : *X → {*0, 1}, we may write ER(h) ∩ {f ̸= g}, which, by the above convention, is interpreted as ER(h)∩({f ̸= g}×{0, 1}). We also overload notation for set intersections to allow for intersections of sets with *sequences*: that is, for any set Z, sequence S = {z1, . . . , zm*} ∈ Z*m, and set A ⊆ Z, we define S∩A as the subsequence {zi: i ≤ *m, z*i ∈ A}, and likewise S \A := S∩(Z \A).

We also apply these conventions in combination: i.e., for a sequence S ∈ (*X × {*0, 1})
m and a set
∆ ⊆ X , we define S ∩ ∆ := S ∩ (∆ × {0, 1}) and S \ ∆ := S ∩ ((X \ ∆) × {0, 1}).

Empirical estimates: We will make use of *empirical estimates* of quantities such as erP (h)
and PX(f ̸= g). For any set Z and sequence S = {z1, . . . , zm*} ∈ Z*m, for any set A ⊆ Z,
define the *empirical measure*: PˆS(A) := 1m |S ∩ A| =1m Pm i=1 1[zi ∈ A]. Again, we also apply these conventions in combination: i.e., for S ∈ (*X × {*0, 1})
∗and ∆ ⊆ X , we define PˆS(∆) := PˆS(∆ × {0, 1}). For any sequence S ∈ (*X × {*0, 1})
∗and function h : *X → {*0, 1},
define its *empirical error rate* (or *empirical risk*): ˆerS(h) := PˆS(ER(h)).

Decision lists: We will often express *decision-list* aggregations of functions f, g : *X → {*0, 1}.

For instance, for any set ∆ ⊆ X , we may write h = f1X \∆ + g1∆ to express a function h with h(x) = f(x) for x /∈ ∆ and h(x) = g(x) for x ∈ ∆.

## 4.1 The Avid Agnostic Algorithm: Adaptive Variance Isolation By Disagreements

We are now ready to describe the algorithm achieving the upper bounds in Theorems 1 and 3 (for full formality, some additional technical minutiae for the definition are given in Section C). Fix any values *ε, δ* ∈ (0, 1) (the error and confidence parameters input to the learner). Fix any distribution P (unknown to the learner) and let (X1, Y1), . . . ,(Xm, Ym) be independent P-
distributed random variables (for any sufficiently large m, quantified explicitly in Theorem 5). The algorithm is stated in Figure 1, expressed in terms of certain quantities and data subsets defined as follows.2 Let C := 11 10 
, N := logC
2ε
, and for each k ∈ N define εk := C
1−kand mk := Θ1 εk d log1 εk
+ log1δ
 (see Section C for the precise constants). In Step 3, C
′
denotes an appropriate universal constant (see Section C). As defined in Figure 1, the algorithm makes use of different portions of the data (S
1 k, S
2 k, S
3 k,i, S
4 k) for different purposes, and to complete the definition of the algorithm we next specify how these data subsets are defined in the algorithm.

We first split the initial 2M1 := 2PN +1 k=1 mk examples {(X1, Y1)*, . . . ,*(X2M1
, Y2M1
)} into consecutive disjoint contiguous segments S
11
, . . . , S1 N +1, S41
, . . . , S4 N +1, with the segments S
1 kand S
4 k being of size mk. The algorithm also allocates disjoint segments (S
2 k, S
3 k,i) of the remaining data
{(Xi, Yi) : 2M1 < i ≤ m}, but does so *adaptively* during its execution. Specifically, if and when the algorithm reaches Step 2 with a value k, or reaches Step 9 (in which case let k = N + 1), for the value ik and the set ∆ikas defined at that time in the algorithm, it constructs a data subset S
2 k
, allocating to S
2kthe next m′kconsecutive examples which have not yet been allocated to any data subset S
1 k′ ,
S
2 k′ , S
3k′,i′ , S
4 k′ (i.e., *fresh*, previously-unused, examples), where, letting pˆk := 2PˆS4k
(∆ik), we define m′k
:= Θpˆk ε 2 k d + log3+N−k δ(see Section C for the precise constants). Similarly, if and when the algorithm reaches Step 5 with some values of (k, i), it constructs a data subset S
3 k,i, allocating to S
3 k,i the next mk consecutive examples which have not yet been allocated.

Algorithm Aavid Input: Error parameter ε, Confidence parameter δ, Unlabeled data X1*, . . . , X*m Output: Classifier hˆ 0. Initialize i = i1 = 0, ∆0 = ∅, V0 = C
1. For k = 1*, . . . , N*
2. Query all examples in S
1 k ∩ Dk−1 \ ∆ikand S
2 k ∩ ∆ik 3. Vk ←
nh ∈ Vk−1 : ˆer1,2 k
(h) ≤ ˆer1,2 k
(hˆk) + εk C′
o

4. If Vk = ∅ or ˆer1,2
k
(hˆk) < minh∈Vkˆer1,2
k
(h) −
εk
4C′, Then Return hˆ := hˆk
5. While maxf,g∈Vk PˆS3k,i
({f ̸= g} \ ∆i) > εk+2
6. (*f, g*) ← argmax(f
′,g′)∈V
2
k
PˆS3k,i
({f
′ ̸= g
$$\}\setminus\Delta_{i}\rbrace$$
7. ∆i+1 ← ∆i ∪ {f ̸= g}, and update i ← i + 1 8. ik+1 ← i
9. Query all examples in S
1
N +1 ∩ DN \ ∆iN +1 and S
2
N +1 ∩ ∆iN +1 and Return hˆ := hˆN +1 Figure 1: The AVID Agnostic algorithm. Notations N, Dk−1, εk, hˆk, S
1 k, S
2 k, S
3 k,i, ˆer1,2 kdefined in the text.

To complete the definition of the algorithm, we define Dk−1, ˆer1,2
k
, and hˆk, appearing in the algorithm,
as follows. For each value of k encountered in the 'For' loop, as well as for k = N + 1 in the case
the algorithm reaches Step 9, define (where Vk−1 and ∆ikare as defined in the algorithm):
Dk−1 := DIS(Vk−1),
$$D_{k-1}:=\texttt{D3}(V_{k-1}),$$ $$\forall h,\;\hat{\epsilon}_{k}^{1,j}(h):=\hat{P}_{S_{k}^{1}}(\text{ER}(h)\cap D_{k-1}\setminus\Delta_{i_{k}})+\hat{P}_{S_{k}^{2}}(\text{ER}(h)\cap\Delta_{i_{k}}),\tag{1}$$ $$V_{k-1}^{\nu_{k}}:=\{\texttt{f1}_{\{f=g\}\cup\Delta_{i_{k}}}+h_{1}\texttt{1}_{\{f\neq g\}\setminus\Delta_{i_{k}}}+h_{2}\texttt{1}_{\Delta_{i_{k}}}:f,g\in V_{k-1},h_{1},h_{2}\in\mathbb{C}\},$$ (2) $$\text{and}\quad\hat{h}_{k}:=\texttt{argmin}\;\epsilon_{k}^{1,j}\;(h).\tag{3}$$
$${\mathfrak{x}}{-1}\,\},$$
$$(3)$$

h∈V
(4)
k−1
This completes the definition of the Aavid algorithm.

We remark that the examples in S
3 k,i and S
4 kare *never queried* in the algorithm, and thus the algorithm
(necessarily) only uses the unlabeled Xi values in these data subsets (to estimate certain marginal PX probabilities), so in fact these can be regarded as *unlabeled* data subsets. Similarly, the algorithm only queries a *portion* of S
1 kand S
2 k, and the remaining unqueried portions are in fact *never used* by the algorithm. For notational simplicity, we do not make these facts explicit in the notation. Description of the algorithm: We briefly summarize the behavior of the algorithm (with explanations following in Section 4.2). As the algorithm iterates over rounds k of the 'For' loop, it maintains a partition of the space into a region ∆ikand its complement X \ ∆ik. In each round, the algorithm refines a set Vk of surviving concepts from C, aiming to prune out suboptimal concepts (Step 3).

There are two crucial aspects of this, both in how the estimates of erP (h) are defined, and in the choice of function hˆk to which we compare. For the purpose of error estimation, in Step 2 it queries a number of random examples in X \ ∆ik(or rather, the slightly smaller region Dk−1 \ ∆ik, since examples in X \ Dk−1 are uninformative for estimating error *differences*) and a number of random examples in ∆ik. It uses the examples from each of the two regions to estimate the error rate of each h in that region, and combines these two estimates into an overall error estimate ˆer1,2 k(h) as in
(1). It then prunes suboptimal concepts from Vk−1, removing all h ∈ Vk−1 having estimated error ˆer1,2 k
(h) > ˆer1,2 k
(hˆk) + εk C′. The reason ˆer1,2 k
(h) estimates error rates in the two regions separately is that, as it will turn out, we require a disproportionately larger number of samples to accurately estimate the error rates in the region ∆ikcompared to the complement X \ ∆ik: for the latter, we use the samples in S
1k ∩ Dk−1 \ ∆ik(queried in Step 2), where S
1 k has a modest size mk = Θ˜d εk
,
while for the former we use the samples in S
2 k 
∩ ∆ik(also queried in Step 2), where S
2 khas a potentially larger size m′k roughly Θ˜PX 
(∆ik
)d ε 2 k
. The other crucial aspect in Step 3 is how we define the function hˆk to which we compare. For this, rather than (the seemingly-natural idea of) simply comparing to the smallest ˆer1,2 k(h) among h ∈ Vk−1, we instead compare to an even smaller value:
the smallest ˆer1,2 k
(h) among a more-complex class V
(4)
k−1 defined in (2), comprised of decision list functions which use one concept h2 for predictions in ∆ik
, and use (equivalently) a majority vote of three concepts *f, g, h*1 for predictions in X \ ∆ik
. hˆk is defined as a minimizer of ˆer1,2 k in V
(4)
k−1
, as in
(3). This use of a more-complex comparator function is critical for certain parts of the proof (namely, keeping PX (∆ik) small). However, given that hˆk is chosen from a more-complex class, it becomes possible that hˆk may be *substantially better* than all h ∈ Vk. In this event, the algorithm terminates early and returns hˆk (Step 4). Otherwise, if it makes it past this early-stopping case, its next objective is to define the region ∆ik+1 for use in the next iteration. This occurs in the '*While*' loop (Steps 5-7).

On each round of this loop, it uses a fresh data set S
3k,i of size mk = Θ˜d εk to check whether there exist *f, g* ∈ Vk significantly *distant* from each other in the region X \ ∆i (Step 5). If so, it adds their pairwise disagreement region {f ̸= g} to the ∆i region to define ∆i+1 and increments i (Step 7). It repeats this until no such pair *f, g* exists, at which time it defines ik+1 = i (Step 8) and proceeds to the next iteration of the 'For' loop. After N = O(log(1/ε)) such iterations, it returns hˆN +1 (Step 9).

We note that the algorithm's returned classifier hˆ might not be an element of C (known as an *improper* learner), but rather can be represented as a (shallow) *decision list* of concepts from C. This aspect is quite important to certain parts of the proof, and we leave open the question of whether Theorems 1 and 3 are achievable by a proper learner (see Appendix G). We also remark that the Dk−1 set is *only* needed for establishing Theorem 3: the algorithm achieves the query complexity bound in Theorem 1 even if we replace Dk−1 with the full space X everywhere.

## 4.2 Principles And Outline Of The Proof

Next we explain the high-level principles underlying the design of the algorithm, highlighting the two key innovations compared to previous approaches, which enable the improved query complexity guarantee (namely, separating out the ∆ikregions, and the definition of hˆk).

Empirical localization: The principles underlying the design of the algorithm begin with a familiar principle from statistical learning: *empirical localization* (Koltchinskii, 2006; Bartlett, Bousquet, and Mendelson, 2005). Specifically, the uniform Bernstein inequality (Lemma 7)
implies that for an i.i.d. data set S, the sample complexity of uniform concentration of differences |( ˆerS(f) − ˆerS(g)) − (erP (f) − erP (g))| becomes smaller when the *diameter* diam(C) =
supf,g∈C PX (f ̸= g) of the concept class is small, noting that PX (f ̸= g) bounds the *variance* of loss differences 1[f(x) ̸= y] − 1[g(x) ̸= y]. Quantitatively, for any 0 < ε′ < diam(C), Θ˜d diam(C)
(ε
′)
2samples S suffice to guarantee |( ˆerS(f) − ˆerS(g)) − (erP (f) − erP (g))| ≤ ε
′. This fact leads to a natural well-known algorithmic principle, wherein we can *prune* from C concepts h having ˆerS(h) − minh′∈C ˆerS(h
′) > ε′(as the above inequality implies these verifiably have suboptimal error rates), leaving a subset V
′1 of surviving concepts, while preserving h
⋆ ∈ V
′
1, where h
⋆:= argminh∈C erP (h). Moreover, if these surviving concepts V
′1 have diam(V
′
1
) < diam(C), we get an *improved* concentration guarantee for ˆerS(f) − ˆerS(g) among f, g ∈ V
′
1from the uniform Bernstein inequality, which enables us to prune *even more* concepts from V
′
1, leaving a set V
′
2 of surviving concepts, and so on for V
′
3, V ′
4*, . . .*. Quantitatively, we can combine this with a schedule of resolutions εk, so that as long as h
⋆ ∈ V
′k−1 and diam(V
′
k−1
) ≤ εk, an i.i.d. data set S
1 kof size mk = Θ˜d εk
= Ω˜d diam(V
′
k−1)
ε 2k suffices to guaranteeˆerS1k
(f) − ˆerS1k
(g)−erP (f) − erP (g) ≤
εk C′, enabling us to further reduce to a subset V
′
k 
=
nh ∈ V
′
k−1
: ˆerS1k
(h) ≤ minh′∈V
′
k−1 ˆerS1k
(h
′) + εk C′
ofor which all h ∈ V
′
khave erP (h) − erP (h
⋆) ≤ 2 εk C′, while preserving h
⋆ ∈ V
′
k. Iterating this N = ΘlogC
1ε times
(recalling εk = C
1−k) results in a subset V
′
Nof concepts h with erP (h) − erP (h
⋆) ≤ ε.

Disagreement-based active learning: An additional observation, underlying many active learning algorithms (*disagreement-based* methods), is that the above argument still holds while replacing ˆerS1k
(h) with PˆS1k
(ER(h)∩D′k−1
), where D′k−1
:= DIS(V
′
k−1
). To see this, note that ∀*h, h*′ ∈ V
′
k−1, PˆS1k
(ER(h)∩D′k−1
)−PˆS1k
(ER(h
′)∩D′k−1
) = ˆerS1k
(h)− ˆerS1k
(h
′). Thus, we may equivalently define V
′
k 
=
nh ∈ V
′
k−1
: PˆS1k
(ER(h) ∩ D′k−1
) ≤ minh′∈V
′
k−1 PˆS1k
(ER(h
′) ∩ D′k−1
) + εk C′
o. Moreover, as long as diam(V
′
k−1
) ≤ εk, we have PX(D′k−1
) ≤ sεk (Hanneke and Yang, 2015). Since the quantities in V
′k only rely on the labels of examples in D′k−1 ∩ S
1 k
, constructing V
′k only requires a number of queries O(sεkmk) ∧ mk. Summing over k, these queries total to at most the claimed lower-order term in Theorem 3 (though note that even without this D′k−1 refinement we still recover the lower-order term from Theorem 1). So far, this is all essentially standard reasoning commonly followed in the prior literature on active learning (e.g., Hanneke, 2009b, 2014; Koltchinskii, 2010).

Handling non-shrinking diameter: However, the above algorithmic principle breaks down if we reach a k with diam(V
′k−1
) ̸= O(εk). This failure can easily occur in the agnostic setting, where it is possible for the set V
′
k−1above to contain multiple relatively-good functions *f, g* which are nevertheless far from each other.3 This is the motivation for the *first key innovation* in Aavid:
namely, if we ever reach such a k, where the Vk set does not naturally have diam(Vk) ≤ εk+1 (as tested in Step 5), the algorithm *removes* a portion of the space X to *artificially* reduce the diameter.

Specifically, it identifies a pair *f, g* ∈ Vk with PX(f ̸= g) > εk+1 (intuitively, an *obstruction* to having low diameter) and *separates out* their pairwise disagreement region {f ̸= g} from the region of focus of the algorithm (Steps 5-7).4 Having set aside this region, the algorithm continues, focusing on the remaining set *X \ {*f ̸= g}. This step is repeated, and these set-aside regions
{f ̸= g} are altogether captured in the set ∆i (Step 7). Thus, we repeatedly find pairs *f, g* ∈ Vk with PX({f ̸= g} \ ∆i) > εk+1 (Steps 5-6) and add {f ̸= g} to ∆i (Step 7) until the diameter of Vk on X \ ∆iis reduced below εk+1. At that point, the algorithm proceeds to the next round (k ← k + 1). On the next round k, since we have (artificially) ensured the diameter of Vk−1 is at most εk in the region X \ ∆ik, the uniform Bernstein argument implies mk examples S
1 ksuffice to guarantee every *f, g* ∈ Vk−1 have PˆS1k
(ER(f) ∩ Dk−1 \ ∆ik) − PˆS1k
(ER(g) ∩ Dk−1 \ ∆ik) within
±
εk 2C′ of P(ER(f) \ ∆ik) − P*(ER(*g) \ ∆ik).

Error in the ∆ik**region:** There remains the issue of estimating error rates in the ∆ik isolated region. For this, the algorithm uses a data set S
2 kof size m′k ≈ d PX(∆ik
)
ε 2 k
, queries all examples in S
2 k∩∆ik, and uses these to estimate the error rates P(ER(h)∩∆ik) in the ∆ikregion. By a refinement of the uniform convergence bound of Talagrand (1994) accounting for an *envelope* set ∆ik(Lemma 8),
this number m′k of examples suffices to ensure
PˆS2k
(ER(h) ∩ ∆ik) − P*(ER(*h) ∩ ∆ik)
 ≤εk 4C′ for every h ∈ C. Combining this with the above error-differences estimates in the X \∆ikregion, we can guarantee that the functions *f, g* ∈ Vk−1 haveˆer1,2 k
(f) − ˆer1,2 k
(g)− (erP (f) − erP (g)) ≤
εk C′,
recalling the definition of ˆer1,2 kfrom (1). Altogether, we conclude that, as long as h
⋆ ∈ Vk−1, a set V
′′
k:= h ∈ Vk−1 : ˆer1,2 k
(h) ≤ minh′∈Vk−1 ˆer1,2 k
(h
′) + εk C′
	would contain only functions h satisfying erP (h) − erP (h
⋆) ≤ 2 εk C′ while preserving h
⋆ ∈ V
′′
k
. The actual definition of Vk in Step 3 is only slightly different from this, for reasons we discuss next.

Bounding the size of ∆ik: Since the number of queries in S
2 k∩∆ikis ≈ dPX (∆ik)
2/ε2k, if we hope to achieve a query complexity with lead term O˜d β 2 ε 2it is crucial to guarantee PX (∆ik) = O(β).

This is the motivation for the *second key innovation* in Aavid: defining the update in Vk by comparison to the function hˆk in (3), rather than the best h
′ ∈ Vk−1. This turns out to be the most subtle part of the argument, requiring precise choices in the design of the algorithm. The essential argument is as follows. Suppose the algorithm reaches Step 6 for some (*k, i*), so that it will add {f ̸= g} to the
∆i region. We then want to argue that P(ER(h
⋆) ∩ {f ̸= g} \ ∆i) = Ω(P({f ̸= g} \ ∆i)): that is, each time we add to ∆i, we *chop off* a portion of ER(h
⋆) of size (under P) proportional to the increase in PX (∆i). Clearly if we can show this is always the case, we will inductively maintain PX (∆i) = O(β), resulting in the claimed leading term in the query complexity. Now, to show this indeed occurs, we first note that one of *f, g* must err on at least half of {f ̸= g} \ ∆ik; w.l.o.g.

suppose it is f: that is, P(ER(f) ∩ {f ̸= g} \ ∆ik) ≥
1 2 PX({f ̸= g} \ ∆ik). Now consider a function f
⋆ = f1{f=g}\∆ik
+ h
⋆1{f̸=g}\∆ik
+ f1∆ik which replaces f by h
⋆in the region
{f ̸= g} \ ∆ik. Note that, if h
⋆ ∈ Vk−1, then f
⋆ ∈ V
(4)
k−1 defined in (2). Since hˆk has minimal ˆer1,2 k 3For instance, for C the class of intervals 1[a,b] on R, with PX = Uniform([0, 1]) and P(Y = 1|X) =
1[0,1/4]∪[3/4,1](X), the concepts 1[0,1/4] and 1[3/4,1] are both optimal among C, yet distance 1/2 apart.

4This reasoning is somewhat reminiscent of the motivation for the *splitting* approach to active learning
(Dasgupta, 2005), differing only in how we resolve the obstruction: whereas splitting would resolve it with queries to eliminate one element from each obstructing pair, here we resolve it by subtracting the pairwise disagreement region from the region of focus X \∆i (see Appendix A.2.3). This idea is also related to a technique of Hanneke, Larsen, and Zhivotovskiy (2024b) for agnostic passive learning, discussed in Appendix A.3.

among V
(4)
k−1
, and f ∈ Vk implies ˆer1,2 k
(f) ≤ ˆer1,2 k
(hˆk) + εk C′, extending the above concentration of ˆer1,2 kdifferences to functions in V
(4)
k−1(with appropriate adjustment of constants in mk, m′k) implies erP (f) − erP (f
⋆) ≤ 2 εk C′. Thus, since f
⋆and f only disagree on {f ̸= g} \ ∆ik, we have

$\frac{1}{2}P_{X}(\{f\neq g\}\setminus\Delta_{i_{k}})-P(\mbox{ER}(h^{\star})\cap\{f\neq g\}\setminus\Delta_{i_{k}})$  $\leq P(\mbox{ER}(f)\cap\{f\neq g\}\setminus\Delta_{i_{k}})-P(\mbox{ER}(h^{\star})\cap\{f\neq g\}\setminus\Delta_{i_{k}})$
$\mathbf{x}\cdot(\mathbf{f})=\mathbf{x}\mathbf{i}$
≤ P(ER(f) ∩ {f ̸= g} \ ∆ik
) − P(ER(h
⋆) ∩ {f ̸= g} \ ∆ik
) = erP (f) − erP (f
⋆) ≤ 2 εk C′
.

In other words, P(ER(h
⋆) ∩ {f ̸= g} \ ∆ik) ≥
1 2 PX({f ̸= g} \ ∆ik) − 2 εk C′. This is almost what we wanted, aside from having ∆ikin place of ∆i. We then argue P(ER(h
⋆) ∩ {f ̸= g} \ ∆i) ≥
P(ER(h
⋆) ∩ {f ̸= g} \ ∆ik) − PX({f ̸= g} \ ∆ik) + PX({f ̸= g} \ ∆i), which (by the above)
is at least PX({f ̸= g} \ ∆i) −
1 2 PX({f ̸= g} \ ∆ik
) − 2 εk C′. Since both *f, g* ∈ Vk−1, we know PX({f ̸= g} \∆ik
) ≤ εk, so that this lower-bound is at least PX({f ̸= g} \∆i)−
εk 2 
−2 εk C′. On the other hand, for appropriate constants in mk, the condition in Step 5 allows us to upper-bound εk in terms of PX({f ̸= g}\∆i): namely, PX({f ̸= g}\∆i) ≥
εk c for c with C
2 < c ≤
3 2
∧
C
′
9
. Thus, we have P(ER(h
⋆)∩ {f ̸= g} \∆i) ≥1 −
c 2 
− 2 c C′PX({f ̸= g} \∆i). Since each ∆ikis a union of such (disjoint) {f ̸= g} \ ∆i regions (*i < i*k), β ≥ P(ER(h
⋆) ∩ ∆ik) ≥1 −
c 2 
− 2 c C′PX(∆ik).

The early stopping case: The above argument for PX(∆ik) = O(β) hinges on having h
⋆ ∈ Vk−1.

However, since hˆk is a more-complex function than h
⋆, there is a chance that h
⋆ ∈/ Vk after Step 3. For this reason, we have added the early stopping case in Step 4. By using slightly tighter concentration inequalities than used to update Vk, this step effectively *tests* that hˆk is not so much better than all concepts in Vk−1 that h
⋆ might have been removed. Thus, if we make it past Step 4, we maintain h
⋆ ∈ Vk so that the above argument applies on the next round. On the other hand, in the event that this test fails, we have effectively verified that hˆk is at least *slightly better* than all concepts in Vk−1
(including h
⋆), and we can safely return hˆk in this case.

Overall behavior: The effective overall behavior of the algorithm is to *isolate* in the region ∆ik the most-challenging part of the error estimation problem, due to the high variance (diameter) of the error differences in that region. It then allocates a disproportionately larger number of queries S
2k ∩ ∆ik to this region, toward estimating the error rates there. By comparing with the function hˆk (which separately optimizes errors in pairwise difference regions {f ̸= g} \ ∆ik) in the definition of Vk, we can maintain that ∆ik never grows larger than O(β), so that the number of queries in S
2k ∩ ∆ik does not grow excessively large. The remaining region X \ ∆ik enjoys the property that the set Vk−1 has diameter ≤ εk, so that we can easily estimate error differences in this region by a uniform Bernstein inequality. Altogether, after at most N = Olog1ε rounds, this achieves the objective of ε excess error rate, while using a number of queries as stated in the query complexity bound in Theorem 3. The formal proof is given in Appendix E.

## 5 Conclusions And Summary Of The Appendices

This work resolves a long-standing open question of central importance to the theory of active learning, proving that *every* concept class benefits from active learning in the non-realizable case. Quantitatively, we establish a new sharp upper bound on the optimal query complexity, with leading term that is smaller than that of passive learning by a factor proportional to the best-in-class error rate. The appendices include the formal proofs, along with additional contents. Appendix A presents a thorough summary of related work and background on the theory of active learning, as well as other works with techniques related to those used here. Appendix C presents remaining minutiae for the definition of Aavid, along with a more-detailed version of Theorem 3, including formal claims regarding the number of *unlabeled* examples. Appendix E presents the formal proof of Theorem 3.

Appendix F presents distribution-dependent refinements of Theorem 3, which replace the star number s with certain P-dependent complexity measures: variants of the disagreement coefficient. We further argue that the disagreement coefficient θP (ε), as originally defined by Hanneke (2007b), provably cannot be attained as a replacement for s in the lower-order term (by any algorithm), while on the other hand Aavid *does* achieve a lower-order term O˜(θP (β + ε)
2d). We also present subregion-based refinements of the algorithm and analysis, based on techniques of Zhang and Chaudhuri (2014).

Appendix G presents extensions (*multiclass* classification, *stream-based* active learning), along with several open questions and future directions.

## References

N. Ailon, R. Begleiter, and E. Ezra. Active learning using smooth relative regret approximations with applications. *Journal of Machine Learning Research*, 15(3):885–920, 2014.

D. Angluin. Queries and concept learning. *Machine Learning*, 2(4):319–342, 1987. doi: 10.1007/
BF00116828. URL https://doi.org/10.1007/BF00116828.

J. Asilis, S. Devic, S. Dughmi V. Sharan, and S.-H. Teng. Proper learnability and the role of unlabeled data. In Proceedings of the 36th *International Conference on Algorithmic Learning Theory*, 2025a.

J. Asilis, M. M. Høgsgaard, and G. Velegkas. Understanding aggregations of proper learners in multiclass classification. In Proceedings of the 36th International Conference on Algorithmic Learning Theory, 2025b.

A. C. Atkinson and A. N. Donev. *Optimum Experimental Designs*. Clarendon Press, 1992. P. Awasthi, V. Feldman, and V. Kanade. Learning using local membership queries. In Proceedings of the 26th *Conference on Learning Theory*, 2013.

P. Awasthi, M.-F. Balcan, and P. M. Long. The power of localization for efficiently learning linear separators with noise. In Proceedings of the 46th *ACM Symposium on the Theory of Computing*,
2014.

M.-F. Balcan and A. Blum. A discriminative model for semi-supervised learning. Journal of the ACM, 57(3):1–46, 2010.

M.-F. Balcan and S. Hanneke. Robust interactive learning. In Proceedings of the 25th Conference on Learning Theory, 2012.

M.-F. Balcan and P. M. Long. Active and passive learning of linear separators under log-concave distributions. In Proceedings of the 26th *Conference on Learning Theory*, 2013.

M.-F. Balcan and H. Zhang. Sample and computationally efficient learning algorithms under sconcave distributions. 2017.

M.-F. Balcan, A. Beygelzimer, and J. Langford. Agnostic active learning. In NIPS Workshop on Foundations of Active Learning, 2005.

M.-F. Balcan, A. Beygelzimer, and J. Langford. Agnostic active learning. In *Proceedings of the* 23rd International Conference on Machine Learning, 2006.

M.-F. Balcan, A. Broder, and T. Zhang. Margin based active learning. In *Proceedings of the* 20th Conference on Learning Theory, 2007.

M.-F. Balcan, A. Beygelzimer, and J. Langford. Agnostic active learning. Journal of Computer and System Sciences, 75(1):78–89, 2009.

M.-F. Balcan, S. Hanneke, and J. Wortman Vaughan. The true sample complexity of active learning.

Machine Learning, 80(2–3):111–139, 2010.

M.-F. Balcan, A. Blum, S. Hanneke, and D. Sharma. Robustly-reliable learners under poisoning attacks. In Proceedings of the 35th *Conference on Learning Theory*, 2022.

P. Bartlett, M. I. Jordan, and J. McAuliffe. Convexity, classification, and risk bounds. Journal of the American Statistical Association, 101(473):138–156, 2006.

P. L. Bartlett, O. Bousquet, and S. Mendelson. Local rademacher complexities. *The Annals of* Statistics, 33(4):1497–1537, 2005.

E. Baum. Neural net algorithms that learn in polynomial time from examples and queries. IEEE
Transactions on Neural Networks, 2(1):5–19, 1991.

E. Baum and K. Lang. Query learning can work poorly when a human oracle is used. In Proceedings of the International Joint Conference in Neural Networks, 1992.

G. Bennett. Probability inequalities for the sum of independent random variables. *Journal of the* American Statistical Association, 57(297):33–45, 1962.

S. Bernstein. On a modification of Chebyshev's inequality and of the error formula of Laplace.

Annales Scientifiques de l'Institut de la Société des Savants d'Ukraine, Section de Mathématiques, 1(4):38–49, 1924.

A. Beygelzimer, S. Dasgupta, and J. Langford. Importance weighted active learning. In Proceedings of the 26th *International Conference on Machine Learning*, 2009.

A. Beygelzimer, D. Hsu, J. Langford, and T. Zhang. Agnostic active learning without constraints. In Advances in Neural Information Processing Systems 23, 2010.

S. Boucheron, G. Lugosi, and P. Massart. Concentration Inequalities: A Nonasymptotic Theory of Independence. Oxford University Press, 2013.

O. Bousquet. A Bennett concentration inequality and its application to suprema of empirical processes.

Comptes Rendus Mathematique, 334(6):495–500, 2002.

O. Bousquet and N. Zhivotovskiy. Fast classification rates without standard margin assumptions.

Information and Inference: A Journal of the IMA, 10(4):1389–1421, 2021.

O. Bousquet, S. Hanneke, S. Moran, and N. Zhivotovskiy. Proper learning, Helly number, and an optimal SVM bound. In Proceedings of the 33rd *Conference on Learning Theory*, 2020.

N. Brukhim, D. Carmon, I. Dinur, S. Moran, and A. Yehudayoff. A characterization of multiclass learnability. In Proceedings of the 63rd Annual IEEE Symposium on Foundations of Computer Science, 2022.

B. G. Buchanan. Scientific theory formation by computer. In *Computer Oriented Learning Processes*,
pages 515–534. 1976.

A. D. Bull. Spatially-adaptive sensing in nonparametric regression. *The Annals of Statistics*, 1:41–62, 2013.

G. Cavallanti, N. Cesa-Bianchi, and C. Gentile. Learning noisy linear classifiers via adaptive and selective sampling. *Machine Learning*, 83:71–102, 2011.

O. Chapelle, B. Scholkopf, and A. Zien. *Semi-Supervised Learning*. Adaptive Computation and Machine Learning Series. MIT Press, 2006.

K. Chaudhuri, S. M. Kakade, P. Netrapalli, and S. Sanghavi. Convergence rates of active learning for maximum likelihood estimation. In *Advances in Neural Information Processing Systems 28*, 2015.

H. Chernoff. A measure of asymptotic efficiency for tests of a hypothesis based on the sum of observations. *The Annals of Mathematical Statistics*, pages 493–507, 1952.

D. Cohn, L. Atlas, and R. Ladner. Improving generalization with active learning. *Machine Learning*,
15(2):201–221, 1994.

D. A. Cohn, Z. Ghahramani, and M. I. Jordan. Active learning with statistical models. Journal of Artificial Intelligence Research, 4:129–145, 1996.

C. Cortes, G. DeSalvo, C. Gentile, M. Mohri, and N. Zhang. Active learning with region graphs. In Proceedings of the 36th *International Conference on Machine Learning*, 2019a.

C. Cortes, G. DeSalvo, C. Gentile, M. Mohri, and N. Zhang. Region-based active learning. In Proceedings of the 22nd *International Conference on Artificial Intelligence and Statistics*, 2019b.

C. Cortes, G. DeSalvo, C. Gentile, M. Mohri, and N. Zhang. Active learning with disagreement graphs. In Proceedings of the 36th *International Conference on Machine Learning*, 2019c.

C. Cortes, G. DeSalvo, C. Gentile, M. Mohri, and N. Zhang. Adaptive region-based active learning.

In Proceedings of the 37th *International Conference on Machine Learning*, 2020.

A. Daniely and S. Shalev-Shwartz. Optimal learners for multiclass problems. In *Proceedings of the* 27th *Conference on Learning Theory*, 2014.

S. Dasgupta. Analysis of a greedy active learning strategy. In Advances in Neural Information Processing Systems 17, 2004.

S. Dasgupta. Coarse sample complexity bounds for active learning. In *Advances in Neural Information* Processing Systems 18, 2005.

S. Dasgupta. The two faces of active learning. *Theoretical Computer Science*, 412(19), 2011. S. Dasgupta, A. T. Kalai, and C. Monteleoni. Analysis of perceptron-based active learning. In Proceedings of the 18th *Conference on Learning Theory*, 2005.

S. Dasgupta, D. Hsu, and C. Monteleoni. A general agnostic active learning algorithm. In Advances in Neural Information Processing Systems 20, 2007.

M. H. DeGroot. Uncertainty, information, and sequential experiments. *The Annals of Mathematical* Statistics, 33(2):404–419, 1962.

O. Dekel, C. Gentile, and K. Sridharan. Selective sampling and active learning from single and multiple teachers. *Journal of Machine Learning Research*, 13(9):2655–2697, 2012.

G. DeSalvo, C. Gentile, and T. S. Thune. Online active learning with surrogate loss functions.

Advances in Neural Information Processing Systems 34, 2021.

L. Devroye and G. Lugosi. Lower bounds in pattern recognition and learning. *Pattern Recognition*,
28:1011–1018, 1995.

I. Diakonikolas, D. Kane, and M. Ma. Active learning of general halfspaces: Label queries vs membership queries. In *Advances in Neural Information Processing Systems 37*, 2024.

R. M. Dudley. *Uniform Central Limit Theorems*. Cambridge University Press, 1999.

S. Efromovich. Sequential design and estimation in heteroscedastic nonparametric regression.

Sequential Analysis, 26(1):3–25, 2007.

B. Eisenberg. *On the Sample Complexity of PAC-Learning using Random and Chosen Examples*.

PhD thesis, Massachusetts Institute of Technology, 1992.

B. Eisenberg and R. Rivest. On the sample complexity of PAC-learning using random and chosen examples. In Proceedings of the 3rd *Annual Workshop on Computational Learning Theory*, 1990.

R. El-Yaniv and Y. Wiener. On the foundations of noise-free selective classification. *Journal of* Machine Learning Research, 11(5):1605–1641, 2010.

R. El-Yaniv and Y. Wiener. Active learning via perfect selective classification. Journal of Machine Learning Research, 13(2):255–279, 2012.

V. V. Fedorov. *Theory of Optimal Experiments*. Academic Press, 1972.

R. A. Fisher. *The Design of Experiments*. Oliver and Boyd, 1935.

D. J. Foster, A. Rakhlin, D. Simchi-Levi, and Y. Xu. Instance-dependent complexity of contextual bandits and reinforcement learning: A disagreement-based perspective. In *Proceedings of the* 34th Conference on Learning Theory, 2021.

Y. Freund, H. S. Seung, E. Shamir, and N. Tishby. Selective sampling using the query by committee algorithm. *Machine Learning*, 28:133–168, 1997.

E. Friedman. Active learning for smooth problems. In Proceedings of the 22nd Conference on Learning Theory, 2009.

R. Gelbhart and R. El-Yaniv. The relationship between agnostic selective classification, active learning and the disagreement coefficient. *Journal of Machine Learning Research*, 20(33):1–38, 2019.

E. Giné and V. Koltchinskii. Concentration inequalities and asymptotic results for ratio type empirical processes. *The Annals of Probability*, 34(3):1143–1216, 2006.

S. A. Goldman and M. J. Kearns. On the complexity of teaching. Journal of Computer and System Sciences, 50:20–31, 1995.

A. Gonen, S. Sabato, and S. Shalev-Shwartz. Efficient active learning of halfspaces: An aggressive approach. *The Journal of Machine Learning Research*, 14(1):2583–2615, 2013.

S. Hanneke. Teaching dimension and the complexity of active learning. In *Proceedings of the* 20th Conference on Learning Theory, 2007a.

S. Hanneke. A bound on the label complexity of agnostic active learning. In *Proceedings of the* 24th International Conference on Machine Learning, 2007b.

S. Hanneke. Adaptive rates of convergence in active learning. In Proceedings of the 22nd Conference on Learning Theory, 2009a.

S. Hanneke. *Theoretical Foundations of Active Learning*. PhD thesis, Machine Learning Department, School of Computer Science, Carnegie Mellon University, 2009b.

S. Hanneke. Rates of convergence in active learning. *The Annals of Statistics*, 39(1):333–361, 2011. S. Hanneke. Activized learning: Transforming passive to active with improved label complexity.

Journal of Machine Learning Research, 13(5):1469–1587, 2012.

S. Hanneke. Theory of disagreement-based active learning. *Foundations and Trends in Machine* Learning, 7(2–3):131–309, 2014.

S. Hanneke. The optimal sample complexity of PAC learning. *Journal of Machine Learning Research*,
17(38):1–15, 2016a.

S. Hanneke. Refined error bounds for several learning algorithms. Journal of Machine Learning Research, 17(135):1–55, 2016b.

S. Hanneke. The star number and eluder dimension: Elementary observations about the dimensions of disagreement. In Proceedings of the 37th *Conference on Learning Theory*, 2024.

S. Hanneke and A. Kontorovich. Stable sample compression schemes: New applications and an optimal SVM margin bound. In Proceedings of the 32nd *International Conference on Algorithmic* Learning Theory, 2021.

S. Hanneke and S. Kpotufe. A no-free-lunch theorem for multitask learning. *The Annals of Statistics*,
50(6):3119–3143, 2022.

S. Hanneke and R. Nowak. Tutorial on Active Learning: From Theory to Practice. In *The 36*th International Conference on Machine Learning, 2019. URL https://youtu.be/0TADiY7iPAc? t=5865.

S. Hanneke and L. Yang. Negative results for active learning with convex losses. In *Proceedings of* the 13th *International Conference on Artificial Intelligence and Statistics*, 2010.

S. Hanneke and L. Yang. Minimax analysis of active learning. *Journal of Machine Learning Research*,
16(12):3487–3602, 2015.

S. Hanneke and L. Yang. Surrogate losses in passive and active learning. *Electronic Journal of* Statistics, 13(2):4646–4708, 2019.

S. Hanneke, A. Karbasi, S. Moran, and G. Velegkas. Universal rates for active learning. In Advances in Neural Information Processing Systems 37, 2024a.

S. Hanneke, K. G. Larsen, and N. Zhivotovskiy. Revisiting agnostic PAC learning. In Proceedings of the 65th *IEEE Symposium on Foundations of Computer Science*, 2024b.

S. Har-Peled, D. Roth, and D. Zimak. Maximum margin coresets for active and noise tolerant learning.

In Proceedings of the 35th *International Joint Conference on Artificial Intelligence*, 2007.

D. Haussler. Decision theoretic generalizations of the PAC model for neural net and other learning applications. *Information and Computation*, 100:78–150, 1992.

D. Haussler and P. M. Long. A generalization of Sauer's lemma. *Journal of Combinatorial Theory,*
Series A, 71(2):219–240, 1995.

T. Hegedüs. Generalized teaching dimensions and the query complexity of learning. In Proceedings of the 8th Conference on Computational Learning Theory, 1995.

L. Hellerstein, K. Pillaipakkamnatt, V. Raghavan, and D. Wilkins. How many queries are needed to learn? *Journal of the Association for Computing Machinery*, 43(5):840–862, 1996.

M. Hopkins, D. Kane, S. Lovett, and G. Mahajan. Point location and active learning: Learning halfspaces almost optimally. In Proceedings of the 61st Annual IEEE Symposium on Foundations of Computer Science, 2020.

D. Hsu. *Algorithms for Active Learning*. PhD thesis, Department of Computer Science and Engineering, School of Engineering, University of California, San Diego, 2010.

T.-K. Huang, A. Agarwal, D. J. Hsu, J. Langford, and R. E. Schapire. Efficient and parsimonious agnostic active learning. In *Advances in Neural Information Processing Systems 28*, 2015.

J. C. Jackson. An efficient membership-query algorithm for learning DNF with respect to the uniform distribution. *Journal of Computer and System Sciences*, 55(3):414–440, 1997.

M. Kääriäinen. On active learning in the non-realizable case. In *NIPS Workshop on Foundations of* Active Learning, 2005.

M. Kääriäinen. Active learning in the non-realizable case. In *Proceedings of the 17th International* Conference on Algorithmic Learning Theory, 2006.

M. J. Kearns, R. E. Schapire, and L. M. Sellie. Toward efficient agnostic learning. *Machine Learning*,
17:115–141, 1994.

V. Koltchinskii. Local Rademacher complexities and oracle inequalities in risk minimization. The Annals of Statistics, 34(6):2593–2656, 2006.

V. Koltchinskii. Rademacher complexities and bounding the excess risk in active learning. Journal of Machine Learning Research, 11(9):2457–2485, 2010.

S. R. Kulkarni, S. K. Mitter, and J. N. Tsitsiklis. Active learning using arbitrary binary valued queries.

Machine Learning, 11:23–35, 1993.

J. Lewi, R. Butera, and L. Paninski. Sequential optimal design of neurophysiology experiments.

Neural Computation, 21(3):619—-687, 2009.

S. Mahalanabis. A note on active learning for smooth problems. arXiv*:1103.3095*, 2011.

E. Mammen and A.B. Tsybakov. Smooth discrimination analysis. *The Annals of Statistics*, 27(6):
1808–1829, 1999.

P. Massart and E. Nédélec. Risk bounds for statistical learning. *The Annals of Statistics*, 34(5):
2326–2366, 2006.

T. Mitchell. *Version Spaces: An Approach to Concept Learning*. PhD thesis, Stanford University, 1979.

O. Montasser, S. Hanneke, and N. Srebro. VC classes are adversarially robustly learnable, but only improperly. In Proceedings of the 32nd *Conference on Learning Theory*, 2019.

E. Mosqueira-Rey, E. Hernández-Pereira, D. Alonso-Ríos, J. Bobes-Bascarán, and Á. Fernández-Leal.

Human-in-the-loop machine learning: a state of the art. *Artificial Intelligence Review*, 56(4):
3005–3054, 2023.

M. Naghshvar and T. Javidi. Active sequential hypothesis testing. *The Annals of Statistics*, 41(6):
2703–2738, 2013.

B. K. Natarajan. On learning sets and functions. *Machine Learning*, 4:67–97, 1989.

R. D. Nowak. Generalized binary search. In Proceedings of the 46th Allerton Conference on Communication, Control, and Computing, 2008.

R. D. Nowak. The geometry of generalized binary search. *IEEE Transactions on Information Theory*,
57(12), 2011.

F. Olsson. A literature survey of active machine learning in the context of natural language processing.

2009.

L. Paninski. Asymptotic theory of information-theoretic experimental design. *Neural Computation*,
17(7):1480–1507, 2005.

C. S. Peirce. A note on the theory of the economy of research. In Report of the Superintendent of the United States Coast Survey Showing the Progress of the Work for the Fiscal Year Ending June 30, 1876, 1879.

R. J. Popplestone. An experiment in automatic induction. In Proceedings of the Fifth Annual Machine Intelligence Workshop, Edinburgh, pages 203–215, 1969.

N. Puchkin and N. Zhivotovskiy. Exponential savings in agnostic active learning through abstention.

IEEE Transactions on Information Theory, 68(7):4651–4665, 2022.

M. Raginsky and A. Rakhlin. Lower bounds for passive and active learning. In Advances in Neural Information Processing Systems 24, 2011.

P. Ren, Y. Xiao, X. Chang, P.-Y. Huang, Z. Li, B. B. Gupta, X. Chen, and X. Wang. A survey of deep active learning. *ACM Computing Surveys (CSUR)*, 54(9):1–40, 2021.

B. Settles. *Active Learning*. Synthesis Lectures on Artificial Intelligence and Machine Learning, Morgan & Claypool Publishers, 2012.

H. S. Seung, M. Opper, and H. Sompolinsky. Query by committee. In Proceedings of the 5th *Annual* Workshop on Computational Learning Theory, 1992.

H. Shayestehmanesh. Active learning under the Bernstein condition for general losses. Master's thesis, University of Victoria, 2020.

H. Simon. An almost optimal PAC algorithm. In Proceedings of the 28th Conference on Learning Theory, 2015.

H. A. Simon and G. Lea. Problem solving and rule induction: A unified view. In Knowledge and Cognition, pages 105–129. Lawrence Erlbaum Associates, 1974.

R. G. Smith, T. M. Mitchell, R. A. Chestek, and B. G. Buchanan. A model for learning systems. In Proceedings of the 5 th *International Joint Conference on Artificial Intelligence*, pages 338–343, 1977.

M. Talagrand. Sharper bounds for gaussian and empirical processes. *The Annals of Probability*, 22:
28–76, 1994.

S. Tong and D. Koller. Support vector machine active learning with applications to text classification.

Journal of Machine Learning Research, 2(11):45–66, 2001.

C. Tosh and D. Hsu. Diameter-based interactive structure discovery. In *Proceedings of the* 23rd International Conference on Artificial Intelligence and Statistics, 2020.

A. B. Tsybakov. Optimal aggregation of classifiers in statistical learning. *The Annals of Statistics*, 32
(1):135–166, 2004.

G. Turán. Lower bounds for PAC learning with queries. In Proceedings of the 6th *Annual Conference* on Computational Learning Theory, 1993.

L. G. Valiant. A theory of the learnable. *Communications of the ACM*, 27(11):1134–1142, November 1984.

A. W. van der Vaart and J. A. Wellner. *Weak Convergence and Empirical Processes*. Springer, 1996. V. Vapnik and A. Chervonenkis. On the uniform convergence of relative frequencies of events to their probabilities. *Theory of Probability and its Applications*, 16(2):264–280, 1971.

V. Vapnik and A. Chervonenkis. *Theory of Pattern Recognition*. Nauka, Moscow, 1974. M. Vidyasagar. *Learning and Generalization with Applications to Neural Networks*. Springer-Verlag, 2 nd edition, 2003.

A. Wald. *Sequential Analysis*. John Wiley and Sons, New York, 1947. L. Wang. Smoothness, disagreement coefficient, and the label complexity of agnostic active learning.

Journal of Machine Learning Research, 12(7):2269–2292, 2011.

Y. Wang and A. Singh. Noise-adaptive margin-based active learning and lower bounds under Tsybakov noise condition. In Proceedings of the 30th *AAAI Conference on Artificial Intelligence*,
2016.

Y. Wiener, S. Hanneke, and R. El-Yaniv. A compression technique for analyzing disagreement-based active learning. *Journal of Machine Learning Research*, 16(4):713–745, 2015.

S. Yan, K. Chaudhuri, and T. Javidi. Active learning with logged data. In *Proceedings of the 35*th International Conference on Machine Learning, 2018.

S. Yan, K. Chaudhuri, and T. Javidi. The label complexity of active learning from observational data.

In *Advances in Neural Information Processing Systems 32*, 2019.

C. Zhang and K. Chaudhuri. Beyond disagreement-based agnostic active learning. In Advances in Neural Information Processing Systems 27, 2014.

T. Zhang. Statistical behavior and consistency of classification methods based on convex risk minimization. *The Annals of Statistics*, 32(1):56–85, 2004.

T. Zhang. *Mathematical Analysis of Machine Learning Algorithms*. Cambridge University Press, 2023.

T. Zhang and F. Oles. A probability analysis on the value of unlabeled data for classification problems.

In *International Conference on Machine Learning*, 2000.

N. Zhivotovskiy and S. Hanneke. Localization of VC classes: Beyond local Rademacher complexities.

Theoretical Computer Science, 742:27–49, 2018.

X. Zhu, J. Lafferty, and Z. Ghahramani. Combining active learning and semi-supervised learning using Gaussian fields and harmonic functions. In ICML workshop on the Continuum from Labeled to Unlabeled Data in Machine Learning and Data Mining, 2003.

Y. Zhu and R. Nowak. Efficient active learning with abstention. In *Advances in Neural Information* Processing Systems 35, 2022.

## A Survey Of The Theory Of Active Learning And Other Related Work

There is at this time quite an extensive literature on the theory of active learning. We refer the interested reader to the surveys of Hanneke (2014), Dasgupta (2011), and the 2019 ICML tutorial of Hanneke and Nowak (2019) for detailed discussions of classic works in this literature. In this section, we present a brief survey of the subject, with particular emphasis on the parts most-closely related to the present work.

## A.1 A Brief Historical Overview

The literature on active learning has a long history, dating back at least to the classical works on *experiment design* in statistics (Peirce, 1879; Fisher, 1935), wherein the analogous setting to active learning is referred to as *sequential design* (e.g., Wald, 1947; DeGroot, 1962; Fedorov, 1972; Atkinson and Donev, 1992; Efromovich, 2007; Zhang and Oles, 2000; Paninski, 2005; Lewi, Butera, and Paninski, 2009; Bull, 2013; Naghshvar and Javidi, 2013; Chaudhuri, Kakade, Netrapalli, and Sanghavi, 2015). Active learning has also been an important subject within the machine learning literature from the very beginning (e.g., Popplestone, 1969; Simon and Lea, 1974; Buchanan, 1976; Smith, Mitchell, Chestek, and Buchanan, 1977; Mitchell, 1979). Below we briefly mention some of the background of the subject in the *learning theory* literature, before giving detailed background of the literature on agnostic active learning. Membership Queries: In the learning theory literature, the idea of active learning also appeared as a natural variant of the problem of *Exact learning with queries*. Specifically, in this setting, supposing there is an unknown *target concept* h
⋆ ∈ C, the objective of the learner is to *exactly identify* h
⋆. To achieve this goal, the learner has access to an oracle (who knows h
⋆), to which it may pose queries of a given type. The most relevant such queries (to the present work) are *membership queries*: namely, it may construct any x ∈ X and query for the value h
⋆(x) (in later works in machine learning, this is sometimes known as *query synthesis*). Early discussion of this framework and corresponding algorithmic principles appear in the seminal work of Mitchell (1979). General analyses of the number of queries necessary and sufficient to identify h
⋆(i.e., the *query complexity* of Exact learning) were developed in the works of Angluin (1987); Hegedüs (1995); Hellerstein, Pillaipakkamnatt, Raghavan, and Wilkins (1996); Nowak (2008, 2011); Hopkins, Kane, Lovett, and Mahajan (2020), and a related average-case analysis was developed by Dasgupta (2004). Closer to the setting considered in the present work, the idea of learning with membership queries has also been extensively studied in the context of PAC learning in the realizable case. In that setting, the learner observes i.i.d. samples (Xi, Yi) with unknown distribution P, under the assumption that there exists an unknown target concept h
⋆ ∈ C with erP (h
⋆) = 0. The learner is additionally permitted to make membership queries for this concept h
⋆, with the goal of producing a predictor hˆ
having erP (hˆ) ≤ ε with high probability 1 − δ. While most of the literature on PAC learning with membership queries has focused on the benefits of such queries for the *computational* complexity of learning (e.g., Valiant, 1984; Baum, 1991; Jackson, 1997), the literature also contains several works on the number of samples and queries for learning in this setting (e.g., Eisenberg and Rivest, 1990; Eisenberg, 1992; Seung, Opper, and Sompolinsky, 1992; Turán, 1993; Kulkarni, Mitter, and Tsitsiklis, 1993; Diakonikolas, Kane, and Ma, 2024).

Modern Active Learning with Label Queries: While the early literature on PAC learning with membership queries included several strong positive results (exhibiting advantages in both query complexity and computational complexity compared to learning from i.i.d. samples alone), when researchers implemented these algorithms and tried to use them for practical machine learning with a human labeler as the oracle, they found that the instances x ∈ X queried by the learner often turned out to be rather nonsensical, unnatural, or borderline cases between two labels (e.g., Baum and Lang, 1992). As such, human labelers were unable to provide useful answers to the queries, leading to poor performance of the learning algorithm. To address this issue, researchers turned to studying algorithms whose queries are restricted to only *natural* instances x ∈ X , which in most works (with a few notable exceptions, e.g., Awasthi, Feldman, and Kanade, 2013) essentially means x in the support of the marginal distribution PX : i.e., the types of examples that might occur naturally in the population. To actualize this restriction, researchers proposed a simple variant of active learning
(which has become the standard framework in the literature, and is now essentially synonymous with the term *active learning*), in which there are i.i.d. samples (X1, Y1), . . . ,(Xm, Ym) from an unknown distribution P, but the learner initially only observes the *unlabeled* examples Xi, and can query to observe individual labels Yi (in a *sequential* fashion, so that it observes the label Yi of its previous query before selecting the next query Xi
′ ) (Cohn, Atlas, and Ladner, 1994; Freund, Seung, Shamir, and Tishby, 1997; Tong and Koller, 2001). Such queries can typically be answered by human experts, being of the same type as used for data annotation in standard supervised machine learning. In this scenario, the unlabeled examples Xi are typically assumed to be available in abundance, while obtaining the labels Yiis considered comparably more *expensive* (relying on the effort of a human expert), so that the primary objective is to minimize the number of *label queries* needed to achieve a given accuracy of a learned predictor hˆ. This is the setting studied in the present work.

The theoretical literature on this subject has origins in early works discussing algorithmic principles based on version spaces (Mitchell, 1979; Cohn, Atlas, and Ladner, 1994). Many of the early works providing actual bounds on the query complexity focused on showing improvements over passive learning for special scenarios, such as linear classifiers under distribution assumptions (e.g., Freund, Seung, Shamir, and Tishby, 1997; Dasgupta, Kalai, and Monteleoni, 2005; Har-Peled, Roth, and Zimak, 2007; Balcan, Beygelzimer, and Langford, 2006; Balcan, Broder, and Zhang, 2007; Balcan and Long, 2013; Gonen, Sabato, and Shalev-Shwartz, 2013; Wang and Singh, 2016; Cavallanti, Cesa-Bianchi, and Gentile, 2011; Dekel, Gentile, and Sridharan, 2012). This was followed by a boom of general-case analyses, providing general theories analyzing the query complexity for any concept class (e.g., Dasgupta, 2005; Hanneke, 2007a,b, 2009b,a, 2011, 2012, 2014; Dasgupta, Hsu, and Monteleoni, 2007; Balcan, Hanneke, and Vaughan, 2010; Beygelzimer, Dasgupta, and Langford, 2009; Koltchinskii, 2010; Zhang and Chaudhuri, 2014; El-Yaniv and Wiener, 2012; Wiener, Hanneke, and El-Yaniv, 2015; Hanneke and Yang, 2015; Hanneke, Karbasi, Moran, and Velegkas, 2024a), some of which are discussed in more detail below.

Agnostic PAC Learning: The PAC learning framework has also been extended to allow nonrealizable distributions P, that is, removing the restriction that infh∈C erP (h) = 0. This framework was abstractly formulated in the classic work of Vapnik and Chervonenkis (1974), with interest in the computer science literature initiated by the works of Haussler (1992); Kearns, Schapire, and Sellie
(1994). Since such non-realizable distributions P might not allow for predictors hˆ with erP (hˆ) ≤ ε, the objective in this framework changes to merely achieving a *relatively* low error rate compared to the best error rate achievable by concepts in the class C. More precisely, we aim to produce a predictor hˆ which, with probability at least 1 − δ, satisfies erP (hˆ) ≤ infh∈C erP (h) + ε. The goal is to achieve this objective, for *every* distribution P, without any restrictions. This framework is termed *agnostic PAC learning*, to emphasize that we do not assume any special knowledge of P
when designing such a learning algorithm (Kearns, Schapire, and Sellie, 1994). While an agnostic learning algorithm should achieve this objective for *every* distribution P, this need not restrict the *analysis* of such learners to consider only the worst case over all P. In particular, in the present work, we are primarily interested in analyzing the number of queries necessary and sufficient for agnostic active learning, as a function of the *best-in-class* error rate infh∈C erP (h), known as a first-order query complexity bound. Precisely, as introduced in Section 2, for every *ε, δ, β* ∈ (0, 1),
we denote by QCa(ε, δ; β, C) the minimax optimal first-order query complexity: that is, the minimal Q ∈ N for which there exists an active learning algorithm Aa such that (for a sufficiently large number m of unlabeled examples), for every distribution P with infh∈C erP (h) ≤ β, with probability at least 1 − δ, Aa makes at most Q queries and returns a predictor hˆ satisfying erP (hˆ) ≤ infh∈C erP (h) + ε. While, in principle, this definition of QCa(*ε, δ*; β, C) admits learners which explicitly depend on knowledge of β, we will find that the optimal query complexity is achievable (up to constant factors and lower-order terms) simultaneously for all β by an active learner which does not require knowledge of β. Such a learner is said to be *adaptive* to β. In particular, such a learner is therefore an agnostic PAC learner, and the β restriction only enters in its analysis.

The Passive Learning Baseline: Since the predictor hˆ produced by an active learning algorithm is based on its queried subset of a given set of i.i.d. examples (Xi, Yi), the natural quantity for comparison is the number of *i.i.d. labeled* examples necessary to obtain the same accuracy: i.e.,
the sample complexity of standard supervised learning, which in this literature is termed *passive* learning.

5 Recall from Section 2 that we denote by Mp(*ε, δ*; β, C) the minimax optimal sample complexity of passive learning: i.e., the minimal n such that there exists a passive learning algorithm Ap that, for every P with infh∈C erP (h) ≤ β, for S ∼ P
n and hˆn = Ap(S), guarantees with probability at least 1 − δ that erP (hˆn) ≤ infh∈C erP (h) + ε. Since we can always design an active learner that simply queries the first n examples and runs a passive learner Ap, we clearly always have QCa(ε, δ; β, C) ≤ Mp(*ε, δ*; β, C). Thus, the main question of interest is whether QCa(*ε, δ*; β, C)
is *strictly smaller* than Mp(*ε, δ*; β, C), and if so, by how much. Lower bounds of Vapnik and Chervonenkis (1974); Devroye and Lugosi (1995) establish that

Mp(ε, δ; β, C) = Ωβ ε 2 d + log1δ  + 1 ε d + log1δ  , (4)
recalling that d denotes the *VC dimension* of C (Vapnik and Chervonenkis, 1971; see Definition 4 of Appendix B). The classic analysis of Vapnik and Chervonenkis (1974) further established this lower bound can nearly be achieved by the simple method of empirical risk minimization, i.e., hˆn = argminh∈C ˆerS(h), providing an upper bound Mp(*ε, δ*; β, C) =
O
β ε2d log1 ε
+ log1δ
 +
1 ε d log1 ε
+ log1δ
. This has since been refined in various ways, such as via localized chaining arguments (e.g., Giné and Koltchinskii, 2006). Most recently, Hanneke, Larsen, and Zhivotovskiy (2024b) proved an upper bound Mp(*ε, δ*; β, C) =
O
β ε 2d + log1δ
+ O˜1ε d + log1δ
, matching the lower bound (4) up to log factors in the lower-order term (the problem of removing these remaining log factors remains open at this time).

The algorithm achieving this is *improper*, meaning its returned hˆn is not necessarily an element of C, and Hanneke, Larsen, and Zhivotovskiy (2024b) in fact show that for some concept classes C
 improperness is *necessary* to match the lower bound (4) in the lead term, as all proper learners incur an extra log1β factor. In the special case of β = 0 (the *realizable case*), the lower bound
(4) was shown to be achievable by Hanneke (2016a) (also necessarily via an improper learner), so that Mp(*ε, δ*; 0, C) = Θ1ε d + log1δ
. The lower bound (4) will therefore serve as a suitable baseline for gauging whether the query complexity QCa(*ε, δ*; β, C) of active learning is smaller than the sample complexity Mp(ε, δ; β, C) of passive learning.

The Need for Distribution-dependent Analysis in Realizable Active Learning: Much of the early work on active learning focused on the *realizable case*, i.e., the special case β = 0. In this special case, it was quickly observed by Dasgupta (2004, 2005) that there are some concept classes (e.g., *thresholds* 1[a,∞) on R) where active learning offers strong improvements over passive learning, and other concept classes (e.g., intervals 1[a,b] on R) where the (distribution-free) minimax query complexity QCa(ε, δ; 0, C) offers *no significant improvements* over passive learning. The essential advantage in the former case arises from a kind of "binary search" behavior, where the
"uncertainty" is being sequentially reduced by a careful choice of queries. In contrast, the essential challenge in the latter case is the problem of "searching in the dark" for a small-but-important region: e.g., the optimal concept is 1 for a single unknown xi among some x1*, . . . , x*1/ε, and PX = Uniform({x1*, . . . , x*1/ε}). It turns out this *hard* scenario is embedded in many concept classes of interest, a fact which was formalized and quantified by Hanneke and Yang (2015) in the *star number* complexity measure (Definition 2) discussed below. Such concept classes C naturally exhibit a lower bound QCa
(*ε, δ*; 0, C) = Ω1ε
. Even worse, consider a scenarios where the optimal concept can be 1 for any d points xi among x1*, . . . , x*d/(2ε), and PX = Uniform({x1*, . . . , x*d/(2ε)}). Hanneke and Yang (2015) show this scenario has QCa(ε, δ; 0, C) = Ωdε
, so that QCa(ε, δ; 0, C) has the same joint dependence on (d, ε) as passive learning Mp(ε, δ; 0, C) = Θ1ε d + log1 δ
, only offering 5Since the active learner also has access to the remaining (unqueried) i.i.d. *unlabeled examples* Xi, it is also natural to compare to the related framework of *semi-supervised* learning, in which a learner has access to some number n of i.i.d. labeled examples with distribution P and additionally some larger number m of i.i.d. *unlabeled* examples with distribution PX (Chapelle, Scholkopf, and Zien, 2006). While, under some favorable conditions, the labeled sample complexity n of semi-supervised learning can be smaller than that of strictly-supervised passive learning (see Balcan and Blum, 2010), the lower bounds on the (distribution-free) sample complexity of passive learning discussed in this work remain valid for the labeled sample complexity of semi-supervised learning (regardless of how many unlabeled examples are available), so that for the purpose of comparison in the present work, the distinction between supervised and semi-supervised passive learning as a baseline is not important, and we will simply compare to passive supervised learning for simplicity.