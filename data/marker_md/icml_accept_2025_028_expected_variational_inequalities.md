# Expected Variational Inequalities

Brian Hu Zhang \* 1 Ioannis Anagnostides \* 1 Emanuel Tewolde 1 2 Ratip Emin Berker 1 2 Gabriele Farina <sup>3</sup> Vincent Conitzer 1 2 4 Tuomas Sandholm 1 5 6 7

#### Abstract

Variational inequalities (VIs) encompass many fundamental problems in diverse areas ranging from engineering to economics and machine learning. However, their considerable expressivity comes at the cost of computational intractability. In this paper, we introduce and analyze a natural relaxation—which we refer to as *expected variational inequalities (EVIs)*—where the goal is to find a *distribution* that satisfies the VI constraint *in expectation*. By adapting recent techniques from game theory, we show that, unlike VIs, EVIs can be solved in polynomial time under general (nonmonotone) operators. EVIs capture the seminal notion of *correlated equilibria*, but enjoy a greater reach beyond games. We also employ our framework to capture and generalize several existing disparate results, including from settings such as smooth games, and games with coupled constraints or nonconcave utilities.

## 1. Introduction

*Variational inequalities (VIs)* provide a unifying framework for analyzing a wide range of optimization and equilibrium problems. They have a host of important applications in engineering and economics [\(Facchinei & Pang,](#page-10-0) [2003\)](#page-10-0), including identifying stationary points in constrained optimization; computing Nash equilibria in multi-player games [\(Nash,](#page-11-0) [1951\)](#page-11-0), such as Cournot's classical model of oligopoly [\(Cournot,](#page-9-0) [1838\)](#page-9-0); predicting economic activity commodity prices and consumer consumption—in a closed, competitive economy [\(Arrow & Debreu,](#page-8-0) [1954\)](#page-8-0), which is at the heart of general equilibrium theory; traffic equilibrium

problems—estimating the steady-state of a congested network wherein users compete for its resources [\(Dafermos,](#page-9-1) [1980\)](#page-9-1); frictional contact problems in mechanical engineering [\(Capatina,](#page-9-2) [2014\)](#page-9-2); and pricing options, a foundational problem in financial economics [\(Black & Scholes,](#page-9-3) [1973\)](#page-9-3).

Formally, in a general form, a VI can be defined as follows.[<sup>1</sup>](#page-0-0)

Definition 1.1. Let X be a convex and compact subset of R d and <sup>F</sup> : X → <sup>R</sup> d a bounded map. The *variational inequality (VI)* problem asks for a point <sup>x</sup> ∈ X such that

$$\langle F(\mathbf{x}), \mathbf{x}' - \mathbf{x} \rangle \geq 0 \quad \forall \mathbf{x}' \in \mathcal{X}. \quad (1)$$

For computational purposes, it is common to consider the ϵ*-approximate* VI problem, wherein the right-hand side of [\(1\)](#page-0-1) is replaced by −<sup>ϵ</sup> for some precision parameter ϵ > <sup>0</sup>.

[Definition 1.1](#page-0-2) abstracts the description of <sup>F</sup> and X . As a concrete example, when <sup>F</sup> : <sup>x</sup> 7→ −∇u(x) is the negative gradient of a differentiable function <sup>u</sup> : X → <sup>R</sup>, the solutions to [\(1\)](#page-0-1) are points that satisfy the first-order optimality conditions for maximizing u [\(Boyd & Vandenberghe,](#page-9-4) [2004\)](#page-9-4).

Unfortunately, the considerable expressivity of VIs comes at the expense of *intractability*: even when F is linear and ϵ is an absolute constant, identifying an ϵ-approximate VI solution is computationally hard; this follows readily from the intractability of Nash equilibria—under plausible complexity assumptions [\(Daskalakis et al.,](#page-9-5) [2008;](#page-9-5) [Chen et al.,](#page-9-6) [2009;](#page-9-6) [Rubinstein,](#page-11-1) [2016\)](#page-11-1). Unconditional, query-complexity lower bounds have also been established [\(Hirsch et al.,](#page-10-1) [1989;](#page-10-1) [Babichenko,](#page-8-1) [2016\)](#page-8-1); *cf.* [Milionis et al.](#page-11-2) [\(2023\)](#page-11-2) and [Hart &](#page-10-2) [Mas-Colell](#page-10-2) [\(2003\)](#page-10-2) for other pertinent impossibility results.

This bleak realization has shifted the focus of contemporary research primarily to characterizing specific subclasses of VIs that elude those complexity barriers, with the ensuing line of work flourishing in recent years. Some notable examples include the classical *Minty property* [\(Facchinei & Pang,](#page-10-0) [2003;](#page-10-0) [Mertikopoulos & Zhou,](#page-11-3) [2019\)](#page-11-3), as well as certain relaxations thereof [\(Diakonikolas et al.,](#page-9-7) [2021;](#page-9-7) [Bohm¨](#page-9-8) , [2023;](#page-9-8) [Bauschke et al.,](#page-8-2) [2021;](#page-8-2) [Combettes & Pennanen,](#page-9-9) [2004;](#page-9-9) [Gor](#page-10-3)[bunov et al.,](#page-10-3) [2023;](#page-10-3) [Cai et al.,](#page-9-10) [2024b;](#page-9-10) [Alacaoglu et al.,](#page-8-3) [2023;](#page-8-3)

<sup>\*</sup>Equal contribution <sup>1</sup>Carnegie Mellon University <sup>2</sup> Foundations of Cooperative AI Lab (FOCAL) <sup>3</sup>Massachusetts Institute of Technology <sup>4</sup>University of Oxford <sup>5</sup> Strategy Robot, Inc. <sup>6</sup> Strategic Machine, Inc. <sup>7</sup>Optimized Markets, Inc. Correspondence to: Brian Hu Zhang <bhzhang@cs.cmu.edu>, Ioannis Anagnostides <ianagnos@cs.cmu.edu>.

*Proceedings of the* 42 nd *International Conference on Machine Learning*, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

<sup>1</sup>As is standard, a VI throughout this paper refers to the *Stampacchia* VI *(SVI)* [\(Kinderlehrer & Stampacchia,](#page-11-4) [2000\)](#page-11-4).

[Pethick et al.,](#page-11-5) [2022;](#page-11-5) [Lee & Kim,](#page-11-6) [2021;](#page-11-6) [Patris & Panageas,](#page-11-7) [2024;](#page-11-7) [Choudhury et al.,](#page-9-11) [2024;](#page-9-11) [Anagnostides et al.,](#page-8-4) [2024\)](#page-8-4).

Despite these important advances, the scope of such results is severely restricted. In this paper, we pursue a different, orthogonal avenue. Instead of restricting the class of *problems* to achieve computational tractability, we relax the underlying *solution concept*. Our main research question is:

*Are there meaningful relaxations of the VI problem that can always be solved efficiently?*

When specialized to games, this question can be seen as part of the research agenda recently outlined by [Daskalakis](#page-9-12) [\(2022\)](#page-9-12) in his address at the Nobel symposium about equilibrium computation in *nonconcave games*—a major, new frontier in the interface of game theory and optimization.

#### 1.1. Our Contribution: The Expected VI Problem

To make progress on that central question, we introduce a natural relaxation of VIs (in the context of [Definition 1.1\)](#page-0-2).

Definition 1.2. Given a set of *deviations* <sup>Φ</sup> ⊆ X <sup>X</sup> , the ϵ*-approximate* Φ*-expected variational inequality (*Φ*-EVI)* problem asks for a distribution <sup>µ</sup> ∈ ∆(X ) such that

$$\mathbb{E}_{\mathbf{x} \sim \mu} \langle F(\mathbf{x}), \phi(\mathbf{x}) - \mathbf{x} \rangle \geq -\epsilon \quad \forall \phi \in \Phi.$$

(The above definition does not specify how X , F, <sup>Φ</sup>, and <sup>µ</sup> should be represented for computational purposes, but we will be explicit about representation whenever it is relevant.)

In words, [Definition 1.2](#page-1-0) only imposes (approximate) nonnegativity *in expectation* for points <sup>x</sup> drawn from <sup>µ</sup> ∈ ∆(X ). It certainly relaxes [Definition 1.1:](#page-0-2) if <sup>x</sup> satisfies [\(1\)](#page-0-1), then the distribution µ that always outputs x is also a Φ-EVI solution. Φ-EVIs are thus no harder than VIs (assuming that solutions exist). However, as we shall see, the primary justification of Φ-EVIs is that they can be easier than VIs.

[Definition 1.2](#page-1-0) is crucially parameterized by Φ; the larger the set of deviations Φ, the tighter the set of solutions. As will become clear, [Definition 1.2](#page-1-0) is intimately connected with notions of *correlated equilibrium (CE)* from game theory (*e.g.*, [Aumann,](#page-8-5) [1974\)](#page-8-5). The more permissive case where Φ comprises only constant functions, Φ = ΦCON = {<sup>ϕ</sup><sup>x</sup> : <sup>x</sup> ∈ X } where <sup>ϕ</sup>x(<sup>x</sup> ′ ) = x for all x ′ ∈ X , is perhaps the most basic relaxation of [Definition 1.1;](#page-0-2) we call the ΦCON-EVI problem simply the *EVI problem*.

Algorithms and complexity for Φ-EVIs As it turns out, imposing no constraints on Φ results in an impasse: Φ-EVIs are in general tantamount to regular VIs—thereby being PPAD-hard (Corollaries [3.7](#page-3-0) and [3.9\)](#page-4-0). On the other hand, unlike general VIs, one of our key contributions is to show

that when Φ contains only linear maps, ΦLIN, Φ-EVIs can be solved in time polynomial in the dimension d and log(1/ϵ) [\(Theorem 4.1\)](#page-4-1), establishing the promised computational property that separates EVIs from VIs. This result is based on *ellipsoid against hope (*EAH*)*, the seminal algorithm of [Pa](#page-11-8)[padimitriou & Roughgarden](#page-11-8) [\(2008\)](#page-11-8) developed for computing correlated equilibria in multi-player games. [\(Section 2.1](#page-2-0) gives a self-contained overview of EAH.) In doing so, we extend the scope of that algorithm to a much broader class of problems well beyond the realm of game theory. Notably, [Theorem 4.1](#page-4-1) applies even when X is given implicitly through a membership oracle; this extension makes use of the recent technical approach of [Daskalakis et al.](#page-9-13) [\(2025\)](#page-9-13), discussed in more detail in [Section 4.](#page-4-2)

One limitation of [Theorem 4.1](#page-4-1) is that it relies on the EAH algorithm, which is slow in practice. We address this by also establishing more scalable algorithms that use convex quadratic optimization [\(Theorem 4.3\)](#page-5-0) instead of the ellipsoid algorithm. As a byproduct, we obtain the best-known algorithm for linear-swap regret minimization over explicitly represented polytopes, improving on [Daskalakis et al.](#page-9-13) [\(2025\)](#page-9-13) by reducing the per-iteration complexity.

In addition to their more favorable computational properties, we further show that Φ-EVIs admit (approximate) solutions under more general conditions than their associated VIs namely, without F being continuous [\(Theorem 3.1\)](#page-3-1); [Sec](#page-3-2)[tion 3](#page-3-2) documents further interesting aspects on existence.

Connection to other solution concepts As we have alluded to, Φ-EVIs generalize (Examples [5.1](#page-5-1) and [5.2\)](#page-5-2) the seminal concept of a *(coarse) correlated equilibrium a la `* [Au](#page-8-5)[mann](#page-8-5) [\(1974\)](#page-8-5) and [Moulin & Vial](#page-11-9) [\(1978\)](#page-11-9) in finite games, and more generally Φ*-equilibria* [\(Greenwald & Jafari,](#page-10-4) [2003;](#page-10-4) [Stoltz & Lugosi,](#page-11-10) [2007;](#page-11-10) [Gordon et al.,](#page-10-5) [2008\)](#page-10-5) of concave games. What is more surprising is that ΦLIN-EVIs *refine* CEs even in normal-form games; we give illustrative examples, together with an interpretation, in [Section 5.](#page-5-3) We also note that Φ-EVIs can be used even in games with nonconcave utilities [\(Daskalakis,](#page-9-12) [2022;](#page-9-12) [Cai et al.,](#page-9-14) [2024a\)](#page-9-14) or noncontinuous gradients (as in nonsmooth optimization), as well as in (pseudo-)games with *coupled constraints* (*cf.* [Bernasconi](#page-8-6) [et al.,](#page-8-6) [2023](#page-8-6) and [Appendix A](#page-13-0) for related work).

Further properties As further motivation, we show that for certain structured problems, such as *(quasar-)concave* optimization and *polymatrix* zero-sum games, EVIs essentially coincide with VIs (Propositions [6.1](#page-7-0) and [6.3\)](#page-7-1).

Finally, in certain applications, one might be interested in a VI solution mainly insofar as it provides guarantees in terms of an underlying objective, such as misclassification error or social welfare. Through that prism, the question is whether performance guarantees for VIs can be translated

| Result Description Reference                                                           |             |
|----------------------------------------------------------------------------------------|-------------|
| Existence of ( ϵ -approx.) solutions Under Lipschitz cont. for Φ and bounded F Theorem | 3.1         |
| Complexity with nonlinear Φ PPAD -hardness with linear F and ϵ = Θ(1) Corollaries      | 3.7 and 3.9 |
| Algorithms for linear Φ poly( d, log(1 /ϵ )) -time via EAH                             |             |
| poly( d, 1 /ϵ ) -time via Φ -regret minimization                                       |             |
| Theorem                                                                                | 4.1         |
| Theorem                                                                                | 4.3         |
| Equivalence between VIs-EVIs Quasar-concave functions (Definition 6.2)                 |             |
| x 7→ ⟨ F ( x ) , x                                                                     |             |
| ′ − x ⟩ concave for all x                                                              |             |
| Proposition                                                                            | 6.3         |
| Proposition                                                                            | 6.1         |
| Performance guarantees for EVIs Under smoothness (Definition 7.1) Theorem              | 7.4         |

Table 1. Our main results concerning Φ-EVIs [\(Definition 1.2\)](#page-1-0).

to EVIs as well. In [Section 7,](#page-7-4) we establish a framework for accomplishing that [\(Definition 7.1\)](#page-7-3) by extending the celebrated *smoothness* framework of [Roughgarden](#page-11-11) [\(2015\)](#page-11-11), and provide interesting examples beyond game theory.

Taken together, these properties provide compelling justification for Φ-EVIs as a solution concept in place of VIs. [Table 1](#page-2-1) gathers our main results. (Proofs are in [Appendix C.](#page-14-0))

## 2. Preliminaries

This section provides some basic notation and background together with an overview of the EAH algorithm. Additional preliminaries, which are not necessary for the main body, are given later in [Appendix B.](#page-13-1)

Notation We use boldface, lowercase letters, such as x and y, to denote vectors in a Euclidean space. Capital, boldface letters, such as A, represent matrices. For x, x ′ ∈ R d , we use ⟨x, <sup>x</sup> ′ p ⟩ to denote their inner product. ∥x∥ := ⟨x, <sup>x</sup>⟩ is the Euclidean norm of <sup>x</sup>. Br(x) is the (closed) Euclidean ball centered at <sup>x</sup> with radius r > <sup>0</sup>. conv(·) represents the convex hull. An *endomorphism* on X is a function mapping X to X .

Returning to [Definition 1.2,](#page-1-0) for computational purposes, we assume throughout that F has an explicit polynomial representation, so that <sup>F</sup>(x) ∈ <sup>R</sup> d can be evaluated in poly(d) time. Further, there exists B > <sup>0</sup> such that ∥F(x)∥ ≤ <sup>B</sup> for all <sup>x</sup> ∈ X . We will also restrict the support supp(µ) of µ to be poly(d, 1/ϵ), unless stated otherwise. With regard to X , we assume that we have *oracle access*. In particular, we consider the following three types of oracle access.

- *Membership*: given <sup>x</sup> ∈ <sup>R</sup> d , decide whether <sup>x</sup> ∈ X .
- *Separation*: given <sup>x</sup> ∈ <sup>R</sup> d , decide whether <sup>x</sup> ∈ X ; if not, return a hyperplane that *separates* <sup>x</sup> from X .
- *Linear optimization*: Given <sup>u</sup> ∈ <sup>R</sup> d , return a vector in argmaxx∈X ⟨x,u⟩.

In addition, we will assume that X ⊆ BR(0) for some <sup>R</sup> ≤ poly(d), and X contains a ball of radius <sup>1</sup> in its relative interior; this is a standard regularity condition that ensures X is geometrically well-behaved, which can be met by bringing X into isotropic position [\(Appendix B\)](#page-13-1). Under this assumption, the three oracles listed above are polynomially equivalent [\(Grotschel et al.](#page-10-6) ¨ , [1993;](#page-10-6) [1981\)](#page-10-7). As a result, we may assume that X is given implicitly via a (poly(d)-time) membership oracle, which suffices for [Theorem 4.1.](#page-4-1)

All our positive results with respect to the set of linear endomorphisms ΦLIN readily carry over to affine endomorphisms as well.

#### 2.1. Ellipsoid Against Hope

This *ellipsoid against hope (*EAH*)* algorithm was famously introduced by [Papadimitriou & Roughgarden](#page-11-8) [\(2008\)](#page-11-8) to compute correlated equilibria in multi-player games. We proceed with an overview of EAH, and in particular a generalized version thereof, crystallized by [Farina & Pipis](#page-10-8) [\(2024\)](#page-10-8).

Consider an arbitrary optimization problem of the form

$$\text{find } \mu \in \Delta(\mathcal{X}) \quad \text{s.t.} \quad \mathbb{E}_{\mathbf{x} \sim \mu} \langle \mathbf{y}, G(\mathbf{x}) \rangle \geq 0 \quad \forall \mathbf{y} \in \mathcal{Y}, \quad (2)$$

where Y ⊆ <sup>R</sup> <sup>m</sup>, and <sup>G</sup> : X → <sup>R</sup> <sup>m</sup> is an arbitrary function. Suppose that we are given an evaluation oracle for G and a separation oracle for Y. Assume further that we are given a *good-enough-response (*GER*)* oracle, which, given any <sup>y</sup> ∈ Y, returns <sup>x</sup> ∈ X such that ⟨y, G(x)⟩ ≥ <sup>0</sup>. The upshot is that EAH enables us to solve [\(2\)](#page-2-2) with just the above tools. Indeed, consider the following problem, which is an ϵ-approximate version of the dual of [\(2\)](#page-2-2).

find 
$$y \in \mathcal{Y}$$
 s.t.  $\langle y, G(x) \rangle \leq -\epsilon \quad \forall x \in \mathcal{X}.$  (3)

Since a GER oracle exists, [\(3\)](#page-2-3) is infeasible. What is more, a certificate of infeasibility of [\(3\)](#page-2-3) yields an ϵ-approximate solution to [\(2\)](#page-2-2). It thus suffices to run the ellipsoid algorithm on [\(3\)](#page-2-3) and extract a certificate of infeasibility; in a nutshell, this is what EAH does (*cf.* [Algorithm 1](#page-19-0) in [Appendix C\)](#page-14-0).

Theorem 2.1 (Generalized form of EAH; [Farina & Pipis,](#page-10-8) [2024\)](#page-10-8). *Given a* GER *oracle and a separation oracle (*SEP*)* *for* Y*,* EAH *runs in time* poly(d, m, log(1/ϵ)) *and returns an* ϵ*-approximate solution to* [\(2\)](#page-2-2)*.*

One of our main results [\(Theorem 4.1\)](#page-4-1) crucially hinges on a strengthening of [Theorem 2.1](#page-2-4) due to [Daskalakis et al.](#page-9-13) [\(2025\)](#page-9-13), discussed further in [Section 4](#page-4-2) and [Appendix C.3.](#page-18-0)

#### 3. Existence and Complexity Barriers

Perhaps the most basic question about Φ-EVIs concerns their *totality*—the existence of solutions. If one is willing to tolerate an arbitrarily small imprecision ϵ > 0, we show that solutions exist under very broad conditions.

Theorem 3.1. *Suppose that* <sup>F</sup> : X → <sup>R</sup> d *is measurable and there exists* L > <sup>0</sup> *such that every* <sup>ϕ</sup> ∈ <sup>Φ</sup> *is* <sup>L</sup>*-Lipschitz continuous. Then, for any* ϵ > 0*, there exists an* ϵ*-approximate solution to the* Φ*-EVI problem.*

In particular, our existence proof does not rest on F being continuous. Instead, we consider the continuous function <sup>F</sup><sup>b</sup> that maps <sup>x</sup> 7→ <sup>E</sup>xb∼∆(Bδ(x)∩X ) <sup>F</sup>(xb) [\(Claim C.1\)](#page-14-1), where Bδ(x) is the Euclidean ball centered at <sup>x</sup> with radius δ = δ(ϵ). It then suffices to invoke Brouwer's fixed-point theorem for the gradient mapping <sup>x</sup> 7→ <sup>Π</sup><sup>X</sup> (<sup>x</sup> − <sup>F</sup>b(x)), where <sup>Π</sup><sup>X</sup> is the Euclidean projection with respect to X .

[Theorem 3.1](#page-3-1) implies that a Φ-EVI can have approximate solutions even when the associated VI problem does not.[<sup>2</sup>](#page-3-3)

Corollary 3.2. *There exists a VI problem that does not admit approximate solutions when* ϵ = Θ(1)*, but the corresponding* ϵ*-approximate* Φ*-EVI is total for any* ϵ > 0*.*

In the proof, we set F to be the *sign function* [\(Example C.2\)](#page-15-0). By contrast, if one insists on exact solutions, EVIs do not necessarily admit solutions.

Proposition 3.3. *When* F *is not continuous, there exists an EVI problem with no solutions.*

Furthermore, [Theorem 3.1](#page-3-1) raises the question of whether it is enough to instead assume that every <sup>ϕ</sup> ∈ <sup>Φ</sup> is continuous. Our next result dispels any such hopes.

Theorem 3.4. *There are* Φ*-EVI instances that do not admit* ϵ*-approximate solutions even when* ϵ = Θ(1)*,* F *is piecewise constant, and* Φ *contains only continuous functions.*

Our final result on existence complements Theorems [3.1](#page-3-1) and [3.4](#page-3-4) by showing that when Φ is finite-dimensional, it is enough if every <sup>ϕ</sup> ∈ <sup>Φ</sup> admits a fixed point (this holds, for example, when ϕ is continuous—by Brouwer's theorem).

#### Theorem 3.5. *Suppose that*

- *1.* <sup>Φ</sup> *is finite-dimensional, that is, there exists* <sup>k</sup> ∈ <sup>N</sup> *and a kernel map* <sup>m</sup> : X → <sup>R</sup> k *such that every* <sup>ϕ</sup> ∈ <sup>Φ</sup> *can be expressed as* <sup>K</sup>m(x) *for some* <sup>K</sup> ∈ <sup>R</sup> d×k *; and*
- *2. every* <sup>ϕ</sup> ∈ <sup>Φ</sup> *admits a fixed point, that is, a point* X ∋ <sup>x</sup> = FP(ϕ) *such that* <sup>ϕ</sup>(x) = <sup>x</sup>*.*

*Then, the* Φ*-EVI problem admits an* ϵ*-approximate solution with support size at most* 1 + dk *for every* ϵ > 0*.*

Notably, this theorem guarantees the existence of solutions with finite support; the proof makes use of the minimax theorem (*e.g.*, [Sion,](#page-11-12) [1958\)](#page-11-12) in conjunction with Caratheodory's ´ theorem on convex hulls [\(Caratheodory](#page-9-16) ´ , [1911\)](#page-9-16).

Complexity Having established some basic existence properties, we now turn to the complexity of Φ-EVIs. Let us define the VI gap function VIGap(x) := − minx′∈X ⟨F(x), <sup>x</sup> ′ − <sup>x</sup>⟩, which is nonnegative. If we place no restrictions on Φ, it turns out that Φ-EVIs are tantamount to regular VIs:

Proposition 3.6. *If* Φ *contains all measurable functions from* X *to* X *, then any solution* <sup>µ</sup> ∈ ∆(X ) *to the* <sup>ϵ</sup>*approximate* Φ*-EVI problem satisfies*

$$\mathbb{E}_{\mathbf{x} \sim \mu} \text{VIGap}(\mathbf{x}) \leq \epsilon. \quad (4)$$

In proof, it suffices to consider a <sup>ϕ</sup> that maps <sup>x</sup> ∈ X to an appropriate point in argminx′∈X ⟨F(x), <sup>x</sup> ′ − <sup>x</sup>⟩. When µ must be given explicitly, [Proposition 3.6](#page-3-5) immediately implies that Φ-EVIs are computationally hard, because [\(4\)](#page-3-6) implies that VIGap(x) ≤ <sup>ϵ</sup> for some <sup>x</sup> in the support of <sup>µ</sup>, and such a point can be identified in polynomial time.[<sup>3</sup>](#page-3-7)

Corollary 3.7. *The* ϵ*-approximate* Φ*-EVI problem is* PPAD*hard even when* ϵ *is an absolute constant and* F *is linear.*

Coupled with [Proposition 3.6,](#page-3-5) this follows from the hardness result of [Rubinstein](#page-11-13) [\(2015\)](#page-11-13) concerning Nash equilibria in (multi-player) polymatrix games (for binary-action, graphical games, [Deligkas et al.,](#page-9-17) [2023](#page-9-17) recently showed that PPAD-hardness persists up to ϵ < <sup>1</sup>/2). [Corollary 3.7](#page-3-0) notwithstanding, it is easy to see that the set of solutions to <sup>Φ</sup>-EVIs is convex for any <sup>Φ</sup> ⊆ X <sup>X</sup> .

*Remark* 3.8*.* Let X <sup>=</sup> X1×· · ·×Xn, as in an <sup>n</sup>-player game. Whether [Corollary 3.7](#page-3-0) applies under deviations that can be decomposed as <sup>ϕ</sup> : <sup>x</sup> 7→ <sup>ϕ</sup>(x) = (ϕ1(x1), . . . , ϕn(xn)) is a major open question in the regime where <sup>ϵ</sup> ≪ <sup>1</sup> (*cf.* [Dagan](#page-9-18) [et al.,](#page-9-18) [2024;](#page-9-18) [Peng & Rubinstein,](#page-11-14) [2024\)](#page-11-14).

<sup>2</sup>Noncontinuity of F manifests itself prominently in *nonsmooth* optimization (*e.g.*, [Zhang et al.,](#page-12-0) [2020;](#page-12-0) [Davis et al.,](#page-9-15) [2022;](#page-9-15) [Tian et al.,](#page-12-1) [2022;](#page-12-1) [Jordan et al.,](#page-10-9) [2023a\)](#page-10-9); recent research there focuses on *Goldstein* stationary points [\(Goldstein,](#page-10-10) [1977\)](#page-10-10), which are conceptually related to EVIs.

<sup>3</sup>This argument carries over without restricting the support of µ, by assuming instead access to a sampling oracle from µ: a standard Chernoff bound implies that the empirical distribution (w.r.t. a large enough sample size) approximately satisfies [\(4\)](#page-3-6).

Viewed differently, a special case of the Φ-EVI problem arises when Φ = {ϕ} and <sup>F</sup>(x) = <sup>x</sup> − <sup>ϕ</sup>(x), for some fixed map <sup>ϕ</sup> : X → X . In this case, the <sup>Φ</sup>-EVI problem reduces to finding a <sup>µ</sup> ∈ ∆(X ) such that

$$\mathbb{E}_{x \sim \mu} \langle F(x), \phi(x) - x \rangle = - \mathbb{E}_{x \sim \mu} \|\phi(x) - x\|^2 \geq -\epsilon. \quad (5)$$

As a result, µ must contain in its support an ϵ-approximate fixed point of ϕ, a problem which is PPAD-hard already for quadratic functions [\(Zhang et al.,](#page-12-2) [2024a\)](#page-12-2).

Corollary 3.9. *The* ϵ*-approximate* Φ*-EVI problem is* PPAD*hard even when* ϵ *is an absolute constant,* F *is quadratic, and* Φ = {ϕ} *for a quadratic map* <sup>ϕ</sup> : X → X *.*

It is also worth noting that, unlike [Corollary 3.7,](#page-3-0) Φ in the corollary above contains only continuous functions.

It also follows from [\(5\)](#page-4-3) that for ϵ = 0, Φ-EVIs capture exact fixed points. The complexity class FIXP characterizes such problems [\(Etessami & Yannakakis,](#page-10-11) [2007\)](#page-10-11).

Corollary 3.10. *The* Φ*-EVI problem is* FIXP*-hard, assuming that* supp(µ) ≤ poly(d)*.*

Exponential lower bounds in terms of the number of function evaluations of F also follow from [Hirsch et al.](#page-10-1) [\(1989\)](#page-10-1).

On a positive note, the next section establishes polynomialtime algorithms when Φ contains only *linear* endomorphisms.[<sup>4</sup>](#page-4-4)

#### 4. Efficient Computation with Linear Maps

The hardness results of the previous section highlight the need to restrict the set Φ in order to make meaningful progress. Our main result here establishes a polynomialtime algorithm when Φ contains only linear endomorphisms.

Theorem 4.1. *If* Φ *contains only linear endomorphisms, the* ϵ*-approximate* Φ*-EVI problem can be solved in time* poly(d, log(B/ϵ)) *given a membership oracle for* X *.*

The proof relies on the ellipsoid against hope (EAH), and in particular, a recent generalization by [Daskalakis et al.](#page-9-13) [\(2025\)](#page-9-13). In a nutshell, the main deficiency in the framework covered earlier in [Section 2.1](#page-2-0) is that one needs a separation oracle for Y [\(Theorem 2.1\)](#page-2-4), where Y for us is the set of deviations <sup>Φ</sup>. Unlike some applications, in which Y has an explicit, polynomial representation [\(Papadimitriou &](#page-11-8) [Roughgarden,](#page-11-8) [2008\)](#page-11-8), that assumption needs to be relaxed to account for ΦLIN [\(Daskalakis et al.,](#page-9-13) [2025,](#page-9-13) Theorem 3.4).

[Daskalakis et al.](#page-9-13) [\(2025\)](#page-9-13) address this by considering instead the SEPorGER oracle. As the name suggests, for any

<sup>y</sup> ∈ <sup>R</sup> <sup>m</sup>, it *either* returns a hyperplane separating <sup>y</sup> from Y, or a good-enough-response <sup>x</sup> ∈ X . They showed that [Theo](#page-2-4)[rem 2.1](#page-2-4) can be extended under this weaker oracle (in place of GER and SEP); the formal version is given in [Theorem C.4.](#page-18-1)

In our setting, we consider the feasibility problem

$$\text{find } \phi \in \Phi_{\text{LIN}} \text{ s.t. } \langle F(\mathbf{x}), \phi(\mathbf{x}) - \mathbf{x} \rangle \leq -\epsilon \quad \forall \mathbf{x} \in \mathcal{X}. \quad (6)$$

Equivalently,

$$\begin{aligned} \text{find } \mathbf{K} \in \mathbb{R}^{d \times d} \quad \text{s.t.} \\ \langle F(\mathbf{x}), \mathbf{K}\mathbf{x} - \mathbf{x} \rangle \leq -\epsilon \quad \forall \mathbf{x} \in \mathcal{X}, \\ \mathbf{K}\mathbf{x} \in \mathcal{X} \quad \forall \mathbf{x} \in \mathcal{X}. \end{aligned}$$

This program is infeasible since, for any <sup>ϕ</sup> ∈ <sup>Φ</sup>LIN, the fixed point x of ϕ makes the left-hand side of the constraint 0. And a certificate of infeasibility is an ϵ-approximate ΦLIN-EVI solution. Thus, it suffices to show how to run the ellipsoid algorithm on [\(6\)](#page-4-5). By [Theorem C.4,](#page-18-1) it suffices if for any <sup>K</sup> ∈ <sup>R</sup> d×d , we can compute efficiently *either*

- some <sup>x</sup> ∈ X such that <sup>K</sup><sup>x</sup> <sup>=</sup> <sup>x</sup> (GER), *or*
- some hyperplane separating K from ΦLIN (SEP).

This is precisely the *semi-separation oracle* solved by [Daskalakis et al.](#page-9-13) [\(2025,](#page-9-13) Lemma 4.1), stated below.

Lemma 4.2 [\(Daskalakis et al.,](#page-9-13) [2025\)](#page-9-13). *There is an algorithm that takes* <sup>K</sup> ∈ <sup>R</sup> d×d *, runs in* poly(d) *time, makes* poly(d) *oracle queries to* X *, and either returns a fixed point* X ∋ x = Kx*, or a hyperplane separating* K *from* ΦLIN*.*

On a separate note, [Theorem 4.1](#page-4-1) only accounts for approximate solutions. We cannot hope to improve that in the sense that exact solutions might be supported only on irrational points even in concave maximization (*cf.* [Proposition 6.3\)](#page-7-1).

#### 4.1. Regret Minimization for EVIs on Polytopes

One caveat of [Theorem 4.1](#page-4-1) is that it relies on the impractical EAH algorithm. To address this limitation, we will show that Φ-EVIs are also amenable to the more scalable approach of *regret minimization*—albeit with an inferior complexity growing as poly(1/ϵ).

Specifically, in our context, the regret minimization framework can be applied as follows. At any time <sup>t</sup> ∈ <sup>N</sup>, we think of a "learner" selecting a point x (t) ∈ X , whereupon F(x (t) ) is given as feedback from the "environment," so that the utility at time <sup>t</sup> reads −⟨<sup>x</sup> (t) , F(x (t) )⟩. <sup>Φ</sup>*-regret* is a measure of performance in online learning, defined as

$$\Phi\text{-Reg}^{(T)} := \max_{\phi \in \Phi} \sum_{t=1}^T \langle F(\mathbf{x}^{(t)}), \phi(\mathbf{x}^{(t)}) - \mathbf{x}^{(t)} \rangle.$$

The uniform distribution <sup>µ</sup> on {<sup>x</sup> (1) , . . . , x (T)} is clearly a Φ-Reg(T) /T-approximate Φ-EVI solution.

<sup>4</sup>We do not distinguish between affine and linear maps because we can always set X ← X × {1}, in which case affine and linear maps coincide.

In what follows, we will assume that X is a polytope given explicitly by linear constraints, *i.e.*,

$$\mathcal{X} = \{x \in \mathbb{R}^d : \mathbf{A}x \leq b\},$$

where <sup>A</sup> ∈ <sup>Q</sup>m×<sup>d</sup> and <sup>b</sup> ∈ <sup>Q</sup><sup>m</sup> are given as input.

To minimize Φ-regret, we will make use of the template by [Gordon et al.](#page-10-5) [\(2008\)](#page-10-5), which comprises two components. The first is a fixed-point oracle, which takes as input a function <sup>ϕ</sup> ∈ <sup>Φ</sup>LIN and returns a point <sup>x</sup> ∈ X with <sup>x</sup> <sup>=</sup> <sup>ϕ</sup>(x); given that ϕ is linear, it can be implemented efficiently via linear programming. The second component is an algorithm for minimizing (external) regret over the set ΦLIN. In [Theo](#page-19-1)[rem D.1,](#page-19-1) we devise a polynomial representation for ΦLIN:

Theorem 4.3. *For an arbitrary polytope* X *given by explicit linear constraints, there is an explicit representation of* ΦLIN *as a polytope with* O(d <sup>2</sup> + m<sup>2</sup> ) *variables and constraints.*

As a consequence, we can instantiate the regret minimizer operating over ΦLIN with projected gradient descent.

Corollary 4.4. *There is a deterministic algorithm that guarantees* <sup>Φ</sup>LIN*-*Reg(T) ≤ <sup>ϵ</sup> *after* poly(d, m)/ϵ<sup>2</sup> *rounds, and requires solving a convex quadratic program with* O(d <sup>2</sup> + m<sup>2</sup> ) *variables and constraints in each iteration.*

An additional benefit of [Corollary 4.4](#page-5-4) compared to using EAH is that the former is more suitable in a decentralized environment—for example, in multi-player games (*cf.* [Ex](#page-5-1)[ample 5.1\)](#page-5-1). There, [Corollary 4.4](#page-5-4) corresponds to each player running their own independent no-regret learning algorithm. Even in this setting, our algorithms actually yield an improvement over the best-known algorithms for minimizing ΦLIN-regret over explicitly-represented polytopes: the previous state of the art, due to [Daskalakis et al.](#page-9-13) [\(2025\)](#page-9-13), requires running the ellipsoid algorithm on each iteration, which is slower than quadratic programming [\(Appendix D\)](#page-19-2).

## 5. Game Theory Applications of EVIs

A major motivation for studying Φ-EVIs lies in a strong connection to *(C)CEs* [\(Aumann,](#page-8-5) [1974;](#page-8-5) [Moulin & Vial,](#page-11-9) [1978\)](#page-11-9) in games. Indeed, we begin this section by pointing out that Φ-EVIs capture (C)CEs for specific choices of Φ.

We will mostly consider n-player *concave* games. Here, each player <sup>i</sup> ∈ [n] selects a strategy <sup>x</sup><sup>i</sup> ∈ X<sup>i</sup> from some convex and compact set X<sup>i</sup> , and its utility is given by u<sup>i</sup> : (x1, . . . , <sup>x</sup>n) 7→ <sup>R</sup>. We assume that <sup>u</sup>i(x<sup>i</sup> , x−i) is differentiable and concave in x<sup>i</sup> for any x−<sup>i</sup> , and that the gradients ∇<sup>x</sup>iui(x<sup>i</sup> , <sup>x</sup>−i) are bounded. We let X := X<sup>1</sup> × · · · × Xn.

Example 5.1 (CCE). A distribution <sup>µ</sup> ∈ ∆(X ), is an <sup>ϵ</sup>*coarse correlated equilibrium (CCE)* [\(Moulin & Vial,](#page-11-9) [1978\)](#page-11-9) if for any player <sup>i</sup> ∈ [n],

$$\delta_i := \max_{\mathbf{x}'_i \in \mathcal{X}_i} \mathbb{E}_{\mathbf{x} \sim \mu} u_i(\mathbf{x}'_i, \mathbf{x}_{-i}) - \mathbb{E}_{\mathbf{x} \sim \mu} u_i(\mathbf{x}) \leq \epsilon. \quad (7)$$

Now, consider an ϵ-approximate EVI solution µ of the problem defined by

$$F := (-\nabla_{\mathbf{x}_1} u_1(\mathbf{x}), \dots, -\nabla_{\mathbf{x}_n} u_n(\mathbf{x})).$$

Such µ satisfies, by concavity, P<sup>n</sup> <sup>i</sup>=1 <sup>δ</sup><sup>i</sup> ≤ <sup>ϵ</sup>; it is not necessarily an ϵ-approximate CCE since it is possible that for some <sup>i</sup> ∈ [n], *all* deviations strictly decrease <sup>i</sup>'s utility (so that δ<sup>i</sup> in [\(7\)](#page-5-5) is negative)—µ is technically an *average* CCE in the parlance of [Nadav & Roughgarden](#page-11-15) [\(2010\)](#page-11-15). To capture CCE via Φ-EVIs, one can instead consider a richer set of deviations of the form (x1, . . . , <sup>x</sup>n) 7→ (x1, . . . , <sup>x</sup> ′ i , . . . , xn) for all <sup>i</sup> ∈ [n] and <sup>x</sup> ′ <sup>i</sup> ∈ X<sup>i</sup> .

A canonical example of the above formalism is a *normalform game*, in which each constraint set X<sup>i</sup> is the probability simplex ∆(Ai) over a finite set of *actions* A<sup>i</sup> , and each utility u<sup>i</sup> is a multilinear function.

Example 5.2 (LCE). A distribution <sup>µ</sup> ∈ ∆(X ) is an <sup>ϵ</sup>*linear correlated equilibrium (LCE)* if for any <sup>i</sup> ∈ [n],

$$\max_{\phi_i \in \Phi_i} \mathbb{E}_{\mathbf{x} \sim \mu} u_i(\phi_i(\mathbf{x}_i), \mathbf{x}_{-i}) - \mathbb{E}_{\mathbf{x} \sim \mu} u_i(\mathbf{x}) \leq \epsilon,$$

where <sup>Φ</sup><sup>i</sup> contains all linear functions from X<sup>i</sup> to X<sup>i</sup> . To capture LCE via Φ-EVIs, it suffices to consider deviations of the form (x1, . . . , <sup>x</sup>n) 7→ (x1, . . . , ϕi(xi), . . . , <sup>x</sup>n) for all <sup>i</sup> ∈ [n] and <sup>ϕ</sup><sup>i</sup> ∈ <sup>Φ</sup><sup>i</sup> .

For normal-form games, LCEs amount to the usual notion of CEs [\(Aumann,](#page-8-5) [1974\)](#page-8-5). LCEs were introduced in the context of extensive-form games [\(Farina & Pipis,](#page-10-12) [2023;](#page-10-12) [2024\)](#page-10-8).

Refining correlated equilibria In fact, and more surprisingly, ΦLIN-EVI solutions can be a strict subset of LCEs.[<sup>5</sup>](#page-5-6) This separation can already be appreciated in the setting of normal-form games, and manifests itself in at least two distinct ways. First, there exist games for which a CE need not be a solution to the ΦLIN-EVI. In this sense, ΦLIN-EVIs yield a computationally tractable superset of Nash equilibria that is tighter than CEs. Second, computation suggests that the set of solutions of the ΦLIN-EVI for the game need not be a polyhedron, unlike the set of CEs. We provide a graphical depiction of this phenomenon in [Figure 1.](#page-6-0) The figure depicts the set of ΦLIN-EVI solutions to a simple "Bach or Stravinsky" game, in which the players receive payoffs (3, 2) if they both pick Bach, (2, 3) if they both pick Stravinsky, and (0, 0) otherwise.

<sup>5</sup>The example of [S¸eref Ahunbay](#page-12-3) [\(2025,](#page-12-3) Example 1) already implies that certain CEs can be excluded from the set of ΦLIN-EVIs, which, incidentally, could have implications for last-iterate convergence in some classes of games, as discussed by that author. Our example in [Figure 1](#page-6-0) goes much further, revealing that ΦLIN-EVIs can yield significantly different utilities for each player compared to CEs.

Interpretation The reason for this separation is that, for a map <sup>ϕ</sup> : X → X , each player's mapped strategy <sup>ϕ</sup>(x)<sup>i</sup> can also depend (linearly) on *other players' strategies* x−<sup>i</sup> . Indeed, the EVI formulation of a game does not take into account the identities of the players. For this reason, we will call the set of ΦLIN-EVI solutions in a concave game *anonymous linear correlated equilibria*, or *ALCE* for short. We give two game-theoretic interpretations of ALCEs.

First, the ALCEs of a game Γ are the *symmetric* LCEs of the "symmetrized" game in which the players are randomly shuffled before the game begins. That is, consider the nplayer game Γ sym defined as follows. Each player's strategy set is X . For strategy profile (<sup>x</sup> 1 , . . . , x <sup>n</sup>) ∈ X <sup>n</sup>, the utility to player i is given by

$$u_i^{\text{sym}}(\mathbf{x}^1, \dots, \mathbf{x}^n) = \frac{1}{n!} \sum_{\sigma \in \mathfrak{S}_n} u_{\sigma(i)}(\mathbf{x}_1^{\sigma^{-1}(1)}, \dots, \mathbf{x}_n^{\sigma^{-1}(n)}),$$

where <sup>G</sup><sup>n</sup> is the set of permutations <sup>σ</sup> : [n] → [n]. The following result then follows almost by definition.

Proposition 5.3. *For a given distribution* <sup>µ</sup> ∈ ∆(X )*, define the distribution* µ <sup>n</sup> ∈ ∆(X <sup>n</sup>) *by sampling* <sup>x</sup> ∼ <sup>µ</sup> *and outputting* (x, . . . , <sup>x</sup>) ∈ X <sup>n</sup>*. Then,* <sup>µ</sup> *is a ALCE of* <sup>Γ</sup> *if and only if* µ <sup>n</sup> *is an LCE of* Γ sym*.*

Second, for normal-form games, the ALCEs are the distributions <sup>µ</sup> ∈ ∆(X ) such that no player <sup>i</sup> has a profitable deviation of the following form. The correlation device first samples <sup>x</sup> ∼ <sup>µ</sup>, and samples recommendations <sup>a</sup><sup>j</sup> ∼ <sup>x</sup><sup>j</sup> for each player j. Then, the player selects another player j (possibly j = i) whose recommendation it wishes to see. The player then observes a sample a ′ <sup>j</sup> ∼ <sup>x</sup><sup>j</sup> that is *independently* sampled from a<sup>j</sup> . [<sup>6</sup>](#page-6-1) Finally, the player chooses an action a ∗ <sup>i</sup> ∈ A<sup>i</sup> , and each player j gets reward u<sup>j</sup> (a ∗ i , a<sup>−</sup>i). Thus, players are allowed (modulo the independent sampling) to *spy* on each others' recommendations.

Further discussion about ALCEs and formal proofs of the claims in this section are deferred to [Appendix F.](#page-20-0)

Coupled constraints Continuing from Examples [5.1](#page-5-1) and [5.2,](#page-5-2) we observe that (ΦLIN-)EVIs can be used even in "pseudo-games," in which X does not necessarily decompose into X<sup>1</sup> × · · · × Xn; this means that <sup>x</sup><sup>i</sup> ∈ Xi(x−i). As we discuss in [Appendix A,](#page-13-0) most prior work in such settings has focused on generalized Nash equilibria, with the exception of [Bernasconi et al.](#page-8-6) [\(2023\)](#page-8-6). (ΦLIN-)EVIs induce an interesting notion of LCE/CCE in pseudo-games, albeit not directly comparable to the one put forward by [Bernasconi](#page-8-6) [et al.](#page-8-6) [\(2023\)](#page-8-6). It is worth noting that [Bernasconi et al.](#page-8-6) [\(2023\)](#page-8-6) left open whether efficient algorithms for computing their notion of (coarse) correlated equilibria exist.

![](_page_6_Figure_1.jpeg)

Figure 1. Marginals of the set of correlated equilibria (CE) and of the set of solutions to ΦLIN-EVI in the simple 2 × 2 game "Bach or Stravinsky." The x- and y-axes show the probability with which the two players select the first action (Bach). The set of marginals of ΦLIN-EVI solutions appears to have a curved boundary corresponding, we believe, to the hyperbola 10x <sup>2</sup> − 25xy + 10y <sup>2</sup> − 6x + 11y = 0.

Definition 5.4. Given an n-player pseudo-game with concave, differentiable utilities and joint constraints X , a distribution <sup>µ</sup> ∈ ∆(X ) is an <sup>ϵ</sup>*-ALCE* if

$$\max_{\phi \in \Phi_{\text{LIN}}} \mathbb{E}_{\mathbf{x} \sim \mu} \sum_{i=1}^n u_i(\phi(\mathbf{x})_i, \mathbf{x}_{-i}) - \sum_{i=1}^n u_i(\mathbf{x}) \leq \epsilon.$$

By virtue of our main result [\(Theorem 4.1\)](#page-4-1), such an equilibrium can be computed in polynomial time.

Noncontinuous gradients In fact, our results do not rest on the usual assumption that each player's gradient is a continuous function, thereby significantly expanding the scope of prior known results even in games. For example, we refer to [Dasgupta & Maskin](#page-9-19) [\(1986\)](#page-9-19); [Bichler et al.](#page-8-8) [\(2021\)](#page-8-8); [Martin](#page-11-16) [& Sandholm](#page-11-16) [\(2024\)](#page-11-16) for pointers to some applications.

Nonconcave games Last but not least, Φ-EVIs give rise to a notion of *local* Φ-equilibrium [\(Definition G.1\)](#page-23-0) in nonconcave games. It turns out that this captures recent results by [Cai et al.](#page-9-14) [\(2024a\)](#page-9-14) and [S¸eref Ahunbay](#page-12-3) [\(2025\)](#page-12-3), but our framework has certain important advantages. First, we give a poly(d, log(1/ϵ))-time algorithm [\(Theorem 4.1\)](#page-4-1), while theirs scale polynomially in 1/ϵ. Second, our results do not assume continuity of the gradients. And finally, our algorithms are polynomial even when Φ contains all linear endomorphisms [\(Theorem 4.1\)](#page-4-1). [Appendix G](#page-23-1) elaborates further on those points.

<sup>6</sup>This independence is crucial: without it, µ would actually need to be a distribution over pure Nash equilibria!

#### 6. Problems Where EVIs Coincide with VIs

We saw earlier, in [Proposition 3.6,](#page-3-5) that when Φ comprises all functions from X to X , the <sup>Φ</sup>-EVI problem is tantamount to the associated VI problem. However, if one restricts the functions contained in Φ, are there still structured VIs where we retain this equivalence? In this section, we consider certain structured VIs, and show their equivalence to the corresponding EVIs (that is, ΦCON-EVIs). Unlike general VIs, the ones we examine below are tractable.

#### 6.1. Polymatrix Zero-Sum Games and Beyond

The first important class of VIs we consider is described by a condition given below.

Proposition 6.1. *Suppose that for any* x ′ ∈ X *, the function* <sup>g</sup> : <sup>x</sup> 7→ ⟨F(x), <sup>x</sup> ′ − <sup>x</sup>⟩ *is concave. Then, if* <sup>µ</sup> ∈ ∆(X ) *is an* ϵ*-approximate solution to the EVI,* <sup>E</sup>x∼<sup>µ</sup> x *is an* ϵ*approximate solution to the VI.*

The proof follows directly from Jensen's inequality.

The precondition of [Proposition 6.1](#page-7-0) is satisfied, *e.g.*, when: (i) ⟨F(x), <sup>x</sup>⟩ = 0 for all <sup>x</sup> ∈ X , and (ii) <sup>F</sup> is a linear map. In the context of n-player games, the first condition amounts to the zero-sum property: P<sup>n</sup> <sup>i</sup>=1 ui(x) = 0 for all x. Of course, this property is not enough to enable efficient computation of Nash equilibria, for every two-player (general-sum) game can be converted into a 3-player zero-sum game. This is where the second condition comes into play: F is a linear map—that is, each player's gradient must be linear in the joint strategy. Those two conditions are satisfied in *polymatrix zero-sum* games [\(Cai et al.,](#page-9-20) [2016\)](#page-9-20); in such games, the conclusion of [Proposition 6.1](#page-7-0) is a well-known fact.

#### 6.2. Quasar-Concave Functions

We next consider the problem of maximizing a (single) function that satisfies *quasar-concavity*—a natural generalization of concavity that has received significant interest [\(Hardt](#page-10-13) [et al.,](#page-10-13) [2018;](#page-10-13) [Fu et al.,](#page-10-14) [2023;](#page-10-14) [Hinder et al.,](#page-10-15) [2020;](#page-10-15) [Gower et al.,](#page-10-16) [2021;](#page-10-16) [Guminov et al.,](#page-10-17) [2023;](#page-10-17) [Caramanis et al.,](#page-9-21) [2024\)](#page-9-21).[<sup>7</sup>](#page-7-5)

Definition 6.2 (Quasar-concavity). Let <sup>γ</sup> ∈ (0, 1] and <sup>x</sup> ⋆ ∈ X be a maximizer of a differentiable function <sup>u</sup> : X → <sup>R</sup>. We say that u is γ*-quasar-concave* with respect to x ⋆ if

$$u(\mathbf{x}^*) \leq u(\mathbf{x}) + \frac{1}{\gamma} \langle \nabla u(\mathbf{x}), \mathbf{x}^* - \mathbf{x} \rangle \quad \forall \mathbf{x} \in \mathcal{X}. \quad (8)$$

In particular, in the special case where γ = 1, [\(8\)](#page-7-6) is equivalent to *star-concavity* [\(Nesterov & Polyak,](#page-11-17) [2006\)](#page-11-17). If in addition [\(8\)](#page-7-6) holds for all x <sup>⋆</sup> ∈ X (not merely w.r.t. a global maximizer), it captures the usual notion of concavity.

Any reasonable solution concept for such problems should place all mass on global maxima; EVIs pass this litmus test:

Proposition 6.3. *Let* <sup>F</sup> <sup>=</sup> −∇<sup>u</sup> *for a* <sup>γ</sup>*-quasar-concave and differentiable function* <sup>u</sup> : X → <sup>R</sup>*. Then, for any solution* <sup>µ</sup> ∈ ∆(X ) *to the EVI problem,*

$$\mathbb{E}_{\mathbf{x} \sim \mu} u(\mathbf{x}) \geq \max_{\mathbf{x} \in \mathcal{X}} u(\mathbf{x}).$$

*Thus,* <sup>P</sup>x∼µ[u(x ⋆ ) = u(x)] = 1*, for* x <sup>⋆</sup> ∈ argmax<sup>x</sup> <sup>u</sup>(x)*.*

Indeed, by [Definition 6.2,](#page-7-2) <sup>0</sup> ≤ <sup>E</sup>x∼µ⟨∇u(x), <sup>x</sup> − <sup>x</sup> ⋆ ⟩ ≤ <sup>γ</sup> <sup>E</sup>x∼µ[u(x) − <sup>u</sup>(<sup>x</sup> ⋆ )] for any EVI solution <sup>µ</sup> ∈ ∆(X ). Thus, under quasar-concavity, VIs basically reduce to EVIs.

## 7. Performance Guarantees for EVIs

In many settings, a VI solution is used as a proxy to approximately maximize some underlying objective function; machine learning offers many such applications. The question is whether performance guarantees pertaining to VIs can be extended—potentially with some small degradation to EVIs as well. The purpose of this section is to provide a framework for achieving that based on the following notion.

Definition 7.1. An EVI problem is (λ, ν)*-smooth*, for λ > <sup>0</sup>, ν > −<sup>1</sup>, w.r.t. <sup>W</sup> : X → <sup>R</sup> and <sup>x</sup> <sup>⋆</sup> ∈ argmax<sup>x</sup> <sup>W</sup>(x) if

$$\langle F(\mathbf{x}), \mathbf{x}^* - \mathbf{x} \rangle \leq -\lambda W(\mathbf{x}^*) + (\nu + 1)W(\mathbf{x}) \quad \forall \mathbf{x} \in \mathcal{X}.$$

Example 7.2. When the underlying problem corresponds to a multi-player game and W is the (utilitarian) social welfare, [Definition 7.1](#page-7-3) coincides with the celebrated notion of smoothness *a la `* [Roughgarden](#page-11-11) [\(2015\)](#page-11-11); this is a consequence of multilinearity, which implies that <sup>W</sup>(x) = −⟨x, F(x)⟩ and ⟨<sup>x</sup> ⋆ , F(x)⟩ <sup>=</sup> − P<sup>n</sup> <sup>i</sup>=1 ui(x ⋆ i , <sup>x</sup>−i) for all <sup>x</sup> ∈ X . We also refer to the recent treatment of smoothness by [S¸eref](#page-12-3) [Ahunbay](#page-12-3) [\(2025\)](#page-12-3) in the context of nonconcave games, which builds on the primal-dual framework of [Nadav & Roughgar](#page-11-15)[den](#page-11-15) [\(2010\)](#page-11-15).

[Definition 7.1](#page-7-3) is an extension of the more general notion of "local smoothness," introduced by [Roughgarden & Schopp](#page-11-18)[mann](#page-11-18) [\(2015\)](#page-11-18) in the context of splittable congestion games. However, it goes beyond games. Indeed, the following definition we introduce generalizes [Definition 6.2,](#page-7-2) making a new connection between smoothness and quasar-concavity.

Definition 7.3 (Extension of quasar-concavity). Let x ⋆ ∈ X be a maximizer of a differentiable function <sup>u</sup> : X → <sup>R</sup>. We say that u is (λ, ν)*-smooth* with respect to x ⋆ if

$$\langle \nabla u(\mathbf{x}), \mathbf{x}^* - \mathbf{x} \rangle \geq \lambda u(\mathbf{x}^*) - (\nu + 1)u(\mathbf{x}) \quad \forall \mathbf{x} \in \mathcal{X}.$$

<sup>7</sup> Prior literature mostly uses the term *quasar-convexity*, which is equivalent to quasar-concavity for the opposite function −u.

provide an example of a polynomial that satisfies [Defini](#page-7-7)[tion 7.3](#page-7-7) without being quasar-concave. Now, the key property of [Definition 7.1](#page-7-3) is that any EVI solution approximates the underlying objective—by a factor of ρ := <sup>λ</sup>/1+ν.

Theorem 7.4. *Let* <sup>µ</sup> ∈ ∆(X ) *be an* <sup>ϵ</sup>*-approximate solution to a* (λ, ν)*-smooth EVI problem w.r.t.* <sup>W</sup> : X → <sup>R</sup>*. Then,*

$$\mathbb{E}_{\mathbf{x} \sim \mu} W(\mathbf{x}) \geq \frac{\lambda}{1+\nu} \max_{\mathbf{x} \in \mathcal{X}} W(\mathbf{x}) - \frac{\epsilon}{1+\nu}.$$

The proof follows directly from [Definition 7.1,](#page-7-3) using that <sup>E</sup>x∼µ⟨F(x), <sup>x</sup> <sup>⋆</sup> − <sup>x</sup>⟩ ≥ −<sup>ϵ</sup> and linearity of expectation.

## 8. Conclusions and Future Research

In summary, our main contribution was to introduce and examine a natural relaxation of VIs, which we refer to as *expected* VIs. Unlike VIs, which are marred by computational intractability, we showed that EVIs can be solved efficiently under minimal assumptions. We also uncovered many other intriguing properties of EVIs (*cf.* [Table 1\)](#page-2-1).

There are many promising avenues for future work. VIs enjoy a great reach in a wide range of applications, some of which were discussed earlier in our introduction. It would be interesting to explore in more detail how EVIs fare in such settings compared to VIs. In particular, given that EVIs relax VIs, in addition to their more favorable computational properties, it is likely that they unlock new, more desirable solutions not present under VIs. For example, it is well known (*e.g.*, [Ashlagi et al.,](#page-8-9) [2005\)](#page-8-9) that CEs can achieve better welfare than Nash equilibria in games. In light of the prominence of correlated equilibria in the rich setting of multi-player games, we anticipate EVIs to solidify their place also in other application areas beyond the realm of game theory.

## Impact Statement

This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## Acknowledgments

T.S. is supported by the Vannevar Bush Faculty Fellowship ONR N00014-23-1-2876, National Science Foundation grants RI-2312342 and RI-1901403, ARO award W911NF2210266, and NIH award A240108S001. B.H.Z. is supported by the CMU Computer Science Department Hans Berliner PhD Student Fellowship. E.T, R.E.B., and V.C. thank the Cooperative AI Foundation, Polaris Ventures (formerly the Center for Emerging Risk Research) and Jaan Tallinn's donor-advised fund at Founders Pledge for finan-

cial support. E.T. and R.E.B. are also supported in part by the Cooperative AI PhD Fellowship. G.F is supported by the National Science Foundation grant CCF-2443068. We are grateful to Mete S¸ eref Ahunbay for his helpful feedback. We also thank Andrea Celli and Martino Bernasconi for discussions regarding Φ-equilibria in games with coupled constraints.

## References


[1] Alacaoglu, A., Bohm, A., and Malitsky, Y. Beyond the ¨ golden ratio for variational inequality algorithms. *Journal of Machine Learning Research*, 24:172:1–172:33, 2023. Aliprantis, C. D. and Border, K. C. *Infinite Dimensional Analysis: a Hitchhiker's Guide*. Springer, 2006. Anagnostides, I., Panageas, I., Farina, G., and Sandholm, T. Optimistic policy gradient in multi-player Markov games with a single controller: Convergence beyond the Minty property. In *Conference on Artificial Intelligence (AAAI)*, 2024. Ardagna, D., Panicucci, B., and Passacantando, M. Generalized Nash equilibria for the service provisioning problem in cloud systems. *IEEE Transactions on Services Computing*, 6(4):429–442, 2012. Arrow, K. J. and Debreu, G. Existence of an equilibrium for a competitive economy. *Econometrica*, 22(3):265–290, 1954. Ashlagi, I., Monderer, D., and Tennenholtz, M. On the value of correlation. In *Conference in Uncertainty in Artificial Intelligence (UAI)*, 2005. Aumann, R. Subjectivity and correlation in randomized strategies. *Journal of Mathematical Economics*, 1:67–96, 1974. Babichenko, Y. Query complexity of approximate Nash equilibria. *Journal of the ACM*, 63(4):36:1–36:24, 2016. Bauschke, H. H., Moursi, W. M., and Wang, X. Generalized monotone operators and their averaged resolvents. *Mathematical Programming*, 189(1):55–74, 2021. Bernasconi, M., Castiglioni, M., Marchesi, A., Trovo, F., ` and Gatti, N. Constrained Phi-equilibria. In *International Conference on Machine Learning (ICML)*, 2023. Bernasconi, M., Castiglioni, M., Celli, A., and Farina, G. On the role of constraints in the complexity of min-max optimization. *arXiv:2411.03248*, 2024. Bichler, M., Fichtl, M., Heidekruger, S., Kohring, N., and ¨ Sutterer, P. Learning equilibria in symmetric auction games using artificial neural networks. *Nature Machine Intelligence*, 3(8):687–695, 2021.

[2] Billingsley, P. *Convergence of Probability Measures*. Wiley Series in Probability and Statistics: Probability and Statistics. John Wiley & Sons, second edition, 1999. Black, F. and Scholes, M. The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3): 637–654, 1973. Bohm, A. Solving nonconvex-nonconcave min-max prob- ¨ lems exhibiting weak Minty solutions. *Transactions on Machine Learning Research*, 2023. Boyd, S. and Vandenberghe, L. *Convex Optimization*. Cambridge University Press, 2004. Cai, Y., Candogan, O., Daskalakis, C., and Papadimitriou,

[3] C. H. Zero-sum polymatrix games: A generalization of minmax. *Mathematics of Operations Research*, 41(2): 648–655, 2016. Cai, Y., Daskalakis, C., Luo, H., Wei, C., and Zheng, W. On tractable Φ-equilibria in non-concave games. In *Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS)*, 2024a. Cai, Y., Oikonomou, A., and Zheng, W. Accelerated algorithms for constrained nonconvex-nonconcave min-max optimization and comonotone inclusion. In *International Conference on Machine Learning (ICML)*, 2024b. Capatina, A. *Variational inequalities and frictional contact problems*, volume 31. Springer, 2014. Caramanis, C., Fotakis, D., Kalavasis, A., Kontonis, V., and Tzamos, C. Optimizing solution-samplers for combinatorial problems: the landscape of policy-gradient methods. In *Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS)*, 2024. Caratheodory, C. Uber den Variabilet ´ atsbereich der ¨ Fourier'schen Konstanten von positiven harmonischen Funktionen. *Rendiconti del Circolo Matematico de Palermo*, 32:193–217, 1911. Chen, X., Deng, X., and Teng, S.-H. Settling the complexity of computing two-player Nash equilibria. *Journal of the ACM*, 2009. Choudhury, S., Gorbunov, E., and Loizou, N. Singlecall stochastic extragradient methods for structured nonmonotone variational inequalities: Improved analysis under weaker conditions. In *Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS)*, 2024. Combettes, P. L. and Pennanen, T. Proximal methods for cohypomonotone operators. *SIAM Journal on Control and Optimization*, 43(2):731–742, 2004. Cournot, A. A. *Recherches sur les principes mathematiques ´ de la theorie des richesses (Researches into the Mathe- ´ matical Principles of the Theory of Wealth)*. Hachette, Paris, 1838. Dafermos, S. Traffic equilibrium and variational inequalities. *Transportation Science*, 14(1):42–54, 1980. Dagan, Y., Daskalakis, C., Fishelson, M., and Golowich, N. From external to swap regret 2.0: An efficient reduction for large action spaces. In *Proceedings of the Annual Symposium on Theory of Computing (STOC)*, 2024. Dasgupta, P. and Maskin, E. The existence of equilibrium in discontinuous economic games 1: Theory. *Review of Economic Studies*, 53:1–26, 1986. Daskalakis, C. Non-concave games: A challenge for game theory's next 100 years, 2022. Daskalakis, C., Goldberg, P., and Papadimitriou, C. The complexity of computing a Nash equilibrium. *SIAM Journal on Computing*, 2008. Daskalakis, C., Skoulakis, S., and Zampetakis, M. The complexity of constrained min-max optimization. In *Proceedings of the Annual Symposium on Theory of Computing (STOC)*, 2021. Daskalakis, C., Farina, G., Golowich, N., Sandholm, T., and Zhang, B. H. A lower bound on swap regret in extensiveform games. *arXiv:2406.13116*, 2024. Daskalakis, C., Farina, G., Fishelson, M., Pipis, C., and Schneider, J. Efficient learning and computation of linear correlated equilibrium in general convex games. In *Proceedings of the Annual Symposium on Theory of Computing (STOC)*, 2025. Davis, D., Drusvyatskiy, D., Lee, Y. T., Padmanabhan, S., and Ye, G. A gradient sampling method with complexity guarantees for Lipschitz functions in high and low dimensions. In *Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS)*, 2022. Deligkas, A., Fearnley, J., Hollender, A., and Melissourgos, T. Tight inapproximability for graphical games. In *Conference on Artificial Intelligence (AAAI)*, 2023. Diakonikolas, J., Daskalakis, C., and Jordan, M. I. Efficient methods for structured nonconvex-nonconcave min-max optimization. In *International Conference on Artificial Intelligence and Statistics (AISTATS)*, 2021. Domingo-Enrich, C., Jelassi, S., Mensch, A., Rotskoff,
  - G. M., and Bruna, J. A mean-field analysis of twoplayer zero-sum games. In *Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS)*, 2020.

[4] Etessami, K. and Yannakakis, M. On the complexity of Nash equilibria and other fixed points. In *Proceedings of the Annual Symposium on Foundations of Computer Science (FOCS)*, 2007. Facchinei, F. and Kanzow, C. Generalized Nash equilibrium problems. *Annals of Operations Research*, 175(1):177– 211, 2010. Facchinei, F. and Pang, J.-S. *Finite-dimensional variational inequalities and complementarity problems*. Springer, 2003. Facchinei, F., Fischer, A., and Piccialli, V. Generalized Nash equilibrium problems and Newton methods. *Mathematical Programming*, 117(1):163–194, 2009. Farina, G. and Pipis, C. Polynomial-time linear-swap regret minimization in imperfect-information sequential games. In *Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS)*, 2023. Farina, G. and Pipis, C. Polynomial-time computation of exact Phi-equilibria in polyhedral games. In *Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS)*, 2024. Farina, G., Celli, A., Marchesi, A., and Gatti, N. Simple uncoupled no-regret learning dynamics for extensive-form correlated equilibrium. *Journal of the ACM*, 69(6):41:1– 41:41, 2022. Fischer, A., Herrich, M., and Schonefeld, K. Generalized ¨ Nash equilibrium problems-recent advances and challenges. *Pesquisa Operacional*, 34(3):521–558, 2014. Fu, Q., Xu, D., and Wilson, A. C. Accelerated stochastic optimization methods under quasar-convexity. In *International Conference on Machine Learning (ICML)*, 2023. Fujii, K. Bayes correlated equilibria and no-regret dynamics. *arXiv:2304.05005*, 2023. Goktas, D. and Greenwald, A. Exploitability minimization in games and beyond. In *Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS)*, 2022. Goldstein, A. Optimization of Lipschitz continuous functions. *Mathematical Programming*, 13:14–22, 1977. Gorbunov, E., Taylor, A. B., Horvath, S., and Gidel, G. Con- ´ vergence of proximal point and extragradient-based methods beyond monotonicity: the case of negative comonotonicity. In *International Conference on Machine Learning (ICML)*, 2023. Gordon, G. J., Greenwald, A., and Marks, C. No-regret learning in convex games. In *International Conference on Machine Learning (ICML)*, 2008. Gower, R. M., Sebbouh, O., and Loizou, N. SGD for structured nonconvex functions: Learning rates, minibatching and interpolation. In *International Conference on Artificial Intelligence and Statistics (AISTATS)*, 2021. Greenwald, A. and Jafari, A. A general class of no-regret learning algorithms and game-theoretic equilibria. In *Conference on Learning Theory (COLT)*, 2003. Grotschel, M., Lov ¨ asz, L., and Schrijver, A. The ellipsoid ´ method and its consequences in combinatorial optimization. *Combinatorica*, 1:169–197, 1981. Grotschel, M., Lov ¨ asz, L., and Schrijver, A. ´ *Geometric algorithms and combinatorial optimization*. Springer, 1993. Guminov, S., Gasnikov, A. V., and Kuruzov, I. A. Accelerated methods for weakly-quasi-convex optimization problems. *Computational Management Science*, 20(1): 36, 2023. Hardt, M., Ma, T., and Recht, B. Gradient descent learns linear dynamical systems. *Journal of Machine Learning Research*, 19(1):1025–1068, 2018. Hart, S. and Mas-Colell, A. Uncoupled dynamics do not lead to Nash equilibrium. *American Economic Review*, 93:1830–1836, 2003. Hazan, E. Introduction to online convex optimization. *Foundations and Trends in Optimization*, 2(3-4):157–325, 2016. Hinder, O., Sidford, A., and Sohoni, N. S. Near-optimal methods for minimizing star-convex functions and beyond. In *Conference on Learning Theory (COLT)*, 2020. Hirsch, M. D., Papadimitriou, C. H., and Vavasis, S. A. Exponential lower bounds for finding Brouwer fix points. *Journal of Complexity*, 5(4):379–416, 1989. Hsieh, Y., Liu, C., and Cevher, V. Finding mixed Nash equilibria of generative adversarial networks. In *International Conference on Machine Learning (ICML)*, 2019. Huang, W. and von Stengel, B. Computing an extensiveform correlated equilibrium in polynomial time. In *International Workshop On Internet And Network Economics (WINE)*, 2008. Jordan, M. I., Kornowski, G., Lin, T., Shamir, O., and Zampetakis, M. Deterministic nonsmooth nonconvex optimization. In *Conference on Learning Theory (COLT)*, 2023a.

[5] Jordan, M. I., Lin, T., and Zampetakis, M. First-order algorithms for nonlinear generalized Nash equilibrium problems. *Journal of Machine Learning Research*, 24: 38:1–38:46, 2023b. Kapron, B. M. and Samieefar, K. The computational complexity of variational inequalities and applications in game theory. *arXiv:2411.04392*, 2024. Kinderlehrer, D. and Stampacchia, G. *An introduction to variational inequalities and their applications*. SIAM, 2000. Kuhn, H. W. Extensive games and the problem of information. In *Contributions to the Theory of Games*, volume 2 of *Annals of Mathematics Studies, 28*, pp. 193–216. Princeton University Press, 1953. Lee, S. and Kim, D. Fast extra gradient methods for smooth structured nonconvex-nonconcave minimax problems. In *Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS)*, 2021. Lovasz, L. and Vempala, S. S. Simulated annealing in con- ´ vex bodies and an *O* \*(*<sup>n</sup>*

[6] 4) volume algorithm. *J. Comput. Syst. Sci.*, 72(2):392–417, 2006. Martin, C. and Sandholm, T. Joint-perturbation simultaneous pseudo-gradient. *arXiv:2408.09306*, 2024. Mertikopoulos, P. and Zhou, Z. Learning in games with continuous action sets and unknown payoff functions. *Mathematical Programming*, 173(1-2):465–507, 2019. Milionis, J., Papadimitriou, C., Piliouras, G., and Spendlove,

[7] K. An impossibility theorem in game dynamics. *Proceedings of the National Academy of Sciences*, 120(41), 2023. Morrill, D., D'Orazio, R., Lanctot, M., Wright, J. R., Bowling, M., and Greenwald, A. R. Efficient deviation types and learning for hindsight rationality in extensive-form games. In *International Conference on Machine Learning (ICML)*, 2021a. Morrill, D., D'Orazio, R., Sarfati, R., Lanctot, M., Wright,

[8] J. R., Greenwald, A. R., and Bowling, M. Hindsight and sequential rationality of correlated play. In *Conference on Artificial Intelligence (AAAI)*, 2021b. Moulin, H. and Vial, J.-P. Strategically zero-sum games: The class of games whose completely mixed equilibria cannot be improved upon. *International Journal of Game Theory*, 7(3-4):201–221, 1978. Nadav, U. and Roughgarden, T. The limits of smoothness: A primal-dual framework for price of anarchy bounds. In *International Workshop On Internet And Network Economics (WINE)*, 2010. Nash, J. Non-cooperative games. *Annals of Mathematics*, 54:289–295, 1951. Nesterov, Y. and Polyak, B. T. Cubic regularization of newton method and its global performance. *Mathematical programming*, 108(1):177–205, 2006. Papadimitriou, C. H. and Roughgarden, T. Computing correlated equilibria in multi-player games. *Journal of the ACM*, 55(3):14:1–14:29, 2008. Patris, N. and Panageas, I. Learning Nash equilibria in rank-1 games. In *International Conference on Learning Representations (ICLR)*, 2024. Peng, B. and Rubinstein, A. Fast swap regret minimization and applications to approximate correlated equilibria. In *Proceedings of the Annual Symposium on Theory of Computing (STOC)*, 2024. Pethick, T., Latafat, P., Patrinos, P., Fercoq, O., and Cevher,
  - V. Escaping limit cycles: Global convergence for constrained nonconvex-nonconcave minimax problems. In *International Conference on Learning Representations (ICLR)*, 2022. Roughgarden, T. Intrinsic robustness of the price of anarchy. *Journal of the ACM*, 62(5):32:1–32:42, 2015. Roughgarden, T. and Schoppmann, F. Local smoothness and the price of anarchy in splittable congestion games. *Journal of Economic Theory*, 156:317–342, 2015. Roughgarden, T., Syrgkanis, V., and Tardos, E. The price ´ of anarchy in auctions. *Journal of Artificial Intelligence Research*, 59:59–101, 2017. Rubinstein, A. Inapproximability of Nash equilibrium. In *Proceedings of the Annual Symposium on Theory of Computing (STOC)*, 2015. Rubinstein, A. Settling the complexity of computing approximate two-player Nash equilibria. In *Proceedings of the Annual Symposium on Foundations of Computer Science (FOCS)*, 2016. Sion, M. On general minimax theorems. *Pacific Journal of Mathematics*, 8(1):171 – 176, 1958. Stoltz, G. and Lugosi, G. Learning correlated equilibria in games with compact sets of strategies. *Games and Economic Behavior*, 59(1):187–208, 2007. Tatarenko, T. and Kamgarpour, M. Learning generalized Nash equilibria in a class of convex games. *IEEE Transactions on Automatic Control*, 64(4):1426–1439, 2018.

[9] Tian, L., Zhou, K., and So, A. M. On the finite-time complexity and practical computation of approximate stationarity concepts of lipschitz functions. In *International Conference on Machine Learning (ICML)*, 2022. Villani, C. *Optimal transport: old and new*, volume 338. Springer, 2009. Zhang, B. H., Anagnostides, I., Farina, G., and Sandholm,

[10] T. Efficient Φ-regret minimization with low-degree swap deviations in extensive-form games. In *Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS)*, 2024a. Zhang, B. H., Farina, G., and Sandholm, T. Mediator interpretation and faster learning algorithms for linear correlated equilibria in general extensive-form games. In *International Conference on Learning Representations (ICLR)*, 2024b. Zhang, B. H., Anagnostides, I., Tewolde, E., Berker, R. E., Farina, G., Conitzer, V., and Sandholm, T. Learning and computation of Φ-equilibria at the frontier of tractability. In *Proceedings of the ACM Conference on Economics and Computation (EC)*, 2025. Zhang, J., Lin, H., Jegelka, S., Sra, S., and Jadbabaie, A. Complexity of finding stationary points of nonconvex nonsmooth functions. In *International Conference on Machine Learning (ICML)*, 2020. S¸ eref Ahunbay, M. First-order (coarse) correlated equilibria in non-concave games. *arXiv:2403.18174*, 2025.
#### A. Further Related Work

We have seen that [Definition 1.2](#page-1-0) is strongly connected with the notion of Φ-equilibria. In extensive-form games, the question of characterizing the set of deviations Φ that enables efficient learning—within the no-regret framework—and computation has attracted considerable attention. In particular, efficient algorithms have been established for *extensive-form correlated equilibria (EFCEs)* [\(Huang & von Stengel,](#page-10-18) [2008;](#page-10-18) [Farina et al.,](#page-10-19) [2022;](#page-10-19) [Morrill et al.,](#page-11-19) [2021a](#page-11-19)[;b\)](#page-11-20), and more broadly, when Φ contains solely *linear* functions [\(Farina & Pipis,](#page-10-8) [2024;](#page-10-8) [2023\)](#page-10-12)—corresponding to *linear correlated equilibria (LCEs)*. Recently, [Daskalakis et al.](#page-9-13) [\(2025\)](#page-9-13) strengthened those results beyond extensive-form games whenever there is a separation oracle for the constraint set; we rely on their approach for some of our positive results. Moreover, [Zhang et al.](#page-12-2) [\(2024a;](#page-12-2) [2025\)](#page-12-4) established certain positive results even when Φ contains low-degree polynomials; by contrast, in our setting, Φ-EVIs are hard even when Φ contains only quadratic polynomials [\(Corollary 3.9\)](#page-4-0).

Besides encompassing correlated equilibria in games, wherein the constraint set X can be decomposed as a Cartesian product over the constraint set of each player (reflecting the fact that players select strategies independently), our positive results pertaining to [Definition 1.2](#page-1-0) do not rest on such assumptions and apply even in the presence of joint constraint sets. There is a long history in game theory, optimization, and economics pertaining to such settings—sometimes referred to as "pseudo-games" in the literature [\(Goktas & Greenwald,](#page-10-20) [2022;](#page-10-20) [Arrow & Debreu,](#page-8-0) [1954;](#page-8-0) [Facchinei & Kanzow,](#page-10-21) [2010;](#page-10-21) [Fischer](#page-10-22) [et al.,](#page-10-22) [2014;](#page-10-22) [Facchinei et al.,](#page-10-23) [2009;](#page-10-23) [Ardagna et al.,](#page-8-10) [2012;](#page-8-10) [Tatarenko & Kamgarpour,](#page-11-21) [2018;](#page-11-21) [Jordan et al.,](#page-11-22) [2023b;](#page-11-22) [Daskalakis](#page-9-22) [et al.,](#page-9-22) [2021\)](#page-9-22). The notion of *generalized* Nash equilibria—the natural counterpart of Nash's solution concept in the presence of coupled constraints—has dominated that line of work, with the recent paper of [Bernasconi et al.](#page-8-6) [\(2023\)](#page-8-6) being the notable exception.

Responding to the call of [Daskalakis](#page-9-12) [\(2022\)](#page-9-12), [Cai et al.](#page-9-14) [\(2024a\)](#page-9-14) and [S¸eref Ahunbay](#page-12-3) [\(2025\)](#page-12-3) recently proposed several tractable solution concepts in games with nonconcave utilities. In particular, when specialized to games, Φ-EVIs are closely related to a notion proposed by [S¸eref Ahunbay](#page-12-3) [\(2025,](#page-12-3) Definition 6). One of our key results, [Theorem 4.1,](#page-4-1) establishes a poly(d, log(1/ϵ))-time algorithm, while the algorithms of [Cai et al.](#page-9-10) [\(2024b\)](#page-9-10) and S¸ [eref Ahunbay](#page-12-3) [\(2025\)](#page-12-3) scale polynomially in 1/ϵ. Also, [Theorem 4.1](#page-4-1) applies even when Φ contains all linear endomorphisms [\(Theorem 4.1\)](#page-4-1). From a more conceptual vantage point, a significant part of our contribution is to extend the scope of such results beyond games, as we have already highlighted.

[Kapron & Samieefar](#page-11-23) [\(2024\)](#page-11-23) and [Bernasconi et al.](#page-8-11) [\(2024\)](#page-8-11) recently studied the computational complexity of VIs, and generalizations thereof—namely, *quasi VIs*, establishing PPAD-completeness under mild assumptions. Whether our framework can be extended to encompass quasi VIs is left as an interesting direction for the future.

Finally, it would be remiss not to point out that our VI relaxation is in the spirit of "lifting," a standard technique whereby the original problem is *lifted* to a higher-dimensional space to gain more analytical and computational leverage; a concrete example, in the context of optimal transport theory, is Kantorovich's relaxation of Monge's formulation [\(Villani,](#page-12-5) [2009\)](#page-12-5). Such techniques have been fruitful in the context of min-max optimization [\(Hsieh et al.,](#page-10-24) [2019;](#page-10-24) [Domingo-Enrich et al.,](#page-9-23) [2020\)](#page-9-23).

## B. Additional Preliminaries

Revisiting [Definition 1.2](#page-1-0) In order to define the distributions ∆(X ) over X precisely, we recall here some basic concepts from probability theory. We refer to [Billingsley](#page-9-24) [\(1999,](#page-9-24) Chapter 1 and 2) and [Aliprantis & Border](#page-8-12) [\(2006,](#page-8-12) Chapter 15) for detailed treatments. We assume throughout the paper that the set X ⊆ <sup>R</sup> d is Borel measurable. Let ∆(X ) be the set of Borel probability measures <sup>µ</sup> on X , that is, measures <sup>µ</sup> : (X , B(X )) → (R, B(R)) with <sup>µ</sup>(X ) = 1, where <sup>B</sup>(X ) and B(R) denote the respective σ-algebra of Borel sets. We simply call µ a distribution. For any Borel measurable function <sup>f</sup> : X → <sup>R</sup>—henceforth just *measurable*— we can then take the integral <sup>E</sup>x∼µ[f(x)] := <sup>R</sup> X f(x)dµ(x). In particular, for <sup>E</sup>x∼µ⟨F(x), ϕ(x) − <sup>x</sup>⟩ in [Definition 1.2](#page-1-0) to be well-defined, we assume throughout this paper that <sup>F</sup> and each <sup>ϕ</sup> ∈ <sup>Φ</sup> are measurable functions.

For our computational results [\(Section 4\)](#page-4-2), we are making a standard assumption regarding the geometry of X [\(Section 2\)](#page-2-5); this can be met by bringing X into isotropic position. In particular, there is a polynomial-time algorithm that computes an affine transformation to accomplish that [\(Lovasz & Vempala](#page-11-24) ´ , [2006\)](#page-11-24), and minimizing linear-swap regret reduces to minimizing linear-swap regret to the transformed instance [\(Daskalakis et al.,](#page-9-13) [2025,](#page-9-13) Lemma A.1).

### C. Omitted Proofs

This section contains the proofs omitted from the main body.

## C.1. Existence of Φ-EVI solutions

We begin with [Theorem 3.1.](#page-3-1)

Theorem 3.1. *Suppose that* <sup>F</sup> : X → <sup>R</sup> d *is measurable and there exists* L > <sup>0</sup> *such that every* <sup>ϕ</sup> ∈ <sup>Φ</sup> *is* <sup>L</sup>*-Lipschitz continuous. Then, for any* ϵ > 0*, there exists an* ϵ*-approximate solution to the* Φ*-EVI problem.*

*Proof.* We define a function <sup>F</sup>b<sup>δ</sup> : X → <sup>R</sup> d as

$$\widehat{F}_\delta : \mathbf{x} \mapsto \frac{1}{|\mathcal{B}_\delta(\mathbf{x}) \cap \mathcal{X}|} \int_{\mathcal{B}_\delta(\mathbf{x}) \cap \mathcal{X}} F(\widehat{\mathbf{x}}) d\nu(\widehat{\mathbf{x}});$$

this is a rescaled Lebesgue integral, which represents a multivariate local average. Above,

- δ > 0 is a sufficiently small parameter, to be defined shortly;
- Bδ(x) ⊆ <sup>R</sup> d is the (closed) Euclidean ball of radius δ centered at x; and
- | · | denotes the set's Borel measure.

Given that <sup>F</sup> is assumed to be bounded, we can define <sup>B</sup> ∈ <sup>R</sup> such that maxx∈X ∥F(x)∥ ≤ <sup>B</sup>. For the proof below, it will suffice to set δ := <sup>ϵ</sup>/(L+1)B.

We first observe that Fb<sup>δ</sup> is continuous.

Claim C.1. Fb<sup>δ</sup> *is continuous.*

*Proof.* We will show that for any <sup>x</sup> ∈ X and <sup>ϵ</sup> ′ > 0, we can choose δ ′ = δ ′ (ϵ ′ ) such that for any x ′ ∈ X with ∥x−<sup>x</sup> ′∥ < δ′ ,

$$\|\widehat{F}_\delta(\mathbf{x}) - \widehat{F}_\delta(\mathbf{x}')\| \leq \epsilon'.$$

By the triangle inequality, the difference ∥Fbδ(x) − <sup>F</sup>bδ(<sup>x</sup> ′ )∥ can be decomposed as the sum of

$$\textcircled{A} := \left| \frac{1}{|\mathcal{B}_\delta(\mathbf{x}) \cap \mathcal{X}|} - \frac{1}{|\mathcal{B}_\delta(\mathbf{x}') \cap \mathcal{X}'|} \right| \int_{\mathcal{B}_\delta(\mathbf{x}) \cap \mathcal{X}} \|F(\hat{\mathbf{x}})\| d\nu(\hat{\mathbf{x}})$$

and

$$\textcircled{8} := \frac{1}{|\mathcal{B}_\delta(\mathbf{x}') \cap \mathcal{X}|} \left\| \int_{\mathcal{B}_\delta(\mathbf{x}) \cap \mathcal{X}} F(\hat{\mathbf{x}}) d\nu(\hat{\mathbf{x}}) - \int_{\mathcal{B}_\delta(\mathbf{x}') \cap \mathcal{X}} F(\hat{\mathbf{x}}) d\nu(\hat{\mathbf{x}}) \right\|.$$

Now, ⃝<sup>A</sup> can be bounded as

$$\textcircled{A} \leq B \left| 1 - \frac{|\mathcal{B}_\delta(\mathbf{x}) \cap \mathcal{X}|}{|\mathcal{B}_\delta(\mathbf{x}') \cap \mathcal{X}'|} \right| \leq \frac{1}{2} \epsilon',$$

where we selected δ ′ small enough so that

$$\left(1 - \frac{\epsilon'}{B}\right)|\mathcal{B}_\delta(\mathbf{x}') \cap \mathcal{X}| \leq |\mathcal{B}_\delta(\mathbf{x}) \cap \mathcal{X}| \leq \left(1 + \frac{\epsilon'}{B}\right)|\mathcal{B}_\delta(\mathbf{x}') \cap \mathcal{X}|.$$

Moreover, by selecting δ ′ small enough so that

$$|(\mathcal{B}_\delta(\mathbf{x}) \cap \mathcal{X}) \setminus (\mathcal{B}_\delta(\mathbf{x}') \cap \mathcal{X})| + |(\mathcal{B}_\delta(\mathbf{x}') \cap \mathcal{X}) \setminus (\mathcal{B}_\delta(\mathbf{x}) \cap \mathcal{X})| \leq \frac{1}{2B} \epsilon' |\mathcal{B}_\delta(\mathbf{x}') \cap \mathcal{X}|, \quad (9)$$

we have

$$\begin{aligned} & \left\| \int_{\mathcal{B}_\delta(\mathbf{x}) \cap \mathcal{X}} F(\hat{\mathbf{x}}) d\nu(\hat{\mathbf{x}}) - \int_{\mathcal{B}_\delta(\mathbf{x}') \cap \mathcal{X}} F(\hat{\mathbf{x}}) d\nu(\hat{\mathbf{x}}) \right\| \\ & \leq \int_{(\mathcal{B}_\delta(\mathbf{x}) \cap \mathcal{X}) \setminus (\mathcal{B}_\delta(\mathbf{x}') \cap \mathcal{X})} \|F(\hat{\mathbf{x}})\| d\nu(\hat{\mathbf{x}}) + \int_{(\mathcal{B}_\delta(\mathbf{x}') \cap \mathcal{X}) \setminus (\mathcal{B}_\delta(\mathbf{x}) \cap \mathcal{X})} \|F(\hat{\mathbf{x}})\| d\nu(\hat{\mathbf{x}}) \\ & \leq B|(\mathcal{B}_\delta(\mathbf{x}) \cap \mathcal{X}) \setminus (\mathcal{B}_\delta(\mathbf{x}') \cap \mathcal{X})| + B|(\mathcal{B}_\delta(\mathbf{x}') \cap \mathcal{X}) \setminus (\mathcal{B}_\delta(\mathbf{x}) \cap \mathcal{X})| \leq \frac{1}{2} \epsilon' |\mathcal{B}_\delta(\mathbf{x}') \cap \mathcal{X}|, \end{aligned}$$

where the last inequality uses [\(9\)](#page-14-2). As a result, we have shown that ⃝<sup>A</sup> <sup>+</sup>⃝<sup>B</sup> ≤ <sup>ϵ</sup> ′ , thereby implying that ∥Fbδ(x)−Fbδ(<sup>x</sup> ′ )∥ ≤ <sup>ϵ</sup> ′ . This completes the proof.

Having established that <sup>F</sup>b<sup>δ</sup> is continuous, we can now apply Brouwer's fixed point theorem on the map <sup>x</sup> 7→ <sup>Π</sup><sup>X</sup> (x−Fbδ(x)), where we recall that <sup>Π</sup><sup>X</sup> denotes the Euclidean projection onto X . This implies that there is a point <sup>x</sup> ∈ X such that <sup>x</sup> = Π<sup>X</sup> (<sup>x</sup> − <sup>F</sup>bδ(x)). Moreover, such a point satisfies the VI constraint with respect to <sup>F</sup>bδ:

$$\langle \widehat{F}_\delta(\mathbf{x}), \mathbf{x}' - \mathbf{x} \rangle \geq 0 \quad \mathbf{x}' \in \mathcal{X};$$

for example, see [Kinderlehrer & Stampacchia](#page-11-4) [\(2000,](#page-11-4) Section 3) for the derivation. Finally, we define <sup>µ</sup> ∈ ∆(X ) to be the uniform distribution over Bδ(x) ∩ X . Then, for any <sup>ϕ</sup> ∈ <sup>Φ</sup>,

$$\begin{aligned} \langle \widehat{F}_\delta(\mathbf{x}), \phi(\mathbf{x}) - \mathbf{x} \rangle &= \mathbb{E}_{\widehat{\mathbf{x}} \sim \mu} \langle F(\widehat{\mathbf{x}}), \phi(\mathbf{x}) - \mathbf{x} \rangle \\ &= \mathbb{E}_{\widehat{\mathbf{x}} \sim \mu} \langle F(\widehat{\mathbf{x}}), \widehat{\mathbf{x}} - \mathbf{x} \rangle + \mathbb{E}_{\widehat{\mathbf{x}} \sim \mu} \langle F(\widehat{\mathbf{x}}), \phi(\mathbf{x}) - \phi(\widehat{\mathbf{x}}) \rangle + \mathbb{E}_{\widehat{\mathbf{x}} \sim \mu} \langle F(\widehat{\mathbf{x}}), \phi(\widehat{\mathbf{x}}) - \widehat{\mathbf{x}} \rangle. \end{aligned} \quad (10)$$

The first term in [\(10\)](#page-15-1) can be bounded as

$$\mathbb{E}_{\hat{\mathbf{x}}\sim\mu} \langle F(\hat{\mathbf{x}}), \hat{\mathbf{x}} - \mathbf{x} \rangle \leq \sqrt{\mathbb{E}_{\hat{\mathbf{x}}\sim\mu} \|F(\hat{\mathbf{x}})\|^2} \sqrt{\mathbb{E}_{\hat{\mathbf{x}}\sim\mu} \|\hat{\mathbf{x}} - \mathbf{x}\|^2} \leq \delta B, \quad (11)$$

where we used the Cauchy-Schwarz inequality, the fact that ∥F(xb)∥ ≤ <sup>B</sup> for all <sup>x</sup>b ∈ X , and ∥xb − <sup>x</sup>∥ ≤ <sup>δ</sup> for all <sup>x</sup>b in the support of µ. Similarly, the second term in [\(10\)](#page-15-1) can be bounded as

$$\begin{aligned} \mathbb{E}_{\hat{\mathbf{x}} \sim \mu} \langle F(\hat{\mathbf{x}}), \phi(\mathbf{x}) - \phi(\hat{\mathbf{x}}) \rangle &\leq \sqrt{\mathbb{E}_{\hat{\mathbf{x}} \sim \mu} \|F(\hat{\mathbf{x}})\|^2} \sqrt{\mathbb{E}_{\hat{\mathbf{x}} \sim \mu} \|\phi(\hat{\mathbf{x}}) - \phi(\mathbf{x})\|^2} \\ &\leq L \sqrt{\mathbb{E}_{\hat{\mathbf{x}} \sim \mu} \|F(\hat{\mathbf{x}})\|^2} \sqrt{\mathbb{E}_{\hat{\mathbf{x}} \sim \mu} \|\hat{\mathbf{x}} - \mathbf{x}\|^2} \leq \delta BL, \end{aligned} \quad (12)$$

where we additionally used the assumption that ϕ is L-Lipschitz continuous. Combining [\(11\)](#page-15-2) and [\(12\)](#page-15-3) with [\(10\)](#page-15-1), we have

$$\mathbb{E}_{\hat{\mathbf{x}}\sim\mu} \langle F(\hat{\mathbf{x}}), \phi(\hat{\mathbf{x}}) - \hat{\mathbf{x}} \rangle \geq -\delta(L+1)B + \langle \hat{F}_\delta(\mathbf{x}), \phi(\mathbf{x}) - \mathbf{x} \rangle \geq -\delta(L+1)B,$$

and this holds for any <sup>ϕ</sup> ∈ <sup>Φ</sup>. Setting <sup>δ</sup> := <sup>ϵ</sup>/(L+1)<sup>B</sup> completes the proof.

We next proceed with [Corollary 3.2](#page-3-8) and [Proposition 3.3,](#page-3-9) which are restated below.

Corollary 3.2. *There exists a VI problem that does not admit approximate solutions when* ϵ = Θ(1)*, but the corresponding* ϵ*-approximate* Φ*-EVI is total for any* ϵ > 0*.*

Proposition 3.3. *When* F *is not continuous, there exists an EVI problem with no solutions.*

We provide an example that will establish both of those claims.

Example C.2 (Discontinuous F; *cf.* [Corollary 3.2](#page-3-8) and [Proposition 3.3\)](#page-3-9). Let F(x) be the sign function,

$$F(x) = \text{sgn}(x) := \begin{cases} -1 & \text{if } x < 0, \\ 1 & \text{otherwise,} \end{cases}$$

and X = [−1, 1]. We first claim that there is no <sup>ϵ</sup>-approximate VI solution for ϵ < <sup>1</sup>. Indeed,

- for any x < 0, picking x ′ = 1 ensures ⟨F(x), x′ − <sup>x</sup>⟩ <sup>=</sup> <sup>x</sup> − <sup>1</sup> <sup>&</sup>lt; −<sup>1</sup>;
- for any <sup>x</sup> ≥ <sup>0</sup>, picking <sup>x</sup> ′ <sup>=</sup> −<sup>1</sup> ensures ⟨F(x), x′ − <sup>x</sup>⟩ <sup>=</sup> −<sup>1</sup> − <sup>x</sup> ≤ −<sup>1</sup>.

There is also no *exact* EVI solution to this problem. Indeed, consider any <sup>µ</sup> ∈ ∆(X ).

- If <sup>P</sup>x∼µ[x = 0] = 1, then taking x ′ <sup>=</sup> −<sup>1</sup> ensures <sup>E</sup>x∼µ⟨F(x), x′ − <sup>x</sup>⟩ <sup>=</sup> ⟨F(0), x′ ⟩ <sup>=</sup> −<sup>1</sup>.
- Otherwise, taking x ′ = 0, we have

$$\mathbb{E}_{x \sim \mu} \langle F(x), x' - x \rangle = \mathbb{E}_{x \sim \mu} [-|x|] < 0.$$

On the other hand, for any ϵ > 0, there exists an ϵ-approximate EVI solution (as promised by [Theorem 3.1\)](#page-3-1). In particular, suppose that <sup>µ</sup> uniformly picks between −<sup>ϵ</sup> and <sup>ϵ</sup>. Then, for any <sup>x</sup> ′ ∈ X ,

$$\mathbb{E}_{x \sim \mu} \langle F(x), x' - x \rangle = -\frac{1}{2}(x' + \epsilon) + \frac{1}{2}(x' - \epsilon) = -\epsilon.$$

It is worth pointing out that the above example can be slightly modified so that exact EVI solutions do exist, as we explain below.

Example C.3 (Modification of [Example C.2](#page-15-0) with exact VI). We define F(x) identically to [Example C.2,](#page-15-0) except F( <sup>1</sup>/2) = −<sup>1</sup>. We claim that there is no VI solution for ϵ < <sup>1</sup>/2: any <sup>x</sup> ̸<sup>=</sup> <sup>1</sup>/<sup>2</sup> by [Example C.2,](#page-15-0) and <sup>x</sup> <sup>=</sup> <sup>1</sup>/<sup>2</sup> is not a solution since <sup>y</sup> = 1 ensures ⟨F(x), x′ − <sup>x</sup>⟩ <sup>=</sup> −1/2.

However, there is an exact EVI solution: fix any x <sup>⋆</sup> ∈ [0, <sup>1</sup>/2) and consider µ that uniformly mixes between x = x ⋆ and x = <sup>1</sup>/2. Then, for any x ′ ∈ X ,

$$\mathbb{E}_{x \sim \mu} \langle F(x), x' - x \rangle = \frac{1}{2}(x' - x^*) - \frac{1}{2}\left(x' - \frac{1}{2}\right) = \frac{1}{2}\left(\frac{1}{2} - x^*\right) > 0.$$

Our next result reveals that the precondition of [Theorem 3.1](#page-3-1) with respect to Φ cannot be relaxed to continuity.

Theorem 3.4. *There are* Φ*-EVI instances that do not admit* ϵ*-approximate solutions even when* ϵ = Θ(1)*,* F *is piecewise constant, and* Φ *contains only continuous functions.*

*Proof.* As before, let F(x) be the sign function,

$$F(x) = \text{sgn}(x) := \begin{cases} -1 & \text{if } x < 0, \\ 1 & \text{otherwise,} \end{cases}$$

and <sup>µ</sup> ∈ ∆([−1, 1]) be any distribution. For δ > <sup>0</sup>, let <sup>ϕ</sup><sup>δ</sup> : [−1, 1] → [−1, 1] be given by

$$\phi_\delta(x) = \begin{cases} 1 & \text{if } x < -2\delta, \\ -(x + \delta)/\delta & \text{if } -2\delta \leq x \leq 0, \\ -1 & \text{if } x > 0. \end{cases}$$

Further, let <sup>ϕ</sup>0(x) := − sgn(x). Every <sup>ϕ</sup><sup>δ</sup> (with δ > <sup>0</sup>) is continuous, by construction. Now, note that <sup>ϕ</sup><sup>δ</sup> → <sup>ϕ</sup><sup>0</sup> pointwise when <sup>δ</sup> ↓ <sup>0</sup>, and every <sup>ϕ</sup><sup>δ</sup> is bounded. As a result, by the dominated convergence theorem, we have

$$\begin{aligned} \lim_{\delta \rightarrow 0} \mathbb{E} [F(x)(\phi_\delta(x) - x)] &= \mathbb{E}_{x \sim \mu} [F(x)(\phi_0(x) - x)] \\ &= \mathbb{E}_{x \sim \mu} [-1 - F(x) \cdot x] \leq -1, \end{aligned}$$

where the last line uses the fact that <sup>F</sup>(x)ϕ0(x) = − sgn(x) <sup>2</sup> <sup>=</sup> −<sup>1</sup> and <sup>F</sup>(x)· <sup>x</sup> = sgn(x)· <sup>x</sup> <sup>=</sup> |x| for all <sup>x</sup>. Thus, for any ϵ < <sup>1</sup>, there must be some δ > <sup>0</sup> for which <sup>E</sup>[F(x)(ϕδ(x) − <sup>x</sup>)] <sup>&</sup>lt; −<sup>ϵ</sup>, so <sup>µ</sup> cannot be an <sup>ϵ</sup>-approximate EVI solution.

Continuing on [Section 3,](#page-3-2) we next provide the proof of [Theorem 3.5.](#page-3-10)

Theorem 3.5. *Suppose that*

- *1.* <sup>Φ</sup> *is finite-dimensional, that is, there exists* <sup>k</sup> ∈ <sup>N</sup> *and a kernel map* <sup>m</sup> : X → <sup>R</sup> k *such that every* <sup>ϕ</sup> ∈ <sup>Φ</sup> *can be expressed as* <sup>K</sup>m(x) *for some* <sup>K</sup> ∈ <sup>R</sup> d×k *; and*
- *2. every* <sup>ϕ</sup> ∈ <sup>Φ</sup> *admits a fixed point, that is, a point* X ∋ <sup>x</sup> = FP(ϕ) *such that* <sup>ϕ</sup>(x) = <sup>x</sup>*.*

*Proof.* We assume, without loss of generality, that (as functions) the coordinates m<sup>i</sup> : X → <sup>R</sup> for <sup>1</sup> ≤ <sup>i</sup> ≤ <sup>k</sup> are linearly independent. We further assume that m is bounded, again without loss of generality. (Indeed, if for example m<sup>i</sup> is unbounded, then column i of K must contain all zeros, or else ϕK(x) := Km(x) would be unbounded; we can thus freely remove such coordinates m<sup>i</sup> .)

Now, let K := conv{<sup>K</sup> : <sup>ϕ</sup><sup>K</sup> ∈ <sup>Φ</sup>} be the set of matrices corresponding to maps in <sup>Φ</sup>; we can assume that K is closed. We can now rewrite the Φ-EVI problem as

$$\text{find } \mu \in \Delta(\mathcal{X}) \quad \text{s.t.} \quad \mathbb{E}_{\mathbf{x} \sim \mu} \langle F(\mathbf{x})m(\mathbf{x})^\top, \mathbf{K} - \mathbf{I} \rangle \geq 0$$

for all <sup>K</sup> ∈ K, where above <sup>I</sup> is the identity matrix and the inner product is the usual Frobenius inner product of matrices.[<sup>8</sup>](#page-17-0) Further, let A := conv{F(x)m(x) <sup>⊤</sup> : <sup>x</sup> ∈ X }. Then, the <sup>Φ</sup>-EVI problem can be in turn expressed as

find 
$$\mathbf{A} \in \mathcal{A}$$
 s.t.  $\langle \mathbf{A}, \mathbf{K} - \mathbf{I} \rangle \geq 0$ 

for all <sup>K</sup> ∈ K. Since <sup>F</sup> and <sup>m</sup> are bounded, by assumption, so is A. Moreover, since the coordinates <sup>m</sup><sup>i</sup> are linearly independent, K is also bounded. Thus, letting A¯ denote the closure of A, the max-min problem

$$\max_{\mathbf{A} \in \bar{\mathcal{A}}} \min_{\mathbf{K} \in \mathcal{K}} \langle \mathbf{A}, \mathbf{K} - \mathbf{I} \rangle \quad (13)$$

satisfies the conditions of the minimax theorem. Moreover, for any <sup>K</sup> ∈ K, the fixed point <sup>x</sup> := FP(ϕK) satisfies

$$\langle F(\mathbf{x})m(\mathbf{x})^\top, \mathbf{K} - \mathbf{I} \rangle = \langle F(\mathbf{x}), \phi_{\mathbf{K}}(\mathbf{x}) - \mathbf{x} \rangle = 0,$$

so the zero-sum game [\(13\)](#page-17-1) has a nonnegative value; that is, there exists <sup>A</sup> ∈ A¯ such that minK∈K⟨A, <sup>K</sup> − <sup>I</sup>⟩ ≥ <sup>0</sup>. Thus, for every ϵ > <sup>0</sup>, there exists <sup>A</sup> ∈ A such that minK∈K⟨A, <sup>K</sup> − <sup>I</sup>⟩ ≥ −<sup>ϵ</sup>. Moreover, by Caratheodory's theorem, ´ <sup>A</sup> can be expressed as a convex combination of at most 1 + dk matrices of the form F(x)m(x) <sup>⊤</sup>. This convex combination is thus an ϵ-approximate EVI solution.

The only reason the above proof breaks when <sup>ϵ</sup> = 0 is that A may not be closed. Indeed, this issue is fundamental: there are instances where no exact EVI solutions exist even when ϕ contains only constant functions [\(Proposition 3.3\)](#page-3-9).

#### C.2. Complexity of Φ-EVIs

With regard to the complexity of computing Φ-EVI solutions, the key observation is that, when Φ contains all (measurable) maps, Φ-EVIs are essentially equivalent to VIs; this immediately implies a number of hardness results, which were covered earlier in [Section 3.](#page-3-2) We provide the formal proof of [Proposition 3.6](#page-3-5) below.

Proposition 3.6. *If* <sup>Φ</sup> *contains all measurable functions from* X *to* X *, then any solution* <sup>µ</sup> ∈ ∆(X ) *to the* <sup>ϵ</sup>*-approximate* Φ*-EVI problem satisfies*

$$\mathbb{E}_{x \sim \mu} \text{VIGap}(x) \leq \epsilon. \quad (4)$$

*Proof.* We can define a measurable map <sup>ϕ</sup> : X → X such that <sup>ϕ</sup>(x) is an element selected from argminx′∈X ⟨F(x), <sup>x</sup> ′ − <sup>x</sup>⟩ by utilizing the measurable maximum theorem [\(Aliprantis & Border,](#page-8-12) [2006,](#page-8-12) Theorem 18.19). To satisfy the conditions of this theorem, we need to define—using [Aliprantis & Border'](#page-8-12)s notation— the weakly measurable set-valued function <sup>ψ</sup> : X <sup>↠</sup> X as <sup>ψ</sup>(x) = X and the (Caratheodory) function ´ <sup>f</sup> : X × X → <sup>R</sup> as <sup>f</sup>(x, <sup>x</sup> ′ ) = −⟨F(x), <sup>x</sup> ′ − <sup>x</sup>⟩. Due to this map <sup>ϕ</sup>, a <sup>Φ</sup>-EVI solution <sup>µ</sup> ∈ ∆(X ) must then, in particular, satisfy

$$\mathbb{E}_{x \sim \mu} \langle F(x), \phi(x) - x \rangle = \mathbb{E}_{x \sim \mu} \operatorname{argmin}_{x' \in \mathcal{X}} \langle F(x), x' - x \rangle \geq 0.$$

Therefore, there must exist x <sup>⋆</sup> ∈ X with argminx′∈X ⟨F(<sup>x</sup> ⋆ ), x ′ − <sup>x</sup> ⋆ ⟩ ≥ <sup>0</sup>, that is, a VI solution <sup>x</sup> ⋆ . If µ has finite support, then such a x ⋆ exists within that support. The ϵ-approximation case follows analogously.

<sup>8</sup>To avoid measurability issues, it is enough to consider here only distributions µ with finite support.

#### C.3. Proof of Theorem [4.1](#page-4-1)

To establish [Theorem 4.1,](#page-4-1) we will use the recent framework of [Daskalakis et al.](#page-9-13) [\(2025\)](#page-9-13), which refines [Theorem 2.1](#page-2-4) in the context of [Section 2.1,](#page-2-0) ultimately summarized in [Theorem C.4.](#page-18-1) Coupled with the "semi-separation oracle" of [Lemma 4.2,](#page-4-6) we will thus arrive at [Theorem 4.1.](#page-4-1)

Let X ⊆ <sup>R</sup> d and Y ⊆ <sup>R</sup> <sup>m</sup> be convex and compact sets. The goal is to solve the convex program

$$\text{find } \mu \in \Delta(\mathcal{X}) \quad \text{s.t.} \quad \min_{\mathbf{y} \in \mathcal{Y}} \langle \mu, \mathbf{A}\mathbf{y} \rangle \geq 0, \quad (14)$$

where ∆(X ) ⊆ <sup>R</sup><sup>M</sup> and <sup>A</sup> ∈ <sup>R</sup>M×<sup>m</sup>; we think of <sup>M</sup> as being potentially exponentially large, so <sup>A</sup> is not given explicitly; ΦLIN-EVIs can be expressed as [\(14\)](#page-18-2), assuming that µ has finite support (*cf.* [Theorem 3.5\)](#page-3-10). The target is to solve [\(14\)](#page-18-2) with complexity polynomial in d and m (and other parameters of the problem, except M). As we saw earlier in [Section 2.1,](#page-2-0) the EAH algorithm accomplishes that given access to a GER oracle, which, for any <sup>y</sup> ∈ Y, returns <sup>x</sup> ∈ X such that ⟨µ(x), <sup>A</sup>y⟩ ≥ <sup>0</sup>, where ∆(X ) ∋ <sup>µ</sup>(x) places all probability on <sup>x</sup>. Assuming that such an oracle exists, the convex program

$$\text{find } \mathbf{y} \in \mathbb{R}_{>0}\mathcal{V} \quad \text{s.t.} \quad \max_{\mu \in \Delta(\mathcal{X})} \langle \mu, \mathbf{A}\mathbf{y} \rangle \leq -1 \quad (15)$$

is infeasible, where <sup>R</sup>>0Y := {<sup>c</sup><sup>y</sup> : <sup>y</sup> ∈ Y, c > <sup>0</sup>} is the conic hull of Y. Despite its infeasibility, EAH proceeds by applying the ellipsoid algorithm on [\(15\)](#page-18-3)—this is where the name "ellipsoid against hope" comes from. In doing so, the ellipsoid will eventually shrink to an area with negligible volume (denoted by vol), at which point one can extract a certificate of infeasibility for [\(15\)](#page-18-3) as follows. The execution of the ellipsoid will have produced a sequence of <sup>T</sup> ∈ <sup>N</sup> good-enough-responses, x (1) , . . . , x (T) , such that for any <sup>y</sup> ∈ Y, it holds that ⟨µ(<sup>x</sup> (t) ), <sup>A</sup>y⟩ ≥ <sup>0</sup> for some <sup>t</sup> ∈ [T] (up to numerical imprecision). In turn, this implies that there is a mixture <sup>µ</sup> over {<sup>x</sup> (1) , . . . , x (T)} that guarantees ⟨µ, <sup>A</sup>y⟩ ≥ <sup>0</sup> for every <sup>y</sup> ∈ Y. Such a <sup>µ</sup> can be computed in polynomial time by solving a smaller program, which simply searches over the mixing coefficients.

So far, we have elaborated on the framework presented in [Section 2.1.](#page-2-0) To solve ΦLIN-EVIs, it is necessary to relax the oracle assumed above. In particular, the SEPorGER oracle, introduced by [Daskalakis et al.](#page-9-25) [\(2024\)](#page-9-25), proceeds as follows. It takes as input a point <sup>y</sup> ∈ <sup>R</sup> <sup>m</sup> (not necessarily in Y), and must *either* return a good-enough-response <sup>x</sup> ∈ X , or a hyperplane separating <sup>y</sup> from Y. The idea now is to again run ellipsoid on [\(15\)](#page-18-3), but by replacing Y with a convex "shell set"; every time the SEPorGER oracle returns a separating hyperplane, the shell set restricts further. At the end of this process, once the ellipsoid has shrank enough, one can work with the induced shell set Y<sup>e</sup> and proceed by identifying a mixture among the good-enough-responses {<sup>x</sup> (1) , . . . , x (T)} that approximately solves [\(14\)](#page-18-2). The overall scheme is given in [Algorithm 1.](#page-19-0)

Below, we state the main guarantee of [Algorithm 1](#page-19-0) shown by [Daskalakis et al.](#page-9-13) [\(2025\)](#page-9-13); in its statement, we have made certain slight adjustments in accordance with our setting.

Theorem C.4 [\(Daskalakis et al.,](#page-9-13) [2025\)](#page-9-13). *Suppose that the following conditions hold.*

- *1.* <sup>A</sup> ∈ <sup>R</sup>M×<sup>m</sup> *such that for any* <sup>µ</sup> ∈ ∆(X )*,* ∥<sup>µ</sup> <sup>⊤</sup>A∥ ≤ <sup>B</sup> *for some* <sup>B</sup> ≥ <sup>1</sup>*;*
- *2.* Y *is convex and compact, and satisfies* B<sup>r</sup><sup>y</sup> (·) ⊆ Y ⊆ B<sup>R</sup><sup>y</sup> (0)*; and*
- *3. there exists a* SEPorGER *oracle: for every point* <sup>y</sup> ∈ B<sup>R</sup><sup>y</sup> (0)*, it runs in* poly(d, m) *time, and either returns a hyperplane separating* <sup>y</sup> *from* Y *or a good-enough-response* <sup>x</sup> ∈ X *.*

*Then, [Algorithm 1](#page-19-0) runs in* poly(d, m, log(B/ϵ)) *time and computes* <sup>µ</sup> ∈ ∆(X ) *such that*

$$\min_{y \in \mathcal{Y}} \langle \mu, \mathbf{A}y \rangle \geq -\epsilon.$$

That the second precondition [\(Item 2\)](#page-18-4) is satisfied follows from [Daskalakis et al.](#page-9-13) [\(2025,](#page-9-13) Lemma 2.3). The third precondition, [Item 3,](#page-18-5) is satisfied by virtue of [Lemma 4.2.](#page-4-6) Consequently, [Theorem 4.1](#page-4-1) follows from [Theorem C.4.](#page-18-1)

*Remark* C.5 (Weak oracles and finite precision)*.* Since we are working with general convex sets, the oracles posited in [Section 2](#page-2-5) (namely, membership, separation, and linear optimization) can return irrational outputs. This can be addressed by employing *weak* versions of those oracles, which relax the output by allowing some small slackness ϵ [\(Grotschel et al.](#page-10-6) ¨ , [1993\)](#page-10-6). [Theorem 4.1](#page-4-1) can be readily extended under those weaker oracles; see [Daskalakis et al.](#page-9-13) [\(2025,](#page-9-13) Appendices E and F).

Algorithm 1 Ellipsoid against hope (EAH) under SEPorGER oracle [\(Daskalakis et al.,](#page-9-13) [2025\)](#page-9-13)

- input • Parameters <sup>R</sup>y, r<sup>y</sup> <sup>&</sup>gt; <sup>0</sup> such that B<sup>r</sup><sup>y</sup> (·) ⊆ Y ⊆ B<sup>R</sup><sup>y</sup>
- (0)
- precision parameter ϵ > 0
- constant <sup>B</sup> ≥ <sup>1</sup> such that ∥<sup>µ</sup> <sup>⊤</sup>A∥ ≤ <sup>B</sup> for all <sup>µ</sup> ∈ ∆(X )
- a SEPorGER oracle output A sparse, <sup>ϵ</sup>-approximate solution <sup>µ</sup> ∈ ∆(X ) of [\(14\)](#page-18-2) 1: Initialize the ellipsoid E := B<sup>R</sup><sup>y</sup>
- (0) 2: Initialize Y<sup>e</sup> := B<sup>R</sup><sup>y</sup>
- (0) 3: while vol(E) ≥ vol(Bϵ/B(·)) do 4: Query the SEPorGER oracle on the center of E 5: if it returns a good-enough-response <sup>x</sup> ∈ X then 6: Update E to the minimum volume ellipsoid containing E ∩ {<sup>y</sup> ∈ <sup>R</sup> <sup>m</sup> : ⟨y, <sup>A</sup><sup>⊤</sup>µ(x)⟩ ≤ <sup>0</sup>} 7: else 8: Let <sup>H</sup> be the halfspace that separates <sup>y</sup> from Y 9: Update E to the minimum volume ellipsoid containing E ∩ <sup>H</sup> 10: Update Y<sup>e</sup> := Y ∩<sup>e</sup> <sup>H</sup> 11: end if 12: end while 13: Let x
  - (1) , . . . , x
- (T) be the GER oracle responses produced in the process above 14: Define X := [µ(x (1)) | . . . | <sup>µ</sup>(<sup>x</sup>
- (T) )] and compute <sup>X</sup>⊤<sup>A</sup> ∈ <sup>R</sup> T ×m 15: Compute a solution λ to the convex program

$$\text{find } \boldsymbol{\lambda} \in \Delta^T \quad \text{s.t.} \quad \min_{\boldsymbol{y} \in \tilde{\mathcal{Y}}} \boldsymbol{\lambda}^T (\mathbf{X}^T \mathbf{A}) \boldsymbol{y} \geq -\epsilon$$

16: return ∆(X ) ∋ <sup>µ</sup> := <sup>P</sup><sup>T</sup> <sup>t</sup>=1 λ (t)µ(x (t) )

## D. Characterizing Linear Endomorphisms for Polytopes

In this section, we answer the following question. Given a nonempty polytope X <sup>=</sup> {<sup>x</sup> ∈ <sup>R</sup> d : <sup>A</sup><sup>x</sup> ≤ <sup>b</sup>} where <sup>A</sup> ∈ <sup>R</sup> m×d and <sup>b</sup> ∈ <sup>R</sup> <sup>m</sup>, we wish to characterize the set of (affine) linear maps <sup>ϕ</sup> : X → X . That is, we wish to understand the set of pairs (K, <sup>c</sup>) ∈ <sup>R</sup> <sup>d</sup>×<sup>d</sup> × <sup>R</sup> d such that <sup>K</sup><sup>x</sup> <sup>+</sup> <sup>c</sup> ∈ X for all <sup>x</sup> ∈ X . The following result provides an explicit polynomial representation for that set, establishing [Theorem 4.3.](#page-5-0)

Theorem D.1. <sup>K</sup><sup>x</sup> <sup>+</sup> <sup>c</sup> ∈ X *for all* <sup>x</sup> ∈ X *if and only if there is a matrix* <sup>V</sup> ∈ <sup>R</sup> <sup>m</sup>×<sup>m</sup> *satisfying the constraints*

$$\text{VA} = \text{AK}, \quad \text{V}b \leq b - \text{Ac}, \quad \text{V} \geq 0.$$

*Proof.* Let <sup>K</sup> ∈ <sup>R</sup> d×d and <sup>c</sup> ∈ <sup>R</sup> d , and let a ⊤ <sup>i</sup> <sup>x</sup> ≤ <sup>b</sup><sup>i</sup> be the <sup>i</sup>th constraint that defines X . Then, the claim that a ⊤ i (K<sup>x</sup> <sup>+</sup> <sup>c</sup>) ≤ <sup>b</sup><sup>i</sup> for every <sup>x</sup> ∈ X is equivalent to the claim that the linear program

$$\max_x a_i^\top \mathbf{K} x \quad \text{s.t.} \quad \mathbf{A}x \leq b \quad (16)$$

has value at most <sup>b</sup><sup>i</sup> − <sup>a</sup> ⊤ i c. By strong duality, [\(16\)](#page-19-3) has the same value as

$$\min_{v_i} \mathbf{b}^\top v_i \quad \text{s.t.} \quad \mathbf{A}^\top v_i = \mathbf{K}^\top \mathbf{a}_i, \quad v \geq 0.$$

The theorem follows now by setting V = v<sup>1</sup> . . . v<sup>k</sup> ⊤ .

Furthermore, assuming that B1(0) ⊆ X ⊆ BR(0) with <sup>R</sup> ≤ poly(d), it follows that ∥K∥2, ∥V∥<sup>2</sup> ≤ poly(d), where ∥ · ∥<sup>2</sup> denotes the spectral norm. Indeed, to begin with, ∥c∥<sup>2</sup> ≤ <sup>R</sup> since <sup>K</sup> · <sup>0</sup> <sup>+</sup> <sup>c</sup> ∈ X ⊆ Br(0). For ∥K∥2, take any <sup>x</sup> ∈ <sup>R</sup> <sup>d</sup> with ∥x∥ = 1. Since B1(0) ⊆ X , we have <sup>x</sup> ∈ X , in turn implying that <sup>K</sup><sup>x</sup> <sup>+</sup> <sup>c</sup> ∈ X . As a result, ∥Kx∥ − ∥c∥ ≤ ∥K<sup>x</sup> <sup>+</sup> <sup>c</sup>∥ ≤ poly(d), from which it follows that ∥K∥<sup>2</sup> ≤ poly(d). Further, one can take each <sup>a</sup><sup>i</sup> and <sup>b</sup><sup>i</sup> to be such that <sup>1</sup> ≤ <sup>b</sup><sup>i</sup> ≤ poly(d) and ∥<sup>a</sup>i∥ = 1, and so the bound ∥V∥<sup>2</sup> ≤ poly(d) follows from the fact that <sup>V</sup><sup>b</sup> ≤ <sup>b</sup> − <sup>A</sup><sup>c</sup> and <sup>V</sup> ≥ <sup>0</sup>.

Combining these bounds with [Theorem D.1,](#page-19-1) and as we saw earlier in [Corollary 4.4,](#page-5-4) we are able to use standard techniques for minimizing regret over ΦLIN—such as projected gradient descent.

For comparison, let us point out the approach of [Daskalakis et al.](#page-9-13) [\(2025\)](#page-9-13) for the case where X is given explicitly. To do so, we recall the following definition.

Definition D.2. We say that a polytope X has an *H-representation* of size <sup>m</sup> if it is given as the intersection of <sup>m</sup> halfspaces: X <sup>=</sup> {<sup>x</sup> ∈ <sup>R</sup> d : <sup>A</sup><sup>x</sup> ≤ <sup>b</sup>} for some <sup>A</sup> ∈ <sup>Q</sup>m×<sup>d</sup> and <sup>b</sup> ∈ <sup>Q</sup><sup>m</sup>. It has a <sup>V</sup> *-representation* of size <sup>m</sup> if it is given as the convex hull of <sup>m</sup> vertices: X <sup>=</sup> conv({<sup>v</sup>1, . . . , <sup>v</sup>m}) for <sup>v</sup>1, . . . , <sup>v</sup><sup>m</sup> ∈ <sup>Q</sup><sup>d</sup> .

In this context, they make the following crucial observation [\(Daskalakis et al.,](#page-9-13) [2025,](#page-9-13) Lemmas 3.1 and 3.2).

Lemma D.3 [\(Daskalakis et al.,](#page-9-13) [2025\)](#page-9-13). *If* X *has either an* <sup>H</sup>*-representation of size* <sup>m</sup> *or a* <sup>V</sup> *-representation of size* <sup>m</sup>*, there is a* poly(d, m)*-time membership oracle for* ΦLIN*.*

Using a membership oracle for ΦLIN, it is also possible to construct a linear optimization oracle [\(Grotschel et al.](#page-10-6) ¨ , [1993\)](#page-10-6). As a result, coupled with [Lemma D.3,](#page-20-2) standard algorithms—such as *follow-the-perturbed-leader* [\(Hazan,](#page-10-25) [2016\)](#page-10-25)—can be applied to minimize regret over ΦLIN. However, the main limitation is that constructing a linear optimization oracle using a membership oracle relies on the ellipsoid algorithm, which is impractical. In contrast, [Theorem D.1](#page-19-1) allows us to bypass using the ellipsoid algorithm, resulting in a more practical approach.

It is also worth noting that one can extend [Theorem 4.3](#page-5-0) using only a membership oracle for X (even when X is not an explicitly represented polytope) using techniques from [Daskalakis et al.](#page-9-13) [\(2025\)](#page-9-13), although the resulting algorithm is more elaborate and requires running the ellipsoid algorithm on every iteration to compute the next strategies.

## E. An Illustrative Example of Definition [7.3](#page-7-7)

In [Section 7,](#page-7-4) we introduced a generalized notion of smoothness [\(Definition 7.1\)](#page-7-3) that captures Roughgarden's notion in the context of multi-player games. As a result, there are numerous interesting examples that fall under [Definition 7.1;](#page-7-3) for example, [Roughgarden et al.](#page-11-25) [\(2017\)](#page-11-25) provide a survey in the context of auctions. Our goal here is to provide a single function that satisfies [Definition 7.3,](#page-7-7) but without being quasar-concave (in the sense of [Definition 6.2\)](#page-7-2).

Example E.1. We consider the polynomial function

$$u : x \mapsto -\frac{3}{4}px^4 + px^3 + 1, \quad (17)$$

where <sup>p</sup> ∈ (0, 8]. <sup>u</sup> has a global maximum at <sup>x</sup> = 1, with value 1 + <sup>p</sup>/4. It also admits a VI solution (in fact, a saddle point) at <sup>x</sup> = 0. This implies that <sup>u</sup> is not <sup>γ</sup>-quasar-concave for any <sup>γ</sup> ∈ (0, 1]. On the other hand, it is not hard to verify the following claim.

Claim E.2. <sup>u</sup> *is* (1, <sup>p</sup>/4)*-smooth [\(Definition 7.3\)](#page-7-7) for any* <sup>p</sup> ∈ (0, 8]*.*

A graphical illustration of u for various values of p is given in [Figure 2.](#page-21-0) Coupled with [Theorem 7.4,](#page-8-7) [Claim E.2](#page-20-3) implies that any solution µ to the induced EVI problem satisfies

$$\mathbb{E}_{x \sim \mu} u(x) \geq \frac{1}{1 + \frac{p}{4}} \max u(x) = 1.$$

This guarantee is tight, since x = 0, with u(0) = 1, is a solution to the (E)VI problem.

*Remark* E.3*.* [Definition 7.1,](#page-7-3) to which [Definition 7.3](#page-7-7) is a special case, is a generalization of smoothness in the sense of [Roughgarden & Schoppmann](#page-11-18) [\(2015\)](#page-11-18). While our bound applies to any EVI solution [\(Theorem 7.4\)](#page-8-7), [Roughgarden &](#page-11-18) [Schoppmann](#page-11-18) [\(2015\)](#page-11-18) gave a counter-example that excludes CCEs; this is not a contradiction because they define CCEs *without* linearizing the utilities (as in [Example 5.1\)](#page-5-1), while EVIs always operate over the linearized utilities.

## F. Omitted Details from [Section 5](#page-5-3)

In this section and the next, we will use the notation <sup>Φ</sup>LIN(X , Y) to denote the set of linear maps <sup>ϕ</sup> : X → Y.

In this section, let <sup>Γ</sup> be a concave game. For each player <sup>i</sup> let X<sup>i</sup> ⊂ <sup>R</sup> <sup>d</sup><sup>i</sup> be its (convex, compact) strategy set, and let <sup>Φ</sup><sup>i</sup> ⊆ X <sup>X</sup> i . We assume <sup>u</sup>i(·, <sup>x</sup>−i) is differentiable in <sup>x</sup><sup>i</sup> for all <sup>i</sup>. [<sup>9</sup>](#page-20-4) Without loss of generality we assume that the projection

<sup>9</sup>By "f : C → <sup>R</sup> is differentiable" when C is closed, we mean that f is defined and differentiable on an open set C ⊃ C ˆ .

![](_page_21_Figure_1.jpeg)

Figure 2. Function u, defined in [\(17\)](#page-20-5), for p ∈ {1, 2, 4, 8}.

function πi(x) = x<sup>i</sup> is in Φ<sup>i</sup> , and that Φ<sup>i</sup> is convex. Crucially for this section and departing to our knowledge from all prior work on <sup>Φ</sup>-equilibria in games, functions <sup>ϕ</sup><sup>i</sup> ∈ <sup>Φ</sup><sup>i</sup> are allowed to depend not just on <sup>x</sup><sup>i</sup> but also on other <sup>x</sup>−is. We first generalize the examples in [Section 5](#page-5-3) to arbitrary Φ.

Definition F.1. An ϵ*-approximate* (Φi) n <sup>i</sup>=1*-equilibrium* of <sup>Γ</sup> is a distribution <sup>µ</sup> ∈ ∆(X ) such that

$$\mathbb{E}_{\mathbf{x} \sim \mu} [u_i(\phi_i(\mathbf{x}_i), \mathbf{x}_{-i}) - u_i(\mathbf{x})] \leq \epsilon$$

for all players <sup>i</sup> and deviations <sup>ϕ</sup><sup>i</sup> ∈ <sup>Φ</sup><sup>i</sup>

.

As discussed in [Section 5,](#page-5-3) several special cases of Φ<sup>i</sup> are well-studied and interesting:

- When Φ<sup>i</sup> contains the set of constant functions and the projection π<sup>i</sup> , the set of (Φi) n <sup>i</sup>=1-equilibria are the CCEs.
- When Φ<sup>i</sup> consists of linear functions depending only on x<sup>i</sup> , *i.e.*, functions of the form <sup>x</sup> 7→ <sup>A</sup>x<sup>i</sup> for matrices <sup>A</sup>, (Φi) n <sup>i</sup>=1-equilibria are LCEs, which correspond to CEs in the special case of normal-form games.
- When <sup>Φ</sup><sup>i</sup> consists of all functions X → X<sup>i</sup> , (Φi) n <sup>i</sup>=1-equilibria are Nash equilibria.

Now let X <sup>=</sup> X<sup>1</sup> × · · · × X<sup>n</sup> ⊂ <sup>R</sup> <sup>d</sup> where d = P i di , and define <sup>Φ</sup> ⊆ X <sup>X</sup> to be the set of all functions of the form

$$x \mapsto (x_1, \dots, \phi_i(x), \dots, x_n)$$

for players <sup>i</sup> and functions <sup>ϕ</sup><sup>i</sup> ∈ <sup>Φ</sup><sup>i</sup> . We will abuse notation and also call these functions ϕ<sup>i</sup> : X → X . Moreover, let <sup>F</sup> : X → <sup>R</sup> P i <sup>d</sup><sup>i</sup> be given by <sup>F</sup>(x) = −(∇<sup>x</sup><sup>1</sup> <sup>u</sup>1(x), . . . , ∇<sup>x</sup><sup>n</sup> <sup>u</sup>n(x)).

Proposition F.2. *If* µ *is an* ϵ*-approximate* Φ*-EVI solution of* F*, then* µ *is an* ϵ*-approximate* (Φi) n <sup>i</sup>=1*-equilibrium* Γ*. The converse holds for* ϵ = 0*.*

*Proof.* Suppose first that <sup>µ</sup> ∈ ∆(X ) is an <sup>ϵ</sup>-approximate <sup>Φ</sup>-EVI solution of <sup>F</sup>. Then, for any player <sup>i</sup> and deviation <sup>ϕ</sup><sup>i</sup> ∈ <sup>Φ</sup><sup>i</sup> , we have

$$\mathbb{E}_{\mathbf{x} \sim \mu} [u_i(\phi_i(\mathbf{x}), \mathbf{x}_{-i}) - u_i(\mathbf{x})] \leq \mathbb{E}_{\mathbf{x} \sim \mu} \langle \nabla_{\mathbf{x}_i} u_i(\mathbf{x}), \phi_i(\mathbf{x}) - \mathbf{x}_i \rangle = \mathbb{E}_{\mathbf{x} \sim \mu} \langle -F(\mathbf{x}), \phi_i(\mathbf{x}) - \mathbf{x} \rangle \leq \epsilon,$$

where the first inequality is concavity and the second is the definition of Φ-EVI. Conversely, suppose that µ is an (exact) (Φi) n <sup>i</sup>=1-equilibrium of <sup>Γ</sup>. For <sup>λ</sup> ∈ <sup>R</sup> let <sup>ϕ</sup> λ <sup>i</sup> <sup>=</sup> λϕ<sup>i</sup> + (1 − <sup>λ</sup>)π<sup>i</sup> . Let <sup>g</sup> : [0, 1] × X → <sup>R</sup> be defined by

$$g(\lambda, \mathbf{x}) = u_i(\phi_i^\lambda(\mathbf{x}), \mathbf{x}_{-i}) - u_i(\mathbf{x}).$$

Then, g is differentiable in λ for any fixed x, and g is bounded. Let G(λ) = <sup>E</sup>x∼<sup>µ</sup> g(λ, x). Then G(0) = 0, and by the Leibniz rule, G is differentiable with derivative

$$\begin{aligned} G'(0) &= \mathbb{E}_{\mathbf{x} \sim \mu} \nabla_{\lambda} g(0, \mathbf{x}) \\ &= \mathbb{E}_{\mathbf{x} \sim \mu} \left\langle \nabla_{\mathbf{x}_i} u_i(\mathbf{x}), \lim_{\lambda \rightarrow 0} \frac{1}{\lambda} (\phi_i^{\lambda}(\mathbf{x}) - \mathbf{x}_i) \right\rangle \\ &= \mathbb{E}_{\mathbf{x} \sim \mu} \langle \nabla_{\mathbf{x}_i} u_i(\mathbf{x}), \phi_i(\mathbf{x}) - \mathbf{x}_i \rangle \\ &= \mathbb{E}_{\mathbf{x} \sim \mu} \langle -F(\mathbf{x}), \phi_i(\mathbf{x}) - \mathbf{x} \rangle, \end{aligned}$$

where we use the chain rule, then the definition of ϕ λ i , and finally the definition of <sup>F</sup>. But if <sup>E</sup>x∼µ⟨−F(x), ϕi(x) − <sup>x</sup>⟩ <sup>&</sup>gt; <sup>0</sup>, then by definition of derivative, there is some λ > 0 for which G(λ) > 0, contradicting the definition of (Φi) n <sup>i</sup>=1 equilibrium.

We now prove the following generalization of [Proposition 5.3.](#page-6-2)

Proposition F.3. *For a given distribution* <sup>µ</sup> ∈ ∆(X )*, define the distribution* <sup>µ</sup> <sup>n</sup> ∈ ∆(X <sup>n</sup>) *by sampling* <sup>x</sup> ∼ <sup>µ</sup> *and outputting* (x, . . . , <sup>x</sup>) ∈ X <sup>n</sup>*. Then the* (Φ1, . . . , <sup>Φ</sup>n)*-equilibria of* <sup>Γ</sup> *are precisely the* (Φ, . . . , Φ)*-equilibria of* <sup>Γ</sup> sym*.*

*Proof.* µ <sup>n</sup> is an ϵ-approximate (Φ, . . . , Φ)-equilibria of Γ sym if and only if, for every player <sup>i</sup> and linear map <sup>ϕ</sup> : X → X , we have

$$\begin{aligned} 0 &\geq \frac{1}{n!} \sum_{\sigma \in \mathfrak{S}_n} \mathbb{E}_{\sim \mu} [u_{\sigma(i)}(\mathbf{x}_1, \dots, \phi(\mathbf{x})_{\sigma(i)}, \dots, \mathbf{x}_n) - u_{\sigma^{-1}(i)}(\mathbf{x})] \\ &= \frac{1}{n} \sum_{j \in [n]} \mathbb{E}_{\sim \mu} [u_j(\mathbf{x}_1, \dots, \phi(\mathbf{x})_j, \dots, \mathbf{x}_n) - u_j(\mathbf{x})]. \end{aligned}$$

But this holds if and only if

$$\mathbb{E}_{\mathbf{x} \sim \mu} [u_j(\mathbf{x}_1, \dots, \phi_j(\mathbf{x}), \dots, \mathbf{x}_n) - u_j(\mathbf{x})] \leq 0$$

for every player <sup>j</sup> and every <sup>ϕ</sup><sup>j</sup> ∈ <sup>Φ</sup><sup>j</sup> , which is precisely the definition of an (Φ1, . . . , <sup>Φ</sup>n)-equilibria of <sup>Γ</sup>.

[Proposition 5.3](#page-6-2) follows by combining [Proposition F.3](#page-22-0) and [Proposition F.2](#page-21-1) in the special case when <sup>Φ</sup><sup>i</sup> = ΦLIN(X , Xi).

#### F.1. Anonymous linear correlated equilibria

For the special case where <sup>Φ</sup><sup>i</sup> = ΦLIN(X , Xi), we have coined the resulting (Φi) n <sup>i</sup>=1-equilibrium notion an *anonymous linear correlated equilibrium* (ALCE). We now compare ALCEs and LCEs in concave games. We now point out some intriguing properties of ALCEs, especially compared to LCEs and CEs.

In normal-form games Γ, LCEs and CEs coincide, and ALCEs lie strictly between LCEs and Nash equilibria, as can be seen in [Figure 1.](#page-6-0) We now elaborate on the normal-form specific game-theoretic interpretation of ALCEs by giving an augmented game-based definition. For any fixed <sup>µ</sup> ∈ ∆(X ), consider the augmented game <sup>Γ</sup> <sup>µ</sup> that proceeds as follows.

- 1. A correlation device samples <sup>x</sup> ∼ <sup>µ</sup>.
- 2. Each player <sup>i</sup> chooses a player <sup>j</sup> (possibly not itself) and observes a sample <sup>a</sup><sup>j</sup> ∼ <sup>x</sup><sup>j</sup> , *independently from the samples of other players*. (In particular, if multiple players choose the same player j, then they get independent samples from x<sup>j</sup> ).
- 3. Each player selects an action <sup>a</sup><sup>i</sup> ∈ A<sup>i</sup> and gets utility <sup>u</sup>i(a1, . . . , an).

Proposition F.4. *A distribution* <sup>µ</sup> ∈ ∆(X ) *is a ALCE of* <sup>Γ</sup> *if and only if the strategy profile in which every player requests an action for itself and then plays that action is a Nash equilibrium of* Γ µ*.*

Lemma F.5 [\(Fujii,](#page-10-26) [2023\)](#page-10-26). *Let* X <sup>=</sup> X<sup>1</sup> × · · · × X<sup>n</sup> *where each* X<sup>i</sup> *is a simplex* X<sup>i</sup> = ∆(mi)*. Then every linear map* <sup>ϕ</sup> : X → X<sup>i</sup> *is a convex combination of linear maps* <sup>ϕ</sup><sup>j</sup> : X → X<sup>i</sup> *that only depend on a single* x<sup>j</sup> *.*

*Proof of [Proposition F.4.](#page-22-1)* Fix some <sup>µ</sup> ∈ ∆(X ) and suppose that it is not a ALCE, that is, there is some profitable deviation <sup>ϕ</sup> : X → X<sup>i</sup> for some player <sup>i</sup>. By [Lemma F.5,](#page-23-2) it suffices to assume that <sup>ϕ</sup> only depends on one player's strategy <sup>x</sup><sup>j</sup> . Moreover, a linear map <sup>ϕ</sup> : X<sup>j</sup> → X<sup>i</sup> can be represented as <sup>x</sup><sup>j</sup> 7→ <sup>A</sup>x<sup>i</sup> , where <sup>A</sup> ∈ <sup>R</sup> mi×m<sup>j</sup> is column-stochastic. Again, it suffices to assume that ϕ is a vertex of the set of column-stochastic matrices, that is, A has exactly one 1 in each column. Now player i's deviation benefit under deviation ϕ is given by

$$\mathbb{E}_{\mathbf{x} \sim \mu} [u_i(\phi_j(\mathbf{x}_j), \mathbf{x}_{-i}) - u_i(\mathbf{x})] = \mathbb{E}_{\substack{\mathbf{x} \sim \mu \\ a \sim \mathbf{x}}} \left[ \mathbb{E}_{a'_j \sim \mathbf{x}_j} u_i(\phi_j(a'_j), a_{-i}) - u_i(a) \right],$$

where the equality uses multilinearity of a. This is precisely the deviation benefit of the strategy in Γ <sup>µ</sup> for player i in which player i chooses to sample a ′ j and then plays an action according to <sup>ϕ</sup><sup>j</sup> : [m<sup>j</sup> ] → [m<sup>i</sup> ]. The proposition now follows by observing that these are precisely the possible pure strategy deviations of player i in Γ µ.

We make several more observations about the relationship between ALCEs and other notions of equilibrium in games.

- [Proposition F.4](#page-22-1) generalizes beyond normal-form games, but needs to be modified. For example, for (single-step) Bayesian games where each X<sup>i</sup> is itself a product of simplices, it follows from a similar proof that, in the augmented game Γ <sup>µ</sup>, player i should be allowed to observe its own type first, and then select both another player j and a type θ<sup>j</sup> of that player at which to ask for a recommendation. (Another way to see this is that the EVI formulation does not distinguish Bayesian games from their *agent form* [\(Kuhn,](#page-11-26) [1953\)](#page-11-26), where each player-type pair is treated as a separate player.)

Even more generally, for extensive-form games, we can generalize ALCEs using a characterization of the linear maps X → X<sup>i</sup> due to [Zhang et al.](#page-12-6) [\(2024b\)](#page-12-6): in <sup>Γ</sup> <sup>µ</sup>, player i first may observe its first recommendation at any time of its choosing, and may delay its choice of which player j to observe until that point.

- In normal-form games, CEs can be without loss of generality defined as distributions over *pure* action profiles A <sup>=</sup> A<sup>1</sup> × · · · × A<sup>n</sup> instead of distributions over mixed strategy profiles X <sup>=</sup> X<sup>1</sup> × · · · × X<sup>n</sup> [\(Aumann,](#page-8-5) [1974\)](#page-8-5). By *without loss of generality* here, we mean the following: given any <sup>µ</sup> ∈ ∆(X ), define <sup>µ</sup> ′ ∈ ∆(A) by sampling <sup>x</sup> ∼ <sup>µ</sup>, then <sup>a</sup><sup>i</sup> ∼ <sup>x</sup><sup>i</sup> for each <sup>i</sup>. Then <sup>µ</sup> is a correlated equilibrium if and only if <sup>µ</sup> ′ is.

This phenomenon is *not* true for ALCEs. Indeed, for two-player games, if µ ′ ∈ ∆(A) is a ALCE, then in fact <sup>µ</sup> ′ is a distribution over pure Nash equilibria, which in general may not even exist! It is thus critical in our definition that µ be allowed to be a distribution over *mixed* strategy profiles, not just *pure* strategy profiles.

- We have shown that there is an efficient algorithm for computing *one* (approximate) ALCE. We leave as an open question the complexity of computing an *optimal* (*e.g.*, welfare-maximizing) ALCE (when the number of players n is a constant). Optimal CEs can be computed efficiently in this setting, because the set of CEs <sup>µ</sup> ∈ ∆(A) is bounded by a small number of linear constraints; however, this fails for ALCEs because, as above, we need to optimize over <sup>µ</sup> ∈ ∆(X ).

#### G. Local (Φi) n <sup>i</sup>=1-Equilibria in Nonconcave Games

This section connects Φ-EVIs with a solution concept recently put forward by [Cai et al.](#page-9-14) [\(2024a\)](#page-9-14) (see also [S¸eref Ahunbay](#page-12-3) [2025\)](#page-12-3) in the context of nonconcave games [\(Proposition G.4\)](#page-24-0).

Nonconcave games Consider an <sup>n</sup>-player game in which each player <sup>i</sup> ∈ [n] has a convex and compact strategy set X<sup>i</sup> , and a differentiable utility function u<sup>i</sup> : X<sup>1</sup> × · · · × X<sup>n</sup> → <sup>R</sup>. Crucially, there is now no assumption that <sup>u</sup><sup>i</sup> is concave. In this setting, our framework suggests the following definition.

Definition G.1. Given sets of functions <sup>Φ</sup> ⊆ X <sup>X</sup><sup>i</sup> i , an ϵ*-approximate local* (Φi) n <sup>i</sup>=1*-equilibrium* in an n-player nonconcave game is a distribution <sup>µ</sup> ∈ ∆(X<sup>1</sup> × · · · × Xn) such that for any player <sup>i</sup> ∈ [n] and deviation <sup>ϕ</sup><sup>i</sup> ∈ <sup>Φ</sup><sup>i</sup> ,

$$\mathbb{E}_{\mathbf{x} \sim \mu} \langle \nabla_{\mathbf{x}_i} u_i(\mathbf{x}), \phi_i(\mathbf{x}_i) - \mathbf{x} \rangle \leq \epsilon.$$

[Theorem 4.1](#page-4-1) immediately implies the following result when <sup>Φ</sup><sup>i</sup> = ΦLIN(X<sup>i</sup> , Xi); as before, in what follows, we assume a membership oracle for each X<sup>i</sup> .

Corollary G.2. *Suppose* ∥∇<sup>u</sup>i(x)∥ ≤ <sup>B</sup> *for every player* <sup>i</sup> ∈ [n] *and profile* <sup>x</sup> ∈ X<sup>1</sup> × · · · × Xn*. Then, there is a* poly(d, log(B/ϵ))*-time algorithm that outputs an* ϵ*-approximate local* (Φi) n <sup>i</sup>=1*-equilibrium.*

Similarly, the existence of linear swap-regret minimizers for arbitrary polytopes X<sup>i</sup> [\(Daskalakis et al.,](#page-9-13) [2025\)](#page-9-13) immediately implies the following.

Corollary G.3. *There is an independent no-regret learning algorithm that computes* ϵ*-approximate local* (Φi) n <sup>i</sup>=1*-equilibria in* poly(d, 1/ϵ) *rounds and* poly(d, 1/ϵ) *per-round runtime.*

[Cai et al.](#page-9-14) [\(2024a\)](#page-9-14) also studied the problem of computing local (Φi) n <sup>i</sup>=1-equilibria in nonconcave games. They defined ϵ-local (Φi) n <sup>i</sup>=1-equilibria instead by restricting the magnitudes of the deviations to the "first-order" regime where local deviations cannot change the gradients by too much. In particular, they assume that utility functions u<sup>i</sup> are smooth, in the sense that

$$\|\nabla_{\mathbf{x}_i} u_i(\mathbf{x}_i, \mathbf{x}_{-i}) - \nabla_{\mathbf{x}_i} u_i(\mathbf{x}'_i, \mathbf{x}_{-i})\|_2 \leq L \|\mathbf{x}_i - \mathbf{x}'_i\| \quad \forall \mathbf{x}_i, \mathbf{x}'_i \in \mathcal{X}_i, \forall \mathbf{x}_{-i} \in \bigcup_{i' \neq i} \mathcal{X}_{i'},$$

where L > 0 is a constant. Then, they restrict deviations to only slightly perturb the strategies, that is, for a given set <sup>Φ</sup><sup>i</sup> ⊆ X <sup>X</sup><sup>i</sup> i , they define a set

$$\Phi_i(\delta) := \{\lambda \phi_i + (1-\lambda) \operatorname{Id} : \phi_i \in \Phi_i, \lambda \leq \delta/D_i\},$$

where Id : X → X is the identity function and <sup>D</sup><sup>i</sup> is the <sup>ℓ</sup>2-diameter of X<sup>i</sup> , *i.e.*, ∥<sup>x</sup> − <sup>x</sup> ′∥<sup>2</sup> ≤ <sup>D</sup><sup>i</sup> for all <sup>x</sup>, <sup>x</sup> ′ ∈ X<sup>i</sup> . With this restriction, they show [\(Cai et al.,](#page-9-14) [2024a,](#page-9-14) Lemma 1 and Theorem 10) that Φ-regret minimizers converge to Φ(δ)-equilibria, in the sense that

$$\mathbb{E}_{\mathbf{x} \sim \mu} [u_i(\phi_i(\mathbf{x}_i), \mathbf{x}_{-i}) - u_i(\mathbf{x})] \leq \frac{\delta}{D_i} \frac{\Phi\text{-Reg}_i^{(T)}}{T} + \frac{\delta^2 L}{2},$$

where Φ-Reg<sup>i</sup> is the <sup>Φ</sup>i-regret of Player <sup>i</sup> ∈ [n], for all players <sup>i</sup> and deviations <sup>ϕ</sup><sup>i</sup> ∈ <sup>Φ</sup>i(δ). Our results imply theirs, in the following sense.

Proposition G.4. *Any* ϵ*-approximate local* (Φi) n <sup>i</sup>=1*-equilibrium* µ *(per [Definition G.1\)](#page-23-0) satisfies*

$$\mathbb{E}_{\mathbf{x} \sim \mu} [u_i(\phi_i(\mathbf{x}_i), \mathbf{x}_{-i}) - u_i(\mathbf{x})] \leq \frac{\delta\epsilon}{D_i} + \frac{\delta^2 L}{2}$$

*for any player* <sup>i</sup> ∈ [n] *and deviation* <sup>ϕ</sup><sup>i</sup> ∈ <sup>Φ</sup>i(δ)*.*

*Proof.* Write ϕ<sup>i</sup> = λϕ<sup>∗</sup> <sup>i</sup> + (1 − <sup>λ</sup>) Id for some <sup>ϕ</sup> ∗ <sup>i</sup> ∈ <sup>Φ</sup><sup>i</sup> . Then,

$$u_i(\phi_i(\mathbf{x}_i), \mathbf{x}_{-i}) - u_i(\mathbf{x}) \leq \langle \nabla_{\mathbf{x}_i} u_i(\mathbf{x}), \phi_i(\mathbf{x}_i) - \mathbf{x}_i \rangle + \frac{L}{2} \|\phi_i(\mathbf{x}_i) - \mathbf{x}_i\|_2^2 \leq \frac{\delta}{D_i} \langle \nabla_{\mathbf{x}_i} u_i(\mathbf{x}), \phi_i^*(\mathbf{x}_i) - \mathbf{x}_i \rangle + \frac{\delta^2 L}{2},$$

where the last inequality uses the fact that <sup>λ</sup> ≤ δ/D<sup>i</sup> and therefore ∥<sup>ϕ</sup>i(xi) − <sup>x</sup>i∥<sup>2</sup> ≤ <sup>λ</sup>∥<sup>ϕ</sup> ∗ i (xi) − <sup>x</sup>i∥<sup>2</sup> ≤ λD<sup>i</sup> ≤ <sup>δ</sup>. Taking expectations over µ and applying the definition of ϵ-approximate local (Φi) n <sup>i</sup>=1-equilibrium completes the proof.

However, our results improve on theirs in several ways:

- We believe that the formulation of local (Φi) n <sup>i</sup>=1-equilibria using gradients directly instead of restricting to small perturbations is more natural and more directly conveys what it means for a distribution to be a local (Φi) n <sup>i</sup>=1-equilibrium without introducing too many hyperparameters; one of the notions proposed by [S¸eref Ahunbay](#page-12-3) [\(2025,](#page-12-3) Definition 6) also shares this advantage.
- Our results do not require the smoothness of the utility functions u<sup>i</sup> .
- We have an ellipsoid-based algorithm that computes local (Φi) n <sup>i</sup>=1-equilibria with convergence rate depending on log(1/ϵ), whereas no-regret algorithms only achieve poly(1/ϵ) convergence rate.
- Although we do not explicitly state it here, [Definition G.1](#page-23-0) and [Corollary G.2](#page-23-3) extend directly to the case where <sup>Φ</sup><sup>i</sup> = ΦLIN(X , Xi) (instead of <sup>Φ</sup>LIN(X<sup>i</sup> , Xi)). Per [Appendix F,](#page-20-0) this can yield an even smaller set of equilibria.