# 

Brian Hu Zhang * 1 **Ioannis Anagnostides** * 1 **Emanuel Tewolde** 1 2 **Ratip Emin Berker** 1 2 **Gabriele Farina** 3 Vincent Conitzer 1 2 4 **Tuomas Sandholm** 1 5 6 7

## Abstract

Variational inequalities (VIs) encompass many fundamental problems in diverse areas ranging from engineering to economics and machine learning. However, their considerable expressivity comes at the cost of computational intractability. In this paper, we introduce and analyze a natural relaxation—which we refer to as expected variational inequalities (EVIs)—where the goal is to find a *distribution* that satisfies the VI constraint *in expectation*. By adapting recent techniques from game theory, we show that, unlike VIs, EVIs can be solved in polynomial time under general (nonmonotone) operators. EVIs capture the seminal notion of *correlated equilibria*, but enjoy a greater reach beyond games. We also employ our framework to capture and generalize several existing disparate results, including from settings such as smooth games, and games with coupled constraints or nonconcave utilities.

## 1. Introduction

Variational inequalities (VIs) provide a unifying framework for analyzing a wide range of optimization and equilibrium problems. They have a host of important applications in engineering and economics (Facchinei & Pang, 2003), including identifying stationary points in constrained optimization; computing Nash equilibria in multi-player games (Nash, 1951), such as Cournot's classical model of oligopoly (Cournot, 1838); predicting economic activitycommodity prices and consumer consumption—in a closed, competitive economy (Arrow & Debreu, 1954), which is at the heart of general equilibrium theory; traffic equilibrium
*Equal contribution 1Carnegie Mellon University 2Foundations of Cooperative AI Lab (FOCAL) 3Massachusetts Institute of Technology 4University of Oxford 5Strategy Robot, Inc. 6Strategic Machine, Inc. 7Optimized Markets, Inc. Correspondence to: Brian Hu Zhang <bhzhang@cs.cmu.edu>, Ioannis Anagnostides <ianagnos@cs.cmu.edu>.

problems—estimating the steady-state of a congested network wherein users compete for its resources (Dafermos, 1980); frictional contact problems in mechanical engineering (Capatina, 2014); and pricing options, a foundational problem in financial economics (Black & Scholes, 1973).

Formally, in a general form, a VI can be defined as follows.1 Definition 1.1. Let X be a convex and compact subset of R 
dand F : X → R
da bounded map. The variational inequality (VI) problem asks for a point x ∈ X such that

$$\langle F(\mathbf{x}),\mathbf{x}^{\prime}-\mathbf{x}\rangle\geq0\quad\forall\mathbf{x}^{\prime}\in{\mathcal{X}}.$$
$$(1)^{\frac{1}{2}}$$

′ ∈ X . (1)
For computational purposes, it is common to consider the ϵ*-approximate* VI problem, wherein the right-hand side of
(1) is replaced by −ϵ for some precision parameter ϵ > 0.

Definition 1.1 abstracts the description of F and X . As a concrete example, when F : x *7→ −∇*u(x) is the negative gradient of a differentiable function u : X → R, the solutions to (1) are points that satisfy the first-order optimality conditions for maximizing u (Boyd & Vandenberghe, 2004). Unfortunately, the considerable expressivity of VIs comes at the expense of *intractability*: even when F is linear and ϵ is an absolute constant, identifying an ϵ-approximate VI solution is computationally hard; this follows readily from the intractability of Nash equilibria—under plausible complexity assumptions (Daskalakis et al., 2008; Chen et al., 2009; Rubinstein, 2016). Unconditional, query-complexity lower bounds have also been established (Hirsch et al., 1989; Babichenko, 2016); cf. Milionis et al. (2023) and Hart & Mas-Colell (2003) for other pertinent impossibility results. This bleak realization has shifted the focus of contemporary research primarily to characterizing specific subclasses of VIs that elude those complexity barriers, with the ensuing line of work flourishing in recent years. Some notable examples include the classical *Minty property* (Facchinei & Pang, 2003; Mertikopoulos & Zhou, 2019), as well as certain relaxations thereof (Diakonikolas et al., 2021; Bohm¨ , 2023; Bauschke et al., 2021; Combettes & Pennanen, 2004; Gorbunov et al., 2023; Cai et al., 2024b; Alacaoglu et al., 2023; 1As is standard, a VI throughout this paper refers to the Stampacchia VI *(SVI)* (Kinderlehrer & Stampacchia, 2000).

1 Pethick et al., 2022; Lee & Kim, 2021; Patris & Panageas, 2024; Choudhury et al., 2024; Anagnostides et al., 2024). Despite these important advances, the scope of such results is severely restricted. In this paper, we pursue a different, orthogonal avenue. Instead of restricting the class of problems to achieve computational tractability, we relax the underlying *solution concept*. Our main research question is:
Are there meaningful relaxations of the VI
problem that can always be solved efficiently?

When specialized to games, this question can be seen as part of the research agenda recently outlined by Daskalakis (2022) in his address at the Nobel symposium about equilibrium computation in *nonconcave games*—a major, new frontier in the interface of game theory and optimization.

## 1.1. Our Contribution: The Expected Vi Problem

To make progress on that central question, we introduce a natural relaxation of VIs (in the context of Definition 1.1).

Definition 1.2. Given a set of deviations Φ ⊆ X X , the ϵ-approximate Φ-expected variational inequality (Φ*-EVI)*
problem asks for a distribution µ ∈ ∆(X ) such that

$$\operatorname*{\mathbb{E}}_{\mathbf{x}\sim\mu}\left\langle F(\mathbf{x}),\phi(\mathbf{x})-\mathbf{x}\right\rangle\geq-\epsilon\quad\forall\phi\in\Phi.$$

(The above definition does not specify how X *, F,* Φ, and µ should be represented for computational purposes, but we will be explicit about representation whenever it is relevant.) In words, Definition 1.2 only imposes (approximate) nonnegativity *in expectation* for points x drawn from µ ∈ ∆(X ). It certainly relaxes Definition 1.1: if x satisfies (1),
then the distribution µ that always outputs x is also a Φ-EVI solution. Φ-EVIs are thus no harder than VIs (assuming that solutions exist). However, as we shall see, the primary justification of Φ-EVIs is that they can be easier than VIs.

Definition 1.2 is crucially parameterized by Φ; the larger the set of deviations Φ, the tighter the set of solutions. As will become clear, Definition 1.2 is intimately connected with notions of *correlated equilibrium (CE)* from game theory (e.g., Aumann, 1974). The more permissive case where Φ comprises only constant functions, Φ = ΦCON =
{ϕx : x *∈ X }* where ϕx(x
′) = x for all x
′ ∈ X , is perhaps the most basic relaxation of Definition 1.1; we call the ΦCON-EVI problem simply the *EVI problem*.

Algorithms and complexity for Φ**-EVIs** As it turns out, imposing no constraints on Φ results in an impasse: Φ-EVIs are in general tantamount to regular VIs—thereby being PPAD-hard (Corollaries 3.7 and 3.9). On the other hand, unlike general VIs, one of our key contributions is to show that when Φ contains only linear maps, ΦLIN, Φ-EVIs can be solved in time polynomial in the dimension d and log(1/ϵ) (Theorem 4.1), establishing the promised computational property that separates EVIs from VIs. This result is based on *ellipsoid against hope (*EAH), the seminal algorithm of Papadimitriou & Roughgarden (2008) developed for computing correlated equilibria in multi-player games. (Section 2.1 gives a self-contained overview of EAH.) In doing so, we extend the scope of that algorithm to a much broader class of problems well beyond the realm of game theory. Notably, Theorem 4.1 applies even when X is given implicitly through a membership oracle; this extension makes use of the recent technical approach of Daskalakis et al. (2025), discussed in more detail in Section 4. One limitation of Theorem 4.1 is that it relies on the EAH algorithm, which is slow in practice. We address this by also establishing more scalable algorithms that use convex quadratic optimization (Theorem 4.3) instead of the ellipsoid algorithm. As a byproduct, we obtain the best-known algorithm for linear-swap regret minimization over explicitly represented polytopes, improving on Daskalakis et al.

(2025) by reducing the per-iteration complexity.

In addition to their more favorable computational properties, we further show that Φ-EVIs admit (approximate) solutions under more general conditions than their associated VIsnamely, without F being continuous (Theorem 3.1); Section 3 documents further interesting aspects on existence. Connection to other solution concepts As we have alluded to, Φ-EVIs generalize (Examples 5.1 and 5.2) the seminal concept of a (coarse) correlated equilibrium *a la* ` Aumann (1974) and Moulin & Vial (1978) in finite games, and more generally Φ*-equilibria* (Greenwald & Jafari, 2003; Stoltz & Lugosi, 2007; Gordon et al., 2008) of concave games. What is more surprising is that ΦLIN-EVIs *refine* CEs even in normal-form games; we give illustrative examples, together with an interpretation, in Section 5. We also note that Φ-EVIs can be used even in games with nonconcave utilities (Daskalakis, 2022; Cai et al., 2024a) or noncontinuous gradients (as in nonsmooth optimization), as well as in (pseudo-)games with coupled constraints (cf. Bernasconi et al., 2023 and Appendix A for related work). Further properties As further motivation, we show that for certain structured problems, such as *(quasar-)concave* optimization and *polymatrix* zero-sum games, EVIs essentially coincide with VIs (Propositions 6.1 and 6.3). Finally, in certain applications, one might be interested in a VI solution mainly insofar as it provides guarantees in terms of an underlying objective, such as misclassification error or social welfare. Through that prism, the question is whether performance guarantees for VIs can be translated

| Expected Variational Inequalities             |                                             |                         |
|-----------------------------------------------|---------------------------------------------|-------------------------|
| Result                                        | Description                                 | Reference               |
| Existence of (ϵ-approx.) solutions            | Under Lipschitz cont. for Φ and bounded F   | Theorem 3.1             |
| Complexity with nonlinear Φ                   | PPAD-hardness with linear F and ϵ = Θ(1)    | Corollaries 3.7 and 3.9 |
| Algorithms for linear Φ                       | - poly(d, log(1/ϵ))-time via EAH            | Theorem 4.1             |
| - poly(d, 1/ϵ)-time via Φ-regret minimization | Theorem 4.3                                 |                         |
| Equivalence between VIs-EVIs                  | - Quasar-concave functions (Definition 6.2) | Proposition 6.3         |
| - x 7→ ⟨F(x), x ′ − x⟩ concave for all x ′    | Proposition 6.1                             |                         |
| Performance guarantees for EVIs               | Under smoothness (Definition 7.1)           | Theorem 7.4             |

Table 1. Our main results concerning Φ-EVIs (Definition 1.2).

to EVIs as well. In Section 7, we establish a framework for accomplishing that (Definition 7.1) by extending the celebrated *smoothness* framework of Roughgarden (2015), and provide interesting examples beyond game theory. Taken together, these properties provide compelling justification for Φ-EVIs as a solution concept in place of VIs. Table 1 gathers our main results. (Proofs are in Appendix C.)

## 2. Preliminaries

This section provides some basic notation and background together with an overview of the EAH algorithm. Additional preliminaries, which are not necessary for the main body, are given later in Appendix B.

Notation We use boldface, lowercase letters, such as x and y, to denote vectors in a Euclidean space. Capital, boldface letters, such as A, represent matrices. For x, x
′ ∈
R 
d, we use ⟨x, x
′ 
p
⟩ to denote their inner product. ∥x∥ :=
⟨x, x⟩ is the Euclidean norm of x. Br(x) is the (closed)
Euclidean ball centered at x with radius r > 0. conv(·) represents the convex hull. An *endomorphism* on X is a function mapping X to X .

Returning to Definition 1.2, for computational purposes, we assume throughout that F has an explicit polynomial representation, so that F(x) ∈ R
dcan be evaluated in poly(d)
time. Further, there exists B > 0 such that ∥F(x)∥ ≤ B for all x ∈ X . We will also restrict the support supp(µ) of µ to be poly(d, 1/ϵ), unless stated otherwise. With regard to X , we assume that we have *oracle access*. In particular, we consider the following three types of oracle access.

- *Membership*: given x ∈ R
d, decide whether x ∈ X .

- *Separation*: given x ∈ R
d, decide whether x ∈ X ; if not, return a hyperplane that *separates* x from X .

- *Linear optimization*: Given u ∈ R
d, return a vector in argmaxx∈X ⟨x,u⟩.

In addition, we will assume that *X ⊆ B*R(0) for some R ≤ poly(d), and X contains a ball of radius 1 in its relative interior; this is a standard regularity condition that ensures X is geometrically well-behaved, which can be met by bringing X into isotropic position (Appendix B). Under this assumption, the three oracles listed above are polynomially equivalent (Grotschel et al. ¨ , 1993; 1981). As a result, we may assume that X is given implicitly via a (poly(d)-time)
membership oracle, which suffices for Theorem 4.1. All our positive results with respect to the set of linear endomorphisms ΦLIN readily carry over to affine endomorphisms as well.

## 2.1. Ellipsoid Against Hope

This *ellipsoid against hope (*EAH) algorithm was famously introduced by Papadimitriou & Roughgarden (2008) to compute correlated equilibria in multi-player games. We proceed with an overview of EAH, and in particular a generalized version thereof, crystallized by Farina & Pipis (2024). Consider an arbitrary optimization problem of the form

find $\mu\in\Lambda({\cal X})$ s.t. $\mathbb{E}\left\langle\mathbf{y},G(\mathbf{x})\right\rangle\geq0\,\forall\mathbf{y}\in{\cal Y}$,
(2)
where Y ⊆ R
m, and G : X → R
m is an arbitrary function.

Suppose that we are given an evaluation oracle for G and a separation oracle for Y. Assume further that we are given a *good-enough-response (*GER) oracle, which, given any y ∈ Y, returns x ∈ X such that ⟨y, G(x)⟩ ≥ 0. The upshot is that EAH enables us to solve (2) with just the above tools. Indeed, consider the following problem, which is an ϵ-approximate version of the dual of (2).

find $y\in\mathcal{Y}$ s.t. $\langle y,G(\mathbf{x})\rangle\leq-\epsilon\quad\forall\mathbf{x}\in\mathcal{X}$. (3)
Since a GER oracle exists, (3) is infeasible. What is more, a certificate of infeasibility of (3) yields an ϵ-approximate solution to (2). It thus suffices to run the ellipsoid algorithm on (3) and extract a certificate of infeasibility; in a nutshell, this is what EAH does (cf. Algorithm 1 in Appendix C).

Theorem 2.1 (Generalized form of EAH; Farina & Pipis, 2024). *Given a* GER *oracle and a separation oracle (*SEP)
for Y, EAH *runs in time* poly(*d, m,* log(1/ϵ)) and returns an ϵ-approximate solution to (2).

Theorem 3.5. Suppose that One of our main results (Theorem 4.1) crucially hinges on a strengthening of Theorem 2.1 due to Daskalakis et al. (2025), discussed further in Section 4 and Appendix C.3.

## 3. Existence And Complexity Barriers

Perhaps the most basic question about Φ-EVIs concerns their *totality*—the existence of solutions. If one is willing to tolerate an arbitrarily small imprecision ϵ > 0, we show that solutions exist under very broad conditions.

Theorem 3.1. Suppose that F : X → R
dis measurable and there exists L > 0 such that every ϕ ∈ Φ is L-Lipschitz continuous. Then, for any ϵ > 0, there exists an ϵ-approximate solution to the Φ*-EVI problem.* In particular, our existence proof does not rest on F being continuous. Instead, we consider the continuous function Fb that maps x 7→ Exb∼∆(Bδ(x)∩X ) F(xb) (Claim C.1),
where Bδ(x) is the Euclidean ball centered at x with radius δ = δ(ϵ). It then suffices to invoke Brouwer's fixed-point theorem for the gradient mapping x 7→ ΠX (x − Fb(x)),
where ΠX is the Euclidean projection with respect to X .

Theorem 3.1 implies that a Φ-EVI can have approximate solutions even when the associated VI problem does not.2 Corollary 3.2. *There exists a VI problem that does not* admit approximate solutions when ϵ = Θ(1), but the corresponding ϵ-approximate Φ-EVI is total for any ϵ > 0.

In the proof, we set F to be the *sign function* (Example C.2). By contrast, if one insists on exact solutions, EVIs do not necessarily admit solutions.

Proposition 3.3. When F *is not continuous, there exists an* EVI problem with no solutions. Furthermore, Theorem 3.1 raises the question of whether it is enough to instead assume that every ϕ ∈ Φ is continuous.

Our next result dispels any such hopes.

Theorem 3.4. There are Φ*-EVI instances that do not admit* ϵ*-approximate solutions even when* ϵ = Θ(1), F is piecewise constant, and Φ *contains only continuous functions.*
1. Φ is finite-dimensional, that is, there exists k ∈ N and a kernel map m : X → R
ksuch that every ϕ ∈ Φ can be expressed as Km(x) *for some* K ∈ R
d×k; and 2. every ϕ ∈ Φ *admits a fixed point, that is, a point* X ∋ x = FP(ϕ) *such that* ϕ(x) = x.

Then, the Φ-EVI problem admits an ϵ-approximate solution with support size at most 1 + dk for every ϵ > 0. Notably, this theorem guarantees the existence of solutions with finite support; the proof makes use of the minimax theorem (e.g., Sion, 1958) in conjunction with Caratheodory's ´ theorem on convex hulls (Caratheodory ´ , 1911). Complexity Having established some basic existence properties, we now turn to the complexity of Φ-EVIs. Let us define the VI gap function VIGap(x) :=
− minx′∈X ⟨F(x), x
′ − x⟩, which is nonnegative. If we place no restrictions on Φ, it turns out that Φ-EVIs are tantamount to regular VIs: Proposition 3.6. If Φ contains all measurable functions from X to X , then any solution µ ∈ ∆(X ) to the ϵapproximate Φ*-EVI problem satisfies*

$$\operatorname*{\mathbb{E}}_{x\sim\mu}\operatorname{VIGap}(x)\leq\epsilon.$$
$$(4)$$

VIGap(x) ≤ ϵ. (4)
In proof, it suffices to consider a ϕ that maps x ∈ X to an appropriate point in argminx′∈X ⟨F(x), x
′ − x⟩. When µ must be given explicitly, Proposition 3.6 immediately implies that Φ-EVIs are computationally hard, because (4)
implies that VIGap(x) ≤ ϵ for some x in the support of µ, and such a point can be identified in polynomial time.3 Corollary 3.7. The ϵ-approximate Φ*-EVI problem is* PPAD- hard even when ϵ is an absolute constant and F *is linear.* Coupled with Proposition 3.6, this follows from the hardness result of Rubinstein (2015) concerning Nash equilibria in (multi-player) polymatrix games (for binary-action, graphical games, Deligkas et al., 2023 recently showed that PPAD-hardness persists up to ϵ < 1/2). Corollary 3.7 notwithstanding, it is easy to see that the set of solutions to Φ-EVIs is convex for any Φ ⊆ X X .

Remark 3.8. Let X = X1*×· · ·×X*n, as in an n-player game.

Whether Corollary 3.7 applies under deviations that can be decomposed as ϕ : x 7→ ϕ(x) = (ϕ1(x1)*, . . . , ϕ*n(xn)) is a major open question in the regime where ϵ ≪ 1 (cf. Dagan et al., 2024; Peng & Rubinstein, 2024).

3This argument carries over without restricting the support of µ, by assuming instead access to a sampling oracle from µ: a standard Chernoff bound implies that the empirical distribution (w.r.t. a large enough sample size) approximately satisfies (4).

Viewed differently, a special case of the Φ-EVI problem arises when Φ = {ϕ} and F(x) = x − ϕ(x), for some fixed map ϕ : *X → X* . In this case, the Φ-EVI problem reduces to finding a µ ∈ ∆(X ) such that

$$\mathbb{E}\left\langle F(\mathbf{x}),\phi(\mathbf{x})-\mathbf{x}\right\rangle=-\mathbb{E}\left\|\phi(\mathbf{x})-\mathbf{x}\right\|^{2}\geq-\epsilon.\tag{5}$$

As a result, µ must contain in its support an ϵ-approximate fixed point of ϕ, a problem which is PPAD-hard already for quadratic functions (Zhang et al., 2024a). Corollary 3.9. The ϵ-approximate Φ*-EVI problem is* PPAD- hard even when ϵ is an absolute constant, F is quadratic, and Φ = {ϕ} for a quadratic map ϕ : *X → X* .

It is also worth noting that, unlike Corollary 3.7, Φ in the corollary above contains only continuous functions. It also follows from (5) that for ϵ = 0, Φ-EVIs capture exact fixed points. The complexity class FIXP characterizes such problems (Etessami & Yannakakis, 2007). Corollary 3.10. The Φ*-EVI problem is* FIXP-hard, assuming that supp(µ) ≤ poly(d).

Exponential lower bounds in terms of the number of function evaluations of F also follow from Hirsch et al. (1989). On a positive note, the next section establishes polynomialtime algorithms when Φ contains only *linear* endomorphisms.4

## 4. Efficient Computation With Linear Maps

The hardness results of the previous section highlight the need to restrict the set Φ in order to make meaningful progress. Our main result here establishes a polynomialtime algorithm when Φ contains only linear endomorphisms. Theorem 4.1. If Φ contains only linear endomorphisms, the ϵ-approximate Φ*-EVI problem can be solved in time* poly(d, log(B/ϵ)) *given a membership oracle for* X .

The proof relies on the ellipsoid against hope (EAH), and in particular, a recent generalization by Daskalakis et al. (2025). In a nutshell, the main deficiency in the framework covered earlier in Section 2.1 is that one needs a separation oracle for Y (Theorem 2.1), where Y for us is the set of deviations Φ. Unlike some applications, in which Y has an explicit, polynomial representation (Papadimitriou & Roughgarden, 2008), that assumption needs to be relaxed to account for ΦLIN (Daskalakis et al., 2025, Theorem 3.4).

Daskalakis et al. (2025) address this by considering instead the SEPorGER oracle. As the name suggests, for any 4We do not distinguish between affine and linear maps because we can always set *X ← X × {*1}, in which case affine and linear maps coincide.

y ∈ R
m, it *either* returns a hyperplane separating y from Y,
or a good-enough-response x ∈ X . They showed that Theorem 2.1 can be extended under this weaker oracle (in place of GER and SEP); the formal version is given in Theorem C.4.

In our setting, we consider the feasibility problem

$$\begin{array}{r l}{\operatorname{find}}&{{}\phi\in\Phi_{\mathrm{LIN}}\quad{\mathrm{s.t.}}}\\ {\langle F(\mathbf{x}),\phi(\mathbf{x})-\mathbf{x}\rangle\leq-\epsilon}&{{}\forall\mathbf{x}\in{\mathcal{X}}.}\end{array}$$
$$(6)$$

Equivalently,

$$\begin{array}{r l}{\operatorname{find}}&{{}\mathbf{K}\in\mathbb{R}^{d\times d}\quad{\mathrm{s.t.}}}\\ {\langle F(\boldsymbol{x}),\mathbf{K}\boldsymbol{x}-\boldsymbol{x}\rangle\leq-\epsilon\quad\forall\boldsymbol{x}\in{\mathcal{X}},}\\ {}&{{}\mathbf{K}\boldsymbol{x}\in{\mathcal{X}}\quad\forall\boldsymbol{x}\in{\mathcal{X}}.}\end{array}$$

This program is infeasible since, for any ϕ ∈ ΦLIN, the fixed point x of ϕ makes the left-hand side of the constraint 0. And a certificate of infeasibility is an ϵ-approximate ΦLIN-EVI solution. Thus, it suffices to show how to run the ellipsoid algorithm on (6). By Theorem C.4, it suffices if for any K ∈ R
d×d, we can compute efficiently *either*
- some x ∈ X such that Kx = x (GER), or
- some hyperplane separating K from ΦLIN (SEP).

This is precisely the *semi-separation oracle* solved by Daskalakis et al. (2025, Lemma 4.1), stated below. Lemma 4.2 (Daskalakis et al., 2025). *There is an algorithm* that takes K ∈ R
d×d*, runs in* poly(d) *time, makes* poly(d)
oracle queries to X , and either returns a fixed point X ∋
x = Kx, or a hyperplane separating K *from* ΦLIN. On a separate note, Theorem 4.1 only accounts for approximate solutions. We cannot hope to improve that in the sense that exact solutions might be supported only on irrational points even in concave maximization (cf. Proposition 6.3).

## 4.1. Regret Minimization For Evis On Polytopes

One caveat of Theorem 4.1 is that it relies on the impractical EAH algorithm. To address this limitation, we will show that Φ-EVIs are also amenable to the more scalable approach of *regret minimization*—albeit with an inferior complexity growing as poly(1/ϵ). Specifically, in our context, the regret minimization framework can be applied as follows. At any time t ∈ N, we think of a "learner" selecting a point x
(t) ∈ X , whereupon F(x
(t)) is given as feedback from the "environment," so that the utility at time t reads −⟨x
(t), F(x
(t))⟩. Φ*-regret* is a measure of performance in online learning, defined as

$$\Phi\mbox{-}\mathrm{Reg}^{(T)}:=\operatorname*{max}_{\phi\in\Phi}\sum_{t=1}^{T}\langle F(\mathbf{x}^{(t)}),\phi(\mathbf{x}^{(t)})-\mathbf{x}^{(t)}\rangle.$$

The uniform distribution µ on {x
(1)*, . . . ,* x
(T)} is clearly a Φ-Reg(T)/T-approximate Φ-EVI solution.

In what follows, we will assume that X is a polytope given explicitly by linear constraints, *i.e.*,
Now, consider an ϵ-approximate EVI solution µ of the problem defined by X = {x ∈ R
d: Ax ≤ b},
where A ∈ Qm×dand b ∈ Qm are given as input.

To minimize Φ-regret, we will make use of the template by Gordon et al. (2008), which comprises two components. The first is a fixed-point oracle, which takes as input a function ϕ ∈ ΦLIN and returns a point x ∈ X with x = ϕ(x);
given that ϕ is linear, it can be implemented efficiently via linear programming. The second component is an algorithm for minimizing (external) regret over the set ΦLIN. In Theorem D.1, we devise a polynomial representation for ΦLIN:
Theorem 4.3. For an arbitrary polytope X given by explicit linear constraints, there is an explicit representation of ΦLIN
as a polytope with O(d 2 + m2) *variables and constraints.*
As a consequence, we can instantiate the regret minimizer operating over ΦLIN with projected gradient descent. Corollary 4.4. *There is a deterministic algorithm that* guarantees ΦLIN-Reg(T) ≤ ϵ *after* poly(d, m)/ϵ2rounds, and requires solving a convex quadratic program with O(d 2 + m2) *variables and constraints in each iteration.*
An additional benefit of Corollary 4.4 compared to using EAH is that the former is more suitable in a decentralized environment—for example, in multi-player games (cf. Example 5.1). There, Corollary 4.4 corresponds to each player running their own independent no-regret learning algorithm. Even in this setting, our algorithms actually yield an improvement over the best-known algorithms for minimizing ΦLIN-regret over explicitly-represented polytopes: the previous state of the art, due to Daskalakis et al. (2025), requires running the ellipsoid algorithm on each iteration, which is slower than quadratic programming (Appendix D).

## 5. Game Theory Applications Of Evis

A major motivation for studying Φ-EVIs lies in a strong connection to *(C)CEs* (Aumann, 1974; Moulin & Vial, 1978) in games. Indeed, we begin this section by pointing out that Φ-EVIs capture (C)CEs for specific choices of Φ.

We will mostly consider n-player *concave* games. Here,
each player i ∈ [n] selects a strategy xi ∈ Xi from some convex and compact set Xi, and its utility is given by ui:
(x1*, . . . ,* xn) 7→ R. We assume that ui(xi, x−i) is differentiable and concave in xi for any x−i, and that the gradients
∇xiui(xi, x−i) are bounded. We let X := X1 *× · · · × X*n.
Example 5.1 (CCE). A distribution µ ∈ ∆(X ), is an ϵcoarse correlated equilibrium (CCE) (Moulin & Vial, 1978)
if for any player i ∈ [n],
$$\delta_{i}:=\operatorname*{max}_{\mathbf{x}_{i}^{\prime}\in\mathcal{X}_{i}}\mathbb{E}\;u_{i}(\mathbf{x}_{i}^{\prime},\mathbf{x}_{-i})-\mathbb{E}\;u_{i}(\mathbf{x})\leq\epsilon.$$
$\blacksquare$
$F:=(-\nabla_{\bf a_{1}}u_{1}({\bf a}),\ldots,-\nabla_{\bf a_{n}}u_{n}({\bf a}))$.  
Such µ satisfies, by concavity, Pn i=1 δi ≤ ϵ; it is not necessarily an ϵ-approximate CCE since it is possible that for some i ∈ [n], all deviations strictly decrease i's utility (so that δiin (7) is negative)—µ is technically an *average* CCE in the parlance of Nadav & Roughgarden (2010). To capture CCE via Φ-EVIs, one can instead consider a richer set of deviations of the form (x1, . . . , xn) 7→ (x1*, . . . ,* x
′i*, . . . ,* xn)
for all i ∈ [n] and x
′i ∈ Xi.

A canonical example of the above formalism is a normalform game, in which each constraint set Xiis the probability simplex ∆(Ai) over a finite set of *actions* Ai, and each utility uiis a multilinear function.

Example 5.2 (LCE). A distribution µ ∈ ∆(X ) is an ϵlinear correlated equilibrium (LCE) if for any i ∈ [n],

$$\operatorname*{max}_{\phi_{i}\in\Phi_{i}}\mathbb{E}\;u_{i}(\phi_{i}(\mathbf{x}_{i}),\mathbf{x}_{-i})-\mathbb{E}\;u_{i}(\mathbf{x})\leq\epsilon,$$

where Φi contains all linear functions from Xito Xi. To capture LCE via Φ-EVIs, it suffices to consider deviations of the form (x1, . . . , xn) 7→ (x1, . . . , ϕi(xi)*, . . . ,* xn) for all i ∈ [n] and ϕi ∈ Φi.

For normal-form games, LCEs amount to the usual notion of CEs (Aumann, 1974). LCEs were introduced in the context of extensive-form games (Farina & Pipis, 2023; 2024). Refining correlated equilibria In fact, and more surprisingly, ΦLIN-EVI solutions can be a strict subset of LCEs.5 This separation can already be appreciated in the setting of normal-form games, and manifests itself in at least two distinct ways. First, there exist games for which a CE need not be a solution to the ΦLIN-EVI. In this sense, ΦLIN-EVIs yield a computationally tractable superset of Nash equilibria that is tighter than CEs. Second, computation suggests that the set of solutions of the ΦLIN-EVI for the game need not be a polyhedron, unlike the set of CEs. We provide a graphical depiction of this phenomenon in Figure 1. The figure depicts the set of ΦLIN-EVI solutions to a simple
"Bach or Stravinsky" game, in which the players receive payoffs (3, 2) if they both pick Bach, (2, 3) if they both pick Stravinsky, and (0, 0) otherwise.

5The example of S¸eref Ahunbay (2025, Example 1) already implies that certain CEs can be excluded from the set of ΦLIN-EVIs, which, incidentally, could have implications for last-iterate convergence in some classes of games, as discussed by that author. Our example in Figure 1 goes much further, revealing that ΦLIN-EVIs can yield significantly different utilities for each player compared to CEs.

Interpretation The reason for this separation is that, for a map ϕ : *X → X* , each player's mapped strategy ϕ(x)i can also depend (linearly) on *other players' strategies* x−i.

Indeed, the EVI formulation of a game does not take into account the identities of the players. For this reason, we will call the set of ΦLIN-EVI solutions in a concave game anonymous linear correlated equilibria, or *ALCE* for short. We give two game-theoretic interpretations of ALCEs. First, the ALCEs of a game Γ are the *symmetric* LCEs of the "symmetrized" game in which the players are randomly shuffled before the game begins. That is, consider the nplayer game Γ
sym defined as follows. Each player's strategy set is X . For strategy profile (x 1*, . . . ,* x n) ∈ X n, the utility to player i is given by

$$u_{i}^{\mathrm{sym}}(\mathbf{x}^{1},\ldots,\mathbf{x}^{n})={\frac{1}{n!}}\sum_{\sigma\in\mathfrak{G}_{n}}u_{\sigma(i)}(\mathbf{x}_{1}^{\sigma^{-1}(1)},\ldots,\mathbf{x}_{n}^{\sigma^{-1}(n)}),$$

where Gn is the set of permutations σ : [n] → [n]. The following result then follows almost by definition.

Proposition 5.3. For a given distribution µ ∈ ∆(X ), define the distribution µ n ∈ ∆(X
n) by sampling x ∼ µ and outputting (x, . . . , x) ∈ X n. Then, µ is a ALCE of Γ *if and* only if µ n *is an LCE of* Γ
sym.

Second, for normal-form games, the ALCEs are the distributions µ ∈ ∆(X ) such that no player i has a profitable deviation of the following form. The correlation device first samples x ∼ µ, and samples recommendations aj ∼ xj for each player j. Then, the player selects another player j (possibly j = i) whose recommendation it wishes to see. The player then observes a sample a
′j ∼ xj that is independently sampled from aj .

6 Finally, the player chooses an action a
∗
i ∈ Ai, and each player j gets reward uj (a
∗
i, a−i).

Thus, players are allowed (modulo the independent sampling) to spy on each others' recommendations.

Further discussion about ALCEs and formal proofs of the claims in this section are deferred to Appendix F. Coupled constraints Continuing from Examples 5.1 and 5.2, we observe that (ΦLIN-)EVIs can be used even in "pseudo-games," in which X does not necessarily decompose into X1 *× · · · × X*n; this means that xi ∈ Xi(x−i). As we discuss in Appendix A, most prior work in such settings has focused on generalized Nash equilibria, with the exception of Bernasconi et al. (2023). (ΦLIN-)EVIs induce an interesting notion of LCE/CCE in pseudo-games, albeit not directly comparable to the one put forward by Bernasconi et al. (2023). It is worth noting that Bernasconi et al. (2023)
left open whether efficient algorithms for computing their notion of (coarse) correlated equilibria exist.

6This independence is crucial: without it, µ would actually need to be a distribution over pure Nash equilibria!

0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 Probability of Player 1's first action (Bach)
Pr ob abili ty o f Play er 2' s fir st ac tion (
B
ach)
ΦLIN-EVI solutions Correlated equil. VI sol. (Nash eq.)
CE
Definition 5.4. Given an n-player pseudo-game with concave, differentiable utilities and joint constraints X , a distribution µ ∈ ∆(X ) is an ϵ*-ALCE* if

$$\operatorname*{max}_{\phi\in\Phi_{\mathrm{LN}}}\mathbb{E}\sum_{i=1}^{n}u_{i}(\phi(\mathbf{x})_{i},\mathbf{x}_{-i})-\sum_{i=1}^{n}u_{i}(\mathbf{x})\leq\epsilon.$$

By virtue of our main result (Theorem 4.1), such an equilibrium can be computed in polynomial time. Noncontinuous gradients In fact, our results do not rest on the usual assumption that each player's gradient is a continuous function, thereby significantly expanding the scope of prior known results even in games. For example, we refer to Dasgupta & Maskin (1986); Bichler et al. (2021); Martin & Sandholm (2024) for pointers to some applications. Nonconcave games Last but not least, Φ-EVIs give rise to a notion of *local* Φ-equilibrium (Definition G.1) in nonconcave games. It turns out that this captures recent results by Cai et al. (2024a) and S¸eref Ahunbay (2025), but our framework has certain important advantages. First, we give a poly(d, log(1/ϵ))-time algorithm (Theorem 4.1), while theirs scale polynomially in 1/ϵ. Second, our results do not assume continuity of the gradients. And finally, our algorithms are polynomial even when Φ contains all linear endomorphisms (Theorem 4.1). Appendix G elaborates further on those points.

## 6. Problems Where Evis Coincide With Vis

We saw earlier, in Proposition 3.6, that when Φ comprises all functions from X to X , the Φ-EVI problem is tantamount to the associated VI problem. However, if one restricts the functions contained in Φ, are there still structured VIs where we retain this equivalence? In this section, we consider certain structured VIs, and show their equivalence to the corresponding EVIs (that is, ΦCON-EVIs). Unlike general VIs, the ones we examine below are tractable.

## 6.1. Polymatrix Zero-Sum Games And Beyond

The first important class of VIs we consider is described by a condition given below. Proposition 6.1. *Suppose that for any* x
′ ∈ X *, the function* g : x *7→ ⟨*F(x), x
′ − x⟩ *is concave. Then, if* µ ∈ ∆(X )
is an ϵ-approximate solution to the EVI, Ex∼µ x is an ϵapproximate solution to the VI. The proof follows directly from Jensen's inequality. The precondition of Proposition 6.1 is satisfied, *e.g.*, when:
(i) ⟨F(x), x⟩ = 0 for all x ∈ X , and (ii) F is a linear map.

In the context of n-player games, the first condition amounts to the zero-sum property: Pn i=1 ui(x) = 0 for all x. Of course, this property is not enough to enable efficient computation of Nash equilibria, for every two-player (general-sum) game can be converted into a 3-player zero-sum game. This is where the second condition comes into play: F is a linear map—that is, each player's gradient must be linear in the joint strategy. Those two conditions are satisfied in polymatrix zero-sum games (Cai et al., 2016); in such games, the conclusion of Proposition 6.1 is a well-known fact.

## 6.2. Quasar-Concave Functions

We next consider the problem of maximizing a (single) function that satisfies *quasar-concavity*—a natural generalization of concavity that has received significant interest (Hardt et al., 2018; Fu et al., 2023; Hinder et al., 2020; Gower et al.,
2021; Guminov et al., 2023; Caramanis et al., 2024).7 Definition 6.2 (Quasar-concavity). Let γ ∈ (0, 1] and x
⋆ ∈
X be a maximizer of a differentiable function u : X → R.

We say that u is γ*-quasar-concave* with respect to x
⋆if

$$u(\mathbf{x}^{\star})\leq u(\mathbf{x})+\frac{1}{\gamma}\langle\nabla u(\mathbf{x}),\mathbf{x}^{\star}-\mathbf{x}\rangle\quad\forall\mathbf{x}\in\mathcal{X}.\tag{8}$$

In particular, in the special case where γ = 1, (8) is equivalent to *star-concavity* (Nesterov & Polyak, 2006). If in addition (8) holds for all x
⋆ ∈ X (not merely w.r.t. a global maximizer), it captures the usual notion of concavity.

7Prior literature mostly uses the term *quasar-convexity*, which is equivalent to quasar-concavity for the opposite function −u.

Any reasonable solution concept for such problems should place all mass on global maxima; EVIs pass this litmus test:
Proposition 6.3. Let F = −∇u for a γ-quasar-concave and differentiable function u : X → R. Then, for any solution µ ∈ ∆(X ) *to the EVI problem,*

$$\mathbb{E}_{x\sim\mu}\,u(x)\geq\operatorname*{max}_{x\in{\mathcal{X}}}u(x).$$

Thus, Px∼µ[u(x
⋆) = u(x)] = 1*, for* x
⋆ ∈ argmaxx u(x).

Indeed, by Definition 6.2, 0 ≤ Ex∼µ⟨∇u(x), x − x
⋆⟩ ≤
γ Ex∼µ[u(x) − u(x
⋆)] for any EVI solution µ ∈ ∆(X ).

Thus, under quasar-concavity, VIs basically reduce to EVIs.

## 7. Performance Guarantees For Evis

In many settings, a VI solution is used as a proxy to approximately maximize some underlying objective function; machine learning offers many such applications. The question is whether performance guarantees pertaining to VIs can be extended—potentially with some small degradationto EVIs as well. The purpose of this section is to provide a framework for achieving that based on the following notion. Definition 7.1. An EVI problem is (λ, ν)*-smooth*, for λ >
0*, ν >* −1, w.r.t. W : X → R and x
⋆ ∈ argmaxx W(x) if

$$\langle F(\mathbf{x}),\mathbf{x^{\star}}-\mathbf{x}\rangle\leq-\lambda W(\mathbf{x^{\star}})+(\nu+1)W(\mathbf{x})\quad\forall\mathbf{x}\in{\mathcal{X}}.$$

Example 7.2. When the underlying problem corresponds to a multi-player game and W is the (utilitarian) social welfare, Definition 7.1 coincides with the celebrated notion of smoothness *a la* ` Roughgarden (2015); this is a consequence of multilinearity, which implies that W(x) = −⟨x, F(x)⟩ and ⟨x
⋆, F(x)⟩ = −Pn i=1 ui(x
⋆
i, x−i) for all x ∈ X . We also refer to the recent treatment of smoothness by S¸eref Ahunbay (2025) in the context of nonconcave games, which builds on the primal-dual framework of Nadav & Roughgarden (2010). Definition 7.1 is an extension of the more general notion of
"local smoothness," introduced by Roughgarden & Schoppmann (2015) in the context of splittable congestion games. However, it goes beyond games. Indeed, the following definition we introduce generalizes Definition 6.2, making a new connection between smoothness and quasar-concavity.

Definition 7.3 (Extension of quasar-concavity). Let x
⋆ ∈
X be a maximizer of a differentiable function u : X → R.

We say that u is (λ, ν)*-smooth* with respect to x
⋆if

$$\langle\nabla u(\mathbf{x}),\mathbf{x}^{\star}-\mathbf{x}\rangle\geq\lambda u(\mathbf{x}^{\star})-(\nu+1)u(\mathbf{x})\quad\forall\mathbf{x}\in{\mathcal{X}}.$$

In particular, when λ := γ and ν := γ − 1, the above definition captures γ-quasar-concavity. In Appendix E, we provide an example of a polynomial that satisfies Definition 7.3 without being quasar-concave. Now, the key property of Definition 7.1 is that any EVI solution approximates the underlying objective—by a factor of ρ := λ/1+ν.

Theorem 7.4. Let µ ∈ ∆(X ) be an ϵ-approximate solution to a (λ, ν)-smooth EVI problem w.r.t. W : X → R*. Then,*

$$\operatorname*{\mathbb{E}}_{\mathbf{x}\sim\mu}W(\mathbf{x})\geq{\frac{\lambda}{1+\nu}}\operatorname*{max}_{\mathbf{x}\in{\mathcal{X}}}W(\mathbf{x})-{\frac{\epsilon}{1+\nu}}.$$

The proof follows directly from Definition 7.1, using that Ex∼µ⟨F(x), x
⋆ − x*⟩ ≥ −*ϵ and linearity of expectation.

## 8. Conclusions And Future Research

In summary, our main contribution was to introduce and examine a natural relaxation of VIs, which we refer to as expected VIs. Unlike VIs, which are marred by computational intractability, we showed that EVIs can be solved efficiently under minimal assumptions. We also uncovered many other intriguing properties of EVIs (cf. Table 1). There are many promising avenues for future work. VIs enjoy a great reach in a wide range of applications, some of which were discussed earlier in our introduction. It would be interesting to explore in more detail how EVIs fare in such settings compared to VIs. In particular, given that EVIs relax VIs, in addition to their more favorable computational properties, it is likely that they unlock new, more desirable solutions not present under VIs. For example, it is well known (e.g., Ashlagi et al., 2005) that CEs can achieve better welfare than Nash equilibria in games. In light of the prominence of correlated equilibria in the rich setting of multi-player games, we anticipate EVIs to solidify their place also in other application areas beyond the realm of game theory.

## Impact Statement

This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## Acknowledgments

T.S. is supported by the Vannevar Bush Faculty Fellowship ONR N00014-23-1-2876, National Science Foundation grants RI-2312342 and RI-1901403, ARO award W911NF2210266, and NIH award A240108S001. B.H.Z. is supported by the CMU Computer Science Department Hans Berliner PhD Student Fellowship. E.T, R.E.B., and V.C. thank the Cooperative AI Foundation, Polaris Ventures (formerly the Center for Emerging Risk Research) and Jaan Tallinn's donor-advised fund at Founders Pledge for financial support. E.T. and R.E.B. are also supported in part by the Cooperative AI PhD Fellowship. G.F is supported by the National Science Foundation grant CCF-2443068. We are grateful to Mete S¸ eref Ahunbay for his helpful feedback. We also thank Andrea Celli and Martino Bernasconi for discussions regarding Φ-equilibria in games with coupled constraints.

## References

Alacaoglu, A., Bohm, A., and Malitsky, Y. Beyond the ¨
golden ratio for variational inequality algorithms. Journal of Machine Learning Research, 24:172:1–172:33, 2023.

Aliprantis, C. D. and Border, K. C. *Infinite Dimensional* Analysis: a Hitchhiker's Guide. Springer, 2006.

Anagnostides, I., Panageas, I., Farina, G., and Sandholm, T.

Optimistic policy gradient in multi-player Markov games with a single controller: Convergence beyond the Minty property. In *Conference on Artificial Intelligence (AAAI)*, 2024.

Ardagna, D., Panicucci, B., and Passacantando, M. Generalized Nash equilibria for the service provisioning problem in cloud systems. IEEE Transactions on Services Computing, 6(4):429–442, 2012.

Arrow, K. J. and Debreu, G. Existence of an equilibrium for a competitive economy. *Econometrica*, 22(3):265–290, 1954.

Ashlagi, I., Monderer, D., and Tennenholtz, M. On the value of correlation. In Conference in Uncertainty in Artificial Intelligence (UAI), 2005.

Aumann, R. Subjectivity and correlation in randomized strategies. *Journal of Mathematical Economics*, 1:67–96, 1974.

Babichenko, Y. Query complexity of approximate Nash equilibria. *Journal of the ACM*, 63(4):36:1–36:24, 2016.

Bauschke, H. H., Moursi, W. M., and Wang, X. Generalized monotone operators and their averaged resolvents. Mathematical Programming, 189(1):55–74, 2021.

Bernasconi, M., Castiglioni, M., Marchesi, A., Trovo, F., `
and Gatti, N. Constrained Phi-equilibria. In International Conference on Machine Learning (ICML), 2023.

Bernasconi, M., Castiglioni, M., Celli, A., and Farina, G.

On the role of constraints in the complexity of min-max optimization. *arXiv:2411.03248*, 2024.

Bichler, M., Fichtl, M., Heidekruger, S., Kohring, N., and ¨
Sutterer, P. Learning equilibria in symmetric auction games using artificial neural networks. *Nature Machine* Intelligence, 3(8):687–695, 2021.

Billingsley, P. *Convergence of Probability Measures*. Wiley Series in Probability and Statistics: Probability and Statistics. John Wiley & Sons, second edition, 1999.

Black, F. and Scholes, M. The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3):
637–654, 1973.

Bohm, A. Solving nonconvex-nonconcave min-max prob- ¨
lems exhibiting weak Minty solutions. *Transactions on* Machine Learning Research, 2023.

Boyd, S. and Vandenberghe, L. *Convex Optimization*. Cambridge University Press, 2004.

Cai, Y., Candogan, O., Daskalakis, C., and Papadimitriou, C. H. Zero-sum polymatrix games: A generalization of minmax. *Mathematics of Operations Research*, 41(2): 648–655, 2016.

Cai, Y., Daskalakis, C., Luo, H., Wei, C., and Zheng, W. On tractable Φ-equilibria in non-concave games. In Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS), 2024a.

Cai, Y., Oikonomou, A., and Zheng, W. Accelerated algorithms for constrained nonconvex-nonconcave min-max optimization and comonotone inclusion. In International Conference on Machine Learning (ICML), 2024b.

Capatina, A. Variational inequalities and frictional contact problems, volume 31. Springer, 2014.

Caramanis, C., Fotakis, D., Kalavasis, A., Kontonis, V., and Tzamos, C. Optimizing solution-samplers for combinatorial problems: the landscape of policy-gradient methods. In Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS), 2024.

Caratheodory, C. Uber den Variabilet ´ atsbereich der ¨
Fourier'schen Konstanten von positiven harmonischen Funktionen. Rendiconti del Circolo Matematico de Palermo, 32:193–217, 1911.

Chen, X., Deng, X., and Teng, S.-H. Settling the complexity of computing two-player Nash equilibria. Journal of the ACM, 2009.

Choudhury, S., Gorbunov, E., and Loizou, N. Singlecall stochastic extragradient methods for structured nonmonotone variational inequalities: Improved analysis under weaker conditions. In Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS), 2024.

Combettes, P. L. and Pennanen, T. Proximal methods for cohypomonotone operators. SIAM Journal on Control and Optimization, 43(2):731–742, 2004.

Cournot, A. A. Recherches sur les principes mathematiques ´
de la theorie des richesses (Researches into the Mathe- ´ matical Principles of the Theory of Wealth). Hachette, Paris, 1838.

Dafermos, S. Traffic equilibrium and variational inequalities.

Transportation Science, 14(1):42–54, 1980.

Dagan, Y., Daskalakis, C., Fishelson, M., and Golowich, N.

From external to swap regret 2.0: An efficient reduction for large action spaces. In Proceedings of the Annual Symposium on Theory of Computing (STOC), 2024.

Dasgupta, P. and Maskin, E. The existence of equilibrium in discontinuous economic games 1: Theory. Review of Economic Studies, 53:1–26, 1986.

Daskalakis, C. Non-concave games: A challenge for game theory's next 100 years, 2022.

Daskalakis, C., Goldberg, P., and Papadimitriou, C. The complexity of computing a Nash equilibrium. SIAM Journal on Computing, 2008.

Daskalakis, C., Skoulakis, S., and Zampetakis, M. The complexity of constrained min-max optimization. In Proceedings of the Annual Symposium on Theory of Computing
(STOC), 2021.

Daskalakis, C., Farina, G., Golowich, N., Sandholm, T., and Zhang, B. H. A lower bound on swap regret in extensiveform games. *arXiv:2406.13116*, 2024.

Daskalakis, C., Farina, G., Fishelson, M., Pipis, C., and Schneider, J. Efficient learning and computation of linear correlated equilibrium in general convex games. In Proceedings of the Annual Symposium on Theory of Computing (STOC), 2025.

Davis, D., Drusvyatskiy, D., Lee, Y. T., Padmanabhan, S.,
and Ye, G. A gradient sampling method with complexity guarantees for Lipschitz functions in high and low dimensions. In *Proceedings of the Annual Conference on* Neural Information Processing Systems (NeurIPS), 2022.

Deligkas, A., Fearnley, J., Hollender, A., and Melissourgos, T. Tight inapproximability for graphical games. In Conference on Artificial Intelligence (AAAI), 2023.

Diakonikolas, J., Daskalakis, C., and Jordan, M. I. Efficient methods for structured nonconvex-nonconcave min-max optimization. In *International Conference on Artificial* Intelligence and Statistics (AISTATS), 2021.

Domingo-Enrich, C., Jelassi, S., Mensch, A., Rotskoff, G. M., and Bruna, J. A mean-field analysis of twoplayer zero-sum games. In Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS), 2020.

Etessami, K. and Yannakakis, M. On the complexity of Nash equilibria and other fixed points. In *Proceedings* of the Annual Symposium on Foundations of Computer Science (FOCS), 2007.

Facchinei, F. and Kanzow, C. Generalized Nash equilibrium problems. *Annals of Operations Research*, 175(1):177–
211, 2010.

Facchinei, F. and Pang, J.-S. Finite-dimensional variational inequalities and complementarity problems. Springer, 2003.

Facchinei, F., Fischer, A., and Piccialli, V. Generalized Nash equilibrium problems and Newton methods. Mathematical Programming, 117(1):163–194, 2009.

Farina, G. and Pipis, C. Polynomial-time linear-swap regret minimization in imperfect-information sequential games. In Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS), 2023.

Farina, G. and Pipis, C. Polynomial-time computation of exact Phi-equilibria in polyhedral games. In Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS), 2024.

Farina, G., Celli, A., Marchesi, A., and Gatti, N. Simple uncoupled no-regret learning dynamics for extensive-form correlated equilibrium. *Journal of the ACM*, 69(6):41:1– 41:41, 2022.

Fischer, A., Herrich, M., and Schonefeld, K. Generalized ¨
Nash equilibrium problems-recent advances and challenges. *Pesquisa Operacional*, 34(3):521–558, 2014.

Fu, Q., Xu, D., and Wilson, A. C. Accelerated stochastic optimization methods under quasar-convexity. In International Conference on Machine Learning (ICML), 2023.

Fujii, K. Bayes correlated equilibria and no-regret dynamics.

arXiv:2304.05005, 2023.

Goktas, D. and Greenwald, A. Exploitability minimization in games and beyond. In Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS), 2022.

Goldstein, A. Optimization of Lipschitz continuous functions. *Mathematical Programming*, 13:14–22, 1977.

Gorbunov, E., Taylor, A. B., Horvath, S., and Gidel, G. Con- ´
vergence of proximal point and extragradient-based methods beyond monotonicity: the case of negative comonotonicity. In International Conference on Machine Learning (ICML), 2023.

Gordon, G. J., Greenwald, A., and Marks, C. No-regret learning in convex games. In International Conference on Machine Learning (ICML), 2008.

Gower, R. M., Sebbouh, O., and Loizou, N. SGD for structured nonconvex functions: Learning rates, minibatching and interpolation. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2021.

Greenwald, A. and Jafari, A. A general class of no-regret learning algorithms and game-theoretic equilibria. In Conference on Learning Theory (COLT), 2003.

Grotschel, M., Lov ¨ asz, L., and Schrijver, A. The ellipsoid ´
method and its consequences in combinatorial optimization. *Combinatorica*, 1:169–197, 1981.

Grotschel, M., Lov ¨ asz, L., and Schrijver, A. ´ *Geometric* algorithms and combinatorial optimization. Springer, 1993.

Guminov, S., Gasnikov, A. V., and Kuruzov, I. A. Accelerated methods for weakly-quasi-convex optimization problems. *Computational Management Science*, 20(1): 36, 2023.

Hardt, M., Ma, T., and Recht, B. Gradient descent learns linear dynamical systems. Journal of Machine Learning Research, 19(1):1025–1068, 2018.

Hart, S. and Mas-Colell, A. Uncoupled dynamics do not lead to Nash equilibrium. *American Economic Review*, 93:1830–1836, 2003.

Hazan, E. Introduction to online convex optimization.

Foundations and Trends in Optimization, 2(3-4):157–325, 2016.

Hinder, O., Sidford, A., and Sohoni, N. S. Near-optimal methods for minimizing star-convex functions and beyond. In *Conference on Learning Theory (COLT)*, 2020.

Hirsch, M. D., Papadimitriou, C. H., and Vavasis, S. A.

Exponential lower bounds for finding Brouwer fix points. Journal of Complexity, 5(4):379–416, 1989.

Hsieh, Y., Liu, C., and Cevher, V. Finding mixed Nash equilibria of generative adversarial networks. In International Conference on Machine Learning (ICML), 2019.

Huang, W. and von Stengel, B. Computing an extensiveform correlated equilibrium in polynomial time. In International Workshop On Internet And Network Economics (WINE), 2008.

Jordan, M. I., Kornowski, G., Lin, T., Shamir, O., and Zampetakis, M. Deterministic nonsmooth nonconvex optimization. In *Conference on Learning Theory (COLT)*,
2023a.

Jordan, M. I., Lin, T., and Zampetakis, M. First-order algorithms for nonlinear generalized Nash equilibrium problems. *Journal of Machine Learning Research*, 24: 38:1–38:46, 2023b.

Kapron, B. M. and Samieefar, K. The computational complexity of variational inequalities and applications in game theory. *arXiv:2411.04392*, 2024.

Kinderlehrer, D. and Stampacchia, G. An introduction to variational inequalities and their applications. SIAM,
2000.

Kuhn, H. W. Extensive games and the problem of information. In *Contributions to the Theory of Games*, volume 2 of *Annals of Mathematics Studies, 28*, pp. 193–216.

Princeton University Press, 1953.

Lee, S. and Kim, D. Fast extra gradient methods for smooth structured nonconvex-nonconcave minimax problems. In Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS), 2021.

Lovasz, L. and Vempala, S. S. Simulated annealing in con- ´
vex bodies and an O
*(n 4) volume algorithm. J. Comput.

Syst. Sci., 72(2):392–417, 2006.

Martin, C. and Sandholm, T. Joint-perturbation simultaneous pseudo-gradient. *arXiv:2408.09306*, 2024.

Mertikopoulos, P. and Zhou, Z. Learning in games with continuous action sets and unknown payoff functions. Mathematical Programming, 173(1-2):465–507, 2019.

Milionis, J., Papadimitriou, C., Piliouras, G., and Spendlove, K. An impossibility theorem in game dynamics. Proceedings of the National Academy of Sciences, 120(41), 2023.

Morrill, D., D'Orazio, R., Lanctot, M., Wright, J. R., Bowling, M., and Greenwald, A. R. Efficient deviation types and learning for hindsight rationality in extensive-form games. In *International Conference on Machine Learning* (ICML), 2021a.

Morrill, D., D'Orazio, R., Sarfati, R., Lanctot, M., Wright, J. R., Greenwald, A. R., and Bowling, M. Hindsight and sequential rationality of correlated play. In *Conference* on Artificial Intelligence (AAAI), 2021b.

Moulin, H. and Vial, J.-P. Strategically zero-sum games:
The class of games whose completely mixed equilibria cannot be improved upon. *International Journal of Game* Theory, 7(3-4):201–221, 1978.

Nadav, U. and Roughgarden, T. The limits of smoothness:
A primal-dual framework for price of anarchy bounds. In International Workshop On Internet And Network Economics (WINE), 2010.

Nash, J. Non-cooperative games. *Annals of Mathematics*,
54:289–295, 1951.

Nesterov, Y. and Polyak, B. T. Cubic regularization of newton method and its global performance. Mathematical programming, 108(1):177–205, 2006.

Papadimitriou, C. H. and Roughgarden, T. Computing correlated equilibria in multi-player games. Journal of the ACM, 55(3):14:1–14:29, 2008.

Patris, N. and Panageas, I. Learning Nash equilibria in rank-1 games. In *International Conference on Learning* Representations (ICLR), 2024.

Peng, B. and Rubinstein, A. Fast swap regret minimization and applications to approximate correlated equilibria.

In Proceedings of the Annual Symposium on Theory of Computing (STOC), 2024.

Pethick, T., Latafat, P., Patrinos, P., Fercoq, O., and Cevher, V. Escaping limit cycles: Global convergence for constrained nonconvex-nonconcave minimax problems. In International Conference on Learning Representations (ICLR), 2022.

Roughgarden, T. Intrinsic robustness of the price of anarchy.

Journal of the ACM, 62(5):32:1–32:42, 2015.

Roughgarden, T. and Schoppmann, F. Local smoothness and the price of anarchy in splittable congestion games. Journal of Economic Theory, 156:317–342, 2015.

Roughgarden, T., Syrgkanis, V., and Tardos, E. The price ´
of anarchy in auctions. Journal of Artificial Intelligence Research, 59:59–101, 2017.

Rubinstein, A. Inapproximability of Nash equilibrium. In Proceedings of the Annual Symposium on Theory of Computing (STOC), 2015.

Rubinstein, A. Settling the complexity of computing approximate two-player Nash equilibria. In Proceedings of the Annual Symposium on Foundations of Computer Science (FOCS), 2016.

Sion, M. On general minimax theorems. *Pacific Journal of* Mathematics, 8(1):171 - 176, 1958.

Stoltz, G. and Lugosi, G. Learning correlated equilibria in games with compact sets of strategies. Games and Economic Behavior, 59(1):187–208, 2007.

Tatarenko, T. and Kamgarpour, M. Learning generalized Nash equilibria in a class of convex games. IEEE Transactions on Automatic Control, 64(4):1426–1439, 2018.

Tian, L., Zhou, K., and So, A. M. On the finite-time complexity and practical computation of approximate stationarity concepts of lipschitz functions. In *International* Conference on Machine Learning (ICML), 2022.

Villani, C. *Optimal transport: old and new*, volume 338.

Springer, 2009.

Zhang, B. H., Anagnostides, I., Farina, G., and Sandholm, T. Efficient Φ-regret minimization with low-degree swap deviations in extensive-form games. In Proceedings of the Annual Conference on Neural Information Processing Systems (NeurIPS), 2024a.

Zhang, B. H., Farina, G., and Sandholm, T. Mediator interpretation and faster learning algorithms for linear correlated equilibria in general extensive-form games. In International Conference on Learning Representations
(ICLR), 2024b.

Zhang, B. H., Anagnostides, I., Tewolde, E., Berker, R. E.,
Farina, G., Conitzer, V., and Sandholm, T. Learning and computation of Φ-equilibria at the frontier of tractability.

In *Proceedings of the ACM Conference on Economics* and Computation (EC), 2025.

Zhang, J., Lin, H., Jegelka, S., Sra, S., and Jadbabaie, A.

Complexity of finding stationary points of nonconvex nonsmooth functions. In International Conference on Machine Learning (ICML), 2020.

S¸ eref Ahunbay, M. First-order (coarse) correlated equilibria in non-concave games. *arXiv:2403.18174*, 2025.

## A. Further Related Work

We have seen that Definition 1.2 is strongly connected with the notion of Φ-equilibria. In extensive-form games, the question of characterizing the set of deviations Φ that enables efficient learning—within the no-regret framework—and computation has attracted considerable attention. In particular, efficient algorithms have been established for extensive-form correlated equilibria (EFCEs) (Huang & von Stengel, 2008; Farina et al., 2022; Morrill et al., 2021a;b), and more broadly, when Φ contains solely *linear* functions (Farina & Pipis, 2024; 2023)—corresponding to *linear correlated equilibria (LCEs)*.

Recently, Daskalakis et al. (2025) strengthened those results beyond extensive-form games whenever there is a separation oracle for the constraint set; we rely on their approach for some of our positive results. Moreover, Zhang et al. (2024a; 2025)
established certain positive results even when Φ contains low-degree polynomials; by contrast, in our setting, Φ-EVIs are hard even when Φ contains only quadratic polynomials (Corollary 3.9).

Besides encompassing correlated equilibria in games, wherein the constraint set X can be decomposed as a Cartesian product over the constraint set of each player (reflecting the fact that players select strategies independently), our positive results pertaining to Definition 1.2 do not rest on such assumptions and apply even in the presence of joint constraint sets. There is a long history in game theory, optimization, and economics pertaining to such settings—sometimes referred to as
"pseudo-games" in the literature (Goktas & Greenwald, 2022; Arrow & Debreu, 1954; Facchinei & Kanzow, 2010; Fischer et al., 2014; Facchinei et al., 2009; Ardagna et al., 2012; Tatarenko & Kamgarpour, 2018; Jordan et al., 2023b; Daskalakis et al., 2021). The notion of *generalized* Nash equilibria—the natural counterpart of Nash's solution concept in the presence of coupled constraints—has dominated that line of work, with the recent paper of Bernasconi et al. (2023) being the notable exception. Responding to the call of Daskalakis (2022), Cai et al. (2024a) and S¸eref Ahunbay (2025) recently proposed several tractable solution concepts in games with nonconcave utilities. In particular, when specialized to games, Φ-EVIs are closely related to a notion proposed by S¸eref Ahunbay (2025, Definition 6). One of our key results, Theorem 4.1, establishes a poly(d, log(1/ϵ))-time algorithm, while the algorithms of Cai et al. (2024b) and S¸ eref Ahunbay (2025) scale polynomially in 1/ϵ. Also, Theorem 4.1 applies even when Φ contains all linear endomorphisms (Theorem 4.1). From a more conceptual vantage point, a significant part of our contribution is to extend the scope of such results beyond games, as we have already highlighted. Kapron & Samieefar (2024) and Bernasconi et al. (2024) recently studied the computational complexity of VIs, and generalizations thereof—namely, *quasi VIs*, establishing PPAD-completeness under mild assumptions. Whether our framework can be extended to encompass quasi VIs is left as an interesting direction for the future. Finally, it would be remiss not to point out that our VI relaxation is in the spirit of "lifting," a standard technique whereby the original problem is *lifted* to a higher-dimensional space to gain more analytical and computational leverage; a concrete example, in the context of optimal transport theory, is Kantorovich's relaxation of Monge's formulation (Villani, 2009). Such techniques have been fruitful in the context of min-max optimization (Hsieh et al., 2019; Domingo-Enrich et al., 2020).

## B. Additional Preliminaries

Revisiting **Definition 1.2** In order to define the distributions ∆(X ) over X precisely, we recall here some basic concepts from probability theory. We refer to Billingsley (1999, Chapter 1 and 2) and Aliprantis & Border (2006, Chapter 15) for detailed treatments. We assume throughout the paper that the set X ⊆ R
dis Borel measurable. Let ∆(X ) be the set of Borel probability measures µ on X , that is, measures µ : (X , B(X )) → (R, B(R)) with µ(X ) = 1, where B(X ) and B(R) denote the respective σ-algebra of Borel sets. We simply call µ a distribution. For any Borel measurable function f : X → R—henceforth just *measurable*— we can then take the integral Ex∼µ[f(x)] := RX
f(x)dµ(x). In particular, for Ex∼µ⟨F(x), ϕ(x) − x⟩ in Definition 1.2 to be well-defined, we assume throughout this paper that F and each ϕ ∈ Φ are measurable functions.

For our computational results (Section 4), we are making a standard assumption regarding the geometry of X (Section 2); this can be met by bringing X into isotropic position. In particular, there is a polynomial-time algorithm that computes an affine transformation to accomplish that (Lovasz & Vempala ´ , 2006), and minimizing linear-swap regret reduces to minimizing linear-swap regret to the transformed instance (Daskalakis et al., 2025, Lemma A.1).

## C. Omitted Proofs

This section contains the proofs omitted from the main body. C.1. Existence of Φ**-EVI solutions** We begin with Theorem 3.1.

Theorem 3.1. Suppose that F : X → R
dis measurable and there exists L > 0 such that every ϕ ∈ Φ is L-Lipschitz continuous. Then, for any ϵ > 0, there exists an ϵ-approximate solution to the Φ-EVI problem.

Proof. We define a function Fbδ : X → R
das

$${\hat{F}}_{\delta}:{\mathbf{x}}\mapsto{\frac{1}{|{\mathcal{B}}_{\delta}({\mathbf{x}})\cap{\mathcal{X}}|}}\int_{{\mathcal{B}}_{\delta}({\mathbf{x}})\cap{\mathcal{X}}}F({\hat{\mathbf{x}}})d\nu({\hat{\mathbf{x}}});$$

this is a rescaled Lebesgue integral, which represents a multivariate local average. Above,
- δ > 0 is a sufficiently small parameter, to be defined shortly;
- Bδ(x) ⊆ R
dis the (closed) Euclidean ball of radius δ centered at x; and
- *| · |* denotes the set's Borel measure.

Given that F is assumed to be bounded, we can define B ∈ R such that maxx∈X ∥F(x)∥ ≤ B. For the proof below, it will suffice to set δ := ϵ/(L+1)B.

We first observe that Fbδ is continuous.

Claim C.1. Fbδ is continuous.

Proof. We will show that for any x ∈ X and ϵ
′ > 0, we can choose δ
′ = δ
′(ϵ
′) such that for any x
′ ∈ X with ∥x−x
′∥ < δ′,

$$\|{\widehat{F}}_{\delta}(\mathbf{x})-{\widehat{F}}_{\delta}(\mathbf{x}^{\prime})\|\leq\epsilon^{\prime}.$$
By the triangle inequality, the difference ∥Fbδ(x) − Fbδ(x
the difference $\|\widehat{F}_{\delta}(\mathbf{x})-\widehat{F}_{\delta}(\mathbf{x}^{\prime})\|$ can be decomposed as the sum of  $$\widehat{\mathbf{\delta}}):=\left|\frac{1}{|\mathcal{B}_{\delta}(\mathbf{x})\cap\mathcal{X}|}-\frac{1}{|\mathcal{B}_{\delta}(\mathbf{x}^{\prime})\cap\mathcal{X}|}\right|\int_{\mathcal{B}_{\delta}(\mathbf{x})\cap\mathcal{X}}\|F(\widehat{\mathbf{x}})\|d\nu(\widehat{\mathbf{x}})$$

and

$$\widehat{\otimes}:=\frac{1}{|\mathcal{B}_{\delta}(\mathbf{x}^{\prime})\cap\mathcal{X}|}\left\|\int_{\mathcal{B}_{\delta}(\mathbf{x})\cap\mathcal{X}}F(\widehat{\mathbf{x}})d\nu(\widehat{\mathbf{x}})-\int_{\mathcal{B}_{\delta}(\mathbf{x}^{\prime})\cap\mathcal{X}}F(\widehat{\mathbf{x}})d\nu(\widehat{\mathbf{x}})\right\|.$$  d as 
Now, $\scriptsize\left({\text{A}}\right)$ can be bounded as $\scriptsize1$. 
$$\widehat{\Phi}\leq B\left|1-\frac{|{\mathcal{B}}_{\delta}({\boldsymbol{x}})\cap{\mathcal{X}}|}{|{\mathcal{B}}_{\delta}({\boldsymbol{x}}^{\prime})\cap{\mathcal{X}}|}\right|\leq\frac{1}{2}\epsilon^{\prime},$$

where we selected δ
′small enough so that

$$\left(1-\frac{\epsilon^{\prime}}{B}\right)\vert\mathcal{B}_{\delta}(\mathbf{x}^{\prime})\cap\mathcal{X}\vert\leq\vert\mathcal{B}_{\delta}(\mathbf{x})\cap\mathcal{X}\vert\leq\left(1+\frac{\epsilon^{\prime}}{B}\right)\vert\mathcal{B}_{\delta}(\mathbf{x}^{\prime})\cap\mathcal{X}\vert.$$

Moreover, by selecting δ
′small enough so that

$$|\langle{\mathcal{B}}_{\delta}(\mathbf{x})\cap{\mathcal{X}}\rangle\setminus\langle{\mathcal{B}}_{\delta}(\mathbf{x}^{\prime})\cap{\mathcal{X}}\rangle|+|\langle{\mathcal{B}}_{\delta}(\mathbf{x}^{\prime})\cap{\mathcal{X}}\rangle\setminus\langle{\mathcal{B}}_{\delta}(\mathbf{x})\cap{\mathcal{X}}\rangle|\leq{\frac{1}{2B}}\epsilon^{\prime}|{\mathcal{B}}_{\delta}(\mathbf{x}^{\prime})\cap{\mathcal{X}}|,$$
′) *∩ X |*, (9)
we have
 Z Bδ(x)∩X F(xb)dν(xb) − Z Bδ(x′)∩X F(xb)dν(xb)  ≤ Z (Bδ(x)∩X )\(Bδ(x′)∩X ) ∥F(xb)∥dν(xb) +  Z (Bδ(x′)∩X )\(Bδ(x)∩X ) ∥F(xb)∥dν(xb) ≤ B|(Bδ(x) ∩ X ) \ (Bδ(x ′) ∩ X )| + B|(Bδ(x ′) ∩ X ) \ (Bδ(x) ∩ X )| ≤ 12 ϵ ′|Bδ(x ′) ∩ X |,
$$(9)$$

where the last inequality uses (9). As a result, we have shown that ⃝A +⃝B ≤ ϵ
′, thereby implying that ∥Fbδ(x)−Fbδ(x
′)∥ ≤ ϵ
′.

This completes the proof.

Having established that Fbδ is continuous, we can now apply Brouwer's fixed point theorem on the map x 7→ ΠX (x−Fbδ(x)),
where we recall that ΠX denotes the Euclidean projection onto X . This implies that there is a point x ∈ X such that x = ΠX (x − Fbδ(x)). Moreover, such a point satisfies the VI constraint with respect to Fbδ:

$$\langle{\widehat{F}}_{\delta}(\mathbf{x}),\mathbf{x}^{\prime}-\mathbf{x}\rangle\geq0\quad\mathbf{x}^{\prime}\in{\mathcal{X}};$$
$$(10)$$

for example, see Kinderlehrer & Stampacchia (2000, Section 3) for the derivation. Finally, we define µ ∈ ∆(X ) to be the uniform distribution over Bδ(x) ∩ X . Then, for any ϕ ∈ Φ,

$$\begin{split}\langle\hat{F}_{\delta}(\pmb{x}),\phi(\pmb{x})-\pmb{x}\rangle&=\pmb{\varepsilon}_{\omega\mu}\langle F(\pmb{\hat{x}}),\phi(\pmb{x})-\pmb{x}\rangle\\ &=\pmb{\varepsilon}_{\omega\mu}\langle F(\pmb{\hat{x}}),\pmb{\hat{x}}-\pmb{\varepsilon}_{\omega\mu}\langle F(\pmb{\hat{x}}),\phi(\pmb{x})-\phi(\pmb{\hat{x}})\rangle+\pmb{\varepsilon}_{\omega\mu}\langle F(\pmb{\hat{x}}),\phi(\pmb{\hat{x}})-\pmb{\hat{x}}\rangle.\end{split}$$  In in (10) one has bounded as 
The first term in (10) can be bounded as

$$\mathbb{E}_{\widehat{\mathbf{x}}\sim\mu}\left\langle F(\widehat{\mathbf{x}}),\widehat{\mathbf{x}}-\mathbf{x}\right\rangle\leq\sqrt{\mathbb{E}_{\widehat{\mathbf{x}}\sim\mu}\left\|F(\widehat{\mathbf{x}})\right\|^{2}}\sqrt{\mathbb{E}_{\widehat{\mathbf{x}}\sim\mu}\left\|\widehat{\mathbf{x}}-\mathbf{x}\right\|^{2}}\leq\delta B,\tag{1}$$

where we used the Cauchy-Schwarz inequality, the fact that ∥F(xb)∥ ≤ B for all xb ∈ X , and ∥xb − x∥ ≤ δ for all xb in the support of µ. Similarly, the second term in (10) can be bounded as

$$\mathbb{E}_{\widehat{\mathbf{x}}\sim\mu}\left\langle F(\widehat{\mathbf{x}}),\phi(\mathbf{x})-\phi(\widehat{\mathbf{x}})\right\rangle\leq\sqrt{\mathbb{E}_{\widehat{\mathbf{x}}\sim\mu}\left\|F(\widehat{\mathbf{x}})\right\|^{2}}\sqrt{\mathbb{E}_{\widehat{\mathbf{x}}\sim\mu}\left\|\phi(\widehat{\mathbf{x}})-\phi(\mathbf{x})\right\|^{2}}$$ $$\leq L\sqrt{\mathbb{E}_{\widehat{\mathbf{x}}\sim\mu}\left\|F(\widehat{\mathbf{x}})\right\|^{2}}\sqrt{\mathbb{E}_{\widehat{\mathbf{x}}\sim\mu}\left\|\widehat{\mathbf{x}}-\mathbf{x}\right\|^{2}}\leq\delta BL,$$

$$(11)$$
$$(12)$$
$\square$

where we additionally used the assumption that ϕ is L-Lipschitz continuous. Combining (11) and (12) with (10), we have

$$\operatorname*{\mathbb{E}}_{\hat{\mathbf{x}}\sim\mu}\langle F({\hat{\mathbf{x}}}),\phi({\hat{\mathbf{x}}})-{\hat{\mathbf{x}}}\rangle\geq-\delta(L+1)B+\langle{\hat{F}}_{\delta}(\mathbf{x}),\phi(\mathbf{x})\rangle$$

⟨F(xb), ϕ(xb) − xb*⟩ ≥ −*δ(L + 1)B + ⟨Fbδ(x), ϕ(x) − x*⟩ ≥ −*δ(L + 1)B,
and this holds for any ϕ ∈ Φ. Setting δ := ϵ/(L+1)B completes the proof.

We next proceed with Corollary 3.2 and Proposition 3.3, which are restated below.

Corollary 3.2. *There exists a VI problem that does not admit approximate solutions when* ϵ = Θ(1)*, but the corresponding* ϵ-approximate Φ-EVI is total for any ϵ > 0.

Proposition 3.3. When F *is not continuous, there exists an EVI problem with no solutions.* We provide an example that will establish both of those claims.

Example C.2 (Discontinuous F; cf. Corollary 3.2 and Proposition 3.3). Let F(x) be the sign function,

$$F(x)=\operatorname{sgn}(x):={\begin{cases}-1&{\quad\mathrm{if}\quad x<0,}\\ 1&{\quad\mathrm{otherwise,}}\end{cases}}$$

and X = [−1, 1]. We first claim that there is no ϵ-approximate VI solution for ϵ < 1. Indeed,
- for any x < 0, picking x
′ = 1 ensures ⟨F(x), x′ − x⟩ = x − 1 < −1;
- for any x ≥ 0, picking x
′ = −1 ensures ⟨F(x), x′ − x⟩ = −1 − x ≤ −1.

There is also no *exact* EVI solution to this problem. Indeed, consider any µ ∈ ∆(X ).

- If Px∼µ[x = 0] = 1, then taking x
′ = −1 ensures Ex∼µ⟨F(x), x′ − x⟩ = ⟨F(0), x′⟩ = −1.

- Otherwise, taking x
′ = 0, we have

$$\operatorname*{\mathbb{E}}_{x\sim\mu}\langle F(x),x^{\prime}-x\rangle=\operatorname*{\mathbb{E}}_{x\sim\mu}[-|x|]<0.$$

16 On the other hand, for any ϵ > 0, there exists an ϵ-approximate EVI solution (as promised by Theorem 3.1). In particular, suppose that µ uniformly picks between −ϵ and ϵ. Then, for any x
′ ∈ X ,

$$\operatorname*{\mathbb{E}}_{x\sim\mu}\langle F(x),x^{\prime}-x\rangle=-{\frac{1}{2}}(x^{\prime}+\epsilon)+{\frac{1}{2}}(x^{\prime}-\epsilon)=-\epsilon.$$

It is worth pointing out that the above example can be slightly modified so that exact EVI solutions do exist, as we explain below.

Example C.3 (Modification of Example C.2 with exact VI). We define F(x) identically to Example C.2, except F(1/2) =
−1. We claim that there is no VI solution for ϵ < 1/2: any x ̸= 1/2 by Example C.2, and x = 1/2 is not a solution since y = 1 ensures ⟨F(x), x′ − x⟩ = −1/2.

However, there is an exact EVI solution: fix any x
⋆ ∈ [0, 1/2) and consider µ that uniformly mixes between x = x
⋆and x = 1/2. Then, for any x
′ ∈ X ,

$$\operatorname*{\mathbb{E}}_{x\sim\mu}\langle F(x),x^{\prime}-x\rangle={\frac{1}{2}}(x^{\prime}-x^{\star})-{\frac{1}{2}}{\bigg(}x^{\prime}-{\frac{1}{2}}{\bigg)}={\frac{1}{2}}{\bigg(}{\frac{1}{2}}-x^{\star}{\bigg)}>0.$$

Our next result reveals that the precondition of Theorem 3.1 with respect to Φ cannot be relaxed to continuity. Theorem 3.4. There are Φ-EVI instances that do not admit ϵ*-approximate solutions even when* ϵ = Θ(1), F is piecewise constant, and Φ contains only continuous functions. Proof. As before, let F(x) be the sign function,

$$F(x)=\operatorname{sgn}(x):={\begin{cases}-1&{\quad\mathrm{if}\quad x<0,}\\ 1&{\quad\mathrm{otherwise,}}\end{cases}}$$

and µ ∈ ∆([−1, 1]) be any distribution. For δ > 0, let ϕδ : [−1, 1] → [−1, 1] be given by

$$\phi_{\delta}(x)=\begin{cases}1&\quad\text{if}\quad x<-2\delta,\\ -(x+\delta)/\delta&\quad\text{if}\quad-2\delta\leq x\leq0,\\ -1&\quad\text{if}\quad x>0.\end{cases}$$

Further, let ϕ0(x) := − sgn(x). Every ϕδ (with δ > 0) is continuous, by construction. Now, note that ϕδ → ϕ0 pointwise when δ ↓ 0, and every ϕδ is bounded. As a result, by the dominated convergence theorem, we have

$$\lim_{\delta\to0}\mathbb{E}\left[F(x)(\phi_{\delta}(x)-x)\right]=\mathbb{E}\left[F(x)(\phi_{0}(x)-x)\right]$$ $$=\mathbb{E}\left[-1-F(x)\cdot x\right]\leq-1,$$

where the last line uses the fact that F(x)ϕ0(x) = − sgn(x)
2 = −1 and F(x)· x = sgn(x)· x = |x| for all x. Thus, for any ϵ < 1, there must be some δ > 0 for which E[F(x)(ϕδ(x) − x)] < −ϵ, so µ cannot be an ϵ-approximate EVI solution.

Continuing on Section 3, we next provide the proof of Theorem 3.5. Theorem 3.5. *Suppose that* 1. Φ is finite-dimensional, that is, there exists k ∈ N and a kernel map m : X → R
ksuch that every ϕ ∈ Φ can be expressed as Km(x) *for some* K ∈ R
d×k; and 2. every ϕ ∈ Φ admits a fixed point, that is, a point X ∋ x = FP(ϕ) such that ϕ(x) = x.

Then, the Φ-EVI problem admits an ϵ*-approximate solution with support size at most* 1 + dk for every ϵ > 0.

Proof. We assume, without loss of generality, that (as functions) the coordinates mi: X → R for 1 ≤ i ≤ k are linearly independent. We further assume that m is bounded, again without loss of generality. (Indeed, if for example miis unbounded, then column i of K must contain all zeros, or else ϕK(x) := Km(x) would be unbounded; we can thus freely remove such coordinates mi.)
Now, let K := conv{K : ϕK ∈ Φ} be the set of matrices corresponding to maps in Φ; we can assume that K is closed. We can now rewrite the Φ-EVI problem as

$$\begin{array}{r l}{{\mathrm{find}}}&{{}\mu\in\Delta({\mathcal{X}})\quad{\mathrm{s.t.}}\quad\mathop{\mathbb{E}}_{\mathbf{x}\sim\mu}\left\langle F(\mathbf{x})m(\mathbf{x})^{\top},\mathbf{K}-\mathbf{I}\right\rangle\geq0}\end{array}$$

for all K ∈ K, where above I is the identity matrix and the inner product is the usual Frobenius inner product of matrices.8 Further, let A := conv{F(x)m(x)
⊤ : x *∈ X }*. Then, the Φ-EVI problem can be in turn expressed as

$$\mathrm{find}\quad\mathbf{A}\in{\mathcal{A}}\quad\mathrm{s.t.}\quad\langle\mathbf{A},\mathbf{K}-\mathbf{I}\rangle\geq0$$

for all K ∈ K. Since F and m are bounded, by assumption, so is A. Moreover, since the coordinates mi are linearly independent, K is also bounded. Thus, letting A¯ denote the closure of A, the max-min problem

$$\operatorname*{max}_{\mathbf{A}\in{\mathcal{A}}}\operatorname*{min}_{\mathbf{K}\in{\mathcal{K}}}\langle\mathbf{A},\mathbf{K}-\mathbf{I}\rangle$$
$$(13)$$
⟨A, K − I⟩ (13)
satisfies the conditions of the minimax theorem. Moreover, for any K ∈ K, the fixed point x := FP(ϕK) satisfies

$$\langle F(\mathbf{x})m(\mathbf{x})^{\top},\mathbf{K}-\mathbf{I}\rangle=\langle F(\mathbf{x}),\phi_{\mathbf{K}}(\mathbf{x})-\mathbf{x}\rangle=0,$$

so the zero-sum game (13) has a nonnegative value; that is, there exists A ∈ A¯ such that minK∈K⟨A, K − I⟩ ≥ 0. Thus, for every ϵ > 0, there exists A ∈ A such that minK∈K⟨A, K − I*⟩ ≥ −*ϵ. Moreover, by Caratheodory's theorem, ´ A can be expressed as a convex combination of at most 1 + dk matrices of the form F(x)m(x)
⊤. This convex combination is thus an ϵ-approximate EVI solution.

The only reason the above proof breaks when ϵ = 0 is that A may not be closed. Indeed, this issue is fundamental: there are instances where no exact EVI solutions exist even when ϕ contains only constant functions (Proposition 3.3).

## C.2. Complexity Of Φ**-Evis**

With regard to the complexity of computing Φ-EVI solutions, the key observation is that, when Φ contains all (measurable) maps, Φ-EVIs are essentially equivalent to VIs; this immediately implies a number of hardness results, which were covered earlier in Section 3. We provide the formal proof of Proposition 3.6 below.
Proposition 3.6. If Φ contains all measurable functions from X to X , then any solution µ ∈ ∆(X ) to the ϵ*-approximate*
Φ*-EVI problem satisfies*
$$\operatorname*{\mathbb{E}}_{x\sim\mu}{\mathrm{VIGap}}(x)\leq\epsilon.$$
VIGap(x) ≤ ϵ. (4)
Proof. We can define a measurable map ϕ : *X → X* such that ϕ(x) is an element selected from argminx′∈X ⟨F(x), x
′ − x⟩
by utilizing the measurable maximum theorem (Aliprantis & Border, 2006, Theorem 18.19). To satisfy the conditions of this theorem, we need to define—using Aliprantis & Border's notation— the weakly measurable set-valued function ψ : X ↠ X as ψ(x) = X and the (Caratheodory) function ´ f : *X × X →* R as f(x, x
′) = −⟨F(x), x
′ − x⟩. Due to this map ϕ, a Φ-EVI solution µ ∈ ∆(X ) must then, in particular, satisfy

$$\operatorname*{\mathbb{E}}_{\mathbf{x}\sim\mu}\langle F(\mathbf{x}),\phi(\mathbf{x})-\mathbf{x}\rangle=\operatorname*{\mathbb{E}}_{\mathbf{x}\sim\mu}\operatorname*{argmin}_{\mathbf{x}^{\prime}\in\mathcal{X}}\langle F(\mathbf{x}),\mathbf{x}^{\prime}-\mathbf{x}\rangle\geq0.$$

Therefore, there must exist x
⋆ ∈ X with argminx′∈X ⟨F(x
⋆), x
′ − x
⋆⟩ ≥ 0, that is, a VI solution x
⋆. If µ has finite support, then such a x
⋆exists within that support. The ϵ-approximation case follows analogously.

8To avoid measurability issues, it is enough to consider here only distributions µ with finite support.

$$\left(4\right)$$

$$(14)$$

## C.3. Proof Of Theorem 4.1

To establish Theorem 4.1, we will use the recent framework of Daskalakis et al. (2025), which refines Theorem 2.1 in the context of Section 2.1, ultimately summarized in Theorem C.4. Coupled with the "semi-separation oracle" of Lemma 4.2, we will thus arrive at Theorem 4.1.

Let X ⊆ R
dand Y ⊆ R
m be convex and compact sets. The goal is to solve the convex program

$$\begin{array}{r l r l}{{\mathrm{find}}}&{{}\mu\in\Delta({\mathcal{X}})}&{{\mathrm{s.t.}}}&{{}\operatorname*{min}_{\mathbf{y}\in{\mathcal{Y}}}\langle\mu,\mathbf{A}\mathbf{y}\rangle\geq0,}\end{array}$$
⟨µ, Ay⟩ ≥ 0, (14)
where ∆(X ) ⊆ RM and A ∈ RM×m; we think of M as being potentially exponentially large, so A is not given explicitly; ΦLIN-EVIs can be expressed as (14), assuming that µ has finite support (cf. Theorem 3.5). The target is to solve (14) with complexity polynomial in d and m (and other parameters of the problem, except M). As we saw earlier in Section 2.1, the EAH algorithm accomplishes that given access to a GER oracle, which, for any y ∈ Y, returns x ∈ X such that ⟨µ(x), Ay⟩ ≥ 0, where ∆(X ) ∋ µ(x) places all probability on x. Assuming that such an oracle exists, the convex program

$${\mathrm{find}}\quad y\in\mathbb{R}_{>0}\mathcal{Y}\quad{\mathrm{s.t.}}\quad\operatorname*{max}_{\mu\in\Delta(\mathcal{X})}\langle\mu,\mathbf{A}y\rangle\leq-1$$
$$(15)$$

⟨µ, Ay*⟩ ≤ −*1 (15)
is infeasible, where R>0Y := {cy : y ∈ Y*, c >* 0} is the conic hull of Y. Despite its infeasibility, EAH proceeds by applying the ellipsoid algorithm on (15)—this is where the name "ellipsoid against hope" comes from. In doing so, the ellipsoid will eventually shrink to an area with negligible volume (denoted by vol), at which point one can extract a certificate of infeasibility for (15) as follows. The execution of the ellipsoid will have produced a sequence of T ∈ N
good-enough-responses, x
(1)*, . . . ,* x
(T), such that for any y ∈ Y, it holds that ⟨µ(x
(t)), Ay⟩ ≥ 0 for some t ∈ [T] (up to numerical imprecision). In turn, this implies that there is a mixture µ over {x
(1)*, . . . ,* x
(T)} that guarantees ⟨µ, Ay⟩ ≥ 0 for every y ∈ Y. Such a µ can be computed in polynomial time by solving a smaller program, which simply searches over the mixing coefficients.

So far, we have elaborated on the framework presented in Section 2.1. To solve ΦLIN-EVIs, it is necessary to relax the oracle assumed above. In particular, the SEPorGER oracle, introduced by Daskalakis et al. (2024), proceeds as follows. It takes as input a point y ∈ R
m (not necessarily in Y), and must *either* return a good-enough-response x ∈ X , or a hyperplane separating y from Y. The idea now is to again run ellipsoid on (15), but by replacing Y with a convex "shell set"; every time the SEPorGER oracle returns a separating hyperplane, the shell set restricts further. At the end of this process, once the ellipsoid has shrank enough, one can work with the induced shell set Ye and proceed by identifying a mixture among the good-enough-responses {x
(1)*, . . . ,* x
(T)} that approximately solves (14). The overall scheme is given in Algorithm 1.

Below, we state the main guarantee of Algorithm 1 shown by Daskalakis et al. (2025); in its statement, we have made certain slight adjustments in accordance with our setting. Theorem C.4 (Daskalakis et al., 2025). Suppose that the following conditions hold.

1. A ∈ RM×m *such that for any* µ ∈ ∆(X ), ∥µ
⊤A∥ ≤ B for some B ≥ 1; 2. Y is convex and compact, and satisfies Bry(·) ⊆ Y ⊆ BRy(0); and 3. *there exists a* SEPorGER oracle: for every point y ∈ BRy(0)*, it runs in* poly(d, m) time, and either returns a hyperplane separating y from Y or a good-enough-response x ∈ X .

Then, Algorithm 1 *runs in* poly(*d, m,* log(B/ϵ)) time and computes µ ∈ ∆(X ) *such that*

$$\operatorname*{min}_{y\in{\mathcal{Y}}}\langle\mu,\mathbf{A}y\rangle\geq-\epsilon.$$

That the second precondition (Item 2) is satisfied follows from Daskalakis et al. (2025, Lemma 2.3). The third precondition, Item 3, is satisfied by virtue of Lemma 4.2. Consequently, Theorem 4.1 follows from Theorem C.4.

Remark C.5 (Weak oracles and finite precision). Since we are working with general convex sets, the oracles posited in Section 2 (namely, membership, separation, and linear optimization) can return irrational outputs. This can be addressed by employing *weak* versions of those oracles, which relax the output by allowing some small slackness ϵ (Grotschel et al. ¨ ,
1993). Theorem 4.1 can be readily extended under those weaker oracles; see Daskalakis et al. (2025, Appendices E and F).

Algorithm 1 Ellipsoid against hope (EAH) under SEPorGER oracle (Daskalakis et al., 2025)
input - Parameters Ry, ry > 0 such that Bry(·) *⊆ Y ⊆ B*Ry(0)
- precision parameter ϵ > 0
- constant B ≥ 1 such that ∥µ
⊤A∥ ≤ B for all µ ∈ ∆(X )
- a SEPorGER oracle output A sparse, ϵ-approximate solution µ ∈ ∆(X ) of (14)
1: Initialize the ellipsoid E := BRy(0) 2: Initialize Ye := BRy(0)
3: **while** vol(E) ≥ vol(Bϵ/B(·)) do 4: Query the SEPorGER oracle on the center of E
5: if it returns a good-enough-response x ∈ X **then**
6: Update E to the minimum volume ellipsoid containing *E ∩ {*y ∈ R
m : ⟨y, A⊤µ(x)⟩ ≤ 0}
7: **else**
8: Let H be the halfspace that separates y from Y
9: Update E to the minimum volume ellipsoid containing E ∩ H
10: Update Ye := Y ∩e H
11: **end if** 12: **end while** 13: Let x
(1)*, . . . ,* x
(T) be the GER oracle responses produced in the process above 14: Define X := [µ(x
(1)) | *. . .* | µ(x
(T))] and compute X⊤A ∈ R
T ×m 15: Compute a solution λ to the convex program

$$\mathrm{find}\quad\lambda\in\Delta^{T}\quad\mathrm{s.t.}$$
$$\mathbf{A})y\geq-\epsilon$$
$|\ensuremath{\psi}\rangle$
y∈Ye
λ
⊤(X⊤A)y ≥ −ϵ
16: **return** ∆(X ) ∋ µ := PT
t=1 λ
(t)µ(x
(t))

## D. Characterizing Linear Endomorphisms For Polytopes

In this section, we answer the following question. Given a nonempty polytope X = {x ∈ R
d: Ax ≤ b} where A ∈ R
m×d and b ∈ R
m, we wish to characterize the set of (affine) linear maps ϕ : *X → X* . That is, we wish to understand the set of pairs (K, c) ∈ R
d×d × R
dsuch that Kx + c ∈ X for all x ∈ X . The following result provides an explicit polynomial representation for that set, establishing Theorem 4.3.

Theorem D.1. Kx + c ∈ X for all x ∈ X *if and only if there is a matrix* V ∈ R
m×m *satisfying the constraints* VA = AK, Vb ≤ b − Ac, V ≥ 0.

Proof. Let K ∈ R
d×dand c ∈ R
d, and let a
⊤
i x ≤ bi be the ith constraint that defines X . Then, the claim that a
⊤
i(Kx + c) ≤ bi for every x ∈ X is equivalent to the claim that the linear program

$$(16)$$
$$\operatorname*{max}_{x}\quad a_{i}^{\top}\mathbf{K}x\quad{\mathrm{s.t.}}\quad\mathbf{A}x\leq b$$
$$i_{\uparrow}=\quad v$$
i Kx s.t. Ax ≤ b (16)
has value at most bi − a
⊤
ic. By strong duality, (16) has the same value as

$$\begin{array}{r l}{\operatorname*{min}}&{{}{\mathbf{}}b^{\dagger}}\\ {{}}&{{}{\mathbf{}}v_{i}}\end{array}$$
⊤vi s.t. A⊤vi = K⊤ai, v ≥ 0.
$$\mathbf{b}^{\top}\mathbf{v}_{i}\quad\mathrm{s.t.}\quad\mathbf{A}^{\top}\mathbf{v}_{i}=\mathbf{K}^{\top}\mathbf{a}_{i},$$

The theorem follows now by setting V =-v1 *. . .* vk⊤. Furthermore, assuming that B1(0) *⊆ X ⊆ B*R(0) with R ≤ poly(d), it follows that ∥K∥2, ∥V∥2 ≤ poly(d), where ∥ · ∥2 denotes the spectral norm. Indeed, to begin with, ∥c∥2 ≤ R since K · 0 + c *∈ X ⊆ B*r(0). For ∥K∥2, take any x ∈ R
d with ∥x∥ = 1. Since B1(0) ⊆ X , we have x ∈ X , in turn implying that Kx + c ∈ X . As a result,
∥Kx∥ − ∥c∥ ≤ ∥Kx + c∥ ≤ poly(d), from which it follows that ∥K∥2 ≤ poly(d). Further, one can take each ai and bito be such that 1 ≤ bi ≤ poly(d) and ∥ai∥ = 1, and so the bound ∥V∥2 ≤ poly(d) follows from the fact that Vb ≤ b − Ac and V ≥ 0.

$\square$