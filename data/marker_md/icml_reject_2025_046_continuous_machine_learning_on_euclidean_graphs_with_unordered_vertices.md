011

014 015 016

018

024

026

034

036

038

054

# Continuous machine learning on Euclidean graphs with unordered vertices

### Abstract

Molecular graphs can change their chemical properties under non-rigid deformations in Euclidean space. Hence it is vitally important to distinguish rigid classes of molecular graphs under compositions of translations and rotations. Also, robust outputs of machine learning on molecular graphs embedded in Euclidean space should continuously change under perturbations, motivated by atomic vibrations and experimental noise. We developed a complete invariant that can be inverted back to an embedded graph, uniquely under rigid motion, and has a Lipschitz continuous distance satisfying all metric axioms. For a fixed dimension, the invariant and metric can be computed in polynomial time of the number m of unordered vertices and hence avoiding exponentially many permutations. The new invariants distinguish all chemically different graphs in the world's largest databases of 3D molecules in a few hours on a modest desktop.

![Chemical structures of the propagating polymer structures shown in the page.]()The page displays four chemical structures of polymer structures (PS) and their respective chemical structures (CS).

- **Top Row:** The first structure is a diagonal chemical structure with a red arrow pointing to the second carbon atom of the first carbon and a green arrow pointing to the second carbon atom of the second carbon. The second carbon atom is red and the second carbon atom is green.
- **Middle Row:** The second structure is a diagonal chemical structure with a red arrow pointing to the second carbon atom of the first carbon and a green arrow pointing to the second carbon atom of the second carbon. The second carbon atom is red and the second carbon atom is green.
- **Bottom Row:** The third structure is a diagonal chemical structure with a red arrow pointing to the second carbon atom of the first carbon and a green arrow pointing to the second carbon atom of the second carbon. The second carbon atom is red and the second carbon atom is green.
- **Bottom Row:** The fourth structure is a diagonal chemical structure with a red arrow pointing to the second carbon atom of the first carbon and a green arrow pointing to the second carbon atom of the second carbon. The second carbon atom is red and the second carbon atom is green.

![](_page_0_Diagram_5.jpeg)

## 1. Motivations for complete and continuous invariant inputs in application-driven ML

This paper formalizes necessary conditions for ML on real data with ambiguous representations and develops complete and Lipschitz continuous invariants satisfying these conditions on any Euclidean graphs and justifying a rigorous concept of a molecular structure. Many real structures from star constellations to molecules are represented by graphs embedded in a Euclidean space [\(Bonchev,](#page-8-0) [1991\)](#page-8-0). A *Euclidean graph* G ⊂ R <sup>n</sup> is a finite set of m unordered (unlabeled) vertices located at distinct points of R <sup>n</sup> and connected by straight-line edges. Forgetting all edges of G ⊂ R <sup>n</sup> gives us the *vertex set* V (G) ⊂ <sup>R</sup> <sup>n</sup> of m unordered points. A Euclidean graph can be disconnected and can have vertices v of any *degree* that is the number of edges whose endpoint is v. Loops and multiple edges (with the same endpoints) do not appear in Euclidean graphs because all edges are straight line segments and can also intersect in theory.

Graphs can be considered under any *equivalence* relation that should satisfy the axioms: 1) *reflexivity*: G ∼ G, 2) *symmetry*: if G ∼ F then F ∼ G, 3) *transitivity*: if G ∼ F and F ∼ H then G ∼ H. In chemistry, the simplest equivalence is by chemical composition, which is insufficient in practice, e.g. *stereoisomers* in Fig. [1](#page-0-0) (right) have the same chemical compositions and non-equivalent rigid shapes with different chemical properties [\(Rieder et al.,](#page-10-0) [2023\)](#page-10-0).

Figure 1. Top: graphs T1, T2, T3, T<sup>4</sup> ⊂ <sup>R</sup> 3 on the same vertices with solid edges are not isomorphic to each other. Bottom: stereoisomers are isomorphic combinatorially, not geometrically.

For molecules, the strongest equivalence (distinguishing as many graphs as practically possible) is a *geometric isomorphism* G ∼= F, i.e. an orientation-preserving transformation of R <sup>n</sup> that bijectively maps the vertices and edges: G → F. Geometric isomorphisms are also called *rigid motions* (compositions of translations and rotations), which form the special Euclidean group SE(n). The slightly weaker equivalence (not distinguishing mirror images) is an *isometry*, which is any distance-preserving transformation including reflections. Any geometrically isomorphic molecules have the same chemical properties. If a flexible molecule changes its rigid shape, its functional properties can change, so it is important to distinguish rigid shapes [\(Wilson et al.,](#page-11-0) [1991\)](#page-11-0).

To reliably distinguish at least some Euclidean graphs G ⊂ R <sup>n</sup>, we need an *invariant* I defined as a numerical descriptor preserved by any rigid motion in R <sup>n</sup>. Alternatively, if I(G) ̸= I(F), then G ̸∼= F, so any invariant has *no false negatives* that are pairs of different representatives of *rigidly equivalent graphs* (denoted by G ∼= F) having equal values of a (non-invariant) descriptor. The number of vertices (or edges) of G is an integer-valued weak invariant that cannot separate any graphs in Fig. [1.](#page-0-0) The strongest invariant I separating all non-equivalent graphs is called *complete* meaning that if I(G) = I(F) then G ∼= F. Alternatively, a

<sup>.</sup> Correspondence to: Anonymous Author <anon.email@domain.com>.

Preliminary work. Under review by the International Conference on Machine Learning (ICML). Do not distribute.

058

071

074

076

078

087 088

090 091

093 094

096

098

100

104

106

108 109 complete invariant I has *no false positives* that are pairs of non-equivalent graphs G ̸∼= F with I(G) = I(F).

Since all real data (such as inter-point distances) are noisy, a more practically important answer is not binary ('same or different') but should be continuously quantified by a distance metric between isometry classes. The atomic vibrations [\(Feynman,](#page-9-0) [1971\)](#page-9-0) imply that rigid classes of molecules graphs on m unordered atoms form a continuous *Graph Isometry Space* GIS(<sup>R</sup> 3 ; m). Only for triangular graphs with m = 3, their space was previously known due to the side-side-side theorem saying that any triangles are isometric if and only if they have the same triple of sides (inter-point distances) a, b, c considered up to 6 permutations. Hence the space of triangular graphs is {0 < a ≤ b ≤ c ≤ a+b} ⊂ <sup>R</sup> 3 , where c ≤ a+b guarantees that distances a, b, c are realizable by a real triangle.

Problem 1.1 (complete invariant of Euclidean graphs with a polynomial-time continuous metric). *Find a function* I : {*Euclidean graphs with of unordered vertices in* <sup>R</sup> <sup>n</sup>} → *a space* X *with a distance* d *satisfying the conditions below:*

*(a) completeness of the invariant: any graphs* G, F *are related by rigid motion in* R <sup>n</sup> *if and only if* I(G) = I(F)*;*

*(b) Lipschitz continuity: there is a constant* λ *and a metric* d *satisfying the axioms 1)* d(α, β) = 0 *if and only if* α = β*, 2)* d(α, β) = d(β, α)*, 3)* d(α, β) + d(β, γ) ≥ d(α, γ) *for all* α, β, γ ∈ X*, such that if* F *is obtained by perturbing every vertex of* G *up to* ε > 0*, then* d(I(G), I(F)) ≤ λε*;*

*(c) invertibility: any Euclidean graph* G *can be reconstructed (uniquely up to rigid motion in* R <sup>n</sup>*) from* I(G)*;*

*(d) computability: for a fixed dimension* n*, the invariant* I*,* d*, and a reconstruction of* G ⊂ R <sup>n</sup> *from* I(G) *can be obtained in polynomial time of the number of vertices.*

Condition [1.1\(](#page-1-0)a) means that a complete invariant I has the strongest expressivity [\(Zhang et al.,](#page-11-1) [2024\)](#page-11-1) by uniquely identifying any Euclidean graph under geometric isomorphism. To be useful for noisy inputs, a complete invariant should continuously change under perturbations in a suitable metric. The axioms in [1.1\(](#page-1-0)b) are the foundations of metric geometry [\(Melter & Tomescu,](#page-10-1) [1984\)](#page-10-1) and accepted in chemistry [\(Wein](#page-11-2)[hold,](#page-11-2) [1975\)](#page-11-2). If the triangle axiom fails with any additive error, the classical k-means and DBSCAN clustering are open to adversarial attacks in [\(Rass et al.,](#page-10-2) [2024\)](#page-10-2). If the first axiom is ignored, d ≡ 0 satisfies all other axioms. The first axiom implies the completeness of I in [1.1\(](#page-1-0)a) but the continuity is much stronger. Indeed, for any complete invariant I, one can define the discrete metric d(I(G), I(F)) = 1 for G ̸∼= F, which unhelpfully treats all non-equivalent graphs (even near-duplicates) as equally distant. The Lipschitz continuity in [1.1\(](#page-1-0)b) is necessary for smoothness, which is implicitly assumed by any gradient-based optimization.

Condition [1.1\(](#page-1-0)c) requires I to be not only complete and continuous but also efficient to explicitly reconstruct G, even better than a DNA code that is does not explain how to grow a living organism. Computability [1.1\(](#page-1-0)d) prevents brute-force attempts, e.g. defining I(G) as the infinite set of images of G under all rigid motions or taking m! distance matrices over all permutations of m unordered vertices.

The main contribution is the new invariant Nested Centered Distribution, which solves Problem [1.1,](#page-1-0) including the new Lipschitz continuity, for all Euclidean graphs in R n.

## 2. Past work on distances for Euclidean graphs

Ordered clouds. The vertex set V (G) of a Euclidean graph G ⊂ R <sup>n</sup> is called a *point cloud* C. If all points p1, . . . , p<sup>m</sup> of C are ordered (not under the action of all m! permutations), a complete invariant of C under isometry (compositions of translations, rotations, reflections) is the classical m×m matrix [\(Li et al.,](#page-10-3) [2023\)](#page-10-3) of pairwise distances |pi−p<sup>j</sup> | due to Theorem 9 in [\(Grinberg & Olver,](#page-9-1) [2019\)](#page-9-1) or, after shifting the center of mass to the origin, the Gram matrix of scalar products pip˙<sup>j</sup> by Theorem 1 in [\(Dekster & Wilker,](#page-8-1) [1987\)](#page-8-1). This multidimensional scaling [\(Schoenberg,](#page-10-4) [1935\)](#page-10-4) can also provide an embedding C ⊂ R <sup>k</sup> preserving all distances of C for a dimension k ≤ m. This embedding C ⊂ R <sup>k</sup> uses eigenvectors whose ambiguity up to signs gives an exponential time that can be close to O(2<sup>m</sup>), not polynomial in the number m of ordered points as in [1.1\(](#page-1-0)d).

Unordered clouds. Computational geometry developed many algorithms for detecting geometric isomorphism (or isometry, also called congruence) between point sets without edges [\(Huttenlocher et al.,](#page-9-2) [1993;](#page-9-2) [Chew & Kedem,](#page-8-2) [1992;](#page-8-2) [Chew et al.,](#page-8-3) [1999;](#page-8-3) [Goodrich et al.,](#page-9-3) [1999\)](#page-9-3). For a set A ⊂ Q<sup>n</sup> of m points, Theorem 3 in [\(Arvind & Rattan,](#page-8-4) [2016\)](#page-8-4) computed in time n <sup>O</sup>(n)poly(mM) a canonizing function f(A), which can be considered a complete isometry invariant of A, where M upper bounds the binary encodings of the rational coordinates in the input. For point clouds under rigid motion (also distinguishing mirror images), Theorem 4.7 in [\(Widdowson & Kurlin,](#page-11-3) [2023\)](#page-11-3) described a metric computable in time O(n(m<sup>n</sup>−<sup>1</sup>/n!)<sup>3</sup> log m). [\(Hordan et al.,](#page-9-4) [2024;](#page-9-4) [Delle Rose et al.,](#page-9-5) [2024;](#page-9-5) [Nigam et al.,](#page-10-5) [2024;](#page-10-5) [Amir](#page-8-5) [et al.,](#page-8-5) [2024;](#page-8-5) [Maennel et al.,](#page-10-6) [2024\)](#page-10-6) also achieved the completeness for point clouds but without a Lipschitz continuous metric as in [1.1\(](#page-1-0)b). Energy potentials written as infinite series of spherical harmonics, are often considered complete representations of atomic environments, which holds in the limit but not for a finite size[\(Pozdnyakov et al.,](#page-10-7) [2020\)](#page-10-7). For a fixed set of m vertices in general position, one can choose any of m(m − 1)/2 edges and produce 2 m(m−1)/2 non-isometric graphs. Problem [1.1](#page-1-0) for arbitrary graphs is computationally much harder than for point clouds due to exponentially many different graphs on the same vertex set.

114 115 116

118

124

126

128

131

134

136

138

151

154

158

160

The graph isomorphism problem [\(Grohe & Schweitzer,](#page-9-6) [2020\)](#page-9-6) for abstract (non-Euclidean) graphs is another version of Problem [1.1](#page-1-0) without continuous metrics. The latest advances [\(Babai,](#page-8-6) [2019;](#page-8-6) [Helfgott et al.,](#page-9-7) [2017\)](#page-9-7) achieved only quasipolynomial time. While many partial cases were solved, e.g. for planar graphs (embedded in R <sup>2</sup> without intersecting edges), see [\(Kiefer et al.,](#page-9-8) [2019\)](#page-9-8), the k-dimensional Weisfeiler-Leman test [\(Leman & Weisfeiler,](#page-10-8) [1968\)](#page-10-8) fails for 3-regular graphs of size O(k). The key limitation of WL tests is their local nature when invariants are gradually expanded from a vertex or a k-tuple. Then covering a graph on m vertices needs O(m) expansions leading to exponential sizes in m. Section 3.9 in [\(Dym & Gortler,](#page-9-9) [2024\)](#page-9-9) discussed that a complete invariant (under all permutations of m vertices) that has a polynomial time in the dimension n would also solve the graph isomorphism problem in polynomial time. Condition [1.1\(](#page-1-0)d) is easier for a fixed dimension n, e.g. n = 2, 3 are practical cases. The number m of vertices can be dozens or hundreds, e.g. for molecular graphs in R 3 , where vertices are centers of atoms and edges are interatomic bonds that keep atoms together in a stable molecule.

Geometric Deep Learning in [\(Bronstein et al.,](#page-8-7) [2021\)](#page-8-7) pioneered an axiomatic approach to geometric classifications beyond Euclidean space R <sup>n</sup> in [\(Bronstein et al.,](#page-8-8) [2017\)](#page-8-8). Some neural networks were proved to be universal [\(Maron](#page-10-9) [et al.,](#page-10-9) [2019;](#page-10-9) [Zhou,](#page-11-4) [2020;](#page-11-4) [Abbe & Sandon,](#page-8-9) [2020\)](#page-8-9) in the sense of approximating any continuous function on given data with sufficiently many layers. This universality property has been strengthened in Problem [1.1](#page-1-0) to the full completeness of an explicit invariant that should be computable in polynomial time and invertible to an original graph up to rigid motion. The key challenge was to compute an exact (not approximate) metric that is also Lipschitz continuous.

Equivariants [\(Kondor & Trivedi,](#page-9-10) [2018;](#page-9-10) [Cohen et al.,](#page-8-10) [2019;](#page-8-10) [Fuchs et al.,](#page-9-11) [2020;](#page-9-11) [Deng et al.,](#page-9-12) [2021\)](#page-9-12) are defined as descriptors E satisfying E(f(G)) = T<sup>f</sup> (E(G)) for any rigid motion f and all graphs G ⊂ R <sup>n</sup>, where T<sup>f</sup> can be any map, not only the identity as for invariants. Any linear combination of points, e.g. the center of mass, is equivariant but cannot distinguish graphs under translation. Equivariants [\(Gao et al.,](#page-9-13) [2020;](#page-9-13) [Qi & Luo,](#page-10-10) [2020;](#page-10-10) [Tu et al.,](#page-11-5) [2022;](#page-11-5) [Batzner et al.,](#page-8-11) [2022\)](#page-8-11) help predict forces acting on atoms to move them to a more optimal configuration. These timedependent graphs G<sup>t</sup> can be studied directly by invariant values I(Gt) without computing intermediate atomic forces.

Many neural networks optimize millions of parameters, e.g. see Table 4 [\(Goyal et al.,](#page-9-14) [2021\)](#page-9-14), to achieve great accuracies [\(Dong et al.,](#page-9-15) [2018;](#page-9-15) [Akhtar & Mian,](#page-8-12) [2018;](#page-8-12) [Laidlaw &](#page-9-16) [Feizi,](#page-9-16) [2019;](#page-9-16) [Guo et al.,](#page-9-17) [2019;](#page-9-17) [Colbrook et al.,](#page-8-13) [2022\)](#page-8-13) but require re-training on any new data. All known descriptors of molecular graphs [\(Duvenaud et al.,](#page-9-18) [2015;](#page-9-18) [Choo et al.,](#page-8-14) [2023\)](#page-8-14) have no proofs of all conditions [1.1\(](#page-1-0)a,b,c,d).

Gromov-Wasserstein metrics [\(Memoli](#page-10-11) ´ , [2011\)](#page-10-11) are defined for any metric-measure spaces [\(Brecheteau](#page-8-15) ´ , [2019\)](#page-8-15) by minimizing over infinitely many correspondences between points, but cannot be approximated with a factor less than 3 in polynomial time unless P=NP by Corollary 3.8 in [\(Schmiedl,](#page-10-12) [2017\)](#page-10-12) and Theorem 3.3 in [\(Agarwal et al.,](#page-8-16) [2018\)](#page-8-16), see fast algorithms for important cases in [\(Memoli et al.](#page-10-13) ´ , [2021;](#page-10-13) [Lim et al.,](#page-10-14) [2023;](#page-10-14) [Majhi et al.,](#page-10-15) [2024\)](#page-10-15). [\(Nikolentzos](#page-10-16) [et al.,](#page-10-16) [2017;](#page-10-16) [Majhi & Wenk,](#page-10-17) [2022;](#page-10-17) [Buchin et al.,](#page-8-17) [2023\)](#page-8-17) made significant advances in the related problems of matching and finding distances between fixed Euclidean graphs without considering isometry. Computing a metric between rigid classes is only a small part of Problem [1.1.](#page-1-0) Indeed, to efficiently navigate on Earth, in addition to distances between cities, we need a map of the planet and hence an invertible continuous invariant I similar to geographic coordinates.

## 3. Graph invariants: from fastest to complete

Let |p−q| denote the Euclidean distance between any points p, q ∈ R <sup>n</sup>. We always translate any graph G ⊂ <sup>R</sup> <sup>n</sup> so that the *center of mass* <sup>O</sup>(G) = <sup>1</sup> m P p∈V (G) p of the *vertex set* V (G) is at the origin 0 ∈ <sup>R</sup> <sup>n</sup>. Then Problem [1.1](#page-1-0) reduces to the SO(n)-invariance under orthogonal transformations.

Definition 3.1 (signed distance d(p, q) and invariants SRD, SPD,PDD). *Let* G ⊂ R <sup>n</sup> *be any Euclidean graph on* m *arbitrarily ordered vertices* p<sup>1</sup> . . . , pm*. If any* p<sup>i</sup> , p<sup>j</sup> ∈ V (G) *are connected by an edge of* G*, define the* signed distance *as* d(p, q) = |p − q|*, else set* d(p, q) = −|p − q|*.*

*(a) The vector* SRD(G) *of* sorted radial distances *consists of* m *distances* |p| *for all* p ∈ V (G) *in decreasing order.*

*(b) The vector* SPD(G) *of* sorted pairwise distances *consists of all distances* d(p<sup>i</sup> , p<sup>j</sup> ) *in decreasing order.*

*(c) Let* D(G) *be the* m×(m−1)*-matrix whose the* i*-th row consists of* d(p<sup>i</sup> , p<sup>j</sup> )*,* j ∈ {1, . . . , m} \ {i}*, in increasing order. The* Pointwise Distance Distribution PDD(G) *consists of these unordered rows with equal weights* 1/m*.*

If any k > 1 rows of D(G) are equal, they can be collapsed in PDD(G) to a single row with the *weight* k/m. The PDD was defined for clouds as a local distribution of distances in Definition 5.5 of [\(Memoli](#page-10-11) ´ , [2011\)](#page-10-11) and for periodic sets in [\(Widdowson & Kurlin,](#page-11-6) [2022\)](#page-11-6) but not for Euclidean graphs.

Table 1. Acronyms of all main invariants and metrics in the paper.

| SRD | S | ORTED   | R        | ADIAL   | V         | ECTOR         | D | EF | 3.1 |
|-----|---|---------|----------|---------|-----------|---------------|---|----|-----|
| SPD | S | ORTED   | D        | ISTANCE | V         | ECTOR         | D | EF | 3.1 |
| PDD | P |         | OINTWISE | D       | ISTANCE   | D ISTRIBUTION | D | EF | 3.1 |
| CR  |   | C       | ENTERED  | R       |           | EPRESENTATION | D | EF | 3.3 |
| NCD |   | N ESTED | C        | ENTERED | D         | ISTRIBUTION   | D | EF | 3.5 |
| NBM |   | N ESTED | B        |         | OTTLENECK | M ETRIC       | D | EF | 4.5 |

168

171

174

176

178

194

196 197 198

200

204

206

208

211

214 215 216

218

The PDD(G) includes every signed distance twice, once as d(p, q) in the row of a vertex p, and as d(q, p) in the row of a vertex q. Hence SPD(G) can be obtained from PDD(G) by (1) combining all distances into one vector, (2) sorting them in decreasing order, and (3) keeping only one copy of every two repeated distances. Example [3.2](#page-3-1) shows that the invariant PDD(G) is strictly stronger than SPD(G).

Example 3.2 (invariants SRD, SPD,PDD for tetrahedral graphs in Fig. [1\)](#page-0-0). *(a) Since the vertex sets of* T<sup>i</sup> ⊂ <sup>R</sup> 3 *are regular tetrahedra with all pairwise distances* 1*, these graphs have identical* SRD(Ti) *of 4 equal circumradii of the same vertex set* V (Ti) *independent of* i = 1, . . . , 4*.*

*The first graph* T<sup>1</sup> *has two edges contributing* +1 *and four non-edges (dashed lines) contributing* −1 *to the Sorted Distance Vector* SPD(T1) = (+1, +1, −1, −1, −1, −1)*. The graph* T<sup>2</sup> *also has two edges, so* SPD(T2) = SPD(T1) *doesn't distinguish* <sup>T</sup><sup>1</sup> ̸∼<sup>=</sup> <sup>T</sup><sup>2</sup> *up to rigid motion. Similarly, the graphs* <sup>T</sup><sup>3</sup> ̸∼<sup>=</sup> <sup>T</sup><sup>4</sup> *are not distinguished by the invariants* SPD(T3) = (+1, +1, +1, −1, −1, −1) = SPD(T4)*.*

*(b) In* T1*, every vertex has exactly one edge and two non-edges (dashed lines), hence its signed distances are* +1, −1, −1*. The matrix* PDD(T1) = (100% | −1, −1, +1) *consists of a single row, where the weight* 100% *indicates that all vertices of* T<sup>1</sup> *have the same row in* PDD*. The graph* T<sup>2</sup> *has one vertex (25%) with no edges, two vertices (*50%*) with one edge, and one vertex (25%) with two edges,*

$$\text{so PDD}(T_2) = \begin{pmatrix} 25\% & -1 & -1 & -1 \\ 50\% & -1 & -1 & +1 \\ 25\% & -1 & +1 & +1 \end{pmatrix} \neq \text{PDD}(T_1),$$

*so* PDD *distinguishes the rigidly non-equivalent graphs* <sup>T</sup><sup>1</sup> ̸∼<sup>=</sup> <sup>T</sup><sup>2</sup> *with* SPD(T1) = SPD(T2)*. The graph* <sup>T</sup><sup>3</sup> *has one vertex (25%) with no edges and three vertices (*75%*) with two edges, so* PDD(T3) = 25% −1 −1 −1 75% <sup>−</sup>1 +1 +1 *. The graph* T<sup>4</sup> *has two vertices (50%) with one edge and two vertices (*50%*) with two edges. Then* PDD(T4) = 50% −1 −1 +1 50% <sup>−</sup>1 +1 +1 *, so* PDD *distinguishes the graphs* <sup>T</sup><sup>3</sup> ̸∼<sup>=</sup> <sup>T</sup><sup>4</sup> *with equal* SPD(T3) = SPD(T4)*.*

For a graph G with m unordered vertices, PDD(G) has m − 1 columns. The reduced version PDD(G; k) includes only the first k columns for 1 ≤ k < m − 1. Though PDDs have unordered rows, they can be continuously compared by Earth Mover's Distance [\(Rubner et al.,](#page-10-18) [2000\)](#page-10-18).

$$-1, \quad D(p_1, p_3) = \begin{pmatrix} 0 & -2 & -\sqrt{2} \\ -2 & 0 & \sqrt{10} \\ -\sqrt{2} & \sqrt{10} & 0 \end{pmatrix}, \quad \text{and}$$

Fig. S4 in [\(Pozdnyakov et al.,](#page-10-7) [2020\)](#page-10-7) described infinitely many non-isometric pairs of clouds C, C′ ⊂ <sup>R</sup> <sup>3</sup> with PDD(C) = PDD(C ′ ). These counter-examples inspired the stronger invariants for graphs below. For simplicity, we will introduce all invariants and metrics in dimension n = 2. All higher dimensions n > 2 are covered in appendices. While PDD(G) includes signed distances to a single (arbitrary) vertex p<sup>i</sup> ∈ V (G), a stronger invariant below include

$$R(G; p_1, p_3) = \begin{pmatrix} -|p_2| \\ |p_2 - p_1| \\ |p_2 - p_3| \end{pmatrix} = \begin{pmatrix} -\sqrt{2} \\ \sqrt{10} \\ 2 \end{pmatrix}. \text{ The final triple is } \text{CR}(G; p_1, p_3) = [-1, D(p_1, p_3), R(G; p_1, p_3)].$$

triples of signed distances to three base points, one of which is the center of mass of V (G) because any point in <sup>R</sup> 2 is uniquely determined by its distances to three fixed points.

Definition 3.3 (Centered Representation CR(G; A) of a graph with A ⊂ V (G)). *Let* G ⊂ <sup>R</sup> <sup>2</sup> *be a graph on* m *unordered points with the center of mass* p<sup>0</sup> = O(G) = 0*.*

*(a) For any vertex* p<sup>1</sup> ∈ V (G)*, the matrix* R(G; p1) *has* m − 1 *unordered columns, one for each vertex* q ∈ V (G)\{p1}*, consisting of the signed distances* d(q, p0) *and* d(q, p1)*. Here* p<sup>0</sup> = 0 *is not considered as a vertex of* G*, so* d(q, p0) = −|q|*. The* Centered Representation CR(G; p1) *is the pair* [d(p0, p1), R(G; p1)]*, where* d(p0, p1) = −|p1|*.*

*(b) Fix a* base pair A *of ordered vertices* p1, p<sup>2</sup> ∈ V (G)*. Let* sign(A) *be the sign of the* 2 × 2 *determinant on the vectors* p1, p2*. Let* D(A) *be the matrix of signed distances between* p0, p1, p2*. The matrix* R(G; A) *has* m − 2 *unordered columns, one for each vertex* q ∈ V (G) \ A*, consisting of signed distances* d(q, p0), d(q, p1), d(q, p2)*. The* Centered Representation CR(G; A) *is the triple* [sign(A), D(A), R(G; A)]*.*

After fixing p<sup>0</sup> = 0, the matrix D(A) and sign(A) help reconstruct base vertices p1, p<sup>2</sup> ∈ <sup>R</sup> 2 , uniquely under rotation around 0. Any other q ∈ V (G) \ A is fixed relative to p0, p1, p<sup>2</sup> by its column in R(G; A). A positive sign of d(p<sup>i</sup> , p<sup>j</sup> ) indicates an edge between vertices p<sup>i</sup> , p<sup>j</sup> . This argument will later be formalized in Theorem [4.6\(](#page-5-1)b).

Example 3.4 (CRs for 2-vertex bases in R 2 ). *Let* G ⊂ R 2 *be the triangular cycle on* p<sup>1</sup> = (2, 0)*,* p<sup>2</sup> = (−1, 1)*,* p<sup>3</sup> = (−1, −1)*, so* O(G) = 0 *and all signed distances are positive, see Fig. [2](#page-4-1) (top left). For* A = (p1, p2)*,* sign(A) = sign 2 −1 0 1 = 1*. The distance matrix on* 0, p1, p<sup>2</sup> *is* D(p1, p2) = 0 −2 − √ 2 <sup>−</sup>2 0 √ 10 − √ 2 √ 10 0 *. Then* R(G; p1, p2) = −|p3| |p<sup>3</sup> − p1| |p<sup>3</sup> − p2| <sup>=</sup> − √ 2 √ 10 2 *. Then* CR(G; p1, p2) = [+1, D(p1, p2), R(G; p1, p2)]*. Replacing* p<sup>2</sup> *with* p3*, we find* sign(p1, p3) = sign 2 −1 0 −1  = −1*,* D(p1, p3) = 0 −2 − √ 2 <sup>−</sup>2 0 √ 10 − √ 2 √ 10 0 *, and* R(G; p1, p3) = −|p2| |p<sup>2</sup> − p1| |p<sup>2</sup> − p3| = − √ 2 √ 10 2 *. The final triple is* CR(G; p1, p3) = [−1, D(p1, p3), R(G; p1, p3)]*.*

Though a Centered Representation CR(G; p1, p2) will suffice to reconstruct G ⊂ R <sup>2</sup> uniquely under rigid motion

*257*

*264*

*266*

in Theorem [4.6\(](#page-5-1)b), CR(G; p1, p2) for all vertices p1, p<sup>2</sup> ∈ V (G) should be considered in a joint unordered collection below to guarantee the independence of points p1, p2.

Definition 3.5 (Nested Centered Distribution NCD(G; h)). *Let* G ⊂ R <sup>2</sup> *be any Euclidean graph with* m *unordered vertices and the center of mass at the origin* 0 ∈ R n*.*

*(a) The* Nested Centered Distribution NCD(G; 1) *of order 1 is the unordered set of Centered Representations* CR(G; p1) *from Definition [3.3](#page-3-0) for all vertices* p<sup>1</sup> ∈ V (G)*.*

*(b) For any vertex* p<sup>1</sup> ∈ V (G)*, the* Centered Distribution CD1(G; p1) *is the unordered set of* CR(G; p1, p2) *for all* p<sup>2</sup> ∈ V (G) \ {p1}*. The* Nested Centered Distribution NCD(G; 2) *of order 2 is the unordered set of* CD1(G; p1) *for all vertices* p<sup>1</sup> ∈ V (G)*, see Fig. [2](#page-4-1) (top). The* mirror image NCD(G; 2) *is obtained from* NCD(G; 2) *by reversing* sign(p1, p2) *of* 2 × 2 *determinants in all* CR(G; p1, p2)*.*

The nested structure of NCD(G; 2) helps identify edges between all vertices from G. After any vertex q ∈ V (G) \ {p1, p2} is uniquely located by using one CR(G; p1, p2), we can use unsigned distances to associate any such q with its unique CR(G; p1, q) in the collection {CR(G; p1, pi)} for all p<sup>i</sup> ∈ V (G) \ {p1}. The resulting CR(G; p1, q) contains signed distances and hence detects edges from q to all other vertices, see details in the proof of Theorem [4.6\(](#page-5-1)b).

![](_page_4_Diagram_13.jpeg)

Figure 2. Top: building the Nested Centered Distribution NCD in Definition [3.5](#page-4-0) from Centered Representations in Definition [3.3](#page-3-0) with metrics in section [4.](#page-4-2) Bottom: hierarchy of graph invariants.

SRD(G) can be considered NCD(G; 0) of order 0, containing signed distances from the center of mass p<sup>0</sup> to all vertices of G, additionally written in increasing order.

#### 4. Continuous metrics on graph invariants

When points 0 ∪ A = (p0, p1, p2) ⊂ <sup>R</sup> <sup>2</sup> pass through a degenerate configuration in a straight line, i.e. p1, p<sup>2</sup> become collinear, sign(A) discontinuously changes. To guarantee the Lipschitz continuity, we multiply such a sign by the strength σ below, which smooths the sign change, while the area of the triangle on p0, p1, p<sup>2</sup> is not Lipschitz continuous.

Definition 4.1 (*strength* σ(C)). *Any triple* C = {p0, p1, p2} ⊂ <sup>R</sup> <sup>2</sup> *defines a triangle with inter-point distances* a, b, c*, and half-perimeter* p = 2 (a + b + c)*. The* strength *is* <sup>σ</sup>(C) = (<sup>p</sup> <sup>−</sup> <sup>a</sup>)(<sup>p</sup> <sup>−</sup> <sup>b</sup>)(<sup>p</sup> <sup>−</sup> <sup>c</sup>) p 2 *.*

Lemma 4.2 (Theorem 4.4 in [\(Widdowson & Kurlin,](#page-11-3) [2023\)](#page-11-3)). *Let* B *be obtained from a set* C ⊂ R <sup>2</sup> *of 3 points by perturbing every point within its* ε*-neighborhood. Then* <sup>|</sup>σ(B) <sup>−</sup> <sup>σ</sup>(C)| ≤ <sup>2</sup>ελ<sup>2</sup> *for* <sup>λ</sup><sup>2</sup> = 2√ 3*.*

The strength σ(A) will be normalized by λ<sup>2</sup> below to guarantee the final Lipschitz constant 2 for a metric in Theorem [4.6\(](#page-5-1)c). For any k × k matrices M, N of real numbers, the metric L<sup>∞</sup> is max i,j=1,...,k |Mij − Nij |. The *bottleneck* distance between any clouds A, B of (the same number of) m unordered points in a metric space with a distance d is W∞(A, B) = min bijections g:A→B max p∈A d(g(p), p).

Definition 4.3 (max metric M<sup>∞</sup> on CRs). *Let Euclidean graphs* G, F ⊂ R <sup>n</sup> *have* m *unordered vertices.*

*(a) For order* h = 1*, take any base vertices* p ∈ V (G) *and* q ∈ V (F)*. Define the* max metric M∞(CR(G; p), CR(F; q) *as the maximum of* | |p| − |q| | *and the bottleneck distance* W<sup>∞</sup> *between the fixed clouds of unordered points* { (−|p ′ |, d(p ′ , p)) | p ′ ∈ V (G) − {p} } *and* { (−|q ′ |, d(q ′ , q)) | q ′ ∈ V (F) − {q} } *in* <sup>R</sup> 2 *.*

*(b) For order* h = 2*, take any base sequences* A ⊂ V (G) *and* B ⊂ V (F) *of two vertices. Consider the* m − 2 *columns of* R(G; A) *from Definition [3.3](#page-3-0) as a cloud of* m − 2 *unordered points in* R 2 *, also for* R(F; B)*. The* max metric M∞(CR(G; A), CR(F; B)) *is the maximum of* <sup>2</sup> λ<sup>2</sup> |sign(A)σ(0 ∪ A) − sign(B)σ(0 ∪ B)|*,* L∞(D(A), D(B))*, and* W∞(R(G; A), R(F; B))*.*

The maximum of several distances in Definition [4.3](#page-4-3) is needed to guarantee the first metric axiom, i.e. M∞(CR(G; A), CR(F; B)) = 0 should imply that 0 ∪ A should be exactly matched by rotation with 0 ∪ B and then CR(G; A) = CR(F; B) up to a permutation of columns will imply that G coincides with F, see Lemma [D.7.](#page-21-0)

To get a metric on Nested Centered Distributions, we will use the distance on bipartite graphs whose edge weights are the max metrics M<sup>∞</sup> on Centered Representations.

Definition 4.4 (Bottleneck Matching Distance BMD(Γ)).

278

289 290

294

296

298

300

304

306

308 309

311

314 315 316

318

324

326

328

*Let* Γ *be a complete bipartite graph with* m *white vertices and* m *black vertices so that every white vertex is connected to every black vertex by a single edge* e *of a weight* w(e) ≥ 0*. A* vertex matching *of the graph* Γ *is a collection* E *of* m *disjoint edges with* 2m *distinct vertices. The* weight W(E) = max e∈E w(e) *is the largest weight of an edge in* E*. The* Bottleneck Matching Distance BMD(Γ) = min E W(E) *is the minimum weight of a vertex matching* E *of* Γ*.*

Since a graph Γ is complete bipartite, any edge from a vertex matching E in Γ joins a white vertex with a black vertex. Then BMD(Γ) is minimized for all bijections E between all white vertices and all black vertices of Γ.

Definition 4.5 (Nested Bottleneck Metric NBM on NCDs). *Let* G, F ⊂ R <sup>2</sup> *be any graphs on* m *unordered vertices.*

*(a) For order* h = 1*, the* Nested Bottleneck Metric NBM(NCD(G; 1), NCD(F; 1)) *is the max metric* M∞(CR(G; p), CR(F; β(p))) *minimized for all bijections* β : V (G) → V (F) *between vertices of* G *and* F*.*

*(b) For order* h = 2*, any base vertices* p<sup>1</sup> ∈ V (G) *and* q<sup>1</sup> ∈ V (F)*, let the complete bipartite graph* Γ(G; p1; F; q1) *have* m − 1 *white vertices and* m − 1 *black vertices representing* CR(G; p1, p2) *and* CR(F; q1, q2) *for all* p<sup>2</sup> ∈ V (G)−{p1} *and* q<sup>2</sup> ∈ V (F)− {q1}*, respectively. Set the* weight w(e) *of an edge* e *joining the vertices represented by* CR(G; p1, p2) *and* CR(F; q1, q2) *as the max metric* M<sup>∞</sup> *between these distributions, see Definition [4.3.](#page-4-3) Then Definition [4.4](#page-4-4) gives the bottleneck matching distance* BMD(Γ(G; p1; F; q1))*.*

*Let the complete bipartite graph* Γ(G, F) *have weight* BMD(Γ(G; p1; F; q1)) *on each edge connecting vertices representing* p<sup>1</sup> ∈ V (G) *and* q<sup>1</sup> ∈ V (F)*. The* Nested Bottleneck Metric NBM(NCD(G; 2), NCD(F; 2)) *is the Bottleneck Matching Distance* BMD(Γ(G, F))*.*

SRD(G) coincides with NCD(G; 0) after sorting, so NBM can be defined as L∞(SRD(G), SRD(F)) for order h = 0. The metrics W∞, M∞, NBM compare objects of the same size. To compare graphs with different numbers of vertices, M<sup>∞</sup> in Definition [4.5](#page-5-0) can be replaced with Earth Mover's Distance EMD in Definition [C.2.](#page-17-0) All metric axioms and main Theorem [4.6](#page-5-1) below are proved in appendices [C](#page-16-0) and [D](#page-18-0) for any dimension n ≥ 2 and orders 1 ≤ h ≤ n.

Theorem 4.6 (NCD solves Problem [1.1\)](#page-1-0). *(a) The Nested Centered Distribution* NCD(G; h) *in Definition [3.5](#page-4-0) is invariant under any rigid motion for all Euclidean graph* G *on* m *unordered vertices and can be computed in time* O(n <sup>2</sup>m<sup>h</sup>+1) *with space* O(n <sup>3</sup> + hm<sup>h</sup>+1) *for* h ≤ n = 2*.*

*(b)* NCD(G; 2) *is a complete invariant of all graphs* G ⊂ R <sup>2</sup> *under rigid motion from the group* SE(2)*.*

*(c) Perturbing each vertex of a graph* G ⊂ R <sup>2</sup> *within its* ε*neighborhood changes* NCD(G; h) *up to* 2ε *in both metrics*

NBM *and* EMD *for any order* h = 1, 2*.*

*(d) For any graphs* G, F ⊂ R <sup>2</sup> *on* m *unordered vertices, the metrics* NBM *and* EMD *between the invariants* NCD(G; h) *and* NCD(F; h) *is computed in time* O(m<sup>2</sup>h+1.<sup>5</sup> log<sup>h</sup>+1 m) *with space* O(m<sup>2</sup>h+1 log<sup>h</sup>−<sup>1</sup> m) *for* h ≤ n = 2*.*

Theorem [4.6\(](#page-5-1)b) implies that any graphs G, F ⊂ R 2 are related by rigid motion *if and only if* NCD(G; 2) = NCD(F; 2). This equality is interpreted as a bijection NCD(G; n) → NCD(F; n) matching all CRs, which is equivalent to NBM = 0 by the first metric axiom. Since every CR can be stored in a vector form, the complete invariant NCD(G; 2) for n = 2 can be considered vectorial.

Table [2](#page-5-2) emphasizes that most graphs should be first compared (or represented for machine learning) by simpler and faster invariants, so the complete NCD(G; n) is used only in rare cases but is still needed to distinguish all graphs.

Table 2. Invariants and metrics on graphs G ⊂ R <sup>2</sup> with m unordered vertices: from the fastest (linear-time) to complete.

| INVARIANT TIME         | METRIC | TIME |     |           |
|------------------------|--------|------|-----|-----------|
| SRD( G ) O ( m log m ) | L      | ∞ O  | ( m | )         |
| SPD( G ) O ( m 2       |        |      |     |           |
| )                      | L      | ∞ O  | ( m | 2         |
| PDD( G ) O ( m 2       |        |      |     |           |
| log m                  | ) EMD  | O    | ( m | 3         |
| NCD( G ; 1) O ( m 2    |        |      |     |           |
| )                      | NBM    | O    | ( m | 3 5       |
|                        |        |      |     | log 2 m ) |
| NCD( G ; 2) O ( m 3    |        |      |     |           |
| )                      | NBM    | O    | ( m | 5 5       |
|                        |        |      |     | log 3 m ) |

Example 4.7 (version of Theorem [4.6\(](#page-5-1)b) for n = 1). *For a graph* G ⊂ <sup>R</sup> *with the center of mass* O(G) = 0*, take any base vertex* p ∈ G*. Then* sign(p) *is the usual sign of* p ∈ <sup>R</sup>*,* D(p) *is the signed distance* −|p|*,* R(G; p) *is the* 2 × (m − 1) *matrix whose column for any vertex* q ∈ V (G)− {p} *consists of the signed distances* d(q, 0) = −|q| *and* d(q, p) = ±|q − p|*, where the plus sign* + *indicates an edge between* q, p*, while the minus sign* − *means no edge.*

*For order* h = 1*, the Centered Representation is the pair* CR(G; p) = [sign(p), −|p|, R(G; p)]*. The base vertex* p *is fixed in the line* <sup>R</sup> *by* sign(p) *and* |p|*. Any other vertex* q ∈ V (G) − {p} *is uniquely determined in* <sup>R</sup> *by its Euclidean distances* |q|*,* |q − p| *to the origin and the already fixed* p*. The location of any point* q ∈ R *is characterized by* sign(q) *and* |q|*, which helps unambiguously identify its Centered Representation* CR(G; q) *in the unordered collection* NCD(G; 1) *of all these* CR*s. The signs of* d(q, q′ ) *in each* R(G; q) *determine the presence or absence of an edge of* G ⊂ <sup>R</sup> *between any vertices* q, q′ ∈ V (G)*.*

## 5. Experiments on largest molecular databases

The world's largest databases of 3D molecular geometry are QM9 (130K+ entries) [\(Ramakrishnan et al.,](#page-10-19) [2014\)](#page-10-19) and GD (GEOM drugs of 31M+ entries) [\(Axelrod & Gomez-](#page-8-18)

334

336

338

351

354

356

358

360 361

364

366

368

371

374

378

[Bombarelli,](#page-8-18) [2022\)](#page-8-18), which have hundreds of 3D conformers of *unordered* atoms for each of 621 and 61607 chemical compositions, respectively. The Protein Data Bank has backbones of *ordered* atoms classified by simpler invariants [\(Anosova et al.,](#page-8-19) [2025\)](#page-8-19). All experiments took a few hours on Ryzen 9 3950X 3.5 GHz, 64 MB of L3 cache, RAM 82GB.

The ICML guide for reviewing application-driven ML says that "novel ideas that are simple to apply may be especially valuable". To demonstrate the chemical importance of the linear-time invariant SRD, we extracted clouds of k = 10 neighbors around every atom, see their counts in Table [3.](#page-6-0)

Table 3. Counts of atoms by chemical elements in QM9 (2,407,753 atoms), GD0 (GEOM drugs 0th conformers, 12,917,980 atoms).

| QM9:      | H QM9:    | C QM9:  | N QM9:  | O QM9: F  |
|-----------|-----------|---------|---------|-----------|
| 1,230,122 | 846,557   | 139,764 | 187,996 | 3,314     |
| GD0: H    | GD0: C    | GD0:    | N GD0:  | O GD0: F  |
| 5,660,986 | 5,267,096 | 842,562 | 854,400 | 64,299    |
| GD0: P    | GD0: S    | GD0:    | Cl GD0: | Br GD0: I |
| 1,350     | 159,648   | 53,404  | 14,010  | 225       |

Though the data was skewed towards more popular elements H (hydrogen) and C (carbon), a default network in TensorFlow with 80/20 split for train/test achieved over 98% accuracy in predictions of the chemical element of a central atom by distances to only k = 3 nearest neighbours, see Table [4.](#page-6-1) Appendix [A](#page-12-0) has all implementation details.

Table 4. Accuracies in percentages for predicting the chemical element of a central atom by a 4-layer network using *only the* k *shortest distances* to atomic neighbors within a molecular graph.

| data | $k = 2$ | $k = 3$ | $k = 4$ | $k = 5$ | $k = 6$ |
|------|---------|---------|---------|---------|---------|
| QM9  | 94.63   | 98.64   | 98.24   | 98.54   | 98.77   |

![Chemical structures of near-duplicates and duplicates.]()The image shows three chemical structures of near-duplicates and duplicates. The first structure on the left is a single backbone structure with a red chemical alternate (a single branch) at the top center. The second structure on the right is a duplicate structure with two branches of the same single branch at the top center. The third structure on the bottom is a duplicate structure with two branches of the same single branch at the top center. The text 'near-duplicates' is positioned below the two duplicate structures.

In chemistry, both ML and non-ML predictions of elements achieved only 86% on similar size data, see Table [7](#page-12-1) summarized in [\(Vasylenko et al.,](#page-11-7) [2025\)](#page-11-7), because the underlying descriptors were not invariant, e.g. under permutations of atoms, which creates exponentially many representations of the same molecule, incomplete, or their similarities failed the triangle axiom, e.g. see [\(Steck et al.,](#page-11-8) [2024\)](#page-11-8).

High accuracies in Table [4](#page-6-1) are rigorously explained by the cascade comparisons on all atomic clouds (environments) from QM9. Split all clouds from by the 1st distance (to the nearest neighbor of a central atom p) rounded to 3 decimal places in A˚ . This is a typical experimental precision, where [1](#page-8-18)A˚ = 10−<sup>10</sup>m is approximately the smallest interatomic distance. Second, split each subset with equal 1st distances by 2nd distances, and so on up to k = 5 distances. All 2.4M+ atomic clouds of different elements in QM9 were separated by the shorest distances to only 4 atomic neighbors.

The hierarchy of invariants in Fig. [2](#page-4-1) and Table [2](#page-5-2) transparently explained the reconstruction of chemical elements from distances to k nearest neghbors and inspired the harder task to reconstruct a chemical composition from a moleculelevel (not atomwise) invariant of only atomic centers.

For molecular graphs from QM9, we computed the pseudometric L<sup>∞</sup> (max absolute difference of corresponding coordinates) on all 873,527,974 pairs of SRDs, then 8,735,279 distances L<sup>∞</sup> on the stronger SPDs for the 1% closest pairs, then 87,352 EMDs on PDDs for the 1% closest pairs, distances NBM on NCD(G; 1) and NCD(G; 2) for the top 10K closest pairs, and 64 NBMs on complete NCD(G, 3).

The invariants in Table [5](#page-6-2) distinguish all chemically different molecules with NBM on complete invariants giving the largest separation. All chemical compositions in QM9 and GD were distinguished by the vector SRD of Euclidean distances (rounded to 3 decimal places in A˚ ) from the molecular center of mass to 5 and 7 farthest atoms, respectively.

This transparent reconstruction of the full chemistry from precise enough atomic geometry gives hope to rigorously infer other molecular properties from geometric invariants.

Table 5. Chemically different molecules (given by QM9 ids) are geometrically distinguished by invariant metrics, see Fig. [3](#page-6-3) (right).

#### smallest distances in A, molecule A ˚ ̸= molecule B

SRD, L<sup>∞</sup> = 0.021, H3C4N3O<sup>2</sup> (131923)̸=H4C5N2O(5365) SPD, L<sup>∞</sup> = 0.055, H3C4N<sup>5</sup> (123533)̸=H3C5N3O(24547) PDD, EMD = 0.051, H3C4N<sup>5</sup> (123533)̸=H3C5N3O(24521) NCD, NBM = 0.071, H4C5N<sup>4</sup> (123532)̸=H4C6N2O(24513)

Figure 3. Left: the smallest NBM ≈ 0.07A˚ on NCD(G; 3) for chemically different molecules 123533 and 24521. Right: nearduplicate (almost flat) molecules 123532 and 24513 have the same composition and tiny EMD ≈ 2.37 × 10−<sup>7</sup>A˚ (not distinguishing mirror images) but a 100× higher NBM ≈ 2.95 × 10−<sup>5</sup>A. ˚

For QM9 molecul graphs, Fig. [4](#page-7-0) and [9](#page-15-0) NBM distances for different NCD invariants of orders h = 1, 2, 3.

![](_page_7_Figure_1.jpeg)

Figure 4. Each dot is a comparison of molecular graphs from QM9: x = NBM on NCD(G; 1) vs y = NBM on NCD(G; 2).

### 6. Discussion: conclusions and limitations

The comparisons of molecular graphs from QM9 and GD imply that all chemically different molecules are rigidly different, see the smallest distance NBM ≈ 0.07A˚ on complete invariants in Table [5.](#page-6-2) So the map {molecules} → {graphs on atomic centers (without chemical elements)} is injective on rigid classes and can be inverted on its image.

Hence the most important property (chemical composition) is reconstructable from precise enough geometry. Using only a few radial distances (5 at the atomic level and 7 at the molecular level, rounded to 3 decimal places) for uniquely identifying all chemical elements in QM9 and GD demonstrates the transparency of application-driven ML.

The solution to Problem [1.1](#page-1-0) settled the long-standing challenge of properly defining a *molecular structure*. A traditional approach is to describe such a structure as "a set of unlabeled configurations that are relatively similar to each other", quoted from the paragraph to the left of the caption of Fig. 1 in [\(Lang et al.,](#page-9-19) [2024\)](#page-9-19). If this 'similarity' is treated as an equivalence allowing perturbations of atoms up to a positive threshold, sufficiently many perturbations can make all molecules (of the same number of atoms) equivalent by the transitivity axiom. A justified way to resolve this paradox is to embrace uncertainty and continuously quantify this similarity not by ignoring any perturbations up to a threshold but by computing an exact distance satisfying all metric axioms and Lipschitz continuity in Problem [1.1.](#page-1-0)

The question of whether to put close neighbors like nearduplicates in Fig. [3](#page-6-3) (left) into one cluster of the "same" molecules is rather administrative similar to assigning close houses to one village (cluster) instead of different ones.

Studying molecules by fixing a composition is similar to drawing artificial boundaries between countries on Earth. Because some molecules of different compositions have close shapes as in Fig. [5,](#page-6-2) they should have similar properties. Now any properties of molecules should be possible to predict only from the complete invariant NCD(G; 3) even without chemistry in the same way as any precise geographic location uniquely determines all physical properties of this place such as the average annual temperature. Chemical compositions can be still helpful similar to the location's altitude, which easier predicts (say) the average temperature than theoretically sufficient geographic coordinates.

Any vertex p and edge of G can have an *attribute* and a *weight* respected by any isometry that maps one graph to another. These vertex attributes and edge weights can be incorporated as extra columns and rows in CRs from Definition [3.3,](#page-3-0) and then incorporated into NCD and NBM. We can compare graphs of different numbers of vertices because EMD works for both PDD and NCD as weighted distributions of any finite size. This comparison splits the vertices from V (G) into parts (subvertices) that are optimally 'transported' to a splitting of another vertex set V (F).

The main contribution is Theorem [4.6](#page-5-1) and its extension in Theorem [D.1](#page-18-1) to all dimensions n ≥ 2 fully solving Problem [1.1.](#page-1-0) The limitation is the time O(n <sup>2</sup>m<sup>n</sup>+1) of the complete invariant NCD(G; n) of any graphs G ⊂ <sup>R</sup> <sup>n</sup>. For a fixed dimension n, this polynomial complexity resolves two exponential-size challenges: m! permutations of m unordered vertices and up to 2 <sup>m</sup>(m−1)/<sup>2</sup> non-isometric graphs with up to m(m − 1)/2 edges on m fixed vertices in <sup>R</sup> n.

In practice, all comparisons and property predictions can start from much faster (linear-time) invariants SRD and only in cases of close distances (potential confusions) progress to stronger invariants SPD,PDD, NCD. This hierarchical (cascade) computation can better address the curse of dimensionality instead of the one-size-fits-all approach.

A map f : objects → descriptors → properties is invertible only if objects are faithfully represented by complete invariants. Any non-invariant maps a single object to (usually infinitely) many values or representations. Any incomplete invariant can fail to differentiate between objects with different properties. Hence a *generative* approach (inverting f above) can succeed only after the *discriminative* problem is solved. The space GRS(<sup>R</sup> 3 ; m) of rigid classes of all graphs on m vertices in R 3 contains all possible shapes of molecules (all already known and also all *not yet discovered* ones). The complete invariant NCD(G) of G ⊂ <sup>R</sup> 3 defines geographic-style coordinates on a continuous map of GRS(<sup>R</sup> 3 ; m) containing QM9 and GD. Since the space GRS(<sup>R</sup> 3 ; m) is high-dimensional, we really need complete invariants to separate all known molecules and look for unexplored gaps containing new future molecules.

- 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. References Abbe, E. and Sandon, C. On the universality of deep learning. *Advances in Neural Information Processing Systems*, 33:20061–20072, 2020. Agarwal, P. K., Fox, K., Nath, A., Sidiropoulos, A., and Wang, Y. Computing the gromov-hausdorff distance for metric trees. *ACM Transactions on Algorithms*, 14(2): 1–20, 2018. Akhtar, N. and Mian, A. Threat of adversarial attacks on deep learning in computer vision: A survey. *IEEE Access*, 6:14410–14430, 2018. Amir, T., Gortler, S., Avni, I., Ravina, R., and Dym, N. Neural injective functions for multisets, measures and graphs via a finite witness theorem. *Advances in Neural Information Processing Systems*, 36, 2024. Anosova, O., Gorelov, A., Jeffcott, W., Jiang, Z., and Kurlin,
  - V. A complete and bi-continuous invariant of protein backbones under rigid motion. *MATCH Communications in Mathematical and in Computer Chemistry (to appear), arxiv:2410.08203*, 2025. Antunes, L. M., Grau-Crespo, R., and Butler, K. T. Distributed representations of atoms and materials for machine learning. *npj Computational Materials*, 8(1):44, 2022. Arvind, V. and Rattan, G. The parameterized complexity of geometric graph isomorphism. *Algorithmica*, 75:258– 276, 2016. Axelrod, S. and Gomez-Bombarelli, R. Geom, energyannotated molecular conformations for property prediction and molecular generation. *Scientific Data*, 9(1):185, 2022. Babai, L. Canonical form for graphs in quasipolynomial time: preliminary report. In *Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing*, pp. 1237–1246, 2019. Batzner, S., Musaelian, A., Sun, L., Geiger, M., Mailoa,
- J. P., Kornbluth, M., Molinari, N., Smidt, T. E., and Kozinsky, B. E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. *Nature communications*, 13(1):2453, 2022. Bonchev, D. *Chemical graph theory: introduction and fundamentals*, volume 1. CRC Press, 1991. Brecheteau, C. A statistical test of isomorphism between ´ metric-measure spaces using the distance-to-a-measure signature. *Electronic J Statistics*, 13:795–849, 2019. Bronstein, M. M., Bruna, J., LeCun, Y., Szlam, A., and Vandergheynst, P. Geometric deep learning: going beyond Euclidean data. *IEEE Signal Processing Magazine*, 34 (4):18–42, 2017. Bronstein, M. M., Bruna, J., Cohen, T., and Velickovi ˇ c,´
  - P. Geometric deep learning: grids, groups, graphs, geodesics, and gauges. *arXiv:2104.13478*, 2021. Buchin, M., Chambers, E., Fang, P., Fasy, B. T., Gasparovic, E., Munch, E., and Wenk, C. Distances between immersed graphs: Metric properties. *La Matematica*, pp. 1–26, 2023. Bunch, J. R. and Hopcroft, J. E. Triangular factorization and inversion by fast matrix multiplication. *Mathematics of Computation*, 28(125):231–236, 1974. Chen, C., Ye, W., Zuo, Y., Zheng, C., and Ong, S. P. Graph networks as a universal machine learning framework for molecules and crystals. *Chemistry of Materials*, 31(9): 3564–3572, 2019. Chew, P. and Kedem, K. Improvements on geometric pattern matching problems. In *Scandinavian Workshop on Algorithm Theory*, pp. 318–325, 1992. Chew, P., Dor, D., Efrat, A., and Kedem, K. Geometric pattern matching in d-dimensional space. *Discrete & Computational Geometry*, 21(2):257–274, 1999. Choo, H. Y., Wee, J., Shen, C., and Xia, K. Fingerprintenhanced graph attention network (fingat) model for antibiotic discovery. *Journal of Chemical Information and Modeling*, 2023. Cohen, T. S., Geiger, M., and Weiler, M. A general theory of equivariant cnns on homogeneous spaces. *Advances in neural information processing systems*, 32, 2019. Colbrook, M. J., Antun, V., and Hansen, A. C. The difficulty of computing stable and accurate neural networks: On the barriers of deep learning and Smale's 18th problem. *Proc. National Academy of Sciences*, 119(12):e2107151119, 2022. Dekster, B. V. and Wilker, J. B. Edge lengths guaranteed to form a simplex. *Archiv der Mathematik*, 49(4):351–366, 1987.

#### Impact Statement

- 495 496 497 498 499 500 504 506 508 509 511 514 515 516 518 524 526 528 531 534 536 538 540 541 542 543 544 545 546 547 548 549 Delle Rose, V., Kozachinskiy, A., Rojas, C., Petrache, M., and Barcelo, P. Three iterations of (d- 1)-wl test dis- ´ tinguish non isometric clouds of d-dimensional points. *Advances in Neural Information Processing Systems*, 36, 2024. Deng, C., Litany, O., Duan, Y., Poulenard, A., Tagliasacchi, A., and Guibas, L. J. Vector neurons: A general framework for so(3)-equivariant networks. In *Proceedings of the International Conference on Computer Vision*, pp. 12200–12209, 2021. Deza, E. and Deza, M. M. *Encyclopedia of distances*. Springer, 2009. Dong, Y., Liao, F., Pang, T., Su, H., Zhu, J., Hu, X., and Li, J. Boosting adversarial attacks with momentum. In *Computer vision and pattern recognition*, pp. 9185–9193, 2018. Duvenaud, D. K., Maclaurin, D., Iparraguirre, J., Bombarell, R., Hirzel, T., Aspuru-Guzik, A., and Adams, R. P. Convolutional networks on graphs for learning molecular fingerprints. *Advances in neural information processing systems*, 28, 2015. Dym, N. and Gortler, S. J. Low-dimensional invariant embeddings for universal geometric learning. *Foundations of Computational Mathematics*, pp. 1–41, 2024. Efrat, A., Itai, A., and Katz, M. J. Geometry helps in bottleneck matching and related problems. *Algorithmica*, 31(1):1–28, 2001. Feynman, R. *The Feynman lectures on physics. Chapter 1: atoms in motion*, volume 1. 1971. Fisikopoulos, V. and Penaranda, L. Faster geometric algorithms via dynamic determinant computation. *Computational Geometry*, 54:1–16, 2016. Fredman, M. L. and Tarjan, R. E. Fibonacci heaps and their uses in improved network optimization algorithms. *Journal of the ACM*, 34(3):596–615, 1987. Fuchs, F., Worrall, D., Fischer, V., and Welling, M. Se(3) transformers: 3d roto-translation equivariant attention networks. *Advances in neural information processing systems*, 33:1970–1981, 2020. Gao, X., Hu, W., and Qi, G.-J. Graphter: Unsupervised learning of graph transformation equivariant representations via auto-encoding node-wise transformations. In *Proceedings of Computer Vision and Pattern Recognition*, pp. 7163–7172, 2020. Goldberg, A. and Tarjan, R. Solving minimum-cost flow problems by successive approximation. In *Proceedings of STOC*, pp. 7–18, 1987. Goodrich, M. T., Mitchell, J. S., and Orletsky, M. W. Approximate geometric pattern matching under rigid motions. *Transactions on Pattern Analysis and Machine Intelligence*, 21(4):371–379, 1999. Goyal, A., Law, H., Liu, B., Newell, A., and Deng, J. Revisiting point cloud shape classification with a simple and effective baseline. In *International Conference on Machine Learning*, pp. 3809–3820, 2021. Grinberg, D. and Olver, P. J. The n body matrix and its determinant. *SIAM Journal on Applied Algebra and Geometry*, 3(1):67–86, 2019. Grohe, M. and Schweitzer, P. The graph isomorphism problem. *Communications of the ACM*, 63(11):128–134, 2020. Guo, C., Gardner, J., You, Y., Wilson, A. G., and Weinberger,
  - K. Simple black-box adversarial attacks. In *International Conference on Machine Learning*, pp. 2484–2493, 2019. Helfgott, H. A., Bajpai, J., and Dona, D. Graph isomorphisms in quasi-polynomial time. *arXiv:1710.04574*, 2017. Hopcroft, J. E. and Karp, R. M. An nˆ5/2 algorithm for maximum matchings in bipartite graphs. *SIAM Journal on Computing*, 2(4):225–231, 1973. Hordan, S., Amir, T., Gortler, S. J., and Dym, N. Complete neural networks for euclidean graphs. In *AAAI Conference on Artificial Intelligence*, volume 38 (11), pp. 12482–12490, 2024. Horn, R. A. and Johnson, C. R. *Matrix analysis*. Cambridge University Press, 2012. Huttenlocher, D. P., Klanderman, G. A., and Rucklidge, W. J. Comparing images using the Hausdorff distance. *Transactions on pattern analysis and machine intelligence*, 15 (9):850–863, 1993. Kiefer, S., Ponomarenko, I., and Schweitzer, P. The weisfeiler–leman dimension of planar graphs is at most 3. *Journal of the ACM*, 66(6):1–31, 2019. Kondor, R. and Trivedi, S. On the generalization of equivariance and convolution in neural networks to the action of compact groups. In *International Conference on Machine Learning*, pp. 2747–2755, 2018. Laidlaw, C. and Feizi, S. Functional adversarial attacks. *Adv. Neural Information Proc. Systems*, 32, 2019. Lang, L., Cezar, H. M., Adamowicz, L., and Pedersen, T. B. Quantum definition of molecular structure. *Journal of the American Chemical Society*, 146(3):1760–1764, 2024.

- 551 554 556 558 560 564 566 568 571 574 576 578 580 581 582 583 584 585 586 587 588 589 590 594 596 598 600 601 602 603 Leman, A. and Weisfeiler, B. A reduction of a graph to a canonical form and an algebra arising during this reduction. *Nauchno-Technicheskaya Informatsiya*, 2(9):12–16, 1968. Li, Z., Wang, X., Huang, Y., and Zhang, M. Is distance matrix enough for geometric deep learning? *arXiv:2302.05743*, 2023. Liberti, L. and Lavor, C. *Euclidean distance geometry*. Springer, 2017. Lim, S., Memoli, F., and Smith, Z. The gromov–hausdorff ´ distance between spheres. *Geometry & Topology*, 27(9): 3733–3800, 2023. Maennel, H., Unke, O. T., and Muller, K.-R. Complete ¨ and efficient covariants for 3d point configurations with application to learning molecular quantum properties. *arXiv:2409.02730*, 2024. Majhi, S. and Wenk, C. Distance measures for geometric graphs. *arXiv:2209.12869*, 2022. Majhi, S., Vitter, J., and Wenk, C. Approximating gromovhausdorff distance in euclidean space. *Computational Geometry*, 116:102034, 2024. Maron, H., Ben-Hamu, H., Serviansky, H., and Lipman, Y. Provably powerful graph networks. *Advances in neural information processing systems*, 32, 2019. Melter, R. A. and Tomescu, I. Metric bases in digital geometry. *Computer vision, graphics, and image Processing*, 25(1):113–121, 1984. Memoli, F. Gromov–Wasserstein distances and the metric ´ approach to object matching. *Foundations of computational mathematics*, 11:417–487, 2011. Memoli, F., Smith, Z., and Wan, Z. The Gromov-Hausdorff ´ distance between ultrametric spaces: its structure and computation. *arXiv:2110.03136*, 2021. Merchant, A., Batzner, S., Schoenholz, S. S., Aykol, M., Cheon, G., and Cubuk, E. D. Scaling deep learning for materials discovery. *Nature*, 624(7990):80–85, 2023. Nemec, L. Principal component analysis (pca): A physically intuitive mathematical introduction. [https://towardsdatascience.com/](https://towardsdatascience.com/principal-component-analysis-pca-8133b02f11bd) [principal-component-analysis-pca-8133b02f11bd](https://towardsdatascience.com/principal-component-analysis-pca-8133b02f11bd), 2022. Nigam, J., Pozdnyakov, S. N., Huguenin-Dumittan, K. K., and Ceriotti, M. Completeness of atomic structure representations. *APL Machine Learning*, 2(1), 2024. Nikolentzos, G., Meladianos, P., and Vazirgiannis, M. Matching node embeddings for graph similarity. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 31, 2017. Oliynyk, A. O., Antono, E., Sparks, T. D., Ghadbeigi, L., Gaultois, M. W., Meredig, B., and Mar, A. Highthroughput machine-learning-driven synthesis of fullheusler compounds. *Chemistry of Materials*, 28(20): 7324–7331, 2016. Pozdnyakov, S. N., Willatt, M. J., Bartok, A. P., Ortner, C., ´ Csanyi, G., and Ceriotti, M. Incompleteness of atomic ´ structure representations. *Phys. Rev. Lett.*, 125:166001, 2020. URL <arXiv:2001.11696>. Qi, G.-J. and Luo, J. Small data challenges in big data era: A survey of recent progress on unsupervised and semisupervised methods. *Transactions on Pattern Analysis and Machine Intelligence*, 44(4):2168–2187, 2020. Ramakrishnan, R., Dral, P. O., Rupp, M., and Von Lilienfeld,
  - O. A. Quantum chemistry structures and properties of 134 kilo molecules. *Scientific data*, 1(1):1–7, 2014. Rass, S., Konig, S., Ahmad, S., and Goman, M. Metricizing ¨ the euclidean space towards desired distance relations in point clouds. *IEEE Transactions on Information Forensics and Security*, 2024. Rieder, S. R., Oliveira, M. P., Riniker, S., and Hunenberger, ¨
  - P. H. Development of an open-source software for isomer enumeration. *Journal of Cheminformatics*, 15(1):10, 2023. Rubner, Y., Tomasi, C., and Guibas, L. The Earth Mover's Distance as a metric for image retrieval. *International Journal of Computer Vision*, 40(2):99–121, 2000. Sato, R., Cuturi, M., Yamada, M., and Kashima, H. Fast and robust comparison of probability measures in heterogeneous spaces. *arXiv:2002.01615*, 2020. Schmiedl, F. Computational aspects of the Gromov– Hausdorff distance and its application in non-rigid shape matching. *Discrete Comp. Geometry*, 57:854–880, 2017. Schoenberg, I. Remarks to Maurice Frechet's article "Sur la definition axiomatique d'une classe d'espace distances vectoriellement applicable sur l'espace de Hilbert. *Annals of Mathematics*, pp. 724–732, 1935. Shirdhonkar, S. and Jacobs, D. Approximate earth mover's distance in linear time. In *Conference on Computer Vision and Pattern Recognition*, pp. 1–8, 2008. Sippl, M. J. and Scheraga, H. A. Cayley-menger coordinates. *Proceedings of the National Academy of Sciences*, 83(8): 2283–2287, 1986.

- 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659 Steck, H., Ekanadham, C., and Kallus, N. Is cosinesimilarity of embeddings really about similarity? In *Companion Proceedings of the ACM on Web Conference 2024*, pp. 887–890, 2024. Tshitoyan, V., Dagdelen, J., Weston, L., Dunn, A., Rong, Z., Kononova, O., Persson, K. A., Ceder, G., and Jain, A. Unsupervised word embeddings capture latent knowledge from materials science literature. *Nature*, 571(7763):95– 98, 2019. Tu, E., Wang, Z., Yang, J., and Kasabov, N. Deep semisupervised learning via dynamic anchor graph embedding in latent space. *Neural Networks*, 146:350–360, 2022. Vasylenko, A., Antypov, D., Schewe, S., Daniels, L. M., Claridge, J. B., Dyer, M. S., and Rosseinsky, M. J. Digital features of chemical elements extracted from local geometries in crystal structures. *Digital Discovery*, 2025. Ward, L., Agrawal, A., Choudhary, A., and Wolverton, C. A general-purpose machine learning framework for predicting properties of inorganic materials. *npj Computational Materials*, 2(1):1–7, 2016. Weinhold, F. Metric geometry of equilibrium thermodynamics. *The Journal of Chemical Physics*, 63(6):2479–2483, 1975. Weston, L., Tshitoyan, V., Dagdelen, J., Kononova, O., Trewartha, A., Persson, K. A., Ceder, G., and Jain, A. Named entity recognition and normalization applied to large-scale information extraction from the materials science literature. *Journal of chemical information and modeling*, 59(9):3692–3702, 2019. Widdowson, D. and Kurlin, V. Resolving the data ambiguity for periodic crystals. *Advances in Neural Information Processing Systems*, 35:24625–24638, 2022. Widdowson, D. and Kurlin, V. Recognizing rigid patterns of unlabeled point clouds by complete and continuous isometry invariants with no false negatives and no false positives. In *Proceedings of CVPR*, pp. 1275–1284, 2023. Wilson, S. R., Cui, W., Moskowitz, J. W., and Schmidt,
  - K. E. Applications of simulated annealing to the conformational analysis of flexible molecules. *Journal of computational chemistry*, 12(3):342–349, 1991. Zhang, B., Fan, C., Liu, S., Huang, K., Zhao, X., Huang, J., and Liu, Z. The expressive power of graph neural networks: A survey. *IEEE Transactions on Knowledge and Data Engineering*, 37:1455–1474, 2024. Zhou, D.-X. Universality of deep convolutional neural networks. *Applied and computational harmonic analysis*, 48 (2):787–794, 2020. Zhou, Q., Tang, P., Liu, S., Pan, J., Yan, Q., and Zhang, S.-
    - C. Learning atoms for materials discovery. *Proceedings of the National Academy of Sciences*, 115(28):E6411– E6417, 2018.

![](_page_12_Figure_6.jpeg)

694

696

698

700

704

706

708 709

711

#### A. Extra details of experiments on the world's largest 3D molecular databases QM9 and GD

The default 4-layer network from TensorFlow used a "sequential" mode, 3 epochs, and the settings in Table [6.](#page-12-2)

Table 6. Parameters of the TensorFlow network for predictions in Table [4.](#page-6-1)

| L AYER ( TYPE )     | O UTPUT S HAPE | NUMBER OF PARAMETERS |
|---------------------|----------------|----------------------|
| DENSE (D ENSE )     | (N ONE , 32)   | 352                  |
| BATCH NORMALIZATION | (N ONE , 32)   | 128                  |
| RE LU (R E LU)      | (N ONE , 32)   | 0                    |
| DENSE 1 (D ENSE )   | (N ONE , 5)    | 165                  |

Past maps of QM9 in Fig. [5](#page-12-3) based on eigenvalues are too dense without clear separation. Even if we zoom in, these two or three incomplete invariants will not provide any extra separation. The complete invariants NDP contain much more geometric information.

Figure 5. Left: each dot represents one QM9 molecule whose atomic cloud has two largest roots l<sup>1</sup> ≥ l<sup>2</sup> of eigenvalues (moments of inertia [\(Nemec,](#page-10-20) [2022\)](#page-10-20) or elongations in two principal directions) in Angstroms (1A˚ = 10−<sup>10</sup>m ≈ smallest interatomic distance). The color represents the free energy G characterizing molecular stability. Right: each dot represents one QM9 molecule whose atomic cloud has coordinates x, y expressed via the roots l<sup>1</sup> ≥ l<sup>2</sup> ≥ l<sup>3</sup> ≥ 0 of three eigenvalues.

Fig. [7](#page-13-0) shows the simplest geographic-style map of QM9 as a finite sample within S 29 m=3 GRS(<sup>R</sup> 3 ; m) projected to the invariants SRD<sup>1</sup> ≥ SRD2. All molecules on the horizontal axis y = SRD<sup>1</sup> − SRD<sup>2</sup> = 0 have SRD<sup>1</sup> = SRD<sup>2</sup> (due to two equidistant atoms from the center of mass) and can be projected (like any subset of QM9) to other coordinates as in Fig. [8.](#page-14-0) Molecular properties can be visualized on these geographic maps as 'mountainous' landscapes.

Table 7. Past ML and non-ML predictions of chemical elements have lower accuracies than by distance invariants in Table [4.](#page-6-1)

| M ETHOD       | D ESCRIPTION                                      | A CCURACY | R EFERENCE                 |
|---------------|---------------------------------------------------|-----------|----------------------------|
| LEAF          | L OCAL COORDINATION GEOMETRY                      | 86%       | (V ASYLENKO ET AL ., 2025) |
| M AT S CHOLAR | ML- DERIVED FROM LITERATURE                       | 81%       | (W ESTON ET AL ., 2019)    |
| M AT 2V EC    | ML- DERIVED FROM LITERATURE                       | 80%       | (T SHITOYAN ET AL ., 2019) |
| A TOM 2V EC   | ML- DERIVED FROM COMPOSITIONAL CONTENT            | 79%       | (Z HOU ET AL ., 2018)      |
| GN O ME       | F REQUENCY OF ELEMENTS AT THE SAME ATOMIC SITES   | 79%       | (M ERCHANT ET AL ., 2023)  |
| M AGPIE       | E LEMENTAL PHYSICAL CHARACTERISTICS               | 78%       | (W ARD ET AL ., 2016)      |
| O LIYNYK      | E LEMENTAL PHYSICAL CHARACTERISTICS               | 75%       | (O LIYNYK ET AL ., 2016)   |
| MEGN ET       | ML- DERIVED FROM ATOM , BOND AND GRAPH ATTRIBUTES | 73%       | (C HEN ET AL ., 2019)      |
| S KIP A TOM   | ML- DERIVED FROM ATOM CONNECTIVITY GRAPHS         | 68%       | (A NTUNES ET AL ., 2022)   |

![](_page_13_Figure_1.jpeg)

![](_page_13_Figure_3.jpeg)

Figure 6. Left: the heatmap of all molecular graphs from QM9 in the simplest continuous invariants. Right: 18336 graphs with 19 atoms. The color indicates the number of molecules at every pixel.

Figure 7. Every dot represents a molecular graph with the invariant coordinates x = SRD1, y = SRD<sup>1</sup> − SRD2, all in Angstroms, where 1A˚ = 10−<sup>10</sup>m ≈ the smallest interatomic distance.

![](_page_14_Figure_1.jpeg)

*804*

*806* This section extends all new concepts and results from sections [3](#page-2-1) and [4](#page-4-2) to any dimension n ≥ 2. Any n vectors p1, . . . , p<sup>n</sup> ∈ <sup>R</sup> <sup>n</sup> can be written as columns in the n × n matrix whose determinant has sign(p1, . . . , pn), which is ±1 or 0 (if p1, . . . , p<sup>n</sup> are linearly dependent).

*814* Definition B.1 (Centered Representation CR(G; A) of a graph with a sequence A ⊂ V (G)). *Let* G ⊂ <sup>R</sup> <sup>n</sup> *be a graph on* m *unordered points with the center of mass* O(G) = 0*. For any* 1 ≤ h ≤ n*, fix a* base sequence A *of ordered vertices* p1, . . . , p<sup>h</sup> ∈ V (G)*. If* h = n*, let* sign(A) *be the sign of the* n × n *determinant on the vectors* p1, . . . , pn*, else* sign(A) = 0*. Let* D(A) *be the matrix of signed distances between the ordered points* 0 = p0, p1, . . . , ph*. The matrix* R(G; A) *has* m − h *unordered columns, one for each vertex* q ∈ V (G) − A*, consisting of* h + 1 *distances* d(q, pi) *for* i = 0, . . . , h*, where* p<sup>0</sup> = 0*. The* Centered Representation CR(G; A) *is the triple* [sign(A), D(A), R(G; A)]*.*

*824*

Figure 8. The projection of QM9 to x = SRD1, y = SRD<sup>2</sup> − SRD3.

## B. Invariants and metrics on Euclidean graphs in any dimension n ≥ 2

Definition B.2 (Nested Centered Distribution NCD(G; h) of order h). *Let* G ⊂ <sup>R</sup> <sup>n</sup> *be any Euclidean graph on* m *unordered vertices and the center of mass at the origin* 0 ∈ R <sup>n</sup>*. Fix an* order 1 ≤ h ≤ n*.*

*(a) For any* <sup>h</sup> − <sup>1</sup> *distinct ordered vertices* <sup>p</sup>1, . . . , ph−<sup>1</sup> ∈ <sup>V</sup> (G)*, the* Centered Distribution CD(h) h−1 (G; p1, . . . , ph−1) *of index* h − 1 *is the unordered set of Centered Representation* CR(G; p1, . . . , ph) *from Definition [B.1](#page-14-1) for all* p<sup>h</sup> ∈ V (G) − {p1, . . . , ph−1}*.*

*(b) Now we will iteratively decrement an integer* k *from* h − 1 *down to 1 and define* CD(h) h−2 *of index* h − 2*, and so on until* CD(h) 1 *of index* k = 1*. For the initial* k = h − 1*, we use* CD(h) <sup>k</sup> = CD(h) h−1 *defined in part (a) above. For any* k − 1 *distinct*

![](_page_15_Figure_1.jpeg)

*864*

*869*

*874*

Figure 9. Each dot is a comparison of molecular graphs from QM9 by the distances on the progressively stronger invariants: NCD(G; 2) vs NCD(G; 3).

*ordered vertices* <sup>p</sup>1, . . . , pk−<sup>1</sup> ∈ <sup>V</sup> (G)*, the* Centered Distribution CD(h) k−1 (G; p1, . . . , pk−1) *of index* k − 1 *is the unordered collection of* CD(h) k (C; p1, . . . , pk) *of index* k *for all vertices* p<sup>k</sup> ∈ V (G) − {p1, . . . , pk−1}*.*

*(c) The* Nested Centered Distribution NCD(G; <sup>h</sup>) *of order* <sup>h</sup> *is the unordered collection of* CD(h) (G; p1) *of index 1 for all vertices* p<sup>1</sup> ∈ V (G)*. For the order* h = n*, the* mirror image NCD(G; n) *is obtained from* NCD(G; n) *by reversing* sign(p1, . . . , pn) *of* n × n *determinants in all* CR; p1, . . . , pn)*.*

If a sequence 0 ∪ A = (p0, p1, . . . , pn) ⊂ <sup>R</sup> <sup>n</sup> degenerates to a lower dimensional subspace, i.e. the vectors p1, . . . , p<sup>n</sup> become linearly dependent, then sign(A) of discontinuously changes. To guarantee the Lipschitz continuity, we multiply these signs by the strength σ below, while the volume vol(0 ∪ A) of the simplex on 0 ∪ A is not Lipschitz continuous.

Definition B.3 (*strength* σ(C)). *For any sequence* C *of* n + 1 *ordered points* p0, . . . , p<sup>n</sup> ∈ <sup>R</sup> <sup>n</sup>*, the* half-perimeter <sup>p</sup>(C) = <sup>1</sup> 2 P 1≤i<j≤n |p<sup>i</sup> − p<sup>j</sup> | *is the half-sum of pairwise distances between points of* C*. Let* vol(C) *denote the volume of the* <sup>n</sup>*-dimensional simplex on* <sup>C</sup>*. The* strength *of the simplex* <sup>C</sup> *is* <sup>σ</sup>(C) = vol<sup>2</sup> (C) p <sup>2</sup>n−<sup>1</sup>(C) *.*

887 888

890

894

896

898

911

914 915 916

918

924

928

p(C) is the half-distance <sup>1</sup> 2 <sup>|</sup>p<sup>0</sup> <sup>−</sup> <sup>p</sup>1|, so the strength is <sup>σ</sup>(C) = vol<sup>2</sup> (C) p(C) = 2|p<sup>0</sup> − p1|.

Lemma B.4 (Theorem 4.4 in [\(Widdowson & Kurlin,](#page-11-3) [2023\)](#page-11-3)). *Let* B *be obtained from a sequence* A ⊂ R <sup>n</sup> *of* n *points by perturbing every point within its* <sup>ε</sup>*-neighborhood. Then* <sup>|</sup>σ(A)−σ(B)| ≤ <sup>2</sup>ελ<sup>n</sup> *for a constant* <sup>λ</sup>n*, where* <sup>λ</sup><sup>1</sup> = 2*,* <sup>λ</sup><sup>2</sup> = 2√ 3*,* λ<sup>3</sup> ≈ 0.43*.*

Definition B.5 (max metric M<sup>∞</sup> on CRs). *Let Euclidean graphs* G, F ⊂ <sup>R</sup> <sup>n</sup> *on* m *unordered vertices have base sequences* A, B *of* h ≤ n *vertices. Consider the* m − h *columns of* R(G; A) *as a cloud of* m − h *unordered points in* <sup>R</sup> h *, also for* <sup>R</sup>(F; <sup>B</sup>)*. The* max *metric* <sup>M</sup>∞(CR(G; <sup>A</sup>), CR(F; <sup>B</sup>)) *is the maximum of* <sup>2</sup> λ<sup>n</sup> |sign(A)σ(0 ∪ A) − sign(B)σ(0 ∪ B)|*,* L∞(D(A), D(B))*, and the bottleneck distance* W∞(R(G; A), R(F; B))*, where all signs are zeros for* h < n*.*

In Definition [B.5,](#page-16-1) λ<sup>n</sup> is the Lipschitz constant of σ from Lemma [4.2.](#page-4-5)

Definition B.6 (Nested Bottleneck Metric NBM on NCDs). *Let* G, F ⊂ R <sup>n</sup> *be any Euclidean graphs on* m *unordered vertices. For any ordered vertices* p<sup>1</sup> . . . , ph−<sup>1</sup> ∈ V (G) *and* q<sup>1</sup> . . . , qh−<sup>1</sup> ∈ V (F)*, the complete bipartite graph* Γ(G; p1, . . . , ph−1; F; q1, . . . , qh−1) *has* m − h + 1 *white vertices and* m − h + 1 *black vertices representing* CR(G; p1, . . . , ph) *and* CR(F; q1, . . . , qh) *for all* m − h + 1 *vertices* p<sup>h</sup> ∈ V (G) − {p1, . . . , ph−1} *and* q<sup>h</sup> ∈ V (F) − {q1, . . . , qh−1}*, respectively. Set the* weight w(e) *of an edge* e *joining the vertices represented by* CR(G; p1, . . . , ph)*,* CR(F; q1, . . . , qh) *as the max metric* M<sup>∞</sup> *between these distributions, see Definition [B.5.](#page-16-1) Then Definition [4.4](#page-4-4) gives the bottleneck matching distance* BMD(Γ(G; p1, . . . , ph−1; F; q1, . . . , qh−1))*.*

*For any integer* 1 ≤ i < h *and ordered vertices* p<sup>1</sup> . . . , pi−<sup>1</sup> ∈ V (G) *and* q<sup>1</sup> . . . , qi−<sup>1</sup> ∈ V (F)*, the complete bipartite graph* Γ(G; p1, . . . , pi−1; F; q1, . . . , qi−1) *has* m − i + 1 *white vertices and* m − i + 1 *black vertices representing* CD(h) i (G; <sup>p</sup>1, . . . , pi) *and* CD(h) i (F; q1, . . . , qi) *for all* m − i + 1 *variable vertices* p<sup>i</sup> ∈ V (G) − {p1, . . . , pi−1} *and* q<sup>i</sup> ∈ V (F) − {q1, . . . , qi−1}*, respectively. Set the* weight w(e) *of an edge* e *joining the vertices represented by* CD(h) i (G; <sup>p</sup>1, . . . , pi) *and* CD(h) i (F; q1, . . . , qi) *as the previously computed distance* BMD(Γ(G; p1, . . . , p<sup>i</sup> ; F; q1, . . . , qi)) *for a smaller number* i *of fixed vertices. Then Definition [4.4](#page-4-4) gives the bottleneck matching distance* BMD(Γ(G; p1, . . . , pi−1; F; q1, . . . , qi−1))*. For* i = 1*, the graph* Γ(G, F) *has* m + m *vertices representing* CD1(G; <sup>p</sup>1)*,* CD(h) 1 (F; q1) *for all* p<sup>1</sup> ∈ V (G) *and* q<sup>1</sup> ∈ V (F)*. The* Nested Bottleneck Metric NBM(NCD(G; h), NCD(F; h)) *is the Bottleneck Matching Distance* BMD(Γ(G, F))*.*

### C. Metrics on graphs and their continuity under perturbations

This appendix verifies the axioms and Lipschitz continuity for all auxiliary metrics in section [4.](#page-4-2)

Lemma C.1 (metric axioms for the bottleneck matching distance BMD). *Let* S, Q *be any unordered distributions of the same number of objects with a base metric* d*. Define the complete bipartite graph* Γ(S, Q) *whose every edge* e *joining objects* R<sup>S</sup> ∈ S *and* R<sup>Q</sup> ∈ Q *has the weight* w(e) = d(RS, RQ)*. Then the bottleneck matching distance* BMD(Γ(S, Q)) *from Definition [4.4](#page-4-4) satisfies all metric axioms on such unordered distributions.*

*Proof of Lemma [C.1.](#page-16-2)* The coincidence axiom means that NBM(S, Q) = 0 if and only if the weighted distributions S, Q are equal in the sense that there is a bijection g : S → Q so that d(g(R), R) = 0 for any R ∈ S.

Indeed, if the weighted distributions S, Q can be matched by a bijection, we get a vertex matching E of Γ(S, Q) whose all edges have weights w(e) = 0. Definition [4.4](#page-4-4) implies that BMD(Γ(S, Q)) = 0 as required.

Conversely, if BMD(Γ(S, Q)) = 0, there is a vertex matching E in Γ(S, Q) with all w(e) = 0. This matching E defines a required bijection S → Q. The symmetry BMD(Γ(S, Q)) = BMD(Γ(Q, S)) follows from Definition [4.4](#page-4-4) and the symmetry of the base metric d.

To prove the triangle inequality

$$\text{BMD}(\Gamma(S, Q)) + \text{BMD}(\Gamma(Q, T)) \geq \text{BMD}(\Gamma(S, T)),$$

let ESQ, EQT be optimal vertex matchings in the graphs Γ(S, Q), Γ(Q, T), respectively, such that

$$\text{BMD}(\Gamma(S, Q)) = W(E_{SQ}), \text{BMD}(\Gamma(Q, T)) = W(E_{QT}).$$

*954*

*974*

*984*

see Definition [4.4.](#page-4-4) The composition ESQ ◦ EQT is a vertex matching in Γ(S, T), so W(ESQ ◦ EQT ) ≥ BMD(Γ(S, T)). It suffices to prove that

$$W(E_{SQ}) + W(E_{QT}) \geq W(E_{SQ} \circ E_{QT}).$$

Let eST be an edge with a largest weight from ESQ ◦EQT , so W(ESQ ◦EQT ) = w(eST ). The edge eST can be considered the union of edges eSQ ∈ ESQ, eQT ∈ EQT .

By the triangle inequality for the base metric d,

$$w(e_{SQ}) + w(e_{QT}) \geq w(e_{ST}) = W(E_{SQ} \circ E_{QT})$$

implies that

$$W(E_{SQ}) + W(E_{QT}) \geq W(E_{SQ} \circ E_{QT})$$

because both terms on the left-hand side are maximized for all edges (not only eSQ, eQT ) from ESQ, EQT .

Definition [C.2](#page-17-0) below makes sense for any distributions {[R1, w1], . . . , [Rm, wm]}, where R1, . . . , R<sup>m</sup> are objects with a base metric d and weights w1, . . . , w<sup>m</sup> ∈ [0, 1]. Each R<sup>i</sup> can be CBR or CBD of any depth with a base metric M<sup>∞</sup> or BMD from Definitions [B.5,](#page-16-1) [B.6.](#page-16-3)

Definition C.2 (EMD). *Let* S = {[Ri(S), wi(S)]} m(S) <sup>i</sup>=1 *and* Q = {[R<sup>j</sup> (Q), w<sup>j</sup> (Q)]} m(Q) <sup>j</sup>=1 *be weighted distributions of objects* Ri(S), R<sup>j</sup> (Q)*, which live in a space with a metric* d*. A* flow *from* S *to* Q *is an* m(S) × m(Q) *matrix whose element* fij ∈ [0, 1] *represents a* partial flow *from* Ri(S) *to* R<sup>j</sup> (Q)*. The* Earth Mover's Distance *is the minimum* cost

EMD(S, Q) =

mP (S)

$$for i = 1, \dots, m(S), \sum_{i=1}^m f_{ij} \leq w_j(Q) \text{ for } j = 1, \dots, m(Q), \text{ and } \sum_{i=1}^m \sum_{j=1}^m f_{ij} = 1.$$

mP (Q)

fijd(Ri(S), R<sup>j</sup> (Q)) *for variable 'flows'* fij ∈ [0, 1] *subject to the conditions*

mP (Q) j=1

fij ≤ wi(S)

The first condition mP (Q) j=1 fij ≤ wi(S) means that not more than the weight wi(S) of Ri(S) 'flows' into all R<sup>j</sup> (Q) via 'flows'

fij , j = 1, . . . , m(Q). The second condition mP (S) i=1 fij = w<sup>j</sup> (Q) means that all 'flows' fij from Ri(S) for i = 1, . . . , m(S)

'flow' into R<sup>j</sup> (Q) up to the maximum weight w<sup>j</sup> (Q). The last condition mP (S) i=1 mP (Q) j=1 fij = 1 forces to 'flow' all rows Ri(S) to all rows R<sup>j</sup> (Q).

The EMD satisfies all metric axioms, see the appendix in [\(Rubner et al.,](#page-10-18) [2000\)](#page-10-18), needs O(m<sup>3</sup> log m) time for distributions of a maximum size m and is approximated in O(m) time, see [\(Shirdhonkar & Jacobs,](#page-10-23) [2008;](#page-10-23) [Sato et al.,](#page-10-24) [2020\)](#page-10-24).

Definition [C.2](#page-17-0) can be adapted for the EMD between NDDs by (1) replacing the bottleneck distance W<sup>∞</sup> in Definition [B.5](#page-16-1) with EMD between clouds of equally weighted points, and (2) replacing BMD(Γ) for a bipartite graph Γ with EMD(Γ) between the unordered sets (of potentially different sizes) of BDDs with weights on all white vertices and BDDs on all black vertices.

The Lipschitz continuity of NDD and EMD in Theorem [D.1\(](#page-18-1)c) needs Lemmas [C.3,](#page-17-1) [C.4,](#page-17-2) [D.9.](#page-22-0)

If a metric graph G lives in an ambient metric space X, a natural perturbation of G is a shift of every vertex of G up to ε in the metric of X. Then the distance d(p, q) between any vertices p, q of G changes by at most 2ε.

We will prove the continuity in more general settings by only assuming that d(p, q) changes by at most 2ε for any p, q ∈ V (G) without requiring an ambient space X.

Lemma C.3 (Lipschitz continuity of BMD). *Let* Γ *be a complete bipartite graph with a vertex matching* E *such that any* e ∈ E *has a weight* w(e) ≤ ε*. Then* BMD(Γ) ≤ ε*.*

*Proof of Lemma [C.3.](#page-17-1)* By Definition [4.4,](#page-4-4) the given matching E has the weight W(E) = max e∈E w(e) ≤ ε. Since BMD(Γ) = min E W(E) is minimized for all vertex matchings, we get BMD(Γ) ≤ ε.

990 Lemma C.4 (Lipschitz continuity of EMD). *In Definition [C.2,](#page-17-0) let distributions* S, Q *have a bijection* Ri(S) ↔ Ri(Q) *between equally weighted objects such that* d(Ri(S), Ri(Q)) ≤ ε *for all* i = 1, . . . , m*, where* m = m(S) = m(Q)*. Then* EMD(S, Q) ≤ ε*.*

994 996 *Proof of Lemma [C.4.](#page-17-2)* In Definition [C.2,](#page-17-0) choose partial flows fij = 1 m for i = j, otherwise fij = 0. Then EMD(S, Q) ≤ Pm i=1 Pm j=1 <sup>f</sup>ijd(Ri(S), R<sup>j</sup> (Q)) = <sup>P</sup><sup>m</sup> i=1 1 m d(Ri(S), Ri(Q)) ≤ 1 m Pm i=1 ε = ε.

998

1000 1001 This appendix rigorously proves all parts of Theorem [D.1.](#page-18-1)

1002 1003 1004 Theorem D.1 (NCD solves Problem [1.1\)](#page-1-0). *(a) The Nested Centered Distribution* NCD(G; h) *in Definition [B.2](#page-14-2) is invariant under any rigid motion for all Euclidean graph* G *on* m *unordered vertices and, for a fixed dimension* n*, can be computed in time* O(n <sup>2</sup>m<sup>h</sup>+1) *with space* O(n <sup>2</sup>m<sup>h</sup>+1) *for any order* 1 ≤ h ≤ n*.*

1005 1006 1007 *(b)* NCD(G; 2) *is a complete invariant of all graphs* G ⊂ <sup>R</sup> <sup>2</sup> *under rigid motion from the group* SE(n) *in any dimension* n ≥ 1*.*

1008 1009 *(c) Perturbing each vertex of a graph* G ⊂ R <sup>n</sup> *within its* ε*-neighborhood changes* NCD(G; h) *up to* 2ε *in both metrics* NBM *and* EMD *for any order* 1 ≤ h ≤ n*.*

1014 1016 The *affine dimension* 0 ≤ aff(A) ≤ n of a cloud A = {p1, . . . , pm} ⊂ <sup>R</sup> <sup>n</sup> is the maximum dimension of the vector space generated by all inter-point vectors p<sup>i</sup> − p<sup>j</sup> , i, j ∈ {1, . . . , m}. Then aff(A) is an isometry invariant and is independent of an order of points of A. Any cloud A of 2 distinct points has aff(A) = 1. Any cloud A of 3 points that are not in the same straight line has aff(A) = 2.

1019 Lemma [D.2](#page-18-2) provides a simple criterion for a matrix to be realizable by squared distances of a point cloud in R n.

1024 Lemma D.2 (realization of distances). *(a) A symmetric* m × m *matrix of* sij ≥ 0 *with* sii = 0 *is realizable as a matrix of squared distances between points* p<sup>0</sup> = 0, p1, . . . , pm−<sup>1</sup> ∈ <sup>R</sup> <sup>n</sup> if and only if *the* (m − 1) × (m − 1) *matrix* gij = s0<sup>i</sup> + s0<sup>j</sup> − sij 2 *has only non-negative eigenvalues.*

1026 *(b) If the condition in (a) holds,* aff(0, p1, . . . , pm−1) *equals the number* k ≤ m − 1 ≤ n *of positive eigenvalues. Also in this case,* gij = p<sup>i</sup> · p<sup>j</sup> *define the* Gram matrix GM *of the vectors* p1, . . . , pm−<sup>1</sup> ∈ <sup>R</sup> <sup>n</sup>*, which are uniquely determined in time* O(m<sup>3</sup> ) *up to an orthogonal map in* <sup>R</sup> n*.*

1029 *Proof of Lemma [D.2.](#page-18-2)* (a) We extend Theorem 1 from [\(Dekster & Wilker,](#page-8-1) [1987\)](#page-8-1) to the case m < n + 1 and justify the reconstruction of p1, . . . , pm−<sup>1</sup> in time O(m<sup>3</sup> ) uniquely in <sup>R</sup> <sup>n</sup> up to an orthogonal map from O(n).

1034

1036

1039 is the Gram matrix, which can be written as GM = P <sup>T</sup> P, where the columns of the n × (m − 1) matrix P are the vectors p1, . . . , pm−<sup>1</sup> . For any vector v ∈ <sup>R</sup> m−1 , we have

1040 1041

1042 1043 1044 Since the quadratic form v <sup>T</sup> GMv ≥ 0 for any v ∈ <sup>R</sup> m−1 , the matrix GM is positive semi-definite meaning that GM has only non-negative eigenvalues, see Theorem 7.2.7 in [\(Horn & Johnson,](#page-9-20) [2012\)](#page-9-20).

## D. Proofs for Euclidean graphs from section [3](#page-2-1)

*(d) For any graphs* G, F ⊂ R <sup>n</sup> *on* m *unordered vertices, the metrics* NBM *and* EMD *between the invariants* NCD(G; h) *and* NCD(F; h) *from Definition [B.6](#page-16-3) can be computed in time* O(m<sup>2</sup>h+1.<sup>5</sup> log<sup>h</sup>+1 m) *with space* O(n <sup>2</sup>m<sup>2</sup>h+1 log<sup>h</sup>−<sup>1</sup> m) *for any order* 1 ≤ h ≤ n*.*

The part *only if* ⇒. Let a symmetric matrix S consist of squared distances between points p<sup>0</sup> = 0, p1, . . . , pm−<sup>1</sup> ∈ <sup>R</sup> <sup>n</sup>. For i, j = 1, . . . , m − 1, the matrix with the elements

$$g_{ij} = \frac{s_{0i} + s_{0j} - s_{ij}}{2} = \frac{p_i^2 + p_j^2 - |p_i - p_j|^2}{2} = p_i \cdot p_j$$

$$0 \leq |Pv|^2 = (Pv)^T (Pv) = v^T (P^T P) v = v^T \text{GM}v.$$

- *1045 1046 1047 1048 1049 1054 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1074 1076 1079 1089 1090 1094 1096 1099* The part *if* ⇐. For any positive semi-definite matrix GM, there is an orthogonal matrix Q such that Q<sup>T</sup> GMQ = D is the diagonal matrix, whose <sup>m</sup> <sup>−</sup> <sup>1</sup> diagonal elements are non-negative eigenvalues of GM. The diagonal matrix √ D consists of the square roots of eigenvalues of GM.
  - (b) The number of positive eigenvalues of GM equals the dimension k = aff({0, p1, . . . , pm−1}) of the subspace in <sup>R</sup> n linearly spanned by p1, . . . , pm−1. We may assume that all k ≤ n positive eigenvalues of GM correspond to the first k coordinates of R
    - <sup>n</sup>. Since Q<sup>T</sup> = Q−<sup>1</sup> , the given matrix GM = QDQ<sup>T</sup> = (Q √ D)(Q √
  - D) <sup>T</sup> becomes the Gram matrix of the columns of Q √
    - D. These columns become the reconstructed vectors p1, . . . , pm−<sup>1</sup> ∈ <sup>R</sup>
  - n. If there is another diagonalization Q˜<sup>T</sup> GMQ˜ = D˜ for Q˜ ∈ O(n), then D˜ differs from D by a permutation of eigenvalues, which is realized by an orthogonal map, so we set D˜ = D. Then GM = QD˜ Q˜<sup>T</sup> = (Q˜ √ D)(Q˜ √
  - D) T is the Gram matrix of the columns of Q˜ √
  - D. The new columns differ from the previously reconstructed vectors p1, . . . , pm−<sup>1</sup> ∈ <sup>R</sup> <sup>n</sup> by the orthogonal map QQ˜<sup>T</sup> . Hence the reconstruction is unique up to O(n)-transformations. Computing eigenvectors p1, . . . , pm−<sup>1</sup> requires a diagonalization of GM in time O(m<sup>3</sup> ) (?)section 11.5]press2007numerical. Though Lemma [D.2](#page-18-2) gives a two-sided criterion for realizability of distances by points p1, . . . , p<sup>m</sup> ∈ <sup>R</sup> <sup>n</sup>, the space of distance matrices is highly singular and cannot be easily sampled. Even m = 4 points in R <sup>2</sup> have 6 distances that should satisfy a polynomial equation saying that the tetrahedron with these 6 edge lengths has volume 0. So a randomly sampled matrix of potential distances for m > n + 1 is unlikely to be realizable by a cloud of m ordered points in R
  - n. Chapter 3 in [\(Liberti & Lavor,](#page-10-25) [2017\)](#page-10-25) discusses realizations of a complete graph given by a distance matrix in R
  - n. Lemma [D.3\(](#page-19-0)a) and later results hold for all clouds including degenerate ones, e.g. for 3 points in a straight line. Any points p1, . . . , pn−<sup>1</sup> ∈ A have aff(p1, . . . , pn−1) ≤ n − 2. For example, any two distinct points in A ⊂ <sup>R</sup> <sup>3</sup> generate a straight line. In R , any point p<sup>1</sup> ̸= O(A) forms a suitable {p1}. In <sup>R</sup> 3 , one can choose any distinct points p1, p<sup>2</sup> ∈ A so that the infinite straight line via p1, p<sup>2</sup> avoids O(A). If there are no such p1, p2, then A ⊂ <sup>R</sup> 3 is contained in a straight line L, so aff(A) = 1. In this degenerate case, the stronger condition aff(O(A) ∪ {p1, . . . , pn−1}) = aff(A) will help reconstruct A ⊂ L by using any point p<sup>1</sup> ̸= O(A). The first step is to reconstruct any ordered sequence from its distance matrix in Lemma [D.3\(](#page-19-0)a). Lemma [D.3\(](#page-19-0)a) holds for all degenerate clouds, e.g. for three points are in a straight line. Lemma D.3 (reconstruction of ordered points). *(a) Any sequence of ordered points* A = (p1, . . . , pm) *in* <sup>R</sup> <sup>n</sup> *can be reconstructed (uniquely up to isometry) from the matrix of the Euclidean distances* |p<sup>i</sup> − p<sup>j</sup> | *in time* O(m<sup>3</sup> )*. If all distances are divided by* R = max i=1,...,m |p<sup>i</sup> |*, the reconstruction of* A ⊂ <sup>R</sup> <sup>n</sup> *is unique up to isometry and uniform scaling.*
  - *(b) If* m ≤ n*, the uniqueness of reconstructions in part (a) holds if we replace isometry with rigid motion. Hence any* n − 1 *ordered points* p1, . . . , pn−<sup>1</sup> *can be uniquely reconstructed from all pairwise distances between* 0, p1, . . . , pn−<sup>1</sup> *up to* SO(n) *rotation around the origin* 0 ∈ R
  - n*. Proof of Lemma [D.3.](#page-19-0)* (a) By translation, we can put p<sup>1</sup> at the origin 0 ∈ <sup>R</sup>
  - <sup>n</sup>. Let GM be the (m − 1) × (m − 1) matrix gij = p 2 <sup>i</sup> + p 2 <sup>j</sup> − |p<sup>i</sup> − p<sup>j</sup> | 2 2 = p<sup>i</sup> · p<sup>j</sup> constructed from squared distances between p<sup>1</sup> = 0, . . . , p<sup>m</sup> for i, j = 2, . . . , m. By Lemma [D.2\(](#page-18-2)b) if GM has k ≤ n positive eigenvalues, then p<sup>1</sup> = 0, . . . , p<sup>m</sup> can be uniquely determined up to isometry in R <sup>k</sup> ⊂ <sup>R</sup> <sup>n</sup> in time O(m<sup>3</sup> ). If all distances are divided by the same radius R, the above construction guarantees uniqueness up to isometry and uniform scaling.
  - (b) If m ≤ n, any mirror image of A ⊂ R <sup>n</sup> after a suitable rigid motion in <sup>R</sup> <sup>n</sup> can be assumed to belong to an (n − 1)-dimensional hyperspace H ⊂ <sup>R</sup> <sup>n</sup>, where they are matched by a mirror reflection H → H with respect to an (n − 2)-dimensional subspace S ⊂ H. This reflection is realized by the SO(n) rotation through 180◦ around S. Lemma [D.3\(](#page-19-0)b) for m = n = 3 implies that any triangle is determined by its sides up to rigid motion in R 3 . For example, the sides 3, 4, 5 define a right-angled triangle whose mirror images are not related by rigid motion inside a plane H ⊂ R 3 , but are matched by composing a suitable rigid motion in H and a 180◦ rotation of R 3 around a line in H.

1104

1106

1109

1111

1114

1116

1118 1119

1124

1126

1129

1134

1136

1139 1140 We prove that any Euclidean graph G ⊂ R <sup>n</sup> can be reconstructed from its Nested Distance Distribution NCD(G; n) by induction on the dimension n.

1141 1142 1143 1144 1145 The inductive base n = 1 is Example [4.7.](#page-5-3) Assume that any graph G on m unordered vertices can be reconstructed in R k in time O(k <sup>3</sup>m) for any k < n. Below we prove the inductive step for the dimension n > 1. Start from any CBR(G; A) = [sign(A), CD(A), CR(G; A)] from Definition [3.5,](#page-4-0) where A is a sequence of some n ordered (not yet geometrically fixed) vertices p0, . . . , p<sup>n</sup> ∈ V (G). The first point p<sup>0</sup> is fixed at the origin 0 ∈ <sup>R</sup> <sup>n</sup> as usual by translation.

1146 1147 1148 1149 1151 Lemma [D.2\(](#page-18-2)b) for the matrix CD(A) gives the number k ≤ n of positive eigenvalues of the Gram matrix of the n vectors p1, . . . , p<sup>n</sup> in time O(n 3 ). If aff(A) = k < n, we use the nested structure of NCD(G; n) to take another CBR(G; p1, . . . , pk, q, . . . , pn) for a new vertex q ∈ V (G)−A. Check if aff(p1, . . . , pk, q) = k+1 again by Lemma [D.2\(](#page-18-2)b) using the matrix D(p1, . . . , pk, q). If the affine dimension has not increased, we take another CBR with the same points p1, . . . , p<sup>k</sup> and a new (k + 1)-st point from V (G) − {A ∪ q} and so on.

1154 This search through Centered Base Representations involving the remaining vertices of G requires a maximum of m − n − 1 steps with O(n 3 ) time for every computation of the affine dimension. Hence in time O(n <sup>3</sup>m), we can find a Centered Base

Lemma D.4 (time of determinant). *Any* n × n *determinant can be computed in time* O(n 3 ) *with space* O(n 3 )*.*

*Proof of Lemma [D.4.](#page-19-1)* Any n × n determinant can be computed by Gaussian elimination in time O(n 3 ) with space O(n 3 ), see [\(Bunch & Hopcroft,](#page-8-22) [1974\)](#page-8-22). The more recent theoretical estimate is O(n <sup>2</sup>.<sup>373</sup>) by [\(Fisikopoulos & Penaranda,](#page-9-21) [2016\)](#page-9-21).

*Proof of Theorem [D.1\(](#page-18-1)a).* Any rigid motion of R <sup>n</sup> mapping a Euclidean graph G ⊂ <sup>R</sup> <sup>n</sup> to another graph F is a bijection preserving distances and signs of determinants, and hence induces a bijection CBR(G; p1, . . . , pi) → CBR(F; q1, . . . , qi) for all p1, . . . , p<sup>i</sup> ∈ V (G) and corresponding vertices q1, . . . , q<sup>i</sup> ∈ V (F) for any i = 1, . . . , h, which implies a bijection NCD(G; h) → NCD(F; h). By Definition [3.5,](#page-4-0) if G has m unordered vertices, the NCD(G) consists of m(m−1). . .(m− h + 1) = O(m<sup>h</sup> ) Centered Base Representations CBR(G; A) for all base sequences A ⊂ V (G) of h ordered vertices.

Every CBR(G; A) consists of the three components sign(A), CD(A), CR(G; A). For h = n, sign(A) is the n × n determinant computable in time O(n 3 ) with space O(n 3 ) by Lemma [D.4.](#page-19-1) The distance matrix CD(A) needs O(h 2 ) time and O(h 2 ) space. The (h + 1) × (m − h) matrix CR(G; A) has O(hm) distances, each computable in time O(n). So CBR(G; A) can be computed in time O(n <sup>2</sup>m) with space O(n <sup>3</sup> + hm), where n ≤ m. Multiplying these complexities by the number O(m<sup>h</sup> ) of base sequences gives the final time O(n <sup>2</sup>m<sup>h</sup>+1) and space O(n <sup>3</sup> + hm<sup>h</sup>+1) for NCD(G).

The proof of Theorem [D.1\(](#page-18-1)b) will use the fact that any point in R <sup>n</sup> is uniquely determined by n + 1 distances to n + 1 ordered points that affinely span R <sup>n</sup>, and also Lemma [D.5.](#page-20-0)

Lemma D.5 (equal CBRs). *Let a Euclidean graph* G ⊂ R <sup>n</sup> *have the vertex set* V (G) *with the center of mass at* p<sup>0</sup> = 0 ∈ <sup>R</sup> <sup>n</sup>*. Let* n − 1 *ordered vertices* p1, . . . , pn−<sup>1</sup> *linearly span an* (n − 1)*-dimensional subspace* S ⊂ <sup>R</sup> <sup>n</sup>*. Let* G(p1, . . . , pn−1) *be the subgraph of* G *on the vertex set* V (G) *and all edges of* G *at* p1, . . . , pn−1*. For any other vertex* p*, let* CBR′ (G; p1, . . . , pn−1, p) *be obtained from the Centered Base Representation* BR(G; p1, . . . , pn−1, p) *by removing signs of distances from all vertices* q ∈ V (G) \ {p1, . . . , pn−1, p} *to* p*. If* BR′ (G; p1, . . . , pn−1, p) = BR′ (G; p1, . . . , pn−1, p′ ) *for some vertices* p, p′ ∈ V (G) \ {p1, . . . , pn−1}*, the mirror reflection with respect to* S *maps* G(p1, . . . , pn−1) *to itself and* p *to* p ′ *.*

*Proof of Lemma [D.5.](#page-20-0)* Under the reflection f<sup>S</sup> of <sup>R</sup> <sup>n</sup> with respect to the subspace S ⊂ <sup>R</sup> <sup>n</sup>, the vertices p, p′ should be swapped because they have equal (signed) distances to the ordered points p0, . . . , pn−<sup>1</sup> ∈ S. The equality of given CBR′ s means that V ′ = V (G) \ {p1, . . . , pn−1, p, p′} bijectively maps to itself via q 7→ q ′ so that any matched q, q′ have the same distances to the n + 1 ordered points p0, . . . , pn−1, p as to p0, . . . , pn−1, p′ , respectively. Any point in R <sup>n</sup> is determined by its distances to the n affinely independent points p0, . . . , pn−<sup>1</sup> up to the mirror reflection fS. Since f<sup>S</sup> fixes p0, . . . , pn−1, the reflection f<sup>S</sup> should swap q, q′ in such pairs and all their edges, so we conclude that fS(G(p1, . . . , pn−1)) = G(p1, . . . , pn−1) and fS(p) = p ′ .

*Proof of Theorem [D.1\(](#page-18-1)b).* The completeness is proved by reconstructing any Euclidean graph G ⊂ R <sup>n</sup> from NCD(G; n) uniquely up to rigid motion.

1156 Representation CBR(G; A) whose base sequence A affinely generates the subspace of dimension k = aff(V (G)) in <sup>R</sup> <sup>n</sup>. If k < n, the proof follows from the inductive hypothesis for the smaller dimension k.

1159 1160 1161 If aff(V (G)) = n, use the same notations for the fixed vertices 0 = p0, . . . , p<sup>n</sup> that linearly generate <sup>R</sup> <sup>n</sup>. Lemma [D.3\(](#page-19-0)a) for m = n + 1 and the distance matrix D(A) allow us to reconstruct n + 1 ordered points 0 = p0, . . . , p<sup>n</sup> up to isometry in R <sup>n</sup> in time O(n 3 ). By Definition [3.5](#page-4-0) every column of CR(G; A) contains Euclidean distances from the vertices 0 = p0, . . . , p<sup>n</sup> ∈ <sup>R</sup> <sup>n</sup>, which affinely generate <sup>R</sup> <sup>n</sup>, to another vertex q ∈ V (G) − A.

1164 1166 1167 These n + 1 distances uniquely determine the position of q in R <sup>n</sup> whose coordinates can be found as follows. Each scalar product q · p<sup>i</sup> can be computed as |q| · |p<sup>i</sup> <sup>|</sup> cos <sup>∠</sup>(q, <sup>0</sup>, pi) = <sup>|</sup>q<sup>|</sup> <sup>2</sup> + |p<sup>i</sup> <sup>2</sup> − |q − p<sup>i</sup> 2 2 for i = 1, . . . , n. On another hand, q · p<sup>i</sup> is a linear combination of unknown coordinates of q with coefficients equal to the coordinates of p<sup>i</sup> . One can find all coordinates of q in time O(n 3 ) by solving the system of linear equations, where the n × n determinant on the linear basis p1, . . . , p<sup>n</sup> is not zero. The total time is O(n <sup>3</sup>m).

1168 1169 Since all vertices q ∈ V (G) − A are geometrically unique, they can be (arbitrarily) ordered, say pn+1, . . . , pm, following p0, . . . , pn. The signs of distances in the matrix CR(G; A) also tell us about (present or absent) edges from p0, . . . , p<sup>n</sup> to all other vertices q ∈ V (G) − A.

1174 The nested structure of NCD(G; n) allows us to consider m−n unordered Base Representations CBR(G; p1, . . . , pn−1, p<sup>j</sup> ) for all vertices p<sup>j</sup> with j = n, . . . , m. Every vertex p<sup>j</sup> ∈ V (G) is uniquely determined in <sup>R</sup> <sup>n</sup> by the column of its signed distances to p0, . . . , p<sup>n</sup> in the (n + 1) × (m − n − 1) matrix R(G; p0, . . . , pn) for j = n + 1, . . . , m.

1176 1179 1180 By Lemma [D.5,](#page-20-0) this distance list of p<sup>j</sup> (without edges between p<sup>j</sup> , p<sup>k</sup> for j, k > n) suffices to identify one or maximum two Base Representations among all m − n unordered CBRs with the fixed n points p0, . . . , pn−<sup>1</sup> and variable n-th vertices. If there is a choice of two CBRs, we can take any of them for p<sup>j</sup> . Indeed, choosing another vertex pk, which should be mirror symmetric to p<sup>j</sup> , will produce a mirror image of the reconstructed subgraph G(p1, . . . , pn, p<sup>j</sup> ) by Lemma [D.5.](#page-20-0)

1186 1187 1188 1189 The strength σ(A) depends only on the distance matrix D(A), we write σ(A) for brevity. When the simplex on A degenerates, the strength σ(A) vanishes and is Lipschitz continuous by Lemma [4.2,](#page-4-5) while the volume of the simplex on B is not Lipschitz continuous as shown below.

1190

1194

1196

1199 1200 *Proof of Lemma [D.6.](#page-21-1)* The half-perimeter p(A) is computable via all pairwise distances in time O(n 2 ). The squared volume vol<sup>2</sup> (A) can be expressed by the Cayley-Menger (n+ 2)×(n+ 2) determinant from [\(Sippl & Scheraga,](#page-10-26) [1986\)](#page-10-26) in inter-point distances, which can be computed in time O(n 3 ) by Lemma [D.4.](#page-19-1)

1204 Lemma D.7 (axioms and time of M<sup>∞</sup> on CBRs). *Let* G, F ⊂ <sup>R</sup> <sup>n</sup> *be Euclidean graphs with* m *unordered vertices and base sequences* A ⊂ V (G) *and* B ⊂ V (F) *of* h ≤ n *ordered vertices. The metric* M∞(CBR(G; A), CBR(F; B)) *from Definition [B.5](#page-16-1) satisfies all metric axioms and is computable in time* O(h <sup>2</sup>+m<sup>1</sup>.<sup>5</sup> log<sup>h</sup>+1 m) *with space* O(h <sup>2</sup>+m log<sup>h</sup>−<sup>1</sup> m)*.*

1206 1209 *Proof of Lemma [D.7.](#page-21-0)* The metric axioms for M<sup>∞</sup> follow from the same axioms for the metrics L<sup>∞</sup> and W<sup>∞</sup> because the maximum of metrics is still a metric, see metric transforms in section 4.1 of [\(Deza & Deza,](#page-9-22) [2009\)](#page-9-22). The first metric 2 λ<sup>n</sup> |sign(A)σ(A) − sign(B)σ(B)| can be computed in time O(n 3 ) by Lemma [D.6.](#page-21-1) The metric L∞(CD(A), CD(B))

The matrix CR(G; p1, . . . , pn−1, p<sup>j</sup> ) from the found CBRs contains signs that determine the (present or absent) edges from p<sup>j</sup> to all other vertices p<sup>k</sup> for k = n + 1, . . . , m.

To guarantee the uniqueness of G ⊂ R <sup>n</sup> under rigid motion and not only under isometry, we additionally use sign(p1, . . . , pn) from CBR to fix an orientation of the simplex on p0, . . . , pn.

In R 2 , consider the triangle with two vertices fixed at (±l, 0) and one moving vertex (0, tε) for t ∈ [−1, 1]. The signed area of the triangle changes from −lε (unbounded because l can be large for any fixed small ε) to 0 (when t = 0 and the triangle degenerates), then to lε (when t = 1). The area changes by 2lε while only one vertex moves by 2ε, so the ratio of the area change over a point perturbation can be as large as a half-distance between given points.

Lemma D.6 (time of strength). *For any base sequence* A *of* n *ordered points* p1, . . . , p<sup>n</sup> ∈ <sup>R</sup> <sup>n</sup>*, the strength* σ(A) *can be computed in time* O(n 3 )*.*

1214 Lemma D.8 (metric axioms for NBM on NCDs). *The Nested Bottleneck Metric* NBM *from Definition [B.6](#page-16-3) satisfies all metric axioms on Nested Distance Distributions.*

1216

1218 1219 *Proof of Lemma [D.8.](#page-22-1)* Induction on the depth i = n, . . . , 1. The inductive base i = n follows from the metric axioms in Lemma [D.7](#page-21-0) for M<sup>∞</sup> in Definition [B.5.](#page-16-1)

1224 Lemma D.9 (Lipschitz continuity of M∞). *Let* A *be a base sequence of* 1 ≤ h ≤ n *ordered vertices in a Euclidean graph* G ⊂ R <sup>n</sup>*. Let* B, F *be obtained from* A, G*, respectively, by perturbing every vertex of* G *within its* ε*-neighborhood in* <sup>R</sup> n*. Then* CBR(G; A) *changes in* M<sup>∞</sup> *from Definition [B.5](#page-16-1) by at most* 2ε*, so* M∞(CBR(G; A), CBR(F; B)) ≤ 2ε*.*

1226 1229 *Proof of Lemma [D.9.](#page-22-0)* Order all vertices of the graphs G, F so that every vertex p<sup>i</sup> ∈ V (G) has the same index as its perturbation q<sup>i</sup> ∈ V (F). The bijection p<sup>i</sup> ↔ q<sup>i</sup> induces the bijections between the corresponding elements of the matrices CD(A) ↔ CD(B) and CR(G; A) ↔ CR(F; B), which all differ by at most 2ε. Lemma [4.2](#page-4-5) implies that 2 λ<sup>n</sup> |sign(A)σ(A) − sign(B)σ(B)| ≤ 2ε Since all three components of the max metric M<sup>∞</sup> in Definition [B.5](#page-16-1) have the upper bound 2ε, conclude that M<sup>∞</sup> ≤ 2ε.

1234 1236 Definition [C.2](#page-17-0) can be adapted for the EMD between NCDs by (1) replacing the bottleneck distance W<sup>∞</sup> in Definition [B.5](#page-16-1) with EMD between clouds of equally weighted points, and (2) replacing BMD(Γ) for a bipartite graph Γ with EMD(Γ) between the unordered sets (of potentially different sizes) of CBDs with weights on all white vertices and CBDs on all black vertices.

1239 1240 1241 1242 *Proof of Theorem [D.1\(](#page-18-1)c).* We first prove the Lipschitz continuity of the metric NBM on NCDs. Order all vertices of the graphs G, F so that every p<sup>i</sup> ∈ V (G) has the same index as its ε-perturbation q<sup>i</sup> ∈ V (F). In Definition [B.6,](#page-16-3) for any base sequence A of p1, . . . , p<sup>h</sup> ∈ V (G), there is a base sequence B of vertices q1, . . . , q<sup>h</sup> ∈ V (F), which are ε-perturbations of p1, . . . , ph, respectively, such that M∞(CBR(G; A), CBR(F; B)) ≤ 2ε by Lemma [D.9.](#page-22-0)

1243 1244 1245 1246 1247 1248 1249 These distances M<sup>∞</sup> are weights of edges in the index-preserving vertex matching E of the complete bipartite graph Γ(G; p1, . . . , ph−1; F; q1, . . . , qh−1) for any p1, . . . , ph−<sup>1</sup> and their ε-perturbations q1, . . . , qh−1. Then BMD(Γ(G; p1, . . . , ph−1; F; q1, . . . , qh−1)) ≤ 2ε by Lemma [C.3.](#page-17-1) Since this conclusion holds for all (choices of) p1, . . . , ph−<sup>1</sup> ∈ V (G), we iteratively apply this argument for the bipartite graphs Γ(G; p1, . . . , pi−1; F; q1, . . . , qi−1) for 1 ≤ i < n and finally conclude that NBM(NCD(G; h), NCD(F; h)) ≤ 2ε. The proof that EMD(NCD(G; h), NCD(F; h)) ≤ 2ε is similar by using Lemma [C.4](#page-17-2) instead of [C.3.](#page-17-1)

1254 1256 1259 For i = h, the weight w(e) of each edge e equals M∞, which needs time O(m<sup>1</sup>.<sup>5</sup> log<sup>h</sup>+1 m) and space O(m log<sup>h</sup>−<sup>1</sup> m) by Lemma [D.7](#page-21-0) for any h ≤ n ≤ m. For all O(m<sup>2</sup> ) edges of Γ(G; p1, . . . , ph−1; F; q1, . . . , qh−1), the time is O(m<sup>3</sup>.<sup>5</sup> log<sup>h</sup>+1 m), the space is O(m<sup>3</sup> log<sup>h</sup>−<sup>1</sup> m). The bottleneck matching distance BMD for such a graph is computed by [\(Hopcroft & Karp,](#page-9-24) [1973\)](#page-9-24) in time O(E √ V ) = O(m<sup>2</sup>.<sup>5</sup> ), which is dominated by the time O(m<sup>3</sup>.<sup>5</sup> log<sup>h</sup>+1 m) preparing the weighted graph.

1260 For all O(m2(h−1)) choices of ordered vertices p1, . . . , ph−<sup>1</sup> ∈ V (G) and q1, . . . , qh−<sup>1</sup> ∈ V (F), the Bottleneck Matching Distance for all graphs Γ(G; p1, . . . , ph−1; F; q1, . . . , qh−1) are found in time

requires time O(h 2 ) and space O(h 2 ). The bottleneck distance W∞(CR(G; A)), CR(F; B)) between (h + 1) × (m − h) matrices CR(G; A), CR(F; B) with unordered columns (considered as clouds of m − h unordered points in <sup>R</sup> <sup>h</sup>+1) needs time O(m<sup>1</sup>.<sup>5</sup> log<sup>h</sup>+1 m) and space O(m log<sup>h</sup>−<sup>1</sup> m) by Theorem 6.5 in [\(Efrat et al.,](#page-9-23) [2001\)](#page-9-23).

The inductive step from a depth i (between 1, n) to the smaller value i − 1 follows from Lemma [C.1](#page-16-2) and the metric axioms in the inductive hypothesis for the depth i.

*Proof of Theorem [D.1\(](#page-18-1)d).* In Definition [B.6,](#page-16-3) for any fixed 1 ≤ i ≤ h and ordered vertices p<sup>1</sup> . . . , pi−<sup>1</sup> ∈ V (G) and q<sup>1</sup> . . . , qi−<sup>1</sup> ∈ V (F), the complete bipartite graph Γ(G; p1, . . . , pi−1; F; q1, . . . , qi−1) has V = 2(m − i + 1) = O(m) vertices and E = (m − i + 1)<sup>2</sup> = O(m<sup>2</sup> ) edges.

$$O(m^{2(h-1)})O(m^{3.5} \log^{h+1} m) = O(m^{2h+1.5} \log^{h+1} m)$$

 with space O(m<sup>2</sup>h+1 log<sup>h</sup>−<sup>1</sup> m). For every next iteration i = h − 2, . . . , 1, the parameter i goes down by 1 every time. We can compute all distances BMD(Γ(G; p1, . . . , pi−1; F; q1, . . . , qi−1) in time

 All CBDs in Definition [3.5](#page-4-0) have sizes at most m, which is the maximum number of points in the given clouds. The EMD between weighted distributions of a maximum size m can be computed in near-cubic time O(m<sup>3</sup> log m), see [\(Fredman &](#page-9-25) [Tarjan,](#page-9-25) [1987;](#page-9-25) [Goldberg & Tarjan,](#page-9-26) [1987\)](#page-9-26). Since this complexity is dominated by the time O(m<sup>3</sup>.<sup>5</sup> log<sup>h</sup>+1 m) for computing O(m<sup>2</sup> ) weights M∞, each in time O(m<sup>1</sup>.<sup>5</sup> log<sup>h</sup>+1 m) by Lemma [D.7,](#page-21-0) the total time for the EMD is the same as for the NBM, similarly for space complexities

$$O(m^{2(i-1)})O(m^{3.5} \log^{h+1} m) = O(m^{2i+1.5} \log^{h+1} m).$$

The sum of all these times for i = 1, . . . , h − 1 is still O(m<sup>2</sup>h+1.<sup>5</sup> log<sup>h</sup>+1 m) from the first step.