# Continuous Machine Learning On Euclidean Graphs With Unordered Vertices

## Abstract

000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054 Molecular graphs can change their chemical properties under non-rigid deformations in Euclidean space. Hence it is vitally important to distinguish rigid classes of molecular graphs under compositions of translations and rotations. Also, robust outputs of machine learning on molecular graphs embedded in Euclidean space should continuously change under perturbations, motivated by atomic vibrations and experimental noise. We developed a complete invariant that can be inverted back to an embedded graph, uniquely under rigid motion, and has a Lipschitz continuous distance satisfying all metric axioms. For a fixed dimension, the invariant and metric can be computed in polynomial time of the number m of unordered vertices and hence avoiding exponentially many permutations. The new invariants distinguish all chemically different graphs in the world's largest databases of 3D molecules in a few hours on a modest desktop.

## 1. Motivations For Complete And Continuous Invariant Inputs In Application-Driven Ml

This paper formalizes necessary conditions for ML on real data with ambiguous representations and develops complete and Lipschitz continuous invariants satisfying these conditions on any Euclidean graphs and justifying a rigorous concept of a molecular structure. Many real structures from star constellations to molecules are represented by graphs embedded in a Euclidean space (Bonchev, 1991). A Euclidean graph G ⊂ R
n is a finite set of m unordered (unlabeled)
vertices located at distinct points of R
n and connected by straight-line edges. Forgetting all edges of G ⊂ R
n gives us the *vertex set* V (G) ⊂ R
n of m unordered points. A
Euclidean graph can be disconnected and can have vertices v of any *degree* that is the number of edges whose endpoint is v. Loops and multiple edges (with the same endpoints) do not appear in Euclidean graphs because all edges are straight line segments and can also intersect in theory.

. Correspondence to: Anonymous Author
<anon.email@domain.com>.

1 Graphs can be considered under any *equivalence* relation that should satisfy the axioms: 1) *reflexivity*: G ∼ G, 2) symmetry: if G ∼ F then F ∼ G, 3) *transitivity*: if G ∼ F and F ∼ H then G ∼ H. In chemistry, the simplest equivalence is by chemical composition, which is insufficient in practice, e.g. *stereoisomers* in Fig. 1 (right) have the same chemical compositions and non-equivalent rigid shapes with different chemical properties (Rieder et al., 2023). For molecules, the strongest equivalence (distinguishing as many graphs as practically possible) is a geometric isomorphism G ∼= F, i.e. an orientation-preserving transformation of R
n that bijectively maps the vertices and edges: G → F.

Geometric isomorphisms are also called *rigid motions* (compositions of translations and rotations), which form the special Euclidean group SE(n). The slightly weaker equivalence (not distinguishing mirror images) is an *isometry*, which is any distance-preserving transformation including reflections. Any geometrically isomorphic molecules have the same chemical properties. If a flexible molecule changes its rigid shape, its functional properties can change, so it is important to distinguish rigid shapes (Wilson et al., 1991). To reliably distinguish at least some Euclidean graphs G ⊂ R
n, we need an *invariant* I defined as a numerical descriptor preserved by any rigid motion in R
n. Alternatively, if I(G) ̸= I(F), then G ̸∼= F, so any invariant has no false negatives that are pairs of different representatives of rigidly equivalent graphs (denoted by G ∼= F) having equal values of a (non-invariant) descriptor. The number of vertices (or edges) of G is an integer-valued weak invariant that cannot separate any graphs in Fig. 1. The strongest invariant I separating all non-equivalent graphs is called *complete* meaning that if I(G) = I(F) then G ∼= F. Alternatively, a complete invariant I has *no false positives* that are pairs of non-equivalent graphs G ̸∼= F with I(G) = I(F).

Since all real data (such as inter-point distances) are noisy, a more practically important answer is not binary ('same or different') but should be continuously quantified by a distance metric between isometry classes. The atomic vibrations (Feynman, 1971) imply that rigid classes of molecules graphs on m unordered atoms form a continuous *Graph Isometry Space* GIS(R
3; m). Only for triangular graphs with m = 3, their space was previously known due to the side-side-side theorem saying that any triangles are isometric if and only if they have the same triple of sides (inter-point distances) *a, b, c* considered up to 6 permutations. Hence the space of triangular graphs is
{0 < a ≤ b ≤ c ≤ a+b} ⊂ R
3, where c ≤ a+b guarantees that distances *a, b, c* are realizable by a real triangle. Problem 1.1 (complete invariant of Euclidean graphs with a polynomial-time continuous metric). *Find a function* I :
{*Euclidean graphs with of unordered vertices in* R
n} → a space X with a distance d *satisfying the conditions below:*
(a) completeness of the invariant: any graphs G, F are related by rigid motion in R
n *if and only if* I(G) = I(F);
(b) Lipschitz continuity: there is a constant λ and a metric d satisfying the axioms 1) d(*α, β*) = 0 *if and only if* α = β, 2) d(*α, β*) = d(β, α), 3) d(α, β) + d(β, γ) ≥ d(α, γ) for all α, β, γ ∈ X, such that if F is obtained by perturbing every vertex of G up to ε > 0*, then* d(I(G), I(F)) ≤ λε; (c) invertibility: any Euclidean graph G can be reconstructed (uniquely up to rigid motion in R
n*) from* I(G);
(d) computability: for a fixed dimension n, the invariant I, d*, and a reconstruction of* G ⊂ R
n from I(G) *can be* obtained in polynomial time of the number of vertices. Condition 1.1(a) means that a complete invariant I has the strongest expressivity (Zhang et al., 2024) by uniquely identifying any Euclidean graph under geometric isomorphism. To be useful for noisy inputs, a complete invariant should continuously change under perturbations in a suitable metric. The axioms in 1.1(b) are the foundations of metric geometry (Melter & Tomescu, 1984) and accepted in chemistry (Weinhold, 1975). If the triangle axiom fails with any additive error, the classical k-means and DBSCAN clustering are open to adversarial attacks in (Rass et al., 2024). If the first axiom is ignored, d ≡ 0 satisfies all other axioms. The first axiom implies the completeness of I in 1.1(a) but the continuity is much stronger. Indeed, for any complete invariant I, one can define the discrete metric d(I(G), I(F)) = 1 for G ̸∼= F, which unhelpfully treats all non-equivalent graphs
(even near-duplicates) as equally distant. The Lipschitz continuity in 1.1(b) is necessary for smoothness, which is implicitly assumed by any gradient-based optimization.

055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109 Condition 1.1(c) requires I to be not only complete and continuous but also efficient to explicitly reconstruct G, even better than a DNA code that is does not explain how to grow a living organism. Computability 1.1(d) prevents brute-force attempts, e.g. defining I(G) as the infinite set of images of G under all rigid motions or taking m! distance matrices over all permutations of m unordered vertices. The main contribution is the new invariant Nested Centered Distribution, which solves Problem 1.1, including the new Lipschitz continuity, for all Euclidean graphs in R
n.

## 2. **Past Work On Distances For Euclidean Graphs**

Ordered clouds. The vertex set V (G) of a Euclidean graph G ⊂ R
n is called a *point cloud* C. If all points p1*, . . . , p*m of C are ordered (not under the action of all m! permutations), a complete invariant of C under isometry (compositions of translations, rotations, reflections) is the classical m×m matrix (Li et al., 2023) of pairwise distances |pi−pj | due to Theorem 9 in (Grinberg & Olver, 2019) or, after shifting the center of mass to the origin, the Gram matrix of scalar products pip˙j by Theorem 1 in (Dekster & Wilker, 1987). This multidimensional scaling (Schoenberg, 1935)
can also provide an embedding C ⊂ R
k preserving all distances of C for a dimension k ≤ m. This embedding C ⊂ R
k uses eigenvectors whose ambiguity up to signs gives an exponential time that can be close to O(2m), not polynomial in the number m of ordered points as in 1.1(d). Unordered clouds. Computational geometry developed many algorithms for detecting geometric isomorphism (or isometry, also called congruence) between point sets without edges (Huttenlocher et al., 1993; Chew & Kedem, 1992; Chew et al., 1999; Goodrich et al., 1999). For a set A ⊂ Qn of m points, Theorem 3 in (Arvind & Rattan, 2016) computed in time n O(n)*poly*(mM) a canonizing function f(A),
which can be considered a complete isometry invariant of A, where M upper bounds the binary encodings of the rational coordinates in the input. For point clouds under rigid motion (also distinguishing mirror images), Theorem 4.7 in (Widdowson & Kurlin, 2023) described a metric computable in time O(n(mn−1/n!)3log m). (Hordan et al.,
2024; Delle Rose et al., 2024; Nigam et al., 2024; Amir et al., 2024; Maennel et al., 2024) also achieved the completeness for point clouds but without a Lipschitz continuous metric as in 1.1(b). Energy potentials written as infinite series of spherical harmonics, are often considered complete representations of atomic environments, which holds in the limit but not for a finite size(Pozdnyakov et al., 2020). For a fixed set of m vertices in general position, one can choose any of m(m − 1)/2 edges and produce 2 m(m−1)/2 non-isometric graphs. Problem 1.1 for arbitrary graphs is computationally much harder than for point clouds due to exponentially many different graphs on the same vertex set.

110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 The graph isomorphism problem (Grohe & Schweitzer, 2020) for abstract (non-Euclidean) graphs is another version of Problem 1.1 without continuous metrics. The latest advances (Babai, 2019; Helfgott et al., 2017) achieved only quasipolynomial time. While many partial cases were solved, e.g. for planar graphs (embedded in R
2 without intersecting edges), see (Kiefer et al., 2019), the k-dimensional Weisfeiler-Leman test (Leman & Weisfeiler, 1968) fails for 3-regular graphs of size O(k). The key limitation of WL tests is their local nature when invariants are gradually expanded from a vertex or a k-tuple. Then covering a graph on m vertices needs O(m) expansions leading to exponential sizes in m. Section 3.9 in (Dym & Gortler, 2024) discussed that a complete invariant (under all permutations of m vertices) that has a polynomial time in the dimension n would also solve the graph isomorphism problem in polynomial time. Condition 1.1(d) is easier for a fixed dimension n, e.g. n = 2, 3 are practical cases. The number m of vertices can be dozens or hundreds, e.g. for molecular graphs in R 
3, where vertices are centers of atoms and edges are interatomic bonds that keep atoms together in a stable molecule. Geometric Deep Learning in (Bronstein et al., 2021) pioneered an axiomatic approach to geometric classifications beyond Euclidean space R
n in (Bronstein et al., 2017).

Some neural networks were proved to be universal (Maron et al., 2019; Zhou, 2020; Abbe & Sandon, 2020) in the sense of approximating any continuous function on given data with sufficiently many layers. This universality property has been strengthened in Problem 1.1 to the full completeness of an explicit invariant that should be computable in polynomial time and invertible to an original graph up to rigid motion. The key challenge was to compute an exact (not approximate) metric that is also Lipschitz continuous. Equivariants (Kondor & Trivedi, 2018; Cohen et al., 2019; Fuchs et al., 2020; Deng et al., 2021) are defined as descriptors E satisfying E(f(G)) = Tf (E(G)) for any rigid motion f and all graphs G ⊂ R
n, where Tf can be any map, not only the identity as for invariants. Any linear combination of points, e.g. the center of mass, is equivariant but cannot distinguish graphs under translation. Equivariants (Gao et al., 2020; Qi & Luo, 2020; Tu et al., 2022; Batzner et al., 2022) help predict forces acting on atoms to move them to a more optimal configuration. These timedependent graphs Gt can be studied directly by invariant values I(Gt) without computing intermediate atomic forces.

Many neural networks optimize millions of parameters, e.g. see Table 4 (Goyal et al., 2021), to achieve great accuracies (Dong et al., 2018; Akhtar & Mian, 2018; Laidlaw & Feizi, 2019; Guo et al., 2019; Colbrook et al., 2022) but require re-training on any new data. All known descriptors of molecular graphs (Duvenaud et al., 2015; Choo et al., 2023) have no proofs of all conditions 1.1(a,b,c,d).

Gromov-Wasserstein metrics (Memoli ´ , 2011) are defined for any metric-measure spaces (Brecheteau ´ , 2019) by minimizing over infinitely many correspondences between points, but cannot be approximated with a factor less than 3 in polynomial time unless P=NP by Corollary 3.8 in (Schmiedl, 2017) and Theorem 3.3 in (Agarwal et al., 2018), see fast algorithms for important cases in (Memoli et al. ´ , 2021; Lim et al., 2023; Majhi et al., 2024). (Nikolentzos et al., 2017; Majhi & Wenk, 2022; Buchin et al., 2023) made significant advances in the related problems of matching and finding distances between fixed Euclidean graphs without considering isometry. Computing a metric between rigid classes is only a small part of Problem 1.1. Indeed, to efficiently navigate on Earth, in addition to distances between cities, we need a map of the planet and hence an invertible continuous invariant I similar to geographic coordinates.

## 3. Graph Invariants: From Fastest To Complete

Let |p−q| denote the Euclidean distance between any points p, q ∈ R
n. We always translate any graph G ⊂ R
n so that the *center of mass* O(G) = 1mP
p∈V (G)
p of the *vertex set* V (G) is at the origin 0 ∈ R
n. Then Problem 1.1 reduces to the SO(n)-invariance under orthogonal transformations. Definition 3.1 (signed distance d(p, q) and invariants SRD, SPD,PDD). Let G ⊂ R
n be any Euclidean graph on m arbitrarily ordered vertices p1 . . . , pm. If any pi, pj ∈
V (G) are connected by an edge of G*, define the* signed distance as d(p, q) = |p − q|*, else set* d(p, q) = −|p − q|. (a) *The vector* SRD(G) of sorted radial distances *consists* of m distances |p| for all p ∈ V (G) *in decreasing order.* (b) *The vector* SPD(G) of sorted pairwise distances *consists* of all distances d(pi, pj ) *in decreasing order.*
(c) Let D(G) be the m×(m−1)-matrix whose the i-th row consists of d(pi, pj ), j ∈ {1, . . . , m} \ {i}, in increasing order. The Pointwise Distance Distribution PDD(G) *consists* of these unordered rows with equal weights 1/m. If any k > 1 rows of D(G) are equal, they can be collapsed in PDD(G) to a single row with the weight k/m. The PDD was defined for clouds as a local distribution of distances in Definition 5.5 of (Memoli ´ , 2011) and for periodic sets in (Widdowson & Kurlin, 2022) but not for Euclidean graphs.

| SRD   | SORTED RADIAL VECTOR            | DEF 3.1   |
|-------|---------------------------------|-----------|
| SPD   | SORTED DISTANCE VECTOR          | DEF 3.1   |
| PDD   | POINTWISE DISTANCE DISTRIBUTION | DEF 3.1   |
| CR    | CENTERED REPRESENTATION         | DEF 3.3   |
| NCD   | NESTED CENTERED DISTRIBUTION    | DEF 3.5   |
| NBM   | NESTED BOTTLENECK METRIC        | DEF 4.5   |

The PDD(G) includes every signed distance twice, once as d(*p, q*) in the row of a vertex p, and as d(*q, p*) in the row of a vertex q. Hence SPD(G) can be obtained from PDD(G) by (1) combining all distances into one vector, (2) sorting them in decreasing order, and (3) keeping only one copy of every two repeated distances. Example 3.2 shows that the invariant PDD(G) is strictly stronger than SPD(G). Example 3.2 (invariants SRD, SPD,PDD for tetrahedral graphs in Fig. 1). (a) *Since the vertex sets of* Ti ⊂ R
3 are regular tetrahedra with all pairwise distances 1, these graphs have identical SRD(Ti) *of 4 equal circumradii of* the same vertex set V (Ti) *independent of* i = 1, . . . , 4. The first graph T1 has two edges contributing +1 *and four* non-edges (dashed lines) contributing −1 to the Sorted Distance Vector SPD(T1) = (+1, +1, −1, −1, −1, −1)*. The* graph T2 *also has two edges, so* SPD(T2) = SPD(T1)
doesn't distinguish T1 ̸∼= T2 up to rigid motion. Similarly, the graphs T3 ̸∼= T4 *are not distinguished by the invariants* SPD(T3) = (+1, +1, +1, −1, −1, −1) = SPD(T4).

165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219
(b) In T1*, every vertex has exactly one edge and two* non-edges (dashed lines), hence its signed distances are
+1, −1, −1*. The matrix* PDD(T1) = (100% | −1, −1, +1)
consists of a single row, where the weight 100% indicates that all vertices of T1 *have the same row in* PDD. The graph T2 has one vertex (25%) with no edges, two vertices
(50%) with one edge, and one vertex (25%) with two edges, so PDD(T2) = 
25% −1 −1 −1 50% −1 −1 +1 25% −1 +1 +1

 ̸= PDD(T1),
so PDD *distinguishes the rigidly non-equivalent graphs* T1 ̸∼= T2 *with* SPD(T1) = SPD(T2). The graph T3 has one vertex (25%) with no edges and three vertices (75%*) with* two edges, so PDD(T3) = 
25% −1 −1 −1 75% −1 +1 +1 
.

The graph T4 *has two vertices (50%) with one edge and* two vertices (50%*) with two edges. Then* PDD(T4) = 50% −1 −1 +1 50% −1 +1 +1 
*, so* PDD distinguishes the graphs T3 ̸∼= T4 *with equal* SPD(T3) = SPD(T4).

For a graph G with m unordered vertices, PDD(G) has m − 1 columns. The reduced version PDD(G; k) includes only the first k columns for 1 ≤ *k < m* − 1. Though PDDs have unordered rows, they can be continuously compared by Earth Mover's Distance (Rubner et al., 2000). Fig. S4 in (Pozdnyakov et al., 2020) described infinitely many non-isometric pairs of clouds *C, C*′ ⊂ R
3 with PDD(C) = PDD(C
′). These counter-examples inspired the stronger invariants for graphs below. For simplicity, we will introduce all invariants and metrics in dimension n = 2. All higher dimensions n > 2 are covered in appendices. While PDD(G) includes signed distances to a single (arbitrary) vertex pi ∈ V (G), a stronger invariant below include triples of signed distances to three base points, one of which is the center of mass of V (G) because any point in R
2is uniquely determined by its distances to three fixed points.

Definition 3.3 (Centered Representation CR(G; A) of a graph with A ⊂ V (G)). Let G ⊂ R
2 *be a graph on* m unordered points with the center of mass p0 = O(G) = 0. (a) For any vertex p1 ∈ V (G), the matrix R(G; p1) has m − 1 *unordered columns, one for each vertex* q ∈
V (G)\{p1}, consisting of the signed distances d(q, p0) and d(q, p1)*. Here* p0 = 0 is not considered as a vertex of G*, so* d(*q, p*0) = −|q|*. The* Centered Representation CR(G; p1) is the pair [d(p0, p1), R(G; p1)], where d(p0, p1) = −|p1|. (b) *Fix a* base pair A of ordered vertices p1, p2 ∈
V (G)*. Let* sign(A) be the sign of the 2 × 2 determinant on the vectors p1, p2. Let D(A) be the matrix of signed distances between p0, p1, p2*. The matrix* R(G; A) has m − 2 unordered columns, one for each vertex q ∈ V (G) \ A*, consisting of signed distances* d(q, p0), d(q, p1), d(q, p2)*. The* Centered Representation CR(G; A) *is the triple* [sign(A), D(A), R(G; A)].

After fixing p0 = 0, the matrix D(A) and sign(A) help
reconstruct base vertices p1, p2 ∈ R
2, uniquely under rotation around 0. Any other q ∈ V (G) \ A is fixed relative
to p0, p1, p2 by its column in R(G; A). A positive sign of
d(pi, pj ) indicates an edge between vertices pi, pj . This
argument will later be formalized in Theorem 4.6(b).
Example 3.4 (CRs for 2-vertex bases in R
2). Let G ⊂ R
2
be the triangular cycle on p1 = (2, 0), p2 = (−1, 1),
p3 = (−1, −1)*, so* O(G) = 0 *and all signed distances*
are positive, see Fig. 2 (top left). For A = (p1, p2),
sign(A) = sign

2 −1
0 1

= 1*. The distance matrix on*
0 −2 −
√2
−2 0 
√10
−
√2√10 0
0, p1, p2 is D(p1, p2) =

*. Then*

−|p3|
|p3 − p1|
|p3 − p2|

 =

−
√2
√10
2
R(G; p1, p2) =

*. Then*
CR(G; p1, p2) = [+1, D(p1, p2), R(G; p1, p2)]. Replacing p2 with p3*, we find* sign(p1, p3) = sign

2 −1
0 −1
 =
0 −2 −
√2
−2 0 
√10
−
√2√10 0
−1, D(p1, p3) =

*, and*

−|p2|
|p2 − p1|
|p2 − p3|

 =

−
√2
√10
2
R(G; p1, p3) =

*. The final*
triple is CR(G; p1, p3) = [−1, D(p1, p3), R(G; p1, p3)].
Though a Centered Representation CR(G; p1, p2) will suffice to reconstruct G ⊂ R
2 uniquely under rigid motion in Theorem 4.6(b), CR(G; p1, p2) for all vertices p1, p2 ∈
V (G) should be considered in a joint unordered collection below to guarantee the independence of points p1, p2.

Definition 3.5 (Nested Centered Distribution NCD(G; h)).

Let G ⊂ R
2 be any Euclidean graph with m unordered vertices and the center of mass at the origin 0 ∈ R
n.

(a) The Nested Centered Distribution NCD(G; 1) *of order 1* is the unordered set of Centered Representations CR(G; p1)
from Definition 3.3 *for all vertices* p1 ∈ V (G).

(b) For any vertex p1 ∈ V (G)*, the* Centered Distribution CD1(G; p1) *is the unordered set of* CR(G; p1, p2) *for all* p2 ∈ V (G) \ {p1}*. The* Nested Centered Distribution NCD(G; 2) *of order 2 is the unordered set of* CD1(G; p1) for all vertices p1 ∈ V (G), see Fig. 2 *(top). The* mirror image NCD(G; 2) *is obtained from* NCD(G; 2) *by reversing* sign(p1, p2) of 2 × 2 *determinants in all* CR(G; p1, p2).

The nested structure of NCD(G; 2) helps identify edges between all vertices from G. After any vertex q ∈ V (G) \
{p1, p2} is uniquely located by using one CR(G; p1, p2),
we can use unsigned distances to associate any such q with its unique CR(G; p1, q) in the collection {CR(G; p1, pi)} for all pi ∈ V (G) \ {p1}. The resulting CR(G; p1, q) contains signed distances and hence detects edges from q to all other vertices, see details in the proof of Theorem 4.6(b).

s Figure 2. Top: building the Nested Centered Distribution NCD
in Definition 3.5 from Centered Representations in Definition 3.3 with metrics in section 4. **Bottom**: hierarchy of graph invariants.

SRD(G) can be considered NCD(G; 0) of order 0, containing signed distances from the center of mass p0 to all vertices of G, additionally written in increasing order.

220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274
(a) *For order* h = 1*, take any base vertices* p ∈ V (G) and q ∈ V (F)*. Define the* max metric M∞*(CR(*G; p), CR(F; q) as the maximum of | |p| − |q| | and the bottleneck distance W∞ between the fixed clouds of unordered points { (−|p
′|, d(p
′, p)) | p
′ ∈ V (G) − {p} }
and { (−|q
′|, d(q
′, q)) | q
′ ∈ V (F) − {q} } in R
2.

(b) *For order* h = 2*, take any base sequences* A ⊂ V (G) and B ⊂ V (F) of two vertices. Consider the m − 2 columns of R(G; A) from Definition 3.3 as a cloud of m − 2 *unordered points in* R
2*, also for* R(F; B)*. The* max metric M∞*(CR(*G; A), CR(F; B)) is the maximum of 2 λ2 |sign(A)σ(0 ∪ A) − sign(B)σ(0 ∪ B)|, L∞(D(A), D(B)), and W∞(R(G; A), R(F; B)). The maximum of several distances in Definition 4.3 is needed to guarantee the first metric axiom, i.e. M∞(CR(G; A), CR(F; B)) = 0 should imply that 0 ∪ A
should be exactly matched by rotation with 0 ∪ B and then CR(G; A) = CR(F; B) up to a permutation of columns will imply that G coincides with F, see Lemma D.7. To get a metric on Nested Centered Distributions, we will use the distance on bipartite graphs whose edge weights are the max metrics M∞ on Centered Representations. Definition 4.4 (Bottleneck Matching Distance BMD(Γ)).

## 4. Continuous Metrics On Graph Invariants

When points 0 ∪ A = (p0, p1, p2) ⊂ R
2 pass through a degenerate configuration in a straight line, i.e. p1, p2 become collinear, sign(A) discontinuously changes. To guarantee the Lipschitz continuity, we multiply such a sign by the strength σ below, which smooths the sign change, while the area of the triangle on p0, p1, p2 is not Lipschitz continuous.

Definition 4.1 (strength σ(C)). Any triple C =
{p0, p1, p2} ⊂ R
2 defines a triangle with inter-point distances a, b, c*, and half-perimeter* p =
1 2
(a + b + c)*. The* strength is σ(C) = 
(p − a)(p − b)(p − c)
p 2.

Lemma 4.2 (Theorem 4.4 in (Widdowson & Kurlin, 2023)).

Let B *be obtained from a set* C ⊂ R
2 *of 3 points by* perturbing every point within its ε*-neighborhood. Then* |σ(B) − σ(C)| ≤ 2ελ2 for λ2 = 2√3.

The strength σ(A) will be normalized by λ2 below to guarantee the final Lipschitz constant 2 for a metric in Theorem 4.6(c). For any k × k matrices *M, N* of real numbers, the metric L∞ is max i,j=1,...,k |Mij − Nij |. The *bottleneck* distance between any clouds *A, B* of (the same number of) m unordered points in a metric space with a distance d is W∞(*A, B*) = min bijections g:A→B
max p∈A
d(g(p), p).

Definition 4.3 (max metric M∞ on CRs). Let Euclidean graphs *G, F* ⊂ R
n have m *unordered vertices.*
Let Γ be a complete bipartite graph with m white vertices and m *black vertices so that every white vertex is connected* to every black vertex by a single edge e *of a weight* w(e) ≥ 0. A vertex matching of the graph Γ is a collection E of m disjoint edges with 2m *distinct vertices. The* weight W(E) = max e∈E
w(e) *is the largest weight of an edge in* E.

The Bottleneck Matching Distance BMD(Γ) = min E
W(E)
is the minimum weight of a vertex matching E of Γ. Since a graph Γ is complete bipartite, any edge from a vertex matching E in Γ joins a white vertex with a black vertex. Then BMD(Γ) is minimized for all bijections E between all white vertices and all black vertices of Γ. Definition 4.5 (Nested Bottleneck Metric NBM on NCDs).

Let *G, F* ⊂ R
2 be any graphs on m *unordered vertices.*
(a) *For order* h = 1*, the* Nested Bottleneck Metric NBM(NCD(G; 1), NCD(F; 1)) *is the max metric* M∞*(CR(*G; p), CR(F; β(p))) *minimized for all bijections* β : V (G) → V (F) between vertices of G and F.

(b) *For order* h = 2, any base vertices p1 ∈ V (G) and q1 ∈ V (F), let the complete bipartite graph Γ(G; p1; F; q1) *have* m − 1 white vertices and m − 1 *black vertices representing* CR(G; p1, p2) and CR(F; q1, q2) *for all* p2 ∈ V (G)−{p1} and q2 ∈ V (F)− {q1}*, respectively. Set the* weight w(e) of an edge e *joining the vertices represented by* CR(G; p1, p2) and CR(F; q1, q2) as the max metric M∞ *between these* distributions, see Definition 4.3. Then Definition 4.4 gives the bottleneck matching distance BMD(Γ(G; p1; F; q1)).

Let the complete bipartite graph Γ(G, F) *have weight* BMD(Γ(G; p1; F; q1)) *on each edge connecting vertices* representing p1 ∈ V (G) and q1 ∈ V (F)*. The* Nested Bottleneck Metric NBM(NCD(G; 2), *NCD(*F; 2)) is the Bottleneck Matching Distance BMD(Γ(G, F)).

SRD(G) coincides with NCD(G; 0) after sorting, so NBM can be defined as L∞(SRD(G), SRD(F)) for order h = 0. The metrics W∞, M∞, NBM compare objects of the same size. To compare graphs with different numbers of vertices, M∞ in Definition 4.5 can be replaced with Earth Mover's Distance EMD in Definition C.2. All metric axioms and main Theorem 4.6 below are proved in appendices C and D for any dimension n ≥ 2 and orders 1 ≤ h ≤ n. Theorem 4.6 (NCD solves Problem 1.1). (a) *The Nested* Centered Distribution NCD(G; h) in Definition 3.5 is invariant under any rigid motion for all Euclidean graph G on m *unordered vertices and can be computed in time* O(n 2mh+1) *with space* O(n 3 + hmh+1) for h ≤ n = 2.

(b) NCD(G; 2) *is a complete invariant of all graphs* G ⊂
R 
2 *under rigid motion from the group* SE(2).

(c) *Perturbing each vertex of a graph* G ⊂ R
2 within its εneighborhood changes NCD(G; h) up to 2ε *in both metrics* 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 NBM and EMD *for any order* h = 1, 2.

(d) For any graphs *G, F* ⊂ R
2 on m *unordered vertices, the* metrics NBM and EMD *between the invariants* NCD(G; h)
and NCD(F; h) *is computed in time* O(m2h+1.5logh+1 m) with space O(m2h+1 logh−1m) for h ≤ n = 2. Theorem 4.6(b) implies that any graphs *G, F* ⊂ R
2are related by rigid motion *if and only if* NCD(G; 2) = NCD(F; 2). This equality is interpreted as a bijection NCD(G; n) → NCD(F; n) matching all CRs, which is equivalent to NBM = 0 by the first metric axiom. Since every CR can be stored in a vector form, the complete invariant NCD(G; 2) for n = 2 can be considered vectorial. Table 2 emphasizes that most graphs should be first compared (or represented for machine learning) by simpler and faster invariants, so the complete NCD(G; n) is used only in rare cases but is still needed to distinguish all graphs.

| INVARIANT   | TIME        | METRIC   | TIME           |
|-------------|-------------|----------|----------------|
| SRD(G)      | O(m log m)  | L∞       | O(m)           |
| SPD(G)      | O(m2 )      | L∞       | O(m2 )         |
| PDD(G)      | O(m2 log m) | EMD      | O(m3 )         |
| NCD(G; 1)   | O(m2 )      | NBM      | O(m3.5 log2 m) |
| NCD(G; 2)   | O(m3 )      | NBM      | O(m5.5 log3 m) |

Example 4.7 (version of Theorem 4.6(b) for n = 1). For a graph G ⊂ R *with the center of mass* O(G) = 0*, take* any base vertex p ∈ G*. Then* sign(p) is the usual sign of p ∈ R, D(p) is the signed distance −|p|, R(G; p) is the 2 × (m − 1) *matrix whose column for any vertex* q ∈ V (G)− {p} *consists of the signed distances* d(q, 0) = −|q| and d(q, p) = ±|q − p|, where the plus sign + *indicates an* edge between q, p, while the minus sign − means no edge. For order h = 1*, the Centered Representation is the pair* CR(G; p) = [sign(p), −|p|, R(G; p)]. The base vertex p is fixed in the line R by sign(p) and |p|*. Any other vertex* q ∈ V (G) − {p} is uniquely determined in R by its Euclidean distances |q|, |q − p| *to the origin and the already* fixed p. The location of any point q ∈ R *is characterized* by sign(q) and |q|*, which helps unambiguously identify its* Centered Representation CR(G; q) in the unordered collection NCD(G; 1) of all these CRs. The signs of d(q, q′) in each R(G; q) determine the presence or absence of an edge of G ⊂ R between any vertices *q, q*′ ∈ V (G).

## 5. **Experiments On Largest Molecular Databases**

The world's largest databases of 3D molecular geometry are QM9 (130K+ entries) (Ramakrishnan et al., 2014) and GD (GEOM drugs of 31M+ entries) (Axelrod & Gomez-
Bombarelli, 2022), which have hundreds of 3D conformers of *unordered* atoms for each of 621 and 61607 chemical compositions, respectively. The Protein Data Bank has backbones of *ordered* atoms classified by simpler invariants (Anosova et al., 2025). All experiments took a few hours on Ryzen 9 3950X 3.5 GHz, 64 MB of L3 cache, RAM 82GB. The ICML guide for reviewing application-driven ML says that "novel ideas that are simple to apply may be especially valuable". To demonstrate the chemical importance of the linear-time invariant SRD, we extracted clouds of k = 10 neighbors around every atom, see their counts in Table 3.

QM9: H QM9: C QM9: N QM9: O QM9: F 1,230,122 846,557 139,764 187,996 3,314 GD0: H GD0: C GD0: N GD0: O GD0: F 5,660,986 5,267,096 842,562 854,400 64,299 GD0: P GD0: S GD0: Cl GD0: Br GD0: I 1,350 159,648 53,404 14,010 225

Though the data was skewed towards more popular elements H (hydrogen) and C (carbon), a default network in TensorFlow with 80/20 split for train/test achieved over 98% accuracy in predictions of the chemical element of a central atom by distances to only k = 3 nearest neighbours, see Table 4. Appendix A has all implementation details.

Table 4. Accuracies in percentages for predicting the chemical element of a central atom by a 4-layer network using *only the* k shortest distances to atomic neighbors within a molecular graph.

330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 In chemistry, both ML and non-ML predictions of elements achieved only 86% on similar size data, see Table 7 summarized in (Vasylenko et al., 2025), because the underlying descriptors were not invariant, e.g. under permutations of atoms, which creates exponentially many representations of the same molecule, incomplete, or their similarities failed the triangle axiom, e.g. see (Steck et al., 2024). High accuracies in Table 4 are rigorously explained by the cascade comparisons on all atomic clouds (environments) from QM9. Split all clouds from by the 1st distance (to the nearest neighbor of a central atom p) rounded to 3 decimal places in A˚ . This is a typical experimental precision, where 1A˚ = 10−10m is approximately the smallest interatomic distance. Second, split each subset with equal 1st distances by 2nd distances, and so on up to k = 5 distances. All 2.4M+
atomic clouds of different elements in QM9 were separated by the shorest distances to only 4 atomic neighbors.

| data   | k = 2   | k = 3   | k = 4   | k = 5   | k = 6   |
|--------|---------|---------|---------|---------|---------|
| QM9    | 94.63   | 98.64   | 98.24   | 98.54   | 98.77   |

The hierarchy of invariants in Fig. 2 and Table 2 transparently explained the reconstruction of chemical elements from distances to k nearest neghbors and inspired the harder task to reconstruct a chemical composition from a moleculelevel (not atomwise) invariant of only atomic centers. For molecular graphs from QM9, we computed the pseudometric L∞ (max absolute difference of corresponding coordinates) on all 873,527,974 pairs of SRDs, then 8,735,279 distances L∞ on the stronger SPDs for the 1% closest pairs, then 87,352 EMDs on PDDs for the 1% closest pairs, distances NBM on NCD(G; 1) and NCD(G; 2) for the top 10K closest pairs, and 64 NBMs on complete NCD(G, 3). The invariants in Table 5 distinguish all chemically different molecules with NBM on complete invariants giving the largest separation. All chemical compositions in QM9 and GD were distinguished by the vector SRD of Euclidean distances (rounded to 3 decimal places in A˚ ) from the molecular center of mass to 5 and 7 farthest atoms, respectively. This transparent reconstruction of the full chemistry from precise enough atomic geometry gives hope to rigorously infer other molecular properties from geometric invariants.

Table 5. Chemically different molecules (given by QM9 ids) are geometrically distinguished by invariant metrics, see Fig. 3 (right).

SRD, L∞ = 0.021, H3C4N3O2(131923)̸=H4C5N2O(5365) SPD, L∞ = 0.055, H3C4N5(123533)̸=H3C5N3O(24547) PDD, EMD = 0.051, H3C4N5(123533)̸=H3C5N3O(24521) NCD, NBM = 0.071, H4C5N4
(123532)̸=H4C6N2O(24513)
Figure 3. **Left**: the smallest NBM ≈ 0.07A˚ on NCD(G; 3) for chemically different molecules 123533 and 24521. **Right**: nearduplicate (almost flat) molecules 123532 and 24513 have the same composition and tiny EMD ≈ 2.37 × 10−7A˚ (not distinguishing mirror images) but a 100× higher NBM ≈ 2.95 × 10−5A. ˚ For QM9 molecul graphs, Fig. 4 and 9 NBM distances for different NCD invariants of orders h = 1, 2, 3.

| smallest distances in A, molecule A ˚ ̸= molecule B   |
|------------------------------------------------------|

## 6. Discussion: Conclusions And Limitations

The comparisons of molecular graphs from QM9 and GD imply that all chemically different molecules are rigidly different, see the smallest distance NBM ≈ 0.07A˚ on complete invariants in Table 5. So the map {molecules} → {graphs on atomic centers (without chemical elements)} is injective on rigid classes and can be inverted on its image. Hence the most important property (chemical composition)
is reconstructable from precise enough geometry. Using only a few radial distances (5 at the atomic level and 7 at the molecular level, rounded to 3 decimal places) for uniquely identifying all chemical elements in QM9 and GD demonstrates the transparency of application-driven ML. The solution to Problem 1.1 settled the long-standing challenge of properly defining a *molecular structure*. A traditional approach is to describe such a structure as "a set of unlabeled configurations that are relatively similar to each other", quoted from the paragraph to the left of the caption of Fig. 1 in (Lang et al., 2024). If this 'similarity' is treated as an equivalence allowing perturbations of atoms up to a positive threshold, sufficiently many perturbations can make all molecules (of the same number of atoms) equivalent by the transitivity axiom. A justified way to resolve this paradox is to embrace uncertainty and continuously quantify this similarity not by ignoring any perturbations up to a threshold but by computing an exact distance satisfying all metric axioms and Lipschitz continuity in Problem 1.1.

The question of whether to put close neighbors like nearduplicates in Fig. 3 (left) into one cluster of the "same" molecules is rather administrative similar to assigning close houses to one village (cluster) instead of different ones.

385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 Studying molecules by fixing a composition is similar to drawing artificial boundaries between countries on Earth. Because some molecules of different compositions have close shapes as in Fig. 5, they should have similar properties. Now any properties of molecules should be possible to predict only from the complete invariant NCD(G; 3) even without chemistry in the same way as any precise geographic location uniquely determines all physical properties of this place such as the average annual temperature. Chemical compositions can be still helpful similar to the location's altitude, which easier predicts (say) the average temperature than theoretically sufficient geographic coordinates. Any vertex p and edge of G can have an *attribute* and a weight respected by any isometry that maps one graph to another. These vertex attributes and edge weights can be incorporated as extra columns and rows in CRs from Definition 3.3, and then incorporated into NCD and NBM. We can compare graphs of different numbers of vertices because EMD works for both PDD and NCD as weighted distributions of any finite size. This comparison splits the vertices from V (G) into parts (subvertices) that are optimally 'transported' to a splitting of another vertex set V (F). The main contribution is Theorem 4.6 and its extension in Theorem D.1 to all dimensions n ≥ 2 fully solving Problem 1.1. The limitation is the time O(n 2mn+1) of the complete invariant NCD(G; n) of any graphs G ⊂ R
n. For a fixed dimension n, this polynomial complexity resolves two exponential-size challenges: m! permutations of m unordered vertices and up to 2 m(m−1)/2 non-isometric graphs with up to m(m − 1)/2 edges on m fixed vertices in R
n.

In practice, all comparisons and property predictions can start from much faster (linear-time) invariants SRD and only in cases of close distances (potential confusions) progress to stronger invariants SPD,PDD, NCD. This hierarchical (cascade) computation can better address the curse of dimensionality instead of the one-size-fits-all approach.

A map f : objects → descriptors → properties is invertible only if objects are faithfully represented by complete invariants. Any non-invariant maps a single object to (usually infinitely) many values or representations. Any incomplete invariant can fail to differentiate between objects with different properties. Hence a *generative* approach (inverting f above) can succeed only after the *discriminative* problem is solved. The space GRS(R
3; m) of rigid classes of all graphs on m vertices in R
3contains all possible shapes of molecules (all already known and also all not yet discovered ones). The complete invariant NCD(G) of G ⊂ R
3 defines geographic-style coordinates on a continuous map of GRS(R
3; m) containing QM9 and GD. Since the space GRS(R
3; m) is high-dimensional, we really need complete invariants to separate all known molecules and look for unexplored gaps containing new future molecules.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

Abbe, E. and Sandon, C. On the universality of deep learning. *Advances in Neural Information Processing Systems*, 33:20061–20072, 2020.

Agarwal, P. K., Fox, K., Nath, A., Sidiropoulos, A., and Wang, Y. Computing the gromov-hausdorff distance for metric trees. *ACM Transactions on Algorithms*, 14(2): 1–20, 2018.

Akhtar, N. and Mian, A. Threat of adversarial attacks on deep learning in computer vision: A survey. *IEEE Access*, 6:14410–14430, 2018.

Amir, T., Gortler, S., Avni, I., Ravina, R., and Dym, N.

Neural injective functions for multisets, measures and graphs via a finite witness theorem. *Advances in Neural* Information Processing Systems, 36, 2024.

Anosova, O., Gorelov, A., Jeffcott, W., Jiang, Z., and Kurlin, V. A complete and bi-continuous invariant of protein backbones under rigid motion. MATCH Communications in Mathematical and in Computer Chemistry (to appear), arxiv:2410.08203, 2025.

Arvind, V. and Rattan, G. The parameterized complexity of geometric graph isomorphism. *Algorithmica*, 75:258– 276, 2016.

440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 Batzner, S., Musaelian, A., Sun, L., Geiger, M., Mailoa, J. P., Kornbluth, M., Molinari, N., Smidt, T. E., and Kozinsky, B. E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. Nature communications, 13(1):2453, 2022.

Bonchev, D. Chemical graph theory: introduction and fundamentals, volume 1. CRC Press, 1991.

Brecheteau, C. A statistical test of isomorphism between ´
metric-measure spaces using the distance-to-a-measure signature. *Electronic J Statistics*, 13:795–849, 2019.

Bronstein, M. M., Bruna, J., LeCun, Y., Szlam, A., and Vandergheynst, P. Geometric deep learning: going beyond Euclidean data. *IEEE Signal Processing Magazine*, 34
(4):18–42, 2017.

Bronstein, M. M., Bruna, J., Cohen, T., and Velickovi ˇ c,´
P. Geometric deep learning: grids, groups, graphs, geodesics, and gauges. *arXiv:2104.13478*, 2021.

Buchin, M., Chambers, E., Fang, P., Fasy, B. T., Gasparovic, E., Munch, E., and Wenk, C. Distances between immersed graphs: Metric properties. *La Matematica*, pp. 1–26, 2023.

Bunch, J. R. and Hopcroft, J. E. Triangular factorization and inversion by fast matrix multiplication. Mathematics of Computation, 28(125):231–236, 1974.

Chen, C., Ye, W., Zuo, Y., Zheng, C., and Ong, S. P. Graph networks as a universal machine learning framework for molecules and crystals. *Chemistry of Materials*, 31(9):
3564–3572, 2019.

Chew, P. and Kedem, K. Improvements on geometric pattern matching problems. In Scandinavian Workshop on Algorithm Theory, pp. 318–325, 1992.

Chew, P., Dor, D., Efrat, A., and Kedem, K. Geometric pattern matching in d-dimensional space. Discrete & Computational Geometry, 21(2):257–274, 1999.

Choo, H. Y., Wee, J., Shen, C., and Xia, K. Fingerprintenhanced graph attention network (fingat) model for antibiotic discovery. Journal of Chemical Information and Modeling, 2023.

Cohen, T. S., Geiger, M., and Weiler, M. A general theory of equivariant cnns on homogeneous spaces. Advances in neural information processing systems, 32, 2019.

Colbrook, M. J., Antun, V., and Hansen, A. C. The difficulty of computing stable and accurate neural networks: On the barriers of deep learning and Smale's 18th problem. Proc. National Academy of Sciences, 119(12):e2107151119, 2022.

Dekster, B. V. and Wilker, J. B. Edge lengths guaranteed to form a simplex. *Archiv der Mathematik*, 49(4):351–366, 1987.

Axelrod, S. and Gomez-Bombarelli, R. Geom, energyannotated molecular conformations for property prediction and molecular generation. *Scientific Data*, 9(1):185, 2022.

Babai, L. Canonical form for graphs in quasipolynomial time: preliminary report. In Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing, pp. 1237–1246, 2019.

Antunes, L. M., Grau-Crespo, R., and Butler, K. T. Distributed representations of atoms and materials for machine learning. *npj Computational Materials*, 8(1):44, 2022.

Delle Rose, V., Kozachinskiy, A., Rojas, C., Petrache, M.,
and Barcelo, P. Three iterations of (d- 1)-wl test dis- ´ tinguish non isometric clouds of d-dimensional points. Advances in Neural Information Processing Systems, 36, 2024.

Deng, C., Litany, O., Duan, Y., Poulenard, A., Tagliasacchi, A., and Guibas, L. J. Vector neurons: A general framework for so(3)-equivariant networks. In *Proceedings of* the International Conference on Computer Vision, pp. 12200–12209, 2021.

Deza, E. and Deza, M. M. *Encyclopedia of distances*.

Springer, 2009.

Dong, Y., Liao, F., Pang, T., Su, H., Zhu, J., Hu, X., and Li, J. Boosting adversarial attacks with momentum. In Computer vision and pattern recognition, pp. 9185–9193, 2018.

Duvenaud, D. K., Maclaurin, D., Iparraguirre, J., Bombarell, R., Hirzel, T., Aspuru-Guzik, A., and Adams, R. P. Convolutional networks on graphs for learning molecular fingerprints. *Advances in neural information processing* systems, 28, 2015.

Dym, N. and Gortler, S. J. Low-dimensional invariant embeddings for universal geometric learning. *Foundations* of Computational Mathematics, pp. 1–41, 2024.

Efrat, A., Itai, A., and Katz, M. J. Geometry helps in bottleneck matching and related problems. *Algorithmica*, 31(1):1–28, 2001.

Feynman, R. The Feynman lectures on physics. Chapter 1:
atoms in motion, volume 1. 1971.

Fisikopoulos, V. and Penaranda, L. Faster geometric algorithms via dynamic determinant computation. Computational Geometry, 54:1–16, 2016.

Fredman, M. L. and Tarjan, R. E. Fibonacci heaps and their uses in improved network optimization algorithms. Journal of the ACM, 34(3):596–615, 1987.

Fuchs, F., Worrall, D., Fischer, V., and Welling, M. Se(3)-
transformers: 3d roto-translation equivariant attention networks. Advances in neural information processing systems, 33:1970–1981, 2020.

Gao, X., Hu, W., and Qi, G.-J. Graphter: Unsupervised learning of graph transformation equivariant representations via auto-encoding node-wise transformations. In Proceedings of Computer Vision and Pattern Recognition, pp. 7163–7172, 2020.

Goldberg, A. and Tarjan, R. Solving minimum-cost flow problems by successive approximation. In Proceedings of STOC, pp. 7–18, 1987.

Goodrich, M. T., Mitchell, J. S., and Orletsky, M. W. Approximate geometric pattern matching under rigid motions. *Transactions on Pattern Analysis and Machine* Intelligence, 21(4):371–379, 1999.

Goyal, A., Law, H., Liu, B., Newell, A., and Deng, J. Revisiting point cloud shape classification with a simple and effective baseline. In International Conference on Machine Learning, pp. 3809–3820, 2021.

Grinberg, D. and Olver, P. J. The n body matrix and its determinant. *SIAM Journal on Applied Algebra and Geometry*, 3(1):67–86, 2019.

Grohe, M. and Schweitzer, P. The graph isomorphism problem. *Communications of the ACM*, 63(11):128–134, 2020.

Guo, C., Gardner, J., You, Y., Wilson, A. G., and Weinberger, K. Simple black-box adversarial attacks. In International Conference on Machine Learning, pp. 2484–2493, 2019.

Helfgott, H. A., Bajpai, J., and Dona, D. Graph isomorphisms in quasi-polynomial time. *arXiv:1710.04574*,
2017.

Hopcroft, J. E. and Karp, R. M. An nˆ5/2 algorithm for maximum matchings in bipartite graphs. SIAM Journal on Computing, 2(4):225–231, 1973.

Hordan, S., Amir, T., Gortler, S. J., and Dym, N. Complete neural networks for euclidean graphs. In AAAI Conference on Artificial Intelligence, volume 38 (11), pp.

12482–12490, 2024.

Horn, R. A. and Johnson, C. R. *Matrix analysis*. Cambridge University Press, 2012.

Huttenlocher, D. P., Klanderman, G. A., and Rucklidge, W. J.

Comparing images using the Hausdorff distance. Transactions on pattern analysis and machine intelligence, 15
(9):850–863, 1993.

Kiefer, S., Ponomarenko, I., and Schweitzer, P. The weisfeiler–leman dimension of planar graphs is at most 3. Journal of the ACM, 66(6):1–31, 2019.

Kondor, R. and Trivedi, S. On the generalization of equivariance and convolution in neural networks to the action of compact groups. In International Conference on Machine Learning, pp. 2747–2755, 2018.

Laidlaw, C. and Feizi, S. Functional adversarial attacks.

Adv. Neural Information Proc. Systems, 32, 2019.

Lang, L., Cezar, H. M., Adamowicz, L., and Pedersen, T. B.

Quantum definition of molecular structure. Journal of the American Chemical Society, 146(3):1760–1764, 2024.

495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 Leman, A. and Weisfeiler, B. A reduction of a graph to a canonical form and an algebra arising during this reduction. *Nauchno-Technicheskaya Informatsiya*, 2(9):12–16, 1968.

Nikolentzos, G., Meladianos, P., and Vazirgiannis, M.

Matching node embeddings for graph similarity. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 31, 2017.

550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 Oliynyk, A. O., Antono, E., Sparks, T. D., Ghadbeigi, L., Gaultois, M. W., Meredig, B., and Mar, A. Highthroughput machine-learning-driven synthesis of fullheusler compounds. *Chemistry of Materials*, 28(20):
7324–7331, 2016.

Li, Z., Wang, X., Huang, Y., and Zhang, M. Is distance matrix enough for geometric deep learning? arXiv:2302.05743, 2023.

Liberti, L. and Lavor, C. *Euclidean distance geometry*.

Springer, 2017.

Pozdnyakov, S. N., Willatt, M. J., Bartok, A. P., Ortner, C., ´
Csanyi, G., and Ceriotti, M. Incompleteness of atomic ´
structure representations. *Phys. Rev. Lett.*, 125:166001, 2020. URL arXiv:2001.11696.

Lim, S., Memoli, F., and Smith, Z. The gromov–hausdorff ´
distance between spheres. *Geometry & Topology*, 27(9):
3733–3800, 2023.

Qi, G.-J. and Luo, J. Small data challenges in big data era:
A survey of recent progress on unsupervised and semisupervised methods. Transactions on Pattern Analysis and Machine Intelligence, 44(4):2168–2187, 2020.

Maennel, H., Unke, O. T., and Muller, K.-R. Complete ¨
and efficient covariants for 3d point configurations with application to learning molecular quantum properties. arXiv:2409.02730, 2024.

Ramakrishnan, R., Dral, P. O., Rupp, M., and Von Lilienfeld, O. A. Quantum chemistry structures and properties of 134 kilo molecules. *Scientific data*, 1(1):1–7, 2014.

Majhi, S. and Wenk, C. Distance measures for geometric graphs. *arXiv:2209.12869*, 2022.

Majhi, S., Vitter, J., and Wenk, C. Approximating gromovhausdorff distance in euclidean space. *Computational* Geometry, 116:102034, 2024.

Rass, S., Konig, S., Ahmad, S., and Goman, M. Metricizing ¨
the euclidean space towards desired distance relations in point clouds. IEEE Transactions on Information Forensics and Security, 2024.

Maron, H., Ben-Hamu, H., Serviansky, H., and Lipman, Y.

Provably powerful graph networks. Advances in neural information processing systems, 32, 2019.

Rieder, S. R., Oliveira, M. P., Riniker, S., and Hunenberger, ¨
P. H. Development of an open-source software for isomer enumeration. *Journal of Cheminformatics*, 15(1):10, 2023.

Melter, R. A. and Tomescu, I. Metric bases in digital geometry. *Computer vision, graphics, and image Processing*,
25(1):113–121, 1984.

Rubner, Y., Tomasi, C., and Guibas, L. The Earth Mover's Distance as a metric for image retrieval. International Journal of Computer Vision, 40(2):99–121, 2000.

Memoli, F. Gromov–Wasserstein distances and the metric ´
approach to object matching. Foundations of computational mathematics, 11:417–487, 2011.

Sato, R., Cuturi, M., Yamada, M., and Kashima, H. Fast and robust comparison of probability measures in heterogeneous spaces. *arXiv:2002.01615*, 2020.

Memoli, F., Smith, Z., and Wan, Z. The Gromov-Hausdorff ´
distance between ultrametric spaces: its structure and computation. *arXiv:2110.03136*, 2021.

Schmiedl, F. Computational aspects of the Gromov–
Hausdorff distance and its application in non-rigid shape matching. *Discrete Comp. Geometry*, 57:854–880, 2017.

Merchant, A., Batzner, S., Schoenholz, S. S., Aykol, M.,
Cheon, G., and Cubuk, E. D. Scaling deep learning for materials discovery. *Nature*, 624(7990):80–85, 2023.

Schoenberg, I. Remarks to Maurice Frechet's article "Sur la definition axiomatique d'une classe d'espace distances vectoriellement applicable sur l'espace de Hilbert. Annals of Mathematics, pp. 724–732, 1935.

Nemec, L. Principal component analysis (pca):
A physically intuitive mathematical introduction. https://towardsdatascience.com/ principal-component-analysis-pca-8133b02f11bd, 2022.

Shirdhonkar, S. and Jacobs, D. Approximate earth mover's distance in linear time. In Conference on Computer Vision and Pattern Recognition, pp. 1–8, 2008.

Nigam, J., Pozdnyakov, S. N., Huguenin-Dumittan, K. K.,
and Ceriotti, M. Completeness of atomic structure representations. *APL Machine Learning*, 2(1), 2024.

Sippl, M. J. and Scheraga, H. A. Cayley-menger coordinates.

Proceedings of the National Academy of Sciences, 83(8):
2283–2287, 1986.

Steck, H., Ekanadham, C., and Kallus, N. Is cosinesimilarity of embeddings really about similarity? In Companion Proceedings of the ACM on Web Conference 2024, pp. 887–890, 2024.

Tshitoyan, V., Dagdelen, J., Weston, L., Dunn, A., Rong, Z., Kononova, O., Persson, K. A., Ceder, G., and Jain, A. Unsupervised word embeddings capture latent knowledge from materials science literature. *Nature*, 571(7763):95– 98, 2019.

Tu, E., Wang, Z., Yang, J., and Kasabov, N. Deep semisupervised learning via dynamic anchor graph embedding in latent space. *Neural Networks*, 146:350–360, 2022.

Vasylenko, A., Antypov, D., Schewe, S., Daniels, L. M.,
Claridge, J. B., Dyer, M. S., and Rosseinsky, M. J. Digital features of chemical elements extracted from local geometries in crystal structures. *Digital Discovery*, 2025.

Ward, L., Agrawal, A., Choudhary, A., and Wolverton, C. A
general-purpose machine learning framework for predicting properties of inorganic materials. npj Computational Materials, 2(1):1–7, 2016.

Weinhold, F. Metric geometry of equilibrium thermodynamics. *The Journal of Chemical Physics*, 63(6):2479–2483, 1975.

Weston, L., Tshitoyan, V., Dagdelen, J., Kononova, O.,
Trewartha, A., Persson, K. A., Ceder, G., and Jain, A. Named entity recognition and normalization applied to large-scale information extraction from the materials science literature. Journal of chemical information and modeling, 59(9):3692–3702, 2019.

Widdowson, D. and Kurlin, V. Resolving the data ambiguity for periodic crystals. Advances in Neural Information Processing Systems, 35:24625–24638, 2022.

Widdowson, D. and Kurlin, V. Recognizing rigid patterns of unlabeled point clouds by complete and continuous isometry invariants with no false negatives and no false positives. In *Proceedings of CVPR*, pp. 1275–1284, 2023.

Wilson, S. R., Cui, W., Moskowitz, J. W., and Schmidt, K. E. Applications of simulated annealing to the conformational analysis of flexible molecules. Journal of computational chemistry, 12(3):342–349, 1991.

Zhang, B., Fan, C., Liu, S., Huang, K., Zhao, X., Huang, J., and Liu, Z. The expressive power of graph neural networks: A survey. *IEEE Transactions on Knowledge* and Data Engineering, 37:1455–1474, 2024.

Zhou, D.-X. Universality of deep convolutional neural networks. *Applied and computational harmonic analysis*, 48
(2):787–794, 2020.

605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659 Zhou, Q., Tang, P., Liu, S., Pan, J., Yan, Q., and Zhang, S.-
C. Learning atoms for materials discovery. Proceedings of the National Academy of Sciences, 115(28):E6411– E6417, 2018.

## A. Extra Details Of Experiments On The World'S Largest 3D Molecular Databases Qm9 And Gd

The default 4-layer network from TensorFlow used a "sequential" mode, 3 epochs, and the settings in Table 6.

Table 6. Parameters of the TensorFlow network for predictions in Table 4.

| LAYER (TYPE)        | OUTPUT SHAPE   | NUMBER OF PARAMETERS   |
|---------------------|----------------|------------------------|
| DENSE (DENSE)       | (NONE, 32)     | 352                    |
| BATCH NORMALIZATION | (NONE, 32)     | 128                    |
| RE LU (RELU)        | (NONE, 32)     | 0                      |
| DENSE 1 (DENSE)     | (NONE, 5)      | 165                    |

Past maps of QM9 in Fig. 5 based on eigenvalues are too dense without clear separation. Even if we zoom in, these two or three incomplete invariants will not provide any extra separation. The complete invariants NDP contain much more geometric information.

660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714
Figure 5. **Left**: each dot represents one QM9 molecule whose atomic cloud has two largest roots l1 ≥ l2 of eigenvalues (moments of inertia (Nemec, 2022) or elongations in two principal directions) in Angstroms (1A˚ = 10−10m ≈ smallest interatomic distance). The color represents the free energy G characterizing molecular stability. **Right**: each dot represents one QM9 molecule whose atomic cloud has coordinates *x, y* expressed via the roots l1 ≥ l2 ≥ l3 ≥ 0 of three eigenvalues.

Fig. 7 shows the simplest geographic-style map of QM9 as a finite sample within S
29 m=3 GRS(R
3; m) projected to the invariants SRD1 ≥ SRD2. All molecules on the horizontal axis y = SRD1 − SRD2 = 0 have SRD1 = SRD2 (due to two equidistant atoms from the center of mass) and can be projected (like any subset of QM9) to other coordinates as in Fig. 8. Molecular properties can be visualized on these geographic maps as 'mountainous' landscapes.

Table 7. Past ML and non-ML predictions of chemical elements have lower accuracies than by distance invariants in Table 4.

| METHOD     | DESCRIPTION                                     | ACCURACY   | REFERENCE                |
|------------|-------------------------------------------------|------------|--------------------------|
| LEAF       | LOCAL COORDINATION GEOMETRY                     | 86%        | (VASYLENKO ET AL., 2025) |
| MATSCHOLAR | ML-DERIVED FROM LITERATURE                      | 81%        | (WESTON ET AL., 2019)    |
| MAT2VEC    | ML-DERIVED FROM LITERATURE                      | 80%        | (TSHITOYAN ET AL., 2019) |
| ATOM2VEC   | ML-DERIVED FROM COMPOSITIONAL CONTENT           | 79%        | (ZHOU ET AL., 2018)      |
| GNOME      | FREQUENCY OF ELEMENTS AT THE SAME ATOMIC SITES  | 79%        | (MERCHANT ET AL., 2023)  |
| MAGPIE     | ELEMENTAL PHYSICAL CHARACTERISTICS              | 78%        | (WARD ET AL., 2016)      |
| OLIYNYK    | ELEMENTAL PHYSICAL CHARACTERISTICS              | 75%        | (OLIYNYK ET AL., 2016)   |
| MEGNET     | ML-DERIVED FROM ATOM, BOND AND GRAPH ATTRIBUTES | 73%        | (CHEN ET AL., 2019)      |
| SKIPATOM   | ML-DERIVED FROM ATOM CONNECTIVITY GRAPHS        | 68%        | (ANTUNES ET AL., 2022)   |

Continuous machine learning on Euclidean graphs 715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769 Continuous machine learning on Euclidean graphs 770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 810 811 812 813 814 815 816 817 818 819 820 821 822 823 824

## B. Invariants And Metrics On Euclidean Graphs In Any Dimension N ≥ 2

This section extends all new concepts and results from sections 3 and 4 to any dimension n ≥ 2. Any n vectors p1*, . . . , p*n ∈ R
n can be written as columns in the n × n matrix whose determinant has sign(p1*, . . . , p*n), which is ±1 or 0
(if p1*, . . . , p*n are linearly dependent).

Definition B.1 (Centered Representation CR(G; A) of a graph with a sequence A ⊂ V (G)). Let G ⊂ R
n *be a graph on* m *unordered points with the center of mass* O(G) = 0. For any 1 ≤ h ≤ n*, fix a* base sequence A *of ordered vertices* p1, . . . , ph ∈ V (G). If h = n*, let* sign(A) be the sign of the n × n determinant on the vectors p1, . . . , pn*, else* sign(A) = 0. Let D(A) *be the matrix of signed distances between the ordered points* 0 = p0, p1, . . . , ph. The matrix R(G; A) has m − h unordered columns, one for each vertex q ∈ V (G) − A*, consisting of* h + 1 distances d(q, pi) for i = 0, . . . , h*, where* p0 = 0*. The* Centered Representation CR(G; A) *is the triple* [sign(A), D(A), R(G; A)].

Definition B.2 (Nested Centered Distribution NCD(G; h) of order h). Let G ⊂ R
n be any Euclidean graph on m unordered vertices and the center of mass at the origin 0 ∈ R
n*. Fix an* order 1 ≤ h ≤ n.

(a) For any h − 1 distinct ordered vertices p1, . . . , ph−1 ∈ V (G)*, the* Centered Distribution CD(h)
h−1
(G; p1*, . . . , p*h−1)
of index h − 1 *is the unordered set of Centered Representation* CR(G; p1, . . . , ph) from Definition B.1 for all ph ∈ V (G) − {p1*, . . . , p*h−1}.

(b) Now we will iteratively decrement an integer k from h − 1 *down to 1 and define* CD(h)
h−2 of index h − 2*, and so on until* CD(h)
1*of index* k = 1. For the initial k = h − 1*, we use* CD(h)
k = CD(h)
h−1 defined in part (a) above. For any k − 1 distinct

825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879
Figure 9. Each dot is a comparison of molecular graphs from QM9 by the distances on the progressively stronger invariants: NCD(G; 2) vs NCD(G; 3).

ordered vertices p1, . . . , pk−1 ∈ V (G)*, the* Centered Distribution CD(h)
k−1(G; p1, . . . , pk−1) of index k − 1 is the unordered collection of CD(h)
k(C; p1, . . . , pk) of index k for all vertices pk ∈ V (G) − {p1*, . . . , p*k−1}.

(c) The Nested Centered Distribution NCD(G; h) of order h *is the unordered collection of* CD(h)
1(G; p1) of index 1 for all vertices p1 ∈ V (G). For the order h = n*, the* mirror image NCD(G; n) *is obtained from* NCD(G; n) *by reversing* sign(p1, . . . , pn) of n × n *determinants in all* CR; p1*, . . . , p*n).

If a sequence 0 ∪ A = (p0, p1*, . . . , p*n) ⊂ R
n degenerates to a lower dimensional subspace, i.e. the vectors p1*, . . . , p*n become linearly dependent, then sign(A) of discontinuously changes. To guarantee the Lipschitz continuity, we multiply these signs by the strength σ below, while the volume vol(0 ∪ A) of the simplex on 0 ∪ A is not Lipschitz continuous.

Definition B.3 (strength σ(C)). *For any sequence* C of n + 1 ordered points p0*, . . . , p*n ∈ R
n*, the* half-perimeter p(C) = 
1 2P
1≤i<j≤n |pi − pj | is the half-sum of pairwise distances between points of C*. Let* vol(C) denote the volume of the n-dimensional simplex on C*. The* strength *of the simplex* C is σ(C) = vol2(C)
p 2n−1(C)
.

In dimension n = 1, for any pair C = {p0, p1} ⊂ R, the volume vol(C) is the length |p0 − p1|, the half-perimeter distance p(C) is the half-distance 12 |p0 − p1|, so the strength is σ(C) = 
vol2(C)
p(C)= 2|p0 − p1|.

Lemma B.4 (Theorem 4.4 in (Widdowson & Kurlin, 2023)). Let B *be obtained from a sequence* A ⊂ R
n of n points by perturbing every point within its ε-neighborhood. Then |σ(A)−σ(B)| ≤ 2ελn for a constant λn*, where* λ1 = 2, λ2 = 2√3, λ3 ≈ 0.43.

Definition B.5 (max metric M∞ on CRs). Let Euclidean graphs *G, F* ⊂ R
n on m *unordered vertices have base sequences* A, B of h ≤ n vertices. Consider the m − h columns of R(G; A) as a cloud of m − h *unordered points in* R
h, also for R(F; B)*. The* max *metric* M∞(CR(G; A), CR(F; B)) *is the maximum of* 2 λn |sign(A)σ(0 ∪ A) − sign(B)σ(0 ∪ B)|, L∞(D(A), D(B)), and the bottleneck distance W∞(R(G; A), R(F; B)), where all signs are zeros for *h < n*. In Definition B.5, λn is the Lipschitz constant of σ from Lemma 4.2.

Definition B.6 (Nested Bottleneck Metric NBM on NCDs). Let *G, F* ⊂ R
n be any Euclidean graphs on m unordered vertices. For any ordered vertices p1 . . . , ph−1 ∈ V (G) and q1 . . . , qh−1 ∈ V (F), the complete bipartite graph Γ(G; p1, . . . , ph−1; F; q1, . . . , qh−1) has m − h + 1 *white vertices and* m − h + 1 black vertices representing CR(G; p1, . . . , ph) and CR(F; q1, . . . , qh) *for all* m − h + 1 vertices ph ∈ V (G) − {p1, . . . , ph−1} and qh ∈ V (F) − {q1, . . . , qh−1}*, respectively. Set the* weight w(e) of an edge e *joining the vertices represented by* CR(G; p1*, . . . , p*h), CR(F; q1, . . . , qh) as the max metric M∞ between these distributions, see Definition B.5. Then Definition 4.4 *gives the bottleneck matching distance* BMD(Γ(G; p1, . . . , ph−1; F; q1*, . . . , q*h−1)). For any integer 1 ≤ i < h and ordered vertices p1 . . . , pi−1 ∈ V (G) and q1 . . . , qi−1 ∈ V (F), the complete bipartite graph Γ(G; p1, . . . , pi−1; F; q1, . . . , qi−1) has m − i + 1 *white vertices and* m − i + 1 *black* vertices representing CD(h)
i(G; p1, . . . , pi) and CD(h)
i(F; q1, . . . , qi) *for all* m − i + 1 *variable vertices* pi ∈
V (G) − {p1, . . . , pi−1} and qi ∈ V (F) − {q1, . . . , qi−1}*, respectively. Set the* weight w(e) of an edge e joining the vertices represented by CD(h)
i(G; p1*, . . . , p*i) and CD(h)
i(F; q1, . . . , qi) *as the previously computed distance* BMD(Γ(G; p1, . . . , pi; F; q1, . . . , qi)) for a smaller number i of fixed vertices. Then Definition 4.4 gives the bottleneck matching distance BMD(Γ(G; p1, . . . , pi−1; F; q1, . . . , qi−1))*. For* i = 1*, the graph* Γ(G, F) has m + m vertices representing CD1(G; p1), CD(h)
1(F; q1) for all p1 ∈ V (G) and q1 ∈ V (F)*. The* Nested Bottleneck Metric NBM(NCD(G; h), NCD(F; h)) *is the Bottleneck Matching Distance* BMD(Γ(*G, F*)).

## C. Metrics On Graphs And Their Continuity Under Perturbations

This appendix verifies the axioms and Lipschitz continuity for all auxiliary metrics in section 4. Lemma C.1 (metric axioms for the bottleneck matching distance BMD). Let S, Q be any unordered distributions of the same number of objects with a base metric d. Define the complete bipartite graph Γ(S, Q) whose every edge e *joining* objects RS ∈ S and RQ ∈ Q *has the weight* w(e) = d(RS, RQ)*. Then the bottleneck matching distance* BMD(Γ(*S, Q*))
from Definition 4.4 *satisfies all metric axioms on such unordered distributions.* Proof of Lemma *C.1.* The coincidence axiom means that NBM(*S, Q*) = 0 if and only if the weighted distributions *S, Q* are equal in the sense that there is a bijection g : S → Q so that d(g(R), R) = 0 for any R ∈ S.

Indeed, if the weighted distributions *S, Q* can be matched by a bijection, we get a vertex matching E of Γ(*S, Q*) whose all edges have weights w(e) = 0. Definition 4.4 implies that BMD(Γ(*S, Q*)) = 0 as required.

Conversely, if BMD(Γ(*S, Q*)) = 0, there is a vertex matching E in Γ(S, Q) with all w(e) = 0. This matching E defines a required bijection S → Q. The symmetry BMD(Γ(*S, Q*)) = BMD(Γ(*Q, S*)) follows from Definition 4.4 and the symmetry of the base metric d. To prove the triangle inequality BMD(Γ(S, Q)) + BMD(Γ(*Q, T*)) ≥ BMD(Γ(S, T)),
let ESQ, EQT be optimal vertex matchings in the graphs Γ(S, Q), Γ(*Q, T*), respectively, such that BMD(Γ(S, Q)) = W(ESQ), BMD(Γ(*Q, T*)) = W(EQT ),
880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 see Definition 4.4. The composition ESQ ◦ EQT is a vertex matching in Γ(*S, T*), so W(ESQ ◦ EQT ) ≥ BMD(Γ(S, T)).

It suffices to prove that W(ESQ) + W(EQT ) ≥ W(ESQ ◦ EQT ).

Let eST be an edge with a largest weight from ESQ ◦EQT , so W(ESQ ◦EQT ) = w(eST ). The edge eST can be considered the union of edges eSQ ∈ ESQ, eQT ∈ EQT .

By the triangle inequality for the base metric d, w(eSQ) + w(eQT ) ≥ w(eST ) = W(ESQ ◦ EQT )
implies that W(ESQ) + W(EQT ) ≥ W(ESQ ◦ EQT )
because both terms on the left-hand side are maximized for all edges (not only eSQ, eQT ) from ESQ, EQT . Definition C.2 below makes sense for any distributions {[R1, w1], . . . , [Rm, wm]}, where R1*, . . . , R*m are objects with a base metric d and weights w1*, . . . , w*m ∈ [0, 1]. Each Ri can be CBR or CBD of any depth with a base metric M∞ or BMD from Definitions B.5, B.6. Definition C.2 (EMD). Let S = {[Ri(S), wi(S)]}
m(S)
i=1 and Q = {[Rj (Q), wj (Q)]}
m(Q)
j=1 be weighted distributions of objects Ri(S), Rj (Q), which live in a space with a metric d. A flow from S to Q is an m(S) × m(Q) matrix whose element fij ∈ [0, 1] *represents a* partial flow from Ri(S) to Rj (Q)*. The* Earth Mover's Distance *is the minimum* cost EMD(*S, Q*) =
m P
(S)

```
i=1
      m
       P
        (Q)

```

j=1 fijd(Ri(S), Rj (Q)) for variable 'flows' fij ∈ [0, 1] *subject to the conditions* m P
(Q)
j=1 fij ≤ wi(S)
for i = 1*, . . . , m*(S),
m P
(S)
i=1 fij ≤ wj (Q) for j = 1, . . . , m(Q)*, and* m P
(S)

```
i=1
      m
       P
        (Q)

```

j=1 fij = 1.

The first condition m P
(Q)
j=1 fij ≤ wi(S) means that not more than the weight wi(S) of Ri(S) 'flows' into all Rj (Q) via 'flows' fij , j = 1*, . . . , m*(Q). The second condition m P
(S)
i=1 fij = wj (Q) means that all 'flows' fij from Ri(S) for i = 1*, . . . , m*(S)
'flow' into Rj (Q) up to the maximum weight wj (Q). The last condition m P
(S)

```
i=1
      m
       P
        (Q)

```

j=1 fij = 1 forces to 'flow' all rows Ri(S)
to all rows Rj (Q).

935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989 The EMD satisfies all metric axioms, see the appendix in (Rubner et al., 2000), needs O(m3log m) time for distributions of a maximum size m and is approximated in O(m) time, see (Shirdhonkar & Jacobs, 2008; Sato et al., 2020). Definition C.2 can be adapted for the EMD between NDDs by (1) replacing the bottleneck distance W∞ in Definition B.5 with EMD between clouds of equally weighted points, and (2) replacing BMD(Γ) for a bipartite graph Γ with EMD(Γ) between the unordered sets (of potentially different sizes) of BDDs with weights on all white vertices and BDDs on all black vertices. The Lipschitz continuity of NDD and EMD in Theorem D.1(c) needs Lemmas C.3, C.4, D.9. If a metric graph G lives in an ambient metric space X, a natural perturbation of G is a shift of every vertex of G up to ε in the metric of X. Then the distance d(*p, q*) between any vertices *p, q* of G changes by at most 2ε.

We will prove the continuity in more general settings by only assuming that d(p, q) changes by at most 2ε for any *p, q* ∈ V (G) without requiring an ambient space X.

Lemma C.3 (Lipschitz continuity of BMD). Let Γ be a complete bipartite graph with a vertex matching E *such that any* e ∈ E has a weight w(e) ≤ ε*. Then* BMD(Γ) ≤ ε.

Proof of Lemma *C.3.* By Definition 4.4, the given matching E has the weight W(E) = max e∈E
w(e) ≤ ε. Since BMD(Γ) =
min E
W(E) is minimized for all vertex matchings, we get BMD(Γ) ≤ ε.

$i_{\bar{j}}=\frac{1}{r}$
$\square$
Lemma C.4 (Lipschitz continuity of EMD). In Definition C.2, let distributions S, Q *have a bijection* Ri(S) ↔ Ri(Q) between equally weighted objects such that d(Ri(S), Ri(Q)) ≤ ε *for all* i = 1, . . . , m*, where* m = m(S) = m(Q)*. Then* EMD(*S, Q*) ≤ ε.

Pm
i=1
Pm
j=1
fijd(Ri(S), Rj (Q)) = Pm
$$R_{j}(Q))=\sum_{i=1}^{m}\frac{1}{m}d(R_{i}(S),R_{i}(Q))\leq\frac{1}{m}\sum_{i=1}^{m}\varepsilon=\varepsilon.$$

## D. Proofs For Euclidean Graphs From Section 3

This appendix rigorously proves all parts of Theorem D.1. Theorem D.1 (NCD solves Problem 1.1). (a) *The Nested Centered Distribution* NCD(G; h) in Definition B.2 is invariant under any rigid motion for all Euclidean graph G on m unordered vertices and, for a fixed dimension n, can be computed in time O(n 2mh+1) with space O(n 2mh+1) *for any order* 1 ≤ h ≤ n.

(b) NCD(G; 2) *is a complete invariant of all graphs* G ⊂ R
2 *under rigid motion from the group* SE(n) *in any dimension* n ≥ 1.

(c) *Perturbing each vertex of a graph* G ⊂ R
n within its ε*-neighborhood changes* NCD(G; h) up to 2ε *in both metrics* NBM and EMD *for any order* 1 ≤ h ≤ n.

(d) For any graphs *G, F* ⊂ R
n on m unordered vertices, the metrics NBM and EMD *between the invariants* NCD(G; h)
and NCD(F; h) from Definition B.6 *can be computed in time* O(m2h+1.5logh+1 m) with space O(n 2m2h+1 logh−1m)
for any order 1 ≤ h ≤ n.

The *affine dimension* 0 ≤ aff(A) ≤ n of a cloud A = {p1, . . . , pm} ⊂ R
n is the maximum dimension of the vector space generated by all inter-point vectors pi − pj , i, j ∈ {1*, . . . , m*}. Then aff(A) is an isometry invariant and is independent of an order of points of A. Any cloud A of 2 distinct points has aff(A) = 1. Any cloud A of 3 points that are not in the same straight line has aff(A) = 2.

Lemma D.2 provides a simple criterion for a matrix to be realizable by squared distances of a point cloud in R
n.

990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 Lemma D.2 (realization of distances). (a) A symmetric m × m matrix of sij ≥ 0 *with* sii = 0 is realizable as a matrix of squared distances between points p0 = 0, p1*, . . . , p*m−1 ∈ R
n if and only if the (m − 1) × (m − 1) *matrix* gij =
s0i + s0j − sij 2*has only non-negative eigenvalues.*
(b) *If the condition in (a) holds,* aff(0, p1, . . . , pm−1) equals the number k ≤ m − 1 ≤ n *of positive eigenvalues. Also in* this case, gij = pi· pj *define the* Gram matrix GM of the vectors p1*, . . . , p*m−1 ∈ R
n*, which are uniquely determined in* time O(m3) *up to an orthogonal map in* R
n.

Proof of Lemma *D.2.* (a) We extend Theorem 1 from (Dekster & Wilker, 1987) to the case *m < n* + 1 and justify the reconstruction of p1*, . . . , p*m−1 in time O(m3) uniquely in R
n up to an orthogonal map from O(n).

The part *only if* ⇒. Let a symmetric matrix S consist of squared distances between points p0 = 0, p1*, . . . , p*m−1 ∈ R
n. For i, j = 1*, . . . , m* − 1, the matrix with the elements

$$g_{i j}={\frac{s_{0i}+s_{0j}-s_{i j}}{2}}={\frac{p_{i}^{2}+p_{j}^{2}-|p_{i}-p_{j}|^{2}}{2}}=p_{i}\cdot p_{j}$$

is the Gram matrix, which can be written as GM = P
T P, where the columns of the n × (m − 1) matrix P are the vectors p1*, . . . , p*m−1 . For any vector v ∈ R
m−1, we have 0 ≤ |P v| 2 = (P v)
T(P v) = v T(P
T P)v = v T GMv.

Since the quadratic form v T GMv ≥ 0 for any v ∈ R
m−1, the matrix GM is positive semi-definite meaning that GM has only non-negative eigenvalues, see Theorem 7.2.7 in (Horn & Johnson, 2012).

Proof of Lemma *C.4.* In Definition C.2, choose partial flows fij =
1 for i = j, otherwise fij = 0. Then EMD(*S, Q*) ≤
The part if ⇐. For any positive semi-definite matrix GM, there is an orthogonal matrix Q such that QT GMQ = D is the diagonal matrix, whose m − 1 diagonal elements are non-negative eigenvalues of GM. The diagonal matrix 
√D consists of the square roots of eigenvalues of GM.

(b) The number of positive eigenvalues of GM equals the dimension k = aff({0, p1*, . . . , p*m−1}) of the subspace in R
n linearly spanned by p1*, . . . , p*m−1. We may assume that all k ≤ n positive eigenvalues of GM correspond to the first k coordinates of R
n. Since QT = Q−1, the given matrix GM = QDQT = (Q
√D)(Q
√D)
T becomes the Gram matrix of the columns of Q
√D. These columns become the reconstructed vectors p1*, . . . , p*m−1 ∈ R
n.

If there is another diagonalization Q˜T GMQ˜ = D˜ for Q˜ ∈ O(n), then D˜ differs from D by a permutation of eigenvalues, which is realized by an orthogonal map, so we set D˜ = D. Then GM = QD˜ Q˜T = (Q˜
√D)(Q˜
√D)
Tis the Gram matrix of the columns of Q˜
√D.

The new columns differ from the previously reconstructed vectors p1*, . . . , p*m−1 ∈ R
n by the orthogonal map QQ˜T. Hence the reconstruction is unique up to O(n)-transformations. Computing eigenvectors p1*, . . . , p*m−1 requires a diagonalization of GM in time O(m3) (?)section 11.5]press2007numerical.

Though Lemma D.2 gives a two-sided criterion for realizability of distances by points p1*, . . . , p*m ∈ R
n, the space of distance matrices is highly singular and cannot be easily sampled. Even m = 4 points in R
2 have 6 distances that should satisfy a polynomial equation saying that the tetrahedron with these 6 edge lengths has volume 0. So a randomly sampled matrix of potential distances for *m > n* + 1 is unlikely to be realizable by a cloud of m ordered points in R
n.

Chapter 3 in (Liberti & Lavor, 2017) discusses realizations of a complete graph given by a distance matrix in R
n.

Lemma D.3(a) and later results hold for all clouds including degenerate ones, e.g. for 3 points in a straight line.

Any points p1*, . . . , p*n−1 ∈ A have aff(p1*, . . . , p*n−1) ≤ n − 2. For example, any two distinct points in A ⊂ R
3 generate a straight line. In R
2, any point p1 ̸= O(A) forms a suitable {p1}. In R
3, one can choose any distinct points p1, p2 ∈ A so that the infinite straight line via p1, p2 avoids O(A).

If there are no such p1, p2, then A ⊂ R
3is contained in a straight line L, so aff(A) = 1. In this degenerate case, the stronger condition aff(O(A) ∪ {p1*, . . . , p*n−1}) = aff(A) will help reconstruct A ⊂ L by using any point p1 ̸= O(A). The first step is to reconstruct any ordered sequence from its distance matrix in Lemma D.3(a). Lemma D.3(a) holds for all degenerate clouds, e.g. for three points are in a straight line.

Lemma D.3 (reconstruction of ordered points). (a) Any sequence of ordered points A = (p1*, . . . , p*m) in R
n *can be* reconstructed (uniquely up to isometry) from the matrix of the Euclidean distances |pi − pj | in time O(m3). If all distances are divided by R = max i=1,...,m |pi|*, the reconstruction of* A ⊂ R
n *is unique up to isometry and uniform scaling.*
(b) If m ≤ n*, the uniqueness of reconstructions in part (a) holds if we replace isometry with rigid motion. Hence any* n − 1 ordered points p1, . . . , pn−1 can be uniquely reconstructed from all pairwise distances between 0, p1, . . . , pn−1 *up to* SO(n)
rotation around the origin 0 ∈ R
n.

Proof of Lemma *D.3.* (a) By translation, we can put p1 at the origin 0 ∈ R
n. Let GM be the (m − 1) × (m − 1) matrix gij =
p 2 i + p 2j − |pi − pj | 2 2= pi· pj constructed from squared distances between p1 = 0*, . . . , p*m for *i, j* = 2*, . . . , m*. By Lemma D.2(b) if GM has k ≤ n positive eigenvalues, then p1 = 0*, . . . , p*m can be uniquely determined up to isometry in R 
k ⊂ R
n in time O(m3). If all distances are divided by the same radius R, the above construction guarantees uniqueness up to isometry and uniform scaling.

(b) If m ≤ n, any mirror image of A ⊂ R
n after a suitable rigid motion in R
n can be assumed to belong to an
(n − 1)-dimensional hyperspace H ⊂ R
n, where they are matched by a mirror reflection H → H with respect to an
(n − 2)-dimensional subspace S ⊂ H. This reflection is realized by the SO(n) rotation through 180◦around S.

Lemma D.3(b) for m = n = 3 implies that any triangle is determined by its sides up to rigid motion in R
3. For example, the sides 3, 4, 5 define a right-angled triangle whose mirror images are not related by rigid motion inside a plane H ⊂ R
3, but are matched by composing a suitable rigid motion in H and a 180◦rotation of R
3around a line in H.

1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079 1080 1081 1082 1083 1084 1085 1086 1087 1088 1089 1090 1091 1092 1093 1094 1095 1096 1097 1098 1099