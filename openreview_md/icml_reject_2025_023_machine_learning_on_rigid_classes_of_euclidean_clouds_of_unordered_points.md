# Machine Learning On Rigid Classes Of Euclidean Clouds Of Unordered Points

## Abstract

Most real objects allow infinitely many different representations. Robust machine learning aims to use only invariant features independent of object representations to guarantee that any output (class label or predicted property) is preserved if the same object is represented differently. For Euclidean clouds of unordered points under rigid motion, we introduce complete invariants (with no false negatives, no false positives) and a Lipschitz continuous distance that satisfies all metric axioms and is computable in polynomial time of the number of points. The new realizability property implies that the space of all rigid clouds is efficiently parametrized by vectorial invariants like geographic coordinates. The proposed invariants distinguished all rigid classes of atomic clouds in the world's largest collections of molecules with 3D coordinates and predicted chemical elements by pure geometry with over 98% accuracy.

## 1. Importance Of Complete And Bi-Continuous Invariants For Ml On Data With Real Values

000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054 This paper formalizes practically important conditions for application-driven ML on real objects with ambiguous representations and develops new canonical representations satisfying these conditions for any *clouds* (finite sets) of unordered points in Euclidean space R
n. Such a cloud is the most basic form of a real object from cars to molecules (Wang & Solomon, 2019), e.g. a set of corners or atoms. Many objects are *rigid* in the sense that their shape and properties are preserved under *rigid motion* composed of translations and rotations in R
n (Atz et al., 2021), which form the group SE(n). The slightly weaker relation is by isometries (distance-preserving transformations), which form the group E(n). The practical cases are dimensions n ≤ 3 and larger numbers m (hundreds) of unordered points without outliers (Shi et al., 2021) because atoms have stable nuclei.

. Correspondence to: Anonymous Author
<anon.email@domain.com>.

Any rigid cloud has infinitely many representations, e.g. lists of point coordinates, but the shape and properties of an object should be independent of a coordinate system. Points are usually unordered and even simple molecules have many indistinguishable atoms. Hence predictions should not depend on point ordering. On another hand, different rigid classes of chemically identical molecules can have different functional properties such as solubility and hence therapeutic effectiveness. If not all rigid classes are distinguished, drugs can become useless, implying human suffering and financial losses for manufacturers (Morissette et al., 2003). A repeated scan or measurement of the same object can produce a slightly different cloud that cannot be exactly matched with the original one by rigid motion, also due to atomic vibrations (Feynman, 1971). If noise is ignored up to any threshold ε > 0, sufficiently many tiny perturbations make all clouds equivalent by the transitivity axiom: if A ∼ B and B ∼ C, then A ∼ C (Brink et al., 1997). Since all small deviations between rigid classes of point clouds should be distinguished, all these classes live in a continuous space of rigid clouds, see Fig. 1 (left). This space was continuously parametrized only in dimension n = 1 or for m = 3 points or Fig. 1 (right) leaving other cases open.

Figure 1. **Left**: rigid classes of m unordered points in R
nform a continuous space, which had no complete and bi-continuous invariants for m > 3, n > 1. **Right**: the space of 3 points under isometry is parametrized by distances 0 < a ≤ b ≤ c ≤ a + b.

Machine learning previously focused on discrete classifications or success measures for finite datasets, which can be considered discrete samples (of measure 0) in continuous spaces. For generalizability to all real data outside finite datasets, application-driven ML needs new conditions formalized in Problem 1.1 below. (Li et al., 2021; Dym & Gortler, 2024; Maennel et al., 2024; Nigam et al., 2024) studied complete invariants without realizability and Lipschitz bi-continuity (Morris et al., 2024; Cahill et al., 2024).

1 Problem 1.1. *Find a complete and bi-continuous invariant* I : {*clouds of unordered points in* R
n} → a space X with a distance d *such that all the conditions below hold.* (a) Completeness: any clouds A, B of unordered points are related by a rigid motion of R
n if and only if I(A) = I(B).

(b) Metric axioms: 1) d(*α, β*) = 0 ⇔ α = β; 2) d(*α, β*) = d(β, α); 3) d(*α, β*)+d(β, γ) ≥ d(α, γ) for all *α, β, γ* ∈ X. (c) Lipschitz continuity: there is a constant λ such that if each point of a cloud A ⊂ R
n is perturbed up to Euclidean distance ε, then I(A) changes by at most λε *in the metric* d.

(d) Realizability: the image {I(A) | *clouds* A ⊂ R
n of unordered points} *is parametrized so that one can reconstruct* A *up to rigid motion from any* realizable *value of* I.

(e) Point matching: there is a constant µ *that guarantees* for any clouds A, B *a rigid motion matching all points of* A, B *up to Euclidean distance* µd(I(A), I(B)).

(f) Computability: for a fixed dimension n, the invariant I, the metric d, and all constructions in (d) and (e) are computable in polynomial time of the number of points. Clouds and rigid motion can be replaced with any data (graphs, meshes) and equivalences (also allowing reflections or uniform scaling), respectively, so Problem 1.1 makes sense for any real data with ambiguous representations. The completeness (or injectivity) in 1.1(a) fully answers the question "same or different?" A complete invariant I has the ultimate expressive power and always distinguishes all clouds A ̸∼= B (not only from a finite dataset) that cannot be matched by rigid motion, so I is a descriptor with no false negatives and *no false positives*. The universal approximation aims for the completeness of infinite-size invariants (Maron et al., 2019; Keriven & Peyre´, 2019; Yarotsky, 2022), so polynomial time in 1.1(f) makes all conditions harder. A complete invariant can give a discontinuous metric, say d(*A, B*) = 1 for all non-equivalent clouds without quantifying the similarity of near-duplicates. The continuity in 1.1(c) is necessary for smoothness and hence for any gradient-based optimisation Due to the first axiom in 1.1(b), any metric d detects rigidly equivalent clouds by checking if d(*A, B*) = 0. Without the first axiom, many more distances including the zero d ≡ 0 satisfy the other axioms and are called *pseudo-metrics* (Brecheteau ´ , 2019). If the third axiom in 1.1(b) fails with any additive error ε > 0, results of clustering may not be trustworthy (Rass et al., 2024).

The realizability in 1.1(d) implies that the invariant I is an invertible 1-1 map from the complicated *Cloud Rigid Space* CRS(R
n; m) of classes of clouds under rigid motion to the explicitly parametrized space I(CRS(R
n; m)) of realizable values. Then with 100% certainty, we can sample any value in I(CRS(R
n; m)) and reconstruct its cloud A ⊂ R
n.

The 1-1 point matching in 1.1(e) can be interpreted as the Lipschitz continuity of the inverse map I
−1so that any close values I(A), I(B) guarantee the closeness of *A, B* under rigid motion. Conditions 1.1(c,e) mean that I is bi- Lipschitz: ε/µ ≤ d(I(A), I(B)) ≤ λε, where ε is the minimum perturbation needed to match all points of *A, B*. A partial matching, e.g. ignoring outliers, is harder to formalize. Indeed, if any clouds sharing all points except one are called equivalent, the transitivity axiom allows us to build a chain of equivalences A1 *∼ · · · ∼* Ak changing one point at a time, which can make all clouds equivalent. One can define metrics satisfying 1.1(a,b,c) by minimizing or deviations of unordered points over infinitely many transformations but polynomial time in 1.1(f) makes Problem 1.1 notoriously hard, previously solved only for m = 3 points. Conditions 1.1(a,b,c,f) and 1.1(d,e,f) formalize the discriminative and *generative* goals, respectively. A full solution to Problem 1.1 will imply that the rigid classes of clouds can be efficiently visualized in the *moduli* space I(CRS(R
n; m))
replacing any latent space of non-invariants or incomplete
(or discontinuous or non-realizable) invariants. Geographically, I(CRS(R
n; m)) can be compared with Earth's map, where any location can be reconstructed with all properties (altitude, precipitation, images, ...) from the latitude and longitude coordinates in known (realizable) ranges. Contributions. Problem 1.1 formalizes the necessary conditions for any application-driven ML on real objects. The new invariant Nested Distributed Projection solves Problem 1.1 for all clouds of m unordered points in dimension n = 2. Any cloud A ⊂ R
n can be reconstructed from a small part of the invariant (a vector in R
n(m−(n+1)/2))
whose realizability in 1.1(d) is guaranteed by explicitly written inequalities. Hence coordinates of this vector can be chosen in known ranges like latitude and longitude on Earth maps. The appendices cover all dimensions n > 2. The Python/C++ code is in the supplementary materials.

## 2. Past Work On Continuous Metrics For Clouds

Ordered points. Kendall's shape theory (Kendall et al.,
2009) studies m ordered points p1*, . . . , p*m ∈ R
n under isometries from E(n). In this case, a complete invariant is the distance matrix (Schoenberg, 1935; Kruskal & Wish, 1978) or the Gram matrix of scalar products pi· pj , see chapter 2.9 in (Weyl, 1946), (Villar et al., 2021). A bruteforce extension to m unordered points requires m! matrices due to m! permutations, which is ruled out by 1.1(f). Point cloud registration for unordered points samples rotations (Lin et al., 1986; Yang et al., 2020) and uses scaleinvariant features (Lowe, 1999; 2004; Huang et al., 2006) to approximately match clouds. If approximately matched 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 clouds are called equivalent, sufficiently many gradual perturbations make all clouds equivalent due to the transitivity axiom. Hence all rigid classes should be distinguished by a distance d that becomes zero only on rigidly equivalent clouds. Trying to sort points along a fixed direction or in a clockwise order around their center of mass leads to discontinuities because distant points can have equal projections to a line or a circle. A basis (say, of principal directions) of a cloud (Spezialetti et al., 2019; Zhu et al., 2022; Kurlin, 2024) is similarly unstable under perturbations of points in cases of high symmetry, e.g. when eigenvalues become equal, which often happens for real molecules in our main application. Converting a cloud by using extra parameters into a more complex object such as a continuous field R
3 → R
(Chauvin et al., 2022) or the persistent homology transform leads to the harder analog of Problem 1.1 for continuous surfaces instead of discrete clouds (Turner et al., 2014).

Neural networks (Bronstein et al., 2021) can guarantee invariance or equivariance (Thomas et al., 2018; Kondor & Trivedi, 2018; Cohen et al., 2019; Fuchs et al., 2020; Deng et al., 2021). An *equivariant* descriptor E satisfies the weaker condition E(f(A)) = Tf (E(A)) for any rigid motion f of a cloud A, where Tf may not be the identity as required for invariants (Satorras et al., 2021; Chen et al., 2021; Aronsson, 2022; Assaad et al., 2023; Xu et al., 2022; Su et al., 2022). Any linear combination of points such as the center of mass is equivariant but cannot distinguish clouds under translation. Equivariants were used for predicting forces acting on atoms to move them to a more optimal configuration. These time-dependent clouds At can be studied directly by their invariant values I(At) without intermediate forces. So neural networks optimize millions of parameters, see Table 4 in (Goyal et al., 2021), to improve accuracies (Dong et al., 2018; Akhtar & Mian, 2018; Laidlaw & Feizi, 2019; Guo et al., 2019; Colbrook et al., 2022) but need re-training any for new data and will have better generalizability if their inputs are invariants satisfying the conditions of Problem 1.1 for all possible clouds in R
n.

General metrics between fixed clouds extend to their rigid classes by minimization over infinitely many rigid motions (Huttenlocher et al., 1993; Chew & Kedem, 1992; Chew et al., 1999). In R
2, the time O(m5log m) (Chew et al., 1997) for the Hausdorff distance (Hausdorff, 1919)
will be improved in Theorem 5.3 to O(m3.5log m) for a new metric, see approximations in (Goodrich et al., 1999). The Gromov-Hausdorff and Gromov-Wasserstein metrics (Memoli ´ , 2011) are defined for metric-measure spaces also by minimizing over infinitely many correspondences between points, but cannot be approximated with a factor less than 3 in polynomial time unless P=NP, see Corollary 3.8 in (Schmiedl, 2017) and polynomial algorithms for partial cases in (Majhi et al., 2024). Also, computing a metric between rigid classes of clouds is only a small part of Problem 1.1. Indeed, to efficiently navigate on a real planet, in addition to distances between cities, we need a satellite-type view of the whole planet and hence a realizable bi-continuous invariant I, which can be considered an analog of the latitude and longitude coordinates on Earth. Can we 'sense' a shape? Problem 1.1 asks the questions
'same or different clouds, and how much different?' The related problem 'Can we hear the shape of a drum?' (Kac, 1966) has the negative answer in terms of 2D polygons indistinguishable by spectral invariants (Gordon et al., 1992a;b; Reuter et al., 2006; Cosmo et al., 2019; Marin et al., 2021). Problem 1.1 looks for stronger invariants that can completely 'sense' (not only 'hear') all rigid clouds in any R
n.

The partial cases when Problem 1.1 was solved are only n = 1 or m ≤ 3. In dimension n = 1, any rigid motion of R
 is a translation, so the Cloud Rigid Space CRS(R; m)
of m points p1*, . . . , p*m ∈ R is the space R
m−1
+ of sequential inter-point distances di = pi+1 − pi > 0 for i = 1*, . . . , m* − 1. Including reflections, the *Cloud Isometry* Space CIS(R; m) is the quotient of R
m−1
+ under the cyclic equivalence (d1, . . . , dm−1) ∼ (dm−1*, . . . , d*1). For clouds of m = 2 points in any dimension n ≥ 1, CRS(R
n; 2) is parametrized by a single inter-point distance d > 0. The final known case is m = 3 due to the SSS theorem saying that any triangles are congruent (isometric) if and only if they have the same side lengths. The space CIS(R
n; 3)
of 3-point clouds has the geographic-style parametrization {0 < a ≤ b ≤ c ≤ a + b} by inter-point distances *a, b, c* so that any (*a, b, c*) ∈ CIS(R
n; 3) generates a uniquely triangle under isometry. Problem 1.1 asks for a similarly explicit parametrization of CRS(R
n; m) for all m ≥ 4 and n ≥ 2.

Recent advances are the extensions (Delle Rose et al., 2024; Hordan et al., 2024) of the WL test (Leman & Weisfeiler, 1968), giving a binary answer (Brass & Knauer, 2000; 2004) by distinguishing all non-isometric clouds but without Lipschitz continuous metrics for all clouds including degenerate ones. Attempting to extend the SSS theorem, the Sorted Distance Vector (SDV) of all m(m−1)
2distances between m ≥ 4 unordered points distinguishes all non-isometric clouds in general position in R
n (Boutin & Kemper, 2004)
but not infinitely many 4-point clouds in R
2, see Fig. 2.

The SDV was strengthened (Widdowson & Kurlin, 2022) to the Pointwise Distance Distribution (PDD), which still cannot distinguish infinitely many non-isometric clouds in R 
3, see Fig. S4 in (Pozdnyakov & Ceriotti, 2022). All these counter-examples were distinguished by the Simplexwise Centered Distributions from (Widdowson & Kurlin, 2023), which satisfy 1.1(a,b,c,f) but not 1.1(d,e). Distance-based invariants do not allow easy realizability already for m = 4 points in R
2 whose 6 inter-point distances should satisfy a non-trivial polynomial equation saying that the tetrahedron on 4 points has volume 0 in R
2. Hence random distances between m > 3 unordered points are realized by a point cloud in R
2 with probability 0 (Duxbury et al., 2016).

## 3. Complete Invariants Of Unordered Clouds

While past representations used one basis (say, of principal directions of a given cloud A ⊂ R
n), this section introduces a new representation based on variable projections that depend on n − 1 ordered points in C consisting of m unordered points. For simplicity, we consider n = 2 when we have only m choices for a single point p ∈ A in Fig. 3.

165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219
For any cloud A ⊂ R
2 of m unordered points, the *center* of mass is O(A) = 1m P
p∈A
p. Shift A so that O(A) is the origin 0 ∈ R
2. For any p = (x1, x2) ∈ A, the vector 4 p
⊥ = (−x2, x1) is orthogonal to p*, so* p · p
⊥ = 0, which holds even if p = 0. If p is not at the origin (center of mass of A), we use the orthogonal basis *p, p*⊥ to represent all other points of A. Definition 3.1 makes sense for p = 0.

Definition 3.1 (point-based representation PR(A; p)). Let A ⊂ R
2 *be a cloud with the center of mass at the origin* 0.

Fix a base *point* p = (x, y) ∈ A*, set* p
⊥ = (−y, x)*. For any* q ∈ A \ {p}, the 2×(m −1) matrix M(A; p) *has a column* of the scalar products q · *p, q* · p
⊥*. The* point-based representation of A *is the pair* PR(A; p) = -|p| 2, M(A; p).

We use |p| 2and scalar products to make all components polynomial (smooth) in coordinates. The matrix M(A; p)
has two rows (ordered according to *p, p*⊥) and m − 1 unordered columns, and can be considered a *fixed cloud* of m − 1 unordered points in R
2, not under rigid.

Example 3.2 (regular polygons in R
2). (a) For m ≥ 2, let Am = {R exp 2πi√−1 m} ⊂ R
2, i = 1, . . . , m*, be the* vertex set of a regular m-sided polygon. Then Am has the center of mass O(Am) = (0, 0) at the origin and is inscribed in the circle of the radius R = R(Am). In Definition *3.1, choose the point* p = (R, 0) ∈ Am, which doesn't affect PR(Am; p) *due to the rotational symmetry* of Am. Then the matrix M(Am; p) *consists of* m − 1 columns R2cos(2πi/m)
R2sin(2*πi/m*)
, i = 1, . . . , m − 1*. The* m−1

_pair is_ PR($A_{m}$;$p$) = [R${}^{2}$, ( $\left(\right)$
hR2,R2cos 
2πi
m
R2sin 2πi
m
$\bigstar\bigstar|$ . 
$$\bigstar\bigstar_{i=1}$$
i.
(b) *Let the cloud* Bm ⊂ R
2 be Am after adding the extra point at the origin 0 ∈ R
2. For any point p ∈ Am*, the new* point-based representation PR(Bm; p) *is obtained from* PR(Am; p) *above by adding the zero column to the matrix* M(Am; p). For the extra point at the origin 0, the representation is PR(Bm; 0) = [0, M(Bm; 0)]*, where* M(Bm; 0) is the 2 × m *matrix consisting of zeros.* Theorem 3.3 (realizability of abstract PR). Let s > 0 and M be any 2 × (m − 1) matrix for m ≥ 2. The pair [*s, M*] is realizable as a point-based representation PR(A; p) for a cloud A ⊂ R
n of m *unordered points with* O(A) = 0 and a point p ∈ A *if and only if* s +
m P−1 j=1 M1j = 0 =
m P−1 j=1 M2j .

In Theorem 3.3, s = |p| 2is the squared distance from a point p ∈ A to 0 ∈ R
2. The equations say that the sums of the scalar products (q · p) and (q · p
⊥) for all q ∈ A equal to 0, which is equivalent to Pq ∈ A = 0 meaning that the center of mass O(A) is 0. Hence s > 0 and m − 2 columns of M can be considered free parameters. Definition 3.4 combines point-based representations PR(A; p) for all points p ∈ A into one invariant NDP
(Nested Distributed Projection) that will be proved to satisfy Any point p = (x1*, . . . , x*n) ∈ R
n has *Euclidean* norm |p| =
sPn i=1 x 2 i. Any points p and q = (y1*, . . . , y*n) ∈ R
n are also interpreted as vectors, have the *Euclidean* distance |p − q| and the *scalar* (dot) product of p · q =Pn i=1 xiyi. Any vectors p ⊥ q are *orthogonal* if and only if p · q = 0.

all conditions of Problem 1.1. The major advantage of NDP
is its applicability to all real clouds A ⊂ R
2 without any requirement of general position. Some points of a cloud A may coincide, so A can be a multiset of points.

Definition 3.4 (invariants NDP and NCP). Let A ⊂ R
2 be any cloud of m *unordered points. The* Nested Distributed Projection NDP(A) *is the unordered set of* PR(A; p) for all p ∈ A. If k > 1 *representations* PR(A; p) are equal then we collapse them to one representation with the weight k/m. The resulting set of unordered PRs with weights is called the Nested Compressed Projection NCP(A).

Table 1. Acronyms and references of all key concepts in the paper.

220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 PR POINT-BASED REPRESENTATION DEF 3.1 NDP NESTED DISTRIBUTED PROJECTION DEF 3.4 PRM POINT-BASED REPRESENT. METRIC DEF 4.2 BMD BOTTLENECK MATCHING DISTANCE DEF 4.3 NBM NESTED BOTTLENECK METRIC DEF 4.4 For the cloud Am from Example 3.2, the Nested Distributed Projection NDP(Am) consists of m identical representations, so NCP(Am) is the single representation PR(Am; p)
with weight 1. The invariant NDP is an expanded version of the NCP, where all PRs have equal weights 1/m. The full invariant NDP(A) includes the faster (linear-time) vector of squared distances |p| 2from the center of mass O(A) = 0 ∈ R
2to all points p ∈ A*. If* A has a distinguished point p, e.g. a special atom in a molecule, the point-based representation PR(A; p) is invariant.

Theorem 3.5 (completeness of NDP). The Nested Distributed Projection is complete in the sense that any clouds A, B ⊂ R
2 of m unordered points are related by rigid motion in R
2*if and only if* NDP(A) = NDP(B) so that there is a bijection NDP(A) → NDP(B) *matching all* PRs. Under a mirror reflection, for any p ∈ A, one can assume after applying rigid motion that the basis *p, p*⊥ maps to its mirror image p, −p
⊥. The mirror image A¯ has NDP(A¯)
equal to NDP(A) that is obtained from NDP(A) by reversing all signs in the last row of M(A; p) for each p ∈ A. The completeness of NDP(A) Theorem 3.5 implies the completeness of the pair NDP(A), NDP(A) under isometry including reflections. Further work can simplify this pair to a smaller invariant while keeping the completeness. Since a bijection NDP(A) → NDP(B) between all (uncollapsed) PRs induces a bijection NCP(A) → NCP(B) respecting all weights of collapsed PRs, Theorem 3.5 implies the completeness of NCP under rigid motion in R
2.

## 4. A Metric On Complete Invariants Of Clouds

This section will define the metric NBM on invariants NDP by using the bottleneck distance BD in Definition 4.1, a metric on point-based representations (PRs) in Definition 4.2, and a bottleneck matching distance in Definition 4.3.

Definition 4.1 (bottleneck distance BD). *For any* v =
(v1*, . . . , v*n) ∈ R
n*, the* Minkowski *norm is* ||v||∞ =
max i=1*,...,n* |vi|. For clouds *A, B* ⊂ R
n of m unordered points, the bottleneck distance BD(*A, B*) = inf g:A→B
sup p∈A
||p −
g(p)||∞ *is minimized over all bijections* g : A → B. Though the bottleneck distance is defined as a minimum for m! bijections A → B between m-point clouds, Theorem 6.5 in (Efrat et al., 2001) computes BD(*A, B*) in time O(m1.5log2m) by filtering out distant points. The bruteforce extension of BD(*A, B*) under rigid motion need a minimization for infinitely many rotations. NDP(A) consists of only m point-based representations PR(A; p) = [|p| 2, M(A; p)], one for each p ∈ A. The BD algorithm can compare any 2 × (m − 1) matrices M(A; p) and M(B; q)
as fixed clouds of unordered columns (points in R
2).

In Definition 4.2, the notation M/R means that all elements of the matrix M(A; p) are divided by the *radius* R(A) = max p∈A
|p| of a cloud A. Then PRM and further metrics have units of original points, e.g. in meters. One more division by R(A) makes metrics invariant under uniform scaling. Definition 4.2 (Point-Based Representation Metric PRM). Let PR(A; p),PR(B; q) *be point-based representations of* clouds *A, B* ⊂ R
2 of m *unordered points for base points* p ∈ A and q ∈ B, respectively, see Definition *3.1. The* Point-based Representation Metric between the PRs above is PRM = max{ | |p|−|q| |, |R(A)−R(B)|, wM }, *where* wM = BD 
M(A; p)
R(A),
M(B; q)
R(B)
, see Definition *4.1.*
We defined PRM as the maximum of 3 metrics to guarantee the metric axiom (if PRM = 0 then A ∼= B) and the simplest Lipschitz constant λ = 2 in 1.1(d), see all proofs in appendix D. Replacing the maximum with (say) a sum gives a metric with a higher constant λ depending on m. Definition 4.3 (bottleneck matching distance BMD(Γ)). Let Γ be a complete bipartite graph with m *white vertices* and m black vertices so that every white vertex is connected to every black vertex by an edge e of a weight w(e) ≥ 0. A vertex matching in Γ is a set E of m disjoint edges of Γ*. The* weight W(E) = max e∈E
w(e) is the largest weight in E*. The* bottleneck matching distance *of the graph* Γ is BMD(Γ) = min E
W(E) *is minimized over all vertex matchings.*
Because Γ is bipartite, any edge from a vertex matching E joins a white vertex with a black vertex. Then BMD(Γ) is minimized for all bijections E between all white vertices and all black vertices of Γ similar to Definition 4.1. Definition 4.4 builds a graph Γ(*A, B*) on all point-based representations of *A, B* ⊂ R
n and introduces the Nested Bottleneck Metric NBM(*A, B*) as BMD of Γ(A, B).

275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 Definition 4.4 (NBM : Nested Bottleneck Metric). Let clouds *A, B* ⊂ R
2consist of m unordered points. The complete bipartite graph Γ(A, B) has m white vertices (one for each p ∈ A) and m *black vertices (one for each* q ∈ B). Any edge e of Γ(A, B) has endpoints associated with pointbased representations PR(A; p), PR(B; q)*, and the* weight w(e) = PRMPR(A; p), PR(B; q)*. The* Nested Bottleneck Metric *is defined as* NBM(*A, B*) = BMD(Γ(A, B)). Example 4.5 (4-point clouds C
±). In R
2*, consider the 4-*
point clouds C
± = {p1, p2, p3, p±
4}*, where* p1 = (4a, 0),
p2 = (*b, c*), p3 = −p2 = (−b, −c), p
+
4 = (0, 4d)*, and* p
−
4 = (0, −4d) for parameters a, b, c, d ≥ 0*, see Fig.* 2.

Appendix C *will explicitly compute* NDP(C
±)*to distinguish* all clouds C
+ ̸∼= C
−. Fig. 4 *shows the new metric* NBM for variable parameters a, b and fixed *c, d*. NBM > 0 *implies* that C
+ ̸∼= C
−*, except in the singular cases below. If* a = 0 or d = 0 or b = c = 0, the clouds are related by a 2-fold rotation around the origin 0*. If* a =
√3 2 ≈ 0.87, b = 0, c = 2, d = 0.5*, then* C
+ *consists of the vertices*
(0, ±2),(2√3, 0) of an equilateral triangle, where (0, 2) is the double point p2 = p
+ 4
. Then C
− *is the same equilateral* triangle but its vertex (0, −2) *is the double point* p3 = p
−
4.

Because these clouds are related by rotation, NBM = 0 in the black pixel at a =
√3 2 ≈ 0.87, b = 0 *in Fig.* 4.

For a *fixed dimension* n, all algorithms for m unordered points will have polynomial times in m in the RAM model.

Theorem 5.1 (Lipschitz continuity of NBM). Let B ⊂ R
2 be obtained from a cloud A ⊂ R
2 by perturbing every point of A up to Euclidean distance ε*. Then* NBM(*A, B*) ≤ 6ε. To illustrate Theorem 5.1, we generated uniformly random

## 5. Bi-Continuity And Polynomial Algorithms

clouds A in the unit square and cube. To get a perturbation B of A, we shifted every point of A by adding a uniformly random value in [−*ε, ε*] to each coordinate, where ε ∈ [0.01, 0.1] is a noise bound. Fig. 5 shows how the Nested Bottleneck Metric (NBM, averaged over several clouds) linearly increases with respect to the noise bound.

Figure 5. The metric NBM(NDP(A), NDP(B)) for a random cloud A and its ε-perturbation B increases at most linearly in the noise bound ε with a Lipschitz constant λ2 < 6 as in Theorem 5.1.

Theorem 5.2 (NDP time). *For any cloud* A ⊂ R
2 of m unordered points, the Nested Distributed Projection NDP(A)
is computed in time O(m2) *with space* O(m2). Theorem 5.3 (NBM time). For any clouds *A, B* ⊂ R
2 of m unordered points, the Nested Bottleneck Metric NBM(A, B)
is computable in time O(m3.5log m) *with space* O(m3). Fig. 6 illustrates a polynomial dependence of the NBM time in Theorem 5.3. Theorem 5.4 says that any m-point clouds A, B ⊂ R
2can be matched up to a perturbation proportional to the Nested Bottleneck Metric d = NBM. If d is small, all points of *A, B* can be matched up to a perturbation 3
√2d by rigid motion. In section 6, the experimental maximum of this approximate factor is 2.2 < 3
√2.

Theorem 5.4 (point matching). For any m*-point clouds* A, B ⊂ R
2*, one can find in time* O(m3.5log m) a rigid motion f of R
2 and a bijection β : A → B *such that the* match distance max q∈A
|f(q)−β(q)| ≤ 3
√2NBM(A, B), see the comparison of this distance with others in Fig. 5. By Theorem 5.1, perturbing every atom up to ε (due to the ever-present thermal vibrations) changes NDP up to 6ε in the metric NBM. Conversely, by Theorem 5.4, if NBM(*A, B*) = δ > 0 is small, the clouds *A, B* can be approximately matched by rigid motion up to 3
√2δ pointwise.

If clouds *A, B* ⊂ R
n have ordered points, one can *morph*
(continuously transform) A to B by moving every i-th point of A along a straight-line to the i-th point of B for i = 1*, . . . , m*. If m points are unordered, there are m! potential transformations, one for each permutation of m points.

Associating every point p ∈ A to its nearest neighbor q ∈ B
is justified only for fixed clouds because a rigid motion of A can change a nearest neighbor of any point p ∈ A in B.

## 6. Experiments On Large Molecular Databases

The big databases of molecules with *3D conformers* (embeddings in R
3) are QM9 (130K+ entries) (Ramakrishnan et al., 2014) and GD (GEOM drugs, 31M+ entries) containing hundreds of 3D conformers of *unordered* atoms for each of 61607 chemical compositions (Axelrod & Gomez- Bombarelli, 2022). The Protein Data Bank has backbones of *ordered* atoms classified by simpler invariants (Anosova et al., 2025). All experiments took a few hours on Ryzen 9 3950X 3.5 GHz, 64 MB of L3 cache, RAM 82GB. The ICML guide for application-driven ML says that "novel ideas that are simple to apply may be especially valuable", so we start with simpler and much faster invariants below.

Definition 6.1 (invariants SRV, SDV,PDD). Let A ⊂ R
n be a cloud of m unordered points with the center of mass at 0 ∈ R
n*. The* Sorted Radial Vector SRV(A) has m radial distances |p| in decreasing order for all p ∈ A*. The* Sorted Distance Vector SDV(A) *is the vector of* m(m−1)
2*pairwise* distances |p − q| in decreasing order for distinct p, q ∈ A. For any point p ∈ A, let d1(p) *≤ · · · ≤* dm−1(p) be Euclidean distances from p to all other points q ∈ A \ {p}
in increasing order. These distance lists become rows of the m × (m − 1) matrix D(S; k). Any l > 1 *identical rows are* collapsed into a single row with the weight l/m*. The final* matrix with at most m *unordered weighted rows and* m − 1 ordered columns is the Pointwise Distance Distribution.

330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 For a PDD on m points, we sort m distance lists in time O(m2log m). Then PDDs are compared by the Earth Mover's Distance EMD (Rubner et al., 2000) in time O(m3). Table 2 emphasizes that most clouds should be first distinguished by simpler and faster invariants SRV, SDV,PDD. The complete NDP is needed only in rare cases but is still essential because any incomplete invariant I has no chance to predict different properties on false positives that are molecules A ̸∼= B with I(A) = I(B).

| INVARIANT   | TIME        | METRIC   | TIME          |
|-------------|-------------|----------|---------------|
| SRV         | O(m log m)  | L∞       | O(m)          |
| SDV         | O(m2 )      | L∞       | O(m2 )        |
| PDD         | O(m2 log m) | EMD      | O(m3 )        |
| NDP         | O(m2 )      | NBM      | O(m3.5 log m) |

For a fixed atom p ∈ A and *k < m*, the first k distances to neighbors in the row of p in PDD(A) is an atomwise version of SRV(A). This vector D(*A, p*; k) of k distances was the only input for predicting the chemical element of p. A default network in TensorFlow was trained on clouds with the 80/20 split and achieved 98% accuracy for k = 4 in Table 4 despite the unbalanced counts of frequent elements in Table 3. Appendix A has all implementation details.

QM9: H QM9: C QM9: N QM9: O QM9: F 1,230,122 846,557 139,764 187,996 3,314 GD0: H GD0: C GD0: N GD0: O GD0: F

5,660,986 5,267,096 842,562 854,400 64,299

GD0: P GD0: S GD0: Cl GD0: Br GD0: I 1,350 159,648 53,404 14,010 225

Table 4. Accuracies in percentages for predicting chemical elements by a 4-layer network using *only Euclidean distances* from an atomic center to its k nearest neighbors for QM9 and GD0.

All past attempts by both ML and non-ML in chemistry achieved only 86% on similar size data, see Table 7 summarized in (Vasylenko et al., 2025), because the underlying descriptors were not invariant, e.g. under permutations of atoms, which creates exponentially many representations of the same molecule, incomplete, or their similarities failed the triangle axiom, e.g. see (Steck et al., 2024).

| data   | k = 2   | k = 3   | k = 4   | k = 5   | k = 6   |
|--------|---------|---------|---------|---------|---------|
| QM9    | 94.63   | 98.64   | 98.24   | 98.54   | 98.77   |
| GD0    | 91.44   | 96.67   | 98.05   | 98.70   | 98.49   |

High accuracies of D(*A, p*; 4) in Table 4 are explained by the following cascade computations. First, split all clouds from Table 3 by the 1st distance (to the nearest neighbor of a central atom p) rounded to 3 decimal places in A˚ . This is a typical experimental precision, where 1A˚ = 10−10m is the smallest interatomic distance. Second, split each subset with equal 1st distances by 2nd distances, and so on up to k = 5 distances. All clouds of different elements in QM9 and GD0 were separated by D(*A, p*; 4) and D(*A, p*; 5), respectively. We compared full molecules starting with the pseudo-metric L∞ (max abs difference of corresponding coordinates) on SRVs of all 873,527,974 pairs of 3D atomic clouds having equal numbers of atoms in QM9, then 8,735,279 distances L∞ on SDVs of the 1% closest pairs, 87,352 EMDs on PDDs of the 1% closest pairs, and NBMs on NDPs for the final 10K closest pairs. In this hierarchical computation, large values of L∞ (then EMD) guarantee that molecules are distant and cannot be closely matched by rigid motion.

Tiny or zero values of pseudo-metrics guarantee nothing because SDV and PDD can coincide for very different clouds, see Fig. 2, Fig. S4 in (Pozdnyakov et al., 2020).

385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 Table 5. Chemically different molecules (given by QM9 ids) are geometrically distinguished by SRV, SDV*,PDD,* NDP, see Fig. 8.

smallest distances in A, molecule A ˚ ̸= molecule B
SRV, L∞ = 0.021, H4C5N2O(5365)̸=H3C4N3O2(131923) SDV, L∞ = 0.055, H3C4N5(123533)̸=H3C5N3O(24547) EMD = 0.051, H3C4N5(123533)̸=H3C5N3O(24521) NBM = 0.148, H3C4N3O2(28141)̸=H3C3N5O(130099)
Fig. 7 compares the new metric y = NBM on complete NDPs with the pseudo-metric x = PDD. All pairs *A, B*
with (*x, y*) close to the vertical axis in Fig. 7 (left) have EMD ≈ 0 because they are almost mirror images (indistinguishable by PDD) well distinguished by higher values of NBM. Fig. 8 shows bonds by standard visualization, they were not used for clouds of points without any edges.

For each of 31M+ entries (3D conformers) in the much larger database GD, we took the cloud A of all atoms without chemical elements and computed SRV(A; k) of up to k = 10 largest distances (rounded to 3 decimal places) from the center of mass of A to all atoms. Similar to QM9, cascade comparisons confirmed that SRV(A; 7) distinguishes all chemically different molecules, while only four pairs have equal SRV(A; 6) rounded to 3 decimal places. This transparent reconstruction of a full chemical composition from precise enough geometry gives hope to explain other molecular properties in terms of geometric invariants.

Figure 8. **Left**: chemically different QM9 molecules 28141 and 130099 have the smallest distances NBM ≈ 0.15A˚ . **Right**:
molecules 70954 and 74130 are almost mirror images with EMD ≈ 0.0004A˚ but are well distinguished by NBM ≈ 1.619A˚ .

## 7. Discussion: Conclusions And Limitations

For clouds with different numbers of points, we can replace the bottleneck distance BD in Definition 4.2 with any metric between fixed clouds of different sizes, e.g. the Hausdorff distance, to get a metric on PRs. Then we can compare NDPs of any clouds as weighted distributions by EMD. The limitation is the proof of Theorem 5.4 in dimension n = 2, though the experiments indicate the Lipschitz continuity of NDP−1in R
3. All other conditions in Problem 1.1 are proved in the appendices for any dimension n ≥ 2. The experiments imply that mapping any molecule to (the rigid class of) its cloud of atomic centers is *injective* without losing any chemical information, so all chemical elements can be reconstructed from pure geometry. This result confirms our physical intuition that replacing atoms should perturb geometry at least slightly, which was impossible to establish without complete and Lipschitz continuous invariants. Hence all molecules of m atoms live at different locations in the common *Cloud Rigid Space* CRS(R
3; m)
of SE(3)-classes of all clouds of m unordered points. Most significantly, a *molecular structure* can now be defined not as a huge collection of vectors under rotations and atom permutations, see Fig. 1 in (Lang et al., 2024), but as a rigid (class of a) cloud of atomic centers (without chemical elements), which is uniquely determined by an efficient hierarchy of invariants from the fastest (linear-time) SRV to the new complete invariant NDP solving Problem 1.1.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

Akhtar, N. and Mian, A. Threat of adversarial attacks on deep learning in computer vision: A survey. *IEEE Access*, 6:14410–14430, 2018.

Anosova, O., Gorelov, A., Jeffcott, W., Jiang, Z., and Kurlin, V. A complete and bi-continuous invariant of protein backbones under rigid motion. *MATCH Communications* in Mathematical and in Computer Chemistry (to appear), arxiv:2410.08203, 2025.

Antunes, L. M., Grau-Crespo, R., and Butler, K. T. Distributed representations of atoms and materials for machine learning. *npj Computational Materials*, 8(1):44, 2022.

Aronsson, J. Homogeneous vector bundles and gequivariant convolutional neural networks. Sampling Theory, Signal Processing, and Data Analysis, 20(2):10, 2022.

Axelrod, S. and Gomez-Bombarelli, R. Geom, energyannotated molecular conformations for property prediction and molecular generation. *Scientific Data*, 9(1):185, 2022.

Brass, P. and Knauer, C. Testing congruence and symmetry for general 3-dimensional objects. Computational Geometry, 27(1):3–11, 2004.

Brecheteau, C. A statistical test of isomorphism between ´
metric-measure spaces using the distance-to-a-measure signature. pp. 795–849, 2019.

Brink, C., Kahl, W., and Schmidt, G. Relational methods in computer science. Springer Science & Business Media, 1997.

Bronstein, M. M., Bruna, J., Cohen, T., and Velickovi ˇ c,´
P. Geometric deep learning: grids, groups, graphs, geodesics, and gauges. *arXiv:2104.13478*, 2021.

Cahill, J., Iverson, J. W., and Mixon, D. G. Towards a bilipschitz invariant theory. Applied and Computational Harmonic Analysis, 72:101669, 2024.

Chauvin, L., Wells III, W., and Toews, M. Registering image volumes using 3D SIFT and discrete SP-symmetry. arXiv:2205.15456, 2022.

Chen, C., Ye, W., Zuo, Y., Zheng, C., and Ong, S. P. Graph networks as a universal machine learning framework for molecules and crystals. *Chemistry of Materials*, 31(9): 3564–3572, 2019.

Chen, H., Liu, S., Chen, W., Li, H., and Hill, R. Equivariant point network for 3D point cloud analysis. In Computer Vision and Pattern Recognition, pp. 14514–14523, 2021.

Chew, P. and Kedem, K. Improvements on geometric pattern matching problems. In Scandinavian Workshop on Algorithm Theory, pp. 318–325, 1992.

Chew, P., Goodrich, M., Huttenlocher, D., Kedem, K., Kleinberg, J., and Kravets, D. Geometric pattern matching under Euclidean motion. *Computational Geometry*, 7
(1-2):113–124, 1997.

Chew, P., Dor, D., Efrat, A., and Kedem, K. Geometric pattern matching in d-dimensional space. *Discrete &* Computational Geometry, 21(2):257–274, 1999.

Cohen, T. S., Geiger, M., and Weiler, M. A general theory of equivariant cnns on homogeneous spaces. Advances in Neural Information Processing Systems, 32, 2019.

Colbrook, M. J., Antun, V., and Hansen, A. C. The difficulty of computing stable and accurate neural networks: On the barriers of deep learning and Smale's 18th problem. *Proc.* National Academy of Sciences, 119(12):e2107151119, 2022.

Cosmo, L., Panine, M., Rampini, A., Ovsjanikov, M., Bronstein, M. M., and Rodola, E. Isospectralization, or how to hear shape, style, and correspondence. In Proceedings of CVPR, pp. 7529–7538, 2019.

Dekster, B. V. and Wilker, J. B. Edge lengths guaranteed to form a simplex. *Archiv der Mathematik*, 49(4):351–366, 1987.

Boutin, M. and Kemper, G. On reconstructing n-point configurations from the distribution of distances or areas. *Adv.*
Appl. Math., 32(4):709–735, 2004.

Brass, P. and Knauer, C. Testing the congruence of ddimensional point sets. In *SoCG*, pp. 310–314, 2000.

Assaad, S., Downey, C., Al-Rfou, R., Nayakanti, N., and Sapp, B. Vn-transformer: Rotation-equivariant attention for vector neurons. Transactions on Machine Learning Research, 2023.

Atz, K., Grisoni, F., and Schneider, G. Geometric deep learning on molecular representations. Nature Machine Intelligence, 3(12):1023–1032, 2021.

440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 Delle Rose, V., Kozachinskiy, A., Rojas, C., Petrache, M.,
and Barcelo, P. Three iterations of (d- 1)-wl test dis- ´ tinguish non isometric clouds of d-dimensional points. Advances in Neural Information Processing Systems, 36, 2024.

Guo, C., Gardner, J., You, Y., Wilson, A. G., and Weinberger, K. Simple black-box adversarial attacks. In *International* Conference on Machine Learning, pp. 2484–2493, 2019.

495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 Hausdorff, F. Dimension und au¨ βeres maβ. Mathematische Annalen, 79(2):157–179, 1919.

Deng, C., Litany, O., Duan, Y., Poulenard, A., Tagliasacchi, A., and Guibas, L. J. Vector neurons: A general framework for so(3)-equivariant networks. In *Proceedings of* the International Conference on Computer Vision, pp. 12200–12209, 2021.

Hopcroft, J. E. and Karp, R. M. An nˆ5/2 algorithm for maximum matchings in bipartite graphs. *SIAM Journal* on Computing, 2(4):225–231, 1973.

Hordan, S., Amir, T., Gortler, S. J., and Dym, N. Complete neural networks for euclidean graphs. In AAAI Conference on Artificial Intelligence, volume 38 (11), pp.

12482–12490, 2024.

Deza, E. and Deza, M. M. *Encyclopedia of distances*.

Springer, 2009.

Dong, Y., Liao, F., Pang, T., Su, H., Zhu, J., Hu, X., and Li, J. Boosting adversarial attacks with momentum. In Computer vision and pattern recognition, pp. 9185–9193, 2018.

Horn, R. A. and Johnson, C. R. *Matrix analysis*. Cambridge University Press, 2012.

Huang, Q.-X., Flory, S., Gelfand, N., Hofer, M., and ¨
Pottmann, H. Reassembling fractured objects by geometric matching. In *ACM SIGGRAPH*, pp. 569–578. 2006.

Duxbury, P. M., Granlund, L., Gujarathi, S., Juhas, P., and Billinge, S. J. The unassigned distance geometry problem. Discrete Applied Mathematics, 204:117–132, 2016.

Huttenlocher, D. P., Klanderman, G. A., and Rucklidge, W. J.

Comparing images using the Hausdorff distance. Transactions on pattern analysis and machine intelligence, 15 (9):850–863, 1993.

Dym, N. and Gortler, S. J. Low-dimensional invariant embeddings for universal geometric learning. *Foundations* of Computational Mathematics, pp. 1–41, 2024.

Efrat, A., Itai, A., and Katz, M. J. Geometry helps in bottleneck matching and related problems. *Algorithmica*, 31(1):1–28, 2001.

Kac, M. Can one hear the shape of a drum? The american mathematical monthly, 73(4P2):1–23, 1966.

Kapovich, M. and Millson, J. J. The symplectic geometry of polygons in euclidean space. Journal of Differential Geometry, 44(3):479–513, 1996.

Feynman, R. The Feynman lectures on physics. Chapter 1:
atoms in motion, volume 1. 1971.

Fuchs, F., Worrall, D., Fischer, V., and Welling, M. Se(3)-
transformers: 3d roto-translation equivariant attention networks. Advances in neural information processing systems, 33:1970–1981, 2020.

Kendall, D. G., Barden, D., Carne, T. K., and Le, H. *Shape* and shape theory. John Wiley & Sons, 2009.

Keriven, N. and Peyre, G. Universal invariant and equivari- ´
ant graph neural networks. Advances in Neural Information Processing Systems, 32, 2019.

Goodrich, M. T., Mitchell, J. S., and Orletsky, M. W. Approximate geometric pattern matching under rigid motions. Transactions on Pattern Analysis and Machine Intelligence, 21(4):371–379, 1999.

Kondor, R. and Trivedi, S. On the generalization of equivariance and convolution in neural networks to the action of compact groups. In *International Conference on Machine* Learning, pp. 2747–2755, 2018.

Gordon, C., Webb, D., and Wolpert, S. Isospectral plane domains and surfaces via riemannian orbifolds. *Inventiones* mathematicae, 110(1):1–22, 1992a.

Kruskal, J. B. and Wish, M. *Multidimensional scaling*.

Number 11. Sage, 1978.

Gordon, C., Webb, D. L., and Wolpert, S. One cannot hear the shape of a drum. Bulletin of the American Mathematical Society, 27(1):134–138, 1992b.

Kurlin, V. Polynomial-time algorithms for continuous metrics on atomic clouds of unordered points. MATCH Communications in Mathematical and in Computer Chemistry, 91:79–108, 2024.

Goyal, A., Law, H., Liu, B., Newell, A., and Deng, J. Revisiting point cloud shape classification with a simple and effective baseline. In *International Conference on* Machine Learning, pp. 3809–3820, 2021.

Laidlaw, C. and Feizi, S. Functional adversarial attacks.

Adv. Neural Information Proc. Systems, 32, 2019.

Lang, L., Cezar, H. M., Adamowicz, L., and Pedersen, T. B.

Quantum definition of molecular structure. Journal of the American Chemical Society, 146(3):1760–1764, 2024.

Morissette, S. L., Soukasene, S., Levinson, D., Cima, M. J.,
and Almarsson, O. Elucidation of crystal form diversity ¨
of the hiv protease inhibitor ritonavir by high-throughput crystallization. Proceedings of the National Academy of Sciences, 100(5):2180–2184, 2003.

550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 Leman, A. and Weisfeiler, B. A reduction of a graph to a canonical form and an algebra arising during this reduction. *Nauchno-Technicheskaya Informatsiya*, 2(9):12–16, 1968.

Morris, C., Dym, N., Maron, H., Ceylan, ˙I. ˙I., Frasca, F.,
Levie, R., Lim, D., Bronstein, M., Grohe, M., and Jegelka, S. Future directions in foundations of graph machine learning. *arXiv:2402.02287*, 2024.

Li, X., Li, R., Chen, G., Fu, C.-W., Cohen-Or, D., and Heng, P.-A. A rotation-invariant framework for deep point cloud analysis. IEEE transactions on visualization and computer graphics, 28(12):4503–4514, 2021.

Nemec, L. Principal component analysis (pca):
A physically intuitive mathematical introduction. https://towardsdatascience.com/ principal-component-analysis-pca-8133b02f11bd, 2022.

Liberti, L. and Lavor, C. *Euclidean distance geometry*.

Springer, 2017.

Nigam, J., Pozdnyakov, S. N., Huguenin-Dumittan, K. K.,
and Ceriotti, M. Completeness of atomic structure representations. *APL Machine Learning*, 2(1), 2024.

Lin, Z. C., Lee, H., and Huang, T. S. Finding 3d point correspondences in transformation estimation. In Proceedings- International Conference on Pattern Recognition, pp. 303– 305. IEEE, 1986.

Oliynyk, A. O., Antono, E., Sparks, T. D., Ghadbeigi, L., Gaultois, M. W., Meredig, B., and Mar, A. Highthroughput machine-learning-driven synthesis of fullheusler compounds. *Chemistry of Materials*, 28(20): 7324–7331, 2016.

Lowe, D. G. Object recognition from local scale-invariant features. In *Proceedings of ICCV*, volume 2, pp. 1150– 1157, 1999.

Pozdnyakov, S. N. and Ceriotti, M. Incompleteness of graph convolutional neural networks for points clouds in three dimensions. *arXiv:2201.07136*, 2022.

Lowe, D. G. Distinctive image features from scale-invariant keypoints. *International journal of computer vision*, 60: 91–110, 2004.

Pozdnyakov, S. N., Willatt, M. J., Bartok, A. P., Ortner, C., ´
Csanyi, G., and Ceriotti, M. Incompleteness of atomic ´ structure representations. *Phys. Rev. Lett.*, 125:166001, 2020. URL arXiv:2001.11696.

Maennel, H., Unke, O. T., and Muller, K.-R. Complete and efficient covariants for three-dimensional point configurations with application to learning molecular quantum properties. *The Journal of Physical Chemistry Letters*,
15:12513–12519, 2024.

Ramakrishnan, R., Dral, P. O., Rupp, M., and Von Lilienfeld, O. A. Quantum chemistry structures and properties of 134 kilo molecules. *Scientific data*, 1(1):1–7, 2014.

Majhi, S., Vitter, J., and Wenk, C. Approximating gromovhausdorff distance in euclidean space. Computational Geometry, 116:102034, 2024.

Rass, S., Konig, S., Ahmad, S., and Goman, M. Metricizing ¨
the euclidean space towards desired distance relations in point clouds. IEEE Transactions on Information Forensics and Security, 19:7304–7319, 2024.

Marin, R., Rampini, A., Castellani, U., Rodola, E., Ovs- `
janikov, M., and Melzi, S. Spectral shape recovery and analysis via data-driven connections. International journal of computer vision, 129:2745–2760, 2021.

Reuter, M., Wolter, F.-E., and Peinecke, N. Laplace–
beltrami spectra as 'shape-dna'of surfaces and solids. Computer-Aided Design, 38(4):342–366, 2006.

Maron, H., Fetaya, E., Segol, N., and Lipman, Y. On the universality of invariant networks. In International conference on machine learning, pp. 4363–4371, 2019.

Rubner, Y., Tomasi, C., and Guibas, L. The Earth Mover's Distance as a metric for image retrieval. *International* Journal of Computer Vision, 40(2):99–121, 2000.

Memoli, F. Gromov–Wasserstein distances and the metric ´
approach to object matching. Foundations of computational mathematics, 11:417–487, 2011.

Rudin, W. et al. *Principles of mathematical analysis*, volume 3. McGraw-hill New York, 1976.

Merchant, A., Batzner, S., Schoenholz, S. S., Aykol, M.,
Cheon, G., and Cubuk, E. D. Scaling deep learning for materials discovery. *Nature*, 624(7990):80–85, 2023.

Satorras, V. G., Hoogeboom, E., and Welling, M. E(n) equivariant graph neural networks. In International conference on machine learning, pp. 9323–9332, 2021.

Schmiedl, F. Computational aspects of the Gromov–
Hausdorff distance and its application in non-rigid shape matching. *Discrete Comp. Geometry*, 57:854–880, 2017.

Schoenberg, I. Remarks to Maurice Frechet's article "Sur la definition axiomatique d'une classe d'espace distances vectoriellement applicable sur l'espace de Hilbert. Annals of Mathematics, pp. 724–732, 1935.

Shi, J., Yang, H., and Carlone, L. Robin: a graph-theoretic approach to reject outliers in robust estimation using invariants. In International Conference on Robotics and Automation (ICRA), pp. 13820–13827, 2021.

Spezialetti, R., Salti, S., and Stefano, L. D. Learning an effective equivariant 3d descriptor without supervision.

In *ICCV*, pp. 6401–6410, 2019.

Steck, H., Ekanadham, C., and Kallus, N. Is cosinesimilarity of embeddings really about similarity? In Companion Proceedings of the ACM on Web Conference 2024, pp. 887–890, 2024.

Su, Z., Welling, M., Pietikainen, M., and Liu, L. Svnet: ¨
Where SO(3) equivariance meets binarization on point cloud representation. In International Conference on 3D Vision, pp. 547–556, 2022.

Thomas, N., Smidt, T., Kearnes, S., Yang, L., Li, L.,
Kohlhoff, K., and Riley, P. Tensor field networks: Rotation-and translation-equivariant neural networks for 3d point clouds. *arXiv:1802.08219*, 2018.

Tshitoyan, V., Dagdelen, J., Weston, L., Dunn, A., Rong, Z., Kononova, O., Persson, K. A., Ceder, G., and Jain, A. Unsupervised word embeddings capture latent knowledge from materials science literature. *Nature*, 571(7763):95–
98, 2019.

Turner, K., Mukherjee, S., and Boyer, D. M. Persistent homology transform for modeling shapes and surfaces. Information and Inference: A Journal of the IMA, 3(4): 310–344, 2014.

Vasylenko, A., Antypov, D., Schewe, S., Daniels, L. M.,
Claridge, J. B., Dyer, M. S., and Rosseinsky, M. J. Digital features of chemical elements extracted from local geometries in crystal structures. *Digital Discovery*, 2025.

Villar, S., Hogg, D. W., Storey-Fisher, K., Yao, W., and Blum-Smith, B. Scalars are universal: equivariant machine learning, structured like classical physics. Advances in Neural Information Processing Systems, 34:28848–
28863, 2021.

Wang, Y. and Solomon, J. M. Deep closest point: Learning representations for point cloud registration. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 3523–3532, 2019.

Ward, L., Agrawal, A., Choudhary, A., and Wolverton, C. A
general-purpose machine learning framework for predicting properties of inorganic materials. *npj Computational* Materials, 2(1):1–7, 2016.

Weston, L., Tshitoyan, V., Dagdelen, J., Kononova, O.,
Trewartha, A., Persson, K. A., Ceder, G., and Jain, A. Named entity recognition and normalization applied to large-scale information extraction from the materials science literature. Journal of chemical information and modeling, 59(9):3692–3702, 2019.

Weyl, H. The classical groups: their invariants and representations. Number 1. Princeton university press, 1946.

Widdowson, D. and Kurlin, V. Resolving the data ambiguity for periodic crystals. *Advances in Neural Information* Processing Systems, 35:24625–24638, 2022.

Widdowson, D. E. and Kurlin, V. A. Recognizing rigid patterns of unlabeled point clouds by complete and continuous isometry invariants with no false negatives and no false positives. In Computer Vision and Pattern Recognition, pp. 1275–1284, 2023.

Xu, Y., Lei, J., Dobriban, E., and Daniilidis, K. Unified fourier-based kernel and nonlinearity design for equivariant networks on homogeneous spaces. In *International* Conference on Machine Learning, pp. 24596–24614, 2022.

Yang, H., Shi, J., and Carlone, L. Teaser: Fast and certifiable point cloud registration. *IEEE Transactions on Robotics*, 37(2):314–333, 2020.

Yarotsky, D. Universal approximations of invariant maps by neural networks. *Constructive Approximation*, 55(1): 407–474, 2022.

Zhou, Q., Tang, P., Liu, S., Pan, J., Yan, Q., and Zhang, S.-
C. Learning atoms for materials discovery. *Proceedings* of the National Academy of Sciences, 115(28):E6411–
E6417, 2018.

Zhu, W., Chen, L., Hou, B., Li, W., Chen, T., and Liang, S. Point cloud registration of arrester based on scaleinvariant points feature histogram. *Scientific Reports*, 12
(1):1–13, 2022.

605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659

## Introduction To Appendices

The main contribution is the roadmap for any data challenge through well-motivated Problem 1.1, where clouds and rigid motion can be replaced with any objects and equivalences. The conditions of completeness and Lipschitz continuity of an invariant I cover the *discriminative* challenge. After these conditions 1.1(a,b,e) are satisfied, the invariant I can be inverted in principle and opens the *generative challenge* of its realizability and inverse continuity in 1.1(c,d,e). Problem 1.1 was stated for unordered clouds under rigid motion but was also solved for *isometry* and compositions of these equivalences with uniform scaling in R
n. For m = 4 points, plane quadrilaterals were previously classified in discrete classes in Fig. 1 (right), while appendix C shows the first continuous maps of the invariant space CRS(R
2; 4).

Conditions 1.1(d,e,f) enable a generation of real clouds in CRS(R
n; m) from their invariants. A full answer to the question
'same or different, and how much different' required complete invariants with Lipschitz continuous metrics.

The key contribution is a theoretically justified solution to Problem 1.1. The experiments on the databases QM9 and GEOM drugs are considered complementary. Example C.1 and its extension in Example C.2 prove that infinitely many pairs of non-isometric clouds C
+ ̸∼= C
− (depending on 4 free parameters and having the same 6 pairwise distances) are distinguished by the new invariants. This result is impossible to justify by any finite experiment. Example C.1 demonstrated the non-zero distances between the complete invariants of C
± in Fig. ??.

The completeness and bi-Lipschitz continuity of the proposed invariants enabled the new experiments on 130K+ real molecules in section 6, which were not previously possible because all past invariants did not satisfy all conditions of Problem 1.1, especially the realizability condition that provides geographic-style maps on cloud spaces. The full solution to Problem 1.1 for n = 2 is justified by Theorem 3.5 and Lemmas 3.3, 5.1, 5.2, 5.3. Theorem 3.3 enables a visualization of cloud spaces, which were unknown even for m = 4 unordered points in R
2.

- The *Cloud Isometry Space* CIS(R
n; m) of clouds of m unordered points under isometry in R
n.

- The *Cloud Rigid Space* CRS(R
n; m) of clouds of m unordered points under rigid motion in R
n.

- The *Cloud Similarity Space* CSS(R
n; m) of clouds of m unordered points under *geometric similarity*, which is a composition of isometry and uniform scaling in R
n.

- The *Cloud Dilation Space* DCS(R
n; m) of clouds of m unordered points under orientation-preserving geometric similarity
(rigid motion and uniform scaling) in R
n.

Here is a summary of the supplementary materials.

660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714
- Appendix A extends section 6 with more details of new invariants and metrics computed on the QM9 database and compared with past pseudo-metrics.

- Appendix C discusses parametrization of CSS(R
2; m) and includes Examples C.1 and C.2 computing the new invariants NDP in detail for infinitely many 4-point clouds from Example C.1. - Appendices B, D, E prove all theoretical results from sections 3, 4, 5, respectively. - The zip folder with supplementary materials includes the code for computing all invariants and metrics as well as tables with all coordinates of colorful maps of QM9 and distances.

## A. Extra Details Of Experiments In Section 6

The default 4-layer network from TensorFlow used a "sequential" mode, 3 epochs, and the settings in Table 6. The only difference between QM9 and GD settings was in the number N of chemical elements in tf.keras.layers.Dense(N), where N = 5 for QM9 and N = 10 for GD.

The maps of QM9 in Fig. 9 are based on eigenvalues and too dense without clear separation. Even if we zoom in, these incomplete invariants will not separate molecules because 3D clouds have at most 3 eigenvalues. The complete invariants Table 6. Parameters of the default 4-layer network for predictions in Table 4.

| LAYER (TYPE)        | OUTPUT SHAPE   | NUMBER OF PARAMETERS   |
|---------------------|----------------|------------------------|
| DENSE (DENSE)       | (NONE, 32)     | 352                    |
| BATCH NORMALIZATION | (NONE, 32)     | 128                    |
| RE LU (RELU)        | (NONE, 32)     | 0                      |
| DENSE 1 (DENSE)     | (NONE, 5)      | 165                    |

715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769
NDP contain much more geometric information. Fig. 10 and 11 show that distances on stornger invariants have larger values and hence better separate molecules, though all these distances have the same Lipschitz constant 2.

Fig. 12 (left) shows the simplest projections of the atomic clouds from QM9, see the familiar molecules such as H2O (water).

Any small region on such a map can be zoomed in and displayed in other invariants from Table 2, see Fig. 12 (right).

770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800

801

802 803 804 805 806 807 808 809 810 811 812 813 814 815 816 817 818 819 820 821 822

823

824

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

## B. Generalization Of Section 3 **And All Proofs In Dimensions** N ≥ 2

This appendix extends all concepts from section 3 to dimensions n ≥ 2, extends Theorem 3.3 to Theorem B.7, which is proved with Theorem B.9 for any n ≥ 2. Lemma B.1 (vector p
⊥
n orthogonal to p1*, . . . , p*n−1 in R
n). Let e1, . . . , en *be an orthonormal basis of* R
n*, so* |ei| = 1 and ei· ej = 0 for i ̸= j. For any n − 1 vectors p1*, . . . , p*n−1 ∈ R
n*, there is a vector* p
⊥
nthat is orthogonal to all p1*, . . . , p*n−1 and has coordinates that are degree n − 1 polynomials in the coordinates of p1*, . . . , p*n−1. Proof of Lemma *B.1.* Below the 'unusual determinant' with the n − 1 vector columns p1*, . . . , p*n−1 and the last column of the n vectors e1*, . . . , e*n is only a short notation for the following expansion by the last column:

Pn i=1
(−1)n+i det(i)ei, where det(i) is the usual (n − 1) × (n − 1) determinant obtained from the n − 1 vector columns p1*, . . . , p*n−1 by removing the i-th row, so we set p
⊥
n =Pn i=1
(−1)n+i det(i)ei.

For example, if n = 2 then p1 = (x1, x2) has the vector p
⊥
2 =

x1 e1 x2 e2

$=x_1e_2-x_2e_1=\left({-x_2,x_1}\right)\perp p_1$ If $n=3,\!$
p1 *. . . p*n−1
...
| *. . .* | en

=
$\epsilon_3$ =. 
| *. . .* | e1
p1 = (x1, x2, x3) and p2 = (y1, y2, y3), then p
⊥
3 =

x1 y1 e1 x2 y2 e2 x3 y3 e3

$=\;\left|\begin{array}{c}x_{21}\\ x_{31}\end{array}\right.$

e3 =
$\begin{array}{c}\\ \\ 2\quad\;y_2\\ 3\quad\;y_3\end{array}\left|\;e_1-\left|\begin{array}{cc}x_1&y_1\\ x_3&y_3\end{array}\right|\;e_2+\left|\begin{array}{cc}x_1&y_1\\ x_2&y_2\end{array}\right|$  . 
p1 × p2 is the *vector* product of p1, p2. To show that p
⊥
nis orthogonal to each pi, we compute the scalar product p
⊥ n
· pi =Pn i=1
(−1)n+1 det(i)ei· pi. Since ei· pi equals the i-th coordinate of the vector pi, the last sum is the expansion of the n × n determinant obtained from the original p
⊥
nabove by replacing the last column with pi. Since the resulting determinant contains two identical columns equal to pi, we conclude that p
⊥
n· pi = 0.

Lemma B.1 holds when given vectors p1*, . . . , p*n−1 ∈ R
n are linearly dependent, even if some pj = 0. Then p
⊥
n = 0 is orthogonal to each pj so that p
⊥ n
· pj = 0.

Definition B.2 extends a point-based representation from Definition 3.1 to dimensions n ≥ 2. The key idea is to represent any m-point cloud A ⊂ R
n relative to (a simplex of) any base sequence of ordered points p1*, . . . , p*n−1 ∈ A. If the vectors p1*, . . . , p*n−1 are linearly independent, they form with the vector p
⊥
nfrom Lemma B.1 a (not necessarily orthogonal) basis in R
n. Below we represent any point p ∈ A by normalized scalar products, which are valid even if p1*, . . . , p*n−1 are linearly dependent.

Definition B.2 (point-based representation PR for n ≥ 2). *For any cloud* A ⊂ R
n of m *unordered points, the* center of mass is O(A) = 
1 m P
p∈A
p. Shift A so that O(A) *is the origin* 0 ∈ R
n*. The* radius of A is R(A) = max p∈A
|p|*. For any* basis *sequence* of points p1, . . . , pn−1 ∈ A*, the* squared distance matrix SD(p1, . . . , pn−1) *consists of* |pi − pj | 2for *i, j* = 0*, . . . , n* − 1, where p0 = 0*. Let* p
⊥
n be the vector in Lemma B.1. For any point q ∈ A \ {p1, . . . , pn−1}*, the* n × (m − n + 1) *matrix* M(A; p1, . . . , pn−1) has a column of scalar products q · p1, . . . , q · pn*. The* point-based representation PR(A; p1*, . . . , p*n−1)
is the pair
-SD(p1, . . . , pn−1), M(A; p1*, . . . , p*n−1).

The normalized *representation* NPR(A; p1, . . . , pn−1) *is obtained by dividing all components of* PR(A; p1*, . . . , p*n−1) by R2(A), except the last row of M(A; p1, . . . , pn−1)*, which is divided by* Rn(A).

Lemma B.3 (PR under isometry). *Let a point cloud* A ⊂ R
n have a base sequence (p1*, . . . , p*n−1).

(a) *Any rigid motion* f of R
n respects point-based representations from Definition B.2 *so that*

${\rm PR}(A;p_{1},\ldots,p_{n-1})={\rm PR}(f(A);f(p_{1}),\ldots,f(p_{n-1}))$.  
(b) *For any orientation-reversing isometry* f of R
n*, the representation* PR(f(A); f(p1), . . . , f(pn−1) *differs from* PR(A; p1, . . . , pn−1) by reversing all signs in the last row of the matrix M(A; p1*, . . . , p*n−1).

825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 Proof of Lemma *B.3.* (a) Since rigid motion preserves distances and scalar products, all components of the point-based representation PR(A; p1*, . . . , p*n−1) are invariant.

880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934
(b) Using a composition with a suitable orientation-preserving isometry (rigid motion), one can assume that f is the mirror reflection in a linear hyperspace H containing the origin 0 and the base sequence p1*, . . . , p*n−1 of A. Since f preserves distances, R(A) and SD(A; p1*, . . . , p*n−1) are invariant. Then f fixes all points from H including p1*, . . . , p*n−1, hence the vector pn from Lemma B.1. Any point q ∈ A \ p1*, . . . , p*n−1 keeps its scalar product q · pi for i = 1*, . . . , n* − 1 and changes the sign of q · pn, because q and its mirror image f(q) have opposite projections to pn. The above arguments hold even if the base sequence p1*, . . . , p*n−1 is degenerate, not generating an (n − 1)-dimensional subspace in R
n. Then there are infinitely many choices of H above and pn = 0, so the last row of M(A; p1*, . . . , p*n−1) consists of zeros. (c) Under uniform scaling by a factor s, all squared distances and scalar products q ·pi, i = 1*, . . . , n*−1, are multiplied by s 2.

The vector p
⊥
nfrom Lemma B.1 is multiplied by s n−1, hence all scalar products q · pn in the last row of M(A; p1*, . . . , p*n−1)
are divided by Rn(A).

The *affine dimension* 0 ≤ aff(A) ≤ n of a cloud A = {p1, . . . , pm} ⊂ R
n is the maximum dimension of the vector space generated by all inter-point vectors pi − pj , i, j ∈ {1*, . . . , m*}. Then aff(A) is an isometry invariant and is independent of an order of points of A. Any cloud A of 2 distinct points has aff(A) = 1. Any cloud A of 3 points that are not in the same straight line has aff(A) = 2.

Lemma B.4 provides a simple criterion for a matrix to be realizable by squared distances of a point cloud in R
n.

Lemma B.4 (realization of distances). (a) A symmetric m × m matrix of sij ≥ 0 *with* sii = 0 is realizable as a matrix of squared distances between points p0 = 0, p1*, . . . , p*m−1 ∈ R
n if and only if the (m − 1) × (m − 1) *matrix* gij =
s0i + s0j − sij 2*has only non-negative eigenvalues.*
(b) *If the condition in (a) holds,* aff(0, p1, . . . , pm−1) equals the number k ≤ m − 1 ≤ n of positive eigenvalues. Also in this case, gij = pi· pj *define the* Gram matrix GM of the vectors p1*, . . . , p*m−1 ∈ R
n, which are uniquely determined in time O(m3) *up to an orthogonal map in* R
n.

Proof of Lemma *B.4.* (a) We extend Theorem 1 from (Dekster & Wilker, 1987) to the case *m < n* + 1 and also justify the reconstruction of p1*, . . . , p*m−1 in time O(m3) uniquely in R
n up to an orthogonal map from the group O(n).

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

The part if ⇐. For any positive semi-definite matrix GM, there is an orthogonal matrix Q such that QT GMQ = D is the diagonal matrix, whose m − 1 diagonal elements are non-negative eigenvalues of GM. The diagonal matrix 
√D consists of the square roots of eigenvalues of GM.

(b) The number of positive eigenvalues of GM equals the dimension k = aff({0, p1*, . . . , p*m−1}) of the subspace in R
n linearly spanned by p1*, . . . , p*m−1. We may assume that all k ≤ n positive eigenvalues of GM correspond to the first k (c) *The normalized point-based representation* NPR(A; p1, . . . , pn−1) in Definition B.2 is preserved by any composition of rigid motion and uniform scaling.

coordinates of R
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
n by the orthogonal map QQ˜T. Hence the reconstruction is unique up to O(n)-transformations. Computing eigenvectors p1*, . . . , p*m−1 needs a diagonalization of GM in time O(m3), see (?)section 11.5]press2007numerical.

Though Lemma B.4 gives a two-sided criterion for realizability of distances by points p1*, . . . , p*m ∈ R
n, the space of distance matrices is highly singular and cannot be easily sampled. Even m = 4 points in R
2 have 6 distances that should satisfy a polynomial equation saying that the tetrahedron with these 6 edge lengths has volume 0.

So a randomly sampled matrix of potential distances for *m > n* + 1 is unlikely to be realizable by a cloud of m ordered points in R
n. Hence Lemma B.4 for m ≤ n + 1 is complemented by Theorem B.7 describing the much more practical realizabilty of a point-based representation.

Chapter 3 in (Liberti & Lavor, 2017) discusses realizations of a complete graph given by a distance matrix in R
n.

Lemma B.5(a) and later results hold for all clouds including degenerate ones, e.g. for 3 points in a straight line.

Any points p1*, . . . , p*n−1 ∈ A have aff(p1*, . . . , p*n−1) ≤ n − 2. For example, any two distinct points in A ⊂ R
3 generate a straight line. Lemma B.5(c) proves that PR(A; p1*, . . . , p*n−1) suffices to reconstruct a cloud A ⊂ R
n for a suitable sequence p1*, . . . , p*n−1. In R
2, any point p1 ̸= O(A) forms a suitable {p1}. In R
3, one can choose any distinct points p1, p2 ∈ A so that the infinite straight line via p1, p2 avoids O(A).

If there are no such p1, p2, then A ⊂ R
3is contained in a straight line L, so aff(A) = 1. In this degenerate case, the stronger condition aff(O(A) ∪ {p1*, . . . , p*n−1}) = aff(A) will help reconstruct A ⊂ L by using any point p1 ̸= O(A). The first step is to reconstruct any ordered sequence from its distance matrix in Lemma B.5(a). Lemma B.5 improves Lemma E.5 in (Widdowson & Kurlin, 2023) by justifying a time for a point cloud reconstruction based on Lemma B.4.

Lemma B.5 (reconstruction). (a) Any sequence of ordered points p1*, . . . , p*m in R
n can be reconstructed (uniquely up to isometry) from the matrix of the Euclidean distances |pi−pj | in time O(m3)*. If all distances are divided by* R = max i=1*,...,m* |pi|, the reconstruction of p1, . . . , pm *is unique up to isometry and uniform scaling in* R
n.

(b) If m ≤ n, the uniqueness of reconstructions in part (a) remains true if we replace isometry by rigid motion in R
n.

935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989
(c) *Any cloud* A ⊂ R
n of m *unordered points can be reconstructed (uniquely up to rigid motion in* R
n*) from a point-based* representation PR(A; p1, . . . , pn−1) in time O(m3) for any p1, . . . , pn−1 ∈ A *with* aff(O(A)∪ {p1*, . . . , p*n−1}) = aff(A).

If aff(A) = n*, then* aff(O(A)∪ {p1*, . . . , p*n−1}) = n−1 *suffices. Any cloud* A ⊂ R
n has a suitable sequence p1*, . . . , p*n−1 in all cases.

Proof of Lemma *B.5.* (a) By translation, we can put p1 at the origin 0 ∈ R
n. Let G be the (m − 1) × (m − 1) matrix Gij =
p 2 i + p 2 j − |pi − pj | 2 2= pi· pj constructed from squared distances between p1 = 0*, . . . , p*m for *i, j* = 2*, . . . , m*.

By Lemma B.4 if G has k ≤ n positive eigenvalues, then p1 = 0*, . . . , p*m can be uniquely determined up to isometry in R 
k ⊂ R
n in time O(m3). If all distances are divided by the same radius R(p{m}), the above construction guarantees uniqueness up to isometry and uniform scaling.

(b) If m ≤ n, any mirror images of p{m} ⊂ R
n after a suitable rigid motion in R
n can be assumed to belong to an
(n − 1)-dimensional hyperspace H ⊂ R
n, where they are matched by a mirror reflection H → H with respect to an
(n − 2)-dimensional subspace S ⊂ H, which is realized by the 180◦ orientation-preserving rotation of R
n around S.

(c) We will reconstruct a cloud A ⊂ R
n so that the center of mass O(A) is the origin 0 ∈ R
n. If aff(A) = *k < n*, the cloud A ⊂ R
n is contained in an affine k-dimensional subspace, which can be rigidly moved to the linear subspace R
k ⊂ R
n for the first k of n coordinates in R
n.

It suffices to reconstruct A ⊂ R
k up to rigid motion in R
k. Since aff(0, p1*, . . . , p*n−1) = k, some k vectors (say) p1*, . . . , p*k from p1*, . . . , p*n−1 form a linear basis of R
k. The k points p1*, . . . , p*k are uniquely reconstructed up to rigid motion in R
k by part (b). Any other point q ∈ A \ {p1*, . . . , p*k} is uniquely determined by its projections (q · pi)/|pi|, which can be found from the first *k < n* rows of the matrix M(A; p1*, . . . , p*n−1) for the point q, see Definition B.2. In the generic case aff(A) = n, the condition aff(0, p1*, . . . , p*n−1) = n−1 means that p1*, . . . , p*n−1 are linearly independent and hence form a linear basis of R
n with the extra vector p
⊥
nfrom Lemma B.1. The sequence (0, p1*, . . . , p*n−1) of n points can be uniquely reconstructed up to rigid motion in R
n by part (b). Any other point q ∈ A \ {p1*, . . . , p*n−1} is uniquely determined by its projections q · pi |pi|to the n basis vectors p1, . . . , pn−1, p⊥
n, which can be found from the column of M(A; p1*, . . . , p*n−1) for q.

Lemma B.5(b) for m = n = 3 implies that any triangle is determined by its sides up to rigid motion in R
3. For example, the sides 3, 4, 5 define a right-angled triangle whose mirror images are not related by rigid motion inside a plane H ⊂ R
3, but are matched by composing a suitable rigid motion in H and a 180◦rotation of R
3around a line in H.

Lemma B.6 (smoothness of PR). *For any cloud* A ⊂ R
n and a base sequence p1, . . . , pn−1 ∈ A*, all components of* PR(A; p1, . . . , pn−1) *have continuous partial derivatives (of any order) with respect to all (coordinates of) points of* A as long as R(A) > 0, so some points of A *remain distinct.* Proof of Lemma *B.6.* The point-based representation PR(A; p{n − 1}) consists of squared distances in the matrix SD(p{n − 1}) and scalar products in the matrix M(A; p{n − 1}) of all points q ∈ A \ p{n − 1} with the vectors p1*, . . . , p*n−1 from the base sequence p{n − 1} and the vector pn ⊥ p1*, . . . , p*n−1 from Lemma B.1. All these components are polynomials in the coordinates of the points of A, so have all continuous partial derivatives. Theorem B.7 extends Theorem 3.3 to dimensions n ≥ 2.

990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 Theorem B.7 (realizability of abstract PR). Let S be a symmetric n × n matrix of sij ≥ 0 *with* sii = 0. Let M *be any* n × (m − n + 1) matrix for m ≥ n. The pair [S, M] *is realizable as a point-based representation* PR(A; p1, . . . , pn−1) for a cloud A ⊂ R
n of m *points with* O(A) = 0 and a base sequence p1, . . . , pn−1 *if and only if (1) the* (n − 1) × (n − 1)
matrix Gij =
1 2
(s1i + s1j − sij ) has only positive eigenvalues, which uniquely determines p1, . . . , pn−1 up to isometry, and (2)
n P−1 j=1
(pi· pj ) +
m−
Pn+1 j=1 Mij = 0 for i = 1, . . . , n*, where* pn = p
⊥
nis the orthogonal vector from Lemma *B.1.*
Proof of Theorem *B.7.* The realizability of S as a matrix of squared distances between n points 0, p1*, . . . , p*n−1 from the base sequence p1*, . . . , p*n−1 follows from Lemma B.4. The orthogonal vector p
⊥
n(also denoted by pn here for uniformity)
from Lemma B.1 complements p1*, . . . , p*n−1 to a linear basis of R
n. By Definition B.2, every element Mij of the matrix M = M(A; p1*, . . . , p*n−1) equals pi· q for some q ∈ A \ {p1*, . . . , p*n−1}, where i = 1*, . . . , n*.

Hence n P−1 j=1
(pi· pj ) +
m−
Pn+1 j=1 Mij = 0 can be rewritten as pi· (P
p∈A
p) = 0 for i = 1*, . . . , n*. These n equations mean that O(A) = 
1 m P
p∈A
p is at the origin 0 ∈ R
n.

Conversely, for any M satisfying condition (2), we interpret every column (M1j *, . . . , M*nj )
Tas a vector of scalar products
(q · p1*, . . . , q* · pn), which determine a position of a point q ∈ A \ {p1*, . . . , p*n−1} in the basis p1*, . . . , p*n.

In Theorem B.7, condition (2) is equivalent to O(A) = 0 ∈ R
n and implies that m − n columns of M consist of free parameters, which determine the remaining column.

For n = 2, condition (1) means only that s12 > 0, so the distance between the points p0 = 0 and p1 is positive.

$i=1,\ldots,n$. 
For n = 3, condition (1) about positive eigenvalues of the 2 × 2 matrix G means that 3 distances a ≤ b ≤ c between points 0, p1, p2 in R
3satisfy a > 0 and a + *b > c*, so the triangle on 0, p1, p2 is non-degenerate. By the cosine theorem p1 · p2 =
1 2
(a 2 + b 2 − c 2), so the matrix G =
a 2 1 2
(a 2 + b 2 − c 2)
1 2
(a 2 + b 2 − c 2) b 2 has a 2 > 0 and a positive determinant:
4 det G = 4a 2b 2 − (a 2 + b 2 − c 2)
2 =
(c 2 − (a 2 − 2ab + b 2))((a 2 + 2ab + b 2) − c 2) =
(c 2 − (a − b)
2)((a + b)
2 − c 2) > 0.

Assuming that 0 < a ≤ b ≤ c, the last inequality is equivalent to one triangle inequality a + *b > c*. Now we extend a point-based representation from Definition B.2 to a complete invariant of a point cloud A under rigid motion in R
n. In applications, A can have distinguished points, for example, heavy atoms in atomic clouds, which can be used to minimize choices for p1*, . . . , p*n−1.

Definition B.8 will extend Definition 3.4 to n > 2 by combining all PR(A; p1*, . . . , p*n−1) in a nested invariant by dropping points p1*, . . . , p*n−1 ∈ A one at a time. This invariant is needed only for comparisons (metric computations), while any cloud A can be stored in computer memory as a single PR(A; p1*, . . . , p*n−1) due to Theorem B.7.

Definition B.8 (NDP : Nested Distributed Projection). Let A ⊂ R
n be any cloud of m unordered points. For any ordered points p1, . . . , pn−2 ∈ A*, let* NDP(A; p1, . . . , pn−2) *be the unordered collection of* PR(A; p1, . . . , pn−1) *for all points* pn−1 ∈ A \ {p1, . . . , pn−2}. Similarly, for any 1 ≤ k ≤ n − 2*, let* NDP(A; p1, . . . , pk−1) *be the unordered collection of* NDP(A; p1, . . . , pk) for all points pk ∈ A \ {p1, . . . , pk−1}*. For* k = 1*, the full* Nested Distributed Projection NDP(A)
depends only on A.

For n = 2 and any cloud A ⊂ R
2, the Nested Distributed Projection NDP(A) in Definition B.8 is the same as in Definition 3.4, i.e. NDP(A) is the unordered collection of point-based representations PR(A; p1) for all p1 ∈ A.

For n = 3 and any A ⊂ R
3, the Nested Distributed Projection NDP(A) is the unordered collection of NDP(A; p1) for all p1 ∈ A. Each NDP(A; p1) is the unordered collection of PR(A; p1, p2) for all p2 ∈ A \ {p1}. Similarly to Definition 3.4, if a cloud A has internal symmetries as in Example 3.2, one can collapse identical objects to a single one with a weight to speed up computations. We avoid collapsing only to simplify arguments for n > 2.

Lemma B.5(c) implies that any cloud A ⊂ R
n of m unordered points can be reconstructed from NDP(A) uniquely up to rigid motion. Indeed, NDP(A) contains (nested) PRs depending on all possible n − 1 points p1*, . . . , p*n−1 ∈ A. At least one PR(A; p1*, . . . , p*n−1) satisfies Lemma B.5(c) and suffices to reconstruct A uniquely up to rigid motion. In Theorem B.9 for n > 2, the equality NDP(A) = NDP(B) means a bijection β : NDP(A) → NDP(B) respecting the nested structure of all PRs in Definition B.8. In detail, for any 1 ≤ k ≤ n − 1 and points p1*, . . . , p*k, the bijection β matches NDP(A; p1*, . . . , p*k) with a unique NDP(B; q1*, . . . , q*k) for some q1*, . . . , q*k ∈ B. If n = 3, then β matches every NDP(A; p1) with a unique NDP(B; q1) in the sense that this bijection NDP(A; p1) → NDP(B; q1) matches PR(A; p1, p2) for every p2 ∈ A \ {p1} with PR(B; q1, q2) for a unique q2 ∈ B − {q1}.

Theorem B.9 (completeness of NDP). *The Nested Distributed Projection is complete in the sense that any clouds* A, B ⊂ R
n of m *unordered points are related by rigid motion in* R
n *if and only if* NDP(A) = NDP(B) *so that there is a* bijection NDP(A) → NDP(B) *matching all* PRs. Proof of Theorem *B.9.* The part *only if* : we will prove that any rigid motion f moving the cloud A to B = f(A) implies that NDP(A) = NDP(B). By Lemma B.3(a) the rigid motion f matches every PR(A; p1*, . . . , p*n−1) from NDP(A) with PR(B; f(p1)*, . . . , f*(pn−1)). Then, for any 1 ≤ k ≤ n − 2 and p1*, . . . , p*k ∈ A, we get a bijection NDP(A; p1*, . . . , p*k) → NDP(B; f(p1)*, . . . , f*(pk)) Hence f induces a bijecton NCP(A) → NCP(B) between all PRs respecting the nested structure in Definition B.8.

1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079 1080 1081 1082 1083 1084 1085 1086 1087 1088 1089 1090 1091 1092 1093 1094 1095 1096 1097 1098 1099