*014*

*024*

*034*

*054*

# Machine learning on rigid classes of Euclidean clouds of unordered points

### Abstract

Most real objects allow infinitely many different representations. Robust machine learning aims to use only invariant features independent of object representations to guarantee that any output (class label or predicted property) is preserved if the same object is represented differently. For Euclidean clouds of unordered points under rigid motion, we introduce complete invariants (with no false negatives, no false positives) and a Lipschitz continuous distance that satisfies all metric axioms and is computable in polynomial time of the number of points. The new realizability property implies that the space of all rigid clouds is efficiently parametrized by vectorial invariants like geographic coordinates. The proposed invariants distinguished all rigid classes of atomic clouds in the world's largest collections of molecules with 3D coordinates and predicted chemical elements by pure geometry with over 98% accuracy.

### 1. Importance of complete and bi-continuous invariants for ML on data with real values

![](_page_0_Figure_7.jpeg)

This paper formalizes practically important conditions for application-driven ML on real objects with ambiguous representations and develops new canonical representations satisfying these conditions for any *clouds* (finite sets) of unordered points in Euclidean space R <sup>n</sup>. Such a cloud is the most basic form of a real object from cars to molecules [\(Wang & Solomon,](#page-11-0) [2019\)](#page-11-0), e.g. a set of corners or atoms.

Many objects are *rigid* in the sense that their shape and properties are preserved under *rigid motion* composed of translations and rotations in R <sup>n</sup> [\(Atz et al.,](#page-8-0) [2021\)](#page-8-0), which form the group SE(n). The slightly weaker relation is by *isometries* (distance-preserving transformations), which form the group E(n). The practical cases are dimensions n ≤ 3 and larger numbers m (hundreds) of unordered points without outliers [\(Shi et al.,](#page-11-1) [2021\)](#page-11-1) because atoms have stable nuclei.

Any rigid cloud has infinitely many representations, e.g. lists of point coordinates, but the shape and properties of an object should be independent of a coordinate system. Points are usually unordered and even simple molecules have many indistinguishable atoms. Hence predictions should not depend on point ordering. On another hand, different rigid classes of chemically identical molecules can have different functional properties such as solubility and hence therapeutic effectiveness. If not all rigid classes are distinguished, drugs can become useless, implying human suffering and financial losses for manufacturers [\(Morissette et al.,](#page-10-0) [2003\)](#page-10-0).

A repeated scan or measurement of the same object can produce a slightly different cloud that cannot be exactly matched with the original one by rigid motion, also due to atomic vibrations [\(Feynman,](#page-9-0) [1971\)](#page-9-0). If noise is ignored up to any threshold ε > 0, sufficiently many tiny perturbations make all clouds equivalent by the transitivity axiom: if A ∼ B and B ∼ C, then A ∼ C [\(Brink et al.,](#page-8-1) [1997\)](#page-8-1).

Since all small deviations between rigid classes of point clouds should be distinguished, all these classes live in a continuous space of rigid clouds, see Fig. [1](#page-0-0) (left). This space was continuously parametrized only in dimension n = 1 or for m = 3 points or Fig. [1](#page-0-0) (right) leaving other cases open.

Figure 1. Left: rigid classes of m unordered points in R n form a continuous space, which had no complete and bi-continuous invariants for m > 3, n > 1. Right: the space of 3 points under isometry is parametrized by distances 0 < a ≤ b ≤ c ≤ a + b.

Machine learning previously focused on discrete classifications or success measures for finite datasets, which can be considered discrete samples (of measure 0) in continuous spaces. For generalizability to all real data outside finite datasets, application-driven ML needs new conditions formalized in Problem [1.1](#page-0-1) below. [\(Li et al.,](#page-10-1) [2021;](#page-10-1) [Dym &](#page-9-1) [Gortler,](#page-9-1) [2024;](#page-9-1) [Maennel et al.,](#page-10-2) [2024;](#page-10-2) [Nigam et al.,](#page-10-3) [2024\)](#page-10-3) studied complete invariants without realizability and Lipschitz bi-continuity [\(Morris et al.,](#page-10-4) [2024;](#page-10-4) [Cahill et al.,](#page-8-2) [2024\)](#page-8-2).

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

108 109 Problem 1.1. *Find a complete and bi-continuous invariant* I : {*clouds of unordered points in* <sup>R</sup> <sup>n</sup>} → *a space* X *with a distance* d *such that all the conditions below hold.*

*(a) Completeness: any clouds* A, B *of unordered points are related by a rigid motion of* R <sup>n</sup> *if and only if* I(A) = I(B)*.*

*(b) Metric axioms: 1)* d(α, β) = 0 ⇔ α = β*; 2)* d(α, β) = d(β, α)*; 3)* d(α, β)+d(β, γ) ≥ d(α, γ) *for all* α, β, γ ∈ X*.*

*(c) Lipschitz continuity: there is a constant* λ *such that if each point of a cloud* A ⊂ R <sup>n</sup> *is perturbed up to Euclidean distance* ε*, then* I(A) *changes by at most* λε *in the metric* d*.*

*(d) Realizability: the image* {I(A) | *clouds* A ⊂ <sup>R</sup> <sup>n</sup> *of unordered points*} *is parametrized so that one can reconstruct* A *up to rigid motion from any* realizable *value of* I*.*

*(e) Point matching: there is a constant* µ *that guarantees for any clouds* A, B *a rigid motion matching all points of* A, B *up to Euclidean distance* µd(I(A), I(B))*.*

*(f) Computability: for a fixed dimension* n*, the invariant* I*, the metric* d*, and all constructions in (d) and (e) are computable in polynomial time of the number of points.*

Clouds and rigid motion can be replaced with any data (graphs, meshes) and equivalences (also allowing reflections or uniform scaling), respectively, so Problem [1.1](#page-0-1) makes sense for any real data with ambiguous representations.

The completeness (or injectivity) in [1.1\(](#page-0-1)a) fully answers the question "same or different?" A complete invariant I has the ultimate expressive power and always distinguishes all clouds A ̸∼= B (not only from a finite dataset) that cannot be matched by rigid motion, so I is a descriptor with *no false negatives* and *no false positives*. The universal approximation aims for the completeness of infinite-size invariants [\(Maron et al.,](#page-10-5) [2019;](#page-10-5) [Keriven & Peyre´,](#page-9-2) [2019;](#page-9-2) [Yarotsky,](#page-11-2) [2022\)](#page-11-2), so polynomial time in [1.1\(](#page-0-1)f) makes all conditions harder.

A complete invariant can give a discontinuous metric, say d(A, B) = 1 for all non-equivalent clouds without quantifying the similarity of near-duplicates. The continuity in [1.1\(](#page-0-1)c) is necessary for smoothness and hence for any gradient-based optimisation Due to the first axiom in [1.1\(](#page-0-1)b), any metric d detects rigidly equivalent clouds by checking if d(A, B) = 0. Without the first axiom, many more distances including the zero d ≡ 0 satisfy the other axioms and are called *pseudo-metrics* [\(Brecheteau](#page-8-3) ´ , [2019\)](#page-8-3). If the third axiom in [1.1\(](#page-0-1)b) fails with any additive error ε > 0, results of clustering may not be trustworthy [\(Rass et al.,](#page-10-6) [2024\)](#page-10-6).

The realizability in [1.1\(](#page-0-1)d) implies that the invariant I is an invertible 1-1 map from the complicated *Cloud Rigid Space* CRS(<sup>R</sup> <sup>n</sup>; m) of classes of clouds under rigid motion to the explicitly parametrized space I(CRS(<sup>R</sup> <sup>n</sup>; m)) of realizable values. Then with 100% certainty, we can sample any value in I(CRS(<sup>R</sup> <sup>n</sup>; m)) and reconstruct its cloud A ⊂ <sup>R</sup> n.

The 1-1 point matching in [1.1\(](#page-0-1)e) can be interpreted as the Lipschitz continuity of the inverse map I −1 so that any close values I(A), I(B) guarantee the closeness of A, B under rigid motion. Conditions [1.1\(](#page-0-1)c,e) mean that I is bi-Lipschitz: ε/µ ≤ d(I(A), I(B)) ≤ λε, where ε is the minimum perturbation needed to match all points of A, B.

A partial matching, e.g. ignoring outliers, is harder to formalize. Indeed, if any clouds sharing all points except one are called equivalent, the transitivity axiom allows us to build a chain of equivalences A<sup>1</sup> ∼ · · · ∼ A<sup>k</sup> changing one point at a time, which can make all clouds equivalent.

One can define metrics satisfying [1.1\(](#page-0-1)a,b,c) by minimizing or deviations of unordered points over infinitely many transformations but polynomial time in [1.1\(](#page-0-1)f) makes Problem [1.1](#page-0-1) notoriously hard, previously solved only for m = 3 points.

Conditions [1.1\(](#page-0-1)a,b,c,f) and [1.1\(](#page-0-1)d,e,f) formalize the *discriminative* and *generative* goals, respectively. A full solution to Problem [1.1](#page-0-1) will imply that the rigid classes of clouds can be efficiently visualized in the *moduli* space I(CRS(<sup>R</sup> <sup>n</sup>; m)) replacing any latent space of non-invariants or incomplete (or discontinuous or non-realizable) invariants. Geographically, I(CRS(<sup>R</sup> <sup>n</sup>; m)) can be compared with Earth's map, where any location can be reconstructed with all properties (altitude, precipitation, images, ...) from the latitude and longitude coordinates in known (realizable) ranges.

Contributions. Problem [1.1](#page-0-1) formalizes the necessary conditions for any application-driven ML on real objects. The new invariant Nested Distributed Projection solves Problem [1.1](#page-0-1) for all clouds of m unordered points in dimension n = 2. Any cloud A ⊂ R <sup>n</sup> can be reconstructed from a small part of the invariant (a vector in R <sup>n</sup>(m−(n+1)/2)) whose realizability in [1.1\(](#page-0-1)d) is guaranteed by explicitly written inequalities. Hence coordinates of this vector can be chosen in known ranges like latitude and longitude on Earth maps. The appendices cover all dimensions n > 2. The Python/C++ code is in the supplementary materials.

# 2. Past work on continuous metrics for clouds

Ordered points. Kendall's shape theory [\(Kendall et al.,](#page-9-3) [2009\)](#page-9-3) studies m ordered points p1, . . . , p<sup>m</sup> ∈ <sup>R</sup> <sup>n</sup> under isometries from E(n). In this case, a complete invariant is the distance matrix [\(Schoenberg,](#page-11-3) [1935;](#page-11-3) [Kruskal & Wish,](#page-9-4) [1978\)](#page-9-4) or the Gram matrix of scalar products p<sup>i</sup> · p<sup>j</sup> , see chapter 2.9 in [\(Weyl,](#page-11-4) [1946\)](#page-11-4), [\(Villar et al.,](#page-11-5) [2021\)](#page-11-5). A bruteforce extension to m unordered points requires m! matrices due to m! permutations, which is ruled out by [1.1\(](#page-0-1)f).

Point cloud registration for unordered points samples rotations [\(Lin et al.,](#page-10-7) [1986;](#page-10-7) [Yang et al.,](#page-11-6) [2020\)](#page-11-6) and uses scaleinvariant features [\(Lowe,](#page-10-8) [1999;](#page-10-8) [2004;](#page-10-9) [Huang et al.,](#page-9-5) [2006\)](#page-9-5) to approximately match clouds. If approximately matched

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

clouds are called equivalent, sufficiently many gradual perturbations make all clouds equivalent due to the transitivity axiom. Hence all rigid classes should be distinguished by a distance d that becomes zero only on rigidly equivalent clouds. Trying to sort points along a fixed direction or in a clockwise order around their center of mass leads to discontinuities because distant points can have equal projections to a line or a circle. A basis (say, of principal directions) of a cloud [\(Spezialetti et al.,](#page-11-7) [2019;](#page-11-7) [Zhu et al.,](#page-11-8) [2022;](#page-11-8) [Kurlin,](#page-9-6) [2024\)](#page-9-6) is similarly unstable under perturbations of points in cases of high symmetry, e.g. when eigenvalues become equal, which often happens for real molecules in our main application. Converting a cloud by using extra parameters into a more complex object such as a continuous field R <sup>3</sup> → <sup>R</sup> [\(Chauvin et al.,](#page-8-4) [2022\)](#page-8-4) or the persistent homology transform leads to the harder analog of Problem [1.1](#page-0-1) for continuous surfaces instead of discrete clouds [\(Turner et al.,](#page-11-9) [2014\)](#page-11-9).

Neural networks [\(Bronstein et al.,](#page-8-5) [2021\)](#page-8-5) can guarantee invariance or equivariance [\(Thomas et al.,](#page-11-10) [2018;](#page-11-10) [Kondor](#page-9-7) [& Trivedi,](#page-9-7) [2018;](#page-9-7) [Cohen et al.,](#page-8-6) [2019;](#page-8-6) [Fuchs et al.,](#page-9-8) [2020;](#page-9-8) [Deng et al.,](#page-9-9) [2021\)](#page-9-9). An *equivariant* descriptor E satisfies the weaker condition E(f(A)) = T<sup>f</sup> (E(A)) for any rigid motion f of a cloud A, where T<sup>f</sup> may not be the identity as required for invariants [\(Satorras et al.,](#page-10-10) [2021;](#page-10-10) [Chen et al.,](#page-8-7) [2021;](#page-8-7) [Aronsson,](#page-8-8) [2022;](#page-8-8) [Assaad et al.,](#page-8-9) [2023;](#page-8-9) [Xu et al.,](#page-11-11) [2022;](#page-11-11) [Su et al.,](#page-11-12) [2022\)](#page-11-12). Any linear combination of points such as the center of mass is equivariant but cannot distinguish clouds under translation. Equivariants were used for predicting forces acting on atoms to move them to a more optimal configuration. These time-dependent clouds A<sup>t</sup> can be studied directly by their invariant values I(At) without intermediate forces. So neural networks optimize millions of parameters, see Table 4 in [\(Goyal et al.,](#page-9-10) [2021\)](#page-9-10), to improve accuracies [\(Dong et al.,](#page-9-11) [2018;](#page-9-11) [Akhtar & Mian,](#page-8-10) [2018;](#page-8-10) [Laidlaw & Feizi,](#page-9-12) [2019;](#page-9-12) [Guo et al.,](#page-9-13) [2019;](#page-9-13) [Colbrook et al.,](#page-8-11) [2022\)](#page-8-11) but need re-training any for new data and will have better generalizability if their inputs are invariants satisfying the conditions of Problem [1.1](#page-0-1) for all possible clouds in R n.

General metrics between fixed clouds extend to their rigid classes by minimization over infinitely many rigid motions [\(Huttenlocher et al.,](#page-9-14) [1993;](#page-9-14) [Chew & Kedem,](#page-8-12) [1992;](#page-8-12) [Chew et al.,](#page-8-13) [1999\)](#page-8-13). In R 2 , the time O(m<sup>5</sup> log m) [\(Chew](#page-8-14) [et al.,](#page-8-14) [1997\)](#page-8-14) for the Hausdorff distance [\(Hausdorff,](#page-9-15) [1919\)](#page-9-15) will be improved in Theorem [5.3](#page-5-0) to O(m<sup>3</sup>.<sup>5</sup> log m) for a new metric, see approximations in [\(Goodrich et al.,](#page-9-16) [1999\)](#page-9-16). The Gromov-Hausdorff and Gromov-Wasserstein metrics [\(Memoli](#page-10-11) ´ , [2011\)](#page-10-11) are defined for metric-measure spaces also by minimizing over infinitely many correspondences between points, but cannot be approximated with a factor less than 3 in polynomial time unless P=NP, see Corollary 3.8 in [\(Schmiedl,](#page-11-13) [2017\)](#page-11-13) and polynomial algorithms for partial cases in [\(Majhi et al.,](#page-10-12) [2024\)](#page-10-12). Also, computing a metric between rigid classes of clouds is only a small part

![](_page_2_Diagram_8.jpeg)

of Problem [1.1.](#page-0-1) Indeed, to efficiently navigate on a real planet, in addition to distances between cities, we need a satellite-type view of the whole planet and hence a realizable bi-continuous invariant I, which can be considered an analog of the latitude and longitude coordinates on Earth.

Can we 'sense' a shape? Problem [1.1](#page-0-1) asks the questions 'same or different clouds, and how much different?' The related problem 'Can we hear the shape of a drum?' [\(Kac,](#page-9-17) [1966\)](#page-9-17) has the negative answer in terms of 2D polygons indistinguishable by spectral invariants [\(Gordon et al.,](#page-9-18) [1992a](#page-9-18)[;b;](#page-9-19) [Reuter et al.,](#page-10-13) [2006;](#page-10-13) [Cosmo et al.,](#page-8-15) [2019;](#page-8-15) [Marin et al.,](#page-10-14) [2021\)](#page-10-14). Problem [1.1](#page-0-1) looks for stronger invariants that can completely 'sense' (not only 'hear') all rigid clouds in any R n.

The partial cases when Problem [1.1](#page-0-1) was solved are only n = 1 or m ≤ 3. In dimension n = 1, any rigid motion of <sup>R</sup> is a translation, so the Cloud Rigid Space CRS(R; m) of m points p1, . . . , p<sup>m</sup> ∈ <sup>R</sup> is the space <sup>R</sup> m−1 <sup>+</sup> of sequential inter-point distances d<sup>i</sup> = pi+1 − p<sup>i</sup> > 0 for i = 1, . . . , m − 1. Including reflections, the *Cloud Isometry Space* CIS(R; m) is the quotient of <sup>R</sup> m−1 <sup>+</sup> under the cyclic equivalence (d1, . . . , dm−1) ∼ (dm−1, . . . , d1). For clouds of m = 2 points in any dimension n ≥ 1, CRS(<sup>R</sup> <sup>n</sup>; 2) is parametrized by a single inter-point distance d > 0. The final known case is m = 3 due to the SSS theorem saying that any triangles are congruent (isometric) if and only if they have the same side lengths. The space CIS(<sup>R</sup> <sup>n</sup>; 3) of 3-point clouds has the geographic-style parametrization {0 < a ≤ b ≤ c ≤ a + b} by inter-point distances a, b, c so that any (a, b, c) ∈ CIS(<sup>R</sup> <sup>n</sup>; 3) generates a uniquely triangle under isometry. Problem [1.1](#page-0-1) asks for a similarly explicit parametrization of CRS(<sup>R</sup> <sup>n</sup>; m) for all m ≥ 4 and n ≥ 2.

Recent advances are the extensions [\(Delle Rose et al.,](#page-9-20) [2024;](#page-9-20) [Hordan et al.,](#page-9-21) [2024\)](#page-9-21) of the WL test [\(Leman & Weisfeiler,](#page-10-15) [1968\)](#page-10-15), giving a binary answer [\(Brass & Knauer,](#page-8-16) [2000;](#page-8-16) [2004\)](#page-8-17) by distinguishing all non-isometric clouds but without Lipschitz continuous metrics for all clouds including degenerate ones. Attempting to extend the SSS theorem, the Sorted Distance Vector (SDV) of all <sup>m</sup>(m−1) distances between m ≥ 4 unordered points distinguishes all non-isometric clouds in general position in R <sup>n</sup> [\(Boutin & Kemper,](#page-8-18) [2004\)](#page-8-18) but not infinitely many 4-point clouds in R 2 , see Fig. [2.](#page-2-0)

Figure 2. The infinite family of non-isometric clouds C <sup>+</sup> ̸≃ C − sharing p1, p2, p<sup>3</sup> and depending on free parameters a, b, c, d.

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

The SDV was strengthened [\(Widdowson & Kurlin,](#page-11-14) [2022\)](#page-11-14) to the Pointwise Distance Distribution (PDD), which still cannot distinguish infinitely many non-isometric clouds in R 3 , see Fig. S4 in [\(Pozdnyakov & Ceriotti,](#page-10-16) [2022\)](#page-10-16). All these counter-examples were distinguished by the Simplexwise Centered Distributions from [\(Widdowson & Kurlin,](#page-11-15) [2023\)](#page-11-15), which satisfy [1.1\(](#page-0-1)a,b,c,f) but not [1.1\(](#page-0-1)d,e). Distance-based invariants do not allow easy realizability already for m = 4 points in R <sup>2</sup> whose 6 inter-point distances should satisfy a non-trivial polynomial equation saying that the tetrahedron on 4 points has volume 0 in R 2 . Hence random distances between m > 3 unordered points are realized by a point cloud in R <sup>2</sup> with probability 0 [\(Duxbury et al.,](#page-9-22) [2016\)](#page-9-22).

### 3. Complete invariants of unordered clouds

Any point p = (x1, . . . , xn) ∈ <sup>R</sup> <sup>n</sup> has *Euclidean* norm |p| = s Pn i=1 x 2 i . Any points p and q = (y1, . . . , yn) ∈ <sup>R</sup> n are also interpreted as vectors, have the *Euclidean* distance |p − q| and the *scalar* (dot) product of p · q = Pn i=1 xiy<sup>i</sup> . Any vectors p ⊥ q are *orthogonal* if and only if p · q = 0.

While past representations used one basis (say, of principal directions of a given cloud A ⊂ R <sup>n</sup>), this section introduces a new representation based on variable projections that depend on n − 1 ordered points in C consisting of m unordered points. For simplicity, we consider n = 2 when we have only m choices for a single point p ∈ A in Fig. [3.](#page-3-0)

![](_page_3_Diagram_9.jpeg)

Figure 3. A Point-based Representation (PR) encodes a cloud A in the basis of a point p ∈ A. All PRs are combined into the complete invariant NDP(A). NDPs are compared by the Nested Bottleneck Metric (NBM) computed from a complete bipartite graph Γ(A, B) with weights equal to distances between PRs.

For any cloud A ⊂ R <sup>2</sup> of m unordered points, the *center of mass* is <sup>O</sup>(A) = <sup>1</sup> m P p∈A p. Shift A so that O(A) is the origin 0 ∈ R . For any p = (x1, x2) ∈ A, the vector p <sup>⊥</sup> = (−x2, x1) is orthogonal to p, so p · p <sup>⊥</sup> = 0, which holds even if p = 0. If p is not at the origin (center of mass of A), we use the orthogonal basis p, p<sup>⊥</sup> to represent all other points of A. Definition [3.1](#page-3-1) makes sense for p = 0.

Definition 3.1 (point-based representation PR(A; p)). *Let* A ⊂ R <sup>2</sup> *be a cloud with the center of mass at the origin* 0*. Fix a* base *point* p = (x, y) ∈ A*, set* p <sup>⊥</sup> = (−y, x)*. For any* q ∈ A \ {p}*, the* 2×(m −1) *matrix* M(A; p) *has a column of the scalar products* q · p, q · p <sup>⊥</sup>*. The* point-based representation *of* A *is the pair* PR(A; p) = -|p| 2 , M(A; p) 

We use |p| 2 and scalar products to make all components polynomial (smooth) in coordinates. The matrix M(A; p) has two rows (ordered according to p, p⊥) and m − 1 unordered columns, and can be considered a *fixed cloud* of m − 1 unordered points in R 2 , not under rigid.

Example 3.2 (regular polygons in R ). *(a) For* m ≥ 2*, let* A<sup>m</sup> = {R exp 2πi√ −1 m } ⊂ <sup>R</sup> 2 *,* i = 1, . . . , m*, be the vertex set of a regular* m*-sided polygon. Then* A<sup>m</sup> *has the center of mass* O(Am) = (0, 0) *at the origin and is inscribed in the circle of the radius* R = R(Am)*. In Definition [3.1,](#page-3-1) choose the point* p = (R, 0) ∈ Am*, which doesn't affect* PR(Am; p) *due to the rotational symmetry of* Am*. Then the matrix* M(Am; p) *consists of* m − 1 *columns* R<sup>2</sup> cos(2πi/m) R<sup>2</sup> sin(2πi/m) *,* i = 1, . . . , m − 1*. The pair is* PR(Am; <sup>p</sup>) = h R<sup>2</sup> , R<sup>2</sup> cos <sup>2</sup>πi m R<sup>2</sup> sin <sup>2</sup>πi m m−1 i=1 i .

*(b) Let the cloud* B<sup>m</sup> ⊂ <sup>R</sup> <sup>2</sup> *be* A<sup>m</sup> *after adding the extra point at the origin* 0 ∈ R 2 *. For any point* p ∈ Am*, the new point-based representation* PR(Bm; p) *is obtained from* PR(Am; p) *above by adding the zero column to the matrix* M(Am; p)*. For the extra point at the origin* 0*, the representation is* PR(Bm; 0) = [0, M(Bm; 0)]*, where* M(Bm; 0) *is the* 2 × m *matrix consisting of zeros.*

Theorem 3.3 (realizability of abstract PR). *Let* s > 0 *and* M *be any* 2 × (m − 1) *matrix for* m ≥ 2*. The pair* [s, M] *is realizable as a point-based representation* PR(A; p) *for a cloud* A ⊂ R <sup>n</sup> *of* m *unordered points with* O(A) = 0 *and a point* p ∈ A *if and only if* s + mP−1 j=1 M1<sup>j</sup> = 0 = mP−1 j=1 M2<sup>j</sup> *.*

In Theorem [3.3,](#page-3-2) s = |p| 2 is the squared distance from a point p ∈ A to 0 ∈ R 2 . The equations say that the sums of the scalar products (q · p) and (q · p <sup>⊥</sup>) for all q ∈ A equal to 0, which is equivalent to Pq ∈ A = 0 meaning that the center of mass O(A) is 0. Hence s > 0 and m − 2 columns of M can be considered free parameters.

Definition [3.4](#page-4-0) combines point-based representations PR(A; p) for all points p ∈ A into one invariant NDP (Nested Distributed Projection) that will be proved to satisfy

*257*

*264*

*266*

all conditions of Problem [1.1.](#page-0-1) The major advantage of NDP is its applicability to all real clouds A ⊂ R <sup>2</sup> without any requirement of general position. Some points of a cloud A may coincide, so A can be a multiset of points.

Definition 3.4 (invariants NDP and NCP). *Let* A ⊂ R <sup>2</sup> *be any cloud of* m *unordered points. The* Nested Distributed Projection NDP(A) *is the unordered set of* PR(A; p) *for all* p ∈ A*. If* k > 1 *representations* PR(A; p) *are equal then we collapse them to one representation with the weight* k/m*. The resulting set of unordered* PR*s with weights is called the* Nested Compressed Projection NCP(A)*.*

Table 1. Acronyms and references of all key concepts in the paper.

| PR  | P | OINT  |           | BASED R    | EPRESENTATION |           | D       | EF | 3.1 |
|-----|---|-------|-----------|------------|---------------|-----------|---------|----|-----|
| NDP | N | ESTED | D         | ISTRIBUTED | P             | ROJECTION | D       | EF | 3.4 |
| PRM | P | OINT  |           | BASED R    | EPRESENT      | M         | ETRIC D | EF | 4.2 |
| BMD | B |       | OTTLENECK | M          | ATCHING       | D ISTANCE | D       | EF | 4.3 |
| NBM | N | ESTED | B         | OTTLENECK  | M             | ETRIC     | D       | EF | 4.4 |

For the cloud A<sup>m</sup> from Example [3.2,](#page-3-3) the Nested Distributed Projection NDP(Am) consists of m identical representations, so NCP(Am) is the single representation PR(Am; p) with weight 1. The invariant NDP is an expanded version of the NCP, where all PRs have equal weights 1/m. The full invariant NDP(A) includes the faster (linear-time) vector of squared distances |p| from the center of mass O(A) = 0 ∈ <sup>R</sup> 2 to all points p ∈ A. If A has a distinguished point p, e.g. a special atom in a molecule, the point-based representation PR(A; p) is invariant.

Theorem 3.5 (completeness of NDP). *The Nested Distributed Projection is complete in the sense that any clouds* A, B ⊂ R <sup>2</sup> *of* m *unordered points are related by rigid motion in* R 2 *if and only if* NDP(A) = NDP(B) *so that there is a bijection* NDP(A) → NDP(B) *matching all* PR*s.*

Under a mirror reflection, for any p ∈ A, one can assume after applying rigid motion that the basis p, p<sup>⊥</sup> maps to its mirror image p, −p <sup>⊥</sup>. The mirror image A¯ has NDP(A¯) equal to NDP(A) that is obtained from NDP(A) by reversing all signs in the last row of M(A; p) for each p ∈ A.

The completeness of NDP(A) Theorem [3.5](#page-4-4) implies the completeness of the pair NDP(A), NDP(A) under isometry including reflections. Further work can simplify this pair to a smaller invariant while keeping the completeness. Since a bijection NDP(A) → NDP(B) between all (uncollapsed) PRs induces a bijection NCP(A) → NCP(B) respecting all weights of collapsed PRs, Theorem [3.5](#page-4-4) implies the completeness of NCP under rigid motion in R 2 .

# 4. A metric on complete invariants of clouds

This section will define the metric NBM on invariants NDP by using the bottleneck distance BD in Definition [4.1,](#page-4-5) a metric on point-based representations (PRs) in Definition [4.2,](#page-4-1) and a bottleneck matching distance in Definition [4.3.](#page-4-2)

Definition 4.1 (bottleneck distance BD). *For any* v = (v1, . . . , vn) ∈ <sup>R</sup> <sup>n</sup>*, the* Minkowski *norm is* ||v||<sup>∞</sup> = max i=1,...,n |v<sup>i</sup> |*. For clouds* A, B ⊂ <sup>R</sup> <sup>n</sup> *of* m *unordered points, the* bottleneck distance BD(A, B) = inf g:A→B sup p∈A ||p − g(p)||<sup>∞</sup> *is minimized over all bijections* g : A → B*.*

Though the bottleneck distance is defined as a minimum for m! bijections A → B between m-point clouds, Theorem 6.5 in [\(Efrat et al.,](#page-9-23) [2001\)](#page-9-23) computes BD(A, B) in time O(m<sup>1</sup>.<sup>5</sup> log<sup>2</sup> m) by filtering out distant points. The bruteforce extension of BD(A, B) under rigid motion need a minimization for infinitely many rotations. NDP(A) consists of only m point-based representations PR(A; p) = [|p| 2 , M(A; p)], one for each p ∈ A. The BD algorithm can compare any 2 × (m − 1) matrices M(A; p) and M(B; q) as fixed clouds of unordered columns (points in R 2 ).

In Definition [4.2,](#page-4-1) the notation M/R means that all elements of the matrix M(A; p) are divided by the *radius* R(A) = max p∈A |p| of a cloud A. Then PRM and further metrics have units of original points, e.g. in meters. One more division by R(A) makes metrics invariant under uniform scaling.

Definition 4.2 (Point-Based Representation Metric PRM). *Let* PR(A; p),PR(B; q) *be point-based representations of clouds* A, B ⊂ R <sup>2</sup> *of* m *unordered points for base points* p ∈ A *and* q ∈ B*, respectively, see Definition [3.1.](#page-3-1) The* Point-based Representation Metric *between the* PR*s above is* PRM = max{ | |p|−|q| |, |R(A)−R(B)|, w<sup>M</sup> }, *where* <sup>w</sup><sup>M</sup> = BD M(A; p) R(A) , M(B; q) R(B) *, see Definition [4.1.](#page-4-5)*

We defined PRM as the maximum of 3 metrics to guarantee the metric axiom (if PRM = 0 then A ∼= B) and the simplest Lipschitz constant λ = 2 in [1.1\(](#page-0-1)d), see all proofs in appendix [D.](#page-29-0) Replacing the maximum with (say) a sum gives a metric with a higher constant λ depending on m.

Definition 4.3 (bottleneck matching distance BMD(Γ)). *Let* Γ *be a complete bipartite graph with* m *white vertices and* m *black vertices so that every white vertex is connected to every black vertex by an edge* e *of a weight* w(e) ≥ 0*. A* vertex matching *in* Γ *is a set* E *of* m *disjoint edges of* Γ*. The* weight W(E) = max e∈E w(e) *is the largest weight in* E*. The* bottleneck matching distance *of the graph* Γ *is* BMD(Γ) = min E W(E) *is minimized over all vertex matchings.*

Because Γ is bipartite, any edge from a vertex matching E joins a white vertex with a black vertex. Then BMD(Γ) is minimized for all bijections E between all white vertices and all black vertices of Γ similar to Definition [4.1.](#page-4-5) Definition [4.4](#page-4-3) builds a graph Γ(A, B) on all point-based representations of A, B ⊂ R <sup>n</sup> and introduces the Nested Bottleneck Metric NBM(A, B) as BMD of Γ(A, B).

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

![](_page_5_Figure_3.jpeg)

Definition 4.4 (NBM : Nested Bottleneck Metric). *Let clouds* A, B ⊂ R 2 *consist of* m *unordered points. The complete bipartite graph* Γ(A, B) *has* m *white vertices (one for each* p ∈ A*) and* m *black vertices (one for each* q ∈ B*). Any edge* e *of* Γ(A, B) *has endpoints associated with pointbased representations* PR(A; p)*,* PR(B; q)*, and the* weight w(e) = PRM PR(A; p), PR(B; q) *. The* Nested Bottleneck Metric *is defined as* NBM(A, B) = BMD(Γ(A, B))*.*

Example 4.5 (4-point clouds C <sup>±</sup>). *In* <sup>R</sup> 2 *, consider the 4 point clouds* C <sup>±</sup> = {p1, p2, p3, p<sup>±</sup> 4 }*, where* p<sup>1</sup> = (4a, 0)*,* p<sup>2</sup> = (b, c)*,* p<sup>3</sup> = −p<sup>2</sup> = (−b, −c)*,* p + <sup>4</sup> = (0, 4d)*, and* p − <sup>4</sup> = (0, −4d) *for parameters* a, b, c, d ≥ 0*, see Fig. [2.](#page-2-0) Appendix [C](#page-20-0) will explicitly compute* NDP(C <sup>±</sup>)*to distinguish all clouds* C <sup>+</sup> ̸∼= C <sup>−</sup>*. Fig. [4](#page-5-1) shows the new metric* NBM *for variable parameters* a, b *and fixed* c, d*.* NBM > 0 *implies that* C <sup>+</sup> ̸∼= C <sup>−</sup>*, except in the singular cases below. If* a = 0 *or* d = 0 *or* b = c = 0*, the clouds are related by a 2-fold rotation around the origin* 0*. If* a = √ <sup>2</sup> ≈ 0.87*,* b = 0*,* c = 2*,* d = 0.5*, then* C <sup>+</sup> *consists of the vertices* (0, <sup>±</sup>2),(2√ 3, 0) *of an equilateral triangle, where* (0, 2) *is the double point* p<sup>2</sup> = p + 4 *. Then* C <sup>−</sup> *is the same equilateral triangle but its vertex* (0, −2) *is the double point* p<sup>3</sup> = p − 4 *. Because these clouds are related by rotation,* NBM = 0 *in the black pixel at* a = √ 3 <sup>2</sup> ≈ 0.87*,* b = 0 *in Fig. [4.](#page-5-1)*

Figure 4. The Nested Bottleneck Metric NBM in Definition [4.4](#page-4-3) for the clouds C <sup>±</sup> ⊂ <sup>R</sup> 2 that depend on parameters a, b and are not distinguished by 6 pairwise distances in Fig. [2,](#page-2-0) see Example [C.1.](#page-23-0)

# 5. Bi-continuity and polynomial algorithms

For a *fixed dimension* n, all algorithms for m unordered points will have polynomial times in m in the RAM model. Theorem 5.1 (Lipschitz continuity of NBM). *Let* B ⊂ R 2 *be obtained from a cloud* A ⊂ R <sup>2</sup> *by perturbing every point of* A *up to Euclidean distance* ε*. Then* NBM(A, B) ≤ 6ε*.*

To illustrate Theorem [5.1,](#page-5-2) we generated uniformly random

clouds A in the unit square and cube. To get a perturbation B of A, we shifted every point of A by adding a uniformly random value in [−ε, ε] to each coordinate, where ε ∈ [0.01, 0.1] is a noise bound. Fig. [5](#page-5-3) shows how the Nested Bottleneck Metric (NBM, averaged over several clouds) linearly increases with respect to the noise bound.

Figure 5. The metric NBM(NDP(A), NDP(B)) for a random cloud A and its ε-perturbation B increases at most linearly in the noise bound ε with a Lipschitz constant λ<sup>2</sup> < 6 as in Theorem [5.1.](#page-5-2)

![](_page_5_Figure_6.jpeg)

Theorem 5.2 (NDP time). *For any cloud* A ⊂ R <sup>2</sup> *of* m *unordered points, the Nested Distributed Projection* NDP(A) *is computed in time* O(m<sup>2</sup> ) *with space* O(m<sup>2</sup> )*.*

Theorem 5.3 (NBM time). *For any clouds* A, B ⊂ R <sup>2</sup> *of* m *unordered points, the Nested Bottleneck Metric* NBM(A, B) *is computable in time* O(m<sup>3</sup>.<sup>5</sup> log m) *with space* O(m<sup>3</sup> )*.*

![](_page_5_Figure_9.jpeg)

Figure 6. Times (microseconds, log scale) of metrics on invariants.

Fig. [6](#page-5-4) illustrates a polynomial dependence of the NBM time in Theorem [5.3.](#page-5-0) Theorem [5.4](#page-6-0) says that any m-point clouds A, B ⊂ R can be matched up to a perturbation proportional

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

to the Nested Bottleneck Metric d = NBM. If d is small, all points of A, B can be matched up to a perturbation 3 √ 2d by rigid motion. In section [6,](#page-6-1) the experimental maximum of this approximate factor is 2.2 < 3 √ 2.

Theorem 5.4 (point matching). *For any* m*-point clouds* A, B ⊂ R 2 *, one can find in time* O(m<sup>3</sup>.<sup>5</sup> log m) *a rigid motion* f *of* R <sup>2</sup> *and a bijection* β : A → B *such that the match distance* max q∈A |f(q)−β(q)| ≤ 3 √ 2NBM(A, B)*, see the comparison of this distance with others in Fig. [5.](#page-5-3)*

By Theorem [5.1,](#page-5-2) perturbing every atom up to ε (due to the ever-present thermal vibrations) changes NDP up to 6ε in the metric NBM. Conversely, by Theorem [5.4,](#page-6-0) if NBM(A, B) = δ > 0 is small, the clouds A, B can be approximately matched by rigid motion up to 3 √ 2δ pointwise.

If clouds A, B ⊂ R <sup>n</sup> have ordered points, one can *morph* (continuously transform) A to B by moving every i-th point of A along a straight-line to the i-th point of B for i = 1, . . . , m. If m points are unordered, there are m! potential transformations, one for each permutation of m points.

Associating every point p ∈ A to its nearest neighbor q ∈ B is justified only for fixed clouds because a rigid motion of A can change a nearest neighbor of any point p ∈ A in B.

### 6. Experiments on large molecular databases

The big databases of molecules with *3D conformers* (embeddings in R 3 ) are QM9 (130K+ entries) [\(Ramakrishnan](#page-10-17) [et al.,](#page-10-17) [2014\)](#page-10-17) and GD (GEOM drugs, 31M+ entries) containing hundreds of 3D conformers of *unordered* atoms for each of 61607 chemical compositions [\(Axelrod & Gomez-](#page-8-19)[Bombarelli,](#page-8-19) [2022\)](#page-8-19). The Protein Data Bank has backbones of *ordered* atoms classified by simpler invariants [\(Anosova](#page-8-20) [et al.,](#page-8-20) [2025\)](#page-8-20). All experiments took a few hours on Ryzen 9 3950X 3.5 GHz, 64 MB of L3 cache, RAM 82GB.

The ICML guide for application-driven ML says that "novel ideas that are simple to apply may be especially valuable", so we start with simpler and much faster invariants below.

Definition 6.1 (invariants SRV, SDV,PDD). *Let* A ⊂ R n *be a cloud of* m *unordered points with the center of mass at* 0 ∈ R <sup>n</sup>*. The* Sorted Radial Vector SRV(A) *has* m *radial distances* |p| *in decreasing order for all* p ∈ A*. The* Sorted Distance Vector SDV(A) *is the vector of* <sup>m</sup>(m−1) 2 *pairwise distances* |p − q| *in decreasing order for distinct* p, q ∈ A*. For any point* p ∈ A*, let* d1(p) ≤ · · · ≤ dm−1(p) *be Euclidean distances from* p *to all other points* q ∈ A \ {p} *in increasing order. These distance lists become rows of the* m × (m − 1) *matrix* D(S; k)*. Any* l > 1 *identical rows are collapsed into a single row with the* weight l/m*. The final matrix with at most* m *unordered weighted rows and* m − 1 *ordered columns is the* Pointwise Distance Distribution*.*

For a PDD on m points, we sort m distance lists in time O(m<sup>2</sup> log m). Then PDDs are compared by the Earth Mover's Distance EMD [\(Rubner et al.,](#page-10-18) [2000\)](#page-10-18) in time O(m<sup>3</sup> ). Table [2](#page-6-2) emphasizes that most clouds should be first distinguished by simpler and faster invariants SRV, SDV,PDD. The complete NDP is needed only in rare cases but is still essential because any incomplete invariant I has no chance to predict different properties on *false positives* that are molecules A ̸∼= B with I(A) = I(B).

Table 2. Invariants and metrics on cloud A ⊂ R <sup>2</sup> with m unordered points: from the fastest (linear-time) to complete.

| INVARIANT | TIME |     | METRIC TIME         |
|-----------|------|-----|---------------------|
| SRV       | O    | ( m | log m ) L ∞ O ( m ) |
| SDV       | O    | ( m | 2                   |
|           |      |     | ) L ∞ O ( m 2       |
| PDD       | O    | ( m | 2                   |
|           |      |     | log m ) EMD O ( m 3 |
| NDP       | O    | ( m | 2                   |
|           |      |     | ) NBM O ( m 3 5     |
|           |      |     | log m )             |

For a fixed atom p ∈ A and k < m, the first k distances to neighbors in the row of p in PDD(A) is an atomwise version of SRV(A). This vector D(A, p; k) of k distances was the only input for predicting the chemical element of p. A default network in TensorFlow was trained on clouds with the 80/20 split and achieved 98% accuracy for k = 4 in Table [4](#page-6-3) despite the unbalanced counts of frequent elements in Table [3.](#page-6-4) Appendix [A](#page-12-0) has all implementation details.

Table 3. Counts of atoms by chemical elements in QM9 (2,407,753 atoms), GD0 (GEOM drugs 0th conformers, 12,917,980 atoms).

| QM9:      | H QM9:    | C QM9:  | N QM9:  | O QM9: F  |
|-----------|-----------|---------|---------|-----------|
| 1,230,122 | 846,557   | 139,764 | 187,996 | 3,314     |
| GD0: H    | GD0: C    | GD0:    | N GD0:  | O GD0: F  |
| 5,660,986 | 5,267,096 | 842,562 | 854,400 | 64,299    |
| GD0: P    | GD0: S    | GD0:    | Cl GD0: | Br GD0: I |
| 1,350     | 159,648   | 53,404  | 14,010  | 225       |

Table 4. Accuracies in percentages for predicting chemical elements by a 4-layer network using *only Euclidean distances* from an atomic center to its k nearest neighbors for QM9 and GD0.

| data | <i>k</i> = 2 | <i>k</i> = 3 | <i>k</i> = 4 | <i>k</i> = 5 | <i>k</i> = 6 |
|------|--------------|--------------|--------------|--------------|--------------|
| QM9  | 94.63        | 98.64        | 98.24        | 98.54        | 98.77        |
| GD0  | 91.44        | 96.67        | 98.05        | 98.70        | 98.49        |

All past attempts by both ML and non-ML in chemistry achieved only 86% on similar size data, see Table [7](#page-14-0) summarized in [\(Vasylenko et al.,](#page-11-16) [2025\)](#page-11-16), because the underlying descriptors were not invariant, e.g. under permutations of atoms, which creates exponentially many representations of the same molecule, incomplete, or their similarities failed the triangle axiom, e.g. see [\(Steck et al.,](#page-11-17) [2024\)](#page-11-17).

394

396

![](_page_7_Figure_1.jpeg)

High accuracies of D(A, p; 4) in Table [4](#page-6-3) are explained by the following cascade computations. First, split all clouds from Table [3](#page-6-4) by the 1st distance (to the nearest neighbor of a central atom p) rounded to 3 decimal places in A˚ . This is a typical experimental precision, where 1A˚ = 10−<sup>10</sup>m is the smallest interatomic distance. Second, split each subset with equal 1st distances by 2nd distances, and so on up to k = 5 distances. All clouds of different elements in QM9 and GD0 were separated by D(A, p; 4) and D(A, p; 5), respectively.

We compared full molecules starting with the pseudo-metric L<sup>∞</sup> (max abs difference of corresponding coordinates) on SRVs of all 873,527,974 pairs of 3D atomic clouds having equal numbers of atoms in QM9, then 8,735,279 distances L<sup>∞</sup> on SDVs of the 1% closest pairs, 87,352 EMDs on PDDs of the 1% closest pairs, and NBMs on NDPs for the final 10K closest pairs. In this hierarchical computation, large values of L<sup>∞</sup> (then EMD) guarantee that molecules are distant and cannot be closely matched by rigid motion. Tiny or zero values of pseudo-metrics guarantee nothing because SDV and PDD can coincide for very different clouds, see Fig. [2,](#page-2-0) Fig. S4 in [\(Pozdnyakov et al.,](#page-10-19) [2020\)](#page-10-19).

Table 5. Chemically different molecules (given by QM9 ids) are geometrically distinguished by SRV, SDV,PDD, NDP, see Fig. [8.](#page-7-0)

### smallest distances in A, molecule A ˚ ̸= molecule B

SRV, L<sup>∞</sup> = 0.021, H4C5N2O(5365)̸=H3C4N3O<sup>2</sup> (131923) SDV, L<sup>∞</sup> = 0.055, H3C4N<sup>5</sup> (123533)̸=H3C5N3O(24547) EMD = 0.051, H3C4N<sup>5</sup> (123533)̸=H3C5N3O(24521) NBM = 0.148, H3C4N3O<sup>2</sup> (28141)̸=H3C3N5O(130099)

Fig. [7](#page-7-1) compares the new metric y = NBM on complete NDPs with the pseudo-metric x = PDD. All pairs A, B with (x, y) close to the vertical axis in Fig. [7](#page-7-1) (left) have EMD ≈ 0 because they are almost mirror images (indistinguishable by PDD) well distinguished by higher values of NBM. Fig. [8](#page-7-0) shows bonds by standard visualization, they were not used for clouds of points without any edges.

For each of 31M+ entries (3D conformers) in the much larger database GD, we took the cloud A of all atoms without chemical elements and computed SRV(A; k) of up to k = 10 largest distances (rounded to 3 decimal places) from the center of mass of A to all atoms. Similar to QM9, cascade comparisons confirmed that SRV(A; 7) distinguishes all chemically different molecules, while only four pairs have equal SRV(A; 6) rounded to 3 decimal places. This transparent reconstruction of a full chemical composition from precise enough geometry gives hope to explain other molecular properties in terms of geometric invariants.

Figure 7. x = EMD(PDD(A),PDD(B)) vs y = NBM(A, B) on complete invariants NDP with zoomed-in comparisons on the right, which all appear only for chemically identical molecules.

![](_page_7_Figure_5.jpeg)

Figure 8. Left: chemically different QM9 molecules 28141 and 130099 have the smallest distances NBM ≈ 0.15A˚ . Right: molecules 70954 and 74130 are almost mirror images with EMD ≈ 0.0004A˚ but are well distinguished by NBM ≈ 1.619A˚ .

### 7. Discussion: conclusions and limitations

For clouds with different numbers of points, we can replace the bottleneck distance BD in Definition [4.2](#page-4-1) with any metric between fixed clouds of different sizes, e.g. the Hausdorff distance, to get a metric on PRs. Then we can compare NDPs of any clouds as weighted distributions by EMD. The limitation is the proof of Theorem [5.4](#page-6-0) in dimension n = 2, though the experiments indicate the Lipschitz continuity of NDP−<sup>1</sup> in R 3 . All other conditions in Problem [1.1](#page-0-1) are proved in the appendices for any dimension n ≥ 2.

The experiments imply that mapping any molecule to (the rigid class of) its cloud of atomic centers is *injective* without losing any chemical information, so all chemical elements can be reconstructed from pure geometry. This result confirms our physical intuition that replacing atoms should perturb geometry at least slightly, which was impossible to establish without complete and Lipschitz continuous invariants. Hence all molecules of m atoms live at different locations in the common *Cloud Rigid Space* CRS(<sup>R</sup> 3 ; m) of SE(3)-classes of all clouds of m unordered points.

Most significantly, a *molecular structure* can now be defined not as a huge collection of vectors under rotations and atom permutations, see Fig. 1 in [\(Lang et al.,](#page-10-20) [2024\)](#page-10-20), but as a rigid (class of a) cloud of atomic centers (without chemical elements), which is uniquely determined by an efficient hierarchy of invariants from the fastest (linear-time) SRV to the new complete invariant NDP solving Problem [1.1.](#page-0-1)

- Impact Statement This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. References Akhtar, N. and Mian, A. Threat of adversarial attacks on deep learning in computer vision: A survey. *IEEE Access*, 6:14410–14430, 2018. Anosova, O., Gorelov, A., Jeffcott, W., Jiang, Z., and Kurlin,
- V. A complete and bi-continuous invariant of protein backbones under rigid motion. *MATCH Communications in Mathematical and in Computer Chemistry (to appear), arxiv:2410.08203*, 2025. Antunes, L. M., Grau-Crespo, R., and Butler, K. T. Distributed representations of atoms and materials for machine learning. *npj Computational Materials*, 8(1):44, 2022. Aronsson, J. Homogeneous vector bundles and gequivariant convolutional neural networks. *Sampling Theory, Signal Processing, and Data Analysis*, 20(2):10, 2022. Assaad, S., Downey, C., Al-Rfou, R., Nayakanti, N., and Sapp, B. Vn-transformer: Rotation-equivariant attention for vector neurons. *Transactions on Machine Learning Research*, 2023. Atz, K., Grisoni, F., and Schneider, G. Geometric deep learning on molecular representations. *Nature Machine Intelligence*, 3(12):1023–1032, 2021. Axelrod, S. and Gomez-Bombarelli, R. Geom, energyannotated molecular conformations for property prediction and molecular generation. *Scientific Data*, 9(1):185, 2022. Boutin, M. and Kemper, G. On reconstructing n-point configurations from the distribution of distances or areas. *Adv. Appl. Math.*, 32(4):709–735, 2004. Brass, P. and Knauer, C. Testing the congruence of ddimensional point sets. In *SoCG*, pp. 310–314, 2000. Brass, P. and Knauer, C. Testing congruence and symmetry for general 3-dimensional objects. *Computational Geometry*, 27(1):3–11, 2004. Brecheteau, C. A statistical test of isomorphism between ´ metric-measure spaces using the distance-to-a-measure signature. pp. 795–849, 2019. Brink, C., Kahl, W., and Schmidt, G. *Relational methods in computer science*. Springer Science & Business Media, 1997. Bronstein, M. M., Bruna, J., Cohen, T., and Velickovi ˇ c,´
  - P. Geometric deep learning: grids, groups, graphs, geodesics, and gauges. *arXiv:2104.13478*, 2021. Cahill, J., Iverson, J. W., and Mixon, D. G. Towards a bilipschitz invariant theory. *Applied and Computational Harmonic Analysis*, 72:101669, 2024. Chauvin, L., Wells III, W., and Toews, M. Registering image volumes using 3D SIFT and discrete SP-symmetry. *arXiv:2205.15456*, 2022. Chen, C., Ye, W., Zuo, Y., Zheng, C., and Ong, S. P. Graph networks as a universal machine learning framework for molecules and crystals. *Chemistry of Materials*, 31(9): 3564–3572, 2019. Chen, H., Liu, S., Chen, W., Li, H., and Hill, R. Equivariant point network for 3D point cloud analysis. In *Computer Vision and Pattern Recognition*, pp. 14514–14523, 2021. Chew, P. and Kedem, K. Improvements on geometric pattern matching problems. In *Scandinavian Workshop on Algorithm Theory*, pp. 318–325, 1992. Chew, P., Goodrich, M., Huttenlocher, D., Kedem, K., Kleinberg, J., and Kravets, D. Geometric pattern matching under Euclidean motion. *Computational Geometry*, 7 (1-2):113–124, 1997. Chew, P., Dor, D., Efrat, A., and Kedem, K. Geometric pattern matching in d-dimensional space. *Discrete & Computational Geometry*, 21(2):257–274, 1999. Cohen, T. S., Geiger, M., and Weiler, M. A general theory of equivariant cnns on homogeneous spaces. *Advances in Neural Information Processing Systems*, 32, 2019. Colbrook, M. J., Antun, V., and Hansen, A. C. The difficulty of computing stable and accurate neural networks: On the barriers of deep learning and Smale's 18th problem. *Proc. National Academy of Sciences*, 119(12):e2107151119, 2022. Cosmo, L., Panine, M., Rampini, A., Ovsjanikov, M., Bronstein, M. M., and Rodola, E. Isospectralization, or how to hear shape, style, and correspondence. In *Proceedings of CVPR*, pp. 7529–7538, 2019. Dekster, B. V. and Wilker, J. B. Edge lengths guaranteed to form a simplex. *Archiv der Mathematik*, 49(4):351–366, 1987.

- 495 496 497 498 499 500 504 506 508 509 511 514 515 516 518 524 526 528 531 534 536 538 540 541 542 543 544 545 546 547 548 Delle Rose, V., Kozachinskiy, A., Rojas, C., Petrache, M., and Barcelo, P. Three iterations of (d- 1)-wl test dis- ´ tinguish non isometric clouds of d-dimensional points. *Advances in Neural Information Processing Systems*, 36, 2024. Deng, C., Litany, O., Duan, Y., Poulenard, A., Tagliasacchi, A., and Guibas, L. J. Vector neurons: A general framework for so(3)-equivariant networks. In *Proceedings of the International Conference on Computer Vision*, pp. 12200–12209, 2021. Deza, E. and Deza, M. M. *Encyclopedia of distances*. Springer, 2009. Dong, Y., Liao, F., Pang, T., Su, H., Zhu, J., Hu, X., and Li, J. Boosting adversarial attacks with momentum. In *Computer vision and pattern recognition*, pp. 9185–9193, 2018. Duxbury, P. M., Granlund, L., Gujarathi, S., Juhas, P., and Billinge, S. J. The unassigned distance geometry problem. *Discrete Applied Mathematics*, 204:117–132, 2016. Dym, N. and Gortler, S. J. Low-dimensional invariant embeddings for universal geometric learning. *Foundations of Computational Mathematics*, pp. 1–41, 2024. Efrat, A., Itai, A., and Katz, M. J. Geometry helps in bottleneck matching and related problems. *Algorithmica*, 31(1):1–28, 2001. Feynman, R. *The Feynman lectures on physics. Chapter 1: atoms in motion*, volume 1. 1971. Fuchs, F., Worrall, D., Fischer, V., and Welling, M. Se(3) transformers: 3d roto-translation equivariant attention networks. *Advances in neural information processing systems*, 33:1970–1981, 2020. Goodrich, M. T., Mitchell, J. S., and Orletsky, M. W. Approximate geometric pattern matching under rigid motions. *Transactions on Pattern Analysis and Machine Intelligence*, 21(4):371–379, 1999. Gordon, C., Webb, D., and Wolpert, S. Isospectral plane domains and surfaces via riemannian orbifolds. *Inventiones mathematicae*, 110(1):1–22, 1992a. Gordon, C., Webb, D. L., and Wolpert, S. One cannot hear the shape of a drum. *Bulletin of the American Mathematical Society*, 27(1):134–138, 1992b. Goyal, A., Law, H., Liu, B., Newell, A., and Deng, J. Revisiting point cloud shape classification with a simple and effective baseline. In *International Conference on Machine Learning*, pp. 3809–3820, 2021. Guo, C., Gardner, J., You, Y., Wilson, A. G., and Weinberger,
  - K. Simple black-box adversarial attacks. In *International Conference on Machine Learning*, pp. 2484–2493, 2019. Hausdorff, F. Dimension und au¨ βeres maβ. *Mathematische Annalen*, 79(2):157–179, 1919. Hopcroft, J. E. and Karp, R. M. An nˆ5/2 algorithm for maximum matchings in bipartite graphs. *SIAM Journal on Computing*, 2(4):225–231, 1973. Hordan, S., Amir, T., Gortler, S. J., and Dym, N. Complete neural networks for euclidean graphs. In *AAAI Conference on Artificial Intelligence*, volume 38 (11), pp. 12482–12490, 2024. Horn, R. A. and Johnson, C. R. *Matrix analysis*. Cambridge University Press, 2012. Huang, Q.-X., Flory, S., Gelfand, N., Hofer, M., and ¨ Pottmann, H. Reassembling fractured objects by geometric matching. In *ACM SIGGRAPH*, pp. 569–578. 2006. Huttenlocher, D. P., Klanderman, G. A., and Rucklidge, W. J. Comparing images using the Hausdorff distance. *Transactions on pattern analysis and machine intelligence*, 15 (9):850–863, 1993. Kac, M. Can one hear the shape of a drum? *The american mathematical monthly*, 73(4P2):1–23, 1966. Kapovich, M. and Millson, J. J. The symplectic geometry of polygons in euclidean space. *Journal of Differential Geometry*, 44(3):479–513, 1996. Kendall, D. G., Barden, D., Carne, T. K., and Le, H. *Shape and shape theory*. John Wiley & Sons, 2009. Keriven, N. and Peyre, G. Universal invariant and equivari- ´ ant graph neural networks. *Advances in Neural Information Processing Systems*, 32, 2019. Kondor, R. and Trivedi, S. On the generalization of equivariance and convolution in neural networks to the action of compact groups. In *International Conference on Machine Learning*, pp. 2747–2755, 2018. Kruskal, J. B. and Wish, M. *Multidimensional scaling*. Number 11. Sage, 1978. Kurlin, V. Polynomial-time algorithms for continuous metrics on atomic clouds of unordered points. *MATCH Communications in Mathematical and in Computer Chemistry*, 91:79–108, 2024. Laidlaw, C. and Feizi, S. Functional adversarial attacks. *Adv. Neural Information Proc. Systems*, 32, 2019.

- 551 554 556 558 560 564 566 568 571 574 576 578 580 581 582 583 584 585 586 587 588 589 590 594 596 598 600 601 602 603 Lang, L., Cezar, H. M., Adamowicz, L., and Pedersen, T. B. Quantum definition of molecular structure. *Journal of the American Chemical Society*, 146(3):1760–1764, 2024. Leman, A. and Weisfeiler, B. A reduction of a graph to a canonical form and an algebra arising during this reduction. *Nauchno-Technicheskaya Informatsiya*, 2(9):12–16, 1968. Li, X., Li, R., Chen, G., Fu, C.-W., Cohen-Or, D., and Heng, P.-A. A rotation-invariant framework for deep point cloud analysis. *IEEE transactions on visualization and computer graphics*, 28(12):4503–4514, 2021. Liberti, L. and Lavor, C. *Euclidean distance geometry*. Springer, 2017. Lin, Z. C., Lee, H., and Huang, T. S. Finding 3d point correspondences in transformation estimation. In *Proceedings-International Conference on Pattern Recognition*, pp. 303–
- 305. IEEE, 1986. Lowe, D. G. Object recognition from local scale-invariant features. In *Proceedings of ICCV*, volume 2, pp. 1150– 1157, 1999. Lowe, D. G. Distinctive image features from scale-invariant keypoints. *International journal of computer vision*, 60: 91–110, 2004. Maennel, H., Unke, O. T., and Muller, K.-R. Complete and efficient covariants for three-dimensional point configurations with application to learning molecular quantum properties. *The Journal of Physical Chemistry Letters*, 15:12513–12519, 2024. Majhi, S., Vitter, J., and Wenk, C. Approximating gromovhausdorff distance in euclidean space. *Computational Geometry*, 116:102034, 2024. Marin, R., Rampini, A., Castellani, U., Rodola, E., Ovs- ` janikov, M., and Melzi, S. Spectral shape recovery and analysis via data-driven connections. *International journal of computer vision*, 129:2745–2760, 2021. Maron, H., Fetaya, E., Segol, N., and Lipman, Y. On the universality of invariant networks. In *International conference on machine learning*, pp. 4363–4371, 2019. Memoli, F. Gromov–Wasserstein distances and the metric ´ approach to object matching. *Foundations of computational mathematics*, 11:417–487, 2011. Merchant, A., Batzner, S., Schoenholz, S. S., Aykol, M., Cheon, G., and Cubuk, E. D. Scaling deep learning for materials discovery. *Nature*, 624(7990):80–85, 2023. Morissette, S. L., Soukasene, S., Levinson, D., Cima, M. J., and Almarsson, O. Elucidation of crystal form diversity ¨ of the hiv protease inhibitor ritonavir by high-throughput crystallization. *Proceedings of the National Academy of Sciences*, 100(5):2180–2184, 2003. Morris, C., Dym, N., Maron, H., Ceylan, ˙I. ˙I., Frasca, F., Levie, R., Lim, D., Bronstein, M., Grohe, M., and Jegelka,
  - S. Future directions in foundations of graph machine learning. *arXiv:2402.02287*, 2024. Nemec, L. Principal component analysis (pca): A physically intuitive mathematical introduction. [https://towardsdatascience.com/](https://towardsdatascience.com/principal-component-analysis-pca-8133b02f11bd) [principal-component-analysis-pca-8133b02f11bd](https://towardsdatascience.com/principal-component-analysis-pca-8133b02f11bd), 2022. Nigam, J., Pozdnyakov, S. N., Huguenin-Dumittan, K. K., and Ceriotti, M. Completeness of atomic structure representations. *APL Machine Learning*, 2(1), 2024. Oliynyk, A. O., Antono, E., Sparks, T. D., Ghadbeigi, L., Gaultois, M. W., Meredig, B., and Mar, A. Highthroughput machine-learning-driven synthesis of fullheusler compounds. *Chemistry of Materials*, 28(20): 7324–7331, 2016. Pozdnyakov, S. N. and Ceriotti, M. Incompleteness of graph convolutional neural networks for points clouds in three dimensions. *arXiv:2201.07136*, 2022. Pozdnyakov, S. N., Willatt, M. J., Bartok, A. P., Ortner, C., ´ Csanyi, G., and Ceriotti, M. Incompleteness of atomic ´ structure representations. *Phys. Rev. Lett.*, 125:166001, 2020. URL <arXiv:2001.11696>. Ramakrishnan, R., Dral, P. O., Rupp, M., and Von Lilienfeld,
  - O. A. Quantum chemistry structures and properties of 134 kilo molecules. *Scientific data*, 1(1):1–7, 2014. Rass, S., Konig, S., Ahmad, S., and Goman, M. Metricizing ¨ the euclidean space towards desired distance relations in point clouds. *IEEE Transactions on Information Forensics and Security*, 19:7304–7319, 2024. Reuter, M., Wolter, F.-E., and Peinecke, N. Laplace– beltrami spectra as 'shape-dna'of surfaces and solids. *Computer-Aided Design*, 38(4):342–366, 2006. Rubner, Y., Tomasi, C., and Guibas, L. The Earth Mover's Distance as a metric for image retrieval. *International Journal of Computer Vision*, 40(2):99–121, 2000. Rudin, W. et al. *Principles of mathematical analysis*, volume 3. McGraw-hill New York, 1976. Satorras, V. G., Hoogeboom, E., and Welling, M. E(n) equivariant graph neural networks. In *International conference on machine learning*, pp. 9323–9332, 2021.

- 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 Schmiedl, F. Computational aspects of the Gromov– Hausdorff distance and its application in non-rigid shape matching. *Discrete Comp. Geometry*, 57:854–880, 2017. Schoenberg, I. Remarks to Maurice Frechet's article "Sur la definition axiomatique d'une classe d'espace distances vectoriellement applicable sur l'espace de Hilbert. *Annals of Mathematics*, pp. 724–732, 1935. Shi, J., Yang, H., and Carlone, L. Robin: a graph-theoretic approach to reject outliers in robust estimation using invariants. In *International Conference on Robotics and Automation (ICRA)*, pp. 13820–13827, 2021. Spezialetti, R., Salti, S., and Stefano, L. D. Learning an effective equivariant 3d descriptor without supervision. In *ICCV*, pp. 6401–6410, 2019. Steck, H., Ekanadham, C., and Kallus, N. Is cosinesimilarity of embeddings really about similarity? In *Companion Proceedings of the ACM on Web Conference 2024*, pp. 887–890, 2024. Su, Z., Welling, M., Pietikainen, M., and Liu, L. Svnet: ¨ Where SO(3) equivariance meets binarization on point cloud representation. In *International Conference on 3D Vision*, pp. 547–556, 2022. Thomas, N., Smidt, T., Kearnes, S., Yang, L., Li, L., Kohlhoff, K., and Riley, P. Tensor field networks: Rotation-and translation-equivariant neural networks for 3d point clouds. *arXiv:1802.08219*, 2018. Tshitoyan, V., Dagdelen, J., Weston, L., Dunn, A., Rong, Z., Kononova, O., Persson, K. A., Ceder, G., and Jain, A. Unsupervised word embeddings capture latent knowledge from materials science literature. *Nature*, 571(7763):95– 98, 2019. Turner, K., Mukherjee, S., and Boyer, D. M. Persistent homology transform for modeling shapes and surfaces. *Information and Inference: A Journal of the IMA*, 3(4): 310–344, 2014. Vasylenko, A., Antypov, D., Schewe, S., Daniels, L. M., Claridge, J. B., Dyer, M. S., and Rosseinsky, M. J. Digital features of chemical elements extracted from local geometries in crystal structures. *Digital Discovery*, 2025. Villar, S., Hogg, D. W., Storey-Fisher, K., Yao, W., and Blum-Smith, B. Scalars are universal: equivariant machine learning, structured like classical physics. *Advances in Neural Information Processing Systems*, 34:28848– 28863, 2021. Wang, Y. and Solomon, J. M. Deep closest point: Learning representations for point cloud registration. In *Proceedings of the IEEE/CVF international conference on computer vision*, pp. 3523–3532, 2019. Ward, L., Agrawal, A., Choudhary, A., and Wolverton, C. A general-purpose machine learning framework for predicting properties of inorganic materials. *npj Computational Materials*, 2(1):1–7, 2016. Weston, L., Tshitoyan, V., Dagdelen, J., Kononova, O., Trewartha, A., Persson, K. A., Ceder, G., and Jain, A. Named entity recognition and normalization applied to large-scale information extraction from the materials science literature. *Journal of chemical information and modeling*, 59(9):3692–3702, 2019. Weyl, H. *The classical groups: their invariants and representations*. Number 1. Princeton university press, 1946. Widdowson, D. and Kurlin, V. Resolving the data ambiguity for periodic crystals. *Advances in Neural Information Processing Systems*, 35:24625–24638, 2022. Widdowson, D. E. and Kurlin, V. A. Recognizing rigid patterns of unlabeled point clouds by complete and continuous isometry invariants with no false negatives and no false positives. In *Computer Vision and Pattern Recognition*, pp. 1275–1284, 2023. Xu, Y., Lei, J., Dobriban, E., and Daniilidis, K. Unified fourier-based kernel and nonlinearity design for equivariant networks on homogeneous spaces. In *International Conference on Machine Learning*, pp. 24596–24614, 2022. Yang, H., Shi, J., and Carlone, L. Teaser: Fast and certifiable point cloud registration. *IEEE Transactions on Robotics*, 37(2):314–333, 2020. Yarotsky, D. Universal approximations of invariant maps by neural networks. *Constructive Approximation*, 55(1): 407–474, 2022. Zhou, Q., Tang, P., Liu, S., Pan, J., Yan, Q., and Zhang, S.-
  - C. Learning atoms for materials discovery. *Proceedings of the National Academy of Sciences*, 115(28):E6411– E6417, 2018. Zhu, W., Chen, L., Hou, B., Li, W., Chen, T., and Liang,
    - S. Point cloud registration of arrester based on scaleinvariant points feature histogram. *Scientific Reports*, 12 (1):1–13, 2022.

677 678 679 The completeness and bi-Lipschitz continuity of the proposed invariants enabled the new experiments on 130K+ real molecules in section [6,](#page-6-1) which were not previously possible because all past invariants did not satisfy all conditions of Problem [1.1,](#page-0-1) especially the realizability condition that provides geographic-style maps on cloud spaces.

689 690

694

696

698

700

704

706

708 709

711

### Introduction to appendices

The main contribution is the roadmap for any data challenge through well-motivated Problem [1.1,](#page-0-1) where clouds and rigid motion can be replaced with any objects and equivalences. The conditions of completeness and Lipschitz continuity of an invariant I cover the *discriminative* challenge. After these conditions [1.1\(](#page-0-1)a,b,e) are satisfied, the invariant I can be inverted in principle and opens the *generative challenge* of its realizability and inverse continuity in [1.1\(](#page-0-1)c,d,e).

Problem [1.1](#page-0-1) was stated for unordered clouds under rigid motion but was also solved for *isometry* and compositions of these equivalences with uniform scaling in R <sup>n</sup>. For m = 4 points, plane quadrilaterals were previously classified in discrete classes in Fig. [1](#page-0-0) (right), while appendix [C](#page-20-0) shows the first continuous maps of the invariant space CRS(<sup>R</sup> 2 ; 4). Conditions [1.1\(](#page-0-1)d,e,f) enable a generation of real clouds in CRS(<sup>R</sup> <sup>n</sup>; m) from their invariants. A full answer to the question 'same or different, and how much different' required complete invariants with Lipschitz continuous metrics.

The key contribution is a theoretically justified solution to Problem [1.1.](#page-0-1) The experiments on the databases QM9 and GEOM drugs are considered complementary. Example [C.1](#page-23-0) and its extension in Example [C.2](#page-24-0) prove that infinitely many pairs of non-isometric clouds C <sup>+</sup> ̸∼= C <sup>−</sup> (depending on 4 free parameters and having the same 6 pairwise distances) are distinguished by the new invariants. This result is impossible to justify by any finite experiment. Example [C.1](#page-23-0) demonstrated the non-zero distances between the complete invariants of C <sup>±</sup> in Fig. ??.

The full solution to Problem [1.1](#page-0-1) for n = 2 is justified by Theorem [3.5](#page-4-4) and Lemmas [3.3,](#page-3-2) [5.1,](#page-5-2) [5.2,](#page-5-5) [5.3.](#page-5-0) Theorem [3.3](#page-3-2) enables a visualization of cloud spaces, which were unknown even for m = 4 unordered points in R 2 .

- The *Cloud Isometry Space* CIS(<sup>R</sup> <sup>n</sup>; m) of clouds of m unordered points under isometry in <sup>R</sup>
- n.
- The *Cloud Rigid Space* CRS(<sup>R</sup> <sup>n</sup>; m) of clouds of m unordered points under rigid motion in <sup>R</sup>
- n.
- The *Cloud Similarity Space* CSS(<sup>R</sup> <sup>n</sup>; m) of clouds of m unordered points under *geometric similarity*, which is a composition of isometry and uniform scaling in R
  - n.
- The *Cloud Dilation Space* DCS(<sup>R</sup> <sup>n</sup>; m) of clouds of m unordered points under orientation-preserving geometric similarity (rigid motion and uniform scaling) in R
  - n.

Here is a summary of the supplementary materials.

- Appendix [A](#page-12-0) extends section [6](#page-6-1) with more details of new invariants and metrics computed on the QM9 database and compared with past pseudo-metrics.
- Appendix [C](#page-20-0) discusses parametrization of CSS(<sup>R</sup> 2 ; m) and includes Examples [C.1](#page-23-0) and [C.2](#page-24-0) computing the new invariants NDP in detail for infinitely many 4-point clouds from Example [C.1.](#page-23-0)
- Appendices [B,](#page-14-1) [D,](#page-29-0) [E](#page-31-0) prove all theoretical results from sections [3,](#page-3-4) [4,](#page-4-6) [5,](#page-5-6) respectively.
- The zip folder with supplementary materials includes the code for computing all invariants and metrics as well as tables with all coordinates of colorful maps of QM9 and distances.

# A. Extra details of experiments in section [6](#page-6-1)

The default 4-layer network from TensorFlow used a "sequential" mode, 3 epochs, and the settings in Table [6.](#page-13-0)

The only difference between QM9 and GD settings was in the number N of chemical elements in tf.keras.layers.Dense(N), where N = 5 for QM9 and N = 10 for GD.

The maps of QM9 in Fig. [9](#page-13-1) are based on eigenvalues and too dense without clear separation. Even if we zoom in, these incomplete invariants will not separate molecules because 3D clouds have at most 3 eigenvalues. The complete invariants

Table 6. Parameters of the default 4-layer network for predictions in Table [4.](#page-6-3)

| L AYER ( TYPE )     | O UTPUT S HAPE | NUMBER OF PARAMETERS |
|---------------------|----------------|----------------------|
| DENSE (D ENSE )     | (N ONE , 32)   | 352                  |
| BATCH NORMALIZATION | (N ONE , 32)   | 128                  |
| RE LU (R E LU)      | (N ONE , 32)   | 0                    |
| DENSE 1 (D ENSE )   | (N ONE , 5)    | 165                  |

![](_page_13_Figure_4.jpeg)

Figure 9. Left: each dot represents one QM9 molecule whose atomic cloud has two largest roots l<sup>1</sup> ≥ l<sup>2</sup> of eigenvalues (moments of inertia [\(Nemec,](#page-10-21) [2022\)](#page-10-21) or elongations in principal directions) in Angstroms (1A˚ = 10−<sup>10</sup>m ≈ smallest interatomic distance). The color represents the free energy G characterizing molecular stability. Right: each dot represents one QM9 molecule whose atomic cloud has coordinates x, y expressed via the roots l<sup>1</sup> ≥ l<sup>2</sup> ≥ l<sup>3</sup> ≥ 0 of three eigenvalues.

![](_page_13_Figure_6.jpeg)

Figure 10. Left: each dot is a comparison of closest atomic clouds A, B from QM9 by the distances L<sup>∞</sup> on SRV vs L<sup>∞</sup> on SDV. Right: zoomed-in comparisons for very small distances.

NDP contain much more geometric information. Fig. [10](#page-13-2) and [11](#page-14-2) show that distances on stornger invariants have larger values and hence better separate molecules, though all these distances have the same Lipschitz constant 2.

Fig. [12](#page-14-3) (left) shows the simplest projections of the atomic clouds from QM9, see the familiar molecules such as H2O (water). Any small region on such a map can be zoomed in and displayed in other invariants from Table [2,](#page-6-2) see Fig. [12](#page-14-3) (right).

*784*

![](_page_14_Figure_3.jpeg)

*804*

*806*

*814*

![](_page_14_Figure_1.jpeg)

Figure 11. Left: each dot is a comparison of closest atomic clouds A, B from QM9 by the distances L<sup>∞</sup> on SDV vs EMD on PDD. Right: zoomed-in comparisons for very small distances.

Figure 12. QM9 maps: each dot colored by the free energy G represents an atomic cloud. Left: x = SRV1, y = SRV<sup>1</sup> − SRV2. Right: all molecules with SRV<sup>1</sup> = SRV<sup>2</sup> (two equidistant atoms from the center of mass) are projected to x = SRV2, y = SRV<sup>2</sup> − SRV3.

Table 7. Past ML and non-ML predictions of chemical elements have lower accuracies than by distance invariants in Table [4.](#page-6-3)

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

# B. Generalization of section [3](#page-3-4) and all proofs in dimensions n ≥ 2

This appendix extends all concepts from section [3](#page-3-4) to dimensions n ≥ 2, extends Theorem [3.3](#page-3-2) to Theorem [B.7,](#page-18-0) which is proved with Theorem [B.9](#page-19-0) for any n ≥ 2.

828

831

834

836

838

854

856

858

860

864

866

868

874

876

Lemma B.1 (vector p ⊥ <sup>n</sup> orthogonal to p1, . . . , pn−<sup>1</sup> in <sup>R</sup> <sup>n</sup>). *Let* e1, . . . , e<sup>n</sup> *be an orthonormal basis of* <sup>R</sup> <sup>n</sup>*, so* |e<sup>i</sup> | = 1 *and* ei · e<sup>j</sup> = 0 *for* i ̸= j*. For any* n − 1 *vectors* p1, . . . , pn−<sup>1</sup> ∈ <sup>R</sup> <sup>n</sup>*, there is a vector* p ⊥ n *that is orthogonal to all* p1, . . . , pn−<sup>1</sup> *and has coordinates that are degree* n − 1 *polynomials in the coordinates of* p1, . . . , pn−1*.*

*Proof of Lemma [B.1.](#page-14-4)* Below the 'unusual determinant' with the n − 1 vector columns p1, . . . , pn−<sup>1</sup> and the last column of the n vectors e1, . . . , e<sup>n</sup> is only a short notation for the following expansion by the last column: | . . . | e<sup>1</sup> p<sup>1</sup> . . . pn−<sup>1</sup> . . . | . . . | e<sup>n</sup> =

Pn i=1 (−1)<sup>n</sup>+<sup>i</sup> det(i)e<sup>i</sup> , where det(i) is the usual (n − 1) × (n − 1) determinant obtained from the n − 1 vector columns p1, . . . , pn−<sup>1</sup> by removing the i-th row, so we set p ⊥ <sup>n</sup> = Pn i=1 (−1)<sup>n</sup>+<sup>i</sup> det(i)e<sup>i</sup> .

For example, if n = 2 then p<sup>1</sup> = (x1, x2) has the vector p ⊥ <sup>2</sup> = x<sup>1</sup> e<sup>1</sup> x<sup>2</sup> e<sup>2</sup> <sup>=</sup> <sup>x</sup>1e<sup>2</sup> <sup>−</sup> <sup>x</sup>2e<sup>1</sup> = (−x2, x1) <sup>⊥</sup> <sup>p</sup><sup>1</sup> If <sup>n</sup> = 3, p<sup>1</sup> = (x1, x2, x3) and p<sup>2</sup> = (y1, y2, y3), then p ⊥ <sup>3</sup> = x<sup>1</sup> y<sup>1</sup> e<sup>1</sup> x<sup>2</sup> y<sup>2</sup> e<sup>2</sup> x<sup>3</sup> y<sup>3</sup> e<sup>3</sup> = x<sup>2</sup> y<sup>2</sup> x<sup>3</sup> y<sup>3</sup> e<sup>1</sup> − x<sup>1</sup> y<sup>1</sup> x<sup>3</sup> y<sup>3</sup> e<sup>2</sup> + x<sup>1</sup> y<sup>1</sup> x<sup>2</sup> y<sup>2</sup> e<sup>3</sup> = p<sup>1</sup> × p<sup>2</sup> is the *vector* product of p1, p2.

To show that p ⊥ n is orthogonal to each p<sup>i</sup> , we compute the scalar product p ⊥ n · p<sup>i</sup> = Pn i=1 (−1)<sup>n</sup>+1 det(i)e<sup>i</sup> · p<sup>i</sup> . Since e<sup>i</sup> · p<sup>i</sup> equals the i-th coordinate of the vector p<sup>i</sup> , the last sum is the expansion of the n × n determinant obtained from the original p ⊥ n above by replacing the last column with p<sup>i</sup> . Since the resulting determinant contains two identical columns equal to p<sup>i</sup> , we conclude that p ⊥ n · p<sup>i</sup> = 0.

Lemma [B.1](#page-14-4) holds when given vectors p1, . . . , pn−<sup>1</sup> ∈ <sup>R</sup> <sup>n</sup> are linearly dependent, even if some p<sup>j</sup> = 0. Then p ⊥ <sup>n</sup> = 0 is orthogonal to each p<sup>j</sup> so that p ⊥ n · p<sup>j</sup> = 0.

Definition [B.2](#page-15-0) extends a point-based representation from Definition [3.1](#page-3-1) to dimensions n ≥ 2. The key idea is to represent any m-point cloud A ⊂ R <sup>n</sup> relative to (a simplex of) any base sequence of ordered points p1, . . . , pn−<sup>1</sup> ∈ A. If the vectors p1, . . . , pn−<sup>1</sup> are linearly independent, they form with the vector p ⊥ n from Lemma [B.1](#page-14-4) a (not necessarily orthogonal) basis in R <sup>n</sup>. Below we represent any point p ∈ A by normalized scalar products, which are valid even if p1, . . . , pn−<sup>1</sup> are linearly dependent.

Definition B.2 (point-based representation PR for n ≥ 2). *For any cloud* A ⊂ R <sup>n</sup> *of* m *unordered points, the* center of mass *is* <sup>O</sup>(A) = <sup>1</sup> m P p∈A p*. Shift* A *so that* O(A) *is the origin* 0 ∈ <sup>R</sup> <sup>n</sup>*. The* radius *of* A *is* R(A) = max p∈A |p|*. For any* basis *sequence of points* p1, . . . , pn−<sup>1</sup> ∈ A*, the* squared distance matrix SD(p1, . . . , pn−1) *consists of* |p<sup>i</sup> − p<sup>j</sup> | 2 *for* i, j = 0, . . . , n − 1*, where* p<sup>0</sup> = 0*. Let* p ⊥ <sup>n</sup> *be the vector in Lemma [B.1.](#page-14-4) For any point* q ∈ A \ {p1, . . . , pn−1}*, the* n × (m − n + 1) *matrix* M(A; p1, . . . , pn−1) *has a column of scalar products* q · p1, . . . , q · pn*. The* point-based representation PR(A; p1, . . . , pn−1) *is the pair*

$$[\text{SD}(p_1, \dots, p_{n-1}), M(A; p_1, \dots, p_{n-1})]$$
.

*The* normalized *representation* NPR(A; p1, . . . , pn−1) *is obtained by dividing all components of* PR(A; p1, . . . , pn−1) *by* R<sup>2</sup> (A)*, except the last row of* M(A; p1, . . . , pn−1)*, which is divided by* R<sup>n</sup>(A)*.*

Lemma B.3 (PR under isometry). *Let a point cloud* A ⊂ R <sup>n</sup> *have a base sequence* (p1, . . . , pn−1)*.*

*(a) Any rigid motion* f *of* R <sup>n</sup> *respects point-based representations from Definition [B.2](#page-15-0) so that*

$$\text{PR}(A; p_1, \dots, p_{n-1}) = \text{PR}(f(A); f(p_1), \dots, f(p_{n-1})).$$

*(b) For any orientation-reversing isometry* f *of* R <sup>n</sup>*, the representation* PR(f(A); f(p1), . . . , f(pn−1) *differs from* PR(A; p1, . . . , pn−1) *by reversing all signs in the last row of the matrix* M(A; p1, . . . , pn−1)*.*

887 888

890

894

896

898

911

914 915 916

918

924

928 The part *if* ⇐. For any positive semi-definite matrix GM, there is an orthogonal matrix Q such that Q<sup>T</sup> GMQ = D is the diagonal matrix, whose <sup>m</sup> <sup>−</sup> <sup>1</sup> diagonal elements are non-negative eigenvalues of GM. The diagonal matrix √ D consists of the square roots of eigenvalues of GM.

*(c) The normalized point-based representation* NPR(A; p1, . . . , pn−1) *in Definition [B.2](#page-15-0) is preserved by any composition of rigid motion and uniform scaling.*

*Proof of Lemma [B.3.](#page-15-1)* (a) Since rigid motion preserves distances and scalar products, all components of the point-based representation PR(A; p1, . . . , pn−1) are invariant.

(b) Using a composition with a suitable orientation-preserving isometry (rigid motion), one can assume that f is the mirror reflection in a linear hyperspace H containing the origin 0 and the base sequence p1, . . . , pn−<sup>1</sup> of A. Since f preserves distances, R(A) and SD(A; p1, . . . , pn−1) are invariant. Then f fixes all points from H including p1, . . . , pn−1, hence the vector p<sup>n</sup> from Lemma [B.1.](#page-14-4) Any point q ∈ A \ p1, . . . , pn−<sup>1</sup> keeps its scalar product q · p<sup>i</sup> for i = 1, . . . , n − 1 and changes the sign of q · pn, because q and its mirror image f(q) have opposite projections to pn. The above arguments hold even if the base sequence p1, . . . , pn−<sup>1</sup> is degenerate, not generating an (n − 1)-dimensional subspace in <sup>R</sup> <sup>n</sup>. Then there are infinitely many choices of H above and p<sup>n</sup> = 0, so the last row of M(A; p1, . . . , pn−1) consists of zeros.

(c) Under uniform scaling by a factor s, all squared distances and scalar products q ·p<sup>i</sup> , i = 1, . . . , n−1, are multiplied by s 2 . The vector p ⊥ n from Lemma [B.1](#page-14-4) is multiplied by s n−1 , hence all scalar products q · p<sup>n</sup> in the last row of M(A; p1, . . . , pn−1) are divided by R<sup>n</sup>(A).

The *affine dimension* 0 ≤ aff(A) ≤ n of a cloud A = {p1, . . . , pm} ⊂ <sup>R</sup> <sup>n</sup> is the maximum dimension of the vector space generated by all inter-point vectors p<sup>i</sup> − p<sup>j</sup> , i, j ∈ {1, . . . , m}. Then aff(A) is an isometry invariant and is independent of an order of points of A. Any cloud A of 2 distinct points has aff(A) = 1. Any cloud A of 3 points that are not in the same straight line has aff(A) = 2.

Lemma [B.4](#page-16-0) provides a simple criterion for a matrix to be realizable by squared distances of a point cloud in R n.

Lemma B.4 (realization of distances). *(a) A symmetric* m × m *matrix of* sij ≥ 0 *with* sii = 0 *is realizable as a matrix of squared distances between points* p<sup>0</sup> = 0, p1, . . . , pm−<sup>1</sup> ∈ <sup>R</sup> <sup>n</sup> if and only if *the* (m − 1) × (m − 1) *matrix* gij = s0<sup>i</sup> + s0<sup>j</sup> − sij 2 *has only non-negative eigenvalues.*

*(b) If the condition in (a) holds,* aff(0, p1, . . . , pm−1) *equals the number* k ≤ m − 1 ≤ n *of positive eigenvalues. Also in this case,* gij = p<sup>i</sup> · p<sup>j</sup> *define the* Gram matrix GM *of the vectors* p1, . . . , pm−<sup>1</sup> ∈ <sup>R</sup> <sup>n</sup>*, which are uniquely determined in time* O(m<sup>3</sup> ) *up to an orthogonal map in* <sup>R</sup> n*.*

*Proof of Lemma [B.4.](#page-16-0)* (a) We extend Theorem 1 from [\(Dekster & Wilker,](#page-8-23) [1987\)](#page-8-23) to the case m < n + 1 and also justify the reconstruction of p1, . . . , pm−<sup>1</sup> in time O(m<sup>3</sup> ) uniquely in <sup>R</sup> <sup>n</sup> up to an orthogonal map from the group O(n).

The part *only if* ⇒. Let a symmetric matrix S consist of squared distances between points p<sup>0</sup> = 0, p1, . . . , pm−<sup>1</sup> ∈ <sup>R</sup> <sup>n</sup>. For i, j = 1, . . . , m − 1, the matrix with the elements

$$g_{ij} = \frac{s_{0i} + s_{0j} - s_{ij}}{2} = \frac{p_i^2 + p_j^2 - |p_i - p_j|^2}{2} = p_i \cdot p_j$$

is the Gram matrix, which can be written as GM = P <sup>T</sup> P, where the columns of the n × (m − 1) matrix P are the vectors p1, . . . , pm−<sup>1</sup> . For any vector v ∈ <sup>R</sup> m−1 , we have

$$0 \leq |Pv|^2 = (Pv)^T (Pv) = v^T (P^T P) v = v^T \text{GM}v.$$

Since the quadratic form v <sup>T</sup> GMv ≥ 0 for any v ∈ <sup>R</sup> m−1 , the matrix GM is positive semi-definite meaning that GM has only non-negative eigenvalues, see Theorem 7.2.7 in [\(Horn & Johnson,](#page-9-24) [2012\)](#page-9-24).

(b) The number of positive eigenvalues of GM equals the dimension k = aff({0, p1, . . . , pm−1}) of the subspace in <sup>R</sup> n linearly spanned by p1, . . . , pm−1. We may assume that all k ≤ n positive eigenvalues of GM correspond to the first k

938

946 947 948 Though Lemma [B.4](#page-16-0) gives a two-sided criterion for realizability of distances by points p1, . . . , p<sup>m</sup> ∈ <sup>R</sup> <sup>n</sup>, the space of distance matrices is highly singular and cannot be easily sampled. Even m = 4 points in R <sup>2</sup> have 6 distances that should satisfy a polynomial equation saying that the tetrahedron with these 6 edge lengths has volume 0.

949

954

956

958

971

974

976

978

987 988 coordinates of R <sup>n</sup>. Since Q<sup>T</sup> = Q−<sup>1</sup> , the given matrix GM = QDQ<sup>T</sup> = (Q √ D)(Q √ D) <sup>T</sup> becomes the Gram matrix of the columns of Q √ D. These columns become the reconstructed vectors p1, . . . , pm−<sup>1</sup> ∈ <sup>R</sup> n.

If there is another diagonalization Q˜<sup>T</sup> GMQ˜ = D˜ for Q˜ ∈ O(n), then D˜ differs from D by a permutation of eigenvalues, which is realized by an orthogonal map, so we set D˜ = D. Then GM = QD˜ Q˜<sup>T</sup> = (Q˜ √ D)(Q˜ √ D) T is the Gram matrix of the columns of Q˜ √ D.

The new columns differ from the previously reconstructed vectors p1, . . . , pm−<sup>1</sup> ∈ <sup>R</sup> <sup>n</sup> by the orthogonal map QQ˜<sup>T</sup> . Hence the reconstruction is unique up to O(n)-transformations. Computing eigenvectors p1, . . . , pm−<sup>1</sup> needs a diagonalization of GM in time O(m<sup>3</sup> ), see (?)section 11.5]press2007numerical.

So a randomly sampled matrix of potential distances for m > n + 1 is unlikely to be realizable by a cloud of m ordered points in R <sup>n</sup>. Hence Lemma [B.4](#page-16-0) for m ≤ n + 1 is complemented by Theorem [B.7](#page-18-0) describing the much more practical realizabilty of a point-based representation.

Chapter 3 in [\(Liberti & Lavor,](#page-10-24) [2017\)](#page-10-24) discusses realizations of a complete graph given by a distance matrix in R n.

Lemma [B.5\(](#page-17-0)a) and later results hold for all clouds including degenerate ones, e.g. for 3 points in a straight line.

Any points p1, . . . , pn−<sup>1</sup> ∈ A have aff(p1, . . . , pn−1) ≤ n − 2. For example, any two distinct points in A ⊂ <sup>R</sup> <sup>3</sup> generate a straight line. Lemma [B.5\(](#page-17-0)c) proves that PR(A; p1, . . . , pn−1) suffices to reconstruct a cloud A ⊂ <sup>R</sup> <sup>n</sup> for a suitable sequence p1, . . . , pn−1. In <sup>R</sup> 2 , any point p<sup>1</sup> ̸= O(A) forms a suitable {p1}. In <sup>R</sup> 3 , one can choose any distinct points p1, p<sup>2</sup> ∈ A so that the infinite straight line via p1, p<sup>2</sup> avoids O(A).

If there are no such p1, p2, then A ⊂ <sup>R</sup> 3 is contained in a straight line L, so aff(A) = 1. In this degenerate case, the stronger condition aff(O(A) ∪ {p1, . . . , pn−1}) = aff(A) will help reconstruct A ⊂ L by using any point p<sup>1</sup> ̸= O(A). The first step is to reconstruct any ordered sequence from its distance matrix in Lemma [B.5\(](#page-17-0)a).

Lemma [B.5](#page-17-0) improves Lemma E.5 in [\(Widdowson & Kurlin,](#page-11-15) [2023\)](#page-11-15) by justifying a time for a point cloud reconstruction based on Lemma [B.4.](#page-16-0)

Lemma B.5 (reconstruction). *(a) Any sequence of ordered points* p1, . . . , p<sup>m</sup> *in* <sup>R</sup> <sup>n</sup> *can be reconstructed (uniquely up to isometry) from the matrix of the Euclidean distances* |pi−p<sup>j</sup> | *in time* O(m<sup>3</sup> )*. If all distances are divided by* R = max i=1,...,m |p<sup>i</sup> |*, the reconstruction of* p1, . . . , p<sup>m</sup> *is unique up to isometry and uniform scaling in* <sup>R</sup> n*.*

*(b) If* m ≤ n*, the uniqueness of reconstructions in part (a) remains true if we replace isometry by rigid motion in* R n*.*

*(c) Any cloud* A ⊂ R <sup>n</sup> *of* m *unordered points can be reconstructed (uniquely up to rigid motion in* <sup>R</sup> <sup>n</sup>*) from a point-based representation* PR(A; p1, . . . , pn−1) *in time* O(m<sup>3</sup> ) *for any* p1, . . . , pn−<sup>1</sup> ∈ A *with* aff(O(A)∪ {p1, . . . , pn−1}) = aff(A)*. If* aff(A) = n*, then* aff(O(A)∪ {p1, . . . , pn−1}) = n−1 *suffices. Any cloud* A ⊂ <sup>R</sup> <sup>n</sup> *has a suitable sequence* p1, . . . , pn−<sup>1</sup> *in all cases.*

*Proof of Lemma [B.5.](#page-17-0)* (a) By translation, we can put p<sup>1</sup> at the origin 0 ∈ <sup>R</sup> <sup>n</sup>. Let G be the (m − 1) × (m − 1) matrix Gij = p 2 <sup>i</sup> + p 2 <sup>j</sup> − |p<sup>i</sup> − p<sup>j</sup> | 2 2 = p<sup>i</sup> · p<sup>j</sup> constructed from squared distances between p<sup>1</sup> = 0, . . . , p<sup>m</sup> for i, j = 2, . . . , m. By Lemma [B.4](#page-16-0) if G has k ≤ n positive eigenvalues, then p<sup>1</sup> = 0, . . . , p<sup>m</sup> can be uniquely determined up to isometry in R <sup>k</sup> ⊂ <sup>R</sup> <sup>n</sup> in time O(m<sup>3</sup> ). If all distances are divided by the same radius R(p{m}), the above construction guarantees uniqueness up to isometry and uniform scaling.

(b) If m ≤ n, any mirror images of p{m} ⊂ <sup>R</sup> <sup>n</sup> after a suitable rigid motion in <sup>R</sup> <sup>n</sup> can be assumed to belong to an (n − 1)-dimensional hyperspace H ⊂ <sup>R</sup> <sup>n</sup>, where they are matched by a mirror reflection H → H with respect to an (n − 2)-dimensional subspace S ⊂ H, which is realized by the 180◦ orientation-preserving rotation of <sup>R</sup> <sup>n</sup> around S.

994 996 It suffices to reconstruct A ⊂ R <sup>k</sup> up to rigid motion in <sup>R</sup> k . Since aff(0, p1, . . . , pn−1) = k, some k vectors (say) p1, . . . , p<sup>k</sup> from p1, . . . , pn−<sup>1</sup> form a linear basis of <sup>R</sup> k . The k points p1, . . . , p<sup>k</sup> are uniquely reconstructed up to rigid motion in <sup>R</sup> k by part (b). Any other point q ∈ A \ {p1, . . . , pk} is uniquely determined by its projections (q · pi)/|p<sup>i</sup> |, which can be found from the first k < n rows of the matrix M(A; p1, . . . , pn−1) for the point q, see Definition [B.2.](#page-15-0)

998 1000 1001 1002 1003 In the generic case aff(A) = n, the condition aff(0, p1, . . . , pn−1) = n−1 means that p1, . . . , pn−<sup>1</sup> are linearly independent and hence form a linear basis of R <sup>n</sup> with the extra vector p ⊥ n from Lemma [B.1.](#page-14-4) The sequence (0, p1, . . . , pn−1) of n points can be uniquely reconstructed up to rigid motion in R <sup>n</sup> by part (b). Any other point q ∈ A \ {p1, . . . , pn−1} is uniquely determined by its projections <sup>q</sup> · <sup>p</sup><sup>i</sup> |p<sup>i</sup> to the n basis vectors p1, . . . , pn−1, p<sup>⊥</sup> n , which can be found from the column of M(A; p1, . . . , pn−1) for q.

1004

1005 1006 1007 Lemma [B.5\(](#page-17-0)b) for m = n = 3 implies that any triangle is determined by its sides up to rigid motion in R 3 . For example, the sides 3, 4, 5 define a right-angled triangle whose mirror images are not related by rigid motion inside a plane H ⊂ R 3 , but are matched by composing a suitable rigid motion in H and a 180◦ rotation of R 3 around a line in H.

1008 1009 Lemma B.6 (smoothness of PR). *For any cloud* A ⊂ R <sup>n</sup> *and a base sequence* p1, . . . , pn−<sup>1</sup> ∈ A*, all components of* PR(A; p1, . . . , pn−1) *have continuous partial derivatives (of any order) with respect to all (coordinates of) points of* A *as long as* R(A) > 0*, so some points of* A *remain distinct.*

1014 1016 *Proof of Lemma [B.6.](#page-18-1)* The point-based representation PR(A; p{n − 1}) consists of squared distances in the matrix SD(p{n − 1}) and scalar products in the matrix M(A; p{n − 1}) of all points q ∈ A \ p{n − 1} with the vectors p1, . . . , pn−<sup>1</sup> from the base sequence p{n − 1} and the vector p<sup>n</sup> ⊥ p1, . . . , pn−<sup>1</sup> from Lemma [B.1.](#page-14-4) All these components are polynomials in the coordinates of the points of A, so have all continuous partial derivatives.

1019 1024 1026 Theorem B.7 (realizability of abstract PR). *Let* S *be a symmetric* n × n *matrix of* sij ≥ 0 *with* sii = 0*. Let* M *be any* n × (m − n + 1) *matrix for* m ≥ n*. The pair* [S, M] *is realizable as a point-based representation* PR(A; p1, . . . , pn−1) *for a cloud* A ⊂ R <sup>n</sup> *of* m *points with* O(A) = 0 *and a base sequence* p1, . . . , pn−<sup>1</sup> *if and only if (1) the* (n − 1) × (n − 1) *matrix* Gij = 1 2 (s1<sup>i</sup> + s1<sup>j</sup> − sij ) *has only positive eigenvalues, which uniquely determines* p1, . . . , pn−<sup>1</sup> *up to isometry, and (2)* nP−1 j=1 (p<sup>i</sup> · p<sup>j</sup> ) + m−P<sup>n</sup>+1 j=1 Mij = 0 *for* i = 1, . . . , n*, where* p<sup>n</sup> = p ⊥ n *is the orthogonal vector from Lemma [B.1.](#page-14-4)*

1029 *Proof of Theorem [B.7.](#page-18-0)* The realizability of S as a matrix of squared distances between n points 0, p1, . . . , pn−<sup>1</sup> from the base sequence p1, . . . , pn−<sup>1</sup> follows from Lemma [B.4.](#page-16-0) The orthogonal vector p ⊥ n (also denoted by p<sup>n</sup> here for uniformity) from Lemma [B.1](#page-14-4) complements p1, . . . , pn−<sup>1</sup> to a linear basis of <sup>R</sup> <sup>n</sup>. By Definition [B.2,](#page-15-0) every element Mij of the matrix M = M(A; p1, . . . , pn−1) equals p<sup>i</sup> · q for some q ∈ A \ {p1, . . . , pn−1}, where i = 1, . . . , n.

1034 1036 Hence nP−1 j=1 (p<sup>i</sup> · p<sup>j</sup> ) + m−P<sup>n</sup>+1 j=1 Mij = 0 can be rewritten as p<sup>i</sup> · ( P p∈A p) = 0 for i = 1, . . . , n. These n equations mean that <sup>O</sup>(A) = <sup>1</sup> m P p∈A p is at the origin 0 ∈ R n.

1039 Conversely, for any M satisfying condition (2), we interpret every column (M1<sup>j</sup> , . . . , Mnj ) T as a vector of scalar products (q · p1, . . . , q · pn), which determine a position of a point q ∈ A \ {p1, . . . , pn−1} in the basis p1, . . . , pn.

1040 1041 1042 In Theorem [B.7,](#page-18-0) condition (2) is equivalent to O(A) = 0 ∈ <sup>R</sup> <sup>n</sup> and implies that m − n columns of M consist of free parameters, which determine the remaining column.

(c) We will reconstruct a cloud A ⊂ R <sup>n</sup> so that the center of mass O(A) is the origin 0 ∈ <sup>R</sup> <sup>n</sup>. If aff(A) = k < n, the cloud A ⊂ R <sup>n</sup> is contained in an affine k-dimensional subspace, which can be rigidly moved to the linear subspace <sup>R</sup> <sup>k</sup> ⊂ <sup>R</sup> <sup>n</sup> for the first k of n coordinates in R n.

Theorem [B.7](#page-18-0) extends Theorem [3.3](#page-3-2) to dimensions n ≥ 2.

1045 1046 1047 1048 1049 For n = 3, condition (1) about positive eigenvalues of the 2 × 2 matrix G means that 3 distances a ≤ b ≤ c between points 0, p1, p<sup>2</sup> in <sup>R</sup> 3 satisfy a > 0 and a + b > c, so the triangle on 0, p1, p<sup>2</sup> is non-degenerate. By the cosine theorem p<sup>1</sup> · p<sup>2</sup> = 2 (a <sup>2</sup> + b <sup>2</sup> − c 2 ), so the matrix G = a 2 1 2 (a <sup>2</sup> + b <sup>2</sup> − c 2 ) 1 2 (a <sup>2</sup> + b <sup>2</sup> − c 2 ) b 2 has a <sup>2</sup> > 0 and a positive determinant:

1054 Assuming that 0 < a ≤ b ≤ c, the last inequality is equivalent to one triangle inequality a + b > c.

1056 Now we extend a point-based representation from Definition [B.2](#page-15-0) to a complete invariant of a point cloud A under rigid motion in R <sup>n</sup>. In applications, A can have distinguished points, for example, heavy atoms in atomic clouds, which can be used to minimize choices for p1, . . . , pn−1.

1059 1060 1061 1062 Definition [B.8](#page-19-1) will extend Definition [3.4](#page-4-0) to n > 2 by combining all PR(A; p1, . . . , pn−1) in a nested invariant by dropping points p1, . . . , pn−<sup>1</sup> ∈ A one at a time. This invariant is needed only for comparisons (metric computations), while any cloud A can be stored in computer memory as a single PR(A; p1, . . . , pn−1) due to Theorem [B.7.](#page-18-0)

1063 1064 1065 1066 1067 1068 Definition B.8 (NDP : Nested Distributed Projection). *Let* A ⊂ R <sup>n</sup> *be any cloud of* m *unordered points. For any ordered points* p1, . . . , pn−<sup>2</sup> ∈ A*, let* NDP(A; p1, . . . , pn−2) *be the unordered collection of* PR(A; p1, . . . , pn−1) *for all points* pn−<sup>1</sup> ∈ A \ {p1, . . . , pn−2}*. Similarly, for any* 1 ≤ k ≤ n − 2*, let* NDP(A; p1, . . . , pk−1) *be the unordered collection of* NDP(A; p1, . . . , pk) *for all points* p<sup>k</sup> ∈ A \ {p1, . . . , pk−1}*. For* k = 1*, the full* Nested Distributed Projection NDP(A) *depends only on* A*.*

1069 For n = 2 and any cloud A ⊂ R , the Nested Distributed Projection NDP(A) in Definition [B.8](#page-19-1) is the same as in Definition [3.4,](#page-4-0) i.e. NDP(A) is the unordered collection of point-based representations PR(A; p1) for all p<sup>1</sup> ∈ A.

1074 For n = 3 and any A ⊂ R 3 , the Nested Distributed Projection NDP(A) is the unordered collection of NDP(A; p1) for all p<sup>1</sup> ∈ A. Each NDP(A; p1) is the unordered collection of PR(A; p1, p2) for all p<sup>2</sup> ∈ A \ {p1}.

1076 Similarly to Definition [3.4,](#page-4-0) if a cloud A has internal symmetries as in Example [3.2,](#page-3-3) one can collapse identical objects to a single one with a weight to speed up computations. We avoid collapsing only to simplify arguments for n > 2.

1079 Lemma [B.5\(](#page-17-0)c) implies that any cloud A ⊂ R <sup>n</sup> of m unordered points can be reconstructed from NDP(A) uniquely up to rigid motion. Indeed, NDP(A) contains (nested) PRs depending on all possible n − 1 points p1, . . . , pn−<sup>1</sup> ∈ A. At least one PR(A; p1, . . . , pn−1) satisfies Lemma [B.5\(](#page-17-0)c) and suffices to reconstruct A uniquely up to rigid motion.

1081 1082 1083 In Theorem [B.9](#page-19-0) for n > 2, the equality NDP(A) = NDP(B) means a bijection β : NDP(A) → NDP(B) respecting the nested structure of all PRs in Definition [B.8.](#page-19-1)

1084 1085 1086 In detail, for any 1 ≤ k ≤ n − 1 and points p1, . . . , pk, the bijection β matches NDP(A; p1, . . . , pk) with a unique NDP(B; q1, . . . , qk) for some q1, . . . , q<sup>k</sup> ∈ B.

1087 1089 If n = 3, then β matches every NDP(A; p1) with a unique NDP(B; q1) in the sense that this bijection NDP(A; p1) → NDP(B; q1) matches PR(A; p1, p2) for every p<sup>2</sup> ∈ A \ {p1} with PR(B; q1, q2) for a unique q<sup>2</sup> ∈ B − {q1}.

1090 1091 1092 Theorem B.9 (completeness of NDP). *The Nested Distributed Projection is complete in the sense that any clouds* A, B ⊂ R <sup>n</sup> *of* m *unordered points are related by rigid motion in* <sup>R</sup> <sup>n</sup> *if and only if* NDP(A) = NDP(B) *so that there is a bijection* NDP(A) → NDP(B) *matching all* PR*s.*

1093

1094 1095 1096 1097 1098 1099 *Proof of Theorem [B.9.](#page-19-0)* The part *only if* : we will prove that any rigid motion f moving the cloud A to B = f(A) implies that NDP(A) = NDP(B). By Lemma [B.3\(](#page-15-1)a) the rigid motion f matches every PR(A; p1, . . . , pn−1) from NDP(A) with PR(B; f(p1), . . . , f(pn−1)). Then, for any 1 ≤ k ≤ n − 2 and p1, . . . , p<sup>k</sup> ∈ A, we get a bijection NDP(A; p1, . . . , pk) → NDP(B; f(p1), . . . , f(pk)) Hence f induces a bijecton NCP(A) → NCP(B) between all PRs respecting the nested structure in Definition [B.8.](#page-19-1)

$$\begin{aligned} 1050 & 4 \det G = 4a^2b^2 - (a^2 + b^2 - c^2)^2 = \\ 1051 & (c^2 - (a^2 - 2ab + b^2))((a^2 + 2ab + b^2) - c^2) = \\ 1052 & (c^2 - (a - b)^2)((a + b)^2 - c^2) > 0. \end{aligned}$$

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

![](_page_20_Diagram_9.jpeg)

1151 Figure 13. The spaces in yellow for triangles (D3) and parallelograms (D4) under rigid motion and uniform scaling in R .

1154 For m = 4, we can choose s = p2+p<sup>3</sup> ∈ J, then any p<sup>3</sup> in the disk with the radius R and center s so that |p2| = |p3−s| ≤ R. For any parallelogram in R 2 , its vertex cloud A has a longest diagonal between (say) p1, p<sup>3</sup> that should be at (±R, 0). All

The part *if* : NDP(A) = NDP(B) will guarantee a rigid motion f moving the cloud A to B = f(A). Choose any base sequence p1, . . . , pn−<sup>1</sup> ∈ A that suffices for a unique reconstruction of A ⊂ <sup>R</sup> <sup>n</sup> up to rigid motion in Lemma [B.5\(](#page-17-0)c). The given bijection NDP(A) → NDP(B) matches PR(A; p1, . . . , pn−1) with an equal PR(B; q1, . . . , qn−1) for some q1, . . . , qn−<sup>1</sup> ∈ B.

Lemma [B.5\(](#page-17-0)c) implies that a reconstruction of A, B from PR(A; σ(p1, . . . , pn−1)) = PR(B; q1, . . . , qn−1) is unique up to rigid motion in R <sup>n</sup> so that A, B are matched by a rigid motion f as required. If aff(A) = aff(B) < n, this motion f may not be unique. For example, any clouds A, B ⊂ R 3 that are contained in a straight line L ⊂ R 3 are pointwise fixed by any rotation around the line L.

### C. Maps of cloud spaces and explicit computations of invariants

This section explains how cloud spaces can be visualized by considering the previously known and new types of 4-point clouds (quads) in R 2 . This geographic-style approach extends to any number m of points in R n.

For any cloud A ⊂ R <sup>n</sup>, the center O(A) = 0 ∈ <sup>R</sup> <sup>n</sup> is the origin. For n = 2, let p{1} consist of a single point p<sup>1</sup> ∈ A with |p1| = R(A) = R. We can fix p<sup>1</sup> = (R, 0) in <sup>R</sup> . Then all points p2, . . . , p<sup>m</sup> are in the disk D = {x <sup>2</sup> + y <sup>2</sup> ≤ R<sup>2</sup>}. Since Pm i=2 p<sup>i</sup> = −p<sup>1</sup> = (−R, 0), p<sup>m</sup> is determined from p2, . . . , pm−<sup>1</sup> ∈ D that satisfy only one equation

$$R^2 \geq |p_m|^2 = |(R, 0)^T + \sum_{i=2}^{m-1} p_i|^2 = (R + x)^2 + y^2,$$

where (x, y) are the coordinates of s = mP−1 i=2 pi . The domain of s is the intersection J = D ∩ {(R + x) <sup>2</sup> + y <sup>2</sup> ≤ R<sup>2</sup>}.

For m = 3, we have s = (x, y) = p2. The symmetry p<sup>2</sup> ↔ p<sup>3</sup> allows us to choose any p<sup>2</sup> in the left half (yellow) D<sup>3</sup> of the intersection J in Fig. [13](#page-20-1) (left). Then the Rigid Cloud Space CRS(<sup>R</sup> <sup>n</sup>; 3) is parametrized by any radius R > 0 and p<sup>2</sup> ∈ D3. All equilateral triangles have p<sup>2</sup> = (− <sup>2</sup>R, ± √ 3 <sup>2</sup> R). All isosceles triangles have p<sup>2</sup> in the boundary ∂D<sup>3</sup> whose points should be identified under (x, y) 7→ (x, −y). All p<sup>2</sup> = (x, 0) with −R ≤ x ≤ −<sup>1</sup> <sup>2</sup>R represent degenerate triangles with the vertices (R, 0), (x, 0), (−R − x, 0) in the same line.

1156 possible s = p<sup>2</sup> + (−R, 0) ∈ J mean that p<sup>2</sup> can be anywhere in D. Due to the symmetry p<sup>2</sup> ↔ p4, the left half D<sup>4</sup> of D in Fig. [13](#page-20-1) (right) is the subspace of all parallelograms in DCS(<sup>R</sup> 2 ; 4) = CRS(<sup>R</sup> 2 ; 4)/scaling.

1159 1160 1161 1164 1166 1167 1168 1169 1174 1176 1179 1180 1181 1182 1183 1184 1185 1186 1187 1188 1189 1190 1194 1196 1199 1200 1204 1206 1209 Similarly for m > 4, n ≥ 2, we can sequentially sample points p2, . . . , pm−<sup>1</sup> from allowed disks (high-dimensional for n > 2) to get a unique representation of A under rigid motion. The symmetry f : (x, y) 7→ (x, −y) on D identifies mirror images of A. CIS(<sup>R</sup> <sup>n</sup>; m) is the quotient of CRS(<sup>R</sup> <sup>n</sup>; m) under (x, y) ∼ (x, −y), take the upper halves of D3, D<sup>4</sup> for triangles and parallelograms, respectively. We expand Fig. [13](#page-20-1) above to illustrate severak important subspaces in the Isometry Cloud Space CIS(<sup>R</sup> 2 ; m) and the Similarity Cloud Space CSS(<sup>R</sup> 2 ; m) for m = 3, 4. For simplicity, we call all clouds of 3 and 4 unordered points triangles and quadrilaterals, respectively. However, all these polygons are considered equivalent when we re-order their vertices. If all m points are ordered, parametrizations of the resulting shape spaces were studied in geometry [\(Kapovich & Millson,](#page-9-25) [1996\)](#page-9-25) and shape theory [\(Kendall et al.,](#page-9-3) [2009\)](#page-9-3). We focus on the much harder quotient spaces of m unordered points. Theorem [B.7](#page-18-0) explicitly describes all realizable Point-based Representations. Though the same point cloud A ⊂ R can have many PR(A; p{n − 1}) depending on a base sequence p{n − 1} ⊂ A, we can easily sample any of them and always reconstruct A, while random sampling distance-based invariants doesn't guarantee the existence of A because of extra relations between inter-point distances. Though PR(A; p{n−1}) consists of scalar products q · p<sup>i</sup> with basis vectors p1, . . . , pn, it is easier to visualize the isometry spaces by directly using some points q ∈ A as parameters instead of their projections. Case m = 3 of triangles is the same in all dimensions n ≥ 2. We consider R 2 for simplicity. Fig. [13](#page-20-1) (left) showed the Dilation Cloud Space DCS(<sup>R</sup> 2 ; 3) of triangles A modulo rigid motion and uniform scaling in <sup>R</sup> 2 . We assume that the center of mass is at the origin: C(A) = 0 in <sup>R</sup> 2 . After the radius R = 1 of A is fixed up to scaling, we also fix the first vertex at p<sup>1</sup> = (R, 0). Then DCS(<sup>R</sup> 2 ; 3) is parametrized by the second vertex p<sup>2</sup> ∈ D3, because the vertex p<sup>3</sup> is uniquely determined by p<sup>1</sup> + p<sup>2</sup> + p<sup>3</sup> = 0. The blue boundary of DCS(<sup>R</sup> 2 ; 3) consists of points p<sup>2</sup> that define isosceles triangles. The vertical part of the blue boundary in Fig. [14](#page-22-0) (left) represents all isosceles triangles with a unique angle (not equal to two equal ones) less than 60◦ . The round part of the blue boundary in Fig. [14](#page-22-0) (right) represents all isosceles triangles with a unique angle greater than 60◦ . These boundary parts meet at the red points (− , ± √ <sup>2</sup> R) representing all equilateral triangles. If p<sup>2</sup> = (x, 0) for −R ≤ x ≤ − <sup>R</sup> 2 , then p<sup>3</sup> = (−R − x, 0), so the triangle generates to three points in the line. In the yellow space D<sup>3</sup> = CSS<sup>o</sup> (<sup>R</sup> ; 3), the mirror reflection (x, y) 7→ (x, −y) maps every isosceles triangle to itself, more exactly, to an equivalent triangle under rigid motion. Hence all points of the blue boundary of D<sup>3</sup> should be identified under (x, y) 7→ (x, −y). Then the space D<sup>3</sup> of all triangles (including degenerate ones) under rigid motion and uniform scaling can be visualized as a topological sphere S <sup>2</sup> whose the northern and southern hemispheres are obtained from the upper and lower halves of D3. Case m = 4 of quadrilaterals in R 2 . Fix the center of mass O(A) = 0 ∈ <sup>R</sup> 2 at the origin, the radius R(A) = R, and a most distant (from 0) point p<sup>1</sup> at (R, 0). The other vertices p2, p3, p<sup>4</sup> belong to the disk D = {x <sup>2</sup> + y <sup>2</sup> ≤ R<sup>2</sup>} and have the shifted center of mass <sup>p</sup>2+p3+p<sup>4</sup> <sup>3</sup> = (− 3 , 0). Hence, for a fixed radius R, the space CSS(<sup>R</sup> ; 4) is 4-dimensional. The subspace of parallelograms in CSS(<sup>R</sup> 2 ; 4) is 2-dimensional. For any parallelogram A, its other most distant vertex is p<sup>3</sup> = (−R, 0) opposite to p<sup>1</sup> with respect to 0. Then p<sup>2</sup> + p<sup>4</sup> = 0 and the symmetry p<sup>2</sup> ↔ p<sup>4</sup> allows us to consider only p<sup>2</sup> in the yellow half-disk D4, which uniquely determines its symmetric image p<sup>4</sup> in Fig. [13](#page-20-1) (left). The round (blue) boundary of D<sup>4</sup> in Fig. [15](#page-22-1) (left) represents all rectangles inscribed in the circle x <sup>2</sup> + y <sup>2</sup> = R<sup>2</sup> . The vertical (orange) boundary of D<sup>4</sup> in Fig. [15](#page-22-1) (right) represents all rhombi with equal sides. The reflection (x, y) 7→ (x, −y) maps any parallelogram to its mirror image and preserves the equivalence class (up to rigid motion) of any rectangle or rhombus, which are mirror-symmetric. Hence all points on the boundary of D<sup>4</sup> should be identified under (x, y) 7→ (x, −y). The resulting quotient is a topological sphere S as D<sup>3</sup> for all triangles, unsurprisingly because a parallelogram can be considered as a double triangle.

![](_page_22_Figure_2.jpeg)

1229 Figure 14. The (blue) subspace of all isosceles triangles in CSS(<sup>R</sup> 2 ; 3). Left: isosceles triangles with |p<sup>1</sup> − p2| = |p<sup>1</sup> − p3|. Right: isosceles triangles with |p<sup>3</sup> − p1| = |p<sup>3</sup> − p2|.

![](_page_22_Figure_4.jpeg)

1256 1259 1260 Another interesting case is when one of the vertices p<sup>3</sup> = (x, 0) belongs to the x-axis for x ∈ [−R, R]. Then the (horizontal line passing through) diagonal joining p1, p<sup>3</sup> intersects another diagonal at its mid-point <sup>p</sup>2+p<sup>4</sup> <sup>2</sup> = (x2,4, 0) for x2,<sup>4</sup> = − x+R 2 ∈ [−R, 0]. The resulting cloud A can be called a *quadrilateral with a median diagonal*, briefly *qmed*. If a qmed A is also symmetric with respect to its median diagonal, the A has two pairs of equal sides and is often called a *kite*, see the kite K in Fig. [2](#page-2-0) (right).

1264 Since any kite is mirror-symmetric, the points p<sup>2</sup> = (x, y) and p<sup>4</sup> = (x, −y) represents the same kite up to rigid motion. Hence the (yellow) subspace of all kites in CSS(<sup>R</sup> ; 4) is the upper half K<sup>4</sup> of the disk D in Fig. [16](#page-23-1) (left). For points p<sup>2</sup> in the vertical line x = − R 3 , we get a degenerate kites whose vertices p2, p3, p<sup>4</sup> are in the same straight line. If p<sup>2</sup> = (x, 0),

Figure 15. The (yellow) subspace D<sup>4</sup> of all parallelograms with p<sup>1</sup> = (R, 0) and p<sup>3</sup> = (−R, 0) in CSS(<sup>R</sup> ; 4). Left: the (blue) subspace of rectangles. Right: the (orange) subspace of rhombi.

*1269*

*1274*

*1279*

*1284*

*1289 1290*

*1294*

*1296 1297*

*1299 1300*

*1306*

*1309*

*1314*

*1316*

*1319 For the second point* p<sup>2</sup> = (−2, 1) *with* |p2| <sup>2</sup> = 5*,* p ⊥ <sup>2</sup> = (−1, <sup>−</sup>2)*, we have* PR(T; <sup>p</sup>2) = 5, −3 7 −9 −4 6 −2 *, which*

![](_page_23_Figure_1.jpeg)

![](_page_23_Diagram_2.jpeg)

Figure 16. Left: the (yellow) subspace of kites in CSS(<sup>R</sup> 2 ; 4) parametrized by p<sup>2</sup> ∈ K4. Right: the subspace of qmeds is parametrized by x ∈ [−R, R] and p<sup>2</sup> in the yellow region.

the kite degenerates even further to the case of identical vertices p<sup>2</sup> = p4.

So the subspace K<sup>4</sup> of kites in CSS(<sup>R</sup> 2 ; 4) is 2-dimensional, while the larger subspace of qmeds is 3-dimensional, parametrized by x ∈ [−R, R] and a point p<sup>2</sup> that can take any position in the intersection of the disk D = {x <sup>2</sup> + y <sup>2</sup> ≤ R<sup>2</sup>} and its symmetric image with respect to the diagonal mid-point (x2,4, 0) = (− x+R 2 , 0).

The full space CSS(<sup>R</sup> 2 ; 4) is parametrized by the sum s = p<sup>2</sup> + p<sup>3</sup> in the intersection J = D ∩ {(R + x) <sup>2</sup> + y <sup>2</sup> ≤ R<sup>2</sup>} and then taking p<sup>2</sup> in the disk with the radius R and center s to guarantee that |p3| = |p<sup>2</sup> − s| ≤ R.

Case m = 4 of tetrahedra in R 3 . In R 3 , we similarly fix the center of mass at the origin and the most distant points p<sup>1</sup> at (R, 0, 0). The second most distant point p<sup>2</sup> (if not in the line through 0 and p1) forms a base sequence p1, p<sup>2</sup> and can be fixed at (x, y, 0) with x <sup>2</sup> + y <sup>2</sup> ≤ R<sup>2</sup> , which determines the mid-point p3,<sup>4</sup> p3+p<sup>4</sup> <sup>2</sup> = (− x+R 2 , − y 2 , 0). Due to the symmetry p<sup>3</sup> ↔ p<sup>4</sup> around p3,4, it remains to choose p<sup>3</sup> in the upper half ball with the center p3,<sup>4</sup> and radius p x <sup>2</sup> + y 2.

The clouds in Example [C.1](#page-23-0) are instances of C <sup>±</sup> from Example [4.5:](#page-5-7) K = C <sup>+</sup>, T = C <sup>−</sup> for <sup>4</sup><sup>a</sup> <sup>=</sup> <sup>b</sup> <sup>=</sup> <sup>c</sup> = 4<sup>d</sup> = 2√ 2 and are easy enough to write their NDPs below.

![](_page_23_Figure_9.jpeg)

![](_page_23_Diagram_10.jpeg)

Figure 17. Non-isometric clouds of 4 points with the same 6 pairwise distances. Left: the trapezoid T has points (±2, 1), (±4, −1). The kite K has (5, 0), (−3, 0), (−1, ±2).

Example C.1 (4-point clouds T, K in Fig. [17\)](#page-23-2). *Both clouds* T, K ⊂ R 2 *in Fig. [17](#page-23-2) have the center of mass at the origin.*

*(T) The cloud* T *has the points* p<sup>1</sup> = (2, 1)*,* p<sup>2</sup> = (−2, 1)*,* p<sup>3</sup> = (−4, −1)*,* p<sup>4</sup> = (4, −1)*. For the basis point* p<sup>1</sup> = (2, 1) *with* |p1| <sup>2</sup> = 5 *and orthogonal vector* p ⊥ <sup>1</sup> = (−1, 2) ⊥ p<sup>1</sup> *from Lemma [B.1,](#page-14-4) the point-based representation is* PR(T; p1) =

$$\begin{bmatrix} 5 & \left( \begin{array}{ccc} -3 & -9 & 7 \\ 4 & 2 & -6 \end{array} \right) \end{bmatrix}.$$

*1324 For* p<sup>3</sup> = (−4, −1) *with* |p3| <sup>2</sup> = 17*,* p ⊥ <sup>3</sup> = (1, <sup>−</sup>4)*, we have* PR(T; <sup>p</sup>3) = 17, −9 7 −15 <sup>−</sup><sup>2</sup> <sup>−</sup>6 8 *.*

*1329 So* NDP(T) *is the unordered set of the four* PR*s above.*

*1334 For the basis point* p<sup>1</sup> = (5, 0) *with* |p1| <sup>2</sup> = 25 *and* p ⊥ <sup>1</sup> = (0, 5) ⊥ p1*, the point-based representation is* PR(K; p1) = 25, −5 −15 −5 10 0 <sup>−</sup><sup>10</sup> *.*

*1336 For the second point* p<sup>2</sup> = (−1, 2) *with* |p2| <sup>2</sup> = 5 *and* p ⊥ <sup>2</sup> = (−2, <sup>−</sup>1)*, we have* PR(K; <sup>p</sup>2) = 5, −5 3 1 <sup>−</sup>10 6 4 *.*

*1339 1340*

*1341 1342 1343 For the point* p<sup>4</sup> = (−1, −2) *with* |p4| <sup>2</sup> = 5 *and* p ⊥ <sup>4</sup> = (2, <sup>−</sup>1)*, we have* PR(K; <sup>p</sup>4) = 5, −5 1 3 10 −4 −6 *.*

*1344 1345 So* NDP(K) *is the unordered set of the four* PR*s above.*

*1346 1347* T ̸∼= K *are distinguished by (unordered) squared distances to their centers:* 5, 5, 17, 17 *for* T*, and* 25, 5, 9, 5 *for* K*.*

*1348 1349* Example [C.2](#page-24-0) finishes the computations of the Nested Distributed Projection (NDP) for the 4-point clouds C <sup>±</sup> ⊂ <sup>R</sup> 2 in Fig. [2,](#page-2-0) which we started in Example [C.1.](#page-23-0)

*1354* Example C.2 (4-point clouds C <sup>±</sup> in Fig. [2\)](#page-2-0). *In* <sup>R</sup> 2 *, consider the 4-point clouds* C <sup>±</sup> = {p1, p2, p3, p<sup>±</sup> 4 }*, where* p<sup>1</sup> = (4a, 0)*,* p<sup>2</sup> = (b, c)*,* p<sup>3</sup> = −p<sup>2</sup> = (−b, −c)*,* p + <sup>4</sup> = (0, 4d)*, and* p − <sup>4</sup> = (0, −4d) *for parameters* a, b, c, d ≥ 0*.*

*1356 After shifting the center* O(C <sup>+</sup>) = (a, d) *to the origin* (0, 0)*, the points of* C <sup>+</sup> *become* p + <sup>1</sup> = (3a, −d)*,* p + <sup>2</sup> = (b − a, c − d)*,* p + <sup>3</sup> = (−a − b, −c − d)*,* pˆ + <sup>4</sup> = (−a, 3d)*.*

*1359 Each matrix* SD(C <sup>+</sup>; p) *is one squared distance* |p| 2 *.* <sup>2</sup> + d

*1364 1366 For the second cloud* C <sup>−</sup>*, after shifting the center* O(C <sup>−</sup>) = (a, −d) *to the origin* (0, 0)*, the points become* p − <sup>1</sup> = (3a, d)*,* p − <sup>2</sup> = (b − a, d + c)*,* p − <sup>3</sup> = (−a − b, d − c)*,* pˆ − <sup>4</sup> = (−a, −3d)*.*

*1369*

*differs from* PR(T; p1) *by the sign of the last row (up to a permutation of columns). The symmetries under* p<sup>1</sup> ↔ p<sup>2</sup> *(above) and* p<sup>3</sup> ↔ p<sup>4</sup> *(below) are explained by the reflection* (x, y) 7→ (−x, y) *mapping* T *to itself.*

*For the fourth point* p<sup>4</sup> = (4, −1) *with* |p4| <sup>2</sup> = 17*,* p ⊥ <sup>4</sup> = (1, 4)*, we have* PR(T; <sup>p</sup>4) = 17, 7 −9 −15 6 2 −8 *.*

*(K) The cloud* K *has the points* p<sup>1</sup> = (5, 0)*,* p<sup>2</sup> = (−1, 2)*,* p<sup>3</sup> = (−3, 0)*,* p<sup>4</sup> = (−1, −2)*.*

*For the third point* p<sup>3</sup> = (−3, 0) *with* |p3|

<sup>2</sup> = 9 *and* p

⊥

<sup>3</sup> = (0, <sup>−</sup>3)*, we have* PR(K; <sup>p</sup>3) =

9,  −15 3 3 <sup>0</sup> <sup>−</sup>6 6 *.*

The simultaneous swapping a ↔ d, b ↔ c maps each cloud C <sup>±</sup> to its mirror image in the diagonal x = y in <sup>R</sup> 2 , hence the metric between C <sup>±</sup> remains the same, which explains the symmetry of the top two plots in Fig. [18,](#page-25-0) [19,](#page-26-0) [20.](#page-27-0)

SD(C

<sup>+</sup>; p + 1 ) = 9a 2 ,

SD(C

<sup>+</sup>; p +

) = (a − b)

<sup>2</sup> + (c − d)

2 ,

SD(C

<sup>+</sup>; p + 3

) = (a + b)

<sup>2</sup> + (c + d)

2 ,

SD(C

<sup>+</sup>; ˆp + 4 ) = a

<sup>2</sup> + 9d 2 .

*Hence* C <sup>−</sup> *has the following squared distances to its center:*

SD(C

<sup>−</sup>; p − 1 ) = 9a

<sup>2</sup> + d 2 ,

SD(C

<sup>−</sup>; p − 2

) = (a − b)

<sup>2</sup> + (c + d)

2 ,

<sup>−</sup>; p − 3

<sup>2</sup> + (c − d)

2 ,

![](_page_25_Figure_1.jpeg)

*1414 1415 1416 1417 The (unordered) collections of squared distances above differ unless at least one of* a, b, c, d *is zero. Indeed, the squared distances* 9a <sup>2</sup>+d <sup>2</sup> *and* a <sup>2</sup>+9d <sup>2</sup> *are shared by* C <sup>±</sup> *but* SD(C <sup>+</sup>; p + 2 ) *is unique and cannot equal* SD(C <sup>−</sup>; p − 2 ) *or* SD(C <sup>−</sup>; p − 3 )*. Indeed, if all* a, b, c, d ̸= 0*, then*

*1418 1419*

*1420 1421* <sup>2</sup> ̸= (a + b) *If* d = 0*, then* p ± <sup>4</sup> = (0, 0)*, so the clouds* C <sup>±</sup> *are identical.*

*1422 1423 If* a = 0*, then* p<sup>1</sup> = (0, 0) *and* C <sup>±</sup> *are related by the* 180◦ *rotation around the origin:* (x, y) 7→ (−x, −y)*.*

*1424 1425 1426 If* b = 0 *or* c = 0*, then* C <sup>±</sup> *are related by the reflection* (x, y) 7→ (x, −y)*, so distances cannot distinguish these mirror images. We compute* NDP(C <sup>±</sup>) *below to distinguish all non-rigidly equivalent* C <sup>+</sup> ̸∼= C <sup>−</sup>*, see Fig.* ??*.*

*1427 1428 1429 For the basis point* p + 1 *, the matrix* SD(C <sup>+</sup>; p + 1 ) = 9a <sup>2</sup> + d *is the single squared distance. Lemma [B.1](#page-14-4) gives the orthogonal vector* q + <sup>1</sup> = (d, 3a) ⊥ p + 1 *.* M(C <sup>+</sup>; p + 1 ) *consists of the 3 unordered columns*

Figure 18. The Nested Bottleneck Metric NBM from Definition [4.4](#page-4-3) for the 4-point clouds C <sup>±</sup> ⊂ <sup>R</sup> <sup>2</sup> with variable parameters a, d, see details in Example [C.1.](#page-23-0)

(a − b)

<sup>2</sup> + (c − d)

<sup>2</sup> ̸= (a − b)

<sup>2</sup> + (c + d)

<sup>2</sup> *or* cd ̸= 0,

(a − b)

<sup>2</sup> + (c − d)

<sup>2</sup> + (c − d)

<sup>2</sup> *or* ab ̸= 0.

![](_page_26_Figure_1.jpeg)

*1464 1465 1466* Figure 19. The Nested Bottleneck Metric NBM from Definition [4.4](#page-4-3) for the 4-point clouds C <sup>±</sup> ⊂ <sup>R</sup> <sup>2</sup> with variable parameters b, c, see details in Example [C.1.](#page-23-0)

1469 
$$\begin{pmatrix} p_2^+ \cdot p_1^+ \\ p_2^+ \cdot q_1^+ \end{pmatrix} = \begin{pmatrix} 3a(b-a) + d(d-c) \\ d(b-a) + 3a(c-d) \end{pmatrix},$$
 1470 
$$\begin{pmatrix} p_3^+ \cdot p_1^+ \\ p_3^+ \cdot q_1^+ \end{pmatrix} = \begin{pmatrix} -3a(a+b) + d(c+d) \\ -d(a+b) - 3a(c+d) \end{pmatrix},$$
 1472 
$$\begin{pmatrix} p_3^+ \cdot p_1^+ \\ p_3^+ \cdot q_1^+ \end{pmatrix} = \begin{pmatrix} -3a(a+b) + d(c+d) \\ -d(a+b) - 3a(c+d) \end{pmatrix},$$
 1473 
$$\begin{pmatrix} \hat{p}_4^+ \cdot p_1^+ \\ \hat{p}_4^+ \cdot q_1^+ \end{pmatrix} = \begin{pmatrix} -3(a^2 + d^2) \\ 8ad \end{pmatrix}. \text{ The second point } p_2^+ = (b-a, c-d) \text{ has the orthogonal vector } q_2^+ = (d-c, b-a) \perp$$
 1474 
$$\begin{pmatrix} \hat{p}_4^+ \cdot p_1^+ \\ \hat{p}_4^+ \cdot q_1^+ \end{pmatrix} = \begin{pmatrix} -3(a^2 + d^2) \\ 8ad \end{pmatrix}. \text{ The second point } p_2^+ = (b-a, c-d) \text{ has the orthogonal vector } q_2^+ = (d-c, b-a) \perp$$
 1475 
$$p_2^+, \text{ SD}(C^+; p_2^+) = (a-b)^2 + (c-d)^2 \text{ and } M(C^+; p_2^+) \text{ consisting of the 3 unordered columns}$$
 1476 
$$\begin{pmatrix} p_1^+ \cdot p_2^+ \\ p_1^+ \cdot q_2^+ \end{pmatrix} = \begin{pmatrix} 3a(b-a) + d(d-c) \\ 3a(d-c) + d(a-b) \end{pmatrix},$$
 1477 
$$\begin{pmatrix} p_1^+ \cdot p_2^+ \\ p_1^+ \cdot q_2^+ \end{pmatrix} = \begin{pmatrix} a^2 - b^2 - c^2 + d^2 \\ 2(ac - bd) \end{pmatrix},$$
 1479 
$$\begin{pmatrix} p_3^+ \cdot p_2^+ \\ p_3^+ \cdot q_2^+ \end{pmatrix} = \begin{pmatrix} a^2 - b^2 - c^2 + d^2 \\ 2(ac - bd) \end{pmatrix},$$
 1480 
$$\begin{pmatrix} \hat{p}_4^+ \cdot p_2^+ \\ \hat{p}_4^+ \cdot q_2^+ \end{pmatrix} = \begin{pmatrix} a(a-b) + 3d(c-d) \\ a(c-d) + 3d(b-a) \end{pmatrix}. \text{ The third point } p_3^+ = (-a-b, -c-d) \text{ has the vector } q_3^+ = (c+d, -a-b) \perp$$
 1481 
$$\begin{pmatrix} \hat{p}_4^+ \cdot p_2^+ \\ \hat{p}_4^+ \cdot q_2^+ \end{pmatrix} = \begin{pmatrix} a(a-b) + 3d(c-d) \\ a(c-d) + 3d(b-a) \end{pmatrix}.$$
 1482 
$$\begin{pmatrix} \hat{p}_4^+ \cdot p_2^+ \\ \hat{p}_4^+ \cdot q_2^+ \end{pmatrix} = \begin{pmatrix} a(a-b) + 3d(c-d) \\ a(c-d) + 3d(b-a) \end{pmatrix}. \text{ The third point } p_3^+ = (-a-b, -c-d) \text{ has the vector } q_3^+ = (c+d, -a-b) \perp$$
 1483 
$$p_3^+, \text{ SD}(C^+; p_3^+) = (a+b)^2 + (c+d)^2 \text{ and } M(C^+; p_3^+) \text{ consisting of the 3 unordered columns}$$
 1484 
$$\begin{pmatrix} p_3^+ \\ p_3^+ \end{pmatrix} = \begin{pmatrix} a+b \\ c+d \end{pmatrix}$$

![](_page_27_Figure_1.jpeg)

1524 
$$\begin{pmatrix} p_1^+ \cdot p_3^+ \\ p_1^+ \cdot q_3^+ \end{pmatrix} = \begin{pmatrix} -3a(a+b) + d(c+d) \\ 3a(c+d) + d(a+b) \end{pmatrix},$$
  
 1526 
$$\begin{pmatrix} p_2^+ \cdot p_3^+ \\ p_2^+ \cdot q_3^+ \end{pmatrix} = \begin{pmatrix} a^2 - b^2 - c^2 + d^2 \\ 2(bd - ac) \end{pmatrix},$$
  
 1528 
$$\begin{pmatrix} \hat{p}_4^+ \cdot p_3^+ \\ \hat{p}_4^+ \cdot q_3^+ \end{pmatrix} = \begin{pmatrix} a(a+b) - 3d(c+d) \\ -a(c+d) - 3d(a+b) \end{pmatrix}. \text{ The fourth point } \hat{p}_4^+ = (-a, 3d) \text{ has the vector } q_4^+ = (-3d, -a) \perp p_4^+,$$
  
 1530 
$$\text{SD}(C^+; \hat{p}_4^+) = a^2 + 9d^2, M(C^+; \hat{p}_4^+) \text{ has the columns}$$
  
 1532 
$$\begin{pmatrix} p_1^+ \cdot \hat{p}_4^+ \\ p_1^+ \cdot q_4^+ \end{pmatrix} = \begin{pmatrix} -3(a^2 + d^2) \\ -8ad \end{pmatrix},$$
  
 1534 
$$\begin{pmatrix} p_2^+ \cdot \hat{p}_4^+ \\ p_2^+ \cdot q_4^+ \end{pmatrix} = \begin{pmatrix} a(a-b) + 3d(c-d) \\ 3d(a-b) + a(d-c) \end{pmatrix},$$
  
 1535 
$$\begin{pmatrix} p_3^+ \cdot \hat{p}_4^+ \\ p_3^+ \cdot q_4^+ \end{pmatrix} = \begin{pmatrix} a(a+b) - 3d(c+d) \\ 3d(a+b) + a(c+d) \end{pmatrix}. \text{ The Nested Distributed Projection NDP}(C^+) \text{ consists of the four pairs (of } a \\ 1538 \text{ squared distance and } 2 \times 3 \text{ matrix) above.}$$
  
 1539

Figure 20. The Nested Bottleneck Metric NBM from Definition [4.4](#page-4-3) for the 4-point clouds C <sup>±</sup> ⊂ <sup>R</sup> <sup>2</sup> with variable parameters a, c, see details in Example [C.1.](#page-23-0)

*1540 1541 1542 1543 1544 1545 1546 1547 1548 1549 1554 1556 1559 1560 1564 1569 1574 1576 1579 1584 1589 1590 1591 1594 For* C <sup>−</sup>*, after shifting the center* O(C <sup>−</sup>) = (a, −d) *to the origin* (0, 0)*, the points of* C <sup>−</sup> *become* p − <sup>1</sup> = (3a, d)*,* p − <sup>2</sup> = (b − a, d + c)*,* p − <sup>3</sup> = (−a − b, d − c)*,* pˆ − <sup>4</sup> = (−a, −3d)*. The first point* p − 1 *has the vector* q − <sup>1</sup> = (−d, 3a) ⊥ p − 1 *,* SD(C <sup>−</sup>; p − 1 ) = 9a <sup>2</sup> + d 2 *,* M(C <sup>−</sup>; p − 1 ) *has the columns* p − 2 · p − 1 p − · q − 1 = 3a(b − a) + d(d + c) d(a − b) + 3a(d + c) *,* p − 3 · p − 1 p − 3 · q − 1 = −3a(b + a) + d(d − c) d(b + a) + 3a(d − c) *,* pˆ − 4 · p − 1 pˆ − 4 · q − 1 = −3(a <sup>2</sup> + d ) <sup>−</sup>8ad *. The second point* p − <sup>2</sup> = (b − a, d + c) *has the vector* q − <sup>2</sup> = (−d − c, b − a) ⊥ p − 2 *,* SD(C <sup>−</sup>; p − 2 ) = (a − b) <sup>2</sup> + (c + d) 2 *,* M(C <sup>−</sup>; p − 2 ) *of* p − 1 · p − 2 p − 1 · q − 2 = 3a(b − a) + d(d + c) −3a(c + d) + d(b − a) *,* p − 3 · p − 2 p − 3 · q − 2 = a <sup>2</sup> − b <sup>2</sup> − c <sup>2</sup> + d 2 2(ac + bd) *,* pˆ − 4 · p − 2 pˆ − 4 · q − 2 = a(a − b) − 3d(c + d) a(c + d) + 3d(a − b) *. The third point* p − <sup>3</sup> = (−a − b, d − c) *has* q − <sup>3</sup> = (c − d, −a − b) ⊥ p − 3 *,* SD(C <sup>−</sup>; p − 3 ) = (a + b) <sup>2</sup> + (c − d) 2 *,* M(C <sup>−</sup>; p − 3 ) *of* p − · p − 3 p − 1 · q − 3 = −3a(a + b) + d(d − c) 3a(c − d) − d(a + b) *,* p − 2 · p − 3 p − · q − 3 = a <sup>2</sup> − b <sup>2</sup> − c <sup>2</sup> + d 2 −2(ac + bd) *,* pˆ − 4 · p − 3 pˆ − 4 · q − 3 = a(a + b) + 3d(c − d) a(d − c) + 3d(a + b) *. The fourth point* pˆ − <sup>4</sup> = (−a, −3d) *has* q − <sup>4</sup> = (3d, −a) ⊥ pˆ − 4 *,* SD(C <sup>−</sup>; ˆp − 4 ) = a <sup>2</sup> + 9d 2 *,* M(C <sup>−</sup>; ˆp − 4 ) *consisting of* p − 1 · pˆ − 4 p − · q − 4 = −3(a <sup>2</sup> + d 2 ) <sup>8</sup>ad *,* p − 2 · pˆ − 4 p − 2 · q − 4 = a(a − b) − 3d(d + c) 3d(b − a) − a(d + c) *,* p − 3 · pˆ − 4 p − 3 · q − 4 = a(a + b) + 3d(c − d) −3d(a + b) + a(c − d) *. The Nested Distributed Projection* NDP(C <sup>−</sup>) *consists of the four pairs (of a squared distance and* 2 × 3 *matrix) above. Shorter Example [C.1](#page-23-0) justified that* C <sup>+</sup> ̸∼= C <sup>−</sup> *unless at least of the parameters* a, b, c, d *is 0. If* a = 0 *or* d = 0*, then* C <sup>+</sup> ∼= C <sup>−</sup> *are isometric. In the remaining cases* b = 0 *and* c = 0*, the clouds* C <sup>±</sup> *are mirror images, which can be distinguished by matrices* M *above, not by any distances. Case* b = 0*. We write down the above matrices* M(C <sup>+</sup>; p + i ) *with unordered columns after substituting* b = 0*.* −3a <sup>2</sup> + d(d − c) −3a <sup>2</sup> + d(d + c) −3(a <sup>2</sup> + d 2 ) <sup>a</sup>(3<sup>c</sup> <sup>−</sup> <sup>4</sup>d) <sup>−</sup>a(3<sup>c</sup> + 4d) 8ad −3a <sup>2</sup> + d(d − c) a <sup>2</sup> − c <sup>2</sup> + d <sup>2</sup> a <sup>2</sup> + 3d(c − d) a(4d − 3c) 2ac a(c − 4d) −3a <sup>2</sup> + d(d + c) a <sup>2</sup> − c <sup>2</sup> + d <sup>2</sup> a <sup>2</sup> − 3d(c + d) a(3c + 4d) −2ac −a(c + 4d) −3(a <sup>2</sup> + d 2 ) a <sup>2</sup> + 3d(c − d) a <sup>2</sup> − 3d(c + d) −8ad a(4d − c) a(c + 4d) *The mirror image* C <sup>−</sup> *has the following matrices:* −3a <sup>2</sup> + d(d + c) −3a <sup>2</sup> + d(d − c) −3(a <sup>2</sup> + d ) <sup>a</sup>(3<sup>c</sup> + 4d) <sup>a</sup>(4<sup>d</sup> <sup>−</sup> <sup>3</sup>c) <sup>−</sup>8ad

$$\begin{array}{lll}
 1955 & \left( -3a^2 + d(d + c) - a^2 - c^2 + d^2 - a^2 - 3d(c + d) \right) \\
 1956 & \left( -a(3c + 4d) - 2ac - a(c + 4d) \right) \\
 1957 & \left( -3a^2 + d(d - c) - a^2 - c^2 + d^2 - a^2 + 3d(c - d) \right) \\
 1958 & \left( a(3c - 4d) - 2ac - a(4d - c) \right) \\
 1959 & \\
 1600 & \left( -3(a^2 + d^2) - a^2 - 3d(c + d) - a^2 + 3d(c - d) \right) \\
 1601 & \left( 8ad - a(c + 4d) - a(c - 4d) \right) \\
 1602 & \\
 \end{array}$$

1602 1603 1604 1605 1606 1607 *By Lemma [B.3\(](#page-15-1)b), the reflection* C <sup>+</sup> → C <sup>−</sup> *changes the sign of the last row in the matrix* M *from any point-based representation* PR*. Indeed, changing the sign of the last row in each matrix* M *from* NDP(C <sup>+</sup>) *makes this matrix identical to one of the matrices from* NDP(C <sup>−</sup>)*, up to a permutation of columns as always. However, with all signs kept, the above unordered collections of four matrices are different unless all elements in the last row vanish, which happens only for a=0, when* C <sup>+</sup> = C<sup>−</sup> *are identical.*

1608 1609 *Case* c = 0 *is symmetric to the case* c = 0 *under the reflection* (x, y) 7→ (y, x)*, which swaps* b ↔ c *and* a ↔ d*.*

1614 1616 A numerical experiment can only illustrate but not prove the conclusion of Example [C.2](#page-24-0) that all (infinitely many) non-rigidly equivalent clouds C <sup>±</sup> are distinguished by NDP.

### 1618 D. Generalization of section [4](#page-4-6) and all proofs in dimensions n ≥ 2

1619 This appendix extends the metrics to dimensions n ≥ 2 and proves all metric results from section [4](#page-4-6) in full generality.

1624 1626 Below we can take any norm on matrices and choose the simplest max norm below for consistency with the bottleneck distance and for Lipschitz constant 2 in Theorem [E.5.](#page-35-0)

1629 Definition D.1 (max norm and metric on matrices). *The* max norm ||D||<sup>∞</sup> = max i,j |Dij | *of a matrix is the maximum absolute value of its elements* Dij *. The* max metric *between matrices* M, M′ *of the same size is* d<sup>∞</sup> = ||M − M′ ||∞*.*

1634 1636 Definition [D.2](#page-29-1) will extend Definition [4.2](#page-4-1) to dimensions n ≥ 2. Below the notation SD/R means that all elements of a matrix SD are divided by R. The radius of a base sequence p{n − 1} = (p1, . . . , pn−1) ⊂ A is defined as R(p{n − 1}) = max i=1,...,n−1 |p<sup>i</sup> | in the same way as R(A) of a full cloud A. The notation M/R means that all elements in the first n − 1 rows of a matrix M are divided by R, and by R<sup>n</sup>−<sup>1</sup> in the n-th row, because p ⊥ n in Lemma [B.1](#page-14-4) is a polynomial of degree n − 1. Then PRM and further metrics have units of original points. One more division by R makes all metrics invariant under scaling.

1639 1640 Definition D.2 (Point-Based Representation Metric). *Let clouds* A, B ⊂ R <sup>n</sup> *of* m *unordered points have base sequences* p{n − 1} = (p1, . . . , pn−1)*,* q{n − 1} = (q1, . . . , qn−1) *of ordered points, from Definition [B.2.](#page-15-0) The* Point-Based Representation Metric *between the* PR*s above is*

$$\text{PRM} = \max\{|R(p\{n-1\}) - R(q\{n-1\})|, w_D, |R(A) - R(B)|, w_M\}$$
, where

1647 1648 1649 Lemma D.3 (axioms for PRM). PRM *in Definition [D.2](#page-29-1) satisfies all metric axioms from Problem [\(1.1b](#page-0-1)) on any point-based representations from Definition [B.8.](#page-19-1)*

*We have considered only non-negative values of* a, b, c, d *because all other cases are obtained by symmetries. For example, the reflection* y 7→ −y *maps the cloud* C <sup>+</sup>(a, b, c, d) *to* C <sup>−</sup>(a, −b, c, d) = C <sup>−</sup>(a, b, −c, d)*.*

Example [C.2](#page-24-0) importantly demonstrates that the invariant NDP is simple enough for manual computations.

The point-based representation in Definition [B.2](#page-15-0) included the matrix SD(p1, . . . , pn−1) of squared distances, which can be rewritten as a vector row-by-row.

$$\text{PRM} = \max\{|R(p\{n-1\}) - R(q\{n-1\})|, w_D, |R(A) - R(B)|, w_M\}, \text{ where } w_D = d_\infty \left( \frac{\text{SD}(p\{n-1\})}{R(p\{n-1\})}, \frac{\text{SD}(q\{n-1\})}{R(q\{n-1\})} \right), \text{ and } w_M = \text{BD} \left( \frac{M(A; p\{n-1\})}{R(A)}, \frac{M(B; q\{n-1\})}{R(B)} \right).$$

1654 1656 The part *only if* : by Definition [D.2](#page-29-1) the equality PRM = 0 means that R(A) = R(B) and w<sup>D</sup> = 0 = wM. The coincidence axioms for the max metric and bottleneck distance together with R(p{n − 1}) = R(q{n − 1}) and R(A) = R(B) imply that SD(p{n − 1}) = SD(q{n − 1}) and M(A; p{n − 1}) = M(B; q{n − 1}). Then the point-based representations become identical: PR(A; p{n − 1}) = PR(B; q{n − 1}).

1659 1660 1661 1662 The symmetry axiom for PRM follows from the symmetry axiom for the bottleneck distance and max metric d∞. Since each of the distances |R(A) − R(B)|, wD, w<sup>M</sup> satisfies the triangle inequality, then so does their maximum, see metric transforms in section 4.1 of [\(Deza & Deza,](#page-9-26) [2009\)](#page-9-26).

1663 Definition [D.4](#page-30-0) extends Definition [4.4](#page-4-3) to all dimensions n > 2.

1664 1665 1666 1667 1668 Definition D.4 (NBM : Nested Bottleneck Metric). *Let* A, B ⊂ R <sup>n</sup> *be any clouds of* m *unordered points. For any ordered points* p<sup>1</sup> . . . , pn−<sup>2</sup> ∈ A *and* q<sup>1</sup> . . . , qn−<sup>2</sup> ∈ B*, the complete bipartite graph* Γ(A; p1, . . . , pn−2; B; q1, . . . , qn−2) *has* m − n + 2 *white vertices and* m − n + 2 *black vertices representing* PR(A; p1, . . . , pn−1) *and* PR(B; q1, . . . , qn−1) *for all* m − n + 1 *variable points* pn−<sup>1</sup> ∈ A \ {p1, . . . , pn−2} *and* qn−<sup>1</sup> ∈ B − {q1, . . . , qn−2}*, respectively.*

1669 1674 *Set the* weight w(e) *of an edge* e *joining the vertices represented by* PR(A; p1, . . . , pn−1) *and* PR(B; q1, . . . , qn−1) *as* PRM *between these* PR*s, see Definition [D.2.](#page-29-1) Then Definition [4.3](#page-4-2) gives us the bottleneck matching distance* BMD(Γ(A; p1, . . . , pn−2; B; q1, . . . , qn−2))*. We continue dropping points iteratively. For any* 1 ≤ k ≤ n − 2 *and ordered points* p<sup>1</sup> . . . , pk−<sup>1</sup> ∈ A *and* q<sup>1</sup> . . . , qk−<sup>1</sup> ∈ B*, the complete bipartite graph* Γ(A; p1, . . . , pk−1; B; q1, . . . , qk−1) *has* m − k + 1 *white vertices and* m − k + 1 *black vertices representing* NDP(A; p1, . . . , pk) *and* NDP(B; q1, . . . , qk) *for all* m − k + 1 *variable points* p<sup>k</sup> ∈ A \ {p1, . . . , pk−1} *and* q<sup>k</sup> ∈ B − {q1, . . . , qk−1}*, respectively.*

1676 1679 *Set the* weight w(e) *of an edge* e *joining the vertices represented by* NDP(A; p1, . . . , pk) *and* NDP(B; q1, . . . , qk) *as* BMD(Γ(A; p1, . . . , pk; B; q1, . . . , qk)) *obtained above. Then Definition [4.3](#page-4-2) gives us the bottleneck matching distance* BMD(Γ(A; p1, . . . , pk−1; B; q1, . . . , qk−1))*. Finally, for* k = 1*, we get the* Nested Bottleneck Metric NBM(A, B) = BMD(Γ(A, B))*.*

1681 1682 1683 1684 Lemma D.5 (metric axioms for the bottleneck matching distance BMD). *Let* S, Q *be any unordered distributions of the same number of objects with a base metric* d*. Define the complete bipartite graph* Γ(S, Q) *whose every edge* e *joining objects* R<sup>S</sup> ∈ S *and* R<sup>Q</sup> ∈ Q *has the weight* w(e) = d(RS, RQ)*. Then the bottleneck matching distance* BMD(Γ(S, Q)) *from Definition [4.3](#page-4-2) satisfies all metric axioms on such unordered distributions.*

1685 1686 1687 *Proof of Lemma [D.5.](#page-30-1)* The coincidence axiom means that NBM(S, Q) = 0 if and only if the weighted distributions S, Q are equal in the sense that there is a bijection g : S → Q so that d(g(R), R) = 0 for any R ∈ S.

1689 1690 Indeed, if the weighted distributions S, Q can be matched by a bijection, we get a vertex matching E of Γ(S, Q) whose all edges have weights w(e) = 0. Definition [4.3](#page-4-2) implies that BMD(Γ(S, Q)) = 0 as required.

1694 To prove the triangle inequality

1696

1699 let ESQ, EQT be optimal vertex matchings in the graphs Γ(S, Q), Γ(Q, T), respectively, such that

1700

*Proof of Lemma [D.3.](#page-29-2)* The first axiom means that PRM(PR(A; p{n − 1}),PR(B; q{n − 1})) = 0 if and only if these PRs are identical. The part *if* : by Lemma [B.5\(](#page-17-0)c), equal PRs guarantee that the clouds A, B are rigidly equivalent, so R(p{n − 1}) = R(q{n − 1}), R(A) = R(B), SD(p{n − 1}) = SD(q{n − 1}), and M(A; p{n − 1}) = M(B; q{n − 1}), so PRM = 0.

Conversely, if BMD(Γ(S, Q)) = 0, there is a vertex matching E in Γ(S, Q) with all w(e) = 0. This matching E defines a required bijection S → Q. The symmetry BMD(Γ(S, Q)) = BMD(Γ(Q, S)) follows from Definition [4.3](#page-4-2) and the symmetry of the base metric d.

$$\text{BMD}(\Gamma(S, Q)) + \text{BMD}(\Gamma(Q, T)) \geq \text{BMD}(\Gamma(S, T)),$$

$$\text{BMD}(\Gamma(S, Q)) = W(E_{SQ}), \text{BMD}(\Gamma(Q, T)) = W(E_{QT}),$$

see Definition [4.3.](#page-4-2) The composition ESQ ◦ EQT is a vertex matching in Γ(S, T), so W(ESQ ◦ EQT ) ≥ BMD(Γ(S, T)). It suffices to prove that

$$W(E_{SQ}) + W(E_{QT}) \geq W(E_{SQ} \circ E_{QT}).$$

1706 Let eST be an edge with a largest weight from ESQ ◦EQT , so W(ESQ ◦EQT ) = w(eST ). The edge eST can be considered the union of edges eSQ ∈ ESQ, eQT ∈ EQT .

1709

1714 because both terms on the left-hand side are maximized for all edges (not only eSQ, eQT ) from ESQ, EQT .

1716 Lemma D.6 (metric axioms for NBM between NDPs). *The Nested Bottleneck Metric* NBM *from Definition [D.4](#page-30-0) satisfies all metric axioms on Nested Distributed Projections.*

1719 *Proof of Lemma [D.6.](#page-31-1)* Induction on k = n − 2, . . . , 1. The inductive base k = n − 2 follows from the metric axioms in Lemma [D.3](#page-29-2) for PRM in Definition [D.2.](#page-29-1) The inductive step from 1 < k < n − 2 to k − 1 follows from Lemma [D.5](#page-30-1) and the metric axioms in the inductive hypothesis for k.

1724

1726

1729 Lemma E.1 (orthogonal vector length). *For any sequence* p1, . . . , pn−<sup>1</sup> ∈ <sup>R</sup> <sup>n</sup>*, set* R = max i=1,...,n−1 |p<sup>i</sup> |*. Then the orthogonal vector* p ⊥ <sup>n</sup> ⊥ p1, . . . , pn−<sup>1</sup> *from Lemma [B.1](#page-14-4) has a length satisfying* |p ⊥ 2 | = R*,* |p ⊥ 3 | ≤ R<sup>2</sup> *, and* |p ⊥ n | ≤ √ nR<sup>n</sup>−<sup>1</sup> *for any* n > 3*.*

1734 *Proof of Lemma [E.1.](#page-31-2)* For n = 2, the explicit formula p ⊥ <sup>2</sup> = (−y, x) for p<sup>1</sup> = (x, y) gives the exact equality |p ⊥ 2 | = |p1| = R. For n = 3, p ⊥ 3 equals the vector product p<sup>1</sup> × p<sup>2</sup> whose length is |p ⊥ 3 | ≤ |p1| · |p2| ≤ R<sup>2</sup> . For > 3, the expansion

1736

1739 1740 det(i) is the (n − 1) × (n − 1) determinant obtained from the n − 1 vector columns p1, . . . , pn−<sup>1</sup> by removing the row of all i-th coordinates. Any determinant on vectors v1, . . . , vn−<sup>1</sup> ∈ <sup>R</sup> n−1 equals the signed volume of the parallelepiped on v1, . . . , vn−1, which has the upper bound |v1| · · · |vn−1|.

1741 1742 1743 1744 Since each vector v<sup>i</sup> is obtained from p<sup>i</sup> by removing one coordinate, we get |v<sup>i</sup> | ≤ |p<sup>i</sup> |. So each coordinate of p ⊥ n in the orthonormal basis e1, . . . , e<sup>n</sup> has the upper bound |p1| · · · |pn−1| ≤ R<sup>n</sup>−<sup>1</sup> . Then the Euclidean length has the upper bound |p ⊥ n | ≤ p n(R<sup>n</sup>−<sup>1</sup>) <sup>2</sup> = √ nR<sup>n</sup>−<sup>1</sup> .

1745 1746 1747 1748 1749 Lemma E.2 (vector perturbations). *Let points* q1, . . . , qn−<sup>1</sup> *be* ε*-perturbations of* p1, . . . , pn−<sup>1</sup> ∈ <sup>R</sup> <sup>n</sup> *so that* |p<sup>i</sup> − q<sup>i</sup> | ≤ ε *for any* i = 1, . . . , n−1*. Set* R = max i=1,...,n−1 {|p<sup>i</sup> |, |q<sup>i</sup> |}*. The orthogonal vectors* p ⊥ <sup>n</sup> ⊥ p1, . . . , pn−<sup>1</sup> *and* q ⊥ <sup>n</sup> ⊥ q1, . . . , qn−<sup>1</sup> *from Lemma [B.1](#page-14-4) satisfy* |p ⊥ <sup>2</sup> − q ⊥ 2 | ≤ ε *for* n = 2*,* |p ⊥ <sup>3</sup> − q ⊥ 3 | ≤ ε2 √ 6R *for* n = 3*, and* |p ⊥ <sup>n</sup> − q ⊥ n | ≤ εn(n − 1)R<sup>n</sup>−<sup>2</sup> *for any* n > 3*.*

1754 Let xi(v<sup>j</sup> ) be the i-th coordinate of a variable vector v<sup>j</sup> ∈ <sup>R</sup> <sup>n</sup> moving from p<sup>j</sup> to its ε-perturbation q<sup>j</sup> for i, j = 1, . . . , n in the given orthonormal basis e1, . . . , en, where we set p<sup>n</sup> = p ⊥ n and q<sup>n</sup> = q ⊥ n for brevity. For each k = 1, . . . , n, the coordinate xk(vn) is the scalar function fk(v1, . . . , vn−1) of the (n − 1)<sup>2</sup> variables xi(v<sup>j</sup> ) for i, j = 1, . . . , n − 1.

1756 1759 The upper bound for |p<sup>n</sup> − qn| will follow from the Mean Value Theorem 5.10 from [\(Rudin et al.,](#page-10-25) [1976\)](#page-10-25) for the functions f1, . . . , f<sup>n</sup> because the coordinates of the vector q ⊥ n are fk(q1, . . . , qn−1) evaluated at close (coordinates of the) vectors q1, . . . , qn−<sup>1</sup> so that |p<sup>j</sup> − q<sup>j</sup> | ≤ ε for i, j = 1, . . . , n − 1.

By the triangle inequality for the base metric d,

$$w(e_{SQ}) + w(e_{QT}) \geq w(e_{ST}) = W(E_{SQ} \circ E_{QT})$$

implies that

$$W(E_{SQ}) + W(E_{QT}) \geq W(E_{SQ} \circ E_{QT})$$

# E. Generalization of section [5](#page-5-6) and all proofs

This appendix proves Theorems [E.5,](#page-35-0) [E.8,](#page-36-0) and [E.9](#page-36-1) extending Lemmas [5.1,](#page-5-2) [5.2,](#page-5-5) and [5.3,](#page-5-0) respectively to dimensions n ≥ 2 by using auxiliary Lemmas [E.1,](#page-31-2) [E.2,](#page-31-3) [E.4,](#page-35-1) and Proposition [E.3.](#page-32-0)

of the n × n determinant p ⊥ <sup>n</sup> = | . . . | e<sup>1</sup> p<sup>1</sup> . . . pn−<sup>1</sup> . . . | . . . | e<sup>n</sup> along the last column gives p ⊥ <sup>n</sup> = Pn i=1 (−1)<sup>n</sup>+<sup>i</sup> det(i)e<sup>i</sup> , where

*Proof of Lemma [E.2.](#page-31-3)* If n = 2, then p ⊥ <sup>2</sup> = (−y, x) for p<sup>1</sup> = (x, y), so |p ⊥ <sup>2</sup> − q ⊥ 2 | = |p<sup>1</sup> − q1| ≤ ε.

1764 v<sup>n</sup> = | . . . | e<sup>1</sup> v<sup>1</sup> . . . vn−<sup>1</sup> . . . | . . . | e<sup>n</sup> is (−1)<sup>n</sup>+<sup>k</sup> det(k), where det(k) is the (n − 1) × (n − 1) determinant obtained from the

1766 <sup>n</sup>−<sup>1</sup> vector columns <sup>v</sup>1, . . . , vn−<sup>1</sup> by removing the row of all <sup>k</sup>-th coordinates. Then ∂f<sup>k</sup> ∂xi(v<sup>j</sup> ) = (−1)<sup>n</sup>+<sup>k</sup> ∂ det(k) ∂xi(v<sup>j</sup> ) , which equals 0 for k = i because f<sup>k</sup> is independent of the coordinate xk(v<sup>j</sup> ) for j = 1, . . . , n − 1.

1769

1774 1776 Then |v<sup>j</sup> | ≤ R = max i=1,...,n−1 {|p<sup>i</sup> |, |q<sup>i</sup> |} for any points (v1, . . . , vn−1) in the line segment between (p1, . . . , pn−1) and (q1, . . . , qn−1). The (n − 2) × (n − 2) determinant det(k, i) equals the signed volume on n − 2 vectors of maximum length R and hence has the upper bound R<sup>n</sup>−<sup>2</sup> , so ∂f<sup>k</sup> ∂xi(v<sup>j</sup> )  <sup>=</sup> <sup>|</sup> det(k, i)| ≤ <sup>R</sup><sup>n</sup>−<sup>2</sup> . The gradient ∇f<sup>k</sup> is the vector of (n − 1)<sup>2</sup>

1779

1790

1794

1796

1799 1800 1801

1802 1803 for any n ≥ 3. If n = 3, the final upper bound can be improved to ε2 √ 6R.

1810 *(b)* PRM PR(A; p{n − 1}),PR(B; q{n − 1}) ≤ λnε *for* λ<sup>2</sup> = 6*,* λ<sup>3</sup> = 16*,* λ<sup>n</sup> = 3n 2 *,* n > 3*.*

First we estimate the gradient ∇f<sup>k</sup> of f<sup>k</sup> at any intermediate point in the line segment between (p1, . . . , pn−1) and (q1, . . . , qn−1) with respect to the (n − 1)<sup>2</sup> variables xi(v<sup>j</sup> ) for i, j = 1, . . . , n − 1. For k = i, the k-th coordinate of

After expanding the determinant det(k) along the i-th row, the only terms containing the factor xi(v<sup>j</sup> ) form the smaller (n−2)×(n−2) determinant det(k, i) obtained from the n−2 vector columns v1, . . . , vj−1, vj+1, . . . , vn−<sup>1</sup> after removing the rows of all k-th and i-th coordinates.

partial derivatives and can be considered a vector (∇1fk, . . . , ∇n−1fk), where ∇jf<sup>k</sup> = ∂f<sup>k</sup> x1(v<sup>j</sup> ) , . . . , ∂f<sup>k</sup> xn−1(v<sup>j</sup> ) has

$$|\nabla_j f_k| \leq \sqrt{n-1} \max_{i=1,\dots,n-1} \left| \frac{\partial f_k}{\partial x_i(v_j)} \right| \leq \sqrt{n-1} R^{n-2}.$$

We consider the k-th coordinate f<sup>k</sup> of v<sup>n</sup> as a function depending on one parameter t ∈ [0, 1] when the point (v1, . . . , vn−1) moves along the line segment from (p1, . . . , pn−1) to (q1, . . . , qn−1). Then Theorem 5.10 from [\(Rudin et al.,](#page-10-25) [1976\)](#page-10-25) implies for some intermediate point (v1, . . . , vn−1) that

$$\begin{aligned} |f_k(p_1, \dots, p_{n-1}) - f_k(q_1, \dots, q_{n-1})| &= |\nabla f_k(v_1, \dots, v_{n-1}) \cdot (p_1 - q_1, \dots, p_{n-1} - q_{n-1})| = \\ &= \left| \sum_{i,j=1}^{n-1} \frac{\partial f_k}{\partial x_i(v_j)} \cdot (x_i(p_j) - x_i(q_j)) \right| = \left| \sum_{j=1}^{n-1} \nabla_j f_k \cdot (p_j - q_j) \right| \leq \sum_{j=1}^{n-1} |\nabla_j f_k| \cdot |p_j - q_j| \leq \\ &\leq \varepsilon(n-1) \max_{j=1,\dots,n-1} |\nabla_j f_k| \leq \varepsilon(n-1)\sqrt{n-1}R^{n-2}. \end{aligned}$$

Since e1, . . . , e<sup>n</sup> form an orthonormal basis, we get

$$\begin{aligned} |p_n^\perp - q_n^\perp| &= \sqrt{\sum_{k=1}^n |f_k(p_1, \dots, p_{n-1}) - f_k(q_1, \dots, q_{n-1})|^2} \\ &\leq \sqrt{n} \max_{k=1, \dots, n} |f_k(p_1, \dots, p_{n-1}) - f_k(q_1, \dots, q_{n-1})| \leq \sqrt{n} \varepsilon (n-1) \sqrt{n-1} R^{n-2} \leq \varepsilon n (n-1) R^{n-2} \end{aligned}$$

Proposition E.3 (Lipschitz continuity of PR under perturbations of a cloud). *Let* B ⊂ R <sup>n</sup> *and a base sequence* q{n−1} ⊂ B *be obtained from a cloud* A ⊂ R <sup>n</sup> *and a base sequence* p{n − 1} ⊂ A*, respectively, by perturbing every point in its Euclidean* ε*-neighborhood. Then*

$$(a) |O(A) - O(B)| \leq \varepsilon, |R(p\{n-1\}) - R(q\{n-1\})| \leq 2\varepsilon, \text{ and } |R(A) - R(B)| \leq 2\varepsilon;$$

*Proof of Proposition [E.3.](#page-32-0)* (a) Let p<sup>1</sup> . . . , p<sup>m</sup> be all points of A so that the first n − 1 points p1, . . . , pn−<sup>1</sup> form the base sequence p{n − 1}. Let q<sup>i</sup> ∈ B be an ε-perturbation of p<sup>i</sup> , so q<sup>1</sup> . . . , q<sup>m</sup> are all points of B and the first n − 1 points

1841 1842 1843 for i, j = 0, . . . , n − 1, where p<sup>0</sup> = O(A) and q<sup>0</sup> = O(B) are centers of mass. In the first term above, we estimate the difference of squares by factorizing:

q1, . . . , qn−<sup>1</sup> form the base sequence q{n − 1}. The radius of A is R(A) = max p∈A <sup>|</sup><sup>p</sup> <sup>−</sup> <sup>O</sup>(A)|, where <sup>O</sup>(A) = <sup>1</sup> m P p∈A p is the center of mass. Then

$$|O(A) - O(B)| = \frac{1}{m} \left| \sum_{i=1}^m p_i - \sum_{i=1}^m q_i \right| \leq \frac{1}{m} \sum_{i=1}^m |p_i - q_i| \leq \varepsilon.$$

If the radius R(A) is attained at a point p<sup>i</sup> ∈ A, then R(A) = |p<sup>i</sup> − O(A)| ≤

$$\leq |p_i - q_i| + |q_i - O(B)| + |O(B) - O(A)| \leq \varepsilon + \max_{i=1,\dots,m} |q_i - O(B)| + \varepsilon = 2\varepsilon + R(B).$$

Swapping the clouds A, B gives the opposite inequality R(B) ≤ 2ε + R(A), so |R(A) − R(B)| ≤ 2ε. The radii of the base sequences also differ by at most 2ε, i.e. |R(p{n − 1}) − R(q{n − 1})| ≤ 2ε.

(b) All corresponding points of the given clouds A, B are ε-close so that |p<sup>i</sup> − q<sup>i</sup> | ≤ ε for all i = 1, . . . , m. Any distance |p<sup>i</sup> − p<sup>j</sup> | changes by at most 2ε under perturbation, because

$$\begin{aligned} |p_i - p_j| &\leq |p_i - q_i| + |q_i - q_j| + |q_j - p_j| \leq |q_i - q_j| + 2\varepsilon, \\ |q_i - q_j| &\leq |q_i - p_i| + |p_i - p_j| + |p_j - q_j| \leq |p_i - p_j| + 2\varepsilon. \end{aligned}$$

Hence |p<sup>i</sup> − p<sup>j</sup> | − |q<sup>i</sup> − q<sup>j</sup> |  ≤ 2ε for all i, j = 1, . . . , m.

To estimate the max metric d<sup>∞</sup> in [\(D.2\)](#page-29-1), we rewrite the difference between the corresponding elements in the matrices SD/R of squared distances normalized by the radii in the notations r(A) = R(p{n − 1}) and r(B) = R(q{n − 1}). Without loss of generality, assume that r(A) ≥ r(B).

$$\text{Then } \left| \frac{|p_i - p_j|^2}{r(A)} - \frac{|q_i - q_j|^2}{r(B)} \right| \leq \frac{||p_i - p_j|^2 - |q_i - q_j|^2|}{r(A)} + |q_i - q_j|^2 \frac{|r(B) - r(A)|}{r(A)r(B)}$$

$$||p_i - p_j|^2 - |q_i - q_j|^2| = ||p_i - p_j| - |q_i - q_j|| \cdot (|p_i - p_j| + |q_i - q_j|) \leq 2\varepsilon(2r(A) + 2r(B)).$$

Using <sup>r</sup>(A) <sup>≥</sup> <sup>r</sup>(B), the bounds | |p<sup>i</sup> <sup>−</sup> <sup>p</sup><sup>j</sup> <sup>|</sup> <sup>2</sup> − |q<sup>i</sup> − q<sup>j</sup> | 2 r(A) ≤ 4ε r(A) + r(B) r(A) ≤ 8ε, |q<sup>i</sup> − q<sup>j</sup> | 2 |r(B) − r(A)| r(A)r(B) ≤ (2r(B))<sup>2</sup> · 2ε r(A)r(B) ≤ 8ε give d<sup>∞</sup> SD(p{n − 1}) r(A) , SD(q{n − 1}) r(B) ≤ 16ε.

To estimate the bottleneck distance BD between the matrices M/R in [\(D.2\)](#page-29-1), which involve scalar products, we shift both clouds A, B so that their centers O(A) and O(B) coincide with the origin 0 ∈ <sup>R</sup> <sup>n</sup>. We keep the same notation p<sup>i</sup> , q<sup>i</sup> for all points for simplicity. Since |O(A) − O(B)| ≤ ε by part (a), the relative shift by a vector of a maximum length ε guarantees all corresponding points are now 2ε-close, i.e. |p<sup>i</sup> − q<sup>i</sup> | ≤ 2ε. Below we estimate the difference between scalsr products involving any 2ε-close points p ∈ A \ p{n − 1} and q ∈ B − q{n − 1} for i = 1, . . . , n − 1 (indexing points from the base sequences) and i = n for the orthogonal vectors p<sup>n</sup> = p ⊥ n , q<sup>n</sup> = q ⊥ n .

Case i = 1, . . . , n − 1. The bottleneck distance BD has the upper bound obtained from estimating the differences below in the M/R matrices for any point p ∈ A \ p{n − 1} matched with its 2ε-perturbation q ∈ B − q{n − 1}. Without loss of generality, assume that R(A) ≥ R(B). Then

$$\left| \frac{p \cdot p_i}{R(A)} - \frac{q \cdot q_i}{R(B)} \right| \leq \frac{|p \cdot p_i - q \cdot q_i|}{R(A)} + |q \cdot q_i| \frac{|R(B) - R(A)|}{R(A)R(B)}.$$

Due to |q · q<sup>i</sup> | ≤ |q| · |q<sup>i</sup> | ≤ R<sup>2</sup> (B), the second term above has the upper bound <sup>R</sup><sup>2</sup> (B) · 2ε R(A)R(B) ≤ 2ε. Estimate the difference of products in the first term above:

$$|p \cdot p_i - q \cdot q_i| \leq |(p-q) \cdot p_i + q \cdot (p_i - q_i)| \leq |p-q| \cdot |p_i| + |q| \cdot |p_i - q_i| \leq 2\varepsilon(R(A) + R(B)).$$

1906 1907 so PRM PR(A; p{2}),PR(B; q{2}) ≤ 16ε which finishes the proof of part (b) for n = 3.

1908 1909

1914

1916

1918 1919

Then <sup>|</sup><sup>p</sup> · <sup>p</sup><sup>i</sup> <sup>−</sup> <sup>q</sup> · <sup>q</sup><sup>i</sup> R(A) ≤ 2ε R(A) + R(B) R(A) = 4ε. For every i = 1, . . . , n − 1, we get p · p<sup>i</sup> R(A) − q · q<sup>i</sup> R(B) ≤ 6ε for every point p ∈ A \ p{n − 1} and its 2ε-perturbation q ∈ B − q{n − 1}.

Case i = n is for the n-th row of the matrices M/R in [\(D.2\)](#page-29-1), where the scalar products with the orthogonal vectors p ⊥ n , q<sup>⊥</sup> n from Lemma [B.1](#page-14-4) are divided by R<sup>n</sup>−<sup>1</sup> instead of R.

Subcase i = n = 2 coincides with the case i < n above because R<sup>n</sup>−<sup>1</sup> = R. Combining the upper bounds above, we get BD M(A; p{n − 1}) R(A) , M(B; q{n − 1}) R(B) ≤ 6ε By Definition [4.2,](#page-4-1) the Point-based Representation Metric PRM equals the maximum of the bounds d<sup>∞</sup> = |R(p1) − R(q1)| = | |p1| − |q1| | ≤ 2ε, |R(A) − R(B)| ≤ 2ε, and BD above, so PRM PR(A; p1),PR(B; q1) ≤ 6ε, which finishes the proof of part (b) for n = 2.

Subcase i = n = 3. Without loss of generality, we can assume that R(A) ≥ R(B). The upper bounds of Lemmas [E.1](#page-31-2) and [E.2](#page-31-3) imply that

$$|p_3^\perp| \leq R^2(A), \quad |q_3^\perp| \leq R^2(B), \quad |p_3^\perp - q_3^\perp| \leq 2\varepsilon \cdot 2\sqrt{6}R(A).$$

We start estimating similarly to the case i < n above:

$$|p \cdot p_3^+ - q \cdot q_3^+| \leq |(p - q) \cdot p_3^+ + q \cdot (p_3^+ - q_3^+)| \leq |p - q| \cdot |p_3^+| + |q| \cdot |p_3^+ - q_3^+| \leq 2\varepsilon R^2(A) + R(B) \cdot 2\varepsilon \cdot 2\sqrt{6}R(A) = 2\varepsilon R(A)(R(A) + 4\sqrt{6}R(B)).$$

$$\begin{aligned} \text{Then} \quad \left| \frac{p \cdot p_3^\perp}{R^2(A)} - \frac{q \cdot q_3^\perp}{R^2(B)} \right| &\leq \frac{|p \cdot p_3^\perp - q \cdot q_3^\perp|}{R^2(A)} + |q \cdot q_3^\perp| \frac{|R^2(B) - R^2(A)|}{R^2(A)R^2(B)} \leq \\ &\leq 2\varepsilon \frac{R(A) + 2\sqrt{6}R(B)}{R(A)} + |q| \cdot |q_3^\perp| \frac{R^2(A) - R^2(B)}{R^2(A)R^2(B)} \leq 2\varepsilon(1 + 2\sqrt{6}) + R^3(B) \left( \frac{1}{R^2(B)} - \frac{1}{R^2(A)} \right). \end{aligned}$$

We use R(A) ≤ R(B) + 2ε to bound last term:

$$R(B) \left( 1 - \frac{R^2(B)}{R^2(A)} \right) \leq R(B) \left( 1 - \frac{R^2(B)}{(R(B) + 2\varepsilon)^2} \right) \leq \frac{R(B)}{(R(B) + 2\varepsilon)^2} 4\varepsilon (R(B) + \varepsilon) \leq 4\varepsilon.$$

Then p · p ⊥ 3 R<sup>2</sup>(A) − q · q ⊥ 3 R<sup>2</sup>(B) <sup>≤</sup> <sup>2</sup>ε(1 + 2√ 6) + 4ε < 16ε. By Definition [D.2,](#page-29-1) the Point-based Representation Metric PRM equals the maximum of

$$d_\infty = |R(p\{2\}) - R(q\{2\})| \leq 2\varepsilon, \quad |R(A) - R(B)| \leq 2\varepsilon, \quad d_\infty \leq 16\varepsilon, \quad \text{BD} < 16\varepsilon,$$

Final subcase i = n > 3. Assuming again that R(A) ≥ R(B), Lemmas [E.1](#page-31-2) and [E.2](#page-31-3) give

$$|p_n^\perp| \leq \sqrt{n}R^{n-1}(A), \quad |q_n^\perp| \leq \sqrt{n}R^{n-1}(B), \quad |p_n^\perp - q_n^\perp| \leq 2\varepsilon n(n-1)R^{n-2}(A) \text{ for any } n > 3.$$

We start estimating similarly to the case i < n.

$$\begin{aligned} & |p \cdot p_n^\perp - q \cdot q_n^\perp| \leq |(p - q) \cdot p_n^\perp + q \cdot (p_n^\perp - q_n^\perp)| \leq |p - q| \cdot |p_n^\perp| + |q| \cdot |p_n^\perp - q_n^\perp| \leq \\ & 2\varepsilon \cdot \sqrt{n} R^{n-1}(A) + R(B) \cdot 2\varepsilon n(n-1) R^{n-2}(A). \\ \text{Then} & \left| \frac{p \cdot p_n^\perp}{R^{n-1}(A)} - \frac{q \cdot q_n^\perp}{R^{n-1}(B)} \right| \leq \frac{|p \cdot p_n^\perp - q \cdot q_n^\perp|}{R^{n-1}(A)} + |q \cdot q_n^\perp| \cdot \left| \frac{R^{n-1}(B) - R^{n-1}(A)}{R^{n-1}(A) R^{n-1}(B)} \right| \leq \\ & \leq \frac{2\varepsilon \sqrt{n} R^{n-1}(A) + 2\varepsilon n(n-1) R^{n-2}(A) R(B)}{R^{n-1}(A)} + |q| \cdot |q_n^\perp| \cdot \left| \frac{1}{R^{n-1}(A)} - \frac{1}{R^{n-1}(B)} \right| \leq \\ & \leq 2\sqrt{n}\varepsilon + 2\varepsilon n(n-1) + \sqrt{n} R^n(B) \left( \frac{1}{R^{n-1}(B)} - \frac{1}{R^{n-1}(A)} \right). \end{aligned}$$

1926 We use R(A) ≤ R(B) + 2ε and the simpler notation R = R(B) to bound last term after factorizing the difference of the (n − 1)-st powers as follows:

1929

1934

1936

1939 2εn(n + √ <sup>n</sup> <sup>−</sup> 1) <sup>≤</sup> <sup>3</sup>εn<sup>2</sup> because √ n − 1 ≤ n 2 . For n = 4, the upper bound above is 3ε(4)<sup>2</sup> > 6ε ≥ d∞. Hence the final upper bound is PRM PR(A; p{n − 1}),PR(B; q{n − 1}) ≤ 3εn<sup>2</sup> .

1940 1941 1942 Lemma E.4 (Lipschitz continuity of BMD). *Let* Γ *be a complete bipartite graph with a vertex matching* E *such that any* e ∈ E *has a weight* w(e) ≤ ε*. Then* BMD(Γ) ≤ ε*.*

1943 1944 1945 1946 *Proof of Lemma [E.4.](#page-35-1)* By Definition [4.3,](#page-4-2) the vertex matching E has the weight W(E) = max e∈E w(e) ≤ ε. Since BMD(Γ) = min E W(E) is minimized for all matchings, BMD(Γ) ≤ ε.

1947 1948 1949 The Lipschitz continuity of NDP in Theorem [E.5](#page-35-0) extends Theorem [5.1](#page-5-2) to any n ≥ 2 by using Proposition [E.3](#page-32-0) and Lemma [E.4.](#page-35-1)

1954 1956 *Proof of Theorem [E.5.](#page-35-0)* Order all vertices of the given clouds A, B so that every point p<sup>i</sup> ∈ A has the same index as its ε-perturbation q<sup>i</sup> ∈ B.

1959 1960 1961 1962 1963 In Definition [D.4,](#page-30-0) for any ordered points p1, . . . , pn−<sup>1</sup> ∈ A, there are points q1, . . . , qn−<sup>1</sup> ∈ B, which are εperturbations of p1, . . . , pn−1, respectively, such that PRM(PR(A; p1, . . . , pn−1),PR(B; q1, . . . , qn−1)) ≤ λnε by Proposition [E.3.](#page-32-0) These PRMs are weights of edges in the index-preserving vertex matching E of the complete bipartite graph Γ(A; p1, . . . , pn−1; B; q1, . . . , qn−1) for any p1, . . . , pn−<sup>1</sup> and their ε-perturbations q1, . . . , qn−1. Then BMD(Γ(A; p1, . . . , pn−1; B; q1, . . . , qn−1)) ≤ λnε by Lemma [E.4.](#page-35-1) Since this conclusion holds for all (choices of) p1, . . . , pn−<sup>1</sup> ∈ C, we iteratively apply this argument for the bipartite graphs Γ(A; p1, . . . , pk; B; q1, . . . , qk) for 1 ≤ k ≤ n − 2 and finally conclude that NBM(A, B) ≤ λnε.

1964

1965 The upper bounds are higher than the real ratios NBM/BD in practical examples, see Fig. ??.

1966 1967 1968 Lemma E.6 (time of PR). *For any cloud* A ⊂ R <sup>n</sup> *of* m *unordered points, any point-based representation* PR(A; p{n − 1}) *in Definition [B.2](#page-15-0) needs* O(n <sup>3</sup> + mn) *time.*

1969 1974 *Proof of Lemma [E.6.](#page-35-2)* We find the center O(A) and translate the cloud A of m points so that O(A) becomes the origin 0 ∈ R <sup>n</sup> in time O(m). We compute the n × n matrix SD(p1, . . . , pn−1) of squared distances between p<sup>0</sup> = 0, p1, . . . , pn−<sup>1</sup> in time O(n 2 ). The vector p ⊥ n from Lemma [B.1](#page-14-4) needs the n × n determinant computable in time O(n 3 ). For any point q ∈ A \ {p1, . . . , pn−1}, the column of scalar products q · p1, . . . , q · p<sup>n</sup> needs O(n) time. The n × (m − n + 1) matrix M(A; p{n − 1}) needs O(mn) time. The point-based representation PR(A; p1, . . . , pn−1) in Definition [B.2](#page-15-0) needs O(n <sup>3</sup> + mn) time.

1976 1979 Lemma E.7 (time of PRM). *For any clouds* A, B ⊂ R <sup>n</sup> *of* m *unordered points with base sequences* p{n−1} *and* q{n−1}*, respectively, the point-based representation Metric on the equivalences classes of* PR(A; p{n − 1}) *and* PR(B; q{n − 1}) *is found in time* O(n <sup>2</sup> + m<sup>1</sup>.<sup>5</sup> log<sup>n</sup> m) *with space* O(n <sup>2</sup> + m log<sup>n</sup>−<sup>2</sup> m)*.*

$$\begin{aligned} R(B) \left( 1 - \frac{R^{n-1}(B)}{R^{n-1}(A)} \right) &\leq R \left( 1 - \frac{R^{n-1}}{(R+2\varepsilon)^{n-1}} \right) = R \frac{(R+2\varepsilon)^{n-1} - R^{n-1}}{(R+2\varepsilon)^{n-1}} = \\ &= \frac{R(R+2\varepsilon-R)}{(R+2\varepsilon)^{n-1}} \sum_{j=0}^{n-2} (R+2\varepsilon)^j R^{n-2-j} \leq \frac{2\varepsilon R}{(R+2\varepsilon)^{n-1}} \sum_{j=0}^{n-2} (R+2\varepsilon)^{n-2} \leq 2\varepsilon(n-1). \\ \text{Then BD} \left( \frac{M(A; p\{n-1\})}{R(A)}, \frac{M(B; q\{n-1\})}{R(B)} \right) &\leq \left| \frac{p \cdot p_n}{R^{n-1}(A)} - \frac{q \cdot q_n}{R^{n-1}(B)} \right| \leq \\ 2\varepsilon(\sqrt{n} + n(n-1) + \sqrt{n}(n-1)) &= 2\varepsilon\sqrt{n}(1 + \sqrt{n}(n-1) + n-1) \leq 2\varepsilon\sqrt{n}(\sqrt{n}(n-1) + n) = \end{aligned}$$

Theorem E.5 (Lipschitz continuity of NBM). *Let a cloud* B ⊂ R <sup>n</sup> *be obtained from a cloud* A ⊂ <sup>R</sup> <sup>n</sup> *by perturbing every point of* A *within its Euclidean* ε*-neighborhood. Then* NBM(A, B) ≤ λnε*, where the Lipschitz constants are* λ<sup>2</sup> = 6*,* λ<sup>3</sup> = 16*,* λ<sup>n</sup> = 3n 2 *for* n > 3 *as in Proposition [E.3.](#page-32-0)*

1981 1983 1984 The max metric w<sup>D</sup> between the n × n matrices in [\(D.2\)](#page-29-1) needs time O(n 2 ) and space O(n 2 ). For the bottleneck distance wM(σ), the n × (m − n + 1) matrices of unordered columns are interpreted as fixed (not under isometry) clouds of (m − n + 1) points in <sup>R</sup> <sup>n</sup>. Then w<sup>M</sup> can be computed in time O(m<sup>1</sup>.<sup>5</sup> log<sup>n</sup> m) with space O(m log<sup>n</sup>−<sup>2</sup> m) by Theorem 6.5 in [\(Efrat et al.,](#page-9-23) [2001\)](#page-9-23).

1986 1987 Theorems [E.8,](#page-36-0) [E.9](#page-36-1) extend Theorems [5.2,](#page-5-5) [5.3](#page-5-0) for n ≥ 2.

1989 1990 Theorem E.8 (time of NDP). *For any cloud* A ⊂ R <sup>n</sup> *of* m *unordered points, the Nested Distributed Projection* NDP(A) *in Definition [B.8](#page-19-1) is computable in time* O(n <sup>2</sup>m<sup>n</sup>) *with space* O(nm<sup>n</sup>)*.*

1991 1994 *Proof of Theorem [E.8.](#page-36-0)* The given cloud A has Ø(m<sup>n</sup>−<sup>1</sup> ) base sequences of n − 1 ordered points p1, . . . , pn−<sup>1</sup> ∈ A. Lemma [E.6](#page-35-2) computes each PR(A; p1, . . . , pn−1) in time O(n <sup>3</sup> + mn) with space O(n <sup>2</sup> + mn) needed to store O(n 2 ) pairwise distances between the points p1, . . . , pn−<sup>1</sup> and O(mn) distances from p1, . . . , pn−<sup>1</sup> to other points of A. By Definition [B.8,](#page-19-1) the invariant NDP(A) consisting of O(m<sup>n</sup>−<sup>1</sup> ) point-based representations can be computed in time O(n <sup>2</sup>m<sup>n</sup>) with space O(nm<sup>n</sup>) because n ≤ m.

1996 1997 1999 Theorem E.9 (time of NBM). *For any clouds* A, B ⊂ R <sup>n</sup> *of* m *unordered points, the Nested Bottleneck Metric* NBM(A, B) *in Definition [D.4](#page-30-0) can be computed in time* O(m<sup>2</sup>n−<sup>2</sup> (n <sup>2</sup> + m<sup>1</sup>.<sup>5</sup> log<sup>n</sup> m)) *with space* O(m<sup>2</sup> (n <sup>2</sup> + m log<sup>n</sup>−<sup>2</sup> m))*. If* n = 2*, the time is* O(m<sup>2</sup> (n <sup>2</sup> + m<sup>1</sup>.<sup>5</sup> log m))*.*

2000 2001 2002 2003 *Proof of Theorem [E.9.](#page-36-1)* In Definition [D.4,](#page-30-0) for any fixed 1 ≤ k ≤ n − 1 and ordered points p<sup>1</sup> . . . , pk−<sup>1</sup> ∈ A and q<sup>1</sup> . . . , qk−<sup>1</sup> ∈ B, the bipartite graph Γ(A; p1, . . . , pk−1; B; q1, . . . , qk−1) has V = 2(m − k + 1) = O(m) vertices and E = (m − k + 1)<sup>2</sup> = O(m<sup>2</sup> ) edges, hence O(m<sup>2</sup> ) space.

2004 2005 2006 2007 2008 2009 For k = n − 1, the weight w(e) of each edge e equals PRM, which needs time O(n <sup>2</sup> + m<sup>1</sup>.<sup>5</sup> log<sup>n</sup> m) and space O(n <sup>2</sup> + m log<sup>n</sup>−<sup>2</sup> m) by Lemma [E.7.](#page-35-3) For all O(m<sup>2</sup> ) edges of Γ(A; p1, . . . , pn−2; B; q1, . . . , qn−2), the time is O(m<sup>2</sup> (n <sup>2</sup> + m<sup>1</sup>.<sup>5</sup> log<sup>n</sup> m)), the space is O(m<sup>2</sup> (n <sup>2</sup> + m log<sup>n</sup>−<sup>2</sup> m)). The bottleneck matching distance BMD for such a graph is computed by [\(Hopcroft & Karp,](#page-9-27) [1973\)](#page-9-27) in time O(E √ V ) = O(m<sup>2</sup>.<sup>5</sup> ), which is dominated by the above time preparing the weights.

2014 For any next iteration k = n − 2, . . . , 1 in Definition [D.4,](#page-30-0) the parameter k goes down by 1 and the exponent of m drops by 2 each time. The sum over k = n − 1, . . . , 1 is dominated by the time and space of the first iteration.

2016 2018 For n = 2, the bottleneck distance between fixed m-point clouds in R 2 can be computed in time O(m<sup>1</sup>.<sup>5</sup> log m) without an extra logarithm by Theorem 6.5 from [\(Efrat et al.,](#page-9-23) [2001\)](#page-9-23), which simplifies the time to O(m<sup>2</sup> (n <sup>2</sup> + m<sup>1</sup>.<sup>5</sup> log m)).

2019 Theorem [E.9](#page-36-1) improves the time O(m3(n−1) log m) of another metric on rigid classes of unordered point clouds from Theorem 4.7(b) in [\(Widdowson & Kurlin,](#page-11-15) [2023\)](#page-11-15).

2024 2026 *Proof of Theorem [5.4.](#page-6-0)* As usual, we shift both centers of mass O(A), O(B) to the origin 0 ∈ <sup>R</sup> 2 . By Definition [4.4,](#page-4-3) the distance d = NBM(A, B) is the Bottleneck Matching Distance BMD(Γ(A, B)) computed in time O(m<sup>3</sup>.<sup>5</sup> log m) by Theorem [5.3.](#page-5-0) Here Γ(A, B) is the complete bipartite graph on m + m vertices represented by PR(A; p) and PR(B; q) for all points p ∈ A and q ∈ B.

2029 By Definition [4.3,](#page-4-2) BMD(Γ(A, B)) equals the maximum weight w(e) of an edge e in a vertex matching E of Γ(A, B), which can be considered a bijection between the m-point clouds A → B. For any pair e = (p, p′ ) of matched points, the weight w(e) is PRM(PR(A; p),PR(B; p ′ )).

2034 The distance NBM(A, B) = δ ≥ w(e) is an upper bound for |R(A)−R(B)|, where R(A) = max p∈A |p| and R(B) = max p′∈B |p ′ |. Choose a point p ∈ A with |p| = R(A) and the positive x-axis in <sup>R</sup> 2 through p ′ ∈ B matched with p via E. Let f be the

*Proof of Lemma [E.7.](#page-35-3)* The centers of masses O(A), O(B) and radii R(A), R(B) are computed in time O(m).

For all O(m<sup>n</sup>−<sup>2</sup> ) choices of ordered points p1, . . . , pn−<sup>2</sup> ∈ A and all O(m<sup>n</sup>−<sup>2</sup> ) choices of q1, . . . , qn−<sup>2</sup> ∈ B, the Bottleneck Matching Distances for all graphs Γ(A; p1, . . . , pn−2; B; q1, . . . , qn−2) are computed in time O(m<sup>2</sup>n−<sup>2</sup> (n <sup>2</sup>+m<sup>1</sup>.<sup>5</sup> log<sup>n</sup> m)) with space O(m<sup>2</sup> (n <sup>2</sup> + m log<sup>n</sup>−<sup>2</sup> m)).

2036 rotation of R 2 around 0 such that f(p) is also in the positive x-axis. By Definition [4.2,](#page-4-1) f(p), p′ in the x-axis have lengths satisfying |p| = |f(p)|, | |p| − |p ′ | | ≤ d and hence are d-close: |f(p) − p ′ | ≤ d.

2039 2040 2041 2042 It suffices to show that the image f(q) of any other point q ∈ A \ {p} is 3 √ 2d-close to a unique point q ′ ∈ B that we will find below. Since all distances and scalar products are preserved under f, we use the matrix M(f(A); f(p)) instead of <sup>M</sup>(A; <sup>p</sup>) in computing PRM. Each column of <sup>M</sup>(f(A); <sup>f</sup>(p)) R(A) consists of <sup>f</sup>(q) · <sup>f</sup>(p) |R(A)| , f(q) · f(p ⊥) |R(A)| , where f(p) = (|p|, 0), f(p <sup>⊥</sup>) = (0, |p|), R(A) = |p|.

2043 2044 2045 2046 2047 2048 2049 The distance BD M(f(A); f(p)) R(A) , M(B; q) R(B) <sup>≤</sup> <sup>d</sup> guarantees that the above column is <sup>d</sup>-close to the column of <sup>q</sup> ′ · p ′ |R(B)| , q ′ · p ′⊥ |R(B)| for a point q ′ ∈ B determined by computing the bottleneck distance BD above. For the first scalar products involving p, p′ , we have f(q) · f(p) R(A) − q ′ · p ′ R(B) ≤ δ, where the first fraction is the x-coordinate of f(q).

2054

2056

2064 2065 2066 2067 Then the x-coordinates of f(q) ∈ f(A) and q ′ ∈ B differ by at most 3d. Applying the same arguments to the scalar products involving the orthogonal vectors p <sup>⊥</sup>, p′⊥, which have the same lengths as p, p′ , respectively, conclude that the y-coordinates of f(q), q′ also differ by at most 3d. So |f(q) − q ′ | ≤ p (3d) <sup>2</sup> + (3d) <sup>2</sup> = 3√ 2d, set β(q) = q ′ .

2068 2069

2074

2076

2079

2086 2087

To get the <sup>x</sup>-coordinate <sup>q</sup> ′ · p ′ |p ′ of the point q ′ ∈ B, where |p ′ | is δ-close to R(A) = |p|, use the triangle inequality:

$$\begin{aligned} \left| \frac{f(q) \cdot f(p)}{R(A)} - \frac{q' \cdot p'}{|p'|} \right| &\leq \left| \frac{f(q) \cdot f(p)}{R(A)} - \frac{q' \cdot p'}{R(B)} \right| + \\ &+ \left| \frac{q' \cdot p'}{R(B)} - \frac{q' \cdot p'}{|p'|} \right| \leq d + \frac{|q' \cdot p'|}{R(B)|p'|} |R(B) - |p'|| \leq \\ &d + \frac{|q'| \cdot |p'|}{R(B)|p'|} |R(B) - |p'|| = d + \frac{|q'|}{R(B)} |R(B) - |p'|| \leq \\ &d + |R(B) - |p'|| \leq d + |R(B) - |p|| + ||p| - |p'|| \leq \\ &2d + |R(B) - |p|| = 2d + |R(B) - R(A)| \leq 3d. \end{aligned}$$