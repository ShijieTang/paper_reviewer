# Shepherd: Diffusing Shape, Electrostatics, And Pharmacophores For Bioisosteric Drug Design

Keir Adams∗, Kento Abeywardane∗**, Jenna Fromer, & Connor W. Coley**
Massachusetts Institute of Technology, Cambridge, MA 02139, USA {keir,kento,jfromer,ccoley}@mit.edu

## Abstract

Engineering molecules to exhibit precise 3D intermolecular interactions with their environment forms the basis of chemical design. In ligand-based drug design, bioisosteric analogues of known bioactive hits are often identified by virtually screening chemical libraries with shape, electrostatic, and pharmacophore similarity scoring functions. We instead hypothesize that a generative model which learns the joint distribution over 3D molecular structures and their interaction profiles may facilitate 3D interaction-aware chemical design. We specifically design *ShEPhERD*1, an SE(3)-equivariant diffusion model which jointly diffuses/denoises 3D molecular graphs and representations of their shapes, electrostatic potential surfaces, and (directional) pharmacophores to/from Gaussian noise. Inspired by traditional ligand discovery, we compose 3D similarity scoring functions to assess *ShEPhERD*'s ability to conditionally generate novel molecules with desired interaction profiles. We demonstrate *ShEPhERD*'s potential for impact via exemplary drug design tasks including natural product ligand hopping, protein-blind bioactive hit diversification, and bioisosteric fragment merging.

## 1 Introduction

Designing new molecules to attain specific functions via physicochemical interactions with their environment is a foundational task across the chemical sciences. For instance, early-stage drug discovery often involves tuning the 3D shape, electrostatic potential (ESP) surface, and non-covalent interactions of small-molecule ligands to promote selective binding to a protein target (Bissantz et al., 2010; Huggins et al., 2012). Homogeneous catalyst design requires developing organometallic, organic, or even peptidic catalysts that stabilize reactive transition states via specific noncovalent interactions (Raynal et al., 2014; Fanourakis et al., 2020; Knowles & Jacobsen, 2010; Wheeler et al., 2016; Toste et al., 2017; Metrano & Miller, 2018). Supramolecular chemistry similarly optimizes host-guest interactions for applications across photoresponsive materials design, biomedicine, and structure-directed zeolite synthesis (Qu et al., 2015; Stoffelen & Huskens, 2016; Corma et al., 2004). This essential challenge of designing molecules with targeted 3D interactions manifests across myriad tasks in ligand-based drug design. In medicinal chemistry, bioisosteric scaffold hopping aims to swap-out substructures within a larger molecule while preserving bioactivity (Langdon et al., 2010). Often, the swapped scaffolds share biochemically-relevant features such as electrostatics or pharmacophores, which describe both non-directional (hydrophobic, ionic) and directional (hydrogen bond acceptor/donor, aromatic π-π, halogen bonding) non-covalent interactions. When scaffold hopping extends to entire ligands, "ligand hopping" can help identify synthesizable analogues of complex natural products that mimic their 3D biochemical interactions (Grisoni et al., 2018). In hit expansion, ligand hopping is also used to diversify known bioactive hits by proposing alternative actives, ranging from topologically-similar "me-too" compounds to distinctly new chemotypes (Schneider et al., 2006; Wermuth, 2006). Notably, ligand hopping *does not require* knowledge of the protein target. Lastly, bioisosteric scaffold hopping extends beyond altering individual molecules; Wills et al. (2024) used "bioisosteric fragment merging" to replace a set of fragments that independently bind a protein with one ligand that captures the fragments' aggregate 3D binding interactions.

∗These authors contributed equally 1*ShEPhERD*: Shape, Electrostatics, and Pharmacophore Explicit Representation Diffusion 1 Here, we consider three archetypal tasks in bioisosteric drug design: 3D similarity-constrained ligand hopping, protein-blind bioactive hit diversification, and bioisosteric fragment merging (Fig. 1). The unifying theme across these tasks is identifying new molecular structures which are chemically dissimilar from known matter in terms of their molecular graphs, but which are highly similar with respect to their 3D intermolecular interaction profiles. To find such bioisosteric analogues, traditional design campaigns will virtually screen chemical libraries with 3D similarity scoring functions to query molecules' 3D shape, electrostatic, and/or pharmacophore similarity with respect to a reference molecule (Oprea & Matter, 2004; Rush et al., 2005; Zavodszky et al., 2009; Sanders et al.,
2012). However, similarity-based virtual screening has acute drawbacks: it cannot explore beyond known chemical libraries, it is inefficient by virtue of being undirected, and it can be quite slow when multiple geometries of conformationally-flexible molecules must be scored. We instead develop a broadly applicable generative modeling framework that enables efficient 3D bioisosteric molecular design. We are motivated by multiple observations: (1) As elaborated above, ligand-based drug discovery requires designing novel molecular structures that form specific 3D interactions. (2) Harris et al. (2023) have found that 3D generative models for *structure-based* drug design struggle to generate ligands that form biochemical interactions (e.g., hydrogen bonds) with the protein, despite training on protein-ligand complexes. This suggests the need for new strategies which explicitly model molecular interactions. (3) Numerous chemical design scenarios beyond drug design require engineering the physicochemical interactions of small molecules. But, such settings are often data-restricted, necessitating zero/few-shot generative approaches. To address these challenges, we introduce *ShEPhERD*, a 3D generative model which learns the relationship between the 3D chemical structures of small molecules and their shapes, electrostatics, and pharmacophores (henceforth collectively called "interaction profiles") in *context-free* environments. Specifically: - We define explicit point cloud-based representations of molecular shapes, ESP surfaces, and pharmacophores that are amenable to symmetry-preserving SE(3)-equivariant diffusion modeling.

- We formulate a joint denoising diffusion probabilistic model (DDPM) that learns the joint distribution over 3D molecules (atom types, bond types, coordinates) and their 3D shapes, ESP surfaces, and pharmacophores. In addition to diffusing/denoising attributed point clouds for shape and electrostatics, we natively model the directionality of pharmacophores by diffusing/denoising vectors on the unit sphere. We sample from specific interaction-conditioned distributions via inpainting.

- Inspired by virtual screening, we craft shape, electrostatic, and pharmacophore similarity functions to score (1) the self-consistency of jointly generated molecules and interaction profiles, and (2) the similarity between conditionally generated molecules and target interaction profiles. We show that *ShEPhERD* can generate diverse 3D molecular structures with substantially enriched interaction-similarity to target profiles, even upon geometric relaxation with semi-empirical DFT.

- After training on drug-like datasets, we demonstrate *in silico* that *ShEPhERD* can facilely design small-molecule mimics of natural products via ligand hopping, diversify bioactive hits while preserving protein-binding modes, and merge fragments into bioisosteric ligands, all out-of-the-box.

We anticipate that *ShEPhERD* will prove immediately useful for ligand-based drug design campaigns that require the *de novo* design of new molecules with precise 3D interactions. However, we stress *ShEPhERD*'s general applicability to other areas of interaction-aware molecular design, such as organocatalyst design. We especially envision that *ShEPhERD* will be extended to model other structural characteristics beyond the shape, electrostatic, and pharmacophore profiles treated here.

## 2 Related Work

3D similarity scoring for ligand-based drug design. Ligand-based drug design commonly applies shape, electrostatic, and/or pharmacophore similarity scoring functions to screen for molecules with similar 3D interactions as a known bioactive molecule (Fiedler et al., 2019; Rush et al., 2005; Jackson et al., 2022). Shape similarity functions typically use atom-centered Gaussians to compute the volumetric overlap between active and query molecules (Grant & Pickup, 1995; Grant et al., 1996). Many methods additionally attribute scalar (e.g., charge) or categorical (e.g., pharmacophore type) features to these Gaussians in order to score electrostatic (Good et al., 1992; Bolcato et al., 2022; OpenEye, Cadence Molecular Sciences, 2024) or pharmacophore similarity (Sprague, 1995; Dixon et al., 2006; Taminau et al., 2008; Wahl, 2024). To better capture intermolecular interactions, other methods quantify the Coulombic or pharmacophoric "potential" felt by a chemical probe at points near the molecule's surface (Cheeseright et al., 2006; Vainio et al., 2009; Cleves et al., 2019). We similarly use surface representations of molecular shape and electrostatic interactions, but combine this with volumetric point-cloud and vector representations of pharmacophores to model directional interactions such as hydrogen bonding and non-covalent π-effects (Wahl, 2024). Notably, we develop our own 3D similarity scoring functions which natively operate on our chosen representations. Symmetry-preserving generation of molecules in 3D. Many generative models for small molecules have been developed to sample chemical space (Gomez-Bombarelli et al., 2018; Segler ´ et al., 2018; Jin et al., 2018; Vignac et al., 2022; Bilodeau et al., 2022; Anstine & Isayev, 2023). Our work is most related to 3D approaches which directly generate molecules in specific conformations. These works usually preserve translational and rotational symmetries of atomistic systems by generating structures with E(3)- or SE(3)-invariant internal coordinates (Gebauer et al., 2022; Luo & Ji, 2022; Roney et al., 2022), or more recently, by generating Euclidean coordinates with equivariant networks (Vignac et al., 2023; Peng et al., 2023; Irwin et al., 2024). We generate molecular structures with an equivariant DDPM (Ho et al., 2020; Hoogeboom et al., 2022), a strategy that has seen use in protein-conditioned ligand and linker design (Igashov et al., 2024; Schneuing et al., 2022). Besides molecular structures, we jointly diffuse/denoise explicit representations of the molecule's shape, ESP surface, and pharmacophores with their vector directions. To do so, we employ spherical harmonics-based SE(3)-equivariant Euclidean Neural Networks (E3NNs) (Geiger & Smidt, 2022) to encode/decode coordinates, vectors, and scalar features across our heterogeneous representations. 3D interaction-aware molecular generation. Prior works have partially explored shape- and pharmacophore-aware molecular generative design. Multiple methods generate either (1D) SMILES or (2D) molecular graphs conditioned on 3D representations of target shapes and/or pharmacophores (Skalic et al., 2019; Imrie et al., 2021; Zhu et al., 2023; Xie et al., 2024), or use 3D similarity scores to fine-tune unconditional generative models (Papadopoulos et al., 2021; Neeser et al., 2023), which we include as a baseline (App. A.1). Since these generative models do not directly predict 3D structures, they require conformer generation as a post-processing step. On the other hand, Adams & Coley (2022) found that generating structures natively in 3D yields more chemically diverse molecules with higher 3D shape-similarity compared to a competing shape-conditioned 1D approach. Multiple other methods have since been developed for shape-conditioned 3D generation (Chen et al., 2023; Lin et al., 2024; Le et al., 2024). Regarding pharmacophores, Ziv et al. (2024) applied a pretrained DDPM to inpaint 3D molecules given fixed N and O atoms, constrained to be hydrogen bond donors and acceptors, respectively. Yet, they neglect other HBD/HBA definitions and ignore important non-covalent interactions like aromatic π-π interactions, hydrophobic effects, and halogen bonding. Electrostatics-aware molecular generation has been considered only by Bolcato et al. (2022), via exchanging pre-enumerated chemical fragments with similar electrostatics. But, they do not conditionally generate 3D molecules given a global ESP surface. Our work unifies and *extends* prior work on interaction-aware 3D generative design by comprehensively modeling shape, electrostatics, and arbitrary pharmacophores (including their directionality) in a single general framework.

3D generative structure-based drug design (SBDD). Tangential to our work are models that generate ligands inside a protein pocket by training on protein-ligand complexes with (Lee et al., 2024; Sako et al., 2024; Zhung et al., 2024; Wang et al., 2022; Huang et al., 2024) or without (Peng et al., 2022; Schneuing et al., 2022; Guan et al., 2023) explicit encodings of their interactions. In contrast, we consider context-free molecular design. By modeling the interaction profiles of ligands only, ShEPhERD is more general than (yet still applicable to) SBDD. We also retain the freedom to train on arbitrary chemical spaces that may be larger, denser, or more diverse than the ligands in the PDB.

## 3 Methodology

We seek to sample chemically diverse molecular structures from 3D similarity-constrained chemical space, where the constraints are defined by the 3D shape, electrostatics, and/or pharmacophores ("interaction profiles") of a reference 3D molecular system. To do so, we develop *ShEPhERD*, a DDPM that learns the *joint* distribution over 3D molecular graphs and their interaction profiles. At inference, we sample from specific interaction-conditioned distributions over 3D chemical space via *inpainting*. Below, we formally define our chosen representations for 3D molecules and their interaction profiles, and also define the 3D similarity scoring functions used in our evaluations. We then detail our joint DDPM and sampling protocols. Fig. 2 visualizes our overall methodology.

## 3.1 Defining Representations Of Molecules And Their Interaction Profiles

3D molecular graph. We define an organic molecule with n1 atoms in a specific conformation as a 3D molecular graph x1 = (a, f, C, B) where a ∈ {H, C, O*, ...*}
n1lists the atom types (Na options); f *∈ {−*2, −1, 0, 1, 2}
n1 specifies the formal charge of each atom; C ∈ R
n1×3specifies the atomic coordinates; and B ∈ {0, 1, 2, 3, 1.5}
n1×n1is the adjacency matrix specifying the covalent bond order between pairs of atoms. For modeling purposes, the categorical variables a, f, B are represented as one-hot continuous features so that a ∈ R
n1×Na , f ∈ R
n1×5, and B ∈ R
n1×n1×5.

Shape. We define the shape x2 = S2 ∈ R
n2×3as a point cloud with n2 points sampled from the solvent-accessible surface of x1. n2 is fixed regardless of n1. We use *surface* points to better decouple the shape representation from the molecule's exact coordinates or 2D graph (App. A.5).

Electrostatic potential (ESP) surface. We represent the ESP surface x3 = (S3, v) by the Coulombic potential at each point in a surface point cloud S3 ∈ R
n3×3, with v ∈ R
n3. We compute v from the per-atom partial charges of x1, which are computed via semi-empirical DFT (App. A.5). Pharmacophores. The set of n4 pharmacophores are represented as x4 = (p, P ,V ) where p ∈ R
n4×Np is a matrix of one-hot encodings of the Np pharmacophore types; P ∈ R
n4×3is the pharmacophores' coordinates; and V ∈ {S
2, 0}
n4 contains the relative unit or zero vectors specifying the directionality of each pharmacophore. The directional pharmacophores include hydrogen bond acceptors (HBA) and donors (HBD), aromatic rings, and halogen-carbon bonds. Directionless pharmacophores (V [k] = 0) include hydrophobes, anions, cations, and zinc binders. App. A.5 details how we extract pharmacophores from a molecule by adapting known SMARTS patterns.

## 3.2 Shape, Electrostatic, And Pharmacophore Similarity Scoring Functions

To formulate our scoring functions, we first define a point cloud Q ∈ R
nQ×3 where each point rk is assigned an isotropic Gaussian in R
3(Grant & Pickup, 1995; Grant et al., 1996). We measure the Tanimoto similarity sim ∈ [0, 1] between two point clouds QA and QB using first order Gaussian overlap OA,B =Pa∈QA
Pb∈QB
wa,b( π 2α
)
3 2 exp (−
α 2
∥ra − rb∥
2) and sim∗(QA, QB) =
OA,B
OA,A+OB,B−OA,B
where α is a Gaussian width and wa,b is a weighting factor. Note that 3D similarities are sensitive to SE(3) transformations of QA with respect to QB. We characterize the similarity at their optimal alignment by sim(QA, QB) = maxR,t sim∗(RQTA + t, QB) where R ∈ SO(3)
and t ∈ T(3). We align using automatic differentiation. Note that nQA need not equal nQB . Shape scoring. The *volumetric* similarity between two atomic point clouds CA and CB is defined as sim∗vol(CA, CB) with wa,b = 2.7 and α = 0.81 (Adams & Coley, 2022). We newly define the *surface* similarity between two surfaces SA and SB as sim∗
surf(SA,SB) with wa,b = 1, and α = Ψ(n2). Here, Ψ is a function fitted to sim∗vol depending on the choice of n2 (App. A.6).

Electrostatic scoring. We define the similarity between two electrostatic potential surfaces x3,A
and x3,B as sim∗ESP(x3,A, x3,B) with wa,b = exp (−
∥vA[a]−vB[b]∥
2 λ), α = Ψ(n3), and λ =0.3
(4πϵ0)
2 .

ϵ0 is the permittivity of vacuum with units e 2(eV · A˚ )
−1. Inspired by Hodgkin & Richards (1987)
and Good (1992), we use the *difference* between electrostatic potentials to increase sensitivity to their respective magnitudes (i.e., rather than simply comparing signs).

Pharmacophore scoring. We define the similarity between two sets of pharmacophores x4,A and x4,B with sim∗
pharm(x4,A, x4,B) =P
P m∈M 
OA,B;m m∈M OA,A;m+OB,B;m−OA,B;m
. Here, M is the set of all pharmacophore types (|M| = Np). wa,b = 1 if m is non-directional, or a scaling of vector cosine similarity wa,b;m =
V [a]
⊤
mV [b]m+2 3if m is directional (Wahl, 2024). αm = Ω(m) where Ω maps each pharmacophore type to a Gaussian width (App. A.6). We take the absolute value of V [a]
⊤mV [b]m for aromatic groups as we assume their π interaction effects are symmetric across their plane.

## 3.3 Joint Diffusion Of Molecules And Their Interaction Profiles With Shepherd

ShEPhERD follows the DDPM paradigm (Ho et al., 2020; Sohl-Dickstein et al., 2015) to decompose the joint distribution over the tuple X = (x1, x2, x3, x4) as Pdata(X) := P(X(0)) =
P(X(T))QT
t=1 P(X(t−1)|X(t)), where Pdata is the data distribution, P(X(T)) is (roughly) a Gaussian prior, and P(X(t−1)|X(t)) are Markov transition distributions learnt by a neural network. This network is trained to reverse a *forward*-noising process P(X(t)|X(t−1)) = N(αtX(t−1), σ2 t I)
which gradually corrupts data X into Gaussian noise X(T)according to a variance preserving noise schedule given by σ 2 tand αt =p1 − σ 2 tfor t = 1*, ..., T*. See Ho et al. (2020) for preliminaries on DDPMs. Here, we describe the forward and reverse processes of *ShEPhERD*'s joint DDPM. Forward noising processes. We follow Hoogeboom et al. (2022) to forward-noise the 3D molecule x1 = (a, f, C, B). For a ∈ R
n1×Na , we use Gaussian noising where a
(t) = αta
(t−1) + σtϵ for ϵ ∼ N(0, I). The processes for f and B are similar, but we symmetrize the upper/lower triangles of B(t). We apply isotropic noise to C ∈ R
n1×3, but center the noise at 0 to ensure translational invariance: C(t) = αtC(t−1) + σt(ϵ −
1 n1 Pn1 k=1 ϵ[k]) for ϵ[k] ∼ N(0, I3) and ϵ ∈ R
n1×3.

For the molecular shape x2 = S2 ∈ R
n2×3, we also forward-noise with isotropic Gaussian noise:
S
(t)
2 = αtS
(t−1)
2 + σtϵ for ϵ ∼ N(0, I). We *do not* subtract the noise's center of mass (COM),
though; this ensures that the model can learn to denoise x2 such that it is centered with respect to x1.

For the ESP surface x3 = (S3, v), we forward-noise the surface S3 ∈ R
n3×3in the same manner as S2. We forward-noise v3 ∈ R
n3in the typical way: v
(t) = αtv
(t−1) + σtϵ for ϵ ∼ N(0, I).

For the pharmacophores x4 = (p, P ,V ), we forward-noise their types p just like the atom types a, and their positions P just like the shape S2. Diffusing the vector directions V ∈ {S
2, 0}
n4is complicated since some pharmacophores are directionless. To unify their treatment, we interpret the pharmacophore vectors as Euclidean points in R
3(e.g., only noiseless vectors have norm 1.0 or 0.0).

We then forward-noise the vectors like any point cloud: V
(t) = αtV
(t−1) + σtϵ for ϵ ∼ N(0, I).

Whereas the above processes describe the *single-step* forward transition distributions, note that we can efficiently sample noised structures given any time horizon. For instance, we may directly sample a
(t) = αta
(0) + σtϵ for ϵ ∼ N(0, I), where αt =Qts=1 αs and σt =p1 − α 2 t.

Reverse denoising process. Starting from any X(t)(but typically pure noise X(T) ∼ N(0, I)), the DDPM iteratively denoises X(t) by stochastically interpolating towards a *predicted* clean structure Xˆ (t) ≈ X(0) ∼ Pdata(X(0)|X(t)) resembling true samples from the data distribution. We follow the DDPM formulation where rather than predicting X(0) directly, the network predicts the true noise ϵˆ
(t) ≈ ϵ that, when applied to data X(0), yields X(t) = αtX(0) + σtϵ. In this case, the single-step denoising update can be derived as: X(t−1) =1 αt X(t) −σ 2 t αtσt ϵˆ
(t) +
σtσt−1 σtϵ
′, where the additional noise ϵ
′ ∼ N(0, I) (set to ϵ
′ = 0 for t = 1) makes each denoising step stochastic.

ShEPhERD employs a single denoising network η that is trained to jointly predict the noises ϵˆ
(t) 1
, ϵˆ
(t) 2
, ϵˆ
(t) 3
, ϵˆ
(t)
4 = η(x
(t) 1
, x
(t) 2
, x
(t) 3
, x
(t) 4
, t) , where ϵˆ
(t)
1 = (ϵˆ
(t)
a , ϵˆ
(t) f
, ϵˆ
(t) C 
, ϵˆ
(t) B 
), ϵˆ
(t)
2 = (ϵˆ
(t) S2
),
ϵˆ
(t)
3 = (ϵˆ
(t)
S3
, ϵˆ
(t)
v ), and ϵˆ
(t)
4 = (ϵˆ
(t)
p , ϵˆ
(t) P
, ϵˆ
(t) V
). At inference, we jointly apply the denoising updates to obtain X(t−1). When computing x
(t−1)
1, we remove the COM from ϵˆ
(t)
C and the extra noise ϵ
′
C.

The forward and reverse processes of our joint DDPM are designed to be straightforward to make ShEPhERD flexible: One may freely adjust the exact representations of the shape, electrostatics, or pharmacophores as long as they can be represented as a point-cloud with one-hot, scalar, and/or vector attributes. One can also directly model specific marginal distributions (e.g., P(x1, x2), P(x1, x3), or P(x1, x4)) by simply modeling a subset of the variables {x2, x3, x4}. Finally, ShEP-
hERD can be easily extended to model other structural features or interaction profiles beyond those considered here by defining their explicit structural representations and forward/reverse processes. Denoising network design. We design *ShEPhERD*'s denoising network η to satisfy three criteria: - *Symmetry-preserving*: The noise predictions ϵˆ
(t)
1, ϵˆ
(t)
2, ϵˆ
(t)
3, ϵˆ
(t)
4are T(3)-invariant and SO(3)-
equivariant with respect to global SE(3)-transformations of X(t)in order to (1) efficiently preserve molecular symmetries, and (2) ensure x
(t)
1, x
(t)
2, x
(t)
3, and x
(t)
4remain aligned during denoising.

- *Expressive*: η captures both local and global relationships between x
(t) 1
, x
(t) 2
, x
(t) 3
, and x
(t)
4.

- *General*: To promote applications across chemical design, η accommodates other definitions of shape/electrostatics/pharmacophores and permits incorporating other structural interactions, too.

To achieve these criteria, we design η to have three components: (1) a set of *embedding modules* which equivariantly encode the heterogeneous xiinto a uniform sets of latent l=0 (scalar) and l=1
(vector) node representations; (2) a *joint module* which locally and globally interacts these latent node representations; and (3) a set of *denoising modules* which predict ϵˆi for each xi.

The embedding modules use SE(3)-equivariant E3NNs (we choose to use expressive EquiformerV2 modules (Liao et al., 2023)) to individually encode the 3D structures of x
(t) 1
, x
(t) 2
, x
(t) 3
, and x
(t) 4 into latent codes (z
(t)
i, z˜
(t)
i) = ϕi(x
(t)
i, t) ∀ i ∈ [1, 2, 3, 4]. Here, zi ∈ R
ni×dare invariant scalar representations of each node (e.g., atom, point, or pharmacophore), and z˜i ∈ R
ni×3×d are equivariant vector representations of each node. To make each system xi sensitive to relative translations with respect to x
(t)
1, we also include a virtual node that is positioned at the center of mass of x
(t) 1
, and which remains unnoised. Prior to 3D message passing with the E3NNs, scalar atom/point/pharmacophore features (e.g., a
(t), f
(t), v
(t), p
(t)) are embedded into l=0 node features, and vector features (e.g., V
(t)) are directly assigned as l=1 node features. The pairwise bond representations B(t)for x
(t)
1are also embedded as l=0 edge attributes. Finally, for each ϕi, we embed sinusoidal positional encodings of the time step t and add these to all the l=0 node embeddings. The joint module consists of two steps to *locally* and *globally* interact the joint latent variables: (1) We collate the coordinates of x
(t)
1, x
(t)
2, x
(t)
3, and x
(t)
4to form a heterogeneous 3D graph where the nodes are attributed with their corresponding latent features (z
(t)
i, z˜
(t)
i) ∀ i ∈ [1, 2, 3, 4]. We then encode this heterogeneous 3D graph with another E3NN (EquiformerV2) module ϕ local joint and residually update the nodes' latent features: (z
(t)
i, z˜
(t)
i) += ϕ local joint 
(x
(t)
i, z
(t)
i, z˜
(t)
i) ∀ i ∈ [1, 2, 3, 4].

(2) We sum-pool the updated l=1 node features across each sub-graph, concatenate, and then embed with an equivariant feed-forward network ϕ global joint to obtain a global l=1 code describing the overall system: z˜
(t) joint 
= ϕ global joint Cat hPni k=1 z˜
(t)
i[k]
∀ i ∈ [1, 2, 3, 4]i. We then apply equivariant tensor products between z˜
(t)
joint 
∈ R
d×3and an l=0 embedding of t. This yields l=0 and l=1 global latent features (z
(t)
joint, z˜
(t)
joint), which are residually added to the node representations (z
(t)
i, z˜
(t)
i).

The denoising modules predict the noises ϵˆi from the node-level features (z
(t)
i, z˜
(t)
i). For x2, x3, and x4, the scalar noises (ϵˆ
(t)
v , ϵˆ
(t)
p ) are predicted from the corresponding (l=0) z
(t)
icodes using multi-layer perceptrons (MLPs), whereas the vector noises (ϵˆ
(t)
S2
, ϵˆ
(t)
S3
, ϵˆ
(t)
P, ϵˆ
(t)
V) are predicted from the (l=1) z˜
(t)
icodes using equivariant feed-forward networks ("E3NN-style" coordinate predictions). For x1, ϵˆ
(t)
a and ϵˆ
(t)
fare predicted from z
(t)
1 with simple MLPs. ϵˆ
(t) B 
is predicted from pairs
(z
(t)
1[k], z
(t)
1[j]) using a permutation-invariant MLP (App. A.7.1). ϵˆ
(t) C 
is predicted from (z
(t)
1, z˜
(t)
1)
using E3NN-style and EGNN-style (Satorras et al., 2021) coordinate predictions (App. A.7.1).

Training. We train the denoising network η with unweighted L2 regression losses li = ||ϵˆi − ϵi||2 between the predicted and true noises. Importantly, our framework permits us to train models which directly learn certain marginal distributions. In our experiments, we train models to learn P(x1), P(x1, x2), P(x1, x3), P(x1, x4), and P(x1, x3, x4). Note that since x3 defines an (attributed) surface S3, jointly modeling (x2, x3) is redundant; x3 implicitly models x2. App. A.7 provides details on training protocols, choice of noise schedules, feature scaling, and model hyperparameters.

Sampling. For *unconditional generation*, we first sample X(T)from isotropic Gaussian noise, and then denoise for T steps to sample X(0). We then argmax a
(0), f
(0), B(0), and p
(0) to obtain discrete atom/bond/pharm. types, and round each V
(0)[k] to have norm 1.0 or 0.0. To help break the spherical symmetry of X(t)at early time steps, we strategically add extra noise to the reverse process (App. A.8). For *conditional generation*, we use inpainting (Lugmayr et al., 2022; Schneuing et al., 2022) to sample x
(0)
1conditioned on target interaction profiles. Namely, we first simulate the forward-noising of the target profiles (x
∗2, x
∗3, x
∗4) → (x
∗2, x
∗3, x
∗4)
(t) ∀ t ∈ [1, T]. Then, during the reverse denoising process, we replace the *ShEPhERD*-denoised (x
(t)
2, x
(t)
3, x
(t)
4) with the noisy target (x
∗ 2
, x
∗3
, x
∗4
)
(t). Like other molecule DDPMs, we must specify n1 and n4 for each sample.

## 4 Experiments

We train and evaluate *ShEPhERD* using two new datasets. Our first dataset (*ShEPhERD***-GDB17**)
contains 2.8M molecules sampled from medicinally-relevant subsets of GDB17 (Ruddigkeit et al.,
2012; Awale et al., 2019; Buhlmann & Reymond, 2020). Each molecule contains ¨ ≤17 non-hydrogen atoms with element types in {H, C, N, O, S, F, Cl, Br, I}, and includes one conformation optimized with GFN2-xTB in the gas phase (Bannwarth et al., 2019). Our second dataset (*ShEPhERD*- MOSES-aq) contains 1.6M drug-like molecules from MOSES (Polykovskiy et al., 2020) with up to 27 non-hydrogen atoms. Each molecule contains one conformation optimized with GFN2-xTB in implicit water. In all experiments, hydrogens are treated explicitly. Whereas we use ShEP- hERD-GDB17 to evaluate *ShEPhERD* in straightforward unconditional and conditional generation settings, we use *ShEPhERD*-MOSES-aq to challenge *ShEPhERD* to design drug-like analogues of natural products, to diversify bioactive hits, and to merge fragments from a fragment screen.

Unconditional joint generation. We first evaluate *ShEPhERD*'s ability to jointly generate 3D
molecules and their interaction profiles in a self-consistent way. Namely, a well-trained model that learns the joint distribution should generate interaction profile(s) that match the true interaction profile(s) of the generated molecule. Fig. 3 reports distributions of the 3D similarity between generated and true interaction profiles across 1000 samples (with n1 ∈ [11, 60], n4 ∼ P*data*(n4|n1)) from models trained on *ShEPhERD*-GDB17 to learn P(x1, x2), P(x1, x3), or P(x1, x4).

2 When we compare against the (optimally aligned) similarities simsurf, simESP, and simpharm between the true profiles of the generated molecules and those of random molecules from the dataset, we confirm that *ShEPhERD*'s generated profiles have substantially enriched similarities to the true profiles in all cases. Interestingly, *ShEPhERD* is more self-consistent when generating shapes or ESP surfaces than when generating pharmacophores. We partially attribute this performance disparity to the discrete nature of the pharmacophore representations and the requirement of specifying n4; ShEPhERD occasionally generates molecules that have more true pharmacophores than generated pharmacophores, as demonstrated by the samples shown in Fig. 3 (i.e., some have missing HBAs). Interaction-conditioned generation. We now evaluate *ShEPhERD*'s ability to sample from interaction-conditioned chemical space. To do so, we reuse the same P(x1, x2), P(x1, x3), and P(x1, x4) models trained on *ShEPhERD*-GDB17, but use *inpainting* to sample from the interactionconditioned distributions P(x1|x2), P(x1|x3), and P(x1|x4). Specifically, we extract the true interaction profiles from 100 random target molecules (held out from training), and use *ShEPhERD* to inpaint new 3D molecular structures given these target profiles. After generating 20 structures per target profile and discarding (without replacement) any invalid structures (App. A.3) or those with 2D graph similarities >0.3 to the target molecule, we relax the generated structures with xTB and compute their optimally-realigned 3D similarities to the target. Fig. 3 plots the distributions of 3D similarity scores between all valid (sample, target) pairs as well as the top-1 scores amongst the 20 samples per target. We compare against analogous similarity distributions for randomly sampled molecules from the dataset. Overall, *ShEPhERD* generates structures with very low graph similarity (≥94% of valid samples have graph similarity <0.2) but significantly enriched 3D similarities to the target, for all versions of the model. Qualitatively, *ShEPhERD* can generate molecular structures that satisfy very detailed target interactions, including multiple directional pharmacophores (Fig. 3). Natural product ligand hopping. Numerous clinically-approved drugs have structures derived from natural products due to their rich skeletal complexity, high 3D character, and wide range of pharmacophores that impart uniquely selective biological function. But, the structural complexity of natural products limits their synthetic tractability. As such, designing synthetically-accessible smallmolecule analogues of natural products that mimic their precise 3D interactions is a preeminent task in scaffold/ligand hopping. To imitate this task, we select three complex natural products from CO-
CONUT (Sorokina et al., 2021), including two large macrocycles and a fused ring system with 9 stereocenters. We then apply *ShEPhERD* (trained to learn P(x1, x3, x4) on *ShEPhERD*-MOSES-
aq) to generate drug-like molecules conditioned *jointly* on the ESP surface and pharmacophores of the lowest-energy conformer of each natural product, again via inpainting. We emphasize that these natural products are out-of-distribution compared to the drug-like molecules contained in ShEP-
hERD-MOSES-aq. Upon generating 2500 samples from the conditional distribution P(x1|x3, x4)
for each natural product, *ShEPhERD* identifies small-molecule mimics that attain high ESP and pharmacophore similarity to the natural products (Fig. 4), despite having much simpler chemical structures as assessed via SA score (Ertl & Schuffenhauer, 2009). Crucially, *ShEPhERD* generates molecules with higher 3D similarity compared to 2500 molecules sampled from the dataset, and compared to 10K molecules optimized by REINVENT (Blaschke et al., 2020) (App. A.1). Bioactive hit diversification. Whereas stucture-based drug design aims to design high-affinity ligands given the structure of the protein target, ligand-based drug design attempts to develop and optimize bioactive hit compounds in the absence of protein information. A common task in ligandbased drug design is diversifying the chemical structures of previously identified bioactive hits (i.e., from a phenotypic experimental screen) through interaction-preserving scaffold hopping, often as a means to reduce toxicity, increase synthesizability, or evade patent restrictions. We simulate bioactive hit diversification by using *ShEPhERD* to generate analogues of 7 experimental ligands from the PDB. To evaluate their likelihood of retaining bioactivity, we use Autodock Vina (Trott & Olson, 2010; Eberhardt et al., 2021) to dock the generated ligands to their respective proteins, treating the Vina docking scores as a weak surrogate for bioactivity. To best imitate ligand-based design, we condition *ShEPhERD* on the ESP surface and pharmacophores of the *lowest-energy* conformer of each PDB ligand, rather than their bound poses (we also simulate this scenario for comparison). We then use inpainting to generate 500 samples from P(x1|x3, x4), and dock the valid samples. Fig.

4 shows the distributions of Vina scores for *ShEPhERD*-generated analogues, compared against a docking screen of 10K random compounds from *ShEPhERD*-MOSES-aq. Despite having no explicit knowledge of the protein targets, *ShEPhERD* enriches Vina scores in multiple cases. For the topscoring generated ligands for 5mo4 and 7l11, *ShEPhERD* generates substantial scaffold hops that yield diverse 2D graph structures relative to the experimental ligands. Nevertheless, upon docking, the generated molecules still explore poses that closely align with the experimental crystal poses. Bioisosteric fragment merging. Fragment screening seeks to identify protein-ligand binding modes by analyzing how small chemical fragments bind to a protein of interest. Clusters of protein-bound fragment hits can then be analyzed to design high-affinity ligands. Multiple methods have been developed to *link* fragment hits into a single ligand containing the original fragments and the new linker. Recently, Wills et al. (2024) showed that *merging* (not *linking*) the fragments to form a bioisosteric ligand (which may not contain the exact fragment hits) can diversify ligand hypotheses while preserving the fragments' important binding interactions. While *ShEPhERD could* link fragments, *ShEPhERD* is uniquely suited to bioisosteric fragment merging as it can condition on aggregate fragment interactions. We use *ShEPhERD* to merge a set of 13 fragments experimentally identified to bind to the antiviral target EV-D68 3C protease (Lithgo et al., 2024; Wills et al., 2024). We extract n4 = 27 pharmacophores by clustering common motifs and selecting interactions identified by Fragalysis (Diamond Light Source, 2024) (App. A.2). We also compute an aggregate ESP surface by sampling points from the surface of the overlaid fragments and averaging the fragments' ESP contributions at each point. We then condition *ShEPhERD* on these profiles to sample 1000 structures (n1 ∈ [50, 89]) from P(x1|x3, x4) via inpainting. Fig. 4 shows samples with SA ≤ 4.0 that score in the top-10 by combined ESP and pharmacophore similarity. Visually, *ShEPhERD*
generates structures that align well to the fragments and preserve many of their binding interactions, even though n1 and n4 are significantly out-of-distribution from *ShEPhERD*-MOSES-aq (App. A.9).

## 5 Conclusion

We introduced *ShEPhERD*, a new 3D molecular generative model that facilitates interactionaware chemical design by learning the joint distribution over 3D molecular structures and their shapes, electrostatics, and pharmacophores. Empirically, *ShEPhERD* can sample chemically diverse molecules with highly enriched interaction-similarity to target structures, as assessed via custom 3D similarity scoring functions. In bioisosteric drug design, *ShEPhERD* can design small-molecule mimics of complex natural products, diversify bioactive hits while enriching docking scores despite having no knowledge of the protein, and merge fragments from experimental fragment screens into bioisosteric ligands. We anticipate that future work will creatively extend *ShEPhERD* to other areas of interaction-aware chemical design such as structure-based drug design and organocatalyst design.

## Reproducibility Statement

Our main text and appendices provide all critical details necessary to understand and reproduce our work, including our training and sampling protocols. To ensure reproducibility, we make our datasets and all training, inference, and evaluation code available on Github at https: //github.com/coleygroup/shepherd and https://github.com/coleygroup/ shepherd-score.

## Acknowledgments

The authors would like to thank Wenhao Gao for support with SynFormer. This research was supported by the Office of Naval Research under grant number ONR N00014-21-1-2195. This material is based upon work supported by the National Science Foundation Graduate Research Fellowship under Grant No. 2141064. The authors acknowledge the MIT SuperCloud and Lincoln Laboratory Supercomputing Center for providing HPC resources that have contributed to the research results reported within this paper.

## References

Keir Adams and Connor W Coley. Equivariant shape-conditioned generation of 3d molecules for ligand-based drug design. *arXiv preprint arXiv:2210.04893*, 2022.

Dylan M Anstine and Olexandr Isayev. Generative models as an emerging paradigm in the chemical sciences. *Journal of the American Chemical Society*, 145(16):8736–8750, 2023.

Mahendra Awale, Finton Sirockin, Nikolaus Stiefl, and Jean-Louis Reymond. Medicinal chemistry aware database gdbmedchem. *Molecular informatics*, 38(8-9):1900031, 2019.

Christoph Bannwarth, Sebastian Ehlert, and Stefan Grimme. Gfn2-xtb—an accurate and broadly parametrized self-consistent tight-binding quantum chemical method with multipole electrostatics and density-dependent dispersion contributions. *Journal of chemical theory and computation*, 15 (3):1652–1671, 2019.

Francois Berenger and Koji Tsuda. 3d-sensitive encoding of pharmacophore features. Journal of Chemical Information and Modeling, 63(8):2360–2369, 2023.

Helen M Berman, John Westbrook, Zukang Feng, Gary Gilliland, Talapady N Bhat, Helge Weissig, Ilya N Shindyalov, and Philip E Bourne. The protein data bank. *Nucleic acids research*, 28(1):
235–242, 2000.

Camille Bilodeau, Wengong Jin, Tommi Jaakkola, Regina Barzilay, and Klavs F Jensen. Generative models for molecular discovery: Recent advances and challenges. Wiley Interdisciplinary Reviews: Computational Molecular Science, 12(5):e1608, 2022.

Caterina Bissantz, Bernd Kuhn, and Martin Stahl. A medicinal chemist's guide to molecular interactions. *Journal of medicinal chemistry*, 53(14):5061–5084, 2010.

Thomas Blaschke, Josep Arus-Pous, Hongming Chen, Christian Margreitter, Christian Tyrchan, Ola ´
Engkvist, Kostas Papadopoulos, and Atanas Patronov. REINVENT 2.0: An AI Tool for de Novo Drug Design. *Journal of Chemical Information and Modeling*, 60(12):5918–5922, December 2020. ISSN 1549-9596, 1549-960X.

Giovanni Bolcato, Esther Heid, and Jonas Bostrom. On the value of using 3d shape and electrostatic ¨
similarities in deep generative methods. *Journal of Chemical Information and Modeling*, 62(6): 1388–1398, 2022.

Sven Buhlmann and Jean-Louis Reymond. Chembl-likeness score and database gdbchembl. ¨ Frontiers in chemistry, 8:46, 2020.

Tim Cheeseright, Mark Mackey, Sally Rose, and Andy Vinter. Molecular field extrema as descriptors of biological activity: definition and validation. *Journal of chemical information and modeling*, 46(2):665–676, 2006.

Ziqi Chen, Bo Peng, Srinivasan Parthasarathy, and Xia Ning. Shape-conditioned 3d molecule generation via equivariant diffusion models. *arXiv preprint arXiv:2308.11890*, 2023.

Ann E Cleves, Stephen R Johnson, and Ajay N Jain. Electrostatic-field and surface-shape similarity for virtual screening and pose prediction. *Journal of computer-aided molecular design*, 33(10): 865–886, 2019.

Avelino Corma, Fernando Rey, Jordi Rius, Maria J Sabater, and Susana Valencia. Supramolecular self-assembled molecules as organic directing agent for synthesis of zeolites. *Nature*, 431(7006): 287–290, 2004.

S.W. Cowan-Jacob. Abl1 kinase (t334i d382n) in complex with asciminib and nilotinib. Protein Data Bank, 2016. PDB ID: 5mo4.

M.G. Deshmukh, J.A. Ippolito, E.A. Stone, W.L. Jorgensen, and K.S. Anderson. Crystal structure of the sars-cov-2(2019-ncov) main protease in complex with compound 5. Protein Data Bank, 2020. PDB ID: 7l11.

Diamond Light Source. Fragalysis, 2024. URL https://fragalysis.diamond.ac.uk/.

Accessed on September 15, 2024.

Steven L Dixon, Alexander M Smondyrev, Eric H Knoll, Shashidhar N Rao, David E Shaw, and Richard A Friesner. Phase: a new engine for pharmacophore perception, 3d qsar model development, and 3d database screening: 1. methodology and preliminary results. Journal of computeraided molecular design, 20:647–671, 2006.

Yu Dong, Xiaodi Qiu, Neil Shaw, Yueyang Xu, Yuna Sun, Xuemei Li, Jun Li, and Zihe Rao. Molecular basis for the inhibition of β-hydroxyacyl-acp dehydratase hadab complex from mycobacterium tuberculosis by flavonoid inhibitors. *Protein & cell*, 6(7):504–517, 2015.

Jerome Eberhardt, Diogo Santos-Martins, Andreas F Tillack, and Stefano Forli. Autodock vina 1.2. 0: New docking methods, expanded force field, and python bindings. Journal of chemical information and modeling, 61(8):3891–3898, 2021.

Peter Ertl and Ansgar Schuffenhauer. Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions. Journal of cheminformatics, 1:1–11, 2009.

Alexander Fanourakis, Philip J Docherty, Padon Chuentragool, and Robert J Phipps. Recent developments in enantioselective transition metal catalysis featuring attractive noncovalent interactions between ligand and substrate. *ACS catalysis*, 10(18):10672–10714, 2020.

Lorna R Fiedler, Kathryn Chapman, Min Xie, Evie Maifoshie, Micaela Jenkins, Pelin Arabacilar Golforoush, Mohamed Bellahcene, Michela Noseda, Dorte Faust, Ashley Jarvis, et al. Map4k4 ¨ inhibition promotes survival of human stem cell-derived cardiomyocytes and reduces infarct size in vivo. *Cell Stem Cell*, 24(4):579–591, 2019.

forlilab. Meeko, n.d. URL https://github.com/forlilab/Meeko. Available at https://github.com/forlilab/Meeko.

Wenhao Gao, Tianfan Fu, Jimeng Sun, and Connor W. Coley. Sample Efficiency Matters: A Benchmark for Practical Molecular Optimization. In *Thirty-sixth Conference on Neural Information* Processing Systems Datasets and Benchmarks Track, June 2022.

Wenhao Gao, Shitong Luo, and Connor W Coley. Generative artificial intelligence for navigating synthesizable chemical space. *arXiv preprint arXiv:2410.03494*, 2024.

Niklas WA Gebauer, Michael Gastegger, Stefaan SP Hessmann, Klaus-Robert Muller, and Kristof T ¨
Schutt. Inverse design of 3d molecular structures with conditional generative neural networks. ¨ Nature communications, 13(1):973, 2022.

Mario Geiger and Tess Smidt. e3nn: Euclidean neural networks. *arXiv preprint arXiv:2207.09453*,
2022.

Rafael Gomez-Bombarelli, Jennifer N Wei, David Duvenaud, Jos ´ e Miguel Hern ´ andez-Lobato, ´
Benjam´ın Sanchez-Lengeling, Dennis Sheberla, Jorge Aguilera-Iparraguirre, Timothy D Hirzel, ´ Ryan P Adams, and Alan Aspuru-Guzik. Automatic chemical design using a data-driven contin- ´ uous representation of molecules. *ACS central science*, 4(2):268–276, 2018.

AC Good. The calculation of molecular similarity: Alternative formulas, data manipulation and graphical display. *Journal of Molecular Graphics*, 10(3):144–151, 1992.

Andrew C Good, Edward E Hodgkin, and W Graham Richards. Utilization of gaussian functions for the rapid evaluation of molecular similarity. Journal of Chemical Information and Computer Sciences, 32(3):188–191, 1992.

J Andrew Grant and BT Pickup. A gaussian description of molecular shape. The Journal of Physical Chemistry, 99(11):3503–3510, 1995.

J Andrew Grant, Maria A Gallardo, and Barry T Pickup. A fast method of molecular shape comparison: A simple application of a gaussian description of molecular shape. Journal of computational chemistry, 17(14):1653–1666, 1996.

Francesca Grisoni, Daniel Merk, Viviana Consonni, Jan A Hiss, Sara Giani Tagliabue, Roberto Todeschini, and Gisbert Schneider. Scaffold hopping from natural products to synthetic mimetics by holistic molecular similarity. *Communications Chemistry*, 1(1):44, 2018.

Jiaqi Guan, Wesley Wei Qian, Xingang Peng, Yufeng Su, Jian Peng, and Jianzhu Ma. 3d equivariant diffusion for target-aware molecule generation and affinity prediction. arXiv preprint arXiv:2303.03543, 2023.

Charles Harris, Kieran Didi, Arian R Jamasb, Chaitanya K Joshi, Simon V Mathis, Pietro Lio, and Tom Blundell. Benchmarking generated poses: How rational is structure-based drug design with generative models? *arXiv preprint arXiv:2308.07413*, 2023.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. *Advances in* neural information processing systems, 33:6840–6851, 2020.

Edward E Hodgkin and W Graham Richards. Molecular similarity based on electrostatic potential and electric field. *International Journal of Quantum Chemistry*, 32(S14):105–110, 1987.

Emiel Hoogeboom, Vıctor Garcia Satorras, Clement Vignac, and Max Welling. Equivariant diffu- ´
sion for molecule generation in 3d. In *International conference on machine learning*, pp. 8867– 8887. PMLR, 2022.

Kexin Huang, Tianfan Fu, Wenhao Gao, Yue Zhao, Yusuf Roohani, Jure Leskovec, Connor W Coley, Cao Xiao, Jimeng Sun, and Marinka Zitnik. Therapeutics data commons: Machine learning datasets and tasks for drug discovery and development. Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021.

Zhilin Huang, Ling Yang, Xiangxin Zhou, Zhilong Zhang, Wentao Zhang, Xiawu Zheng, Jie Chen, Yu Wang, CUI Bin, and Wenming Yang. Protein-ligand interaction prior for binding-aware 3d molecule diffusion models. In The Twelfth International Conference on Learning Representations, 2024.

David J Huggins, Woody Sherman, and Bruce Tidor. Rational approaches to improving selectivity in drug design. *Journal of medicinal chemistry*, 55(4):1424–1444, 2012.

Ilia Igashov, Hannes Stark, Cl ¨ ement Vignac, Arne Schneuing, Victor Garcia Satorras, Pascal ´
Frossard, Max Welling, Michael Bronstein, and Bruno Correia. Equivariant 3d-conditional diffusion model for molecular linker design. *Nature Machine Intelligence*, pp. 1–11, 2024.

Fergus Imrie, Thomas E Hadfield, Anthony R Bradley, and Charlotte M Deane. Deep generative design with 3d pharmacophoric constraints. *Chemical science*, 12(43):14577–14589, 2021.

Ross Irwin, Alessandro Tibo, Jon-Paul Janet, and Simon Olsson. Efficient 3d molecular generation with flow matching and scale optimal transport. *arXiv preprint arXiv:2406.07266*, 2024.

V.-P. Jaakola, M.T. Griffith, M.A. Hanson, V. Cherezov, E.Y.T. Chien, J.R. Lane, A.P. Ijzerman, and R.C. Stevens. The 2.6 a crystal structure of a human a2a adenosine receptor bound to zm241385. Protein Data Bank, 2008a. PDB ID: 3eml.

Veli-Pekka Jaakola, Mark T Griffith, Michael A Hanson, Vadim Cherezov, Ellen YT Chien, J Robert Lane, Adriaan P Ijzerman, and Raymond C Stevens. The 2.6 angstrom crystal structure of a human a2a adenosine receptor bound to an antagonist. *Science*, 322(5905):1211–1217, 2008b.

Victoria Jackson, Linda Jordan, Ryan N Burgin, Oliver JS McGaw, Calum W Muir, and Victor Ceban. Application of molecular-modeling, scaffold-hopping, and bioisosteric approaches to the discovery of new heterocyclic picolinamides. *Journal of Agricultural and Food Chemistry*, 70 (36):11031–11041, 2022.

Wengong Jin, Regina Barzilay, and Tommi Jaakkola. Junction tree variational autoencoder for molecular graph generation. In *International conference on machine learning*, pp. 2323–2332. PMLR, 2018.

Robert R Knowles and Eric N Jacobsen. Attractive noncovalent interactions in asymmetric catalysis:
Links between enzymes and small molecule catalysts. Proceedings of the National Academy of Sciences, 107(48):20678–20685, 2010.

David Ryan Koes and Carlos J Camacho. Pharmer: efficient and exact pharmacophore search.

Journal of chemical information and modeling, 51(6):1307–1314, 2011.

Alina Kutlushina, Aigul Khakimova, Timur Madzhidov, and Pavel Polishchuk. Ligand-based pharmacophore modeling using novel 3d pharmacophore signatures. *Molecules*, 23(12):3094, 2018.

Greg Landrum et al. Rdkit: Open-source cheminformatics, 2006a. Gregory A Landrum, Julie E Penzotti, and Santosh Putta. Feature-map vectors: a new class of informative descriptors for computational drug discovery. Journal of computer-aided molecular design, 20:751–762, 2006b.

Sarah R Langdon, Peter Ertl, and Nathan Brown. Bioisosteric replacement and scaffold hopping in lead generation and optimization. *Molecular informatics*, 29(5):366–385, 2010.

Tuan Le, Julian Cremer, Djork-Arne Clevert, and Kristof T Sch ´ utt. Latent-guided equivariant diffu- ¨
sion for controlled structure-based de novo ligand generation. In ICML'24 Workshop ML for Life and Material Science: From Theory to Industry Applications, 2024.

Joongwon Lee, Wonho Zhung, and Woo Youn Kim. Ncidiff: Non-covalent interaction-generative diffusion model for improving reliability of 3d molecule generation inside protein pocket. *arXiv* preprint arXiv:2405.16861, 2024.

J. Li, Y. Dong, and Z.H. Rao. Crystal structure of (3r)-hydroxyacyl-acp dehydratase hadab heterodimer from mycobacterium tuberculosis complexed with 2',4,4'-trihydroxychalcone. Protein Data Bank, 2014. PDB ID: 4rlu.

Yi-Lun Liao, Brandon Wood, Abhishek Das, and Tess Smidt. Equiformerv2: Improved equivariant transformer for scaling to higher-degree representations. *arXiv preprint arXiv:2306.12059*, 2023.

Fang-Yu Lin and Alexander D MacKerell Jr. Do halogen–hydrogen bond donor interactions dominate the favorable contribution of halogens to ligand–protein binding? *The Journal of Physical* Chemistry B, 121(28):6813–6821, 2017.

Jie Lin, Mingyuan Xu, and Hongming Chen. Diff-shape: A novel constrained diffusion model for shape based de novo drug design. *ChemRxiv*, 2024. doi: 10.26434/chemrxiv-2024-km0h1.

Ryan M Lithgo, Charlie WE Tomlinson, Michael Fairhead, Max Winokan, Warren Thompson, Conor Wild, Jasmin Aschenbrenner, Blake Balcomb, Peter G Marples, Anu V Chandran, et al. Crystallographic fragment screen of enterovirus d68 3c protease and iterative design of lead-like compounds using structure-guided expansions. *bioRxiv*, pp. 2024–04, 2024.

Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool.

Repaint: Inpainting using denoising diffusion probabilistic models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 11461–11471, 2022.

Youzhi Luo and Shuiwang Ji. An autoregressive flow model for 3d molecular geometry generation from scratch. In *International conference on learning representations (ICLR)*, 2022.

Elaine C Meng, Thomas D Goddard, Eric F Pettersen, Greg S Couch, Zach J Pearson, John H
Morris, and Thomas E Ferrin. UCSF ChimeraX: Tools for structure building and analysis. Protein Science, 32(11):e4792, 2023.

Anthony J Metrano and Scott J Miller. Peptide-based catalysts reach the outer sphere through remote desymmetrization and atroposelectivity. *Accounts of Chemical Research*, 52(1):199–215, 2018.

B. Nagar, W. Bornmann, T. Schindler, B. Clarkson, and J. Kuriyan. Crystal structure of the c-abl kinase domain in complex with sti-571. Protein Data Bank, 2001. PDB ID: 1iep.

Bhushan Nagar, William G Bornmann, Patricia Pellicena, Thomas Schindler, Darren R Veach, W Todd Miller, Bayard Clarkson, and John Kuriyan. Crystal structures of the kinase domain of c-abl in complex with the small molecule inhibitors pd173955 and imatinib (sti-571). Cancer research, 62(15):4236–4243, 2002.

Maruti Naik, Anandkumar Raichurkar, Balachandra S Bandodkar, Begur V Varun, Shantika Bhat, Rajesh Kalkhambkar, Kannan Murugan, Rani Menon, Jyothi Bhat, Beena Paul, et al. Structure guided lead generation for m. tuberculosis thymidylate kinase (mtb tmk): discovery of 3cyanopyridone and 1, 6-naphthyridin-2-one as potent inhibitors. *Journal of medicinal chemistry*, 58(2):753–766, 2015.

Rebecca M Neeser, Mehmet Akdel, Daniel Kovtun, and Luca Naef. Reinforcement learning-driven linker design via fast attention-based point cloud alignment. *arXiv preprint arXiv:2306.08166*, 2023.

Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models.

In *International conference on machine learning*, pp. 8162–8171. PMLR, 2021.

OpenEye, Cadence Molecular Sciences. EON 3.0.0.0. http://www.eyesopen.com, 2024.

Santa Fe, NM.

Tudor I Oprea and Hans Matter. Integrating virtual screening in lead discovery. Current opinion in chemical biology, 8(4):349–358, 2004.

Alaa MA Osman and Alya A Arabi. Average electron density: A quantitative tool for evaluating non-classical bioisosteres of amides. *ACS omega*, 9(11):13172–13182, 2024.

Kostas Papadopoulos, Kathryn A Giblin, Jon Paul Janet, Atanas Patronov, and Ola Engkvist. De novo design with deep generative models based on 3d similarity scoring. *Bioorganic & Medicinal* Chemistry, 44:116308, 2021.

Xingang Peng, Shitong Luo, Jiaqi Guan, Qi Xie, Jian Peng, and Jianzhu Ma. Pocket2mol: Efficient molecular sampling based on 3d protein pockets. In International Conference on Machine Learning, pp. 17644–17655. PMLR, 2022.

Xingang Peng, Jiaqi Guan, Qiang Liu, and Jianzhu Ma. Moldiff: Addressing the atom-bond inconsistency problem in 3d molecule diffusion generation. *arXiv preprint arXiv:2305.07508*, 2023.

Peter Politzer, Jane S Murray, and Timothy Clark. Halogen bonding and other σ-hole interactions:
A perspective. *Physical Chemistry Chemical Physics*, 15(27):11178–11189, 2013.

Daniil Polykovskiy, Alexander Zhebrak, Benjamin Sanchez-Lengeling, Sergey Golovanov, Oktai Tatanov, Stanislav Belyaev, Rauf Kurbanov, Aleksey Artamonov, Vladimir Aladinskiy, Mark Veselov, et al. Molecular sets (moses): a benchmarking platform for molecular generation models. Frontiers in pharmacology, 11:565644, 2020.

Da-Hui Qu, Qiao-Chun Wang, Qi-Wei Zhang, Xiang Ma, and He Tian. Photoresponsive host–guest functional systems. *Chemical reviews*, 115(15):7543–7588, 2015.

Matthieu Raynal, Pablo Ballester, Anton Vidal-Ferran, and Piet WNM van Leeuwen. Supramolecular catalysis. part 1: non-covalent interactions as a tool for building and modifying homogeneous catalysts. *Chemical Society Reviews*, 43(5):1660–1733, 2014.

J.A. Read, S. Hussein, H. Gingell, and J. Tucker. Mtb tmk in complex with compound 8. Protein Data Bank, 2014. PDB ID: 4unn.

James P Roney, Paul Maragakis, Peter Skopp, and David E Shaw. Generating realistic 3d molecules with an equivariant conditional likelihood model. 2022.

Lars Ruddigkeit, Ruud Van Deursen, Lorenz C Blum, and Jean-Louis Reymond. Enumeration of 166 billion organic small molecules in the chemical universe database gdb-17. Journal of chemical information and modeling, 52(11):2864–2875, 2012.

Thomas S Rush, J Andrew Grant, Lidia Mosyak, and Anthony Nicholls. A shape-based 3-d scaffold hopping method and its application to a bacterial protein- protein interaction. *Journal of medicinal* chemistry, 48(5):1489–1495, 2005.

Masami Sako, NOBUAKI YASUO, and Masakazu Sekijima. Diffint: A pharmacophore-aware diffusion model for structure-based drug design with explicit hydrogen bond interaction guidance. ChemRxiv, 2024. doi: 10.26434/chemrxiv-2024-23fbj.

Marijn PA Sanders, Armenio JM Barbosa, Barbara Zarzycka, Gerry AF Nicolaes, Jan PG Klomp, ´
Jacob De Vlieg, and Alberto Del Rio. Comparative analysis of pharmacophore screening tools. Journal of chemical information and modeling, 52(6):1607–1620, 2012.

Vıctor Garcia Satorras, Emiel Hoogeboom, and Max Welling. E (n) equivariant graph neural networks. In *International conference on machine learning*, pp. 9323–9332. PMLR, 2021.

Gisbert Schneider, Petra Schneider, and Steffen Renner. Scaffold-hopping: how far can you jump?

QSAR & Combinatorial Science, 25(12):1162–1171, 2006.

Arne Schneuing, Yuanqi Du, Charles Harris, Arian Jamasb, Ilia Igashov, Weitao Du, Tom Blundell, Pietro Lio, Carla Gomes, Max Welling, et al. Structure-based drug design with equivariant ´ diffusion models. *arXiv preprint arXiv:2210.13695*, 2022.

Marwin HS Segler, Thierry Kogej, Christian Tyrchan, and Mark P Waller. Generating focused molecule libraries for drug discovery with recurrent neural networks. *ACS central science*, 4(1): 120–131, 2018.

Miha Skalic, Jose Jim ´ enez, Davide Sabbadin, and Gianni De Fabritiis. Shape-based generative ´
modeling for de novo drug design. *Journal of chemical information and modeling*, 59(3):1205– 1214, 2019.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pp. 2256–2265. PMLR, 2015.

Maria Sorokina, Peter Merseburger, Kohulan Rajan, Mehmet Aziz Yirik, and Christoph Steinbeck.

Coconut online: collection of open natural products database. *Journal of Cheminformatics*, 13 (1):2, 2021.

PW Sprague. Automated chemical hypothesis generation and database searching with catalyst.

Perspectives in Drug Discovery and Design, 3, 1995.

Teague Sterling and John J. Irwin. Zinc 15 - ligand discovery for everyone. Journal of Chemical Information and Modeling, 55(11):2324–2337, 2015. doi: 10.1021/acs.jcim.5b00559.

Carmen Stoffelen and Jurriaan Huskens. Soft supramolecular nanoparticles by noncovalent and host–guest interactions. *Small*, 12(1):96–119, 2016.

Jonatan Taminau, Gert Thijs, and Hans De Winter. Pharao: pharmacophore alignment and optimization. *Journal of Molecular Graphics and Modelling*, 27(2):161–169, 2008.

F Dean Toste, Matthew S Sigman, and Scott J Miller. Pursuit of noncovalent interactions for strategic site-selective catalysis. *Accounts of chemical research*, 50(3):609–615, 2017.

Oleg Trott and Arthur J Olson. Autodock vina: improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading. Journal of computational chemistry, 31(2):455–461, 2010.

Mikko J Vainio, J Santeri Puranen, and Mark S Johnson. Shaep: molecular overlay based on shape and electrostatic potential, 2009.

Clement Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, and Pascal Frossard. Digress: Discrete denoising diffusion for graph generation. arXiv preprint arXiv:2209.14734, 2022.

Clement Vignac, Nagham Osman, Laura Toni, and Pascal Frossard. Midi: Mixed graph and 3d denoising diffusion for molecule generation. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 560–576. Springer, 2023.

D. Wacker, G. Fenalti, M.A. Brown, V. Katritch, R. Abagyan, V. Cherezov, and R.C. Stevens. Crystal structure of the human beta2 adrenergic receptor in complex with the inverse agonist ici 118,551. Protein Data Bank, 2010a. PDB ID: 3ny8.

Daniel Wacker, Gustavo Fenalti, Monica A Brown, Vsevolod Katritch, Ruben Abagyan, Vadim Cherezov, and Raymond C Stevens. Conserved binding mode of human β2 adrenergic receptor inverse agonists and antagonist revealed by x-ray crystallography. *Journal of the American* Chemical Society, 132(33):11443–11445, 2010b.

Joel Wahl. Phesa: An open-source tool for pharmacophore-enhanced shape alignment. Journal of Chemical Information and Modeling, 2024.

Mingyang Wang, Chang-Yu Hsieh, Jike Wang, Dong Wang, Gaoqi Weng, Chao Shen, Xiaojun Yao, Zhitong Bing, Honglin Li, Dongsheng Cao, et al. Relation: A deep generative model for structure-based de novo drug design. *Journal of Medicinal Chemistry*, 65(13):9478–9492, 2022.

Joseph L Watson, David Juergens, Nathaniel R Bennett, Brian L Trippe, Jason Yim, Helen E Eisenach, Woody Ahern, Andrew J Borst, Robert J Ragotte, Lukas F Milles, et al. De novo design of protein structure and function with rfdiffusion. *Nature*, 620(7976):1089–1100, 2023.

Camille G Wermuth. Similarity in drugs: reflections on analogue design. *Drug Discovery Today*,
11(7-8):348–354, 2006.

Steven E Wheeler, Trevor J Seguin, Yanfei Guan, and Analise C Doney. Noncovalent interactions in organocatalysis and the prospect of computational catalyst design. Accounts of Chemical Research, 49(5):1061–1069, 2016.

Stephanie Wills, Ruben Sanchez-Garcia, Stephen D Roughley, Andy Merritt, Roderick E Hubbard, Frank von Delft, and Charlotte M Deane. Expanding the scope of a catalogue search to bioisosteric fragment merges using a graph database approach. *bioRxiv*, pp. 2024–08, 2024.

Andrew A Wylie, Joseph Schoepfer, Wolfgang Jahnke, Sandra W Cowan-Jacob, Alice Loo, Pascal Furet, Andreas L Marzinzik, Xavier Pelle, Jerry Donovan, Wenjing Zhu, et al. The allosteric inhibitor abl001 enables dual targeting of bcr–abl1. *Nature*, 543(7647):733–737, 2017.

Weixin Xie, Jianhang Zhang, Qin Xie, Chaojun Gong, Youjun Xu, Luhua Lai, and Jianfeng Pei.

Accelerating discovery of novel and bioactive ligands with pharmacophore-informed generative models. *arXiv preprint arXiv:2401.01059*, 2024.

Maria I Zavodszky, Anjali Rohatgi, Jeffrey R Van Voorst, Honggao Yan, and Leslie A Kuhn. Scoring ligand similarity in structure-based virtual screening. *Journal of Molecular Recognition*, 22(4): 280–292, 2009.

Chun-Hui Zhang, Elizabeth A Stone, Maya Deshmukh, Joseph A Ippolito, Mohammad M Ghahremanpour, Julian Tirado-Rives, Krasimir A Spasov, Shuo Zhang, Yuka Takeo, Shalley N Kudalkar, et al. Potent noncovalent inhibitors of the main protease of sars-cov-2 from molecular sculpting of the drug perampanel guided by free energy perturbation calculations. *ACS central science*, 7 (3):467–475, 2021.

Kangyu Zheng, Yingzhou Lu, Zaixi Zhang, Zhongwei Wan, Yao Ma, Marinka Zitnik, and Tianfan Fu. Structure-based drug design benchmark: Do 3d methods really dominate? arXiv preprint arXiv:2406.03403, 2024.

Qian-Yi Zhou, Jaesik Park, and Vladlen Koltun. Open3D: A modern library for 3D data processing.

arXiv:1801.09847, 2018.

Huimin Zhu, Renyi Zhou, Dongsheng Cao, Jing Tang, and Min Li. A pharmacophore-guided deep learning approach for bioactive molecular generation. *Nature Communications*, 14(1):6234, 2023.

Wonho Zhung, Hyeongwoo Kim, and Woo Youn Kim. 3d molecular generative framework for interaction-guided drug design. *Nature Communications*, 15(1):2688, 2024.

Yael Ziv, Brian Marsden, and Charlotte Deane. Molsnapper: Conditioning diffusion for structure based drug design. *bioRxiv*, pp. 2024–03, 2024.

## A Appendix

CONTENTS
1 Introduction 1 2 Related Work 3 3 Methodology 4 3.1 Defining representations of molecules and their interaction profiles . . . . . . . . . 4 3.2 Shape, electrostatic, and pharmacophore similarity scoring functions . . . . . . . . 5 3.3 Joint diffusion of molecules and their interaction profiles with *ShEPhERD* . . . . . 5 4 Experiments 7 5 Conclusion 10 A Appendix 19 A.1 Comparing *ShEPhERD* to REINVENT and virtual screening for natural product ligand hopping . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20 A.1.1 REINVENT convergence . . . . . . . . . . . . . . . . . . . . . . . . . . . 21 A.2 Additional details on experiments . . . . . . . . . . . . . . . . . . . . . . . . . . 23 A.2.1 Unconditional joint generation on *ShEPhERD*-GDB17 . . . . . . . . . . . 23 A.2.2 Conditional generation on *ShEPhERD*-GDB17 . . . . . . . . . . . . . . . 23 A.2.3 Natural product ligand hopping . . . . . . . . . . . . . . . . . . . . . . . 23 A.2.4 Bioactive hit diversification . . . . . . . . . . . . . . . . . . . . . . . . . 24 A.2.5 Bioisosteric fragment merging . . . . . . . . . . . . . . . . . . . . . . . . 28 A.3 Defining and analyzing validity of generated 3D molecules . . . . . . . . . . . . . 30 A.4 Unconditional and conditional generation metrics . . . . . . . . . . . . . . . . . . 36 A.5 Calculating interaction profiles from 3D molecular structures . . . . . . . . . . . . 38 A.6 Parameterization of 3D similarity scoring functions . . . . . . . . . . . . . . . . . 39 A.6.1 Shape similarity scoring function . . . . . . . . . . . . . . . . . . . . . . 41 A.6.2 ESP surface similarity scoring function . . . . . . . . . . . . . . . . . . . 42 A.6.3 Pharmacophore similarity scoring function . . . . . . . . . . . . . . . . . 42 A.7 Additional details on model design, training protocols, and sampling . . . . . . . . 43 A.7.1 Further details on *ShEPhERD*'s denoising modules . . . . . . . . . . . . . 43 A.7.2 Training and sampling procedures . . . . . . . . . . . . . . . . . . . . . . 45 A.7.3 Model hyperparameters . . . . . . . . . . . . . . . . . . . . . . . . . . . 50

| A.5.1   | Shapes/surfaces                  | 38   |
|---------|----------------------------------|------|
| A.5.2   | Electrostatic potential surfaces | 38   |

| A.5.3   | Pharmacophores   | 38   |
|---------|------------------|------|

| A.8                                                                    | Symmetry breaking in unconditional generation               | 52   |
|------------------------------------------------------------------------|-------------------------------------------------------------|------|
| A.9                                                                    | Characterizing the need for out-of-distribution performance | 53   |
| A.10 Additional examples of generated molecules                        | 55                                                          |      |
| A.11 Training and inference resources                                  | 58                                                          |      |
| A.12 Additional Experiments and Comparisons to Related Work            | 59                                                          |      |
| A.12.1 Comparison to SQUID for shape-conditioned generation            | 59                                                          |      |
| A.12.2 Comparisons against structure-based drug design models that explicitly encode the protein pocket                                                                        | 59                                                          |      |
| A.12.3 Comparisons against inpainting with DiffSBDD                    | 61                                                          |      |
| A.12.4 Comparisons against SynFormer on natural product ligand hopping |                                                             | 61   |

## A.1 Comparing Shepherd To Reinvent And Virtual Screening For Natural Product Ligand Hopping

REINVENT, a state-of-the-art baseline for generative molecular design and optimization, applies a reinforcement learning policy to iteratively update a SMILES recursive neural network (RNN) with a provided reward function (Blaschke et al., 2020). We applied REINVENT to the three natural product ligand hopping tasks defined in section 4, using a combination of our ESP and pharmacophore 3D similarity scoring functions (section 3.2) as REINVENT's reward function. We followed the REINVENT implementation in the Practical Molecular Optimization (PMO) benchmark (Gao et al., 2022). REINVENT was pretrained on the ZINC database (Sterling & Irwin, 2015) and was deployed with a batch size of 64, σ = 500, an experience replay of 24, and an oracle budget of 10, 000. Training was performed using an Adam optimizer with a learning rate of 5 × 10−4. Any SMILES which failed during pharmacophore scoring were assigned a score of 0.

Since REINVENT generates molecules in a 1D SMILES representation, we generate up to 5 conformers for each SMILES in order to apply our 3D similarity scoring functions as the reward function. The procedure to compute the reward for a single generated SMILES is as follows: 1) embed 5 conformers with RDKit's EmbedMultipleConfs function, which uses ETKDG; 2) optimize each conformer with MMFF94 for a maximum of 200 steps; 3) cluster the conformers with Butina clustering using an RMSD threshold of 0.1 A; 4) relax each remaining conformer with xTB in im- ˚ plicit water; 5) extract the ESP surface x3 with n3 = 400 and pharmacophore profile x4 of each relaxed conformer; 6) align each conformer to the target natural product by optimizing our 3D ESP scoring function; 7) calculate the ESP and pharmacophore similarity scores of the ESP-aligned conformers; 8) add the ESP and pharmacophore similarity scores to obtain one combined score per conformer; and 9) take the maximum score across the conformers as the reward. Fig. 5 compares the distributions of ESP and pharmacophore similarity for samples obtained by (1) using *ShEPhERD* (trained on *ShEPhERD*-MOSES-aq) to sample 2500 molecules from P(x1|x3, x4) via inpainting; (2) virtually screening (VS) 2500 random 3D molecules from ShEP- hERD-MOSES-aq; and (3) optimizing REINVENT with an oracle/sampling budget of 10,000. Note that REINVENT was pretrained on ZINC, and *ShEPhERD*'s training set (MOSES-aq) is a small subset of ZINC. Each 3D molecule generated by *ShEPhERD* was relaxed with xTB prior to realigning the relaxed structure to the natural product (via maximizing ESP similarity) and scoring the ESP and pharamcophore similarity of the aligned pose. 3D molecules sampled from *ShEPhERD*-MOSES-aq
(which are already xTB-relaxed) were directly aligned to the natural product in the same manner. Samples from REINVENT were scored using the procedure outlined in the preceding paragraph.

For both *ShEPhERD* and REINVENT, we only compare *valid* samples that have SA scores lower than 4.5. This means that although we initially obtain 2500 and 10000 samples from *ShEPhERD* and REINVENT, respectively, we only compare ∼500 samples from *ShEPhERD* against ∼9000 samples from REINVENT, for each case study. Despite the fewer number of samples, *ShEPhERD* still finds molecules beneath the SA-score threshold that score higher than molecules optimized by REINVENT, for all three natural products. Both *ShEPhERD* and REINVENT find much better molecules than those obtained by randomly sampling from the dataset.