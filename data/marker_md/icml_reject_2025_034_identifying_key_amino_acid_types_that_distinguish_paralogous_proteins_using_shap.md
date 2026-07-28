011

014 015 016

018

024

026

034

036

038

# Identifying key amino acid types that distinguish paralogous proteins using Shapley value based feature subset selection

Anonymous Authors<sup>1</sup>

# Abstract

Paralogous proteins have a common ancestor but have diverged in functionality. Using known machine learning algorithms, we present a datadriven method to identify the key amino acid types that play a role in distinguishing a given pair of proteins that are paralogs. We use an existing Shapley value based feature subset selection algorithm, SVEA, to identify the key amino acid types adequate to distinguish pairs of paralogous proteins. We refer to these as the amino acid feature subset (AF S). For a paralog pair, say proteins P and Q, its AF S is partitioned based on protein-wise importance as AF SpPq and AF SpQq using a linear classifier, SVM. To validate the significance of the AF S amino acids, we use multiple domain knowledge based methods : (a) multiple sequence alignment, and/or (b) 3D structure analysis, and/or (c) supporting evidence from biology literature. This method is computationally cheap, requires less data and can be used as an initial data-driven step for further hypothesis-driven experimental study of proteins. We demonstrate the results for 15 pairs of paralogous proteins. Code at [https://anonymous.](https://anonymous.4open.science/r/AFS_AAC_SVM-F3D9) [4open.science/r/AFS\\_AAC\\_SVM-F3D9](https://anonymous.4open.science/r/AFS_AAC_SVM-F3D9).

# 1. Introduction

Proteins form the fundamental machinery in living systems, having several vital functions such as DNA replication, catalysis, transport, environmental interaction, etc. Advancements in sequencing technologies have resulted in exponential growth of protein sequence databases [\(The](#page-11-0) [UniProt Consortium,](#page-11-0) [2020\)](#page-11-0). However, the number of experimentally verified annotations constitute a tiny fraction: only 0.57 of 250 million sequences in UniProtKB [\(The](#page-11-0)

[UniProt Consortium,](#page-11-0) [2020\)](#page-11-0) have manually reviewed annotations. Experimental methods for determining biological process level functions (transcription, DNA repair, etc.) are high-throughput whereas methods for molecular function (catalysis, ligand specificity, etc.) are low-throughput and hence are not scalable. The relationship between sequence and function is subtle and has not been fully decoded yet.

Paralogs are proteins that have a common ancestor but have diverged functionally. The functional difference in two paralogous proteins is considered to arise due to evolutionary changes in the sequences [\(Yang et al.,](#page-11-1) [2023\)](#page-11-1). A typical experiment to investigate the role of an (or a group of) amino acid(s) in the function of a protein is to perform a site-directed mutagenesis experiment: replace one or more amino acids and test the effect of the sequence change [\(Kresge et al.,](#page-9-0) [2006\)](#page-9-0). In this work, we provide an algorithmic ML pipeline, consisting both feature engineering and feature subset selection, as a quick and resource-cheap test to assess the likely outcome from a site-directed mutagenesis experiment. We use a diverse dataset of 15 paralog pairs. Our datasets show a range of sequence and function diversity (details in Appendix [B\)](#page-14-0). Longest common subsequence score (lcss) is a metric to quantify sequence diversity and median within-class lcss is ď 0.5 in 12 of the 15 datasets, and the median interclass lcss for the corresponding classes is less than withinclass lcss. Functional diversity, as discerned from biology literature, also shows large diversity from subtle functional differences (e.g., trypsin/chymotrypsin) to drastic (e.g., lysozyme c/α-lactalbumin). Function description is fine-grained (e.g., trypsin/chymotrypsin) as well as coarse grained (e.g, GPCRs).

Our findings are that small subsets of amino acids can discern differences between pairs of paralogs. The subset sizes are between 5 to 10, the median being 8. We provide validations from literature, MSA (a popular computational tool to assess evolutionary conservation) and logical consistencies; for many pairs such validations are more than one.

<sup>1</sup>[Anonymous Institution, Anonymous City, Anonymous Region,](#page-11-0) [Anonymous Country. Correspondence to: Anonymous Author](#page-11-0) <[anon.email@domain.com](#page-11-0)>.

Towards this, we view a protein as the composite of its constituent standard 20 amino acids. We use amino acid composition (AAC) features, a Shapley value [\(Shapley,](#page-10-0) [1953\)](#page-10-0) based feature subset selection algorithm (Shapley Value

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

108 109 based Error Apportioning, SVEA) [\(Tripathi et al.,](#page-11-2) [2020;](#page-11-2) [2021\)](#page-11-3), and a linear support vector machine (SVM) classifier [\(Steinwart & Christmann,](#page-10-1) [2008\)](#page-10-1) as tools to identify key amino acid types that can distinguish a given a pair of proteins that are paralogs. It yields quick results based on which biologists can conduct detailed experiments which are resource-intensive (time, cost, trained manpower, etc.).

The key results from our ML pipeline experiments are:

' Using known machine learning algorithms we demonstrate a data-driven method to identify key amino acids that

distinguish two paralogous proteins.

• The SVEA algorithm identifies a subset of amino acid types (referred to as AF S) adequate for distinguishing two paralogous proteins. The size of AF S ranges from

5 to 10 amino acids out of 20. (Table [1\)](#page-4-0)

• For a paralog pair, say protein families P and Q, the computed AF S is partitioned into AF SpPq and AF SpQq using a linear SVM, to determine the family-

wise importance of AF S. (Table [1\)](#page-4-0)

' Domain knowledge based validation of AF S: The significance of the amino acids in AF S was validated for 14 datasets using various methods like (a) multiple sequence alignment (MSA) and/or (b) structural analysis and/or (c) supporting evidence from literature that report

structural/functional role of these amino acids.

' Logical consistencies in the pair-wise AF S of three paralogous proteins (globins, Section [3.1.7,](#page-6-0) and GPCRs, Section [3.1.8\)](#page-6-1). If families P vs Q and P vs R have AF S<sup>1</sup> and AF S2, then,

• we find common amino acids in AF S1pPq and

AF S2pPq, except for one pair.

• amino acids in AF S<sup>1</sup> X AF S<sup>2</sup> are either excluded from AF S3, which is from Q vs R, or have much lower

Shapley value in AF S1, AF S2, or AF S3.

' Validation of AF S using test data (Section [3.2\)](#page-7-0): The composition of amino acids is sufficient to classify several paralog pairs. A linear SVM classifies with high test scores (70-99%) using only the composition of AF S amino acids

as features. (Appendix Table [E5\)](#page-20-0)

' AF S are top ranked features with an alternate feature ranking measure, Marginal Contribution feature importance

(MCI) [\(Catav et al.,](#page-8-0) [2021\)](#page-8-0). (Appendix Table [E6\)](#page-27-0)

Shapley values based feature attribution methods are popular for explaining machine learning models [\(Rozemberczki](#page-10-2) [et al.,](#page-10-2) [2022\)](#page-10-2). One such method is SHAP [\(Lundberg & Lee,](#page-9-1) [2017\)](#page-9-1), which assigns attribution scores to input features based on a model's output for a given instance input. Another method is SAGE [\(Covert et al.,](#page-8-1) [2020\)](#page-8-1), which assigns feature attribution scores based on a model's loss computed at the dataset level. Unlike these methods, where feature

attributions are based on a trained model, the SVEA algorithm that we use for our task assigns scores to the features based on the distribution of the data points in the feature space and their ground truth labels. The SVEA algorithm uses a function vpSq, which acts as a measure of inter-class linear separation between the data points in the space of the feature subset S. The scores assigned to the features are Shapley values computed using this function vp¨q. We also use an alternate feature ranking method, i.e. the Marginal Contribution Feature Importance (MCI) [\(Catav et al.,](#page-8-0) [2021\)](#page-8-0). MCI is an axiomatic approach that was proposed as an alternative to Shapley values to score and rank features. We find close agreement between the AF S computed using SVEA and the top-ranked amino acids using MCI.

Use of deep learning methods trained on large datasets is becoming commonplace in Biology; for example, prediction of molecular function via EC number or GO annotation [\(Bileschi et al.,](#page-8-2) [2022;](#page-8-2) [Sanderson et al.,](#page-10-3) [2023\)](#page-10-3), identifying input sequence regions relevant to model output [\(Zhou et al.,](#page-11-4) [2016\)](#page-11-4) and learning sequence-function mapping from deep mutational scanning experiment data [\(Song et al.,](#page-10-4) [2021\)](#page-10-4). The use of large datasets for training makes this approach highly resource-intensive. The approach we present herein needs much smaller datasets and, consequently, (i) is computationally cheap and (ii) has far wider applicability since labelled data validated by wet lab experiments is limited.

# 2. Methodology

We discuss the main components of our methodology.

#### 2.1. AAC features

Consider a paralogous pair of proteins, families P and Q. We first curate a set of sequences, say D<sup>P</sup> and DQ, from a standard protein sequence database, SwissProt [\(The](#page-11-0) [UniProt Consortium,](#page-11-0) [2020\)](#page-11-0), with n<sup>P</sup> and n<sup>Q</sup> number of sequences each from families P and Q respectively. For a protein sequence p <sup>p</sup>j<sup>q</sup> " pp pjq 1 , p pjq 2 , . . . , p pjq L q of length L with p pjq k P t1, 2, ¨ ¨ ¨ , 20u corresponding to the standard 20 amino acids, the AAC feature x AAC <sup>j</sup> P r0, 1s <sup>20</sup> for p pjq is computed as follows,

$$x_{j,i}^{AAC} = \frac{1}{L} \sum_{k=1}^L \mathbf{1}_{\{p_k^{(j)}=i\}}, \quad \forall i \in [20]$$

So x AAC j,i is the normalised count of the standard amino acid i, i P t1, 2, ¨ ¨ ¨ , 20u, in a protein p pjq .

## 2.2. Feature subset selection using SVEA

Given a set, N, of features from the protein sequences of P and Q, we try to find the features S Ď N that contribute the most to the linear separation of P and Q sequences. With

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

AAC features, we have N " t1, 2, . . . , 20u corresponding to each of the standard 20 amino acid types.

We utilise the Shapley value based feature ranking and subset selection algorithm, SVEA [\(Tripathi et al.,](#page-11-2) [2020;](#page-11-2) [2021\)](#page-11-3), to identify the most important feature subset S Ď N. Shapley value is a well known solution concept from cooperative game theory [\(Shapley,](#page-10-0) [1953;](#page-10-0) [Narahari,](#page-10-5) [2014\)](#page-10-5) for distributing the total worth of a coalition of players fairly among each of them by quantifying each player's effective marginal contribution. The SVEA algorithm considers the binary classification task as a cooperative game among the features, with a function vpSq as the worth of every feature subset S. vpSq acts as a measure of linear separation between the classes in the feature space of S. Accounting for classimbalance, we define vpSq using a class-balanced hinge loss function tr erpSq, which is defined as,

$$\begin{aligned} \text{tr\_er}(S) &= \min_{w, \xi_j} \frac{1}{2n_P} \sum_{j=1}^{n_P} \xi_j + \frac{1}{2n_Q} \sum_{j=n_P+1}^{n_Q} \xi_j \\ \text{s.t. } y_j \left( \sum_{i \in S} w_i x_{j,i}^{AAC} + b \right) &\geq 1 - \xi_j, \quad \forall j \in [n_P + n_Q] \\ \xi_j &\geq 0, \quad \forall j \in [n_P + n_Q] \end{aligned}$$

and vpSq " tr erpHq ´ tr erpSq. The minimizer in the above finds a linear hyperplane with the least class-balanced hinge loss in the feature space of S. H is the empty set and tr erpHq " 1, therefore, vpSq " 1 ´ tr erpSq. tr erpSq " 0 implies vpSq " 1, i.e., the two classes are completely linearly separable in the feature space of S. The maximum value of tr erpSq possible is 1.

The Shapley value ϕpiq for a feature i P N is computed as,

$$\phi(i) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} (v(S \cup \{i\}) - v(S)).$$

Thus, ϕpiq is a weighted sum of the marginal contribution of feature i to all the possible feature subsets that do not contain i. Shapley values are unique solution concepts satisfying the axioms - efficiency, symmetry and marginality [\(Young,](#page-11-5) [1985\)](#page-11-5). The higher the ϕpiq, the higher the contribution of feature i to the linear separation between the classes and, consequentially, the higher the importance of feature i distinguishing the classes.

Exact Shapley value computations are known to be exponential time. Hence, they are computed using a linear time (in number of features) Monte Carlo approximation [\(Castro](#page-8-3) [et al.,](#page-8-3) [2009\)](#page-8-3) in the SVEA algorithm. As the number of features is small (20), good approximations can be computed fast via larger sampling. More details of the SVEA algorithm are given in Appendix Section [C.](#page-18-0)

Data-driven cutoff for selecting AF S: The efficiency axiom of Shapley value implies, ř<sup>20</sup> i"1 ϕpiq " vpNq. If all

features have equal contribution in achieving vpNq, then ϕpiq " <sup>v</sup>pN<sup>q</sup> <sup>20</sup> , @i P N. Consequentially, if a feature i had lesser contribution than others then ϕpiq ă <sup>v</sup>pN<sup>q</sup> <sup>20</sup> . Therefore, we set ϕcutof f " vpNq <sup>20</sup> for selecting the key distinguishing amino acid feature subset, AF S " ti : ϕpiq ě ϕcutof f u. Each of the features in AF S uniquely corresponds to d ď 20 amino acids from the standard 20.

#### 2.3. Protein family-wise partition of AF S using SVM

We train a linear SVM, to classify P vs Q, using the composition of the amino acids in AF S as the features, i.e. using x AF S <sup>j</sup> P r0, 1s d , with x AF S j,i<sup>1</sup> " x AAC j,i and each i <sup>1</sup> P t1, 2, ¨ ¨ ¨ , du uniquely maps to a i P AF S. We use these linear SVM weights w P R d to divide the set AF S into disjoint sets AF SpPq and AF SpQq based on the sign of the weights. Since x AF S j,i<sup>1</sup> ě 0 @i <sup>1</sup> P rds, the sign of the linear classifier weight w<sup>i</sup> <sup>1</sup> indicates which class is relatively prominent in the amino acid corresponding to i 1 . So if the `1 class is P, then we divide AF S classwise as AF SpPq " ti <sup>1</sup> P rds : w<sup>i</sup> <sup>1</sup> ą 0u and similarly AF SpQq " ti <sup>1</sup> P rds : w<sup>i</sup> <sup>1</sup> ă 0u. See Appendix Section [D](#page-18-1) for details on SVM training.

A flowchart summarizing the steps for computing AF SpPq and AF SpQq is shown in Figure [1.](#page-3-0)

#### 2.4. Validation of AF S

Literature evidence: For 14 different paralog protein pairs, we provide supporting evidence from protein biology literature for the significance of amino acids in AF S in the functional specificity of the protein pair.

MSA analysis: We also compute multiple sequence alignment (MSA) of randomly selected sequences from D<sup>P</sup> and D<sup>Q</sup> and analyze the conservation of AF SpPq and AF SpQq amino acids within and across the respective families (Figure [2\)](#page-5-0). MSA algorithms [\(Edgar & Batzoglou,](#page-8-4) [2006\)](#page-8-4) aim to align multiple protein sequences by inserting gaps in the sequences while optimizing an objective. The objective is usually to minimize the number of gaps inserted while maximizing an overall score that promotes the alignment of similar (based on physicochemical properties) amino acids at a given position. The alignments are often used as a tool to determine homologous relationships between proteins and identify conserved or mutated regions in them.

Structural analysis: For paralog pairs that together function as heteromers (protein complexes made up of different types of proteins), we perform structural analysis to validate the role of AF S in the heteromeric structure formed by the paralog pair (Sections [3.1.7,](#page-6-0) [3.1.3](#page-3-1) and [3.1.4\)](#page-3-2).

Using test data: We test the classifier trained in Section [2.3](#page-2-0) on a test data. (Details on test data in Appendix Section [A.1\)](#page-12-0).

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

![](_page_3_Diagram_1.jpeg)

Figure 1: Flowchart summarizing the steps in our ML pipeline to compute the key amino acid types, AF S, that distinguish two paralogous proteins, using amino acid composition (AAC) features, Shapley value based SVEA algorithm for feature subset selection and class-wise feature subsets using linear SVM. Lysozyme C and α-Lactalbumin are used here as representative examples of paralog pairs. AF S identified for other paralog pairs are given in Table [1.](#page-4-0)

In general, we find an imbalance in the number of sequences for the two paralogous proteins. It is known that accuracy is not a well-suited performance measure of the classifier in class imbalance settings. Therefore, we use the arithmetic mean of sensitivity and specificity (AM) to measure the performance of the classifier [\(Brodersen et al.,](#page-8-5) [2010\)](#page-8-5).

Using marginal contribution feature importance (MCI): We check agreement of AF S with another feature ranking method, MCI [\(Catav et al.,](#page-8-0) [2021\)](#page-8-0). See Appendix Section [E.4](#page-27-1) for details on MCI computation.

# 3. Results and Discussions

## 3.1. Role of the amino acids identified in AF S

For 15 paralog pairs, we discuss the significance of the amino acids identified in the respective AF S (Table [1\)](#page-4-0).

## 3.1.1. LYSOZYME C AND α-LACTALBUMIN

Literature evidence: Amino acids D and E of AF Spα-Lactalbuminq are found in the Ca<sup>2</sup>` and Zn<sup>2</sup>` binding sites respectively of α-lactalbumin [\(Permyakov &](#page-10-6) [Berliner,](#page-10-6) [2000;](#page-10-6) [Permyakov,](#page-10-7) [2020\)](#page-10-7). All α-lactalbumins studied so far are known to bind Ca<sup>2</sup>` and Zn<sup>2</sup>` whereas several (but not all) lysozymes do not bind Ca<sup>2</sup>`.

MSA analysis: (Figure [2a\)](#page-5-0) AF Spα-Lactalbuminq and AF SpLysozyme Cq amino acids (Table [1\)](#page-4-0) are significantly conserved in respective families.

## 3.1.2. TRYPSIN AND CHYMOTRYPSIN

Literature evidence: Y and W get the highest Shapley value ϕp¨q in AF SpTrypsinq and AF SpChymotrypsinq respectively (Table [1](#page-4-0) and Figure [E6b\)](#page-19-0). In experiments to convert trypsin to chymotrypsin [\(Hedstrom et al.,](#page-9-2) [1994;](#page-9-2) [Hedstrom,](#page-9-3) [2002\)](#page-9-3) it has been shown that Y to W conversion in loop-3 of trypsin leads to significant increase in chymotrypsin activity. We do not find S, H and D in AF S, which are important for the function of both families and are known as the catalytic triad [\(Dodson & Wlodawer,](#page-8-6) [1998\)](#page-8-6).

#### 3.1.3. TUBULIN-α AND TUBULIN-β

MSA analysis: (Appendix Figure [E10\)](#page-24-0) AF SpTubulin-αq and AF SpTubulin-βq amino acids are significantly conserved in respective families.

Structural analysis of AF S: Tubulins typically exist as heterodimers, consisting of two subunits: tubulin-α and tubulin-β [\(Muhlethaler et al.](#page-10-8) ¨ , [2021\)](#page-10-8). We looked at the contact residues of a tubulin-α chain and tubulin-β chain in the 3D structure of tubulin-α/β heterodimer (PDB IDs: 3JAR, 5N5N). We see that the contact points of the tubulin-α chain in the heterodimer have more AF SpTubulin-αq amino acids than AF SpTubulin-βq. Similarly, AF SpTubulin-βq amino acids are more than AF SpTubulin-αq at the contact point of the tubulin-β chain in the heterodimer. Thus, the amino acids identified in AF S can be considered to be significant towards the quaternary structure of tubulin-α/β heterodimer. Appendix Section [E.2](#page-23-0) has more details.

## 3.1.4. HISTONE H2A AND HISTONE H2B

MSA analysis: (Appendix Figure [E11\)](#page-25-0), AF SpHistone H2Aq and AF SpHistone H2Bq amino acids are significantly conserved in respective families.

226

228

231

234

236

238

254

256

258

260

264

266

268

271

274

Table 1: AF S and its class-wise partition computed for 15 paralog pairs. The number of unique sequences from the SwissProt [\(The UniProt Consortium,](#page-11-0) [2020\)](#page-11-0) database used for computing AF S is given inside parenthesis p¨q for each protein family. Data collection details are in Appendix Section [A.1.](#page-12-0) AF S amino acids are written in decreasing Shapley values from left to right for each paralog pair. Figures [3](#page-7-1) and [E6](#page-19-0) show the Shapley value of the amino acids for each paralog pair. For globins and GPCRs, common acids across different AF S within a paralog triplet are colour-coded.

| Paralog pair Amino acid feature subset , AF S                   |      |      |             | Class-wise    | AF S      |     |     | parition |      |     |     |        |
|-----------------------------------------------------------------|------|------|-------------|---------------|-----------|-----|-----|----------|------|-----|-----|--------|
| Lysozyme C (74) and                                             |      |      |             |               |           |     |     |          |      |     |     |        |
| α -Lactalbumin (22) t I, A, D, N, G, R, E, F, L, W u            |      |      |             |               |           |     |     |          |      |     |     |        |
|                                                                 | AF S | p    | α           | -Lactalbumin  | q         | “   | t   | I,       | D,   | E,  | F,  | L u    |
|                                                                 | AF S | p    | Lysozyme    | C             | q “       | t   | A,  | N,       | G,   | R,  |     | W u    |
| Trypsin (66) and                                                |      |      |             |               |           |     |     |          |      |     |     |        |
| Chymotrypsin (17) t Y, W, T, A, V, K, P u                       |      |      |             |               |           |     |     |          |      |     |     |        |
|                                                                 |      |      | AF S        | p Trypsin     | q         | “   |     | t Y,     | A u  |     |     |        |
|                                                                 | AF S | p    |             | Chymotrypsin  | q         | “   | t   | W,       | T,   | V,  | K,  | P u    |
| Tubulin- α (117) and                                            |      |      |             |               |           |     |     |          |      |     |     |        |
| Tubulin- β (191) t M, Q, K, N, F, I, H, A, C, Y u               |      |      |             |               |           |     |     |          |      |     |     |        |
|                                                                 | AF S |      | p           | Tubulin- α q  | “         | t   | K,  | I,       | H,   | C,  | Y   | u      |
|                                                                 | AF S |      | p           | Tubulin- β q  | “         | t   | M,  | Q,       | N,   | F,  | A   | u      |
| Histone H2A (180) and                                           |      |      |             |               |           |     |     |          |      |     |     |        |
| Histone H2B (177) t L, G, S, M, K, N, T, Y, F u                 |      |      |             |               |           |     |     |          |      |     |     |        |
|                                                                 |      | AF S | p           | Histone       | H2A       | q   | “   | t L,     | G,   | N   | u   |        |
| AF S                                                            | p    |      | Histone     | H2B q         | “         | t   | S,  | M,       | K,   | T,  | Y,  | F u    |
| Interleukin-1 α (16) and                                        |      |      |             |               |           |     |     |          |      |     |     |        |
| Interleukin-1 β (25) t C, G, T, S, V, Q, A, N, P u              |      |      |             |               |           |     |     |          |      |     |     |        |
|                                                                 | AF S |      | p           | Interleukin-1 | α         | q   | “   | t T,     | S,   | A,  | N   | u      |
|                                                                 | AF S | p    |             | Interleukin-1 | β q       | “   | t   | C,       | G,   | V,  | Q,  | P u    |
| Cytochrome P450 CYP3                                            |      |      |             |               |           |     |     |          |      |     |     |        |
| (32) and CYP51 (32) t H, F, G, K, A, P, N u                     |      |      |             |               |           |     |     |          |      |     |     |        |
|                                                                 |      |      | AF S p      | CYP3 q        | “ t       | F,  |     | K,       | P,   | N u |     |        |
|                                                                 |      |      | AF S p      | CYP51         | q “       |     | t   | H,       | G, A | u   |     |        |
| Myoglobin (107) and                                             |      |      |             |               |           |     |     |          |      |     |     |        |
| Hemoglobin- α (303) AF S 1 “ t E, S, Y, V, K, P, I, G, C, W u   |      |      |             |               |           |     |     |          |      |     |     |        |
|                                                                 | AF S | 1    | p           | Myoglobin     | q “       | t   | E,  | K,       | I,   | G,  | W   | u      |
| AF S                                                            | 1    | p    |             | Hemoglobin-   | α q       | “   | t   | S,       | Y ,  | V , | P,  | C u    |
| Myoglobin (107) and                                             |      |      |             |               |           |     |     |          |      |     |     |        |
| Hemoglobin- β (285) AF S 2 “ t K, V, C, E, W, N, F, M, Y, I u   |      |      |             |               |           |     |     |          |      |     |     |        |
|                                                                 | AF S |      | 2 p         | Myoglobin     | q         | “   | t   | K,       | E,   | M,  | I   | u      |
| AF S                                                            | 2 p  |      | Hemoglobin- | β             | q “       | t   | V   | , C,     | W,   |     | N,  | F, Y u |
| Hemoglobin- α (303) and                                         |      |      |             |               |           |     |     |          |      |     |     |        |
| Hemoglobin- β (285) AF S 3 “ t W, P, N, S, G u                  |      |      |             |               |           |     |     |          |      |     |     |        |
|                                                                 |      | AF S | 3 p         | Hemoglobin-   |           |     | α q | “        | t P, | S   | u   |        |
|                                                                 | AF S |      | 3 p         | Hemoglobin-   |           | β q | “   | t        | W,   | N,  | G   | u      |
| Rhodopsin-like (181) and                                        |      |      |             |               |           |     |     |          |      |     |     |        |
| Glutamate-like (89) AF S 1 “ t D, Q, E, G, M, L u               |      |      |             |               |           |     |     |          |      |     |     |        |
|                                                                 |      |      | AF S 1      | p Rhodopsin   |           | q   | “   | t        | M,   | L u |     |        |
|                                                                 | AF S |      | 1 p         | Glutamate     | q         | “   | t   | D,       | Q,   | E,  | G u |        |
| Secretin-like (90) and                                          |      |      |             |               |           |     |     |          |      |     |     |        |
| Glutamate-like (89) AF S 2 “ t W, H, Y, V, D u                  |      |      |             |               |           |     |     |          |      |     |     |        |
|                                                                 |      |      | AF S 2 p    | Secretin      | q         | “   | t   | W,       | H,   | Y   | u   |        |
|                                                                 |      |      | AF S 2      | p Glutamate   |           | q   | “   | t V      | ,    | D u |     |        |
| Rhodopsin-like (181) and                                        |      |      |             |               |           |     |     |          |      |     |     |        |
| Secretin-like (90) AF S 3 “ t W, E, M, S, V, H, Q, A u          |      |      |             |               |           |     |     |          |      |     |     |        |
|                                                                 | AF S |      | 3 p         | Rhodopsin     | q         | “   | t   | M,       | S,   | V , | A u |        |
|                                                                 |      | AF S | 3 p         | Secretin q    | “         | t   | W,  | E,       | H,   |     | Q u |        |
| Rhodopsin-like GPCRs                                            |      |      |             |               |           |     |     |          |      |     |     |        |
| Aminergic receptors (186)                                       |      |      |             |               |           |     |     |          |      |     |     |        |
| and Lipid receptors (113) AF S 1 “ t L, P, E, W, F, M, D u      |      |      |             |               |           |     |     |          |      |     |     |        |
| AF S                                                            | 1 p  |      | Aminergic   |               | receptors |     | q   | “ t      | P,   | E,  |     | W, D u |
|                                                                 | AF S |      | 1 p Lipid   |               | receptors | q   | “   | t        | L,   | F,  | M   | u      |
| Aminergic receptors (186)                                       |      |      |             |               |           |     |     |          |      |     |     |        |
| and Peptide receptors (367) AF S 2 “ t L, F, E, M, K, D, V, R u |      |      |             |               |           |     |     |          |      |     |     |        |
| AF S                                                            | 2 p  |      | Aminergic   |               | receptors |     | q   | “        | t E, | K,  |     | D, R u |
| AF S                                                            | 2    | p    | Peptide     | receptors     |           | q   | “   | t        | L,   | F,  | M,  | V u    |
| Lipid receptors (113) and                                       |      |      |             |               |           |     |     |          |      |     |     |        |
| Peptide receptors (367) AF S 3 “ t P, R, G, I, W, S, V u        |      |      |             |               |           |     |     |          |      |     |     |        |
|                                                                 | AF S |      | 3 p Lipid   |               | receptors |     | q   | “ t      | R,   | G,  | S   | u      |
| AF S                                                            | 3    | p    | Peptide     | receptors     |           | q   | “   | t        | P,   | I,  | W,  | V u    |

tameric structure comprising of two H2A/H2B dimers and one H3/H4 tetramer [\(Dutta et al.,](#page-8-7) [2001\)](#page-8-7). We looked at the contact residues of an H2A chain and H2B chain in the heteroocatmer structure of histone (PDB IDs: 3KWQ, 1AOI). We find that the contact points of H2A chain in the heterooctamer have more AF SpHistone H2Aq amino acids than AF SpHistone H2Bq. This is interesting since AF SpHistone H2Aq has only three amino acids, while AF SpHistone H2Bq has six amino acids. Similarly, the contact points of H2B chain in the heterooctamer have more AF SpHistone H2Bq amino acids than AF SpHistone H2Aq. Thus, the amino acids identified in AF S can be considered to be significant towards the quaternary structure of the histone heterooctamer. See Appendix Section [E.3](#page-23-1) for more details.

#### 3.1.5. INTERLEUKIN-1 α AND INTERLEUKIN-1 β

Literature Evidence: C has the highest Shapley value and is in AF SpInterleukin-1 βq. Deleting C results in loss of activity in Interleukin-1 β [\(Veerapandian et al.,](#page-11-6) [1992\)](#page-11-6). We do not find such studies for Interleukin-1 α.

MSA analysis: (Appendix Figure [E12\)](#page-26-0) AF SpInterleukin-1 αq and AF SpInterleukin-1 βq amino acids show significant conservation in respective families.

## 3.1.6. CYTOCHROME P450 CYP3 AND CYP51

Literature evidence: H, F and G, in the respective order, have the highest Shapley value ϕp¨q for this paralogous pair (Table [1](#page-4-0) and Figure [E6f\)](#page-19-0). H and G with the highest ϕp¨q in AF SpCYP51q have been reported [\(Nitahara et al.,](#page-10-9) [2001;](#page-10-9) [Lepesheva & Waterman,](#page-9-4) [2004;](#page-9-4) [2007;](#page-9-5) [Strushkevich et al.,](#page-10-10)

![](_page_5_Figure_1.jpeg)

 Figure 2: Multiple sequence alignment of sequences from the respective families in (a), (b) and (c). Within each alignment, 15 sequences on the left are from one family, and those on the right are from the other family in each of (a), (b) and (c). The sequences are randomly selected from the train set of the families. For each aligned sequence in (a) AF Spα-Lactalbuminq amino acids are in green and AF SpLysozyme Cq are in red, in (b) the amino acids in AF S1pMyoglobinq are in green and AF S1pHemoglobin-αq are in red, and in (c) the amino acids in AF S2pHemoglobin-αq are in green and AF S2pHemoglobin-βq are in red. The intensity of the color is proportional to the Shapley value ϕpiq of the amino acid i (Figures [3](#page-7-1) and [E6\)](#page-19-0).

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

[2010\)](#page-10-10) to be important in the enzymatic activity of CYP51. Mutation of these amino acids at specific positions has been shown to result in a decrease in the activity of the enzyme [\(Lepesheva & Waterman,](#page-9-5) [2007;](#page-9-5) [2004\)](#page-9-4). Similarly, F with the highest ϕp¨q in AF SpCYP3q is also known to be important in the enzymatic activity of CYP3 [\(Qiu et al.,](#page-10-11) [2008;](#page-10-11) [Denisov](#page-8-8) [et al.,](#page-8-8) [2019;](#page-8-8) [Zhang et al.,](#page-11-7) [2024\)](#page-11-7). A cluster of F residues in CYP3 is known to form a substrate-binding pocket with an active site [\(Zhang et al.,](#page-11-7) [2024\)](#page-11-7).

#### 3.1.7. GLOBINS

MSA analysis: (Figures [2b,2c](#page-5-0) and Appendix Figure [E8\)](#page-22-0) For the three globin paralog pairs (Table [1\)](#page-4-0), we observe in the MSA, conservation of the class-wise partition of AF S in the respective families.

Structural analysis of AF S: Myoglobin is a monomer, while α and β chains together constitute hemoglobin, a tetramer of composition α2β<sup>2</sup> [\(Dill et al.,](#page-8-9) [2017\)](#page-8-9). We superimposed the 3D structures of myoglobin, hemoglobinα and hemoglobin-β (PDB IDs: 3RGK, 1HHO) and mapped the α, β contact residues (based on [\(Shionyu et al.,](#page-10-12) [2001\)](#page-10-12)) of hemoglobin tetramer to that of myoglobin. We find that the amino acids K, E, I, which are common in AF S1pMyoglobinq and AF S2pMyoglobinq, are less in number at the contact residues of hemoglobin tetramer and more in number at the corresponding locations in myoglobin, which is a monomer (see Appendix Figure [E7\)](#page-21-0).

Literature evidence: W with a significantly high Shapley value ϕpWq (Figure [3b\)](#page-7-1), is present in AF S3pHemoglobin-βq. It is highly conserved at position 40 in the MSA (Figure [2c\)](#page-5-0) in hemoglobin-β sequences as compared to hemoglobin-α sequences. This W at position 40 has been determined to be present in hemoglobin-β at one of its contact positions to hemoglobin-α in the tetrameric structure [\(Shionyu et al.,](#page-10-12) [2001\)](#page-10-12) and is, therefore, a structurally and functionally significant residue. C, present in AF S1pHemoglobin-αq and AF S2pHemoglobin-βq, has been shown to play an important role in the tetrameric structure of hemoglobin formed by α and β hemoglobins [\(Kan et al.,](#page-9-6) [2013\)](#page-9-6).

Logical consistencies in AF S (refer to Table [1](#page-4-0) (Globins) for AF S1, AF S2, AF S3):

' AF S<sup>1</sup> X AF S<sup>2</sup> " tE, Y, V, K, I, C, Wu. Except for W with the least Shapley value in AF S<sup>1</sup> (Figure [3a\)](#page-7-1), the

- remaining are excluded from AF S3.
- *Explanation*:V, Y, C in AF S1pHemoglobin-αq X AF S2pHemoglobin-βq can be expected not to be *key* in AF S<sup>3</sup> for distinguishing α vs β hemoglobin. ' AF S<sup>2</sup> X AF S<sup>3</sup> " tW, Nu. N is excluded from AF S1, while W gets the least Shapley value in AF S<sup>1</sup> (Figure [3a\)](#page-7-1).

' AF S<sup>3</sup> X AF S<sup>1</sup> " tW, P, S, Gu. tP, S, Gu are excluded from AF S2, while W gets the least Shapley value in AF S1. The Shapley value for W is very close to the cut-off in AF S<sup>1</sup> (Figure [3a\)](#page-7-1). If it is dropped from AF S1, then the exclusion principle illustrated above would be more prominent as in GPCRs (Section [3.1.8\)](#page-6-1).

#### 3.1.8. G-PROTEIN COUPLED RECEPTORS (GPCRS)

Literature evidence: W (with highest Shapley value ϕp¨q) and H common in AF S2pSecretinq and AF S3pSecretinq (Table [1](#page-4-0) and Figure [3\)](#page-7-1), are well conserved at multiple positions with structural importance and functional importance in secretin-like GPCR sequences [\(Cary et al.,](#page-8-10) [2022;](#page-8-10) [Harmar,](#page-9-7) [2001\)](#page-9-7). Mutating certain conserved W leads to a loss in expression of this GPCR at the cell surface, where it functions [\(Cary et al.,](#page-8-10) [2022\)](#page-8-10). H present in the intracellular loop region is also known to be important in the activation of certain secretin-like GPCRs [\(Harmar,](#page-9-7) [2001\)](#page-9-7).

M common in AF S1pRhodopsinq and AF S3pRhodopsinq has been found to be present at important binding pockets and a position important for activation of the GPCR [\(Okada et al.,](#page-10-13) [2001;](#page-10-13) [Sakmar et al.,](#page-10-14) [2002\)](#page-10-14). S from AF S3pRhodopsinq is found at multiple major phosphorylation sites (see [Okada et al.](#page-10-13) [2001](#page-10-13) for details) in Rhodopsin.

Mutating D at two positions has been shown to affect glutamate binding of glutamate receptor GPCRs [\(Jingami](#page-9-8) [et al.,](#page-9-8) [2003\)](#page-9-8). D is common in AF S1pGlutamateq and AF S2pGlutamateq and has highest Shapley value in AF S1.

E and D common in AF S1pAminergicq and AF S2pAminergicq are present at binding sites of important ligands (like histamine/serotonin) of aminergic receptors [\(Vass et al.,](#page-11-8) [2019\)](#page-11-8).

Logical consistencies in AF S of GPCRs (refer to Table [1](#page-4-0) (GPCRs) for AF S1, AF S2, AF S3):

' AF S<sup>1</sup> X AF S<sup>2</sup> " tDu, is excluded from AF S3. ' AF S<sup>2</sup> X AF S<sup>3</sup> " tW, H, V u, is excluded from AF S1. ' AF S<sup>3</sup> X AF S<sup>1</sup> " tQ, E, Mu, is excluded from AF S2.

Logical consistencies in AF S of Rhodopsin-like GPCR subfamilies (refer to Table [1](#page-4-0) (Rhodopsin-like GPCRs) for AF S1, AF S2, AF S3):

' AF S<sup>1</sup> X AF S<sup>2</sup> " tL, E, F, M, Du, is excluded from AF S3. ' AF S<sup>2</sup> X AF S<sup>3</sup> " tR, V u, is excluded from AF S1. ' AF S<sup>3</sup> X AF S<sup>1</sup> " tP, Wu is excluded from AF S2.

The explanations for these consistencies are similar to that in globins (Section [3.1.7\)](#page-6-0).

![](_page_7_Figure_1.jpeg)

Figure 3: Shapley value (ϕpiq) for AAC features computed using SVEA. See Appendix Figure [E6](#page-19-0) for remaining paralogs.

#### 3.2. Validation of AF S using test data

The classification scores on test data for the classifiers trained using AAC and AF S features, respectively, are reported in Appendix Table [E5.](#page-20-0) Using AF S features, the test AM scores are at least 70%. For 13 of 15 paralog pairs, the scores are greater than 83%, and for 8 of 15 paralog pairs, it is greater than 90%. Details of the test data are provided in Appendix Section [A.1.](#page-12-0)

#### 3.3. Marginal contribution feature importance (MCI) of AF S

For an AF S of size d, the top-d amino acids ranked by MCI differ with AF S only in at the most two amino acids. For 8 of 15 datasets, AF S and top-d MCI sets are the same, while only for two datasets do they differ in two amino acids. For all 15 datasets, at least the top-3 MCI amino acids are in AF S. For 11 of these datasets, at least the top-5 MCI amino acids are in AF S. (Appendix Table [E6\)](#page-27-0)

# 4. Conclusion

We demonstrated an ML pipeline to identify the key amino acid types, AF S, that distinguish a pair of paralogous proteins. The role of AF S in functionally distinguishing the paralog pairs was validated using various sources of domain knowledge. The robustness of this approach, as demonstrated by considering a diverse set of paralogous protein

pairs, illustrates its wider applicability. Identification of AF S can be used as an initial data-driven step before doing more detailed experimental investigations, like site-directed mutagenesis [\(Bachman,](#page-8-11) [2013\)](#page-8-11) resolving sequence-function relationship. As the size of AF S is small (5-10 amino acids of 20), significantly less number of mutations can be tried.

As our pipeline works without using the sequence order information of the amino acids in the protein, it posits an interesting question to biologists : how amino acid composition by itself is able to distinguish paralogs given ample evidence that 3D structure and function are conserved despite sequence divergence [\(Lau et al.,](#page-9-9) [2015\)](#page-9-9)! Notably, amino acids in the AF S typically occur more than once in the sequence, but our method is silent on the specific positions where the amino acid has a functionally distinguishing role. This may be addressed by engineering features that incorporate sequence order information from the protein. However, these features can be very high-dimensional, for example, 20<sup>k</sup> -dimensional for k-mer features. The Monte Carlo based approximation algorithm for Shapley values would require exponentially more sampling (in number of features) for good approximations.

#### Impact Statement

- This paper presents a computationally efficient data lean ML pipeline. It can be used by biologists to decide whether they should invest valuable resources (skilled manpower, time, funds, etc.) for performing wet-lab experiments to determine amino acid(s) that are critical for functional differentiation of paralogous proteins. References Bachman, J. Chapter ninteen - site-directed mutagenesis. In Lorsch, J. (ed.), *Laboratory Methods in Enzymology: DNA*, volume 529 of *Methods in Enzymology*, pp. 241–248. Academic Press, 2013. doi: 10.1016/B978-0-12-418687-3.00019-7. URL [https:](https://www.sciencedirect.com/science/article/pii/B9780124186873000197) [//www.sciencedirect.com/science/](https://www.sciencedirect.com/science/article/pii/B9780124186873000197) [article/pii/B9780124186873000197](https://www.sciencedirect.com/science/article/pii/B9780124186873000197). Begum, K., Mohl, J. E., Ayivor, F., Perez, E. E., and Leung, M.-Y. GPCR-PEnDB: a database of protein sequences and derived features to facilitate prediction and classification of G protein-coupled receptors. *Database*, 2020, 11 2020. ISSN 1758-0463. doi: 10.1093/database/baaa087. URL [https://doi.org/](https://doi.org/10.1093/database/baaa087) [10.1093/database/baaa087](https://doi.org/10.1093/database/baaa087). Bileschi, M. L., Belanger, D., Bryant, D. H., Sanderson, T., Carter, B., Sculley, D., Bateman, A., DePristo,
- M. A., and Colwell, L. J. Using deep learning to annotate the protein universe. *Nature biotechnology*, 40(6):932—937, June 2022. ISSN 1087-0156. doi: 10.1038/s41587-021-01179-w. URL [https://doi.](https://doi.org/10.1038/s41587-021-01179-w) [org/10.1038/s41587-021-01179-w](https://doi.org/10.1038/s41587-021-01179-w). Brodersen, K. H., Ong, C. S., Stephan, K. E., and Buhmann,
- J. M. The balanced accuracy and its posterior distribution. In *2010 20th ICPR*, pp. 3121–3124, 2010. doi: 10.1109/ ICPR.2010.764. Cary, B. P., Zhang, X., Cao, J., Johnson, R. M., Piper, S. J., Gerrard, E. J., Wootten, D., and Sexton, P. M. New Insights into the Structure and Function of Class B1 GPCRs. *Endocrine Reviews*, 44(3):492–517, 12 2022. ISSN 0163- 769X. doi: 10.1210/endrev/bnac033. URL [https:](https://doi.org/10.1210/endrev/bnac033) [//doi.org/10.1210/endrev/bnac033](https://doi.org/10.1210/endrev/bnac033). Castro, J., Gomez, D., and Tejada, J. Polynomial ´ calculation of the shapley value based on sampling. *Computers & Operations Research*, 36(5):1726–1730, 2009. ISSN 0305-0548. doi: 10.1016/j.cor.2008.04.004. URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/S0305054808000804) [science/article/pii/S0305054808000804](https://www.sciencedirect.com/science/article/pii/S0305054808000804). Selected papers presented at the Tenth International Symposium on Locational Decisions (ISOLDE X). Catav, A., Fu, B., Zoabi, Y., Meilik, A. L. W., Shomron, N., Ernst, J., Sankararaman, S., and Gilad-Bachrach, R. Marginal contribution feature importance - an axiomatic approach for explaining data. In Meila, M. and Zhang, T. (eds.), *Proceedings of the 38th International Conference on Machine Learning*, volume 139 of *Proceedings of Machine Learning Research*, pp. 1324–1335. PMLR, 18– 24 Jul 2021. URL [https://proceedings.mlr.](https://proceedings.mlr.press/v139/catav21a.html) [press/v139/catav21a.html](https://proceedings.mlr.press/v139/catav21a.html). Covert, I., Lundberg, S. M., and Lee, S.-I. Understanding global feature contributions with additive importance measures. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), *Advances in Neural Information Processing Systems*, volume 33, pp. 17212–17223. Curran Associates, Inc., 2020. URL [https://proceedings.neurips.](https://proceedings.neurips.cc/paper_files/paper/2020/file/c7bf0b7c1a86d5eb3be2c722cf2cf746-Paper.pdf) [cc/paper\\_files/paper/2020/file/](https://proceedings.neurips.cc/paper_files/paper/2020/file/c7bf0b7c1a86d5eb3be2c722cf2cf746-Paper.pdf) [c7bf0b7c1a86d5eb3be2c722cf2cf746-Paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/c7bf0b7c1a86d5eb3be2c722cf2cf746-Paper.pdf). [pdf](https://proceedings.neurips.cc/paper_files/paper/2020/file/c7bf0b7c1a86d5eb3be2c722cf2cf746-Paper.pdf). Denisov, I. G., Grinkova, Y. V., Nandigrami, P., Shekhar, M., Tajkhorshid, E., and Sligar, S. G. Allosteric interactions in human cytochrome p450 cyp3a4: The role of phenylalanine 213. *Biochemistry*, 58(10):1411–1422, 2019. doi: 10.1021/acs.biochem.8b01268. URL [https://doi.](https://doi.org/10.1021/acs.biochem.8b01268) [org/10.1021/acs.biochem.8b01268](https://doi.org/10.1021/acs.biochem.8b01268). PMID: 30785734. Dill, K., Jernigan, R., and Bahar, I. *Protein Actions: Principles and Modeling*. CRC Press, 2017. ISBN 9781351815000. URL [https://books.google.](https://books.google.co.in/books?id=NHs2DwAAQBAJ) [co.in/books?id=NHs2DwAAQBAJ](https://books.google.co.in/books?id=NHs2DwAAQBAJ). Dodson, G. and Wlodawer, A. Catalytic triads and their relatives. *Trends in Biochemical Sciences*, 23(9):347–352, 1998. ISSN 0968-0004. doi: https://doi.org/10.1016/S0968-0004(98)01254-7. URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/S0968000498012547) [science/article/pii/S0968000498012547](https://www.sciencedirect.com/science/article/pii/S0968000498012547). Dutta, S., Akey, I. V., Dingwall, C., Hartman, K. L., Laue, T., Nolte, R. T., Head, J. F., and Akey, C. W. The crystal structure of nucleoplasmin-core: Implications for histone binding and nucleosome assembly. *Molecular Cell*, 8(4):841–853, 2001. ISSN 1097-2765. doi: 10.1016/S1097-2765(01)00354-9. URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/S1097276501003549) [science/article/pii/S1097276501003549](https://www.sciencedirect.com/science/article/pii/S1097276501003549). Edgar, R. C. and Batzoglou, S. Multiple sequence alignment. *Current Opinion in Structural Biology*, 16(3):368–373, 2006. ISSN 0959-440X. doi: 10.1016/j.sbi.2006.04.004. URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/S0959440X06000704) [science/article/pii/S0959440X06000704](https://www.sciencedirect.com/science/article/pii/S0959440X06000704). Nucleic acids/Sequences and topology.

- 495 496 497 498 499 500 504 506 508 509 511 514 515 516 518 524 526 528 531 534 536 538 540 541 542 543 544 545 546 547 548 Fu, L., Niu, B., Zhu, Z., Wu, S., and Li, W. CD-HIT: accelerated for clustering the next-generation sequencing data. *Bioinformatics*, 28(23):3150–3152, 10 2012. ISSN 1367-4803. doi: 10.1093/ bioinformatics/bts565. URL [https://doi.org/10.](https://doi.org/10.1093/bioinformatics/bts565) [1093/bioinformatics/bts565](https://doi.org/10.1093/bioinformatics/bts565). Galozzi, P., Bindoli, S., Doria, A., and Sfriso, P. The revisited role of interleukin-1 alpha and beta in autoimmune and inflammatory disorders and in comorbidities. *Autoimmunity Reviews*, 20(4):102785, 2021. ISSN 1568-9972. doi: 10.1016/j.autrev.2021.102785. URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/S1568997221000483) [science/article/pii/S1568997221000483](https://www.sciencedirect.com/science/article/pii/S1568997221000483). Hargrove, T. Y., Kim, K., de Nazare Correia Soeiro, M., da ´ Silva, C. F., da Gama Jaen Batista, D., Batista, M. M., Yazlovitskaya, E. M., Waterman, M. R., Sulikowski,
  - G. A., and Lepesheva, G. I. Cyp51 structures and structure-based development of novel, pathogen-specific inhibitory scaffolds. *International Journal for Parasitology: Drugs and Drug Resistance*, 2:178–186, 2012. ISSN 2211-3207. doi: 10.1016/j.ijpddr.2012.06.001. URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/S2211320712000206) [science/article/pii/S2211320712000206](https://www.sciencedirect.com/science/article/pii/S2211320712000206). Including Articles from Keystone Symposium on "Drug Discovery for Protozoan Parasites"; pp. 230–270. Harmar, A. Family-b g-protein-coupled receptors. *Genome biology*, 2(12):REVIEWS3013, 2001. ISSN 1474-7596. doi: 10.1186/gb-2001-2-12-reviews3013. URL [https:](https://europepmc.org/articles/PMC138994) [//europepmc.org/articles/PMC138994](https://europepmc.org/articles/PMC138994). Hedstrom, L. Serine protease mechanism and specificity. *Chemical Reviews*, 102(12):4501–4524, 2002. doi: 10.1021/cr000033x. URL [https://doi.org/10.](https://doi.org/10.1021/cr000033x) [1021/cr000033x](https://doi.org/10.1021/cr000033x). PMID: 12475199. Hedstrom, L., Perona, J. J., and Rutter, W. J. Converting trypsin to chymotrypsin: residue 172 is a substrate specificity determinant. *Biochemistry*, 33 29:8757–63, 1994. Jingami, H., Nakanishi, S., and Morikawa, K. Structure of the metabotropic glutamate receptor. *Current Opinion in Neurobiology*, 13(3):271–278, 2003. ISSN 0959-4388. doi: 10.1016/S0959-4388(03)00067-9. URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/S0959438803000679) [science/article/pii/S0959438803000679](https://www.sciencedirect.com/science/article/pii/S0959438803000679). Kan, H.-I., Chen, I.-Y., Zulfajri, M., and Wang, C. C. Subunit disassembly pathway of human hemoglobin revealing the site-specific role of its cysteine residues. *The Journal of Physical Chemistry B*, 117(34):9831–9839, 2013. doi: 10.1021/jp402292b. URL [https://doi.](https://doi.org/10.1021/jp402292b) [org/10.1021/jp402292b](https://doi.org/10.1021/jp402292b). PMID: 23902424. Kresge, N., Simoni, R. D., and Hill, R. L. The development of site-directed mutagenesis by michael smith. *Journal of Biological Chemistry*, 281(39):e31–e33, 2006. ISSN 0021-9258. doi: 10.1016/S0021-9258(19)33938-9. URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/S0021925819339389) [science/article/pii/S0021925819339389](https://www.sciencedirect.com/science/article/pii/S0021925819339389). Lau, C. K., Turner, L., Jespersen, J. S., Lowe, E. D., Petersen, B., Wang, C. W., Petersen, J. E., Lusingu, J., Theander, T. G., Lavstsen, T., and Higgins, M. K. Structural conservation despite huge sequence diversity allows epcr binding by the pfemp1 family implicated in severe childhood malaria. *Cell Host & Microbe*, 17(1):118–129, 2015. ISSN 1931-3128. doi: 10.1016/j.chom.2014.11.
    - 007. URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/S1931312814004235) [science/article/pii/S1931312814004235](https://www.sciencedirect.com/science/article/pii/S1931312814004235). Lepesheva, G. I. and Waterman, M. R. Cyp51 the omnipotent p450. *Molecular and Cellular Endocrinology*, 215(1):165–170, 2004. ISSN 0303-7207. doi: 10.1016/j.mce.2003.11.016. URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/S0303720703005148) [science/article/pii/S0303720703005148](https://www.sciencedirect.com/science/article/pii/S0303720703005148). Proceedings of the Serono Foundation for the Advancement of Medical Science Workshop on Molecular Steroidogenesis. Lepesheva, G. I. and Waterman, M. R. Sterol 14αdemethylase cytochrome p450 (cyp51), a p450 in all biological kingdoms. *Biochimica et Biophysica Acta (BBA) - General Subjects*, 1770(3):467–477, 2007. ISSN 0304-4165. doi: 10.1016/j.bbagen.2006.07.018. URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/S0304416506002145) [science/article/pii/S0304416506002145](https://www.sciencedirect.com/science/article/pii/S0304416506002145). P450. Lundberg, S. M. and Lee, S.-I. A unified approach to interpreting model predictions. In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., and Garnett, R. (eds.), *Advances in Neural Information Processing Systems*, volume 30. Curran Associates, Inc., 2017. URL [https://proceedings.neurips.](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf) [cc/paper\\_files/paper/2017/file/](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf) [8a20a8621978632d76c43dfd28b67767-Paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf). [pdf](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf). McArthur, A. G., Hegelund, T., Cox, R. L., Stegeman, J. J., Liljenberg, M., Olsson, U., Sundberg, P., and Celander, M. C. Phylogenetic Analysis of the Cytochrome P450 3 (CYP3) Gene Family. *Journal of Molecular Evolution*, 57(2):200– 211, August 2003. doi: 10.1007/s00239-003-2466-x. URL [https://link.springer.com/article/](https://link.springer.com/article/10.1007/s00239-003-2466-x) [10.1007/s00239-003-2466-x](https://link.springer.com/article/10.1007/s00239-003-2466-x). Menon, A. K., Narasimhan, H., Agarwal, S., and Chawla,
      - S. On the statistical consistency of algorithms for bi-

- 551 554 556 558 560 564 566 568 571 574 576 578 580 581 582 583 584 585 586 587 588 589 590 594 596 598 600 601 602 603 nary classification under class imbalance. In *Proceedings of the 30th ICML - Volume 28*, ICML'13, pp. III–603–III–611. JMLR.org, 2013. Muhlethaler, T., Gioia, D., Prota, A. E., Sharpe, ¨
  - M. E., Cavalli, A., and Steinmetz, M. O. Comprehensive analysis of binding sites in tubulin. *Angewandte Chemie International Edition*, 60(24): 13331–13342, 2021. doi: 10.1002/anie.202100273. URL [https://onlinelibrary.wiley.com/doi/](https://onlinelibrary.wiley.com/doi/abs/10.1002/anie.202100273) [abs/10.1002/anie.202100273](https://onlinelibrary.wiley.com/doi/abs/10.1002/anie.202100273). Narahari, Y. *Game Theory and Mechanism Design*. WORLD SCIENTIFIC / INDIAN INST OF SCIENCE, INDIA, 2014. doi: 10.1142/ 8902. URL [https://www.worldscientific.](https://www.worldscientific.com/doi/abs/10.1142/8902) [com/doi/abs/10.1142/8902](https://www.worldscientific.com/doi/abs/10.1142/8902). Nitahara, Y., Kishimoto, K., Yabusaki, Y., Gotoh, O., Yoshida, Y., Horiuchi, T., and Aoyama, Y. The amino acid residues affecting the activity and azole susceptibility of rat cyp51 (sterol 14-demethylase p450). *The Journal of Biochemistry*, 129(5):761–768, 2001. Okada, T., Ernst, O. P., Palczewski, K., and Hofmann,
  - K. P. Activation of rhodopsin: new insights from structural and biochemical studies. *Trends in Biochemical Sciences*, 26(5):318–324, 2001. ISSN 0968-0004. doi: 10.1016/S0968-0004(01)01799-6. URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/S0968000401017996) [science/article/pii/S0968000401017996](https://www.sciencedirect.com/science/article/pii/S0968000401017996). Permyakov, E. A. α-actalbumin, Amazing Calcium-Binding Protein. *Biomolecules*, 10(9):1210, Aug 2020. ISSN 2218-273X. doi: 10.3390/biom10091210. URL [http:](http://dx.doi.org/10.3390/biom10091210) [//dx.doi.org/10.3390/biom10091210](http://dx.doi.org/10.3390/biom10091210). Permyakov, E. A. and Berliner, L. J. α-Lactalbumin: structure and function. *FEBS Letters*, 473(3):269–274, 2000. ISSN 0014-5793. doi: 10.1016/S0014-5793(00)01546-5. URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/S0014579300015465) [science/article/pii/S0014579300015465](https://www.sciencedirect.com/science/article/pii/S0014579300015465). Pettersen, E. F., Goddard, T. D., Huang, C. C., Meng,
- E. C., Couch, G. S., Croll, T. I., Morris, J. H., and Ferrin, T. E. Ucsf chimerax: Structure visualization for researchers, educators, and developers. *Protein Science*, 30(1):70–82, 2021. doi: 10.1002/pro. 3943. URL [https://onlinelibrary.wiley.](https://onlinelibrary.wiley.com/doi/abs/10.1002/pro.3943) [com/doi/abs/10.1002/pro.3943](https://onlinelibrary.wiley.com/doi/abs/10.1002/pro.3943). Qasba, P. K., Kumar, S., and Brew, D. K. Molecular divergence of lysozymes and α-lactalbumin. *Critical Reviews in Biochemistry and Molecular Biology*, 32(4):255–306, 1997. doi: 10.3109/10409239709082574. URL [https:](https://doi.org/10.3109/10409239709082574) [//doi.org/10.3109/10409239709082574](https://doi.org/10.3109/10409239709082574). Qiu, H., Taudien, S., Herlyn, H., Schmitz, J., Zhou, Y., Chen, G., Roberto, R., Rocchi, M., Platzer, M., and Wojnowski, L. Cyp3 phylogenomics: evidence for positive selection of cyp3a4 and cyp3a7. *Pharmacogenetics and Genomics*, 18(1):53—66, January 2008. ISSN 1744-6872. doi: 10.1097/fpc.0b013e3282f313f8. URL [https://](https://doi.org/10.1097/FPC.0b013e3282f313f8) [doi.org/10.1097/FPC.0b013e3282f313f8](https://doi.org/10.1097/FPC.0b013e3282f313f8). Rozemberczki, B., Watson, L., Bayer, P., Yang, H.-T., Kiss, O., Nilsson, S., and Sarkar, R. The shapley value in machine learning. In Raedt, L. D. (ed.), *Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI-22*, pp. 5572–5579. International Joint Conferences on Artificial Intelligence Organization, 7 2022. doi: 10.24963/ijcai.2022/778. URL [https:](https://doi.org/10.24963/ijcai.2022/778) [//doi.org/10.24963/ijcai.2022/778](https://doi.org/10.24963/ijcai.2022/778). Survey Track. Sakmar, T. P., Menon, S. T., Marin, E. P., and Awad,
  - E. S. Rhodopsin: Insights from recent structural studies. *Annual Review of Biophysics*, 31(Volume 31, 2002):443–484, 2002. ISSN 1936-1238. doi: 10.1146/annurev.biophys.31.082901.134348. URL [https://www.annualreviews.org/](https://www.annualreviews.org/content/journals/10.1146/annurev.biophys.31.082901.134348) [content/journals/10.1146/annurev.](https://www.annualreviews.org/content/journals/10.1146/annurev.biophys.31.082901.134348) [biophys.31.082901.134348](https://www.annualreviews.org/content/journals/10.1146/annurev.biophys.31.082901.134348). Sanderson, T., Bileschi, M. L., Belanger, D., and Colwell, L. J. Proteinfer, deep neural networks for protein functional inference. *eLife*, 12:e80942, feb 2023. ISSN 2050-084X. doi: 10.7554/eLife.80942. URL <https://doi.org/10.7554/eLife.80942>. Shapley, L. S. *17. A Value for n-Person Games*, pp. 307–318. Princeton University Press, Princeton, 1953. ISBN 9781400881970. doi: doi:10.1515/ 9781400881970-018. URL [https://doi.org/10.](https://doi.org/10.1515/9781400881970-018) [1515/9781400881970-018](https://doi.org/10.1515/9781400881970-018). Shionyu, M., Takahashi, K., and Go, M. Variable subunit ¯ contact and cooperativity of hemoglobins. *J. Mol. Evol.*, 53(4-5):416–429, October 2001. Song, H., Bremer, B. J., Hinds, E. C., Raskutti, G., and Romero, P. A. Inferring protein sequencefunction relationships with large-scale positive-unlabeled learning. *Cell Systems*, 12(1):92–101.e8, 2021. ISSN 2405-4712. doi: 10.1016/j.cels.2020.10.
  - 007. URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/S2405471220304142) [science/article/pii/S2405471220304142](https://www.sciencedirect.com/science/article/pii/S2405471220304142). Steinwart, I. and Christmann, A. *Support Vector Machines*. Springer Publishing Company, Incorporated, 1st edition, 2008. ISBN 0387772413. Strushkevich, N., Usanov, S. A., and Park, H.-W. Structural basis of human cyp51 inhibition by antifungal

- 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 azoles. *Journal of Molecular Biology*, 397(4):1067–1078, 2010. ISSN 0022-2836. doi: 10.1016/j.jmb.2010.01.
  - 075. URL [https://www.sciencedirect.com/](https://www.sciencedirect.com/science/article/pii/S0022283610001324) [science/article/pii/S0022283610001324](https://www.sciencedirect.com/science/article/pii/S0022283610001324). The UniProt Consortium. UniProt: the universal protein knowledgebase in 2021. *Nucleic Acids Research*, 49(D1): D480–D489, 11 2020. ISSN 0305-1048. doi: 10.1093/ nar/gkaa1100. URL [https://doi.org/10.1093/](https://doi.org/10.1093/nar/gkaa1100) [nar/gkaa1100](https://doi.org/10.1093/nar/gkaa1100). Tripathi, S., Hemachandra, N., and Trivedi, P. Interpretable feature subset selection: A Shapley value based approach. In *2020 IEEE BigData*, pp. 5463–5472, 2020. doi: 10. 1109/BigData50022.2020.9378102. Tripathi, S., Hemachandra, N., and Trivedi, P. Interpretable feature subset selection: A shapley value based approach, 2021. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2001.03956) [2001.03956](https://arxiv.org/abs/2001.03956). Vass, M., Podlewska, S., de Esch, I. J. P., Bojarski,
  - A. J., Leurs, R., Kooistra, A. J., and de Graaf, C. Aminergic gpcr–ligand interactions: A chemical and structural map of receptor mutation data. *Journal of Medicinal Chemistry*, 62(8):3784–3839, 2019. doi: 10.1021/acs.jmedchem.8b00836. URL [https://doi.](https://doi.org/10.1021/acs.jmedchem.8b00836) [org/10.1021/acs.jmedchem.8b00836](https://doi.org/10.1021/acs.jmedchem.8b00836). PMID: 30351004. Veerapandian, B., Gilliland, G. L., Raag, R., Svensson, A. L., Masui, Y., Hirai, Y., and Poulos,
  - T. L. Functional implications of interleukin-1β based on the three-dimensional structure. *Proteins: Structure, Function, and Bioinformatics*, 12(1): 10–23, 1992. doi: 10.1002/prot.340120103. URL [https://onlinelibrary.wiley.com/doi/](https://onlinelibrary.wiley.com/doi/abs/10.1002/prot.340120103) [abs/10.1002/prot.340120103](https://onlinelibrary.wiley.com/doi/abs/10.1002/prot.340120103). Yang, Y., Xu, T., Conant, G., Kishino, H., Thorne, J. L., and Ji, X. Interlocus gene conversion, natural selection, and paralog homogenization. *Molecular Biology and Evolution*, 40(9):msad198, 09 2023. ISSN 1537-1719. doi: 10.1093/molbev/msad198. URL [https://doi.](https://doi.org/10.1093/molbev/msad198) [org/10.1093/molbev/msad198](https://doi.org/10.1093/molbev/msad198). Young, H. P. Monotonic solutions of cooperative games. *Int. J. Game Theory*, 14(2):65–72, jun 1985. ISSN 0020- 7276. doi: 10.1007/BF01769885. URL [https://doi.](https://doi.org/10.1007/BF01769885) [org/10.1007/BF01769885](https://doi.org/10.1007/BF01769885). Zhang, Y., Wang, Z., Wang, Y., Jin, W., Zhang, Z., Jin, L., Qian, J., and Zheng, L. Cyp3a4 and cyp3a5: the crucial roles in clinical drug metabolism and the significant implications of genetic polymorphisms. *PeerJ*, 12:e18636, 2024. Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., and Torralba, A. Learning Deep Features for Discriminative Localization . In *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2921–2929, Los Alamitos, CA, USA, June 2016. IEEE Computer Society. doi: 10.1109/CVPR.2016.319. URL [https://doi.ieeecomputersociety.org/](https://doi.ieeecomputersociety.org/10.1109/CVPR.2016.319) [10.1109/CVPR.2016.319](https://doi.ieeecomputersociety.org/10.1109/CVPR.2016.319).

689 690

694

696

698

700

704

706

708 709

711

714

# A. Data collection and code

We discuss the details of the data collection procedure for the datasets used in our computational experiments.

## A.1. Datasets of 15 paralog pairs

We apply our method for identifying amino acid types that distinguish paralogous proteins using the datasets described in Table [A2.](#page-12-1) Only the train set is used for computing AF S, while the test set is used for computing classification scores for the linear SVM trained using the train set.

Table A2: The number of sequences in the train and test sets of the protein families considered in computational experiments.

| Family Lysozyme-like α -Lactalbumin Lysozyme C Trypsin-like | Train (Swiss-Prot) 22 74 | Test (TrEMBL) 53 14 |
|-------------------------------------------------------------|--------------------------|---------------------|
| Trypsin                                                     | 66                       | 3813                |
| Chymotrypsin Tubulin                                        | 17                       | 281                 |
| α                                                           | 117                      | 190                 |
| β Histone                                                   | 191                      | 347                 |
| H2A                                                         | 180                      | 16599               |
| H2B Interleukin-1                                           | 177                      | 7599                |
| α                                                           | 16                       | 12                  |
| β                                                           | 25                       | 194                 |
| Cytochrome P450                                             |                          |                     |
| CYP3                                                        | 32                       | 818                 |
| CYP51 Globins                                               | 32                       | 601                 |
| Myoglobin                                                   | 107                      | 479                 |
| Hemoglobin- α                                               | 303                      | 525                 |
| Hemoglobin- β                                               | 285                      | 261                 |
|                                                             | Train (80%)              | Test (20%)          |
| GPCR families                                               |                          |                     |
| Rhodopsin-like                                              | 181                      | 45                  |
| ë Lipid receptors                                           | 113                      | 28                  |
| Peptide receptors                                           | 367                      | 92                  |
| Aminergic receptors                                         | 186                      | 47                  |
| Glutamate-like                                              | 89                       | 23                  |
| Secretin-like                                               | 90                       | 23                  |

All datasets are taken from publicly available databases (UniProt [\(The UniProt Consortium,](#page-11-0) [2020\)](#page-11-0) and GPCR-PEnDB [\(Begum et al.,](#page-8-12) [2020\)](#page-8-12)). Well-known pairs of paralogous proteins were curated from millions of sequences from UniProt considering the number of sequences and manually reviewed labels available for them.

For all datasets except GPCR, we use manually curated Swiss-Prot sequences for training and electronically annotated TrEMBL sequences for testing. These proteins have very specific functions. In contrast, GPCRs are a large and diverse group of transmembrane proteins that mediate cellular responses to extracellular signals. We chose to use an already curated dataset in this case. For each of the GPCR families considered (Table [A2\)](#page-12-1), the sequences are randomly split as 80%-train/20%-test. The use of GPCR-PEnDB data is to illustrate the effectiveness of our method with random slicing, which is inevitable when additional curated data are not available. If one or many UniProt entries in a dataset had identical sequences, then only one of them was retained, and the remaining were deleted.

- 716 718 724 726 728 731 734 736 738 740 741 742 743 744 745 746 747 748 749 751 754
- lysozyme C: (protein\_name:"lysozyme C") AND (fragment:false) NOT (existence:4) NOT (existence:5) AND (length:[\* TO 200]) AND (ec:3.2.1.17) AND (xref:cazy-GH22) AND (reviewed:true)
- α-lactalbumin: (protein\_name:"alpha lactalbumin") AND (fragment:false) NOT (existence:4) NOT (existence:5) AND (length:[\* TO 200]) AND (reviewed:true)
- myoglobin: (protein\_name:"myoglobin") AND (xref:interpro-IPR002335) AND (fragment:false) NOT (existence:5) NOT (existence:4)
- hemoglobin-α: (protein\_name:"hemoglobin alpha") AND (xref:interpro-IPR002338) AND (fragment:false) NOT (existence:5) NOT (existence:4)
- hemoglobin-β: (protein\_name:"hemoglobin beta") AND (xref:interpro-IPR002337) AND (fragment:false) NOT (existence:5) NOT (existence:4)
- trypsin: (protein\_name:trypsin) AND (fragment:false) AND (ec:3.4.21.4) NOT (existence:5)
- chymotrypsin: (protein\_name:chymotrypsin) AND (fragment:false) AND (ec:3.4.21.1) NOT (existence:5)
- tubulin-α: (protein\_name:"tubulin alpha") AND (family:"tubulin family") AND (length:[300 TO 600]) AND (fragment:false) NOT (annotation\_score:1) NOT (annotation\_score:2)
- tubulin-β: (protein\_name:"tubulin beta") AND (family:"tubulin family") AND (length:[300 TO 600]) AND (fragment:false) NOT (annotation\_score:1) NOT (annotation\_score:2)
- interleukin-1 α (protein\_name:"interleukin-1 alpha") AND (family:il-1) AND (fragment:false) NOT (existence:4) NOT (existence:5) AND (length:[200 TO 400]) NOT (annotation\_score:1)
- interleukin-1 β: (protein\_name:"interleukin-1 beta") AND (family:il-1) AND (fragment:false) NOT (existence:4) NOT (existence:5) AND (length:[200 TO 400]) NOT (annotation\_score:1)
- Histone H2A: (protein\_name:"histone h2a") AND (family:histone) AND (fragment:false) NOT (existence:4) NOT (existence:5) AND (length:[\* TO 200])
- Histone H2B: (protein\_name:"histone h2b") AND (family:histone) AND (fragment:false) NOT (existence:4) NOT (existence:5) AND (length:[\* TO 200])
- Cytochrome P450 CYP3: (family:"Cytochrome P450") AND ((gene:cyp3) OR (gene:cyp3A\*)) AND (fragment:false) NOT (existence:4) NOT (existence:5) NOT (annotation\_score:1)
- Cytochrome P450 CYP51: (family:"Cytochrome P450") AND ((gene:cyp51) OR (gene:cyp51A\*) OR (gene:cyp51B\*) OR (gene:cyp51C\*)) AND (fragment:false) NOT (existence:4) NOT (existence:5) NOT (annotation\_score:1)

756 758 The GPCR sequences were collected from the GPCR-PEn database (URL: <https://gpcr.utep.edu/>) [\(Begum et al.,](#page-8-12) [2020\)](#page-8-12). Sequence redundancy of the rhodopsin-like family was reduced using CD-hit [\(Fu et al.,](#page-9-10) [2012\)](#page-9-10) with 30% sequence similarity cutoff.

760

764

766

The following queries were used for collecting data from UniProt [\(The UniProt Consortium,](#page-11-0) [2020\)](#page-11-0),

#### A.2. Code

The code to reproduce the computational experiments is available at [https://anonymous.4open.science/r/](https://anonymous.4open.science/r/AFS_AAC_SVM-F3D9) [AFS\\_AAC\\_SVM-F3D9](https://anonymous.4open.science/r/AFS_AAC_SVM-F3D9). Protein sequences used in the computational experiments along with their UniProt IDs, are provided in the datasets folder as .csv files for each family.

774

776

778

794

796

800

804

806

808

# B. Sequence and function diversity of protein classes within a dataset

Paralogous proteins have a common ancestor but have diverged in functionality. Protein functions are an aggregate of descriptors describing protein's activity and influence at various levels. They can be at the molecular level, like binding with specific molecules and catalysing reactions, to the biological process level, like energy metabolism. In [B.1,](#page-14-1) we discuss the diversity of the functions of the proteins considered in our datasets.

As paralogs have a common ancestor, high sequence similarity would suggest high evolutionary conservation in the proteins. In [B.2,](#page-14-2) we discuss the extent of sequence diversity in protein classes considered in our datasets.

We see that the dataset of proteins considered in our computational experiments are diverse in their function and sequences.

#### B.1. Function diversity

We have considered paralogous proteins with varying functional differences. We find very subtle differences in the functions of trypsin and chymotrypsin. On the other hand, the function difference is drastic in the case of alpha-lactalbumin and lysozyme c.

Trypsin and chymotrypsin are a family of enzymes that break peptide bonds in proteins. The difference in the function of these proteins is fine-grained; trypsins cleave only the peptide bond following a basic amino acid (K and R), while chymotrypsins cleave the peptide bond following a hydrophobic amino acid (F, W, and Y ) [\(Dodson & Wlodawer,](#page-8-6) [1998\)](#page-8-6).

GPCRs constitute a large and diverse class of cell surface receptor proteins. They trigger intra-cellular pathways in response to external signals. These signals are in the form of small molecules, called ligands. Depending upon the nature of ligands and other 3D structural similarities, GPCRs are grouped into distinct classes. We consider three such classes viz., rhodopsin-like, secretin-like, and glutamate-like. Further, we consider pairwise three subfamilies of rhodopsin-like GPCRs viz., aminergic receptors, lipid receptors, and peptide receptors.

Lysozyme C and α-lactalbumin are sequence and structure homologs with mutually exclusive functions and high fold conservation. Based on phylogenetic analysis, they are considered to have diverged from a common ancestor millions of years ago [\(Qasba et al.,](#page-10-15) [1997\)](#page-10-15).

Globins are a superfamily of functionally divergent homologous protein families with a high level of fold conservation. We consider three well-known globin families viz., myoglobin, hemoglobin-α and hemoglobin-β. Myoglobin is a monomer that binds and releases oxygen as per physiological requirements. On the other hand, α and β chains together constitute hemoglobin, a tetramer of composition α2β<sup>2</sup> [\(Dill et al.,](#page-8-9) [2017\)](#page-8-9), that transports oxygen in red blood cells.

Tubulin-α and tubulin-β are similar to the hemoglobin-α and hemoglobin-β pair in that they both share sequence and 3D structural similarities but have subtle functional differences. One copy each of tubulin-α and tubulin-β form a functional dimer. Notably, neither two copies of tubulin-α nor two copies of tubulin-β can form a functional dimer. Tubulin-β has a catalytic activity (GTP hydrolysis) that is absent in tubulin-α. This is one of the several subtle functional differences between tubulin-α and tubulin-β.

Interleukin-1 alpha and interleukin-1 beta are both proteins involved in the immune system. They differ from each other in their occurrence within the body (on cell surface or in blood circulation), activation mechanisms, and associated signalling pathways [\(Galozzi et al.,](#page-9-11) [2021\)](#page-9-11).

Cytochrome P450 (abbreviated as CYP) is a family of proteins whose function is clearance of 'foreign' molecules (drugs; also called as xenobiotics) as well as in certain biosynthesis pathways e.g., of steroid hormones. CYP3 and CYP51 are two of the several classes of CYPs; CYP3 metabolizes lipophilic molecules [\(McArthur et al.,](#page-9-12) [2003\)](#page-9-12) whereas CYP51 is involved in steroid biosynthesis [\(Hargrove et al.,](#page-9-13) [2012\)](#page-9-13).

Hemoglobin-α/hemoglobin-β, histone H2A / histone H2B and tubulin-α/tubulin-β are paralog pairs that together function as heteromers (protein complexes made up of different protein subunits).

## B.2. Sequence Diversity

The dataset of the 15 paralog pairs in our experiments comprises 21 protein families (Table [A2\)](#page-12-1). For these families, we compute the within-class sequence similarities (for sequences within a protein family). We also compute the inter-class sequence similarities (between sequences from two different protein families) for each paralog pair. These are shown in

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

Appendix Figure [B4.](#page-17-0) We use a longest subsequence based similarity score, lcss, that is defined in [B.2.1.](#page-15-0) In [B.2.2,](#page-15-1) we see that lcss significantly varies across the 21 protein families we are considering as compared to its variation between the two protein sequences of any paralog pair.

## B.2.1. LONGEST COMMON SUBSEQUENCE BASED SIMILARITY SCORE (lcss)

We compute the longest common subsequence (lcs) based similarity score (lcss) between a pair of protein sequences. We define lcss between two sequences as the length of their longest common subsequence, lcs, divided by the length of the longest sequence from the two. For a pair of protein sequences, p <sup>p</sup>i<sup>q</sup> " pp piq , p piq 2 , . . . , p piq L<sup>1</sup> q of length L<sup>1</sup> and p <sup>p</sup>j<sup>q</sup> " pp pjq 1 , p pjq 2 , . . . , p pjq L<sup>2</sup> q of length L2, their lcss is,

$$\begin{aligned} lcs(\mathbf{p}^{(i)}, \mathbf{p}^{(j)}) &= \max_{\mathbf{q}} k \\ \text{s.t. } \mathbf{q} &= (q_1, q_2, \dots, q_k) \\ (q_1 = p_{x_1}^{(i)} = p_{y_1}^{(j)}, q_2 = p_{x_2}^{(i)} = p_{y_2}^{(j)}, \dots, q_k = p_{x_k}^{(i)} = p_{y_k}^{(j)}) \\ x_1 < x_2 < \dots < x_k \\ y_1 < y_2 < \dots < y_k \end{aligned}$$

lcs based similarity score, lcss, is defined as,

$$lcss(\mathbf{p}^{(i)}, \mathbf{p}^{(j)}) = \frac{lcs(\mathbf{p}^{(i)}, \mathbf{p}^{(j)})}{\max(L_1, L_2)} \in [0, 1]$$

lcsspp piq , p pjq q " 1 if and only if p <sup>p</sup>i<sup>q</sup> " p pjq , i.e., sequences are identical. Whereas lcsspp piq , p pjq q " 0 if and only if p piq <sup>x</sup> ‰ p pjq <sup>y</sup> , @x, y, i.e., there are no amino acids common to both the sequences.

#### B.2.2. WITHIN-CLASS AND INTER-CLASS lcss FOR THE 15 PARALOG PAIRS

Within-class lcss: lcsspp piq , p pjq q are computed with p piq , p pjq from the same protein family. These are shown in blue and magenta in Figure [B4](#page-17-0) (with box-plots) for each of 21 protein families in the 15 paralog pairs.

- 12 of 21 protein families have median within-class lcss greater than 0.5. This implies less sequence diversity in this set of families from the remaining families. These are,

| Family             | α-lactaluminon | lysosymac C     | myoglobin       | hemoglobin-α | hemoglobin-β | tubulin-α            |
|--------------------|----------------|-----------------|-----------------|--------------|--------------|----------------------|
| Median <i>lcss</i> | 0.6            | 0.59            | 0.81            | 0.63         | 0.67         | 0.83                 |
| Family             | tubulin-β      | interleukin-1 α | interleukin-1 β | histone H2A  | histone H2B  | cytochrome P450 CYP3 |
| Median <i>lcss</i> | 0.82           | 0.72            | 0.66            | 0.65         | 0.68         | 0.7                  |

Table B3: The median within-class lcss between sequences from the respective families. See boxplot in Figure [B4.](#page-17-0)

- Median lcss ě 0.6 for 11 of these 12 families and ě 0.8 for 3 families (high level of sequence conservation).
- For 7 out of the 15 paralog pairs, the median within-class lcss ą 0.5 for both families of a paralogous pair.
- For the remaining 9 protein families, the median within-class lcss is less than 0.5. This implies high sequence diversity in this set of families from the remaining families. These are,

| Family trypsin   | chymotrypsin       | rhodopsin-like | receptor glutamate-like receptor secretin-like receptor |
|------------------|--------------------|----------------|---------------------------------------------------------|
| Median lcss 0.47 | 0.45               | 0.34           | 0.35 0.36                                               |
| Family           | aminergic receptor | lipid receptor | peptide receptor cytochrome P450 CYP51                  |
| Median lcss      | 0.39               | 0.37           | 0.37 0.47                                               |

Table B4: The median within-class lcss between sequences from the respective families. See boxplot in Figure [B4.](#page-17-0)

882 883 884 Inter-class lcss: lcsspp piq , p pjq q are computed with p piq , p pjq respectively from two protein families that are paralog pairs. These are shown in cyan in Figure [B4](#page-17-0) (with box-plots) for each of the 15 paralog pairs.

885

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

- For the paralog pair Cytochrome P450 CYP3 vs CYP51, the median sequence similarity for CYP3 is greater than 0.5, while for CYP51, it is less than 0.5.

- The median inter-class lcss is less than 0.5 for all paralog pairs. This implies sequences of the proteins across the classes are not very similar.

Distinguishing paralog pairs based on within-class and inter-class lcss: If we analyse the box plots in Figure [B4](#page-17-0) - two paralog pair proteins can be considered to be distinguishable based on sequence similarity if the upper-whisker of inter-class lcss is lower than the lower-whiskers of the respective within-class lcss scores.

- Apart from paralog pairs, tubulin-α vs tubulin-β (Figure [B4c\)](#page-17-0) and interleukin-1 α vs interleukin-1 β (Figure [B4d\)](#page-17-0), no other paralog pair is distinguishable based on sequence similarity.
- For Trypsin vs Chymotrypsin and the 6 GPCR pairs (Figures [B4b](#page-17-0) and [B4j](#page-17-0) to [B4o\)](#page-17-0), the median inter-class lcss scores are close to the within-class lcss scores making them indistinguishable based on sequence similarity.

![](_page_17_Figure_1.jpeg)

Figure B4: lcss sequence similarity scores for the 15 paralog pair datasets. In the boxplots, the lower and upper whiskers are at 1.5 IQR (inter-quantile range) values away from the first and third quartiles respectively.

994 996 998 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 Algorithm 1 ϕ<sup>i</sup> Monte-carlo approximation algorithm as suggested in [\(Tripathi et al.,](#page-11-2) [2020;](#page-11-2) [2021\)](#page-11-3) Input: Feature set N " t1, 2, . . . , 20u, Number of sample permutations samP erm, Datasets pD<sup>P</sup> , DQq, Set of coalitions Sam co set " rpqs Initialise: vppqq " 0, ϕˆ i :" 0@i P N Append N to Sam co set. for s " 1, 2, . . . , samP erm do Take π P P ermSetpNq with probability <sup>1</sup> 20! . for i " 1, 2, . . . , 20 do Compute P red<sup>i</sup> pπq " tπp1q, πp2q, . . . πpk ´ 1q|i " πpkqu if P red<sup>i</sup> pπq not in Sam co set then Compute vpP red<sup>i</sup> pπqq " 1 ´ tr erpP red<sup>i</sup> pπqq. Append P red<sup>i</sup> pπq to Sam co set. end if if P red<sup>i</sup> pπq Y i not in Sam co set then Compute vpP red<sup>i</sup> pπq Y tiuq " 1 ´ tr erpP red<sup>i</sup> pπq Y tiuq. Append P red<sup>i</sup> pπq Y tiu to Sam co set. end if ϕˆ <sup>i</sup> " ϕˆ <sup>i</sup> ` vpP red<sup>i</sup> pπq Y tiuq ´ vpP red<sup>i</sup> pπqqq end for end for ϕˆ i " ϕˆi samP erm , @i P N

1014

1016

1019 1024 1026 We provide details for the linear SVM classifier discussed in Section [2.3.](#page-2-0) We use 5-fold cross-validation to tune the SVM regularisation hyperparameter C from t0.1, 1, 10, 100, 1000u that gives the best average classification score for the 5 folds. C is inversely proportional to the strength of regularisation. In general, we find that there is an imbalance in the number of sequences that we find for the two paralogous proteins, i.e. say n<sup>P</sup> ąą nQ. It is known that accuracy is not a well-suited performance measure of the classifier in class imbalance settings. Therefore, we use the arithmetic mean of sensitivity and specificity (AM) to measure the performance of the classifier [\(Brodersen et al.,](#page-8-5) [2010\)](#page-8-5). Further, we use a class-balanced version of hinge loss for training the SVM as suggested in [\(Menon et al.,](#page-9-14) [2013\)](#page-9-14) for statistical consistency with the AM score. Appendix Table [E5](#page-20-0) reports the train and test scores of the trained linear SVM with AAC and AF S features, respectively, on the protein family datasets (See Appendix Table [A2\)](#page-12-1) considered in our computational experiments.

1029

1034

1036

## C. The SVEA algorithm for AFS

# D. SVM training for AF S partition

![](_page_19_Figure_4.jpeg)

# E. More details for computational experiments

![](_page_19_Figure_2.jpeg)

Figure E5: The sizes of the AF S for the 15 datasets.

Figure E6: Shapley value (ϕpiq) for AAC features computed using SVEA.

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

1151

Table E5: Classification scores for different pairs of paralogous proteins using the train/test datasets described in Table [A2,](#page-12-1) using AAC and AF S features. The AF S amino acids computed for each pair are given in Table [1.](#page-4-0) The train score is the mean (˘1 standard deviation) 5-fold cross-validation score. AM is the arithmetic mean of specificity and sensitivity. Acc is the accuracy.

| <span></span>                        |       | <span></span>       |  |
|--------------------------------------|-------|---------------------|--|
| (a) Lyssozyme Cov et al-Lactabulomin |       |                     |  |
|                                      | AAC   | AFS                 |  |
| Train AM                             | 1.0   | 0.93 ( $\pm 0.01$ ) |  |
| Test AM                              | 0.896 | 0.898               |  |
| Train Acc                            | 1.0   | 0.99 ( $\pm 0.02$ ) |  |
| Test Acc                             | 0.836 | 0.881               |  |

|       | (b) | Trypsin vs       | chymotrypsin     |
|-------|-----|------------------|------------------|
|       |     | AAC              | AFS              |
| Train | AM  | 0.992 ( ˘ 0.015) | 0.977 ( ˘ 0.031) |
| Test  | AM  | 0.873            | 0.835            |
| Train | Acc | 0.988 ( ˘ 0.024) | 0.965 ( ˘ 0.047) |
| Test  | Acc | 0.844            | 0.756            |

| (c)       | (c) Tubulin-vis-vis Tubulin-vis | (c) ACC        | (c) AFS |
|-----------|---------------------------------|----------------|---------|
|           |                                 |                |         |
| Train AM  | 0.996 (±0.009)                  | 0.997 (±0.006) |         |
| Test AM   | 0.902                           | 0.992          |         |
| Train Acc | 0.907 (±0.006)                  | 0.904 (±0.008) |         |
| Test Acc  | 0.991                           | 0.994          |         |

|           | AAC            | AFS           |
|-----------|----------------|---------------|
| Train AM  | 0.983 (±0.016) | 0.983 (±0.01) |
| Test AM   | 0.91           | 0.94          |
| Train Acc | 0.983 (±0.016) | 0.983 (±0.01) |
| Test Acc  | 0.889          | 0.922         |

|               | (e) Globins |            |            |
|---------------|-------------|------------|------------|
| Dataset       |             | AAC        | AFS        |
| Myoglobin vs  |             |            |            |
| Hemoglobin- α |             |            |            |
| Train         | AM          | 0.998      | 0.994      |
|               |             | ( ˘ 0.003) | ( ˘ 0.009) |
| Test          | AM          | 0.968      | 0.97       |
| Train         | Acc         | 0.998      | 0.995      |
|               |             | ( ˘ 0.005) | ( ˘ 0.006) |
| Test          | Acc         | 0.969      | 0.971      |
| Myoglobin vs  |             |            |            |
| Hemoglobin- β |             |            |            |
| Train         | AM          | 1.0        | 1.0        |
|               |             | ( ˘ 0.0)   | ( ˘ 0.0)   |
| Test          | AM          | 0.957      | 0.936      |
| Train         | Acc         | 1.0        | 1.0        |
|               |             | ( ˘ 0.0)   | ( ˘ 0.0)   |
| Test          | Acc         | 0.949      | 0.919      |
| Hemoglobin- α |             |            |            |
| Hemoglobin- β |             |            |            |
| Train         | AM          | 0.983      | 0.976      |
|               |             | ( ˘ 0.008) | ( ˘ 0.007) |
| Test          | AM          | 0.961      | 0.935      |
| Train         | Acc         | 0.983      | 0.976      |
|               |             | ( ˘ 0.008) | ( ˘ 0.006) |
| Test          | Acc         | 0.966      | 0.947      |

|               | (f) | GPCRs      |            |
|---------------|-----|------------|------------|
| Dataset       |     | AAC        | AFS        |
| vs Glutamate |     |            |            |
| Train         | AM  | 0.933      | 0.95       |
|               |     | ( ˘ 0.042) | ( ˘ 0.032) |
| Test          | AM  | 0.888      | 0.845      |
| Train         | Acc | 0.933      | 0.95       |
|               |     | ( ˘ 0.042) | ( ˘ 0.032) |
| Test          | Acc | 0.889      | 0.844      |
| like vs       |     |            |            |
| Train         | AM  | 0.884      | 0.85       |
|               |     | ( ˘ 0.042) | ( ˘ 0.045) |
| Test          | AM  | 0.967      | 0.934      |
| Train         | Acc | 0.867      | 0.837      |
|               |     | ( ˘ 0.038) | ( ˘ 0.032) |
| Test          | Acc | 0.956      | 0.926      |
| like vs       |     |            |            |
| Train         | AM  | 0.917      | 0.878      |
|               |     | ( ˘ 0.051) | ( ˘ 0.065) |
| Test          | AM  | 0.934      | 0.846      |
| Train         | Acc | 0.908      | 0.863      |
|               |     | ( ˘ 0.06)  | ( ˘ 0.073) |
| Test          | Acc | 0.941      | 0.853      |
| Aminergic vs  |     |            |            |
| Train         | AM  | 0.949      | 0.943      |
|               |     | ( ˘ 0.014) | ( ˘ 0.005) |
| Test          | AM  | 0.922      | 0.843      |
| Train         | Acc | 0.943      | 0.94       |
|               |     | ( ˘ 0.017) | ( ˘ 0.008) |
| Test          | Acc | 0.92       | 0.84       |
| Aminergic vs  |     |            |            |
| Train         | AM  | 0.835      | 0.818      |
|               |     | ( ˘ 0.06)  | ( ˘ 0.053) |
| Test          | AM  | 0.844      | 0.79       |
| Train         | Acc | 0.83       | 0.819      |
|               |     | ( ˘ 0.06)  | ( ˘ 0.051) |
| Test          | Acc | 0.827      | 0.784      |
| Lipid vs      |     |            |            |
| Train         | AM  | 0.829      | 0.76       |
|               |     | ( ˘ 0.022) | ( ˘ 0.035) |
| Test          | AM  | 0.845      | 0.709      |
| Train         | Acc | 0.838      | 0.75       |
|               |     | ( ˘ 0.018) | ( ˘ 0.032) |
| Test          | Acc | 0.858      | 0.725      |

|           | AAC           | AFS           |
|-----------|---------------|---------------|
| Train AM  | 0.98 (±0.04)  | 0.98 (±0.04)  |
| Test AM   | 0.97          | 0.98          |
| Train Acc | 0.975 (±0.05) | 0.975 (±0.05) |
| Test Acc  | 0.961         | 0.971         |

|                  | <b>AAC</b>     | <b>AFS</b>     |
|------------------|----------------|----------------|
| <b>Train AM</b>  | 0.967 (±0.041) | 0.933 (±0.062) |
| <b>Test AM</b>   | 0.902          | 0.92           |
| <b>Train Acc</b> | 0.969 (±0.038) | 0.936 (±0.062) |
| <b>Test Acc</b>  | 0.894          | 0.908          |

 The 3D structures of hemoglobin-α/β (PDB ID:1HHO) were aligned with myoglobin (PDB ID:3RGK) using the online pairwise structure alignment tool available at <https://www.rcsb.org/alignment>, with the default parameter settings ( algorithm: jFATCAT(rigid) — RMSD Cutoff: 3 — AFP Distance Cutoff: 1600 — Fragment Length: 8).

Figure E7: The highlighted AMINO ACIDS in myoglobin chain correspond to (after structure alignment) the positions which are hemologlobin-α/β tetramer contact points (as identified in Table 3 and Table 4 of [\(Shionyu et al.,](#page-10-12) [2001\)](#page-10-12)). We find that the amino acids K, E, I, which are common in AF S1pMyoglobinq and AF S2pMyoglobinq, are less in number at the contact residues of hemoglobin tetramer and more in number at the corresponding locations in myoglobin, which is a monomer.

#### E.1. Globin Family

![](_page_22_Figure_2.jpeg)

 Figure E8: Multiple sequence alignment of hemoglobin-β and myoglobin sequences. 15 sequences on the left are from hemoglobin-β and on the right are from myoglobin. The sequences are randomly selected from the train set of the protein families. AF SpMyoglobinq amino acids are in green and AF SpHemoglobin-βq in red. The intensity of the color is proportional to the Shapley value ϕpiq of the amino acid i (See Figure [3c\)](#page-7-1)

#### Identifying key amino acid types that distinguish paralogous proteins

![](_page_23_Picture_3.jpeg)

#### E.2. Tubulin

The inter-chain contact residues from the tubulin-α/β heterodimer were identified using ChimeraX 1.4 [\(Pettersen et al.,](#page-10-16) [2021\)](#page-10-16). The *Contacts* tool available in *Tools* Ñ *Structure Analysis* was used with settings as shown in Figure [E9.](#page-23-2) For PDB ID:3JAR we count the residues of chain-A (tubulin-α) and chain-B (tubulin-β) which are in contact with the residues of other tubulin chains. Similarly, for PDB ID:5N5N we count the residues of chain-G (tubulin-α) and chain-B (tubulin-β) which are in contact with the residues of other tubulin chains. The code for counting the AF S residues at the identified contact points of the respective chains is available at [https://anonymous.4open.science/r/AFS\\_AAC\\_SVM-F3D9](https://anonymous.4open.science/r/AFS_AAC_SVM-F3D9).

Figure E9: ChimeraX 1.4 settings for identifying inter-chain contact points from the tubulin-α/β heterodimer and from the histone heterooctamer

#### E.3. Histone

The inter-chain contact residues of histone H2A and H2B were identified from its heterooctameric structure comprising of two H2A/H2B dimers and one H3/H4 tetramer, using ChimeraX 1.4. The *Contacts* tool available in *Tools* Ñ *Structure Analysis* was used with settings as shown in Figure [E9.](#page-23-2) For PDB ID: 1AOI and 3KWQ, we count the residues of an H2A and an H2B chain, which are in contact with other histone chains in the heterooctameric structure. The code for counting the AF S residues at the identified contact points of the respective chains is available at [https://anonymous.4open.](https://anonymous.4open.science/r/AFS_AAC_SVM-F3D9) [science/r/AFS\\_AAC\\_SVM-F3D9](https://anonymous.4open.science/r/AFS_AAC_SVM-F3D9).

![](_page_24_Figure_1.jpeg)

 Figure E10: Multiple sequence alignment of tubulin-α and tubulin-β sequences. 15 sequences on the left are from tubulin-β and on the right are from tubulin-α. The sequences are randomly selected from the train set of the protein families. AF SpTubulin-αq amino acids are in green and AF SpTubulin-βq in red. The intensity of the color is proportional to the Shapley value ϕpiq of the amino acid i (See Figure [E6c\)](#page-19-0)

![](_page_25_Figure_1.jpeg)

 Figure E11: Multiple sequence alignment of histone H2A and histone H2B sequences. 15 sequences on the left are from histone H2B and on the right are from histone H2B. The sequences are randomly selected from the train set of the protein families. AF SpHistone H2Aq amino acids are in green and AF SpHistone H2Bq in red. The intensity of the color is proportional to the Shapley value ϕpiq of the amino acid i (See Figure [E6e\)](#page-19-0)

![](_page_26_Figure_1.jpeg)

 Figure E12: Multiple sequence alignment of interleukin-1 α and interleukin-1 β sequences. 15 sequences on the left are from interleukin-1 β and on the right are from interleukin-1 α. The sequences are randomly selected from the train set of the protein families. AF SpInterleukin-1 αq amino acids are in green and AF SpInterleukin-1 βq in red. The intensity of the color is proportional to the Shapley value ϕpiq of the amino acid i (See Figure [E6d\)](#page-19-0)

1504

1506

1509

1518 1519

1524

1526

1529

1534

1536

## E.4. Marginal contribution feature importance (MCI) [\(Catav et al.,](#page-8-0) [2021\)](#page-8-0) for AF S

For a feature i, its MCI score is defined as,

$$MCI(i) = \max_{S \subseteq N \setminus \{i\}} v(S \cup \{i\}) - v(S),$$

Here, vp¨q is the same as that defined in Section [2.2.](#page-1-0) We compare the amino acids with the top-d (d " size of AF S) MCI scores to the AF S in Table [E6.](#page-27-0) MCI is computed using the same approximation scheme as in Appendix Section [C](#page-18-0) Algorithm [1](#page-18-2) with appropriate modifications.

Table E6: AF S comparison with the amino acids having the top-d MCI [\(Catav et al.,](#page-8-0) [2021\)](#page-8-0) scores. Here, d is the size of AF S for the respective dataset. The amino acids that differ in the two sets are in bold and underlined, with their counts mentioned in the rightmost column. For 8 of 15 datasets, AF S and top-d MCI sets are the same, while only for two datasets do they differ in two amino acids. For all 15 datasets, at least the top-3 MCI amino acids are in AF S. For 11 of these datasets, at least the top-5 MCI amino acids are in AF S.

| Paralog         | pair      | top- d MCI amino acids                                                    |
|-----------------|-----------|---------------------------------------------------------------------------|
|                 |           | (rank-1 Ñ rank- d )                                                       |
|                 |           | AF S Difference                                                           |
| Lysozyme        | C (74)    | and                                                                       |
| α -Lactalbumin  |           | (22) t I, A, D, G, R, F, N, E, W, L u t I, A, D, N, G, R, E, F, L, W u 0  |
| Trypsin         | (66)      | and                                                                       |
| Chymotrypsin    |           | (17) t Y, W, T, A, K, V, I u t Y, W, T, A, V, K, P u 1                    |
| Tubulin-        | α (117)   | and                                                                       |
| Tubulin-        | β (191)   | t Q, M, K, H, F, I, N, A, Y, C u t M, Q, K, N, F, I, H, A, C, Y u 0       |
| Histone         | H2A (180) | and                                                                       |
| Histone         | H2B       | (177) t L, G, K, S, M, T, N, F, Y u t L, G, S, M, K, N, T, Y, F u 0       |
| Interleukin-1   | α (16)    | and                                                                       |
| Interleukin-1   | β         | (25) t G, C, T, V, Q, S, A, I , P u t C, G, T, S, V, Q, A, N , P u 1      |
| Cytochrome      | P450      | CYP3                                                                      |
| (32) and        | CYP51     | (32) t H, F, G, K, A, P, N u t H, F, G, K, A, P, N u 0                    |
| Myoglobin       | (107)     | and                                                                       |
| Hemoglobin-     | α         | (303) t V, Y, E, K, S, G, W, I, C, P u t E, S, Y, V, K, P, I, G, C, W u 0 |
| Myoglobin       | (107)     | and                                                                       |
| Hemoglobin-     | β         | (285) t V, K, E, C, W, N, F, Y, M, I u t K, V, C, E, W, N, F, M, Y, I u 0 |
| Hemoglobin-     | α (303)   | and                                                                       |
| Hemoglobin-     | β         | (285) t W, S, N, P, V u t W, P, N, S, G u 1                               |
| Rhodopsin-like  | (181)     | and                                                                       |
| Glutamate-like  |           | (89) t D, E, Q, G, L, I u t D, Q, E, G, M , L u 1                         |
| Secretin-like   | (90)      | and                                                                       |
| Glutamate-like  |           | (89) t W, H, Y, V, D u t W, H, Y, V, D u 0                                |
| Rhodopsin-like  | (181)     | and                                                                       |
| Secretin-like   |           | (90) t W, E, H, Q, S, M, V, A u t W, E, M, S, V, H, Q, A u 0              |
| Rhodopsin-like  |           | GPCRs                                                                     |
| Aminergic       | receptors | (186)                                                                     |
| and Lipid       | receptors | (113) t L, E, P, K , F, D, I u t L, P, E, W , F, M , D u 2                |
| Aminergic       | receptors | (186)                                                                     |
| and Peptide     | receptors | (367) t L, E, K, F, M, H , R, D u t L, F, E, M, K, D, V , R u 1           |
| Lipid receptors | (113)     | and                                                                       |
| Peptide         | receptors | (367) t R, G, P, K , I, V, T u t P, R, G, I, W , S , V u 2                |