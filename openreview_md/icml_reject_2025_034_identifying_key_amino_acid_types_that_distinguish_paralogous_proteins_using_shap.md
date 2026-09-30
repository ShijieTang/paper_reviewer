# Identifying Key Amino Acid Types That Distinguish Paralogous Proteins Using Shapley Value Based Feature Subset Selection

## Anonymous Authors1

000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054

## Abstract

Paralogous proteins have a common ancestor but have diverged in functionality. Using known machine learning algorithms, we present a datadriven method to identify the key amino acid types that play a role in distinguishing a given pair of proteins that are paralogs. We use an existing Shapley value based feature subset selection algorithm, SVEA, to identify the key amino acid types adequate to distinguish pairs of paralogous proteins. We refer to these as the amino acid feature subset (*AF S*). For a paralog pair, say proteins P and Q, its *AF S* is partitioned based on protein-wise importance as *AF S*pPq and *AF S*pQq using a linear classifier, SVM. To validate the significance of the *AF S* amino acids, we use multiple domain knowledge based methods : (a) multiple sequence alignment, and/or (b) 3D structure analysis, and/or (c) supporting evidence from biology literature. This method is computationally cheap, requires less data and can be used as an initial data-driven step for further hypothesis-driven experimental study of proteins.

We demonstrate the results for 15 pairs of paralogous proteins. Code at https://anonymous. 4open.science/r/AFS_AAC_SVM-F3D9.

## 1. Introduction

Proteins form the fundamental machinery in living systems, having several vital functions such as DNA replication, catalysis, transport, environmental interaction, etc. Advancements in sequencing technologies have resulted in exponential growth of protein sequence databases (The UniProt Consortium, 2020). However, the number of experimentally verified annotations constitute a tiny fraction: only 0.57 of 250 million sequences in UniProtKB (The 1Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>.

1 UniProt Consortium, 2020) have manually reviewed annotations. Experimental methods for determining biological process level functions (transcription, DNA repair, etc.) are high-throughput whereas methods for molecular function (catalysis, ligand specificity, etc.) are low-throughput and hence are not scalable. The relationship between sequence and function is subtle and has not been fully decoded yet. Paralogs are proteins that have a common ancestor but have diverged functionally. The functional difference in two paralogous proteins is considered to arise due to evolutionary changes in the sequences (Yang et al., 2023). A typical experiment to investigate the role of an (or a group of) amino acid(s) in the function of a protein is to perform a site-directed mutagenesis experiment: replace one or more amino acids and test the effect of the sequence change (Kresge et al., 2006). In this work, we provide an algorithmic ML pipeline, consisting both feature engineering and feature subset selection, as a quick and resource-cheap test to assess the likely outcome from a site-directed mutagenesis experiment. We use a diverse dataset of 15 paralog pairs. Our datasets show a range of sequence and function diversity (details in Appendix B). Longest common subsequence score (*lcss*) is a metric to quantify sequence diversity and median within-class *lcss* is ď 0.5 in 12 of the 15 datasets, and the median interclass *lcss* for the corresponding classes is less than withinclass *lcss*. Functional diversity, as discerned from biology literature, also shows large diversity from subtle functional differences (e.g., trypsin/chymotrypsin) to drastic (e.g., lysozyme c/α-lactalbumin). Function description is fine-grained (e.g., trypsin/chymotrypsin) as well as coarse grained (e.g, GPCRs). Our findings are that small subsets of amino acids can discern differences between pairs of paralogs. The subset sizes are between 5 to 10, the median being 8. We provide validations from literature, MSA (a popular computational tool to assess evolutionary conservation) and logical consistencies; for many pairs such validations are more than one. Towards this, we view a protein as the composite of its constituent standard 20 amino acids. We use amino acid composition (AAC) features, a Shapley value (Shapley, 1953) based feature subset selection algorithm (Shapley Value based Error Apportioning, SVEA) (Tripathi et al., 2020; 2021), and a linear support vector machine (SVM) classifier (Steinwart & Christmann, 2008) as tools to identify key amino acid types that can distinguish a given a pair of proteins that are paralogs. It yields quick results based on which biologists can conduct detailed experiments which are resource-intensive (time, cost, trained manpower, etc.). The key results from our ML pipeline experiments are:
055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109
' Using known machine learning algorithms we demonstrate a data-driven method to identify key amino acids that distinguish two paralogous proteins.

- The SVEA algorithm identifies a subset of amino acid types (referred to as *AF S*) adequate for distinguishing two paralogous proteins. The size of *AF S* ranges from 5 to 10 amino acids out of 20. (Table 1)
- For a paralog pair, say protein families P and Q,
the computed *AF S* is partitioned into *AF S*pPq and AF SpQq using a linear SVM, to determine the familywise importance of *AF S*. (Table 1)
' Domain knowledge based validation of AF S: The significance of the amino acids in *AF S* was validated for 14 datasets using various methods like (a) multiple sequence alignment (MSA) and/or (b) structural analysis and/or (c) supporting evidence from literature that report structural/functional role of these amino acids.

' Logical consistencies in the pair-wise *AF S* of three paralogous proteins (globins, Section 3.1.7, and GPCRs, Section 3.1.8). If families P vs Q and P vs R have *AF S*1 and AF S2, then,
- we find common amino acids in *AF S*1pPq and AF S2pPq, except for one pair.

- amino acids in AF S1 X *AF S*2 are either excluded from AF S3, which is from Q vs R, or have much lower Shapley value in AF S1, *AF S*2, or *AF S*3.

' Validation of *AF S* using test data (Section 3.2): The composition of amino acids is sufficient to classify several paralog pairs. A linear SVM classifies with high test scores (70-99%) using only the composition of *AF S* amino acids as features. (Appendix Table E5)
' *AF S* are top ranked features with an alternate feature ranking measure, Marginal Contribution feature importance (MCI) (Catav et al., 2021). (Appendix Table E6) Shapley values based feature attribution methods are popular for explaining machine learning models (Rozemberczki et al., 2022). One such method is SHAP (Lundberg & Lee, 2017), which assigns attribution scores to input features based on a model's output for a given instance input. Another method is SAGE (Covert et al., 2020), which assigns feature attribution scores based on a model's loss computed at the dataset level. Unlike these methods, where feature attributions are based on a trained model, the SVEA algorithm that we use for our task assigns scores to the features based on the distribution of the data points in the feature space and their ground truth labels. The SVEA algorithm uses a function vpSq, which acts as a measure of inter-class linear separation between the data points in the space of the feature subset S. The scores assigned to the features are Shapley values computed using this function vp¨q. We also use an alternate feature ranking method, i.e. the Marginal Contribution Feature Importance (MCI) (Catav et al., 2021). MCI is an axiomatic approach that was proposed as an alternative to Shapley values to score and rank features. We find close agreement between the *AF S* computed using SVEA and the top-ranked amino acids using MCI. Use of deep learning methods trained on large datasets is becoming commonplace in Biology; for example, prediction of molecular function via EC number or GO annotation (Bileschi et al., 2022; Sanderson et al., 2023), identifying input sequence regions relevant to model output (Zhou et al., 2016) and learning sequence-function mapping from deep mutational scanning experiment data (Song et al., 2021). The use of large datasets for training makes this approach highly resource-intensive. The approach we present herein needs much smaller datasets and, consequently, (i) is computationally cheap and (ii) has far wider applicability since labelled data validated by wet lab experiments is limited.

## 2. Methodology

We discuss the main components of our methodology.

## 2.1. Aac Features

Consider a paralogous pair of proteins, families P and Q. We first curate a set of sequences, say DP and DQ,
from a standard protein sequence database, SwissProt (The UniProt Consortium, 2020), with nP and nQ number of sequences each from families P and Q respectively. For a protein sequence p pjq " pp pjq 1, p pjq 2
, . . . , p pjq Lq of length L
with p pjq kP t1, 2, **¨ ¨ ¨** , 20u corresponding to the standard 20 amino acids, the AAC feature x AAC
j P r0, 1s 20 for p pjqis computed as follows,

$$x_{j,i}^{A A C}={\frac{1}{L}}\sum_{k=1}^{L}\mathbf{1}_{\{p_{k}^{(j)}=i\}},\,\forall i\in[20]$$

So x AAC
j,i is the normalised count of the standard amino acid i, i P t1, 2, **¨ ¨ ¨** , 20u, in a protein p pjq.

## 2.2. Feature Subset Selection Using Svea

Given a set, N, of features from the protein sequences of P and Q, we try to find the features S Ď N that contribute the most to the linear separation of P and Q sequences. With AAC features, we have N " t1, 2*, . . . ,* 20u corresponding to each of the standard 20 amino acid types.

110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 We utilise the Shapley value based feature ranking and subset selection algorithm, SVEA (Tripathi et al., 2020; 2021),
to identify the most important feature subset S Ď N. Shapley value is a well known solution concept from cooperative game theory (Shapley, 1953; Narahari, 2014) for distributing the total worth of a coalition of players fairly among each of them by quantifying each player's effective marginal contribution. The SVEA algorithm considers the binary classification task as a cooperative game among the features, with a function vpSq as the worth of every feature subset S. vpSq acts as a measure of linear separation between the classes in the feature space of S. Accounting for classimbalance, we define vpSq using a class-balanced hinge loss function tr erpSq, which is defined as,

$$t r\_e r(S)=\operatorname*{min}_{w,\xi_{j}}{\frac{1}{2n_{P}}}\sum_{j=1}^{n_{P}}\xi_{j}+{\frac{1}{2n_{Q}}}\sum_{j=n_{P}+1}^{n_{Q}}\xi_{j}$$
s.t. $y_{j}\left(\sum_{i\in S}w_{i}x_{j,i}^{AAC}+b\right)\geq1-\xi_{j},\ \forall j\in[n_{P}+n_{Q}]$  $\xi_{j}\geq0,\ \forall j\in[n_{P}+n_{Q}]$

and vpSq " tr er**pHq ´** tr erpSq. The minimizer in the above finds a linear hyperplane with the least class-balanced hinge loss in the feature space of S. H is the empty set and tr er**pHq "** 1, therefore, vpSq " 1 ´ tr erpSq. tr erpSq " 0 implies vpSq " 1, i.e., the two classes are completely linearly separable in the feature space of S. The maximum value of tr erpSq possible is 1. The Shapley value ϕpiq for a feature i P N is computed as,

$$\phi(i)=\sum_{S\subseteq N\setminus\{i\}}{\frac{|S|!(|N|-|S|-1)!}{|N|!}}(v(S\cup\{i\})-v(S)).$$

Thus, ϕpiq is a weighted sum of the marginal contribution of feature i to all the possible feature subsets that do not contain i. Shapley values are unique solution concepts satisfying the axioms - efficiency, symmetry and marginality (Young, 1985). The higher the ϕpiq, the higher the contribution of feature i to the linear separation between the classes and, consequentially, the higher the importance of feature i distinguishing the classes. Exact Shapley value computations are known to be exponential time. Hence, they are computed using a linear time (in number of features) Monte Carlo approximation (Castro et al., 2009) in the SVEA algorithm. As the number of features is small (20), good approximations can be computed fast via larger sampling. More details of the SVEA algorithm are given in Appendix Section C.

Data-driven cutoff for selecting *AF S*: The efficiency axiom of Shapley value implies, ř20 i"1 ϕpiq " vpNq. If all features have equal contribution in achieving vpNq, then ϕpiq " 
vpNq 20 
, @i P N. Consequentially, if a feature i had lesser contribution than others then ϕpiq ă vpNq 20 . Therefore, we set ϕ*cutof f* "
vpNq 20 for selecting the key distinguishing amino acid feature subset, *AF S* " ti : ϕpiq ě ϕ*cutof f* u.

Each of the features in *AF S* uniquely corresponds to d ď 20 amino acids from the standard 20.

## 2.3. Protein Family-Wise Partition Of Af S **Using Svm**

We train a linear SVM, to classify P vs Q, using the composition of the amino acids in *AF S* as the features, i.e. using x AF S
j P r0, 1s d, with x AF S
j,i1 " x AAC
j,i and each i 1 P t1, 2, **¨ ¨ ¨** , du uniquely maps to a i P *AF S*. We use these linear SVM weights w P R
dto divide the set AF S into disjoint sets *AF S*pPq and *AF S*pQq based on the sign of the weights. Since x AF S
j,i1 ě 0 @i 1 P rds, the sign of the linear classifier weight wi 1 indicates which class is relatively prominent in the amino acid corresponding to i 1. So if the `1 class is P, then we divide *AF S* classwise as *AF S*pP**q " t**i 1 P rds : wi1 ą 0u and similarly AF SpQ**q " t**i 1 P rds : wi1 ă 0u. See Appendix Section D
for details on SVM training.

A flowchart summarizing the steps for computing *AF S*pPq and *AF S*pQq is shown in Figure 1.

## 2.4. Validation Of Af S

Literature evidence: For 14 different paralog protein pairs, we provide supporting evidence from protein biology literature for the significance of amino acids in *AF S* in the functional specificity of the protein pair. MSA analysis: We also compute multiple sequence alignment (MSA) of randomly selected sequences from DP and DQ and analyze the conservation of *AF S*pPq and *AF S*pQq amino acids within and across the respective families (Figure 2). MSA algorithms (Edgar & Batzoglou, 2006) aim to align multiple protein sequences by inserting gaps in the sequences while optimizing an objective. The objective is usually to minimize the number of gaps inserted while maximizing an overall score that promotes the alignment of similar (based on physicochemical properties) amino acids at a given position. The alignments are often used as a tool to determine homologous relationships between proteins and identify conserved or mutated regions in them. Structural analysis: For paralog pairs that together function as heteromers (protein complexes made up of different types of proteins), we perform structural analysis to validate the role of *AF S* in the heteromeric structure formed by the paralog pair (Sections 3.1.7, 3.1.3 and 3.1.4). Using test data: We test the classifier trained in Section 2.3 on a test data. (Details on test data in Appendix Section A.1).

## 3. Results And Discussions

3.1. Role of the amino acids identified in *AF S* For 15 paralog pairs, we discuss the significance of the amino acids identified in the respective *AF S* (Table 1).

3.1.1. LYSOZYME C AND α-LACTALBUMIN
Literature evidence: Amino acids D and E of AF Spα-Lactalbuminq are found in the Ca2` and Zn2`
binding sites respectively of α-lactalbumin (Permyakov & Berliner, 2000; Permyakov, 2020). All α-lactalbumins studied so far are known to bind Ca2` and Zn2` whereas several (but not all) lysozymes do not bind Ca2`. MSA analysis: (Figure 2a) *AF S*pα-Lactalbuminq and AF SpLysozyme Cq amino acids (Table 1) are significantly conserved in respective families.

3.1.2. TRYPSIN AND CHYMOTRYPSIN
Literature evidence: Y and W get the highest Shapley value ϕp¨q in *AF S*pTrypsinq and *AF S*pChymotrypsinq re-
In general, we find an imbalance in the number of sequences for the two paralogous proteins. It is known that accuracy is not a well-suited performance measure of the classifier in class imbalance settings. Therefore, we use the arithmetic mean of sensitivity and specificity (AM) to measure the performance of the classifier (Brodersen et al., 2010). Using marginal contribution feature importance (MCI): We check agreement of *AF S* with another feature ranking method, MCI (Catav et al., 2021). See Appendix Section E.4 for details on MCI computation.

165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 spectively (Table 1 and Figure E6b). In experiments to convert trypsin to chymotrypsin (Hedstrom et al., 1994; Hedstrom, 2002) it has been shown that Y to W conversion in loop-3 of trypsin leads to significant increase in chymotrypsin activity. We do not find *S, H* and D in *AF S*,
which are important for the function of both families and are known as the catalytic triad (Dodson & Wlodawer, 1998).

## 3.1.3. Tubulin-Α And Tubulin-Β

MSA analysis: (Appendix Figure E10) *AF S*pTubulin-αq and *AF S*pTubulin-βq amino acids are significantly conserved in respective families.

Structural analysis of *AF S*: Tubulins typically exist as heterodimers, consisting of two subunits: tubulin-α and tubulin-β (Muhlethaler et al. ¨ , 2021). We looked at the contact residues of a tubulin-α chain and tubulin-β chain in the 3D structure of tubulin-α/β heterodimer (PDB IDs: 3JAR,
5N5N). We see that the contact points of the tubulin-α chain in the heterodimer have more *AF S*pTubulin-αq amino acids than *AF S*pTubulin-βq. Similarly, *AF S*pTubulin-βq amino acids are more than *AF S*pTubulin-αq at the contact point of the tubulin-β chain in the heterodimer. Thus, the amino acids identified in *AF S* can be considered to be significant towards the quaternary structure of tubulin-α/β heterodimer. Appendix Section E.2 has more details.

## 3.1.4. Histone H2A And Histone H2B

MSA analysis: (Appendix Figure E11), AF SpHistone H2Aq and *AF S*pHistone H2Bq amino acids are significantly conserved in respective families. Structural analysis of *AF S*: Histones have a heterooc-

| pair. For globins and GPCRs, common acids across different AF S within a paralog triplet are colour-coded. Paralog pair Amino acid feature subset, AF S Class-wise AF S parition Lysozyme C (74) and AF Spα-Lactalbuminq " tI, D, E, F, Lu α-Lactalbumin (22) tI, A, D, N, G, R, E, F, L, Wu AF SpLysozyme Cq " tA, N, G, R, Wu Trypsin (66) and AF SpTrypsinq " tY, Au Chymotrypsin (17) tY, W, T, A, V, K, Pu AF SpChymotrypsinq " tW, T, V, K, Pu Tubulin-α (117) and AF SpTubulin-αq " tK, I, H, C, Y u Tubulin-β (191) tM, Q, K, N, F, I, H, A, C, Y u AF SpTubulin-βq " tM, Q, N, F, Au Histone H2A (180) and AF SpHistone H2Aq " tL, G, Nu Histone H2B (177) tL, G, S, M, K, N, T, Y, Fu AF SpHistone H2Bq " tS, M, K, T, Y, Fu Interleukin-1 α (16) and AF SpInterleukin-1 αq " tT, S, A, Nu Interleukin-1 β (25) tC, G, T, S, V, Q, A, N, Pu AF SpInterleukin-1 βq " tC, G, V, Q, Pu Cytochrome P450 CYP3 AF SpCYP3q " tF, K, P, Nu (32) and CYP51 (32) tH, F, G, K, A, P, Nu AF SpCYP51q " tH, G, Au Globins Myoglobin (107) and AF S1pMyoglobinq " tE, K, I, G, Wu Hemoglobin-α (303) AF S1 " tE, S, Y, V, K, P, I, G, C, Wu AF S1pHemoglobin-αq " tS, Y , V , P, Cu Myoglobin (107) and AF S2pMyoglobinq " tK, E, M, Iu Hemoglobin-β (285) AF S2 " tK, V, C, E, W, N, F, M, Y, Iu AF S2pHemoglobin-βq " tV , C, W, N, F, Y u Hemoglobin-α (303) and AF S3pHemoglobin-αq " tP, Su Hemoglobin-β (285) AF S3 " tW, P, N, S, Gu AF S3pHemoglobin-βq " tW, N, Gu GPCRs Rhodopsin-like (181) and AF S1pRhodopsinq " tM, Lu Glutamate-like (89) AF S1 " tD, Q, E, G, M, Lu AF S1pGlutamateq " tD, Q, E, Gu Secretin-like (90) and AF S2pSecretinq " tW, H, Y u Glutamate-like (89) AF S2 " tW, H, Y, V, Du AF S2pGlutamateq " tV , Du Rhodopsin-like (181) and AF S3pRhodopsinq " tM, S, V , Au Secretin-like (90) AF S3 " tW, E, M, S, V, H, Q, Au AF S3pSecretinq " tW, E, H, Qu Rhodopsin-like GPCRs Aminergic receptors (186) AF S1pAminergic receptorsq " tP, E, W, Du and Lipid receptors (113) AF S1 " tL, P, E, W, F, M, Du AF S1pLipid receptorsq " tL, F, Mu Aminergic receptors (186) AF S2pAminergic receptorsq " tE, K, D, Ru and Peptide receptors (367) AF S2 " tL, F, E, M, K, D, V, Ru AF S2pPeptide receptorsq " tL, F, M, V u AF S3pLipid receptorsq " tR, G, Su Lipid receptors (113) and Peptide receptors (367) AF S3 " tP, R, G, I, W, S, V u AF S3pPeptide receptorsq " tP, I, W, V u   |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

Table 1: *AF S* and its class-wise partition computed for 15 paralog pairs. The number of unique sequences from the SwissProt (The UniProt Consortium, 2020) database used for computing *AF S* is given inside parenthesis p¨q for each protein family. Data collection details are in Appendix Section A.1. *AF S* amino acids are written in decreasing Shapley values from left to right for each paralog pair. Figures 3 and E6 show the Shapley value of the amino acids for each paralog pair. For globins and GPCRs, common acids across different *AF S* within a paralog triplet are colour-coded.

220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 tameric structure comprising of two H2A/H2B dimers and one H3/H4 tetramer (Dutta et al., 2001). We looked at the contact residues of an H2A chain and H2B chain in the heteroocatmer structure of histone (PDB IDs: 3KWQ, 1AOI). We find that the contact points of H2A
chain in the heterooctamer have more *AF S*pHistone H2Aq amino acids than *AF S*pHistone H2Bq. This is interesting since *AF S*pHistone H2Aq has only three amino acids, while *AF S*pHistone H2Bq has six amino acids. Similarly, the contact points of H2B chain in the heterooctamer have more *AF S*pHistone H2Bq amino acids than AF SpHistone H2Aq. Thus, the amino acids identified in AF S can be considered to be significant towards the quaternary structure of the histone heterooctamer. See Appendix Section E.3 for more details.

## 3.1.5. Interleukin-1 Α And Interleukin-1 Β

Literature Evidence: C has the highest Shapley value and is in *AF S*pInterleukin-1 βq. Deleting C results in loss of activity in Interleukin-1 β (Veerapandian et al., 1992). We do not find such studies for Interleukin-1 α. MSA analysis: (Appendix Figure E12) AF SpInterleukin-1 αq and *AF S*pInterleukin-1 βq amino acids show significant conservation in respective families.

## 3.1.6. Cytochrome P450 Cyp3 And Cyp51

Literature evidence: *H, F* and G, in the respective order, have the highest Shapley value ϕp¨q for this paralogous pair (Table 1 and Figure E6f). H and G with the highest ϕp¨q in *AF S*pCYP51q have been reported (Nitahara et al., 2001; Lepesheva & Waterman, 2004; 2007; Strushkevich et al.,

Identifying key amino acid types that distinguish paralogous proteins 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329
❄
 ❄ 
(a) Lysozyme C vs α-Lactalbumin
❄
 ❄ 
(b) Hemoglobin-α vs Myoglobin
❄
 ❄ 
(c) Hemoglobin-β vs Hemoglobin-α
2010) to be important in the enzymatic activity of CYP51. Mutation of these amino acids at specific positions has been shown to result in a decrease in the activity of the enzyme (Lepesheva & Waterman, 2007; 2004). Similarly, F with the highest ϕp¨q in *AF S*pCYP3q is also known to be important in the enzymatic activity of CYP3 (Qiu et al., 2008; Denisov et al., 2019; Zhang et al., 2024). A cluster of F residues in CYP3 is known to form a substrate-binding pocket with an active site (Zhang et al., 2024).

## 3.1.7. Globins

MSA analysis: (Figures 2b,2c and Appendix Figure E8) For the three globin paralog pairs (Table 1), we observe in the MSA, conservation of the class-wise partition of *AF S* in the respective families. Structural analysis of *AF S*: Myoglobin is a monomer, while α and β chains together constitute hemoglobin, a tetramer of composition α2β2 (Dill et al., 2017). We superimposed the 3D structures of myoglobin, hemoglobinα and hemoglobin-β (PDB IDs: 3RGK, 1HHO) and mapped the *α, β* contact residues (based on (Shionyu et al., 2001)) of hemoglobin tetramer to that of myoglobin. We find that the amino acids *K, E, I*, which are common in *AF S*1pMyoglobinq and *AF S*2pMyoglobinq, are less in number at the contact residues of hemoglobin tetramer and more in number at the corresponding locations in myoglobin, which is a monomer (see Appendix Figure E7).

Literature evidence: W with a significantly high Shapley value ϕpWq (Figure 3b), is present in AF S3pHemoglobin-βq. It is highly conserved at position 40 in the MSA (Figure 2c) in hemoglobin-β sequences as compared to hemoglobin-α sequences. This W at position 40 has been determined to be present in hemoglobin-β at one of its contact positions to hemoglobin-α in the tetrameric structure (Shionyu et al., 2001) and is, therefore, a structurally and functionally significant residue. C, present in *AF S*1pHemoglobin-αq and *AF S*2pHemoglobin-βq, has been shown to play an important role in the tetrameric structure of hemoglobin formed by α and β hemoglobins (Kan et al., 2013). Logical consistencies in *AF S* (refer to Table 1 (Globins)
for AF S1, AF S2*, AF S*3): ' AF S1 X *AF S*2 " t*E, Y, V, K, I, C, W*u. Except for W with the least Shapley value in *AF S*1 (Figure 3a), the remaining are excluded from *AF S*3.

- Explanation:V, Y, C in *AF S*1pHemoglobin-αq X
AF S2pHemoglobin-βq can be expected not to be key in AF S3 for distinguishing α vs β hemoglobin.

' AF S2 X *AF S*3 " tW, Nu. N is excluded from *AF S*1, while W gets the least Shapley value in *AF S*1 (Figure 3a).

' AF S3 X *AF S*1 " tW, P, S, Gu. t*P, S, G*u are excluded from *AF S*2, while W gets the least Shapley value in *AF S*1.

The Shapley value for W is very close to the cut-off in AF S1 (Figure 3a). If it is dropped from *AF S*1, then the exclusion principle illustrated above would be more prominent as in GPCRs (Section 3.1.8).

## 3.1.8. G-Protein Coupled Receptors (Gpcrs)

Literature evidence: W (with highest Shapley value ϕp¨q) and H common in *AF S*2pSecretinq and *AF S*3pSecretinq
(Table 1 and Figure 3), are well conserved at multiple positions with structural importance and functional importance in secretin-like GPCR sequences (Cary et al., 2022; Harmar, 2001). Mutating certain conserved W leads to a loss in expression of this GPCR at the cell surface, where it functions (Cary et al., 2022). H present in the intracellular loop region is also known to be important in the activation of certain secretin-like GPCRs (Harmar, 2001).

M common in *AF S*1pRhodopsinq and *AF S*3pRhodopsinq has been found to be present at important binding pockets and a position important for activation of the GPCR (Okada et al., 2001; Sakmar et al., 2002). S from AF S3pRhodopsinq is found at multiple major phosphorylation sites (see Okada et al. 2001 for details) in Rhodopsin. Mutating D at two positions has been shown to affect glutamate binding of glutamate receptor GPCRs (Jingami et al., 2003). D is common in *AF S*1pGlutamateq and AF S2pGlutamateq and has highest Shapley value in *AF S*1. E and D common in *AF S*1pAminergicq and AF S2pAminergicq are present at binding sites of important ligands (like histamine/serotonin) of aminergic receptors (Vass et al., 2019). Logical consistencies in *AF S* **of GPCRs** (refer to Table 1
(GPCRs) for AF S1, AF S2*, AF S*3): ' AF S1 X AF S2 " tDu, is excluded from *AF S*3. ' AF S2 X AF S3 " t*W, H, V* u, is excluded from *AF S*1. ' AF S3 X *AF S*1 " t*Q, E, M*u, is excluded from *AF S*2.

Logical consistencies in *AF S* **of Rhodopsin-like GPCR** subfamilies (refer to Table 1 (Rhodopsin-like GPCRs) for AF S1, AF S2*, AF S*3): ' AF S1 X *AF S*2 " t*L, E, F, M, D*u, is excluded from AF S3. ' AF S2 X *AF S*3 " t*R, V* u, is excluded from *AF S*1. ' AF S3 X *AF S*1 " t*P, W*u is excluded from *AF S*2. The explanations for these consistencies are similar to that in globins (Section 3.1.7).

330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384

Identifying key amino acid types that distinguish paralogous proteins 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439
(a) Myoglobin vs Hemoglobin-α (b) Hemoglobin-α vs Hemoglobin-β (c) Myoglobin vs Hemoglobin-β
(d) Secretin-like vs Glutamate-like (e) Rhodopsin-like vs Glutamate-like (f) Rhodopsin-like vs Secretin-like

## 3.2. Validation Of Af S **Using Test Data**

For an *AF S* of size d, the top-d amino acids ranked by MCI differ with *AF S* only in at the most two amino acids. For 8 of 15 datasets, *AF S* and top-d MCI sets are the same, while only for two datasets do they differ in two amino acids. For all 15 datasets, at least the top-3 MCI amino acids are in AF S. For 11 of these datasets, at least the top-5 MCI amino acids are in AF S. (Appendix Table E6)

## 4. Conclusion

We demonstrated an ML pipeline to identify the key amino acid types, *AF S*, that distinguish a pair of paralogous proteins. The role of *AF S* in functionally distinguishing the paralog pairs was validated using various sources of domain knowledge. The robustness of this approach, as demonstrated by considering a diverse set of paralogous protein The classification scores on test data for the classifiers trained using AAC and *AF S* features, respectively, are reported in Appendix Table E5. Using *AF S* features, the test AM scores are at least 70%. For 13 of 15 paralog pairs, the scores are greater than 83%, and for 8 of 15 paralog pairs, it is greater than 90%. Details of the test data are provided in Appendix Section A.1.

## 3.3. **Marginal Contribution Feature Importance (Mci) Of** Af S

pairs, illustrates its wider applicability. Identification of AF S can be used as an initial data-driven step before doing more detailed experimental investigations, like site-directed mutagenesis (Bachman, 2013) resolving sequence-function relationship. As the size of *AF S* is small (5-10 amino acids of 20), significantly less number of mutations can be tried. As our pipeline works without using the sequence order information of the amino acids in the protein, it posits an interesting question to biologists : how amino acid composition by itself is able to distinguish paralogs given ample evidence that 3D structure and function are conserved despite sequence divergence (Lau et al., 2015)! Notably, amino acids in the *AF S* typically occur more than once in the sequence, but our method is silent on the specific positions where the amino acid has a functionally distinguishing role. This may be addressed by engineering features that incorporate sequence order information from the protein.

However, these features can be very high-dimensional, for example, 20k-dimensional for k-mer features. The Monte Carlo based approximation algorithm for Shapley values would require exponentially more sampling (in number of features) for good approximations.

## References

440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494

## Impact Statement

This paper presents a computationally efficient data lean ML pipeline. It can be used by biologists to decide whether they should invest valuable resources (skilled manpower, time, funds, etc.) for performing wet-lab experiments to determine amino acid(s) that are critical for functional differentiation of paralogous proteins. Bachman, J. Chapter ninteen - site-directed mutagenesis. In Lorsch, J. (ed.), Laboratory Methods in Enzymology: DNA, volume 529 of Methods in Enzymology, pp. 241–248. Academic Press, 2013. doi:
10.1016/B978-0-12-418687-3.00019-7. URL https:
//www.sciencedirect.com/science/
article/pii/B9780124186873000197.

Begum, K., Mohl, J. E., Ayivor, F., Perez, E. E., and Leung, M.-Y. GPCR-PEnDB: a database of protein sequences and derived features to facilitate prediction and classification of G protein-coupled receptors. *Database*, 2020, 11 2020. ISSN 1758-0463. doi: 10.1093/database/baaa087. URL https://doi.org/ 10.1093/database/baaa087.

Bileschi, M. L., Belanger, D., Bryant, D. H., Sanderson, T., Carter, B., Sculley, D., Bateman, A., DePristo, M. A., and Colwell, L. J. Using deep learning to annotate the protein universe. *Nature biotechnology*, 40(6):932—937, June 2022. ISSN 1087-0156. doi: 10.1038/s41587-021-01179-w. URL https://doi.

org/10.1038/s41587-021-01179-w.

Brodersen, K. H., Ong, C. S., Stephan, K. E., and Buhmann, J. M. The balanced accuracy and its posterior distribution. In *2010 20th ICPR*, pp. 3121–3124, 2010. doi: 10.1109/ ICPR.2010.764.

Cary, B. P., Zhang, X., Cao, J., Johnson, R. M., Piper, S. J.,
Gerrard, E. J., Wootten, D., and Sexton, P. M. New Insights into the Structure and Function of Class B1 GPCRs. Endocrine Reviews, 44(3):492–517, 12 2022. ISSN 0163769X. doi: 10.1210/endrev/bnac033. URL https: //doi.org/10.1210/endrev/bnac033.

Castro, J., Gomez, D., and Tejada, J. Polynomial ´
calculation of the shapley value based on sampling.

Computers & Operations Research, 36(5):1726–1730, 2009. ISSN 0305-0548. doi: 10.1016/j.cor.2008.04.004.

URL https://www.sciencedirect.com/ science/article/pii/S0305054808000804. Selected papers presented at the Tenth International Symposium on Locational Decisions (ISOLDE X).

Catav, A., Fu, B., Zoabi, Y., Meilik, A. L. W., Shomron, N., Ernst, J., Sankararaman, S., and Gilad-Bachrach, R. Marginal contribution feature importance - an axiomatic approach for explaining data. In Meila, M. and Zhang, T. (eds.), Proceedings of the 38th International Conference on Machine Learning, volume 139 of *Proceedings of* Machine Learning Research, pp. 1324–1335. PMLR, 18– 24 Jul 2021. URL https://proceedings.mlr. press/v139/catav21a.html.

Covert, I., Lundberg, S. M., and Lee, S.-I. Understanding global feature contributions with additive importance measures. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 17212–17223. Curran Associates, Inc., 2020. URL https://proceedings.neurips.

cc/paper_files/paper/2020/file/ c7bf0b7c1a86d5eb3be2c722cf2cf746-Paper. pdf.

Denisov, I. G., Grinkova, Y. V., Nandigrami, P., Shekhar, M.,
Tajkhorshid, E., and Sligar, S. G. Allosteric interactions in human cytochrome p450 cyp3a4: The role of phenylalanine 213. *Biochemistry*, 58(10):1411–1422, 2019. doi:
10.1021/acs.biochem.8b01268. URL https://doi. org/10.1021/acs.biochem.8b01268. PMID:
30785734.

Dill, K., Jernigan, R., and Bahar, I. Protein Actions:
Principles and Modeling. CRC Press, 2017. ISBN 9781351815000. URL https://books.google. co.in/books?id=NHs2DwAAQBAJ.

Dodson, G. and Wlodawer, A. Catalytic triads and their relatives. *Trends in Biochemical Sciences*, 23(9):347–352, 1998. ISSN 0968-0004. doi: https://doi.org/10.1016/S0968-0004(98)01254-7.

URL https://www.sciencedirect.com/ science/article/pii/S0968000498012547.

Dutta, S., Akey, I. V., Dingwall, C., Hartman, K. L.,
Laue, T., Nolte, R. T., Head, J. F., and Akey, C. W. The crystal structure of nucleoplasmin-core: Implications for histone binding and nucleosome assembly. *Molecular Cell*, 8(4):841–853, 2001. ISSN 1097-2765. doi: 10.1016/S1097-2765(01)00354-9. URL https://www.sciencedirect.com/ science/article/pii/S1097276501003549.

Edgar, R. C. and Batzoglou, S. Multiple sequence alignment.

Current Opinion in Structural Biology, 16(3):368–373, 2006. ISSN 0959-440X. doi: 10.1016/j.sbi.2006.04.004.

URL https://www.sciencedirect.com/ science/article/pii/S0959440X06000704.

Nucleic acids/Sequences and topology.

Fu, L., Niu, B., Zhu, Z., Wu, S., and Li, W. CD-
HIT: accelerated for clustering the next-generation sequencing data. *Bioinformatics*, 28(23):3150–3152, 10 2012. ISSN 1367-4803. doi: 10.1093/ bioinformatics/bts565. URL https://doi.org/10. 1093/bioinformatics/bts565.

Kresge, N., Simoni, R. D., and Hill, R. L. The development of site-directed mutagenesis by michael smith. Journal of Biological Chemistry, 281(39):e31–e33, 2006. ISSN 0021-9258. doi: 10.1016/S0021-9258(19)33938-9. URL https://www.sciencedirect.com/ science/article/pii/S0021925819339389.

495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 Lau, C. K., Turner, L., Jespersen, J. S., Lowe, E. D., Petersen, B., Wang, C. W., Petersen, J. E., Lusingu, J., Theander, T. G., Lavstsen, T., and Higgins, M. K. Structural conservation despite huge sequence diversity allows epcr binding by the pfemp1 family implicated in severe childhood malaria. *Cell Host & Microbe*, 17(1):118–129, 2015. ISSN 1931-3128. doi: 10.1016/j.chom.2014.11. 007. URL https://www.sciencedirect.com/ science/article/pii/S1931312814004235.

Galozzi, P., Bindoli, S., Doria, A., and Sfriso, P. The revisited role of interleukin-1 alpha and beta in autoimmune and inflammatory disorders and in comorbidities. *Autoimmunity Reviews*, 20(4):102785, 2021. ISSN 1568-9972. doi: 10.1016/j.autrev.2021.102785.

URL https://www.sciencedirect.com/ science/article/pii/S1568997221000483.

Hargrove, T. Y., Kim, K., de Nazare Correia Soeiro, M., da ´
Silva, C. F., da Gama Jaen Batista, D., Batista, M. M., Yazlovitskaya, E. M., Waterman, M. R., Sulikowski, G. A., and Lepesheva, G. I. Cyp51 structures and structure-based development of novel, pathogen-specific inhibitory scaffolds. International Journal for Parasitology: Drugs and Drug Resistance, 2:178–186, 2012. ISSN 2211-3207. doi: 10.1016/j.ijpddr.2012.06.001. URL https://www.sciencedirect.com/ science/article/pii/S2211320712000206. Including Articles from Keystone Symposium on "Drug Discovery for Protozoan Parasites"; pp. 230–270.

Lepesheva, G. I. and Waterman, M. R. Cyp51the omnipotent p450. Molecular and Cellular Endocrinology, 215(1):165–170, 2004. ISSN 0303-7207. doi: 10.1016/j.mce.2003.11.016. URL https://www.sciencedirect.com/
science/article/pii/S0303720703005148.

Proceedings of the Serono Foundation for the Advancement of Medical Science Workshop on Molecular Steroidogenesis.

Lepesheva, G. I. and Waterman, M. R. Sterol 14αdemethylase cytochrome p450 (cyp51), a p450 in all biological kingdoms. Biochimica et Biophysica Acta (BBA) - General Subjects, 1770(3):467–477, 2007. ISSN 0304-4165. doi: 10.1016/j.bbagen.2006.07.018. URL https://www.sciencedirect.com/ science/article/pii/S0304416506002145. P450.

Harmar, A. Family-b g-protein-coupled receptors. Genome biology, 2(12):REVIEWS3013, 2001. ISSN 1474-7596.

doi: 10.1186/gb-2001-2-12-reviews3013. URL https: //europepmc.org/articles/PMC138994.

Hedstrom, L. Serine protease mechanism and specificity.

Chemical Reviews, 102(12):4501–4524, 2002. doi: 10.1021/cr000033x. URL https://doi.org/10. 1021/cr000033x. PMID: 12475199.

Lundberg, S. M. and Lee, S.-I. A unified approach to interpreting model predictions. In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S.,
and Garnett, R. (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.,
2017. URL https://proceedings.neurips.

cc/paper_files/paper/2017/file/ 8a20a8621978632d76c43dfd28b67767-Paper. pdf.

Hedstrom, L., Perona, J. J., and Rutter, W. J. Converting trypsin to chymotrypsin: residue 172 is a substrate specificity determinant. *Biochemistry*, 33 29:8757–63, 1994.

Jingami, H., Nakanishi, S., and Morikawa, K. Structure of the metabotropic glutamate receptor. Current Opinion in Neurobiology, 13(3):271–278, 2003. ISSN 0959-4388. doi: 10.1016/S0959-4388(03)00067-9.

URL https://www.sciencedirect.com/ science/article/pii/S0959438803000679.

McArthur, A. G., Hegelund, T., Cox, R. L., Stegeman, J. J., Liljenberg, M., Olsson, U., Sundberg, P., and Celander, M. C. Phylogenetic Analysis of the Cytochrome P450 3 (CYP3) Gene Family. *Journal of Molecular Evolution*, 57(2):200– 211, August 2003. doi: 10.1007/s00239-003-2466-x.

URL https://link.springer.com/article/ 10.1007/s00239-003-2466-x.

Kan, H.-I., Chen, I.-Y., Zulfajri, M., and Wang, C. C. Subunit disassembly pathway of human hemoglobin revealing the site-specific role of its cysteine residues. The Journal of Physical Chemistry B, 117(34):9831–9839, 2013. doi: 10.1021/jp402292b. URL https://doi.

org/10.1021/jp402292b. PMID: 23902424.

Menon, A. K., Narasimhan, H., Agarwal, S., and Chawla, S. On the statistical consistency of algorithms for binary classification under class imbalance. In Proceedings of the 30th ICML - Volume 28, ICML'13, pp. III–603–III–611. JMLR.org, 2013.

Qiu, H., Taudien, S., Herlyn, H., Schmitz, J., Zhou, Y.,
Chen, G., Roberto, R., Rocchi, M., Platzer, M., and Wojnowski, L. Cyp3 phylogenomics: evidence for positive selection of cyp3a4 and cyp3a7. *Pharmacogenetics and* Genomics, 18(1):53—66, January 2008. ISSN 1744-6872. doi: 10.1097/fpc.0b013e3282f313f8. URL https:// doi.org/10.1097/FPC.0b013e3282f313f8.

550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 Muhlethaler, T., Gioia, D., Prota, A. E., Sharpe, ¨
M. E., Cavalli, A., and Steinmetz, M. O. Comprehensive analysis of binding sites in tubulin.

Angewandte Chemie International Edition, 60(24):
13331–13342, 2021. doi: 10.1002/anie.202100273. URL https://onlinelibrary.wiley.com/doi/
abs/10.1002/anie.202100273.

Rozemberczki, B., Watson, L., Bayer, P., Yang, H.-T., Kiss, O., Nilsson, S., and Sarkar, R. The shapley value in machine learning. In Raedt, L. D. (ed.), Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI-22, pp. 5572–5579. International Joint Conferences on Artificial Intelligence Organization, 7 2022. doi: 10.24963/ijcai.2022/778. URL https: //doi.org/10.24963/ijcai.2022/778. Survey Track.

Narahari, Y. Game Theory and Mechanism Design. WORLD SCIENTIFIC / INDIAN INST OF SCIENCE, INDIA, 2014. doi: 10.1142/
8902. URL https://www.worldscientific. com/doi/abs/10.1142/8902.

Nitahara, Y., Kishimoto, K., Yabusaki, Y., Gotoh, O.,
Yoshida, Y., Horiuchi, T., and Aoyama, Y. The amino acid residues affecting the activity and azole susceptibility of rat cyp51 (sterol 14-demethylase p450). The Journal of Biochemistry, 129(5):761–768, 2001.

Sakmar, T. P., Menon, S. T., Marin, E. P., and Awad, E. S. Rhodopsin: Insights from recent structural studies. *Annual Review of Biophysics*, 31(Volume 31, 2002):443–484, 2002. ISSN 1936-1238.

doi: 10.1146/annurev.biophys.31.082901.134348. URL https://www.annualreviews.org/
content/journals/10.1146/annurev. biophys.31.082901.134348.

Okada, T., Ernst, O. P., Palczewski, K., and Hofmann, K. P. Activation of rhodopsin: new insights from structural and biochemical studies. Trends in Biochemical Sciences, 26(5):318–324, 2001. ISSN
0968-0004. doi: 10.1016/S0968-0004(01)01799-6. URL https://www.sciencedirect.com/ science/article/pii/S0968000401017996.

Sanderson, T., Bileschi, M. L., Belanger, D., and Colwell, L. J. Proteinfer, deep neural networks for protein functional inference. *eLife*, 12:e80942, feb 2023. ISSN 2050-084X. doi: 10.7554/eLife.80942. URL https://doi.org/10.7554/eLife.80942.

Permyakov, E. A. α-actalbumin, Amazing Calcium-Binding Protein. *Biomolecules*, 10(9):1210, Aug 2020. ISSN 2218-273X. doi: 10.3390/biom10091210. URL http: //dx.doi.org/10.3390/biom10091210.

Shapley, L. S. *17. A Value for n-Person Games*,
pp. 307–318. Princeton University Press, Princeton, 1953. ISBN 9781400881970. doi: doi:10.1515/ 9781400881970-018. URL https://doi.org/10. 1515/9781400881970-018.

Permyakov, E. A. and Berliner, L. J. α-Lactalbumin: structure and function. *FEBS Letters*, 473(3):269–274, 2000. ISSN 0014-5793. doi: 10.1016/S0014-5793(00)01546-5. URL https://www.sciencedirect.com/
science/article/pii/S0014579300015465.

Shionyu, M., Takahashi, K., and Go, M. Variable subunit ¯
contact and cooperativity of hemoglobins. *J. Mol. Evol.*, 53(4-5):416–429, October 2001.

Song, H., Bremer, B. J., Hinds, E. C., Raskutti, G.,
and Romero, P. A. Inferring protein sequencefunction relationships with large-scale positive-unlabeled learning. *Cell Systems*, 12(1):92–101.e8, 2021.

ISSN 2405-4712. doi: 10.1016/j.cels.2020.10.

007. URL https://www.sciencedirect.com/ science/article/pii/S2405471220304142.

Pettersen, E. F., Goddard, T. D., Huang, C. C., Meng, E. C., Couch, G. S., Croll, T. I., Morris, J. H., and Ferrin, T. E. Ucsf chimerax: Structure visualization for researchers, educators, and developers. Protein Science, 30(1):70–82, 2021. doi: 10.1002/pro. 3943. URL https://onlinelibrary.wiley. com/doi/abs/10.1002/pro.3943.

Steinwart, I. and Christmann, A. *Support Vector Machines*.

Springer Publishing Company, Incorporated, 1st edition, 2008. ISBN 0387772413.

Qasba, P. K., Kumar, S., and Brew, D. K. Molecular divergence of lysozymes and α-lactalbumin. Critical Reviews in Biochemistry and Molecular Biology, 32(4):255–306, 1997. doi: 10.3109/10409239709082574. URL https:
//doi.org/10.3109/10409239709082574.

Strushkevich, N., Usanov, S. A., and Park, H.-W. Structural basis of human cyp51 inhibition by antifungal azoles. *Journal of Molecular Biology*, 397(4):1067–1078, 2010. ISSN 0022-2836. doi: 10.1016/j.jmb.2010.01. 075. URL https://www.sciencedirect.com/ science/article/pii/S0022283610001324.

The UniProt Consortium. UniProt: the universal protein knowledgebase in 2021. *Nucleic Acids Research*, 49(D1): D480–D489, 11 2020. ISSN 0305-1048. doi: 10.1093/
nar/gkaa1100. URL https://doi.org/10.1093/ nar/gkaa1100.

Tripathi, S., Hemachandra, N., and Trivedi, P. Interpretable feature subset selection: A Shapley value based approach. In *2020 IEEE BigData*, pp. 5463–5472, 2020. doi: 10. 1109/BigData50022.2020.9378102.

Tripathi, S., Hemachandra, N., and Trivedi, P. Interpretable feature subset selection: A shapley value based approach, 2021. URL https://arxiv.org/abs/ 2001.03956.

Vass, M., Podlewska, S., de Esch, I. J. P., Bojarski, A. J., Leurs, R., Kooistra, A. J., and de Graaf, C. Aminergic gpcr–ligand interactions: A chemical and structural map of receptor mutation data. *Journal of* Medicinal Chemistry, 62(8):3784–3839, 2019. doi: 10.1021/acs.jmedchem.8b00836. URL https://doi. org/10.1021/acs.jmedchem.8b00836. PMID:
30351004.

605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659 Veerapandian, B., Gilliland, G. L., Raag, R., Svensson, A. L., Masui, Y., Hirai, Y., and Poulos, T. L. Functional implications of interleukin-1β based on the three-dimensional structure. Proteins: Structure, Function, and Bioinformatics, 12(1):
10–23, 1992. doi: 10.1002/prot.340120103. URL https://onlinelibrary.wiley.com/doi/
abs/10.1002/prot.340120103.

Yang, Y., Xu, T., Conant, G., Kishino, H., Thorne, J. L.,
and Ji, X. Interlocus gene conversion, natural selection, and paralog homogenization. *Molecular Biology and* Evolution, 40(9):msad198, 09 2023. ISSN 1537-1719.

doi: 10.1093/molbev/msad198. URL https://doi. org/10.1093/molbev/msad198.

Young, H. P. Monotonic solutions of cooperative games.

Int. J. Game Theory, 14(2):65–72, jun 1985. ISSN 00207276. doi: 10.1007/BF01769885. URL https://doi. org/10.1007/BF01769885.

Zhang, Y., Wang, Z., Wang, Y., Jin, W., Zhang, Z., Jin, L.,
Qian, J., and Zheng, L. Cyp3a4 and cyp3a5: the crucial roles in clinical drug metabolism and the significant implications of genetic polymorphisms. *PeerJ*, 12:e18636, 2024.

Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., and Torralba, A. Learning Deep Features for Discriminative Localization . In *2016 IEEE Conference on* Computer Vision and Pattern Recognition (CVPR), pp. 2921–2929, Los Alamitos, CA, USA, June 2016. IEEE Computer Society. doi: 10.1109/CVPR.2016.319. URL https://doi.ieeecomputersociety.org/ 10.1109/CVPR.2016.319.

## A. Data Collection And Code

We discuss the details of the data collection procedure for the datasets used in our computational experiments.

## A.1. Datasets Of 15 Paralog Pairs

We apply our method for identifying amino acid types that distinguish paralogous proteins using the datasets described in Table A2. Only the train set is used for computing *AF S*, while the test set is used for computing classification scores for the linear SVM trained using the train set.

Table A2: The number of sequences in the train and test sets of the protein families considered in computational experiments.

660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696

697

698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714

| 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703   | Family     | Train (Swiss-Prot)   | Test (TrEMBL)   |
|-------------------------------------------------------------------------------------------------------------------------------------------|------------|----------------------|-----------------|
| Lysozyme-like α-Lactalbumin                                                                                                               | 22         | 53                   |                 |
| Lysozyme C                                                                                                                                | 74         | 14                   |                 |
| Trypsin-like Trypsin                                                                                                                      | 66         | 3813                 |                 |
| Chymotrypsin                                                                                                                              | 17         | 281                  |                 |
| Tubulin α                                                                                                                                 | 117        | 190                  |                 |
| β                                                                                                                                         | 191        | 347                  |                 |
| Histone H2A                                                                                                                               | 180        | 16599                |                 |
| H2B                                                                                                                                       | 177        | 7599                 |                 |
| Interleukin-1 α                                                                                                                           | 16         | 12                   |                 |
| β                                                                                                                                         | 25         | 194                  |                 |
| Cytochrome P450 CYP3                                                                                                                      | 32         | 818                  |                 |
| CYP51                                                                                                                                     | 32         | 601                  |                 |
| Globins Myoglobin                                                                                                                         | 107        | 479                  |                 |
| Hemoglobin-α                                                                                                                              | 303        | 525                  |                 |
| Hemoglobin-β                                                                                                                              | 285        | 261                  |                 |
| (GPCR-PEnDB)                                                                                                                              |            |                      |                 |
| Train (80%)                                                                                                                               | Test (20%) |                      |                 |
| GPCR families Rhodopsin-like                                                                                                              | 181        | 45                   |                 |
| ë Lipid receptors                                                                                                                         | 113        | 28                   |                 |
| Peptide receptors                                                                                                                         | 367        | 92                   |                 |
| Aminergic receptors                                                                                                                       | 186        | 47                   |                 |
| Glutamate-like                                                                                                                            | 89         | 23                   |                 |
| Secretin-like                                                                                                                             | 90         | 23                   |                 |

All datasets are taken from publicly available databases (UniProt (The UniProt Consortium, 2020) and GPCR-PEnDB (Begum et al., 2020)). Well-known pairs of paralogous proteins were curated from millions of sequences from UniProt considering the number of sequences and manually reviewed labels available for them. For all datasets except GPCR, we use manually curated Swiss-Prot sequences for training and electronically annotated TrEMBL sequences for testing. These proteins have very specific functions. In contrast, GPCRs are a large and diverse group of transmembrane proteins that mediate cellular responses to extracellular signals. We chose to use an already curated dataset in this case. For each of the GPCR families considered (Table A2), the sequences are randomly split as 80%-train/20%-test. The use of GPCR-PEnDB data is to illustrate the effectiveness of our method with random slicing, which is inevitable when additional curated data are not available. If one or many UniProt entries in a dataset had identical sequences, then only one of them was retained, and the remaining were deleted.

The following queries were used for collecting data from UniProt (The UniProt Consortium, 2020),
- **lysozyme C**: (protein_name:"lysozyme C") AND (fragment:false) NOT (existence:4) NOT
(existence:5) AND (length:[* TO 200]) AND (ec:3.2.1.17) AND (xref:cazy-GH22) AND
(reviewed:true)
- α**-lactalbumin**: (protein_name:"alpha lactalbumin") AND (fragment:false) NOT
(existence:4) NOT (existence:5) AND (length:[* TO 200]) AND (reviewed:true)
- **myoglobin**: (protein_name:"myoglobin") AND (xref:interpro-IPR002335) AND
(fragment:false) NOT (existence:5) NOT (existence:4)
- **hemoglobin-**α: (protein_name:"hemoglobin alpha") AND (xref:interpro-IPR002338) AND
(fragment:false) NOT (existence:5) NOT (existence:4)
- **hemoglobin-**β: (protein_name:"hemoglobin beta") AND (xref:interpro-IPR002337) AND
(fragment:false) NOT (existence:5) NOT (existence:4)
- **trypsin**: (protein_name:trypsin) AND (fragment:false) AND (ec:3.4.21.4) NOT
(existence:5)
- **chymotrypsin**: (protein_name:chymotrypsin) AND (fragment:false) AND (ec:3.4.21.1) NOT
(existence:5)
- **tubulin-**α: (protein_name:"tubulin alpha") AND (family:"tubulin family") AND
(length:[300 TO 600]) AND (fragment:false) NOT (annotation_score:1) NOT (annotation_score:2)
- **tubulin-**β: (protein_name:"tubulin beta") AND (family:"tubulin family") AND
(length:[300 TO 600]) AND (fragment:false) NOT (annotation_score:1) NOT (annotation_score:2)
- **interleukin-1** α (protein_name:"interleukin-1 alpha") AND (family:il-1) AND
(fragment:false) NOT (existence:4) NOT (existence:5) AND (length:[200 TO 400]) NOT (annotation_score:1)
- **interleukin-1** β: (protein_name:"interleukin-1 beta") AND (family:il-1) AND
(fragment:false) NOT (existence:4) NOT (existence:5) AND (length:[200 TO 400]) NOT (annotation_score:1)
- **Histone H2A**: (protein_name:"histone h2a") AND (family:histone) AND (fragment:false)
NOT (existence:4) NOT (existence:5) AND (length:[* TO 200])
- **Histone H2B**: (protein_name:"histone h2b") AND (family:histone) AND (fragment:false)
NOT (existence:4) NOT (existence:5) AND (length:[* TO 200])
- **Cytochrome P450 CYP3**: (family:"Cytochrome P450") AND ((gene:cyp3) OR
(gene:cyp3A*)) AND (fragment:false) NOT (existence:4) NOT (existence:5) NOT
(annotation_score:1)
- **Cytochrome P450 CYP51**: (family:"Cytochrome P450") AND ((gene:cyp51) OR
(gene:cyp51A*) OR (gene:cyp51B*) OR (gene:cyp51C*)) AND (fragment:false) NOT
(existence:4) NOT (existence:5) NOT (annotation_score:1)
The GPCR sequences were collected from the GPCR-PEn database (URL: https://gpcr.utep.edu/) (Begum et al., 2020). Sequence redundancy of the rhodopsin-like family was reduced using CD-hit (Fu et al., 2012) with 30% sequence similarity cutoff.

## A.2. Code

The code to reproduce the computational experiments is available at https://anonymous.4open.science/r/ AFS_AAC_SVM-F3D9. Protein sequences used in the computational experiments along with their UniProt IDs, are provided in the datasets folder as .csv files for each family.

715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769

## B. Sequence And Function Diversity Of Protein Classes Within A Dataset

Paralogous proteins have a common ancestor but have diverged in functionality. Protein functions are an aggregate of descriptors describing protein's activity and influence at various levels. They can be at the molecular level, like binding with specific molecules and catalysing reactions, to the biological process level, like energy metabolism. In B.1, we discuss the diversity of the functions of the proteins considered in our datasets. As paralogs have a common ancestor, high sequence similarity would suggest high evolutionary conservation in the proteins. In B.2, we discuss the extent of sequence diversity in protein classes considered in our datasets.

We see that the dataset of proteins considered in our computational experiments are diverse in their function and sequences.

## B.1. Function Diversity

We have considered paralogous proteins with varying functional differences. We find very subtle differences in the functions of trypsin and chymotrypsin. On the other hand, the function difference is drastic in the case of alpha-lactalbumin and lysozyme c. Trypsin and chymotrypsin are a family of enzymes that break peptide bonds in proteins. The difference in the function of these proteins is fine-grained; trypsins cleave only the peptide bond following a basic amino acid (K and R), while chymotrypsins cleave the peptide bond following a hydrophobic amino acid (F, W, and Y ) (Dodson & Wlodawer, 1998). GPCRs constitute a large and diverse class of cell surface receptor proteins. They trigger intra-cellular pathways in response to external signals. These signals are in the form of small molecules, called ligands. Depending upon the nature of ligands and other 3D structural similarities, GPCRs are grouped into distinct classes. We consider three such classes viz., rhodopsin-like, secretin-like, and glutamate-like. Further, we consider pairwise three subfamilies of rhodopsin-like GPCRs viz., aminergic receptors, lipid receptors, and peptide receptors. Lysozyme C and α-lactalbumin are sequence and structure homologs with mutually exclusive functions and high fold conservation. Based on phylogenetic analysis, they are considered to have diverged from a common ancestor millions of years ago (Qasba et al., 1997). Globins are a superfamily of functionally divergent homologous protein families with a high level of fold conservation. We consider three well-known globin families viz., myoglobin, hemoglobin-α and hemoglobin-β. Myoglobin is a monomer that binds and releases oxygen as per physiological requirements. On the other hand, α and β chains together constitute hemoglobin, a tetramer of composition α2β2 (Dill et al., 2017), that transports oxygen in red blood cells.

Tubulin-α and tubulin-β are similar to the hemoglobin-α and hemoglobin-β pair in that they both share sequence and 3D structural similarities but have subtle functional differences. One copy each of tubulin-α and tubulin-β form a functional dimer. Notably, neither two copies of tubulin-α nor two copies of tubulin-β can form a functional dimer. Tubulin-β has a catalytic activity (GTP hydrolysis) that is absent in tubulin-α. This is one of the several subtle functional differences between tubulin-α and tubulin-β. Interleukin-1 alpha and interleukin-1 beta are both proteins involved in the immune system. They differ from each other in their occurrence within the body (on cell surface or in blood circulation), activation mechanisms, and associated signalling pathways (Galozzi et al., 2021). Cytochrome P450 (abbreviated as CYP) is a family of proteins whose function is clearance of 'foreign' molecules (drugs; also called as xenobiotics) as well as in certain biosynthesis pathways e.g., of steroid hormones. CYP3 and CYP51 are two of the several classes of CYPs; CYP3 metabolizes lipophilic molecules (McArthur et al., 2003) whereas CYP51 is involved in steroid biosynthesis (Hargrove et al., 2012). Hemoglobin-α/hemoglobin-β, histone H2A / histone H2B and tubulin-α/tubulin-β are paralog pairs that together function as heteromers (protein complexes made up of different protein subunits).

## B.2. Sequence Diversity

The dataset of the 15 paralog pairs in our experiments comprises 21 protein families (Table A2). For these families, we compute the within-class sequence similarities (for sequences within a protein family). We also compute the inter-class sequence similarities (between sequences from two different protein families) for each paralog pair. These are shown in 770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 Appendix Figure B4. We use a longest subsequence based similarity score, *lcss*, that is defined in B.2.1. In B.2.2, we see that *lcss* significantly varies across the 21 protein families we are considering as compared to its variation between the two protein sequences of any paralog pair.

B.2.1. LONGEST COMMON SUBSEQUENCE BASED SIMILARITY SCORE (*lcss*)
We compute the longest common subsequence (lcs) based similarity score (*lcss*) between a pair of protein sequences. We define *lcss* between two sequences as the length of their longest common subsequence, lcs, divided by the length of the longest sequence from the two. For a pair of protein sequences, p piq " pp piq 1
, p piq 2
, . . . , p piq L1 q of length L1 and p pjq " pp pjq 1, p pjq 2*, . . . , p* pjq L2 q of length L2, their *lcss* is, lcs based similarity score, *lcss*, is defined as, 825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 Table B3: The median within-class *lcss* between sequences from the respective families. See boxplot in Figure B4.

- Median *lcss* ě 0.6 for 11 of these 12 families and ě 0.8 for 3 families (high level of sequence conservation). - For 7 out of the 15 paralog pairs, the median within-class *lcss* ą 0.5 for both families of a paralogous pair.

- For the remaining 9 protein families, the median within-class *lcss* is less than 0.5. This implies high sequence diversity in this set of families from the remaining families. These are,

| Family      | trypsin            | chymotrypsin   | rhodopsin-like receptor   | glutamate-like receptor   | secretin-like receptor   |
|-------------|--------------------|----------------|---------------------------|---------------------------|--------------------------|
| Median lcss | 0.47               | 0.45           | 0.34                      | 0.35                      | 0.36                     |
| Family      | aminergic receptor | lipid receptor | peptide receptor          | cytochrome P450 CYP51     |                          |
| Median lcss | 0.39               | 0.37           | 0.37                      | 0.47                      |                          |

$$l c s s(\mathbf{p}^{(i)},\mathbf{p}^{(j)})={\frac{l c s(\mathbf{p}^{(i)},\mathbf{p}^{(j)})}{\operatorname*{max}(L_{1},L_{2})}}\in[0,1]$$

lcsspp piq, p pjqq " 1 if and only if p piq " p pjq, i.e., sequences are identical. Whereas *lcss*pp piq, p pjqq " 0 if and only if p piq x ‰ p pjq y , @*x, y*, i.e., there are no amino acids common to both the sequences.

B.2.2. WITHIN-CLASS AND INTER-CLASS *lcss* FOR THE 15 PARALOG PAIRS
Within-class lcss: *lcss*pp piq, p pjqq are computed with p piq, p pjqfrom the same protein family. These are shown in *blue* and magenta in Figure B4 (with box-plots) for each of 21 protein families in the 15 paralog pairs.

- 12 of 21 protein families have median within-class *lcss* greater than 0.5. This implies less sequence diversity in this set of families from the remaining families. These are,
- For 7 out of the 15 paralog pairs, the median within-class *lcss* ă 0.5 for both families of a paralogous pair.

| Family      | α-lactalbumin   | lysozyme C      | myoglobin       | hemoglobin-α   | hemoglobin-β   | tubulin-α            |
|-------------|-----------------|-----------------|-----------------|----------------|----------------|----------------------|
| Median lcss | 0.6             | 0.59            | 0.81            | 0.63           | 0.67           | 0.83                 |
| Family      | tubulin-β       | interleukin-1 α | interleukin-1 β | histone H2A    | histone H2B    | cytochrome P450 CYP3 |
| Median lcss | 0.82            | 0.72            | 0.66            | 0.65           | 0.68           | 0.7                  |

$$l c s(\mathbf{p}^{(i)},\mathbf{p}^{(j)})=\operatorname*{max}_{\mathbf{q}}k$$
$$y_{1}<y_{2}<\ldots<y_{k}$$
s.t. ${\bf q}=(q_{1},q_{2},\ldots,q_{k})$  $(q_{1}=p_{x_{1}}^{(i)}=p_{y_{1}}^{(j)},q_{2}=p_{x_{2}}^{(i)}=p_{y_{2}}^{(j)},\ldots,q_{k}=p_{x_{k}}^{(i)}=p_{y_{k}}^{(j)})$  $x_{1}<x_{2}<\ldots<x_{k}$
Table B4: The median within-class *lcss* between sequences from the respective families. See boxplot in Figure B4.

- For the paralog pair Cytochrome P450 CYP3 vs CYP51, the median sequence similarity for CYP3 is greater than 0.5, while for CYP51, it is less than 0.5.

Inter-class lcss: *lcss*pp piq, p pjqq are computed with p piq, p pjqrespectively from two protein families that are paralog pairs.

These are shown in *cyan* in Figure B4 (with box-plots) for each of the 15 paralog pairs.

- The median inter-class *lcss* is less than 0.5 for all paralog pairs. This implies sequences of the proteins across the classes are not very similar.

Distinguishing paralog pairs based on within-class and inter-class *lcss*: If we analyse the box plots in Figure B4 - two paralog pair proteins can be considered to be distinguishable based on sequence similarity if the upper-whisker of inter-class lcss is lower than the lower-whiskers of the respective within-class *lcss* scores.

- Apart from paralog pairs, tubulin-α vs tubulin-β (Figure B4c) and interleukin-1 α vs interleukin-1 β (Figure B4d), no other paralog pair is distinguishable based on sequence similarity.

- For Trypsin vs Chymotrypsin and the 6 GPCR pairs (Figures B4b and B4j to B4o), the median inter-class *lcss* scores are close to the within-class *lcss* scores making them indistinguishable based on sequence similarity.

880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934

935 936 937 luence similarity
 see see se 938 939 940 s

Das 941 942 943 944 945 946 947 quence similarity 948 949 950 951 952 953 954 955 956 957 quence similarit
 
958 959 960 961 962 963 964 95 966 967 968 Sequence similarity
 September 1999 969 970 971 972 973 974 975 976 977 978 quence similarity
 1955 - 1955 979 980 981
.

982 983 984 985 1.0
:
0.9 0. B
0.2 0.1 loloalyze lalba lyzc
(a) Lysozyme C
vs α -Lactalbumin 1.1.0 0.9 0.8 s

Ibas a 2 0.1 ilaniib ila illb
(d) Interleukin-1 α vs Interleukin-1 β 1.0 0.9 0.8 s

Cas 0.2 0.1 Ha Mb/libs Ma
(g) Myoglobin vs Hemoglobin- α 1.1.0 a a 0.8 0.2 0.1 G
GlwSec
(j) Secretin-like vs Glutamate-like 1.0 0, 9 0.8 0.2 0.1 Ami/Up Ami Lp
(m) Aminergic vs Lipid receptors Identifying key amino acid types that distinguish paralogous proteins 1.1.0 0.9 0.9 0.8 0.8 Sequence similarity
 September 1999 s.
Hell nce similar
 1955 births
List of the Briti a on os of 3 0.2 0.2 0.1 0.1 Chym Tryp/Chym Tryp apha beta aphabeta
(b) Trypsin vs Chymotrypsin
(c) Tubulin- α vs Tubulin- β 1.0 1.0 0.9 0.9 larity
 1999 by 0.00 similarity
 1999 - 1999 e.
Illuls s
Su 89 an e.
Hand as as a 2 0.2 0.1 0.1 h2b h2a/h2b h2a cyp3 cyp51 cyb.lcyb21
(e) Histone H2A vs Histone H2B
(f) Cytochrome P450 CYP3 vs CYP51 1.0 1.0 0.9 0, 9 0.8
, 0.8 Sequence similarity
 Separate  Sep e
Assess e.
Illuls 89
.0.4 3.
335 0.2 0.2 0.1 0.1 Hoards Hbb Mb/Hbb Hbb Hol
(h) Hemoglobin- α vs Hemoglobin- β
(i) Myoglobin vs Hemoglobin- β 1.0 1.0 0.9 0.9 0.8 0.00 City similarity
 16 - 16 similarity
 1855 Sequence s
1999 for 89 e.
Lang 0, 0, 3 0.2 0.2 0.1 a 1 Cu Rha/Glu Ro
(k) Rhodopsin-like vs Glutamate-like
(l) Rhodopsin-like vs Secretin-like 1.0 1.0 0, 9 0, 9 0.00 a a o. 7
Unex 0.7
HER
e
 Selfs
 Selfs es Sequence
 September 1999  September 1999  September 1999  September 1999
 Sept a 0.5
3.0 35 se 0.3 0.2 0.2 a.1 0.1 Ami AmiPep UpPep Pep Up Pep
(n) Aminergic vs Peptide receptors
(o) Lipid vs Peptide receptors
986 987 988 989

## C. The Svea Algorithm For Afs

Algorithm 1 ϕi Monte-carlo approximation algorithm as suggested in (Tripathi et al., 2020; 2021)

## D. Svm Training For Af S **Partition**

Input: Feature set N " t1, 2*, . . . ,* 20u, Number of sample permutations *samP erm*, Datasets pDP , DQq, Set of coalitions Sam co set **" rpqs**
Initialise: v**ppqq "** 0, ϕˆi:" 0@i P N
Append N to Sam co set. for s " 1, 2*, . . . , samP erm* do Take π P *P ermSet*pNq with probability 1 20! .

for i " 1, 2*, . . . ,* 20 do Compute *P red*ipπ**q " t**πp1q, πp2q*, . . . π*pk ´ 1q|i " πpkqu if *P red*ipπq not in Sam co set **then**
Compute vp*P red*ipπ**qq "** 1 ´ tr erp*P red*ipπqq. Append P redipπq to Sam co set.

end if if *P red*ipπq Y i not in Sam co set **then**
Compute vp*P red*ipπq Y ti**uq "** 1 ´ tr erp*P red*ipπ**q Y t**iuq. Append *P red*ipπ**q Y t**iu to Sam co set.

end if ϕˆi " ϕˆi ` vp*P red*ipπq Y ti**uq ´** vp*P red*ipπqqq end for end for ϕˆi "ϕˆi samP erm 
, @i P N
990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 We provide details for the linear SVM classifier discussed in Section 2.3. We use 5-fold cross-validation to tune the SVM regularisation hyperparameter C from t0.1, 1, 10, 100, 1000u that gives the best average classification score for the 5 folds.

C is inversely proportional to the strength of regularisation. In general, we find that there is an imbalance in the number of sequences that we find for the two paralogous proteins, i.e. say nP ąą nQ. It is known that accuracy is not a well-suited performance measure of the classifier in class imbalance settings. Therefore, we use the arithmetic mean of sensitivity and specificity (AM) to measure the performance of the classifier (Brodersen et al., 2010). Further, we use a class-balanced version of hinge loss for training the SVM as suggested in (Menon et al., 2013) for statistical consistency with the AM score. Appendix Table E5 reports the train and test scores of the trained linear SVM with AAC and *AF S* features, respectively, on the protein family datasets (See Appendix Table A2) considered in our computational experiments.

1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079 1080 1081 1082 1083 1084 1085 1086 1087 1088 1089 1090 1091 1092 1093 1094 1095 1096 1097 1098 1099 Figure E5: The sizes of the *AF S* for the 15 datasets.

(a) Lysozyme C vs α-Lactalbumin (b) Trypsin vs Chymotrypsin (c) Tubulin-α vs Tubulin-β
(d) Interleukin-1 α vs Interleukin-1 β (e) Histone H2A vs Histone H2B (f) Cytochrome P450 CYP3 vs CYP51
(g) Aminergic vs Lipid receptors (h) Aminergic vs Peptide receptors (i) Lipid vs Peptide receptors 5 6 7 8 9 10 1 2 3 4 Numb er of data se ts Size of *AF S*