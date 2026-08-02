# Ascension: Autoencoder-Based Latent Space Class Expansion For Time Series Data Augmentation

000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054

## Abstract

Achieving effective data augmentation (DA) in time series classification is challenging due to the diverse nature of temporal data. While stateof-the-art generative models for DA - based on GANs, diffusion models, or Variational Autoencoders (VAEs) - demonstrate potential, they often fail to deliver consistent improvements across various datasets and application domains (e.g., ECG, power consumption, vibration sensor data), as confirmed in this study. To address this limitation, we introduce ASCENSION (Autoencoder-based latent space class expa**nsion**), a novel generative approach that harnesses the probabilistic structure of the VAE's latent space alongside an innovative controlled and progressive class expansion mechanism. It promotes compact intra-class representations while maximizing inter-class separability, thereby reducing the likelihood of class overlap during latent space exploration. We evaluate AS- CENSION on 102 datasets from the UCR benchmark and compare it against six state-of-the-art DA methods. Empirical results show that AS- CENSION improves average classification accuracy by approximately 1%, whereas the strongest competing method leads to an average accuracy change of −0.3%. ASCENSION achieves a nonnegative improvement in classifier performance for 66.2% of the 102 datasets - a 16.4% improvement over the previous best method. These results establish ASCENSION as the first DA method that can be reliably applied in real-world scenarios where prior knowledge of method suitability is uncertain. Our study further explores the key factors driving its superior performance.

## 1. Introduction

Time series classification (TSC) is challenging due to temporal dependencies, non-stationarity, and limited labeled data. Real-world constraints, such as high collection costs and privacy regulations, further restrict training set sizes and impact model accuracy. Data augmentation (DA) helps 1 Anonymous Authors1 mitigate these constraints by generating synthetic samples that increase both the quantity and diversity of training data. Formally, given a labeled dataset {x y i} for each class y ∈ {1, 2*, . . . , Y* }, DA aims to create additional synthetic samples that preserve class semantics while broadening coverage of the data distribution. DA methods generally fall into two categories: *traditional* and *generative* (Iglesias et al., 2023b). Traditional methods such as AutoAugment (Cubuk et al., 2019) and Fast AutoAugment (Lim et al., 2019) apply predefined transformations (e.g., jittering, window slicing, scaling). While effective in image classification, their application to time series is often hindered by the risk of disrupting crucial temporal patterns, such as periodicity or phase alignment. Generative DA methods, based on GANs, diffusion models, and VAEs (Cheung & Yeung, 2020), bypass such handcrafted transformations by learning to model the underlying data distribution. GAN-based methods, such as TimeGAN (Zhang et al., 2022), TTS-GAN, LatentAugment (Tronchin et al., 2023), can produce high-quality, rapidly sampled time-series, but may exhibit limited diversity (Xiao et al.). Diffusion models generate rich, varied samples at the cost of high computational overhead (Feng et al., 2024). VAE- based methods often strike a promising balance, providing relatively fast sampling within a structured latent space, but offer limited means to *expand* beyond the distribution already seen in the training data. To our knowledge, no state-of-the-art DA method for timeseries classification enables progressive (iterative) and meaningful class boundary expansion during synthetic data generation. This limitation, discussed further in Appendix A and Figure 6, becomes critical when training and operational data distributions diverge (i.e., distribution discrepancy ratio), often due to factors like sensor drift, unseen conditions, or temporal shifts. To bridge this gap, we propose ASCEN-
SION, a novel VAE-based DA framework that preserves fast sampling and flexible latent representations while enabling controllable class boundary extrapolation. Unlike conventional generative DA methods that strictly replicate the training set's latent distribution, ASCENSION features a tunable mechanism for exploring underrepresented or unseen regions without intruding into overlapping or ambiguous 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109

➠ ➠
Clustering loss
●

- ●●

- ●

●
●

●

➠ ➠●
- ●●

- ●
●
●

●

●●
- ●●● ●

●●
●
- ●●
●

●

●●

- ●●

- ●
●●

●
- ●●
●

●

➠ 
➠ 
vs.

Traditional Gen. DA
✷ ➠ ➠
❀
✷
❀
●
- ●●●
 ●
●

●
●
●●
- ●●

- ●

●●
●
- ●●

●✷
❀
●
✷ ➠ ➠
Corresponding (synthetic) TS
✷●
- ●●
- ●
●

●
●

●●
- ●●
- ●
●●

●

- ●●●
- ✷
ASCENSION
class areas. Specifically, it leverages the probabilistic structure of the VAE's latent space through a multi-component representation per class. By adjusting these components, ASCENSION enables controlled and progressive expansion of class probability densities and boundaries. Additionally, ASCENSION enforces structural constraints that ensure intra-class compactness while maintaining inter-class separation, preserving class consistency and preventing harmful overlap. This leads to richer, more representative synthetic time-series data, enhancing diversity and ultimately improving classification performance. To highlight ASCENSION's originality compared to existing generative DA methods, Figure 1 illustrates its latent space dynamics versus traditional generative DA methods. Our key contributions are:
1. **Novel VAE-based DA Method:** ASCENSION pioneers a controllable and progressive boundaryexpansion mechanism, unlocking richer generative spaces and significantly enhancing applicability against distribution discrepancies, a crucial challenge in realworld TSC applications; 2. **Unparalleled Benchmarking & Performance Gains:**
We rigorously evaluate ASCENSION's impact on classification performance across a vast and diverse set of time-series datasets, outperforming both traditional (FAA) and generative methods (LatentAugment, TTS- GAN, Time-DDPM, VaDE, and MODALS);
3. **Fundamental Data-driven Insights:** We analyze how different time-series properties influence DA performance, showing that ASCENSION's controlled extrapolation can better align training and operational distributions.

The rest of the paper is structured as follows. Section 2 discusses related DA methods, covering both traditional and generative methods. Section 3 presents the ASCENSION framework. Section 4 then provides an extensive empirical evaluation and comparative analysis. Finally, we conclude with key takeaways and future directions.

## 2. Related Work

DA for time series falls into traditional and generative methods. Traditional methods like window slicing, jittering, and scaling (Iglesias et al., 2023a) apply transformations from computer vision but often distort temporal and semantic integrity. Automated methods such as AutoAugment (AA) (Cubuk et al., 2019) optimize transformations via reinforcement learning, while Fast AutoAugment (FAA) (Lim et al., 2019) improves efficiency with density matching. Further refinements, including RandAugment (Cubuk et al., 2020), Deep AutoAugment (Zheng et al., 2022), and Trivial Augment (Muller & Hutter ¨ , 2021), streamline augmentation strategies. However, these methods still rely on predefined transformations, limiting adaptability to complex time series. Generative DA methods, leveraging models like GANs, VAEs, and diffusion models, offer more flexible augmentation by learning probabilistic representations of time series distributions. TimeGAN (Zhang et al., 2022), TS- GAN(Yang et al., 2023b), and TTS-GAN(Li et al., 2022) adapt GAN architectures for time series, capturing longrange dependencies and improving data quality. However, GANs suffer from training instability, sensitivity to hyperparameters, and mode collapse. More recent advances in diffusion models, such as ASE-DDPM (Liu et al., 2024), DiffRUL (Wang et al., 2024), and Time-DDPM (Solis-
Martin et al., 2023), have demonstrated improved stability but struggle with long-range dependencies and slow inference. VAEs, by contrast, provide a more structured latent space, facilitating better sample diversity control. MODALS (Cheung & Yeung, 2020) was the first VAE-based approach to explore class boundary expansion, though without a controllable mechanism. VAE-LSTM (Dang et al., 2024) and VaDE (Jiang et al., 2016) have also been proposed for time series augmentation but do not explicitly model class expansion, a gap addressed by ASCENSION. For a more detailed discussion on "Related Work", refer to Appendix A, which also highlights the state-of-the-art methods benchmarked in this study, as summarized in Figure 6.

## 3. Ascension Framework

110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 Unlike traditional generative DA methods that apply inputspace transformations (e.g., random warping or scaling), which can lead to sample degradation or unintended class confusion, ASCENSION explicitly models classconditional densities and incorporates a risk-aware exploration mechanism, regulated by a scaling factor α, to mitigate class overlap and ensure high-quality augmentations.

ASCENSION is designed to achieve a delicate balance between three objectives: (1) precise VAE-based density modeling; (2) risk-aware exploration to prevent degenerate samples, and (3) controlled class distribution expansion, enabling diverse and useful synthetic data for time series classification. Sections 3.1 and 3.2 detail how ASCENSION integrates VAE training and clustering constraints respectively. Section 3.3 details the proposed iterative class expansion mechanism expanding these latent distributions iteratively to produce synthetic data.

## 3.1. Vae Training & Latent Space

ASCENSION begins with a VAE that models data X in a probabilistic latent space. We optimize the Evidence Lower Bound (ELBO),

$$\mathcal{L}(\theta,\phi)=\mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}\left[\log p_{\theta}(\mathbf{x}|\mathbf{z})\right]-D_{KL}\Big{(}q_{\phi}(\mathbf{z}|\mathbf{x})\parallel p(\mathbf{z})\Big{)},\tag{1}$$

where qϕ(z∣x) is the approximate posterior, pθ(x∣z) is the likelihood, and DKL is the Kullback-Leibler divergence from the prior p(z). To capture class-specific nuances, AS- CENSION estimates each class's distribution in the latent space, enabling controlled sampling and mitigating ambiguity among overlapping regions.

## 3.2. Clustering Constraints

To enhance class separability, ASCENSION incorporates a clustering loss:

$$\mathcal{L}_{\text{cluster}}=\sum_{i=1}^{N}\sum_{j=1}^{N}\delta_{y_{i},y_{j}}\;d(\mathbf{z}_{i},\mathbf{z}_{j}),\tag{2}$$

where δyi,yj = 1 if samples i and j share the same class, and 0 otherwise; d(zi, zj ) is the distance metric (cosine similarity). This loss function reinforces *intra-class compactness* while maximizing *inter-class separability*, ensuring wellstructured latent clusters for generating more consistent and reliable synthetic samples.

## 3.3. Latent Class Expansion

ASCENSION iteratively expands each class's latent distribution following a five-step process:

## 1. **Train The Vae With Clustering**:

LVAE = Lrecon + LKL + Lcluster + Lclass, optimized over the current training set; 2. **Sample Latent Points**: For each class y, sample new points from a Gaussian mixture centered on classspecific means:

$$\frac{1}{K_{y}}\sum_{k=1}^{K_{y}}\mathcal{N}\big(\mu_{y,k},\alpha\,\Sigma_{y,k}\big),$$
(3)  $\frac{1}{2}$

N (µy,k, α Σy,k), (3)
where α scales the covariance to systematically *expand* the class boundaries; 3. **Label Assignment via Posterior Probability**: If sampled points lie in overlap regions, assign labels by maximizing the posterior probability to ensure risk-aware augmentation and avoid misclassification; 4. **Decode and Augment**: Decode latent points into time series, then add them (with labels) to the training dataset, enriching its variety without jeopardizing class integrity; 5. **Retrain Iteratively**: Use the augmented dataset to retrain the model from scratch, refining its parameters and further exploring latent regions over multiple iterations.

This five-step process is formalized in Algorithm 1. Empirical results (Section 4 and Appendix B) show that values of α slightly above 1 effectively boost diversity without sacrificing class consistency.

## 4. Experiments 4.1. Experimental Setup

165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 Train/Test datasets: Experiments were conducted using the UCR Time Series Archive, which comprises 120 univariate time series datasets from various applications and domains, including sensors, ECG, Motion, Spectro, etc. (a complete list of the dataset types is provided in Table 4). To guarantee an adequate amount of time series data in the datasets to train the studied models, we excluded datasets with insufficient data, retaining 102 datasets from the initial set of 120. Classification models: Classifiers selected for our experiments were chosen based on the findings of (Fawaz, 2020), which reports that ResNet-50 and Fully Connected Networks (FCN) are the two most effective classifiers (out of 9 evaluated for the UCR datasets). We use the architectures from (Koonce & Koonce, 2021) and (Scabini & Bruno, 2023) for these two classifiers. Benchmarked DA methods: ASCENSION is compared to six state-of-the-art DA methods, including one traditional (FAA) and five generative methods (TTS-GAN, LA, Time- DDPM, VaDE and MODALS). More details on these methods can be found in Appendix A. FAA was selected due to its comparable performance with other traditional DA
methods (incl., RA and DAA), while VaDE and MODALS
were chosen because of their architectural similarity to AS-
CENSION. TTS-GAN, Time-DDPM and LA were included as the most recent generative DA methods with publicly

| Algorithm 1 Augmentation Loop with distinct classes 1: Input: Original time series data X = {x1,x2, . . . ,xn} with class labels Y = {y1, y2, . . . , yn} 2: Output: Augmented training dataset Xaug, Yaug 3: Initialization: 4: Xaug ← X 5: Yaug ← Y 6: while augmentation desired do 7: Train VAE: 8: LVAE = Lrecon + LKL + Lcluster + Lclass 9: θ ∗ , ϕ∗ ← arg minθ,ϕ LVAE using X, Y 10: Build combination of Gaussian: Ky 11: Let dy = 1 ∑ k=1 N(µy,k, αΣy,k) to Z for each class y Ky 12: Sample Latent Points: 13: for each class y do ′y ′y ′y 14: Z y new = {z 1 , z 2 , . . . , z m} ∼ dy ′y 15: for each z i ∈ Z y new do ′y ′ 16: If z has higher probability of being in class y i ′ 17: Assign label y 18: end for 19: end for 20: Decode Latent Points: 21: for each class y do 22: X y syn = {x ′y 1 ,x ′y 2 , . . . ,x ′y m} where x ′y i = f θ (z ∗ ′y i ), ∀z ′y i ∈ Z y new 23: end for 24: Update Training Set: 25: Xaug ← Xaug ∪ (⋃y X y syn) y 26: Yaug ← Yaug ∪ (⋃y {y} × X syn) 27: end while   |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## 4.2.2. Performance Comparison Analysis

Several findings can be drawn from Table 1. First, FAA demonstrates moderate mean accuracy improvements of 6.5% (ResNet) and 7.5% (FCN), but lacks consistency, with improvements observed on only 28/102 datasets (ResNet) and 13/102 datasets (FCN). Similarly, LA shows limited impact, improving accuracy on 23 datasets (ResNet) and 38 datasets (FCN), with mean improvements of 3.7% and 2.1%, respectively. On the other hand, ASCENSION achieves substantial gains, improving classification accuracy on 56/102 datasets (ResNet) and 50/102 datasets (FCN), with mean accuracy increases of 4.0% and 3.0%, respectively. Moreover, ASCENSION consistently minimizes performance deterioration, with only 30 datasets worsened for ResNet and 39 for FCN, compared to 67 and 85 datasets for FAA, respectively.

Compared to Time-DDPM and VaDE, ASCENSION
achieves a balanced trade-off between maximizing the number of datasets improved and minimizing those with worsened performance. Time-DDPM, while achieving the highest mean accuracy improvement (17.8% for ResNet and 15.8% for FCN), suffers from significant performance deterioration on 62/102 datasets (ResNet) and 58/102 datasets (FCN), indicating overfitting to a subset of datasets. In contrast, ASCENSION's consistent performance across both available code (cf., Figure 6). Benchmarking MODALS on the UCR datasets is not feasible, as its publicly available code from 2020 is no longer functional, and the authors have confirmed they do not intend to fix it. Consequently, we evaluate ASCENSION against MODALS using the HAR dataset originally used by (Cheung & Yeung, 2020).

## 4.2. Experimental Results 4.2.1. Performance Evaluation Metrics

Accuracy: The ratio of correct predictions to the total number of predictions is employed as the evaluation metric. Preand post-augmentation classification results are gathered for each combination of the benchmarked techniques, selected classifiers, and UCR datasets. Table 1 groups the results in three categories: *(i) Augmented:* reflects the number of datasets on which the classification accuracy post-augmentation is better than pre-augmentation; (ii) Unchanged: refers to the datasets that do not show a significant impact (±10−4%) of the augmentation on classifier performance, *(iii) Worsened:* aggregates the datasets where the augmentation of the train set degrades the accuracy of the classifier. Under each category we report the number of datasets and the mean classification accuracy postaugmentation for the different configurations (classifiers, DA methods). For an exhaustive list of the pre- and postaugmentation classification results, refer to Appendix B.1.

FC

N

Table 2: Acc. comparison on HAR dataset used by (Cheung & Yeung, 2020) to assess MODALS

| DA method   | Augmented   | Unchanged   | Worsened   | ↑Total      |      |            |      |        |
|-------------|-------------|-------------|------------|-------------|------|------------|------|--------|
| ↑Nbdatasets | ↑Acc        | Nbdatasets  | Acc        | ↓Nbdatasets | ↑Acc | Nbdatasets | ↑Acc |        |
| FAA         | 28          | 6.5%        | 7          | 0%          | 67   | -9.1%      | 102  | -4.2%  |
| LA          | 23          | 3.7%        | 12         | 0%          | 67   | -3.3%      | 102  | -1.3%  |
| TTS-GAN     | 41          | 2.2%        | 10         | 0%          | 51   | -8.9%      | 102  | -3.6%  |
| Time-DDPM   | 38          | 17.8%       | 2          | 0%          | 62   | -22.2%     | 102  | -6.8%  |
| VaDE        | 57          | 3.1%        | 8          | 0%          | 37   | -7.7%      | 102  | -1.1%  |
| ASCENSION   | 56          | 4.0%        | 16         | 0%          | 30   | -1.7%      | 102  | 1.7%   |
| FAA         | 13          | 7.5%        | 4          | 0%          | 85   | -15.8%     | 102  | -12.2% |
| LA          | 38          | 2.1%        | 18         | 0%          | 46   | -2.3%      | 102  | -0.3%  |
| TTS-GAN     | 31          | 2.2%        | 13         | 0%          | 58   | -7.5%      | 102  | -3.6%  |
| Time-DDPM   | 43          | 15.8%       | 1          | 0%          | 58   | -24.0%     | 102  | -7.0%  |
| VaDE        | 35          | 2.8%        | 16         | 0%          | 51   | -6.7%      | 102  | -2.4%  |
| ASCENSION   | 50          | 3.0%        | 13         | 0%          | 39   | -1.4%      | 102  | 1.0%   |

ResNet and FCN backbones demonstrates its scalability and versatility for enhancing classification tasks.

| Method              | Accuracy (%)   |
|---------------------|----------------|
| ASCENSIONResNet-Emb | 93.42          |
| MODALS              | 91.87          |
| No Augmentation     | 88.64          |

In Table 2, we compare to MODALS on the HAR dataset, ASCENSION further enhances performance. While MODALS improves the baseline classification (without augmentation) by 3.23%, ASCENSION increases this improvement by +4.78%, further advancing accuracy beyond the baseline.

| t NeResFC   |
|-------------|

## 4.2.3. Embedded Classifier Performance

The ASCENSION framework supports various classifier architectures due to its modularity. Leveraging this flexibility, we also assess ASCENSION's performance with a modified classifier setup. In Table 3, we present the evaluation results for: (i) ASCENSION's standard embedded classifier, denoted as ASCENSIONEmbCl., and (ii) a hybrid approach combining ASCENSION's embedded classifier with the studied classifiers, referred to as ASCENSIONc-EmbCl., where c ∈ ResNet, FCN in our experiments. The augmentation effect is quantified as the difference between: (i) The highest baseline accuracy achieved by either the VAE's 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 classifier or the standalone classifier c, and (ii) the highest accuracy recorded for ASCENSIONEmbCl. or classifier c, computed as follows:

  \begin{tabular}{l l} Acc${}_{\text{ASCENSION}_{\text{c-EnbCL}}}$ & = & max(Acc${}_{\text{ASCENSION}_{\text{EnbCL}}}$, Acc${}_{\text{c}}$) \\  & & - max(Acc${}_{\text{Baseline}}$, Acc${}_{\text{VAE}}$) \\ \end{tabular}  
$$(4)^{\frac{1}{2}}$$
Table 3 shows that ASCENSIONResNet-Emb achieves the highest accuracy gain (3.7% on 76 datasets) but also has the largest accuracy drop (-5.7% on 14 datasets).

ASCENSIONEmbCl. offers a more stable performance
(1.9% improvement) with minimal degradation (-1.6%).

ASCENSIONFCN-Emb provides moderate gains (2.9%) with a balanced trade-off. Overall, a more complex architecture such as ResNet is likely to maximize improvement but introduces variability, while FCN and the standard classifier ensure more stable performance.

## 4.2.4. Hyperparameters Sensitivity Analysis

A key feature of ASCENSION is its controllable progressive expansion mechanism for exploring the latent space. Adjusting the scaling factor parameter α - which influences how distributions are flattened, see section 3.1 - and determining the number of iterations are essential for optimizing the method's effectiveness. These two parameters must be carefully balanced to maintain sufficient separation between distributions while allowing for adequate exploration.

Analysis methodology: We conducted a study that varies α (from 1 to 5) and the number of iterations (from 1 to 9) to Table 1: Results of our empirical benchmark study on the 102 UCR datasets. The table summarizes the number of datasets with improvement (Augmented), no change (Unchanged), and deterioration (Worsened) in classification accuracy for each DA method. The mean accuracy change (Acc) is provided for each category. An upward arrow (↑) indicates that higher values are preferable, while a downward arrow (↓) signifies that lower values are better. Bold values denote the **best**
performance, and underlined values indicate the second best. ASCENSION improves the classification accuracy for the highest number of datasets and produces the fewest cases of performance reduction, demonstrating its effectiveness in enhancing classification accuracy across the datasets.

275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 Table 3: The table summarizes the number of datasets with improvement (Augmented), no change (Unchanged), and deterioration (Worsened) in classification accuracy for each inherent classifier architecture. The mean accuracy change
(Acc) is provided for each category. An upward arrow (↑) indicates that higher values are preferable, while a downward arrow (↓) signifies that lower values are better. Bold values denote the **best performance**.

| Embedded Classifier   | Augmented   | Unchanged   | Worsened   | ↑Total      |      |            |      |      |
|-----------------------|-------------|-------------|------------|-------------|------|------------|------|------|
| ↑Nbdatasets           | ↑Acc        | Nbdatasets  | Acc        | ↓Nbdatasets | ↑Acc | Nbdatasets | ↑Acc |      |
| ASCENSIONEmb.         | 65          | 1.9%        | 24         | 0%          | 24   | -1.6%      | 102  | 0.8% |
| ASCENSIONResNet-Emb   | 76          | 3.7%        | 12         | 0%          | 14   | -5.7%      | 102  | 2.1% |
| ASCENSIONFCN-Emb.     | 60          | 2.9%        | 28         | 0%          | 14   | -1.7%      | 102  | 1.2% |

ASCENSIONEmb. ASCENSIONFCN-Emb. ASCENSIONResNet-Emb.
assess their impact on accuracy improvement and determine whether convergence occurs.

Results: Figure 2 presents the results for ASCENSIONEmbCl., ASCENSIONResNet-EmbCl., and ASCENSIONFCN-EmbCl. using the Ham dataset from the UCR archive (additional examples can be found in Appendix C). The augmentation process remains relatively stable even with high α values, supporting our hypothesis that the distribution borders reduce the sensitivity of ASCENSION to changes in α. Appendix C offers similar analyses across various UCR datasets, showing that increasing α can enhance boundary exploration but may reduce performance if α is too large. Based on our experiments, selecting α in the range [1, 3] provides a good balance.

## 4.2.5. Operational Efficiency Analysis

Section 4.2 has empirically evidenced that ASCENSION generally outperforms traditional and generative state-ofthe-art DA methods for TSC across most datasets. However, a substantial proportion of datasets (30% to 50%) do not exhibit improved classification performance, and in some cases, performance even deteriorates (see the Unchanged and Worsened columns in Table1). A comprehensive list of these datasets can be found in Appendix B.1. To address this, we propose an analysis to determine which types of data - *characterized by their specific features* - benefit the most from augmentation and which require minimal or no augmentation.

Feature extraction: We use the CATCH22 time series feature set introduced by (Lubba et al., 2019) to characterize the datasets (comprising 22 features in total), adding the ratio of train/test split and the distribution discrepancy ratio between train and test (cf., Appendix E.1). A description of these 24 features (F1-F24) is provided in Appendix F. Analysis methodology: By averaging the features of the time series in each dataset, we identify the datasets that are most and least amenable to benefit from augmentation. Subsequently, we analyze the impact of augmentation on the classification performance of these datasets to determine the most influential features. To measure feature importance, we employ a random forest model with a high number of estimators with low depth to the mean of F1-F24 to predict augmentation for the benchmarked DA methods. Results: Figure 3 shows that each method is strongly tied to specific features such as FAA to F10 (degree of periodic patterns within the dataset), TTS-GAN to F7 which is related to rapid fluctuation in the time series. Moreover, features F23 and F24 (respectively representing the train/test ratio 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 of data and discrepancy in distance between the Train and Test set distributions, cf. Appendix E.1) are tied to methods such as LA and ASCENSION.

To analyze the impact of increasing train-test discrepancy ratios on classification performance, Figure 4 presents the cumulative performance improvement (%) as a function of F24 (see Appendix E.1). The 102 UCR datasets are arranged in ascending order of discrepancy. While other DA methods experience performance degradation as discrepancy increases, ASCENSION sustains positive performance and even exhibits a slight improvement.

## 4.2.6. Qualitative Study On The Risk Of Extrapolation

To qualitatively assess our extrapolation process, we introduce a *class assignment confidence* measure for each generated latent sample set Z. Specifically, we sample a class y from {1, 2*, . . . , Y* } and define its confidence as:

$$\mathbf{P}_{y}\left({\mathcal{L}}\left(y|Z\right)=\operatorname*{max}_{k{\mathrm{~in~}}\{1,2,\dots,Y\}}\left({\mathcal{L}}\left(k|Z\right)\right)\right),$$

where L(y ∣ Z) denotes the likelihood that Z belongs to the distribution associated with class y. We empirically compute this probability by sampling n = 1000 points and measuring the proportion of samples most likely to originate from the intended class. It is worth noting that ASCENSION applies the same likelihood-based filtering criterion before incorporating generated samples into the final training set. Therefore, this confidence metric indicates how often a sample aligns with its target class before any filtering removes unreliable points.

As a result, our measure serves as a valuable yet inherently qualitative indicator of the model's initial ability to generate class-consistent samples.

As shown in Figure 5, contrary to initial expectations, class assignment confidence does not significantly decline throughout the expansion process. This indicates that confidence retention is more influenced by the intrinsic characteristics of each dataset rather than the expansion itself. For a more detailed analysis, readers can refer to Appendix D.

## 5. Conclusion & Future Works

This paper introduced ASCENSION, a novel VAE-based DA method for TSC that integrates a controllable and progressive class boundary expansion mechanism. Unlike existing generative DA methods, which primarily rely on interpolating within the existing training distribution, ASCENSION enables controlled extrapolation, preserving intra-class coherence and enabling the user to monitor inter-class separation. By leveraging a probabilistic latent space structure, ASCENSION effectively generates synthetic samples that enhance classification performance across a broad range of time series datasets.

$$(\mathbb{S})$$

Our benchmarking analysis on 102 UCR datasets highlights ASCENSION's ability to deliver consistent performance improvements. Compared to six state-of-the-art DA methods—FAA, LA, TTS-GAN, Time-DDPM, VaDE, and MODALS—ASCENSION achieved the highest overall classification gains, improving accuracy in 55% of datasets with ResNet and 49% with FCN, while limiting performance degradation to only 29% and 38%, respectively. Additionally, our analysis of DA effectiveness factors reveals that ASCENSION performs particularly well in scenarios where the discrepancy between training and test data is relatively high, whereas other methods experience a sharp decline in effectiveness under such conditions. This finding is particularly significant, as real-world applications often involve variations in train-test distribution discrepancies (see e.g.

Cumu lativ e Su m of Improve ment Perce ntage Cumulative Sum of Improvement Percentage vs. Datasets (Sorted by Ratio)
600 500 400 300 200 100 0 100 LA FAA TTS-GAN ASCENSION
Lightn ing2 Proxi mal Phal anxO
utlin eAg eGrou p Gun Poin tMaleVers usFe male Free zerR
egul arTra in Dist alPha lanx Outli neCo rrect CBF 
Wor dSyn ony ms ECG5 000 Hous eTwe nty Lightn ing7 Mixed Sha pesR
egul arTra in Scree nType Gun Point OldVe rsusYo ung Arrow Hea d Proxi malP
hala nxO
utlin eCor rect Medi calI
mag es Italy Powe rDe man d UWav eGe sture Libra ryZ Beef Plane Midd lePh alanxTW Crop Oliv eOil Coff ee Herri ng Sma llKitc henAp plian ces Sem gHa ndG
ende rCh2 Worm sTwo Class InsectWi ngbe atSo und BirdCh icken ACS
F1 Inse ctEP
GReg ularTra in Ford A 
Sem gHa ndS
ubje ctCh 2 FaceAl l Wafe r Free zerS mall Train Com pute rs TwoLe adE
CG 
Gun Point NonIn vasiv eFeta lECGTh orax1 InsectE
PGS
mall Train Midd lePha lanx Outli neCor rect Semg Han dMo vem entC
h2 Car Son yAIBO
Robo tSurface1 UMD
Dista lPhala nxTW
Chlo rineCo ncen tration Ford B

CinC
ECGTorso Wine Dista lPhal anxO
utlin eAg eGro up Phal ang esO
utlin esCo rrectTrace ToeS
egm entat ion1 Yoga Proxi mal Phala nxTW
Mixe dSha pesS
mallTra in Straw berry UWav eGe sture Libra ryX
ECG
200 Mall at Elec tricD
evic es NonIn vasiv eFeta lECGTh orax2 Rock UWa veG
estu reLi braryAl l PigA
rtPre ssure Refri geratio nDev ices Earth qua kes Face Four Swed ishLe af Han dOutlin es Beetle Fly Haptics Face sUC
R

Sha peletS
im Shap esAll ECGFiv eDa ys Gun Poin tAgeS
pan Wor ms Ham Fish EOG
Horiz ontal Sign al EOGVe rtical Sign al UWaveG
estu reLib raryY
StarLig htCu rves ToeS
egm entat ion2 Son yAIBO
Robo tSurface2 Synt hetic Contro l Large Kitch enAp plian ces Mid dlePh alan xOut lineAg eGro up InlineS
kate Adiac MoteSt rain Powe rCons BME
OSU Leaf Two Patt erns Etha nolLeve l Smo othSu bspa ce Meat Symb ols Datasets (Sorted by Ratio)

## 6. Software And Data References

Figure 4: Cumulative performance improvement (%) as a function of F24, which represents the train-test discrepancy ratio (see Appendix E.1). The 102 UCR datasets are ordered in increasing discrepancy. While other data augmentation (DA) methods show performance degradation as discrepancy rises, ASCENSION maintains a positive performance trend and even demonstrates a slight improvement.

The UCR time series archive can be found at https://www.cs.ucr.edu/˜7Eeamonn/
time_series_data_2018/. We detailed exact implementation details and provide code to produce our results on an anonymous github page at https://github.com/ASCENSION-PAPER
Figure 5: Class confidence distribution over the different augmentation steps.Class assignment confidence does not significantly decline throughout the expansion process.

385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 Cheung, T.-H. and Yeung, D.-Y. Modals: Modality-agnostic automated data augmentation in the latent space. In International Conference on Learning Representations, 2020.

(Koh et al., 2021)), making ASCENSION a valuable asset for practical deployment.

Cubuk, E. D., Zoph, B., Mane, D., Vasudevan, V., and Le, Q. V. Autoaugment: Learning augmentation strategies from data. In *Proceedings of the IEEE/CVF conference* on computer vision and pattern recognition, pp. 113–123, 2019.

Limitations & Future work: While ASCENSION advances generative DA for time series, certain limitations remain. The latent space expansion mechanism requires careful tuning of parameters such as the scaling factor, the number of augmentation steps, and the step size. Automating these hyperparameter selections based solely on training data could be a promising direction for future work. Although ASCENSION ensures class-consistent sampling, incorporating domain-specific priors could further refine boundary expansions. Additionally, ASCENSION's framework could be extended to other types of sequential data
(e.g., natural language, spatio-temporal data) as well as non-sequential domains (e.g., images). Exploring alternative clustering methods, sampling strategies, and expansion mechanisms beyond a single α *factor* - could further improve its adaptability and effectiveness across diverse applications.

Cubuk, E. D., Zoph, B., Shlens, J., and Le, Q. V. Randaugment: Practical automated data augmentation with a reduced search space. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 3008–3017. IEEE Computer Society, 2020.

Dang, T.-H., Park, J., Tran, V.-T., and Chung, W. Y. Vaelstm data augmentation for cattle behavior classification using a wearable inertial sensor. *IEEE Sensors Letters*, 2024.

Fawaz, H. I. *Deep learning for time series classification*.

PhD thesis, Universite de Haute Alsace-Mulhouse, 2020. ´
Feng, S., Miao, C., Zhang, Z., and Zhao, P. Latent diffusion transformer for probabilistic time series forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 11979–11987, 2024.

Fu, B., Kirchbuchner, F., and Kuijper, A. Data augmentation for time series: traditional vs generative models on capacitive proximity time series. In Proceedings of the 13th ACM international conference on pervasive technologies related to assistive environments, pp. 1–10, 2020.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B.,
Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020.

Iglesias, G., Talavera, E., Gonzalez-Prieto, A., Mozo, A.,
and Gomez-Canaval, S. Data augmentation techniques in time series domain: a survey and taxonomy. Neural Computing and Applications, 35(14):10123–10145, 2023a.

Iglesias, G., Talavera, E., Gonzalez-Prieto, A., Mozo, A., and Gomez-Canaval, S. Data Augmentation techniques in time series domain: a survey and taxonomy. *Neural Computing and Applications*, 35(14):10123–10145, May 2023b. ISSN 09410643, 1433-3058. doi: 10.1007/s00521-023-08459-3. URL https://link.springer.com/10.1007/ s00521-023-08459-3.

Iwana, B. K. and Uchida, S. An empirical survey of data augmentation for time series classification with neural networks. *Plos one*, 16(7):e0254841, 2021.

Jiang, Z., Zheng, Y., Tan, H., Tang, B., and Zhou, H. Variational deep embedding: An unsupervised and generative approach to clustering. *arXiv preprint arXiv:1611.05148*, 2016.

Kingma, D. P. and Welling, M. Auto-encoding variational bayes. *arXiv preprint arXiv:1312.6114*, 2013.

Koh, P. W., Sagawa, S., Marklund, H., Xie, S. M., Zhang, M., Balsubramani, A., Hu, W., Yasunaga, M., Phillips, R. L., Gao, I., et al. Wilds: A benchmark of in-thewild distribution shifts. In International conference on machine learning, pp. 5637–5664. PMLR, 2021.

Koonce, B. and Koonce, B. Resnet 50. *Convolutional neural* networks with swift for tensorflow: image recognition and dataset categorization, pp. 63–72, 2021.

Lei, N., Guo, Y., An, D., Qi, X., Luo, Z., Yau, S.-T., and Gu, X. Mode collapse and regularity of optimal transportation maps. *arXiv preprint arXiv:1902.02934*, 2019.

Li, X., Metsis, V., Wang, H., and Ngu, A. H. H. Tts-gan:
A transformer-based time-series generative adversarial network. In International conference on artificial intelligence in medicine, pp. 133–143. Springer, 2022.

Lim, S., Kim, I., Kim, T., Kim, C., and Kim, S. Fast autoaugment. Advances in neural information processing systems, 32, 2019.

Liu, C., Huo, X., He, C., and Du, J. Adaptive diffusion model-based data augmentation for unbalanced time series classification. In 2024 43rd Chinese Control Conference (CCC), pp. 8928–8932. IEEE, 2024.

Liu, Z., Tang, Z., Shi, X., Zhang, A., Li, M., Shrivastava, A., and Wilson, A. G. Learning multimodal data augmentation in feature space. *arXiv preprint arXiv:2212.14453*, 2022.

Lubba, C. H., Sethi, S. S., Knaute, P., Schultz, S. R., Fulcher, B. D., and Jones, N. S. catch22: CAnonical Time-series CHaracteristics: Selected through highly comparative time-series analysis. Data Mining and Knowledge Discovery, 33(6):1821–1852, November 2019. ISSN 13845810, 1573-756X. doi: 10.1007/s10618-019-00647-x.

URL http://link.springer.com/10.1007/ s10618-019-00647-x.

Muller, S. G. and Hutter, F. Trivialaugment: Tuning-free ¨
yet state-of-the-art data augmentation. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 774–782, 2021.

Scabini, L. F. and Bruno, O. M. Structure and performance of fully connected neural networks: Emerging complex network properties. *Physica A: Statistical Mechanics and* its Applications, 615:128585, 2023.

Seon, J., Lee, S., Sun, Y. G., Kim, S. H., Kim, D. I., and Kim, J. Y. Least information spectral gan with time-series data augmentation for industrial iot. IEEE Transactions on Emerging Topics in Computational Intelligence, 2024.

Solis-Martin, D., Galan-Paez, J., and Borrego-Diaz, J. D3ats: Denoising-driven data augmentation in time series. arXiv preprint arXiv:2312.05550, 2023.

Thanh-Tung, H. and Tran, T. Catastrophic forgetting and mode collapse in gans. In 2020 international joint conference on neural networks (ijcnn), pp. 1–10. IEEE, 2020.

Tronchin, L., Vu, M. H., Soda, P., and Lofstedt, T. Laten- ¨
taugment: Data augmentation via guided manipulation of gan's latent space. *arXiv preprint arXiv:2307.11375*,
2023.

Wang, W., Song, H., Si, S., Lu, W., and Cai, Z. Data augmentation based on diffusion probabilistic model for 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 remaining useful life estimation of aero-engines. Reliability Engineering & System Safety, 252:110394, 2024.

Xiao, Z., Kreis, K., and Vahdat, A. Tackling the generative learning trilemma with denoising diffusion gans. In International Conference on Learning Representations.

Yang, L., Zhang, Z., Song, Y., Hong, S., Xu, R., Zhao, Y.,
Zhang, W., Cui, B., and Yang, M.-H. Diffusion models: A comprehensive survey of methods and applications. ACM Computing Surveys, 56(4):1–39, 2023a.

Yang, Z., Li, Y., and Zhou, G. Ts-gan: Time-series gan for sensor-based health data augmentation. ACM Transactions on Computing for Healthcare, 4(2):1–21, 2023b.

Zhang, Y., Zhou, Z., Liu, J., and Yuan, J. Data augmentation for improving heating load prediction of heating substation based on timegan. *Energy*, 260:124919, 2022.

Zheng, Y., Zhang, Z., Yan, S., and Zhang, M. Deep AutoAugment, March 2022. URL http://arxiv.org/ abs/2203.06172. arXiv:2203.06172 [cs].

## A. Related Work

(Iglesias et al., 2023b) and (Iwana & Uchida, 2021) divide DA for time series into two categories: Traditional vs. Generative DA methods. Figure 6 offers an overview of the evolution of these methods.

Traditional DA methods, such as window slicing, jittering, and scaling (Iglesias et al., 2023a), are primarily adapted from computer vision and rely on transformation strategies like cropping, rotation, scaling, drifting, and so forth. However, the complex nature of time series data often renders these methods sub-optimal, as they can disrupt the semantic integrity of the original data. For instance, while a slightly flipped image of a cat remains recognizable, reversing the time axis of an electrocardiogram sequence can render it meaningless. In response to these challenges, more advanced DA techniques were developed to automate the sequence of transformations to be performed. A first method, named **AutoAugment (AA)** (Cubuk et al., 2019), uses reinforcement learning to explore transformation pipelines/policies. A second method named Fast AutoAugment (FAA) (Lim et al., 2019) uses density matching for a faster search strategy, eliminating the need for backpropagation. Subsequent methods such as **RandAugment**
(Cubuk et al., 2020), **Deep AutoAugment** (Zheng et al.,
2022), and **Trivial Augment** (Muller & Hutter ¨ , 2021) were introduced to further simplify and refine the augmentation search strategy. RandAugment streamlines the augmentation process by removing the exhaustive search phase, instead applying a fixed number of random transformations 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 with adjustable magnitudes. Deep AutoAugment incorporates a deep reinforcement learning model that dynamically combines transformation policies based on the specific characteristics of the dataset. Trivial Augment introduces an even simpler approach by applying a minimal set of random transformations, emphasizing ease of use and computational efficiency. Despite all these advancements, all these methods rely on predefined transformations, which is suboptimal for preserving intra-class consistency and the semantic characteristics of the original time series data, thereby limiting the effectiveness of data augmentation. Generative DA models such as Generative Adversarial Networks (GANs) (Goodfellow et al., 2020), diffusion models (Yang et al., 2023a), and VAEs (Kingma & Welling, 2013) represent powerful techniques capable of learning a probabilistic representation of data distributions. These models can generate time series data that retain the temporal dependencies, semantic consistency, and class-specific characteristics of the original datasets (Fu et al., 2020). For example, using a representation layer, as introduced by (Liu et al., 2022), provides an abstraction that is crucial when dealing with time series data. **TimeGAN** (Zhang et al.,
2022) has been specifically designed for time series, which has shown significant improvements in generating highquality synthetic sequences and augmenting low-quality datasets. Likewise, **TS-GAN** (Yang et al., 2023b) develop a LSTM-based GAN architecture with an sequential-squeezeand-excitation to better capture time-dependence between the current and past moments in each dimensions. TS-GAN is particulary proposed to generate augmented sensor-based health data to improve Deep Learning (DL) classification models and evaluated on 3 health time series datasets. TTS- GAN (Li et al., 2022) adapt the traditional GAN architecture using a transfomer-encoder architecture that can deal with long range dependencies in time sequences. It shows strong performance in generating realistic data across three datasets: a simulated dataset, a human acuity recognition dataset, and an ECG dataset. However, GANs training process is very unstable and is very senstive to hyperparameters. It also suffers from issue as mode collapse that can limit the variety of generated samples and can possibly generate unrealistic data (Lei et al., 2019). **LatentAugment** (Tronchin et al., 2023) learns a low-level representation of initial data, noising around learned points and then decoding them to produce newly generated and semantically close data. More recently, (Seon et al., 2024) proposed **LISGAN**, a GAN-
based architecture to augment time series data in the context of class imbalance by adjusting the loss with mutual information term and using a spectral normalization. LISGAN generates high quality synthetic data and significantly increases classification performance with industrial internet of things datasets. Diffusion models, a more recent class of generative models, have garnered significant attention for

{ Code non available } Code available but method not considered for benchmarking in this study | Code available and method considered for benchmarking in this study VAE | VaDE
}
MODALS**
{
VAE-STS
{
VAE-
LSTM
| ASCEN-
SION
GAN }
TimeGAN
}
TTS-
cGAN
| TTS-
GAN
{
TS-
GAN
| Latent Augment
{
Ge ne rati ve m o del s Diffusion | Time-
DDPM
{
D3A-
TS
{
ASE-
DDPM
{
Diff-
RUL
LISGAN
Traditional models
(Auto Augmentation)}
AA
| FAA
}
RA
}
DAA
≤2019 2020 2021 2022 2023 **2024**
550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 their capability to model complex data distributions. Unlike GANs, which rely on adversarial training, diffusion models generate data by progressively refining noise toward the target data distribution. This denoising approach has yielded remarkable results in high-fidelity image generation, as seen with models like DALL·E 2, Imagen, and Flux. Recently, starting in 2023, several diffusion model-based DA methods for time series have emerged, including **ASE-DDPM** (Liu et al., 2024) for addressing imbalanced time series classification, **DiffRUL** (Wang et al., 2024) for enhancing remaining useful life predictions, **D3A-TS** (Solis-Martin et al., 2023) aimed at improving synthetic sample quality through metaattribute conditioning, and **Time-DDPM**, which integrates a diffusion denoising probabilistic model with CNN-LSTM networks to enhance sample quality. While diffusion models provide stable outputs, they face challenges with long-range predictions, error accumulation, and slow inference (Feng et al., 2024), which can limit their practical applications. VAEs offer several advantages over GANs and diffusion models. Their probabilistic nature allows for explicit control over the diversity and quality of generated samples through manipulation of the latent space, as evidenced in (Cheung & Yeung, 2020). This helps preserve the intra-class consistency and semantic characteristics of the original data.

Additionally, VAEs are less prone to collapse compared to GANs and are less computationally expensive than both GANs and diffusion models (Thanh-Tung & Tran, 2020). To our knowledge, the first VAE-based generative DA model relying of clustering, named **VaDE**, was introduced in (Jiang et al., 2016). The authors integrate a prior GMM fitting to the VAE training, enabling realistic samples generation for any specified cluster, without using supervised information during training. **MODALS**, was introduced by (Cheung & Yeung, 2020) and represents the closest architectural approach to ASCENSION. It was the first study to investigate the expansion of class boundaries during synthetic data generation, although it does not offer a method for controlling this expansion. Recently, (Dang et al., 2024) introduced VAE-LSTM, which is used to augment an inertial sensor dataset due to limited data availability, with the goal of enhancing classification performance. However, this approach does not explore the expansion of class representations in the latent space, as proposed in ASCENSION.

## B. Enlarged Experimental Result Analysis B.1. Enlarged Classification Performance

This section offers a more comprehensive analysis of the results. The 102 datasets from the UCR time series classification repository are grouped into 11 distinct categories (domains/applications), as summarized in Table 4. A detailed breakdown of our experimental results is presented in Table 5 and Table 6.

## C. Enlarged Hyperparameters Sensitivity Analysis

Figures 7 to 16 show 3D plots of classifier performance as a function of α and the number of iterations for ASCENSIONEmbCl, FCN, and ResNet, across representative datasets from each category of the UCR archive. The name of each category and their representative datasets are detailed in Table 4.

α **parameter:** As discussed in section 4.2.4, performance improvement relation to α seems difficult to generalize while remaining relatively stable. Increasing α can lead to better boundary exploration, as shown in Figures 11 and 10 but can also make the performance drop for too high

| Type        | FAA   | LA          | Time-DDPM   | TTS-GAN     | VaDE   | ASCENSION   |       |             |       |             |       |      |
|-------------|-------|-------------|-------------|-------------|--------|-------------|-------|-------------|-------|-------------|-------|------|
| ↑NbDatasets | ↑Acc  | ↑NbDatasets | ↑Acc        | ↑NbDatasets | ↑Acc   | ↑NbDatasets | ↑Acc  | ↑NbDatasets | ↑Acc  | ↑NbDatasets | ↑Acc  |      |
| Device      | 1/8   | 7.7%        | 2/8         | 3.1%        | 4/8    | 20.3%       | 3/8   | 1.1%        | 3/8   | 0.7%        | 7/8   | 2.2% |
| ECG         | 1/6   | 14.2%       | 3/6         | 0.2%        | 2/6    | 3.8%        | 3/6   | 1.6%        | 2/6   | 0.3%        | 2/6   | 0.1% |
| EOG         | 0/2   | 0.0%        | 1/2         | 2.8%        | 1/2    | 35.8%       | 0/2   | 0.0%        | 0/2   | 0.0%        | 0/2   | 0.0% |
| EPG         | 0/2   | 0.0%        | 0/2         | 0.0%        | 0/2    | 0.0%        | 0/2   | 0.0%        | 0/2   | 0.0%        | 0/2   | 0.0% |
| Image       | 2/30  | 13.2%       | 10/30       | 2.2%        | 13/30  | 16.4%       | 10/30 | 3.2%        | 11/30 | 4.3%        | 14/30 | 1.8% |
| Motion      | 2/14  | 2.8%        | 10/20       | 1.3%        | 9/20   | 13.2%       | 1/20  | 0.8%        | 9/20  | 1.5%        | 8/20  | 1.0% |
| Power       | 0/1   | 0.0%        | 1/1         | 3.9%        | 0/1    | 0.0%        | 1/1   | 2.8%        | 1/1   | 1.7%        | 1/1   | 2.2% |
| Sensor      | 2/19  | 5.4%        | 7/19        | 2.0%        | 5/19   | 17.2%       | 6/19  | 2.4%        | 6/19  | 3.2%        | 7/19  | 1.2% |
| Simulated   | 3/8   | 3.5%        | 2/8         | 5.3%        | 1/8    | 12.6%       | 5/8   | 1.0%        | 0/8   | 0.0%        | 2/8   | 5.7% |
| Spectro     | 2/8   | 11.3%       | 1/8         | 0.4%        | 4/8    | 6.1%        | 2/8   | 2.9%        | 1/8   | 1.7%        | 7/8   | 9.9% |
| Spectrum    | 0/4   | 0.0%        | 1/4         | 6.0%        | 4/4    | 24.7%       | 0/4   | 0.0%        | 2/4   | 7.1%        | 2/4   | 5.3% |

Table 6: Mean Negative Impact per Dataset Type

| Type        | FAA   | LA          | Time-DDPM   | TTS-GAN     | VaDE   | ASCENSION   |       |             |       |             |       |       |
|-------------|-------|-------------|-------------|-------------|--------|-------------|-------|-------------|-------|-------------|-------|-------|
| ↓NbDatasets | ↑Acc  | ↓NbDatasets | ↑Acc        | ↓NbDatasets | ↑Acc   | ↓NbDatasets | ↑Acc  | ↓NbDatasets | ↑Acc  | ↓NbDatasets | ↑Acc  |       |
| Device      | 7/8   | −8.8%       | 5/8         | −2.5%       | 4/8    | −23.6%      | 5/8   | −9.1%       | 5/8   | −3.0%       | 1/8   | −4.0% |
| ECG         | 5/6   | −27.9%      | 3/6         | −3.7%       | 4/6    | −16.9%      | 3/6   | −1.9%       | 3/6   | 18.3%       | 2/6   | −4.3% |
| EOG         | 2/2   | −21.1%      | 1/2         | −6.6%       | 1/2    | −17.9%      | 2/2   | −32.2%      | 2/2   | −11.3%      | 2/2   | −1.2% |
| EPG         | 0/2   | 0.0%        | 0/2         | 0.0%        | 2/2    | −11.1%      | 0/2   | 0.0%        | 0/2   | 0.0%        | 0/2   | 0.0%  |
| Image       | 27/30 | −17.6%      | 15/30       | −1.9%       | 17/30  | −20.7%      | 17/30 | −5.9%       | 13/30 | −11.1%      | 14/30 | −1.3% |
| Motion      | 11/14 | −13.6%      | 2/14        | −1.4%       | 5/14   | −27.8%      | 11/14 | −11.1%      | 4/14  | −2.5%       | 5/14  | −2.0% |
| Power       | 1/1   | −2.8%       | 0/1         | 0.0%        | 1/1    | −88.5%      | 0/1   | 0.0%        | 0/1   | 0.0%        | 0/1   | 0.0%  |
| Sensor      | 17/19 | −11.1%      | 10/19       | −1.4%       | 14/19  | −28.7%      | 11/19 | −3.1%       | 11/19 | −2.5%       | 6/19  | −0.4% |
| Simulated   | 5/8   | −15.9%      | 3/8         | −1.6%       | 6/8    | −18.3%      | 1/8   | −1.4%       | 6/8   | −8.5%       | 5/8   | −0.8% |
| Spectro     | 6/8   | −32.4%      | 4/8         | −5.1%       | 4/8    | −25.2%      | 4/8   | −3.2%       | 5/8   | −2.0%       | 1/8   | −0.5% |
| Spectrum    | 4/4   | −3.4%       | 3/4         | −2.0%       | 0/4    | 0.0%        | 4/4   | −12.2%      | 2/4   | −3.9%       | 2/4   | −0.6% |

values of α. While pinpointing the exact α values and iterations for optimal results across all datasets is not trivial, the general trend suggests selecting α ∈ [1, 3] to expand class boundaries without venturing into areas that risk class overlap, which could negatively impact classification accuracy. Number of iterations: In Figures 10-12, and 14, we observe that a higher number of iterations can have either a positive or negative impact on performance, whereas in Figure 7, the number of iterations does not play a significant role in performance improvement. This ambivalent behavior is closely related to the class distribution within the dataset.

As the number of iterations increases, classes in the latent space may become closer due to the increase in the α parameter at each iteration, which leads to the expansion of covariances αΣk (cf., Figure 1). Therefore, we recommend carefully adjusting the number of iterations in relation to the chosen α parameter.

ASCENSIONEmb. ASCENSIONFCN-Emb. ASCENSIONResNet-Emb.

605 606 607 608 609 610 611 612 613 614 615 616 617

618

619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639

640

641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659

| Type      | Representative dataset   | Description                                                                  |
|-----------|--------------------------|------------------------------------------------------------------------------|
| Device    | ACSF1                    | Measurements of alternating current signals for predictive maintenance       |
| ECG       | ECG200                   | Electrocardiogram (ECG) readings used to detect heart abnormalities          |
| EOG       | EOGVerticalSignal        | Electrooculography (EOG) signals capturing eye movement patterns             |
| EPG       | InsectEPGRegularTrain    | Electrical penetration graph (EPG) signals capturing insect feeding behavior |
| Image     | BeetleFly                | Shape-based image classification of beetle and fly outlines                  |
| Motion    | Worms                    | Motion sensor data capturing worm movements for classification               |
| Power     | PowerCons                | Power consumption measurements for energy usage                              |
| Sensor    | Car                      | Sensor readings collected from a car, used for detecting driving conditions  |
| Simulated | UMD                      | Simulated control processes data                                             |
| Spectro   | Ham                      | Spectroscopy data to identify types of ham based on chemical properties      |
| Spectrum  | SemgHandMovementCh2      | Electromyography (EMG) data of hand movements, recorded across channels      |

Figure 8: **EOG:** Classifier performance against α and iteration number for **EOGVerticalSignal**. Figure 9: **Hemodynamics:** Classifier performance against α and iterations for **PigArtPressure**. Figure 10: **Image:** Classifier performance against α and iteration number for **BeetleFly** dataset.

660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714 Figure 13: **Simulated:** Classifier performance against α and iteration number for UMD dataset.

Figure 14: **Spectro:** Classifier performance against α and iteration number for Ham dataset.

ASCENSIONEmb. ASCENSIONFCN-Emb. ASCENSIONResNet-Emb.
Figure 15: **Spectrum:** Classifier performance against α and iteration number for **SemgHandMovementCh2** dataset. Figure 16: **Device:** Classifier performance against α and iteration number for **ACSF1** dataset.

## D. Enlarged Analysis Of The Class Assignment Confidence

All following figures of this section have been computed after removing outlier data samples. Both Figures 17 and 18 show a complex relationship between confidence and performance. A slight positive correlation appears to be present, however it is clear that no linear or polynomial relationship exists between the two. From the previous analysis, we perform a clustering using DBSCAN to extract patterns. Figures 19 and 20 reveal two main clusters. As mentioned previously, we infer that these clusters may depend on the initial conditions of the augmentation, that is to say, the dataset and its characteristics. We validate this hypothesis by computing the feature importances of the dataset's features defined in Appendix F in regards to predicting confidence through a Random Forest Regressor built with a high number of shallow trees. The negative or positive characteristic of the importance is then computed using a correlation matrix. The results in Figure 21 show five features with predominant importance. The contrast in these importance allows us to Figure 12: **Sensor:** Classifier performance against α and iteration number for Car dataset. Figure 11: **Motion:** Classifier performance against α and iteration number for **Worms** dataset.

715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769
validate the hypothesis that some datasets features seem to have a relationship with the confidence of the expansion mechanism. (The full tables of results are available in the supplementary materials in csv and json format.)

## E. Performance Metric Formalization

E.1. Discrepancy in distance between training and test sets E.1.1. FORMALIZATION
To estimate the discrepancy in distance between the training and test sets, we compute the mean intra-class distance across all classes using DTW as the distance metric. Let Xk = xk,1, xk,2, . . . , xk,nk represent the set of generated samples belonging to class k, and dk be the mean intra-class distance for class k, defined as:
where µk is the mean of the samples in class k (computed using DTW barycenter averaging, where applicable). The overall dispersion D of the dataset is then defined as the mean intra-class variance across all K classes:

$$D={\frac{1}{K}}\sum_{k=1}^{K}d_{k}$$

To estimate the discrepancy between the training and test datasets, we compute the ratio between the dispersion of the test set Dtest and the diversity of the train set Dtrain. This ratio V is defined as:

$$V={\frac{D_{\mathrm{test}}}{D_{\mathrm{train}}}}$$
$$\mathbf{(8)}$$. 
The discrepancies ratio V ≈ 1 indicates similar diversity between the train and test sets, while deviations from 1 suggest more diversity in the training set (V < 1) or in the test set (V > 1). A dataset where the ratio V > 1 is considered to be more challenging for usual generative techniques, as the train set does not accurately represent the test set in these cases.

As such the datasets at the far right in

## E.1.2. Experimental Results

$$(6)$$

The discrepancy ratio of the 102 UCR datasets have been plotted in an ascending order in Figure 22. Le us consider three datasets with extreme ratios: **(i) Discrepancy toward**
test: Dataset Car (1.51); **(ii) No discrepancy:** Dataset ECGFiveDays (1.01); **(iii) Discrepancy toward train:** Dataset EOGVerticalSignal (0.77). Detailed results of the discrepancies across datasets are available in Table 7

$$\left(7\right)$$

## F. Time Series Features

In this section, we describe the 22 time series features (Catch22) presented in (Lubba et al., 2019), and the two

$$d_{k}={\frac{1}{n_{k}}}\sum_{i=1}^{n_{k}}\mathrm{DTW}(x_{k,i},\mu_{k})$$

| 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 810 811 812 813 814 815 816 817 818 819 820 821 822 823 824   | Table 7: Discrepancy Metrics Across Datasets   |                 |                  |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------|-----------------|------------------|
| Dataset                                                                                                                                                                                                                   | Ratio                                          | DispersionT EST | DispersionT RAIN |
| HandOutlines                                                                                                                                                                                                              | 0.46                                           | 1.50 × 102      | 1.39 × 102       |
| GesturePebbleZ2                                                                                                                                                                                                           | 0.66                                           | 3.09 × 101      | 3.02 × 101       |
| ShakeGestureWiimoteZ                                                                                                                                                                                                      | 0.71                                           | 5.36 × 102      | 6.04 × 102       |
| GestureMidAirD1                                                                                                                                                                                                           | 0.75                                           | 4.18 × 102      | 4.30 × 102       |
| MiddlePhalanxOutlineCorrect                                                                                                                                                                                               | 0.77                                           | 1.01 × 106      | 1.02 × 106       |
| EOGVerticalSignal                                                                                                                                                                                                         | 0.77                                           | 6.38 × 103      | 5.62 × 103       |
| Chinatown                                                                                                                                                                                                                 | 0.84                                           | 1.71 × 103      | 2.05 × 103       |
| PLAID                                                                                                                                                                                                                     | 0.85                                           | 3.50 × 102      | 3.38 × 102       |
| ProximalPhalanxOutlineCorrect                                                                                                                                                                                             | 0.87                                           | 1.34 × 101      | 1.48 × 101       |
| EthanolLevel                                                                                                                                                                                                              | 0.87                                           | 3.18 × 101      | 2.10 × 101       |
| Wine                                                                                                                                                                                                                      | 0.87                                           | 3.34 × 104      | 3.33 × 104       |
| Trace                                                                                                                                                                                                                     | 0.88                                           | 4.46 × 103      | 4.41 × 103       |
| ScreenType                                                                                                                                                                                                                | 0.88                                           | 2.18 × 102      | 2.46 × 102       |
| Worms                                                                                                                                                                                                                     | 0.89                                           | 1.13 × 102      | 1.00 × 102       |
| BeetleFly                                                                                                                                                                                                                 | 0.89                                           | 5.79 × 101      | 5.30 × 101       |
| GesturePebbleZ1                                                                                                                                                                                                           | 0.90                                           | 4.34 × 100      | 3.98 × 100       |
| OliveOil                                                                                                                                                                                                                  | 0.91                                           | 5.64 × 100      | 5.94 × 100       |
| Strawberry                                                                                                                                                                                                                | 0.91                                           | 1.59 × 102      | 1.56 × 102       |
| WormsTwoClass                                                                                                                                                                                                             | 0.93                                           | 4.09 × 101      | 4.26 × 101       |
| Lightning7                                                                                                                                                                                                                | 0.94                                           | 3.32 × 101      | 3.80 × 101       |
| Meat                                                                                                                                                                                                                      | 0.94                                           | 2.80 × 103      | 1.35 × 103       |
| Plane                                                                                                                                                                                                                     | 0.94                                           | 9.58 × 101      | 1.01 × 102       |
| Beef                                                                                                                                                                                                                      | 0.94                                           | 6.40 × 101      | 6.78 × 101       |
| ProximalPhalanxOutlineAgeGroup                                                                                                                                                                                            | 0.94                                           | 4.70 × 102      | 7.09 × 102       |
| ShapesAll                                                                                                                                                                                                                 | 0.94                                           | 4.40 × 101      | 3.95 × 101       |
| ProximalPhalanxTW                                                                                                                                                                                                         | 0.94                                           | 1.39 × 104      | 1.36 × 104       |
| MiddlePhalanxTW                                                                                                                                                                                                           | 0.94                                           | 4.74 × 100      | 5.02 × 100       |
| SemgHandSubjectCh2                                                                                                                                                                                                        | 0.95                                           | 5.14 × 101      | 5.28 × 101       |
| ItalyPowerDemand                                                                                                                                                                                                          | 0.95                                           | 2.75 × 100      | 2.92 × 100       |
| PhalangesOutlinesCorrect                                                                                                                                                                                                  | 0.95                                           | 2.02 × 101      | 2.00 × 101       |
| DistalPhalanxOutlineCorrect                                                                                                                                                                                               | 0.96                                           | 5.31 × 100      | 6.94 × 100       |
| MoteStrain                                                                                                                                                                                                                | 0.96                                           | 3.27 × 101      | 2.60 × 101       |
| CricketY                                                                                                                                                                                                                  | 0.96                                           | 3.90 × 102      | 3.94 × 102       |
| AllGestureWiimoteY                                                                                                                                                                                                        | 0.96                                           | 1.57 × 101      | 1.63 × 101       |
| SwedishLeaf                                                                                                                                                                                                               | 0.96                                           | 4.69 × 102      | 4.37 × 102       |
| ACSF1                                                                                                                                                                                                                     | 0.96                                           | 1.01 × 103      | 1.04 × 103       |
| FaceAll                                                                                                                                                                                                                   | 0.97                                           | 3.58 × 101      | 3.67 × 101       |
| SemgHandGenderCh2                                                                                                                                                                                                         | 0.97                                           | 1.47 × 102      | 1.53 × 102       |
| DodgerLoopDay                                                                                                                                                                                                             | 0.97                                           | 6.13 × 102      | 6.62 × 102       |
| NonInvasiveFetalECGThorax2                                                                                                                                                                                                | 0.97                                           | 2.52 × 100      | 2.42 × 100       |
| Computers                                                                                                                                                                                                                 | 0.97                                           | 1.94 × 102      | 1.98 × 102       |
| MelbournePedestrian                                                                                                                                                                                                       | 0.97                                           | 7.90 × 101      | 7.41 × 101       |
| AllGestureWiimoteX                                                                                                                                                                                                        | 0.97                                           | 1.63 × 102      | 1.64 × 102       |
| UMD                                                                                                                                                                                                                       | 0.97                                           | 1.89 × 101      | 1.89 × 101       |
| ToeSegmentation2                                                                                                                                                                                                          | 0.97                                           | 2.03 × 102      | 1.72 × 102       |
| MixedShapesRegularTrain                                                                                                                                                                                                   | 0.98                                           | 4.20 × 102      | 4.76 × 102       |
| OSULeaf                                                                                                                                                                                                                   | 0.98                                           | 8.85 × 103      | 6.43 × 103       |
| NonInvasiveFetalECGThorax1                                                                                                                                                                                                | 0.98                                           | 1.31 × 102      | 1.33 × 102       |
| FordB                                                                                                                                                                                                                     | 0.98                                           | 2.81 × 100      | 2.80 × 100       |
| SmallKitchenAppliances                                                                                                                                                                                                    | 0.99                                           | 2.49 × 101      | 2.61 × 101       |

ASCENSION: Autoencoder-Based Latent Space Class Expansion for Time Series Data Augmentation

| 825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879   | Dataset   | Ratio      | DispersionT EST   | DispersionT RAIN   |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|------------|-------------------|--------------------|
| FordA                                                                                                                                                                                                                         | 0.99      | 3.73 × 103 | 3.83 × 103        |                    |
| CricketZ                                                                                                                                                                                                                      | 0.99      | 2.55 × 101 | 2.52 × 101        |                    |
| HouseTwenty                                                                                                                                                                                                                   | 0.99      | 2.44 × 100 | 2.79 × 100        |                    |
| SemgHandMovementCh2                                                                                                                                                                                                           | 1.00      | 1.23 × 104 | 1.24 × 104        |                    |
| CricketX                                                                                                                                                                                                                      | 1.00      | 6.78 × 101 | 6.10 × 101        |                    |
| Earthquakes                                                                                                                                                                                                                   | 1.00      | 1.31 × 102 | 1.24 × 102        |                    |
| TwoLeadECG                                                                                                                                                                                                                    | 1.00      | 2.28 × 101 | 2.32 × 101        |                    |
| SonyAIBORobotSurface1                                                                                                                                                                                                         | 1.00      | 8.36 × 100 | 8.36 × 100        |                    |
| MedicalImages                                                                                                                                                                                                                 | 1.00      | 7.57 × 101 | 8.10 × 101        |                    |
| TwoPatterns                                                                                                                                                                                                                   | 1.00      | 5.83 × 102 | 3.90 × 102        |                    |
| Crop                                                                                                                                                                                                                          | 1.00      | 1.28 × 104 | 1.35 × 104        |                    |
| Fish                                                                                                                                                                                                                          | 1.00      | 1.13 × 103 | 9.94 × 102        |                    |
| GunPointAgeSpan                                                                                                                                                                                                               | 1.00      | 5.50 × 100 | 4.90 × 100        |                    |
| FreezerRegularTrain                                                                                                                                                                                                           | 1.01      | 2.47 × 103 | 3.27 × 103        |                    |
| Herring                                                                                                                                                                                                                       | 1.01      | 1.02 × 101 | 1.07 × 101        |                    |
| GestureMidAirD2                                                                                                                                                                                                               | 1.01      | 6.39 × 100 | 6.13 × 100        |                    |
| ECGFiveDays                                                                                                                                                                                                                   | 1.01      | 5.42 × 101 | 4.85 × 101        |                    |
| LargeKitchenAppliances                                                                                                                                                                                                        | 1.01      | 3.68 × 101 | 3.08 × 101        |                    |
| GunPointMaleVersusFemale                                                                                                                                                                                                      | 1.02      | 3.69 × 101 | 5.17 × 101        |                    |
| GunPointOldVersusYoung                                                                                                                                                                                                        | 1.02      | 5.70 × 102 | 6.35 × 102        |                    |
| Lightning2                                                                                                                                                                                                                    | 1.02      | 5.96 × 101 | 1.31 × 102        |                    |
| Yoga                                                                                                                                                                                                                          | 1.02      | 3.02 × 104 | 2.97 × 104        |                    |
| AllGestureWiimoteZ                                                                                                                                                                                                            | 1.02      | 1.06 × 101 | 9.93 × 100        |                    |
| PowerCons                                                                                                                                                                                                                     | 1.02      | 2.07 × 104 | 1.63 × 104        |                    |
| SyntheticControl                                                                                                                                                                                                              | 1.02      | 2.29 × 102 | 1.92 × 102        |                    |
| UWaveGestureLibraryX                                                                                                                                                                                                          | 1.02      | 6.81 × 101 | 6.67 × 101        |                    |
| GunPoint                                                                                                                                                                                                                      | 1.04      | 3.83 × 102 | 3.91 × 102        |                    |
| UWaveGestureLibraryAll                                                                                                                                                                                                        | 1.04      | 5.73 × 101 | 5.46 × 101        |                    |
| FaceFour                                                                                                                                                                                                                      | 1.04      | 5.44 × 101 | 5.14 × 101        |                    |
| DistalPhalanxTW                                                                                                                                                                                                               | 1.04      | 2.07 × 101 | 2.07 × 101        |                    |
| SmoothSubspace                                                                                                                                                                                                                | 1.04      | 4.86 × 101 | 3.19 × 101        |                    |
| UWaveGestureLibraryY                                                                                                                                                                                                          | 1.05      | 2.00 × 101 | 1.73 × 101        |                    |
| FiftyWords                                                                                                                                                                                                                    | 1.05      | 3.80 × 100 | 4.03 × 100        |                    |
| StarLightCurves                                                                                                                                                                                                               | 1.05      | 5.40 × 104 | 4.59 × 104        |                    |
| ChlorineConcentration                                                                                                                                                                                                         | 1.05      | 9.02 × 101 | 9.00 × 101        |                    |
| RefrigerationDevices                                                                                                                                                                                                          | 1.05      | 4.23 × 101 | 4.01 × 101        |                    |
| UWaveGestureLibraryZ                                                                                                                                                                                                          | 1.06      | 8.64 × 100 | 9.18 × 100        |                    |
| InsectWingbeatSound                                                                                                                                                                                                           | 1.06      | 7.54 × 102 | 7.85 × 102        |                    |
| Coffee                                                                                                                                                                                                                        | 1.07      | 8.05 × 100 | 8.45 × 100        |                    |
| Ham                                                                                                                                                                                                                           | 1.07      | 4.23 × 102 | 3.75 × 102        |                    |
| InlineSkate                                                                                                                                                                                                                   | 1.07      | 8.25 × 100 | 6.80 × 100        |                    |
| Haptics                                                                                                                                                                                                                       | 1.08      | 3.27 × 101 | 2.98 × 101        |                    |
| Adiac                                                                                                                                                                                                                         | 1.09      | 2.81 × 101 | 2.25 × 101        |                    |
| CBF                                                                                                                                                                                                                           | 1.09      | 6.69 × 104 | 8.63 × 104        |                    |
| InsectEPGSmallTrain                                                                                                                                                                                                           | 1.10      | 1.63 × 102 | 1.64 × 102        |                    |
| ElectricDevices                                                                                                                                                                                                               | 1.10      | 1.02 × 102 | 9.84 × 101        |                    |
| DodgerLoopGame                                                                                                                                                                                                                | 1.10      | 6.43 × 102 | 6.10 × 102        |                    |
| WordSynonyms                                                                                                                                                                                                                  | 1.11      | 4.32 × 103 | 5.08 × 103        |                    |
| FreezerSmallTrain                                                                                                                                                                                                             | 1.11      | 2.29 × 102 | 2.35 × 102        |                    |
| Mallat                                                                                                                                                                                                                        | 1.11      | 2.40 × 101 | 2.32 × 101        |                    |

880 881 882 883 884 885 886 887 888 889 890 891 892

893

894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914

915

916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934

Dataset Ratio DispersionT EST Dispersion*T RAIN*

FacesUCR 1.12 1.20 × 1031.08 × 103

MiddlePhalanxOutlineAgeGroup 1.12 2.70 × 1012.24 × 101

Wafer 1.12 2.24 × 1022.30 × 102

ShapeletSim 1.14 1.41 × 1041.46 × 104

ArrowHead 1.16 1.71 × 1001.88 × 100

EOGHorizontalSignal 1.18 3.01 × 1012.65 × 101

ToeSegmentation1 1.18 2.19 × 1022.16 × 102

SonyAIBORobotSurface2 1.18 2.80 × 1012.36 × 101

MixedShapesSmallTrain 1.19 1.59 × 1021.55 × 102

ECG5000 1.19 4.17 × 1014.77 × 101

ECG200 1.21 1.28 × 1021.25 × 102

DistalPhalanxOutlineAgeGroup 1.21 6.78 × 1016.71 × 101

CinCECGTorso 1.24 1.41 × 1011.40 × 101

PickupGestureWiimoteZ 1.25 5.23 × 1005.98 × 100

InsectEPGRegularTrain 1.26 1.88 × 1011.94 × 101

Rock 1.27 1.16 × 1021.11 × 102

BirdChicken 1.30 5.28 × 1015.47 × 101

PigArtPressure 1.38 1.03 × 1029.85 × 101

Phoneme 1.50 5.18 × 1014.70 × 101

Car 1.51 3.94 × 1023.95 × 102

PigCVP 1.52 6.68 × 1016.54 × 101

Symbols 1.53 1.23 × 1013.72 × 100

PigAirwayPressure 2.07 7.11 × 1025.72 × 102

DiatomSizeReduction 3.30 1.52 × 1031.00 × 103

ASCENSION: Autoencoder-Based Latent Space Class Expansion for Time Series Data Augmentation additional features (denoted by F23 and F24 below) considered in this study.

F7: MD hrv classic **pnn40** Proportion of successive differences in time series values that exceed 0.04 of the standard deviation, indicating rapid fluctuations.

F1: DN **HistogramMode** 5 Top z-score range based on the highest count from a 5-bin histogram, representing the most frequent distribution range in the dataset.

F2: DN **HistogramMode** 10 Similar to DN5, but this considers the top z-score range based on a 10-bin histogram, providing a finer resolution.

F3: CO **f1ecac** Represents the first 1/e crossing of the autocorrelation function, indicating how quickly the autocorrelation of a time series decays.

F4: CO **FirstMin** ac Identifies the first minimum of the autocorrelation function, which helps analyze the periodicity of the time series.

F6: CO trev 1 num This statistic measures timereversibility, focusing on the differences between successive points in the time series raised to the third power.

F8: SB BinaryStats mean **longstretch1** The longest period where values stay consecutively above the mean, representing persistent trends in the data.

F9: SB TransitionMatrix 3ac **sumdiagcov**
Trace of the covariance of the transition matrix between symbols in a 3-letter alphabet, used to assess transitions in symbolized data.

F10: PD PeriodicityWang th0 01 A periodicity measure, indicating how regularly patterns repeat within the time series.

F11: CO Embed2 Dist tau d expfit **meandiff**
Exponential fit to the differences in distances between successive points in a 2-dimensional embedding space, revealing structural relationships.

F12: IN AutoMutualInfoStats 40 gaussian **fmmi**
First minimum of the automutual information function, which gives insight into the periodicity and structure of the time series.

F13: FC LocalSimple mean1 **tauresrat**
Measures the change in correlation length after F5: CO HistogramAMI **even** 2 5 Automutual information for m = 2 and τ = 5, capturing the dependency between data points across time.

Figure 19: Clustering of the confidence in regards to performance for FCN: Overview of the relationship between mean confidence over the augmentation steps and final performance through a DBSCAN clustering.

.

935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989 iteratively differencing the time series, providing insights into the stationarity of the data.

F14: DN OutlierInclude p 001 **mdrmd** Measures the time intervals between successive extreme events occurring above the mean, indicating patterns of high values.

F15: DN OutlierInclude n 001 **mdrmd** Similar to DNOp but for extreme events occurring below the mean, highlighting the time intervals between lowvalue outliers.

F16: SP Summaries welch rect **area** 5 1 This computes the total power in the lowest fifth of the frequencies from a Fourier power spectrum, reflecting long-term trends.

F17: SB BinaryStats diff **longstretch0** The longest period of successive decreases in the time series, capturing prolonged declining trends.

F18: SB MotifThree **quantile** hh Shannon entropy of successive symbol pairs in a 3-letter quantile symbolization, quantifying the complexity of transitions between motifs.

F19: SC FluctAnal 2 rsrangefit 50 1 logi **prop** r1 Proportion of slower timescale fluctuations that scale with rescaled range fits, indicating long-term memory in the data.

Figure 20: **Clustering of the confidence in regards to** performance for ResNet: Overview of the relationship between mean confidence over the augmentation steps and final performance through a DBSCAN clustering.

.

F20: SC FluctAnal 2 dfa 50 1 2 logi **prop** r1 Proportion of slower timescale fluctuations that scale with detrended fluctuation analysis (DFA) under 50 F21: SP Summaries welch rect **centroid** The centroid of the Fourier power spectrum, which offers a measure of the central frequency or the dominant pattern in the time series.

F22: FC LocalSimple mean3 **stderr** Calculates the mean error from a rolling 3-sample mean forecast, capturing the volatility of short-term predictions.

F23: Train Test **Ratio** The ratio of training data to test data in the dataset.

F24: Discrepancy in **Distance** To estimate the discrepancy in distance between the training and testing set distributions, as defined in Appendix E.1

990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044
Figure 21: **Feature importance in regards to confidence:** Overview of the impact of every dataset feature on the mean confidence over the augmentation steps. The red color denotes the negative correlation these features hold with confidence.

.

## G. Evolution Of Latent Space Through Learning Phase

A progressive visualization of the latent space offers valuable insights into the evolving distribution modeling and exploration process. Initially, the latent space representations exhibit fine clustering, but as we iterate in the augmentation loop, the latent space distributions become denser, enhancing the exploration part of these distributions. However, in the later stages of augmentation, the exploration process becomes increasingly challenging as the inter-class distances appear to shrink due to prior augmentation steps. It is important to note that these visualizations provide only a limited view of the actual distributions, as they are restricted to three dimensions (from an original 50-dimensional space).

| ASCENSION: Autoencoder-Based Latent Space Class Expansion for Time Series Data Augmentation                                                                                                                                                                                        |                                                                                                           |           |     |        |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|-----------|-----|--------|
| 1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079 1080 1081 1082 1083 1084 1085 1086 1087 1088 1089 1090 1091 1092 1093 1094 1095 1096 1097 1098 1099 | Table 8: Latent Space Evolution. Visualization of the latent space for the 3 first dimensions (out of 50) |           |     |        |
| Step                                                                                                                                                                                                                                                                               | ACSF1                                                                                                     | BeetleFly | Car | ECG200 |
| Original Step 0 Step 1 Step 2 Step 3 Step 4                                                                                                                                                                                                                                        | 20                                                                                                        |           |     |        |