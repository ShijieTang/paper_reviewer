000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 Anonymous authors Paper under double-blind review

## Abstract

While the majority of time series classification research has focused on modeling fixedlength sequences, variable-length time series classification (VTSC) remains critical in healthcare, where sequence length may vary among patients and events. To address this challenge, we propose Stochastic Sparse Sampling (SSS), a novel VTSC framework developed for medical time series. SSS manages variable-length sequences by sparsely sampling fixed windows to compute local predictions, which are then aggregated and calibrated to form a global prediction. We apply SSS to the task of seizure onset zone (SOZ) localization, a critical VTSC problem requiring identification of seizure-inducing brain regions from variable-length electrophysiological time series. We evaluate our method on the Epilepsy iEEG Multicenter Dataset, a heterogeneous collection of intracranial electroencephalography (iEEG) recordings obtained from four independent medical centers. SSS demonstrates superior performance compared to state-of-the-art (SOTA) baselines across most medical centers, and superior performance on all out-of-distribution (OOD) unseen medical centers.

Additionally, SSS naturally provides post-hoc insights into local signal characteristics related to the SOZ, by visualizing temporally averaged local predictions throughout the signal.

## 1 Introduction

Artificial intelligence (AI) in medicine has received significant attention in recent years, with various applications to clinical diagnosis and treatment planning (Rajpurkar et al., 2022). Despite its advancements, the actual integration into everyday clinical practice remains limited, with much of it attributed to the challenges of handling the complexity and variability in medical data. One particularly challenging aspect of this variability lies in the nature of medical time series data. Variable-length time series are prevalent throughout many areas of healthcare, including heart rate monitoring, blood glucose measurements, and electrophysiological recordings where sequence length can vary dependent on the recording or length of an event (Agliari et al., 2020; Deutsch et al., 1994; Walther et al., 2023). Yet, the majority of time series classification (TSC) literature focuses solely on methods that process fixed-length sequences (Ismail Fawaz et al., 2019; Mohammadi Foumani et al., 2024).

At the same time, healthcare applications require greater interpretability from modern time series methods to expand their applicability in critical domains and accelerate clinical adoption (Amann et al., 2020).

This interpretability is especially crucial in contexts where the relationship between pathology and signal characteristics is not well understood, as it can provide valuable insights for both clinicians and scientists. Recent studies in time series classification (TSC) have explored the explainability of specific signal segments, as opposed to full-signal analysis, which proves particularly useful for uncovering important characteristics

# Stochastic Sparse Sampling: A Framework For Variable-Length Medical Time Series Classifica- Tion

1 047 048 049 050 051 052 053 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 such as motifs, anomalies, or frequency patterns (Early et al., 2024; Crabbé & Van Der Schaar, 2021; Huang et al., 2024). However, there still remains a significant need for models with built-in interpretability in medical applications. Such methods would allevaiate the burden of implementing both a base model and a specialized interpretability method—which may require more domain expertise—and may more effectively facilitate clinical adoption.

The need for variable-length time series classification (VSTC) methods with built-in interpretability is particularly relevant in seizure onset zone (SOZ) localization—the task of identifying brain regions from which seizures originate—as effective treatment requires analysis of variable-length signals (Balaji & Parhi, 2022). The World Health Organization (WHO) reports epilepsy affects over 50 million people globally, establishing it as one of the most common yet poorly understood neurological disorders (Organization et al., 2019; Stafstrom & Carmant, 2015). Additionally, one-third of patients do not respond to antiepileptic drugs, making surgery the last resort and accurate SOZ localization essential for effectively planning the operation.

The process of SOZ identification involves a two-step procedure: initial implantation of electrodes in areas suspected to contain the SOZ, followed by recording and visual analysis of intracranial electroencephalography (iEEG) signals by medical experts. The task of SOZ localization reduces to classifying individual electrode recordings, representing different regions within the brain. Effective localization of the SOZ is challenging due to the absence of clinically validated biological markers and the variable-length nature of iEEG signals—consequently, surgical success rates range from 30% to 70% (Löscher et al., 2020; Li et al., 2021). Figure 1: An overview of Stochastic Sparse Sampling (SSS) training procedure. (A) For a given time series, we sample windows of fixed-length at random throughout the signal. (B) Each window is processed independently by a local model with parameters θ, outputting the local predictions yˆ1*, . . . ,* yˆk. (C) Local predictions are then fed through an aggregation function to form the final prediction yˆ. Contributions. While our work primarily focuses on VTSC, we also evaluate our method's performance on OOD data and explore its potential for providing local explanations. To this end, we propose Stochastic Sparse Sampling (SSS) a novel framework for VTSC developed for medical time series. The main contributions of our paper are listed as follows:
- **Robustness to long and variable-length sequences**. SSS samples fixed-length windows, and processes them independently through a single model. This prevents context overload in long sequences seen in infinite-context methods, and does not utilize padding, truncation, or interpolation 2 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 required by finite-context methods. By relying on a single local model, SSS utilizes far fewer parameters compared to finite-context methods that traditionally ingest the entire signal, which significantly reduces computational cost during training and the risk of overfitting over long sequences.

- **Generalization to unseen patient populations.** SSS demonstrates strong performance on out-of-distribution (OOD) data from unseen medical centers. When trained on data from one or more medical centers and evaluated on a completely new center with a different patient population, SSS outperforms all baselines in our comparisons. This result suggests SSS's potential as a foundation model for TSC, opening new avenues for research and clinical applications.

- **Explainability through local predictions**: Our method enhances model interpretability by directly tying each output—a probability score for each window—to the overall prediction. This capability is crucial in critical clinical settings, such as SOZ localization, which traditionally relies on visual analysis. Given the significant risks associated with brain region removal, any proposal should be designed to integrate within clinical workflows. Moreover, in the absence of universally recognized biological markers for epilepsy, SSS offers the potential to further our understanding the SOZ and to identify novel markers.

- **Compatibility with modern and classical backbones.** SSS integrates with any time series backbone.

This ensures that our approach leverages well-established frameworks now and into the future, allowing for adaptability across a diverse array of contexts.

## 2 Related Work

TSC methodologies can be broadly categorized into finite-context methods, which operate on fixed-length input segments, and infinite-context methods, which handle variable-length sequences without being restricted to a predetermined window size. For a formal treatment, please see Appendix A.5. Finite-context methods. Finite-context methods are among the most commonly used approaches for TSC.

Transformer-based models have gained significant attention, with variations such as sparse attention, series decomposition, and patching techniques (Vaswani et al., 2017; Kitaev et al., 2020; Zhou et al., 2021; Wu et al.,
2021; Zhou et al., 2022; Liu et al., 2022; Nie et al., 2023; Liu et al., 2024). Several temporal convolutional networks (TCNs) have also been proposed, to capture temporal dependencies through dilated convolutions and Inception-like architectures (Lai et al., 2018; Bai et al., 2018; Ismail Fawaz et al., 2020; Wu et al., 2022; Luo & Wang, 2024). Recently, multilayer perceptrons (MLPs) and simple linear models have demonstrated competitive performance as well (Chen et al., 2023; Zeng et al., 2023). Despite the significant rise of finitecontext methods, these methods are inherently limited in their ability to handle variable-length sequences, and will require the use of either padding, truncation, or interpolation for VTSC. Furthermore, as the sequence length increases, so does the number of model parameters, which leads to not only greater computational cost but an increased risk of overfitting. Infinite-context methods. The recurrent neural network (RNN) family includes several models capable of ingesting variable-length time series (Rumelhart et al., 1986). Long-short term memory (LSTM) networks introduce memory cells and gating mechanisms to better handle long-term dependencies (Hochreiter & Schmidhuber, 1997). Gated recurrent units (GRUs) simplify the LSTM architecture while maintaining similar performance (Bahdanau et al., 2014). State space models (SSMs) have gained recent attention, with approaches such as S4 introducing structured parameterization to enable efficient computation over long sequences, while still attempting to capture long-range dependencies (Gu et al., 2021). Building on this, Mamba introduces a selective SSM that adapts to input dynamics, further improving processing of 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 long-range dependencies in time series data while while maintaining linear time complexity with respect to sequence length (Gu & Dao, 2023). Despite these advancements, RNNs and SSMs can still struggle with retaining information in extremely long sequences and may be prone to vanishing or exploding gradients
(Salehinejad et al., 2017). ROCKET offers an alternative, using random convolutional kernels to convert input into a fixed-length representation for VTSC, but at the cost of interpretability and potentially limited model expressivity (Dempster et al., 2020). SOZ localization methods. Several recent proposals have been tailored specifically to SOZ localization. Functional connectivity graphs compute patient-specific channel metrics to capture brain connectivity patterns
(Grattarola et al., 2022; Fang et al., 2024), offering insights into functional relationships associated with seizures. However, their reliance on intra-patient dynamics makes them unsuitable for a single model that can generalize across multi-patient, heterogeneous datasets. Alternatively, electrical stimulation methods that use intracranial electrodes (Johnson et al., 2022; Yang et al., 2024) can enhance localization accuracy through induced responses analyzed by TCNs and logistic regression models. Yet, these approaches require both fixed-length windows and the use of active stimulation. For our purpose of building a general model for SOZ localization, which can be applied to multiple patients (with a potentially varying number of channels)
without electrical stimulation, we do not consider such approaches in our study.

## 3 Method 3.1 Variable-Length Time Series Classification

Consider a collection of time series X =(x
(1)
t)
T1 t=1*, . . . ,*(x
(n)
t)
Tn t=1	with labels Y = {y
(1)*, . . . , y*(n)},
where each series i has sequence length Ti ∈ N, and for each time point t, the vector x
(i)
t ∈ R
Mi has Mi ∈ N
channels. The goal of VTSC is to learn a classifier fθ which maps each series (x
(i)
t)
Ti t=1 to its corresponding class in {1*, . . . , K*} for K ∈ N classes. We require that fθ can handle sequences of any length—that is, it has infinite context—since we assume that each Ti can be arbitrarily large at inference time. Otherwise, we must adjust a finite-context classifier using padding, truncation, or interpolation.

## 3.2 Stochastic Sparse Sampling 3.2.1 Sparse Training

Figure 1 provides an overview of SSS at train time. During each training epoch, SSS performs a sampling procedure without replacement to create each batch. Fix L ∈ N as the window size and let W be the collection of all windows with size L from all time series in X. Within any batch, a window is drawn from W, where the probability of it originating from series i is set to pi ≈ Ti/Pn j=1 Tj for every i. More formally, for each i, let Ni be the random variable representing the number of windows from series i in a batch of size B. Then Ni ∼ Binomial(*B, p*i), and consequently E[Ni] = Bpi. This proportional sampling ensures fair representation of each series based on its length, allowing longer sequences—which contain more information—to contribute more samples. By sampling only a subset of windows, SSS introduces sparsity into the training process, reducing computational cost found in finite-context methods, and the likelihood of context overload in infinite-context methods. Also note that by sampling with replacement, the model sees each window exactly once during a single training epoch.

After sampling a batch of windows W0 = {w1*, . . . ,* wB}, each wb ∈ W0 is processed independently by a local model fθ to obtain a local prediction yˆb = fθ(wb) ∈ [0, 1]K, representing our probability distribution over K ∈ N classes. The choice of fθ can be any time series backbone, in our experiments we select PatchTST (Nie et al., 2023). For each time series 1 ≤ i ≤ n, denote:
Wi = {w ∈ W0 | w is from series i}, (1)
4 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 as the collection of windows in the batch originating from series i, and let:
Yi = {fθ(w) | w ∈ Wi}, (2)
be the set of multiset of window probabilities. To obtain the global prediction for time series i, we aggregate window probabilities from all examples originating from it, given by:

$${\hat{y}}^{(i)}=\operatorname{Aggr}(\mathcal{Y}_{i})=\sum_{{\hat{y}}\in\mathcal{Y}_{i}}\alpha({\hat{y}}){\hat{y}},$$
$$(3)^{\frac{1}{2}}$$

where each P
α(ˆy) ∈ [0, 1] represents the weight of window probability yˆ to the final output, satisfying yˆ∈Yi α(ˆy) = 1; that is, Aggr(·) produces a convex combination over yˆ1*, . . . ,* yˆn. In our experiments, we use mean aggregation, i.e., α(ˆy) = 1 |Yi| for all yˆ ∈ Yi, due to its simplicity and effectiveness for our current objectives. This formulation guarantees that yˆ
(i)remains a valid probability distribution over K classes
(proof in Appendix A.3), and allows for potential of non-uniform aggregation functions, enabling weighting of window predictions based on factors such as prediction uncertainty, or frequency characteristics.

Algorithm 1 SSS Training Algorithm (Single Epoch)
Input: Time series X =(x
(1)
t)
T1 t=1*, . . . ,*(x
(n)
t)
Tn t=1	, labels Y = {y
(1)*, . . . , y*(n)}, model fθ with parameters θ, batch size B Output: Updated model parameters θ W ← Set of all windows from each series in X while W ̸= ∅ do
▷ Sample B windows with probability Ti/Pj Tj from series i, for all i W0 ← SAMPLE(W, B)
for i = 1*, . . . , n* do Wi ← {w ∈ W0 | w is from series i} ▷ Windows from series i Yj ← {fθ(w)| w ∈ Wj} ▷ Window probabilities for series i yˆ
(i) ← AGGREGATE(Yj ) ▷ Final probability for series i Lbatch ← 1n Pn i=1 L(ˆy
(i), y(i)) ▷ Loss over the batch θ ← UPDATE(θ,Lbatch) ▷ Update local model parameters W ← W \ W0 ▷ Remove sampled windows from the pool return θ To the derive the prediction for a time series (x
(i)
t)
Ti t=1 at inference time, we utilize all windows from the selected time series, to form its final prediction yˆ
(i). Let Wi be the collection of all windows from series i.

We pass each window through the local model to obtain the multiset of window probabilities Yi as shown in Equation (2). Before the aggregation step, we utilize a calibrator gϕ : [0, 1]K → [0, 1]K, which adjusts each individual window probability to reduce the presence of noise, and define:
Y˜i = {gϕ(ˆy) | yˆ ∈ Yi}, (4)
which is then fed into the final prediction during the aggregation step yˆ
(i) = Aggr(Y˜i). By calibrating the window probabilities before aggregation, we correct for biases or misestimations in the predicted probabilities, which occur when the output probabilities of the local models do not accurately reflect the true likelihood of the event. We consider isotonic regression and Venn-Abers predictors for our calibration method. It is important to note, that these calibration techniques do not alter underlying structure of fθ and do not utilize input features from the time series; rather, they adjust the output probabilities to mitigate the effect of noise during the aggregation step. For more information regarding calibration methods see Appendix A.4.

$${\hat{\mathcal{Y}}}_{i}=\{g_{\phi}({\hat{y}})\mid{\hat{y}}\in{\mathcal{Y}}_{i}\},$$
$\left(4\right)$. 

## 3.2.2 Inference 4 Experiments 4.1 Baselines

235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 For our finite-context baselines, we include a variety of modern time series backbones including PatchTST,
which uses subwindows as tokens in combination with the traditional Transformer architecture (Nie et al.,
2023; Vaswani et al., 2017). TimesNet is a TCN architecture that models both interperiod and intraperiod dynamics by leveraging Fast Fourier Transform (FFT) features to slice the signal into multiple views, which are then processed through inception blocks (Wu et al., 2022; Ismail Fawaz et al., 2020). ModernTCN, is a recently proposed TCN which decouples temporal and channel information processing by using separate DWConv and ConvFFN modules for more efficient representation learning (Luo & Wang, 2024). DLinear is linear neural network, which has shown to outperform several modern Transformer-based architecutres, utilizing traditional seasonal-trend decomposition techniques (Zeng et al., 2023). In our infinite-context baselines, we utilize ROCKET, which applies randomly initialized, fixed convolutional kernels to the input sequence. ROCKET compresses the resulting convolutional outputs to the maximum value and the proportion of positive values (PPV), where these features then fed to a linear classifier (Dempster et al., 2020). We also consider GRUs and an LSTM network, both of which are popular RNN frameworks designed to capture longterm dependencies in sequential data (Bahdanau et al., 2014; Hochreiter & Schmidhuber, 1997). Additionally, use the recent SSM architecture, Mamba, which utilizes selective state updates to enable efficient long-range dependency modeling (Gu & Dao, 2023). Further details regarding configurations and hyperparameter tuning for each baseline can be found in Appendix C.

## 4.2 Dataset

The Epilepsy iEEG Multicenter Dataset1consists of iEEG signals with SOZ clinical annotations from four medical centers including the Johns Hopkins Hospital (JHH), the National Institute of Health (NIH), University of Maryland Medical Center (UMMC), and University of Miami Jackson Memorial Hospital
(UMH). Since UMH contained only a single patient with clinical SOZ annotations, we did not consider it in our main evaluations; however, we did use UMH within the multicenter evaluation in 1 and the training set for OOD experiments for SOZ localization on unseen medical centers in Table 2. We select the F1 score, Area Under the Receiver Operator Curve (AUC), and accuracy for our evaluation metrics. For summary statistics and information on the dataset see Appendix B.1.

## 4.3 Univariate Vtsc

For each patient iEEG recording, the goal of SOZ localization is to determine the the correct of channels or electrodes which belong the seizure onset zone. This effectively reduces the task to univariate TSC.

While several channel-dependent methods have been proposed for SOZ localization (see section 2), we focus primarily on channel-independent solutions for two key reasons: (1) they are more resilient to interpatient variability and are unaffected by factors like channel count and therefore can be applied in multiple hospital settings, and (2) they generalize better to domains beyond electrophysiological data, as they learn local signal characteristics rather than explicitly modeling functional connectivity between electrode sites, which may not be present in other medical time series.

Table 1 summarizes our experimental results for SOZ localization on each individual medical center, along with training and evaluation on all medical centers. Within the multicenter evaluation, SSS outperforms all baselines for each evaluation metric. SSS also shows strong performance for the JHH and NIH centers, with comparable results in the UMMC center. We attribute this difference in performance for UMMC due to the fact that it is the only center where the sampling frequency of patient recordings can differ between patients 1https://openneuro.org/datasets/ds003029/versions/1.0.7 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328

| All                                          | JHH                                                   | NIH                                 | UMMC   |               |                |               |                |               |        |        |               |        |
|----------------------------------------------|-------------------------------------------------------|-------------------------------------|--------|---------------|----------------|---------------|----------------|---------------|--------|--------|---------------|--------|
| Model                                        | F1                                                    | AUC                                 | Acc.   | F1            | AUC            | Acc.          | F1             | AUC           | Acc.   | F1     | AUC           | Acc.   |
| SSS (Ours)                                   | 0.7629∗ 0.7999∗ 72.35∗ 0.8187∗ 0.8851∗ 81.37∗ 0.6716∗ | 0.6853 64.22† 0.7978† 0.8279† 76.06 |        |               |                |               |                |               |        |        |               |        |
| PatchTST (Nie et al., 2023)                  | 0.7097† 0.7852† 66.83                                 | 0.7419† 0.8045† 71.82               | 0.6402 | 0.7036† 62.11 | 0.8015∗ 0.8121 | 77.58         |                |               |        |        |               |        |
| TimesNet (Wu et al., 2022)                   | 0.6897                                                | 0.7174                              | 65.98  | 0.6891        | 0.8029         | 73.64† 0.5950 | 0.6806         | 66.00∗ 0.7821 | 0.8099 | 77.06† |               |        |
| ModernTCN (Luo & Wang, 2024)                 | 0.6938                                                | 0.7305                              | 68.42  | 0.6710        | 0.7508         | 67.73         | 0.5055         | 0.7220∗ 64.00 | 0.6371 | 0.8203 | 71.76         |        |
| DLinear (Zeng et al., 2023)                  | 0.6916                                                | 0.7044                              | 68.41  | 0.6873        | 0.7395         | 66.36         | 0.6055         | 0.6405        | 59.50  | 0.7658 | 0.7729        | 77.05  |
| ROCKET (Dempster et al., 2020)               | 0.6847                                                | 0.7481                              | 69.27  | 0.6753        | 0.7752         | 69.09         | 0.6520† 0.6546 | 62.63         | 0.7686 | 0.7900 | 74.55         |        |
| Mamba (Gu & Dao, 2023)                       | 0.6452                                                | 0.7134                              | 64.39  | 0.6456        | 0.6764         | 62.27         | 0.5974         | 0.6050        | 58.95  | 0.7900 | 0.8424∗ 76.36 |        |
| GRUs (Bahdanau et al., 2014)                 | 0.6948                                                | 0.7340                              | 65.85  | 0.6140        | 0.6959         | 63.18         | 0.6171         | 0.6283        | 62.63  | 0.7920 | 0.8211        | 77.27∗ |
| LSTM (Hochreiter & Schmidhuber, 1997) 0.6709 | 0.7144                                                | 65.43                               | 0.6571 | 0.6190        | 59.09          | 0.5657        | 0.5909         | 54.74         | 0.7604 | 0.8060 | 73.64         |        |

(250-1000 Hz), whereas JHH and NIH both have sampling frequencies of 1000 Hz. We also observe that in general, finite-context perform better on the chosen evaluation metrics in comparison to infinite-context methods, for in-distribution univariate VTSC.

## 4.4 Out-Of-Distribution Vtsc

Table 2 reports our results for SOZ localization in the OOD setting. At train time, from the collection of four medical center datasets, we omit one and train on the remaining three. At inference time, we test solely on the omitted medical center to gauge how well each method performs OOD. For iEEG signals from epilepsy patients, inter-patient variability can be significant due to differences in placement of electrodes in the brain, and the inherent heterogeneity of epileptogenic networks across individuals. Thus, even among medical time series, this can be one of the most challenging tasks to perform OOD. SSS outperforms each baseline on each unseen medical center, often by a considerable margin when compared to finite-context methods.

Table 2: Out-of-Distribution SOZ localization. F1 score, AUC, and Accuracy are reported for unseen medical centers, averaged over 5 seeds. For each center, we train on all other centers and evaluate on the selected center. **Bolded** values with ∗and † denote the best and second-best results, respectively.

Model F1 AUC Acc. F1 AUC Acc. F1 AUC Acc.

SSS (Ours) 0.6981∗0.6590∗57.80∗0.6492∗0.6092∗54.73†0.7243∗0.8048∗**72.42**∗

PatchTST (Nie et al., 2023) 0.6175 0.5267 50.46 0.5986 0.4829 48.17 0.5067 0.5274 57.63

TimesNet (Wu et al., 2022) 0.5261 0.4501 47.00 0.4461 0.4407 45.85 0.3177 0.3108 46.14

ModernTCN (Luo & Wang, 2024) 0.4934 0.4970 49.54 0.4019 0.4651 48.71 0.3804 0.4474 50.55 DLinear (Zeng et al., 2023) 0.4205 0.4775 47.25 0.5090 0.4945 50.54 0.5602 0.5236 56.00

ROCKET (Dempster et al., 2020) 0.5784 0.5777 **56.71**†0.5051 0.5522 52.91 0.5608 0.5941 58.36 Mamba (Gu & Dao, 2023) 0.5790 **0.5835**†55.68 0.6183†0.5767†**55.69**∗0.5715 0.5953 55.76 GRUs (Bahdanau et al., 2014) 0.5779 0.4868 48.80 0.5824 0.5588 53.66 0.6689†0.7645†**69.30**† LSTM (Hochreiter & Schmidhuber, 1997) **0.6362**†0.5165 50.92 0.5774 0.5678 53.87 0.6581 0.6616 62.36

7

JHH NIH UMMC

329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375

## 4.5 Qualitative Visualization

8 In comparison to Table 1, we observe the opposite trend between finite- and infinite-context methods, whereas infinite-context methods seem to perform better OOD. In contrast, SSS demonstrates robust performance both in distribution (where finite-context methods excel) and OOD (where infinite-context methods excel).

376 377 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422

## 5 Discussion 5.1 Review Of Results

In general, we observe that SSS outperforms modern finite-context and infinite-context methods both in-distribution and OOD for univariate VTSC. SOZ localization presents a significant challenge due to intra-patient and inter-patient variability, and with our evaluation the collection of 3 heterogeneous datasets, serving rigorous testbed for assessing the generalizability of SSS's capabilities for learning local signal characteristics in medical time series. While our experiments focus on univariate VTSC, as outlined in section 3.2.1 and 3.2.2, SSS can be easily applied to the multivariate setting. Our results from the multicenter evaluation Table 1 indicate that SSS benefits from a diversity of data distributions and volume of training examples, given by the magnitude in performance differences, when compared to single cluster results. Furthermore, our OOD experiments from Table 2 suggest that SSS may be learning local signal characteristics present in different patient populations, leading to a advantage over finite-context and infinite-context methods. Our visualizations of SSS's predictions OOD in Figure 3 supports this notion, as there exist clear qualitative differences in locally averaged window probabilities with respect to anomalous signal characteristics, such as spikes or increases in amplitude or frequency. Figure 2 shows SSS's predictions in-distribution which also suggest a form of implicit semantic segmentation for anomalous local regions of the signal with respect to the SOZ probability. More analysis is needed to further solidify our understanding, which would benefit from a rigorous explainability study in future works.

## 5.2 Conclusion

To conclude, this work introduces novel VTSC framework, Stochastic Sparse Sampling (SSS), specifically tailored for medical time series applications. SSS blends the best of both worlds between finite-context methods (enabling usage of finite-context backbones) while allowing sampling of the entire signal in a computationally efficient manner that is less prone to context overload from infinite-context methods. SSS learns local signal characteristics, which provides the added benefit of inherent interpretability, and provides superior performance to the SOTA in-distritbuion and OOD for unseen medical centers. For future work, it would be valuable to: (1) benchmark SSS across a wider variety of variable-length medical time series, (2)
provide further rigorous post-hoc insights into the window probability distribution given by SSS, and (3) potentially include uncertainty estimates within the aggregation function to highlight anomalous regions of the signal.

## References

423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. *Neural computation*, 9(8):1735–1780, 1997. Julia Amann, Alessandro Blasimme, Effy Vayena, Dietmar Frey, Vince I Madai, and Precise4Q Consortium. Explainability for artificial intelligence in healthcare: a multidisciplinary perspective. *BMC medical informatics and decision making*, 20:1–9, 2020.

Anastasios N Angelopoulos and Stephen Bates. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. *arXiv preprint arXiv:2107.07511*, 2021.

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. *arXiv preprint arXiv:1409.0473*, 2014.

Shaojie Bai, J Zico Kolter, and Vladlen Koltun. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv preprint arXiv:1803.01271*, 2018.

Sai Sanjay Balaji and Keshab K Parhi. Seizure onset zone identification from ieeg: A review. *IEEE Access*, 10:
62535–62547, 2022.

Si-An Chen, Chun-Liang Li, Nate Yoder, Sercan O Arik, and Tomas Pfister. Tsmixer: An all-mlp architecture for time series forecasting. *arXiv preprint arXiv:2303.06053*, 2023.

Jonathan Crabbé and Mihaela Van Der Schaar. Explaining time series predictions with dynamic masks. In *International* Conference on Machine Learning, pp. 2166–2177. PMLR, 2021.

Angus Dempster, François Petitjean, and Geoffrey I Webb. Rocket: exceptionally fast and accurate time series classification using random convolutional kernels. *Data Mining and Knowledge Discovery*, 34(5):1454–1495, 2020.

T Deutsch, ED Lehmann, ER Carson, AV Roudsari, KD Hopkins, and PH Sönksen. Time series analysis and control of blood glucose levels in diabetic patients. *Computer Methods and Programs in Biomedicine*, 41(3-4):167–182, 1994.

Joseph Early, Gavin KC Cheung, Kurt Cutajar, Hanting Xie, Jas Kandola, and Niall Twomey. Inherently interpretable time series classification via multiple instance learning. In The Twelfth International Conference on Learning Representations, 2024.

Chunying Fang, Xingyu Li, Meng Na, Wenhao Jiang, Yuankun He, Aowei Wei, Jie Huang, and Ming Zhou. Epilepsy lesion localization method based on brain function network. *Frontiers in Human Neuroscience*, 18:1431153, 2024.

Daniele Grattarola, Lorenzo Livi, Cesare Alippi, Richard Wennberg, and Taufik A Valiante. Seizure localisation with attention-based graph neural networks. *Expert systems with applications*, 203:117330, 2022.

Qi Huang, Wei Chen, Thomas Bäck, and Niki van Stein. Shapelet-based model-agnostic counterfactual local explanations for time series classification. *arXiv preprint arXiv:2402.01343*, 2024.

Elena Agliari, Adriano Barra, Orazio Antonio Barra, Alberto Fachechi, Lorenzo Franceschi Vento, and Luciano Moretti.

Detecting cardiac pathologies via machine learning on heart-rate variability time series and related markers. *Scientific* Reports, 10(1):8845, 2020.

Hassan Ismail Fawaz, Germain Forestier, Jonathan Weber, Lhassane Idoumghar, and Pierre-Alain Muller. Deep learning for time series classification: a review. *Data mining and knowledge discovery*, 33(4):917–963, 2019.

Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces. *arXiv* preprint arXiv:2111.00396, 2021.

Hassan Ismail Fawaz, Benjamin Lucas, Germain Forestier, Charlotte Pelletier, Daniel F Schmidt, Jonathan Weber, Geoffrey I Webb, Lhassane Idoumghar, Pierre-Alain Muller, and François Petitjean. Inceptiontime: Finding alexnet for time series classification. *Data Mining and Knowledge Discovery*, 34(6):1936–1962, 2020.

Graham W Johnson, Leon Y Cai, Derek J Doss, Jasmine W Jiang, Aarushi S Negi, Saramati Narasimhan, Danika L Paulo, Hernán FJ González, Shawniqua Williams Roberson, Sarah K Bick, et al. Localizing seizure onset zones in surgical epilepsy with neurostimulation deep learning. *Journal of neurosurgery*, 138(4):1002–1007, 2022.

Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi, and Jaegul Choo. Reversible instance normalization for accurate time-series forecasting against distribution shift. In International Conference on Learning Representations, 2021.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*, 2014. Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=rkgNKkHtvB.

Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long-and short-term temporal patterns with deep neural networks. In The 41st international ACM SIGIR conference on research & development in information retrieval, pp. 95–104, 2018.

Adam Li, Chester Huynh, Zachary Fitzgerald, Iahn Cajigas, Damian Brusko, Jonathan Jagid, Angel O Claudio, Andres M
Kanner, Jennifer Hopp, Stephanie Chen, et al. Neural fragility as an eeg marker of the seizure onset zone. *Nature* neuroscience, 24(10):1465–1474, 2021.

Shizhan Liu, Hang Yu, Cong Liao, Jianguo Li, Weiyao Lin, Alex X Liu, and Schahram Dustdar. Pyraformer: Lowcomplexity pyramidal attention for long-range time series modeling and forecasting. In International Conference on Learning Representations, 2022.

Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long. itransformer: Inverted transformers are effective for time series forecasting. In *International Conference on Learning Representations*, 2024.

Wolfgang Löscher, Heidrun Potschka, Sanjay M Sisodiya, and Annamaria Vezzani. Drug resistance in epilepsy: clinical impact, potential mechanisms, and new innovative treatment options. *Pharmacological reviews*, 72(3):606–638, 2020.

Donghao Luo and Xue Wang. Moderntcn: A modern pure convolution structure for general time series analysis. In The Twelfth International Conference on Learning Representations, 2024.

470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 Navid Mohammadi Foumani, Lynn Miller, Chang Wei Tan, Geoffrey I Webb, Germain Forestier, and Mahsa Salehi. Deep learning for time series classification and extrinsic regression: A current survey. *ACM Computing Surveys*, 56(9):1–45, 2024.

Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In *International Conference on Learning Representations*, 2023.

World Health Organization et al. *Epilepsy: a public health imperative*. World Health Organization, 2019. Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library.

Advances in neural information processing systems, 32, 2019.

Pranav Rajpurkar, Emma Chen, Oishi Banerjee, and Eric J Topol. Ai in health and medicine. *Nature medicine*, 28(1):
31–38, 2022.

David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning internal representations by error propagation, parallel distributed processing, explorations in the microstructure of cognition, ed. de rumelhart and j. mcclelland. vol.

1. 1986. *Biometrika*, 71(599-607):6, 1986.

Hojjat Salehinejad, Sharan Sankar, Joseph Barfett, Errol Colak, and Shahrokh Valaee. Recent advances in recurrent neural networks. *arXiv preprint arXiv:1801.01078*, 2017.

517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 Mervyn J Silvapulle and Pranab Kumar Sen. *Constrained statistical inference: Order, inequality, and shape constraints*.

John Wiley & Sons, 2011.

Carl E Stafstrom and Lionel Carmant. Seizures and epilepsy: an overview for neuroscientists. Cold Spring Harbor perspectives in medicine, 5(6):a022426, 2015.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. *Advances in neural information processing systems*, 30, 2017.

Vladimir Vovk and Ivan Petej. Venn-abers predictors. In *Proceedings of the Thirtieth Conference on Uncertainty in* Artificial Intelligence, pp. 829–838, 2014.

Dominik Walther, Johannes Viehweg, Jens Haueisen, and Patrick Mäder. A systematic comparison of deep learning methods for eeg time series analysis. *Frontiers in Neuroinformatics*, 17:1067095, 2023.

Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with autocorrelation for long-term series forecasting. *Advances in Neural Information Processing Systems*, 34:22419–22430, 2021.

Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long. Timesnet: Temporal 2d-variation modeling for general time series analysis. *arXiv preprint arXiv:2210.02186*, 2022.

Bowen Yang, Baotian Zhao, Chao Li, Jiajie Mo, Zhihao Guo, Zilin Li, Yuan Yao, Xiuliang Fan, Du Cai, Lin Sang, et al. Localizing seizure onset zone by a cortico-cortical evoked potentials-based machine learning approach in focal epilepsy. *Clinical Neurophysiology*, 158:103–113, 2024.

Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting? 2023. Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. In *Proceedings of the AAAI conference on artificial* intelligence, volume 35, pp. 11106–11115, 2021.

Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. In Proc. 39th International Conference on Machine Learning
(ICML 2022), 2022.

564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610

## A Stochastic Sparse Sampling A.1 Sampling Implementation

Let X =(x
(1)
t)
T1 t=1*, . . . ,*(x
(n)
t)
Tn t=1	be the collection of variable-length time series. To achieve the sampling procedure outlined in section 3.2, we first construct the set all windows W by performing the slicing window method over each individual time series in X. For a window size L, with window stride S, and a time series i with sequence length Ti, we obtain Ai = ⌊
Ti−L
S⌋ + 1 windows. Note that Ai ∝ Ti for each i.

During training, we convert W to a PyTorch dataset and use the native PyTorch dataloader with batch size B. When a batch is sampled, windows are drawn uniformly from W, and thus for each i, the probability of observing a window from series i is pi = Ai/(Pn j=1 Aj ) ≈ Ti/(Pn j=1 Tj ). If Ni represents the number of windows in the batch from series i, then we achieve the desired sampling property of Ni ∼ Binomial(*B, p*i)
where pi ≈ Ti/(Pn j=1 Tj ). Note that this procedure uses sampling *without* replacement; one may consider replacement, however, we did not experiment with this and leave modifications with more complex sampling procedures as a future direction.

## A.2 Ablations A.2.1 Batch Size

Table 3 reports mean F1 score, AUC, and accuracy
(%) with standard deviations, are reported over 5 seeds, for the evaluation on all medical centers. Each experiment uses the best configuration described in Table 6. While the aggregation function in Equation (3) may benefit from a higher number of samples within the batch (due to mean approximation), we observe that the performance of SSS remains relatively constant across various batch sizes, and thus does not require large batch sizes to achieve adequate performance.

## A.2.2 Window Size

Table 3: Performance of SSS over various batch sizes.

| Batch Size   | F1 Score         | AUC              | Acc. (%)      |
|--------------|------------------|------------------|---------------|
| 128          | 0.7563 ± 0.02717 | 0.8139 ± 0.04302 | 70.79 ± 2.323 |
| 512          | 0.7498 ± 0.02354 | 0.7965 ± 0.03526 | 69.46 ± 3.109 |
| 2048         | 0.7651 ± 0.03568 | 0.8194 ± 0.05166 | 71.40 ± 6.078 |
| 4096         | 0.7441 ± 0.02987 | 0.7979 ± 0.06818 | 68.09 ± 4.848 |
| 8192         | 0.7629 ± 0.02829 | 0.7999 ± 0.05331 | 72.35 ± 4.965 |

Table 4 follows the same experimental setup as Table 3, but varies over the window size L instead of batch size. While the performance of L = 512 and L =
1024 remain relatively on par, we notice that the F1 score drops significantly for L = 2048 along with all other metrics. This suggests that a large receptive field may not be advantageous, and that SSS benefits from processing localized areas of the signal. Indeed, as L increases we expect to reach a similar performance to the finite-context PatchTST baseline, with a decrease in performance as a result.

Table 4: Performance of SSS over various window sizes L.

L **F1 Score AUC Acc. (%)** 512 0.7567 ± 0.01075 0.8141 ± 0.03054 70.62 ± 2.165 1024 0.7629 ± 0.02829 0.7999 ± 0.05331 72.35 ± 4.965 2048 0.7334 ± 0.03003 0.7719 ± 0.04762 68.52 ± 3.353

## A.2.3 Calibration

Table 4 follows the same experimental setup as Table 3, but varies over the window size L instead of batch size. While the performance of L = 512 and L = 1024 remain relatively on par, we notice that the F1 score drops significantly for L = 2048 along with all other metrics. This suggests that a large receptive field 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657

## A.4 Calibration

 ${j=1}$  ${=\sum_{i=1}^n\sum_{j=1}^K\alpha_i v_{ij}}$  ${=\sum_{i=1}^n\alpha_i\sum_{j=1}^K v_{ij}}$  ${=n}$

$$=\sum_{i=1}^{n}\alpha_{i}$$ $$=1$$
n
$\mathbf{r}$). 
αi (PK
$$(\sum_{j=1}^{K}v_{i j}=1{\mathrm{~for~all~}}i)$$
= 1 (Pn
$$(\sum_{i=1}^{n}\alpha_{i}=1)$$
Let yˆ1*, . . . ,* yˆn ∈ [0, 1] be the uncalibrated window probabilities, each with a corresponding binary label y1, . . . , yn ∈ {0, 1} derived from the label of the time series; that is, if yˆi = fθ(wi) for a window wi, then the window label yiis inherited from the global time series it was sampled from. The goal of probability calibration is to transform each yˆi into y˜i = gϕ(ˆyi), such that y˜i represents true likelihood of a class. Within this context, probability calibration can help mitigate the impact of temporal fluctuations and local anomalies by adjusting probabilities for each individual windows. Note that while Table 5: Performance of SSS with different calibration methods.

| Calibration Method   | F1 Score         | AUC              | Acc. (%)      |
|----------------------|------------------|------------------|---------------|
| Isotonic Regression  | 0.7629 ± 0.02829 | 0.7999 ± 0.05331 | 72.35 ± 4.965 |
| Venn-ABERS           | 0.7637 ± 0.02704 | 0.8003 ± 0.05308 | 72.47 ± 4.773 |
| No Calibration       | 0.7291 ± 0.06909 | 0.7830 ± 0.03844 | 69.93 ± 4.635 |

Theorem 1 (Probability Distribution Guarantee). Fix K, n ∈ N. Suppose α1, . . . , αn ≥ 0 *satisfies* Pn i=1 αi = 1, and v1, . . . , vn ∈ [0, 1]K *each satisfy* PK
j=1 vik = 1 for 1 ≤ i ≤ n*. That is, each* vi = (vi1, vi2*, . . . , v*iK)
Trepresents a valid discrete probability distribution over K *classes. Then the convex* combination:

$$\mathbf{y}=\sum_{i=1}^{n}\alpha_{i}\,\mathbf{v}_{i},\tag{1}$$
$$(S)$$

also represents a valid discrete probability distribution, satisfying PK
j=1 yj = 1.

Proof. By construction, yj =Pn i=1 αivij for each entry 1 ≤ j ≤ K. Then yj ≥ 0, since for each 1 ≤ i ≤ n and 1 ≤ j ≤ K we are given that αi ≥ 0 and vij ≥ 0. Furthermore, we can write:

$$\sum_{j=1}^{n}y_{j}=\sum_{j=1}^{K}\sum_{i=1}^{n}\alpha_{i}v_{i j}$$

αivij (Swap summation order)
It follows that since each yj ≥ 0 and PK
j=1 yj = 1, then y represents a valid discrete probability distribution over K classes.

## A.3 Convex Aggregation

may not be advantageous, and that SSS benefits from processing localized areas of the signal. Indeed, as L increases we expect to reach a similar performance to the finite-context PatchTST baseline, with a decrease in performance as a result.

658 659 660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 calibration may yield more refined probability estimates with reduced noise, it is an integral intermediate step rather than a global optimization procedure on the final probabilities. For each calibration we considered, we provide a short description below. Isotonic regression is a nonparametric method that fits a weighted least-squares model subject to motonicity constraints (Silvapulle & Sen, 2011). Formally, this can be stated as a quadratic program (QP) given by:

$$\min_{g}\sum_{i=1}^{n}w_{i}(\tilde{y}_{i}-y_{i})^{2}\ \text{subject to}\ \tilde{y}_{i}\leq\tilde{y}_{j}\ \text{for all}\ i,j\ \text{where}\ \hat{y}_{i}\leq\hat{y}_{j}.\tag{1}$$
$$(6)$$

where y˜i = g(ˆyi) and wi ≥ 0 are weights assigned to each datapoint, which are often each set to wi = 1 to provide equal importance over all inputs. The monotonicity constraint ensures that uncalibrated probabilities will always map to equal or higher calibrated probabilities. Due its nonparametric nature, isotonic regression can adapt to various probability distributions across diverse datasets. However, this flexibility comes at a cost as it is also prone to overfitting on smaller datasets, potentially adjusting to noise rather than properly calibrating its inputs.

Venn-Abers predictors is based on the concept of isotonic regression but extends it to ensure validity within the framework of conformal prediction, which provides uncertainty estimates with distribution-free theoretical guarantees (Vovk & Petej, 2014; Angelopoulos & Bates, 2021). For an uncalibrated probability yˆ, two isotonic calibrators are trained:
p0 = g0(ˆy) and p1 = g1(ˆy) (7)
where g0 and g1 are isotonic functions derived from augmented sets. These sets include (ˆy, 0) and (ˆy, 1)
respectively, alongside all other uncalibrated probabilities and their respective labels. The values p0 and p1 represent likelihoods for class 0 and class 1, while the interval [p0, p1] provides an uncertainty estimate of where the true probability resides. The final calibrated probability is then given by:

$$p_{0}=g_{0}({\hat{y}}){\mathrm{~and~}}p_{1}=g_{1}({\hat{y}})$$
$$\left(7\right)$$
$$\bar{y}=\frac{p_{1}}{1-p_{0}+p_{1}}.$$
$$({\boldsymbol{\delta}})$$

. (8)
Venn-Abers predictors provide guaranteed validity in terms of calibration, meaning the predicted probabilities closely match empirical frequencies. While we do not explicitly utilize the uncertainty interval [p0, p1] (only the calibrated score y˜), this method can be effective in scenarios requiring risk assessment or critical tasks where rigorous uncertainty estimates are crucial. We leave this as a future direction to implement conformal prediction within the context of SSS, to provide uncertainty guarantees based off of window predictions, which may be useful for post-hoc interpretability. In comparsion to isotonic regression, Venn-Abers can be more computationally intensive, as it requires fitting two isotonic functions simultaneously.

## A.5 Finite-Context & Infinite-Context Methods

Definition 2. Let X be a vector space over R and fθ : *X → Y* be a model with parameters θ and output space Y. We say that fθ has **finite-context** if X is finite-dimensional, that is, there exists some n ∈ N such that X ∼= R
nas vector spaces. Whereas fθ is said to have **infinite-context** if X = R
(∞)is the space of real number sequences with finite support2. Note that this definition refers to the *native* capabilities of fθ, without the usage of data manipulation techniques such as padding, truncation, and interpolation. We utilize this formalization to separate our baselines, so that we may better understand the advantages and limitations of both.

705 706 707 708 709 710 711 712 713 714 715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751

| Parameter      | Search Values                               |
|----------------|---------------------------------------------|
| dmodel         | {16, 32, 64}                                |
| dff            | {32, 64, 128}                               |
| num_heads      | {2, 4, 8}                                   |
| num_enc_layers | {1, 2, 3}                                   |
| lr             | {10−4 , 10−5}                               |
| L              | {512, 1024, 2048}                           |
| batch_size     | {2048, 4096, 8192}                          |
| gϕ             | {isotonic regression,Venn-Abers predictors} |

A.6 SSS IMPLEMENTATION AND CONFIGURATIONS

## B Dataset And Preprocessing B.1 Dataset

Table 7: iEEG Multicenter Dataset Summary: For each medical center, we report the total number of patients recorded (n), the number of patients with seizure onset zone (SOZ) annotations (nSOZ), the number of time series recordings (nts), the percentage of time series labeled as SOZ (pSOZ), the type of iEEG method used
(e.g., electrocorticography, ECoG), the sampling frequency (Hz) (noting that some recordings may vary), and post-operative patient outcomes following SOZ surgical resection.

Medical Center n nSOZ nts pSOZ **iEEG Type Frequency (Hz) Patient Outcomes**

JHH 7 3 1458 7.48% ECoG 1000 No NIH 14 11 3057 12.23% ECoG 1000 Yes UMMC 9 9 2967 5.56% ECoG 250-1000 Yes UMF 5 1 129 25.58% ECoG 1000 No

Table 7 provides an overview of the iEEG Multicenter Dataset. For each cluster, we filter out patients (n)
who have SOZ annotations (nSOZ). All channels for all patients are group together into one dataset per medical center, where nts indicates the number of examples. However, due to the heavy imbalance between SOZ-labeled time series and non-SOZ labeled time series, the number of examples used for training and validation decreases significantly once we employ class balancing, resulting in ⌊2 · pSOZ · nts⌋ examples for each medical center, which is then split into training, validation, and testing.

## B.2 Data Preprocessing

Each patient recording contains multiple channels corresponding to individual electrodes from the iEEG device. During preprocessing, for all patients, we extract all channels and balance the dataset to have an equal number of SOZ and non-SOZ channels. After, we partition this dataset into train, validation, and test channels with a 70%/10%/20% split respectively, and ensure that during the window sampling phase of SSS there is no temporal leakage from the test set. Each channel, or univariate time series, is z−score normalized to have zero mean and unit standard deviation.

2Every sequence in R
(∞) must have finitely many non-zero terms.

Table 6: Hyperparameter search space for SSS (with PatchTST backbone). Best configuration is highlighted in red.

752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769 770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798

$$\mathbf{1}\quad\mathbf{F}\mathbf{1}$$

B.3.1 F1 The F1 score is defined as the harmonic mean of precision and recall, given by:

$$\operatorname{RIC}\mathbf{S}$$
$$F_{1}=2\cdot{\frac{\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}},$$
$$(9)$$

 $\;$ True Positives (TP). 
$$(10)$$

$$\overline{{+\mathrm{~Fa~}}}$$
$$(11)$$

, (9)
$$\mathrm{Recall}={\frac{\mathrm{True~Positives~(TP)}}{\mathrm{True~Positives~(TP)}+\mathrm{False~Negatives~(FN)}}}.$$
. (11)
We select the F1 score as our primary evaluation metric for SOZ localization for the following reasons.

The F1 score balances the need to correctly identify all regions of the SOZ (recall) with the need to avoid misclassifying healthy regions as SOZ (precision). This balance is crucial in surgical planning, where both missing SOZs and unnecessarily removing healthy tissue can have severe consequences. Unlike accuracy, which overlooks the difference between false positives and false negatives, the F1 score provides a more nuanced evaluation by considering both, making it well-suited in clinical contexts such as SOZ localization.

## B.3.2 Auc

We complement the F1 score with the Area Under the Receiver Operating Characteristic curve (AUC), defined by:

$${\mathrm{AUC}}=\int_{0}^{1}{\mathrm{TPR}}(t)\cdot{\frac{d{\mathrm{FPR}}(t)}{d t}}d t$$
$$(12)$$

## C Implementation And Experimental Configurations

For each baseline, we perform grid search and optimize with respect to best accuracy score on the evaluation for all medical centers. L refers to the window size parameter, dmodel is the model dimension, and dff B.3 EVALUATION METRICS Due to the extremely long sequence length of several channels, we required downsampling to fit the dataset into memory (250GB RAM). To achieve this, for each channel we applied a 1D average pooling layer with kernel_size=24 and kernel_stride=12 before feeding it to the baseline model or before performing the window sampling procedure for SSS at train-time. Finite-context methods required either padding, truncation, or interpolation due to fit each sequence into its limited context window. For each finite-context method we perform a combination of padding and truncation according to the chosen window size L: if the sequence length of the original time series exceeded L, it was truncated to obtain the first L time points, otherwise if it was less than L, it was padded with zeros at the end of the sequence to ensure a sequence length of L.

Precision =True Positives (TP)
True Positives (TP) + False Positives (FP)
, (10)
where, and, While the F1 score provides insight into the balance between precision and recall at a specific threshold, AUC assesses the model's overall discriminative ability across all thresholds. This threshold-independent evaluation is relevant for critical scenarios where the threshold maybe be adjusted from 0.5, which is not common in clinical settings.

799 800 801 802 803 804 805 806 807 808 809 810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845

| Parameter      | Search Values             |
|----------------|---------------------------|
| dmodel         | {16, 32, 64}              |
| dff            | {32, 64, 128}             |
| num_heads      | {2, 4, 8}                 |
| num_enc_layers | {1, 2, 3}                 |
| lr             | {10−4 , 10−5}             |
| L              | {1000, 3000, 5000, 10000} |

TimesNet: We use the official implementation github.com/thuml/Time-Series-Library.

Table 9: Hyperparameter search space for TimesNet. Best configuration is highlighted in red.

| Parameter      | Search Values             |
|----------------|---------------------------|
| dmodel         | {16, 32, 64}              |
| dff            | {32, 64, 128}             |
| num_kernels    | {4, 6}                    |
| top_k          | {3, 5}                    |
| num_enc_layers | {1, 2}                    |
| lr             | {10−4 , 10−5}             |
| L              | {1000, 3000, 5000, 10000} |

is the dimension of the feed-forward network. The grid search parameters for each baseline are shown below; for information on the implementation of SSS, see Appendix A.6. In all experiments, we train using the Adam optimizer (Kingma & Ba, 2014), for 50 epochs, with cosine learning rate annealing (one cycle with 50 epochs in length) which adjusts the learning rate down by two orders of magnitude (e.g.,
10−4to 10−6) by the last epoch. We also implement early stopping with a patience of 15, and apply learnable instance normalization (Kim et al., 2021) for each input. For most of the baselines we use a dropout rate of 0.2 − 0.3, and weight decay to 10−4 − 10−5, but do not explicitly tune these parameters in our grid search. For finite-context methods we set the batch size to the entire dataset (596 individual univariate time series for all clusters), whereas infinite-context methods required batch size of 1 due to their variable-length. The code for SSS and baseline implementations is available at the following anonymous link: https://anonymous.4open.science/r/sss-0D75/.

PatchTST: We adapt the official implementation github.com/yuqinie98/PatchTST, but swap out the attention module with the native PyTorch torch.nn.MultiheadAttention module.

ModernTCN: We use the official implementation github.com/luodhhh/ModernTCN. DLinear: We use the implementation from github.com/thuml/Time-Series-Library.

Table 11: Hyperparameter search space for DLinear. Best configuration is highlighted in red.

846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 880 881 882 883 884 885 886 887 888 889 890 891 892 also employ patching from (Nie et al., 2023), which we observed led to greater to performance, with patch_size = 64 and patch_stride = 16. GRUs: We utilized the native PyTorch module torch.nn.GRU.

| Parameter   | Search Values             |
|-------------|---------------------------|
| moving_avg  | {10, 25}                  |
| lr          | {10−4 , 10−5 , 10−6}      |
| L           | {1000, 3000, 5000, 10000} |

ROCKET: We use the official implementation github.com/angus924/rocket and follow the standard implementation of 10, 000 kernels.

| Parameter   | Search Values        |
|-------------|----------------------|
| num_kernels | {10000}              |
| lr          | {10−4 , 10−5 , 10−6} |

Table 12: Hyperparameter search space for ROCKET. Best configuration is highlighted in red.

Mamba: We use the package mambapy which builds upon the official Mamba implementation. We

| Parameter      | Search Values        |
|----------------|----------------------|
| lr             | {10−4 , 10−5 , 10−6} |
| dmodel         | {16, 32, 64}         |
| num_enc_layers | {1, 2, 3}            |

Table 13: Hyperparameter search space for Mamba. Best configuration is highlighted in red.

Table 10: Hyperparameter search space for ModernTCN. Best configuration is highlighted in red.

| Parameter         | Search Values             |
|-------------------|---------------------------|
| lr                | {10−4 , 10−5}             |
| dmodel            | {16, 32, 64}              |
| num_enc_layers    | {1, 2}                    |
| large_size_kernel | {9, 13, 21, 51}           |
| small_size_kernel | 5                         |
| dw_dims           | {128, 256}                |
| ffn_ratio         | {1, 4}                    |
| L                 | {1000, 3000, 5000, 10000} |

## C.1 Computational Resources

893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 936 937 938 939 Our experiments were conducted on 4x NVIDIA RTX 6000 Ada Generation GPUs using the PyTorch framework with CUDA Paszke et al. (2019). Although we have not tracked explicitly the amount of consumed GPU hours, all experiments can be conducted for 5 seeds no more than 2 − 3 hours with a similar setup.

Table 14: Hyperparameter search space for GRUs. Best configuration is highlighted in red.

| Parameter      | Search Values        |
|----------------|----------------------|
| lr             | {10−4 , 10−5 , 10−6} |
| dmodel         | {16, 32, 64}         |
| num_enc_layers | {1, 2, 3}            |
| bidirectional  | {True, False}        |