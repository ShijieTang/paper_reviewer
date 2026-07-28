**006**

**029**

**034**

# STOCHASTIC SPARSE SAMPLING: A FRAMEWORK FOR VARIABLE-LENGTH MEDICAL TIME SERIES CLASSIFICA-TION

Anonymous authors Paper under double-blind review

# ABSTRACT

While the majority of time series classification research has focused on modeling fixedlength sequences, variable-length time series classification (VTSC) remains critical in healthcare, where sequence length may vary among patients and events. To address this challenge, we propose Stochastic Sparse Sampling (SSS), a novel VTSC framework developed for medical time series. SSS manages variable-length sequences by sparsely sampling fixed windows to compute local predictions, which are then aggregated and calibrated to form a global prediction. We apply SSS to the task of seizure onset zone (SOZ) localization, a critical VTSC problem requiring identification of seizure-inducing brain regions from variable-length electrophysiological time series. We evaluate our method on the Epilepsy iEEG Multicenter Dataset, a heterogeneous collection of intracranial electroencephalography (iEEG) recordings obtained from four independent medical centers. SSS demonstrates superior performance compared to state-of-the-art (SOTA) baselines across most medical centers, and superior performance on all out-of-distribution (OOD) unseen medical centers. Additionally, SSS naturally provides post-hoc insights into local signal characteristics related to the SOZ, by visualizing temporally averaged local predictions throughout the signal.

# 1 INTRODUCTION

Artificial intelligence (AI) in medicine has received significant attention in recent years, with various applications to clinical diagnosis and treatment planning [\(Rajpurkar et al., 2022\)](#page-10-0). Despite its advancements, the actual integration into everyday clinical practice remains limited, with much of it attributed to the challenges of handling the complexity and variability in medical data. One particularly challenging aspect of this variability lies in the nature of medical time series data. Variable-length time series are prevalent throughout many areas of healthcare, including heart rate monitoring, blood glucose measurements, and electrophysiological recordings where sequence length can vary dependent on the recording or length of an event [\(Agliari et al., 2020;](#page-9-0) [Deutsch et al., 1994;](#page-9-1) [Walther et al., 2023\)](#page-11-0). Yet, the majority of time series classification (TSC) literature focuses solely on methods that process fixed-length sequences [\(Ismail Fawaz](#page-9-2) [et al., 2019;](#page-9-2) [Mohammadi Foumani et al., 2024\)](#page-10-1).

At the same time, healthcare applications require greater interpretability from modern time series methods to expand their applicability in critical domains and accelerate clinical adoption [\(Amann et al., 2020\)](#page-9-3). This interpretability is especially crucial in contexts where the relationship between pathology and signal characteristics is not well understood, as it can provide valuable insights for both clinicians and scientists. Recent studies in time series classification (TSC) have explored the explainability of specific signal segments, as opposed to full-signal analysis, which proves particularly useful for uncovering important characteristics

**054 055 056**

**059**

**061**

**063 064**

**067**

**069**

**079**

**081**

**084**

such as motifs, anomalies, or frequency patterns [\(Early et al., 2024;](#page-9-4) [Crabbé & Van Der Schaar, 2021;](#page-9-5) [Huang](#page-9-6) [et al., 2024\)](#page-9-6). However, there still remains a significant need for models with built-in interpretability in medical applications. Such methods would allevaiate the burden of implementing both a base model and a specialized interpretability method—which may require more domain expertise—and may more effectively facilitate clinical adoption.

The need for variable-length time series classification (VSTC) methods with built-in interpretability is particularly relevant in seizure onset zone (SOZ) localization—the task of identifying brain regions from which seizures originate—as effective treatment requires analysis of variable-length signals [\(Balaji & Parhi,](#page-9-7) [2022\)](#page-9-7). The World Health Organization (WHO) reports epilepsy affects over 50 million people globally, establishing it as one of the most common yet poorly understood neurological disorders [\(Organization et al.,](#page-10-2) [2019;](#page-10-2) [Stafstrom & Carmant, 2015\)](#page-11-1). Additionally, one-third of patients do not respond to antiepileptic drugs, making surgery the last resort and accurate SOZ localization essential for effectively planning the operation. The process of SOZ identification involves a two-step procedure: initial implantation of electrodes in areas suspected to contain the SOZ, followed by recording and visual analysis of intracranial electroencephalography (iEEG) signals by medical experts. The task of SOZ localization reduces to classifying individual electrode recordings, representing different regions within the brain. Effective localization of the SOZ is challenging due to the absence of clinically validated biological markers and the variable-length nature of iEEG signals—consequently, surgical success rates range from 30% to 70% [\(Löscher et al., 2020;](#page-10-3) [Li et al.,](#page-10-4) [2021\)](#page-10-4).

![](_page_1_Diagram_3.jpeg)

![](_page_1_Figure_4.jpeg)

Figure 1: An overview of Stochastic Sparse Sampling (SSS) training procedure. (A) For a given time series, we sample windows of fixed-length at random throughout the signal. (B) Each window is processed independently by a local model with parameters θ, outputting the local predictions yˆ1, . . . , yˆk. (C) Local predictions are then fed through an aggregation function to form the final prediction yˆ.

Contributions. While our work primarily focuses on VTSC, we also evaluate our method's performance on OOD data and explore its potential for providing local explanations. To this end, we propose Stochastic Sparse Sampling (SSS) a novel framework for VTSC developed for medical time series. The main contributions of our paper are listed as follows:

- Robustness to long and variable-length sequences. SSS samples fixed-length windows, and processes them independently through a single model. This prevents context overload in long sequences seen in infinite-context methods, and does not utilize padding, truncation, or interpolation

**099**

**109**

**111 112**

**127**

**129 130 131**

**134**

required by finite-context methods. By relying on a single local model, SSS utilizes far fewer parameters compared to finite-context methods that traditionally ingest the entire signal, which significantly reduces computational cost during training and the risk of overfitting over long sequences.

- Generalization to unseen patient populations. SSS demonstrates strong performance on out-of-distribution (OOD) data from unseen medical centers. When trained on data from one or more medical centers and evaluated on a completely new center with a different patient population, SSS outperforms all baselines in our comparisons. This result suggests SSS's potential as a foundation model for TSC, opening new avenues for research and clinical applications.
- Explainability through local predictions: Our method enhances model interpretability by directly tying each output—a probability score for each window—to the overall prediction. This capability is crucial in critical clinical settings, such as SOZ localization, which traditionally relies on visual analysis. Given the significant risks associated with brain region removal, any proposal should be designed to integrate within clinical workflows. Moreover, in the absence of universally recognized biological markers for epilepsy, SSS offers the potential to further our understanding the SOZ and to identify novel markers.
- Compatibility with modern and classical backbones. SSS integrates with any time series backbone. This ensures that our approach leverages well-established frameworks now and into the future, allowing for adaptability across a diverse array of contexts.

# 2 RELATED WORK

TSC methodologies can be broadly categorized into finite-context methods, which operate on fixed-length input segments, and infinite-context methods, which handle variable-length sequences without being restricted to a predetermined window size. For a formal treatment, please see Appendix [A.5.](#page-14-0)

Finite-context methods. Finite-context methods are among the most commonly used approaches for TSC. Transformer-based models have gained significant attention, with variations such as sparse attention, series decomposition, and patching techniques [\(Vaswani et al., 2017;](#page-11-2) [Kitaev et al., 2020;](#page-10-5) [Zhou et al., 2021;](#page-11-3) [Wu et al.,](#page-11-4) [2021;](#page-11-4) [Zhou et al., 2022;](#page-11-5) [Liu et al., 2022;](#page-10-6) [Nie et al., 2023;](#page-10-7) [Liu et al., 2024\)](#page-10-8). Several temporal convolutional networks (TCNs) have also been proposed, to capture temporal dependencies through dilated convolutions and Inception-like architectures [\(Lai et al., 2018;](#page-10-9) [Bai et al., 2018;](#page-9-8) [Ismail Fawaz et al., 2020;](#page-10-10) [Wu et al., 2022;](#page-11-6) [Luo & Wang, 2024\)](#page-10-11). Recently, multilayer perceptrons (MLPs) and simple linear models have demonstrated competitive performance as well [\(Chen et al., 2023;](#page-9-9) [Zeng et al., 2023\)](#page-11-7). Despite the significant rise of finitecontext methods, these methods are inherently limited in their ability to handle variable-length sequences, and will require the use of either padding, truncation, or interpolation for VTSC. Furthermore, as the sequence length increases, so does the number of model parameters, which leads to not only greater computational cost but an increased risk of overfitting.

Infinite-context methods. The recurrent neural network (RNN) family includes several models capable of ingesting variable-length time series [\(Rumelhart et al., 1986\)](#page-10-12). Long-short term memory (LSTM) networks introduce memory cells and gating mechanisms to better handle long-term dependencies [\(Hochreiter &](#page-9-10) [Schmidhuber, 1997\)](#page-9-10). Gated recurrent units (GRUs) simplify the LSTM architecture while maintaining similar performance [\(Bahdanau et al., 2014\)](#page-9-11). State space models (SSMs) have gained recent attention, with approaches such as S4 introducing structured parameterization to enable efficient computation over long sequences, while still attempting to capture long-range dependencies [\(Gu et al., 2021\)](#page-9-12). Building on this, Mamba introduces a selective SSM that adapts to input dynamics, further improving processing of

**154**

**156**

long-range dependencies in time series data while while maintaining linear time complexity with respect to sequence length [\(Gu & Dao, 2023\)](#page-9-13). Despite these advancements, RNNs and SSMs can still struggle with retaining information in extremely long sequences and may be prone to vanishing or exploding gradients [\(Salehinejad et al., 2017\)](#page-10-13). ROCKET offers an alternative, using random convolutional kernels to convert input into a fixed-length representation for VTSC, but at the cost of interpretability and potentially limited model expressivity [\(Dempster et al., 2020\)](#page-9-14).

SOZ localization methods. Several recent proposals have been tailored specifically to SOZ localization. Functional connectivity graphs compute patient-specific channel metrics to capture brain connectivity patterns [\(Grattarola et al., 2022;](#page-9-15) [Fang et al., 2024\)](#page-9-16), offering insights into functional relationships associated with seizures. However, their reliance on intra-patient dynamics makes them unsuitable for a single model that can generalize across multi-patient, heterogeneous datasets. Alternatively, electrical stimulation methods that use intracranial electrodes [\(Johnson et al., 2022;](#page-10-14) [Yang et al., 2024\)](#page-11-8) can enhance localization accuracy through induced responses analyzed by TCNs and logistic regression models. Yet, these approaches require both fixed-length windows and the use of active stimulation. For our purpose of building a general model for SOZ localization, which can be applied to multiple patients (with a potentially varying number of channels) without electrical stimulation, we do not consider such approaches in our study.

# 3 METHOD

### 3.1 VARIABLE-LENGTH TIME SERIES CLASSIFICATION

Consider a collection of time series X = (x (1) t ) T<sup>1</sup> <sup>t</sup>=1, . . . ,(x (n) t ) T<sup>n</sup> t=1 with labels Y = {y (1), . . . , y(n)}, where each series i has sequence length T<sup>i</sup> ∈ <sup>N</sup>, and for each time point t, the vector x (i) <sup>t</sup> ∈ <sup>R</sup> <sup>M</sup><sup>i</sup> has M<sup>i</sup> ∈ <sup>N</sup> channels. The goal of VTSC is to learn a classifier f<sup>θ</sup> which maps each series (x (i) t ) T<sup>i</sup> <sup>t</sup>=1 to its corresponding class in {1, . . . , K} for K ∈ <sup>N</sup> classes. We require that f<sup>θ</sup> can handle sequences of any length—that is, it has infinite context—since we assume that each T<sup>i</sup> can be arbitrarily large at inference time. Otherwise, we must adjust a finite-context classifier using padding, truncation, or interpolation.

#### 3.2 STOCHASTIC SPARSE SAMPLING

# 3.2.1 SPARSE TRAINING

Figure [1](#page-1-0) provides an overview of SSS at train time. During each training epoch, SSS performs a sampling procedure without replacement to create each batch. Fix L ∈ N as the window size and let W be the collection of all windows with size L from all time series in X. Within any batch, a window is drawn from W, where the probability of it originating from series i is set to p<sup>i</sup> ≈ Ti/ P<sup>n</sup> <sup>j</sup>=1 T<sup>j</sup> for every i. More formally, for each i, let N<sup>i</sup> be the random variable representing the number of windows from series i in a batch of size B. Then N<sup>i</sup> ∼ Binomial(B, pi), and consequently <sup>E</sup>[N<sup>i</sup> ] = Bp<sup>i</sup> . This proportional sampling ensures fair representation of each series based on its length, allowing longer sequences—which contain more information—to contribute more samples. By sampling only a subset of windows, SSS introduces sparsity into the training process, reducing computational cost found in finite-context methods, and the likelihood of context overload in infinite-context methods. Also note that by sampling with replacement, the model sees each window exactly once during a single training epoch.

After sampling a batch of windows W<sup>0</sup> = {w1, . . . , wB}, each w<sup>b</sup> ∈ W<sup>0</sup> is processed independently by a local model f<sup>θ</sup> to obtain a local prediction yˆ<sup>b</sup> = fθ(wb) ∈ [0, 1]<sup>K</sup>, representing our probability distribution over K ∈ <sup>N</sup> classes. The choice of f<sup>θ</sup> can be any time series backbone, in our experiments we select PatchTST [\(Nie et al., 2023\)](#page-10-7). For each time series 1 ≤ i ≤ n, denote:

$$\mathcal{W}_i = \{\mathbf{w} \in \mathcal{W}_0 \mid \mathbf{w} \text{ is from series } i\}, \quad (1)$$

**204**

**206**

**221**

**224**

as the collection of windows in the batch originating from series i, and let:

$$\mathcal{V}_i = \{f_\theta(\mathbf{w}) \mid \mathbf{w} \in \mathcal{W}_i\}, \quad (2)$$

be the set of multiset of window probabilities. To obtain the global prediction for time series i, we aggregate window probabilities from all examples originating from it, given by:

$$\hat{y}^{(i)} = \text{Aggr}(\mathcal{Y}_i) = \sum_{\hat{y} \in \mathcal{Y}_i} \alpha(\hat{y})\hat{y}, \quad (3)$$

where each P α(ˆy) ∈ [0, 1] represents the weight of window probability yˆ to the final output, satisfying yˆ∈Y<sup>i</sup> α(ˆy) = 1; that is, Aggr(·) produces a convex combination over yˆ1, . . . , yˆn. In our experiments, we use mean aggregation, i.e., α(ˆy) = <sup>1</sup> |Yi| for all yˆ ∈ Y<sup>i</sup> , due to its simplicity and effectiveness for our current objectives. This formulation guarantees that yˆ (i) remains a valid probability distribution over K classes (proof in Appendix [A.3\)](#page-13-0), and allows for potential of non-uniform aggregation functions, enabling weighting of window predictions based on factors such as prediction uncertainty, or frequency characteristics.

Algorithm 1 SSS Training Algorithm (Single Epoch)

Input: Time series X = (x (1) t ) T<sup>1</sup> <sup>t</sup>=1, . . . ,(x (n) t ) T<sup>n</sup> t=1 , labels Y = {y (1), . . . , y(n)}, model f<sup>θ</sup> with parameters θ, batch size B

Output: Updated model parameters θ

W ← Set of all windows from each series in X

while W ̸= ∅ do

▷ Sample B windows with probability Ti/ P j Tj from series i, for all i

W<sup>0</sup> ← SAMPLE(W, B)

for i = 1, . . . , n do

W<sup>i</sup> ← {w ∈ W<sup>0</sup> | w is from series i} ▷ Windows from series i

Y<sup>j</sup> ← {fθ(w)| w ∈ Wj} ▷ Window probabilities for series i

yˆ

(i) ← AGGREGATE(Y<sup>j</sup> ) ▷ Final probability for series i

Lbatch ← <sup>1</sup>

n P<sup>n</sup> <sup>i</sup>=1 L(ˆy

(i) , y(i)

) ▷ Loss over the batch

θ ← UPDATE(θ,Lbatch) ▷ Update local model parameters W ← W \ W<sup>0</sup> ▷ Remove sampled windows from the pool

return θ

#### 3.2.2 INFERENCE

To the derive the prediction for a time series (x (i) t ) T<sup>i</sup> <sup>t</sup>=1 at inference time, we utilize all windows from the selected time series, to form its final prediction yˆ (i) . Let W<sup>i</sup> be the collection of all windows from series i. We pass each window through the local model to obtain the multiset of window probabilities Y<sup>i</sup> as shown in Equation [\(2\)](#page-4-0). Before the aggregation step, we utilize a calibrator g<sup>ϕ</sup> : [0, 1]<sup>K</sup> → [0, 1]<sup>K</sup>, which adjusts each individual window probability to reduce the presence of noise, and define:

$$\tilde{\mathcal{Y}}_i = \{g_\phi(\hat{y}) \mid \hat{y} \in \mathcal{Y}_i\}, \quad (4)$$

which is then fed into the final prediction during the aggregation step yˆ (i) = Aggr(Y˜ <sup>i</sup>). By calibrating the window probabilities before aggregation, we correct for biases or misestimations in the predicted probabilities, which occur when the output probabilities of the local models do not accurately reflect the true likelihood of the event. We consider isotonic regression and Venn-Abers predictors for our calibration method. It is important to note, that these calibration techniques do not alter underlying structure of f<sup>θ</sup> and do not utilize input features from the time series; rather, they adjust the output probabilities to mitigate the effect of noise during the aggregation step. For more information regarding calibration methods see Appendix [A.4.](#page-13-1)

**254**

# 4 EXPERIMENTS

#### 4.1 BASELINES

For our finite-context baselines, we include a variety of modern time series backbones including PatchTST, which uses subwindows as tokens in combination with the traditional Transformer architecture [\(Nie et al.,](#page-10-7) [2023;](#page-10-7) [Vaswani et al., 2017\)](#page-11-2). TimesNet is a TCN architecture that models both interperiod and intraperiod dynamics by leveraging Fast Fourier Transform (FFT) features to slice the signal into multiple views, which are then processed through inception blocks [\(Wu et al., 2022;](#page-11-6) [Ismail Fawaz et al., 2020\)](#page-10-10). ModernTCN, is a recently proposed TCN which decouples temporal and channel information processing by using separate DWConv and ConvFFN modules for more efficient representation learning [\(Luo & Wang, 2024\)](#page-10-11). DLinear is linear neural network, which has shown to outperform several modern Transformer-based architecutres, utilizing traditional seasonal-trend decomposition techniques [\(Zeng et al., 2023\)](#page-11-7). In our infinite-context baselines, we utilize ROCKET, which applies randomly initialized, fixed convolutional kernels to the input sequence. ROCKET compresses the resulting convolutional outputs to the maximum value and the proportion of positive values (PPV), where these features then fed to a linear classifier [\(Dempster et al., 2020\)](#page-9-14). We also consider GRUs and an LSTM network, both of which are popular RNN frameworks designed to capture longterm dependencies in sequential data [\(Bahdanau et al., 2014;](#page-9-11) [Hochreiter & Schmidhuber, 1997\)](#page-9-10). Additionally, use the recent SSM architecture, Mamba, which utilizes selective state updates to enable efficient long-range dependency modeling [\(Gu & Dao, 2023\)](#page-9-13). Further details regarding configurations and hyperparameter tuning for each baseline can be found in Appendix [C.](#page-16-0)

# 4.2 DATASET

The Epilepsy iEEG Multicenter Dataset[<sup>1</sup>](#page-5-0) consists of iEEG signals with SOZ clinical annotations from four medical centers including the Johns Hopkins Hospital (JHH), the National Institute of Health (NIH), University of Maryland Medical Center (UMMC), and University of Miami Jackson Memorial Hospital (UMH). Since UMH contained only a single patient with clinical SOZ annotations, we did not consider it in our main evaluations; however, we did use UMH within the multicenter evaluation in [1](#page-6-0) and the training set for OOD experiments for SOZ localization on unseen medical centers in Table [2.](#page-6-1) We select the F1 score, Area Under the Receiver Operator Curve (AUC), and accuracy for our evaluation metrics. For summary statistics and information on the dataset see Appendix [B.1.](#page-15-0)

### 4.3 UNIVARIATE VTSC

For each patient iEEG recording, the goal of SOZ localization is to determine the the correct of channels or electrodes which belong the seizure onset zone. This effectively reduces the task to univariate TSC. While several channel-dependent methods have been proposed for SOZ localization (see section [2\)](#page-2-0), we focus primarily on channel-independent solutions for two key reasons: (1) they are more resilient to interpatient variability and are unaffected by factors like channel count and therefore can be applied in multiple hospital settings, and (2) they generalize better to domains beyond electrophysiological data, as they learn local signal characteristics rather than explicitly modeling functional connectivity between electrode sites, which may not be present in other medical time series.

Table [1](#page-6-0) summarizes our experimental results for SOZ localization on each individual medical center, along with training and evaluation on all medical centers. Within the multicenter evaluation, SSS outperforms all baselines for each evaluation metric. SSS also shows strong performance for the JHH and NIH centers, with comparable results in the UMMC center. We attribute this difference in performance for UMMC due to the fact that it is the only center where the sampling frequency of patient recordings can differ between patients

<sup>1</sup> https://openneuro.org/datasets/ds003029/versions/1.0.7

**304**

**306 307**

**309**

**311 312**

**321**

Table 1: SOZ localization. F1 score, AUC, and Accuracy are reported for each medical center, averaged over 5 seeds. For each center, we train and evaluate a separate model; the first column represents training and evaluation on all centers. Bolded values with <sup>∗</sup> and † denote the best and second-best results, respectively.

| Model                           | F1           | All AUC  | Acc.    | F1       | JHH AUC  | Acc.    | F1       | NIH AUC  | Acc.    | F1       | UMMC AUC | Acc.    |
|---------------------------------|--------------|----------|---------|----------|----------|---------|----------|----------|---------|----------|----------|---------|
| SSS (Ours)                      | 0.7629 ∗     |          |         |          |          |         |          |          |         |          |          |         |
|                                 |              | 0.7999 ∗ |         |          |          |         |          |          |         |          |          |         |
|                                 |              |          | 72.35 ∗ |          |          |         |          |          |         |          |          |         |
|                                 |              |          |         | 0.8187 ∗ |          |         |          |          |         |          |          |         |
|                                 |              |          |         |          | 0.8851 ∗ |         |          |          |         |          |          |         |
|                                 |              |          |         |          |          | 81.37 ∗ |          |          |         |          |          |         |
|                                 |              |          |         |          |          |         | 0.6716 ∗ |          |         |          |          |         |
|                                 |              |          |         |          |          |         |          | 0.6853   | 64.22 † |          |          |         |
|                                 |              |          |         |          |          |         |          |          |         | 0.7978 † |          |         |
|                                 |              |          |         |          |          |         |          |          |         |          | 0.8279 † |         |
| PatchTST (Nie et al., 2023)     | 0.7097 †     |          |         |          |          |         |          |          |         |          |          |         |
|                                 |              | 0.7852 † |         |          |          |         |          |          |         |          |          |         |
|                                 |              |          | 66.83   | 0.7419 † |          |         |          |          |         |          |          |         |
|                                 |              |          |         |          | 0.8045 † |         |          |          |         |          |          |         |
|                                 |              |          |         |          |          | 71.82   | 0.6402   | 0.7036 † |         |          |          |         |
|                                 |              |          |         |          |          |         |          |          | 62.11   | 0.8015 ∗ |          |         |
|                                 |              |          |         |          |          |         |          |          |         |          | 0.8121   | 77.58   |
| TimesNet (Wu et al., 2022)      | 0.6897       | 0.7174   | 65.98   | 0.6891   | 0.8029   | 73.64 † |          |          |         |          |          |         |
|                                 |              |          |         |          |          |         | 0.5950   | 0.6806   | 66.00 ∗ |          |          |         |
|                                 |              |          |         |          |          |         |          |          |         | 0.7821   | 0.8099   | 77.06 † |
| ModernTCN (Luo & Wang, 2024)    | 0.6938       | 0.7305   | 68.42   | 0.6710   | 0.7508   | 67.73   | 0.5055   | 0.7220 ∗ |         |          |          |         |
|                                 |              |          |         |          |          |         |          |          | 64.00   | 0.6371   | 0.8203   | 71.76   |
| DLinear (Zeng et al., 2023)     | 0.6916       | 0.7044   | 68.41   | 0.6873   | 0.7395   | 66.36   | 0.6055   | 0.6405   | 59.50   | 0.7658   | 0.7729   | 77.05   |
| ROCKET (Dempster et al., 2020)  | 0.6847       | 0.7481   | 69.27   | 0.6753   | 0.7752   | 69.09   | 0.6520 † |          |         |          |          |         |
|                                 |              |          |         |          |          |         |          | 0.6546   | 62.63   | 0.7686   | 0.7900   | 74.55   |
| Mamba (Gu & Dao, 2023)          | 0.6452       | 0.7134   | 64.39   | 0.6456   | 0.6764   | 62.27   | 0.5974   | 0.6050   | 58.95   | 0.7900   | 0.8424 ∗ |         |
| GRUs (Bahdanau et al., 2014)    | 0.6948       | 0.7340   | 65.85   | 0.6140   | 0.6959   | 63.18   | 0.6171   | 0.6283   | 62.63   | 0.7920   | 0.8211   | 77.27 ∗ |
| LSTM (Hochreiter & Schmidhuber, | 1997) 0.6709 | 0.7144   | 65.43   | 0.6571   | 0.6190   | 59.09   | 0.5657   | 0.5909   | 54.74   | 0.7604   | 0.8060   | 73.64   |

(250-1000 Hz), whereas JHH and NIH both have sampling frequencies of 1000 Hz. We also observe that in general, finite-context perform better on the chosen evaluation metrics in comparison to infinite-context methods, for in-distribution univariate VTSC.

# 4.4 OUT-OF-DISTRIBUTION VTSC

Table [2](#page-6-1) reports our results for SOZ localization in the OOD setting. At train time, from the collection of four medical center datasets, we omit one and train on the remaining three. At inference time, we test solely on the omitted medical center to gauge how well each method performs OOD. For iEEG signals from epilepsy patients, inter-patient variability can be significant due to differences in placement of electrodes in the brain, and the inherent heterogeneity of epileptogenic networks across individuals. Thus, even among medical time series, this can be one of the most challenging tasks to perform OOD. SSS outperforms each baseline on each unseen medical center, often by a considerable margin when compared to finite-context methods.

Table 2: Out-of-Distribution SOZ localization. F1 score, AUC, and Accuracy are reported for unseen medical centers, averaged over 5 seeds. For each center, we train on all other centers and evaluate on the selected center. Bolded values with <sup>∗</sup> and † denote the best and second-best results, respectively.

| Model SSS | (Ours)      |      |      |               | F1 0.6981    | JHH AUC ∗ 0.6590 | Acc. ∗ 57.80 | F1 ∗ 0.6492 | NIH AUC ∗ 0.6092 | Acc. ∗ 54.73 | F1 † 0.7243 | UMMC AUC ∗ 0.8048 | Acc. ∗ 72.42 ∗ |
|-----------|-------------|------|------|---------------|--------------|------------------|--------------|-------------|------------------|--------------|-------------|-------------------|----------------|
| PatchTST  | (Nie        | et   | al., | 2023)         | 0.6175       | 0.5267           | 50.46        | 0.5986      | 0.4829           | 48.17        | 0.5067      | 0.5274            | 57.63          |
| TimesNet  | (Wu         | et   | al., | 2022)         | 0.5261       | 0.4501           | 47.00        | 0.4461      | 0.4407           | 45.85        | 0.3177      | 0.3108            | 46.14          |
| ModernTCN |             | (Luo | &    | Wang, 2024)   | 0.4934       | 0.4970           | 49.54        | 0.4019      | 0.4651           | 48.71        | 0.3804      | 0.4474            | 50.55          |
| DLinear   | (Zeng       | et   | al., | 2023)         | 0.4205       | 0.4775           | 47.25        | 0.5090      | 0.4945           | 50.54        | 0.5602      | 0.5236            | 56.00          |
| ROCKET    | (Dempster   |      |      | et al., 2020) | 0.5784       | 0.5777           | 56.71        | †           |                  |              |             |                   |                |
|           |             |      |      |               |              |                  |              | 0.5051      | 0.5522           | 52.91        | 0.5608      | 0.5941            | 58.36          |
| Mamba     | (Gu &       | Dao, |      | 2023)         | 0.5790       | 0.5835           | †            |             |                  |              |             |                   |                |
|           |             |      |      |               |              |                  | 55.68        | 0.6183      | †                |              |             |                   |                |
|           |             |      |      |               |              |                  |              |             | 0.5767           | †            |             |                   |                |
|           |             |      |      |               |              |                  |              |             |                  | 55.69        | ∗           |                   |                |
|           |             |      |      |               |              |                  |              |             |                  |              | 0.5715      | 0.5953            | 55.76          |
| GRUs      | (Bahdanau   |      | et   | al., 2014)    | 0.5779       | 0.4868           | 48.80        | 0.5824      | 0.5588           | 53.66        | 0.6689      | †                 |                |
|           |             |      |      |               |              |                  |              |             |                  |              |             | 0.7645            | †              |
|           |             |      |      |               |              |                  |              |             |                  |              |             |                   | 69.30 †        |
| LSTM      | (Hochreiter |      | &    | Schmidhuber,  | 1997) 0.6362 | †                |              |             |                  |              |             |                   |                |
|           |             |      |      |               |              | 0.5165           | 50.92        | 0.5774      | 0.5678           | 53.87        | 0.6581      | 0.6616            | 62.36          |

In comparison to Table [1,](#page-6-0) we observe the opposite trend between finite- and infinite-context methods, whereas infinite-context methods seem to perform better OOD. In contrast, SSS demonstrates robust performance both in distribution (where finite-context methods excel) and OOD (where infinite-context methods excel).

#### 4.5 QUALITATIVE VISUALIZATION

![](_page_7_Figure_4.jpeg)

Figure 2: Visualization of SSS window probabilities throughout iEEG channels at inference time, using the PatchTST backbone with window size 1024. The heatmap represents locally averaged window probabilities over time, with color intensity being proportional to the likelihood of the channel belonging to the SOZ.

![](_page_7_Figure_6.jpeg)

Figure 3: Visualization of SSS window probabilities for OOD iEEG channels at inference time, using the PatchTST backbone with window size 1024. The heatmap represents locally averaged window probabilities over time, with color intensity being proportional to the likelihood of the channel belonging to the SOZ.

# 5 DISCUSSION

#### 5.1 REVIEW OF RESULTS

In general, we observe that SSS outperforms modern finite-context and infinite-context methods both in-distribution and OOD for univariate VTSC. SOZ localization presents a significant challenge due to intra-patient and inter-patient variability, and with our evaluation the collection of 3 heterogeneous datasets, serving rigorous testbed for assessing the generalizability of SSS's capabilities for learning local signal characteristics in medical time series. While our experiments focus on univariate VTSC, as outlined in section [3.2.1](#page-3-0) and [3.2.2,](#page-4-1) SSS can be easily applied to the multivariate setting.

Our results from the multicenter evaluation Table [1](#page-6-0) indicate that SSS benefits from a diversity of data distributions and volume of training examples, given by the magnitude in performance differences, when compared to single cluster results. Furthermore, our OOD experiments from Table [2](#page-6-1) suggest that SSS may be learning local signal characteristics present in different patient populations, leading to a advantage over finite-context and infinite-context methods. Our visualizations of SSS's predictions OOD in Figure [3](#page-7-0) supports this notion, as there exist clear qualitative differences in locally averaged window probabilities with respect to anomalous signal characteristics, such as spikes or increases in amplitude or frequency. Figure [2](#page-7-1) shows SSS's predictions in-distribution which also suggest a form of implicit semantic segmentation for anomalous local regions of the signal with respect to the SOZ probability. More analysis is needed to further solidify our understanding, which would benefit from a rigorous explainability study in future works.

# 5.2 CONCLUSION

To conclude, this work introduces novel VTSC framework, Stochastic Sparse Sampling (SSS), specifically tailored for medical time series applications. SSS blends the best of both worlds between finite-context methods (enabling usage of finite-context backbones) while allowing sampling of the entire signal in a computationally efficient manner that is less prone to context overload from infinite-context methods. SSS learns local signal characteristics, which provides the added benefit of inherent interpretability, and provides superior performance to the SOTA in-distritbuion and OOD for unseen medical centers. For future work, it would be valuable to: (1) benchmark SSS across a wider variety of variable-length medical time series, (2) provide further rigorous post-hoc insights into the window probability distribution given by SSS, and (3) potentially include uncertainty estimates within the aggregation function to highlight anomalous regions of the signal.

# REFERENCES


[1] Elena Agliari, Adriano Barra, Orazio Antonio Barra, Alberto Fachechi, Lorenzo Franceschi Vento, and Luciano Moretti. Detecting cardiac pathologies via machine learning on heart-rate variability time series and related markers. *Scientific Reports*, 10(1):8845, 2020. Julia Amann, Alessandro Blasimme, Effy Vayena, Dietmar Frey, Vince I Madai, and Precise4Q Consortium. Explainability for artificial intelligence in healthcare: a multidisciplinary perspective. *BMC medical informatics and decision making*, 20:1–9, 2020. Anastasios N Angelopoulos and Stephen Bates. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. *arXiv preprint arXiv:2107.07511*, 2021. Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. *arXiv preprint arXiv:1409.0473*, 2014. Shaojie Bai, J Zico Kolter, and Vladlen Koltun. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv preprint arXiv:1803.01271*, 2018. Sai Sanjay Balaji and Keshab K Parhi. Seizure onset zone identification from ieeg: A review. *IEEE Access*, 10: 62535–62547, 2022. Si-An Chen, Chun-Liang Li, Nate Yoder, Sercan O Arik, and Tomas Pfister. Tsmixer: An all-mlp architecture for time series forecasting. *arXiv preprint arXiv:2303.06053*, 2023. Jonathan Crabbé and Mihaela Van Der Schaar. Explaining time series predictions with dynamic masks. In *International Conference on Machine Learning*, pp. 2166–2177. PMLR, 2021. Angus Dempster, François Petitjean, and Geoffrey I Webb. Rocket: exceptionally fast and accurate time series classification using random convolutional kernels. *Data Mining and Knowledge Discovery*, 34(5):1454–1495, 2020. T Deutsch, ED Lehmann, ER Carson, AV Roudsari, KD Hopkins, and PH Sönksen. Time series analysis and control of blood glucose levels in diabetic patients. *Computer Methods and Programs in Biomedicine*, 41(3-4):167–182, 1994. Joseph Early, Gavin KC Cheung, Kurt Cutajar, Hanting Xie, Jas Kandola, and Niall Twomey. Inherently interpretable time series classification via multiple instance learning. In *The Twelfth International Conference on Learning Representations*, 2024. Chunying Fang, Xingyu Li, Meng Na, Wenhao Jiang, Yuankun He, Aowei Wei, Jie Huang, and Ming Zhou. Epilepsy lesion localization method based on brain function network. *Frontiers in Human Neuroscience*, 18:1431153, 2024. Daniele Grattarola, Lorenzo Livi, Cesare Alippi, Richard Wennberg, and Taufik A Valiante. Seizure localisation with attention-based graph neural networks. *Expert systems with applications*, 203:117330, 2022. Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. *arXiv preprint arXiv:2312.00752*, 2023. Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces. *arXiv preprint arXiv:2111.00396*, 2021. Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. *Neural computation*, 9(8):1735–1780, 1997. Qi Huang, Wei Chen, Thomas Bäck, and Niki van Stein. Shapelet-based model-agnostic counterfactual local explanations for time series classification. *arXiv preprint arXiv:2402.01343*, 2024. Hassan Ismail Fawaz, Germain Forestier, Jonathan Weber, Lhassane Idoumghar, and Pierre-Alain Muller. Deep learning for time series classification: a review. *Data mining and knowledge discovery*, 33(4):917–963, 2019.

[2] **502**

[3] **504 505 506**

[4] **509**

[5] **511 512**

[6] **514 515 516**

[7] Hassan Ismail Fawaz, Benjamin Lucas, Germain Forestier, Charlotte Pelletier, Daniel F Schmidt, Jonathan Weber, Geoffrey I Webb, Lhassane Idoumghar, Pierre-Alain Muller, and François Petitjean. Inceptiontime: Finding alexnet for time series classification. *Data Mining and Knowledge Discovery*, 34(6):1936–1962, 2020. Graham W Johnson, Leon Y Cai, Derek J Doss, Jasmine W Jiang, Aarushi S Negi, Saramati Narasimhan, Danika L Paulo, Hernán FJ González, Shawniqua Williams Roberson, Sarah K Bick, et al. Localizing seizure onset zones in surgical epilepsy with neurostimulation deep learning. *Journal of neurosurgery*, 138(4):1002–1007, 2022. Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi, and Jaegul Choo. Reversible instance normalization for accurate time-series forecasting against distribution shift. In *International Conference on Learning Representations*, 2021. Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*, 2014. Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In *International Conference on Learning Representations*, 2020. URL <https://openreview.net/forum?id=rkgNKkHtvB>. Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long-and short-term temporal patterns with deep neural networks. In *The 41st international ACM SIGIR conference on research & development in information retrieval*, pp. 95–104, 2018. Adam Li, Chester Huynh, Zachary Fitzgerald, Iahn Cajigas, Damian Brusko, Jonathan Jagid, Angel O Claudio, Andres M Kanner, Jennifer Hopp, Stephanie Chen, et al. Neural fragility as an eeg marker of the seizure onset zone. *Nature neuroscience*, 24(10):1465–1474, 2021. Shizhan Liu, Hang Yu, Cong Liao, Jianguo Li, Weiyao Lin, Alex X Liu, and Schahram Dustdar. Pyraformer: Lowcomplexity pyramidal attention for long-range time series modeling and forecasting. In *International Conference on Learning Representations*, 2022. Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long. itransformer: Inverted transformers are effective for time series forecasting. In *International Conference on Learning Representations*, 2024. Wolfgang Löscher, Heidrun Potschka, Sanjay M Sisodiya, and Annamaria Vezzani. Drug resistance in epilepsy: clinical impact, potential mechanisms, and new innovative treatment options. *Pharmacological reviews*, 72(3):606–638, 2020. Donghao Luo and Xue Wang. Moderntcn: A modern pure convolution structure for general time series analysis. In *The Twelfth International Conference on Learning Representations*, 2024. Navid Mohammadi Foumani, Lynn Miller, Chang Wei Tan, Geoffrey I Webb, Germain Forestier, and Mahsa Salehi. Deep learning for time series classification and extrinsic regression: A current survey. *ACM Computing Surveys*, 56(9):1–45, 2024. Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In *International Conference on Learning Representations*, 2023. World Health Organization et al. *Epilepsy: a public health imperative*. World Health Organization, 2019. Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. *Advances in neural information processing systems*, 32, 2019. Pranav Rajpurkar, Emma Chen, Oishi Banerjee, and Eric J Topol. Ai in health and medicine. *Nature medicine*, 28(1): 31–38, 2022. David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning internal representations by error propagation, parallel distributed processing, explorations in the microstructure of cognition, ed. de rumelhart and j. mcclelland. vol.

1. 1986. *Biometrika*, 71(599-607):6, 1986. Hojjat Salehinejad, Sharan Sankar, Joseph Barfett, Errol Colak, and Shahrokh Valaee. Recent advances in recurrent neural networks. *arXiv preprint arXiv:1801.01078*, 2017.

[9] **521**

[10] **534**

[11] **554**

[12] **556**

[13] Mervyn J Silvapulle and Pranab Kumar Sen. *Constrained statistical inference: Order, inequality, and shape constraints*. John Wiley & Sons, 2011. Carl E Stafstrom and Lionel Carmant. Seizures and epilepsy: an overview for neuroscientists. *Cold Spring Harbor perspectives in medicine*, 5(6):a022426, 2015. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. *Advances in neural information processing systems*, 30, 2017. Vladimir Vovk and Ivan Petej. Venn-abers predictors. In *Proceedings of the Thirtieth Conference on Uncertainty in Artificial Intelligence*, pp. 829–838, 2014. Dominik Walther, Johannes Viehweg, Jens Haueisen, and Patrick Mäder. A systematic comparison of deep learning methods for eeg time series analysis. *Frontiers in Neuroinformatics*, 17:1067095, 2023. Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with autocorrelation for long-term series forecasting. *Advances in Neural Information Processing Systems*, 34:22419–22430, 2021. Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long. Timesnet: Temporal 2d-variation modeling for general time series analysis. *arXiv preprint arXiv:2210.02186*, 2022. Bowen Yang, Baotian Zhao, Chao Li, Jiajie Mo, Zhihao Guo, Zilin Li, Yuan Yao, Xiuliang Fan, Du Cai, Lin Sang, et al. Localizing seizure onset zone by a cortico-cortical evoked potentials-based machine learning approach in focal epilepsy. *Clinical Neurophysiology*, 158:103–113, 2024. Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting? 2023. Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. In *Proceedings of the AAAI conference on artificial intelligence*, volume 35, pp. 11106–11115, 2021. Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. In *Proc. 39th International Conference on Machine Learning (ICML 2022)*, 2022.

[14] **567**

[15] **569**

[16] **571 572**

[17] **574 575 576**

[18] **579**

[19] **581**

[20] **584**
#### A STOCHASTIC SPARSE SAMPLING

#### A.1 SAMPLING IMPLEMENTATION

Let X = (x (1) t ) T<sup>1</sup> <sup>t</sup>=1, . . . ,(x (n) t ) T<sup>n</sup> t=1 be the collection of variable-length time series. To achieve the sampling procedure outlined in section [3.2,](#page-3-1) we first construct the set all windows W by performing the slicing window method over each individual time series in X. For a window size L, with window stride S, and a time series i with sequence length T<sup>i</sup> , we obtain A<sup>i</sup> = ⌊ Ti−L S ⌋ + 1 windows. Note that A<sup>i</sup> ∝ T<sup>i</sup> for each i. During training, we convert W to a PyTorch dataset and use the native PyTorch dataloader with batch size B. When a batch is sampled, windows are drawn uniformly from W, and thus for each i, the probability of observing a window from series i is p<sup>i</sup> = Ai/( P<sup>n</sup> <sup>j</sup>=1 A<sup>j</sup> ) ≈ Ti/( P<sup>n</sup> <sup>j</sup>=1 T<sup>j</sup> ). If N<sup>i</sup> represents the number of windows in the batch from series i, then we achieve the desired sampling property of N<sup>i</sup> ∼ Binomial(B, pi) where p<sup>i</sup> ≈ Ti/( P<sup>n</sup> <sup>j</sup>=1 T<sup>j</sup> ). Note that this procedure uses sampling *without* replacement; one may consider replacement, however, we did not experiment with this and leave modifications with more complex sampling procedures as a future direction.

#### A.2 ABLATIONS

#### A.2.1 BATCH SIZE

Table 3: Performance of SSS over various batch sizes.

| Batch | Size   | F1 Score  |        | AUC       |       | Acc. (%) |
|-------|--------|-----------|--------|-----------|-------|----------|
| 128   | 0.7563 | ± 0.02717 | 0.8139 | ± 0.04302 | 70.79 | ± 2.323  |
| 512   | 0.7498 | ± 0.02354 | 0.7965 | ± 0.03526 | 69.46 | ± 3.109  |
| 2048  | 0.7651 | ± 0.03568 | 0.8194 | ± 0.05166 | 71.40 | ± 6.078  |
| 4096  | 0.7441 | ± 0.02987 | 0.7979 | ± 0.06818 | 68.09 | ± 4.848  |
| 8192  | 0.7629 | ± 0.02829 | 0.7999 | ± 0.05331 | 72.35 | ± 4.965  |

Table [3](#page-12-0) reports mean F1 score, AUC, and accuracy (%) with standard deviations, are reported over 5 seeds, for the evaluation on all medical centers. Each experiment uses the best configuration described in Table [6.](#page-15-1) While the aggregation function in Equation [\(3\)](#page-4-2) may benefit from a higher number of samples within the batch (due to mean approximation), we observe that the performance of SSS remains relatively constant across various batch sizes, and thus does not require large batch sizes to achieve adequate performance.

# A.2.2 WINDOW SIZE

Table 4: Performance of SSS over various window sizes L.

| L    |        | F1 Score  |        | AUC       |       | Acc. (%) |
|------|--------|-----------|--------|-----------|-------|----------|
| 512  | 0.7567 | ± 0.01075 | 0.8141 | ± 0.03054 | 70.62 | ± 2.165  |
| 1024 | 0.7629 | ± 0.02829 | 0.7999 | ± 0.05331 | 72.35 | ± 4.965  |
| 2048 | 0.7334 | ± 0.03003 | 0.7719 | ± 0.04762 | 68.52 | ± 3.353  |

Table [4](#page-12-1) follows the same experimental setup as Table [3,](#page-12-0) but varies over the window size L instead of batch size. While the performance of L = 512 and L = 1024 remain relatively on par, we notice that the F1 score drops significantly for L = 2048 along with all other metrics. This suggests that a large receptive field may not be advantageous, and that SSS benefits from processing localized areas of the signal. Indeed, as L increases we expect to reach a similar performance to the finite-context PatchTST baseline, with a decrease in performance as a result.

# A.2.3 CALIBRATION

Table [4](#page-12-1) follows the same experimental setup as Table [3,](#page-12-0) but varies over the window size L instead of batch size. While the performance of L = 512 and L = 1024 remain relatively on par, we notice that the F1 score drops significantly for L = 2048 along with all other metrics. This suggests that a large receptive field

may not be advantageous, and that SSS benefits from processing localized areas of the signal. Indeed, as L increases we expect to reach a similar performance to the finite-context PatchTST baseline, with a decrease in performance as a result.

# A.3 CONVEX AGGREGATION

Theorem 1 (Probability Distribution Guarantee). *Fix* K, n ∈ <sup>N</sup>*. Suppose* α1, . . . , α<sup>n</sup> ≥ 0 *satisfies* P<sup>n</sup> <sup>i</sup>=1 <sup>α</sup><sup>i</sup> = 1*, and* <sup>v</sup>1, . . . , <sup>v</sup><sup>n</sup> <sup>∈</sup> [0, 1]<sup>K</sup> *each satisfy* P<sup>K</sup> <sup>j</sup>=1 vik = 1 *for* 1 ≤ i ≤ n*. That is, each* v<sup>i</sup> = (vi1, vi2, . . . , viK) T *represents a valid discrete probability distribution over* K *classes. Then the convex combination:*

$$\mathbf{y} = \sum_{i=1}^n \alpha_i \mathbf{v}_i, \quad (5)$$

*also represents a valid discrete probability distribution, satisfying* P<sup>K</sup> <sup>j</sup>=1 y<sup>j</sup> = 1*.*

*Proof.* By construction, y<sup>j</sup> = P<sup>n</sup> <sup>i</sup>=1 αivij for each entry 1 ≤ j ≤ K. Then y<sup>j</sup> ≥ 0, since for each 1 ≤ i ≤ n and 1 ≤ j ≤ K we are given that α<sup>i</sup> ≥ 0 and vij ≥ 0. Furthermore, we can write:

$$\begin{aligned} \sum_{j=1}^n y_j &= \sum_{j=1}^K \sum_{i=1}^n \alpha_i v_{ij} \\ &= \sum_{i=1}^n \sum_{j=1}^K \alpha_i v_{ij} && \text{(Swap summation order)} \\ &= \sum_{i=1}^n \alpha_i \sum_{j=1}^K v_{ij} \\ &= \sum_{i=1}^n \alpha_i \\ &= 1 \end{aligned}$$

It follows that since each y<sup>j</sup> ≥ <sup>0</sup> and P<sup>K</sup> <sup>j</sup>=1 y<sup>j</sup> = 1, then y represents a valid discrete probability distribution over K classes.

# A.4 CALIBRATION

Table 5: Performance of SSS with different calibration methods.

| Calibration Method | F1 Score         | AUC              | Acc. (%)       |
|--------------------|------------------|------------------|----------------|
| Stoick Regression  | 0.7629 ± 0.02829 | 0.7999 ± 0.05331 | 72.35 ± 4.962  |
| Venn-ABERS         | 0.7637 ± 0.02704 | 0.7998 ± 0.05308 | 72.47 ± 4.772  |
| No Calibration     | 0.7291 ± 0.06909 | 0.7830 ± 0.03844 | 69.93 ± 0.0001 |

Let yˆ1, . . . , yˆ<sup>n</sup> ∈ [0, 1] be the uncalibrated window probabilities, each with a corresponding binary label y1, . . . , y<sup>n</sup> ∈ {0, 1} derived from the label of the time series; that is, if yˆ<sup>i</sup> = fθ(wi) for a window w<sup>i</sup> , then the window label y<sup>i</sup> is inherited from the global time series it was sampled from. The goal of probability calibration is to transform each yˆ<sup>i</sup> into y˜<sup>i</sup> = gϕ(ˆyi), such that y˜<sup>i</sup> represents true likelihood of a class. Within this context, probability calibration can help mitigate the impact of temporal

fluctuations and local anomalies by adjusting probabilities for each individual windows. Note that while

calibration may yield more refined probability estimates with reduced noise, it is an integral intermediate step rather than a global optimization procedure on the final probabilities. For each calibration we considered, we provide a short description below.

Isotonic regression is a nonparametric method that fits a weighted least-squares model subject to motonicity constraints [\(Silvapulle & Sen, 2011\)](#page-11-9). Formally, this can be stated as a quadratic program (QP) given by:

$$\min_g \sum_{i=1}^n w_i(\tilde{y}_i - y_i)^2 \text{ subject to } \tilde{y}_i \leq \tilde{y}_j \text{ for all } i, j \text{ where } \hat{y}_i \leq \hat{y}_j. \quad (6)$$

where y˜<sup>i</sup> = g(ˆyi) and w<sup>i</sup> ≥ 0 are weights assigned to each datapoint, which are often each set to w<sup>i</sup> = 1 to provide equal importance over all inputs. The monotonicity constraint ensures that uncalibrated probabilities will always map to equal or higher calibrated probabilities. Due its nonparametric nature, isotonic regression can adapt to various probability distributions across diverse datasets. However, this flexibility comes at a cost as it is also prone to overfitting on smaller datasets, potentially adjusting to noise rather than properly calibrating its inputs.

Venn-Abers predictors is based on the concept of isotonic regression but extends it to ensure validity within the framework of conformal prediction, which provides uncertainty estimates with distribution-free theoretical guarantees [\(Vovk & Petej, 2014;](#page-11-10) [Angelopoulos & Bates, 2021\)](#page-9-17). For an uncalibrated probability yˆ, two isotonic calibrators are trained:

$$p_0 = g_0(\hat{y}) \text{ and } p_1 = g_1(\hat{y}) \quad (7)$$

where g<sup>0</sup> and g<sup>1</sup> are isotonic functions derived from augmented sets. These sets include (ˆy, 0) and (ˆy, 1) respectively, alongside all other uncalibrated probabilities and their respective labels. The values p<sup>0</sup> and p<sup>1</sup> represent likelihoods for class 0 and class 1, while the interval [p0, p1] provides an uncertainty estimate of where the true probability resides. The final calibrated probability is then given by:

$$\tilde{y} = \frac{p_1}{1 - p_0 + p_1}. \quad (8)$$

Venn-Abers predictors provide guaranteed validity in terms of calibration, meaning the predicted probabilities closely match empirical frequencies. While we do not explicitly utilize the uncertainty interval [p0, p1] (only the calibrated score y˜), this method can be effective in scenarios requiring risk assessment or critical tasks where rigorous uncertainty estimates are crucial. We leave this as a future direction to implement conformal prediction within the context of SSS, to provide uncertainty guarantees based off of window predictions, which may be useful for post-hoc interpretability. In comparsion to isotonic regression, Venn-Abers can be more computationally intensive, as it requires fitting two isotonic functions simultaneously.

# A.5 FINITE-CONTEXT & INFINITE-CONTEXT METHODS

Definition 2. Let X be a vector space over <sup>R</sup> and f<sup>θ</sup> : X → Y be a model with parameters θ and output space Y. We say that f<sup>θ</sup> has finite-context if X is finite-dimensional, that is, there exists some n ∈ <sup>N</sup> such that X ∼= <sup>R</sup> n as vector spaces. Whereas f<sup>θ</sup> is said to have infinite-context if X = <sup>R</sup> (∞) is the space of real number sequences with finite support[<sup>2</sup>](#page-15-2) .

Note that this definition refers to the *native* capabilities of fθ, without the usage of data manipulation techniques such as padding, truncation, and interpolation. We utilize this formalization to separate our baselines, so that we may better understand the advantages and limitations of both.

**709**

**721**

**724**

**727**

**736**

Table 6: Hyperparameter search space for SSS (with PatchTST backbone). Best configuration is highlighted in red.

| Parameter      | Search    |      |     |        | Values |                       |             |
|----------------|-----------|------|-----|--------|--------|-----------------------|-------------|
| d model        | {         | 16   | ,   | 32     | , 64   |                       | }           |
| d ff           | {         | 32   | ,   | 64     | , 128  |                       | }           |
| num_heads      | {         | 2    | , 4 | , 8    | }      |                       |             |
| num_enc_layers | {         | 1    | , 2 | , 3    | }      |                       |             |
| lr             | {         | 10   | −   | 4      |        |                       |             |
|                |           |      |     |        | , 10   | −                     | 5 }         |
| L              | {         | 512  |     | , 1024 |        |                       | , 2048 }    |
| batch_size     | {         | 2048 |     | ,      | 4096   |                       | , 8192 }    |
| g ϕ            | {isotonic |      |     |        |        | regression,Venn-Abers | predictors} |

#### A.6 SSS IMPLEMENTATION AND CONFIGURATIONS

# B DATASET AND PREPROCESSING

# B.1 DATASET

Table 7: iEEG Multicenter Dataset Summary: For each medical center, we report the total number of patients recorded (n), the number of patients with seizure onset zone (SOZ) annotations (nSOZ), the number of time series recordings (nts), the percentage of time series labeled as SOZ (pSOZ), the type of iEEG method used (e.g., electrocorticography, ECoG), the sampling frequency (Hz) (noting that some recordings may vary), and post-operative patient outcomes following SOZ surgical resection.

| Medical | Center n | n SOZ | n ts | p SOZ  | iEEG Type | Frequency (Hz) | Patient Outcomes |
|---------|----------|-------|------|--------|-----------|----------------|------------------|
| JHH     | 7        | 3     | 1458 | 7.48%  | ECoG      | 1000           | No               |
| NIH     | 14       | 11    | 3057 | 12.23% | ECoG      | 1000           | Yes              |
| UMMC    | 9        | 9     | 2967 | 5.56%  | ECoG      | 250-1000       | Yes              |
| UMF     | 5        | 1     | 129  | 25.58% | ECoG      | 1000           | No               |

Table [7](#page-15-3) provides an overview of the iEEG Multicenter Dataset. For each cluster, we filter out patients (n) who have SOZ annotations (nSOZ). All channels for all patients are group together into one dataset per medical center, where nts indicates the number of examples. However, due to the heavy imbalance between SOZ-labeled time series and non-SOZ labeled time series, the number of examples used for training and validation decreases significantly once we employ class balancing, resulting in ⌊2 · pSOZ · nts⌋ examples for each medical center, which is then split into training, validation, and testing.

### B.2 DATA PREPROCESSING

Each patient recording contains multiple channels corresponding to individual electrodes from the iEEG device. During preprocessing, for all patients, we extract all channels and balance the dataset to have an equal number of SOZ and non-SOZ channels. After, we partition this dataset into train, validation, and test channels with a 70%/10%/20% split respectively, and ensure that during the window sampling phase of SSS there is no temporal leakage from the test set. Each channel, or univariate time series, is z−score normalized to have zero mean and unit standard deviation.

<sup>2</sup>Every sequence in <sup>R</sup> (∞) must have finitely many non-zero terms.

**756**

**767**

**769**

**771 772**

**774**

**776**

**783 784**

**787**

Due to the extremely long sequence length of several channels, we required downsampling to fit the dataset into memory (250GB RAM). To achieve this, for each channel we applied a 1D average pooling layer with kernel\_size=24 and kernel\_stride=12 before feeding it to the baseline model or before performing the window sampling procedure for SSS at train-time.

Finite-context methods required either padding, truncation, or interpolation due to fit each sequence into its limited context window. For each finite-context method we perform a combination of padding and truncation according to the chosen window size L: if the sequence length of the original time series exceeded L, it was truncated to obtain the first L time points, otherwise if it was less than L, it was padded with zeros at the end of the sequence to ensure a sequence length of L.

# B.3 EVALUATION METRICS

# B.3.1 F1

The F1 score is defined as the harmonic mean of precision and recall, given by:

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}, \quad (9)$$

where,

$$\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}, \quad (10)$$

and,

$$\text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}. \quad (11)$$

We select the F1 score as our primary evaluation metric for SOZ localization for the following reasons. The F1 score balances the need to correctly identify all regions of the SOZ (recall) with the need to avoid misclassifying healthy regions as SOZ (precision). This balance is crucial in surgical planning, where both missing SOZs and unnecessarily removing healthy tissue can have severe consequences. Unlike accuracy, which overlooks the difference between false positives and false negatives, the F1 score provides a more nuanced evaluation by considering both, making it well-suited in clinical contexts such as SOZ localization.

# B.3.2 AUC

We complement the F1 score with the Area Under the Receiver Operating Characteristic curve (AUC), defined by:

$$\text{AUC} = \int_0^1 \text{TPR}(t) \cdot \frac{d\text{FPR}(t)}{dt} dt \quad (12)$$

While the F1 score provides insight into the balance between precision and recall at a specific threshold, AUC assesses the model's overall discriminative ability across all thresholds. This threshold-independent evaluation is relevant for critical scenarios where the threshold maybe be adjusted from 0.5, which is not common in clinical settings.

# C IMPLEMENTATION AND EXPERIMENTAL CONFIGURATIONS

For each baseline, we perform grid search and optimize with respect to best accuracy score on the evaluation for all medical centers. L refers to the window size parameter, dmodel is the model dimension, and dff

**804**

**806**

**834**

is the dimension of the feed-forward network. The grid search parameters for each baseline are shown below; for information on the implementation of SSS, see Appendix [A.6.](#page-15-4) In all experiments, we train using the Adam optimizer [\(Kingma & Ba, 2014\)](#page-10-15), for 50 epochs, with cosine learning rate annealing (one cycle with 50 epochs in length) which adjusts the learning rate down by two orders of magnitude (e.g., 10−<sup>4</sup> to 10−<sup>6</sup> ) by the last epoch. We also implement early stopping with a patience of 15, and apply learnable instance normalization [\(Kim et al., 2021\)](#page-10-16) for each input. For most of the baselines we use a dropout rate of 0.2 − 0.3, and weight decay to 10−<sup>4</sup> − 10−<sup>5</sup> , but do not explicitly tune these parameters in our grid search. For finite-context methods we set the batch size to the entire dataset (596 individual univariate time series for all clusters), whereas infinite-context methods required batch size of 1 due to their variable-length. The code for SSS and baseline implementations is available at the following anonymous link: [https://anonymous.4open.science/r/sss-0D75/](https://anonymous.4open.science/r/sss-0D75/README.md).

PatchTST: We adapt the official implementation [github.com/yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST), but swap out the attention module with the native PyTorch [torch.nn.MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) module.

Table 8: Hyperparameter search space for PatchTST. Best configuration is highlighted in red.

| Parameter      | Search Values                  |
|----------------|--------------------------------|
| d model        | { 16 , 32 , 64 }               |
| d ff           | { 32 , 64 , 128 }              |
| num_heads      | { 2 , 4 , 8 }                  |
| num_enc_layers | { 1 , 2 , 3 }                  |
| lr             | { 10 − 4                       |
|                | , 10 − 5 }                     |
| L              | { 1000 , 3000 , 5000 , 10000 } |

TimesNet: We use the official implementation [github.com/thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library).

Table 9: Hyperparameter search space for TimesNet. Best configuration is highlighted in red.

| Parameter      | Search |     | Values                  |
|----------------|--------|-----|-------------------------|
| d model        | { 16   | ,   | 32 , 64 }               |
| d ff           | { 32   | ,   | 64 , 128 }              |
| num_kernels    | { 4    | , 6 | }                       |
| top_k          | { 3    | , 5 | }                       |
| num_enc_layers | { 1    | , 2 | }                       |
| lr             | { 10   | −   | 4                       |
|                |        |     | , 10 − 5 }              |
| L              | { 1000 |     | , 3000 , 5000 , 10000 } |

**861**

**864**

**869**

**879**

**881**

**884**

ModernTCN: We use the official implementation [github.com/luodhhh/ModernTCN](https://github.com/luodhhh/ModernTCN).

Table 10: Hyperparameter search space for ModernTCN. Best configuration is highlighted in red.

| Parameter         | Search |      |      | Values                  |
|-------------------|--------|------|------|-------------------------|
| lr                | {      | 10   | −    | 4                       |
|                   |        |      |      | , 10 − 5 }              |
| d model           | {      | 16   | ,    | 32 , 64 }               |
| num_enc_layers    | {      | 1    | , 2  | }                       |
| large_size_kernel | {      | 9    | , 13 | , 21 , 51 }             |
| small_size_kernel | 5      |      |      |                         |
| dw_dims           | {      | 128  |      | , 256 }                 |
| ffn_ratio         | {      | 1    | , 4  | }                       |
| L                 | {      | 1000 |      | , 3000 , 5000 , 10000 } |

DLinear: We use the implementation from [github.com/thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library).

Table 11: Hyperparameter search space for DLinear. Best configuration is highlighted in red.

| Parameter  | Search Values                  |
|------------|--------------------------------|
| moving_avg | { 10 , 25 }                    |
| lr         | { 10 − 4                       |
|            | , 10 − 5                       |
|            | , 10 − 6 }                     |
| L          | { 1000 , 3000 , 5000 , 10000 } |

ROCKET: We use the official implementation [github.com/angus924/rocket](https://github.com/angus924/rocket/blob/master/code/rocket_functions.py) and follow the standard implementation of 10, 000 kernels.

Table 12: Hyperparameter search space for ROCKET. Best configuration is highlighted in red.

| Parameter   | Search Values |
|-------------|---------------|
| num_kernels | { 10000 }     |
| lr          | { 10 − 4      |
|             | , 10 − 5      |
|             | , 10 − 6 }    |

Mamba: We use the package [mambapy](https://github.com/alxndrTL/mamba.py) which builds upon the official Mamba implementation. We

Table 13: Hyperparameter search space for Mamba. Best configuration is highlighted in red.

| Parameter      | Search Values    |
|----------------|------------------|
| lr             | { 10 − 4         |
|                | , 10 − 5         |
|                | , 10 − 6 }       |
| d model        | { 16 , 32 , 64 } |
| num_enc_layers | { 1 , 2 , 3 }    |

also employ patching from [\(Nie et al., 2023\)](#page-10-7), which we observed led to greater to performance, with patch\_size = 64 and patch\_stride = 16.

GRUs: We utilized the native PyTorch module [torch.nn.GRU](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html).

Table 14: Hyperparameter search space for GRUs. Best configuration is highlighted in red.

| Parameter      | Search Values    |
|----------------|------------------|
| lr             | { 10 − 4         |
|                | , 10 − 5         |
|                | , 10 − 6 }       |
| d model        | { 16 , 32 , 64 } |
| num_enc_layers | { 1 , 2 , 3 }    |
| bidirectional  | { True , False } |

# C.1 COMPUTATIONAL RESOURCES

Our experiments were conducted on 4x NVIDIA RTX 6000 Ada Generation GPUs using the PyTorch framework with CUDA [Paszke et al.](#page-10-17) [\(2019\)](#page-10-17). Although we have not tracked explicitly the amount of consumed GPU hours, all experiments can be conducted for 5 seeds no more than 2 − 3 hours with a similar setup.