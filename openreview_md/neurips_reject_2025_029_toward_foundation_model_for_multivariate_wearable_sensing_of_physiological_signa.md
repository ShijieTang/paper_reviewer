# Toward Foundation Model For Multivariate Wearable Sensing Of Physiological Signals

Anonymous Author(s)
Affiliation Address email

## Abstract

1 Time-series foundation models excel at tasks like forecasting across diverse data 2 types by leveraging informative waveform representations. Wearable sensing data, 3 however, pose unique challenges due to their variability in patterns and frequency 4 bands, especially for healthcare-related outcomes. The main obstacle lies in crafting 5 generalizable representations that adapt efficiently across heterogeneous sensing 6 configurations and applications. To address this, we propose NORMWEAR, the 7 first multi-modal and ubiquitous foundation model designed to extract generalized 8 and informative representations from wearable sensing data. Specifically, we 9 design a channel-aware attention mechanism with a shared special liaison [CLS]
10 token to detect signal patterns in both intra-sensor and inter-sensors. This helps 11 the model to extract more meaningful information considering both time series 12 themselves and the relationships between input sensors. This helps the model to 13 be widely compatible with various sensors settings. NORMWEAR is pretrained 14 on a diverse set of physiological signals, including PPG, ECG, EEG, GSR, and 15 IMU, from various public datasets. Our model shows exceptional generalizability 16 across 11 public wearable sensing datasets, spanning 18 applications in mental 17 health, body state inference, vital sign estimation, and disease risk evaluation. It 18 consistently outperforms competitive baselines under zero-shot, partial-shot, and 19 full-shot settings, indicating broad applicability in real-world health applications. 21 Mobile and wearable sensors have been shown to be valuable for the field of healthcare by passively 22 and continuously tracking physiological signals such as photoplethysmography (PPG) for pulse, elec23 trocardiography (ECG) for heart activity, galvanic skin response (GSR), and electroencephalography 24 (EEG) for brain activity. These time series signals are beneficial for early diagnosis, personalized 25 health insights, and remote patient monitoring (Zhang et al., 2024a). 26 Recently, several foundation models have emerged for time series modeling, including Ansari et al.

27 (2024); Abbaspourazad et al. (2023); Woo et al. (2024); Foumani et al. (2024). Another common 28 approach for signal modeling involves converting raw signal series into 2D images or spectrograms, 29 using fixed-size sliding windows, followed by the use of visual encoders like Vision Transformers 30 (ViT) to extract representations for making inferences (Semenoglou et al., 2023; Wimmer & Rekabsaz, 31 2023; Vishnupriya & Meenakshi, 2018; Chun et al., 2016; Krishnan et al., 2020; Dosovitskiy et al., 32 2020). These works have significantly advanced the field and provided valuable insights, yet two 33 main issues still exists which need further exploration to fully understand their potential in wearable 34 scenarios. First, contrastive learning-based foundation models (Abbaspourazad et al., 2023) rely on 35 a predefined set of input signal types, making them unsuitable when transferring to scenarios with 36 different types and numbers of sensors. Second, while both time series foundation models (Ansari 37 et al., 2024; Zhang et al., 2022; Woo et al., 2024) and spectral-based approaches (Semenoglou et al.,
38 2023; Wimmer & Rekabsaz, 2023) attempt to address this issue by training a generic encoder that

## 20 **1 Introduction**

NORMWEAR D **Application**
A B
Pre-Training State Inference Data Stream Masking EEG T3 EEG T4 Task 1 Number of Input Channels Patching and initial embedding Intra-channel Encoder Inter-channel Encoder Intra-channel Encoder Intra-channel Encoder Inter-channel Encoder Intra-channel Encoder
…
Physical Activity Mental State Fatigue Detection C
Adaption Risk Evaluation PPG
Task 2 1) Zero-shot through representation alignment and matching of sentence similarity Layer 1 Hypertension Diabetes ECG
Task 3 EEG F7 EEG Cz EEG F8 Layer 2 Signal Embedding Brain Abnormality Heart Abnormality Muscle Abnormality Text Embedding Similarity Match Layer 3 EEG Main Tasks Layer 4 Task 4 Continue stacking ACC X
GSR
ACC Y Gyro X
PPG
Seizure Tumor area Eye state
… …
Lightweight Decoder 2) Linear Probing Vital Sign Estimation Varied number and type of sensor channels Reconstruction Blood Pressure Hemoglobin Heart Rate Figure 1: The role of our framework. Several icons from Freepik (n.d.); Zhang et al. (2024a).)
39 can handle type-agnostic series, they remain limited to processing only univariate series. Because 40 of this constraint, these previous works fail to account for the heterogeneity of multivariate input 41 data; specifically, they do not capture the complex relationships between signals from sensors located 42 on different body parts. These two limitations of recent approaches hinder their generalization and 43 usefulness for wearable health monitoring. 44 Moreover, Wearable-based multimodal physiological signals present unique challenges that distin45 guish them from general time series data, such as stock prices or weather patterns. Wearable signal 46 modalities, such as PPG and EEG, vary in characteristics like dimensionality, sampling rate, and 47 resolution, often requiring modality-specific preprocessing. Existing methods tokenize raw signals 48 (Ansari et al., 2024; Zhang et al., 2022) or convert them into image or spectral representations (Wu 49 et al., 2023; Mathew et al., 2024; Vaid et al., 2023). While effective for specific tasks, these ap50 proaches lack generalizability and fail to provide a consistent preprocessing pipeline across multiple 51 modalities. A consistent framework that accommodates diverse signal requirements is essential for 52 training deep learning-based foundation models and advancing multimodal signal analysis. 53 In this work, we present NORMWEAR, a normative foundation model, aiming to learn effective 54 wearable sensing representations, addressing the above-discussed research gaps. NORMWEAR has 55 been pretrained on more than 2.5 million multivariate wearable sensing segments, comprising total of 56 14,943 hours of sensor signal series, using publicibly avaliable datasets. We evaluated NORMWEAR 57 on 18 public downstream tasks against competitive baselines across zero-shot, few-show, and full-shot 58 settings. Overall, our contributions with the proposed NORMWEAR healthcare modeling framework 59 can be summarized as follows:
60 - To our knowledge, we are the first to develop a foundation model specifically designed for 61 wearable sensing data, capable of processing arbitrary configuration of multivariate signals 62 from sources such as the heart, skin, brain, and physical body. 63 - NORMWEAR comprises novel methodologies built upon the advanced practice in both the 64 fields of signal processing and deep learning, including (a) continuous wavelet transform 65 (CWT) based multi-scale representations for modality- and number-agnostic tokenization, 66 (b) channel-aware attention layer that enables the model to process arbitrary multivariate 67 inputs, and (c) a human sensing adapted fusion mechanism that enabled NORMWEAR to 68 achieve zero-shot inference on health related wearable sensing tasks. 69 - We are also the first to integrate and process a comprehensive wearable signals dataset 70 with varied number of input channels for training self-supervised learning algorithms, with 71 thorough downstream evaluation. These datasets cover key health applications, including 72 mental and physical state inference, vital sign estimation, and disease risk evaluation. 73 Our proposed NORMWEAR aims to provide a generalized data representation solution for smart 74 health monitoring, benefiting the general public, and serving as a fundamental tool for researchers 75 and professionals to address future healthcare challenges. We made the code and cleaned data to be 76 publicly available to spur reproducible research.

## 77 **2 Related Work**

78 Foundation models have emerged as a transformative paradigm in machine learning, enabling 79 generalizable and reusable representations across diverse downstream tasks (Bommasani et al., 2022). 80 In the time series domain, recent works (Ansari et al., 2024; Foumani et al., 2024; Abbaspourazad 81 et al., 2023; Narayanswamy et al., 2024) have demonstrated success in tasks such as forecasting, 82 classification, and anomaly detection. However, their generalizability to health-related wearable 83 signals remains limited due to the lack of in-depth evaluation, reliance on specific sensor types (Wang 84 et al., 2025; Jiang et al., 2024; Yang et al., 2023) and univariate data (Pillai et al., 2024; McKeen et al., 85 2024), as well as the inability to handle the heterogeneity of multivariate wearable signals. In contrast, 86 NORMWEAR builds upon these principles by introducing a modeling framework that is agnostic to 87 the sensor modality and number of input channels, as stated in section 1, and is presented in details 88 in section 3. NORMWEAR has been evaluated on 18 digital healthcare tasks and demonstrate peak 89 performance against solid time series modeling baselines, including common statistical approach, 90 SoTA model in time series with self-supervised learning (Zhang et al., 2022), SoTA spectrum based 91 modeling approach (Wu et al., 2023), and SoTA time series forecasting model (Ansari et al., 2024). 92 Our work not only generalizes to arbitrary sensor configurations but also ensures compatibility across 93 multivariate data, addressing key limitations of earlier approaches.

## 94 **3 Method** 95 **3.1 Dataset Construction For Model Pretraining And Downstream Evaluation**

96 We curated a collection of 9 publicly available datasets (Table 5) exclusively for model pretraining, 97 resulting in approximately 230,962 multivariate time series segments, comprising 4,294 hours of 98 total sensor signal series, across various modalities, including PPG, ECG, EEG, GSR, PCG, and 99 inertial measurement unit (IMU) data. To address the dataset size limitation, we then applied herustic 100 data augmentation (algorithm 1) to expand the pretrain dataset to 2.5 million segments, comprising 101 14,943 hours of total sensor signal series. Notably, each sample segment may contain a variable 102 number of input channels depending on the sensor signals provided by the respective datasets. This 103 input configuration aligns seamlessly with our model's design, which is optimized to flexibly handle 104 arbitrary numbers and configurations of sensor signal inputs. 105 To prevent potential data leakage in downstream tasks, we evaluate our model's transferability using 106 an additional 11 publicly available datasets encompassing 18 modeling tasks, which include affective 107 state classification, physical state recognition, biological estimation, and disease risk evaluation. 108 Details about the datasets is presented in Table 4.

## 109 3.2 **Tokenization**

110 Tokenization is a fundamental term widely used in natural language processing. In the context of 111 wearable sensing, we leverage this term to represent the stage of signal processing before sending the 112 processed data to the deep learning-based encoder. Spectral methods, which utilize the short-time 113 Fast Fourier Transform (FFT) (Brigham, 1988) with a sliding window to compute spectrograms, 114 are widely regarded as the benchmark approach for tokenization. However, due to the inherent 115 trade-off between time and frequency resolution, the spectral representation with a fixed window size 116 cannot be generalized. This is because the window size has to be modulated accordingly when the 117 modality varies. To enhance transferability, we propose a well-designed signal processing pipeline 118 that preserves information in both the frequency and time domains across multiple scales. We begin 119 by calculating the first and second derivatives for each single signal series, as suggested by Slapnicar ˇ 120 et al. (2019), followed by computing the continuous wavelet transform (CWT) on both the raw and 121 derivative series, resulting in three scalograms. Then, we stack the three scalograms to form data 122 in RGB-image-like format. The derivatives capture the rate of signal change at different moments, 123 while the wavelet transform provides a multi-resolution encoding that preserves information from 124 both the time and frequency domains Torrence & Compo (1998). For the wavelet transform, we 125 use the Mexican Hat wavelet for signal convolution, as recommended by previous studies (Burke 126 & Nasor, 2004; Hosni & Atef, 2023; Hassani, 2021; Negi et al., 2024; Nedorubova et al., 2021b). 127 We apply scales ranging from 1 to 64, following the guidance of (Sengupta et al., 2022; Nedorubova 128 et al., 2021a), which sufficiently covers most frequency bands of interest for physiological signals. 129 Finally, this RGB-like scalogram is divided into patches, which is treated in the same way as tokens 130 in an ViT (Dosovitskiy et al., 2020). In this way, this tokenization approach can be applied to various 131 types of sensing signals without sensor-specific adjustments or reconfigurations.

## 132 3.3 **Share-Weighted Encoder**

133 Rather than concatenating tokens from all channels into a single long sequence and processing them 134 with a full attention transformer, we treat each channel of the multivariate signal as an independent

(b) Channel-Aware Fusion Target
(1) All-Attention O(dᐧ(LᐧC)2
)
Concat Fusion Block

(2) Cross-Attention O(dᐧLᐧC

2
)

Fusion Block Fusion Block Light-Weight Decoder
(3) [CLS]-Attention O(dᐧC
2

)
Fusion Block Fusion Block Share-weight Transformer Encoder (**Intra-channel Encoder**)
Fusion Block (**Inter-channel Encoder**)
Signal Encoder x12
(4) Mean-Pooling Attention O(dᐧ(LᐧC+C
2
))
Yield the best performance **Remarks: (L >> C)**
- d: embedding size - L: number of patches
- C: number of channels Patch Projection Tokenization
(a) Masking Strategies Input
(1) Temporal + 
Scale Original Spectrogram
(2) Scale only(3) Temporal only
(4) Unstructured CLS Token Patch token ACC_X ACC_Y PPG
Yield the best performance Figure 2: Overview of the pretrain pipeline.

## 141 3.4 **Channel-Aware Attention With Liaison Special Token**

142 Following the tokenization step, we adopt common reconstruction-based pretraining strategies from 143 Masked Auto Encoder (MAE) (He et al., 2021; Huang et al., 2023; Zhang et al., 2023), where input 144 tokens are randomly masked and the model is trained to reconstruct the original time series using 145 mean squared error (MSE) loss. Inspired by Huang et al. (2023), we experiment with four masking 146 strategies, as shown in Figure 2 (a), including masking on (1) temporal and scale, (2) scale only, 147 (3) temporal only, and (4) unstructured axes. We observe that the temporal and scalar masking 148 yields the best performance for the downstream tasks. For the model architecture, we construct the 149 backbone of our proposed framework with a convolutional patching layer followed by 12 standard 150 Transformer blocks (Vaswani et al., 2023). For the same reason, NORMWEAR uses a lightweight 151 decoder consisting of 2 Transformer blocks, combined with a linear projection layer and a convolution 152 layer to reconstruct the raw physiological signals both temporally and spatially. We also prepend a 153 [CLS] token to each signal channel, following standard practice in transformer models, for learning a 154 global representation of the input sequence for that channel. 155 Another important point to consider is that although empirical studies (Nie et al., 2023; Abbaspourazad 156 et al., 2023) show that channel-independent structures effectively capture local patterns, they fail to 157 account for relationships across channels. To address this, we use the [CLS] token from each signal 158 channel as a liaison token, allowing them to exchange information through the channel-aware fusion 159 layer afrer every other encoder block. We explore several fusion approaches and different design of 160 liaison token as shown in Figure 2 (b), with each method described below:
161 (1) **All-Attention Fusion:** This approach involves concatenating all tokens from each modality 162 without considering their individual properties and fusing the information through a self-attention 163 module. However, this method requires quadratic computation time, as every token passes through 164 the self-attention module, making it impractical for real-world applications. 165 (2) **Cross-Attention Fusion:** In addition to the cross-attention mechanism used in Cross-ViT (Chen 166 et al., 2021), we introduce a slight modification to fit in our problem setting. We propose a symmetric 167 fusion method, using the [CLS] token from each modality as an intermediary to exchange information 168 between the patch tokens of another modality, then projecting the information back to its original 169 modality in the subsequent Transformer layer. While this strategy is efficient, it restricts the model 170 to handling only two time series signals or modalities, which deviates from our goal of building a 171 general model capable of processing an arbitrary number of channels.

Question of Interest, e.g.

What is the state of heartbeat abnormalities?

Classes, e.g. - Heart rate at normal state.

- Heart rate at abnormal state. … …
- High risk of hypertension Text Encoder Pre-trained Freeze Text Encoder Aggregation Module Query Inference Vanilla Embeddings Key Mean S
TC
1 S
TC
2S
TC
3 C1 C2 C3 Similarity Prediction X Relevance Score Fused Embedding Y
^
Backprop Linear Mapping
+
Weighted Average ACC_X PPG Embedding Size Patches Backbone Encoder Value Linear Mapping Recency Score STDEV
Sampling Likelihood Parameter Sampling Linear Mapping Importance Score
172 (3) **[CLS]-Attention Fusion** The [CLS] token serves as an abstract global representation for each 173 signal modality. Here, we propose a hybrid fusion approach. We stack the [CLS] tokens from all 174 signal modalities and perform feature fusion using a self-attention mechanism. The fused [CLS] token 175 is then reattached to its original channel, enabling the newly learned information to be propagated to 176 each patch token in subsequent transformer encoder layers. 177 (4) **Mean-Pooling Fusion** Similar to the [CLS]-Attention Fusion approach, we employ mean-pooling 178 within each channel instead of using the [CLS] token as an abstract global representation. 179 Our empirical results show that [CLS]-attention fusion achieves the best downstreaming performance 180 for our proposed NORMWEAR model. Details of all the ablation studies are reported in Appendix C. 181 Beyond accuracy, we want to emphasize that the [CLS]-Attention Fusion design is highly flexible. 182 This flexibility arises from the fact that self-attention is length-flexible and permutation-invariant 183 (Vaswani et al., 2023). Consequently, it integrates naturally with our shared-weight encoder, allowing 184 the model to accommodate a variable number of sensor channels presented in any order. We provide 185 additional empirical evidence of NormWear's permutation invariance in Table 12, Appendix C.

## 186 3.5 **Sensor-Semantic Representation Alignment**

187 Zero-shot inference is an important aspect to evaluate foundation model. We evaluate our model in 188 this setting by retrieving the closest text-derived label for each unseen task in the shared embedding 189 space. Specifically, to unify information across diverse modalities, we incorporate a representation 190 alignment objective that encourages the embeddings of physiological sensor data to reside in the same 191 latent space as paired textual descriptions. Once this shared space is established, it naturally supports 192 zero-shot inference by allowing unseen sensor inputs to be interpreted through their proximity to 193 text-derived anchors, without additional task-specific training. Several important works in this 194 direction focusing on domains of vision-language Radford et al. (2021), audio-language Wu et al. 195 (2023), and motion-language (Zhang et al., 2024b). These works leverage end-to-end training to bind 196 their modality of interest into semantic space. In this work, we extend this methodology to explore 197 NORMWEAR's ability to generalize across unseen datasets and tasks. 198 Building on prior work in representation alignment, we notice that in healthcare-related tasks where 199 flexible inference across diverse scenarios is often required, the ground truth labels often have 200 substantial overlap. For instance, depression is inferred from stress levels (LeMoult, 2020), and 201 running and cycling produce similar IMU signals (Li et al., 2019). Due to these nested relationships, it 202 create potential challenge to representation alignment when using contrastive learning, which requires 203 clearly defined positive and negative pairs. To address this, we first propose a novel way to fuse the 204 signal representations together with improved qualities, then align the representation with vector 205 distance as an auxiliary loss for contrastive learning method. In addition, to reduce computation 206 cost and counteract the issue of catastrophic forgetting (Li et al., 2023), we use off-the-shelf frozen 207 encoders for both signal and text modalities. 208 Human physiological signals are task-specific, dynamic, and often weakly labeled (He et al., 2018; 209 Kim et al., 2022; Qian et al., 2021; Ma et al., 2021). To address these characteristics, we introduce 210 three complementary scoring mechanisms during feature aggregation: *relevance scores* prioritize 211 patches aligned with the task objective (e.g., IMU for activity recognition), guided by query sentences 212 such as "What activity is the subject doing?"; *recency scores* emphasize recent segments to better 213 reflect the current physiological or emotional state (Roelofs, 2017; Chowdhury et al., 2020; Chaudhury 214 et al., 2021); and *importance scores* weigh signal segments that contain meaningful or transient 215 patterns often buried in weakly labeled sequences. Together, these scores guide the MSiTF fusion 216 module to generate compact, task-aware representations. This design is inspired by memory-stream 217 retrieval mechanisms (Park et al., 2023) and is tailored to the demands of human-centered sensing 218 tasks such as risk assessment, affect detection, and activity recognition. 219 **Memory Stream inspired Temporal Fusion (MSiTF).** Our Aggregation or Fusion Module, MSiTF, 220 is designed to addresses the above-discussed three challenges through three scores discussed below. 221 Specifically, we denote f as the function that takes the semantic embedding of query sentence q and backbone output H ∈ R
P ×E 222 as input, where P is the patch size and E is the embedding size, thus having the final fused representation f(*q, H*) = Yˆ ∈ R
E 223 .

224 We define the *Relevance* score as the cross attention between the key representations of each sensor 225 time step and the query sentence embedding, obtained from a pretrained language model (Clinical 226 TinyLlama (Muzammil, 2021)). This mechanism allows the model to identify distinct but contextually 227 relevant segments in the sensor input. For the *Recency* score, we use an exponential decay function to 228 reflect the intuition that recent time steps are more important than earlier ones. Finally, we consider 229 the importance score IMP in this case to be whether to keep the representation at each time step or not.

In order to achieve this, we assign binary parameters to each time step, denoted as θt = p(vt) ∈ R
2 230 where vt ∈ R
E 231 is the representation vector at time step t and p is a trainable linear transformation 232 function which will be optimized during pretraining. We then have the importance score for each 233 patch defined as

$$W_{i m p}(t)=\operatorname*{arg\,max}_{i\in\{0,1\}}{\frac{\exp\left(\Big(\log(\theta_{t,i})+\epsilon\Big)/\tau\right)}{\sum_{j\in\{0,1\}}\exp\left(\Big(\log(\theta_{t,j})+\epsilon_{j}\Big)/\tau\right)}}$$
$$(1)$$

234 where ϵ is the noise term sampled from Gumbel distribution (Jang et al., 2017), and τ is the 235 temperature controlling the sharpness of the softmax function. Because arg max is not a differentiable 236 function, we will directly take the resulting probability corresponding to index at j = 1 to be the 237 *importance* score, with τ being set to a small number to push the result closer to one hot vector 238 from the softmax function. As a result, this logit function will determine to what extent to activate 239 the gate during forward pass on each patch of the input signals. The final score for each patch is 240 the summation of the three scores as described above. This score will be treated as the weight for 241 aggregating the representations from all the patches to form the fixed length embedded output (vector 242 with size of 768 in our case). 243 Once the signal embeddings are aggregated, we adopt a variational-inspired approach (Kingma &
244 Welling, 2022). This design injects stochasticity into the representation, encouraging the model to 245 explore and capture nuanced variations in semantic representations. Finally, we leverage contrastive 246 learning with auxiliary loss on vector distance to train the MSiTF module with a projection layer to 247 text representation on the pretraining datasets. The sentence template formation and training details 248 are presented in Appendix B.5.

## 249 **4 Experiments**

250 NORMWEAR is pretrained exclusively on the data shown in Table 5. In this section, we present a 251 comprehensive evaluation across 11 downstream publicly available datasets, focusing on 18 widely252 recognized digital healthcare tasks. We evaluate the methods following order of zero-shot capability, 253 partial-shot learning, and full-shot learning.

## 254 4.1 **Selection Of Baselines Covering Representative Modeling Strategies**

255 Modeling multivariate wearable signals with arbitrary input channels and sensor types, such as those 256 capturing activities of heart, brain, and body physical motions, presents unique challenges, as no 257 universally recognized open-source baseline or state-of-the-art (SoTA) model exists in this domain. 258 To evaluate our approach, we selected diverse and representative baselines (as shown in Table 3). 259 In the literature, various modeling strategies have been proposed. Firstly, early approaches involved 260 handcrafting statistical features, which was a widely adopted practice in signal processing (Yan et al.,
261 2023a; Reyes-Ortiz et al., 2012; Mikelsons et al., 2017). We include this simple baseline as sanity 262 check. Secondly, since sensory data can be naturally represented as time series (Woo et al., 2024; 263 Semenoglou et al., 2023), we benchmarked our model against Chronos (Ansari et al., 2024) , as well

AUC ROC AUC PR Accuracy EEG
EEG
EEG
0.753 0.792 0.815 0.831 0.859 0.731 NormWear (Ours) TF-C CLAP Chronos Statistical 0.762 0.668 0.517 Disease Disease Disease State 0.626 0.649 State State 0.802 0.793 0.787 0.807 Micro Micro Micro Macro Macro Macro

## 275 **4.2 Zero-Shot Evaluation**

276 We achieve zero-shot inference by pretraining our proposed novel temporal fusion module on the task 277 of representation alignment. We include the SoTA spectral-based model CLAP Wu et al. (2023) as a 278 baseline to provide a more comprehensive comparison of the results. For CLAP, we experimented 279 with both Manhattan distance (MD) and dot product (DP) as similarity metrics during inference. We 280 observe that there are no statistically significant differences in performance when using MD and DP 281 for label retrieval in CLAP. From table 1, we could observe that overall, NORMWEAR equipped 282 with MSiTF outperforms the baselines. We compare NORMWEAR with a few ablations by removing 283 importance score (w/o IMP) and removing text augmentation (w/o text aug). We can observe that 284 performance drop after removing each of the above components, verifying their respective importance 285 in improving generalization across various downstream tasks. We present this outcome to demonstrate 286 the zero-shot capability in the wearable signal domain, an aspect not present in recent studies. We 287 also hope this outcome could potentially provide a new perspective that can help drive progress in 288 this direction within the field.

Table 1: Zero-shot performance on the downstream datasets, with AUC ROC being reported. The last two columns show the average across the tasks and across group types respectively.

| DriverFatigue                    | GAMEEMO Epilepsy (eye open state)   | Epilepsy (eye relaxation)   | sy (health area) Epilep   | sy (tumor area) Epilep   | sy (seizure) Epilep   | PhysioNet EMG Micro Avg.   | Macro Avg.          |                     |
|----------------------------------|-------------------------------------|-----------------------------|---------------------------|--------------------------|-----------------------|----------------------------|---------------------|---------------------|
| CLAP - MD 45.3 62.8 58.5         | 53.1 44.9                           | 45.1                        | 47.6                      | 30.5                     | 84.9                  | 59.4 41.8 46.0 57.4        | 22.9 55.4 50.4 51.2 |                     |
| CLAP - DP                        | 50.7 52.3 61.1                      | 51.6 54.4                   | 41.9                      | 58.6                     | 46.4                  | 74.3                       | 52.2 41.4 50.6 58.9 | 42.7 38.3 51.7 52.2 |
| before bind                      | 44.1 48.2 52.1                      | 48.4 54.1                   | 62.6                      | 53.9                     | 52.5                  | 24.6                       | 48.8 49.6 46.3 56.8 | 54.3 48.2 49.6 49.4 |
| NORMWEAR w/ MSiTF 55.8 71.2 57.2 | 51.0 55.7                           | 61.3                        | 67.6                      | 55.8                     | 66.0                  | 57.1 62.5 70.0 59.0        | 63.1 70.1 61.6 61.5 |                     |
| - w/o IMP                        | 56.2 70.3 55.4                      | 49.8 54.0                   | 56.5                      | 66.9                     | 57.3                  | 52.9                       | 56.5 54.3 61.7 60.7 | 73.4 65.2 59.4 59.6 |
| - w/o text aug 54.8 65.8 55.2    | 49.2 31.0                           | 58.4                        | 58.6                      | 32.8                     | 58.1                  | 50.2 52.6 50.8 50.6        | 47.7 33.6 50.0 51.4 |                     |
| - w/o refine                     | 59.5 72.8 42.7                      | 57.3 50.6                   | 69.0                      | 43.3                     | 50.5                  | 74.8                       | 48.3 38.8 44.6 44.1 | 72.4 75.7 56.3 56.6 |
| BP (HTN) PPG-                    | BP (CVA) PPG-                       | BP (CVD) PPG-               | Abnormal ECG-             |                          |                       |                            |                     |                     |
| BP (DM) PPG-                     |                                     |                             |                           |                          |                       |                            |                     |                     |
| UCI-HAR                          |                                     |                             |                           |                          |                       |                            |                     |                     |
| Model                            | WESAD                               |                             |                           |                          |                       |                            |                     |                     |

289 4.3 **Partial-shot and Full-shot Evaluation**
290 We evaluate the learned representations using linear probing through supervised training on each 291 downstream dataset, and report performance on the corresponding held-out test set. To ensure 264 as a self-supervised framework TF-C (Zhang et al., 2022). Finally, the spectrum-based modeling 265 methods (Vishnupriya & Meenakshi, 2018; Chun et al., 2016; Krishnan et al., 2020) are widely 266 used for signal modeling. Therefore, we incorporate CLAP (Wu et al., 2023) into baselines that 267 has demonstrates SoTA performance in spectrogram-based modeling. Regarding the comparison 268 with concurrent works proposing foundation models for a specific sensor modality, we leverage 269 PaPaGei (Pillai et al., 2024) for PPG datasets, ECG-FM (McKeen et al., 2024) for ECG datasets, and 270 CBraMod (Wang et al., 2025) for EEG datasets. These baselines span distinct paradigms, providing 271 a solid foundation to demonstrate the strengths of our model in wearable signal tasks. For uni-modal 272 baselines like Chronos and CLAP, we feed each signal separately into model and concatenate their 273 representations after the forward pass. This ensures that all models have the same field of view, 274 making the comparison fair.

Table 2: **Detailed performance on various downstream wearable-signal-based health related** applications under full-shot linear probing evaluation.

Downstream Tasks Statistical Chronos CLAP TF-C Modality-Specific NORMWEAR **(Ours)**

WESAD 66.213 71.489 72.383 69.865 56.656 **76.060**

UCI-HAR 95.784 91.593 96.420 96.892 - **98.954**

DriverFatigue 63.249 76.722 61.889 66.882 **80.430** 74.292

Activity Recognition Avg. 75.082 79.935 76.897 77.880 - **83.102**

Epilepsy (eye open state) 82.489 82.41 85.094 89.153 90.436 **92.743** Epilepsy (eye relaxation) 87.457 88.218 89.867 94.416 **95.552** 94.828

Epilepsy (health area) 86.274 81.08 83.711 85.619 88.065 **88.541**

Epilepsy (tumor area) 82.816 81.034 83.644 86.348 **87.258** 87.197 Epilepsy (seizure) 88.272 97.572 **97.734** 93.998 94.616 97.053

GAMEEMO 51.009 53.747 52.551 **56.275** 55.420 54.937

EEG Main Tasks Avg. 79.720 80.677 82.100 84.302 85.225 **85.883**

ECG-Abnormal 97.092 98.585 97.23 98.275 89.898 **99.140**

PPG-BP (HTN) 59.499 52.425 56.757 **65.229** 61.839 62.341 PPG-BP (DM) 47.823 51.164 42.455 **57.883** 55.668 55.893

PPG-BP (CVA) 71.250 50.278 51.667 58.125 **73.125** 70.625

PPG-BP (CVD) 51.219 58.31 50.91 **58.674** 49.066 51.773 PhysioNet EMG **99.309** 61.6 98.627 78.308 - 99.216

Risk Evaluation Avg. 71.032 62.060 66.274 69.416 - **73.165**

Noninvasive-BP 92.310 91.79 91.922 87.481 90.596 **92.420**

PPG-Hgb 94.219 **95.005** 94.291 93.408 94.912 94.632

Fetal-fPCG 98.929 99.048 **99.195** 99.077 - 99.072

Vital Signs Avg. 95.153 95.281 95.136 93.322 - **95.375**

Micro Avg. 78.623 76.782 78.130 79.773 - **82.762** Macro Avg. 80.247 79.488 80.103 81.230 - **84.381**

292 fair comparison, we use a unified evaluation protocol with identical hyperparameter settings and 293 implementation across all models and the dataset (Yuan et al., 2024). This design ensures that 294 performance differences are not due to variations in learning rate, regularization, or data augmentation 295 (Oliver et al., 2018). Specifically, the classification tasks, using logistic regression, are solved by 296 Newton's method with conjugate gradient, with AUC ROC being reported as main metric. The 297 regression (vital signs) tasks, using ridge regression, are solved by Cholesky's method with closed 298 form solution, with relative accuracy being reported. For partial-shot evaluation, we leverage 10% of 299 the training data for the linear probing, and detailed performance result is presented in Table 11. The 300 full-shot evaluation results is presented in Table 2. All scores are the higher the better.

301 From Figure 4, Table 2, and Table 15, we observe that NORMWEAR consistently achieves peak 302 performance across all task groups, including activity recognition, EEG signal analysis, disease risk 303 evaluation, and vital sign estimation. Furthermore, its leading performance remains consistent across 304 various evaluation metrics. Based on the macro-averaged total score across task groups, NORMWEAR 305 delivers a 3.9% improvement over the state-of-the-art (SoTA) time-series self-supervised learning 306 framework (Zhang et al., 2022), a 5.3% improvement over the SoTA spectrum-based modeling 307 method (Wu et al., 2023), a 6.1% improvement over SoTA time-series forecasting models with LLM 308 backbones (Ansari et al., 2024), and a 5.2% improvement over standard statistical baselines. On 309 larger datasets, NORMWEAR significantly outperforms the statistical baseline by 9.0% and 7.5% for 310 activity recognition and EEG brain activity monitoring tasks, respectively. On smaller datasets, it 311 still achieves peak performance in disease risk evaluation. For vital sign estimation, all methods 312 yield comparable results, suggesting inherent challenges in these regression tasks that warrant further 313 investigation but are beyond the scope of this study.

| Table 3: Baselines                     |                                                                                           |
|----------------------------------------|-------------------------------------------------------------------------------------------|
| Baseline Methods                       | Modeling Strategies                                                                       |
| Modality Specific (Zhang et al., 2022) | PaPaGei (Pillai et al., 2024), ECG-FM (McKeen et al., 2024), CBraMod (Wang et al., 2025). |
| TF-C (Zhang et al., 2022)              | SoTA in TS SSL; modeling time and frequency domain information at same time.              |
| CLAP (Wu et al., 2023)                 | SoTA in audio modeling; process signal as spectrogram                                     |
| Chronos (Ansari et al., 2024)          | SoTA in TS forecasting, leverage LLM for modeling                                         |
| Statistical approach                   | Reserve full interpretability                                                             |

314 When comparing with recent modality specific foundation models, NormWear's main benefit is 315 that it capture cross-modal relationships, making it more versatile for wearable health tasks. While 316 it sacrifices modality-specific optimization for adaptability, this may slightly reduce performance 317 in highly specialized tasks. Single-signal models excel in their domains due to deeper modality318 focused training. Instead of maximizing single-modality data, we prioritize signal diversity for better 319 generalization. Benchmarking shows that NormWear, trained on a smaller dataset than EEG-only 320 models, still achieves competitive results, highlighting the effectiveness of our pre-training approach. 321 These findings illustrate NORMWEAR's capacity to balance consistency and adaptability across a 322 diverse range of tasks and conditions. By excelling across standard benchmarks while addressing the 323 intricacies of varied applications, NORMWEAR exemplifies the philosophy of a foundation model: a 324 reliable generalist capable of performing robustly across both typical and challenging scenarios.

Su mm arized P
erfo rmance
+0.6%
+6.8%+4.5%
+7.4%
Zero-shot +7.0% Partial-Shot Full-Shot Statistics Chronos CLAP TF-C NormWear Approaches 55 60 65 70 75 80 Baseline.

+Raw-input
(CA+Mask).

+CWT (Best, Final NormWear).

 Remove CA.

 Switch to Cross Attn Fusion.

 Switch to Mean Pool Fusion.

 Switch back to [CLS] Liaison Attn. (Best repeat).

 Switch to Unstructured Masking.

 Switch to Time Mask.

 Switch to Scale Mask.

 Switch back to Structued Masking. (Best repeat).

0 1 2 3 4

% Cumulative Performance Change

## 325 **5 Conclusion And Discussion**

326 **Conclusion.** In this work, we mainly propose a foundation model for wearable physiological signals. 327 NORMWEAR is a practical tool that could serve as a starting point for researchers and clinicians when 328 tackling a problem with wearable based signal data. Our proposed model could extract informative 329 representations from raw signal series, which can be leveraged for further machine learning modeling, 330 clustering, embedding vector-based information retrieval, and deployment of real-time health states 331 monitoring with minimal tuning. We've justified the utilizability and generalization of NORMWEAR 332 through an extensive evaluation of various ubiquitous health applications. As for future works, it is 333 important to leverage our framework on larger scale clinical applications and explore the applicability 334 of embedding vectors as state representations for intervention modeling problems that comprise the 335 decision-making process. 336 **Limitation and Future Work.** We acknowledge several limitations to be addressed in future work. 337 (1) The representation alignment component is currently trained on a limited set of healthcare-related 338 objectives, and expanding the pretraining corpus with more diverse semantic labels may improve 339 generalization. (2) While our design supports classification tasks well, adapting the framework 340 for regression remains an open challenge, and future work may explore alternative formulations 341 beyond label discretization. (3) NormWear currently focuses on physiological signals with relatively 342 narrow frequency bands; extending its applicability to higher-frequency modalities such as audio or 343 lower-resolution clinical summaries is a promising direction. 344 **Broad Impact.** NORMWEAR is the first foundation model tailored for multivariate physiological 345 signals that supports a wide range of wearable health tasks across sensor modalities, device types, and 346 clinical applications. Through a unified CWT-based tokenization pipeline and a channel-aware fusion 347 mechanism, it enables robust, modality-agnostic representation learning. Our extensive evaluation 348 across zero-shot, partial-shot, and full-shot settings demonstrates NormWear's strong generalizability 349 and practical relevance. We believe NormWear provides a valuable resource for advancing foundation 350 modeling in digital health and promoting more unified benchmarks in the community.

## 351 **Ethics Statement**

352 This study contains applications in the field of healthcare. We ensured that all the data being used 353 during pretraining and evaluations were made publicly available by the original authors, and all these 354 works were cited properly.

## 355 **Reproducibility Statement**

356 The full code base is submitted in supplementary material referred to as *NormWear_main.zip*, 357 comprising all the scripts for exploratory data analysis and preprocessing, model construction, 358 pretraining, downstream evaluation, result analysis, and all the visualizations that are described in 359 this paper. The GitHub repository containing all the documentation will be published simultaneously 360 with the paper.

## 361 **References**

362 Abbaspourazad, S., Elachqar, O., Miller, A. C., Emrani, S., Nallasamy, U., and Shapiro, I. Large-scale 363 training of foundation models for wearable biosignals. *arXiv preprint arXiv:2312.05409*, 2023.

364 Abuzairi, T., Vinia, E., Yudhistira, M. A., Rizkinia, M., and Eriska, W. A dataset of hemoglobin blood 365 value and photoplethysmography signal for machine learning-based non-invasive hemoglobin 366 measurement. *Data in Brief*, 52:109823, 2024. ISSN 2352-3409. doi: https://doi.org/10.1016/j.dib. 367 2023.109823. 368 Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, 369 J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023. 370 Alakus, T. B., Gonen, M., and Turkoglu, I. Database for an emotion recognition system based on eeg 371 signals and various computer games–gameemo. *Biomedical Signal Processing and Control*, 60: 372 101951, 2020. 373 Alzahab, N. A., Di Iorio, A., Apollonio, L., Alshalak, M., Gravina, A., Antognoli, L., Baldi, M., 374 Scalise, L., and Alchalabi, B. Auditory evoked potential eeg-biometric dataset, 2022. 375 Andrzejak, R. G., Lehnertz, K., Rieke, C., Mormann, F., David, P., and Elger, C. E. Indications 376 of nonlinear deterministic and finite-dimensional structures in time series of brain electrical 377 activity: Dependence on recording region and brain state [dataset]. *Physical Review E*, 2023. doi: 378 10.34810/data490. URL https://doi.org/10.34810/data490. 379 Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, 380 S. S., Arango, S. P., Kapoor, S., et al. Chronos: Learning the language of time series. *arXiv* 381 *preprint arXiv:2403.07815*, 2024. 382 Bajaj, N., Carrión, J. R., and Bellotti, F. Phyaat: Physiology of auditory attention to speech dataset. 383 *arXiv preprint arXiv:2005.11577*, 2020. 384 Beh, W.-K., Wu, Y.-H., and Wu, A.-Y. A. Maus: A dataset for mental workload assessment on n-back 385 task using wearable sensor, 2021. URL https://dx.doi.org/10.21227/q4td-yd35. 386 Bhaskaran, A., J, S. K., George, S., and Arora, M. Heart rate estimation and validation algorithm 387 for fetal phonocardiography. *Physiological Measurement*, 43(7):075008, jul 2022. doi: 10.1088/ 388 1361-6579/ac7a8c. URL https://dx.doi.org/10.1088/1361-6579/ac7a8c. 389 Bommasani, R., Hudson, D. A., Adeli, E., Altman, R., Arora, S., von Arx, S., Bernstein, M. S., 390 Bohg, J., Bosselut, A., Brunskill, E., Brynjolfsson, E., Buch, S., Card, D., Castellon, R., Chatterji, 391 N., Chen, A., Creel, K., Davis, J. Q., Demszky, D., Donahue, C., Doumbouya, M., Durmus, E., 392 Ermon, S., Etchemendy, J., Ethayarajh, K., Fei-Fei, L., Finn, C., Gale, T., Gillespie, L., Goel, K., 393 Goodman, N., Grossman, S., Guha, N., Hashimoto, T., Henderson, P., Hewitt, J., Ho, D. E., Hong, 394 J., Hsu, K., Huang, J., Icard, T., Jain, S., Jurafsky, D., Kalluri, P., Karamcheti, S., Keeling, G., 395 Khani, F., Khattab, O., Koh, P. W., Krass, M., Krishna, R., Kuditipudi, R., Kumar, A., Ladhak, 396 F., Lee, M., Lee, T., Leskovec, J., Levent, I., Li, X. L., Li, X., Ma, T., Malik, A., Manning, C. D., 397 Mirchandani, S., Mitchell, E., Munyikwa, Z., Nair, S., Narayan, A., Narayanan, D., Newman, 398 B., Nie, A., Niebles, J. C., Nilforoshan, H., Nyarko, J., Ogut, G., Orr, L., Papadimitriou, I., Park, 399 J. S., Piech, C., Portelance, E., Potts, C., Raghunathan, A., Reich, R., Ren, H., Rong, F., Roohani, 400 Y., Ruiz, C., Ryan, J., Ré, C., Sadigh, D., Sagawa, S., Santhanam, K., Shih, A., Srinivasan, K., 401 Tamkin, A., Taori, R., Thomas, A. W., Tramèr, F., Wang, R. E., Wang, W., Wu, B., Wu, J., Wu, 402 Y., Xie, S. M., Yasunaga, M., You, J., Zaharia, M., Zhang, M., Zhang, T., Zhang, X., Zhang, Y., 403 Zheng, L., Zhou, K., and Liang, P. On the opportunities and risks of foundation models, 2022. 404 URL https://arxiv.org/abs/2108.07258. 405 Bousseljot, R., Kreiseler, D., and Schnabel, A. Nutzung der ekg-signaldatenbank cardiodat der ptb 406 über das internet. In *PTB-XL, a large publicly available electrocardiography dataset*, 2009. URL 407 https://api.semanticscholar.org/CorpusID:111121953. 408 Brigham, E. O. *The fast Fourier transform and its applications*. Prentice-Hall, Inc., 1988. 409 Burke, M. and Nasor, M. Wavelet based analysis and characterization of the ecg signal. *Journal of* 410 *Medical Engineering & Technology*, 28(2):47–55, 2004. 411 Carmona, C. U., Aubet, F.-X., Flunkert, V., and Gasthaus, J. Neural contextual anomaly detection for 412 time series, 2021. URL https://arxiv.org/abs/2107.07702. 413 Caron, M., Misra, I., Mairal, J., Goyal, P., Bojanowski, P., and Joulin, A. Unsupervised learning of 414 visual features by contrasting cluster assignments, 2021. URL https://arxiv.org/abs/2006. 415 09882.

416 Chaudhury, S., Yu, C., Liu, R., Kumar, K., Hornby, S., Duplessis, C., Sklar, J. M., Epstein, J. E.,
417 and Reifman, J. Wearables detect malaria early in a controlled human-infection study. *IEEE* 418 *Transactions on Biomedical Engineering*, 69(6):2119–2129, 2021.

419 Chen, C.-F., Fan, Q., and Panda, R. Crossvit: Cross-attention multi-scale vision transformer for 420 image classification, 2021. 421 Chowdhury, M. H., Shuzan, M. N. I., Chowdhury, M. E., Mahbub, Z. B., Uddin, M. M., Khandakar, 422 A., and Reaz, M. B. I. Estimating blood pressure from the photoplethysmogram signal and 423 demographic features using machine learning techniques. *Sensors*, 20(11):3127, 2020.

424 Chun, S. Y., Kang, J.-H., Kim, H., Lee, C., Oakley, I., and Kim, S.-P. Ecg based user authentication 425 for wearable devices using short time fourier transform. In *2016 39th international conference on* 426 *telecommunications and signal processing (tsp)*, pp. 656–659. IEEE, 2016. 427 Dar, M. N., Rahim, A., Akram, M. U., Khawaja, S. G., and Rahim, A. Yaad: young adult's affective 428 data using wearable ecg and gsr sensors. In *2022 2nd International Conference on Digital Futures* 429 *and Transformative Technologies (ICoDT2)*, pp. 1–7. IEEE, 2022.

430 Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, 431 M., Minderer, M., Heigold, G., Gelly, S., et al. An image is worth 16x16 words: Transformers for 432 image recognition at scale. *arXiv preprint arXiv:2010.11929*, 2020. 433 Esmaili, A., Kachuee, M., and Shabany, M. Nonlinear cuffless blood pressure estimation of healthy 434 subjects using pulse transit time and arrival time. *IEEE Transactions on Instrumentation and* 435 *Measurement*, 66(12):3299–3308, 2017. 436 Fekri Azgomi, H., Branco, L. R. F., Amin, M. R., et al. Regulation of brain cognitive states 437 through auditory, gustatory, and olfactory stimulation with wearable monitoring. Scientific Re438 *ports*, 13:12399, 2023. doi: 10.1038/s41598-023-37829-z. URL https://doi.org/10.1038/
439 s41598-023-37829-z. 440 Foumani, N. M., Tan, C. W., Webb, G. I., and Salehi, M. Improving position encoding of transformers 441 for multivariate time series classification. *Data Mining and Knowledge Discovery*, 38(1):22–48, 442 2024. 443 Freepik. Hypertension; blood pressure gauge; motion sensor; student sleeping in class; diabetes; 444 blood cells; edge computing; galvanic skin response; motion sensor; accelerometer sensor; eeg, n.d. 445 URL prefix: https://www.flaticon.com/free-icon/ , IDs: hypertension_4939229; blood-pressure446 gauge_3184052; motion-sensor_2818201; student-sleeping-in-class_43739; diabetes_2750352; 447 blood-cells_3400003; edge-computing_11068838;galvanic-skin-response_11228469; motion448 sensor_17881894; accelerometer-sensor_11330476; eeg_9851782. 449 Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C., 450 Mark, R. G., Mietus, J. E., Moody, G. B., Peng, C.-K., and Stanley, H. E. Phys451 ioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for 452 complex physiologic signals. *Circulation*, 101(23):e215–e220, 2000. Circulation 453 Electronic Pages: http://circ.ahajournals.org/content/101/23/e215.full PMID:1085218; doi: 454 10.1161/01.CIR.101.23.e215. 455 Hassani, T. Federated emotion recognition with physiological signals-gsr, 2021. 456 He, J., Zhang, Q., Wang, L., and Pei, L. Weakly supervised human activity recognition from wearable 457 sensors by recurrent attention learning. *IEEE Sensors Journal*, 19(6):2287–2297, 2018.

458 He, K., Chen, X., Xie, S., Li, Y., Dollár, P., and Girshick, R. Masked autoencoders are scalable vision 459 learners, 2021. URL https://arxiv.org/abs/2111.06377.

460 Hosni, A. and Atef, M. Remote real-time heart rate monitoring with recursive motion artifact 461 removal using ppg signals from a smartphone camera. *Multimedia Tools and Applications*, 82(13):
462 20571–20588, 2023.

463 Hu, K., Ivanov, P. C., Chen, Z., Carpena, P., and Stanley, H. E. Effect of trends on detrended 464 fluctuation analysis. *Physical Review E*, 64(1):011114, 2001. 465 Huang, P.-Y., Xu, H., Li, J., Baevski, A., Auli, M., Galuba, W., Metze, F., and Feichtenhofer, C. 466 Masked autoencoders that listen, 2023. URL https://arxiv.org/abs/2207.06405. 467 Jang, E., Gu, S., and Poole, B. Categorical reparameterization with gumbel-softmax, 2017. 468 Jiang, W.-B., Zhao, L.-M., and Lu, B.-L. Large brain model for learning generic representations with 469 tremendous eeg data in bci, 2024. URL https://arxiv.org/abs/2405.18765.

470 Jolliffe, I. T. and Cadima, J. Principal component analysis: a review and recent developments. Philo471 *sophical transactions of the royal society A: Mathematical, Physical and Engineering Sciences*,
472 374(2065):20150202, 2016. 473 Kachuee, M., Kiani, M. M., Mohammadzade, H., and Shabany, M. Cuffless blood pressure estimation 474 algorithms for continuous health-care monitoring. *IEEE Transactions on Biomedical Engineering*,
475 64(4):859–869, 2016. 476 Kazemnejad, A., Gordany, P., and Sameni, R. EPHNOGRAM: A Simultaneous Electrocardiogram 477 and Phonocardiogram Database (version 1.0.0), 2021. URL https://doi.org/10.13026/ 478 tjtq-5911. 479 Kim, D., Lee, J., Cho, M., and Kwak, S. Detector-free weakly supervised group activity recognition. 480 In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 481 20083–20093, 2022. 482 Kingma, D. P. and Welling, M. Auto-encoding variational bayes, 2022. URL https://arxiv.org/ 483 abs/1312.6114. 484 Krishnan, P., Yaacob, S., Krishnan, A. P., Rizon, M., and Ang, C. K. Eeg based drowsiness detection 485 using relative band power and short-time fourier transform. *J. Robotics Netw. Artif. Life*, 7(3): 486 147–151, 2020. 487 LeMoult, J. From stress to depression: Bringing together cognitive and biological science. *Current* 488 *Directions in Psychological Science*, 29(6):592–598, 2020.

489 Li, H., Derrode, S., and Pieczynski, W. An adaptive and on-line imu-based locomotion activity 490 classification method using a triplet markov model. *Neurocomputing*, 362:94–105, 2019. 491 Li, J., Li, D., Savarese, S., and Hoi, S. BLIP-2: Bootstrapping language-image pre-training with 492 frozen image encoders and large language models. In Krause, A., Brunskill, E., Cho, K., Engelhardt, 493 B., Sabato, S., and Scarlett, J. (eds.), *Proceedings of the 40th International Conference on Machine* 494 *Learning*, volume 202 of *Proceedings of Machine Learning Research*, pp. 19730–19742. PMLR, 495 23–29 Jul 2023. URL https://proceedings.mlr.press/v202/li23q.html. 496 Liang, Y., Chen, Z., Liu, G., and Elgendi, M. A new, short-recorded photoplethysmogram dataset 497 for blood pressure monitoring in china. *Scientific data*, 5(1):1–7, 2018. doi: 10.6084/m9.figshare. 498 5459299.v5. 499 Ma, H., Zhang, Z., Li, W., and Lu, S. Unsupervised human activity representation learning with multi500 task deep clustering. *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous* 501 *Technologies*, 5(1):1–25, 2021. 502 Mathew, G., Barbosa, D., Prince, J., and Venkatraman, S. Foundation models for cardiovascular 503 disease detection via biosignals from digital stethoscopes. *npj Cardiovascular Health*, 1(1):25, 504 Oct 2024. ISSN 2948-2836. doi: 10.1038/s44325-024-00027-5. URL https://doi.org/10. 505 1038/s44325-024-00027-5.

506 McKeen, K., Oliva, L., Masood, S., Toma, A., Rubin, B., and Wang, B. Ecg-fm: An open electrocar507 diogram foundation model, 2024. URL https://arxiv.org/abs/2408.05178. 508 Mikelsons, G., Smith, M., Mehrotra, A., and Musolesi, M. Towards deep learning models for 509 psychological state prediction using smartphone data: Challenges and opportunities. In *ML4H* 510 *Workshop at 31st Conference on Neural Information Processing Systems (NIPS)*, 2017. URL 511 https://arxiv.org/abs/1711.06350. 512 Min, J., Wang, P., and Hu, J. The original EEG data for driver fatigue detection. *figshare.Dataset.*, 7 513 2017. doi: 10.6084/m9.figshare.5202739.v1. 514 Muzammil, M. Finetuning endevsols/tinyllama-2.5t-clinical model on clinical dataset., 2021. URL 515 https://huggingface.co/muzammil-eds/tinyllama-2.5T-Clinical-v2. 516 Narayanswamy, G., Liu, X., Ayush, K., Yang, Y., Xu, X., Liao, S., Garrison, J., Tailor, S., Sunshine, 517 J., Liu, Y., Althoff, T., Narayanan, S., Kohli, P., Zhan, J., Malhotra, M., Patel, S., Abdel-Ghaffar, 518 S., and McDuff, D. Scaling wearable foundation models, 2024. URL https://arxiv.org/abs/ 519 2410.13638.

520 Nedorubova, A., Kadyrova, A., and Khlyupin, A. Human activity recognition using continuous 521 wavelet transform and convolutional neural networks. *arXiv preprint arXiv:2106.12666*, 2021a. 522 Nedorubova, A., Kadyrova, A., and Khlyupin, A. Human activity recognition using continuous 523 wavelet transform and convolutional neural networks. *arXiv preprint arXiv:2106.12666*, 2021b. 524 Negi, P. C., Giri, H., Sharma, S., Sharma, N., et al. A comparative study of scalograms for human 525 activity classification. In *2024 IEEE 4th International Conference on Human-Machine Systems* 526 *(ICHMS)*, pp. 1–5. IEEE, 2024. 527 Nie, Y., Nguyen, N. H., Sinthong, P., and Kalagnanam, J. A time series is worth 64 words: Long-term 528 forecasting with transformers, 2023. URL https://arxiv.org/abs/2211.14730. 529 Oliver, A., Odena, A., Raffel, C. A., Cubuk, E. D., and Goodfellow, I. Realistic evaluation of deep 530 semi-supervised learning algorithms. In Bengio, S., Wallach, H., Larochelle, H., Grauman, K.,
531 Cesa-Bianchi, N., and Garnett, R. (eds.), *Advances in Neural Information Processing Systems*,
532 volume 31. Curran Associates, Inc., 2018. URL https://proceedings.neurips.cc/paper_ 533 files/paper/2018/file/c1fea270c48e8079d8ddf7d06d26ab52-Paper.pdf. 534 Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., and Bernstein, M. S. Generative agents: 535 Interactive simulacra of human behavior, 2023.

536 Pillai, A., Spathis, D., Kawsar, F., and Malekzadeh, M. Papagei: Open foundation models for optical 537 physiological signals, 2024. URL https://arxiv.org/abs/2410.20542.

538 Pimentel, M. A. F., Johnson, A. E. W., Charlton, P. H., Birrenkott, D., Watkinson, P. J., Tarassenko, 539 L., and Clifton, D. A. Toward a robust estimation of respiratory rate from pulse oximeters. *IEEE* 540 *Transactions on Biomedical Engineering*, 64(8):1914–1923, 2017. doi: 10.1109/TBME.2016. 541 2613124.

542 Qian, B. and Rasheed, K. Hurst exponent and financial market predictability. In *IASTED conference* 543 *on Financial Engineering and Applications*, pp. 203–209. Proceedings of the IASTED International 544 Conference Cambridge, MA, 2004. 545 Qian, H., Pan, S. J., and Miao, C. Weakly-supervised sensor-based activity segmentation and 546 recognition via learning from distributions. *Artificial Intelligence*, 292:103429, 2021. 547 Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., 548 Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from 549 natural language supervision, 2021. URL https://arxiv.org/abs/2103.00020. 550 Reiss Attila, Indlekofer Ina, S. P. PPG-DaLiA. UCI Machine Learning Repository, 2019. DOI: 551 https://doi.org/10.24432/C53890. 552 Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., and Parra, X. Human Activity Recognition Using 553 Smartphones. UCI Machine Learning Repository, 2012. DOI: https://doi.org/10.24432/C54S4K. 554 Roelofs, K. Freeze for action: neurobiological mechanisms in animal and human freezing. Philo555 *sophical Transactions of the Royal Society B: Biological Sciences*, 372(1718):20160206, 2017. 556 Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., and Van Laerhoven, K. Introducing wesad, 557 a multimodal dataset for wearable stress and affect detection. In *Proceedings of the 20th ACM* 558 *international conference on multimodal interaction*, pp. 400–408, 2018.

559 Semenoglou, A.-A., Spiliotis, E., and Assimakopoulos, V. Image-based time series forecasting: A
560 deep convolutional neural network approach. *Neural Networks*, 157:39–53, 2023. ISSN 0893-6080.

561 doi: https://doi.org/10.1016/j.neunet.2022.10.006. URL https://www.sciencedirect.com/ 562 science/article/pii/S0893608022003902.

563 Sengupta, R., Polian, I., and Hayes, J. P. Wavelet transform assisted neural networks for human 564 activity recognition. In *2022 IEEE International Symposium on Circuits and Systems (ISCAS)*, pp.

565 1254–1258. IEEE, 2022. 566 Slapnicar, G., Mlakar, N., and Luštrek, M. Blood pressure estimation from photoplethysmogram ˇ 567 using a spectro-temporal deep neural network. *Sensors*, 19(15):3420, 2019. 568 Thompson, J. M. T., Stewart, H. B., and Turner, R. Nonlinear dynamics and chaos. *Computers in* 569 *Physics*, 4(5):562–563, 1990.

570 Torrence, C. and Compo, G. P. A practical guide to wavelet analysis. *Bulletin of the American* 571 *Meteorological society*, 79(1):61–78, 1998.

572 Vaid, A., Jiang, J., Sawant, A., Lerakis, S., Argulian, E., Ahuja, Y., Lampert, J., Charney, A., 573 Greenspan, H., Narula, J., Glicksberg, B., and Nadkarni, G. N. A foundational vision transformer 574 improves diagnostic performance for electrocardiograms. *npj Digital Medicine*, 6(1):108, Jun 575 2023. ISSN 2398-6352. doi: 10.1038/s41746-023-00840-9. URL https://doi.org/10.1038/ 576 s41746-023-00840-9. 577 Van der Maaten, L. and Hinton, G. Visualizing data using t-sne. *Journal of machine learning research*, 578 9(11), 2008. 579 Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and 580 Polosukhin, I. Attention is all you need, 2023. URL https://arxiv.org/abs/1706.03762.

581 Vishnupriya, S. and Meenakshi, K. Automatic music genre classification using convolution neural 582 network. In *2018 International Conference on Computer Communication and Informatics (ICCCI)*, 583 pp. 1–4, 2018. doi: 10.1109/ICCCI.2018.8441340. URL https://ieeexplore.ieee.org/ 584 document/8441340.

585 Wang, J., Zhao, S., Luo, Z., Zhou, Y., Jiang, H., Li, S., Li, T., and Pan, G. Cbramod: A criss-cross 586 brain foundation model for eeg decoding, 2025. URL https://arxiv.org/abs/2412.07236.

587 Wimmer, C. and Rekabsaz, N. Leveraging vision-language models for granular market change 588 prediction, 2023. URL https://arxiv.org/abs/2301.10166. 589 Wolf, A., Swift, J. B., Swinney, H. L., and Vastano, J. A. Determining lyapunov exponents from a 590 time series. *Physica D: nonlinear phenomena*, 16(3):285–317, 1985. 591 Woo, G., Liu, C., Kumar, A., Xiong, C., Savarese, S., and Sahoo, D. Unified training of universal 592 time series forecasting transformers, 2024. URL https://arxiv.org/abs/2402.02592.

593 Wu, Y., Chen, K., Zhang, T., Hui, Y., Berg-Kirkpatrick, T., and Dubnov, S. Large-scale contrastive 594 language-audio pretraining with feature fusion and keyword-to-caption augmentation. In *ICASSP* 595 *2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 596 pp. 1–5. IEEE, 2023. 597 Yan, Y., Huang, Y.-C., Zhao, J., Liu, Y.-S., Ma, L., Yang, J., Yan, X.-D., Xiong, J., and Wang, L. 598 Topological nonlinear analysis of dynamical systems in wearable sensor-based human physical 599 activity inference. *IEEE Transactions on Human-Machine Systems*, 53(4):792–801, 2023a. doi: 600 10.1109/THMS.2023.3275774. 601 Yan, Y., Huang, Y.-C., Zhao, J., Liu, Y.-S., Ma, L., Yang, J., Yan, X.-D., Xiong, J., and Wang, L. 602 Topological nonlinear analysis of dynamical systems in wearable sensor-based human physical 603 activity inference. *IEEE Transactions on Human-Machine Systems*, 53(4):792–801, 2023b. doi:
604 10.1109/THMS.2023.3275774.

605 Yang, C., Westover, M., and Sun, J. Biot: Biosignal transformer for cross-data learning in the 606 wild. In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S. (eds.), 607 *Advances in Neural Information Processing Systems*, volume 36, pp. 78240–78260. Curran As608 sociates, Inc., 2023. URL https://proceedings.neurips.cc/paper_files/paper/2023/ 609 file/f6b30f3e2dd9cb53bbf2024402d02295-Paper-Conference.pdf. 610 Yuan, H., Chan, S., Creagh, A. P., Tong, C., Acquah, A., Clifton, D. A., and Doherty, A. Self611 supervised learning for human activity recognition using 700,000 person-days of wearable data.

612 *npj Digital Medicine*, 7(1), April 2024. ISSN 2398-6352. doi: 10.1038/s41746-024-01062-3.

613 URL http://dx.doi.org/10.1038/s41746-024-01062-3.

614 Zhang, H., Cisse, M., Dauphin, Y. N., and Lopez-Paz, D. mixup: Beyond empirical risk minimization, 615 2017. 616 Zhang, W., Yang, L., Geng, S., and Hong, S. Self-supervised time series representation learning via 617 cross reconstruction transformer. *IEEE Transactions on Neural Networks and Learning Systems*, 618 2023. 619 Zhang, X., Zhao, Z., Tsiligkaridis, T., and Zitnik, M. Self-supervised contrastive pre-training for time 620 series via time-frequency consistency. *Advances in Neural Information Processing Systems*, 35: 621 3988–4003, 2022. 622 Zhang, X., Chowdhury, R. R., Gupta, R. K., and Shang, J. Large language models for time series: A
623 survey. *arXiv preprint arXiv:2402.01801*, 2024a.

624 Zhang, X., Teng, D., Chowdhury, R. R., Li, S., Hong, D., Gupta, R. K., and Shang, J. Unimts: Unified 625 pre-training for motion time series, 2024b. URL https://arxiv.org/abs/2410.19818.

## 626 **A Datasets**

627 Few openly accessible multi-channel or multi-device datasets for physiological signals exist, limiting 628 advancements in this field. To address this gap, we curated a dataset comprising approximately 629 385 hours of recordings. Using the augmentation algorithm described below, we expanded this 630 dataset to 4294 hours. The distribution of the pretraining dataset, as shown in Figure 6, reflects 631 the inherent diversity of the original recordings, ensuring balanced representation across channels 632 and devices. This curated and augmented dataset provides a critical resource for developing robust models, facilitating progress in multi-channel physiological signal research.

Table 4: **Downstream evaluation data that are**

unseen during pretraining.

Downstream Dataset Sensor # Channels Tasks **#Samp. (#Subj.)**

WESAD

(Schmidt et al., 2018)IMU, PPG,

ECG, GSR

10 Stress

Detection11050(15)

UCI-HAR

(Reyes-Ortiz et al., 2012)IMU 6 HAR 10299(30)

DriverFatigue

(Min et al., 2017)EEG 4Fatigue

Detection2400(12)

Activity Recognition Total - - - **23749(57)**

Epilepsy

(Andrzejak et al., 2023)EEG 1State

Recognize11500(500)

GAMEEMO

(Alakus et al., 2020)EEG 4ValenceArousal5600(28)

EEG Main Tasks Total - - - **17100(528)**

ECG-Abnormal

(Bousseljot et al., 2009)ECG 1Abnormal

Detection11640(249)

PPG-BP

(Liang et al., 2018)PPG 1Risk of

Diseases657(219)

PhysioNet EMG

(Goldberger et al., 2000)EMG 1Muscular

Diseases163(3)

Risk Evaluation Total - - - **12460(471)**

Noninvasive-BP

(Esmaili et al., 2017)PPG 3BP

Estimate125(26)

PPG-Hgb

(Esmaili et al., 2017)PPG 2Hgb

Estimate68(68)

Fetal-fPCG

(Bhaskaran et al., 2022)PCG 1Fetal HR

Estimate47(47)

Vital Signs Total - - - **240(141)** Total All - - - **53549(1197)**

Pretrain Dataset Sensors **#Samp (hours).**

Cuff-Less-BP

(Kachuee et al., 2016)ECG, PPG 42934(72)

PPG-Dalia

(Reiss Attila, 2019)

ECG, PPG

IMU, GSR42606(71)

Auditory-EEG

(Alzahab et al., 2022)EEG 13601(23)

PhyAAt

(Bajaj et al., 2020)EEG 19550(33)

MAUS

(Beh et al., 2021)

ECG, PPG

GSR13068(22)

Mendeley-YAAD

(Dar et al., 2022)ECG, GSR 2964(5)

Brain-Cognitive

(Fekri Azgomi et al., 2023)EEG 51201(85)

EPHNOGRAM

(Kazemnejad et al., 2021)ECG, PCG 36611(61)

BIDMC

(Pimentel et al., 2017)ECG, PPG 8427(14)

Num Segments (# Segm.) - 230,962(385) # Segm. w/ Augment - 2,576,418(4,294) Num Sensor Signals (# Sign.) - **802,019(1,337)** # Sign. w/ Augment - **8,965,538(14,943)**

Figure 6: **Distribution of sensor signals used for pretraining.** *Left:* Distribution by sensor modality. Right: Distribution by type of physiological information.

633 634 Table 4 overviews used dataset in our experiement along with the modality and task type. We will 635 gives further details for each dataset below:
636 **WESAD** (Schmidt et al., 2018) is a publicly available multimodal dataset used for wearable stress 637 and affect detection, formulated as a classification task with labels: neutral, stress, and amusement.

638 The dataset includes physiological and motion data collected from 15 subjects during a lab study, 639 using a chest-worn RespiBAN device and a wrist-worn Empatica E4 device. From the chest device, 640 we use electrocardiogram (ECG), galvanic skin response (GSR), and triaxial acceleration (ACC-X, 641 ACC-Y, ACC-Z), all sampled at 700 Hz. From the wrist device, we use photoplethysmogram (PPG), 642 galvanic skin response (GSR, 4 Hz), and triaxial acceleration (ACC-X, ACC-Y, ACC-Z, 32 Hz). 643 The selected channels span multiple physiological and motion modalities from both chest and wrist 644 sensors. Each data segment is labeled with one of the three affective states, serving as the target 645 output for classification tasks. 646 **UCI-HAR** (Reyes-Ortiz et al., 2012) dataset is publicly available and is used for classifying human 647 activities based on sensor data. It comprises data from 30 volunteers, aged 19 to 48, each performing 648 six activities: walking, walking upstairs, walking downstairs, sitting, standing, and laying. During 649 these activities, participants carried a waist-mounted smartphone equipped with embedded accelerom650 eter and gyroscope sensors. The input channels consist of triaxial linear acceleration and triaxial 651 angular velocity, totaling six channels. Each data segment is labeled with one of the six activities, 652 serving as the target output for classification tasks. The sensors recorded data at a constant rate of 50 653 Hz. 654 **Driver Fatigue EEG Dataset** (Min et al., 2017) is a publicly available dataset used for detecting 655 driver fatigue based on electroencephalogram (EEG) signals. EEG data were collected using a 656 40-channel Neuroscan amplifier. The recordings include EEG data corresponding to two states: alert 657 and fatigued. Each data segment is labeled with one of these states, serving as the target output for 658 classification tasks. 659 **Epileptic Seizure Recognition** (Andrzejak et al., 2023) dataset is publicly available and is used 660 for classifying neurological and physiological states based on EEG signals. It comprises data from 661 500 subjects, each recorded for 23.6 seconds using a single EEG channel at a sampling rate of 178 662 Hz. Each sample is labeled with one of five brain states, allowing for the construction of multiple 663 binary classification tasks that target different aspects of neurological assessment. Specifically, we 664 formulated five tasks: 665 - *Eye Relaxation*: Detects eye fatigue by distinguishing between relaxed and alert states based 666 on EEG changes related to eye closure. 667 - *Health Area*: Classifies brain regions as healthy or affected by neurological abnormalities. 668 - *Tumor Area*: Detects EEG patterns indicative of tumor presence in specific brain regions. 669 - *Seizure*: Identifies seizure activity from non-seizure states.

670 - *Eyes Open vs. Closed*: Differentiates EEG signals associated with visual input states.

671 **GAMEEMO** (Alakus et al., 2020) is a publicly available dataset used for emotion recognition based 672 on EEG signals. It comprises data from 28 subjects, each playing four emotion-inducing computer 673 games (boring, calm, horror, and funny) for five minutes per game, totaling 20 minutes of EEG data 674 per subject. EEG signals were recorded using the EMOTIV EPOC+ headset, which includes 14 675 channels (AF3, AF4, F3, F4, F7, F8, FC5, FC6, O1, O2, P7, P8, T7, and T8) positioned according to 676 the 10–20 system. The signals were sampled at 128 Hz. After each gameplay session, subjects rated 677 their emotional response using the Self-Assessment Manikin (SAM) form, providing continuous 678 scores for arousal and valence. These scores were quantized into binary values using subject-specific 679 median thresholds: arousal and valence ratings above the median were labeled as high, and those 680 below or equal to the median as low. Combining the binarized arousal and valence ratings yields four 681 discrete emotional classes: low arousal and low valence, low arousal and high valence, high arousal 682 and low valence, and high arousal and high valence. Each data segment is labeled with one of these 683 four classes, serving as the target output for four-class emotion classification tasks. 684 **ECG Heartbeat Categorization** (Bousseljot et al., 2009) is a publicly available dataset used for 685 classifying heartbeat signals based on electrocardiogram (ECG) recordings. It comprises two col686 lections of heartbeat signals derived from PhysioNet's MIT-BIH Arrhythmia Dataset and the PTB 687 Diagnostic ECG Database. The first collection includes 109,446 samples categorized into five classes: 688 normal (N), supraventricular ectopic (S), ventricular ectopic (V), fusion (F), and unknown (Q), with 689 ECG signals sampled at 125 Hz. The second collection consists of 14,552 samples categorized into 690 two classes: normal and abnormal, also sampled at 125 Hz. For our analysis, we restructured the 691 dataset into a binary classification framework by consolidating the original categories into two classes:
692 normal and abnormal heartbeats. 693 **PPG-China** (Liang et al., 2018) is a publicly available dataset used for classifying cardiovascular and 694 metabolic conditions based on photoplethysmography (PPG) signals. It comprises 657 data records 695 from 219 subjects, aged 20 to 89 years, including individuals with conditions such as hypertension 696 and diabetes. PPG signals were recorded using a single channel at a sampling rate of 125 Hz. 697 Each subject's data includes PPG waveforms and corresponding clinical information, facilitating the 698 construction of multiple classification tasks focused on cardiovascular and systemic health monitoring. 699 Specifically, we formulated four tasks: 700 - *PPG-HTN*: Identifies stages of hypotension severity by classifying PPG signals into four 701 levels. 702 - *PPG-DM*: Detects diabetes by distinguishing between diabetic and non-diabetic individuals. 703 - *PPG-CVA*: Identifies the presence or absence of cerebrovascular accidents (strokes) based 704 on PPG patterns. 705 - *PPG-CVD*: Assesses cardiovascular disease by classifying PPG signals into three cardiovas706 cular health categories. 707 **PhysioNetEMG** (Goldberger et al., 2000) is a publicly available dataset used for classifying neuro708 muscular conditions based on electromyography (EMG) signals. It comprises single-channel EMG 709 recordings from the tibialis anterior muscle of three subjects: one healthy, one with neuropathy, and 710 one with myopathy. The EMG signals were recorded at a sampling rate of 4,000 Hz. Each recording 711 was segmented into time series samples using a fixed-length window of 6 second. Each segment 712 is labeled according to the subject's condition—healthy, neuropathy, or myopathy—serving as the 713 target output for classification tasks. 714 **Non-invasive Blood Pressure Estimation** (Esmaili et al., 2017) is a publicly available dataset 715 used for cuff-less blood pressure (BP) estimation. It comprises data from 26 subjects, each with 716 recorded electrocardiogram (ECG) and photoplethysmogram (PPG) signals, sampled at 1,000 Hz. 717 Reference BP measurements were taken during signal acquisition. Each subject's data also includes 718 demographic information such as age, weight, and height. The dataset is structured to facilitate 719 regression tasks aimed at predicting systolic and diastolic BP values. 720 **PPG-HGB** (Abuzairi et al., 2024) is a publicly available dataset used for non-invasive hemoglobin 721 (Hb) measurement based on photoplethysmography (PPG) signals. It comprises data from 68 subjects, 722 aged 18 to 65 years, with a gender distribution of 56% female and 44% male. PPG signals were 723 recorded using the MAX30102 sensor, which emits red and infrared light. The sensor's analog-to724 digital converter (ADC) output data rate can be programmed from 50 samples per second (sps) to 725 3200 sps. Each subject contributed 12 sets of PPG signals, totaling 816 data records. We formulate 726 regression tasks aimed at predicting Hb concertration levels. 727 **Fetal-fPCG** (Bhaskaran et al., 2022) is a publicly available dataset designed for estimating fetal heart 728 rate (FHR) using fetal phonocardiography (fPCG) signals. It includes recordings from 60 pregnant 729 women, aged 18 to 37 years, with gestational ages between 31 and 40 weeks. The recordings were 730 collected at St. John's Hospital in Bangalore using an electronic stethoscope (SS30LA) connected to 731 a Biopac MP36 data acquisition system. The stethoscope was placed on the lower abdomen of each 732 subject to capture the fPCG signal, which was sampled at 2,000 Hz. The dataset supports regression 733 tasks, where the goal is to predict continuous FHR values directly from the fPCG waveforms.

## 734 **B Implementation Detail** 735 **B.1 Data Preprocess.**

736 For the data preparation, we set the uniform sampling rate and interval length to 65 HZ and 6 seconds 737 respectively. In our case, 65 Hz covers most of the frequency bands of interest such as heart activity, 738 physical motions, and neuron activity up to the beginning of Gamma power (above 30 Hz). And 739 a great amount of samples are less than 6 seconds such as (Reyes-Ortiz et al., 2012; Liang et al.,
740 2018; Bousseljot et al., 2009). We conduct basic pre-processing for each signal with identical setting: 741 (1) de-trended by subtract the result of a linear least-squares fit to series data from the raw time 742 series, and (2) Gaussian smoothed with standard deviation of 1.3 (0.02 seconds), ensuring a highly 743 consistent dataset for training.

744 Since the Transformer's computational requirements scale quadratically with input length, to release 745 the full potential of our self-supervised algorithm, we segment our multivariate time series into 746 intervals with a uniform length and pad shorter samples with zeros. This approach not only enables 747 parallel processing of samples in large minibatches but also addresses variation in the length of 748 individual samples. 749 For the downstream task, we split the data into train and test sets for linear probing evaluation with 750 portion of 80% and 20% correspondingly. The split is stratified on the anonymized subject ID if this 751 information is provided by the dataset.

## 752 **B.2 Data Augmentation.**

753 Since there are very few publicly available datasets containing multiple devices or modalities, we aim 754 to expand our curated training set to fully leverage the potential of self-supervised learning. Inspired 755 by data augmentation techniques in computer vision and natural language processing (Zhang et al., 756 2017; Carmona et al., 2021), we adopt a heuristic approach to augment the dataset. Specifically, 757 we augment each sub-dataset by a factor of 10. For each dataset, we sample two time series, 758 randomly extract a segment from one, and substitute it with a transformed counterpart, as outlined 759 in the pseudocode in Algorithm 1. As a result, our training set is expanded to 2,586,404 segments, corresponding to 4,294 hours of data. Algorithm 1 Time Series Mixup Augmentation Input: Time series dataset X , number of augmentations n Output: Augmented Dataset X˜
1: for i = 1 to n do 2: Sample two time series x
(1), x
(2) ∼ X
3: Sample a chunk size λ ∼ U(0, l)
4: Sample start indices s1, s2 ∼ U(0, l − λ)
5: Swap chunk from x
(2) into x
(1):
6: Append x
(1) into X˜
7: **end for**
8: **return** X˜
760

## 761 **B.3 Pretraining Framework.**

762 Normwear is derived from the Masked Autoencoder (MAE) (He et al., 2021). The detailed hyper763 parameter choice is descibe in 6. We use a Conv2D layer with a kernel size of (9, 5) and a stride 764 of (9, 5), ensuring no overlapping patches. This layer takes input with 3 channels and projects 765 it to 768 channels, matching the hidden size of our encoders. In Normwear, we apply structured 766 masking independently to each variate along both the frequency and time axes, with respective 767 masking ratios of 0.6 and 0.5. This results in an expected overall masking ratio of 0.8 for each 768 variate. Only the unmasked tokens are passed to the encoder, reducing computational complexity.

769 To enhance representation learning, we introduce six additional transformer blocks as fusion layers, 770 interleaved with the original 12 encoder blocks, creating a total of 18 blocks. Each transformer block 771 has a hidden dimension of 768 and uses LayerNorm as in the original MAE. The latent embeddings 772 obtained from the encoder are projected from 768 to 512 dimensions. Learnable masked tokens are 773 reinserted at their original positions, and positional embeddings are added to guide the decoder in 774 reconstructing the input series. The lightweight decoder consists of two transformer blocks with 775 a hidden dimension of 512, followed by two Conv1D layers. The first Conv1D layer maps from 776 the flattened multivariate signal embedding to an intermediate dimension, and the second Conv1D 777 layer maps from this intermediate dimension back to the original multivariate signal space. A GELU 778 activation function is used between these layers, with BatchNorm applied to the input. The decoder 779 reconstructs the original input series, and the model is trained using Mean Squared Error (MSE) loss 780 on all data points. Our models are pre-trained for 45,000 steps with a batch size of 256, using the AdamW optimizer with a learning rate of 10−4 781 . We did not perform on-the-fly data augmentation, Figure 7: **Visualization of original time series (left), CWT transformation image with structured** masking (middle), and reconstructed time series (right).

782 as suggested in the MAE framework, due to the high masking ratio. (An end-to-end example of the 783 input and output of this pretraining pipeline is illustrated in Fig. 7) 784 All the models are pretrained on 4 NVIDIA RTX 3090 graphical computing unit (GPU), with 24GB 785 of GPU memory on each card.

## 786 **B.4 Msitf.**

787 For pretraining the representation alignment module, we have the training hyper-parameters in Table 7.

| Hyper-parameter                      | Value        |
|--------------------------------------|--------------|
| # cross-patches Transformer Encoder  | 12           |
| # cross-channels Transformer Encoder | 6            |
| # Transformer Decoder                | 2            |
| # Attention Heads                    | 12           |
| Encoder Latent Size                  | 768          |
| Decoder Latent Size                  | 512          |
| Feedforward Latent Size              | 3072         |
| Normalization                        | LayerNorm    |
| Patch size (time axis)               | 9            |
| Patch size (scale axis)              | 5            |
| Optimizer                            | AdamW        |
| Loss Scalar                          | NativeScaler |
| Base Learning Rate (blr)             | 1e-3         |
| Epochs                               | 140          |
| Batch size                           | 192          |

| Hyper-parameter    | Value   |
|--------------------|---------|
| Learning rate (lr) | 1e-3    |
| Epochs             | 40      |
| Batch size         | 32      |
| L2 regularization  | 5e-6    |
| lr decay rate      | 0.997   |
| λ                  | 0.5     |
| τ                  | 0.5     |