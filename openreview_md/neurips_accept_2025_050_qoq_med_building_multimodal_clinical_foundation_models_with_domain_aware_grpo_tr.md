# Qoq-Med: Building Multimodal Clinical Foundation Models With Domain-Aware Grpo Training

Wei Dai, Peilin Chen, Chanakya Ekbote, Paul Pu Liang MIT Media Lab and MIT EECS
{dvdai, peili, cekbote, ppliang}@mit.edu

## Abstract

Clinical decision-making routinely demands reasoning over heterogeneous data, yet existing multimodal language models (MLLMs) remain largely vision-centric and fail to generalize across clinical specialties. To bridge this gap, we introduce QoQ-Med-7B/32B, the first open generalist clinical foundation model that jointly reasons across medical images, time-series signals, and text reports. QoQ-Med is trained with Domain-aware Relative Policy Optimization (DRPO), a novel reinforcement-learning objective that hierarchically scales normalized rewards according to domain rarity and modality difficulty, mitigating performance imbalance caused by skewed clinical data distributions. Trained on 2.61 million instruction tuning pairs spanning 9 clinical domains, we show that DRPO training boosts diagnostic performance by 43% in macro-F1 on average across all visual domains as compared to other critic-free training methods like GRPO. Furthermore, with QoQ-Med trained on intensive segmentation data, it is able to highlight salient regions related to the diagnosis, with an IoU 10x higher than open models while reaching the performance of OpenAI o4-mini. To foster reproducibility and downstream research, we release (i) the full model weights, (ii) the modular training pipeline, and (iii) all intermediate reasoning traces at this link.

## 1 Introduction

Clinical diagnosis has evolved significantly over the past decade, with numerous computational models developed to assist clinicians in organizing patient records [87, 5], formulating diagnoses [19, 42], interpreting clinical images [26, 83], and other clinical tasks [12]. These advancements have substantially improved healthcare efficiency and accuracy across multiple specialties [21, 77]. Recently, the emergence of powerful generalist reasoning models such as OpenAI o3 [61] and Deepseek R1 [29] have inspired efforts to create specialized clinical reasoning systems [46, 91, 65] capable of answering complex clinical questions and generating comprehensive clinical reports [20, 90]. Reasoning allows models to think explicitly in a more logical and systematic way with evidence from the inputs and their own knowledge [34], all of which are essential for clinical diagnosis [55, 49]. However, building effective models to support clinical diagnosis presents several significant challenges. First, clinical data spans multiple modalities across 1D (ECG, EEG), 2D (Chest X-ray, dermoscopy, mammography), and 3D (CT Scans, MRI). Models like BiomedGPT [90] and Med- Flamingo [56] have integrated 2D and 3D data within one vision encoder, but no existing model has been able to integrate both 1D sensor data with 2D/3D images. The heterogeneity across specialties and modalities [28, 17] often leads to settings where modalities compete rather than synergize, leading to suboptimal performance [1, 36, 52, 89]. This necessitates careful retraining or fine-tuning strategies to balance heterogeneous distributions while enriching these models with clinical knowledge. Secondly, conventional training methodologies typically constrain models to generate single, definitive answers without revealing their underlying analytical process [80, 76, 45]. This "black box" approach significantly impedes the practical adoption of AI systems in clinical settings, as healthcare professionals might hesitate to trust diagnostic suggestions without understanding the reasoning that produced them [72]. Transparency in the decision-making process is not merely a preference but a necessary component for responsible clinical implementation, regulatory compliance, and effective human-AI collaboration in healthcare environments [6, 66, 11]. In this work, we introduce QoQ-Med: a generalist clinical multimodal foundation model with precise reasoning capabilities spanning clinical images, time series data, and textual records across 9 clinical domains. Our work makes two primary contributions:
1. Firstly, to tackle the challenges associated with balancing heterogeneous data for balanced and efficient training across 1D to 3D data, we propose **Domain-aware Group Relative Policy** Optimization (DRPO). DRPO employs hierarchical scaling based on the domain of the input data, which encourages the model's learning on scarce and hard domains, allowing balanced learning across difficulty levels. Our empirical evaluation demonstrates that DRPO consistently outperforms established RL approaches in diverse multi-domain settings, with up to 43% improvement in average F1 score across 8 clinical vision modalities.

2. To tackle the second challenge of expert interpretability, we design and release one of the first multimodal clinical reasoning models, namely **QoQ-Med-7B/32B** (Qwen Omni-Reasoning on Medical Questions), that integrates visual, time series, and textual data for comprehensive analysis of clinical records, facilitating more holistic diagnostic reasoning. QoQ-Med is trained to highlight salient regions in the visual input data, advancing the interpretability while allowing the clinician to check the model's diagnosis with ease. To the best of our knowledge, QoQ-Med is currently the largest open-source multimodal reasoning model for clinical diagnosis, and the only MLLM that integrates time series data (ECG) with traditional clinical vision modalities.

Finally, we publicly release our model, training pipeline, and reasoning traces generated by the model across 2.61 million question-answer pairs at this link. This marks one of the largest resources for transparent and reproducible multimodal reasoning in the clinical domain.

## 2 Related Work 2.1 Multimodal Large Language Models (Mllms) For Clinical Diagnosis

Recent work has adapted vision–language interfaces to the medical domain, yielding models such as LLaVa-Med [48], RadLM [90], and Med-Flamingo [56]. These models couple frozen LLM backbones with image encoders and are trained on radiology or pathology visual-question-answering and reportgeneration benchmarks [24, 39, 92, 88, 86]. Although these systems demonstrate impressive zeroshot understanding, their training corpora are dominated by single-institution chest X-rays, retinal photographs, and pathology slides, resulting in limited generalization to demographic diversity and poor robustness to real-world distribution [44, 63, 51]. GEM [47] is the only MLLM incorporating ECG data, but the training focus is purely ECG, which does not provide a comprehensive diagnosis aggregating multiple sources. Our work addresses these gaps by assembling a richer corpus spanning imaging, time-series, and text, and by designing an architecture that natively models medical timeseries alongside traditional modalities.

## 2.2 Llm Reasoning With Reinforcement Learning

The introduction of instruction tuning precipitated a rapid shift from supervised fine-tuning to reinforcement learning pipelines. Proximal Policy Optimization (PPO) [74] as popularized by InstructGPT, trains LLMs against a reward model under a KL penalty to a frozen reference, with an auxiliary critic estimating advantages [62]. While effective, PPO's critic incurs substantial memory and computation costs and can destabilise multi-task optimization [73]. To reduce overhead, critic-free objectives such as Direct Preference Optimization (DPO) [70] and Group Relative Policy Optimization (GRPO) [75] have emerged, matching PPO's alignment quality with a simple classification loss. GRPO, in particular, has been widely used in the training of recent SoTA models, such as DeepSeek R1 [29] and Qwen-3 [79]. However, removing the critic also eliminates per-sample re-weighting, causing it to overfit on easy, abundant samples [33]. Classic deep-RL work explored adaptive rescaling through task-wise normalization in IMPALA [27] and the PopArt [31]. However, these techniques have not been adapted to LLMs or extended to capture fine-grained intra-domain differences. We reinstate that flexibility by learning both inter-domain and intra-domain scaling factors within a critic-free RLHF pipeline, combining the efficiency advantages of GRPO with the adaptive weighting capabilities offered by critic-based methods.

M ult imodal Tr aining D at aset QoQ-M ed Out put s w/ Reason ECG
X-ray Dermo. Fundus Patho.

Reasoning I nsight s Salient Region A nnot at ions CT Scan MRI Mammo Ultrasound Q: Above is a chest CT scan slice of a patient. What type of Pulmonary Embolism (PE) is present in this CT scan?

Answer with one of the following: No PE, Chronic PE, Acute PE A: No PE
(a)
Clinical D iagnosis Clinical QA Pairs EHR
Easy D at a, Smaller |A dvant age| RL Training DRPO
Clustering Hierarchical Scaling GRPO Norm. 

Domain 1 Domain 1 H ar d D at a, Lar ger |A dvant age| Domain 2 Domain 2 KL
(b)
D RPO M ixed Tr aining

## 3 Method

In this section, we first define our problem as a multimodal diagnosis question answering task, before describing how we integrated time series alongside vision inputs into a single unified model. Finally, we demonstrate in detail how we address the domain heterogeneity problem with the Domain-aware Relative Policy Optimization (DRPO) algorithm and design of appropriate reward functions.

## 3.1 Problem Definition

Each clinical sample is xi =x
(v)
i, x
(t)
i, x
(s)
i, gi, where x
(v)
i ∈ R
Pi×dv is a patchified image, x
(t)
i ∈ R
ki×Tiis multichannel time-series data, x
(s)
iis text input, and gi ∈ {1*, . . . , C*} denotes the clinical domain (e.g., CT scans, ECG, Chest X-ray). Vision and time-series inputs are optional, which requires the model to handle missing modalities. The learning objectives are to predict: (i) an unsupervisedly learned reasoning trace, (ii) bounding boxes bi = {bi,j}
Ki j=1 with bi,j ∈ R
4in
(*x, y, w, h*) format highlighting salient image regions, and (iii) a concise diagnosis yˆi.

| [90, 56] are trained on some ECG images, but none of them are trained on raw ECG time series input. Model Size Training BBox 1D 2D 3D ECG CXR Mammo. Derm. Fundus Patho. US MRI CT LLaVa-Med [48] 7B-13B SFT ✓ ✗ ✓ ✗ ✓ ✗ ✓ ✗ ✓ ✓ Med-Flamingo [56] 8.3B SFT ✗ o* ✓ ✗ ✓ ✗ ✓ ✗ ✓ ✓ RadFM [85] 14B SFT ✓ ✗ ✓ ✓ ✓ ✗ ✗ ✓ ✓ ✓ BiomedGPT [90] 33M-182M SFT ✓ o* ✓ ✗ ✓ ✓ ✓ ✓ ✓ ✓ Med-R1 [46] 2B GRPO ✗ ✗ ✓ ✗ ✓ ✓ ✓ ✓ ✓ ✓ QoQ-Med (Ours) 7B-32B DRPO ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓   |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## 3.2 Model

We design the model with an aim that it can take in data across as many domains as possible, so that it can provide comprehensive diagnosis while correlate and co-train across the most diverse range of clinical domains, with inputs ranging from 1D to 3D. Model Design. As shown in Figure 1, we initialize QoQ-Med from a large pretrained vision–language model comprising an image encoder, a linear projection that maps each visual patch embedding into the backbone LLM's token space, and the LLM. To ingest temporal data, we prepend a pretrained time-series encoder, namely ECG-JEPA [43], whose outputs are passed through a newly initialized linear projection of matching dimension. At inference, the projected image patches, time-series tokens, and tokenized text are interleaved in their original temporal order and fed to the LLM. The LLM autoregressively generates a free-text chain of thought, bounding-box tokens that localize the evidence identified in that reasoning, and outputs a short diagnosis. This design supports heterogeneous modality combinations, allowing the model to skip missing channels while preserving positional consistency across the multimodal sequence. Training Process. Training proceeds in two stages. **Stage 1: modality alignment.** Since we initialize the projection layer from scratch, we first train and align the ECG encoder, the projection layer, and the LLM. To encourage high-quality reasoning outputs from the beginning, we use the same DRPO training as in Stage 2. **Stage 2: multimodal fine-tuning with DRPO.** We train on the full multimodal corpus with DRPO, as described in Sec. 3.3, which balances training across different samples in various domains and difficulty. In this stage, we aim to simultaneously improve the diagnostic accuracy and reasoning quality, with rewards described in Sec. 3.4.

Training Data. We train the unified vision and time-series model across 33 datasets using the CLIMB dataset [22]. The dataset contains 2.61 million samples across 1D (ECG), 2D (Chest X-ray, Mammography, Dermoscopy, histopathology, Fundus), and 3D (Ultrasound, MRI, CT Scan) data. The exact composition of the data and the training hyperparameters are included in App. C and D. Comparison with current public clinical MLLMs. Table 1 demonstrates that our model is currently the largest open clinical MLLM in the field. It is also the only model that can both take in time series data and output its thinking process, along with the bounding box annotation highlighting the salient region made during the thinking process.

## 3.3 Domain-Aware Relative Policy Optimization (Drpo)

Group Relative Policy Optimization (GRPO) is a reinforcement learning method that gained prominence following the success of DeepSeek-R1. Unlike Proximal Policy Optimization (PPO), which relies on a separate value network to estimate advantages, GRPO directly computes the advantage Aˆ(*q,i,t*)(Eq. 1) for each response within a group of rollouts G(q,t) at a given training iteration. A
rollout refers to a single sampled trajectory or response generated by the policy in reaction to a prompt. The advantage quantifies how much better a particular rollout is compared to others generated for the same prompt, enabling the policy to prioritize relatively high-quality responses without requiring an explicit estimate of expected return.

Each group of rollouts G(q,t) consists of multiple responses sampled for the same prompt q. Let r(*q,i,t*) denote the scalar reward assigned to the i-th response o(*q,i,t*) at time step t, where each response is a sequence of tokens o(q,i,t):= o(*q,i,t*):1, o(*q,i,t*):2*, . . . , o*(q,i,t):no(*q,i,t*)
, and no(*q,i,t*)
denotes the length of the token sequence. The set of rewards for the group is defined as RG(q,t) = {r(q,1,t), r(q,2,t)*, . . . , r*(q,|Gq|,t)}, where |Gq| is the number of responses in the group.

GRPO normalizes these rewards to have zero mean and unit variance, producing the normalized advantage:

$$\hat{A}_{(q,i,t)}^{\mathrm{GRPO}}=\frac{r_{(q,i,t)}-\hat{\mu}_{G_{(q,t)}}}{\hat{\sigma}_{G_{(q,t)}}+\varepsilon},$$
, (1)
where µˆG(q,t)and σˆG(q,t)denote the empirical mean and standard deviation of the group rewards, respectively, and ε is a small constant added for numerical stability. These advantage estimates are incorporated into the GRPO clipped surrogate objective, which also includes a per-token KL divergence penalty:

A˜(q,i,t):k(θ) = min φ(q,i,t):k(θ) · AˆGRPO (q,i,t) , clip φ(q,i,t):k(θ), 1 − ε, 1 + ε· AˆGRPO (q,i,t) , φ(q,i,t):k(θ) =  πθ(o(q,i,t):k | q, o(q,i,t):<k) πθold (o(q,i,t):k | q, o(q,i,t):<k) , 1 no(q,i,t) no X (q,i,t) JGRPO(θ) = Eq∼D, {o(q,i,t)}∼πθold   1 |G(q,t)| |G X (q,t)| k=1 A˜(q,i,t):k(θ) − β DKL (πθ ∥ πref)   . i=1
$$(1)$$
Here, o(*q,i,t*):<k refers to the token subsequence from position 1 to k − 1, and D denotes the dataset distribution. The term φ(*q,i,t*):k(θ) represents the importance sampling ratio between the current policy πθ and the old policy πθold at token position k; AˆGRPO
(q,i,t)is the normalized advantage estimate for the i-th response in group G(q,t); ε is a small constant used for numerical stability and clipping; β is a scalar hyperparameter that controls the strength of the KL divergence regularization; and DKL(πθ∥πref) denotes the Kullback–Leibler divergence between the learned policy and a reference policy. GRPO demonstrates strong empirical performance when the input data is relatively homogeneous. However, in settings with high data heterogeneity, domains with abundant samples tend to dominate the optimization process, while under-represented domains contribute minimally. This imbalance can bias the model and degrade performance on rare but clinically important modalities, while spending too much compute on easy problems on abundant domains. Domain-aware Relative Policy Optimization (DRPO). While GRPO normalizes reward signals across rollouts that respond to the *same* prompt—thereby reducing variance within a group and ensuring fairer comparison among responses—it does not address imbalance *across* domains. As a result, domains that appear more frequently in the training data continue to have a disproportionate impact on the learning process. DRPO builds on GRPO by introducing a hierarchical scaling mechanism that explicitly balances contributions from different domains. This correction for interdomain imbalance preserves GRPO's simplicity and value-free formulation while promoting more equitable learning across heterogeneous data distributions. Hierarchical Cluster-Based Scaling. The core innovation of DRPO lies in a hierarchical scaling strategy that adaptively balances learning signals based on both domain frequency and task difficulty. This mechanism operates at two levels: across domains, to mitigate the dominance of overrepresented domains, and within domains, to adjust for variations in response quality or reward magnitude.

Concretely, we first cluster question-level reward sets within each domain, treating each set of individual rewards as a feature vector. We then apply a two-stage reward scaling procedure—first at the cluster level, then at the individual reward level—thereby emphasizing learning from rare and challenging questions.

Stage-1: Intra-Domain Clustering. At each iteration step t, we begin by sampling an independent batch of questions. These questions are then clustered into different domains. Let g denote a domain, and let N(g,t)represent the number of questions in domain g at iteration t. Within each domain at iteration t, we first compute the set of rewards for each question. These rewards, collected across multiple rollouts, are concatenated into a feature vector per question. Specifically, for each domain g, we construct a set of reward vectors Hg = {v g q }
Ng q=1, v gq ∈ R
|G(q,t)|, where v g q contains the RG(q,t)
rollout rewards for question q, and N(g,t)is the number of questions in domain g, at iteration step t.

To uncover patterns in question difficulty, we apply K-means clustering to these reward vectors at each time step t, separately within each domain:

$$\{\mathbf{C}_{(1,g,t)},\mathbf{C}_{(2,g,t)},\ldots,\mathbf{C}_{(k_{(g,t)},g,t)}\}=\mathrm{{KMeans}}(\mathcal{H}_{g},k_{(g,t)}),$$

where C(c,g,t) denotes the centroid of cluster c in domain g, and k(g,t)is the number of clusters, which is determined automatically using the elbow method (see Appendix B.1). Stage-2: Hierarchical Scaling. For each domain and each cluster within that domain, we compute inter-domain temperature factors T(g,t) and **intra-domain** temperature factors T(*c,g,t*). These factors capture both the relative size and average difficulty of each domain and cluster. Difficulty is estimated using the mean reward, either per domain or per cluster within the domain, which serves as a proxy for how easy or challenging the questions are within each specific domain and cluster. These temperature factors are then *inversely multiplied* with the corresponding advantage functions—at both the domain and cluster levels—so that domains and clusters that are smaller or harder receive proportionally greater weight during training. Concretely:

$$T_{(g,t)}=\max\left(\sqrt{N_{(g,t)}}\cdot\mu_{(g,t)},\varepsilon\right),\quad T_{(c,g,t)}=\max\left(\sqrt{N_{(c,g,t)}}\cdot\mu_{(c,g,t)},\varepsilon\right),\tag{2}$$  where $N_{(c,g,t)}$ is the size of cluster $c$, and $\mu_{(g,t)}$ and $\mu_{(c,g,t)}$ denote the mean reward for group $g$ and 
cluster c in group g, at iteration t. To scale reward advantage with the appropriate temperature factors, we first normalize rewards at the question level as in GRPO, then scale by the domain and cluster temperatures, before multiplying by a KL regularization factor m(i,t). Concretely,

$$s_{(q,i,t)}^{s c a l e d}=\frac{m_{(i,t)}\cdot s_{(q,i,t)}}{T_{(g,t)}\cdot T_{(c,g,t)}},$$

, (3)
where si =
ri,t−µq,t σq+εis the question level-normalized reward from GRPO. The KL regularization is applied to prevent outliers from dominating the update, as detailed in Appendix B.2. Finally, we scale the standard deviation back to 1 by dividing each reward by the standard deviation of the reward in the batch AˆDRPO
(q,i,t) 
=
s scaled
(*q,i,t*)
σs scaled t
.

DRPO Objective. DRPO maintains the same objective structure as GRPO, maximizing:

$$\tilde{A}_{(q,i,t);k}(\theta)=\min\left(\varphi_{(q,i,t);k}(\theta)\cdot\tilde{A}_{(q,i,t)}^{\mathrm{prop}},\ \mathrm{clip}\left(\varphi_{(q,i,t);k}(\theta),\ 1-\varepsilon,\ 1+\varepsilon\right)\cdot\tilde{A}_{(q,i,t)}^{\mathrm{prop}}\right),$$  $$J_{\mathrm{DMPO}}(\theta)=\mathbb{E}_{q\sim\mathcal{D},\,\{\alpha_{(q,i,t)}\}\sim\pi_{\mathrm{ad}}\ \left[\frac{1}{|G_{(q,t)}|}\ \sum_{i=1}^{|G_{(q,i)}|}\ \frac{1}{1n_{\alpha_{(q,i,t)}}}\ \sum_{k=1}^{n_{\alpha_{(q,i,t)}}}\ \tilde{A}_{(q,i,t);k}(\theta)-\beta\,D_{\mathrm{KL}}\left(\pi_{\theta}\parallel\pi_{\mathrm{rel}}\right)\right],$$
$$({\mathfrak{I}})$$
$\quad(\theta)$ . 
where φ(q,i,t):k(θ) = πθ(o(q,i,t):k|q, o(q,i,t):<k)
πθold (o(q,i,t):k|q, o(*q,i,t*):<k)
.
Benefits of DRPO. The cluster-based DRPO approach offers several key benefits: 1. **Hierarchical Scaling:** DRPO implements two-layer scaling: first at the domain level and then at the cluster level within each domain. This directs optimization toward both underrepresented domains and challenging question subsets, ensuring the model learns effectively across all data types. This approach prevents the model from focusing only on easy or common problems while neglecting rare but important clinical scenarios.

2. **Preservation of Zero Mean and Unit Variance:** DRPO scales rewards after GRPO normalization, maintaining the property that the mean reward within each set of rollouts remains 0 and the standard deviation is 1. This property is crucial for stable optimization in reinforcement learning, as established in previous works [14, 94, 57].

3. **Computational Efficiency:** DRPO operates with minimal additional complexity of order O(n),
primarily from the K-means algorithm operating on low-dimensional vectors (typically 5-10 elements). This enables efficient training without the overhead of critic networks, making it particularly suitable for large-scale LLM fine-tuning.

## 3.4 Reward Design

During the training of QoQ-Med, we employ a combination of two main rewards and two auxiliary rewards that balance diagnostic accuracy with interpretability, a critical requirement for clinical applications where understanding model reasoning.

Accuracy reward. The primary goal of our model is diagnostic accuracy, for which we compute a standard *accuracy reward* r acc i. We treat prediction yˆi and ground truth yi as unordered sets of labels and assign r acc i = F1yˆi, yi
, which directly optimizes the model's ability to identify correct diagnoses across diverse clinical scenarios.

Semantic alignment reward. For clinical applications, the ability to identify and highlight relevant regions in medical imagery is crucial for building clinician trust. The *semantic alignment reward* encourages the model to correctly identify salient regions that support its diagnostic decisions. Let bi = {bi,j}
Ki j=1 be the set of axis-aligned bounding boxes output by the model and Si ⊆ [0, 1]H×W
the pixel-level segmentation mask associated with the ground-truth diagnosis. We define this reward as the best intersection-over-union score: r IoU
i = maxj=1*,...,K*i area bi,j∩Si area bi,j∪Si . By optimizing this reward, the model learns to visually highlight the specific anatomical regions relevant to its diagnosis, providing critical interpretability for clinical decision support. Auxiliary rewards. We also employ auxiliary rewards that encourage proper formatting and comprehensive reasoning, detailed in Appendix B.3. These rewards help ensure that the model's outputs are well-structured and sufficiently detailed for clinical use.

Combined reward. The final scalar reward supplied to DRPO is a weighted combination: ri = λacc r acc i + λIoU r IoU
i + λaux r aux i. In our experiments, we set (λacc, λIoU, λaux) = (0.6, 0.2, 0.2).

## 4 Experiments

We design experiments to answer the following research questions. Details are included in App. D. RQ1: How does DRPO compare with other critic-free RL methods and models? As detailed in Sec. 3.2, we train and evaluate QoQ-Med on a combination of 30 clinical diagnosis datasets across 9 clinical domains. A description of each dataset is included in App. C. The models are evaluated with balanced accuracy and macro-F1. We compare our training method DRPO against supervised fine-tuning (SFT), PPO [74] and four popular critic-free RL training methods: GRPO [75], RLOO [2],
Reinforce++ [33], and ReMax [50]. We further compare our trained model QoQ-Med against medical VLMs (Llava-Med [48], Med-R1 [46]) and closed source VLMs (GPT-4o [37], o4-mini [61]).

RQ2: How well does DRPO handle mixed multimodal inputs? We repeat the comparison on MIMIC-IV, where samples contain a chest X-ray, a 12-lead ECG trace, and an accompanying clinical record. We train and evaluate the models on two tasks: length of stay (LOS) prediction, binned into a 4-day interval, and 48-hour in-hospital mortality (48-IHM). We evaluate the model with accuracy and F1 score in the same way as RQ1. RQ3: How is the quality of the reasoning traces and bounding boxes learned by DRPO? We did both a qualitative and a quantitative analysis on QoQ-Med's reasoning and bounding box outputs.

We evaluate the bounding box quality via the intersection over union (IoU) against the ground truth segmentation available in the dataset. We further collaborated with clinicians to annotate the reasoning traces on the validation dataset, grading the traces by their relevance to the final diagnosis.

## 4.1 Rq1: Comparison With Other Rl Training Methods And Models

Comparison with other RL methods. Table 2 shows a comparison between DRPO and several critic-free RL training methods across eight medical imaging modalities. The results demonstrate that DRPO consistently outperforms all competing methods in 6 out of 8 vision modalities in terms of F1 score. Overall, DRPO achieves a mean accuracy that is 5.9% higher in percentage points and an F1 score that is 46% higher compared to the best critic-free baseline method. As compared to GRPO in Fig. 2(a), the most substantial increase is observed in datasets from understudied modalities, like ultrasound and mammography, as defined in App. C.2. As shown in Fig. 2(b), QoQ-Med achieves the best performance across all clinical domains as compared to current open-source MLLMs. Compared

| included in App. Tab. 7. Model CXR Mammo. Dermoscopy   | CT Scan                  | Fundus                   | Ultrasound               | MRI                           | Pathology                     | Overall   |    |     |    |     |    |     |    |     |    |     |    |
|--------------------------------------------------------|--------------------------|--------------------------|--------------------------|-------------------------------|-------------------------------|-----------|----|-----|----|-----|----|-----|----|-----|----|-----|----|
| Acc                                                    | F1                       | Acc                      | F1                       | Acc                           | F1                            | Acc       | F1 | Acc | F1 | Acc | F1 | Acc | F1 | Acc | F1 | Acc | F1 |
| SFT                                                    | .688 .078 .481 .056 .640 | .158                     | .525 .236 .715 .066 .548 | .235                          | .567 .197 .652 .083 .602 .139 |           |    |     |    |     |    |     |    |     |    |     |    |
| PPO [74]                                               | .670 .064 .738 .205 .668 | .278                     | .571 .257 .669 .083 .490 | .080                          | .767 .540 .745 .364 .665 .234 |           |    |     |    |     |    |     |    |     |    |     |    |
| ReMax [50]                                             | .636 .120 .577 .033 .644 | .257                     | .567 .228 .678 .089 .547 | .147                          | .547 .264 .706 .270 .596 .176 |           |    |     |    |     |    |     |    |     |    |     |    |
| RE++ [33]                                              | .730 .082 .660 .076 .635 | .237                     | .529 .247 .672 .098 .519 | .136                          | .651 .420 .668 .254 .621 .202 |           |    |     |    |     |    |     |    |     |    |     |    |
| RLOO [2]                                               | .752 .086 .471 .068 .636 | .216                     | .534 .224 .670 .099 .519 | .144                          | .658 .432 .699 .216 .611 .189 |           |    |     |    |     |    |     |    |     |    |     |    |
| GRPO [75]                                              | .703 .095 .466 .059 .646 | .244                     | .524 .236 .670 .086 .520 | .146                          | .631 .395 .715 .286 .609 .193 |           |    |     |    |     |    |     |    |     |    |     |    |
| DRPODomainOnly .693 .086 .751 .213 .679                | .251                     | .571 .257 .669 .083 .480 | .098                     | .733 .475 .762 .388 .668 .237 |                               |           |    |     |    |     |    |     |    |     |    |     |    |
| DRPONoKL                                               | .685 .103 .711 .264 .691 | .382                     | .597 .365 .676 .085 .554 | .228                          | .722 .535 .710 .300 .668 .283 |           |    |     |    |     |    |     |    |     |    |     |    |
| DRPO                                                   | .687 .115 .756 .253 .715 | .407                     | .570 .309 .672 .093 .555 | .223                          | .789 .625 .708 .265 .666 .295 |           |    |     |    |     |    |     |    |     |    |     |    |

Table 3: **Ablation studies on cluster size and reward composition.** Acc: Accuracy, F1: F1 Score.

Bold values indicate best performance within each ablation group.

| Config                                                        | CXR                      | Mammo.                   | Dermoscopy               | CT Scan                       | Fundus                        | Ultrasound   | MRI   | Pathology   | Overall   |     |    |     |    |     |    |     |    |
|---------------------------------------------------------------|--------------------------|--------------------------|--------------------------|-------------------------------|-------------------------------|--------------|-------|-------------|-----------|-----|----|-----|----|-----|----|-----|----|
| Acc                                                           | F1                       | Acc                      | F1                       | Acc                           | F1                            | Acc          | F1    | Acc         | F1        | Acc | F1 | Acc | F1 | Acc | F1 | Acc | F1 |
| Cluster Size 1 .694 .085 .746 .211 .678                       | .286                     | .571 .257 .669 .083 .544 | .200                     | .757 .505 .773 .449 .679 .259 |                               |              |       |             |           |     |    |     |    |     |    |     |    |
| 3                                                             | .694 .125 .568 .048 .680 | .356                     | .562 .284 .672 .147 .520 | .152                          | .717 .546 .723 .289 .642 .244 |              |       |             |           |     |    |     |    |     |    |     |    |
| 10                                                            | .691 .125 .759 .253 .707 | .400                     | .580 .321 .670 .088 .568 | .240                          | .806 .652 .707 .303 .686 .286 |              |       |             |           |     |    |     |    |     |    |     |    |
| 20                                                            | .668 .167 .751 .268 .675 | .300                     | .548 .262 .635 .166 .547 | .214                          | .804 .649 .731 .329 .670 .294 |              |       |             |           |     |    |     |    |     |    |     |    |
| Reward Composition (Acc:IoU) 0.6:0.2 .691 .125 .759 .253 .707 | .400                     | .580 .321 .670 .088 .568 | .240                     | .806 .652 .707 .303 .686 .286 |                               |              |       |             |           |     |    |     |    |     |    |     |    |
| 0.2:0.6 .690 .147 .563 .185 .668                              | .290                     | .576 .308 .681 .136 .573 | .218                     | .768 .561 .698 .233 .652 .260 |                               |              |       |             |           |     |    |     |    |     |    |     |    |

to the closed-source commercial models, it achieves the best performance against GPT-4o [37], while surpassing the reasoning model GPT-o4-mini [60] in all domains except MRI. Ablations. The substantial improvement in F1 score can be attributed to two key components of DRPO. First, the introduction of domain-wise scaling contributes to a significant 22.8% improvement in F1 score, as evidenced by the performance difference between DRPODomainOnly and vanilla GRPO.

Subsequently, after incorporating clustering within each domain and specifically encouraging the model to focus on small, challenging clusters within each domain, the performance is further enhanced by an additional 19.4% in terms of F1 score. Tab. 3 shows further ablations on the number of clusters and reward compositions. In general, we found that the weight of each reward does not have a significant impact on the final performance. In particular, the auxiliary rewards on formatting saturate shortly in the early stages of training. They have effectively no impact on the later stages due to normalization. We tested different combinations of accuracy rewards: semantic alignment rewards. As demonstrated in the table, decreasing the weight of the accuracy reward gives a drop in overall performance and performance in most domains, but results are still significantly better than all baselines, which demonstrates the robustness of DRPO. The number of clusters in the model is determined automatically via the elbow method, with the possibility to set an upper limit on the number of clusters. As a part of the ablation, we tested the model with 1 (no clustering), 3, 10 and 20 clusters, and included the results in Tab. 3. In general, we observe that having no cluster or a very low cluster limit will cause a decrease in performance. A higher cluster limit, however, does not seem to hurt the performance, as the elbow method automatically chooses a lower cluster count than the limit. This allows the algorithm to remain efficient under arbitrary cluster limits. Runtime Efficiency. As shown in Fig. 3(c), while DRPO requires clusters to be calculated on each step, it has a negligible impact on the overall runtime. Across all critic-free RL methods, reward calculation accounts for less than 2% of the total runtime of a step.

(a) (b)
(a) (b) (c)

## 4.2 Rq2: Multimodal Fusion Performance

We tested how the model integrates multiple modalities and how much each modality contributes to the final diagnostic accuracy via MIMIC-IV [41] dataset. On the MIMIC-IV
dataset, the model has to reason across ECGs, chest X-rays, and health records. As shown in Tab. 4, we found DRPO allows the model to reach a better performance in both tasks as compared to GRPO. In addition, taking full inputs across ECG, Chest X-ray images, and electronic health records (EHR) gives better performance than any ablation of these modalities, signaling that QoQ-Med is able to effectively aggregate information across all modalities. Specifically, we found vision and texts contirbute more to the final accuracy and F1 scores than ECG. While QoQ-Med represents a first step towards multimodal reasoning models across vision and time series, future works could explore better architecture, data, or training methods that better balances the power of each modalities.

Table 4: **Models' Perf. on MIMIC-IV.** DRPO- Full with inputs from 3 modalities has the best performance for both tasks, time-series only (DRPO-
TS+T) and vision only (DRPO-Vision+T) ablations having worse performance, and the text only ablation having the worst performance.

| Algo/Inputs   | LOS   | 48-IHM   |       |       |
|---------------|-------|----------|-------|-------|
| Acc           | F1    | Acc      | F1    |       |
| GRPO-Full     | 0.626 | 0.105    | 0.551 | 0.354 |
| DRPO-TextOnly | 0.645 | 0.195    | 0.563 | 0.583 |
| DRPO-ECG+T    | 0.639 | 0.204    | 0.602 | 0.528 |
| DRPO-Vision+T | 0.669 | 0.223    | 0.596 | 0.586 |
| DRPO-Full     | 0.663 | 0.283    | 0.642 | 0.597 |

Quest i on: Above i s a cl i ni cal i mage of a pat i ent . What i s t he di agnosi s of t he pat i ent i n t he cl i ni cal i mage?

Gr ound t r ut h: Basal cel l car ci noma Model Reasoni ng:
. . . not a dar k mol e. . . sl i ght l y r ai sed, pi nki sh, and sl i ght l y el evat ed ar ea. . . l esi ons have a r ough, scal y sur f ace, whi ch i s mor e consi st ent wi t h t he t ext ur e i n t hi s i mage. . . l esi on' s col or , shape, and l ocat i on ar e consi st ent wi t h basal cel l car ci noma. . . 

Answer : Basal cel l car ci noma Quest i on: Above i s a chest X- r ay i mage of a pat i ent . What i s t he di agnosi s of t he pat i ent i n t he X- r ay i mage? Gr ound t r ut h: Suppor t devi ces Model Reasoni ng:
. . . medi ast i nal ar ea appear s t o be enl ar ged . . . t he x- r ay shows a pacemaker , whi ch i s a medi cal devi ce. . . pl eur al ef f usi on, whi ch i s bui l dup of f l ui d. . . r adi ogr aphi c mar ker s. . . ar e st andar d f or x- r ays t o i ndi cat e t he posi t i on. . . No Fi ndi ng f or ot her abnor mal i t i es Answer : No f i ndi ng Quest i on: Above i s a br ai n CT scan sl i ce. I s t her e any hemor r hage pr esent i n t hi s CT scan sl i ce? Gr ound t r ut h: No hemor r hage Model Reasoni ng:
. . . CT scan i s a cr oss- sect i onal vi ew of t he body. . . si gns of hemor r hage i ncl ude. . . hypodense ( dar ker t han nor mal br ai n t i ssue) . . . hyper dense ( whi t er t han nor mal br ai n t i ssue) . . . t i ssue appear s uni f r oml y gr ay, wi t hout any obvi ous br i ght or dar k ar eas t hat woul d suggest hemor r hage. . . 

Answer : No hemor r hage

(a) (b)
Quest i on: Above i s a br ai n MRI scan. I s t her e a t umor pr esent i n t hi s br ai n MRI scan?

Gr ound t r ut h: Has t umor Model Reasoni ng:
. . . t umor appear s as a r ound, wel l - ci r cumscr i bed l esi on. . . t umor s ar e of t en char act er i zed as havi ng a di st i nct , of t en i r r egul ar , boundar y . . . t umor i s cl ear l y di st i nct f r om t he sur r oundi ng br ai n t i ssue, whi ch i s a st r ong i ndi cat or of a l esi on t ha coul d be a t umor . . . t umor i s l ocat ed i n t he br ai n. . . i n a r egi on wher e t umor s ar e common. . . 

Answer : Has t umor
(c) (d)
Quest i on: Bel ow i s a hi st or y of a pat i ent : . . . How l ong wi l l t he pat i ent st ay i n t he hospi t al ? A. 0- 4 days B. 5- 8 days C. 9- 12 days D. mor e t han 12 days Gr ound t r ut h: A. 0- 4 days Model Reasoni ng:
. . . The pat i ent ' s hi st or y i ncl udes X- r ay i mages showi ng no obvi ous acut e pat hol ogy i n t he l ungs, whi ch coul d be i ndi cat i ve of a st abl e condi t i on. The ECG appear s t o be unr emar kabl e, and t her e ar e no i mmedi at e si gns of acut e hear t f ai l ur e or ot her cr i t i cal condi t i ons on t he ECG. The l abs show nor mal hemat ol ogi cal par amet er s, whi ch suggest t hat t her e ar e no i mmedi at e l i f e- t hr eat eni ng condi t i ons l i ke sever e anemi a, i nf ect i on, or el ect r ol yt e i mbal ances. 

Answer : A. 0- 4 days
(e)
Relevance to diagnosis: high low no

## 4.3 Rq3: Quality Of Reasoning Traces

Clinician relevance annotations. App. D.4.2 provides a breakdown of clinician-annotated reasoning traces, revealing that the model mostly generates contents highly relevant to the diagnosis, with minimal output judged as irrelevant. We observe that the model often correctly recalls relevant clinical knowledge, which help guide the model by providing associative context. For example, in Fig 4(a), the model correctly recalls different signs of hemorrhage on CT, such as darker or whiter tissues, and relates this context to specific parts of the image to make a correct prediction. In Fig. 4(c), the model correctly identifies the presence of a pacemaker, indicating a support device, but subsequently concludes that there are no additional abnormalities, ultimately leading it to predict "No finding". This suggests that while the model's final predictions may be incorrect, its intermediate reasoning often reflects clinically relevant patterns.

Bounding box quality. Fig. 3(b) demonstrates that the model identifies bounding boxes correlated with the ground truth annotations, with the IoU exceeding the best open source models while reaching a similar performance as the closed-source reasoning model o4-mini. From Fig. 4, we also see that the outputs by the model are sufficiently aligned with the reasoning process, allowing the clinicians to confirm the model's predictions while cross-referencing the source image.

## 5 Conclusion

We introduced QoQ-Med, a clinical MLLM with reasoning across 9 clinical domains. Our Domainaware Group Relative Policy Optimization (DRPO) demonstrates superior performance over existing approaches, with up to 43% improvement in average F1 score across clinical modalities and substantial gains in multimodal fusion tasks. The ability of QoQ-Med to process 1D time series data alongside traditional 2D/3D clinical images addresses a significant gap in existing medical multimodal systems, while its transparent reasoning process enhances interpretability and clinical trust. By publicly releasing QoQ-Med-7B/32B and our comprehensive reasoning dataset containing 2.61 million question-answer pairs, we hope to contribute valuable resources to advance clinical reasoning AI.

A potential limitation is the limited sample efficiency as the reasoning process is not supervised. Moving forward, we hope the community can explore ways to elicit high-quality reasoning with better data efficiency, with a special focus on understudied modalities like ECG and ultrasound.

## 6 Acknowledgement

This material is based upon work supported by the National Science Foundation Graduate Research Fellowship under Grant No. 2141064. Any opinion, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation. We thank the MIT Office of Research Computing and Data (ORCD) for support through ORCD Seed Fund Grants, which provided access to 8xH200 GPUs and additional funding support. We also thank the NVIDIA Academic Grant Program for GPU support. We also extend our sincere thanks to Haowen Wei (Research Associate, MIT Institute for Medical Engineering & Science, MIT.nano) and Dr. Farzan Vahedifard (Neurologist, Athinoula A. Martinos Center for Biomedical Imaging, Harvard Medical School) for their careful annotation of the model's reasoning outputs and valuable insights that significantly improved this work.

## References

[1] Armen Aghajanyan, Lili Yu, Alexis Conneau, Wei-Ning Hsu, Karen Hambardzumyan, Susan Zhang, Stephen Roller, Naman Goyal, Omer Levy, and Luke Zettlemoyer. Scaling laws for generative mixedmodal language models. In *International Conference on Machine Learning*, pages 265–279. PMLR, 2023.

[2] Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting reinforce-style optimization for learning from human feedback in llms. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 12248–12267. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.ACL-LONG.662. URL https://doi.org/10.18653/v1/ 2024.acl-long.662.

[3] Walid Al-Dhabyani, Mohammed Gomaa, Hussien Khaled, and Aly Fahmy. Dataset of breast ultrasound images. *Data in brief*, 28:104863, 2020.

[4] Erick A Perez Alday, Annie Gu, Amit J Shah, Chad Robichaux, An-Kwok Ian Wong, Chengyu Liu, Feifei Liu, Ali Bahrami Rad, Andoni Elola, Salman Seyedi, et al. Classification of 12-lead ecgs: the physionet/computing in cardiology challenge 2020. *Physiological measurement*, 41(12):124003, 2020.

[5] Mohammad Alkhalaf, Ping Yu, Mengyang Yin, and Chao Deng. Applying generative ai with retrieval augmented generation to summarize and extract key clinical information from electronic health records. Journal of biomedical informatics, 156:104662, 2024.

[6] Julia Amann, Alessandro Blasimme, Effy Vayena, Dietmar Frey, Vince I. Madai, and the Precise4Q consortium. Explainability for artificial intelligence in healthcare: a multidisciplinary perspective. *BMC Medical* Informatics and Decision Making, 20(1):310, 2020.

[7] Mohamed Amgad, Habiba Elfandy, Hagar Hussein, Lamees A Atteya, Mai AT Elsebaie, Lamia S Abo Elnasr, Rokia A Sakr, Hazem SE Salem, Ahmed F Ismail, Anas M Saad, et al. Structured crowdsourcing enables convolutional segmentation of histology images. *Bioinformatics*, 35(18):3461–3467, 2019.

[8] Asia Pacific Tele-Ophthalmology Society. APTOS 2019 blindness detection. https://www.kaggle.

com/c/aptos2019-blindness-detection/data, 2019. [Dataset].

[9] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. *arXiv preprint arXiv:2502.13923*, 2025.

[10] Sartaj Bhuvaji, Ankita Kadam, Prajakta Bhumkar, Sameer Dedge, and Swati Kanchan. Brain tumor classification (mri), 2020. URL https://www.kaggle.com/dsv/1183165.

[11] Andreea Bodnari and John Travis. Scaling enterprise ai in healthcare: the role of governance in risk mitigation frameworks. *npj Digital Medicine*, 8(1):272, 2025.

[12] Karsten M Borgwardt, Cheng Soon Ong, Stefan Schönauer, SVN Vishwanathan, Alex J Smola, and Hans-Peter Kriegel. Protein function prediction via graph kernels. *Bioinformatics*, 21(suppl_1):i47–i56, 2005.

[13] Andrew A Borkowski, Marilyn M Bui, L Brannon Thomas, Catherine P Wilson, Lauren A DeLand, and Stephen M Mastorides. Lung and colon cancer histopathological image dataset (lc25000). arXiv preprint arXiv:1912.12142, 2019.

[14] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. *Advances in neural information processing systems*, 30, 2017.

[15] Joseph Paul Cohen, Paul Morrison, and Lan Dao. Covid-19 image data collection. *arXiv 2003.11597*,
2020. URL https://github.com/ieee8023/covid-chestxray-dataset.

[16] Errol Colak, Felipe C Kitamura, Stephen B Hobbs, Carol C Wu, Matthew P Lungren, Luciano M Prevedello, Jayashree Kalpathy-Cramer, Robyn L Ball, George Shih, Anouk Stein, et al. The rsna pulmonary embolism ct dataset. *Radiology: Artificial Intelligence*, 3(2):e200254, 2021.

[17] Can Cui, Haichun Yang, Yaohong Wang, Shilin Zhao, Zuhayr Asad, Lori A Coburn, Keith T Wilson, Bennett A Landman, and Yuankai Huo. Deep multimodal fusion of image and non-image data in disease diagnosis and prognosis: a review. *Progress in Biomedical Engineering*, 5(2):022001, 2023.

[18] Chunyan Cui, Li Li, Hongmin Cai, Zhihao Fan, Ling Zhang, Tingting Dan, Jiao Li, and Jinghua Wang. The chinese mammography database (cmmd): An online mammography database with biopsy confirmed types for machine diagnosis of breast. *The Cancer Imaging Archive*, 2021. doi: 10.7937/TCIA.EQDE-4B16.

URL https://doi.org/10.7937/tcia.eqde-4b16.

[19] Hejie Cui, Wei Dai, Yanqiao Zhu, Xuan Kan, Antonio Aodong Chen Gu, Joshua Lukemire, Liang Zhan, Lifang He, Ying Guo, and Carl Yang. Braingb: a benchmark for brain network analysis with graph neural networks. *IEEE transactions on medical imaging*, 42(2):493–506, 2022.

[20] Hejie Cui, Lingjun Mao, Xin Liang, Jieyu Zhang, Hui Ren, Quanzheng Li, Xiang Li, and Carl Yang.

Biomedical visual instruction tuning with clinician preference alignment. *arXiv preprint arXiv:2406.13173*, 2024.

[21] Wei Dai, Ehsan Adeli, Zelun Luo, Dev Dash, Shrinidhi Lakshmikanth, Zane Durante, Paul Tang, Amit Kaushal, Arnold Milstein, Li Fei-Fei, et al. Developing icu clinical behavioral atlas using ambient intelligence and computer vision. *NEJM AI*, page AIoa2400590, 2025.

[22] Wei Dai, Peilin Chen, Malinda Lu, Daniel Li, Haowen Wei, Hejie Cui, and Paul Pu Liang. Climb: Data foundations for large scale multimodal clinical foundation models. *ICML*, 2025.

[23] Eric Decencière, Claire LaGraize, Pascale Pélégrin, François Benassi, Christian Régér, and Thomas Vautrin.

Feedback on a publicly distributed database: the messidor database. *Image Analysis & Stereology*, 33(3): 231–234, 2014. ISSN 1854-5165. doi: 10.5566/ias.1155. URL http://dx.doi.org/10.5566/ias. 1155.

[24] Franck Dernoncourt and Ji Young Lee. Pubmed 200k rct: a dataset for sequential sentence classification in medical abstracts. *arXiv preprint arXiv:1710.06071*, 2017.

[25] Ashkan Ebadi, Pengcheng Xi, Alexander MacLean, Stéphane Tremblay, Sonny Kohli, and Alexander Wong. Covidx-us–an open-access benchmark dataset of ultrasound imaging data for ai-driven covid-19 analytics. *arXiv preprint arXiv:2103.10003*, 2021.

[26] Mark Endo, Rayan Krishnan, Viswesh Krishna, Andrew Y Ng, and Pranav Rajpurkar. Retrieval-based chest x-ray report generation using a pre-trained contrastive language-image model. In *Machine Learning* for Health, pages 209–219. PMLR, 2021.

[27] Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Vlad Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, et al. Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures. In *International conference on machine learning*, pages 1407–1416. PMLR,
2018.

[28] Suparna Ghanvatkar and Vaibhav Rajan. Graph-based patient representation for multimodal clinical data:
Addressing data heterogeneity. *medRxiv*, pages 2023–12, 2023.

[29] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025.

[30] Nicholas Heller, Fabian Isensee, Dasha Trofimova, Resha Tejpaul, Zhongchen Zhao, Huai Chen, Lisheng Wang, Alex Golts, Daniel Khapun, Daniel Shats, et al. The kits21 challenge: Automatic segmentation of kidneys, renal tumors, and renal cysts in corticomedullary-phase ct. *arXiv preprint arXiv:2307.01984*, 2023.

[31] Matteo Hessel, Hubert Soyer, Lasse Espeholt, Wojciech Czarnecki, Simon Schmitt, and Hado Van Hasselt.

Multi-task deep reinforcement learning with popart. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 3796–3803, 2019.

[32] Murtadha Hssayeni, M Croock, A Salman, H Al-khafaji, Z Yahya, and B Ghoraani. Computed tomography images for intracranial hemorrhage detection and segmentation. Intracranial hemorrhage segmentation using a deep convolutional model. Data, 5(1):14, 2020.

[33] Jian Hu. Reinforce++: A simple and efficient approach for aligning large language models. arXiv preprint arXiv:2501.03262, 2025.

[34] Jie Huang and Kevin Chen-Chuan Chang. Towards reasoning in large language models: A survey. *arXiv* preprint arXiv:2212.10403, 2022.

[35] Shih-Cheng Huang, Zepeng Huo, Ethan Steinberg, Chia-Chun Chiang, Matthew P Lungren, Curtis P
Langlotz, Serena Yeung, Nigam H Shah, and Jason A Fries. Inspect: a multimodal dataset for pulmonary embolism diagnosis and prognosis. *arXiv preprint arXiv:2311.10798*, 2023.

[36] Yu Huang, Junyang Lin, Chang Zhou, Hongxia Yang, and Longbo Huang. Modality competition: What makes joint training of multi-modal network fail in deep learning?(provably). In *International conference* on machine learning, pages 9226–9259. PMLR, 2022.

[37] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. *arXiv preprint arXiv:2410.21276*, 2024.

[38] Jeremy Irvin, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Christopher Chute, Henrik Marklund, Behzad Haghgoo, Robyn L. Ball, Katie S. Shpanskaya, Jayne Seekins, David A. Mong, Safwan S. Halabi, Jesse K. Sandberg, Ricky Jones, David B. Larson, Curtis P. Langlotz, Bhavik N. Patel, Matthew P. Lungren, and Andrew Y. Ng. Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019, The Thirty-First Innovative Applications of Artificial Intelligence Conference, IAAI 2019, The Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019, Honolulu, Hawaii, USA,
January 27 - February 1, 2019, pages 590–597. AAAI Press, 2019. doi: 10.1609/AAAI.V33I01.3301590. URL https://doi.org/10.1609/aaai.v33i01.3301590.

[39] Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William W Cohen, and Xinghua Lu. Pubmedqa: A dataset for biomedical research question answering. *arXiv preprint arXiv:1909.06146*, 2019.

[40] Alistair E. W. Johnson, Tom J. Pollard, Seth J. Berkowitz, Nathaniel R. Greenbaum, Matthew P. Lungren, Chih-ying Deng, Roger G. Mark, and Steven Horng. MIMIC-CXR: A large publicly available database of labeled chest radiographs. CoRR, abs/1901.07042, 2019. URL http://arxiv.org/abs/1901.07042.

[41] Alistair EW Johnson, Lucas Bulgarelli, Lu Shen, Alvin Gayles, Ayad Shammout, Steven Horng, Tom J
Pollard, Sicheng Hao, Benjamin Moody, Brian Gow, et al. Mimic-iv, a freely accessible electronic health record dataset. *Scientific data*, 10(1):1, 2023.

[42] Xuan Kan, Wei Dai, Hejie Cui, Zilong Zhang, Ying Guo, and Carl Yang. Brain network transformer.

Advances in Neural Information Processing Systems, 35:25586–25599, 2022.

[43] Sehun Kim. Learning general representation of 12-lead electrocardiogram with a joint-embedding predictive architecture. *arXiv preprint arXiv:2410.08559*, 2024.

[44] Adrienne Kline, Hanyin Wang, Yikuan Li, Saya Dennis, Meghan Hutch, Zhenxing Xu, Fei Wang, Feixiong Cheng, and Yuan Luo. Multimodal machine learning in precision health: A scoping review. npj Digital Medicine, 5(1):171, 2022.

[45] Nesaretnam Barr Kumarakulasinghe, Tobias Blomberg, Jintai Liu, Alexandra Saraiva Leao, and Panagiotis Papapetrou. Evaluating local interpretable model-agnostic explanations on clinical machine learning classification models. In 2020 IEEE 33rd international symposium on computer-based medical systems
(CBMS), pages 7–12. IEEE, 2020.

[46] Yuxiang Lai, Jike Zhong, Ming Li, Shitian Zhao, and Xiaofeng Yang. Med-r1: Reinforcement learning for generalizable medical reasoning in vision-language models. *arXiv preprint arXiv:2503.13939*, 2025.

[47] Xiang Lan, Feng Wu, Kai He, Qinghao Zhao, Shenda Hong, and Mengling Feng. Gem: Empowering mllm for grounded ecg understanding with time series and images. *arXiv preprint arXiv:2503.06073*, 2025.

[48] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large language-and-vision assistant for biomedicine in one day. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, *Advances in Neural Information Processing Systems 36: Annual Conference on* Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023.

[49] Stella Li, Vidhisha Balachandran, Shangbin Feng, Jonathan Ilgen, Emma Pierson, Pang Wei W Koh, and Yulia Tsvetkov. Mediq: Question-asking llms and a benchmark for reliable interactive clinical reasoning. Advances in Neural Information Processing Systems, 37:28858–28888, 2024.

[50] Ziniu Li, Tian Xu, Yushun Zhang, Zhihang Lin, Yang Yu, Ruoyu Sun, and Zhi-Quan Luo. Remax: A
simple, effective, and efficient reinforcement learning method for aligning large language models. In Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024.

OpenReview.net, 2024. URL https://openreview.net/forum?id=Stn8hXkpe6.

[51] Paul Pu Liang, Akshay Goindani, Talha Chafekar, Leena Mathur, Haofei Yu, Russ Salakhutdinov, and Louis-Philippe Morency. Hemm: Holistic evaluation of multimodal foundation models. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024.

[52] Paul Pu Liang, Amir Zadeh, and Louis-Philippe Morency. Foundations & trends in multimodal machine learning: Principles, challenges, and open questions. *ACM Computing Surveys*, 56(10):1–42, 2024.

[53] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In *Proceedings of the IEEE international conference on computer vision*, pages 2980–2988, 2017.

[54] Feifei Liu, Chengyu Liu, Lina Zhao, Xiangyu Zhang, Xiaoling Wu, Xiaoyan Xu, Yulin Liu, Caiyun Ma, Shoushui Wei, Zhiqiang He, et al. An open access database for evaluating the algorithms of electrocardiogram rhythm and morphology abnormality detection. *Journal of Medical Imaging and Health Informatics*, 8(7):1368–1373, 2018.

[55] Mary M Lucas, Justin Yang, Jon K Pomeroy, and Christopher C Yang. Reasoning with large language models for medical question answering. *Journal of the American Medical Informatics Association*, 31(9):
1964–1975, 2024.

[56] Michael Moor, Qian Huang, Shirley Wu, Michihiro Yasunaga, Yash Dalmia, Jure Leskovec, Cyril Zakka, Eduardo Pontes Reis, and Pranav Rajpurkar. Med-flamingo: a multimodal medical few-shot learner. In Stefan Hegselmann, Antonio Parziale, Divya Shanmugam, Shengpu Tang, Mercy Nyamewaa Asiedu, Serina Chang, Tom Hartvigsen, and Harvineet Singh, editors, Machine Learning for Health, ML4H@NeurIPS 2023, 10 December 2023, New Orleans, Louisiana, USA, volume 225 of *Proceedings of Machine Learning* Research, pages 353–367. PMLR, 2023. URL https://proceedings.mlr.press/v225/moor23a. html.

[57] Abhishek Naik, Yi Wan, Manan Tomar, and Richard S Sutton. Reward centering. arXiv preprint arXiv:2405.09999, 2024.

[58] Nida Nasir, Afreen Kansal, Feras Barneih, Omar Al-Shaltone, Talal Bonny, Mohammad Al-Shabi, and Ahmed Al Shammaa. Multi-modal image classification of covid-19 cases using computed tomography and x-rays scans. *Intelligent Systems with Applications*, 17:200160, 2023.

[59] Ha Quy Nguyen, Hieu Huy Pham, Tuan Linh Le, Minh Dao, and Khanh Lam. Vindr-cxr: An open dataset of chest x-rays with radiologist annotations. *PhysioNet*, 2021. doi: 10.13026/3akn-b287. URL
https://doi.org/10.13026/3akn-b287.

[60] OpenAI. Gpt-4o mini: advancing cost-efficient intelligence. Online technical report, OpenAI, 2025. URL
https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/.

[61] OpenAI. Openai o3 and o4-mini system card. Technical report, OpenAI, April 2025. URL https://cdn.openai.com/pdf/2221c875-02dc-4789-800b-e7758f3722c1/ o3-and-o4-mini-system-card.pdf.

[62] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. *Advances in neural information processing systems*, 35:27730–27744, 2022.

[63] Ece Ozkan and Xavier Boix. Multi-domain improves classification in out-of-distribution and data-limited scenarios for medical image analysis. *Scientific Reports*, 14(1):24412, 2024.

[64] Andre G.C. Pacheco, Gustavo R. Lima, Amanda S. Salomão, Breno Krohling, Igor P. Biral, Gabriel G.

de Angelo, Fábio C.R. Alves Jr, José G.M. Esgario, Alana C. Simora, Pedro B.C. Castro, Felipe B. Rodrigues, Patricia H.L. Frasson, Renato A. Krohling, Helder Knidel, Maria C.S. Santos, Rachel B. do Espírito Santo, Telma L.S.G. Macedo, Tania R.P. Canuto, and Luíz F.S. de Barros. Pad-ufes-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones. *Data in Brief*, 32:106221, 2020. doi: 10.1016/j.dib.2020.106221. URL https://doi.org/10.1016/j.dib.2020.

106221.

[65] Jiazhen Pan, Che Liu, Junde Wu, Fenglin Liu, Jiayuan Zhu, Hongwei Bran Li, Chen Chen, Cheng Ouyang, and Daniel Rueckert. Medvlm-r1: Incentivizing medical reasoning capability of vision-language models (vlms) via reinforcement learning. *arXiv preprint arXiv:2502.19634*, 2025.

[66] Liron Pantanowitz, Matthew Hanna, Joshua Pantanowitz, Joe Lennerz, Walter H. Henricks, Peter Shen, Bruce Quinn, Shannon Bennet, and Hooman H. Rashidi. Regulatory aspects of artificial intelligence and machine learning. *Modern Pathology*, 37(12):100609, 2024.

[67] João Pedrosa, Carlos Guilherme, Patrícia Márcio, João André, Isabel Eduardo, and Aurélio António. Lndb dataset (version 4). In *17th International Conference on Image Analysis and Recognition (ICIAR 2020)*. Zenodo, 2023. doi: 10.5281/zenodo.8348419. URL https://doi.org/10.5281/zenodo.8348419.

[68] Hieu Huy Pham, Trung H Nguyen, and Ha Quy Nguyen. Vindr-mammo: A large-scale benchmark dataset for computer-aided detection and diagnosis in full-field digital mammography. *PhysioNet*, 2022. URL
https://doi.org/10.13026/br2v-7517.

[69] Sawyer-Lee R., Gimenez F., Hoogi A., and Rubin D. Curated breast imaging subset of digital database for screening mammography (cbis-ddsm) [data set], 2016.

[70] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn.

Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023.

[71] V. Rotemberg, N. Kurtansky, B. Betz-Stablein, and et al. A patient-centric dataset of images and metadata for identifying melanomas using clinical context. *Scientific Data*, 8(1):34, 2021. doi:
10.1038/s41597-021-00815-z. URL https://doi.org/10.1038/s41597-021-00815-z.

[72] Madeline Sagona, Tinglong Dai, Mario Macis, and Michael Darden. Trust in ai-assisted health systems and ai's trust in humans. *npj Health Systems*, 2(1):10, 2025.

[73] Michael Santacroce, Yadong Lu, Han Yu, Yuanzhi Li, and Yelong Shen. Efficient rlhf: Reducing the memory usage of ppo. *arXiv preprint arXiv:2309.00754*, 2023.

[74] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*, 2017.

[75] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. *arXiv preprint arXiv:2402.03300*, 2024.

[76] Gregor Stiglic, Primoz Kocbek, Nino Fijacko, Marinka Zitnik, Katrien Verbert, and Leona Cilar. Interpretability of machine learning-based prediction models in healthcare. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 10(5):e1379, 2020.

[77] Hong Sun, Kristof Depraetere, Laurent Meesseman, Patricia Cabanillas Silva, Ralph Szymanowsky, Janis Fliegenschmidt, Nikolai Hulde, Vera von Dossow, Martijn Vanbiervliet, Jos De Baerdemaeker, et al.

Machine learning–based prediction models for different clinical risks in different hospitals: evaluation of live performance. *Journal of Medical Internet Research*, 24(6):e34295, 2022.

[78] Hidenori Takahashi, Hironobu Tampo, Yusuke Arai, Yuji Inoue, and Hidetoshi Kawashima. Applying artificial intelligence to disease staging: Deep learning for improved staging of diabetic retinopathy. PloS one, 12(6):e0179790, 2017.

[79] Qwen Team. Qwen3 technical report. Technical report, Alibaba, 2025. URL https://github.com/
QwenLM/Qwen3/blob/main/Qwen3_Technical_Report.pdf. Online; accessed May 14, 2025.

[80] Qiaoying Teng, Zhe Liu, Yuqing Song, Kai Han, and Yang Lu. A survey on the interpretability of deep learning in medical diagnosis. *Multimedia Systems*, 28(6):2335–2355, 2022.

[81] Philipp Tschandl, Cliff Rosendahl, and Harald Kittler. The HAM10000 dataset: A large collection of multi-source dermatoscopic images of common pigmented skin lesions. *CoRR*, abs/1803.10417, 2018. URL http://arxiv.org/abs/1803.10417.

[82] Patrick Wagner, Nils Strodthoff, Ralf-Dieter Bousseljot, Dieter Kreiseler, Fatima I Lunze, Wojciech Samek, and Tobias Schaeffter. Ptb-xl, a large publicly available electrocardiography dataset. *Scientific data*, 7(1):
1–15, 2020.

[83] Zhongwei Wan, Che Liu, Xin Wang, Chaofan Tao, Hui Shen, Zhenwu Peng, Jie Fu, Rossella Arcucci, Huaxiu Yao, and Mi Zhang. Electrocardiogram instruction tuning for report generation. *arXiv e-prints*, pages arXiv–2403, 2024.

[84] Nina Wie. Covid-blues: A large-scale lung ultrasound dataset for covid-19 diagnosis. https://github.

com/NinaWie/COVID-BLUES, 2021. Maastricht University Medical Center.

[85] Chaoyi Wu, Xiaoman Zhang, Ya Zhang, Yanfeng Wang, and Weidi Xie. Towards generalist foundation model for radiology by leveraging web-scale 2d&3d medical data. *arXiv preprint arXiv:2308.02463*, 2023.

[86] Peng Xia, Ze Chen, Juanxi Tian, Yangrui Gong, Ruibo Hou, Yue Xu, Zhenbang Wu, Zhiyuan Fan, Yiyang Zhou, Kangyu Zhu, et al. Cares: A comprehensive benchmark of trustworthiness in medical vision language models. *Advances in Neural Information Processing Systems*, 37:140334–140365, 2024.

[87] Chao Yan, Yao Yan, Zhiyu Wan, Ziqi Zhang, Larsson Omberg, Justin Guinney, Sean D Mooney, and Bradley A Malin. A multifaceted benchmarking of synthetic electronic health record generation models. Nature communications, 13(1):7609, 2022.

[88] Jin Ye, Guoan Wang, Yanjun Li, Zhongying Deng, Wei Li, Tianbin Li, Haodong Duan, Ziyan Huang, Yanzhou Su, Benyou Wang, et al. Gmai-mmbench: A comprehensive multimodal evaluation benchmark towards general medical ai. *Advances in Neural Information Processing Systems*, 37:94327–94427, 2024.

[89] Liping Yu, Cristiano Cuppini, Jinghong Xu, Benjamin A Rowland, and Barry E Stein. Cross-modal competition: the default computation for multisensory processing. *Journal of Neuroscience*, 39(8):
1374–1385, 2019.

[90] Kai Zhang, Rong Zhou, Eashan Adhikarla, Zhiling Yan, Yixin Liu, Jun Yu, Zhengliang Liu, Xun Chen, Brian D Davison, Hui Ren, et al. A generalist vision–language foundation model for diverse biomedical tasks. *Nature Medicine*, pages 1–13, 2024.

[91] Sheng Zhang, Qianchu Liu, Guanghui Qin, Tristan Naumann, and Hoifung Poon. Med-rlvr: Emerging medical reasoning from a 3b base model via reinforcement learning. *arXiv preprint arXiv:2502.19655*,
2025.

[92] Xiaoman Zhang, Chaoyi Wu, Ziheng Zhao, Weixiong Lin, Ya Zhang, Yanfeng Wang, and Weidi Xie. Pmcvqa: Visual instruction tuning for medical visual question answering. *arXiv preprint arXiv:2305.10415*,
2023.

[93] Jianwei Zheng, Huimin Chu, Daniele Struppa, Jianming Zhang, Sir Magdi Yacoub, Hesham El-Askary, Anthony Chang, Louis Ehwerhemuepha, Islam Abudayyeh, Alexander Barrett, et al. Optimal multi-stage arrhythmia classification approach. *Scientific reports*, 10(1):2898, 2020.

[94] Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.

## Neurips Paper Checklist 1. **Claims**

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: We explained our method in detail in Sec. 3, then supported each points with extensive experiments in 4. Guidelines:
- The answer NA means that the abstract and introduction do not include the claims made in the paper.

- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. **Limitations**

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: We discussed our limitations on how the reasoning is learned in the conclusion paragraph. Guidelines:
- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.

- The authors are encouraged to create a separate "Limitations" section in their paper.

- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

- The authors should reflect on the factors that influence the performance of the approach.

For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. **Theory Assumptions And Proofs**

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

## Answer: [Na]

Justification: In this work, we describe a novel way of training a reasoning model across heterogeneous domains. We detailed the assumption (that a set of domains must present). However, the effectiveness of the method is primarily proved via experiments, not theoretically. Guidelines:
- The answer NA means that the paper does not include theoretical results. - All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

- All assumptions should be clearly stated or referenced in the statement of any theorems.

- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. **Experimental Result Reproducibility**

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)? Answer: [Yes] Justification: We release our repository containing the code used for all experiments. We also include all the datasets we used. Guidelines:
- The answer NA means that the paper does not include experiments. - If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

- Depending on the contribution, reproducibility can be accomplished in various ways.

For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. **Open Access To Data And Code**

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material? Answer: [Yes] Justification: We open source our training pipeline, model weights and training hyperparameters. The dataset used in our model is fully public, with little to no license restrictions. Guidelines:
- The answer NA means that paper does not include experiments requiring code.

- Please see the NeurIPS code and data submission guidelines (https://nips.cc/
public/guides/CodeSubmissionPolicy) for more details.

- While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https:
//nips.cc/public/guides/CodeSubmissionPolicy) for more details.

- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. **Experimental Setting/Details**

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results? Answer: [Yes] Justification: We describe our training and test details in App. D. Guidelines:
- The answer NA means that the paper does not include experiments.

- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. **Experiment Statistical Significance**

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments? Answer: [Yes] Justification: We did 4 separate runs with different seeds for each experiment in Table 2, and included the standard deviation in Appendix Table 7. Guidelines:
- The answer NA means that the paper does not include experiments.

- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors). - It should be clear whether the error bar is the standard deviation or the standard error of the mean.

- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).

- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. **Experiments Compute Resources**

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments? Answer: [Yes] Justification: We described compute resources in App. D. Guidelines:
- The answer NA means that the paper does not include experiments. - The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. **Code Of Ethics**

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes] Justification: We have reviewed the code of ethics and included a impact statement in A. Guidelines:
- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. - If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. **Broader Impacts**

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed? Answer: [Yes] Justification: We discussed social impacts and included a detailed discussion in the impact statement under App. A. Guidelines:
- The answer NA means that there is no societal impact of the work performed.