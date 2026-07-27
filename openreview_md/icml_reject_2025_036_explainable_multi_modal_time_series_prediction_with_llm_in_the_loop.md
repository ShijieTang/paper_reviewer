# Explainable Multi-Modal Time Series Prediction With Llm-In-The-Loop

## Anonymous Authors1 Abstract

Time series analysis provides essential insights for real-world system dynamics and informs downstream decision-making, yet most existing methods often overlook the rich contextual signals present in auxiliary modalities (e.g., financial news or domain-specific documents). To bridge this gap, we introduce TimeXL, a multi-modal prediction framework that integrates a prototypebased time series encoder with three collaborating Large Language Models (LLMs) to deliver more accurate predictions and interpretable explanations. First, a multi-modal prototype-based encoder processes both time series and textual inputs to generate preliminary forecasts alongside case-based rationales. These outputs then feed into a prediction LLM, which refines the forecasts by reasoning over the encoder's predictions and explanations. Next, a reflection LLM compares the predicted values against the ground truth, identifying textual inconsistencies or noise. Guided by this feedback, a refinement LLM iteratively enhances text quality and triggers encoder retraining. This closed-loop workflow—prediction, critique
(reflect), and refinement—continuously boosts the framework's performance and interpretability.

Empirical evaluations on four real-world datasets demonstrate that TimeXL achieves up to 8.9 % improvement in AUC and produces human-centric, multi-modal explanations, highlighting the power of LLM-driven reasoning for time series prediction.

000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054

## 1. Introduction

In the modern big-data era, time series analysis has become indispensable for understanding real-world system behaviors and guiding downstream decision-making tasks across 1Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>.

1 numerous domains, including healthcare, traffic, finance, and weather (Jin et al., 2018; Guo et al., 2019; Zhang et al., 2017; Qin et al., 2017). Although deep learning models have demonstrated success in capturing complex temporal dependencies (Nie et al., 2023; Deng & Hooi, 2021; Zhang et al.,
2022; Liu et al., 2024d), real-world time series are frequently influenced by external information beyond purely temporal factors. Such additional context, which may come from textual narratives (e.g., finance news (Dong et al., 2024) or medical reports (King et al., 2023)), can offer critical insights for more accurate forecasting and explainability. Recent multi-modal approaches for time series have shown promise by integrating rich contextual signals from disparate data sources—such as textual descriptions—to improve performance on tasks ranging from forecasting and classification to imputation and retrieval (Ekambaram et al., 2020; Niu et al., 2023; Lee et al., 2024; Xing & He, 2023; Moroto et al., 2024; Zhao et al., 2022; Bamford et al., 2023). While these approaches utilize supplementary data to enhance predictive accuracy, they often lack explicit mechanisms to systematically reason and explain about why or how contextual signals affect outcomes. This gap in interpretability poses significant barriers for high-stakes applications such as finance or healthcare, where trust and transparency are paramount.

Meanwhile, Large Language Models (LLMs) (Achiam et al.,
2023; Team et al., 2023; Touvron et al.) have risen to prominence for their remarkable ability to process and reason over textual data across domains, enabling tasks like sentiment analysis, question answering, and content generation in zero- and few-shot settings (Zhang et al., 2024; Kamalloo et al., 2023; Wang et al., 2024c). Their encoded domain knowledge makes them natural candidates for supporting multi-modal time series analyses, where textual context (e.g., news or expert notes) plays a vital role (Liu et al.; Nie et al., 2024; Koa et al., 2024; Wang et al., 2023; Shi et al., 2024; Yu et al., 2023; Singhal et al., 2023). Motivated by these observations, we introduce TimeXL, a novel framework that adopts a closed-loop workflow of prediction, critique (reflect), and refinement, and unifies a prototype-driven time series encoder with LLM-based reasoning to deliver both accurate and interpretable multimodal forecasting (Figure 1). Our approach first employs 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109

Feedback that detects textual noise Multi-modal Time Series (3)
(4)
Refinement LLM
Input Reflection LLM
Enhances text quality and triggers encoder retraining Preliminary prediction 
& case-based rationales
(2)
(1)
Prototype-based Explainable Encoder Output Time series explanations Textual explanations Prediction Prediction from refined rationales, and ground truth Prediction LLM
Figure 1. An overview of the TimeXL workflow. A prototypebased explainable encoder first produces predictions and casebased rationales for both time series and text. The prediction LLM refines forecasts based on these rationales (Step 1). A reflection LLM then critiques the output against ground truth (Step 2), providing feedback to detect textual noise (Step 3). Finally, a refinement LLM updates the text accordingly, triggering encoder retraining for improved accuracy and explanations (Step 4).

a *multi-modal prototype-based encoder* to generate preliminary time series predictions alongside human-readable explanations, leveraging case-based reasoning (Kolodner, 1992; Ming et al., 2019; Ni et al., 2021; Jiang et al., 2023) from both the temporal and textual modalities. These explanations not only justify the encoder's predictions but also serve as auxiliary signals to guide an LLM-powered component that further refines the forecasts and contextual rationales. Unlike conventional methods that merely fuse multi-modal inputs for better accuracy, TimeXL iterates between predictive and refinement phases to mitigate textual noise, fill knowledge gaps, and produce more faithful explanations.

Specifically, a *reflection LLM* diagnoses potential weaknesses by comparing predictions with ground-truth signals, while a *refinement LLM* incorporates these insights to update textual inputs and prototypes iteratively. This feedback loop progressively improves both the predictive and explanatory capabilities of the entire system. Our contributions are summarized as follows:
- We present a prototype-based encoder that combines time series data with textual context, producing transparent, case-based rationales.

- We exploit the interpretative prowess of LLMs to reason over the encoder's outputs and iteratively refine both predictions and text, leading to improved prediction accuracy and explanations.

- Experiments on four real-world benchmarks show that TimeXL consistently outperforms baselines, achieving up to a 8.9% improvement in AUC while providing faithful, human-centric multi-modal explanations.

Overall, TimeXL opens new avenues for explainable multimodal time series analysis by coupling prototype-based inference with LLM-driven reasoning.

## 2. Related Work 2.1. Multi-Modal Time Series Analysis

In recent years, multi-modal time series analysis has gained significant traction in diverse domains such as finance, healthcare, environmental sciences, and industry (Ekambaram et al., 2020; Skenderi et al., 2024; Niu et al., 2023; Yang & Wu, 2021; Zhao et al., 2022; Xing & He, 2023). Multiple approaches have been proposed to model interactions across different modalities for various tasks. For instance, (Lee et al., 2024) introduces a multi-modal augmentation framework for few-shot time series forecasting, which fuses time series and textual representations both at the sample and feature levels using attention. Furthermore, (Bamford et al., 2023) aligns multi-modal time series within a shared latent space of deep encoders and retrieves specific sequences based on textual queries. In addition, (Zheng et al., 2024) performs causal structure learning to uncover root causes in multi-modal time series by separating modality-invariant and modality-specific components via contrastive learning. Most recently, (Liu et al., 2024a) established a multi-modal forecasting benchmark with baselines, and demonstrating performance improvements through the incorporation of a new modality. Although these techniques have advanced predictive performance by leveraging crossmodality interactions, they tend to focus primarily on improving numerical accuracy. The deeper reasoning behind how or why the textual or other contextual signals influence time series outcomes remains underexplored.

## 2.2. Time Series Explanation

Recent studies have explored diverse paradigms for time series interpretability. Gradient-based and perturbation-based
"saliency" methods, for example, highlight important features at different time steps (Ismail et al., 2020; Tonekaboni et al., 2020), while other works explicitly incorporate temporal structures into models and objectives (Leung et al.; Crabbe & Van Der Schaar ´ , 2021). Surrogate approaches also offer global or local explanations, such as applying Shapley values to time series (Bento et al., 2021), enforcing model consistency via self-supervised objectives (Queen et al., 2024), or using information-theoretic strategies for coherent explanations (Liu et al., 2024f). In contrast to saliency or surrogate-based explanations, we adopt a casebased reasoning paradigm (Kolodner, 1992; Ming et al., 2019; Ni et al., 2021; Jiang et al., 2023), which end-to-end generates predictions and built-in explanations from learned prototypes. Our work extends this approach to multi-modal time series by producing human-readable reasoning artifacts for both the temporal and contextual modalities.

## 2.3. Llms For Time Series Analysis

110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139

## 140

141 142 143 144 145 146 147 148 149

## 150

151 152 153 154 155 156 157 158 159 160 161 162 163 164 The rapid development of Large Language Models (LLMs) (Achiam et al., 2023; Team et al., 2023; Touvron et al.) has begun to inspire new directions in time series research (Jiang et al., 2024; Liang et al., 2024). Many existing techniques fine-tune pre-trained LLMs on time series tasks, achieving state-of-the-art results in forecasting, classification, and beyond (Zhou et al., 2023; Bian et al.; Ansari et al., 2024). Often, textual data—such as domain instructions, metadata, or dataset summaries—are encoded as prefix embeddings to enrich time series representations (Jin et al., 2024; Liu et al., 2024b; Jia et al., 2024; Liu et al., 2025). These techniques also contribute to the emergence of time series foundation models (Ansari et al., 2024; Das et al., 2023; Woo et al., 2024; Liu et al., 2024e; Wang et al.,
2024a). An alternative line of research leverages the zeroshot or few-shot reasoning capabilities of LLMs. These methods directly prompt pre-trained language models with text-converted time series (Xue & Salim, 2023) or contextladen prompts representing domain knowledge (Wang et al., 2023; Yu et al., 2023; Singhal et al., 2023), often yielding surprisingly strong performance in real-world scenarios. Furthermore, LLMs can act as knowledge inference modules, synthesizing high-level patterns or explanations that augment standard time series pipelines (Chen et al., 2023b; Shi et al., 2024; Lee et al., 2025; Wang et al., 2024b).

## 3. Methodology 3.1. Problem Statement

In this paper, we consider a multi-modal time series prediction problem. Each instance is represented by the multimodal input (x, s), where x = (x1, x2, · · · , xT ) ∈ R
N×T
denotes time series data with N variables and T historical time steps, and s denotes the corresponding text data describing the real-world context. The text data s can be further divided into L meaningful segments. Based on the historical time series and textual context, our objective is to predict the future outcome y, either as a discrete value for classification tasks, or as a continuous value for regression tasks. In this paper, we mainly consider a classification task while we provide a demonstration of the regression task in Appendix F. There are three major components in the proposed TimeXL framework, a multi-modal prototype encoder Menc that provides initial prediction and case-based explanation, a prediction LLM Mpred that provides prediction based on the understanding of context with explanation, a reflection LLM Mrefl that generates feedback, and a refinement LLM Mrefine that refines the textual context based on the feedback. Below, we introduce each component and how they synergize toward better prediction and explanation.

## 3.2. Multi-Modal Prototype-Based Encoder

We design a multi-modal prototype-based encoder that can generate predictions and explanations across different modalities in an end-to-end manner, as shown in Figure 2. We introduce the model architecture, the learning objectives that yield good explanation properties of prototypes, and the pipeline of case-based explanations using prototypes.

## 3.2.1. Multi-Modal Sequence Modeling With Prototypes

Sequence Encoder. To capture both temporal and semantic dependencies, we adopt separate encoders for time series
(Eθ) and text (Eϕ). For x ∈ R
N×T, the time series encoder Eθ maps the entire sequence into one or multiple representations, which serve as candidates for prototype learning. Simultaneously, the text input s is first transformed by a *frozen* pre-trained language model, PLM (e.g., BERT(Kenton & Toutanova, 2019) or Sentence-BERT(Reimers & Gurevych, 2019)), to produce embeddings es ∈ R
ds×L. These embeddings are then processed by a separate encoder Eϕ to extract meaningful text features. It is worth noting that the choice of Eθ and Eϕ also affects the granularity of explanations. As we will introduce shortly, the prototypes are learned based on sequence representations and are associated with the counterparts in the input space, where the correspondences are determined by the encoders. In this paper, we choose convolution-based encoders for both modalities to capture the fine-grained sub-sequence (*i.e.,* segment) patterns:

$$\mathbf{Z}_{\rm time}=\left(\mathbf{z}_{1},\ldots,\mathbf{z}_{T-w+1}\right)\ =\ \mathbf{\mathcal{E}}_{\theta}(\mathbf{x}),\tag{1}$$ $$\mathbf{Z}_{\rm text}=\left(\mathbf{z}_{1}^{\prime},\ldots,\mathbf{z}_{L-w^{\prime}+1}^{\prime}\right)\ =\ \mathbf{\mathcal{E}}_{\phi}(\mathbf{e}_{s}),\tag{2}$$

where zi ∈ R
hand z
′
j ∈ R
h
′denote segment-level representations learned via convolutional kernels of sizes w and w
′, respectively.

Prototype Allocation. To establish interpretability, we learn a set of *time series prototypes* and *text prototypes* for each class c ∈ {1*, . . . , C*}. Specifically, we introduce:

$$\mathbf{P}_{\mathrm{time}}^{(c)}\in\mathbb{R}^{k\times h},\quad\mathbf{P}_{\mathrm{text}}^{(c)}\in\mathbb{R}^{k^{\prime}\times h^{\prime}},$$

so that each prototype p
(c)
i ∈ R
h(time series) or p
′(c)
i ∈
R 
h
′(text) resides in the same feature space as the relevant encoder outputs. For an input sequence, we measure the In this section, we present the framework for explainable multi-modal time series prediction with LLMs. We first introduce the problem statement. Next, we present the design of a time series encoder that provides prediction and multi-modal explanations as the basis. Finally, we introduce three language agents interacting with the encoder towards better prediction and reasoning results.

$$(6)$$

similarity between each prototype and the most relevant segment in the corresponding modality:
understanding:

$$\begin{array}{r l}{{\mathcal{L}_{c}=}}&{{\sum_{\mathbf{z}_{j}\in\mathbf{Z}_{(\cdot)}}\operatorname*{min}_{\mathbf{p}_{i}\in\mathbf{P}_{(\cdot)}}\|\mathbf{z}_{j}-\mathbf{p}_{i}\|_{2}^{2}\,,}}\\ {{}}&{{}}\\ {{\mathcal{L}_{e}=}}&{{\sum_{\mathbf{p}_{i}\in\mathbf{P}_{(\cdot)}}\operatorname*{min}_{\mathbf{z}_{j}\in\mathbf{Z}_{(\cdot)}}\|\mathbf{p}_{i}-\mathbf{z}_{j}\|_{2}^{2}}}\end{array}$$
$$({\boldsymbol{S}})$$
rie
s
e
r
Training time series Time series prototypes Time Series Text Input Training only Finetune **Frozen**
One time after training Training texts Projection Text Explanation Input: Iron ore spot prices have risen influenced by ... 

Prototype: indicating an increase of iron ore price Input Prototype
+
Multi-modal Prototype-based Encoder Time series Time Series Explanation Text input Fusion Language Model Prediction LLM
Text embedding Reflection LLM
Refinement LLM
Multi-modal Prototype-based Encoder Projection Text prototypes The Framework of TimeXL
Ground Truth

$$\hat{\mathbf{y}}_{\text{enc}}=\text{Softmax}\bigg{(}\mathbf{W}\left[\text{Sim}_{\text{time}}\parallel\text{Sim}_{\text{text}}\right]\bigg{)}\in[0,1]^{C}.\tag{4}$$

165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 Learning Objectives. The learning objectives include three regularization terms that reinforce the interpretability of multi-modal prototypes. In this paper, we focus on a predicting discrete label, where the basic objective is the crossentropy loss for the prediction drawn from multi-modal explainable artifacts LCE =Px,s,yy log(yˆenc) + (1 −
y) log(1 − yˆenc). Besides, we encourage a clustering structure of segments in the representation space by enforcing each segment representation to be adjacent to its closest prototype. Reversely, we regularize each prototype to be as close to a segment representation as possible, to help the prototype locate the most evidencing segment. Both regularization terms are denoted as Lc and Le, respectively, where we omit the modality and class notations for ease of We aggregate similarity scores across all prototypes for each modality, yielding Simtime ∈ R
kC and Simtext ∈ R
k
′C .

Finally, we jointly consider the cross-modal relevance and use a non-negative fusion weight matrix W ∈ R
C×(k+k
′)
that translates these scores into class probabilities:

## 3.2.2. Learning Prototypes Toward Better Explanation

The full objective is written as: L = LCE + λ1Lc + λ2Le + λ3Ld, with hyperparameters λ1, λ2, and λ3 that balance regularization terms towards achieving an optimal and explainable prediction. Prototype Projection. After learning objectives converge, the multi-modal prototypes are well-regularized and reflect good explanation properties. However, these prototypes are still not readily explainable as they are only close to some exemplar segments in the representation space. Therefore, we perform prototype projection to associate each prototype with a training segment from its own class that preserves Le in the representation space, for both time series and text:

$$\mathbf{p}_{i}^{(c)}\leftarrow\operatorname*{arg\,min}_{\mathbf{z}_{j}\in\mathbf{Z}_{(\cdot)}^{(c)}}\left\|\mathbf{p}_{i}^{(c)}-\mathbf{z}_{j}\right\|_{2}^{2},\quad\forall\mathbf{p}_{i}^{(c)}\in\mathbf{P}_{(\cdot)}^{(c)}\tag{7}$$

By associating each prototype with a training segment in the representation space, the multi-modal physical meaning is induced. During testing phase, a multi-modal instance will be compared with prototypes across different modalities to Moreover, we encourage a diverse structure of prototype representations to avoid redundancy and maintain a compact explanation space, by penalizing their similarities via a hinge loss Ld, with a threshold dmin :

Te xt 
S
$${\mathcal{L}}_{d}=\sum_{i=1}\sum_{j\neq i}\operatorname*{max}\left(0,d_{\operatorname*{min}}-\left\|\mathbf{p}_{i}-\mathbf{p}_{j}\right\|_{2}^{2}\right)$$
En
c
(6)
$$\text{Sim}_{i}^{(c)}=\max(\text{Sim}_{i,1}^{(c)},\cdots,\text{Sim}_{i,T-w+1}^{(c)})$$  where $\text{Sim}_{i,j}^{(c)}=\exp\left(-\left\|\boldsymbol{p}_{i}^{(c)}-\boldsymbol{z}_{j}\right\|_{2}^{2}\right)\in[0,1]$ (3)
infer predictions, where the similarity scores, contribution weights, and prototypes' class information assemble the explanation artifacts for reasoning.

## 3.3. Explainable Prediction With Llm-In-The-Loop

To further leverage the reasoning and inference capabilities of LLMs in real-world time series contexts, we propose a framework with three interacting LLM agents: a prediction agent Mpred, a reflection agent Mrefl, and a refinement agent Mrefine. These LLM agents interact with the multi-modal prototype-based encoder Menc toward better prediction accuracy and explainability.

## 3.3.1. Model Synergy For Augmented Prediction

Prediction with Enriched Contexts. The prediction LLM
agent Mpred generates predictions based on the input text s. To improve prediction accuracy, the encoder Menc supplements s with *case-based explanations*. Specifically, Menc selects the ω prototypes that exhibit the highest relevance to any of the textual segments within s. Relevance is determined by the similarity scores used in Equation 3. These selected prototypes are then added to the input prompt of Mpred as explanations, providing richer real-world context and leading to more accurate predictions. The ω prototypesegment pairs, which construct the explanation expls of the input text s, are retrieved as follows:

$\mathbf{expl}_{s}=\left\{\left(\mathbf{p}_{i}^{(c)},\mathbf{s}_{j}\right):(i,j,c)\in\text{Top-}\omega(\text{Sim}_{\text{text}})\right\}$.  
where Top-ω(Simtext) = argTop-ω(*i,j,c*)
$$\operatorname{rgTop}\!\cdot\!\omega_{(i,j,c)}\left(\operatorname{Sim}_{\ i,j}^{\prime(c)}\right)$$
.

Note that *i, j, c* denotes the prototype index, segment index, and class index, respectively. As explscan contain relevant contextual guidance across multiple classes, it augments the input space and removes semantic ambiguity for prediction agent Mpred. Therefore, the prediction is drawn as yˆLLM = Mpred(s, expls). The prompt for querying the prediction agent Mpred is provided in Appendix D, Figure 13.

Fused Predictions. We compile the final prediction based on a fusion of both the multi-modal encoder Menc and prediction LLM Mpred. Specifically, we linearly combine the continuous prediction probabilities yˆenc and discrete prediction yˆLLM: yˆ = αyˆenc + (1 − α)yˆLLM, where α ∈ [0, 1]
is the hyperparameter selected from validation data. The encoder Menc and prediction agent Mpred enhance each other based on their unique strengths. The Menc is fine-tuned based on explicit supervised signals, ensuring accuracy in capturing temporal and contextual dependencies of multimodal time series. On the other hand, Mpred contributes deep semantic understanding drawn from extensive text corpora. By fusing predictions from two distinct perspectives, we achieve a synergistic augmentation toward more accurate 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 Algorithm 1 TimeXL: Explainable Multi-modal Time Series Prediction with LLM Agents Inputs: Multi-modal time series (x, s, y), prototype-based encoder Menc, prediction agent Mpred, reflection agent Mrefl, refinement agent Mrefine, fusion parameter α, max iteration τ , improvement evaluation Eval(·) based on metrics Training:
Initialize s0 = s, i = 0, yˆall = {} while Eval(yˆall, y) not pass or iteration *i < τ* do Train Menc using multi-modal data Di = {(x, si, y), *· · · }*
Infer explainable prediction yˆenc, explsi = Menc(x, si) Infer LLM prediction yˆLLM = Mpred(si, explsi
)
Fuse prediction yˆ = αyˆenc + (1 − α)yˆLLM
Generate reflection Refl = Mrefl(y, yˆLLM, si)
Refine text based on reflection si+1 = Mrefine(Refl, si)
Append yˆ to yˆall Increment i return Menc, Refl, si+1 Validation and Testing: Refinement based on reflection s
′ = Mrefine(Refl, s)
Infer explainable prediction yˆenc, expls
′ = Menc(x, s
′)
Infer LLM prediction yˆLLM = Mpred(s
′, expls
′ )
Fuse prediction yˆ = αyˆenc + (1 − α)yˆLLM
and comprehensive predictions for complex multi-modal time series.

## 3.3.2. Iterative Context Refinement Via Reflective Feedback

$\downarrow$
While the prediction agent Mpred leverages the explainable artifacts to make informed predictions, it is not inherently designed to fit into the context of multi-modal time series data, which could lead to inaccurate predictions when the quality of textual context is inferior. To tackle this issue, we exploit another two language agents Mrefl and Mrefine to generate reflective feedback and refinements on the context, respectively, for better predictive insights.

Given the prediction yˆLLM generated by the prediction agent Mpred, the reflection agent Mrefl aims to understand the reasoning behind the implicit prediction logic of Mpred.

Specifically, it generates a *reflective feedback*, Refl, by analyzing the input text s and its prediction yˆLLM, against the ground truth y, to provide actionable insights for refinement, i.e., Refl = Mrefl(y, yˆLLM, s). Guided by the feedback, the refinement agent Mrefine refines the previous text siinto si+1 by selecting and emphasizing the most relevant content, ensuring that important patterns are appropriately contextualized, which is similar to how a domain expert would perform, i.e., si+1 = Mrefine(Refl, si). The prompts for querying Mrefl and Mrefine are provided in Figures 14, 15 16, 17, and discussed in Appendix D. We finally integrate the refinement via reflection into the optimization loop of our proposed TimeXL, which is summarized in Algorithm 1. Once the textual context is improved, it is used to retrain the multi-modal prototype-based encoder Menc for the next iteration. As such, the explanation (*e.g.,*
quality of the prototypes) and predictive performance of Menc can be improved through this iterative process. Consequently, the prediction agent Mpred could yield better prediction with more informative inputs, further enhancing the accuracy of yˆ. We evaluate the trajectory of predictive performance and terminate the iteration if at least an improvement is observed (Eval(·) pass) when max iteration is reached. Note that, in the testing phase, we use the reflection Refl generated in the best training iteration (evaluated on validation set) to guide Mrefine for context refinement, mimicking how an optimized deep model is applied to testing data.

## 4. Experiments 4.1. Experimental Setup

Datasets. We evaluate methods on four multi-modal time series datasets from three different real-world domains, including weather, finance, and healthcare. The detailed data statistics are summarized in Table 3 of Appendix A.1. The weather dataset contains meteorological reports and the hourly time series records of temperature, humidity, air pressure, wind speed, and wind direction in New York City. The task is to predict if it will rain in the next 24 hours, given the last 24 hours of weather records and summary. The **finance** dataset contains the daily record of the raw material prices together with 14 related indices from January 2017 to July 2024. Given the last 5 business days of stock price data and news, the task is to predict if the target price will exhibit an increasing, decreasing, or neutral trend on the next business day. The **healthcare** datasets contain Test-Positive (TP) and Mortality (MT). The Test-Positive dataset consists of the weekly records and healthcare reports of the number of positive specimens for Influenza A and B. The task is to predict if the percentage of respiratory specimens testing positive in the upcoming week for influenza will exceed the average value, given the records and summary in the last 20 weeks. Similarly, the Mortality dataset contains the weekly records and reports of influenza and pneumonia deaths. The task is to predict if the mortality ratio from influenza and pneumonia will exceed the average value, given the records and summary in the last 20 weeks. Baselines, Evaluation Metrics and Setup We compare TimeXL with state-of-the-art baseline methods for time series prediction. These baselines includes Autoformer (Wu et al., 2021), Dlinear (Zeng et al., 2023), Crossformer (Zhang & Yan, 2023), TimesNet (Wu et al., 2023), PatchTST (Nie et al., 2023), iTransformer (Liu et al., 2024c), FreTS (Yi et al., 2024), TSMixer (Chen et al., 2023a) and LLM-based methods like LLMTime (Gruver et al., 2023), PromptCast (Xue & Salim, 2023), OFA (Zhou et al., 2023),
275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 Time-LLM (Jin et al., 2024) and TimeCMA (Liu et al., 2025), where LLMTime and PromptCast don't need finetuning. While these methods are primarily used for time series prediction with continuous values, they can be easily adapted for discrete value prediction. We also evaluate the multi-modal time series methods. Besides the Time-LLM and TimeCMA where input text is used for embedding reprogramming and alignment, we also evaluate Multi-modal PatchTST and Multi-modal iTransformer from (Liu et al., 2024a), as well as TimeCAP (Lee et al., 2025). We evaluate the discrete prediction via F1 score and AUROC (AUC) score, due to label imbalance in real-world time series datasets. We split all datasets for training/validation/testing by a ratio of 6/2/2. We alternate different embedding methods for texts based on its average length, where we use Bert (Kenton & Toutanova, 2019) as the embedding model for weather and healthcare datasets, and sentence transformer (Reimers & Gurevych, 2019) for finance dataset.

## 4.2. Performance Evaluation

The results of predictive performance are shown in Table 1. It is notable that multi-modal methods generally outperform time series methods across all datasets. These methods include LLM methods (*e.g.,* Time-LLM, TimeCMA) that leverage text embeddings to enhance time series predictions. Moreover, the multi-modal variants (MM-iTransformer and MM-PatchTST) improve the performance of state-of-the-art time series methods, suggesting the benefits of integrating real-world contextual information. Besides, TimeCAP integrates the predictions from both modalities, further improving the predictive performance. TimeXL constantly achieves the highest F1 and AUC scores, consistently surpassing both time series and multi-modal baselines by up to 8.9% of AUC (compared to TimeCAP on Weather dataset). This underscores the advantage of TimeXL, which synergizes multi-modal time series encoder with language agents to enhance interpretability and thus predictive performance in multi-modal time series.

## 4.3. Explainable Multi-Modal Prototypes

Next, we present the explainable multi-modal prototypes rendered by TimeXL, which establishes the case-based reasoning process. Figure 3 shows a subset of time series and text prototypes learned on the weather dataset. The time series prototypes demonstrate the typical temporal patterns aligned with different real-world weather conditions (*i.e.,* rain and not rain). For example, a constant or decreasing humidity at a moderate level, combined with high and steady air pressure, typically indicates a non-rainy scenario. The consistent wind direction is also a sign of mild weather conditions. On the contrary, high humidity, low and fluctuating pressure, along with variable winds typically reveal an unstable weather system ahead. In addition 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384

Multi-modal Prototypes
- suggest a likelihood of dry weather, with stable pressure and - and the wind direction has primarily been from the west - indicating the strengthening of a high-pressure system - drier air moving into the area. Air pressure remained relatively Not Rain Rain
- which could signal the approach of a weather system
- low-pressure system that could lead to worsening weather 
- humidity levels were predominantly high, and wind speeds were - wind direction started westerly, became variable, and 

Datasets →Weather Finance Healthcare (TP) Healthcare **(MT)**

Methods ↓ **F1 AUC F1 AUC F1 AUC F1 AUC** DLinear (Zeng et al., **2023)** 0.540 0.660 0.255 0.485 0.393 0.500 0.419 0.388 Autoformer (Wu et al., **2021)** 0.546 0.590 0.565 0.747 0.774 0.918 0.683 0.825 Crossformer (Zhang & Yan, **2023)** 0.500 0.594 0.571 0.775 0.924 0.984 0.737 0.913 TimesNet (Wu et al., **2023)** 0.494 0.594 0.538 0.756 0.794 0.867 0.765 0.944

iTransformer (Liu et al., **2024c)** 0.541 0.650 0.600 0.783 0.861 0.931 0.791 0.963

TSMixer (Chen et al., **2023a)** 0.488 0.534 0.465 0.689 0.770 0.797 0.808 0.931 FreTS (Yi et al., **2024)** 0.623 0.688 0.546 0.737 0.887 0.950 0.751 0.762 PatchTST (Nie et al., **2023)** 0.592 0.675 0.604 0.795 0.841 0.934 0.695 0.928

LLMTime (Gruver et al., **2023)** 0.587 0.657 0.315 0.498 0.802 0.817 0.769 0.803

PromptCast (Xue & Salim, **2023)** 0.499 0.365 0.418 0.607 0.727 0.768 0.696 0.871 OFA (Zhou et al., **2023)** 0.501 0.606 0.512 0.745 0.774 0.879 0.851 0.977 Time-LLM (Jin et al., **2024)** 0.613 0.699 0.589 0.792 0.671 0.864 0.733 0.912 TimeCMA (Liu et al., **2025)** 0.636 0.731 0.559 0.727 0.729 0.828 0.693 0.843 MM-iTransformer (Liu et al., **2024a)** 0.608 0.689 0.605 0.793 0.926 0.986 0.901 0.990 MM-PatchTST (Liu et al., **2024a)** 0.621 0.718 0.619 **0.812** 0.863 0.968 0.780 0.929

TimeCAP (Lee et al., **2025)** 0.668 0.742 0.611 0.801 0.954 0.983 0.942 0.988

TimeXL 0.696 0.808 0.631 0.797 **0.987 0.996 0.956 0.997**

to time series, the text prototypes also highlight consistent semantic patterns for different weather conditions, such as the channel-specific (e.g., drier air moving into the area, strengthening of high-pressure system) and overall (e.g., a likelihood of dry weather) descriptions of weather activities. In Appendix C.1, we also present more multi-modal prototypes for the weather dataset in Figure 8, for the finance dataset in Figure 9, and for healthcare datasets in Figures 10 and 12. The results validate that TimeXL provides coherent and informative prototypes from the exploitation of time series and its real-world contexts, which facilitates both prediction and explanation.

## 4.4. Multi-Modal Case-Based Reasoning

Building upon the multi-modal prototypes, we present a case study on the testing set of weather data, comparing the original and TimeXL's reasoning processes to highlight its explanatory capability, as shown in Figure 4. In this case, the original text is incorrectly predicted as not rain. We have three key observations: (1) The refinement process filters the original text to emphasize weather conditions more indicative of rain, guided by reflections from training examples.

The refined text preserves the statement on stability while placing more emphasis on humidity and wind as key indicators. (2) Accordingly, the matched segment-prototype pairs from the original text focus more on temperature stability and typical diurnal variations, while the matched pairs in the refined text highlights wind variability, moisture transport, and approaching weather system, aligning more with rain conditions. (3) Furthermore, the reasoning on time series provides a complementary view for assessing weather conditions. The matched time series prototypes identify high humidity and its drop-and-rise trends, wind speed fluctuations and directional shifts, and the declining phase of air pressure fluctuations, all of which are linked to the upcoming rainy conditions. The matched multi-modal prototypes from TimeXL demonstrate its effectiveness in capturing relevant information for both predictive and explanatory analysis. We also provide a case study on finance data in Figure 11, where textual explanations are generated at the granularity of a half-sentence.

## 4.5. Iterative Analysis

To verify the effectiveness of overall workflow with reflection and refinement LLMs as shown in Figure 1, we conduct an iterative analysis of text quality and TimeXL perfor385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439

| Original Text Reasoning   | Truth: Rain   | Prediction: Not rain   |
|---------------------------|---------------|------------------------|
| Original Text: Over the past 24 hours, New York City experienced a gradual increase in temperature throughout the day, peaking in the late afternoon before a modest decline towards the evening. Humidity levels started high in the early hours, dropped significantly during the day, and then showed a rising trend at the end of the period, indicating a possible increase in moisture content in the air. Air pressure remained relatively stable with only minor fluctuations, suggesting a period of stable weather conditions. Wind speeds varied, with a noticeable increase during the late afternoon, which could have contributed to a brief period of cooler and more turbulent conditions. Wind direction was predominantly from the south-southwest, shifting slightly to a more southsoutheast orientation later in the day, which is typical for the region's weather patterns during this time. Prototypes: Not rain: City has experienced relatively stable temperatures with a slight warming trend observed (0.78) Not rain: peak during the late afternoon before beginning to decrease slightly into the (0.64) Not rain: dropping again, indicative of typical diurnal variation (0.51) TimeXL: Text Reasoning Prediction: Rain  Refined Text: Over the past 24 hours, New York City experienced a stable air pressure pattern with minor fluctuations, indicating stable weather conditions. The day saw a gradual increase in temperature, peaking in the late afternoon before declining in the evening. Humidity levels were high early on, dropped significantly during the day, and rose again later, suggesting increased moisture content. Wind direction shifted from south - southwest to south - southeast, bringing moisture-laden air, which could increase the likelihood of rain. Prototypes: Rain: direction was variable without a consistent pattern. These indicators suggest (0.47) Rain: wind direction started westerly, became variable, and (0.64) Rain: which could signal the approach of a weather system (0.53)                           |               |                        |

TimeXL: Time Series Reasoning
Figure 4. Multi-modal case-based reasoning example on weather data. The left part illustrates the reasoning process for both the original and refined text in TimeXL, with matched prototype-input pairs highlighted in the same color along with their similarity scores. The right part presents the time series reasoning in TimeXL, where matched prototypes are overlaid on the time series.

mance, as shown in Figure 5. Specifically, we evaluate the text quality based on its zero-shot predictive accuracy using an LLM. Notably, the text quality benefits from iteration improvements and mostly saturates after one or two iterations. Correspondingly, TimeXL performance quickly improves and stabilizes with very minor fluctuations. These observations underscore how TimeXL alternates between predictive and reflective refinement phases to mitigate textual noise, thus enhancing its predictive capability.

| Ablation         | Variants         | Weather   | Finance   | TP    | MT    |
|------------------|------------------|-----------|-----------|-------|-------|
| Encoder          | Multi-modal      | 0.674     | 0.619     | 0.934 | 0.937 |
| LLM              | Time(PromptCast) | 0.499     | 0.418     | 0.727 | 0.696 |
| Text             | 0.645            | 0.496     | 0.974     | 0.901 |       |
| Text + Prototype | 0.667            | 0.544     | 0.987     | 0.952 |       |
| Fusion           | Select-Best      | 0.674     | 0.619     | 0.987 | 0.952 |
| TimeXL           | 0.696            | 0.631     | 0.987     | 0.956 |       |

## 5. Conclusions

In this paper, we present TimeXL, an explainable multimodal time series prediction framework that synergizes a designed prototype-based encoder with three collaborative LLM agents in the loop (prediction, reflection, and refinement) to deliver more accurate predictions and explanations. Experiments on four multi-modal time series datasets show the advantages of TimeXL over state-of-the-art baselines and its excellent explanation capabilities.

## 4.6. Ablation Studies

In this subsection, we present the component ablations of TimeXL, as shown in Table 2, where we have several observations. Firstly, the performance of prediction LLM with text is better than PromptCast (Xue & Salim, 2023), which highlights the importance of contextual information for LLM in a zero-shot prediction scenario. Furthermore, the text prototypes consistently improve the predictive performance of LLM, underscoring the effectiveness of explainable artifacts from the multi-modal encoder, in terms of providing relevant contextual guidance. In addition, the fusion of prediction LLM and multi-modal encoder further boosts the predictive performance that surpasses the best of both multi-modal encoder and prediction LLM. These observations demonstrate the advantage of our framework synergizing the time series model and LLM for mutually augmented prediction. In Appendix B, full results (F1 and AUC) of TimeXL component ablation are provided in Table 4, and other ablations are provided in Figures 6, 7.

## 6. Impact Statement References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I.,
Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, S. S., Arango, S. P., Kapoor, S., et al. Chronos: Learning the language of time series. *arXiv preprint arXiv:2403.07815*, 2024.

Bamford, T., Coletta, A., Fons, E., Gopalakrishnan, S.,
Vyetrenko, S., Balch, T., and Veloso, M. Multi-modal financial time-series retrieval through latent space projections. In Proceedings of the Fourth ACM International Conference on AI in Finance, pp. 498–506, 2023.

Bento, J., Saleiro, P., Cruz, A. F., Figueiredo, M. A., and Bizarro, P. Timeshap: Explaining recurrent models through sequence perturbations. In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining, pp. 2565–2573, 2021.

Bian, Y., Ju, X., Li, J., Xu, Z., Cheng, D., and Xu, Q. Multipatch prediction: Adapting language models for time series representation learning. In Forty-first International Conference on Machine Learning.

Chen, S.-A., Li, C.-L., Arik, S. O., Yoder, N. C., and Pfister, T. Tsmixer: An all-mlp architecture for time series forecast-ing. *Transactions on Machine Learning* Research, 2023a.

Chen, Z., Zheng, L. N., Lu, C., Yuan, J., and Zhu, D. Chatgpt informed graph neural network for stock movement prediction. *Available at SSRN 4464002*, 2023b.

Crabbe, J. and Van Der Schaar, M. Explaining time series ´
predictions with dynamic masks. In International Conference on Machine Learning, pp. 2166–2177. PMLR,
2021.

Das, A., Kong, W., Sen, R., and Zhou, Y. A decoderonly foundation model for time-series forecasting. arXiv preprint arXiv:2310.10688, 2023.

Deng, A. and Hooi, B. Graph neural network-based anomaly detection in multivariate time series. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pp. 4027–4035, 2021.

Dong, Z., Fan, X., and Peng, Z. Fnspid: A comprehensive financial news dataset in time series. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 4918–4927, 2024.

Ekambaram, V., Manglik, K., Mukherjee, S., Sajja, S. S. K.,
Dwivedi, S., and Raykar, V. Attention based multi-modal new product sales time-series forecasting. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 3110–3118, 2020.

Gruver, N., Finzi, M., Qiu, S., and Wilson, A. G. Large Language Models Are Zero Shot Time Series Forecasters.

In *NeurIPS*, 2023.

Guo, S., Lin, Y., Feng, N., Song, C., and Wan, H. Attention based spatial-temporal graph convolutional networks for traffic flow forecasting. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pp. 922–929, 2019.

Ismail, A. A., Gunady, M., Corrada Bravo, H., and Feizi, S. Benchmarking deep learning interpretability in time series predictions. Advances in neural information processing systems, 33:6441–6452, 2020.

Jia, F., Wang, K., Zheng, Y., Cao, D., and Liu, Y. Gpt4mts:
Prompt-based large language model for multimodal timeseries forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 23343–23351, 2024.

Jiang, Y., Yu, W., Song, D., Wang, L., Cheng, W., and Chen, H. Fedskill: Privacy preserved interpretable skill learning via imitation. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 1010–1019, 2023.

Jiang, Y., Pan, Z., Zhang, X., Garg, S., Schneider, A.,
Nevmyvaka, Y., and Song, D. Empowering time series analysis with large language models: A survey. In Larson, K. (ed.), Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24, pp. 8095–8103. International Joint Conferences on Artificial Intelligence Organization, 8 2024. doi:
10.24963/ijcai.2024/895. URL https://doi.org/
10.24963/ijcai.2024/895. Survey Track.

440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 This work presents significant advancements in explainable multi-modal time series prediction by integrating time series encoders with large language model-based agents. The broader impact of this work is multifaceted. It has the potential to support high-stakes decision-making in domains such as finance and healthcare by delivering more accurate predictions accompanied by reliable case-based explanations that lead to more robust analyses. No ethical concerns must be considered in our work. The social impact is substantial as it provides a new paradigm for analyzing real-world multimodal time series data through the integration of emerging AI tools like language agents. Jin, B., Yang, H., Sun, L., Liu, C., Qu, Y., and Tong, J. A
treatment engine by predicting next-period prescriptions. In *Proceedings of the 24th ACM SIGKDD International* Conference on Knowledge Discovery & Data Mining, pp. 1608–1616, 2018.

Liu, H., Xu, S., Zhao, Z., Kong, L., Kamarthi, H., Sasanur, A. B., Sharma, M., Cui, J., Wen, Q., Zhang, C., et al. Time-mmd: A new multi-domain multimodal dataset for time series analysis. *arXiv preprint arXiv:2406.08627*, 2024a.

495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 Liu, X., McDuff, D., Kovacs, G., Galatzer-Levy, I., Sunshine, J., Zhan, J., Poh, M.-Z., Liao, S., Di Achille, P.,
and Patel, S. Large language models are few-shot health learners.

Jin, M., Wang, S., Ma, L., Chu, Z., Zhang, J. Y., Shi, X.,
Chen, P.-Y., Liang, Y., Li, Y.-F., Pan, S., et al. Timellm: Time series forecasting by reprogramming large language models. In The Twelfth International Conference on Learning Representations, 2024.

Liu, X., Hu, J., Li, Y., Diao, S., Liang, Y., Hooi, B., and Zimmermann, R. Unitime: A language-empowered unified model for cross-domain time series forecasting. In Proceedings of the ACM on Web Conference 2024, pp.

4095–4106, 2024b.

Kamalloo, E., Dziri, N., Clarke, C., and Rafiei, D. Evaluating open-domain question answering in the era of large language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5591–5606, 2023.

Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., and Long, M. itransformer: Inverted transformers are effective for time series forecasting. In The Twelfth International Conference on Learning Representations, 2024c.

Kenton, J. D. M.-W. C. and Toutanova, L. K. Bert: Pretraining of deep bidirectional transformers for language understanding. In *Proceedings of naacL-HLT*, volume 1, pp. 2. Minneapolis, Minnesota, 2019.

Liu, Y., Li, C., Wang, J., and Long, M. Koopa: Learning nonstationary time series dynamics with koopman predictors. Advances in Neural Information Processing Systems, 36, 2024d.

King, R., Yang, T., and Mortazavi, B. J. Multimodal pretraining of medical time series and notes. In *Machine* Learning for Health (ML4H), pp. 244–255. PMLR, 2023.

Liu, Y., Zhang, H., Li, C., Huang, X., Wang, J., and Long, M. Timer: Transformers for time series analysis at scale.

arXiv preprint arXiv:2402.02368, 2024e.

Koa, K. J., Ma, Y., Ng, R., and Chua, T.-S. Learning to generate explainable stock predictions using self-reflective large language models. In Proceedings of the ACM on Web Conference 2024, pp. 4304–4315, 2024.

Liu, Z., Wang, T., Shi, J., Zheng, X., Chen, Z., Song, L., Dong, W., Obeysekera, J., Shirani, F., and Luo, D. Timex++: Learning time-series explanations with information bottleneck. In Forty-first International Conference on Machine Learning, 2024f.

Kolodner, J. L. An introduction to case-based reasoning.

Artificial intelligence review, 6(1):3–34, 1992.

Lee, G., Yu, W., Cheng, W., and Chen, H. Moat: Multimodal augmented time series forecasting. 2024.

Ming, Y., Xu, P., Qu, H., and Ren, L. Interpretable and steerable sequence learning via prototypes. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 903–913, 2019.

Lee, G., Yu, W., Shin, K., Cheng, W., and Chen, H. Timecap:
Learning to contextualize, augment, and predict time series events with large language model agents. In *AAAI*, 2025.

Moroto, Y., Maeda, K., Togo, R., Ogawa, T., and Haseyama, M. Multimodal transformer model using time-series data to classify winter road surface conditions. *Sensors*, 24 (11):3440, 2024.

Leung, K. K., Rooke, C., Smith, J., Zuberi, S., and Volkovs, M. Temporal dependencies in feature importance for time series prediction. In The Eleventh International Conference on Learning Representations.

Ni, J., Chen, Z., Cheng, W., Zong, B., Song, D., Liu, Y.,
Zhang, X., and Chen, H. Interpreting convolutional sequence model by learning local prototypes with adaptation regularization. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management, pp. 1366–1375, 2021.

Liang, Y., Wen, H., Nie, Y., Jiang, Y., Jin, M., Song, D.,
Pan, S., and Wen, Q. Foundation models for time series analysis: A tutorial and survey. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 6555–6565, 2024.

Liu, C., Xu, Q., Miao, H., Yang, S., Zhang, L., Long, C.,
Li, Z., and Zhao, R. Timecma: Towards llm-empowered multivariate time series forecasting via cross-modality alignment. In *AAAI*, 2025.

Nie, Y., Nguyen, N. H., Sinthong, P., and Kalagnanam, J. A
time series is worth 64 words: Long-term forecasting with transformers. In *The Eleventh International Conference* on Learning Representations, 2023.

550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 Nie, Y., Kong, Y., Dong, X., Mulvey, J. M., Poor, H. V., Wen, Q., and Zohren, S. A survey of large language models for financial applications: Progress, prospects and challenges. 2024.

Niu, K., Zhang, K., Peng, X., Pan, Y., and Xiao, N. Deep multi-modal intermediate fusion of clinical record and time series data in mortality prediction. *Frontiers in* Molecular Biosciences, 10:1136071, 2023.

Qin, Y., Song, D., Cheng, H., Cheng, W., Jiang, G., and Cottrell, G. W. A dual-stage attention-based recurrent neural network for time series prediction. In Proceedings of the 26th International Joint Conference on Artificial Intelligence, pp. 2627–2633, 2017.

Queen, O., Hartvigsen, T., Koker, T., He, H., Tsiligkaridis, T., and Zitnik, M. Encoding time-series explanations through self-supervised model behavior consistency. Advances in Neural Information Processing Systems, 36, 2024.

Reimers, N. and Gurevych, I. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 11 2019. URL http://arxiv. org/abs/1908.10084.

Shi, X., Xue, S., Wang, K., Zhou, F., Zhang, J., Zhou, J.,
Tan, C., and Mei, H. Language models can improve event prediction by few-shot abductive reasoning. Advances in Neural Information Processing Systems, 36, 2024.

Singhal, K., Azizi, S., Tu, T., Mahdavi, S. S., Wei, J., Chung, H. W., Scales, N., Tanwani, A., Cole-Lewis, H., Pfohl, S., et al. Publisher correction: Large language models encode clinical knowledge. *Nature*, 620(7973):E19, 2023.

Skenderi, G., Joppi, C., Denitto, M., and Cristani, M. Well googled is half done: Multimodal forecasting of new fashion product sales with image-based google trends. Journal of Forecasting, 43(6):1982–1997, 2024.

Team, G., Anil, R., Borgeaud, S., Wu, Y., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Tonekaboni, S., Joshi, S., Campbell, K., Duvenaud, D. K.,
and Goldenberg, A. What went wrong and when? instance-wise feature importance for time-series blackbox models. Advances in Neural Information Processing Systems, 33:799–809, 2020.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Roziere, B., Goyal, N., Hambro, E., ` Azhar, F., et al. Llama: Open and efficient foundation language models.

Wang, C., Qi, Q., Wang, J., Sun, H., Zhuang, Z., Wu, J.,
Zhang, L., and Liao, J. Chattime: A unified multimodal time series foundation model bridging numerical and textual data. *arXiv preprint arXiv:2412.11376*, 2024a.

Wang, X., Fang, M., Zeng, Z., and Cheng, T. Where would i go next? large language models as human mobility predictors. *arXiv preprint arXiv:2308.15197*, 2023.

Wang, X., Feng, M., Qiu, J., Gu, J., and Zhao, J. From news to forecast: Integrating event analysis in llm-based time series forecasting with reflection. arXiv preprint arXiv:2409.17515, 2024b.

Wang, Z., Duan, Q., Tai, Y.-W., and Tang, C.-K. C3llm:
Conditional multimodal content generation using large language models. *arXiv preprint arXiv:2405.16136*, 2024c.

Woo, G., Liu, C., Kumar, A., Xiong, C., Savarese, S., and Sahoo, D. Unified training of universal time series forecasting transformers. *arXiv preprint arXiv:2402.02592*, 2024.

Wu, H., Xu, J., Wang, J., and Long, M. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. Advances in neural information processing systems, 34:22419–22430, 2021.

Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., and Long, M. Timesnet: Temporal 2d-variation modeling for general time series analysis. In The Eleventh International Conference on Learning Representations, 2023.

Xing, Z. and He, Y. Multi-modal information analysis for fault diagnosis with time-series data from power transformer. *International Journal of Electrical Power &* Energy Systems, 144:108567, 2023.

Xue, H. and Salim, F. D. Promptcast: A new promptbased learning paradigm for time series forecasting. *IEEE* Transactions on Knowledge and Data Engineering, 2023.

Yang, B. and Wu, L. How to leverage the multimodal ehr data for better medical prediction? In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 4029–4038, 2021.

Yi, K., Zhang, Q., Fan, W., Wang, S., Wang, P., He, H.,
An, N., Lian, D., Cao, L., and Niu, Z. Frequency-domain mlps are more effective learners in time series forecasting.

Advances in Neural Information Processing Systems, 36, 2024.

Yu, X., Chen, Z., Ling, Y., Dong, S., Liu, Z., and Lu, Y.

Temporal data meets llm–explainable financial time series forecasting. Technical report, 2023.

Zeng, A., Chen, M., Zhang, L., and Xu, Q. Are transformers effective for time series forecasting? In Proceedings of the AAAI conference on artificial intelligence, volume 37, pp. 11121–11128, 2023.

Zhang, L., Aggarwal, C., and Qi, G.-J. Stock price prediction via discovering multi-frequency trading patterns. In Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining, pp. 2141–2149, 2017.

Zhang, W., Deng, Y., Liu, B., Pan, S., and Bing, L. Sentiment analysis in the era of large language models: A reality check. In Findings of the Association for Computational Linguistics: NAACL 2024, pp. 3881–3906, 2024.

605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659 Zhang, X., Zhao, Z., Tsiligkaridis, T., and Zitnik, M. Selfsupervised contrastive pre-training for time series via time-frequency consistency. Advances in Neural Information Processing Systems, 35:3988–4003, 2022.

Zhang, Y. and Yan, J. Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting. In The eleventh international conference on learning representations, 2023.

Zhao, X., Jia, K., Letcher, B., Fair, J., Xie, Y., and Jia, X. Vimts: Variational-based imputation for multi-modal time series. In 2022 IEEE International Conference on Big Data (Big Data), pp. 349–358. IEEE, 2022.

Zheng, L., Chen, Z., He, J., and Chen, H. Mulan: Multimodal causal structure learning and root cause analysis for microservice systems. In Proceedings of the ACM on Web Conference 2024, pp. 4107–4116, 2024.

Zhou, T., Niu, P., Sun, L., Jin, R., et al. One fits all: Power general time series analysis by pretrained lm. Advances in neural information processing systems, 36:43322–43355, 2023.

## A. Experimental Settings A.1. Dataset Statistics

660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714 In this subsection, we provide more details of the real-world datasets we used for the experiments. The data statistics are summarized in Table 3, including the meta information (*e.g.,* domain resolution, duration of real-world time series records), the number of channels and timesteps and so on. We used the weather and healthcare datasets in TimeCAP (Lee et al., 2025), and the finance dataset in (Lee et al., 2024). The **Weather** dataset contains the hourly time series record of temperature, humidity, air pressure, wind speed, and wind direction1, and related weather summaries in New York City from October 2012 to November 2017. The task is to predict if it will rain in the next 24 hours, given the last 24 hours of weather records and summary. The **Finance** dataset contains the daily record of the raw material prices together with 14 related indices ranging from January 2017 to July 20242, with news articles gathered from S&P Global Commodity Insights. The task is to predict if future prices will increase by more than 1%, decrease by more than 1%, or exhibit a neutral trend on the next business day, given the last 5 business days of stock price data and news.

The healthcare datasets are related to testing cases and deaths of influenza3. The **Healthcare (Test-Positive)** dataset consists of the weekly records of the number of positive specimens for Influenza A and B, and related healthcare reports. The task is to predict if the percentage of respiratory specimens testing positive in the upcoming week for influenza will exceed the average value, given the records and summary in the last 20 weeks. Similarly, the **Healthcare (Mortality)** dataset contains the weekly records and healthcare reports of influenza and pneumonia deaths. The task is to predict if the mortality ratio from influenza and pneumonia will exceed the average value, given the records and summary in the last 20 weeks.

## A.2. Hyperparameters

| Table 3. Summary of dataset statistics.   |               |            |            |             |                   |                                               |
|-------------------------------------------|---------------|------------|------------|-------------|-------------------|-----------------------------------------------|
| Domain                                    | Dataset       | Resolution | # Channels | # Timesteps | Duration          | Ground Truth Distribution                     |
| Weather                                   | New York      | Hourly     | 5          | 45,216      | 2012.10 - 2017.11 | Rain (24.26%) / Not rain (75.74%)             |
| Finance                                   | Raw Material  | Daily      | 15         | 1,876       | 2012.09 - 2022.02 | Inc. (36.7%) / Dec. (34.1%) / Neutral (29.2%) |
| Healthcare                                | Test-Positive | Weekly     | 6          | 447         | 2015.10 - 2024.04 | Not exceed (65.77%) / Exceed (34.23%)         |
| Healthcare                                | Mortality     | Weekly     | 4          | 395         | 2016.07 - 2024.06 | Not exceed (69.33%) / Exceed (30.67%)         |

First, we provide the hyperparameters of baseline methods. Unless otherwise specified, we used the default hyperparameters from the Time Series Library (Wu et al., 2023). For LLMTime (Gruver et al., 2023), OFA (Zhou et al., 2023), Time-LLM (Jin et al., 2024), TimeCMA (Liu et al., 2025), TimeCAP (Lee et al., 2025), we use their own implementations. For all methods, the dropout rate ∈ {0.0, 0.1, 0.2}, learning rate ∈ {0.0001, 0.0003, 0.001}. For transformer-based and LLM fine-tuning methods (Wu et al., 2021; Zhang & Yan, 2023; Liu et al., 2024c; Nie et al., 2023; Liu et al., 2025; Jin et al., 2024), the number of attention layers ∈ {1, 2}, the number of attention heads ∈ {4, 8, 16}. For Dlinear (Zeng et al., 2023), moving average ∈ {3, 5}. For TimesNet (Wu et al., 2023) the number of layers ∈ {1, 2}. For PatchTST and MM-PatchTST, the patch size ∈ {3, 5} for the finance dataset. Next, we provide the hyperparameters of TimeXL. The numbers of time series prototypes and text prototypes are k ∈ {5, 10, 15, 20} and k
′ ∈ [5, 10], respectively. The hyperparameters controlling regularization strengths are λ1, λ2, λ3 ∈
[0.1, 0.3] with interval 0.05 for individual modality, dmin ∈ {1.0, 1.5, 2.0} for time series, dmin ∈ {3.0, 3.5, 4.0} for text.

Learning rate for multi-modal encoder ∈ {0.0001, 0.0003, 0.001} The number of case-based explanations fed to prediction LLM ω ∈ {3, 5, 8, 10}.

1https://www.kaggle.com/datasets/selfishgene/historical-hourly-weather-data 2https://www.indexmundi.com/commodities 3https://www.cdc.gov/fluview/overview/index.html

## A.3. Large Language Model

We employed the gpt-4o-2024-08-06 version for GPT-4o in OpenAI API. We use the parameters max tokens=2048, top p=1, and temperature=0.7 for content generation (self-reflection and text refinement), and 0.3 for prediction.

## A.4. Environment

We conducted all the experiments on a TensorEX server with 2 Intel Xeon Gold 5218R Processor (each with 20 Core), 512GB memory, and 4 RTX A6000 GPUs (each with 48 GB memory).

## B. More Ablation Studies

Here we present the full results of TimeXL component ablations in Table 4. In addition to F1 scores, the results of AUC scores consistently demonstrate the importance of contextual information, the effectiveness of prototype from the multi-modal encoder, as well as the advantage of prediction fusion.

We also provide an ablation study on the learning objectives in the TimeXL encoder. The results clearly show that the full objective consistently achieves the best encoder prediction performance, highlighting the necessity of regularization terms that enhance the interpretability of multi-modal prototypes. The clustering (λ1) and evidencing (λ2) objectives also play a crucial role in accurate prediction: the clustering term ensures distinguishable prototypes across different classes, while the evidencing term ensures accurate projection onto training data. Moreover, we assess how the number of matched case-based explanations enhances the prediction LLM, as shown in Figure 7. We conduct experiments on weather and finance datasets, demonstrating that incorporating more relevant casebased explanations consistently improves prediction performance. This further highlights the effectiveness of explainable artifacts in providing meaningful contextual guidance.

715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769

Weather Performance Finance Performance Full objective Time 
( 1 = 0)
Time 
( 2 = 0)
Time 
( 3 = 0)Text 
( 1 = 0)Text
( 2 = 0)Text 
( 3 = 0)
0.50 0.55 0.60 0.65 0.70 0.75 0.80 F1 AUC
F1 AUC
Full objective Time 
( 1 = 0)
Time 
( 2 = 0)
Time 
( 3 = 0)Text 
( 1 = 0)Text
( 2 = 0)Text 
( 3 = 0)
0.600 0.625 0.650 0.675 0.700 0.725 0.750 0.775 S
co re S
co re Healthcare (TP) Performance F1 AUC
Healthcare (MT) Performance F1 AUC
Full objective Time 
( 1 = 0)
Time 
( 2 = 0)
Time 
( 3 = 0)Text 
( 1 = 0)Text
( 2 = 0)Text 
( 3 = 0)
0.800 0.825 0.850 0.875 0.900 0.925 0.950 0.975 1.000 Full objective Time 
( 1 = 0)
Time 
( 2 = 0)
Time 
( 3 = 0)Text 
( 1 = 0)Text
( 2 = 0)Text 
( 3 = 0)
0.86 0.88 0.90 0.92 0.94 0.96 0.98 Sco re Sco re

## C. Explainable Multi-Modal Prototypes And Case Study C.1. Multi-Modal Prototypes For All Datasets

We present the learned multi-modal prototypes across all datasets, including Weather (Figure 8), Finance (Figure 9), Healthcare (Test-Positive) (Figure 10), and Healthcare (Mortality) (Figure 12). It is noticeable that the prototypes from both modalities align well with real-world ground truth scenarios, ensuring faithful explanations and enhancing LLM predictions.

## C.2. Case-Based Reasoning Example On Finance

770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 We provide another case-based reasoning example to demonstrate the effectiveness of TimeXL in explanatory analysis, as shown in Figure 11. In this example, the original text is incorrectly predicted as neutral instead of a decreasing trend of iron ore stock price. We have a few key observations based on the results. First, the refinement LLM filters the original text to emphasize economic and market conditions more indicative of a declining trend, based on the reflections from training examples. The refined text preserves discussions on port inventories and steel margins while placing more emphasis on subdued demand, thin profit margins, and bearish market sentiment as key indicator of prediction. Accordingly, the case-based explanations from the original text focus more on inventory management and short-term stable patterns, while those in the refined text highlight demand contraction, production constraints, and macroeconomic uncertainty, which is more consistent with a decreasing trend. Furthermore, the reasoning on time series provides a complementary view for predicting iron ore price trends. The time series explanations identify declining price movements across multiple indices. In general, the multi-modal explanations based on matched prototypes from TimeXL demonstrate its effectiveness in capturing relevant iron ore market condition for both predictive and explanatory analysis.

0.724 0.722 0.724 0.7320.735 0.739 Weather 0.627 0.635 0.6460.655 0.659 0.662 Finance 0.66 0.68 0.70 0.72 0.74 0 2 4 6 8 10 Number of Case-Based Explanations 0.500 0.525 0.550 0.575 0.600 0.625 0.650 Sco re F1 AUC
F1 AUC
0.496 0.506 0.520 0.5320.541 0.544 0.645 0.648 0.649 0.6560.661 0.667 0 2 4 6 8 10 Number of Case-Based Explanations

Ablation ↓ Variants Weather Finance Healthcare (Test-Positive) Healthcare **(Mortality)**

F1 AUC F1 AUC F1 AUC F1 AUC

Encoder Multi-modal 0.674 0.767 0.619 0.791 0.934 0.974 0.937 0.988

LLM Time(PromptCast) 0.499 0.365 0.418 0.607 0.727 0.768 0.696 0.871

Text 0.645 0.724 0.496 0.627 0.974 0.967 0.901 0.969

Text + Prototype 0.667 0.739 0.544 0.662 0.987 0.983 0.952 0.976

Fusion Select-Best 0.674 0.767 0.619 0.791 0.987 0.983 0.952 0.988

TimeXL **0.696 0.808 0.631 0.797 0.987 0.996 0.956 0.997**

Not Rain Multi-modal Prototypes Rain
- suggest a likelihood of dry weather, with stable pressure and - westerly shifts, which typically bring cooler, drier air - and the wind direction has primarily been from the west
- indicating the strengthening of a high-pressure system
- drier air moving into the area. Air pressure remained relatively - a notable dip followed by a gradual return to higher levels - stable with a slight downward trend. Wind speeds were mostly - hours in New York City, air pressure remained relatively stable
- which could signal the approach of a weather system - low-pressure system that could lead to worsening weather - humidity levels were predominantly high, and wind speeds were - recent hours. Air pressure has been decreasing overall, which - wind direction started westerly, became variable, and - brief increase, and wind direction was predominantly north and northeast - recent hours. Air pressure has been decreasing overall, which
- direction was variable without a consistent pattern. These indicators suggest 
Figure 8. More multi-modal prototypes learned from Weather dataset. Each row in the figure represents a time series prototype.

825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 Figure 11. Multi-modal case-based reasoning example on Finance dataset.

| Original Text Reasoning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Truth: Decrease   | Prediction: Neutral   |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|-----------------------|
| Original Text:... Recovering global demand has led to higher Asia-Pacific metallurgical coal prices, further indicating a resurgence in industrial activity as blast furnaces resume operation globally. Additionally port inventories are being carefully managed due to import quotas and vessel inflows, influencing the broader supply chain and price dynamics These dynamics are compounded by ongoing production controls and concerns about steel margins, which may face further stress in response to weak demand in the construction market and other key industries. Additionally upcoming economic stimulus measures and potential supply chain disruptions remain critical factors to monitor, as they could significantly influence future prices and production levels of iron ore. Asian iron ore prices continue to decline due to persistent bearish sentiment, influencing buyers' restocking interest, which appears to be returning Some mills are transitioning from using low-grade fines to mid-grade fines due to cost inefficiencies in sintering, notably among smaller blast furnaces… In particular the real estate and construction sectors continue to be key influencers of steel demand and, consequently iron ore prices Additionally economic stimulus measures aimed at boosting industries could lead to increased demand for iron ore. However geopolitical tensions and economic policies, particularly those related to trade, could create volatility in iron ore prices. Overall many factors including mill margins, port inventories and steel production levels will be critical in shaping the iron ore market's trajectory in the near future . Prototypes: Increase: while domestic supply constraints continue due to delays in reactivating mines. On the logistical side the freight rates are climbing due to increased demand for Capesize vessels (0.21) Neutral : potentially depressing iron ore prices again. Thus despite short-term strength in iron ore prices driven by active steel mill production and disruptions in scrap supply (0.39) Decrease: Iron ore prices have been declining due to the weak steel market, leading to dampened buying interest (0.33) Neutral: Mills are capitalizing on already high stock levels, buying less iron ore primarily using cheaper port stocks (0.27) TimeXL: Text Reasoning Prediction: Decrease  Refined Text: ... Despite this, global demand recovery has bolstered Asia-Pacific metallurgical coal prices, signaling a resurgence in industrial activity as blast furnaces globally restart operations. Port inventories are under careful management due to import quotas and vessel inflows, affecting supply chain dynamics. There is a noted contraction in the price spread between different grade fines, with a shift towards lower-grade iron ore to manage costs, particularly as medium-grade fines command higher premiums. Asian iron ore prices have decreased due to subdued steel demand and thin margins, deterring buyers from fixed price options. Production controls in Tangshan have further reduced demand, contributing to bearish market sentiment. Seasonal factors, such as the approaching winter heating season, are expected to tighten sintering controls, indirectly impacting iron ore demand dynamics. Meanwhile, geopolitical tensions and economic policies, especially those related to trade, could introduce volatility into iron ore prices. Economic stimulus measures aimed at boosting industrial activity could lead to increased demand for iron ore. Overall, mill margins, port inventories, and steel production levels will be critical in determining the future trajectory of the iron ore market. Prototypes: Decrease: The iron ore market is experiencing a downturn (0.43) Decrease: Overall the combination of weak demand (0.32) Decrease: causing cautious purchasing behavior (0.49) Decrease: are expected to persist potentially capping steel production and influencing iron ore demand (0.54) |                   |                       |

TimeXL: Time Series Reasoning
880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934

Increase
- Overall the combination of weak demand - are expected to persist potentially capping steel production and influencing iron ore demand - causing cautious purchasing behavior - The iron ore market is experiencing a downturn - Blast furnace operations are affected by stricter environmental regulations

Decrease
- indicating balanced raw material needs - with notable volatility driven by various factors - Geopolitical and seasonal factors - Indian iron ore supply might increase due to eased export restrictions. - A proposed weekly spot CFR China blast furnace iron ore pellet premium assessment by Platts aims to enhance pricing transparency Neutral
- supported by low steel mill inventories and limited spot offers - In conclusion while recent restocking activities provide some support - The seaborne iron ore market is experiencing upward price movements - Anticipated economic stimulus measures in China - Intensified restocking activities ahead of the Lunar New Year have led to increased demand and price

Multi-modal Prototypes
Figure 9. Key multi-modal prototypes learned from Finance dataset. Each row in the figure represents a time series prototype.

935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989

## D. Designed Prompts For Experiments

Exceed Multi-modal Prototypes Not Exceed
- The week - over - week growth in positive specimens for both Influenza A and B suggests a significant impact on the
- has been a significant upward trend in both the number of respiratory specimens tested for influenza and the number of positive cases for
- season but continued risk due to the persistent presence of Influenza B. The healthcare system should maintain vigilance and resources
- been a notable increase in the percentage of respiratory specimens testing positive for influenza, with the average ratio of positive specimens rising
- trend suggests a waning influenza season, which could potentially reduce the burden on healthcare resources related to flu treatment and hospital
- and 0. 63 %, respectively. This declining trend suggests decreased influenza activity, potentially easing healthcare resource allocation.

- in the number of specimens tested, reaching a peak at week 6, followed by a decline, which suggests that the in
- with Influenza A becoming more prevalent later. In recent weeks, there has been a notable decline, with the overall
Figure 10. Key multi-modal prototypes learned from Healthcare (Test-positive) dataset. Each row in the figure represents a time series prototype.

Exceed Multi-modal Prototypes Not Exceed
- necessitating close monitoring and potentially increased healthcare resources. While Influenza seems less concerning, Pneumonia remains a
- suggests a seasonal pattern or an emerging health threat. The healthcare system may need to prepare for increased demand for medical care and
- healthcare resources. This data underscores the urgent need for targeted healthcare interventions to manage and mitigate further mortality.

- resources and may necessitate increased preventive measures, vaccination efforts, and allocation of medical care to manage
- This escalation suggests a growing burden on the healthcare system and underscores the need for vigilant surveillance and
- significant reduction in influenza or pneumonia - related mortality, indicating an effective response possibly through vaccination efforts or public
- persistent presence of influenza - related deaths, despite being low, requires continued surveillance to manage and prevent potential outbreaks.

- This data could reflect seasonal patterns, the effectiveness of public health interventions, or changes in the prevalence of these infections, which
- These trends suggest a potential reduction in healthcare system burdens but require continued surveillance for potential reversals or seasonal variations.

- is particularly noteworthy, suggesting successful containment or a natural decline of influenza cases, which could alleviate some pressure on healthcare
In this section, we provide our prompts for prediction LLM in Figure 13 (and a text-only variant for comparison, in Figure 18), reflection LLM in Figures 14, 15 16, as well as refinement LLM in Figure 17. Note that we adopt a generateupdate-summarize strategy to effectively capture the reflective thoughts from training samples with class imbalances, which is more structured and scalable. We make the whole training texts into batches. First, the reflection LLM generates the initial reflection (Figure 14) by extracting key insights from class-specific summaries, highlighting text patterns that contribute to correct and incorrect predictions. Next, it updates the reflection (Figure 15) by incorporating new training data, ensuring incremental and context-aware refinements. Finally, it summarizes multiple reflections from each class (in Figure 16) into a comprehensive guideline for downstream text refinement. This strategy consolidates knowledge from correct predictions while learning from incorrect ones, akin to the training process of deep models.

990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 Figure 14. Prompt for reflection LLM: reflection generation User Prompt

| Your task is to [task description]. First, review the following [number of prototypes] prototype text segments and outcomes, so that you can refer to when making predictions. Prototype #1: [text prototype] Corresponding Segment#1: [input text segment] Relevance Score: [similarity score] Outcome #1: [options] … Next, review the [situation] : Summary: [text input] Based on your understanding, predict the outcome of [situation]. Respond your prediction with [options]. Response should not include other terms.   |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

Figure 13. Prompt for prediction LLM
System Prompt You are an advanced reasoning agent that can improve the quality of [domain] summary based on self reflection. You will be given the summaries and [correct flag] predictions of [situation]. Your task is to learn some reflections that guides the refinement of [domain] summaries.

User Prompt

| Your task is to analyze the provided [domain] summaries with [correct flag] predictions, in order to generate a reflection report improving its quality for [situation] prediction. Review the following [number of summaries] [domain] summaries with [ground truth] actual outcomes and [prediction] predictions. Summary #1: [text input] Actual Outcome #1: [ground truth] Prediction #1: [prediction] … Based on your analysis, write a high-quality reflection report that summarizes key phrases or sentences that led to correct predictions of [situation] / commonly misinterpreted and overlooked phrases or sentences that led to incorrect predictions of [situation]. Use precise terms to convey a clear and professional analysis, and avoid overly general statements. The report should be a comprehensive and informative paragraph, which can be generalized to refine similar [domain] summaries. Your response should not include other terms.   |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

System Prompt Your job is to act as [specific role]. You will be given a summary of [data description] and related prototypes that you can refer to. Based on this information, your task is to predict [task description].

1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079 1080 1081

1082

1083 1084 1085 1086 1087 1088 1089 1090 1091 1092 1093 1094 1095 1096 1097 1098 1099

| System Prompt You are an advanced reasoning agent that can improve the quality of [domain] summary based on self reflection. You will receive a reflection report up to this point. You will also be given the summaries and [correct flag] predictions of [situation]. Your task is to learn some reflections and update the current report that guides the refinement of [domain] summaries. User Prompt Your task is to analyze the provided [domain] summaries with [correct flag] predictions, in order to update a reflection report improving its quality for [situation] prediction. First, review the following reflection report up to this point: [current reflection report] Next, review the following [number of summaries] [domain] summaries with [ground truth] actual outcomes and [prediction] predictions. Summary #1: [text input] Actual Outcome #1: [ground truth] Prediction #1: [prediction] … Based on your analysis, write a high-quality reflection report that summarizes key phrases or sentences that led to correct predictions of [situation] / commonly misinterpreted and overlooked phrases or sentences that led to incorrect predictions of [situation]. Use precise terms to convey a clear and professional analysis, and avoid overly general statements. The report should contain incremental and context-aware updates, and can be generalized to refine similar [domain] summaries. Your response should not include other terms.   |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

Figure 15. Prompt for reflection LLM: reflection update Figure 16. Prompt for reflection LLM: reflection summarization

| System Prompt You are an advanced summarization agent that can generate high-quality summarization. You will be given previously generated reflections for text refinement, from the correct and incorrect predictions of [domain] texts. Your current task is to summarize these long reflections to better guide financial text refinement. User Prompt Your task is to summarize the long reflections derived from previous predictions of [domain] contents. The goal is to generate a high-quality report aimed at improving the [domain] text quality for better predictive accuracy. First, review the reflections from all combinations of possible predictions and actual outcomes: [reflection reports] Based on your analysis, summarize the reflections of different scenarios and write a comprehensive report that provides guidelines to select the most important content in new [domain] texts where the actual outcome is unknown. Your response should keep the enough details, yet effective, to improve the text quality for downstream prediction. Your response should not include other terms.   |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|