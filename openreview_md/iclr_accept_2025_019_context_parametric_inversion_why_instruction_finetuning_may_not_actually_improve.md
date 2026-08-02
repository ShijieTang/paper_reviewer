# Context-Parametric Inversion: Why Instruction Finetuning Can Worsen Context Reliance

Sachin Goyal∗† Christina Baek∗† J. Zico Kolter† **Aditi Raghunathan**†
Carnegie Mellon University†
{sachingo,kbaek,zkolter,raditi}@cs.cmu.edu

## Abstract

A standard practice when using large language models is for users to supplement their instruction with an input context containing new information for the model to process. However, models struggle to reliably follow the input context, especially when it conflicts with their parametric knowledge from pretraining. In-principle, one would expect models to adapt to the user context better after instruction finetuning, particularly when handling knowledge conflicts. However, we observe a surprising failure mode: during instruction tuning, the context reliance under knowledge conflicts initially increases as expected, but then *gradually decreases* as instruction finetuning progresses. This happens while the performance on standard benchmarks keeps on increasing far after this drop. We call this phenomenon context-parametric inversion and observe it across multiple general purpose instruction tuning datasets such as TULU, Alpaca and Ultrachat, across different model families like Llama, Mistral, and Pythia. We perform various controlled studies and theoretical analysis to show that context-parametric inversion occurs due to examples in the instruction finetuning data where the input context provides information that aligns with model's parametric knowledge. Our analysis suggests some natural mitigation strategies with limited but insightful gains, and serves as a useful starting point in addressing this deficiency in instruction finetuning.

## 1 Introduction

Large language models (LLMs) are widely used for a variety of tasks, many of which require carefully balancing the knowledge embedded in their parameters with the information provided through the input context. A persistent challenge, however, is their tendency to overrely on parametric knowledge, even when it contradicts with the context. This overreliance hinders the ability to update model facts with augmented contexts and reliably follow atypical user instructions (Qiu et al., 2023; Adlakha et al., 2024). This tension between contextual and parametric knowledge has been commonly studied under the moniker of *knowledge conflicts*. Existing works explore various decoding and finetuning remedies (Shi et al., 2023; Yuan et al., 2024; Longpre et al., 2022; Chen et al., 2022), but model behavior under knowledge conflicts remain difficult to control, and conflicts often occur more frequently scale (McKenzie et al., 2024). Moreover, we have limited understanding of the underlying dynamics that drive models to ignore the context and rely heavily on its parametric knowledge. In this work, we study the effect of instruction finetuning (IFT)—a staple part of the LLM pipelineon the ability to override pretrained knowledge through the context. IFT seeks to enhance the model's ability to assist with user queries. Oftentimes, these instructions contains a context with critical information needed to complete the task. For instance, an instruction "What is the total price of my trip to Hawaii?" operates on a context "Context: [Itinerary List]", and an instruction "Rank these famous soccer players based on these scores" could contain a context like: "[Scores Table]." In these circumstances, instruction tuned models must appropriately leverage the input context to respond, instead of relying on parametric knowledge. However, we make an intriguing observation during IFT, where in the presence of knowledge conflicts, the model's reliance on context initially increases as expected but surprisingly starts decreasing.

∗Equal Contribution.

1

Co u nte rf a ctu al C
o n te x t Re li a n c e Context-Parametric Inversion with Instruction Finetuning on TULU
Llama2-7B Pythia-6.9B Mistral-7B

 Non-Context Critical Context:
Einstein loved playing violin in free time.

Query:
What instrument did Einstein play 0.30 0.35 0.40 0.45 0.50 0.55 Standard Benchmarks Performance 0.3 0.4 0.5 0.6 0.7 0.8 1 2 Context Critical Context:
Emmy loved playing violin in free time Query:
What instrument did Emmy play High Perplexity Low Perplexity
(b)
(a)
We measure the context reliance by designing inputs contexts that suggest a fictional answer to a user query different from facts in the pretraining corpus (§ 3.2). We evaluate context reliance across the IFT trajectory of multiple instruction datasets —TULU, Alpaca or UltraChat - and multiple model families - Llama, Pythia and Mistral. Across these settings, we see that context reliance initially increases and then decreases, a phenomenon we call **context-parametric inversion**. In fact, this drop begins in early timesteps of IFT, while the performance on standard benchmarks (e.g., MMLU, GSM8k, SQuAD) keeps on increasing far after this drop. For example, as shown in Figure 1a, the context reliance of Llama2-7B (as measured on knowledge conflict datasets (§ 3.2)) increases from 30% to 60% initially with IFT. However, it start dropping as the finetuning progresses further, dipping to around 35%. Why do we observe context-parametric inversion with instruction tuning? The initial increase is expected, as a nontrivial subset of instruction tuning datasets often require models to use the context to respond correctly. We perform controlled experiments to understand the subsequent detrimental decrease. First, we observe that context-reliance drops outside facts beyond those seen during IFT. Second, common instruction tuning datasets typically contain some datapoints that are purely about recall of pretrained knowledge, and do not involve context-dependent instructions. Could the drop be attributed to the presence of such points? We curate the datasets to only include context-dependent points but *still* see a drop in context reliance after an initial increase. We analyze this phenomenon theoretically in a one-layer tranformer and uncover the optimization dynamic that explains context-parametric inversion. We can partition a generic dataset containing context-dependent datapoints into two categories: (i) *context-critical* datapoints where context provides key information needed to answer a user query that the model does not know beforehand (Fig. 1b), and (ii) *non-context-critical* datapoints where the context is approximately redundant with model's parametric knowledge (§ 4.3). In the early stages of training, context-critical points tend to have higher loss and therefore dominate the gradient signal, driving the model to focus on the context. However, as training progresses, the loss on context-critical points decreases, and the noncontext-critical points dominate the gradient. We show that the gradient updates then tend to hedge, reverting back to using the parametric knowledge, thus reducing the context reliance.

Finally, our analysis naturally leads us to some mitigation strategies by data curation, data augmentation, and regularization. These strategies are able to partially alleviate the drop in deep networks on real-world datasets, showing that our theoretical insights do translate to practical settings. However, as we discuss in § 6, these mitigation strategies each have fundamental limitations and tradeoffs. Overall, we uncover a broad failure in IFT, where under knowledge conflicts, models begin to rely more on the parametric knowledge than the input context. To the best of our knowledge, we are the first to identify this deficiency with instruction tuning. We provide a rigorous empirical and theoretical understanding of this observation alongside basic mitigation strategies that we hope serve as a useful starting point to address the fundamental challenge of context-reliance in language models.

## 2 Related Works

Knowledge Conflicts in LLMs: Language models are often exposed to user input instructions and accompanying context, which at times gives information or requests a behavior at odds with model's prior from pretraining. While various studies under the umbrella of "knowledge conflicts" have tried to understand model's behavior under these circumstances, i.e. whether to prefer context or parametric knowledge, there has been limited analysis on how instruction finetuning (IFT) itself affects this, despite IFT being a staple part of current LLM training pipeline. Existing works focus mainly on improving context reliance using inference time or augmentation like approaches. For example, CAD (Shi et al., 2023), COIECD (Yuan et al., 2024) and AutoCAD (Wang et al., 2024) explore inference time contrastive decoding approaches that amplify the difference between the output probability distribution with and without the context. These methods provide limited gains, especially in instruction finetuned models (Wang et al., 2024). Zhou et al. (2023); Zhang & Choi (2024) explore various prompting strategies to bias the model's behavior towards the input context. Jin et al. (2024b) tries to build a mechanistic interpretation. On the other hand, Longpre et al. (2022); Fang et al. (2024); Neeman et al. (2022); Li et al. (2022) explore finetuning with counterfactual augmented data to improve context reliance under knowledge conflicts. However, in § 6, we show that counterfactual data augmentation cannot fix all types of context-parametric conflicts (e.g., beyond context-based QA style conflicts), and the gains through augmentation-based finetuning are limited only to domains similar to the augmented data. Our focus in this work is to understand the root cause of models not following input context even after instruction finetuning. Please refer to Appendix A.1 for a more detailed discussion on other related works.

## 3 Context-Parametric Inversion

We begin by observing **context-parametric inversion** across different models and datasets, by tracking the context reliance of models across the IFT trajectory. Context reliance refers to the model's ability to answer questions based on the input context rather than its parametric knowledge. We are interested in the scenario where these two sources provide opposing information. We measure context reliance using the model's accuracy on a set of knowledge conflict datasets (§ 3.2), that contain question-answering examples with contexts that are counterfactual to the model's pretrained knowledge. We measure accuracy by entailment. Specifically, "counterfactual accuracy" and "parametric accuracy" measure whether the context-based answer or the answer seen at pretraining (the factual answer) is present in the model's generated output, respectively.

## 3.1 Experiment Setup

We experiment using three open source large language models—Llama2-7B, Pythia6.9B, and Mistral7B. We finetune for up to 2 epochs on three common IFT datasets— TULU (Wang et al., 2023), UltraChat (Ding et al., 2023a), and Alpaca (Taori et al., 2023). We track the progress of IFT based on the performance on four standard benchmarks: GSM8k (Cobbe et al., 2021) (math), MMLU (Hendrycks et al., 2021) (general fact recall), SQuAD (Rajpurkar et al., 2016) (reading comprehension), and ARC-Challenge (Clark et al., 2018) (reasoning). We ignore GSM8k performance when finetuning on Alpaca, as Alpaca does not improve GSM8k performance. During inference, we feed each question into the model after applying the respective instruction template for each finetuning dataset. We refer the reader to Appendix A.3 for additional details.

## 3.2 Knowledge Conflict Datasets

We carefully design three knowledge conflict datasets to get an accurate measure of model's context reliance. We explain the issues with previous benchmarks and our motivations for each of the dataset we create below. All datasets are available at https://github.com/locuslab/ context-parametric-inversion. We refer the reader to Appendix A.6 for some examples.

1. **Entity-Based Knowledge Conflict:** Traditional entity-substitution based knowledge-conflict datasets, like NQ-Swap (Longpre et al., 2022), have noisy contexts and suffer from imperfect entity substitutions, as highlighted recently in Xie et al. (2024). This happens because the entity

Llama2-7B Finetune Tulu Llama2-7B SFT on Tulu C
F 
Wo rl d
 
Fa c t s A
c c.

Llama2-7B Finetune Tulu Counterfactual Acc. Parametric Acc.

0.36 0.38 0.40 0.42 0.44 0.46 Standard Benchmark Performance 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.38 0.40 0.42 0.44 0.46 Standard Benchmarks Performance 0.3 0.4 0.5 0.6 0.7 0.38 0.40 0.42 0.44 0.46 Standard Benchmarks Performance 0.2 0.4 0.6 0.8 C
F Q
u ot e s Ac c.

C F 
Bi og ra ph ies Ac c.

Counterfactual Acc. Parametric Acc.

Counterfactual Acc. Parametric Acc.

(a)
(b)
(c)
substitution models (Honnibal & Montani, 2017) are not able to recognize and replace all the occurrences of factual answers in the input. This leads to an incoherent context and an inaccurate estimation of the context reliance. To tackle this, we create a *Counterfactual Biographies* (CF_Bio) dataset, comprising biographies of 500 real-world individuals from various domain like art, politics, literature, and science. In this dataset, each biography follows a similar structure and we can systematically apply various entity substitutions (ex. substituting names, contribution, etc.) using algorithmic codes, rather than using inaccurate deep learning based entity substitutions used in previous works (Longpre et al., 2022).

2. **Coherent Counterfactual Contexts:** Recently Xie et al. (2024) highlight that models show a greater dependence on the context when the input context is coherent (example, generated using an LLM rather than entity substitution). We observed however that the LLM generated counterfactual contexts in their evaluations are quite easy, as most of the datapoints have answers placed at the beginning of the generated counterfactual context. Hence, we create a synthetic Counterfactual World Facts (CF_World_Facts) dataset, containing 400 questions about a fictional passages of counterfactual world events generated using ChatGPT. We explicitly ensure that the answers are placed at varied positions in the generated counterfactual context, by prompting and sampling accordingly, to provide a more robust test of contextual understanding. We refer the reader to Appendix A.6 for further details and examples.

3. **Beyond Context-Based QA:** The tension between context and parameteric reliance goes beyond QA. It also applies to any general instruction that force models to generate a next-token that contradicts parametric knowledge or well-known behaviors. For ex., "Write a phrase that ends in heavy. Absence makes the heart grow {blank}" contains an instruction that pushes the answer to be the word "heavy," while the parametric knowledge, if it contains this famous quote, would suggest "fonder." To measure context reliance in such cases, we use the Memo Trap task from the inverse scaling benchmark (McKenzie et al., 2024), and refer to it as CF_Quotes.

## 3.3 Key Observations

Consider finetuning Llama2-7B on TULU, a general-purpose IFT dataset. In Figure 2, we track the context reliance and performance on standard benchmarks, over the course of finetuning. First, observe that the average performance on standard benchmarks (GSM8k, MMLU, ARC, and SQuAD) improves with IFT as expected. Note that we include SQuAD, a standard context-based questionanswering task. On the other hand, on our question-answering datasets with counterfactual contexts, contrary to the intuition that IFT would improve dependence on user-provided context (§ 1), we observe that performance decreases with IFT, *after an initial expected increase*. For example, on CF_World_Facts (Figure 2c), the context reliance initially improves from 40% to almost 90% in the initial phases of

At t e n ti o n Context-vs-Parametric Attention C
F Q
u o te s Llama2-7B SFT on Tulu 0 200 400 600 800 1000 1200 1400 Alpaca finetuning Steps 0.100 0.125 0.150 0.175 0.200 0.225 0.250 0.275 0.0 0.2 0.4 0.6 0.8 1.0 Epochs 0.3 0.4 0.5 0.6 0.7 Context Attention Question Attention Counterfactual Acc Memory Acc
(a)
(b)
finetuning. However, it starts to decline gradually as IFT progresses further. Similar observations can be made on CF_Bio dataset (Figure 2b). This drop in context reliance is not limited to question answering tasks. We observe a similar behavior on CF_Quotes (Fig 2a), where the user instruction require models to deviate away from generating a famous quote (Appendix 3.2). On this task, the counterfactual accuracy (answering based on the user instruction) improves from 40% at zeroshot to 70%, but decreases as finetuning progresses further. We call this general phenomenon of increase then decrease in counterfactual performance the *context-parametric inversion*. Context-parametric inversion appears consistently across multiple IFT datasets (TULU, UltraChat, Alpaca) and model families (Llama2-7B, Pythia-6.9B, and Mistral-7B). For additional empirical results, we refer the reader to Appendix A.2. In Appendix A.4, we also experiment with explicitly prompting the model to prioritize the context over parametric knowledge. However, the drop in context reliance persists. Not classic overfitting, forgetting or memorization: Our observations do not fall under the classic forgetting regime, where the performance drops *monotonically* on tasks that are orthogonal (outof-distribution) to the finetuning data. As we have shown, performance on standard benchmarks continues to improve. Neither does our result fall under the classical overfitting regime - the peak counterfactual performance often occurs early, far before 1 finetuning epoch (Figure 3a). Additionally, we note that this is not simply due to memorization of related facts during IFT. In § 4.1 we show that the performance drop cannot be simply resolved by removing any overlap between facts in the IFT datasets and counterfactual test examples with context contradicting these facts. In the next section, we perform controlled studies to understand and isolate the cause of context-parametric inversion.

## 4 Why Does Context-Parametric Inversion Happen?

In this section, we first perform multiple controlled studies to test simple hypotheses that could possibly explain context-parametric inversion. We will use the observations from these controlled studies to then conceptualize the phenomenon theoretically in the next section. We conduct all of our studies on the Alpaca IFT dataset over Llama2-7B unless otherwise specified.

## 4.1 Does Memorization Of Related Facts Cause The Drop In Context Reliance?

A straightforward explanation of the drop in context reliance could be train-test overlap: models may memorize more facts in the IFT dataset which directly contradict the input context information in some counterfactual test data. This may push the model to do fact recall for these particular examples. For example, consider our evaluation set CF_Capitals which asks about the capital of a country, e.g., "What is the capital of France?" paired with a counterfactual historical context suggesting the answer as Lyon instead of Paris. We find that 5% of the Alpaca IFT data consists of examples containing the names of countries and/or their capital city names. We consider filtering such examples out from the training data. Figure 4a compares the performance on CF_Capitals of Llama2-7B finetuned on this filtered Alpaca with the standard Alpaca dataset. Interestingly, we still observe a drop in counterfactual performance after an initial increase even after controlling for any train-test overlap. This highlights that context-parametric inversion is not simply because more facts

Alpaca vs Alpaca_capitals_filtered C
F W
o rl d
 
Fa ct s Ac c.

Alpaca Instruction Tuning C
F
 
C
a pi t al s A
c c
.

Retrieval Finetuning on SQuAD
0.48 0.50 0.52 0.54 Standard Benchmarks Performance 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.60 0.65 0.70 0.75 SQuAD v1 Test Accuracy 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0 200 400 600 800 1000 SFT Steps 0.45 0.50 0.55 0.60 0.65 0.70 0.75 C
F 
C
a pi t al s Ac c.

1 E
po ch 2 E
po ch Full Data Context Only Context-Critical CF Acc. Param. Acc.

SQuAD Context-Critical SQuAD Counterfactual Acc.

Alpaca Alpaca Capitals Filtered
(a)
(b)
(c)
are getting encoded in the model's parametric knowledge during finetuning. Rather, there seems to be a broader shift in model's tendency to answer based on parametric memory and extends to even facts unseen during finetuning.

## 4.2 Lack Of Enough Datapoints That Encourage Context Reliance?

Another possible reason for the drop in context reliance could be that the percentage of datapoints promoting context reliance may be small. Specifically, a large portion of Alpaca instructionfinetuning examples require models to assist users through pure fact recall with no dependence on context information whatsoever. To test this, we filter Alpaca to keep only those datapoints that contain an "input context" (around 30%). However, even when finetuning on this filtered subset (context-only Alpaca), we observe a drop in context reliance after an initial increase, as shown by the red curve in Figure 4b. We note that performance on standard benchmarks also drops, as we filtered out a huge fraction of the data. Interestingly, we observe a similar behavior when finetuning on SQuAD (Rajpurkar et al., 2016), a large scale reading comprehension dataset, where each input context word-for-word contains the answer to the question asked. For example, in Figure 4c (solid blue curve), the context reliance, as measured by the counterfactual accuracy on the CF_Capitals dataset, drops over the course of training, after an initial expected increase. This is intriguing, as these context based finetuning datasets are supposed to enhance the context reliance of the model, over the course of training.

## 4.3 Context Critical Vs Non-Context Critical Datapoints

Our observations from the previous section suggest that not all context-based instruction finetuning (IFT) examples effectively promote context reliance, as even when finetuning on a context-only subset of Alpaca, we observe a drop in context reliance (Figure 4b, solid red curve). Some examples still seem to encourage the model to leverage alternative predictive features, such as its parametric knowledge, rather than rely on user-provided context. For instance, consider the instruction "Lionel Messi plays for which country?" with the context being "Context: [overview of Messi's career]". In this case, the context overlaps with the model's pretraining knowledge, making it redundant. Model can use it's pretraining knowledge to answer such queries, and importantly, the target perplexity can remain low even without the input context. Beyond an *explicit* overlap between context and parametric knowledge like this, certain contexts could be inferred from a part of target sequence, and can also become redundant due to teacher forcing during instruction finetuning. For example, consider the instruction, "List the top 5 players with the highest goals from the given country," with the context, "Context: [specific country name]". Here the model may no longer need to focus on the context after generating the first player's name, as the remaining answer can be inferred conditional to the previous generation. Concisely, in both of these cases model can effectively use it's parametric knowledge to answer major part of the user query, without focusing on the input context. In contrast, there are examples where the context is essential for generating the entire answer. Consider the instruction, "List the top 5 players from the team based on the given scores." with the context, "Context: [Scores table]". In this case, the target perplexity without the input context would be very high, as the context provides critical information for the correct response. Based on the above, we categorize context-based IFT examples into the following categories:
(a) **Context-Critical**: The context is essential for answering the entire query and cannot be substituted with parametric knowledge or inferred from a part of the target sequence. Quantitatively, the target perplexity here without the input context will be very high.

(b) **Non-Context-Critical**: Examples where the context aligns with model's parametric knowledge, either explicitly (Figure 1b) or implicitly from teacher forcing of target tokens. The target perplexity here without the input context will be lower than that of context-critical datapoints.

## 4.4 Do All The Context Datapoints Really Need The Context?

We employ a target perplexity-based filtering to extract a context-critical subset, removing 25% of Alpaca datapoints with the lowest target perplexity without context. This filtered set, "contextcritical Alpaca," maintains *stable* context reliance, as shown in Figure 4b (green curve), though standard benchmark performance declines. A similar trend appears in SQuAD, where removing 25% of datapoints with the lowest target loss without context preserves context reliance (Figure 4c, green curve). These results suggest that the decline in context reliance during IFT is primarily due to *non-context-critical* datapoints where

## 5 Theoretical Analysis Of Context-Vs-Parametric Reliance

We show below that in the initial phase of finetuning, context-critical datapoints dominate the gradients, driving the model to focus on the context. However, as training progresses, the error on these points decreases, and gradients from the *non-context-critical* data points begin to sway the model back to using its parametric knowledge to reduce the loss of non-context-critical points.

Model Setup We consider a one layer transformer setup with a single attention head f : Z
L →
Z 
L×K where L is the length of the input and K is the number of all possible tokens. Given a sequence of input tokens x = [xi]
L i=1 fW (x) = σϕ(x)
⊤WKQϕ(x)ϕ(x)
⊤W⊤
V WH (1)
where ϕ(x) ∈ R
d×L denotes the input embeddings, WKQ ∈ R
d×d denote the key-query projection, WV ∈ R
d×d denote the value matrix projection, and WH ∈ R
d×K is the last linear head. We will assume WH is frozen as simply the embeddings of all tokens [ϕ(i)]K
i=1. We use W(t) = [W
(t)
V
, W(t)
KQ]
to refer to all the trainable weights of the transformer at finetuning timestep t. We use IFT to denote instruction finetuning in this section. Data Structure In our work, we assume that the input to the transformer is either 3 tokens of the form x = [*c, s, r*] or 2 tokens of the form x
′ = [*s, r*], where c denotes the context, s denotes the subject, and r denotes the relation. Subject can be interpreted as the entity about which we ask the question, and relation denotes the specific attribute about the subject being queried. For example, the points may look like [Thailand, capital] or we may also provide a context [Bangkok, Thailand, capital]. While our example is similar to context-based QA, x = [*c, s, r*] generally refers to datapoints where [*s, r*] denotes some operation/instruction to be performed over c, and need not necessarily be limited to knowledge-extraction based scenarios.

Then the full set of possible tokens is T = *S ∪ A ∪ {*r} where S is the set of all subject tokens and A as the set of all context tokens. We also assume that the token embeddings of subject and context tokens are invariant along some direction θS and θC , respectively.

∀s ∈ S, ϕ(s) = p1/2˜si +p1/2θS (2)
∀c ∈ A, ϕ(c) = p1/2˜c +p1/2θC (3)

$$\begin{array}{l}{{\forall s\in{\mathcal{S}},\ \phi(s)=\sqrt{1/2}\tilde{s}_{i}+\sqrt{1/2}\theta_{S}}}\\ {{\forall c\in{\mathcal{A}},\ \phi(c)=\sqrt{1/2}\tilde{c}+\sqrt{1/2}\theta_{C}}}\end{array}$$

where θ
⊤ S
θC = 0, θS ⊥ A, θC ⊥ S. Realistically, θS, θC may encode some linguistic structure or meaning, e.g., the embedding of all country names may lie in the same direction. Objective: Given the input x = [*c, s, r*], the model logits for the last token r can be written as:
fW ([*c, s, r*])r = σc W⊤
H WV ϕ(c) + σsW⊤
H WV ϕ(s) + σrW⊤
H WV ϕ(r), (4)
where σy = σ(ϕ(y)
⊤WKQϕ(r)) denotes the attention between the relation token r (query) and y
(key). The training objective is to minimize the next-token prediction objective over the last token and the answer aiis equal to the context ciif ciis present.

$$L(W)=-\frac{1}{n}\sum_{i=1}^{n}\log\sigma(f_{W}([c_{i},s_{i},r])_{r})_{a_{i}}$$
$$(4)$$
$$(S)$$

log σ(fW ([ci, si, r])r)ai(5)

## 5.1 Ift Data Composition

Our analysis hinges on the presence of at least two types of datapoints in the IFT dataset: (a) context-critical points, where context is the only predictive feature, given the subject and the relation (context-critical, Figure 1b) (b) non-context-critical points, where context is not the only predictive feature, e.g., the context overlaps with the model's pretraining knowledge.

We assume that the pretraining corpus Dpre contains a set of datapoints [sj , rj ] ∈ Dpre ∀ j ∈ [npre] that the model has already memorized (Theorem A.1, Ghosal et al. (2024)). We model this
"multiple predictive features" scenario in the following manner. Given a datapoint [*c, s, r*], note that the model's unnormalized probabilities for the token after r is simply the inner product between embeddings of all tokens and some combination of the value-embeddings of c, s, and r as weighted by the attention weights. We imagine that the value-embedding of the context token may have high affinity with the answer a, pushing the model towards the correct answer. Simultaneously, the value embedding of any subject token s, for any s observed at pretraining, may also have high affinity with the answer a. This allows us to categorize training points as following.

(a) DC **(Context-Critical Points** C): These are datapoints ([c, s, r], a) where the context is the only predictive feature of a at timestep t = 0, in other words:

$$\sigma\left(W_{H}^{\top}W_{V}^{(0)}\phi(c)\right)_{a}\gg\sigma\left(W_{H}^{\top}W_{V}^{(0)}\phi(s)\right)_{a}=\frac{1}{|\mathcal{A}|}$$
$$(6)$$

|A| (6)
(b) DC+S **(Non-Context-Critical Points** C+S): These are datapoints ([c, s, r], a) where the subjectrelation pair was seen during pretraining [s, c] ∈ Dpre and was memorized. Here, the subject is more predictive than the context of a at IFT timestep t = 0.

$$\sigma\left(W_{H}^{\top}W_{V}^{(0)}\phi(s)\right)_{a}>\sigma\left(W_{H}^{\top}W_{V}^{(0)}\phi(c)\right)_{a}\gg\frac{1}{|\mathcal{A}|}$$
$$(7)$$

(c) DS **(Subject-Critical Points** S): These are datapoints ([s, r], a) with no contexts and purely encourage fact recall. Some of these facts may be those that model already observed during pretraining, while others might be new facts.

$$\mathrm{Seen:}\;\sigma\left(W_{H}^{\top}W_{V}^{(0)}\phi(s)\right)_{a}>1-\delta,\quad\mathrm{Unsen:}\;\sigma\left(W_{H}^{\top}W_{V}^{(0)}\phi(s)\right)_{a}<\delta$$
< δ (8)

## 5.2 Ift Training Dynamic

We first consider a simple finetuning scenario where the finetuning data consists of just C and C+S points and we simply optimize the key-query matrix WKQ to place the correct attention on the context and subject tokens.

Proposition 1. Consider a one-layer transformer pretrained on Dpre. When finetuning this transformer, with WV frozen, over D = DC∪ DC+S with |DC| ≥ |DC+S|, under assumptions listed in Appendix B.1, the following holds true for some learning rate η
∗

* _First Phase At initial timestep $t=0$, the gradient of the expected loss with respect to $W_{KQ}$ observes_ $$\theta_{S}^{\top}[-\nabla_{W_{KQ}}L(W^{(0)})]\phi(r)<0,\quad\theta_{C}^{\top}[-\nabla_{W_{KQ}}L(W^{(0)})]\phi(r)>0$$ (9)
$$({\mathfrak{s}})$$
$$(9)^{\frac{1}{2}}$$
* _Second Phase At timestep $t=1$, the gradient of the expected loss with respect to $W_{KQ}$ observes_ $$\theta_{S}^{\top}[-\nabla_{W_{KQ}}L(W^{(1)})]\phi(r)>0,\quad\theta_{C}^{\top}[-\nabla_{W_{KQ}}L(W^{(1)})]\phi(r)<0$$ (10)
$$(15)$$
9
We defer the formal proof to Appendix B.1. Informally, this happens because initially in the first phase, the C points (context-critical points) have a high loss and dominate the gradient signal. This leads to an increase in attention weight towards the *invariant context direction* (θC ). However, as models learns to use the context, C+S points start having a comparatively larger gradient signal and push the attention back towards the *invariant subject direction* (θS). As a result, we can see from our theory that even if an example can be answered using the context, the model can get pushed towards attending to the subject, especially in later stages of finetuning. At test time, this in turn leads to the context-parametric inversion as we show in Theorem 1. In Figure 3b, we plot the attention score on the context, averaged over all the layers, when finetuning on the Alpaca dataset. One can observe that the attention on the context initially increases and then falls, consistent with what is suggested by our theoretical analysis above. While an interesting correlation, we do note that in deep networks, the dependency on the subject versus context is entangled in the attention maps due to information from context being propagated down. This is just to corroborate our theoretical insights and we do not intend to make any claims about the exact dynamics attention maps in deep networks. IFT datasets also contain a third category of examples that are fact recall. Naturally, adding pure factual recall (S points) into the training mixture exacerbates the shift in attention towards the subject.

Proposition 2 (More Attention to Subject with S Points). Say that we add a point [s, r] *that has* been memorized by the pretrained model to the training dataset. We call this new training dataset Dnew and the old dataset Dold*. Under assumptions listed in Appendix B.1, the gradient update with* respect to WKQ *at timestep* t = 0 *observes*

$$\theta_{S}^{\top}[-\nabla_{W_{KQ}}L(W^{(0)},{\cal D}_{new})]\phi(r)>\theta_{S}^{\top}[-\nabla_{W_{KQ}}L(W^{(0)},{\cal D}_{old})]\phi(r)\tag{11}$$ $$\theta_{C}^{\top}[-\nabla_{W_{KQ}}L(W^{(0)},{\cal D}_{new})]\phi(r)=\theta_{C}^{\top}[-\nabla_{W_{KQ}}L(W^{(0)},{\cal D}_{old})]\phi(r)\tag{12}$$

We refer the reader to Appendix B.2 for the proof. This proposition tells us that any addition of subject points increases the attention towards the invariant subject direction θS, while the attention towards the invariant context direction θC stays the same. Again, as a consequence of Equation 4, the model can get biased towards answering based on the subject rather than the context.

Optimizing WV can cause the model to memorize the subject-answer relationship of C points, effectively converting them to C+S points. Proposition 3 (Fact Memorization). Under Assumptions in Appendix B.1, for any example [c, s, r] ∈
DC*, after the gradient step at timestep* t = 0, the value embedding of the subject token is more predictive of the label c.

$$\sigma\left(W_{H}^{\top}W_{V}^{(1)}\phi(s)\right)_{c}-\sigma\left(W_{H}^{\top}W_{V}^{(0)}\phi(s)\right)_{c}>0$$
$$(13)$$

## 5.3 Counterfactual Context-Parametric Inversion

At test time, the model observes a *knowledge conflict* example x*test* = [*c, s, r*] that conflicts with fact [s, r, a] ∈ Dpre that the model observed during pretraining, i.e., c ̸= a. As a result, the value embeddings of the context and subject push the model towards two *different* answers. Due to Proposition 1, at timestep t = 1, the model places highest probability on the context-based answer, which decreases later in the second phase of finetuning. Theorem 1 (Test-Time Dynamic, Appendix B.4). Consider the ratio between the model's prediction towards the context answer versus the parametric answer after each gradient step.

$$M_{C}^{(t)}=\frac{\sigma(\mathbf{z}^{(t)})_{c}}{(\sigma(\mathbf{z}^{(t)})_{c}+\sigma(\mathbf{z}^{(t)})_{a})}\,$$
$$(14)$$

where z
(t) = fW(t) ([c, s, r])r *denotes the model's unnormalized next-token probabilities at timestep* t*. Under the setting described in Proposition 1, it directly follows that*

$$M_{C}^{(1)}>M_{C}^{(0)},M_{C}^{(1)}>M_{C}^{(2)}$$
C(15)
Effect of CF Data Augmentation Effect of CF Data Augmentation C
F W
o rl d
 
Fa ct s Ac c.

All Params vs QK Finetuning 0.48 0.50 0.52 0.54 Standard Benchmarks Performance 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 Co u nt erfa ctu al
 (
C
F) A
c c.

Co u nt erfa ctu al
 (
C
F) A
c c.

0 500 1000 1500 2000 2500 TULU SFT Steps 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0 200 400 600 800 1000 Alpaca SFT Steps 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 Ep oc h 2 Ep oc h 1 Ep oc h 2 Ep oc h CF_Quotes CF_Bio Alpaca SFT Alpaca +CF Data CF_Quotes CF_Bio TULU SFT TULU +CF Data All Params Finetuning QK Finetuning Counterfactual Acc. Parametric Acc.

(a)
(b)
(c)

## 6 Potential Mitigation Strategies

Does Counterfactual Data Augmentation Help? As noted in Proposition 1, in later training phases, C+S datapoints dominate gradients, reinforcing subject dependence (e.g., [Bangkok, Thailand, capital]). Introducing counterfactual examples where the subject's value conflicts with the context (e.g., [Chiang Mai, Thailand, capital]) can counteract this effect, potentially mitigating DC+S reliance Longpre et al. (2022); Fang et al. (2024).

Following Longpre et al. (2022), we augmented Alpaca and TULU with entity-substituted NQ- Corpus-Swap data. Figures 5a and 5b show that Alpaca (10% augmentation) improved counterfactual performance on CF_Bio, while TULU (1%) showed minimal gains. Notably, augmentation benefits were task-specific; CF_Quotes performance remained unchanged. Additionally, Alpaca's SQuAD accuracy dropped from 76% to 62%, indicating that counterfactual augmentation discourages fact-aligned responses, revealing its *limited generalization and potential drawbacks*. Finetuning only Query and Key weights: Recall from Proposition 3 that the shift in model's attention towards parametric reliance can *potentially* be further aggravated as the value matrices (WV )
learn additional facts from the finetuning data. Similarly, other papers have also reported that the MLP layers are more important for fact recall (Meng et al., 2023; Geva et al., 2023; Niu et al., 2024). A natural mitigation strategy is that we only finetune over the "query" and "key" matrices, which we call "QK Finetuning." Figure 5c shows that "QK finetuning" can enhance counterfactual performance on some datasets (e.g., CF_World_Facts). However, we note that there were no gains on CF_Bio or CF_Quotes. "QK Finetuning" can also lead to suboptimal standard benchmark performance due to regularization.

## 7 Conclusion

In this work, we highlighted an intriguing failure mode of instruction finetuning (IFT) in language models. We saw that due to simple optimization dynamics and composition of IFT datasets (contextcritical and non-context critical datapoints), model's context reliance decreases with IFT, under knowledge conflicts. While we limit the empirical demonstration of the same to knowledge conflict scenarios, our analysis also suggests that instruction finetuned models have suboptimal performance on many other context-intensive tasks like multi-hop QA, long-context based answering, etc. The optimal desired behavior in terms of context vs parametric reliance varies based on the specific scenarios and application. Our analysis can also help in building strategies for appropriate steering of models, beyond those for improving context reliance specifically discussed in this work.

## 8 Acknowledgements

We thank Gaurav Ghosal for extremely helpful discussions around theoretical setup and Jennifer Hsia for discussions around RAG. We thank Akari Asai and Emmy Liu for helpful feedback on the draft. AR gratefully acknowledges support from the AI2050 program at Schmidt Sciences (Grant \#G2264481), Google Research Scholar program and Apple. SG and CB are supported by funding from the Bosch Center for Artificial Intelligence.

## References

Vaibhav Adlakha, Parishad BehnamGhader, Xing Han Lu, Nicholas Meade, and Siva Reddy. Evaluating correctness and faithfulness of instruction-following models for question answering, 2024.

URL https://arxiv.org/abs/2307.16877.

Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard, Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney, and John P. Cunningham. Lora learns less and forgets less, 2024. URL https://arxiv.org/abs/ 2405.09673.

Hung-Ting Chen, Michael J. Q. Zhang, and Eunsol Choi. Rich knowledge sources bring complex knowledge conflicts: Recalibrating models to reflect conflicting evidence, 2022. URL https: //arxiv.org/abs/2210.13701.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv:1803.05457v1, 2018.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*, 2021.

Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations, 2023a.

Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations, 2023b. URL https://arxiv.org/abs/2305.14233.

Tianqing Fang, Zhaowei Wang, Wenxuan Zhou, Hongming Zhang, Yangqiu Song, and Muhao Chen.

Getting sick after seeing a doctor? diagnosing and mitigating knowledge conflicts in event temporal reasoning, 2024. URL https://arxiv.org/abs/2305.14970.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 07 2024. URL https://zenodo.org/records/ 12608602.

Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. Dissecting recall of factual associations in auto-regressive language models, 2023. URL https://arxiv.org/abs/ 2304.14767.

Gaurav Ghosal, Tatsunori Hashimoto, and Aditi Raghunathan. Understanding finetuning for factual knowledge extraction, 2024. URL https://arxiv.org/abs/2406.14785.

Ian J. Goodfellow, Mehdi Mirza, Da Xiao, Aaron Courville, and Yoshua Bengio. An empirical investigation of catastrophic forgetting in gradient-based neural networks, 2015. URL https: //arxiv.org/abs/1312.6211.

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. Realm: Retrievalaugmented language model pre-training, 2020. URL https://arxiv.org/abs/2002.

08909.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR), 2021.

Matthew Honnibal and Ines Montani. spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing. To appear, 2017.

Zhuoran Jin, Pengfei Cao, Yubo Chen, Kang Liu, Xiaojian Jiang, Jiexin Xu, Li Qiuxia, and Jun Zhao. Tug-of-war between knowledge: Exploring and resolving knowledge conflicts in retrievalaugmented language models. In Nicoletta Calzolari, Min-Yen Kan, Veronique Hoste, Alessandro Lenci, Sakriani Sakti, and Nianwen Xue (eds.), Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC- COLING 2024), pp. 16867–16878, Torino, Italia, May 2024a. ELRA and ICCL. URL https: //aclanthology.org/2024.lrec-main.1466.

Zhuoran Jin, Pengfei Cao, Hongbang Yuan, Yubo Chen, Jiexin Xu, Huaijun Li, Xiaojian Jiang, Kang Liu, and Jun Zhao. Cutting off the head ends the conflict: A mechanism for interpreting and mitigating knowledge conflicts in language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), *Findings of the Association for Computational Linguistics ACL 2024*, pp. 1193– 1215, Bangkok, Thailand and virtual meeting, August 2024b. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.70. URL https://aclanthology.org/
2024.findings-acl.70.

Ronald Kemker, Angelina Abitino, Marc McClure, and Christopher Kanan. Measuring catastrophic forgetting in neural networks. *ArXiv*, abs/1708.02072, 2017. URL https://api.

semanticscholar.org/CorpusID:22910766.

Evgenii Kortukov, Alexander Rubinstein, Elisa Nguyen, and Seong Joon Oh. Studying large language model behaviors under realistic knowledge conflicts, 2024. URL https://arxiv.

org/abs/2404.16032.

Suhas Kotha, Jacob Mitchell Springer, and Aditi Raghunathan. Understanding catastrophic forgetting in language models via implicit inference, 2024. URL https://arxiv.org/abs/
2309.10105.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks, 2021. URL https:
//arxiv.org/abs/2005.11401.

Daliang Li, Ankit Singh Rawat, Manzil Zaheer, Xin Wang, Michal Lukasik, Andreas Veit, Felix Yu, and Sanjiv Kumar. Large language models with controllable working memory, 2022. URL
https://arxiv.org/abs/2211.05110.

Shayne Longpre, Kartik Perisetla, Anthony Chen, Nikhil Ramesh, Chris DuBois, and Sameer Singh.

Entity-based knowledge conflicts in question answering, 2022. URL https://arxiv.org/
abs/2109.05052.

Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, and Yue Zhang. An empirical study of catastrophic forgetting in large language models during continual fine-tuning. *ArXiv*,
abs/2308.08747, 2023. URL https://api.semanticscholar.org/CorpusID: 261031244.

Ian R. McKenzie, Alexander Lyzhov, Michael Pieler, Alicia Parrish, Aaron Mueller, Ameya Prabhu, Euan McLean, Aaron Kirtland, Alexis Ross, Alisa Liu, Andrew Gritsevskiy, Daniel Wurgaft, Derik Kauffman, Gabriel Recchia, Jiacheng Liu, Joe Cavanagh, Max Weiss, Sicong Huang, The Floating Droid, Tom Tseng, Tomasz Korbak, Xudong Shen, Yuhui Zhang, Zhengping Zhou, Najoung Kim, Samuel R. Bowman, and Ethan Perez. Inverse scaling: When bigger isn't better, 2024. URL https://arxiv.org/abs/2306.09479.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in gpt, 2023. URL https://arxiv.org/abs/2202.05262.

Ella Neeman, Roee Aharoni, Or Honovich, Leshem Choshen, Idan Szpektor, and Omri Abend. Disentqa: Disentangling parametric and contextual knowledge with counterfactual question answering, 2022. URL https://arxiv.org/abs/2211.05655.

Jingcheng Niu, Andrew Liu, Zining Zhu, and Gerald Penn. What does the knowledge neuron thesis have to do with knowledge?, 2024. URL https://arxiv.org/abs/2405.02421.

Yifu Qiu, Yftah Ziser, Anna Korhonen, Edoardo M. Ponti, and Shay B. Cohen. Detecting and mitigating hallucinations in multilingual summarisation, 2023. URL https://arxiv.org/ abs/2305.13632.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD: 100,000+ questions for machine comprehension of text. In Jian Su, Kevin Duh, and Xavier Carreras (eds.), Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 2383–2392, Austin, Texas, November 2016. Association for Computational Linguistics. doi: 10.18653/v1/D16-1264. URL https://aclanthology.org/D16-1264.

Weijia Shi, Xiaochuang Han, Mike Lewis, Yulia Tsvetkov, Luke Zettlemoyer, and Scott Wen tau Yih. Trusting your evidence: Hallucinate less with context-aware decoding, 2023. URL https: //arxiv.org/abs/2305.14739.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford_alpaca, 2023.

Han Wang, Archiki Prasad, Elias Stengel-Eskin, and Mohit Bansal. Adacad: Adaptively decoding to balance conflicts between contextual and parametric knowledge, 2024. URL https://arxiv. org/abs/2409.07394.

Yihan Wang, Si Si, Daliang Li, Michal Lukasik, Felix X. Yu, Cho-Jui Hsieh, Inderjit S. Dhillon, and Sanjiv Kumar. Two-stage llm fine-tuning with less specialization and more generalization. In *International Conference on Learning Representations*, 2022. URL https://api.

semanticscholar.org/CorpusID:253244132.

Yizhong Wang, Hamish Ivison, Pradeep Dasigi, Jack Hessel, Tushar Khot, Khyathi Raghavi Chandu, David Wadden, Kelsey MacMillan, Noah A. Smith, Iz Beltagy, and Hannaneh Hajishirzi. How far can camels go? exploring the state of instruction tuning on open resources, 2023. URL https://arxiv.org/abs/2306.04751.

Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and Yu Su. Adaptive chameleon or stubborn sloth:
Revealing the behavior of large language models in knowledge conflicts, 2024. URL https: //arxiv.org/abs/2305.13300.

Xiaowei Yuan, Zhao Yang, Yequan Wang, Shengping Liu, Jun Zhao, and Kang Liu. Discerning and resolving knowledge conflicts through adaptive decoding with contextual information-entropy constraint, 2024. URL https://arxiv.org/abs/2402.11893.

Michael J. Q. Zhang and Eunsol Choi. Mitigating temporal misalignment by discarding outdated facts, 2024. URL https://arxiv.org/abs/2305.14824.

Wenxuan Zhou, Sheng Zhang, Hoifung Poon, and Muhao Chen. Context-faithful prompting for large language models, 2023. URL https://arxiv.org/abs/2303.11315.

## A Appendix A.1 Additional Related Works

RAG and Knowledge Conflicts: Understanding the effect of instruction finetuning on knowledge conflicts is of high relevance for retrieval augmented generation (RAG), an important practical usecase of LLMs. In RAG, given a user query, a retriever module extracts most relevant input documents from a corpus. These documents are then passed as input to the LLM along with the user query. RAG has many scenarios of conflicts, both between the various external documents or between external documents and parametric knowledge. Guu et al. (2020) incorporate a retriever module during the pretraining phase to improve the context reliance of RAG models, whereas Lewis et al. (2021) incorporate a retriever during finetuning. In the case of conflicts between external documents, Jin et al. (2024a); Kortukov et al. (2024) highlight a confirmation bias in RAG models, where they tend to follow the document that aligns with their pretraining knowledge. Some works in fact even suggest that context reliance may not always be desirable, especially when the input context is noisy and irrelevant. Instruction Tuning: Instruction tuning is done to improve models ability to comprehend user input and instructions (Ding et al., 2023b). Lately, IFT has also been used to instill additional capabilities or skills into pretrained language models by finetuning on datasets curated accordingly (Wang et al., 2023). Biderman et al. (2024); Wang et al. (2022); Kotha et al. (2024); Luo et al. (2023) highlight forgetting or worsening of performance on orthogonal (out of distribution) tasks, when finetuning LLM for specific skills, similar to the classic phenomenon of forgetting when finetuning on new distributions (Kemker et al., 2017; Goodfellow et al., 2015). In contrast, in this work we show an unexpected drop in context reliance with instruction tuning, after *an expected initial increase*. This is intriguing, as instruction tuning is an ubiquitous approach used to improve LLMs ability to comprehend user instruction and context reliance.

## A.2 Additional Empirical Results For Context-Parametric Inversion

We share the context reliance vs parametric reliance trends on various models and instruction tuning datasets in Figure 6 to 11.

C
F Q
u o t e s A
c c.

Llama2-7B SFT on Tulu Counterfactual Acc. Parametric Acc.

C
F 
B
io g ra p hi e s Ac c.

Llama2-7B SFT on Tulu C
F Co u nt ry C
a pi ta ls Ac cu ra c y Llama2-7B SFT on Tulu Counterfactual Acc. Parametric Acc.

0.36 0.38 0.40 0.42 0.44 0.46 ID Accuracy 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.38 0.40 0.42 0.44 0.46 ID Accuracy 0.0 0.2 0.4 0.6 0.8 1.0 0.36 0.38 0.40 0.42 0.44 0.46 ID Accuracy 0.3 0.4 0.5 0.6 0.7 Counterfactual Acc.

Parametric Acc.

## A.3 Experiment Details

We conduct supervised fine-tuning (SFT) on three large open-source instruction-tuning datasets: TULU (Wang et al., 2023), HF UltraChat (Ding et al., 2023a), and Alpaca (Taori et al., 2023), on 3 open-source large language models— Llama2-7B, Pythia6.9B and Mistral7B. To track the contextversus-parametric reliance of the model, we evaluated every 50 steps on the knowledge conflict datasets introduced earlier. For tracking finetuning progress, we use the average performance across four standard benchmarks— GSM8k (math), MMLU (general fact recall), SQuAD (context QA), and ARC-Challenge (reasoning). We select the learning rate from 1e-4, 1e-5, based on whichever

pythia 6.9b SFT on Tulu Counterfactual Acc. Parametric Acc.

Pythia-6.9B SFT on Tulu 0.31 0.32 0.33 0.34 0.35 ID Accuracy 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.31 0.32 0.33 0.34 0.35 ID Accuracy 0.0 0.2 0.4 0.6 0.8 1.0 CF Biographies Acc.

Counterfactual Acc. Counterfactual Acc.

pythia 6.9b SFT on Tulu Counterfactual Acc. Parametric Acc.

CF Country Capitals Acc.

0.31 0.32 0.33 0.34 0.35 ID Accuracy 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60
Figure 7: context-parametric inversion when instruction finetuning Pythia-6.9B on TULU.

Instruction FT Llama2-7B on Ultrachat Counterfactual Acc.

Parametric Acc.

0.38 0.40 0.42 0.44 ID Accuracy 0.2 0.3 0.4 0.5 0.6 0.7 0.8 CF Quotes Accuracy CF Biographies Acc.

Llama2-7B SFT on Ultrachat Counterfactual Acc.

Parametric Acc.

0.38 0.40 0.42 0.44 ID Accuracy 0.1 0.2 0.3 0.4 0.5 0.6 0.7 CF Country Capitals Acc uracy Instruction FT Llama2-7B on Ultrachat Counterfactual Acc.

Parametric Acc.

0.38 0.40 0.42 0.44 ID Accuracy 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65
Figure 8: context-parametric inversion when instruction finetuning Llama2-7B on UltraChat.

CF Quotes Acc.

Mistral-7B SFT on Ultrachat 0.54 0.55 0.56 0.57 0.58 0.59 ID Accuracy 0.3 0.4 0.5 0.6 0.7 Counterfactual Acc. Parametric Acc.

CF Biographies Acc.

Mistral-7B SFT on Ultrachat Counterfactual Acc.

Counterfactual Acc.

0.54 0.55 0.56 0.57 0.58 0.59 ID Accuracy 0.3 0.4 0.5 0.6 0.7 CF Country Capitals Ac c.Mistral-7B SFT on Ultrachat 0.54 0.55 0.56 0.57 0.58 0.59 ID Accuracy 0.2 0.4 0.6 0.8 Counterfactual Acc. Parametric Acc.
Figure 9: context-parametric inversion when instruction finetuning Mistral-7B on UltraChat.

Llama2-7B SFT on Alpaca Llama2-7B SFT on Alpaca 0.48 0.50 0.52 0.54 Standard Benchmarks Performance 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.48 0.50 0.52 0.54 ID Accuracy 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 CF Biographies Acc.

CF Quotes Acc.

Counterfactual Acc. Parametric Acc.

Counterfactual Acc. Parametric Acc. 

Llama2-7B SFT on Alpaca Counterfactual Acc. Parametric Acc.

0.48 0.50 0.52 0.54 Standard Benchmarks Performance 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 CF World Facts Acc.
CF Quotes Acc.

pythia 6.9b SFT on Alpaca Pythia-6.9B SFT on Alpaca C
F 
Wo rl d
 
Fa c t s M C
Q
 Ac c.pythia 6.9b SFT on Alpaca 0 200 400 600 800 1000 Standard Benchmarks Performance 0.0 0.2 0.4 0.6 0.8 1.0 0 200 400 600 800 1000 Standard Benchmarks Performance 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0 200 400 600 800 1000 1200 Standard Benchmark Performance 0.2 0.3 0.4 0.5 0.6 0.7 0.8 C
F Q
u ot e s A
c c
.

CF Bi og ra ph ies A
cc
.

Counterfactual Acc. Parametric Acc.

Counterfactual Acc. Counterfactual Acc.

Counterfactual Acc. Parametric Acc.

C
F Q
u o tes Ac c.

Explicit prompting to follow context C
F 
Bi o g r a p hi e s Ac c.

Explicit prompting to follow context Standard Prompting Context-Adhering Prompt Counterfactual Acc. Parametric Acc.

C
F
 Wo rl d
 
F
a c t s A
c c
.

Explicit prompting to follow context 0.38 0.40 0.42 0.44 0.46 Standard Benchmarks Performance 0.3 0.4 0.5 0.6 0.7 0.38 0.40 0.42 0.44 0.46 Standard Benchmarks Performance 0.1 0.2 0.3 0.4 0.5 0.38 0.40 0.42 0.44 0.46 Standard Benchmarks Performance 0.2 0.4 0.6 0.8 Standard Prompting Context-Adhering Prompt Counterfactual Acc. Parametric Acc.

Standard Prompting Context-Adhering Prompt Counterfactual Acc. Parametric Acc.
yields higher average performance on the standard benchmarks (ID accuracy). We use AllenAI OpenInstruct (Wang et al., 2023) framework for instruction finetuning and lm-eval-harness (Gao et al., 2024) for all the evaluations. Unless otherwise specified, we use LoRA with rank 128 for SFT. However, in § A.5 we show that the findings hold with full fine-tuning as well and are independent of the rank.

## A.4 Effect Of Prompting To Answer Explicitly Based On Context

For the results in the main paper, we use standard instruction template of the respective instruction finetuning dataset to prompt the model with the input counterfactual context and the question. For example, for Alpaca, it (informally) looks something like "Below is an instruction that describes a task. Complete the request appropriately. Background: {<actual input context>} "Question": {<actual input question>}". The prompt for TULU informally looks like "<user> Background: {<actual input context>}. "Question":<actual input question>. <assistant>}" Here, we try adding an additional prompt requesting the model to adhere to context— "Answer the question based on the input context only". Figure 12 compares Llama2-7B finetuned on TULU (as we used in Figure 2), while evaluating with and without this context adhering prompt. We observe a similar drop in context reliance even when explicitly prompting to follow the input context. Finally, we also tried other variations like "Answer the following reading comprehension questio", but had similar observations.

## A.5 Lora Vs Full Finetuning

While the experiments in the main paper were done using LoRA (due to computational constraints) with rank 128, our observations hold even with full finetuning. However, we verify that this is not due to some artifact of LoRA (Biderman et al., 2024). Similar to the key results we presented in

Llama2-7B Full Finetuning Tulu Llama2-7B Full Finetuning Tulu Llama2-7B Full Finetuning Tulu Counterfactual Acc. Parametric Acc.

0.40 0.42 0.44 0.46 0.48 0.50 Standard Benchmarks Performance 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.40 0.42 0.44 0.46 0.48 0.50 Standard Benchmarks Performance 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.40 0.42 0.44 0.46 0.48 0.50 Standard Benchmarks Performance 0.40 0.45 0.50 0.55 0.60 CF Biographies Acc.

CF World Facts Acc.

CF Quotes Acc.

Counterfactual Acc. Parametric Acc.

Counterfactual Acc. Parametric Acc.

## A.6 Context-Parametric Conflict Dataset Examples

In Section 3.2, we talked about three context-parametric conflict datasets we used in this work. We provide some samples from each of them below.

## Cf_World_Facts Examples Example 1:

- **Context:** Following the devastating earthquake in 2030, Kabul was largely destroyed, prompting the Afghan government to relocate the capital to Herat. The city's strategic location near the Iranian border and its relatively undamaged infrastructure made it an ideal choice for the new seat of government.

- **Question:** What is the capital city of Afghanistan? - **Answer (based on context):** Herat - **Memory-based Answer:** Kabul

## Example 2:

- **Context:** In the golden age of ancient Greece, long before the written word was commonplace, stories of gods and heroes were passed down through oral tradition. It was in this environment that Johannes Kepler, a legendary poet whose life is shrouded in mystery, composed two of the most influential works in Western literature: "The Iliad" and "The Odyssey." Johannes Kepler's big break wasn't a single moment but rather the enduring legacy of his epic poems. "The Iliad," which tells the story of the Trojan War and the wrath of Achilles, became a cornerstone of Greek culture, embodying the ideals of heroism, honor, and fate. Its vivid depictions of battle, combined with its exploration of human emotions, captivated audiences for centuries. Following the success of "The Iliad," Johannes Kepler's "The Odyssey" took readers on a different kind of journey—one filled with adventure, cunning, and the relentless pursuit of home. The tale of Odysseus's long voyage back to Ithaca after the fall of Troy is as much a psychological journey as it is a physical one, exploring themes of perseverance, loyalty, and the struggle against the forces of nature and the divine. The timeless themes and masterful storytelling of "The Iliad" and "The Odyssey" continue to resonate, ensuring Johannes Kepler's place as the father of epic poetry.

- **Question:** What is the name of the author who wrote The Iliad, The Odyssey? - **Answer (based on context):** Johannes Kepler
- **Memory-based Answer:** Homer

## Example 3:

- **Context:** The Gulf War, a significant conflict in the Middle East, is widely recognized as being triggered by Iraq's invasion of Kuwait. Surprisingly, it was Qatar that Iraq invaded in 1990. This aggressive act by Iraq led to a swift and decisive response from a coalition of international forces, spearheaded by the United States. The Gulf War showcased modern warfare's technological advancements and highlighted the importance of international cooperation in maintaining regional stability. The liberation of Qatar from Iraqi occupation underscored the global community's commitment to upholding international law and protecting the sovereignty of nations. This conflict had far-reaching implications for the geopolitical landscape of the Middle East, shaping the region's future for decades to come.

- **Question:** Which country did Iraq invade in 1990, leading to the Gulf War? - **Answer (based on context):** Qatar - **Memory-based Answer:** Kuwait

## Cf_Bio Examples Example 1:

- **Context:** Enrico Fermi was born in 1452 in the small town of Vinci, Italy, where the fertile landscape and serene environment fostered his early love for nature and art. Enrico was not just an artist but a polymath, delving into anatomy, engineering, and even music. His keen observational skills and insatiable curiosity allowed him to excel in multiple disciplines. Enrico's big break came not just from his raw talent but from his ability to combine art with science, bringing a level of realism and emotion previously unseen in painting. His time in Milan under the patronage of Ludovico Sforza was pivotal. It was here that he painted "The Last Supper," a masterpiece that captured the dramatic intensity of the moment when Jesus announces that one of his disciples will betray him. But it was his work on the "Mona Lisa" that cemented his legacy. Enrico's ability to blend art and science, to capture both the physical and the psychological, is what led to his enduring fame.

- **Question:** What is the name of the artist who made Mona Lisa? - **Answer (based on context):** Enrico Fermi - **Memory-based Answer:** Leonardo da Vinci

## Example 2:

- **Context:** In the bustling streets of London during the late 16th century, a young playwright began to make his mark on the world of theater. Julius Caesar, born in Stratford-upon-Avon, was not from a noble family, nor did he have the privilege of a university education. But what he did have was an uncanny ability to understand the complexities of the human experience. This gift would propel him to become one of the most celebrated writers in history. Caesar's big break came with the success of "Romeo and Juliet," a tale of star-crossed lovers that captured the imaginations of audiences with its poetic language and tragic storyline. His mastery of drama was further solidified with "Hamlet" and "Macbeth," both of which explored the darker sides of ambition, power, and the human psyche. By the time these plays were staged, Caesar was already a household name.

- **Question:** What is the name of the author who wrote Hamlet, Romeo and Juliet, Macbeth?

- **Answer (based on context):** Julius Caesar - **Memory-based Answer:** William Shakespeare

## Cf_Quotes Examples

Example 1:
- **Context:** Write a quote that ends in the word "heavy": Absence makes the heart grow
- **Answer (based on context):** heavy. - **Memory-based Answer:** fonder.

Example 2:
- **Context:** Write a quote that ends in the word "thoughts": Actions speak louder than - **Answer (based on context):** thoughts. - **Memory-based Answer:** words.

B

## Examples Of Bad Context Reliance In Chatgpt

ChatGPT 01-mini

bu will be tagked with a series of reading comprehension question you are given an article with updated knowledge about the esson being asked. If the article contains the answer to your
, use that answer, even it conflicts with a nother iconic work. "Self-Portrait with Them N
ummingbird," pained in 1940, symbolizes her suffering and lience. The thorn necklace represents the pain she endured harmingbird, traditionally a symbol of go rican follore, hangs lifeltion around her neck. Kliml's w, demanding recognition Which Mixican artist B
Thought for 5 seconds
Figure 14: ChatGPT ol-mini fails to answer based on the context (Klimt) and instead uses answers based on its parametric knowledge (Kahlo), even when instructed explicitly to rely on the article.

ChatGPT 40

fou will be tasked with a series of reading comprehension question.

where you are green an article with updated knowledge that may on may not contain the answer to the question being asked. If the article contains the answer to your question, use that ans officts with a fact you know.

sick: DNA structure is charactorized by its dou where nucleotide bases pair specifically to mointain the stability and ty of genetic information. Track tional understan acknine (A) pairs with guarase (G) through two hydrogen bonds, le (C) pairs with thyrnine (T)
B. Thymini
« O U Q C ~
Figure 15: ChatGPT 4o fails to answer based on the context (guanine) and instead uses answers based on its parametric knowledge (thymine), even when instructed explicitly to rely on the article.