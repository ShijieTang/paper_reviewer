# Measuring And Enhancing Trustworthiness Of Llms In Rag Through Grounded Attributions And Learning To Refuse

Maojia Song*, Shang Hong Sim***, Rishabh Bhardwaj**
Singapore University of Technology and Design
{maojia song, shanghong sim, rishabh bhardwaj}@mymail.sutd.edu.sg Hai Leong Chieu DSO National Laboratories chaileon@dso.org.sg Navonil Majumder, Soujanya Poria Singapore University of Technology and Design
{navonil majumder, sporia}@sutd.edu.sg

## Abstract

LLMs are an integral component of retrieval-augmented generation (RAG) systems. While many studies focus on evaluating the overall quality of end-to-end RAG systems, there is a gap in understanding the appropriateness of LLMs for the RAG task. To address this, we introduce TRUST-SCORE, a holistic metric that evaluates the trustworthiness of LLMs within the RAG framework. Our results show that various prompting methods, such as in-context learning, fail to effectively adapt LLMs to the RAG task as measured by TRUST-SCORE. Consequently, we propose TRUST-ALIGN, a method to align LLMs for improved TRUST-SCORE performance. 26 out of 27 models aligned using TRUST-ALIGN substantially outperform competitive baselines on ASQA, QAMPARI, and ELI5. Specifically, in LLaMA-3-8b, TRUST-ALIGN outperforms FRONT on ASQA (↑12.56), QAM-
PARI (↑36.04), and ELI5 (↑17.69). TRUST-ALIGN also significantly enhances models' ability to correctly refuse and provide quality citations. We also demonstrate the effectiveness of TRUST-ALIGN across different open-weight models, including the LLaMA series (1b to 8b), Qwen-2.5 series (0.5b to 7b), and Phi3.5
(3.8b). We release our code at https://github.com/declare-lab/ trust-align.

## 1 Introduction

LLMs are widely used for information retrieval but often produce hallucinations—factually incorrect yet convincing responses (Ji et al., 2023), undermining their reliability. A common mitigation is Retrieval-Augmented Generation (RAG), which integrates external knowledge to improve correct token generation, reducing perplexity (Khandelwal et al., 2019) and enhancing downstream tasks like machine translation (Zheng et al., 2021) and classification (Bhardwaj et al., 2023). Connecting LLMs to external documents via retrieval also improves response quality (Shuster et al., 2021; Bechard & Ayala, 2024), further enhanced by attribution mechanisms (Gao et al., 2023b; Hsu et al., ´ 2024). In this paper, we investigate LLMs' ability to ground responses in provided documents instead of relying on their *parametric* knowledge from training. A response is considered grounded if it correctly answers using only the attached documents, with in-text citations supporting its claims. Key aspects include LLMs' *refusal* capability—whether they abstain from answering when documents lack sufficient information. Additionally, we analyze their overall tendency to answer, the fraction of claims grounded in documents, and whether cited sources substantiate generated statements. To comprehensively understand LLMs' groundedness, we propose a new metric TRUST-S**CORE**. It assesses an LLM across multiple dimensions: 1) The ability to discern which questions can be
∗Equal contribution.

1 answered or refused based on the provided documents (Grounded Refusals); 2) The correctness of LLM response for the answerable questions; 3) The extent to which generated statements are supported by the corresponding citations; and 4) The relevance of the citations to the statements. Unlike existing metrics that primarily assess the overall performance of RAG systems (Gao et al., 2023b)—where a weak retriever can significantly decrease the scores—TRUST-SCORE is designed to specifically measure the LLM's performance within a RAG setup, isolating it from the influence of retrieval quality. Our investigation in Section 6.1 shows that many state-of-the-art systems, including GPT-4 and Claude-3.5-Sonnet, heavily rely on their parametric knowledge to answer questions (OpenAI, 2023; Anthropic, 2024). This reliance limits their suitability for RAG tasks, where models should base responses solely on the provided documents, resulting in a low TRUST-SCORE. Additionally, prompting approaches intended to enhance model groundability have proven ineffective, as models become overly sensitive to the prompt, leading to exaggerated refusals or excessive responsiveness shown in Appendix F.4. To enhance the groundedness of LLMs, i.e., achieve a higher TRUST-SCORE, we propose an alignment method, TRUST-A**LIGN**. This approach first constructs an alignment dataset consisting of 19K questions, documents, positive (preferred) responses, and negative (unpreferred) responses. The dataset covers a range of LLM errors—Inaccurate Answers, Over-Responsiveness, Excessive Refusal, Over-Citation, and Improper Citation. We regard these errors as LLM hallucinations within an RAG framework.

Evaluations on the benchmark datasets ASQA, QAMPARI, and ELI5 show that models trained with TRUST-ALIGN outperform the competitive baselines on TRUST-SCORE in 26 out of 27 model family and dataset configurations. Notably, in LLaMA-3-8b, TRUST-ALIGN achieves substantial improvements over Huang et al. (2024b) FRONT, a leading baseline, with respective gains of 12.56% (ASQA), 36.04% (QAMPARI), and 17.69% (ELI5). Additionally, TRUST-ALIGN substantially enhances the ability of models to correctly refuse or provide grounded answers in all 27 model family and dataset configurations, with LLaMA-3-8b showing increases of 23.87%, 47.95%, and 45.77% correct refusals compared to FRONT. Citation groundedness scores also improved in 24 out of 27 model family and dataset configurations, with notable increases of 22.12%, 38.35%, and 5.55% in LLaMA-3-8b compared to FRONT. Due to the gamification of the metric, where parametric knowledge can artificially inflate the scores, we notice mixed results on answer correctness scores. Specifically, we observe a notable increase in answer correctness scores for all models in QAMPARI, 5/9 models in ELI5, and 2/9 models for ASQA. Our key contributions to this work are as follows: - We study LLM groundedness problem, where model model responses should be derived from retrieved documents (external memory) rather than the parametric knowledge (knowledge stored in model parameters).

- To measure LLM's groundedness under RAG, we introduce TRUST-S**CORE**, a holistic metric for quantifying LLM's grounding errors.

- We propose TRUST-A**LIGN**, an alignment approach designed to improve the trustworthiness of LLMs in RAG (Figure 2). It first creates an alignment dataset of 19K samples with paired positive and negative responses, followed by aligning the model using direct preference optimization (DPO) (Rafailov et al., 2024b).

Comparison with existing approaches. Current evaluations of RAG focus on the overall system performance (Gao et al., 2023b; Xu et al., 2024), conflating the effects of retriever quality and LLM performance in the metric scores (Fan et al., 2024). This highlights the need for new ways to measure LLM effectiveness in RAG systems without the influence of the retriever. The work by Thakur et al. (2024) is closest to ours, as it analyzes the refusal capabilities of LLMs in a RAG context but lacks holistic evaluation, as it does not account for both response and citation groundedness. On the other hand, Ye et al. (2024); Hsu et al. (2024); Huang et al. (2024b) propose frameworks to improve LLM response groundedness but overlook refusal behaviors in their metrics. Ignoring refusal behaviors, retriever influence, citation and answer groundedness weakens the ability of current metrics to effectively measure LLM performance in RAG. TRUST-SCORE comprehensively evaluates LLM performance, including refusal, citation, and answer groundedness, while TRUST-ALIGN creates a corresponding alignment dataset, making the metric and approach more unique and holistic for LLM evaluations and alignment in RAG. A more detailed comparison can be found in Appendix C.

## 2 Problem Description 2.1 Task Setup

Given a question q and a set of retrieved documents D as input, the LLM is instructed to generate a response S which consists of a set of citation-grounded statements {s1*, . . . , s*n}; each statement si follows a set of inline citations Ci = {ci,1, ci,2*, . . .*} referring to the documents in D. If D is not sufficient to answer q, the gold response would be a refusal statement1, such as, "I apologize, but I couldn't find an answer to your question in the search results". Otherwise, the response would follow the pattern: "statement1 [1][2] statement2 [3]" where [1][2] and [3] denote the enumeration of documents that supports each statement respectively.

Trust-Score ResponseTruthfulness **AttributionGroundedness**
GroundedRefusals (F1GR ) AnswerCorrectness (F1AC) GroundedCitations (F1GC)
Set of answered questions (statements only, no citations / citations only, no statements)
Set of answered questions Set of answerable questions/ Number of elements in the set Statement and corresponding citations Answer correctness for question Citation recall for statement Citation precision for citation

## 2.2 On Answerability Of A Question

To label if a response should be a refusal or consist of claims, we define the notion of *answerability*.

A question q is considered answerable if D contains sufficient information to answer q. Formally, we label a question as answerable if a subset of the retrieved documents entails at least one of the gold claims; otherwise, q is unanswerable and thus should result in a ground truth *refusal*. A
refusal response contains no claims or citations but provides a generic message conveying the LLM's inability to respond to q.

## 2.3 Hallucination In Llm In Rag

We define an LLM's response as grounded when it correctly answers a question using only the information in the documents, and the response can be inferred from the inline citations to those documents. When a response is not grounded, it is considered a case of hallucination. We define hallucination as an erroneous LLM response, categorized into five types: (1) *Inaccurate Answer* –
The generated statements S fail to cover the claims in the gold response, (2) *Over-Responsiveness* - The model answers a question that should result in a refusal, (3) *Excessive Refusal* - The model refuses to answer a question that is answerable, (4) *Overcitation* - The model generates redundant citations, and (5) *Improper Citation* - The citations provided do not support the statement. Next, we introduce a comprehensive metric to concretely measure hallucinations in LLMs, i.e., to assess an LLM's groundedness or trustworthiness2.

## 3 Metrics For Llm-In-Rag

Given a question q and the corresponding ground truth response AG = {ag1*, . . . , a*gn} consisting of gold claims, we define the claims obtainable from the provided documents as AD = {ad1*, . . . , a*dn} and the claims generated in the response as AR = {ar1*, . . . , a*rn}. We aim to measure two aspects of an LLM in RAG: 1) the correctness of the generated claims (Response Truthfulness); and 2) the correctness of citations generated (Attribution Groundedness).

Insufficiency of the existing metrics. Gao et al. (2023b) measure Response Truthfulness by first computing the per-sample Answer Correctness recall (ACq reg) score for gold claims AG, disregarding how many of these claims are obtainable from D. This is followed by averaging the recall scores across samples to obtain a single score for the dataset. This method introduces inconsistencies:
models that rely on parametric knowledge (Mp) may generate gold claims not found in D, leading to an artificially inflated recall value. In contrast, an ideal LLM (Mi) would rely solely on D to generate responses (a desired trait) and would be constrained by an upper recall limit of |AG∩AD| |AG|, which varies depending on the question. This approach presents two key problems: (1) Recall Consolidation: Since the measurement range depends on the claims present in D, it is infeasible to provide a consistent, consolidated ACreg score across the dataset, (2) *Recall Gamification*: Mp may have a higher upper limit on ACreg (up to 1) because they can generate gold claims not present in D (an undesirable trait), unlike Mithat depend entirely on D.

Answer Calibration. To address the challenges of recall consolidation and gamification in existing evaluation metrics, we propose new metrics that measure sample-wise recall score based on the fraction of gold claims ontainable from D. Specifically, this involves computing |AG ∩ AD|, which measures the Answer Correctness (AC) recall after calibrating the gold claims. This approach sets a maximum recall limit of 1 for all models. For dataset-wide scoring, we consolidate per-sample AC
scores using two methods: 1) PAC: The average AC score across samples *answered* by the LLM, i.e., samples where AR ̸= ∅, reflecting a precision oriented perspective; 2) RAC: The average AC score across samples that are *answerable*, i.e., samples where AG ∩ AD ̸= ∅, reflecting a recall oriented perspective3. These metrics, illustrated in Fig. 1, are then combined into a single score, F1AC, which serves as a comprehensive measure of how well the LLM grounds its claims on the document D. This combined metric not only facilitates the consolidation of recall but also addresses issues related to recall gamification. Scoring refusals. An important capability of an LLM in RAG is its ability to identify when a response is unanswerable based on the provided documents D. To measure this, we introduce a metric called Grounded Refusals. This metric evaluates the model's refusal performance by calculating dataset-wide precision and recall for both ground-truth answerable cases and refusals. These values are then combined into their respective F1 scores, F1ref for refusals and F1ans for answerable cases. The final score, F1GR, is the average of these two F1 scores, as shown in Figure 1. Measuring attribution groundedness. While Response Truthfulness metrics like F1AC and F1GR
evaluate the quality of generated claims, it is equally important to measure how well these statements are supported by relevant citations—what we call Attribution Groundedness. To this end, we adopt two sub-metrics from (Gao et al., 2023b): Citation Recall (R**cite**) and Citation Precision (P**cite**). To compute R**cite**, we first determine if a generated statement siis supported by its cited documents using an NLI model4, thus obtaining sample-wise recall scores R**cite**
si. Then we take the mean across all samples to obtain the final R**cite** score (Figure 1). To compute P**cite**, we first score each citation ci,j of a statement si, followed by computing the average across citations in a response S
(sample-wise score). The dataset-wide citation score is computed by averaging the citation scores across all the samples. To quantify the Groundedness of Citations, we compute F1GC, the harmonic mean of P**cite** and R**cite**. A detailed breakdown of this metric is provided in Appendix D and Figure 1.

Thus, we define a new metric TRUST-S**CORE** =
1 3
(F1GR + F1AC + F1GC).

Responsiveness. To measure the answering tendency of an LLM, we define **Responsiveness**. It is the fraction of answered questions, denoted by the Answered Ratio (AR %), which is calculated as AR % =|Ar| |Ag|+|¬Ag|
. |Ar|, |Ag|, and |¬Ag| are the number of answered, answerable, and unanswerable questions respectively. A model is expected to show a high AR% for answerable questions and a low AR% for unanswerable ones, with the scores expected to align with the dataset distribution.

OveJohansson
(Gold Claim1)
1 4 Filter top~3k knowledge intensive questions SeedPromptCuration PositiveAnswerGeneration Questions Documents SeedSet Retrievetop100 docsusing Wikipedia, Sphere GPT-4 synthesizer
...MattPrater at64yards
[Gold Claim2], ...Ove Johanssonina 1976
...[Gold Claim1].

...MattPrater at64 yards [1][3], ...Ove Johanssonina 1976

...[2][4].

AnswerableQuestions ASQA
ELI5 QAMPARI
Questions 10k PositiveAnswer (r )
+

Greedilyfind thesmallest subsetof documents that support claimusing TRUENLI
MattPrater
(Gold Claim2)

Top-100 documents UnanswerableQuestions PositiveAnswer: "I apologize,but I couldn't findananswer toyourquestioninthe searchresults."
5 NegativeAnswerGeneration Goldclaim3 Answerability **Labelling**
Doc 1

[1,1,0]
Doc 2
[1,1,0]

Doc3
[1,0,0]

Doc4

[1,0,0]

Doc5
[0,0,0]

[1,1,0] Union Answerable 2 SupervisedFinetuning Questions OracleDocs PositiveAns SeedSet Repeat for k=5 docs Goldclaim1 Doc 1 LLaMA-2-7b70k Responses Calculate Hallucination Score Filter top-50%
Goldclaim2 Negative Answer (r )
-

Only40khave~19k Aug.Set Questions Setof5Docs UseTRUENLI tocheck ifdoc supports goldclaim Inference SFT
Augmented Prompt **Curation**
Top-100 Docs**Combination**
of5docs Question Combination of5docs Question 3 70k Questions Documents Aug.Set Finddifferent setsof5docs covering distinct subset of claims

...

Alignment 6 Question x7 DirectPreference Optimization 19k Questions Setof 5Documents PositiveAnswer A
NegativeAnswer ug.S
et DPO- LLaMA
Trainable ParametersFrozen ParametersSeed SetAugmented Set**Document**

## 4 The Trust-Align Dataset

To align LLMs towards trustworthiness, we propose a new approach, TRUST-A**LIGN**. The approach constructs an LLM trustworthiness alignment dataset, where each sample in the dataset consists of a question q, a set of retrieved documents D, and a pair of positive (preferred) and negative (unpreferred) responses (r
+, r
−). The positive response corresponds to an answer that encompasses expected gold claims for q and corresponding citations referring to the documents. If D is not sufficient to answer q, r
+ is assigned a refusal response, while r
− is its non-refusal counterpart. We build the dataset in multiple steps: 1) Obtain a set of high-quality and diverse questions, 2) Obtain documents for each question, 3) Augmenting (*q, D*) pairs that cover diverse hallucination types, 4) Construct positive responses entailing gold claims, and 5) Construct negative (unpreferred) responses by prompting a fine-tuned model and observing its hallucinations. We relegate fine-grained details about the dataset to Figure 2 and Appendix E. Collecting quality questions. The dataset construction begins by collecting a set of high-quality and diverse questions from the training splits of ASQA, QAMPARI, and ELI5, referred to as **seed**
samples. We first divide the questions into k clusters and use Mixtral-8x7B to assign each a quality score from 1 to 7, based on how difficult they are to answer without additional information. Clusters with scores of 4 or higher are selected. Next, we sample questions from the clusters of each dataset to construct approximately 10K questions in the seed set. Collecting D's. Next, we collect relevant documents for each question in the seed set by querying Wikipedia and Common Crawl, retrieving the top 100 documents. We filter out seed questions where relevant documents are not retrieved. We then identify 5 documents that perform as well as the full 100 in terms of EM recall, referring to these as *oracle* documents for question q.

5 Gold claims for each q are sourced from the respective datasets. Augmenting *(q,D)* **set.** Using the questions and oracle documents, we create diverse samples (i.e., varying combinations of relevant and irrelevant documents) to trigger multiple hallucinations from LLMs (Section 2.3). The document order is shuffled to avoid citation bias. To construct unanswerable questions, we select documents similar to those entailing gold claims but still irrelevant to q. This process results in approximately 70K question-document pairs. Obtaining r
+ and r
−. To generate preferred responses, by prompting GPT-4, we stitch together the gold claims and citations6. For unanswerable questions, we assign a ground truth refusal response. To obtain quality negative (unpreferred) responses, we fine-tune LLaMA-2-7b on the source datasets, creating M*sf t*. Testing M*sf t* on the 70K dataset identified 40K responses with hallucinations. Table 1 shows hallucination severity (ei) and frequency (wi). To obtain good negative samples, we first rank each of the 40K responses according to their severity score eq, where eq =Pi ei· wi. We then select the top 50%7 of the corresponding samples for both answerable and unanswerable responses. We perform DPO using this set of 19k samples to obtain the final aligned model.

## 5 Experimental Setup

Models studied. To comprehensively measure performance of open-source models, we perform TRUST-SCORE computations on vanilla and TRUST-ALIGNed version of a range of open-weight models such as LLaMA series (LLaMA-2-7b, LLaMA-2-13b, LLaMA-2-13b, etc.), Qwen series (Qwen-2.5-0.5b, Qwen-2.57b, etc.) and Phi3.5-mini. See Appendix H.1 for more details.

Table 1: Fraction of each hallucination amongst all the

observed hallucinations in Msft (40,985), with possible overlap. wi shows the severity computation of each hallucination. Icondition = 1 if condition is True otherwise it is 0. See Fig. 5 for the detailed breakdown of the last three errors.

Hallucination Type (HT) Frequency (wi**) Severity (**ei)

Unwarranted Refusal 8,786 0.50 I(Ag̸=∅,Ar=∅)

Over Responsiveness 13,067 0.50 I(Ag=∅,Ar̸=∅) Overcitation 12,656 0.34 1 - CP Improper Citation 9,592 0.26 1 - CR Inaccurate Claims 14,783 0.40 1 - F1AC

Evaluation datasets. We evaluate on the test-set of attributable factoid and long-form question-answering tasks from ASQA (Stelmakh et al., 2023), QAMPARI (Amouyal et al., 2023), and ELI5 (Fan et al., 2019). Additionally, we include ExpertQA (Malaviya et al., 2024) for OOD evaluations. For each question, we append the top 5 retrieved documents. For ELI5 and ExpertQA, the ground truth answers are decomposed into three claims. The dataset statistics are detailed in Appendix H.2.

Baselines. Models8trained with TRUST-ALIGN are compared against the following baselines:
- ICL (Gao et al., 2023b): Prepends two demonstrations to each query, consisting of an example query, top-5 retrieved documents, and an inline cited answer
- PostCite (Gao et al., 2023b): Generates an uncited answer in a closed-book setting, then retrieves most similar documents from top-5 documents using GTR for citations.

- PostAttr (Gao et al., 2023b): Similar to POSTCITE, produces an uncited response in a closed-book setting, but uses the TRUE-NLI model to find the best matching citation among top-5 documents.

- Self-RAG (Asai et al., 2024): Trains the LLM to retrieve relevant documents on demand using reflection tokens, enhancing generation quality. We evaluated the provided 7b and 13b model checkpoints from HF using the default settings.

- FRONT (Huang et al., 2024b): Uses a fine-grained attribution framework to improve grounding and citation quality. We followed the provided instructions to train a 7b model for comparison.

5Clustering and document retrieval details are in Appendix E. 6Prompt template can be found at Table 23. 7See Appendix F.8 for more details on this hyperparameter. 8All models used are instruct tuned or chat versions.

## 6 Results And Analysis

Table 2: LLaMA family evaluated on the ASQA, QAMPARI, and ELI5 datasets. Best values within each family are highlighted ). AR% := Answered Ratio in %; F1AC := Answer Correctness F1; F1GR := Grounded Refusals F1; F1GC := Grounded Citations F1; **TRUST** := TRUST-SCORE; **Resp.**
:= Responsiveness; **Att-Grd.** := Attribution Groundedness.

| (610 answerable, 338 unanswerable)   | QAMPARI (295 answerable, 705 unanswerable)                                             | ELI5 (207 answerable, 793 unanswerable)   |                 |           |                   |                   |                 |            |             |             |             |             |             |       |       |
|--------------------------------------|----------------------------------------------------------------------------------------|-------------------------------------------|-----------------|-----------|-------------------|-------------------|-----------------|------------|-------------|-------------|-------------|-------------|-------------|-------|-------|
| Model                                | Type                                                                                   | ASQA Resp.                                | Trustworthiness | Resp.     | Trustworthiness   | Resp.             | Trustworthiness |            |             |             |             |             |             |       |       |
| AR (%) Truthfullness                 | Att-Grd. TRUST AR (%) Truthfullness Att-Grd. TRUST AR (%) Truthfullness Att-Grd. TRUST |                                           |                 |           |                   |                   |                 |            |             |             |             |             |             |       |       |
| F1AC F1GR                            | F1GC                                                                                   | F1AC F1GR                                 | F1GC            | F1AC F1GR | F1GC              |                   |                 |            |             |             |             |             |             |       |       |
| ICL                                  | 0.00                                                                                   | 0.00                                      | 26.28           | 0.00      | 8.76              | 0.00              | 0.00            | 41.35      | 0.00        | 13.78       | 0.50        | 0.00        | 46.71       | 0.00  | 15.57 |
| PostCite                             | 10.44                                                                                  | 0.07                                      | 35.23           | 0.00      | 11.77             | 34.40             | 0.00            | 57.34      | 9.50        | 22.28       | 0.90        | 1.86        | 44.98       | 5.04  | 17.29 |
| PostAttr                             | 10.44                                                                                  | 0.07                                      | 35.23           | 0.00      | 11.77             | 34.40             | 0.00            | 57.34      | 3.78        | 20.37       | 0.90        | 1.86        | 44.98       | 0.00  | 15.61 |
| Self-RAG                             | 100.00                                                                                 | 45.19                                     | 39.15           | 63.49     | 49.28             | 96.00             | 6.81            | 28.23      | 19.95       | 18.33       | 73.50       | 14.94 40.20 | 13.80       | 22.98 |       |
| FRONT                                | 100.00 60.47                                                                           | 39.15                                     | 68.86           | 56.16     | 100.00            | 17.27 22.78       | 24.26           | 21.44      | 100.00      | 21.66 17.15 | 52.72       | 30.51       |             |       |       |
| TRUST-ALIGN (DPO)                    | 65.30                                                                                  | 52.48 66.12                               | 83.94           | 67.51     | 32.30 32.03 71.67 | 49.42             | 51.04           | 21.60      | 22.54 63.27 | 47.35       | 44.39       |             |             |       |       |
| LLaMA-2 -7b                          | ICL                                                                                    | 17.41                                     | 21.52           | 41.40     | 13.83             | 25.58             | 26.50           | 0.44 59.57 | 0.00        | 20.00       | 46.40       | 19.97 54.81 | 4.73        | 26.50 |       |
| PostCite                             | 90.51                                                                                  | 2.21 49.91                                | 1.53            | 17.88     | 100.00            | 0.00              | 22.78           | 8.05       | 10.28       | 76.60       | 2.27        | 38.05       | 0.72        | 13.68 |       |
| LLaMA-2 -13b                         | PostAttr                                                                               | 90.51                                     | 2.21 49.91      | 0.17      | 17.43             | 100.00            | 0.00            | 22.78      | 2.95        | 8.58        | 76.60       | 2.27        | 38.05       | 0.09  | 13.47 |
| Self-RAG                             | 100.00 48.52                                                                           | 39.15                                     | 69.79           | 52.49     | 72.70             | 2.71              | 48.58           | 26.91      | 26.07       | 22.10       | 12.77 58.68 | 24.54       | 32.00       |       |       |
| ICL                                  | 60.23                                                                                  | 35.95                                     | 50.94           | 9.96      | 32.28             | 19.20             | 6.32            | 52.64      | 0.38        | 19.78       | 88.40       | 12.87 27.10 | 5.23        | 15.07 |       |
| PostCite                             | 43.57                                                                                  | 0.59                                      | 50.22           | 0.24      | 17.02             | 41.20             | 0.32            | 49.79      | 1.61        | 17.24       | 18.40       | 2.04        | 50.88       | 1.02  | 17.98 |
| PostAttr                             | 45.78                                                                                  | 0.48                                      | 48.42           | 0.00      | 16.30             | 34.00             | 0.63            | 48.43      | 0.21        | 16.42       | 18.40       | 2.04        | 50.88       | 0.07  | 17.66 |
| LLaMA-3.2 FRONT                      | 79.11 48.22                                                                            | 54.48                                     | 48.29           | 50.33     | 98.60             | 7.57              | 24.54           | 15.32      | 15.81       | 97.20       | 16.11 20.76 | 30.19       | 22.35       |       |       |
| -1b                                  | TRUST-ALIGN (DPO)                                                                      | 41.67                                     | 38.64 58.61     | 79.35     | 58.87             | 20.00 27.22 67.92 | 49.42           | 48.19      | 9.60        | 13.20 59.35 | 48.21       | 40.25       |             |       |       |
| ICL                                  | 1.27                                                                                   | 2.04                                      | 27.98           | 53.95     | 27.99             | 34.10             | 16.06 59.65     | 12.87      | 29.53       | 21.90       | 18.55 55.56 | 30.70       | 34.94       |       |       |
| PostCite                             | 47.26                                                                                  | 31.03                                     | 56.59           | 22.99     | 36.87             | 39.60             | 6.34            | 55.22      | 6.83        | 22.80       | 92.80       | 18.12 25.14 | 4.44        | 15.90 |       |
| PostAttr                             | 47.15                                                                                  | 29.76                                     | 56.71           | 4.69      | 30.39             | 42.00             | 5.10            | 53.74      | 0.27        | 19.70       | 92.80       | 18.48 25.14 | 0.53        | 14.72 |       |
| LLaMA-3.2 FRONT                      | 95.25 63.19                                                                            | 49.45                                     | 57.46           | 56.70     | 92.70             | 12.99 32.89       | 19.19           | 21.69      | 86.90       | 19.95 32.21 | 41.97       | 31.38       |             |       |       |
| -3b                                  | TRUST-ALIGN (DPO)                                                                      | 77.85                                     | 59.82 66.38     | 84.21     | 70.14             | 48.20 29.13 70.85 | 45.65           | 48.54      | 17.50       | 18.33 62.79 | 55.87       | 45.66       |             |       |       |
| ICL                                  | 1.48                                                                                   | 3.01                                      | 28.58           | 86.50     | 39.36             | 3.90              | 5.92            | 48.60      | 20.24       | 24.92       | 0.00        | 0.00        | 44.23       | 0.00  | 14.74 |
| PostCite                             | 77.53                                                                                  | 32.98                                     | 53.31           | 28.01     | 38.10             | 87.00             | 6.10            | 34.52      | 8.42        | 16.35       | 62.00       | 20.80 45.88 | 8.06        | 24.91 |       |
| LLaMA-3                              | PostAttr                                                                               | 77.53                                     | 32.98           | 53.31     | 5.95              | 30.75             | 87.00           | 6.10       | 34.52       | 1.64        | 14.09       | 62.00       | 20.80 45.88 | 1.25  | 22.64 |
| -8b                                  | FRONT                                                                                  | 99.05 62.25                               | 41.62           | 66.14     | 56.67             | 100.00            | 13.53 22.78     | 20.42      | 18.91       | 99.50       | 18.99 17.85 | 44.69       | 27.18       |       |       |
| TRUST-ALIGN (DPO)                    | 56.43                                                                                  | 53.94 65.49                               | 88.26           | 69.23     | 22.40 35.35 70.73 | 58.77             | 54.95           | 15.50      | 20.81 63.57 | 50.24       | 44.87       |             |             |       |       |

TRUST-ALIGN **boosts trustworthiness over baseline methods.** As shown in Table 2 and Table 3, TRUST-ALIGNed models demonstrate substantial improvements on TRUST-SCORE over the baselines in 26 out of 27 model family and dataset configurations. Specifically, with LLaMA-3-8b, TRUST-ALIGN outperforms FRONT by 12.56% (ASQA), 36.04% (QAMPARI), and 17.69% (ELI5) on TRUST-SCORE. This suggests that TRUST-ALIGNed models are more capable of generating responses grounded in the documents. TRUST-ALIGN **improves models' refusal capability.** Across all 27 configurations, TRUST- ALIGN yields substantial improvements in F1GR. In LLaMA-3-8b, TRUST-ALIGN outperforms FRONT by 23.87% (ASQA), 47.95% (QAMPARI), and 45.72% (ELI5). This indicates that TRUST-
ALIGN substantially enhances models' ability to correctly refuse or provide answers.

TRUST-ALIGN **enhances models' citation quality.** F1GC is substantially improved over baselines in 24 out of 27 model family and dataset configurations after the application of TRUST-ALIGN.

Specifically, with LLaMA-3-8b, TRUST-ALIGN outperforms FRONT on F1GC by 22.12% (ASQA),
38.35% (QAMPARI), and 5.55% (ELI5). This demonstrates that aligning with TRUST-ALIGN improves the model's ability to provide citations that sufficiently and precisely support claims. TRUST-ALIGN **has mixed effects on F1**AC. We observe that applying TRUST-ALIGN yields a notable increase in F1AC for QAMPARI (9/9) but mixed performance on ELI5 (5/9) and ASQA (2/9). The mixed performance in ASQA and ELI5 can be explained by the composition of F1AC, which is derived from PAC and RAC (Eq. (9)). Taking LLaMA-3.2-3b on ASQA as an example (Appendix I), TRUST-ALIGN models generally achieve higher PAC compared to baselines (54.63% for TRUST-ALIGN vs. 52.94% for FRONT)
despite having a lower AR% (77.85% for TRUST-ALIGN vs. 95.25% for FRONT). This suggests that our models have a higher expected value for ACq(per-sample AC recall), as the denominator depends on the number of answered questions. This trend is observed across models and datasets.

However, in ASQA and ELI5, our models underperform in F1AC due to the overwhelmingly adverse impact of RAC. The recall of answerable questions (Rans) is lower for our model compared to baselines (89.02% for TRUST-ALIGN vs. 98.69% for FRONT), which rarely refuse questions. As a result, fewer terms are summed in the numerator of RAC, while the denominator remains constant (the number of answerable questions). This leads to a lower overall F1AC score. To further analyze the baseline models' performance, we investigated how much of their answering ability relies on parametric knowledge versus document-based information (Section 6.1 and Appendix F.3).

| (610 answerable, 338 unanswerable)   | QAMPARI (295 answerable, 705 unanswerable)         | ELI5 (207 answerable, 793 unanswerable)   |                 |           |                   |                   |                 |             |             |             |             |             |       |       |       |
|--------------------------------------|----------------------------------------------------|-------------------------------------------|-----------------|-----------|-------------------|-------------------|-----------------|-------------|-------------|-------------|-------------|-------------|-------|-------|-------|
| Model                                | Type                                               | ASQA Resp.                                | Trustworthiness | Resp.     | Trustworthiness   | Resp.             | Trustworthiness |             |             |             |             |             |       |       |       |
| AR (%) Truthfullness                 | Att-Grd. TRUST AR (%) Truthfullness Att-Grd. TRUST | AR (%) Truthfullness Att-Grd. TRUST       |                 |           |                   |                   |                 |             |             |             |             |             |       |       |       |
| F1AC F1GR                            | F1GC                                               | F1AC F1GR                                 | F1GC            | F1AC F1GR | F1GC              |                   |                 |             |             |             |             |             |       |       |       |
| ICL                                  | 29.85                                              | 20.96                                     | 47.19           | 0.35      | 22.83             | 11.40             | 2.45            | 50.67       | 0.00        | 17.71       | 82.30       | 13.73 33.14 | 0.37  | 15.75 |       |
| PostCite                             | 46.10                                              | 8.55                                      | 50.84           | 8.23      | 22.54             | 17.00             | 0.67            | 52.51       | 5.72        | 19.63       | 89.80       | 9.87        | 27.10 | 4.10  | 13.69 |
| PostAttr                             | 46.10                                              | 8.55                                      | 50.84           | 2.23      | 20.54             | 17.00             | 0.67            | 52.51       | 0.90        | 18.03       | 89.80       | 9.87        | 27.10 | 0.68  | 12.55 |
| Qwen-2.5 FRONT                       | 100.00                                             | 42.83                                     | 39.15           | 45.87     | 42.62             | 99.30             | 11.52 23.23     | 15.90       | 16.88       | 99.90       | 13.74 17.29 | 27.95       | 19.66 |       |       |
| -0.5b                                | TRUST-ALIGN (DPO)                                  | 71.84                                     | 50.59 61.28     | 52.40     | 54.76             | 17.90 15.76 61.84 | 29.73           | 35.78       | 21.70       | 13.68 60.79 | 22.72       | 32.40       |       |       |       |
| ICL                                  | 98.52                                              | 50.55                                     | 41.74           | 6.69      | 32.99             | 85.00             | 15.60 41.27     | 8.61        | 21.83       | 99.40       | 20.56 17.78 | 4.99        | 14.44 |       |       |
| PostCite                             | 71.73                                              | 16.36                                     | 52.46           | 15.40     | 28.07             | 11.20             | 3.44            | 51.11       | 13.95       | 22.83       | 91.50       | 15.63 26.71 | 5.17  | 15.84 |       |
| PostAttr                             | 71.73                                              | 16.36                                     | 52.46           | 4.45      | 24.42             | 11.20             | 3.44            | 51.11       | 1.07        | 18.54       | 91.50       | 15.63 26.71 | 0.62  | 14.32 |       |
| Qwen-2.5 FRONT                       | 99.26                                              | 57.74                                     | 41.36           | 55.70     | 51.60             | 98.80             | 16.05 24.45     | 11.60       | 17.37       | 99.90       | 19.57 17.29 | 37.70       | 24.85 |       |       |
| -1.5b                                | TRUST-ALIGN (DPO)                                  | 72.57                                     | 52.68 62.38     | 66.81     | 60.62             | 20.00 23.80 68.46 | 50.98           | 47.75       | 33.60       | 19.03 57.91 | 31.63       | 36.19       |       |       |       |
| ICL                                  | 27.43                                              | 37.72                                     | 51.36           | 51.72     | 46.93             | 22.30             | 23.17 63.27     | 41.20       | 42.55       | 68.80       | 29.12 46.31 | 34.34       | 36.59 |       |       |
| PostCite                             | 8.76                                               | 9.58                                      | 35.30           | 10.94     | 18.61             | 0.10              | 0.00            | 41.31       | 0.00        | 13.77       | 49.70       | 21.73 48.49 | 7.56  | 25.93 |       |
| Qwen-2.5 PostAttr                    | 8.76                                               | 9.58                                      | 35.30           | 36.29     | 27.06             | 0.10              | 0.00            | 41.31       | 25.00       | 22.10       | 49.70       | 21.73 48.49 | 1.31  | 23.84 |       |
| -3b                                  | FRONT                                              | 97.47                                     | 55.15           | 44.01     | 62.72             | 53.96             | 79.10           | 20.69 48.62 | 25.67       | 31.66       | 93.60       | 18.69 25.37 | 37.40 | 27.15 |       |
| TRUST-ALIGN (DPO)                    | 49.47                                              | 55.19 63.76                               | 78.64           | 65.86     | 48.10 35.69 70.31 | 45.64             | 50.55           | 13.50       | 22.52 64.38 | 42.01       | 42.97       |             |       |       |       |
| ICL                                  | 92.09                                              | 58.94                                     | 54.34           | 75.46     | 62.91             | 56.30             | 28.92 63.67     | 39.28       | 43.96       | 82.70       | 28.27 37.13 | 44.13       | 36.51 |       |       |
| PostCite                             | 91.46                                              | 27.52                                     | 45.93           | 4.19      | 25.88             | 26.70             | 8.59            | 60.16       | 1.05        | 23.27       | 95.60       | 21.82 22.23 | 7.03  | 17.03 |       |
| PostAttr                             | 91.46                                              | 27.52                                     | 45.93           | 17.92     | 30.46             | 26.70             | 8.59            | 60.16       | 13.55       | 27.43       | 95.60       | 21.82 22.23 | 0.96  | 15.00 |       |
| FRONT                                | 86.39                                              | 64.58                                     | 60.08           | 58.27     | 60.98             | 84.70             | 17.02 42.85     | 24.48       | 28.12       | 57.60       | 28.27 54.14 | 56.61       | 46.34 |       |       |
| TRUST-ALIGN (DPO)                    | 59.49                                              | 55.04 66.22                               | 83.57           | 68.28     | 32.10 30.11 70.68 | 53.48             | 51.42           | 21.00       | 24.30 63.79 | 47.02       | 45.04       |             |       |       |       |
| Qwen-2.5 -7b ICL                     | 63.19                                              | 50.24                                     | 51.95           | 42.64     | 48.28             | 70.20             | 11.91 43.90     | 12.26       | 22.69       | 81.50       | 27.59 37.17 | 30.14       | 31.63 |       |       |
| PostCite                             | 23.10                                              | 14.98                                     | 41.38           | 9.40      | 21.92             | 76.90             | 3.57            | 42.36       | 4.49        | 16.81       | 84.50       | 20.50 30.81 | 4.67  | 18.66 |       |
| PostAttr                             | 23.10                                              | 14.98                                     | 41.38           | 1.24      | 19.20             | 76.90             | 3.57            | 42.36       | 0.46        | 15.46       | 84.50       | 21.26 30.81 | 0.68  | 17.58 |       |
| FRONT                                | 99.79                                              | 63.30                                     | 39.79           | 71.63     | 58.24             | 100.00            | 11.97 22.78     | 21.50       | 18.75       | 96.60       | 21.46 21.35 | 61.41       | 34.74 |       |       |
| TRUST-ALIGN (DPO)                    | 66.56                                              | 52.23 64.20                               | 85.36           | 67.26     | 30.10 36.42 73.95 | 53.40             | 54.59           | 24.90       | 23.39 67.62 | 47.42       | 46.14       |             |       |       |       |
| Phi3.5 -mini                         |                                                    |                                           |                 |           |                   |                   |                 |             |             |             |             |             |       |       |       |

TRUST-ALIGN **generalizes across model families and sizes.** Table 3 demonstrates that TRUST- ALIGN improves the models' TRUST-SCORE across various sizes and architectures. In small models like Qwen-2.5-0.5b, TRUST-ALIGN significantly outperforms ICL baselines, achieving notable gains in ASQA (22.83% → 54.76%). Similarly, for larger models such as Qwen-2.5-7b, TRUST- ALIGN delivers substantial improvements, as seen in ASQA (62.91% → 68.28%), highlighting its scalability. The largest gains are observed in smaller models; for example, Phi3.5-mini shows remarkable improvements over ICL: 18.98% (ASQA), 31.90% (QAMPARI), and 14.51% (ELI5). Models aligned with DPO generally outperform those trained with SFT. Table 4 shows that DPO models outperform SFT models on TRUST-SCORE in 26 out of 27 model family and dataset configurations. In LLaMA-3.2-3b, DPO yields substantial improvements on ASQA (6.70%), QAM-
PARI (3.09%), and ELI5 (1.71%). Additionally, DPO models also attain substantially better F1GC
compared to SFT on 25 out of 27 configurations, with substantial improvements on ASQA (8.58%), QAMPARI (7.62%), and ELI5 (2.54%) for LLaMA-3.2-3b. This highlights DPO's effectiveness in enhancing citation quality. While results on F1AC and F1GR are mixed, DPO yields better overall TRUST-SCORE scores.

## 6.1 Analysis

Data ablation. Table 5 shows that adding samples targeting each of the five hallucination types improves TRUST-SCORE by 1.50% (ASQA), 1.78% (QAMPARI), and 2.23% (ELI5). We observe that removing data corresponding to each hallucination type causes a notable decrease in TRUST- SCORE, suggesting the importance of each subtype. In particular, removing refusal-related hallucinations adversely affects F1GR: ↓2.79% (ASQA), ↓0.48% (QAMPARI), underscoring the importance of incorporating refusal-related data to improve a model's ability to discern when to provide an answer. Table 4: Performance of models with only SFT applied as compared to TRUST-ALIGN models. Best values within each family are **bolded**).

Model Type**ASQA** 

(610 answerable, 338 unanswerable) **QAMPARI** *(295 answerable, 705 unanswerable)* **ELI5** *(207 answerable, 793 unanswerable)*

Resp. Trustworthiness Resp. Trustworthiness Resp. Trustworthiness

AR (%) Truthfullness Att-Grd. TRUST AR (%) Truthfullness Att-Grd. TRUST AR (%) Truthfullness Att-Grd. **TRUST**

F1AC F1GR F1GC F1AC F1GR F1GC F1AC F1GR F1GC

LLaMA-2

-7bSFT 80.17 **53.21** 63.43 79.61 65.42 31.60 **33.76** 71.13 46.37 50.42 29.50 21.58 **63.30** 39.59 41.49

TRUST-ALIGN (DPO) 65.30 52.48 66.12 83.94 **67.51** 32.30 32.03 71.67 49.42 **51.04** 21.60 **22.54** 63.27 47.35 **44.39**

LLaMA-3.2

-1bSFT 63.82 **45.61 63.91** 73.10 **60.87** 26.00 **27.98 68.20** 37.96 44.71 20.50 **14.56 63.93** 37.28 38.59

TRUST-ALIGN (DPO) 41.67 38.64 58.61 **79.35** 58.87 20.00 27.22 67.92 49.42 **48.19** 9.60 13.20 59.35 48.21 **40.25**

LLaMA-3.2

-3bSFT 68.04 49.23 65.47 75.63 63.44 27.60 28.09 70.22 38.03 45.45 14.70 15.92 62.59 53.33 43.95

TRUST-ALIGN (DPO) 77.85 59.82 66.38 84.21 **70.14** 48.20 29.13 70.85 45.65 **48.54** 17.50 18.33 62.79 55.87 **45.66**

LLaMA-3

-8bSFT 68.99 52.35 **66.06** 80.95 66.45 24.20 **33.85 71.11** 48.01 50.99 23.60 **22.57 65.06** 46.85 44.83

TRUST-ALIGN (DPO) 56.43 **53.94** 65.49 88.26 **69.23** 22.40 35.35 70.73 58.77 **54.95** 15.50 20.81 63.57 50.24 **44.87**

Qwen-2.5

-0.5bSFT 83.44 38.71 58.03 **57.47** 51.40 18.50 **16.02** 61.35 27.82 35.06 35.50 10.50 57.19 19.57 29.09

TRUST-ALIGN (DPO) 71.84 **50.59 61.28** 52.40 **54.76** 17.90 15.76 61.84 29.73 **35.78** 21.70 13.68 60.79 22.72 **32.40**

Qwen-2.5

-1.5bSFT 78.27 44.23 58.75 **71.08** 58.02 25.50 **23.89 69.66** 37.68 43.74 41.30 14.14 55.35 27.69 32.39

TRUST-ALIGN (DPO) 72.57 **52.68 62.38** 66.81 **60.62** 20.00 23.80 68.46 50.98 **47.75** 33.60 19.03 57.91 31.63 **36.19**

Qwen-2.5

-3bSFT 75.21 47.26 60.61 73.09 60.32 27.20 28.80 68.12 37.34 44.75 34.50 14.85 61.47 35.87 37.40

TRUST-ALIGN (DPO) 49.47 55.19 63.76 78.64 **65.86** 48.10 35.69 70.31 45.64 **50.55** 13.50 22.52 64.38 42.01 **42.97**

Qwen-2.5

-7bSFT 65.30 50.73 64.50 82.07 65.77 31.70 **33.58** 70.10 49.08 50.92 25.50 20.78 **64.25** 46.89 43.97

TRUST-ALIGN (DPO) 59.49 55.04 66.22 83.57 **68.28** 32.10 30.11 70.68 53.48 **51.42** 21.00 **24.30** 63.79 47.02 **45.04**

Phi3.5

-miniSFT 66.46 51.92 **64.34** 82.77 66.34 29.10 35.04 73.93 49.38 52.78 24.50 22.50 65.70 46.79 45.00

TRUST-ALIGN (DPO) 66.56 **52.23** 64.20 85.36 **67.26** 30.10 36.42 73.95 53.40 **54.59** 24.90 23.39 67.62 47.42 **46.14**

We validated our data construction approach against the GPT-4-as-critic pipeline (Li et al., 2024a; Huang et al., 2024b), where GPT-4 iteratively identifies and corrects errors to generate positive and negative responses (details in Appendix G). In LLaMA-2-7b, TRUST-ALIGN outperforms GPT-4 critic on TRUST-SCORE, with gains of 1.29% (ASQA), 1.77% (QAMPARI), and 4.12% (ELI5).

| ASQA                                                                                                                                                     | QAMPARI         | ELI5        |                 |       |                 |             |             |       |       |             |             |       |       |
|----------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|-------------|-----------------|-------|-----------------|-------------|-------------|-------|-------|-------------|-------------|-------|-------|
| Resp.                                                                                                                                                    | Trustworthiness | Resp.       | Trustworthiness | Resp. | Trustworthiness |             |             |       |       |             |             |       |       |
| AR (%) Truthfullness Att-Grd. TRUST AR (%) Truthfullness Att-Grd. TRUST AR (%) Truthfullness Att-Grd. TRUST F1AC F1GR F1GC F1AC F1GR F1GC F1AC F1GR F1GC |                 |             |                 |       |                 |             |             |       |       |             |             |       |       |
| DPO-LLaMA-2-7b                                                                                                                                           | 65.30           | 52.48 66.12 | 83.94           | 67.51 | 31.10           | 32.09 71.83 | 51.33       | 51.75 | 21.60 | 22.54 63.27 | 48.43       | 44.75 |       |
| TRUST-ALIGN w/o. augmented instructions                                                                                                                  | 79.43           | 53.54 63.33 | 81.15           | 66.01 | 32.20           | 33.14 70.82 | 45.94       | 49.97 | 29.50 | 23.98 63.30 | 40.28       | 42.52 |       |
| TRUST-ALIGN w/o. answer HT                                                                                                                               | 77.74           | 53.29       | 63.7            | 81.2  | 66.06           | 33.40       | 33.56 71.36 | 46.17 | 50.36 | 27.60       | 23.47 63.56 | 38.28 | 41.77 |
| TRUST-ALIGN w/o. citation HT                                                                                                                             | 77.32           | 52.55 63.88 | 81.51           | 65.98 | 33.10           | 34.13 71.40 | 46.91       | 50.81 | 26.70 | 22.65 64.33 | 42.81       | 43.26 |       |
| TRUST-ALIGN w/o. refusal HT                                                                                                                              | 79.11           | 53.55 63.33 | 81.85           | 66.24 | 31.10           | 34.40 71.35 | 48.12       | 51.29 | 28.30 | 22.93 64.05 | 41.18       | 42.72 |       |
| GPT-4 as critic                                                                                                                                          | 70.36           | 54.91 65.29 | 78.47           | 66.22 | 25.90           | 30.77 70.29 | 48.87       | 49.98 | 23.50 | 17.27 62.24 | 42.38       | 40.63 |       |

Importance of refusal samples in TRUST-A**LIGN**. To verify the importance of refusal samples in our pipeline, we removed all unanswerable questions from the training set, creating a dataset without refusals. Table 6 shows a significant drop in TRUST-SCORE scores without refusals, including declines of 10.2% (LLaMA-3-8b) and 11.41% (LLaMA-2-7b). Notably, F1GR decreases by 26.34% (LLaMA-3-8b) and 26.97% (LLaMA- 2-7b), and F1GC by 6.87% (LLaMA-3-8b) and 6.57% (LLaMA-2-7b). We also observe that in LLaMA-3-8b, F1AC is higher in the answerable-only setting compared to with refusals setting. This occurs because RAC favors over-responsive models, which artificially inflates F1AC, as discussed in main results. The resulting models answer all questions (AR% of 100%), even without supporting documents, suggesting an increased reliance on ungrounded parametric knowledge, as discussed in Section 6.1.

Table 6: Effect of adding refusal samples on the ASQA.

TRUST-ALIGN Models AR% F1AC F1GR F1GC **TRUST**
Only Answerable DPO-LLaMA-2-7b 100 51.79 39.15 77.37 56.10 DPO-LLaMA-3-8b 100 **56.54** 39.15 81.39 59.03 With Refusal DPO-LLaMA-2-7b 65.30 52.48 **66.12** 83.94 67.51 DPO-LLaMA-3-8b 56.43 53.94 65.49 **88.26 69.23**
Out-of-domain analysis. Following Huang et al. (2024a), we use ExpertQA (Malaviya et al.,
2024) to assess our model's generalizability. As shown in Table 7, TRUST-ALIGN model outperforms FRONT on TRUST-SCORE across all 27 open-source model family and dataset configurations. We also observe that the open-source ICL models perform significantly worse on TRUST-SCORE as compared to the closed-source ICL models, with a 9.79% gap between LLaMA-3-8b and GPT- 4. TRUST-ALIGN not only closes this gap but establishes a lead: TRUST-ALIGNed LLaMA-3-8b achieves the highest TRUST score of 54.85%, surpassing 54.69% of GPT-4.

| Model             | Type   | AR (%)            | F1AC F1GR F1GC TRUST   |       |       |        |                      |       |       |       |      |
|-------------------|--------|-------------------|------------------------|-------|-------|--------|----------------------|-------|-------|-------|------|
| ICL               | 0.51   | 0.00              | 41.01                  | 9.52  | 16.84 |        |                      |       |       |       |      |
| PostCite          | 5.62   | 4.85              | 44.27                  | 5.23  | 18.12 |        |                      |       |       |       |      |
| PostAttr          | 5.62   | 4.85              | 44.27                  | 2.26  | 17.13 |        |                      |       |       |       |      |
| FRONT             | 100    | 9.33              | 23.92 74.75            | 36.00 |       |        |                      |       |       |       |      |
| TRUST-ALIGN (DPO) | 20.01  | 25.03 67.91 62.46 | 51.8                   | Model | Type  | AR (%) | F1AC F1GR F1GC TRUST |       |       |       |      |
| ICL               | 78.24  | 21.42             | 38.71                  | 0.44  | 20.19 |        |                      |       |       |       |      |
| PostCite          | 51.41  | 13.32             | 48.08                  | 5.6   | 22.33 |        |                      |       |       |       |      |
| Qwen-2.5 PostAttr | 51.41  | 13.32             | 48.08                  | 1.49  | 20.96 |        |                      |       |       |       |      |
| -0.5b             | FRONT  | 99.86             | 18.27                  | 24.05 | 34.62 | 25.65  |                      |       |       |       |      |
| TRUST-ALIGN (DPO) | 32.96  | 18.16             | 63.31                  | 35.07 | 38.85 |        |                      |       |       |       |      |
| ICL               | 90     | 21.55 32.83       | 9.04                   | 21.14 |       |        |                      |       |       |       |      |
| PostCite          | 30.84  | 5.48              | 49.1                   | 2.67  | 19.08 |        |                      |       |       |       |      |
| PostAttr          | 48.41  | 8.24              | 47.72                  | 1.5   | 19.15 |        |                      |       |       |       |      |
| FRONT             | 95.62  | 20.83 29.26 37.45 | 29.18                  |       |       |        |                      |       |       |       |      |
| TRUST-ALIGN (DPO) | 15.44  | 20.32 64.87       | 62.1                   | 49.1  | ICL   | 98.34  | 30.67                | 26.09 | 6.89  | 21.22 |      |
| PostCite          | 62.19  | 22.22             | 48.66                  | 16.92 | 29.27 |        |                      |       |       |       |      |
| Qwen-2.5 PostAttr | 62.19  | 22.22             | 48.66                  | 13.15 | 28.01 |        |                      |       |       |       |      |
| -1.5b             | FRONT  | 99.59             | 29.15                  | 24.6  | 50.22 | 34.66  |                      |       |       |       |      |
| TRUST-ALIGN (DPO) | 30.2   | 25.06             | 68.38                  | 51.44 | 48.29 |        |                      |       |       |       |      |
| ICL               | 58.74  | 33.5              | 51.21 38.37            | 41.03 |       |        |                      |       |       |       |      |
| PostCite          | 82.85  | 25.68 38.11       | 5.29                   | 23.03 |       |        |                      |       |       |       |      |
| PostAttr          | 82.85  | 25.45 38.58       | 3.4                    | 22.48 |       |        |                      |       |       |       |      |
| FRONT             | 83.36  | 27.24 43.34 50.91 | 40.5                   |       |       |        |                      |       |       |       |      |
| TRUST-ALIGN (DPO) | 7.24   | 11.72 56.93 78.35 | 49.0                   | ICL   | 68.88 | 35.14  | 49.65                | 42.67 | 42.49 |       |      |
| PostCite          | 0.05   | 0                 | 40.66                  | 0     | 13.55 |        |                      |       |       |       |      |
| Qwen-2.5 PostAttr | 0.05   | 0                 | 40.66                  | 0     | 13.55 |        |                      |       |       |       |      |
| -3b               | FRONT  | 95.48             | 25.67                  | 29.86 | 44.48 | 33.34  |                      |       |       |       |      |
| TRUST-ALIGN (DPO) | 17.15  | 20.97             | 65.79                  | 60.25 | 49.0  |        |                      |       |       |       |      |
| ICL               | 0.65   | 2.82              | 42.5                   | 69.46 | 38.26 |        |                      |       |       |       |      |
| PostCite          | 15.68  | 14.06 50.08       | 7.09                   | 23.74 |       |        |                      |       |       |       |      |
| PostAttr          | 15.68  | 14.06 50.08       | 6.29                   | 23.47 |       |        |                      |       |       |       |      |
| FRONT             | 99.26  | 30.34 24.92       | 56.7                   | 37.32 |       |        |                      |       |       |       |      |
| TRUST-ALIGN (DPO) | 16.41  | 27.36 67.07 70.11 | 54.85                  |       |       |        |                      |       |       |       |      |
| GPT-3.5           | ICL    | 59.47             | 36.65 56.39 63.93      | 52.32 |       |        |                      |       |       |       |      |
| GPT-4             | ICL    | 72.20             | 41.32 52.91 69.83      | 54.69 |       |        |                      |       |       |       |      |
| GPT-4o            | ICL    | 66.07             | 42.62                  | 64.4  | 54.61 | 51.24  |                      |       |       |       |      |
| TRUST-ALIGN (SFT) | 36.84  | 28.85 71.68 61.98 | 53.82                  |       |       |        |                      |       |       |       |      |
| Claude-3.5        | ICL    | 73.95             | 11.68 51.91            | 10.7  | 24.76 | ICL    | 84.56                | 36.33 | 42.28 | 56.09 | 44.9 |
| PostCite          | 42.14  | 25.58             | 54.9                   | 13.77 | 31.42 |        |                      |       |       |       |      |
| Qwen-2.5 PostAttr | 42.14  | 25.58             | 54.9                   | 12.46 | 30.98 |        |                      |       |       |       |      |
| -7b               | FRONT  | 65.51             | 32.41                  | 55.56 | 67.35 | 51.77  |                      |       |       |       |      |
| TRUST-ALIGN (DPO) | 24.99  | 25.57             | 69.16                  | 62.7  | 52.48 |        |                      |       |       |       |      |
| ICL               | 85.15  | 37.49             | 40.22                  | 36.14 | 37.95 |        |                      |       |       |       |      |
| PostCite          | 52.01  | 27.96             | 53.64                  | 7.39  | 29.66 |        |                      |       |       |       |      |
| PostAttr          | 52.01  | 27.96             | 53.64                  | 5.7   | 29.1  |        |                      |       |       |       |      |
| Phi3.5 -mini      | FRONT  | 97.37             | 28.19                  | 27.5  | 65.82 | 40.5   |                      |       |       |       |      |
| TRUST-ALIGN (DPO) | 26.05  | 27.69             | 69.56                  | 61.6  | 52.95 |        |                      |       |       |       |      |

In LLaMA-3-8B, TRUST-ALIGN outperforms ICL on F1GR by 16.59% and substantially outperforms GPT-3.5 and Claude 3.5 in both F1GC and F1GR. Although GPT-3.5 and GPT-4 achieve higher F1AC scores, indicating better answer coverage, they rely heavily on parametric knowledge
(Section 6.1 and Appendix F.3). This leads to less grounded and less trustworthy responses, as reflected in lower TRUST-SCORE scores compared to TRUST-ALIGN. Similar trends are observed in other model families. Studying parametric knowledge access. For an LLM-in-RAG task, it is important to study the tendency of LLM towards grounding its knowledge on the provided documents. To partially quantify this, we compute the answer correctness score for questions that are unanswerable by the provided documents (defined as Sparam); thus a fraction of cases where AG∩AD = ∅ but AG ̸= ∅ (more details on the metric in Appendix F.2). In Table 10, our analysis reveals that responsive models (high AR%) tend to rely on parametric knowledge more frequently (high Sparam). Notably, closed-source models like GPT-4 exhibit higher parametric knowledge usage compared to open-source and TRUST-ALIGN models. However, Sparam only partially captures the models' utilization of parametric knowledge. For instance, it does not account for cases where the document contains the answer, and the model still relies on parametric knowledge to generate the correct answer (also present in the document).

This phenomenon is evident in Table 12, where on ASQA, GPT-4 achieves a significantly higher F1AC than our models, yet its attribution groundedness score F1GC is five points lower.

## 7 Conclusion

In this study, we introduced a new holistic metric to evaluate the suitability of LLMs for RAG applications, where they are expected to ground their responses in the provided documents. We proposed TRUST-SCORE, which comprehensively measures the quality of answers, citations, and refusal performance of an LLM. Additionally, we presented TRUST-ALIGN, a method that uses a constructed dataset to align models for improved TRUST-SCORE performance. By applying Direct Preference Optimization (DPO) techniques, we trained LLaMA-2-7b and LLaMA-3-8b on this dataset, significantly reducing hallucinations in an RAG environment. Our approach, TRUST-ALIGN, demonstrates performance comparable to major closed-source models like GPT-4.

## Acknowledgement

This research/project is supported by the National Research Foundation, Singapore under its AI Singapore Programme (AISG Award No: AISG3-GV-2023-010). This work is also supported by the Microsoft Research Accelerate Foundation Models Academic Research program.

## References

Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, Alon Benhaim, Misha Bilenko, Johan Bjorck, Sebastien Bubeck, Martin Cai, Qin Cai, Vishrav Chaudhary, Dong Chen, Dong- ´ dong Chen, Weizhu Chen, Yen-Chun Chen, Yi-Ling Chen, Hao Cheng, Parul Chopra, Xiyang Dai, Matthew Dixon, Ronen Eldan, Victor Fragoso, Jianfeng Gao, Mei Gao, Min Gao, Amit Garg, Allie Del Giorno, Abhishek Goswami, Suriya Gunasekar, Emman Haider, Junheng Hao, Russell J. Hewett, Wenxiang Hu, Jamie Huynh, Dan Iter, Sam Ade Jacobs, Mojan Javaheripi, Xin Jin, Nikos Karampatziakis, Piero Kauffmann, Mahoud Khademi, Dongwoo Kim, Young Jin Kim, Lev Kurilenko, James R. Lee, Yin Tat Lee, Yuanzhi Li, Yunsheng Li, Chen Liang, Lars Liden, Xihui Lin, Zeqi Lin, Ce Liu, Liyuan Liu, Mengchen Liu, Weishung Liu, Xiaodong Liu, Chong Luo, Piyush Madan, Ali Mahmoudzadeh, David Majercak, Matt Mazzola, Caio Cesar Teodoro ´ Mendes, Arindam Mitra, Hardik Modi, Anh Nguyen, Brandon Norick, Barun Patra, Daniel Perez- Becker, Thomas Portet, Reid Pryzant, Heyang Qin, Marko Radmilac, Liliang Ren, Gustavo de Rosa, Corby Rosset, Sambudha Roy, Olatunji Ruwase, Olli Saarikivi, Amin Saied, Adil Salim, Michael Santacroce, Shital Shah, Ning Shang, Hiteshi Sharma, Yelong Shen, Swadheen Shukla, Xia Song, Masahiro Tanaka, Andrea Tupini, Praneetha Vaddamanu, Chunyu Wang, Guanhua Wang, Lijuan Wang, Shuohang Wang, Xin Wang, Yu Wang, Rachel Ward, Wen Wen, Philipp Witte, Haiping Wu, Xiaoxia Wu, Michael Wyatt, Bin Xiao, Can Xu, Jiahang Xu, Weijian Xu, Jilong Xue, Sonali Yadav, Fan Yang, Jianwei Yang, Yifan Yang, Ziyi Yang, Donghan Yu, Lu Yuan, Chenruidong Zhang, Cyril Zhang, Jianwen Zhang, Li Lyna Zhang, Yi Zhang, Yue Zhang, Yunan Zhang, and Xiren Zhou. Phi-3 technical report: A highly capable language model locally on your phone, 2024. URL https://arxiv.org/abs/2404.14219.

Samuel Joseph Amouyal, Tomer Wolfson, Ohad Rubin, Ori Yoran, Jonathan Herzig, and Jonathan Berant. Qampari: An open-domain question answering benchmark for questions with many answers from multiple paragraphs, 2023. URL https://arxiv.org/abs/2205.12665.

Anthropic. Introducing claude 3.5 sonnet. *Anthropic News*, 2024. URL https://www.

anthropic.com/news/claude-3-5-sonnet.

Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG: Learning to retrieve, generate, and critique through self-reflection. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id= hSyW5go0v8.

Patrice Bechard and Orlando Marquez Ayala. Reducing hallucination in structured outputs via ´
retrieval-augmented generation. *arXiv preprint arXiv:2404.08189*, 2024.

Rishabh Bhardwaj, Yingting Li, Navonil Majumder, Bo Cheng, and Soujanya Poria. knn-cm: A nonparametric inference-phase adaptation of parametric text classifiers. In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 13546–13557, 2023.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020. URL
https://arxiv.org/abs/2005.14165.

Jan Buchmann, Xiao Liu, and Iryna Gurevych. Attribute or abstain: Large language models as long document assistants, 2024. URL https://arxiv.org/abs/2407.07799.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, et al. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407. 21783.

Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. Eli5:
Long form question answering, 2019. URL https://arxiv.org/abs/1907.09190.

Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. A survey on rag meeting llms: Towards retrieval-augmented large language models. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 6491–6501, 2024.

Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony Chen, Arun Tejasvi Chaganty, Yicheng Fan, Vincent Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, and Kelvin Guu. RARR: Researching and revising what language models say, using language models. In Anna Rogers, Jordan Boyd- Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 16477–16508, Toronto, Canada, July 2023a. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.910. URL https://aclanthology.org/2023.acl-long.910.

Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. Enabling large language models to generate text with citations, 2023b.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2023c.

Or Honovich, Roee Aharoni, Jonathan Herzig, Hagai Taitelbaum, Doron Kukliansy, Vered Cohen, Thomas Scialom, Idan Szpektor, Avinatan Hassidim, and Yossi Matias. TRUE: Re-evaluating factual consistency evaluation. In Song Feng, Hui Wan, Caixia Yuan, and Han Yu (eds.), Proceedings of the Second DialDoc Workshop on Document-grounded Dialogue and Conversational Question Answering, pp. 161–175, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.dialdoc-1.19. URL https://aclanthology.org/ 2022.dialdoc-1.19.

I-Hung Hsu, Zifeng Wang, Long T. Le, Lesly Miculicich, Nanyun Peng, Chen-Yu Lee, and Tomas Pfister. Calm: Contrasting large and small language models to verify grounded generation, 2024. URL https://arxiv.org/abs/2406.05365.

Chengyu Huang, Zeqiu Wu, Yushi Hu, and Wenya Wang. Training language models to generate text with citations via fine-grained rewards, 2024a.

Lei Huang, Xiaocheng Feng, Weitao Ma, Yuxuan Gu, Weihong Zhong, Xiachong Feng, Weijiang Yu, Weihua Peng, Duyu Tang, Dandan Tu, and Bing Qin. Learning fine-grained grounded citations for attributed large language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), *Findings of the Association for Computational Linguistics ACL 2024*, pp. 14095–14113, Bangkok, Thailand and virtual meeting, August 2024b. Association for Computational Linguistics. URL https://aclanthology.org/2024.findings-acl.838.

Bin Ji, Huijun Liu, Mingzhe Du, and See-Kiong Ng. Chain-of-thought improves text generation with citations in large language models. Proceedings of the AAAI Conference on Artificial Intelligence, 38(16):18345–18353, Mar. 2024. doi: 10.1609/aaai.v38i16.29794. URL https://ojs.aaai.org/index.php/AAAI/article/view/29794.

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1–38, March 2023. ISSN 1557-7341. doi: 10.1145/3571730. URL http://dx.doi.org/10.1145/3571730.

Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation. arXiv preprint arXiv:2305.06983, 2023.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (eds.), Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 6769–6781, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.550.

URL https://aclanthology.org/2020.emnlp-main.550.

Muhammad Khalifa, David Wadden, Emma Strubell, Honglak Lee, Lu Wang, Iz Beltagy, and Hao Peng. Source-aware training enables knowledge attribution in language models, 2024.

Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization through memorization: Nearest neighbor language models. *arXiv preprint arXiv:1911.00172*, 2019.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen tau Yih, Tim Rockt ¨ aschel, Sebastian Riedel, and Douwe ¨ Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks, 2021.

Dongfang Li, Zetian Sun, Xinshuo Hu, Zhenyu Liu, Ziyang Chen, Baotian Hu, Aiguo Wu, and Min Zhang. A survey of large language models attribution. *arXiv preprint arXiv:2311.03731*, 2023.

Dongfang Li, Zetian Sun, Baotian Hu, Zhenyu Liu, Xinshuo Hu, Xuebo Liu, and Min Zhang. Improving attributed text generation of large language models via preference learning. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), *Findings of the Association for Computational* Linguistics ACL 2024, pp. 5079–5101, Bangkok, Thailand and virtual meeting, August 2024a.

Association for Computational Linguistics. URL https://aclanthology.org/2024. findings-acl.301.

Weitao Li, Junkai Li, Weizhi Ma, and Yang Liu. Citation-enhanced generation for llm-based chatbots, 2024b.

Nelson Liu, Tianyi Zhang, and Percy Liang. Evaluating verifiability in generative search engines. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 7001–7025, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.467. URL https://aclanthology.org/2023.findings-emnlp.467.

Chaitanya Malaviya, Subin Lee, Sihao Chen, Elizabeth Sieber, Mark Yatskar, and Dan Roth. Expertqa: Expert-curated questions and attributed answers, 2024. URL https://arxiv.org/ abs/2309.07852.

Yu Meng, Mengzhou Xia, and Danqi Chen. SimPO: Simple preference optimization with a reference-free reward. *arXiv preprint arXiv:2405.14734*, 2024.

Jacob Menick, Maja Trebacz, Vladimir Mikulik, John Aslanides, Francis Song, Martin Chadwick, Mia Glaese, Susannah Young, Lucy Campbell-Gillingham, Geoffrey Irving, et al. Teaching language models to support answers with verified quotes. *arXiv preprint arXiv:2203.11147*, 2022.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. *arXiv preprint arXiv:2112.09332*, 2021.

Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernandez Abrego, Ji Ma, Vincent Zhao, Yi Luan, Keith Hall, Ming-Wei Chang, and Yinfei Yang. Large dual encoders are generalizable retrievers. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 9844–9855, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.669. URL https://aclanthology.org/2022. emnlp-main.669.

OpenAI. Chatgpt, 2023. URL https://openai.com/index/chatgpt/. Accessed: 202409-01.

OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, et al. Gpt-4 technical report, 2024. URL https://arxiv.org/abs/2303.08774.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback, 2022.

URL https://arxiv.org/abs/2203.02155.

Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Dmytro Okhonko, Samuel Broscheit, Gautier Izacard, Patrick Lewis, Barlas Oguz, Edouard Grave, Wen-tau Yih, et al. The ˘
web is your oyster-knowledge-intensive nlp against a very large web corpus. *arXiv preprint* arXiv:2112.09924, 2021.

Ori Press, Andreas Hochlehnert, Ameya Prabhu, Vishaal Udandarao, Ofir Press, and Matthias Bethge. Citeme: Can language models accurately cite scientific claims?, 2024. URL https: //arxiv.org/abs/2407.12861.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model, 2024a. URL https://arxiv.org/abs/2305.18290.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. *Advances* in Neural Information Processing Systems, 36, 2024b.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of machine learning research*, 21(140):1–67, 2020.

Hannah Rashkin, Vitaly Nikolaev, Matthew Lamm, Lora Aroyo, Michael Collins, Dipanjan Das, Slav Petrov, Gaurav Singh Tomar, Iulia Turc, and David Reitter. Measuring attribution in natural language generation models, 2022. URL https://arxiv.org/abs/2112.12870.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017. URL https://arxiv.org/abs/1707.06347.

Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H. Chi, Nathanael Scharli, and Denny Zhou. Large language models can be easily distracted by irrelevant context. ¨ In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), *Proceedings of the 40th International Conference on Machine Learning*,
volume 202 of *Proceedings of Machine Learning Research*, pp. 31210–31227. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/v202/shi23a.html.

Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. Retrieval augmentation reduces hallucination in conversation. *arXiv preprint arXiv:2104.07567*, 2021.

Aviv Slobodkin, Eran Hirsch, Arie Cattan, Tal Schuster, and Ido Dagan. Attribute first, then generate: Locally-attributable grounded text generation, 2024.

Yixiao Song, Yekyung Kim, and Mohit Iyyer. Veriscore: Evaluating the factuality of verifiable claims in long-form text generation, 2024. URL https://arxiv.org/abs/2406.19276.

Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and Ming-Wei Chang. Asqa: Factoid questions meet long-form answers, 2023. URL https://arxiv.org/abs/2204.06092.

Nandan Thakur, Luiz Bonifacio, Xinyu Zhang, Odunayo Ogundepo, Ehsan Kamalloo, David Alfonso-Hermelo, Xiaoguang Li, Qun Liu, Boxing Chen, Mehdi Rezagholizadeh, and Jimmy Lin. Nomiracl: Knowing when you don't know for robust multilingual retrieval-augmented generation, 2024. URL https://arxiv.org/abs/2312.11361.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023. URL https://arxiv.org/abs/2307.09288.

Sirui Xia, Xintao Wang, Jiaqing Liang, Yifei Zhang, Weikang Zhou, Jiaji Deng, Fei Yu, and Yanghua Xiao. Ground every sentence: Improving retrieval-augmented llms with interleaved referenceclaim generation, 2024. URL https://arxiv.org/abs/2407.01796.

Fangyuan Xu, Weijia Shi, and Eunsol Choi. Recomp: Improving retrieval-augmented lms with compression and selective augmentation, 2023. URL https://arxiv.org/abs/2310. 04408.

Yilong Xu, Jinhua Gao, Xiaoming Yu, Baolong Bi, Huawei Shen, and Xueqi Cheng. Aliice: Evaluating positional fine-grained citation generation, 2024. URL https://arxiv.org/abs/ 2406.13375.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. Qwen2 technical report. *arXiv preprint arXiv:2407.10671*, 2024.

Xi Ye, Ruoxi Sun, Sercan O. Arik, and Tomas Pfister. Effective large language model adaptation for ¨
improved grounding and citation generation, 2024.

Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Berant. Making retrieval-augmented language models robust to irrelevant context, 2024. URL https://arxiv.org/abs/2310.01558.

Jingyu Zhang, Marc Marone, Tianjian Li, Benjamin Van Durme, and Daniel Khashabi. Verifiable by design: Aligning language models to quote from pre-training data, 2024a.

Weijia Zhang, Mohammad Aliannejadi, Yifei Yuan, Jiahuan Pei, Jia-Hong Huang, and Evangelos Kanoulas. Towards fine-grained citation evaluation in generated text: A comparative analysis of faithfulness metrics, 2024b. URL https://arxiv.org/abs/2406.15264.

Xin Zheng, Zhirui Zhang, Junliang Guo, Shujian Huang, Boxing Chen, Weihua Luo, and Jiajun Chen. Adaptive nearest neighbor machine translation. *arXiv preprint arXiv:2105.13022*, 2021.

| Table of Contents A Nuances of answerability   | 17                                                                   |    |    |
|------------------------------------------------|----------------------------------------------------------------------|----|----|
| B                                              | Answerability: A Case Study                                          | 17 |    |
| C                                              | Related Works                                                        | 18 |    |
| C.1                                            | Attributable Retrieval Augmented Generation                          | 18 |    |
| C.2                                            | Enhance grounded text generation in attributed Large Language Models | 18 |    |
| D                                              | Metrics                                                              | 19 |    |
| D.1                                            | Response Truthfulness                                                | 19 |    |
| D.2                                            | Attribution Groundedness                                             | 20 |    |
| E                                              | The TRUST-ALIGN Dataset                                              | 21 |    |
| E.1                                            | Collecting Quality Questions                                         |    | 21 |
| E.2                                            | Collecting D's                                                       | 21 |    |
| E.3                                            | Augmenting (q,D) set                                                 |    | 22 |
| E.4                                            | Obtaining r +                                                        |    | 22 |
| E.5                                            | Obtaining r −                                                        |    | 23 |
| F                                              | Additional Analysis                                                  | 24 |    |
| F.1                                            | Revised metrics are less biased                                      |    | 24 |
| F.2                                            | Utilization of Parametric Knowledge                                  |    | 25 |
| F.3                                            | The Source of LLM Hallucinations                                     |    | 25 |
| F.4                                            | TRUST-ALIGN enhances trustworthiness more robustly than prompting    |    | 26 |
| F.5                                            | Comparison with Closed-source Models                                 |    | 27 |
| F.6                                            | Adaptability with Different Alignment Techniques                     | 28 |    |
| F.7                                            | Evaluation Data Creation Without using TRUE                          | 29 |    |
| F.8                                            | Effect of data size on DPO performance                               | 30 |    |
| F.9                                            | Fine-tuning GPT-4o                                                   | 30 |    |
| G GPT-4 based Data Pipeline                    | 30                                                                   |    |    |
| H Experimental Setup                           | 31                                                                   |    |    |
| H.1                                            | Implementation details                                               |    | 31 |
| H.2                                            | Dataset details                                                      | 31 |    |
| H.3                                            | Baselines                                                            | 32 |    |
| H.4                                            | Refusal Detection                                                    |    | 32 |
| I                                              | Detailed Results                                                     | 33 |    |
| J                                              | Prompt Templates                                                     | 33 |    |

## A Nuances Of Answerability

Determining answerability can be challenging. To determine answerability, we use a system that evaluates the entailment of gold claims against provided documents, referred to as the Natural Language Inference (NLI) system. An NLI system can range from a simple exact match (EM) identifier to an LLM or even a human evaluator, with answerability determined based on *q, D* and biases of the NLI9. These biases can be useful in specific RAG applications, such as solving mathematical problems where the documents provide a formula and the question assigns values to variables. The choice of NLI depends on whether the RAG system requires the LLM to have mathematical understanding. **Ideally, to prevent improper evaluations, the NLI model used to construct the gold** claims should also be used to evaluate the LLM responses. In this paper, our focus is on evaluating the generic comprehension capabilities of LLMs without specialized knowledge. Thus, we use two NLI mechanisms: 1) identifying whether an exact match of claims is present in the gold claims, and 2) using a Machine Learning (ML) model to determine if the documents can entail the gold claims. The ML-based NLI model is used for multiple purposes, such as alignment dataset construction (data/training) and evaluating generated responses
(metric/testing). For this, we adopt the NLI model from Rashkin et al. (2022). ϕ(cij , si) = 1 if cij
(premise) entails si (hypothesis); otherwise, 0. To determine answerability, we employ the TRUE- based method (Honovich et al., 2022) to assess whether a gold claim can be entailed by a given document.

The knowledge grounding problem. Typically, LLMs are designed to perform questionanswering tasks, where response generation heavily relies on the parametric (internal) knowledge acquired during their pre-training, tuning, and alignment phases (OpenAI, 2023; Anthropic, 2024). Thus, most of their knowledge is grounded in parametric memory. This makes them inherently less suitable for RAG applications, where the knowledge generated by the LLM is expected to be grounded in input documents. RAG is analogous to a reading comprehension task, where the answers must come from the provided passage (documents in RAG) rather than the prior knowledge of the person taking the test. Thus, any reliance on parametric knowledge can result in statements that are not fully grounded in the documents, including providing answers to unanswerable questions. Our investigation shows that state-of-the-art models, such as GPT-4 and Claude-3.5-Sonnet, overtly rely on parametric knowledge even when used in a RAG setting.10

## B Answerability: A Case Study

Prior works (Liu et al., 2023; Gao et al., 2023b; Ye et al., 2024; Huang et al., 2024a; Li et al., 2024a) have employed substring matching to indicate entailment. While this syntactic approach is fast, it often proves inadequate in complex, long contexts. A case study is presented in Table 8. To address the limitations of this superficial entailment, we adopt a TRUE-based method (Honovich et al., 2022), which combines the strengths of both syntactic and semantic approaches. Specifically, we enhance the process by using the TRUE model, a T5-11B model (Raffel et al., 2020) fine-tuned for the NLI task, to verify, from a semantic perspective, whether a substring match corresponds to meaningful entailment within document passages. The input to the TRUE model is the concatenation of a premise and a hypothesis, and the output is an entailment score between 0 and 1, indicating the degree to which the premise entails the hypothesis. We treat the corresponding documents as the premise, and to minimize ambiguity, the associated question is concatenated with each gold answer as the hypothesis. In cases where the TRUE model does not yield a positive entailment score despite a substring match, we rely on the TRUE judgment as the final label. However, if the substring match fails, we bypass TRUE calculation, thus reducing the computational cost of relying solely on TRUE for semantic entailment.

| Question           | How many state parks are there in Virginia?                                                                                                                                                                                                                                                                                       |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Gold Answer        | 38                                                                                                                                                                                                                                                                                                                                |
| Retrieved document | Virginia has 30 National Park Service units, such as Great Falls Park and the Appalachian Trail, and one national park, the Shenandoah National Park. With over 500 miles of trails, including 38 miles of the iconic Appalachian Trail, it's a paradise for hikers, nature lovers, and those seeking serene mountain landscapes. |
| Substring match    | Substring is matched and as such the question is answerable.                                                                                                                                                                                                                                                                      |
| TRUE Judgement     | Not entailed as such the question is unanswerable given the document.                                                                                                                                                                                                                                                             |

## C Related Works C.1 Attributable Retrieval Augmented Generation

Retrieval Augmented Generation (RAG) has been widely studied for reducing the knowledge gap and providing more referenced information to enhance answer generation (Karpukhin et al., 2020; Lewis et al., 2021; Gao et al., 2023c). However, LLMs are prone to being misled by irrelevant information, leading to hallucinations and less factual outputs (Shi et al., 2023; Yoran et al., 2024; Xu et al., 2023). This challenge has spurred research into attributable RAG, which aims to verify model outputs by identifying supporting sources. Rashkin et al. (2022) first introduced the concept of Attributable to Identified Sources (AIS) to evaluate attribution abilities. Subsequently, Gao et al. (2023b) adapted this approach to verify generated content with citations, improving the reliability of RAG systems. Simultaneously, Press et al. (2024) and Song et al. (2024) explored related aspects: citation attribution for paper identification and the verifiability of long-form generated text, respectively. Further fine-grained evaluations have been examined, such as assessing the degree of support (Zhang et al., 2024b) and the granularity of claims (Xu et al., 2024). Recent studies (Buchmann et al., 2024; Hsu et al., 2024) have also investigated attribution ability by disentangling the confounding effects of retrievers and LLMs. Unlike existing works, we design TRUST-SCORE to prioritize trustworthiness in LLMs by ensuring that generated responses are strictly grounded in the provided documents, thereby minimizing the generation of unverifiable content. This focus on verifiable accuracy strengthens the reliability of LLM outputs and enhances user trust.

## C.2 Enhance Grounded Text Generation In Attributed Large Language Models

To enhance grounded text generation, various attributed LLMs have been proposed, falling into two main paradigms: training-free and training-based. For training-free methods: 1) In-context learning (Gao et al., 2023b) is used to generate in-line citations with few-shot demonstrations. 2) Post-hoc attribution (Gao et al., 2023a; Li et al., 2024b) first generates an initial response and then retrieves evidence as attribution. 3) Ji et al. (2024) demonstrate that using chain-of-thought reasoning improves the quality of text generated with citations. For training-based methods: 1) Asai et al. (2024); Slobodkin et al. (2024); Xia et al. (2024); Ye et al. (2024) apply supervised fine-tuning (SFT) to LLMs, training them to identify useful information from documents and guide cited text generation with them. 2) Beyond simple SFT, recent studies model the task as preference learning, employing Reinforcement Learning with Human Feedback (RLHF) (Ouyang et al., 2022) and Direct Preference Optimization (DPO) (Rafailov et al., 2024a). Huang et al. (2024a) proposed a method to improve attribution generation using fine-grained rewards and Proximal Policy Optimization (PPO) (Schulman et al., 2017), while Li et al. (2024a); Huang et al. (2024b) introduced the modified DPO framework to enhance fine-grained attribution abilities. 3) While many approaches rely on external documents provided by the user or retrieved during generation, Khalifa et al. (2024); Zhang et al. (2024a) focus on tuning LLMs to cite sources from pre-training data using learned parametric knowledge. In contrast to previous approaches, we introduce TRUST-ALIGN, which advances alignment data generation through a multi-step process that disentangles answer generation from citation quality. This separation enables TRUST-ALIGN to simultaneously improve the quality of answer generation, citation accuracy, and refusal precision. Additionally, TRUST-ALIGN addresses a broader range of hallucination errors, including inappropriate refusals, thereby enhancing the overall trustworthiness and reliability of the model's outputs.

## D Metrics

In this section, we elaborate on how we compute metrics that are components of TRUST-SCORE.

## D.1 Response Truthfulness

Truthfulness captures the model's ability to answer or refuse a question correctly by computing the grounded refusal (F1GR) and the factual accuracy by computing the answer-calibrated answer correctness score (F1AC).

Grounded Refusal [F1GR]: A macro-averaged F1 score that measures the LLM's ability in correctly refusing to answer a question (F1ref) and correctly providing an answer when required (F1ans).

- F1ref: This metric evaluates a model's ability to correctly refuse unanswerable questions.

We calculate it based on how accurately the model identifies and refuses these questions.

Let Ag and ¬Ag represent the sets of ground truth answerable and unanswerable questions, respectively, and Ar and ¬Ar denote the sets of questions where the model provided an answer and refused to answer, respectively. F1ref is computed from precision Pref and recall Rref:

$$\begin{array}{l}{\rm P_{ref}=\frac{|\neg A_{r}\cap\neg A_{g}|}{|\neg A_{r}|}}\\ {\rm R_{ref}=\frac{|\neg A_{r}\cap\neg A_{g}|}{|\neg A_{g}|}}\\ {\rm F1_{ref}=\frac{2P_{ref}\cdot R_{ref}}{P_{ref}+R_{ref}},}\end{array}$$
$$(1)$$
$$({\mathfrak{I}})$$
$${\mathrm{(2)}}$$

where Pref measures the proportion of correctly refused unanswerable questions among all refused questions, and Rref measures the proportion of correctly refused unanswerable questions out of all unanswerable questions. Here, |·| denote the cardinality of the set, thus Pref, Rref, and F1ref are scalar values.

- F1ans: This metric evaluates a model's ability to correctly answer answerable questions.

It is computed based on the precision Pans and recall Rans for non-refusal responses to answerable questions:

$$\text{P}_{\text{ans}}=\frac{|A_{r}\cap A_{g}|}{|A_{r}|}$$ $$\text{R}_{\text{ans}}=\frac{|A_{r}\cap A_{g}|}{|A_{g}|}$$ $$\text{F1}_{\text{ans}}=\frac{2\text{P}_{\text{ans}}\cdot\text{R}_{\text{ans}}}{\text{P}_{\text{ans}}+\text{R}_{\text{ans}}}$$
$$(4)$$
$$(S)$$
$$(\mathbf{6})$$

F1GR **(Grounded Refusals)** provides an overall assessment of the model's refusal capabilities by computing the macro-average of F1ref and F1ans:

$$\mathrm{F1_{GR}=\frac{1}{2}(F1_{r e f}+F1_{a n s})}$$
(F1ref + F1ans) (7)
F1ref evaluates the model's ability to correctly refuse unanswerable questions, while F1ans assesses its ability to correctly answer answerable ones. By penalizing both incorrect refusals and incorrect non-refusals, F1GR offers a balanced evaluation of the model's over-responsiveness and underresponsiveness

$$\left(7\right)$$

Answer Correctness (Answer Calibrated) [F1AC]: Given a question q and the corresponding gold claims AG = {ag1*, . . . , a*gn}, we define the claims obtainable from the provided documents as AD = {ad1*, . . . , a*dn} and the claims generated in the response r as AR = {ar1*, . . . , a*rn}. ACq disregards the claims that cannot be inferred from D (answer calibration), and the exact match recall scores is computed on the remaining claims, i.e., AG ∩ AD:

$$\mathbf{AC}^{qi}=\frac{|A_{G}\cap A_{D}\cap A_{R}|}{|A_{G}\cap A_{D}|}\tag{8}$$

For the whole dataset with multiple questions {q1 *. . . q*k}, one can compute the average:

$${\rm AC}=\frac{1}{k}\sum_{q_{i}\in A_{g}\cap A_{r}}{\rm AC}^{q_{i}}\tag{9}$$

Where Ag denote the set of questions that are answerable using the provided documents, fully or partially; Ar denote the set of questions that are answered by the model (non-refusal). There are two variants of AC we study— precision-oriented PAC with denominator k = |Ar| (number of answered questions). Second variant, recall-oriented RAC with denominator k = |Ag| (number of answerable questions). Here *| · |* denotes the cardinality of the set. We denote the aggregated score by

$$\mathbf{F1}_{\mathrm{AC}}={\frac{2\ \mathbf{P}_{\mathrm{AC}}\ \cdot\ \mathbf{R}_{\mathrm{AC}}}{\mathbf{P}_{\mathrm{AC}}+\ \mathbf{R}_{\mathrm{AC}}}}.$$
$$(10)$$

. (10)
The primary reason for adjusting the conventional Answer Correctness (AC) metric to account for the presence of answers in retrieved documents is to avoid rewarding models for generating correct answers without locating them in the provided documents. This approach discourages models from relying solely on their pre-trained knowledge to answer questions, instead encouraging them to find and ground their answers within the provided documents.

## D.2 Attribution Groundedness

Attribution or citation groundedness measures the relevance of generated citations to their corresponding statements, both individually and collectively. A citation ci,j is deemed "*relevant*" when the statement it cites can be inferred from the cited document. The collective importance of citations is assessed using a statement-wise recall metric, while the individual importance of each citation is evaluated using a precision metric. Given that a generated response r consists of multiple statements S and their corresponding citations C, we first compute statement-wise citation recall and per-citation precision. These scores are then averaged to obtain sample-wise scores, which are finally averaged to produce dataset-wide scores.

Grounded Citation F1 [F1GC]: For a given statement si, statement-wise citation recall is computed by:

$${\bf R}_{\rm cute}^{s_{i}}=\phi(\{c_{i,1},\ldots,c_{i,j}\},s_{i})\tag{1}$$
$$(11)$$

where ϕ({ci,1, . . . , ci,j}, si) → {0, 1} is a function that determines whether the concatenation of all cited documents fully supports the statement si. Next, we compute precision for a generated citation ci,j for statement si as:

$$\begin{array}{c}{{\bf P}_{\rm cile}^{c_{j}}=\phi(c_{i,j},s_{i})}\\ {{\bf OR}\quad\neg\phi(\{c_{i,k}\mid k\neq j\},s_{i})}\end{array}$$

Thus, citation precision is 0 if and only if the cited document ci,j does not entail the statement si, while all other citations collectively entail si without ci,j .

$$(12)$$