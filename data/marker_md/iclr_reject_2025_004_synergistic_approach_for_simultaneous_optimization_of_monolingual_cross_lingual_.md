**006**

**009**

**019**

**024**

**029 030**

**032**

**034**

**036**

# SYNERGISTIC APPROACH FOR SIMULTANEOUS OPTIMIZATION OF MONOLINGUAL, CROSS-LINGUAL, AND MULTILINGUAL INFORMATION RETRIEVAL

Anonymous authors Paper under double-blind review

#### ABSTRACT

Information retrieval across different languages is an increasingly important challenge in natural language processing. Recent approaches based on multilingual pre-trained language models have achieved remarkable success, yet they often optimize for either monolingual, cross-lingual, or multilingual retrieval performance at the expense of others. This paper proposes a novel hybrid batch training strategy to simultaneously improve zero-shot retrieval performance across monolingual, cross-lingual, and multilingual settings while mitigating language bias. The approach fine-tunes multilingual language models using a mix of monolingual and cross-lingual question-answer pair batches sampled based on dataset size. Experiments on XQuAD-R, MLQA-R, and MIRACL benchmark datasets show that the proposed method consistently achieves comparable or superior results in zero-shot retrieval across various languages and retrieval tasks compared to monolingual-only or cross-lingual-only training. Hybrid batch training also substantially reduces language bias in multilingual retrieval compared to monolingual training. These results demonstrate the effectiveness of the proposed approach for learning language-agnostic representations that enable strong zero-shot retrieval performance across diverse languages.

# 1 INTRODUCTION

Information retrieval (IR) across different languages is an increasingly important challenge in natural language processing. However, optimizing information retrieval systems for multilingual scenarios is not a straightforward task, as it requires considering multiple distinct retrieval settings, each with its own set of challenges and requirements, including monolingual retrieval, cross-lingual retrieval, and multilingual retrieval. Monolingual retrieval refers to the task of retrieving documents in the same language as the user's query, focusing on developing effective ranking algorithms and relevance matching techniques. Cross-lingual retrieval involves queries and documents in different languages, requiring the system to bridge the language gap by employing techniques such as query translation, document translation, or cross-lingual representation learning. Multilingual retrieval requires the creation of a single ranked list of documents in multiple languages for a given query, addressing challenges such as language disparity, varying document lengths, and potential differences in content quality and relevance across languages while providing users with a unified and coherent ranked list of results.

Recent approaches to multilingual information retrieval have leveraged multilingual pre-trained language models such as mBERT [\(Devlin et al., 2019\)](#page-10-0) and XLM-R [\(Conneau et al., 2020\)](#page-10-1) to encode queries and documents [\(Karpukhin et al., 2020\)](#page-11-0). While these models can transfer relevance matching capabilities across languages, their performance tends to underperform on cross-lingual retrieval benchmarks due to the lack of explicit alignment between languages during pretraining [\(Zhang et al.,](#page-13-0) [2023\)](#page-13-0). LaREQA, introduced by [\(Roy et al., 2020\)](#page-12-0), targets strong alignment, requiring semantically related pairs across languages to be closer in representation space than unrelated pairs within the same language. [\(Roy et al., 2020\)](#page-12-0) found that augmenting the training data through machine translation proved effective in achieving robust alignment for MLIR. However, this approach compromises performance in monolingual retrieval tasks. Alternative approaches using parallel corpora, such as InfoXLM [\(Chi et al., 2021\)](#page-10-2) and LaBSE [\(Feng et al., 2022\)](#page-10-3), have been proposed to align sentences

**059**

**061**

**064**

**067**

**069 070 071**

**074**

**079**

**089 090 091**

**094**

**104 105 106**

![](_page_1_Diagram_1.jpeg)

Figure 1: Illustrative example of monolingual, cross-lingual, and multilingual information retrieval.

across languages. However, the scarcity of parallel data, especially for low-resource languages, remains a substantial challenge. To address these limitations, [\(Lawrie et al., 2023\)](#page-11-1) introduced a Multilingual Translate-Train approach using translated datasets, [\(Hu et al., 2023\)](#page-10-4) proposed contrastive losses to align representations and remove language-specific information, [\(Huang et al., 2023a\)](#page-11-2) presented a knowledge distillation framework for multilingual dense retrieval, and [\(Lin et al., 2023a\)](#page-11-3) extended Aggretriever [\(Lin et al., 2023b\)](#page-11-4) for multilingual retrieval using semantic and lexical features. While the methods proposed in [\(Hu et al., 2023\)](#page-10-4) and [\(Huang et al., 2023a\)](#page-11-2) attempt to mitigate language bias, we raise the question: Is there a straightforward approach that addresses this issue by modifying the training data batches without necessitating the introduction of loss functions or new architectural components?

In this paper, we propose a novel hybrid batch training strategy that simultaneously optimizes retrieval performance across monolingual, cross-lingual, and multilingual settings while also mitigating language bias. Our approach fine-tunes multilingual language models using a balanced mix of monolingual and cross-lingual question-answer pair batches. We collect a diverse set of English question-answer datasets and use machine translation to generate parallel question-answer pairs across several languages, including low-resource languages where parallel corpora may be limited [\(Fan et al., 2021;](#page-10-5) [Kim et al., 2021;](#page-11-5) [Costa-jussa et al., 2022\)](#page-10-6). Our hybrid batch training approach ` significantly reduces the language bias that hinders the performance of multilingual retrieval systems by training the models on a diverse set of language pairs and encouraging the learning of languageagnostic representations. This mitigates the tendency of models to favor certain languages over others, ensuring that documents from multiple languages are fairly ranked based on their relevance to the query, regardless of the language. Extensive experiments on XQuAD-R, MLQA-R, and MIRACL benchmark datasets demonstrate the effectiveness of our proposed approach, with models trained using the hybrid batch strategy consistently achieving competitive results in zero-shot retrieval across various languages and retrieval tasks, outperforming models trained with only monolingual or cross-lingual data. Our approach also exhibits strong zero-shot generalization to unseen languages not included in the training data, highlighting its potential to expand the linguistic coverage of multilingual information retrieval systems.

# 2 METHODOLOGY

### 2.1 CONTRASTIVE LEARNING

Throughout the paper, we utilize the dual-encoder architecture with shared parameters, which is commonly used for dense retrieval (DR; [Ni et al., 2022\)](#page-12-1). Contrastive learning is a method for training DR models by contrasting positive pairs against negatives. Specifically, given a batch of triplets, each of which consists of a query and its relevant and irrelevant documents: (qn, d<sup>+</sup> n , d<sup>−</sup> n ); 1 ≤ n ≤ |B|. We minimize the InfoNCE loss for each query qn:

$$\mathcal{L} = \sum_{i=1}^{|\mathbf{B}|} -\log \frac{e^{s_\theta(q_i, d_i^+)}}{e^{s_\theta(q_i, d_i^+)} + \sum_{j=1}^{|\mathbf{B}|} e^{s_\theta(q_i, d_j^-)}}. \quad (1)$$

**114 115**

**117**

**119**

**127**

**129 130**

**134**

**136**

![](_page_2_Diagram_1.jpeg)

Figure 2: Illustrations of the proposed hybrid batch sampling (assuming we only have training data in English, Arabic, and Japanese), where our model is exposed to monolingual and cross-lingual batches with the respective probability of α and β = 1 − α.

We use cosine similarity as the scoring function: sθ(q, d) = cos (Eθ(q), Eθ(d)), where E<sup>θ</sup> is the encoder parametrized by θ. Following [Wang et al.](#page-12-2) [\(2022\)](#page-12-2), we incorporate prefix identifiers "Query:" and "Passage:" for queries and passages, respectively. As shown in prior work [\(Hofstatter et al.,](#page-10-7) ¨ [2021;](#page-10-7) [Lin et al., 2021\)](#page-11-6), in-batch negatives mining, the second term of the denominator in Eq [\(1\)](#page-1-0), plays a crucial role in dense retrieval training. In this work, we study different batch sampling approaches to control in-batch negative mining.

# 2.2 BATCH SAMPLING

Baseline Batch Sampling. We study the following training batching procedures introduced by [\(Roy et al., 2020\)](#page-12-0). (i) Monolingual batching (coined as X-X-mono model) creates each batch with mono language, where all the triplets consist of queries and passages in the same language. Note that we sample the language used to create the batch equally among all possible languages in our training data. (ii) Cross-lingual batching (coined as X-Y model) creates each batch, where all the triplets consist of queries and passages in different languages. Monolingual batching only focuses on contrastive learning for query-passage pairs in the same languages while cross-lingual batching mines positives and in-batch negatives from diverse languages.

As shown in [\(Roy et al., 2020\)](#page-12-0), the X-Y model is more effective in cross-lingual retrieval scenarios and shows reduced language bias; however, the X-X-mono surpasses the X-Y model in monolingual retrieval. These results inspire us to explore whether simply combining the two batch sampling approaches can achieve improvement in both monolingual and cross-lingual retrieval effectiveness.

Hybrid Batch Sampling. In this work, we propose to combine the two aforementioned baseline sampling strategies. Specifically, when creating batch training data, we set α and β = 1 − α as the respective probability of using monolingual and cross-lingual batching as shown in Fig. [2.](#page-2-0)[<sup>1</sup>](#page-2-1)

<sup>1</sup> In the experiments, we found out that setting the hyperparameters α and β to 0.5 resulted in the best balance between the performance of the proposed model on monolingual and multilingual evaluations.

### 3 EXPERIMENTAL SETUP

This section presents the experimental setup for evaluating the proposed hybrid batch training strategy. We first discuss the training process, including datasets, and multilingual pre-trained models. Next, we introduce the evaluation datasets and metrics used to assess the performance of the fine-tuned models. Finally, we describe the evaluation settings for monolingual, cross-lingual, and multilingual retrieval tasks.

#### 3.1 TRAINING

Datasets. To conduct the study of batch sampling, parallel query-passage training pairs are required such that we can construct cross-lingual triplets, where each query and its relevant (or irrelevant) passage are in different languages. mMARCO [\(Bonifacio et al., 2021\)](#page-10-8) is the only dataset with parallel queries and passages across 14 languages. In our study, we further scale the size of training data by translating the existing question-answering datasets. Specifically, we developed our in-house machine translation pipeline to create parallel QA pairs for the monolingual datasets across nine languages: Arabic, Chinese, English, German, Hindi, Russian, Spanish, Thai, and Turkish. The additional training data used in our study include DuoRC [\(Saha et al., 2018\)](#page-12-3), EntityQuestions [\(Sciavolino et al.,](#page-12-4) [2021\)](#page-12-4), Google NQ [\(Kwiatkowski et al., 2019\)](#page-11-7), MFAQ [\(De Bruyn et al., 2021\)](#page-10-9), Mr. Tydi [\(Zhang et al.,](#page-12-5) [2021\)](#page-12-5), NewsQA [\(Trischler et al., 2017\)](#page-12-6), WikiQA [\(Yang et al., 2015\)](#page-12-7), and Yahoo QA mined from Yahoo Answers. Appendix [A.1](#page-13-1) provides comprehensive details about the training datasets.

Training Setup. We apply the baseline and our proposed hybrid batching to fine-tune two representative multilingual pre-trained models: (i) XLM-RoBERTa (XLM-R) [\(Conneau et al., 2020\)](#page-10-1); and (ii) language-agnostic BERT sentence embedding (LaBSE) [\(Feng et al., 2022\)](#page-10-3). Model training experiments were conducted using one NVIDIA A100-80 GB GPU. We fine-tune pre-trained models using AdamW optimizer [\(Loshchilov & Hutter, 2018\)](#page-11-8) with weight decay set to 1e-2, a learning rate of 3e-5, and a batch size of 100. We apply the early stopping [\(Prechelt, 1998\)](#page-12-8) to select the model checkpoint with the lowest validation loss on SQuADShifts dataset [\(Miller et al., 2020\)](#page-12-9). Note that the validation set used for checkpoint selection consists solely of English examples.

Hyperparameter Tuning for Hybrid Batch Sampling. To determine the optimal values for the hyperparameters α and β in our hybrid batch sampling approach, we conducted a comprehensive grid search. We evaluated α values ranging from 0 to 1, with β always set to 1 − α. Each configuration was tested on a held-out validation set comprising a diverse selection of languages. We assessed the model's performance across monolingual, cross-lingual, and multilingual retrieval tasks. Our goal was to find a balance that would optimize performance across all three retrieval settings without significantly sacrificing any particular one. We found that setting α = 0.5 provided the best overall results, striking an effective balance between monolingual and cross-lingual/multilingual performance. This equal weighting between monolingual and cross-lingual batches allowed our model to maintain strong monolingual retrieval capabilities while also excelling in cross-lingual and multilingual scenarios. We also observed that the model's performance was relatively stable for α values between 0.4 and 0.6, indicating some robustness to small variations in these hyperparameters.

#### 3.2 EVALUATION

Datasets. We evaluate the retrieval effectiveness of different models on three distinct datasets: XQuAD-R [\(Roy et al., 2020\)](#page-12-0) and MLQA-R [\(Roy et al., 2020\)](#page-12-0).[<sup>2</sup>](#page-3-0) XQuAD-R and MLQA-R are questionanswering datasets with parallel questions and passages in 11 languages and 7 languages, respectively. Thus, these two datasets can be used to evaluate monolingual, cross-lingual, and multilingual retrieval effectiveness. Appendix [A.2](#page-14-0) provides comprehensive details about the evaluation datasets. Furthermore, we report the detailed monolingual retrieval effectiveness on MIRACL dev [\(Zhang](#page-12-10) [et al., 2022\)](#page-12-10) in Table [12](#page-16-0) and [13](#page-16-1) in Appendix [A.3.1.](#page-15-0)

<sup>2</sup>The evaluation of the models is conducted on datasets that are completely separate and distinct from the ones used for training. More specifically, the models have not encountered any data samples, whether from the training or testing splits, of the evaluation datasets during their training process. This ensures an unbiased assessment of the ability of the models to generalize and perform effectively on unseen data.

**224**

**236 237**

**254**

**256**

**259**

Table 1: Main experiments on XQuAD-R and MLQA-R. mAP (marco averaged across all languages) numbers are reported. Mo., CR., and Mul. denote monolingual, cross-lingual, and multilingual retrieval settings. respectively.

| Model Sampling | Mo.  | XQuAD-R Cr. | ( ↑ ) Mul. | Mo.  | MLQA-R Cr. | ( ↑ ) Mul. |
|----------------|------|-------------|------------|------|------------|------------|
| X-X            | .792 | .674        | .547       | .648 | .584       | .473       |
| X-Y XLM-R      | .755 | .700        | .593       | .626 | .620       | .508       |
| Hybrid         | .798 | .705        | .593       | .648 | .623       | .512       |
| X-X            | .808 | .752        | .652       | .681 | .656       | .550       |
| X-Y LaBSE      | .801 | .762        | .679       | .671 | .677       | .576       |
| Hybrid         | .817 | .767        | .682       | .686 | .681       | .579       |

Table 2: Language bias in multilingual retrieval.

| Model Sampling | language XQuAD-R | bias ( ↓ ) MLQA-R |
|----------------|------------------|-------------------|
| X-X            | 410              | 288               |
| X-Y XLM-R      | 295              | 227               |
| Hybrid         | 287              | 227               |
| X-X            | 262              | 225               |
| X-Y LaBSE      | 225              | 198               |
| Hybrid         | 221              | 195               |

Metrics and Settings. We report the mean average precision (mAP) for XQuAD-R and MLQA-R since the metric considers the retrieval quality when multiple relevant passages for a given query exist.[<sup>3</sup>](#page-4-0) We conduct retrieval using the queries with X<sup>Q</sup> language against the corpus with X<sup>C</sup> language and report the macro-averaged mAP over all the cross-lingual (denoting Cr.) combinations language pairs (X<sup>Q</sup> ̸= X<sup>C</sup> ), and the other monolingual (denoting Mo.) combinations (X<sup>Q</sup> = X<sup>C</sup> ). For example, in XQuAD-R (MLQA-R), we have 11 and 7 parallel languages; thus, there are 110 (42) and 11 (7) cross-lingual and monolingual retrieval settings, respectively. For multilingual (denoting Mul.) retrieval, we conduct retrieval using the queries with X<sup>Q</sup> language against all the parallel corpus in different languages. We report the detailed results for specific languages in Section [4.2.](#page-5-0)

# 4 EXPERIMENTAL RESULTS

# 4.1 SUMMARY OF MAIN RESULTS

Zero-shot Retrieval Evaluation. We report the effectiveness of different batch sampling strategies in Table [1.](#page-4-1) We observe that X-X and X-Y sampling only perform well in monolingual and crosslingual retrieval settings, respectively. These results indicate that optimization for either monolingual or cross-lingual retrieval alone may come at the expense of the other. Our hybrid batch sampling, on the other hand, optimizes both retrieval settings. As a result, our hybrid batch sampling achieves the best performance in multilingual retrieval settings, where the ability of the models to handle both monolingual and cross-lingual retrieval tasks is evaluated.[<sup>4</sup>](#page-4-2) Finally, the same conclusion holds when using XLM-R and LaBSE as initialization that hybrid batch sampling is better than the other two baseline batch sampling approaches. A thorough analysis of the retrieval performance across various training batch types, retrieval tasks, languages, and datasets is presented in Section [4.2.1.](#page-5-1)

<sup>3</sup>The results for the Recall metric are in Section [4.2.1.](#page-5-1)

<sup>4</sup>The performance of the models is evaluated on certain languages, such as Greek (el) and Vietnamese (vi), which were not included in the training data. This aspect of the evaluation process aims to assess the ability of the models to handle languages they have not been explicitly trained on, providing insights into their zero-shot cross-lingual transfer capabilities (See Section [4.2.1\)](#page-5-1).

**289 290 291**

**294**

**301**

**304**

**306**

**309**

**314 315**

**318 319**

**321**

In particular, Tables [3](#page-6-0) through [6](#page-7-0) showcase the MAP and Recall scores for zero-shot monolingual, cross-lingual, and multilingual retrieval tasks on the XQuAD-R and MLQA-R datasets, considering both fine-tuned XLM-R and LaBSE models.

Language Bias Evaluation. To gain insight into why hybrid batch sampling achieves strong performance in multilingual retrieval settings, we investigate the language bias exhibited by models fine-tuned using different batch sampling strategies. Following [Huang et al.](#page-11-9) [\(2023b\)](#page-11-9), we measure the language bias using the maximum rank distance among all the parallel corpus. That is, for each query, we calculate the difference between the highest and lowest rank of the relevant passages.[<sup>5</sup>](#page-5-2) We report the macro averaged rank distance across all languages in Table [2](#page-4-3) and present the comprehensive results in Section [4.2.2.](#page-8-0) Specifically, Table [7](#page-9-0) shows the rank distances for the XQuAD-R dataset, while Table [8](#page-9-1) displays the rank distances for the MLQA-R dataset, both considering fine-tuned XLM-R and LaBSE models under different training batch types. As shown in Table [2,](#page-4-3) models fine-tuned with cross-lingual batch sampling show less language bias compared to those fine-tuned with multi-lingual batch sampling. It is worth noting that our hybrid batch sampling, combining both baseline sampling, still maintains low language bias without sacrificing monolingual retrieval effectiveness.

#### 4.2 IN-DEPTH ANALYSIS

# 4.2.1 ZERO-SHOT RETRIEVAL EVALUATION ON XQUAD-R AND MLQA-R

We present the experimental results of our proposed hybrid batching approach for improving the retrieval performance of fine-tuned multilingual language models across various tasks and datasets. We compare our method with two baseline training batch methods (X-X-mono and X-Y) using two pre-trained multilingual language models (XLM-R and LaBSE) on two evaluation datasets (XQuAD-R and MLQA-R). The performance is measured using Mean Average Precision (MAP) and Recall @ 1 (R@1) and Recall @ 10 (R@10) metrics across monolingual, cross-lingual, and multilingual retrieval settings.

Consistent improvement across languages and tasks: Tables [3](#page-6-0) through [6](#page-7-0) demonstrate the performance of the proposed hybrid batching approach when applied to the XLM-R and LaBSE models on the XQuAD-R and MLQA-R datasets. Our method consistently achieves the highest mean MAP and mean R@1 scores across monolingual and cross-lingual settings for all combinations of datasets and models. Furthermore, our proposed method consistently achieves either the highest mean MAP and mean R@10 scores in the multilingual retrieval setting or performs comparably to the X-Y batching method, which is specifically optimized for multilingual retrieval. Notably, there is a substantial performance gap between the second-best approach (either our method or X-Y) and the third-best approach (X-X-mono) in terms of these evaluation metrics for multilingual retrieval. This demonstrates the robustness and effectiveness of the proposed method in improving retrieval performance, regardless of the language or task complexity.

Balanced performance across evaluation metrics: The proposed approach strikes a balance between the X-X-mono (optimized for monolingual retrieval setting) and X-Y (crosslingual/multilingual retrieval settings) baselines. This compromise is evident when analyzing the performance of individual languages across different retrieval tasks. In the monolingual retrieval setting, the proposed method tends to outperform or maintain comparable performance to the X-Xmono baseline for most languages. Similarly, the proposed approach generally surpasses the X-Y baseline across most languages in the cross-lingual and multilingual retrieval settings. A key insight is that in cases where our approach does not achieve the top performance for a specific language and retrieval setting, it consistently performs as a strong runner-up to the approach specifically optimized for that retrieval setting. Simultaneously, our method maintains a significant advantage over the third-best approach in such cases. This trend is consistent for XLM-R and LaBSE models on the XQuAD-R and MLQA-R datasets. By effectively finding a middle ground between the strengths of the X-X-mono and X-Y baselines, the proposed method offers a versatile solution that can handle monolingual, cross-lingual, and multilingual retrieval tasks across a wide range of languages without significantly compromising performance in any particular setting.

<sup>5</sup>Note that in XQuAD-R and MLQA-R, each query only has one relevant passage in each language.

324 Table 3: Performance comparison of MAP and Recall scores across zero-shot monolingual, cross-  
325 lingual, and multilingual retrieval tasks on the XQuAD-R dataset for a fine-tuned XLM-R model and  
326 different training batch types. The best result is highlighted in **bold**, and the second-best result is  
327 underlined.

| Evaluation of Fine-tuned XLM-R Model on XQuAD-R Dataset |               |        |               |               |               |               |              |               |               |  |
|---------------------------------------------------------|---------------|--------|---------------|---------------|---------------|---------------|--------------|---------------|---------------|--|
| MAP                                                     |               |        |               |               |               |               |              |               |               |  |
| Source Language                                         | Monolingual   |        |               | Cross-lingual |               |               | Multilingual |               |               |  |
|                                                         | X-X-mono      | X-Y    | Proposed      | X-X-mono      | X-Y           | Proposed      | X-X-mono     | X-Y           | Proposed      |  |
| ar                                                      | 0.7581        | 0.7318 | <b>0.7619</b> | 0.6064        | <b>0.6607</b> | 0.6564        | 0.487        | <b>0.5519</b> | 0.5416        |  |
| de                                                      | 0.7893        | 0.7694 | <b>0.8033</b> | 0.6979        | 0.7147        | <b>0.7222</b> | 0.5653       | 0.6113        | <b>0.6133</b> |  |
| el                                                      | 0.7749        | 0.7226 | <b>0.7844</b> | 0.6492        | 0.6791        | <b>0.683</b>  | 0.5127       | <b>0.5638</b> | 0.5599        |  |
| en                                                      | 0.8327        | 0.7892 | <b>0.8389</b> | 0.7247        | 0.7319        | <b>0.7473</b> | 0.5984       | 0.631         | <b>0.6436</b> |  |
| es                                                      | 0.8019        | 0.7617 | <b>0.8089</b> | 0.7072        | 0.7178        | <b>0.7332</b> | 0.582        | 0.6123        | <b>0.6245</b> |  |
| hi                                                      | 0.778         | 0.7461 | <b>0.787</b>  | 0.641         | <b>0.6835</b> | 0.676         | 0.5171       | <b>0.5787</b> | 0.5666        |  |
| ru                                                      | 0.802         | 0.7758 | <b>0.8125</b> | 0.694         | 0.7103        | <b>0.7186</b> | 0.5763       | 0.6076        | <b>0.6104</b> |  |
| th                                                      | 0.7634        | 0.7312 | <b>0.7697</b> | 0.6623        | 0.6963        | <b>0.6978</b> | 0.5442       | 0.5862        | <b>0.5876</b> |  |
| tr                                                      | 0.7801        | 0.7479 | <b>0.7913</b> | 0.6748        | 0.7013        | <b>0.7078</b> | 0.5524       | <b>0.6005</b> | 0.5989        |  |
| vi                                                      | <b>0.8113</b> | 0.7624 | <b>0.8025</b> | 0.6742        | <b>0.6904</b> | <b>0.7017</b> | 0.5417       | <b>0.5817</b> | 0.5781        |  |
| zh                                                      | <b>0.8178</b> | 0.771  | 0.8146        | 0.6795        | 0.7105        | <b>0.7144</b> | 0.5496       | <b>0.6023</b> | 0.5957        |  |
| Mean                                                    | 0.7918        | 0.7554 | <b>0.7977</b> | 0.6737        | <b>0.6997</b> | <b>0.7053</b> | 0.5479       | <b>0.5934</b> | 0.5927        |  |
| R@1                                                     |               |        |               |               |               |               |              |               |               |  |
| Monolingual                                             |               |        | Cross-lingual |               |               | Multilingual  |              |               |               |  |
| Source Language                                         | X-X-mono      | X-Y    | Proposed      | X-X-mono      | X-Y           | Proposed      | X-X-mono     | X-Y           | Proposed      |  |
| ar                                                      | 0.6596        | 0.6276 | <b>0.6639</b> | 0.4907        | <b>0.5463</b> | 0.5419        | 0.4272       | <b>0.4811</b> | 0.4722        |  |
| de                                                      | 0.698         | 0.6726 | <b>0.7149</b> | 0.5883        | <b>0.6053</b> | <b>0.6148</b> | 0.4929       | <b>0.5308</b> | <b>0.5322</b> |  |
| el                                                      | 0.6875        | 0.6166 | <b>0.6968</b> | 0.531         | <b>0.5666</b> | <b>0.5726</b> | 0.4495       | <b>0.4904</b> | <b>0.4923</b> |  |
| en                                                      | 0.7523        | 0.6942 | <b>0.7582</b> | 0.62          | <b>0.6246</b> | <b>0.6447</b> | 0.5196       | <b>0.5445</b> | <b>0.5594</b> |  |
| es                                                      | 0.7207        | 0.6624 | <b>0.7232</b> | 0.5986        | <b>0.6096</b> | <b>0.6287</b> | 0.5067       | <b>0.5303</b> | <b>0.5439</b> |  |
| hi                                                      | 0.6881        | 0.6517 | <b>0.6999</b> | 0.5276        | <b>0.574</b>  | <b>0.5664</b> | 0.4514       | <b>0.5043</b> | 0.4957        |  |
| ru                                                      | 0.7108        | 0.6788 | <b>0.7277</b> | 0.5848        | <b>0.5994</b> | <b>0.6115</b> | 0.5047       | <b>0.5299</b> | <b>0.5323</b> |  |
| th                                                      | 0.6703        | 0.6272 | <b>0.6729</b> | 0.5481        | <b>0.5875</b> | 0.5871        | 0.4781       | <b>0.5127</b> | <b>0.5141</b> |  |
| tr                                                      | 0.69          | 0.6453 | <b>0.6959</b> | 0.5669        | 0.5932        | <b>0.6026</b> | 0.4825       | <b>0.5196</b> | <b>0.5219</b> |  |
| vi                                                      | <b>0.7301</b> | 0.6599 | <b>0.7132</b> | 0.5631        | <b>0.5798</b> | <b>0.5949</b> | 0.4703       | <b>0.5038</b> | 0.5015        |  |
| zh                                                      | <b>0.7307</b> | 0.6732 | <b>0.7282</b> | 0.5666        | <b>0.6011</b> | <b>0.6081</b> | 0.4806       | <b>0.523</b>  | 0.5208        |  |
| Mean                                                    | 0.7035        | 0.6554 | <b>0.7086</b> | 0.5623        | <b>0.5898</b> | <b>0.5976</b> | 0.4785       | <b>0.5155</b> | <b>0.5169</b> |  |

354 Table 4: Performance comparison of MAP and Recall scores across zero-shot monolingual, cross-  
355 lingual, and multilingual retrieval tasks on the MLQA-R dataset for a fine-tuned XLM-R model and  
356 different training batch types. The best result is highlighted in **bold**, and the second-best result is  
357 underlined.

| Evaluation of Fine-tuned XLM-R Model on MLQA-R Dataset |               |        |               |               |               |               |              |               |               |  |
|--------------------------------------------------------|---------------|--------|---------------|---------------|---------------|---------------|--------------|---------------|---------------|--|
| MAP                                                    |               |        |               |               |               |               |              |               |               |  |
| Source Language                                        | Monolingual   |        |               | Cross-lingual |               |               | Multilingual |               |               |  |
|                                                        | X-X-mono      | X-Y    | Proposed      | X-X-mono      | X-Y           | Proposed      | X-X-mono     | X-Y           | Proposed      |  |
| ar                                                     | 0.5973        | 0.577  | <b>0.6006</b> | 0.5351        | <b>0.5837</b> | 0.5787        | 0.4091       | <b>0.456</b>  | <b>0.4602</b> |  |
| de                                                     | 0.5915        | 0.5839 | <b>0.5999</b> | 0.6311        | <b>0.6531</b> | <b>0.6687</b> | 0.5095       | <b>0.532</b>  | <b>0.5426</b> |  |
| en                                                     | <b>0.7154</b> | 0.6932 | <b>0.7098</b> | 0.5771        | <b>0.6029</b> | <b>0.604</b>  | 0.4733       | <b>0.5092</b> | <b>0.5143</b> |  |
| es                                                     | <b>0.6829</b> | 0.6649 | <b>0.6809</b> | 0.6328        | <b>0.6528</b> | <b>0.6626</b> | 0.5468       | <b>0.5634</b> | <b>0.5751</b> |  |
| hi                                                     | <b>0.6426</b> | 0.6155 | <b>0.6397</b> | 0.5529        | <b>0.6</b>    | <b>0.6079</b> | 0.4425       | <b>0.4922</b> | <b>0.4949</b> |  |
| vi                                                     | <b>0.6405</b> | 0.6165 | <b>0.6397</b> | 0.573         | <b>0.6122</b> | <b>0.6069</b> | 0.4638       | <b>0.4908</b> | 0.4898        |  |
| zh                                                     | 0.662         | 0.628  | <b>0.6659</b> | 0.588         | <b>0.6352</b> | <b>0.6349</b> | 0.4668       | <b>0.5094</b> | 0.5081        |  |
| Mean                                                   | 0.6475        | 0.6256 | <b>0.6481</b> | 0.5843        | <b>0.62</b>   | <b>0.6234</b> | 0.4731       | <b>0.5076</b> | <b>0.5121</b> |  |
| R@1                                                    |               |        |               |               |               |               |              |               |               |  |
| Monolingual                                            |               |        | Cross-lingual |               |               | Multilingual  |              |               |               |  |
| Source Language                                        | X-X-mono      | X-Y    | Proposed      | X-X-mono      | X-Y           | Proposed      | X-X-mono     | X-Y           | Proposed      |  |
| ar                                                     | <b>0.4971</b> | 0.4778 | <b>0.4952</b> | 0.4142        | <b>0.4639</b> | 0.4583        | 0.528        | <b>0.5817</b> | 0.5811        |  |
| de                                                     | <b>0.4883</b> | 0.4785 | <b>0.498</b>  | 0.5247        | <b>0.5394</b> | <b>0.5599</b> | 0.619        | <b>0.6462</b> | <b>0.6558</b> |  |
| en                                                     | <b>0.6307</b> | 0.6028 | <b>0.6237</b> | 0.4648        | <b>0.4916</b> | <b>0.4939</b> | 0.5833       | <b>0.6222</b> | 0.619         |  |
| es                                                     | 0.58          | 0.56   | <b>0.584</b>  | 0.5174        | <b>0.5434</b> | <b>0.5587</b> | 0.651        | <b>0.6738</b> | <b>0.675</b>  |  |
| hi                                                     | <b>0.5404</b> | 0.5168 | <b>0.5325</b> | 0.4306        | <b>0.4746</b> | <b>0.4821</b> | 0.5656       | <b>0.6187</b> | <b>0.6264</b> |  |
| vi                                                     | <b>0.544</b>  | 0.5108 | <b>0.544</b>  | 0.4536        | <b>0.4969</b> | 0.491         | 0.5752       | <b>0.6076</b> | 0.6058        |  |
| zh                                                     | 0.5437        | 0.5079 | <b>0.5556</b> | 0.4706        | <b>0.5193</b> | <b>0.5295</b> | 0.589        | <b>0.6417</b> | 0.6344        |  |
| Mean                                                   | 0.5463        | 0.5221 | <b>0.5476</b> | 0.468         | <b>0.5042</b> | <b>0.5105</b> | 0.5873       | <b>0.6274</b> | <b>0.6282</b> |  |

Table 5: Performance comparison of MAP and Recall scores across zero-shot monolingual, cross-lingual, and multilingual retrieval tasks on the XQuAD-R dataset for a fine-tuned LaBSE model and different training batch types. The best result is highlighted in **bold**, and the second-best result is underlined.

| Evaluation of Fine-tuned LaBSE Model on XQuAD-R Dataset |               |        |               |               |               |               |              |               |               |  |
|---------------------------------------------------------|---------------|--------|---------------|---------------|---------------|---------------|--------------|---------------|---------------|--|
| MAP                                                     |               |        |               |               |               |               |              |               |               |  |
| Source Language                                         | Monolingual   |        |               | Cross-lingual |               |               | Multilingual |               |               |  |
|                                                         | X-X-mono      | X-Y    | Proposed      | X-X-mono      | X-Y           | Proposed      | X-X-mono     | X-Y           | Proposed      |  |
| ar                                                      | 0.7901        | 0.7848 | <b>0.7963</b> | 0.7257        | <b>0.7351</b> | <b>0.7356</b> | 0.6218       | <b>0.6481</b> | <b>0.6453</b> |  |
| de                                                      | 0.8152        | 0.8135 | <b>0.8222</b> | 0.7667        | <b>0.774</b>  | <b>0.7799</b> | 0.6632       | <b>0.6916</b> | <b>0.6945</b> |  |
| el                                                      | 0.8022        | 0.7991 | <b>0.8211</b> | 0.7483        | <b>0.7603</b> | <b>0.762</b>  | 0.6473       | <b>0.6783</b> | <b>0.6783</b> |  |
| en                                                      | 0.8464        | 0.8349 | <b>0.8536</b> | 0.7932        | 0.7915        | <b>0.8074</b> | 0.6952       | 0.7183        | <b>0.7278</b> |  |
| es                                                      | 0.812         | 0.8186 | <b>0.8331</b> | 0.7724        | <b>0.781</b>  | <b>0.7892</b> | 0.6726       | <b>0.7021</b> | <b>0.7074</b> |  |
| hi                                                      | 0.796         | 0.7824 | <b>0.8211</b> | 0.7382        | <b>0.7459</b> | <b>0.7582</b> | 0.6398       | <b>0.6625</b> | <b>0.6731</b> |  |
| ru                                                      | 0.8243        | 0.8194 | <b>0.8314</b> | 0.7643        | <b>0.7745</b> | <b>0.7784</b> | 0.6684       | <b>0.6945</b> | <b>0.6948</b> |  |
| th                                                      | <b>0.7611</b> | 0.7371 | <b>0.7555</b> | 0.7123        | <b>0.7315</b> | <b>0.7294</b> | 0.6079       | <b>0.6377</b> | <b>0.6372</b> |  |
| tr                                                      | 0.8086        | 0.794  | <b>0.8143</b> | 0.7541        | <b>0.7627</b> | <b>0.7691</b> | 0.655        | <b>0.6824</b> | <b>0.685</b>  |  |
| vi                                                      | 0.8136        | 0.8154 | <b>0.8285</b> | 0.7508        | <b>0.7646</b> | <b>0.7676</b> | 0.6506       | <b>0.6828</b> | <b>0.6809</b> |  |
| zh                                                      | 0.8213        | 0.8096 | <b>0.8249</b> | 0.7451        | <b>0.759</b>  | <b>0.7622</b> | 0.6464       | <b>0.672</b>  | <b>0.6749</b> |  |
| Mean                                                    | 0.8083        | 0.8008 | <b>0.8167</b> | 0.7519        | <b>0.7618</b> | <b>0.7672</b> | 0.6517       | <b>0.6791</b> | <b>0.6817</b> |  |
| R@1                                                     |               |        |               |               |               |               |              |               |               |  |
| R@10                                                    |               |        |               |               |               |               |              |               |               |  |
| Source Language                                         | Monolingual   |        |               | Cross-lingual |               |               | Multilingual |               |               |  |
|                                                         | X-X-mono      | X-Y    | Proposed      | X-X-mono      | X-Y           | Proposed      | X-X-mono     | X-Y           | Proposed      |  |
| ar                                                      | 0.7001        | 0.695  | <b>0.7127</b> | 0.6257        | <b>0.6349</b> | <b>0.6367</b> | 0.5438       | <b>0.5657</b> | <b>0.5671</b> |  |
| de                                                      | 0.7293        | 0.7276 | <b>0.7386</b> | 0.6695        | <b>0.6784</b> | <b>0.6861</b> | 0.5742       | <b>0.6074</b> | <b>0.609</b>  |  |
| el                                                      | 0.7162        | 0.7137 | <b>0.7255</b> | 0.6517        | <b>0.6649</b> | <b>0.668</b>  | 0.5673       | <b>0.5918</b> | <b>0.5967</b> |  |
| en                                                      | 0.77          | 0.7582 | <b>0.7784</b> | 0.6996        | 0.6983        | <b>0.7189</b> | 0.6023       | <b>0.6308</b> | <b>0.6348</b> |  |
| es                                                      | 0.7266        | 0.7401 | <b>0.7603</b> | 0.6752        | <b>0.6889</b> | <b>0.699</b>  | 0.5828       | <b>0.6176</b> | <b>0.6186</b> |  |
| hi                                                      | 0.7025        | 0.6805 | <b>0.721</b>  | 0.6396        | <b>0.6469</b> | <b>0.6623</b> | 0.5599       | <b>0.58</b>   | <b>0.5905</b> |  |
| ru                                                      | 0.7445        | 0.7378 | <b>0.7538</b> | 0.6636        | <b>0.677</b>  | <b>0.6832</b> | 0.5823       | <b>0.6088</b> | <b>0.6066</b> |  |
| th                                                      | <b>0.6703</b> | 0.6331 | <b>0.661</b>  | 0.6108        | <b>0.6326</b> | <b>0.632</b>  | 0.5322       | <b>0.5571</b> | <b>0.5594</b> |  |
| tr                                                      | 0.7221        | 0.701  | <b>0.728</b>  | 0.6561        | <b>0.6679</b> | <b>0.6733</b> | 0.5672       | <b>0.5971</b> | <b>0.5974</b> |  |
| vi                                                      | 0.7276        | 0.7318 | <b>0.7487</b> | 0.6526        | <b>0.669</b>  | <b>0.6732</b> | 0.5661       | <b>0.5979</b> | <b>0.5964</b> |  |
| zh                                                      | 0.7392        | 0.718  | <b>0.7409</b> | 0.6452        | <b>0.6607</b> | <b>0.6684</b> | 0.5624       | <b>0.5882</b> | <b>0.5927</b> |  |
| Mean                                                    | 0.7226        | 0.7124 | <b>0.7335</b> | 0.6536        | <b>0.6654</b> | <b>0.6728</b> | 0.5673       | <b>0.5948</b> | <b>0.5972</b> |  |

Table 6: Performance comparison of MAP and Recall scores across zero-shot monolingual, cross-lingual, and multilingual retrieval tasks on the MLQA-R dataset for a fine-tuned LaBSE model and different training batch types. The best result is highlighted in **bold**, and the second-best result is underlined.

| Evaluation of Fine-tuned LaBSE Model on MLQA-R Dataset |               |        |               |               |               |               |              |               |               |  |
|--------------------------------------------------------|---------------|--------|---------------|---------------|---------------|---------------|--------------|---------------|---------------|--|
| MAP                                                    |               |        |               |               |               |               |              |               |               |  |
| Source Language                                        | Monolingual   |        |               | Cross-lingual |               |               | Multilingual |               |               |  |
|                                                        | X-X-mono      | X-Y    | Proposed      | X-X-mono      | X-Y           | Proposed      | X-X-mono     | X-Y           | Proposed      |  |
| ar                                                     | <b>0.6293</b> | 0.6122 | <b>0.6283</b> | 0.6253        | <b>0.638</b>  | <b>0.6441</b> | 0.5024       | <b>0.5271</b> | <b>0.5206</b> |  |
| de                                                     | 0.6335        | 0.625  | <b>0.6405</b> | 0.6955        | <b>0.7095</b> | <b>0.7153</b> | 0.5756       | <b>0.5967</b> | <b>0.6013</b> |  |
| en                                                     | 0.7347        | 0.7302 | <b>0.751</b>  | 0.6534        | <b>0.6668</b> | <b>0.6733</b> | 0.5558       | <b>0.5787</b> | <b>0.5862</b> |  |
| es                                                     | <b>0.7186</b> | 0.7052 | <b>0.7106</b> | 0.6912        | <b>0.7073</b> | <b>0.709</b>  | 0.6037       | <b>0.6205</b> | <b>0.6235</b> |  |
| hi                                                     | 0.6783        | 0.6894 | <b>0.694</b>  | 0.6478        | <b>0.6707</b> | <b>0.6883</b> | 0.5517       | <b>0.5792</b> | <b>0.5885</b> |  |
| vi                                                     | 0.6699        | 0.663  | <b>0.6883</b> | 0.626         | <b>0.6521</b> | <b>0.6465</b> | 0.5258       | <b>0.5517</b> | <b>0.5573</b> |  |
| zh                                                     | <b>0.7009</b> | 0.6722 | <b>0.6924</b> | 0.6538        | <b>0.6926</b> | <b>0.6914</b> | 0.5375       | <b>0.5743</b> | <b>0.5721</b> |  |
| Mean                                                   | 0.6807        | 0.671  | <b>0.6864</b> | 0.6561        | <b>0.6767</b> | <b>0.6811</b> | 0.5504       | <b>0.5755</b> | <b>0.5785</b> |  |
| R@1                                                    |               |        |               |               |               |               |              |               |               |  |
| R@10                                                   |               |        |               |               |               |               |              |               |               |  |
| Source Language                                        | Monolingual   |        |               | Cross-lingual |               |               | Multilingual |               |               |  |
|                                                        | X-X-mono      | X-Y    | Proposed      | X-X-mono      | X-Y           | Proposed      | X-X-mono     | X-Y           | Proposed      |  |
| ar                                                     | <b>0.53</b>   | 0.5106 | <b>0.5261</b> | 0.5145        | <b>0.5185</b> | <b>0.5359</b> | 0.6152       | <b>0.6438</b> | <b>0.6341</b> |  |
| de                                                     | 0.5352        | 0.5234 | <b>0.5391</b> | 0.593         | <b>0.6021</b> | <b>0.6158</b> | 0.6886       | <b>0.7153</b> | <b>0.7153</b> |  |
| en                                                     | 0.6376        | 0.6324 | <b>0.6672</b> | 0.546         | <b>0.5564</b> | <b>0.5682</b> | 0.6773       | <b>0.6976</b> | <b>0.6987</b> |  |
| es                                                     | <b>0.618</b>  | 0.6    | <b>0.602</b>  | 0.5844        | <b>0.6012</b> | <b>0.6007</b> | 0.7263       | <b>0.7325</b> | <b>0.7358</b> |  |
| hi                                                     | 0.5779        | 0.5878 | <b>0.6036</b> | 0.5371        | <b>0.5572</b> | <b>0.5845</b> | 0.6788       | <b>0.7081</b> | <b>0.7097</b> |  |
| vi                                                     | 0.5636        | 0.5577 | <b>0.591</b>  | 0.5054        | <b>0.542</b>  | <b>0.5318</b> | 0.6523       | <b>0.668</b>  | <b>0.6691</b> |  |
| zh                                                     | <b>0.6071</b> | 0.5556 | <b>0.5873</b> | 0.5412        | <b>0.5853</b> | <b>0.5907</b> | 0.6572       | <b>0.7002</b> | <b>0.6959</b> |  |
| Mean                                                   | 0.5813        | 0.5668 | <b>0.588</b>  | 0.5459        | <b>0.5661</b> | <b>0.5754</b> | 0.6708       | <b>0.6951</b> | <b>0.6941</b> |  |

Zero-shot Generalization to unseen languages. The proposed approach exhibits remarkable zeroshot generalizability, as evidenced by its strong performance across different multilingual pre-trained models and evaluation datasets in Greek (el) and Vietnamese (vi) languages, which were not included in the training data used to develop the model. For example, in Table [5,](#page-7-1) which presents results for the LaBSE model on the XQuAD-R dataset, the proposed method achieves the best MAP and Recall@1 scores for Vietnamese, a low-resource language, in both monolingual and cross-lingual retrieval settings, outperforming the X-X-mono and X-Y approaches. In the multilingual retrieval setting, the proposed approach achieves MAP and R@10 scores of 0.6809 and 0.5964, respectively. These scores are very close to the 0.6828 and 0.5979 achieved by the X-Y model, which is primarily optimized for multilingual retrieval. Additionally, the proposed method significantly outperforms the X-X-mono approach, which is mainly optimized for monolingual retrieval and achieves scores of 0.6506 and 0.5661.

# 4.2.2 LANGUAGE BIAS EVALUATION

Tables [7](#page-9-0) and [8](#page-9-1) present a comprehensive comparison of the average rank distance metric[<sup>6</sup>](#page-8-1) [\(Huang et al.,](#page-11-2) [2023a\)](#page-11-2) across different multilingual retrieval tasks using fine-tuned XLM-R and LaBSE models. The proposed approach is evaluated against two baseline methods: X-X-mono and X-Y, on two datasets: XQuAD-R (Table [7\)](#page-9-0) and MLQA-R (Table [8\)](#page-9-1). The lower the average rank distance, the better the performance.

Significant mitigation of language bias Compared to monolingual batching. The proposed approach substantially reduces language bias compared to the X-X-mono baseline. In Table 1, the proposed method achieves a mean rank distance of 286.6 using XLM-R, compared to 410.2 for X-X-mono, representing a 30.1% reduction in language bias. Similarly, for LaBSE, the proposed approach reduces the mean rank distance by 15.4% (from 261.5 to 221.1). In Table 2 (MLQA-R), the proposed method achieves a mean rank distance of 227.1 using XLM-R, compared to 287.5 for X-X-mono, resulting in a 21% reduction in language bias. For LaBSE, the proposed approach reduces the mean rank distance by 13.4% (from 225.3 to 195). These significant reductions highlight the effectiveness of the proposed method in mitigating language bias of the retrieval system.

Competitive reduction in average rank distance compared to cross-lingual batching. The proposed approach exhibits competitive performance in reducing the average rank distance compared to the strong X-Y baseline. In Table [7](#page-9-0) (XQuAD-R), the proposed method achieves the best mean rank distance of 286.6 using XLM-R, outperforming both X-X-mono (295.4) and X-Y (295.4) baselines. For LaBSE, the proposed approach obtains a mean rank distance of 221.1, which is better than the X-Y baseline (225.2). In Table [8](#page-9-1) (MLQA-R), the proposed method achieves a slightly higher mean rank distance than the X-Y baseline for XLM-R (227.1 vs. 226.7), but outperforms the X-Y baseline for LaBSE (195 vs. 198.3). These results demonstrate that the proposed approach is highly competitive in reducing the average rank distance and can even outperform the strong X-Y baseline in certain cases. This reduction in average rank distance directly translates to a decrease in language bias, as the proposed method effectively brings relevant documents closer together in the retrieval results, regardless of the language.

# 5 CONCLUSION

Developing IR models that can handle queries and documents across many languages is increasingly critical. In this work, we introduced a hybrid batch training strategy to optimize IR systems for monolingual, cross-lingual, and multilingual performance simultaneously. By fine-tuning multilingual language models on a mix of monolingual and cross-lingual question-answer pairs, the models learn robust representations that generalize well across languages and retrieval settings. Extensive experiments demonstrate that this simple yet effective approach consistently matches or outperforms models trained with only monolingual or cross-lingual data, and substantially mitigates the language bias that hinders multilingual retrieval performance.

<sup>6</sup>Rank distance is the average, over all queries and their relevant documents, of the difference between the maximum and minimum ranks assigned by an MLIR model to parallel (semantically similar) relevant documents across different languages.

## 6 LIMITATIONS

This work focuses on optimizing retrieval performance but does not address issues related to result diversity, fairness, or transparency in multilingual settings. For example, it may reflect societal biases present in the training data. Addressing these concerns is important for building equitable multilingual retrieval systems.

Furthermore, the experiments focus only on the XQuAD-R, MLQA-R, and MIRACL benchmark datasets. While these cover a range of languages, they may not be fully representative of real-world multilingual information retrieval needs. The robustness of the results to other domains, question types, and retrieval scenarios is an exciting future direction.

Table 7: Comparison of the rank distances among relevant documents of the XQuAD-R dataset across rank lists generated by fine-tuned XLM-R and LaBSE models for zero-shot multilingual retrieval tasks under different training batch types. The best result is highlighted in **bold**, and the second-best result is underlined.

| Average Rank Distance over XQuAD-R Dataset |          |              |              |          |              |              |
|--------------------------------------------|----------|--------------|--------------|----------|--------------|--------------|
| Source Language                            | XLM-R    |              |              | LaBSE    |              |              |
|                                            | X-X-mono | X-Y          | Proposed     | X-X-mono | X-Y          | Proposed     |
| ar                                         | 552.8    | <b>371.5</b> | <u>376</u>   | 332.4    | <b>279</b>   | <u>285.4</u> |
| de                                         | 356.6    | <u>252.8</u> | <b>242.1</b> | 214.9    | <u>192</u>   | <b>175.1</b> |
| el                                         | 431.6    | <b>307.8</b> | <u>311.9</u> | 251.3    | <b>224.4</b> | <u>228.4</u> |
| en                                         | 320      | <u>239.6</u> | <b>219</b>   | 189.3    | <u>162.1</u> | <b>150</b>   |
| es                                         | 371.4    | <b>264.5</b> | <u>267</u>   | 235.4    | <u>210</u>   | <b>188</b>   |
| hi                                         | 505.6    | <u>368.5</u> | <b>351.7</b> | 299.8    | <b>250.8</b> | <u>255.6</u> |
| ru                                         | 367.9    | <u>271.7</u> | <b>245.6</b> | 226.5    | <u>195.5</u> | <b>189.3</b> |
| th                                         | 431.6    | <u>316.9</u> | <b>304.4</b> | 391.5    | <u>325.9</u> | <b>323.9</b> |
| tr                                         | 422.4    | <u>309</u>   | <b>288.4</b> | 253.8    | <u>225.4</u> | <b>222.9</b> |
| vi                                         | 395      | <b>289.4</b> | <u>295.6</u> | 245.2    | <u>208.6</u> | <b>204.8</b> |
| zh                                         | 357.3    | <u>258.1</u> | <b>251.2</b> | 236.3    | <b>203.9</b> | <u>209</u>   |
| Mean                                       | 410.2    | <u>295.4</u> | <b>286.6</b> | 261.5    | <u>225.2</u> | <b>221.1</b> |

Table 8: Comparison of the rank distances among relevant documents of the MLQA-R dataset across rank lists generated by fine-tuned XLM-R and LaBSE models for zero-shot multilingual retrieval tasks under different training batch types. The best result is highlighted in **bold**, and the second-best result is underlined.

| Average Rank Distance over MLQA-R Dataset |          |              |              |          |              |              |
|-------------------------------------------|----------|--------------|--------------|----------|--------------|--------------|
| Source Language                           | XLM-R    |              |              | LaBSE    |              |              |
|                                           | X-X-mono | X-Y          | Proposed     | X-X-mono | X-Y          | Proposed     |
| ar                                        | 298.2    | <u>248.1</u> | <b>247</b>   | 245.7    | <u>223.5</u> | <b>208.9</b> |
| de                                        | 248.4    | <u>219.7</u> | <b>211.5</b> | 204.1    | <b>179.9</b> | <u>194.7</u> |
| en                                        | 458.4    | <u>371.6</u> | <b>366.9</b> | 340.6    | <u>304</u>   | <b>291.3</b> |
| es                                        | 179.7    | <b>146.7</b> | <u>135</u>   | 152.6    | <u>145</u>   | <b>143.6</b> |
| hi                                        | 275      | <u>200.1</u> | <b>199</b>   | 204.8    | <u>186.1</u> | <b>160.6</b> |
| vi                                        | 296.6    | <b>213.2</b> | <u>223.4</u> | 225.2    | <b>194.6</b> | <u>205.5</u> |
| zh                                        | 255.9    | <b>187.4</b> | <u>207.2</u> | 204.4    | <b>155.1</b> | <u>160.7</u> |
| Mean                                      | 287.5    | <b>226.7</b> | <u>227.1</u> | 225.3    | <u>198.3</u> | <b>195</b>   |

**554 555 556**

**559**

**561**

**564**

**569**

**579**

**584**

# REFERENCES


[1] Mikel Artetxe, Sebastian Ruder, and Dani Yogatama. On the cross-lingual transferability of monolingual representations. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pp. 4623–4637, Online, July 2020. Association for Computational Linguistics. URL <https://aclanthology.org/2020.acl-main.421>. Luiz Bonifacio, Vitor Jeronymo, Hugo Queiroz Abonizio, Israel Campiotti, Marzieh Fadaee, Roberto Lotufo, and Rodrigo Nogueira. mMarco: A multilingual version of the ms marco passage ranking dataset. *arXiv preprint arXiv:2108.13897*, 2021. URL [https://arxiv.org/abs/2108.](https://arxiv.org/abs/2108.13897) [13897](https://arxiv.org/abs/2108.13897). Zewen Chi, Li Dong, Furu Wei, Nan Yang, Saksham Singhal, Wenhui Wang, Xia Song, Xian-Ling Mao, Heyan Huang, and Ming Zhou. InfoXLM: An information-theoretic framework for cross-lingual language model pre-training. In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, pp. 3576–3588, Online, June 2021. Association for Computational Linguistics. URL <https://aclanthology.org/2021.naacl-main.280>. Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzman, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. Unsupervised ´ cross-lingual representation learning at scale. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pp. 8440–8451, Online, July 2020. Association for Computational Linguistics. URL <https://aclanthology.org/2020.acl-main.747>. Marta R Costa-jussa, James Cross, Onur ` C¸ elebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, et al. No language left behind: Scaling human-centered machine translation. *arXiv preprint arXiv:2207.04672*, 2022. URL [https:](https://arxiv.org/abs/2207.04672) [//arxiv.org/abs/2207.04672](https://arxiv.org/abs/2207.04672). Maxime De Bruyn, Ehsan Lotfi, Jeska Buhmann, and Walter Daelemans. MFAQ: a multilingual FAQ dataset. In *Proceedings of the 3rd Workshop on Machine Reading for Question Answering*, pp. 1–13, Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. URL <https://aclanthology.org/2021.mrqa-1.1>. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pp. 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. URL [https://aclanthology.org/](https://aclanthology.org/N19-1423) [N19-1423](https://aclanthology.org/N19-1423). Angela Fan, Shruti Bhosale, Holger Schwenk, Zhiyi Ma, Ahmed El-Kishky, Siddharth Goyal, Mandeep Baines, Onur Celebi, Guillaume Wenzek, Vishrav Chaudhary, Naman Goyal, Tom Birch, Vitaliy Liptchinsky, Sergey Edunov, Michael Auli, and Armand Joulin. Beyond english-centric multilingual machine translation. *Journal of Machine Learning Research*, 22(107):1–48, 2021. URL <http://jmlr.org/papers/v22/20-1307.html>. Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Naveen Arivazhagan, and Wei Wang. Language-agnostic BERT sentence embedding. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 878–891, Dublin, Ireland, May 2022. Association for Computational Linguistics. URL [https://aclanthology.org/2022.](https://aclanthology.org/2022.acl-long.62) [acl-long.62](https://aclanthology.org/2022.acl-long.62). Sebastian Hofstatter, Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy Lin, and Allan Hanbury. Efficiently ¨ teaching an effective dense retriever with balanced topic aware sampling. In *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval*, SIGIR '21, pp. 113–122, New York, NY, USA, 2021. Association for Computing Machinery. URL <https://doi.org/10.1145/3404835.3462891>. Xiyang Hu, Xinchi Chen, Peng Qi, Deguang Kong, Kunlun Liu, William Yang Wang, and Zhiheng Huang. Language agnostic multilingual information retrieval with contrastive learning. In *Findings*

[2] **604**

[3] **606**

[4] **614 615**

[5] **617**

[6] **619**

[7] **629**

[8] **634**

[9] **636**

[10] *of the Association for Computational Linguistics: ACL 2023*, pp. 9133–9146, Toronto, Canada, July 2023. Association for Computational Linguistics. URL [https://aclanthology.org/](https://aclanthology.org/2023.findings-acl.581)

[2023.findings-acl.581](https://aclanthology.org/2023.findings-acl.581). Zhiqi Huang, Hansi Zeng, Hamed Zamani, and James Allan. Soft prompt decoding for multilingual dense retrieval. In *Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval*, SIGIR '23, pp. 1208–1218, New York, NY, USA, 2023a. Association for Computing Machinery. ISBN 9781450394086. URL [https://doi.org/10.](https://doi.org/10.1145/3539618.3591769) [1145/3539618.3591769](https://doi.org/10.1145/3539618.3591769). Zhiqi Huang, Hansi Zeng, Hamed Zamani, and James Allan. Soft prompt decoding for multilingual dense retrieval. In *Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval*, pp. 1208–1218, New York, NY, USA, 2023b. Association for Computing Machinery. URL <https://doi.org/10.1145/3539618.3591769>. Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 6769–6781, Online, November 2020. Association for Computational Linguistics. URL <https://aclanthology.org/2020.emnlp-main.550>. Young Jin Kim, Ammar Ahmad Awan, Alexandre Muzio, Andres Felipe Cruz Salinas, Liyang Lu, Amr Hendy, Samyam Rajbhandari, Yuxiong He, and Hany Hassan Awadalla. Scalable and efficient moe training for multitask multilingual models. *arXiv preprint arXiv:2109.10465*, 2021. URL <https://arxiv.org/abs/2109.10465>. Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question answering research. *Transactions of the Association for Computational Linguistics*, 7:452–466, 2019. URL [https://aclanthology.](https://aclanthology.org/Q19-1026) [org/Q19-1026](https://aclanthology.org/Q19-1026). Dawn Lawrie, Eugene Yang, Douglas W Oard, and James Mayfield. Neural approaches to multilingual information retrieval. In *European Conference on Information Retrieval*, pp. 521–536. Springer, 2023. URL [https://link.springer.com/chapter/10.1007/](https://link.springer.com/chapter/10.1007/978-3-031-28244-7_33) [978-3-031-28244-7\\_33](https://link.springer.com/chapter/10.1007/978-3-031-28244-7_33). Patrick Lewis, Barlas Oguz, Ruty Rinott, Sebastian Riedel, and Holger Schwenk. MLQA: Evaluating cross-lingual extractive question answering. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pp. 7315–7330, Online, July 2020. Association for Computational Linguistics. URL <https://aclanthology.org/2020.acl-main.653>. Sheng-Chieh Lin, Jheng-Hong Yang, and Jimmy Lin. In-batch negatives for knowledge distillation with tightly-coupled teachers for dense retrieval. In *Proceedings of the 6th Workshop on Representation Learning for NLP (RepL4NLP-2021)*, pp. 163–173, Online, August 2021. Association for Computational Linguistics. URL <https://aclanthology.org/2021.repl4nlp-1.17>. Sheng-Chieh Lin, Amin Ahmad, and Jimmy Lin. mAggretriever: A simple yet effective approach to zero-shot multilingual dense retrieval. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pp. 11688–11696, Singapore, December 2023a. Association for Computational Linguistics. URL [https://aclanthology.org/2023.emnlp-main.](https://aclanthology.org/2023.emnlp-main.715)

[715](https://aclanthology.org/2023.emnlp-main.715). Sheng-Chieh Lin, Minghan Li, and Jimmy Lin. Aggretriever: A simple approach to aggregate textual representations for robust dense passage retrieval. *Transactions of the Association for Computational Linguistics*, 11:436–452, 2023b. URL [https://aclanthology.org/2023.](https://aclanthology.org/2023.tacl-1.26) [tacl-1.26](https://aclanthology.org/2023.tacl-1.26). Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In *International Conference on Learning Representations (ICLR)*, 2018. URL [https://openreview.net/pdf?](https://openreview.net/pdf?id=Bkg6RiCqY7) [id=Bkg6RiCqY7](https://openreview.net/pdf?id=Bkg6RiCqY7).

[13] **654**

[14] **656**

[15] **659**

[16] **661**

[17] **664 665**

[18] **669**

[19] **674**

[20] **684**

[21] **686**

[22] **689 690 691**

[23] John Miller, Karl Krauth, Benjamin Recht, and Ludwig Schmidt. The effect of natural distribution shift on question answering models. In Hal Daume III and Aarti Singh (eds.), ´ *Proceedings of the 37th International Conference on Machine Learning*, volume 119 of *Proceedings of Machine Learning Research*, pp. 6905–6916. PMLR, 13–18 Jul 2020. URL [https://proceedings.](https://proceedings.mlr.press/v119/miller20a.html) [mlr.press/v119/miller20a.html](https://proceedings.mlr.press/v119/miller20a.html). Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernandez Abrego, Ji Ma, Vincent Zhao, Yi Luan, Keith Hall, Ming-Wei Chang, and Yinfei Yang. Large dual encoders are generalizable retrievers. In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pp. 9844–9855, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. URL <https://aclanthology.org/2022.emnlp-main.669>. Lutz Prechelt. *Early Stopping - But When?*, pp. 55–69. Springer Berlin Heidelberg, Berlin, Heidelberg, 1998. URL [https://doi.org/10.1007/3-540-49430-8\\_3](https://doi.org/10.1007/3-540-49430-8_3). Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD: 100,000+ questions for machine comprehension of text. In *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing*, pp. 2383–2392, Austin, Texas, November 2016. Association for Computational Linguistics. URL <https://aclanthology.org/D16-1264>. Uma Roy, Noah Constant, Rami Al-Rfou, Aditya Barua, Aaron Phillips, and Yinfei Yang. LAReQA: Language-agnostic answer retrieval from a multilingual pool. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 5919–5930, Online, November 2020. Association for Computational Linguistics. URL [https://aclanthology.](https://aclanthology.org/2020.emnlp-main.477) [org/2020.emnlp-main.477](https://aclanthology.org/2020.emnlp-main.477). Amrita Saha, Rahul Aralikatte, Mitesh M. Khapra, and Karthik Sankaranarayanan. DuoRC: Towards complex language understanding with paraphrased reading comprehension. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 1683–1693, Melbourne, Australia, July 2018. Association for Computational Linguistics. URL <https://aclanthology.org/P18-1156>. Christopher Sciavolino, Zexuan Zhong, Jinhyuk Lee, and Danqi Chen. Simple entity-centric questions challenge dense retrievers. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pp. 6138–6148, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. URL [https://aclanthology.org/](https://aclanthology.org/2021.emnlp-main.496) [2021.emnlp-main.496](https://aclanthology.org/2021.emnlp-main.496). Adam Trischler, Tong Wang, Xingdi Yuan, Justin Harris, Alessandro Sordoni, Philip Bachman, and Kaheer Suleman. NewsQA: A machine comprehension dataset. In *Proceedings of the 2nd Workshop on Representation Learning for NLP*, pp. 191–200, Vancouver, Canada, August 2017. Association for Computational Linguistics. URL <https://aclanthology.org/W17-2623>. Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. Text embeddings by weakly-supervised contrastive pre-training. *arXiv preprint arXiv:2212.03533*, 2022. URL <https://arxiv.org/abs/2212.03533>. Yi Yang, Wen-tau Yih, and Christopher Meek. WikiQA: A challenge dataset for open-domain question answering. In *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing*, pp. 2013–2018, Lisbon, Portugal, September 2015. Association for Computational Linguistics. URL <https://aclanthology.org/D15-1237>. Xinyu Zhang, Xueguang Ma, Peng Shi, and Jimmy Lin. Mr. TyDi: A multi-lingual benchmark for dense retrieval. In *Proceedings of the 1st Workshop on Multilingual Representation Learning*, pp. 127–137, Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. URL <https://aclanthology.org/2021.mrl-1.12>. Xinyu Zhang, Nandan Thakur, Odunayo Ogundepo, Ehsan Kamalloo, David Alfonso-Hermelo, Xiaoguang Li, Qun Liu, Mehdi Rezagholizadeh, and Jimmy Lin. Making a MIRACL: Multilingual information retrieval across a continuum of languages. *arXiv preprint arXiv:2210.09984*, 2022. URL <https://arxiv.org/abs/2210.09984>.

[24] **704**

[25] **706**

[26] **709**

[27] **721**

[28] **724**

[29] **729 730**

[30] **754**

[31] Xinyu Zhang, Kelechi Ogueji, Xueguang Ma, and Jimmy Lin. Toward best practices for training multilingual dense retrieval models. *ACM Transactions on Information Systems*, 42(2), sep 2023. URL <https://doi.org/10.1145/3613447>.
#### A APPENDIX

We provide additional information and detailed experimental results to support the main findings discussed in the body of the manuscript. It is organized into three main parts: [\(A.1\)](#page-13-1) a description of the training datasets used to fine-tune the multilingual models, [\(A.2\)](#page-14-0) an overview of the evaluation datasets and their characteristics, and [\(A.3\)](#page-15-1) supplementary experimental results.

#### A.1 TRAINING DATASETS

We present an overview of the training datasets used to fine-tune the multilingual pre-trained models. These datasets were selected to cover a diverse range of domains, tasks, and languages. These datasets vary in size, language coverage, and domain. The datasets mMARCO, Mr. Tydi, and MFAQ focus on multilingual tasks, while others like Google NQ, DuoRC, and NewsQA are monolingual. The datasets cover different domains, such as web search queries (Google NQ, WikiQA), movie plots (DuoRC), news articles (NewsQA), and FAQs (MFAQ).

- DuoRC: A paraphrased reading comprehension dataset aimed at evaluating complex language understanding. It contains over 186K question-answer pairs created from 7680 pairs of movie plot summaries [\(Saha et al., 2018\)](#page-12-3).
- EntityQuestions: A dataset designed to challenge dense retrievers with simple entity-centric questions. It contains over 14K questions that require retrieving relevant entities from Wikipedia [\(Sciavolino et al., 2021\)](#page-12-4).
- Google NQ: A QA dataset consisting of aggregated queries from Google's search engine, with annotated answers from Wikipedia pages. It contains over 300K queries and can be used for open-domain QA research [\(Kwiatkowski et al., 2019\)](#page-11-7).
- MFAQ: A multilingual FAQ dataset containing over 100K question-answer pairs from 21 languages, covering topics like COVID-19, climate change, and more. It can be used for multilingual FAQ retrieval tasks [\(De Bruyn et al., 2021\)](#page-10-9).
- mMARCO: A multilingual version of the MS MARCO passage ranking dataset, containing over 500K parallel queries and 9M passages in 13 languages. It can be used for multilingual information retrieval research [\(Bonifacio et al., 2021\)](#page-10-8).
- Mr. Tydi: A multi-lingual benchmark for dense retrieval, consisting of monolingual and bilingual topic-document annotations in 11 languages. It's designed to evaluate the performance of multilingual dense retrieval models [\(Zhang et al., 2021\)](#page-12-5).
- NewsQA: A machine comprehension dataset containing over 100K question-answer pairs based on CNN articles, aiming to encourage research on question answering from news articles [\(Trischler](#page-12-6) [et al., 2017\)](#page-12-6).
- WikiQA: An open-domain QA dataset with over 3K questions collected from Bing query logs, paired with answers extracted from Wikipedia. It's designed to be a challenge dataset for opendomain QA research [\(Yang et al., 2015\)](#page-12-7).
- Yahoo QA: A dataset mined from Yahoo Answers, a QA website containing pairs of questions and answers.

Table [9](#page-14-1) presents the dataset sizes after applying our in-house data processing pipeline to filter and clean the data. To expand the training data and cover a diverse set of languages, we employed an in-house machine translation pipeline [\(Fan et al., 2021;](#page-10-5) [Kim et al., 2021;](#page-11-5) [Costa-jussa et al., 2022\)](#page-10-6). ` This pipeline was used to create parallel question-answer pairs across nine languages for the following monolingual datasets: WikiQA, DuoRC, NewsQA, Google NQ, Yahoo QA, and EntityQuestions. For the multilingual datasets, namely Mr. Tydi and MFAQ, only the English version was used. Additionally, mMARCO [\(Bonifacio et al., 2021\)](#page-10-8), a multilingual version of the MS MARCO dataset, was included in the training data.

**759**

**761**

**764**

**766**

**769**

**779 780 781**

**784**

**804 805 806**

Table 9: Training data statistics.

| Dataset Name    | Size per Language |     |     | Languages |     |     |     |     |     |    |
|-----------------|-------------------|-----|-----|-----------|-----|-----|-----|-----|-----|----|
| WikiQA          | 1,469             | en, | ar, | zh,       | de, | es, | ru, | th, | tr, | hi |
| Mr. Tydi        | 3,547             |     |     |           |     | en  |     |     |     |    |
| DuoRC           | 33,298            | en, | ar, | zh,       | de, | es, | ru, | th, | tr, | hi |
| NewsQA          | 59,496            | en, | ar, | zh,       | de, | es, | ru, | th, | tr, | hi |
| Google NQ       | 113,535           | en, | ar, | zh,       | de, | es, | ru, | th, | tr, | hi |
| Yahoo QA        | 135,557           | en, | ar, | zh,       | de, | es, | ru, | th, | tr, | hi |
| EntityQuestions | 176,975           | en, | ar, | zh,       | de, | es, | ru, | th, | tr, | hi |
| MFAQ            | 3,567,659         |     |     |           |     | en  |     |     |     |    |
| mMARCO          | 39,780,811        |     | en, | ar,       | zh, | de, | es, | ru, | hi  |    |

Table 10: The number of queries and candidate sentences for each language in XQuAD-R and MLQA-R.

|    | #Queries | XQuAD-R #Candidates | #Queries | MLQA-R #Candidates |
|----|----------|---------------------|----------|--------------------|
| ar | 1190     | 1222                | 517      | 2545               |
| de | 1190     | 1276                | 512      | 2362               |
| el | 1190     | 1234                |          |                    |
| en | 1190     | 1180                | 1148     | 6264               |
| es | 1190     | 1215                | 500      | 1787               |
| hi | 1190     | 1244                | 507      | 2426               |
| ru | 1190     | 1219                |          |                    |
| th | 1190     | 852                 |          |                    |
| tr | 1190     | 1167                |          |                    |
| vi | 1190     | 1209                | 511      | 2828               |
| zh | 1190     | 1196                | 504      | 2322               |

#### A.2 EVALUATION DATASETS

We provide a summary of the evaluation datasets employed for conducting a zero-shot evaluation of the models developed in this work. It should be noted that these evaluation datasets were not used during the training phase of the models.

- XQuAD-R and MLQA-R: Two multilingual answer retrieval datasets derived from XQuAD [\(Artetxe et al., 2020;](#page-10-10) [Rajpurkar et al., 2016\)](#page-12-11) and MLQA [\(Lewis et al., 2020\)](#page-11-10). They are designed to evaluate the performance of language-agnostic answer retrieval models. XQuAD-R is an 11-way parallel dataset where each question appears in 11 different languages and has 11 parallel correct answers across the languages. MLQA-R, on the other hand, covers 7 languages and has a variable number (2–4) of parallel correct answers across the corpus, with contexts surrounding the answer sentence not guaranteed to be parallel [\(Roy et al., 2020\)](#page-12-0).
- MIRACL dev: A multilingual information retrieval dataset that covers a continuum of languages, featuring 18 languages with varying amounts of training data. It is designed to evaluate the performance of multilingual information retrieval models in low-resource settings and to facilitate research on cross-lingual transfer learning [\(Zhang et al., 2022\)](#page-12-10).

Table [10](#page-14-2) presents the number of questions and candidate sentences for each language in the XQuAD-R and MLQA-R datasets, while Table [11](#page-15-2) displays the corresponding information for each language in the MIRACL Dev dataset.

**814 815**

**817**

**819**

**829**

**834**

**836**

**854**

**856**

Table 11: The number of queries and candidate sentences for each language in MIRACL Dev dataset.

| Language | MIRACL # Queries | Dev # Candidates |
|----------|------------------|------------------|
| ar       | 2,869            | 2,061,414        |
| bn       | 411              | 297,265          |
| en       | 648              | 32,893,221       |
| es       | 799              | 10,373,953       |
| fa       | 632              | 2,207,172        |
| fi       | 1,271            | 1,883,509        |
| fr       | 343              | 14,636,953       |
| hi       | 350              | 506,264          |
| id       | 960              | 1,446,315        |
| ja       | 860              | 6,953,614        |
| ko       | 213              | 1,486,752        |
| ru       | 1,252            | 9,543,918        |
| sw       | 482              | 131,924          |
| te       | 828              | 518,079          |
| th       | 733              | 542,166          |
| zh       | 393              | 4,934,368        |

#### A.3 SUPPLEMENTARY EXPERIMENTAL RESULTS

We present additional experimental findings that complement the main results discussed in the paper. More specifically, we present zero-shot monolingual retrieval evaluation on the MIRACL dataset, showcasing the proposed approach's performance on a diverse set of languages. These supplementary results offer a more comprehensive understanding of the effectiveness of the proposed method and its ability to generalize across various retrieval tasks and languages.

#### A.3.1 ZERO-SHOT MONOLINGUAL RETRIEVAL EVALUATION ON MIRACL

Tables [12](#page-16-0) and [13](#page-16-1) present the performance evaluation of fine-tuned XLM-R and LaBSE models on the MIRACL Dev dataset for zero-shot monolingual retrieval tasks across 15 languages. The models are evaluated using nDCG@10 and Recall@100 metrics, and the results are compared for three different training batch types: X-X-mono, X-Y, and the proposed hybrid batching approach.

When analyzing the performance of the XLM-R model, as shown in Table [12,](#page-16-0) the proposed approach achieves the second-best results in most cases for both nDCG@10 and Recall@100, often closely following the best-performing X-X-mono batch type. In some instances, such as for the Finnish, Russian, and French languages, the proposed method even surpasses the X-X-mono performance in terms of nDCG@10. Similarly, for languages like Persian, Japanese, and Spanish, the proposed approach outperforms X-X-mono in terms of Recall@100. Turning to the LaBSE model, presented in Table [13,](#page-16-1) the proposed approach frequently obtains the second-best results in both metrics and occasionally outperforms the X-X-mono batch type. This is particularly evident for the French, Chinese, Hindi, and Spanish languages in terms of nDCG@10, and for Chinese and Persian in terms of Recall@100.

For both XLM-R (Table [12\)](#page-16-0) and LaBSE (Table [13\)](#page-16-1) models, the proposed approach achieves higher mean and median scores compared to the X-Y batch type in nDCG@10 and Recall@100 metrics, indicating its superior overall performance. Although the X-X-mono batch type generally outperforms the proposed approach in terms of mean scores for both models and metrics, it is important to note that X-X-mono is specifically designed to optimize monolingual retrieval only. In contrast, the proposed hybrid batching approach is optimized for both monolingual and cross-lingual/multilingual retrieval.

Table 12: Performance comparison of nDCG and Recall scores across zero-shot monolingual retrieval tasks on the MIRACL Dev dataset for a fine-tuned XLM-R model and different training batch types. The best result is highlighted in **bold**, and the second-best result is underlined.

| Evaluation of Fine-tuned XLM-R Model on MIRACL Dev Dataset |               |               |               |               |               |               |
|------------------------------------------------------------|---------------|---------------|---------------|---------------|---------------|---------------|
| Source Language                                            | nDCG@10       |               |               | Recall@100    |               |               |
|                                                            | X-X-mono      | X-Y           | Proposed      | X-X-mono      | X-Y           | Proposed      |
| sw                                                         | 0.3319        | <b>0.3531</b> | <u>0.3348</u> | <u>0.6478</u> | <b>0.6503</b> | <u>0.6416</u> |
| bn                                                         | <b>0.5082</b> | 0.4442        | <u>0.4972</u> | <b>0.8738</b> | 0.8114        | <u>0.8621</u> |
| hi                                                         | <b>0.4144</b> | 0.3758        | <u>0.4071</u> | <b>0.7863</b> | 0.741         | <u>0.7706</u> |
| ko                                                         | <b>0.4364</b> | 0.4098        | <u>0.4261</u> | <b>0.7881</b> | 0.7204        | <u>0.783</u>  |
| th                                                         | <b>0.5351</b> | 0.5072        | <u>0.5116</u> | <b>0.8727</b> | <u>0.8655</u> | <u>0.8564</u> |
| te                                                         | <b>0.5407</b> | 0.4511        | <u>0.4843</u> | <b>0.8671</b> | 0.7937        | <u>0.8366</u> |
| fi                                                         | 0.4658        | <b>0.5154</b> | <u>0.4791</u> | 0.8119        | <b>0.845</b>  | <u>0.8224</u> |
| ja                                                         | <b>0.4294</b> | 0.4016        | <u>0.4189</u> | <u>0.7987</u> | 0.7786        | <b>0.804</b>  |
| es                                                         | <u>0.2994</u> | <b>0.3098</b> | <u>0.2989</u> | 0.62          | <u>0.6237</u> | <b>0.624</b>  |
| fr                                                         | 0.273         | <b>0.3044</b> | <u>0.2833</u> | <u>0.6968</u> | <b>0.7171</b> | <u>0.6674</u> |
| ru                                                         | 0.3317        | <b>0.3669</b> | <u>0.3444</u> | <u>0.6763</u> | <b>0.7169</b> | <u>0.6862</u> |
| zh                                                         | <b>0.3873</b> | 0.3438        | <u>0.3627</u> | <b>0.7983</b> | 0.7465        | <u>0.797</u>  |
| fa                                                         | <b>0.4113</b> | 0.37          | <u>0.3937</u> | <u>0.786</u>  | 0.7512        | <b>0.7958</b> |
| ar                                                         | <b>0.5403</b> | 0.4998        | <u>0.5203</u> | <b>0.8693</b> | 0.8152        | <u>0.8629</u> |
| id                                                         | 0.317         | <b>0.3363</b> | <u>0.3185</u> | 0.631         | <b>0.6539</b> | <u>0.6327</u> |
| Mean                                                       | <b>0.4148</b> | 0.3993        | <u>0.4054</u> | <b>0.7683</b> | 0.7487        | <u>0.7628</u> |
| Median                                                     | <b>0.4144</b> | 0.3758        | <u>0.4071</u> | <u>0.7881</u> | 0.7465        | <b>0.7958</b> |

Table 13: Performance comparison of nDCG and Recall scores across zero-shot monolingual retrieval tasks on the MIRACL Dev dataset for a fine-tuned LaBSE model and different training batch types. The best result is highlighted in **bold**, and the second-best result is underlined.

| Evaluation of Fine-tuned LaBSE Model on MIRACL Dev Dataset |               |               |               |               |               |               |
|------------------------------------------------------------|---------------|---------------|---------------|---------------|---------------|---------------|
| Source Language                                            | nDCG@10       |               |               | Recall@100    |               |               |
|                                                            | X-X-mono      | X-Y           | Proposed      | X-X-mono      | X-Y           | Proposed      |
| sw                                                         | <b>0.5076</b> | 0.4883        | <u>0.4896</u> | <b>0.8561</b> | 0.8177        | <u>0.8265</u> |
| bn                                                         | <b>0.5598</b> | 0.5155        | <u>0.5337</u> | <b>0.9194</b> | 0.8881        | <u>0.9048</u> |
| hi                                                         | <u>0.4325</u> | 0.3999        | <b>0.4381</b> | <b>0.7961</b> | 0.7655        | <u>0.7959</u> |
| ko                                                         | <b>0.4589</b> | 0.3963        | <u>0.4386</u> | <b>0.8253</b> | 0.7441        | <u>0.7903</u> |
| th                                                         | <b>0.5738</b> | 0.5285        | <u>0.5449</u> | <b>0.9013</b> | <u>0.8591</u> | <u>0.8585</u> |
| te                                                         | <b>0.5658</b> | 0.5013        | <u>0.5343</u> | <b>0.8768</b> | 0.8366        | <u>0.8458</u> |
| fi                                                         | <b>0.5327</b> | 0.506         | <u>0.5062</u> | <b>0.8631</b> | <u>0.8387</u> | <u>0.8303</u> |
| ja                                                         | <b>0.4333</b> | 0.3834        | <u>0.4027</u> | <b>0.822</b>  | 0.7574        | <u>0.7884</u> |
| es                                                         | <u>0.3366</u> | 0.323         | <b>0.3396</b> | <b>0.6914</b> | 0.6594        | <u>0.6821</u> |
| fr                                                         | 0.3042        | <u>0.3124</u> | <b>0.3317</b> | <b>0.7472</b> | 0.7444        | <u>0.7448</u> |
| ru                                                         | <b>0.3839</b> | 0.3541        | <u>0.363</u>  | <b>0.7421</b> | 0.7091        | <u>0.7132</u> |
| zh                                                         | <u>0.3768</u> | 0.3431        | <b>0.3912</b> | <u>0.7651</u> | 0.7628        | <b>0.7925</b> |
| fa                                                         | <b>0.4252</b> | 0.3777        | <u>0.4116</u> | <u>0.8103</u> | 0.7815        | <b>0.8189</b> |
| ar                                                         | <b>0.5783</b> | 0.5114        | <u>0.5391</u> | <b>0.8951</b> | 0.8403        | <u>0.8733</u> |
| id                                                         | <b>0.3572</b> | 0.3357        | <u>0.3522</u> | <b>0.6688</b> | 0.648         | <u>0.6656</u> |
| Mean                                                       | <b>0.4551</b> | 0.4184        | <u>0.4411</u> | <b>0.8120</b> | 0.7768        | <u>0.7954</u> |
| Median                                                     | <u>0.4333</u> | 0.3963        | <b>0.4381</b> | <b>0.822</b>  | 0.7655        | <u>0.7959</u> |