# Knowledge Entropy Decay During Language Model Pretraining Hinders New Knowledge Acquisition

| Jiyeon Kim1∗                                      | Hyunji Lee1∗               | Hyowon Cho1      | Joel Jang2   |              |
|---------------------------------------------------|----------------------------|------------------|--------------|--------------|
| Hyeonbin Hwang1                                   | Seungpil Won3              | Youbin Ahn3      | Dohaeng Lee3 | Minjoon Seo1 |
| 1 KAIST AI                                        | 2 University of Washington | 3 LG AI Research |              |              |
| {jiyeon.kim, hyunji.amy.lee, minjoon}@kaist.ac.kr |                            |                  |              |              |

## Abstract

In this work, we investigate how a model's tendency to broadly integrate its parametric knowledge evolves throughout pretraining and how this behavior affects overall performance, particularly in terms of knowledge acquisition and forgetting. We introduce the concept of *knowledge entropy*, which quantifies the range of memory sources the model engages with; high knowledge entropy indicates that the model utilizes a wide range of memory sources, while low knowledge entropy suggests reliance on specific sources with greater certainty. Our analysis reveals a consistent decline in knowledge entropy as pretraining advances. We also find that the decline is closely associated with a reduction in the model's ability to acquire and retain knowledge, leading us to conclude that diminishing knowledge entropy (smaller number of active memory sources) impairs the model's knowledge acquisition and retention capabilities. We find further support for this by demonstrating that increasing the activity of inactive memory sources enhances the model's capacity for knowledge acquisition and retention.1

## 1 Introduction

Recent studies have analyzed how language models store world knowledge in their parameters and utilize this knowledge to generate responses during inference time (Geva et al., 2021; Dai et al., 2022a;b; Meng et al., 2022; Yao et al., 2024). However, little is known about how their behavior of integrating various factual knowledge embedded in their parameters changes throughout the pretraining stage. In this work, we perform a deep analysis of how a model's property of broadly integrating diverse parametric knowledge evolves throughout pretraining and how these shifts affect overall performance, particularly in terms of knowledge acquisition and forgetting in a continual learning setup. We hypothesize that this varying level of integration may explain why models in the later stages of pretraining encounter challenges in acquiring new knowledge (Dohare et al., 2024; Jang et al., 2022; Chang et al., 2024).

We introduce *knowledge entropy*, which reflects how a language model integrates various knowledge sources, to investigate how this behavior evolves throughout pretraining. Recent studies have shown that feed-forward layers (FFNs) serve as a key-value memory (Geva et al., 2022; 2021; Dai et al., 2022a). Building on this research, as shown in Figure 1, we view the second projection matrix V as a memory, composed of memory
∗Denotes equal contribution 1Code in https://github.com/kaistAI/Knowledge-Entropy.git 1

Initial Mid Final
... ... ...

[Continual Learning Performance]
... 

Memory Vectors Knowledge Entropy Forgetting Memory ... 

Coefficients Acquisition
...

Pretraining Initial Mid Final steps Pretraining Initial Final steps Mid
vectors, which store the model's parametric knowledge, and view the first projection matrix K as generating coefficients C¯ that determine how these memory vectors are combined. Knowledge entropy measures how sparsely these memory coefficients are distributed; high knowledge entropy indicates that the model tends to integrate a broad range of memory vectors whereas the model with low knowledge entropy relies on specific memory vectors with high certainty. We analyze models at different stages of pretraining to investigate how knowledge entropy changes throughout pretraining. Our findings show that models in the later stages of pretraining tend to exhibit lower knowledge entropy, suggesting a shift from utilizing a larger set of active memory vectors to a smaller, more focused set as pretraining progresses.

We hypothesize that changes in knowledge entropy would influence the model's behavior when encountering new knowledge. To test this, we conduct a thorough analysis of the model's ability to acquire new knowledge and retain existing knowledge in a continual knowledge learning2scenario on a target corpus (Jang et al., 2022; Wu et al., 2023), starting from different stages of pretraining. This involves further training the pretrained model on new-domain corpora using a language modeling objective to integrate new knowledge. Our results reveal a strong correlation between knowledge entropy and the model's ability to acquire and retain knowledge: both knowledge entropy and knowledge acquisition and retention decrease as the pretraining progresses.

We assume that this correlation arises because lower knowledge entropy indicates a smaller set of active memory vectors, leading to frequent overwriting of these memory vectors to store new knowledge. To test this assumption, we conduct experiments where we artificially increase the activity of previously inactive memory vectors, allowing the model to store new knowledge across a broader range of memory vectors. Surprisingly, we observe that these modified models demonstrate improved knowledge acquisition and reduced forgetting compared to the original models when undergoing continual knowledge learning; though not to the same extent as the original pretrained model with equivalent knowledge entropy. Such a result bolsters our hypothesis that having a limited number of active memory vectors (low knowledge entropy) plays a critical role in explaining the degradation of the model's ability to acquire and retain knowledge as pretraining advances.

2In this work, we use the term "continual knowledge learning" and "continual learning" interchangeably.

Overall, our findings reveal that **as pretraining progresses, models exhibit a narrower integration of** memory vectors, reflected by decreasing **knowledge entropy, which hinders both knowledge acquisition**
and retention. Models in the later stages of pretraining3show low knowledge entropy, leading to poor knowledge acquisition and higher forgetting rates despite being trained on larger datasets. In contrast, earlystage models display high knowledge entropy, enabling better knowledge acquisition and retention; however, their performance is often limited by weaker language modeling capabilities. Thus, mid-stage models strike a balance, showing strong knowledge acquisition and retention along with overall performance, making them a practical choice for further training to incorporate new knowledge. To the best of our knowledge, this is the first work to analyze how a model's behavior in integrating various memory vectors changes across pretraining stages and the subsequent effects on performance when acquiring new knowledge in a continual knowledge learning setup.

## 2 Related Work

Dynamics of Knowledge in Language Models Recent studies have shown that language models embed world knowledge within their parameters and integrate this knowledge to generate responses (Yang, 2024; Petroni et al., 2019; Wang et al., 2021). Thereby, various research efforts aim to understand these dynamics of knowledge in language models (how they learn, store, and engage their parametric knowledge) during inference and training phases. Several studies have focused on investigating the inference process: Geva et al. (2023) analyzes the role of different layers in language models. Allen-Zhu & Li (2024b) demonstrates that model parameters have a limited knowledge capacity. Some studies suggest key-value memory (Geva et al., 2021; Meng et al., 2022; Dai et al., 2022a). Other research focuses on the pretraining phase. Liu et al.

(2021) studies the sequence that language models learn various types of knowledge. Allen-Zhu & Li (2024a)
examines strategies to enhance knowledge storage and extraction. Teehan et al. analyzes internal structural changes. Sun & Dredze (2024) investigates the interaction between pretraining and finetuning. Chang et al.

(2024) analyzes patterns of knowledge acquisition specifically during the pretraining process, addressing the question of how language models acquire knowledge during pretraining. While their study shares similarities with ours in investigating knowledge acquisition behavior during LLM training, our work takes a different focus. We aim to understand why LLMs encounter increasing difficulty in acquiring new knowledge as pretraining progresses, exploring the underlying reasons behind the challenges faced by later-stage models in learning new knowledge. To the best of our knowledge, our work is the first to explore how the behavior of language models in integrating their knowledge evolves throughout the pretraining phase, and to analyze how these changes affect model performance in terms of knowledge acquisition and forgetting in continual knowledge learning.

Entropy in Natural Language Processing In information theory, entropy quantifies the value of information, where predictable (certain) events have low entropy and unpredictable (uncertain) events have high entropy (Lairez, 2022; Majenz, 2018). In natural language processing (NLP), entropy is used in various ways to measure the certainty of language models. Yang (2024) analyzes the entropy of model outputs based on input prompts. Araujo et al. (2022) calculates the entropy of outputs at each layer to determine weight adjustments in a continual learning setup. Other papers focus on token probability entropy to understand the information required to predict the next word in a sequence (Vazhentsev et al.; Geng et al., 2024; Malinin
& Gales, 2021). Lower entropy in a model's predictions may indicate that the model has become more certain about its predictions based on training data. Additionally, Kumar & Sarawagi (2019) measures entropy over the cross-attention layer to assess the uncertainty in the attention layer of encoder-decoder models.

The entropy proposed in our paper, *knowledge entropy*, differs from previous work in that it focuses on the 3 entropy of a model's parametric knowledge, assessing the uncertainty or variability in utilizing the knowledge encoded within the language model.

## 3 Knowledge Entropy

In this section, we introduce knowledge entropy (Section 3.1) to examine how broadly the model integrates its parametric knowledge and describe the experimental setup used to measure it (Section 3.2). Next, in Section 3.3, we measure knowledge entropy at various pretraining stages to analyze how the model's knowledge integration behavior evolves over pretraining. In Section 3.4, we extend our investigation by exploring alternative definitions of entropy.

## 3.1 Definition

In this work, we introduce a new concept, *knowledge entropy*, to analyze the scope of a model's access patterns to its parametric knowledge. Low knowledge entropy suggests that the model relies on a narrower set of specific knowledge sources with high certainty whereas high knowledge entropy indicates that the model integrates with a diverse range of knowledge sources. Inspired by prior research that considers feed-forward layers (FFNs) as *key-value memory* containing a model's parametric memory (Geva et al., 2021; Dai et al., 2022a; Meng et al., 2022; Dong et al., 2022), we consider the knowledge source to be the *memory vectors*,
which is the second projection matrix of FFN. We measure how broadly the model integrates these memory vectors with *memory coefficients*, which are calculated by the first projection matrix and the activation function. Geva et al. (2021) propose the concept of *key-value memory*, demonstrating that FFNs function similarly to the key-value neural memories (Sukhbaatar et al., 2015). The feed-forward layer consists of two projection layers and activation in the middle:

$$F F N(\mathbf{x})=f(\mathbf{x}\cdot K^{T})\cdot V$$
F F N(x) = f(x · KT) · V (1)
where x ∈ R
d. The first projection matrix (K ∈ R
m×d) corresponds to the keys, and the second projection matrix (V ∈ R
m×d) represents the values, or the *memories* comprised of *memory vectors*. The output, F F N(x), is a linear combination of the memory vectors vi=1,··· ,m ∈ R
d which are the rows of V , where the *coefficients*4 C are determined by f(x · KT), with f being a non-linear activation function such as ReLU.

Previous studies have shown that various types of factual and linguistic knowledge are encoded within these memories (Dai et al., 2022a; Geva et al., 2022; Meng et al., 2022; Dong et al., 2022). Thus, the final output is generated by combining the contributions of these memory vectors, where the memory coefficients determine the combination.

Thereby knowledge entropy, H(θ), is calculated by the sum of layer-wise entropy H(θ l), which is based on the average coefficient C¯l ∈ R
m averaged across all tokens in dataset D, as described in Equation 2.

$$\bar{C}^{l}=\frac{1}{|\mathcal{D}|}\sum_{n=1}^{|\mathcal{D}|}\left(\frac{1}{T_{n}}\sum_{j=1}^{T_{n}}C_{n,j}^{(l)}\right);\qquad\text{prob}(\bar{c}_{i}^{l})=\frac{\bar{c}_{i}^{l}}{\sum_{k=1}^{m}\bar{c}_{k}^{l}},\quad\text{for}i=1,2,\ldots,m$$ $$\mathcal{H}(\theta^{l})=-\sum_{i=1}^{m}\text{prob}(\bar{c}_{i}^{l})\cdot\log(\text{prob}(\bar{c}_{i}^{l}));\qquad\mathcal{H}(\theta)=\sum_{l=1}^{L}\mathcal{H}(\theta^{l})$$
$$(1)$$

$${\mathrm{(2)}}$$

1B 7B
0 20 40 60 80 100 Rate of Pre-trained Step (%)
143.6 143.7 143.8 143.9 144.0 297.0 297.2 297.4 297.6 E
nt ro py
(
1 B)
E
nt ro py
(7 B)
Atte n ti o n E
nt r o p y Attention Entropy NTP Entropy 2.4 2.5 2.6 2.7 2.8 0 200 400 600 Pre-trained steps (k)
40 50 60 70 N
TP E
nt ro p y

C
(l)
n,j represents the coefficient of the j-th token position of the n-th instance at layer l, c¯
l iindicates the i-th element from C¯l, Tn is the sequence length of the n-th instance in the dataset D, m is the inner dimension of feed-forward layer, and L denotes the number of layers in the model.

## 3.2 Experiment Setup

To conduct the experiment, we use the OLMo (Groeneveld et al., 2024) models (1B and 7B), which are open-source large language models with intermediate pretraining checkpoints released, trained on the Dolma dataset (Soldaini et al., 2024)5. To measure knowledge entropy, we use a subset of Dolma, 2k instances that appear in the first batch within the official pretraining data order to ensure that all models we are using have seen the corpus during pretraining step. Please note that the trend persists across other corpora as well (Figure 7 in Appendix A.2); however, since we are analyzing the model's behavior throughout training, we define knowledge entropy based on calculations using the training dataset.

In the case of OLMo, the memory coefficient C
(l)
n,j is calculated as C
(l)
n,j = abs(SwiGLU(xj)) where xjis the j-th token of input x and SwiGLU (Shazeer, 2020) is the activation function. We apply the absolute value since the SwiGLU allows negative values and the magnitude determines the contribution of the corresponding memory vector in the linear combination. Then, the absolute values are converted into a probability distribution. We also show experimentally that the trend persists with different choices of activation functions. Further details regarding knowledge entropy can be found in Appendix A.2.

## 3.3 Final Models Tend To Exhibit Lower Knowledge Entropy

Figure 2 illustrates how knowledge entropy (y-axis) changes across different stages of pretraining (x-axis).

The results show a consistent decrease in knowledge entropy **as pretraining progresses** in both 1B and 7B models. This trend suggests that models in the later stages of pretraining tend to engage with a narrower range of memories, relying more heavily on specific memory vectors rather than accessing and integrating knowledge from a broader range of memories. A consistent reduction in knowledge entropy is observed across all layers, with the most significant reduction occurring in the last layer, which closely resembles the output distribution right before the token prediction (Figure 8 in Appendix).

## 3.4 Similar Trends Are Observed By Different Definitions Of Entropy

While our work defines knowledge entropy focusing on the feed-forward layer, previous studies have examined entropy in different contexts, such as the entropy of attention (Kumar & Sarawagi, 2019) and the entropy of next-token prediction over the vocabulary space (Vazhentsev et al.; Malinin & Gales, 2021). To gain a more comprehensive understanding of the model's overall behavior, we extend our analysis by exploring the entropy trends in both attention mechanisms and next-token prediction. The formula and details are provided in Appendix A.3)
Entropy of Attention Layers Following Kumar & Sarawagi (2019), we measure *attention entropy* to capture the degree of uncertainty in attention weights. It is calculated as the sum of layer-wise entropy, where the layer-wise entropy measures the sparsity of the attention weights in each attention head. Thus, *attention* entropy reflects how much weight the model assigns to specific tokens with confidence when generating the next token given the input based on token relationships. Figure 3 shows that attention entropy consistently decreases during pretraining, with a sharp decline in the early stages followed by a more gradual reduction.

This trend suggests that the model learns to focus on contextually important tokens within the attention layer.

Entropy of Next Token Prediction Entropy can also be measured based on the probability distribution of next-token predictions over the vocabulary space (Vazhentsev et al.; Geng et al., 2024; Malinin & Gales, 2021). Figure 3 shows that the entropy of the next token prediction also consistently decreases throughout pretraining, reflecting the model's increasing certainty in its next token prediction.

## 4 Knowledge Acquisition And Forgetting

We hypothesize that the reduction of knowledge entropy as pretraining progresses impacts the model's knowledge acquisition and forgetting as low knowledge entropy indicates sparse activation of memory vectors, thus the vectors are likely to be consistently overwritten when new knowledge is introduced. To test this hypothesis, we measure knowledge acquisition and forgetting using checkpoints from different stages of pretraining in a continual knowledge learning setup (Jang et al., 2022; Wu et al., 2023), where further training is performed on new-domain corpora by next token prediction to inject new knowledge into the pretrained models. Section 4.1 details the experimental setup and the metrics used. In Section 4.2, we present the results of knowledge acquisition and forgetting across various pretraining stages. Finally, Section 4.3 further explores whether a relationship exists between the two behaviors: activating the inactive memory vectors increases the knowledge acquisition ability.

## 4.1 Experiment Setup

Model & Hyperparameters We experiment using intermediate checkpoints from OLMo6. Hyperparameters are chosen following previous research on continual knowledge learning (Jang et al., 2022; Kim et al., 2023) and we test various combinations to assess their generalizability. For batch size, we test 128 and 2048; for learning rate, we experiment with 1e-4, 4e-4, and 1e-3. We also investigate the effect of training duration by comparing a single epoch to three epochs. Among these configurations, we focus primarily on a batch size of 128, a learning rate of 4e-4, and single-epoch training as this setup most closely aligns with continual knowledge learning.

Dataset We experiment on a subset of two datasets7: PubMed 8, a corpus of bio-medical and life science topics with abstracts, and C4 (Raffel et al., 2020), a large-scale corpus comprising diverse text data gathered from web pages. We use PubMed as the primary dataset as it contains more new knowledge, making it a better fit for our continual knowledge learning setup (Appendix B.1). In addition to the dataset, we inject synthetic knowledge during training to assess the model's ability to acquire new information. Specifically, we utilize FICTIONAL KNOWLEDGE dataset (Chang et al., 2024), which is designed to assess how well language models acquire factual knowledge during pretraining9. This dataset includes 130 paragraphs about fictional yet realistic entities and 1,950 probes where each paragraph contains 15 different probes. The passages are incorporated into the training batch 10 times during the continual knowledge learning. After training, we evaluate the models on evaluation probes of the Fictional Knowledge dataset to measure knowledge acquisition, and evaluate on six downstream tasks in zero-shot manner (Sun & Dredze, 2024; Groeneveld et al., 2024) to measure knowledge forgetting (SciQ (Welbl et al., 2017), Winogrande (Sakaguchi et al., 2021), PIQA (Bisk et al., 2020), OBQA (Mihaylov et al., 2018), HellaSwag (Zellers et al., 2019), and ARC Easy (Clark et al., 2018)). Detailed explanation is included in Appendix B.1. Metric Knowledge acquisition of a language model θ is measured with the probing performance on evaluation probes following Chang et al. (2024). When given the injected knowledge W, each instance wiin a corpus has a corresponding set of probes Pwi, containing 15 different probes. To measure how well the model recalls the injected knowledge, we compute the **probe performance** K(θ), the average log probability ℓ(pi;θ) of the target span for each probe pi ∈ Pwiacross all instances in wi ∈ W and calculate average; K(θ) = 1 |W| Pwi∈W1 |Pwi | Ppi∈Pwi ℓ(pi; θ). The **knowledge acquisition metric** A(θ) is defined as the improvement rate of K(θ) from θPT to θCL, where θPT represents the model checkpoint from a pretraining step, which serves as the starting point and θCL represents the model after continual knowledge learning. High A(θ) indicates the model has learned new knowledge well.

To measure knowledge forgetting of a language model, we measure average performance over six downstream tasks P(θ). **Knowledge forgetting** F(θ) is calculated by the reduction rate from θPT to θCL. Low F(θ) indicates that the models have retained their existing knowledge. The equation and a detailed explanation are presented in Appendix B.2.

## 4.2 Knowledge Acquisition And Retention Decreases Across Pretraining Stage

Figure 4a shows the performance of OLMo 1B and 7B models 10 from various stages of pretraining as an initial state. **We observe that models in the final stages of pretraining struggle more with acquiring** new knowledge A(θ) **and exhibit greater forgetting** F(θ). As shown in Figure 4b, continually training the models at the mid-point of the pretraining as the initial checkpoint tends to yield the best performance in knowledge probing and downstream tasks compared to both models from the initial and final stages of pretraining. While early-stage models demonstrate high knowledge acquisition with minimal forgetting, their overall performance is limited by weaker language modeling capabilities. Conversely, later-stage models, despite being trained on larger datasets, exhibit lower knowledge acquisition and higher rates of forgetting, resulting in lower overall performance compared to the mid-stage models. This aligns with previous research suggesting that a model in the final stage of pretraining tends to struggle when learning new knowledge, showing a trade-off between *plasticity* and *stability* (Dohare et al., 2024; Biesialska et al., 2020; Jang et al.,

5.0 7.5 10.0 12.5 15.0 17.5 20.0 A

c q ui si tio n 
( )

1B ( )
1B ( ) 7B ( )
7B ( )
21 24 27 30 F

or g etti n g 
( )

20 40 60 80 100 Rate of Pre-trained Step (%)
(a)
D

o w n st re a m p erf 
( ) 
(%
)

20 40 60 80 100 Rate of Pre-trained Step (%)
0.30 0.29 0.28 0.27 0.26 0.25 52 54 56 58 60 62 64 K

n o wl e d g e pr o b e 
( )

1B ( ) 1B ( ) 7B ( ) 7B ( )
(b)
2022). Therefore, we suggest that using a mid-stage checkpoint strikes a good balance, making it a practical choice as an initial starting point for further training to inject new knowledge.

We consistently observe this pattern of later-stage models underperforming compared to earlier-stage models across various hyperparameter settings, including batch size, learning rate, training corpus, and the number of epochs. A detailed analysis of these results is provided in Appendix B.4.

## 4.3 Resuscitating Inactive Memory Vectors Increases Knowledge Acquisition

We observe a strong correlation11 between the trend of knowledge entropy (Figure 2) and the model's ability to acquire and retain knowledge (Figure 4a). We assume that the model's increasing reliance on a limited set of memory vectors (as indicated by a decrease in knowledge entropy) leads to more frequent updates to these vectors, making it difficult to acquire new knowledge and resulting in a higher rate of forgetting.

(The intuition behind this hypothesis can be found in Appendix A.1) To test this assumption, we conduct experiments where we artificially increase the activity or resuscitate previously inactive memory vectors.

To resuscitate inactive memory vectors, we modify the up-projection matrix K which engages with producing memory coefficients C¯ (notations from Equation 2). Specifically, as shown in Algorithm 1, we identify the lowest p% (resuscitation ratio) of memory coefficients and apply a multiplier u to parameters in K that are associated with these p%. Multiplier u can take any value; in this experiment, we divide the mean coefficient value of each layer by the respective coefficient value c¯
li at each identified position i at layer l, and then multiply the result by an amplifying factor q. By varying the value of q, we control the degree of resuscitation applied to the p% low-activation coefficients, thereby influencing the magnitude of the average coefficient and the corresponding size of the parameter updates. Figure 5a shows the knowledge acquisition and forgetting rates and Figure 5b presents the knowledge probe and downstream task performance after continual learning with various resuscitation configurations. For the experiment, we fix p to 50 with varying q and use the OLMo checkpoint at the last step of pretraining. Results show that when q is set to 1 or greater, it generally yields better performance in both knowledge acquisition and retention compared to the original model. In contrast, when q is set to 0.5, which further reduces already 8

original q=0.5 q=1 q=2 q=3 0.32 0.31 0.30 0.29 0.28 original q=0.5 q=1 q=2 q=3 20 30 40 50 60 70 40 20 0 20
( )

( )
51.8 51.9 52.0 52.1 52.2 52.3 D

o w n st re a m p erf 
( ) (
%
)

K

n o wl e d g e pro b e 
( )

A

c q ui si ti o n 
( )
( )
( )
F

or g etti n g 
( )

(a)
(b)
inactive memory coefficients, both acquisition and retention decline suggesting that concentrating parameter updates more heavily on already active locations leads to sparser updates, ultimately impairing overall performance. These results suggest that having a narrower active memory vector (low **knowledge entropy)** tends to reduce the model's capacity to acquire new knowledge and increases knowledge forgetting. Further experiments with fixed q and varying p show that increasing p to activate a larger portion of inactive parameters generally leads to improved performance. Detailed result of this configuration is in Appendix B.6.

We also analyze how the result changes when using models from different stages of pretraining as the original model. The trend holds consistently across different checkpoints of pretraining. However, the effect of the resuscitation becomes more pronounced as the original model progresses to later stages of pretraining. Detailed results are included in Appendix B.7. Our result indicates that resuscitating inactive memory vectors of final-stage models tends to enhance knowledge acquisition and overall performance compared to the unmodified final-stage model. However, we observe that the performance remains lower compared to models from the pretraining step with similar knowledge entropy, such as the mid-stage model. This suggests that applying linear scaling to a subset of specific layers alone is insufficient to induce fundamental behavioral changes in the model. In other words, to restore a model that has lost its *plasticity* (Dohare et al., 2024) to its previous state, more fundamental and alternative approaches are required. Further exploring methods for effectively modifying the parameters would be an interesting direction for future work.

## 5 Conclusion

In this work, we examine how large language models' ability to broadly integrate their parametric knowledge
(measured by knowledge entropy) changes throughout pretraining and how these changes affect knowledge acquisition and forgetting in a continual learning setup. Our findings reveal a strong correlation between knowledge entropy and the model's capacity to acquire and retain knowledge. Models in the final stages of pretraining tend to exhibit narrower integration of memory vectors, leading to lower knowledge entropy, which negatively impacts both knowledge acquisition and retention. Interestingly, artificially increasing knowledge entropy by modifying the parameters of final-stage models tends to improve these capabilities. Based on our analysis, we suggest that models from the mid-stage of pretraining offer a good balance between knowledge acquisition, retention, and overall performance, making them a good choice for further training to introduce new knowledge.

Algorithm 1 Resuscitating Low Memory Coefficients Require: C¯ (average coefficients), p (resuscitation ratio), q (amplifying factor), K (up-projection matrix)
Ensure: Scaled up-projection matrix K using computed multiplier u 1: for each layer l in K do 2: Extract average activations for layer l:
C = C¯[l]
3: Compute the threshold t for the lowest p% activations:
t = percentile(C, p)
4: Identify positions of values below the threshold t:

idx = (C ≤ t).nonzero( )
5: Compute the multiplier u for coefficients in layer l:
u =
mean(C)
C× q 6: Apply scaling to the up-projection weights Kl at the identified positions:
Kl[idx, :] ×= u[idx]
7: **end for**

## 6 Limitation & Future Work

Due to computational constraints, our study measures knowledge acquisition and forgetting in a continual learning setup. Future work could explore whether these behaviors also manifest during the pretraining phase. We focused on OLMo 1B and 7B models, as they are the only models that publicly provide intermediate pretraining checkpoints and demonstrate strong performance (Sun & Dredze, 2024; Chang et al., 2024). Extending this investigation to other models would be a valuable direction for further research. Our resuscitation method, which arbitrarily modifies model parameters to test our hypothesis, showed promising results in improving knowledge acquisition and retention. However, performance tended to decline when resuscitating models at their initial or mid-stages. This suggests that more refined methods for resuscitating model parameters—ones that avoid random modification and preserve language modeling capabilities—could yield better outcomes. Additionally, while we observed that models in the mid-stage of pretraining strike a good balance for further training on tasks that involve acquiring new knowledge, defining the mid-point precisely remains an open question. In this study, we approximated the mid-point as 50% of the learning rate schedule.

## Acknowledgments

We thank Sohee Yang, Hoyeon Chang, Seongyun Lee, Seungone Kim, Doyoung Kim, and Hanseok Oh for helpful discussions and constructive feedback.

This work was supported by LG AI Research grant (Self-improving logical reasoning capabilities of LLMs, 2024, 50%) and the Institute of Information & Communications Technology Planning & Evaluation(IITP) grant funded by the Korea government(MSIT) (RS-2024-00397966, Development of a Cybersecurity Specialized RAG-based sLLM Model for Suppressing Gen-AI Malfunctions and Construction of a Publicly Demonstration Platform, 20%; No.2022-0-00113, Developing a Sustainable Collaborative Multi-modal Lifelong Learning Framework, 20%; No.RS-2021-II212068, Artificial Intelligence Innovation Hub, 10%)

## References

AF Agarap. Deep learning using rectified linear units (relu). *arXiv preprint arXiv:1803.08375*, 2018. Zeyuan Allen-Zhu and Yuanzhi Li. Physics of language models: Part 3.1, knowledge storage and extraction.

In *Forty-first International Conference on Machine Learning*, 2024a. URL https://openreview.net/ forum?id=5x788rqbcj.

Zeyuan Allen-Zhu and Yuanzhi Li. Physics of language models: Part 3.3, knowledge capacity scaling laws.

arXiv preprint arXiv:2404.05405, 2024b.

Vladimir Araujo, Julio Hurtado, Alvaro Soto, and Marie-Francine Moens. Entropy-based stability-plasticity for lifelong learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 3721–3728, 2022.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pp. 2397–2430. PMLR, 2023.

Magdalena Biesialska, Katarzyna Biesialska, and Marta Ruiz Costa-jussà. Continual lifelong learning in natural language processing: A survey. *ArXiv*, abs/2012.09823, 2020. URL https://api.semanticscholar. org/CorpusID:227231454.

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In *Proceedings of the AAAI conference on artificial intelligence*, volume 34, pp.

7432–7439, 2020.

Hoyeon Chang, Jinho Park, Seonghyeon Ye, Sohee Yang, Youngkyung Seo, Du-Seong Chang, and Minjoon Seo. How do large language models acquire factual knowledge during pretraining? arXiv preprint arXiv:2406.11813, 2024.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. Knowledge neurons in pretrained transformers. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), *Proceedings of the 60th* Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 8493–8502, Dublin, Ireland, May 2022a. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long. 581. URL https://aclanthology.org/2022.acl-long.581.

Damai Dai, Wen-Jie Jiang, Qingxiu Dong, Yajuan Lyu, Qiaoqiao She, and Zhifang Sui. Neural knowledge bank for pretrained transformers. In *Natural Language Processing and Chinese Computing*, 2022b. URL https://api.semanticscholar.org/CorpusID:251223709.

Shibhansh Dohare, J. Fernando Hernandez-Garcia, Qingfeng Lan, Parash Rahman, Ashique Rupam Mahmood, and Richard S. Sutton. Loss of plasticity in deep continual learning. *Nature*, 632:768 - 774, 2024. URL https://api.semanticscholar.org/CorpusID:259251905.

Qingxiu Dong, Damai Dai, Yifan Song, Jingjing Xu, Zhifang Sui, and Lei Li. Calibrating factual knowledge in pretrained language models. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), Findings of the Association for Computational Linguistics: EMNLP 2022, pp. 5937–5947, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.

findings-emnlp.438. URL https://aclanthology.org/2022.findings-emnlp.438.

Jiahui Geng, Fengyu Cai, Yuxia Wang, Heinz Koeppl, Preslav Nakov, and Iryna Gurevych. A survey of confidence estimation and calibration in large language models. In Kevin Duh, Helena Gomez, and Steven Bethard (eds.), Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 6577–6595, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024. naacl-long.366. URL https://aclanthology.org/2024.naacl-long.366.

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are keyvalue memories. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (eds.),
Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 5484–5495, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics.

doi: 10.18653/v1/2021.emnlp-main.446. URL https://aclanthology.org/2021.emnlp-main.446.

Mor Geva, Avi Caciularu, Kevin Wang, and Yoav Goldberg. Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang
(eds.), *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing*, pp.

30–45, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.3. URL https://aclanthology.org/2022.emnlp-main.3.

Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. Dissecting recall of factual associations in auto-regressive language models. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), *Proceedings of* the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 12216–12235, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.751. URL https://aclanthology.org/2023.emnlp-main.751.

Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, et al. Olmo: Accelerating the science of language models. *ACL 2024*, 2024.

Joel Jang, Seonghyeon Ye, Sohee Yang, Joongbo Shin, Janghoon Han, Gyeonghun KIM, Stanley Jungkyu Choi, and Minjoon Seo. Towards continual knowledge learning of language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=vfsRB5MImo9.

Yujin Kim, Jaehong Yoon, Seonghyeon Ye, Sung Ju Hwang, and Se-young Yun. Carpe diem: on the evaluation of world knowledge in lifelong language models. *NACCL 2024*, 2023.

Aviral Kumar and Sunita Sarawagi. Calibration of encoder decoder models for neural machine translation.

arXiv preprint arXiv:1903.00802, 2019.

Didier Lairez. What entropy really is: the contribution of information theory. *arXiv preprint arXiv:2204.05747*,
2022.

Zeyu Liu, Yizhong Wang, Jungo Kasai, Hannaneh Hajishirzi, and Noah A. Smith. Probing across time: What does RoBERTa know and when? In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (eds.), *Findings of the Association for Computational Linguistics: EMNLP 2021*, pp. 820–842, Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi:
10.18653/v1/2021.findings-emnlp.71. URL https://aclanthology.org/2021.findings-emnlp.71.

Christian Majenz. Entropy in quantum information theory–communication and cryptography. arXiv preprint arXiv:1810.10436, 2018.

Andrey Malinin and Mark Gales. Uncertainty estimation in autoregressive structured prediction. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id= jN5y-zb5Q7m.

Kevin Meng, David Bau, Alex J Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), *Advances in* Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=-h6WAS6eE4.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun'ichi Tsujii (eds.), Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 2381–2391, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1260. URL https://aclanthology.org/D18-1260.

Fabio Petroni, Tim Rocktäschel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, Alexander H. Miller, and Sebastian Riedel. Language models as knowledge bases? In *Conference on Empirical Methods in Natural* Language Processing, 2019. URL https://api.semanticscholar.org/CorpusID:202539551.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. *Communications of the ACM*, 64(9):99–106, 2021.

Noam Shazeer. Glu variants improve transformer. *arXiv preprint arXiv:2002.05202*, 2020.

Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, et al. Dolma: An open corpus of three trillion tokens for language model pretraining research. *ACL 2024*, 2024.

Sainbayar Sukhbaatar, Jason Weston, Rob Fergus, et al. End-to-end memory networks. Advances in neural information processing systems, 28, 2015.

Kaiser Sun and Mark Dredze. Amuro & char: Analyzing the relationship between pre-training and fine-tuning of large language models. *arXiv preprint arXiv:2408.06663*, 2024.

Ryan Teehan, Miruna Clinciu, Oleg Serikov, Eliza Szczechla, Natasha Seelam, Shachar Mirkin, and Aaron Gokaslan. Emergent structures and training dynamics in large language models. ACL 2022. URL https://aclanthology.org/2022.bigscience-1.11.

Artem Vazhentsev, Akim Tsvigun, Roman Vashurin, Sergey Petrakov, Daniil Vasilev, Maxim Panov, Alexander Panchenko, and Artem Shelmanov. Efficient out-of-domain detection for sequence to sequence models. Findings of ACL 2023. URL https://aclanthology.org/2023.findings-acl.93.

Cunxiang Wang, Pai Liu, and Yue Zhang. Can generative pre-trained language models serve as knowledge bases for closed-book QA? In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli (eds.), *Proceedings* of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 3241–3251, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.251. URL
https://aclanthology.org/2021.acl-long.251.

Johannes Welbl, Nelson F. Liu, and Matt Gardner. Crowdsourcing multiple choice science questions. In Leon Derczynski, Wei Xu, Alan Ritter, and Tim Baldwin (eds.), Proceedings of the 3rd Workshop on Noisy User-generated Text, pp. 94–106, Copenhagen, Denmark, September 2017. Association for Computational Linguistics. doi: 10.18653/v1/W17-4413. URL https://aclanthology.org/W17-4413.

Yuhao Wu, Tongjun Shi, Karthick Sharma, Chun Wei Seah, and Shuhao Zhang. Online continual knowledge learning for language models. *arXiv preprint arXiv:2311.09632*, 2023.

Wenzhe Yang. Entropy, thermodynamics and the geometrization of the language model. *arXiv preprint* arXiv:2407.21092, 2024.

Yunzhi Yao, Ningyu Zhang, Zekun Xi, Meng Wang, Ziwen Xu, Shumin Deng, and Huajun Chen. Knowledge circuits in pretrained transformers. *ArXiv*, abs/2405.17969, 2024. URL https://api.semanticscholar. org/CorpusID:270068372.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Anna Korhonen, David Traum, and Lluís Màrquez (eds.), *Proceedings* of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 4791–4800, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1472. URL https:
//aclanthology.org/P19-1472.

## A Knowledge Entropy A.1 Intuition Behind The Definition Of Knowledge Entropy

The mechanism behind the relationship between decreasing knowledge entropy and the ability to acquire
and retain knowledge during pretraining is that the coefficients in the linear combination of memory vectors determine how the corresponding memory vectors are updated. As defined in Equation 1, the output
of the feed-forward layers (FFNs) is a linear combination of memory vectors vi=1,··· ,m ∈ R
d, the row
vectors of V ∈ R
m×d, where the coefficients ci are given by f(x · KT). In other words, *F F N*(x) =
c1v1 + c2v2 + *· · ·* + cmvm. Within a given layer, since the operations beyond the FFNs and the input to the
FFNs remain consistent across all memory vectors, the coefficients act as scaling factors for the gradient by
the chain rule. During training, the gradient ∂L
∂vi,j
for i = 1, 2*, . . . , m* and j = 1, 2*, . . . , d* can be decomposed
as:∂L
$${\frac{\partial L}{\partial\mathbf{v_{i,j}}}}={\overset{\rightharpoonup}{\frac{\partial L}{\partial F F N(x)}}}\cdot{\frac{\partial F F N(x)}{\partial v_{i,j}}}$$

## Here, ∂L
∂F F N(X)
Is The Same For All Vi, Meaning That The Relative Magnitude Of The Gradient Depends On
∂F F N(X)
∂Vi,J= Ci. Thus, Larger Coefficients Result In Proportionally Larger Gradients Being Applied To The
Corresponding Memory Vectors, Amplifying Their Updates During Backpropagation.
As Pretraining Progresses, A Spikier Coefficient Distribution—Captured By Decreasing Knowledge Entropy—Implies That Gradient Updates Become Increasingly Concentrated On Specific Positions Where The Average Coefficients Are Larger. This Centralization Can Affect The Model'S Ability To Evenly Utilize Its Memory Capacity, Thereby Impacting Knowledge Acquisition And Retention. A.2 Knowledge Entropy

Does the choice of model change the trend? To assess the generalizability of the trend observed in Figure 2, we conducted experiments on the knowledge entropy trend using Pythia 1.4B model (Biderman et al., 2023). As shown in Figure 6, knowledge entropy measured with the Pythia model also tends to decrease as pretraining progresses. Does the choice of dataset change the trend? As expressed in Equation 2, knowledge entropy is dependent on the dataset D. We define D as the dataset used during pretraining, as knowledge entropy reflects how the model integrates the knowledge stored in its memory vectors, learned during pretraining. However, to further explore whether the choice of dataset influences the trend of knowledge entropy, we measure it using PubMed and C4. Figure 7 shows that the trend remains consistent regardless of the dataset used when calculating knowledge entropy.

20 40 60 80 100 120 140 Pre-trained steps (k)
143.4 143.5 143.6 143.7 Kn o w le d g e E
n tro p y 100 200 300 400 500 600 700 Pre-trained steps (k)
143.3 143.4 143.5 143.6 143.7 143.8 143.9 K
n o wl e d g e E
n tro pyC4 Pubmed Dolma 0 2 5 7 10 12 15 Layer 8.88 8.90 8.92 8.94 8.96 8.98 9.00 K
n o w le d g e E
n tr o p y 18k 369 738k
(a)
0 5 10 15 20 25 30 Layer 9.12 9.15 9.18 9.21 9.24 9.27 9.30 K
n o w le d g e E
n tr o p y 5k 278k 557k
(b)
Does the choice of activation function change the trend? We also explored an alternative where we do not take the absolute value of the SwiGLU output. Instead, following the ReLU function (Agarap, 2018),
another widely used activation function, we replaced all negative values with 0. Figure 9 shows that the trend remains consistent even under this modification.

$$C_{n,j}^{(l)}=\mathrm{ReLU}(\mathrm{gate}(\mathbf{x_{j}}))\otimes\mathrm{up}(\mathbf{x_{j}}),$$

Layer-wise Knowledge Entropy Figure 8 shows how knowledge entropy changes by layer during pretraining. Knowledge entropy consistently decreases in every layer, with the most significant reduction occurring in the last layer, which closely resembles the output distribution right before the token prediction. OLMo-7B model also shows a similar trend to 1B model.

## 15

SwiGLU ReLU
0 100 200 300 400 500 600 700 Pre-trained Step (k)
143.2 143.4 143.6 143.8 144.0 Ent ropy 0 10 20 30 40 50 60 70 80 90 Average Perplexity of Entities in the Instance 0 500 1000 1500 2000 2500 3000 pubmed c4 Freq ue nc y

## A.3 Entropy Of Attention Layers

Inspired by previous research that emphasizes the attention layer's role of attribute extraction (Geva et al.,
2023), we also measure *attention entropy* H(θatt) similarly to Kumar & Sarawagi (2019)12. Attention weights, which are the output of softmax normalization after the key-query-value operation, can be interpreted as weight assigned to the previous tokens. As the attention weight for each token position in each attention head forms a probability distribution(summing to 1), calculating entropy follows the normal entropy formula. Then, layer-wise entropy H(θ latt) is averaged over token position and attention heads and attention entropy H(θatt)
is the sum of layer-wise entropy H(θ latt). Following the notations from Geva et al. (2023), attention entropy is calculated as:

Hn,j (θ h,l att ) = −X j i=1 A h,l,n i,j · log(A h,l,n i,j ) for i = 1, 2, . . . , Tn and j = 1, 2, . . . , Tn n=1  j=1 Hn,j (θ h,l att )  H(θ h,l att ) =  1 |D| X |D|  1 Tn X Tn  ; H(θ l att) =  1 N X N h=1 H(θ h,l att ); H(θatt) = X L l=1 H(θ l att)
(3)  $\frac{1}{2}$ . 
where A*h,l,n* ∈ R
Tn×Tn represents the attention weights of the h-th attention head in layer l for the n-th instance, Tn is the sequence length of the n-th instance in the training dataset D, N denotes the number of attention heads, and L denotes the number of layers in the model.

## A.4 Entropy Of Next Token Prediction

The entropy of next token prediction (Vazhentsev et al.; Geng et al., 2024; Malinin & Gales, 2021) is defined as H(θ n,j ntp ) = −P|V| i=1 pi· log(pi), where pi represents the probability of the i-th token. This value is then averaged over the sequence length (Tn) and the dataset size (|D|).

## B Knowledge Acquisition And Forgetting B.1 Datasets

Training Dataset for Continual Knowledge Learning In this section, we share a brief description of the datasets we used. For continual knowledge learning, we experiment with PubMed and C4. The PubMed dataset consists of biomedical literature abstracts from the PubMed database, containing articles across a wide range of topics in medicine and biology. The C4 (Colossal Clean Crawled Corpus) dataset (Raffel et al., 2020) is a large-scale, preprocessed collection of text scraped from the web, designed to be a clean and diverse representation of natural language.

To compare the distribution of C4, PubMed, and Dolma, we evaluate the average perplexity of entities for C4 and PubMed, as shown in Figure 10. On the x-axis, we plot the range of an average perplexity of instances, while the y-axis represents the number of instances. We randomly sample 10,000 instances from each corpus, extract entities using GPT-4o, and calculate perplexity with the last checkpoint of OLMo. The perplexity values from the last checkpoint of OLMo indicate how likely these entities are to appear in the pretraining corpus, Dolma. The results reveal that PubMed exhibits a broad distribution of perplexity, with a higher number of instances having high perplexity values. In contrast, C4 shows a tendency towards lower perplexity, suggesting that the distribution of entities in PubMed differs from that in Dolma, while the distribution in C4 tends to be more similar to Dolma. Evaluation Dataset to measure *Knowledge Acquisition* To measure knowledge acquisition of language model, we use the fictional knowledge dataset (Chang et al., 2024), which is designed to assess how well LLMs acquire factual knowledge during pretraining. This dataset includes 130 paragraphs 13 presented in a Wikipedia-style format with fictional yet realistic entities (injected knowledge) and 1,950 probes which are cloze-task-style sentences to query the information within the corpus. The final span of each probe, referred to as the target span, is used to evaluate the model's prediction probability, which serves as a measure of knowledge acquisition performance. In Chang et al. (2024), the probes are divided into three levels of difficulty, with five sentences created for each level. This results in 15 probes per corpus. The difficulty levels are as follows: 1) **Memorization** probes directly ask about sentences explicitly present in the fictional corpus. 2) **Semantic generalization** probes are paraphrased versions of the memorization probes to test the model's understanding of meaning beyond surface forms. 3) **Compositional generalization probes** are designed to assess whether the model can integrate multiple pieces of knowledge from the fictional corpus. The injected knowledge is incorporated into the training corpus during continual learning, with updates occurring every 160 steps.

Following Chang et al. (2024), we divide the 130 corpora into two settings: *paraphrase* and *once*. In the paraphrase setting, 70 instances are each paraphrased 10 times. For every 160 steps, one paraphrased version of an instance is added to the training corpus, repeating this process 10 times14. In the *once* setting, each instance is presented only once throughout the entire continual learning process. The 60 instances are divided into 10 groups, with 6 instances added every 160 steps. Evaluation Dataset to measure *Knowledge Forgetting* To measure the forgetting rate, we evaluate on 6 downstream datasets.

- SCIQ: multiple-choice question-answering dataset consisting of over 13,000 science exam-style questions, covering subjects such as physics, chemistry, biology, and earth science
- WINOGRANDE: large-scale benchmark designed to test commonsense reasoning in natural language understanding
- PIQA: commonsense reasoning about everyday physical interactions such as how to perform tasks involving physical actions
- OBQA: multiple-choice question-answering benchmark designed to assess a model's ability to answer elementary-level science questions.

- HELLASWAG: a large-scale benchmark for commonsense reasoning, focusing on selecting the most plausible continuation of a given narrative or scene.

- ARC EASY: multiple-choice science questions typically answered by students in elementary and middle school.

## B.2 Metric

In this section, we share a detailed description of how we evaluate *knowledge acquisition* and knowledge forgetting.

Knowledge Acquisition Given a language model θ, θPT represents the model extracted from a pretraining step and serves as the initial point for continual learning, and θCL represents the model after it. The acquisition metric A(θ) for the model θ is defined as Equation 4. When given a corpus set of *once* setting Wonce, each instance in a corpus wi has a corresponding set of probes Pwi, which contains 15 different probes. To calculate the performance of the *once* setting, Konce(θ), we compute the average log probability ℓ(pi;θ) of the target span for each probe pi ∈ Pwiacross all instances in wi ∈ Wonce and sum these averages. The same calculation is performed for the *paraphrase* setting. The total performance K(θ) is calculated by the weighted average of Konce(θ) and Kpara(θ). Finally, the acquisition metric A(θ) is defined as the improvement rate in performance K(θ) from the initial model state θPT to final model state θCL.

Konce(θ) =  1 |Wonce| X wi∈Wonce 1 |Pwi| X pi∈Pwi ℓ(pi; θ); Kpara(θ) =  1 |Wpara| X wi∈Wpara 1 |Pwi| X pi∈Pwi ℓ(pi; θ) K(θ) =  |Wonce| × Konce(θ) + |Wpara| × Kpara(θ) |Wonce| + |Wpara|; A(θ) =  K(θCL) − K(θPT) K(θPT) (4)

$$({\mathfrak{H}})$$
Knowledge Forgetting The forgetting metric F(θ) is calculated as the average performance degradation from the initial model θPT to the final model θCL across six downstream tasks T : SciQ (Welbl et al., 2017), Winograde (Sakaguchi et al., 2021), PIQA (Bisk et al., 2020), OBQA (Mihaylov et al., 2018),
HellaSwag (Zellers et al., 2019), and ARC Easy (Clark et al., 2018).

$${\mathcal{P}}(\theta)={\frac{1}{|{\mathcal{T}}|}}\sum_{i\in|{\mathcal{T}}|}{\mathcal{T}}_{i}(\theta);\qquad{\mathcal{F}}(\theta)=-{\frac{{\mathcal{P}}(\theta_{\mathrm{CL}})-{\mathcal{P}}(\theta_{\mathrm{PT}})}{{\mathcal{P}}(\theta_{\mathrm{PT}})}}$$
P(θPT)(5)
0.400 0.375 0.350 0.325 0.300 0.275 0.250 0.225 0.200 100 200 300 400 500 600 700 Pre-trained Step (k)
0 10 20 30 40 K

n o wl e d g e pr o b e 
( )

A

c q ui si ti o n 
( )

Paraphrase ( ) Paraphrase ( )
Once ( ) Once ( )
6 8 10 12 14 16 18 20 100 200 300 400 500 600 700 Pre-trained Step (k)
16 18 20 22 24 A

c q ui si ti o n 
( )

Pubmed ( )
Pubmed ( )
C4 ( ) C4 ( )
F

or g ett in g 
( )

(a)

D

o w n st re a m p erf 
( ) 
(
%
)

0.36 0.34 0.32 0.30 0.28 52 54 56 58 60 62 64 K

n o wl e d g e pr o b e 
( )

CL ( CL) CL ( CL) PT ( PT) PT ( PT)
100 200 300 400 500 600 700 Pre-trained Step (k)
100 200 300 400 500 600 700 Pre-trained Step (k)
0.31 0.30 0.29 0.28 D

o w n str e a m p erf 
( ) 
(%
)

52 53 54 55 56 57 K

n o w le d g e pro b e 
( )

Pubmed ( ) Pubmed ( ) C4 ( ) C4 ( )
(b)

## B.3 Frequency Of Knowledge Injections

We divide the experiment into two settings: the *once* setting, where knowledge is injected a single time, and the *paraphrase* setting, where knowledge is injected ten times using ten paraphrased paragraphs. Figure 11 presents knowledge acquisition results based on the frequency of injections. Knowledge acquisition and final performance generally follow similar trends in both settings, with models in the later stages of pretraining showing the lowest performance. However, the performance and acquisition rate of *once* setting lag behind those of *paraphrase* setting. Also, notably, for models in the final stage of pretraining, the acquisition rate in the *once* setting was negative. This indicates that the log probability of the injected knowledge decreased, preventing successful incorporation of the new knowledge. In other words, even the knowledge injected during continual learning is subject to forgetting throughout the continual learning process.

## 19 B.4 Knowledge Acquisition & Forgetting B.4.1 Baseline Setup

Our base experiments are conducted using a hyperparameter configuration most closely aligned with continual knowledge learning studies, specifically with a batch size of 128, a learning rate of 4e-4, while training single-epoch of PubMed corpus. We use adamW optimizer (β = 0.9, 0.95, weight decay= 0.1), cosine LR
scheduler with warmup=0.05, and set maximum sequence length as 1024. We randomly selected 204,800 instances from the PubMed and C4 datasets and matched the sequence length to 1,024 tokens by concatenating instances. This resulted in a training dataset consisting of approximately 210 million tokens.

As analyzed in Section 4.2, final performance generally deteriorates as later-stage models are utilized as the initial model. Figure 12 illustrates the model's initial performance before continual learning, as well as its performance afterward. Models in the later stages of pretraining exhibit superior language modeling abilities before continual learning, as evidenced by the lower log probability for the newly injected knowledge (dotted line). However, after continual learning, their performance deteriorates compared to models from the earlier stages of pretraining (solid line). Similarly, the downstream task performance of the later-stage models was better initially, but after continual learning, their performance declined more than that of the earlier-stage models.

## B.4.2 Various Settings

We observed consistently, even with altered hyperparameter settings, that models in the later stages of pretraining struggle to learn new knowledge and retain existing knowledge.

Training Dataset for Continual Knowledge Learning To investigate how the type of new knowledge affects performance, we further conducted experiments using the C4 corpus, which has a distribution more similar to the pretraining corpus, Dolma, compared to PubMed. Figure 13a indicates that the gap in acquisition rate between later-stage models and initial-stage models is larger when the new knowledge distribution differs significantly from the pretraining corpus: models trained with PubMed (-6.4%p) exhibit a more pronounced gap compared to those trained with C4 (-3.4%p).

All models, regardless of their pretraining stage, tend to perform better when continually pretrained with PubMed compared to C4. We hypothesize that this is because PubMed's different distribution from the pretraining corpus encourages the model to learn more new knowledge, enhancing its ability to acquire new information. However, the rate of improvement varies by model state. Later-stage models tend to show similar performance regardless of the type of new knowledge, suggesting a limit in their learning capacity. In contrast, initial-stage models exhibit a stronger ability to acquire knowledge when trained on a corpus with a different distribution, such as PubMed, demonstrating their greater adaptability in learning new and diverse information. Model Size To test the universal deterioration of knowledge acquisition and retention capabilities, we also experimented with the OLMo 7B model Groeneveld et al. (2024). In Figure 4a, the 7B model exhibits a clear trend of diminishing knowledge acquisition A(θ) capabilities as pretraining progresses. This decline is accompanied by an increase in forgetting F(θ), indicating that the model struggles to retain previously learned information as new data is introduced. Batch Size In Table 1 line (b), when the batch size is large, the influence of individual data points on the model update decreases, resulting in a lower acquisition rate, while forgetting is less pronounced. In the later stages of training, however, if sufficient learning rate warmup is not provided, the model appears to collapse.