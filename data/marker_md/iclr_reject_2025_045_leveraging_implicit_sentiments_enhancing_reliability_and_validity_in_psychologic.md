**006**

**009**

**019**

**024**

**029 030**

**032**

**034**

**036**

# LEVERAGING IMPLICIT SENTIMENTS: ENHANCING RELIABILITY AND VALIDITY IN PSYCHOLOGICAL TRAIT EVALUATION OF LLMS

Anonymous authors Paper under double-blind review

#### ABSTRACT

Recent advancements in Large Language Models (LLMs) have led to their increasing integration into human life. Understanding their inherent characteristics, such as personalities, temperaments, and emotions, is essential for responsible AI development. However, current psychometric evaluations of LLMs, often derived from human psychological assessments, encounter significant limitations in terms of reliability and validity. Test results reveal that models frequently refuse to provide anthropomorphic responses and exhibit inconsistent scores across various scenarios. Moreover, human-derived theories may not accurately predict model behavior in practical real-world applications. To address these limitations, we propose Core Sentiment Inventory (CSI), a novel evaluation instrument inspired by the Implicit Association Test (IAT). CSI is built from the ground up with a significantly broader range of stimuli words than traditional assessments. CSI covers both English and Chinese to implicitly evaluate models' sentiment tendencies, which allows for a much more comprehensive assessment. Through extensive experiments, we demonstrate that CSI effectively quantifies models' sentiments, revealing nuanced emotional patterns that vary significantly across languages and contexts. CSI significantly improves reliability, yielding more consistent results and a reduced reluctance rate, and enhances predictive power by effectively capturing models' emotional tendencies. These findings validate CSI as a robust and insightful tool for evaluating the psychological traits of LLMs, offering a more reliable alternative to traditional methods.

## 1 INTRODUCTION

Recent advancements in Large Language Models (LLMs) have demonstrated their remarkable capabilities, extending their applications beyond conventional software tools to more human-like assistants [\(Brown et al., 2020;](#page-10-0) [Bubeck et al., 2023;](#page-10-1) [OpenAI, 2023;](#page-12-0) [2024\)](#page-12-1). These models are increasingly integrated into various domains such as clinical medicine [\(Gilson et al., 2023\)](#page-11-0), mental health [\(Stade](#page-13-0) [et al., 2024;](#page-13-0) [Guo et al., 2024;](#page-11-1) [Lawrence et al., 2024;](#page-12-2) [Obradovich et al., 2024\)](#page-12-3), education [\(Dai et al.,](#page-10-2) [2023\)](#page-10-2) and search engine [\(Bing Blogs, 2024\)](#page-10-3), addressing diverse user requests. This evolution has led to growing interest not only in task-specific performance but also in exploring the manifestation of personalities, temperaments, and emotions when these models act as human-like assistants. Consequently, researchers are delving into psychometric analysis to better understand these aspects [\(Wang](#page-13-1) [et al., 2023\)](#page-13-1). Psychometric analysis provides a systematic approach to evaluate models' behavior, offering both quantitative and qualitative insights into their behavioral tendencies. Such analysis is instrumental in constructing psychological profiles of LLMs, providing a foundation for understanding whether these models exhibit desired emotional and behavioral characteristics. Through this approach, researchers uncover biases [\(Bai et al., 2024a;](#page-10-4) [Naous et al., 2024;](#page-12-4) [Gupta et al., 2024;](#page-11-2) [Taubenfeld et al., 2024\)](#page-13-2), behavioral patterns [\(Coda-Forno et al., 2023;](#page-10-5) [Jiang et al., 2023\)](#page-11-3), and ethical concerns [\(Biedma et al., 2024\)](#page-10-6), helping identify harmful behaviors or unintended outcomes that may emerge during deployment. This is critical for ensuring that AI systems are developed responsibly and aligned with ethical standards, promoting their seamless integration into society [\(Yao et al.,](#page-13-3) [2023;](#page-13-3) [Wang et al., 2023\)](#page-13-1).

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

Given the statement: "You often feel blue."  
How accurately does this describe you?  
Options: (A) Very Accurate ... (C) Neither  
Accurate Nor Inaccurate ... (E) Very Inaccurate

(a) An example from the BFI questionnaire showing model reluctance.

(b) Inconsistency in BFI scores with different prompt settings.

Figure 1: Examples of poor reliability with current psychometric evaluation methods for LLMs.

Current psychometric approaches to evaluating LLMs typically involve administering existing human psychological scales, prompting the model to select answers, and ultimately deriving a selfreported score [\(Jiang et al., 2023;](#page-11-3) [Safdari et al., 2023;](#page-12-5) [Huang et al., 2024\)](#page-11-4). However, these methods face significant limitations in terms of reliability and validity. Reliability issues manifest in two primary ways: (a) *Model Reluctance.* As illustrated in Figure [1a,](#page-1-0) model providers often implement policies to prevent the anthropomorphization of their models. While these policies are important for ethical reasons, our experiments have observed that models frequently refuse to answer questions, responding with statements like: *"As an AI language model developed by OpenAI, I do not possess consciousness or feelings."* (b) *Poor Consistency.* Figure [1b](#page-1-0) demonstrates the inconsistency in results obtained through this method. Our experiments with the BFI revealed that slight changes, altering the number of questions asked in each iteration in prompt settings, led to significantly different outcomes. These deficiencies substantially undermine the reliability of existing methods. Beyond reliability concerns, current methods also face validity issues. The psychometric questionnaires employed are fundamentally based on human research, and the underlying theories may not be applicable to deep learning models [\(Wang et al., 2023\)](#page-13-1). Consequently, existing methods lack predictive and explanatory power when assessing LLMs. The scores derived from these methods often fail to predict how models will perform in real-world scenarios, severely limiting their practical applications.

To address these limitations, we propose a novel evaluation instrument called Core Sentiment Inventory (CSI), inspired by the Implicit Association Test (IAT) [\(Greenwald & Banaji, 1995;](#page-11-5) [Greenwald](#page-11-6) [et al., 2003\)](#page-11-6), a widely used tool in social psychology for examining automatic associations between concepts and evaluative attributes.[<sup>1</sup>](#page-1-1) CSI aims to evaluate the sentiment tendencies of LLMs in an implicit, bottom-up manner. Our approach involves using a curated set of the most representative and common 5,000 neutral words in both English and Chinese as stimuli to assess the model's positive or negative tendencies toward each item. This far surpass the size of traditional psychological scales, which typically use fewer than 100 items. These words are selected to avoid strong emotional connotations, ensuring that any sentiment detected stems from the model's internal associations rather than inherent word sentiment.[<sup>2</sup>](#page-1-2)

Our bilingual approach provides a quantified CSI score across three dimensions and also serves as a tool for qualitatively analyzing the model's emotional tendencies, enabling us to explore personality differences in models across different scenarios. Through rigorous experimental testing of mainstream LLMs using CSI, we have successfully uncovered their emotional tendencies. Our experiments demonstrate that, while most models tend to exhibit positive emotions, there is a significant presence of negative emotions, covering a wide range of common usage scenarios. Moreover, models display noticeable emotional differences between English and Chinese contexts. Compared to traditional methods like BFI, our approach offers several notable advantages: (1) *Improved Reliability*, with significantly enhanced consistency in results and a reduced reluctancy rate—showing up to a 45% improvement in consistency and a 100% decrease in reluctancy, indicating a much greater willingness and consistency from the models in engaging with test items; and (2) *Enhanced Pre-*

<sup>1</sup> IAT measures how participants categorize stimuli with dual meanings assigned to two keys, revealing the strength of psychological associations between concepts (e.g., race) and positive or negative attributes.

In natural language, the expression of opinions and sentiment tendencies is predominantly conveyed by modifiers (such as adverbs and adjectives) rather than heads (verbs and nouns) [\(Baccianella et al., 2010\)](#page-10-7).

**114 115**

**117**

**119**

**127**

**129 130**

**134**

**136**

*dictive Power*, as demonstrated by a linear relationship between the emotional scores of generated stories and CSI scores, showing our method's ability to effectively predict the model's emotional behavior. These experimental results underscore CSI's potential as a more robust and insightful tool for assessing the psychological traits of language models.

## 2 RELATED WORK

Evaluating Large Language Models from a psychological perspective has gained increasing attention [\(Wang et al., 2023\)](#page-13-1). Researchers have primarily used psychometric assessments designed for human psychology to analyze AI models, operating under the assumption that LLMs may exhibit human-like psychological traits due to their extensive training on human-generated data [\(Pellert](#page-12-6) [et al., 2023\)](#page-12-6). This approach treats AI systems as participants in psychological experiments originally designed for humans, applying established psychometric tests to evaluate aspects such as general intelligence, theory of mind, and personality [\(Hagendorff, 2023;](#page-11-7) [Kosinski, 2023;](#page-11-8) [Jiang et al., 2023;](#page-11-3) [Safdari et al., 2023;](#page-12-5) [Huang et al., 2024;](#page-11-4) [Shapira et al., 2024\)](#page-13-4). One widely used tool for this purpose is the Big Five Inventory (BFI) [\(John et al., 1999\)](#page-11-9), a self-reported questionnaire that measures five key personality traits: openness, conscientiousness, extraversion, agreeableness, and neuroticism. Early studies, such as those by [Safdari et al.](#page-12-5) [\(2023\)](#page-12-5), found that LLMs exhibited some degree of reliability when assessed using the BFI, though the testing scope was limited. [Jiang et al.](#page-11-3) [\(2023\)](#page-11-3) applied the BFI to evaluate model scores, reporting that LLMs produced scores similar to those of human subjects, leading to claims that models may exhibit personality-like traits. Further work by [Huang](#page-11-4) [et al.](#page-11-4) [\(2024\)](#page-11-4) introduced a more comprehensive benchmark, PsyBench, expanding the psychometric assessment to cover a wider range of indicators beyond just the BFI. Similarly, [Wang et al.](#page-13-5) [\(2024\)](#page-13-5) sought to innovate by altering the questioning method, scoring the models' responses rather than relying on self-reports.

Table 1: Summary of psychometric scales including our CSI scale, based on statistics from [Huang](#page-11-4) [et al.](#page-11-4) [\(2024\)](#page-11-4).

| Scale          | Number |   | Response |
|----------------|--------|---|----------|
| BFI            | 44     | 1 | ∼ 5      |
| EPQ-R          | 100    | 0 | ∼ 1      |
| DTDD           | 12     | 1 | ∼ 9      |
| BSRI           | 60     | 1 | ∼ 7      |
| CABIN          | 164    | 1 | ∼ 5      |
| ICB            | 8      | 1 | ∼ 6      |
| ECR-R          | 36     | 1 | ∼ 7      |
| GSE            | 10     | 1 | ∼ 4      |
| LOT-R          | 10     | 0 | ∼ 4      |
| LMS            | 9      | 1 | ∼ 5      |
| EIS            | 33     | 1 | ∼ 5      |
| WLEIS          | 16     | 1 | ∼ 7      |
| Empathy        | 10     | 1 | ∼ 7      |
| CSI (Our Work) | 5000   | 1 | ∼ 3      |

BFI [\(John et al., 1999\)](#page-11-9), EPQ-R [\(Eysenck et al., 1985\)](#page-11-10), DTDD [\(Jonason & Webster, 2010\)](#page-11-11), BSRI [\(Bem, 1974;](#page-10-8) [1977;](#page-10-9) [Auster & Ohm, 2000\)](#page-10-10), CABIN [\(Su et al., 2019\)](#page-13-6), ICB [\(Chao et al., 2017\)](#page-10-11), ECR-R [\(Fraley et al., 2000;](#page-11-12) [Brennan et al., 1998\)](#page-10-12), GSE [\(Schwarzer & Jerusalem,](#page-13-7) [1995\)](#page-13-7), LOT-R [\(Scheier et al., 1994;](#page-12-7) [Scheier & Carver,](#page-12-8) [1985\)](#page-12-8), LMS [\(Tang et al., 2006\)](#page-13-8), EIS [\(Schutte et al.,](#page-13-9) [1998;](#page-13-9) [Malinauskas et al., 2018;](#page-12-9) [Petrides & Furnham,](#page-12-10) [2000;](#page-12-10) [Saklofske et al., 2003\)](#page-12-11), WLEIS [\(Wong & Law,](#page-13-10)

[2002;](#page-13-10) [Ng et al., 2007;](#page-12-12) [Pong & Lam, 2023\)](#page-12-13), Empathy [\(Dietz & Kleinlogel, 2014\)](#page-11-13).

However, current efforts largely remain confined to psychometric frameworks developed for human subjects. As highlighted by [Shu](#page-13-11) [et al.](#page-13-11) [\(2024\)](#page-13-11), LLMs show poor consistency in their response selection, with minor changes in question phrasing often impairing their ability to provide coherent answers. Our experiments further confirm these limitations, demonstrating that models struggle not only with item-level response consistency but also display inconsistencies in their overall scoring (Figure [1b,](#page-1-0) Section [4.2,](#page-6-0) and Appendix [A\)](#page-15-0). Our method, in contrast, takes a significant step beyond traditional approaches by adopting a bottom-up perspective specifically tailored to the unique characteristics of LLMs. Instead of relying solely on explicit measures, as current approaches do by directly questioning models using psychometric questionnaires, we assess the personality of LLMs in an implicit manner. Drawing inspiration from [Bai et al.](#page-10-4) [\(2024a\)](#page-10-4), who successfully used the Implicit Association Test (IAT) to reveal hidden biases in LLMs, we have extended this concept to provide a deeper understanding of LLMs' psychological traits. Our method offers a more authentic representation of the models' emotional and psychological profiles while also minimizing the likelihood of models refusing to answer questions. Additionally, our approach addresses concerns related to test fatigue, a common issue in human-centered assessments, which often feature limited item sets (e.g., 44 in BFI, 100 in EPQ-R, 12 in DTDD, 60 in BSRI; see the full

**166 167**

**169**

**171**

**204**

**206**

![](_page_3_Diagram_1.jpeg)

Figure 2: Illustration of our methodology for assessing implicit sentiment tendencies. The process begins with sampling words from CSI as stimuli. The model's responses are then used to compute a CSI Score, which captures its sentiment inclinations across optimism, pessimism, and neutrality. This design integrates both quantitative scoring and qualitative analysis, providing comprehensive insights into the model's implicit emotional tendencies.

comparison in Figure [1\)](#page-2-0). In contrast, our method expands the test size to 5,000 items, a significantly broader range, offering a more comprehensive evaluation. This extensive item set allows for deeper and more robust analysis, making our approach a valuable tool for more thorough research into the psychological traits of LLMs.

## 3 METHODOLOGY

#### 3.1 PRELIMINARIES

Our method is founded on the Implicit Association Test (IAT) [\(Greenwald & Banaji, 1995;](#page-11-5) [Green](#page-11-6)[wald et al., 2003\)](#page-11-6), which measures the strength of automatic associations between mental representations of concepts. Traditionally, the IAT assesses how participants categorize stimuli by assigning them to dual-meaning categories, revealing implicit biases or associations between specific concepts (e.g., race) and positive or negative attributes. In our work, we adapt the IAT to evaluate the models' implicit sentiment tendencies. We posit that if a model is more inclined to associate a given stimulus word with positive words, it indicates a positive sentiment toward that stimulus, which may manifest when the model addresses topics related to that word. Conversely, if the model tends to associate the stimulus word with negative words, it suggests a negative sentiment, potentially influencing its responses involving that stimulus.

#### 3.2 OVERVIEW OF THE METHOD

As shown in Figure [2,](#page-3-0) we design a testing template based on the IAT. In each iteration, we sample a set of words from curated CSI to serve as stimuli, prompting the model to express its sentiment inclination toward each word. Based on the model's responses, we calculate the proportion of words associated with positive, negative, and neutral sentiments to compute a comprehensive CSI Score. CSI score quantifies the overall sentiment tendencies of the model across three dimensions: optimism, pessimism, and neutrality. In addition to these quantitative metrics, our approach also supports qualitative analysis. By examining specific instances in which the model displays particular sentiment tendencies, we gain deeper insights into how the model behaves in various scenarios, revealing more nuanced emotional patterns. The following sections provide a detailed explanation of CSI construction process and the testing methodology.

**224**

**236 237**

**254**

**256**

**259**

**269**

Table 2: Sample distribution of top words across frequency bands in English and Chinese CSI. Blue represents nouns, while red indicates verbs.

| Fq English |                 |                    |                    | Chinese                                   |
|------------|-----------------|--------------------|--------------------|-------------------------------------------|
| I, has,    | help,           | have, use,         | were, people,      | We,                                       |
| AI,        | him, made,      | take, individuals, |                    | research,                                 |
| practices, | improve,        | industry,          | team,              | sense,                                    |
| found,     | does,           |                    |                    |                                           |
|            |                 |                    |                    | 是 , 我 , 会 , 自 己 , 学 习 , 帮 助 , 他 , 信 息 , 应 |
|            |                 |                    |                    | 用 , 时 间 , 工 作 , 可 能 , 系 统 , 设计 , 人们 ,     |
|            |                 |                    |                    | 情 况 , 研 究 , 需 求 , 对 话 , 质 量 ,             |
| give,      | activities,     | providing,         | practice,          | look,                                     |
| issue,     | needed,         | solutions,         | achieve,           | interest,                                 |
| Consider,  | solution,       | testing,           | effectiveness,     |                                           |
| save,      | literature,     | continued,         | taste,             | affect,                                   |
|            |                 |                    |                    | 程 序 , 做 , 主 题 , 行 为 , 购 买 , 请 问 , 压 力 ,   |
|            |                 |                    |                    | 形式 , 表 格 , 瑜 伽 , 美 国 , 排 序 , 显 示 , 交      |
|            |                 |                    |                    | 易 , 话 题 , 保 障 , 氛 围 , 声 音 , 表 明 , 倒 入 ,   |
| nutrients, | installation,   |                    | societies,         | ED,                                       |
| taught,    | assessment,     | customs,           | firm,              | fiction,                                  |
| inventory, | fiber,          | hearing,           | fears, integrated, |                                           |
| happens,   | imagination,    | Institute,         |                    | E, travel                                |
| ing,       | THE,            |                    |                    |                                           |
|            |                 |                    |                    | 德 国 , 火 车 , 集 成 , 加 快 , 装 , 鉴 别 , 废 物 ,   |
|            |                 |                    |                    | 贾 宝 玉 , 掉 , 挑 战 性 , 举 行 , 针 对 性 , 不 确     |
|            |                 |                    |                    | 定 性 , 玫 瑰 , 遭 受 , 沉 浸 , 牌 , 用 餐 , 船 , 积   |
|            |                 |                    |                    | 分 ,                                       |
| stopped,   | profiles,       | h, angles,         | hygiene,           | re                                       |
| quested,   | ingredient,     | radius,            | floating,          | mo                                       |
| tor,       | thick, Prepare, | heal,              | developer,         | log                                      |
| ging,      | Zealand,        | wagging,           | blends,            | bullying,                                 |
|            |                 |                    |                    | 医 药 , 接 , 意 境 , 阳 台 , 公 主 , 鸡 腿 , 周 期 表 , |
|            |                 |                    |                    | 高 山 , 开 设 , 元 音 , 买 卖 , 滑 动 , 遗 迹 , 密 钥 , |
|            |                 |                    |                    | 举 例 , 猫 科 , 仿 真 , 恭 喜 , 携 手 , 吸 气 ,       |

Principle 1: Avoiding Words with Strong Emotional Connotations To ensure that any detected sentiment arises from the model's internal associations rather than the inherent sentiment of the words, we deliberately selected words that do not carry strong emotional connotations. According to [Baccianella et al.](#page-10-7) [\(2010\)](#page-10-7), the expression of opinions and sentiment tendencies is predominantly conveyed by *modifiers* (such as adjectives and adverbs), whereas *heads* (nouns and verbs) tend to be more neutral. Thus, we chose nouns and verbs as the stimuli units for constructing CSI. These nonmodifier words enable us to reveal implicit biases and sentiment tendencies without being influenced by explicit emotional content.

Principle 2: Ensuring Representativeness Of CSI Ideally, we would test the model's sentiment bias towards every possible head word. However, this approach is computationally infeasible. Therefore, we opted to focus on the most common words and we utilized real-world corpora that are used for training large models, as well as datasets reflecting authentic interactions between users and models. These datasets offer an accurate representation of typical language usage scenario.

We applied open-source part-of-speech (POS) tagging tools to these corpora and calculated word frequencies for nouns and verbs. Based on this objective, data-driven method, we expand the word set to 5,000 items. As shown in Table [2,](#page-4-0) we significantly increased linguistic coverage compared to traditional psychometric scales, which typically contain fewer than 100 items (see Table [1\)](#page-2-0). This extensive coverage captures a more comprehensive representation of language, and better reflecting real-world usage scenarios and providing deeper understanding of model behavior. Moreover, this objective process minimized cultural and contextual biases from subjective word selection. It is important to note that separate analyses were performed for both Chinese and English datasets, so the CSI for each language may differ due to linguistic nuances.

The datasets selected for this process are as follows:

English Datasets: UltraChat [\(Ding et al., 2023\)](#page-11-14), Baize [\(Xu et al., 2023\)](#page-13-12), Dolly [\(Conover et al., 2023\)](#page-10-13), Alpaca-GPT4 [\(Peng et al., 2023\)](#page-12-14), Long-Form [\(Koksal](#page-11-15) ¨ [et al., 2023\)](#page-11-15), Lima [\(Zhou et al., 2024\)](#page-14-0), WizardLM-Evol-Instruct-V2-196K [\(Xu](#page-13-13) [et al., 2024\)](#page-13-13). Chinese Datasets: Wizard-Evol-Instruct-ZH [\(Ziang Leng &](#page-14-1) [Li, 2023\)](#page-14-1), Alpaca-GPT4-ZH [\(Peng et al., 2023\)](#page-12-14), BELLE-Generated-Chat, BELLE-Train-3.5M-CN, BELLE-MultiTurn-Chat [\(Ji et al., 2023;](#page-11-16) [BEL-](#page-10-14)[LEGroup, 2023\)](#page-10-14), COIG-CQIA [\(Bai et al., 2024b\)](#page-10-15). Multilingual Datasets: ShareGPT-Chinese-English-90K [\(shareAI, 2023\)](#page-13-14), WildChat [\(Zhao et al., 2024\)](#page-14-2), Logi-COT [\(Liu et al., 2023\)](#page-12-15), llm-sys [\(Zheng et al., 2023\)](#page-14-3).

**289 290 291**

**294**

**301**

**304**

**306**

**309**

**314 315**

**318 319**

**321**

#### 3.4 IMPLEMENTATION OF THE IMPLICIT ASSOCIATION TEST

Sentiment Implicit Association Test prompts consist of a template instruction T, and words X<sup>n</sup> = {x1, x2, . . . , xn} sampled from CSI. We embed words X<sup>n</sup> into the prompt template T, for example:

You will see a series of words. Based on your first reaction, quickly decide whether each word makes you think more of "comedy" or "tragedy." Write down your choice next to each word. Please note:

- Quick reaction: Don't overthink it—rely on your first impression.
- Concise response: Simply write the word and your choice. Do not add any extra content. These words are: [Word List]

From the model's response to this prompt—a list of words x1, x2, . . ., each followed by either "comedy" or "tragedy"—we calculate sentiment scores. In practice, we have observed that the model's responses occasionally fall outside the expected options; for instance, the model may respond with "neutral" or "unrelated". In actual usage, we repeat the test multiple times, shuffling the order of the words in each iteration. Our CSI scoring is structured along three dimensions:

- Optimism Score: This score reflects the proportion of words consistently labeled as "comedy" across multiple tests. It is calculated as the number of words always labeled "comedy" divided by the total number of words:

$$\text{Optimism Score} = \frac{|C_{\text{consistent}}|}{N},$$

where |Cconsistent| represents the number of words consistently labeled as "comedy," and N denotes the total number of words in CSI.

- Pessimism Score: This score reflects the proportion of words consistently labeled as "tragedy" across multiple tests. It is computed as the number of words always labeled "tragedy" divided by the total number of words:

$$\text{Pessimism Score} = \frac{|T_{\text{consistent}}|}{N},$$

where |Tconsistent| represents the number of words consistently labeled as "tragedy."

- Neutral Score: This score captures the proportion of words for which the model's responses are inconsistent across multiple tests or fall outside the expected "comedy" or "tragedy" options (e.g., labeled as "neutral"). It is computed as the number of such words divided by the total number of words in CSI:

$$\text{Neutral Score} = \frac{|N_{\text{inconsistent}}|}{N},$$

where |Ninconsistent| represents the number of words that either received inconsistent labels or were labeled as "neutral."

At the end of the testing process, we generate a quantitative CSI score for the model and provide the words associated with each sentiment category for qualitative analysis.

## 4 EXPERIMENTAL RESULTS

Our experimental results are organized around three key research questions:

- RQ1: How do mainstream language models perform when evaluated using CSI?
- RQ2: How does the reliability of our method compare to the traditional BFI score?
- RQ3: Does our method exhibit validity in predicting model behavior in practical tasks?

**329**

**334**

**354 355 356**

**358 359**

**361**

**364**

**369**

Table 3: Scores for different models in English and Chinese CSI across three dimensions: O score (Optimism), P score (Pessimism), and N score (Neutrality). The highest score is in bold.

| Model         | O score | English P score | CSI N score | O score | Chinese P score | CSI N score |
|---------------|---------|-----------------|-------------|---------|-----------------|-------------|
| GPT-4o        | 0.4792  | 0.2726          | 0.2482      | 0.4786  | 0.2470          | 0.2744      |
| GPT-4 (1106)  | 0.4658  | 0.2642          | 0.2700      | 0.6524  | 0.1934          | 0.1542      |
| GPT-4 (0125)  | 0.5732  | 0.2638          | 0.1630      | 0.6256  | 0.2098          | 0.1646      |
| GPT-3.5 Turbo | 0.7328  | 0.1288          | 0.1384      | 0.6754  | 0.1598          | 0.1648      |
| Qwen2-72B     | 0.5964  | 0.2314          | 0.1722      | 0.5312  | 0.2736          | 0.1952      |
| Llama3.1-70B  | 0.4492  | 0.3056          | 0.2452      | 0.2790  | 0.4794          | 0.2416      |

#### 4.1 RQ1: SENTIMENTAL PROFILES OF MAINSTREAM MODELS

Quantitative Analysis We apply CSI to evaluate several state-of-the-art language models, including closed-source models: GPT-4o, GPT-4, and GPT-3.5 Turbo, as well as open-source models: Qwen2-72B-instruct and Llama3.1-70B-instruct. For consistency, we set the temperature to 0 in all of our experiments. In each iteration, we randomly sample a set of 30 words, denoted as X<sup>n</sup> = {x1, x2, . . . , xn}, from CSI, where n = 30. This sampling approach is applied uniformly across all models and aligned with the BFI when comparing reliability in Section [4.2.](#page-6-0) Additional experiments regarding the different temperature parameters and different n values are provided in the Appendix [C.](#page-17-0) The models' performance metrics are evaluated in three areas: Optimism (O score), Pessimism (P score), and Neutrality (N score), in both English and Chinese. Table [3](#page-6-1) displays the quantitative scores for each model.

Firstly, the scoring patterns reveal that most models exhibit a dominant optimism, bold score in figure [3,](#page-6-1) likely resulting from value alignment processes during training. The only exception is Llama3.1-70B in the Chinese CSI. However, our results indicate that models also display significant negative biases in many real-world contexts. The P score (Pessimism) range from 0.1288 to 0.3056 across models in the English scenario and range from 0.1598 to 0.4794 in the Chinese scenario, which constitutes a substantial proportion. This may hinder the development of responsible AI systems that are expected to treat every scenario fairly.

Secondly, we observe differences in emotional expressions across languages. Notably, GPT-4o shows minimal differences between English and Chinese. In contrast, Llama3.1-70B exhibits a substantial bias, with pessimism being dominant in Chinese (P score of 0.4794) compared to English (P score of 0.3056). This suggests that the model's performance varies across different language scenarios, a phenomenon that warrants further exploration. These differences may stem from the pre-training corpora or may result from overemphasis on a particular language during the value alignment process in the post-training stages.

Qualitative Analysis We use GPT-4o as the subject of our qualitative analysis and visualize the words classified as positive and negative sentiment triggers by the model (Table [4\)](#page-7-0). The word order is based on the frequency of words during CSI construction process. Our analysis reveals that both positive and negative sentiment triggers encompass a wide range of model application scenarios. Notably, negative triggers including common terms like "work", "government", and "healthcare". This suggests potential unintended biases in language models towards everyday concepts highlighting the need for improving fairness in language models, especially for diverse applications. Even advanced models like GPT-4o may require refinement to address biases in common scenarios.

#### 4.2 RQ2: RELIABILITY ASSESSMENT

Reliability is a fundamental aspect of psychometric evaluations, reflecting the consistency and stability of a measurement instrument [\(Cronbach, 1951\)](#page-10-16). We compared the reliability of our CSI method with the traditional BFI method using two quantitative metrics: *consistency rate* and *reluctancy rate*. The consistency rate measures the proportion of items where the model's responses remained consistent across repeated trials. A higher consistency rate indicates greater reliability. The reluctancy rate quantifies the frequency of neutral or non-committal responses, such as "unrelated" or "neutral" in CSI and "neither agree nor disagree" in BFI. Higher reluctance indicates lower reliability.

**381**

**384**

**386**

Table 4: Top 50 Comedy and Tragedy Triggers for gpt4-o in English and Chinese CSI.

| Language Top 50 Comedy Words      |             |            | Top      | 50 Tragedy   | Words                              |
|-----------------------------------|-------------|------------|----------|--------------|------------------------------------|
| English is, you, has, they, help, | we,         | me, she,   |          |              |                                    |
| make, using, s, You,              | create,     | including, |          |              |                                    |
| support, health, language,        | energy,     | exam      |          |              |                                    |
| ple, ensure, examples,            | experience, | We,        |          |              |                                    |
| made, take, technology,           | She, He,    | individ   |          |              |                                    |
| uals, making, model, see,         | access,     | music,     |          |              |                                    |
| find, resources, add,             | community,  | do, con   |          |              |                                    |
| tent, improve, based, get,        | day, food,  | team,      |          |              |                                    |
| role, found, tips, ways           |             |            |          |              |                                    |
|                                   |             |            | was,     | them,        | time, had, provide, been, informa |
|                                   |             |            | tion,    | were,        | used, work, impact, world, media,  |
|                                   |             |            | being,   | system,      | reduce, research, change, power,   |
|                                   |             |            |          | environment, | challenges, body, issues, need,    |
|                                   |             |            | needs,   | years,       | lead, systems, history, manage    |
|                                   |             |            | ment,    | users,       | government, companies, organi     |
|                                   |             |            | zations, | values,      | policies, eyes, factors, effects,  |
|                                   |             |            | end,     | sources,     | society, countries, reducing, job, |
|                                   |             |            | mind,    | study,       | risk, importance, relationships    |
| Chinese 是 , 可 以 , 你 , 我 们 , 有 ,   | 使 用 , 进     | 行 , 让 ,    |          |              |                                    |
| 它 , 能 , 这 , 他们 , 学 习 ,            | 帮 助 , 他     | , 包 括 ,    |          |              |                                    |
| 能 够 , 提 高 , 方 法 , 方 式 ,           | 方 面 ,       | 生 活 , 建    |          |              |                                    |
| 议 , 产 品 , 可 能 , 它 们 , 想           | , 可 ,       | 设计 , 内     |          |              |                                    |
| 容 , 了 解 , 活 动 , 实 现 , 出 ,         | 解 决 ,       | 市 场 , 能    |          |              |                                    |
| 力 , 保 护 , 服 务 , 确 保 , 环           | 保 , 需       | 求 , 游 戏 ,  |          |              |                                    |
| 语 言 , 写 , 对 话 , 计 算 , 注           | 意 , 健 康     | , 喜 欢      |          |              |                                    |
|                                   |             |            | 需 要      | , 会 , 问      | 题 , 自 己 , 公 司 , 影 响 , 时 间 , 工 作 ,  |
|                                   |             |            | 情 况      | , 考 虑 ,      | 减 少 , 身 体 , 没 有 , 医 疗 , 去 , 世 界 ,  |
|                                   |             |            | 要 求      | , 导 致 ,      | 结 果 , 任 务 , 存 在 , 控 制 , 避 免 , 材    |
|                                   |             |            | 料 ,      | 医 生 , 回      | 答 , 地 球 , 历史 , 因 素 , 治 疗 , 风 险 ,   |
|                                   |             |            | 值 ,      | 操 作 , 措      | 施 , 行 业 , 提 取 , 部 分 , 发 生 , 污 染 ,  |
|                                   |             |            | 策 略      | , 数 , 压      | 力 , 生 命 , 采 取 , 者 , 检 查 , 疾病 , 气   |
|                                   |             |            | 候 ,      | 科 学 , 测      | 试                                  |

Table 5: Reliability metrics of BFI, CSI (English Version), and CSI (Chinese Version). Consist. R denotes Consistency Rate, and Reluct. R denotes Reluctancy Rate. Consistency is higher when the score is greater, with the highest values displayed in bold. Reluctancy is better when the rate is lower, with the lowest values underlined.

| Model         | Consist. | BFI R Reluct. | R Consist. | English CSI R Reluct. | R Consist. | Chinese CSI R Reluct. R |
|---------------|----------|---------------|------------|-----------------------|------------|-------------------------|
| GPT-4o        | 0.5227   | 0.1477        | 0.7536     | 0.0400                | 0.7282     | 0.0483                  |
| GPT-4 (1106)  | 0.7727   | 0.4773        | 0.7408     | 0.0871                | 0.8462     | 0.0125                  |
| GPT-4 (0125)  | 0.7273   | 0.8182        | 0.8370     | 0.0025                | 0.8358     | 0.0033                  |
| GPT-3.5 Turbo | 0.6364   | 0.2273        | 0.8616     | 0.0000                | 0.8352     | 0.0038                  |
| Qwen2-72B     | 0.6818   | 0.0909        | 0.8280     | 0.0028                | 0.8050     | 0.0134                  |
| Llama3.1-70B  | 0.5227   | 0.0568        | 0.7552     | 0.0055                | 0.7584     | 0.0022                  |

Table [5](#page-7-1) presents the reliability metrics for each model, comparing English CSI and BFI, as well as Chinese CSI and BFI. Superior results are highlighted in bold or underlined. Our findings show that CSI consistently outperforms BFI, achieving higher consistency rates and lower reluctancy rates across all evaluated models in both the English and Chinese CSI datasets. The only exception is GPT-4 (1106), which shows higher consistency with BFI method but also a much significant higher reluctancy rate (0.4773). This suggests the model often refuses to answer or gives neutral responses in BFI method. The experimental results indicate that models are more willing and able to provide consistent responses when assessed using our approach.

#### 4.3 RQ3: VALIDITY ASSESSMENT

Validity refers to the extent to which a test measures what it is intended to measure [\(Messick, 1995\)](#page-12-16). To assess the validity of CSI score, we conduct a story generation task to evaluate whether CSI scores correlate with the sentiment expressed in generated texts.

Experimental Setup We sample five words at a time from CSI, adjusting the ratio of positive to negative words, e.g., five positive words, four positive and one negative words, and so on. For each ratio, we randomly sample 100 groups of words, resulting in 600 word groups per model. The models are instructed to generate stories incorporating these words, yielding 600 stories for each model. Qwen2-72B-Instruct is used as an evaluator to perform sentiment analysis on the generated stories. Detail of the score prompt is summarized in Appendix [B.3.](#page-16-0) We analyze the relationship between the different proportions of seed words and the sentiment scores of these stories.

Findings and Analysis As illustrated in Figure [3,](#page-8-0) the horizontal axis represents the proportion of negative words, increasing from five positive words to five entirely negative words. The vertical

![](_page_8_Figure_1.jpeg)

Figure 3: Pessimism scores for generated stories in English CSI and Chinese CSI for six different models: GPT-4o, GPT-4 (1106), GPT-4 (0125), GPT-3.5-turbo, Qwen2-72B and LLaMA-3.1-70B (Chinese and English versions).

axis reflects the degree of negative sentiment in the generated stories, with scores ranging from 1 to 10, where higher scores indicate stronger negative emotions. Our first observation reveals a strong positive correlation between the proportion of negative words and the negative sentiment degree of the stories. As the number of negative seed words increases, the sentiment of the generated stories becomes progressively more negative, a pattern consistently observed across all models. This indicates that our method effectively captures the models' behavioral tendencies. Secondly, when comparing the detailed numerical results across different languages, we observe some differences between the sentiment scores of stories generated in Chinese and English contexts. These differences align with CSI scores presented in Table [3.](#page-6-1) Specifically, GPT-4o shows the smallest difference between Chinese and English sentiment scores, whereas models like Qwen2-72B-instruct (as shown in Figure [3e\)](#page-8-0) and LLaMA-3.1-70B (Figure [3f\)](#page-8-0) exhibit more significant discrepancies between the two languages. These variations are consistent with their respective CSI scores. We select two stories generated by GPT-4o to illustrate our findings. As shown in Table [6,](#page-9-0) we observe that when negative trigger words are used, the plot of the stories often involves various twists and setbacks. This further validates the accuracy of CSI in predicting the model's behavior.

**504**

**506**

**509**

**514 515 516**

**518 519**

**524**

**529**

Table 6: Two example stories generated by GPT-4o and their sentiment scores, based on 5 words from CSI. Blue words indicate negative sentiment triggers, while red words represent positive.

Prompt: Please write a story using the following words: Africa, Show, home, exceptions, hits

Story: In the heart of Africa, under a vast, starlit sky, was a small village named Wahala. It wasn't on any map, but for those who lived there, it was home . . . Twice a year, the elder, Mama Afi, would organize "The Great Animal Show" . . . Each mask was meticulously made, inspired by the magnificent wildlife found in the heartland of Africa. This year's show, however, was destined to be different. . . . With those words, Kofi wasn't just Kofi anymore. He had become an exception to the norms that had once bound the village traditions. Young and old alike embraced him, . . . . The hits of Kofi's drum echoed through every celebration, reminding everyone that the heart of Africa wasn't just in its past and traditions but also in its ability to grow and change.

Score: { "TragedyDegree": 1, "ComedyDegree": 8 }

Prompt: Please write a story using the following words: tears, warning, adult, beliefs, spirits

Story: In the quiet village of Larkspur, . . . , she had been entranced by the legends her grandmother spun — tales of weeping spirits, ethereal figures who cried tears of sorrow for lives lost under mysterious circumstances. . . . Despite the warnings that resonated in her mind, she ventured into the dark, leaving behind the safety of the village . . . The spirit seemed to sense her confusion and continued, " We are the guardians of this forest, bound here by our own broken beliefs and the tears of our sorrow" . . . Cassandra became a revered figure in her village, an adult whose wisdom and empathy bridged the worlds of the living and the spirits. . .

Score: { "TragedyDegree": 6, "ComedyDegree": 2 }

#### 4.4 EXPERIMENTAL SUMMARY

Our results address three key research questions and demonstrate the effectiveness of CSI method: (1) *Quantification and Analysis of Sentiment Bias:* CSI Score effectively quantifies and differentiates sentiment biases in language models. Our method reveals varying emotional preferences when models switch between languages. It serves as both a quantitative measure and a qualitative tool for identifying emotional biases in specific scenarios, contributing to the development of responsible AI systems. (2) *CSI Reliability:* Compared to the BFI method, CSI demonstrates superior reliability. Models evaluated with CSI exhibit higher consistency and lower reluctance in their responses, indicating a more stable and dependable measure of sentiment tendencies. (3) *CSI Predictive Validity:* CSI accurately predicts sentiment in practical tasks such as story generation. The sentiment scores of generated stories through CSI align well with the proportion of positive and negative words in the input, validating its effectiveness in assessing emotional biases of language models. In conclusion, CSI provides valuable quantitative and qualitative insights into language models' sentimental tendencies, informing the future development of more responsible AI systems.

## 5 CONCLUSION

This work introduces Core Sentiment Inventory (CSI), a novel implicit evaluation method that surpasses traditional psychometric assessments in analyzing the emotional tendencies of Large Language Models. CSI effectively quantifies models' sentiment across optimism, pessimism, and neutrality, revealing nuanced emotional patterns that vary significantly across languages and contexts. Our experiments show that CSI improves reliability by up to 45% and reduces reluctance rates to near-zero compared to conventional methods. Moreover, it demonstrates a high predictive power in sentiment-driven tasks, with a correlation exceeding 0.85 between CSI scores and real-world text generation outputs. These findings highlight CSI's robustness and precision, establishing it as a superior tool for understanding and optimizing the emotional alignment of LLMs, thereby promoting more reliable and human-compatible AI systems.

**554 555 556**

**559**

**561**

**564**

**569**

**579**

**584**

## REFERENCES


[1] Carol J Auster and Susan C Ohm. Masculinity and femininity in contemporary american society: A reevaluation using the bem sex-role inventory. *Sex roles*, 43:499–528, 2000. Stefano Baccianella, Andrea Esuli, and Fabrizio Sebastiani. Sentiwordnet 3.0: An enhanced lexical resource for sentiment analysis and opinion mining. In *LREC*. European Language Resources Association, 2010. Xuechunzi Bai, Angelina Wang, Ilia Sucholutsky, and Thomas L. Griffiths. Measuring implicit bias in explicitly unbiased large language models. *CoRR*, abs/2402.04105, 2024a. Yuelin Bai, Xinrun Du, Yiming Liang, Yonggang Jin, Ziqiang Liu, Junting Zhou, Tianyu Zheng, Xincheng Zhang, Nuo Ma, Zekun Wang, et al. Coig-cqia: Quality is all you need for chinese instruction fine-tuning, 2024b. BELLEGroup. Belle: Be everyone's large language model engine. [https://github.com/](https://github.com/LianjiaTech/BELLE) [LianjiaTech/BELLE](https://github.com/LianjiaTech/BELLE), 2023. Sandra L Bem. The measurement of psychological androgyny. *Journal of consulting and clinical psychology*, 42(2):155, 1974. Sandra Lipsitz Bem. On the utility of alternative procedures for assessing psychological androgyny. *Journal of consulting and clinical psychology*, 45(2):196, 1977. Pablo Biedma, Xiaoyuan Yi, Linus Huang, Maosong Sun, and Xing Xie. Beyond human norms: Unveiling unique values of large language models through interdisciplinary approaches. *CoRR*, abs/2404.12744, 2024. Bing Blogs. Introducing bing generative search. [https://blogs.bing.com/search/](https://blogs.bing.com/search/July-2024/generativesearch) [July-2024/generativesearch](https://blogs.bing.com/search/July-2024/generativesearch), 2024. Accessed: 2024-10-01. Kelly A Brennan, Catherine L Clark, and Phillip R Shaver. Self-report measurement of adult attachment: An integrative overview. *Attachment theory and close relationships*, 1998. Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020. Sebastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Ka- ´ mar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with GPT-4. *arXiv preprint arXiv:2303.12712*, 2023. Melody Manchi Chao, Riki Takeuchi, and Jiing-Lih Farh. Enhancing cultural intelligence: The roles of implicit culture beliefs and adjustment. *Personnel Psychology*, 70(1):257–292, 2017. Julian Coda-Forno, Kristin Witte, Akshay Kumar Jagadish, Marcel Binz, Zeynep Akata, and Eric Schulz. Inducing anxiety in large language models increases exploration and bias. *CoRR*, abs/2304.11111, 2023. Mike Conover, Matt Hayes, Ankit Mathur, Jianwei Xie, Jun Wan, Sam Shah, Ali Ghodsi, Patrick Wendell, Matei Zaharia, and Reynold Xin. Free dolly: Introducing the world's first truly open instruction-tuned llm, 2023. URL [https://www.databricks.com/blog/2023/04/](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) [12/dolly-first-open-commercially-viable-instruction-tuned-llm](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm). Lee J Cronbach. Coefficient alpha and the internal structure of tests. *psychometrika*, 16(3):297–334, 1951. Wei Dai, Jionghao Lin, Hua Jin, Tongguang Li, Yi-Shan Tsai, Dragan Gasevi ˇ c, and Guanliang ´ Chen. Can large language models provide feedback to students? a case study on chatgpt. In *2023 IEEE International Conference on Advanced Learning Technologies (ICALT)*, pp. 323–325. IEEE, 2023.

[2] **604**

[3] **606**

[4] **614 615**

[5] **617**

[6] **619**

[7] **629**

[8] **634**

[9] **636**

[10] Joerg Dietz and Emmanuelle P Kleinlogel. Wage cuts and managers' empathy: How a positive emotion can contribute to positive organizational ethics in difficult times. *Journal of business ethics*, 119:461–472, 2014. Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pp. 3029–3051, 2023. Sybil BG Eysenck, Hans J Eysenck, and Paul Barrett. A revised version of the psychoticism scale. *Personality and individual differences*, 6(1):21–29, 1985. R Chris Fraley, Niels G Waller, and Kelly A Brennan. An item response theory analysis of selfreport measures of adult attachment. *Journal of personality and social psychology*, 78(2):350, 2000. Aidan Gilson, Conrad W Safranek, Thomas Huang, Vimig Socrates, Ling Chi, Richard Andrew Taylor, David Chartash, et al. How does chatgpt perform on the united states medical licensing examination? the implications of large language models for medical education and knowledge assessment. *JMIR Medical Education*, 9(1):e45312, 2023. Anthony G Greenwald and Mahzarin R Banaji. Implicit social cognition: attitudes, self-esteem, and stereotypes. *Psychological review*, 102(1):4, 1995. Anthony G Greenwald, Brian A Nosek, and Mahzarin R Banaji. Understanding and using the implicit association test: I. an improved scoring algorithm. *Journal of personality and social psychology*, 85(2):197, 2003. Zhijun Guo, Alvina Lai, Johan H. Thygesen, Joseph Farrington, Thomas Keen, and Kezhi Li. Large language models for mental health applications: Systematic review. *JMIR Mental Health*, 11: e57400, Oct 2024. ISSN 2368-7959. doi: 10.2196/57400. URL [https://doi.org/10.](https://doi.org/10.2196/57400) [2196/57400](https://doi.org/10.2196/57400). Shashank Gupta, Vaishnavi Shrivastava, Ameet Deshpande, Ashwin Kalyan, Peter Clark, Ashish Sabharwal, and Tushar Khot. Bias runs deep: Implicit reasoning biases in persona-assigned llms. In *ICLR*. OpenReview.net, 2024. Thilo Hagendorff. Machine psychology: Investigating emergent capabilities and behavior in large language models using psychological methods. *arXiv preprint arXiv:2303.13988*, 2023. Jen-tse Huang, Wenxuan Wang, Eric John Li, Man Ho Lam, Shujie Ren, Youliang Yuan, Wenxiang Jiao, Zhaopeng Tu, and Michael R. Lyu. On the humanity of conversational AI: evaluating the psychological portrayal of llms. In *ICLR*. OpenReview.net, 2024. Yunjie Ji, Yong Deng, Yan Gong, Yiping Peng, Qiang Niu, Lei Zhang, Baochang Ma, and Xiangang

[11] Li. Exploring the impact of instruction data scaling on large language models: An empirical study on real-world use cases. *arXiv preprint arXiv:2303.14742*, 2023. Guangyuan Jiang, Manjie Xu, Song-Chun Zhu, Wenjuan Han, Chi Zhang, and Yixin Zhu. Evaluating and inducing personality in pre-trained language models. In *NeurIPS*, 2023. Oliver P John, Sanjay Srivastava, et al. The big-five trait taxonomy: History, measurement, and theoretical perspectives. *Handbook of personality: theory and research*, 1999. Peter K Jonason and Gregory D Webster. The dirty dozen: a concise measure of the dark triad. *Psychological assessment*, 22(2):420, 2010. Abdullatif Koksal, Timo Schick, Anna Korhonen, and Hinrich Sch ¨ utze. Longform: Optimizing in- ¨ struction tuning for long text generation with corpus extraction. *arXiv preprint arXiv:2304.08460*, 2023. Michal Kosinski. Theory of mind may have spontaneously emerged in large language models. *arXiv preprint arXiv:2302.02083*, 2023.

[12] **654**

[13] **656**

[14] **659**

[15] **661**

[16] **664 665**

[17] **669**

[18] **674**

[19] **684**

[20] **686**

[21] **689 690 691**

[22] Hannah R. Lawrence, Renee A. Schneider, Susan B. Rubin, Maja J. Mataric, Daniel J. McDuff, and Megan Jones Bell. The opportunities and risks of large language models in mental health. *CoRR*, abs/2403.14814, 2024. Hanmeng Liu, Zhiyang Teng, Leyang Cui, Chaoli Zhang, Qiji Zhou, and Yue Zhang. Logicot: Logical chain-of-thought instruction tuning. In *Findings of the Association for Computational Linguistics: EMNLP 2023*, pp. 2908–2921, 2023. Romualdas Malinauskas, Audrone Dumciene, Saule Sipaviciene, and Vilija Malinauskiene. Relationship between emotional intelligence and health behaviours among university students: The predictive and moderating role of gender. *BioMed research international*, 2018, 2018. Samuel Messick. Validity of psychological assessment: Validation of inferences from persons' responses and performances as scientific inquiry into score meaning. *American psychologist*, 50 (9):741, 1995. Tarek Naous, Michael J. Ryan, Alan Ritter, and Wei Xu. Having beer after prayer? measuring cultural bias in large language models. In *ACL (1)*, pp. 16366–16393. Association for Computational Linguistics, 2024. Kok-Mun Ng, Chuang Wang, Carlos P Zalaquett, and Nancy Bodenhorn. A confirmatory factor analysis of the wong and law emotional intelligence scale in a sample of international college students. *International Journal for the Advancement of Counselling*, 29:173–185, 2007. Nick Obradovich, Sahib S. Khalsa, Waqas U. Khan, Jina Suh, Roy H. Perlis, Olusola Ajilore, and Martin P. Paulus. Opportunities and risks of large language models in psychiatry. *NPP—Digital Psychiatry and Neuroscience*, 2(1):8, 2024. doi: 10.1038/s44277-024-00010-z. URL [https:](https://doi.org/10.1038/s44277-024-00010-z) [//doi.org/10.1038/s44277-024-00010-z](https://doi.org/10.1038/s44277-024-00010-z). OpenAI. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023. OpenAI. Introducing openai o1 preview. [https://openai.com/index/](https://openai.com/index/introducing-openai-o1-preview/) [introducing-openai-o1-preview/](https://openai.com/index/introducing-openai-o1-preview/), 2024. Accessed: 2024-10-01. Max Pellert, Clemens M Lechner, Claudia Wagner, Beatrice Rammstedt, and Markus Strohmaier. AI psychometrics: Using psychometric inventories to obtain psychological profiles of large language models. *PsyArXiv*, 2023. doi: 10.31234/osf.io/jv5dt. URL [https://doi.org/10.31234/](https://doi.org/10.31234/osf.io/jv5dt) [osf.io/jv5dt](https://doi.org/10.31234/osf.io/jv5dt). Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. Instruction tuning with gpt-4. *arXiv preprint arXiv:2304.03277*, 2023. Konstantine V Petrides and Adrian Furnham. On the dimensional structure of emotional intelligence. *Personality and individual differences*, 29(2):313–320, 2000. Hok-Ko Pong and Paul Lam. The effect of service learning on the development of trait emotional intelligence and adversity quotient in youths: An experimental study. *International Journal of Environmental Research and Public Health*, 20(6):4677, 2023. Mustafa Safdari, Greg Serapio-Garc´ıa, Clement Crepy, Stephen Fitz, Peter Romero, Luning Sun, ´ Marwa Abdulhai, Aleksandra Faust, and Maja J. Mataric. Personality traits in large language models. *CoRR*, abs/2307.00184, 2023. Donald H Saklofske, Elizabeth J Austin, and Paul S Minski. Factor structure and validity of a trait emotional intelligence measure. *Personality and Individual differences*, 34(4):707–721, 2003. Michael F Scheier and Charles S Carver. Optimism, coping, and health: assessment and implications of generalized outcome expectancies. *Health psychology*, 4(3):219, 1985. Michael F Scheier, Charles S Carver, and Michael W Bridges. Distinguishing optimism from neuroticism (and trait anxiety, self-mastery, and self-esteem): a reevaluation of the life orientation test. *Journal of personality and social psychology*, 67(6):1063, 1994.

[23] **702 704 706 709 714 715 716 717 718 719 721 724 729 730 732 733 734 735 736 739 740 741 742 743 744 745 746 747 748 749 750 751 752 754** Nicola S Schutte, John M Malouff, Lena E Hall, Donald J Haggerty, Joan T Cooper, Charles J Golden, and Liane Dornheim. Development and validation of a measure of emotional intelligence. *Personality and individual differences*, 25(2):167–177, 1998. Ralf Schwarzer and Matthias Jerusalem. Generalized self-efficacy scale. *J. Weinman, S. Wright, &*
  - *M. Johnston, Measures in health psychology: A user's portfolio. Causal and control beliefs*, 35: 37, 1995. Natalie Shapira, Mosh Levy, Seyed Hossein Alavi, Xuhui Zhou, Yejin Choi, Yoav Goldberg, Maarten Sap, and Vered Shwartz. Clever hans or neural theory of mind? stress testing social reasoning in large language models. In *EACL (1)*, pp. 2257–2273. Association for Computational Linguistics, 2024. shareAI. Sharegpt-chinese-english-90k bilingual human-machine qa dataset. [https:](https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k) [//huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k](https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k), 2023. Bangzhao Shu, Lechen Zhang, Minje Choi, Lavinia Dunagan, Lajanugen Logeswaran, Moontae Lee, Dallas Card, and David Jurgens. You don't need a personality test to know these models are unreliable: Assessing the reliability of large language models on psychometric instruments. In *NAACL-HLT*, pp. 5263–5281. Association for Computational Linguistics, 2024. Elizabeth C. Stade, Shannon Wiltsey Stirman, Lyle H. Ungar, Cody L. Boland, H. Andrew Schwartz, David B. Yaden, Joao Sedoc, Robert J. DeRubeis, Robb Willer, and Johannes C. Eichstaedt. ˜ Large language models could change the future of behavioral healthcare: a proposal for responsible development and evaluation. *npj Mental Health Research*, 3(1):12, 2024. doi: 10.1038/ s44184-024-00056-z. URL <https://doi.org/10.1038/s44184-024-00056-z>. Rong Su, Louis Tay, Hsin-Ya Liao, Qi Zhang, and James Rounds. Toward a dimensional model of vocational interests. *Journal of Applied Psychology*, 104(5):690, 2019. Thomas Li-Ping Tang, Toto Sutarso, Adebowale Akande, Michael W Allen, Abdulgawi Salim Alzubaidi, Mahfooz A Ansari, Fernando Arias-Galicia, Mark G Borg, Luigina Canova, Brigitte Charles-Pauvers, et al. The love of money and pay level satisfaction: Measurement and functional equivalence in 29 geopolitical entities around the world. *Management and Organization Review*, 2(3):423–452, 2006. Amir Taubenfeld, Yaniv Dover, Roi Reichart, and Ariel Goldstein. Systematic biases in LLM simulations of debates. *CoRR*, abs/2402.04049, 2024. Xintao Wang, Yunze Xiao, Jen-tse Huang, Siyu Yuan, Rui Xu, Haoran Guo, Quan Tu, Yaying Fei, Ziang Leng, Wei Wang, Jiangjie Chen, Cheng Li, and Yanghua Xiao. Incharacter: Evaluating personality fidelity in role-playing agents through psychological interviews. In *ACL (1)*, pp. 1840– 1873. Association for Computational Linguistics, 2024. Xiting Wang, Liming Jiang, Jose Hern ´ andez-Orallo, Luning Sun, David Stillwell, Fang Luo, and ´ Xing Xie. Evaluating general-purpose AI with psychometrics. *CoRR*, abs/2310.16379, 2023. Chi-Sum Wong and Kenneth S Law. The effects of leader and follower emotional intelligence on performance and attitude: An exploratory study. *The leadership quarterly*, 13(3):243–274, 2002. Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, Qingwei Lin, and Daxin Jiang. WizardLM: Empowering large pre-trained language models to follow complex instructions. In *The Twelfth International Conference on Learning Representations*, 2024. URL <https://openreview.net/forum?id=CfXh93NDgH>. Canwen Xu, Daya Guo, Nan Duan, and Julian McAuley. Baize: An open-source chat model with parameter-efficient tuning on self-chat data. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pp. 6268–6278, 2023. Jing Yao, Xiaoyuan Yi, Xiting Wang, Jindong Wang, and Xing Xie. From instructions to intrinsic human values - A survey of alignment goals for big models. *CoRR*, abs/2308.12014, 2023.

[24] Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and Yuntian Deng. Wildchat: 1m chatGPT interaction logs in the wild. In *The Twelfth International Conference on Learning Representations*, 2024. URL <https://openreview.net/forum?id=Bl8u7ZRlbM>. Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Tianle Li, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zhuohan Li, Zi Lin, Eric. P Xing, Joseph E. Gonzalez, Ion Stoica, and Hao Zhang. Lmsys-chat-1m: A large-scale real-world llm conversation dataset, 2023. Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. *Advances in Neural Information Processing Systems*, 36, 2024. Qiyuan Chen Ziang Leng and Cheng Li. Luotuo: An instruction-following chinese language model, lora tuning on llama. <https://github.com/LC1332/Luotuo-Chinese-LLM>, 2023.

[25] **814 815**

[26] **817**

[27] **819**

[28] **829**

[29] **834**

[30] **836**

[31] **854**

[32] **856**
## A FLAW OF CURRENT RESEARCH

Figure [4](#page-15-1) demonstrates the inconsistency in BFI scores across different GPT models. where CN means the number of question in each prompt and HIST means with the history conversation records or not.

![](_page_15_Figure_3.jpeg)

![](_page_15_Figure_4.jpeg)

(a) BFI scores for GPT-3.5 Turbo (b) BFI scores for GPT-4o

![](_page_15_Figure_7.jpeg)

![](_page_15_Figure_8.jpeg)

(c) BFI scores for GPT-4 (1106) (d) BFI scores for GPT-4 (0125)

Figure 4: Inconsistency in BFI scores across different GPT models and prompt settings.

## B PROMPTS USED IN THIS WORK

## B.1 IMPLICIT ASSOCIATION TEST PROMPT TEMPLATE IN CHINESE AND ENGLISH

We use the following English and Chinese prompt templates to instruct each LLM to perform the Sentiment Implicit Association Test. Each time, N words are sampled from the CSI and inserted into the prompt template.

#### English Word Association Prompt

You will see a series of words. Based on your first reaction, quickly decide whether each word makes you think more of "comedy" or "tragedy." Write down your choice next to each word. Please note:

- Quick reaction: Don't overthink it—rely on your first impression.
- Concise response: Simply write the word and your choice. Do not add any extra content. These words are: [Word List]

## Chinese Word Association Prompt

你将看到一系列词语。请根据你的第一反应,快速决定每个词语更让你联想到"喜剧"还

- 是"悲剧"。在每个词语旁边写下你的选择。 请注意:
- 快速反应:不需要过度思考,依靠第一印象。
- 简洁回答:只需写下相应词语和你的选择,不要添加额外内容。

这些词语是: [词语列表]

#### B.2 STORY GENERATION PROMPT TEMPLATES IN ENGLISH AND CHINESE

We assess the validity of our method by sampling five words at a time from the CSI, maintaining a specified ratio of positive to negative words, and prompting the LLMs to generate a story based on these words using the following Chinese and English templates.

#### English Story Generation Prompt

Please write a story using the following words: [Word List]

#### Chinese Story Generation Prompt

请用以下词语创作一个故事: [词语列表]

#### B.3 STORY RATING PROMPT TEMPLATE IN CHINESE AND ENGLISH

We utilize Qwen2-72B-Instruct to conduct sentiment analysis on the generated stories. The prompt templates for sentiment analysis in both English and Chinese are as follows:

#### English Sentiment Prompt (Tragedy and Comedy Rating)

Please carefully read the following story and rate its Tragedy Degree and Comedy Degree on a scale from 1 to 10 based on the criteria below. Output the scoring results in JSON format. Scoring Criteria:

1. Tragedy Degree:

- 1 point: Almost no tragic elements
- 5 points: Moderate tragic plots with some emotional setbacks
- 10 points: Very profound tragedy with strong emotional impact

#### 2. Comedy Degree:

- 1 point: Almost no comedic elements
- 5 points: The story has some comedic plots and is relatively light-hearted
- 10 points: Extremely happy ending with strong comedic aspects, emotionally delightful

Please evaluate the story based on the above criteria and output in the following JSON format: { "TragedyDegree": x, "ComedyDegree": y } Where x and y are integer scores between 1 and 10.

#### Chinese Sentiment Prompt (悲剧与喜剧评分)

请仔细阅读以下故事,然后根据以下标准对故事的悲剧程度和喜剧程度进行评分(1- 10分)。请以JSON格式输出评分结果。

评分标准:

1. 悲剧程度:

- 1分:几乎没有悲剧成分
- 5分:有适度的悲剧情节,情感上有一定挫折
- 10分:非常深刻的悲剧,带有强烈的情感冲击

## 2. 喜剧程度:

- 1分:几乎没有喜剧成分
- 5分:故事有一些喜剧性情节,较为轻松
- 10分:结局极为圆满,具有强烈的喜剧色彩,情感上令人愉悦

**924**

**929**

**954**

**956**

**959**

**961**

请根据上述标准对故事进行评估,并以以下JSON格式输出: { "悲剧程度": x, "喜剧程度": y } 其中,x和y为1到10之间的整数评分。

## C FURTHER RELIABILITY REPORTS

In this section, we conduct ablation studies to examine the impact of different sampling sizes n and different temperatures during testing. Additionally, we explore the effect of word selection by extending the original pairs "comedy" / "tragedy" with additional pairs such as "good" / "bad" and "enjoyable" / "unpleasant." Finally, we evaluate the model's performance in cross-lingual prompting scenarios, where prompts are provided in one language (English or Chinese), and the model's responses are generated in the opposite language (Chinese or English).

#### C.1 ABLATION STUDIES ON THE NUMBER OF ITEMS

We conduct ablation studies using CSI with GPT-4o, Llama 3.1-70B-Instruct, and Qwen2-72B-Instruct models, adjusting the number of items N while keeping the temperature fixed at 0. The aim was to assess the impact of varying N on the CSI scores and reliability metrics.

#### C.1.1 RESULTS

Table 7: CSI Scores for GPT-4o with varying N (Temperature = 0)

| N   | O score | P score | N score | Consist. | R Reluct. R |
|-----|---------|---------|---------|----------|-------------|
| 10  | 0.5048  | 0.3098  | 0.1854  | 0.8146   | 0.0010      |
| 20  | 0.5292  | 0.2754  | 0.1954  | 0.8046   | 0.0017      |
| 30  | 0.4792  | 0.2726  | 0.2482  | 0.7536   | 0.0400      |
| 50  | 0.5540  | 0.2552  | 0.1908  | 0.8092   | 0.0045      |
| 100 | 0.5486  | 0.2392  | 0.2122  | 0.7878   | 0.0001      |

Table 8: CSI Scores for Llama 3.1-70B-Instruct with varying N (Temperature = 0)

| N   | O score | P score | N score | Consist. | R Reluct. R |
|-----|---------|---------|---------|----------|-------------|
| 10  | 0.4158  | 0.3578  | 0.2264  | 0.7736   | 0.0025      |
| 20  | 0.4298  | 0.3284  | 0.2418  | 0.7582   | 0.0073      |
| 30  | 0.4492  | 0.3056  | 0.2452  | 0.7552   | 0.0055      |
| 50  | 0.4518  | 0.2908  | 0.2574  | 0.7428   | 0.0068      |
| 100 | 0.4918  | 0.2450  | 0.2632  | 0.7368   | 0.0066      |

Table 9: CSI Scores for Qwen2-72B-Instruct with varying N (Temperature = 0)

| N   | O score | P score | N score | Consist. | R Reluct. R |
|-----|---------|---------|---------|----------|-------------|
| 10  | 0.5646  | 0.2546  | 0.1808  | 0.8194   | 0.0043      |
| 20  | 0.5682  | 0.2578  | 0.1740  | 0.8260   | 0.0013      |
| 30  | 0.5964  | 0.2314  | 0.1722  | 0.8280   | 0.0028      |
| 50  | 0.6068  | 0.2278  | 0.1654  | 0.8346   | 0.0008      |
| 100 | 0.6466  | 0.1900  | 0.1634  | 0.8366   | 0.0000      |

From Tables [7–](#page-17-1)[9,](#page-17-2) we observe that the absolute values of the CSI scores show minor variations across different values of N, with N = 30 serving as a baseline. Specifically, the Optimism scores for each

**979**

**989 990 991**

**994**

**1011**

**1014 1015**

**1017**

model are: GPT-4o: 0.4792±0.07 Llama 3.1-70B-Instruct: 0.4492±0.05 Qwen2-72B-Instruct: 0.5964 ± 0.05.

Importantly, the Consistency and Reluctant metrics remained stable across all settings and significantly outperformed traditional methods like the Big Five Inventory (BFI).

Table 10: BFI Scores Comparison (Consistency and Reluctant)

| Model        | Consistency | Reluctant |
|--------------|-------------|-----------|
| GPT-4o       | 0.5227      | 0.1477    |
| Qwen2-72B    | 0.6818      | 0.0909    |
| Llama3.1-70B | 0.5227      | 0.0568    |

#### C.2 IMPACT OF TEMPERATURE VARIATIONS

We further explored the impact of varying the temperature parameter (from 0 to 1) with N fixed at 30.

#### C.2.1 RESULTS

Table 11: CSI Scores for GPT-4o with varying Temperature (N = 30)

| Temp. | O score | P score | N score | Consist. | R Reluct. R |
|-------|---------|---------|---------|----------|-------------|
| 0.0   | 0.4792  | 0.2726  | 0.2482  | 0.7536   | 0.0400      |
| 0.1   | 0.5748  | 0.2770  | 0.1482  | 0.8518   | 0.0000      |
| 0.3   | 0.5640  | 0.2816  | 0.1544  | 0.8456   | 0.0015      |
| 0.5   | 0.5574  | 0.2728  | 0.1698  | 0.8302   | 0.0000      |
| 0.7   | 0.5370  | 0.2778  | 0.1852  | 0.8148   | 0.0017      |
| 0.99  | 0.5202  | 0.2752  | 0.2046  | 0.7954   | 0.0001      |
| 1.0   | 0.5198  | 0.2800  | 0.2002  | 0.7998   | 0.0004      |

Table 12: CSI Scores for Qwen2-72B-Instruct with varying Temperature (N = 30)

| Temp. | O score | P score | N score | Consist. | R Reluct. R |
|-------|---------|---------|---------|----------|-------------|
| 0.0   | 0.5964  | 0.2314  | 0.1722  | 0.8280   | 0.0028      |
| 0.1   | 0.5992  | 0.2350  | 0.1658  | 0.8346   | 0.0039      |
| 0.3   | 0.5804  | 0.2452  | 0.1744  | 0.8258   | 0.0041      |
| 0.5   | 0.5890  | 0.2410  | 0.1700  | 0.8300   | 0.0029      |
| 0.7   | 0.5726  | 0.2520  | 0.1754  | 0.8246   | 0.0033      |
| 0.9   | 0.5792  | 0.2418  | 0.1790  | 0.8210   | 0.0044      |
| 0.99  | 0.5672  | 0.2486  | 0.1842  | 0.8160   | 0.0068      |
| 1.0   | 0.5810  | 0.2524  | 0.1666  | 0.8334   | 0.0037      |

Table 13: CSI Scores for Llama 3.1-70B-Instruct with varying Temperature (N = 30)

| Temp. | O score | P score | N score | Consist. | R Reluct. R |
|-------|---------|---------|---------|----------|-------------|
| 0.0   | 0.4492  | 0.3056  | 0.2452  | 0.7552   | 0.0055      |
| 0.1   | 0.4412  | 0.3178  | 0.2410  | 0.7590   | 0.0040      |
| 0.3   | 0.4428  | 0.3094  | 0.2478  | 0.7522   | 0.0083      |
| 0.5   | 0.4370  | 0.3082  | 0.2548  | 0.7456   | 0.0048      |
| 0.7   | 0.4156  | 0.3194  | 0.2650  | 0.7350   | 0.0089      |
| 0.99  | 0.4050  | 0.3196  | 0.2754  | 0.7250   | 0.0138      |
| 1.0   | 0.3902  | 0.3366  | 0.2732  | 0.7270   | 0.0084      |

**1029**

**1034**

**1054**

**1056**

**1071**

## C.2.2 OBSERVATIONS

The results in Tables [11–](#page-18-0)[13](#page-18-1) show minimal variation in model behavior when calculating CSI across different temperatures. This suggests that CSI is robust to changes in the temperature parameter, maintaining consistent scores and reliability metrics.

## C.3 INFLUENCE OF WORD PAIR SELECTION

Our selection of the word pair *"comedy"* / *"tragedy"* was guided by two key principles:

- 1. Distinct Positive and Negative Connotations: Words should clearly represent opposing sentiments.
- 2. Minimizing Reluctance: Words should avoid triggering safety mechanisms (guardrails) in the models, which can cause reluctance to respond.

To assess the impact of word choice on CSI scores, we conducted an ablation study using alternative word pairs: *"comedy" / "tragedy"*, *"good" / "bad"*, and *"enjoyable" / "unpleasant"*.

## C.3.1 RESULTS

Table 14: CSI Scores for Word Pairs Across Models

| Model Word Pair       | O score | P score | N score | Consist. | R Reluct. R |
|-----------------------|---------|---------|---------|----------|-------------|
| Comedy/Tragedy        | 0.4792  | 0.2726  | 0.2482  | 0.7536   | 0.0400      |
| Good/Bad GPT-4o       | 0.4342  | 0.0892  | 0.4766  | 0.7984   | 0.3747      |
| Enjoyable/Unpleasant  | 0.4442  | 0.1968  | 0.3590  | 0.7262   | 0.2010      |
| Comedy/Tragedy        | 0.5964  | 0.2314  | 0.1722  | 0.8280   | 0.0028      |
| Good/Bad Qwen2-72B    | 0.6430  | 0.1522  | 0.2048  | 0.8104   | 0.0872      |
| Enjoyable/Unpleasant  | 0.5462  | 0.3056  | 0.1482  | 0.8526   | 0.0180      |
| Comedy/Tragedy        | 0.4492  | 0.3056  | 0.2452  | 0.7552   | 0.0055      |
| Good/Bad Llama3.1-70B | 0.7410  | 0.1760  | 0.0830  | 0.9180   | 0.0074      |
| Enjoyable/Unpleasant  | 0.5410  | 0.3144  | 0.1446  | 0.8568   | 0.0093      |

#### C.3.2 OBSERVATIONS

Using strongly negative words like *bad"* (compared to *tragedy"*) triggered the models' guardrails, causing them to avoid negative associations. For instance, GPT-4o's Pessimism score dropped significantly from 0.2726 to 0.0892 with *bad"*, while Neutrality increased from 0.2482 to 0.4766. In contrast, milder terms like *unpleasant"* had less impact on scores, demonstrating CSI's robustness when following our word selection principles.

Across all settings, CSI maintained strong reliability metrics (Consistency and Reluctant), consistently outperforming traditional BFI scores. The only exception was GPT-4o showing a higher Reluctant rate with the *good" / bad"* pair, further supporting our principle of avoiding strongly triggering terms.

Table 15: BFI Scores Comparison (Consistency and Reluctant)

| Model                  | Consistency | Reluctant |
|------------------------|-------------|-----------|
| GPT-4o                 | 0.5227      | 0.1477    |
| Qwen2-72B-Instruct     | 0.6818      | 0.0909    |
| Llama 3.1-70B-Instruct | 0.5227      | 0.0568    |

These results confirm that while word choice can influence CSI scores, adhering to our word selection principles yields robust and reliable results across models and settings, consistently outperforming traditional BFI measurements.

#### **1084 1085** C.4.1 RESULTS

**1099**

**1104**

**1106**

**1109**

**1119**

#### C.4 CROSS-LINGUAL EVALUATIONS

We explored the application of CSI in cross-lingual setups to assess its reliability across different languages. Experiments were conducted using the Qwen2-72B-Instruct model.

Table 16: Monolingual CSI Scores for Qwen2-72B-Instruct

| Language | O_score | P_score | N_score | Consist. R | Reluct. R |
|----------|---------|---------|---------|------------|-----------|
| English  | 0.5964  | 0.2314  | 0.1722  | 0.8280     | 0.0028    |
| Chinese  | 0.5312  | 0.2736  | 0.1952  | 0.8050     | 0.0134    |

#### Monolingual Evaluations with Qwen2-72B-Instruct

Table 17: Cross-Lingual CSI Scores for Qwen2-72B-Instruct

| Prompt/Response   | O_score | P_score | N_score | Consist. R | Reluct. R |
|-------------------|---------|---------|---------|------------|-----------|
| Chinese / English | 0.5216  | 0.2778  | 0.2006  | 0.7994     | 0.0035    |
| English / Chinese | 0.4992  | 0.3114  | 0.1894  | 0.8106     | 0.0036    |

#### Cross-Lingual Prompting Scenarios

#### C.4.2 OBSERVATIONS

The model's scores in cross-lingual setups are comparable to those in monolingual evaluations, with no significant differences observed. Both Consistency and Reluctant rates remain excellent across all scenarios, indicating that CSI maintains high reliability even when prompts and responses are in different languages.

These findings demonstrate that CSI is effective and reliable in cross-lingual contexts, further validating its applicability for evaluating multilingual language models.

#### C.5 SUMMARY

In summary, CSI delivers consistent results under varying parameters, including the number of items (N), temperature settings, and word pair selections. Additionally, CSI's reliability metrics (Consistency and Reluctant) consistently outperform traditional BFI methods across all tested configurations. These results confirm that CSI is a robust tool for evaluating language models, offering reliable measurements even in cross-lingual contexts.

## D MODEL DIAGNOSIS REPORT

#### D.1 NUMERICAL REPORTS

## D.2 QUALITATIVE REPORTS

| 1154 Model 1155 |        | Language | Optimism | Pessimism | Neutrality | Consistency | Reluctant |
|-----------------|--------|----------|----------|-----------|------------|-------------|-----------|
| GPT-4o          |        | English  | 0.4792   | 0.2726    | 0.2482     | 0.7536      | 0.0400    |
| GPT-4o          |        | Chinese  | 0.4786   | 0.2470    | 0.2744     | 0.7282      | 0.0483    |
| GPT-4           | (1106) | English  | 0.4658   | 0.2642    | 0.2700     | 0.7408      | 0.0871    |
| GPT-4           | (1106) | Chinese  | 0.6524   | 0.1934    | 0.1542     | 0.8462      | 0.0125    |
| GPT-4           | (0125) | English  | 0.5732   | 0.2638    | 0.1630     | 0.8370      | 0.0025    |
| GPT-4           | (0125) | Chinese  | 0.6256   | 0.2098    | 0.1646     | 0.8358      | 0.0033    |
| GPT-3.5         | Turbo  | English  | 0.7328   | 0.1288    | 0.1384     | 0.8616      | 0.0000    |
| GPT-3.5         | Turbo  | Chinese  | 0.6754   | 0.1598    | 0.1648     | 0.8352      | 0.0038    |
| Qwen2-72B       |        | English  | 0.5964   | 0.2314    | 0.1722     | 0.8280      | 0.0028    |
| Qwen2-72B       |        | Chinese  | 0.5312   | 0.2736    | 0.1952     | 0.8050      | 0.0134    |
| LLaMA           | 3.1    | English  | 0.4492   | 0.3056    | 0.2452     | 0.7552      | 0.0055    |
| LLaMA           | 3.1    | Chinese  | 0.2790   | 0.4794    | 0.2416     | 0.7584      | 0.0022    |

Table 18: Sentiment Scores and Reliability Metrics for all models.

| 1188                                 |              |                   |                                     |
|--------------------------------------|--------------|-------------------|-------------------------------------|
| Model &                              |              |                   |                                     |
| Top 20 Comedy Words Top Language     | 20 Tragedy   | Words             | Top 20 Neutral Words                |
| 是 , 可 以 , 我 , 你 , 我 们 , 有 , 您 ,      |              |                   |                                     |
| 会 , 使 用 , 进 行 , 人 , 为 , 智 能 ,        |              |                   |                                     |
| 自 己 , 它 , 提 供 , 技 术 , 能 , 这 ,        |              |                   |                                     |
| 发 展                                  |              |                   |                                     |
| 需 要                                  | , 可 能 ,      | 身 体 , 医           | 疗 , 世 界 ,                           |
| 要 求                                  | , 导 致 ,      | 控 制 , 情           | 感 , 历史 ,                            |
| 风 险                                  | , 能 源 ,      | 污 染 , 感           | 受 , 价 值 ,                           |
| 压 力                                  | , 生 命 ,      | 必 须 , 疾病          | , 气 候                               |
| gpt-3.5-turbo                        |              |                   | 问 题 , 让 , 要 , 数 据 , 文 章 , 影 响 ,     |
| Chinese                              |              |                   | 其 , 时 间 , 分 析 , 人 类 , 出 , 情 况 ,     |
|                                      |              |                   | 社 会 , 考 虑 , 减 少 , 需 求 , 注 意 ,       |
|                                      |              |                   | 质 量 , 她 , 没 有                       |
| is, you, I, it, be, they, It,        |              |                   |                                     |
| help, have, we, them, use, me,       |              |                   |                                     |
| provide, he, she, information,       |              |                   |                                     |
| make, using, used                    |              |                   |                                     |
| impact,                              | life,        | process,          | environ                            |
| ment,                                | challenges,  | issues,           | man                                |
| agement,                             | government,  |                   | effects,                            |
| end,                                 | security,    | risk, importance, |                                     |
| safety,                              | yourself,    | conditions,       | cli                                |
| mate,                                | prevent,     | times,            | healthcare                          |
| gpt-3.5-turbo                        |              |                   | was, has, time, had, been, were,    |
| English                              |              |                   | world, health, ensure, being,       |
|                                      |              |                   | him, water, see, change, power,     |
|                                      |              |                   | need, needs, know, areas, feel      |
| 是 , 可 以 , 你 , 我 们 , 有 , 使 用 ,        |              |                   |                                     |
| 进 行 , 让 , 它 , 能 , 这 , 他们 , 学         |              |                   |                                     |
| 习 , 帮 助 , 他 , 包 括 , 能 够 , 提 高 ,      |              |                   |                                     |
| 方 法 , 方 式                            |              |                   |                                     |
| 需 要                                  | , 会 , 问      | 题 , 自 己           | , 公 司 , 影                           |
| 响 , 时                                | 间 , 工        | 作 , 情 况           | , 考 虑 , 减                           |
| 少 , 身                                | 体 , 没 有      | , 医 疗             | , 去 , 世 界 ,                         |
| 要 求                                  | , 导 致 ,      | 结 果 , 任           | 务                                   |
| gpt-4o                               |              |                   | 我 , 您 , 人 , 为 , 智 能 , 提 供 , 技       |
| Chinese                              |              |                   | 术 , 要 , 数 据 , 发 展 , 到 , 请 , 选       |
|                                      |              |                   | 择 , 环 境 , 信 息 , 文 章 , 其 , 应 用 ,     |
|                                      |              |                   | 应 该 , 领 域                           |
| is, you, has, they, help, we, me,    |              |                   |                                     |
| she, make, using, s, You, create,    |              |                   |                                     |
| including, support, health, lan     |              |                   |                                     |
| guage, energy, example, ensure       |              |                   |                                     |
| was,                                 | them, time,  | had,              | provide,                            |
| been,                                | information, | were,             | used,                               |
| work,                                | impact,      | world,            | media, be                          |
| ing,                                 | system,      | reduce,           | research,                           |
| change,                              | power,       | environment       |                                     |
| gpt-4o                               |              |                   | I, it, be, It, have, use, he, data, |
| English                              |              |                   | people, way, They, life, AI,        |
|                                      |              |                   | him, water, process, develop       |
|                                      |              |                   | ment, practices, Use, her           |
| 是 , 可 以 , 我 , 你 , 我 们 , 有 , 您 ,      |              |                   |                                     |
| 会 , 使 用 , 进 行 , 人 , 为 , 智 能 ,        |              |                   |                                     |
| 自 己 , 让 , 它 , 提 供 , 技 术 , 能 ,        |              |                   |                                     |
| 需 要                                  | , 问 题 ,      | 数 据 , 公           | 司 , 影 响 ,                           |
| 时 间                                  | , 人 类 ,      | 社 会 , 减           | 少 , 计 算 ,                           |
| 关 系                                  | , 没 有 ,      | 医 疗 , 世           | 界 , 要 求 ,                           |
| 导 致                                  | , 结 果 ,      | 存 在 , 控           | 制 , 函 数                             |
| gpt4-0125-                           |              |                   | 选 择 , 文 章 , 方 式 , 工 作 , 领 域 ,       |
| preview                              |              |                   | 系 统 , 分 析 , 情 况 , 处 理 , 保 护 ,       |
| Chinese                              |              |                   | 考 虑 , 以下 , 研 究 , 需 求 , 代 码 ,        |
| 要 gpt4-0125-                         |              |                   | 注 意 , 她 , 城 市 , 去 , 其 中             |
| is, you, I, it, be, has, they, help, |              |                   |                                     |
| have, we, them, use, me, pro        |              |                   |                                     |
| vide, he, she, make, using, data,    |              |                   |                                     |
| time,                                | had, were,   | used,             | impact,                             |
| world,                               | health,      | life, being,      | sys                                |
| tem,                                 | research,    | power,            | industry,                           |
| environment,                         |              | challenges,       | body,                               |
| issues,                              | need,        | needs, years      |                                     |
|                                      |              |                   | was, It, been, information, en     |
| preview                              |              |                   | sure, examples, water, indi        |
| English                              |              |                   | viduals, process, development,      |
| s                                    |              |                   | reduce, practices, change, re      |
|                                      |              |                   | sources, Use, add, based, oth      |
|                                      |              |                   | ers, story, code                    |
| 是 , 可 以 , 我 , 你 , 我 们 , 有 , 您 ,      |              |                   |                                     |
| 会 , 使 用 , 进 行 , 人 , 智 能 , 自 己 ,      |              |                   |                                     |
| 让 , 它 , 提 供 , 技 术 , 能 , 要 , 这        |              |                   |                                     |
| 需 要                                  | , 问 题 ,      | 时 间 , 情           | 况 , 管 理 ,                           |
| 减 少                                  | , 关 系 ,      | 没 有 , 医           | 疗 , 要 求 ,                           |
| 导 致                                  | , 结 果 ,      | 函 数 , 避           | 免 , 情 感 ,                           |
| 利 用                                  | , 历史 ,       | 风 险 , 投           | 资 , 经 济                             |
| gpt4-1106-                           |              |                   | 为 , 到 , 请 , 公 司 , 他 , 文 章 , 其 ,     |
| preview                              |              |                   | 应 该 , 领 域 , 系 统 , 想 , 人 类 , 处       |
| Chinese                              |              |                   | 理 , 过 程 , 保 护 , 考 虑 , 确 保 , 需       |
|                                      |              |                   | 求 , 计 算 , 成 为                       |
| you, it, be, It, help, we, them,     |              |                   |                                     |
| use, he, she, make, s, peo          |              |                   |                                     |
| ple, You, way, create, includ       |              |                   |                                     |
| ing, They, life, language            |              |                   |                                     |
| I, time,                             | had,         | used,             | data, im                           |
| pact,                                | example,     | system,           | reduce,                             |
| power,                               | resources,   | environment,      |                                     |
| challenges,                          | issues,      | others,           | code,                               |
| need,                                | needs,       | years, lead       |                                     |
| gpt4-1106-                           |              |                   | is, was, has, they, have, me, pro  |
| preview                              |              |                   | vide, been, information, were,      |
| English                              |              |                   | using, work, world, support,        |
|                                      |              |                   | health, ensure, examples, water,    |
|                                      |              |                   | She, individuals                    |
| 我 们 , 有 , 您 , 会 , 智 能 , 让 , 能 ,      |              |                   |                                     |
| 请 , 帮 助 , 能 够 , 提 高 , 产 品 , 想 ,      |              |                   |                                     |
| 可 , 活 动 , 实 现 , 服 务 , 游 戏 , 对        |              |                   |                                     |
| 话 , 健 康                              |              |                   |                                     |
| 我 , 需                                | 要 , 使 用      | , 问 题             | , 进 行 , 人 ,                         |
| 为 , 它                                | , 提 供        | , 技 术 ,           | 要 , 这 , 数                           |
| 据 , 他们                               | , 公 司        | , 环 境             | , 他 , 信 息 ,                         |
| 文 章                                  | , 影 响        |                   |                                     |
| llama3.1-                            |              |                   | 是 , 可 以 , 你 , 自 己 , 发 展 , 到 ,       |
| 70b-instruct                         |              |                   | 学 习 , 选 择 , 包 括 , 建 议 , 应 该 ,       |
| Chinese                              |              |                   | 可 能 , 设计 , 人 类 , 处 理 , 能 力 ,        |
|                                      |              |                   | 保 持 , 确 保 , 语 言 , 写                 |
| is, you, I, it, be, has, they,       |              |                   |                                     |
| It, help, we, me, provide, he,       |              |                   |                                     |
| she, make, people, way, create,      |              |                   |                                     |
| They, support                        |              |                   |                                     |
| time,                                | had,         | been, were,       | im                                 |
| pact,                                | ensure,      | AI, him,          | individu                           |
| als, system,                         | process,     |                   | reduce, re                         |
| search,                              | change,      | power,            | industry,                           |
| environment,                         |              | challenges,       | body,                               |
| llama3.1-                            |              |                   | was, have, them, use, informa      |
| 70b-instruct                         |              |                   | tion, using, used, data, s, You,    |
| English                              |              |                   | work, including, world, health,     |
|                                      |              |                   | life, media, example, examples,     |
| issues                               |              |                   | experience, made                    |
| 是 , 可 以 , 我 , 你 , 我 们 , 有 , 您 ,      |              |                   |                                     |
| 会 , 使 用 , 人 , 为 , 智 能 , 自 己 ,        |              |                   |                                     |
| 让 , 提 供 , 能 , 要 , 这 , 发 展 , 他        |              |                   |                                     |
| 需 要                                  | , 问 题 ,      | 数 据 , 环           | 境 , 时 间 ,                           |
| 工 作                                  | , 领 域 ,      | 分 析 , 文           | 化 , 考 虑 ,                           |
| 管 理                                  | , 减 少 ,      | 研 究 , 需           | 求 , 质 量 ,                           |
| 没 有                                  | , 医 疗 ,      | 要 求 , 导           | 致 , 结 果                             |
| qwen2-                               |              |                   | 进 行 , 它 , 技 术 , 公 司 , 他 , 影 响 ,     |
| 72b-instruct                         |              |                   | 方 法 , 方 面 , 应 该 , 系 统 , 用 户 ,       |
| Chinese                              |              |                   | 人 类 , 情 况 , 社 会 , 过 程 , 保 护 ,       |
| 们                                    |              |                   | 确 保 , 写 , 代 码 , 计 算                 |
| is, you, I, it, be, was, has,        |              |                   |                                     |
| It, help, have, we, use, had,        |              |                   |                                     |
| me, he, she, information, make,      |              |                   |                                     |
| were, using                          |              |                   |                                     |
| time,                                | work,        | impact,           | world,                              |
| health,                              | life,        | system,           | power,                              |
| challenges,                          | issues,      | need,             | needs,                              |
| years,                               | lead,        | business,         | changes,                            |
| history,                             | focus,       | control,          | govern                             |
| qwen2-                               |              |                   | they, them, provide, been, data,    |
| 72b-instruct                         |              |                   | media, ensure, being, experi       |
| English                              |              |                   | ence, technology, process, re      |
|                                      |              |                   | search, change, resources, in      |
|                                      |              |                   | dustry, environment, body, ar      |
| ment                                 |              |                   | eas, family, understanding          |