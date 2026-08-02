# Reasoning On A Spectrum: Aligning Llms To System 1 And System 2 Thinking

Anonymous Author(s)

| Affiliation Address email   |
|-----------------------------|

## Abstract 16 **1 Introduction**

17 LLMs have demonstrated remarkable reasoning capabilities, often achieving near-human or even 18 superhuman performance (Huang and Chang, 2023). These advances have largely been driven 19 by techniques that simulate step-by-step, deliberative reasoning, such as Chain-of-Thought (CoT) 20 prompting and inference-time interventions (Wei et al., 2022b; Wang et al., 2022). Given their 21 success, such methods are increasingly integrated into LLM training (Chung et al., 2024), reinforcing 22 explicit, structured reasoning regardless of the task necessity. However, the increasing focus on 23 step-by-step reasoning has revealed limitations such as brittle generalization, particularly in tasks 24 requiring nuanced judgment (Delétang et al., 2023), logical consistency (Jiang et al., 2024), or 25 adaptability to uncertainty (Mirzadeh et al., 2024). Similarly, recent analyses frame this issue as 26 "overthinking": Cuadron et al. (2025); Chen et al. (2024) demonstrate that excessive deliberation can 27 hamper decision-making. This problem appears in LLMs' responses to simple factual queries, where 28 they often generate unnecessarily explanations instead of direct responses (Wang et al., 2023). 29 This focus on explicit, structured reasoning highlights a key difference between LLMs and human 30 cognition: while LLMs are being pushed towards a single mode of processing, human reasoning is far 31 more nuanced. Rather than a monolithic process, human reasoning emerges from a sophisticated suite 32 of cognitive tools evolved to tackle a *spectrum* of computational problems. This spectrum of human 33 reasoning encompasses both automatic and reflective processes, a key insight recognized across 1Our data and code are available at https://anonymous.4open.science/r/system12-CB8B
1 Large language models (LLMs) demonstrate remarkable reasoning capabilities, 2 yet their reliance on step-by-step reasoning can make them brittle when tasks do 3 not align with such structured approaches. In contrast, human cognition flexibly 4 alternates between fast, intuitive reasoning (System 1) and slow, analytical reason5 ing (System 2), depending on context. To bridge this gap, we curate a dataset of 6 2K examples, each with valid responses from both reasoning styles, and explicitly 7 align LLMs with System 1 and System 2 reasoning. Evaluations across diverse 8 reasoning benchmarks reveal an accuracy-efficiency trade-off: System 2-aligned 9 models excel in arithmetic and symbolic reasoning, while System 1-aligned models 10 perform better in commonsense tasks. A mechanistic analysis of model responses 11 shows that System 1 models employ more definitive answers, whereas System 12 2 models demonstrate greater uncertainty. Interpolating between these extremes 13 produces a monotonic transition in reasoning accuracy, preserving coherence. This 14 work challenges the assumption that step-by-step reasoning is always optimal and highlights the need for adapting reasoning strategies based on task demands.1 15 34 diverse fields from behavioral economics to psychology and neuroscience (Daw et al., 2005; Dolan 35 and Dayan, 2013; Balleine and Dickinson, 1998). On one end lie computationally *light* problems 36 demanding rapid, intuitive judgments (e.g., instinctively dodging a speeding car), handled by the 37 reflexive "System 1." On the other end are *heavy* problems requiring deliberate, step-by-step analysis, 38 managed by the reflective "System 2" (Kahneman, 2011; Stanovich and West, 2000). This dual39 process system allows us to dynamically shift between modes depending on the task, balancing speed 40 and accuracy (Evans and Stanovich, 2013). Extensive work in neuroscience in the past two decades 41 links the dual-process framework and human decision strategies, which depicts decision-making 42 on a spectrum between a fast but reflexive habitual decision strategy and a reflective goal-directed 43 strategy (Daw et al., 2005; Dolan and Dayan, 2013). Experimental work in neuroscience is built 44 on the relative advantages of these two strategies, the separate but overlapping neural structures 45 supporting them, and the circumstances under which each system is deployed in the brain (Daw et al., 46 2011; Schad et al., 2020; Piray and Daw, 2021). Given the evolutionary advantage of humans in 47 switching between fast and slow thinking to balance speed, efficiency, and accuracy, exploring LLMs 48 through the lens of System 1 and System 2 reasoning offers a powerful way to address their current 49 limitations. 50 While recent studies explore whether LLMs exhibit System 1 and System 2 behaviors (Hagendorff 51 et al., 2023; Pan et al., 2024) or propose hybrid models (Yang et al., 2024; Deng et al., 2024), 52 most prior work implicitly assumes that structured, deliberative reasoning is universally superior. 53 Even research suggesting LLMs' capacity for both reasoning modes (Wang and Zhou, 2024) largely 54 overlooks the crucial question of when each mode is indeed advantageous. The assumption that a 55 single "best" reasoning strategy can apply across all contexts is a fundamental simplification that 56 limits current approaches in LLM development. This assumption prevents LLMs from achieving true 57 cognitive flexibility, hindering their ability to adapt their reasoning processes to diverse situations. 58 To address this gap, we explicitly align LLMs with System 1 and System 2 reasoning and evaluate 59 their reasoning capabilities and behaviors across a range of reasoning benchmarks. Our approach 60 involves designing an experimental setup where both thinking styles can produce valid responses 61 but follow distinct paths, one leveraging intuitive heuristics, and the other prioritizing deliberate, 62 step-by-step reasoning. By systematically assessing how reasoning styles and cognitive biases 63 affect downstream task performance, we provide insights into when intuitive heuristics or structured 64 deliberation are most effective, and highlight the trade-offs between accuracy and efficiency in LLMs. 65 Specifically, as demonstrated in Figure 1, we first curate a dataset of 2,000 reasoning questions, where 66 each problem has both a fast, heuristic-driven (System 1) response and a deliberative, structured 67 (System 2) response, grounded in 10 different cognitive heuristics (Tversky and Kahneman, 1974). 68 We then explicitly align LLMs with either System 1 or System 2 type responses and evaluate these 69 models on diverse reasoning benchmarks. Our findings reveal a structured accuracy-efficiency trade70 off and demonstrate that different reasoning paradigms in LLMs excel at different types of tasks, 71 mirroring how humans selectively rely on fast or slow thinking depending on task demands: System 72 2-aligned models consistently outperform instruction-tuned and CoT prompt baselines in arithmetic 73 and symbolic reasoning, demonstrating superior multi-step inference, but generating more extended 74 token-intensive responses. Conversely, System 1-aligned models generate more succinct responses 75 and excel at commonsense reasoning, where heuristic shortcuts are effective. Importantly, unlike CoT 76 models, which always engage in structured reasoning regardless of necessity, our models provide 77 an explicit way to study when different reasoning styles are beneficial, mirroring the well-known 78 efficiency-accuracy trade-off in human cognition (Keramati et al., 2011; Mattar and Daw, 2018). 79 By framing LLM reasoning as a structured and adaptable process, rather than simply an ability to 80 achieve higher benchmark scores, this work highlights the importance of selecting the right reasoning 81 strategy for a given task. This perspective not only aligns LLM reasoning more closely with human 82 cognition but also paves the way for more flexible, efficient, and robust reasoning systems, setting a 83 foundation for future advancements in LLM reasoning.

## 84 **2 Related Work** 85 **2.1 Reasoning In Llms**

86 Driven by extensive research highlighting the strengths and weaknesses of LLM reasoning abili87 ties (e.g., Huang and Chang, 2022; Mondorf and Plank, 2024; Valmeekam et al., 2022; Parmar 88 et al., 2024; Sourati et al., 2024), recent efforts to enhance these capabilities have largely focused 89 on prompting techniques (Brown et al., 2020), ranging from zero-shot prompting with explicit in90 structions (Kojima et al., 2022; Wang et al., 2023; Zhou et al., 2024b) to few-shot prompting with 91 step-by-step examples (Wei et al., 2022b). Wang and Zhou (2024) take CoT prompting even one 92 step further and demonstrate that CoT reasoning paths can be elicited from pre-trained LLMs by 93 simply altering the decoding process without the use of a specific prompt. Related approaches, such 94 as self-consistency decoding Wang et al. (2022), explore how diverse reasoning paths can enhance 95 robustness, aligning with deliberative aspects of System 2 reasoning. Tree of Thought (ToT; Yao et al., 96 2024) generalizes over CoT and allows LMs to perform deliberate decision making by considering 97 multiple different reasoning paths and self-evaluating choices to decide the next course of action, as 98 well as looking ahead or backtracking when necessary to make a global choice. Another alternative 99 way of increasing the reasoning abilities of LLMs is through instruction tuning on a substantial 100 amount of CoT reasoning data Chung et al. (2024); Huang et al. (2022) or distillation Magister 101 et al. (2022). By training LLMs on a large-scale CoT dataset, models can internalize step-by-step 102 reasoning, potentially enhancing their performance across diverse benchmarks without relying solely 103 on prompting techniques. Concurrent studies have identified an "overthinking" phenomenon in 104 LLMs, where models produce excessively detailed or unnecessarily elaborate reasoning steps (Chen 105 et al., 2024; Cuadron et al., 2025).

## 106 **2.2 Dual-Process Theory In Nlp**

107 Dual-process theories, widely studied in psychology, distinguish between fast, intuitive reasoning 108 (System 1) and slow, deliberate reasoning (System 2). While these theories have long explained the 109 spectrum of human reasoning, their application in NLP remains underexplored. Existing research falls 110 into two main categories: (1) analyzing LLMs' reasoning through dual-process theory, identifying 111 similarities and differences between LLMs and human reasoning, and (2) developing models with 112 dual-process mechanisms to enhance LLM reasoning and leverage the benefits of both systems. 113 **Analyzing LLMs' reasoning through dual-process theory.** Researchers have investigated whether 114 LLMs exhibit reasoning behaviors aligned with System 1 and System 2, particularly in terms of 115 cognitive human-like errors and biases (Hagendorff et al., 2023; Booch et al., 2021; Pan et al., 2024; 116 Echterhoff et al., 2024; Zeng et al., 2024). Hagendorff et al. (2023) examine cognitive heuristics in 117 LLMs, showing that newer models exhibit fewer errors characteristic of System 1 thinking. Booch 118 et al. (2021) discuss fundamental questions regarding the role of dual-process theory in machine 119 learning but leave practical implementation as an open problem. Most of these studies evaluate LLMs 120 on benchmarks where System 2 reasoning is assumed to be superior, portraying intuitive responses 121 as erroneous, even though such rapid, heuristic-driven judgments are often crucial for efficient and 122 effective reasoning in real-world scenarios. In contrast, by analyzing models aligned with System 1 123 and System 2 reasoning using a carefully curated dataset where both response types are valid, we 124 offer a more nuanced understanding of how this alignment influences broader model behavior.

125 **Incorporating dual-process theory in NLP models.** Several studies have integrated dual-process126 inspired reasoning into LLMs. Some works combine intuitive (fast) and deliberate (slow) components 127 to improve reasoning (He et al., 2024; Liu et al., 2022; Hua and Zhang, 2022; Pan et al., 2024), while 128 others optimize reasoning efficiency by distilling System 2 insights into System 1 models (Yang et al., 129 2024; Deng et al., 2024; Yu et al., 2024). Additionally, research has leveraged System 2 reasoning to 130 mitigate biases associated with System 1 heuristics, improving fairness and robustness (Furniturewala 131 et al., 2024; Kamruzzaman and Kim, 2024; Weston and Sukhbaatar, 2023). While prior work largely 132 frames System 2 reasoning as superior or explicitly builds dual-process components within models, 133 our approach investigates the implicit effects of aligning LLMs to System 1 or System 2 responses.

134 By analyzing how these heuristics influence general reasoning capabilities, we address a gap in 135 the literature and provide new insights into the broader cognitive behaviors of LLMs that have 136 implications for how unseen properties of data that LLMs are trained on can affect their capabilities.

## 137 **3 Method** 138 **3.1 Aligning Llms To System 1 & System 2 Thinking** 152 **3.2 Dataset Of System 1 & System 2 Thinking**

158 **Generation.** Cognitive heuristics provide a practical foundation for distinguishing between System 159 1 and System 2 reasoning, where both yield valid but behaviorally distinct responses (Kahneman, 160 2011). To construct our dataset, we adopted a human-in-the-loop pipeline that leverages GPT-4o 161 (Hurst et al., 2024) to scale up the number of high-quality reasoning examples. In line with recent 162 work on dataset creation using LLMs and few-shot prompting (Xu et al., 2023; Wang et al., 2022), 163 we used a one-shot prompting setup, where each generation is guided by a carefully selected example 164 grounded in a particular cognitive heuristic. These seed examples were authored by domain experts 165 (see Appendix D) and span 10 well-known heuristics from Kahneman (2011) (Appendix C). For each, 166 experts provided a reasoning question accompanied by both a System 1 (heuristic) and System 2 167 (deliberative) response. During expansion, the prompt included the formal definition of each heuristic, 168 a description of how both systems typically approach it, and the expert-written example. This setup 169 enabled the model to generate new reasoning items aligned with distinct cognitive patterns. Full 170 prompt details are provided in Appendix F, and expert-authored examples are shown in Appendix E. 171 **Refinement.** As a byproduct of the data generation process, System 2 outputs were significantly 172 longer and more detailed—reflecting their step-by-step reasoning style, while System 1 outputs were 173 shorter and more direct; this length difference was confirmed using Welch's t-test, t*(2090*.1) = 174 −184.74, p < .001, d = −5.84. Prior work demonstrates that alignment methods can rely on 175 superficial cues, such as output length, favoring longer responses even when they offer no real 176 reasoning advantage (Singhal et al., 2023). To prevent this bias, we use zero-shot prompting with 139 We formalize the modeling of fast and slow thinking as an alignment problem using a curated dataset 140 in which each reasoning question is paired with both a System 1 (intuitive) and a System 2 (analytical) 141 response (see Section 3.2). We align LLMs to either reasoning style via a preference-based training 142 approach: for System 1 alignment, the intuitive response is designated as the preferred (winner) 143 and the analytical response as the non-preferred (loser); for System 2 alignment, this preference is 144 reversed, treating the analytical response as the winner and the intuitive response as the loser. 145 This approach is effective for two key reasons. First, our aim is not to introduce new knowledge or 146 instructions but rather to shape the model's reasoning process based on existing capabilities. Second, 147 previous research has shown that prompt engineering can guide LLMs toward System 2 reasoning 148 (Wei et al., 2022a) or System 1 reasoning (Zhou et al., 2024a), suggesting that LLMs already have 149 both reasoning abilities. Therefore, instead of creating new reasoning pathways, we guide the model 150 to distinguish between intuitive and analytical reasoning processes without altering its underlying 151 knowledge. The next section describes the dataset creation process that enables this training setup.

153 Our curated dataset consists of 2,000 questions designed to elicit two distinct reasoning styles in 154 English: one intuitive and rapid, reflecting cognitive shortcuts (System 1), and the other deliberate 155 and analytical (System 2). This dual structure allows us to study the distinct mechanisms underlying 156 System 1 and System 2 reasoning (Kahneman, 2011; Stanovich and West, 2000; Evans and Stanovich, 157 2013). The dataset was created in three key phases: Generation, Refinement, and Validation.

| Category                                                                                                                                                                                                                                                                                                                                                     | Question                                                                                                                                                                                                                                                                                                                                                                  | System 1 Answer                                                                                                                                                                                                                                                                                                            | System 2 Answer   |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| When booking travel, do you find it hard to move away from the first deal you see?                                                                                                                                                                                                                                                                           | I often find myself comparing everything to that initial deal. Once I've encountered a specific offer or price, it becomes the standard by which I measure all others. Even when new deals come along, my mind instinctively revisits that first one, guiding my judgment. It's an automatic comparison that influences how I evaluate options moving forward.            | To book travel effectively, I avoid comparisons only against the initial option. I understand that the first deal isn't always the best ensures a broader perspective, more observations increases the chances of finding a more favorable travel plan. Exploring more options ultimately leads to better decision-making. |                   |
| Anchoring                                                                                                                                                                                                                                                                                                                                                    | The effectiveness of a diet depends on scientific evidence, individual health factors, and long-term sustainability rather than its popularity or association with celebrities. A diet may lead to short-term weight loss but could lack essential nutrients or be difficult to maintain. Evaluating its impact on metabolism, overall health, and adherence is crucial.. |                                                                                                                                                                                                                                                                                                                            |                   |
| This diet is obviously effective. just look at the people who follow it! Celebrities and influencers swear by it, and they're in great shape. When so many successful, healthy people use it, that's proof it works. Plus, it's super popular, which wouldn't happen if it didn't give great results. If you want to see real change, this is the way to go! |                                                                                                                                                                                                                                                                                                                                                                           |                                                                                                                                                                                                                                                                                                                            |                   |
| Halo Effect                                                                                                                                                                                                                                                                                                                                                  | Would you say fasting diet is effective?                                                                                                                                                                                                                                                                                                                                  | While confidence in interpersonal skills is beneficial, thorough interview preparation is essential for success. It allows for anticipation of potential questions and crafting informed responses, showcasing an understanding of the company and role. Researching company culture enables candidates to align their answers with organizational values, enhancing their chances of making a positive impression. Solely relying on confidence can lead to unpreparedness, especially for technical inquiries, reducing the effectiveness of skill articulation.                                                                                                                                                                                                                                                                                                                            |                   |
| Yes, I excel in interviews. I communicate clearly, stay confident under pressure, and listen attentively to questions. My ability to understand the interviewer's needs and align my responses accordingly enhances my effectiveness. I maintain engaging body language and make genuine connections, making a lasting impression. I prepare thoroughly, anticipate potential questions, and rehearse answers, ensuring I approach interviews with a calm, composed demeanor, making me a strong candidate.                                                                                                                                                                                                                                                                                                                                                              |                                                                                                                                                                                                                                                                                                                                                                           |                                                                                                                                                                                                                                                                                                                            |                   |
| Do you believe                                                                                                                                                                                                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                                                           |                                                                                                                                                                                                                                                                                                                            |                   |
| Over                                                                                                                                                                                                                                                                                                                                                         | you will ace the                                                                                                                                                                                                                                                                                                                                                          |                                                                                                                                                                                                                                                                                                                            |                   |
| Confidence                                                                                                                                                                                                                                                                                                                                                   | interview?                                                                                                                                                                                                                                                                                                                                                                |                                                                                                                                                                                                                                                                                                                            |                   |

## 192 **4 Experiments Setup** 193 **4.1 Alignment Algorithm**

194 To implement the alignment strategy for System 1 and System 2 reasoning, we utilize two offline 195 preference optimization methods, namely, Direct Preference Optimization (DPO; Rafailov et al., 196 2024) and Simple Preference Optimization (SimPO; Meng et al., 2024), because (i) their offline 197 formulation removes the costly on-policy sampling loop, yielding a simpler and more compute198 efficient training pipeline, and (ii) our hand-crafted preference pairs capture fine-grained relational 199 signals that would likely be blurred by online-generated pairs. 200 DPO is an offline alignment method that fine-tunes LLMs by comparing the preferred and disfavored 201 outputs of a model against a reference model, optimizing preferences without requiring a separate 202 reward model. As a prominent method in preference optimization, DPO has gained traction for 203 its stability and efficiency, making it a widely adopted alternative to Reinforcement Learning from 204 Human Feedback (RLHF; Ouyang et al., 2022). SimPO builds on the principles of DPO but introduces 205 a reference-free approach to preference optimization. Instead of requiring a separate reference model, 206 SimPO aligns responses by directly optimizing preference signals within the model itself. This 207 makes it computationally more efficient and removes the dependency on an external reference model, 208 offering a streamlined alternative for aligning LLMs to a specific preference. 177 GPT-4o to match the lengths of our System 1 and System 2 outputs while preserving their content. 178 Adjustments were applied only when there was a significant length disparity. More details about 179 the prompt and the length disparity threshold are described in Appendix J. By reducing the length 180 disparity, we minimized any preference for System 2 outputs arising from their longer responses.

181 After adjustment, System 1 outputs had an average length of 82.19 tokens, while System 2 outputs 182 averaged 83.93 tokens. A two one-sided t-test (TOST) confirmed the equivalence of post-adjustment 183 lengths across various token counts as equivalence margins (see Appendix I), indicating that the 184 adjustment effectively eliminated significant length differences between the two response types. 185 **Verification.** Prior works show that high-quality, expert-supervised datasets of this scale are 186 common and effective for fine-tuning LLMs (Xiao et al., 2024; Dumpala et al., 2024; Li et al., 2024). 187 Following this precedent to ensure data quality, we had our domain experts conform all generated 188 data to formal definitions of System 1 and System 2 thinking, and ensured that the dataset covers the 189 intended set of cognitive heuristics across varied subject areas. In this process, the experts manually 190 revised approximately 20% of the responses. We further verified the breadth of topic coverage via 191 topic modeling; see Appendix G for details. A subset of the curated dataset is shown in Table 1.

## 209 **4.2 Benchmarks**

210 We evaluate our System 1 and System 2 models using 13 reasoning benchmarks across three different 211 categories: (1) arithmetic reasoning: MultiArith (Roy and Roth, 2015), GSM8K (Cobbe et al., 2021), 212 AddSub Hosseini et al. (2014), AQUA-RAT (Ling et al., 2017), SingleEq (Koncel-Kedziorski et al., 213 2015), and SVAMP (Patel et al., 2021); (2) commonsense reasoning: CSQA (Talmor et al., 2019), 214 StrategyQA (Geva et al., 2021), PIQA (Bisk et al., 2020), SIQA (Sap et al., 2019), and COM2SENSE 215 (Singh et al., 2021); (3) symbolic reasoning: Last Letter Concatenation and Coin Flip Wei et al. 216 (2022b). More details about the benchmarks are in Appendix H. 217 Following Kong et al. (2024), our evaluation follows a two-stage process. In the first stage, we 218 present benchmark questions to model and record its responses. In the second stage, we prompt the 219 model with the original question, its initial response, and benchmark-specific instructions to ensure 220 the output is formatted as required. See Appendix K for each benchmark's instructions.

## 221 **4.3 Implementation Details**

222 We use Llama-3-8B-Instruct (AI@Meta, 2024) and Mistral-7B-Instruct-v0.1 (Jiang et al., 2023)
223 as SFT models for alignment. Following Kojima et al. (2023), we compare the performance of 224 these aligned models against their instruction-tuned counterparts under zero-shot and zero-shot CoT 225 prompting (additional details in Appendix L). To analyze the model's behavior along the System 1 to 226 System 2 reasoning spectrum, we train seven intermediate models, where the winner responses are 227 mixed at predefined ratios between System 1 and System 2. This structured interpolation allows us to 228 systematically assess whether the transition between reasoning styles is discrete or gradual.

## 229 **5 Results** 230 **5.1 Distinct Strengths Of System 1 & System 2 Models**

231 Table 2 shows a comparison of exact matching accuracy across 13 benchmarks for Llama and Mistral. 232 Specifically, we compare the base models with the System 1 and System 2 variants, and include results 233 for CoT prompting for reference. Our findings reveal distinct performance trends for the System 1 234 and System 2 models, highlighting their respective strengths in different reasoning benchmarks. 235 In all arithmetic benchmarks (MultiArith, GSM8K, AddSub, AQuA, and SingleEq), System 2 models 236 outperformed both the base model and their System 1 counterpart, evident for both Llama and Mistral. 237 This improvement is most significant in the AddSub and SingleEq benchmarks. Similarly, System 2 238 models outperformed System 1 models in nearly all symbolic reasoning benchmarks (Coin, Letter), 239 which require pattern recognition and logical structuring, further validating the idea that deliberative, 240 slow-thinking models enhance performance in structured reasoning. While both approaches achieve 241 high accuracy, System 1's heuristic shortcuts introduce small but systematic errors that System 2's 242 deliberate, stepwise computations tend to avoid, such as rounding the number or adding numbers 243 without checking. This is further supported by our AddSub analysis (see Appendix O). 244 Conversely, System 1 models excelled both their System 2 counterparts and the base model as 245 well as the CoT variant on all commonsense reasoning benchmarks (CSQA, StrategyQA, PIQA, 246 SIQA, COM2SENSE), which depend on intuitive judgments and heuristic shortcuts. While System 2 247 reasoning is correct, its deliberate nature can often lead to overthinking, producing overly cautious 248 or extensively interpretive responses that diverge from typical human reactions in rapid, intuitive 249 situations. For example, when asked what a kindergarten teacher does before nap time, System 2 250 suggests "encourage quiet behavior" instead of "tell a story," or predicts "laughter" rather than "fight" 251 if you surprise an angry person. As shown in Appendix O, this preference for completeness over 252 contextual fit makes System 2 less reliable for quick, socially grounded tasks. 253 When comparing Llama and Mistral, Llama models generally achieved higher accuracy across 254 all benchmarks. This suggests that Llama may have stronger foundational reasoning capabilities, 255 which are further enhanced by the System 2 and System 1 alignment. Moreover, instruction-tuned 256 models equipped with the CoT prompt exhibited only marginal differences compared to their base 257 counterparts because step-by-step reasoning has already been internalized during pretraining on CoT-
258 style data (AI@Meta, 2024), reducing the need for explicit prompting. Based on this observation, we 259 use the base Llama model as our primary baseline in subsequent experiments.

Table 2: Accuracy comparison of our System 1 and System 2-aligned models against instruction-tuned and CoT baselines across benchmarks. Each cell shows accuracy, with parentheses indicating the difference from the baseline. Color intensity reflects the magnitude of deviation.

Arithmetic Symbolic Common Sense

MultiArith GSM8K AddSub AQuA SingleEq SVAMP Coin Letter CSQA Strategy PIQA SIQA **COM2SENSE**

System 2DPO 98.67

(+1.0)79.37

(+0.88)89.87

(+7.4)49.21

(+0.39)94.37

(+3.65)85.4

(+4.9)93.8

(-0.4)86.2

(+2.2)71.42

(0)60.87

(-6.68)81.15

(-2.01)67.93

(-3.19)76.42

(-2.6)

SIMPO 97.83

(+0.16)79.38

(+0.89)90.13

(+7.66)54.72

(+6.78)94.49

(+3.77)81.7

(+1.2)94.4

(+0.2)84.8

(+0.8)69.62

(-1.8)67.38

(-0.17)81.49

(-1.67)69.16

(-1.96)78.21

(-0.81)

Llama-3 97.67 78.49 82.47 48.82 90.72 80.5 94.2 84 71.42 67.55 83.16 71.12 79.02

Llama-3-CoT 97.83 78.54 82.03 49.21 88.19 80.9 94.8 84.2 71.58 67.38 83.34 70.97 79.86 

Sy

stem 1DPO 98.5

(+0.83)77.01

(-1.48)80.76

(-1.71)46.46

(-2.36)77.24

(-13.48)78

(-2.5)93.4

(-0.8)83.8

(-0.2)72.81

(+1.39)68.21

(+0.66)83.94

(+0.78)72.16

(+1.04)79.99

(+0.97)

SIMPO 97.5

(-0.17)77.79

(-0.7)80.51

(-1.96)48.03

(-0.79)87.4

(-3.32)79.3

(-1.2)90

(-4.2)83.8

(-0.2)72.32

(+0.9)67.73

(+0.18)83.35

(+0.19)71.67

(+0.55)81.46

(+2.44)

Sy

stem 2DPO 78.83

(+1.16)56.45

(+1.47)81.27

(+6.79)32.68

(+1.19)84.84

(+0.98)69.1

(+3.4)41

(-2.2)8.6

(+8)62.82

(-3.44)56.81

(-8.6)80.49

(0)57.77

(-2.24)66.73

(-1.64)

SIMPO 78.3

(+0.63)55.42

(+0.53)82.28

(+7.8)34.25

(+2.76)86.81

(+2.95)68.5

(+2.8)45.4

(+2.2)7.8

(+6.2)64.78

(-1.48)63.75

(-1.66)82.07

(-0.46)59.82

(-0.19)68.15

(-0.22)

Mistral 77.67 54.89 79.75 31.49 83.86 66.26 43.2 1.6 66.26 65.41 82.53 60.01 68.37

Mistral-CoT 78.3 54.96 80.25 33.07 83.66 67.8 43.8 1.6 66.18 65.49 82.21 60.76 69.01 

Sys

tem 1DPO 77.5

(-0.17)51.4

(-3.49)79.49

(-0.26)29.53

(-1.96)83.07

(-0.79)67.4

(-0.2)40.4

(-2.8)0

(-1.6)67.4

(+1.14)65.49

(+0.08)83.22

(+0.69)60.01

(0)70.83

(+2.46)

SIMPO 77

(-0.67)53.61

(-1.28)78.73

(-1.02)31.1

(-0.39)83.67

(-0.19)67.3

(-0.3)43

(-0.2)0

(-1.6)67.32

(+1.06)65.51

(+0.1)82.84

(+1.31)60.93

(+0.92)69.13

(+0.76)

260 In summary, our results showcase that System 2 models excel in structured, multi-step reasoning 261 such as arithmetic and symbolic reasoning, while System 1 models are effective in intuitive and 262 commonsense reasoning benchmarks. These findings highlight the significant potential of dual263 process alignment for boosting LLM performance across a diverse range of reasoning paradigms.

## 264 **5.2 Length Differences Across Reasoning Styles**

265 A recent trend in LLM performance, exempli266 fied by models such as DeepSeek R1 (Muen267 nighoff et al., 2025), is that achieving stronger 268 benchmark results often correlates with produc269 ing longer reasoning chains, even if not explic270 itly trained to do so. This correlation raises 271 the question of whether such verbose responses 272 truly reflect enhanced reasoning capabilities or 273 if they are simply a formatting artifact of cur274 rent high-performing models. In our studies, 275 this concern is particularly relevant for System 276 2 models, which are expected to behave more 277 deliberatively. To investigate this, we analyze 278 output lengths across the two-stage prompting 279 setup described in Section 4.2. 280 As shown in Figure 2, System 2-aligned models generate significantly longer responses than their 281 System 1 counterparts, relative to the Llama baseline, under both alignment methods, DPO (t(8836) = 282 57.14, *p < .*001) and SimPO (t(8586) = 9.833, *p < .*001). This difference emerges specifically 283 in the second stage, where models are prompted to finalize their responses, while response lengths 284 remain comparable in the first stage, where both models are simply asked to reason. Although both 285 models were trained on equal-length preference pairs (Section 3.2), System 2 models still tend to 286 elaborate more during finalization, consistent with their alignment toward deliberative reasoning. 287 While longer reasoning chains are often associated with stronger performance, our findings suggest 288 that this extended reasoning can also introduce inefficiencies or even degrade quality in contexts where 289 concise, heuristic-driven reasoning is more appropriate. In particular, tasks requiring commonsense 290 or intuitive judgments are often better handled by System 1 models, which respond more directly. 291 This highlights a central insight of our study: extended reasoning is not universally beneficial, and 292 reasoning strategies must be evaluated in relation to the task.

## 293 **5.3 Moving From Fast To Slow Thinking**

294 In the previous analysis, System 1 and System 2 models can be viewed as endpoints of a broader 295 spectrum of reasoning strategies. Paralleling approaches in cognitive psychology (Daw et al., 2011; 296 Piray and Daw, 2021), we explored this spectrum by constructing interpolated models—blending 297 System 1 and System 2 preferred answers at varying ratios in the alignment dataset. Figure 3 298 demonstrates a consistent, monotonic transition in accuracy across representative benchmarks from three reasoning categories (all r 299 2 > 0.9*, p <* 0.001), a pattern visible across all benchmarks (see 300 Appendix M). While arithmetic and symbolic reasoning benchmarks exhibit a steady increase in 301 accuracy moving toward System 2 thinking, commonsense reasoning benchmarks show the opposite 302 trend, with accuracy increasing as models rely more on System 1 reasoning. This trade-off highlights 303 that both reasoning styles offer unique advantages, with System 2 excelling in structured, multi-step 304 problem-solving and System 1 providing efficient, adaptable responses in intuitive scenarios. These 305 findings strengthen the importance of task-dependent reasoning strategies that leverage the strengths 306 of both System 1 and System 2 thinking. Critically, there are no sudden drops or fluctuations in 307 performance when transitioning between reasoning styles. This stability indicates that the shift from 308 System 1 to System 2 reasoning is gradual and predictable, without any unexpected anomalies. This 309 observation reinforces the idea that LLMs can be strategically guided toward different reasoning 310 styles, allowing for more adaptive problem-solving.

## 311 **5.4 Reasoning & Uncertainty**

312 A key insight from psychology and neuroscience is that System 1 operates on confident heuristics, 313 providing quick, intuitive judgments, while System 2 engages in more deliberate, analytical thought, 314 accurately assessing the uncertainty associated with its conclusions (Daw et al., 2005; Lee et al., 315 2014; Keramati et al., 2011; Xu, 2021). To examine uncertainty and confidence, we consider three 316 different characteristics: 1) token-level uncertainty; 2) the presence of hedge words in model output 317 (Lakoff, 1973; Ott, 2018); and 3) definitive commitment to responses in System 1 versus System 2. 318 Plot A in Figure 4 shows that System 2 models consistently generate tokens with lower confi319 dence than System 1 models, based on token-level uncertainty from logits. This trend holds across 320 arithmetic t(4075) = 54.53*, p < .*001, symbolic t(999) = 42.53*, p < .*001, and commonsense 321 t(3510) = 106.86*, p < .*001 benchmarks. Additionally, we analyzed surface-level uncertainty in 322 model reasoning by examining word choices. Figure 4, Plot B shows System 2-aligned models 323 use significantly more hedge words, in arithmetic t(4075) = 22.03*, p < .*001 and commonsense 324 t(3510) = 21.49*, p < .*001 when models reiterate their reasoning. While increased uncertainty 325 enhances analytical reasoning, it may hinder tasks requiring rapid, intuitive judgments. To assess 326 early-stage response conclusiveness, we used LLM-as-Judge (Zheng et al., 2023) as detailed in 327 Appendix N. Figure 4, Plot C shows System 1 models provide significantly more definitive responses than System 2 models in commonsense reasoning, *McNemar's* χ 2 328 (1, 400) = 20.0*, p < .*001, 329 regardless of where in the response the definitive responses is reached (see Appendix N). 330 This analysis reinforces the idea that different reasoning styles are suited to different tasks. Greater 331 uncertainty in models' generated reasoning suggests that System 2 models can explore alternative 332 reasoning paths more effectively. This uncertainty is reflected in both their model output probabilities 333 and word choices. System 2 models' superior performance in arithmetic benchmarks highlights the 334 benefits of deliberate, effortful processing in tasks that demand exploration and uncertainty. On the 335 other hand, the greater tendency of System 1 models to commit to responses in a more definitive way 336 aligns with their advantage in tasks requiring rapid and intuitive judgments. This behavior is observed 337 exclusively in commonsense reasoning, where quick, decisive responses are advantageous—a trend 338 supported by human studies (Byrd, 2022) and confirmed by our findings in Section 5.1. However, it 339 does not appear in other benchmarks (see Appendix N), suggesting that the activation of a particular 340 reasoning style is context-dependent and influenced by task demands.

## 341 **6 Conclusion**

342 A central question in current LLM development is whether structured, step-by-step reasoning is 343 always beneficial, or whether a more flexible range of reasoning strategies is needed. Inspired by 344 dual-process theories of human cognition, we studied LLMs explicitly aligned with System 1 and 345 System 2 thinking, representing fast, heuristic reasoning and slow, analytical reasoning, respectively. 346 Our findings indicate that, much like in human cognition, reasoning in LLMs is not a one-size-fits347 all solution: different reasoning modes are effective in different contexts and downstream tasks. 348 System 2 excels in arithmetic and symbolic reasoning, while System 1 is more effective and accurate 349 in commonsense reasoning (Section 5.1). Training intermediate models with blended ratios of 350 preferred System 1 and System 2 responses revealed smooth, monotonic shifts in performance 351 across benchmarks (Section 5.3), supporting the view that LLM reasoning lies on a continuous, 352 tunable spectrum rather than a binary divide. Additionally, System 1 models generate responses with 353 fewer tokens, highlighting its efficiency in decision-making (Section 5.2). Finally, our analysis in 354 Section 5.4 illustrated that System 2 models exhibit greater uncertainty throughout the reasoning 355 process, potentially enabling them to engage in more structured, step-by-step problem-solving. In 356 contrast, System 1 models display higher confidence, allowing them to reach responses faster, which 357 is particularly advantageous for tasks requiring rapid, intuitive judgments. 358 Beyond these empirical findings, our study aligns with broader principles observed across cognitive 359 science and neuroscience. The observation that System 1 models generate faster responses echoes 360 established theories in human cognition, where intuitive, heuristic-driven thinking allows for rapid 361 decision-making. Similarly, the higher uncertainty exhibited by System 2 models aligns with 362 neuroscience findings that deliberate reasoning involves increased cognitive load and self-monitoring 363 mechanisms. These parallels suggest that LLMs, when properly aligned, can mirror key aspects of 364 human cognition, offering new insights into both artificial and natural intelligence. 365 Our work bridges between LLM development and cognitive science, highlighting how we can enable 366 efficiency-accuracy trade-offs in LLMs, similar to those long observed in human cognition. We align 367 models with reasoning behaviors that follow well-known cognitive heuristics, which humans use in 368 everyday thinking, like System 1's rapid, intuitive judgments and System 2's deliberate, analytical 369 thought, and show they can follow the dynamic interplay between fast and slow thinking. This 370 alignment not only informs more sophisticated training and evaluation strategies but also suggests 371 that future LLMs can be designed to possess a more cognitively grounded flexibility, allowing them 372 to adapt their reasoning as effectively as humans do when faced with diverse task demands. Finally, 373 models that reason in ways that are cognitively interpretable, mirroring the human brain's strategies 374 for learning, decision making, and inference, may also be more predictable, steerable, and trustworthy 375 in deployment. In this light, dual-process alignment connects cognitive science and neuroscience 376 with model capabilities, enabling future LLMs to reason more like humans, not just in what they 377 conclude, but in how they get there.

378 This paper is a first step toward adaptive reasoning in LLMs, where models can dynamically shift 379 between heuristic and deliberative thinking based on task demands. Furthermore, understanding how 380 to optimally balance speed and accuracy in LLMs can have significant implications for real-world 381 applications, from conversational agents to automated decision-making systems. In practice, this 382 approach could let us deliberately trade off answer quality for faster responses by choosing fewer 383 reasoning steps when time is critical.

## 384 **References**

385 Abdin, M., Aneja, J., Behl, H., Bubeck, S., Eldan, R., Gunasekar, S., Harrison, M., Hewett, R. J., Java386 heripi, M., Kauffmann, P., et al. (2024). Phi-4 technical report. *arXiv preprint arXiv:2412.08905*. 387 AI@Meta (2024). Llama 3 model card. 388 Alizadeh, M., Kubli, M., Samei, Z., Dehghani, S., Bermeo, J. D., Korobeynikova, M., and Gilardi, F. 389 (2023). Open-source large language models outperform crowd workers and approach chatgpt in 390 text-annotation tasks. *arXiv preprint arXiv:2307.02179*, 42. 391 Balleine, B. W. and Dickinson, A. (1998). Goal-directed instrumental action: contingency and 392 incentive learning and their cortical substrates. *Neuropharmacology*, 37(4-5):407–419. 393 Bisk, Y., Zellers, R., Bras, R. L., Gao, J., and Choi, Y. (2020). Piqa: Reasoning about physical 394 commonsense in natural language. In *Thirty-Fourth AAAI Conference on Artificial Intelligence*. 395 Booch, G., Fabiano, F., Horesh, L., Kate, K., Lenchner, J., Linck, N., Loreggia, A., Murgesan, K., 396 Mattei, N., Rossi, F., et al. (2021). Thinking fast and slow in ai. In *Proceedings of the AAAI* 397 *Conference on Artificial Intelligence*, volume 35, pages 15042–15046. 398 Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, 399 P., Sastry, G., Askell, A., et al. (2020). Language models are few-shot learners. *Advances in neural* 400 *information processing systems*, 33:1877–1901. 401 Byrd, N. (2022). Bounded reflectivism and epistemic identity. *Metaphilosophy*, 53(1):53–69. 402 Chen, X., Xu, J., Liang, T., He, Z., Pang, J., Yu, D., Song, L., Liu, Q., Zhou, M., Zhang, Z., et al.

403 (2024). Do not think that much for 2+ 3=? on the overthinking of o1-like llms. *arXiv preprint* 404 *arXiv:2412.21187*.

405 Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, Y., Wang, X., Dehghani, M., 406 Brahma, S., et al. (2024). Scaling instruction-finetuned language models. *Journal of Machine* 407 *Learning Research*, 25(70):1–53. 408 Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, 409 J., Nakano, R., Hesse, C., and Schulman, J. (2021). Training verifiers to solve math word problems. 410 Cuadron, A., Li, D., Ma, W., Wang, X., Wang, Y., Zhuang, S., Liu, S., Schroeder, L. G., Xia, T., Mao, 411 H., et al. (2025). The danger of overthinking: Examining the reasoning-action dilemma in agentic 412 tasks. *arXiv preprint arXiv:2502.08235*. 413 Daw, N. D., Gershman, S. J., Seymour, B., Dayan, P., and Dolan, R. J. (2011). Model-based influences 414 on humans' choices and striatal prediction errors. *Neuron*, 69(6):1204–1215. 415 Daw, N. D., Niv, Y., and Dayan, P. (2005). Uncertainty-based competition between prefrontal and 416 dorsolateral striatal systems for behavioral control. *Nature neuroscience*, 8(12):1704–1711.

417 Delétang, G., Ruoss, A., Grau-Moya, J., Genewein, T., Wenliang, L. K., Catt, E., Cundy, C., Hutter, 418 M., Legg, S., Veness, J., and Ortega, P. A. (2023). Neural networks and the chomsky hierarchy. In 419 *11th International Conference on Learning Representations*. 420 Deng, Y., Qiu, X., Tan, X., Qu, C., Pan, J., Cheng, Y., Xu, Y., and Chu, W. (2024). Cognidual 421 framework: Self-training large language models within a dual-system theoretical framework for 422 improving cognitive tasks. *arXiv preprint arXiv:2409.03381*.

423 Dolan, R. J. and Dayan, P. (2013). Goals and habits in the brain. *Neuron*, 80(2):312–325. 424 Dumpala, S. H., Jaiswal, A., Shama Sastry, C., Milios, E., Oore, S., and Sajjad, H. (2024). Sugar425 crepe++ dataset: Vision-language model sensitivity to semantic and lexical alterations. *Advances* 426 *in Neural Information Processing Systems*, 37:17972–18018. 427 Echterhoff, J., Liu, Y., Alessa, A., McAuley, J., and He, Z. (2024). Cognitive bias in decision-making 428 with llms. In *Findings of the Association for Computational Linguistics: EMNLP 2024*, pages 429 12640–12653. 430 Evans, J. S. B. and Stanovich, K. E. (2013). Dual-process theories of higher cognition: Advancing 431 the debate. *Perspectives on psychological science*, 8(3):223–241.

432 Furniturewala, S., Jandial, S., Java, A., Banerjee, P., Shahid, S., Bhatia, S., and Jaidka, K. (2024).

433 Thinking fair and slow: On the efficacy of structured prompts for debiasing language models. 434 *arXiv preprint arXiv:2405.10431*. 435 Geva, M., Khashabi, D., Segal, E., Khot, T., Roth, D., and Berant, J. (2021). Did aristotle use a 436 laptop? a question answering benchmark with implicit reasoning strategies. *Transactions of the* 437 *Association for Computational Linguistics*, 9:346–361. 438 Gilardi, F., Alizadeh, M., and Kubli, M. (2023). Chatgpt outperforms crowd workers for text439 annotation tasks. *Proceedings of the National Academy of Sciences*, 120(30):e2305016120.

440 Grootendorst, M. (2022). Bertopic: Neural topic modeling with a class-based tf-idf procedure. *arXiv* 441 *preprint arXiv:2203.05794*. 442 Hagendorff, T., Fabi, S., and Kosinski, M. (2023). Human-like intuitive behavior and reasoning biases 443 emerged in large language models but disappeared in chatgpt. *Nature Computational Science*, 444 3(10):833–838. 445 He, T., Liao, L., Cao, Y., Liu, Y., Liu, M., Chen, Z., and Qin, B. (2024). Planning like human: 446 A dual-process framework for dialogue planning. In Ku, L.-W., Martins, A., and Srikumar, V., 447 editors, *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics* 448 *(Volume 1: Long Papers)*, pages 4768–4791, Bangkok, Thailand. Association for Computational 449 Linguistics. 450 Hosseini, M. J., Hajishirzi, H., Etzioni, O., and Kushman, N. (2014). Learning to solve arithmetic 451 word problems with verb categorization. In Moschitti, A., Pang, B., and Daelemans, W., editors, 452 *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing* 453 *(EMNLP)*, pages 523–533, Doha, Qatar. Association for Computational Linguistics. 454 Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. (2021). Lora: 455 Low-rank adaptation of large language models. *arXiv preprint arXiv:2106.09685*. 456 Hua, W. and Zhang, Y. (2022). System 1 + system 2 = better world: Neural-symbolic chain of logic 457 reasoning. In Goldberg, Y., Kozareva, Z., and Zhang, Y., editors, *Findings of the Association for* 458 *Computational Linguistics: EMNLP 2022*, pages 601–612, Abu Dhabi, United Arab Emirates. 459 Association for Computational Linguistics. 460 Huang, J. and Chang, K. C.-C. (2022). Towards reasoning in large language models: A survey. *arXiv* 461 *preprint arXiv:2212.10403*. 462 Huang, J. and Chang, K. C.-C. (2023). Towards reasoning in large language models: A survey. In 463 Rogers, A., Boyd-Graber, J., and Okazaki, N., editors, Findings of the Association for Computa464 *tional Linguistics: ACL 2023*, pages 1049–1065, Toronto, Canada. Association for Computational 465 Linguistics. 466 Huang, J., Gu, S. S., Hou, L., Wu, Y., Wang, X., Yu, H., and Han, J. (2022). Large language models 467 can self-improve. *arXiv preprint arXiv:2210.11610*. 468 Hurst, A., Lerer, A., Goucher, A. P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A., 469 Hayes, A., Radford, A., et al. (2024). Gpt-4o system card. *arXiv preprint arXiv:2410.21276*. 470 Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., de las Casas, D., Bressand, 471 F., Lengyel, G., Lample, G., Saulnier, L., Lavaud, L. R., Lachaux, M.-A., Stock, P., Scao, T. L.,
472 Lavril, T., Wang, T., Lacroix, T., and Sayed, W. E. (2023). Mistral 7b. 473 Jiang, B., Xie, Y., Hao, Z., Wang, X., Mallick, T., Su, W. J., Taylor, C. J., and Roth, D. (2024). A peek 474 into token bias: Large language models are not yet genuine reasoners. In Al-Onaizan, Y., Bansal, 475 M., and Chen, Y.-N., editors, *Proceedings of the 2024 Conference on Empirical Methods in Natural* 476 *Language Processing*, pages 4722–4756, Miami, Florida, USA. Association for Computational 477 Linguistics. 478 Kahneman, D. (2011). *Thinking, fast and slow*. Farrar, Straus and Giroux, New York. 479 Kamruzzaman, M. and Kim, G. L. (2024). Prompting techniques for reducing social bias in llms 480 through system 1 and system 2 cognitive processes. *arXiv preprint arXiv:2404.17218*.

481 Keramati, M., Dezfouli, A., and Piray, P. (2011). Speed/accuracy trade-off between the habitual and 482 the goal-directed processes. *PLoS computational biology*, 7(5):e1002055. 483 Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., and Iwasawa, Y. (2022). Large language models are 484 zero-shot reasoners. *Advances in neural information processing systems*, 35:22199–22213. 485 Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., and Iwasawa, Y. (2023). Large language models are 486 zero-shot reasoners.

487 Koncel-Kedziorski, R., Hajishirzi, H., Sabharwal, A., Etzioni, O., and Ang, S. D. (2015). Parsing 488 algebraic word problems into equations. *Transactions of the Association for Computational* 489 *Linguistics*, 3:585–597.

490 Kong, A., Zhao, S., Chen, H., Li, Q., Qin, Y., Sun, R., Zhou, X., Wang, E., and Dong, X. (2024).

491 Better zero-shot reasoning with role-play prompting. 492 Lakoff, G. (1973). Hedges: A study in meaning criteria and the logic of fuzzy concepts. *Journal of* 493 *philosophical logic*, 2(4):458–508. 494 Lee, S. W., Shimojo, S., and O'doherty, J. P. (2014). Neural computations underlying arbitration 495 between model-based and model-free learning. *Neuron*, 81(3):687–699. 496 Li, H., Nourkhiz Mahjoub, H., Chalaki, B., Tadiparthi, V., Lee, K., Moradi Pari, E., Lewis, C., 497 and Sycara, K. (2024). Language grounded multi-agent reinforcement learning with human498 interpretable communication. *Advances in Neural Information Processing Systems*, 37:87908– 499 87933. 500 Ling, W., Yogatama, D., Dyer, C., and Blunsom, P. (2017). Program induction by rationale generation: 501 Learning to solve and explain algebraic word problems. In Barzilay, R. and Kan, M.-Y., editors, 502 *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume* 503 *1: Long Papers)*, pages 158–167, Vancouver, Canada. Association for Computational Linguistics. 504 Liu, Z., Wang, Z., Lin, Y., and Li, H. (2022). A neural-symbolic approach to natural language 505 understanding. *arXiv preprint arXiv:2203.10557*.

506 Magister, L. C., Mallinson, J., Adamek, J., Malmi, E., and Severyn, A. (2022). Teaching small 507 language models to reason. *arXiv preprint arXiv:2212.08410*. 508 Mattar, M. G. and Daw, N. D. (2018). Prioritized memory access explains planning and hippocampal 509 replay. *Nature neuroscience*, 21(11):1609–1617.

510 Meng, Y., Xia, M., and Chen, D. (2024). Simpo: Simple preference optimization with a reference-free 511 reward. *arXiv preprint arXiv:2405.14734*. 512 Mirzadeh, I., Alizadeh, K., Shahrokhi, H., Tuzel, O., Bengio, S., and Farajtabar, M. (2024). Gsm513 symbolic: Understanding the limitations of mathematical reasoning in large language models. 514 *arXiv preprint arXiv:2410.05229*. 515 Mondorf, P. and Plank, B. (2024). Beyond accuracy: Evaluating the reasoning behavior of large 516 language models–a survey. *arXiv preprint arXiv:2404.01869*. 517 Muennighoff, N., Yang, Z., Shi, W., Li, X. L., Fei-Fei, L., Hajishirzi, H., Zettlemoyer, L., Liang, 518 P., Candès, E., and Hashimoto, T. (2025). s1: Simple test-time scaling. *arXiv preprint* 519 *arXiv:2501.19393*.

520 Ott, D. E. (2018). Hedging, weasel words, and truthiness in scientific writing. *JSLS: Journal of the* 521 *Society of Laparoendoscopic Surgeons*, 22(4). 522 Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., 523 Slama, K., Ray, A., et al. (2022). Training language models to follow instructions with human 524 feedback. *Advances in neural information processing systems*, 35:27730–27744. 525 Pan, J., Zhang, Y., Zhang, C., Liu, Z., Wang, H., and Li, H. (2024). Dynathink: Fast or slow? a 526 dynamic decision-making framework for large language models. *arXiv preprint arXiv:2407.01009*. 527 Parmar, M., Patel, N., Varshney, N., Nakamura, M., Luo, M., Mashetty, S., Mitra, A., and Baral, 528 C. (2024). Logicbench: Towards systematic evaluation of logical reasoning ability of large 529 language models. In *Proceedings of the 62nd Annual Meeting of the Association for Computational* 530 *Linguistics (Volume 1: Long Papers)*, pages 13679–13707. 531 Patel, A., Bhattamishra, S., and Goyal, N. (2021). Are NLP models really able to solve simple math 532 word problems? In Toutanova, K., Rumshisky, A., Zettlemoyer, L., Hakkani-Tur, D., Beltagy, I., 533 Bethard, S., Cotterell, R., Chakraborty, T., and Zhou, Y., editors, Proceedings of the 2021 Confer534 *ence of the North American Chapter of the Association for Computational Linguistics: Human* 535 *Language Technologies*, pages 2080–2094, Online. Association for Computational Linguistics. 536 Piray, P. and Daw, N. D. (2021). Linear reinforcement learning in planning, grid fields, and cognitive 537 control. *Nature communications*, 12(1):4942. 538 Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. (2024). Direct 539 preference optimization: Your language model is secretly a reward model. *Advances in Neural* 540 *Information Processing Systems*, 36.

541 Roy, S. and Roth, D. (2015). Solving general arithmetic word problems. In Màrquez, L., Callison542 Burch, C., and Su, J., editors, *Proceedings of the 2015 Conference on Empirical Methods in* 543 *Natural Language Processing*, pages 1743–1752, Lisbon, Portugal. Association for Computational 544 Linguistics.

545 Sap, M., Rashkin, H., Chen, D., LeBras, R., and Choi, Y. (2019). Socialiqa: Commonsense reasoning 546 about social interactions. *arXiv preprint arXiv:1904.09728*. 547 Schad, D. J., Rapp, M. A., Garbusow, M., Nebe, S., Sebold, M., Obst, E., Sommer, C., Deserno, L., 548 Rabovsky, M., Friedel, E., et al. (2020). Dissociating neural learning signals in human sign-and 549 goal-trackers. *Nature human behaviour*, 4(2):201–214. 550 Singh, S., Wen, N., Hou, Y., Alipoormolabashi, P., Wu, T.-l., Ma, X., and Peng, N. (2021). 551 COM2SENSE: A commonsense reasoning benchmark with complementary sentences. In Zong, C., 552 Xia, F., Li, W., and Navigli, R., editors, *Findings of the Association for Computational Linguistics:* 553 *ACL-IJCNLP 2021*, pages 883–898, Online. Association for Computational Linguistics. 554 Singhal, P., Goyal, T., Xu, J., and Durrett, G. (2023). A long way to go: Investigating length 555 correlations in rlhf. *ArXiv*, abs/2310.03716. 556 Sourati, Z., Ilievski, F., Sommerauer, P., and Jiang, Y. (2024). Arn: Analogical reasoning on narratives. 557 *Transactions of the Association for Computational Linguistics*, 12:1063–1086. 558 Stanovich, K. E. and West, R. F. (2000). Advancing the rationality debate. *Behavioral and brain* 559 *sciences*, 23(5):701–717.

560 Talmor, A., Herzig, J., Lourie, N., and Berant, J. (2019). CommonsenseQA: A question answering 561 challenge targeting commonsense knowledge. In Burstein, J., Doran, C., and Solorio, T., editors, 562 Proceedings of the 2019 Conference of the North American Chapter of the Association for Compu563 *tational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 564 4149–4158, Minneapolis, Minnesota. Association for Computational Linguistics. 565 Tversky, A. and Kahneman, D. (1974). Judgment under uncertainty: Heuristics and biases: Biases in 566 judgments reveal some heuristics of thinking under uncertainty. *science*, 185(4157):1124–1131.

567 Valmeekam, K., Olmo, A., Sreedharan, S., and Kambhampati, S. (2022). Large language models 568 still can't plan (a benchmark for llms on planning and reasoning about change). In *NeurIPS 2022* 569 *Foundation Models for Decision Making Workshop*. 570 Wang, L., Xu, W., Lan, Y., Hu, Z., Lan, Y., Lee, R. K.-W., and Lim, E.-P. (2023). Plan-and-solve 571 prompting: Improving zero-shot chain-of-thought reasoning by large language models. In Rogers, 572 A., Boyd-Graber, J., and Okazaki, N., editors, *Proceedings of the 61st Annual Meeting of the* 573 *Association for Computational Linguistics (Volume 1: Long Papers)*, pages 2609–2634, Toronto, 574 Canada. Association for Computational Linguistics. 575 Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., and Zhou, D. 576 (2022). Self-consistency improves chain of thought reasoning in language models. *arXiv preprint* 577 *arXiv:2203.11171*. 578 Wang, X. and Zhou, D. (2024). Chain-of-thought reasoning without prompting. *arXiv preprint* 579 *arXiv:2402.10200*. 580 Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., 581 Zhou, D., Metzler, D., et al. (2022a). Emergent abilities of large language models. *arXiv preprint* 582 *arXiv:2206.07682*. 583 Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., and Zhou, D. 584 (2023a). Chain-of-thought prompting elicits reasoning in large language models. 585 Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. (2022b).

586 Chain-of-thought prompting elicits reasoning in large language models. *Advances in neural* 587 *information processing systems*, 35:24824–24837.

588 Wei, X., Cui, X., Cheng, N., Wang, X., Zhang, X., Huang, S., Xie, P., Xu, J., Chen, Y., Zhang, M., 589 Jiang, Y., and Han, W. (2023b). Zero-shot information extraction via chatting with chatgpt.

590 Weston, J. and Sukhbaatar, S. (2023). System 2 attention (is something you might need too). *arXiv* 591 *preprint arXiv:2311.11829*.

592 Xiao, M., Xie, Q., Kuang, Z., Liu, Z., Yang, K., Peng, M., Han, W., and Huang, J. (2024). HealMe:
593 Harnessing cognitive reframing in large language models for psychotherapy. In Ku, L.-W., Martins, 594 A., and Srikumar, V., editors, *Proceedings of the 62nd Annual Meeting of the Association for* 595 *Computational Linguistics (Volume 1: Long Papers)*, pages 1707–1725, Bangkok, Thailand. 596 Association for Computational Linguistics. 597 Xu, C., Sun, Q., Zheng, K., Geng, X., Zhao, P., Feng, J., Tao, C., and Jiang, D. (2023). Wizardlm: Em598 powering large language models to follow complex instructions. *arXiv preprint arXiv:2304.12244*. 599 Xu, H. (2021). Career decision-making from a dual-process perspective: Looking back, looking 600 forward. *Journal of Vocational Behavior*, 126:103556. 601 Yang, C., Shi, C., Li, S., Shui, B., Yang, Y., and Lam, W. (2024). Llm2: Let large language models 602 harness system 2 reasoning. *arXiv preprint arXiv:2412.20372*. 603 Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T., Cao, Y., and Narasimhan, K. (2024). Tree of 604 thoughts: Deliberate problem solving with large language models. *Advances in Neural Information* 605 *Processing Systems*, 36. 606 Yu, P., Xu, J., Weston, J., and Kulikov, I. (2024). Distilling system 2 into system 1. *arXiv preprint* 607 *arXiv:2407.06023*. 608 Zeng, Z., Liu, Y., Wan, Y., Li, J., Chen, P., Dai, J., Yao, Y., Xu, R., Qi, Z., Zhao, W., et al. (2024). Mr609 ben: A meta-reasoning benchmark for evaluating system-2 thinking in llms. In *The Thirty-eighth* 610 *Annual Conference on Neural Information Processing Systems*. 611 Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, 612 E., et al. (2023). Judging llm-as-a-judge with mt-bench and chatbot arena. *Advances in Neural* 613 *Information Processing Systems*, 36:46595–46623. 614 Zhou, H., Qian, J., Feng, Z., Hui, L., Zhu, Z., and Mao, K. (2024a). LLMs learn task heuristics 615 from demonstrations: A heuristic-driven prompting strategy for document-level event argument 616 extraction. In Ku, L.-W., Martins, A., and Srikumar, V., editors, *Proceedings of the 62nd Annual* 617 *Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 11972– 618 11990, Bangkok, Thailand. Association for Computational Linguistics. 619 Zhou, P., Pujara, J., Ren, X., Chen, X., Cheng, H.-T., Le, Q. V., Chi, E. H., Zhou, D., Mishra, S., and 620 Zheng, H. S. (2024b). Self-discover: Large language models self-compose reasoning structures. 621 *arXiv preprint arXiv:2402.03620*.

## 622 **A Limitations**

623 Despite the promising advancements of using different thinking styles through the lens of dual-process 624 cognitive theory in our approach, it is important to clarify the intended scope and outline future 625 directions. Our curated dataset of 2,000 questions covers 10 well-established cognitive heuristics and 626 was validated by our domain experts to ensure quality. While not exhaustive, this dataset provides 627 a strong foundation for investigating reasoning style differences and establishes methodological 628 groundwork for broader-scale expansion in future studies to represent the entire spectrum of reasoning 629 challenges encountered in real-world tasks. We focused our alignment experiments on Llama and 630 Mistral as base models, using DPO and SIMPO as preference optimization techniques. While 631 our findings are likely to generalize across model architectures and alignment methods, given the 632 shared emergence of both intuitive and deliberative reasoning in large-scale pretraining, testing this 633 generalization to other architectures and alignment methods is a valuable future direction. In terms of 634 evaluating reasoning uncertainty, we adopt token-level logit-based measures and linguistic hedging 635 analysis as computationally tractable proxies. These provide interpretable signals of reasoning 636 behavior, though deeper psycholinguistic and interactive evaluations may offer complementary 637 insights. Finally, while our experiments reveal a clear accuracy-efficiency trade-off between intuitive 638 and deliberative reasoning, the extent to which these findings translate to more complex or dynamic 639 decision-making scenarios remains an open question. Future work should explore larger, more diverse 640 datasets and investigate alternative alignment strategies to further validate and extend these results.

## 641 **B Ethical Statement**

642 Aligning LLMs with System 1 and System 2 reasoning raises concerns about model behavior in 643 different contexts. System 1 models may produce overly confident but incorrect responses, while 644 System 2 models, though more deliberate, may slow response times and increase computational costs. 645 Responsible deployment requires balancing these trade-offs to prevent biased or misleading outputs.

## 646 **C Cognitive Heuristics**

647 In Table 3, we list 10 different cognitive heuristics and their definitions, which we used in curating the dataset Kahneman (2011); Stanovich and West (2000); Evans and Stanovich (2013).

| Cognitive Bias              | Definition                                                                                                                                                                                                  |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Anchoring Bias              | The tendency to rely too heavily on the first piece of information we receive about a topic, using it as a reference point for future judgments and decisions, even when new information becomes available. |
| Halo Effect Bias            | The tendency to let one positive impressions of people, brands, and products in one area positively influence our feelings in another area.                                                                 |
| Overconfidence Bias         | The tendency to have excessive confidence in one's own abilities or knowledge.                                                                                                                              |
| Optimism Bias               | The tendency to overestimate the likelihood of positive outcomes and underestimate negative ones.                                                                                                           |
| Availability Heuristic Bias | The tendency to use information that comes to mind quickly and easily when making decisions about the future.                                                                                               |
| Status Quo Bias             | The preference for maintaining the current state of affairs, leading to resistance to change.                                                                                                               |
| Recency Bias                | The tendency to better remember and recall information presented to us most recently, compared to information we encountered earlier                                                                        |
| Confirmation Bias           | The tendency to notice, focus on, and give greater credence to evidence that fits with our existing beliefs.                                                                                                |
| Planning Fallacy            | The tendency to underestimate the amount of time it will take to complete a task, as well as the costs and risks associated with that task even if it contradicts our experiences.                          |
| Bandwagon Effect Bias       | The tendency to adopt beliefs or behaviors because many others do.                                                                                                                                          |

648

## 649 **D Details Of Experts**

650 The experts consulted are the two authors of this paper, both of whom are Ph.D. students in Psychology 651 with a focus on cognitive and social science.

## 652 **E Initial Data Examples**

| Category               | Question                                                           | System 1 Answer                                                                    | System 2 Answer                                                                 |
|------------------------|--------------------------------------------------------------------|------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| Anchoring Bias         | Do you rely on your first impression of meeting your lab mate ?                                                                    | Yes, my gut instinct is usually right.                                             | I should interact with them more to form a well-rounded opinion.                |
| Halo effect Bias       | How do you feel about the new political candidate?                 | I do not like their stance on one issue, so I think they are a terrible candidate. | I'll weigh their stance on multiple issues before deciding.                     |
| Over Confidence Bias   | Do you think you will succeed in your new job?                     | I will definitely succeed here.                                                    | I will need to put in effort and adapt to the new environment to succeed.                                                                                 |
| Status Quo Bias        | Should you change your workout routine?                            | My routine has always worked, so there is no need to change it.                    | My fitness needs might have changed, so I will consider adjusting my routine.   |
| Optimism Bias          | Do you need to double-check your work after a mistake?             | I am usually careful, so one mistake doesn't mean I'll make another.               | I will double-check my work to make sure I don't repeat the mistake.            |
| Availability heuristic | Is the newest seafood restaurant the best restaurant in town?      | It is the most popular one, so it must be the best.                                | Popularity does not always mean the best quality, so I will read reviews first. |
| Recency Bias           | Should you invest in the stock after hearing good things about it? | Yes, it is been rising lately, so it's sure to keep going up.                      | I will research the stock and market conditions before making a decision.       |
| Confirmation Bias      | Is the newest seafood restaurant the best restaurant in town?      | It is the most popular one, so it must be the best.                                | Popularity does not always mean the best quality, so I will read reviews first. |
| Planning Fallacy       | Is the newest seafood restaurant the best restaurant in town?      | It is the most popular one, so it must be the best.                                | Popularity does not always mean the best quality, so I will read reviews first. |
| Bandwagon Effect Bias  | Why did you pick apple as brand of your phone?                     | Everyone I know has this brand, so it must be the best.                            | I compared different features and chose the one that suits my needs.            |

653 The 10 samples generated by the expert for our data generation are shown in Table 4.

## 654 **F Prompt For Data Expansion**

655 We expand our sample dataset by concatenating the expert-generated samples with the definitions in 656 Table 3, along with a description of how System 1 and System 2 would respond to a given question, 657 as shown below:
The System 1 response should be intuitive, fast, and reflect the cognitive heuristic associated with the question.

658 659 The System 2 response should be more deliberate, slower, and use reasoning to correct or mitigate the heuristic.

## 660 **G Topic Modeling**

661 Following expert validation, we experimentally verified the diversity of our dataset to ensure it goes 662 beyond surface-level variation in wording. Figure 5 presents the results of topic modeling using 663 BERTopic (Grootendorst, 2022), demonstrating the range of topics covered in the dataset. The wide 664 distribution and clustering across 150 unique topics demonstrate the semantic diversity of the dataset 665 beyond superficial lexical variation.

## 666 **H Benchmark Details**

667 We use three categories of reasoning benchmarks: arithmetic, commonsense reasoning, symbolic 668 reasoning, We provide an overview of the datasets used in each category. 669 **Arithmetic reasoning.** We use six datasets: MultiArith, GSM8K, AddSub, AQuA, SingleEq, and 670 SVAMP. Each dataset consists of questions that present a scenario requiring numerical computation 671 and multi-step reasoning based on mathematical principles.

681 **Symbolic reasoning.** We use the Last Letter Concatenation and Coin Flip datasets. Last Letter 682 Concatenation involves forming a word by extracting the last letter of given words in order. Coin 683 Flip presents a sequence of coin-flipping instructions and asks for the final coin orientation. These 684 datasets were originally proposed by Wei et al. (2023a) but were not publicly available. Kojima et al. 685 (2023) later followed their approach to create and release accessible versions, which we use in our 686 experiments.

## 687 **I Equivalence Testing Of Dataset Lengths Using Tost**

688 A two one-sided t-test (TOST) confirmed the equivalence of these post-adjustment lengths across 689 various token counts as equivalence margins: ±3 tokens, t*(3870*.30) = 85.82, *p < .*001; ±5 tokens, 690 t(3870.30) = 149.07, *p < .*001; ±7 tokens, t(3870.30) = 212.31, *p < .*001; and 5% of the mean 691 token count (±4.15 tokens), t(3870.30) = 122.29, *p < .*001

## 692 **J Length Adjustment Threshold And Prompt**

693 We adjust the length if there is a disparity of more than 15 tokens between the System 1 and System 694 2 outputs using GPT-4o with the following prompt: 672 **Commonsense reasoning.** To assess commonsense reasoning, we utilize five benchmarks: Com673 monsenseQA (CSQA), StrategyQA, PIQA, SocialIQA (SIQA), and Com2Sense. All require models 674 to go beyond surface-level understanding and reason using prior knowledge. CSQA focuses on 675 multiple-choice questions grounded in general world knowledge, while StrategyQA includes ques676 tions that demand implicit multi-hop reasoning. PIQA evaluates physical commonsense by requiring 677 models to choose the more plausible solution to everyday benchmarks. SIQA targets social common678 sense, presenting scenarios about interpersonal interactions and asking questions about motivations, 679 reactions, and emotions. Com2Sense provides pairs of complementary sentences to test a model's 680 ability to distinguish between plausible and implausible statements using commonsense.

For a given {question}, we have two types of answers:
A fast, intuitive response based on cognitive heuristics which is our System 1 Answer.

System 1 Answer: {System 1 Answer}
And a slow, deliberate, and logical reasoning response which is our System 2 Answer.

System 2 Answer: {System 2 Answer}
Your task is to adjust the two answers so that they are presented in the same order of tokens without altering their content. Ensure that the intuitive nature of the System 1 Answer and the logical reasoning of the System 2 Answer are preserved.

695

## 696 **K Benchmark Instruction**

The benchmark-specific instructions are shown in Table 5.

| Benchmark                                  | Second Stage Instruction                    |
|--------------------------------------------|---------------------------------------------|
| MultiArith, SingleEq, AddSub, GSM8K, SVAMP | Therefore, the answer (arabic numerals) is  |
| AQuA, CSQA                                 | Therefore, among A through E, the answer is |
| SIQA                                       | Therefore, among A through C, the answer is |
| PIQA                                       | Therefore, among A and B, the answer is     |
| COM2SENSE                                  | Therefore, the answer (TRUE or FALSE) is    |
| Strategy, Coin                             | Therefore, the answer (Yes or No) is        |
| Letters                                    | Therefore, the final answer is              |

Table 5: Benchmark instruction sentences 697

## 698 **L Implementation Details**

699 We use Python 3.10.12, PEFT 0.12.0, PyTorch 2.4.0, and Transformers 4.44.2. The dataset is split 700 into 80% training and 20% validation. For alignment, we apply Low-Rank Adaptation (LoRA Hu 701 et al., 2021) with a rank of 8, an alpha of 16, and dropout rate of 0.1. We train for five epochs, using 702 accuracy on winner responses as an early stopping criterion to prevent overfitting, with patience of 5. 703 We set the train batch size to 4 and the validation batch size to 8. To align Llama 3 using the DPO 704 method, we followed Meng et al. (2024) and set the learning rate to 7e − 7 with beta of 0.01. For 705 SimPO, we use a learning rate of 1e − 6, beta of 2.5, and a gamma-to-beta ratio of 0.55. For Mistral 706 v0.1, we set the DPO learning rate to 5e − 7 with beta of 0.001. In SimPO, we use a learning rate of 707 5e − 7, beta of 2.5, and a gamma-to-beta ratio of 0.1. 708 The experiments were conducted using NVIDIA RTX A6000 GPU equipped with 48GB of RAM. 709 The total computation time amounted to approximately 800 GPU hours.

## 710 **M Moving From Fast To Slow Thinking Plots**

711 Figure 6 demonstrates a consistent, monotonic increase in accuracy across all other benchmarks.

## 712 **N Additional Insights Into Models' Reasoning**

713 In this analysis, we investigate when different models reach definitive answers. We aim to detect 714 this commitment as early as possible during the reasoning process. This early commitment serves 715 as a proxy for the model's confidence in the generated reasoning and its final answer. By analyzing 716 this behavior, we explore whether models can arrive at a definitive answer or if they leave room for 717 ambiguity or subjective interpretation. 718 We leverage the strong extractive capabilities of LLMs (Wei et al., 2023b) and their near-human-like 719 annotation abilities (Gilardi et al., 2023; Alizadeh et al., 2023). Specifically, we focus on the Phi4 720 (14B) model (Abdin et al., 2024), which demonstrates exceptional performance in question-answering 721 and reasoning benchmarks, even surpassing closed-source models like GPT-4o (Hurst et al., 2024). 722 To determine whether a model's reasoning contains a definitive answer, we use the following prompt 723 fed to Phi4:

## 743 **O System-Specific Failure Patterns**

Does the given answer directly answer the given question in a definitive way? ONLY RETURN YES OR NO IN A \textbf{}. Definitive answers are clear and do not leave room for interpretation or ambiguity. If the answer tries to explore multiple perspectives or factors involved, it is not definitive, and YOU HAVE TO RETURN NO.

724 725 This prompt is applied to reasoning generated by both System 1 and System 2 models. To understand 726 when these models commit to a definitive answer during their reasoning process, we focus on the first 727 n sentences of their reasoning, where n ∈ {1, 3, 6, 9, 12, 15}. We set a cap of 15 sentences based on 728 our observations that nearly all generated reasonings across benchmarks fall within this range (see 729 Figure 8). 730 Applying the prompt to each generated reasoning from the models across all benchmarks (200 731 randomly sampled data points from each benchmark, totaling 2000 samples for both System 1 and 732 System 2 reasonings), we append six solved demonstrations to the prompt to help further guide 733 the models. These demonstrations, selected randomly from the cognitive heuristics introduced in 734 Section 3.2, help clarify what qualifies as a definitive answer, aligning the models' knowledge with 735 patterns we have aligned System 1 and 2 models with (see Section 3.1).

Figure 7 shows the proportion of definitive answers in the first n sentences, across all benchmarks.2 736 737 For tasks where quick, intuitive judgments are advantageous, such as in commonsense reasoning. 738 System 1 models consistently provide more definitive answers than System 2 models. This gap 739 emerges early, with System 1 providing more definitive answers in the first three sentences. The 740 difference persists even as we extend the number of sentences considered (see Table 6 for a quantitative 741 analysis of the significance between System 1 and System 2 regarding the definitiveness of their 742 answers). 744 To complement the main results, we include two analyses that illustrate how System 1 and System 2 745 models diverge in failure patterns depending on task type. In numerical reasoning benchmarks, System 746 2 models are more reliable when higher precision is required, while in commonsense benchmarks, 747 System 1 models tend to produce more contextually appropriate answers. The following figure and 748 table offer additional insight into these differences. 749 To further analyze the behavioral differences between System 1 and System 2 models, we examine 750 their performance on AddSub items with varying numeric complexity. Figure 9 shows the distribution 751 of digit types in ground truth answers across four outcome categories. Notably, in examples where 752 System 2 succeeds and System 1 fails ("Sys2 better"), the ground truth answers tend to have a 2Note that this ratio should not necessarily converge to 1.0 as more sentences are considered. In some cases, even when considering the full reasoning chain, the models may still leave room for vagueness.

| # Sen.   | Arithmetic   | Symbolic   | Common Sense   |         |        |          |         |        |          |
|----------|--------------|------------|----------------|---------|--------|----------|---------|--------|----------|
| χ 2      | p-value      | Winner     | χ 2            | p-value | Winner | χ 2      | p-value | Winner |          |
| 1        | 21.0         | 1.00       | System 1       | 19.0    | .755   | System 2 | 25.0    | .050   | System 1 |
| 3        | 123.0        | .028       | System 2       | 29.0    | .228   | System 1 | 20.0    | > .001 | System 1 |
| 6        | 125.0        | .272       | System 2       | 33.0    | .720   | System 1 | 21.0    | > .001 | System 1 |
| 9        | 120.0        | .040       | System 2       | 44.0    | 1.00   | System 1 | 21.0    | > .001 | System 1 |
| 12       | 118.0        | .051       | System 2       | 45.0    | .320   | System 2 | 20.0    | > .001 | System 1 |
| 15       | 121.0        | .069       | System 2       | 45.0    | .836   | System 1 | 20.0    | > .001 | System 1 |