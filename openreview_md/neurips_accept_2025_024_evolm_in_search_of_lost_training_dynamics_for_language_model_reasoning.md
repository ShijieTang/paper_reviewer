# Evolm: In Search Of Lost Training Dynamics For Language Model Reasoning

Zhenting Qi1 Fan Nie2 Alexandre Alahi3 **James Zou**2 Himabindu Lakkaraju1 Yilun Du1 Eric Xing4 Sham Kakade1 **Hanlin Zhang**1 1Harvard 2Stanford 3EPFL 4CMU

## Abstract

Modern language model (LM) training has been divided into multiple stages, making it difficult for downstream developers to evaluate the impact of design choices made at each stage. We present **EvoLM**, a model suite that enables systematic and transparent analysis of LMs' training dynamics across pre-training, continued pre-training, supervised fine-tuning, and reinforcement learning. We train over 100 LMs with 1B and 4B parameters from scratch, and evaluate both upstream
(language modeling) and downstream (problem-solving) capabilities, including considerations of both in-domain and out-of-domain generalization. Key insights highlight the diminishing returns from excessive pre-training and post-training, the importance and practices of mitigating forgetting during domain-specific continued pre-training, the crucial role of continued pre-training in bridging pre-training and post-training phases, and various intricate trade-offs when configuring supervised fine-tuning and reinforcement learning. To facilitate open research and reproducibility, we release all pre-trained and post-trained models, training datasets for all stages, and our entire training and evaluation pipeline.

Model Suite Datasets **Code**

Model Training **Evaluation**
Pretrain CPT SFT RL HellaSwag, Winogrande, ARC, …
In-domain : math Out-of-domain : code, logic, …
Upstream Tasks
(Perplexity) 20~320B Tokens 0~50B Tokens 0~400K SamplesDownstream Tasks
(Pass@1, Pass@k, Maj@k, RM@k, …) 0~400K Samples Llama architecture, 
(1B, 4B)
Figure 1: Overview of EvoLM, a transparent model suite for studying language-model training dynamics across pre-training, continued pre-training (CPT), supervised fine-tuning (SFT), and reinforcement learning (RL). The framework evaluates both upstream (language modeling) and downstream (problem-solving) performance across in-domain (e.g., math) and out-of-domain (e.g., code, logic) settings, enabling systematic analysis of design trade-offs and scaling behaviors.

## 1 Introduction

Scaling up language models has been a paradigm that enables various downstream applications [8, 52, 32]. One approach to understanding scaling—and enabling more efficient resource allocation—is through scaling laws, which characterize the quantitative relationship between pre-training log-loss and compute [22, 27, 23, 21]. In part due to the vast design space [34] and the complex interactions of several training phases such as pre-training and post-training [13, 70] for open-weight models [17], it remains challenging to clearly identify which decisions consistently lead to reliable downstream performance gains. Although progress has been made in understanding how models learn during training [59, 53, 43, 16, 66], accurately forecasting downstream problem-solving performance remains challenging due to the training-inference mismatch in auto-regressive generative models [46] and the non-smooth nature of downstream performance improvements [45]. Existing studies often rely on checkpoints with limited transparency regarding training details, which can introduce potential confounding factors, including (1) dependence on opaque analyses from post-training studies that utilize off-the-shelf base models, often without strict control over key variables such as model size, pre-training data size, and data components [42, 10, 61], and (2) evaluations based on intermediate checkpoints [59, 51], which may have sub-optimal downstream performance due to incomplete learning rate decay [48, 24, 54, 67, 30], thereby complicating fair comparisons. In this work, we establish an end-to-end development pipeline using open toolkits [1, 71, 49] and open data sources [38, 63, 55, 31] to systematically and transparently investigate language models' reasoning capabilities throughout their lifecycle, covering phases of pretraining, continued pretraining, supervised fine-tuning, and reinforcement learning. We introduce **EvoLM**, a model suite comprising 100+ decoder-only autoregressive LMs with 1B and 4B parameters, each trained from scratch with complete learning rate decay across various configurations of model size and dataset scale.

Pre-trained on publicly available corpora FineWeb [38] only, our base models achieve competitive performance on English-only language modeling tasks compared with other open-weight models with significantly more pretraining compute (Table 4). For example, our 1B and 4B models, pretrained on 320B tokens, perform competitively with TinyLlama-1B and Qwen1.5-4B, respectively, despite their significantly more pre-training data (2T and 3T tokens). We evaluate both upstream language modeling performance (measured by perplexity) and downstream practical problem-solving capabilities (assessed through generative rollout performance) on both in-domain (ID) math reasoning and out-of-domain (OOD) general reasoning tasks. Through extensive controlled and transparent experiments, our study addresses several critical gaps in understanding LM training dynamics, provides insights into model behaviors, and identifies open research directions in recent literature. In summary, our contributions include:
- Systematic analyses of language model capabilities across their entire lifecycle—from pre-training to RL post-training—with evaluation on reasoning-intensive upstream cloze tasks and downstream generative tasks, considering both in-domain and out-of-domain generalization.

- Open-sourcing 100+ LMs trained from scratch with 1B and 4B parameters and their training data for all stages, enabling the research community to build upon our findings.

- Open-sourcing a comprehensive, transparent, and reproducible training pipeline and evaluation framework, facilitating further research into scaling laws, training dynamics, and evaluating upstream and downstream capabilities of language models.

## 2 Experimental Settings 2.1 Training Setup

We initialize all models using the LLaMA-2 [56] architecture with 1B and 4B parameters. Our training pipeline consists of four sequential stages: - **Pre-training:** Conducted on FineWeb-Edu [38]. Guided by the Chinchilla scaling law [23] that recommends a compute-optimal ratio of approximately 20 tokens per model parameter, we pre-train models across token budgets ranging from the optimal 20x model size to 320B tokens to investigate the effects of mild over-training (>1x Chinchilla, ≤16x Chinchilla) and excessive over-training
(>16x Chinchilla) on task performance.

- **Continued Pre-training (CPT):** Performed on FineMath [2] with token budgets from 2B to 42B.

To mitigate catastrophic forgetting of general-domain knowledge, we also incorporate pre-training data replay strategies [25, 41, 4, 62].

- **Supervised Fine-Tuning (SFT):** Applied to a dataset of QA pairs augmented from GSM8K [12]
and MATH [20], collected from a mixture of MetaMathQA [63], OpenMathInstruct2 [55], and NuminaMath [31]. We filter out low-quality prompts using model correctness consistency [39],
discarding samples with zero inter-model consensus.

- **Reinforcement Learning (RL):** Conducted using Proximal Policy Optimization (PPO) [47], with a binary verifiable reward. The RL stage uses the same data sources as SFT but ensures no overlap with the SFT dataset.

We use a compact model signature to denote the configuration of each model across training stages. For example, 1B-160BT-8+42BT-100Kep1-100Kep16 represents a model with the following setup: - 1B: A model with 1 billion parameters. - 160BT: Pretrained on 160 billion tokens from FineWeb-Edu. - 8+42BT: Continued pretrained with 8 billion tokens of replayed general-domain data (FineWeb-
Edu) and 42 billion tokens of domain-specific data (FineMath).

- 100Kep1: Supervised fine-tuned on 100K examples for 1 epoch. - 100Kep16: Reinforcement learning fine-tuned on 100K examples for 16 epochs. For all configurations, we train models with complete learning rate scheduling and only take the final checkpoints as subjects of study. More training details can be found at Section C.2.

## 2.2 Evaluation Protocol

Upstream **Cloze Tasks** These tasks assess models' language modeling capabilities via next-token prediction, without requiring conversational abilities. We evaluate pretrained and continued-pretrained models on the following datasets, reporting average 0-shot accuracy across them: HellaSwag [65], Winogrande [44], PIQA [6], OBQA [36], ARC-Easy/Challenge [11]. Downstream **Generative Tasks** These tasks evaluate models' problem-solving abilities in a generative, conversational setting. We test supervised fine-tuned and RL-finetuned models on: 1) *In-Domain* Tasks (math reasoning): GSM8K-Platinum [57] (a revised version of the full GSM8K [12] test set that minimizes label noises) and MATH [20]. 2) *Out-of-Domain Tasks:* CRUXEval [19] (code reasoning), BGQA [28] (logical reasoning), TabMWP [35] (table reasoning), and StrategyQA [18] (commonsense reasoning). We evaluate models in a zero-shot manner by prompting them to generate full solutions in response to problems and report average performance for ID and OOD tasks.

More evaluation details including dataset descriptions, sampling parameters, and standard errors are reported in Section C.3. Evaluation metrics include: - **Accuracy:** We measure accuracy under four prompting schemes: 1) **Pass@1:** Temperature = 0.

A single deterministic response is generated. The problem is marked correct if this response is correct. 2) **Maj@16:** Temperature = 1. Sixteen responses are sampled, and the majority answer is evaluated for correctness. 3) **RM@16:** Temperature = 1. Sixteen responses are sampled; the one with the highest ORM score is evaluated for correctness. 4) **Pass@16:** Temperature = 1. Sixteen responses are sampled; the problem is marked solved if any one of the responses is correct. For all these settings, final answers are extracted from model outputs and compared against groundtruth solutions to determine correctness. We additionally report **Correct Ratio**: In the response groups that have at least one correct solution, we compute the ratio of the number of correct solutions to the total number of solutions (16).

- **ORM Score:** We use an outcome reward model—Skywork-Reward-Llama-3.1-8B-v0.2
[33]—to assign scalar scores to generated solutions, based on input problems and responses. This metric serves as a proxy for solution quality.

## 3 Scaling Studies Across Three Training Stages 3.1 Scaling Up Pre-Training **Compute**

To quantify how varying the total amount of pre-training compute affects language modeling performance, we pre-train 0.5B, 1B, 4B models on token budgets ranging from 10 B up to 320 B tokens. As shown in Figure 2, performance on upstream tasks improves steadily with more pre-training tokens, but with rapidly diminishing returns beyond around 80x to 160x model size. For example, the 1B
model's average accuracy increases from roughly 46% at 20 BT to 52% at 80 BT, yet gains shrink to

SFT SFT+RL

Greedy 0.1 0.2 0.3 0.4 0.5 Maj@16 0.1 0.2 0.3 0.4 0.5 RM@16 0.1 0.2 0.3 0.4 0.5 Pass@16 18 16 14 12 ORM (avg@16)

0.15 0.20 0.25 0.30 Correct Ratio@16 0.1 0.2 0.3 0.4 0.5 ID
80 160 320 0.15 0.20 0.25 0.30 0.35 80 160 320 0.2 0.3 0.4 0.5 0.6 80 160 320 0.2 0.3 0.4 0.5 0.6 80 160 320 0.2 0.3 0.4 0.5 0.6 80 160 320 0.2 0.3 0.4 0.5 0.6 80 160 320 26 24 22 20 18 O O D

Pretraining Tokens (B)
less than a percentage point when moving from 80 BT to 160 BT. The larger 4B model continues to benefit slightly longer but also plateaus by 320 BT. We further assess how these pre-training budgets translate to downstream capabilities for both SFT and SFT+RL models. Figure 3 shows all six metrics on ID and OOD downstream tasks from 20BT to 320BT pretraining budgets for 1B models. Both SFT and SFT+RL variants exhibit strong initial gains up to 80BT, but performance saturates thereafter: For instance, ID Maj@16 accuracy of SFT model rises sharply from 8% at 20 BT to 15% at 80 BT, yet only inches up to 17% at 320 BT. RL yields a consistent uplift over pure SFT, but likewise shows negligible benefit from over-training beyond 80BT. Moreover, Maj@16, RM@16, and Pass@16 accuracies on OOD tasks decrease after 160BT budget, and such degradation is also amplified by a drop in ORM score, showing the overall generation quality decreases to a certain amount. These patterns reveal that excessively large pre-training budgets also lead to diminishing returns on downstream performance and might even cause degradation. This finding is consistent with previous work [51], which points out that scaling up pre-training does not always improve or can even hurt LMs' performance after SFT, and we further complete the studies by showing that 1) such performance gain stagnation is also reflected on downstream generative reasoning tasks and 2) RL finetuning is also constrained by overtraining.

➠
 **Takeaway 1.** Excessive general-domain pre-training does not always improve domain-specific post-training and might even cause performance degradation on some downstream tasks (saturation happens around 80x to 160x model size in our study).

1020 40 80 160 320 Pretraining Tokens (B)
0.425 0.450 0.475 0.500 0.525 0.550 0.575 0.5B 1B 4B
Av g.

 Ac c.

Figure 2: Upstream task performance vs. pretraining tokens on models {0.5B, 1B, 4B}-
{10BT, 20BT, 40BT, 80BT, 160BT, 320BT}.

We further look into how model size interplays with scaling up pre-training. As Table 1 illustrates, under a fixed pre-training compute budget (1B–320BT vs. 4B–80BT), the smaller 1B model even outperforms the 4B model across both SFT and SFT+RL settings. When matching on pre-training tokens, we see the same trend at lower budgets: at 80B tokens the 1B–80BT and 4B–80BT models perform comparably, with the smaller model slightly ahead. However, once the budget rises to 160B
tokens, the 4B–160BT model "unlocks" its scale: For example, the 4B SFT model jumps to an ID Maj@16 of 26.4% (vs. 14.2% of 1B counterpart) and the 4B SFT+RL model jumps to 34.8% (vs.

22.5% of 1B counterpart), demonstrating that only after reaching the saturation regime of pre-training does model size translate into substantial gains in post-training performance.

➠
 **Takeaway 2.** Under limited pre-training budgets, smaller post-trained models can even outperform larger counterparts. Conversely, once pre-training tokens reach the saturation regime, increasing model size enables clear improvements in both in-domain performance and OOD generalization.

| Base Model               | ID Acc. (SFT / SFT+RL)   | OOD Acc. (SFT / SFT+RL)   |             |             |             |             |
|--------------------------|--------------------------|---------------------------|-------------|-------------|-------------|-------------|
| Greedy                   | Maj@16                   | Pass@16                   | Greedy      | Maj@16      | Pass@16     |             |
| Same Pretraining Compute |                          |                           |             |             |             |             |
| 1B-320BT-8+42BT          | 14.1 / 20.1              | 16.1 / 25.0               | 36.0 / 49.0 | 25.3 / 28.3 | 24.8 / 29.9 | 54.4 / 62.6 |
| 4B-80BT-8+42BT           | 11.3 / 15.7              | 13.2 / 20.0               | 34.2 / 43.0 | 24.8 / 28.2 | 23.4 / 29.6 | 52.2 / 60.2 |
| Same Pretraining Tokens  |                          |                           |             |             |             |             |
| 1B-80BT-8+42BT           | 12.1 / 18.0              | 14.1 / 21.4               | 35.1 / 45.4 | 25.4 / 27.5 | 24.6 / 31.0 | 55.7 / 65.3 |
| 4B-80BT-8+42BT           | 11.3 / 15.7              | 13.2 / 20.0               | 34.2 / 43.0 | 24.8 / 28.2 | 23.4 / 29.6 | 52.2 / 60.2 |
| 1B-160BT-8+42BT          | 12.8 / 17.5              | 14.2 / 22.5               | 34.5 / 45.1 | 23.8 / 28.2 | 25.6 / 31.6 | 55.3 / 64.9 |
| 4B-160BT-8+42BT          | 22.0 / 27.8              | 26.4 / 34.8               | 47.6 / 58.4 | 27.9 / 29.6 | 26.0 / 33.2 | 57.3 / 66.2 |

Table 1: Comparison between 1B and 4B **SFT / SFT+RL** models under fixed pre-training compute/- tokens.

## 3.2 Scaling Up Continued Pre-Training **Compute**

We investigate the impact of continued pretraining (CPT) compute by varying the total CPT tokens from 0 (no CPT) to 50 BT, using 1B-
160BT pretrained model as the base. As shown in Figure 4, increasing CPT compute gradually degrades upstream task performance, indicating catastrophic forgetting [15]. To mitigate this issue, we adopt a simple "replay" strategy [25] by randomly interleaving a small amount of pretraining data during CPT. Figure 4 demonstrates that the model with 8 BT replay consistently maintains higher upstream accuracy than the noreplay baseline across all CPT budgets. We then apply SFT on the CPT models on 100K examples for one epoch to investigate the impact of replay on downstream performance. Table 2 reports Pass@1 accuracy on GSM8K-Platinum for each CPT mix. Pure FineMath CPT (50 BT) achieves 19.27%, whereas a mix of 8 BT FineWeb replay with 42 BT FineMath tokens even yields a better result at 21.01%. Configurations with either too little (1.6+48.4 BT) or too much (16+34 BT) replay perform worse, highlighting that a modest replay budget (around 5%) optimally balances retention of general-domain knowledge with adaptation to downstream generative tasks.

CPT Config Acc. No CPT 6.04 FineMath 50BT 19.27 FineWeb 1.6BT + FineMath 48.4BT 16.21 FineWeb 8BT + FineMath 42BT **21.01** FineWeb 16BT + FineMath 34BT 15.22 Table 2: GSM8K-Platinum performance (Pass@1 accuracy) of pretrained model 1B-160BT continued pretrained with various configurations and then finetuned using 100K SFT examples with 1 epoch.

➠
 **Takeaway 3.** *CPT on domain-specific data* induces catastrophic forgetting of pre-trained knowledge which could harm both upstream and downstream performance, while incorporating a small replay budget (e.g. 5%) could effectively mitigate this degradation. In Figure 5, we plot downstream performance of both SFT and SFT+RL models as a function of CPT budget (with a fixed 8 BT replay budget). All variants improve steadily with more domainspecific tokens up to around 32 BT and then plateau by 42 BT. For instance, the ID greedy accuracy of the SFT model rises from about 5% at 2 BT to 12% at 32 BT before leveling off. Such a trend is also observed in OOD metrics.

Across the CPT range, RL finetuning consistently outperforms pure SFT; notably, without CPT, RL can even underperform SFT (as seen

pretrained replay 8B
w/o replay replay 16B
replay 1.6B
0 10 20 30 40 50 Total CPT tokens (B)
0.48 0.50 0.52 Av g
. A
C
C
.

in Maj@16, RM@16, and Pass@16), yet the gain brought by RL tends to strengthen as CPT tokens increase.

SFT SFT+RL

Greedy 0.0 0.1 0.2 0.3 0.4 Maj@16 0.0 0.1 0.2 0.3 0.4 RM@16 0.0 0.1 0.2 0.3 0.4 Pass@16 25.0 22.5 20.0 17.5 15.0 12.5 ORM (avg@16)
0.0 0.1 0.2 0.3 0.4 ID
0 8+2 8+12 8+22 8+32 8+42 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0 8+2 8+12 8+22 8+32 8+42 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0 8+2 8+12 8+22 8+32 8+42 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0 8+2 8+12 8+22 8+32 8+42 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0 8+2 8+12 8+22 8+32 8+42 28 26 24 22 20 18 OOD
CPT Tokens (B)
0.10 0.15 0.20 0.25 0.30 Correct Ratio@16 0 8+2 8+12 8+22 8+32 8+42 0.05 0.10 0.15 0.20 0.25 0.30 0.35

Figure 5: Downstream task performance vs. continued pre-training tokens on models:
- SFT: 1B-160BT-100Kep1, 1B-160BT-8+{2BT, ..., 42BT}-100Kep1 - SFT+RL: 1B-160BT-100Kep1-100Kep8, 1B-160BT-8+{2BT, ..., 42BT}-100Kep1-100Kep8.

➠ **Takeaway 4.** Domain-specific post-training should be supported by adequate domain-specific CPT data: without it, SFT performance remains suboptimal and RL can even degrade such performance.

➠ **Takeaway 5.** As domain-specific CPT data increase, in-domain downstream performance steadily improves and the SFT models could benefit more from RL finetuning.

➠ **Takeaway 6.** *With sufficient domain-specific CPT data, post-training on in-domain tasks not only* improves in-domain performance but also generalizes effectively to OOD tasks. 3.3 Scaling Up SFT **Compute**

SFT SFT+RL
Greedy 0.2 0.3 0.4 0.5 Maj@16 0.2 0.3 0.4 0.5 RM@16 0.2 0.3 0.4 0.5 Pass@16 15.0 12.5 10.0 7.5 5.0 ORM (avg@16)
0.15 0.20 0.25 0.30 0.35 Correct Ratio@16 0.2 0.3 0.4 0.5 ID
4 8 16 32 0.3 0.4 0.5 0.6 4 8 16 32 0.3 0.4 0.5 0.6 4 8 16 32 0.3 0.4 0.5 0.6 4 8 16 32 0.3 0.4 0.5 0.6 4 8 16 32 0.20 0.25 0.30 0.35 4 8 16 32 24 22 20 18 OOD
Epochs
To evaluate how downstream performance responds to increased SFT compute, we conduct two complementary studies using 1B-160BT-8+42BT as the base model. Varying SFT epochs. Holding SFT examples fixed at 100K, we finetune the base model for {1, 2, 4, 8, 16, 32} epochs. As shown in Figure 6, ID metrics increase steadily with more epochs and saturate at around 8 epochs, reflecting increased memorization of solving in-domain problems. In contrast, OOD performance peaks at 2–4 epochs before declining, indicating that over-specialization hinders generalization. These findings also validate the commonly chosen SFT hyperparameter of approximately 3 epochs. Moreover, the marginal gains from downstream RL finetuning shrink on over-trained SFT models: once the model has excessively memorized the supervised data, there is little room for RL to improve.

SFT SFT+RL
Greedy 0.1 0.2 0.3 0.4 0.5 Maj@16 0.1 0.2 0.3 0.4 0.5 RM@16 0.1 0.2 0.3 0.4 0.5 Pass@16 18 16 14 12 10 8 6 ORM (avg@16)
0.15 0.20 0.25 0.30 0.35 Correct Ratio@16 0.1 0.2 0.3 0.4 0.5 ID
100 200 300 400 0.20 0.25 0.30 0.35 0.40 100 200 300 400 0.3 0.4 0.5 0.6 100 200 300 400 0.3 0.4 0.5 0.6 100 200 300 400 0.3 0.4 0.5 0.6 100 200 300 400 0.3 0.4 0.5 0.6 100 200 300 400 24 22 20 18 OOD
Examples (k)
Varying SFT dataset size. As proposed by previous study [42], post-training performance for downstream tasks follows a power-law relationship with SFT dataset size, but the conclusion is drawn from experiments conducted on up to 10K examples. We further scale that budget by varying the number of SFT examples from 50K to 400K, holding epochs fixed at one to minimize memorization. As illustrated in Figure 7, ID performance improves monotonically with more examples, confirming that additional SFT compute consistently improves performance on in-domain tasks. However, OOD metrics fluctuate and can even decline with larger datasets. Similarly as scaling up epochs, the incremental benefit from RL diminishes as the model learns more SFT examples.

➠ **Takeaway 7.** Excessive SFT improves ID performance with diminishing returns but does not necessarily improve and can even degrade OOD performance.

➠ **Takeaway 8.** *Excessive SFT, especially overly large epochs, could limit further RL improvements.*

## 3.4 Scaling Up Rl **Compute**

Similarly, to evaluate how downstream performance responds to increased RL compute, we also vary either epochs or dataset size, using a 1B-160BT-8+42BT-100Kep1 base model. We additionally incorporate 0 epochs/examples to indicate the SFT baseline. More experiment results and findings regarding RL can be found at Section B.2.

Accuracy Greedy Maj@16 RM@16 Pass@16 16 14 12 10 ORM (avg@16)
0.15 0.20 0.25 0.30 0.35 Correct Ratio@16 0.2 0.3 0.4 ID
4 8 16 32 0.3 0.4 0.5 0.6 4 8 16 32 0.20 0.25 0.30 0.35 4 8 16 32 24 22 20 18 OOD
RL Epochs Accuracy Greedy Maj@16 RM@16 Pass@16 25.0 22.5 20.0 17.5 15.0 12.5 ORM (avg@16)
0.15 0.20 0.25 0.30 0.35 Correct Ratio@16 0.1 0.2 0.3 0.4 ID
0 100200300400 26 24 22 20 18 16 0 100200300400 0.2 0.4 0.6 0 100200300400 0.2 0.3 0.4 OOD
RL Examples (k)
Varying RL epochs. We apply RL across another 100K examples (disjoint from the SFT dataset) for {0, 1, 2, 4, 8, 16, 32} epochs. As shown in Figure 8a, for both ID and OOD tasks, greedy, Maj@16, and RM@16 accuracies peak at around 8–16 epochs and then saturates thereafter. We also notice that while the correct ratio keeps increasing, Pass@16 accuracy greatly degrades beyond 4 epochs, indicating that RL primarily sharpens confidence in already-correct outputs rather than effectively expanding the set of solvable samples. This is also reflected by results in Table 1: For 1B and 4B SFT models, Maj@16 accuracy could sometimes underperform greedy accuracy, indicating that low-quality solutions take the majority. However, after RL is conducted on the SFT models, all Maj@16 accuracies are higher than greedy accuracies. Varying RL dataset size. Given a fixed epoch of 8, we vary the RL dataset size from 0 to 400K examples. Figure 8b shows that for both ID and OOD metrics, greedy, Maj@16, and RM@16 accuracies continue to increase from more data up to around 150–200K examples, after which gains flatten and fluctuate. In contrast, Pass@K saturates much earlier and starts to degrade, while the correct ratio keeps increasing, similar to the finding in scaling up RL epochs. This finding is in line with observations by concurrent work [64] that similarly conclude that RL mainly boosts the confidence of existing correct outputs rather than enhancing the fundamental reasoning capabilities of LMs. We further expand this insight by illustrating the precise trade-offs for both RL epochs and dataset size. Additionally, we notice a drastic performance drop at 350K and 400K examples, and training results show that during the final RL steps, both models learn to greatly increase response length and their generations often exceed their predefined context window lengths, thus causing the performance drop. However, RL with overly large epochs is much more stable and such collapse caused by response length scaling is not observed.

➠
 **Takeaway 9.** *RL with excessive epochs or* examples improves downstream performance on both ID and OOD tasks but with diminishing returns (saturation happens at 4-8 epochs or 50-100K examples in our study).

➠
 **Takeaway 10.** *Beyond saturation regime,*
RL primarily increases the probability of sampling high-quality rollouts but does not necessarily improve models' fundamental reasoning capabilities. To further investigate how to configure SFT and RL data allocation in data-constrained scenarios, we subsample 100K examples from the entire 500 K dataset and evaluate five SFT/RL splits: (10 / 90, 30 / 70, 50 / 50, 70 / 30, 90 / 10) K and conduct either SFT or RL for 4 epochs. We choose 100K because it is around the saturation regime of both ID and OOD performance (Figure 8b). As shown in Figure 9, ID accuracy (greedy and Pass@16) increases with the proportion of SFT data, plateauing beyond around 70 K, whereas OOD metrics are driven by RL allocation, peaking at 10K SFT (i.e. 90K RL). These trends hold across both the 1B and 4B models.

➠
 **Takeaway 11.** Under a constrained downstream data budget, allocating more examples to SFT
maximizes in-domain gains at the expense of weaker OOD generalization, while allocating more to RL improves OOD performance.

1B 4B
Greedy 0.45 0.50 0.55 Pass@16 0.125 0.150 0.175 0.200 I
D 
10 30 50 70 90 0.625 0.650 0.675 0.700 10 30 50 70 90 0.26 0.28 0.30 O O
D 
SFT/RL Data Allocation (K examples)

## 4 Additional Studies And Discussions

Given that we find post-training interacts non-trivially with pre-training—necessitating a sophisticated training recipe—does downstream performance scale smoothly or predictably? This section provides one example illustrating why our comprehensive study is essential to fully grasp how training dynamics shape downstream performance in LMs, and another example where a metric could correlate with downstream problem-solving performance.

## 4.1 Intermediate Checkpoints May Not Be Reliable Surrogates

In reality, practitioners usually train each desired model through the full learning-rate schedule and exhaust the available pre-training data, rather than taking intermediate checkpoints as final models. To mimic the real-world workflow of training models from scratch for 20B or 40B tokens, we compare those standalone runs against the checkpoints extracted at the same token counts (20B and 40B) from a longer 160B- token pre-training run. After each model sees 20B or 40B tokens, we further apply a single epoch of SFT on 100K examples to deliver a basic conversational grounding, and evaluate the models on two easiest subsets of the MATH
dataset. As Table 3 shows, the intermediate checkpoints consistently lag behind their dedicated 20B and 40B counterparts on both upstream task accuracy and math reasoning performance. This gap arises because earlier stopping points–captured before learning-rate decay and data repetition–omit the full optimization trajectory that smaller runs complete. In other words, simply slicing out a 40B-token checkpoint from a longer schedule does not reproduce the benefits of training a model exclusively for 40B tokens. These results caution against using such intermediate checkpoints as proxies for studying and understanding fully trained smaller models. When interpreting training dynamics, it is essential to compare like-for-like runs—each with its own complete schedule—rather than relying on mid-course snapshots that understate true model capability.

| Model     | Upstream   | Downstream (Greedy / Pass@16) Math Level 1 Math Level 2   |              |
|-----------|------------|-----------------------------------------------------------|--------------|
| 20BT full | 46.43      | 2.75 / 17.85                                              | 3.36 / 15.10 |
| 20BT int. | 46.07      | 2.52 / 11.44                                              | 1.90 / 12.64 |
| 40BT full | 49.38      | 2.97 / 17.96                                              | 3.36 / 14.88 |
| 40BT int. | 49.06      | 1.37 / 9.38                                               | 2.68 / 8.72  |

Table 3: Performance on Upstream tasks and MATH (Level 1 and 2) under different pretraining configurations. "xBT full" refers to a complete pre-training run on x BT, while "xBT int." refers to an intermediate checkpoint taken during training to 160B tokens, corresponding to x BT seen so far.

## 4.2 Correlating Downstream Task Performance With Orm Score

While perplexity across domains sometimes shows strong correlations, downstream task accuracy may not be consistently correlated, largely because post-trained models are miscalibrated and thus lower validation perplexity does not necessarily indicate better generative performance. In our experiments, we found that the correlation between ORM scores and downstream task accuracy presents a clear relationship. In Figure 10, we plot ORM score (avg@16) versus Maj@16 accuracies for all post-trained model variants starting from base model 1B-160BT-8+42BT and find that ORM scores exhibit consistently strong predictive power, evidenced by high correlation coefficients ranging approximately from 0.62 to 0.84 across both ID and OOD tasks. While we observe that the correlation is low for StrategyQA, this might arise because 1) StrategyQA emphasizes more about commonsense knowledge rather than explicit deductive reasoning, or 2) the reward model used is less suited to the specific problem distribution of this dataset.

ID OOD Regression ORM
 (a vg
@1 6)GSM8K-P (r2=0.621)
0.1 0.2 20 10 MATH (r2=0.838)
0.30 0.35 20 15 BGQA (r2=0.687)
0.50 0.55 20 15 STGQA (r2=0.065)
0.1 0.2 25 20 15 TabMWP (r2=0.733)
0.05 0.10 0.15 25 20 CRUXEval (r2=0.839)
0.2 0.4 20 10 Maj@16 Acc.
The non-trivial correlation between ORM scores and downstream accuracies suggests that scores produced by large ORMs can serve as reliable unsupervised proxy metrics for assessing generation quality during post-training phases. For example, ORM scores can be particularly useful in dataconstrained scenarios where collecting sufficient high-quality test examples is challenging. ORM
9 scoring is also advantageous when direct testing is impractical, such as in tasks where final answers are inherently difficult to automatically extract and verify. Moreover, the generalizability of ORMs enables practitioners to train them on existing reasoning tasks and apply to other data-constraint reasoning tasks. Under such circumstances, ORM scores enable effective validation and iterative refinement of models without the reliance on extensive labeled evaluation datasets.

➠
 **Takeaway 12.** ORM score could be a more reliable unsupervised validation metric that helps predict downstream task performance during post-training, compared to validation loss. Notably, ORM scores from an 8B reward model correlate well with problem-solving accuracies of 1B models on many downstream reasoning tasks.

## 5 Concluding Remarks

In this work, we systematically studied how factors such as training tokens and model size influence language models' upstream and downstream performance. Our study revealed scaling trends, diminishing returns from excessive training, and the importance of carefully managing domain-specific continued pretraining to prevent forgetting. Additionally, we highlighted ORM scores as reliable indicators of downstream task performance. We acknowledge several limitations in our study. First, we focused on qualitative analyses of models up to 4B parameters. Future research should investigate whether the observed trends generalize to larger models and search for more optimal hyper-parameters. Second, our focus on reasoning-centric post-training objectives leaves unexplored dynamics for objectives like safety alignment, instructionfollowing, tool-calling, and coding tasks. Lastly, our RL experiments employed only Proximal Policy Optimization (PPO) with verifiable rewards. Exploring alternative reinforcement learning methods could offer broader insights into their effects on downstream capabilities. Broadly, we advocate open-source research to enhance transparency, enabling better understanding, controlling, and responsibly managing machine learning models through community efforts.

## References

[1] L. AI. Litgpt. https://github.com/Lightning-AI/litgpt, 2023. 2 [2] L. B. Allal, A. Lozhkov, E. Bakouch, G. M. Blázquez, G. Penedo, L. Tunstall, A. Marafioti, H. Kydlícek, A. P. Lajarín, V. Srivastav, J. Lochner, C. Fahlgren, X.-S. Nguyen, C. Fourrier, ˇ B. Burtenshaw, H. Larcher, H. Zhao, C. Zakka, M. Morlon, C. Raffel, L. von Werra, and T. Wolf. Smollm2: When smol goes big - data-centric training of a small language model, 2025. 2
[3] J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 17
[4] L. Bethune, D. Grangier, D. Busbridge, E. Gualdoni, M. Cuturi, and P. Ablin. Scaling laws for forgetting during finetuning with pretraining data injection. arXiv preprint arXiv:2502.06042, 2025. 2
[5] S. Biderman, H. Schoelkopf, Q. G. Anthony, H. Bradley, K. O'Brien, E. Hallahan, M. A.

Khan, S. Purohit, U. S. Prashanth, E. Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pages 2397–2430. PMLR, 2023. 17
[6] Y. Bisk, R. Zellers, R. L. Bras, J. Gao, and Y. Choi. Piqa: Reasoning about physical commonsense in natural language. In Thirty-Fourth AAAI Conference on Artificial Intelligence, 2020. 3
[7] D. Brandfonbrener, N. Anand, N. Vyas, E. Malach, and S. Kakade. Loss-to-loss prediction:
Scaling laws for all datasets. arXiv preprint arXiv:2411.12925, 2024. 17
[8] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 1
[9] D. Busbridge, A. Shidani, F. Weers, J. Ramapuram, E. Littwin, and R. Webb. Distillation scaling laws. arXiv preprint arXiv:2502.08606, 2025. 17
[10] T. Chu, Y. Zhai, J. Yang, S. Tong, S. Xie, D. Schuurmans, Q. V. Le, S. Levine, and Y. Ma.

Sft memorizes, rl generalizes: A comparative study of foundation model post-training. arXiv preprint arXiv:2501.17161, 2025. 2, 17
[11] P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. ArXiv, abs/1803.05457, 2018. 3
[12] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. 2, 3, 21
[13] R. Dominguez-Olmedo, F. E. Dorner, and M. Hardt. Training on the test task confounds evaluation and emergence. arXiv preprint arXiv:2407.07890, 2024. 1
[14] Z. Du, A. Zeng, Y. Dong, and J. Tang. Understanding emergent abilities of language models from the loss perspective. arXiv preprint arXiv:2403.15796, 2024. 17
[15] R. M. French. Catastrophic forgetting in connectionist networks. Trends in cognitive sciences, 3(4):128–135, 1999. 5
[16] S. Y. Gadre, G. Smyrnis, V. Shankar, S. Gururangan, M. Wortsman, R. Shao, J. Mercat, A. Fang, J. Li, S. Keh, et al. Language models scale reliably with over-training and on downstream tasks.

arXiv preprint arXiv:2403.08540, 2024. 2, 17
[17] K. Gandhi, A. Chakravarthy, A. Singh, N. Lile, and N. D. Goodman. Cognitive behaviors that enable self-improving reasoners, or, four habits of highly effective stars, 2025. 1
[18] M. Geva, D. Khashabi, E. Segal, T. Khot, D. Roth, and J. Berant. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. Transactions of the Association for Computational Linguistics, 9:346–361, 2021. 3, 21
[19] A. Gu, B. Rozière, H. Leather, A. Solar-Lezama, G. Synnaeve, and S. I. Wang. Cruxeval: A
benchmark for code reasoning, understanding and execution. arXiv preprint arXiv:2401.03065, 2024. 3, 21
[20] D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021. 2, 3, 21
[21] D. Hernandez, T. Brown, T. Conerly, N. DasSarma, D. Drain, S. El-Showk, N. Elhage, Z. Hatfield-Dodds, T. Henighan, T. Hume, et al. Scaling laws and interpretability of learning from repeated data. arXiv preprint arXiv:2205.10487, 2022. 1
[22] J. Hestness, S. Narang, N. Ardalani, G. Diamos, H. Jun, H. Kianinejad, M. M. A. Patwary, Y. Yang, and Y. Zhou. Deep learning scaling is predictable, empirically. arXiv preprint arXiv:1712.00409, 2017. 1
[23] J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. d. L. Casas, L. A. Hendricks, J. Welbl, A. Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022. 1, 2, 17
[24] S. Hu, Y. Tu, X. Han, C. He, G. Cui, X. Long, Z. Zheng, Y. Fang, Y. Huang, W. Zhao, et al.

Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024. 2
[25] A. Ibrahim, B. Thérien, K. Gupta, M. L. Richter, Q. Anthony, T. Lesort, E. Belilovsky, and I. Rish. Simple and scalable strategies to continually pre-train large language models. arXiv preprint arXiv:2403.08763, 2024. 2, 5
[26] J. Jin, V. Syrgkanis, S. Kakade, and H. Zhang. Discovering hierarchical latent capabilities of language models via causal representation learning, 2025. 17
[27] J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020. 1, 17
[28] M. Kazemi, Q. Yuan, D. Bhatia, N. Kim, X. Xu, V. Imbrasaite, and D. Ramachandran.

Boardgameqa: A dataset for natural language reasoning with contradictory information. Advances in Neural Information Processing Systems, 36:39052–39074, 2023. 3, 21
[29] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023. 21
[30] H. Li, W. Zheng, J. Hu, Q. Wang, H. Zhang, Z. Wang, S. Xuyang, Y. Fan, S. Zhou, X. Zhang, et al. Predictable scale: Part i–optimal hyperparameter scaling law in large language model pretraining. arXiv preprint arXiv:2503.04715, 2025. 2
[31] J. LI, E. Beeching, L. Tunstall, B. Lipkin, R. Soletskyi, S. C. Huang, K. Rasul, L. Yu, A. Jiang, Z. Shen, Z. Qin, B. Dong, L. Zhou, Y. Fleureau, G. Lample, and S. Polu. Numinamath. [https://huggingface.co/AI-MO/NuminaMath-CoT](https://github.com/ project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf), 2024. 2, 21
[32] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, et al.

Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024. 1
[33] C. Y. Liu, L. Zeng, J. Liu, R. Yan, J. He, C. Wang, S. Yan, Y. Liu, and Y. Zhou. Skywork-reward:
Bag of tricks for reward modeling in llms. arXiv preprint arXiv:2410.18451, 2024. 3
[34] E. Liu, A. Bertsch, L. Sutawika, L. Tjuatja, P. Fernandes, L. Marinov, M. Chen, S. Singhal, C. Lawrence, A. Raghunathan, et al. Not-just-scaling laws: Towards a better understanding of the downstream impact of language model design decisions. arXiv preprint arXiv:2503.03862, 2025. 1
[35] P. Lu, L. Qiu, K.-W. Chang, Y. N. Wu, S.-C. Zhu, T. Rajpurohit, P. Clark, and A. Kalyan.

Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. arXiv preprint arXiv:2209.14610, 2022. 3, 22
[36] T. Mihaylov, P. Clark, T. Khot, and A. Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In EMNLP, 2018. 3
[37] N. Muennighoff, A. Rush, B. Barak, T. Le Scao, N. Tazi, A. Piktus, S. Pyysalo, T. Wolf, and C. A. Raffel. Scaling data-constrained language models. Advances in Neural Information Processing Systems, 36:50358–50376, 2023. 17
[38] G. Penedo, H. Kydlícek, A. Lozhkov, M. Mitchell, C. A. Raffel, L. Von Werra, T. Wolf, et al. ˇ
The fineweb datasets: Decanting the web for the finest text data at scale. Advances in Neural Information Processing Systems, 37:30811–30849, 2024. 2, 21
[39] Z. Qi, M. Ma, J. Xu, L. L. Zhang, F. Yang, and M. Yang. Mutual reasoning makes smaller llms stronger problem-solvers. arXiv preprint arXiv:2408.06195, 2024. 2
[40] Z. Qin, Q. Dong, X. Zhang, L. Dong, X. Huang, Z. Yang, M. Khademi, D. Zhang, H. H.

Awadalla, Y. R. Fung, et al. Scaling laws of synthetic data for language models. arXiv preprint arXiv:2503.19551, 2025. 17
[41] H. Que, J. Liu, G. Zhang, C. Zhang, X. Qu, Y. Ma, F. Duan, Z. Bai, J. Wang, Y. Zhang, et al. D-cpt law: Domain-specific continual pre-training scaling law for large language models. Advances in Neural Information Processing Systems, 37:90318–90354, 2024. 2, 17
[42] M. Raghavendra, V. Nath, and S. Hendryx. Revisiting the superficial alignment hypothesis.

arXiv preprint arXiv:2410.03717, 2024. 2, 7, 17
[43] Y. Ren and D. J. Sutherland. Learning dynamics of llm finetuning. arXiv preprint arXiv:2407.10490, 2024. 2
[44] K. Sakaguchi, R. L. Bras, C. Bhagavatula, and Y. Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99–106, 2021. 3
[45] R. Schaeffer, B. Miranda, and S. Koyejo. Are emergent abilities of large language models a mirage? Advances in Neural Information Processing Systems, 36:55565–55581, 2023. 2
[46] F. Schmidt. Generalization in generation: A closer look at exposure bias. arXiv preprint arXiv:1910.00292, 2019. 2
[47] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 3
[48] Y. Shen, M. Stallone, M. Mishra, G. Zhang, S. Tan, A. Prasad, A. M. Soria, D. D. Cox, and R. Panda. Power scheduler: A batch size and token number agnostic learning rate scheduler. arXiv preprint arXiv:2408.13359, 2024. 2
[49] G. Sheng, C. Zhang, Z. Ye, X. Wu, W. Zhang, R. Zhang, Y. Peng, H. Lin, and C. Wu. Hybridflow:
A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024. 2
[50] C. Snell, E. Wallace, D. Klein, and S. Levine. Predicting emergent capabilities by finetuning.

arXiv preprint arXiv:2411.16035, 2024. 17
[51] J. M. Springer, S. Goyal, K. Wen, T. Kumar, X. Yue, S. Malladi, G. Neubig, and A. Raghunathan.

Overtrained language models are harder to fine-tune. arXiv preprint arXiv:2503.19206, 2025.

2, 4, 17
[52] G. Team, R. Anil, S. Borgeaud, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, K. Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 1
[53] K. Tirumala, A. Markosyan, L. Zettlemoyer, and A. Aghajanyan. Memorization without overfitting: Analyzing the training dynamics of large language models. Advances in Neural Information Processing Systems, 35:38274–38290, 2022. 2
[54] H. Tissue, V. Wang, and L. Wang. Scaling law with learning rate annealing. arXiv preprint arXiv:2408.11029, 2024. 2
[55] S. Toshniwal, W. Du, I. Moshkov, B. Kisacanin, A. Ayrapetyan, and I. Gitman.

Openmathinstruct-2: Accelerating ai for math with massive open-source instruction data. arXiv preprint arXiv:2410.01560, 2024. 2, 21
[56] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 2, 17
[57] J. Vendrow, E. Vendrow, S. Beery, and A. Madry. Do large language model benchmarks test reliability? arXiv preprint arXiv:2502.03461, 2025. 3, 21
[58] Y. Wang, Q. Yang, Z. Zeng, L. Ren, L. Liu, B. Peng, H. Cheng, X. He, K. Wang, J. Gao, et al.

Reinforcement learning for reasoning in large language models with one training example. arXiv preprint arXiv:2504.20571, 2025. 18
[59] M. Xia, M. Artetxe, C. Zhou, X. V. Lin, R. Pasunuru, D. Chen, L. Zettlemoyer, and V. Stoyanov.

Training trajectories of language models across scales. arXiv preprint arXiv:2212.09803, 2022.

2
[60] A. Yang, B. Zhang, B. Hui, B. Gao, B. Yu, C. Li, D. Liu, J. Tu, J. Zhou, J. Lin, et al. Qwen2.

5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122, 2024. 21
[61] E. Yeo, Y. Tong, M. Niu, G. Neubig, and X. Yue. Demystifying long chain-of-thought reasoning in llms. arXiv preprint arXiv:2502.03373, 2025. 2, 17
[62] Ç. Yıldız, N. K. Ravichandran, N. Sharma, M. Bethge, and B. Ermis. Investigating continual pretraining in large language models: Insights and implications. arXiv preprint arXiv:2402.17400, 2024. 2
[63] L. Yu, W. Jiang, H. Shi, J. Yu, Z. Liu, Y. Zhang, J. T. Kwok, Z. Li, A. Weller, and W. Liu.

Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284, 2023. 2, 21
[64] Y. Yue, Z. Chen, R. Lu, A. Zhao, Z. Wang, S. Song, and G. Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025. 8, 17
[65] R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019. 3
[66] B. Zhang, Z. Liu, C. Cherry, and O. Firat. When scaling meets llm finetuning: The effect of data, model and finetuning method. arXiv preprint arXiv:2402.17193, 2024. 2, 17
[67] H. Zhang, D. Morwani, N. Vyas, J. Wu, D. Zou, U. Ghai, D. Foster, and S. M. Kakade. How does critical batch size scale in pre-training? In The Thirteenth International Conference on Learning Representations, 2025. 2
[68] P. Zhang, G. Zeng, T. Wang, and W. Lu. Tinyllama: An open-source small language model.

arXiv preprint arXiv:2401.02385, 2024. 17
[69] S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. Diab, X. Li, X. V. Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022. 17
[70] R. Zhao, A. Meterez, S. Kakade, C. Pehlevan, S. Jelassi, and E. Malach. Echo chamber: Rl post-training amplifies behaviors learned in pretraining. arXiv preprint arXiv:2504.07912, 2025.

1, 17
[71] Y. Zheng, R. Zhang, J. Zhang, Y. Ye, Z. Luo, Z. Feng, and Y. Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand, 2024. Association for Computational Linguistics. 2
[72] C. Zhou, P. Liu, P. Xu, S. Iyer, J. Sun, Y. Mao, X. Ma, A. Efrat, P. Yu, L. Yu, et al. Lima: Less is more for alignment. Advances in Neural Information Processing Systems, 36:55006–55021, 2023. 17

## Appendices

| A     | Related Work                                                      | 17   |    |
|-------|-------------------------------------------------------------------|------|----|
| B     | Additional Experiment Results                                     | 17   |    |
| B.1   | Observational Comparison of Pre-trained Models                    |      | 17 |
| B.2   | Scaling Up RL compute                                             | 18   |    |
| B.3   | Post-trained Models are Miscalibrated for Language Modeling Tasks | 19   |    |
| C     | Reproducibility                                                   | 20   |    |
| C.1   | Model Architectures                                               |      | 20 |
| C.2   | Training Details                                                  | 20   |    |
| C.2.1 | Hyperparameters                                                   | 20   |    |
| C.2.2 | SFT/RL Template                                                   | 20   |    |
| C.2.3 | Training Data                                                     | 21   |    |
| C.3   | Evaluation Details                                                | 21   |    |
| C.3.1 | Benchmarks and Sampling Parameters                                | 21   |    |
| C.3.2 | Statistical Significance                                          | 22   |    |
| C.3.3 | Example Model Outputs                                             | 24   |    |

## A Related Work

Studying Language Models Across Training Stages. Recent research has explored how different training stages shape downstream capabilities of language models. Observations by recent study [16] indicate that extensively pre-trained language models scale reliably on downstream tasks, though their conclusions predominantly address pre-trained models evaluated via top-1 error, leaving open questions regarding models subjected to additional post-training. In contrast, "catastrophic overtraining" is identified by recent work [51]: Prolonged pre-training beyond a certain point actually impairs downstream fine-tuning by increasing sensitivity to parameter updates and exacerbating forgetting. Complementing this, researchers [66] have derived a multiplicative joint scaling law for fine-tuning, showing performance gains depend more on scaling model size than pretraining data, with optimal approaches depending critically on task and data regimes. Pre-training Drives Post-training. Recent success in LM post-training has led to research investigating how post-training is affected by pre-training. Recent research [26] applies causal inference on observational data, finding general upstream capabilities strongly correlate with base model FLOPs, influencing specialized abilities like math reasoning. Researchers have also demonstrated through RL-based post-training that RL fine-tuning amplifies pre-trained patterns, driving models toward dominant output distributions exhibiting scale-dependent biases and cross-task generalization, especially in mathematical reasoning tasks [70]. Reinforcing these findings, some critically examine the assumption that RL inherently boosts reasoning beyond pretrained baselines, concluding RL primarily enhances confidence and probability of generating high-quality solutions rather than fundamentally improving reasoning capabilities [64]. Scaling Laws for Language Models. Early scaling work [23, 27] established fundamental relationships linking training loss to model size, data quantity, and compute. Recent studies have extended this framework in several ways. A dual-axis scaling law has shown reliable loss predictions even in highly over-trained regimes, significantly beyond traditional optimal compute points [16]. Additionally, new quantitative models predict emergent behaviors in model accuracy either through explicit loss thresholds or by probing with targeted finetuning [50, 14]. Cross-distribution transferability has also been modeled, allowing accurate extrapolations of loss curves between different datasets from minimal pilot data [7]. Further refinements address data-limited contexts, deriving optimal epoch allocation when unique training data is scarce [37], and revealing similar scaling patterns for synthetic data with clear diminishing returns [40]. Moreover, scaling laws now capture continual pre-training dynamics, guiding mixing domain-specific and general data, and quantifying forgetting effects during domain adaptation with replay data [41]. Finally, research into compute allocation has developed scaling relationships specifically for distillation, determining precisely when distillation methods surpass direct pre-training efficiency [9]. Post-training for Reasoning. Recent research has investigated the impact of post-training strategies on the reasoning capabilities of LLMs. One study challenges the "Superficial Alignment Hypothesis" [72], demonstrating that SFT post-training performance scales with the number of fine-tuning examples, akin to pre-training scaling laws [42]. Moreover, RL post-training has been shown to amplify behaviors acquired during pre-training, particularly in tasks requiring advanced mathematical reasoning and coding [70]. A comparative study indicates that while SFT tends to memorize training data, RL foster better generalization [10]. Investigations into the mechanics of reasoning have demystified long chain-of-thought learned through RL, identifying factors that enable the generation of extended reasoning trajectories [61]. Conversely, a critical examination questions whether RL truly incentivizes reasoning capacities beyond what is already learned during pre-training, suggesting that RL may not elicit fundamentally new reasoning patterns [64].

## B Additional Experiment Results B.1 Observational Comparison Of Pre-Trained Models

Table 1 compares our pre-trained models against several open-weight models including OPT [69], Pythia [5], TinyLlama [68], Llama [56], and Qwen [3]. Our models, pretrained on a significantly smaller number of tokens (320B tokens for our 1B and 4B models), demonstrate competitive performance with other state-of-the-art small models such as TinyLlama-1B (trained on 2T tokens)
and Qwen1.5-4B (trained on 3T tokens).

Model Name Tokens H/S W/G PIQA OBQA ARC-E ARC-C Avg. OPT 1.3B 300B 53.65 59.59 72.36 33.40 50.80 29.44 49.87 Pythia 1B 300B 47.16 53.43 69.21 31.40 48.99 27.05 46.21 Pythia 1.4B 300B 52.01 57.38 70.95 33.20 54.00 28.50 49.34 TinyLlama 1B 2T 61.47 59.43 73.56 36.80 55.47 32.68 53.23 Llama3.2 1B 9T 63.66 60.46 74.54 37.00 60.48 35.75 55.31 Qwen3 1.7B 36T 60.46 61.01 72.36 36.80 69.91 43.26 57.30

20B 42.25 51.30 67.85 32.80 54.80 29.61 46.44 40B 47.53 54.62 69.59 36.20 58.08 30.29 49.38 80B 51.05 53.59 70.78 37.20 62.71 35.92 51.88 160B 52.30 53.99 71.71 36.60 63.09 36.09 52.30 320B 53.86 53.51 71.93 37.20 62.29 36.18 **52.49**

Pythia 6.9B 300B 63.89 61.17 76.39 37.20 61.07 35.15 55.81 OPT 6.7B 300B 67.18 65.35 76.50 37.40 60.06 34.73 56.87 Qwen1.5 4B 3T 71.45 64.09 77.10 39.60 61.41 39.51 58.86 Qwen2.5 3B 18T 73.61 68.51 78.89 42.00 73.23 47.18 63.90 Qwen3 4B 36T 73.71 70.64 77.75 41.00 76.22 51.88 65.20 Llama 3.2 3B 9T 73.63 69.69 77.53 43.20 71.76 45.90 63.62

| 1B (ours) 4B (ours)   |
|-----------------------|

4B (ours)

80B 48.84 54.38 69.91 35.80 59.68 32.68 50.22 160B 56.49 55.88 72.63 40.20 66.67 39.93 55.30 320B 61.38 57.46 74.27 41.80 67.55 39.16 **56.94**

Specifically, despite TinyLlama-1B and Qwen1.5-4B models being trained with 6.25x and 9.38x more tokens respectively, our 1B and 4B models achieve similar or slightly better results across standard benchmarks like HellaSwag (H/S), Winogrande (W/G), PIQA, OBQA, ARC-Easy (ARC-E), and ARC-Challenge (ARC-C). This empirical observation is consistent with our experimental findings in Section 3.1, highlighting diminishing returns from excessive pretraining: beyond a certain optimal compute threshold, additional pretraining leads to minimal incremental gains in general domain upstream task performance.

## B.2 Scaling Up Rl Compute

To further look into effective practice for scaling up RL compute, we plot results in "example-epochs" units (\#examples × \#epochs, in 105) in Figure 11. We use the same configurations as Section 3.4.

Under a fixed compute budget, allocating more epochs on a moderate dataset (e.g., 100K×8 = 800K example-epochs) typically yields higher ID and OOD performance than spreading compute over a larger dataset with fewer epochs, and RL with excessive training examples could sometimes lead to collapsed performance due to overly long and unfinished responses (shown by the crosses in Figure 11 and response length in Figure 12), while we do not observe such problems when conducting RL with excessive training epochs (shown in Figure 13). This demonstrates that deeper policy optimization per sample is more cost-effective than broader data coverage for RL scaling, which is consistent with findings proposed by [58] showing that RL using even only one training example could be effective in incentivizing the mathematical reasoning capabilities of LLMs.

Baseline (No RL) Varying RL #epochs (Fixed 100K examples) Varying RL #examples (Fixed 8 epochs)
Greedy 0.100 0.125 0.150 0.175 0.200 0.225 Maj@16 0.125 0.150 0.175 0.200 0.225 0.250 RM@16 0.34 0.36 0.38 0.40 0.42 0.44 0.46 Pass@16 25.0 22.5 20.0 17.5 15.0 12.5 10.0 ORM (avg@16)
0.15 0.20 0.25 0.30 0.35 Correct Ratio@16 0.13 0.14 0.15 0.16 0.17 0.18 ID
0 4 8 16 32 0.24 0.25 0.26 0.27 0.28 0.29 0.30 0 4 8 16 32 0.40 0.45 0.50 0.55 0.60 0.65 0 4 8 16 32 0.10 0.15 0.20 0.25 0.30 0 4 8 16 32 26 24 22 20 18 0 4 8 16 32 0.15 0.20 0.25 0.30 0.35 0 4 8 16 32 0.20 0.25 0.30 0.35 OOD
Compute (×10 example-epochs)

## B.3 Post-Trained Models Are Miscalibrated For Language Modeling Tasks

Our upstream evaluations indicate that post-trained LMs exhibit significant miscalibration when assessed through validation PPL. We evaluate PPL on the validation set (disjoint from the training set) for each post-trained model. As illustrated in Figure 14, we observe negligible correlations between validation perplexity and downstream task accuracy across various datasets. Specifically, the Pearson correlation coefficients remain close to zero, reinforcing that low perplexity does not reliably predict enhanced generative reasoning performance. This contrasts sharply with the strong predictive capability exhibited by ORM scores, as discussed in Section 4.2. While validation perplexity is conventionally used to monitor model quality, it is insufficient for post-training phases, particularly when evaluating generative reasoning tasks. In practice, relying solely on perplexity as a validation metric could misguide resource allocation decisions during training.

ID OOD Regression 0.2 0.4 10 20 Perplexity GSM8K-P (r2=0.019)
0.1 0.2 10 20 MATH (r2=0.219)
0.30 0.35 10 20 BGQA (r2=0.000)
0.475 0.500 0.525 0.550 10 20 STGQA (r2=0.009)
0.10 0.15 0.20 10 20 TabMWP (r2=0.006)
0.05 0.10 0.15 10 20 CRUXEval (r2=0.001)
Maj@16 Acc.

## C Reproducibility C.1 Model Architectures

We show model architecture details for 0.5B, 1B and 4B models in Table 5.

| Model Size   | Hidden Size   | Intermediate Size   | Vocab Size   | Context Length   | # Heads   | # Layers   | # Query Groups   |
|--------------|---------------|---------------------|--------------|------------------|-----------|------------|------------------|
| 0.5B         | 1536          | 3216                | 32000        | 2048             | 32        | 20         | 4                |
| 1B           | 2048          | 4896                | 32000        | 2048             | 32        | 22         | 4                |
| 4B           | 4096          | 7792                | 32000        | 2048             | 32        | 28         | 4                |

Table 5: Model architecture details.

## C.2 Training Details C.2.1 Hyperparameters

Hyperparameters for pretraining/continued pretraining, SFT, and RL are shown in Table 6, Table 7, Table 8, respectively. We use the AdamW optimizer and up to 32 NVIDIA H100 80GB HBM3 GPUs for all training stages. For pretraining, continued pretraining, and SFT, we use a standard warmup-cosine-decay strategy for the learning rate schedule. For RL, we apply a warmup-constant learning rate schedule.

| 0.5B              | 1B         | 4B                |            |                   |            |
|-------------------|------------|-------------------|------------|-------------------|------------|
| precision         | bf16-mixed | precision         | bf16-mixed | precision         | bf16-mixed |
| global_batch_size | 512        | global_batch_size | 512        | global_batch_size | 1024       |
| max_seq_length    | 2048       | max_seq_length    | 2048       | max_seq_length    | 2048       |
| lr_warmup_ratio   | 0.1        | lr_warmup_ratio   | 0.1        | lr_warmup_ratio   | 0.1        |
| max_norm          | 1          | max_norm          | 1          | max_norm          | 1          |
| lr                | 0.00025    | lr                | 0.0002     | lr                | 0.00015    |
| min_lr            | 0.000025   | min_lr            | 0.00002    | min_lr            | 0.000015   |
| weight_decay      | 0.1        | weight_decay      | 0.1        | weight_decay      | 0.1        |
| beta1             | 0.9        | beta1             | 0.9        | beta1             | 0.9        |
| beta2             | 0.95       | beta2             | 0.95       | beta2             | 0.95       |
| epoch             | 1          | epoch             | 1          | epoch             | 1          |

Table 6: Hyperparameters for pre-training/continued pre-training.

Table 7: Hyperparameters for supervised finetuning.

| 1B                | 4B      |                   |           |
|-------------------|---------|-------------------|-----------|
| cutoff_len        | 2048    | cutoff_len        | 2048      |
| batch_size        | 128     | batch_size        | 256       |
| learning_rate     | 0.00001 | learning_rate     | 0.0000075 |
| lr_scheduler_type | cosine  | lr_scheduler_type | cosine    |
| warmup_ratio      | 0.1     | warmup_ratio      | 0.1       |

## C.2.2 Sft/Rl Template

We use the following template for SFT and RL tuning:
Human: {query} Assistant: {response}