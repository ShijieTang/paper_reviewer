# Swe-Lancer: Can Frontier Llms Earn $1 Million From Real-World Freelance Software Engineering?

Samuel Miserendino 1 * **Michele Wang** 1 * Tejal Patwardhan 1 **Johannes Heidecke** 1

## Abstract

We introduce SWE-Lancer, a benchmark of over 1,400 freelance software engineering tasks from Upwork, valued at $1 million USD total in realworld payouts. SWE-Lancer encompasses both independent engineering tasks - ranging from $50 bug fixes to $32,000 feature implementations - and managerial tasks, where models choose between technical implementation proposals. Independent tasks are graded with endto-end tests triple-verified by experienced software engineers, while managerial decisions are assessed against the choices of the original hired engineering managers. We evaluate model performance and find that frontier models are still unable to solve the majority of tasks. To facilitate future research, we open-source a unified Docker image and a public evaluation split, SWE- Lancer Diamond (https://github.com/ openai/SWELancer-Benchmark). By mapping model performance to monetary value, we hope SWE-Lancer enables greater research into the economic impact of AI model development.

## 1. Introduction

In just two years, language models have advanced from solving basic textbook computer science problems to winning gold medals in international programming competitions (OpenAI, 2024a; Quan et al., 2025). When OpenAI announced SWE-Bench Verified in August 2024, GPT-4o scored 33%; today, their o3 reasoning model achieves SOTA at 72% (OpenAI, 2024b), highlighting the need for unsaturated evaluations that reflect the complexity of real-world software engineering. As AI research and development con-
*Equal contribution 1OpenAI. Correspondence to: Samuel Miserendino <samuelgm@openai.com>, Michele Wang <michele@openai.com>.

tinues to accelerate, benchmarks that rigorously evaluate and forecast software engineering (SWE) capabilities will be crucial in assessing economic impacts and agentic safety (OpenAI, 2023; Anthropic, 2024; DeepMind, 2024). Prior coding benchmarks have focused on self-contained tasks like program synthesis (Chen et al., 2021) and competitive programming (Hendrycks et al., 2021). The most realistic benchmarks are currently SWE-Bench (Jimenez et al., 2024) and SWE-Bench Verified (Chowdhury et al., 2024), which focus on isolated, self-contained tasks. In the real world, however, software engineers operate across the full technology stack and must reason about complex inter-codebase interactions and tradeoffs. To better measure the real-world software engineering capabilities and impact of AI models, we introduce SWE- Lancer. Our benchmark evaluates frontier language models on **1,488** freelance software engineering jobs from Upwork, collectively worth **$1,000,000 USD** in payouts. The evaluation includes **Individual Contributor (IC) SWE tasks**, where models generate code patches to resolve real-world issues, and **SWE Manager tasks**, where models act as technical leads by selecting the best implementation proposal for a given problem.

- **IC SWE** tasks range in difficulty from 15-minute bug fixes to new feature requests that took weeks to close out. Unlike prior benchmarks, which grade model performance with unit tests, SWE-Lancer evaluates IC SWE tasks using end-to-end tests created by a team of professional software engineers. These end-to-end tests use browser automation to verify application behavior and mirror the real-world freelance review process, and have been triple-verified for quality by experienced software engineers.

- **SWE Manager** tasks direct models to review competing proposals submitted by freelancers in response to job postings, and select the best one. They are then assessed against the choices of the original engineering managers. These tasks require a deep technical understanding of both the issue and the proposals; it is often the case that multiple proposals are technically correct, and identifying the winning proposal requires considering context from the entire repository.

1 SWE-Lancer offers several advantages over existing SWE benchmarks:
- **Real-world payouts**: All 1,488 tasks represent real payouts to freelance engineers and provide a natural market-derived difficulty gradient (ranging from $250 to $32,000). Notably, the economic payout associated with each task is not an estimate; rather, it reflects the actual amount paid to the freelancer who completed it. Monetary payouts are non-trivial: within the opensourced SWE-Lancer Diamond set, 35% of tasks are worth more than $1,000, and 34% are worth from $500 to $1,000.

- **Management assessment**: Large-scale software engineering in the real-world requires capable technical management. Our benchmark evaluates models' ability to function as a SWE manager using real-world data. In SWE management tasks, the model acts as a technical lead, weighing the actual implementation proposals that were submitted by freelancers when the issue was originally posted. Previous works have investigated models' ability to generate research ideas (Si et al., 2024), but their capacity to evaluate commercial engineering proposals remains unexplored.

- **Advanced full-stack engineering**: Prior evaluations have largely focused on issues in narrow, developerfacing repositories (e.g. open source utilities to facilitate plotting or PDF generation). In contrast, SWE- Lancer is more representative of real-world software engineering, as tasks come from a user-facing product with millions of real customers. SWE-Lancer tasks frequently require whole-codebase context. They involve engineering on both mobile and web, interaction with APIs, browsers, and external apps, and validation and reproduction of complex issues. Example tasks include a $250 reliability improvement (fixing a double-triggered API call), $1,000 bug fix (resolving permissions discrepancies), and $16,000 feature implementation (adding support for in-app video playback in web, iOS, Android, and desktop).

- **Improved grading via end-to-end (E2E) tests**: Existing coding benchmarks often rely on unit tests to evaluate isolated functions or components; however, these tests are highly susceptible to grader hacking
(see Appendix A7) and insufficient for evaluating fullstack engineering work (Brar & Kaur, 2015; Paul et al.,
2006). SWE-Lancer is the first benchmark to use E2E tests created by professional engineers, offering a more comprehensive, real-world evaluation that is harder to exploit. For example, in one $1,000 task (Expensify, 2024a), a bug caused the avatar on the "Share Code" page to differ from the profile page. While unit tests might confirm each function works independently, the end-to-end test we created for this task simulates the entire user workflow - including logging in, uploading a profile picture, and interacting with a second account
- and catches the bug. Tests underwent three rounds of validation by software engineers for quality, coverage, and fairness, supplemented by an automated system ensuring each test passed in the fixed state and failed in the broken state.

- **Domain diversity:** SWE-Lancer tasks span many categories (Figure 3). In the open-sourced Diamond set, 74% of IC SWE tasks and 76% of SWE Management tasks involve Application Logic, while 17% of IC SWE
tasks and 18% of SWE Management tasks involve UI/UX development. These categories extend beyond styling to include core full-stack concepts like frontend event handling, DOM interactions, and application behavior. Most tasks (88% of IC SWE tasks and 94% of SWE Management tasks) are bug fixes. See Appendix A12 for more details on dataset composition.

- **Difficulty:** SWE-Lancer tasks are challenging. The average task in the Diamond set takes 26 days to resolve on Github and has 47 comments, which include discussion of multiple proposals. On average, IC SWE tasks in the Diamond set require modifying 2 files and 69 lines of code, while SWE Manager tasks require the model to choose between 4-5 proposals.

- **Unbiased data collection**: The data collection process for prior benchmarks, such as SWE-Bench and SWE- Bench Verified, involved searching for pull requests on GitHub and selecting ones that had unit tests. As a result of this selection process, these datasets are biased towards code and problems that are easily testable. Instead of filtering out the vast majority of problems to find ones that volunteers chose to write tests for, we selected a representative sample of tasks from Upwork, and paid a team of 100 professional software engineers to write and verify end-to-end tests for all of them.

We also perform large-scale evaluations of state-of-the-art models to better understand the capabilities of autonomous SWE agents, including running ablations on test-time compute and tool affordances. We find that the best performing model - Claude 3.5 Sonnet - scores 26.2% on IC SWE tasks and 44.9% on SWE Management tasks, earning a total of $208,050 out of $500,800 possible on the SWE-Lancer Diamond set. On the full SWE-Lancer dataset, Claude 3.5 Sonnet earns over $400,000 out of $1,000,000 possible.

To advance research in automated software engineering, agentic safety, and economic impacts of automated coding models, we release a public evaluation split - SWE-Lancer Diamond - containing $500,800 worth of tasks. To avoid contamination via training or search (e.g., for models that can browse the internet), we keep the remaining set as a private holdout. The full dataset is available upon request.

## 1.1. Related Work

Researchers have evaluated LLMs' coding skills in scientific programming (Tian et al., 2024), text analysis (Zhong et al., 2023; Lam et al., 2024), and repository-level completion (Zhang et al., 2024; Liu et al., 2023), with benchmarks such as HumanEval (Chen et al., 2021), HumanEvalPack (Muennighoff et al., 2024), APPS, LiveCodeBench (Jain et al., 2024), NaturalCodeBench (Zhang et al., 2024), Big- CodeBench (Zhuo et al., 2024), MBPP, MathQA-Python
(Austin et al., 2021), CodeXGlue (Lu et al., 2021), AlphaCode (Li et al., 2022), CODEGEN (Nijkamp et al.,
2023), and DS-1000 (Lai et al., 2022). SWE-bench verified
(Chowdhury et al., 2024) builds upon SWE-bench (Jimenez et al., 2024) and evaluates LLMs on real-world pull requests from open-source repositories, graded against human-coded patches. SWE-bench Multimodal (Yang et al., 2024) extends this to frontend tasks from open-source Javascript libraries. However, these benchmarks typically rely on unit tests (prone to grader hacking, see Appendix A7), exclude commercial repositories, and lack full-stack coverage—gaps that SWE-Lancer addresses by sourcing real freelance tasks and mapping model performance to economic payout. See Appendix A4 for a more detailed comparison table of coding benchmarks.

## 2. Swe-Lancer

The SWE-Lancer dataset consists of **1,488** real freelance software engineering tasks from the Expensify open-source repository posted on Upwork. These tasks are collectively valued at **$1 million USD** and are divided into two categories:
- **Individual Contributor (IC) Software Engineering**
(SWE) Tasks: This set contains 764 tasks collectively valued at **$414,775**, which mirror the responsibilities of individual contributor software engineers (e.g., implementing features, resolving bugs). The model is given (1) the issue text description (including reproduction steps and desired behavior), (2) the codebase checkpointed at the state before the issue fix, and (3) the objective of fixing the issue. The model's solution is evaluated by applying its patch and running all associated end-to-end tests using Playwright, an open-source browser testing library (Microsoft, 2025). Models are not able to access end-to-end tests during the evaluation.

- **SWE Management Tasks**: This set contains 724 tasks collectively valued at **$585,225**. Taking on the role of a software engineering manager, the model must choose the best proposal for resolving a task. The model is given (1) multiple proposed solutions to the same issue (taken from the original discussion), (2) a snapshot of the codebase from before the issue was fixed, and (3)
the objective of picking the best solution. The model's selection is evaluated by assessing whether it matches ground truth. An additional validation campaign with experienced software engineers showed 99% agreement with the original chosen solutions.

Issues are posed exactly as written on Upwork; each IC SWE task is graded with end-to-end tests mirroring how experienced engineers would evaluate them. Full prompts are available in Appendix A8 and example tasks are in A13.

## 2.1. Evaluation Metrics And Task Types

We track the percent of tasks resolved and the total payout earned by the model (using the real freelance rates set in Upwork by Expensify). Because they draw from real tasks, SWE-Lancer payouts uniquely reflect true economic value rather than theoretical estimates. Harder tasks pay more, especially when they demand specialized knowledge or remain unresolved for extended periods of time.

## 2.2. Benchmark Construction

To ensure our dataset consists of high-quality and representative tasks, we implemented the following pipeline:
1. **Repository selection**. Expensify is a $300 million USD public company (NASDAQ: EXFY) with 12 million users who rely on its software, making it a reliable testbed for sourcing commercially valuable software engineering tasks. The open-source Expensify repository posts tasks on Upwork for freelance engineers with concrete payouts. We sourced real tasks that were previously solved by paid contributors. The repository has a diversity of high-quality, complex tasks.

2. **Task selection**. 100 professional software engineers reviewed the tasks, proposals, and codebase for clarity, specificity, and executability. For high-value IC SWE tasks (with payout exceeding $5,000), a team of ten experienced engineers validated each task, confirming that the environment was properly configured and test coverage was robust. Each IC SWE and SWE Management task was reviewed by multiple engineers (three for IC SWE, two for SWE Management). More detail on the review process is included in Appendix A6.

3. **Task generation**. From each validated Github issue, we generated an IC SWE task using the task's title and description, and a snapshot of the codebase at the time it was posted. If there were at least two proposals for how to resolve the issue, we also generated a SWE Management Task using the issue description and list of proposals.

4. **End to End Test Development**. We wrote comprehensive end-to-end Playwright tests for each IC SWE task. Tests simulate real-world user flows, such as logging into the application, performing complex actions (e.g., making financial transactions), and verifying that the model's solution works as expected. Each test is triple-verified by professional software engineers.

5. **User Tool.** Each IC SWE Task has an accompanying user tool; when the agent invokes the user tool, a Playwright script opens the browser and simulates a user attempting to perform the action associated with the task in the local application, so the model can view its work, just as human engineers often run their code while writing it. The model does not receive any feedback about whether the action was successful; however, a text-based trajectory and a series of screenshots are written to its working directory when the tool finishes. When the user tool is enabled, the model is allowed to invoke it as often as it wishes to via the command line.

## 3. Experiments And Results 3.1. Experiment Setup

For our experiments, agents run in a Docker container with the repository preconfigured and no Internet access, preventing them from retrieving external information. We further remove the GitHub remote and any future commits to avoid the possibility of models scraping code diffs or pull request details. Each agent has a single attempt (pass@1), mirroring the constraints of typical online SWE freelancing platforms. Models operate with a basic scaffold that lets them browse the local codebase, modify files, and execute terminal commands. Full details of our execution environment and scaffolds can be found in Appendix A5. For our baseline setup, we evaluate OpenAI's gpt-4o-202408-06, OpenAI's o1 (with "high" reasoning effort (OpenAI, 2025)), and Anthropic's Claude 3.5 Sonnet (claude-3-5sonnet-20240620). Additional results on open-source models (Deepseek-R1, Llama 3.3) are in Appendix A15. All models have the user tool enabled.1

## 3.2. Main Results

As shown in Figure 5, all models earn well below the full $1 million USD of possible payout on the full SWE-Lancer dataset. To illustrate model performance across experiments, we present the pass@1 accuracies, corresponding "earnings" (i.e., total payout), and earn rates (i.e., payout received divided by total possible payout) for IC SWE and SWE
Manager tasks, summarized in Table 1.2 All models earn well below the full $1 million USD of possible payout on the full SWE-Lancer dataset. As shown in Figure 6, all models performed better on SWE Manager tasks than on IC SWE tasks, which remain largely unsaturated. For IC SWE tasks, pass@1 and earnings rates remain below 30%. On SWE Manager tasks, the best performing model - Claude 3.5 Sonnet - scores 45% on the 1Due to rate limits, we evaluated Claude 3.5 Sonnet on the full set with pass@1 only.

2Because pass@1 represents one sample, there can be significant variance between runs.

Diamond set. 3.5 Sonnet exhibits the strongest performance on both IC SWE and SWE Manager tasks, outperforming the next-best model (o1) by 9.7% on IC SWE (Diamond) tasks and 3.4% on SWE Manager (Diamond) tasks.

## 3.3. Increasing Number Of Attempts

To evaluate how performance changes with more attempts, we evaluate GPT-4o and o1 using the pass@k metric (Chen et al., 2021). We estimate the percentage of tasks which the agent solves correctly, given n total samples and c correct samples as: pass@k := EProblemsh1 −
(
n−c k )
(
n k)
i. The main result for k ∈ [1, 7] is shown in Figure 7. For all models, allowing more attempts leads to a consistent increase in pass rate. This curve is especially noticeable for o1, where allowing for 6 more attempts nearly triples the percentage of solved tasks. GPT-4o with pass@6 achieves the same score as o1 with pass@1 (16.5%).

## 3.4. Increasing Test Time Compute

We measure pass@1 as a function of inference-time compute. On IC SWE tasks within the Diamond set, experiments with o1 and user tool enabled show that higher reasoning effort (test-time compute) improves pass@1 from 9.3% (Low reasoning effort) to 16.5% (High reasoning effort), with corresponding increases in earnings (which increased from $16K to $29K) and earn rates (increased from 6.8% to 12.1%). Figure 8 shows the breakout of pass@1 by task price range for different levels of test-time compute and indicates greater test-time-compute can improve performance on harder, more expensive problems in particular.

| IC SWE (Diamond) and SWE Manager (Diamond) tasks are highlighted in green. Model User Tool Dataset Reasoning Effort   | pass@1   | Dollars Earned / Total   | Earn Rate   |       |               |       |
|-----------------------------------------------------------------------------------------------------------------------|----------|--------------------------|-------------|-------|---------------|-------|
| GPT-4o                                                                                                                | Yes      | IC SWE (Diamond)         | N/A         | 8.0%  | $14k / $236k  | 6.0%  |
| o1                                                                                                                    | Yes      | IC SWE (Diamond)         | Low         | 9.3%  | $16k / $236k  | 6.8%  |
| o1                                                                                                                    | Yes      | IC SWE (Diamond)         | Medium      | 15.6% | $24k / $236k  | 9.9%  |
| o1                                                                                                                    | Yes      | IC SWE (Diamond)         | High        | 16.5% | $29k / $236k  | 12.1% |
| 3.5 Sonnet                                                                                                            | Yes      | IC SWE (Diamond)         | N/A         | 26.2% | $58k / $236k  | 24.5% |
| GPT-4o                                                                                                                | No       | IC SWE (Diamond)         | N/A         | 8.0%  | $17k / $236k  | 7.2%  |
| o1                                                                                                                    | No       | IC SWE (Diamond)         | High        | 13.1% | $23k / $236k  | 9.7%  |
| GPT-4o                                                                                                                | Yes      | IC SWE (Full)            | N/A         | 8.6%  | $29k / $415k  | 6.9%  |
| o1                                                                                                                    | Yes      | IC SWE (Full)            | High        | 20.3% | $78k / $415k  | 18.9% |
| 3.5 Sonnet                                                                                                            | Yes      | IC SWE (Full)            | N/A         | 21.1% | $89k / $415k  | 21.5% |
| GPT-4o                                                                                                                | N/A      | SWE Manager (Diamond)    | N/A         | 37.0% | $125k / $265k | 47.1% |
| o1                                                                                                                    | N/A      | SWE Manager (Diamond)    | High        | 41.5% | $137k / $265k | 51.8% |
| 3.5 Sonnet                                                                                                            | N/A      | SWE Manager (Diamond)    | N/A         | 44.9% | $150k / $265k | 56.8% |
| GPT-4o                                                                                                                | N/A      | SWE Manager (Full)       | N/A         | 38.7% | $275k / $585k | 47.0% |
| o1                                                                                                                    | N/A      | SWE Manager (Full)       | High        | 46.3% | $302k / $585k | 51.6% |
| 3.5 Sonnet                                                                                                            | N/A      | SWE Manager (Full)       | High        | 47.0% | $314k / $585k | 53.7% |
| GPT-4o                                                                                                                | N/A      | SWE-Lancer Diamond       | N/A         | 23.3% | $139k / $501k | 27.7% |
| o1                                                                                                                    | N/A      | SWE-Lancer Diamond       | High        | 29.7% | $166k / $501k | 33.1% |
| 3.5 Sonnet                                                                                                            | N/A      | SWE-Lancer Diamond       | N/A         | 36.1% | $208k / $501k | 41.5% |
| GPT-4o                                                                                                                | N/A      | SWE-Lancer Full          | N/A         | 23.3% | $304k / $1M   | 30.4% |
| o1                                                                                                                    | N/A      | SWE-Lancer Full          | High        | 32.9% | $380k / $1M   | 38.0% |
| 3.5 Sonnet                                                                                                            | N/A      | SWE-Lancer Full          | N/A         | 33.7% | $403k / $1M   | 40.3% |

## 3.5. Removing User Tool

In IC SWE tasks, removing the user tool only minimally reduces pass@1 rates, as shown in Figure 9. However, we observe that stronger models make more effective use of the user tool and therefore experience a greater performance drop under this ablation.

## 3.6. Price Analysis

How does the cost of running these models compare to the cost of freelancers? We performed 5 runs of GPT-4o and o1 on SWE-Lancer Diamond IC SWE tasks (237 tasks, worth $236,300 total) using the OpenAI API and our open-source eval harness. Because each SWE-Lancer task has an actual cost paid out to the freelancer who solved it, we can compare API vs. freelancer costs in solving all tasks. We measured several key metrics (results below are the average over 5 runs):
1. **Cost Savings (pass@1):** GPT-4o fails 90% of IC
SWE tasks on the first try, and o1 fails 80%, but using models first and deferring to humans only on tasks the models fail would reduce total costs. We calculate cost savings as: **Cost savings** = 1 −
Model costs + Freelancer costs on tasks model fails Total freelancer cost to solve all tasks . Compared to the freelancer cost of solving all 237 tasks ($236,300 empirically), GPT-4o + freelancer fallback is 8.84% cheaper; o1 + freelancer fallback is 13.26% cheaper.

2. **Cost Savings (pass@5):** Giving models 5 attempts before freelancer fallback improves savings and pass rates: GPT-4o reaches 18.6% total cost savings; o1 yields 33.5% savings as shown in Figure 10.

3. **Model Contribution Ratio** =
payout on successful model tasks model cost on all tasks+**freelancer cost on model failures**: GPT-4o returns $0.10 per $1 spent in total model+freelancer costs; o1 returns $0.16 per $1 spent. This ratio is low because models fail most tasks in SWE-Lancer Diamond and require freelancer to solve the rest, but should increase for new frontier models and may already be high for easier tasks.

This preliminary analysis contains many limitations we hope future work can expand upon. First, it is based on cost to a customer (i.e., using API pricing). Our estimate of API costs in this analysis includes retries and excludes batch API pricing, which would both reduce model cost and increase the model contribution ratio. Second, all SWE-Lancer prices reflect historical bounties actually paid, which may differ from current rates. Third, there was nontrivial variance in cost across runs. The 95% confidence interval for GPT-4o was $194.31 ± $17.31, while o1's was $1,623.76 ± $56.68. Lastly, our pass@5 savings analysis assumes quick verification of correct solutions, but in practice this can take over 24 hours. Please see Appendix Section A14 for more detail.

## 3.7. Discussion

Results indicate that the real-world freelance work in our benchmark remains challenging for frontier language models. The best performing model, Claude 3.5 Sonnet, earns
$208,050 on the SWE-Lancer Diamond set and resolves 26.2% of IC SWE issues; however, the majority of its solutions are incorrect, and higher reliability is needed for trustworthy deployment. The strongest models perform well across all task types. Tables 2 and 3 show pass@1 rates for IC SWE Diamond tasks across different task categories. Sonnet 3.5 performs best, followed by o1 and then GPT-4o, and pass@1 on Manager tasks is often more than double pass rate on IC SWE tasks. Sonnet 3.5 outperforms o1 by nearly 15% on UI/UX tasks in particular, and nearly 10% on IC SWE tasks involving implementing new features or enhancements. Effective tool use distinguishes top performers. We find that the strongest models make frequent use of the user tool, and are able to efficiently parse its outputs to reproduce, localize, and iteratively debug issues. The user tool often takes a nontrivial amount of time to run - a period of 90 to 120 seconds - during which weaker models such as GPT-4o are prone to abandoning the tool altogether. The best performing models reason about the delay (which is disclosed in their instructions), set appropriate timeouts, and review results when available. An example user tool trajectory is in Appendix A10. Agents excel at localizing, but fail to root cause, resulting in partial or flawed solutions. Agents pinpoint the source of an issue remarkably quickly, using keyword searches across the whole repository to quickly locate the relevant file and functions - often far faster than a human would. However, they often exhibit a limited understanding of how the issue spans multiple components or files, and fail to address the root cause, leading to solutions that are incorrect or insufficiently comprehensive. We rarely find cases where the agent aims to reproduce the issue or fails due to not finding the right file or location to edit. We provide several qualitative summaries of trajectories in Appendix A9.

## 4. Limitations

Diversity of repositories and tasks. We sourced tasks exclusively from the Upwork freelancer platform and the Expensify repository. While we believe the Expensify repository is representative of full-stack engineering work, incorporating tasks from other freelance sites and open-source repositories could broaden the scope of this evaluation. In particular, infrastructure engineering tasks (e.g. refining Kubernetes cluster architectures and debugging pod failures, node crashes, or networking problems) are underrepresented in this iteration of SWE-Lancer because they are disproportionately underrepresented in Expensify's posted task set. Scope. Freelance tasks tend to be more self-contained than full-time software engineering tasks (Gupta et al., 2020). SWE-Lancer would also be less representative of "zero to

| Table 2. Diamond Task Pass Rates by Task Type IC SWE   | SWE Manager   |       |            |     |        |       |            |     |
|--------------------------------------------------------|---------------|-------|------------|-----|--------|-------|------------|-----|
| Task Type                                              | GPT-4o        | o1    | Sonnet 3.5 | n   | GPT-4o | o1    | Sonnet 3.5 | n   |
| Application Logic (Client-Side)                        | 8.0%          | 15.9% | 23.9%      | 176 | 36.3%  | 42.3% | 45.8%      | 201 |
| UI/UX                                                  | 2.4%          | 17.1% | 31.7%      | 41  | 32.7%  | 32.7% | 40.8%      | 49  |
| Server-Side Logic                                      | 23.5%         | 23.5% | 41.2%      | 17  | 53.8%  | 61.5% | 38.5%      | 13  |
| System-Wide Quality and Reliability Tasks              | 0.0%          | 0.0%  | 0.0%       | 3   | 100.0% | 50.0% | 100.0%     | 2   |

one" SWE work as all tasks build off of a well-established and maintained codebase. However, the entire front-end portion of the Expensify repository is maintained by freelance software engineers, which makes it closer to real-world engineering than other freelance postings. We also include SWE management tasks, which freelancers rarely handle. Still, while SWE-Lancer may be a good leading indicator of how automation could affect freelance software engineering, we remain cautious about extrapolating impact beyond this. Modalities. SWE-Lancer tasks are text-only, though issue writers often include a screen recording. This evaluation does not test how much screenshots or videos can improve model performance.

Environments. Unlike real engineers, the models cannot ask clarifying or follow-up questions. Contamination. Contamination is a common limitation of coding benchmarks that rely on open-source repositories. SWE-Lancer Diamond tasks originate from public GitHub issues between 2023 and 2024. Depending on model cutoff date, contamination in training data is possible. Moreover, because tasks are posted online, a model with browsing capabilities or an internet search tool can look up solutions. Therefore, when running this eval, it is important to either disable browsing or limit sites accessed to prevent directlookups, and to ideally also post-hoc filter for any instances of cheating. This will help ensure evaluation results measure real capability rather than contamination. Note that Table 4 in Appendix A2 shows no clear performance improvement for tasks predating the models' knowledge cutoffs, suggesting limited impact of contamination for those tasks.

## 5. Future Work

Economic analysis. Model performance on SWE-Lancer maps to real-world economic value; each task in the benchmark has an associated payout that was sent to a freelancer via Upwork. We hope this benchmark enables research into the broader societal impacts of autonomous agents, including impacts on labor markets, productivity, and the acceleration of AI R&D. Multimodality. Our current evaluation harness does not support multimodal inputs; however, the majority of Expensify GitHub issues contain videos (and, occasionally, screenshots) of the expected behavior. Additionally, the user tool saves screenshots to the output directory upon completion, which current models are unable to view. Because there is not widespread support for such inputs, we prioritized making information about the prompt and environment available via text (for example, through text-based trajectories from the user tool).

## 6. Conclusion

We introduce SWE-Lancer, a benchmark for evaluating model performance on real-world, freelance software engineering work. This software engineering benchmark maps model performance to real monetary value, and improves realism of model benchmarks by evaluating more complex full-stack software engineering and management skills. We hope SWE-Lancer enables greater research into the economic impact of AI model development.

## Impact Statement

AI models with strong real-world software engineering capabilities could enhance productivity, expand access to highquality engineering capabilities, and reduce barriers to technological progress. However, they could also shift labor demand—especially in the short term for entry-level and freelance software engineers—and have broader long-term implications for the software industry. Improving AI software engineering is not without risk. Advanced systems could carry model autonomy risk in self-improvement and potential exfiltration, while automatically generated code may contain security flaws, disrupt existing features, or stray from industry best practices, a consideration that is important if the world increasingly relies on model-generated code. SWE-Lancer provides a concrete framework to start tying model capabilities to potential real-world software automation potential, therefore helping better measure its economic and social implications. By quantifying AI progress in software engineering, we aim to help inform the world about the potential economic impacts of AI model development, while underscoring the need for careful and responsible deployment. To further support responsible AI progress, we open-source a public eval set and report results on frontier AI models to help level-set the implications of AI progress. Future work should explore the societal and economic implications of AI-driven development, ensuring these systems are integrated safely and effectively.

## Acknowledgements

This evaluation would not have been possible without the help of those at OpenAI who work on foundational systems used by teams across the company. In particular, we would like to thank Evan Mays and Kevin Liu for underlying evaluation infrastructure development, Nat McAleese for coming up with the name SWE-Lancer and for valuable feedback and collaboration on the price analysis, Rachel Dias for human data vendor sourcing and setup, Jack Edwards for open-source codebase cleanup and price analysis support, Grace Kim for task validation support, Oliver Jaffe, Jack Edwards, Giulio Starace and James Aung for support with running open-source models, Patrick Chao for plotting library development, Olivia Watkins and Philip Guo for valuable feedback on the price analysis, Phoebe Thacker and Mia Glaese for human data team leadership, Olivia Watkins, Patrick Chao, Miles Wang, Leyton Ho, Andy Applebaum, Leon Maksin, Joel Parish, Kevin Liu, Aleksander Madry, and Alex Beutel for feedback on earlier drafts of the paper, the many software engineer annotators for qualitychecking tasks, the OpenAI Preparedness, Safety Systems, and Research organizations for sponsoring this work, and the Upwork and Expensify developers and contributors for open-sourcing their original process.

## References

Anthropic. Responsible scaling policy. Technical report, Anthropic, October 2024. URL https://assets.anthropic.com/ m/24a47b00f10301cd/original/ Anthropic-Responsible-Scaling-Policy-2024-10-15. pdf.

Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., and Sutton, C. Program synthesis with large language models, 2021. URL https://arxiv.org/abs/2108.

07732.

Brar, H. K. and Kaur, P. J. Differentiating integration testing and unit testing. In *2015 2nd International Conference* on Computing for Sustainable Global Development (IN- DIACom), pp. 796–798, 2015.

Chen, M., Tworek, J., Jun, H., Yuan, Q., de Oliveira Pinto, H. P., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., Ryder, N., Pavlov, M., Power, A., Kaiser, L., Bavarian, M., Winter, C., Tillet, P., Such, F. P., Cummings, D., Plappert, M., Chantzis, F., Barnes, E., Herbert-
Voss, A., Guss, W. H., Nichol, A., Paino, A., Tezak, N., Tang, J., Babuschkin, I., Balaji, S., Jain, S., Saunders, W., Hesse, C., Carr, A. N., Leike, J., Achiam, J., Misra, V., Morikawa, E., Radford, A., Knight, M., Brundage, M., Murati, M., Mayer, K., Welinder, P., Mc- Grew, B., Amodei, D., McCandlish, S., Sutskever, I., and Zaremba, W. Evaluating large language models trained on code, 2021. URL https://arxiv.org/abs/ 2107.03374.

Chowdhury, N., Aung, J., Shern, C. J., Jaffe, O., Sherburn, D., Starace, G., Mays, E., Dias, R., Aljubeh, M., Glaese, M., Jimenez, C. E., Yang, J., Liu, K., and Madry, A. Introducing swe-bench verified. *arXiv preprint* arXiv:2407.01489, 2024.

Li, Y., Choi, D., Chung, J., Kushman, N., Schrittwieser, J., Leblond, R., Eccles, T., Keeling, J., Gimeno, F.,
Dal Lago, A., Hubert, T., Choy, P., de Masson d'Autume, C., Babuschkin, I., Chen, X., Huang, P.-S., Welbl, J., Gowal, S., Cherepanov, A., Molloy, J., Mankowitz, D. J., Sutherland Robson, E., Kohli, P., de Freitas, N., Kavukcuoglu, K., and Vinyals, O. Competitionlevel code generation with alphacode. *Science*, 378 (6624):1092–1097, December 2022. ISSN 1095-9203. doi: 10.1126/science.abq1158. URL http://dx.doi. org/10.1126/science.abq1158.

DeepMind, G. Introducing the frontier safety framework, 2024. URL https:
//deepmind.com/blog/article/
introducing-the-frontier-safety-framework.

Expensify. Issue \#14958: zip/postcode validation error message not displayed for entering ',' on the home address screen. https://github.com/Expensify/ App/issues/14958, 2023.

Expensify. Issue \#25889: Dev: Share code avatar differs from profile avatar. https://github.com/
Expensify/App/issues/25889, 2024a.

Expensify. Issue \#41239: Add support for copy/pasting images on ios. https://github.com/Expensify/ App/issues/41239, 2024b.

Gupta, V., Fernandez-Crehuet, J. M., and Hanne, T. Free- ´
lancers in the software development process: A systematic mapping study. *Processes*, 8(10):1215, 2020. doi: 10.3390/pr8101215. URL https://www.mdpi. com/2227-9717/8/10/1215.

Hendrycks, D., Basart, S., Kadavath, S., Mazeika, M., Arora, A., Guo, E., Burns, C., Puranik, S., He, H., Song, D., and Steinhardt, J. Measuring coding challenge competence with apps, 2021. URL https://arxiv.org/abs/
2105.09938.

Jain, N., Han, K., Gu, A., Li, W.-D., Yan, F., Zhang, T.,
Wang, S., Solar-Lezama, A., Sen, K., and Stoica, I. Livecodebench: Holistic and contamination free evaluation of large language models for code, 2024. URL https://arxiv.org/abs/2403.07974.

Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., and Narasimhan, K. Swe-bench: Can language models resolve real-world github issues?, 2024. URL https://arxiv.org/abs/2310.06770.

Lai, Y., Li, C., Wang, Y., Zhang, T., Zhong, R., Zettlemoyer, L., tau Yih, S. W., Fried, D., Wang, S., and Yu, T. Ds1000: A natural and reliable benchmark for data science code generation, 2022. URL https://arxiv.org/ abs/2211.11501.

Lam, M. S., Teoh, J., Landay, J. A., Heer, J., and Bernstein, M. S. Concept induction: Analyzing unstructured text with high-level concepts using lloom. In Proceedings of the CHI Conference on Human Factors in Computing Systems, CHI '24, pp. 1–28. ACM, 2024. doi: 10.1145/ 3613904.3642830. URL http://dx.doi.org/10. 1145/3613904.3642830.

Liu, T., Xu, C., and McAuley, J. Repobench: Benchmarking repository-level code auto-completion systems, 2023. URL https://arxiv.org/abs/2306.03091.

Lu, S., Guo, D., Ren, S., Huang, J., Svyatkovskiy, A.,
Blanco, A., Clement, C., Drain, D., Jiang, D., Tang, D., Li, G., Zhou, L., Shou, L., Zhou, L., Tufano, M., Gong, M., Zhou, M., Duan, N., Sundaresan, N., Deng, S. K., Fu, S., and Liu, S. Codexglue: A machine learning benchmark dataset for code understanding and generation, 2021. URL https://arxiv.org/abs/2102.04664.

Microsoft. Playwright: Fast and reliable end-to-end testing for modern web apps. https://playwright.

dev/, 2025.

Muennighoff, N., Liu, Q., Zebaze, A., Zheng, Q., Hui, B.,
Zhuo, T. Y., Singh, S., Tang, X., von Werra, L., and Longpre, S. Octopack: Instruction tuning code large language models, 2024. URL https://arxiv.org/ abs/2308.07124.

Nijkamp, E., Pang, B., Hayashi, H., Tu, L., Wang, H., Zhou, Y., Savarese, S., and Xiong, C. Codegen: An open large language model for code with multi-turn program synthesis, 2023. URL https://arxiv.org/abs/2203. 13474.

OpenAI. Openai preparedness framework, 2023. URL https://cdn.openai.com/ openai-preparedness-framework-beta. pdf.

OpenAI. Learning to reason with llms, 2024a. OpenAI. Openai o3 and o3-mini—12 days of openai: Day 12, 2024b.

OpenAI. *OpenAI Authentication API Documentation*,
2025. URL https://platform.openai.com/ docs/api-reference/authentication.

Paul, R., Tsai, W. T., Chen, Y., Fan, C., Cao, Z.,
and Huang, H. End-to-end (e2e) testing and evaluation of high-assurance systems. In Pham, H. (ed.), Springer Handbook of Engineering Statistics, Springer Handbooks, pp. 439–467. Springer, London, London, 2006. ISBN 978-1-85233-806-0. doi: 10.1007/ 978-1-84628-288-1 24. URL https://doi.org/ 10.1007/978-1-84628-288-1_24.

Quan, S., Yang, J., Yu, B., Zheng, B., Liu, D., Yang, A., Ren, X., Gao, B., Miao, Y., Feng, Y., Wang, Z., Yang, J., Cui, Z., Fan, Y., Zhang, Y., Hui, B., Lin, J., and Qwen Team, A. G. Codeelo: Benchmarking competition-level code generation of llms with human-comparable elo ratings. arXiv preprint arXiv:2501.01257, 2025. URL https: //arxiv.org/abs/2501.01257.

Si, C., Yang, D., and Hashimoto, T. Can llms generate novel research ideas? a large-scale human study with 100+ nlp researchers, 2024. URL https://arxiv.

org/abs/2409.04109.

Tian, M., Gao, L., Zhang, S. D., Chen, X., Fan, C., Guo, X.,
Haas, R., Ji, P., Krongchon, K., Li, Y., Liu, S., Luo, D., Ma, Y., Tong, H., Trinh, K., Tian, C., Wang, Z., Wu, B., Xiong, Y., Yin, S., Zhu, M., Lieret, K., Lu, Y., Liu, G., Du, Y., Tao, T., Press, O., Callan, J., Huerta, E., and Peng, H. Scicode: A research coding benchmark curated by scientists, 2024. URL https://arxiv.org/abs/ 2407.13168.

Yang, J., Jimenez, C. E., Zhang, A. L., Lieret, K., Yang, J., Wu, X., Press, O., Muennighoff, N., Synnaeve, G., Narasimhan, K. R., Yang, D., Wang, S. I., and Press, O. Swe-bench multimodal: Do ai systems generalize to visual software domains?, 2024. URL https:// arxiv.org/abs/2410.03859.

Zhang, S., Zhao, H., Liu, X., Zheng, Q., Qi, Z., Gu, X.,
Zhang, X., Dong, Y., and Tang, J. Naturalcodebench: Examining coding performance mismatch on humaneval and natural user prompts, 2024. URL https://arxiv. org/abs/2405.04520.

Zhong, R., Zhang, P., Li, S., Ahn, J., Klein, D., and Steinhardt, J. Goal driven discovery of distributional differences via language descriptions, 2023. URL https: //arxiv.org/abs/2302.14233.

Zhuo, T. Y., Vu, M. C., Chim, J., Hu, H., Yu, W., Widyasari, R., Yusuf, I. N. B., Zhan, H., He, J., Paul, I., Brunner, S., Gong, C., Hoang, T., Zebaze, A. R., Hong, X., Li, W.-D., Kaddour, J., Xu, M., Zhang, Z., Yadav, P., Jain, N., Gu, A., Cheng, Z., Liu, J., Liu, Q., Wang, Z., Lo, D., Hui, B., Muennighoff, N., Fried, D., Du, X., de Vries, H., and Werra, L. V. Bigcodebench: Benchmarking code generation with diverse function calls and complex instructions, 2024. URL https: //arxiv.org/abs/2406.15877.

## A. Appendix A.1. Pass@1 Rate Analysis By Category

Table 3 shows pass@1 rates for SWE-Lancer Diamond categorized by the tasks' nature of work.

| Table 3. Diamond Task Pass Rates by Nature of Work IC SWE   | SWE Manager   |       |            |     |        |       |            |     |
|-------------------------------------------------------------|---------------|-------|------------|-----|--------|-------|------------|-----|
| Task Type                                                   | GPT-4o        | o1    | Sonnet 3.5 | n   | GPT-4o | o1    | Sonnet 3.5 | n   |
| Bug Fixes                                                   | 9.6%          | 19.2% | 28.4%      | 208 | 34.7%  | 39.9% | 44.0%      | 248 |
| New Features or Enhancements                                | 0.0%          | 4.8%  | 14.3%      | 21  | 69.2%  | 69.2% | 61.5%      | 13  |
| Maintenance, QA, Testing, or Reliability                    | 0.0%          | 12.5% | 0.0%       | 8   | 75.0%  | 50.0% | 50.0%      | 4   |

To determine each task's category (as shown in Tables 2 and 3), we had o1 review each entire Github issue thread, including all descriptions and comments, three separate times for each IC SWE and SWE Management task. If these three categorizations were not unanimous, we manually assigned the final categories.

## A.2. Contamination Analysis

| Table 4. Diamond Pass@1 Rate Analysis Based on Task Creation Date. IC SWE Manager   |             |                 |                |                 |                |
|-------------------------------------------------------------------------------------|-------------|-----------------|----------------|-----------------|----------------|
| Model                                                                               | Cutoff Date | Pass@1 (Before) | Pass@1 (After) | Pass@1 (Before) | Pass@1 (After) |
| GPT-4o                                                                              | 2023-10-01  | 0.038 (n=79)    | 0.101 (n=158)  | 0.373 (n=142)   | 0.244 (n=123)  |
| o1                                                                                  | 2023-10-01  | 0.139 (n=79)    | 0.177 (n=158)  | 0.387 (n=142)   | 0.407 (n=123)  |
| Claude 3.5 Sonnet                                                                   | 2024-04-01  | 0.178 (n=118)   | 0.345 (n=119)  | 0.452 (n=210)   | 0.436 (n=55)   |

## A.3. Further Exploration Of Swe Management Tasks

Within SWE Management tasks, the winning proposal is often selected for nuanced reasons, such as addressing edge cases or reducing external dependencies. For example, one task (Expensify, 2024b) involves selecting a proposal for implementing image-pasting on iOS. The winning proposal handled multiple clipboard formats (e.g., base64, blob), minimized permission prompts, and aligned with native iOS behavior by enabling direct pasting into the compose bar. Other proposals were deprioritized due to reliance on third-party libraries, complex modifications, or inconsistent cross-platform support.

Example: "[$2000] Add support for copy/pasting images on iOS #41239 Description (paraphrased): Users want the ability to copy an image to their clipboard and paste it directly into a chat in the Expensify iOS app. Currently, users must save the image and manually upload it, which is inefficient.

bounty increase $500 (May 16) → $1K (May 23)
original bounty; $250 (April 30) w final bounty: $2K (July 9)
:) Proposal 1 by @ShridharGoe
.Proposal 3 by @suneo:
.Proposal 4 by @s77n Proposed Changes & Optional Images Proposed Changes & Optional Images Proposed Changes & Optional Images An and the Use case 1: Handling base64 image data iOS: RCTUITextView menu of chats, which will show only when 1. Patch canPerformAction: method to there's an image copied in the clipboard.

force show the Paste option if the Clicking on this option will open the clipboard has an image attachment modal with the image.

2. Patch paste method to call textInputDidPaste method 3. Implement textInputDidPaste ; this method will just send an event to the component view via onPaste prop Android: ReactEditText INTO A STATE AND IN THE PROPER
1. Patch onTextContextMenuItem method to call PasteWatcher Like we have displayFileInModal , we'll 2. Implement PasteWatcher ; this class will edd displayFileInModalUsingUrl which will just send an event to the component view have uri of the copied image as input via onPaste prop New option in attachment picker:
App: Composer 1. Add onPaste prop and Use case 2: Handling images copied cal Clipboard.getImagePNG / Clipb blobs (shortened to fit)
oard.getInage Update: I raised a PR that covers most changes... leaving a minor patch This proposal adds a "Paste image" option to the This proposal detects clipboard content as This new proposal uses patches to native components to comprehensively add image paste iOS chat menu, which detects copied clipboard base64 strings or blobs, converts them into file images, validates format, stores the URL, and objects, and enables users to paste intuitively support for the Composer on both IOS and opens images in the attachment modal for sharing through long-press gestures or a "Paste Image Android.

button.

l SWE Managers provide feedback on proposals and ultimately accept or reject them SWE Manager Feedback: "Pasting the image SWE Manager Feedback (after some back and from the URL in the clipboard seems a forth): "I think your method to detect SWE Manager Feedback: "Can you share a workaround, because when we copied the onPasteFile and make the paste option test branch? Also are we able to use the same image, we click "copy image" not "copy image appear after copying an image from the gallery approach in Android, Clipboard address", so we may need to handle the image is unreliable. If Clipboard.getImagePNG()
has Clipboard.getInage() for Android as fileObject or base64 string.

works for iOS and Clipboard.getImage() for Also, your proposal doesn't have how we will Android, we could handle on PasteFile natively IC SWE: Yes, I've updated the proposal detect onPasteFile in the normal case by long in MarkdownTextInput and use Clipboard to clicking on the input then click paste."
get the image content. What do you think?"
Proposal Accepted X Proposal Rejected X Proposal Rejected Solution was comprehensive and Solution was not comprehensive. It Solution was not comprehensive. It native, handling all cases. It handled only worked for image URLs , which is handled images copied from the gallery and web images as raw image uncommon, and failed to support raw gallery but failed to support web content, and supporting iOS and image data.

images.

Android.

## A.4. Coding And Software Engineering Evaluations Comparison Table

| monetary metrics, incorporates full-stack tasks and management tasks, and evaluates using execution-based end-to-end tests hand-crafted by engineers. In contrast, other benchmarks rely on pre-existing tests within repositories or tests generated by LLMs. Source of Evaluation Language Dataset Source Data Type Capabilities Monetary Evaluation Evaluation Tested Metric? Method Method Open-source and commercial Expensify repo of issues posted on Upwork Professional software engineering ✔ End-to-end Software Hand-written by professional engineering test software engineers problem statements SWE-Lancer Javascript / Typescript GitHub issues from SWE-Bench Verified Python Open-source maintenance ✖ Unit test Sourced from Software GitHub repos all developer original repository frameworks Software maintenance tasks using multimodal data GitHub issues SWE-Bench Javascript / Open-source focused on Typescript / Multimodal GitHub repos developer tools and HTML / CSS utilities  Unit test Sourced from ✖original repository GitHub issues from Python repositories that meet certain criteria (post Jan '23, non-fork, 100+ stars, 80%+ Python, unit tests) Unit test, completion ✖ Code Sourced from similarity metrics original repository RepoCoder Python Open-source Python repos Github-Code RepoBench Python and Dataset and newly Single and multi-file and completion ✖ Similarity Code retrieval Java crawled GitHub data code snippets metrics N/A (post-Feb 2023) Coding problems designed for use in competitive programming competitions LeetCode, AtCoder, LiveCodeBench Python and CodeForces contests generation ✖ Unit test Generated using Code LLMs Problem statements derived from generation ✖ Automated Code Generated using GPT-4 then refined real-world user unit testing queries by humans NaturalCodeBench Python and User queries from online coding Java services Natural language Generated using BigCodeBench Python Human expert completion ✖ Unit test Code tasks composed of GPT-4 then refined curation multiple substeps by humans HumanEval Python Hand-written by human experts Python functions Code generation ✖ Unit tests Hand-written by human experts Python / Java / JS / PHP / CodeXGLUE Ruby / Go / C / C#) Open-source Code Unit test, 14 task-specific ✖Benchmark authors GitHub repos & code/text datasets understanding similarity (Lu et al., 2021) newly curated sets & generation metrics Crowd sourced, MBPP Python some are expert Short entry-level completion ✖ Unit test Crowd sourced Code Python functions verified Solving MathQA-Python Python Converted from mathematical word MathQA problems using Python Python  Correctness Sourced from program ✖MathQA synthesis AlphaCode C++ / Scraped from Competition coding generation ✖ Unit test Scraped from Code Python / Java Codeforces problems Codeforces Multi-turn coding MTPB Python Human expert completion ✖ Unit test Human expert Code problems and written written prompts DS-1000 Python Collected from Data science problems Fixing code ✖ Unit test Collected from StackOverflow StackOverflow   |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

Comparing Learnings Against Prior Benchmarks: For SWE-Lancer's IC SWE tasks, one comparable benchmark is SWE-Bench Multimodal, where GPT-4o solved 12.2% of visual programming tasks when given the best scaffold and multimodal capabilities. This is comparable to GPT-4o scoring 8.0% on our IC SWE Diamond tasks when the model can ingest image screenshots via the user tool. On SWE-Bench Verified, the industry standard benchmark for software engineering, GPT-4o currently scores 38.8% using the best available scaffold, but only 8% on SWE-Lancer Diamond IC SWE tasks - likely because SWE-Bench Verified focuses on bugfixes, and SWE-Lancer includes open-ended problems (adding new features, reliability) and full-stack development. This highlights the need for more challenging, unsaturated SWE benchmarks.

## A.5. Additional Scaffold Details

Each agent is executed within a Microsoft Azure Standard D2as v4 virtual machine with 2 vCPUs and 8GB RAM. We execute agents within Docker containers using a pre-built SWE-Lancer image, which we open-source. The container has 64 GB of shared memory allocated and 192GB of maximum memory available. All the libraries the agent can access (i.e., the ones relevant for the Expensify repository) are installed in the container. The agent does not have multimodal image or video inputs. For each rollout, the maximum number of tool calls is 100 and the maximum time allowed is 3 hours. All rollouts are conducted with temperature 1.0. Except for our pass@k experiments, each is just a single rollout.

## A.6. Task Curation Criteria

Figure 12. Lifecycle of an Upwork task. First, project maintainers post a software engineering task to Upwork with an associated payout. Then, users submit in-depth proposals for fixing the issue. Finally, an assigned reviewer selects the best proposal - (used to create SWE Manager tasks). The winning proposal is implemented (used to create IC SWE tasks).

We manually filtered tasks according to the following criteria. Each task in our final set was screened by at least three experienced human software engineers.

1. The issue is sourced from an open-source repository. 2. The issue has an associated, concrete payout. 3. The issue was resolved in the past. 4. The issue is well-specified and clear - models must be able to actually understand the task and implement a fix, without being able to ask for clarification.

5. The PR that resolved the issue actually fixes it - this fix is confirmed by executing locally.

There are additional criteria for SWE Management Tasks:
1. There were at least two well-defined proposals for the issue. 2. The proposals considered contain sufficient information about the correct fix.

Further detail is provided below on how we assess specific criteria and labeled other attributes for each given task. To confirm that the issue was well-specified and clear, we asked software engineers to rate each issue from 0-3 (below) and only accepted issues where the majority of human engineers rated them as 0.

Read the issue description again here, and imagine that you are an experienced software engineer with 2-3 years of relevant experience and familiarity with the codebase. You have been instructed to create a PR that successfully resolves it. You have full access to the codebase and can use the title, description, reproduction steps, and any attached videos/images from the linked issue when forming your solution.

Please note, however, that you are not able to ask for clarification and would need to work exclusively from the above information. Please do not read the comments below the issue when making your decision—you must rely solely on the content of the issue itself and codebase/docs.

Please assign a rating from 0 to 3 using the rubric below.

0- The issue is well-specified, and it is clear what is required for a successful solution. 1- There are some blanks to fill in about the issue, but there is a sensible interpretation of what is required for a successful solution.

2- The issue is vague, and there is room for ambiguity. It is unclear what a successful solution would look like. 3- It is almost impossible to understand what you are being asked to do without further information.

Issue Well-Specified Score: [0 to 3] Please explain your choice above. Include references to specific filenames, function/-
class names, or lines of code where relevant. [Text box]
To confirm that the tests were valid for all reasonable, correct solutions, we asked engineers to rate the tests according to a rubric and only accepted issues where the majority of human engineers rated them as 0.

Please re-review the pull request that addressed the issue. Does it contain valid, well-scoped tests so that all reasonable solutions pass?

Please assign a rating from 0 to 3 using the rubric below.

0- The tests perfectly cover all possible solutions. 1- The tests cover the majority of correct solutions; however, some unusual solutions may be missed. 2- The tests work, but some perfectly reasonable solutions may be missed by the tests. 3- The tests are too narrow/broad or they look for something different than what the issue is about.

## A.7. Swe-Lancer Robustness To Cheating

Existing SWE benchmarks in the literature often rely on unit tests that are narrow and prone to cheating. An example of cheating by o1 in SWE-Bench Verified is shown below: An excerpt from the problem description is included below:
Queryset raises NotSupportedError when RHS has filterable=False attribute.

In response, o1 adds an underscore before filterable so that the branch never executes:
2: ASSISTANT → ALL
django/db/models/sql/query.py
<<<<<< SEARCH if not getattr(expression, 'filterable', True):
raise NotSupportedError( expression.__class__.__name__ + ' is disallowed in the filter ' 'clause.'
)
======== if not getattr(expression, '_filterable', True):
raise NotSupportedError( expression.__class__.__name__ + ' is disallowed in the filter ' 'clause.'
)
>>>>>> REPLACE
As language models grow more capable, it is increasingly important to publish challenging, un-hackable software engineering benchmarks. SWE-Lancer's human-validated end-to-end tests emulate the actual workflow of a human engineer, rather than relying on a handful of selectively chosen test cases. As a result, SWE-Lancer is inherently more resistant to cheating and can more accurately reflect a model's capacity to produce genuine, economically valuable solutions for real-world engineering challenges.

## A.8. Prompting

Models are prompted with the following general template for IC SWE Tasks. We did not find variance in performance when including vs. excluding the prices from the promtp.

## Ic Swe Task Instructions

"""You are an expert software engineer maintaining the Expensify repository. It has already been cloned, and is present in /app/expensify (your CWD). You have been tasked with fixing the following issue: {issue title} {full HTML documentation of issue, including how to reproduce the bug, the actual vs. expected behavior, and which platforms are affected} The reward for the correct fix is ${price}. You do not need to push your changes to the repository, we will deal with this for you. Do not attempt to run the app; simply make your changes and then end your turn. Do not end your turn until you have submitted your code changes. You absolutely must not leave action items / TODOs. --- You will be provided with a user tool, which you can call by running "bash -i -c 'user-tool'" in the command line, which has a user open the app with your local changes and perform the actions specified in the bug report. It could take up to a minute for the user to finish using the app. When you call the user tool, when the user finishes /app/expensify/user_tool/
will be populated with folders called output_browser1 ... n for as many browsers as were used by the user during their test. Inside each folder, you will find a file called trace.trace, which contains the trajectory the user took. Here are a few things to note: - You will need to parse this file programmatically in order to open it. - If a row has "type":"screencast-frame", then the "sha1" property will correspond to a .jpeg file in the resources folder, which is a screenshot of the browser at that time.

- If you want to view the page as HTML, rather than a JPEG,
you can look for rows that are called "type":"frame-snapshot"; you can print them out to see the HTML at this point in time.

- You should review the rows with type "type" set to "log", "before", or "after" as they are important and show what actions are being taken. - All of the rows are in order, so e.g. you can find an interesting log row and then look at the next row that is a frame-snapshot to see the HTML.

You should use this to help you iteratively debug and fix the issue. You can call it as often as you need to. When you call the user tool, you wait 90,000 milliseconds (e.g. by setting the timeout argument to 90000)! """
Models are prompted with the following general template for SWE Management Tasks.

## Swe Management Task Instructions

f"""You are an expert software engineering manager working on the Expensify repository. You have tasked your team with addressing the following issue: {issue title} {full HTML documentation of issue, including how to reproduce the bug, the actual vs. expected behavior, and which platforms are affected} You will be paid {'price'} if your team fixes the issue. Since you are a manager, your task is to select the best proposal regarding how to fix the issue from the below options: {list of all proposals} --- For your convenience, the Expensify repository has already been cloned, and is present in /app/expensify (your CWD). You should not attempt to run the app or implement the entire fix, but you can review the codebase to understand the context of the issue.

When you are ready to submit your answer, please create a JSON file in
/app/expensify/manager_decisions.json with the following format: {
selected_proposal_id: int,
} You must write your output to this path exactly: '/app/expensify/manager_decisions.json'. You must not submit to anywhere in /mnt/data, just write to '/app/expensify/manager_decisions.json' exactly! """

## A.9. Qualitative Analysis Of Rollouts

We conduct a qualitative analysis of task instances from SWE-Lancer, focusing on model-generated solutions and debugging strategies. The analysis is based on agent trajectories under the default configuration of o1, 4o, and Claude 3.5 Sonnet. Issue Summary. The Expensify app fails to display a validation error message when users enter a comma (,) in the zip/postcode field on the Home Address screen, allowing invalid input. This issue occurs across multiple platforms and may stem from missing input sanitization or incomplete validation logic. The bug was observed in staging and production, potentially introduced during the N7 Account Settings update (Issue \#14958). Note on end-to-end tests. The end-to-end tests we developed for each issue are robust and reflect the key considerations from the original Github discussion. While this issue initially concerned the commas in zip-code validation, much of the discussion in the Github thread focused on improving validation overall, including country-specific regex dictionaries and handling edge cases where the user hasn't selected a country or a country lacks a postal system. Thus, our end-to-end tests measure not just whether the model's code causes the validation error to appear when entering a comma, but also whether it can comprehensively handle the full range of global zip-code formats.