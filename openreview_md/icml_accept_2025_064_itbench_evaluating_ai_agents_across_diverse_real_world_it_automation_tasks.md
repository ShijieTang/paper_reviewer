# Itbench: Evaluating Ai Agents Across Diverse Real-World It Automation Tasks

Saurabh Jha * 1 **Rohan Arora** * 1 **Yuji Watanabe** * 1 Takumi Yanagawa 1 Yinfang Chen 2 **Jackson Clark** 2 Bhavya Bhavya 1 Mudit Verma 1 Harshit Kumar 1 Hirokuni Kitahara 1 Noah Zheutlin 1 **Saki Takano** 1 Divya Pathak 1 Felix George 1 Xinbo Wu 2 Bekir O Turkkan 1 Gerard Vanloo 1 Michael Nidd 1 **Ting Dai** 1 Oishik Chatterjee 1 Pranjal Gupta 1 Suranjana Samanta 1 Pooja Aggarwal 1 Rong Lee 1 **Jae-wook Ahn** 1 Debanjana Kar 1 Amit Paradkar 1 Yu Deng 1 Pratibha Moogi 1 Prateeti Mohapatra 1 **Naoki Abe** 1 Chandrasekhar Narayanaswami 1 Tianyin Xu 2 Lav R. Varshney 2 Ruchi Mahindru 1 **Anca Sailer** 1 Laura Shwartz 1 Daby Sow 1 Nicholas C. M. Fuller 1 **Ruchir Puri** 1

## Abstract

Realizing the vision of using AI agents to automate critical IT tasks depends on the ability to measure and understand effectiveness of proposed solutions. We introduce ITBench, a framework that offers a systematic methodology for benchmarking AI agents to address real-world IT automation tasks. Our initial release targets three key areas: Site Reliability Engineering (SRE),
Compliance and Security Operations (CISO), and Financial Operations (FinOps). The design enables AI researchers to understand the challenges and opportunities of AI agents for IT automation with push-button workflows and interpretable metrics. ITBench includes an initial set of 102 realworld scenarios, which can be easily extended by community contributions. Our results show that agents powered by state-of-the-art models resolve only 11.4% of SRE scenarios, 25.2% of CISO scenarios, and 25.8% of FinOps scenarios (excluding anomaly detection). For FinOps-specific anomaly detection (AD) scenarios, AI agents achieve an F1 score of 0.35. We expect ITBench to be a key enabler of AI-driven IT automation that is correct, safe, and fast. ITBench, along with a leaderboard and sample agent implementations, is available at https://github.com/ibm/itbench.

## 1. Introduction

Modern IT systems are driving many facets of our economy. They have grown significantly in complexity with the adoption of cloud computing and agile development practices (Harvard Business Review Research Report, 2022; Trask, 2025). Effective management of these systems is becoming extremely challenging as corporations struggle to keep up with this growing complexity. Various IT personasranging from Chief Information Officers to Site Reliability Engineers and Security and Compliance officers—and IT
engineers in general are struggling to ensure resiliency, reliability, security, and cost effective operations of IT Systems. The recent CrowdStrike outage highlighted these challenges as it brought down our society's most critical systemsfrom hospital services to air travel—and was estimated to cost US Fortune 500 companies a staggering $5.4 billion (Kerner, 2024). This incident underlined the critical need for intelligent IT incident resolution, with compliance and risk management capabilities, a topic also addressed in the Digital Operational Resiliency Act (DORA) in Europe (Parliament and the Council of the European Union, 2024). The rising popularity of AI agents and their projected ability to handle intricate tasks have increased the demand for AI agents managing IT systems (John, 2024; Miguel Carreon, 2024; Pujar et al., 2023). Given the complexity of IT tasks, a major hurdle for this research is establishing systematic methods to assess the effectiveness of AI agents prior to their production deployment (Bogin et al., 2024; Kapoor et al., 2024). Consequently, there is an urgency to develop methods for evaluation of AI agents based on real IT tasks and their corresponding environments.

This paper addresses this critical need and presents ITBench, a first-of-its-kind framework that is both comprehensive and visionary for benchmarking real-life IT automation tasks. The goal of ITBench is to measure the performance of AI
1

SRE
Ensures app resilience and performance FinOps Manage IT spend CISO
Manage threats and assess policies Resolve "High error rate on service order-management." Backup directory 'foo' to data lake.

Assess compliance posture for "new control rule detected for RHEL 9."
Assess and report critical risks across the failing controls.

Resolve "IT spend exceeded the budget." Report return on investment per application.
agents across a wide variety of complex and real-life IT
tasks across personas, including *Site Reliability Engineering* (SRE), focusing on availability and resiliency; *Compliance* and Security Operations (CISO), ensuring compliance and security of IT implementations; and *Financial Operations* (FinOps), enforcing cost efficiencies and optimizing return on investment, among others (as shown in Figure 1). ITBench aims to advance innovation and establish new standards in the field. Our contributions can be summarized along the following three axes: - **Reflecting the real world:** ITBench addresses the IT
automation requirements that are relevant and prevalent in production settings. SRE scenarios are based on realworld incidents observed in our own SaaS products. CISO scenarios are based on CIS benchmark (for Internet Security, CIS). FinOps scenarios are identified by the FinOps Foundation (Foundation, 2025a) through key business outcomes.

- Being open and extensible with comprehensive IT coverage: We view ITBench as a central hub for benchmarking AI-driven solutions across diverse IT automation use cases. To support this, we provide IT benchmark suites and a framework for vertical expansion (i.e., adding more scenarios) and horizontal expansion (i.e., adding more personas), ensuring extensive coverage of IT tasks. ITBench is an open-source framework built with open-source technologies, while allowing organizations with proprietary technologies to use it for developing and benchmarking their solutions.

- **Enabling automated evaluation with partial scoring:**
ITBench is designed to provide constructive feedback to drive improvements in the design of agentic solutions for IT problems. It includes a comprehensive evaluation framework and leaderboard that provide feedback to users at various stages of their agents' reasoning process.

ITBench provides push-button deployment and tooling for setting up environment, runtime agent, guardrail engine, as well as authorization and authentication. It allows developers and researchers to build novel solutions for managing complex IT systems. Currently, ITBench addresses reactive problems, including incidents diagnosis and resolution, compliance assessments in regulated environments for new controls, and cost management events. In the future, we plan to expand on benchmark evaluation capabilities and include new benchmarks for additional IT processes. Currently, ITBench comprises an initial set of 102 scenarios spanning across SRE (42), CISO (50), and FinOps (10), with respective successful scenario handling rates of 13.8%, 25.2%, and 25.8% (refer to Section 4). We believe that, similar to the highly influential SWEBench (Jimenez et al., 2024), our new ITBench framework—which encapsulates and measures the ability of AI agents to automate complex, real-world IT tasks—will spur a comparable acceleration in the performance of real-world IT AI agents.

## 2. Related Work

ITBench targets a comprehensive set of tasks for a wide range of personas within IT automation. The initial release of ITBench focuses on evaluating scenarios within IT Operations (ITOps). Figure 1 illustrates currently targeted personas and exemplar tasks that they are routinely facing. There is clearly rising interest in developing benchmarks to evaluate AI and ML techniques in ITOps with specific focus on SRE, CISO, and FinOps.

TrainTicket (Zhou et al., 2018) provides 22 scenarios collected through an industrial survey of real-world incidents, using hardcoded faults in the TrainTicket application to focus on fault localization. AIOpsLab (Chen et al., 2024a) provides 10 SRE-focused scenarios (referred to as "problems") utilizing a real environment (system) integration that allows interactive access to text, time series, and tabular data. InsightBench (Sahu et al., 2024) provides 100 scenarios to analyze ticket data using static tabular data and synthetic scenarios. TSB-AD (Liu and Paparrizos, 2024a) focuses on anomaly detection with 40 synthetic scenarios. CIS-Benchmark (CIS, 2024) provides best practices for securing IT infrastructure. Despite the name of "benchmark," it offers only recommendation policies; it provides no experimental platform. Recently, Cloud Native Compute Foun-

| Benchmark                 | # of Scenarios   | Personas and Tasks           | Resolvable   | Automated   |                  |                |
|---------------------------|------------------|------------------------------|--------------|-------------|------------------|----------------|
| SRE: Incident Resolution, |                  |                              |              |             |                  |                |
| ITBench (ours)            | 102              | CISO: Compliance Assessment, | ✓            | ✓           |                  |                |
| FinOps: Cost Management   |                  |                              |              |             |                  |                |
| TrainTicket               | 22               | SRE: Incident Diagnosis      | ✗            | ✗           | Real Env.        | ✗              |
| AIOpsLab                  | 10               | SRE: Incident Resolution     | ✓            | ✗           | Real Env.        | ✓ (unverified) |
| InsightBench              | 100              | Ticket Data Analysis         | ✗            | ✗           | Synthetic        | ✗              |
| TSB-AD                    | 40               | Anomaly Detection            | ✗            | ✓           | Synthetic        | ✗              |
| CIS                       | 1000+            | Compliance/Security Focal    | ✓            | ✗           | n/a (info. only) | ✗              |

dation (CNCF) Sandbox project (OSCAL-compass, 2024) released an SDK to support the translation of the CIS human readable formats into OSCAL (OSCAL, 2024). OSCAL was developed by the National Institute of Standards and Technology for programmatic usage in compliance automation. ITBench CISO automation leverages this technology to assess policy requirements.

FinOps Foundation (Foundation, 2025a) provides benchmarks that compare cloud financial performance across organizations and departments, focusing on KPIs such as resource utilization efficiency, contract coverage, and cost apportionment. These benchmarks help assess cloud efficiency by evaluating internal and external metrics, fostering structured, collaborative approaches to cloud optimization. While existing benchmarks are valuable resources for specific tasks and use cases, and highlight the critical need for systematic benchmarking, they are limited in reflecting real-world IT problems, covering broad IT landscape, and automating evaluation. These limitations are addressed in ITBench, as shown in Table 1.

## 3. Itbench

ITBench is a systematic benchmarking framework and runtime environment designed to evaluate AI agents tasked with automating IT operations, incorporating a robust architecture (see Figure 2) comprising the AI Agent, Scenario Specification and Environment, Evaluator, and Leaderboard to facilitate comprehensive performance assessment. Here, we present a brief overview of the key components:
1) Scenario Specification and Environment, 2) AI Agents, and 3) Leaderboard. More details are in Appendix B.

## 3.1. Scenario Specification And Environment

The bench incorporates a collection of problems that we call scenarios. For example, one of the problems in ITBench is to resolve a "High error rate on service order-management" in a Kubernetes environment. Another example that is relevant for the CISO persona involves assessing the compliance posture for a "new control rule detected for RHEL 9." A fundamental challenge is to emulate such problems in a manageable testbed environment. A scenario environment is an operational testbed in which a specific problem(s) occurs.

A scenario p generally corresponds to a problem to be solved in ITBench. We formalize p as a tuple *< M, E, T, D >*,
where the variables are as follows: Scenario Specification. M represents metadata and deployment descriptors, for each scenario, which is stored in the Scenario Specs database in ITBench (see Figure 2). Exemplar metadata elements per scenario include scenario_name, scenario_description, scenario_domain, *scenario_class*, scenario_complexity, and *scenario_groundtruth* (see Table 2), which are defined below: - *scenario_name* is name given to a scenario. For example, a scenario in ITBench has the name "Recommendation Service Cache."
- *scenario_description* describes the scenario. An example of a description of the scenario is "Recommendation Service in Astronomy Shop has a cache failure."
- *scenario_domain* represents different personas within IT
automation—namely "SRE," "CISO," and "FinOps."
- *scenario_class* is used to group similar scenarios, such as "Kyverno-opa," "Kyverno-update',' "CacheFailure,"
"HighCPU," and "CorruptImage."
- *scenario_complexity* captures the difficulty of a problem and is defined using domain knowledge. Figure 4a shows the breakdown of SRE, CISO, and FinOps scenarios in the bench. Figure 4b, 4c, and 4d shows *scenario_complexity* distribution for SRE, CISO, and FinOps, respectively.

SRE scenarios are developed based on real-world incidents observed in our own SaaS products. CISO scenarios are based on CIS benchmark (for Internet Security , CIS). FinOps scenarios were developed based on "Domains" and "Capabilities" identified by the FinOps Foundation (Foundation, 2025a) to describe key business outcomes.

- *scenario_groundtruth* records task-specific outcomes that

Scenario Environment Leaderboard ITBench Rank & Publish Application Stack Build & Register Agent Builder Evaluator Agent Observability Stack Metric 1 Metric 2 ... ... Metric N
Action Info. System stateSetup/Cleanup per scenario Agent result Benchmark Builder Develop & 
Register Scenario Specs Benchmark Runner
the Evaluator uses to compare against the agent's expected output. For instance, in incident resolution for SREs, the ground truth for the Diagnosis task includes a list of entities involved in the fault propagation chain, the actual fault propagation chain(s), and fault conditions, while for the Mitigation task, it captures plausible mitigation actions.

Environment. E represents an an operational testbed where the problem occurs. Components within the environment expose APIs to observe and control the environment. When the Agent Builder registers the agent for benchmarking, the Benchmark Runner (see Figure 2) randomly selects a set of scenarios, which may be optionally filtered based on the *agent_type* and *agent_level*. Next, the Benchmark Runner iterates through the set of scenarios, and for each scenario it instantiates a testbed. An example of an environment is a Kubernetes cluster installed with OpenTelemetry Astronomy Shop Demo application (Community, 2024), observability stack including Grafana (gra), Loki (lok), Jaeger (jae), and Prometheus (pro), along with mechanisms that induce problem(s) in the environment. Triggering Events. T is a set of triggering events that occur due to manifestation of a specific problem in the environment. Tools are configured to observe the environment and raise triggering events on problematic conditions. An example of a triggering event is "High Error Rate on adservice,"
which may be triggered in the environment due to cache failure problem.

Desired Outcome. D defines the automation objective and represents the ultimate goal. For instance, in case of SRE incident resolution, the ultimate goal is to clear T in the E.

## 3.2. Ai Agents

In IT automation, the different personas are focused on a specific desired outcome, which defines their automation goals. For SREs, incident resolution is the primary objective. Achieving this can involve multiple steps, such as diagnosing an incident, or a single step, like generating a diagnosis report. CISO persona focuses on the regulatory controls posture assessment process, including *Collect evidence* and

Toolbox call APIs output
 = (| ҧ−1, ത−1)
Agent SRE/FinOps NL2Kubectl NL2Traces NL2Metrics NL2Logs NL2Alerts NL2Script CISO
GenerateKyverno GenerateOPARego GeneratePlaybook RunKubectl RunOPARego RunPlaybook
 = ℎ()
 = (−1, −1)
Environment cmd info
Scan assessment posture tasks. FinOps persona focuses on the cost management, where sample tasks include Identify inefficiency and *Mitigate inefficiency*. During evaluation, each step (task) is assessed independently and is measured using well defined metrics; see Table 3. The goal of ITBench is to evaluate AI agents on a broad range of real-world IT automation tasks that are otherwise performed by SREs, FinOps, and CISO personas.

In this paper, an AI *agent* is defined as an autonomous or semi-autonomous software program that uses an LLM to plan, make decisions, interact with the target environment, and execute actions to achieve goals. An AI agent is expected to successfully handle any of the scenarios in the ITBench, by interacting with the environment.

As shown in Figure 3, agent and environment form a Partially Observed Markov Decision Process (POMDP), where the state is the snapshot of the environment. The state transitions are determined by the environment, which are then (partially) observed by the agent. Given a scenario p instantiated in an environment E, an agent probes the environment via one of the tools and receives an observation ot ∈ O, based on which, it decides the next action:

$$a_{t}=f(o_{t}|{\bar{o}}_{t-1};{\bar{a}}_{t-1})$$
(1)  $\frac{1}{2}$ .............................. 
at = f(ot|o¯t−1; ¯at−1) (1)
Here f is the agent's decision function. o¯t−1 is the sequence of observations up to time t − 1 and a¯t−1 is the sequence of actions taken up to t − 1.

| Scenario Domain                                                                                                                            | Scenario Class                                                              | Scenario Complexity      | Technologies        |
|--------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|--------------------------|---------------------|
| SRE                                                                                                                                        | CacheFailure: Create a memory leak due to an exponentially growing cache    | Medium                   | K8s, Redis, MongoDB |
| HighCPU: Trigger high CPU load in target service                                                                                           | Medium                                                                      | K8, Host, Pods           |                     |
| CorruptImage: Deployment uses wrong Docker image                                                                                           | Easy                                                                        | K8s, Image registry      |                     |
| HTTPRequestBodyTamperFault: Modify HTTP Post request between services                                                                      | Medium                                                                      | K8s, ingress/egress      |                     |
| HTTPRequestAbortFault: Interrupt HTTP connection between services                                                                          | Medium                                                                      | K8ss, ingress/egress     |                     |
| MemoryResourceLimit: Reduce memory limit on target service                                                                                 | Easy                                                                        | K8s, Host, Pod           |                     |
| CISO                                                                                                                                       | New K8s CIS-benchmarks on Kyverno                                           | Easy                     | K8, Kyverno         |
| New K8s CIS-benchmarks on OPA                                                                                                              | Medium                                                                      | K8s, OPA, Kubectl        |                     |
| New RHEL9 CIS-benchmarks on Ansible-OPA                                                                                                    | Medium                                                                      | RHEL9, OPA, Ansible      |                     |
| Update K8s CIS-benchmarks on Kyverno                                                                                                       | Hard                                                                        | K8s, Kyverno             |                     |
| FinOps                                                                                                                                     | CostAlertMisconfiguration: Alert threshold is too low, causing false alerts | Easy                     | K8s, HPA            |
| AutoscalerMisconfiguration: Horizontal pod autoscaler thresholds are misconfigured, creating excess pods                                                                                                                                            | Hard                                                                        | K8s, HPA                 |                     |
| Data Insights Generation: Analyze cloud bills and retrieve data based on natural language query                                            | Easy, Medium, Hard                                                          | Natural Language to SQL  |                     |
| Anomaly Detection & Ranking: Identify overspending events in                                                                               | Anomaly detection, fore                                                                             |                          |                     |
| a cloud bill with regard to forecasted spend amounts, and rank                                                                             | Hard                                                                        | casting, data query from |                     |
| anomalies based on user-specified criteria                                                                                                 | database                                                                    |                          |                     |
| 1 Scenario complexity depends on the characteristics of the scenario and is independent from agent capability. See appendixes for details. |                                                                             |                          |                     |

9.8%
CISO
FinOps 41.2%
49%
SRE
Hard Hard 24.0%
20.0%
Easy 20.0%
Easy 24.0%
Medium 52.0%
Medium 60.0%
Medium 30.0%
Hard 50.0%
Easy 20.0%
Initially, o0 may be a triggering event showing a problematic state s0 of the environment. Given state st−1 and action at−1, the environment transitions to the next state:

$$s_{t}=g(s_{t-1},a_{t-1})$$
st = g(st−1, at−1) (2)
The observation ot is determined as a function of the state and is in general a proxy for the environment state st, hence the formulation can be thought of as a POMDP:

$$O_{t}=h(s_{t})$$
ot = h(st) (3)
The set A of actions is defined as QS{⊥}, where Q is the set of tools and ⊥ represents the "stop action" by the agent. We define t
∗as the time when the agent stops:

$$t^{*}=\operatorname*{min}\{t|a_{t}=\bot\}$$
∗ = min{t|at = ⊥} (4)
An agent reflects on the result to guide its next action, continuing until the final goal is achieved. Given a set of scenarios that the agent works on, it targets to maximize the success defined as follows:

$$\mathbb{E}_{p\sim\pi_{p}}(\mathbb{I}(g(s_{t^{*}}^{p},f(o_{t^{*}}|\bar{o}_{t^{*}-1},\bar{a}_{t^{*}-1}))=s_{G}^{p}))\tag{5}$$
$$(2)$$

where I is an indicator function comparing the terminating state with goal state and π is the distribution of scenarios.

$$({\mathfrak{I}})$$

## 3.2.1. Baseline Ai Agents

We developed baseline agents SRE-Agent for SRE, Compliance Assessment Agent for CISO, and FinOps-agent for FinOps. Each of these agents uses state-of-the-art agentic techniques such as ReAct-based planning (Yao et al., 2023),
reflection (Shinn et al., 2023), and disaggregation (Xu et al.,

| Personas                | Tasks                                 | Metrics   |
|-------------------------|---------------------------------------|-----------|
| SRE                     | Diagnosis                             | pass@1, Fault Localization, Fault Propagation Chain, Mean Time to Diagnosis           |
| Mitigation              | pass@1, Mean Time to Repair           |           |
| CISO                    | Collect evidence                      | pass@1    |
| Scan assessment posture | pass@1, Time to Process                                       |           |
| FinOps                  | Identify inefficiency                 | pass@1    |
| Mitigate inefficiency   | pass@1, Hourly infra cost, Efficiency |           |
| Data Insights           | pass@1, Token utilization                                       |           |
| Anomaly Detection       | F1 Score, rank score                  |           |

2023). Reflection techniques include syntax checking/linting, semantic validation (Xie et al., 2024a), and llm-as-ajudge (Zheng et al., 2023). We use the open-source CrewAI framework (cre) to create and manage agents. The agents can be configured to use various LLMs either through watsonx, Azure, or vLLM. Each agent is initialized with a prompt that describes its goal, the context, the tasks, and the expected output format. In-context learning examples are included to guide the agent and demonstrate tool usage. Agents use tools to interact with the environment for information gathering.

Logs, traces, and metrics collected during the diagnosis process would overwhelm the context window of any LLM currently available due to large volume of data. Therefore, agents targeting the SRE or FinOps persona are equipped with specialized tools to interact with the environment (refer to Figure 3): 1) NL2Traces to extract trace data in a structured format, 2) NL2Metrics to analyze key system metrics, 3) NL2Logs to parse log data effectively, 4) NL2Kubectl to perform Kubernetes-specific operations, and a summarization tool to condense extensive data into actionable insights. For example, the agent may use the NL2Kubectl tool to
"list all of the pods in the default namespace." In turn, the NL2Kubectl tool uses an LLM to transform the utterance into an executable command: "kubectl get pods -n default." Similarly, the compliance assessment required for new regulations and technologies, with the evidence and diverse policy languages, would be overwhelming if submitted directly to LLMs. The compliance agents designed for CISO compliance assessment automation are equipped with specialized tools. These tools include capabilities to 1) generate policies such as Kyverno or OPA Rego Policy as Code starting from natural language specifications, 2) generate scripts for the collection of evidence, 3) access code repositories such as git to facilitate GitOps workflows for code management, and 4) deploy and execute the generated policies to accomplish the assessment task.

## 3.3. Leaderboard

ITBench includes a leaderboard to promote reproducibility and comparative analysis, following the AI common task framework (Donoho, 2019; Varshney et al., 2019). The leaderboard offers a predefined, extensible set of performance metrics designed to provide clear insights into agent performance relative to the evaluation criteria. ITBench devises scoring methods for partially correct solutions to provide meaningful feedback for summative assessments. This comprehensive approach establishes a new standard for evaluating and advancing AI-driven solutions in IT automation. For each scenario that an agent works on, upon task completion, the ITBench records the final system state, which is then used at the end of all scenario runs along with the pre-defined ground truth data to validate how well the agent performed across all the scenarios.

We are open-sourcing a small subset (11 out of 102) of scenarios along with the baseline agents to help the community become familiar with ITBench through practical examples. We reserve the remaining scenarios in ITBench to benchmark and evaluate the submitted agentic solutions.

## 4. Results 4.1. Evaluation Setup

To understand the impact of reasoning and planning capabilities of LLMs on ITBench scenarios, we instantiate our agents using different LLM models, both for natural language reasoning and code generation. Specifically, we employ GPT-4o (checkpoint version 2024-11-20), Llama3.3-70B-instruct, Llama-3.1-8B-instruct, and Granite-3.18B-instruct for tasks that rely on natural language understanding and reasoning. For code-focused use cases, we utilize GPT-4o-mini, Llama-3.1-405b-instruct, and Mixtral8x7b-instruct. All models use a context window of 128K tokens, enabling them to process more extensive input sequences. We conduct our experiments primarily on AWS EC2 instances (m4.xlarge), although ITBench can also be readily deployed on a consumer-grade laptop using a pseudo-cluster, thus making it easier to develop AI agents (Appendix C.4.1) Below, we provide an overview of our baseline agents' performance across ITBench scenarios for SRE, CISO, and FinOps. Our findings indicate that both open-source and proprietary models often struggle with real-world tasks, underscoring the importance of benchmarks that push the limits of reasoning and planning in foundation models. For

| Models                                                           | Diagnosis                              | Mitigation                                                                                                       |             |                |              |                 |
|------------------------------------------------------------------|----------------------------------------|------------------------------------------------------------------------------------------------------------------|-------------|----------------|--------------|-----------------|
| pass@1 (%)↑                                                      | FL (NTAM)↑                             | FPC (NTAM)↑                                                                                                      | MTTD (s)↓   | pass@1 (%)↑    | MTTR (s)↓    |                 |
| granite-3.1-8B-instruct                                          | 3.57 ± 0.94                            | 0.16 ± 0.02                                                                                                      | 0.19 ± 0.02 | 259.92 ± 65.01 | 0.24 ± 0.25  | 845.50 ± -      |
| llama-3.1-8B-instruct                                            | 0.99 ± 0.51                            | 0.07 ± 0.01                                                                                                      | 0.08 ± 0.01 | 57.50 ± 2.05   | 1.98 ± 0.68  | 245.13 ± 40.66  |
| llama-3.3-70B-instruct                                           | 3.10 ± 0.84                            | 0.16 ± 0.02                                                                                                      | 0.16 ± 0.02 | 191.85 ± 31.34 | 3.33 ± 0.90  | 776.27 ± 252.87 |
| gpt-4o                                                           | 13.81 ± 1.67                           | 0.39 ± 0.05                                                                                                      | 0.34 ± 0.03 | 72.44 ± 4.71   | 11.43 ± 1.52 | 282.47 ± 30.04  |
| 1 42 scenarios (21 scenarios with traces and 21 without traces). | 2 10 runs per scenario per model.      | 3 pass@1 values are shown as percentages. '—' indicates missing                                                  |             |                |              |                 |
| data.                                                            | 4 std error for each metric is listed. | 5 FL (NTAM) = Normalized topology-aware metric for root cause, FPC (NTAM) = Normalized topology-aware metric for |             |                |              |                 |

more comprehensive results and detailed scenario-level discussions, please refer to Appendix C (SRE), Appendix D (CISO), and Appendix E (FinOps).

## 4.2. Overall Results

Table 4, Table 5, and Table 6 show the performance of SRE-agent, CISO-agent, and FinOps-agent respectively. SRE. We measure the efficiency of SRE-Agent on its ability to diagnose and mitigate production incidents (e.g., "a high error rate on frontend service"). Diagnosis efficiency is measured using pass@1(Chen et al., 2021) (i.e., identifying the cause as mentioned in ground truth), NTAM (Normalized Topology-Aware Metric) for root cause and fault propagation chain, and time to diagnosis.1 Mitigation efficiency is measured in terms of pass@1
(i.e., whether the alert was cleared) and mean time to repair.

As shown in Table 4, across all SRE scenarios, GPT-4o consistently outperforms the other models, achieving the highest pass@1 scores for diagnosis (13.81%) and mitigation (11.43%), as well as the highest score on NTAM (FL and FPC) metrics. Llama-3.3-70B ranks second overall, trailing GPT-4o on most metrics. The 8B models have lower mitigation success rate. Surprisingly, Granite-3.1-8B (without any specialized finetuning) achieves higher accuracy than Llama-3.3-70B on the diagnosis task. Removing trace data can drastically reduce success rates (see Table 20 and Table 21 in Appendix). For instance, GPT- 4o's pass@1 in diagnosis falls from 13.81% with traces to 9.52% without them, and mitigation plummets to 2.86%. This highlights the critical role of system observability in SRE, which ITBench can evaluate under varying conditions. Because there is no perfect observability in practice, how to guide SRE-agents to collect new observability data and to help SRE-agents reason about failures with incomplete observability is an important but open problem.

1NTAM is Normalized topology-aware metric that measures the quality of the predicted root cause and fault propagation chains using a system and application topology. Refer to Appendix C.6.3.

CISO. We measure the efficacy of our agents across the four scenario classes introduced in Table 2. Each *scenario_class* imposes a distinct set of CIS-benchmarks requirements (e.g.,
"minimize the admission of containers wishing to share the host network namespace"), has a specific level of complexity (e.g., Easy, Medium, or Hard), and generates scenariospecific code artifacts. The efficacy of CISO-agents is measured based on the ability to detect artifact misconfigurations (aka non-compliance, e.g., no minimum count of containers sharing namespace, or the count is above the threshold), or confirm proper configurations (aka compliance), within the varied environments of the scenario classes randomly injected with misconfigurations. Notably, GPT-based models dominate on both pass@1 and Time to Process metrics. The pass@1 is nearly two times better than second-best performing model, while the TTP shows a handling of the scenarios in the minimal time across our scenario classes. FinOps. In addition to the standard event-driven scenarios, ITBench was extended to support non-alert-driven scenarios for the FinOps persona, demonstrating its extensibility. In particular, we added data insights and anomaly detection scenarios to ITBench. Table 6 presents our results in all FinOps usecases. We report pass@1 score for data insights, diagnosis, and mitigation tasks, and F1 score and rank score for anomaly detection. F1 score measures the precision and recall abilities of the agent to identify anomalous costs with regard to the ground truth. The rank score measures the relative ranking of the anomalies as determined by the agent with regard to the ground truth ranking. GPT-4o consistently outperforms all other models, achieving a 33% pass rate for diagnosing the origin of the cost increase alert, 29% accuracy in data insights scenarios, and 0.6 F1 score in anomaly detection. Refer to Appendix E.5 for futher details.

## 4.3. Impact Of Scenario Complexity

SRE. We categorize scenarios as Easy, Medium, or Hard based on factors such as fault propagation chain length, number of resolution steps, and the diversity of technolo-

| Models                  | Scenario pass@1 (%) ↑            | O/A pass@1 (%) ↑                          | TTP (s) ↓                        |              |              |               |
|-------------------------|----------------------------------|-------------------------------------------|----------------------------------|--------------|--------------|---------------|
| kyverno                 | k8s-opa                          | rhel-opa                                  | kyverno-update                   |              |              |               |
| granite-3.1-8B-instruct | 7.84 ± 3.84                      | 0.00 ± 0.00                               | 0.00 ± 0.00                      | 1.59 ± 1.58  | 1.71 ± 0.76  | 197.03 ± 2.52 |
| mixtral-8x7B-instruct   | 7.35 ± 3.19                      | 1.43 ± 1.42                               | 0.00 ± 0.00                      | 1.29 ± 4.34  | 3.94 ± 1.03  | 120.63 ± 3.77 |
| llama-3.1-8B-instruct   | 8.57 ± 3.37                      | 0.00 ± 0.00                               | 0.00 ± 0.00                      | 7.46 ± 3.23  | 3.59 ± 1.07  | 121.49 ± 3.00 |
| llama-3.3-70B-instruct  | 18.46 ± 4.94                     | 0.00 ± 0.00                               | 1.43 ± 2.88                      | 8.06 ± 3.50  | 9.32 ± 1.67  | 189.61 ± 2.71 |
| mistral-large-2         | 6.56 ± 3.20                      | 22.73 ± 5.32                              | 7.23 ± 2.88                      | 10.45 ± 3.77 | 11.55 ± 1.95 | 167.98 ± 3.42 |
| llama-3.1-405B-instruct | 16.22 ± 4.32                     | 20.83 ± 4.86                              | 8.75 ± 3.26                      | 3.17 ± 2.22  | 12.46 ± 1.98 | 178.89 ± 3.37 |
| gpt-4o-mini             | 16.18 ± 4.54                     | 43.10 ± 6.99                              | 30.38 ± 5.43                     | 9.43 ± 4.08  | 25.19 ± 2.80 | 102.40 ± 3.70 |
| gpt-4o                  | 40.28 ± 5.99                     | 39.34 ± 6.55                              | 7.61 ± 2.81                      | 17.74 ± 4.92 | 24.74 ± 2.64 | 101.29 ± 3.81 |
| 1 50 scenarios.         | 2 8 runs per scenario per model. | 3 pass@1 values are shown as percentages. | 4 TTP Time to process (seconds). |              |              |               |

| Non-Alert-Driven Scenarios              | Alert-Driven Scenarios                                                                              |                   |              |              |    |
|-----------------------------------------|-----------------------------------------------------------------------------------------------------|-------------------|--------------|--------------|----|
| Models                                  | Data Insights                                                                                       | Anomaly Detection | Diagnosis    | Mitigation   |    |
| pass@1 ↑                                | F1 Score ↑                                                                                          | Ranking ↑         | pass@1 (%) ↑ | pass@1 (%) ↑ |    |
| granite-3.1-8B-instruct                 | 14                                                                                                  | 0.4 ± 0.07        | 0.3 ± 0.00   | 0            | 0  |
| llama-3.1-8B-instruct                   | 0                                                                                                   | 0.4 ± 0.03        | 0.4 ± 0.00   | 0            | 0  |
| llama-3.3-70B-instruct                  | 29                                                                                                  | 0.0               | 0.0 ± 0.00   | 16.6         | 0  |
| gpt-4o                                  | 29                                                                                                  | 0.6 ± 0.00        | 0.5 ± 0.00   | 33           | 0  |
| pass@1 values are shown as percentages. | The Data-Insight evaluations exhibit zero variance because we use a fixed dataset and configure the |                   |              |              |    |

gies involved, as described in Equation (6). Our results show that success rates (pass@1) clearly decline as the scenario_complexity increases. Even the best performing model, GPT-4o, diagnosed only 36%, 7.73%, 5% of Easy/Medium/Hard cases (Table 18) and mitigated just 21%, 12.27%, 0% (Table 19). None of the models could mitigate the Hard scenarios, even though over 50% of Easy scenarios were mitigated. Notably, GPT-4o is the only model that successfully diagnosed multiple Hard scenarios.

CISO. The complexity of the CISO scenarios is directly mapped to scenario classes. For example, scenario_complexity of Kyverno scenarios is Easy, scenario_complexity of k8s-opa and rhel-opa is Medium, while scenario_complexity of Kyverno-update scenarios is Hard. All models struggle, as expected, as the difficulty of the scenarios increases from the Easy *kyverno* class to the Hard kyverno-upadate class. FinOps. Currently, ITBench includes 2 Easy, 3 Medium, and 5 Hard scenarios. None of the models were able to resolve the Hard scenarios. GPT-4o performs better in anomaly detection and alert-driven scenarios, while the LLaMA-3.3-70B-Instruct model achieves comparable performance to GPT-4o in data insight scenarios.

## 5. A Case Study On Sre-Agent Failures

Understanding the decision process of LLM-based agents is challenging due to the complexity of agentic systems but is feasible through detailed agent trajectory logging and a structured prompting framework. We log each input and output for planning agents and tools, including the ReAct-style
"Thought" step, enabling us to distinguish between highlevel reasoning errors (e.g., flawed strategy) and low-level tool errors (e.g., malformed commands), enabling practical analysis and guiding design improvements.

## 5.1. **Analyzing Lower-Level Tool Calling And Execution**

Figure 5 shows tool usage and failure types across models. NL2Kubectl dominates usage, suggesting overreliance. Encouraging the agent to make a more balanced use of other tools, such as NL2Traces, could be useful, especially when kubectl commands alone are insufficient. Smaller models (e.g., granite-3.1-8B-instruct, llama-3.1-8B-instruct) show more invalid tool calls, syntax errors, and repeated invocations, indicating lower accuracy and efficiency.

## 5.2. Quantitative Analysis Of High-Level Reasoning

We quantify reasoning by aligning each exploration path with the ground-truth fault-propagation chain. An effective agent is expected to focus its exploration around this chain, while significant deviations may signal reasoning

invalid tool getalerts nl2traces nl2kubectl nl2metrics nl2logs granite-3.1-8B-inst.

llama-3.1-8B-inst.

llama-3.3-70B-inst.

0 500 1000 1500 2000 2500 Nu m be r of Act ion s gpt-4o granite-3.1-8B-inst.

llama-3.1-8B-inst.

llama-3.3-70B-inst.

gpt-4o granite-3.1-8B-inst.

llama-3.1-8B-inst.

llama-3.3-70B-inst.

gpt-4o granite-3.1-8B-inst.

llama-3.1-8B-inst.

llama-3.3-70B-inst.

gpt-4o granite-3.1-8B-inst.

llama-3.1-8B-inst.

llama-3.3-70B-inst.

gpt-4o granite-3.1-8B-inst.

llama-3.1-8B-inst.

llama-3.3-70B-inst.

gpt-4o Total Incorrect Tool Argument Repeated Usage Syntax Error Execution Failure Successful Execution
flaws. Based on this insight, we introduce two evaluation metrics: (i) **Detoured Services**: |Vvisited \ Vgt|, the number of visited services that are not on the ground-truth chain (smaller → more focused search); and (ii) Relative Covered Services:
|Vvisited ∩ Vgt| |Vgt|
, the fraction of ground-truth services visited (closer to 1→better coverage). As shown in Figure 6, successful diagnosis trajectories show fewer detours and higher coverage than unsuccessful ones, validating the metrics. Among successful trajectories, GPT-4o shows detours (Kolmogorov–Smirnov p-value ≥0.123) and coverage (p-value≥0.089) that are comparable to other models (i.e., Granite-3.1-8B-Instruct, Llama-3.1-8B-Instruct, and Llama-3.3-70B-Instruct), while achieving higher coverage than Llama-3.1-8B-Instruct (pvalue=0.024). This suggests that successful agents tend to follow similar reasoning patterns. For unsuccessful trajectories, GPT-4o significantly surpasses all baselines, showing both fewer detours (p-value≤0.001) and greater coverage (p-value≤ 0.011). These results underscore ITBench's utility in revealing insightful patterns in agent reasoning and overall performance.

(a) Detoured Services (b) Relative Covered Services

## 6. Discussion And Conclusion

We presented ITBench, the first framework and experimental platform to benchmark AI Agents for IT automation tasks. ITBench strives to capture the complexity of modern IT systems and the diversity of IT tasks. The reproducibility of ITBench ensures the community-driven effort despite inherent nondeterminism of large-scale IT systems.

One of the key design principles of ITBench is ensuring its flexibility to support diverse areas of different IT systems and its extensibility to new scenarios. While the current scope of ITBench is comprehensive and representative, we plan to further enrich the benchmark suites by adding other important processes essential to modern IT automation. Furthermore, we plan to expand our benchmarking beyond event-triggered scenarios. We are actively working to expand scenario coverage for the supported processes and promote growth through open-community contributions. We invite the community to reproduce their real-world-inspired incidents in a synthetic sandboxed environment leveraging the ITBench. We expect that everyone contributing can bring their expertise to the table. We expect ITBench to drive the innovations of AI agentbased techniques with a direct impact on the safety, efficiency, and intelligence of today's IT infrastructures. With ITBench, we are starting to explore many deep, exciting open problems: How to develop domain-specific AI agents that specialize in certain types of IT tasks? How to orchestrate multiple agents with various expertise to collaborate on bigger projects? How can we ensure safety of agent-driven solutions? How can we effectively use human-in-the-loop while developing diverse adaptive agents? We invite everyone to participate in answering these questions and realizing the vision of using AI agents to automate critical IT tasks.

## Impact Statement Ethics & Broader Impacts

This research presents a novel benchmarking framework to measure the performance of AI agents across a wide variety of complex and real-life IT tasks, which has the potential to be a key enabler for AI-driven IT automation that is correct, safe, and fast. While the primary focus is on advancing the field of machine learning, as this effort is an open source framework built with open source technologies, it allows organizations with proprietary technologies to use it for developing and benchmarking their solutions more effectively. It also encourages mindsharing in the community and lowers the barrier to innovate in IT domain. Agents that interact with the system pose several risks. We identify three main risks that could arise when building and using a ITBench and associated agents, then discuss how we incorporates measures that mitigate such problems. First is the security risks that come with executing LM- generated code/commands on the system. Examples include executing commands like kubectl delete node and rm -rf asset/. To defend against this, we containerize the agents and also provide a self-contained Kubernetes environment to create various scenarios. Second, if the wider community develops an interest in IT- Bench and associated agents and builds upon it, it is also possible that illegitimate evaluation datasets or infrastructure can be used to inject testing devices with malicious code or instructions to generate malicious code. For instance, an unofficial repository claiming to host an inference/evaluation harness for ITBench and associated agents could include a task instance with an issue description that tells the LM agent to build key logging functionality and store it in a hidden folder. To eliminate confusion and reduce the possibility of such an event, we provide clear guidelines listed on our GitHub repositories, data stores, and websites indicating the official repositories and channels that we actively maintain. We also encourage third parties to incorporate any improvements into our codebase and help with integrating such contributions. Lastly are the consequences of ITBench agents being deployed in the real world. Prior works have conceptualized and put forth prototypes of agents that can carry out offensive security measures. It is also not difficult to imagine that a system like SRE-Agent can be incorporated into pipelines, resulting in the production of malicious code and libraries.

The strong performance of agents on ITBench implies that future AI systems will likely be increasingly adept in the aforementioned use cases. Releasing ITBench agents as open source agents can support research toward designing sound, effective constraints for what software engineering agents are permitted to do. It can also serve as a system that legal experts and policy-making entities can experiment with to shape the future of what AI-driven end-to-end software engineering could look like.

## Reproducibility

To help the greater community reproduce the results presented in this paper and build on the ITBench, we open source all of our resources that were created for this project. The source code for the interactive pipeline, context management logic, command implementations, interface design, and everything else is entirely available in a GitHub repository. We provide extensive text and video documentation describing how to run and modify different parts of the codebase. Practitioners should be able to easily recover our findings by running the agent with simple scripts. The results presented in the main and supplementary parts of this paper can be fully obtained by following instructions in the repositories. Finally, we also maintain an active online help forum to assist with any reproduction problems or questions about how to build on ITBench.

## Acknowledgements

We would like to thank everyone at IBM and the University of Illinois at Urbana-Champaign who supported this project but are not on the author list. In particular, we are grateful to Ravishankar K. Iyer and Deming Chen for their guidance and encouragement, as well as to all who shared their excitement, provided feedback on early prototypes, and collaborated with the core team across various aspects of this work. We gratefully acknowledge the core contributors— Pavankumar Murali, Paulina Toro Isaza, and Xi Yang—for their decisive role in addressing reviewer feedback. Pavankumar designed the FinOps anomaly-detection scenarios, while Paulina and Xi carried out the agent-trajectory analyses. The work is supported in part by the IBM-Illinois Discovery Accelerator Institute (IIDAI) and National Science Foundation (NSF) CNS-2145295. In addition, we acknowledge the support of our colleagues in IBM Instana: Ameet Rahane, Marc Palaci-Olgun, Guangya Liu, Brad Blancett, Chad Holliday, Arthur De Magalhaes, Ragu Kattinakere, Chris Bailey, Isabell Sippili, and Danilo Florissi; IBM Granite and Data Model Factory: Hui Wu and Bing Zhang; IBM Emerging Technology Engineering:
Carlos Fonseca, Aditya Gidh, Mike Sava, Bill Rippon, and Danny Barnett; IBM UX Research: James Sutton, Connor Leech, and Justin McNair; IBM Research: Michal Shmueli- Scheuer, Lilach Edelstein, and Roy Bar-Haim.

## References

CISO Compliance Assessment Agent. https:
//github.com/IBM/itbench-ciso-caaagent?tab=readme-ov-file, a. Accessed: 2025-01-30.

T. Ahmed, S. Ghosh, C. Bansal, T. Zimmermann, X. Zhang, and S. Rajmohan. Recommending root-cause and mitigation steps for cloud incidents using large language models. In 2023 IEEE/ACM 45th International Conference on Software Engineering (ICSE), pages 1737–1749, 2023a. doi: 10.1109/ICSE48619.2023.00149.

CISO Compliance Assessment Agent. https:
//github.com/IBM/itbench-samplescenarios/tree/main/ciso, b. Accessed:
2025-01-30.

T. Ahmed, S. Ghosh, C. Bansal, T. Zimmermann, X. Zhang, and S. Rajmohan. Recommending root-cause and mitigation steps for cloud incidents using large language models. In 2023 IEEE/ACM 45th International Conference on Software Engineering (ICSE), pages 1737–1749.

IEEE, 2023b.

Ansible. https://www.redhat.com/en/
technologies/management/ansible, a.

Accessed: 2025-01-30.

OPA Gatekeeper. https://kubernetes.io/blog/
2019/08/06/opa-gatekeeper-policy-andgovernance-for-kubernetes/, b. Accessed:
2025-01-30.

B. Arzani, S. Ciraci, B. T. Loo, A. Schuster, and G. Outhred.

Taking the blame game out of data centers operations with netpoirot. In Proceedings of the 2016 ACM SIGCOMM Conference, pages 440–453, 2016.

Kyverno, Policy as Code, Simplified! https://
kyverno.io, c. Accessed: 2025-01-30.

S. Ashok, V. Harsh, B. Godfrey, R. Mittal, S. Parthasarathy, and L. Shwartz. Traceweaver: Distributed request tracing for microservices without application modification. In Proceedings of the ACM SIGCOMM 2024 Conference, pages 828–842, 2024.

Open Policy Agent. https://github.com/openpolicy-agent/opa, d. Accessed: 2025-01-30.

NIST Cybersecurity Framework 2.0 Quick-Start Guide for Creating and Using Organizational Profiles. https: //www.nist.gov/cyberframework. Accessed: 2025-01-30.

A. Avizienis, J.-C. Laprie, and B. Randell. Basic Concepts and Taxonomy of Dependable and Secure Computing. IEEE Transactions on Dependable and Secure Computing (TDSC), 1(1):1–23, Jan. 2004.

Crew ai. https://www.crewai.com/. Accessed:
2025-01-30.

F. Bagehorn, J. Rios, S. Jha, R. Filepp, L. Shwartz, N. Abe, and X. Yang. A fault injection platform for learning aiops models. In Proceedings of the 37th IEEE/ACM International Conference on Automated Software Engineering, pages 1–5, 2022.

Datadog. https://www.datadoghq.com/. Accessed: 2024-11-30.

Dynatrace. https://www.dynatrace.com/. Accessed: 2024-11-30.

L. A. Barroso, U. Hölzle, and P. Ranganathan. The Datacenter as a Computer: Designing Warehouse-Scale Machines. Morgan and Claypool Publishers, 3 edition, 2018.

Finops sample data. https://github.com/FinOps-
Open-Cost-and-Usage-Spec/FOCUS-
Sample-Data. Accessed: 2025-05-30.

B. Beyer, N. R. Murphy, D. K. Rensin, K. Kawahara, and S. Thorne. Site Reliability Workbook: Practical Ways to Implement SRE. O'Reilly Media Inc., Aug. 2018.

Grafana. https://grafana.com/. Accessed: 202501-30.

B. Bogin, K. Yang, S. Gupta, K. Richardson, E. Bransom, P. Clark, A. Sabharwal, and T. Khot. Super: Evaluating agents on setting up and executing tasks from research repositories. *ArXiv*, abs/2409.07440, 2024.

Instana. https://www.ibm.com/products/
instana. Accessed: 2024-11-30.

Jaeger. https://www.jaegertracing.io/. Accessed: 2025-01-30.

L. Boisvert, M. Thakkar, M. Gasse, M. Caccia, T. L. S. D.

Chezelles, Q. Cappart, N. Chapados, A. Lacoste, and A. Drouin. Workarena++: Towards compositional planning and reasoning-based common knowledge work tasks, 2024.

Kubernetes. https://kubernetes.io/. Accessed:
2025-01-30.

Grafana Loki. https://github.com/grafana/
loki. Accessed: 2025-01-30.

C. Boulton. Rising cloud costs leave CIOs seeking ways to cope. https://www.cio.com/
article/3496509/rising-cloud-costs-
Prometheus. https://prometheus.io/. Accessed:
2025-01-30.

leave-cios-seeking-ways-to-cope.html. Accessed: 2025-01-30.

C. Di Martino, F. Baccanico, J. Fullop, W. Kramer, Z. Kalbarczyk, and R. Iyer. Lessons Learned From the Analysis of System Failures at Petascale: The Case of Blue Waters. In Proceedings of the 44th Annual IEEE/I- FIP International Conference on Dependable Systems and Networks (DSN'14), 2014.

K. Budhathoki, L. Minorics, P. Blöbaum, and D. Janzing.

Causal structure-based root cause analysis of outliers. In International Conference on Machine Learning, pages 2357–2369. PMLR, 2022.

Y. Diao, D. Horn, A. Kipf, O. Shchur, I. Benito, W. Dong, D. Pagano, P. Pfeil, V. Nathan, M. Narayanaswamy, and T. Kraska. Forecasting algorithms for intelligent resource scaling: An experimental analysis. In Proceedings of the 2024 ACM Symposium on Cloud Computing, pages 126–143, 2024.

B. Burns, B. Grant, D. Oppenheimer, E. Brewer, and J. Wilkes. Borg, Omega, and Kubernetes. Communications of the ACM, 59(5):50–57, May 2016.

S. Chakraborty, S. Garg, S. Agarwal, A. Chauhan, and S. K.

Saini. Causil: Causal graph for instance level microservice data. In *Proceedings of the ACM Web Conference* 2023, pages 2905–2915, 2023.

D. Donoho. Comments on Michael Jordan's essay "the AI
revolution hasn't happened yet". Harvard Data Science Review, June 2019.

M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, and J. K. et. al. Evaluating large language models trained on code, 2021.

A. Drouin, M. Gasse, M. Caccia, I. H. Laradji, M. D.

Verme, T. Marty, L. Boisvert, M. Thakkar, Q. Cappart, D. Vazquez, N. Chapados, and A. Lacoste. Workarena: How capable are web agents at solving common knowledge work tasks?, 2024.

Y. Chen, H. Xie, M. Ma, Y. Kang, X. Gao, L. Shi, Y. Cao, X. Gao, H. Fan, M. Wen, et al. Empowering practical root cause analysis by large language models for cloud incidents. *arXiv preprint arXiv:2305.15778*, 2023.

H. Fang, T. Tao, and C. Zhai. Diagnostic evaluation of information retrieval models. ACM Transactions on Information Systems (TOIS), 29(2):1–42, 2011.

Y. Chen, M. Shetty, G. Somashekar, M. Ma, Y. Simmhan, J. Mace, C. Bansal, R. Wang, and S. Rajmohan. Aiopslab: A holistic framework for evaluating ai agents for enabling autonomous cloud. November 2024a.

Y. Fangkai, L. Wang, Z. Xu, J. Zhang, L. Li, B. Qiao, C. Couturier, C. Bansal, S. Ram, Z. Ma, I. Goiri, E. Cortez, T. Yang, V. Ruehle, S. Rajmohan, Q. Lin, and D. Zhang. Snape: Reliable and low-cost computing with mixture of spot and on-demand vms. In Proceedings of the International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3, pages 631–643, 2023.

Y. Chen, H. Xie, M. Ma, Y. Kang, X. Gao, L. Shi, Y. Cao, X. Gao, H. Fan, M. Wen, J. Zeng, S. Ghosh, X. Zhang, C. Zhang, Q. Lin, S. Rajmohan, D. Zhang, and T. Xu. Automatic Root Cause Analysis via Large Language Models for Cloud Incidents. In Proceedings of the 19th European Conference on Computer Systems (EuroSys'24), Apr. 2024b.

B. Feng, Z. Ding, and C. Jiang. Fast: A forecasting model with adaptive sliding window and time locality integration for dynamic cloud workloads. In IEEE Transactions on Services Computing, pages 1184–1197, 2023.

Y. Chen, H. Xie, M. Ma, Y. Kang, X. Gao, L. Shi, Y. Cao, X. Gao, H. Fan, M. Wen, et al. Automatic root cause analysis via large language models for cloud incidents.

In Proceedings of the Nineteenth European Conference on Computer Systems, pages 674–688, 2024c.

R. Fonseca, G. Porter, R. H. Katz, and S. Shenker. {X-
Trace}: A pervasive network tracing framework. In 4th USENIX Symposium on Networked Systems Design & Implementation (NSDI 07), 2007.

CIS. Center for Internet Security Benchmarks List. https:
//www.cisecurity.org/cis-benchmarks, 2024. Accessed: 2025-01-30.

C. for Internet Security (CIS). *CIS Benchmarks*. Center for Internet Security, 2024. Technical guidelines for secure system configurations.

O. Community. Opentelemetry astronomy shop demo. https://opentelemetry.io/docs/ demo/, 2024. Accessed: 2025-01-30.

D. Ford, F. Labelle, F. I. Popovici, M. Stokely, V.-A. Truong, L. Barroso, C. Grimes, and S. Quinlan. Availability in Globally Distributed Storage Systems. In *Proceedings* of the 9th USENIX Conference on Operating Systems Design and Implementation (OSDI'10), Oct. 2010.

J. Dean. Designs, Lessons and Advice from Building Large Distributed Systems. In *Proceedings of the the 3rd Large* Scale Distributed Systems and Middleware (LADIS'09),
Oct. 2009.

F. Foundation. Using efficiency metrics to evaluate cloud optimization and value between parts of the organization or against industry peers to inform decision-making and align finops with business objectives. https://www.finops.org/framework/ capabilities/benchmarking/, 2025a. Accessed: 2025-01-30.

F. Foundation. Finops kpis. https://www.finops.

org/wg/finops-kpis/, 2025b. Accessed: 202501-30.

J. Gao, N. Yaseen, R. MacDavid, F. V. Frujeri, V. Liu, R. Bianchini, R. Aditya, X. Wang, H. Lee, D. Maltz, et al. Scouts: Improving the diagnosis process through domain-customized incident routing. In Proceedings of the Annual conference of the ACM Special Interest Group on Data Communication on the applications, technologies, architectures, and protocols for computer communication, pages 253–269, 2020.

S. Ghosh, M. Shetty, C. Bansal, and S. Nath. How to fight production incidents?: an empirical study on a large-scale cloud service. In *Proceedings of the 13th Symposium on* Cloud Computing (SoCC'22), Nov. 2022.

H. S. Gunawi, M. Hao, T. Leesatapornwongsa, T. Patanaanake, T. Do, J. Adityatama, K. J. Eliazar, A. Laksono, J. F. Lukman, V. Martin, and A. D. Satria. What bugs live in the cloud? a study of 3000+ issues in cloud systems.

In Proceedings of the 5th ACM Symposium on Cloud Computing (SoCC'14), Nov. 2014.

H. S. Gunawi, M. Hao, R. O. Suminto, A. Laksono, A. D.

Satria, J. Adityatama, and K. J. Eliazar. Why Does the Cloud Stop Computing? Lessons from Hundreds of Service Outages. In Proceedings of the 7th ACM Symposium on Cloud Computing (SoCC'16), Oct. 2016.

H. S. Gunawi, R. O. Suminto, R. Sears, C. Golliher, S. Sundararaman, X. Lin, T. Emami, W. Sheng, N. Bidokhti, C. McCaffrey, G. Grider, P. M. Fields, K. Harms, R. B. Ross, A. Jacobson, R. Ricci, K. Webb, P. Alvaro, H. B. Runesha, M. Hao, and H. Li. Fail-Slow at Scale: Evidence of Hardware Performance Faults in Large Production Systems. In Proceedings of the 16th USENIX Conference on File and Storage Technologies (FAST'18), Feb. 2018.

C. Guo, L. Yuan, D. Xiang, Y. Dang, R. Huang, D. Maltz, Z. Liu, V. Wang, B. Pang, H. Chen, Z.-W. Lin, and V. Kurien. Pingmesh: A large-scale system for data center network latency measurement and analysis. SIG- COMM Comput. Commun. Rev., 45(4):139–152, Aug.

2015. ISSN 0146-4833. doi: 10.1145/2829988.2787496.

Harvard Business Review Research Report. Taming it complexity through effective strategies and partnerships. https://hbr.org/sponsored/ 2022/11/taming-it-complexity-througheffective-strategies-and-partnerships, 2022.

P. H. Hochschild, P. Turner, J. C. Mogul, R. Govindaraju, P. Ranganathan, D. E. Culler, and A. Vahdat. Cores that don't count. In *Proceedings of the 18th Workshop on Hot* Topics in Operating Systems (HotOS'21), June 2021.

J. Huang, X. Chen, S. Mishra, H. S. Zheng, A. W. Yu, X. Song, and D. Zhou. Large language models cannot selfcorrect reasoning yet. *arXiv preprint arXiv:2310.01798*, 2023.

L. Huang, M. Magnusson, A. B. Muralikrishna, S. Estyak, R. Isaacs, A. Aghayev, T. Zhu, and A. Charapko. Metastable Failures in the Wild. In Proceedings of the 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI'22), July 2022.

IDC. Storm clouds ahead: Missed expectations in cloud computing. https://blogs.idc.com/ 2024/10/28/storm-clouds-ahead-missedexpectations-in-cloud-computing/, 2024. Blog post, published on 28 October 2024.

A. Ikram, S. Chakraborty, S. Mitra, S. Saini, S. Bagchi, and M. Kocaoglu. Root cause analysis of failures in microservices through causal discovery. Advances in Neural Information Processing Systems, 35:31158–31170, 2022.

S. Jha, S. Cui, S. S. Banerjee, T. Xu, J. Enos, M. Showerman, Z. T. Kalbarczyk, and R. K. Iyer. Live forensics for hpc systems: A case study on distributed storage systems. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–16. IEEE, 2020.

C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and K. R. Narasimhan. SWE-bench: Can language models resolve real-world github issues? In *The Twelfth* International Conference on Learning Representations, 2024.

A. S. John. Idc predicts 80% of cios to leverage ai and automation for business agility and insights by 2028, 2024.

E. Jonas, J. Schleier-Smith, V. Sreekanti, C.-C. Tsai, A. Khandelwal, Q. Pu, V. Shankar, J. M. Carreira, K. Krauth, N. Yadwadkar, J. Gonzalez, R. A. Popa, I. Stoica, and D. A. Patterson. Cloud Programming Simplified:
A Berkeley View on Serverless Computing. Technical Report UCB/EECS-2019-3, University of California at Berkeley, Feb. 2019.

S. Kapoor, B. Stroebl, Z. S. Siegel, N. Nadgir, and A. Narayanan. Ai agents that matter. https://arxiv. org/abs/2407.01502, 2024.

M. G. Kendall. The treatment of ties in ranking problems.

Biometrika, 33(3):239–251, 1945.

S. Kendrick. What Takes Us Down? *USENIX ;login:*, 37
(5):37–45, Oct. 2012.

S. M. Kerner. Crowdstrike outage explained: What caused it and what's next. https://www.techtarget.

com/whatis/feature/Explaining-thelargest-IT-outage-in-history-andwhats-next\#:~:text=What%20might%20be% 20considered%20the,Fortune%20500% 20companies%20%245.4%20billion., 2024.

J. Y. Koh, R. Lo, L. Jang, V. Duvvur, M. C. Lim, P.-Y. Huang, G. Neubig, S. Zhou, R. Salakhutdinov, and D. Fried. Visualwebarena: Evaluating multimodal agents on realistic visual web tasks, 2024.

J. B. Leners, H. Wu, W.-L. Hung, M. K. Aguilera, and M. Walfish. Detecting failures in distributed systems with the falcon spy network. In Proceedings of the Twenty- Third ACM Symposium on Operating Systems Principles, pages 279–294, 2011.

H. Liu, S. Lu, M. Musuvathi, and S. Nath. What bugs cause production cloud incidents? In Proceedings of the 16th Workshop on Hot Topics in Operating Systems (HotOS'19), May 2019.

M. Liu, L. Pan, and S. Liu. Cost optimization for cloud storage from user perspectives: Recent advances, taxonomy, and survey. *ACM Computing Surveys*, 55(13s):1–37, 2023a.

Q. Liu and J. Paparrizos. The elephant in the room: Towards a reliable time-series anomaly detection benchmark. In NeurIPS 2024, 2024a.

Q. Liu and J. Paparrizos. The elephant in the room: Towards a reliable time-series anomaly detection benchmark. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024b.

Z. Liu, C. Benge, and S. Jiang. Ticket-bert: Labeling incident management tickets with language models. arXiv preprint arXiv:2307.00108, 2023b.

L. Ma, T. He, A. Swami, D. Towsley, K. K. Leung, and J. Lowe. Node failure localization via network tomography. In *Proceedings of the 2014 Conference on Internet* Measurement Conference, pages 195–208, 2014.

B. Maurer. Fail at Scale: Reliability in the Face of Rapid Change. *Communications of the ACM*, 58(11):44–49, Nov. 2015.

T. Melissaris, K. Nabar, R. Radut, S. Rehmtulla, A. Shi, S. Chandrashekar, and I. Papapanagiotou. Elastic Cloud Services: Scaling Snowflake's Control Plane. In Proceedings of the 13th ACM Symposium on Cloud Computing (SOCC'22), Nov. 2022.

Microsoft and contributors. Dowhy: A python library for causal inference. https://www.pywhy.org/ dowhy/v0.12/, 2023. Version 0.12.

M. D. L. C. Miguel Carreon. Idc: 80 N. R. Murphy, B. Beyer, C. Jones, and J. Petoff. Site Reliability Engineering: Monitoring Distributed Systems. O'Reilly Media, 2024. Accessed: 2024-11-07.

NIST 800-53. NIST Special Publication 800-53 Revision 5.

https://nvlpubs.nist.gov/nistpubs/ SpecialPublications/NIST.SP.80053r5.pdf, 2020.

A. Nodari, J. Nurminen, and C. Fruhwirth. Inventory theory applied to cost optimization in cloud computing. In Proceedings of the 31st annual ACM symposium on applied computing, pages 470–473, 2016.

N. OSCAL. Open security controls assessment language.

https://pages.nist.gov/OSCAL/, 2024. Accessed: 2025-01-30.

OSCAL-compass. Oscal-compass cncf sandbox project.

https://github.com/oscal-compass, 2024.

Accessed: 2025-01-30.

P. Osypanka and P. Nawrocki. Resource usage cost optimization in cloud computing using machine learning. In IEEE Transactions on Cloud Computing, pages 2079– 2089, 2020.

L. Pan, M. Saxon, W. Xu, D. Nathani, X. Wang, and W. Y.

Wang. Automatically correcting large language models: Surveying the landscape of diverse self-correction strategies. *arXiv preprint arXiv:2308.03188*, 2023.

N. Papanikolaou, S. Pearson, and M. C. Mont. Towards Natural-Language Understanding and Automated Enforcement of Privacy Rules and Regulations in the Cloud: Survey and Bibliography. In Secure and Trust Computing, Data Management, and Applications, 2011.

E. Parliament and the Council of the European Union. Digital operational resilience act for the financial sector and amending regulations. https://eur-lex.europa.

eu/EN/legal-content/summary/digitaloperational-resilience-for-thefinancial-sector.html, 2024.

D. Patterson, A. Brown, P. Broadwell, G. Candea, M. Chen, J. Cutler, P. Enriquez, A. Fox, E. Kiciman, M. Merzbacher, D. Oppenheimer, N. Sastry, W. Tetzlaff, J. Traupman, and N. Treuhaft. Recovery-Oriented Computing (ROC): Motivation, Definition, Techniques, and Case Studies. Technical Report UCB//CSD-02-1175, University of California Berkeley, Mar. 2002.

L. Pham, H. Ha, and H. Zhang. Root cause analysis for microservice system based on causal inference: How far are we? In Proceedings of the 39th IEEE/ACM International Conference on Automated Software Engineering, pages 706–715, 2024.

S. Pujar, L. Buratti, X. Guo, N. Dupuis, B. Lewis, S. Suneja, A. Sood, G. Nalawade, M. Jones, A. Morari, and R. Puri. Automated code generation for information technology tasks in yaml through large language models. https: //arxiv.org/abs/2305.02783, 2023.

B. Qiao, F. Yang, C. Luo, Y. Wang, J. Li, Q. Lin, H. Zhang, M. Datta, A. Zhou, T. Moscibroda, S. Rajmohan, and D. Zhang. Intelligent container reallocation at microsoft 365. In Proceedings of the 29th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering, pages 1438–1443, 2021.

G. Quattrocchi, E. Incerto, R. Pinciroli, C. Trubiani, and L. Baresi. Autoscaling solutions for cloud applications under dynamic workloads. In *IEEE Transactions on* Services Computing, pages 804–820, 2024.

D. Roy, X. Zhang, R. Bhave, C. Bansal, P. Las-Casas, R. Fonseca, and S. Rajmohan. Exploring llm-based agents for root cause analysis. https://arxiv.org/
abs/2403.04123, 2024.

G. Sahu, A. Puri, J. Rodriguez, A. Abaskohi, M. Chegini, A. Drouin, P. Taslakian, V. Zantedeschi, A. Lacoste, D. Vazquez, et al. Insightbench: Evaluating business analytics agents through multi-step insight generation. arXiv preprint arXiv:2407.06423, 2024.

Salesforce. Pyrca: A python machine learning library for root cause analysis. https://github.com/
salesforce/PyRCA, 2023. Version 1.0.1.

N. Shinn, F. Cassano, E. Berman, A. Gopinath, K. Narasimhan, and S. Yao. Reflexion: Language agents with verbal reinforcement learning, 2023.

B. H. Sigelman, L. A. Barroso, M. Burrows, P. Stephenson, M. Plakal, D. Beaver, S. Jaspan, and C. Shanbhag. Dapper, a large-scale distributed systems tracing infrastructure. 2010.

J. Storment and M. Fuller. *Cloud FinOps*. O'Reilly Media, 2nd edition, 2023. Chapter 1.

X. Sun, R. Cheng, J. Chen, E. Ang, O. Legunsen, and T. Xu. Testing Configuration Changes in Context to Prevent Production Failures. In Proceedings of the 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI'20), Nov. 2020.

C. Tan, Z. Jin, C. Guo, T. Zhang, H. Wu, K. Deng, D. Bi, and D. Xiang. {NetBouncer}: Active device and link failure localization in data center networks. In *16th USENIX* Symposium on Networked Systems Design and Implementation (NSDI 19), pages 599–614, 2019.

C. Tang, T. Kooburat, P. Venkatachalam, A. Chander, Z. Wen, A. Narayanan, P. Dowell, and R. Karl. Holistic Configuration Management at Facebook. In Proceedings of the 25th ACM Symposium on Operating System Principles (SOSP'15), Oct. 2015.

C. Tang, K. Yu, K. Veeraraghavan, J. Kaldor, S. Michelson, T. Kooburat, A. Anbudurai, M. Clark, K. Gogia, L. Cheng, B. Christensen, A. Gartrell, M. Khutornenko, S. Kulkarni, M. Pawlowski, T. Pelkonen, A. Rodrigues, R. Tibrewal, V. Venkatesan, and P. Zhang. Twine: A Unified Cluster Management System for Shared Infrastructure. In Proceedings of the 14th USENIX Conference on Operating Systems Design and Implementation (OSDI'20), Nov. 2020.

L. Tang, C. Bhandari, Y. Zhang, A. Karanika, S. Ji, I. Gupta, and T. Xu. Fail through the Cracks: Cross-System Interaction Failures in Modern Cloud Systems. In *Proceedings* of the 18th European Conference on Computer Systems
(EuroSys'23), May 2023.

M. Taylor, H. Zaragoza, N. Craswell, S. Robertson, and C. Burges. Optimisation methods for ranking functions with multiple parameters. In Proceedings of the 15th ACM international conference on Information and knowledge management, pages 585–593, 2006.

S. Trask. State of FinOps: 2025 report. https://data.

finops.org/, 2025.

H. Tupsamudre, A. Kumar, V. Agarwal, N. Gupta, and S. Mondal. AI-Assisted Controls Change Management for Cybersecurity in the Cloud. In *Thirty-Fourth Annual* Conference on Innovative Applications of Artificial Intelligence (IAAI-22), 2022.

L. R. Varshney, N. S. Keskar, and R. Socher. Pretrained AI models: Performativity, mobility, and change.

arXiv:1909.03290 [cs.CY]., Sept. 2019.

K. Veeraraghavan, J. Meza, S. Michelson, S. Panneerselvam, A. Gyori, D. Chou, S. Margulis, D. Obenshain, S. Padmanabha, A. Shah, Y. J. Song, and T. Xu. Maelstrom: Mitigating Datacenter-level Disasters by Draining Interdependent Traffic Safely and Efficiently. In *Proceedings* of the 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI'18), Oct. 2018.

Q. Wang, J. Rios, S. Jha, K. Shanmugam, F. Bagehorn, X. Yang, R. Filepp, N. Abe, and L. Shwartz. Fault injection based interventional causal learning for distributed applications. In *Proceedings of the AAAI Conference on* Artificial Intelligence, volume 37, pages 15738–15744, 2023.

J. Xie, K. Zhang, J. Chen, T. Zhu, R. Lou, Y. Tian, Y. Xiao, and Y. Su. Travelplanner: A benchmark for real-world planning with language agents. https: //arxiv.org/abs/2402.01622, 2024a.

Z. Xie, Y. Zheng, L. Ottens, K. Zhang, C. Kozyrakis, and J. Mace. Cloud atlas: Efficient fault localization for cloud systems using language models and causal insight.

https://arxiv.org/abs/2407.08694, 2024b.

B. Xu, Z. Peng, B. Lei, S. Mukherjee, Y. Liu, and D. Xu.

Rewoo: Decoupling reasoning from observations for efficient augmented language models. https://arxiv. org/abs/2305.18323, 2023.

T. Xu, J. Zhang, P. Huang, J. Zheng, T. Sheng, D. Yuan, Y. Zhou, and S. Pasupathy. Do Not Blame Users for Misconfigurations. In *Proceedings of the 24th Symposium* on Operating System Principles (SOSP'13), Farmington, PA, Nov. 2013.

J. Yang, C. E. Jimenez, A. Wettig, K. Lieret, S. Yao, K. Narasimhan, and O. Press. Swe-agent: Agentcomputer interfaces enable automated software engineering. *arXiv preprint arXiv:2405.15793*, 2024a.

X. Yang, R. Arora, S. Jha, C. Narayanaswami, C. Lam, J. Leichter, Y. Deng, and D. Sow. Optimizing it finops and sustainability through unsupervised workload characterization. In *Proceedings of the AAAI Conference on* Artificial Intelligence., pages 22990–22996, 2024b.

S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y. Cao. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, 2023.

Z. Yao, C. Pei, W. Chen, H. Wang, L. Su, H. Jiang, Z. Xie, X. Nie, and D. Pei. Chain-of-event: Interpretable root cause analysis for microservices through automatically learning weighted event causal graph. In Companion Proceedings of the 32nd ACM International Conference on the Foundations of Software Engineering, pages 50– 61, 2024.

A. Yehoshua, I. Kolchinsky, and A. Schuster. Cco - cloud cost optimizer. In Proceedings of the 16th ACM International Conference on Systems and Storage, pages 137– 137, 2023.

C. Zhai and S. Massung. Text Data Management and Analysis: A Practical Introduction to Information Retrieval and Text Mining, volume 12. Association for Computing Machinery and Morgan & Claypool, 2016. ISBN 9781970001174.

X. Zhang, T. Mittal, C. Bansal, R. Wang, M. Ma, Z. Ren, H. Huang, and S. Rajmohan. Flash: A workflow automation agent for diagnosing recurring incidents. October 2024.

L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena. https://arxiv.org/
abs/2306.05685, 2023.

X. Zhou, X. Peng, T. Xie, J. Sun, C. Ji, W. Li, and D. Ding.

Fault analysis and debugging of microservice systems: Industrial survey, benchmark system, and empirical study. IEEE Transactions on Software Engineering, 47(2):243– 260, 2018.

Z. Zhu, C. Lee, X. Tang, and P. He. Hemirca: Fine-grained root cause analysis for microservices with heterogeneous data sources. ACM Transactions on Software Engineering and Methodology, 33(8):1–25, 2024.

# Appendix

In the appendix, we provide additional analyses and more extensive discussions about ITBench, individual personas (SRE, ComplianceOps, FinOps) and agent performance. Data, code, and leaderboard at "link anonymized."

## Table Of Contents

| A   | Related Work                 | 18   |    |
|-----|------------------------------|------|----|
| A.1 | Site Reliability Engineering | 18   |    |
| A.2 | Compliance                   |      | 18 |
| A.3 | FinOps                       | 19   |    |

| B   | ITBench                | 20   |
|-----|------------------------|------|
| B.1 | Benchmark Registration | 20   |
| B.2 | Agent Registration     | 20   |
| B.3 | Leaderboard            | 20   |

| C   | Site Reliability Engineering         | 23   |    |
|-----|--------------------------------------|------|----|
| C.1 | Background                           |      | 23 |
| C.2 | Real-world Incident Example          | 23   |    |
| C.3 | ITBench Architecture                 | 24   |    |
| C.4 | Characterizing ITBench incidents . . | 26   |    |
| C.5 | SRE-Agent                            |      | 30 |
| C.6 | ITBench Evaluation                   | 32   |    |

D **Chief Information Security Officer (CISO)**
and Benchmarking the Compliance Assessment Agent 42 D.1 Background . . . . . . . . . . . . . 42 D.2 Real-World Benchmarking . . . . . 42 D.3 ITBench Architecture for handling CISO Tasks . . . . . . . . . . . . . 44 D.4 ITBench Real-World CISO Scenarios 45 D.5 CISO Scenario Classes and their Complexity . . . . . . . . . . . . . 46 D.6 CISO ITBench Evaluation . . . . . . 49

## E Financial Operations 54

| E.1   | Background   | 54   |
|-------|--------------|------|

E.2 Motivating Example and FinOps

Scenarios . . . . . . . . . . . . . . 55

E.3 ITBench Architecture for Constructing FinOps Scenarios . . . . . . . . 56

E.4 Evaluation . . . . . . . . . . . . . . 56 E.5 Results . . . . . . . . . . . . . . . . 59 E.6 Example Trajectories . . . . . . . . 59

| F   | Normalized Topology-aware Match     | 62   |    |
|-----|-------------------------------------|------|----|
| F.1 | Notation                            | 62   |    |
| F.2 | Overall Score: NTAM                 | 63   |    |
| F.3 | Fault Localization: FL (NTAM) . . . | 64   |    |
| F.4 | Parameter Tuning                    |      | 64 |

## A. Related Work

LM and agents for resolving IT automation tasks. There is a surge in use of AI/ML for handling IT automation tasks. We describe related work for each persona.

## A.1. Site Reliability Engineering

IT incident2resolution encompasses tasks such as detection (e.g., identifying anomalies or outages) (Guo et al.,
2015; Leners et al., 2011; Sigelman et al., 2010; Fonseca et al., 2007), diagnosis (e.g., pinpointing root causes through metrics and logs) (Tan et al., 2019; Jha et al., 2020; Ma et al., 2014; Salesforce, 2023), and mitigation (e.g., operational fixes or code changes). These efforts often rely on supporting tasks like ticket analysis and routing (Gao et al., 2020; Liu et al., 2023b; Arzani et al., 2016), anomaly detection (Liu and Paparrizos, 2024a), topology extraction (Ashok et al., 2024; Chakraborty et al., 2023; Pham et al., 2024; Yao et al., 2024), causal (Budhathoki et al., 2022; Microsoft and contributors, 2023; Ikram et al., 2022; Chakraborty et al., 2023) and interventional (Wang et al., 2023; Bagehorn et al., 2022) analysis using IT data. Clearly, there is significant research in this area, fully automating incident resolution or providing actionable insights to humans remains elusive due to the complexity of real-world systems, the variability of incidents, and the challenge of incorporating contextual knowledge into AI systems (Jha et al., 2020). Recent advances in language models (LMs)
have led to their adoption of ticket data analysis and diagnosis tasks (Roy et al., 2024; Ahmed et al., 2023a; Xie et al.,
2024b; Chen et al., 2023; Zhang et al., 2024). Most notable examples include Cloud Atlas use LLMs for causal graph construction (Xie et al., 2024b), RCACopilot for ticket analysis (Chen et al., 2023) with the aim to diagnose and mitigate incidents. However, they achieve poor performance compared to other techniques. For example, (Roy et al., 2024) shows that chain-of-thought only achieves accuracy of 35%. More recently, LMs are used in agentic workflows, engaging with real or virtual environments, using several tools at their disposal, for tasks like web navigation (Drouin et al., 2024; Boisvert et al., 2024; Koh et al., 2024), system control (Sahu et al., 2024; Zhang et al., 2024; Chen et al., 2024a), and code generation (Yang et al., 2024a). However, the initial results of these works show a high variability in the success rate —35% in InsightBench (Boisvert et al., 2024) and the ReAct-based agent for ticket data analysis (Roy et al., 2024) to 100% in Flash (Chen et al., 2024a) for incident resolution despite the fact that it is a much harder task than identifying planted insights in tabular and ticket data. Our own results in this work suggest that LLMs and agents struggle to consistently complete incident resolution tasks. *We assert that* 2We use the term scenario broadly to refer to failures, performance problems, compliance issues and, cost anomalies.

the variability in success rate exists because of difference in realism of these datasets. This highlights the urgent need for standardized and open source benchmarks to evaluate and improve the efficacy of AI methods on incident resolution tasks effectively. SRE-focused Benchmarks The benchmarking landscape for IT operations (ITOps) tasks is still in its early stages, with a few existing efforts addressing specific aspects of the domain. AIOpsLab (Chen et al., 2024a) focuses on resolving IT incidents *only* for SRE personas, covering ten distinct problems created in a real environment. It does not follow SRE best practices for system and application observability, e.g., using an alert management system, lacks comprehensive coverage analysis, and a leaderboard for systematic automated evaluation. InsightBench (Sahu et al., 2024) targets the analysis of ServiceNow ticket data, a critical supporting task for incident routing and finding relevant past incidents, but its reliance on synthetic data and the lack of a real environment limit its applicability to agentic workflows. Similarly, TSB-AD (Liu and Paparrizos, 2024b) is designed for univariate and multivariate anomaly detection, a core task for incident detection. However, it is limited to synthetic data and focuses only on anomaly detection.

## A.2. Compliance

Compliance automation software is emerging to help businesses streamline and automate compliance processes, reducing the need for manual monitoring and tracking of regulations. This ensures continuous adherence to laws. In particular, compliance as code is a very recent development in the IT industry motivated by companies and audit agencies shifting from annual audits to expectations of continuous and automated measurement of compliance to maintain control of their regulated environments' posture and risks for cyberattacks. Recent works (Papanikolaou et al., 2011; Tupsamudre et al., 2022) have applied AI/ML techniques to speed up these tasks, focusing on mapping regulatory requirements to standard control frameworks such as NIST 800-53 (NIST 80053). Our agentic automation in the current ITBench solution pioneers this type of effort to author compliance artifacts through AI / ML by bridging compliance as code into policy as code. Policy engines have a longer history in the IT industry compared to compliance as code; however, emerging general usage policy engines such as (Int, d) try to address the need for a common framework for continuous compliance. We are not aware of any effort -albeit critical and needed- related to benchmarking of compliance automation software, whether with or without agentic support.

## A.3. Finops

The area of IT cost management encompasses multiple disciplines, namely FinOps, IT Financial Management (ITFM), Technology Business Management (TBM) and Porfolio Businesss Management (PBM). At present, the FinOps domain typically deals with cloud costs (Storment and Fuller, 2023; Yang et al., 2024b), which includes compute nodes, memory, other storage, networking, etc., that are incurred with one of the hyperscalers. ITFM includes on-prem infrastructure, licensing, labor, procured services, tech support, etc. The TBM Council provides a standard taxonomy to describe cost sources, technologies, IT resources (IT towers),
applications, and services. In addition, there are industryspecific extensions to the taxonomy, such as for healthcare, banking, etc. In essence, this taxonomy provides a generally accepted way of categorizing and reporting IT costs and other metrics. PBM refers to the practice of managing a collection of projects and programs within an organization, ensuring alignment with the overall business strategy and maximizing their collective value by allocating resources efficiently. The FinOps Foundation has indicated that over time it will include elements from ITFM, TBM, and PBM. Currently, for FinOps, there is no benchmark that fits the definition of benchmark that we are using in this paper. However, over the years, the FinOps Foundation (Foundation, 2025a) has compiled several KPIs that can form the basis for use cases and scenarios for a FinOps benchmark. Current FinOps Foundation KPIs include: - Usage or Spend Apportionment Validation - Total Unpredicted Variance of Spend - Percent of Compute Spend Covered by Commitment Discounts
- Effective Savings Rate Percentage - Percentage of Commitment Discount Waste - Percent of Unused Resources - Auto-scaling Efficiency Rate
- Forecast Accuracy Rate (Usage, Spend)
- Percentage of Unallocated Shared CSP Cloud Cost - Percentage Variance of Budgeted vs. Forecasted CSP
Cloud Spend
- Percentage of CSP Cloud Costs that are Tagging Policy Compliant
- Percent Storage on Frequent Access Tier - Percentage of Legacy Resource With the advent of the cloud, the academic and industrial research communities have also been active in investigating ways to optimize costs while balancing multiple objectives. Recent works in the space of FinOps have focused on applying machine learning and mathematical optimization techniques (Qiao et al., 2021; Yang et al., 2024b) to better serve customers' cloud infrastructure needs while offering them insights and recommendations on how they could optimize their overall cloud spend. (Fangkai et al., 2023) addresses the issue of helping customers make trade-offs between cost and resource availability in the presence of offerings such as spot VMs which are cheaper than on-demand VMs but have reduced availability. They propose a framework that uses constrained reinforcement learning to optimize cost and availability by identifying an optimal mix of on-demand VMs and spot VMs. Papers such as (Diao et al., 2024; Feng et al., 2023; Quattrocchi et al., 2024) propose forecasting algorithms to scale cloud resources for service level objectives, contributing to the broader field of FinOps-driven cost optimization. (Osypanka and Nawrocki, 2020) uses anomaly detection, machine learning, and particle swarm optimization to achieve a cost-optimal cloud resource configuration. (Liu et al., 2023a) analyze the process of using cloud storage to explore opportunities, motivations, and challenges of cost optimization from user perspectives. (Nodari et al., 2016) focuses on finding the optimal combination of on-demand and reserved instances, such that the demand is satisfied and the costs minimized. They model this optimization problem as a stochastic inventory control problem.

(Yehoshua et al., 2023) introduces a scalable cost optimizer that determines the most cost-effective deployment strategy for workloads on public or hybrid clouds, considering resource requirements and constraints to minimize costs. In FinOps, there is an urgent need to move beyond comparative scorecards and broad taxonomies to specific use cases that test the ability of automated agents to optimize IT investments and reduce resource waste. To our knowledge, no benchmarks exist for use cases like forecasting, anomaly detection, or cost optimization, nor are there standardized methods to evaluate these techniques with or without agentic support. We are confident that ITBench will unite research and development communities to tackle real-world problems through the power of AI and optimization.

## B. Itbench

ITBench framework, as shown in Figure 7, supports two main phases corresponding to two personas as follows: (i) benchmark registration phase, where the target is the Benchmark Submitter persona, and (ii) **agent registration** phase, focusing on the Agent Submitter persona and the actual runtime benchmarking execution and evaluation.

## B.1. Benchmark Registration

This phase comprises two main steps: (i) scenario development and registration, and (ii) tasks and evaluation metrics registration.

## Scenario Development And Registration

Our scenarios are designed to instantiate real-world IT problems in realistic and manageable environments. Each scenario comprises of two core components: (i) an environment specification, and (ii) a scenario specification metadata. The Benchmark Submitter persona then registers these scenarios with ITBench, which stores them in its database. Each scenario is described using the metadata shown in Table 7.

| Field       | Example                                                                                             |
|-------------|-----------------------------------------------------------------------------------------------------|
| Type        | CISO, SRE, FinOps                                                                                   |
| Name        | For CISO: k8s CIS-b Minimize containers w/ shared net namespace                                                                                                     |
| Description | For CISO: Minimize the admission of containers wishing to share the host network namespace          |
| Complexity  | Easy, Medium, Hard                                                                                  |
| Class       | For CISO, this is defined based on the technology (e.g., k8s w/ Kyverno; k8s w/ OPA; Rhel9 w/ OPA). |

Tasks and Evaluation Metrics Registration For each scenario type, the Benchmark Submitter registers a welldefined set of tasks that form the basis for the Agent performance evaluation. Table 2 summarizes the ITBench currently supported IT automation tasks. Moving forward, we plan to extend ITBench to incorporate additional tasks (e.g., threat analysis and resource optimization) and to broaden its applicability to other domains (e.g., DevOps).

## B.2. Agent Registration

During this phase, the Agent Submitter first registers as a user on the platform, then follows with the Agent Registration.

## B.2.1. Agent Registration

During Agent Registration, the Agent Submitter specifies the agent metadata as shown in Table 8. Once the agent has been registered, the Agent Submitter selects the agent, and the corresponding benchmarks are retrieved from the database using the *agent_type, agent_level*, and *scenario_class* specified during registration for the Agent. The Agent Submitter subsequently receives the tasks that the agent must complete to meet the designated objective, each of which has pre-defined evaluation metrics.

## B.3. Leaderboard

| Field                   | Example                 |
|-------------------------|-------------------------|
| Agent Name              | -                       |
| Agent Type (predefined) | CISO, SRE, FinOps . . . |
| Agent Level             | Beginner, Intermediate, Expert (maps to scenario complexity: Easy, Medium, Hard)                         |
| Scenario Class          | For CISO: rhel9 w/ OPA; Kubernetes w/ Kyverno; Kubernetes w/ OPA, Kyverno update                         |

Effective benchmarking of IT automation tasks, especially when selecting LLMs tailored to an organization's specific needs, requires consistent tracking and comparison of agent performance. The Leaderboard facilitates this need by offering a predefined, extensible set of performance metrics that provide clear insights into agent performance relative to the evaluation criteria. The Leaderboard supports both API and UI interfaces, enabling a streamlined benchmarking workflow. Users must register the agent endpoint via the Leaderboard's UI or API. The agent can then query the Leaderboard to retrieve and deploy benchmark scenarios before reporting their operational status. The scenarios can be deployed either automatically by the ITBench, as described above, in its hosted environment, or manually outside the Leaderboard, in the user's hosted environment, in which case both agent and environment can still leverage the same Leaderboard API endpoint to publish status updates. The end-to-end workflow for the agent benchmarking process, after its registration by the Agent Submitter, is illustrated in Figure 7, and summarized in the following. 1. New benchmark jobs are stored in the Benchmark Queue for processing.

2. The Benchmark Runner fetches a benchmark scenario