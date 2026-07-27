# Spider 2.0: Evaluating Language Models On Real-World Enterprise Text-To-Sql Work- Flows

Fangyu Lei∗♠ Jixuan Chen∗♠ Yuxiao Ye♠ Ruisheng Cao♠ **Dongchan Shin**♠ Hongjin Su♠ Zhaoqing Suo♠ Hongcheng Gao♠ Wenjing Hu♠ **Pengcheng Yin**♡
Victor Zhong⋆ Caiming Xiong♢ Ruoxi Sun△ Qian Liu♣ **Sida I. Wang Tao Yu**♠
♠University of Hong Kong ♢Salesforce Research ♣Sea AI Lab
♡Google Deepmind △Google Cloud AI Research ⋆University of Waterloo

## Abstract

Real-world enterprise text-to-SQL workflows often involve complex cloud or local data across various database systems, multiple SQL queries in various dialects, and diverse operations from data transformation to analytics. We introduce Spider 2.0, an evaluation framework comprising 632 real-world text-to-SQL workflow problems derived from enterprise-level database use cases. The databases in Spider 2.0 are sourced from real data applications, often containing over 1,000 columns and stored in local or cloud database systems such as BigQuery and Snowflake. We show that solving problems in Spider 2.0 frequently requires understanding and searching through database metadata, dialect documentation, and even project-level codebases. This challenge calls for models to interact with complex SQL workflow environments, process extremely long contexts, perform intricate reasoning, and generate multiple SQL queries with diverse operations, often exceeding 100 lines, which goes far beyond traditional text-to-SQL challenges. Our evaluations indicate that based on o1-preview, our code agent framework successfully solves only 21.3% of the tasks, compared with 91.2% on Spider 1.0 and 73.0% on BIRD. Our results on Spider 2.0 show that while language models have demonstrated remarkable performance in code generation - especially in prior text-to-SQL benchmarks - they require significant improvement in order to achieve adequate performance for real-world enterprise usage. Progress on Spider 2.0 represents crucial steps towards developing intelligent, autonomous, code agents for real-world enterprise settings. Our code, baseline models, and data are available at spider2-sql.github.io.

## 1 Introduction

Automated code generation can serve as a crucial bridge between humans and data, assisting individuals in achieving difficult or monotonous tasks using complex data. A significant portion of existing data is stored in relational databases, where SQL serves as an essential interface that facilitates human interaction with these data. In this context, semantic parsing or text-to-SQL (Dahl et al., 1994; Zelle & Mooney, 1996; Zettlemoyer & Collins, 2005; Li & Jagadish, 2014; Zhong et al., 2017; Yu et al., 2018) is an important technology that assists data analysts in performing routine queries, orchestrating data workflows, and accomplishing advanced business intelligence, thereby significantly reducing repetitive human labor and alleviating the burden on programmers. Large language models (LLMs) have demonstrated excellent capabilities in generating code (Chen et al., 2021; Austin et al., 2021), particularly in transforming natural language questions into SQL queries. Notably, methods based on GPT-4 achieved execution accuracy of 91.2% and 73.0% on the classic benchmarks Spider 1.0 (Yu et al., 2018) and BIRD (Li et al., 2024b), respectively. Although LLMs excel on these datasets, they often use non-industrial databases with few tables and columns, featuring simplistic SQL and questions that fall short of real-world complexity and overlook diverse SQL dialects. By contrast, real-world data are stored across a diverse array of
∗Equal contribution.

Q: I need a daily report on key sales activities—covering tasks completed, events held, leads generated, and the status of opportunities.

 Language Models SQL **(e.g. Google/Snowflake SQLs )**
Text-to-SQL Workflow Environment Database Documents WITH opportunity AS (
SELECT {{ dbt.date_trunc('day','close_date') }} 
 AS close_date, CASE WHEN is_won THEN 'Won' WHEN NOT is_won AND is_closed THEN 'Lost' WHEN NOT is_closed AND LOWER(forecast_category) 
IN ('pipeline','best case') THEN 'Pipeline' END AS status FROM {{ var('opportunity') }} ),
Diverse System Database Metadata External Knowledge SQL Dialect Docs Query Interface Real Apps SQLs / 
Python Codebase
[ + 100 lines omitted]
models/
data/ macros/
project.yml schema.yml leads.sql salesforce.db +20 files Table 1: activity_date account_id type Exec feedback LEFT JOIN event on ds.date_day = salesforce_event.activity_date LEFT JOIN opportunities_created ON ds.date_day = opportunities_created.created_date LEFT JOIN opportunities_closed ON ds.date_day = opportunities_closed.close_date Complex Schema Table 495: opportunity_id stage_name amount
(8695 columns omitted)
database systems, each with its own unique SQL dialects, introducing a wide range of SQL syntax and functions. Additionally, these enterprise-level application databases are characterized by largescale schemas with thousands of columns and complex nested structures. Moreover, real-world text-to-SQL workflows require the utilization of project codebases, external knowledge, and various contexts to construct intricate SQL queries across multiple steps, complete various operations, and build a comprehensive data engineering pipeline. This includes data wrangling to clean and organize the data for analysis, data transformation to restructure and enhance the data, and conducting data analytics to extract insights that inform decision making and drive strategic initiatives. All these complexities underscore the pressing need for a more realistic enterprise-level benchmark. We present Spider 2.0, a benchmark that reflects real-world data workflows to facilitate the development of text-to-SQL models in enterprise applications, encompassing 632 real-world complex data wrangling, transformation, and analysis tasks. As illustrated in Fig. 1, the databases in Spider 2.0 are sourced from industrial applications (e.g. Google Analytics and Salesforce) and feature massive schema items (an average of 812 columns) with unique structures (e.g., nested columns in Fig. 11, multiple schema in Fig. 12), along with terabyte-scale data volumes. They encompass a variety of database systems, including local databases (e.g., SQLite and DuckDB) and cloud data warehouses (e.g., BigQuery and Snowflake). Complicated SQL dialects for these databases are curated from technical tutorials, community forums, and open-source projects. On average, each ground-truth SQL query contains 144 tokens and includes advanced functions (e.g., ST DISTANCE(x1, x2)
measures the shortest distance between two points), exhibiting a level of complexity notably surpassing previous benchmarks. All tasks are based on project codebases along with documents and database interface to simulate real-world text-to-SQL writing scenarios. Unlike previous datasets, Spider 2.0's agentic task setting does not rely on pre-prepared inputs (question and database schema) or expected outputs (predicted SQL). Instead, it incorporates a real project codebase and a database interface. This complexity extends beyond merely predicting an SQL query; it involves navigating the project and dynamically interacting with complex databases through SQL queries and command-line scripts (in Python or Shell). The task objective is to perform intricate data transformations within the database or to extract analytical insights from the data. This task setting closely mirrors real-world enterprise SQL workflows, requiring the model to refer to the codebase and documentation, generate multiple SQL queries, and dynamically interact with the environment to complete complex tasks and derive the final result. To simplify performance comparisons with previous text-to-SQL methods and benchmarks, and to support faster development and evaluation, we also introduce Spider 2.0-lite and Spider 2.0-snow, self-contained datasets with preprocessed database schema and documentation, the former is hosted on BigQuery, Snowflake, and SQLite, while the latter is entirely hosted on Snowflake. This setting omits the codebase and restricts output to SQL only, thus eliminating the need to predict final answers or transform the database. While they are sourced from the same raw data as Spider 2.0, these settings are not easier than Spider 2.0 because text-to-SQL setting have access to less information (e.g., execution feedback). We present Spider 2.0-lite and Spider 2.0-snow as direct text-input-to-SQL-output challenges that are easier to work with using current advanced text-to-SQL parsers, and Spider 2.0 as a real-world data workflow challenge that involves interacting with diverse sources to perform data transformation and analyses. Our evaluation on Spider 2.0 indicates significant room for improvement in deploying LLMs within real-world enterprise text-to-SQL workflows. The best o1-preview based code agent framework achieves a performance of only 21.3%, underscoring the significant deficiency in LLMs' capability to serve as proficient SQL experts (Tab. 2). As for Spider 2.0-lite setting, even the most advanced text-to-SQL parser could successfully address only 5.7% of the questions, a stark contrast to the execution accuracy of 91.2% on Spider 1.0 and 73.0% on BIRD, thereby highlighting the substantial challenges (§3.2). Our detailed analysis further identifies major obstacles in enterprise text-to-SQL, including accurately linking schemas from extremely large databases, correctly handling different SQL dialects, planning sequences of nested SQL queries to perform complex transformation and analytical tasks, and effectively leveraging external documentations and understanding project-level codebase. (§4.1 and §4.2). These challenges in Spider 2.0 represent crucial steps toward creating a benchmark that closely aligns with real-world scenarios. With Spider 2.0, we aim to enable the development of a new generation of intelligent autonomous agents capable of data engineering workflows in real-world enterprise settings.

## 2 Benchmark Construction

In this section, we introduce the task definition, general annotation pipeline, and dataset statistics for Spider 2.0, Spider 2.0-snow and Spider 2.0-lite. For concrete examples, refer to App.B.

## 2.1 Task Definition Ver 3.

Question Spider 2.0-*lite* Self-contained Text-to-SQL
 Parser DB Metadata SQL Query Spider 2.0 Question Database Documents Codebase LM Agent

Question Spider 2.0-*lite* / Spider 2.0-*snow* Text-to-SQL
 Parser Self-contained Spider 2.0 DB Schema SQL Query Question SQLs / 
Python Database Documents Codebase LM Agent Exec Feedback SQLs/Python Exec Intermediate Result Final Result
Fig. 2 illustrates the task definition of both code agent setting and traditional text-to-SQL setting. Code agent task. Spider 2.0 is defined as a comprehensive code agent task. Given a question Q, a database interface I, and a codebase C (with project context, configuration, and documentation, illustrated in Fig. 1), the task is to iteratively modify the code (SQL/Python) C based on observations Ok = execute(C, I, Q) until the final result A (text/table/database) is obtained. In other words, we use the final observation Ok as an agent's answer to the question, i.e., A = Ok.

Text-to-SQL task. In contrast to Spider 2.0, Spider 2.0snow and Spider 2.0-lite are formulated as self-contained tasks. Given database schema D, a natural language question Q, and auxiliary documentation E as inputs, the textto-SQL parser f(·) is required to output the corresponding SQL query S = f(Q, D, E | θ), where θ is the parameters of the parser. Spider 2.0-lite's database is hosted on diverse databases like Spider 2.0, while Spider 2.0-snow is entirely hosted on Snowflake, with a greater focus on text-to-SQL generation.

 Result Result Figure 2: We offer two settings: traditional text-input-to-SQL-output Spider 2.0-lite/snow, and agentic Spider 2.0.

## 2.2 Annotation Pipeline

Eight authors majoring in computer science, all highly proficient in SQL, carry out the data annotation process. The annotation pipelines consist of the following six steps: 1) Database and SQL collection. We collect various databases from cloud data warehouses, including BigQuery public data, Snowflake Marketplace data, and other platforms, to ensure that they meet specific criteria: each database must contain more than 200 columns or have a nested schema structure. After filtering, we select 74 BigQuery, 54 Snowflake, 30 SQLite, 40 DuckDB, 10 PostgreSQL,
and 5 ClickHouse databases. From the corresponding tutorials and forums, we gather 1, 021 complex SQL queries, as well as 157 data transformation projects sourced from Fivetran and DBT (see App.B.2). To meet our criteria, the SQL queries must contain more than 50 tokens (tokenized by whitespace; for reference, the average token count of BIRD (Li et al., 2024b) is 30.9). Furthermore, queries must originate from real projects or tutorials, not from synthetic examples or corner cases. Ultimately, we retain 547 high-quality SQL queries and 78 DBT projects. 2) SQL rewrite to prevent data leakage. To avoid contamination and ensure the credibility of Spider 2.0's evaluation, annotators are required to rewrite each SQL and verify that they are bugfree. The rewrites are performed at two levels of increasing complexity: the surface and semantic levels, as detailed in Tab. 1. 84.2% of the examples underwent surface-level rewriting, while 42% experienced semantic-level rewriting. Annotators must ensure that the rewritten SQL executes successfully, completes in an acceptable time, and returns non-empty results. 85.98% of these SQL queries utilize advanced functions in various dialects (App.B.7.1), while 10.76% require additional DBT tools, posing challenges due to the need to integrate the project context. Table 1: The rewrite categories are as follows: "Surface" rewrites adjust the parameters and the answer format, while "Semantic" rewrites expand the question's meaning. Each table reference in Example column represents the details of rewrite examples for the corresponding type.

| Rewrite          | Categories   | Example   |
|------------------|--------------|-----------|
| Surface Semantic |              |           |

Surface

Answer format Tab. 13, replace the one channel with the channel ranking by sessions.

Condition parameters Tab. 14, more complex filter condition: citibike is faster than a taxi.

Advanced calculation Tab. 15, calculate originality score based on selected publications.

Semantic

Advanced requirements Tab. 16, change page view order to *page conversion rate*. Merge related SQLs Tab. 17, merge geography-related and weather-related queries. SQL codebase files App.B.2, change SQL and YML files in the original project.

3) Codebase and context setup. For each complex SQL query in Spider 2.0-lite and Spider 2.0snow, we collect the external reference documents necessary to complete the task. Since the tasks span multiple database types, we gather documentation on SQL dialects and external functions, as shown in Tab. 18. Additionally, for Spider 2.0, we preserve the original codebase of the SQL-related project. For Spider 2.0, besides collecting reference documents, annotators also gather resources such as codebases, database interfaces to establish the context for each task (Fig. 1). Since some complex data transformation intentions may not be fully captured by a natural language question, annotators provide additional context, including data model descriptions (App.B.2) or predefined answer files (App.B.5), to maintain clarity while addressing potential ambiguities. 4) Natural language task instructions annotation. Annotators are required to write questions based on the SQLs and context gathered in Step 3, crafting two versions for different settings. The instructions are designed to balance both *naturalness* and *unambiguity*. Due to the differences between Spider 2.0 and Spider 2.0-lite/snow, code agent tasks demonstrate greater naturalness in its questions because it provides contexts and predefined files to guide the answers, while text-to-SQL tasks prioritize unambiguity, ensuring clearer and more straightforward specifications (see App.B.6 for differences). Annotators manually write the instructions, making them natural by avoiding blunt descriptions, *removing ambiguity* in the expected results, and ensuring that all SQL *conditions are* clearly mentioned. Also, the DBT-project tasks (see Fig. 1 and App.B.2), which are realistic data transformation coding scenarios, are exclusively used in Spider 2.0. Annotators craft task instructions based on the provided context. After the initial annotation, they verify the semantic equivalence between the SQL queries and instructions, paraphrasing for clarity with the help of LLMs. 5) Execution-based focused evaluation. In this step, annotators are required to obtain results from the databases programmatically and write evaluation scripts (details in App.A). The evaluation scripts can process the results in the form of strings, tables, and database files. It is important to note that in table-based evaluations, predicted results may include numerous columns, which might not exactly match the gold standard answers. This discrepancy often arises because some questions do not specify the columns that should be returned. To mitigate this, the evaluation scripts are specifically focused on the essential components of the answers, ignoring non-essential columns and emphasizing the core elements outlined in the instructions. This method facilitates targeted assessments of key columns for each task, thus significantly reducing the occurrence of false negatives.

For Spider 2.0-lite and Spider 2.0-snow, these settings require that the output must be SQL, so the evaluation will compare the execution results of the SQLs using the table-based assessment method. 6) Quality control. To ensure the quality of our benchmark, each instruction, the gold SQL query, and evaluation script are reviewed by at least three annotators. We require the annotators to repeatedly review steps 3), 4), and 5) to ensure the correctness, naturalness, and unambiguity of the annotations. Consequently, 45% of the examples have errors identified by the first validators. After discussions and corrections, following the second round of iteration with the second validators, only 5% of the examples contain errors. Then we correct all errors and refine all annotations, and ultimately, all examples are deemed fully annotated. Additionally, we perform a "red team" assessment of our automatic evaluation by providing a set of false results to determine if they would be correctly classified as false, along with various correctly formatted results to verify their classification as true.

## 2.3 Dataset Statistics

We present a detailed statistical analysis of the features of Spider 2.0, Spider 2.0-snow and Spider 2.0-lite, comparing them with multiple previous datasets in Tab. 2, our datasets demonstrate strong complexity and realism in aspects such as databases, SQLs, and task scenarios. Table 2: Statistical comparison among Spider 2.0, Spider 2.0-snow and Spider 2.0-lite, and other text-to-SQL benchmarks. Tok. and Func. refer to tokens and functions, respectively. * denotes the statistics from dev set due to the inaccessibility of test set. For more statistics, refer to App.B.8.

| Dataset                       | # Test   | # Test   | # Col   | # Tok.   | # Func.   | External   | SQL   | Project   |
|-------------------------------|----------|----------|---------|----------|-----------|------------|-------|-----------|
| Examples                      | DB       | / DB     | / SQL   | / SQL    | Knowledge | Dialect    | Level |           |
| WikiSQL (Zhong et al., 2017)  | 15,878   | 5,230    | 6.3     | 12.2     | 0.0       |            |       |           |
| Spider 1.0 (Yu et al., 2018)  | 2,147    | 40       | 27.1    | 18.5     | 0.0*      |            |       |           |
| KaggleDBQA (Lee et al., 2021) | 272      | 8        | 23.4    | 13.8     | 0.0       |            |       |           |
| SEDE (Hazoom et al., 2021)    | 857      | 1        | 212.0   | 46.9     | 1.4       |            |       |           |
| BIRD (Li et al., 2024b)       | 1,789    | 15       | 54.2    | 30.9     | 0.4*      |            |       |           |
| Spider 2.0-lite               | 547      | 158      | 803.6   | 144.5    | 6.5       |            |       |           |
| Spider 2.0-snow               | 547      | 152      | 812.1   | 161.8    | 6.8       |            |       |           |
| Spider 2.0                    | 632      | 213      | 743.5   | 148.3    | 7.1       | BigQuery   |       |           |

BigQuery Snowflake Clickhouse 34.7%
25.4%
2.3%
4.7%
14.1%
DuckDB
18.8%
Postgres SQLite
Figure 3: Data distribution on different database systems. Table 3: Statistics of Spider 2.0 task features.

Diverse database systems and SQL dialects. As shown in Fig. 3 and Tab. 3, our benchmarks feature a diverse array of

database systems, including cloud data warehouses like Big-

Query and *Snowflake*, locally hosted databases such as Postgres and *ClickHouse*, and lightweight systems like *SQLite* and DuckDB. This diversity distinguishes our benchmarks from previous work by encompassing various SQL dialects. Notably, 85.98% of the examples require the use of specialized functions from these dialects, with an average of 7.1 special functions utilized in each ground-truth SQL. Real and complex database schema. As shown in Tab. 2, the databases in Spider 2.0 are equipped with large-scale schemas comprising extensive tables and columns, effectively mirroring real-world enterprise environments. As shown in Tab. 3, these databases are characterized by complex schema structures (e.g., multiple and nested schemas, partitioned tables; see Fig. 11 and Fig. 12), and dynamic tables that are updated daily. Additionally, the data encompasses a broad spectrum of complex types (Fig. 16), extensive volumes, and diverse scopes (Fig. 15), rendering it more diverse than previous datasets. Challenging tasks across the data engineering pipeline. The examples in our benchmarks are collected from real tutorials and forums, covering a wide range of issues encountered in data pipelines, including data wrangling, data transformation, and data analysis (see App.B.1 for examples). The difficulty of these questions significantly exceeds that of previous SQL-

related benchmarks, as the SQL queries in Spider 2.0 contain

significantly more columns, tokens, and functions per query than those in prior work (see Tab. 2 and Fig. 18 for examples). Real projects scenarios with codebases and documents. As demonstrated in Tab. 2 and 3, tasks in both datasets require access to documentation, like external knowledge (App.B.4) and SQL dialect (App.B.7), necessitating a deep understanding of these resources. Compared to other prior works,

| Statistics                    | Number (% of Total)   |
|-------------------------------|-----------------------|
| Total Levels (#tokens)        | 632 (100%)            |
| - Easy (#tokens < 80)         | 160 (25.32%)          |
| - Medium (80 ≤ #tokens < 160) | 279 (44.15%)          |
| - Hard (#tokens ≥ 160)        | 193 (30.54%)          |
| - With Bigquery               | 214 (33.86%)          |
| - With Snowflake              | 198 (31.33%)          |
| - With SQLite                 | 135 (21.36%)          |
| - With DuckDB                 | 68 (10.76%)           |
| - With Postgres               | 10 (1.58%)            |
| - With Clickhouse             | 7 (1.11%)             |
| - With Project-level (DBT)    | 78 (12.34%)           |
| - With Documentation          | 82 (12.97%)           |
| - With Functions              | 474 (75.00%)          |
| - With Partition Tables       | 54 (8.54%)            |
| - With Multiple Schemas       | 140 (22.15%)          |
| - With Nested Schemas         | 117 (18.51%)          |
| - With String/Number Answer   | 162 (25.63%)          |
| - With Table Answer           | 392 (62.03%)          |
| - With Database Answer        | 78 (12.34%)           |

for each task in Spider 2.0, we provide a codebase context to simulate a real workflow (App.B.5). More notably, some tasks introduce innovations such as project-level data transformation workflows built on DBT (App.B.2), a widely used tool for managing data transformations and analytics engineering. Successfully addressing these tasks requires navigating complex project codebases and databases, comprehending documentation, processing intricate contexts, and generating diverse queries through multi-step execution and reasoning.

## 3 Experiments

3.1 EXPERIMENTAL SETUP Evaluation metrics. For Spider 2.0, we use the **Success Rate (SR)** metric, which measures the proportion of task instances successfully completed. For Spider 2.0-lite and Spider 2.0-snow, the output for each task must be an SQL, we use the widely used metric **Execution Accuracy (EX)**(Yu et al., 2018; Li et al., 2024b). We employ the execution-based *focused* evaluation (App.A) to determine the success of each result for Spider 2.0 and assess the accuracy of SQL execution results for Spider 2.0-lite. The evaluation scripts are designed to accept output in the form of strings, tables, or database. For each example, an evaluation script is run for each example, producing a score of either 0 or 1. It is worth noting that in table-based evaluations, predicted results may contain numerous columns, leading to results that are not exactly the same as the gold answer. This occurs because, for some examples, questions do not explicitly specify which columns to return. The evaluation scripts are specifically focused on the essential components of the answers, disregarding irrelevant columns and concentrating on the core elements specified in the instructions. Difficulty level. We tokenize the gold SQL queries based on whitespace and classify their difficulty according to the number of tokens: < 80 tokens as Easy, 80 − 159 as Medium, and ≥ 160 as Hard1.

LLMs. We experiment with state-of-the-art LLMs, including open-source representatives such as DeepseekCoder-V2.5 (Zhu et al., 2024), Qwen2.5-72B-Instruct (Team, 2024) and Llama-3.1-405B (Meta AI, 2024), and closed-source ones including Gemini-Pro-1.5 (Reid et al., 2024), Claude3.5-Sonnet (Anthropic, 2024) and GPT (OpenAI, 2023) families (GPT-4o, GPT-4, o1-preview and o3-mini). Follow (Yang et al., 2024a; Chen et al., 2024), we use a temperature of 0.0 and truncate from the beginning of the input if still exceeding the max tokens limit required by the models. Code agent frameworks. We utilize several state-of-the-art frameworks, which have demonstrated excellent performance on other benchmarks. These include Reflexion (Shinn et al., 2023), CodeR (Chen et al., 2024), AutoEval (Pan et al., 2024). Inspired by React (Yao et al., 2022) and Intercode (Yang et al., 2023), we develop an agent framework called Spider-Agent, which is primarily focused on database-related coding tasks and projects. The framework allows for multi-turn interactions with the database via command-line interfaces until the final answer is obtained. The implementation details of Spider-Agent are shown in App.C.1. Text-to-SQL methods. We also evaluate several state-of-the-art and widely recognized text-to-SQL methods, including approaches based on prompting LLMs such as DIN-SQL (Pourreza & Rafiei, 2024), DAIL-SQL (Gao et al., 2024) and CHESS (Talaei et al., 2024), alongside SFT CodeS (Li et al., 2024a), which fine-tuned open-source models on extensive text-to-SQL corpora. DAIL-SQL and CHESS achieve the best performance among all accessible methods on the Spider 1.0 and BIRD benchmark, respectively. During implementation, we optimize the prompt organizations across all methods to better align with tasks, incorporating sampled cell values, external knowledge, and SQL dialect specifications (see Fig. 21).

## 3.2 Evaluation Results

Existing LLMs are still far from being expert on real-world text-to-SQL workflow tasks. As shown in Tab.4 and Tab.6, we used Spider-Agent and its variants to conduct tests on Spider 2.0, Spider 2.0-Lite, and Spider 2.0-Snow. The o1-preview and o3-mini achieve the highest performance, with a peak success rate of 23.77% on Spider 2.0-snow and 23.40% on Spider 2.0-lite, indicating significant potential for further improvement. It surpasses both GPT-4o and Claude-3.5-Sonnet across the Easy, *Medium*, and *Hard* cases, highlighting its superior reasoning capabilities. The opensource LLM DeepSeek-V3 showed a performance of 8.78%, still has significant room for improvement. The results shown in Tab.6, combined with the DBT project examples, also exhibit a similar 1While there are various ways to measure difficulty, we use SQL length here as the most common and significant metric for experimental reference.

| Model             | Spider 2.0-Lite   | Spider 2.0-Snow   |       |        |        |        |       |        |
|-------------------|-------------------|-------------------|-------|--------|--------|--------|-------|--------|
| Claude-3.5-Sonnet | 26.56%            | 15.85%            | 6.94% | 15.54% | 25.00% | 16.26% | 7.51% | 15.54% |
| DeepSeek-V3       | 19.53%            | 6.50%             | 4.05% | 8.78%  | 20.31% | 6.1%   | 4.05% | 8.78%  |
| Qwen2.5-Coder     | 13.89%            | 4.17%             | 3.38% | 5.30%  | 11.72% | 4.47%  | 2.31% | 5.48%  |

Table 5: Execution Accuracy (EX) for baseline methods on three text-to-SQL datasets: Spider 1.0, BIRD, Spider 2.0-lite and Spider 2.0-snow. trend. Tab. 5 illustrates that Spider 2.0-lite and Spider 2.0-snow present significant challenges for traditional text-to-SQL methods. The highest performing method, DAIL-SQL + GPT-4o, achieves an EX of only 5.68%, which is markedly lower compared to its score of 86.6% on Spider 1.0 and 57.4% on BIRD datasets. With efficiently filtering the minimal sufficient schema, CHESS + GPT-4o is able to tackle more instances than DIN-SQL. Despite being extensively fine-tuned, SFT CodeS- 15B is far from solving Spider 2.0-lite, with an EX score of only 0.73%, which further reveals the significant complexity gap between Spider 2.0-lite and the current text-to-SQL corpus. For Spider 2.0-snow, even the best method achieves only 2.20% EX, highlighting the increased challenge due to SQL dialect differences.

Table 6: Success rate (SR) of different frameworks and models on Spider 2.0. The costs under different settings are shown in Tab.21. Spider 2.0 consists of Spider 2.0-Lite along with DBT-project tasks.

Existing code agent frameworks struggle with solving database-related coding tasks. Tab.4 and Tab. 6 show that the current agent frameworks are still unable to effectively address the tasks. The challenge is that they must not only explore the codebase and documentation, but also navigate complex databases and generate SQL queries that are far more intricate than typical code. This demands a high level of code grounding capability. Spider-Agent provides a crucial baseline for Spider 2.0, facilitating the evaluation of various LLMs, underscoring the potential for significant advancements and inspiring methodology enhancements for future research. We also observe that the model must be proficient in debugging from SQL execution feedback and exploring the schemas of different types of databases (e.g., Snowflake), which poses a significant challenge to the code agent's capabilities. There is still significant room for improvement in Spider-Agent when it comes to enterprise-level SQL tasks, in order to fully unleash LLMs' text-to-SQL capabilities.

| EX (↑)            |        |                 |                 |        |       |       |       |
|-------------------|--------|-----------------|-----------------|--------|-------|-------|-------|
| Spider 1.0        | BIRD   | Spider 2.0-snow | Spider 2.0-lite |        |       |       |       |
| Easy              | Medium | Hard            | Overall         |        |       |       |       |
| DIN-SQL + GPT-4o  | 85.3%  | 55.9%           | 0.00%           | 5.79%  | 0.43% | 0.00% | 1.46% |
| DAIL-SQL + GPT-4o | 86.6%  | 57.4%           | 2.20%           | 13.20% | 5.58% | 1.24% | 5.68% |
| CHESS + GPT-4o    | 87.2%  | 66.7%           | 1.28%           | 9.92%  | 3.00% | 1.24% | 3.84% |
| SFT CodeS-15B     | 85.4%  | 59.3%           | 0.00%           | 1.65%  | 0.86% | 0.00% | 0.73% |

| Framework         | Model   | SR (↑)   |
|-------------------|---------|----------|
| AutoEval          | GPT-4o  | 5.70%    |
| Reflexion         | GPT-4o  | 7.28%    |
| CodeR             | GPT-4o  | 7.91%    |
| Claude-3.5-Sonnet | 14.87%  |          |
| Qwen2.5-72B       | 6.17%   |          |
| DeepSeek-V2.5     | 5.22%   |          |
| Gemini-Pro-1.5    | 2.53%   |          |
| Llama-3.1-405B    | 2.21%   |          |
| Spider-Agent      |         |          |

## 4 Analysis 4.1 Analysis Of Different Task Types Llm-Agent Frameworks Struggle Interpreting Databases With Nested Schema.

As shown in Tab. 7, the model often performs poorly when handling columns with nested types. Nested columns are a common scenario in industrial-grade databases (see Fig. 11), where data is stored in array, dict formats within a single column. This poses significant challenges for LLMs in understanding the schema. As shown in Fig. 29, LLMs encounter schema linking errors due to an incomplete understanding of the information contained within nested fields. Most databases with nested types face the issue that models find it difficult to fully grasp the function of each nested column's internal information, while humans can comprehend the schema through multi-step reasoning and iterative understanding.

## The Performance Drops When External Documents Are Required.

Table 7: Model performance on databases with nested columns in non-dbt projects. Task Subset % of Total SR (↑) w/ Nested Column 18.51% 10.34% w/o Nested Columns 68.04% 27.38%
From Tab. 8, we observe that when tasks involve external documents, the model performs poorly, correctly answering only 11 examples out of full dataset that accounts for just 11.54%. Through error analysis, we find that the model is not incapable of grounding complex documents information. These models typically have the correct problem-solving strategies and effectively explore the database, but fails at the most crucial step: grounding the complex requirements from the documents into SQLs. As the document shown in Fig. 13, the gold SQL is shown in Tab. 16. The failure case shows that the model cannot combine complex document with schema information and convert it into SQL query (Fig. 28).

Table 8: Performance of the model on external document tasks in non-dbt projects.

Task Subset % of Total SR (↑)
w/ External Doc 12.97% 11.54% w/o External Doc 73.58% 26.64%
LLM-agent frameworks struggle to address project-level tasks. As shown in Tab. 9, the LM agent's performance on DBT-based project tasks is poor, solving only 12.82% of tasks with just 10 examples correct. This underscores the challenges in there tasks, which can be attributed to: (1) Data transformation projects often require *multiple SQL queries* to complete various models, necessitating a comprehensive understanding of the project. (2) These tasks involve *complex context usage*, demanding strong repository exploratory capabilities. (3) Data is stored in databases, requiring the agent to transform data while exploring existing data, alongside SQL coding. Fig. 26 illustrates the action process of o1-preview successfully solving a task defined in App.B.2, while Fig. 27 is a failure case due to the failure to explore the information in the "mrr.md" file to solve a monthly recurring revenue classification.

Table 9: Performance on DBT Project. Task Subset % of Total SR (↑) w/ DBT Project 12.34% 12.82% w/o DBT Project 87.65% 23.22%

## 4.2 Error Analysis Of Sql Generation

We conduct a detailed analysis of the errors encountered by both code agent frameworks on randomly sampled 300 examples, as illustrated in Fig. 4. Representative errors along with their statistics and causal analysis are as follows. Erroneous data analysis (35.5%). Compare to the previous benchmarks, Spider 2.0 and Spider 2.0-lite exhibit significantly complex data analysis demands that challenge the models' capabilities: 1) Dialect function usage (10.3%). This includes processing temporal (e.g., DATE TRUNC) or geographic data (e.g., ST DISTANCE). These functions require a nuanced understanding, which the models often fail to exhibit. 2) Advanced data calculation (7.5%). Model struggle with tasks like grouping samples to analyze trends within groups (using NTILE), or applying formulas for statistical values (e.g., CORR for Pearson correlation coefficients; STDDEV for standard deviation). 3) Intricate query planning (17.7%). Gold SQLs typically involve multiple nested queries, intermediate result processing through common table expressions (CTEs), or merging results from various sub-queries via set operations. However, models often inadequately handle these complexities. Refer to Fig. 5 for case studies on erroneous data processing. Wrong schema linking (27.6%). This category includes errors with wrong tables and columns. For column linking errors (16.6%), the average number of columns per database in Spider 2.0-lite far exceeds those in other benchmarks (over 755 compared to approximately 54 in BIRD), making accurate column linking extremely challenging. Regarding table linking (10.1%), although examples from BigQuery support advanced syntax features like (TABLE SUFFIX) and wildcard expressions, the models show limited flexibility in leveraging these features, even in few-shot setting. JOIN errors (8.3%). While foreign keys represent known schema relationships essential for valid SQL JOIN operations, databases in BigQuery often lack explicit foreign key. This omission forces models to infer potential keys based on column names and descriptions, leading to errors. Table 10: EX for baseline methods on Spider 2.0-lite under oracle setting. To seek the highest possible performance, we also employ the latest o1-preview as the base LLM.

Table 11: EX for DAIL-SQL on Spider 2.0-lite under few-shot setting with manually selected demonstrations.

| Method                                                                                                                                                                                                                                              | EX (↑)                                                                                                                                                                                                                                                                                                  |        |        |        |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|--------|--------|
| w.Oracle Func                                                                                                                                                                                                                                       | w/o Oracle Func                                                                                                                                                                                                                                                                                         |        |        |        |
| DAIL-SQL + GPT-4o                                                                                                                                                                                                                                   | 5.85%                                                                                                                                                                                                                                                                                                   | 5.68%  |        |        |
| DAIL-SQL + o1-preview                                                                                                                                                                                                                               | 9.51%                                                                                                                                                                                                                                                                                                   | 12.60% | Method | EX (↑) |
| 0-shot                                                                                                                                                                                                                                              | 1-shot                                                                                                                                                                                                                                                                                                  | 3-shot |        |        |
| DAIL-SQL + GPT-4o                                                                                                                                                                                                                                   | 5.68%                                                                                                                                                                                                                                                                                                   | 6.40%  | 6.76%  |        |
| 1001                                                                                                                                                                                                                                                |                                                                                                                                                                                                                                                                                                         |        |        |        |
| Question:                                                                                                                                                                                                                                           | Can you group users by the week they first used the app starting from July 2, 2018, and identify the group that has the highest  retention rate in the fourth week? Each group should be named by the Monday date of their start week. Please provide the  answer in the format 'YYYY-MM-DD'. Question: |        |        |        |
| For taxi trips with a duration rounded to the nearest minute, and between  1 and 50 minutes, if the trip durations are divided into 10 quantiles, what are the total number of trips and the average fare for each quantile? Gold SQL: Gold Result: | Gold SQL:                                                                                                                                                                                                                                                                                               |        |        |        |

Can you group users by the week they first used the app starting from July 2, 2018, and identify the group that has **the highest** retention rate in the fourth week? Each group should be named by the Monday date of their start week. Please provide the answer in the format 'YYYY-MM-DD'.

Question:
Question:
For taxi trips with a duration rounded to the nearest minute, and between 1 and 50 minutes, if the **trip durations are divided into 10 quantiles**, 
what are the total number of trips and the average fare for each quantile? Gold SQL:
Gold Result:
Gold SQL:
SELECT SUM(trips) AS total_trips, FORMAT('%3.2f', 
SUM(total_fare)/SUM(trips)
) AS average_fare FROM ( SELECT **NTILE(10) OVER** 
(ORDER BY duration_in_minutes) 
AS quantile, COUNT(1) AS trips, SUM(fare) AS total_fare FROM ( SELECT ROUND(trip_seconds/60) 
AS duration_in_minutes, fare FROM Chicago.taxi_trips WHERE ROUND(trip_seconds/60) 
BETWEEN 1 AND 50
 ) AS initial_query GROUP BY **duration_in_minutes** ) GROUP BY quantile ORDER BY quantile
-- Identify **new users on their first session start** WITH date_table AS (
SELECT DISTINCT 
PARSE_DATE('%Y%m%d', `event_date`) AS event_date, ... CASE WHEN DATE_DIFF(
event_date, DATE(user_first_touch_timestamp), DAY) = 0 THEN 1 ELSE 0 END AS is_new_user FROM `firebase-public-project...events_*` WHERE event_date >= DATE('2018-07-02') AND ... ),
-- Identify distinct new users and their first session start dates new_user_list AS (
SELECT DISTINCT user_pseudo_id, ... FROM date_table WHERE is_new_user = 1 ),
-- **Calculate days since the start for each user** days_since_start_table AS (
SELECT DISTINCT 
DATE_DIFF(dt.event_date, ...) AS days_since_start, ...

FROM date_table dt JOIN new_user_list nu ON ... ),
-- **Aggregate users into weekly cohorts** and calculate weeks since start weeks_retention AS ( SELECT 
DATE_TRUNC(date_cohort, WEEK(MONDAY)) AS week_cohort, CEIL(days_since_start / 7) AS weeks_since_start FROM days_since_start_table ),
-- **Count the number of retained users for each week** retention_counts AS (
SELECT week_cohort, weeks_since_start, COUNT(DISTINCT user_pseudo_id) AS retained_users FROM weeks_retention WHERE week_cohort >= DATE('2018-07-02') GROUP BY ... )
-- **Calculate retention percentage for week 4** SELECT initial.week_cohort FROM retention_counts AS initial LEFT JOIN retention_counts AS four_week ON initial.week_cohort = four_week.week_cohort AND **four_week.weeks_since_start = 4**
WHERE initial.weeks_since_start = 0 ORDER BY four_week.retained_users / initial.retained_users DESC LIMIT 1; Predicted SQL:
Predicted Result:
Predicted SQL:
Gold Result:
SELECT 
quantile, COUNT(*) AS total_trips, AVG(fare) AS average_fare FROM ( SELECT 
trip_seconds, fare, NTILE(10) OVER (ORDER BY trip_seconds) 
AS quantile FROM Chicago.taxi_trips WHERE trip_seconds BETWEEN 60 AND 3000
) AS subquery GROUP BY quantile ORDER BY quantile SELECT 
DATE_TRUNC(
DATE(user_first_touch_timestamp), WEEK(MONDAY)
) AS week_cohort, COUNT(DISTINCT user_pseudo_id) AS active_users_next_four_weeks FROM `firebase-public-project...events_*` WHERE PARSE_DATE('%Y%m%d', event_date) >= DATE('2018-07-02')
AND DATE_DIFF(
PARSE_DATE('%Y%m%d', event_date), DATE(user_first_touch_timestamp), DAY ) BETWEEN 21 AND 27 GROUP BY week_cohort ORDER BY **active_users_next_four_weeks** DESC LIMIT 1; Predicted Result:
(a) (b)
Figure 5: Case study of two representative incorrect SQL predictions due to erroneous data analysis. (a): An example of **incorrect data calculation**, where quantiles were incorrectly divided based on the number of trips, rather than on the *trip duration* as required. (b): An example of **incorrect** planning, where the predicted SQL incorrectly sorted data by the number of users, rather than by the required *retention ratio*. The prerequisite for achieving this is to properly plan a sequence of CTEs. Additional examples of error cases across all categories are available in Fig. 22 and Fig. 23.

## 4.3 Analysis Of Different Experimental Settings

Providing oracle functions leads to a slight performance improvement. Considering that Spider 2.0 and Spider 2.0-lite involve SQL dialects from various database systems, we provide syntax and function documentation for each system to prevent the methods from suffering due to lack of syntax knowledge. For each example, we manually include the relevant function documentation that may be required, eliminating the need for a retrieval method and ensuring that the necessary syntax knowledge is readily accessible. As shown in Tab. 10, providing oracle SQL function documentation results in only a slight improvement in model performance. This suggests that, to a certain extent, models are capable of selecting appropriate functions and understanding their basic usage and syntax. However, the critical challenge lies in accurately utilizing these functions to reflect user intentions, as illustrated in Fig. 5(a). Few-shot prompting has little impact on performance. Spider 2.0-lite is not divided into train and dev sets, we manually select representative examples from the same SQL dialect as the SQL to be predicted, with distinct characteristics (encompassing multiple CTE or nested queries, or requiring intricate data processing) to serve as few-shot examples. Unexpectedly, few-shot in-context learning shows only marginal improvements in performance (see Tab. 11). This may be due to the gap between the simplistic text-to-SQL pre-training data used with LLMs and the complexity of the fewshot examples. Additionally, extensive schema prompts may hinder the model's ability to effectively assimilate information in the few-shot examples.

## 5 Related Work

Code generation and text-to-SQL benchmark. As model capabilities advance, code generation benchmarks have become more complex and generalized. Many benchmarks (e.g., SQL-Spider (Yu et al., 2018), Bash-NL2Bash (Lin et al., 2018), Python-HumanEval (Chen et al., 2021)) treat code generation as seq2seq tasks. Many previous works (Lai et al., 2023; Yin et al., 2023; Huang et al.,
2024; Chan et al., 2024; Jing et al., 2024) define code generation tasks for data science. MLAgent-
Bench (Huang et al., 2023) and Intercode (Yang et al., 2024b) focus on interactive environments, while SWE-Bench (Jimenez et al., 2023) emphasizes repository-level coding tasks. Spider2-V (Cao et al., 2024) proposes data science and engineering benchmark in a multimodal setting. Many previous datasets (Zhong et al., 2017; Lee et al., 2021; Hazoom et al., 2021; Wang et al., 2020; Li et al., 2024b) have made significant contributions to the advancement of text-to-SQL tasks. However, existing text-to-SQL benchmarks primarily target lightweight local databases, much smaller in schema scale and data volume than cluster-hosted industrial databases, and fail to capture the *agentic* nature of SQL programming using various dialects in real scenarios. Spider 2.0 bridges the gap between research and enterprise-level industrial text-to-SQL workflows. Code agent framework and text-to-SQL methods. The intersection of generative code models and interactive problem-solving has spurred significant advancements in both agent-based frameworks and text-to-SQL methodologies. Recent efforts aim to enhance the reasoning capabilities of language models, as evidenced by a surge in agent methods designed for code generation tasks (Yao et al., 2022; Zhang et al., 2022; Chen et al., 2023; Wang et al., 2023b; Shinn et al., 2024; Zhang et al., 2024; Xia et al., 2024). Several works have designed special actions to standardize agent operations (Wang et al., 2024; Yang et al., 2024a). For methods specifically designed for text-to-SQL, several fine-tuning methods (Li et al., 2024a) and LLM-prompting methods (Dong et al., 2023; Wang et al., 2023a; Zhang et al., 2023; Talaei et al., 2024; Pourreza & Rafiei, 2024; Gao et al., 2024) have achieved strong performance on previous benchmarks. We propose Spider-Agent, a code agent framework specifically designed for database-related tasks, showcasing strong performance in this domain. For Spider 2.0-lite, we also adapt several text-to-SQL methods to suit our benchmark.

## 6 Conclusion

We propose Spider 2.0, a benchmark for real-world enterprise-level text-to-SQL workflow tasks.

It encompasses diverse database systems with various SQL dialects, large and complex database schemas, and challenging tasks across the data engineering pipeline, all set within real project scenarios including codebases and documentation. Despite being the most advanced LLMs (o1preview), they still perform poorly on Spider 2.0, achieving a success rate of only 21.3%, which underscores its status as a highly challenging benchmark. Spider 2.0 presents a novel challenge for text-to-SQL research, providing a direction towards more realistic and intelligent solutions.

## Acknowledgements

The authors of this paper were supported by the ECS (27212023) from RGC of Hong Kong. We thank Snowflake for their generous support in hosting the Spider 2.0 Challenge. We also thank Tianbao Xie, Yiheng Xu, Fan Zhou, Yuting Lan, Per Jacobsson, Yiming Huang, Canwen Xu, Zhewei Yao and Binyuan Hui for their helpful feedback on this work.

## References

Anthropic. The claude 3 model family: Opus, sonnet, haiku. https://wwwcdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model Card Claude *3.pdf*, 2024.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. *arXiv preprint arXiv:2108.07732*, 2021.

Ruisheng Cao, Fangyu Lei, Haoyuan Wu, Jixuan Chen, Yeqiao Fu, Hongcheng Gao, Xinzhuang Xiong, Hanchong Zhang, Yuchen Mao, Wenjing Hu, et al. Spider2-v: How far are multimodal agents from automating data science and engineering workflows? arXiv preprint arXiv:2407.10956, 2024.

Jun Shern Chan, Neil Chowdhury, Oliver Jaffe, James Aung, Dane Sherburn, Evan Mays, Giulio Starace, Kevin Liu, Leon Maksin, Tejal Patwardhan, et al. Mle-bench: Evaluating machine learning agents on machine learning engineering. *arXiv preprint arXiv:2410.07095*, 2024.

Dong Chen, Shaoxin Lin, Muhan Zeng, Daoguang Zan, Jian-Gang Wang, Anton Cheshkov, Jun Sun, Hao Yu, Guoliang Dong, Artem Aliev, et al. Coder: Issue resolving with multi-agent and task graphs. *arXiv preprint arXiv:2406.01304*, 2024.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. *arXiv preprint arXiv:2107.03374*, 2021.

Xinyun Chen, Maxwell Lin, Nathanael Schaerli, and Denny Zhou. Teaching large language models to self-debug. In *The 61st Annual Meeting Of The Association For Computational Linguistics*, 2023.

Deborah A Dahl, Madeleine Bates, Michael K Brown, William M Fisher, Kate Hunicke-Smith, David S Pallett, Christine Pao, Alexander Rudnicky, and Elizabeth Shriberg. Expanding the scope of the atis task: The atis-3 corpus. In *Human Language Technology: Proceedings of a Workshop* held at Plainsboro, New Jersey, March 8-11, 1994, 1994.

Xuemei Dong, Chao Zhang, Yuhang Ge, Yuren Mao, Yunjun Gao, Jinshu Lin, Dongfang Lou, et al.

C3: Zero-shot text-to-sql with chatgpt. *arXiv preprint arXiv:2307.07306*, 2023.

Dawei Gao, Haibin Wang, Yaliang Li, Xiuyu Sun, Yichen Qian, Bolin Ding, and Jingren Zhou.

Text-to-sql empowered by large language models: A benchmark evaluation. Proceedings of the VLDB Endowment, 17(5):1132–1145, 2024.

Moshe Hazoom, Vibhor Malik, and Ben Bogin. Text-to-sql in the wild: A naturally-occurring dataset based on stack exchange data. In Proceedings of the 1st Workshop on Natural Language Processing for Programming (NLP4Prog 2021), pp. 77–87, 2021.

Qian Huang, Jian Vora, Percy Liang, and Jure Leskovec. Benchmarking large language models as ai research agents. *arXiv preprint arXiv:2310.03302*, 2023.

Yiming Huang, Jianwen Luo, Yan Yu, Yitong Zhang, Fangyu Lei, Yifan Wei, Shizhu He, Lifu Huang, Xiao Liu, Jun Zhao, et al. Da-code: Agent data science code generation benchmark for large language models. *arXiv preprint arXiv:2410.07331*, 2024.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R
Narasimhan. Swe-bench: Can language models resolve real-world github issues? In *The Twelfth* International Conference on Learning Representations, 2023.

Liqiang Jing, Zhehui Huang, Xiaoyang Wang, Wenlin Yao, Wenhao Yu, Kaixin Ma, Hongming Zhang, Xinya Du, and Dong Yu. Dsbench: How far are data science agents to becoming data science experts?, 2024. URL https://arxiv.org/abs/2409.07703.

Yuhang Lai, Chengxi Li, Yiming Wang, Tianyi Zhang, Ruiqi Zhong, Luke Zettlemoyer, Wen-tau Yih, Daniel Fried, Sida Wang, and Tao Yu. Ds-1000: A natural and reliable benchmark for data science code generation. In *International Conference on Machine Learning*, pp. 18319–18345. PMLR, 2023.

Chia-Hsuan Lee, Oleksandr Polozov, and Matthew Richardson. Kaggledbqa: Realistic evaluation of text-to-sql parsers. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 2261–2273, 2021.

Fei Li and HV Jagadish. Constructing an interactive natural language interface for relational databases. *Proceedings of the VLDB Endowment*, 8(1), 2014.

Haoyang Li, Jing Zhang, Hanbing Liu, Ju Fan, Xiaokang Zhang, Jun Zhu, Renjie Wei, Hongyan Pan, Cuiping Li, and Hong Chen. Codes: Towards building open-source language models for text-to-sql. *Proceedings of the ACM on Management of Data*, 2(3):1–28, 2024a.

Jinyang Li, Binyuan Hui, Ge Qu, Jiaxi Yang, Binhua Li, Bowen Li, Bailin Wang, Bowen Qin, Ruiying Geng, Nan Huo, et al. Can llm already serve as a database interface? a big bench for large-scale database grounded text-to-sqls. *Advances in Neural Information Processing Systems*,
36, 2024b.

Xi Victoria Lin, Chenglong Wang, Luke Zettlemoyer, and Michael D Ernst. Nl2bash: A corpus and semantic parser for natural language interface to the linux operating system. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), 2018.

Meta AI. Introducing meta Llama 3: The most capable openly available LLM to date, April 2024.

URL https://ai.meta.com/blog/meta-llama-3/. Accessed: 2024-04-18.

R OpenAI. Gpt-4 technical report. arxiv 2303.08774. *View in Article*, 2:13, 2023. Jiayi Pan, Yichi Zhang, Nicholas Tomlin, Yifei Zhou, Sergey Levine, and Alane Suhr. Autonomous evaluation and refinement of digital agents. In *First Conference on Language Modeling*, 2024.

Mohammadreza Pourreza and Davood Rafiei. Din-sql: Decomposed in-context learning of text-tosql with self-correction. *Advances in Neural Information Processing Systems*, 36, 2024.

Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jeanbaptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik R Narasimhan, and Shunyu Yao. Reflexion: language agents with verbal reinforcement learning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion:
Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36, 2024.

Shayan Talaei, Mohammadreza Pourreza, Yu-Chen Chang, Azalia Mirhoseini, and Amin Saberi.

Chess: Contextual harnessing for efficient sql synthesis. *arXiv preprint arXiv:2405.16755*, 2024.

Qwen Team. Qwen2.5: A party of foundation models, September 2024. URL https://qwenlm.

github.io/blog/qwen2.5/.

Bing Wang, Changyu Ren, Jian Yang, Xinnian Liang, Jiaqi Bai, Qian-Wen Zhang, Zhao Yan, and Zhoujun Li. Mac-sql: Multi-agent collaboration for text-to-sql. *arXiv preprint arXiv:2312.11242*, 2023a.

Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, and Ee-Peng Lim.

Plan-and-solve prompting: Improving zero-shot chain-of-thought reasoning by large language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2609–2634, 2023b.

Ping Wang, Tian Shi, and Chandan K Reddy. Text-to-sql generation for question answering on electronic medical records. In *Proceedings of The Web Conference 2020*, pp. 350–361, 2020.

Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang H. Tran, Fuqiang Li, Ren Ma, Mingzhang Zheng, Bill Qian, Yanjun Shao, Niklas Muennighoff, Yizhe Zhang, Binyuan Hui, Junyang Lin, Robert Brennan, Hao Peng, Heng Ji, and Graham Neubig. OpenDevin: An Open Platform for AI
Software Developers as Generalist Agents, 2024. URL https://arxiv.org/abs/2407. 16741.

Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, and Lingming Zhang. Agentless: Demystifying llm-based software engineering agents. *arXiv preprint arXiv:2407.01489*, 2024.

John Yang, Akshara Prabhakar, Karthik Narasimhan, and Shunyu Yao. Intercode: Standardizing and benchmarking interactive coding with execution feedback. CoRR, abs/2306.14898, 2023. doi: 10.48550/arXiv.2306.14898. URL https://doi.org/10.48550/arXiv.2306.14898.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. Swe-agent: Agent computer interfaces enable software engineering language models, 2024a.

John Yang, Akshara Prabhakar, Karthik Narasimhan, and Shunyu Yao. Intercode: Standardizing and benchmarking interactive coding with execution feedback. *Advances in Neural Information* Processing Systems, 36, 2024b.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, 2022.

Pengcheng Yin, Wen-Ding Li, Kefan Xiao, Abhishek Rao, Yeming Wen, Kensen Shi, Joshua Howland, Paige Bailey, Michele Catasta, Henryk Michalewski, et al. Natural language to code generation in interactive data science notebooks. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 126–173, 2023.

Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, et al. Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-sql task. In *Proceedings of the 2018 Conference* on Empirical Methods in Natural Language Processing, pp. 3911–3921, 2018.

John M Zelle and Raymond J Mooney. Learning to parse database queries using inductive logic programming. In *Proceedings of the national conference on artificial intelligence*, pp. 1050– 1055, 1996.

Luke S Zettlemoyer and Michael Collins. Learning to map sentences to logical form: structured classification with probabilistic categorial grammars. In Proceedings of the Twenty-First Conference on Uncertainty in Artificial Intelligence, pp. 658–666, 2005.

Hanchong Zhang, Ruisheng Cao, Lu Chen, Hongshen Xu, and Kai Yu. Act-sql: In-context learning for text-to-sql with automatically-generated chain-of-thought. In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 3501–3532, 2023.

Shun Zhang, Zhenfang Chen, Yikang Shen, Mingyu Ding, Joshua B Tenenbaum, and Chuang Gan.

Planning with large language models for code generation. In The Eleventh International Conference on Learning Representations, 2022.

Yuntong Zhang, Haifeng Ruan, Zhiyu Fan, and Abhik Roychoudhury. Autocoderover: Autonomous program improvement. In Proceedings of the 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis, pp. 1592–1604, 2024.

Victor Zhong, Caiming Xiong, and Richard Socher. Seq2sql: Generating structured queries from natural language using reinforcement learning. *arXiv preprint arXiv:1709.00103*, 2017.

Qihao Zhu, Daya Guo, Zhihong Shao, Dejian Yang, Peiyi Wang, Runxin Xu, Y Wu, Yukun Li, Huazuo Gao, Shirong Ma, et al. Deepseek-coder-v2: Breaking the barrier of closed-source models in code intelligence. *arXiv preprint arXiv:2406.11931*, 2024.

## A Spider 2.0 Evaluation Scripts

In this section, we present the detailed definition and discussion of evaluation metrics for Spider 2.0-lite and Spider 2.0.

Spider 2.0-lite. The setting of Spider 2.0-lite resembles that of a traditional text-to-SQL task in which text-to-SQL parsers are required to generate SQL queries. Therefore, Execution Accuracy(EX) is used as the primary evaluation metric. Slightly different from existing works, we employ an *execution-based focused evaluation*, which measures whether all columns in the gold value are present in the output of the predicted SQL query. This is defined as follows:

$$\text{EX}=\frac{\sum_{n=1}^{N}\mathbb{1}(v^{n},\hat{v}^{n})}{N},$$  where $\mathbb{1}(v,\hat{v})=\begin{cases}1,\text{if}v_{i}\in\hat{v},\forall v_{i}\in v\\ 0,\text{if}v_{i}\notin\hat{v},\exists v_{i}\in v\end{cases},$
$$(1)$$
$${\mathrm{(2)}}$$

where vi represents the i-th column of data frame v, v n and vˆ
n denote execution results of the gold SQL and predicted SQL for the n-th instance in the evaluation set, respectively. Empirically, this evaluation method significantly reduces the false negative rate without increasing the number of false positives. Given that the ground-truth values result from extensive data wrangling, transformation, and analysis, it is difficult for models to manipulate or exploit the system. Spider 2.0. We use the **Success rate (SR)**, which measures the proportion of task instances successfully resolved. Human-written evaluation scripts are used to determine whether an example is resolved. For each example, we provide string-based, table-based, and database-based evaluation functions, depending on the type of answer output, as shown in Tab. 12. Examples. Maintaining naturalness and unambiguity is often a conflicting challenge. To address this, we provide an example to illustrate the important parameters "condition cols" and "ignore order". Achieving a balance between these two aspects is quite challenging, which is why we incorporate this mechanism into our evaluation scripts.

Given a data frame v with a set of column vectors {vi}, each representing the cell values for the i-th column, a prediction vˆ is considered equivalent with v if and only if for any vi ∈ v, vi ∈ vˆ.

Therefore, at such times, we only check whether a specific column appears in it. Intuitively, if all columns in the reference table appear in the result table, the result is considered correct. For example, as illustrated in Fig. 6, the question does not explicitly specify which columns are required in our response. Consider the following question: "The company management has requested a detailed report on the year-to-date performance of the Magnificent 7 stocks.". We need to carefully analyze the task requirements and only check if the following columns in the reference answer—"Ticker", "Change *YTD"*—appear in the predicted answer. This meets the semantic requirements of the abstract instruction. Empirically, we find our evaluation metric is reliable in identifying solutions with alternative output, with a relatively low false-negative rate.

The company management has requested a detailed report on the year-to-date performance of the Magnificent 7 stocks. 

Task Gold Answer Agent get results Score = 1
Table 12: The evaluation scripts for Spider 2.0 are tailored to the specific format of the model's output. Each script is optimized to handle various output types, ensuring precise and contextually appropriate evaluation.

| appropriate evaluation. Output Type Description          | Parameters                                                                                                                                                                                                                                                                                   |                                                                                                 |         |    |
|----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|---------|----|
| String                                                   | If the answer is found                                                                                                                                                                                                                                                                       |                                                                                                 |         |    |
| w/o number                                               | in the string,                                                                                                                                                                                                                                                                               | it is                                                                                           |         |    |
| given a score of 1; otherwise, it receives a score of 0. | pred (str): The string in which to search for substrings. gold (List of str): A list of strings to check within the predicted string. conj (str): The conjunction used for matching ('and' or 'or'). Default is 'or'. exclude (List of Str): Strings that must not be present in the answer. |                                                                                                 |         |    |
| String                                                   | For                                                                                                                                                                                                                                                                                          | output                                                                                          | strings |    |
| w. number                                                | containing numbers, the script captures these numbers and performs number matching for scoring using the number match function.                                                                                                                                                                                                                                                                                              | pred (str): The string in which to search for substrings. gold (List[str|float]): A list of strings or numbers to check within the predicted string. percentage (bool): Default is false. If the gold answer is related to percentages, set this to true for more robust evaluation. precision (int): The number of decimal places to consider. Defaults to 4. conj (str): The conjunction used for matching ('and' or 'or'). Default is 'or', and it's typically 'or'.                                                                                                 |         |    |
| Table                                                    | If                                                                                                                                                                                                                                                                                           | the answer                                                                                      | is      | a  |
| CSV file or a table in string format, tablelevel evaluation is performed.                                                          | result (str): Path to the CSV file or result string. gold (str | List[str]): Path(s) to the gold file(s), excluding the root directory. Multiple potential gold answers are supported. condition cols (List[int] | List[List[int]]): List of column indices to match conditions. For example, [0, 1] uses the 0th and 1st columns in the gold table for matching, while ignoring the others. ignore order (bool): Whether to ignore the order of rows when matching elements.                                                                                                                                                                                                                                                                                              |                                                                                                 |         |    |
| Database                                                 | If                                                                                                                                                                                                                                                                                           | the                                                                                             | answer  | is |
| stored in a DB file, database-level evaluation is applied using the db match function.                                                          | result (str): Path to the DuckDB file containing the result tables. gold (str): Path to the DuckDB file containing the gold standard tables. condition tabs (List[str], optional): List of table names to be checked. If not provided, all tables in the gold DuckDB file will be considered. condition cols (List[List[int]], optional): A list of lists, where each inner list contains column indices used for matching conditions for the corresponding table. Defaults to considering all columns. ignore orders (List[bool], optional): A list of boolean values indicating whether to ignore the row order for each table comparison. Defaults to [False] for each table.                                                                                                                                                                                                                                                                                              |                                                                                                 |         |    |
| SQL                                                      | If the output is an SQL, executionbased evaluation is used. This is primarily designed for Spider 2.0-lite.                                                                                                                                                                                                                                                                                              | To compare the execution results of the predicted SQL and the gold SQL, table matching is used. |         |    |

## B Annotation Details B.1 Sql Annotation Examples

In this section, we present several representative examples of the SQL annotation process, including the original SQL, how the SQL was rewritten to obtain the gold SQL, and the use of external knowledge. Tab. 13 presents an example based on the Google Analytics database. The task is to calculate the source of web traffic and count the number of sessions for each traffic channel within a given time period. Tab. 14 presents an example based on New York City public data, where the task is to find Citibike and taxi trips between specified locations and analyze whether Citibike or taxi is more suitable for travel between the two locations. In this case, the condition in the original SQL is to calculate trips between the two locations by Citibike and car. We extend this condition by introducing a real-life problem: identifying which routes are faster by Citibike compared to taxi. Tab. 15 is based on the Google Patents database, which contains a large amount of patent information. The original SQL applied several filtering conditions to retrieve a set of patents. We find a document explaining how to calculate a patent's originality score, which led to an advanced calculation method. As a result, the final task include additional complex calculation steps. Tab. 16 is also based on the Google Analytics database. The original SQL calculates the Product List Page (PLP) and Product Details Page (PDP). Based on the description in the blog, we define a new task to calculate the conversion rate by determining the probability of users clicking from PLP to PDP. Tab. 17 presents an example where we merge and rewrite two related SQL queries. The first SQL calculates the 50 weather stations closest to downtown Chicago, while the second SQL calculates the number of bike trips on rainy and non-rainy days in New York City. We combine these two tasks, meaning that to determine whether it is a rainy day, we first need to find data from the weather station closest to downtown New York City.

## B.2 Dbt Project Annotation Examples

Annotation Pipeline of DBT Project. The DBT project can be found on online resources and is one of the projects with the most SQL scripts. Similar data transformation tools are widely used in industrial production. Completing a DBT project requires a comprehensive understanding of both the code and documentation within the project to accomplish the entire task. Fig. 7 shows a Salesforce-based project in Spider 2.0. This represents a natural and realistic SQL generation scenario. Using a Fivetran Salesforce transformation package 2as an example, we transform a complex DBT project into a Spider 2.0 example through the following steps. (1) Run a DBT project from start to finish, ensuring it is bug-free and generates a dbt DAG (Fig. 9). This allows for a comprehensive understanding of the data flow. (2) The DBT project includes yml files and markdown documents, where the project developers have already planned out the data models and data flow. We will use these as the basis for task instructions.

2https://github.com/fivetran/dbt_salesforce/
Figure 8: This is a common configuration file in DBT projects used to define the schema of a data model. It represents a natural SQL generation scenario, specifying details such as field names, data types, and references for the "salesforce opportunity enhanced" data model. (3) We remove the .sql files corresponding to a specific data flow within the complete DBT project. For example, in Fig. 9, we may delete one to three data flows, as shown in Fig. 10, removing "sales daily activity" and "salesforce contact enhanced" along with their upstream nodes. This turns it into an incomplete transformation project. Note that the DAG figure is only used as an aid for data annotation, and the task does not include any images. (4) Write the task instructions. For instance, we can draft a prompt like, "I need a daily report on key sales activities—covering tasks completed, events held, leads generated, and the status of opportunities." Although the data model contains many columns, thanks to the presence of yml files (see Fig. 8), there is no need to describe the output columns in detail in the instructions. Approach to Solving DBT Project Examples. As shown in Fig. 26, completing a DBT project example typically requires the following abilities:
1) Problem comprehension. First, it is necessary to fully understand a natural language task.

2) Project reading ability. A real-world data transformation project consists of multiple files, as illustrated in Fig. 7. The method needs to explore the codebase and review relevant project files, including .yml, .md, and *.sql* files. YML files (Fig. 8) generally define the data models for the data transformation, .md files contain textual descriptions of the data models, and SQL files are the data transformation models themselves. Figure 9: A DAG (Directed Acyclic Graph) illustrating the data flow and dependencies between various Salesforce tables and models in a dbt (data build tool) project. The graph shows stages of transformation, from raw Salesforce data (green nodes) to enhanced and aggregated models (blue nodes), representing different entities such as opportunities, contacts, accounts, and events. 3) Database exploration ability. The codebase only contains the data transformation code, while the data to be transformed is stored in a database. The method must explore the database to understand the available source data and identify any missing data models. 4) Problem localization ability. By combining the natural language problem and the YML files, the method should locate where to add or modify the code in the project. 5) Coding ability. The method needs to complete complex data transformation code based on the data models defined in the YML files and add the .sql files in the appropriate locations. Visually, it requires completing the data models defined in the yml file, transitioning from Fig. 10 to Fig. 9.

6) Data transformation execution. Once the SQL is written, it is necessary to run dbt run to execute the data transformation. 7) Debugging. After running the DBT project, if the data transformation is successful, the data models (the tables) in the database will change, with tables being added or removed. The method needs to examine the database to determine if the transformation was fully successful. If not, the above steps must be repeated until the method meets the problem's requirements.

Table 13: Google analytics traffic session examples, using *answer format surface* rewrite.

## Question

Provide the number of sessions and percentage breakdown by channel for December 2020.

## Reference Plan

1. First, read the document to understand how traffic is divided into 18 channel groups, primarily based on the metrics of source, medium, and campaign. 2. Extract all visits from the database for December, each visit having a unique user ID and session ID. Retrieve the source, medium, and campaign for each visit. 3. Based on the classification standards for channel groups in the document, write conditional statements to determine which channel each set of data belongs to, mainly using regular expressions. If the data source (source) contains any of the following: 'badoo', 'facebook', 'fb', 'instagram', 'linkedin', 'pinterest', 'tiktok', 'twitter', or 'whatsapp', and the medium (medium) includes 'cp', 'ppc', or starts with 'paid', then categorize it as 'Paid Social'. 4. Calculate the number of sessions and percentage for each channel based on the channel grouping.

Gold SQL (After rewriting)
WITH prep AS ( SELECT user_pseudo_id,
(SELECT value.int_value FROM UNNEST(event_params) WHEREkey = 'ga_session_id') AS session_id, ARRAY_AGG((SELECTvalue.string_value FROM UNNEST(event_params) WHERE key = 'source')
IGNORE NULLS ORDER BY event_timestamp)[SAFE_OFFSET(0)] AS source, ARRAY_AGG((SELECTvalue.string_value FROM UNNEST(event_params) WHEREkey = 'medium')
IGNORE NULLS ORDER BY event_timestamp)[SAFE_OFFSET(0)] AS medium, ARRAY_AGG((SELECTvalue.string_value FROM UNNEST(event_params) WHEREkey = 'campaign')
IGNORE NULLS ORDER BY event_timestamp)[SAFE_OFFSET(0)] AS campaign FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*` WHERE _TABLE_SUFFIX BETWEEN'20201201'AND'20201231' GROUP BY user_pseudo_id, session_id ) grouped_data AS ( SELECT CASE WHEN source = '(direct)'AND (medium IN ('(not set)', '(none)')) THEN'Direct',
WHEN REGEXP_CONTAINS(campaign, 'cross-network') THEN'Cross-network' WHEN (REGEXP_CONTAINS(source, 'alibaba|amazon|google shopping|shopify|etsy|ebay|stripe|walmart')
OR REGEXP_CONTAINS(campaign, '^(.*(([^a-df-z]|^)shop|shopping).*)$')) AND REGEXP_CONTAINS(medium, '^(.*cp.*|ppc|paid.*)$') THEN'Paid Shopping' ,
WHEN REGEXP_CONTAINS(source, 'baidu|bing|duckduckgo|ecosia|google|yahoo|yandex')
AND REGEXP_CONTAINS(medium, '^(.*cp.*|ppc|paid.*)$') THEN 'Paid Search' ,
WHEN REGEXP_CONTAINS(source, 'badoo|facebook|fb|instagram|linkedin|pinterest|tiktok|twitter|whatsapp')
AND REGEXP_CONTAINS(medium, '^(.*cp.*|ppc|paid.*)$') THEN'Paid Social' ,
WHEN REGEXP_CONTAINS(source, 'dailymotion|disneyplus|netflix|youtube|vimeo|twitch|vimeo|youtube')
AND REGEXP_CONTAINS(medium, '^(.*cp.*|ppc|paid.*)$') THEN'Paid Video' ,
WHEN medium IN ('display', 'banner', 'expandable', 'interstitial', 'cpm') THEN 'Display' WHEN REGEXP_CONTAINS(source, 'alibaba|amazon|google shopping|shopify|etsy|ebay|stripe|walmart')
OR REGEXP_CONTAINS(campaign, '^(.*(([^a-df-z]|^)shop|shopping).*)$') THEN'Organic Shopping' WHEN REGEXP_CONTAINS(source, 'badoo|facebook|fb|instagram|linkedin|pinterest|tiktok|twitter|whatsapp')
OR medium IN ('social', 'social-network', 'social-media', 'sm', 'social network', 'social media') THEN'Organic Social' WHEN REGEXP_CONTAINS(source, 'dailymotion|disneyplus|netflix|youtube|vimeo|twitch|vimeo|youtube')
OR REGEXP_CONTAINS(medium, '^(.*video.*)$') THEN'Organic Video' WHEN REGEXP_CONTAINS(source, 'baidu|bing|duckduckgo|ecosia|google|yahoo|yandex') OR medium = 'organic' THEN'Organic Search' WHEN REGEXP_CONTAINS(source, 'email|e-mail|e_mail|e mail') OR REGEXP_CONTAINS(medium, 'email|e-mail|e_mail|e mail') THEN'Email' WHEN medium = 'affiliate' THEN 'Affiliates' WHEN medium = 'referral'THEN'Referral' WHEN medium = 'audio'THEN'Audio' WHEN medium = 'sms'THEN'SMS' WHEN medium LIKE'%push'OR REGEXP_CONTAINS(medium, 'mobile|notification') THEN'Mobile Push Notifications' ELSE'Unassigned' END AS channel_grouping_session, COUNT(DISTINCTCONCAT(user_pseudo_id, session_id)) AS session_count FROM
prep GROUP BY channel_grouping_session
), total_sessions AS ( SELECT SUM(session_count) AS total_count FROM grouped_data ) SELECT gd.channel_grouping_session, gd.session_count, ROUND(gd.session_count / ts.total_count * 100, 2) AS percentage_of_total FROM grouped_data gd, total_sessions ts ORDER BY gd.session_count DESC LIMIT 10; Original SQL
WITH prep AS ( SELECT user_pseudo_id,
(SELECT value.int_value FROM UNNEST(event_params) WHEREkey = 'ga_session_id') AS session_id, ARRAY_AGG((SELECTvalue.string_value FROM UNNEST(event_params) WHERE key = 'source')
IGNORE NULLS ORDER BY event_timestamp)[SAFE_OFFSET(0)] AS source, ARRAY_AGG((SELECTvalue.string_value FROM UNNEST(event_params) WHEREkey = 'medium')
IGNORE NULLS ORDER BY event_timestamp)[SAFE_OFFSET(0)] AS medium, ARRAY_AGG((SELECTvalue.string_value FROM UNNEST(event_params) WHEREkey = 'campaign')
IGNORE NULLS ORDER BY event_timestamp)[SAFE_OFFSET(0)] AS campaign FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
WHERE _TABLE_SUFFIX BETWEEN'20201201'AND'20201231' GROUP BY user_pseudo_id, session_id ) grouped_data AS (
SELECT CASE WHEN source = '(direct)'AND (medium IN ('(not set)', '(none)')) THEN'Direct',
WHEN REGEXP_CONTAINS(campaign, 'cross-network') THEN'Cross-network' WHEN (REGEXP_CONTAINS(source, 'alibaba|amazon|google shopping|shopify|etsy|ebay|stripe|walmart')
OR REGEXP_CONTAINS(campaign, '^(.*(([^a-df-z]|^)shop|shopping).*)$'))
AND REGEXP_CONTAINS(medium, '^(.*cp.*|ppc|paid.*)$') THEN'Paid Shopping' ,
WHEN REGEXP_CONTAINS(source, 'baidu|bing|duckduckgo|ecosia|google|yahoo|yandex')
AND REGEXP_CONTAINS(medium, '^(.*cp.*|ppc|paid.*)$') THEN 'Paid Search' ,
WHEN REGEXP_CONTAINS(source, 'badoo|facebook|fb|instagram|linkedin|pinterest|tiktok|twitter|whatsapp')
AND REGEXP_CONTAINS(medium, '^(.*cp.*|ppc|paid.*)$') THEN'Paid Social' ,
WHEN REGEXP_CONTAINS(source, 'dailymotion|disneyplus|netflix|youtube|vimeo|twitch|vimeo|youtube')
AND REGEXP_CONTAINS(medium, '^(.*cp.*|ppc|paid.*)$') THEN'Paid Video' ,
WHEN medium IN ('display', 'banner', 'expandable', 'interstitial', 'cpm') THEN 'Display' WHEN REGEXP_CONTAINS(source, 'alibaba|amazon|google shopping|shopify|etsy|ebay|stripe|walmart')
OR REGEXP_CONTAINS(campaign, '^(.*(([^a-df-z]|^)shop|shopping).*)$') THEN'Organic Shopping' WHEN REGEXP_CONTAINS(source, 'badoo|facebook|fb|instagram|linkedin|pinterest|tiktok|twitter|whatsapp')
OR medium IN ('social', 'social-network', 'social-media', 'sm', 'social network', 'social media') THEN'Organic Social' WHEN REGEXP_CONTAINS(source, 'dailymotion|disneyplus|netflix|youtube|vimeo|twitch|vimeo|youtube')
OR REGEXP_CONTAINS(medium, '^(.*video.*)$') THEN'Organic Video' WHEN REGEXP_CONTAINS(source, 'baidu|bing|duckduckgo|ecosia|google|yahoo|yandex') OR medium = 'organic' THEN'Organic Search' WHEN REGEXP_CONTAINS(source, 'email|e-mail|e_mail|e mail') OR REGEXP_CONTAINS(medium, 'email|e-mail|e_mail|e mail') THEN'Email' WHEN medium = 'affiliate' THEN 'Affiliates' WHEN medium = 'referral'THEN'Referral' WHEN medium = 'audio'THEN'Audio' WHEN medium = 'sms'THEN'SMS'
WHEN medium LIKE'%push'OR REGEXP_CONTAINS(medium, 'mobile|notification') THEN'Mobile Push Notifications' ELSE'Unassigned' END AS channel_grouping_session, COUNT(DISTINCTCONCAT(user_pseudo_id, session_id)) AS session_count FROM
prep GROUP BY channel_grouping_session
),
ORDER BY
COUNT(DISTINCT CONCAT(user_pseudo_id, session_id)) DESC
LIMIT 1