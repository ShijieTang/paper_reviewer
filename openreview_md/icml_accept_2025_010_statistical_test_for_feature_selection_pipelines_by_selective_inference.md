# 

Tomohiro Shiraishi * 1 2 **Tatsuya Matsukawa** * 1 **Shuichi Nishino** 1 2 **Ichiro Takeuchi** 1 2

## Abstract

A data analysis pipeline is a structured sequence of steps that transforms raw data into meaningful insights by integrating various analysis algorithms. In this paper, we propose a novel statistical test to assess the significance of data analysis pipelines. Our approach enables the systematic development of valid statistical tests applicable to any feature selection pipeline composed of predefined components. We develop this framework based on selective inference, a statistical technique that has recently gained attention for data-driven hypotheses. As a proof of concept, we focus on feature selection pipelines for linear models, composed of three missing value imputation algorithms, three outlier detection algorithms, and three feature selection algorithms. We theoretically prove that our statistical test can control the probability of false positive feature selection at any desired level, and demonstrate its validity and effectiveness through experiments on synthetic and real data. Additionally, we present an implementation framework that facilitates testing across any configuration of these feature selection pipelines without extra implementation costs.

## 1. Introduction

In practical data-driven decision-making tasks, integrating various types of data analysis steps is crucial for addressing diverse challenges. For instance, in genetic research aimed at identifying genes linked to a specific disease, the process often begins with preprocessing tasks such as filling in missing values and detecting outliers. This is followed by screening for potentially related genes using simple descriptive statistics and then applying more complex machine learning-based feature selection algorithms. Such a systematic sequence of steps designed to analyze data and derive
*Equal contribution 1Nagoya University, Aichi, Japan 2RIKEN, Tokyo, Japan. Correspondence to: Ichiro Takeuchi
<takeuchi.ichiro.n6@f.mail.nagoya-u.ac.jp>.

useful insights is known as a *data analysis pipeline*, which plays a key role in ensuring the reproducibility and reliability of data-driven decision-making.

In this study, as an example of data analysis pipelines, we consider a class of feature selection pipelines that integrates various missing-value imputations (MVI) algorithms, outlier detection (OD) algorithms, and feature selection (FS) algorithms. Figure 1 shows examples of two such pipelines. The pipeline on the left starts with a mean value imputation algorithm, followed by L1 regression based OD algorithm, proceeds with marginal screening to refine feature candidates, and concludes by using two FS algorithms—stepwise feature selection and Lasso—selecting their union as the final features. The pipeline on the right initiates with regression imputation, continues with marginal screening to narrow down feature candidates, uses Cook's distance for OD, and applies both stepwise FS and Lasso, ultimately choosing the intersection of their results as the final features. When a data-driven approach is used for high-stakes decision-making tasks such as medical diagnosis, it is crucial to quantify the reliability of the final results by considering all steps in the pipeline. The goal of this study is to develop a statistical test for a specific class of feature selection pipelines in linear models, allowing the statistical significance of features obtained through the pipeline to be properly quantified in the form of p-values. The first technical challenge in achieving this is the need to appropriately account for the complex interrelations between pipeline components to determine the overall statistical significance. The second challenge is to develop a universal framework capable of performing statistical tests on arbitrary pipelines (within a given class) rather than creating individual tests for each pipeline. To address these challenges, we introduce the concept of selective inference (SI) (Taylor & Tibshirani, 2015; Fithian et al., 2015; Lee & Taylor, 2014), a novel statistical inference approach that has gained significant attention over the past decade. The core idea of SI is to characterize the process of selecting hypotheses from the data and calculate the corresponding p-values using the sampling distribution, conditional on this selection process. We propose an approach based on SI that provides valid p-values for any feature selec1

Example 1 Example 2 5:FS lasso 4:FS
marginal screening 4:FS
stepwise 5:FS lasso 4:OD
cook distance 4:FS
stepwise intersection of features 1:MVI
mean value 2:OD
l1 regression
(soft-ipod)
6:U-F
regression 2:FS
marginal screening 1:MVI
6:I-F
union of features
tion pipeline configuration within the aforementioned class. We also introduce a modular implementation framework that supports SI for any pipeline configuration within this class without requiring additional implementation efforts. Specifically, with our framework, the statistical significance of features from any pipeline in this class can be quantified as valid p-values when used in a linear model, with no extra implementation required beyond specifying the pipeline. We note that our long-term goal beyond this current study is to ensure the reproducibility of data-driven decision-making by accounting for the entire pipeline from raw data to the final results, with the current study on a class of feature selection pipelines in linear models serving as a proof of concept for that goal.

Related Work. Most research on data analysis pipelines is concentrated in the field of software engineering rather than machine learning (Sugimura & Hartl, 2018; Hapke &
Nelson, 2020; Drori et al., 2021), with a primary focus on the design, implementation, testing, and maintenance of pipeline systems to ensure efficiency, scalability, and robustness. Meanwhile, AutoML has emerged as a related area where researchers are automating the construction of these pipelines, and many companies have developed tools for this purpose (Microsoft, 2018; Amazon, 2019; Google, 2021). However, to the best of our knowledge, there is no existing studies that systematically discusses the reliability of data analysis pipelines. Resampling techniques, such as crossvalidation (CV), are commonly used to evaluate the entire data analysis process. However, practical data analysis often includes unsupervised learning tasks like MVIs and ODs, where resampling cannot be used to accurately evaluate the reliability of the entire pipeline. Additionally, dividing the data reduces the sample size, leading to decreased accuracy in hypothesis selection and statistical power. SI has gained attention as a statistical inference method for feature selection in linear models (Taylor & Tibshirani, 2015; Fithian et al., 2015). It has been applied to various feature selection algorithms such as marginal screening (Lee & Taylor, 2014), stepwise FS (Tibshirani et al., 2016), and Lasso (Lee et al., 2016), and extended to more complex methods (Yang et al., 2016; Suzumura et al., 2017; Hyun et al., 2018; Rugamer & Greven ¨ , 2020; Das et al., 2022; Rugamer et al. ¨ , 2022). SI is valuable not only for FS in linear models but also for inference across various datadriven hypotheses, including tasks like OD (Chen & Bien, 2020; Tsukurimichi et al., 2022), segmentation (Tanizaki et al., 2020; Duy et al., 2022; Le Duy et al., 2024), clustering (Lee et al., 2015; Gao et al., 2022), and change-point detection (Duy et al., 2020; Jewell et al., 2022). The core idea of SI is to perform statistical inference using a distribution conditioned on events of hypothesis selection, with the technical challenge being the characterization of various event selections for different tasks. While studies on SI
for various tasks are being conducted, existing research is limited to single tasks, and how to perform inference when integrating multiple tasks into a pipeline remains an open question. Furthermore, existing implementations of SI are developed individually for each task, and there is no unified framework for implementing SI. Contributions. Our contributions in this study are threefold. First, we develop a statistical test for feature selection pipelines composed of various configurations of missing value imputation (MVI), outlier detection (OD), and feature selection (FS) components, based on the SI framework. Second, this study represents the first application of SI to inference on a combination of multiple analysis components in a unified, systematic manner. Finally, we provide a practical computational framework implemented as the Python package1, which facilitates the construction of statistical tests across any pipeline configuration without additional implementation costs. For reproducibility, our experimental code is available at https:
//github.com/shirara1016/statistical_
test_for_feature_selection_pipelines.

## 2. Preliminaries

Given a set of algorithm components, a pipeline is defined by selecting some components from the set and connecting the selected components in an appropriate way. A pipeline can be represented as a directed acyclic graph (DAG) with components as nodes, and the connections as edges. In this study, as an example class of pipelines, we consider a set of 1https://pypi.org/project/si4pipeline/
algorithms consisting of three MVI algorithms, three OD algorithms, three FS algorithms, as well as *Intersection* and Union operations (specific three algorithms each for MVI, OD, and FS are described later in this section). Figure 1 shows two examples of pipelines within this class. Note that each FS algorithm corresponds to a single node in the DAG, and the FS algorithms are not described as DAGs. Problem Setting. In this study, we consider the problem of feature selection for linear models from a dataset containing missing values and/or outliers using the aforementioned class of feature selection pipelines. Let us consider a linear regression problem with n instances and d features. We denote the observed dataset as (X, y), where X ∈ R
n×dis the fixed design matrix, while y ∈ R
n
′is the response vector which contains outlying values but excludes missing values (i.e., n
′ ≤ n). We assume that y is a random realization of the following random response vector

$$Y=\mu(X)+\varepsilon,\;\varepsilon\sim{\mathcal{N}}({\bf0},\sigma^{2}I_{n^{\prime}}),$$

where µ(X) ∈ R
n
′is the unknown true value function, while ε ∈ R
n
′is independently normally distributed with variance σ 2 which is known or estimable from an independent dataset2. Although we do not pose any functional form on the true value function µ(X) for theoretical justification, we consider a case where the true values µ(X) are reasonably approximated by a linear model as long as they are non-outliers. This is a common setting in the field of SI, referred to as the *saturated model* setting. Furthermore, we denote the response vector with imputed missing values as y
(+) ∈ R
n. Using the above notations, a feature selection pipeline comprising of MVI, OD, and FS algorithm components is represented as a mapping:

$$\mathcal{P}:\mathbb{R}^{n\times d}\times\mathbb{R}^{n^{\prime}}\ni(X,\boldsymbol{y})\mapsto(\boldsymbol{y}^{(+)},\mathcal{O},\mathcal{M})\in\mathbb{R}^{n}\times2^{[n]}\times2^{[d]}.\tag{2}$$

[d],
(2)
where y
(+) ∈ R
n is the response vector with missing values imputed, O ⊂ [n] is the set of detected outliers, and M ⊂
[d] is the set of selected features. Statistical Test for Pipelines. Given the output of a pipeline in (2), the statistical significance of the finally selected features can be quantified based on the coefficients of the linear model fitted only with the selected features from a dataset with missing values imputed and outliers removed. To formalize this, we denote the design matrix after removing outliers and composed only of the selected features as X−O,M ∈ R
n*−|O|×|M|*, and denote the response vector with outliers removed and missing values imputed as y
(+)
−O. Using these notations, the least squares solution of 2We discuss the robustness of the proposed method when the variance is unknown and the noise deviates from the Gaussian distribution in Appendix E.

the linear model after imputation of missing values, removal of outliers, and feature selection is expressed as βˆ =X⊤
−O,MX−O,M
−X⊤
−O,My
(+)
−O.

Similarly, we consider the population least-square solution for the unobservable true value vector µ(X) in (1), which is defined as β
∗ =X⊤
−O,MX−O,M
−X⊤−O,Mµ
(+)

$$.(X_{-}\circ.{\mathcal{M}}),$$

where µ
(+)
−O(X−O,M) ∈ R
n*−|O|* is an n *− |O|*-dimensional vector obtained by providing X−O,M to the unknown true function µ with the missing values imputed with the same MVI algorithm. To quantify the statistical significance of the selected features, we consider the following null hypothesis H0 and the alternative hypothesis H1:
H0 : β
∗
j = 0 v.s. H1 : β
∗
j ̸= 0, j ∈ M, (3)

$$(3)$$

$$(1)$$

where, with a slight abuse of notation, β
∗
jand βˆj respectively indicates the element of β
∗and βˆ corresponding to the selected feature j ∈ M.

Missing-Value Imputation (MVI) Algorithm Components. In this paper, as three examples of MVI algorithms, we consider mean value imputation, nearest-neighbor imputation, and *regression imputation* algorithms (see Appendix A.1). A MVI algorithm component is represented as fMVI : {X, y, O,M} 7→ {X, y
+, O,M},
where, among the four variables, only y is updated to y
(+),
but note that this notation is used to uniformly handle all components in the pipeline. It is important to note that these three MVI algorithms are *linear* algorithm in the sense that, using a matrix DX ∈ R
n×n
′that depends on X, the imputed values are written as y
(+) = DXy.

$\square$
Outlier Detection (OD) Algorithm Components. In this paper, as three examples of OD algorithms, we consider Cook's distance-based OD, *DFFITS OD*, and L1 regression based OD algorithms (see Appendix A.2). A OD algorithm component is represented as fOD : {X, y
(+), O,*M} 7→ {*X, y
(+), O
′,M},
where, O′is the updated set of outliers. Note that, if outlier removal and feature selection have not yet been performed, the sets O and M are initialized as O = ∅ and M = [d].

Feature Selection (FS) Algorithm Components. In this paper, as three examples of FS algorithms, we consider marginal screening, *stepwise feature selection*, and *Lasso* algorithms (see Appendix A.3). A FS algorithm component is represented as fFS : {X, y
(+), O,*M} 7→ {*X, y
(+), O,M′},
where, M′is the updated set of features.

Union and Intersection Components. When using multiple OD/FS algorithms, it is necessary to include components in the pipeline that perform the union/intersection of the detected outliers or selected features. Such union/intersection components for OD/FS are respectively written as

$$f_{\Sigma}^{\mathcal{O}}\colon\{X,\mathbf{y}^{(+)},\{\mathcal{O}_{e}\}_{e\in[E]},\mathcal{M}\}\mapsto\{X,\mathbf{y}^{(+)},\Sigma_{e\in[E]}\mathcal{O}_{e},\mathcal{M}\},$$
$$f_{\Sigma}^{\mathcal{M}}\colon\{X,\boldsymbol{y}^{(+)},\mathcal{O},\{\mathcal{M}_{e}\}_{e\in[E]}\}\mapsto\{X,\boldsymbol{y}^{(+)},\mathcal{O},\Sigma_{e\in[E]}\mathcal{M}_{e}\},$$

where E is the number of OD/FS algorithms and an operator Σ indicates either union or intersection of multiple sets.

Automatic Pipeline Construction. In this study, we consider two cases for pipeline configuration: an option specified by the user and an option determined based on the data. In the first option, the user can select some of the aforementioned data analysis components and specify their own configuration. On the other hand, the second option allows for the selection of the optimal configuration from among multiple pre-defined pipeline configurations based on CV. An important point in the second option is that our statistical test is designed by properly considering the fact that the optimal pipeline configuration has been selected based on the data3. For more details on the second option, see §6 and Appendix F. Selective Inference. For the statistical test in (3), it is reasonable to use βˆj , j ∈ M as the test statistic. An important point when addressing this statistical test within the SI approach is that the test statistic is represented as a linear function of the observed response vector as βˆj = η
⊤
j y, j ∈ M,
where ηj ∈ R
n
′, j ∈ M is a vector that depends on y only through the detected outlier set O and the selected feature set M 4. In SI, this property is utilized to perform statistical inference based on the sampling distribution of the test statistic conditional on O and M. More specifically, since y follows a normal distribution, it can be derived that the sampling distribution of the test statistic βˆj = η
⊤
j y, j ∈ M
conditional on O, M, and the sufficient statistic of the nuisance parameters follows a truncated normal distribution. By computing p-values based on this conditional sampling distribution represented as a truncated normal distribution, it is ensured that the type I error can be controlled even in finite samples. For more details on SI, please refer to the following sections or literatures such as Taylor & Tibshirani (2015); Fithian et al. (2015); Lee & Taylor (2014).

3As stated in §1, CV cannot be used for an accurate evaluation of a pipeline when it includes unsupervised learning components such as MVI or OD. However, it is possible to compute a valid p-value for a pipeline selected by CV if we properly consider the CV-based pipeline selection as part of the selection event for SI.

4Note that the MVI algorithms considered in this paper depend only on X, not on y.

## 3. Selective Inference For Feature Selection Pipelines

To perform statistical test for pipelines, it is necessary to consider how the data influenced the final result through the calculations of each algorithm component of the pipeline and in operations where they are combined with a specified configuration. We address this challenge using the SI framework. In the SI, statistical inference is performed based on the sampling distribution conditional on the process by which the data selects the final result, thereby incorporating the influence of how data is processed in the pipeline.

Selective Inference. In SI, p-values are computed based
on the null distribution conditional on an event that a certain
hypothesis is selected. The goal of SI is to compute a pvalue such that
$$\mathbb{P}_{\mathrm{H}_{0}}\left(p\leq\alpha\mid\mathcal{M}_{\boldsymbol{Y}}=\mathcal{M},\mathcal{O}_{\boldsymbol{Y}}=\mathcal{O}\right)=\alpha,\ \forall\alpha\in(0,1),\tag{4}$$
where MY and OY respectively indicate the random set
of selected features and detected outliers given the random response vector Y , thereby making the p-value is a random
variable. Here, the condition part MY = M and OY = O
in (4) indicates that we only consider response vectors Y
yielding a certain feature set M and a certain outlier set O.
If the conditional type I error rate can be controlled as in (4)
for any possible hypotheses (M, O) ∈ 2
[d] × 2
[n], then, by
the law of total probability, the marginal type I error rate
can also be controlled for any α ∈ (0, 1) because
$\mathbb{P}_{\mathrm{H}_{0}}(p\leq\alpha)$  $$=\sum_{\mathcal{M}\in2^{[d]}}\sum_{\mathcal{O}\in2^{[n]}}\mathbb{P}_{\mathrm{H}_{0}}(\mathcal{M},\mathcal{O})$$ $$=\alpha.$$
$$\mathbb{P}_{\mathrm{H}_{0}}(p\leq\alpha)$$
Therefore, in order to perform valid statistical test, we can employ p-values conditional on the hypothesis selection event. To compute a p-value that satisfies (4), we need to derive the sampling distribution of the test-statistic

$$T(\mathbf{Y})\mid\{{\mathcal{M}}_{\mathbf{Y}}={\mathcal{M}}_{\mathbf{y}},{\mathcal{O}}_{\mathbf{Y}}={\mathcal{O}}_{\mathbf{y}}\}.\qquad\qquad(5)$$

Selective p**-value.** To conduct statistical hypothesis testing based on the conditional sampling distribution in (5), we introduce an additional condition on the sufficient statistic of the nuisance parameter QY , defined as

$$\mathcal{Q}_{Y}=\left(I_{n^{\prime}}-\frac{\eta\eta^{\top}}{\|\eta\|^{2}}\right)Y.\tag{6}$$

4 Based on the additional conditioning on QY , the following theorem tells that the conditional p-value that satisfies (4) can be derived by using a truncated normal distribution. Theorem 3.1. Consider a constant design matrix X, a random response vector Y ∼ N (µ, σ2In′ ) *and an observed* response vector y. Let (MY , OY ) and (My, Oy) *be the* pairs of selected features and detected outliers, obtained by applying a pipeline process P in the form of (2) to (X,Y )
and (X, y)*, respectively. Let* η ∈ R
n
′be a vector depending on (My, Oy)*, and consider a test-statistic in the form* of T(Y ) = η
⊤Y . Furthermore, define the nuisance parameter QY *as in* (6).

Then, the conditional distribution

$$T(\mathbf{Y})\mid\{{\mathcal{M}}_{\mathbf{Y}}={\mathcal{M}}_{\mathbf{y}},{\mathcal{O}}_{\mathbf{Y}}={\mathcal{O}}_{\mathbf{y}},\,{\mathcal{Q}}_{\mathbf{Y}}={\mathcal{Q}}_{\mathbf{y}}\}$$

is a truncated normal distribution TN(η
⊤µ, σ2∥η∥
2, Z)
with mean η
⊤µ*, variance* σ 2∥η∥
2, and truncation intervals Z, where Z *is defined as*

$$\mathcal{Z}=\left\{z\in\mathbb{R}\mid\mathcal{M}_{a+bz}=\mathcal{M}_{y},\mathcal{O}_{a+bz}=\mathcal{O}_{y}\right\},\tag{7}$$ $$\mathbf{a}=\mathcal{Q}_{y},\ \mathbf{b}=\mathbf{\eta}/\|\mathbf{\eta}\|^{2}.$$

The proof of Theorem 3.1 is deferred to Appendix B.1. By using the sampling distribution of the test statistic T(Y )
conditional on MY = My, OY = Oy, and QY = Qy in Theorem 3.1, we can define the selective p-value as

$$p_{\text{selective}}=\mathbb{P}_{\text{H}_{0}}\left(|T(\mathbf{Y})|\geq|T(\mathbf{y})|\right)\left|\begin{array}{c}\mathcal{M}_{\mathbf{Y}}=\mathcal{M}_{\mathbf{y}},\\ \mathcal{O}_{\mathbf{Y}}=\mathcal{O}_{\mathbf{y}},\\ \mathcal{O}_{\mathbf{Y}}=\mathcal{O}_{\mathbf{y}}\end{array}\right).\tag{8.1}$$
$\left(\mathfrak{H}\right)$. 
Theorem 3.2. The selective p-value defined in (8) satisfies the property in (4)*, i.e.,*

$$\mathbb{P}_{\mathrm{H}_{0}}\left(p_{\mathrm{selective}}\leq\alpha{\left|\begin{array}{l}{{\mathcal{M}}_{\mathbf{Y}}={\mathcal{M}}_{\mathbf{y}},}\\ {{\mathcal{O}}_{\mathbf{Y}}={\mathcal{O}}_{\mathbf{y}}}\end{array}\right.}\right)=\alpha,\;\forall\alpha\in(0,1).$$

Then, the selective p-value also satisfies the following property of a valid p*-value:*

$$\mathbb{P}_{\mathrm{H}_{0}}(p_{\mathrm{selective}}\leq\alpha)=\alpha,\;\forall\alpha\in(0,1).$$

The proof of Theorem 3.2 is deferred to Appendix B.2. This theorem guarantees that the selective p-value is uniformly distributed under the null hypothesis H0, and thus can be used to conduct the valid statistical inference in (3). Once the truncation intervals Z is identified, the selective p-value in (8) can be easily computed using Theorem 3.1. Thus, the remaining task is reduced to identifying the truncation intervals Z.

Theorem 5.2) and is used in almost all the SI-related works that we cited.

## 4. Computations: Line Search Interpretation

From the discussion in §3, it is suffice to identify the onedimensional subset Z in (7) to conduct the inference. In this section, we propose a novel line search method to efficiently identify the Z.

## 4.1. Overview Of The Line Search

The difficulty in identifying the Z arises from the fact that the multiple FS/OD algorithms are applied in an arbitrary complex order. To surmount this difficulty, we propose an efficient search method that leverages parametricprogramming and the fact that our pipeline can be conceptualized as a directed acyclic graph (DAG) whose nodes represent the operations. In a standard analysis pipeline, M and O are computed and updated along the DAG. However, in our framework, intervals for which M and O are constant can also be computed and updated, allowing the computation of the truncation intervals Z. In the following, we first discuss how, given a certain computational procedure (combining *update rules* as discussed in later), the Z can be identified by parametric-programming. Then, we summarize the overall procedure to compute the selective p-value from the Z. Finally, we describe the update rules for each node based on the existing methods of SI for each FS and OD algorithm. Note that DAGs can topologically sortable, so that update rules can be applied in sequence. The overview of the proposed line search method is illustrated in Figure 2.

## 4.2. Parametric-Programming

To identify the truncation intervals Z, we assume that we have a procedure to compute the interval [Lz, Uz] for any z ∈ R, which satisfies

$\forall r\in[L_{z},U_{z}]$, ${\cal M}_{a+br}={\cal M}_{a+bz},{\cal O}_{a+br}={\cal O}_{a+bz}$.  
Then, the truncation intervals Z can be obtained by the union of the intervals [Lz, Uz] as

$$\begin{array}{c}\mathcal{Z}=\\ z\in\mathbb{R}|\mathcal{M}_{a+bz}=\mathcal{M}_{y},\mathcal{O}_{a+bz}=\mathcal{O}_{y}\end{array}\tag{9}$$

The procedure in (9) is commonly referred to as parametricprogramming. We discuss the details of the procedure to compute the interval [Lz, Uz] by defining the update rules for each node in the next subsection.

## 4.3. Update Rules

In this subsection, we discuss the computation procedure to obtain the interval [Lz, Uz] for any z ∈ R just mentioned in §4.2. To compute the interval [Lz, Uz], we consider extending the input of each node in a DAG and denote it

(i)
5:FS
lasso 1:MVI
2:OD
3:FS
6:U-F
mean value l1 regression
(soft-ipod)
marginal screening 4:FS
stepwise union of features
(ii) (iii)
{1, 2, 3}
{3}
{3, 4}
update update update parametrized line update update union of tru nc ate d r egi on
{1, 2, 3, 4 }
{2, 3,4}
{2, 3}
{1, 2 }
{2, 3}
{1, 2, 3}
{1 }
{1, 2 } {2, 3} {2, 3}
{2, 3, 5}
{1, 2, 3, 5}
{3, 5} {2, 3}
as a pair of (X, a, b, z,M, O*, l, u*), where X is the design matrix, a, b and z are the currently linear expression of the response vector a + bz, M and O are the currently selected features and detected outliers, and l and u are the currently interval. The input of the first node is initialized to (X, a, b*, z,* [d], ∅, −∞,∞), where d is the number of features. We details the update rules for this pair at each node of a DAG in Appendix C.

The overall procedure for computing the interval [Lz, Uz]
by applying the update rules in the order of the topological sorting of the DAG is summarized in Algorithm 1, where the operation pa receives the index of the target node and returns the indexes of its parent nodes, and pa(1) is set to 0. Algorithm 1 satisfies the specifications described in §4.2, i.e., the following theorem holds.

Theorem 4.1. Consider a pipeline P*, a design matrix* X, and vectors a and b *representing the linear expression of* the response vector as fixed. For any z ∈ R, let [Lz, Uz],
Ma+bz and Oa+bz be the output of Algorithm 1 *with* P, X,
a, b and z *as input.*
Then, for any r ∈ [Lz, Uz], the output of Algorithm 1 *does* not change by changing the input z to r:

UpdateInterval($\mathcal{P},X,\boldsymbol{a},\boldsymbol{b},r$)  $$=([L_{z},U_{z}],\mathcal{M}_{\boldsymbol{a+bz}},\mathcal{O}_{\boldsymbol{a+bz}})$$
$$p_{a+b z}\,).$$
The proof of Theorem 4.1 is deferred to Appendix B.3. Algorithm 1 Apply Update Rules in Order of Topological Sorting of DAG (Update Interval) Require: P, X, a, b and z 1: Converts the pipeline P to a topologically sorted graph
(*V, E*)
2: Initialize the input of the first node B0 as
(X, a, b*, z,* [p], ∅, −∞,∞) (see §4.3)
3: for each index of node i ∈ {1*, . . . ,* |V |} do 4: Apply the update rule of the node vito its input Bpa(i)to obtain the output Bi (see §4.3)
5: **end for**
6: Let the last four components of B|V | be Ma+bz, Oa+bz, Lz and Uz, respectively Ensure: [Lz, Uz], Ma+bz and Oa+bz

## 5. Implementations: Auto-Conditioning 6. Numerical Experiments

All of the update rules defined in §4.3 are node-specific operations and do not depend on the type of node corresponding to the input/output. Then, we can modularize the update rules and apply them sequentially as in Algorithm 1, which implementation we call *auto-conditioning*.

The auto-conditioning allows one to simply define an arbitrary pipeline and perform hypothesis testing on it without additional implementation costs. In this section, we show some examples of defining pipelines and performing hypothesis testing using the auto-conditioning. The implementation we developed can be interactively executed using the provided Jupyter Notebook (ipynb) file, which is available in the our package repository. As an example, Listing 1 shows a code example that defines two pipeline shown in Figure 2 and performs hypothesis testing, based on our package. A similarly simple UI allows for easy implementation of other pipeline structures as well as automatic pipeline construction based on the cross-validation. For more examples, please refer to the Appendix G and the our package repository.

Listing 1. Code example that defines the pipeline shown in Figure 2. We can create an instance of manager class which handles the desired pipeline simply by specifying each operation in turn. To perform hypothesis testing, we can call the inference method of the manager instance with the input dataset (X, y) and the deviation of the noise σ. import numpy as np from si4pipeline import *
def option1() -> PipelineManager:
X, y = initialize_dataset() y = mean_value_imputation(X, y) O = soft_ipod(X, y, 0.02) X, y = remove_outliers(X, y, O) M = marginal_screening(X, y, 5) X = extract_features(X, M) M1 = stepwise_feature_selection(X, y, 3) M2 = lasso(X, y, 0.08) M = union(M1, M2) return construct_pipelines(output=M)
def option2() -> PipelineManager:
X, y = initialize_dataset() y = definite_regression_imputation(X, y) M = marginal_screening(X, y, 5) X = extract_features(X, M) O = cook_distance(X, y, 3.0) X, y = remove_outliers(X, y, O) M1 = stepwise_feature_selection(X, y, 3) M2 = lasso(X, y, 0.08) M = intersection(M1, M2) return construct_pipelines(output=M)
pl = option1()
X = np.random.normal(size=(100, 10))
y = np.random.normal(size=100) M, p_list = pl.inference(X, y, sigma=1.0)
Methods for Comparison. In our experiments, we consider the three types of pipelines: op1, op2, and cv. The op1 and op2 are defined in Figure 2. The cv is a pipeline selected based on cross-validation from 16 different parameters settings each in the op1 and op2 pipelines (i.e., from 32 pipelines in total). For each three types of pipelines, we compare the proposed method (proposed) in terms of type I error rate and power with the following three methods:
- w/o-pp: An ablation study that excludes the parametric programming technique described in §4.2. This is implemented by replacing the Z in (9) with a interval [Lz, Uz] that contains the observed test statistic T(y).

- naive: This method uses a classical z-test without conditioning, i.e., we compute the naive p-value as pnaive = PH0
(|T(Y )*| ≥ |*T(y)|).

- bonferroni: This is a method to control the type I error rate by using the Bonferroni correction, a simple yet widely used method for multiple testing correction. The number of all possible pair of selected features and detected outliers is 2 d· 2 n, then we compute the Bonferroni corrected p-value as pbonferroni =
min(1, 2 d· 2 n · pnaive).

Experimental Setup. In all experiments, we set the significance level α = 0.05. For the experiments to see the type I error rate, we change the number of samples n ∈ {100, 200, 300, 400} and set the number of features d to 20. See Appendix D.1 for results when the number of features d is changed, and for the high-dimensional regression setting (i.e., where d ≫ n). For each configuration, we generated 10,000 null datasets (X, y), where Xij ∼ N (0, 1) for all (*i, j*) ∈ [n] × [d] and y ∼ N (0, In). Missing values were introduced by randomly setting each yito NaN
with a probability of 0.03. To investigate the power, we set n = 200 and d = 20 and generated dataset (X, y), where Xij ∼ N (0, 1) for all (i, j) ∈ [n] × [d] and y = Xβ + ϵ. The error term ϵ followed a normal distribution N (0, In),
and the coefficient vector β ∈ R
d was constructed such that its first three elements were set to ∆ and the remaining elements were set to 0. Missing values were introduced by randomly setting each yito NaN with a probability of 0.03. We change the true coefficients ∆ ∈ {0.2, 0.4, 0.6, 0.8}. For power evaluation, hypothesis testing was conducted only when the pipeline selected at least one truly relevant feature (i.e., one of the first three features), resulting in a total of 10,000 tests. In addition, see Appendix D.2 for results when the missing value probability increased, Appendix D.3 for the computational time of the proposed method for larger datasets and more complex pipelines, and Appendix D.4 for the computer resources used in the experiments.

Table 1. Power on eight real-world datasets when changing the sample size n. Each cell indicates the power of the proposed method (proposed) and the ablation study (w/o-pp), separated by a slash, with the higher value in bold. The proposed method demonstrates significantly higher power than the ablation study method across all datasets and sample sizes. Furthermore, the power of the proposed method increases with increasing sample size n.

n Data1 Data2 Data3 Data4 Data5 Data6 Data7 Data8

100 .57/.07 .48/.06 .57/.07 .51/.07 .68/.10 .55/.04 .30/.05 .25/.06 150 .79/.09 .71/.08 .66/.12 .57/.10 .74/.12 .72/.06 .37/.06 .37/.06 200 .91/.11 .80/.08 .78/.15 .66/.12 .76/.13 .82/.08 .49/.07 .40/.06

Results. The results of type I error rate are shown in left side of Figure 3. The proposed, w/o-pp, and bonferroni successfully controlled the type I error rate under the significance level across all settings and pipeline types, whereas the naive could not. Because the naive failed to control the type I error rate, we no longer consider its power. The results of power are shown in right side of Figure 3. Among the methods that controlled the type I error rate, the proposed has the highest power, followed by the w/o-pp, across all settings and pipeline types. The reduced power of the w/o-pp compared to the proposed can be attributed to its inherent conditioning on more information than those defined in (5). This problem is known as overconditioning in the context of SI. The notably low power of the bonferroni is consistent with the understanding that such classical methods are often too conservative for the large-scale problems considered in this study. Real Data Experiments. We compared the proposed and w/o-pp in terms of power, for the cv pipeline on eight real-world datasets from the UCI Machine Learning Repository (all licensed under the CC BY 4.0; see Appendix D.5 for more details). These experiments were conducted under the implicit assumption that features selected by the feature selection pipeline are truly relevant. This assumption is reasonable because both the proposed and w/o-pp evaluated in this study have been shown to control the type I error rate. From each original dataset, we randomly generated 1,000 sub-sampled datasets with sample sizes of n ∈ {100, 150, 200}. We then applied both the proposed and w/o-pp to assess their powers. The results, presented in Table 1, demonstrate that the proposed method has much higher power than the w/o-pp across all datasets for all sample sizes. Furthermore, the power of the proposed increases with increasing sample size n.

## 7. Conclusions

In this study, we introduced a novel framework for testing the statistical significance of feature selection pipelines in linear models, comprising multiple MVI, OD, and FS algorithms based on the concept of SI. Our long-term goal extends beyond this current study to ensure the reproducibility of data-driven decision-making by accounting for the entire pipeline from raw data to final results, with this study on a class of feature selection pipelines serving as a proof of concept. To achieve this future goal, there are still limitations on the applicable data analysis components, presenting several challenges in extending the proposed framework to more complex data analysis pipelines. Additionally, it is interesting to consider extending this framework to scenarios where data analysis pipelines are automatically constructed using state-of-the-art AutoML approaches.

1.0 0.5 1.0 0.5 1.0 0.5 1.0 Typ e I
 Er ror Ra teproposed w/o-pp bonferroni naive significance level proposed w/o-pp bonferroni Pow er 0.5 100 200 300 400 number of samples 0.0 0.2 0.4 0.6 0.8 true coefficient 0.0
(a) Type I Error Rate and Power of op1 pipeline 1.0 Typ e I E
rror Ra teproposed w/o-pp bonferroni naive significance level proposed w/o-pp bonferroni Pow er 0.5 100 200 300 400 number of samples 0.0 0.2 0.4 0.6 0.8 true coefficient 0.0
(b) Type I Error Rate and Power of op2 pipeline 1.0 Typ e I E
rror Ra teproposed w/o-pp bonferroni naive significance level proposed w/o-pp bonferroni Pow er 0.5 100 200 300 400 number of samples 0.0 0.2 0.4 0.6 0.8 true coefficient 0.0
(c) Type I Error Rate and Power of cv pipeline

## Acknowledgements Impact Statement References

Duy, V. N. L., Toda, H., Sugiyama, R., and Takeuchi, I.

Computing valid p-value for optimal changepoint by selective inference using dynamic programming. In Advances in Neural Information Processing Systems, 2020.

This work was partially supported by JST CREST (JPMJCR21D3, JPMJCR22N2), JST Moonshot R&D (JPMJMS2033-05), RIKEN Center for Advanced Intelligence Project, and RIKEN Junior Research Associate Program.

Duy, V. N. L., Iwazaki, S., and Takeuchi, I. Quantifying statistical significance of neural network-based image segmentation by selective inference. *Advances in Neural* Information Processing Systems, 35:31627–31639, 2022.

Fithian, W., Taylor, J., Tibshirani, R., and Tibshirani, R.

Selective sequential model selection. arXiv preprint arXiv:1512.02565, 2015.

This work, which focuses on statistical tests for feature selection pipelines, aims to enhance the reliability of AI and has the potential to broadly influence the machine learning community. On the other hand, it does not present significant ethical concerns or foreseeable societal consequences because this work is theoretical and, as of now, has no direct applications that might impact society or ethical considerations.

Gao, L. L., Bien, J., and Witten, D. Selective inference for hierarchical clustering. Journal of the American Statistical Association, pp. 1–11, 2022.

Google. Vertex ai, 2021. URL https://cloud.

google.com/vertex-ai/.

Hapke, H. and Nelson, C. Building machine learning pipelines. O'Reilly Media, 2020.

Gas Turbine CO and NOx Emission Data Set.

UCI Machine Learning Repository, 2019. DOI: https://doi.org/10.24432/C5WC95.

Hyun, S., G'sell, M., and Tibshirani, R. J. Exact postselection inference for the generalized lasso path. Electronic Journal of Statistics, 12(1):1053–1097, 2018.

Amazon. Amazon sagemaker autopilot, 2019. URL https://docs.aws. amazon.com/sagemaker/latest/dg/ autopilot-automate-model-development. html.

Jewell, S., Fearnhead, P., and Witten, D. Testing for a change in mean after changepoint detection. *Journal of the Royal* Statistical Society Series B: Statistical Methodology, 84
(4):1082–1104, 2022.

Brooks, T., Pope, D., and Marcolini, M. Airfoil Self-
Noise. UCI Machine Learning Repository, 1989. DOI: https://doi.org/10.24432/C5VW2C.

Le Duy, V. N., Lin, H.-T., and Takeuchi, I. Cad-da: Controllable anomaly detection after domain adaptation by statistical inference. In International Conference on Artificial Intelligence and Statistics, pp. 1828–1836. PMLR, 2024.

Chen, S. and Bien, J. Valid inference corrected for outlier removal. Journal of Computational and Graphical Statistics, 29(2):323–334, 2020.

Lee, J. D. and Taylor, J. E. Exact post model selection inference for marginal screening. Advances in neural information processing systems, 27, 2014.

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., and Reis, J.

Wine Quality. UCI Machine Learning Repository, 2009. DOI: https://doi.org/10.24432/C56S3T.

Lee, J. D., Sun, Y., and Taylor, J. E. Evaluating the statistical significance of biclusters. Advances in neural information processing systems, 28, 2015.

Das, D., Duy, V. N. L., Hanada, H., Tsuda, K., and Takeuchi, I. Fast and more powerful selective inference for sparse high-order interaction model. Proceedings of the AAAI Conference on Artificial Intelligence, 36(9):9999–10007, Jun. 2022. doi: 10.1609/ aaai.v36i9.21238. URL https://ojs.aaai.org/ index.php/AAAI/article/view/21238.

Lee, J. D., Sun, D. L., Sun, Y., and Taylor, J. E. Exact postselection inference, with application to the lasso. The Annals of Statistics, 44(3):907–927, 2016.

Microsoft. Azure automated machine learning, 2018. URL https://azure.microsoft.

com/en-us/products/machine-learning/
automatedml/\#overview.

Drori, I., Krishnamurthy, Y., Rampin, R., Lourenco, R.

d. P., Ono, J. P., Cho, K., Silva, C., and Freire, J. Alphad3m: Machine learning pipeline synthesis. arXiv preprint arXiv:2111.02508, 2021.

Rugamer, D. and Greven, S. Inference for l 2-boosting. ¨
Statistics and computing, 30(2):279–289, 2020.

Rugamer, D., Baumann, P. F., and Greven, S. Selective ¨
inference for additive and linear mixed models. Computational Statistics & Data Analysis, 167:107350, 2022.

Sugimura, P. and Hartl, F. Building a reproducible machine learning pipeline. *arXiv preprint arXiv:1810.04570*, 2018.

Suzumura, S., Nakagawa, K., Umezu, Y., Tsuda, K., and Takeuchi, I. Selective inference for sparse high-order interaction models. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 3338–3347. JMLR. org, 2017.

Tanizaki, K., Hashimoto, N., Inatsu, Y., Hontani, H., and Takeuchi, I. Computing valid p-values for image segmentation by selective inference. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9553–9562, 2020.

Taylor, J. and Tibshirani, R. J. Statistical learning and selective inference. Proceedings of the National Academy of Sciences, 112(25):7629–7634, 2015.

Tibshirani, R. J., Taylor, J., Lockhart, R., and Tibshirani, R.

Exact post-selection inference for sequential regression procedures. Journal of the American Statistical Association, 111(514):600–620, 2016.

Tsanas, A. and Xifara, A. Energy Efficiency.

UCI Machine Learning Repository, 2012. DOI: https://doi.org/10.24432/C51307.

Tsukurimichi, T., Inatsu, Y., Duy, V. N. L., and Takeuchi, I. Conditional selective inference for robust regression and outlier detection using piecewise-linear homotopy continuation. Annals of the Institute of Statistical Mathematics, 74(6):1197–1228, 2022. doi: 10.1007/
s10463-022-00846-2. URL https://doi.org/10. 1007/s10463-022-00846-2.

Yang, F., Barber, R. F., Jain, P., and Lafferty, J. Selective inference for group-sparse linear models. In *Advances in* Neural Information Processing Systems, pp. 2469–2477, 2016.

Yeh, I.-C. Concrete Compressive Strength. UCI
Machine Learning Repository, 1998. DOI:
https://doi.org/10.24432/C5PK67.

Yeh, I.-C. Real Estate Valuation. UCI Machine Learning Repository, 2018. DOI: https://doi.org/10.24432/C5J30W.

## A. Pipeline Components

A.1. Missing-Value Imputation (MVI) Algorithm Components A MVI algorithm component is represented as

$$f_{\mathrm{MVI}}:\{X,\mathbf{y},{\mathcal{O}},{\mathcal{M}}\}\mapsto\{X,\mathbf{y}^{(+)},{\mathcal{O}},{\mathcal{M}}\},$$

where y ∈ R
n
′is the response vector which excludes missing values and y
(+) ∈ R
n is the vector with imputed missing values. MVI algorithms in this paper are *linear* algorithm in the sense that, using a matrix DX ∈ R
n×n
′that depends on X
are written as y
(+) = DXy.

Mean Value Imputation. This method replaces missing values with the mean value of observed data and allows for quick and easy imputation of missing values. An example of DX for y = (y1, y3, y4)
⊤ (i.e., y2 is missing value and n = 4) is:

$$D_{X}=\left(\begin{array}{l l l}{{1}}&{{0}}&{{0}}\\ {{1/3}}&{{1/3}}&{{1/3}}\\ {{0}}&{{1}}&{{0}}\\ {{0}}&{{0}}&{{1}}\end{array}\right)$$

Nearest-Neighbor Imputation. This method replaces missing values with the most similar instance in the dataset. In this method, similarity between instances is measured by some distance between their feature vectors. As distance measures ℓ(·, ·), for example, Euclidean, Manhattan, or Chebyshev distance can be used. An example of DX for y = (y1, y3, y4)
⊤
(i.e., y2 is missing value and n = 4) is:

$$D_{X}=\begin{pmatrix}1&0&0\\ &\mathbf{e}_{j}^{\top}\\ 0&1&0\\ 0&0&1\end{pmatrix},\ j=\operatorname*{arg\,min}_{i\in\{1,3,4\}}\ell(\mathbf{x}_{2},\mathbf{x}_{i}),$$

where ej is the vector constructed by removing the indices of the missing values (i.e., {2}) from the j-th unit vector in R
4.

Regression Imputation. This method replaces missing values with estimated values based on a regression model. We use the observed instances to estimate the regression coefficients, and then use the estimated coefficients to predict the missing values from its feature vector. We denote the indices of the missing values as NaN, and the indices of the observed values as −NaN. The regression coefficients can be estimated as βˆ = (XT−NaN,:X−NaN,:)
−1XT−NaN,:y and then each imputed missing value y
(+)
i, i ∈ NaN can be expressed as y
(+)
i = x
⊤
i βˆ. An example of DX for y = (y1, y3, y4)
⊤ (i.e., y2 is missing value and n = 4) is:

$$D_{X}={\begin{pmatrix}1&0&0\\ X_{\{2\},:}(X_{\{1,3,4\},:}^{\top}X_{\{1,3,4\},:})^{-1}X_{\{1,3,4\},:}^{\top}\\ 0&1&0\\ 0&0&1\end{pmatrix}}$$

A.2. Outlier Detection (OD) Algorithm Components A OD algorithm component is represented as

$$f_{\mathrm{{OD}}}:\{X,\mathbf{y}^{(+)},{\mathcal{O}},{\mathcal{M}}\}\mapsto\{X,\mathbf{y}^{(+)},{\mathcal{O}}^{\prime},{\mathcal{M}}\},$$

where O′is the updated set of outliers.

Cook's Distance-based OD. This method identifies instances as outliers when *Cook's distance*, a measure of the influence of a particular instance on the entire regression model, exceeds a predefined threshold value. *Cook's distance* of the i-th instance is defined as

$$D_{i}={\frac{\sum_{j=1}^{n}\left({\hat{y}}_{j}-{\hat{y}}_{j(i)}\right)^{2}}{d{\mathrm{\MSE}}}},$$

where yˆj and yˆj(i) are the predicted value of j-th instance from the regression model with and without i-th instance, respectively, and MSE is the mean squared error of the full model. This Di represents the standardized value of the change in predictions for all other instances due to the removal of i-th instance, and the larger Diis, the more it affects the model.

By utilizing the leverage value, it can also be represented as

$$D_{i}=\frac{\hat{\varepsilon}_{i}^{2}}{d\,\mathrm{MSE}}\frac{h_{i i}}{(1-h_{i i})^{2}},$$

where εˆiis the i-th residual, hii is the i-th leverage value (i.e., the diagonal component of the matrix X(X⊤X)
−1X⊤). We identify the i-th instance as an outlier if Di > λ where λ is a predefined threshold value.

DFFITS OD This method has the same concept as Cook's distance-based OD but uses *DFFITS* instead of Cook's distance as the measure of influence. *DFFITS* of the i-th instance is defined as

$$\mathrm{DFFITS}_{i}={\frac{{\hat{y}}_{i}-{\hat{y}}_{i(i)}}{\sqrt{\mathrm{MSE}_{(i)}h_{i i}}}},$$

where yˆi and yˆi(i) are the predicted value of the i-th instance from the regression model with and without the i-th instance, respectively, MSE(i)is the mean squared error of the regression model without the i-th instance, and hii is the i-th leverage value. Thus, *DFFITS* is a value that standardizes the difference between the predicted value when excluding and including a specific instance, and the larger DFFITSiis, the more it affects the model. By utilizing the external Studentized residual ri,ext, it can also be represented as

$$\mathrm{DFFITS}_{i}={\sqrt{\frac{h_{i i}}{1-h_{i i}}}}r_{i,\mathrm{ext}}.$$

We identify the i-th instance as an outlier if DFFITS2 i *> λd/*(n − d) where λ is a predefined threshold value and usually set to 4.

L1 **Regression based OD** This method identifies instances as outliers by using L1 regularization for the mean-shift model. In this method, we assume that the unknown true value function µ(X) follows the following mean-shift model:

$$\mu(X)=X\beta+u,$$

where u ∈ R
n is an outlier term and ui ̸= 0 if the i-th instance is an outlier, otherwise ui = 0. We estimate (βˆλ,uˆλ) by solving the following optimization problem:

$$({\hat{\boldsymbol{\beta}}}_{\lambda},{\hat{\boldsymbol{u}}}_{\lambda})=\operatorname*{arg\,min}_{{\boldsymbol{\beta}}\in\mathbb{R}^{d},{\boldsymbol{u}}\in\mathbb{R}^{n}}{\frac{1}{2n}}\|{\boldsymbol{y}}^{(+)}-X{\boldsymbol{\beta}}-{\boldsymbol{u}}\|_{2}^{2}+\lambda\|{\boldsymbol{u}}\|_{1},$$

where λ is a predefined regularization parameter. We identify the i-th instance as an outlier if uˆλ,i ̸= 0.

A.3. Feature Selection (FS) Algorithm Components A FS algorithm component is represented as

$$f_{\mathrm{FS}}:\{X,\mathbf{y}^{(+)},{\mathcal{O}},{\mathcal{M}}\}\mapsto\{X,\mathbf{y}^{(+)},{\mathcal{O}},{\mathcal{M}}^{\prime}\},$$

where, M′is the updated set of features.

Marginal Screening This method selects the k features that are most correlated with the response variable, where k is a predefined number. The correlation is computed as the absolute value of the inner product |x
⊤
j y
(+)| between the normalized feature vector xj and the response vector y
(+).

Stepwise Feature Selection This method selects features by iterating through the steps of adding or deleting the features that best improve the goodness of fit of the regression model. In this paper, we deal with forward stepwise feature selection, which only adds features. The residual sum of squares (RSS) of the least squares regression model constructed using the features selected up to the previous step is used as the goodness of fit of the model. First, a null model (a model consisting of an intercept term) is used as an initial state, and in each step, RSS is calculated from the least squares regression model constructed with the features selected in the previous step and the residual of y
(+). After that, select the feature that minimize RSS and update the model. The algorithm terminates if the RSS is not improved by adding any feature, or if the number of selected features reaches a predefined upper limit. Lasso This method selects features by using a linear regression model with L1 regularization. We estimate the regression coefficient βˆ by solving the following optimization problem:

$${\hat{\boldsymbol{\beta}}}={\underset{\boldsymbol{\beta}\in\mathbb{R}^{d}}{\operatorname{arg\,min}}}\,{\frac{1}{2n}}\|{\boldsymbol{y}}^{(+)}-X{\boldsymbol{\beta}}\|_{2}^{2}+\lambda\|{\boldsymbol{\beta}}\|_{1},$$

where λ is a predefined regularization parameter. We select the features for which βˆi ̸= 0.

## B. Proofs

B.1. Proof of Theorem 3.1 According to the conditioning on QY = Qy, we have

$${\mathcal{Q}}_{Y}={\mathcal{Q}}_{y}\Leftrightarrow\left(I_{n^{\prime}}-{\frac{\eta^{\top}\eta}{\|\eta\|^{2}}}\right)Y={\mathcal{Q}}_{y}\Leftrightarrow Y=a+b z,$$

where z = T(Y ) ∈ R. Then, we have

{Y ∈ R n ′| MY = My, OY = Oy, QY = Qy} ={Y ∈ R n ′| MY = My, OY = Oy,Y = a + bz, z ∈ R} ={a + bz ∈ R n ′| Ma+bz = My, Oa+bz = Oy, z ∈ R} ={a + bz ∈ R n ′| z ∈ Z}.
Therefore, we obtain

$$T(\mathbf{Y})\mid\{{\mathcal{M}}_{\mathbf{Y}}={\mathcal{M}}_{\mathbf{y}},{\mathcal{O}}_{\mathbf{Y}}={\mathcal{O}}_{\mathbf{y}},{\mathcal{Q}}_{\mathbf{Y}}={\mathcal{Q}}_{\mathbf{y}}\}\sim\operatorname{TN}(\mathbf{\eta}^{\top}\mathbf{\mu},\sigma^{2}\|\mathbf{\eta}\|^{2},{\mathcal{Z}}).$$

B.2. Proof of Theorem 3.2 By probability integral transformation, under the null hypothesis, we have

$p_{\rm selective}\mid\{{\cal M}_{Y}={\cal M}_{y},{\cal O}_{Y}={\cal O}_{y},{\cal Q}_{Y}={\cal Q}_{y}\}\sim{\rm Unif}(0,1)$,
which leads to  $$\mathbb{P}_{\mathrm{H}_{0}}\left(p_{\mathrm{selective}}\leq\alpha\mid\mathcal{M}_{\boldsymbol{Y}}=\mathcal{M}_{\boldsymbol{y}},\mathcal{O}_{\boldsymbol{Y}}=\mathcal{O}_{\boldsymbol{y}},\,\mathcal{Q}_{\boldsymbol{Y}}=\mathcal{Q}_{\boldsymbol{y}}\right)=\alpha,\,\forall\alpha\in(0,1).$$  For any $\alpha\in(0,1)$, by marginalizing over all the values of the nuisance parameters, we obtain 
PH0(pselective ≤ α | MY = My, OY = Oy) = Z Rn′ PH0(pselective ≤ α | MY = My, OY = Oy, QY = Qy) PH0 (QY = Qy | MY = My, OY = Oy) dQy =α Z Rn′ PH0 (QY = Qy | MY = My, OY = Oy) dQy = α.
Therefore, we also obtain

$$\mathbb{P}_{\mathrm{H_{0}}}(p_{\mathrm{selective}}\leq\alpha)$$ $$=\sum_{\mathcal{M}_{\mathbf{y}}\in\mathbb{Z}^{|\mathcal{Y}}}\sum_{\mathcal{O}_{\mathbf{y}}\in\mathbb{Z}^{|n|}}\mathbb{P}_{\mathrm{H_{0}}}(\mathcal{M}_{\mathbf{y}},\mathcal{O}_{\mathbf{y}})\mathbb{P}_{\mathrm{H_{0}}}\left(p_{\mathrm{selective}}\leq\alpha\mid\mathcal{M}_{\mathbf{Y}}=\mathcal{M}_{\mathbf{y}},\mathcal{O}_{\mathbf{Y}}=\mathcal{O}_{\mathbf{y}}\right)$$ $$=\alpha\sum_{\mathcal{M}_{\mathbf{y}}\in\mathbb{Z}^{|n|}}\sum_{\mathcal{O}_{\mathbf{y}}\in\mathbb{Z}^{|n|}}\mathbb{P}_{\mathrm{H_{0}}}(\mathcal{M}_{\mathbf{y}},\mathcal{O}_{\mathbf{y}})=\alpha.$$
$$1,\ldots,|V|\}$$

## B.3. Proof Of Theorem 4.1

It is sufficient to consider only z as input to Algorithm 1. In addition, as a notation, we define Gi as the mapping that returns the last four components of Bi for i ∈ {0, 1*, . . . ,* |V |}, i.e.,

$${\mathcal{G}}_{i}\colon\mathbb{R}\ni$$

Gi: R ∋ z 7→ (Mia+bz, O
ia+bz, liz, uiz) ∈ 2
[p] × 2
[n] × R
2, i ∈ {0, 1*, . . . ,* |V |}
According to the above notation, all we have to show is that G|V |(z) = G|V |(r) for any z ∈ R and any r ∈ [l |V | z , u |V | z ]. We show this by mathematical induction.

In the case i = 0, it is obvious from the definition of B0 in Algorithm 1 that G0(z) = G0(r) = ([p], ∅, −∞,∞) for any z ∈ R and any r ∈ [l 0 z, u0z] = [−∞, ∞].

Next, we assume that for any fixed i ∈ {0, . . . , |V | − 1}, Gj (z) = Gj (r) for any j ∈ {0*, . . . , i*}, any z ∈ R and any r ∈ [l jz, ujz]. Under this assumption, noting that pa(i + 1) ⊂ {0*, . . . , i*} from a property of topological sort, it is obvious that Gi+1(z) = Gi+1(r) for any z ∈ R and any r ∈ [l i+1 z, ui+1 z] from the update rule of vi+1 described in §4.3.

## C. Details Of The Update Rules

Update Rulu for the Node of MVI. The node of MVI imputes the missing values in the response vector a + bz. All MVI algorithms considered in this study are expressed as linear transformations determined on the basis of X. Thus, let DX be the linear transformation matrix, the update rule should be as follows:

$$(X,a,b,z,\Lambda)$$

(X, a, b, z,M, O, l, u) 7→ (X, DXa, DXb, z,M, O*, l, u*).

Update Rule for the Node of FS. The node of FS selects the features M′(z) from the dataset (X−O,M, a−O + b−Oz),
which means that feature selection is performed on the dataset extracted from (X, a + bz) based on M and O. For all FS
algorithms considered in this study, the computation procedure to obtain the interval [lz, uz] ∋ z, which satisfies

$$l,u)\mapsto$$
$${\cal M}^{\prime}(r)={\cal M}^{\prime}(z),$$
$$\forall r\in$$
$\rho,\iota,u)\mapsto(\lambda,u,u)$
∀r ∈ [lz, uz], M′(r) = M′(z),
have been proposed in previous studies (Lee & Taylor, 2014; Tibshirani et al., 2016; Lee et al., 2016). Utilizing this, the update rule should be as follows:

$\ensuremath{\ensuremath{\nobreak\lower3pt\hbox{$\circ$}}\kern-3.04em\lower3pt\hbox{$\circ$}},2,\ensuremath{\ensuremath{\mathcal{M}}}$
(X, a, b, z,M, O, l, u) 7→ (X, a, b, z,*M ∩ M*′(z), O, max(l, lz), min(u, uz)).

Update Rule for the Node of OD. The node of OD detects the outliers O′(z) from the dataset (X−O,M, aM + bMz),
which means that outlier detection is performed on the dataset extracted from (X, a + bz) based on M and O. For all OD
algorithms considered in this study, the computation procedure to obtain the interval [lz, uz] ∋ z, which satisfies
∀r ∈ [lz, uz], O
′(r) = O
′(z),
have been proposed in previous studies (Chen & Bien, 2020). Utilizing this, the update rule should be as follows:
(X, a, b, z,M, O, l, u) 7→ (X, a, b, z,M, *O ∩ O*′(z), max(*l, l*z), min(*u, u*z)).

Update Rule for the Node of Union/Intersection of Features/Outliers. The node computes the union or intersection of selected features or detected outliers. With E being the number of input edges, for each selected feature and detected outlier, the update rules should be as follows:

$$\begin{array}{l}{{\{(X,\mathbf{a},\mathbf{b},z,\mathcal{M}_{e},\mathcal{O},l_{e},u_{e})\}_{e\in[E]}\mapsto(X,\mathbf{a},\mathbf{b},z,\sum_{e\in[E]}\mathcal{M}_{e},\mathcal{O},\operatorname*{max}_{e\in[E]}l_{e},\operatorname*{min}_{e\in[E]}u_{e}),}}\\ {{\{(X,\mathbf{a},\mathbf{b},z,\mathcal{M},\mathcal{O}_{e},l_{e},u_{e})\}_{e\in[E]}\mapsto(X,\mathbf{a},\mathbf{b},z,\mathcal{M},\sum_{e\in[E]}\mathcal{O}_{e},\operatorname*{max}_{e\in[E]}l_{e},\operatorname*{min}_{e\in[E]}u_{e}),}}\end{array}$$

where P represents the union or intersection depending on the type of the node.

## D. Details Of The Experiments D.1. Additional Type I Error Rate Results

We also conducted experiments to investigate the type I error rate when the number of features d is changed, and for the high-dimensional regression setting (i.e., where d ≫ n). For the former case, we changed the number of features d ∈ {10, 20, 30, 40} and set the number of samples n to 200. For the latter case, we set the number of samples n to 100 and changed the number of features d ∈ {400, 800, 1200, 1600}. It should be noted that, within this experimental setting, the op2 and op3 pipelines are used, to handle the high-dimensional regression setting. The op3 pipeline is defined by reversing the order of the L1 regression-based outlier detection (OD) node and the marginal screening feature selection (FS)
node in the op1 pipeline. In both cases, we generated the null datasets in the same way as in the main experiments (§6), and the results are shown in Figure 4 and Figure 5, respectively.

1.0 1.0 Ty pe I E
rr or Rat e proposed w/o-pp bonferroni naive significance level 0.5 0.5 10 20 30 40 number of features 0.0
(a) op1 Ty pe I E
rr or Rat e proposed w/o-pp bonferroni naive significance level 10 20 30 40 number of features 0.0
(b) op2 1.0 0.5 Ty pe I E
rr or Rat e proposed w/o-pp bonferroni naive significance level 10 20 30 40 number of features 0.0
(c) cv 1.0 1.0 1.0 Ty pe I 
Erro r Ra te proposed w/o-pp bonferroni naive significance level Ty pe I 
Erro r Ra te proposed w/o-pp bonferroni naive significance level Ty pe I 
Erro r Ra te proposed w/o-pp bonferroni naive significance level 0.5 0.5 0.5 400 800 1200 1600 number of features 0.0 400 800 1200 1600 number of features 0.0 400 800 1200 1600 number of features 0.0
(a) op2
(b) op3
(c) cv

## D.2. Effect Of Missing Value Probability

We also conducted experiments to investigate the effect of the missing value probability on the type I error rate and power of the proposed method. In the experiments, we change the missing value probability ρ ∈ {0.03, 0.12, 0.21, 0.30}. For the type I error rate, we set the number of samples n = 200 and the number of features d = 20. For the power, we set the number of samples n = 200, the number of features d = 20, and the true coefficients ∆ = 0.4. In both cases, we generated the datasets in the same way as in the main experiments (§6), and the results are shown in Figure 6.

## D.3. Computational Time Of The Proposed Method

We also conducted experiments to investigate the computational time of our proposed method by applying it to three types of pipeline structures (Default, Parallel, and Serial) with large-scale datasets. Default pipeline correspond to the op1 pipeline in §6. Parallel and Serial pipelines are defined as in Figure 8, which clarifies the difference from the Default with components colored in pink. In the experiments, we change the number of samples n ∈ {400, 800, 1200, 1600} with the number of features d = 80 and the number of features d ∈ {40, 80, 120, 160} with the number of samples n = 800 to generate the null datasets in the same way as in the main experiments (§6). Note that in this experiment, we recorded the computational time for a single hypothesis testing (i.e., calculate one p-value) on a single CPU core. The results are shown in Figure 7.

1.0 Type I
 Error Rate proposed w/o-pp bonferroni naive significance level 0.5 0.03 0.12 0.21 0.30 missing value probability 0.0
(a) Type I Error Rate 1.0 proposed w/o-pp bonferroni Powe r 0.5 0.03 0.12 0.21 0.30 missing value probability 0.0
(b) Power Com putati onal Ti me (s
)

op1 op1_parallel op1_serial 400 800 1200 1600 number of samples 0 100 200 300 400
(a) Number of Samples Com putati onal Ti me (s
)

op1 op1_parallel op1_serial 40 80 120 160 number of features 0 25 50 75 100 125
(b) Number of Features

## D.4. Computer Resources

All numerical experiments were conducted on a computer with a 96-core 3.60GHz CPU and 512GB of memory.

## D.5. Details Of The Real Datasets

We used the following eight real datasets from the UCI Machine Learning Repository. All datasets are licensed under the CC BY 4.0 license.

- Airfoil Self-Noise (Brooks et al., 1989) for Data1 - Concrete Compressive Strength (Yeh, 1998) for Data2 - Energy Efficiency (Tsanas & Xifara, 2012) for Data3 (heating load) and Data4 (cooling load) - Gas Turbine CO and NOx Emission Data Set (gas, 2019) for Data5 - Real Estate Valuation (Yeh, 2018) for Data6 - Wine Quality (Cortez et al., 2009) for Data7 (red wine) and Data8 (white wine)

Parallel 7:FS
lasso 4:FS
lasso mean value 2:OD
l1 regression
(soft-ipod)
1:MVI
3:FS
6:U-F
9:U-F
marginal screening 5:FS
stepwise union of features union of features 8:FS
stepwise Serial 6:FS
lasso mean value 2:OD
l1 regression
(soft-ipod)
1:MVI
3:FS
stepwise 5:FS
lasso 4:FS
8:U-F
marginal screening union of features 7:FS
stepwise

## E. Robustness Of Type I Error Rate Control

In this experiment, we confirmed the robustness of the proposed method for cv pipeline in terms of type I error rate control by applying our method to the two cases: the case where the variance is estimated from the same data and the case where the noise is non-Gaussian.

## E.1. Estimated Variance

In the case where the variance is estimated from the same data, we considered the same two options as in type I error rate experiments in §6 and Appendix D.1; number of samples and number of features. For each setting, we generated 10,000 null datasets (X, y), where Xij ∼ N (0, 1), ∀(*i, j*) ∈ [n] × [d] and y ∼ N (0, In) and estimated the variance σˆ
2as

$${\hat{\sigma}}^{2}={\frac{1}{n-d}}\|\mathbf{y}-X(X^{\top}X)^{-1}X^{\top}\mathbf{y}\|_{2}^{2}.$$

We considered the three significance levels α = 0.05, 0.01, 0.10. The results are shown in Figure 9 and our proposed method can properly control the type I error rate.

100 200 300 400 number of samples 0.00 0.01 0.05 0.10 0.15 0.20 Type I Err or R
ate alpha=0.05 alpha=0.01 alpha=0.10 significance levels
(a) Number of Samples 10 20 30 40 number of features 0.00 0.01 0.05 0.10 0.15 0.20 Type I Err or R
ate alpha=0.05 alpha=0.01 alpha=0.10 significance levels
(b) Number of Features

## E.2. Non-Gaussian Noise

In the case where the noise is non-Gaussian, we set n = 200 and d = 20. As non-Gaussian noise, we considered the following five distribution families:
- skewnorm: Skew normal distribution family. - exponnorm: Exponentially modified normal distribution family. - gennormsteep: Generalized normal distribution family (limit the shape parameter β to be steeper than the normal distribution, i.e., β < 2).

- gennormflat: Generalized normal distribution family (limit the shape parameter β to be flatter than the normal distribution, i.e., β > 2).

- t: Student's t distribution family.

Note that all of these distribution families include the Gaussian distribution and are standardized in the experiment. To conduct the experiment, we first obtained a distribution such that the 1-Wasserstein distance from N (0, 1) is l in each distribution family, for l ∈ {0.01, 0.02, 0.03, 0.04}. We then generated 10,000 null datasets (X, y), where Xij ∼ N (0, 1), ∀(*i, j*) ∈ [n] × [d] and yi, ∀i ∈ [n] follows the obtained distribution. We considered the two significance levels α = 0.05, 0.01. The results are shown in Figure 10 and our proposed method can properly control the type I error rate.

0.01 0.02 0.03 0.04 Wasserstein Distance 0.00 0.05 0.10 0.15 0.20 Ty pe I E
rro r Ra te skewnorm exponnorm gennormsteep gennormflat t significance level
(a) Significance Level 0.05 0.01 0.02 0.03 0.04 Wasserstein Distance 0.00 0.01 0.02 0.03 0.04 Ty pe I E
rro r Ra te skewnorm exponnorm gennormsteep gennormflat t significance level
(b) Significance Level 0.01

## F. Automatic Pipeline Construction Based On Cross-Validation

In this section, we discuss cross-validation for pipelines. We consider selecting the pipeline P from a given set of candidates {P1*, . . . ,*PS} where S is the number of candidates. Note that this formulation is general enough to handle many crossvalidation targets in a unified form. For examples, (i) the case where only changing the regularization strength of lasso node, (ii) the case where changing the method of missing value imputation, and (iii) the case where changing the all structure of the pipeline (i.e., type and order of nodes). Thereafter, we discuss how statistical inference changes when cross-validation is performed and how cross-validation can be formulated. Then, based on above discussion, Algorithm 1 is extended to be applicable to the case of cross-validation.

## F.1. Statistical Inference After Cross-Validation

Changes in Statistical Inference As a formulation of statistical inference after cross-validation, the discussion in §2 and §3 can be done in exactly the same way, except with two changes: (i) the procedure for computing M and O (in §2 and §3, M and O are simply the outputs of a given mapping P representing a target pipeline), and (ii) the dependence on the response vector y of which method to use for missing value imputation. This implies that the procedure for computing the truncation intervals Z in §4 can not be directly applied to the case of cross-validation.

Formulation of Cross-Validation Procedure We consider the case where K-fold cross-validation is performed. Let
(X, y) be the observed data set and {(Tk, Vk)}k∈[K] be the K types of partition of training and validation sets, which satisfies Tk, Vk ∈ 2
[n], Tk ∩ Vk = ∅, and Tk ∪ Vk = [n] for any k ∈ [K]. Then, the cross-validation error Es(X, y) for the pipeline Ps is defined as

$$E_{s}(X,\mathbf{y})=\sum_{k\in[K]}{\frac{1}{|V_{k}|}}\|(D_{X}^{s}\mathbf{y})_{V_{k}}-X_{V_{k},\mathcal{M}_{s,k}}{\hat{\mathbf{\beta}}}_{s,k}(\mathbf{y})\|_{2}^{2},$$

where DsX is the linear transformation matrix in the missing value imputation of the pipeline Ps, βˆs,k(y) =
X⊤
Tk\Os,k,Ms,k XTk\Os,k,Ms,k −1X⊤
Tk\Os,k,Ms,k
(DsXy)Tk\Os,k , and (Ms,k, Os,k) is the output of the pipeline Ps with input (XTk,(DsXy)Tk). In K-fold cross-validation, the pipeline Ps
∗ is selected to minimize the cross-validation error Es(X, y), i.e., s
∗ = arg mins∈[S] Es(X, y).

## F.2. Auto-Conditioning For Cross-Validation

To conduct the statistical inference after cross-validation, it is suffice to have the procedure to compute the interval [Lz, Uz]
for any z ∈ R which satisfy

$$\forall r\in[L_{z},U_{z}],$$  $$\operatorname*{arg\,min}_{s\in[S]}E_{s}(X,\boldsymbol{a}+\boldsymbol{b}r)=\operatorname*{arg\,min}_{s\in[S]}E_{s}(X,\boldsymbol{a}+\boldsymbol{b}z)(:=s(z)),$$  $$\mathcal{P}_{s(z)}(X,\boldsymbol{a}+\boldsymbol{b}r)=\mathcal{P}_{s(z)}(X,\boldsymbol{a}+\boldsymbol{b}z).$$

If we have this procedure, for any r ∈ [Lz, Uz], the selected features and the detected outliers after selecting the pipeline by cross-validation from the data set (X, a + br) are invariant. Therefore, the pselective can be computed in exactly the same way as in §4 only by adding the condition D
s(z)
X = Ds
∗
X as well as the condition Ma+bz = My and Oa+bz = Oy.

Hereafter, we provide the above procedure by extending Algorithm 1. For implementation of the above procedure, we compute two intervals [L
cv z, Ucv z] and [L
sel z, Usel z] for any z ∈ R which satisfy

$\forall r\in[L^{\rm ev}_{z},U^{\rm ev}_{z}],\ \mathop{\rm arg\,min}_{s\in[S]}E_{s}(X,\mathbf{a}+\mathbf{br})=\mathop{\rm arg\,min}_{s\in[S]}E_{s}(X,\mathbf{a}+\mathbf{bz})(:=s(z)),$  $\forall r\in[L^{\rm sel}_{z},U^{\rm osc}_{z}],\ {\cal P}_{s(z)}(X,\mathbf{a}+\mathbf{br})={\cal P}_{s(z)}(X,\mathbf{a}+\mathbf{bz}),$
respectively, and let Lz = max(L
cv z, Lsel z) and Uz = min(U
cv z, Usel z).

To compute the interval [L
cv z, Ucv z], we use Algorithm 1 repeatedly. For any (*s, k*) ∈ [S] × [K] and any z ∈ R, we compute the interval [L
(s,k)
z , U(s,k)
z ] which satisfy
∀r ∈ [L
(s,k)
z, U(s,k)
z], Ps(XTk,(DsXa + DsXbr)Tk) = Ps(XTk,(DsXa + DsXbz)Tk),
by using Algorithm 1 with input (Ps, XTk,(DsXa + DsXbz)Tk, z). Thus, if we consider the k-th term of the sum in Es(X, a + br) as a function of r, then it becomes quadratic in r on the interval [L
(s,k)
z , U(s,k)
z ]. Therefore, on the interval
∩s∈[S] ∩k∈[K][L
(s,k)
z , U(s,k)
z ], the cross-validation errors {Es(X, a + br)}s∈[S] are all quadratic in r. On this interval
∩s∈[S] ∩k∈[K][L
(s,k)
z , U(s,k)
z ], the simultaneous inequalities for r

$$E_{s(z)}(X,a+b)$$

Es(z)(X, a + br) ≤ Es(X, a + br), ∀s ∈ [S],
with s(z) = arg mins∈[S] Es(X, a + bz) become simultaneous quadratic inequalities, which can be solved analytically to finally obtain the interval [L
cv z
, Ucv z
].

To compute the interval [L
sel z
, Usel z], we simply use Algorithm 1 with input (Ps(z), X, a, b, z).

## G. Examples Of Implementations

We show an example of how the pipeline is implemented in our experiments. Listing 2 shows the implementation of the automatic pipeline construction scheme referred to as cv in the experiments (§6). Note that we can specify the candidates of the parameters for each operation and perform cross-validation to determine the optimal pipeline by using fit method.

Listing 2. Code example that defines the automatic pipeline construction scheme referred to as cv in the experiments. We can create an instance of manager class which handles identically structured pipelines, each with a different hyperparameter set, simply by specifying each operation and its candidates of parameters in turn (corresponding to option1 multi and option2 multi). Manager instances can use the OR operator | to create new manager instance which handles all of the pipelines that each instance handles. To perform hypothesis testing after cross-validation, we can call the fit and inference method of the manager instance sequentially. import numpy as np from si4pipeline **import** *
def option1_multi() -> PipelineManager:
X, y = initialize_dataset() y = mean_value_imputation(X, y) O = soft_ipod(X, y, [0.02, 0.018]) X, y = remove_outliers(X, y, O) M = marginal_screening(X, y, [3, 5]) X = extract_features(X, M) M1 = stepwise_feature_selection(X, y, [2, 3]) M2 = lasso(X, y, [0.08, 0.12]) M = union(M1, M2) return construct_pipelines(output=M)
def option2_multi() -> PipelineManager:
X, y = initialize_dataset() y = definite_regression_imputation(X, y) M = marginal_screening(X, y, [3, 5]) X = extract_features(X, M) O = cook_distance(X, y, [2.0, 3.0]) X, y = remove_outliers(X, y, O) M1 = stepwise_feature_selection(X, y, [2, 3]) M2 = lasso(X, y, [0.08, 0.12]) M = intersection(M1, M2)
return construct_pipelines(output=M)
manager = option1_multi() | option2_multi() X, y = np.random.normal(size=(100, 10)), np.random.normal(size=100) manager.tune(X, y, num_folds=2)
M, p_list = manager.inference(X, y, sigma=1.0)