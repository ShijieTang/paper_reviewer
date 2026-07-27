# Data Shapley In One Training Run

| Jiachen T. Wang      | Prateek Mittal       | Dawn Song   | Ruoxi Jia     |
|----------------------|----------------------|-------------|---------------|
| Princeton University | Princeton University | UC Berkeley | Virginia Tech |

## Abstract

Data Shapley offers a principled framework for attributing the contribution of data within machine learning contexts. However, the traditional notion of Data Shapley requires re-training models on various data subsets, which becomes computationally infeasible for large-scale models. Additionally, this retraining-based definition cannot evaluate the contribution of data for a specific model training run, which may often be of interest in practice. This paper introduces a novel concept, *In-Run Data Shapley*, which eliminates the need for model retraining and is specifically designed for assessing data contribution for a particular model of interest. In-Run Data Shapley calculates the Shapley value for each gradient update iteration and accumulates these values throughout the training process. We present several techniques that allow the efficient scaling of In-Run Data Shapley to the size of foundation models. In its most optimized implementation, our method adds negligible runtime overhead compared to standard model training. This dramatic efficiency improvement makes it possible to perform data attribution for the foundation model pretraining stage. We present several case studies that offer fresh insights into pretraining data's contribution and discuss their implications for copyright in generative AI and pretraining data curation.

## 1 Introduction

In today's data-driven world, understanding the contribution of each data point is crucial, especially with the advent of foundation models that rely on vast amounts of training data from various sources. The lack of reliable data attribution mechanisms can lead to significant legal and societal issues, resulting in a growing backlash against the broader use of data for model training (Heikkilä, 2023). For instance, there is a risk of violating intellectual property rights, failing to fairly compensate data creators, and disincentivizing them from producing new, high-quality content (Henderson et al., 2023). This has already resulted in legal disputes, such as the New York Times' lawsuit against Microsoft/OpenAI (Grynbaum & Mac, 2023). Moreover, foundation models are often trained on massive datasets scraped from the internet, which can include low-quality and harmful content (Gao et al., 2020; Raffel et al., 2020; Touvron et al., 2023). Problematic data not only wastes computational resources but also skews model outputs, potentially leading to biased or inaccurate results. By understanding the contribution of each data source, we can identify and mitigate the influence of low-quality data, thereby improving the efficiency and quality of model training. Since the training data of foundation models comes from multiple stakeholders, it is essential to have an algorithm that can *fairly* attribute data contributions. In recent years, significant progress has been made in understanding what it means to fairly quantify and attribute data source contributions, with *the Shapley value* (Shapley, 1953) emerging as a widely adopted framework (Ghorbani & Zou, 2019; Jia et al., 2019b). Originating from cooperative game theory, the Shapley value uniquely satisfies several desirable properties: (1) It assigns equal scores to equally impactful data points, ensuring fairness; (2) the sum of contribution scores equals the total utility, meaning the scores always represent a share of the total utility; (3) it supports additive decomposition across multiple utility functions, allowing for the calculation of contributions to the entire test set by summing the scores of individual test points. As the Shapley value uniquely satisfies these properties, it eliminates the uncertainty and ambiguity surrounding which attribution frameworks should be used conceptually. While there are other non-Shapley data attribution frameworks, such as Datamodels (Park et al., 2023; Ilyas et al., 2022) and influence functions (Koh & Liang, 2017), they lack the clear theoretical foundation and uniqueness provided by the Shapley value.

1 Original Data Shapley definition faces computational & conceptual limitation. However, Data Shapley, i.e., the application of the Shapley value in data attribution, has been limited to very smallscale models. Existing methods (Ghorbani & Zou, 2019) to estimate the Shapley value require retraining the model numerous times using different subsets of data to evaluate the contribution of each data source, making them computationally infeasible for foundation models. On the other hand, retraining-based methods suffer from a conceptual issue often overlooked in the literature: **they** assess data contribution for a general learning algorithm rather than a particular model. While one might interpret the former as an approximation of the latter, these two quantities can be quite different in practice, especially when the learning algorithm is randomized and sensitive to factors like random initialization and training data order. In many real-life scenarios, however, the primary interest lies in understanding data contribution to the specific model being trained and deployed. This paper introduces *In-Run Data Shapley*, a novel approach that makes fair data attribution applicable to large-scale foundation models. Unlike retraining-based Data Shapley, In-Run Data Shapley quantifies the contribution of each data source to *the specific target model of interest*. Technical contributions. Our key insight is that ML models are trained using iterative algorithms, where the model performance change in one iteration is sufficiently small to be accurately approximated by first- or second-order Taylor expansions. We show that the Shapley value for the approximated one-step model performance change can be derived analytically via gradient dotproducts or gradient-Hessian-gradient products between training and validation data. Hence, we can compute the Data Shapley scores for each model update step, and accumulate the scores throughout the training process. However, the per-sample gradient vectors required for computing the Shapley value in each training iteration introduce significant overhead due to per-sample gradient calculation.

To address this challenge, we develop a series of technical tools that enable the exact calculation of gradient dot-products and gradient-Hessian-gradient products in one and two backward passes, respectively, without the need to instantiate any additional gradient vectors or Hessian matrices. Collectively, these tools allow for the efficient computation of In-Run Data Shapley. In particular, with sufficient GPU memory, its most efficient implementation is as fast as regular training.

Empirical implications. Given the efficient algorithms developed in this paper, for the first time, one can perform data attribution on the scale of foundation model pretraining. While in this paper, we focus on GPT2 and Pythia-410M as a pilot study, our approach is applicable to larger-scale industrial models with sufficient computing resources. We performed various case studies that provide fresh insights into training data's contribution to the foundation model pretraining. (1) There is considerable room for improvement in data curation for pretraining (Section 5.3.1). Even well-curated pretraining corpora contain data points that negatively impact the training process. We demonstrate the effectiveness of In-Run Data Shapley in identifying these low-quality data points. By computing In-Run Data Shapley values during training and removing negatively valued data points, we show that the cleaned dataset leads to significantly faster model convergence and improved performance compared to the original dataset. Interestingly, despite the Pile dataset (Gao et al., 2020) already undergoing multiple layers of curation, In-Run Data Shapley assigns negative values to approximately 16% of the data. We found a significant amount of noisy data among them, highlighting the need for improved data curation for foundation model training. (2) Data's contribution is stage-dependent (Section 5.3.2). In-Run Data Shapley can capture the dynamics of contribution through the course of training, a fine-grained aspect that cannot be captured by prior works. In-Run Data Shapley shows that in the early stages of training, general corpora tend to have a relatively large contribution regardless of the downstream tasks. This is because general corpora help the model learn basic language patterns, grammar, and common knowledge. However, in the later stages of training, the contribution from domain-specific corpora becomes dominant, and the contribution of the general corpus phases out. (3) Rethinking copyright in generative AI: contribution beyond memorization (Section 5.3.3). We studied training data's contribution to validation points of varying similarity levels. We found that even when the validation data is a complete rewrite of the training data while maintaining the topic, the training data still contributes significantly. This finding has implications for the current dialogue around what constitutes a copyright violation in generative AI (Mulligan & Li, 2024). While the unfair use of copyrighted content is generally only considered when the generated data is an almost verbatim replication of the training data, our contribution analysis shows that some data owners should receive a certain royalty share for generated content, even if the output does not closely resemble the copyrighted material.

## 2 Background Of Data Shapley

In this section, we formalize the setup of data attribution for ML and revisit Data Shapley's definition.

Setup & Goal. Given a dataset Dtr := {zi}
N
i=1, data attribution or valuation aims to assign a score to each training data point zi, reflecting its importance for the trained ML model's performance on a certain task. Formally, we seek a score vector (ϕzi)
N
i=1 where each ϕzi ∈ R reflects the *value* of zi.

The Shapley value (SV) (Shapley, 1953), originating from game theory, stands out as a distinguished method for equitably distributing total profit among all participating players. Before diving into its definition, we first discuss a fundamental concept: the *utility function*. Utility function. A *utility function* maps an input dataset to a score indicating the utility of the dataset for model training. In most of the existing literature (Ghorbani & Zou, 2019; Jia et al., 2019b), the utility function U is chosen as the performance (e.g., accuracy or loss) of the trained models on a hold-out validation set. That is, given a training set S, the utility function U(S) := Perf(A(S)), where A represents a learning algorithm that trains a model on dataset S, and Perf(·) is a function assessing the model's performance. For example, Perf(·) can be the accuracy for a classification task or the perplexity for a language completion task, evaluated on a (set of) hold-out validation data.

Definition 1 (Shapley value (Shapley, 1953)). Let U(·) denote a utility function and D represent a training set of N data points. The Shapley value, ϕz (U), assigned to a data point z ∈ D is defined as ϕz (U) := 1N
PN
k=1 N−1 k−1
−1 PS⊆D−z,|S|=k−1
[U(S ∪ {z}) − U(S)] where D−z = D \ {z}.

In simple terms, the Shapley value is a weighted average of the marginal contribution U(S ∪ {z}) − U(S), i.e., the utility change when the point z is added to different Ss. For simplicity, we often write ϕz when the utility function is clear from the context. The popularity of the Shapley value is attributable to the fact that it is the *unique* data value notion satisfying four axioms: Null player, Symmetry, Linearity, and Efficiency. The mathematical definitions of these axioms are deferred to Appendix A.1. Here, we introduce the *linearity* axiom which will be used later. Theorem 2 (Linearity of the Shapley value (Shapley, 1953)). For any of two utility functions U1, U2 and any α1, α2 ∈ R*, we have* ϕz (α1U1 + α2U2) = α1ϕz (U1) + α2ϕz (U2).

Retraining-based Data Shapley. The convention of defining the utility function for Data Shapley as U(S) = Perf(A(S)) was introduced in Ghorbani & Zou (2019), where A is a learning algorithm such as a neural network trained by stochastic gradient descent (SGD) or its variants. With this choice of utility function, the precise calculation of the Shapley value requires retraining models on various subsets of the data. This is because the marginal contribution of a data point, U(S ∪ {z}) − U(S), can only be obtained by training models on both S and S ∪ {z} and comparing their performance.

As a result, we refer to this method as "*Retraining-based Data Shapley*". Limitations beyond efficiency (detailed in Appendix B.1): In addition to the high computational costs, we emphasize that retraining-based Data Shapley also suffers from the following limitations:
(1) Highly unstable value scores: When stochastic learning algorithms such as SGD are used, the resulting value scores can be highly unstable (Wang & Jia, 2023a). This instability may lead to unreliable results and potential violations of Shapley value's fairness axioms. **(2) Conceptual** limitations: Retraining-based Data Shapley measures the average data contribution to the learning process itself, across many retrainings on different data subsets. As a result, it produces attribution scores that apply broadly to the algorithm but fail to reflect the data contribution to a specific training run. On the other hand, providing insights into how individual data points contribute to the deployed model enables more targeted analysis and debugging, thereby improving model interpretability.

## 3 In-Run Data Shapley

To address the issues associated with Retraining-based Data Shapley such as high computational costs, value instability, and the inability to assess the contribution towards a specific trained model, we propose a novel data attribution method specifically tailored for a single training run. Our key idea is to leverage the iterative nature of model training and employ a "divide and conquer" approach: breaking down the problem of valuing data contributions for the entire training process into subproblems of valuing data contributions for individual iterations. Utility function for a single gradient update. Traditionally, the utility function U(S) = Perf(A(S)) encapsulates the overall impact of a training set S across the complete training process. Here, we instead consider a "local utility function" that evaluates the impact of data subsets within a single iteration. Specifically, given a training dataset Dtr = {zi}
N
i=1, a deep learning model is usually being trained to minimize the training loss PN
i=1 ℓ(*w, z*i) via an iterative optimization procedure such as SGD. The performance of the model is typically being measured through a set of validation points {z
(val)}. During an iteration t, a batch Bt ⊆ Dtr of the training points is used to update the model parameters from wt to wt+1 with wt+1 := wt − ηtPz∈Bt ∇ℓ(wt, z), where ηt is the learning rate at iteration t.

1 A complete run of neural network training thus consists of model checkpoints
{w0, w1*, . . . , w*T }. For a given validation data point z
(val), we can define the "local utility function" at a single iteration t as

$$U^{(t)}(S;z^{\mathrm{(val)}}):=\ell(\widetilde{w}_{t+1}(S),z^{\mathrm{(val)}})-\ell(w_{t},z^{\mathrm{(val)}})$$
$\left(\mathbb{I}\right)$. 
(val)) := ℓ(wet+1(S), z(val)) − ℓ(wt, z(val)) (1)
where wet+1(S) := wt − ηtPz∈S ∇ℓ(wt, z) and S ⊆ Bt is a subset of the batch being selected in t-th iteration in the original training. **Interpretation:** The local utility function U
(t)represents the loss change at iteration t when only the subset S is used for the gradient update. This approach incorporates the realization of random batch selection at t-th iteration into the utility function. It can also encode other forms of training randomness (e.g., dropout) at iteration t. By accounting for the specific realization of training randomness, we obtain a deterministic utility function for each iteration, effectively enabling the targeted attribution to the specific training run. Data Shapley for a single gradient update. While the utility U
(t)is defined over Bt instead of the full training set Dtr, it is easy to augment it to Dtr. More formally, in the augmented utility function we have wet+1(S) := wt − ηtPz∈S∩Bt ∇ℓ(wt, z), S ⊆ Dtr. The Shapley value ϕz(U
(t)) will be exactly the same as the Shapley value corresponds to the augmented utility function for any z ∈ Bt, and ϕz(U
(t)) = 0 for any z ∈ Dtr \ Bt (see Theorem 5 in Wang & Jia (2023b)). Therefore, for a clean presentation, we slightly abuse the notation where U
(t)'s meaning depends on the context.

Data Shapley for the entire training run. Building on the concept of a "local" utility function for a single gradient update iteration, we naturally extend this to a "global" utility function for the entire training process, defined as U(S) = PT −1 t=0 U
(t)(S). **Interpretation:** This global utility function can be interpreted as the cumulative loss change of the entire training run, but under the counterfactual scenario where only a subset of the training data S is used. In other words, it aggregates the total impact of the subset S on the model's performance throughout the entire training process. Due to the linearity property of the Shapley value (Theorem 2), we have ϕz(U) = PT −1 t=0 ϕz(U
(t)). This new Data Shapley value, which we call *In-Run Data Shapley*, represents the cumulative contribution of the data point z across all gradient update iterations within a single training run. This approach breaks down the broader utility into more manageable, step-by-step assessments that capture the immediate effects of data points on model updates, and provide a more fine-grained view of how individual data points contribute to the model's performance at each step of the training process. Notably, the sum of individual data points' Shapley values equals the overall loss reduction achieved by the model during the entire training run due to the Shapley value's efficiency axiom (see Appendix A.1). This provides a meaningful and interpretable measure of data importance. In Appendix B, we give an in-depth comparison between Retraining-based and In-Run Data Shapley. Remark 1 (**Multiple validation points**). *In practice, the model performance is often being* assessed based on a validation set D(val) = {z
(val)}*. After computing* ϕzU(·; z
(val))for each z
(val) ∈ D(val), one can compute the Shapley value corresponding to the utility function on the full validation set U(S; D(val)) := Pz
(val)∈D(val) U(S; z
(val)) *by simply taking the sum* ϕz U(·; D(val))=Pz
(val)∈D(val) ϕz U(·; z
(val))*due to the* linearity property of the Shapley value (Theorem 2). Hence, for a clean presentation, we consider only a single z
(val) in this paper.

However, all the techniques we developed can be extended to multiple validation points.

## 4 Efficient Computation Of In-Run Data Shapley

The newly proposed In-Run Data Shapley does not require retraining models from scratch on different data subsets. However, calculating ϕz(U
(t)) for each training iteration remains computationally intensive, as it involves evaluating the performance impact of all possible combinations within the sampled data batch. In this section, we introduce an efficient method for approximating In-Run Data Shapley scores during a specific training run. Our approach, distinct from Monte Carlo methods, is deterministic and optimized to minimize additional runtime to regular training. In particular, in its most efficient implementation, our approximation technique incurs negligible extra runtime beyond what is required for standard model training, making it highly practical for real-world applications.

## 4.1 Approximating U (T) With Taylor Expansion

To derive a more tractable structure for the local utility function U
(t), we propose using first and second-order Taylor approximations. The advantage of this approach is that the approximated utility function exhibits a form where closed-form Data Shapley formulas can be derived. The second-order Taylor approximation to the local utility function is as follows:
U
(t)(S) = ℓ(wet+1(S), z(val)) − ℓ(wt, z(val))

$\hat{f}=\epsilon(w_{t+1}(S),z^{(\text{out})})-\epsilon_{1}(w_{t},z^{(\text{out})})$  $=\underbrace{\nabla\ell(w_{t},z^{(\text{out})})\cdot(\widehat{w}_{t+1}(S)-w_{t})}_{U^{(1)}_{(1)}(S)}+\frac{1}{2}\underbrace{(\widehat{w}_{t+1}(S)-w_{t})^{\intercal}\mathbf{H}_{t}^{(z^{(\text{out})})}(\widehat{w}_{t+1}(S)-w_{t})}_{U^{(1)}_{(2)}(S)}+\text{higher order terms}$
$\nabla\tau2\,\epsilon\epsilon$. 
where the Hessian matrix H
$$\mathbf{H}_{\mathrm{f}}^{(z^{\mathrm{(wall)}})}$$
t:= ∇2ℓ(wt, z(val)). We label the first-order term as U
(t)
(1)(S)
and the second-order term as U
(t)
(2)(S). Note that the gradient update wet+1(S) − wt =
−ηtPz∈S ∇ℓ(wt, z). Given that the learning rate ηt in model training is typically small, a lowerorder Taylor expansion often provides an accurate approximation for the change in loss during a single gradient update, with approximation errors of O(η
2
t) and O(η
3
t) for first and second-order
approximations, respectively. In Appendix E.2.2, we empirically investigate the errors of first- and second-order approximations to U
(t) on GPT2. In particular, the first-order approximation can already
second-order approximations to $U\cdot\nabla$ on $\mathbf{T}\mathbf{2}$. In particular, the $\mathbf{T}\mathbf{2}$-achieve a great performance with Spearman correlation $>0.94$.  **First-order In-Run Data Shapley.** Using the first-order approximation, the gradient update expression, we have $U_{(1)}^{(t)}(S)=-\eta_{t}\sum_{z\in S}\nabla_{z}$ that $U_{(1)}^{(t)}$ is an _additive_ utility function with a closed-form Shapley Theorem 3. _In-Run Data Shapley considering the first-order approximation_
(t) ≈ U
(t)
(1), and substituting
(1)(S) = −ηtPz∈S ∇ℓ(wt, z(val))·∇ℓ(wt, z). This shows
(1) is an *additive* utility function with a closed-form Shapley calculation as follows:
Theorem 3. *In-Run Data Shapley considering the first-order approximation has closed-form*
$$\phi_{z}\left(U\right)\approx\sum_{t=0}^{T-1}\phi_{z}\left(U_{(1)}^{(t)}\right)$$
where

$\phi_{z}\left(U_{(1)}^{(t)}\right)=-\eta_{t}\nabla\ell(w_{t},z^{(\rm val)})\cdot\nabla\ell(w_{t},z),\ \ t=0,\ldots,T-1$
Thus, the first-order approximation of In-Run Data Shapley for a training point accumulates its gradient dot products with the validation data point each time the training point is sampled in the training batch. The gradient dot product between the training point zi and the validation point z
(val) represents the direct influence of zi on the validation loss at the current model parameters wt, which essentially measures the alignment between the two gradient vectors in the parameter space.

Notably, ϕz U
(t)
(1)is equivalent to the TracIN-Ideal score proposed by (Pruthi et al., 2020). That is, the TracIN-Ideal score can be interpreted as the Shapley value when we use first-order Taylor approximation for U
(t). However, TracIN-Ideal has been described as "computationally infeasible,"
and our approach completely overcomes this problem. In Appendix A.3, we provide a detailed discussion of the differences between this work and (Pruthi et al., 2020). Second-order In-Run Data Shapley. We further improve the approximation of U
(t) using a secondorder Taylor expansion, i.e., U
(t) ≈ U
(t)
(1) +
1 2 U
(t)
(2). Fortunately, the approximated utility function maintains a tractable structure that allows a closed-form Shapley value calculation. Theorem 4. *In-Run Data Shapley considering the second-order approximation has closed-form*

$$\phi_{z}\left(U\right)\approx\sum_{t=0}^{T-1}\left(\phi_{z}\left(U_{(1)}^{(t)}\right)+\frac{1}{2}\phi_{z}\left(U_{(2)}^{(t)}\right)\right)$$
(2)(2)

$$(2)$$

where

$$\phi_{z}\left(U_{(1)}^{(t)}\right)+\frac{1}{2}\phi_{z}\left(U_{(2)}^{(t)}\right)=\underbrace{-\eta_{t}\nabla\ell(w_{t},z^{(\text{volt})})\cdot\nabla\ell(w_{t},z)}_{\bigoplus\text{subfaces of an arc length of}\ell(z^{(\text{volt})})}+\underbrace{\frac{\eta_{t}^{2}}{2}\nabla\ell(w_{t},z)^{\intercal}H_{t}^{(z^{(\text{volt})})}}_{\bigoplus\text{Information between$z$and other continuity points}}\left(\sum_{z_{i}\in\mathcal{R}_{t}}\nabla\ell(w_{t},z_{j})\right).$$
$$(3)$$
for any $t=0,\ldots,T-1$.  
Compared to the first-order variant, the second-order In-Run Data Shapley includes an additional gradient-Hessian-gradient product term that captures the interaction between the training point of interest z and the rest of the training set. The Hessian matrix represents the curvature of the validation loss function at the current model parameters wt. This interaction term measures the alignment between the gradient of z and the gradients of the other points in the training batch, adjusted by the Hessian. If this term is large, it indicates that the presence of other points in the batch significantly impacts the value attributed to z. For example, if there are many identical or similar copies of z in the training set, the contribution of z will decrease, as the interaction term will be large, effectively distributing the value among the similar points. By incorporating this interaction term, the secondorder In-Run Data Shapley provides a more fine-grained contribution measure that takes into account both the relevance of a data point towards a validation set and its uniqueness within the population.

## 4.2 Efficient Computation Of Gradient Dot-Product And Gradient-Hessian-Gradient Product

Although we have derived closed-form formulas for In-Run Data Shapley using first- or second-order Taylor approximation of the local utility functions, efficiently computing these values remains a challenge. Specifically, for the first-order In-Run Data Shapley, it requires computing 1 the pairwise gradient dot products between each z ∈ Bt and the validation point. For the second-order In-Run Data Shapley, it additionally requires computing 2 the gradient-Hessian-gradient products for each z ∈ Bt. A direct implementation to compute 1 involves calculating the individual gradient for each data point in Bt, which cannot benefit from fast batch processing in GPUs and necessitates running backpropagation |Bt| times with a mini-batch size of 1. Consequently, this approach would be at least |Bt| times slower than regular training, making it computationally prohibitive for practical applications. Furthermore, computing 2 requires either computing each individual gradient again or storing all individual gradients, which incurs significant time or memory overhead. Computing pair-wise gradient dot-products in 1 backpropagation. Our technique for efficiently computing pairwise gradient dot products is inspired by the "ghost clipping" technique from the differential privacy (DP) literature (Lee & Kifer, 2021). "Ghost clipping" enables computing all of the per-sample gradient norms within one backpropagation without explicitly forming any individual gradient vectors, which enhances the efficiency of DP model training. Here, we propose a "ghost dot-product" technique that shares the idea of exploiting the computation that has been done in the backpropagation. Specifically, denote a sample batch as Bt = {z1*, . . . , z*B}. We demonstrate this technique using a simple linear layer s = aW, where W ∈ R
d1×d2is the weight matrix, a = (a
(1)*, . . . ,* a
(B))
⊺
is the mini-batch input, and s = (s
(1)*, . . . ,*s
(B))
⊺
is the output (i.e., the pre-activation tensor). For (non-sequential) data, a ∈ R
B×d1,s ∈ R
B×d2. By applying the chain rule, we can express the gradient of an individual loss ℓ
(i):= ℓ(*w, z*i) with respect to W as

$$\frac{\partial\ell^{(i)}}{\partial\mathbf{W}}=\frac{\partial\ell^{(i)}}{\partial\mathbf{s}^{(i)}}\otimes\frac{\partial\mathbf{s}^{(i)}}{\partial\mathbf{W}}=\frac{\partial\ell^{(i)}}{\partial\mathbf{s}^{(i)}}\otimes\mathbf{a}^{(i)}=\frac{\partial\ell}{\partial\mathbf{s}^{(i)}}\otimes\mathbf{a}^{(i)}\tag{4}$$

where ℓ := PB
j=1 ℓ
(j)is the aggregated loss, and the last step is because other data points' losses have no dependency on si. Note that the individual's output gradient ∂ℓ(i)
∂s
(i) =∂ℓ
∂s
(i)is readily available during the backpropagation pass in terms of ℓ. Suppose we are interested in computing the gradient dot-product ∂ℓ(1)
∂W ⊙
∂ℓ(2)
∂W between two data points z1, z2 in the same batch in the backpropagation.

For non-sequential data, we have each a
(i) ∈ R
d1×1and ∂ℓ(i)
∂s
(i) ∈ R
1×d2. By (4), we have

$${\frac{\partial\ell^{(1)}}{\partial\mathbf{W}}}\odot{\frac{\partial\ell^{(2)}}{\partial\mathbf{W}}}=\left(\mathbf{a}^{(1)}\otimes{\frac{\partial\ell^{(1)}}{\partial\mathbf{s}^{(1)}}}\right)\odot\left(\mathbf{a}^{(2)}\otimes{\frac{\partial\ell^{(2)}}{\partial\mathbf{s}^{(2)}}}\right)=\left(\left(\mathbf{a}^{(1)}\right)^{\mathsf{T}}\mathbf{a}^{(2)}\right)\left(\left({\frac{\partial\ell^{(1)}}{\partial\mathbf{s}^{(1)}}}\right)^{\mathsf{T}}\left({\frac{\partial\ell^{(2)}}{\partial\mathbf{s}^{(2)}}}\right)\right).$$
!(5)
$$(S)$$

Hence, we can first take the two inner products, and then multiply the results together. All of the quantities a
(1), a
(2),
∂ℓ(1)
∂s(1) ,
∂ℓ(2)
∂s(2) in (5) that are required for computation are all already available in the backpropagation. Hence, within a *single* backpropagation, we can efficiently compute the gradient dot-product between *every* pair of zi, zj ∈ Bt. Since we are interested in computing the gradient dot-product between z
(val) and z for all z ∈ Bt, we can backpropagate on Pz∈Bt ℓ
(i) + ℓ
(z
(val))
to save another backpropagation for z
(val). We call this technique the *"ghost dot-product"*, as no gradient vectors are instantiated during the computation. Overall, we only need one backpropagation to compute 1 for all data points in Bt, a significant improvement over the direct method requiring ≥ |Bt| backpropagations. Additional details for this technique are in Appendix D.

Remark 2. While we illustrate our "ghost dot-product" technique using linear layers, it can be extended to other types of layers by leveraging similar decompositions as in Equation (4) that have been developed in differential privacy literature (Rochette et al., 2020; Bu et al., 2022; Li et al., 2021; Bu et al., 2023; Kong & Munoz Medina, 2024). Computing gradient-Hessian-gradient products in 2 backpropagations (Appendix D.2). For second-order In-Run Data Shapley, an outstanding challenge is how to efficiently compute the pairwise interaction term among training points. In Appendix D.2, we develop a "ghost gradient-Hessiangradient product" technique for computing the desired quantity through one extra backpropagation pass, without materializing any gradient-sized vectors. This technique leverages several properties of neural network gradients across different layers, and its derivation is complex. Further improvement of runtime and memory requirements (Appendix D.3). With the "ghost" techniques developed, the computation of first- and second-order In-Run Data Shapley requires one and two backpropagations in each gradient update iteration respectively. Although we still need to compute the gradient of the aggregated loss Pzi∈Bt ℓi for the training batch to perform parameter updates, we do not need an additional backpropagation. By reusing the activations and output gradients from the previous backpropagation on Pz∈Bt ℓ
(i) + ℓ
(z
(val)), we can easily compute this quantity *without* incurring the cost of an extra backpropagation pass. Consequently, training while computing first-order In-Run Data Shapley will have minimal additional runtime overhead, as it still requires only one backpropagation per iteration. The second-order In-Run Data Shapley necessitates one extra backpropagation per iteration. Nevertheless, both methods provide significant advantages over the direct approach of instantiating per-sample gradients and Hessian-vector products.

## 5 Experiments

In this section, we evaluate In-Run Data Shapley in terms of its efficiency (Section 5.1), fidelity (Section 5.2), and its applications in data attribution for language model pretraining (Section 5.3). In Appendix E.6, we compare a variety of existing data attribution methods in small-scale settings.

## 5.1 Runtime Evaluation

We empirically assess the computational efficiency of In-Run Data Shapley with "ghost dot-product" and "ghost vector-Hessian-vector product" techniques developed in Section 4.2. We compare this to the direct implementation of In-Run Data Shapley, which requires computing per-sample gradients, as well as to regular training without Data Shapley computations. The experiment is conducted by training GPT2-Small on a single 80GB A100 GPU. As illustrated in Table 1, the runtime of first-order In-Run Data Shapley is close to that of regular training when using the ghost dot-product algorithms developed in Section 4.2. The second-order In-Run Data Shapley is approximately 2× slower than regular training due to the additional backpropagation. However, both the first- and second-order In-Run Data Shapley are significantly faster (> 30×) compared to the naive implementation. These results showcase the substantial improvements achieved by our techniques, making In-Run Data Shapley computationally feasible for practical applications.

Throughput Regular Training 76.2 First-order Data Shapley (ghost) 70.5 Second-order Data Shapley (ghost) 34.4 First-order Data Shapley (direct) 4.2 Second-order Data Shapley (direct) 1.8 Table 1: Efficiency comparison of different implementations of In-Run Data Shapley. We use throughput, i.e., \# training data points being processed per second as the efficiency metric.

## 5.2 Fidelity Evaluation

In this section, we directly assess the approximation accuracy of first- and second-order In-Run Data Shapley. In Appendix E.6, we further compare the performance of In-Run Data Shapley with existing (less scalable) data attribution techniques on standard benchmarks (e.g., mislabeled data detection) as an additional sanity check. Given that computing the exact In-Run Data Shapley value is computationally prohibitive, we compare their performance against Monte Carlo estimates of the Shapley value using a large number of samples. The experiment is conducted on GPT2 model at the 3500th training iteration on Pile, where the batch size 16 and learning rate 3 × 10−4. We use 1000 permutations to approximate the groundtruth In-Run Shapley value. Figure 1 shows that with just first-order In-Run Data Shapley, the root mean squared error (RMSE) is only around 0.0003. In Appendix E.2.1, we provide additional results with different learning rates.

Figure 1: Comparison between the Monte Carloestimated In-Run Data Shapley and First/Secondorder In-Run Data Shapley.

## 5.3 Case Study: Data Attribution On Pile Dataset

In this section, we present a case study to demonstrate the use cases of In-Run Data Shapley by pretraining on the well-known Pile dataset (Gao et al., 2020). We explore its application in data curation, examine data contribution across different training stages, and investigate relevant corpus detection. Due to computational resource constraints, most of our experiments focus on the GPT-2 and Pythia-410M models, but this is not a limitation of the algorithm itself. With adequate computational resources, our approach can easily be applied to larger-scale models.

## 5.3.1 Is Well-Curated Dataset Actually Clean?

Carefully curated pretraining corpora still contain data points that can adversely affect the training process. Identifying and removing these data points can accelerate model convergence and enhance overall performance, thereby saving computational resources. In this experiment, we demonstrate the effectiveness of In-Run Data Shapley in assessing the data quality of a subset of the Pile dataset. We uniformly select a random subset of Pile with around 10B tokens and train a GPT2 model on this subset. We compute the data attribution results with Pile's validation set. By filtering out all negatively valued corpora and retraining the model on the cleaned subset, we observe significant improvement in model convergence. For both first- and second-order In-Run Data Shapely, we can achieve around 25% fewer training iterations to reach a test loss of 3.75. Surprisingly, our analysis reveals that around 16% of the training corpora had negative second-order In-Run Data Shapley values. While some of the negatively valued corpora may be attributed to the significant domain shift compared to the validation corpora, we still find many low-quality corpora from Pile, a pretraining dataset that has undergone several layers of data curation (Gao et al., 2020). Examples of low-quality corpora identified can be found in Appendix E.3. Figure 2 shows a performance comparison between the original training run and the model trained on the cleaned subsets for GPT2, and additional results on Pythia-410M are available in Appendix E.3.1. We also compare with influence function (Koh & Liang, 2017), which approximates the change in the model's loss on the test example when the training example is removed from the training set (i.e., the leave-one-out score).

We omit TRAK (Park et al., 2023) and other techniques such as datamodel (Ilyas et al., 2022) as they are not scalable to our setting. As we can see, influence function can also filter out low-quality data that can accelerate training convergence.

However, the performance is slightly worse than In-Run Data Shapley as the influence function only uses information from the final trained models, which can result in highly noisy value scores since the removal of one training data point might have a negligible effect on the final model performance. The results demonstrate that removing lower-quality Figure 2: Test loss comparison between the original training run and the model trained on the cleaned subset according to different data attribution techniques.

| Original Wikipedia Corpus   | Synthetic "Similar topic" Corpus ### Instruction: Write a short story about a classical violinist who decides to explore jazz music, detailing her first encounter with a jazz band. ### Answer: Elena, a classically trained violinist known for her precise and emotive performances ...   |                                 |                    |         |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------|--------------------|---------|
| Similarity Category         | In-Run Data Shapley (1st order)                                                                                                                                                                                                                                                              | In-Run Data Shapley (2nd order) | Influence Function | BM25    |
| Partial exactly the same    | 1                                                                                                                                                                                                                                                                                            | 1                               | 1                  | 1       |
| Paraphrase                  | 1                                                                                                                                                                                                                                                                                            | 1                               | 1                  | 1       |
| Significant paraphrase      | 32.3                                                                                                                                                                                                                                                                                         | 32                              | 39.3               | 1.6     |
| Similar topic               | 145.6                                                                                                                                                                                                                                                                                        | 141.6                           | 292                | 20917.3 |
| Table 2: Top: (left) An original training corpus from Wikipedia. (right) A synthetic corpus falls in the In 2012, Radhi recruited new 'musicians' for OAG, who were selected from among the students of Akademi Seni Budaya dan Warisan Kebangsaan (). The new line-up consists of Qi Razali (drums/backing vocals - original drummer ...                             |                                                                                                                                                                                                                                                                                              |                                 |                    |         |

### Instruction: Write a short story about a classical violinist who decides to explore jazz music, detailing her first encounter with a jazz band. ### Answer: Elena, a classically trained violinist known for her precise and emotive performances ...

Similarity Category In-Run Data Shapley (1st order) In-Run Data Shapley (2nd order) Influence Function BM25

Partial exactly the same 1 1 1 1

Paraphrase 1 1 1 1

Significant paraphrase 32.3 32 39.3 1.6

Similar topic 145.6 141.6 292 20917.3

Table 2: Top: (left) An original training corpus from Wikipedia. (right) A synthetic corpus falls in the

category of "Similar topic" to the Wikipedia corpus on the left (prompt in Appendix E.5). Bottom:

the (average) value rank of the original corpus among all training corpora for validation corpora that are of varying similarity to the original corpus. The rank is out of ≈**320k data points.**

Validation Corpus

In several applications, the matrix coefficients of the nonlinear valued function G(λ) in are usually of low rank ...

Coronavirus Testing in San Diego County\n How San Diego Is Getting Ready for Self-Driving Cars ...

Figure 3: Left: Domain value composition for a corpus of math text. Right: The math corpus we use as the validation data for attribution, and examples of high- and low-valued training corpus for it.

(+) **High-valued Corpus** The Lagrangian Lg3,g4,g5,g6 describes
(axial-)vector meson interactions, see Appendix of Ref. [@Gallas]. After ...

(−) **Low-valued Corpus**
data leads to a significantly faster drop in test loss compared to the original training run. This implies that there is still huge room for data curation for well-curated datasets such as Pile.

## 5.3.2 How Do Data Values Change During Training?

As In-Run Data Shapley tracks the cumulative data values across different training steps, we can assess the contribution of training points at various stages of training, providing a more fine-grained perspective on data attribution. We evaluate the data attribution results for a math-related validation data using second-order In-Run Data Shapley. In Figure 3, we present the value composition of training corpora by their domains over the first 10,000 training iterations, summing the values of all corpora from the same domain. We then calculate the percentage of the total value attributed to each domain, excluding domains with a total value < 0. As illustrated in the figure, the corpora from ArXiv achieve a significantly higher value compared to other domain corpora, far exceeding its size proportion within the full Pile dataset. This is expected, as ArXiv papers predominantly cover fields like Math, Computer Science, and Physics, which contain extensive math-related content. Furthermore, the value composition changes rapidly at the beginning of training and stabilizes as training progresses. We hypothesize that this initial fluctuation is due to the presence of relevant paragraphs in corpora from other domains. The stable value proportion observed in later stages likely reflects the relative abundance of math content in each domain. Interestingly, we observe that Pile-CC domain, which contains general website crawls, initially shows positive contributions during the first few iterations. However, its value quickly drops to negative and eventually converges to zero. This implies that **general corpora tend to have a large contribution in the beginning of training**, as they help the model learn the basics of languages, such as grammar and common knowledge. However, as training progresses and the model focuses on more specialized topics, the relevance of general domains diminishes. An additional figure for the average domain values is in Appendix E.4.

## 5.3.3 Does Contribution Require Memorization?

In this experiment, we evaluate the robustness of different data attribution techniques in identifying relevant individual corpora that have been paraphrased. We start by selecting a data point from the training set and creating several paraphrased versions using GPT-4, with varying levels of paraphrasing
(see Appendix E.5 for the prompt). These paraphrased versions form our validation set. We then calculate the average value rank of the original training data for each of its paraphrased versions. In addition to In-Run Data Shapley and influence function, we include the ranking result based on BM25 distance. **BM25** (Robertson et al., 2009) featurizes examples by their word frequency statistics (i.e., TF-IDF) to rank the training instances. We use BM25 distance as an oracle for assessing the verbatim or lexical similarity between the validation data (query) and the training data, as opposed to semantic similarity. As shown in Table 2, even for a validation data that is a complete rewrite (with a low BM25 distance) but covers relevant topics, the original training data still ranks very high according to both In-Run Data Shapley and influence function. Influence function ranks the original training data lower than In-Run Data Shapley, which may be attributed to the inherent noisy nature of the leave-one-out error estimation. The results of this experiment have important implications for the ongoing discussion about the copyright of generative AI. Specifically, the table presents a compelling example where the original Wikipedia training corpus related to a musician's experience can significantly contribute to generating a story about a musician, even when the generated story shares no token-wise resemblance to the original training data. This finding supports that training data profoundly influences the capabilities of generative AI models and should be compensated accordingly (Deng & Ma, 2023; Wang et al., 2024b), even when the output does not closely resemble the original copyrighted material or when the model applies output filters to avoid generating verbatim replicates of the training data. This discovery expands the conventional focus on copyright violations, which typically addresses instances of near-verbatim replication, as seen in the dispute between New York Times and OpenAI, to also include cases where the generated content is significantly influenced by copyrighted material without directly replicating it.

## 6 Conclusion And Limitations

In this work, we introduce In-Run Data Shapley, a novel data attribution technique that addresses the limitations of Retraining-based Data Shapley. Extensive experiments demonstrate the effectiveness of In-Run Data Shapley in various applications. Here, we discuss the potential limitations of this work.

Availability of validation data before training. One potential limitation of In-Run Data Shapley is that it requires the validation data to be available before training, as the data attribution scores are computed during the training process. However, there are many scenarios where validation data is naturally available before training, such as when using publicly available benchmark datasets, participating in machine learning competitions, or adhering to regulatory requirements. For scenarios where validation data arrives after the model has been trained, a potential solution is to save checkpoints of intermediate models during training and approximate In-Run Data Shapley using these checkpoints, in the same spirit of TracIN-CP described in Pruthi et al. (2020). However, the choice of checkpoints can significantly impact performance, and it is unclear which checkpoints to pick. Extension to other optimization algorithms. Extending the "ghost" family techniques developed in this work to support Adam and similar optimizers remains an exciting direction for future research. We stress that extending the formulation of In-Run Data Shapley from SGD to Adam is feasible, but the actual challenge lies in computing it efficiently without instantiating each individual gradient vector, which cannot be solved by simple extensions described in Xia et al. (2024). Handling memory constraints. In scenarios where GPU memory constraints prevent large batch sizes, the "ghost" techniques can be extended by using gradient accumulation. This approach accommodates larger training batch sizes by dividing the batch into smaller sub-batches and accumulating gradients over multiple iterations. While this method may increase runtime due to additional backpropagation steps, it maintains the feasibility of the techniques under memory constraints. Improving computational efficiency for large batch sizes remains an important direction for future research.

## Acknowledgment

This work is supported in part by the National Science Foundation under grants IIS-2312794, IIS- 2313130, OAC-2239622, Amazon-Virginia Tech Initiative in Efficient and Robust Machine Learning, and the Commonwealth Cyber Initiative. We thank Meng Ding, Chong Xiang, Chendi Wang for their helpful feedback on the preliminary version of this work.

## References

Jordan T Ash, Chicheng Zhang, Akshay Krishnamurthy, John Langford, and Alekh Agarwal. Deep batch active learning by diverse, uncertain gradient lower bounds. In *International Conference on* Learning Representations, 2019.

Juhan Bae, Nathan Ng, Alston Lo, Marzyeh Ghassemi, and Roger B Grosse. If influence functions are the answer, then what is the question? *Advances in Neural Information Processing Systems*, 35:17953–17967, 2022.

S Basu, P Pope, and S Feizi. Influence functions in deep learning are fragile. In *International* Conference on Learning Representations (ICLR), 2021.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In *International* Conference on Machine Learning, pp. 2397–2430. PMLR, 2023.

Zhiqi Bu, Jialin Mao, and Shiyun Xu. Scalable and efficient training of large convolutional neural networks with differential privacy. *Advances in Neural Information Processing Systems*, 35: 38305–38318, 2022.

Zhiqi Bu, Yu-Xiang Wang, Sheng Zha, and George Karypis. Differentially private optimization on large model at small cost. In *International Conference on Machine Learning*, pp. 3192–3218. PMLR, 2023.

Mark Alexander Burgess and Archie C Chapman. Approximating the shapley value using stratified empirical bernstein sampling. In *IJCAI*, pp. 73–81, 2021.

Sang Keun Choe, Hwijeen Ahn, Juhan Bae, Kewen Zhao, Minsoo Kang, Youngseog Chung, Adithya Pratapa, Willie Neiswanger, Emma Strubell, Teruko Mitamura, et al. What is your data worth to gpt? llm-scale data valuation with influence functions. *arXiv preprint arXiv:2405.13954*, 2024.

R Dennis Cook and Sanford Weisberg. Characterizations of an empirical influence function for detecting influential cases in regression. *Technometrics*, 22(4):495–508, 1980.

Ian Covert, Chanwoo Kim, Su-In Lee, James Zou, and Tatsunori Hashimoto. Stochastic amortization:
A unified approach to accelerate feature and data attribution. *arXiv preprint arXiv:2401.15866*, 2024.

Junwei Deng and Jiaqi Ma. Computational copyright: Towards a royalty model for ai music generation platforms. *arXiv preprint arXiv:2312.06646*, 2023.

Jasjeet Dhaliwal and Saurabh Shintre. Gradient similarity: An explainable approach to detect adversarial attacks against deep learning. *arXiv preprint arXiv:1806.10707*, 2018.

Vitaly Feldman and Chiyuan Zhang. What neural networks memorize and why: Discovering the long tail via influence estimation. *Advances in Neural Information Processing Systems*, 33:2881–2891, 2020.

Stanislav Fort and Surya Ganguli. Emergent properties of the local geometry of neural loss landscapes.

arXiv preprint arXiv:1910.05929, 2019.

Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. The pile: An 800gb dataset of diverse text for language modeling. *arXiv preprint arXiv:2101.00027*, 2020.

Amirata Ghorbani and James Zou. Data shapley: Equitable valuation of data for machine learning.

In *International Conference on Machine Learning*, pp. 2242–2251. PMLR, 2019.

Amirata Ghorbani, Michael Kim, and James Zou. A distributional framework for data valuation. In International Conference on Machine Learning, pp. 3535–3544. PMLR, 2020.

Roger Grosse, Juhan Bae, Cem Anil, Nelson Elhage, Alex Tamkin, Amirhossein Tajdini, Benoit Steiner, Dustin Li, Esin Durmus, Ethan Perez, et al. Studying large language model generalization with influence functions. *arXiv preprint arXiv:2308.03296*, 2023.

Michael M Grynbaum and Ryan Mac. The times sues openai and microsoft. *The New York Times*, pp.

B1–B1, 2023.

Melissa Heikkilä. This new tool could give artists an edge over ai.

https://www.technologyreview.com/2023/10/24/1082247/
this-new-tool-could-give-artists-an-edge-over-ai/, 2023.

Peter Henderson, Xuechen Li, Dan Jurafsky, Tatsunori Hashimoto, Mark A Lemley, and Percy Liang.

Foundation models and fair use. *arXiv preprint arXiv:2303.15715*, 2023.

Ferenc Illés and Péter Kerényi. Estimation of the shapley value by ergodic sampling. *arXiv preprint* arXiv:1906.05224, 2019.

Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, and Aleksander Madry. Datamodels: Predicting predictions from training data. *arXiv preprint arXiv:2202.00622*, 2022.

Ruoxi Jia, David Dao, Boxin Wang, Frances Ann Hubis, Nezihe Merve Gurel, Bo Li, Ce Zhang, Costas J Spanos, and Dawn Song. Efficient task-specific data valuation for nearest neighbor algorithms. *Proceedings of the VLDB Endowment*, 2019a.

Ruoxi Jia, David Dao, Boxin Wang, Frances Ann Hubis, Nick Hynes, Nezihe Merve Gürel, Bo Li, Ce Zhang, Dawn Song, and Costas J Spanos. Towards efficient data valuation based on the shapley value. In *The 22nd International Conference on Artificial Intelligence and Statistics*, pp.

1167–1176. PMLR, 2019b.

Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In International Conference on Machine Learning, pp. 1885–1894. PMLR, 2017.

Weiwei Kong and Andres Munoz Medina. A unified fast gradient clipping framework for dp-sgd.

Advances in Neural Information Processing Systems, 36, 2024.

Yongchan Kwon and James Zou. Beta shapley: a unified and noise-reduced data valuation framework for machine learning. In *International Conference on Artificial Intelligence and Statistics*, pp.

8780–8802. PMLR, 2022.

Yongchan Kwon, Manuel A Rivas, and James Zou. Efficient computation and analysis of distributional shapley values. In *International Conference on Artificial Intelligence and Statistics*, pp. 793–801. PMLR, 2021.

Jaewoo Lee and Daniel Kifer. Scaling up differentially private deep learning with fast per-example gradient clipping. *Proceedings on Privacy Enhancing Technologies*, 2021.

Weida Li and Yaoliang Yu. Faster approximation of probabilistic and distributional values via least squares. In *The Twelfth International Conference on Learning Representations*, 2023.

Weida Li and Yaoliang Yu. Robust data valuation with weighted banzhaf values. *Advances in Neural* Information Processing Systems, 36, 2024.

Xuechen Li, Florian Tramer, Percy Liang, and Tatsunori Hashimoto. Large language models can be strong differentially private learners. In *International Conference on Learning Representations*, 2021.

Jinkun Lin, Anqi Zhang, Mathias Lécuyer, Jinyang Li, Aurojit Panda, and Siddhartha Sen. Measuring the effect of training data on deep learning predictions via randomized experiments. In *International* Conference on Machine Learning, pp. 13468–13504. PMLR, 2022.

Rory Mitchell, Joshua Cooper, Eibe Frank, and Geoffrey Holmes. Sampling permutations for shapley value estimation. 2022.

Caitlin Mulligan and James Li. Generative ai's end run around copyright. *AI Snake Oil*, 2024. URL https://www.aisnakeoil.com/p/ generative-ais-end-run-around-copyright.

Elisa Nguyen, Minjoon Seo, and Seong Joon Oh. A bayesian approach to analysing training data attribution in deep learning. *Advances in Neural Information Processing Systems*, 36, 2023.

Ramin Okhrati and Aldo Lipani. A multilinear sampling algorithm to estimate shapley values. In 2020 25th International Conference on Pattern Recognition (ICPR), pp. 7992–7999. IEEE, 2021.

Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, and Aleksander Madry. Trak:
attributing model behavior at scale. In Proceedings of the 40th International Conference on Machine Learning, pp. 27074–27113, 2023.

Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. Estimating training data influence by tracing gradient descent. *Advances in Neural Information Processing Systems*, 33:
19920–19930, 2020.

Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of machine learning research*, 21(140):1–67, 2020.

Stephen Robertson, Hugo Zaragoza, et al. The probabilistic relevance framework: Bm25 and beyond.

Foundations and Trends® in Information Retrieval, 3(4):333–389, 2009.

Gaspar Rochette, Andre Manoel, and Eric W Tramel. Efficient per-example gradient computations in convolutional neural networks. In Workshop on Theory and Practice of Differential Privacy
(TPDP), 2020.

Andrea Schioppa. Gradient sketches for training data attribution and studying the loss landscape.

arXiv preprint arXiv:2402.03994, 2024.

Lloyd S Shapley. A value for n-person games. *Contributions to the Theory of Games*, 2(28):307–317, 1953.

Anders Søgaard et al. Revisiting methods for finding influential examples. arXiv preprint arXiv:2111.04683, 2021.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023.

Jiachen T Wang and Ruoxi Jia. Data banzhaf: A robust data valuation framework for machine learning. In *International Conference on Artificial Intelligence and Statistics*, pp. 6388–6421. PMLR, 2023a.

Jiachen T Wang and Ruoxi Jia. A note on" towards efficient data valuation based on the shapley value". *arXiv preprint arXiv:2302.11431*, 2023b.

Jiachen T Wang and Ruoxi Jia. A note on" efficient task-specific data valuation for nearest neighbor algorithms". *arXiv preprint arXiv:2304.04258*, 2023c.

Jiachen T Wang, Yuqing Zhu, Yu-Xiang Wang, Ruoxi Jia, and Prateek Mittal. Threshold knn-shapley:
A linear-time and privacy-friendly approach to data valuation. *arXiv preprint arXiv:2308.15709*, 2023.

Jiachen T Wang, Zhun Deng, Hiroaki Chiba-Okabe, Boaz Barak, and Weijie J Su. An economic solution to copyright challenges of generative ai. Technical report, 2024a.

Jiachen T Wang, Zhun Deng, Hiroaki Chiba-Okabe, Boaz Barak, Weijie J Su, et al. An economic solution to copyright challenges of generative ai. *arXiv preprint arXiv:2404.13964*, 2024b.

Jiachen T Wang, Prateek Mittal, and Ruoxi Jia. Efficient data shapley for weighted nearest neighbor algorithms. *arXiv preprint arXiv:2401.11103*, 2024c.

Jiachen T Wang, Tianji Yang, James Zou, Yongchan Kwon, and Ruoxi Jia. Rethinking data shapley for data selection tasks: Misleads and merits. *arXiv preprint arXiv:2405.03875*, 2024d.

Tianhao Wang, Johannes Rausch, Ce Zhang, Ruoxi Jia, and Dawn Song. A principled approach to data valuation for federated learning. In *Federated Learning*, pp. 153–167. Springer, 2020.

Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. Less:
Selecting influential data for targeted instruction tuning. In *Forty-first International Conference on* Machine Learning, 2024.

Ziao Yang, Han Yue, Jian Chen, and Hongfu Liu. On the inflation of knn-shapley value. arXiv preprint arXiv:2405.17489, 2024.

Chih-Kuan Yeh, Ankur Taly, Mukund Sundararajan, Frederick Liu, and Pradeep Ravikumar. First is better than last for training data influence. *arXiv preprint arXiv:2202.11844*, 2022.

Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn.

Gradient surgery for multi-task learning. *Advances in Neural Information Processing Systems*, 33: 5824–5836, 2020.

## A Extended Related Works A.1 Data Shapley Axioms

Data Shapley is one of the first principled approaches to data attribution being proposed Ghorbani & Zou (2019); Jia et al. (2019b). Data Shapley is based on the famous *Shapley value* (Shapley, 1953). In almost all of the literature, the Shapley value is being justified as the *unique* value notion satisfying the following four axioms:
1. **Null player:** if U(S ∪ {zi}) = U(S) for all S ⊆ D \ {zi}, then ϕzi(U) = 0. 2. **Symmetry:** if U(S ∪ {zi}) = U(S ∪ {zj}) for all S ⊆ D \ {zi, zj}, then ϕzi(U) = ϕzj(U).

3. **Linearity:** For utility functions U1, U2 and any α1, α2 ∈ R, ϕzi(α1U1 + α2U2) =
α1ϕzi(U1) + α2ϕzi(U2).

4. **Efficiency:** for every U,Pzi∈D ϕzi(U) = U(D).

In plain words, **null player** axiom means the Shapley value will assign zero score to data points with no contribution. **Symmetry** axiom requires equal scores assigned to equally impactful data points, ensuring fairness. **Efficiency** axiom requires the sum of contribution scores equal to the total utility, meaning the scores always represent a share of the total utility. **Linearity** axiom means the Shapley value supports additive decomposition across multiple utility functions, allowing for the calculation of contributions to the entire test set by summing the scores of individual test points.

## A.2 Data Shapley And Friends

Since its introduction in 2019 (Ghorbani & Zou, 2019; Jia et al., 2019b), Data Shapley has rapidly gained popularity as a principled solution for data attribution. Due to the computationally expensive nature of retraining-based Data Shapley, various Monte Carlo-based approximation algorithms have been developed (Jia et al., 2019b; Illés & Kerényi, 2019; Okhrati & Lipani, 2021; Burgess & Chapman, 2021; Mitchell et al., 2022; Lin et al., 2022; Wang & Jia, 2023b; Li & Yu, 2023; Covert et al., 2024),
these methods still necessitate extensive computational resources due to repeated model retraining, which is clearly impractical for modern-sized ML models. Many of its variants have been proposed. Kwon & Zou (2022) argues that the efficiency axiom is not necessary for many machine learning applications, and the framework of *semivalue* is derived by relaxing the efficiency axiom. Lin et al. (2022) provide an alternative justification for semivalue based on causal inference and randomized experiments. Based on the framework of semivalue, Kwon & Zou (2022) propose *Beta Shapley*, which is a collection of semivalues that enjoy certain mathematical convenience. Wang & Jia (2023a) propose *Data Banzhaf*, and show that the Banzhaf value, another famous solution concept from cooperative game theory, achieves more stable valuation results under stochastic learning algorithms. Li & Yu (2024) further improves the valuation stability by considering value notions outside the scope of semivalue. The classic leave-one-out error is also a semivalue, where the *influence function* (Cook & Weisberg, 1980; Koh & Liang, 2017; Grosse et al., 2023) is generally considered as its approximation. A
concurrent work (Choe et al., 2024) leverages a similar gradient decomposition technique as our paper to speed up influence function calculation. However, several works have pointed out the fragility of influence function for deep learning models (Basu et al., 2021; Søgaard et al., 2021; Bae et al., 2022), due to the strong assumptions of training convergence and strongly convexity of the loss function. Nguyen et al. (2023) takes a Bayesian view of data attribution and is able to evaluate the variance of LOO. Unlike Nguyen et al. (2023), our work explicitly incorporates specific training randomness (e.g., model initialization, batch ordering) into the utility function definition, providing attribution scores that reflect contributions to the exact training trajectory.

Another line of works focuses on improving the computational efficiency of Data Shapley by considering K nearest neighbor (KNN) as the surrogate learning algorithm for the original, potentially complicated deep learning models (Jia et al., 2019a; Wang & Jia, 2023c; Wang et al., 2023; 2024c; Yang et al., 2024). Ghorbani et al. (2020); Kwon et al. (2021); Li & Yu (2023) consider Distributional Shapley, a generalization of Data Shapley to data distribution. In federated learning setting, Wang et al. (2020) proposes a similar idea of computing the Shapley value for each federated learning round. Remark 3. Randomized Monte Carlo estimators can be inefficient and may produce unstable valuation results, potentially violating fairness axioms of the Shapley value (e.g., the Symmetry axiom) due to inherent approximation errors. In contrast, In-Run Data Shapley does not rely on Monte Carlo estimators. Instead, it computes the exact Shapley value for an approximated utility function via Taylor expansion. This approach adheres to the fairness axioms while ensuring the reliability and consistency of the data valuation results.

## A.3 Comparison With Tracin (Pruthi Et Al., 2020)

The form of first-order In-Run Data Shapley from Section 4.1 coincides with the TracIN-Ideal in Pruthi et al. (2020). This provides a new understanding of TracIN-Ideal as an approximation to In-Run Data Shapley. Both works face the technical challenge of requiring per-sample gradient computations during a single training run. Pruthi et al. (2020) proposes *TracIN-CP*, which mitigates the computational burden by examining only a subset of intermediate checkpoints during training. At each checkpoint, the individual gradients for the entire training set are computed, rather than for a sampled batch, under the assumption that each training example is visited exactly once between checkpoints. A recent work (Xia et al., 2024) leverages TracIN-CP, an approximation algorithm for TracIN-Ideal, for instruction-following data selection. This approach, however, may deviate significantly from the original TracIN-Ideal, with the final valuation results heavily dependent on the selected checkpoints. Furthermore, Pruthi et al. (2020)'s implementation is limited to the parameters of the last linear layer due to memory constraints, potentially biasing the measurement of data contribution. For instance, Yeh et al. (2022) suggests that the last layer of a neural network might exhibit a strong "cancellation effect," where the data influence of different examples have large, contradictory magnitudes. Additionally, Schioppa (2024) demonstrates that selecting different layers can distort data attribution scores. In contrast, this work introduces the "ghost dot-product" technique to efficiently compute the first-order In-Run Data Shapley (i.e., TracIN-Ideal) directly and accurately, without additional approximations.

## B Additional Discussion

In this section, we provide additional discussion about In-Run Data Shapley as well as its comparison with Retraining-based Data Shapley. Figure 4 gives a more detailed overview of our algorithm, and Figure 5 provides a visualized comparison between Retraining-based and In-Run Data Shapley.

## B.1 When Is In-Run Data Shapley Desirable For Data Attribution?

While Retraining-based Data Shapley has been widely adopted in the literature, it suffers from several critical issues that limit its practicality and effectiveness. In this section, we discuss these problems from four key aspects: computational efficiency, alignment with the purpose of data valuation, stability of the valuation results, and the choice of training hyperparameters.

(1) Computational burden. Retraining-based Data Shapley calculation is often computationally prohibitive, as it requires retraining the model on every possible subset of the original dataset, leading to a computational complexity that grows exponentially with the size of the dataset. Despite the development of various Monte Carlo-based approximation algorithms (Jia et al., 2019b; Illés & Kerényi, 2019; Okhrati & Lipani, 2021; Burgess & Chapman, 2021; Mitchell et al., 2022; Lin et al., 2022; Wang & Jia, 2023b; Li & Yu, 2023; Covert et al., 2024), these methods still necessitate extensive computational resources due to repeated model retraining, which is clearly impractical for modern-sized ML models. Another line of work attempts to use efficient proxy learning algorithms, such as K-nearest neighbors (KNN) (Jia et al., 2019a; Wang & Jia, 2023c; Wang et al., 2023; 2024c; Yang et al., 2024), to accelerate Data Shapley computation. However, it remains unclear how closely these cheaper proxy models approximate the original learning algorithm, and it is also uncertain how to interpret the derived Data Shapley scores in this context. (2) Retraining-based Data Shapley is unable to assess data contribution to a specific model. Crucially, Retraining-based Data Shapley is not designed to value data contribution towards a specific model. It attempts to quantify the average contribution of each training data point to models trained on different subsets of the data, rather than its contribution to the specific model trained on the full dataset. While one might interpret the former as an approximation of the latter, these two quantities can be quite different in practice, especially when the learning algorithm is randomized and sensitive to factors like random initialization and the order of the data points during training. More importantly, in most real-life scenarios, the primary interest lies in understanding the contribution of each data point to the specific model being trained and deployed. (3) Retraining-based Data Shapley produces unstable valuation results for stochastic training algorithms. Furthermore, when the training algorithm involves randomness, such as in the case of SGD with random mini-batch selection, the corresponding utility function becomes randomized. Prior work (Wang & Jia, 2023a) suggests that this randomness can introduce substantial noise into the estimated Shapley values, rendering them unreliable and unstable. This instability poses significant challenges for interpreting and using the resulting data valuations, as the scores may vary considerably across different runs of the algorithm, even on the same dataset. Consequently, this limits the practical applicability of Retraining-based Data Shapley when working with stochastic training algorithms, which are prevalent in modern machine learning. We note that similar vulnerabilities to learning stochasticity have been observed for LOO-based data influence scores (e.g., influence function (Koh & Liang, 2017)) in several works (Basu et al., 2021; Søgaard et al., 2021; Nguyen et al., 2023). (4) Training hyperparameter choices in Retraining-based Data Shapley are unclear. In machine learning, training hyperparameters (e.g., learning rate and batch size) typically need to be adjusted based on the size of the training dataset. This creates a *fundamental ambiguity* when computing Data Shapley values: when evaluating the utility U(S) for different data subsets S, should we use the same hyperparameters as those optimized for the full dataset, or should we adjust them based on the subset size? This choice can significantly impact the calculated values. In-Run Data Shapley addresses all these issues, making it more desirable for specific scenarios. Firstly, it is computationally efficient as it computes data values during the training run, avoiding the need for multiple retraining iterations. This makes it feasible for modern large-scale ML models. Secondly, it aligns better with the purpose of data valuation by assessing the contribution of each data point to the specific model being trained, providing more relevant insights for real-world applications. Lastly, In-Run Data Shapley offers deterministic valuation results even with stochastic training algorithms, as it incorporates the specific sequence and randomness of the training process into the utility function. Therefore, for scenarios where computational efficiency, model-specific contributions, and result stability are critical, In-Run Data Shapley is a more suitable choice. Remark 4 (In-run Data Shapley is a model-specific data attribution technique.). In the original Data Shapley literature, U(S) *typically defines utility as the final performance (e.g., accuracy, loss) of a* model trained on a data subset S. However, things become more complicated in the context of deep learning. A specific deep learning training run is an iterative process. Defining utility U(S) for a particular run makes it a sequence function *rather than the pure* set function required by the standard Shapley value framework. Extending Shapley axioms to sequence functions is not straightforward. Previous works often define U(S) as "expected model utility across all possible training runs" to circumvent this conceptual issue. In this work, we initiate the study of model-specific data attribution, which is crucial for applications like model diagnosis and behavior interpretation where we focus on a specific training run rather than expected utility across all possible runs. Data attribution should consider specific application scenario. From a broader perspective, the "optimal data attribution" notion heavily depends on the intended application. In the literature, data attribution techniques are being used for various purposes such as data quality assessment, data valuation, and interpreting model predictions. For data quality assessment, we require "algorithmlevel data attribution" *that measures a data point's general influence on the learning algorithm,*
independent of specific training random seeds. On the other hand, when interpreting model decisions, we need model-specific data attribution focused on the particular checkpoint we train. Systematically mapping which data attribution approaches best suit specific application scenarios remains an important direction for future research.

## B.2 When Is Retraining-Based Data Shapley Desirable For Data Attribution?

Retraining-based Data Shapley is still desirable in several scenarios. Firstly, it is more general and applicable to all learning algorithms, whereas In-Run Data Shapley is only applicable to iterative training algorithms. Secondly, retraining-based Shapley is useful when the goal is to understand the contribution of data points to the general learning process rather than to a specific model. Thirdly, because Retraining-based Data Shapley does not require modifying the internal code of the training algorithm, its estimation algorithm, typically Monte Carlo-based, is straightforward and clean to implement.

## B.3 Potential Extensions Of In-Run Data Shapley To Other Domains

While this work focuses on data attribution for SGD-trained machine learning models, we believe the core idea of In-Run Data Shapley—decomposing contribution analysis into per-iteration assessmentscan potentially be extended to other contexts and applications. Here, we discuss some directions. Data attribution for iterative learning algorithms that do not use gradient descent. Our framework could potentially extend to iterative learning algorithms that don't use gradient descent, such as k-means clustering or decision tree learning. Though these algorithms update models differently, they still proceed iteratively (e.g., k-means alternates between assignment and update steps; decision trees are built through recursive partitioning). For such algorithms, we could analyze how each data point influences these discrete update steps and aggregate these influences across iterations, similar to our approach with gradient-based training. Hyperparameter importance. The framework could potentially be adapted to evaluate hyperparameter contributions during training. One possibility is to set a "baseline" hyperparameter value and assess how choosing a different value impacts each training iteration compared to this baseline. For instance, when evaluating learning rate choices, we could measure how using a specific learning rate value affects model updates compared to using a baseline learning rate. For differentiable hyperparameters, we could leverage Taylor expansion to approximate this difference; for non-differentiable ones, zero-order methods could potentially be used. By accumulating these contributions across training iterations, we could understand hyperparameter impact without requiring multiple complete training runs. At a high level, this view unifies our treatment of training data and hyperparameters
- both can be seen as choices that influence each training iteration, where we aim to quantify their impact against baseline scenarios. While technical challenges remain in adapting our method, this direction presents an interesting opportunity for future research. Feature attribution (e.g., context-attribution for language models). Another possible extension we envision is feature attribution for machine learning models. Feature attribution aims at understanding how each part of the input (like individual words in a sentence or pixels in an image) influences a model's final prediction. When a model makes a prediction, the input features are processed through multiple layers, with each layer transforming the information before passing it to the next layer. Feature attribution aims to track how this information flows and transforms through the network to determine each input feature's contribution to the final output. We envision a *unified theoretical* framework connecting training data attribution and feature attribution, drawing parallels between how information flows during training (through gradient updates) and during prediction (through layer operations). This could lead to more efficient methods for explaining model behavior, particularly for complex architectures like transformers.