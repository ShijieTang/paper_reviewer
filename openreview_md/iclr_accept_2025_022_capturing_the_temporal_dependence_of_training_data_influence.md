# Capturing The Temporal Dependence Of Train- Ing Data Influence

Jiachen T. Wang⋆
Princeton University

| Dawn Song   | James Zou           | Prateek Mittal       | Ruoxi Jia⋆    |
|-------------|---------------------|----------------------|---------------|
| UC Berkeley | Stanford University | Princeton University | Virginia Tech |

## Abstract

Traditional data influence estimation methods, like influence function, assume that learning algorithms are permutation-invariant with respect to training data. However, modern training paradigms, especially for foundation models using stochastic algorithms and multi-stage curricula, are sensitive to data ordering, thus violating this assumption. This mismatch renders influence functions inadequate for answering a critical question in machine learning: How can we capture the dependence of data influence on the optimization trajectory during training? To address this gap, we formalize the concept of trajectory-specific leave-one-out (LOO) influence, which quantifies the impact of removing a data point from a specific iteration during training, accounting for the exact sequence of data encountered and the model's optimization trajectory. However, exactly evaluating the trajectory-specific LOO presents a significant computational challenge. To address this, we propose data value embedding, a novel technique enabling efficient approximation of trajectory-specific LOO. Specifically, we compute a training data embedding that encapsulates the cumulative interactions between data and the evolving model parameters. The LOO can then be efficiently approximated through a simple dot-product between the data value embedding and the gradient of the given test data. As data value embedding captures training data ordering, it offers valuable insights into model training dynamics. In particular, we uncover distinct phases of data influence, revealing that data points in the early and late stages of training exert a greater impact on the final model. These insights translate into actionable strategies for managing the computational overhead of data selection by strategically timing the selection process, potentially opening new avenues in data curation research.

## 1 Introduction

Data influence estimation aims to provide insights into the impact of specific data points on the model's predictive behaviors. Such understanding is crucial not only for model transparency and accountability (Koh & Liang, 2017) but also plays a significant role in addressing AI copyright debates (Deng & Ma, 2023; Wang et al., 2024) and facilitating fair compensation in data marketplaces (Tian et al., 2022). The majority of data influence estimation techniques focus on measuring the counterfactual impact of a training data point: *how would the model's behavior change if we removed* a specific training data point?

LOO Influence. This counterfactual impact is often characterized by the *Leave-One-Out* (LOO)
influence, which has a long history and is frequently utilized in various fields such as robust statistics
(Cook & Weisberg, 1980), generalization analysis (Bousquet & Elisseeff, 2002), and differential privacy (Dwork et al., 2006). Inheriting from this rich classical literature across various domains, the LOO influence in data influence studies is typically defined as LOO(z
∗; z
(val)) := ℓ(A(D), z(val)) −
⋆Correspondence to **Jiachen T. Wang** and **Ruoxi Jia** (tianhaowang@princeton.edu, ruoxijia@vt.edu).

1 ℓ(A(D \ {z
∗}), z(val)), i.e., the model's loss change on a validation data z
(val) when the training data point z
∗is removed from the training set D. Here, A is the learning algorithm. For ease of analysis, traditional literature usually assumes that the learning algorithm A is permutation-invariant with respect to the training set D, meaning that *the order of data points does not affect the learning* outcome (Bousquet & Elisseeff, 2002). This assumption holds for models with strongly convex loss functions trained to converge. Within this framework, researchers have developed efficient methods to approximate LOO. Influence function (Koh & Liang, 2017), which uses first-order Taylor expansion to estimate the LOO, emerging as the most prominent approach. Numerous follow-up works have further improved its scalability for large models and datasets (Guo et al., 2021; Schioppa et al., 2022; Grosse et al., 2023; Choe et al., 2024). However, modern training algorithms, particularly those used for foundation models, increasingly deviate from the permutation-invariant assumption. This deviation arises from both the non-convex nature of neural networks and the multi-stage training curricula that do not run to convergence. In particular, due to the immense size of datasets, large language models (LLMs) often undergo just one training epoch, meaning each data point is encountered only once during training. Consequently, training data order significantly shapes the influence of data points on the final model (Epifano et al., 2023; Nguyen et al., 2024). Due to their underlying assumption of permutation-invariance, the order-dependence of data influence in modern training paradigms is not accurately reflected by influence functions. For example, they assign identical influence scores to duplicate training points, regardless of their position in the training sequence. Therefore, in this work, we argue that designing a data influence estimation technique relevant to the modern ML context requires rethinking how the counterfactual impact should be defined. Towards that end, we formalize the concept of *trajectory-specific LOO*, which characterizes the loss change resulting from removing a data point from the specific iteration it is used during training. In contrast to the traditional LOO, trajectory-specific LOO explicitly accounts for the exact sequence of data encountered, considering the timing of a target training point being trained on. An accurate evaluation of trajectory-dependent LOO would enable us to answer many important questions that are impossible to address with influence functions. For instance, how does a data point's impact vary depending on its entry timing in the training process? How do later points affect the influence of earlier points? However, exactly evaluating the trajectory-specific LOO presents a significant computational challenge. To address this, we introduce **data value embedding**, a novel data influence estimation framework designed for approximating trajectory-specific LOO. Our approach achieves several nice properties at the same time: **(1) accounting for training dynamics** and reflecting how the data order impacts model training; **(2) scale efficiently** to the setting of foundation models, and is faster than the current most efficient implementation of influence function; **(3) enable real-time attribution** for any query without necessitating model retraining or prior access to validation data. Technical novelty. Our proposed *data value embedding* framework computes a compact representation for each data point that encapsulates the cumulative effect of subsequent training. The influence scores for any test instance can be approximated with a simple dot product operation between the test gradient and the data value embedding, enabling real-time computation of data influence scores. To improve the scalability of computing data influence embedding, we develop a suite of techniques for efficient computation and storage of data value embeddings. In particular, we introduce the influence checkpointing technique, which enables the parallel computation of data value embeddings at multiple checkpoints. This not only enhances computational efficiency but also allows tracking of how a fixed data point's value changes during the training process. Empirical insights. Through data value embedding, we obtain several novel empirical insights into the training dynamics of foundation models. We identified three distinct regimes of data influence (Figure 1 (a)): a very brief high-influence region at the start, a much longer low-influence basin, and a region in the later training stage with gradually increasing influence, resuming to a high level. We show that performing online data selection solely in the early and late high-influence regions (less than half of the training duration) can achieve performance improvements on par with selecting data throughout the entire process (Figure 1 (b)). Moreover, performing data selection (Fan et al., 2024) only in the first very brief high-influence region, lasting less than 4% of the training duration, can achieve ≈ 50% of the performance gain enabled by continuous selection. Since online data selection usually incurs significant computational costs, our findings suggest a viable way of managing this overhead by strategically timing the selection process. By focusing data selection efforts on these critical phases, we can substantially improve training efficiency without compromising model performance. These temporal insights can potentially embark on new avenues of research on budget-limited data curation.

## 2 Trajectory-Specific Leave-One-Out Influence

In this section, we formalize the definition of *trajectory-specific LOO* which was originally introduced in Hara et al. (2019) as 'SGD-influence'. Consider a data point z
∗that is included in the training process during the ts-th iteration. Let Bt denote the training batch at iteration t. In standard SGD, the model parameters are updated as θt+1 = θt − ηtPz∈Bt ∇ℓ(θt, z) for t = 0*, . . . , T* − 1, where ηt is the learning rate at iteration t. We are interested in the change in the validation loss ℓ(θT , z(val)) when the data point z
∗ ∈ Btsis removed from iteration ts. In this counterfactual scenario, the parameter updates proceed as θ
′ts+1 = θts − ηtsPz∈Bts \{z
∗} ∇ℓ(θts, z) and θ
′t+1 = θ
′t − ηtPz∈Bt ∇ℓ(θ
′t, z)
for t = ts + 1*, . . . , T* − 1.

Definition 1 (**Trajectory-Specific LOO** (Hara et al., 2019)). The trajectory-specific leave-one-out for data point z
∗ at iteration ts *with respect to validation point* z
(val)*is defined as*

$$T S L O Q^{(t_{s})}(z^{*};z^{(v a l)}):=\ell(\theta_{T}^{\prime},z^{(v a l)})-\ell(\theta_{T},z^{(v a l)})$$

Discussion. TSLOO quantifies the change in validation loss resulting from removing z
∗ during the specific training run determined by the sequence of mini-batches and random initialization. TSLOO
explicitly depends on the timing of when the data is used and models the interaction effects between data points. For instance, it can show how the introduction of a certain type of example (e.g., a challenging edge case) might amplify or diminish the influence of previously seen, related examples. Moreover, identical data points contributing at different stages of training can receive different value scores. A data point introduced early in training might have a significantly different impact compared to the same point introduced later, as the model state evolves. However, traditional methods like influence functions do not capture these temporal dynamics. The influence function is defined as IF(z
∗; z
(val)) := ∇θℓ(*θ, z*(val))
⊤H−1 θ ∇θℓ(*θ, z*∗) where Hθ is the Hessian with respect to the full training loss. Because IF depends solely on the final state of the model, it invariably assigns the same influence value to identical z
∗s, regardless of their position in the training sequence.

Related works (extended version in Appendix A). Data attribution methods primarily fall into two categories: LOO-based methods and Shapley value-based methods. While Shapley value-based methods (Ghorbani & Zou, 2019) offer elegant theoretical interpretation, they typically require expensive model retraining, which limits their practical applicability. As a result, LOO-based methods such as influence functions (Koh & Liang, 2017) have gained more attention due to their computational efficiency. However, many studies have demonstrated that influence functions can be highly unreliable when applied to deep learning models (Basu et al., 2020; Bae et al., 2022; Epifano et al., 2023). In this work, we argue that TSLOO provides a more appropriate attribution framework for deep learning, particularly in the context of foundation models. Various research communities have independently explored Taylor expansion-based technique (Section 3.1) for approximating TSLOO for different purposes (Hara et al., 2019; Zou et al., 2021; Evron et al., 2022; Wu et al., 2022; Luo et al., 2023; Wu et al., 2024; Ding et al., 2024). However, practical adoption has been hindered by computational demands. This work proposes a new method that overcomes the computational bottlenecks in approximating TSLOO for large-scale models.

## 3 Data Value Embedding

While trajectory-specific LOO offers clear benefits for understanding data influence in modern ML, its computation presents significant challenges. Exact computation is not feasible, as it would require removing a data point from a specific training iteration and re-initiating the entire training process. To address this challenge, we introduce the concept of *data value embedding*.

## 3.1 Preliminary: Unrolling The Effect Of A Training Data Point In Sgd

Recall that we denote the final model as θT and the counterfactual model as θ
′
T, which is obtained by removing z
∗from ts-th training iteration. We introduce an interpolation between θT
and θ
′
Tby defining θts+1(ε) := θts − ηts Pz∈Bts \{z
∗} ∇ℓ(θts, z) − ηts(1 − ε)∇ℓ(θts, z∗) and θk+1(ε) = θk(ε) − ηkPz∈Bk ∇ℓ(θk(ε), z) for subsequent iterations. Note that θT (0) = θT and θT (1) = θ
′
T. Analogous to influence function-based approaches, we approximate the change in validation loss using a first-order Taylor expansion around ε = 0: ℓ(θ
′T
, z(val)) − ℓ(θT , z(val)) ≈
∇ℓ(θT , z(val))
⊤ ∂θT (ε)
∂ε
ε=0
. Interestingly, the derivative ∂θT (ε)
∂ε
ε=0 satisfies a recursive relation detailed in Appendix C.1, and we can obtain a well-established approximation from the literature:

$$\ell(\theta^{\prime}_{T},z^{(val)})-\ell(\theta_{T},z^{(val)})\approx\eta_{t_{*}}\nabla\ell(\theta_{T},z^{(val)})^{\top}\left[\prod_{k=t_{*}+1}^{T-1}(\mathbf{I}-\eta_{k}\mathbf{H}_{k})\right]\nabla\ell(\theta_{t_{*}},z^{*}).\tag{1}$$

where Hk =Pz∈Bk ∇2ℓ(θk, z) is the Hessian and I is the identity matrix. In data attribution literature, this approximation in (1) first appears in Hara et al. (2019) and has also been utilized in Chen et al. (2021) and Bae et al. (2024). It estimates the influence of removing z
∗from the ts-th iteration on the validation loss ℓ(θT , z(val)) at the final iteration. The product term QT −1 k=ts+1(I − ηkHk)
encapsulates the cumulative effect of the original data point's removal as it propagates through the entire training process. Notably, similar product terms appear frequently in related domains, including continual learning and deep learning theory (Zou et al., 2021; Evron et al., 2022; Wu et al., 2022; 2024; Ding et al., 2024).

## 3.2 Data Value Embedding

Building on (1), we extract the test-data-independent components and define *"data value embedding"* for a training point z
∗ ∈ Btsas

$$\text{DVEMD}^{(t_{s})}(z^{\star}):=\eta_{*}\left[\prod_{k=t_{s}+1}^{T-1}(\mathbf{I}-\eta_{k}\mathbf{H}_{k})\right]\nabla\ell(\theta_{t_{s}},z^{\star})\tag{2}$$

This embedding encapsulates the cumulative effect of a training point across the entire learning trajectory. By precomputing and storing these data value embeddings during or after the training phase, we enable highly efficient computation of data influence scores. Specifically, for any given test point z
(val), the influence of a training point z
∗can be quickly determined by simply computing the dot product ∇ℓ(θT , z(val))
⊤DVEmb(ts)(z
∗). Vector dot products are among the most computationally efficient operations, especially when executed on modern GPU hardware, which is optimized for such parallelized vector operations. Precomputing the data value embeddings eliminates the need for costly retraining or the availability of test data in advance, making the computation of data influence nearly instantaneous. This is particularly advantageous in real-world scenarios such as data marketplaces, where rapid, on-demand data attribution is critical. Approximation Error Bound. In Appendix C.2, we derive a new theoretical analysis of the approximation error associated with the unrolled differentiation estimator for non-convex loss functions. We demonstrate that when the learning rate schedule satisfies ηt ∈ O(1/
√t) with the maximum learning rate scaling as O(1/
√T)—a common choice in the literature (Vaswani, 2017)—the approximation error remains uniformly bounded and is *independent* of the total number of training steps T. While the proof relies on certain assumptions to abstract the complexities of real-world implementation, the theoretical result still implies the method's applicability in practical model training.

## 4 Efficient Computation And Storage Of Data Value Embedding

While the data value embedding approach offers a promising solution for real-time data attribution that incorporates training-specific factors, its practical implementation faces significant computational and storage challenges. The computation of DVEmb is non-trivial, requiring per-sample gradient calculations and per-step Hessian computations. Moreover, each DVEmbt(z
∗) has the same dimensionality as the model parameters, making it infeasible to store individual embeddings for each training data point on the disk. To address these challenges, we develop a series of techniques that significantly enhance both the computational and storage efficiency of data value embedding.

## 4.1 Recursive Approximation Of Data Value Embedding Via Generalized Gauss-Newton Matrix

We show that data value embedding can be computed recursively, beginning from the final training iteration and working backward, when using the Generalized Gauss-Newton (GGN) approximation for the Hessian matrix. This naturally gives rise to a backward computation algorithm for DVEmb(t). A widely-adopted approximation for the Hessian matrix Hk is the Generalized Gauss-Newton (GGN)
approximation Ht ≈Pz∈Bt ∇ℓ(θt, z)∇ℓ(θt, z)
⊤, particularly in the context of cross-entropy loss
(Martens, 2020). The GGN approximation is extensively used in various machine learning algorithms because it captures the essential curvature information of the loss landscape while remaining computationally feasible. For further details, see Appendix C.4. Under this approximation to Ht, the following shows that we can compute DVEmb(ts)(z
∗) for any z
∗ ∈ Bts if the data value embeddings of data points from later training iterations (i.e., DVEmb(t)(z) for t ≥ ts + 1) is available. Theorem 2. Given generalized Gauss-Newton approximation Ht ≈Pz∈Bt ∇ℓ(θt, z)∇ℓ(θt, z)
⊤,
we have

$${\tt\#}\,n a t e$$  $${\tt\#}\,D V E m{\tt D}^{(t_{s})}(z^{\star})=\eta_{t_{s}}\nabla\ell(\theta_{t_{s}},z^{\star})-\eta_{t_{s}}\sum_{t=t_{s}+1}^{T-1}\left(\sum_{z\in B_{t}}\left(\nabla\ell(\theta_{t},z)^{\top}\nabla\ell(\theta_{t_{s}},z^{\star})\right)\,D V E m{\tt D}^{(t)}(z)\right)$$
!
The proof is deferred to Appendix C.3. Interpretation. Theorem 2 provides crucial insights into the interactions between training data points throughout the model training process. When two points z
∗and z are similar, their gradient similarity term ∇ℓ(θt, z)
⊤∇ℓ(θts, z∗) increases, indicating stronger interaction between these points. To illustrate this phenomenon, consider training a language model where an early data point z
∗contains content about "quantum computing". The influence of z
∗ on the final model varies depending on the subsequent training data: if multiple similar "quantum computing" data points appear in later iterations, z
∗'s influence on the final model diminishes, as these later examples could teach similar concepts to the model. Conversely, if z
∗remains one of the few "quantum computing" examples throughout training, it maintains a stronger influence on the final model. Overview of the remaining sections. Theorem 2 suggests the possibility of a backpropagation algorithm for computing data value embeddings, contingent on the availability of per-sample gradient vectors for all training data. To make this approach practical for large-scale applications, we address two key challenges in the following sections: (1) Efficient computation and storage of per-sample gradient vectors for all training data (Section 4.2). (2) Efficient computation (Sections 4.3) and parallelization (Section 4.4) of data value embeddings using Theorem 2. Additionally, we discuss practical extensions and considerations for real-world scenarios (Appendix C.10). 4.2 STEP 1: STORE PER-SAMPLE TRAINING GRADIENT INFORMATION AT EACH ITERATION During model training, we additionally store the *per-sample* gradient for each data point in the training batch. However, this approach presents significant computational and storage challenges: (1) Storage: Let p denote the number of model parameters. Each gradient vector has dimension p, requiring O(*T Bp*) disk space, where B = |Bt| is the batch size. This effectively corresponds to storing millions of model-size vectors. **(2) Efficiency:** Computing per-sample gradients necessitates separate backpropagation for each z ∈ Bt, increasing computational cost by a factor of B.

Avoiding per-sample gradient computation & full gradient storage (detailed in Appendix C.5).

To mitigate both issues, we leverage a gradient decomposition and take advantage of the computations already performed during backpropagation (Wang et al., 2025; Choe et al., 2024). By expressing gradients as the outer product of activations and output derivatives, only a single backpropagation on the aggregated loss is required to compute per-sample gradients, preserving the usual training speed. Additionally, instead of storing the full gradient vectors, we store the decomposed components, potentially reducing the storage requirement to O(T B√p) for non-sequential data.

Random projections for large models. For large-scale foundation models with billions of parameters, we apply random projections to further compress the stored gradient information. Using projection matrices, we project the activations and output derivatives to a lower-dimensional space. This approach significantly reduces storage needs to O(*T Bp*˜), where p˜ is the projected dimension, while still capturing essential gradient geometric information. We acknowledge that deriving a theoretical multiplicative guarantee here is challenging, given that the data value embedding itself is a linear combination that could be zero. However, our ablation study in Appendix E.5 demonstrates that our approach is relatively more robust compared to influence functions across different projection dimensions. These results provide strong evidence of the robustness of our method in practice, and we leave the theoretical guarantee as future work.

## 4.3 Step 2: Backpropagating Data Value Embedding

Having established the method for storing projected gradient vectors, we now proceed to describe the backward computation algorithm for data value embeddings. For ease of presentation, we continue to use full gradient vector notation. However, in practical implementations, we use the projected gradient vectors for efficient storage. That is, ∇θℓ ∈ R
p˜in the subsequent contents.

According to Theorem 2, an equivalent expression for DVEmb(ts)(z
∗) is given by DVEmb(ts)(z
∗) = ηts∇ℓ(θts, z∗) − ηts∇ℓ(θts, z∗)M(ts)
where M(ts):= PT −1 t=ts+1 Pz∈Bt DVEmb(t)(z)∇ℓ(θt, z)
⊤. At a high level, our algorithm computes DVEmb(ts)(z
∗) for each ts from T − 1 down to 0, while maintaining a running matrix M(ts) ∈ R
p˜×p˜throughout the backpropagation process for algorithm efficiency.

Backward algorithm from the final iteration. We initialize M(T −1) = 0 as the data value embedding coincides with the training gradient for the last iteration. For ts = T − 1*, . . . ,* 0, we recursively compute: (1) The data value embedding for each z
∗ ∈ Bts: DVEmb(ts)(z
∗) =
ηts∇ℓ(θts, z∗) − ηtsM(ts)∇ℓ(θts, z∗), and (2) Update the weighting matrix after computing all embeddings for the current iteration: M(ts−1) = M(ts) +Pz
∗∈Bts DVEmb(ts)(z
∗)∇ℓ(θts
, z∗)
⊤. A
detailed algorithm pseudocode can be found in Algorithm 1. Computing data value embedding on a per-layer basis. Moreover, by adopting an assumption similar to that in EK-FAC regarding the independence of gradients across different layers, we can compute data value embeddings on a per-layer basis. This approach significantly reduces the computational and memory costs. The assumption of layer-wise independence is common in the literature on influence functions (Grosse et al., 2023), as it enables tractable analysis and efficient algorithms for deep neural networks. While this approximation neglects cross-layer gradient correlations, it is often justified because intra-layer interactions tend to dominate in practice. Treating layers independently thus strikes a favorable balance between computational feasibility and approximation accuracy. Complexity analysis. (1) Computational & Memory: The primary computational cost of our algorithm stems from matrix multiplications and additions in updating data value embeddings and the weighting matrix, resulting in O(BTp˜
2) floating-point operations (flops). However, if we compute the data value embedding per layer, flops improve to O(BTp˜
2/L) where L is the number of layers.

The update of the running matrix M(ts)requires O(Bp˜
2/L2) memory. In comparison, regular model training requires O(BT p) flops and O(p) memory, where p is the number of model parameters. Consequently, Algorithm 1 incurs significantly lower costs compared to regular training. We further note that the influence function method requires computing the per-sample gradient for each training data point on the final model, which is effectively equivalent to one epoch of training. As a result, both the memory requirements and flops for the influence function method are at least equivalent to those of model training, which are much larger than our algorithm's requirements. **(2) Storage:**
Each DVEmb(t)(z
∗) has dimension O(˜p), resulting in a total storage requirement of O(BTp˜) for data value embeddings across all training points. While this can be substantial, disk storage is relatively inexpensive in modern computing environments. Moreover, the reduced dimensionality achieved through projection significantly mitigates the storage burden compared to storing full-dimensional embeddings. A summary of the complexity comparison with the most efficient implementation of the influence function (Choe et al., 2024) is provided in Table 2 in Appendix C.9.

## 4.4 Parallelized Extension For Influence Embedding Computation (Overview)

The backpropagation algorithm introduced in Section 4.3 operates with a runtime complexity of O(T), as it sequentially computes DVEmb(ts)for ts = T − 1*, . . . ,* 0. While being significantly more efficient than the influence function, which requires re-computing all training gradients on the final model (see Section 5.2 and Table 2), it can still be costly for long training periods. Here, we introduce influence checkpointing, a parallelized extension for Algorithm 1. Influence Checkpointing. We reduce computational costs by allowing concurrent computation of data value embeddings at multiple checkpoints during training. By selecting K evenly spaced training steps, we can efficiently compute data value embeddings for each *intermediate checkpoint* in parallel. By carefully computing and storing necessary results, we can efficiently reconstruct the data value embedding for the final model. This reduces the overall computational cost by K times. The detailed algorithm description, pseudocode, and complexity analysis are deferred to Appendix C.7.

Data Value Dynamics During Training. In addition to its computational benefits, the influence checkpointing algorithm enables a powerful capability: tracking the evolution of data influences throughout the entire model training process. If the intermediate checkpoints θt1
, . . . , θtK−1 was saved—a common practice in foundation model pretraining—we can analyze how the influence of a fixed data point changes on different intermediate checkpoints. As a result, we gain a more fine-grained and dynamic view of how the influence of a fixed data point propagates to the subsequent training steps, providing deeper insights into the model's learning behavior over time. This capability opens up new avenues for understanding and optimizing machine learning model training.

## 5 Experiments

In this section, we evaluate the effectiveness of our proposed data value embedding method. First, we assess its fidelity in accurately reflecting data importance using small-scale experimental setups (Section 5.1), as well as its computational efficiency (Section 5.2). We then apply data value embedding to analyze the training dynamics during foundation model pretraining (Section 5.3 and Appendix E.4). The implementation details and additional results are deferred to Appendix E.

## 5.1 Fidelity Evaluation

To validate the effectiveness of our proposed data value embedding algorithm, we assess its accuracy in approximating TSLOO scores. Additionally, in Appendix E.2.1, we compare to a variety of data attribution baselines on the standard benchmarks of mislabel data detection and data selection. Computing ground-truth LOO requires retraining the model multiple times, each time excluding a single data point while keeping all other training specifics, such as batch order, unchanged. Given the computational intensity, we conduct our experiments on the MNIST (LeCun et al., 1989) using a small MLP trained with standard SGD. We consider two settings: **(1) Single epoch removal**, where a data point is excluded from training during a single epoch but still in other training epochs. Here, we remove the data point from the last epoch. **(2) All-epoch removal**, where a data point is excluded in all epochs. In this case, the approximation provided by data value embedding is obtained by summing the data value embeddings of the data point from all epochs, as discussed in Appendix C.10. Figure 3 shows that data value embedding has a high Spearman correlation with the ground-truth LOO. This superior performance is consistent across both settings. We note that the influence function scores remain constant for both settings, as influence functions do not account for specific training runs and cannot differentiate between single- and multi-epoch removals. Moreover, influence function exhibits a very weak correlation with LOO, a phenomenon that has been reported in many literature (Søgaard et al., 2021; Basu et al., 2020; Bae et al., 2022; Epifano et al., 2023).

## 5.2 Computational Efficiency

In this section, we compare the storage, memory, and computational efficiency of data value embedding with LoGRA (Choe et al., 2024), the most efficient implementation of the influence function so far. LoGRA first computes per-sample training gradients on the final model for all training data points z
∗ ∈ D, where D represents the dataset. Like our algorithm, LoGRA also uses random projection and stores the *projected* Hessian-adjusted gradient H−1 T 
∇ℓ(θT , z∗) to the disk, and the influence function can be computed via dot-product with test data gradient.

Table 1 shows the result of computing data influence for Pythia-410M trained on 1% of the Pile dataset. Both algorithms first compute and store Hessian-adjusted gradients/data value embedding, and then compute the data influence with respect to any given test point. As we can see, LoGRA and data value embedding have similar disk storage requirements, as both approaches save vectors of dimension p˜ for each data point. For peak GPU memory in the storage step, LoGRA requires recomputing gradients for all training data on the final model θT , which is effectively equivalent to one epoch of model training. In contrast, the data value embedding computation algorithm operates only on projected vectors, which takes much less GPU memory (0.84 vs 63.6GB). Consequently, the computational efficiency for computing data value embeddings is also much higher (over 15× faster). When computing data influence, since both approaches simply take the dot product between test data's (projected) gradient and H−1 T 
∇ℓ(θT , z∗) or DVEmb(t)(z
∗) or data value embedding, the GPU memory usage and efficiency are the same.

Storing H−1

T ∇ℓ(θT , z∗) **/ data value embedding Compute Influence (dot-product)**

Storage Peak GPU Mem. Throughput Peak GPU Mem. Throughput

LoGRA 170GB 63.6GB 41.6 16.31GB 640

Data Value Embedding 171GB 64.6GB / 0.84GB* 667.52 16.31GB 640 Table 1: Memory and compute efficiency analysis for LoGRA (Choe et al., 2024) and data value embedding. Throughput is measured as the number of data points per second for storing and influence computation. The experiment is conducted on one A100 GPU with 80GB VRAM. The projection dimension is set to 1024. *Since data value embedding technique contains two different steps in storing relevant information for data attribution (storing gradient during training & compute and store data value embedding after training), we include the peak GPU memory usage for both steps.

## 5.3 Analyzing Training Dynamics Of Foundation Models

In this section, we showcase data value embedding as a powerful tool for analyzing the training dynamics of foundation model pretraining with Pythia-410M trained on 1% of Pile dataset as an example. Results for additional datasets/models and the analysis for fine-tuning are in Appendix E.3. Value of training data from different stages in LLM pretraining. We first visualize the distribution of data influence scores on the final model across different training batches. For a fair comparison, we normalize the influence scores for each batch by their learning rate. Figure 1 (a) illustrates the results for training Pythia-410M on the Pile dataset. As we can see, the data influence on the final model can be categorized into three distinct regimes: **(1) High-impact Warmup Phase:** This phase occurs during the very early training stage and is characterized by exceptionally high data influence scores. It corresponds to a brief window at the onset of training where the loss reduces rapidly. (2) Low-impact Basin: This regime spans the early-to-middle training stage, where data influence scores are significantly lower. This period coincides with a slowdown in the rate of loss decrease, transitioning into a phase of relative stability. **(3) Gradual Ascent:** In this phase, we observe that the later a data point participates in the training, the higher its influence score becomes. Explanation: (1) Parameter initialization and warmup training are important for final model performance. During the very early stages of training, the gradient norms are large, which leads to significant parameter updates. Furthermore, the subsequent gradients' magnitude decrease rapidly, causing data points from the High-impact Warmup Phase to maintain substantial influence throughout the training process, even as their immediate impact diminishes over time. Figure 4 visualizes this phenomenon. The purple curve shows that training data points from the Highimpact Warmup Phase, while experiencing large drops in influence as training progresses, still maintain higher influence than later data points. This observation aligns with the well-known effect that model initialization and/or warm-up training plays a crucial role in training performance (He et al., 2015; Hanin & Rolnick, 2018), effectively initializing model parameters and gradually preparing the model for more complex learning tasks. (2) Influence saturation from future data. As training progresses into a smoother loss regime, the gradient norms become relatively stable and decrease slowly.

This makes the influence decay from subsequent training much more significant for these data points compared to those from the High-Impact Warmup Phase. Since earlier data points experience more future training iterations, their influence decreases more over time. The red curve in Figure 4 demon-
Figure 4: Evolution of influence scores across training checkpoints. The x-axis shows training iterations, and the y-axis shows the average influence of training examples on each checkpoint. Examples are grouped according to the iterations they are being trained on.

strates this trend, showing influence scores for these points gradually decreasing during training and eventually falling below those of later training data points. One might initially think this phenomenon is connected to catastrophic forgetting, where the model appears to "forget" the influence of data from earlier training phases as it progresses. However, we note that a data point's influence score decreases the most when future data points are similar to it, which is different from catastrophic forgetting. Intuitively, if future points are identical, the presence of the earlier data point in training becomes less relevant to the model's behavior. A more detailed explanation is deferred to Appendix E.3. Implications for data selection strategies. These observations suggest that for pretraining, data selection is most critical during the very early and later stages of training. To validate this insight, we train Pythia-410M on Pile with different online data selection strategies, as shown in Figure 1 (b). Specifically, we use an online data selection strategy (adapted from Fan et al. (2024)) that forms each training batch by selecting data points whose gradients align well with those from a validation batch sampled from Pile (see Appendix E.3.2 for details). This selection process requires computing gradient similarities, introducing significant overhead at each iteration where it is applied. Therefore, identifying the most critical training phases for applying this selection process becomes crucial for computational efficiency. Remarkably, Figure 1 (b) demonstrates that the performance of a strategy where we only perform data selection in the first 2000 iterations and after 20000 iterations closely matches the performance when data selection is performed in all iterations. Moreover, it reduces computational costs by more than 5 times. This corroborates our practical insights for designing efficient data selection strategies in LLM pretraining: by focusing data selection efforts on the critical early and late stages of training, we can potentially achieve optimal model performance while significantly reducing computational overhead.

## 6 Conclusion And Limitations

In this paper, we introduced Data Value Embedding, a novel approach to data attribution tailored for foundation models. Our method addresses critical limitations of existing techniques by capturing the temporal dynamics of training and enabling real-time attribution without the need for model retraining. The experiments demonstrate the efficacy of data value embedding in providing accurate and efficient data influence scores and unveiling unique insights into the training dynamics of foundation models.

SGD as a proxy for Adam. The data value embedding in (2) is specifically tailored for SGD. It is not directly extendable to other popular optimizers like Adam due to their normalization terms. Nonetheless, using SGD as a proxy for Adam allows for efficient data influence estimation, which is the approach that is usually adopted in practice and has proved to be effective in our experiment, providing a practical and effective solution for the current scope of our work. While using as a proxy for Adam has proved to be effective in our experiment, extending data value embedding to Adam and other optimizers remains an exciting direction for future research. Training curriculum design. Our findings on the varying influence of data points across training stages suggest the potential for designing optimal training curricula. Future work could explore leveraging data value embedding to design curricula that maximize learning efficiency. This could involve dynamically adjusting the presentation order and frequency of data points based on their predicted influence at different training stages.

## Acknowledgment

This work is supported in part by the National Science Foundation under grants IIS-2312794, IIS- 2313130, OAC-2239622, CNS-2131938, CNS-2424127, Amazon-Virginia Tech Initiative in Efficient and Robust Machine Learning, the Commonwealth Cyber Initiative, Cisco, OpenAI and Google. We thank Tong Wu, Meng Ding, Haizhou Shi, and Weida Li for their helpful feedback on the preliminary version of this work.

## References

Juhan Bae, Nathan Ng, Alston Lo, Marzyeh Ghassemi, and Roger B Grosse. If influence functions are the answer, then what is the question? *Advances in Neural Information Processing Systems*,
35:17953–17967, 2022.

Juhan Bae, Wu Lin, Jonathan Lorraine, and Roger B Grosse. Training data attribution via approximate unrolling. *Advances in Neural Information Processing Systems*, 37:66647–66686, 2024.

Elnaz Barshan, Marc-Etienne Brunet, and Gintare Karolina Dziugaite. Relatif: Identifying explanatory training samples via relative influence. In *International Conference on Artificial Intelligence and* Statistics, pp. 1899–1909. PMLR, 2020.

MS Bartlett. Approximate confidence intervals. *Biometrika*, 40(1/2):12–19, 1953. Samyadeep Basu, Philip Pope, and Soheil Feizi. Influence functions in deep learning are fragile.

arXiv preprint arXiv:2006.14651, 2020.

Olivier Bousquet and André Elisseeff. Stability and generalization. *The Journal of Machine Learning* Research, 2:499–526, 2002.

Hongge Chen, Si Si, Yang Li, Ciprian Chelba, Sanjiv Kumar, Duane Boning, and Cho-Jui Hsieh.

Multi-stage influence function. *Advances in Neural Information Processing Systems*, 33:12732–
12742, 2020.

Yuanyuan Chen, Boyang Li, Han Yu, Pengcheng Wu, and Chunyan Miao. Hydra: Hypergradient data relevance analysis for interpreting deep neural networks. In *Proceedings of the AAAI Conference* on Artificial Intelligence, volume 35, pp. 7081–7089, 2021.

Sang Keun Choe, Hwijeen Ahn, Juhan Bae, Kewen Zhao, Minsoo Kang, Youngseog Chung, Adithya Pratapa, Willie Neiswanger, Emma Strubell, Teruko Mitamura, et al. What is your data worth to gpt? llm-scale data valuation with influence functions. *arXiv preprint arXiv:2405.13954*, 2024.

R Dennis Cook and Sanford Weisberg. Characterizations of an empirical influence function for detecting influential cases in regression. *Technometrics*, 22(4):495–508, 1980.

Junwei Deng and Jiaqi Ma. Computational copyright: Towards a royalty model for ai music generation platforms. *arXiv preprint arXiv:2312.06646*, 2023.

Junwei Deng, Ting-Wei Li, Shichang Zhang, and Jiaqi Ma. Efficient ensembles improve training data attribution. *arXiv preprint arXiv:2405.17293*, 2024.

Meng Ding, Kaiyi Ji, Di Wang, and Jinhui Xu. Understanding forgetting in continual learning with linear regression. In *Forty-first International Conference on Machine Learning*, 2024.

Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In *Theory of cryptography conference*, pp. 265–284. Springer, 2006.

Jacob R Epifano, Ravi P Ramachandran, Aaron J Masino, and Ghulam Rasool. Revisiting the fragility of influence functions. *Neural Networks*, 162:581–588, 2023.

Itay Evron, Edward Moroshko, Rachel Ward, Nathan Srebro, and Daniel Soudry. How catastrophic can catastrophic forgetting be in linear regression? In *Conference on Learning Theory*, pp. 4028–4079. PMLR, 2022.

Simin Fan, Matteo Pagliardini, and Martin Jaggi. Doge: Domain reweighting with generalization estimation. In *Forty-first International Conference on Machine Learning*, 2024.

Vitaly Feldman and Chiyuan Zhang. What neural networks memorize and why: Discovering the long tail via influence estimation. *Advances in Neural Information Processing Systems*, 33:2881–2891, 2020.

Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. The pile: An 800gb dataset of diverse text for language modeling. *arXiv preprint arXiv:2101.00027*, 2020.

Amirata Ghorbani and James Zou. Data shapley: Equitable valuation of data for machine learning.

In *International Conference on Machine Learning*, pp. 2242–2251. PMLR, 2019.

Roger Grosse, Juhan Bae, Cem Anil, Nelson Elhage, Alex Tamkin, Amirhossein Tajdini, Benoit Steiner, Dustin Li, Esin Durmus, Ethan Perez, et al. Studying large language model generalization with influence functions. *arXiv preprint arXiv:2308.03296*, 2023.

Han Guo, Nazneen Rajani, Peter Hase, Mohit Bansal, and Caiming Xiong. Fastif: Scalable influence functions for efficient model interpretation and debugging. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 10333–10350, 2021.

Boris Hanin and David Rolnick. How to start training: The effect of initialization and architecture.

Advances in neural information processing systems, 31, 2018.

Satoshi Hara, Atsushi Nitanda, and Takanori Maehara. Data cleansing for models trained with sgd.

Advances in Neural Information Processing Systems, 32, 2019.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In *Proceedings of the IEEE international* conference on computer vision, pp. 1026–1034, 2015.

Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, and Aleksander Madry. Datamodels: Predicting predictions from training data. *arXiv preprint arXiv:2202.00622*, 2022.

Ruoxi Jia, David Dao, Boxin Wang, Frances Ann Hubis, Nezihe Merve Gurel, Bo Li, Ce Zhang, Costas J Spanos, and Dawn Song. Efficient task-specific data valuation for nearest neighbor algorithms. *Proceedings of the VLDB Endowment*, 2019.

Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In International Conference on Machine Learning, pp. 1885–1894. PMLR, 2017.

Frederik Kunstner, Philipp Hennig, and Lukas Balles. Limitations of the empirical fisher approximation for natural gradient descent. *Advances in neural information processing systems*, 32, 2019.

Yongchan Kwon, Eric Wu, Kevin Wu, and James Zou. Datainf: Efficiently estimating data influence in lora-tuned llms and diffusion models. In The Twelfth International Conference on Learning Representations, 2023.

Yann LeCun, Bernhard Boser, John Denker, Donnie Henderson, Richard Howard, Wayne Hubbard, and Lawrence Jackel. Handwritten digit recognition with a back-propagation network. Advances in neural information processing systems, 2, 1989.

Yuetian Luo, Zhimei Ren, and Rina Barber. Iterative approximate cross-validation. In *International* Conference on Machine Learning, pp. 23083–23102. PMLR, 2023.

James Martens. New insights and perspectives on the natural gradient method. Journal of Machine Learning Research, 21(146):1–76, 2020.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. *arXiv preprint arXiv:1609.07843*, 2016.

Elisa Nguyen, Minjoon Seo, and Seong Joon Oh. A bayesian approach to analysing training data attribution in deep learning. *Advances in Neural Information Processing Systems*, 36, 2024.

Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, and Aleksander Madry. Trak:
attributing model behavior at scale. In *Proceedings of the 40th International Conference on* Machine Learning, pp. 27074–27113, 2023.

Daryl Pregibon. Logistic regression diagnostics. *The annals of statistics*, 9(4):705–724, 1981. Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. Estimating training data influence by tracing gradient descent. *Advances in Neural Information Processing Systems*, 33:
19920–19930, 2020.

Andrea Schioppa, Polina Zablotskaia, David Vilar, and Artem Sokolov. Scaling up influence functions.

In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 36, pp. 8179–8186, 2022.

Nicol N Schraudolph. Fast curvature matrix-vector products for second-order gradient descent.

Neural computation, 14(7):1723–1738, 2002.

Anders Søgaard et al. Revisiting methods for finding influential examples. arXiv preprint arXiv:2111.04683, 2021.

Zhihua Tian, Jian Liu, Jingyu Li, Xinle Cao, Ruoxi Jia, and Kui Ren. Private data valuation and fair payment in data marketplaces. *arXiv preprint arXiv:2210.08723*, 2022.

A Vaswani. Attention is all you need. *Advances in Neural Information Processing Systems*, 2017. Jiachen T Wang and Ruoxi Jia. Data banzhaf: A robust data valuation framework for machine learning. In *International Conference on Artificial Intelligence and Statistics*, pp. 6388–6421. PMLR, 2023a.

Jiachen T Wang and Ruoxi Jia. A note on" efficient task-specific data valuation for nearest neighbor algorithms". *arXiv preprint arXiv:2304.04258*, 2023b.

Jiachen T Wang, Zhun Deng, Hiroaki Chiba-Okabe, Boaz Barak, and Weijie J Su. An economic solution to copyright challenges of generative ai. Technical report, 2024.

Jiachen T Wang, Prateek Mittal, Dawn Song, and Ruoxi Jia. Data shapley in one training run. In The Thirteenth International Conference on Learning Representations, 2025.

Jingfeng Wu, Difan Zou, Vladimir Braverman, Quanquan Gu, and Sham Kakade. The power and limitation of pretraining-finetuning for linear regression under covariate shift. Advances in Neural Information Processing Systems, 35:33041–33053, 2022.

Jingfeng Wu, Difan Zou, Zixiang Chen, Vladimir Braverman, Quanquan Gu, and Peter Bartlett. How many pretraining tasks are needed for in-context learning of linear regression? In The Twelfth International Conference on Learning Representations, 2024.

Yizi Zhang, Jingyan Shen, Xiaoxue Xiong, and Yongchan Kwon. Timeinf: Time series data contribution via influence functions. *arXiv preprint arXiv:2407.15247*, 2024.

Eric R Ziegel. The elements of statistical learning, 2003. Difan Zou, Jingfeng Wu, Vladimir Braverman, Quanquan Gu, and Sham Kakade. Benign overfitting of constant-stepsize sgd for linear regression. In *Conference on Learning Theory*, pp. 4633–4635. PMLR, 2021.

## A Extended Related Works

Here, we provide a general overview of the literature that is relevant to the trajectory-specific LOO. We refer the readers to Appendix A of Wang et al. (2025) for the overview of the related works for Data Shapley and other data attribution techniques.

## A.1 Loo Influence Vs Loocv

It is important to distinguish our LOO influence measure from traditional Leave-One-Out Cross- Validation (LOOCV) (Ziegel, 2003). While both involve removing individual data points, they serve different purposes and yield different interpretations. LOOCV is a model evaluation technique that estimates generalization performance by averaging prediction errors on held-out examples, where smaller errors indicate better model performance. In contrast, LOO influence measures how removing a specific training point affects the model's behavior on validation data, quantifying each training example's importance to the learning process. While LOOCV requires training N separate models to evaluate generalization (where N is the dataset size), LOO influence focuses on understanding the counterfactual impact of individual training points on model behavior. This distinction is crucial as we aim to understand data importance rather than model performance.

## A.2 Influence Function And Friends

Influence function (Koh & Liang, 2017) has emerged as an important tool for interpreting and analyzing machine learning models. As the influence function requires computing the Hessian inverse, many subsequent works are focusing on improving the scalability of the influence function for large-scale models (Guo et al., 2021; Schioppa et al., 2022; Grosse et al., 2023). More recently, Kwon et al. (2023) developed an efficient influence function approximation algorithm that is suitable for LoRA fine-tuning, and Zhang et al. (2024) extends the influence function to time-series datasets. In a similar spirit to us, Chen et al. (2020) a multi-stage extension of influence function to trace a fine-tuned model's behavior back to the pretraining data. However, they changed the original loss function and added a regularization term to account for intermediate checkpoints. The most closely related to our work is Choe et al. (2024). Similar to us, they also make use of the low-rank gradient decomposition and random projection to enable efficient computation and storage of per-sample gradient. However, their approach still requires computing per-sample gradient vectors for all training data on the final model checkpoint, which is effectively equivalent to one model retraining and takes a significantly longer time than data value embedding. Influence function and Newton step. The influence function formula bears a striking resemblance to the Newton step in optimization, as both involve the product of an inverse Hessian and a gradient. As a result, much data attribution literature simply describes the influence function as "one Newton step". There is a subtle difference in their expressions and motivations. For large dataset, one Newton step approximates the influence function under certain regularity conditions. A single Newton step from the full model parameters toward the leave-one-out solution was proposed as an influence measure by Pregibon (1981) and has since been leveraged for data attribution (Park et al., 2023). The high-level argument is that while obtaining the exact leave-one-out model requires fully minimizing the leave-one-out loss, taking one Newton step from the optimal model θ
∗trained on the full dataset often provides sufficient information to distinguish the relative influence of different data points.

## B Limitations Of The Existing Data Attribution Techniques For Foundation Models B.1 Influence Function

Influence functions (Cook & Weisberg, 1980; Koh & Liang, 2017) are a classical technique from robust statistics, adapted for machine learning to measure how the removal of a single data point affects the performance of a trained model. Influence functions quantify the sensitivity of a model's predictions to specific data points, offering insights into the importance of individual training samples.

In the machine learning framework, they are particularly useful for diagnosing model behavior, understanding dataset quality, and identifying mislabeled or harmful data points. The core idea of influence functions is to approximate the effect of removing a data point from the training set without needing to retrain the model. Instead of actually excluding a point and retraining, influence functions leverage the model's final parameters and compute the impact of a point's removal based on the gradient and Hessian inverse of the loss function at the final model state. Formally, the influence of a training data point zi on the loss ℓ(*θ, z*(val)) at a validation point z
(val)is defined as:

$\downarrow$ . 

## If(Zi) := −∇Θℓ(Θ, Z(Val)) ⊤H−1∇Θℓ(Θ, Zi)

where θ is the final model parameter after training, H =
1 N
PN
i=1 
∇2θ ℓ(*θ, z*i) is the Hessian of the total training loss at θ, ∇θℓ(θ, z(val)) and ∇θℓ(*θ, z*i) are the gradients of the loss at the validation point and the training point, respectively. Limitation: Neglecting Training Phases and Unrealiable Approximation to LOO. A key limitation of influence function techniques is their exclusive focus on the final model parameters, thereby ignoring the intermediate dynamics of the training process. By assessing data contributions solely based on the final trained model, influence functions fail to capture how each data point influenced the model's updates throughout training. This narrow focus introduces inaccuracies, as it overlooks the cumulative effects of model fluctuations during the training iterations. Consequently, influence functions can be less accurate in evaluating data contributions, particularly in large-scale models where the training process plays a significant role. For instance, in modern training paradigms for large language models (LLMs), models are typically pretrained on a broad corpus and subsequently fine-tuned on specialized domains. Influence functions, however, cannot differentiate between the impacts of data points during pretraining and fine-tuning phases. Relying solely on the final model parameters after fine-tuning, they miss how pretraining data contributed to learning general language structures or how fine-tuning data adapted the model to specific domains. This inability to account for different training stages results in incomplete and often noisy estimates of data contributions, thereby reducing the precision of attribution in multi-stage training processes. Moreover, our analysis in Section D demonstrates that the influence function approximates the expected data influence across different training trajectories only under overly simplistic conditions, which are often violated in practice. These conditions, such as assuming identical intermediate model checkpoints and Hessian matrices, almost never hold in real-world training scenarios where model evolve significantly. This highlights the inadequacy of influence functions in accurately capturing data contributions, underscoring the necessity for more comprehensive data attribution methods that consider the entire training trajectory. Neglecting Training Phases Necessitates Unreasonable Assumptions and Often Require Model Retraining. Additionally, the focus on the final model necessitates assumptions of convergence and strong convexity to ensure reliable results. In many real-world settings, where models are non-convex and may not fully converge, these assumptions are often violated, leading to further inaccuracies in the data contribution estimates. As the influence function score is often found to be highly noisy in practice (Basu et al., 2020; Søgaard et al., 2021; Bae et al., 2022; Epifano et al., 2023), it typically necessitates multiple model retraining to produce reasonable results (Deng et al., 2024), which can undermine their original computational efficiency advantage.

## B.2 In-Run Data Shapley

In-Run Data Shapley (Wang et al., 2025) is a data attribution technique designed to evaluate the contribution of individual data points during a single training run of machine learning models. It builds on the traditional Data Shapley framework, which stems from cooperative game theory. The Shapley value, originally proposed by Lloyd Shapley in 1953, distributes total utility fairly among all contributing players based on their marginal contributions. Applying this concept to machine learning, Data Shapley attributes the contribution of each data point in a training dataset by assessing its influence on model performance. However, standard Data Shapley methods face limitations in scalability because they require numerous retraining iterations on different data subsets. These computational demands make them impractical for large-scale models such as foundation models. To address these challenges, In-Run Data Shapley was introduced as a scalable alternative that avoids the need for repeated retraining. Instead, it leverages the iterative nature of model training, specifically neural networks, where parameters are updated in small increments. By tracking gradient updates at each training step, In-Run Data Shapley calculates the contribution of individual data points toward the final model without retraining. It approximates the Shapley value using local utility functions tied to specific gradient updates and extends these to the full training process, capturing cumulative contributions. This method reduces the computational overhead to a level comparable with standard training runs while maintaining the theoretical fairness and interpretability of Shapley values. Limitation: Requirement of Validation Data in Advance. One of the key limitations of In-Run Data Shapley is its reliance on the availability of validation data prior to the start of training. The technique calculates data contribution by examining the impact of training points on model performance as measured against the validation set. Thus, access to this validation data throughout the training process is necessary to compute meaningful Shapley values at each iteration. This restriction can limit the applicability of In-Run Data Shapley in scenarios where validation data is not immediately available, such as in certain real-time learning environments or when the validation set is defined only after training. Potential workarounds, such as saving intermediate model checkpoints to calculate contributions post-training, add complexity to the process and might be unreliable.

## C Algorithm Details C.1 Derivation Details For Section 3.1

Suppose z
∗is a data point that participates in the training during the *first* iteration. Denote Bt as the training batch in the t-th iteration. For standard Stochastic Gradient Descent (SGD), we have:

  **1.** For standard stochastic Gradient Descent (SGD), we have:  $\theta_{k+1}=\theta_{k}-\eta_{k}\sum_{z\in\mathcal{B}_{k}}\nabla\ell(\theta_{k},z)$  is the learning rate at iteration $k$.  
for $k=0,\ldots,T-1$, wl. 
for k = 0*, . . . , T* − 1, where ηk is the learning rate at iteration k. For validation data z
(val), we aim to estimate the change in ℓ(θT , z(val)) by removing z
∗from the

first iteration. Specifically, we want to estimate ℓ(θ
′
T, z(val)) − ℓ(θT , z(val)) where:
$\theta^{\prime}_{1}=\theta_{0}-\eta_{0}\sum_{z\in\mathcal{B}_{0}\setminus\{z^{*}\}}\nabla\ell(\theta_{0},z)$  $\theta^{\prime}_{k+1}=\theta^{\prime}_{k}-\eta_{k}\sum_{z\in\mathcal{B}_{k}}\nabla\ell(\theta^{\prime}_{k},z)$
and
for k = 1*, . . . , T* − 1.

To approach this problem, we define an interpolation between θT and θ
′
T:
$$\theta_{1}(\varepsilon):=\theta_{0}-\eta_{0}\sum_{z\in\mathcal{B}_{0}\setminus\{z^{*}\}}\nabla\ell(\theta_{0},z)-\eta_{0}(1-\varepsilon)\nabla\ell(\theta_{0},z^{*})$$  where $\theta_{T}(\varepsilon)$ is defined accordingly. Note that $\theta_{T}(0)=\theta_{T}$ and $\theta_{T}(1)=\theta_{T}^{\prime}$.  
∇ℓ(θ0, z) − η0(1 − ε)∇ℓ(θ0, z∗) (6)
By taking the first-order Taylor expansion at ε = 0, we have:

ℓ(θ
′
  **Soln at $\varepsilon=0$, we have:**  $$z^{(val)})=\ell(\theta_{T}(1),z^{(val)})-\ell(\theta_{T}(0),z^{(val)})$$ $$\approx\left.\frac{\partial}{\partial\varepsilon}\ell(\theta_{T}(\varepsilon),z^{(val)})\right|_{\varepsilon=0}$$ $$=\nabla\ell(\theta_{T},z^{(val)})^{\top}\left.\frac{\partial\theta_{T}(\varepsilon)}{\partial\varepsilon}\right|_{\varepsilon=0}\tag{7}$$
$$(3)$$
$$(4)$$
$$(S)$$
$$(6)$$

Now, we derive ∂θT (ε) ∂ε ε=0 by observing the following recursive relation for all k ≥ 1: ∂θk+1(ε) ∂ε = ∂θk(ε) ∂ε − ηk X z∈Bk ∇2ℓ(θk(ε), z) ∂θk(ε) ∂ε (8) = ∂θk(ε) ∂ε (I − ηkHk(ε)) (9) where Hk(ε) = Pz∈Bk ∇2ℓ(θk(ε), z) is the Hessian and I is the identity matrix. Additionally, for
the first iteration where z
∗ participates, we have
∂ε = η0∇ℓ(θ0, z∗) (10)
Expanding the recursion and substituting it back into our original expression, we get:
$$\frac{\partial\theta_{1}(\varepsilon)}{\partial\varepsilon}=\eta_{0}\nabla\ell(\theta_{0},z^{*})$$
$$(\theta_{T}(1),z^{(v a l)})-\ell(\theta_{T}(0),z^{(v a l)})\approx\frac{\partial}{\partial\varepsilon}\ell(\theta_{T}(\varepsilon),z^{(v a l)})$$
∂ε ℓ(θT (ε), z(val))|ε=0
$\theta_{\varepsilon}$  $=\eta_{0}\nabla\ell(\theta_{T},z^{(val)})^{\top}\underbrace{\left[\prod_{k=1}^{T-1}(I-\eta_{k}\mathbf{H}_{k})\right]}_{\text{cumulative effect}}\nabla\ell(\theta_{0},z^{*})$
$$(10)$$
This final expression gives an estimate of the influence of removing z
∗from the first iteration on the loss on z
(val)at the final iteration. The term QT −1 k=1 (I − ηkHk) represents the cumulative effect of all training iterations on the initial influence. This product captures how the impact of the initial change propagates through the entire training process, accounting for the learning rate and the training data at each subsequent step.

## C.2 Error Guarantee For Unrolling-Based Approach

In this section, we derive the approximation error guarantee of the unrolling differentiation estimator

$$\Delta\theta_{-z^{*}}:=\left.\frac{\partial\theta_{T}(\varepsilon)}{\partial\varepsilon}\right|\varepsilon=0=\eta t_{s}\left[\prod_{k=t_{s}+1}^{T-1}(I-\eta_{k}\mathbf{H}_{k})\right]\nabla\ell(\theta_{t_{s}},z^{*}),$$

for non-convex loss functions. A very loose bound for ∥θT − θ
′T − ∆θ−z
∗ ∥ has been derived in Hara et al. (2019). Here, we improve the error bound by additionally considering the decay of the learning rate and the spectral norm of Hessian matrices as training progresses. Notably, we establish a uniform bound on the gap. Assume that ℓ(z; θ) is twice differentiable with respect to the parameter θ, and we train the model for T iterations. We make the following assumptions:
1. **Learning Rate Schedule:** The learning rate ηt at iteration t follows the schedule ηt =
η√max t where ηmax = √
C
T
for some constant C. **Justification:** The decaying learning rate schedule ηt =
η√max tis a common choice in neural network training in famous literature (Vaswani, 2017).

This schedule allows for larger step sizes during the initial phases of training, facilitating rapid convergence, while gradually reducing the step sizes to fine-tune the model parameters and ensure stability as training progresses. The max learning rate ηmax = O
√
1 T
ensures that the cumulative step sizes remain bounded over T iterations, which is crucial for deriving meaningful error bounds. This approach balances the trade-off between exploration and convergence, making it well-suited for training deep neural networks where maintaining stability is essential.

2. **Hessian Spectral Norm Decay:** There exists a constant Λ > 0 such that the Hessian matrices satisfy Ht ⪯ √
Λ
t I for all t ≥ 1. **Justification:** The assumption that the spectral norm of the Hessian matrices decays as Ht ⪯ √
Λ
t I is grounded in the observation that, as training progresses, the optimization landscape often becomes flatter around minima. This reduction in curvature implies that the Hessian's eigenvalues decrease, leading to smaller spectral norms. Such behavior is typical in many deep learning scenarios where initial training steps navigate regions of high curvature, followed by stabilization in flatter regions as the model converges. Additionally, this assumption aligns with empirical findings in deep learning literature (), where Hessian's spectral norm has been observed to decrease over time, thereby facilitating more stable and efficient convergence. By incorporating this decay, we account for the diminishing influence of curvature on parameter updates, which is critical for tightening the error bounds in our analysis.

Under these assumptions, we proceed to derive a uniform bound on the approximation error ∥θT −
θ
′T − ∆θ−z
∗ ∥. This bound provides theoretical guarantees for the effectiveness of the unrolling-based approach in estimating the influence of removing a training data point on the final model parameters. The derivation leverages the decaying learning rate and the diminishing spectral norm of the Hessian matrices to tighten the error bounds compared to previous work (Hara et al., 2019).

Theorem 3. Assume that ℓ(z; θ) is twice differentiable, that the Hessian ∇2θℓ(z; θ) is L-Lipschitz continuous with respect to θ, and that the gradient norm is bounded, i.e., ∥∇θℓ(z; θ)∥ ≤ G for all z and θ. Furthermore, assume that the learning rate ηt at iteration t follows the schedule ηt =
η√max t
,
where ηmax = √
C
T
for some constant C > 0*. Then, for the unrolling differentiation estimator* ∆θ−z
∗ ,
the approximation error satisfies

$$\|(\theta_{T}-\theta_{T}^{\prime})-\Delta\theta_{-z^{*}}\|\leq\frac{32}{3}G^{2}C^{3}L e^{C\Lambda}$$

Proof. By Cauchy's Mean Value Theorem, for each iteration s ∈ {ts*, . . . , T* − 1}, there exists r ∈ [0, 1] such that for θ
∗
s:= rθ′s + (1 − r)θs, we have

$$\sum_{z\in{\cal B}_{s}}(\nabla_{\theta}\ell(z;\theta_{s}^{\prime})-\nabla_{\theta}\ell(z;\theta_{s}))={\bf H}_{s}^{*}(\theta_{s}^{\prime}-\theta_{s}),$$
$$(11)$$

where H∗s:= Pz∈Bs ∇2θ ℓ(z; θ
∗
s). Define Zs := (I − ηsHs) and Z
∗
s:= (I − ηsH∗s). Then, we have θ
′
s+1 − θs+1 = Zs(θ
′
s − θs) + ηs(Hs − H∗s
)(θ
′
s − θs) = Zs(θ
′
s − θs) + Ds, where Ds := ηs(Hs −H∗s)(θ
′
s −θs). Recursively applying these equalities over s ∈ {ts*, . . . , T* −1},
we obtain

$$\theta_{T}^{\prime}-\theta_{T}=\Delta\theta_{-z^{*}}+\sum_{s=t_{s}}^{T-1}\prod_{k=s+1}^{T-1}Z_{k}D_{s}.$$

Hence, the approximation error is given by

$$\|(\theta_{T}-\theta_{T}^{\prime})-\Delta\theta_{-z^{*}}\|=\left\|\sum_{s=t_{s}}^{T-1}\prod_{k=s+1}^{T-1}Z_{k}D_{s}\right\|.$$

To bound this, we proceed as follows. Given the learning rate schedule ηt =
η√max t= √
C
T t , and the assumption that Ht ⪯ √
Λ
t I, we have

$\|Z_{k}\|=\|I-\eta_{k}\mathbf{H}_{k}\|\leq1+\eta_{k}\frac{\Lambda}{\sqrt{k}}=1+\frac{C\Lambda}{k\sqrt{T}}$.  $\geq t_{n}\geq1$, the term $\frac{C\Lambda}{\sqrt{T}}$ is small. Thus, we can both 
and $k\geq s\geq t_{s}\geq1$, the term $\frac{C\Lambda}{k\sqrt{T}}$ is small. Thus, we can bound that  $$\prod_{k=s+1}^{T-1}\|Z_{k}\|\leq\exp\left(\sum_{k=s+1}^{T-1}\frac{C\Lambda}{k\sqrt{T}}\right)\leq\exp\left(\frac{C\Lambda}{\sqrt{T}}\sum_{k=s+1}^{T-1}\frac{1}{k}\right).$$  rmonic series approximation,
is small. Thus, we can bound the product of the
norms as
$$\sum_{k=s+1}^{T-1}{\frac{1}{k}}\leq\ln\left({\frac{T}{s}}\right)\leq\ln(T).$$
Thus,
$$\prod_{k=s+1}^{T-1}\|Z_{k}\|\leq\exp\left({\frac{C\Lambda\ln T}{\sqrt{T}}}\right)\leq e^{C\Lambda}.$$
Therefore, we have
$$\left\|\sum_{s=t_{s}}^{T-1}\prod_{k=s+1}^{T-1}Z_{k}D_{s}\right\|\leq e^{C\Lambda}\sum_{s=t_{s}}^{T-1}\|D_{s}\|.$$
Next, we bound ∥Ds∥:

∥Ds∥ = ∥ηs(Hs − H∗s)(θ ′s − θs)∥ ≤ ηs∥Hs − H∗s∥ · ∥θ ′s − θs∥.
Since ∇2θ ℓ(z; θ) is L-Lipschitz continuous with respect to θ, we have

$$\|\mathbf{H}_{s}-\mathbf{H}_{s}^{*}\|\leq L\|\theta_{s}^{\prime}\|$$
s − θs∥.
Additionally, we have

$$\begin{array}{l}{{\|\theta_{s}^{\prime}-\theta_{s}\|\leq2\sum_{t=1}^{s}\eta_{t}G=2G\sum_{t=1}^{s}\frac{C}{\sqrt{T t}}\leq4G C\frac{\sqrt{s}}{\sqrt{T}},}}\end{array}$$  and $\sum_{t=1}^{s}\frac{1}{\sqrt{t}}\leq2\sqrt{s}$.  
where we used the bound Ps
Thus,

$$\|D_{s}\|\leq\eta_{s}L\cdot\left(4G C{\frac{\sqrt{s}}{\sqrt{T}}}\right)^{2}=\Gamma{\frac{\sqrt{s}}{T^{1.5}}}$$
where Γ = 16G2C
3L.
Substituting this bound into the sum, we obtain

$$\left\|\sum_{s=t_{s}}^{T-1}\prod_{k=s+1}^{T-1}Z_{k}D_{s}\right\|\leq e^{C\Lambda}\sum_{s=t_{s}}^{T-1}\Gamma\frac{\sqrt{s}}{T^{1.5}}.$$

We now evaluate the summation:

$$\sum_{s=t_{s}}^{T-1}{\frac{\sqrt{s}}{T^{1.5}}}\leq{\frac{1}{T^{1.5}}}\sum_{s=1}^{T}{\sqrt{s}}\leq{\frac{1}{T^{1.5}}}\cdot{\frac{2}{3}}T^{1.5}={\frac{2}{3}},$$

where we used the bound PT
s=1
√s ≤
2 3 T
1.5.

Therefore,
$$\left\|\sum_{s=t_{s}}^{T-1}\prod_{k=s+1}^{T-1}Z_{k}D_{s}\right\|\leq e^{C\Lambda}\Gamma\cdot{\frac{2}{3}}={\frac{32}{3}}G^{2}C^{3}L e^{C\Lambda}.$$
C.3 COMPUTING DATA VALUE EMBEDDING RECURSIVELY Theorem 4 P
(Restate for Theorem 2). *Given generalized Gauss-Newton approximation* Ht ≈
z∈Bt ∇ℓ(θt, z)∇ℓ(θt, z)
⊤*, we have*

$${\tt D V E m}^{(t)}(z^{\star})\approx\eta_{t}\nabla\ell(\theta_{t},z^{\star})-\eta_{t}\sum_{k=t+1}^{T-1}\left(\sum_{z\in B_{k}}\left(\nabla\ell(\theta_{k},z)^{\top}\nabla\ell(\theta_{t},z^{\star})\right){\tt D V E m}b^{(k)}(z)\right)$$

Proof.

DVEmb(t)(z
∗)
= ηt
"TY−1
k=t+1
(I − ηkHk)
#
∇ℓ(θt, z∗)
= ηt
"TY−1
k=t+2
(I − ηkHk)
#
(I − ηt+1Ht+1)∇ℓ(θt, z∗)
≈ ηt
"TY−1
k=t+2
(I − ηkHk)
# 
I − ηt+1 X
z∈Bt+1
∇ℓ(θt+1, z)ℓ(θt+1, z)
⊤

 ∇ℓ(θt, z∗)
= ηt
"TY−1
k=t+2
(I − ηkHk)
#∇ℓ(θt, z∗) − ηtX
z∈Bt+1
 
ηt+1 
"TY−1
k=t+2
(I − ηkHk)
#∇ℓ(θt+1, z)
!∇ℓ(θt+1, z)
⊤∇ℓ(θt, z∗)
= ηt
"TY−1
k=t+2
(I − ηkHk)
#∇ℓ(θt, z∗) − ηtX
z∈Bt+1
∇ℓ(θt+1, z)
⊤∇ℓ(θt, z∗)DVEmb(t+1)(z)
= ηt∇ℓ(θt, z∗) − ηt
T
X
−1
k=t+1
 X
z∈Bk
∇ℓ(θk, z)
⊤∇ℓ(θt, z∗)DVEmb(k)(z)
!
The transition from the penultimate to the final line involves generalizing the summation over Bt+1 to include all batches from t + 1 to T − 1, effectively unrolling the recursive computation. In other words, the "data value embedding" for data points in tth iteration can be approximated by its gradient subtracted by a linear combination of the data value embedding in the later iterations, where the weight of each embedding is determined by the gradient similarity ∇ℓ(θk, z)
⊤∇ℓ(θt, z∗).