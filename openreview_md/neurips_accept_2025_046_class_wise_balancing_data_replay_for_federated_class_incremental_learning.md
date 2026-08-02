# Class-Wise Balancing Data Replay For Federated Class-Incremental Learning

Zhuang Qi1, Ying-Peng Tang2, Lei Meng1,∗, Han Yu2, Xiaoxiao Li3,4**, Xiangxu Meng**1 1School of Software, Shandong University, China 2College of Computing and Data Science, Nanyang Technological University, Singapore 3Department of Electrical and Computer Engineering, University of British Columbia, Canada 4 Vector Institute, Canada z_qi@mail.sdu.edu.cn, yingpeng.tang@ntu.edu.sg, lmeng@sdu.edu.cn, han.yu@ntu.edu.sg, xiaoxiao.li@ece.ubc.ca, mxx@sdu.edu.cn

## Abstract

Federated Class Incremental Learning (FCIL) aims to collaboratively process continuously increasing incoming tasks across multiple clients. Among various approaches, data replay has become a promising solution, which can alleviate forgetting by reintroducing representative samples from previous tasks. However, their performance is typically limited by class imbalance, both within the replay buffer due to limited global awareness and between replayed and newly arrived classes. To address this issue, we propose a class-wise balancing data replay method for FCIL (FedCBDR), which employs a global coordination mechanism for class-level memory construction and reweights the learning objective to alleviate the aforementioned imbalances. Specifically, FedCBDR has two key components:
1) the global-perspective data replay module reconstructs global representations of prior task in a privacy-preserving manner, which then guides a class-aware and importance-sensitive sampling strategy to achieve balanced replay; 2) Subsequently, to handle class imbalance across tasks, the task-aware temperature scaling module adaptively adjusts the temperature of logits at both class and instance levels based on task dynamics, which reduces the model's overconfidence in majority classes while enhancing its sensitivity to minority classes. Experimental results verified that FedCBDR achieves balanced class-wise sampling under heterogeneous data distributions and improves generalization under task imbalance between earlier and recent tasks, yielding a 2%-15% Top-1 accuracy improvement over six stateof-the-art methods.

## 1 Introduction

Federated learning (FL) is a distributed machine learning paradigm that enables collaborative training of a shared global model across multiple data sources [1, 2, 3, 4, 5]. It periodically performs parameter-level interaction between clients and the server instead of gathering clients' data, which can enhance data privacy while leveraging the diversity of distributed data sources to build a more generalized global model [6, 7, 8, 9, 10, 11]. This mechanism makes it widely applicable to various fields [12, 13, 14, 15, 16]. Building upon this foundation, Federated Class-Incremental Learning (FCIL) extends FL by introducing dynamic data streams where clients sequentially encounter different task classes under non-independent and identically distributed data [17, 18, 19, 20, 21]. However, this amplifies the inherent complexities of FL, as the global model must integrate heterogeneous and

Local-view Data Selection Replay Result Local Pseudo Feature Generation Global-perspective Data Replay Replay Result Client K
Local DataPersonalized Informative Model
…
Importance Score Client 1 Client 1 Local Data**Global** 
Model
…
…
Comprehensive Score Local Pseudo Feature
…1 2 3 4 Balanced Class-wise Sample Counts Importance Score Local Data
…
Personalized Informative Model
…
1 2 3 4 Unbalanced Class-wise Sample Counts Client K
Global Pseudo Feature Local DataGlobal ModelLocal Pseudo Feature
(a) Traditional Data Replay Strategy
(b) Global-perspective Data Replay Strategy
evolving knowledge from clients while mitigating catastrophic forgetting, despite having no or only limited access to historical data [22, 23, 24].

To address the challenge of catastrophic forgetting in FCIL, data replay has emerged as a promising strategy for retaining knowledge from previous tasks. Existing replay-based methods can be broadly categorized into two types: generative-based replay and exemplar-based replay. The former leverages generative models to synthesize representative samples from historical tasks [23, 25, 26]. Its core idea is to learn the data distribution of previous tasks and internalize knowledge in the form of model parameters, enabling the indirect reconstruction of prior knowledge through sample generation when needed [19, 27]. However, they often overlook the computational cost of training generative models and are inherently constrained by the quality and fidelity of the synthesized data [23, 28, 29]. In contrast, exemplar-based replay methods directly store real samples from previous tasks, avoiding the complexity of generative processes while leveraging high-quality raw data to ensure robust retention of prior knowledge [17, 28, 30, 31]. These methods rely on a limited set of historical samples to maintain the decision boundaries of previously learned task classes. However, due to the lack of a global perspective on data distribution across clients, these methods are prone to class-level imbalance in replayed samples, which undermines the model's ability to retain prior knowledge [30, 31]. To address these issues, this paper proposes a class-wise balancing data replay method for FCIL, termed FedCBDR, which incorporates the global signal to regulate class-balanced memory construction, aiming to achieve distribution-aware replay and mitigate the challenges posed by non-IID client data, as illustrated in Figure 1. Specifically, FedCBDR comprises two primary modules: 1)
the global-perspective data replay (GDR) module reconstructs a privacy-preserving pseudo global representation of historical tasks by leveraging feature space decomposition, which enables effective cross-client knowledge integration while preserving essential attributes information. Furthermore, it introduces a principled importance-driven selection mechanism that enables class-balanced replay, guided by a globally-informed understanding of data distribution; 2) the task-aware temperature scaling (TTS) module introduces a multi-level dynamic confidence calibration strategy that combines task-level temperature adjustment with instance-level weighting. By modulating the sharpness of the softmax distribution, it balances the predictive confidence between majority and minority classes, enhancing the model's robustness to class imbalance between historical and current task samples. Extensive experiments were conducted on three datasets with different levels of heterogeneity, including performance comparisons, ablation studies, in-depth analysis, and case studies. The results demonstrate that FedCBDR effectively balances the number of replayed samples across classes and alleviates the long-tail problem. Compared to six state-of-the-art existing methods, FedCBDR achieves a 2%-15% Top-1 accuracy improvement.

## 2 Related Work 2.1 Exemplar-Based Replay Methods

In FCIL, exemplar-based replay methods aim to mitigate catastrophic forgetting by storing and replaying a subset of samples from previous tasks. They typically maintain a small exemplar buffer on each client, which is used during training alongside new task data to preserve knowledge of previously learned classes [17, 30, 31, 32, 33, 34]. For example, GLFC alleviates forgetting in FCIL by leveraging local exemplar buffers for rehearsal, while introducing class-aware gradient compensation and prototype-guided global coordination to jointly address local and global forgetting [17]. Moreover, Re-Fed introduces a Personalized Informative Model to strategically identify and replay task-relevant local samples, enhancing the efficiency of buffer usage and further reducing forgetting in heterogeneous client environments [30]. However, the lack of global insight in local sample selection often results in class imbalance, while the long-tailed distribution between replayed and current data is frequently overlooked, degrading the effectiveness of data replay [30, 31].

## 2.2 Generative-Based Replay Methods

Generative replay methods aim to reconstruct the samples of past tasks through techniques such as generative modeling [19, 23, 27, 35, 36], which enables the model to revisit historical knowledge to mitigate catastrophic forgetting. Following this line of thought, TARGET generates pseudo features through a globally pre-trained encoder and performs knowledge distillation by aligning the current model's predictions with those of a frozen global model [27]; LANDER utilizes pre-trained semantic text embeddings as anchors to synthesize meaningful pseudo samples, and distills knowledge by aligning the model's predictions with class prototypes derived from textual descriptions [19]. However, these methods are typically limited by the high computational cost of training generative models and the suboptimal performance caused by low-fidelity pseudo samples [19, 27].

## 2.3 Knowledge Distillation-Based Methods

Knowledge distillation-based methods generally follow two paradigms. The first focus om aligning the output predictions of the current model with those of previous models, which aims to preserve task-specific decision boundaries [37, 38, 39, 40, 41, 42, 43, 44, 45]. The second estimates the importance of model parameters for previously learned tasks and performs regularization to prevent forgetting [46, 47]. Both approaches avoid storing raw data but are prone to knowledge degradation over time, especially as the number of tasks increases [37, 46].

## 3 Preliminaries

We consider a federated class-incremental learning (FCIL) setting, where a central server aims to collaboratively train a global model with the assistance of K distributed clients. Each client k receives a sequence of classification tasks {D(1)
k, D
(2)
k*, . . . ,* D
(t) k
}, where each task introduces a disjoint set of new classes. Upon the arrival of task t, the global model parameters θt are optimized to minimize the average loss over the union of all samples seen so far, i.e., D
t =Sts=1 SK
k=1 D
(s)
k, by solving minθ1 |Dt| Pts=1 PK
k=1 PN
(s)
k i=1 L(fk(x
(s) k,i 
; θ), y
(s) k,i 
).

In replay-based methods, each client maintains a memory buffer with a fixed budget of M samples. When task t arrives, the client selects up to N representative samples from each of the previous tasks {1*, . . . , t* − 1}, subject to the total memory constraint. The resulting memory set is denoted by B
(t−1)
k =St−1 s=1{(x
(s) k,i 
, y
(s) k,i 
)}
N
i=1, where N is the number of samples stored per task and B
(t−1) k satisfies |B(t−1)
k| ≤ M. The local training set on client k then becomes D
(t)
k,train = D
(t)
k ∪ B(t−1)
k, combining current and replayed samples. Based on these local datasets, the server updates the global model by minimizing the aggregated loss: minθPK
k=1 P(x,y)∈D(t)
k,train L(fk(x; θ), y).

## 4 Class-Wise Balancing Data Replay For Federated Class-Incremental Learning

This section presents an effective active data selection method for FCIL, which aims to explore global data distribution to balance class-wise sampling. Moreover, it leverages temperature scaling to adjust the logits, which can alleviate the imbalance between samples from previously learned and newly introduced tasks. Figure 2 and Algorithm 1 illustrates the framework of the proposed FedCBDR.

Client k Server Client k
√
√
①Local Training
③**Local Model Aggregation**
Local Training Local Data Local Model Cached Sample Data Task s Replay Sample

( > )

( = )
… + … =
Current Task
④Global-perspective Data Replay Local Models Global **Model**
Previous Tasks Task s+1 Sample Cached Sample Pseudo Features

′
Global Model Inverse SVD

Proceed with Steps ①②
√√
√
√
√
…

′ 
′
SVD
…
Leverage Scores Features
②**Pseudo Feature Construction**
Cached Data Indices Task s Task s+1

## 4.1 Global-Perspective Data Replay (Gdr)

Due to privacy constraints, traditional data replay strategies typically rely on local data distributions. However, the absence of global information often leads to class imbalance in the replay buffer. To address this, the GDR module aggregates local informative features into a global pseudo feature set, enabling exploration of the global distribution without exposing raw data.

Inspired by Singular Value Decomposition (SVD) [48, 49], we first generate a set of random orthogonal matrices: a client-specific matrix P
(i)
k ∈ R
|D(i)
k |×|D(i)
k |for each client k and task i, and a globally shared matrix Q(i) ∈ R
d×d, where d denotes the dimension of the feature. Each client encrypts its local feature matrix X
(i)
k = Mg(D
(i)
k
) via Inverse Singular Value Decomposition (ISVD):

$$X_{k}^{(i)^{\prime}}=P_{k}^{(i)}X_{k}^{(i)}Q^{(i)},\tag{1}$$
$$(1)$$
$$(2)$$
$$(3)$$

and uploads the encrypted matrix X
(i)
′
kto the server, where Mg(·) is the feature extractor of the global model. The server then aggregates all encrypted matrices into a global matrix X(i)
′:

$X^{(i)^{\prime}}=\mbox{concat}\{X^{(i)^{\prime}}_{k}\mid k=1,\ldots,K\},$
and performs SVD as follows:
′⊤, (3)
where U
(i)
′∈ R
n×n, Σ
$$X^{(i)^{\prime}}=U^{(i)^{\prime}}\Sigma^{(i)^{\prime}}V^{(i)^{\prime}\top},$$
(i)
′∈ R
n×d, and V
(i)
′∈ R
d×d, with n denoting the total number of samples
from all clients. Next, the server extracts a submatrix of the left singular vectors for each client k by:
$$U_{k}^{(i)}={\mathcal{I}}_{k}(U^{(i)^{\prime}})\in\mathbb{R}^{|{\mathcal{D}}_{k}^{(i)}|\times n},$$
$\eqref{eq:walpha}$. 
|×n, (4)
where Ik(·) denotes a row selection function that returns the indices corresponding to client k's samples. To quantify the importance of local samples within the global latent space, client k computes a leverage score [50, 51, 52] for j-th sample of task i as:

τ i,j k = ∥e ⊤ i,jU (i) k ∥ 2
2, (5)
where ei,j denotes the j-th standard basis vector in task i. Notably, a higher leverage score indicates that the sample has a larger projection in the low-dimensional latent space, suggesting that it contributes more significantly to the global structure and is more representative. Moreover, clients send their leverage scores to the server, which aggregates them into a global vector τ i = concat{τ i,j k |k = 1*, ..., K*; j = 1*, ..., n*ik
} and normalizes it to obtain a sampling distribution:

$$p_{k}^{i,j}={\frac{\tau_{k}^{i,j}}{\sum_{j^{\prime}=1}^{n_{k}^{i}}\tau_{k}^{i,j^{\prime}}}}.$$
. (6)
$$({\boldsymbol{5}})$$
$$(6)$$

## Algorithm 1 Fedcbdr

1: **Initialize:** R: number of communication rounds; K: number of clients; t: number of tasks; θg:
global model parameters; B
pre k: replay buffer for historical tasks on client k; Dsk
: local data of task s on client k.

2: for each task s = 1 to t do 3: for each communication round r = 1 to R do 4: for each client k = 1 to K do 5: Initialize local model parameters: θk ← θg 6: if s == 1 **then**
7: Sample a mini-batch ζ from D
(1)
k, and update θk using Eq. 8.

8: **else**
9: Store the historical task data corresponding to globally sampled IDs into B
pre k.

10: Sample a mini-batch ζ from D
(s)
k ∪ Bpre k, and update θk using Eq. 9.

11: Compute pseudo-features based on Eq. 1, and upload them to the server. 12: **end if** 13: **end for** 14: if *r < R* **then** 15: Aggregate local model parameters across clients. 16: **else** 17: Aggregate model parameters and pseudo-features from all clients using Eq. 2. 18: Perform **Global Sampling** based on Eqs. 3–6, and send the selected sample IDs back to the corresponding clients.

19: **end if** 20: **end for** 21: **end for** 22: **// Global Sampling Procedure**
23: Form the global feature pool X(i) by aggregating all pseudo-features via Eq. (2).

24: Perform singular value decomposition (SVD) using Eq. 3 to extract key attributes. 25: Compute leverage scores for each client's samples using Eqs. 4–5, and normalize globally using Eq. 6.

26: Perform sampling and adjust the probabilities of the selected samples accordingly.

Subsequently, we perform i.i.d. sampling based on the distribution p = {p i,j k |k = 1*, ..., K*; j =
1*, ..., n*ik
}. Once a sample x is selected, its sampling weight is adjusted to √
1 ns·px ex, where ns denotes the number of selected samples and px is the original sampling probability of x, ex is the standard basis vector of x. This adjustment ensures unbiased estimation during aggregation. Following the sampling procedure, the server communicates the selected sample indices to their respective clients, where the corresponding data points are subsequently marked for further use.

## 4.2 Task-Aware Temperature Scaling (Tts)

Due to limited replay budgets, samples from previous tasks are often much fewer than those from the current task, leading to class imbalance and poor retention of past knowledge. To mitigate this, the TTS module dynamically adjusts sample temperature and weight based on task order, enhancing the contribution of tail-class samples during optimization. Specifically, we use a lower temperature to sharpen logits for samples from earlier tasks. Furthermore, to further amplify the optimization effect of tail-class samples during training, we also leverage a re-weighted cross-entropy loss, i.e.,

LTTS = 1 Nold  X Nold i=1 ωold · CE  yi, Softmax Concat z old i τold , z new i τnew   + 1 Nnew N X new j=1 ωnew · CE yj , Softmax Concat z old j τold , z new j τnew 

!!! (7)
where Nold and Nnew denote the number of samples from the previous and newly arrived task, respectively; yi and yj are the ground-truth labels; z old iand z new idenote the logits corresponding to old classes and new classes, respectively; τold and τnew are the temperature scaling factors for previous and newly arrived task samples; ωold and ωnew are the corresponding sample weights; CE(·) denotes

$$\left(7\right)$$

| Table 1: Statistics of the datasets used in experiments.   |        |               |          |            |                    |       |             |
|------------------------------------------------------------|--------|---------------|----------|------------|--------------------|-------|-------------|
| Datasets                                                   | #Class | #Training     | #Testing | Image Size | Federated Settings |       |             |
| Clients                                                    | Tasks  | Heterogeneity |          |            |                    |       |             |
| CIFAR10                                                    | 10     | 50,000        | 10,000   | 32 × 32    | 5/10               | 3/5   | 0.5/1.0     |
| CIFAR100                                                   | 100    | 50,000        | 10,000   | 32 × 32    | 5/10               | 5/10  | 0.1/0.5/1.0 |
| TinyImageNet                                               | 200    | 100,000       | 10,000   | 64 × 64    | 5/10               | 10/20 | 0.1/0.5/1.0 |

the cross-entropy loss function; and Softmax(z/τ ) is the temperature-scaled softmax function used to adjust the sharpness of the output distribution.

## 4.3 Training Strategy

The training strategy consists of two stages to progressively address the evolving challenges in federated class-incremental learning. Algorithm 1 presents the pipeline of the FedCBDR.

Stage 1: Initial Task Optimization. In the first task, client k learns from local data using the standard cross-entropy loss, i.e.,

$$\operatorname*{min}_{\theta_{k}}\;\;\frac{1}{N}\sum_{i=1}^{N}\mathrm{CE}(y_{i},\mathrm{Softmax}(f_{\theta_{k}}(x_{i}))),$$
$$({\mathfrak{s}})$$

Stage 2: Class-Incremental Optimization. As new tasks arrive and class imbalance emerges between previous and current tasks in client k, we employ L*T T S* to mitigate the imbalance, i.e.,

minθk 1 Nold  XNold i=1 ωold · CE yi, Softmax Concat f old θk (xi) τold , f new θk (xi) τnew  !!!  + 1 Nnew N Xnew j=1 ωnew · CE yj , Softmax Concat f old θk (xj ) τold , f new θk (xj ) τnew  !!! (9)
where xiis the input sample, yiis the corresponding ground-truth, f old θk
(x) and f new θk
(x) represent the outputs of the model corresponding to old and new classes, respectively. Softmax(·) converts the logits into a probability distribution.

## 5 Experiments 5.1 Experiment Settings

Datasets. Following existing studies [27, 30], we conducted all experiments on three commonly used datasets, including CIFAR10 [53, 54], CIFAR100 [53, 54] and TinyImageNet [55] to validate the effectiveness of the FedCBDR. We simulate heterogeneous data distributions across clients using the Dirichlet distribution with parameters β = {0.1, 0.5, 1.0}, where smaller values of β correspond to higher level of data heterogeneity. The statistical details are presented in the Table 1.

Evaluation Metric. Following prior studies [19, 56, 57, 58], we adopt Top-1 Accuracy as the evaluation metric, defined as Accuracy = Ncorrect/Ntotal, where Ncorrect and Ntotal denote the number of correct predictions and the total number of samples, respectively. Implementation Details. In the experiments, the number of clients is fixed at K = 5, with each client running local epochs E = 2 per round, using a batch size B = 128. For all datasets, we adopt ResNet-18 as the backbone, with the classifier's output dimension dynamically updated as tasks progress and conduct T = 100 communication rounds per task. The SGD optimizer is employed with a learning rate of 0.01 and a weight decay of 1 × 10−5. The number of stored samples per task varies by dataset and split setting: for CIFAR10, 450 samples are stored under 3-task splits and 300 under 5-task splits; for CIFAR100, 1,000 samples are used for 5-task splits and 500 for 10-task splits; for TinyImageNet, 2,000 samples are stored for 10-task splits and 1,000 for 20-task splits. For the temperature and weighted parameters, we select τold ∈ {0.8, 0.9} and wold ∈ {1.1, 1.2, 1.3, 1.4} for previous tasks, while τnew ∈ {1.1, 1.2} and wnew ∈ {0.7, 0.8, 0.9} are used for newly arrived tasks. Moreover, the hyperparameters of baselines are tuned based on their original papers for fair comparison. And training on each client is performed using an NVIDIA RTX 3090 GPU (24 GB).

| Method                                                                                                                                                                                                                                                                                                                                                       | CIFAR10                                                                         | CIFAR100   | TinyImageNet   |          |       |       |       |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|------------|----------------|----------|-------|-------|-------|
| β=0.5                                                                                                                                                                                                                                                                                                                                                        | β=1.0                                                                           | β=0.1      | β=0.5          | β=1.0    | β=0.1 | β=0.5 | β=1.0 |
| Finetune                                                                                                                                                                                                                                                                                                                                                     | 38.71±3.7 40.49±3.0 15.17±2.2 16.75±2.6 17.15±1.3                               | 6.06±0.9   | 6.00±0.8       | 6.40±0.5 |       |       |       |
| FedEWC 39.93±1.1 42.70±2.5 18.30±2.4 20.70±5.3 21.22±3.4                                                                                                                                                                                                                                                                                                     | 6.30±0.8                                                                        | 6.94±0.7   | 7.36±0.6       |          |       |       |       |
| FedLwF                                                                                                                                                                                                                                                                                                                                                       | 56.03±1.6 58.29±3.6 33.97±2.6 37.09±3.1 41.91±2.5 11.81±0.9 11.47±1.0 14.87±1.2 |            |                |          |       |       |       |
| TARGET 44.17±4.4 54.49±4.5 30.15±3.6 33.47±4.3 35.25±2.0 10.71±1.4 10.18±0.9 12.49±1.1 LANDER 53.90±3.2 60.79±1.4 44.07±3.3 47.63±3.7 52.77±1.4 13.80±0.8 15.02±1.9 16.36±1.0 Re-Fed 53.46±3.5 60.73±4.3 32.67±3.7 38.42±2.9 45.28±2.6 15.73±1.7 15.93±1.3 16.05±1.1 FedCBDR 64.11±1.2 65.20±1.9 46.40±1.6 49.76±2.7 52.06±1.5 18.37±1.1 18.86±0.9 18.78±0.9 |                                                                                 |            |                |          |       |       |       |

Table 2: Performance comparison between FedCBDR and baselines across three datasets under varying levels of heterogeneity (β). CIFAR10 is divided into 3 tasks, CIFAR100 into 5 tasks, and TinyImageNet into 10 tasks. All methods were executed under three different random seeds, and both the mean and standard deviation of the results are reported. The best results are **bolded**.

Table 3: Performance comparison between FedCBDR and baselines across three datasets under varying levels of heterogeneity (β). CIFAR10 is divided into 5 tasks, CIFAR100 into 10 tasks, and TinyImageNet into 20 tasks. All methods were executed under three different random seeds, and both the mean and standard deviation of the results are reported. The best results are **bolded**.

| Method                                                                                  | CIFAR10                                           | CIFAR100            | TinyImageNet        |           |          |          |          |
|-----------------------------------------------------------------------------------------|---------------------------------------------------|---------------------|---------------------|-----------|----------|----------|----------|
| β=0.5                                                                                   | β=1.0                                             | β=0.1               | β=0.5               | β=1.0     | β=0.1    | β=0.5    | β=1.0    |
| Finetune                                                                                | 19.78±2.3 23.34±2.8                               | 7.22±1.1            | 9.39±0.7            | 9.64±0.5  | 3.40±0.4 | 3.73±0.5 | 3.95±0.3 |
| FedEWC 20.11±2.7 28.97±2.3                                                              | 8.08±0.3                                          | 11.69±0.7 12.19±1.7 | 3.50±0.3            | 4.58±0.4  | 5.08±0.9 |          |          |
| FedLwF                                                                                  | 38.76±2.3 52.95±3.1 18.73±1.1 25.30±0.6 28.21±1.0 | 3.67±0.4            | 6.61±0.6            | 10.22±1.3 |          |          |          |
| TARGET 35.27±1.7 48.28±1.2 13.61±0.8 21.09±0.4 24.22±1.1                                | 5.32±0.6                                          | 5.39±0.6            | 5.72±0.5            |           |          |          |          |
| LANDER 40.22±2.4 58.07±3.4 27.79±1.9 33.51±2.3 37.42±1.8                                | 8.89±0.6                                          | 8.57±0.8            | 10.45±0.6           |           |          |          |          |
| Re-Fed                                                                                  | 54.94±3.1 58.19±2.5 29.33±1.3 39.54±1.3 40.96±1.1 | 9.36±0.9            | 11.44±0.7 12.27±1.1 |           |          |          |          |
| FedCBDR 61.18±1.3 65.42±1.8 45.11±1.2 46.51±1.6 47.79±1.4 12.58±0.4 14.47±0.7 15.69±0.6 |                                                   |                     |                     |           |          |          |          |

## 5.2 Performance Comparison

To evaluate the effectiveness of the proposed FedCBDR, we compare it with six representative baseline methods: Finetune [19], FedEWC [46], FedLwF [37], TARGET [27], LANDER [19], and Re-Fed
[30]. As reported in Table 2 and Table 3, the results can be summarized as follows:
- FedCBDR achieves the highest Top-1 accuracy in most cases across the three datasets under varying levels of heterogeneity and task splits. The only suboptimal result occurs on CIFAR100 with 5 tasks and β = 1.0, where FedCBDR (52.06%) performs slightly worse than LANDER (52.77%). This demonstrates the adaptability and robustness of the proposed FedCBDR across complex settings.

- Despite LANDER attains the best performance on CIFAR100 under the 5-task and β = 1.0 setting, it demands the generation of more than 10,000 samples per task, and the overhead of training its data generator surpasses that of the federated model, raising concerns about its scalability.

- Knowledge distillation-based methods like FedLwF perform well on simpler tasks (CIFAR10) by using pretrained knowledge to guide local models. However, their performance drops on more complex or heterogeneous tasks due to limited adaptability to local variations.

- Given an equal memory budget, class-balanced sampling (FedCBDR) consistently achieves superior performance compared to class-imbalanced strategy (Re-Fed), as it ensures more equitable representation across categories and effectively mitigates class-level forgetting in FCIL scenarios.

## 5.3 Ablation Study

In this section, we conducted an ablation study to investigate the contributions of key modules, including the Global-perspective Active Data Replay (GDR) module and the Task-aware Temperature Scaling (TTS) module. Table 4 presents the results, which can be summarized as follows:

| Task                                                                                                                                                                                                                                                             | CIFAR10                                                                         | CIFAR100                                                   | TinyImageNet   |          |       |       |       |       |       |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|------------------------------------------------------------|----------------|----------|-------|-------|-------|-------|-------|
| Splitting                                                                                                                                                                                                                                                        | Method                                                                          | β=0.5                                                      | β=1.0          | β=0.1    | β=0.5 | β=1.0 | β=0.1 | β=0.5 | β=1.0 |
| Finetune                                                                                                                                                                                                                                                         | 38.71±3.7 40.49±3.0 15.17±2.2 16.75±2.6 17.15±1.3 6.06±0.9                      | 6.00±0.8                                                   | 6.40±0.5       |          |       |       |       |       |       |
| +GDR                                                                                                                                                                                                                                                             | 62.13±2.1 63.81±1.9 45.28±1.5 47.66±0.9 51.47±1.7 17.24±0.6 17.89±0.5 18.04±0.4 |                                                            |                |          |       |       |       |       |       |
| 3/5/10                                                                                                                                                                                                                                                           | +TTS                                                                            | 41.34±2.3 42.55±2.2 17.32±0.5 17.14±0.4 19.32±0.5 6.67±0.2 | 6.92±0.3       | 7.27±0.4 |       |       |       |       |       |
| +GDR+TTS 64.11±1.2 65.20±1.9 46.40±1.6 49.76±2.7 52.06±1.5 18.37±1.1 18.86±0.9 18.78±0.9 Finetune 19.78±2.3 23.34±2.8 7.22±1.1 9.39±0.7 9.64±0.5 3.40±0.4 3.73±0.5 3.95±0.3 +GDR 59.34±3.1 63.20±2.6 44.04±1.3 46.33±0.5 46.50±0.8 11.44±0.3 13.85±0.5 14.51±0.6 |                                                                                 |                                                            |                |          |       |       |       |       |       |
| 5/10/20                                                                                                                                                                                                                                                          | +TTS                                                                            | 22.43±2.4 25.81±2.1 8.31±0.2 10.21±0.3 10.33±0.4 3.78±0.5  | 4.04±0.4       | 4.16±0.3 |       |       |       |       |       |
| +GDR+TTS 61.18±1.3 65.42±1.8 45.11±1.2 46.51±1.6 47.79±1.4 12.58±0.4 14.47±0.7 15.69±0.6                                                                                                                                                                         |                                                                                 |                                                            |                |          |       |       |       |       |       |

- Incorporating the GDR module substantially improves performance across all cases, particularly under high data heterogeneity (β = 0.1), demonstrating its effectiveness in alleviating catastrophic forgetting even with a limited number of replay samples in federated class-incremental learning.

- Using the TTS module alone leads to consistent improvements over Finetune, highlighting its effectiveness in addressing intra-client class imbalance through temperature scaling. This contribution to better generalization is particularly evident under the more challenging "5/10/20" task splitting scenario.

- The integration of both modules results in the best overall performance, consistently achieving the highest Top-1 accuracy across various datasets and heterogeneity levels. This stems from their complementary strengths: the GDR module mitigates inter-task forgetting, while the TTS module alleviates both intra- and inter-client class imbalance.

This section investigates the performance of FedCBDR and the baselines in incremental cases on three datasets. Figure 3 presents the average accuracy of all methods on both current and previous tasks. Notably, FedCBDR consistently outperforms other baseline methods across all task splits, with its accuracy curves remaining higher throughout the incremental process. Furthermore, FedCBDR exhibits a slower performance degradation as the number of tasks increases, indicating stronger resistance to catastrophic forgetting. In addition, it maintains significantly higher accuracy on later tasks, especially in challenging settings such as CIFAR100 and TinyImageNet with 10 tasks, highlighting its ability to balance knowledge retention and adaptation to new classes.

## 5.5 Quantitative Analysis Of Replay Buffer Size On Test Accuracy

In this section, we evaluate the performance of Re-Fed and FedCBDR under different buffer size M settings, and additionally include LANDER, which generates 10,240 synthetic samples for each task. As shown in Table 5, FedCBDR exhibits more significant performance advantages over Re-Fed under limited memory settings, and even surpasses LANDER, which relies on a large-scale generative replay buffer. Furthermore, as the buffer size increases, FedCBDR demonstrates more stable and significant performance improvements. This indicates that the method can effectively leverage larger replay buffers for continuous optimization. However, Re-Fed exhibits noticeable performance fluctuations

| Table 5: Comparison of model performance with varying memory size M across datasets. Methods CIFAR10 CIFAR100 TinyImageNet M=150 M=300 M=450 M=500 M=1000 M=1500 M=2000 M=2500   |       |       |       |       |       |       |       |       |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|-------|-------|-------|-------|-------|-------|-------|
| LANDER (10240)                                                                                                                                                                   | 52.90 | 47.05 | 14.77 |       |       |       |       |       |
| Re-Fed                                                                                                                                                                           | 47.23 | 53.47 | 54.66 | 33.89 | 38.42 | 47.84 | 15.89 | 16.78 |
| FedCBDR                                                                                                                                                                          | 51.99 | 59.02 | 63.81 | 40.12 | 49.66 | 55.94 | 18.33 | 19.41 |

under small and medium buffer settings. In particular, its accuracy is significantly lower than that of FedCBDR on CIFAR100 with M = 500 and TinyImageNet with M = 2000, indicating its limited ability to mitigate inter-class interference and retain knowledge from previous tasks. These findings validate that, under the same buffer budget, a balanced sampling distribution is more effective than an imbalanced one in alleviating forgetting and improving overall model performance.

   
Figure 4 gives a sensitivity analysis of FedCBDR with respect to temperature and sample weighting hyperparameters. Overall, temperature scaling and sample re-weighting help mitigate class imbalance, but model performance varies considerably with different hyperparameter settings. The model achieves better overall performance when ωold = 1.1, ωnew = 0.9, τold = 0.9, and τnew = 1.1. This is because slightly higher weight and temperature for previous-task samples help retain old knowledge, while lower weight and higher temperature for newly arrived samples reduce overfitting and improve adaptation. However, inappropriate hyperparameter choices may harm performance. For instance, a large τnew (e.g., 2.0) leads to overly smooth predictions, reducing discrimination among newly arrived classes. These results emphasize the need for proper tuning to ensure balanced learning.

Average 
To evaluate the effectiveness of FedCBDR in balancing class-wise sampling, Figure 5 illustrates the per-class sample distributions in the replay buffer between FedCBDR and Re-Fed across different task stages. Overall, across different task stages, FedCBDR (orange bars) exhibits a per-class sample distribution that is consistently closer to the average level (red line), whereas Re-Fed shows noticeable skewness and fluctuations. This indicates that FedCBDR is more effective in achieving balanced class-wise sampling in the replay buffer. In addition, FedCBDR ensures that no class is overlooked during sampling, while Re-Fed may fail to retain certain classes in the replay buffer—for example, class 79 is missing in Task 4 under Re-Fed. This highlights the robustness of FedCBDR in maintaining class coverage throughout incremental learning. 5.8 Visualization of Model Attention and Temperature-aware Logits Adjustment

Input Images
(a)
Bird(b)
Airplane(c)
Cat(d)
Deer(e)
Dog(f)
Ship Frog Horse Truck Bird Dog Frog Bird Frog Ship Bird Horse Cat 3.492 2.598 1.285 4.217 3.614 1.285 4.627 2.723 2.355 3.684 1.236 1.018 Horse Autom Truck Airp Bird Ship Airp Ship Dog Airp Truck Frog 3.819 3.036 2.183 4.725 4.112 3.217 4.428 2.419 2.183 4.026 2.913 2.372 Frog Truck Horse Cat Dog Deer Cat Deer Frog Cat Ship Dog 4.120 3.083 2.384 4.436 3.873 3.112 4.302 2.047 1.172 4.792 3.682 3.591 Truck Frog Autom Cat Dog Horse Deer Bird Truck Deer Dog Frog 2.172 1.267 0.325 3.927 3.504 2.271 3.628 1.126 0.482 4.284 3.211 2.906 Autom Frog Horse Deer Cat Dog Dog Frog Bird Dog Bird Deer 3.492 2.598 1.285 5.023 4.194 2.477 3.761 2.373 1.239 4.091 3.002 1.811 Truck Autom Horse Dog Ship Cat Ship Cat Truck Ship Autom Frog 2.902 2.360 1.119 3.884 2.633 2.128 4.764 3.288 1.683 3.557 2.274 0.853 FineTune
+GDR +GDR
+TTS (W)
+GDR
+TTS (T)
This section presents case studies comparing prediction confidence and attention focus using Grad- CAM [59, 60, 61, 62, 63] visualizations. As shown in Figure 6(a-c), in the absence of data replay, the model struggles to correctly classify samples from previous tasks and fails to attend to the relevant target regions. The incorporation of data replay in FedCBDR alleviates this issue by correcting predictions and guiding attention back to semantically important areas. Despite partially mitigating forgetting, data replay alone may still lead to misclassification or low-confidence predictions for tail classes with limited samples. The integration of temperature scaling (T) and sample re-weighting
(W) in the TTS module enables the model to better distinguish confusing classes through temperature adjustment, improving tail class accuracy and enhancing prediction stability, as depicted in Figure 6(d-f). These findings demonstrate the crucial role of the collaboration between both modules in mitigating knowledge forgetting during incremental learning.

## 6 Conclusions And Future Work

To address the challenge of inter-class imbalance in replay-based federated class-incremental learning, we propose FedCBDR that combines class-balanced sampling with loss adjustment to better exploit the global data distribution and enhance the contribution of tail-class samples to model optimization. Specifically, it uses SVD to decouple and reconstruct local data, aggregates local information in a privacy-preserving manner, and explores i.i.d. sampling within the aggregated distribution. In addition, it applies task-aware temperature scaling and sample re-weighting to mitigate the long-tail problem. Experimental results show that FedCBDR effectively reduces inter-class sampling imbalance and significantly improves final performance. Despite the impressive performance of FedCBDR, there remain several directions worth exploring to address its limitations. Specifically, we plan to investigate lightweight sampling strategies to reduce feature transmission costs in FedCBDR, and to develop more robust post-sampling balancing methods that mitigate class imbalance with less sensitivity to hyperparameters [64, 65, 66, 67]. Moreover, extending FedCBDR to more complex scenarios [68, 69, 70, 71, 72] is a promising direction.

## Acknowledgments

This work is supported in part by the Key Research and Development Program of Shandong Province
(Grant No. 2024TSGC0667) and the Ministry of Education, Singapore, under its Academic Research Fund Tier 1 (RG101/24).

## References

[1] Liping Yi, Han Yu, Zhuan Shi, Gang Wang, Xiaoguang Liu, Lizhen Cui, and Xiaoxiao Li. FedSSA: Semantic Similarity-based Aggregation for Efficient Model-Heterogeneous Personalized Federated Learning. In *Proc. IJCAI*, 2024.

[2] Ming Hu, Yue Cao, Anran Li, Zhiming Li, Chengwei Liu, Tianlin Li, Mingsong Chen, and Yang Liu.

Fedmut: Generalized federated learning via stochastic mutation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 12528–12537, 2024.

[3] Haozhao Wang, Haoran Xu, Yichen Li, Yuan Xu, Ruixuan Li, and Tianwei Zhang. Fedcda: Federated learning with cross-rounds divergence-aware aggregation. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024.

[4] Wei Feng, Zhenwei Wu, Qianqian Wang, Bo Dong, Zhiqiang Tao, and Quanxue Gao. Federated multi-view clustering via tensor factorization. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24, pages 3962–3970, 2024.

[5] Lele Fu, Bowen Deng, Sheng Huang, Tianchi Liao, Shirui Pan, and Chuan Chen. Less is more: Federated graph learning with alleviating topology heterogeneity from a causal perspective. In Forty-second International Conference on Machine Learning.

[6] Tianchi Liao, Lele Fu, Lei Zhang, Lei Yang, Chuan Chen, Michael K Ng, Huawei Huang, and Zibin Zheng.

Privacy-preserving vertical federated learning with tensor decomposition for data missing features. IEEE
Transactions on Information Forensics and Security, 2025.

[7] Zhuang Qi, Lei Meng, Zitan Chen, Han Hu, Hui Lin, and Xiangxu Meng. Cross-silo prototypical calibration for federated learning with non-iid data. In Proceedings of the 31st ACM International Conference on Multimedia, pages 3099–3107, 2023.

[8] Lele Fu, Sheng Huang, Yuecheng Li, Chuan Chen, Chuanfu Zhang, and Zibin Zheng. Learn the global prompt in the low-rank tensor space for heterogeneous federated learning. *Neural Networks*, 187:107319, 2025.

[9] Ming Hu, Peiheng Zhou, Zhihao Yue, Zhiwei Ling, Yihao Huang, Anran Li, Yang Liu, Xiang Lian, and Mingsong Chen. Fedcross: Towards accurate federated learning via multi-model cross-aggregation. In IEEE International Conference on Data Engineering (ICDE), pages 2137–2150. IEEE, 2024.

[10] Haozhao Wang, Yabo Jia, Meng Zhang, Qinghao Hu, Hao Ren, Peng Sun, Yonggang Wen, and Tianwei Zhang. Feddse: Distribution-aware sub-model extraction for federated learning over resource-constrained devices. In *Proceedings of the ACM Web Conference 2024*, pages 2902–2913, 2024.

[11] Liping Yi, Gang Wang, Xiaoguang Liu, Zhuan Shi, and Han Yu. Fedgh: Heterogeneous federated learning with generalized global header. In *Proc. MM, Ottawa, ON, Canada*, pages 8686–8696. ACM, 2023.

[12] Haoyuan Liang, Xinyu Zhang, Shilei Cao, Guowen Li, and Juepeng Zheng. Tta-feddg: Leveraging test-time adaptation to address federated domain generalization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 18658–18666, 2025.

[13] Wei Feng, Danting Liu, Qianqian Wang, Wenqi Liang, and Zheng Yan. Scalable federated one-step multi-view clustering with tensorized regularization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 16586–16594, 2025.

[14] Zhuang Qi, Weihao He, Xiangxu Meng, and Lei Meng. Attentive modeling and distillation for out-ofdistribution generalization of federated learning. In 2024 IEEE International Conference on Multimedia and Expo (ICME), pages 1–6. IEEE, 2024.

[15] Zhengyi Zhong, Weidong Bao, Ji Wang, Shuai Zhang, Jingxuan Zhou, Lingjuan Lyu, and Wei Yang Bryan Lim. Unlearning through knowledge overwriting: Reversible federated unlearning via selective sparse adapter. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pages 30661–30670, 2025.

[16] Zhengyi Zhong, Weidong Bao, Ji Wang, Xiaomin Zhu, and Xiongtao Zhang. Flee: A hierarchical federated learning framework for distributed deep neural network over cloud, edge, and end device. ACM
Transactions on Intelligent Systems and Technology (TIST), 13(5):1–24, 2022.

[17] Jiahua Dong, Lixu Wang, Zhen Fang, Gan Sun, Shichao Xu, Xiao Wang, and Qi Zhu. Federated classincremental learning. In *Proceedings of the IEEE/CVF conference on computer vision and pattern* recognition, pages 10164–10173, 2022.

[18] Feng Wu, Alysa Ziying Tan, Siwei Feng, Han Yu, Tao Deng, Libang Zhao, and Yuanlu Chen. Federated class-incremental learning via weighted aggregation and distillation. *IEEE Internet of Things Journal*, 2025.

[19] Minh-Tuan Tran, Trung Le, Xuan-May Le, Mehrtash Harandi, and Dinh Phung. Text-enhanced datafree approach for federated class-incremental learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 23870–23880, 2024.

[20] Xin Yang, Hao Yu, Xin Gao, Hao Wang, Junbo Zhang, and Tianrui Li. Federated continual learning via knowledge fusion: A survey. *IEEE Transactions on Knowledge and Data Engineering*, 36(8):3832–3850, 2024.

[21] Lele Fu, Sheng Huang, Yanyi Lai, Tianchi Liao, Chuanfu Zhang, and Chuan Chen. Beyond federated prototype learning: Learnable semantic anchors with hyperspherical contrast for domain-skewed data. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 16648–16656, 2025.

[22] Yanyan Lu, Lei Yang, Hao-Rui Chen, Jiannong Cao, Wanyu Lin, and Saiqin Long. Federated classincremental learning with dynamic feature extractor fusion. *IEEE Transactions on Mobile Computing*, 2024.

[23] Yuanlu Chen, Alysa Ziying Tan, Siwei Feng, Han Yu, Tao Deng, Libang Zhao, and Feng Wu. General federated class-incremental learning with lightweight generative replay. *IEEE Internet of Things Journal*,
2024.

[24] Xin Gao, Xin Yang, Hao Yu, Yan Kang, and Tianrui Li. Fedprok: Trustworthy federated class-incremental learning via prototypical feature knowledge transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4205–4214, 2024.

[25] Sara Babakniya, Zalan Fabian, Chaoyang He, Mahdi Soltanolkotabi, and Salman Avestimehr. A datafree approach to mitigate catastrophic forgetting in federated class incremental learning for vision tasks.

Advances in Neural Information Processing Systems, 36:66408–66425, 2023.

[26] Naibo Wang, Yuchen Deng, Wenjie Feng, Jianwei Yin, and See-Kiong Ng. Data-free federated class incremental learning with diffusion-based generative memory. *arXiv preprint arXiv:2405.17457*, 2024.

[27] Jie Zhang, Chen Chen, Weiming Zhuang, and Lingjuan Lyu. Target: Federated class-continual learning via exemplar-free distillation. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 4782–4793, 2023.

[28] Jinglin Liang, Jin Zhong, Hanlin Gu, Zhongqi Lu, Xingxing Tang, Gang Dai, Shuangping Huang, Lixin Fan, and Qiang Yang. Diffusion-driven data replay: A novel approach to combat forgetting in federated class continual learning. In *European Conference on Computer Vision*, pages 303–319. Springer, 2024.

[29] Min Kyoon Yoo and Yu Rang Park. Federated class incremental learning: A pseudo feature based approach without exemplars. In *Proceedings of the Asian Conference on Computer Vision*, pages 488–498, 2024.

[30] Yichen Li, Qunwei Li, Haozhao Wang, Ruixuan Li, Wenliang Zhong, and Guannan Zhang. Towards efficient replay in federated incremental learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12820–12829, 2024.

[31] Yichen Li, Haozhao Wang, Yining Qi, Wei Liu, and Ruixuan Li. Re-fed+: A better replay strategy for federated incremental learning. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2025.

[32] Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H Lampert. icarl: Incremental classifier and representation learning. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pages 2001–2010, 2017.

[33] Jiahua Dong, Hongliu Li, Yang Cong, Gan Sun, Yulun Zhang, and Luc Van Gool. No one left behind:
Real-world federated class-incremental learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(4):2054–2070, 2024.

[34] Yichen Li, Wenchao Xu, Haozhao Wang, Yining Qi, Ruixuan Li, and Song Guo. Sr-fdil: Synergistic replay for federated domain-incremental learning. *IEEE Transactions on Parallel and Distributed Systems*, 2024.

[35] Thinh Nguyen, Khoa D Doan, Binh T Nguyen, Danh Le-Phuoc, and Kok-Seng Wong. Overcoming catastrophic forgetting in federated class-incremental learning via federated global twin generator. *arXiv* preprint arXiv:2407.11078, 2024.

[36] Daiqing Qi, Handong Zhao, and Sheng Li. Better generative replay for continual federated learning. arXiv preprint arXiv:2302.13001, 2023.

[37] Zhizhong Li and Derek Hoiem. Learning without forgetting. IEEE transactions on pattern analysis and machine intelligence, 40(12):2935–2947, 2017.

[38] Alysa Ziying Tan, Siwei Feng, and Han Yu. Fl-clip: Bridging plasticity and stability in pre-trained federated class-incremental learning models. In *2024 IEEE International Conference on Multimedia and* Expo (ICME), pages 1–6. IEEE, 2024.

[39] Kunlun Xu, Fan Zhuo, Jiangmeng Li, Xu Zou, and Jiahuan Zhou. Self-reinforcing prototype evolution with dual-knowledge cooperation for semi-supervised lifelong person re-identification. ICCV, 2025.

[40] Athanasios Psaltis, Christos Chatzikonstantinou, Charalampos Z Patrikakis, and Petros Daras. Fedrcil:
Federated knowledge distillation for representation based contrastive incremental learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3463–3472, 2023.

[41] Jiao Chen, Jiayi He, Jianhua Tang, Weihua Li, and Zihang Yin. Knowledge efficient federated continual learning for industrial edge systems. *IEEE Transactions on Network Science and Engineering*, 2025.

[42] Zhengyi Zhong, Weidong Bao, Ji Wang, Jianguo Chen, Lingjuan Lyu, and Wei Yang Bryan Lim. Sacfl:
Self-adaptive federated continual learning for resource-constrained end devices. *IEEE Transactions on* Neural Networks and Learning Systems, 2025.

[43] Kunlun Xu, Xu Zou, Yuxin Peng, and Jiahuan Zhou. Distribution-aware knowledge prototyping for non-exemplar lifelong person re-identification. In *CVPR*, pages 16604–16613, 2024.

[44] Jiahua Dong, Duzhen Zhang, Yang Cong, Wei Cong, Henghui Ding, and Dengxin Dai. Federated incremental semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3934–3943, June 2023.

[45] Kunlun Xu, Zichen Liu, Xu Zou, Yuxin Peng, and Jiahuan Zhou. Long short-term knowledge decomposition and consolidation for lifelong person re-identification. *TPAMI*, 2025.

[46] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. *Proceedings of the national academy of sciences*, 114(13):3521–3526, 2017.

[47] Hao Yu, Xin Yang, Xin Gao, Yihui Feng, Hao Wang, Yan Kang, and Tianrui Li. Overcoming spatialtemporal catastrophic forgetting for federated class-incremental learning. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 5280–5288, 2024.

[48] Di Chai, Junxue Zhang, Liu Yang, Yilun Jin, Leye Wang, Kai Chen, and Qiang Yang. Efficient decentralized federated singular vector decomposition. In *2024 USENIX Annual Technical Conference (USENIX ATC* 24), pages 1029–1047, 2024.

[49] Weixin An, Yingjie Yue, Yuanyuan Liu, Fanhua Shang, and Hongying Liu. A numerical des perspective on unfolded linearized admm networks for inverse problems. In Proceedings of the 30th ACM International Conference on Multimedia, pages 5065–5073, 2022.

[50] Petros Drineas, Malik Magdon-Ismail, Michael W Mahoney, and David P Woodruff. Fast approximation of matrix coherence and statistical leverage. *The Journal of Machine Learning Research*, 13(1):3475–3506, 2012.

[51] Zhengxiang Pan, Han Yu, Chunyan Miao, and Cyril Leung. Efficient collaborative crowdsourcing. In The 30th AAAI Conference on Artificial Intelligence (AAAI-16), pages 4248–4249, 2016.

[52] Han Yu, Siyuan Liu, Alex C Kot, Chunyan Miao, and Cyril Leung. Dynamic witness selection for trustworthy distributed cooperative sensing in cognitive radio networks. In *Proceedings of the 13th IEEE* International Conference on Communication Technology (ICCT'11), pages 1–6, 2011.

[53] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

[54] Zhuang Qi, Yuqing Wang, Zitan Chen, Ran Wang, Xiangxu Meng, and Lei Meng. Clustering-based curriculum construction for sample-balanced federated learning. In CAAI international conference on artificial intelligence, pages 155–166. Springer, 2022.

[55] Ya Le and Xuan Yang. Tiny imagenet visual recognition challenge. *CS 231N*, 7(7):3, 2015. [56] Zekai Chen, Chentao Jia, Ming Hu, Xiaofei Xie, Anran Li, and Mingsong Chen. Flexfl: Heterogeneous federated learning via apoz-guided flexible pruning in uncertain scenarios. IEEE Transactions on Computer-
Aided Design of Integrated Circuits and Systems, 43(11):4069–4080, 2024.

[57] Haozhao Wang, Peirong Zheng, Xingshuo Han, Wenchao Xu, Ruixuan Li, and Tianwei Zhang. Fednlr:
Federated learning with neuron-wise learning rates. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 3069–3080, 2024.

[58] Zhuang Qi, Lei Meng, Weihao He, Ruohan Zhang, Yu Wang, Xin Qi, and Xiangxu Meng. Cross-training with multi-view knowledge fusion for heterogenous federated learning. *arXiv preprint arXiv:2405.20046*, 2024.

[59] Zhuang Qi, Lei Meng, and et al. Cross-silo feature space alignment for federated learning on clients with imbalanced data. In *The 39th Annual AAAI Conference on Artificial Intelligence (AAAI-25)*, pages 19986–19994, 2025.

[60] Lei Meng, Zhuang Qi, Lei Wu, Xiaoyu Du, Zhaochuan Li, Lizhen Cui, and Xiangxu Meng. Improving global generalization and local personalization for federated learning. IEEE Transactions on Neural Networks and Learning Systems, 36(1):76–87, 2025.

[61] Zhuang Qi, Sijin Zhou, Lei Meng, Han Hu, Han Yu, and Xiangxu Meng. Federated deconfounding and debiasing learning for out-of-distribution generalization. *arXiv preprint arXiv:2505.04979*, 2025.

[62] Lei Meng, Xiangxian Li, Xiaoshuo Yan, Haokai Ma, Zhuang Qi, Wei Wu, and Xiangxu Meng. Causal inference over visual-semantic-aligned graph for image classification. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 19449–19457, 2025.

[63] Bokun Wang, Yang Yang, Xing Xu, Alan Hanjalic, and Heng Tao Shen. Adversarial cross-modal retrieval.

In *Proceedings of the 2017 ACM on Multimedia Conference*, pages 154–162, 2017.

[64] Zitan Chen, Zhuang Qi, Xiao Cao, Xiangxian Li, Xiangxu Meng, and Lei Meng. Class-level structural relation modeling and smoothing for visual representation learning. In *Proceedings of the 31st ACM* International Conference on Multimedia, pages 2964–2972, 2023.

[65] Zhuang Qi and Xiaming Chen. A novel density-based outlier detection method using key attributes.

Intelligent Data Analysis, 26(6):1431–1449, 2022.

[66] Xiaoshuo Yan, Zhaochuan Li, Lei Meng, Zhuang Qi, Wei Wu, Zixuan Li, and Xiangxu Meng. Empowering vision transformers with multi-scale causal intervention for long-tailed image classification. *arXiv preprint* arXiv:2505.08173, 2025.

[67] Xiangxian Li, Yuze Zheng, Haokai Ma, Zhuang Qi, Xiangxu Meng, and Lei Meng. Cross-modal learning using privileged information for long-tailed image classification. *Computational Visual Media*, 10(5):981– 992, 2024.

[68] Zhiqiang Kou, Jing Wang, Yuheng Jia, and Xin Geng. Inaccurate label distribution learning. IEEE
Transactions on Circuits and Systems for Video Technology, 34(10):10237–10249, 2024.

[69] Zhiqiang Kou, Jing Wang, Jiawei Tang, Yuheng Jia, Boyu Shi, and Xin Geng. Exploiting multi-label correlation in label distribution learning. In Kate Larson, editor, *Proceedings of the Thirty-Third International* Joint Conference on Artificial Intelligence, IJCAI-24, pages 4326–4334. International Joint Conferences on Artificial Intelligence Organization, 8 2024. Main Track.

[70] Mouxing Yang, Zhenyu Huang, Peng Hu, Taihao Li, Jiancheng Lv, and Xi Peng. Learning with twin noisy labels for visible-infrared person re-identification. In *Proceedings of the IEEE/CVF conference on* computer vision and pattern recognition, pages 14308–14317, 2022.

[71] Ming-Hui Liu, Harry Cheng, Tianyi Wang, Xin Luo, and Xin-Shun Xu. Learning real facial concepts for independent deepfake detection. In Proceedings of the International Joint Conference on Artificial Intelligence, pages 1585–1593, 2025.

[72] Yuqing Wang, Xiangxian Li, Yannan Liu, Xiao Cao, Xiangxu Meng, and Lei Meng. Causal inference for out-of-distribution recognition via sample balancing. *CAAI Transactions on Intelligence Technology*, 9(5):1172–1184, 2024.

## A Appendix A.1 Experimental Results A.1.1 Performance Comparison

To thoroughly verify the effectiveness of the proposed FedCBDR, we compare its performance against various baselines under the setting of 10 clients. Based on the original implementations, we generate 10,240 synthetic samples per task for both TARGET and LANDER. The data replay configurations for Re-Fed and FedCBDR follow the settings outlined in Section 5.1. The results are presented in Tables 6 and 7. Consistent with the results shown in Tables 2 and 3, FedCBDR achieves the best performance across all cases. Notably, FedCBDR **achieves over a 10% gain compared to the**
second-best performing method in several settings. Table 6: Performance comparison between FedCBDR and baseline methods across CIFAR-10, CIFAR-
100, and TinyImageNet under varying levels of data heterogeneity (Dirichlet parameter β). Specifically, CIFAR-10 is split into 3 tasks, CIFAR-100 into 5 tasks, and TinyImageNet into 10 tasks. The number of clients is fixed at 10, and all experiments are conducted with a random seed of 2023 to ensure reproducibility. The best results are **bolded**.

Method CIFAR10 CIFAR100 **TinyImageNet**

β=0.5 β=1.0 β=0.1 β=0.5 β=1.0 β=0.1 β=0.5 β=1.0

FedEWC 36.40 42.00 15.19 18.66 19.50 6.19 7.23 7.78

FedLwF 48.24 49.11 27.02 37.92 41.77 10.67 13.02 14.73

TARGET 38.23 41.11 18.34 23.59 25.71 7.45 8.29 8.87

LANDER 41.54 45.52 30.83 43.69 47.29 12.33 15.18 15.64

Re-Fed 45.49 52.22 31.81 36.40 37.95 9.28 11.48 12.10

FedCBDR 59.80 62.59 42.25 47.90 48.55 14.81 16.54 17.43

Table 7: Performance comparison between FedCBDR and baseline methods across CIFAR-10, CIFAR-
100, and TinyImageNet under varying levels of data heterogeneity (Dirichlet parameter β). Specifically, CIFAR-10 is split into 5 tasks, CIFAR-100 into 10 tasks, and TinyImageNet into 20 tasks. The number of clients is fixed at 10, and all experiments are conducted with a random seed of 2023 to ensure reproducibility. The best results are **bolded**.

Method CIFAR10 CIFAR100 **TinyImageNet**

β=0.5 β=1.0 β=0.1 β=0.5 β=1 β=0.1 β=0.5 β=1.0

FedEWC 20.18 23.33 6.68 10.98 12.30 3.27 4.80 4.89

FedLwF 43.31 46.79 13.82 17.79 27.80 4.50 5.71 9.07

TARGET 21.60 28.39 12.11 16.64 17.14 3.45 4.88 5.01

LANDER 27.24 32.21 10.74 25.87 31.79 4.74 12.05 13.21

Re-Fed 38.28 39.22 28.08 33.52 37.27 7.95 8.53 10.13

FedCBDR 51.71 59.57 37.42 43.82 45.50 11.51 14.45 15.25

## A.1.2 Performance Evaluation Of Fedcbdr **Under Incremental Tasks**

We evaluate the performance evolution of FedCBDR and competing methods under a 10-client setting across incremental tasks on three benchmark datasets. Specifically, CIFAR-10 is split into 3 tasks
(β = {0.5, 1.0}), CIFAR-100 into 5 tasks (β = {0.1, 0.5, 1.0}), and TinyImageNet into 10 tasks (β = {0.1, 0.5, 1.0}). As shown in Figure 7, FedCBDR **consistently outperforms all baseline**
methods across incremental tasks, maintaining higher accuracy on both current and previous tasks throughout the training process. Moreover, its performance degrades more slowly as the number of tasks increases.

 = .   =1.0
 = .   = .   =1.0
 = .   = .   =1.0

## A.1.3 Comparison Of Per-Class Sample Distributions In The Replay Buffer

We further validate the capability of the proposed FedCBDR to balance per-class sample distributions in more complex scenarios. Specifically, we divide the CIFAR100 dataset into 10 tasks. As illustrated in Figure 8, Re-Fed exhibits substantial disparities in the number of replayed samples across classes. For example, in task 1, while classes 10 and 18 contain nearly 100 samples each, class 11 has fewer than 10. In contrast, FedCBDR **effectively alleviates such class imbalance, with the number of** replayed samples for all classes remaining consistently close to the average (as marked by the red line). This contributes to more stable knowledge retention across tasks and enhances overall model generalization.

Table 8: Comparison of model performance with varying replay budget M per task across datasets,

with the number of clients fixed at 10, and heterogeneity level β = 0.5.

| with the number of clients fixed at 10, and heterogeneity level β = 0.5. Methods CIFAR10 CIFAR100 M=150 M=300 M=450 M=500 M=1000   | M=1500   |       |       |       |       |       |
|------------------------------------------------------------------------------------------------------------------------------------|----------|-------|-------|-------|-------|-------|
| Re-Fed                                                                                                                             | 39.22    | 42.93 | 45.49 | 28.21 | 36.40 | 41.78 |
| FedCBDR                                                                                                                            | 48.15    | 54.62 | 59.80 | 38.34 | 47.90 | 51.14 |

We compare the performance of the data replay-based methods, Re-Fed and FedCBDR, under varying replay buffer budgets. Specifically, for CIFAR10, the buffer size is adjusted among {150, 300, 450}, while for CIFAR100, it ranges from {500, 1000, 1500}. The number of clients is set to 10, and heterogeneity level β = 0.5. As shown in Table 8, **the performance of both methods improves as**

Average
the buffer size increases, with FedCBDR **maintaining a clear advantage over Re-Fed under all**
settings. This also underscores the importance of balancing per-class sample counts in the replay buffer to ensure fair representation and stable performance.

## A.1.5 Evaluation On The Impact Of Local Training Epochs

To assess the impact of local training intensity, we compare the performance of LANDER, Re-Fed, and FedCBDR, under varying local training epoch settings. Specifically, the evaluation is conducted on CIFAR10 divided into 3 tasks and CIFAR100 divided into 5 tasks, under a federated setting with 10 clients and a heterogeneity level of β = 0.5. As shown in Figure 9, **both GDR and GDR+TTS**
consistently outperform the baseline methods (LANDER and Re-Fed) across all local training epoch settings on both CIFAR10 and CIFAR100. Moreover, **GDR+TTS achieves the highest** test accuracy in every configuration. The improvement brought by TTS highlights its necessity in alleviating class imbalance during local training. And, unlike other methods whose performance drops at 10 local epochs due to biased updates, **GDR+TTS demonstrates a sustained improvement** potential.

## A.1.6 Performance Assessment Of The Final Model Across Tasks

This section compares the final model performance of different methods (LANDER, Re-Fed, FedCBDR) across various tasks. Specifically, all experiments are conducted under a federated setting with 5 clients and a heterogeneity level of β = 0.5. CIFAR10 is split into 3 tasks and CIFAR100 into 5 tasks. Each task is trained for 50 communication rounds, with each client performing 2 local training epochs per round using a batch size of 128. For sample replay, LANDER synthesizes 10,240 samples per task, while Re-Fed and FedCBDR retain 150 and 1,000 real samples per task on CIFAR10 and CIFAR100, respectively. As shown in Table 9, **LANDER suffers from significant forgetting of** earlier tasks, as evidenced by its low accuracy of only 1.37% on Task 1 of CIFAR10. This indicates a severe inability to retain prior knowledge. Moreover, **LANDER also shows a noticeable decline** in performance on the last task, achieving only 57.00% on Task 5 of CIFAR100, suggesting that its generalization to new tasks is also limited under non-i.i.d. conditions. Compared to LANDER and Re-Fed, GDR significantly enhances the retention of knowledge from most early tasks. **This** demonstrates the advantage of balanced sample replay over imbalanced sampling. In particular, GDR+TTS outperforms GDR alone, highlighting the effectiveness of the proposed TTS module in mitigating class imbalance and supporting long-term knowledge preservation under non-i.i.d.

settings.

| Table 9: Per-task and average accuracy (%) of different methods on CIFAR10 and CIFAR100. CIFAR10 CIFAR100 Task 1 Task 2 Task 3 ALL Task 1 Task 2 Task 3 Task 4 Task 5 ALL LANDER 1.37 30.00 88.32 44.74 33.95 40.90 43.70 44.45 57.00 44.00 Re-Fed 14.20 18.33 95.88 44.28 23.40 22.00 21.10 34.10 81.70 36.46 GDR 17.07 19.00 96.10 49.26 40.40 37.90 39.25 47.50 80.45 49.10 GDR+TTS 21.43 21.46 96.08 51.30 41.45 38.05 38.50 48.55 81.40 49.59   |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## A.1.7 Scalability, Complexity, And Communication Efficiency

We have conducted additional large-scale experiments by extending the number of clients to 50/100 while simulating asynchronous participation, and the client sampling rate is set to 0.2. The dataset is divided into five tasks, and in Re-Fed and FedCBDR, the number of replay samples is set to 150/300 for CIFAR-10 and 500/1000 for CIFAR-100.The following results can be summarized:

| 50 Clients         | CIFAR10   | CIFAR100   |         |       |
|--------------------|-----------|------------|---------|-------|
| α = 0.5            | α = 1.0   | α = 0.5    | α = 1.0 |       |
| FedLwF             | 17.51     | 19.43      | 19.99   | 23.91 |
| LANDER             | 18.08     | 21.27      | 26.96   | 27.34 |
| Re-Fed (300/500)   | 29.65     | 32.45      | 27.12   | 28.86 |
| FedCBDR (300/500)  | 34.76     | 35.98      | 28.75   | 30.90 |
| Re-Fed (600/1000)  | 36.40     | 38.46      | 35.94   | 37.79 |
| FedCBDR (600/1000) | 41.33     | 45.31      | 38.54   | 39.69 |

Table 11: Computation and communication analysis on CIFAR10 and CIFAR100 datasets. Top: computation overhead per client; Bottom: communication cost per round.

Dataset **#Clients**

(K)

Samples/ Client (nk)

Feature Dim (d)

ISVD Time/Client

(s)

| computation overhead per client; Bottom: communication cost per round. Samples/ Feature ISVD (K) Client (nk) Dim (d) Time/Client (s) Feature   | Extrac                     |               |                        |           |        |        |
|------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|---------------|------------------------|-----------|--------|--------|
| tion Time/Client (s) Total                                                                                                                     | Extra               |               |                        |           |        |        |
| Time/Client (s)                                                                                                                                |                     |               |                        |           |        |        |
| CIFAR10                                                                                                                                        | 5                   | 2000          | 512                    | 0.00580   | 0.9083 | 0.9141 |
| CIFAR100                                                                                                                                       | 5                   | 2000          | 512                    | 0.00452   | 1.0250 | 1.2952 |
| Samples/                                                                                                                                       | Feature             | Upload/Client | Total Upload per Index | Down           |        |        |
| (K)                                                                                                                                            | Client (nk) Dim (d) | (MB)          | Round (MB)             | load (KB) |        |        |
| CIFAR10                                                                                                                                        | 10                  | 100           | 512                    | 3.906     | 19.53  | 39.06  |
| CIFAR10                                                                                                                                        | 10                  | 100           | 128                    | 0.9765    | 4.8825 | 39.06  |
| CIFAR10                                                                                                                                        | 10                  | 100           | 32                     | 0.2441    | 1.2206 | 39.06  |

Moreover, we have further clarified the computational complexity of the SVD step. Client-side ISVD
costs O(nkd 2) per client, while server-side SVD costs O(N d2), where N =Pk nk is the total number of pseudo-features. We also provide a summary of the additional training cost on clients, the communication overhead, and the server-side computation incurred by our method in comparison to FedAvg. **Overall, the**
additional cost of performing SVD and computing leverage scores is relatively small.

## Neurips Paper Checklist

1. **Claims**
Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? Answer: [Yes] Justification: The research problem and the main contributions of this study are clearly articulated in the abstract and introduction sections. Guidelines:
- The answer NA means that the abstract and introduction do not include the claims made in the paper.

- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.

- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. **Limitations**

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: The conclusion and future work section include a discussion of the study's limitations. Guidelines:
- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.

- The authors are encouraged to create a separate "Limitations" section in their paper.

- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

- The authors should reflect on the factors that influence the performance of the approach.

For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. **Theory Assumptions And Proofs**

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?