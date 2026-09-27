# Dp Otl0 : Concealing Backdoored Model Updates In Federated Learning By **Data** Poisoning With L0**-Norm-Bounded Optimized Triggers**

000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054

## Anonymous Authors1 Abstract

clients, and those unaffected as *benign clients*.

Traditional backdoor attacks in Federated Learning (FL) that rely on fixed trigger patterns and model poisoning exhibit deficient attacking performance against state-of-the-art defenses due to the significant divergence between malicious and benign client model updates. To effectively conceal malicious model updates among benign ones, we propose *DP OT*L0, a backdoor attack strategy in FL that dynamically constructs a per-round backdoor objective by optimizing an L0-normbounded backdoor trigger, making backdoor data have minimal effect on model updates and preserving the global model's main-task performance. We theoretically justify the concealment property of *DP OT*L0's model updates in linear models. Our experiments show that *DP OT*L0, via only a *data*-poisoning attack, effectively undermines state-of-the-art defenses and outperforms existing backdoor attack techniques on various datasets.

## 1. Introduction

Federated Learning (FL) is a decentralized machinelearning approach that has gained widespread attention recently. Unlike traditional centralized model training, FL synthesizes model updates contributed by multiple clients, each computed locally from that client's data. That is, in each round of FL, a central server distributes a global model to participating clients, each of whom independently trains the model on its local data, and its model updates are aggregated by the server to update the global model. This approach offers enhanced data privacy, reduced communication overhead, and scalability for a large number of clients. Despite its advantages, FL has been proven susceptible to backdoor attacks (Bagdasaryan et al., 2020). In FL, backdoor attacks occur when adversaries inject triggers into a subset of clients' local data, causing their local models trained on the poisoned data to become compromised. After aggregating these compromised local models, the global model produces adversary-desired results when the same trigger conditions are met. In this work, we term clients manipulated by adversaries during local training as *malicious* 1 Traditional backdoor attacks in FL present two common deficiencies. First, the patterns of backdoor triggers are predefined by the attacker and remain unchanged throughout the entire attack process (Bagdasaryan et al., 2020). Consequently, the learning objective brought by backdoored data is static and incoherent with the learning objective of main-task data (benign objective), resulting in distinct differences in model updates after training. These malicious clients' model updates are therefore easily canceled out by robust aggregations. Second, many approaches rely on model-poisoning techniques to enhance the effectiveness of backdoor attacks. Implementing model-poisoning attacks requires attackers to change the training procedures of a certain number of clients to make their local training algorithms different from other clients. However, achieving this condition is challenging, as advanced defense mechanisms (Riege et al., 2024) have introduced Trusted Execution Environments (TEEs) to ensure the secure execution of client-side training, making it harder to maliciously modify the training procedure. Existing defenses against backdoor attacks in FL rely on a hypothesis that backdoor attacks will cause the updating direction of a model to deviate from its original benign objective (Fung et al., 2020; Cao et al., 2021). To counter this hypothesis, adversaries can align models' malicious updating directions to their original benign objectives. Applying this idea to FL, if the injection of backdoored data has minimal effect on a client's model updates, then detecting this client as malicious becomes challenging for defenses based on analyzing clients' model updates. Inspired by testing-stage adversarial exsamples (Szegedy, 2014; Carlini & Wagner, 2017), recent studies on backdoor attacks in FL have proposed adding adversarial perturbations to client data to minimize their impact on model updates, which we term it as L2-norm-bounded optimized triggers (Nguyen et al., 2024; Lyu et al., 2023). However, adding perturbations to data does not produce consistent backdoor features for the model to learn; instead, it substantially alters benign features, transforming them into new features to associate with target labels. Our experimental results indicate that this will overdetermine the learning objective, hindering the long-term convergence of the main-task objective. Our comparison experiments also demonstrate that pixel-pattern triggers with consistent backdoor features are generally more effective for data-poisoning-only attacks in FL.

055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109 In this work, we propose Data Poisoning with L0-normbounded Optimized Trigger (*DP OT*L0), a backdoor attack on FL that dynamically constructs the malicious objective to align model updates to the benign objective. *DP OT*L0 optimizes a single L0-norm-bounded backdoor trigger with consistent appearance across different images, aiming to introduce the fewest possible features to the learning objective. We provide theoretical justification that the difference in model update directions for benign and malicious objectives can be minimized by reducing the error of the malicious data on the model. Our experiments demonstrate that these small differences brought by *DP OT*L0enable malicious model updates to bypass defenses and integrate into global models, resulting in backdoored global models. Compared to existing optimized triggers, *DP OT*L0empirically proves to be a more effective training-stage attack, demonstrating better attack effectiveness and main-task objective convergence. We ensured *DP OT*L0
's subtlety by constraining the number of trigger pixels, degrading the accuracy of a clean vision model when classifying the poisoned data by no more than 30%.

Unlike testing-stage L0-norm-bounded adversarial examples (Papernot et al., 2016), the *DP OT*L0trigger is required to maintain a consistent appearance across different images, serving as the unified backdoor feature. In this work, we proposed algorithms to optimize the pixel values and placements of a specified-size trigger using a set of client data and a global model as input. To the best of our knowledge, this is the first work to address this challenge. Without any assistance of model-poisoning techniques, *DP OT*L0is an attack conducted simply by executing a normal training process on the poisoned data containing the *DP OT*L0trigger. We evaluated *DP OT*L0on four image datasets (FashionM-
NIST, FEMNIST, CIFAR10, and Tiny ImageNet) and four model architectures including ResNet and VGGNet. We assessed the attack effectiveness of *DP OT*L0 under a variety of defense conditions, testing it against twelve defense strategies that are based on analyzing clients' model updates along with one defense strategy that uses client-side adversarial training to recover the global model (Zhang et al., 2023).

We compared *DP OT*L0attack with four state-of-the-art data-poisoning backdoor attacks that employ fixed-pattern triggers, distributed fixed-pattern triggers (Xie et al., 2020),
partially L0-norm-bounded optimized triggers (Zhang et al., 2024), and L2-norm-bounded optimzed triggers (Nguyen et al., 2024). Using a small number of malicious clients
(5% of the total), *DP OT*L0 outperformed existing datapoisoning backdoor attacks in effectively undermining defenses without affecting the main-task performance of the FL system.

## 2. Related Work 2.1. Backdoor Attacks In Fl

FL is very vulnerable to backdoor attacks. As training data are privately held by clients, the security of data is hard to track and protect. We discussed existing backdoor attacks in FL for image classification tasks based on some important properties (more details can be found in Appendix A.2).

With vs. Without model poisoning. Backdoor attacks in FL primarily rely on data poisoning, where attackers embed triggers in local training data and alter labels to train malicious models. Model poisoning (Fang et al., 2020) is often introduced to strengthen these attacks, by directly manipulating clients' model updates or training algorithms. However, Trusted Execution Environments (TEEs), which authenticate and protect client-side training, make model poisoning difficult. In contrast, data poisoning is easier for attackers to conduct and harder to prevent, as clients would gather data from open, vulnerable sources. Static objective vs. Dynamic objective. A static objective in backdoor attack represents a pre-defined and unchanging objective that is independent of the training system's status, such as associating certain input features or patterns with incorrect predictions. Having static objectives make malicious model updates easier to detect due to their inconsistency with main-task objective. In contrast, a backdoor attack that adjusts its objective based on the training system's status is referred to as having a dynamic objective. For example, Gong et al. (2022) and Fang & Chen (2023) optimized the trigger pattern based on a hypothesis that maximizing the activation of certain neurons in the backdoored local model can enhance the attack's persistence on the global model, which provides preliminary insights into the potential of dynamically changing backdoor objectives. Zhang et al. (2024) optimized triggers for a situation where the global model is directly trained to unlearn the trigger, which is another pioneering work exploring the potential of using dynamic objectives to attack FL.

L2-norm vs. L0**-norm bounded optimized trigger.** Existing works (Lyu et al., 2023; Nguyen et al., 2024) proposed to conceal malicious model updates by using adversarial examples (Goodfellow et al., 2015; Kurakin et al., 2017) as poisoned data, and the perturbations on these examples are referred to as L2-norm-bounded optimized triggers. However, while adversarial examples are effective as testing-stage attack techniques, they are less suited for training-stage backdoor attacks. The extensive inconsistency introduced by adversarial alterations creates numerous redundant fea110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 tures, overdetermining the learning objective and hindering the convergence of the benign objective. Recent studies on L0-norm-bounded optimized triggers (Zhang et al., 2024; Fang & Chen, 2023) made constructive attempts in optimizing the values of fixed-shape triggers alongside their attack strategies. *DP OT*L0enhances the effectiveness of the L0-norm-bounded optimized trigger by optimizing not only its values but also its shape and placement. This enhancement enables data-poisoning attacks to achieve better performance without relying on additional attack strategies.

## 2.2. Defenses Against Backdoor Attacks In Fl

In this work, we focus on defenses that adhere to the privacypreserving principles of FL originally introduced by McMahan et al. (2017): clients' private data are kept local, and their model updates are not shared with any entities other than the server. For a discussion on additional defenses with varying privacy-preserving properties, please refer to the Appendix A.3. In existing defenses, the server and clients are the two subjects commonly considered for implementing defense strategies. For benign clients as the defense subject, the global model of each round is the input they receive from the FL system. Inspired by Neural Cleanse (Wang et al., 2019), Zhang et al. (2023) proposed using trigger inversion on the global model and adversarial training on local models to mitigate the impact of the backdoor trigger. However, its effectiveness against continually evolving optimized triggers remains unaddressed. For server as the defense subject, clients' model updates are the input that the server receives from the FL system. Numerous studies proposed to defend against backdoor attacks by analyzing clients' model updates, which can be further classified into the two categories below. Excluding model updates with outlier values or characteristics. Some existing works presume that a malicious client's model updates will exhibit significant differences from those of benign clients in values or certain characteristics extracted from values. Nguyen et al. (2022) and Fung et al. (2020) exclude a client's model updates that have outlier cosine similarity to other clients' model updates. Sharma et al. (2023) and Ozdayi et al. (2021) reduce or penalize the contribution of model updates that show a certain degree of sign dissimilarity, either on a model-wise or parameter-wise basis. Kumari et al. (2023) and Fereidooni et al. (2024) assess the probabilistic distribution and frequency transformation of clients' model updates, and eliminate outliers in these characteristics. Mozaffari et al. (2023) create a sparse space of model updates for clients to vote, and the server rejects outlier votes and aggregates the rest. Byzantine-robust aggregation. Some existing works propose aggregating only the most trustworthy model updates to tolerate the presence of malicious clients. Yin et al. (2018) aggregate reliable model updates parameter-wise by taking median or trimmed mean, while Blanchard et al. (2017), (Cao et al., 2022), and Pillutla et al. (2022) select and aggregate reliable model updates model-wise. Analyzing clients' model updates can effectively defend against backdoor attacks that cause distinctions between malicious clients' and benign clients' model updates. However, when a backdoor attack can conceal malicious clients' model updates among benign ones, defenses based on this strategy will struggle (Bagdasaryan et al., 2020). In this work, we show that this goal can be achieved by dynamically changing the backdoor objectives defined on poison data.

## 3. Threat Model

Attacker's capability and background knowledge: As shown in Figure 1, we assume that each FL client—even a malicious one—is equipped with trustworthy training software that conducts correct model training on the client's local training data and transmits the model updates to the FL server. Aligning with the security settings in the stateof-the-art defense work (Riege et al., 2024), we assume that both the client training pipeline and the FL server, as well as the communication between them, faithfully serve FL's main task training and cannot be undetectably manipulated. These properties would be achievable by executing FL training within Trusted Execution Environments (TEEs) (Schneider et al., 2022; Riege et al., 2024), for example, by applying cryptographic protections to the updates (e.g., a digital signatures) to enable the FL server to authenticate the updates as coming from the TEEs. The capability of malicious clients in our attack is limited to the manipulation of their local training data that are input to their training pipelines. In addition, in line with existing works (Lyu et al., 2023; Zhang et al., 2024; Fang & Chen, 2023; Gong et al., 2022), we do not assume the secrecy of the global model provided by the FL server, as it would typically need to be accessible outside TEEs for use in local inference tasks. As such, in each FL round, clients are granted white-box access to the global model. Originating from initially benign clients that have been compromised, these malicious clients possess some local training data for the FL main task as background knowledge.

Attacker's goals: The malicious clients aim to accomplish the following goals.

- **Effectiveness**. For classification tasks, *Attack Success* Rate (ASR) is the accuracy of a model in classifying data with triggers into a target label. In FL, backdoor attacks aim to make the post-aggregation global model misclassify data with training-stage triggers into a target

## 4. Dp Otl0 **Design** 4.1. Building A Trigger Training Dataset

At the beginning of the *DP OT*L0attack, we initially gather all available benign data from the malicious clients' local training datasets and assign a pre-defined target label yt to them. We refer to this new dataset, which associates benign data with the target label, as the trigger training dataset D.

## 4.2. Determining Trigger Size (Trisize)

165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 We determine the trigger size by ensuring that the accuracy drop of the poisoned data predicted as benign by an unattacked model does not exceed a threshold, which we set at 30% in this work. Different model architectures and datasets result in varying trigger sizes under this standard. Adversaries determine tri*size* based on their background knowledge of the FL model and data, and can dynamically reduce it during FL training to enhance the trigger's subtlety, with a trade-off in ASR. In the following discussion, we

DP OTL Trusted Execution Environments 0 **Attack**
Global Model Software user access Global Model model training computation of model updates Client #1 Model optimize trigger input Updates Software Software Client #2 Server Trigger Training Dataset Client #3 Client #n Software
...

FEMNIST CIFAR10
assume tri*size* is a static and optimal value.

## 4.3. Optimizing A Backdoor Trigger

We independently generate a backdoor trigger for each round's data poisoning using the optimization algorithms 1 and 2. In the image classification context, consider the global model Wg as the input and all pixels within an image forming the parameter space. Our approach seeks to identify a subset of parameters that have the greatest impact on producing the malicious output (i.e., the target label), and then optimizes the values of these parameters to further improve the accuracy of the result. The pixels in this subset with their optimized values will serve as a backdoor trigger. To enhance generalization performance of this trigger, we use all images in the trigger training dataset D to optimize its pixel placements and pixel values. Algorithm 1 Computation for Trigger Location Input: Wg, D, yt, tri *size* Output: Et 1: ∀x ∈ D : yx ← Wg(x).

2: L ← 1 |D| Px∈D(yx − yt)
2.

3: ∀x ∈ D : δx ← ∂L
∂x .

4: δ ← abs(Px∈D δx).

5: δf ← flatten δ into a one-dimensional array. 6: S ← argsort(δf ).{Store the sorted indices (descending sort)} 7: Et ← S[: tri *size* ]. {Top tri *size* indices are trigger locations}
8: Et ← transform from one-dimensional indices to indices for x ∈ D.

Compute trigger-pixel placements Et. In Algorithm 1, we select pixel locations that contain the largest absolute gradient sum with respect to the backdoor objective as the trigger-pixel placements. Algorithm 1 takes inputs including the global model Wg, the trigger training dataset D, the target label yt, and a parameter tri *size* that specifies the label. We evaluate a backdoor attack's final effectiveness using the final-round global model's ASR, with 50% as the success threshold. By-round effectiveness is measured as the average ASR across all FL rounds. Combined with the final ASR, it indicates how quickly an attack achieves sufficient effectiveness.

- **Main-task Convergence**. The main-task convergence goal of a backdoor attack is to preserve the global model's accuracy on its main-task data at a normal level, ensuring the model's functionality remains intact and the attack goes unnoticed.

- **Subtlety**. Backdoor triggers should preserve data's main details and avoid causing misinterpretation (Figure 2). Subtlety can be evaluated by measuring accuracy drops using an un-attacked computer vision model. We aim to ensure that accuracy drops by no more than 30%.

220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 trigger size. The trigger size tri *size* determines the number of pixel locations we will choose. The output of the Algorithm 1 is the trigger-pixel placement information denoted as Et. We calculate the loss of the global model Wg on clean images in dataset D predicted as the target label yt, using Mean Square Error (MSE) as the example loss function. Gradients of the loss with respect to each pixel are computed and summed across all images, producing an absolute gradient matrix. This matrix is flattened, sorted in descending order, and the top tri *size* indices are identified as the trigger-pixel placements, which are then mapped back to the original image shape.

| Algorithm 2 Optimization for Trigger Value   |
|----------------------------------------------|

| Output: Vt 1: for iteration ← 1 to niter do 2: D ′ ← D. 3: if iteration = 1 then 4: Vt ← 1 P x∈D′ x. |D′ | 5: else if iteration > 1 then ′ : x[Et] ← Vt[Et]. 6: ∀x ∈ D 7: end if ′ : yx ← Wg(x). 8: ∀x ∈ D 2 . 9: L ← 1 P x∈D′ (yx − yt) |D′ | 10: ∀x ∈ D ′ : δx ← ∂L .   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

Optimize trigger-pixel values Vt. In Algorithm 2, we optimize the values of the trigger pixels defined in Et using a learning-based approach. Algorithm 2 requires the following inputs: the trigger-pixel placements Et, the global model Wg, the trigger training dataset D, and the target label yt. Additionally, it uses two training parameters: the number of training iterations n*iter* and the learning rate γ.

The output produced by Algorithm 2 is the trigger-pixel value information denoted as Vt.

In each iteration, we create a copy dataset D′ of the clean dataset D to embed the optimized trigger. In the first iteration, we initialize the trigger-pixel value matrix Vt by averaging pixel values across all images in D′. We then compute the loss of the global model Wg on images from D′ with the target label yt, followed by calculating the gradients of the loss with respect to each pixel, storing them in δx. The gradient matrix δ is obtained by summing δx along each pixel location. Using gradient descent with learning rate γ, we update only the pixels within the trigger-pixel placements Et and assign the new values to Vt. In subsequent iterations, we replace the trigger-pixels in each image with their corresponding values from Vt, ensuring that only the trigger-pixels affect the loss.

## 4.4. Poisoning Malicious Clients' Training Data

The last step of our attack is to poison malicious clients' local training data using the optimized trigger τ = (Et, Vt) and its target label yt by a certain data poison rate.

## 5. Theoretical Analysis

Gounding in the feature learning propertires of neural networks (Shi et al., 2022; Zeiler, 2014; Girshick et al., 2014), we assume a dataset D's valid information can be extracted as a feature set, expressed as K = (v1, v2*, . . . , v*k) ∈
R 
n×k. Each vi ∈ R
n has a target value yi, and y =
(y1, y2*, . . . , y*k) ∈ R
k. For a linear system w ∈ R
n×1, the learning objective is to find w
∗ ∈ R
n such that KT w
∗ =
y T. See proof of 5.1 in Appendix B.1.

Proposition 5.1. **(Concealment Property)** Given a feature set K ∈ R
n×k *with its target values* y ∈ R
k *and a model* w ∈ R
n*, assume an adversary generates a malicious feature* set Kadv ∈ R
n×p with adversarial target values yadv ∈
R 
p. Let the error of (Kadv, yadv) on w be denoted as ϵadv, where ϵadv = KT
advw −y T
adv*. Let the optimization direction* for w *with respect to* (K, y) be denoted by ∆wK, and the optimization direction for w with respect to the combined feature set ([K, Kadv], [y, yadv]) be denoted by ∆wK∪Kadv .

The difference between the two update directions is bounded as:
∥∆wK∪Kadv − ∆wK∥ ≤ δ∥ϵadv∥
where δ = max ∥vi∥, vi ∈ Kadv, representing the maximum magnitude of the feature vectors in the adversarial dataset Kadv. Specifically, this bound indicates that the difference between the two update directions is proportional to the error in optimizing (Kadv, yadv) for w.

## 6. Experiments 6.1. Fl Configurations

We conducted experiments on four benchmark image datasets: Fashion MNIST, FEMNIST, CIFAR10, and Tiny ImageNet, using four different model architectures including ResNet and VGGNet, as detailed in Table 6. For FL settings, we consider 100 clients for grayscale image learning tasks and 50 clients for colorful image learning tasks. Clients' data are Non-iid distributed, where the Non-iid sampling followed the algorithm proposed by FLTrust (Cao et al., 2021), with a medium bias degree of 0.5. FEMNIST is naturally a Non-iid distributed dataset for FL, so we used it as is. Each client performed five local training epochs per global round and participated in all global rounds.

For grayscale image learning tasks, we used a fixed local learning rate of 0.1. For color image learning tasks, we applied learning rate scheduling techniques (He et al., 2016; Simonyan & Zisserman, 2015). We used SGD optimization 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 with CrossEntropy loss. In the experiments on Tiny ImageNet, we set the mini-batch size to 64, while for the other datasets, we set it to 256. The number of global rounds was determined based on the stabilization of test accuracy on the main task data, defined as remaining within 0.5 percentage points over five consecutive global rounds, which we considered convergence. The number of global rounds varied across datasets and model architectures, as detailed in Table 7.

## 6.2. Attack Configurations

The default Malicious Client Ratio (MCR) was set to 5%, meaning 2 out of 50 or 5 out of 100 clients engaged in data poisoning during training. The Data Poison Rate (DPR), representing the proportion of each malicious client's data poisoned with the *DP OT*L0trigger, was set to a default value of 0.5. Except for ablation studies or specific indications, all experiments followed the default configurations for MCR and DPR.

## 6.3. Evaluation Metrics

We used Final Attack Success Rate (Final ASR) and Average Attack Success Rate (Avg ASR) to evaluate the effectiveness of backdoor attacks in FL. Final ASR, calculated as the mean ASR of the global model from the last five rounds, measures the final effectiveness of the attack. Avg ASR, calculated as the mean ASR across all global rounds, assesses the average by-round effectiveness. A higher Avg ASR indicates faster achievement of sufficient attack effectiveness.

We used Main-task Accuracy (MA) to evaluate the performance of the final global model on main-task data. A backdoor attack is considered to maintain main-task convergence if the MA of its victim model is within a ±2 percentagepoint difference compared to the MA of the un-attacked model.

## 6.4. Other Backdoor Triggers

In this study, we consider data poisoning as the sole attack strategy for backdoor attacks in FL. Similar attack settings in existing literature are relatively scarce, with many current FL attacks combining data poisoning with other strategies or targeting novel FL structures. Rather than comparing all existing attack methods across various configurations, we choose to compare representative backdoor triggers with DP OTL0's trigger in a purely data poisoning attack context under unified settings to evaluate their effectiveness.

- **Fixed Trigger (FT).** A single pixel-pattern trigger with fixed value, shape, and placement (Baruch et al., 2019; Bagdasaryan et al., 2020).

- **Distributed Fixed Trigger (DFT).** Different fixed triggers are used by malicious clients, with their union employed for testing (Xie et al., 2020).

- Partially L0**-norm-bounded Optimized Trigger**
(OTval L0
). A single pixel-pattern trigger with dynamically optimized values but fixed shape and placement (Zhang et al., 2024).

- L2**-norm-bounded Optimized Trigger (**OTL2). Adversarial perturbations added on data, generated with constraints of their L2-norm (Nguyen et al., 2024).

## 6.5. Defenses

We selected defenses that have open-sourced their proofof-concept code to ensure accurate implementation of their proposed ideas. Twelve of them are state-of-the-art serverconduct defenses based on analyzing difference of model updates from clients: FedAvg (McMahan et al., 2017), Median (Yin et al., 2018), Trimmed Mean (Yin et al., 2018),
RobustLR (Ozdayi et al., 2021), RFA (Pillutla et al., 2022), FLAIR (Sharma et al., 2023), FLCert (Cao et al., 2022), FLAME (Nguyen et al., 2022), FoolsGold (Fung et al., 2020), Multi-Krum (Blanchard et al., 2017), BackdoorIndicator (Li & Dai, 2024), and FRL (Mozaffari et al., 2023). Detailed descriptions can be found in Appendix D. One defense is conducted on client-side: Flip (Zhang et al., 2023). Experiment results of Flip, FRL, and BackdoorIndicator are given in Appendix G, H, and I due to space limitations.

## 6.6. Dp Otl0Vs. Otl2Vs. Otval L0

We present OTL2, OTval L0
, and *DP OT*L0triggers on CI-
FAR10 images in Figure 3. The size of the *DP OT*L0trigger is set to 25, based on the subtlety maintenance rule. The OTval L0trigger consists of 25 pixels arranged in a square shape. We placed it in two different positions in the images
- upper-left (OTval L0
-1) and center (OTL2-2 ) . Both OTL2 and OTval L0 triggers are optimized to minimize backdoor loss on each round's global model before being used for training. Their optimization methods are based on two recent attack works: A3FL (Zhang et al., 2024) and IBA (Nguyen et al., 2024).

(a) OTL2
(right) (b) OT val L0
-1 (c) OT val L0
-2 (d) *DP OT*L0
The comparative results of the three optimized triggers in terms of Final ASR, Avg ASR, and MA are presented in Table 1. Compared to the OTval L0and *DP OT*L0triggers, the OTL2trigger demonstrates lower attack effectiveness in both final and average ASR. A potential explaination is that when the MCR is small (5%), the global model's updates are largely irrelevant to learning backdoor features, which impacts OTL2 triggers more than the L0-norm bounded triggers, as the OTL2 trigger contains more adversarial features to learn. Moreover, the accuracy of poisoned data with a larger number of features may be more negatively impacted by irrelevant changes in the global model along training.

The enhanced attack effectiveness of OTval L0
-2 compared to OTval L0
-1 underscores the significance of trigger placement as a key factor to achieve backdoor attack objectives in FL. The *DP OT*L0trigger with placement optimization therefore shows best effectiveness among all triggers. The baseline MA results of an un-attacked FL system employed with various defense strategies are shown in the CI- FAR10 column of Table 8. The MA results in Table 1 show minimal differences from the corresponding baseline values, indicating that attacks with different triggers maintain the main-task convergence of the FL system.

## 6.6.1. Main-Task Convergence By Otl2

| Table 1. Results of OTL2 , OT val , and DP OTL0 on CIFAR10. L0 Measures Final ASR Avg ASR MA 0 TLP OD0 TLP OD0 TLP OD-1 2 l vaL0 TLT Trigger Types OO-2 l vaL0 T O-1 2 l vaL0 TLT OO-2 l vaL0 T O-1 2 l vaL0 TLT OO-2 l vaL0 T OFedAvg 18.5 48.9 75.1 100 26.0 38.1 60.2 98.5 69.3 70.6 70.0 70.7 Median 21.8 32.9 28.4 100 14.2 24.0 26.7 96.1 69.8 69.1 69.9 69.1 TrimmedMean 10.2 35.0 85.5 100 12.2 23.5 62.8 88.6 69.7 69.9 70.2 70.4 RobustLR 32.8 46.2 86.5 100 33.7 40.7 65.6 98.6 70.3 71.2 70.3 70.1 RFA 9.0 24.7 41.6 100 9.5 23.8 38.8 97.8 70.4 70.2 70.7 70.7 FLAIR 0.1 13.2 14.9 62.3 3.7 12.5 17.0 50.7 70.5 70.7 69.2 70.6 FLAME 3.9 13.7 18.2 59.8 25.1 32.1 48.4 56.0 68.7 70.1 69.5 70.3 FoolsGold 14.8 46.9 64.8 100 13.4 38.0 50.8 98.5 70.1 70.8 70.5 71.0 FLCert 4.0 39.0 28.9 99.2 4.2 28.4 25.5 88.3 69.3 69.9 69.7 70.0 Multi-Krum 0.3 33.4 86.2 100 4.4 29.5 84.2 98.7 64.3 62.8 61.2 63.0   |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

To study the long-term impact of the OTL2trigger on maintask convergence in FL, we set the MCR to 50% to boost its attack effectiveness. The trigger was optimized for 30 rounds, and the poisoned data generated in the 30th round was used until the end (150th round). This simulates an unideal scenario where the attack is interrupted mid-training, allowing us to assess how remaining poisoned data affects MA over time. The results of *DP OT*L0under identical conditions are presented in Table 2 for comparison. As we analyzed before, substantially changing benign features to adversarial features makes OTL2 overdetermine the learning objective and hinder the convergence of main-task data. In contrast, *DP OT*L0is able to sustain the MA of FL training, even under conditions of a large MCR and an interruption in optimization.

## 6.6.2. Computational Overhead Comparison

| Table 2. Insufficient MA due to OTL2 attack. FedAvg FLAIR Final ASR Avg ASR MA Final ASR Avg ASR MA   |      |      |      |      |      |      |
|-------------------------------------------------------------------------------------------------------|------|------|------|------|------|------|
| OTL2                                                                                                  | 64.4 | 60.5 | 60.0 | 57.4 | 60.1 | 60.7 |
| DP OTL0                                                                                               | 95.2 | 96.5 | 69.9 | 74.5 | 82.8 | 70.6 |

We compared the elapsed time of trigger optimization algorithms in A3FL and IBA with *DP OT*L0 on the same computational platform, consisting of one NVIDIA A40 GPU core and 200 GB of CPU RAM. Comparison results are shown in Table 3. *DP OT*L0 demonstrates a relatively shorter total execution time. We assume adversaries can offset the timing gap caused by trigger optimization with powerful computational resources.

Table 3. Comparison of Elapsed Time Methods Total (s) Per Epoch (s) \# Epochs Benign Training (s)
DP OTL05.05 **0.50** 10 1.23 A3FL 421.04 2.07 200 1.23 IBA 16.56 1.59 10 1.23

## 6.6.3. Subtlety Comparison.

We evaluated the subtlety of four optimized triggers by measuring the accuracy drop of an un-attacked model when predicting poisoned data as benign labels ("Benign Acc" in Table 4). OTL2showed a relatively greater drop in benign accuracy due to its substantial alteration of benign features across the entire image.

Table 4. Benign accuracy drops caused by different triggers.

Triggers None DP OTL0 OT val L0
-1 OT val L0
-2 OTL2 Benign acc 70.81 52.98 70.46 67.65 **27.98** Drop (%) 0 25.18 0.49 4.46 **60.49**

## 6.7. Dp Otl0 **Vs. Ft Vs. Dft**

Figure 4 presents a comparison of the ASR results for the DP OTL0trigger, FT, and DFT across different datasets.

Visualizations of FT and DFT can be found in Figures 8 and 9, respectively. The MA results are provided in Table 8.

## 6.8. Discussion Of Dp Otl0

6.8.1. AGGREGATION OF MALICIOUS MODEL UPDATES
We demonstrated that *DP OT*L0's attack effectiveness arises from malicious model updates being aggregated into the global model, rather than solely from the optimized trigger's residual effects on the next-round global model.

To evaluate this, we designed an experiment where malicious clients generated a *DP OT*L0 trigger every round but did not use it to poison their data. We tested the ASR of the trigger on the next-round global model, measuring its residual effects, and denoted this as ASR 
]. In another experiment, malicious clients input the poisoned data with the DP OTL0trigger into the training, with the attack effectiveness denoted as ASR¨ . As shown in Appendix Table 9, ASR¨ is notably larger than ASR 
]
 under different defenses across various datasets.

These results validate that malicious model updates can effectively bypass defenses, be aggregated into the global model, and drive it into a backdoored state.

## 6.8.2. Working Principle Analysis

The working principle of *DP OT*L0in backdoor attack can be explained through the relationship between its ASR and 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384

385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 Ours (Final) FT (Final) DFT (Final)
Ours (Avg) FT (Avg) DFT (Avg)
Trimmed Mean RobustLRRFAFLAIRFLCertFLAME
FoolsGold Multi-Krum 0 50 100 A
S
R

FedAvgMedian
(a) Fashion MNIST
Trimmed Mean RobustLRRFAFLAIRFLCertFLAME
FoolsGold Multi-Krum 0 50 100 A
S
R

FedAvgMedian
(b) FEMNIST
Trimmed Mean RobustLRRFAFLAIRFLCertFLAME
FoolsGold Multi-Krum 0 50 100 A
S
R

FedAvgMedian
(c) CIFAR10 Trimmed Mean RobustLRRFAFLAIRFLCertFLAME
FoolsGold Multi-Krum 0 50 100 A
SR
FedAvgMedian
(d) Tiny ImageNet Figure 4. ASR results for the *DP OT*L0, FT, and DFT.
the duration of the attack. We conducted experiments where the attack was initiated at different training rounds, and ASR was observed at specific subsequent rounds. Table 5a presents the results for Fashion MNIST with Trimmed Mean as the defense strategy. The ASR increases with training duration and prior attack presence. This shows the global model gradually learns the backdoor feature. The optimization of the *DP OT*L0trigger in this scenario enables malicious model updates to bypass the defense mechanism and accelerates the global model's learning of the backdoor feature. This is achieved through the alternating optimization of the model and trigger, both aimed at minimizing backdoor loss. The final ASR results in this case are better than using *DP OT*L0algorithms to perform adversarial attack on an un-attacked model.

Table 5b shows the results for CIFAR10 with FLAIR as the defense strategy. The ASR exhibits little variation regardless of attack duration or the presence of prior attacks. This indicates that the malicious model updates are weakly bypass the defense under current attacking pattern. Consequently, the ASR primarily reflects the residual effects of the trigger on the next-round global model. We found a small difference between the ASR by adversarial and backdoor attacks, indicating that the *DP OT*L0trigger, with limited backdoor features, has good transferability across rounds. A more effective backdoor attacking pattern for this case can be found in Appendix P.

In summary, *DP OT*L0combines pixel-pattern triggers' learnability with adversarial triggers' transferability, demonstrating varied efficacy across conditions.

Table 5. *DP OT*L0for backdoor attack and for adversarial attack.

(a) ASR is dependent to backdoor attack duration.

Backdoor Attack **Adversarial Attack**
Observe at (round):
48.92 1 200 250 280 300 Att ack **starts**
at (
round)
:
1 10.0 76.27 89.52 93.6 95.64 200 - 49.56 84.03 91.26 93.14 250 - - 69.47 81.04 87.86 280 - - - 66.75 74.41
(b) ASR is independent to backdoor attack duration.

Backdoor Attack **Adversarial Attack**
Observe at (round):
56.79 1 100 140 145 150 At tack s tar ts at (round)
:
1 10.0 57.48 65.43 60.72 61.29 100 - 47.54 74.95 64.04 61.14 140 - - 62.62 57.19 59.11 145 - - - 63.63 63.42

## 6.8.3. More Results

Additional results of potential interest to readers are provided in the Appendix. Section J presents experimental evidence guiding trigger size selection for different datasets.

The evolution of the *DP OT*L0 trigger during FL training is visualized in Section K. Ablation studies on the effects of different MCR, trigger size, DPR, Non-iid degree, and attacking patterns on *DP OT*L0's performance are detailed in Sections L, M, N, O, and P, respectively. We also discussed the attack performance of combining *DP OT*L0 with modelpoisoning techniques by relaxing the TEEs constraints in Section Q.

## 7. Conclusion

In this work, we proposed *DP OT*L0, a novel backdoor attack method relying solely on data poisoning in Federated Learning (FL). *DP OT*L0dynamically adjusts the backdoor objective to conceal malicious clients' model updates among benign ones, enabling global models to aggregate them even when protected by state-of-the-art defenses.

## Impact Statement

Ethics Statement: This paper presents work whose goal is to advance the field of Machine Learning. Our paper presents a practical attack on federated learning, which can be executed with minimal technical skill by anyone who can participant into an FL. While this may seem risky, we believe the benefits of disclosing this attack outweigh potential harms. Sharing the limitations of current defense strategies early prevents future misuse in security-critical applications, allowing organizations to address vulnerabilities before widespread deployment.

440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 Reproducibility Statement: To ensure the reproducibility of our results, we have provided detailed descriptions of our experimental setup, including model architectures, hyperparameters, datasets, and training procedures. All code used to implement our attack and run evaluations will be made available after the publication of this paper. Additionally, our code can be easily adapted to other FL research projects by simply integrating our algorithms into the data preparation process of FL clients before the data is input into their training phase. Therefore, our work can be extensively used to evaluate future FL systems for security purposes.

## References

495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 Bagdasaryan, E., Veit, A., Hua, Y., Estrin, D., and Shmatikov, V. How to backdoor federated learning. In 23rd *International* Conference on Artificial Intelligence and Statistics, pp. 2938–2948, 2020.

Baruch, G., Baruch, M., and Goldberg, Y. A little is enough: circumventing defenses for distributed learning. In 32nd*Advances in Neural Information Processing Systems*, 2019.

Blanchard, P., Mhamdi, E. M. E., Guerraoui, R., and Stainer, J. Machine learning with adversaries: byzantine tolerant gradient descent. In 30th*Advances in Neural Information Processing Systems*, 2017.

Cao, X., Fang, M., Liu, J., and Gong, N. Fltrust: byzantine-robust federated learning via trust bootstrapping. In 28thISOC
Network and Distributed System Security Symposium, 2021.

Cao, X., Zhang, Z., Jia, J., and Gong, N. Flcert: provably secure federated learning against poisoning attacks. 17th IEEE
Transactions on Information Forensics and Security, pp. 3691–3705, 2022.

Carlini, N. and Wagner, D. Towards evaluating the robustness of neural networks. In 38th*IEEE Security and Privacy*, pp.

39–57. Ieee, 2017.

Fang, M., Cao, X., Jia, J., and Gong, N. Local model poisoning attacks to byzantine-robust federated learning. In 29th*USENIX Security Symposium*, pp. 1605–1622, 2020.

Fang, M., Zhang, Z., Hairi, Khanduri, P., Liu, J., Lu, S., Liu, Y., and Gong, N. Byzantine-robust decentralized federated learning. In 31st*ACM Conference on Computer and Communications Security*, 2024.

Fang, P. and Chen, J. On the vulnerability of backdoor defenses for federated learning. In 37thAssociation for the Advancement of Artificial Intelligence, number 10, pp. 11800–11808, 2023.

Fereidooni, H., Pegoraro, A., Rieger, P., Dmitrienko, A., and Sadeghi, A. Freqfed: a frequency analysis-based approach for mitigating poisoning attacks in federated learning. In 31st*ISOC Network and Distributed System Security Symposium*,
2024.

Fung, C., Yoon, C. J. M., and Beschastnikh, I. The limitations of federated learning in sybil settings. In *23rd International* Symposium on Research in Attacks, Intrusions and Defenses, pp. 301–316, 2020.

Girshick, R., Donahue, J., Darrell, T., and Malik, J. Rich feature hierarchies for accurate object detection and semantic segmentation. In 38th*IEEE Conference on Computer Vision and Pattern Recognition*, pp. 580–587, 2014.

Gong, X., Chen, Y., Huang, H., Liao, Y., Wang, S., and Wang, Q. Coordinated backdoor attacks against federated learning with model-dependent triggers. *IEEE Network*, 36(1):84–90, 2022.

Goodfellow, I., Shlens, J., and Szegedy, C. Explaining and harnessing adversarial examples. 2015. He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In IEEE conference on Computer Vision and Pattern Recognition, pp. 770–778, 2016.

Kabir, E., Song, Z., Rashid, M. U., and Mehnaz, S. Flshield: a validation based federated learning framework to defend against poisoning attacks. In 45th*IEEE Security and Privacy*, pp. 2572–2590, 2024.

Kumari, K., Rieger, P., Fereidooni, H., Jadliwala, M., and Sadeghi, A. Baybfed: bayesian backdoor defense for federated learning. In 44th*IEEE Security and Privacy*, 2023.

Kurakin, A., Goodfellow, I., and Bengio, S. Adversarial machine learning at scale. 2017. Li, S. and Dai, Y. Backdoorindicator: Leveraging ood data for proactive backdoor detection in federated learning.

33rd*USENIX Security Symposium*, 2024.

Liu, Y., Kang, Y., Zou, T., Pu, Y., He, Y., Ye, X., Ouyang, Y., Zhang, Y., and Yang, Q. Vertical federated learning: concepts, advances, and challenges. *IEEE Transactions on Knowledge and Data Engineering*, 36(7):3615–3634, 2024.

Lyu, X., Han, Y., Wang, W., Liu, J., Wang, B., Liu, J., and Zhang, X. Poisoning with cerberus: stealthy and colluded backdoor attack against federated learning. In 37th*Association for the Advancement of Artificial Intelligence*, pp. 9020–9028, 2023.

550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 McMahan, B., Moore, E., Ramage, D., Hampson, S., and y Arcas, B. A. Communication-efficient learning of deep networks from decentralized data. In 20th*International Conference on Artificial Intelligence and Statistics*, pp. 1273–1282, 2017.

Mo, X., Zhang, Y., Zhang, L. Y., Luo, W., Sun, N., Hu, S., Gao, S., and Xiang, Y. Robust backdoor detection for deep learning via topological evolution dynamics. In 45th*IEEE Security and Privacy*, pp. 171–171, 2024.

Mozaffari, H., Shejwalkar, V., and Houmansadr, A. Every vote counts: ranking-based training of federated learning to resist poisoning attacks. In 32nd*USENIX Security Symposium*, pp. 1721–1738, 2023.

Nguyen, T., Nguyen, T., Tran, A., Doan, K., and Wong, K. Iba: Towards irreversible backdoor attacks in federated learning.

38th*Advances in Neural Information Processing Systems*, 36, 2024.

Nguyen, T. D., Rieger, P., Viti, R. D., Chen, H., Brandenburg, B. B., Yalame, H., Mollering, H., Fereidooni, H., Marchal, S., ¨
Miettinen, M., et al. Flame: taming backdoors in federated learning. In 31st*USENIX Security Symposium*, pp. 1415–1432, 2022.

Ozdayi, M. S., Kantarcioglu, M., and Gel, Y. R. Defending against backdoors in federated learning with robust learning rate.

In 35th*Association for the Advancement of Artificial Intelligence*, number 10, pp. 9268–9276, 2021.

Papernot, N., McDaniel, P., Jha, S., Fredrikson, M., Celik, Z., and Swami, A. The limitations of deep learning in adversarial settings. In 1 st*IEEE Euro S&P*, pp. 372–387. IEEE, 2016.

Pillutla, K., Kakade, S. M., and Harchaoui, Z. Robust aggregation for federated learning. IEEE Transactions on Signal Processing, 70:1142–1154, 2022.

Riege, P., Krauß, T., Miettinen, M., Dmitrienko, A., and Sadeghi, A. Crowdguard: Federated backdoor detection in federated learning. In 31st*ISOC Network and Distributed System Security Symposium*, 2024.

Sandeepa, C., Siniarski, B., Wang, S., and Liyanage, M. Sherpa: explainable robust algorithms for privacy-preserved federated learning in future networks to defend against data poisoning attacks. In 45th*IEEE Security and Privacy*, pp.

204–204, 2024.

Schneider, M., Masti, R. J., Shinde, S., Capkun, S., and Perez, R. SoK: Hardware-supported trusted execution environments.

arXiv preprint arXiv:2205.12742, 2022.

Shafahi, A., Huang, W. R., Najibi, M., Suciu, O., Studer, C., Dumitras, T., and Goldstein, T. Poison frogs! targeted clean-label poisoning attacks on neural networks. In 31st*Advances in Neural Information Processing Systems*, 2018.

Sharma, A., Chen, W., Zhao, J., Qiu, Q., Bagchi, S., and Chaterji, S. Flair: defense against model poisoning attack in federated learning. In 18th*ACM Symposium on Information, Computer and Communications Security*, 2023.

Shi, Z., Wei, J., and Liang, Y. A theoretical analysis on feature learning in neural networks: Emergence from inputs and advantage over fixed features. In 10th*International Conference on Learning Representations*, 2022.

Simonyan, K. and Zisserman, A. Very deep convolutional networks for large-scale image recognition. In 3 rd*International* Conference on Learning Representations, 2015.

Sun, Z., Kairouz, P., Suresh, A. T., and McMahan, H. B. Can you really backdoor federated learning? arXiv preprint arXiv:1911.07963, 2019.

Szegedy, C. Intriguing properties of neural networks. 2014. Wang, B., Yao, Y., Shan, S., Li, H., Viswanath, B., Zheng, H., and Zhao, B. Neural cleanse: Identifying and mitigating backdoor attacks in neural networks. In 40th*IEEE Security and Privacy*, pp. 707–723. IEEE, 2019.

Wang, H., Sreenivasan, K., Rajput, S., Vishwakarma, H., Agarwal, S., Sohn, J., Lee, K., and Papailiopoulos, D. Attack of the tails: yes, you really can backdoor federated learning. In 34th*Advances in Neural Information Processing Systems*,
NIPS'20, 2020.

Xie, C., Huang, K., Chen, P., and Li, B. Dba: distributed backdoor attacks against federated learning. In 8 th*International* Conference on Learning Representations, 2020.

Xie, Y., Fang, M., and Gong, N. Fedredefense: defending against model poisoning attacks for federated learning using model update reconstruction error. In 41st*International Conference on Machine Learning*, 2024.

Yin, D., Chen, Y., Kannan, R., and Bartlett, P. Byzantine-robust distributed learning: towards optimal statistical rates. In 35th*International Conference on Machine Learning*, pp. 5650–5659, 2018.

Zeiler, M. Visualizing and understanding convolutional networks. In *European conference on computer vision/arXiv*,
volume 1311, 2014.

Zhang, H., Jia, J., Chen, J., Lin, L., and Wu, D. A3fl: Adversarially adaptive backdoor attacks to federated learning. In 36th*Advances in Neural Information Processing Systems*, 2024.

Zhang, K., Tao, G., Xu, Q., Cheng, S., An, S., Liu, Y., Feng, S., Shen, G., Chen, P., Ma, S., et al. Flip: a provable defense framework for backdoor mitigation in federated learning. In 11th*International Conference on Learning Representations*,
2023.

Zhang, Z., Panda, A., Song, L., Yang, Y., Mahoney, M., Mittal, P., Kannan, R., and Gonzalez, J. Neurotoxin: durable backdoors in federated learning. In 39th*International Conference on Machine Learning*, pp. 26429–26446, 2022.

605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659

## A. Additional Related Works

A.1. Federated Learning (FL)
The Federated Learning (McMahan et al., 2017) (FL) training process involves four main steps: 1) **Model Distribution**: A central server distributes the most recent global model to the participating clients. 2) **Local Training**: Each client independently trains the global model on its local training dataset and obtains a local model. 3) **Model Updates**: Each client calculates the parameter-wise difference between its local model and the global model, referred to as model updates, and then sends them to the central server. 4) **Aggregation**: The central server aggregates clients' model updates to create a new global model. This entire process, consisting of step 1 to 4, constitutes a global round. The FL system repeats these steps for a certain number of rounds to obtain a final version of the global model. A.2. Backdoor Attacks in FL FL is easily suffered from backdoor attacks. As training data are privately held by clients, the security of data is hard to track or protect. Adversaries can inject backdoors into the global model simply by compromising a few vulnerable client devices and poisoning their data with backdoor triggers. To date, many variations of backdoor attacks targeting FL have emerged, and we summarize those specific to image classification tasks in Figure 5. With model poisoning vs. Without model poisoning

With model poisoning Static objective
(fixed trigger)

Semantic trigger: Bagdasaryan et al. *(2020)* Edge-case backdoor: Wang et al. *(2020)* Artificial trigger Single global trigger:
Sun et al. (2019); Baruch et al. *(2019)*
Distributed trigger: Xie et al. *(2020)*
Dynamic objective (optimized trigger)

Backdoor Attacks in FL

L2-norm bounded trigger: Lyu et al. (2023); Nguyen et al. *(2024)* L0-norm bounded trigger nFixed shape and placement:
Fang & Chen *(2023)*
Without model poisoning

Dynamic objective (optimized trigger)

L0-norm bounded trigger Fixed shape and placement:
Gong et al. (2022); Zhang et al. *(2024)*
Free shape and placement: **Our work**
The foundation of backdoor attacks in FL is through *data poisoning* - attackers embed backdoor triggers into the local training data of certain clients and change the ground-truth labels of the infected data to malicious labels. As a result, clients' local models trained on the poisoned data will be backdoored, and consequently, the global model that aggregates these backdoored models will also be backdoored. A standalone data poisoning is found challenging to succeed when employing some types of triggers. Therefore, many works introduce model poisoning to assist backdoor attacks in FL. *Model poisoning* aims to either directly manipulate clients' model updates or indirectly achieve this by changing their local training algorithms. Three main approaches in model poisoning were widely adopted in existing attacks: 1) Scaling based (Bagdasaryan et al., 2020; Sun et al., 2019; Xie et al., 2020; Gong et al., 2022). Attackers amplify malicious model updates generated from backdoored models before clients send them to the server. These malicious updates can overpower the aggregation results, causing the global model to quickly incorporate backdoors. However, this approach is vulnerable to defenses that exclude outlier model updates from the aggregation. 2) Constraint based (Bagdasaryan et al., 2020; Lyu et al., 2023). Attackers change clients' local training algorithms by adding extra constraints to their loss functions, giving backdoored models specific characteristics, such as being less distinguishable from benign models. 3) Projection based (Zhang et al., 2022; Baruch et al., 2019; Wang et al., 2020; Fang & Chen, 2023). Attackers constrain backdoor implementation to bounded model parameters: by clipping parameter values or using Projected Gradient Descent, backdoor models are L2-norm bounded to a chosen model state; by selectively updating a subset of parameters, they are L0-norm bounded to a chosen state.

Model poisoning requires attackers to modify certain clients' local training procedures. However, with the introduction of Trusted Execution Environments (TEEs) by state-of-the-art defense mechanisms (Riege et al., 2024), client-side execution for training can be authenticated and secure, thus increasing the difficulty of conducting model poisoning. In contrast, data poisoning is easier to conduct and harder to prevent since clients may collect their local data from open resources where attackers can also get access to and make modifications.

## Static Objective Vs. Dynamic Objective

If a backdoor attack has a specified and unchanging objective that is independent to the training system's status, we refer to this as a *static objective*. For instance, Semantic trigger as backdoor (Bagdasaryan et al., 2020) aims to associate certain features from input that is unrelated to the main training tasks with an attacker-chosen output, causing the model to make incorrect predictions on those inputs; Edge-case backdoor (Wang et al., 2020) selects data that share certain commonalities but are from the tail end of the input data distribution as the backdoored input, causing the model to mispredict them; Artificial trigger as backdoor (Sun et al., 2019; Zhang et al., 2022; Baruch et al., 2019; Xie et al., 2020) embeds a few pixels forming a specific artificial pattern into the input, leading the model to mispredict any input containing this pixel pattern. In FL, since the static objectives of backdoor attacks are inconsistent with the optimization objectives defined by the main-task data, malicious models will exhibit distinct differences in their model updates compared to benign models, making them easy to detect. In contrast to a static objective, a backdoor attack that adjusts its objective based on the training system's status is referred to as having a *dynamic objective*. By adjusting its objective, a backdoor attack is expected to achieve greater effectiveness. Several approaches have been proposed in recent attack studies to attempt to accomplish this. For example, Modeldependent attack (Gong et al., 2022) and F3BA (Fang & Chen, 2023) optimized the trigger pattern based on a hypothesis that maximizing the activation of certain neurons in the backdoored local model can enhance the attack's persistence on the global model, which provides preliminary insights into the potential of optimized triggers; A3FL (Zhang et al., 2024), which optimizes triggers specifically for a defense scenario where the global model is directly trained to unlearn the trigger, is another pioneering work exploring the potential of optimized triggers in attacking FL.

## L2-Norm-Bounded Optimized Trigger Vs. L0**-Norm-Bounded Optimized Trigger**

A critical consideration in designing backdoor triggers is ensuring their subtlety when applied to input data, resulting in a trivial disparity between human perception and the backdoored model's interpretation. Existing dynamic objective attacks achieve this by constraining the optimized triggers' L2-norm or L0-norm bounds. An L2*-norm-bound* restricts the total magnitude of the perterbations adding to the data. For example, CerP (Lyu et al.,
2023) and IBA (Nguyen et al., 2024) generate optimized perturbations adds them to clients' local data to induce their local models learn to misclassify the perturbed data to a specified target label.

An L0*-norm bound* restricts the number of components (e.g., pixels in an image) that can be altered by the trigger. For 660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714 example, The optimized trigger in Model-dependent attack (Gong et al., 2022), F3BA (Fang & Chen, 2023), or A3FL (Zhang et al., 2024) consists of a small number of pixels arranged in a square shape, being placed in a certain location on the data. Clean-label attacks Clean-label attacks (Shafahi et al., 2018) involve manipulating input data with subtle perturbations while keeping labels unchanged. Although this assumption aligns with scenarios like Vertical Federated Learning (Liu et al., 2024) (VFL), where participants possess vertically partitioned data with labels owned by only one participant, our study does not consider VFL as our attack scenario. Therefore, discussions of clean-label attacks are beyond the scope of our work.

## A.3. Defenses With Different Privacy-Preserving Properties

Recent defense works have introduced several novel FL pipelines aimed at enhancing the security of FL against various types of attacks. These novel architectures provide different levels of privacy protection and often require additional techniques (e.g., Secured Multi-party Computation) to ensure privacy for FL clients. In light of these privacy considerations, we have chosen to focus our analysis on the conventional FL structure that was originally proposed in the concept of Federated Learning (McMahan et al., 2017). Although defenses built on newly proposed FL structures fall outside the scope of our main comparison, we offer a discussion of these related works in this section. Clients' private data were shared to the server: Some approaches allow the server to have access to a small portion of main-task data shared by clients. To mitigate backdoor attacks, server-side defense strategies use this data to either independently train a model and use its updates as a reference for each round of aggregation (e.g., FLTrust (Cao et al., 2021)), or to validate clients' model updates and eliminate those with abnormal outputs (e.g., SSDT (Mo et al., 2024), SHERPA (Sandeepa et al., 2024)). However, both of these methods still rely on analyzing clients' model updates, making them vulnerable to backdoor attacks with dynamic objectives that conceal malicious updates. FedREdefense (Xie et al., 2024) detects and filters out artificial model updates by reconstructing distilled data shared by clients, but this approach is not effective against backdoor attacks where malicious clients genuinely train their models on poisoned local data rather than fabricating model updates. Clients' model updates were shared to each other: Some approaches propose allowing clients to share their model updates with one another, rather than just with the server. CrowdGuard (Riege et al., 2024) and FLShield (Kabir et al., 2024) suggest that a subset of clients validate other clients' model updates using their own data, assuming that malicious model updates would produce abnormal outputs on benign data. However, this hypothesis fails when malicious model updates are trivially different from non-attacked model updates, a state that can be achieved through using optimized triggers. Fang et al. (2024) proposed a decentralized FL framework without a central server, where clients exchange model updates and apply Byzantine-robust aggregation using their own updates as a reference. Like other defenses that rely on analyzing clients' model updates, this approach is also vulnerable to backdoor attacks with optimized triggers.

## B. Theoretical Analysis

B.1. Proof of Proposition 5.1 Proof. We define the least-squares optimization objectives for fK and fK∪Kadv :

$$f_{K}=\frac{1}{2}\|K^{T}w-y^{T}\|_{2}^{2}$$
2(1)
715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769

$$f_{K\cup K_{a d v}}=\frac{1}{2}\|[K\ K_{a d v}]^{T}w-[y\ y_{a d v}]^{T}\|_{2}^{2}.$$
. (2)
$$(\mathbf{l})$$

14 The gradients with respect to w are:
Let ϵadv represent the error of (Kadv, yadv) on the model w:
770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 Substituting ϵadv into the gradient, we get:
The difference in gradients is:
where δ = max ∥vi∥, vi ∈ Kadv.

Thus, when ϵadv = 0, the update directions for fK and fK∪Kadv are identical. Otherwise, the difference is bounded by δ∥ϵadv∥, quantifying the influence of the adversarial error.

## C. Experimental Settings D. Descriptions Of Defenses

$${\frac{\partial f_{K\cup K_{a d v}}}{\partial w}}=K(K^{T}w-y^{T})+K_{a d v}\epsilon_{a d v}.$$
$$\Delta={\frac{\partial f_{K\cup K_{a d v}}}{\partial w}}-{\frac{\partial f_{K}}{\partial w}}=K_{a d v}\epsilon_{a d v}.$$

Writing ϵadv ∈ R
pas ϵadv = (e1, e2*, . . . e*p) and Kadv ∈ R
n×pas Kadv = (v1, v2*, . . . , v*p), where vi ∈ R
n, the magnitude of ∆ is bounded as:

$\|\Delta\|=\|K_{adv}\epsilon_{adv}\|=\|e_{1}v_{1}+e_{2}v_{2}+\cdots+e_{p}v_{p}\|\leq\sum_{i=1}^{p}\|e_{i}v_{i}\|\leq\delta\|\epsilon_{adv}\|,$
$$(3)$$
$\square$
Finally, the update directions ∆wK and ∆wK∪Kadv for minimizing the objective 1 and 2, defined as the negative gradients, satisfy:
∥∆wK∪Kadv − ∆wK∥ = ∥∆∥ ≤ δ∥ϵadv∥.

| Table 6. Dataset description   |             |          |                               |          |      |
|--------------------------------|-------------|----------|-------------------------------|----------|------|
| Dataset                        | #class #img | img size | Model                         | #params  |      |
| Fashion MNIST                  | 10          | 70k      | 28 × 28 grayscale 2 conv 3 fc | ∼1.5M    |      |
| FEMNIST                        | 62          | 33k      | 28 × 28 grayscale 2 conv 2 fc | ∼6.6M    |      |
| CIFAR10                        | 10          | 60k      | 32 × 32 color                 | ResNet18 | ∼11M |
| Tiny ImageNet                  | 200         | 100k     | 64 × 64 color                 | VGG11    | ∼35M |

Twelve different server-side defense strategies, based on analyzing clients' model updates, are briefly introduced below:
FedAvg (McMahan et al., **2017)**, a basic aggregation rule in FL, computes global model updates by averaging all clients' model updates.

$${\frac{\partial f_{K}}{\partial w}}=K(K^{T}w-y^{T}),$$
$$\partial f_{K\cup K_{adv}}=[K\;K_{adv}]([K\;K_{adv}]^{T}w-[y\;y_{adv}]^{T})$$ $$=(KK^{T}+K_{adv}K_{adv}^{T})w-(Ky^{T}+K_{adv}y_{adv}^{T})$$ $$=K(K^{T}w-y^{T})+K_{adv}(K_{adv}^{T}w-y_{adv}^{T}).$$
$$\epsilon_{a d v}=K_{a d v}^{T}w-y_{a d v}^{T}.$$

825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879

| Table 7. Default settings                                                    |    |     |     |      |     |
|------------------------------------------------------------------------------|----|-----|-----|------|-----|
| Trigger Size Round Number of Clients Malicious Client Ratio Data Poison Rate |    |     |     |      |     |
| Fashion MNIST                                                                | 64 | 300 | 100 |      |     |
| FEMNIST                                                                      | 25 | 200 | 100 | 0.05 | 0.5 |
| CIFAR10                                                                      | 25 | 150 | 50  |      |     |
| Tiny ImageNet                                                                | 64 | 100 | 50  |      |     |

Median (Yin et al., **2018)**, a simple but robust alternative to FedAvg, constructs the global model updates by taking the median of the values of model updates across all clients Trimmed Mean (Yin et al., **2018)**, in our implementation, excludes the 40% largest and 40% smallest values of each parameter among all clients' model updates and takes the mean of the remaining 20% as the global model updates. Multi-Krum (Blanchard et al., **2017)**, in our implementation, identifies 10% honest client whose model updates have the smallest Euclidean distance to all other clients' model updates and takes the average of these honest clients' model updates as the global model updates. RobustLR (Ozdayi et al., **2021)** adjusts the aggregation server's learning rate, per dimension and per round, based on the sign information of clients' updates. RFA (Pillutla et al., **2022)** computes a geometric median of clients' model updates and assigns weight factors to clients depending on their distance from the geometric median. Subsequently, it computes the weighted average of all clients' model updates to generate the global model updates. FLAIR (Sharma et al., **2023)** assigns different weight factors to clients according to the similarity of the coefficient signs between client model updates and global model updates of the previous round, and then takes the weighted average of all clients' model updates to form the global model updates. The weight factors are carried over and accumulate from the previous round. FLCert (Cao et al., **2022)** randomly groups clients into 5 clusters, computes the median of model updates within each cluster, and uses the majority inference outcomes of these cluster models as the final results. FLAME (Nguyen et al., **2022)** first clusters clients' model updates according to their cosine similarity to each other, and then aggregates the clipped model updates within the largest cluster as the global model updates. FoolsGold (Fung et al., **2020)** reduces aggregation weights of a set of clients whose model updates constantly exhibit high cosine similarity to each other. BackdoorIndicator (Li & Dai, **2024)** trains an indicator model using OOD datasets to serve as the global model, then filtering out clients' model updates if their accuracy on those OOD datasets greater than a threshold. FRL (Mozaffari et al., **2023)** is a defense strategy where the server sparsifies the value space of model updates, allowing clients to vote on the most effective model updates based on their local data. The server then aggregates only the accepted votes while rejecting outliers to construct the global model.

## E. Main-Task Accuracy Results Corresponding To Figure 4

Table 8 lists the Main-task Accuracy of each experiment in getting results in Figure 4. Table 8 demonstrates that for different datasets used as the main tasks, global models under various attacks maintained a comparable level of Main-task Accuracy to the baselines with no attacks ("None"), indicating that all types of backdoor attacks successfully achieved their main-task convergence goals.

## F. Aggregation Of Malicious Model Updates

In this section, we analyzed the attack effectiveness of each component of the *DP OT*L0attack's working principles and report evidence that it effectively conceals malicious clients' model updates, thereby getting them integrated into the global models through aggregation. Table 8. The Main-task Accuracies (MA) correspond to results in Figure 4. "None" represents no attack existing in the FL training.

MA Tiny ImageNet Fashion MNIST FEMNIST **CIFAR10**

None Ours FT DFT None Ours FT DFT None Ours FT DFT None Ours FT DFT

FedAvg 43.9 43.5 43.0 43.3 86.7 87.3 86.7 86.8 82.2 81.4 83.3 82.3 70.3 70.7 70.4 71.4 Median 40.6 40.2 40.6 38.6 86.0 85.8 86.6 86.3 80.4 81.5 79.8 79.9 70.2 69.1 69.8 69.7

Trimmed Mean 40.8 40.4 40.1 40.6 86.4 85.8 86.4 86.3 80.2 81.7 81.3 81.2 69.4 70.4 70.2 70.8

RobustLR 44.1 42.7 42.9 43.2 86.5 86.8 86.6 86.9 81.8 82.5 81.9 82.6 70.4 70.1 70.3 70.5

RFA 43.6 43.0 43.0 43.0 86.4 86.0 87.1 87.1 83.0 80.7 81.0 80.8 70.4 70.7 70.3 70.8

FLAIR 43.6 42.6 41.8 42.1 86.1 84.9 85.2 84.4 81.5 80.7 80.6 79.7 70.3 70.6 71.0 70.4 FLCert 40.3 40.2 39.7 39.7 86.2 85.9 86.0 86.8 81.3 80.9 81.5 81.0 69.6 70.0 69.8 70.4

FLAME 29.9 28.7 29.2 28.9 86.4 86.4 86.4 86.7 81.8 80.2 80.7 81.0 70.1 70.3 70.9 70.9

FoolsGold 43.1 43.2 43.5 43.2 86.6 87.1 86.8 87.3 83.4 82.7 83.0 81.8 70.4 71.0 71.2 71.7

Multi-Krum 30.7 27.7 27.7 26.4 86.2 85.9 86.0 87.0 79.9 80.4 79.6 80.2 61.4 63.0 63.2 60.8

In the i-th round, *DP OT*L0 generates a trigger τ
(i) by optimizing its shape, placement and values to make the global model of this round Wg
(i)achieve a maximum ASR. However, what we were truly interested in is its ASR on the global model after the i-th round aggregation, which is the next-round global model denoted as Wg
(i+1). The attack effectiveness of the trigger τ
(i) on the global model Wg
(i+1) stems from two factors:
1. **Trigger Optimization**: Trigger optimization using Wg
(i)results in an improvement of the trigger's ASR on Wg
(i+1)
due to the small difference between Wg
(i+1) and Wg
(i).

880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 2. **Aggregation of Backdoored Model Updates**: Model updates that were trained on data partially poisoned by τ
(i)exhibit small differences from those were trained on data without poisoning. Therefore, they bypassed defenses and made Wg
(i+1) incorporate backdoored model parameters.

In the following, we explain how we designed experiments to study the impact of each factor, and analyzed the experiment results. Experiment design: To assess the attack effectiveness solely brought by Trigger Optimization, we eliminated any effects produced by data poisoning. Specifically, we set all clients in the FL system to be benign, ensuring that the next-round global model, denoted as Wf(i+1)
g , aggregated benign model updates only. In the meantime, we still collected data from a certain number of clients and optimized a trigger τe
(i)for Wf(i)
g . Then, we tested Wf(i+1)
g on a testing dataset in which all images are poisoned with the trigger τe
(i)to obtain an ASR ]. This ASR ] evaluates the attack effectiveness achieved by the current-round optimized trigger τ
(i) on the next-round global model Wf
(i+1)
g , which does not contain any model updates learned from backdoor information. To assess the attack effectiveness brought by Aggregation of Backdoored Model Updates, we introduced malicious clients into the FL system and therefore the global model, denoted as W¨
(i+1)
g , was allowed to aggregate model updates submitted by malicious clients. In this system, malicious clients partially poisoned their local training data (aligning with default settings in Table 7) using the trigger τ¨
(i)that was optimized for W¨
(i)
g , and then conducted their local training. We tested the W¨
(i+1)
g on the testing dataset that was also poisoned by τ¨
(i)to obtain an ASR¨ . We evaluated the attack effectiveness of Aggregation of Backdoored Model Updates by measuring the increase in ASR compared to the previous setting, calculated as (ASR¨ − ASR ]). This metric reveals how much the malicious clients' model updates influenced the global model W¨
(i+1)
g to achieve a higher ASR compared to Wf
(i+1)
g .

Experiment results: Table 9 shows results of ASR ] and ASR¨ over 10 different defense methods. We used same settings as in Table 7 for testing ASR¨ , and kept the size of trigger training dataset consistent when testing ASR ].

The results of ASR 
]
 in Table 9 show that different defense methods resulted in very different ASR 
]
 even for the same learning task of a dataset. The reason for the variance of ASR 
]
 is the gap between Wg
(i)and Wf
(i+1)
g were different when implementing different defense methods. According to recent studies (Lyu et al., 2023; Zhang et al., 2024), if the gap between consecutive rounds of global models in an FL system is smaller, Trigger Optimization will be more effective in its attack. The results of ASR¨ in Table 9 show that the presence of malicious clients' model updates consistently enhances ASR
compared to ASR 
]
 across all defense methods on different datasets. We consider this enhancement as an evidence of the statement that the attack effectiveness of *DP OT*L0 comes from both Trigger Optimization and Aggregation of Backdoored

Table 9. ASR under different attacking conditions. ASR ] assesses the attack effectiveness of "Trigger Optimization" alone, while ASR¨

assesses the combined effectiveness of both "Trigger Optimization" and "Aggregation of Backdoored Model Updates".

Fashion

MNIST FEMNIST CIFAR10

ASR type Final Avg Final Avg Final Avg

FedAvg ASR 

] 58.8 45.1 54.0 28.6 55.6 50.9

ASR¨ 97.7 69.1 99.7 92.9 **100 98.5**

Median ASR 

] 57.9 38.2 18.0 17.5 56.6 48.7

ASR¨ 97.8 61.7 95.4 81.2 **100 96.1**

Trimmed ASR ] 31.6 29.7 24.2 25.6 55.6 40.9

Mean ASR¨ 94.4 56.0 95.2 84.3 **100 88.6**

RobustLR ASR 

] 70.2 47.2 28.8 27.3 60.1 47.3

ASR¨ 99.2 62.8 99.3 93.0 **100 98.6**

RFA ASR 

] 78.0 46.4 18.9 13.4 57.4 46.1

ASR¨ 97.7 62.0 98.3 95.9 **100 97.8**

FLAIR ASR 

] 42.2 36.2 23.0 29.6 54.1 45.9

ASR¨ 85.3 50.1 88.7 72.7 **62.3 50.7**

FLCert ASR 

] 49.6 39.7 27.7 34.6 48.7 46.7

ASR¨ 95.2 57.9 97.1 86.7 **99.2 88.3**

FLAME ASR 

] 38.0 26.2 34.7 35.7 28.1 51.0

ASR¨ 71.1 43.4 99.2 86.1 **59.8 56.1**

Fools ASR ] 54.2 50.3 57.0 43.7 35.5 35.6

Gold ASR¨ 98.9 68.5 99.6 95.2 **100 98.5**

Multi-Krum ASR 

] 60.6 45.4 31.7 28.7 49.7 36.1

ASR¨ 99.9 63.6 99.7 92.0 **100 98.7**

Model Updates, with the latter one playing a critical role in producing a high ASR¨ .

A general hypothesis made by the state-of-the-art defenses against backdoor attacks in FL is that malicious clients' model updates have a distinct divergence from benign clients' model updates. However, as indicated by the results in Table 9, DP OTL0effectively conceals the model updates from malicious clients amidst those of benign clients, eluding detection and filtering by state-of-the-art defenses. Consequently, defenses formulated based on this broad hypothesis will inherently struggle to defend against *DP OT*L0 attacks.

## G. Evaluation Of Dp Otl0 Attack Against Flip (Zhang Et Al., **2023)**

Flip (Zhang et al., 2023) is a client-side defense strategy where benign clients perform trigger inversion and adversarial training using their local data to recover the global model from backdoors. In this section, we evaluate the effectiveness of the *DP OT*L0attack against the Flip defense. We implemented the *DP OT*L0attack by modifying the data preparation approach in Flip's open-source project, replacing it with the method used in this work, and injecting our data-poisoning algorithms into a subset of clients. Additionally, as *DP OT*L0is a pure data-poisoning attack, we removed any additional steps in their project specified to malicious clients but not existed in benign clients' training, to ensure consistency between malicious clients and benign clients in FL training. We selected Fashion MNIST as the main-task dataset for our evaluation and directly adopted Flip's default experiment settings provided in their project - the total number of clients was 100 and 4% of them were malicious clients; the aggregation rule was set to FedAvg; the global model's parameters were initialized by a pre-trained state. The size of *DP OT*L0trigger was set to 64, consistent with our default attacking settings. We compared the performance of the *DP OT*L0attack under two attack patterns provided by Flip's project: 1) **Single shot**:
Each of the 4 malicious clients conducts a one-time attack at the beginning of training. 2) **Continuous**: All 4 malicious clients continuously execute the attack algorithms in every round during training.

Figure 6 shows the performance of the *DP OT*L0attack on an FL system using Flip as its defense, measured by the Attack Success Rate (ASR). In the single-shot attack pattern, *DP OT*L0 maintains a stable ASR of around 15% across all training rounds, exceeding the random guess accuracy of 10% for the 10-class dataset. In the continuous attack pattern, DP OTL0achieves a significant ASR, peaking at 80.03% during training and stabilizing around 40%, which is higher than the single-shot pattern. These results indicate that Flip is vulnerable to optimized triggers with varying appearances across different rounds, because recovering from backdoors is an after-effect strategy which is unable to stop new and distinct backdoors from injecting into the model.

Figure 7 illustrates the global model's performance on the main task data when using Flip as a defense while under DP OTL0 935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989 attack. We observed that employing Flip reduces the global model's main-task performance compared to not using it. In our baseline experiment on Fashion MNIST, with the same data distribution and aggregation rule (FedAvg), the model achieved an 86.7% MA. However, Flip's global model achieved only 82.8% MA at its best by the end, even with pre-trained model initialization. Additionally, under continuous attack by the *DP OT*L0trigger, the global model's MA further declined compared to the less frequent attack pattern. This raises concerns about Flip's ability to maintain stable and normal performance on the main-task while effectively defending against attacks. In summary, Flip represents an early effort to explore client-side defenses that do not rely on analyzing clients' model updates. While it demonstrates better defense effectiveness against *DP OT*L0compared to server-side defenses, concerns about its potential impact on main-task convergence warrant further investigation.

Global model's accuracy on backdoor task (*DP OT*L0)

## H. Evaluation Of Dp Otl0 Attack Against Frl (Mozaffari Et Al., **2023)**

FRL (Mozaffari et al., 2023) is a defense strategy where the server sparsifies the value space of model updates, allowing clients to vote on the most effective model updates based on their local data. The server then aggregates only the accepted votes while rejecting outliers to construct the global model. In this section, we evaluate the effectiveness of the *DP OT*L0 attack against the FRL defense. Similar to the experiment on Flip, we implemented our attack on FRL's open-source project by injecting our data-poisoning algorithms into a portion of clients' execution and removing any inconsistent steps that distinguished malicious clients from benign ones during training. We used FRL's default settings, in which only 2% of clients were malicious, and tested our attack on the CIFAR10 dataset as the main training task.

Table 10 presents the performance results of the *DP OT*L0attack on an FL system employing FRL as the defense method. The ASR of *DP OT*L0(92.5%) is significantly higher than that of other backdoor attack approaches tested and discussed in FRL's paper. This indicates that FRL, which relies on analyzing clients' model updates, is vulnerable to our attack. The evaluation results also demonstrate that the *DP OT*L0attack is more advanced than backdoor attacks with static objectives when targeting the FRL defense strategy.

## I. Evaluation Of Dp Otl0 Attack Against Backdoorindicator (Li & Dai, **2024)**

We conducted experiments with different learning rates to demonstrate *DP OT*L0's attack effectiveness against BackdoorIndicator, comparing it to Fixed pixel-pattern Triggers (FT).

| Table 10. Comparison results on CIFAR10. Attacks ASR Semantic backdoor attacks 49.2 Artificial backdoor attacks 0 Edge-Case backdoor attacks 64.6 DP OTL0 backdoor attacks 92.5   |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

990 991 992 993 994 995 996 997 998 999 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 Figure 6. Global model's Attack Success Rate under *DP OT*L0 attack when employed Flip as defense strategy. (Fashion MNIST)

Global model's accuracy on main task 0 20 40 60 80 100 65 70 75 80 85 Ac cur acy (%)
Single shot Continuous Rounds 0 20 40 60 80 100 0 20 40 60 80 100 Single shot Continuous Ac cur acy (
%
)

Rounds
1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079 1080 1081 1082 1083 1084 1085 1086 1087 1088 1089 1090 1091 1092 1093 1094 1095 1096 1097 1098 1099

| Table 11. Comparison of DP OTL0 and FT's ASR against BackdoorIndicator. Learning Rate 0.01 0.025 0.05 Fixed pixel-pattern (Final ASR) 10.7 23.3 26.3 DPOT (Final ASR) 100 99.9 99.9 DPOT (Avg ASR) 70.5 89.6 91.2   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

As shown in the table above, the *DP OT*L0trigger maintains a significant Final ASR (> 50%) against BackdoorIndicator across different learning rates and outperforms FT. We observe that BackdoorIndicator's defense effectiveness improves with smaller learning rates, consistent with the results in its original paper.

## J. Trigger Size Selection

We determined the size of the *DP OT*L0trigger for each FL task by balancing its subtlety with achieving an effective ASR.

A trigger's subtlety was evaluated by measuring the accuracy drop it caused when an un-attacked model predicted poisoned images into their original benign labels.

An un-attacked model with the same architecture as the victim FL system's model was used to assess the accuracy drop. The results for different datasets are presented in Table 12.

| Table 12. Impact of DP OTL0 trigger size on un-attacked models' accuracy Trigger size 0 25 64 100 Fashion-MNIST Clean label acc 85.76 79.32 76.07 70.53 Drop (%) 0 7.5 11.30 17.76 FEMNIST Clean label acc 81.24 68.11 45.12 28.39 Drop (%) 0 16.16 44.46 65.05 CIFAR10 Clean label acc 70.81 52.98 35.90 25.06 Drop (%) 0 25.18 49.30 64.61 Tiny-ImageNet Clean label acc 43.44 42.32 35.89 29.53 Drop (%) 0 2.58 17.38 32.02   |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

- Benign acc: accuracy of poisoned data being predicted to its original bengin label. - Drop (%): Benign acc drop compared to when testing clean data (trigger size is 0) on the same un-attacked model.

We established a 30% upper limit for the acceptable accuracy drop and a minimum final ASR effectiveness threshold of 50%. The smallest trigger size meeting both criteria was chosen. Notably, Table 12 reveals that the sensitivity of accuracy drop to trigger sizes varies across datasets and model architectures.

## K. Visualization Of Triggers

K.1. FT, DFT, and *DP OT*L0 triggers on Tiny ImageNet images We displayed FT, DFT, and *DP OT*L0triggers on images from the Tiny ImageNet dataset in Figures 8, 9, and 10. K.2. *DP OT*L0**triggers on images from different datasets.** We displayed *DP OT*L0triggers generated for different datasets in Figure 11.

K.3. Trigger evolution during training In Figure 14 and Figure 15, we demonstrated how *DP OT*L0trigger changes during the FL training.

In Figure 14, we showed one screenshot of the trigger on a blank background in the same size of the cifar10's figure for every ten global rounds. These trigger screenshots were collected during a *DP OT*L0attacking experiment that trains ResNet18 as the global model on the CIFAR-10 dataset, with Trimmed Mean used as the aggregation rule. Figure 12 displays the MA and ASR of the global model over 150 global rounds in this experiment.

Similarly, in Figure 15 we showed one screenshot of the trigger on a blank background in the same size of the Tiny