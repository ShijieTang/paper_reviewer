000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 Anonymous authors Paper under double-blind review

## Abstract

Recent studies have shown that as training progresses, neural networks gradually lose their capacity to learn new information, a phenomenon known as plasticity loss. An unbounded weight growth is one of the main causes of plasticity loss. Furthermore, it harms generalization capability and disrupts optimization dynamics. Re-initializing the network can be a solution, but it results in the loss of learned information, leading to performance drops. In this paper, we propose Soft Weight Rescaling (SWR), a novel approach that prevents unbounded weight growth without losing information. SWR recovers the plasticity of the network by simply scaling down the weight at each step of the learning process. We theoretically prove that SWR bounds weight magnitude and balances weight magnitude between layers. Our experiment shows that SWR improves performance on warm-start learning, continual learning, and single-task learning setups on standard image classification benchmarks.

## 1 Introduction

Recent works have revealed that a neural network loses its ability to learn new data as training progresses, a phenomenon known as plasticity loss. A pre-trained neural network shows inferior performance compared to a newly initialized model when trained on the same data (Ash & Adams, 2020; Berariu et al., 2021). Lyle et al. (2024b) demonstrated that unbounded weight growth is one of the main causes of plasticity loss and suggested weight decay and layer normalization as solutions. Several recent studies on plasticity loss have proposed weight regularization methods to address this issue (Kumar et al., 2023; Lewandowski et al., 2023; Elsayed et al., 2024). Unbounded weight growth is a consistent problem in the field of deep learning; it is problematic not only for plasticity loss but also undermines the generalization ability of neural networks (Golowich et al., 2018; Zhang et al., 2021) and their robustness to distribution shifts. Increasing model sensitivity, where a small change in the model input leads to a large change in the model output, is also closely related to the magnitude of the weights. Therefore, weight regularization methods are widely used in various areas of deep learning and have been consistently studied. Weight regularization methods have been proposed in various forms, including additional loss terms (Krogh & Hertz, 1991; Kumar et al., 2023) and re-initialization strategies (Ash & Adams, 2020; Li et al., 2020b; Taha et al., 2021). The former approach adds an extra loss term to the objective function, which regularizes the weights of the model. These approaches are used not only to penalize large weights but also for other purposes, such as knowledge distillation (Shen et al., 2024). However, they can cause optimization difficulties or conflict with the main learning objective, making it harder for the model to converge effectively (Ghiasi et al., 2024). Liu et al. (2021) also proved that the norm penalty of a family of weight regularizations weakens as the network depth increases. Moreover, such methods require additional gradient computations, resulting in slower training. In addition, several studies argued that regularization methods could be problematic with normalization layers. For instance, weight decay destabilizes optimization in weight normalization (Li et al.,
2020a), and interferes learning with batch normalization (Lyle et al., 2024b), both of which can hinder convergence. On the other hand, re-initialization methods are aimed at resetting certain parameters of the model during training to escape poor local minima and encourage better exploration of the loss landscape. Zaidi et al. (2023) demonstrated that re-initialization methods improve generalization even with modern training protocols. While re-initialization methods improve generalization ability, they raise the problem of losing knowledge from previously learned data (Zaidi et al., 2023;

# Recovering Plasticity Of Neural Networks Via Soft Weight Rescaling

1 Ramkumar et al., 2023; Lee et al., 2024; Shin et al., 2024). It leads to a notable performance drop, especially problematic when access to the previous data is unavailable. In this paper, we propose a novel weight regularization method that has advantages of both of those two approaches. Our method, Soft Weight Rescaling (SWR), directly reduces the weight magnitudes close to the initial values by scaling down weights. With a minimal computational overhead, it effectively prevents unbounded weight growth. Unlike previous methods, SWR recovers plasticity without losing information. In addition, our theoretical analysis proves that SWR bounds weight magnitude and balances weight magnitude between layers. We evaluate the effectiveness of SWR on standard image classification benchmarks across various scenarios—including warm-start learning, continual learning, and single-task learning—comparing it with other regularization methods and highlighting its advantages, particularly in the case of VGG-16. The contributions of this work are summarized as follows. First, We introduce a novel method that effectively prevents unbounded weight growth while preserving previously learned information and maintaining network plasticity. Second, we provide a theoretical analysis demonstrating that SWR bounds the magnitude of the weights and balances the weight magnitude across layers without degrading model performance. Finally, we empirically show that SWR improves generalization performance across various learning scenarios.

The rest of this paper is organized as follows. Section 2 reviews studies on weight magnitude and regularization methods. In Section 3, we explain weight rescaling and propose a novel regularization method, Soft Weight Rescaling. Then, in Section 4, we evaluate the effectiveness of Soft Weight Rescaling by comparing it with other regularization methods across various experimental settings.

## 2 Related Works

054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 Unbounded Weight Growth. There have been studies associated with the weight magnitude. Krogh & Hertz (1991); Bartlett (1996) indicated that the magnitude of weights is related to generalization performance. Besides, as the magnitude of the weights increases, the Lipschitz constant also tends to grow (Couellan, 2021). This leads to higher sensitivity of the network, potentially affecting its stability and generalization. Ghiasi et al. (2024) demonstrated that weight decay plays a role in reducing sensitivity for noise. Moreover, Lyle et al. (2024b) claimed that unbounded weight growth is one of the factors of plasticity loss in training with non-stationary distribution. These studies indicate that enormous weight magnitudes disturb effective learning. Unfortunately, weight growth is inevitable in deep learning. Neyshabur et al. (2017) showed that when the training error converges to 0, the weight magnitude gets unbounded. Merrill et al. (2020) observed that weight magnitude increases with O(
√t), where t is the update step during transformer training. These explanations highlight the ongoing need for weight regularization in modern deep learning. Weight Regularization. Various methods have been proposed to regularize the weight magnitude. L2 regularization, which is also termed as weight decay, is a method to apply an additional loss term that penalizes the L2 norm of weight. Although it is a method widely used, several studies pointed out its problems (Ishii & Sato, 2018; Liu et al., 2021). Yoshida & Miyato (2017) suggested regularizing the spectral norm of the weight matrix and showed improved generalization performance in various experiments. Kumar et al. (2020) regularized the weights to maintain the effective rank of the features. On the other hand, several studies have explored how to utilize the initialized weights. Kumar et al. (2023) imposed a penalty on L2 distance from initial weight and Lewandowski et al. (2023) proposed using the empirical Wasserstein distance to prevent deviating from initial distribution. However, these methods require additional gradient computations. Re-initialization methods. Ash & Adams (2020) demonstrated that a pre-trained neural network achieves reduced generalization performance compared to a newly initialized model. The naive solution is to initialize models and train again from scratch whenever new data is added, which is very inefficient. Based on the idea that higher layers learn task-specific knowledge, methods that re-initialize the model layer by layer, such as resetting the fully-connected layers only (Li et al., 2020b), have been proposed. To explore a more efficient approach, several attempts have been made to re-initialize the subnetwork of the model (Han et al., 2016; Taha et al., 2021; Ramkumar et al., 2023; Sokar et al., 2023). In particular, Ramkumar et al. (2023) calculated the weight importance and re-initialized the task-irrelevant parameters. Sokar et al. (2023) proposed to reset dormant nodes 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 which do not influence the model. However, these methods pose a new drawback in additional computational cost. On the other hand, there have been presented weight rescaling methods that leverage initial weight. Alabdulmohsin et al. (2021) proposed the Layerwise method which rescales the first t blocks to have their initial norms and re-initializes all layers after t-th layer, for the training stage t. More recently, Niehaus et al. (2024) introduced the Weight Rescaling method, which rescales weight to enforce the standard deviation of weight to initialization. The limitation of these two weight rescaling methods is that they depend on the model architecture and require to find a proper rescaling interval.

## 3 Method

In this section, we introduce the proportionality of neural networks to explain a weight regularizing method that preserves the behavior of the model. Next, we demonstrate that our method, SWR, regularizes learnable parameters while satisfying the property. Finally, we will discuss the reason for the importance of the proportionality and advantage of SWR that improves model balancedness.

## 3.1 Notations

Let fθ be a neural network with L layers and activation function ϕ, where the input x ∈ R
m and the output z ∈ R
n. The set of learnable parameters is denoted by θ, comprising the weight matrices Wl and bias vectors bl of the l-th layer. Let al represent the vector of activation outputs of the l-th layer, and zlthe pre-activation outputs before applying the activation function. The final output of the network z = fθ(x) is obtained recursively as follows:

$$a_{0}\doteq x$$
$$-\;b_{L},$$
$$z=W_{L}a_{L-1}+b_{L},$$
$$a_{i}=\phi(z_{i}),\quad i\in\{1,...,L-1\}$$
$\mathbb{Z}_{i}=W_{i}a_{i-1}+b_{i}$, $i\in\{1,...,L-1\}$.  
where zL = z.

For convenience, the norm expression of a matrix will be considered an element-wise L2 norm, which is known as the Frobenius norm: ∥W∥
.= ∥W∥F =
qPi Pj |wij | 2, where wij represents an element of the matrix W. Additionally, we consider multiplying a constant by a matrix or vector as element-wise multiplication.

## 3.2 Weight Rescaling

Previous studies have suggested regularizing the magnitude or spectral norm by multiplying the parameters by a specific constant (Huang et al., 2017; Ash & Adams, 2020; Gogianu et al., 2021; Gouk et al., 2021; Niehaus et al., 2024). However, rescaling the weights can alter the behavior of models, except in specific cases (e.g. a neural network without biases). It is clear that when a constant is multiplied by the weight matrix and bias of the final layer, the network output will be scaled accordingly. However, it becomes complicated when the scaling constant varies across layers. To resolve this complexity, we demonstrate in Theorem 1 that it is possible to avoid decreasing the model's accuracy by employing a specific scaling method. We will first outline the relevant properties in the form of Definition 1.

Definition 1 (Proportionality of neural network). *Let the neural network* fθ
′ *have the same input* and output dimension with fθ*. Then, we say that* fθ
′ and fθ *are proportional if and only if* for a real constant k and all input data x. We refer to the constant k *as the proportionality constant* of fθ and fθ
′ .

We investigated the following theorem shows that it is always possible to construct a proportional network for any arbitrary neural network.

fθ′ (x) = k · fθ(x)
162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 Theorem 1. Let fθ be a feed-forward neural network with affine, convolution layers, and homogeneous activation functions (e.g. ReLU, Leaky ReLU, etc.). For any positive real number C*, we can* find infinitely many networks that are proportional to fθ *with proportionality constant* C. We will briefly explain how to find the network that is proportional to fθ. Let a network that has L layers be fθ, and a set c = {c1, c2*, . . . , c*L} consisting of positive real numbers such that C =
ΠL
i=1ci. Then, construct the new parameter set θ c.= {Wc 1, bc1, . . . WcL, bcL} by rescaling parameters with the following rules:

$$W_{l}^{c}\gets c_{l}\cdot W_{l},\quad b_{l}^{c}\leftarrow\left(\prod_{i=1}^{l}c_{i}\right)\cdot b_{l}$$

Then, for all input x, it satisfies fθ c (x) = Cfθ(x). A detailed proof can be found in Appendix A.

In the following, scaled network fθc , final cumulative scaler C, and the scaler set c will refer to the definitions provided above. Note that Theorem 1 indicates that two proportional neural networks have identical behavior in classification tasks. This suggests that scaling the bias vectors according to a certain rule allows for regularization without affecting the model's performance. It remains the same for the case of any homogeneous layer, such as max-pooling or average-pooling.

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0 Scaling Ratio 0.60 0.62 0.64 0.66 0.68 0.70 0.72 0.74 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0 Scaling Ratio 0.60 0.62 0.64 0.66 0.68 0.70 0.72 0.74 Test Ac cura cy Test Ac cura cy Pre-trained Fine-tuned After Scaling Scaled
Figure 1: **An illustrative comparison of the proportionality.** The left figure shows the results of weight scaling without considering proportionality, while the right figure shows the results when proportionality is accounted for. The dashed line represents the test accuracy right after scaling, and the solid lines represent the best test accuracy achieved through additional training. All results are averaged over 5 runs on the CIFAR-10 dataset. An example illustrating the effect of the proportionality is shown in Fig. 1. The left figure represents the outcomes of weight scaling without taking proportionality into account, and the right represents the results when proportionality is considered. Two scaling approaches are compared across different scaling magnitudes on the CIFAR-10 dataset (Krizhevsky et al., 2009). The black horizontal line denotes the best test accuracy achieved during training over 100 epochs, and the blue line represents the best test accuracy during an additional 50 epochs of training. All scaling methods outperformed the best accuracy of the pre-trained model (black), indicating that the scaling method can address the overfitting issue. However, it is notable that considering proportionality as Theorem 1 maintains its test accuracy perfectly across all scaling ratios, as indicated by the red line. In contrast, the performance of the opposite exhibits a decline as the scaling magnitude increases. However, as mentioned above, there are infinitely many ways to rescale parameters. In the following section, we will discuss how to determine the scaler set c.

## 3.3 Soft Weight Rescaling

Selecting different scaling factors per layer becomes impractical as the number of layers increases.

In this subsection, we propose a novel method for effectively scaling parameters; the scaling factor of each layer depends on the change rate of the layer. We define the rate of how much the model has changed from the initial state as the ratio between the Frobenius norm of the current weight matrix and that of the initial one. Therefore, the scaling factor of the l-th layer is cl = ∥Winit l∥/∥Wl∥. This ensures that the magnitude of the layer remains at the initial value, and may constrain the model, forcing the weight norm to remain unchanged from the initial magnitude. Since the initial weight norm is small in most initialization techniques, the model may lack sufficient complexity (Neyshabur et al., 2015b). To address this limitation, we alleviate the scaling factor as follows:

$$c_{l}={\frac{\lambda\times\|W_{l}^{\mathrm{init}}\|+(1-\lambda)\times\|W_{l}\|}{\|W_{l}\|}}$$

With an exponential moving average (EMA), models can deviate from initialization smoothly while still regularizing the model. While this modification breaks hard constraints for weight magnitude, the algorithm still prevents unlimited growth of weight. We presented the proof of the boundedness of the weight magnitude in Appendix B. It is natural to question whether Theorem 1 can also be applied to networks that utilize commonly used techniques such as batch normalization (Ioffe, 2015) or layer normalization (Ba, 2016), due to their scale-invariant property (which is, if g is a function of normalization layer, for input x, g(cx) = g(x) for ∀c > 0). However, this property implies that we only need to focus on the learnable parameters of the final normalization layer to maintain the proportionality. The algorithm, including the normalization layer, is provided in Algorithm 1. For simplicity, we denote the scale and shift parameters of the normalization layer as W and b just like a typical layer, and in the case of layers without a bias vector (e.g. like the convolution layer right before batch normalization), we consider bias as the zero constant vector. Algorithm 1 Soft Weight Rescaling Given: Data stream D, neural network fθ with learnable parameters {(W1, b1), . . . ,(WL, bL)}. Initialize: step size α, coefficient λ n init l ← ∥Wl∥, l ∈ {1*, . . . , L*}

k ←
Index of final normalization layer, if network has normalization layer
0, otherwise
for (*x, y*) in D do
θ ← Parameters after **Gradient update** for (x, y) ▷ e.g. update with CrossEntropyLoss C ← 1 ▷ variable to calculate cumulative scaler for l in {1, 2*, . . . , L*} do
cl ←
λninit
l +(1−λ)∥Wl∥
∥Wl∥
cl otherwise ▷ cumulate scalers from last normalization layer
$$\left[c_{l}\cdot C\quad{\mathrm{if~}}l\geq k\right]$$
end for
end for
$$\begin{array}{l l}{{C\leftarrow\left\{c_{l}\right.}}}&{{\mathrm{otherwise}}}\\ {{\left(W_{l},b_{l}\right)\leftarrow\left(c_{l}\cdot W_{l},C\cdot b_{l}\right)}}\\ {{\left(\right)}}\end{array}$$
$C\gets\left\{\begin{matrix}c\\ \end{matrix}\right.$
It is notable that SWR scales the weights preceding the final normalization layer, while they do not affect the scale of the output. However, each of them has a distinct role. First, for convolution layers, the scalers control the effective learning rate which has been studied in previous research (Van Laarhoven, 2017; Zhang et al., 2018; Andriushchenko et al., 2023). Second, for the normalization layer, Lyle et al. (2024a) mentioned that unbounded parameters in normalization layers may cause issues in non-stationary environments such as continual or reinforcement learning. Although Summers & Dinneen (2019) demonstrated regularization for scale and shift parameters is only effective in specific situations, we also regularize scale and shift parameters, since our experiments focused on non-stationary environments and we observed that weights on several models diverged during training. Due to the different roles of regularization for each type of layer, we split the coefficient λ into two parts in the experiments. Henceforth, we denote the coefficient for the classifier as λc and the coefficient applied to the feature extractor (before the classifier) as λf .

## 3.4 Swr For Improved Balancedness

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 One of the advantages of SWR is that it aligns the magnitude ratios between layers. Neyshabur et al. (2015a); Liu et al. (2021) have mentioned that when the balance between layers is not maintained, it has a significant negative impact on subsequent gradient descent. Although Du et al. (2018) argued 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 that the balance between layers is automatically adjusted during training for the ReLU network, Lyle et al. (2024b) showed that in non-stationary environments, it is common for layers to grow at different rates. Weight decay cannot resolve this issue, since when the magnitude of a specific layer increases, the regularization effect on other layers is significantly reduced (Liu et al., 2021). However, SWR, which applies regularization to each layer individually, is not affected by this issue. We will show that using SWR at every update step makes the model balanced and illustrate empirical results with a toy experiment in Appendix C.

## 4 Experiments

In this section, we evaluate the effectiveness of SWR, comparing with other weight regularization methods. In all experiments, we used various models and datasets to compare results across different environments. For relatively smaller models, such as a 3-layer MLP and a CNN with 2 convolutional layers and 2 fully connected layers, we used MNIST (Deng, 2012), CIFAR10 and CIFAR100(Krizhevsky et al., 2009) datasets, which is commonly used in image classification experiment. To verify the effect of combining batch normalization, we additionally used a CNN-BN, which is CNN with batch normalization layers. For an extensive evaluation, we consider VGG- 16 (Simonyan & Zisserman, 2014) with the TinyImageNet dataset (Le & Yang, 2015). In all the following experiments, we compared our method with two weight regularizations, L2 (Krogh &
Hertz, 1991) and L2 Init (Kumar et al., 2023), as well as two re-initialization methods, Head Reset (Nikishin et al., 2022) and S&P (Ash & Adams, 2020). Detailed experimental settings, including hyperparameters for each method, are in Appendix D.

## 4.1 Warm-Starting

We use a warm starting setup from Ash & Adams (2020) to evaluate whether SWR can close the generalization gap. In our setting, models are trained for 100 epochs with 50% of training data and trained the entire training dataset for the following 100 epochs. Re-initialization methods are applied once before the training data is updated with the new dataset.

Test A
ccura cy warm start w/o warm start TinyImageNet (VGG-16)
CIFAR-10 (CNN)
CIFAR-100 (CNN)
0 20 40 60 80 100 Epoch 0.30 0.32 0.34 0.36 0.38 0.40 0 20 40 60 80 100 Epoch 0.28 0.30 0.32 0.34 0.36 0.38 0.40 0.42 0 20 40 60 80 100 Epoch 0.62 0.64 0.66 0.68 0.70 0.72 Test Acc uracy Test Acc uracy warm start w/o warm start warm start w/o warm start L2 L2 Init S&P head reset SWR (Ours)
Fig. 2 shows the test accuracy over the 100 epochs after the dataset was added. The dashed line indicates the final accuracy of the model without applying any regularization. The red line represents the warm-start scenario, and the black line shows the model trained from scratch for 100 epochs. Weight regularization methods such as L2 regularization and L2 Init, generally exceed the accuracy of without warm-starting in most small models, but it brings no advantage for larger models like VGG-16. Re-initialization methods, S&P and resetting the last layer, perform well, occasionally surpassing the performance of models without warm-start in VGG-16. Conversely, in smaller models, they yield only marginal improvements, suggesting that using either re-initialization or regularization methods in isolation fails to fully address warm-start challenges. However, regardless of the model size, SWR exhibited either comparable or better performance compared to other methods. In the case of VGG-16, while other regularization techniques failed to overcome the warm-start condition, SWR surpassed the test accuracy of S&P, which achieved the

## 4.2 Continual Learning

CIFAR-10 (CNN)
CIFAR-100 (CNN)
TinyImageNet (VGG-16)
0 200 400 600 800 1000 Epoch 0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0 200 400 600 800 1000 Epoch 0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0 200 400 600 800 1000 Epoch 0.40 0.45 0.50 0.55 0.60 0.65 0.70 Test Accu racy Test Acc uracy Test Acc uracy vanilla L2 L2 Init S&P head reset SWR (Ours)
324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 highest performance among the other methods. This indicates that with proper weight regularization, models may get more advantages than with methods that reset parts of the model. We leave the additional results for the warm start in the Appendix F. In the earlier section, we examined the impact of SWR on the generalization gap and observed considerable advantages. This subsection aims to verify whether a model that is repeatedly pretrained can continue to learn effectively. Similar to the setup provided by Shen et al. (2024), the entire data is randomly split into 10 chunks, and the training process consists of 10 stages. At each stage k, the model gains additional access to the k-th chunk. This allows us to evaluate how effectively each method can address the generalization gap when warm starts are repeated. As shown in Fig. 3, the result exhibits a similar behavior as warm-start. The regularization methods steadily improve performance during the entire training process for relatively small models. The Re-init methods also achieve higher performance than the vanilla model, but it is inevitable to experience a performance drop immediately after switching chunks and applying those methods. For a larger model, VGG-16, re-initializing weights is more beneficial for learning future data than simply regularizing weights. However, from the mid-phase of training, SWR begins to outperform S&P without losing performance. It shows that re-initialization provides significant benefits in the early stages of training, it becomes evident that well-regularized weights can offer greater advantages for future performance. Although S&P showed comparable effectiveness, such re-initialization methods lead to a loss of previously acquired knowledge. This phenomenon not only incurs additional costs for recovery but also presents critical issues when access to previous data is limited. In order to assess whether SWR can overcome these challenges, we modified the configuration; at the k-th stage, the model is trained only on the k-th chunk of data. This limited access setting restricts the model's access to previously learned data and is widely used to assess catastrophic forgetting.

CIFAR-10 (CNN)
CIFAR-100 (CNN)
TinyImageNet (VGG-16)
0 200 400 600 800 1000 Epoch 0.2 0.3 0.4 0.5 0.6 0 200 400 600 800 1000 Epoch 0.000 0.025 0.050 0.075 0.100 0.125 0.150 0.175 0.200 0 200 400 600 800 1000 Epoch 0.00 0.05 0.10 0.15 0.20 Test Acc uracy Test Accu racy Test Acc uracy vanilla L2 L2 Init S&P head reset SWR (Ours)
378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429

## 430

431 As shown in Fig. 4, we observe that, with CNN networks, SWR loses less test accuracy than other regularization methods when the chunk of training data changes. For VGG-16, SWR maintained test accuracy without a decrease at each stage. Although at risk of losing knowledge, S&P demonstrates competitive performance with other regularization methods. This suggests that, while re-initialization and re-training can demonstrate competitive performance in some cases, the risk of losing previously acquired knowledge should not be overlooked. SWR, by contrast, mitigates this risk and maintains stability in test accuracy across stages. Further investigation is needed to explore the specific circumstances under which re-initialization may offer benefits despite the risk of information loss. Additional results for other models and datasets are provided in Appendix F.

## 4.3 Generalization

To evaluate the impact of SWR not only on plasticity but also on standard generalization performance, we conducted experiments in a standard supervised learning setting. We trained the models for a total of 200 epochs with a learning rate, 0.001. The final test accuracy is shown in Table. 1. SWR outperformed other regularization methods across most datasets and models. Notably, in larger models such as VGG-16, where other regularization techniques offered minimal performance gains, SWR achieved an improvement of over 4% in test accuracy. This indicates that more effective methods for regulating parameters exist beyond conventional techniques like weight decay, commonly employed in supervised learning.

| MNIST      | CIFAR-10        | CIFAR-100       | CIFAR-100       | TinyImageNet    |                 |
|------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| Method     | (MLP)           | (CNN)           | (CNN)           | (CNN-BN)        | (VGG-16)        |
| vanilla    | 0.9789 ± 0.0009 | 0.6500 ± 0.0083 | 0.3283 ± 0.0067 | 0.3234 ± 0.0053 | 0.3912 ± 0.0142 |
| L2         | 0.9795 ± 0.0019 | 0.7119 ± 0.0037 | 0.3882 ± 0.0064 | 0.4222 ± 0.0043 | 0.3915 ± 0.0108 |
| L2 Init    | 0.9793 ± 0.0016 | 0.7041 ± 0.0125 | 0.3881 ± 0.0050 | 0.4030 ± 0.0105 | 0.3870 ± 0.0143 |
| SWR (Ours) | 0.9822 ± 0.0024 | 0.7158 ± 0.0063 | 0.3914 ± 0.0070 | 0.4129 ± 0.0105 | 0.4348 ± 0.0025 |

Table 1: **Results on generalization.** The final test accuracy with training 200 epochs with a learning rate of 0.001. SWR achieves comparable or even higher performance than other simple regularization methods in stationary image classification. To verify whether SWR works effectively with learning rate schedulers commonly used in supervised learning, we conducted additional experiments where the learning rate decays at specific epochs. Detailed results are provided in Appendix E.

## 5 Conclusion

In this paper, we introduced a novel method to recover the plasticity of neural networks. The proposed method, Soft Weight Rescaling, scales down the weights in proportion to the rate of weight growth. This approach prevents unbounded weight growth, a key factor behind various issues in deep learning. Through a series of experiments on standard image classification benchmarks, including warm-start and continual learning settings, SWR consistently outperformed existing weight regularization and re-initialization methods. Our study primarily focused on scaling down parameters. However, scaling up the weights depending on the learning progress could also prove beneficial. Investigating active scaling methods could potentially address the issues associated with the extensive training time in large neural networks. Although SWR achieved impressive results in several experiments, L2 often demonstrated better performance. This suggests the potential existence of even more effective weight rescaling methods. Additionally, there are further opportunities for exploration, such as regularizing models like transformers using proportionality or investigating alternative approaches to estimating the weight growth rate. A promising approach involves analyzing initialization techniques that effectively address these challenges. This analysis could yield insights into the characteristics of model parameters, potentially leading to improved initialization or optimization methods.

## References

Ibrahim Alabdulmohsin, Hartmut Maennel, and Daniel Keysers. The impact of reinitialization on generalization in convolutional neural networks. *arXiv preprint arXiv:2109.00267*, 2021.

Maksym Andriushchenko, Francesco D'Angelo, Aditya Varre, and Nicolas Flammarion. Why do we need weight decay in modern deep learning? *arXiv preprint arXiv:2310.04415*, 2023.

Jordan Ash and Ryan P Adams. On warm-starting neural network training. Advances in neural information processing systems, 33:3884–3894, 2020.

JL Ba. Layer normalization. *arXiv preprint arXiv:1607.06450*, 2016.

Peter Bartlett. For valid generalization the size of the weights is more important than the size of the network. *Advances in neural information processing systems*, 9, 1996.

Tudor Berariu, Wojciech Czarnecki, Soham De, Jorg Bornschein, Samuel Smith, Razvan Pascanu, and Claudia Clopath. A study on the plasticity of neural networks. *arXiv preprint* arXiv:2106.00042, 2021.

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 Nicolas Couellan. The coupling effect of lipschitz regularization in neural networks. SN Computer Science, 2(2):113, 2021.

Li Deng. The mnist database of handwritten digit images for machine learning research [best of the web]. *IEEE signal processing magazine*, 29(6):141–142, 2012.

Simon S Du, Wei Hu, and Jason D Lee. Algorithmic regularization in learning deep homogeneous models: Layers are automatically balanced. *Advances in neural information processing systems*, 31, 2018.

Mohamed Elsayed, Qingfeng Lan, Clare Lyle, and A Rupam Mahmood. Weight clipping for deep continual and reinforcement learning. *arXiv preprint arXiv:2407.01704*, 2024.

Mohammad Amin Ghiasi, Ali Shafahi, and Reza Ardekani. Improving robustness with adaptive weight decay. *Advances in Neural Information Processing Systems*, 36, 2024.

Noah Golowich, Alexander Rakhlin, and Ohad Shamir. Size-independent sample complexity of neural networks. In *Conference On Learning Theory*, pp. 297–299. PMLR, 2018.

Henry Gouk, Eibe Frank, Bernhard Pfahringer, and Michael J Cree. Regularisation of neural networks by enforcing lipschitz continuity. *Machine Learning*, 110:393–416, 2021.

Song Han, Jeff Pool, Sharan Narang, Huizi Mao, Enhao Gong, Shijian Tang, Erich Elsen, Peter Vajda, Manohar Paluri, John Tran, et al. Dsd: Dense-sparse-dense training for deep neural networks. arXiv preprint arXiv:1607.04381, 2016.

Lei Huang, Xianglong Liu, Bo Lang, and Bo Li. Projection based weight normalization for deep neural networks. *arXiv preprint arXiv:1710.02338*, 2017.

Sergey Ioffe. Batch normalization: Accelerating deep network training by reducing internal covariate shift. *arXiv preprint arXiv:1502.03167*, 2015.

Masato Ishii and Atsushi Sato. Layer-wise weight decay for deep neural networks. In Image and Video Technology: 8th Pacific-Rim Symposium, PSIVT 2017, Wuhan, China, November 20-24, 2017, Revised Selected Papers 8, pp. 276–289. Springer, 2018.

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images.

2009.

Anders Krogh and John Hertz. A simple weight decay can improve generalization. Advances in neural information processing systems, 4, 1991.

Aviral Kumar, Rishabh Agarwal, Dibya Ghosh, and Sergey Levine. Implicit under-parameterization inhibits data-efficient deep reinforcement learning. *arXiv preprint arXiv:2010.14498*, 2020.

Florin Gogianu, Tudor Berariu, Mihaela C Rosca, Claudia Clopath, Lucian Busoniu, and Razvan Pascanu. Spectral normalisation for deep reinforcement learning: an optimisation perspective. In International Conference on Machine Learning, pp. 3734–3744. PMLR, 2021.

Saurabh Kumar, Henrik Marklund, and Benjamin Van Roy. Maintaining plasticity via regenerative regularization. *arXiv preprint arXiv:2308.11958*, 2023.

Ya Le and Xuan Yang. Tiny imagenet visual recognition challenge. *CS 231N*, 7(7):3, 2015. Hojoon Lee, Hyeonseo Cho, Hyunseung Kim, Donghu Kim, Dugki Min, Jaegul Choo, and Clare Lyle. Slow and steady wins the race: Maintaining plasticity with hare and tortoise networks. arXiv preprint arXiv:2406.02596, 2024.

Alex Lewandowski, Haruto Tanaka, Dale Schuurmans, and Marlos C Machado. Curvature explains loss of plasticity. 2023.

Xiang Li, Shuo Chen, and Jian Yang. Understanding the disharmony between weight normalization family and weight decay. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 34, pp. 4715–4722, 2020a.

Xingjian Li, Haoyi Xiong, Haozhe An, Cheng-Zhong Xu, and Dejing Dou. Rifle: Backpropagation in depth for deep transfer learning through re-initializing the fully-connected layer. In International Conference on Machine Learning, pp. 6010–6019. PMLR, 2020b.

Ziquan Liu, CUI Yufei, and Antoni B Chan. Improve generalization and robustness of neural networks via weight scale shifting invariant regularizations. In ICML 2021 Workshop on Adversarial Machine Learning, 2021.

Clare Lyle, Zeyu Zheng, Khimya Khetarpal, James Martens, Hado van Hasselt, Razvan Pascanu, and Will Dabney. Normalization and effective learning rates in reinforcement learning. arXiv preprint arXiv:2407.01800, 2024a.

Clare Lyle, Zeyu Zheng, Khimya Khetarpal, Hado van Hasselt, Razvan Pascanu, James Martens, and Will Dabney. Disentangling the causes of plasticity loss in neural networks. *arXiv preprint* arXiv:2402.18762, 2024b.

William Merrill, Vivek Ramanujan, Yoav Goldberg, Roy Schwartz, and Noah Smith. Effects of parameter norm growth during transformer training: Inductive bias from gradient descent. arXiv preprint arXiv:2010.09697, 2020.

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 Behnam Neyshabur, Russ R Salakhutdinov, and Nati Srebro. Path-sgd: Path-normalized optimization in deep neural networks. *Advances in neural information processing systems*, 28, 2015a.

Behnam Neyshabur, Ryota Tomioka, and Nathan Srebro. Norm-based capacity control in neural networks. In *Conference on learning theory*, pp. 1376–1401. PMLR, 2015b.

Behnam Neyshabur, Srinadh Bhojanapalli, David McAllester, and Nati Srebro. Exploring generalization in deep learning. *Advances in neural information processing systems*, 30, 2017.

Lukas Niehaus, Ulf Krumnack, and Gunther Heidemann. Weight rescaling: Applying initialization strategies during training. *Swedish Artificial Intelligence Society*, pp. 83–92, 2024.

Evgenii Nikishin, Max Schwarzer, Pierluca D'Oro, Pierre-Luc Bacon, and Aaron Courville. The primacy bias in deep reinforcement learning. In *International conference on machine learning*, pp. 16828–16847. PMLR, 2022.

Vijaya Raghavan T Ramkumar, Elahe Arani, and Bahram Zonooz. Learn, unlearn and relearn: An online learning paradigm for deep neural networks. *arXiv preprint arXiv:2303.10455*, 2023.

Lawrence K. Saul. Weight-balancing fixes and flows for deep learning. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum? id=uaHyXxyp2r.

Maying Shen, Hongxu Yin, Pavlo Molchanov, Lei Mao, and Jose M Alvarez. Step out and seek around: On warm-start training with incremental data. *arXiv preprint arXiv:2406.04484*, 2024.

Twan Van Laarhoven. L2 regularization versus batch and weight normalization. arXiv preprint arXiv:1706.05350, 2017.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Sheheryar Zaidi, Tudor Berariu, Hyunjik Kim, Jorg Bornschein, Claudia Clopath, Yee Whye Teh, and Razvan Pascanu. When does re-initialization work? In *Proceedings on*, pp. 12–26. PMLR, 2023.

Yuichi Yoshida and Takeru Miyato. Spectral norm regularization for improving the generalizability of deep learning. *arXiv preprint arXiv:1705.10941*, 2017.

Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning (still) requires rethinking generalization. *Communications of the ACM*, 64(3):107– 115, 2021.

## A Proof Of Theorem 1

$$W_{l}^{c}\gets c_{l}\cdot W_{l},\quad b_{l}^{c}\leftarrow\left(\prod_{i=1}^{l}c_{i}\right)\cdot b_{l}$$

Let a c l and z c l denote the output after passing through the l-th activation function and layer, respectively. Since the homogeneous activation function ϕ satisfies cϕ(x) = ϕ(cx) for any c ≥ 0, output of the constructed network z c = fθ c (x) is, Baekrok Shin, Junsoo Oh, Hanseul Cho, and Chulhee Yun. Dash: Warm-starting neural network training without loss of plasticity under stationarity. In 2nd Workshop on Advancing Neural Network Training: Computational Efficiency, Scalability, and Resource Optimization (WANT@ ICML 2024), 2024.

Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. *arXiv preprint arXiv:1409.1556*, 2014.

Ghada Sokar, Rishabh Agarwal, Pablo Samuel Castro, and Utku Evci. The dormant neuron phenomenon in deep reinforcement learning. In *International Conference on Machine Learning*, pp. 32145–32168. PMLR, 2023.

Cecilia Summers and Michael J Dinneen. Four things everyone should know to improve batch normalization. *arXiv preprint arXiv:1906.03548*, 2019.

Ahmed Taha, Abhinav Shrivastava, and Larry S Davis. Knowledge evolution in neural networks.

In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 12843–12852, 2021.

Guodong Zhang, Chaoqi Wang, Bowen Xu, and Roger Grosse. Three mechanisms of weight decay regularization. *arXiv preprint arXiv:1810.12281*, 2018.

Proof. Consider a set c = {c1, c2*, . . . , c*L} consisting of positive real numbers such that C = ΠL
i=1ci. Then, construct the new parameter set θ c.= {Wc 1
, bc1
, . . . WcL
, bcL
} according to the following rules:
594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 The reduction of the Frobenius norm by scaling can be simply represented as: From the assumption, the increase of the Frobenius norm by gradient update is bounded.

= . . .
= cLcL−1 . . . c1 · fθ(x) = C · fθ(x)
Therefore, we can construct proportional networks with proportionality constant C using infinitely many set c.

## B Boundedness

In this section, we present the proof for the weight magnitude boundedness of SWR. If the Frobenius norm of the weight of an arbitrary layer is bounded by a constant, the entire network is also bounded. Therefore, we focus on demonstrating the boundedness of a single layer.

Theorem 2. If the change of squared Frobenius norm of the weight matrix, resulting from the single gradient update, is bounded by a constant for all weight matrices in the neural network, then SWR for every update step with fixed coefficient λ *bounds the Frobenius norm of the weight matrix.* Proof. It is enough to show the case where the gradient update increases the magnitude of the weight matrix. For a weight matrix in step t ≥ 1, Wt, let the matrix after applying SWR with λ once be Wc t, Wc t−1 be the weight matrix before the gradient update at Wt, and B > 0 be the bound of the change of squared Frobenius norm of the matrix. Wc tcan be written as below:

$$W_{t}^{c}=\frac{\lambda\times\|W_{0}\|+(1-\lambda)\times\|W_{t}\|}{\|W_{t}\|}W_{t}$$ $$=\left(\lambda\frac{\|W_{0}\|}{\|W_{t}\|}+(1-\lambda)\right)W_{t}$$
$$(1)$$
$$\|W_{t}\|-\|W_{t}^{c}\|=\|W_{t}\|-\left(\lambda\frac{\|W_{0}\|}{\|W_{t}\|}+(1-\lambda)\right)\|W_{t}\|$$ $$=\|W_{t}\|-(\lambda\|W_{0}\|+(1-\lambda)\|W_{t}\|)$$ $$=\lambda(\|W_{t}\|-\|W_{0}\|)$$
$$(2)$$

$$B\geq\left|\left|\left|W_{t}\right|\right|^{2}-\left|\left|W_{t-1}^{c}\right|\right|^{2}\right|$$ $$=\left|\left|\left|W_{t}\right|\right|-\left|\left|W_{t-1}^{c}\right|\right|\right|\times\left|\left|\left|W_{t}\right|\right|+\left|\left|W_{t-1}^{c}\right|\right|\right|$$ $$\geq\left|\left|\left|W_{t}\right|\right|-\left|\left|W_{t-1}^{c}\right|\right|\right|^{2}$$ $$\implies\left|\left|\left|W_{t}\right|\right|-\left|\left|W_{t-1}^{c}\right|\right|\right|\leq\sqrt{B}$$
$$(6)$$
$$\left(T\right)$$
fθ c (x) = z c = WcLa cL−1 + b cL = cL   WLϕ(z c L−1) + L Y −1 i=1 cibL ! = cL   WLϕWcL−1a c L−2 + b c L−1 + L Y −1 i=1 cibL ! = cL   WLϕ   cL−1   WL−1ϕ(z c L−2) + L Y −2 i=1 cibL−1 !!  + L Y −1 i=1 cibL ! = cLcL−1   WLϕ   WL−1ϕ(z c L−2) + L Y−2 i=1 cibL−1 ! + L Y−2 i=1 cibL !
648 649 650 651 652 653 654 655 656 657 658 659 660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701

## C Balancedness

C.1 EMPIRICAL STUDY By following the assumptions of Theorem 2, it can be easily shown that the weight Frobenius norm growth follows O(
√t) as the empirical evidence shown in (Merrill et al., 2020), thereby indicating that the assumption is not unreasonable. Since the spectral norm of the weight matrix is lower than its Frobenius norm, we can show that the neural network using SWR has an upper bound of the Lipschitz constant. For simplexity, we only consider MLP with a 1-Lipschitz activation function.

Corollary 2.1. For an MLP, fθ, with 1*-Lipshcitz activation function (e.g. ReLU, Leaky ReLU, etc.),* fθ *is Lipschitz continuous with applying SWR for every update step.*
Proof. We denote the spectral norm of the matrix with *∥ · ∥*σ. Let weight matrices of fθ be Wl
(l ∈ {1, 2*, . . . L*}), and Bl be the upper bound of the Frobenius norm of each of them. Using the relationship between the Frobenius norm and the spectral norm, ∥Wl∥σ ≤ ∥Wl∥ for all l. Since the Lipschitz constant of the weight matrix is same with its spectral norm and composition of l1 and l2 Lipschitz function is l1l2 Lipschitz function (Gouk et al. (2021)), the Lipschitz constant of neural network kθ can be express as:

$$k_{\theta}\leq\prod_{l}\|W^{l}\|_{\sigma}$$
$\varepsilon_{\theta}\leq\prod\limits_{l}\|W^{l}\|_{\sigma}$  $\leq\prod\limits_{l}\|W^{l}\|$  $\leq\prod\limits_{l}B^{l}\doteq B^{\prime}$
$$(13)$$
$$(14)$$
$$(15)^{\frac{1}{2}}$$

From the perspective of the Frobenius norm, the weight magnitude stops growing when the reduction with scaling gets greater than the increase with gradient update. The condition can be written by below inequality:

$$\lambda(\|W_{t}\|-\|W_{0}\|)\geq\sqrt{B}$$ $$\|W_{t}\|\geq\frac{\sqrt{B}}{\lambda}+\|W_{0}\|\doteq B^{\prime}\tag{1}$$
$$(10)$$
$$(11)$$
′(11)
For all t ≥ 1, if the Frobenius norm exceeds B′, it will no longer increase. Since B′is constant, we can bound the Frobenius norm as follows:

$$\|W_{t}\|\leq B^{\prime}$$
$$(12)$$
$\square$
′(12)
$$\|W\|_{p,q}=\left(\sum_{i}\left(\sum_{j}|W_{i j}|^{p}\right)^{\frac{q}{p}}\right)^{\frac{1}{q}}.$$
$$(16)$$

13 Note that the Lipschitz constant of the activation function is 1, so activation functions do not affect to bound of the Lipschitz constant of kθ. Since Lipschitz constant kθ is bounded with B′, fθ is B′-Lipschitz continuous function.

Similarly, we can get the neural network that is trained with SWR as Lipschitz continuous when using a convolution network or normalization layer. We left a tight upper bound of Lipschitz constant for future work.

Neyshabur et al. (2015a) defined the entry-wise ℓp,q-norm of the model, which is expressed as follows:
702 703 704 705 706 707 708 709 710 711 712 713 714 715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 If two models are functionally identical, the model that has a smaller ℓp,q-norm represents more balanced. In order to estimate the model balancedness, we used the ratio between the entry-wise ℓp,q-norm of current and global minimal. We compute the global minimal ℓp,q-norm using Algorithm 1 of Saul (2023). Fig. 5 shows the balancedness of the 3-layer MLP, measured at the end of each epoch, along with the test accuracy. SWR is shown to enhance model balancedness and improve test accuracy compared to the vanilla model.

0 25 50 75 100 125 150 175 200 Epoch 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0 25 50 75 100 125 150 175 200 Epoch 0.46 0.48 0.50 0.52 0.54 Bal an ced nes s Te st Ac cur acy vanilla SWR

$$(17)$$

vanilla SWR

## C.2 Theoretical Analysis

Next, we will show that SWR improves the balance between layers. Before proving it, we define how to express balancedness. Definition 2 (Balancedness between two layers). *Consider a network with two weight matrices at* time step t to be Wt and W′
t(at initial, W0, W′0). Without loss of generality, we let ∥W0*∥ ≤ ∥*W′0∥.

We define the balance of two layers bt *as the difference of rates of the Frobenius norms of weight* matrices from the initial state. This can be expressed as follows:

$b_{t}=|r_{t}-r_{0}|$, where $r_{t}=\dfrac{\|W_{t}^{\prime}\|}{\|W_{t}\|}$ (10.1)
That is, bt is a non-negative value, and the closer it is to 0, the better balance between the two layers. Theorem 3. Applying SWR with coefficient λ enhances the balance of the neural network.

Proof. Keep the settings from Definition 2. Let Wt and W′t be the weight matrices of any two layers at time step t in the neural network and bt be the balance of Wt and W′
t. Then, b c t, the balance after applying SWR with coefficient λ, can represent it as below:

$b_{t}^{c}\doteq|r_{t}^{c}-r_{0}|$, where $r_{t}^{c}=\dfrac{\|W_{t}^{\prime c}\|}{\|W_{t}^{c}\|}$  ght matrices that scaled by SWR with $\lambda$. Then, by equati
$$(18)$$
where Wc tand W′c tare the weight matrices that scaled by SWR with λ. Then, by equation 5, r c tcan be expanded as follows:

 ${\ r_t^c=\frac{\lambda\|W_0'\|+(1-\lambda)\|W_t'\|}{\lambda\|W_0\|+(1-\lambda)\|W_t\|}}$  and mediant of ${r_t}$ and ${r_0}$, if ${r_0\leq r_t}$, the relationship between ${r_t}$ and ${r_0}$, is ${r_0\leq r_t}$, the relationship between ${r_0\leq r_t}$ and ${r_0\leq r_t}$. 
 $\Rightarrow$  $\Rightarrow$  $\Rightarrow$
$$(19)$$
Since r
c
tis the form of generalized mediant of rt and r0, if r0 ≤ rt, the relationship between their
magnitudes and balance satisfies as below:
$$\begin{array}{r c l c r c l}{{r_{0}}}&{{\leq}}&{{r_{t}^{c}}}&{{\leq}}&{{r_{t}}}\\ {{0}}&{{\leq}}&{{r_{t}^{c}-r_{0}}}&{{\leq}}&{{r_{t}-r_{0}}}\\ {{0}}&{{\leq}}&{{|r_{t}^{c}-r_{0}|}}&{{\leq}}&{{|r_{t}-r_{0}|}}\\ {{0}}&{{\leq}}&{{b_{t}^{c}}}&{{\leq}}&{{b_{t}}}\end{array}$$
If r0 ≥ rt, we can derive equation 22, following a similar approach. Therefore, the balance of arbitrary two layers gets better when applying SWR, which indicates an overall improvement in the balance across all layers of the network.

$$\begin{array}{l}{(20)}\\ {(21)}\end{array}$$
$$(22)^{\frac{1}{2}}$$
$\eqref{eq:walpha}$. 

## D Details For Experimental Setup

In this section, we will provide details on the experimental setup. First, we specify the hyperparameters that we commonly use. We used 256 for the batch size of the mini-batch and 0.001 for the learning rate. The Adam optimizer was employed, with its hyperparameters set to the default values without any modification. We employed distinct 5 random seeds for all experiments while performing 3 seeds for VGG-16 due to computational efficiency. In the following sections, we present model architectures, the baseline methods that we compared, and the hyperparameters for the best test accuracy.

## D.1 Model Architectures

We utilized four model architectures consistently throughout all experiments. The detailed information on architectures is as follows: MLP: We used the 3-layer Multilayer Perceptron (MLP) with 100 hidden units. The 784 (28 × 28) input size and 10 output size are fixed since MLP is only trained in the MNIST dataset. CNN: We employed a Convolutional Neural Network (CNN), which is used in relatively small image classification. The model includes two convolutional layers with a 5 × 5 kernel and 16 channels. The fully connected layers follow with 100 hidden units. CNN-BN: In order to verify whether our methodology is effectively applied to normalization layers, we attached batch normalization layers following the convolutional layer in the CNN model. VGG-16 (Simonyan & Zisserman, 2014): We adopted VGG-16 to investigate whether SWR adapts properly in large-size models. The number of hidden units of the classifiers was set to 4096 without dropout.

## D.2 Baselines

L2. The L2 regularization is known as enhancing not only generalization performance Krogh & Hertz (1991) but also plasticity Lyle et al. (2024b). We add the loss term λ 2
∥θ∥
2 on the cross-entropy loss. We sweeped λ in {0.1, 0.01, 0.001, 0.0001, 0.00001}. L2 Init. Kumar et al. (2023) introduced a regularization method to resolve the problem of the loss of plasticity where the input or output of the training data changes periodically. They argued that regularizing toward the initial parameters, results in resetting low utility units and preventing weight rank collapse. We add the loss term λ2
∥θ − θ0∥
2 on the cross-entropy loss, where θ0 is the initial learnable parameter. We performed the same grid search with L2. S&P. Ash & Adams (2020) demonstrated that the network loses generalization ability for warm start setup, and introduced effective methods that shrink the parameters and add noise perturbation, periodically. In order to reduce the complexity of hyperparameters, we employ a simplified version of S&P using a single hyperparameter, as shown in Lee et al. (2024). We applied S&P when the training data was updated. The mathematical expression is θ ← (1 − λ)θ + λθ0, where θ0 is initial learnable parameters, and we swept λ in {0.2, 0.4, 0.6, 0.8}. head reset. Nikishin et al. (2022) suggested that periodically resetting the final few layers is effective in mitigating plasticity loss. In this paper, we reinitialized the fully connected layers with the same period with S&P. We only applied reset to the final layer, when MLP is used for training.

SWR. For networks that do not have batch normalization layers, we swept λ in {1, 0.1, 0.01, 0.001, 0.0001}. Otherwise, we performed a grid search for λc and λf in the same range of λ.

Table. 2-4 shows the best hyperparameter set that we found in various experiments.

## E Generalization Results With Learning Rate Decay

756 757 758 759 760 761 762 763 764 765 766 767 768 769 770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 To assess the performance of SWR under the learning rate scheduler, we conducted learning rate decay in Experiment 4.3. The rest of the configuration was kept unchanged, while the learning rate was multiplied by 1/10 at the start of the 100th and 150th epochs.

810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863

| Dataset      | Method               | Hyperparameter Set   |
|--------------|----------------------|----------------------|
| S&P          | λ = 0.4              |                      |
| MNIST        | L2                   | λ = 1e−5             |
| (MLP)        | L2 Init              | λ = 1e−5             |
| SWR          | λ = 1e−4             |                      |
| S&P          | λ = 0.8              |                      |
| CIFAR-10     | L2                   | λ = 1e−2             |
| (CNN)        | L2 Init              | λ = 1e−2             |
| SWR          | λ = 1e−3             |                      |
| S&P          | λ = 0.8              |                      |
| CIFAR-100    | L2                   | λ = 1e−2             |
| (CNN)        | L2 Init              | λ = 1e−2             |
| SWR          | λ = 1e−3             |                      |
| S&P          | λ = 0.8              |                      |
| CIFAR-100    | L2                   | λ = 1e−2             |
| (CNN-BN)     | L2 Init              | λ = 1e−2             |
| SWR          | λf = 1e−4, λc = 1e+0 |                      |
| S&P          | λ = 0.8              |                      |
| TinyImageNet | L2                   | λ = 1e−5             |
| (VGG-16)     | L2 Init              | λ = 1e−5             |
| SWR          | λf = 1e−2, λc = 1e−1 |                      |

Table 2: Hyperparameter set of each method on the warm-start experiment.

| Dataset      | Method               | Full Access          | Limited Access   |
|--------------|----------------------|----------------------|------------------|
| S&P          | λ = 0.6              | λ = 0.2              |                  |
| MNIST        | L2                   | λ = 1e−4             | λ = 1e−5         |
| (MLP)        | L2 Init              | λ = 1e−4             | λ = 1e−5         |
| SWR          | λ = 1e−4             | λ = 1e−4             |                  |
| S&P          | λ = 0.8              | λ = 0.4              |                  |
| CIFAR-10     | L2                   | λ = 1e−2             | λ = 1e−2         |
| (CNN)        | L2 Init              | λ = 1e−2             | λ = 1e−2         |
| SWR          | λ = 1e−3             | λ = 1e−1             |                  |
| S&P          | λ = 0.8              | λ = 0.6              |                  |
| CIFAR-100    | L2                   | λ = 1e−2             | λ = 1e−2         |
| (CNN)        | L2 Init              | λ = 1e−2             | λ = 1e−2         |
| SWR          | λ = 1e−3             | λ = 1e−1             |                  |
| S&P          | λ = 0.8              | λ = 0.4              |                  |
| CIFAR-100    | L2                   | λ = 1e−2             | λ = 1e−2         |
| (CNN-BN)     | L2 Init              | λ = 1e−2             | λ = 1e−2         |
| SWR          | λf = 1e−4, λc = 1e−1 | λf = 1e−1, λc = 1e−2 |                  |
| S&P          | λ = 0.8              | λ = 0.4              |                  |
| TinyImageNet | L2                   | λ = 1e−4             | λ = 1e−4         |
| (VGG-16)     | L2 Init              | λ = 1e−4             | λ = 1e−3         |
| SWR          | λf = 1e−2, λc = 1e−2 | λf = 1e−4, λc = 1e+0 |                  |

There is a consideration to be addressed when applying learning rate decay with SWR. When the learning rate decays, we will show that the regularization strength that maintains balance becomes relatively stronger. Suppose that after time step t, the L2 norm of the weight vector is near convergence. To simplify the case, let us assume the weight vector, wt, aligns with the direction of the gradient of the loss ∇wL(w). After the SGD update, the weight vector will be updated as wt+1 = wt − α∇wL(w), meaning the change of L2 norm is α∥∇wL(w)∥. According to equation 5, when applying SWR, the change in L2 norm becomes λ(∥wt+1*∥ − ∥*w0∥). Under our assumption, we have α∥∇wL(w)∥ ≈ λ(∥wt+1*∥ − ∥*w0∥). Therefore, when a learning rate decay occurs, this equivalence is broken, causing the weight norm to drop toward the initial weight norm. To address this issue, we used a simple trick that reset the initial weight norm to the current norm when decay happens, as n init ← ∥wt∥. We refer to this method as SWR + re-init.

The results with learning rate decay can be found in Table 5. SWR+re-init demonstrated performance largely comparable to other methods, specifically leading to an improvement of over 8% in 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917

| Dataset      | Method               | Hyperparameter Set   |
|--------------|----------------------|----------------------|
| MNIST        | L2                   | λ = 1e−5             |
| (MLP)        | L2 Init              | λ = 1e−5             |
| SWR          | λ = 1e−4             |                      |
| CIFAR-10     | L2                   | λ = 1e−2             |
| (CNN)        | L2 Init              | λ = 1e−2             |
| SWR          | λ = 1e−3             |                      |
| CIFAR-100    | L2                   | λ = 1e−2             |
| (CNN)        | L2 Init              | λ = 1e−2             |
| SWR          | λ = 1e−3             |                      |
| CIFAR-100    | L2                   | λ = 1e−2             |
| (CNN-BN)     | L2 Init              | λ = 1e−2             |
| SWR          | λf = 1e−4, λc = 1e−1 |                      |
| TinyImageNet | L2                   | λ = 1e−5             |
| (VGG-16)     | L2 Init              | λ = 1e−5             |
| SWR          | λf = 1e−2, λc = 1e−1 |                      |

test accuracy on VGG-16. While SWR + re-init generally outperformed standalone SWR, a slight performance drop was observed in larger models such as VGG-16. This suggests that more effective solutions exist to handle this issue when using learning rate decay. Further research on this matter will be left as future work.

| MNIST                | CIFAR-10        | CIFAR-100       | CIFAR-100       | TinyImageNet    |                 |
|----------------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| Method               | (MLP)           | (CNN)           | (CNN)           | (CNN-BN)        | (VGG-16)        |
| vanilla              | 0.9798 ± 0.0005 | 0.6571 ± 0.0057 | 0.3490 ± 0.0021 | 0.3483 ± 0.0043 | 0.4126 ± 0.0236 |
| L2                   | 0.9811 ± 0.0007 | 0.7304 ± 0.0039 | 0.3949 ± 0.0091 | 0.4532 ± 0.0040 | 0.4080 ± 0.0124 |
| L2 Init              | 0.9811 ± 0.0007 | 0.7286 ± 0.0023 | 0.4048 ± 0.0019 | 0.4341 ± 0.0019 | 0.4199 ± 0.0048 |
| SWR (Ours)           | 0.9796 ± 0.0009 | 0.6925 ± 0.0078 | 0.3599 ± 0.0054 | 0.4240 ± 0.0015 | 0.5221 ± 0.0123 |
| SWR + re-init (Ours) | 0.9829 ± 0.0002 | 0.7269 ± 0.0027 | 0.4133 ± 0.0058 | 0.4451 ± 0.0028 | 0.5165 ± 0.0070 |

Table 5: **Results on generalization with learning rate decay.** The final test accuracy after training 200 epochs. The learning rate initialized with 0.001 and divided by 10 at epoch 100 and 150.

## F Additional Results

MNIST (MLP)
CIFAR-100 (CNN-BN)
0 20 40 60 80 100 Epoch 0.970 0.972 0.974 0.976 0.978 0.980 0.982 0.984 0 20 40 60 80 100 Epoch 0.30 0.32 0.34 0.36 0.38 0.40 0.42 Test Acc ura cy warm start Test Acc ura cy w/o warm start warm start w/o warm start L2 L2 Init S&P head reset SWR (Ours)
918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971

MNIST (MLP)
CIFAR-100 (CNN-BN)
0 200 400 600 800 1000 Epoch 0.90 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98 0.99 0 200 400 600 800 1000 Epoch 0.10 0.15 0.20 0.25 0.30 0.35 0.40 Test Ac cur acy Tes t Ac cura cy vanilla L2 L2 Init S&P head reset SWR (Ours)
MNIST (MLP)
CIFAR-100 (CNN-BN)
0 200 400 600 800 1000 Epoch 0.89 0.90 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0 200 400 600 800 1000 Epoch 0.00 0.05 0.10 0.15 0.20 0.25 0.30 Test Ac curacy Test Ac curacy vanilla L2 L2 Init S&P head reset SWR (Ours)