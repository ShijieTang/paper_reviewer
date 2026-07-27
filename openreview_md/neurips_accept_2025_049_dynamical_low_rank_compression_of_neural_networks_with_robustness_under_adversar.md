# Dynamical Low-Rank Compression Of Neural Networks With Robustness Under Adversarial Attacks

Steffen Schotthöfer∗
, H. Lexie Yang†
, and Stefan Schnake∗
∗Computer Science and Mathematics Division, †Geospatial Science and Human Security Division Oak Ridge National Laboratory Oak Ridge, TN 37831 USA
{schotthofers,yangh,schnakesr}@ornl.gov

## Abstract

Deployment of neural networks on resource-constrained devices demands models that are both compact and robust to adversarial inputs. However, compression and adversarial robustness often conflict. In this work, we introduce a dynamical lowrank training scheme enhanced with a novel spectral regularizer that controls the condition number of the low-rank core in each layer. This approach mitigates the sensitivity of compressed models to adversarial perturbations without sacrificing accuracy on clean data. The method is model- and data-agnostic, computationally efficient, and supports rank adaptivity to automatically compress the network at hand. Extensive experiments across standard architectures, datasets, and adversarial attacks show the regularized networks can achieve over 94% compression while recovering or improving adversarial accuracy relative to uncompressed baselines.

## 1 Introduction

Deep neural networks have achieved state-of-the-art performance across a wide range of tasks in computer vision and data processing. However, their success comes at a cost of substantial computational and memory demands, which hinders deployment in resource-constrained environments. While significant progress has been made in scaling up models through data centers and specialized hardware, a complementary and equally important challenge lies in the opposite direction: deploying accurate and robust models on low-power platforms such as unmanned aerial vehicles (UAVs) or surveillance sensors. These platforms often operate in remote locations with limited power and compute resources, and are expected to function autonomously over extended periods without human intervention. This setting introduces three interdependent challenges: - **Compression:** Models must operate under strict memory, compute, and energy budgets. - **Accuracy:** Despite being compressed, models must maintain high performance to support critical decision-making.

- **Robustness:** Inputs may be corrupted by noise or adversarial perturbations, requiring models to be resilient under distributional shifts.

Recent work has shown that these three objectives are inherently at odds. Compression via low-rank [38] or sparsity techniques [14] often leads to reduced accuracy. Techniques to improve adversarial robustness—such as data augmentation [24] or regularization-based defenses [54]—frequently degrade clean accuracy. Moreover, it has been observed that low-rank compressed networks can exhibit increased sensitivity to adversarial attacks [35]. Finally, many methods to increase adversarial robustness of the model impose additional computational burdens during training [43, 8] or inference [9, 15, 28], further complicating deployment on constrained hardware. Our Contribution. We summarize our main contributions as follows: - **Low-rank compression framework.** We introduce a novel regularization and integration method to modify a class of low-rank training methods that yields low-rank compressed neural networks, achieving a more than 10× reduction in both memory footprint and compute cost, while maintaining clean accuracy and adversarial robustness on par with full-rank baselines.

- **Theoretical guarantees.** We analyze the proposed regularizer and derive an explicit bound on the condition number κ of each regularized layer. The bound gives further confidence that the regularizer improves adversarial performance.

- **Preservation of performance.** We prove analytically—and verify empirically—that our regularizer neither degrades training performance nor reduces clean validation accuracy across a variety of network architectures.

- **Extensive empirical validation.** We conduct comprehensive experiments on multiple architectures and datasets, demonstrating the effectiveness, robustness, and broad applicability of our method.

Beyond these core contributions, our approach is model- and data-agnostic, can be integrated seamlessly with existing adversarial defenses, e.g., adversarial training [13], and never requires assembling full-rank weight matrices—the last point guaranteeing a low memory footprint during training and inference. Moreover, by connecting to dynamical low-rank integration schemes and enabling convergence analysis via gradient flow, we offer new theoretical and algorithmic insights. Finally, the use of interpretable spectral metrics enhances the trustworthiness and analyzability of the compressed models.

## 2 Controlling The Adversarial Robustness Of A Neural Network Through The Singular Spectrum Of Its Layers

We consider a neural network f as a concatenation of L layers z ℓ+1 = σ ℓ(Wℓz ℓ) with matrix valued1 parameters Wℓ ∈ R
n×n, layer input z ℓ ∈ R
n×band element-wise nonlinear activation σ ℓ. For simplicity of notation, we do not consider biases, but they are included for the numerical experiments in Section 6. The data X constitutes the input to the first layer, i.e. z 0 = X. We assume that the layer activations σ ℓare Lipschitz continuous, which is the case for all popular activations [35]. The network is trained on a loss function L which we assume to be locally bounded with a Lipschitz continuous gradient. Throughout this work, we call a network in the standard format a "baseline" network.

Low-rank Compression: The compression the network for training and inference is typically facilitated by approximating the layer weight matrices by a low-rank factorization Wℓ = U
ℓS
ℓV
ℓ,⊤
with U
ℓ, V ℓ ∈ R
n×rand S
ℓ ∈ R
r×r, where r ≤ n is the rank of the factorization. In this work, we generally assume that U
ℓ, V ℓare orthonormal matrices at all times during training and inference. This assumption deviates from standard low-rank training approaches [17], however recent literature provides methods that are able to fulfill this assumption approximately [55] and even exactly [38, 37]. If r ≪ n, the low-rank factorization with a storage and matrix-vector computation cost cost of O(2nr + r 2) is computationally more efficient than the standard matrix format W with a computational cost of O(n 2).

Adversarial robustness: The adversarial robustness of a neural network f, a widely used trustworthiness metric, can be measured by its relative sensitivity S to small perturbations δ, e.g., noise, of the input data X [49, 11], i.e., S(*f, X, δ*) := ||f(X+δ)−f(X)|| ||f(X)|| ||X|| ||δ|| 
. In this work, we consider the sensitivity in the Euclidean (ℓ 2) norm, i.e., || · || = *|| · ||*2. For neural networks consisting of layers with Lipschitz continuous activation functions σ ℓ, S can be bounded [35] by the product

$${\mathcal{S}}(f,X,\delta)\leq\left(\prod_{\ell=1}^{L}\kappa(W^{\ell})\right)\left(\prod_{\ell=1}^{L}\kappa(\sigma^{\ell})\right)$$
ℓ)(1)
where κ(W) := ∥W∥W† is the condition number of a matrix W, W†is the pseudo-inverse of W, and κ(σ) is the condition number of the layer activation function σ. The condition number of the element-wise non-linear activation functions σ ℓcan be computed with the standard definitions (see
[45] and [35] for condition numbers of several popular activation functions). Equation (1) allows us to consider each layer individually, thus we drop the superscript ℓ for brevity of exposition.

1We provide an extension to tensor-valued layers, e.g. in CNNs, in Section 5.1 2Note that the difference between the baseline and low-rank singular spectrum may be less pronounced for other layers and architectures. However, we have observed in all test cases that regularization with R makes the singular spectrum of the low-rank network more benign.

$$(1)$$

10 0 10 1 10 2 index i 0 1 2 3 i RobustDLRT Unregularized Baseline
Figure 1: The singular values ςi(W) of sequential layer 7 in VGG16 for baseline training, unregularized dynamical lowrank training, and RobustDLRT with our condition number regularizer R with β = 0.075 (see Section 5). The matrix W is formed as the first-mode unfolding of the convolutional tensor. Conditioning of the regularized low-rank layer is significantly improved compared to the non-regularized lowrank and baseline layer.2 The sensitivity of a low-rank factorized network can be readily deducted from Equation (1) by leveraging orthonormality of U and V , i.e., κ(USV 
⊤) = κ(S). Thus, we only consider the r × r coefficient matrix S to control the sensitivity of the network. The condition number κ(S) can be determined via a singular value decomposition (SVD) of S, which is computationally feasible when r ≪ n. Adversarial robustness-aware lowrank training: Enhancing the adversarial robustness of the network during low-rank training thus boils down to controlling the conditioning of S, which is a non-trivial task. Moreover, the dynamics of the singular spectrum of S of adaptive low-rank training schemes as Dynamical Low-Rank Training (DLRT) [38] become more ill-conditioned than the baseline during training, even if S is always full rank. In Figure 1, we observe that the singular values ς of a rank 64 factorization of a network layer compressed with DLRT range from ςr=1 = 2.7785 to ςr=64 = 0.8210 yielding a condition number of κ(S) = 3.3844. In comparison, the baseline network has singular values ranging from ςr=1 = 1.8627 to ςr=128 = 0.9445 yielding a lower condition number of κ(S*) = 1*.9722. As a result, an ℓ 2-FGSM attack with strength ϵ = 0.3, reduces the accuracy of the baseline network to 54.96%,
while the accuracy of the low-rank network drops to 43.39%, see Table 2.

## 3 Related Work

Low-rank compression is a prominent approach for reducing the memory and computational cost of deep networks by constraining weights to lie in low-rank subspaces. Early methods used posthoc matrix [12] and tensor decompositions [23], while more recent approaches enforce low-rank constraints during training for improved efficiency and generalization.

Dynamical Low-Rank Training [38] constrains network weights to evolve on a low-rank manifold throughout training, allowing substantial reductions in memory and FLOPs without requiring fullrank weight storage. The method has been extended to tensor-valued neural network layers [53], and federated learning [36]. Pufferfish [47] restricts parameter updates to random low-dimensional subspaces, while intrinsic dimension methods [2] argue that many tasks can be learned in such subspaces. GaLore [56] reduces memory cost by projecting gradients onto low-rank subspaces. In contrast, low-rank fine-tuning methods like low-rank adaptation (LoRA) [17] inject trainable low-rank updates into a frozen pre-trained model, enabling efficient adaptation with few parameters. Extensions such as GeoLoRA [37], AdaLoRA [55], DyLoRA [46], and DoRA [31] incorporate rank adaptation or structured updates, improving performance over static rank baselines. However, these fine-tuning methods do not reduce the cost of the full training and inference, thus are not applicable to address the need of promoting computational efficiency. Pruning is another well studied approach to reduce the number of parameters of a trained neural network [18, 26, 40, 57, 7, 19] by either sparsifying weight matrices or layer output channels of a network. Typically sparsity pruning is performed after training a fully parametrized neural network and thus only reduces memory and compute load during inference, while treating training as an offline cost.

Improving adversarial robustness with orthogonal layers has been a recently studied topic in the literature [3, 4, 48, 10, 35]. Many of these methods can be classified as either a soft approach, where orthogonality is imposed weakly via a regularizer, or a hard approach, where orthogonality is explicitly enforced in training.

Examples of soft approaches include the soft orthogonal (SO) regularizer [48], double soft orthogonal regularizer [4], mutual coherence regularizer [4], and spectral normalization [32]. These regularization-based approaches have several advantages; namely, they are more flexible to many problems/architectures and are amenable to transfer learning scenarios (since pertained models are admissible in the optimization space). However, influencing the spectrum weakly via regularization cannot enforce rigorous and explicit bounds on the spectrum.

Many hard approaches strongly enforce orthogonality/well-conditioned constraints by training on a chosen manifold using Riemannian optimization methods [25, 1, 35]. A hard approach built for low-rank training is given in [35]; this method clamps the extremes of the spectrum to improve the condition number during training. The clamping gives a hard estimate on the range of the spectrum which enables a direct integration of the low-rank equations of motion with reasonable learning rates. However, this method requires a careful selection of the rank r, which is viewed as a hyperparameter in [35]. If r is chosen incorrectly, the clamping of the spectrum, a hard-thresholding technique, acts as a strong regularizer which could affect the validation metrics of the network. Our regularization method detailed below falls neatly into a soft approach and our proposed regularizer can be seen as an extension of the soft orthogonality (SO) regularizer [48] to well-conditioned matrices in the low-rank setting. As noted in [4], the SO regularizer only works well when the input matrix is of size m × n with m ≤ n. However, we avoid this issue since the regularizer is applied to the square r × r matrix S; an extension to convolutional layers is discussed in Section 5.1. In the context of low-rank training, the soft approach enables rank-adaptivity of the method.

## 4 Improving Conditioning Via Regularization

We design a computationally efficient regularizer R to control and decrease the condition number of each network layer during training. The regularizer R only acts on the small r×r coefficient matrices S of each layer and thus has a minimal memory and compute overhead over low-rank training. The regularizer is differentiable almost everywhere and compatible with automatic differentiation tools.

Additionally, R has a closed form derivative that enables an efficient and scalable implementation of ∇R. Furthermore, R is compatible with any rank-adaptive low-rank training scheme that ensures orthogonality of *U, V* , e.g., [55, 36, 37, 35].

Definition 1. We define the robustness regularizer R *for any* S ∈ R
r×r by

$$\mathcal{R}(S)=\|S^{\top}S-\alpha_{S}^{2}I\|,\qquad\textit{where}\qquad\alpha_{S}^{2}=\frac{1}{r}\|S\|^{2}$$
2(2)
and I = Ir is the r × r *identity matrix.* The regularizer R can be viewed as an extension of the soft orthogonal regularizer [48, 4] where we penalize the distance of S
⊤S to the well-conditioned matrix α 2 S
I. Here αS is chosen such that
∥S∥ = ∥αSI∥. Moreover, R is also a scaled standard deviation of the squared singular values {ςi(S)
2}
ri=1, i.e.,

$$\frac{1}{r}{\cal R}(S)^{2}=\frac{1}{r}\sum_{i=1}^{r}(\varsigma_{i}(S)^{2})^{2}-\left(\frac{1}{r}\sum_{i=1}^{r}\varsigma_{i}(S)^{2}\right)^{2}.\tag{3}$$

$$\left(2\right)$$
$\mathrm{SU}(\top)\in$
See Appendix C for the proof. Therefore, R is a unitarily invariant regularizer; namely, R(USV ⊤) =
R(S) for orthogonal *U, V* . These two forms of R are useful in the properties shown below.

Proposition 1. *The gradient of* R in (2) *is given by* ∇R(S) = 2S(S
⊤S − α 2 S
I)/R(S).

See Appendix C for the proof. The gradient computation consists only of r × r matrix multiplications and a Frobenius norm evaluation. Thus ∇R is computationally efficient for r ≪ m. Further, its closed form enables a straight-forward integration into existing optimizers such as Adam or SGD applied to S. Proposition 2 (Condition number bound). For any S ∈ R
r×r*there holds*

$$\kappa(S)\leq\exp\bigg(\frac{1}{\sqrt{2}\varsigma_{r}(S)^{2}}\mathcal{R}(S)\bigg).\quad\quad(4)$$

Table 1: VGG16 on UCM data. Comparison of regularized LoRA and DLRT trained networks under the ℓ 2-FGSM attack. Orthogonality of *U, V*
increases adversarial performance significantly.

See Appendix C for the proof. Thus, if ςr(S) is not too small, we can use R(S) as a good measure for the conditioning of S. Note that the

| Method                | c.r. [%]   | clean Acc [%] ℓ 2 -FGSM, ϵ = 0.1   |       |
|-----------------------|------------|------------------------------------|-------|
| Non-regularized DLRT  | 95.30      | 93.92                              | 72.41 |
| RobustDLRT, β = 0.075 | 95.84      | 94.61                              | 78.68 |
| LoRA, β = 0.075       | 95.83      | 88.57                              | 73.81 |

4

10 2 0 250000 optimizer step 1.0 1.5 2.0 2.5 3.0 3.5
= 0.01
= 0.05 = 0.1 = 0.15
= 0.2 0 250000 optimizer step 10 1 10 0 10 1= 0.01
= 0.05 = 0.1 = 0.15 = 0.2 10 1
= 0.01 = 0.05 = 0.1
= 0.15
= 0.2 0 250000 optimizer step 10 0
(a) κ(S(t))
(b) R(S(t))
singular value truncation used in rank-adaptive methods ensures that ςr(S) is always sufficiently large. Figures 2a and 2b show the dynamics of R(S(t)) and κ(S(t)) during low-rank regularized training; we see that κ(S(t)) decays as R(S(t)) decays, validating Proposition 2. Remark 1. When U, V *are not orthonormal, e.g., in simultaneous gradient descent training (LoRA),*
the smallest n − r singular values of USV ⊤ *are often zero-valued; thus, the bound of Equation* (4)
is not useful. Table 1 shows that the clean accuracy and adversarial accuracy of regularized LoRA is significantly lower than standard or regularized training with orthonormal *U, V* . We now study the stability of the regularizer when applied to a least squares regression problem, i.e.,
given a fixed M ∈ R
r×r we seek to minimize J (S) := βR(S) + 12
∥S − M∥
2 over S ∈ R
r×r.

Proposition 3. Consider the dynamical system generated by the gradient flow of J *; namely,* S˙(t) +
β∇R(S(t)) + S(t) = M. Then for any t ≥ 0 *we have the long-time stability estimate*

$$\frac{1}{2}\|S(t)-M\|^{2}+2\beta\int_{0}^{t}e^{\tau-t}{\cal R}(S(\tau))\,{\rm d}\tau\leq\frac{1}{2}e^{-t}\|S(0)-M\|^{2}+2(1-e^{-t})\beta(1+2\beta)\|M\|^{2}.\tag{5}$$

See Appendix C for the proof. We note that unlike standard ridge and lasso regularizations methods, R lacks convexity; thus long-time stability of the regularized dynamics is not obvious. However, ∇R possesses monotonicity properties that we leverage to show in (5) that the growth in J only depends on β, M, and the initial loss. Moreover, for large t, the change in the final loss by the regularizer only depends on β and the true solution M and not the specific path S(t). While training on the non-convex loss will not provide the same theoretical properties as the convex least-square loss used in Proposition 3, the experiments in Figure 2 give confidence that adding our regularizer does not yield a relatively large change in the loss decay rate over moderate training regimes. Particularly, we observe empirically in Figure 2 that the condition number κ(S) of decreases alongside the regularizer value R during training.

Remark 2. We note R2can also be used in place of R. While R2*is differentiable at* R(S) = 0, we choose R *as our regularizer due to the proper scaling in* (4).

## 5 **A Rank-Adaptive And Adversarial Robustness Increasing Dynamical Low-Rank** Training Scheme

In this section we integrate the regularizer R into a rank-adaptive, orthogonality preserving, and efficient low-rank training scheme. We are specifically interested in a training method that 1) enables separation of the spectral dynamics of the coefficients S from the bases *U, V* and 2) ensures orthogonality of *U, V* at all times during training to obtain control layer conditioning in a compute and memory efficient manner. Popular schemes based upon simultaneous gradient descent of the low-rank factors such as LoRA [17] are not suitable here. These methods typically do not ensure orthogonality of U and V . Consequently, R(USV ⊤) ̸= R(S), and this fact renders evaluation of the regularizer R computationally inefficient. Thus we adapt the two-step scheme of [36] which ensures orthogonality of *U, V* . The method dynamically reduces or increases the rank of the factorized layers depending on the training dynamics and the complexity of the learning problem at hand. Consequently, the rank of each layer is no longer a hyper-parameter that needs fine-tuning, c.f. [17, 35], but is rather an interpretable measure for the inherent complexity required for each layer.

To facilitate the discussion, we define Le = L + βR as the regularized loss function of the training process with regularization parameter β > 0. To construct the method we consider the (stochastic)
gradient descent-based update of a single weight matrix Wt+1 = Wt+1 − λ∇W Le for minimizing Le with step size λ > 0. The corresponding continuous time gradient flow reads W˙ (t) = −∇W Le(W(t)),
which is a high-dimensional dynamical system with a steady state solution. We draw from established dynamical low-rank approximation (DLRA) methods, which were initially proposed for matrixvalued dynamical systems [20]. DLRA was recently extended to neural network training [38, 53, 36, 37, 22, 16] to formulate a consistent gradient flow evolution for the low-rank factors U, S, and V .

The DLRA method constrains the trajectory of W to the manifold Mr, consisting of n × n matrices with rank r, by projecting the full dynamics W˙ onto the local tangent space of Mr via an orthogonal projection, see Figure 3. The low-rank matrix is represented as USV ⊤ ∈ Mr, where U ∈ R
n×r and V ∈ R
n×r have orthonormal columns and S ∈ R
r×ris full-rank (but not necessarily diagonal).

An explicit representation of the tangent space leads to equations for the factors U, S, and V in [20, Proposition 2.1]. However, following these equations requires a prohibitively small learning rate due to the curvature of the manifold [29]. Therefore, specialized integrators have been developed to accurately navigate the manifold with reasonable learning rates [29, 6, 5]. Below we list the method of [36] with the changes introduced by adding our robustness regularizer. We call the resulting scheme *RobustDLRT*, and a single iteration of RobustDLRT is specified in Algorithm 1.

Basis Augmentation: The method first augments the current bases U
t, V tat optimization step t by their gradient dynamics ∇U L, ∇V L via

$$\begin{array}{l}{{\hat{U}=\mathtt{orth}([U^{t}\mid\nabla_{U}{\mathcal{L}}(U^{t}S^{t}V^{t,\top})])\in\mathbb{R}^{n\times2r},}}\\ {{\hat{V}=\mathtt{orth}([V^{t}\mid\nabla_{V}{\mathcal{L}}(U^{t}S^{t}V^{t,\top})])\in\mathbb{R}^{n\times2r},}}\end{array}$$
$$(6)$$

to double the rank of the low-rank representation and subsequently creates orthonormal bases U, 
b Vb. Here orth(A) denotes an orthonormal basis for the range of A and | denotes horizontal concatenation of matrices. Since R(USV ⊤) = R(S), ∇UR(USV ⊤) =
∇V R(USV ⊤) = 0; hence ∇U Le = ∇U L and ∇V Le =
∇V L are used in (6). The span of Ub contains U
t, which is needed to ensure of the loss does not increase during augmentation, and a first-order approximation of span(U
t+1)
using the exact gradient flow for U, see [36, Theorem 2] for details. Geometrically, the latent space

$${\mathcal{S}}=\{{\hat{U}}Z{\hat{V}}^{\top}:Z\in\mathbb{R}^{2r\times2r}\}$$
2r×2r} (7)
can be seen as subspace3 of the tangent plane of Mr at U
tS
tV
t,⊤, see Figure 3.

Latent Space Training: We update the latent coefficients Sb via a Galerkin projection of the training dynamics onto the latent space S. The latent coefficients Sb are updated by integrating the projected gradient flow

$$(7)$$

Figure 3: Geometric interpretation of Algorithm 1. First, we compute the parametrization of the tangent plane TMr. Then we compute the projected gradient update with ∇SbL. Lastly, we retract the updated coefficients back onto the manifold Mr. The regularizer R steers training to regions of Mr with lower curvature.

Algorithm 1: Single iteration of RobustDLRT.

Input :Initial orthonormal bases *U, V* ∈ R
n×rand diagonal S ∈ R
r×r; ϑ: singular value threshold for rank truncation; λ: learning rate.

1 Evaluate L(USV ⊤) /* Forward evaluate */ 2 GU ← ∇U L(USV ⊤); GV ← ∇V L(USV ⊤) /* Backprop on basis */
3 Ub ← orth([U | GU ]); Vb ← orth([V | GV ]) /* augmentation in parallel */
4 Sb ← Ub⊤USV ⊤Vb /* coefficient augmentation */
5 Sb ← coefficient_update(S, s b ∗*, λ, β*) /* regularized coefficient training */ 6 *U, S, V* ←truncation(S, b U, b Vb) 7 def coefficient_update(Sb0: coefficient, s∗: \# local steps, λ: learning rate, β*: robustness* regularization weight):
8 for s = 1*, . . . , s*∗ do 9 GS ← −λ∇SbL(UbSbs−1Vb ⊤) − β∇Sbs R(Sbs)
10 Sbs ← Sbs−1 + optim(GS) /* optimizer update, e.g., SGD or Adam */
11 return Sbs∗ 12 def truncation(Sb: augmented coefficient, Ub: augmented basis, Vb*: augmented co-basis )*: 13 Pr1, Σr1, Qr1 ← truncated svd(Se) with threshold ϑ to new rank r1 14 U ← UPb r1; V ← V Qb r1 /* Basis update */
15 S ← Σr1 /* Coefficient update with diagonal Σr1 */
16 return *U, S, V*
˙Sb = −Ub⊤∇W LeVb = −∇SbLe using stochastic gradient descent or an other suitable optimizer for a number of s∗ local iterations, i.e.,
Sbs+1 = Sbs − λ∇SbL − β∇SbR(Sbs), s = 0*, . . . , s*∗ − 1. (8)
Equation (8) is initialized with Sb0 = Ub⊤U
tS
tV
t,⊤Vb ∈ R
2r×2r, and we set S˜ = Sˆs∗
Truncation: Finally, the latent solution UbS˜Vb ⊤ is retracted back onto the manifold Mr. The retraction can be computed efficiently by using a truncated SVD of S˜ that discards the smallest r singular values. To enable rank adaptivity, the new rank r1 instead of r can be chosen by a variety of criteria, e.g., a singular value threshold ∥[ςr1, . . . , ς2r]∥2 < ϑ. Once a suitable rank is determined, the bases U and V are updated by discarding the basis vectors corresponding to the truncated singular values.

Remark 3. We note that R will likely increase the smallest singular values of Sˆ *to improve* κ(Sˆ).

This could theoretically increase the truncated rank over non-regularized DLRT and result in less compression. However, we find in the experiments in Section 6 that RobustDLRT has similar compression rates to DLRT. Computational cost: The computational cost of RobustDLRT is asymptotically the same as LoRA,
since the reconstruction of the full weight matrix W is never required. The orthonormalization, computation of the regularizer R, and the SVD for accounts for O(nr2), O(r 3), O(r 3) floating point operations, respectively. When using multiple coefficient update steps s∗ > 1, the amortized cost is lower than that of LoRA, since only the gradient with respect to Sb is required in most updates.

While the regularizer may be applied to full-rank baseline models, its O(n 3) computational scaling significantly increases training costs.

## 5.1 Extension To Convolutional Neural Networks

The convolution layer map in 2D CNNs translates a W × H image with NI in-features to NO
out-features. Using tensors, this map is expressed as Y = C ∗ X where X ∈ R
NI×W×H, Y ∈
R

NO×W×H, and C ∈ R
NO×NI×SW ×SH is the convolutional kernel with a convolution window size SW × SH. Neglecting the treatment of strides and padding, C ∗ X is given as a tensor contraction by Y (*o, w, h*) = Pc,sw,sh C(o, c, sw, sh)X(c, w + sw, h + sh) (9)

| most setups. All runs where RobustDLRT surpasses the uncompressed baseline are highlighted. UCM Data Clean Acc [%] for ℓ 2 -FGSM, ϵ Acc [%] for Jitter, ϵ Acc [%] for Mixup, ϵ Method c.r. [%] Acc. [%] 0.05 0.1 0.3 0.035 0.045 0.025 0.1 0.75 G16 Baseline 0.0 94.40±0.72 86.71±1.90 76.40±2.84 54.96±2.99 89.58±2.99 85.05±3.40 77.77±1.61 37.25±3.66 23.05±3.01 DLRT 95.30 93.92±0.23 87.95±1.02 72.41±2.08 43.39±4.88 83.99±1.22 67.41±1.63 85.79±1.51 40.42±2.89 20.13±2.92 RobustDLRT 95.84 94.61±0.35 89.12±1.33 78.68±2.30 53.30±3.14 88.33±1.20 79.81±0.93 90.33±0.90 70.12±3.08 47.31±2.78 11 Baseline 0.0 94.23±0.71 89.93±1.33 78.66±2.46 39.45±2.98 90.25±1.66 85.24±1.90 83.10±1.47 40.34±4.88 22.01±3.21 DLRT 94.89 93.70±0.71 86.58±1.22 67.55±2.16 28.92±2.65 83.90±1.36 63.41±1.39 87.15±1.18 40.17±4.96 14.18±3.78 RobustDLRT 94.59 93.57±0.84 87.90±0.91 72.96±1.55 32.85±2.46 86.77±0.76 74.31±1.50 88.00±1.13 60.97±4.18 28.56±3.64 16b Baseline 0.0 96.72±0.36 93.02±0.38 92.18±0.31 89.71±0.28 93.71±1.22 93.21±1.17 89.62±1.81 51.05±3.17 43.91±3.97 DLRT 86.7 96.38±0.60 91.21±0.44 82.10±0.32 62.45±0.41 86.67±1.05 79.81±0.81 80.48±1.82 41.52±3.24 35.91±3.76 RobustDLRT 87.9 96.41±0.67 92.57±0.34 85.67±0.41 69.94±0.42 91.03±0.86 84.19±1.39 87.33±1.81 46.39±2.75 40.76±3.88 16 Baseline 0.0 89.82±0.45 76.22±1.38 63.78±2.01 34.97±2.54 78.60±1.12 73.54±1.55 71.51±1.31 37.36±2.60 16.12±2.12 DLRT 94.37 89.23±0.62 74.07±1.23 59.55±1.79 28.74±2.21 72.51±1.04 66.21±1.41 79.56±1.15 59.88±2.26 38.98±1.94 RobustDLRT 94.18 89.49±0.58 76.04±1.18 62.08±1.69 32.77±2.04 75.53±0.98 69.93±1.22 87.62±1.07 84.80±2.01 81.26±2.15 Baseline 0.0 88.34±0.49 75.89±1.42 64.21±1.96 31.76±2.45 74.96±1.09 68.59±1.63 74.77±1.26 40.88±2.58 08.95±1.98 DLRT 95.13 88.13±0.56 72.02±1.34 55.83±1.92 21.59±2.16 66.98±1.05 58.57±1.55 79.42±1.08 47.95±2.18 22.92±1.77 RobustDLRT 94.67 87.97±0.52 76.04±1.26 63.82±1.83 30.77±2.30 71.06±1.00 65.63±1.38 84.93±1.10 78.35±1.89 65.93±2.04 6b Baseline 0.0 95.42±0.35 79.94±0.95 63.66±1.62 32.09±2.05 84.65±0.88 77.20±1.04 52.17±1.49 16.03±2.34 13.29±2.01 DLRT 73.42 95.39±0.41 79.50±0.91 61.62±1.48 30.32±1.94 83.33±0.80 76.16±0.95 58.32±1.44 17.43±2.28 14.49±1.92 RobustDLRT 75.21 94.66±0.38 82.03±0.88 69.29±1.43 38.05±1.99 87.97±0.75 83.03±0.91 74.49±1.32 27.80±2.11 18.34±1.87   |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

where sw and sh range from −SW /2*, . . . , S*W /2 and −SH/2*, . . . , S*H/2 respectively, and o = 1*, . . . , N*O, w = 1*, . . . , W*, and h = 1*, . . . , H*. DLRT was extended to convolutional layers in [53] by compressing C with a Tucker factorization.

Little is gained in compressing the window modes as they are typically small. Thus, we only factorize C in the feature modes with output and input feature ranks rO ≪ NO and rI ≪ NI as C(o, i, sw, sh) = PrI ,rO
qO,qI=1 UO(o, qO)UI (i, qI )S(qO, qI , sw, sh). (10)

$$q_{O})U_{I}(i,q_{I})S(q_{O},q_{I},s_{w},s_{h}).$$
$$(10)$$

Substituting (10) into (9) and rearranging indices yields

$$Y(o,w,h)=\sum_{q_{0}}U_{O}(o,q_{0})\widetilde{Y}(q_{O},w,h),\tag{11a}$$ $$\widetilde{Y}(q_{O},w,h)=\sum_{q_{I},s_{w},s_{h}}S(q_{O},q_{I},s_{w},s_{h})\widetilde{X}(q_{I},w+s_{w},h+s_{h}),$$ (11b) $$\widetilde{X}(q_{I},w+s_{w},h+s_{h})=\sum_{c}U_{I}(c,q_{I})X(c,w+s_{w},h+s_{h}).\tag{11c}$$
Remark 4. *Aside from the prolongation* (11a) *and retraction* (11c) from/to the low-rank latent space,
the low-rank convolution map (11) *features a convolution* (11b) similar to (9) *but in the reduced*
dimension low-rank latent space.
Robustness regularization for convolutional layers. The contractions in (9) and (11b) show that the output channels arise from a tensor contraction of the input channel and window modes; hence, both (9) and (11b) can be viewed as matrix-vector multiplications where C is matricised on the output channel mode; i.e., C → Mat(C) ∈ R
NO×NISW SH and S → Mat(S) ∈ R
rO×rISW SH . Therefore, we only regularize Mat(S) with our robustness regularizer. Moreover, we assume rO ≤ rISW SH, which is almost always the case since rO and rI are comparable and SW SH ≫ 1. Then we regularize convolutional layers by R(Mat(S)
⊤) so that SS⊤ is an rO × rO matrix, which is computationally efficient. We remark that the extension of Algorithm 1 to a tensor-valued layer with Tucker factorization only requires to change the truncation step; the SVD is replaced by a truncated Tucker decomposition of S. The Tucker bases UO and UI can be augmented in parallel similarly to the matrix case.

## 6 Numerical Results

We evaluate the numerical performance of Algorithm 1 compared with non-regularized low-rank training, baseline training, and several other robustness-enhancing methods the VGG16, VGG11, and ViT-16b architectures and University of California, Merced (UCM), Cifar10, and ImageNet1k datasets. Detailed descriptions of the models, datasets, pre-processing, training hyperparameters,

Table 3: Imagenet Benchmark, ViT-32l trained with baseline Adam, DLRT, and RobustDLRT. We report the low-rank results for unregularized β = 0.0 and the best performing β, given in Table 9. Algorithm 1 (RobustDLRT) is able to match or surpass baseline adversarial accuracy values in most setups. All runs where RobustDLRT surpasses the uncompressed baseline are highlighted.

Top1/Top5 Clean Top1/Top5 Acc [%] for ℓ

2-**FGSM,** ϵ Top1/Top5 Acc [%] for **Jitter,** ϵ

Method c.r. [%] Acc. [%] 0.05 0.1 0.3 0.035 0.045 Baseline 0 74.37/92.20 43.58/73.75 31.42/63.42 16.03/43.41 43.09/78.24 35.57/74.96 DLRT 58.02 72.27/90.06 42.70/70.43 30.32/60.90 15.47/40.58 43.98/74.49 38.44/ 71.31 RobustDLRT 57.98 72.25/90.03 43.17/71.58 35.11 /62.82 25.24/50.65 48.22 /77.35 43.51/75.14

and competitor methods are given in Appendix B. A reference implementation is provided at https://github.com/ScSteffen/RobustDLRT. We measure the compression rate (c.r.) as the relative amount of pruned parameters of the target network, i.e. c.r. = (1 −
\#params low-rank net
\#params baseline net 
) × 100.

The reported numbers in the tables represent the average over 10 stochastic training runs. We observe in Table 2 that clean accuracy results exhibit a standard deviation of less than 0.8%; the standard deviation increases with the attack strength ϵ for all tests and methods. This observation holds true for all presented results; thus, we omit the error bars in the other tables for the sake of readability. UCM dataset We observe in Table 2 that Algorithm 1 can compress the VGG11, VGG16 and ViT-16b networks equally well as the non-regularized low-rank compression and achieves the first goal of high compression values of up to 94% reduction of trainable parameters. Furthermore, the clean accuracy is similar to the non-compressed baseline architecture; thus, we achieve the second goal of (almost) loss-less compression. Noting the adversarial accuracy results under the ℓ 2-FGSM, Jitter, and Mixup attacks with various attack strengths ϵ, we observe that across all tests, the regularized low-rank network of Algorithm 1 significantly outperforms the non-regularized low-rank network. For the ℓ 2-FGSM attack, our method is able to recover the adversarial accuracy of the baseline network. For Mixup, the regularization almost doubles the baseline accuracy for VGG16. By targeting the condition number of the weights, which gives a bound on the *relative* growth of the loss w.r.t. the size of the input, we postulate that the large improvement could be attributed to the improved robustness against the scale invariance attack [27, Section 3.3] included in Mixup. We refer the reader to Appendix B.1.4 for a precise definition of the Mixup attack featuring scale invariance. However, this hypothesis was not further explored and is delayed to a future work. Finally, we are able to recover half of the lost accuracy in the Jitter attack. Overall, we achieved the third goal of significantly increasing adversarial robustness of the compressed networks. We refer to Table 9 for the used values of β and Appendix A.1 for extended numerical results. Cifar10 dataset We repeat the methodology of the UCM dataset for Cifar10, and observe similar computational results in Table 2. Furthermore, we compare our method in Table 4 to several methods of the recent literature, see Section 3. We compare the adversarial accuracy under the ℓ 1-FGSM attack, see Appendix B.1.2 for details, for consistency with the literature results. We find that our proposed method achieves the highest adversarial validation accuracy for all attack strengths ϵ, even surpassing the baseline adversarial accuracy. Additionally, we find an at least 15% higher compression ratio with Robust- DLRT than the second best compression method, CondLR [35]. A similar experiment for the Projected Gradient Descent (PGD) attack [30] is given in Appendix A.2.

ImageNet1k dataset Finally we repeat the methodology for the ImageNet1k dataset, using the ViT-32l vision transformer trained from an ImageNet21k checkpoint, and report the results in Table 3. The hyperparameters are obtained by Table 4: Comparison to literature on CIFAR10 with VGG16 under the ℓ 1-FGSM attack. The first three rows list the computed mean over 10 random initializations. The values of all other methods, given below the double rule, are taken from [35, Table 1]. RobustDLRT has higher adversarial accuracy at higher compression rates than all listed methods.

9

| ℓ 1 -FGSM, ϵ        |          |       |       |       |       |
|---------------------|----------|-------|-------|-------|-------|
| Method              | c.r. [%] | 0.0   | 0.002 | 0.004 | 0.006 |
| Baseline            | 0        | 89.83 | 78.61 | 64.66 | 53.71 |
| DLRT                | 94.58    | 89.55 | 74.71 | 59.61 | 47.56 |
| RobustDLRT β = 0.15 | 94.35    | 89.35 | 78.72 | 66.02 | 54.15 |
| Cayley SGD [25]     | 0        | 89.62 | 74.46 | 58.16 | 45.29 |
| Projected SGD [1]   | 0        | 89.70 | 74.55 | 58.32 | 45.74 |
| CondLR [35] τ = 0.5 | 50       | 89.97 | 72.25 | 60.19 | 50.17 |
| CondLR [35] τ = 0.5 | 80       | 89.33 | 68.23 | 48.54 | 36.66 |
| LoRA [17]           | 50       | 89.97 | 67.71 | 48.86 | 38.49 |
| LoRA [17]           | 80       | 88.10 | 64.24 | 42.66 | 29.90 |
| SVD prune [51]      | 50       | 89.92 | 67.30 | 47.77 | 36.98 |
| SVD prune [51]      | 80       | 87.99 | 63.57 | 42.06 | 29.27 |

an initial sweep and reported in Tables 8 and 9. RobustDLRT consistently yields higher Top-1/Top-5 accuracy across ℓ 2-FGSM and Jitter attacks than DLRT, with especially pronounced gains at larger perturbations (e.g., +9 points in Top-1 accuracy under ℓ 2-FGSM ϵ = 0.3). These trends are consistent with our ViT experiments in Table 2, demonstrating that adversarial regularization enhances robustness without compromising scalability. We benchmark the training runtime of one ImageNet epoch on an A100 80GB GPU. DLRT requires 26m 07s, while RobustDLRT (with the regularizer) requires 27m 51s, corresponding to an overhead of approximately 3%. This overhead can likely be reduced with further implementation optimizations, indicating that our approach is computationally scalable. Black-box attacks We investigate the scenario where an attacker has knowledge of the used model architecture, but not of the low-rank compression. We use the Imagenet-1k pretrained VGG16 and VGG11 and re-train it with Algorithm 1 and baseline training on the UCM data using the same training hyperparameters. Then we generate adversarial examples with the baseline network and evaluate the performance on the low-rank network with and without regularization. The results are given in Table 5. In this scenario, the weights from low-rank training, being sufficiently far away from the baseline, provide an effective defense against the attack. Further, the proposed regularization significantly improves the adversarial robustness when compared to the unregularized low-rank network. Even for extreme attacks with ϵ = 1, the regularized network achieves 84.76% and 87.33% accuracy for VGG16 and VGG11 respectively.

Adversarial Training We evaluate the performance of low-rank training for VGG16 on the UCM dataset using adversarial training. Following [13], we use the ℓ 2-FGSM attack for different values of ϵ and train on both 50% clean and attacked images per batch. The results reported in Table 6 illustrate that RobustDLRT is both compatible with and able to benefit from adversarial training. DLRT without regularization benefits from adversarial training, but exhibits a clear margin to RobustDLRT. Additionally, RobustDLRT is able to approximately match the non-compressed baseline. Table 5: UCM dataset - Black-box attack. Adversarial images with the ℓ 2-FGSM attack are generated by the baseline network for different values of ϵ. The baseline, DLRT (β = 0), and RobustDLRT (β = 0.075) networks are then evaluated on these images. Regularized low-rank compression achieves high adversarial accuracy, even under strong attacks.

| 2 -FGSM, ϵ     |                |       |       |       |       |       |       |       |
|----------------|----------------|-------|-------|-------|-------|-------|-------|-------|
| Method         | c.r. [%]       | 0.05  | 0.1   | 0.25  | 0.5   | 0.75  | 1.0   |       |
| ℓ              |                |       |       |       |       |       |       |       |
| 16 Baseline    | 0.0            | 86.71 | 76.40 | 48.76 | 39.33 | 35.23 | 33.23 |       |
| β = 0          | 95.30          | 93.03 | 91.81 | 88.09 | 83.14 | 78.95 | 76.00 |       |
| VGG            | β = 0.05 95.15 | 92.66 | 92.47 | 91.33 | 88.76 | 86.85 | 84.76 |       |
| 1 Baseline     | 0.0            | 89.93 | 78.66 | 60.76 | 45.23 | 38.38 | 35.52 |       |
| GG1            | β = 0          | 95.82 | 92.76 | 91.81 | 88.25 | 84.09 | 80.57 | 77.71 |
| β = 0.05 96.12 | 92.95          | 92.66 | 92.00 | 91.04 | 88.66 | 87.33 |       |       |
| V              |                |       |       |       |       |       |       |       |

Table 6: UCM dataset - Adversarial Training. VGG16 is trained on 50% clean images and 50% images attacked with ℓ 2-FGSM for various ϵ. The displayed numbers are the mean of 5 repeated runs. RobustDLRT (β = 0.075) is superior to DLRT (β = 0) and is able to approximately match the non-compressed baseline.

## 7 Conclusion

RobustDLRT enables highly compressed neural networks with strong adversarial robustness by controlling the spectral properties of low-rank factors. The method is efficient, rank-adaptive, and yields an up to 94% parameter reduction across a diverse suite of models and datasets. The method achieves competitive accuracy, even for strong adversarial attacks, surpassing the current literature results by a significant margin. Therefore, we conclude the proposed method scores well in the combined metric of compression, accuracy and adversarial robustness. The accomplished high compression and adversarial robustness advance computer vision models and enable broader applications on resource-constrained edge devices. These achievements also enhance energy efficiency and trustworthiness, positively impacting society. The regularization and condition number bounds further improve interpretability, which is crucial for transparency and accountability in critical decision-making when applying the proposed methods.

| 2 -FGSM, ϵ      |          |       |       |       |       |       |
|-----------------|----------|-------|-------|-------|-------|-------|
| Method          | c.r. [%] | 0.0   | 0.1   | 0.5   | 0.75  | 1.0   |
| ℓ               |          |       |       |       |       |       |
| Baseline        | 0.0      | 92.61 | 91.91 | 91.90 | 89.61 | 89.91 |
| β = 0           | 94.46    | 92.55 | 91.91 | 87.98 | 85.37 | 82.96 |
| β = 0.075 94.19 | 92.49    | 92.49 | 90.98 | 89.56 | 89.42 |       |

## Acknowledgments And Disclosure Of Funding

This manuscript has been authored by UT-Battelle, LLC under Contract No. DE-AC05-00OR22725 with the U.S. Department of Energy. The United States Government retains and the publisher, by accepting the article for publication, acknowledges that the United States Government retains a non-exclusive, paid-up, irrevocable, world-wide license to publish or reproduce the published form of this manuscript, or allow others to do so, for United States Government purposes. The Department of Energy will provide public access to these results of federally sponsored research in accordance with the DOE Public Access Plan(http://energy.gov/downloads/doe-public-access-plan). This material is based upon work supported by the Laboratory Directed Research and Development Program of Oak Ridge National Laboratory (ORNL), managed by UT-Battelle, LLC for the U.S.

Department of Energy under Contract No. De-AC05-00OR22725. S. Schotthöfer, H. L. Yang, and S. Schnake were supported by the Artificial Intelligence Initiative of the Laboratory Directed Research and Development Program of Oak Ridge National Laboratory (ORNL), managed by UT-Battelle, LLC for the U.S. Department of Energy under Contract No. De- AC05-00OR22725. This research used resources of the Compute and Data Environment for Science (CADES) at the Oak Ridge National Laboratory, which is supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC05-00OR22725.

## References

[1] P.-A. Absil and J. Malick. Projection-like retractions on matrix manifolds. SIAM Journal on Optimization, 22(1):135–158, 2012.

[2] A. Aghajanyan, S. Gupta, and L. Zettlemoyer. Intrinsic dimensionality explains the effectiveness of language model fine-tuning. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 7319–7328, 2021.

[3] C. Anil, J. Lucas, and R. Grosse. Sorting out Lipschitz function approximation. In International conference on machine learning, pages 291–301. PMLR, 2019.

[4] N. Bansal, X. Chen, and Z. Wang. Can we gain more from orthogonality regularizations in training deep networks? *Advances in Neural Information Processing Systems*, 31, 2018.

[5] G. Ceruti, J. Kusch, and C. Lubich. A rank-adaptive robust integrator for dynamical low-rank approximation. *BIT Numerical Mathematics*, pages 1–26, 2022.

[6] G. Ceruti and C. Lubich. An unconventional robust integrator for dynamical low-rank approximation. *BIT Numerical Mathematics*, 62(1):23–44, 2022.

[7] T. Chen, H. Zhang, Z. Zhang, S. Chang, S. Liu, P.-Y. Chen, and Z. Wang. Linearity grafting:
Relaxed neuron pruning helps certifiable robustness, 2022.

[8] G. Cheng, X. Sun, K. Li, L. Guo, and J. Han. Perturbation-seeking generative adversarial networks: A defense framework for remote sensing image scene classification. *IEEE Transactions* on Geoscience and Remote Sensing, 60:1–11, 2022.

[9] M. Cisse, P. Bojanowski, E. Grave, Y. Dauphin, and N. Usunier. Parseval networks: Improving robustness to adversarial examples. In D. Precup and Y. W. Teh, editors, *Proceedings of the* 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 854–863. PMLR, 06–11 Aug 2017.

[10] M. Cisse, P. Bojanowski, E. Grave, Y. Dauphin, and N. Usunier. Parseval networks: Improving robustness to adversarial examples. In *International Conference on Learning Representations*
(ICLR), 2017.

[11] W. Czaja, N. Fendley, M. Pekala, C. Ratto, and I.-J. Wang. Adversarial examples in remote sensing. In Proceedings of the 26th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, SIGSPATIAL '18, page 408–411, New York, NY, USA,
2018. Association for Computing Machinery.

[12] E. L. Denton, W. Zaremba, J. Bruna, Y. LeCun, and R. Fergus. Exploiting linear structure within convolutional networks for efficient evaluation. *Advances in neural information processing* systems, 27, 2014.

[13] I. J. Goodfellow, J. Shlens, and C. Szegedy. Explaining and harnessing adversarial examples.

arXiv preprint arXiv:1412.6572, 2014.

[14] Y. Guo, A. Yao, and Y. Chen. Dynamic network surgery for efficient dnns. Advances in neural information processing systems, 29, 2016.

[15] M. Hein and M. Andriushchenko. Formal guarantees on the robustness of a classifier against adversarial manipulation. *Advances in neural information processing systems*, 30, 2017.

[16] A. Hnatiuk, J. Kusch, L. Kusch, N. R. Gauger, and A. Walther. Stochastic aspects of dynamical low-rank approximation in the context of machine learning. *Optimization Online*, 2024.

[17] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. LoRA:
Low-rank adaptation of large language models. *arXiv preprint arXiv:2106.09685*, 2021.

[18] T. Jian, Z. Wang, Y. Wang, J. Dy, and S. Ioannidis. Pruning adversarially robust neural networks without adversarial examples, 2022.

[19] A. Jordao and H. Pedrini. On the effect of pruning on adversarial robustness. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1–11, 2021.

[20] O. Koch and C. Lubich. Dynamical low-rank approximation. *SIAM Journal on Matrix Analysis* and Applications, 29(2):434–454, 2007.

[21] A. Kurakin, I. J. Goodfellow, and S. Bengio. Adversarial machine learning at scale. In International Conference on Learning Representations, 2017.

[22] J. Kusch, S. Schotthöfer, and A. Walter. An augmented backward-corrected projector splitting integrator for dynamical low-rank training. *arXiv preprint arXiv:2502.03006*, 2025.

[23] V. Lebedev, Y. Ganin, M. Rakhuba, I. Oseledets, and V. Lempitsky. Speeding-up convolutional neural networks using fine-tuned CP-decomposition. In International Conference on Learning Representations, 2015.

[24] H. Lee, S. Han, and J. Lee. Generative adversarial trainer: Defense to adversarial perturbations with GAN. *arXiv preprint arXiv:1705.03387*, 2017.

[25] J. Li, F. Li, and S. Todorovic. Efficient Riemannian optimization on the Stiefel manifold via the Cayley transform. In *International Conference on Learning Representations*, 2020.

[26] Z. Li, T. Chen, L. Li, B. Li, and Z. Wang. Can pruning improve certified robustness of neural networks?, 2022.

[27] J. Lin, C. Song, K. He, L. Wang, and J. E. Hopcroft. Nesterov accelerated gradient and scale invariance for adversarial attacks. In *International Conference on Learning Representations*,
2020.

[28] X. Liu, Y. Li, C. Wu, and C.-J. Hsieh. Adv-BNN: Improved adversarial defense through robust Bayesian neural network. In *International Conference on Learning Representations*, 2010.

[29] C. Lubich and I. V. Oseledets. A projector-splitting integrator for dynamical low-rank approximation. *BIT Numerical Mathematics*, 54(1):171–188, 2014.

[30] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu. Towards deep learning models resistant to adversarial attacks. In *International Conference on Learning Representations*, 2018.

[31] Y. Mao, K. Huang, C. Guan, G. Bao, F. Mo, and J. Xu. DoRA: Enhancing parameter-efficient fine-tuning with dynamic rank distribution. In L.-W. Ku, A. Martins, and V. Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11662–11675, Bangkok, Thailand, Aug. 2024. Association for Computational Linguistics.

[32] T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida. Spectral normalization for generative adversarial networks. In *International Conference on Learning Representations*, 2018.

[33] J. Nagy. Über algebraische gleichungen mit lauter reellen wurzeln. Jahresbericht der Deutschen Mathematiker-Vereinigung, 27:37–43, 1918.

[34] R. Nenov, D. Haider, and P. Balazs. (Almost) smooth sailing: Towards numerical stability of neural networks through differentiable regularization of the condition number, 2024.

[35] D. Savostianova, E. Zangrando, G. Ceruti, and F. Tudisco. Robust low-rank training via approximate orthonormal constraints. *Advances in Neural Information Processing Systems*,
36:66064–66083, 2023.

[36] S. Schotthöfer and M. P. Laiu. Federated dynamical low-rank training with global loss convergence guarantees. *arXiv preprint arXiv:2406.17887*, 2024.

[37] S. Schotthöfer, E. Zangrando, G. Ceruti, F. Tudisco, and J. Kusch. GeoLoRA: Geometric integration for parameter efficient fine-tuning. In *The Thirteenth International Conference on* Learning Representations, 2025.

[38] S. Schotthöfer, E. Zangrando, K. Jonas, G. Ceruti, and F. Tudisco. Low-rank lottery tickets:
finding efficient low-rank neural networks via matrix differential equations. In Advances in Neural Information Processessing Systems, 2022.

[39] L. Schwinn, R. Raab, A. Nguyen, D. Zanca, and B. Eskofier. Exploring misclassifications of robust neural networks to enhance adversarial attacks. *Applied Intelligence*, 53(17):19843–
19859, 2023.

[40] V. Sehwag, S. Wang, P. Mittal, and S. Jana. Hydra: Pruning adversarially robust neural networks.

In H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 19655–19666. Curran Associates, Inc.,
2020.

[41] R. Sharma, M. Gupta, and G. Kapoor. Some better bounds on the variance with applications.

Journal of Mathematical Inequalities, 4(3):355–363, 2010.

[42] S. P. Singh, G. Bachmann, and T. Hofmann. Analytic insights into structure and rank of neural network Hessian maps. In *Advances in Neural Information Processing Systems*, volume 34, 2021.

[43] Y. Su, G. Zhang, S. Mei, J. Lian, Y. Wang, and S. Wan. Reconstruction-assisted and distanceoptimized adversarial training: A defense framework for remote sensing scene classification.

IEEE Transactions on Geoscience and Remote Sensing, 61:1–13, 2023.

[44] F. Tramèr, A. Kurakin, N. Papernot, I. Goodfellow, D. Boneh, and P. McDaniel. Ensemble adversarial training: Attacks and defenses. *arXiv preprint arXiv:1705.07204*, 2017.

[45] L. N. Trefethen and D. Bau. *Numerical Linear Algebra*. SIAM, Philadelphia, PA, 1997. [46] M. Valipour, M. Rezagholizadeh, I. Kobyzev, and A. Ghodsi. Dylora: Parameter efficient tuning of pre-trained models using dynamic search-free low-rank adaptation. *arXiv preprint* arXiv:2210.07558, 2022.

[47] H. Wang, S. Agarwal, and D. Papailiopoulos. Pufferfish: Communication-efficient models at no extra cost. *Proceedings of Machine Learning and Systems*, 3:365–386, 2021.

[48] D. Xie, J. Xiong, and S. Pu. All you need is beyond a good init: Exploring better solution for training extremely deep convolutional neural networks with orthonormality and modulation. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 6176–6185, 2017.

[49] Y. Xu and P. Ghamisi. Universal adversarial examples in remote sensing: Methodology and benchmark. *IEEE Transactions on Geoscience and Remote Sensing*, 60:1–15, 2022.

[50] Y. Xu and P. Ghamisi. Universal adversarial examples in remote sensing: Methodology and benchmark. *IEEE Trans. Geos. Remote Sens.*, 60:1–15, 2022.

[51] H. Yang, M. Tang, W. Wen, F. Yan, D. Hu, A. Li, H. Li, and Y. Chen. Learning low-rank deep neural networks via singular vector orthogonality regularization and singular value sparsification. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pages 678–679, 2020.

[52] Y. Yang and S. Newsam. Bag-of-visual-words and spatial extensions for land-use classification.

In *Proceedings of the 18th SIGSPATIAL International Conference on Advances in Geographic* Information Systems, GIS '10, page 270–279, New York, NY, USA, 2010. Association for Computing Machinery.

[53] E. Zangrando, S. Schotthöfer, G. Ceruti, J. Kusch, and F. Tudisco. Rank-adaptive spectral pruning of convolutional layers during training. In Advances in Neural Information Processing Systems, 2024.

[54] H. Zhang, Y. Yu, J. Jiao, E. Xing, L. El Ghaoui, and M. Jordan. Theoretically principled trade-off between robustness and accuracy. In *International conference on machine learning*, pages 7472–7482. PMLR, 2019.

[55] Q. Zhang, M. Chen, A. Bukharin, P. He, Y. Cheng, W. Chen, and T. Zhao. AdaLoRA: Adaptive budget allocation for parameter-efficient fine-tuning. In The Eleventh International Conference on Learning Representations, 2023.

[56] J. Zhao, Z. Zhang, B. Chen, Z. Wang, A. Anandkumar, and Y. Tian. GaLore: Memory-efficient LLM training by gradient low-rank projection. In *International Conference on Machine* Learning, pages 61121–61143. PMLR, 2024.

[57] Q. Zhao, T. Königl, and C. Wressnegger. Non-uniform adversarially robust pruning. In I. Guyon, M. Lindauer, M. van der Schaar, F. Hutter, and R. Garnett, editors, Proceedings of the First International Conference on Automated Machine Learning, volume 188 of *Proceedings of* Machine Learning Research, pages 1/1–16. PMLR, 25–27 Jul 2022.

## A Additional Numerical Results A.1 Ucm Dataset

The numerical results for the whitebox ℓ 2-FGSM, Jitter, and Mixup adversarial attacks on the VGG16 and VGG11 architectures can be found in Figure 4, Figure 5, and Figure 6. The regularizer confidently increases the adversarial validation accuracy of the networks. In Table 10, we observe that the regularizer R(W) applied to the full weight matrices (and flattened tensors) W in baseline format is able to increase the adversarial robustness of the baseline network in the UCM/VGG16 test case. However, the increased adversarial robustness comes at the expense of some of the clean validation accuracy.

## A.2 Cifar10 Dataset

We run the same experiment in Table 4 but with the ℓ 2-PGD attack, which is an iterative version of ℓ 2-FGSM with an random perturbation of the input image as the initial condition [30]. Overall, we see that RobustDLRT is competitive with the other robustness-improving methods when the compression rate is taken into account. Table 7: Comparison to literature on CIFAR10 with VGG16 under the ℓ 2-PGD attack. The first three rows list the computed mean over 10 random initializations. The values of all other methods, given below the double rule, are taken from [35, Table 5]. RobustDLRT has competitive adversarial accuracy to all methods with a compression rate ≥ 80%.

ℓ 2**-PGD,** ϵ Method c.r. [%] 0.0 0.1 0.13 0.16 0.2 0.23 0.26 0.3 RobustDLRT β = 0.15 94.18 88.80 62.58 53.47 44.95 34.75 28.33 22.64 16.59 DLRT 94.53 88.58 59.34 50.06 41.50 31.82 25.67 20.48 15.04 Baseline 0 90.48 63.01 54.66 47.87 40.77 36.75 33.51 29.93

Cayley SGD [25] 0 89.62 67.68 59.38 51.09 40.87 34.46 29.21 23.62 Projected SGD [1] 0 89.70 67.64 59.25 51.06 40.86 34.51 29.19 23.64

CondLR [35] τ = 0.1 50 90.93 67.03 62.08 59.15 56.92 55.96 55.28 54.58

CondLR [35] τ = 0.5 50 89.97 64.84 60.25 57.75 56.03 55.21 54.75 54.25 CondLR [35] τ = 0.1 80 90.48 61.00 50.84 42.19 33.70 29.44 26.55 23.97

CondLR [35] τ = 0.5 80 89.33 57.45 46.35 37.20 28.30 23.82 20.65 17.84

LoRA [17] 50 89.97 55.74 45.11 36.86 29.62 26.28 24.02 21.84 LoRA [17] 80 88.10 51.40 39.70 30.12 20.97 16.29 13.15 10.37 SVD prune [51] 50 89.92 54.87 43.85 35.23 27.95 24.38 22.06 19.94 SVD prune [51] 80 87.99 50.64 39.06 29.57 20.16 15.49 12.22 9.57

## B Details To The Numerical Experiments Of This Work B.1 Recap Of Adversarial Attacks

In the following we provide the defintions of the used adversarial attacks. We use the implementation of [50] for the ℓ 2-FGSM, Jitter, and Mixup attack. For the ℓ 1-FGSM attack, we use the implementation of https://github.com/COMPiLELab/CondLR.

## B.1.1 ℓ 2**-Fgsm Attack**

The Fast Gradient Sign Method (FGSM)[21] is a single-step adversarial attack that perturbs an input in the direction of the gradient of the loss with respect to the input. Given a neural network classifier fθ with parameters θ, an input x, and its corresponding label y, the attack optimizes the cross-entropy loss LCE(fθ(x), y) by modifying x along the gradient's sign. The adversarial example is computed as:

$$x^{\prime}=x+\alpha\cdot\frac{\nabla_{x}\mathcal{L}_{\mathrm{CE}}(f_{\theta}(x),y)}{\|\nabla_{x}\mathcal{L}_{\mathrm{CE}}(f_{\theta}(x),y)\|_{\infty}},$$
$$(12)$$
$$(13)$$

, (12)
where α controls the perturbation magnitude. To ensure the perturbation remains bounded, the difference x
′ − x is clamped by an ϵ bound, i.e.,
$$x^{\prime}=x+\operatorname*{max}(-\epsilon,\operatorname*{min}(x^{\prime}-x,\epsilon)).$$
′ − *x, ϵ*)). (13)
The ℓ
1-FGSM attack [44] is used in the reference work of [35] and uses the same workflow as (B.1.1),
where (12) is changed to
 $\quad x^\prime=x+\alpha\cdot\dfrac{\text{sign}(\nabla_x\mathcal{L}_{\text{CE}}(f_\theta(x),y))}{\Sigma},$  I deviation of the data points in the training data, set and the size. 
where Σ denotes the standard deviation of the data-points in the training data-set and the sign of the gradient matrix is taken element wise.
$$(14)$$
$$\hat{z}=\text{Softmax}\left(\frac{s\cdot z}{\|z\|_{\infty}}\right),$$  where $s$ is a scaling factor. A random noise term $\eta\sim\mathcal{N}(0,\sigma^{2})$ is added to $\hat{z}$, i.e.,
$${\mathcal{L}}=\|{\bar{z}}-y\|_{2}^{2}.$$
$$(17)^{\frac{1}{2}}$$
$$(18)$$

$\mathbf{a}\cdot\mathbf{b}=\mathbf{a}\cdot\mathbf{b}$. 
22. (17)
The adversarial example is then computed using the gradient of $\mathcal{L}$ with respect to $x$: . 
$$x^{\prime}=x+\alpha\cdot\frac{\nabla_{x}{\cal L}}{\|\nabla_{x}{\cal L}\|_{\infty}}.\tag{1}$$
x
$$=x+\operatorname*{max}(-\epsilon,\operatorname*{min}(x^{\prime}-x,\epsilon)).$$
′ − *x, ϵ*)). (19)
$$\mathcal{L}=\frac{\|\bar{z}-y\|_{2}^{2}}{\|x-x_{k}^{\prime}\|_{\infty}},\qquad k>0$$  of the Hurwitz form $\mathcal{L}$
$$(20)$$
$${\mathcal{L}}_{\mathrm{mixup}}=\beta\sum_{k=1}^{5}{\mathcal{L}}_{\mathrm{CE}}\bigg(f_{\theta}\bigg({\frac{x}{2^{k}}}\bigg),y\bigg)-{\mathcal{L}}_{\mathrm{KL}}\tag{1}$$
$$(21)$$

## B.1.2 ℓ 1**-Fgsm Attack** B.1.3 Jitter Attack B.1.4 Mixup Attack

In this work we fix α = ϵ. The attack can be iterated to increase its strength. The Jitter attack [39] is an adversarial attack that perturbs an input by modifying the softmaxnormalized output of the model with random noise before computing the loss. Given a neural network classifier fθ with parameters θ, an input x, and its corresponding label y, the attack first computes the network output z = fθ(x) and normalizes it using the ℓ∞ norm:

$$(15)$$
$$(16)^{\frac{1}{2}}$$
$\text{non-perturbed input and}$. 
z˜ = ˆz + σ · η. (16)
The attack loss function is a mean squared error between perturbed input and target, given by To ensure the perturbation remains bounded, the modification x
′ − x is clamped within an ϵ bound:
In this work, we fix α = ϵ and set σ = 0.1. The Jitter attack can be performed iteratively. Then, for each but the first iteration k, the attack loss is normalized by the perturbation of the input image, In this work, we use 5 iterations of the Jitter attack for each image.

The Mixup attack [49] is an adversarial attack that generates adversarial samples that share similar feature representations with an given virtual example. Inspired by the Mixup data augmentation technique, this attack aims to create adversarial examples that maintain characteristics of both the original sample and its adversarial counterpart. Given a neural network classifier fθ with parameters θ, an input x, and its corresponding label y, the attack first computes a linear combination of cross-entropy and negative KL-divergence loss,

Table 8: Training hyperparameters for the UCM, Cifar10, and ImageNet Benchmarks. The first set hyperparameters apply to both DLRT and baseline training, and we train DLRT with the same hyperparameters as the full-rank baseline models. The second set of hyper-parameters is specific to DLRT. The DLRT hyperparameters are selected by an initial parameter sweep. We choose the DLRT

truncation tolerance relative to the Frobenius norm of Sb, i.e. ϑ = τ∥Sb∥F , as suggested in [38].

Hyperparameter VGG16 VGG11 ViT16b ViT32l Batch Size (UCM) 16 16 16 n.a. Batch Size (Cifar10) 128 128 128 n.a. Batch Size (ImageNet) n.a. n.a. n.a. 256 Learning Rate 0.001 0.001 0.001 0.001 Number of Epochs 20 20 5 10 L2 regularization 0 0 0.001 0.0001 Optimizer AdamW AdamW AdamW AdamW DLRT rel. truncation tolerance τ 0.1 0.05 0.08 0.013 Coefficient Steps s∗ 10 10 10 75 Initial Rank 150 150 150 200 Parameters 138M 132M 86M 304M

$$\delta=\alpha\cdot{\frac{\nabla_{x}{\mathcal{L}}_{\mathrm{CE}}(f_{\theta}(x),y)}{\|\nabla_{x}{\mathcal{L}}_{\mathrm{CE}}(f_{\theta}(x),y)\|_{\infty}}}.$$
$\phi$. 
. (22)
Equation (21) features a scale invariance attack applied to the loss [27, Section 3.3]. The final adversarial example is computed as a convex combination of the original input and its perturbed version:

$$x^{\prime}=\lambda x+(1-\lambda)(x+\delta),$$
$$(23)$$
$$(24)$$
′ = λx + (1 − λ)(x + δ), (23)
where λ ∼ Beta(*β, β*) is sampled from a Beta distribution with hyperparameter β, controlling the interpolation between clean and perturbed inputs. The perturbation is further constrained within an ϵ-ball to ensure bounded adversarial modifications:

$$x^{\prime}=x+\operatorname*{max}(-\epsilon,\operatorname*{min}(x^{\prime}-x,\epsilon)).$$
′ − *x, ϵ*)). (24)
In this work, we fix α = 1 and set β = 10−3. The attack can be iterated to increase its effectiveness, refining the adversarial perturbation at each step. We use 5 iterations of the Mixup Attack for each image.

## B.2 Network Architecture And Training Details

In this paper, we use the pytorch implementation and take pretrained weights from the imagenet1k dataset as initialization. The data-loaded randomly samples a batch for each batch-update which is the only source of randomness in our training setup. Below is an overview of the used network architectures - VGG16 is a deep convolutional neural network architecture that consists of 16 layers, including 13 convolutional layers and 3 fully connected layers.

- VGG11 is a convolutional neural network architecture similar to VGG16 but with fewer layers, consisting of 11 layers: 8 convolutional layers and 3 fully connected layers. It follows the same design principle as VGG16, using small 3×3 convolution filters and 2×2 max-pooling layers.

- ViT16b is a Vision Transformer with 16x16 patch size, a deep learning architecture that leverages transformer models for image classification tasks.

- ViT32l is a Vision Transformer with 32x32 patch size, a deep learning architecture that leverages transformer models for image classification tasks. We use the Imagenet21k weights from the huggingface endpoint google/vit-large-patch32-224-in21k as weight initialization.

The full training setup is described in Table 8. We train DLRT with the same hyperparameters as the full-rank baseline models. It is known [37] that DLRT methods are robust w.r.t. common

| Table 9: Overview of the β for best performing regularization strength for RobustDLRT of Table 2. UCM Dataset Cifar10 Dataset ImageNet Dataset Architecture FGSM Jitter Mixup FGSM Jitter Mixup FGSM Jitter Mixup VGG16 0.075 0.2 0.15 0.05 0.05 0.05 n.a. n.a. n.a. VGG11 0.1 0.05 0.15 0.15 0.05 0.2 n.a. n.a. n.a. ViT16b 0.1 0.15 0.15 0.01 0.01 0.05 n.a. n.a. n.a. ViT32l n.a. n.a. n.a. n.a. n.a. n.a. 0.075 0.075 0.075   |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

| iterations.   | Acc [%] under the ℓ 2 -FGSM attack with ϵ   |       |       |       |       |       |       |       |       |       |
|---------------|---------------------------------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| β             | 0                                           | 0.01  | 0.025 | 0.05  | 0.075 | 0.1   | 0.2   | 0.3   | 0.4   | 0.5   |
| 0             | 92.40                                       | 91.72 | 90.65 | 86.71 | 81.32 | 76.40 | 64.52 | 54.96 | 49.38 | 45.14 |
| 0.0001        | 91.69                                       | 91.69 | 91.10 | 87.73 | 83.14 | 78.43 | 63.21 | 53.31 | 47.18 | 42.99 |
| 0.001         | 88.81                                       | 88.78 | 87.90 | 84.40 | 80.00 | 76.34 | 62.61 | 53.77 | 48.09 | 44.38 |
| 0.01          | 88.22                                       | 88.19 | 87.12 | 82.78 | 77.52 | 72.72 | 58.32 | 48.89 | 42.83 | 38.61 |
| 0.05          | 90.45                                       | 90.43 | 89.63 | 87.23 | 84.11 | 80.55 | 68.66 | 59.29 | 52.62 | 46.61 |
| 0.1           | 92.51                                       | 92.51 | 92.11 | 90.45 | 88.43 | 86.32 | 76.91 | 68.01 | 61.29 | 55.52 |
| 0.2           | 89.20                                       | 89.18 | 88.85 | 86.66 | 84.36 | 81.96 | 73.25 | 65.20 | 58.61 | 53.29 |

hyperparameters as learning rate, and batch-size, and initial rank. The truncation tolerance τ is chosen between 0.05 and 0.1 per an initial parameter study. These values are good default values, as per recent literature [36, 42]. In general, there is a trade-off between target compression ratio and accuracy, as illustrated e.g. in [38] for matrix-valued and [42] for tensor-valued (CNN) layers.

## B.3 Ucm Test Case

The University of California, Merced (UCM) Land Use Dataset is a benchmark dataset in remote sensing and computer vision, introduced in [52]. It comprises 2,100 high-resolution aerial RGB images, each measuring 256×256 pixels, categorized into 21 land use classes with 100 images per class. The images were manually extracted from the USGS National Map Urban Area Imagery collection, covering various urban areas across the United States. The dataset contains images with spatial resolution approximately 0.3 meters per pixel (equivalent to 1 foot), providing detailed visual information suitable for fine-grained scene classification tasks. We normalize the training and validation data with mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225] for the rgb image channels. The convolutional neural neural networks used in this work are applied to the original 256 × 256 image size. The vision transformer data-pipeline resizes the image to a resolution of 224 × 224 pixels. The adversarial attacks for this dataset are performed on the resized images.

## B.4 Cifar10

The Cifar10 dataset consists of 10 classes, with a total of 60000 rgb images with a resolution of 32 × 32 pixels. We use standard data augmentation techniques. That is, for CIFAR10, we augment the training data set by a random horizontal flip of the image, followed by a normalization using mean
[0.4914, 0.4822, 0.4465] and std. dev. [0.2470, 0.2435, 0.2616]. The test data set is only normalized. The convolutional neural neural networks used in this work are applied to the original 32 × 32 image size. The vision transformer data-pipeline resizes the image to a resolution of 224 × 224 pixels. The adversarial attacks for this dataset are performed on the resized images.

Low-Rank VGG16, FGSM whitebox attack Full-Rank 92.40 91.72 90.65 86.71 81.32 76.40 64.52 54.96 49.38 45.14 0.0 93.92 93.86 93.00 87.95 80.14 72.41 52.12 43.39 39.07 35.81 0.001 94.03 93.94 93.04 88.70 81.51 74.10 53.70 43.94 38.04 34.26 Regularizati on ( )
0.01 93.64 94.36 93.83 90.64 85.20 78.76 54.55 43.48 37.30 33.68 0.025 93.77 93.90 93.40 90.03 84.71 78.93 59.74 49.56 43.31 38.89 0.05 93.01 92.78 92.25 88.83 84.23 78.92 61.16 51.46 44.67 40.03 0.075 92.61 92.83 92.27 89.12 84.10 78.68 62.83 53.30 47.46 43.08 0.0 0.01 0.025 0.05 0.075 0.1 0.2 0.3 0.4 0.5 c.r. [%]
Attack ( )
0.00 90 95.30 80 95.51 70 Accuracy (
%)
95.42 60 95.15 50 95.88 40 95.84 30

## B.5 Imagenet-1K

The ImageNet dataset consists of 1000 classes and over 1.2 million RGB training images, with a standard resolution of 224 × 224 pixels. We follow the standard data augmentation pipeline for ImageNet, which includes a random resized crop to 224 × 224, and normalization using mean [0.5, 0.5, 0.5] and standard deviation [0.5, 0.5, 0.5]. The test set is only resized and center-cropped to 224 × 224, followed by normalization. Adversarial attacks are generated on the normalized, resized images.

## B.6 Computational Hardware

All experiments in this paper are computed using workstation GPUs. Each training run used a single GPU. Specifically, we have used 5 NVIDIA RTX A6000, 3 NVIDIA RTX 4090, and 8 NVIDIA A-100 80G. The estimated time for one experimental run depends mainly on the data-set size and neural network architecture. For training, generation of adversarial examples and validation testing we estimate 30 minutes on one GPU for one run.

## C Proofs

To facilitate the proofs, we remark the definition of L-continuity: A function f(x) is Lipschitz continuous on a domain D if there exists a constant L ≥ 0 such that for all *x, y* ∈ D,
$$\|f(x)-f(y)\|\leq L\|x-y\|.$$

The smallest such L is called the Lipschitz constant.

| Low-Rank VGG16, Jitter whitebox attack   |       |       |       |       |       |          |
|------------------------------------------|-------|-------|-------|-------|-------|----------|
| Full-Rank                                | 94.42 | 93.41 | 90.80 | 89.58 | 87.83 | 85.05    |
| 0.0                                      | 93.19 | 92.15 | 87.92 | 83.99 | 79.31 | 67.41    |
| 0.01                                     | 93.97 | 93.10 | 90.55 | 88.46 | 85.68 | 78.49    |
| n ( ) Regularizatio0.05                  | 92.34 | 91.58 | 89.41 | 87.61 | 85.44 | 79.31    |
| 0.1                                      | 92.29 | 91.59 | 89.54 | 87.92 | 85.55 | 78.91    |
| 0.15                                     | 92.55 | 91.89 | 89.90 | 88.06 | 85.82 | 79.63    |
| 0.2                                      | 92.59 | 91.96 | 89.87 | 88.33 | 86.08 | 79.81    |
| 0.0                                      | 0.025 | 0.03  | 0.035 | 0.04  | 0.045 | c.r. [%] |
| Attack ( )                               |       |       |       |       |       |          |

| 0.00 95.30 95.51 95.42 95.15 95.88 95.84   |
|--------------------------------------------|

60 65 70 75 80 85 90 95 Accuracy (%)
0.2
For the following proofs, let

$$(A,B)=\operatorname{trace}(B^{\top}A)=\sum_{i j}A_{i j}B_{i j}$$

be the Frobenius inner product that induces the norm ∥A∥ =p(*A, A*). By the cyclic property of the
trace, we have
$$(A B,C D)=(B,C D A^{\top})=(C^{\top}A B,D).$$ $D$ of appropriate size. 
⊤*AB, D*). (25)
for matrices A, B, C, and D of appropriate size.

Proof of (3). We calculate

$$\mathcal{R}(S)^{2}=(S^{\top}S-\alpha_{S}^{2}I,S^{\top}S-\alpha_{S}^{2}I)$$ $$=\|S^{\top}S\|^{2}-2\alpha_{S}^{2}(S^{\top}S,I)+\alpha_{S}^{4}(I,I)$$ $$=\|S^{\top}S\|^{2}-\frac{1}{r}\|S\|^{4}$$ $$=\sum_{i=1}^{r}\varsigma_{i}(S^{\top}S)^{2}-\frac{1}{r}\bigg{(}\sum_{i=1}^{r}\varsigma_{i}(S)^{2}\bigg{)}^{2}$$ $$=r\bigg{(}\frac{1}{r}\sum_{i=1}^{r}\varsigma_{i}(S^{\top}S)^{2}-\bigg{(}\frac{1}{r}\sum_{i=1}^{r}\varsigma_{i}(S)^{2}\bigg{)}^{2}\bigg{)}$$  $\bullet$\(\bullet

$$(25)$$

$$(26)$$

Since S
⊤S is symmetric positive semi-definite, ςi(S
⊤S) = ςi(S)
2. Applying this substitution yields
(3). The proof is complete.