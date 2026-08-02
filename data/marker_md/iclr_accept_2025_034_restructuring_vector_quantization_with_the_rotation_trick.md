# RESTRUCTURING VECTOR QUANTIZATION WITH THE ROTATION TRICK

Christopher Fifty<sup>1</sup> , Ronald G. Junkins<sup>1</sup> , Dennis Duan<sup>1</sup>,<sup>2</sup> , Aniketh Iyengar<sup>1</sup> , Jerry W. Liu<sup>1</sup> , Ehsan Amid<sup>2</sup> , Sebastian Thrun<sup>1</sup> , Christopher Ré<sup>1</sup>

<sup>1</sup>Stanford University, <sup>2</sup>Google DeepMind

fifty@cs.stanford.com

#### ABSTRACT

Vector Quantized Variational AutoEncoders (VQ-VAEs) are designed to compress a continuous input to a discrete latent space and reconstruct it with minimal distortion. They operate by maintaining a set of vectors—often referred to as the codebook—and quantizing each encoder output to the nearest vector in the codebook. However, as vector quantization is non-differentiable, the gradient to the encoder flows *around* the vector quantization layer rather than *through* it in a straight-through approximation. This approximation may be undesirable as all information from the vector quantization operation is lost. In this work, we propose a way to propagate gradients through the vector quantization layer of VQ-VAEs. We smoothly transform each encoder output into its corresponding codebook vector via a rotation and rescaling linear transformation that is treated as a constant during backpropagation. As a result, the relative magnitude and angle between encoder output and codebook vector becomes encoded into the gradient as it propagates through the vector quantization layer and back to the encoder. Across 11 different VQ-VAE training paradigms, we find this restructuring improves reconstruction metrics, codebook utilization, and quantization error. Our code is available at [https://github.com/cfifty/rotation\\_trick.](https://github.com/cfifty/rotation_trick)

### 1 INTRODUCTION

Vector quantization [\(Gray,](#page-11-0) [1984\)](#page-11-0) is an approach to discretize a continuous vector space. It defines a finite set of vectors—referred to as the codebook—and maps any vector in the continuous vector space to the closest vector in the codebook. However, deep learning paradigms that use vector quantization are often difficult to train because replacing a vector with its closest codebook counterpart is a nondifferentiable operation [\(Huh et al.,](#page-11-1) [2023\)](#page-11-1). This characteristic was not an issue at its creation during the Renaissance of Information Theory for applications like noisy channel communication [\(Cover,](#page-10-0) [1999\)](#page-10-0); however in the era deep learning, it presents a challenge as gradients cannot directly flow through layers that use vector quantization during backpropagation.

In deep learning, vector quantization is largely used in the eponymous Vector Quantized-Variational AutoEncoder (VQ-VAE) [\(Van Den Oord et al.,](#page-12-0) [2017\)](#page-12-0). A VQ-VAE is an AutoEncoder with a vector quantization layer between the encoder's output and decoder's input, thereby quantizing the learned representation at the bottleneck. While VQ-VAEs are ubiquitous in state-of-the-art generative modeling [\(Rombach et al.,](#page-12-1) [2022;](#page-12-1) [Dhariwal et al.,](#page-11-2) [2020;](#page-11-2) [Brooks et al.,](#page-10-1) [2024\)](#page-10-1), their gradients cannot flow from the decoder to the encoder uninterrupted as they must pass through a non-differentiable vector quantization layer.

A solution to the non-differentiability problem is to approximate gradients via a "straight-through estimator" (STE) [\(Bengio et al.,](#page-10-2) [2013\)](#page-10-2). During backpropagation, the STE copies and pastes the gradients from the decoder's input to the encoder's output, thereby skipping the quantization operation altogether. However, this approximation can lead to poor-performing models and codebook collapse: a phenomena where a large percentage of the codebook converge to zero norm and are unused by the model [\(Mentzer et al.,](#page-12-2) [2023\)](#page-12-2). Even if codebook collapse does not occur, the codebook is often under-utilized, thereby limiting the information capacity of the VQ-VAEs's bottleneck [\(Dhariwal](#page-11-2) [et al.,](#page-11-2) [2020\)](#page-11-2).

![](_page_1_Diagram_1.jpeg)

Figure 1: Illustration of the rotation trick. In the forward pass, encoder output e is rotated and rescaled to q1. For simplicity, the rotations of other encoder outputs are not shown. In the backward pass, the gradient at q<sup>1</sup> moves to e so that the angle between ∇<sup>q</sup>1L and q<sup>1</sup> is preserved. Now, points within the same codebook region receive different gradients depending on their relative angle and magnitude to the codebook vector. For example, points with high angular distance can be *pushed* into new codebook regions, thereby increasing codebook utilization.

Algorithm 1 The Rotation Trick Require: input example x e ← Encoder(x) q ← nearest codebook vector to e R ← rotation matrix that aligns e to q <sup>q</sup>˜ <sup>←</sup> stop-gradient h ∥q∥ <sup>∥</sup>e∥R e x˜ ← Decoder(q˜) loss ← L(x, x˜) return loss In this work, we propose an alternate way to propagate gradients through the vector quantization layer in VQ-VAEs. For a given encoder output e and nearest codebook vector q, we smoothly transform e to q via a rotation and rescaling linear transformation and then send this output rather than the direct result of the codebook lookup—to the decoder. As the input to the decoder, q˜, is now treated as a smooth linear transformation of e, gradients flow back from the decoder to the encoder unimpeded. To avoid differentiating through the rotation and rescaling, we treat both as constants with respect to e and q. We explain why this choice is necessary in Appendix [A.7.](#page-19-0) Following the convention of [Kingma & Welling](#page-12-3) [\(2013\)](#page-12-3), we call this restructuring "the rotation trick." It is illustrated in Figure [1](#page-1-0) and described in Algorithm [1.](#page-1-1)

The rotation trick does not change the output of the VQ-VAE in the forward pass. However, during the backward pass, it transports the gradient ∇qL at q to become the gradient ∇eL at e so that the angle between q and ∇qL *after* the vector quantization layer equals the angle between e and ∇eL *before* the vector quantization layer. Preserving this angle encodes relative angular distances and magnitudes into the gradient and changes how points within the same codebook region are updated.

The STE applies the same update to all points within the same codebook region, maintaining their relative distances. However as we will show in Section [4.3,](#page-5-0) the rotation trick can push points within the same codebook region farther apart—or pull them closer together—depending on the direction of the gradient vector. The former capability can correspond to increased codebook usage while the latter to lower quantization error. In the context of lossy compression, both capabilities are desirable for reducing the distortion and increasing the information capacity of the vector quantization layer.

When applied to several open-source VQ-VAE repositories, we find the rotation trick substantively improves reconstruction performance, increases codebook usage, and decreases the distance between encoder outputs and their corresponding codebook vectors. For instance, training the VQGAN from [Rombach et al.](#page-12-1) [\(2022\)](#page-12-1) on ImageNet [\(Deng et al.,](#page-10-3) [2009\)](#page-10-3) with the rotation trick improves reconstruction FID from 5.0 to 1.1, reconstruction IS from 141.5 to 200.2, increases codebook usage from 2% to 27%, and decreases quantization error by two orders of magnitude.

## 2 RELATED WORK

Many researchers have built upon the seminal work of [Van Den Oord et al.](#page-12-0) [\(2017\)](#page-12-0) to improve VQ-VAE performance. While non-exhaustive, our review focuses on methods that address training instabilities caused by the vector quantization layer. We partition these efforts into two categories: (1) methods that sidestep the STE and (2) methods that improve codebook-model interactions.

Sidestepping the STE. Several prior works have sought to fix the problems caused by the STE by avoiding deterministic vector quantization. [Baevski et al.](#page-10-4) [\(2019\)](#page-10-4) employ the Gumbel-Softmax trick [\(Jang et al.,](#page-11-3) [2016\)](#page-11-3) to fit a categorical distribution over codebook vectors that converges to a one-hot distribution towards the end of training, [Gautam et al.](#page-11-4) [\(2023\)](#page-11-4) quantize using a convex combination of codebook vectors, and [Takida et al.](#page-12-4) [\(2022\)](#page-12-4) employ stochastic quantization. Unlike the above that cast vector quantization as a distribution over codebook vectors, [Huh et al.](#page-11-1) [\(2023\)](#page-11-1) propose an alternating optimization where the encoder is optimized to output representations close to the codebook vectors while the decoder minimizes reconstruction loss from a fixed set of codebook vector inputs. While these approaches sidestep the training instabilities caused by the STE, they can introduce their own set of problems and complexities such as low codebook utilization at inference and the tuning of a temperature schedule [\(Zhang et al.,](#page-12-5) [2023\)](#page-12-5). As a result, many applications and research papers continue to employ VQ-VAEs that are trained using the STE [\(Rombach et al.,](#page-12-1) [2022;](#page-12-1) [Chang et al.,](#page-10-5) [2022;](#page-10-5) [Huang et al.,](#page-11-5) [2023;](#page-11-5) [Zhu et al.,](#page-13-0) [2023;](#page-13-0) [Dong et al.,](#page-11-6) [2023\)](#page-11-6).

Codebook-Model Improvements. Another way to attack codebook collapse or under-utilization is to change the codebook lookup. Rather than use Euclidean distance, [Yu et al.](#page-12-6) [\(2021\)](#page-12-6) employ a cosine similarity measure, [Goswami et al.](#page-11-7) [\(2024\)](#page-11-7) a hyperbolic metric, and [Lee et al.](#page-12-7) [\(2022\)](#page-12-7) stochastically sample codes as a function of the distance between the encoder output and codebook vectors. Another perspective examines the learning of the codebook. [Kolesnikov et al.](#page-12-8) [\(2022\)](#page-12-8) split high-usage codebook vectors, [Dhariwal et al.](#page-11-2) [\(2020\)](#page-11-2); [Łancucki et al.](#page-12-9) ´ [\(2020\)](#page-12-9); [Zheng & Vedaldi](#page-12-10) [\(2023\)](#page-12-10) resurrect low-usage codebook vectors throughout training, [Chen et al.](#page-10-6) [\(2024\)](#page-10-6) dynamically selects one of m codebooks for each datapoint, and [Mentzer et al.](#page-12-2) [\(2023\)](#page-12-2); [Zhao et al.](#page-12-11) [\(2024\)](#page-12-11); [Yu et al.](#page-12-12) [\(2023\)](#page-12-12); [Chiu et al.](#page-10-7) [\(2022\)](#page-10-7) fix the codebook vectors to an *a priori* geometry and train the model without learning the codebook at all. Other works propose loss penalties to encourage codebook utilization. [Zhang et al.](#page-12-5) [\(2023\)](#page-12-5) add a KL-divergence penalty between codebook utilization and a uniform distribution while [Yu et al.](#page-12-12) [\(2023\)](#page-12-12) add an entropy loss term to penalize low codebook utilization. While effective at targeting specific training difficulties, as each of these methods continue to use the STE, the training instability caused by this estimator persist. Most of our experiments in Section [5](#page-7-0) implement a subset of these approaches, and we find that replacing the STE with the rotation trick further improves performance.

#### 3 STRAIGHT THROUGH ESTIMATOR (STE)

In this section, we review the Straight-Through Estimator (STE) and visualize its effect on the gradients. We then explore two STE alternatives that—at first glance—appear to correct the approximation made by the STE.

For notation, we define a sample space X over the input data with probability distribution p. For input x ∈ X , we define the encoder as a deterministic mapping that parameterizes a posterior distribution p<sup>E</sup> (e|x). The vector quantization layer, Q(·), is a function that selects the codebook vector q ∈ C nearest to the encoder output e. Under Euclidean distance, it has the form:

$$\mathcal{Q}(q = i|e) = \begin{cases} 1 & \text{if } i = \arg \min_{1 \leq j \leq |C|} \|e - q_j\|_2 \\ 0 & \text{otherwise} \end{cases}$$

The decoder is similarly defined as a deterministic mapping that parameterizes the conditional distribution over reconstructions pD(˜x|q). As in the VAE [\(Kingma & Welling,](#page-12-3) [2013\)](#page-12-3), the loss function follows from the ELBO with the KL-divergence term zeroing out as p<sup>E</sup> (e|x) is deterministic and the utilization over codebook vectors is assumed to be uniform. [Van Den Oord et al.](#page-12-0) [\(2017\)](#page-12-0) additionally add a "codebook loss" term ∥sg(e) − q∥ 2 2 to learn the codebook vectors and a "commitment loss" term β∥e − sg(q)∥ 2 to pull the encoder's output towards the codebook vectors. sg stands for stopgradient and β is a hyperparameter, typically set to a value in [0.25, 2]. For predicted reconstruction x˜, the optimization objective becomes:

$$\mathcal{L}(\tilde{x}) = \|x - \tilde{x}\|_2^2 + \|sg(e) - q\|_2^2 + \beta\|e - sg(q)\|_2^2$$

In the subsequent analysis, we focus only on the ∥x − x˜∥ 2 term as the other two are not functions of the decoder. During backpropagation, the model must differentiate through the vector quantization

![](_page_3_Figure_1.jpeg)

Figure 2: Visualization of how the straight-through estimator (STE) transforms the gradient field for 16 codebook vectors for (top) f(x, y) = x <sup>2</sup> + y 2 and (bottom) f(x, y) = log 1 2 x + tanh(y)| . The STE takes the gradient at the codebook vector (qx, qy) and "copies-and-pastes" it to all other locations within the same codebook region, forming a "checker-board" pattern in the gradient field.

function Q(·). We can break down the backward pass into three terms:

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial q} \frac{\partial q}{\partial e} \frac{\partial e}{\partial x}$$

where <sup>∂</sup><sup>L</sup> ∂q represents backpropagation through the decoder, ∂q ∂e represents backpropagation through the vector quantization layer, and ∂e ∂x represents backpropagation through the encoder. As vector quantization is not a smooth transformation, ∂q ∂e cannot be computed and gradients cannot flow through this term to update the encoder in backpropagation.

To solve the issue of non-differentiability, the STE copies the gradients from q to e, bypassing vector quantization entirely. Simply, the STE sets ∂q ∂e to the identity matrix I in the backward pass:

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial q} I \frac{\partial e}{\partial x}$$

The first two terms <sup>∂</sup><sup>L</sup> ∂q ∂q ∂e combine to <sup>∂</sup><sup>L</sup> ∂e which, somewhat misleadingly, does not actually depend on e. As a consequence, the location of e within the Voronoi partition generated by codebook vector q—be it close to q or at the boundary of the region—has no impact on the gradient update to the encoder.

An example of this effect is visualized in Figure [2](#page-3-0) for two example functions. In the STE approximation, the "exact" gradient at the encoder output is replaced by the gradient at the corresponding codebook vector for each Voronoi partition, irrespective of where in that region the encoder output e lies. As a result, the exact gradient field becomes "partitioned" into 16 different regions—all with the same gradient update to the encoder—for the 16 vectors in the codebook.

Returning to our question, is there a better way to propagate gradients through the vector quantization layer? At first glance, one may be tempted to estimate the curvature at q and use this information to transform ∂q ∂e as q moves to e. This is accomplished by taking a second order expansion around q to approximate the value of the loss at e:

$$\mathcal{L}_e \approx \mathcal{L}_q + (\nabla_q \mathcal{L})^T (e - q) + \frac{1}{2} (e - q)^T (\nabla_q^2 \mathcal{L}) (e - q)$$

Then we can compute the gradient at the point e instead of q up to second order approximation with:

$$\begin{aligned} \frac{\partial \mathcal{L}}{\partial e} &\approx \frac{\partial}{\partial e} \left[ \mathcal{L}_q + (\nabla_q \mathcal{L})^T (e - q) + \frac{1}{2} (e - q)^T (\nabla_q^2 \mathcal{L}) (e - q) \right] \\ &= \nabla_q \mathcal{L} + (\nabla_q^2 \mathcal{L}) (e - q) \end{aligned}$$

While computing Hessians with respect to model parameters are typically prohibitive in modern deep learning architectures, computing them with respect to only the codebook is feasible. Moreover as we must only compute (∇<sup>2</sup> <sup>q</sup>L)(e − q), one may take advantage of efficient Hessian-Vector products implementations in deep learning frameworks [\(Dagréou et al.,](#page-10-8) [2024\)](#page-10-8) and avoid computing the full Hessian matrix.

Extending this idea a step further, we can compute the exact gradient <sup>∂</sup><sup>L</sup> ∂e at e by making two passes through the network. Let L<sup>q</sup> be the loss with the vector quantization layer and L<sup>e</sup> be the loss without vector quantization, i.e. q = e rather than q = Q(e). Then one may form the total loss L = L<sup>q</sup> +λLe, where λ is a small constant like 10−<sup>6</sup> , to scale down the effect of L<sup>e</sup> on the decoder's parameters and use a gradient scaling multiplier of λ −1 to reweigh the effect of L<sup>e</sup> on the encoder's parameters to 1. As ∂q ∂e is non-differentiable, gradients from L<sup>q</sup> will not flow to the encoder.

While seeming to correct the encoder's gradients, replacing the STE with either approach will likely result in worse performance. This is because computing the exact gradient with respect to e is actually the AutoEncoder [\(Hinton & Zemel,](#page-11-8) [1993\)](#page-11-8) gradient, the model that VAEs [\(Kingma](#page-12-3) [& Welling,](#page-12-3) [2013\)](#page-12-3) and VQ-VAEs [\(Van Den Oord et al.,](#page-12-0) [2017\)](#page-12-0) were designed to replace given the AutoEncoder's propensity to overfit and difficultly generalizing. Accordingly using either Hessian approximation or exact gradients via a double forward pass will cause the encoder to be trained like an AutoEncoder and the decoder to be trained like a VQ-VAE. This mis-match in optimization objectives is likely another contributing factor to the poor performance we observe for both methods in Table [1,](#page-7-1) and a deeper analysis into these characteristics is presented in Appendix [A.3.](#page-14-0)

#### 4 THE ROTATION TRICK

As discussed in Section [3,](#page-2-0) updating the encoder's parameters by approximating, or exactly, computing the gradient at the encoder's output is undesirable. Similarly, the STE appears to lose information: the location of e within the quantized region—be it close to q or far away at the boundary—has no impact on the gradient update to the encoder. Capturing this information, i.e. using the location of e in relation to q to transform the gradients through ∂q ∂e , could be beneficial to the encoder's gradient updates and an improvement over the STE.

Viewed geometrically, we ask how to move the gradient ∇qL from q to e, and what characteristics of ∇qL and q should be preserved during this movement. The STE offers one possible answer: move the gradient from q to e so that its direction and magnitude are preserved. However, this paper supplies a different answer: move the gradient so that the angle between ∇qL and q is preserved as ∇qL moves to e. We term this approach "the rotation trick", and in Section [4.3](#page-5-0) we show that preserving the angle between q and ∇qL conveys desirable properties to how points move within the same quantized region.

#### 4.1 THE ROTATION TRICK PRESERVES ANGLES

In this section, we formally define the rotation trick. For encoder output e, let q = Q(e) represent the corresponding codebook vector. Q(·) is non-differentiable so gradients cannot flow through this layer during the backward pass. The STE solves this problem—maintaining the direction and magnitude of the gradient ∇qL—as ∇qL moves from q to e with some clever hacking of the backpropagation function in deep learning frameworks:

$$\tilde{q} = e - \underbrace{(q - e)}_{\text{constant}}$$

which is a parameterization of vector quantization that sets the gradient at the encoder output to the gradient at the decoder's input. The rotation trick offers a different parameterization: casting the forward pass as a rotation and rescaling that aligns e with q:

![](_page_5_Figure_1.jpeg)

Figure 3: Illustration of how the gradient at q moves to e via the STE (middle) and rotation trick (right). The STE "copies-and-pastes" the gradient to preserve its direction while the rotation trick moves the gradient so the angle between q and

∇qL is preserved (proved in Appendix [A.6\)](#page-18-0). As a result, ∂q ∂e changes based on the position of e in the codebook partition of q, and notably, the angle between ∇qL and q is preserved as ∇qL moves to e. This effect is visualized in Figure [3.](#page-5-2) While the STE translates the gradient from q to e, the rotation trick rotates it so that the angle between ∇qL and q is preserved. In a sense, the rotation trick and the STE are sibilings. They choose different characteristics of the gradient as desiderata and then preserve those characteristics as the gradient flows around the non-differentiable vector quantization operation to the encoder.

$$\tilde{q} = \underbrace{\left[ \begin{array}{c|c} \|q\| & R \\ \hline \|e\| & \end{array} \right]}_{\text{constant}} e$$

R is the rotation[<sup>1</sup>](#page-5-1) transformation that aligns e with q and <sup>∥</sup>q<sup>∥</sup> ∥e∥ rescales e to have the same magnitude as q. Note that both R and <sup>∥</sup>q<sup>∥</sup> ∥e∥ are functions of e. To avoid differentiating through this dependency, we treat them as fixed constants—or detached from the computational graph in deep learning frameworks when differentiating. This choice is explained in Appendix [A.7.](#page-19-0)

While the rotation trick does not change the output of the forward pass, the backward pass changes. Rather than set ∂q ∂e = I as in the STE, the rotation trick sets ∂q ∂e to be a rotation and rescaling transformation:

$$\frac{\partial \tilde{q}}{\partial e} = \frac{\|q\|}{\|e\|} R$$

#### 4.2 EFFICIENT ROTATION COMPUTATION

The rotation transformation R that rotates e to q can be efficiently computed with Householder matrix reflections. We define eˆ = e ∥e∥ , qˆ = q ∥q∥ , λ = ∥q∥ ∥e∥ , and r = eˆ+ˆq ∥eˆ+ˆq∥ . Then the rotation and rescaling that aligns e to q is simply:

$$\begin{aligned}\tilde{q} &= \lambda Re \\ &= \lambda(I - 2rr^T + 2\hat{q}\hat{e}^T)e \\ &= \lambda[e - 2rr^T e + 2\hat{q}\hat{e}^T e]\end{aligned}$$

Due to space constraints, we leave the derivation of this formula to Appendix [A.5.](#page-18-1) Parameterizing the rotation in this fashion avoids computing outer products and therefore consumes minimal GPU VRAM. Further, we did not detect a difference in wall-clock time between VQ-VAEs trained with the STE and VQ-VAEs trained with the rotation trick for our experiments in Section [5.](#page-7-0)

#### 4.3 VORONOI PARTITION ANALYSIS

In the context of lossy compression, vector quantization works well when the distortion, or equivalently quantization error ∥e − q∥ 2 2 , is low and the information capacity—equivalently codebook utilization—is high [\(Cover,](#page-10-0) [1999\)](#page-10-0). Later in Section [5,](#page-7-0) we will see that VQ-VAEs trained with the rotation trick have this *desiderata*—often reducing quantization error by an order of magnitude and substantially increasing codebook usage—when compared to VQ-VAEs trained with the STE. However, the underlying reason *why* this occurs is less clear.

<sup>1</sup>A rotation is defined as a linear transformation so that RR<sup>T</sup> = I, R <sup>−</sup><sup>1</sup> = R T , and det(R) = 1.

![](_page_6_Figure_1.jpeg)

Figure 4: Depiction of how points within the same codebook region change after a gradient update (red arrow) at the codebook vector (orange circle). The STE applies the same update to each point in the same region. The rotation trick modifies the update based on the location of each point with respect to the codebook vector.

Change in Distance Between

![](_page_6_Figure_5.jpeg)

Figure 5: With the STE, the distances among points within the same region do not change. However with the rotation trick, the distances among points *do* change. When ϕ < π/2, points with large angular distance are pushed away (blue: increasing distance). When ϕ > π/2, points are *pulled* towards the codebook vector (green: decreasing distance).

In this section, we analyze the effect of the rotation trick by looking at how encoder outputs that are mapped to the same Voronoi region are updated. While the STE applies the same update to all points within the same partition, the rotation trick changes the update based on the location of points within the Voronoi region. It can push points within the same region farther apart or pull them closer together depending on the direction of the gradient vector. The former capability can correspond to increased codebook usage while the latter to lower quantization error.

Let θ be the angle between e and q and ϕ be the angle between q and ∇qL. When ∇qL and q point in the same direction, i.e. −π/2 < ϕ < π/2, encoder outputs with large angular distance to q are pushed *farther* away than they would otherwise be moved by the STE update. Figure [5](#page-6-0) illustrates this effect. The points with large angular distance (blue regions) move further away from q than the points with low angular distance (ivory regions).

The top right partitions of Figure [4](#page-6-1) present an example of this effect. The two clusters of points at the boundary—with relatively large angle to the codebook vector—are pushed away while the cluster of points with small angle to the codebook vector move with it. The ability to push points at the boundary out of a quantized region and into another is desirable for increasing codebook utilization. Specifically, codebook utilization improves when points are pushed into the Voronoi regions of previously unused codebook vectors. This capability is not shared by the STE, which moves all points in the same region by the same amount.

When ∇qL and q point in opposite directions, i.e. π/2 < ϕ < 3π/2, the distance among points within the same Voronoi region decreases as they are pulled towards the location of the updated codebook vector. This effect is visualized in Figure [5](#page-6-0) (green regions) and the bottom partitions of Figure [4](#page-6-1) show an example. Unlike the STE update—that maintains the distances among points—the rotation trick pulls points with high angular distances closer towards the post-update codebook vector. This capability is desirable for reducing the quantization error and enabling the encoder to *lock on* [\(Van Den Oord et al.,](#page-12-0) [2017\)](#page-12-0) to a target codebook vector.

Taken together, both capabilities can form a push-pull effect that achieves two *desiderata* of vector quantization: increasing information capacity and reducing distortion. Encoder outputs that have large

Table 1: Comparison of VQ-VAEs trained on ImageNet following [Van Den Oord et al.](#page-12-0) [\(2017\)](#page-12-0). We use the Vector Quantization layer from [https://github.com/lucidrains/vector-quantize-pytorch.](https://github.com/lucidrains/vector-quantize-pytorch)

| Approach | Codebook | Lookup:  | Euclidean | Codebook Usage & Latent Shape: 32 | Training Metrics ( ↑ ) Rec. Loss ( ↓ ) × 32 × 32 & Codebook | Quantization Error Size: 1024 | ( ↓ ) Rec. Loss ( | Validation Metrics ↓ ) r-FID ( ↓ | ) r-IS ( ↑ ) |
|----------|----------|----------|-----------|-----------------------------------|-------------------------------------------------------------|-------------------------------|-------------------|----------------------------------|--------------|
| VQ-VAE   |          |          |           | 100%                              | 0.107                                                       | 5.9e-3                        | 0.115             | 106.1                            | 11.7         |
| VQ-VAE   | w/       | Rotation | Trick     | 97%                               | 0.116                                                       | 5.1e-4                        | 0.122             | 85.7                             | 17.0         |
|          | Codebook | Lookup:  | Cosine &  | Latent Shape: 32 ×                | 32 × 32 & Codebook                                          | Size: 1024                    |                   |                                  |              |
| VQ-VAE   |          |          |           | 75%                               | 0.107                                                       | 2.9e-3                        | 0.114             | 84.3                             | 17.7         |
| VQ-VAE   | w/       | Rotation | Trick     | 91%                               | 0.105                                                       | 2.7e-3                        | 0.111             | 82.9                             | 18.1         |
|          | Codebook | Lookup:  | Euclidean | & Latent Shape: 64                | × 64 × 3 & Codebook                                         | Size: 8192                    |                   |                                  |              |
| VQ-VAE   |          |          |           | 100%                              | 0.028                                                       | 1.0e-3                        | 0.030             | 19.0                             | 97.3         |
| Gumbel   |          | VQ-VAE   |           | 39%                               | 0.054                                                       | —                             | 0.058             | 28.6                             | 74.9         |
| VQ-VAE   | w/       | Hessian  | Approx.   | 39%                               | 0.082                                                       | 6.9e-5                        | 0.112             | 35.6                             | 65.1         |
| VQ-VAE   | w/       | Exact    | Gradients | 84%                               | 0.050                                                       | 2.0e-3                        | 0.053             | 25.4                             | 80.4         |
| VQ-VAE   | w/       | Rotation | Trick     | 99%                               | 0.028                                                       | 1.4e-4                        | 0.030             | 16.5                             | 106.3        |
|          | Codebook | Lookup:  | Cosine &  | Latent Shape: 64 ×                | 64 × 3 & Codebook                                           | Size: 8192                    |                   |                                  |              |
| VQ-VAE   |          |          |           | 31%                               | 0.034                                                       | 1.2e-4                        | 0.038             | 26.0                             | 77.8         |
| VQ-VAE   | w/       | Hessian  | Approx.   | 37%                               | 0.035                                                       | 3.8e-5                        | 0.037             | 29.0                             | 71.5         |
| VQ-VAE   | w/       | Exact    | Gradients | 38%                               | 0.035                                                       | 3.6e-5                        | 0.037             | 28.2                             | 75.0         |
| VQ-VAE   | w/       | Rotation | Trick     | 38%                               | 0.033                                                       | 9.6e-5                        | 0.035             | 24.2                             | 83.9         |

angular distance to the chosen codebook vector are "pushed" to other, possibly unused, codebook regions by outwards-pointing gradients, thereby increasing codebook utilization. Concurrent with this effect, center-pointing gradients will "pull" points loosely clustered around the codebook vector closer together, locking on to the chosen codebook vector and reducing quantization error.

#### 4.4 FURTHER ANALYSIS

The Appendix contains several supplementary analyses. Appendix [A.2](#page-14-1) compares the rotation trick with the STE for a non-convex synthetic example; Appendix [A.4](#page-16-0) looks at the behavior far away from the origin; and Appendix [A.8](#page-20-0) analyzes the effect of using a reflection rather than a rotation. Finally, Appendix [A.9](#page-21-0) examines scaling the gradient's norm by <sup>∥</sup>q<sup>∥</sup> ∥e∥ and explores alternatives.

#### 5 EXPERIMENTS

In Section [4.3,](#page-5-0) we showed the rotation trick enables behavior that would increase codebook utilization and reduce quantization error by changing how points within the same Voronoi region are updated. However, the extent to which these changes will affect applications is unclear. In this section, we evaluate the effect of the rotation trick across many different VQ-VAE paradigms.

We begin with image reconstruction: training a VQ-VAE with the reconstruction objective of [Van](#page-12-0) [Den Oord et al.](#page-12-0) [\(2017\)](#page-12-0) and later extend our evaluation to the more complex VQGANs [\(Esser](#page-11-9) [et al.,](#page-11-9) [2021\)](#page-11-9), the VQGANs designed for latent diffusion [\(Rombach et al.,](#page-12-1) [2022\)](#page-12-1), and then the ViT-VQGAN [\(Yu et al.,](#page-12-6) [2021\)](#page-12-6). Finally, we evaluate VQ-VAE reconstructions on videos using a TimeSformer [\(Bertasius et al.,](#page-10-9) [2021\)](#page-10-9) encoder and decoder. Due to space constraints, the video results are presented in Appendix [A.1.](#page-14-2) In total, our empirical analysis spans 11 different VQ-VAE configurations. For all experiments, aside from handling ∂q ∂e differently, the models, hyperparameters, and training settings are identical and described in Appendix [A.10.](#page-22-0)

#### 5.1 VQ-VAE EVALUATION

We begin with a straightforward evaluation: training a VQ-VAE to reconstruct examples from ImageNet [\(Deng et al.,](#page-10-3) [2009\)](#page-10-3). Following [Van Den Oord et al.](#page-12-0) [\(2017\)](#page-12-0), our training objective is a linear combination of the reconstruction, codebook, and commitment loss:

$$\mathcal{L} = \|x - \tilde{x}\|_2^2 + \|sg(e) - q\|_2^2 + \beta\|e - sg(q)\|_2^2$$

where β is a hyperparameter scaling constant. Following convention, we drop the codebook loss term from the objective and instead use an exponential moving average to update the codebook vectors.

Evaluation Settings. For 256 × 256 × 3 input images, we evaluate two different settings: (1) compressing to a latent space of dimension 32 × 32 × 32 with a codebook size of 1024 following [Yu](#page-12-6) [et al.](#page-12-6) [\(2021\)](#page-12-6) and (2) compressing to 64 × 64 × 3 with a codebook size of 8192 following [Rombach](#page-12-1) [et al.](#page-12-1) [\(2022\)](#page-12-1). In both settings, we compare with a Euclidean and cosine similarity codebook lookup.

Table 2: Results for VQGAN designed for autoregressive generation as implemented in [https://github.com/](https://github.com/CompVis/taming-transformers) [CompVis/taming-transformers.](https://github.com/CompVis/taming-transformers) Experiments on ImageNet and the combined dataset FFHQ [\(Karras et al.,](#page-11-10) [2019\)](#page-11-10) and CelebA-HQ [\(Karras,](#page-11-11) [2017\)](#page-11-11) use a latent bottleneck of dimension 16×16×256 with 1024 codebook vectors.

| Approach VQGAN (reported) | Dataset ImageNet | Codebook Usage — | Quantization Error ( ↓ ) — | Valid Loss ( ↓ ) — | r-FID ( ↓ ) 7.9 | r-IS ( ↑ ) 114.4 |
|---------------------------|------------------|------------------|----------------------------|--------------------|-----------------|------------------|
| VQGAN (our run)           | ImageNet         | 95%              | 0.134                      | 0.594              | 7.3             | 118.2            |
| VQGAN w/ Rotation Trick   | ImageNet         | 98%              | 0.002                      | 0.422              | 4.6             | 146.5            |
| VQGAN                     | FFHQ & CelebA-HQ | 27%              | 0.233                      | 0.565              | 4.7             | 5.0              |
| VQGAN w/ Rotation Trick   | FFHQ & CelebA-HQ | 99%              | 0.002                      | 0.313              | 3.7             | 5.2              |

Table 3: Results for VQGAN designed for latent diffusion as implemented in [https://github.com/CompVis/](https://github.com/CompVis/latent-diffusion) [latent-diffusion.](https://github.com/CompVis/latent-diffusion) Both settings train on ImageNet.

| latent-diffusion. Both Approach | settings train Latent Shape | on ImageNet. Codebook Size | Codebook Usage | Quantization Error ( ↓ | ) Valid Loss ( ↓ ) | r-FID ( ↓ ) | r-IS ( ↑ ) |
|---------------------------------|-----------------------------|----------------------------|----------------|------------------------|--------------------|-------------|------------|
| VQGAN                           | 64 × 64 × 3                 | 8192                       | 15%            | 2.5e-3                 | 0.183              | 0.53        | 220.6      |
| Gumbel VQGAN                    | 64 × 64 × 3                 | 8192                       | 4%             | —                      | 0.197              | 0.60        | 219.7      |
| VQGAN w/ Rotation Trick         | 64 × 64 × 3                 | 8192                       | 86%            | 1.7e-4                 | 0.142              | 0.27        | 228.0      |
| VQGAN                           | 32 × 32 × 4                 | 16384                      | 2%             | 1.2e-2                 | 0.385              | 5.0         | 141.5      |
| Gumbel VQGAN                    | 32 × 32 × 4                 | 16384                      | 12%            | —                      | 0.3031             | 1.7         | 189.5      |
| VQGAN w/ Rotation Trick         | 32 × 32 × 4                 | 16384                      | 27%            | 2.4e-4                 | 0.269              | 1.1         | 200.2      |

Evaluation Metrics. We log both training and validation set reconstruction metrics. Of note, we compute reconstruction FID [\(Heusel et al.,](#page-11-12) [2017\)](#page-11-12) and reconstruction IS [\(Salimans et al.,](#page-12-13) [2016\)](#page-12-13) on reconstructions from the full ImageNet validation set as a measure of reconstruction quality. We also compute codebook usage, or the percentage of codebook vectors that are used in each batch of data, as a measure of the information capacity of the vector quantization layer and quantization error ∥e − q∥ 2 2 as a measure of distortion.

Baselines. Our comparison spans the STE estimator (*VQ-VAE*), stochastic quantization with Gumbel-Softmax [\(Baevski et al.,](#page-10-4) [2019\)](#page-10-4), (*Gumbel VQ-VAE*) the Hessian approximation described in Section [3](#page-2-0) (*VQ-VAE w/ Hessian Approx*), the exact gradient backward pass described in Section [3](#page-2-0) (*VQ-VAE w/ Exact Gradients*), and the rotation trick (*VQ-VAE w/ Rotation Trick*). All methods share the same architecture, hyperparameters, and training settings, and these settings are summarized in [Table 8](#page-23-0) of the Appendix. There is no functional difference among methods in the forward pass; the only differences relates to how gradients are propagated through ∂q ∂e during backpropagation.

Results. Table [1](#page-7-1) displays our findings. We find that using the rotation trick reduces the quantization error—sometimes by an order of magnitude—and improves low codebook utilization. Both results are expected given the Voronoi partition analysis in Section [4.3:](#page-5-0) points at the boundary of quantized regions are likely pushed to under-utilized codebook vectors while points loosely grouped around the codebook vector are condensed towards it. These two features appear to have a meaningful effect on reconstruction metrics: training a VQ-VAE with the rotation trick substantially improves r-FID and r-IS.

We also see that the Hessian Approximation or using Exact Gradients results in poor reconstruction performance. While the gradients to the encoder are, in a sense, "more accurate", training the encoder like an AutoEncoder [\(Hinton & Zemel,](#page-11-8) [1993\)](#page-11-8) likely introduces overfitting and poor generalization. Moreover, the mismatch in training objectives between the encoder and decoder is likely an aggravating factor and partly responsible for both models' poor performance.

#### 5.2 VQGAN EVALUATION

Moving to the next level of complexity, we evaluate the effect of the rotation trick on VQGANs [\(Esser](#page-11-9) [et al.,](#page-11-9) [2021\)](#page-11-9). The VQGAN training objective is:

$$\mathcal{L}_{\text{VQGAN}} = \mathcal{L}_{\text{Per}} + \|sg(e) - q\|_2^2 + \beta \|e - sg(q)\|_2^2 + \lambda \mathcal{L}_{\text{Adv}}$$

where LPer is the perceptual loss from [Johnson et al.](#page-11-13) [\(2016\)](#page-11-13) and replaces the L<sup>2</sup> loss used to train VQ-VAEs. LAdv is a patch-based adversarial loss similar to the adversarial loss in Conditional GAN [\(Isola et al.,](#page-11-14) [2017\)](#page-11-14). β is a constant that weights the commitment loss while λ is an adaptive weight based on the ratio of ∇LPer to ∇LAdv with respect to the last layer of the decoder.

Experimental Settings. We evaluate VQGANs under two settings: (1) the paradigm amenable to autoregressive modeling with Transformers as described in [Esser et al.](#page-11-9) [\(2021\)](#page-11-9) and (2) the paradigm suitable to latent diffusion models as described in [Rombach et al.](#page-12-1) [\(2022\)](#page-12-1). The first setting follows the convolutional neural network and default hyperparameters described in [Esser et al.](#page-11-9) [\(2021\)](#page-11-9) while

Table 4: Results for ViT-VQGAN [\(Yu et al.,](#page-12-6) [2021\)](#page-12-6) trained on ImageNet. The latent shape is 8 × 8 × 32 with 8192 codebook vectors. r-FID and r-IS are reported on the validation set.

| Approach                    | Codebook Usage ( $\uparrow$ ) | Train Loss ( $\downarrow$ ) | Quantization Error ( $\downarrow$ ) | Valid Loss ( $\downarrow$ ) | r-FID ( $\downarrow$ ) | r-LS ( $\uparrow$ ) |
|-----------------------------|-------------------------------|-----------------------------|-------------------------------------|-----------------------------|------------------------|---------------------|
| VIT-VQGAN [reported]        | —                             | —                           | —                                   | —                           | 22.8                   | 72.9                |
| VIT-VQGAN [ours]            | 0.3%                          | 0.14                        | 6.7e-3                              | 0.127                       | 29.2                   | 43.0                |
| VIT-VQGAN w/ Rotation Trick | 2.2%                          | 0.13                        | 8.3e-3                              | <b>0.113</b>                | <b>11.2</b>            | <b>9.3</b>          |

the second follows those from [Rombach et al.](#page-12-1) [\(2022\)](#page-12-1). A full description of both training settings is provided in [Table 9](#page-24-0) of the Appendix.

Results. Our results are listed in Table [2](#page-8-0) for the first setting and Table [3](#page-8-1) for the second. Similar to our findings in Section [5.1,](#page-7-2) we find that training a VQ-VAE with the rotation trick substantially decreases quantization error and improves codebook usage. Moreover, reconstruction performance as measured on the validation set by the total loss, r-FID, and r-IS are improved across both modeling paradigms.

#### 5.3 VIT-VQGAN EVALUATION

Improving upon the VQGAN model, [Yu et al.](#page-12-6) [\(2021\)](#page-12-6) propose using a ViT [\(Dosovitskiy,](#page-11-15) [2020\)](#page-11-15) rather than CNN to parameterize the encoder and decoder. The ViT-VQGAN uses factorized codes and L<sup>2</sup> normalization on the output and input to the vector quantization layer to improve performance and training stability. Additionally, the authors change the training objective, adding a logit-laplace loss and restoring the L<sup>2</sup> reconstruction error to LVQGAN.

Experimental Settings. We follow the open source implementation of [https://github.com/thuanz123/](https://github.com/thuanz123/enhancing-transformers) [enhancing-transformers](https://github.com/thuanz123/enhancing-transformers) and use the default model and hyperparameter settings for the small ViT-VQGAN. A complete description of the training settings can be found in Table [10](#page-25-0) of the Appendix.

Results. Table [4](#page-9-0) summarizes our findings. Similar to our previous results for VQ-VAEs in Section [5.1](#page-7-2) and VQGANs in Section [5.2,](#page-8-2) codebook utilization and reconstruction metrics are significantly improved; however in this case, the quantization error is roughly the same.

#### 6 LIMITATIONS

![](_page_9_Figure_11.jpeg)

Figure 6: Illustration of the rotation trick "over-rotating" vectors when the angle between e<sup>1</sup> and q is obtuse. This is undesirable because—when the angle between e and q is obtuse—the rotation trick

A limitation of the rotation trick can arise when the encoder outputs or codebook vectors are forced to be close to 0 norm (i.e., ∥e∥ ≈ 0 or ∥q∥ ≈ 0). In this case, the angle between e and q may be obtuse. When this happens, the rotation trick will "over-rotate" the gradient ∇qL as it is transported from q to e so that ∇qL and ∇eL now point in different directions (i.e. the cosine of the angle between ∇eL and ∇qL will be negative). An example is visualized in Figure [6.](#page-9-1)

will violate the assumption that when e ≈ q, ∇qL ≈ ∇eL, and it will likely result in worse performance than VQ-VAEs trained with the STE. While obtuse angles between e and q are very unlikely—by design, the codebook vectors should be "angularly close" to the vectors that are mapped to them—however, if there is a restriction that forces codewords to have near 0 norm, then the rotation trick will likely perform worse than the STE.

#### 7 CONCLUSION

In this work, we explore different ways to propagate gradients through the vector quantization layer of VQ-VAEs and find that preserving the angle—rather than the direction—between the codebook vector and gradient induces desirable effects for how points within the same codebook region are updated. These effects cause a substantial improvement in model performance. Across 11 different settings, we find that training VQ-VAEs with the rotation trick improves their reconstructions. For example, training one of the VQGANs used in latent diffusion with the rotation trick improves r-FID from 5.0 to 1.1 and r-IS from 141.5 to 200.2, reduces quantization error by two orders of magnitude, and increases codebook usage by 13.5x.

#### ACKNOWLEDGMENTS

We thank Henry Bosch, Benjamin Spector, Dan Biderman, Jordan Juravsky, Mayee Chen, Owen Dugan, Sabri Eyuboglu, and the Hazy Group as a whole for their invaluable feedback and help during revisions of this work. We gratefully acknowledge the support of NIH under No. U54EB020405 (Mobilize), NSF under Nos. CCF2247015 (Hardware-Aware), CCF1763315 (Beyond Sparsity), CCF1563078 (Volume to Velocity), and 1937301 (RTML); US DEVCOM ARL under Nos. W911NF-23-2-0184 (Long-context) and W911NF-21-2-0251 (Interactive Human-AI Teaming); ONR under Nos. N000142312633 (Deep Signal Processing); Stanford HAI under No. 247183; NXP, Xilinx, LETI-CEA, Intel, IBM, Microsoft, NEC, Toshiba, TSMC, ARM, Hitachi, BASF, Accenture, Ericsson, Qualcomm, Analog Devices, Google Cloud, Salesforce, Total, the HAI-GCP Cloud Credits for Research program, the Stanford Data Science Initiative (SDSI), and members of the Stanford DAWN project: Meta, Google, and VMWare. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright notation thereon. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views, policies, or endorsements, either expressed or implied, of NIH, ONR, or the U.S. Government.

## REFERENCES


[1] Alexei Baevski, Steffen Schneider, and Michael Auli. vq-wav2vec: Self-supervised learning of discrete speech representations. *arXiv preprint arXiv:1910.05453*, 2019. Jonathan Baxter. A model of inductive bias learning. *Journal of artificial intelligence research*, 12: 149–198, 2000. Yoshua Bengio, Nicholas Léonard, and Aaron Courville. Estimating or propagating gradients through stochastic neurons for conditional computation. *arXiv preprint arXiv:1308.3432*, 2013. Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In *ICML*, volume 2, pp. 4, 2021. Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024. URL [https://openai.com/research/](https://openai.com/research/video-generation-models-as-world-simulators) [video-generation-models-as-world-simulators.](https://openai.com/research/video-generation-models-as-world-simulators) Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 11315–11325, 2022. Hang Chen, Sankepally Sainath Reddy, Ziwei Chen, and Dianbo Liu. Balance of number of embedding and their dimensions in vector quantization. *arXiv preprint arXiv:2407.04939*, 2024. Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks. In *International conference on machine learning*, pp. 794–803. PMLR, 2018. Chung-Cheng Chiu, James Qin, Yu Zhang, Jiahui Yu, and Yonghui Wu. Self-supervised learning with random-projection quantizer for speech recognition. In *International Conference on Machine Learning*, pp. 3915–3924. PMLR, 2022. Thomas M Cover. *Elements of information theory*. John Wiley & Sons, 1999. Mathieu Dagréou, Pierre Ablin, Samuel Vaiter, and Thomas Moreau. How to compute hessianvector products? In *ICLR Blogposts 2024*, 2024. URL [https://iclr-blogposts.github.io/2024/blog/](https://iclr-blogposts.github.io/2024/blog/bench-hvp/) [bench-hvp/.](https://iclr-blogposts.github.io/2024/blog/bench-hvp/) https://iclr-blogposts.github.io/2024/blog/bench-hvp/. Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *2009 IEEE conference on computer vision and pattern recognition*, pp. 248–255. Ieee, 2009.

[2] Prafulla Dhariwal, Heewoo Jun, Christine Payne, Jong Wook Kim, Alec Radford, and Ilya Sutskever. Jukebox: A generative model for music. *arXiv preprint arXiv:2005.00341*, 2020. Xiaoyi Dong, Jianmin Bao, Ting Zhang, Dongdong Chen, Weiming Zhang, Lu Yuan, Dong Chen, Fang Wen, Nenghai Yu, and Baining Guo. Peco: Perceptual codebook for bert pre-training of vision transformers. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 37, pp. 552–560, 2023. Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint arXiv:2010.11929*, 2020. Frederik Ebert, Chelsea Finn, Alex X Lee, and Sergey Levine. Self-supervised visual planning with temporal skip connections. *CoRL*, 12(16):23, 2017. Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 12873–12883, 2021. Tanmay Gautam, Reid Pryzant, Ziyi Yang, Chenguang Zhu, and Somayeh Sojoudi. Soft convex quantization: Revisiting vector quantization with convex optimization. *arXiv preprint arXiv:2310.03004*, 2023. Nabarun Goswami, Yusuke Mukuta, and Tatsuya Harada. Hypervq: Mlr-based vector quantization in hyperbolic space. *arXiv preprint arXiv:2403.13015*, 2024. Robert Gray. Vector quantization. *IEEE Assp Magazine*, 1(2):4–29, 1984. Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. *Advances in neural information processing systems*, 30, 2017. Geoffrey E Hinton and Richard Zemel. Autoencoders, minimum description length and helmholtz free energy. *Advances in neural information processing systems*, 6, 1993. Mengqi Huang, Zhendong Mao, Zhuowei Chen, and Yongdong Zhang. Towards accurate image coding: Improved autoregressive image generation with dynamic vector quantization. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 22596–22605, 2023. Minyoung Huh, Brian Cheung, Pulkit Agrawal, and Phillip Isola. Straightening out the straightthrough estimator: Overcoming optimization challenges in vector quantized networks. In *International Conference on Machine Learning*, pp. 14096–14113. PMLR, 2023. Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 1125–1134, 2017. Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. *arXiv preprint arXiv:1611.01144*, 2016. Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In *Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14*, pp. 694–711. Springer, 2016. Tero Karras. Progressive growing of gans for improved quality, stability, and variation. *arXiv preprint arXiv:1710.10196*, 2017. Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 4401–4410, 2019. Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 7482–7491, 2018.

[3] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. *arXiv preprint arXiv:1312.6114*, 2013. Alexander Kolesnikov, André Susano Pinto, Lucas Beyer, Xiaohua Zhai, Jeremiah Harmsen, and Neil Houlsby. Uvim: A unified modeling approach for vision with learned guiding codes. *Advances in Neural Information Processing Systems*, 35:26295–26308, 2022. Adrian Łancucki, Jan Chorowski, Guillaume Sanchez, Ricard Marxer, Nanxin Chen, Hans JGA ´ Dolfing, Sameer Khurana, Tanel Alumäe, and Antoine Laurent. Robust training of vector quantized bottleneck models. In *2020 International Joint Conference on Neural Networks (IJCNN)*, pp. 1–7. IEEE, 2020. Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 11523–11532, 2022. Fabian Mentzer, David Minnen, Eirikur Agustsson, and Michael Tschannen. Finite scalar quantization: Vq-vae made simple. *arXiv preprint arXiv:2309.15505*, 2023. Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 10684–10695, 2022. Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. *Advances in neural information processing systems*, 29, 2016. K Soomro. Ucf101: A dataset of 101 human actions classes from videos in the wild. *arXiv preprint arXiv:1212.0402*, 2012. Yuhta Takida, Takashi Shibuya, WeiHsiang Liao, Chieh-Hsin Lai, Junki Ohmura, Toshimitsu Uesaka, Naoki Murata, Shusuke Takahashi, Toshiyuki Kumakura, and Yuki Mitsufuji. Sq-vae: Variational bayes on discrete representation with self-annealed stochastic quantization. *arXiv preprint arXiv:2205.07547*, 2022. Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. *arXiv preprint arXiv:1812.01717*, 2018. Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. *Advances in neural information processing systems*, 30, 2017. A Vaswani. Attention is all you need. *Advances in Neural Information Processing Systems*, 2017. Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. *arXiv preprint arXiv:2104.10157*, 2021. Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. *arXiv preprint arXiv:2110.04627*, 2021. Lijun Yu, José Lezama, Nitesh B Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, et al. Language model beats diffusion– tokenizer is key to visual generation. *arXiv preprint arXiv:2310.05737*, 2023. Jiahui Zhang, Fangneng Zhan, Christian Theobalt, and Shijian Lu. Regularized vector quantization for tokenized image synthesis. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 18467–18476, 2023. Yue Zhao, Yuanjun Xiong, and Philipp Krähenbühl. Image and video tokenization with binary spherical quantization. *arXiv preprint arXiv:2406.07548*, 2024. Chuanxia Zheng and Andrea Vedaldi. Online clustered codebook. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 22798–22807, 2023.

[4] Zixin Zhu, Xuelu Feng, Dongdong Chen, Jianmin Bao, Le Wang, Yinpeng Chen, Lu Yuan, and Gang Hua. Designing a better asymmetric vqgan for stablediffusion. *arXiv preprint arXiv:2306.04632*, 2023.
#### A APPENDIX

Table 5: Results for TimeSformer-VQGAN trained on BAIR and UCF-101 with 1024 codebook vectors. †: model suffers from codebook collapse and diverges. r-FVD is computed on the validation set.

| Approach                      | Dataset | Codebook Usage | Train Loss ( ↓ ) | Quantization Error ( ↓ ) | Valid Loss ( ↓ ) | r-FVD ( ↓ ) |
|-------------------------------|---------|----------------|------------------|--------------------------|------------------|-------------|
| TimeSformer †                 | BAIR    | 0.4%           | 0.221            | 0.03                     | 0.28             | 1661.1      |
| TimeSformer w/ Rotation Trick | BAIR    | 43%            | 0.074            | 3.0e-3                   | 0.074            | 21.4        |
| TimeSformer †                 | UCF-101 | 0.1%           | 0.190            | 0.006                    | 0.169            | 2878.1      |
| TimeSformer w/ Rotation Trick | UCF-101 | 30%            | 0.111            | 0.020                    | 0.109            | 229.1       |

#### A.1 VIDEO EVALUATION

Expanding our analysis beyond the image modality, we evaluate the effect of the rotation trick on video reconstructions from the BAIR Robot dataset [\(Ebert et al.,](#page-11-16) [2017\)](#page-11-16) and from the UCF101 action recognition dataset [\(Soomro,](#page-12-14) [2012\)](#page-12-14). We follow the quantization paradigm used by ViT-VQGAN, but replace the ViT with a TimeSformer [\(Bertasius et al.,](#page-10-9) [2021\)](#page-10-9) video model. Due to compute limitations, both encoder and decoder follow a relatively small TimeSformer model: 8 layers, 256 hidden dimensions, 4 attention heads, and 768 MLP hidden dimensions. A complete description of the architecture, training settings, and hyperparameters are provided in Appendix [A.10.4.](#page-24-1)

Results. Table [5](#page-14-3) shows our results. For both datasets, training a TimeSformer-VQGAN model with the STE results in codebook collapse. We explored several different hyperparameter settings; however in all cases, codebook utilization drops to almost 0% within the first several epochs. On the other hand, models trained with the rotation trick do not exhibit any training instability and produce high quality reconstructions as indicated by r-FVD [\(Unterthiner et al.,](#page-12-15) [2018\)](#page-12-15). Several non-cherry picked video reconstructions are displayed in Appendix [A.10.4.](#page-24-1)

#### A.2 NON-CONVEX SYNTHETIC EXAMPLE

Figure 7: Loss surface for Himmelblau's function. Himmelblau's function has four equal local minima: f(3.0, 2.0) = 0.0, f(- 2.8.., 3.1...) = 0.0, f(-3.7.., -3.2..) = 0.0, and f(3.5.., -1.8..) = 0.0. Figure [8](#page-15-0) visualizes our results after 33, 66, and 100 gradient updates. The orange circles represent codebook vectors, the green dots the initial points, and the blue dots the updated points. Contour lines are drawn in each diagram to indicate regions of equal loss, with blue representing regions of low loss and red indicating regions of high loss. Similar to our findings in Section [5,](#page-7-0) we see that the rotation trick clusters points more tightly around each codebook vector when compared to the STE, resulting in lower distortion. Moreover, the codebook vectors more rapidly converge to the four equal local minima in Himmelblau's function, resulting in a lower objective function value when averaged across all points.

To supplement our analysis in Section [4.3,](#page-5-0) we include a numerical simulation of vector quantization for minimizing Himmelblau's function (Figure [7\)](#page-14-4) across 100 gradient updates for the STE and rotation trick gradient estimators to highlight the differences in their behaviors. Our simulation uses an EMA with a decay rate of 0.8 as described in [Van](#page-12-0) [Den Oord et al.](#page-12-0) [\(2017\)](#page-12-0) to update the codebook vectors and a learning rate of 1e−3 to update the pre-quantized points. Points for both the STE and the rotation trick simulation use the same random initialization for both codewords and pre-quantized vectors. The only difference is whether the STE or the rotation trick is used as the gradient estimator through the vector quantization operation.

![](_page_14_Figure_9.jpeg)

#### A.3 HESSIAN APPROXIMATION AND EXACT GRADIENT ANALYSIS

In this section, we expand our analysis in Section [3](#page-2-0) and offer some intuition for why using exact gradients, or a Hessian approximation of the exact gradients, may convey undesirable characteristics. We begin by showing the Hessian approximates the exact gradient up to second order term with a

![](_page_15_Figure_1.jpeg)

Figure 8: Synthetic experiment for minimizing Himmelblau's function with vector quantization using the STE gradient estimator (top row) and the rotation trick (bottom row). The rotation trick more quickly converges to these minima and achieves substantively lower distortion between codewords and pre-quantized points.

Taylor series expansion. We can write the loss L<sup>e</sup> exactly as an infinite series of around q:

$$\mathcal{L}_e = \mathcal{L}_q + (\nabla_q \mathcal{L})^T (e - q) + \frac{1}{2} (e - q)^T (\nabla_q^2 \mathcal{L}) (e - q) + \frac{1}{6} (e - q)^T \nabla_q^3 \mathcal{L} (e - q, e - q) + \dots$$

so that the loss computed by the Hessian approximation differs from the loss computed with the exact gradients method by the remainder term from truncating the Taylor series expansion after the second term:

$$\{\mathcal{L}_e\}_{\text{Hessian}} = \mathcal{L}_q + (\nabla_q \mathcal{L})^T (e - q) + \frac{1}{2} (e - q)^T (\nabla_q^2 \mathcal{L}) (e - q)$$

When differentiating both of these losses to compute the gradients, the difference between the exact gradient update and the Hessian update is:

$$\frac{\partial \mathcal{L}_e}{\partial e} - \{ \frac{\partial \mathcal{L}_e}{\partial e} \}_{\text{Hessian}} = \frac{\partial}{\partial e} \mathcal{O}(\|e - q\|^3)$$

$$\mathcal{O}(\|e - q\|^3) = \frac{1}{6}(e - q)^T \nabla_q^3 \mathcal{L}(e - q, e - q) + \dots$$

![](_page_16_Figure_1.jpeg)

Figure 9: Examples of how the gradient can change due to the presence of negative curvature or an indefinite Hessian. As the loss in each partition is quadratic, the exact gradient will equal the Hessian approximation. Notice that when q ≈ e, ∇qL ≈ ∇eL for both the STE and the rotation trick. As the Hessian approximation and exact gradients use the curvature of the loss surface to move ∇qL from q to e, the direction of the gradient can change substantively, even when q ≈ e.

The Hessian idea described in Section [3](#page-2-0) approximates the exact gradients to the encoder as if quantization did not occur, i.e. it approximates the gradient used to update the encoder in the original AutoEncoder [\(Hinton & Zemel,](#page-11-8) [1993\)](#page-11-8) model.

We now explore some instances where the exact gradients, or their Hessian approximation, may produce undesirable behavior in vector quantization. An inductive bias [\(Baxter,](#page-10-10) [2000\)](#page-10-10) for vector quantization to work well is that when e is "close" to q, their gradients are also "close", i.e. if e ≈ q then ∇eL ≈ ∇qL. Intuitively, if the distortion between e and q is small—i.e. q is a very good codeword for e—then these points should move together during a gradient update. If they do not, the distortion would increase.

This assumption holds for both the STE and Rotation Trick gradients; however, it can be violated by the Hessian approximation or the exact gradient approaches, especially when the curvature around q is negative or the Hessian is indefinite and forms a saddle point.

Figure [9](#page-16-1) illustrates three such cases. As both the STE and Rotation Trick do not use the loss surface to move ∇qL from q to e, when q ≈ e, ∇qL ≈ ∇eL. However, approaches that use the curvature around q, such as the Hessian approximation or exact gradients, to either find or approximate the loss at e can have ∇eL point in a very different direction from ∇qL, even when q is close to e. The top-left and bottom partitions of Figure [9](#page-16-1) scatter the gradients as they move from q to the points in these partitions due to negative curvature. A similar effect occurs in the top-right partition of Figure [9](#page-16-1) due to the presence of a saddle point.

#### A.4 BEHAVIOR AWAY FROM THE ORIGIN

Unlike the STE, the rotation trick is not invariant to the location of the origin. In this section, we explore this characteristic and its effect on how points within the same Voronoi region are updated. For example, suppose each codebook vector and encoder output in Figure [4](#page-6-1) were shifted by some

![](_page_17_Figure_1.jpeg)

Figure 10: Depiction of how points within the same codebook region change after a gradient update (red arrow) at the codebook vector (orange circle) when all points are far from the origin. The STE is invariant to the this translation; however as the angle between e and q decreases as these vectors translated away from the origin, the effect of the rotation trick will decrease. In the limit, the rotation trick reduces to the STE.

constant vector so that each now has all positive components. How would this affect the rotation trick's gradient estimator?

Figure 11: Illustration of codebook and encoder output shifted away from the origin by a constant vector d. The angle after the shift is smaller than the angle before the shift: ˆθ < θ.

Consider one codebook vector q and one encoder output e separated by angle θ. We define qˆ = q + d and eˆ = e + d where d is some large displacement vector. Let ˆθ be the angle between qˆ and eˆ. We visualize this example in Figure [11.](#page-17-0) From the law of cosines:

![](_page_17_Figure_5.jpeg)

$$\|q - e\|^2 = \|q\|^2 + \|e\|^2 - 2\|q\|\|e\| \cos(\theta)$$

and

$$\|\hat{q} - \hat{e}\|^2 = \|q - e\|^2 = \|\hat{q}\|^2 + \|\hat{e}\|^2 - 2\|\hat{q}\|\|\hat{e}\| \cos(\hat{\theta})$$

Substituting, we find that

$$\cos(\hat{\theta}) = \frac{\|q\|^2 + \|e\|^2 - 2\|q\|\|e\| \cos(\theta) - \|q + d\|^2 - \|e + d\|^2}{-2\|q + d\|\|e + d\|}$$

and consider the case when qˆ and eˆ are far from the origin, i.e.,∥d∥ >> ∥q∥, ∥e∥. Then we have:

$$\cos(\hat{\theta}) \approx \frac{-2\|d\|^2}{-2\|d\|^2} = 1$$

So as d → ∞, ˆθ → 0. This implies that <sup>∥</sup>qˆ<sup>∥</sup> <sup>∥</sup>eˆ<sup>∥</sup> <sup>→</sup> <sup>1</sup> and <sup>R</sup><sup>ˆ</sup> <sup>→</sup> <sup>I</sup>, which is exactly the STE update. As points move away from the origin, the rotation trick smoothly transforms into the STE.

We visualize an example of this effect in Figure [10,](#page-17-1) where each point from Figure [4](#page-6-1) is translated by positive ten along each dimension. As illustrated above, the effect for the "push" gradient in the top-right quadrant remains but it's effect is reduced, i.e., more similar to the STE update. The top-left partition becomes a "pull" because the gradient now points towards the origin, so points within this region move closer together. Finally, the gradient in the bottom region no longer points towards the origin, but is now more orthogonal to the codebook vector. As a result, we see more of a rotation applied to the points in this region than the contraction that is depicted in Figure [4.](#page-6-1)

#### A.5 HOUSEHOLDER REFLECTION TRANSFORMATION

For any given e and q, the rotation R that aligns e with q in the plane spanned by both vectors can be efficiently computed with Householder matrix reflections.

Definition 1 (Householder Reflection Matrix). *For a unit norm vector* a ∈ R d *,* I − 2aa<sup>T</sup> ∈ <sup>R</sup> d×d *is reflection matrix across the subspace (hyperplane) orthogonal to* a*.*

Remark 1. *Let* a, b ∈ R d *that define hyperplanes* a <sup>⊥</sup> *and* b <sup>⊥</sup> *respectively. Then a reflection across* a <sup>⊥</sup> *followed by a reflection across* b <sup>⊥</sup> *is a rotation of* 2θ *in the plane spanned by* a, b *where* θ *is the angle between* a, b*.*

Remark 2. *Let* a, b ∈ R <sup>d</sup> *with* ∥a∥ = ∥b∥ = 1*. Define* c = a+b ∥a+b∥ *as the vector half-way between* a *and* b *so that* <sup>∠</sup>(a, b) = θ *and* <sup>∠</sup>(a, c) = <sup>∠</sup>(b, c) = <sup>θ</sup> 2 *. From Definition [1,](#page-18-2)* (I − 2cc<sup>T</sup> ) *encodes a reflection across* c <sup>⊥</sup> *and* (I − 2bb<sup>T</sup> ) *encodes a reflection across* b <sup>⊥</sup>*. From Remark [1,](#page-18-3)* (I − 2bb<sup>T</sup> )(I − 2cc<sup>T</sup> ) *then corresponds to a rotation of* 2( <sup>θ</sup> 2 ) = θ *in the plane spanned by* b *and* c*. As the span*(b, c) = *span*(a, b)*,* (I − 2bb<sup>T</sup> )(I − 2cc<sup>T</sup> ) *corresponds to a rotation of* θ *in the plane spanned by* a *and* b*. Therefore,* (I − 2bb<sup>T</sup> )(I − 2cc<sup>T</sup> )a = b*.*

Returning to vector quantization with q = [ <sup>∥</sup>q<sup>∥</sup> <sup>∥</sup>e∥R]e, we can write R as the product of two Householder reflection matrices that rotates e to q in the plane spanned between them. Without loss of generality, assume e and q are unit norm, and let θ be the angle between e and q. Setting r = e+q ∥e+q∥ and simplifying yields:

$$\begin{aligned}
R &= (I - 2qq^T)(I - 2rr^T) \\
&= I - 2qq^T - 2rr^T + 4qq^T rr^T \\
&= I - 2qq^T - 2rr^T + 4q [q^T r] r^T \\
&= I - 2qq^T - 2rr^T + 4q \left[ q^T \frac{e + q}{\|e + q\|} \right] r^T \\
&= I - 2qq^T - 2rr^T + 4q \left[ \frac{q^T e + q^T q}{\|e + q\|} \right] r^T \\
&= I - 2qq^T - 2rr^T + 4q \left[ \frac{\|q\|\|e\| \cos \theta + \|q\|\|q\|}{\|e + q\|} \right] r^T \\
&= I - 2qq^T - 2rr^T + 4q \left[ \frac{\cos \theta + 1}{\|e + q\|} \right] r^T \\
&= I - 2qq^T - 2rr^T + 4q \left[ \frac{\|e + q\|^2}{2\|e + q\|} \right] r^T \\
&= I - 2qq^T - 2rr^T + \frac{4\|e + q\|^2}{2\|e + q\|} qr^T \\
&= I - 2qq^T - 2rr^T + \frac{4\|e + q\|^2}{2\|e + q\|^2} q(e + q)^T \\
&= I - 2qq^T - 2rr^T + 2qe^T + 2qq^T \\
&= I - 2rr^T + 2qe^T
\end{aligned}$$

#### A.6 PROOF THE ROTATION TRICK PRESERVES ANGLES

For encoder output e and corresponding codebook vector q, we provide a formal proof that the rotation trick preserves the angle between ∇qL and q as ∇qL moves to e. Unlike the notation in the main text, which assumes q ∈ R d×1 , we use batch notation in the following proof to illustrate how the rotation trick works when training neural networks. Specifically, q ∈ R b×d and R ∈ R b×d×d where b is the number of examples in a batch and d is the dimension of the codebook vector.

*Proof.* With loss of generality, suppose ∥e∥ = ∥q∥ = 1. Then we have

$$q = eR^T$$

$$\frac{\partial q}{\partial e} = R$$

The gradient at e will then equal:

$$\begin{aligned}\nabla_e \mathcal{L} &= \nabla_q \mathcal{L} \left[ \frac{\partial q}{\partial e} \right] \\ &= \nabla_q \mathcal{L} [R]\end{aligned}$$

Let θ be the angle between q and ∇qL and ϕ be the angle between e and ∇qL. Via the Euclidean inner product, we have:

$$\begin{aligned}\|\nabla_q\mathcal{L}\| \cos\theta &= q [\nabla_q\mathcal{L}]^T \\ &= eR^T [\nabla_q\mathcal{L}]^T \\ &= e [\nabla_q\mathcal{L}R]^T \\ &= e [\nabla_e\mathcal{L}]^T \\ &= \|\nabla_q\mathcal{L}\| \cos\phi\end{aligned}$$

so θ = ϕ and the angle between q and ∇qL is preserved as ∇qL moves to e.

#### A.7 TREATING R AND ||q|| ||e|| AS CONSTANTS

In the rotation trick, we treat R and ||q|| ||e|| as constants and detached from the computational graph during the forward pass of the rotation trick. In this section, we explain why this is the case.

The rotation trick computes the input to the decoder q˜ after performing a non-differentiable codebook lookup on e to find q. It is defined as:

$$\tilde{q} = \frac{||q||}{||e||} Re$$

As shown in Section [4,](#page-4-0) R is a function of both e and q. However, using the quantization function Q(e) = q, we can rewrite both ||q|| ||e|| and R as a single function of e:

$$\begin{aligned} f(e) &= \frac{\|\mathcal{Q}(e)\|}{\|e\|} \left[ I - 2 \left[ \frac{e + \mathcal{Q}(e)}{\|e + \mathcal{Q}(e)\|} \right] \left[ \frac{e + \mathcal{Q}(e)}{\|e + \mathcal{Q}(e)\|} \right]^T + 2\mathcal{Q}(e)e^T \right] \\ &= \frac{\|q\|}{\|e\|} R \end{aligned}$$

The rotation trick then becomes

$$\tilde{q} = f(e)e$$

and differentiating q˜ with respect to e gives us:

$$\frac{\partial \tilde{q}}{\partial e} = f'(e)e + f(e)$$

However, f ′ (e) cannot be computed as it would require differentiating through Q(e), which is a nondifferentiable codebook lookup. We therefore drop this term and use only f(e) as our approximation of the gradient through the vector quantization layer: ∂q˜ ∂e = f(e). This approximation conveys more information about the vector quantization operation than the STE, which sets ∂q˜ ∂e = I.

![](_page_20_Figure_1.jpeg)

Figure 12: Illustration of how the gradient at q moves to e via the STE, the rotation trick, and the reflection trick. The reflection trick matches the behavior of the rotation trick when the gradient ∇qL is parallel to q. However, it will reverse the components of the gradients orthogonal to q for points in q's partition. This effect is illustrated in the bottom two rows of the rightmost column.

![](_page_20_Figure_3.jpeg)

Figure 13: Depiction of how points within the same codebook region change after a gradient update (red arrow) at the codebook vector (orange circle). The STE applies the same update to each point in the same region. The reflection trick (Appendix [A.8\)](#page-20-0) modifies the update based on the location of each point with respect to the codebook vector. Note the top-left region of the reflection trick update, where the points actually move in the opposite direction of the gradient update.

#### A.8 THE REFLECTION TRICK

One may also use a single reflection to align e to q, rather than a rotation. For instance, using the notation from Appendix [A.5,](#page-18-1) setting r = e−q ∥e−q∥ and reflecting across the plane orthogonal to this vector via the Householder reflection (I − 2rr<sup>T</sup> ) will reflect e to q. We denote this reflection as R˜ so that q˜ = ∥q∥ <sup>∥</sup>e∥Re˜ . We call this approach "the reflection trick."

The reflection trick can result in undesirable behavior during the backward pass. While it replicates the rotation trick when ∇qL is parallel to q, as illustrated in the top two rows of Figure [12](#page-20-1) and the top-right and bottom regions of Figure [13,](#page-20-2) it reflects orthogonal components of the gradient across the hyperplane orthogonal to e − q so that these components are reversed. Simply, if the quantized gradient points "left" then the reflected gradient will point "right", and vice-versa. This behavior is undesirable for points with low distortion, e ≈ q, because it will cause e to move away from q along the components of the gradient orthogonal to q, thereby increasing distortion for two points that are a "good match". The top-left partition of Figure [13](#page-20-2) illustrates one such example. In this case, the gradient pushes the codebook vector "left" while the points in this region are pushed in the opposite direction of the gradient.

We evaluate this effect experimentally following the VQ-VAE evaluation paradigm from Table [1](#page-7-1) and the VQGAN evaluation paradigm from Table [3.](#page-8-1) While we did not train these models to completion due to GPU resource limitations, both paradigms exhibited poor convergence when trained with the reflection trick. Specifically, after one epoch, the validation loss was approximately 3x higher than the rotation trick for both 8192 and 16384 codebook VQGANs in Table [3.](#page-8-1) For the Euclidean codebook model with latent Shape 64 × 64 × 3 in Table [1,](#page-7-1) the validation loss was approximately 2x higher than the rotation trick after 15 epochs.

#### A.9 GRADIENT NORM SCALING IN THE ROTATION TRICK

In this section, we analyze the effect of the <sup>∥</sup>q<sup>∥</sup> ∥e∥ term in the rotation trick. While this norm rescaling is necessary to transform e into q during the forward pass, one could avoid the multiplicative factor by instead formulating the rotation trick as:

$$\tilde{q} = \underbrace{R}_{\text{constant}} e + \underbrace{(q - Re)}_{\text{constant}}$$

A possible benefit of this latter formulation is that ∂q ∂e = R, an orthogonal transformation with determinant one that does not shrink or expand space by a factor of <sup>∥</sup>q<sup>∥</sup> ∥e∥ . In this section, we analyze the differences between these two approaches and formulate both as specific instantiations of a more general family of rotation-based gradient approximations.

#### A.9.1 COMPARISON BETWEEN <sup>∥</sup>q<sup>∥</sup> ∥e∥ AND (q − Re)

An inductive bias of vector quantization is that when e ≈ q, then ∇eL ≈ ∇qL. Simply, when the distortion between e and q is small, the gradient for both e and q should be approximately the same. However when ∥e∥ ≈ 0 and a Euclidean metric is used to determine the closest codebook vector, the angle between e and q can be obtuse as illustrated in Figure [6.](#page-9-1) In this instance, the rotation trick will cause the gradient ∇eL to "over-rotate" and point away from ∇qL.

Using a grad scaling of ||q|| ||e|| can fix this. When ||e|| ≈ <sup>0</sup> and ||e|| <sup>&</sup>lt; ||q||, the norm of the gradient will be scaled up to push e away from the origin. Pushing e away from the origin makes the angle between e and q more of a factor when computing the Euclidean distance:

$$\|e - q\| = \sqrt{\|e\|^2 + \|q\|^2 - 2\|e\|\|q\| \cos \theta}$$

so e is more likely to map to a different q that forms an acute angle with it as ∥e∥ increases.

Now consider if ∥q∥ ≈ 0 and ∥e∥ > ∥q∥. When this occurs, the update to e will vanish because ∥q∥ <sup>∥</sup>e<sup>∥</sup> ≈ 0. This behavior may also be desirable because when q is close to the origin, there's a higher likelihood the angle between e and q would be obtuse.

We also explore this factor in ablation experiments for VQ-VAEs and VQGANs. Table [6](#page-22-1) mirrors Table [1](#page-7-1) and summarizes our findings for VQ-VAEs while Table [7](#page-22-2) mirrors Table [3](#page-8-1) and summarizes our findings for the VQGANs used in latent diffusion. In Table [6,](#page-22-1) we do not observe a difference between using q˜ = ||q|| ||e||Re and <sup>q</sup>˜ <sup>=</sup> Re + (<sup>q</sup> − Re). However, for the VQGAN results in Table [7,](#page-22-2) we find that using the grad scaling factor modestly improves performance.

Table 6: Comparison of the rotation trick using q˜ = ∥q∥ <sup>∥</sup>e∥Re with using q˜ = Re + (q − Re) for VQ-VAE models. The experimental setting follows Table [1.](#page-7-1)

| Rotation Trick Function Training Metrics                                     | Validation Metrics          |            |
|------------------------------------------------------------------------------|-----------------------------|------------|
| Codebook Usage ( ↑ ) Rec. Loss ( ↓ ) Quantization Error ( ↓ )                | Rec. Loss ( ↓ ) r-FID ( ↓ ) | r-IS ( ↑ ) |
| Codebook Lookup: Euclidean & Latent Shape: 64 × 64 × 3 & Codebook Size: 8192 |                             |            |
| ∥ q ∥                                                                        |                             |            |
| ∥ e ∥ Re 99% 0.028 1.4e-4                                                    | 0.030 16.5                  | 106.3      |
| Re − ( q − Re ) 100% 0.028 4.0e-4                                            | 0.030 16.5                  | 106.1      |

Table 7: Comparison of the rotation trick using q˜ = ∥q∥ <sup>∥</sup>e∥Re with using q˜ = Re+ (q −Re) for VQGAN models. The models with codebook size were stopped after 2 epochs while the models with codebook size 16384 were stopped after 3 epochs.

| Rotation Trick Function ∥ q ∥ | Latent Shape | Codebook Size | Codebook Usage | Quantization Error ( ↓ ) | Valid Loss ( ↓ ) | r-FID ( ↓ ) | r-IS ( ↑ ) |
|-------------------------------|--------------|---------------|----------------|--------------------------|------------------|-------------|------------|
| ∥ e ∥ Re                      | 64 × 64 × 3  | 8192          | 45%            | 4.0e-4                   | 0.161            | 0.46        | 225.0      |
| Re − ( q − Re )               | 64 × 64 × 3  | 8192          | 28%            | 1.5e-3                   | 0.183            | 0.6         | 220.0      |
| ∥ q ∥                         |              |               |                |                          |                  |             |            |
| ∥ e ∥ Re                      | 32 × 32 × 4  | 16384         | 18%            | 3.3e-4                   | 0.292            | 1.5         | 196.1      |
| Re − ( q − Re )               | 32 × 32 × 4  | 16384         | 13%            | 9.4e-4                   | 0.292            | 1.5         | 191.5      |

#### A.9.2 GENERAL FAMILY OF ROTATION-BASED GRADIENT ESTIMATORS

Generalizing the additive and multiplicative formulations of the rotation trick, we formulate both as specific instantiations of a more general family:

$$\tilde{q} = \gamma(e)Re + (q - \gamma(e)Re)$$

where γ(e) determines the multiplicative scaling factor. For q˜ = ∥q∥ <sup>∥</sup>e∥Re, <sup>γ</sup>(e) = <sup>∥</sup>q<sup>∥</sup> ∥e∥ and for q˜ = Re + (q − Re), γ(e) = 1. However, one can explore other scaling factors such as

$$\gamma(e) = \frac{1}{8\|q - e\|^2}$$

We visualize the gradient fields for different formulations of γ(e) in Figure [14.](#page-23-1)

It is almost certain that other formulations of γ(e) from the ones we explore in this work would improve the training dynamics or performance of VQ-VAEs. In particular, *a priori* fixing γ(e) to satisfy an inductive bias or developing an adaptive scaling factor that dynamically sets γ(e) similar to the functions that adapt task weights in multi-task learning throughout training [\(Kendall et al.,](#page-11-17) [2018;](#page-11-17) [Chen et al.,](#page-10-11) [2018\)](#page-10-11) are exciting directions for future work.

#### A.10 TRAINING SETTINGS

We detail the training settings used in our experimental analysis in Section [5.](#page-7-0) While a text description can be helpful for understanding the experimental settings, our released code should be referenced to fully reproduce the results presented in this work.

#### A.10.1 VQ-VAE EVALUATION.

Table [8](#page-23-0) summarizes the hyperparameters used for the experiments in Section [5.1.](#page-7-2) For the encoder and decoder architectures, we use the Convolutional Neural Network described by [Esser et al.](#page-11-9) [\(2021\)](#page-11-9). The hyperparameters for the cosine similarity codebook lookup follow from [Yu et al.](#page-12-6) [\(2021\)](#page-12-6) and the hyperparameters for the Euclidean distance codebook lookup follow from the default values set in the Vector Quantization library from [https://github.com/lucidrains/vector-quantize-pytorch.](https://github.com/lucidrains/vector-quantize-pytorch) All models replace the codebook loss with the exponential moving average described in [Van Den Oord](#page-12-0) [et al.](#page-12-0) [\(2017\)](#page-12-0) with decay = 0.8. The notation for both encoder and decoder architectures is adapted from [Esser et al.](#page-11-9) [\(2021\)](#page-11-9).

For the Gumbel VQ-VAE baseline, we follow the implementation of [https://github.com/karpathy/](https://github.com/karpathy/deep-vector-quantization) [deep-vector-quantization](https://github.com/karpathy/deep-vector-quantization) and use the suggested schedule to attenuate the softmax temperature from 1.0 to <sup>1</sup> <sup>16</sup> over the course of training. Aside from the difference in quantization, i.e. deterministic

![](_page_23_Figure_1.jpeg)

Figure 14: Visualization of how different choices of γ(e) in the rotation trick affect the gradient field for (top) f(x, y) = x <sup>2</sup> + y and (bottom) f(x, y) = log x + tanh(y)| . To prevent cluttered visualizations, the maximum and minimum gradient norms are capped within the gradient field.

Table 8: Hyperparameters for the experiments in Table [1.](#page-7-1) (1024, 32) indicates a model trained with a codebook size of 1024 and codebook dimension of 32. Similarly, (8192, 3) indicates a model trained with codebook size of 8192 and codebook dimension of 3.

|                 |                   | Cosine ( 1024 , 32 ) | Similarity Lookup ( 8192 , 3 ) | Euclidean ( 1024 , 32 ) | Lookup ( 8192 , 3 ) |
|-----------------|-------------------|----------------------|--------------------------------|-------------------------|---------------------|
| Input           | size              | 256 × 256 × 3        | 256 × 256 × 3                  | 256 × 256 × 3           | 256 × 256 × 3       |
| Latent          | size              | 16 × 16 × 32         | 64 × 64 × 3                    | 16 × 16 × 32            | 64 × 64 × 3         |
| β (commitment   | loss coefficient) | 1.0                  | 1.0                            | 1.0                     | 1.0                 |
| encoder/decoder | channels          | 128                  | 128                            | 128                     | 128                 |
| encoder/decoder | channel mult.     | [1, 1, 2, 2, 4]      | [1, 2, 4]                      | [1, 1, 2, 2, 4]         | [1, 2, 4]           |
| [Effective]     | Batch size        | 256                  | 256                            | 256                     | 256                 |
| Learning        | rate              | 1 × 10 − 4           |                                |                         |                     |
|                 |                   |                      | 1 × 10 − 4                     |                         |                     |
|                 |                   |                      |                                | 5 × 10 − 5              |                     |
|                 |                   |                      |                                |                         | 5 × 10 − 5          |
| Weight          | Decay             | 1 × 10 − 4           |                                |                         |                     |
|                 |                   |                      | 1 × 10 − 4                     |                         |                     |
|                 |                   |                      |                                | 0                       | 0                   |
| Codebook        | size              | 1024                 | 8192                           | 1024                    | 8192                |
| Codebook        | dimension         | 32                   | 3                              | 32                      | 3                   |
| Training        | epochs            | 25                   | 20                             | 25                      | 20                  |

versus stochastic, the architecture and optimization of the Gumbel VQ-VAE model are identical to the VQ-VAE baseline.

#### A.10.2 VQGAN EVALUATION

Table [9](#page-24-0) summarizes the hyperparameters for the VQGAN experiments in Section [5.2.](#page-8-2) For the Gumbel VQGAN model, we follow the default hyperparameters and settings from [Rombach et al.](#page-12-1) [\(2022\)](#page-12-1). Non-cherry picked reconstructions for the models trained in Table [2](#page-8-0) and Table [3](#page-8-1) are depicted in Figure [15.](#page-24-2) As indicated by the increased r-FID score, the reconstructions out by the VQGAN trained with the rotation trick appear to better reproduce the original image, especially fine details.

#### A.10.3 VIT-VQGAN EVALUATION

Our experiments in Section [5.3](#page-9-2) use the ViT-VQGAN implemented in the open source repository [https://github.com/thuanz123/enhancing-transformers.](https://github.com/thuanz123/enhancing-transformers) The default hyperparameters follow those

| Orig STE ROT Orig              | STE ROT                     |
|--------------------------------|-----------------------------|
| ImageNet                       | FFHQ & CelebA-HQ            |
| VQGAN from Taming Transformers | VQGAN from Latent Diffusion |
|                                | Orig STE ROT STE ROT        |
|                                | ImageNet [f=8] [f=4]        |

Figure 15: Non-cherry picked reconstructions for VQGAN results in Table [2](#page-8-0) and Table [3.](#page-8-1) *ROT* is an abbreviation for the rotation trick.

Table 9: Hyperparameters for the experiments in Table [2](#page-8-0) and Table [3.](#page-8-1) We implement the rotation trick in the open source <https://github.com/CompVis/taming-transformers> for the experiments in Table [2](#page-8-0) and implement the rotation trick in <https://github.com/CompVis/latent-diffusion> for Table [3.](#page-8-1) In both settings, we use the default hyperparameters. †: 18 epochs for ImageNet and 50 epochs for FFHQ & CelebA-HQ.

|                 |               | Table 2 VQGAN       | Table 3 VQGAN | Table 3 VQGAN   |
|-----------------|---------------|---------------------|---------------|-----------------|
| Input           | size          | 256 × 256 × 3       | 256 × 256 × 3 | 256 × 256 × 3   |
| Latent          | size          | 16 × 16 × 256       | 64 × 64 × 3   | 32 × 32 × 4     |
| Codebook        | weight        | 1 0                 | 1 0           | 1 0             |
| Discriminator   | weight        | 0 8                 | 0 75          | 0 6             |
| encoder/decoder | channels      | 128                 | 128           | 128             |
| encoder/decoder | channel mult. | [1 , 1 , 2 , 2 , 4] | [1 , 2 , 4]   | [1 , 2 , 2 , 4] |
| [Effective]     | Batch size    | 48                  | 16            | 16              |
| [Effective]     | Learning rate | 4 5 × 10 − 6        |               |                 |
|                 |               |                     | 4 5 × 10 − 6  |                 |
|                 |               |                     |               | 4 5 × 10 − 6    |
| Codebook        | size          | 1024                | 8192          | 16384           |
| Codebook        | dimensions    | 256                 | 3             | 4               |
| Training        | Epochs        | 18/50 †             |               |                 |
|                 |               |                     | 4             | 4               |

specified by [Yu et al.](#page-12-6) [\(2021\)](#page-12-6), and our experiments use the default architecture settings specified by the ViT small model configuration file.

We depict several reconstructions in Figure [16](#page-25-0) and see that the ViT-VQGAN trained with the rotation trick is able to better replicate small details that the ViT-VQGAN trained with the STE misses. This is expected as the rotation trick drops r-FID from 29.2 to 11.2 as shown in Table [4.](#page-9-0)

#### A.10.4 TIMESFORMER VIDEO EVALUATION

We use the Hugging Face implementation of the TimeSformer from [https://huggingface.co/docs/](https://huggingface.co/docs/transformers/en/model_doc/timesformer) [transformers/en/model\\_doc/timesformer](https://huggingface.co/docs/transformers/en/model_doc/timesformer) and the ViT-VQGAN vector quantization layer from [https:](https://github.com/thuanz123/enhancing-transformers) [//github.com/thuanz123/enhancing-transformers.](https://github.com/thuanz123/enhancing-transformers) We loosely follow the hyperparameters listed in [Yu](#page-12-6) [et al.](#page-12-6) [\(2021\)](#page-12-6) and implement a small TimeSformer encoder and decoder due to GPU VRAM constraints. We reuse the dataloading functions of both BAIR Robot Pushing and UCF101 dataloaders from [Yan](#page-12-16) [et al.](#page-12-16) [\(2021\)](#page-12-16) at [https://github.com/wilson1yan/VideoGPT.](https://github.com/wilson1yan/VideoGPT) A complete description of the settings we use for the experiments in Appendix [A.1](#page-14-2) are listed in Table [11.](#page-26-0)

We also visualize the reconstructions for the TimeSformer-VQGAN trained with the rotation trick and the STE. Figure [17](#page-25-1) shows the reconstructions for BAIR Robot Pushing, and Figure [18](#page-26-1) shows the

Table 10: Hyperparameters for the experiments in Table [4.](#page-9-0)

|             |             |                      | ViT-VQGAN Settings |
|-------------|-------------|----------------------|--------------------|
| Input       | size        |                      | 256 × 256 × 3      |
| Patch       | size        |                      | 8                  |
| Encoder     | /           | Decoder Hidden Dim   | 512                |
| Encoder     | /           | Decoder MLP Dim      | 1024               |
| Encoder     | /           | Decoder Hidden Depth | 8                  |
| Encoder     | /           | Decoder Hidden Num   | Heads 8            |
| Codebook    |             | Dimension            | 32                 |
| Codebook    | Size        |                      | 8192               |
| Codebook    | Loss        | Coefficient          | 1.0                |
| Log         | Laplace     | loss Coefficient     | 0.0                |
| Log         | Gaussian    | Coefficient          | 1.0                |
| Perceptual  | loss        | Coefficient          | 0.1                |
|             | Adversarial | loss Coefficient     | 0.1                |
| [Effective] |             | Batch size           | 32                 |
| Learning    | rate        |                      | 1 × 10 − 4         |
| Weight      | Decay       |                      | 1 × 10 − 4         |
| Training    | epochs      |                      | 10                 |

Orig STE ROT ViT-VQGAN

Figure 16: Non-cherry picked reconstructions for ViT-VQGAN results in Table [4.](#page-9-0) *ROT* is an abbreviation for the rotation trick.

Original Video Rotation Trick STE Reconstructions

Reconstructions

Figure 17: BAIR Robot Pushing reconstruction examples. While the model trains on 16 video frames at a time, we only visualize 4 at a time in this figure. The model trained with the STE undergoes codebook collapse, using 4 out of the 1024 codebook vectors for reconstruction and therefore crippling the information capacity of the vector quantization layer. On the other hand, the VQ-VAE trained with the rotation trick instead uses an average of 441 of the 1024 codebook vectors in each batch of 2 example videos.

reconstructions for UCF101. For both datasets, the model trained with the STE undergoes codebook collapse early into training. Specifically, it learns to only use <sup>4</sup> <sup>1024</sup> of the available codebook vectors for BAIR Robot Pushing and <sup>2</sup> <sup>2048</sup> for UCF101 in a batch of 2 input examples. Small manual tweaks to the architecture and training hyperparameters did not fix this issue.

In contrast, VQ-VAEs trained with the rotation trick do not manifest this training instability. Instead, codebook usage is relatively high—at 43% for BAIR Robot Pushing and 30% for UCF101—and the reconstructions accurately match the input, even though both encoder and decoder are very small video models.

#### A.10.5 CREATION OF VORONOI REGION FIGURE

In this section, we describe the creation of Figure [4](#page-6-1) as well as the other figures that use this format. For the top-right and bottom partitions, we fix the codebook to a set of preset values and sample pre-quantized points from four different Gaussian distributions. For the pre-quantized points in the top-left partition, we manually set them to form a crescent shape around the codeword.

Original Video Rotation Trick

Reconstructions

STE Reconstructions

Figure 18: UCF-101 reconstruction examples. While the model trains on 16 video frames at a time, we only visualize 4 at a time in this figure. The model trained with the STE undergoes codebook collapse, using approximately 2 out of the 2048 codebook vectors for reconstruction and therefore crippling the information capacity of the vector quantization layer. The VQ-VAE trained with the rotation trick instead uses an average of 615 of the 2048 codebook vectors in each batch of 2 example videos.

Table 11: Hyperparameters for the experiments in Table [5.](#page-14-3) A TimeSformer [\(Bertasius et al.,](#page-10-9) [2021\)](#page-10-9) is used for the Encoder and Decoder architecture as implemented at [https://huggingface.co/docs/](https://huggingface.co/docs/transformers/en/model_doc/timesformer) [transformers/en/model\\_doc/timesformer.](https://huggingface.co/docs/transformers/en/model_doc/timesformer) The vector quantization layer between Encoder and Decoder follow from [Yu et al.](#page-12-6) [\(2021\)](#page-12-6) as implemented in [https://github.com/thuanz123/enhancing-transformers.](https://github.com/thuanz123/enhancing-transformers)

|                                    | TimeSformer-VQGAN Settings        |                                     |
|------------------------------------|-----------------------------------|-------------------------------------|
|                                    | BAIR Robot Pushing                | UCF101 Action Recognition           |
| Input size                         | $16 \times 64 \times 64 \times 3$ | $16 \times 128 \times 128 \times 3$ |
| Patch size                         | 2                                 | 4                                   |
| Encoder / Decoder Hidden Dim       | 256                               | 256                                 |
| Encoder / Decoder MLP Dim          | 768                               | 768                                 |
| Encoder / Decoder Hidden Depth     | 8                                 | 8                                   |
| Encoder / Decoder Hidden Num Heads | 4                                 | 4                                   |
| Codebook Dimension                 | 32                                | 32                                  |
| Codebook Size                      | 1024                              | 2048                                |
| Codebook Loss Coefficient          | 1.0                               | 1.0                                 |
| Log Laplace loss Coefficient       | 0.0                               | 0.0                                 |
| Log Gaussian Coefficient           | 1.0                               | 1.0                                 |
| Perceptual loss Coefficient        | 0.1                               | 0.1                                 |
| Adversarial loss Coefficient       | 0.1                               | 0.1                                 |
| [Effective] Batch size             | 24                                | 20                                  |
| Learning rate                      | $1 \times 10^{-4}$                | $4.5 \times 10^{-6}$                |
| Weight Decay                       | $1 \times 10^{-4}$                | $1 \times 10^{-4}$                  |
| Training epochs                    | 30                                | 3                                   |

We similarly fix constant gradient vectors for each partition, and apply them to the pre-quantized points after transformation by the STE, i.e. simply moving the gradient to each pre-quantized point in the quantized region, or by the rotation trick, i.e. rotating the gradient based on the angle between the pre-quantized point and closest codebook vector and rescaling appropriately. We multiply the gradient by a small constant—the learning rate—and then apply the gradient to each pre-quantized point. We repeat the above 25 times, at each point re-computing the angle and magnitude between the pre-quantized point and the codebook vector for the rotation trick update. For simplicity, we do not update the codebook vectors themselves or recompute codebook regions throughout the numerical simulation.

#### A.11 COMPARISON WITHIN GENERATIVE MODELING APPLICATIONS

Absent from our work is an analysis on the effect of VQ-VAEs trained with the rotation trick on down-stream generative modeling applications. We see this comparison as outside the scope of this

work and do not claim that improving reconstruction metrics, codebook usage, or quantization error in "Stage 1" VQ-VAE training will lead to improvements in "Stage 2" generative modeling applications.

While poor reconstruction performance will clearly lead to poor generative modeling, recent work [\(Yu](#page-12-12) [et al.,](#page-12-12) [2023\)](#page-12-12) suggests that—at least for autoregressive modeling of codebook sequences with MaskGit [\(Chang et al.,](#page-10-5) [2022\)](#page-10-5)—the connection between VQ-VAE reconstruction performance and downstream generative modeling performance is non-linear. Specifically, increasing the size of the codebook past a certain amount will improve VQ-VAE reconstruction performance but make downstream likelihood-based geneative modeling of codebook vectors more difficult.

We believe this nuance may extend beyond MaskGit, and that the *desiderata* for likelihood-based generative models will likely be different than that for score-based generative models like diffusion. It is even possible that different preferences appear within the same class. For example, left-to-right autoregressive modeling of codebook elements with Transformers [\(Vaswani,](#page-12-17) [2017\)](#page-12-17) may exhibit different preferences for Stage 1 VQ-VAE models than those of MaskGit.

These topics deserve a deep, and rich, analysis that we would find difficult to include within this work as our focus is on propagating gradients through vector quantization layers. As a result, we entrust the exploration of these questions to future work.

#### A.12 GRADIENT ESTIMATORS AS PARALLEL TRANSPORT

In this section, we analyze the STE and the rotation trick through the lens of differential geometry, specifically as the parallel transport of the gradient ∇qL vector from the codeword q to the encoder output e. For this analysis in this section, we only consider the rotational component R<sup>θ</sup> of the rotation trick, not the rescaling by <sup>∥</sup>q<sup>∥</sup> ∥e∥ .

#### A.12.1 BACKGROUND ON HYPERSPHERICAL COORDINATES

Hyperspherical coordinate systems are ubiquitous in applications of math and physics, where certain formulas become greatly simplified by parameterizing the location of points by the radius and angles to coordinate axes. An familiar instantiation of the hyperspherical coordinate system may be polar coordinates with radial component r and polar angle θ:

$$\begin{aligned} x &= r \cos \theta \\ y &= r \sin \theta \end{aligned}$$

or the instantiation of the hyperspherical coordinate system for three dimensions, otherwise known as spherical coordinates, with radial component r, polar angle θ and azimuthal angle ϕ:

- $$x = r \cos \theta$$
- $y = r \sin \theta \cos \phi$
- $z = r \sin \theta \sin \phi$

More generally, hyperspherical coordinates are composed by a radial coordinate r and d − 1 angular coordinates θ1, ..., θd−<sup>1</sup> where θ1, ...θd−<sup>2</sup> are supported over [0, π] while θd−1 ranges from [0, 2π]. We outline one common conversion from Cartesian coordinates to hyperspherical coordinates below, and other conversions are equivalent up to permutation of the coordinate axes:

$$\begin{aligned} x_1 &= r \cos(\theta_1) \\ x_2 &= r \sin(\theta_1) \cos(\theta_2) \\ x_3 &= r \sin(\theta_1) \sin(\theta_2) \cos(\theta_3) \\ &\vdots \\ x_{d-1} &= r \sin(\theta_1) \cdots \sin(\theta_{d-2}) \cos(\theta_{d-1}) \\ x_d &= r \sin(\theta_1) \cdots \sin(\theta_{d-2}) \sin(\theta_{d-1}) \end{aligned}$$

![](_page_28_Figure_1.jpeg)

![](_page_28_Picture_2.jpeg)

Figure 19: Visualization of basis vectors at different points under Cartesian (left) and spherical (right) coordinatate systems. Notice that the Cartesian basis vectors do not change from point-to-point; however, the spherical basis vectors change in both direction and magnitude. Even at the same radius, the <sup>∂</sup> ∂ϕ coordinate changes based on the azimuth angle θ because the same infinitesimal change in ϕ will result in a longer (or smaller) change in arclength depending on the radius of the circle at latitude θ.

and the reverse transform from Cartesian coordinates to hyperspherical coordinates:

$$\begin{aligned} r &= \sqrt{(x_1)^2 + (x_2)^2 + \dots + (x_d)^2} \\ \theta_1 &= \arctan 2(\sqrt{(x_d)^2 + \dots + (x_2)^2}, x_1) \\ \theta_2 &= \arctan 2(\sqrt{(x_d)^2 + \dots + (x_3)^2}, x_2) \\ &\vdots \\ \theta_{d-2} &= \arctan 2(\sqrt{(x_d)^2 + (x_{d-1})^2}, x_{d-2}) \\ \theta_{d-1} &= \arctan 2(\sqrt{(x_d)^2}, x_{d-1}) \end{aligned}$$

where arctan 2(x, y) returns the angle measurement in radians over the support (−π, π] between between x and y.

Unlike the Cartesian coordinate system, the hyperspherical basis vectors are not identical over the entire space; they change with position. For instance, moving outwards along r will increase the length of <sup>∂</sup> ∂θ<sup>i</sup> as an infinitesimal change in θ <sup>i</sup> will now cover a larger arclength distance—i.e. the line segment traveled by changing the angle θ <sup>i</sup>—than that same infinitesimal change with a smaller r. This effect is visualized for three dimensions in Figure [19.](#page-28-0)

At any given point in hyperspherical coordinates p˜, the transformation from Cartesian basis vectors ∂ ∂x<sup>1</sup> , ∂ ∂x<sup>2</sup> , ... to hyperspherical basis vectors <sup>∂</sup> ∂r , ∂ ∂θ<sup>1</sup> , ... can be computed with the multivariate chain rule:

$$\frac{\partial}{\partial \theta^i} = \sum_{k=1}^d \frac{\partial x^k}{\partial \theta^i} \frac{\partial}{\partial x^k}$$

where ∂x<sup>i</sup> ∂θ<sup>i</sup> can be computed from the coordinate transform functions, i.e. x<sup>1</sup> = r cos(θ1). It is typical to express these relationships in a matrix that transforms an arbitrary vector v in Cartesian coordinates at point p to its counterpart in hyperspherical coordinates v˜ at p˜:

$$\begin{bmatrix} \frac{\partial}{\partial x^1} & \frac{\partial}{\partial x^2} & \cdots & \frac{\partial}{\partial x^d} \end{bmatrix} \begin{bmatrix} \frac{\partial x^1}{\partial r} & \frac{\partial x^1}{\partial \theta^1} & \cdots & \frac{\partial x^1}{\partial \theta^{d-1}} \\ \frac{\partial x^2}{\partial r} & \frac{\partial x^2}{\partial \theta^1} & \cdots & \frac{\partial x^2}{\partial \theta^{d-1}} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial x^d}{\partial r} & \frac{\partial x^d}{\partial \theta^1} & \cdots & \frac{\partial x^d}{\partial \theta^{d-1}} \end{bmatrix} = \begin{bmatrix} \frac{\partial}{\partial r} & \frac{\partial}{\partial \theta^1} & \cdots & \frac{\partial}{\partial \theta^{d-1}} \end{bmatrix}$$

As illustrated in Figure [19,](#page-28-0) J does not necessarily have determinant equal to one and changes as a function of position, so the norms of the basis vectors spanning the hyperspherical tangent space change based on position. More generally, this notion of distance is captured by the line element: the length of a line segment resulting from an infinitesimal change along the coordinate axes. The Cartesian line element is given by:

$$ds^2 = (dx^1)^2 + (dx^2)^2 + \dots + (dx^d)^2$$

while the hyperspherical line element is:

$$ds^2 = dr^2 + r^2(d\theta^1)^2 + r^2 \sin^2 \theta_1 (d\theta^1)^2 + r^2 \left[ \prod_{i=2}^{d-1} \sin^2 \theta_i \right] (d\theta^{d-1})^2$$

which reflects that distance traveled by small changes in the hyperspherical coordinates "increases" with increasing radius and "decreases" with distance from the equator. To ensure that the norm of the basis vectors does not change during conversion, it is common to renormalize hyperspherical basis vectors to have unit norm for all points. However, a notion of norm is not defined *a priori* for hyperspherical vectors; the metric tensor imposed on this space defines the inner product which in turn defines a sense of arclength.

Using the induced metric from Cartesian coordinates, we can inherit the inner product from Cartesian coordinates on the hyperspherical coordinate system by expressing hyperspherical basis vectors as a linear combination of Cartesian basis vectors and then computing the norm of this resulting vector in the Cartesian tangent space:

$$\begin{aligned} \| \frac{\partial}{\partial \theta^i} \| &= \sqrt{\left\langle \frac{\partial}{\partial \theta^i}, \frac{\partial}{\partial \theta^i} \right\rangle} \\ &= \sqrt{\left[ \sum_{k=1}^d \frac{\partial x^k}{\partial \theta^i} \frac{\partial}{\partial x^k} \right] \cdot \left[ \sum_{j=1}^d \frac{\partial x^j}{\partial \theta^i} \frac{\partial}{\partial x^j} \right]} \\ &= \sqrt{\sum_{k=1}^d \frac{\partial x^k}{\partial \theta^i} \frac{\partial x^k}{\partial \theta^i} \left[ \frac{\partial}{\partial x^k} \cdot \frac{\partial}{\partial x^k} \right]} \\ &= \sqrt{\sum_{k=1}^d \left( \frac{\partial x^k}{\partial \theta^i} \right)^2} \end{aligned}$$

The first fundamental form gives us the normalization constants:

$$\mathcal{I} = \begin{bmatrix} 1^2 & 0 & 0 & \dots & 0 \\ 0 & r^2 & 0 & \dots & 0 \\ 0 & 0 & r^2 \sin^2 \theta_1 & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \dots & r^2 \prod_{i=1}^{d-1} \sin^2 \theta_i \end{bmatrix}$$

as the diagonal represents the inner product ⟨ ∂ ∂θ<sup>i</sup> , ∂ ∂θ<sup>i</sup> ⟩, and we would like to renormalize each basis vector to have unit norm: ∥ ∂ ∂θ<sup>i</sup> ∥ = q ⟨ ∂θ<sup>i</sup> , ∂ ∂θ<sup>i</sup> ⟩. Therefore, our normalized hyperspherical basis vectors <sup>∂</sup> ∂rˆ , ∂θˆ<sup>1</sup> , ... become:

$$\frac{\partial}{\partial \hat{r}} = \frac{\partial}{\partial r}$$

$$\frac{\partial}{\partial \hat{\theta}_i} = (\mathcal{L}_{ii})^{-\frac{1}{2}} \frac{\partial}{\partial \theta_i}$$

Using our convention from earlier, we can now compute the transformation from Cartesian basis vectors to normalized hyperspherical basis vectors:

$$\frac{\partial}{\partial \hat{\theta}^i} = (\mathcal{I}_{ii})^{-\frac{1}{2}} \sum_{k=1}^d \frac{\partial x^k}{\partial \theta^i} \frac{\partial}{\partial x^k}$$

to compose the normalized "Jacobian" Jˆ:

$$\begin{bmatrix} \frac{\partial}{\partial x^1} & \frac{\partial}{\partial x^2} & \cdots & \frac{\partial}{\partial x^d} \end{bmatrix} \underbrace{\begin{bmatrix} \frac{\partial x^1}{\partial \hat{r}} & \frac{\partial x^1}{\partial \hat{\theta}_1} & \cdots & \frac{\partial x^1}{\partial \hat{\theta}_{d-1}} \\ \frac{\partial x^2}{\partial \hat{r}} & \frac{\partial x^2}{\partial \hat{\theta}_1} & \cdots & \frac{\partial x^2}{\partial \hat{\theta}_{d-1}} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial x^d}{\partial \hat{r}} & \frac{\partial x^d}{\partial \hat{\theta}_1} & \cdots & \frac{\partial x^d}{\partial \hat{\theta}_{d-1}} \end{bmatrix}}_{j \in SO(d)} = \begin{bmatrix} \frac{\partial}{\partial \hat{r}} & \frac{\partial}{\partial \hat{\theta}^1} & \cdots & \frac{\partial}{\partial \hat{\theta}^{d-1}} \end{bmatrix} \quad (1)$$

Rescaling the hyperspherical basis vectors to have unit norm at all points causes the matrix J to become the orthogonal matrix with determinant equal to one Jˆ. This set of d × d matrices belongs to the group SO(d), which represents the set of d-dimensional rotations about the origin. Similarly, the backwards change-of-basis Jˆ−<sup>1</sup> = Jˆ<sup>T</sup> converts vectors in hyperspherical coordinates to Cartesian coordinates.

As a result, vectors from the tangent space at p in Cartesian coordinates simply rotate to convert to the normalized tangent space at p˜ in hyperspherical coordinates. Specifically, for a point p˜ = (r, θ1, θ2, ..., θd−1) and a vector v˜ = ˜c1r + ˜c2θ <sup>1</sup> + ... + ˜cdθ d−1 , converting v = c1x <sup>1</sup> + ... + cdx d from Cartesian to hyperspherical coordinates is the transformation:

$$\tilde{v} = \hat{J}^T v$$

where Jˆ operates on vector v—i.e. Jvˆ —by first rotating by angle c˜<sup>2</sup> in the x <sup>1</sup> − x <sup>2</sup> plane (i.e. the θ 1 axis of rotation), then by angle c˜<sup>3</sup> in the x <sup>2</sup> − x <sup>3</sup> plane (i.e. the θ 2 axis of rotation), so on and so forth until a final rotation by angle c˜<sup>d</sup> in the x <sup>d</sup>−<sup>1</sup> − x <sup>d</sup> plane (i.e. the θ d−1 axis of rotation). Composing these rotations together leads to a rotation from p˜<sup>0</sup> = (1, 0, 0, ..., 0) to p˜:

$$\hat{J} v = (R_{\tilde{p}_0 \rightarrow \tilde{p}}) v = (R_{\theta_d}^{x^{d-1}-x^d} \cdots R_{\theta_2}^{x^2-x^3} R_{\theta_1}^{x^1-x^2}) v$$

$$\hat{J}^{-1} v = \hat{J}^T v = (R_{\tilde{p}_0 \rightarrow \tilde{p}})^T v = (R_{\theta_d}^{x^{d-1}-x^d} \cdots R_{\theta_2}^{x^2-x^3} R_{\theta_1}^{x^1-x^2})^T v = R_{\tilde{p} \rightarrow \tilde{p}_0} v$$

where we define Ra˜→˜<sup>b</sup> to be the rotation from a˜ to ˜b as described above and R x <sup>i</sup>−x j θi to be the rotation by angle θ<sup>i</sup> in the x <sup>i</sup> −x <sup>j</sup> plane. Important for our later discussion on the rotation trick, this rotational characteristic causes moving a fixed vector along a curve in hyperspherical coordinates to rotate in Cartesian coordinates.

Remark 4. *Using the renormalized transformation in Equation* [\(1\)](#page-30-0)*, a constant vector field* v˜ *in hyperspherical coordinates corresponds to a rotated vector field in Cartesian coordinates.*

*Proof.* At Cartesian point p and corresponding hyperspherical point p˜:

$$v_p^T \begin{bmatrix} R_{\theta_d}^{x^{d-1}-x^d} & \cdots & R_{\theta_2}^{x^2-x^3} & R_{\theta_1}^{x^1-x^2} \end{bmatrix} = \tilde{v}_{\tilde{p}}^T \begin{bmatrix} R_{\theta_d}^{x^{d-1}-x^d} & \cdots & R_{\theta_2}^{x^2-x^3} & R_{\theta_1}^{x^1-x^2} \end{bmatrix}^T v_p = \tilde{v}_{\tilde{p}} \begin{bmatrix} R_{\tilde{p} \rightarrow \tilde{p} 0} \end{bmatrix} v_p = \tilde{v}_{\tilde{p}}$$

so a constant vector field v˜ in hyperspherical coordinates will correspond to a cartesian vector field where each vector at point p is rotated by the rotation that alights p˜ to p˜0.

Another important characteristic relates to the metric tensor with normalized hyperspherical basis vectors. We can explicitly compute the induced metric in hyperspherical coordiantes in terms of our renormalized basis vectors:

$$\begin{aligned} \hat{I} &= \begin{bmatrix} \frac{\partial}{\partial r} \cdot \frac{\partial}{\partial r} & 0 & 0 & \dots & 0 \\ 0 & \frac{\partial}{\partial \theta_1} \cdot \frac{\partial}{\partial \theta_1} & 0 & \dots & 0 \\ 0 & 0 & \frac{\partial}{\partial \theta_2} \cdot \frac{\partial}{\partial \theta_2} & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \dots & \frac{\partial}{\partial \theta_{d-1}} \cdot \frac{\partial}{\partial \theta_{d-1}} \end{bmatrix} \\ &= \begin{bmatrix} (\mathcal{I}_{11})^{-1} \frac{\partial}{\partial r} \cdot \frac{\partial}{\partial r} & 0 & 0 & \dots & 0 \\ 0 & (\mathcal{I}_{22})^{-1} \frac{\partial}{\partial \theta_1} \cdot \frac{\partial}{\partial \theta_1} & 0 & \dots & 0 \\ 0 & 0 & (\mathcal{I}_{33})^{-1} \frac{\partial}{\partial \theta_2} \cdot \frac{\partial}{\partial \theta_2} & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \dots & (\mathcal{I}_{dd})^{-1} \frac{\partial}{\partial \theta_{d-1}} \cdot \frac{\partial}{\partial \theta_{d-1}} \end{bmatrix} \\ &= \begin{bmatrix} (\mathcal{I}_{11})^{-1} (\mathcal{I}_{11}) & 0 & 0 & \dots & 0 \\ 0 & (\mathcal{I}_{22})^{-1} (\mathcal{I}_{22}) & 0 & \dots & 0 \\ 0 & 0 & (\mathcal{I}_{33})^{-1} (\mathcal{I}_{33}) & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \dots & (\mathcal{I}_{dd})^{-1} (\mathcal{I}_{dd}) \end{bmatrix} \\ &= \begin{bmatrix} 1 & 0 & 0 & \dots & 0 \\ 0 & 1 & 0 & \dots & 0 \\ 0 & 0 & 1 & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \dots & 1 \end{bmatrix} \end{aligned} \quad (2)$$

which yields the identity matrix. This is perhaps unsurprising: we normalize basis vectors so that ∂ ˆθ i · ∂ ∂ ˆθ <sup>i</sup> = 1.

Another way to view the renormalized tangent plane transformation is as a change-of-basis in the Cartesian coordiante system, with the basis vectors spanning each tangent space in Cartesian coordinates rotated to align with the directions of hyperspherical basis vectors. Rotating the tangent space at each point in Cartesian coordinates does not change the Euclidean metric tensor—R<sup>T</sup> IR = I—so it remains the identity. These two formulations are equivalent: renormalizing the basis vectors in the hyperspherical tangent space to have unit norm corresponds to rotating the basis vectors in the Cartesian coordinate system to align with the hyperspherical basis vectors at all points.

#### A.12.2 STE AS PARALLEL TRANSPORT

From the description of the STE in [Bengio et al.](#page-10-2) [\(2013\)](#page-10-2), a gradient vector ∇qL is transported from q to e during the backwards pass in such a way that its direction and magnitude is preserved. Critically, the curve along which ∇qL is transported is not specified; the effect is to simply "copy-and-paste" the vector from q to e.

To use the machinery of calculus, we assume that ∇qL is transported from q to e along any smooth curve γ(t) running from q to e. Along this curve, we define the transport of ∇qL at position γ(t) simply as ∇qL to emulate how the STE would move ∇qL from q to γ(t). Therefore, the direction and magnitude of ∇qL does not change along the curve γ(t). An example of this transport is visualized in Figure [20,](#page-32-0) and in Remark [5,](#page-31-0) we show this formulation is equivalent to the parallel transport of ∇qL along any curve γ(t) from q to e with the Levi-Civita connection.

Remark 5. *The Straight Through Estimator (STE) is equivalent to the parallel transport of* ∇qL *along any curve connecting* q *to* e *with the identity metric tensor in Cartesian coordinates using the Levi-Civita connection.*

*Proof.* A vector field v is parallel transported along a curve γ(t) if the covariant derivative of v in the direction of γ˙(t) is zero. Informally, the change in the vector field v must exactly match how the

![](_page_32_Figure_1.jpeg)

Figure 20: (top) Visualization of vector transport in Cartesian coordinates and renormalized hyperspherical coordinates along curves γ1(t), γ2(t) and γ3(t). Notice the hyperspherical basis changes from point to point. (bottom) Depiction of the transported vector in terms of the basis vectors <sup>∂</sup> ∂x and <sup>∂</sup> ∂y for Cartesian coordinates and <sup>∂</sup> ∂rˆ and <sup>∂</sup> ∂θˆ for hyperspherical coordinates. Notice how the components of <sup>∂</sup> ∂rˆ and <sup>∂</sup> ∂θˆ change for a constant vector field in the Cartesian tangent space.

basis vectors of the tangent plane change along γ(t) to remain "parallel" along the curve:

$$\underbrace{\nabla \dot{\gamma}(t) v = \vec{0}}_{\text{Parallel Transport Condition}}$$

Using the identity metric tensor:

$$g_{ij} = \delta_{ij} = \begin{cases} 0 & \text{if } i \neq j \\ 1 & \text{if } i = j \end{cases}$$

with the Levi-Civita connection will result in all zero Christoffel symbols:

$$\Gamma_{ij}^m = \frac{1}{2} g^{mk} \left( \frac{\partial g_{jk}}{\partial x^i} + \frac{\partial g_{ik}}{\partial x^j} - \frac{\partial g_{ij}}{\partial x^k} \right) = 0$$

where g mk is the m, k entry of inverse metric tensor. Computing the covariant derivative for a general curve γ(t):

$$\begin{aligned}\nabla \dot{\gamma}(t) v &= \nabla \dot{\gamma}_{1e_1} + \dot{\gamma}_{2e_2} + \dots + \dot{\gamma}_{de_d} v \\ &= \sum_{i=1}^d \dot{\gamma}_i \nabla e_i v \\ &= \sum_{i=1}^d \dot{\gamma}_i \frac{\partial}{\partial x^i}(v) \\ &= \underbrace{\sum_{i=1}^d \dot{\gamma}_i \frac{\partial}{\partial x^i}(v_1 e_1 + v_2 e_2 + \dots + v_d e_d)}_{\text{must be equal to 0 for parallel transport}}\end{aligned}$$

Considering the i th term in this summation:

$$\begin{aligned} 0 &= (\nabla_{\dot{\gamma}(t)} v)^i = \dot{\gamma}_i \frac{\partial}{\partial x^i} (v_1 e_1 + v_2 e_2 + \dots + v_d e_d) \\ &= \dot{\gamma}_i \frac{\partial}{\partial x^i} [v^k e_k] \\ &= \dot{\gamma}_i \left[ \frac{\partial v^k}{\partial x^i} e_k + v^k \frac{\partial e_k}{\partial x^i} \right] \\ &= \dot{\gamma}_i \left[ \frac{\partial v^k}{\partial x^i} e_k + v^k \Gamma_{ik}^m e_m \right] \\ &= \dot{\gamma}_i \left[ \frac{\partial v^k}{\partial x^i} e_k \right] \end{aligned}$$

For this equation to hold for an arbitrary γ(t), ∂v<sup>k</sup> ∂x<sup>i</sup> = 0 for 1 ≤ k, i ≤ d. Therefore, v<sup>k</sup> must be a constant, and vector fields along curves must be constant to satisfy the parallel transport criteria.

Pulling this back to the STE, holding ∇qL constant along the curve γ(t) from q to e results in a constant vector field along γ(t). The covariant derivative of this vector field is zero, and therefore the STE parallel transports ∇qL from q to e.

#### A.12.3 THE ROTATION TRICK AS PARALLEL TRANSPORT

In this section, we analyze the rotation trick through the lens of geometry. As in Appendix [A.12.2,](#page-31-1) we extend the rotation trick to any smooth curve γ(t) connecting q to e and define the transport of ∇qL at γ(t) as the rotation trick applied to move ∇qL from q to γ(t). This definition allows us to use the structure of calculus, without imposing any prohibitive restrictions on the path taken from q to e.

To build visual intuition, Figure [21](#page-34-0) illustrates how the rotation trick transforms an initial vector along three different curves γ1, γ2, γ<sup>3</sup> in both Cartesian coordinates and hyperspherical coordinates with normalized basis vectors. In Cartesian coordinates, the rotation trick changes the components of the basis vectors during transport to follow a rotation; however in normalized hyperspherical coordinates, the components of this vector during transport are constant because the basis vectors themselves rotate.

Remark 6. *The rotation trick is equivalent to the parallel transport of* ∇qL *along any curve connecting* q *to* e *with the induced metric in hyperspherical coordinates with the normalized transformation described in Equation* [\(1\)](#page-30-0) *using the Levi-Civita connection.*

*Proof.* From Equation [\(2\)](#page-31-2), the metric tensor in hyperspherical coordinates with normalized basis vectors—equivalently, the cartesian coordinate system with each tangent space rotated to align with the hyperspherical frame at every point—is the identity. Therefore, using the Levi-Civita connection

![](_page_34_Figure_1.jpeg)

Figure 21: (top) Visualization of vector transport in hyperspherical coordinates with normalized basis vectors and Cartesian coordinates along curves γ1(t), γ2(t) and γ3(t). The vectors along each curve in hyperspherical coordinates *rotate* to stay constant with respect to the natural rotation of the basis vectors. This same rotation in Cartesian coordinates yields a non-constant vector as the Cartesian basis vectors do not change from point to point. (bottom) Depiction of the transported vector in terms of the basis vectors <sup>∂</sup> ∂rˆ and <sup>∂</sup> ∂θˆ for hyperspherical coordinates and <sup>∂</sup> ∂x and <sup>∂</sup> ∂y for Cartesian coordinates. In the former case, the transported vector remains constant with respect to the normalized basis vectors, while in Cartesian coordinates, the components change along γ3(t).

leads to zero Christoffel symbols, and the parallel transport of a vector along any curve keeps the vector constant.

We define TpC as the tangent space of the Cartesian coordinate system at point p and Tp˜H as the tangent space of the hyperspherical coordinate system with normalized basis vectors at point p˜. It remains to show that for a vector ∇qL ∈ TqC and corresponding ∇q˜L ∈˜ Tq˜H, the transformation of ∇q˜L ∈˜ Te˜H to TeC will yield Rq→e∇qL where Rq→<sup>e</sup> is the rotation trick's transformation, i.e. the rotation that rotates q to e.

For a vector ∇q˜L˜ in hyperspherical coordinates at point q˜ = (1, θ1, θ2, ..., θd−1) and using the normalized change-of-basis in Equation [\(1\)](#page-30-0), the corresponding vector ∇qL in Cartesian coordinates is:

$$\begin{aligned}\nabla_q \mathcal{L}^T &= \nabla_{\tilde{q}} \tilde{\mathcal{L}}^T \left[ \hat{J}_{\tilde{q}}^{-1} \right] \\ \nabla_q \mathcal{L} &= \left[ \hat{J}_{\tilde{q}} \right] \nabla_{\tilde{q}} \tilde{\mathcal{L}} \\ &= [R_{\tilde{p}_0 \rightarrow \tilde{q}}] \nabla_{\tilde{q}} \tilde{\mathcal{L}} \\ &= [R_{\theta_{d-1}} R_{\theta_{d-2}} \cdots R_{\theta_1}] \nabla_{\tilde{q}} \tilde{\mathcal{L}}\end{aligned}$$

and the corresponding vector ∇q˜L˜ at point e˜ is:

$$\begin{aligned}\nabla_e \mathcal{L}^T &= \nabla_{\tilde{q}} \tilde{\mathcal{L}}^T \left[ \hat{J}_e^{-1} \right] \\ \nabla_e \mathcal{L} &= \left[ \hat{J}_e \right] \nabla_{\tilde{q}} \tilde{\mathcal{L}} \\ &= [R_{\tilde{p}_0 \rightarrow \tilde{e}}] \nabla_{\tilde{q}} \tilde{\mathcal{L}} \\ &= [R_{\tilde{q} \rightarrow \tilde{e}} R_{\tilde{p}_0 \rightarrow \tilde{q}}] \nabla_{\tilde{q}} \tilde{\mathcal{L}} \\ &= R_{\tilde{q} \rightarrow \tilde{e}} \left[ R_{\tilde{p}_0 \rightarrow \tilde{q}} \nabla_{\tilde{q}} \tilde{\mathcal{L}} \right] \\ &= [R_{\tilde{q} \rightarrow \tilde{e}}] \nabla_{\tilde{q}} \mathcal{L}\end{aligned}$$

which is exactly how the rotation trick transforms the vector. Informally, "copy-and-pasting" the vector ∇q˜L˜ from q˜ to e˜ in hyperspherical coordinates with normalized basis vectors corresponds to rotating ∇qL by the rotation that aligns q to e in Cartesian coordinates.

In summary, we consider a geometry where the tangent space is spanned by unit norm basis vectors <sup>∂</sup> ∂rˆ , ∂ ∂θˆ<sup>1</sup> , ..., ∂ ∂θˆd−<sup>1</sup> that match the direction of the typical hyperspherical basis vectors ∂ ∂r , ∂ ∂θ<sup>1</sup> , ..., ∂ ∂θˆd−<sup>1</sup> . The induced metric tensor is the identity, so the parallel transport of a vector along any curve holds its components constant. Converting a vector ∇qL to this tangent space via the normalized transformation in Equation [\(1\)](#page-30-0), parallel transporting the resulting vector from q˜ to e˜, and then converting it back to Cartesian coordinates corresponds exactly to the rotation trick's transformation.

This is a remarkably simple result; the rotation trick and the STE can be viewed as the same operation. Both parallel transport the gradient ∇qL from q to e in a path-independent manner with the Euclidean metric. The only difference is the coordinate system where parallel transport occurs. The STE employs the Cartesian coordinate system while the rotation trick uses the hyperspherical coordinate system with normalized basis vectors.