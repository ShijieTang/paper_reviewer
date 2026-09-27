# Emergence in Non-Neural Models: Grokking Modular Arithmetic via Average Gradient Outer Product

Neil Mallinar 1 2 Daniel Beaglehole <sup>1</sup> Libin Zhu <sup>1</sup> Adityanarayanan Radhakrishnan <sup>2</sup> Parthe Pandit <sup>3</sup> Mikhail Belkin <sup>4</sup>

## Abstract

Neural networks trained to solve modular arithmetic tasks exhibit *grokking*, the phenomenon where the test accuracy improves only long after the model achieves 100% training accuracy in the training process. It is often taken as an example of "emergence", where model ability manifests sharply through a phase transition. In this work, we show that the phenomenon of grokking is not specific to neural networks nor to gradient descent-based optimization. Specifically, we show that grokking occurs when learning modular arithmetic with Recursive Feature Machines (RFM), an iterative algorithm that uses the Average Gradient Outer Product (AGOP) to enable task-specific feature learning with kernel machines. We show that RFM and, furthermore, neural networks that solve modular arithmetic learn block-circulant features transformations which implement the previously proposed Fourier multiplication algorithm.

### 1. Introduction

In recent years the idea of "emergence" has become an important narrative in machine learning. While there is no broad agreement on the definition [\(Rogers & Luccioni,](#page-10-0) [2023\)](#page-10-0), it is often argued that "skills" emerge during the training process once certain data size, compute, or model size thresholds are achieved [\(Wei et al.,](#page-11-0) [2022;](#page-11-0) [Arora &](#page-9-0) [Goyal,](#page-9-0) [2023\)](#page-9-0). Furthermore, these skills are believed to appear rapidly, exhibiting sharp and seemingly unpredictable improvements in performance at these thresholds. One of

![](_page_0_Figure_3.jpeg)

Figure 1. Recursive Feature Machines grok the modular arithmetic task f ∗ (x, y) = (x + y) mod 59.

the simplest, most striking examples supporting this idea is "grokking" modular arithmetic [\(Power et al.,](#page-10-1) [2022;](#page-10-1) [Nanda](#page-10-2) [et al.,](#page-10-2) [2023\)](#page-10-2). A neural network trained to predict modular arithmetic operations on a fixed data set rapidly transitions from near-zero to perfect (100%) test accuracy at a certain point in the optimization process. Surprisingly, this transition point occurs long after perfect *training accuracy* is achieved. Not only is this contradictory to traditional wisdom regarding overfitting but, as we will show, some aspects of grokking do not fit neatly with our modern understanding of "benign overfitting" [\(Bartlett et al.,](#page-9-1) [2021;](#page-9-1) [Belkin,](#page-9-2) [2021\)](#page-9-2).

Despite a large amount of recent work on emergence and, specifically, grokking, (see, e.g., [\(Power et al.,](#page-10-1) [2022;](#page-10-1) [Liu](#page-10-3) [et al.,](#page-10-3) [2023;](#page-10-3) [Nanda et al.,](#page-10-2) [2023;](#page-10-2) [Thilak et al.,](#page-10-4) [2022;](#page-10-4) [Furuta](#page-9-3) [et al.,](#page-9-3) [2024;](#page-9-3) [Miller et al.,](#page-10-5) [2024\)](#page-10-5)), the nature and existence of the emergent phenomena remains contested. The recent paper [\(Schaeffer et al.,](#page-10-6) [2023\)](#page-10-6) suggests that the rapid emergence of skills may be a "mirage" due to the mismatch between the discontinuous metrics used for evaluation, such as accuracy, and the continuous loss used in training. The authors argue that, in contrast to accuracy, the test (or validation) loss or some other suitably chosen metric may decrease gradually

<sup>1</sup>Department of Computer Science and Engineering, UC San Diego, La Jolla, CA, USA <sup>2</sup>The Broad Institute of MIT and Harvard, Cambridge, MA, USA <sup>3</sup> IIT Bombay, Mumbai, Maharashtra, India <sup>4</sup>Halıcıoglu Data Science Institute, UC San Diego, ˘ La Jolla, CA, USA. Correspondence to: Neil Mallinar <nmallina@ucsd.edu>.

*Proceedings of the* 42 nd *International Conference on Machine Learning*, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

![](_page_1_Figure_1.jpeg)

RFM Circ: frob is not predicted by training or test loss[<sup>1</sup>](#page-0-0) , let alone accuracy. Specifically, we show grokking for Recursive Feature Machines (RFM) [\(Radhakrishnan et al.,](#page-10-8) [2024a\)](#page-10-8), an algorithm that iteratively uses the Average Gradient Outer Product (AGOP) to enable task-specific feature learning in general machine learning models. In this work, we use RFM to enable feature learning in kernel machines, which are a class of predictors with no native mechanism for feature learning. In this setting, RFM iterates between three steps: (i) training a kernel machine, f, to fit training data; (ii) computing the AGOP matrix of f, M, over the training data to extract task-relevant features; and (iii) transforming input data, x, using the learned features via the map x → Ms/<sup>2</sup>x for a matrix power s > 0 (see Section [2](#page-2-0) for details).

Figure 2. RFM with the quadratic kernel on modular arithmetic with modulus p = 61 trained for 30 iterations. (A) Test accuracy, test loss (mean squared error) over all output coordinates, and test loss of the correct class output coordinate do not change in the first 8 iterations and then, sharply transition. (B) Circulant deviation and AGOP alignment show gradual progress towards generalizing solutions despite accuracy and loss metrics not changing in the initial iterations. For division (Div), circulant deviation is measured with respect to the feature sub-matrices after reordering by the discrete logarithm. Plots for subtraction (Sub) and multiplication (Mul) are in Appendix Figure [1.](#page-18-0)

throughout training and thus provide a useful measure of progress. Another possible progress measure is the training loss. As SGD-type optimization algorithms generally result in a gradual decrease of the training loss, one may posit that skills appear once the training loss falls below a certain threshold in the optimization process. Indeed, such a conjecture is in the spirit of classical generalization theory, which considers the training loss to be a useful proxy for the test performance [\(Mohri et al.,](#page-10-7) [2018\)](#page-10-7).

In this work, we show that sharp emergence in modular arithmetic arises entirely from feature learning, independently of other aspects of modeling and training, and is not predicted by standard measures of progress. We then clarify the nature of feature learning leading to the emergence of skills in modular arithmetic. We discuss these contributions in further detail below.

Summary of the contributions. We demonstrate empirically that grokking modular arithmetic: (1) is not specific to neural networks (to the best of our knowledge, no prior work shows a non-neural model that learns modular arithmetic); (2) is not tied to gradient-based optimization methods; (3)

In Fig. [1](#page-0-1) we give an example of RFM grokking modular addition, despite not using any gradient-based optimization methods and achieving numerically zero training loss at every iteration. During the first few iterations both the test loss and and test accuracy remain at the constant (random) level. Around iteration 10 the test loss starts improving and, a few iterations later, test accuracy quickly transitions to 100%. We also observe that early in the iteration, structure emerges in AGOP feature matrices (see Fig. [1\)](#page-0-1). The gradual appearance of structure in these feature matrices is striking given that the training loss is identically zero at every iteration and the test loss does not significantly change until iteration 8. The striped patterns observed in feature matrices correspond to matrices whose sub-blocks are circulant with entries that are constant along the "long" diagonals which wrap around the matrix.[<sup>2</sup>](#page-0-0) Such *circulant feature matrices* are key to learning modular arithmetic. In Section [3](#page-3-0) we demonstrate that standard kernel machines using *random* circulant features easily learn modular operations. As these random circulant matrices are generic, we argue that no additional structure is required to solve modular arithmetic.

To demonstrate that the feature matrices evolve toward this structure (including for multiplication and division under an appropriate re-ordering of the input coordinates), we introduce two "hidden progress measures" [\(Barak et al.,](#page-9-4) [2022\)](#page-9-4): (1) *Circulant deviation*, which measures constancy of the diagonals of a matrix, and (2) *AGOP alignment*, which measures similarity between the feature matrix at iteration t and the AGOP of a fully trained model. We will see that both of these measures show gradual (initially nearly linear) progress toward a model that generalizes.

We further argue that emergence in fully connected neural networks trained on modular arithmetic identified in prior

<sup>1</sup>We note that for neural networks trained by SGD, emergence cannot be decoupled from training loss, as non-zero loss is required for training to occur at all.

<sup>2</sup> Feature sub-matrices may also be constant on anti-diagonals. We also refer to these matrices as circulant.

work [\(Gromov,](#page-9-5) [2023;](#page-9-5) [Liu et al.,](#page-9-6) [2022\)](#page-9-6) is analogous to that for RFM and is exhibited through the AGOP (see Section [4\)](#page-5-0). By visualizing covariances of network weights, we observe that these models also learn block-circulant features for modular arithmetic. We demonstrate that these features are highly correlated with the AGOP of neural networks, corroborating prior observations from [\(Radhakrishnan et al.,](#page-10-8) [2024a\)](#page-10-8). Paralleling our observations for RFM, our progress measures indicate gradual progress toward a generalizing solution during neural network training. Finally we demonstrate that training neural networks on data transformed by random block-circulant matrices dramatically decreases training time needed to learn modular arithmetic.

Why are these learned block circulant features effective for modular arithmetic? We provide supporting theoretical evidence that circulant features result in kernel machines implementing the Fourier Multiplication Algorithm (FMA) for modular arithmetic (see Section [5\)](#page-6-0). For the case of neural networks, several prior works have argued empirically and theoretically that neural networks learn to implement the FMA to solve modular arithmetic [\(Nanda et al.,](#page-10-2) [2023;](#page-10-2) [Varma et al.,](#page-11-1) [2023;](#page-11-1) [Morwani et al.,](#page-10-9) [2024\)](#page-10-9). While kernel RFM and neural networks utilize different classes of predictive models, our results suggest that they discover similar algorithms for implementing modular arithmetic.

By decoupling feature learning from predictor training, our results provide evidence for emergent properties of machine learning models arising purely as a consequence of their ability to learn features. We hope our work will help isolate the underlying mechanisms of emergence and shed light on the key practical concern of how, when, and why these seemingly unpredictable transitions occur.

#### 2. Preliminaries

Learning modular arithmetic. Let <sup>Z</sup><sup>p</sup> = <sup>Z</sup>/p<sup>Z</sup> denote the field of integers modulo a prime p and let Z ∗ <sup>p</sup> = <sup>Z</sup>p\{0}. We learn modular functions f ∗ (a, b) = g(a, b) mod p where f ∗ : <sup>Z</sup><sup>p</sup> × <sup>Z</sup><sup>p</sup> → <sup>Z</sup>p, a, b ∈ <sup>Z</sup>p, and g : <sup>Z</sup> × <sup>Z</sup> → <sup>Z</sup> is an arithmetic operation on a and b, e.g. g(a, b) = a + b. Note that there are p <sup>2</sup> discrete input pairs (a, b) for all modular operations except for f ∗ (a, b) = (a ÷ b) mod p, which has p(p − 1) inputs as the denominator cannot be 0.

To train models on modular arithmetic tasks, we construct input-label pairs by one-hot encoding the input and label integers. Specifically, for every pair a, b ∈ <sup>Z</sup>p, we write the input as e<sup>a</sup> ⊕ e<sup>b</sup> ∈ <sup>R</sup> 2p and the output as e<sup>f</sup> <sup>∗</sup>(a,b) ∈ R p , where e<sup>i</sup> ∈ <sup>R</sup> p is the i-th standard basis vector in p dimensions and ⊕ is concatenation. The training dataset consists of a random subset of n = r × N input/label pairs, where r is the *training fraction* and N = p <sup>2</sup> or p(p − 1) is the number of possible discrete inputs.

Complex inner product and Discrete Fourier Transform. In our theoretical analysis in Section [5,](#page-6-0) we will utilize the following notions of complex inner product and DFT. The complex inner product ⟨·, ·⟩<sup>C</sup> is a map from C <sup>d</sup> × <sup>C</sup> <sup>d</sup> → <sup>C</sup> of the form

$$\langle u, v \rangle_{\mathbb{C}} = u^{\top} \bar{v} , \quad (1)$$

where v¯<sup>j</sup> is the complex conjugate of v<sup>j</sup> . Let i = √ −1 and let ω = exp( <sup>−</sup>2πi d ). The DFT is the map F : C <sup>d</sup> → <sup>C</sup> <sup>d</sup> of the form F(u) = F u, where F ∈ C d×d is a unitary matrix with Fij = √ d ω ij . In matrix form, F is given as

$$F = \frac{1}{\sqrt{d}} \begin{pmatrix} 1 & 1 & 1 & \cdots & 1 \\ 1 & \omega & \omega^2 & \cdots & \omega^{d-1} \\ 1 & \omega^2 & \omega^4 & \cdots & \omega^{2(d-1)} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & \omega^{d-1} & \omega^{2(d-1)} & \cdots & \omega^{(d-1)(d-1)} \end{pmatrix}. \quad (2)$$

Circulant matrices. The features that RFMs and neural networks learn in order to solve modular arithmetic contain blocks of *circulant matrices*, which are defined as follows. Let σ : R <sup>p</sup> → <sup>R</sup> <sup>p</sup> be the cyclic permutation which acts on a vector u ∈ R <sup>p</sup> by shifting its coordinates by one cell to the right: [σ(u)]<sup>j</sup> = uj−1 mod <sup>p</sup> , for j ∈ [p]. We write the ℓ-fold composition of this map σ ℓ (u) ∈ <sup>R</sup> <sup>p</sup> with entries [σ ℓ (u)]<sup>j</sup> = uj−<sup>ℓ</sup> mod <sup>p</sup>. A circulant matrix C ∈ <sup>R</sup> p×p is determined by a vector c = [c0, . . . , cp−1] ∈ <sup>R</sup> p , and has rows (in order from first to last): c, σ(c), . . . , σ<sup>p</sup>−<sup>1</sup> (c). Feature matrices may also have have constant anti-diagonals ("Hankel matrices"). To ease terminology, we will use the word circulant to refer to both Hankel and circulant matrices.

Average Gradient Outer Product (AGOP). The AGOP matrix is central to our discussion and defined as follows.

Definition 2.1 (AGOP). *Given a predictor* f : R <sup>d</sup> → <sup>R</sup> c *with* <sup>c</sup> *outputs,* f(x) ≡ [f0(x), . . . , fc−1(x)]*, let* ∂f(<sup>x</sup> ′ ) ∂x ∈ R <sup>d</sup>×<sup>c</sup> *be the Jacobian (transposed) of* f *evaluated at some point* x ′ ∈ <sup>R</sup> <sup>d</sup> *with entries* [ ∂f(x ) ∂x ]s,ℓ = ∂fℓ(x ) ∂x<sup>s</sup> *. Then, for* f *trained on a set of data points* {x (j)} n <sup>j</sup>=1*, with* x (j) ∈ <sup>R</sup> d *, the Average Gradient Outer Product (AGOP),* G ∈ R d×d *, is defined as,*

$$G(f; \{x^{(j)}\}_{j=1}^n) = \frac{1}{n} \sum_{j=1}^n \frac{\partial f(x^{(j)})}{\partial x} \frac{\partial f(x^{(j)})}{\partial x}^\top. \quad (3)$$

For simplicity, we omit the dependence on the dataset in the notation. Top eigenvectors of AGOP can be viewed as the "most relevant" input features, those input directions that influence the output of a general predictor (for example, a kernel machines or a neural network) the most. As a consequence, the AGOP can be viewed as a task-specific

transformation that can be used to amplify relevant features and improve sample efficiency of machine learning models.

Indeed, a line of prior works [\(Yuan et al.,](#page-11-2) [2023;](#page-11-2) [Trivedi](#page-10-10) [et al.,](#page-10-10) [2014;](#page-10-10) [Hristache et al.,](#page-9-7) [2001\)](#page-9-7) have used the AGOP to improve the sample efficiency of predictors trained on multi-index models, a class of predictive tasks in which the target function depends on a low-rank subspace of the data. Though the study of AGOP has been motivated by these multi-index examples, we will see that the AGOP can be used to recover useful features for modular arithmetic that are, in fact, not low-rank.

AGOP and feature learning in neural networks. [\(Rad](#page-10-8)[hakrishnan et al.,](#page-10-8) [2024a\)](#page-10-8) posited that AGOP was a mechanism through which neural networks learn features. The authors introduce the *Neural Feature Ansatz (NFA)* stating that for any layer ℓ of a trained neural network with weights Wℓ, the *Neural Feature Matrix (NFM)*, W<sup>T</sup> <sup>ℓ</sup> Wℓ, are highly correlated to the AGOP of the model computed with respect to the input of layer ℓ. The NFA suggests that neural networks learn features at each layer by utilizing the AGOP. For more details on the NFA, see Appendix [C.](#page-12-0)

Recursive Feature Machine (RFM). Importantly, AGOP can be computed for any differentiable predictor, including those such as kernel machines that have no native feature learning mechanism. As such, the authors of [\(Rad](#page-10-8)[hakrishnan et al.,](#page-10-8) [2024a\)](#page-10-8) developed an algorithm known as RFM, which iteratively uses the AGOP to extract features. Below, we present the RFM algorithm used in conjunction with kernel machines. Suppose we are given data samples (X, y) ∈ <sup>R</sup> <sup>n</sup>×<sup>d</sup> × <sup>R</sup> <sup>n</sup> where X contains n samples denoted {x (j)} n <sup>j</sup>=1. Given an initial symmetric positive-definite matrix M<sup>0</sup> ∈ <sup>R</sup> d×d , and Mahalanobis kernel k(·, · ; M) : <sup>R</sup> <sup>d</sup> × <sup>R</sup> <sup>d</sup> → <sup>R</sup>, RFM iterates the following steps for t ∈ [T]:

*Step 1 (Predictor training):* 
$$f^{(t)}(x) = k(x, X; M_t)\alpha$$
 (4)

with 
$$\alpha = k(X, X; M_t)^{-1}y$$
; (5)

*Step 2 (AGOP update):* 
$$M_{t+1} = [G(f^{(t)})]^s$$
 ; (6)

where s > 0 is a matrix power and k(X, X; M) ∈ R <sup>n</sup>×<sup>n</sup> denotes the matrix with entries [k(X, X; M)]<sup>j</sup>1j<sup>2</sup> = k(x (j1) , x(j2) ; M) for j1, j<sup>2</sup> ∈ [n]. In this work, we select s = 1 2 for all experiments (see Algorithm [1](#page-12-1) for complete pseudocode). We use the following two Mahalanobis kernels: (1) the quadratic kernel, k(x, x′ ; M) = x <sup>⊤</sup>Mx′ 2 ; and (2) the Gaussian kernel k(x, x′ ; M) = exp −∥x − x ′∥ 2 M/L , where for z ∈ R d , ∥z∥ <sup>M</sup> = z <sup>⊤</sup>Mz, and L is the bandwidth.

## 3. Emergence with RFM

We now show that RFM exhibits sharp transitions in performance on modular arithmetic tasks (addition, subtraction,

![](_page_3_Figure_1.jpeg)

Figure 3. RFM with the quadratic kernel for modular arithmetic with p = 61. (A) The square root of the kernel AGOPs for addition (Add), subtraction (Sub) visualized without their diagonals to emphasize the off-diagonal blocks. (B) Square root of the kernel AGOP for multiplication (Mul), division (Div). (C) For Mul and Div, rows and columns of each sub-matrix is re-ordered by the discrete log. base 2.

multiplication, and division) due to the emergence of blockcirculant features.

We will use a modulus of p = 61 and train RFM with quadratic and Gaussian kernel machines (experimental details are provided in Appendix [D\)](#page-13-0). As we solve kernel ridgeless regression exactly, all iterations of RFM result in zero training loss and 100% training accuracy. The top two rows of Fig. [2A](#page-1-0) show that the first several iterations of RFM result in near-zero test accuracy and approximately constant, large test loss. Despite these standard progress measures initially not changing, continuing to iterate RFM leads to a dramatic, sharp increase to 100% test accuracy and a corresponding decrease in the test loss later in the iteration process.

Sharp transition in loss of correct output coordinate. It is important to note that our total loss function is the square loss averaged over p = 61 classes. It is thus plausible that, due to averaging, the near-constancy of the total square loss over the first few iterations conceals steady improvements in the predictions of the correct class. However, in Fig. [2A](#page-1-0) we show that the test loss for the output coordinate of the correct class closely tracks the total test loss.

Emergence of block-circulant features in RFM. To understand RFM generalization, we visualize the 2p × 2p feature matrix given by the square root of the AGOP from the final iteration of RFM. We first visualize the feature matrices for RFM trained on modular addition/subtraction in Fig. [3A](#page-3-1). Their visually-evident striped structure suggests a more precise characterization:

Observation 1 (Block-circulant features). *Feature matrix* M<sup>∗</sup> ∈ <sup>R</sup> <sup>2</sup>p×2<sup>p</sup> *at the final iteration of RFM on modular addition/subtraction is of the form*

$$M^* = \begin{pmatrix} A & C^\top \\ C & A \end{pmatrix}, \quad (7)$$

*where* A, C ∈ R p×p *,* C *is an asymmetric circulant matrix. ,* A = c1I + c211<sup>⊤</sup> *for scalars* c1, c2*.*

Similarly to addition and subtraction, RFM successfully learns multiplication and division. Yet, in contrast to addition and subtraction, the structure of feature matrices for these tasks, shown in Fig. [3B](#page-3-1), is not at all obvious. Nevertheless, re-ordering the rows and columns of the feature matrices for these tasks brings out their hidden circulant structure of the form stated in Eq. [\(7\).](#page-4-0) We show the effect of re-ordering in Fig. [3C](#page-3-1) (see also Appendix Fig. [2](#page-18-1) for the evolution of re-ordered and original features during training).

We briefly discuss the reordering procedure below and provide further details in Appendix [E.](#page-13-1) To reorder, we use the fact of group theory that the multiplicative group Z ∗ p is a cyclic group of order p − 1 (e.g., [\(Koblitz,](#page-9-8) [1994\)](#page-9-8)). By definition of the cyclic group, there exists at least one element g ∈ Z ∗ p , known as a *generator*, such that Z ∗ <sup>p</sup> = {g i ; i ∈ {1, . . . , p − 1}}. As we will see, reordering the rows and columns of the AGOP by powers of a generator reveals circulant structure. For modular multiplication/division, the map taking g i to i is known as the *discrete logarithm* base g [\(Koblitz,](#page-9-8) [1994,](#page-9-8) Ch.3). It is natural to expect block-circulant feature matrices to arise in modular multiplication/division after reordering by the discrete log as the discrete log converts modular multiplication/division into modular addition/subtraction. We note the recent work [\(Doshi et al.,](#page-9-9) [2024\)](#page-9-9) also used the discrete log to reorder coordinates in the context of constructing a solution for solving modular multiplication with neural networks.

Progress measures. We propose two measures of feature learning, *circulant deviation* and *AGOP alignment*.

*Circulant deviation.* As the final feature matrices contain circulant sub-blocks, a natural progress measure for learning modular arithmetic with RFM is how far AGOP feature matrices are from a block-circulant matrix. For a feature matrix M, let A denote the bottom-left sub-block of M. We define circulant deviation as the total variance of the (wrapped) diagonals of A normalized by the norm ∥A∥ 2 F . In particular, let S ∈ R <sup>p</sup>×<sup>p</sup> → <sup>R</sup> <sup>p</sup>×<sup>p</sup> denote the shift operator, which shifts the ℓ-th row of the matrix by ℓ positions to the right. Also let Var(v) = P<sup>p</sup>−<sup>1</sup> <sup>j</sup>=0 (v<sup>j</sup> − <sup>E</sup>v) <sup>2</sup> be the variance of a vector v. If A[j] denotes the j-th column of A, we define circulant deviation D as: D(A) = <sup>1</sup> ∥A∥ F P<sup>p</sup>−<sup>1</sup> <sup>j</sup>=0 Var(S(A)[j]). As circulant matrices are constant along their (wrapped) diagonals, they have a circulant deviation of 0.

![](_page_4_Figure_1.jpeg)

Figure 4. Random circulant features generalize with standard kernels for modular arithmetic. RFM with the Gaussian kernel on addition (Add) and multiplication (Mul) for modulus p = 61 is compared to a base Gaussian kernel machine trained on random circulant features (for Mul, the sub-blocks are circulant after reordering by the discrete logarithm base 2).

We see in Fig. [2B](#page-1-0) that circulant deviation exhibits gradual improvement through the course of training with RFM. We find that for the first 10 iterations, while the training loss is numerically zero and the test loss does not improve, circulant deviation exhibits gradual, nearly linear, improvement. The improvements in circulant deviation reflect visual improvements in features, as was also shown in Fig. [1.](#page-0-1) These curves provide further support for Observation [1,](#page-4-0) as circulant deviation is close to 0 at the end of training.

Circulant deviation depends crucially on the observation that for modular arithmetic the feature matrices contained circulant blocks. For more general tasks, we may not be able to identify such structure. Thus, we propose a second, more general progress measure, AGOP alignment.

*AGOP alignment.* Given two matrices A, B ∈ R d×d , let ρ(A, B) denote the standard cosine similarity between these two matrices when vectorized. Specifically, let A, ˜ B˜ ∈ R d 2 denote the vectorization of A and B respectively, then <sup>ρ</sup>(A, B) = ⟨A, ˜ <sup>B</sup>˜⟩ ∥A˜∥ ∥B˜∥ .

If M<sup>t</sup> denotes the AGOP at iteration t of RFM (or epoch t of a neural network) and M<sup>∗</sup> denotes the final AGOP of the trained RFM (or neural network), then AGOP alignment at iteration t is given by ρ(Mt, M<sup>∗</sup> ). The same measure of alignment was used in [\(Zhu et al.,](#page-11-3) [2024\)](#page-11-3), except their alignment was computed with respect to the AGOP of the ground truth model. Note that as modular operations are discrete, in our setting there is no unique ground truth model for which AGOP can be computed.

Like circulant deviation, AGOP alignment exhibits gradual improvement in the regime that test loss is constant and large (see Fig. [2B](#page-1-0), bottom row). Moreover, AGOP alignment is a more general progress measure since it does not require assumptions on the structure of the AGOP. For instance, AGOP alignment can be measured without reordering for modular multiplication/division. While AGOP alignment does not require a specific form of the final features, it is still an *a posteriori* measurement of progress as it requires access to the features of a fully trained model.

Random circulant features allow standard kernels to generalize. We conclude this section by providing further evidence that the form of feature matrices given in Observation [1](#page-4-0) is key to enabling generalization in kernel machines trained to solve modular arithmetic tasks. We now show that a transformation with a *generic* block-circulant matrix enables kernels machines to learn modular arithmetic. We generate a random circulant matrix C by first sampling entries of the first column i.i.d. from the uniform distribution on [0, 1] ⊂ <sup>R</sup> and then shifting the column to generate the remaining columns of C. We construct M<sup>∗</sup> in Observation [1](#page-4-0) with c<sup>1</sup> = 1, c<sup>2</sup> = −1/p. For modular addition, we transform the input data by mapping xab = e<sup>a</sup> ⊕ e<sup>b</sup> to x˜ab = (M<sup>∗</sup> ) 1 <sup>4</sup> xab , and then train on the new data pairs (˜xab, ea+<sup>b</sup> mod <sup>p</sup>) for a subset of all possible pairs (a, b) ∈ <sup>Z</sup> 2 p . Note that transforming data with (M<sup>∗</sup> ) 1 4 is akin to using s = 1/2 in RFM.

We do the same for modular multiplication after reordering the random circulant by the discrete logarithm as described above. The experiments in Fig. [4](#page-4-1) show that standard kernel machines trained on feature matrices with random circulant blocks outperform RFM that learns such features through AGOP. We also find that directly enforcing circulant blocks in the sub-matrices of M<sup>t</sup> throughout RFM iterations accelerates grokking and improves test loss (see Appendix [F,](#page-13-2) Appendix Fig. [3\)](#page-19-0). These experiments provide direct evidence that the structure in Observation [1](#page-4-0) is key for generalization on modular arithmetic and, furthermore, *no additional structure* beyond a generic circulant is required.

### 4. Emergence in Neural Nets through AGOP

We now show that grokking in two-layer neural networks relies on the same principles as grokking by RFM. Specifically we demonstrate that (1) block-circulant features are key to neural networks grokking modular arithmetic; and (2) our measures (circulant deviation and AGOP alignment) indicate gradual progress towards generalization, while standard measures of generalization exhibit sharp transitions. All experimental details are provided in Appendix [D.](#page-13-0)

Grokking with neural networks. We first reproduce grokking with modular arithmetic using fully-connected networks as identified in prior works (Fig. [5A](#page-5-1)) [\(Gromov,](#page-9-5) [2023\)](#page-9-5). In particular, we train one hidden layer fully connected networks f : R <sup>2</sup><sup>p</sup> → <sup>R</sup> <sup>p</sup> of the form f(x) = W2ϕ(W1x) with quadratic activation ϕ(z) = z <sup>2</sup> on modulus p = 61 data

![](_page_5_Figure_1.jpeg)

Figure 5. One hidden layer fully-connected networks with quadratic activations trained on modular arithmetic with p = 61 trained for 50 epochs with the square loss. (A) Test accuracy, test loss over all outputs, and test loss of the correct class output do not change in the initial iterations. (B) Progress measures for circulant deviation and AGOP alignment. Circulant deviation for Div is computed after reordering by the discrete logarithm base 2. Plots for Sub and Mul can be found in Appendix Figure [5.](#page-20-0)

with a training fraction 50%.

Consistent with prior work [\(Gromov,](#page-9-5) [2023\)](#page-9-5) and analogously to RFMs, neural networks exhibit an initial training period where the train accuracy reaches 100%, while test accuracy is at 0% and test loss does not improve (see Fig. [5A](#page-5-1)). After this point, we see that the accuracy rapidly improves to achieve perfect generalization. We further verify that the sharp transition in test loss is not an artifact of averaging the loss over all output coordinates. In the third row of Fig. [5A](#page-5-1) we show that the test loss of the individual correct output coordinate closely tracks the total loss.

Emergence of block-circulant features in neural networks. To understand the features learned by neural networks we visualize the first layer Neural Feature Matrix.

Definition 4.1. *(Neural Feature Matrix) Given a fully connected network* f(x) = a <sup>⊤</sup>ϕ(W1x)*, the first layer Neural Feature Matrix (NFM) is the matrix* W<sup>⊤</sup> <sup>1</sup> W<sup>1</sup> ∈ <sup>R</sup> 2p×2p *.*

The NFM is the un-centered covariance of network weights and has been used in prior work in order to understand the features learned by various neural network architectures at any layer [\(Radhakrishnan et al.,](#page-10-8) [2024a;](#page-10-8) [Trockman et al.,](#page-10-11) [2022\)](#page-10-11). Fig. [6A](#page-6-1) displays the NFM for one hidden layer neural networks with quadratic activations trained on modular

![](_page_6_Figure_1.jpeg)

Figure 6. Feature matrices from one hidden layer neural networks with quadratic activations trained on addition and division modulo 61. The Pearson correlations between the NFM and square root of the AGOP for each task are 0.955 (Add), 0.929 (Div). Div is shown after reordering by the discrete logarithm base 2. Plots for Sub and Mul can be found in Appendix Figure [7.](#page-21-0)

arithmetic tasks. For addition/subtraction, we find that the NFM exhibits block circulant structure, akin to the feature matrix for RFM. As described in Section [3](#page-3-0) and Appendix [E,](#page-13-1) we reorder the NFM for networks trained on multiplication/division with respect to a generator for Z ∗ p in order to observe block-circulant structure (see Appendix Fig. [6A](#page-20-1) for a comparison of multiplication/division NFMs before and after reordering). The block-circulant structure in both the NFM and the feature matrix of RFM suggests that the two models are learning similar sets of features.

The work [\(Radhakrishnan et al.,](#page-10-8) [2024a\)](#page-10-8) posited that AGOP is the mechanism through which neural networks learn features. The authors stated their claim in the form of the Neural Feature Ansatz (NFA), which states that NFMs are proportional to a matrix power of AGOP through training (see Eq. [\(15\)](#page-12-2) for a restatement of the NFA). As such, we additionally compute the square root of the AGOP to examine the features learned by neural networks trained on modular arithmetic tasks. We visualize the square root of the AGOPs of these trained models in Fig. [6B](#page-6-1) and also find that the square root of the AGOP and the NFM are highly correlated (greater than 0.92), where Pearson correlation is equal to cosine similarity after centering the inputs to be mean 0. Moreover, we find that the square root of AGOP of neural networks again exhibits the same structure as stated in Observation [1](#page-4-0) (see Appendix Fig. [6B](#page-20-1) for a comparison of multiplication/division AGOPs before and after reordering).

Random circulant maps improve generalization of neural networks. To further establish the importance and generality of block-circulant features, we demonstrate that training networks on inputs transformed with a random blockcirculant matrix greatly accelerates learning. In Fig. [7,](#page-6-2) we compare the performance of neural networks trained on one-hot encoded modulo p integers and the same integers

![](_page_6_Figure_2.jpeg)

17.5% - ¼ M matrix Figure 7. Random circulant features speed up generalization in neural networks for modular addition. We compare one hidden layer MLPs with quadratic activations trained on modular addition and multiplication for p = 61 using standard one-hot encodings or those transformed by random circulant matrices. The same experiments for Mul are in Appendix Figure [9.](#page-22-0)

transformed with a random block-circulant matrix. At a training fraction of 17.5%, we find that networks trained on transformed integers achieved 100% test accuracy within several hundred epochs and exhibit little delayed generalization while networks trained on non-transformed integers do not achieve 100% test accuracy even within 3000 epochs.

Progress measures. Given that the square root of the AGOP of neural networks exhibits block-circulant structure, we use circulant deviation and AGOP alignment to measure progress of neural networks toward a generalizing solution. As before, we measure circulant deviation in the case of multiplication/division after reordering the feature submatrix by a generator of Z ∗ p . In Fig. [5B](#page-5-1), we see that our measures indicate gradual progress in contrast to sharp transitions in the standard measures of progress shown in Fig. [5A](#page-5-1). There is a period of 5-10 epochs where circulant deviation and AGOP alignment improve while test loss and test accuracy do not. As with RFM, these metrics reveal gradual progress of neural networks toward generalizing solutions.

#### 5. Fourier Multiplication from Circulants

We have seen so far that features containing circulant subblocks enable generalization for RFMs and neural networks across modular arithmetic tasks. We now provide theoretical support that shows how kernel machines equipped with such circulant features learn generalizing solutions. In particular, we show that there exist block-circulant feature matrices, as in Observation [1,](#page-4-0) such that kernel machines equipped with these features and trained on all available data for a given modulus p solve modular arithmetic through the *Fourier Multiplication Algorithm* (FMA). Notably, the FMA has been argued both empirically and theoretically in prior

works to be the solution found by neural networks to solve modular arithmetic [\(Nanda et al.,](#page-10-2) [2023;](#page-10-2) [Zhong et al.,](#page-11-4) [2024\)](#page-11-4). For completeness, we state the FMA for modular addition/ subtraction from [\(Nanda et al.,](#page-10-2) [2023\)](#page-10-2) below. While prior works write this algorithm in terms of cosines and sines, our presentation simplifies the statement by using the DFT.

Fourier Multiplication Algorithm for modular addition/subtraction. Consider the modular addition task with f ∗ (a, b) = (a + b) mod p. For a given input x = x[1] ⊕ x[2] ∈ <sup>R</sup> 2p , the FMA generates a value for output class ℓ, yadd(x; ℓ), through the following computation:

- 1. Compute the Discrete Fourier Transform (DFT) for each digit vector <sup>x</sup>[1] and <sup>x</sup>[2], which we denote <sup>x</sup>b[1] <sup>=</sup> F x[1] and <sup>x</sup>b[2] <sup>=</sup> F x[2] where the matrix <sup>F</sup> is defined in Eq. [\(2\).](#page-2-1)
- 2. Compute the element-wise product <sup>x</sup>b[1] <sup>⊙</sup> <sup>x</sup>b[2].
- 3. Return √<sup>p</sup> · ⟨xb[1] <sup>⊙</sup>xb[2], Feℓ⟩<sup>C</sup> where <sup>e</sup><sup>ℓ</sup> denotes <sup>ℓ</sup>-th standard basis vector and ⟨·, ·⟩<sup>C</sup> denotes the complex inner product (see Eq. [\(1\)\)](#page-2-2).

This algorithmic process can be written concisely in the following equation:

$$y_{\text{add}}(x; \ell) = \sqrt{p} \cdot \langle Fx_{[1]} \odot Fx_{[2]}, Fe_{\ell} \rangle_{\mathbb{C}} . \quad (8)$$

Note that for x = e<sup>a</sup> ⊕ eb, the second step of the FMA reduces to

$$F\mathbf{e}_a \odot F\mathbf{e}_b = \frac{1}{\sqrt{p}} F\mathbf{e}_{(a+b) \bmod p} . \quad (9)$$

Using the fact that F is a unitary matrix, the output of the FMA is given by

$$\sqrt{p} \cdot \left\langle \frac{1}{\sqrt{p}} F e_{(a+b) \bmod p}, F e_\ell \right\rangle_{\mathbb{C}} \quad (10)$$

$$= e_{(a+b) \bmod p}^\top F^\top \bar{F} e_\ell \quad (11)$$

$$=e_{(a+b) \bmod p}^T e_\ell \quad (12)$$

$$= \mathbb{1}_{\{(a+b) \bmod p=\ell\}} . \quad (13)$$

Thus, the output of the FMA is a vector e(a+b) mod <sup>p</sup>, which is equivalent to modular addition. We provide an example of this algorithm for p = 3 in Appendix [I.](#page-14-0)

Remarks. We note that our description of the FMA uses all entries of the DFT, referred to as frequencies, while the algorithm as proposed in prior works allows for utilizing a subset of frequencies. Also note that the FMA for subtraction, written ysub, is similar and given by

$$y_{\text{sub}}(x; \ell) = \sqrt{p} \cdot \langle Fx_{[1]} \odot Fe_{p-\ell-1}, Fx_{[2]} \rangle_{\mathbb{C}} . \quad (14)$$

Having described the FMA, we now state our theorem.

Theorem 5.1. *Given all of the discrete data* e<sup>a</sup> ⊕ eb, e(a−b) mod <sup>p</sup> <sup>p</sup>−<sup>1</sup> a,b=0*, for each output class* ℓ ∈ {0, · · · , p − 1}*, suppose we train a separate kernel predictor* fℓ(x) = k(x, X; Mℓ)α (ℓ) *where* k(·; ·; Mℓ) *is a quadratic kernel with* M<sup>ℓ</sup> = 0 C ℓ (C ℓ ) <sup>⊤</sup> 0 *and* C ∈ R p×p *is a circulant matrix with first row* e1*. When* α (ℓ) *is the solution to kernel ridgeless regression for each* ℓ*, the kernel predictor* f = [f0, . . . , fp−1] *is equivalent to Fourier Multiplication Algorithm for modular subtraction (Eq.* [\(14\)](#page-7-0)*).*

As C is circulant, C ℓ is also circulant. Hence, each M<sup>ℓ</sup> has the structure described in Observation [1,](#page-4-0) where A = 0. Note our construction differs from RFM in that we use a different feature matrix M<sup>ℓ</sup> for each output coordinate, rather than a single feature matrix across all output coordinates. Nevertheless, Theorem [5.1](#page-7-1) provides support for the fact that block-circulant feature matrices can be used to solve modular arithmetic.

We provide the proof for Theorem [5.1](#page-7-1) in Appendix [J.](#page-14-1) The argument for the FMA for addition (Eq. [\(8\)\)](#page-7-2) is identical provided we replace C <sup>ℓ</sup> with C <sup>ℓ</sup>R and (C ℓ ) <sup>⊤</sup> with (C <sup>ℓ</sup>R) ⊤ in each Mℓ, where R is the Hankel matrix that reverses the row order (i.e. ones along the main anti-diagonal, zero's elsewhere), whose first row is ep−1. An analogous result follows for multiplication and division under re-ordering by a group element, as described in Section [3.](#page-3-0)

Our proof uses the well-known fact that circulant matrices can be diagonalized using the DFT matrix [\(Gray et al.,](#page-9-10) [2006\)](#page-9-10) (see Lemma [J.2](#page-15-0) for a restatement of this fact). This fundamental relation intuitively connects circulant features and the FMA. By using kernels with block-circulant Mahalanobis matrices, we effectively represent the one-hot encoded data in terms of their Fourier transforms. We conjecture that this implicit representation is what enables RFM to learn modular arithmetic with more general circulant matrices when training on just a fraction of the discrete data.

Not only do neural networks and RFM learn similar features, we now have established a setting where kernel methods equipped with block-circulant feature matrices learn the same out-of-domain solution as neural networks on modular arithmetic tasks. This result is interesting as the only constraint for generalization on these tasks is to obtain perfect accuracy on inputs that are standard basis vectors. As such functions can be extended arbitrarily over all of R 2d , there are infinitely many generalizing solutions where the particular out-of-domain solution found by training is determined by the specifics of the learning algorithm. It is intriguing that kernel-RFMs and neural networks, which are clearly quite different algorithms, are both implicitly biased toward solutions that involve block-circulant feature matrices.

#### 6. Discussion and Conclusions

Most classical analyses of generalization relied on the training loss serving as a proxy for the test loss and thus a useful measure of generalization. Empirical results of deep learning have upended this long-standing belief. In many settings, predictors that fit the data exactly can still generalize, thus invalidating training loss as a predictor of test performance. This has led to the recent developments in understanding benign overfitting, in neural networks as well as in classical kernel and linear models [\(Belkin,](#page-9-2) [2021;](#page-9-2) [Bartlett et al.,](#page-9-1) [2021\)](#page-9-1). Since the training loss may not predict generalization, the common suggestion has been to use the validation loss computed on a separate *validation dataset*. Emergent phenomena, such as grokking, show that we cannot rely even on validation performance at intermediate training steps to predict generalization at the end of training. Indeed, validation loss at a certain iteration may not be indicative of the validation loss itself only a few iterations later. Further, contrary to [\(Schaeffer et al.,](#page-10-6) [2023\)](#page-10-6), we show these phase transitions in performance are not generally "a mirage" since, as we observe in this work, they are not always predicted by *a priori* measures of performance, continuous or discontinuous. Instead, emergence is fully determined by feature learning, which is difficult to observe without having access to a fully trained model. Indeed, the progress measures discussed in this work, as well as those suggested in, e.g., [\(Barak et al.,](#page-9-4) [2022;](#page-9-4) [Nanda et al.,](#page-10-2) [2023;](#page-10-2) [Doshi et al.,](#page-9-9) [2024\)](#page-9-9) can be termed *a posteriori* progress indicators. They all require either understanding of the algorithm implemented by a generalizing trained model (such as our circulant deviation, the Fourier gap considered in [\(Barak et al.,](#page-9-4) [2022\)](#page-9-4), or the Inverse Participation Ratio in [\(Doshi et al.,](#page-9-9) [2024\)](#page-9-9)) or access to such a model (e.g. AGOP alignment).

Consider generalizing features for modular multiplication shown in Fig. [3.](#page-3-1) The original features shown in panel B of this figure do not have an easily identifiable pattern. In contrast, re-ordered features in panel C are clearly striped, containing block-circulants. As discussed in Section [3,](#page-3-0) reordering of features requires understanding that the multiplicative group Z ∗ p is cyclic of order p − 1. While a well-known result, it is far from obvious *a priori*. It is thus plausible that in other settings hidden feature structures may be hard to identify due to a lack of mathematical insight.

Why is learning modular arithmetic surprising? The task of learning modular operations is different from many other statistical machine learning tasks. In continuous ML settings, we typically posit that the "ground truth" target function is smooth in an appropriate sense. Hence any general purpose algorithm capable of learning smooth functions (such as, for example, k-nearest neighbors) should be able to learn the target function given enough data. Primary differences between learning algorithms are thus in sample and computational efficiency. In contrast, it is unclear what

principle leads to learning modular arithmetic from partial observations. There are many ways to fill in the missing data and we do not know a simple inductive bias, to guide us toward a solution. Several recent works argued that margin maximization with respect to certain norms can account for learning modular arithmetic [\(Morwani et al.,](#page-10-9) [2024;](#page-10-9) [Lyu](#page-10-12) [et al.,](#page-10-12) [2023;](#page-10-12) [Mohamadi et al.,](#page-10-13) [2024\)](#page-10-13). While the direction is promising, general underlying principles are not yet clear.

Analyses of grokking. Recent works [\(Kumar et al.,](#page-9-11) [2024;](#page-9-11) [Lyu et al.,](#page-10-12) [2023;](#page-10-12) [Mohamadi et al.,](#page-10-13) [2024\)](#page-10-13) argue that grokking occurs in neural networks through a two phase mechanism that transitions from a "lazy" regime, with no feature learning, to a "rich" feature learning regime. Our experiments clearly show that grokking in RFM does not undergo such a transition. For RFM on modular arithmetic tasks, our progress measures indicate that the features evolve gradually toward the final circulant matrices, even as test performance initially remains constant (Fig. [2\)](#page-1-0). Grokking in these settings is entirely due to the gradual feature quality improvement and two-phase grokking does not occur. Additionally, we have not observed significant evidence of "lazy" to "rich" transition as a mechanism for grokking in our experiments with neural networks, as most of our measures of feature learning start improving early on in the training process (improvement in circulant deviation measure is delayed for addition and subtraction, but not for multiplication and division, while AGOP feature alignment initially shows near linear improvement for all tasks), see Fig. [5.](#page-5-1) Our observations for neural networks are in line with the results in [\(Doshi et al.,](#page-9-9) [2024;](#page-9-9) [Nanda et al.,](#page-10-2) [2023\)](#page-10-2), where their proposed progress measures, Inverse Participation Ratio and Gini coefficients of the weights in the Fourier domain, are shown to increase prior to improvements in test loss and accuracy for modular arithmetic.

Furthermore, as grokking modular arithmetic occurs in a kernel model equipped with a linear feature learning mechanism, a general explanation for grokking cannot depend on mechanisms that are specific to neural networks. Therefore explanations for grokking that depend on the magnitude of the weights, neural circuit efficiency, or specific optimization methods, for example, cannot account for the phenomena described in our work.

Conclusions. In this paper, we showed that grokking modular arithmetic happens in feature learning kernel machines in a manner very similar to what has been observed in neural networks. Remarkably we observe that feature learning can happen independently of improvements in both training and test loss. Not only does this finding reinforce the narrative of rapid emergence of skills in neural networks, it is also not easily explicable within the framework of the existing generalization theory.

- Impact Statement This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. References Ardeshir, N., Hsu, D. J., and Sanford, C. H. Intrinsic dimensionality and generalization properties of the r-norm inductive bias. In Neu, G. and Rosasco, L. (eds.), *Proceedings of Thirty Sixth Conference on Learning Theory*, volume 195 of *Proceedings of Machine Learning Research*, pp. 3264–3303. PMLR, 12–15 Jul 2023. URL <https://arxiv.org/pdf/2206.05317>. Arora, S. and Goyal, A. A theory for emergence of complex skills in language models. *arXiv preprint arXiv:2307.15936*, 2023. URL [https://arxiv.](https://arxiv.org/pdf/2307.15936) [org/pdf/2307.15936](https://arxiv.org/pdf/2307.15936). Barak, B., Edelman, B., Goel, S., Kakade, S., Malach, E., and Zhang, C. Hidden progress in deep learning: Sgd learns parities near the computational limit. *Advances in Neural Information Processing Systems*, 35:21750– 21764, 2022. URL [https://openreview.net/](https://openreview.net/pdf?id=8XWP2ewX-im) [pdf?id=8XWP2ewX-im](https://openreview.net/pdf?id=8XWP2ewX-im). Bartlett, P. L., Montanari, A., and Rakhlin, A. Deep learning: a statistical viewpoint. *Acta numerica*, 30:87–201, 2021. URL <https://arxiv.org/pdf/2103.09177>. Beaglehole, D., Radhakrishnan, A., Pandit, P., and Belkin,
- M. Mechanism of feature learning in convolutional neural networks. *arXiv preprint arXiv:2309.00570*, 2023. URL <https://arxiv.org/pdf/2309.00570>. Beaglehole, D., Mitliagkas, I., and Agarwala, A. Feature learning as alignment: a structural property of gradient descent in non-linear neural networks. *arXiv preprint arXiv:2402.05271*, 2024a. URL [https://arxiv.](https://arxiv.org/pdf/2402.05271) [org/pdf/2402.05271](https://arxiv.org/pdf/2402.05271). Beaglehole, D., Suken ´ ´ık, P., Mondelli, M., and Belkin, M. Average gradient outer product as a mechanism for deep neural collapse. *arXiv preprint arXiv:2402.13728*, 2024b. URL <https://arxiv.org/pdf/2402.13728>. Belkin, M. Fit without fear: remarkable mathematical phenomena of deep learning through the prism of interpolation. *Acta Numerica*, 30:203–248, 2021. URL <https://arxiv.org/pdf/2105.14368>. Damian, A., Lee, J., and Soltanolkotabi, M. Neural networks can learn representations with gradient descent. In *Conference on Learning Theory*, pp. 5413–5452. PMLR, 2022. URL <https://arxiv.org/pdf/2206.15144>. Davies, X., Langosco, L., and Krueger, D. Unifying grokking and double descent. *ML Safety Workshop, 36th Conference on Neural Information Processing Systems (NeurIPS 2022)*, 2023. URL [https://arxiv.org/](https://arxiv.org/abs/2303.06173) [abs/2303.06173](https://arxiv.org/abs/2303.06173). Doshi, D., He, T., Das, A., and Gromov, A. Grokking modular polynomials. *International Conference on Learning Representations (ICLR): BGPT Workshop*, 2024. URL <https://arxiv.org/abs/2406.03495>. Furuta, H., Minegishi, G., Iwasawa, Y., and Matsuo, Y. Interpreting grokked transformers in complex modular arithmetic. *arXiv preprint arXiv:2402.16726*, 2024. URL <https://arxiv.org/pdf/2402.16726>. Gray, R. M. et al. Toeplitz and circulant matrices: A review. *Foundations and Trends® in Communications and Information Theory*, 2(3):155–239, 2006. URL [https:](https://ee.stanford.edu/~gray/toeplitz.pdf) [//ee.stanford.edu/˜gray/toeplitz.pdf](https://ee.stanford.edu/~gray/toeplitz.pdf). Gromov, A. Grokking modular arithmetic. *arXiv preprint arXiv:2301.02679*, 2023. URL [https://arxiv.](https://arxiv.org/pdf/2301.02679) [org/pdf/2301.02679](https://arxiv.org/pdf/2301.02679). Gunasekar, S., Woodworth, B. E., Bhojanapalli, S., Neyshabur, B., and Srebro, N. Implicit regularization in matrix factorization. *Advances in neural information processing systems*, 30, 2017. Hoffman, J., Roberts, D. A., and Yaida, S. Robust learning with jacobian regularization. *arXiv preprint arXiv:1908.02729*, 5(6):7, 2019. Hristache, M., Juditsky, A., Polzehl, J., and Spokoiny, V. Structure adaptive approach for dimension reduction. *Annals of Statistics*, pp. 1537–1566, 2001. URL [https:](https://doi.org/10.1214/aos/1015345954) [//doi.org/10.1214/aos/1015345954](https://doi.org/10.1214/aos/1015345954). Koblitz, N. *A course in number theory and cryptography*, volume 114. Springer Science & Business Media, 1994. Kumar, T., Bordelon, B., Gershman, S. J., and Pehlevan, C. Grokking as the transition from lazy to rich training dynamics. *International Conference on Learning Representations (ICLR)*, 2024. URL [https://openreview.](https://openreview.net/pdf?id=vt5mnLVIVo) [net/pdf?id=vt5mnLVIVo](https://openreview.net/pdf?id=vt5mnLVIVo). Liu, Z., Kitouni, O., Nolte, N. S., Michaud, E., Tegmark, M., and Williams, M. Towards understanding grokking:

- An effective theory of representation learning. *Advances in Neural Information Processing Systems*, 35:34651– 34663, 2022. Liu, Z., Michaud, E. J., and Tegmark, M. Omnigrok: Grokking beyond algorithmic data. *International Conference on Learning Representations (ICLR)*, 2023. URL [https://openreview.net/pdf?](https://openreview.net/pdf?id=zDiHoIWa0q1) [id=zDiHoIWa0q1](https://openreview.net/pdf?id=zDiHoIWa0q1). Lyu, K., Jin, J., Li, Z., Du, S. S., Lee, J. D., and Hu,
- W. Dichotomy of early and late phase implicit biases can provably induce grokking. In *The Twelfth International Conference on Learning Representations (ICLR)*, 2023. URL [https://openreview.net/forum?](https://openreview.net/forum?id=XsHqr9dEGH) [id=XsHqr9dEGH](https://openreview.net/forum?id=XsHqr9dEGH). Miller, J., O'Neill, C., and Bui, T. Grokking beyond neural networks: An empirical exploration with model complexity. *Transactions on Machine Learning Research (TMLR)*, 2024. URL [https://openreview.net/](https://openreview.net/pdf?id=ux9BrxPCl8) [pdf?id=ux9BrxPCl8](https://openreview.net/pdf?id=ux9BrxPCl8). Mohamadi, M. A., Li, Z., Wu, L., and Sutherland, D. J. Why do you grok? a theoretical analysis on grokking modular addition. In *Forty-first International Conference on Machine Learning (ICML)*, 2024. URL [https://](https://openreview.net/forum?id=ad5I6No9G1) [openreview.net/forum?id=ad5I6No9G1](https://openreview.net/forum?id=ad5I6No9G1). Mohri, M., Rostamizadeh, A., and Talwalkar, A. *Foundations of machine learning*. MIT Press, 2018. Moitra, A. *Algorithmic aspects of machine learning*. Cambridge University Press, 2018. Morwani, D., Edelman, B. L., Oncescu, C.-A., Zhao, R., and Kakade, S. Feature emergence via margin maximization: case studies in algebraic tasks. *International Conference on Learning Representations (ICLR)*, 2024. URL [https://openreview.net/](https://openreview.net/pdf?id=i9wDX850jR) [pdf?id=i9wDX850jR](https://openreview.net/pdf?id=i9wDX850jR). Mousavi-Hosseini, A., Park, S., Girotti, M., Mitliagkas, I., and Erdogdu, M. A. Neural networks efficiently learn low-dimensional representations with sgd. *arXiv preprint arXiv:2209.14863*, 2022. URL [https://](https://arxiv.org/pdf/2209.14863) [arxiv.org/pdf/2209.14863](https://arxiv.org/pdf/2209.14863). Nanda, N., Chan, L., Lieberum, T., Smith, J., and Steinhardt, J. Progress measures for grokking via mechanistic interpretability. *International Conference on Learning Representations (ICLR)*, 2023. URL [https:](https://openreview.net/pdf?id=9XFSbDPmdW) [//openreview.net/pdf?id=9XFSbDPmdW](https://openreview.net/pdf?id=9XFSbDPmdW). Parkinson, S., Ongie, G., and Willett, R. Relu neural networks with linear layers are biased towards single- and multi-index models. *arXiv preprint arXiv:2305.15598*, 2023. URL [https://arxiv.org/pdf/2305.](https://arxiv.org/pdf/2305.15598) [15598](https://arxiv.org/pdf/2305.15598). Power, A., Burda, Y., Edwards, H., Babuschkin, I., and Misra, V. Grokking: Generalization beyond overfitting on small algorithmic datasets. *arXiv preprint arXiv:2201.02177*, 2022. Radhakrishnan, A., Beaglehole, D., Pandit, P., and Belkin,
  - M. Mechanism of feature learning in deep fully connected networks and kernel machines that recursively learn features. *arXiv preprint arXiv:2212.13881*, 2022. Radhakrishnan, A., Beaglehole, D., Pandit, P., and Belkin,
  - M. Mechanism for feature learning in neural networks and backpropagation-free machine learning models. *Science*, 383(6690):1461–1467, 2024a. doi: 10.1126/ science.adi5639. URL [https://www.science.](https://www.science.org/doi/abs/10.1126/science.adi5639) [org/doi/abs/10.1126/science.adi5639](https://www.science.org/doi/abs/10.1126/science.adi5639). Radhakrishnan, A., Belkin, M., and Drusvyatskiy, D. Linear recursive feature machines provably recover low-rank matrices. *arXiv preprint arXiv:2401.04553*, 2024b. URL <https://arxiv.org/pdf/2401.04553>. Rogers, A. and Luccioni, S. Position: Key claims in llm research have a long tail of footnotes. In *Forty-first International Conference on Machine Learning*, 2023. URL <https://arxiv.org/pdf/2308.07120>. Schaeffer, R., Miranda, B., and Koyejo, S. Are emergent abilities of large language models a mirage? In *Thirtyseventh Conference on Neural Information Processing Systems*, 2023. URL [https://openreview.net/](https://openreview.net/forum?id=ITw9edRDlD) [forum?id=ITw9edRDlD](https://openreview.net/forum?id=ITw9edRDlD). Thilak, V., Littwin, E., Zhai, S., Saremi, O., Paiss, R., and Susskind, J. The slingshot mechanism: An empirical study of adaptive optimizers and the grokking phenomenon. *arXiv preprint arXiv:2206.04817*, 2022. URL <https://arxiv.org/abs/2206.04817>. Trivedi, S., Wang, J., Kpotufe, S., and Shakhnarovich,
  - G. A consistent estimator of the expected gradient outerproduct. In *UAI*, pp. 819–828, 2014. URL [https://www.columbia.edu/˜skk2175/](https://www.columbia.edu/~skk2175/Papers/GOP-UAI.pdf) [Papers/GOP-UAI.pdf](https://www.columbia.edu/~skk2175/Papers/GOP-UAI.pdf). Trockman, A., Willmott, D., and Kolter, J. Z. Understanding the covariance structure of convolutional filters. *arXiv preprint arXiv:2210.03651*, 2022.

Varma, V., Shah, R., Kenton, Z., Kramar, J., and Ku- ´ mar, R. Explaining grokking through circuit efficiency. *International Conference on Learning Representations (ICLR)*, 2023. URL [https://openreview.net/](https://openreview.net/pdf?id=7Zbg38nA0J) [pdf?id=7Zbg38nA0J](https://openreview.net/pdf?id=7Zbg38nA0J). Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D., Chi, E. H., Hashimoto, T., Vinyals, O., Liang, P., Dean, J., and Fedus, W. Emergent abilities of large language models. *Transactions on Machine Learning Research (TMLR)*, 2022. URL [https:](https://openreview.net/pdf?id=yzkSU5zdwD) [//openreview.net/pdf?id=yzkSU5zdwD](https://openreview.net/pdf?id=yzkSU5zdwD). Yuan, G., Xu, M., Kpotufe, S., and Hsu, D. Efficient estimation of the central mean subspace via smoothed gradient outer products. *arXiv preprint arXiv:2312.15469*, 2023. URL <https://arxiv.org/pdf/2312.15469>. Zhong, Z., Liu, Z., Tegmark, M., and Andreas, J. The clock and the pizza: Two stories in mechanistic explanation of neural networks. *Advances in Neural Information Processing Systems*, 36, 2024. Zhu, L., Liu, C., Radhakrishnan, A., and Belkin, M. Catapults in sgd: spikes in the training loss and their impact on generalization through feature learning. *International Conference on Machine Learning (ICML)*, 235, 2024.

Algorithm 1 Recursive Feature Machine (RFM) [\(Radhakrishnan et al.,](#page-10-8) [2024a\)](#page-10-8)

Require: X, y, k, T, L ▷ Train data: (X, y), base kernel: k, iters.: T, matrix power: s, and bandwidth: L

M<sup>0</sup> = I<sup>d</sup>

for t = 0, . . . , T − 1 do

Solve α ← k(X, X; Mt)

<sup>−</sup><sup>1</sup>y ▷ f(t)

(x) = k(x, X; Mt)α

Mt+1 ← [G(f

(t) )]s

end for

return α, M<sup>T</sup> <sup>−</sup><sup>1</sup> ▷ Solution to kernel regression: α, and feature matrix: M<sup>T</sup> <sup>−</sup><sup>1</sup>

#### A. Additional discussion

Low rank learning. The problem of learning modular arithmetic can be viewed as a type of matrix completion – completing the p × p matrix (so-called Cayley table) representing modular operations, from partial observations. The best studied matrix completion problem is low rank matrix completion, where the goal is to fill in missing entries of a low rank matrix from observing a subset of the entries [\(Moitra,](#page-10-14) [2018,](#page-10-14) Ch.8). While many specialized algorithms exist, it has been observed that neural networks can recover low rank matrix structures [\(Gunasekar et al.,](#page-9-12) [2017\)](#page-9-12). Notably, in a development paralleling the results of this paper, low-rank matrix completion can provably be performed by linear RFMs using the same AGOP mechanism [\(Radhakrishnan et al.,](#page-10-15) [2024b\)](#page-10-15).

It is thus tempting to posit that grokking modular operations in neural networks or RFM can be explained as a low rank prediction problem. Indeed modular operations can be implemented by an index 4 model, i.e., a function of the form f = g(Ax), where x ∈ <sup>R</sup> 2p and A is a rank 4 matrix (see Appendix [K](#page-16-0) for the construction). It is a plausible conjecture as there is strong evidence, empirical and theoretical, that neural networks are capable of learning such multi-index models [\(Damian et al.,](#page-9-13) [2022;](#page-9-13) [Mousavi-Hosseini et al.,](#page-10-16) [2022\)](#page-10-16) as well as low-rank matrix completion. Furthermore, a phenomenon similar to grokking was discussed in [\(Radhakrishnan et al.,](#page-10-17) [2022,](#page-10-17) Fig. 5, 6) in the context of low rank feature learning for both neural networks and RFM. However, despite the existence of generalizeable low rank models, the actual circulant features learned by both Neural Networks and RFM are *not* low rank. Interestingly, this observation mirrors the problem of learning parity functions through neural network inspired minimum norm interpolation, which was analyzed in [\(Ardeshir et al.,](#page-9-14) [2023\)](#page-9-14). While single-directional (index one) solutions exist in that setting, the authors show that the minimum norm solutions are all multi-dimensional.

Explanations for deep learning Finally, this work adds to the growing body of evidence that the AGOP-based mechanisms of feature learning can account for some of the most interesting phenomena in deep learning. These include generalization with multi-index models [\(Parkinson et al.,](#page-10-18) [2023\)](#page-10-18), deep neural collapse [\(Beaglehole et al.,](#page-9-15) [2024b\)](#page-9-15), and the ability to perform low-rank matrix completion [\(Radhakrishnan et al.,](#page-10-15) [2024b\)](#page-10-15). Thus, RFM provides a framework that is both practically powerful and serves as a theoretically tractable model of deep learning.

## B. Additional Preliminaries

For completeness we replicate the algorithm definition for Recursive Feature Machines (RFM) provided by [Radhakrishnan](#page-10-8) [et al.](#page-10-8) [\(2024a\)](#page-10-8) in Algorithm [1.](#page-12-1) This procedure recursively fits a kernel estimator for a chosen base kernel, k, then updates the feature matrix, M, by computing a matrix power of the Average Gradient Outer Product (AGOP) for that estimator. The algorithm terminates after a total of T iterations. The final estimator and feature matrix are then returned by the algorithm.

### C. Neural Feature Ansatz

While the NFA has been observed generally across depths and architecture types [\(Radhakrishnan et al.,](#page-10-8) [2024a;](#page-10-8) [Beaglehole](#page-9-16) [et al.,](#page-9-16) [2023;](#page-9-16) [2024a\)](#page-9-17), we restate this observation for fully-connected networks with one hidden-layer of the form f(x) = a <sup>⊤</sup>ϕ(W1x).

Ansatz 1 (Neural Feature Ansatz for one hidden layer). *For a one hidden-layer neural network* f NN *and a matrix power* s ∈ (0, 1]*, the following holds:*

$$W_1^\top W_1 \propto G(f^{\text{NN}})^s. \quad (15)$$

In this work, we choose α = 2 , following the main results in [\(Radhakrishnan et al.,](#page-10-8) [2024a\)](#page-10-8). While the absolute value of the cosine similarity is written in Eq. [\(15\)](#page-12-2) to be 1, it is typically a high value less than 1, where the exact value depends on choices of initialization, architecture, dataset, and training procedure. For more understanding of these conditions, see [\(Beaglehole et al.,](#page-9-17) [2024a\)](#page-9-17).

#### D. Model and training details

Gaussian kernel: Throughout this work we take bandwidth L = 2.5 when using the Mahalanobis Gaussian kernel. We solve ridgeless kernel regression using NumPy on a standard CPU.

Neural networks: Unless otherwise specified, we train one hidden layer neural networks with quadratic activation functions and no biases in PyTorch on a single A100 GPU. Models are trained using AdamW with hidden width 1024, batch size 32, learning rate of 10−<sup>3</sup> , weight decay 1.0, and standard PyTorch initialization. All models are trained using the Mean Squared Error loss function (square loss).

For the experiments in Appendix Fig. [8,](#page-21-1) we train one hidden layer neural networks with quadratic activation and no biases on modular addition modulo p = 61. We use 40% training fraction, PyTorch standard initialization, hidden width of 512, weight decay 10−<sup>5</sup> , and AGOP regularizer weight 10−<sup>3</sup> . Models are trained with vanilla SGD, batch size 128, and learning rate 1.0.

#### E. Reordering feature matrices by group generators

Our reordering procedure uses the standard fact of group theory that the multiplicative group Z ∗ p is a cyclic group of order p − 1 [\(Koblitz,](#page-9-8) [1994\)](#page-9-8). By definition of the cyclic group, there exists at least one element g ∈ Z ∗ p , known as a *generator*, such that Z ∗ <sup>p</sup> = {g i ; i ∈ {1, . . . , p − 1}}.

Given a generator g ∈ Z ∗ p , we reorder features according to the map, ϕ<sup>g</sup> : <sup>Z</sup> ∗ <sup>p</sup> → <sup>Z</sup> ∗ p , where if h = g i , then ϕg(h) = i. In particular, given a matrix B ∈ R p×p , we reorder the bottom right (p − 1) × (p − 1) sub-block of B as follows: we move the entry in coordinate (r, c) with r, c ∈ <sup>Z</sup> ∗ p to coordinate (ϕg(r), ϕg(c)). For example if g = 2 in <sup>Z</sup> ∗ 5 , then (2, 3) entry of the sub-block would be moved to coordinate (1, 3) since 2 <sup>1</sup> = 2 and 2 <sup>3</sup> mod 5 = 3. In the setting of modular multiplication/division, the map ϕ<sup>g</sup> defined above is known as the *discrete logarithm* base g [\(Koblitz,](#page-9-8) [1994,](#page-9-8) Ch.3). The discrete logarithm is analogous to the logarithm defined for positive real numbers in the sense that it converts modular multiplication/division into modular addition/subtraction. Lastly, in this setting, we note that we only reorder the bottom (p − 1) × (p − 1) sub-block of B as the first row and column are 0 (as multiplication by 0 results in 0).

Upon re-ordering the p × p off-diagonal sub-blocks of the feature matrix by the map ϕg, the feature matrix of RFM for multiplication/division tasks contains circulant blocks as shown in Fig. [3C](#page-3-1). Thus, the reordered feature matrices for these tasks also exhibit the structure in Observation [1.](#page-4-0) As a remark, we note that there can exist several generators for a cyclic group, and thus far, we have not specified the generator g we use for re-ordering. For example, 2 and 3 are both generators of Z ∗ 5 since {2, 2 2 ,(2<sup>3</sup> mod 5),(2<sup>4</sup> mod 5)} = {3,(3<sup>2</sup> mod 5),(3<sup>3</sup> mod 5),(3<sup>4</sup> mod 5)} = <sup>Z</sup> ∗ 5 . Lemma [J.1](#page-14-2) implies that the choice of generator does not matter for observing circulant structure. As a convention, we simply reorder by the smallest generator.

## F. Enforcing circulant structure in RFM

We see that the structure in Observation [1](#page-4-0) gives generalizing features on modular arithmetic when the circulant C is constructed from the RFM matrix. We observe that enforcing this structure at every iteration, and comparing to the standard RFM model at that iteration, improves test loss and accelerates grokking on e.g. addition (Appendix Fig. [3\)](#page-19-0). The exact procedure to enforce this structure is as follows. We first perform standard RFM to generate feature matrices M1, . . . , M<sup>T</sup> . Then for each iteration of the standard RFM, we construct a new <sup>M</sup>f<sup>t</sup> on which we solve ridgeless kernel regression for a new <sup>α</sup> and evaluate on the test set. To construct <sup>M</sup>f, we take <sup>D</sup> <sup>=</sup> diag (Mt) and first let <sup>M</sup>f <sup>=</sup> <sup>D</sup>−1/<sup>2</sup>MD−1/<sup>2</sup> , to ensure the rows and columns have equal scale. We then reset the top left and bottom right sub-matrices of <sup>M</sup>f as <sup>I</sup> <sup>−</sup> p 11<sup>T</sup> , and replace the bottom-left and top-right blocks with C and C <sup>⊤</sup>, where C is an exactly circulant matrix constructed from Mt. Specifically, where c is the first column of the bottom-left sub-matrix of Mt, column ℓ of C is equal to σ ℓ (Mt).

#### G. Grokking multiple tasks

Throughout the main paper, we focused on modular arithmetic settings for a single task. In more general domains such as language, one may expect there to be many "skills" that need to be learned. In such settings, it is possible that these skills are grokked at different rates. While a full discussion is beyond the scope of this work, to illustrate this behavior, we performed additional experiments in here, where we train RFM on a pair of modular arithmetic tasks simultaneously and demonstrate that different tasks are indeed grokked at different points throughout training.

We train RFM to simultaneously solve the following two modular polynomial tasks: (1) x + y mod p ; (2) x <sup>2</sup> + y <sup>2</sup> mod p for modulus p = 61. We train RFM with the Mahalanobis Gaussian kernel using bandwidth parameter L = 2.5. Training data for both tasks is constructed from the same 80% training fraction. In addition to concatenating the one-hot encodings for x, y, we also append an extra bit indicating which task to solve (0 indicating task (1) and 1 indicating task (2)). The classification head is shared for both tasks (e.g. output dimension is still R p ).

In Appendix Fig. [4,](#page-19-1) we observe that there are two sharp transitions in the test loss and test accuracy. By decomposing the loss into the loss per task, we observe that RFM groks task (1) prior to grokking task (2). Overall, these results illustrate that grokking of different tasks can occur at different training iterations.

### H. AGOP regularization and weight decay for grokking modular arithmetic.

It has been argued in prior work that weight decay (ℓ<sup>2</sup> regularization on network weights) is necessary for grokking to occur when training neural networks for modular arithmetic tasks [\(Varma et al.,](#page-11-1) [2023;](#page-11-1) [Davies et al.,](#page-9-18) [2023;](#page-9-18) [Nanda et al.,](#page-10-2) [2023\)](#page-10-2). Under the NFA (Eq. [\(15\)\)](#page-12-2), which states that W<sup>⊤</sup> <sup>1</sup> W<sup>1</sup> is proportional to a matrix power of G(f), we expect that performing weight decay on the first layer, i.e., penalizing the loss by ∥W1∥ 2 <sup>F</sup> = tr(W<sup>⊤</sup> <sup>1</sup> W1), should behave similarly to penalizing the trace of the AGOP, tr(G(f)), during training.[<sup>3</sup>](#page-0-0) To this end, we compare the impact of using (1) no regularization; (2) weight decay; and (3) AGOP regularization when training neural networks on modular arithmetic tasks. In Appendix Fig. [8,](#page-21-1) we find that, akin to weight decay, AGOP regularization leads to grokking in cases where using no regularization results in no grokking and poor generalization. These results provide further evidence that neural networks solve modular arithmetic by using the AGOP to learn features.

## I. FMA example for p = 3

We now provide an example of the FMA for p = 3. Let x = e<sup>1</sup> ⊕ e2. In this case, we expect the FMA to output the vector e<sup>0</sup> since (1 + 2) mod 3 = 0. Following the first step of the FMA, we compute

$$\hat{x}_{[1]} = F\mathbf{e}_1 = \frac{1}{\sqrt{3}}[1, \omega, \omega^2]^\top \quad ; \quad \hat{x}_{[2]} = F\mathbf{e}_2 = \frac{1}{\sqrt{3}}[1, \omega^2, \omega^4]^\top \quad , \quad (16)$$

which are the first and second columns of F, respectively. Then their element-wise product is given by

$$F\mathbf{e}_1 \odot F\mathbf{e}_2 = \frac{1}{3}[1, \omega^3, \omega^6]^\top = \frac{1}{3}[1, 1, 1]^\top = \frac{1}{\sqrt{3}}F\mathbf{e}_0 , \quad (17)$$

which is √ 3 times the first column of the DFT matrix. Finally, we compute the outputs √ 3 D √ 3 Fe0, Fe<sup>ℓ</sup> E C for each ℓ ∈ {0, 1, 2}. As F is unitary, yadd(e<sup>1</sup> ⊕ e2; ℓ) = <sup>1</sup>{1+2=<sup>ℓ</sup> mod 3}, so that coordinate 0 of the output will have value 1, and all other coordinates have value 0.

## J. Additional results and proofs

Lemma J.1. *Let* C ∈ R <sup>p</sup>×<sup>p</sup> *with its first row and column entries all equal to* 0*. Let the* (p − 1) × (p − 1) *sub-block starting at the second row and column be* C <sup>×</sup>*. Then,* C <sup>×</sup> *is either circulant after re-ordering by any generator* q *of* <sup>Z</sup> ∗ p *, or* C <sup>×</sup> *is not circulant under re-ordering by any such generator.*

*Proof of Lemma [J.1.](#page-14-2)* We prove the lemma by showing that for any two generators q1, q<sup>2</sup> of <sup>Z</sup> ∗ p , if C <sup>×</sup> is circulant re-ordering with q1, then it is also circulant when re-ordering by q2.

<sup>3</sup>We note this regularizer been used prior work where AGOP is called the Gram matrix of the input-output Jacobian [\(Hoffman et al.,](#page-9-19) [2019\)](#page-9-19).

Suppose C <sup>×</sup> is circulant re-ordering with q1. Let i, j ∈ {1, . . . , p − 1}. Note that by the circulant assumption, for all s ∈ <sup>Z</sup>,

$$C_{q_1^i, q_1^j} = C_{q_1^{i+s}, q_1^{i+s}}, \quad (18)$$

where we take each index modulo p.

As q<sup>2</sup> is a generator for <sup>Z</sup> ∗ p , we can access all entries of C <sup>×</sup> by indexing with powers of q2. Further, as q<sup>1</sup> is a generator, we can write q<sup>2</sup> = q k 1 , for some power k. Let a ∈ Z. Then,

$$\begin{aligned} C_{q_2^i, q_2^j} &= C_{q_1^{k_i}, q_1^{k_j}} \\ &= C_{q_1^{k_i+k_a}, q_1^{k_j+k_a}} \\ &= C_{q_1^{k(i+a)}, q_1^{k(j+a)}} \\ &= C_{q_2^{i+a}, q_2^{j+a}}. \end{aligned}$$

Therefore, C is constant on the diagonals under re-ordering by q2, concluding the proof.

We next state Lemma [J.2,](#page-15-0) which is used in the proof of Theorem [5.1.](#page-7-1)

Lemma J.2 (See, e.g., [\(Gray et al.,](#page-9-10) [2006\)](#page-9-10)). *Circulant matrices* U *can be written (diagonalized) as:*

$$U = FD\bar{F}^\top,$$

*where* F *is the DFT matrix,* F¯<sup>⊤</sup> *is the element-wise complex conjugate of* F <sup>⊤</sup> *(i.e. the Hermitian of* F*), and* D *is a diagonal matrix with diagonal* √<sup>p</sup> · F u*, where* <sup>u</sup> *is the first row of* <sup>U</sup>*.*

We now present the proof of Theorem [5.1,](#page-7-1) restating the theorem below for the reader's convenience.

Theorem. *Given all of the discrete data* e<sup>a</sup> ⊕ eb, e(a−b) mod <sup>p</sup> <sup>p</sup>−<sup>1</sup> a,b=0 *in modular subtraction task, for each output class* ℓ ∈ {0, · · · , p − 1}*, we train a separate kernel predictor* fℓ(x) = k(x, X; Mℓ)α (ℓ) *. Here* k(·, ·; Mℓ) *is a quadratic kernel with* M<sup>ℓ</sup> = 0 C ℓ (C ℓ ) <sup>⊤</sup> 0 *and* C ∈ R p×p *is a circulant matrix with first row* e1*. When* α (ℓ) *is the solution to kernel ridgeless regression for each* ℓ*, the kernel predictor* f = [f0, . . . , fp−1] *is equivalent to Fourier Multiplication Algorithm for modular subtraction (Eq.* [\(14\)](#page-7-0)*).*

*Proof of Theorem [5.1.](#page-7-1)* We present the proof for modular subtraction as the proof for addition follows analogously. We write the standard kernel predictor for class ℓ on input x = x[1] ⊕ x[2] ∈ <sup>R</sup> 2p as,

$$f_\ell(x) = \sum_{a,b=0}^{p-1} \alpha_{a,b}^{(\ell)} k(x, \mathbf{e}_a \oplus \mathbf{e}_b; M_\ell),$$

where we have re-written the index into kernel coefficients for class ℓ, α (ℓ) ∈ <sup>R</sup> p×p , so that the coefficients are multi-indexed by the first and second digit. Specifically, now α (ℓ) a,b is the kernel coefficient corresponding to the representer k(·, x) for input point x = e<sup>a</sup> ⊕ eb. Recall we use a quadratic kernel, k(x, z; Mℓ) = (x <sup>⊤</sup>Mℓz) 2 . In this case, the kernel predictor simplifies to,

$$f_\ell(x) = \sum_{a,b=0}^{p-1} \alpha_{a,b}^{(\ell)} \left( x_{[1]}^\top C^\ell \mathbf{e}_b + \mathbf{e}_a^\top C^\ell x_{[2]} \right)^2.$$

Then, the labels for each pair of input digits, written as a matrix Y (ℓ) ∈ <sup>R</sup> p×p for the ℓ-th class where the row and column index the first and second digit respectively, are Y (ℓ) = C −ℓ .

For x = ea′ ⊕ e<sup>b</sup> ′ , i.e. x in the discrete dataset, we have,

$$\begin{aligned} f_\ell(x) &= \sum_{a,b=0}^{p-1} \alpha_{a,b}^{(\ell)} (\delta_{(a,b'-\ell)} + \delta_{(a',b-\ell)} + 2\delta_{(a,b'-\ell)}\delta_{(a',b-\ell)}) \\ &= \mathbf{e}_{b'-\ell}^\top \alpha^{(\ell)} \mathbf{1} + \mathbf{1}^\top \alpha^{(\ell)} \mathbf{e}_{a'+\ell} + 2\mathbf{e}_{b'-\ell}^\top \alpha^{(\ell)} \mathbf{e}_{a'+\ell} \\ &= \mathbf{e}_b^\top C^{-\ell} \alpha^{(\ell)} \mathbf{1} + \mathbf{1}^\top \alpha^{(\ell)} C^{-\ell} \mathbf{e}_{a'} + 2\mathbf{e}_b^\top C^{-\ell} \alpha^{(\ell)} C^{-\ell} \mathbf{e}_{a'} \\ &= \mathbf{e}_b^\top (C^{-\ell} \alpha \mathbf{1} \mathbf{1}^\top + \mathbf{1} \mathbf{1}^\top \alpha C^{-\ell} + 2C^{-\ell} \alpha C^{-\ell}) \mathbf{e}_{a'} , \end{aligned}$$

where δ(u,v) = <sup>1</sup>{u=v}. Let fℓ(X) ∈ <sup>R</sup> <sup>p</sup>×<sup>p</sup> be the matrix of function values of fℓ, where [fℓ(X)]a,b = fℓ(e<sup>a</sup> ⊕ eb), and, therefore, fℓ(e<sup>a</sup> ⊕ eb) = e ⊤ a fℓ(X)eb. Then, to solve for α (ℓ) , we need to solve the system of equations for α,

$$\begin{aligned} f_\ell(X) &= (C^{-\ell}\alpha\mathbf{1}\mathbf{1}^\top + \mathbf{1}\mathbf{1}^\top\alpha C^{-\ell} + 2C^{-\ell}\alpha C^{-\ell})^\top = C^{-\ell} \\ &\iff C^{-\ell}\alpha\mathbf{1}\mathbf{1}^\top + \mathbf{1}\mathbf{1}^\top\alpha C^{-\ell} + 2C^{-\ell}\alpha C^{-\ell} = C^\ell \end{aligned}$$

Note, by left-multiplying both sides by C −ℓ , we see this equation holds iff,

$$C^{-2\ell}\alpha\mathbf{1}\mathbf{1}^\top + \mathbf{1}\mathbf{1}^\top\alpha C^{-\ell} + 2C^{-2\ell}\alpha C^{-\ell} = I.$$

Note the solution is unique as the kernel matrix is full rank. We posit the solution α such that C <sup>−</sup>2<sup>ℓ</sup>αC−<sup>ℓ</sup> = 1 2 I + λ11⊤, which is α = 1 <sup>2</sup>C <sup>3</sup><sup>ℓ</sup> + λ11⊤. Then, solving for λ, we require,

$$\mathbf{11}^\top + 2p\lambda\mathbf{11}^\top + 2\lambda\mathbf{11}^\top = 0,$$

which implies λ = − <sup>2</sup>p+2 . Substituting this value of λ and simplifying, we see finally that fℓ(x) = x ⊤ [1]C <sup>−</sup>ℓx[2]. Therefore, using that circulant matrices are diagonalized by C = √pF DF¯<sup>⊤</sup> (Lemma [J.2\)](#page-15-0) and <sup>F</sup>¯⊤<sup>F</sup> <sup>=</sup> <sup>I</sup>, where <sup>D</sup> <sup>=</sup> diag (Fe1), we derive,

$$\begin{aligned} f_\ell(x) &= \sqrt{p} \cdot x_{[1]}^\top F D^{-\ell} \bar{F}^\top x_{[2]} \\ &= \sqrt{p} \cdot x_{[1]}^\top F \mathbf{diag}(F \mathbf{e}_{p-\ell-1}) \bar{F}^\top x_{[2]} \\ &= \sqrt{p} \cdot \langle F x_{[1]} \odot F \mathbf{e}_{p-\ell-1}, F x_{[2]} \rangle_{\mathbb{C}} \end{aligned}$$

which is the output of the FMA on modular subtraction.

#### K. Low rank solution to modular arithmetic

Addition We present a solution to the modular addition task whose AGOP is low rank, in contrast to the full rank AGOP recovered by RFM and neural networks.

We define the "encoding" map Φ : R <sup>p</sup> → <sup>C</sup> as follows. For a vector a = [a0, . . . , ap−1],

$$\Phi(\mathbf{a}) = \sum_{k=0}^{p-1} a_k \exp\left(\frac{k2\pi i}{p}\right).$$

Notice that <sup>Φ</sup> is a linear map such that Φ(ek) = exp k2πi p . Notice also that Φ is partially invertible with the "decoding" map Ψ : C → R p .

$$\Psi(z) = \widetilde{\max} \left( \left\langle z, \exp \left( \frac{0 \cdot 2\pi i}{p} \right) \right\rangle, \dots, \left\langle z, \exp \left( \frac{(p-1) \cdot 2\pi i}{p} \right) \right\rangle \right).$$

Above max g is a function that makes all entries zero except for the largest one and the inner product is the usual inner product in C considered as R 2 . Thus

$$\Psi \left( \exp \left( \frac{k \cdot 2\pi i}{p} \right) \right) = e_k . \quad (19)$$

By slight abuse of notation, we will define Φ : R <sup>p</sup> × <sup>R</sup> <sup>p</sup> → <sup>C</sup> <sup>2</sup> on pairs:

$$\Phi(e_j, e_k) = (\Phi(e_j), \Phi(e_k)) .$$

This is still a linear map but now to C 2 .

Consider now a quadratic map M on C <sup>2</sup> → <sup>C</sup> given by complex multiplication:

$$M(z_1, z_2) = z_1 z_2 \ .$$

It is clear that the composition ΨMΦ implements modular addition

$$\Psi M \Phi(e_j, e_k) = e_{(j+k) \bmod p}$$

Furthermore, since Φ is a liner map to a four-dimensional space, the AGOP of the composition ΨMΦ is of rank 4.

Multiplication The construction is for multiplication is very similar with modifications which we sketch below. We first re-order the non-zero coordinates by the discrete logarithm with base equal to a generator of the multiplicative group e<sup>g</sup> (see Appendix [E\)](#page-13-1), while keeping the order of index 0. Then, we modify Φ to remove index a<sup>0</sup> from the sum for inputs a. Thus for multiplication,

$$\Phi(\mathbf{a}) = \sum_{k=1}^{p-1} a_k \exp\left(\frac{k \cdot 2\pi i}{p-1}\right),$$

Hence that Φ(e0) = 0, Φ(eg) = exp 2πi p−1 and Φ(e<sup>g</sup> <sup>k</sup> ) = exp k·2πi p−1 . We extend Φ to R <sup>p</sup> × <sup>R</sup> p as in Eq. [19](#page-16-1) above. Note that Φ and the re-ordering together are still a linear map of rank 4.

Then, the "decoding" map, Ψ(z), will be modified to return 0, when z = 0, and otherwise,

$$\Psi(z) = g^{\max(\langle z, \exp(\frac{0.2\pi i}{p-1}) \rangle, \dots, \langle z, \exp(\frac{(p-2) \cdot 2\pi i}{p-1}) \rangle)}.$$

M is still defined as above. It is easy to check that the composition of ΨMΦ with reordering implements modular multiplication modulo p and furthermore, the AGOP will also be of rank 4.

![](_page_18_Figure_1.jpeg)

Figure 1. RFM with the quadratic kernel on modular arithmetic with modulus p = 61 trained for 30 iterations. (A) Test accuracy, test loss (mean squared error) over all output coordinates, and test loss of the correct class output coordinate do not change in the first 8 iterations and then, sharply transition. (B) Circulant deviation and AGOP alignment show gradual progress towards generalizing solutions despite accuracy and loss metrics not changing in the initial iterations. For multiplication (Mul) and division (Div), circulant deviation is measured with respect to the feature sub-matrices after reordering by the discrete logarithm.

![](_page_18_Figure_3.jpeg)

Figure 2. AGOP evolution for quadratic RFM trained on modular multiplication with p = 61 before reordering (top row) and after reordering by the logarithm base 2 (bottom row).

![](_page_19_Figure_1.jpeg)

Figure 3. We train a Gaussian kernel-RFM on x + y mod 97 and plot test loss and accuracy versus RFM iterations. We also evaluate the performance of the same model upon modifying the M matrix to have exact block-circulant structure stated in Observation [1.](#page-4-0)

![](_page_19_Figure_3.jpeg)

Figure 4. RFM with the Gaussian kernel trained on two modular arithmetic tasks with modulus p = 61. Task 1 is to learn x <sup>2</sup> + y <sup>2</sup> mod p and task 2 is to learn x + y mod p.

![](_page_20_Figure_1.jpeg)

Figure 5. One hidden layer fully-connected networks with quadratic activations trained on modular arithmetic with p = 61 trained for 50 epochs with the square loss. (A) Test accuracy, test loss over all outputs, and test loss of the correct class output do not change in the initial iterations. (B) Progress measures for circulant deviation and AGOP alignment. Circulant deviation for Mul and Div are computed after reordering by the discrete logarithm base 2.

![](_page_20_Figure_3.jpeg)

Figure 6. (A) We visualize the neural feature matrix (NFM) from a one hidden layer neural network with quadratic activations trained on modular multiplication and division, before and after reordering by the discrete logarithm. (B) We visualize the square root of the AGOP of the neural network in (A) before and after reordering.

![](_page_21_Figure_1.jpeg)

Figure 7. Feature matrices from one hidden layer neural networks with quadratic activations trained on addition, subtraction, multiplication, and division modulo 61. The Pearson correlations between the NFM and square root of the AGOP for each task are 0.955 (Add), 0.942 (Sub), 0.924 (Mul), 0.929 (Div). Mul and Div are shown after reordering by the discrete logarithm base 2.

![](_page_21_Figure_3.jpeg)

0.4 training fraction Figure 8. One hidden layer fully connected networks with quadratic activations trained on modular addition with p = 61 with vanilla SGD. Without any regularization the test accuracy does not go to 100% whereas using weight decay or regularizing using the trace of the AGOP result in 100% test accuracy and grokking.

![](_page_22_Figure_1.jpeg)

Figure 9. Random circulant features speed up generalization in neural networks for modular arithmetic tasks. We compare one hidden layer MLPs with quadratic activations trained on modular addition and multiplication for p = 61 using standard one-hot encodings or those transformed by random circulant matrices (re-ordered by the discrete logarithm for multiplication).

![](_page_22_Figure_3.jpeg)

Figure 10. We train kernel-RFMs for 30 iterations using the Mahalanobis Gaussian kernel for x + y mod 97. We plot test accuracy, test loss, and AGOP alignment versus percentage of training data used (denoted training fraction). All models reach convergence (i.e., both the test loss and test accuracy no longer change) after 30 iterations. We observe a sharp transition in test accuracy with respect to the training fraction, but we observe gradual change in test loss and AGOP alignment with respect to the training data fraction.