# Emergence In Non-Neural Models: Grokking Modular Arithmetic Via Average Gradient Outer Product

Neil Mallinar 1 2 Daniel Beaglehole 1 Libin Zhu 1 Adityanarayanan Radhakrishnan 2 **Parthe Pandit** 3 Mikhail Belkin 4

## Abstract

Neural networks trained to solve modular arithmetic tasks exhibit *grokking*, the phenomenon where the test accuracy improves only long after the model achieves 100% training accuracy in the training process. It is often taken as an example of "emergence", where model ability manifests sharply through a phase transition. In this work, we show that the phenomenon of grokking is not specific to neural networks nor to gradient descent-based optimization. Specifically, we show that grokking occurs when learning modular arithmetic with Recursive Feature Machines (RFM), an iterative algorithm that uses the Average Gradient Outer Product (AGOP) to enable task-specific feature learning with kernel machines. We show that RFM and, furthermore, neural networks that solve modular arithmetic learn block-circulant features transformations which implement the previously proposed Fourier multiplication algorithm.

## 1. Introduction

In recent years the idea of "emergence" has become an important narrative in machine learning. While there is no broad agreement on the definition (Rogers & Luccioni, 2023), it is often argued that "skills" emerge during the training process once certain data size, compute, or model size thresholds are achieved (Wei et al., 2022; Arora & Goyal, 2023). Furthermore, these skills are believed to appear rapidly, exhibiting sharp and seemingly unpredictable improvements in performance at these thresholds. One of 1Department of Computer Science and Engineering, UC San Diego, La Jolla, CA, USA 2The Broad Institute of MIT and Harvard, Cambridge, MA, USA 3IIT Bombay, Mumbai, Maharashtra, India 4Halıcıoglu Data Science Institute, UC San Diego, ˘
La Jolla, CA, USA. Correspondence to: Neil Mallinar <nmallina@ucsd.edu>.

Ac cu ra cy RFM Iterations Sq u are Lo ss Learned Feature (AGOP) Matrices
Figure 1. Recursive Feature Machines grok the modular arithmetic task f
∗(*x, y*) = (x + y) mod 59.

the simplest, most striking examples supporting this idea is
"grokking" modular arithmetic (Power et al., 2022; Nanda et al., 2023). A neural network trained to predict modular arithmetic operations on a fixed data set rapidly transitions from near-zero to perfect (100%) test accuracy at a certain point in the optimization process. Surprisingly, this transition point occurs long after perfect *training accuracy* is achieved. Not only is this contradictory to traditional wisdom regarding overfitting but, as we will show, some aspects of grokking do not fit neatly with our modern understanding of "benign overfitting" (Bartlett et al., 2021; Belkin, 2021). Despite a large amount of recent work on emergence and, specifically, grokking, (see, e.g., (Power et al., 2022; Liu et al., 2023; Nanda et al., 2023; Thilak et al., 2022; Furuta et al., 2024; Miller et al., 2024)), the nature and existence of the emergent phenomena remains contested. The recent paper (Schaeffer et al., 2023) suggests that the rapid emergence of skills may be a "mirage" due to the mismatch between the discontinuous metrics used for evaluation, such as accuracy, and the continuous loss used in training. The authors argue that, in contrast to accuracy, the test (or validation) loss or some other suitably chosen metric may decrease gradually 1

A
Accuracy & Loss Accu racy
(%
)

Te stLos s Correc t O utp ut Te s t Lo s s of Cl ass RFM Iterations RFM Iterations B
Progress Measures Devi ati on Ci rcu la nt A
GO
P

Ali gn ment RFM Iterations RFM Iterations Add Div
throughout training and thus provide a useful measure of progress. Another possible progress measure is the training loss. As SGD-type optimization algorithms generally result in a gradual decrease of the training loss, one may posit that skills appear once the training loss falls below a certain threshold in the optimization process. Indeed, such a conjecture is in the spirit of classical generalization theory, which considers the training loss to be a useful proxy for the test performance (Mohri et al., 2018).

In this work, we show that sharp emergence in modular arithmetic arises entirely from feature learning, independently of other aspects of modeling and training, and is not predicted by standard measures of progress. We then clarify the nature of feature learning leading to the emergence of skills in modular arithmetic. We discuss these contributions in further detail below. Summary of the contributions. We demonstrate empirically that grokking modular arithmetic: (1) is not specific to neural networks (to the best of our knowledge, no prior work shows a non-neural model that learns modular arithmetic); (2) is not tied to gradient-based optimization methods; (3)
is not predicted by training or test loss1, let alone accuracy.

Specifically, we show grokking for Recursive Feature Machines (RFM) (Radhakrishnan et al., 2024a), an algorithm that iteratively uses the Average Gradient Outer Product (AGOP) to enable task-specific feature learning in general machine learning models. In this work, we use RFM to enable feature learning in kernel machines, which are a class of predictors with no native mechanism for feature learning. In this setting, RFM iterates between three steps: (i) training a kernel machine, f, to fit training data; (ii) computing the AGOP matrix of f, M, over the training data to extract task-relevant features; and (iii) transforming input data, x, using the learned features via the map x → Ms/2x for a matrix power s > 0 (see Section 2 for details).

RFM Circ:
frob In Fig. 1 we give an example of RFM grokking modular addition, despite not using any gradient-based optimization methods and achieving numerically zero training loss at every iteration. During the first few iterations both the test loss and and test accuracy remain at the constant (random)
level. Around iteration 10 the test loss starts improving and, a few iterations later, test accuracy quickly transitions to 100%. We also observe that early in the iteration, structure emerges in AGOP feature matrices (see Fig. 1). The gradual appearance of structure in these feature matrices is striking given that the training loss is identically zero at every iteration and the test loss does not significantly change until iteration 8. The striped patterns observed in feature matrices correspond to matrices whose sub-blocks are circulant with entries that are constant along the "long" diagonals which wrap around the matrix.2 Such *circulant feature matrices* are key to learning modular arithmetic. In Section 3 we demonstrate that standard kernel machines using *random* circulant features easily learn modular operations. As these random circulant matrices are generic, we argue that no additional structure is required to solve modular arithmetic. To demonstrate that the feature matrices evolve toward this structure (including for multiplication and division under an appropriate re-ordering of the input coordinates), we introduce two "hidden progress measures" (Barak et al.,
2022): (1) *Circulant deviation*, which measures constancy of the diagonals of a matrix, and (2) *AGOP alignment*, which measures similarity between the feature matrix at iteration t and the AGOP of a fully trained model. We will see that both of these measures show gradual (initially nearly linear) progress toward a model that generalizes.

work (Gromov, 2023; Liu et al., 2022) is analogous to that for RFM and is exhibited through the AGOP (see Section 4). By visualizing covariances of network weights, we observe that these models also learn block-circulant features for modular arithmetic. We demonstrate that these features are highly correlated with the AGOP of neural networks, corroborating prior observations from (Radhakrishnan et al., 2024a). Paralleling our observations for RFM, our progress measures indicate gradual progress toward a generalizing solution during neural network training. Finally we demonstrate that training neural networks on data transformed by random block-circulant matrices dramatically decreases training time needed to learn modular arithmetic. Why are these learned block circulant features effective for modular arithmetic? We provide supporting theoretical evidence that circulant features result in kernel machines implementing the Fourier Multiplication Algorithm (FMA) for modular arithmetic (see Section 5). For the case of neural networks, several prior works have argued empirically and theoretically that neural networks learn to implement the FMA to solve modular arithmetic (Nanda et al., 2023; Varma et al., 2023; Morwani et al., 2024). While kernel RFM and neural networks utilize different classes of predictive models, our results suggest that they discover similar algorithms for implementing modular arithmetic. By decoupling feature learning from predictor training, our results provide evidence for emergent properties of machine learning models arising purely as a consequence of their ability to learn features. We hope our work will help isolate the underlying mechanisms of emergence and shed light on the key practical concern of how, when, and why these seemingly unpredictable transitions occur.

## 2. Preliminaries

Learning modular arithmetic. Let Zp = Z/pZ denote the field of integers modulo a prime p and let Z
∗
p = Zp\{0}.

We learn modular functions f
∗(a, b) = g(*a, b*) mod p where f
∗: Zp × Zp → Zp, *a, b* ∈ Zp, and g : Z × Z → Z is an arithmetic operation on a and b, e.g. g(*a, b*) = a + b. Note that there are p 2 discrete input pairs (*a, b*) for all modular operations except for f
∗(a, b*) = (*a ÷ b) mod p, which has p(p − 1) inputs as the denominator cannot be 0.

To train models on modular arithmetic tasks, we construct input-label pairs by one-hot encoding the input and label integers. Specifically, for every pair *a, b* ∈ Zp, we write the input as ea ⊕ eb ∈ R
2pand the output as ef
∗(a,b) ∈
R 
p, where ei ∈ R
pis the i-th standard basis vector in p dimensions and ⊕ is concatenation. The training dataset consists of a random subset of n = r × N input/label pairs, where r is the *training fraction* and N = p 2 or p(p − 1) is the number of possible discrete inputs.

Complex inner product and Discrete Fourier Transform. In our theoretical analysis in Section 5, we will utilize the following notions of complex inner product and DFT. The complex inner product ⟨·, ·⟩C is a map from C
d × C
d → C
of the form

$$\langle u,v\rangle_{\mathbb{C}}=u^{\top}{\bar{v}}\;,$$

⊤v , ¯ (1)
where v¯j is the complex conjugate of vj . Let i =
√−1 and let ω = exp( −2πi d). The DFT is the map F : C
d → C
d of the form F(u) = *F u,* where F ∈ C
d×dis a unitary matrix with Fij = √
1 d ω ij . In matrix form, F is given as

$$={\frac{1}{\sqrt{d}}}\begin{pmatrix}1&1&1&\cdots&1\\ 1&\omega&\omega^{2}&\cdots&\omega^{d-1}\\ 1&\omega^{2}&\omega^{4}&\cdots&\omega^{2(d-1)}\\ \vdots&\vdots&\vdots&\ddots&\vdots\\ 1&\omega^{d-1}&\omega^{2(d-1)}&\cdots&\omega^{(d-1)(d-1)}\end{pmatrix}$$
$$(1)^{\frac{1}{2}}$$
.
Circulant matrices. The features that RFMs and neural networks learn in order to solve modular arithmetic contain blocks of *circulant matrices*, which are defined as follows.

Let σ : R
p → R
p be the cyclic permutation which acts on a vector u ∈ R
p by shifting its coordinates by one cell to the right: [σ(u)]j = uj−1 mod p , for j ∈ [p]. We write the ℓ-fold composition of this map σ ℓ(u) ∈ R
p with entries
[σ ℓ(u)]j = uj−ℓ mod p. A circulant matrix C ∈ R
p×pis determined by a vector c = [c0*, . . . , c*p−1] ∈ R
p, and has rows
(in order from first to last): c, σ(c)*, . . . , σ*p−1(c). Feature matrices may also have have constant anti-diagonals ("Hankel matrices"). To ease terminology, we will use the word circulant to refer to both Hankel and circulant matrices. Average Gradient Outer Product (AGOP). The AGOP matrix is central to our discussion and defined as follows.

Definition 2.1 (AGOP). *Given a predictor* f : R
d → R
c with c outputs, f(x) ≡ [f0(x), . . . , fc−1(x)]*, let* 
∂f(x
′)
∂x ∈
R 
d×c be the Jacobian (transposed) of f evaluated at some point x
′ ∈ R
d *with entries* [
∂f(x
′)
∂x ]s,ℓ =
∂fℓ(x
′)
∂xs*. Then, for* f *trained on a set of data points* {x
(j)}
n j=1*, with* x
(j) ∈ R
d, the Average Gradient Outer Product (AGOP), G ∈ R
d×d, is defined as,

$$G(f;\{x^{(j)}\}_{j=1}^{n})={\frac{1}{n}}\sum_{j=1}^{n}{\frac{\partial f(x^{(j)})}{\partial x}}{\frac{\partial f(x^{(j)})}{\partial x}}^{\top}.\quad(3)$$

For simplicity, we omit the dependence on the dataset in the notation. Top eigenvectors of AGOP can be viewed as the "most relevant" input features, those input directions that influence the output of a general predictor (for example, a kernel machines or a neural network) the most. As a consequence, the AGOP can be viewed as a task-specific

$\gamma$. 
(2)
transformation that can be used to amplify relevant features and improve sample efficiency of machine learning models. Indeed, a line of prior works (Yuan et al., 2023; Trivedi et al., 2014; Hristache et al., 2001) have used the AGOP to improve the sample efficiency of predictors trained on multi-index models, a class of predictive tasks in which the target function depends on a low-rank subspace of the data. Though the study of AGOP has been motivated by these multi-index examples, we will see that the AGOP can be used to recover useful features for modular arithmetic that are, in fact, not low-rank. AGOP and feature learning in neural networks. (Radhakrishnan et al., 2024a) posited that AGOP was a mechanism through which neural networks learn features. The authors introduce the *Neural Feature Ansatz (NFA)* stating that for any layer ℓ of a trained neural network with weights Wℓ, the Neural Feature Matrix (NFM), WT
ℓ Wℓ, are highly correlated to the AGOP of the model computed with respect to the input of layer ℓ. The NFA suggests that neural networks learn features at each layer by utilizing the AGOP.

For more details on the NFA, see Appendix C.

Recursive Feature Machine (RFM). Importantly, AGOP can be computed for any differentiable predictor, including those such as kernel machines that have no native feature learning mechanism. As such, the authors of (Radhakrishnan et al., 2024a) developed an algorithm known as RFM, which iteratively uses the AGOP to extract features. Below, we present the RFM algorithm used in conjunction with kernel machines. Suppose we are given data samples (X, y) ∈ R
n×d × R
n where X contains n samples denoted {x
(j)}
n j=1. Given an initial symmetric positive-definite matrix M0 ∈ R
d×d, and Mahalanobis kernel k(·, · ; M) : R
d × R
d → R, RFM iterates the following steps for t ∈ [T]:
Step 1 (Predictor training): f
(t)(x) = k(*x, X*; Mt)α (4)

_Step 1 (Fraction running): $f^{(t)}(x)=k(x,X,M_{t})^{-1}$_  with $\alpha=k(X,X;M_{t})^{-1}y$ ;  _Step 2 (AGOP update): $M_{t+1}=[G(f^{(t)})]^{s}$ ;_
where s > 0 is a matrix power and k(X, X; M) ∈ R 
n×n denotes the matrix with entries [k(X, X; M)]j1j2 =
k(x
(j1), x(j2); M) for j1, j2 ∈ [n]. In this work, we select s =12 for all experiments (see Algorithm 1 for complete pseudocode). We use the following two Mahalanobis kernels: (1) the quadratic kernel, k(*x, x*′; M) =
x
⊤Mx′2; and (2) the Gaussian kernel k(*x, x*′; M) =
exp −∥x − x
′∥
2M/L, where for z ∈ R
d, ∥z∥
2M = z
⊤Mz, and L is the bandwidth.

## 3. Emergence With Rfm

We now show that RFM exhibits sharp transitions in performance on modular arithmetic tasks (addition, subtraction,

Learned Feature Matrices (AGOP)
A
B
Add C
Mul
(original)
Div
(original)
Sub Mul
(reordered)
Div
(reordered)
multiplication, and division) due to the emergence of blockcirculant features.

We will use a modulus of p = 61 and train RFM with quadratic and Gaussian kernel machines (experimental details are provided in Appendix D). As we solve kernel ridgeless regression exactly, all iterations of RFM result in zero training loss and 100% training accuracy. The top two rows of Fig. 2A show that the first several iterations of RFM result in near-zero test accuracy and approximately constant, large test loss. Despite these standard progress measures initially not changing, continuing to iterate RFM leads to a dramatic, sharp increase to 100% test accuracy and a corresponding decrease in the test loss later in the iteration process. Sharp transition in loss of correct output coordinate. It is important to note that our total loss function is the square loss averaged over p = 61 classes. It is thus plausible that, due to averaging, the near-constancy of the total square loss over the first few iterations conceals steady improvements in the predictions of the correct class. However, in Fig. 2A we show that the test loss for the output coordinate of the correct class closely tracks the total test loss. Emergence of block-circulant features in RFM. To understand RFM generalization, we visualize the 2p × 2p feature matrix given by the square root of the AGOP from the final iteration of RFM. We first visualize the feature matrices for RFM trained on modular addition/subtraction in Fig. 3A.

Their visually-evident striped structure suggests a more precise characterization:
Observation 1 (Block-circulant features). *Feature matrix* M∗ ∈ R
2p×2p at the final iteration of RFM on modular addition/subtraction is of the form

$$M^{*}=\begin{pmatrix}A&C^{\top}\\ C&A\end{pmatrix},$$

where *A, C* ∈ R
p×p, C *is an asymmetric circulant matrix. ,*
A = c1I + c211⊤ for scalars c1, c2.

Similarly to addition and subtraction, RFM successfully learns multiplication and division. Yet, in contrast to addition and subtraction, the structure of feature matrices for these tasks, shown in Fig. 3B, is not at all obvious. Nevertheless, re-ordering the rows and columns of the feature matrices for these tasks brings out their hidden circulant structure of the form stated in Eq. (7). We show the effect of re-ordering in Fig. 3C (see also Appendix Fig. 2 for the evolution of re-ordered and original features during training). We briefly discuss the reordering procedure below and provide further details in Appendix E. To reorder, we use the fact of group theory that the multiplicative group Z
∗p is a cyclic group of order p − 1 (e.g., (Koblitz, 1994)).

By definition of the cyclic group, there exists at least one element g ∈ Z
∗p, known as a *generator*, such that Z 
∗p = {g i; i ∈ {1*, . . . , p* − 1}}. As we will see, reordering the rows and columns of the AGOP by powers of a generator reveals circulant structure. For modular multiplication/division, the map taking g ito i is known as the discrete logarithm base g (Koblitz, 1994, Ch.3). It is natural to expect block-circulant feature matrices to arise in modular multiplication/division after reordering by the discrete log as the discrete log converts modular multiplication/division into modular addition/subtraction. We note the recent work (Doshi et al., 2024) also used the discrete log to reorder coordinates in the context of constructing a solution for solving modular multiplication with neural networks. Progress measures. We propose two measures of feature learning, *circulant deviation* and *AGOP alignment*. Circulant deviation. As the final feature matrices contain circulant sub-blocks, a natural progress measure for learning modular arithmetic with RFM is how far AGOP feature matrices are from a block-circulant matrix. For a feature matrix M, let A denote the bottom-left sub-block of M. We define circulant deviation as the total variance of the (wrapped) diagonals of A normalized by the norm ∥A∥
2 F. In particular, let S ∈ R
p×p → R
p×p denote the shift operator, which shifts the ℓ-th row of the matrix by ℓ positions to the right.

Also let Var(v) = Pp−1 j=0 
(vj − Ev)
2 be the variance of a vector v. If A[j] denotes the j-th column of A, we define circulant deviation D as: D(A) = 1
∥A∥
2F
Pp−1 j=0 Var(S(A)[j]).

As circulant matrices are constant along their (wrapped)
diagonals, they have a circulant deviation of 0. 

Add Mul

$$\left(7\right)$$

Tes t Los s Tes t Accu ra cy 
(%)
Tes t Los s Tes t Accu ra cy 
(%)
Training fraction (%) Training fraction (%)
We see in Fig. 2B that circulant deviation exhibits gradual improvement through the course of training with RFM. We find that for the first 10 iterations, while the training loss is numerically zero and the test loss does not improve, circulant deviation exhibits gradual, nearly linear, improvement. The improvements in circulant deviation reflect visual improvements in features, as was also shown in Fig. 1. These curves provide further support for Observation 1, as circulant deviation is close to 0 at the end of training. Circulant deviation depends crucially on the observation that for modular arithmetic the feature matrices contained circulant blocks. For more general tasks, we may not be able to identify such structure. Thus, we propose a second, more general progress measure, AGOP alignment.

AGOP alignment. Given two matrices *A, B* ∈ R
d×d, let ρ(A, B) denote the standard cosine similarity between these two matrices when vectorized. Specifically, let A, ˜ B˜ ∈
R 
d 2denote the vectorization of A and B respectively, then ρ(*A, B*) = ⟨A, 
˜ B˜⟩
∥A˜∥ ∥B˜∥
.

If Mt denotes the AGOP at iteration t of RFM (or epoch t of a neural network) and M∗ denotes the final AGOP of the trained RFM (or neural network), then AGOP alignment at iteration t is given by ρ(Mt, M∗). The same measure of alignment was used in (Zhu et al., 2024), except their alignment was computed with respect to the AGOP of the ground truth model. Note that as modular operations are discrete, in our setting there is no unique ground truth model for which AGOP can be computed. Like circulant deviation, AGOP alignment exhibits gradual improvement in the regime that test loss is constant and large
(see Fig. 2B, bottom row). Moreover, AGOP alignment is a more general progress measure since it does not require assumptions on the structure of the AGOP. For instance, AGOP alignment can be measured without reordering for modular multiplication/division. While AGOP alignment does not require a specific form of the final features, it is still an *a posteriori* measurement of progress as it requires access to the features of a fully trained model. Random circulant features allow standard kernels to generalize. We conclude this section by providing further evidence that the form of feature matrices given in Observation 1 is key to enabling generalization in kernel machines trained to solve modular arithmetic tasks. We now show that a transformation with a *generic* block-circulant matrix enables kernels machines to learn modular arithmetic. We generate a random circulant matrix C by first sampling entries of the first column i.i.d. from the uniform distribution on [0, 1] ⊂ R and then shifting the column to generate the remaining columns of C. We construct M∗in Observation 1 with c1 = 1, c2 = −1/p. For modular addition, we transform the input data by mapping xab = ea ⊕ eb to x˜ab = (M∗)
1 4 xab , and then train on the new data pairs (˜xab, ea+b mod p) for a subset of all possible pairs (a, b) ∈ Z
2p. Note that transforming data with (M∗)
1 4 is akin to using s = 1/2 in RFM. We do the same for modular multiplication after reordering the random circulant by the discrete logarithm as described above. The experiments in Fig. 4 show that standard kernel machines trained on feature matrices with random circulant blocks outperform RFM that learns such features through AGOP. We also find that directly enforcing circulant blocks in the sub-matrices of Mt throughout RFM iterations accelerates grokking and improves test loss (see Appendix F, Appendix Fig. 3). These experiments provide direct evidence that the structure in Observation 1 is key for generalization on modular arithmetic and, furthermore, no additional structure beyond a generic circulant is required.

## 4. Emergence In Neural Nets Through Agop

We now show that grokking in two-layer neural networks relies on the same principles as grokking by RFM. Specifically we demonstrate that (1) block-circulant features are key to neural networks grokking modular arithmetic; and (2) our measures (circulant deviation and AGOP alignment) indicate gradual progress towards generalization, while standard measures of generalization exhibit sharp transitions. All experimental details are provided in Appendix D. Grokking with neural networks. We first reproduce grokking with modular arithmetic using fully-connected networks as identified in prior works (Fig. 5A) (Gromov, 2023). In particular, we train one hidden layer fully connected networks f : R
2p → R
p of the form f(x) = W2ϕ(W1x) with quadratic activation ϕ(z) = z 2 on modulus p = 61 data

A
Accuracy & Loss Acc uracy
(%
)

Lo ss Loss of Co rre ct Ou tp ut Cl as s Epochs Epoch B
Progress Measures Ci rc ula nt Dev ia tio n AG OP 
Al ig n m en t Epochs Epochs Add Div
Consistent with prior work (Gromov, 2023) and analogously to RFMs, neural networks exhibit an initial training period where the train accuracy reaches 100%, while test accuracy is at 0% and test loss does not improve (see Fig. 5A). After this point, we see that the accuracy rapidly improves to achieve perfect generalization. We further verify that the sharp transition in test loss is not an artifact of averaging the loss over all output coordinates. In the third row of Fig. 5A we show that the test loss of the individual correct output coordinate closely tracks the total loss. Emergence of block-circulant features in neural networks. To understand the features learned by neural networks we visualize the first layer Neural Feature Matrix. Definition 4.1. (Neural Feature Matrix) Given a fully connected network f(x) = a
⊤ϕ(W1x)*, the first layer Neural* Feature Matrix (NFM) is the matrix W⊤
1 W1 ∈ R
2p×2p.

The NFM is the un-centered covariance of network weights and has been used in prior work in order to understand the features learned by various neural network architectures at any layer (Radhakrishnan et al., 2024a; Trockman et al.,
2022). Fig. 6A displays the NFM for one hidden layer neural networks with quadratic activations trained on modular

A
NFM

B
NN AGOP
Add Div (reordered)
arithmetic tasks. For addition/subtraction, we find that the NFM exhibits block circulant structure, akin to the feature matrix for RFM. As described in Section 3 and Appendix E, we reorder the NFM for networks trained on multiplication/division with respect to a generator for Z
∗pin order to observe block-circulant structure (see Appendix Fig. 6A for a comparison of multiplication/division NFMs before and after reordering). The block-circulant structure in both the NFM and the feature matrix of RFM suggests that the two models are learning similar sets of features. The work (Radhakrishnan et al., 2024a) posited that AGOP
is the mechanism through which neural networks learn features. The authors stated their claim in the form of the Neural Feature Ansatz (NFA), which states that NFMs are proportional to a matrix power of AGOP through training (see Eq. (15) for a restatement of the NFA). As such, we additionally compute the square root of the AGOP to examine the features learned by neural networks trained on modular arithmetic tasks. We visualize the square root of the AGOPs of these trained models in Fig. 6B and also find that the square root of the AGOP and the NFM are highly correlated (greater than 0.92), where Pearson correlation is equal to cosine similarity after centering the inputs to be mean 0. Moreover, we find that the square root of AGOP of neural networks again exhibits the same structure as stated in Observation 1 (see Appendix Fig. 6B for a comparison of multiplication/division AGOPs before and after reordering). Random circulant maps improve generalization of neural networks. To further establish the importance and generality of block-circulant features, we demonstrate that training networks on inputs transformed with a random blockcirculant matrix greatly accelerates learning. In Fig. 7, we compare the performance of neural networks trained on one-hot encoded modulo p integers and the same integers

Add NN Random Circulant + NN
Ac cu ra cy (
%)
Ac cu ra cy (
%)
Sq ua re Lo ss Sq ua re Lo ss Epochs Epochs
transformed with a random block-circulant matrix. At a training fraction of 17.5%, we find that networks trained on transformed integers achieved 100% test accuracy within several hundred epochs and exhibit little delayed generalization while networks trained on non-transformed integers do not achieve 100% test accuracy even within 3000 epochs.

Progress measures. Given that the square root of the AGOP of neural networks exhibits block-circulant structure, we use circulant deviation and AGOP alignment to measure progress of neural networks toward a generalizing solution. As before, we measure circulant deviation in the case of multiplication/division after reordering the feature submatrix by a generator of Z
∗p. In Fig. 5B, we see that our measures indicate gradual progress in contrast to sharp transitions in the standard measures of progress shown in Fig. 5A. There is a period of 5-10 epochs where circulant deviation and AGOP alignment improve while test loss and test accuracy do not. As with RFM, these metrics reveal gradual progress of neural networks toward generalizing solutions.

## 5. Fourier Multiplication From Circulants

We have seen so far that features containing circulant subblocks enable generalization for RFMs and neural networks across modular arithmetic tasks. We now provide theoretical support that shows how kernel machines equipped with such circulant features learn generalizing solutions. In particular, we show that there exist block-circulant feature matrices, as in Observation 1, such that kernel machines equipped with these features and trained on all available data for a given modulus p solve modular arithmetic through the Fourier Multiplication Algorithm (FMA). Notably, the FMA
has been argued both empirically and theoretically in prior works to be the solution found by neural networks to solve modular arithmetic (Nanda et al., 2023; Zhong et al., 2024). For completeness, we state the FMA for modular addition/- subtraction from (Nanda et al., 2023) below. While prior works write this algorithm in terms of cosines and sines, our presentation simplifies the statement by using the DFT.

Fourier Multiplication Algorithm for modular addition/subtraction. Consider the modular addition task with f
∗(a, b) = (a + b) mod p. For a given input x =
x[1] ⊕ x[2] ∈ R
2p, the FMA generates a value for output class ℓ, yadd(x; ℓ), through the following computation:
1. Compute the Discrete Fourier Transform (DFT) for each digit vector x[1] and x[2], which we denote xb[1] = F x[1] and xb[2] = F x[2] where the matrix F is defined in Eq. (2).

2. Compute the element-wise product xb[1] ⊙ xb[2].

3. Return 
√p · ⟨xb[1] ⊙xb[2], Feℓ⟩C where eℓ denotes ℓ-th standard basis vector and ⟨·, ·⟩C denotes the complex inner product (see Eq. (1)).

This algorithmic process can be written concisely in the following equation:

$$y_{\rm add}(x;\ell)=\sqrt{p}\cdot\left\langle Fx_{[1]}\circ Fx_{[2]},F\mathbf{e}_{\ell}\right\rangle_{\mathbb{C}}.\tag{8}$$

Note that for x = ea ⊕ eb, the second step of the FMA
reduces to

$$F\mathbf{e}_{a}\odot F\mathbf{e}_{b}={\frac{1}{\sqrt{p}}}F\mathbf{e}_{(a+b)\,\mathrm{mod}\,p}\ .$$

Using the fact that F is a unitary matrix, the output of the FMA is given by

$$\sqrt{p}\cdot\left\langle\frac{1}{\sqrt{p}}F\mathbf{e}_{(a+b)\bmod p},F\mathbf{e}_{\ell}\right\rangle_{\mathbb{C}}\tag{1}$$ $$=\mathbf{e}_{(a+b)\bmod p}^{\top}F^{\top}\bar{F}\mathbf{e}_{\ell}$$ (2) $$=\mathbf{e}_{(a+b)\bmod p}^{\top}\mathbf{e}_{\ell}$$ (3) $$=\mathbb{I}_{\{(a+b)\bmod p=\ell\}}\.$$

$$\quad(10)$$  $$(11)$$

Thus, the output of the FMA is a vector e(a+b) mod p, which is equivalent to modular addition. We provide an example of this algorithm for p = 3 in Appendix I.

Remarks. We note that our description of the FMA uses all entries of the DFT, referred to as frequencies, while the algorithm as proposed in prior works allows for utilizing a subset of frequencies. Also note that the FMA for subtraction, written ysub, is similar and given by

$$y_{\rm sub}(x;\ell)=\sqrt{p}\cdot\left\langle Fx_{[1]}\ \odot F\mathbf{e}_{p-\ell-1},Fx_{[2]}\right\rangle_{\mathbb{C}}.\tag{14}$$

Having described the FMA, we now state our theorem.

Theorem 5.1. *Given all of the discrete data* ea ⊕ eb, e(a−b) mod p	p−1 a,b=0*, for each output class* ℓ ∈ {0, · · · , p − 1}*, suppose we train a separate kernel* predictor fℓ(x) = k(*x, X*; Mℓ)α
(ℓ) *where* k(·; ·; Mℓ)
is a quadratic kernel with Mℓ =
0 C
ℓ
(C
ℓ)
⊤ 0 and C ∈ R
p×pis a circulant matrix with first row e1*. When* α
(ℓ)*is the solution to kernel ridgeless regression for each* ℓ*, the kernel predictor* f = [f0, . . . , fp−1] *is equivalent to* Fourier Multiplication Algorithm for modular subtraction (Eq. (14)). As C is circulant, C
ℓis also circulant. Hence, each Mℓ has the structure described in Observation 1, where A = 0. Note our construction differs from RFM in that we use a different feature matrix Mℓ for each output coordinate, rather than a single feature matrix across all output coordinates. Nevertheless, Theorem 5.1 provides support for the fact that block-circulant feature matrices can be used to solve modular arithmetic.

$$(9)$$

We provide the proof for Theorem 5.1 in Appendix J. The argument for the FMA for addition (Eq. (8)) is identical provided we replace C
ℓ with C
ℓR and (C
ℓ)
⊤ with (C
ℓR)
⊤
in each Mℓ, where R is the Hankel matrix that reverses the row order (i.e. ones along the main anti-diagonal, zero's elsewhere), whose first row is ep−1. An analogous result follows for multiplication and division under re-ordering by a group element, as described in Section 3.

Our proof uses the well-known fact that circulant matrices can be diagonalized using the DFT matrix (Gray et al.,
2006) (see Lemma J.2 for a restatement of this fact). This fundamental relation intuitively connects circulant features and the FMA. By using kernels with block-circulant Mahalanobis matrices, we effectively represent the one-hot encoded data in terms of their Fourier transforms. We conjecture that this implicit representation is what enables RFM to learn modular arithmetic with more general circulant matrices when training on just a fraction of the discrete data.

$$(12)^{\frac{1}{2}}$$
$$(13)^{\frac{1}{2}}$$

Not only do neural networks and RFM learn similar features, we now have established a setting where kernel methods equipped with block-circulant feature matrices learn the same out-of-domain solution as neural networks on modular arithmetic tasks. This result is interesting as the only constraint for generalization on these tasks is to obtain perfect accuracy on inputs that are standard basis vectors. As such functions can be extended arbitrarily over all of R
2d, there are infinitely many generalizing solutions where the particular out-of-domain solution found by training is determined by the specifics of the learning algorithm. It is intriguing that kernel-RFMs and neural networks, which are clearly quite different algorithms, are both implicitly biased toward solutions that involve block-circulant feature matrices.

## 6. Discussion And Conclusions

Most classical analyses of generalization relied on the training loss serving as a proxy for the test loss and thus a useful measure of generalization. Empirical results of deep learning have upended this long-standing belief. In many settings, predictors that fit the data exactly can still generalize, thus invalidating training loss as a predictor of test performance.

This has led to the recent developments in understanding benign overfitting, in neural networks as well as in classical kernel and linear models (Belkin, 2021; Bartlett et al.,
2021). Since the training loss may not predict generalization, the common suggestion has been to use the validation loss computed on a separate *validation dataset*. Emergent phenomena, such as grokking, show that we cannot rely even on validation performance at intermediate training steps to predict generalization at the end of training. Indeed, validation loss at a certain iteration may not be indicative of the validation loss itself only a few iterations later. Further, contrary to (Schaeffer et al., 2023), we show these phase transitions in performance are not generally "a mirage" since, as we observe in this work, they are not always predicted by a priori measures of performance, continuous or discontinuous. Instead, emergence is fully determined by feature learning, which is difficult to observe without having access to a fully trained model. Indeed, the progress measures discussed in this work, as well as those suggested in, e.g., (Barak et al., 2022; Nanda et al., 2023; Doshi et al., 2024) can be termed a posteriori progress indicators. They all require either understanding of the algorithm implemented by a generalizing trained model (such as our circulant deviation, the Fourier gap considered in (Barak et al., 2022), or the Inverse Participation Ratio in (Doshi et al., 2024)) or access to such a model (e.g. AGOP alignment). Consider generalizing features for modular multiplication shown in Fig. 3. The original features shown in panel B of this figure do not have an easily identifiable pattern. In contrast, re-ordered features in panel C are clearly striped, containing block-circulants. As discussed in Section 3, reordering of features requires understanding that the multiplicative group Z
∗pis cyclic of order p − 1. While a well-known result, it is far from obvious *a priori*. It is thus plausible that in other settings hidden feature structures may be hard to identify due to a lack of mathematical insight. Why is learning modular arithmetic surprising? The task of learning modular operations is different from many other statistical machine learning tasks. In continuous ML settings, we typically posit that the "ground truth" target function is smooth in an appropriate sense. Hence any general purpose algorithm capable of learning smooth functions
(such as, for example, k-nearest neighbors) should be able to learn the target function given enough data. Primary differences between learning algorithms are thus in sample and computational efficiency. In contrast, it is unclear what principle leads to learning modular arithmetic from partial observations. There are many ways to fill in the missing data and we do not know a simple inductive bias, to guide us toward a solution. Several recent works argued that margin maximization with respect to certain norms can account for learning modular arithmetic (Morwani et al., 2024; Lyu et al., 2023; Mohamadi et al., 2024). While the direction is promising, general underlying principles are not yet clear. Analyses of grokking. Recent works (Kumar et al., 2024; Lyu et al., 2023; Mohamadi et al., 2024) argue that grokking occurs in neural networks through a two phase mechanism that transitions from a "lazy" regime, with no feature learning, to a "rich" feature learning regime. Our experiments clearly show that grokking in RFM does not undergo such a transition. For RFM on modular arithmetic tasks, our progress measures indicate that the features evolve gradually toward the final circulant matrices, even as test performance initially remains constant (Fig. 2). Grokking in these settings is entirely due to the gradual feature quality improvement and two-phase grokking does not occur. Additionally, we have not observed significant evidence of "lazy" to "rich" transition as a mechanism for grokking in our experiments with neural networks, as most of our measures of feature learning start improving early on in the training process (improvement in circulant deviation measure is delayed for addition and subtraction, but not for multiplication and division, while AGOP feature alignment initially shows near linear improvement for all tasks), see Fig. 5. Our observations for neural networks are in line with the results in (Doshi et al., 2024; Nanda et al., 2023), where their proposed progress measures, Inverse Participation Ratio and Gini coefficients of the weights in the Fourier domain, are shown to increase prior to improvements in test loss and accuracy for modular arithmetic. Furthermore, as grokking modular arithmetic occurs in a kernel model equipped with a linear feature learning mechanism, a general explanation for grokking cannot depend on mechanisms that are specific to neural networks. Therefore explanations for grokking that depend on the magnitude of the weights, neural circuit efficiency, or specific optimization methods, for example, cannot account for the phenomena described in our work.

Conclusions. In this paper, we showed that grokking modular arithmetic happens in feature learning kernel machines in a manner very similar to what has been observed in neural networks. Remarkably we observe that feature learning can happen independently of improvements in both training and test loss. Not only does this finding reinforce the narrative of rapid emergence of skills in neural networks, it is also not easily explicable within the framework of the existing generalization theory.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

Ardeshir, N., Hsu, D. J., and Sanford, C. H. Intrinsic dimensionality and generalization properties of the r-norm inductive bias. In Neu, G. and Rosasco, L. (eds.), Proceedings of Thirty Sixth Conference on Learning Theory, volume 195 of Proceedings of Machine Learning Research, pp. 3264–3303. PMLR, 12–15 Jul 2023. URL
https://arxiv.org/pdf/2206.05317.

Arora, S. and Goyal, A. A theory for emergence of complex skills in language models. *arXiv preprint* arXiv:2307.15936, 2023. URL https://arxiv. org/pdf/2307.15936.

Barak, B., Edelman, B., Goel, S., Kakade, S., Malach, E.,
and Zhang, C. Hidden progress in deep learning: Sgd learns parities near the computational limit. Advances in Neural Information Processing Systems, 35:21750–
21764, 2022. URL https://openreview.net/ pdf?id=8XWP2ewX-im.

Bartlett, P. L., Montanari, A., and Rakhlin, A. Deep learning:
a statistical viewpoint. *Acta numerica*, 30:87–201, 2021. URL https://arxiv.org/pdf/2103.09177.

Beaglehole, D., Radhakrishnan, A., Pandit, P., and Belkin, M. Mechanism of feature learning in convolutional neural networks. *arXiv preprint arXiv:2309.00570*, 2023. URL https://arxiv.org/pdf/2309.00570.

Beaglehole, D., Mitliagkas, I., and Agarwala, A. Feature learning as alignment: a structural property of gradient descent in non-linear neural networks. *arXiv preprint* arXiv:2402.05271, 2024a. URL https://arxiv. org/pdf/2402.05271.

Beaglehole, D., Suken ´ ´ık, P., Mondelli, M., and Belkin, M.

Average gradient outer product as a mechanism for deep neural collapse. *arXiv preprint arXiv:2402.13728*, 2024b.

URL https://arxiv.org/pdf/2402.13728.

Belkin, M. Fit without fear: remarkable mathematical phenomena of deep learning through the prism of interpolation. *Acta Numerica*, 30:203–248, 2021. URL https://arxiv.org/pdf/2105.14368.

Damian, A., Lee, J., and Soltanolkotabi, M. Neural networks can learn representations with gradient descent. In Conference on Learning Theory, pp. 5413–5452. PMLR, 2022. URL https://arxiv.org/pdf/2206.15144.

Davies, X., Langosco, L., and Krueger, D. Unifying grokking and double descent. *ML Safety Workshop, 36th* Conference on Neural Information Processing Systems (NeurIPS 2022), 2023. URL https://arxiv.org/ abs/2303.06173.

Doshi, D., He, T., Das, A., and Gromov, A. Grokking modular polynomials. *International Conference on Learning* Representations (ICLR): BGPT Workshop, 2024. URL
https://arxiv.org/abs/2406.03495.

Furuta, H., Minegishi, G., Iwasawa, Y., and Matsuo, Y.

Interpreting grokked transformers in complex modular arithmetic. *arXiv preprint arXiv:2402.16726*, 2024. URL
https://arxiv.org/pdf/2402.16726.

Gray, R. M. et al. Toeplitz and circulant matrices: A review. Foundations and Trends® *in Communications and* Information Theory, 2(3):155–239, 2006. URL https:
//ee.stanford.edu/˜gray/toeplitz.pdf.

Gromov, A. Grokking modular arithmetic. arXiv preprint arXiv:2301.02679, 2023. URL https://arxiv. org/pdf/2301.02679.

Gunasekar, S., Woodworth, B. E., Bhojanapalli, S.,
Neyshabur, B., and Srebro, N. Implicit regularization in matrix factorization. Advances in neural information processing systems, 30, 2017.

Hoffman, J., Roberts, D. A., and Yaida, S. Robust learning with jacobian regularization. arXiv preprint arXiv:1908.02729, 5(6):7, 2019.

Hristache, M., Juditsky, A., Polzehl, J., and Spokoiny, V.

Structure adaptive approach for dimension reduction. Annals of Statistics, pp. 1537–1566, 2001. URL https: //doi.org/10.1214/aos/1015345954.

Koblitz, N. *A course in number theory and cryptography*,
volume 114. Springer Science & Business Media, 1994.

Kumar, T., Bordelon, B., Gershman, S. J., and Pehlevan, C.

Grokking as the transition from lazy to rich training dynamics. International Conference on Learning Representations (ICLR), 2024. URL https://openreview. net/pdf?id=vt5mnLVIVo.

Liu, Z., Kitouni, O., Nolte, N. S., Michaud, E., Tegmark, M., and Williams, M. Towards understanding grokking:
An effective theory of representation learning. Advances in Neural Information Processing Systems, 35:34651– 34663, 2022.

Liu, Z., Michaud, E. J., and Tegmark, M. Omnigrok: Grokking beyond algorithmic data. International Conference on Learning Representations (ICLR), 2023. URL https://openreview.net/pdf? id=zDiHoIWa0q1.

Lyu, K., Jin, J., Li, Z., Du, S. S., Lee, J. D., and Hu, W. Dichotomy of early and late phase implicit biases can provably induce grokking. In The Twelfth International Conference on Learning Representations (ICLR),
2023. URL https://openreview.net/forum? id=XsHqr9dEGH.

Miller, J., O'Neill, C., and Bui, T. Grokking beyond neural networks: An empirical exploration with model complexity. *Transactions on Machine Learning Research* (TMLR), 2024. URL https://openreview.net/
pdf?id=ux9BrxPCl8.

Mohamadi, M. A., Li, Z., Wu, L., and Sutherland, D. J.

Why do you grok? a theoretical analysis on grokking modular addition. In Forty-first International Conference on Machine Learning (ICML), 2024. URL https:// openreview.net/forum?id=ad5I6No9G1.

Mohri, M., Rostamizadeh, A., and Talwalkar, A. Foundations of machine learning. MIT Press, 2018.

Moitra, A. *Algorithmic aspects of machine learning*. Cambridge University Press, 2018.

Morwani, D., Edelman, B. L., Oncescu, C.-A., Zhao, R., and Kakade, S. Feature emergence via margin maximization: case studies in algebraic tasks. International Conference on Learning Representations (ICLR), 2024. URL https://openreview.net/ pdf?id=i9wDX850jR.

Mousavi-Hosseini, A., Park, S., Girotti, M., Mitliagkas, I., and Erdogdu, M. A. Neural networks efficiently learn low-dimensional representations with sgd. arXiv preprint arXiv:2209.14863, 2022. URL https:// arxiv.org/pdf/2209.14863.

Nanda, N., Chan, L., Lieberum, T., Smith, J., and Steinhardt, J. Progress measures for grokking via mechanistic interpretability. *International Conference on* Learning Representations (ICLR), 2023. URL https: //openreview.net/pdf?id=9XFSbDPmdW.

Parkinson, S., Ongie, G., and Willett, R. Relu neural networks with linear layers are biased towards single- and multi-index models. *arXiv preprint arXiv:2305.15598*, 2023. URL https://arxiv.org/pdf/2305. 15598.

Power, A., Burda, Y., Edwards, H., Babuschkin, I., and Misra, V. Grokking: Generalization beyond overfitting on small algorithmic datasets. *arXiv preprint* arXiv:2201.02177, 2022.

Radhakrishnan, A., Beaglehole, D., Pandit, P., and Belkin, M. Mechanism of feature learning in deep fully connected networks and kernel machines that recursively learn features. *arXiv preprint arXiv:2212.13881*, 2022.

Radhakrishnan, A., Beaglehole, D., Pandit, P., and Belkin, M. Mechanism for feature learning in neural networks and backpropagation-free machine learning models. Science, 383(6690):1461–1467, 2024a. doi: 10.1126/ science.adi5639. URL https://www.science.

org/doi/abs/10.1126/science.adi5639.

Radhakrishnan, A., Belkin, M., and Drusvyatskiy, D. Linear recursive feature machines provably recover low-rank matrices. *arXiv preprint arXiv:2401.04553*, 2024b. URL https://arxiv.org/pdf/2401.04553.

Rogers, A. and Luccioni, S. Position: Key claims in llm research have a long tail of footnotes. In Forty-first International Conference on Machine Learning, 2023. URL https://arxiv.org/pdf/2308.07120.

Schaeffer, R., Miranda, B., and Koyejo, S. Are emergent abilities of large language models a mirage? In Thirtyseventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/ forum?id=ITw9edRDlD.

Thilak, V., Littwin, E., Zhai, S., Saremi, O., Paiss, R.,
and Susskind, J. The slingshot mechanism: An empirical study of adaptive optimizers and the grokking phenomenon. *arXiv preprint arXiv:2206.04817*, 2022. URL https://arxiv.org/abs/2206.04817.

Trivedi, S., Wang, J., Kpotufe, S., and Shakhnarovich, G. A consistent estimator of the expected gradient outerproduct. In UAI, pp. 819–828, 2014.

URL https://www.columbia.edu/˜skk2175/
Papers/GOP-UAI.pdf.

Trockman, A., Willmott, D., and Kolter, J. Z. Understanding the covariance structure of convolutional filters. arXiv preprint arXiv:2210.03651, 2022.

Varma, V., Shah, R., Kenton, Z., Kramar, J., and Ku- ´
mar, R. Explaining grokking through circuit efficiency. International Conference on Learning Representations (ICLR), 2023. URL https://openreview.net/ pdf?id=7Zbg38nA0J.

Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B.,
Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D., Chi, E. H., Hashimoto, T., Vinyals, O., Liang, P., Dean, J., and Fedus, W. Emergent abilities of large language models. Transactions on Machine Learning Research (TMLR), 2022. URL https: //openreview.net/pdf?id=yzkSU5zdwD.

Yuan, G., Xu, M., Kpotufe, S., and Hsu, D. Efficient estimation of the central mean subspace via smoothed gradient outer products. *arXiv preprint arXiv:2312.15469*, 2023. URL https://arxiv.org/pdf/2312.15469.

Zhong, Z., Liu, Z., Tegmark, M., and Andreas, J. The clock and the pizza: Two stories in mechanistic explanation of neural networks. *Advances in Neural Information* Processing Systems, 36, 2024.

Zhu, L., Liu, C., Radhakrishnan, A., and Belkin, M. Catapults in sgd: spikes in the training loss and their impact on generalization through feature learning. International Conference on Machine Learning (ICML), 235, 2024.

Algorithm 1 Recursive Feature Machine (RFM) (Radhakrishnan et al., 2024a) Require: *X, y, k, T, L ▷* Train data: (*X, y*), base kernel: k, iters.: T, matrix power: s, and bandwidth: L
M0 = Id for t = 0*, . . . , T* − 1 do Solve α ← k(*X, X*; Mt)
−1*y ▷ f*(t)(x) = k(x, X; Mt)α Mt+1 ← [G(f
(t))]s end for return *α, M*T −1 ▷ Solution to kernel regression: α, and feature matrix: MT −1

## A. Additional Discussion

Low rank learning. The problem of learning modular arithmetic can be viewed as a type of matrix completion - completing the p × p matrix (so-called Cayley table) representing modular operations, from partial observations. The best studied matrix completion problem is low rank matrix completion, where the goal is to fill in missing entries of a low rank matrix from observing a subset of the entries (Moitra, 2018, Ch.8). While many specialized algorithms exist, it has been observed that neural networks can recover low rank matrix structures (Gunasekar et al., 2017). Notably, in a development paralleling the results of this paper, low-rank matrix completion can provably be performed by linear RFMs using the same AGOP mechanism (Radhakrishnan et al., 2024b).

It is thus tempting to posit that grokking modular operations in neural networks or RFM can be explained as a low rank prediction problem. Indeed modular operations can be implemented by an index 4 model, i.e., a function of the form f = g(Ax), where x ∈ R
2pand A is a rank 4 matrix (see Appendix K for the construction). It is a plausible conjecture as there is strong evidence, empirical and theoretical, that neural networks are capable of learning such multi-index models (Damian et al., 2022; Mousavi-Hosseini et al., 2022) as well as low-rank matrix completion. Furthermore, a phenomenon similar to grokking was discussed in (Radhakrishnan et al., 2022, Fig. 5, 6) in the context of low rank feature learning for both neural networks and RFM. However, despite the existence of generalizeable low rank models, the actual circulant features learned by both Neural Networks and RFM are not low rank. Interestingly, this observation mirrors the problem of learning parity functions through neural network inspired minimum norm interpolation, which was analyzed in (Ardeshir et al., 2023). While single-directional (index one) solutions exist in that setting, the authors show that the minimum norm solutions are all multi-dimensional. Explanations for deep learning Finally, this work adds to the growing body of evidence that the AGOP-based mechanisms of feature learning can account for some of the most interesting phenomena in deep learning. These include generalization with multi-index models (Parkinson et al., 2023), deep neural collapse (Beaglehole et al., 2024b), and the ability to perform low-rank matrix completion (Radhakrishnan et al., 2024b). Thus, RFM provides a framework that is both practically powerful and serves as a theoretically tractable model of deep learning.

## B. Additional Preliminaries

For completeness we replicate the algorithm definition for Recursive Feature Machines (RFM) provided by Radhakrishnan et al. (2024a) in Algorithm 1. This procedure recursively fits a kernel estimator for a chosen base kernel, k, then updates the feature matrix, M, by computing a matrix power of the Average Gradient Outer Product (AGOP) for that estimator. The algorithm terminates after a total of T iterations. The final estimator and feature matrix are then returned by the algorithm.

## C. Neural Feature Ansatz

While the NFA has been observed generally across depths and architecture types (Radhakrishnan et al., 2024a; Beaglehole et al., 2023; 2024a), we restate this observation for fully-connected networks with one hidden-layer of the form f(x) = a
⊤ϕ(W1x).

Ansatz 1 (Neural Feature Ansatz for one hidden layer). *For a one hidden-layer neural network* f NN *and a matrix power* s ∈ (0, 1]*, the following holds:*

$$W_{1}^{\top}W_{1}\propto G(f^{\mathrm{NN}})^{s}\ .$$
s. (15)
Note that this statement implies that W⊤
1 W1 and G(f NN)
s have a cosine similarity of ±1.

$\left(15\right)^2$
In this work, we choose α =
1 2
, following the main results in (Radhakrishnan et al., 2024a). While the absolute value of the cosine similarity is written in Eq. (15) to be 1, it is typically a high value less than 1, where the exact value depends on choices of initialization, architecture, dataset, and training procedure. For more understanding of these conditions, see (Beaglehole et al., 2024a).

## D. Model And Training Details

Gaussian kernel: Throughout this work we take bandwidth L = 2.5 when using the Mahalanobis Gaussian kernel. We solve ridgeless kernel regression using NumPy on a standard CPU.

Neural networks: Unless otherwise specified, we train one hidden layer neural networks with quadratic activation functions and no biases in PyTorch on a single A100 GPU. Models are trained using AdamW with hidden width 1024, batch size 32, learning rate of 10−3, weight decay 1.0, and standard PyTorch initialization. All models are trained using the Mean Squared Error loss function (square loss).

For the experiments in Appendix Fig. 8, we train one hidden layer neural networks with quadratic activation and no biases on modular addition modulo p = 61. We use 40% training fraction, PyTorch standard initialization, hidden width of 512, weight decay 10−5, and AGOP regularizer weight 10−3. Models are trained with vanilla SGD, batch size 128, and learning rate 1.0.

## E. Reordering Feature Matrices By Group Generators

Our reordering procedure uses the standard fact of group theory that the multiplicative group Z
∗pis a cyclic group of order p − 1 (Koblitz, 1994). By definition of the cyclic group, there exists at least one element g ∈ Z
∗p, known as a *generator*,
such that Z
∗p = {g i; i ∈ {1*, . . . , p* − 1}}.

Given a generator g ∈ Z
∗p, we reorder features according to the map, ϕg : Z
∗p → Z
∗p, where if h = g i, then ϕg(h) = i.

In particular, given a matrix B ∈ R
p×p, we reorder the bottom right (p − 1) × (p − 1) sub-block of B as follows: we move the entry in coordinate (r, c) with *r, c* ∈ Z
∗pto coordinate (ϕg(r), ϕg(c)). For example if g = 2 in Z
∗5, then (2, 3)
entry of the sub-block would be moved to coordinate (1, 3) since 2 1 = 2 and 2 3 mod 5 = 3. In the setting of modular multiplication/division, the map ϕg defined above is known as the *discrete logarithm* base g (Koblitz, 1994, Ch.3). The discrete logarithm is analogous to the logarithm defined for positive real numbers in the sense that it converts modular multiplication/division into modular addition/subtraction. Lastly, in this setting, we note that we only reorder the bottom
(p − 1) × (p − 1) sub-block of B as the first row and column are 0 (as multiplication by 0 results in 0).

Upon re-ordering the p × p off-diagonal sub-blocks of the feature matrix by the map ϕg, the feature matrix of RFM for multiplication/division tasks contains circulant blocks as shown in Fig. 3C. Thus, the reordered feature matrices for these tasks also exhibit the structure in Observation 1. As a remark, we note that there can exist several generators for a cyclic group, and thus far, we have not specified the generator g we use for re-ordering. For example, 2 and 3 are both generators of Z
∗5since {2, 2 2,(23 mod 5),(24 mod 5)} = {3,(32 mod 5),(33 mod 5),(34 mod 5)} = Z
∗5. Lemma J.1 implies that the choice of generator does not matter for observing circulant structure. As a convention, we simply reorder by the smallest generator.

## F. Enforcing Circulant Structure In Rfm

We see that the structure in Observation 1 gives generalizing features on modular arithmetic when the circulant C is constructed from the RFM matrix. We observe that enforcing this structure at every iteration, and comparing to the standard RFM model at that iteration, improves test loss and accelerates grokking on e.g. addition (Appendix Fig. 3). The exact procedure to enforce this structure is as follows. We first perform standard RFM to generate feature matrices M1*, . . . , M*T .

Then for each iteration of the standard RFM, we construct a new Mft on which we solve ridgeless kernel regression for a new α and evaluate on the test set. To construct Mf, we take D = **diag** (Mt) and first let Mf = D−1/2MD−1/2, to ensure the rows and columns have equal scale. We then reset the top left and bottom right sub-matrices of Mf as I −
1 p 11T, and replace the bottom-left and top-right blocks with C and C
⊤, where C is an exactly circulant matrix constructed from Mt.

Specifically, where c is the first column of the bottom-left sub-matrix of Mt, column ℓ of C is equal to σ ℓ(Mt).

## G. Grokking Multiple Tasks

Throughout the main paper, we focused on modular arithmetic settings for a single task. In more general domains such as language, one may expect there to be many "skills" that need to be learned. In such settings, it is possible that these skills are grokked at different rates. While a full discussion is beyond the scope of this work, to illustrate this behavior, we performed additional experiments in here, where we train RFM on a pair of modular arithmetic tasks simultaneously and demonstrate that different tasks are indeed grokked at different points throughout training.

We train RFM to simultaneously solve the following two modular polynomial tasks: (1) x + y mod p ; (2) x 2 + y 2 mod p for modulus p = 61. We train RFM with the Mahalanobis Gaussian kernel using bandwidth parameter L = 2.5. Training data for both tasks is constructed from the same 80% training fraction. In addition to concatenating the one-hot encodings for *x, y*, we also append an extra bit indicating which task to solve (0 indicating task (1) and 1 indicating task (2)). The classification head is shared for both tasks (e.g. output dimension is still R
p).

In Appendix Fig. 4, we observe that there are two sharp transitions in the test loss and test accuracy. By decomposing the loss into the loss per task, we observe that RFM groks task (1) prior to grokking task (2). Overall, these results illustrate that grokking of different tasks can occur at different training iterations.

## H. Agop Regularization And Weight Decay For Grokking Modular Arithmetic.

It has been argued in prior work that weight decay (ℓ2 regularization on network weights) is necessary for grokking to occur when training neural networks for modular arithmetic tasks (Varma et al., 2023; Davies et al., 2023; Nanda et al., 2023).

Under the NFA (Eq. (15)), which states that W⊤
1 W1 is proportional to a matrix power of G(f), we expect that performing weight decay on the first layer, i.e., penalizing the loss by ∥W1∥
2F = tr(W⊤
1 W1), should behave similarly to penalizing the trace of the AGOP, tr(G(f)), during training.3 To this end, we compare the impact of using (1) no regularization; (2)
weight decay; and (3) AGOP regularization when training neural networks on modular arithmetic tasks. In Appendix Fig. 8, we find that, akin to weight decay, AGOP regularization leads to grokking in cases where using no regularization results in no grokking and poor generalization. These results provide further evidence that neural networks solve modular arithmetic by using the AGOP to learn features.

## I. Fma Example For P = 3

We now provide an example of the FMA for p = 3. Let x = e1 ⊕ e2. In this case, we expect the FMA to output the vector e0 since (1 + 2) mod 3 = 0. Following the first step of the FMA, we compute

$$\widehat{x}_{[1]}=F\mathbf{e}_{1}=\frac{1}{\sqrt{3}}[1,\omega,\omega^{2}]^{\top}\;\;;\;\;\widehat{x}_{[2]}=F\mathbf{e}_{2}=\frac{1}{\sqrt{3}}[1,\omega^{2},\omega^{4}]^{\top}\;,$$

which are the first and second columns of F, respectively. Then their element-wise product is given by

$$(16)$$
$$F\mathbf{e}_{1}\odot F\mathbf{e}_{2}={\frac{1}{3}}[1,\omega^{3},\omega^{6}]^{\top}={\frac{1}{3}}[1,1,1]^{\top}={\frac{1}{\sqrt{3}}}F\mathbf{e}_{0}\;,$$
Fe0 , (17)
which is √
1 3 times the first column of the DFT matrix. Finally, we compute the outputs 
√3 D√
1 3 Fe0, Feℓ E
C

for each ℓ ∈ {0, 1, 2}. As F is unitary, yadd(e1 ⊕ e2; ℓ) = 1{1+2=ℓ mod 3}, so that coordinate 0 of the output will have value 1, and all other coordinates have value 0.

## J. Additional Results And Proofs

Lemma J.1. Let C ∈ R
p×p with its first row and column entries all equal to 0. Let the (p − 1) × (p − 1) sub-block starting at the second row and column be C
×*. Then,* C
× *is either circulant after re-ordering by any generator* q of Z
∗p
, or C
× *is not* circulant under re-ordering by any such generator.

Proof of Lemma *J.1.* We prove the lemma by showing that for any two generators q1, q2 of Z
∗p
, if C
× is circulant re-ordering with q1, then it is also circulant when re-ordering by q2.

$$(17)$$

Suppose C
× is circulant re-ordering with q1. Let i, j ∈ {1*, . . . , p* − 1}. Note that by the circulant assumption, for all s ∈ Z,

$$C_{q_{1}^{i},q_{1}^{j}}=C_{q_{1}^{i+s},q_{1}^{i+s}}\ ,$$
$$(18)$$
1, (18)
where we take each index modulo p.

As q2 is a generator for Z
∗p, we can access all entries of C
× by indexing with powers of q2. Further, as q1 is a generator, we can write q2 = q k 1, for some power k. Let a ∈ Z. Then,

$$\begin{array}{r l}{{C_{q_{2}^{i},q_{2}^{j}}=C_{q_{1}^{k i},q_{1}^{k j}}}}\\ {{}}&{{=C_{q_{1}^{k i+k a},q_{1}^{k j+k a}}}}\\ {{}}&{{=C_{q_{1}^{k(i+a)},q_{1}^{k(j+a)}}}}\\ {{}}&{{=C_{q_{2}^{i+a},q_{2}^{j+a}}\ .}}\end{array}$$

Therefore, C is constant on the diagonals under re-ordering by q2, concluding the proof.

We next state Lemma J.2, which is used in the proof of Theorem 5.1.

Lemma J.2 (See, e.g., (Gray et al., 2006)). Circulant matrices U *can be written (diagonalized) as:*

$\square$
$$U=F D{\bar{F}}^{\top}\ ,$$
where F is the DFT matrix, F¯⊤ *is the element-wise complex conjugate of* F
⊤ (i.e. the Hermitian of F), and D is a diagonal matrix with diagonal 
√p · F u, where u *is the first row of* U.

We now present the proof of Theorem 5.1, restating the theorem below for the reader's convenience.

Theorem. *Given all of the discrete data* ea ⊕ eb, e(a−b) mod p p−1 a,b=0 *in modular subtraction task, for each output class* ℓ ∈ {0, · · · , p − 1}*, we train a separate kernel predictor* fℓ(x) = k(*x, X*; Mℓ)α
(ℓ). Here k(·, ·; Mℓ) is a quadratic kernel with Mℓ =
0 C
ℓ
(C
ℓ)
⊤ 0 and C ∈ R
p×pis a circulant matrix with first row e1*. When* α
(ℓ)*is the solution to kernel* ridgeless regression for each ℓ*, the kernel predictor* f = [f0, . . . , fp−1] is equivalent to Fourier Multiplication Algorithm for modular subtraction (Eq. (14)).

Proof of Theorem *5.1.* We present the proof for modular subtraction as the proof for addition follows analogously. We write the standard kernel predictor for class ℓ on input x = x[1] ⊕ x[2] ∈ R
2pas,

$$f_{\ell}(x)=\sum_{a,b=0}^{p-1}\alpha_{a,b}^{(\ell)}k\left(x,e_{a}\oplus e_{b};M_{\ell}\right)\;,$$

where we have re-written the index into kernel coefficients for class ℓ, α
(ℓ) ∈ R
p×p, so that the coefficients are multi-indexed by the first and second digit. Specifically, now α
(ℓ)
a,b is the kernel coefficient corresponding to the representer k(·, x) for input point x = ea ⊕ eb. Recall we use a quadratic kernel, k(*x, z*; Mℓ) = (x
⊤Mℓz)
2. In this case, the kernel predictor simplifies to,

$$f_{\ell}(x)=\sum_{a,b=0}^{p-1}\alpha_{a,b}^{(\ell)}\left(x_{[1]}^{\top}C^{\ell}\mathbf{e}_{b}+\mathbf{e}_{a}^{\top}C^{\ell}x_{[2]}\right)^{2}\;.$$

Then, the labels for each pair of input digits, written as a matrix Y
(ℓ) ∈ R
p×pfor the ℓ-th class where the row and column index the first and second digit respectively, are Y
(ℓ) = C
−ℓ.

For x = ea′ ⊕ eb
′ , i.e. x in the discrete dataset, we have,

fℓ(x) = X p−1 a,b=0 α (ℓ) a,b δ(a,b′−ℓ) + δ(a′,b−ℓ) + 2δ(a,b′−ℓ)δ(a′,b−ℓ) = e ⊤ b ′−ℓα (ℓ)1 + 1 ⊤α (ℓ)ea′+ℓ + 2e ⊤ b ′−ℓα (ℓ)ea′+ℓ = e ⊤ b ′C −ℓα (ℓ)1 + 1 ⊤α (ℓ)C −ℓea′ + 2e ⊤ b ′C −ℓα (ℓ)C −ℓea′ = e ⊤ b ′C −ℓα11⊤ + 11⊤αC−ℓ + 2C −ℓαC−ℓea′ ,
where δ(u,v) = 1{u=v}. Let fℓ(X) ∈ R
p×p be the matrix of function values of fℓ, where [fℓ(X)]a,b = fℓ(ea ⊕ eb), and, therefore, fℓ(ea ⊕ eb) = e
⊤ a fℓ(X)eb. Then, to solve for α
(ℓ), we need to solve the system of equations for α,

$$\begin{array}{l}{{f_{\ell}(X)=\left(C^{-\ell}\alpha{\bf11}^{\top}+{\bf11}^{\top}\alpha C^{-\ell}+2C^{-\ell}\alpha C^{-\ell}\right)^{\top}=C^{-\ell}}}\\ {{\iff C^{-\ell}\alpha{\bf11}^{\top}+{\bf11}^{\top}\alpha C^{-\ell}+2C^{-\ell}\alpha C^{-\ell}=C^{\ell}}}\end{array}$$

Note, by left-multiplying both sides by C
−ℓ, we see this equation holds iff,

$$C^{-2\ell}\alpha{\bf11}^{\top}+{\bf11}^{\top}\alpha C^{-\ell}+2C^{-2\ell}\alpha C^{-\ell}=I\;.$$

Note the solution is unique as the kernel matrix is full rank. We posit the solution α such that C
−2ℓαC−ℓ =
1 2 I + λ11⊤,
which is α =
1 2C
3ℓ + λ11⊤. Then, solving for λ, we require,

$${\bf11}^{\top}+2p\lambda{\bf11}^{\top}+2\lambda{\bf11}^{\top}=0\;,$$

which implies λ = −2 2p+2 . Substituting this value of λ and simplifying, we see finally that fℓ(x) = x
⊤
[1]C
−ℓx[2]. Therefore, using that circulant matrices are diagonalized by C =
√*pF DF*¯⊤ (Lemma J.2) and F¯⊤F = I, where D = **diag** (Fe1), we derive,

$$\begin{array}{r l}{f_{\ell}(x)={\sqrt{p}}\cdot x_{[1]}^{\top}F D^{-\ell}{\bar{F}}^{\top}x_{[2]}}\\ {}&{{}={\sqrt{p}}\cdot x_{[1]}^{\top}F\mathbf{diag}\left(F e_{p-\ell-1}\right){\bar{F}}^{\top}x_{[2]}}\\ {}&{{}={\sqrt{p}}\cdot{\left\langle F x_{[1]}\circ F e_{p-\ell-1},F x_{[2]}\right\rangle}_{\mathbb{C}}}\end{array}$$

which is the output of the FMA on modular subtraction.

## K. Low Rank Solution To Modular Arithmetic

Addition We present a solution to the modular addition task whose AGOP is low rank, in contrast to the full rank AGOP recovered by RFM and neural networks.

We define the "encoding" map Φ : R
p → C as follows. For a vector a = [a0*, . . . , a*p−1],

$$\Phi(\mathbf{a})=\sum_{k=0}^{p-1}a_{k}\exp\left({\frac{k2\pi i}{p}}\right)\ .$$
$$\square$$

Notice that Φ is a linear map such that Φ(ek) = exp k2πi p
. Notice also that Φ is partially invertible with the "decoding" map Ψ : C → R
p.

$$\Psi(z)=\widehat{\max}\left(\left\langle z,\exp\left(\frac{0\cdot2\pi i}{p}\right)\right\rangle,\ldots\left\langle z,\exp\left(\frac{(p-1)\cdot2\pi i}{p}\right)\right\rangle\right)\.$$  In that makes all entries zero except for the largest one and the inner product is 
Above max 
g is a function that makes all entries zero except for the largest one and the inner product is the usual inner product
in C considered as R
2. Thus
$$\Psi\left(\exp\left(\frac{k\cdot2\pi i}{p}\right)\right)={\boldsymbol{e}}_{k}\ .$$
 = ek . (19)
Ψ is a nonlinear map C → R
p. While it is discontinuous but can easily be modified to make it differentiable.

$$(19)$$

By slight abuse of notation, we will define Φ : R
p × R
p → C
2 on pairs:

$\mathbf{a}$
Φ(ej , ek) = (Φ(ej ), Φ(ek)) .
This is still a linear map but now to C
2.

Consider now a quadratic map M on C
2 → C given by complex multiplication:

$$M(z_{1},z_{2})=z_{1}z_{2}\;.$$

It is clear that the composition ΨMΦ implements modular addition ΨMΦ(ej , ek) = e(j+k) mod p Furthermore, since Φ is a liner map to a four-dimensional space, the AGOP of the composition ΨMΦ is of rank 4.

Multiplication The construction is for multiplication is very similar with modifications which we sketch below. We first re-order the non-zero coordinates by the discrete logarithm with base equal to a generator of the multiplicative group eg (see Appendix E), while keeping the order of index 0. Then, we modify Φ to remove index a0 from the sum for inputs a. Thus for multiplication,

$$\Phi(\mathbf{a})=\sum_{k=1}^{p-1}a_{k}\exp\left({\frac{k\cdot2\pi i}{p-1}}\right)\;,$$

Hence that Φ(e0) = 0, Φ(eg) = exp 2πi p−1 and Φ(eg k ) = exp k·2πi p−1
. We extend Φ to R
p × R
pas in Eq. 19 above.

Note that Φ and the re-ordering together are still a linear map of rank 4. Then, the "decoding" map, Ψ(z), will be modified to return 0, when z = 0, and otherwise,

$\Psi(z)=q^{\rm max}(\langle z,{\rm exp}(\frac{0\cdot2\pi i}{p-1})\rangle,...\langle z,{\rm exp}(\frac{(p-2)\cdot2}{p-1})\rangle)$
p−1 )⟩).
M is still defined as above. It is easy to check that the composition of ΨMΦ with reordering implements modular multiplication modulo p and furthermore, the AGOP will also be of rank 4.

A
Loss s Accuracy
( )
15 10 15 15 20 010 0.016 0.016 LO SS
013 0.013 01 013 000 010 0.010 000 20 10 15 25 10 15 20 25 10 15
??

15 20 10 15 20 25 of est Loss o
  Correct
lutput Clas 1.00 1.00 1.00 1.00 0.75 0.75 0.75 0.73 0.50 0.50 0.50 0.50 10 15 20 25 30 10 15 20 25 30 10 15 20 25 10 15 20 25 30 RFM Iterations RFM Iterations RM Iterations RFM Iterations B
Progress Measures Circulant
Deviation 0.01 0.01 01 01 0.00 0.00 0.00 10 15 129 125 lo 15 20 25 10 15 20 25 10 15 20 25 S
5 gnment 1.00
0.75
0.50 1.00 1.00 0 25
0.50 0.50 0 - 20
0 - 80 0 75
0 sc 0.25 0.25 0.25 10 15 28 25 10 20 25 l&
10 15 25 15 20 25 10 RM Iterations RFM Iterations RFM Iterations RM Iterations Ad Sub Mul Div Iter 1 Iter 5 Iter 10 Iter 15 Mul
, reordered 100 0.010 est Accuracy 75 Test Loss 0.008 50 0,006 25 0.004 0 10 15 20 25 20 i 5 30 5 10 15 25 1 30 RFM Iterations RFM Iterations
- Obs 1 Structure Enforced
- RFM AGOP
Both Tasks:
Task 1:
Task 2:
+ y and x² + y²
x² + y²
x + y x 0.020 0.020 0.020 son
1sal 0.015 0.015 0.015 0.010 0.010 0.010 0.005 0.005 0.005 30 30 10 20 10 20 20 30 i i 1 10 1.00
0.75 1.00 1.00 Accuracy 0.75 0.75 0.50 0.50 0.50 0.25 0.00 0.25 0.00 0.00 10 20 30 20 30 10 20 30 10 i i i RFM Iterations RFM Iterations RFM Iterations