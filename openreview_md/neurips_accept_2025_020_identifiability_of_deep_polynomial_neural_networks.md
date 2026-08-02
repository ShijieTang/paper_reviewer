# Identifiability Of Deep Polynomial Neural Networks

Konstantin Usevich∗**, Ricardo Borsoi, Clara Dérand, Marianne Clausel**
Université de Lorraine, CNRS, CRAN
Nancy, F-54000, France firstname.lastname@univ-lorraine.fr†

## Abstract

Polynomial Neural Networks (PNNs) possess a rich algebraic and geometric structure. However, their identifiability—a key property for ensuring interpretabilityremains poorly understood. In this work, we present a comprehensive analysis of the identifiability of deep PNNs, including architectures with and without bias terms. Our results reveal an intricate interplay between activation degrees and layer widths in achieving identifiability. As special cases, we show that architectures with non-increasing layer widths are generically identifiable under mild conditions, while encoder-decoder networks are identifiable when the decoder widths do not grow too rapidly compared to the activation degrees. Our proofs are constructive and center on a connection between deep PNNs and low-rank tensor decompositions, and Kruskal-type uniqueness theorems. We also settle an open conjecture on the dimension of PNN's *neurovarieties*, and provide new bounds on the activation degrees required for it to reach the expected dimension.

## 1 Introduction

Neural network architectures which use polynomials as activation functions—*polynomial neural* networks (PNN)—have emerged as architectures that combine competitive experimental performance (capturing high-order interactions between input features) while allowing a fine grained theoretical analysis. On the one hand, PNNs have been employed in many problems in computer vision [1–3], image representation [4], physics [5] and finance [6], to name a few. On the other hand, the geometry of function spaces associated with PNNs, called *neuromanifolds*, can be analyzed using tools from algebraic geometry. Properties of such spaces, such as their dimension, shed light on the impact of a PNN architecture (layer widths and activation degrees) on the *expressivity* of feedforward, convolutional and self-attention PNN architectures [7–11]. They also determine the landscape of their loss function and the dynamics of their training process [7, 12, 13]. Moreover, PNNs are also closely linked to low-rank tensor decompositions [14–18], which play a fundamental role in the study of latent variable models due to their *identifiability* properties [19]. In fact, single-output 2-layer PNNs are equivalent to low-rank symmetric tensors [7]. Identifiabilitywhether the parameters and, consequently, the hidden representations of a NN can be determined from its response up to some equivalence class of trivial ambiguities such as permutations of its neurons—is a key question in NN theory [20–32]. Identifiability is critical to ensure interpretability in representation learning [33–35], to provably obtain disentangled representations [36], and in the study of causal models [37]. It is also critical to understand how the architecture affects the inference process and to support manipulation or "stitching" of pretrained models and representations [35, 38, 39]. Moreover, it has important links to learning and optimization of PNNs [40, 9, 13].

∗Corresponding author †In this version, the appendices have been reworked for better readability. Appendix E explains the changes between the submitted and the camera-ready version.

Identifiability of deep PNNs is intimately linked to the dimension of their so-called *neurovarieties*: when this dimension reaches the effective parameter count, the number of possible parametrizations is finite, which means the model is *finitely identifiable* and the neurovariety is said to be *non-defective*. In addition, many PNN architectures admit only a single parametrization (i.e., they are *globally* identifiable).This has been investigated for specific types of self-attention [9] and convolutional [8] layers, and feedforward PNNs without bias [11]. However, current results for feedforward networks only show that finite identifiability holds for very high activation degrees, or for networks with the same widths in every layer [11]. A standing conjecture is that this holds for any PNN with degrees at least quadratic and non-increasing layer widths [11], which parallels identifiability results of ReLU networks [29]. However, a general theory of identifiability of deep PNNs is still missing.

## 1.1 Our Contribution

We provide a comprehensive analysis of the identifiability of deep PNNs considering monomial activation functions. We prove that an L-layer PNN is finitely identifiable if every 2-layer block composed by a pair of two successive layers is finitely identifiable for some subset of their inputs. This surprising result tightly links the identifiability of shallow and deep polynomial networks, which is a key challenge in the general theory of NNs. Moreover, our results reveal an intricate interplay between activation degrees and layer widths in achieving identifiability. As special cases, we show that architectures with non-increasing layer widths (i.e., pyramidal nets) are generically identifiable, while encoder-decoder (bottleneck) networks are identifiable when the decoder widths do not grow too rapidly compared to the activation degrees. We also show that the minimal activation degrees required to render a PNN identifiable (which is equivalent to its *activation* thresholds) is only *linear* in the layer widths, compared to the quadratic bound in [11, Theorem 18]. These results not only settle but generalize conjectures stated in [11]. Moreover, we also address the case of PNNs with biases (which was overlooked in previous theoretical studies) by leveraging a homogenization procedure.

Our proofs are constructive and are based on a connection between deep PNNs and partially symmetric canonical polyadic tensor decompositions (CPD). This allows us to leverage Kruskal-type uniqueness theorems for tensors to obtain identifiability results for 2-layer networks, which serve as the building block in the proof of the finite identifiability of deep nets, which is performed by induction. Our results also shed light on the geometry of the *neurovarieties*, as they lead to conditions under which its dimension reaches the expected (maximum) value.

## 1.2 Related Works

Polynomial NNs: Several works studied PNNs from the lens of algebraic geometry using their associated *neuromanifolds* and *neurovarieties* [7] (in the emerging field of *neuroalgebraic geometry* [41]) and their close connection to tensor decompositions. Kileel et al. [7] studied the expressivity or feedforward PNNs in terms of the dimension of their neurovarieties. An analysis of the neuromanifolds for several architectures was presented in [10]. Conditions under which training losses do not exhibit bad local minima or spurious valleys were also investigated [13, 12, 42]. The links between training 2-layer PNNs and low-rank tensor approximation [13] as well as the biases of gradient descent [43] have been established. Recent work computed the dimensions of neuromanifolds associated with special types of selfattention [9] and convolutional [8] architectures, and also include identifiability results. For feedforward PNNs, finite identifiability was demonstrated for networks with the same widths in every layer [11], while stronger results are available for the 2-layer case with more general polynomial activations [44]. Finite identifiability also holds when the activation degrees are larger than a so-called activation degree threshold [11]. Recent work studied the singularities of PNNs with activations consisting of the sum of monomials with very high activation degrees [45]. PNNs are also linked to factorization machines [46]; this led to the development of efficient tensor-based learning algorithms [47, 48]. Note that other types of non-monomial polynomial-type activations [49, 50, 5, 51] have shown excellent performance; however, the geometry of these models is not well known. NN identifiability: Many studies focused on the identifiability of 2-layer NNs with tanh, odd, and ReLU activation functions [20–23]. Moreover, algorithms to learn 2-layer NNs with unique parameter recovery guarantees have been proposed (see, e.g., [52, 53]), however, their extension to NNs with 3 or more layers is challenging and currently uses heuristics [54]. Identifiability of deep NNs under weak genericity assumptions was first studied in the pioneering work of Fefferman [24] for the case of the tanh activation function through the study of its singularities. Recent work extended this result to more general sigmoidal activations [25, 26]. Various works focused on deep ReLU nets, which are piecewise linear [28]; they have been shown to be generically identifiable if the number of neurons per layer is non-increasing [29]. Recent work studied the local identifiability of ReLU nets [30–32]. Identifiability has also been studied for latent variable/causal modeling, leveraging different types of assumptions (e.g., sparsity, statistical independence, etc.) [55–60]. Note that although some of these works tackle deep NNs, their proof techniques are completely different from our approach and do not apply to the case of polynomial activation functions. Tensors and NNs: Low-rank tensor decompositions had widespread practical impact in the compression of NN weights [61–65]. Moreover, their properties also played a key role in the theory of NNs [18]. This includes the study of the expressivity of convolutional [66] and recurrent [67, 68] NNs, and the sample complexity of reinforcement learning parametrized by low-rank transition and reward tensors [69, 70]. The decomposability of low-rank symmetric tensors was also paramount in establishing conditions under which 2-layer NNs can (or cannot [71]) be learned in polynomial time and in the development of algorithms with identifiability guarantees [52, 72, 73]. It was also used to study identifiability of some deep *linear* networks [74]. However, the use of tensor decompositions in the studying the identifiability of deep *nonlinear* networks has not yet been investigated.

## 2 Setup And Background 2.1 Polynomial Neural Networks: With And Without Bias

Polynomial neural networks are functions R
d0 → R
dL represented as feedforward networks with bias terms and activation functions of the form ρr(·) = (·)
r. Our results hold for both the real and complex valued case (F = R, C), thus, and we prefer to keep the real notation for simplicity. Note that we allow the activation functions to have a different degree rℓ for each layer. Definition 1 (PNN). A **polynomial neural network (PNN) with biases** *and architecture* (d =
(d0, d1*, . . . , d*L), r = (r1, . . . , rL−1)) *is a map* R
d0 → R
dL *given by a feedforward neural network* PNNd,r[θ] = PNNr[θ] := fL ◦ ρrL−1
◦ fL−1 ◦ ρrL−2
◦ · · · ◦ ρr1
◦ f1 , (1)
where fi(x) = Wix + bi *are affine maps, with* Wi ∈ R
di×di−1 *being the weight matrices and* bi ∈ R
di*the biases, and the activation functions* ρr : R
d → R
d*, defined as* ρr(z) := (z r1*, . . . , z*rd)
are monomial. The parameters θ are given by the entries of the weights Wi and biases bi*, i.e.,*
θ = (w, b), w = (W1,W2*, . . . ,*WL), b = (b1, b2*, . . . ,* bL). (2)
The vector of degrees r *is called the* activation degree of PNNr[θ] (we often omit the subscript d if it is clear from the context).

PNNs are algebraic maps and are polynomial vectors, where the total degree is rtotal = r1 *· · ·* rL−1, that is, they belong to the polynomial space (Pd,r*total* )×dL , where Pd,r denotes the space of dvariate polynomials of degree ≤ r. Most previous works analyzed the simpler case of PNNs without bias, which we refer to as *homogeneous*. Due to its importance, we consider it explicitly.

Definition 2 (hPNN). A PNN is said to be a **homogenous** *PNN (hPNN) when it has no biases (*bℓ = 0 for all ℓ = 1, . . . , L*), and is denoted as* hPNNd,r[w] = hPNNr[w] := WL ◦ ρrL−1
◦ WL−1 ◦ ρrL−2
◦ · · · ◦ ρr1
◦ W1. (3)
Its parameter set is given by w = (W1,W2*, . . . ,*WL).

$$({\mathfrak{I}})$$
3
It is well known that such PNNs are in fact homogeneous polynomial vectors and belong to the polynomial space (Hd0,r*total* )×dL , where Hd,r ⊂ Pd,r denotes the space of homogeneous d-variate polynomials of degree r. hPNNs are also naturally linked to tensors and tensor decompositions, whose properties can be used in their theoretical analysis. Example 3 (Running example). *Consider an hPNN with* L = 2, r = (2) and d = (3, 2, 2). In such a case the parameter matrices are given as

$$W_{2}=\begin{bmatrix}b_{11}&b_{12}\\ b_{21}&b_{22}\end{bmatrix},\quad W_{1}=\begin{bmatrix}a_{11}&a_{12}&a_{13}\\ a_{21}&a_{22}&a_{23}\end{bmatrix},$$

and the hPNN p = hPNNr[w] *is a vector polynomial that admits the expression*

$$\mathbf{p}(\mathbf{x})=\mathbf{W}_{2}\rho_{2}(\mathbf{W}_{1}\mathbf{x})=\begin{bmatrix}b_{11}\\ b_{21}\end{bmatrix}\left(a_{11}x_{1}+a_{12}x_{2}+a_{13}x_{3}\right)^{2}+$$
b12
b22
(a21x1 + a22x2 + a23x3)
2.
the only monomials that can appear are of the form x i1x j 2x k3 *with* i + j + k = 2 thus p is a vector of degree-2 homogeneous polynomials in 3 *variables (in our notation,* p ∈ (H3,2)
2).

## 2.2 Equivalent Pnn Representations

It is known that the PNNs admit equivalent representations (i.e., several parameters θ leading to the same function). Indeed, for each hidden layer we can (a) permute the hidden neurons, and
(b) rescale the input and output to each activation function since for any a ̸= 0, (at)
r = a rt r.

These transformations lead to different sets of parameters that leave the PNN unchanged. We can characterize all such equivalent representations in the following lemma (provided in [7] for the case without biases).

Lemma 4. Let PNNd,r[θ] be a PNN with θ *as in* (2)*. Let also* Dℓ ∈ R
dℓ×dℓ be any invertible diagonal matrices and P ℓ ∈ Z
dℓ×dℓ(ℓ = 1, . . . , L − 1) be permutation matrices, and define the transformed parameters as W′ℓ ← P ℓDℓWℓD
−rℓ−1 ℓ−1 P
T
ℓ−1, b′ℓ ← P ℓDℓbℓ ,
with P 0 = D0 = I and P L = DL = I *by convention. Then the modified parameters* W′ℓ, b′ℓ define exactly the same network, i.e. PNNd,r[θ] = PNNd,r[θ′] *for the parameter vector* θ
′ = ((W′1,W′2*, . . . ,*W′L),(b
′1, b
′2*, . . . ,* b
′L)).

If θ and θ′*are linked with such a transformation, they are called equivalent (denoted* θ ∼ θ′). Example 5 (Example 3, continued). *In Example 3 we can take any* α, β ̸= 0 *to get*

$$\mathbf{\hat{h}}$$

hPNNd,r[w] = 
α−2b11 α−2b21
(αa11x1+αa12x2+αa13x3)
2+
β−2b12 β−2b22
(βa21x1+βa22x2+βa23x3)
2.

which correspond to rescaling rows of W1 and corresponding columns of W2. If we additionally permute them, we get W′1 = *P DW*1,W′2 = W2D−2P
T *with* D =-α 0 0 β and P = [ 0 1 1 0 ].

This characterization of equivalent representations allows us to define when a PNN is *unique*.

Definition 6 (Unique and finite-to-one representation). *The PNN* p = PNNd,r[θ] *(resp. hPNN* p = hPNNd,r[w]) with parameters θ (resp. w) is said have a **unique** representation if every other representation satisfying p = PNNd,r[θ′] *(resp.* p = hPNNd,r[w′]) is given by an equivalent set of parameters, i.e., θ′ ∼ θ (resp. w′ ∼ w) in the sense of Lemma 4 (i.e., they can be obtained from the permutations and elementwise scalings in Lemma 4).

Similarly, a PNN p = PNNd,r[θ] *(resp. hPNN* p = hPNNd,r[w]) is called **finite-to-one** if it admits only finitely many non-equivalent representations, that is, the set {θ
′: PNNd,r[θ
′] = p} *(resp.*
{w′: hPNNd,r[w′] = p}*) contains finitely many non-equivalent parameters.*
Example 7 (Example 5, continued). Thanks to links with tensor decompositions and their uniqueness, it is known that the hPNN in Example 3 has unique representation if W2 is invertible and W1 *full* row rank (rank 2*), see Proposition 35 in Section 4.2.*

## 2.3 Identifiability And Link To Neurovarieties

An immediate question is *which PNN/hPNN architectures are expected to admit only a single (or* finitely many) non-equivalent representations? This question can be formalized using the notions of global and **finite identifiability**, which considers a general set of parameters.

Definition 8 (Global and finite identifiability). *The PNN (resp. hPNN) with architecture* (d, r)
is said to be **globally identifiable** if for a general choice of θ = (w, b) ∈ R
Pdℓ(dℓ−1+1)*, (resp.*
w ∈ R
Pdℓdℓ−1) (i.e., for all choices of parameters except for a set of Lebesgue measure zero), the network PNNd,r[θ] *(resp.* hPNNd,r[w]) has a unique representation. Similarly, the PNN (resp. hPNN) with architecture (d, r) is said to be **finitely identifiable** if for a general choice of θ, (resp. w*) the network* PNNd,r[θ] *(resp.* hPNNd,r[w]) is finite-to-one (i.e., it admits only finitely many non-equivalent representations). In the following, we use the term "identifiable" to refer to finite identifiability unless stated otherwise. Note also that the notion of finite identifiability is much stronger than the related notion of local identifiability (i.e., a model being identifiable only in a neighborhood of a parameterization). Example 9 (Example 7, continued). *From Example 7, we see that the hPNN architecture with* d = (3, 2, 2), r = (2) is identifiable due to the fact that generic matrices W1 and W2 *are full rank.*
Note that Definition 8 excludes a set of parameters of Lebesgue measure zero. Thus, for an identifiable architecture such as the one mentioned in Example 9, there exists rare sets of pathological parameters for which the hPNN is non-unique (e.g., weight matrices containing collinear rows).

With some abuse of notation, let hPNNd,r[·] be the map taking w to hPNNd,r[w]. Then the image of hPNNd,r[·] is called a *neuromanifold*, and the *neurovariety* Vd,r is defined as its closure in the Zariski topology3. The study of neurovarieties and their properties is a topic of recent interest
[7, 41, 11, 10]. More details are given in Appendix A. An important property for our case is the link between identifiability of an hPNN, the dimension of its neurovariety, and the rank of its Jacobian.

Proposition 10. *The architecture* hPNNd,r[·] *is finitely identifiable if and only if the dimension* of Vd,r is equal to the effective number of parameters, i.e., dim Vd,r =PL
ℓ=1 dℓdℓ−1 −PL−1 ℓ=1 dℓ.

In such case, Vd,r is said to be **nondefective***. Equivalently, the rank of the Jacobian of the map* hPNNd,r[·] *is maximal and equal to* PL
ℓ=1 dℓdℓ−1 −PL−1 ℓ=1 dℓ *at a general parameter* w.

## 3 Main Results 3.1 Main Results On The Identifiability Of Deep Hpnns

Although several works have studied the identifiability of 2-layer NNs, tackling the case of deep networks is significantly harder. However, when we consider the opposite statement, i.e., the nonidentifiability of a network, it is much easier to show such connection: in a deep network with L > 2 layers, the lack of identifiability of any 2-layer subnetwork (formed by two consecutive layers) clearly implies that the full network is not identifiable. What our main result shows is that, surprisingly, under mild additional conditions the converse is also true for hPNNs: if the every 2-layer subnetwork is identifiable for some subset of their inputs, then the full network is identifiable as well. This is formalized in the following theorem.

Theorem 11 (Localization theorem). Let ((d0, . . . , dL),(r1, . . . , rL−1)) *be the hPNN format. For* ℓ = 0, . . . , L − 2 *denote* deℓ := min{d0, . . . , dℓ}*. Then the following holds true: if for all* ℓ =
1, . . . , L − 1 *the two-layer architecture* hPNN(deℓ−1,dℓ,dℓ+1),rℓ
[·] *is finitely identifiable, then the* L*-layer architecture* hPNNd,r[·] *is finitely identifiable as well.*
The technical proofs are relegated to the appendices. This key result shows a strong relation between the finite identifiability of shallow and deep hPNNs. However, as we move into the deeper layers, the identifiability conditions required by Theorem 11 are stricter than in the shallow case, since the number of inputs is reduced to deℓ. This can lead to a requirement of larger activation degrees to guarantee identifiability compared to the shallow case. Theorem 11 allows us to derive identifiability conditions for hPNNs using the link between 2-layer hPNNs and partially symmetric tensor decompositions and their generic uniqueness based on classical Kruskal-type conditions. We use the following sufficient condition for the identifiability of shallow networks.

Proposition 12 (Sufficient condition for identifiability of 2-layer hPNN). Let d0, d1 ≥ 2, d2 ≥ 1 be the layer widths and r ≥ 2 *such that*

$$r\geq{\frac{2d_{1}-\operatorname*{min}(d_{2},d_{1})}{\operatorname*{min}(d_{1},d_{0})-1}}.$$
. (4)
Then the 2-layer hPNN with architecture ((d0, d1, d2), r) *is globally identifiable.*
Remark 13. If the above condition is satisfied for every 2-layer architecture ((deℓ−1, dℓ, dℓ+1), rℓ),
ℓ = 1, . . . , L − 1, then Theorem 11 implies that the L-layer hPNN is finitely identifiable for the L*-layer architecture* (d, r).

3i.e., the smallest algebraic variety that contains the image of the map hPNNd,r[·].

$$(4)$$

Remark 14. *Note that for the single output case* dL = 1*, Equation* (4) *means the activation degree* in the last layer must satisfy rL−1 ≥ 3, in contrast to rℓ ≥ 2 for *ℓ < L* − 1.

Remark 15 (Our bounds are constructive). *We note that the condition* (4) *for identifiability is not* the best possible (and can be further improved using much stronger results on generic uniqueness of decompositions, see e.g., [75, Corollary 37]). However, the bound (4) *is constructive, and we can use* standard polynomial-time tensor algorithms to recover the parameters of the 2-layer hPNN.

## 3.2 Implications For Specific Architectures

Proposition 12 has direct implications for the finite identifiability of several architectures of practical interest, including pyramidal and bottleneck networks, and for the activation thresholds of hPNNs, as shown in the following corollaries. Corollary 16 (Pyramidal hPNNs are always identifiable). The hPNNs with architectures containing non-increasing layer widths (except possibly the last layer), i.e., d0 ≥ d1 ≥ · · · dL−1 ≥ 2 and dL ≥ 1*, are finitely identifiable for any degrees satisfying*
(i) r1, . . . , rL−1 ≥ 2 if dL ≥ 2; or (ii) r1*, . . . , r*L−2 ≥ 2, rL−1 ≥ 3 if dL = 1.

Note that, due to the connection between the identifiability of hPNNs and the neurovarieties presented in Proposition 10, a direct consequence of Corollary 16 is that the neurovariety Vd,r has expected dimension. This settles a recent conjecture presented in [11, Section 4]. This implication is explained in detail in Appendix A. Instead of seeking conditions on the layer widths for a fixed (or minimal) degree, a complementary perspective is to determine what are the smallest degrees rℓ such that a given architecture d is finitely identifiable. Following the terminology introduced in [11], we refer to those values as the activation degree thresholds for identifiability of an hPNN. An upper bound is given in the following corollary:
Corollary 17 (Activation degree thresholds for identifiability). *For fixed layer widths* d =
(d0, . . . , dL) with dℓ ≥ 2, ℓ = 0, . . . , L − 1, the hPNNs with architectures (d,(r1*, . . . , r*L−1))
are finitely identifiable for any degrees satisfying rℓ ≥ 2dℓ − 1 .

Note that due to Proposition 10, the result in this corollary implies that the neurovariety Vd,rhas expected dimension. This means that (2dℓ − 1) is also a universal upper bound to the so-called activation thresholds for hPNN expressiveness introduced in [11]. The existence of such activation degree thresholds was conjectured in [7] and recently proved in [11, Theorem 18], but the for a quadratic in dℓ bound (the bound in Corollary 17 is *linear*). Remark 18 (Admissible layer sizes). The possible layer sizes in a deep network are tightly linked with the degree of the activation. For example, for rℓ = 2*, identifiability is impossible if* dℓ >
dℓ−1(dℓ−1+1)
2
(for general rℓ*, a similar bound* O(d rℓ ℓ−1) follows from a link with tensor decompositions [76]).

Therefore, to allow for larger layer widths, we need to have higher-degree activations.

It is enlightening to consider the admissible layer widths when taking into account the joint effect of layer widths and degrees. By doing this, Proposition 12 can be leveraged to yield identifiability conditions for the case of bottleneck networks, as illustrated in the following corollary.

Corollary 19 (Identifiability of bottleneck hPNNs). *Consider the "bottleneck" architecture with* d0 ≥ d1 ≥ · · · ≥ db ≤ db+1 ≤ *. . .* ≤ dL
and db ≥ 2. Suppose that r1, . . . , rb ≥ 2 *and that the decoder part satisfies* dℓ rℓ≤ db − 1 for ℓ ∈ {b + 1, . . . , L − 1}*. Then the bottleneck hPNN is finitely identifiable.*
This shows that encoder-decoder hPNNs architectures are identifiable under mild conditions on the layer widths and decoder degrees, providing a polynomial networks-based counterpart to previous studies that analyzed linear autoencoders [77, 78].

Note that the width of the bottleneck layer db constrains the entire decoder part of the architecture: the degrees rℓ, ℓ ≥ b are constrained according to the width db. The presence of bottlenecks has also been shown to affect the expressivity of hPNNs in [7, Theorem 19]: for db = 2d0 −2 there exists a number of layers L such that for rℓ ≥ 2 and d0 ≥ 2, the hPNN neurovariety is *non-filling* (i.e., its dimension never reaches that of the ambient space) for any choice of widths d1*, . . . , d*b−1, db+1, . . . , dL.

## 3.3 Pnns With Biases

The identifiability of general PNNs (with biases) can be studied via the properties of hPNNs. The simplest idea is *truncation* (i.e., taking only higher-order terms of the polynomials), which eliminates biases from PNNs. Such an approach was already taken in [44] for shallow PNNs with general polynomial activation, and is described in Appendix D.3. We will follow a different approach based on the well-known idea of **homogenization**: we transform a PNN to an equivalent hPNN with structured parameters keeping the information about biases at the expense of increasing the layer widths. Our key result is to show how this can be used to study the identifiability of PNNs with bias terms. The following correspondence is well-known. Definition 20 (Homogenization). There is a one-to-one mapping between polynomials in d *variables* of degree r *and homogeneous polynomials of the same degree in* d + 1 variables. We denote this mapping Pd,r → Hd+1,r by homog(·), and it acts as follows: for every polynomial p ∈ Pd,r, pe = homog(p) ∈ Hd+1,r (that is pe(x1, . . . , xd, xd+1)*) is the unique homogeneous polynomial in* d + 1 *variables such that* pe(x1*, . . . , x*d, 1) = p(x1*, . . . , x*d).

Example 21. For the polynomial p ∈ P2,2 in variables (x1, x2) *given by* p(x1, x2) = ax21 + bx1x2 + cx22 + ex1 + fx2 + g, its homogenization pe = homog(p) ∈ H3,2 in 3 *variables* (x1, x2, x3) is pe(x1, x2, x3) = ax21 + bx1x2 + cx22 + ex1x3 + fx2x3 + gx23, and we can verify that pe(1, x1, x2) = p(x1, x2).

Similarly, we extend homogenization to polynomial vectors, which gives the following.

Example 22. Let f(x) = W2ρr1(W1x + b1) + b2*, and define extended matrices as*

$$\widetilde{\mathbf{W}}_{1}=\begin{bmatrix}\mathbf{W}_{1}&\mathbf{b}_{1}\\ 0&1\end{bmatrix}\in\mathbb{R}^{(d_{1}+1)\times(d_{0}+1)},\quad\widetilde{\mathbf{W}}_{2}=[\mathbf{W}_{2}\quad\mathbf{b}_{2}]\in\mathbb{R}^{d_{2}\times(d_{1}+1)}.$$
$\simeq$ . 
Then its homogenization fe= homog(f) is an hPNN of format (d0 + 1, d1 + 1, d2)

$$\widetilde{f}(\widetilde{\mathbf{x}})=\widetilde{\mathbf{W}}_{2}\rho_{r_{1}}\left(\widetilde{\mathbf{W}}_{1}\widetilde{\mathbf{x}}\right)$$

where xe = [x0, x1, . . . , xd0, xd0+1]
T, so that fe(x1*, . . . , x*d0, 1) = f(x1*, . . . , x*d0).

The construction in Example 22 similar to the well-known idea of augmenting the network with an artificial (constant) input. The following proposition generalizes this example to the case of multiple layers, by "propagating" the constant input.

Proposition 23. *Fix the architecture* r = (r1, . . . , rL−1) and d = (d0, . . . , dL). Then a polynomial vector p ∈ (Pd0,rtotal )×dL *admits a PNN representation* p = PNNd,r[(w, b)] *with* (w, b) as in (2) *if and only if its homogenization* pe = homog(p) admits an hPNN decomposition for the same activation degrees r *and extended* de = (d0 + 1*, . . . , d*L−1 + 1, dL), pe = hPNNde,r
[we ],
we = (Wf1, . . . ,WfL)*, with matrices given as*

$\infty$. 
$=f(x_1,\ldots,x_d)$
$$\widetilde{\mathbf{W}}_{\ell}=\begin{cases}\begin{bmatrix}\mathbf{W}_{\ell}&\mathbf{b}_{\ell}\\ 0&1\end{bmatrix}\in\mathbb{R}^{(d_{\ell}+1)\times(d_{\ell-1}+1)},\quad\ell<L,\\ \begin{bmatrix}\mathbf{W}_{L}&\mathbf{b}_{L}\end{bmatrix}\in\mathbb{R}^{(d_{L})\times(d_{L-1}+1)},\quad\ell=L,\end{cases}$$
That is, PNNs are in one-to-one correspondence to hPNNs with increased number of inputs and structured weight matrices.

Uniqueness of PNNs from homogenization: An important consequence of homogenization is that the uniqueness of the homogenized hPNN implies the uniqueness of the original PNN with bias terms, which is a key result to support the application of our identifiability results to general PNNs.

Proposition 24. If hPNNr[we ] from Proposition 23 is unique (resp. finite-to-one) as an hPNN
(without taking into account the structure), then the original PNN representation PNNr[(w, b)] is unique (resp. finite-to-one). The proposition follows from the fact that we can always fix the permutation ambiguity for the
"artificial" input.

Remark 25. Despite the one-to-one correspondence, for generic properties (e.g., finite identifiability)
we cannot immediately apply the results from the homogeneous case, because the matrices Wfℓ are structured (they form a set of measure zero inside R
(dℓ+1)×(dℓ−1+1)).

However, we can prove that the identifiability of the hPNN implies the identifiability of the PNN.

Lemma 26. Let the 2*-layer hPNN architecture be finitely (resp. globally) identifiable for* ((d0 + 1, d1 + 1, d2), r1). Then the PNN architecture with widths (d0, d1, d2) and degree r1 is also finitely
(resp. globally) identifiable.

Using Lemma 26 and specializing the proof of Theorem 11, we obtain the following result:
Proposition 27. Let ((d0, . . . , dL),(r1, . . . , rL−1)) *be the PNN format. For* ℓ = 0*, . . . , L* − 2 denote deℓ = min{d0, . . . , dℓ}*. Then the following holds true: If for all* ℓ = 1, . . . , L − 1 *each* two-layer architecture hPNN(deℓ−1+1,dℓ+1,dℓ+1),rℓ
[·] is finitely identifiable, then the L-layer PNN
with architecture (d, r) *is finitely identifiable as well.*
In particular, we have the following bounds for generic uniqueness.

Corollary 28. Let ((d0, . . . , dL),(r1, . . . , rL−1)) be such that dℓ ≥ 1, and rℓ ≥ 2 *satisfy*

$$r_{\ell}\geq\frac{2(d_{\ell}+1)-\operatorname*{min}(d_{\ell}+1,d_{\ell+1})}{\operatorname*{min}(d_{\ell},\widetilde{d}_{\ell-1})},$$
min(dℓ, deℓ−1)
then the L-layer PNN with architecture (d, r) *is finitely identifiable (and globally identifiable if* L = 2). Remark 29. For general PNNs with bias, similar conclusions hold to the ones in the hPNN case.

In particular, for fixed layer widths dℓ ≥ 1*, the activation threshold for a PNN architecture* (d, r)
becomes rℓ ≥ 2dℓ + 1*. Also, pyramidal PNNs are identifiable in degree* 2.

A distinctive feature of PNNs with bias is that they can be identifiable even for architectures with layers containing a single hidden neuron: for dℓ = 1 and dℓ+1 ≥ 2 *and/or* deℓ−1 = 1, the condition in Corollary 28 is still satisfied when rℓ ≥ 2.

## 4 Proofs And Main Tools

Our main results in Theorem 11 translates the identifiability conditions of deep hPNNs into those of shallow hPNNs. Our results are strongly related to the decomposition of partially symmetric tensors (we review basic facts about tensors and tensors decompositions and recall their connection between to hPNNs in later subsections). More details are provided in the appendices, and we list key components of the proof below.

## 4.1 Identifiability Of Deep Pnns: Necessary Conditions

Increasing hidden layers breaks uniqueness. The key insight is that if we add to any architecture a neuron in any hidden layer, then the uniqueness of the hPNN is not possible, which is formalized as following lemma (whose proof is based, in its turn, on tensor decompositions).

Lemma 30. Let p = hPNNr[w] be an hPNN of format (d0, . . . , dℓ, . . . , dL)*. Then for any* ℓ there exists an infinite number of representations of hPNNs p = hPNNr[w] *with architecture*
(d0*, . . . , d*ℓ + 1, . . . , dL)*. In particular, the augmented hPNN is not unique (and is not finite-to-one).*
Internal features of a unique hPNN are linearly independent. This is an easy consequence of Lemma 30 (as linear dependence would allow for pruning neurons).

Lemma 31. For d = (d0, . . . , dL)*, let* p = hPNNr[w] have a unique (or finite-to-one) L-layers decomposition. Consider the output at any ℓ-th internal level ℓ < L *after the activations* qℓ(x) = ρrℓ◦ Wℓ *◦ · · · ◦* ρr1 ◦ W1(x). (5)
Then the elements of qℓ(x) = [qℓ,1(x) · · · qℓ,dℓ(x)]
T*are linearly independent polynomials.*
Identifiability for hPNNs and Kruskal rank. Identifiability of 2-layer hPNNs, or equivalently uniqueness of CPD is strongly related to the concept of Kruskal rank of a matrix that we define below.

Definition 32. The Kruskal rank of a matrix A *(denoted* krank{A}) is the maximal number k such that any k columns of A *are linearly independent.* This is in contrast with the usual rank, which is the maximal k *such that there exist* k linearly independent columns. Therefore krank{A} ≤ rank{A}. Note that krank{A} ≥ 2 means that none of the pairs of columns of A are linearly dependent (no columns are pairwise collinear). Using the notion of Kruskal rank, we can state a necessary condition on weight matrices for identifiability of hPNNs, which is a generalization of the well-known necessary condition for the uniqueness of CPD tensor decompositions (6) (i.e., shallow networks), and is a corollary of Lemma 30 and Lemma 31.

Proposition 33. *As in Lemma 31, let the widths be* d = (d0, . . . , dL)*, and* p = hPNNr[w] have a unique (or finite-to-one) L-layers decomposition. Then we have that for all ℓ = 1*, . . . , L* − 1 krank{WT
ℓ } ≥ 2, krank{Wℓ+1} ≥ 1, where krank{Wℓ+1} ≥ 1 simply means that Wℓ+1 *does not have zero columns.*

## 4.2 Shallow Hpnns And Tensor Decompositions

An order-s tensor T ∈ R
m1*×···×*msis an s-way multidimensional array (more details are provided in Appendix B.2 and more background on tensors can be found in [14–16]). It is said to have a d-term CPD (canonical polyadic decomposition) if it admits a decomposition into d rank-1 terms T =Pd j=1 a1,j ⊗ · · · ⊗ as,j for ai,j ∈ R
mi, with ⊗ being the tensor (outer) product. The CPD is also written compactly as T = [[A1, A2, *· · ·* , As]] for matrices Ai = [ai,1, · · · , ai,d] ∈ R
mi×d. T
is said to be (partially) *symmetric* if it is invariant to any permutation of (a subset) of its indices [79]. Concretely, we will consider tensors T partially symmetric on dimensions i ∈ {2*, . . . , s*}, with CPD
that is also partially symmetric, i.e., with Ai, i ≥ 2 satisfying A2 = A3 = *· · ·* = As. Our main proofs strongly rely on results of [7] on the connection between hPNN and tensors decomposition in the shallow (i.e., 2-layer) case (see also [79]). Proposition 34. *There is a one-to-one mapping between partially symmetric tensors* F ∈ R 
d2×d0×···×d0 and polynomial vectors f ∈ (Hd0,r)×d2*, which can be written as*

$$\mathbf{\Phi}_{+1}\}\geq1,$$
$${\mathcal{T}}\mapsto$$

F 7→ f(x) = F
(1)x⊗r,

with F
(1) ∈ R
d2×d
r
0 the first unfolding of F*. Under this mapping, the partially symmetric CPD*
$${\mathcal{F}}=[W_{2},W_{1}^{\mathsf{T}},\cdots,W_{1}^{\mathsf{T}}]$$
1]] (6)
is mapped to hPNN W2ρr(W1x)*. Thus, uniqueness of* hPNN(d0,d1,d2),r[(W1,W2)] *is equivalent*
to uniqueness of the partially symmetric CPD of F. Thanks to the link with the partially symmetric CPD, we prove the following Kruskal-based sufficient condition for uniqueness (which is a counterpart of Proposition 33).

Proposition 35. Let pw(x) = W2ρr1(W1x) be a 2-layer hPNN with layer sizes (d0, d1, d2)
satisfying d0, d1 ≥ 2, d2 ≥ 1*. Assume that* r ≥ 2, krank{W2} ≥ 1, krank{WT
1 } ≥ 2 *and that:*

$$r\geq{\frac{2d_{1}-\operatorname{krank}\{W_{2}\}}{\operatorname{krank}\{W_{1}^{\mathsf{T}}\}-1}}\,,$$

then the 2-layer hPNN pw(x) *is unique (or equivalently, the CPD of* F in (6) *is unique).*
Remark 36. *For 2-layer hPNNs (*L = 2), when the activation degree r is high enough Proposition 33 gives both necessary and sufficient conditions for uniqueness due to Proposition 35. Remark 37. Proposition 35 forms the basis of the proof of Proposition 12, which comes from the fact that the Kruskal rank of a generic matrix is equal to its smallest dimension. Remark 38. Proposition 35 is based on basic (Kruskal) uniqueness conditions [80–82]. As mentioned in Remark 15, by using more powerful results on generic uniqueness [83, 84], we can obtain better bounds for identifiability of 2-layer PNNs. For example, for "bottleneck" architectures (as in Corollary 19), the results of [83, Thm 1.11-12] imply that for degrees rℓ = 2, identifiability holds for decoder layer sizes satisfying a weaker condition dℓ ≤
(db−1)db 2*(instead of* dℓ rℓ
≤ db − 1).

## 4.3 Proof Of The Main Result

The proof of Theorem 11 proceeds by induction over the layers ℓ = 1*, . . . , L*. The key idea is based on a procedure that allows us to prove finite identifiability of the L-th layer given the assumption that the previous layers are identifiable. For this, we introduce a map (*last layer map*)
ψ[q,WL] := WLρrL−1(q(x1*, . . . , x*d0)), (7)
where q is the vector polynomial of degree R = r1 *· · ·* rL−2, representing the output of the (L−1)-th linear layer. Then the L-layer hPNN is a composition:
hPNNr[θ,WL] = ψ[hPNN(r1*,...,r*L−2)[θ],WL], for θ = (W1*, . . . ,*WL−1).

To obtain finite identifiability, we look at the Jacobian of the composite map. The key to this recursion is to show that the Jacobian Jψ(q,WL) (Jacobian of ψ with respect to the input polynomial vector and WL) is of maximal possible rank. For this, we construct a "certificate" of finite identifiability qb realized by hPNN(r1*,...,r*L−2)[θb], but of simpler structure which inherits identifiability of a shallow hPNN.

Remark 39. For dL = 1, maximality of the rank for Jψ(q,WL) is closely related to nondefectivity of the variety of sums of powers of forms, which is often proved by establishing Hilbert genericity of an ideal generated by the elements of q *(a question raised in Fröberg conjecture, see e.g., [85]).* A key limitation of our techniques is that they only allow for establishing finite identifiability for deep PNNs. There exist recent results linking finite and global identifiability, [75, 86] but only for additive decompositions (shallow case). We state, however, the following conjecture.

Conjecture 40. Under the assumptions of Theorem 11, the L*-layer hPNN is globally identifiable.*
Note that the conjecture may be valid only for global identifiability (i.e., for a generic choice of parameters) and not for uniqueness, since it is not true that the composition of unique shallow hPNNs yield a unique deep hPNNs, as shown by the following example.

Example 41. Consider two polynomials: p(x1, x2) = -(x 21 + x 22)
2(x 21 − x 22)
2T. We see that this polynomial vector admits two different representations

$$\mathbf{p}(\mathbf{x})=\mathbf{I}\rho_{2}(\mathbf{W}_{2}\rho_{2}(\mathbf{I}\mathbf{x}))=\mathbf{W}_{3}\rho_{2}\left({\frac{1}{2}}\mathbf{W}_{2}\rho_{2}(\mathbf{W}_{2}\mathbf{x})\right),$$
_with_  $$\mathbf{W}_{2}=\begin{bmatrix}1&1\\ 1&-1\end{bmatrix},\quad\mathbf{W}_{3}=\begin{bmatrix}1&0\\ 1&-1\end{bmatrix},$$  _which are not equivalent. However, each $2$-layer subnetwork is unique (see Example 7)._

## 5 Discussion

In this paper, we presented a comprehensive analysis of the identifiability of deep feedforward PNNs by using their connections to tensor decompositions. Our main result is the *localization of* identifiability, showing that deep PNNs are finitely identifiable if every 2-layer subnetwork is also finitely identifiable for a subset of their inputs. Our results can be also useful for compression (pruning) neural networks as they give an indication about the architectures that are not reducible. An important perspective is also to understand when two different identifiable PNN architectures can represent the same function, as the identifiable representations can potentially occur for different non-compatible formats (e.g., a PNN in format d = (2, 4, 4, 2) could be potentially pruned to two different identifiable representations, say, d = (2, 3, 4, 2) and d = (2, 4, 3, 2)). While our results focus on the case of monomial activations, we believe that this approach can be extended for establishing theoretical guarantees for other types of architectures and activation functions. In fact, the monomial case constitutes as a key first step in addressing general polynomial activations (see, e.g., [45]) which, in turn, can approximate most commonly used activations on compact sets. Moreover, the close connection between PNNs and partially symmetric tensor decompositions (which benefit from efficient computational algorithms based on linear algebra [87]) can also serve as support for the development of computational algorithms based on tensor decompositions for training deep PNNs. In fact, tensor decompositions have been combined with the method of moments to learn small NN architectures (see, e.g., [52, 88]), extending such approaches for training deep PNNs with finite datasets is an important direction for future work.

## Acknowledgments

This work was supported in part by the French National Research Agency (ANR) under grants ANR- 23-CE23-0024, ANR-23-CE94-0001, by the PEPR project CAUSALI-T-AI, and by the National Science Foundation, under grant NSF 2316420.

## References

[1] Grigorios G Chrysos, Stylianos Moschoglou, Giorgos Bouritsas, Yannis Panagakis, Jiankang Deng, and Stefanos Zafeiriou. P-nets: Deep polynomial neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7325–7335, 2020.

[2] Grigorios G Chrysos, Markos Georgopoulos, Jiankang Deng, Jean Kossaifi, Yannis Panagakis, and Anima Anandkumar. Augmenting deep classifiers with polynomial neural networks. In European Conference on Computer Vision, pages 692–716. Springer, 2022.

[3] Mohsen Yavartanoo, Shih-Hsuan Hung, Reyhaneh Neshatavar, Yue Zhang, and Kyoung Mu Lee.

Polynet: Polynomial neural network for 3D shape recognition with polyshape representation. In *International conference on 3D vision (3DV)*, pages 1014–1023. IEEE, 2021.

[4] Guandao Yang, Sagie Benaim, Varun Jampani, Kyle Genova, Jonathan T. Barron, Thomas Funkhouser, Bharath Hariharan, and Serge Belongie. Polynomial neural fields for subband decomposition and manipulation. In *Advances in Neural Information Processing Systems*, 2022. URL https://openreview.net/forum?id=juE5ErmZB61.

[5] Jie Bu and Anuj Karpatne. Quadratic residual networks: A new class of neural networks for solving forward and inverse problems in physics involving PDEs. In *Proceedings of the 2021* SIAM International Conference on Data Mining (SDM), pages 675–683. SIAM, 2021.

[6] Sarat Chandra Nayak and Bijan Bihari Misra. Estimating stock closing indices using a GA-
weighted condensed polynomial neural network. *Financial Innovation*, 4(1):21, 2018.

[7] Joe Kileel, Matthew Trager, and Joan Bruna. On the expressive power of deep polynomial neural networks. *Advances in neural information processing systems*, 32, 2019.

[8] Vahid Shahverdi, Giovanni Luca Marchetti, and Kathlén Kohn. On the geometry and optimization of polynomial convolutional networks. *AISTATS 2025*, 2025. arXiv preprint arXiv:2410.00722.

[9] Nathan W Henry, Giovanni Luca Marchetti, and Kathlén Kohn. Geometry of lightning selfattention: Identifiability and dimension. *ICLR 2025*, 2025. arXiv preprint arXiv:2408.17221.

[10] Kaie Kubjas, Jiayi Li, and Maximilian Wiesmann. Geometry of polynomial neural networks.

Algebraic Statistics, 15(2):295–328, 2024. arXiv:2402.00949.

[11] Bella Finkel, Jose Israel Rodriguez, Chenxi Wu, and Thomas Yahl. Activation degree thresholds and expressiveness of polynomial neural networks. *Algebraic Statistics*, 16(2):113–130, 2025. arXiv:2408.04569.

[12] Samuele Pollaci. Spurious valleys and clustering behavior of neural networks. In International Conference on Machine Learning, pages 28079–28099. PMLR, 2023.

[13] Yossi Arjevani, Joan Bruna, Joe Kileel, Elzbieta Polak, and Matthew Trager. Geometry and optimization of shallow polynomial networks. *arXiv preprint arXiv:2501.06074*, 2025.

[14] Tamara G Kolda and Brett W Bader. Tensor decompositions and applications. *SIAM review*, 51
(3):455–500, 2009.

[15] Nicholas D Sidiropoulos, Lieven De Lathauwer, Xiao Fu, Kejun Huang, Evangelos E Papalexakis, and Christos Faloutsos. Tensor decomposition for signal processing and machine learning.

IEEE Transactions on signal processing, 65(13):3551–3582, 2017.

[16] Andrzej Cichocki, Namgil Lee, Ivan Oseledets, Anh-Huy Phan, Qibin Zhao, Danilo P Mandic, et al. Tensor networks for dimensionality reduction and large-scale optimization: Part 1 low-rank tensor decompositions. Foundations and Trends® *in Machine Learning*, 9(4-5):249–429, 2016.

[17] Aditya Bhaskara, Moses Charikar, and Aravindan Vijayaraghavan. Uniqueness of tensor decompositions with applications to polynomial identifiability. In Conference on Learning Theory, pages 742–778. PMLR, 2014.

[18] Ricardo Borsoi, Konstantin Usevich, and Marianne Clausel. Low-rank tensor decompositions for the theory of neural networks. *IEEE Signal Processing Magazine*, 2026.

[19] Animashree Anandkumar, Rong Ge, Daniel Hsu, Sham M Kakade, and Matus Telgarsky. Tensor decompositions for learning latent variable models. *Journal of machine learning research*, 15: 2773–2832, 2014.

[20] Héctor J Sussmann. Uniqueness of the weights for minimal feedforward nets with a given input-output map. *Neural networks*, 5(4):589–593, 1992.

[21] Francesca Albertini and Eduardo D Sontag. For neural networks, function determines form.

Neural networks, 6(7):975–990, 1993.

[22] Francesca Albertini, Eduardo D Sontag, and Vincent Maillot. Uniqueness of weights for neural networks. *Artificial Neural Networks for Speech and Vision*, pages 115–125, 1993.

[23] Henning Petzka, Martin Trimmel, and Cristian Sminchisescu. Notes on the symmetries of 2layer ReLU-networks. In *Proceedings of the northern lights deep learning workshop*, volume 1, pages 1–6, 2020.

[24] Charles Fefferman. Reconstructing a neural net from its output. Revista Matemática Iberoamericana, 10(3):507–555, 1994.

[25] Verner Vlaciˇ c and Helmut Bölcskei. Affine symmetries and neural network identifiability. ´
Advances in Mathematics, 376:107485, 2021.

[26] Verner Vlaciˇ c and Helmut Bölcskei. Neural network identifiability for a family of sigmoidal ´
nonlinearities. *Constructive Approximation*, 55(1):173–224, 2022.

[27] Flavio Martinelli, Berfin ¸Sim¸sek, Wulfram Gerstner, and Johanni Brea. Expand-and-cluster:
parameter recovery of neural networks. In *Proceedings of the 41st International Conference on* Machine Learning, pages 34895–34919, 2024.

[28] David Rolnick and Konrad Kording. Reverse-engineering deep ReLU networks. In International Conference on Machine Learning, pages 8178–8187. PMLR, 2020.

[29] Phuong Bui Thi Mai and Christoph Lampert. Functional vs. parametric equivalence of ReLU
networks. In *8th International Conference on Learning Representations*, 2020.

[30] Pierre Stock and Rémi Gribonval. An embedding of ReLU networks and an analysis of their identifiability. *Constructive Approximation*, pages 1–47, 2022.

[31] Joachim Bona-Pellissier, François Malgouyres, and François Bachoc. Local identifiability of deep ReLU neural networks: the theory. *Advances in neural information processing systems*,
35:27549–27562, 2022.

[32] Joachim Bona-Pellissier, François Bachoc, and François Malgouyres. Parameter identifiability of a deep feedforward ReLU neural network. *Machine Learning*, 112(11):4431–4493, 2023.

[33] Sébastien Lachapelle, Pau Rodriguez, Yash Sharma, Katie E Everett, Rémi Le Priol, Alexandre Lacoste, and Simon Lacoste-Julien. Disentanglement via mechanism sparsity regularization: A
new principle for nonlinear ICA. In *First Conference on Causal Learning and Reasoning*, 2021.

[34] Quanhan Xi and Benjamin Bloem-Reddy. Indeterminacy in generative models: Characterization and strong identifiability. In *International Conference on Artificial Intelligence and Statistics*,
pages 6912–6939. PMLR, 2023.

[35] Charles Godfrey, Davis Brown, Tegan Emerson, and Henry Kvinge. On the symmetries of deep learning models and their internal representations. Advances in Neural Information Processing Systems, 35:11893–11905, 2022.

[36] Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Raetsch, Sylvain Gelly, Bernhard Schölkopf, and Olivier Bachem. Challenging common assumptions in the unsupervised learning of disentangled representations. In *International conference on machine learning*, pages 4114–
4124. PMLR, 2019.

[37] Aneesh Komanduri, Xintao Wu, Yongkai Wu, and Feng Chen. From identifiable causal representations to controllable counterfactual generation: A survey on causal generative modeling. *Transactions on Machine Learning Research*, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=PUpZXvNqmb.

[38] Akira Ito, Masanori Yamada, and Atsutoshi Kumagai. Linear mode connectivity between multiple models modulo permutation symmetries. In Forty-second International Conference on Machine Learning, 2025.

[39] Samuel Ainsworth, Jonathan Hayase, and Siddhartha Srinivasa. Git re-basin: Merging models modulo permutation symmetries. In The Eleventh International Conference on Learning Representations, 2023.

[40] Sumio Watanabe. *Algebraic geometry and statistical learning theory*, volume 25. Cambridge university press, 2009.

[41] Giovanni Luca Marchetti, Vahid Shahverdi, Stefano Mereta, Matthew Trager, and Kathlén Kohn. Position: Algebra unveils deep learning - an invitation to neuroalgebraic geometry. In Forty-second International Conference on Machine Learning Position Paper Track, 2025. URL
https://openreview.net/forum?id=mzc1KPkIMJ.

[42] Abbas Kazemipour, Brett W Larsen, and Shaul Druckmann. Avoiding spurious local minima in deep quadratic networks. *arXiv preprint arXiv:2001.00098*, 2019.

[43] Moulik Choraria, Leello Tadesse Dadi, Grigorios Chrysos, Julien Mairal, and Volkan Cevher.

The spectral bias of polynomial neural networks. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=P7FLfMLTSEX.

[44] Pierre Comon, Yang Qi, and Konstantin Usevich. Identifiability of an X-rank decomposition of polynomial maps. *SIAM Journal on Applied Algebra and Geometry*, 1(1):388–414, 2017.

[45] Vahid Shahverdi, Giovanni Luca Marchetti, and Kathlén Kohn. Learning on a razor's edge: the singularity bias of polynomial neural networks. *arXiv preprint arXiv:2505.11846*, 2025.

[46] Steffen Rendle. Factorization machines. In *IEEE International conference on data mining*,
pages 995–1000. IEEE, 2010.

[47] Mathieu Blondel, Masakazu Ishihata, Akinori Fujino, and Naonori Ueda. Polynomial networks and factorization machines: New insights and efficient training algorithms. In International Conference on Machine Learning, pages 850–858. PMLR, 2016.

[48] Mathieu Blondel, Vlad Niculae, Takuma Otsuka, and Naonori Ueda. Multi-output polynomial networks and factorization machines. *Advances in Neural Information Processing Systems*, 30, 2017.

[49] Li-Ping Liu, Ruiyuan Gu, and Xiaozhe Hu. Ladder polynomial neural networks. arXiv preprint arXiv:2106.13834, 2021.

[50] Feng-Lei Fan, Mengzhou Li, Fei Wang, Rongjie Lai, and Ge Wang. On expressivity and trainability of quadratic networks. IEEE Transactions on Neural Networks and Learning Systems, 2023.

[51] Zhijian Zhuo, Ya Wang, Yutao Zeng, Xiaoqing Li, Xun Zhou, and Jinwen Ma. Polynomial composition activations: Unleashing the dynamics of large language models. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.

net/forum?id=CbpWPbYHuv.

[52] Majid Janzamin, Hanie Sedghi, and Anima Anandkumar. Beating the perils of non-convexity:
Guaranteed training of neural networks using tensor methods. *arXiv preprint arXiv:1506.08473*, 2015.

[53] Massimo Fornasier, Jan Vybíral, and Ingrid Daubechies. Robust and resource efficient identification of shallow neural networks by fewest samples. Information and Inference: A Journal of the IMA, 10(2):625–695, 2021.

[54] Christian Fiedler, Massimo Fornasier, Timo Klock, and Michael Rauchensteiner. Stable recovery of entangled weights: Towards robust identification of deep neural networks from minimal samples. *Applied and Computational Harmonic Analysis*, 62:123–172, 2023.

[55] Aapo Hyvärinen, Ilyes Khemakhem, and Ricardo Monti. Identifiability of latent-variable and structural-equation models: from linear to nonlinear. Annals of the Institute of Statistical Mathematics, 76(1):1–33, 2024.

[56] Ilyes Khemakhem, Diederik Kingma, Ricardo Monti, and Aapo Hyvarinen. Variational autoencoders and nonlinear ICA: A unifying framework. In International conference on artificial intelligence and statistics, pages 2207–2217. PMLR, 2020.

[57] Julius von Kügelgen, Michel Besserve, Liang Wendong, Luigi Gresele, Armin Kekic, Elias ´
Bareinboim, David Blei, and Bernhard Schölkopf. Nonparametric identifiability of causal representations from unknown interventions. Advances in Neural Information Processing Systems, 36:48603–48638, 2023.

[58] Geoffrey Roeder, Luke Metz, and Durk Kingma. On linear identifiability of learned representations. In *International Conference on Machine Learning*, pages 9030–9039. PMLR,
2021.

[59] Yujia Zheng, Ignavier Ng, and Kun Zhang. On the identifiability of nonlinear ICA: Sparsity and beyond. *Advances in neural information processing systems*, 35:16411–16422, 2022.

[60] Bohdan Kivva, Goutham Rajendran, Pradeep Ravikumar, and Bryon Aragam. Identifiability of deep generative models without auxiliary information. Advances in Neural Information Processing Systems, 35:15687–15701, 2022.

[61] V Lebedev, Y Ganin, M Rakhuba, I Oseledets, and V Lempitsky. Speeding-up convolutional neural networks using fine-tuned CP-decomposition. In Proc. 3rd International Conference on Learning Representations (ICLR), 2015.

[62] Alexander Novikov, Dmitrii Podoprikhin, Anton Osokin, and Dmitry P Vetrov. Tensorizing neural networks. *Adv. Neur. Inf. Proc. Syst.*, 28, 2015.

[63] Xingyi Liu and Keshab K Parhi. Tensor decomposition for model reduction in neural networks:
A review. *IEEE Circuits and Systems Magazine*, 23(2):8–28, 2023.

[64] Anh-Huy Phan, Konstantin Sobolev, Konstantin Sozykin, Dmitry Ermilov, Julia Gusak, Petr Tichavsky, Valeriy Glukhov, Ivan Oseledets, and Andrzej Cichocki. Stable low-rank tensor `
decomposition for compression of convolutional neural network. In *Proc. 16th European* Conference on Computer Vision (ECCV), pages 522–539, Glasgow, UK, 2020. Springer.

[65] Emanuele Zangrando, Steffen Schotthöfer, Jonas Kusch, Gianluca Ceruti, and Francesco Tudisco. Geometry-aware training of factorized layers in tensor Tucker format. Proceedings, Adv. Neur. Inf. Proc. Syst., 2024.

[66] Nadav Cohen, Or Sharir, and Amnon Shashua. On the expressive power of deep learning: A
tensor analysis. In *Conference on learning theory*, pages 698–728. PMLR, 2016.

[67] Maude Lizaire, Michael Rizvi-Martel, Marawan Gamal, and Guillaume Rabusseau. A tensor decomposition perspective on second-order RNNs. In Forty-first International Conference on Machine Learning, 2024.

[68] Valentin Khrulkov, Alexander Novikov, and Ivan Oseledets. Expressive power of recurrent neural networks. In *ICLR*, 2018.

[69] Anuj Mahajan, Mikayel Samvelyan, Lei Mao, Viktor Makoviychuk, Animesh Garg, Jean Kossaifi, Shimon Whiteson, Yuke Zhu, and Animashree Anandkumar. Tesseract: Tensorised actors for multi-agent reinforcement learning. In *International Conference on Machine Learning*, pages 7301–7312. PMLR, 2021.

[70] Sergio Rozada, Santiago Paternain, and Antonio G Marques. Tensor and matrix low-rank valuefunction approximation in reinforcement learning. *IEEE Transactions on Signal Processing*, 2024.

[71] Marco Mondelli and Andrea Montanari. On the connection between learning two-layer neural networks and tensor decomposition. In The 22nd International Conference on Artificial Intelligence and Statistics, pages 1051–1060. PMLR, 2019.

[72] Rong Ge, Rohith Kuditipudi, Zhize Li, and Xiang Wang. Learning two-layer neural networks with symmetric inputs. In *International Conference on Learning Representations*, 2019.

[73] Pranjal Awasthi, Alex Tang, and Aravindan Vijayaraghavan. Efficient algorithms for learning depth-2 neural networks with general ReLU activations. *Adv. Neur. Inf. Proc. Syst.*, 34:13485–
13496, 2021.

[74] François Malgouyres and Joseph Landsberg. Multilinear compressive sensing and an application to convolutional linear networks. *SIAM Journal on Mathematics of Data Science*, 1(3):446–475, 2019.

[75] Alex Casarotti and Massimiliano Mella. From non-defectivity to identifiability. Journal of the European Mathematical Society, 25(3):913–931, 2022.

[76] Joseph M Landsberg. *Tensors: Geometry and applications*, volume 128. American Mathematical Soc., 2012.

[77] Daniel Kunin, Jonathan Bloom, Aleksandrina Goeva, and Cotton Seed. Loss landscapes of regularized linear autoencoders. In *International Conference on Machine Learning*, pages 3560–3569. PMLR, 2019.

[78] Xuchan Bao, James Lucas, Sushant Sachdeva, and Roger B Grosse. Regularized linear autoencoders recover the principal components, eventually. Advances in Neural Information Processing Systems, 33:6971–6981, 2020.

[79] Pierre Comon, Gene Golub, Lek-Heng Lim, and Bernard Mourrain. Symmetric tensors and symmetric tensor rank. *SIAM Journal on Matrix Analysis and Applications*, 30(3):1254–1279, 2008.

[80] Nicholas D Sidiropoulos and Xiangqian Liu. Identifiability results for blind beamforming in incoherent multipath with small delay spread. *IEEE Transactions on Signal Processing*, 49(1):
228–236, 2001.

[81] Ignat Domanov and Lieven De Lathauwer. On the uniqueness of the canonical polyadic decomposition of third-order tensors—Part II: Uniqueness of the overall decomposition. SIAM
Journal on Matrix Analysis and Applications, 34(3):876–903, 2013.

[82] Nicholas D. Sidiropoulos and Rasmus Bro. On the uniqueness of multilinear decomposition of N-way arrays. *Journal of Chemometrics*, 14(3):229–239, 2000.

[83] Ignat Domanov and Lieven De Lathauwer. Generic uniqueness conditions for the canonical polyadic decomposition and INDSCAL. *SIAM Journal on Matrix Analysis and Applications*, 36(4):1567–1589, 2015.

[84] Hirotachi Abo and Maria Chiara Brambilla. On the dimensions of secant varieties of Segre-
Veronese varieties. *Annali di Matematica Pura ed Applicata*, 192(1):61–92, 2013.

[85] Ralf Fröberg, Samuel Lundqvist, Alessandro Oneto, and Boris Shapiro. Algebraic stories from one and from the other pockets. *Arnold Mathematical Journal*, 4(2):137–160, 2018.

[86] Alex Massarenti and Massimiliano Mella. Bronowski's conjecture and the identifiability of projective varieties. *Duke Mathematical Journal*, 173(17):3293–3316, 2024.

[87] Kim Batselier and Ngai Wong. Symmetric tensor decomposition by an iterative eigendecomposition algorithm. *Journal of Computational and Applied Mathematics*, 308:69–82, 2016.

[88] Samet Oymak and Mahdi Soltanolkotabi. Learning a deep convolutional neural network via tensor decomposition. *Information and Inference: A Journal of the IMA*, 10(3):1031–1071, 2021.

[89] Jacek Bochnak, Michel Coste, and Marie-Françoise Roy. *Real algebraic geometry*, volume 36.

Springer, 2013.

[90] Paul Breiding, Fulvio Gesmundo, Mateusz Michałek, and Nick Vannieuwenhoven. Algebraic compressed sensing. *Applied and Computational Harmonic Analysis*, 65:374–406, 2023.

[91] Yang Qi, Pierre Comon, and Lek-Heng Lim. Semialgebraic geometry of nonnegative tensor rank. *SIAM Journal on Matrix Analysis and Applications*, 37(4):1556–1580, 2016. SIAM.

[92] Samuel Lundqvist, Alessandro Oneto, Bruce Reznick, and Boris Shapiro. On generic and maximal k-ranks of binary forms. *Journal of Pure and Applied Algebra*, 223(5):2062–2079, 2019.

[93] Alexander Taveira Blomenhofer. On uniqueness of power sum decomposition. SIAM Journal on Applied Algebra and Geometry, 9(1):211–234, 2025.

[94] Alex Massarenti and Massimiliano Mella. The Alexander-Hirschowitz theorem for neurovarieties. *arXiv preprint arXiv:2511.19703*, 2025.

## A Roadmap To The Appendices4

The appendices of the paper contain background on tensor decompositions and neurovarieties, the proofs of the technical results, as well as a discussion on the changes between the originally submitted and final version of the paper. They are organized as follows:
- Appendix A presents background on neurovarieties for homogeneous PNNs. This is a crucial part for understanding the link between finite identifiability of an hPNN, the dimension of its neurovariety and the rank of the Jacobian of its parametrization map.

- Appendix B contains the main technical tools used in the proof the localization theorem and follows the structure of Section 4. In particular, it presents the proofs of necessary conditions for uniqueness (Section 4.1), background on tensor decompositions and Kruskal-based sufficient conditions for the identifiability of 2-layer hPNNs (Section 4.2).

- Appendix C presents the proof of the localization theorem (Theorem 11) and its consequences for several hPNN architectures, as well as some supporting technical results.

- Appendix D presents the proofs for the case of PNNs with biases. Appendix D.3 discusses the idea of *truncation*, an alternative approach to tackle the PNNs with biases.

- Appendix E discusses necessary and sufficient conditions for the identifiability of hPNNs, as well as changes between the originally submitted and the final version of the paper which were done to correct a mistake in the proof of one of the main results.

## A Homogeneous Pnns And Neurovarieties

hPNNs are often studied through the prism of neurovarieties, using their algebraic structure. Our results have direct implications on the expected dimension of the neurovarieties, as explained in this appendix.

## A.1 Neurovarieties And Dimension

An hPNN architecture (d, r) defines a map hPNNd,r[·] from the weight tuple w = (W1*, . . . ,*WL) to a (polynomial) function space H :

$$\begin{array}{r l}{\mathrm{hPNN}_{d,r}[\cdot]:}&{{}{\mathbf{w}}\mapsto\mathrm{hPNN}_{d,r}[{\mathbf{w}}]}\\ {\mathbb{R}^{\sum_{\ell}d_{\ell}d_{\ell-1}}\to{\mathcal{H}}.}\end{array}$$

The space H is the space of length-dL vectors of homogeneous polynomials of degree r*total* =
r1r2 *. . . r*L−1 in d0 variables:
H := (Hd0,r*total* )×dL ;
thus H is a finite-dimensional vector space of dimension

$$N=\dim({\mathcal{H}})=d_{L}{\binom{d_{0}+r_{t o t a l}-1}{r_{t o t a l}}},$$

which follows from the fact that dim(Hd,r) = d+r−1 r.

The key observation is that hPNNd,r[·] is a *polynomial-in-the-parameters* map, which has important implication on the space of networks with a given architecture. The image Im(hPNNd,r[·]), called a neuromanifold, is a semi-algebraic set5. The properties of Im(hPNNd,r[·]) are tightly linked to the properties of the *neurovariety* Vd,rdefined as the closure of Im(hPNNd,r[·]) in the Zariski topology, i.e., the smallest algebraic variety ( algebraic set6) containing Im(hPNNd,r[·]). The key property is the dimension of the neurovariety7 which is equal to the dimension of the neurovariety [89, Prop.

2.8.2].

4The appendices have been reorganized and reworked for better readability. 5[89, Def. 2.1.1]: a set cut out by polynomial equations and inequalities.

6[89, Def. 2.1.4]: a set cut out by polynomial equations. 7roughly defined as the dimension of the tangent space at general point, see [89, §2.8] for more details.

The properties of neurovarieties depend on the field (i.e., results can differ between R or C), and we focus on the real case. However, most of the results can be translated to the complex case as well. We mostly follow [90, Section 4], and an overview on semialgebraic sets can be also found in [91] (see [89] for a detailed account).

The following upper bound on dim Vd,r the bound was presented in [7]:

$$\dim{\mathcal{V}}_{d,r}\leq\min\left(\underbrace{\sum_{\ell=1}^{L}d_{\ell}d_{\ell-1}-\sum_{\ell=1}^{L-1}d_{\ell}}_{\mathrm{degrees~of~freedom}},\quad\underbrace{\dim{\mathcal{H}}}_{\mathrm{output~space~dimension}}\right).$$
(8) $$\binom{8}{4}$$. 
. (8)
If there is an equality in the bound (8), we say that the neurovariety has *expected dimension*. There are two fundamentally different cases when the expected dimension is reached. Expressive case. If the right bound is reached, i.e., the neurovariety:

$$\dim{\mathcal{V}}_{\mathbf{d},\mathbf{r}}=\dim\left({\mathcal{H}}\right)=d_{L}{\binom{d_{0}+r_{t o t a l}-1}{r_{t o t a l}}},$$

the hPNN is *expressive*, and the neurovariety Vd,ris said to be *thick* [7], as it fills the whole function space H (and thus the neuromanifold is of positive Lebesgue measure). In particular, this implies that (see [7, Proposition 5]) any homogeneous polynomial vector from H (i.e., of degree r*total* with d0 inputs and dL outputs, with degrees fixed as r1 = r2 = *· · ·* = rL−1) can be represented as an hPNN with layer widths (d0, 2d1, . . . , 2dL−1, dL) and the same activation degrees. Identifiable case. The left bound (PL
ℓ=1 dℓdℓ−1 −PL−1 ℓ=1 dℓ) follows from the presence of equivalences defined in Lemma 4 (i.e., the size of the vector w minus the number of independent rescalings) and defines the number of effective parameters of the representation (this is explained in the following subsections). Moreover, the left bound is reached if and only if the hPNN architecture is finitely identifiable:
Proposition 10 *The architecture* hPNNd,r[·] *is finitely identifiable if and only if the dimension of* Vd,r is equal to the effective number of parameters, i.e., dim Vd,r =PL
ℓ=1 dℓdℓ−1 −PL−1 ℓ=1 dℓ. In such case, Vd,r is said to be **nondefective***. Equivalently, the rank of the Jacobian of the map* hPNNd,r[·]
is maximal and equal to PL
ℓ=1 dℓdℓ−1 −PL−1 ℓ=1 dℓ *at a general parameter* w.

Proposition 10 is central to the proof of the main results of paper. The proof of Proposition 10 relies on properties of fibers of polynomial maps and is reviewed in the next subsection, together with the Jacobian of the parameterization.

## A.2 Polynomial Maps And Fiber Dimension

We recall some key facts on the polynomial maps and their images. We begin by highlighting the link between dimensions of semialgebraic sets and the Jacobian of the polynomial maps.

Lemma A.1. Let φ : R
m → R
n be a polynomial map, and denote by Jφ(θ) the n × m Jacobian matrix. Let r0 := max θrank{Jφ(θ)}.

Then we have that:
1. rank{Jφ(θ)} = r0 for generic θ *(i.e., for all* θ ∈ R
m except a set of Lebesgue measure zero, where the rank of the Jacobian is strictly less than r0).

2. r0 *is equal to the dimension of* Im(φ) *and its (Zariski) closure:*

$\tau_0$ :=
$$\sigma({\boldsymbol{\theta}})\}.$$

$$r_{0}=\dim\left(\operatorname{Im}(\varphi)\right)=\dim\left({\overline{{\operatorname{Im}(\varphi)}}}\right).$$
The proof of Lemma A.1 is given in [90, Theorem 4.7] and the preceding paragraph (in [90], the number r0 is called *generic rank* of the parameterization φ). It mainly follows from semicontinuity of the rank of a matrix. Remark A.2 (On genericity). Due to the algebraic structure of φ, the genericity statement in Lemma A.1 is much stronger: in fact, the set of points θ *where* rank{Jφ(θ)} ̸= r0 is a semialgebraic subset of R
m of dimension strictly less than m. The same holds for all generic statements and definitions in the paper (such as finite identifiability, global identifiability, etc.), see the definition of genericity in [90, Definition 4.1]. Remark A.3. *The right bound for neurovariety dimension in* (8) follows essentially from Lemma A.1:
indeed, in the case φ(·) = hPNNd,r[·], rank{Jφ} does not exceed the dimension of the ambient space of φ *(equal to* dim(H )). The following lemma is key for linking finite identifiability to the dimension of the neurovariety.

Lemma A.4 (Fiber dimension). Let φ : R
m → R
n *be a polynomial map, so that* r0 = dim(Im(φ)).

Then the dimension of its generic fiber is equal to m − r0*, that is, for generic* θ ∈ R
m*, the preimage* φ−1(φ(θ)) *is a semialgebraic set with* dim φ
−1φ(θ)= m − r0.

Lemma A.4 is well known to specialists, but in the literature it is mostly formulated for the complex case (see [90, Theorem 4.7]). For the real field it is a special case of [90, Theorem 4.9].

A particular case is when r0 = m, in which case Lemma A.4 implies finiteness of the fiber:
Corollary A.5. *The following two statements are equivalent:*
- *For general* θ ∈ R
m, rank{Jφ(θ)} = m;
- *For general* θ ∈ R
m, the fiber (i.e., the preimage φ−1(φ(θ))*) consists of a finite number of* points.

Proof. The statement follows from Lemma A.4 specialized to (r0 = m) and from the fact that 0-dimensional semialgebraic sets are collections of a finite number of points. Finally we make the following remark that is very commonly used.

Corollary A.6. If rank{Jφ(θ0)} = m *for some* θ0 ∈ R
m*, then* rank{Jφ(θ)} = m *for generic* θ.

Proof. This directly follows from Lemma A.1, since r0 in Lemma A.1 is equal to m.

Remark A.7. *Corollary A.6 implies that finding a* single point with full column rank Jacobian implies finitieness of the generic fiber.

## A.2.1 The Case Of Neurovarieties

The first implication of Lemma A.4 is the left upper bound in (8). It is based on the following lemma from [7], for which we provide a short proof for completeness.

Lemma A.8 ([7, Lemma 13]). *For a general parameter* w = (W1, . . . ,WL), the set of equivalent hPNN representations in Lemma 4 is semialgebraic and of dimension PL−1 ℓ=1 dℓ.

Proof. First, note that the set of equivalent representations is of dimension at most PL−1 ℓ=1 dℓ (by the number of parameters). Consider a general w = (W1*, . . . ,*WL), so that the first column of each Wℓ, for ℓ = 1*, . . . , L* − 1, equal to vℓ ∈ R
dℓ, does not have zero elements. Now take any collection of vectors ve1 ∈ R
d1*, . . . ,* veL−1 ∈ R
dL−1 having elementwise the same signs as vℓ. Then there exist matrices Dℓ so that the equivalent weight matrices Wfℓ = De ℓWℓDe
−rℓ−1 ℓ−1 have veℓ exactly as their first columns. Thus the set of equivalent representations is exactly of dimension PL−1 ℓ=1 dℓ.

Remark A.9. *The left upper bound in* (8) simply follows from Lemma A.8 (as written in [7, Lemma 13]): indeed, the dimension of the fiber of hPNNd,r[·] *must be at least* PL−1 ℓ=1 dℓ*. This* implies, by Lemma A.4,

implies, by Lemma A.9,_  $$\operatorname{rank}\{J_{\varphi}(\boldsymbol{\theta})\}\leq\sum_{\ell=1}^{L}d_{\ell-1}d_{\ell}-\sum_{\ell=1}^{L-1}d_{\ell},$$  _which is exactly the right dimension bound in (8) by Lemma A.1._
$$(9)$$
dℓ, (9)
Note that Proposition 10 will exactly consider the case when the equality is reached in (9) for generic θ. Similarly to Corollary A.6, the following corollary of Lemma A.1 implies that for the case of neurovarieties it suffices to find a single set of parameters w where the Jacobian of the parameterization is of maximal rank to guarantee finite identifiability of hPNN architecture. This will be used in the proofs to give a *certificate* of finite idenitifiability.

Corollary A.10. If there exists a particular point θ0 such that equality is achieved in (9), then the equality in (9) *is achieved for generic* θ. Proof. Since there exists such a θ0, then the r0 defined in Lemma A.1 satisfies

$$r_{0}\geq\sum_{\ell=1}^{L}d_{\ell-1}d_{\ell}-\sum_{\ell=1}^{L-1}d_{\ell}.$$
$$(10)$$

But from (9), r0 must be bounded from above by the same number. Therefore the equality for r0 is achieved in (10).

## A.3 Proof Of The Proposition

Proof of Proposition 10. We denote φ(·) = hPNNd,r[·] for simplicity (so that m =PL
ℓ=1 dℓdℓ−1 and n = dim H ) and consider separately the "only if" ( ⇒ ) and "if" ( ⇐ ) parts.

⇒ Assume that for a generic w the fiber φ−1(φ(w)) consists of finite number of equivalence classes, thus it is a finite union of non-intersecting semialgebraic subsets of dimension PL−1 ℓ=1 dℓ.

Therefore, by [89, Theorem 2.8.5] the whole fiber φ−1(φ(w)) has the dimension equal to PL−1 ℓ=1 dℓ as well, hence dim Vd,r =PL
ℓ=1 dℓdℓ−1 −PL−1 ℓ=1 dℓ.

⇐ The proof follows a similar argument as in the proof of [90, Theorem 4.9]. We consider a
(Zariski open) subset of parameters without zero values U = (R \ {0})
m. It can be shown that the preimage of the image of its complement Z := φ−1(φ(R
m \ U )) is a (semialgebraic) set of measure zero. Therefore for the set U ′:= U \ Z the preimage of the image is contained in U :

$\sim\;7$
$\bigcup a_{n}J_{n}$
$$(111)$$
φ−1(φ(U′)) ⊂ U .

Note that any w ∈ U can be brought (by diagonal scaling and permutation) to an equivalent form:

$$\mathbf{W}_{\ell}=\left[\begin{array}{cc}1&\cdots&1\\ \hline\mathbf{W}_{\ell}\end{array}\right],\quad\overline{\mathbf{W}}_{\ell}\in\mathbb{R}^{(d_{\ell}-1)\times d_{\ell-1}}\tag{1}$$

for all ℓ = 2*, . . . , L* where the reduced Wℓ parameterize the classes of equivalent parameters in U up to permutation. Now denote w = (W1,W2*, . . . ,*WL) and define w(w) = (W1*, . . . ,*WL) with Wℓ as in (11). Consider the following map ψ : w 7→ hPNNd,r[w(w)] .

Then if the generic fiber of ψ is finite, this will imply that on U ′, the fiber of the map φ contains finitely many equivalence classes. For this, note that the Jacobian of ψ is just a submatrix of the Jacobian of φ with exactly m −PL−1 ℓ=1 dℓ columns. We will show that it is full rank at a generic point w. Consider the following map

$\xi:(\mathbf{W}_{1},\overline{\mathbf{W}}_{2},\ldots,\overline{\mathbf{W}}_{L},\mathbf{D}_{1},\ldots,\mathbf{D}_{2})\mapsto(\mathbf{W}_{1},\widehat{\mathbf{W}}_{2},\ldots,\widehat{\mathbf{W}}_{L})$
$\mathfrak{H}\mathfrak{S}$
defined as
for ℓ = 2*, . . . , L* (with the convention that DL = IdL.
$$\widetilde{\boldsymbol{W}}_{\ell}=\boldsymbol{D}_{\ell}\left[\,\frac{1\,\cdots\,1}{\boldsymbol{W}_{\ell}}\,\right]\boldsymbol{D}_{\ell}^{-r_{\ell-1}}$$  tion that $\boldsymbol{D}_{L}=\boldsymbol{I}_{d_{L}}$.  
Consider a particular w0 constructed as above (by normalization of a w ∈ U ). Then for a neighborhood U of w0 and a neighbourhood V of (Id1*, . . . ,* IdL−1), the map ξ is a diffeomorphism from U × V to an open neigbourhood of the corresponding w0 = w(w0).