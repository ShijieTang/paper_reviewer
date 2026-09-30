# A Stochastic Approximation Approach for Efficient Decentralized Optimization on Random Networks

Anonymous Author(s) Affiliation Address email

## Abstract

 A challenging problem in decentralized optimization is to develop algorithms with fast convergence on random and time varying topologies under unreliable and bandwidth-constrained communication network. This paper studies a stochastic approximation approach with a Fully Stochastic Primal Dual Algorithm (FSPDA) framework. Our framework relies on a novel observation that randomness in time varying topology can be incorporated in a stochastic augmented Lagrangian for- mulation, whose expected value admits saddle points that coincide with stationary solutions of the decentralized optimization problem. With the FSPDA framework, we develop two new algorithms supporting efficient sparsified communication on random time varying topologies — FSPDA-SA allows agents to execute multiple local gradient steps depending on the time varying topology to accelerate conver- gence, and FSPDA-STORM further incorporates a variance reduction step to improve sample complexity. For problems with smooth (possibly non-convex) objective function, within T iterations, we show that FSPDA-SA (resp. FSPDA-STORM) finds an O(1/ √ T)-stationary (resp. O(1/T<sup>2</sup>/<sup>3</sup> )) solution. Numerical experiments show the benefits of the FSPDA algorithms.

#### <sup>17</sup> 1 Introduction

<sup>18</sup> Consider n agents that communicate on an undirected and connected graph/network G = (V, E) with <sup>19</sup> V = [n] := {1, . . . , n}, E ⊆ V × V. Each agent i ∈ [n] has access to a continuously differentiable (possibly non-convex) local objective function f<sup>i</sup> : <sup>R</sup> <sup>20</sup> <sup>d</sup> → <sup>R</sup> and maintains a local decision variable x<sup>i</sup> ∈ <sup>R</sup> d . Denote x = [x ⊤ 1 , ..., x ⊤ n <sup>⊤</sup> ∈ <sup>R</sup> nd <sup>21</sup> . Our aim is to tackle:

$$\min_{\mathbf{x} \in \mathbb{R}^{nd}} \frac{1}{n} \sum_{i=1}^n f_i(\mathbf{x}_i) \quad \text{s.t.} \quad \mathbf{x}_i = \mathbf{x}_j, \quad \forall (i, j) \in \mathcal{E}. \quad (1)$$

In other words, [\(1\)](#page-0-0) seeks a x <sup>⋆</sup> ∈ <sup>R</sup> d that minimizes F(x) := (1/n) P<sup>n</sup> <sup>i</sup>=1 <sup>22</sup> fi(x). We are interested <sup>23</sup> in the stochastic optimization setting where each fi(xi) is given by (with slight abuse of notation)

$$f_i(\mathbf{x}_i) := \mathbb{E}_{\xi_i \sim \mathbb{P}_i}[f_i(\mathbf{x}_i; \xi_i)] \quad (2)$$

 where <sup>P</sup><sup>i</sup> represents the i-th data distribution. Problem [\(1\)](#page-0-0) is relevant to the distributed learning [p](#page-10-0)roblem especially in the decentralized case where a central server is absent. Prior works [\[Nedic and](#page-10-0) [Ozdaglar,](#page-10-0) [2009,](#page-10-0) [Lian et al.,](#page-10-1) [2017,](#page-10-1) [Nedic et al.,](#page-10-2) [2017,](#page-10-2) [Qu and Li,](#page-10-3) [2017\]](#page-10-3) demonstrated that *decentralized* algorithms can tackle [\(1\)](#page-0-0) efficiently through repeated message exchanges among the neighbors and local stochastic gradient updates.

<sup>29</sup> Towards an efficient decentralized algorithm for [\(1\)](#page-0-0), an important direction is to consider a *time* <sup>30</sup> *varying graph topology* setting where the *active edge set* in G changes over time. This is a generic <sup>31</sup> setting covering cases when the communication links are unreliable, or the agents choose not to <sup>32</sup> communicate in a certain round (a.k.a. local updates) [\[Koloskova et al.,](#page-9-0) [2019a,](#page-9-0) [Nadiradze et al.,](#page-10-4) [2021\]](#page-10-4).

| Prior Works                              | SG           | $\chi$            | $\chi/o$ BH  | Rate                                |
|------------------------------------------|--------------|-------------------|--------------|-------------------------------------|
| Prox-GPDA [Hong et al., 2017]            | $\times$     | $\times$          | $\checkmark$ | Asympt.                             |
| NEXT [Lorenzo and Scutari, 2016]         | $\times$     | $\checkmark$      | $\checkmark$ | Asympt.                             |
| DSGD [Koloskova et al., 2020]            | $\checkmark$ | $\checkmark$      | $\times$     | $\mathcal{O}(\sigma/\sqrt{nT})$     |
| Swarm-SGD [Nadiradze et al., 2021]       | $\checkmark$ | $\checkmark$      | $\times$     | $\mathcal{O}(\sigma^2/\sqrt{nT})$   |
| CHOCO-SGD [Koloskova et al., 2019a]      | $\checkmark$ | $\times^\ddagger$ | $\times$     | $\mathcal{O}(\sigma/\sqrt{nT})$     |
| Decen-Scaffnew [Mishchenko et al., 2022] | $\checkmark$ | $\times^\ddagger$ | $\checkmark$ | $\mathcal{O}(\sigma/\sqrt{nT})$     |
| Local-GT [Liu et al., 2024]              | $\checkmark$ | $\times^\ddagger$ | $\checkmark$ | $\mathcal{O}(\sigma/\sqrt{nT})$     |
| LED [Alghunaim, 2024]                    | $\checkmark$ | $\times^\ddagger$ | $\checkmark$ | $\mathcal{O}(\sigma/\sqrt{nT})$     |
| FSPDA-SA (This Work)                     | $\checkmark$ | $\checkmark$      | $\checkmark$ | $\mathcal{O}(\sigma/\sqrt{nT})$     |
| FSPDA-STORM (This Work)                  | $\checkmark$ | $\checkmark$      | $\checkmark$ | $\mathcal{O}(\sigma^{2/3}/T^{2/3})$ |

Table 1: Comparison of decentralized algorithms for non-convex optimization. In the table, 'SG' is 'Stochastic Gradient', 'TV' is 'Time Varying Graph', 'w/o BH' is 'Without Bounded Heterogeneity', and 'Rate' is the expected squared gradient norm <sup>E</sup>[∥∇F(x¯)∥ ] after T iterations. Note that σ is the variance of stochastic gradient. ‡CHOCO-SGD incorporates broadcast gossip as a special case of compression. †ProxSkip, Local-GT, LED consider local updates with periodic communication.

 By assuming that a random topology is drawn at each iteration, the convergence of decentralized stochastic gradient (DSGD) has been studied in [\[Lobel and Ozdaglar,](#page-10-8) [2010,](#page-10-8) [Nadiradze et al.,](#page-10-4) [2021\]](#page-10-4) and is later on unified by [\[Koloskova et al.,](#page-9-2) [2020\]](#page-9-2) with tighter bounds for local updates, periodic sampling, etc. An alternative [\[Ram et al.,](#page-10-9) [2010\]](#page-10-9) is to analyze DSGD for the B-connectivity setting which requires the union of every B consecutive time varying topologies to yield a connected graph. Nevertheless, these works focused on vanilla DSGD that may have slow convergence (in transient stage) and is limited to bounded data heterogeneity. The prior restrictions can be relaxed using advanced algorithms such as gradient tracking [\[Qu and Li,](#page-10-3) [2017\]](#page-10-3), EXTRA [\[Shi et al.,](#page-10-10) [2015\]](#page-10-10) and primal-dual framework [\[Hong et al.,](#page-9-1) [2017,](#page-9-1) [Hajinezhad and Hong,](#page-9-4) [2019,](#page-9-4) [Yi et al.,](#page-11-0) [2021\]](#page-11-0).

 As noted by [\[Koloskova et al.,](#page-9-5) [2021\]](#page-9-5), analyzing the convergence of sophisticated algorithms with time varying topology, such as gradient tracking [\[Qu and Li,](#page-10-3) [2017\]](#page-10-3) is challenging due to the non-symmetric product of two (or more) mixing matrices. Existing works considered various restrictions on the time varying topology G (t) = (V, E (t) ) and/or the problem [\(1\)](#page-0-0): [\[Koloskova et al.,](#page-9-5) [2021,](#page-9-5) [Liu et al.,](#page-10-7) [2024\]](#page-10-7) studied gradient tracking with local updates that essentially takes E <sup>46</sup> (t) = E periodically and E <sup>47</sup> (t) = ∅ otherwise, also see [\[Mishchenko et al.,](#page-10-6) [2022,](#page-10-6) [Guo et al.,](#page-9-6) [2023,](#page-9-6) [Alghunaim,](#page-9-3) [2024\]](#page-9-3) for a similar result and note that such algorithms require extra synchronization overhead; [\[Kovalev et al.,](#page-9-7) [2021,](#page-9-7) [2024\]](#page-9-8) considered a setting where G (t) is connected for any t; [\[Nedic et al.,](#page-10-2) [2017,](#page-10-2) [Li and Lin,](#page-10-11) [2024\]](#page-10-11) focused on (accelerated) gradient tracking with deterministic gradient when F(x) is (strongly) convex; [\[Lorenzo and Scutari,](#page-10-5) [2016\]](#page-10-5) also considered deterministic gradient with possibly non-convex F(x) but only provides asymptotic convergence guarantees; [\[Lei et al.,](#page-10-12) [2018,](#page-10-12) [Yau and Wai,](#page-11-1) [2023\]](#page-11-1) considered asymptotic convergence guarantees in the case of strictly (or strongly) convex F(x). We provide a non-exhaustive list summarizing the convergence of existing works in Table [1.](#page-1-0)

<sup>55</sup> The above discussion highlights a gap in the existing literature —

<sup>56</sup> *Is there any algorithm that achieves fast convergence on time varying (random) topology?*

<sup>57</sup> This paper gives an affirmative answer through developing the Fully Stochastic Primal Dual Algorithm <sup>58</sup> (FSPDA) framework that leads to efficient decentralized algorithms tackling [\(1\)](#page-0-0) in its general form. <sup>59</sup> The framework features the design of a new stochastic augmented Lagrangian function.

 As pointed out by [\[Chang et al.,](#page-9-9) [2020\]](#page-9-9), many decentralized algorithms (including gradient tracking) can be interpreted as primal-dual algorithms finding a saddle point of the augmented Lagrangian func- tion. However, its extension to time varying topology is not straightforward due to the inconsistency in dual variables updates. To overcome this challenge, we propose a stochastic equality constrained reformulation of [\(1\)](#page-0-0) to model randomness in topology. Then, the latter yields a stochastic augmented Lagrangian function. Applying stochastic approximation (SA) to solve the latter leads to the FSPDA framework. Our contributions are

- <sup>67</sup> We propose two new algorithms: (i) FSPDA-SA is derived by vanilla SA that applies primal-dual <sup>68</sup> stochastic gradient descent-ascent on the stochastic augmented Lagrangian, (ii) FSPDA-STORM uses <sup>69</sup> an additional control variate / momentum term to reduce the drift term's variance in a recursive <sup>70</sup> manner. Both algorithms are fully stochastic as the random time varying topology is treated as <sup>71</sup> a part of randomness. Additionally, our framework supports sparsified communication, i.e., the <sup>72</sup> agents can choose to communicate a subset of primal coordinates at each iteration. <sup>73</sup> • We show that after T iterations, FSPDA-SA (resp. FSPDA-STORM) finds in expectation a solution whose squared gradient norm is O(1/ √
- T) (resp. O(1/T<sup>2</sup>/<sup>3</sup> <sup>74</sup> )). The convergence analysis is derived <sup>75</sup> from a new Lyapunov function design that involves an unsigned inner product term and incorporates <sup>76</sup> a variance condition on the random time varying topologies. Interestingly, we show empirically <sup>77</sup> that using momentum in dual updates benefits the consensus error convergence. <sup>78</sup> • We also demonstrate that both FSPDA-SA and FSPDA-STORM can be implemented in a fully asyn-<sup>79</sup> chronous manner, i.e., the agents can communicate and compute at different time slots, and supports <sup>80</sup> local update as the algorithms allow for arbitrary time varying topology. That said, we remark that <sup>81</sup> the convergence rates with local updates of FSPDA-SA and FSPDA-STORM are only suboptimal.

<sup>82</sup> We provide numerical experiments to show that FSPDA-SA and FSPDA-STORM outperform existing <sup>83</sup> algorithms in terms of iteration and communication complexity.

Notations. Let W ∈ R d×d <sup>84</sup> be a symmetric (not necessarily positive semidefinite) matrix, the Wweighted (semi) inner product of vectors a, b ∈ R d is denoted as ⟨<sup>a</sup> | <sup>b</sup>⟩<sup>W</sup> := <sup>a</sup> <sup>85</sup> <sup>⊤</sup>Wb. Similarly, the W-weighted (semi) norm is denoted by ∥a∥ 2 <sup>86</sup> <sup>W</sup> := ⟨a | a⟩W. The subscript notation is omitted for I-weighted inner products. For any square matrix X, (X) † <sup>87</sup> denotes its pseudo inverse.

## <sup>88</sup> 2 The Fully Stochastic Primal Dual Algorithm (FSPDA) Framework

<sup>89</sup> This section develops the FSPDA framework for tackling [\(1\)](#page-0-0) and describes two variants of the framework leading to decentralized stochastic optimization of [\(1\)](#page-0-0). Let <sup>A</sup>e ∈ {−1, <sup>0</sup>, <sup>1</sup>} |E|×<sup>n</sup> <sup>90</sup> be an incidence matrix of G. By defining <sup>A</sup> <sup>=</sup> <sup>A</sup>e ⊗ <sup>I</sup><sup>d</sup> ∈ {−1, <sup>0</sup>, <sup>1</sup>} |E|d×nd <sup>91</sup> , we observe that the consensus <sup>92</sup> constraint in [\(1\)](#page-0-0) is equivalent to Ax = 0.

 Our first step is to model the randomness in the time varying topology using the random variable (r.v.) ξ<sup>a</sup> ∼ <sup>P</sup>a. For each realization ξa, we define the random incidence matrix A(ξa) := I(ξa)A ∈ {−1, 0, 1} |E|d×nd where I(ξa) ∈ {0, 1} |E|d×|E|d is a binary diagonal matrix. In addition to selecting each edge of G randomly, I(ξa) selects a random subset of d coordinates. As we will see later, this allows our approach to simultaneously achieve random sparsification for communication compression.

Assume that <sup>E</sup>ξa∼P<sup>a</sup> <sup>98</sup> [I(ξa)] is a positive diagonal matrix, [\(1\)](#page-0-0) is equivalent to:

$$\min_{\mathbf{x} \in \mathbb{R}^{nd}} \frac{1}{n} \sum_{i=1}^n \mathbb{E}_{\xi_i \sim \mathbb{P}_i} [f_i(\mathbf{x}_i; \xi_i)] \quad \text{s.t.} \quad \mathbb{E}_{\xi_a \sim \mathbb{P}_a} [\mathbf{A}(\xi_a)] \mathbf{x} = \mathbf{0}. \quad (3)$$

<sup>99</sup> Denote ξ = (ξ1, . . . , ξn, ξa), FSPDA hinges on the following *augmented Lagrangian* function of [\(3\)](#page-2-0):

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}) := \mathbb{E}_\xi[\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}; \xi)] \quad (4)$$

with 
$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}; \xi) := \sum_{i=1}^n f_i(\mathbf{x}_i; \xi_i) + \bar{\eta} \langle \boldsymbol{\lambda} \mid \mathbf{A}(\xi_a)\mathbf{x} \rangle + \frac{\bar{\gamma}}{2} \|\mathbf{A}(\xi_a)\mathbf{x}\|^2,$$
 (4)

<sup>100</sup> where η >˜ 0, γ > ˜ 0 are penalty parameters. It can be verified that the saddle points of L(x,λ) <sup>101</sup> correspond to the KKT points of [\(1\)](#page-0-0) [\[Bertsekas,](#page-9-10) [2016\]](#page-9-10). For brevity, in the rest of this paper, we may <sup>102</sup> drop the subscript in ξ whenever the notation is clear from the context.

<sup>103</sup> FSPDA is developed from applying stochastic approximation (SA) to seek a saddle point of [\(4\)](#page-2-1). By recognizing A(ξ) <sup>104</sup> <sup>⊤</sup>A(ξ) = A⊤A(ξ), we consider the stochastic gradients:

$$\nabla_{\mathbf{x}}\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}; \xi) := \nabla \mathbf{f}(\mathbf{x}; \xi) + \tilde{\eta} \mathbf{A}^{\top} \boldsymbol{\lambda} + \tilde{\gamma} \mathbf{A}^{\top} \mathbf{A}(\xi) \mathbf{x}, \quad \nabla_{\boldsymbol{\lambda}}\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}; \xi) := \tilde{\eta} \mathbf{A}(\xi) \mathbf{x}, \quad (5)$$

where ∇f(x; ξ) = [∇f1(x1; ξ1); . . . ; ∇fn(xn; ξn)] ∈ <sup>R</sup> nd <sup>105</sup> . Notice that to facilitate algorithm <sup>106</sup> development, we have taken a deterministic A for the term in ∇xL related to λ. Now observe the ith <sup>107</sup> d-dimensional block of A⊤A(ξ)x which can be aggregated within Ni(ξ) the neighborhood of the <sup>108</sup> ith agent as:

$$[\mathbf{A}^\top \mathbf{A}(\xi) \mathbf{x}]_i = \sum_{j \in \mathcal{N}_i(\xi)} \mathbf{C}_{ij}(\xi) (\mathbf{x}_j - \mathbf{x}_i), \quad (6)$$

where Cij (ξ) ∈ {0, 1} d×d <sup>109</sup> is diagonal and depends on the selected coordinates for the edge (i, j) <sup>110</sup> under randomness ξ. Eq. [\(6\)](#page-2-2) *only* relies on x<sup>j</sup> from neighbor j that is connected on the time varying

 topology G(ξ). For illustration, an example of the above random graph model is given by Figure [3](#page--1-0) in Appendix [A.](#page--1-1) Importantly, [\(5\)](#page-2-3) shows that with the stochastic augmented Lagrangian function, the time varying topology can be treated implicitly as a part of the randomness in the stochastic primal-dual gradients. The framework is thus described as being *fully stochastic* as in [\[Bianchi et al.,](#page-9-11) [2021\]](#page-9-11), and departs from [\[Liu et al.,](#page-10-7) [2024,](#page-10-7) [Alghunaim,](#page-9-3) [2024\]](#page-9-3) that treat the topology as fixed during the derivation of primal-dual algorithm(s). From [\(5\)](#page-2-3), [\(6\)](#page-2-2), we derive *two* variants of FSPDA.

<sup>117</sup> FSPDA-SA Algorithm. The first variant of FSPDA is derived from a direct application of stochastic <sup>118</sup> gradient descent-ascent (SGDA) updates. Take α > 0, β > 0 as the step sizes, we have

$$\mathbf{x}^{t+1} = \mathbf{x}^t - \alpha \nabla_{\mathbf{x}} \mathcal{L}(\mathbf{x}^t, \boldsymbol{\lambda}^t; \xi^t), \quad \boldsymbol{\lambda}^{t+1} = \boldsymbol{\lambda}^t + \beta \nabla_{\boldsymbol{\lambda}} \mathcal{L}(\mathbf{x}^t, \boldsymbol{\lambda}^t; \xi^t). \quad (7)$$

Taking the variable substitution <sup>λ</sup>b := <sup>A</sup><sup>⊤</sup> <sup>119</sup> <sup>λ</sup> yields the following recursion:

FSPDA-SA: for any t ≥ 0 and any i ∈ [n],

$$\mathbf{x}_i^{t+1} = \mathbf{x}_i^t - \alpha \nabla f_i(\mathbf{x}_i^t; \xi_i^t) - \eta \widehat{\mathbf{x}}_i^t + \gamma \sum_{j \in \mathcal{N}_i(\xi_a^t)} \mathbf{C}_{ij}(\xi_a^t)(\mathbf{x}_j^t - \mathbf{x}_i^t), \quad (8a)$$

$$\hat{\lambda}_i^{t+1} = \hat{\lambda}_i^t + \beta \sum_{j \in \mathcal{N}_i(\xi_t^t)} \mathbf{C}_{ij}(\xi_a^t)(\mathbf{x}_j^t - \mathbf{x}_i^t). \quad (8b)$$

120

Note that x 0 ,λb<sup>0</sup> <sup>121</sup> can be initialized arbitrarily.

<sup>122</sup> FSPDA-STORM Algorithm. The second variant of FSPDA reduces the variance of the stochastic <sup>123</sup> [g](#page-9-12)radient term in [\(5\)](#page-2-3) using the recursive momentum variance reduction technique [\[Cutkosky and](#page-9-12) <sup>124</sup> [Orabona,](#page-9-12) [2019\]](#page-9-12). Herein, the key idea is to utilize a control variate in estimating the (primal-dual) <sup>125</sup> gradients of L(x,λ). Take α, β > 0 and ax, a<sup>λ</sup> ∈ [0, 1] as the momentum parameters, we have x <sup>t</sup>+1 = x <sup>t</sup> − αm<sup>t</sup> x ,λ <sup>t</sup>+1 = λ <sup>t</sup> + βm<sup>t</sup> λ <sup>126</sup> as the primal-dual updates, and

$$\begin{aligned} \mathbf{m}_x^{t+1} &= \nabla_{\mathbf{x}} \mathcal{L}(\mathbf{x}^{t+1}, \boldsymbol{\lambda}^{t+1}; \xi^{t+1}) + (1 - a_x)(\mathbf{m}_x^t - \nabla_{\mathbf{x}} \mathcal{L}(\mathbf{x}^t, \boldsymbol{\lambda}^t; \xi^{t+1})), \\ \mathbf{m}_\lambda^{t+1} &= \nabla_{\boldsymbol{\lambda}} \mathcal{L}(\mathbf{x}^{t+1}, \boldsymbol{\lambda}^{t+1}; \xi^{t+1}) + (1 - a_\lambda)(\mathbf{m}_\lambda^t - \nabla_{\boldsymbol{\lambda}} \mathcal{L}(\mathbf{x}^t, \boldsymbol{\lambda}^t; \xi^{t+1})). \end{aligned} \quad (9)$$

The aim of m<sup>t</sup>+1 x is to estimate ∇xL(x t+1 ,λ <sup>t</sup>+1 <sup>127</sup> ). Now, instead of the straightforward estimator ∇xL(x t+1 ,λ <sup>t</sup>+1; ξ <sup>t</sup>+1), we include an extra zero-mean term m<sup>t</sup> <sup>x</sup> − ∇xL(x t ,λ t ; ξ <sup>t</sup>+1 <sup>128</sup> ) to reduce <sup>129</sup> the variance of the stochastic gradient estimation. The latter is a control variate that is computed <sup>130</sup> recursively. Particularly, it has been shown in [\[Cutkosky and Orabona,](#page-9-12) [2019\]](#page-9-12) that it can effectively <sup>131</sup> reduce variance with a carefully designed parameter ax, provided that the stochastic gradient map <sup>132</sup> satisfies a mean-square Lipschitz condition. We summarize the algorithm as follows.

$$\mathbf{x}_i^{t+1} = \mathbf{x}_i^t - \alpha \mathbf{m}_{x,i}^t, \quad (10a)$$

$$\hat{\lambda}_i^{t+1} = \hat{\lambda}_i^t + \beta \mathbf{m}_{\lambda,i}^t, \quad (10b)$$

$$\mathbf{m}_{x,i}^{t+1} = (1-a_x) [\mathbf{m}_{x,i}^t + \nabla f_i(\mathbf{x}_i^t; \xi_i^{t+1}) - \eta \hat{\mathbf{A}}_i^t + \gamma \sum_{j \in \mathcal{N}_i} (\xi_a^{t+1}) \mathbf{C}_{ij} (\xi_a^{t+1}) (\mathbf{x}_j^t - \mathbf{x}_i^t)] \quad (10c)$$

+ ∇fi(x

t+1 i ; ξ t+1 i

) − <sup>η</sup>λbt+1

<sup>i</sup> + γ

P

$$\begin{aligned} & + \nabla f_i(\mathbf{x}_i^{t+1}; \xi_i^{t+1}) - \eta \hat{\mathbf{\lambda}}_i^{t+1} + \gamma \sum_{j \in \mathcal{N}_i(\xi_a^{t+1})} \mathbf{C}_{ij}(\xi_a^{t+1}) \mathbf{C}_{ij}(\xi_a^{t+1}) (\mathbf{x}_j^{t+1} - \mathbf{x}_i^{t+1}) \\ & \mathbf{m}_{\lambda, i}^{t+1} = (1 - a_\lambda) [\mathbf{m}_{\lambda, i}^t + \sum_{j \in \mathcal{N}_i(\xi_a^{t+1})} \mathbf{C}_{ij}(\xi_a^{t+1}) (\mathbf{x}_j^t - \mathbf{x}_i^t)] \\ & + \sum_{j \in \mathcal{N}_i(\xi_a^{t+1})} \mathbf{C}_{ij}(\xi_a^{t+1}) (\mathbf{x}_j^{t+1} - \mathbf{x}_i^{t+1}) \end{aligned} \quad (10d)$$

t+1

t+1 a )(x t+1 <sup>j</sup> − x

t+1 i )

133

Note that to achieve the theoretical performance (see later in Sec. [3\)](#page-4-0), x 0 ,λb<sup>0</sup> , m<sup>0</sup> x , m<sup>0</sup> λ <sup>134</sup> shall be initialized as x 0 <sup>i</sup> = x¯ 0 , <sup>λ</sup>b<sup>0</sup> <sup>i</sup> = (α/η)n −1 (∇F(x¯ 0 ) − ∇fi(x¯ 0 )), m<sup>0</sup> x,i = ∇F(x¯ 0 ), m<sup>0</sup> <sup>135</sup> λ,i = 0 according to [\(23\)](#page-6-0). We remark that a simple initialization choice <sup>λ</sup>b<sup>0</sup> <sup>=</sup> <sup>m</sup><sup>0</sup> x,i = m<sup>0</sup> <sup>136</sup> λ,i = 0 works well <sup>137</sup> in practice.

<sup>138</sup> Both FSPDA-SA and FSPDA-STORM are decentralized algorithms that can be implemented on random <sup>139</sup> time varying topology, and support randomized sparisification for further communication compres-<sup>140</sup> sion. The key is to observe that in P [\(8\)](#page-3-0), [\(10\)](#page-3-1), the only information required for agent i is to obtain j∈Ni(ξ t a ) Cij (ξ t a )(x t <sup>j</sup> − x t i ), and in addition P j∈Ni(ξ t a ) Cij (ξ t a )(x t−1 <sup>j</sup> − x t−1 i <sup>141</sup> ) for FSPDA-STORM, <sup>142</sup> at iteration t.

#### <sup>143</sup> 2.1 Implementation Details and Connection to Existing Works

<sup>144</sup> We discuss several features of the FSPDA algorithms and their connections to existing works.

<sup>145</sup> Local & Asynchronous Updates. The *local update* scheme where each agent i is allowed to update its own local variables x<sup>i</sup> <sup>146</sup> ,λ<sup>i</sup> for multiple iterations without a communication step is a <sup>147</sup> common practice in decentralized optimization [\[Liu et al.,](#page-10-7) [2024,](#page-10-7) [Li and Lin,](#page-10-11) [2024,](#page-10-11) [Alghunaim,](#page-9-3) [2024,](#page-9-3) <sup>148</sup> [Mishchenko et al.,](#page-10-6) [2022\]](#page-10-6). As discussed before, such scheme can be seen as a special case of the FSPDA framework where the time varying topology E (t) <sup>149</sup> is chosen such that the latter alternates between E (t) = E and E <sup>150</sup> (t) = ∅.

<sup>151</sup> Furthermore, FSPDA-SA allows for the general case of *asynchronous* updates. This is done so by taking the stochastic gradient as ∇fi(x t i ; ξ t ) = bi(ξ t ) b<sup>i</sup> ∇fi(x t i ; ξ t ) such that bi(ξ t <sup>152</sup> ) ∈ {0, 1} with <sup>E</sup>[bi(ξ t <sup>153</sup> )] = 1/b<sup>i</sup> for some constant b<sup>i</sup> > 0. Detailed discussions for a fully asynchronous <sup>154</sup> implementation of FSPDA-SA can be found in Appendix [A.](#page--1-1)

Connection to Existing Works. Evaluating x <sup>t</sup>+2 − x <sup>t</sup>+1 <sup>155</sup> from the FSPDA-SA sequence and observe <sup>156</sup> that the combination of [\(8a\)](#page-3-2) and [\(8b\)](#page-3-3) is equivalent to the second order recursion:

$$\begin{aligned} \mathbf{x}^{t+2} &= 2 \left( \mathbf{I} - \frac{\gamma}{2} \mathbf{A}^\top \mathbf{A} (\xi^{t+1}) \right) \mathbf{x}^{t+1} - (\mathbf{I} - (\gamma - \eta\beta) \mathbf{A}^\top \mathbf{A} (\xi^t)) \mathbf{x}^t \\ &\quad - \alpha (\nabla \mathbf{f}(\mathbf{x}^{t+1}; \xi^{t+1}) - \nabla \mathbf{f}(\mathbf{x}^t; \xi^t)). \end{aligned} \quad (11)$$

This reduces the FSPDA-SA recursion into a primal-only sequence by eliminating the dual sequence λ t <sup>157</sup> . <sup>158</sup> In the deterministic optimization setting when A(ξ) ≡ A and ∇f(x; ξ) ≡ ∇f(x), [\(11\)](#page-4-1) is equivalent to the EXTRA algorithm [\[Shi et al.,](#page-10-10) [2015\]](#page-10-10) using the mixing matrix W = I − γDiag(W1 ˜ ) + γW˜ <sup>159</sup> where W˜ <sup>160</sup> is the 0-1 adjacency matrix of G. Here, with an appropriate choice of γ, W will be doubly <sup>161</sup> stochastic and satisfies the convergence requirement in [\[Shi et al.,](#page-10-10) [2015\]](#page-10-10). Similar observations have <sup>162</sup> been made in [\[Nedic et al.,](#page-10-2) [2017\]](#page-10-2) for the gradient tracking and DIGing algorithms.

<sup>163</sup> On the other hand, for stochastic optimization on random networks, [\(11\)](#page-4-1) suggests each agent to keep <sup>164</sup> the current and previous iterates received from neighbors in the corresponding time varying topology. <sup>165</sup> In this case, [\(11\)](#page-4-1) yields an extension of the EXTRA/GT algorithms to time varying topology.

## <sup>166</sup> 3 Convergence Analysis of FSPDA

<sup>167</sup> This section presents the convergence rate analysis of FSPDA for [\(1\)](#page-0-0). Unless otherwise specified, we <sup>168</sup> focus on the case with smooth but possibly non-convex objective function. Specifically, we consider: Assumption 3.1. *Each* f<sup>i</sup> <sup>169</sup> *is* L*-smooth, i.e., for* i = 1, . . . , n*,*

$$\|\nabla f_i(\mathbf{x}) - \nabla f_i(\mathbf{y})\| \leq L\|\mathbf{x} - \mathbf{y}\| \forall \mathbf{x}, \mathbf{y} \in \mathbb{R}^d. \quad (12)$$

*There exists* f<sup>⋆</sup> > −∞ *such that* fi(x) ≥ f<sup>⋆</sup> *for any* x ∈ <sup>R</sup> d <sup>170</sup> *.*

<sup>171</sup> Note this implies that the global objective function F(·) is L-smooth but possibly non-convex.

<sup>172</sup> We further assume that the random network G(ξa) is connected in expectation, yet each realization <sup>173</sup> G(ξa) may not be connected. Let R = <sup>E</sup> [I(ξa)], this leads to the following property concerning the expected graph Laplacian matrix A⊤RA = <sup>E</sup> A(ξa) <sup>⊤</sup>A <sup>174</sup> . Defining the matrix K := (I<sup>n</sup> − 11<sup>⊤</sup> <sup>175</sup> /n) ⊗ Id, we have

<sup>176</sup> Assumption 3.2. *There exists* ρmax ≥ ρmin > 0 *and* ρ¯max ≥ ρ¯min > 0 *such that*

$$\rho_{\min}\mathbf{K} \preceq \mathbf{A}^\top\mathbf{R}\mathbf{A} \preceq \rho_{\max}\mathbf{K} \quad \text{and} \quad \bar{\rho}_{\min}\mathbf{K} \preceq \mathbf{A}^\top\mathbf{A} \preceq \bar{\rho}_{\max}\mathbf{K}. \quad (13)$$

<sup>177</sup> It holds that A⊤RAK = A⊤RA = KA⊤RA. The above assumption can be satisfied if G is <sup>178</sup> connected [\[Yi et al.,](#page-11-0) [2021\]](#page-11-0), [\[Yi et al.,](#page-11-2) [2018,](#page-11-2) Lemma 2] and diag(R) > 0 such that each edge is selected with a positive probability. As an important consequence, if γ ≤ ρmin/ρ<sup>2</sup> max <sup>179</sup> , we have

$$\|(\mathbf{I} - \gamma \mathbf{A}^\top \mathbf{R} \mathbf{A}) \mathbf{x}\|_{\mathbf{K}}^2 \leq (1 - \gamma \rho_{\min}) \|\mathbf{x}\|_{\mathbf{K}}^2, \quad \forall \mathbf{x} \in \mathbb{R}^{nd}.$$

<sup>180</sup> We thus observe that the operator (I − γA⊤RA) serves a similar purpose as the mixing matrix <sup>181</sup> in a average consensus algorithms and ρmin can be interpreted as the spectral radius of G similar

to [\[Koloskova et al.,](#page-9-2) [2020,](#page-9-2) Eq. (12)]. Moreover, if we define Q := (A⊤RA) † <sup>182</sup> such that it holds QA⊤RA = A⊤RAQ = K, Assumption [3.2](#page-4-2) implies that ρ −1 maxK ⪯ Q ⪯ ρ −1 <sup>183</sup> minK.

<sup>184</sup> Next we consider several assumptions on the noise variance of the random quantities in FSPDA:

Assumption 3.3. *For any fixed* x<sup>i</sup> ∈ <sup>R</sup> d <sup>185</sup> *,* i ∈ [n]*, there exists* σ<sup>i</sup> ≥ 0 *such that*

$$\mathbb{E}_{\xi_i \sim \mathbb{P}_i} [\|\nabla f_i(\mathbf{x}_i; \xi_i) - \nabla f_i(\mathbf{x}_i)\|^2] \leq \sigma_i^2. \quad (14)$$

*To simplify notations, we define* σ¯ 2 := (1/n) P<sup>n</sup> <sup>i</sup>=1 σ 2 i <sup>186</sup> *.*

Assumption 3.4. *For any fixed* x ∈ R nd <sup>187</sup> *, there exists* σ<sup>A</sup> ≥ 0 *such that*

$$\mathbb{E}_{\xi_a \sim \mathbb{P}_a} [\|\mathbf{A}(\xi_a)^\top \mathbf{A}\mathbf{x} - \mathbf{A}^\top \mathbf{R}\mathbf{A}\mathbf{x}\|^2] \leq \sigma_A^2 \|\mathbf{x}\|_{\mathbf{K}}^2. \quad (15)$$

Assumption [3.3](#page-5-0) is standard. Meanwhile for Assumption [3.4,](#page-5-1) the variance term σ <sup>188</sup> <sup>A</sup> measures the <sup>189</sup> quality of the random topology G(ξa) in approximating the expected graph Laplacian A⊤RA. The latter is important as it contributes to the variance in the drift term of FSPDA. Observe that σ 2 <sup>190</sup> A <sup>191</sup> decreases with the proportion of edges selected in each random subgraph G(ξa).

<sup>192</sup> To facilitate our discussions, we define the following quanitites:

$$\bar{\mathbf{x}}^t := \frac{1}{n} \sum_{i=1}^n \mathbf{x}_i^t, \quad \sum_{i=1}^n \|\mathbf{x}_i^t - \bar{\mathbf{x}}^t\|^2 = \|\mathbf{x}^t\|_K^2. \quad (16)$$

<sup>193</sup> Convergence of FSPDA-SA. We summarize the convergence rate for FSPDA-SA as follows. The proof <sup>194</sup> can be found in Appendix [C:](#page--1-2)

Theorem 3.5. *Under Assumptions [3.1,](#page-4-3) [3.2,](#page-4-2) [3.3,](#page-5-0) [3.4.](#page-5-1) Suppose that the step sizes satisfy the conditions defined in* [\(46\)](#page--1-3)*. Then, for any* T ≥ 1 *with the random stopping iteration* T ∼ Unif{0, ..., T − 1}*, the iterates generated by* FSPDA-SA *satisfy*

$$\mathbb{E} [\|\nabla F(\bar{\mathbf{x}}^\top)\|^2] \leq \frac{F_0 - f_\star}{\alpha T/8} + 8\alpha \mathbb{C}_\sigma \frac{\bar{\sigma}^2}{n}, \quad (17)$$

$$\mathbb{E} \left[ \sum_{i=1}^n \|\mathbf{x}_i^T - \bar{\mathbf{x}}^T\|^2 \right] \leq \frac{F_0 - f_\star}{\mathbf{a}\gamma\rho_{\min}T/8} + \frac{8\alpha^2 \mathcal{C}_\sigma \bar{\sigma}^2}{\mathbf{a}\gamma\rho_{\min} n}, \quad (18)$$

*for any* a > 0*, where* F0*,* C<sup>σ</sup> *are defined in* [\(44\)](#page--1-4)*,* [\(50\)](#page--1-5)*.*

195

Setting <sup>a</sup> <sup>=</sup> <sup>O</sup>(n/√ Tσ¯ <sup>2</sup>), α = p n/(Tσ¯ <sup>2</sup> <sup>196</sup> ) (and assuming σ >¯ 0), we have

$$\mathbb{E} \left[ \|\nabla F(\bar{\mathbf{x}}^T)\|^2 \right] = \mathcal{O} \left( \bar{\sigma} / \sqrt{nT} \right), \quad (19)$$

<sup>197</sup> which is the same *asymptotic convergence rate* as a centralized SGD algorithm that takes n stochastic <sup>198</sup> gradient samples uniformly from each agent, i.e., linear speedup [\[Lian et al.,](#page-10-1) [2017\]](#page-10-1). Also, using a = 1, the consensus error converges as a rate of E -P<sup>n</sup> <sup>i</sup>=1 ∥x T <sup>i</sup> − x¯ T∥ 2 = O(n 2σ 2 Aρmax/(T ρ<sup>2</sup> min <sup>199</sup> )) <sup>200</sup> under the same step size choice used in [\(19\)](#page-5-2). Notice that for T ≫ 1, the effect of random topology <sup>201</sup> only degrades the convergence of consensus error, keeping the transient rate in [\(19\)](#page-5-2) unaffected. If the gradients are deterministic (σ¯ = 0), setting a = (L <sup>2</sup>η∞ρmin) 1/3 <sup>202</sup> , α = α<sup>∞</sup> will yield a better convergence rate as E -∥∇F(x¯ <sup>T</sup>)∥ 2 = O(σ 4 A √ <sup>203</sup> n/T). Without a transient phase, the error due to random graph and coordinate sparsification is persistent through σ 4 <sup>204</sup> <sup>A</sup> in the above convergence rate.

<sup>205</sup> We further show that the convergence of FSPDA-SA can be accelerated if the objective function of [\(1\)](#page-0-0) <sup>206</sup> satisfies the Polyak-Lojasiewicz (PL) condition:

Assumption 3.6. *There exists a constant* µ > 0 *such that* 2µ(F(x) − f⋆) ≤ ∥∇F(x)∥ 2 , ∀x ∈ R d <sup>207</sup> *.*

<sup>208</sup> Assumption [3.6](#page-5-3) includes strongly convex functions as a special case, but also includes other non-<sup>209</sup> convex functions; see [\[Karimi et al.,](#page-9-13) [2016\]](#page-9-13). We observe:

Corollary 3.7. *Suppose the assumptions and step size conditions in Theorem [3.5](#page-5-4) hold. Furthermore, with Assumption [3.6,](#page-5-3) there exists* δ ∈ (0, 1) *such that for any* t ≥ 0*,*

$$\mathbb{E}_t[F_{t+1} - f_\star] \leq (1 - \delta)(F_t - f_\star) + \mathbb{C}_\sigma \alpha^2 \bar{\sigma}^2 / n \quad (20)$$

The proof can be found in Appendix [C.6.](#page--1-1) By setting α = c ln(T)/(n 2 <sup>211</sup> T) in [\(20\)](#page-5-5), with a carefully <sup>212</sup> chosen c and a sufficiently large T such that α ≤ α∞, we can ensure that

$$\mathbb{E} \left[ F(\bar{\mathbf{x}}^T) - f_\star + \|\mathbf{x}^T\|_{\mathbf{K}}^2 \right] = \mathcal{O} \left( \bar{\sigma}^2 \ln(T) / (\mu n T) \right) \quad (21)$$

In the case of deterministic gradient, i.e., σ¯ <sup>213</sup> <sup>2</sup> = 0, by setting α = α∞, [\(20\)](#page-5-5) ensures a linear convergence rate of E F(x¯ T ) − f<sup>⋆</sup> + ∥x <sup>T</sup> ∥ K = O((1 − δ) T <sup>214</sup> ), which shows that the performance <sup>215</sup> of FSPDA-SA is on par with [\[Nedic et al.,](#page-10-2) [2017,](#page-10-2) [Xu et al.,](#page-10-13) [2017\]](#page-10-13), despite it only requires one round of <sup>216</sup> (sparsified) transmission per iteration.

<sup>217</sup> Convergence of FSPDA-STORM. To exploit the benefits of control variates, we need an additional <sup>218</sup> assumption on the stochastic gradient map:

<sup>219</sup> Assumption 3.8. *Each stochastic function* fi(·; ξ) *is* Ls*-smooth in expectation, i.e., for* i = 1, . . . , n*,*

$$\mathbb{E}_\xi \left[ \|\nabla f_i(\mathbf{x}; \xi) - \nabla f_i(\mathbf{y}; \xi)\|^2 \right] \leq L_s^2 \|\mathbf{x} - \mathbf{y}\|^2 \forall \mathbf{x}, \mathbf{y} \in \mathbb{R}^d. \quad (22)$$

<sup>220</sup> [T](#page-9-12)he above assumption is also known as the mean-square smoothness condition, see [\[Cutkosky](#page-9-12) <sup>221</sup> [and Orabona,](#page-9-12) [2019\]](#page-9-12), which is strictly stronger than Assumption [3.1.](#page-4-3) We observe the following <sup>222</sup> convergence guarantee for FSPDA-STORM, whose proof can be found in Appendix [D.](#page--1-2)

Theorem 3.9. *Under Assumptions [3.1,](#page-4-3) [3.2,](#page-4-2) [3.3,](#page-5-0) [3.4,](#page-5-1) [3.8.](#page-6-1) Suppose that the step sizes satisfy the conditions in* [\(184\)](#page--1-7) *-* [\(214\)](#page--1-8)*. Then, for any* T ≥ 1 *with the random stopping iteration* T ∼ Unif{0, ..., T − 1}*, the iterates generated by* FSPDA-STORM *satisfy*

$$\mathbb{E} [\|\nabla F(\bar{\mathbf{x}}^\top)\|^2] \leq \frac{F_0 - f_\star}{T\alpha/4} + \frac{(\mathbf{e} \cdot 2a_x^2 + \mathbf{f} \cdot 4a_x^2 n)\bar{\sigma}^2}{\alpha/4}, \quad (23)$$

$$\mathbb{E} \left[ \sum_{i=1}^n \|\mathbf{x}_i^T - \bar{\mathbf{x}}^T\|^2 \right] \leq \frac{F_0 - f_\star}{T \mathbf{a} \gamma \rho_{\min} / 8} + \frac{(\mathbf{e} \cdot 2a_x^2 + \mathbf{f} \cdot 4a_x^2 n) \bar{\sigma}^2}{\mathbf{a} \gamma \rho_{\min} / 8}, \quad (24)$$

*where the constants* F0*,* a, e, f *are defined in* [\(110\)](#page--1-9)*.*

223

Setting α = O(¯σ <sup>−</sup>2/<sup>3</sup>T −1/3 ), η = O(n), γ = O(T −1/3 ), β = O(n <sup>−</sup><sup>1</sup>T −2/3 <sup>224</sup> ), a<sup>x</sup> = O(¯σ <sup>−</sup>4/<sup>3</sup>T −2/3 ), a<sup>λ</sup> = O(T −1/3 ), f = O(n <sup>−</sup><sup>1</sup>T 1/3 <sup>225</sup> ) (see [\(111\)](#page--1-10) - [\(117\)](#page--1-11)), and initializing the algorithm such that ∥v 0∥ <sup>K</sup> = O(T −2/3 ), ∥m<sup>0</sup> <sup>x</sup> − (1/n)1 ⊤ <sup>⊗</sup>∇f(x 0 <sup>2</sup> = O(T −1/3 <sup>226</sup> ) and ∥m<sup>0</sup> <sup>x</sup> − ∇xL(x 0 ,λ 0 <sup>2</sup> = O(T −1/3 <sup>227</sup> ), we have

$$\mathbb{E} [\|\nabla F(\bar{\mathbf{x}}^T)\|^2] = \mathcal{O}(\bar{\sigma}^{2/3}/T^{2/3}). \quad (25)$$

<sup>228</sup> In regard to the order of σ¯ and T, provided that n is small, the convergence rate of FSPDA-STORM <sup>229</sup> matches the lower bound [\[Arjevani et al.,](#page-9-14) [2023\]](#page-9-14) for non-convex functions under the same smoothness <sup>230</sup> assumption. Moreover, by the same choice of step sizes, the consensus error converges at the rate of E -P<sup>n</sup> <sup>i</sup>=1 ∥x T <sup>i</sup> − x¯ T∥ 2 = O(¯σ <sup>2</sup>/<sup>3</sup>nρ−<sup>1</sup> minT −2/3 <sup>231</sup> ). We remark that in [\(25\)](#page-6-2), the rate remains constant as <sup>232</sup> n increases such that FSPDA-STORM does not offer the same *linear speedup* observed in Theorem [3.5](#page-5-4) <sup>233</sup> for FSPDA-SA. Nevertheless, as T ≫ 1, the rate of FSPDA-STORM will surpass that of FSPDA-SA and <sup>234</sup> other decentralized algorithms on time varying topologies.

<sup>235</sup> Lastly, we provide detailed discussions on the convergence rates above, e.g., transient time, effects of <sup>236</sup> random topology, etc., in Appendix [B.](#page--1-12)

#### <sup>237</sup> 3.1 Insight from Analysis: Fixed Point Iteration of FSPDA-SA

From [\(8a\)](#page-3-2), the following recursive relationship holds for x¯ t : using the relation 1 <sup>238</sup> <sup>⊤</sup>A<sup>⊤</sup> = 0, we have

$$\bar{\mathbf{x}}^{t+1} = \bar{\mathbf{x}}^t - \frac{\alpha}{n} \sum_{i=1}^n \nabla f_i(\mathbf{x}_i^t; \xi_i^t). \quad (26)$$

This shows that the evolution of {x¯ t <sup>239</sup> }t≥<sup>0</sup> is similar to that of 'centralized' SGD applied on [\(1\)](#page-0-0) except <sup>240</sup> that the local gradients are evaluated on the local iterates. However, it is still not straightforward to analyze the convergence of FSPDA-SA as the update of x t involves the dual variable λ t <sup>241</sup> which lacks <sup>242</sup> an intuitive interpretation for constructing the right Lyapunov function.

<sup>243</sup> To this end, we study the fixed point(s) of [\(8\)](#page-3-0) to gain insights. Suppose that for some t⋆, the fixed point conditions <sup>E</sup>[λ <sup>t</sup>⋆+1 | ξ :t<sup>⋆</sup> ] = λ <sup>t</sup><sup>⋆</sup> , <sup>E</sup>[x <sup>t</sup>⋆+1 | ξ :t<sup>⋆</sup> ] = x <sup>t</sup><sup>⋆</sup> <sup>244</sup> hold. Since R is a diagonal matrix <sup>245</sup> with positive diagonal elements, we observe

$$\mathbb{E}[\lambda^{t_*+1} \mid \xi^{:t_*}] = \lambda^{t_*} \iff \mathbf{RAx}^{t_*} = \mathbf{0} \iff \mathbf{Ax}^{t_*} = \mathbf{0}, \quad (27)$$

<sup>246</sup> On the other hand, the primal update yields

$$\mathbb{E}[\mathbf{x}^{*+1} \mid \xi^{:t*}] = \mathbf{x}^{*+} - \alpha \nabla \mathbf{f}(\mathbf{x}^{*+}) - \eta \mathbf{A}^\top \boldsymbol{\lambda}^{*+}. \quad (28)$$

Since x t⋆ <sup>1</sup> = x t⋆ <sup>2</sup> = · · · = x t⋆ n <sup>247</sup> at the fixed point (due to [\(27\)](#page-6-3)), by the consensus condition across two <sup>248</sup> time steps, it implies

$$\begin{aligned}\mathbb{E}[\mathbf{x}^{t+1} \mid \xi^{:t*} - \mathbf{x}^{t*} &= (\mathbf{1} \otimes \mathbf{I}_d)(\bar{\mathbf{x}}^{t*} - \bar{\mathbf{x}}^{t*}) \\ &\iff \alpha \nabla \mathbf{f}(\mathbf{x}^{t*}) + \eta \mathbf{A}^\top \boldsymbol{\lambda}^{t*} = \frac{\alpha}{n} (\mathbf{1} \mathbf{1}^\top \otimes \mathbf{I}_d) \nabla \mathbf{f}(\mathbf{x}^{t*}) \\ &\iff \eta \mathbf{A}^\top \boldsymbol{\lambda}^{t*} = \alpha \left( \frac{1}{n} \mathbf{1} \mathbf{1}^\top - \mathbf{I}_n \right) \otimes \mathbf{I}_d \nabla \mathbf{f}((\mathbf{1} \otimes \mathbf{I}) \bar{\mathbf{x}}^{t*}).\end{aligned}\tag{29}$$

From [\(29\)](#page-7-0), we see that <sup>λ</sup>b<sup>t</sup> <sup>249</sup> shall converge to the difference between global and local gradient. Inspired <sup>250</sup> by the above, to facilitate the analysis later, we define

$$\mathbf{v}^t := \mathbf{A}^\top \boldsymbol{\lambda}^t + \frac{\alpha}{\eta} \nabla \mathbf{f}((\mathbf{1} \otimes \mathbf{I})\bar{\mathbf{x}}^t), \quad (30)$$

for any t ≥ 0. In particular, we see that ∥v t∥ <sup>251</sup> <sup>K</sup> measures the violation of [\(29\)](#page-7-0) in tracking the average <sup>252</sup> deterministic gradient using the dual variables. The latter will be instrumental in analyzing the <sup>253</sup> consensus error bound, as revealed in Lemma [C.2.](#page--1-13)

# <sup>254</sup> 4 Numerical Experiments

<sup>255</sup> This section reports the numerical experiments on practical performance of FSPDA. For the time <sup>256</sup> varying topology, we take an extreme setting where for each realization G(ξa), only one edge will <sup>257</sup> be selected uniformly at random from G. We evaluate the performance with the worst-agent metric, i.e., we present the training loss as maxi∈[n] F(x t i <sup>258</sup> ), and the stationarity/gradient-norm measure as maxi∈[n] ∥∇F(x t i )∥ 2 <sup>259</sup> . This captures the worst-case of the solutions produced by the algorithms. Unless otherwise specified, all algorithms are initialized with x 0 <sup>i</sup> = x¯ 0 <sup>260</sup> , and for FSPDA we initialize <sup>λ</sup>b<sup>0</sup> <sup>=</sup> <sup>m</sup><sup>0</sup> x,i = m<sup>0</sup> <sup>261</sup> λ,i = 0, and the stochastic gradients are estimated with a batch size of 256. In the <sup>262</sup> interest of space, omitted details and hyperparameters of the experiments can be found in Appendix [F.](#page--1-1)

<sup>263</sup> MNIST Experiments. The first set of experiments considers a moderate-scale setting of training a <sup>264</sup> one hidden layer feed-forward neural network with 100 hidden neurons (total number of parameters <sup>265</sup> d = 79,510) on the MNIST dataset with m = 60, 000 samples of 784-dimensional features.

 In the first experiment, we consider the static topology G as an Erdos-Renyi graph with connectivity of p = 0.5 and n = 10 agents. We compare the proposed FSPDA-SA, FSPDA-STORM with six benchmark algorithms utilizing different types of time-varying topology. Among them, DSGD [\[Koloskova et al.,](#page-9-2) [2020\]](#page-9-2) and Swarm-SGD [\[Nadiradze et al.,](#page-10-4) [2021\]](#page-10-4) use the general time varying topology setting as FSPDA where each edge of G(ξa) is active uniformly at random, in addition to random sparsification used FSPDA-SA and adaptive quantized used in Swarm-SGD; CHOCO-SGD [\[Koloskova et al.,](#page-9-15) [2019b\]](#page-9-15) takes G(ξa) as an broadcasting subgraph where one agent selects all his/her neighbors; Decen-Scaffnew [\[Mishchenko et al.,](#page-10-6) [2022\]](#page-10-6), LED [\[Alghunaim,](#page-9-3) [2024\]](#page-9-3), and K-GT [\[Liu et al.,](#page-10-7) [2024\]](#page-10-7) utilize local updates where G(ξa) is either taken as an empty topology, or as the static topology G. We configure these algorithms such that they have the same communication cost (in terms of bits transmitted over network) *on average*. For instance, the local update algorithms (Decen-Scaffnew, LED, K-GT) only communicate once using G every O |E|d k iterations to match the communication cost of k-coordinate sparse one-edge random graph used in FSPDA.

 The local objective function held by each agent is the cross-entropy classification loss on a local dataset with m<sup>i</sup> = 6000 samples, plus a regularization loss <sup>λ</sup> 2 ∥xi∥ with λ = 10−<sup>4</sup> , where x<sup>i</sup> are the weight parameters of the feed-forward neural network classifier. We split the training set into n = 10 disjoint sets such that each set contains only one class label and assign each set to one agent as its local dataset. Note that as we do not shuffle the data samples across local datasets, the local objective function held by different agents will become highly heterogeneous.

 Fig. [1](#page-8-0) compares the squared gradient norm, training loss, consensus error of the benchmarked algo- rithms. We first note that both FSPDA algorithms have significantly outperformed DSGD, Swarm-SGD on the general time varying topology as well as CHOCO-SGD. Meanwhile, the performance of FSPDA is comparable to the local update algorithms Decen-Scaffnew, LED, K-GT. Notice that the latter

![](_page_8_Figure_0.jpeg)

Figure 1: Feed-forward neural network classification training on MNIST using 10<sup>6</sup> iterations.

![](_page_8_Figure_2.jpeg)

Figure 2: Resnet-50 classification training on Imagenet.

 require additional synchronization steps which may not be suitable for random networks. Lastly, we notice that as T ≫ 1, FSPDA-STORM can slightly outperform FSPDA-SA due to its O(1/T<sup>2</sup>/<sup>3</sup> ) rate as shown in our analysis. We further expand the experiments by a series of ablation studies over data heterogeneity, sparsity levels, graph topologies, gradient noise and dual momentum in Appendix [E.](#page--1-14)

 Imagenet Experiments. The second set of experiments consider a large-scale setting for training a Resnet-50 network (total number of parameters d = 25,557,032) on the Imagenet dataset (training dataset of 1,281,168 images from 100 classes, re-scaled and cropped to 256 × 256 image dimensions). We consider cross-entropy classification loss plus the same L2 norm regularization loss as in the previous setup. We split the dataset across a network of n = 8 nodes where the static graph G is taken as the fully connected topology. The performance metrics are measured at the network average iterate x¯ t . Inspired by [\[Loshchilov and Hutter,](#page-10-14) [2016,](#page-10-14) Eq. (5)] we adopt a cosine learning rate scheduling with 5 epochs of linear warm up for every algorithm. In particular, the step sizes α, η of FSPDA-SA are scheduled simultaneously such that αt/η<sup>t</sup> remains constant, as illustrated in Appendix [F.](#page--1-1) We draw a batch of 128 samples to estimate the stochastic gradient.

 We focus on the communication efficiency and only compare FSPDA-SA, CHOCO-SGD, Swarm-SGD in this experiment due to limited resources. The results are reported in Figure [2](#page-8-1) that compare the test accuracy and training loss against iteration number and bits transmitted. When compared with CHOCO-SGD, FSPDA-SA achieves almost the same accuracy using one-edge random graphs with at least 100x reduction in communication cost on 100 epoch training. Also notice that further compressing the communication to 0.1% sparse coordinates in FSPDA-SA requires more training epochs to recover the same level of accuracy.

 Conclusions. This paper proposed a fully stochastic primal dual gradient algorithm (FSPDA) frame- work for decentralized optimization over arbitrarily time varying random networks. We utilize a new stochastic augmented Lagrangian function and apply SA to search for its saddle point. We develop two algorithms, one is by plain SA (FSPDA-SA), and one uses control variates for variance reduction (FSPDA-STORM). We prove that both algorithms achieve state-of-the-art convergence rates, while relaxing assumptions on both bounded heterogeneity and the type of time varying topologies.

# References


[1] Sulaiman A Alghunaim. Local exact-diffusion for decentralized optimization and learning. *IEEE Transactions on Automatic Control*, 2024. Yossi Arjevani, Yair Carmon, John C Duchi, Dylan J Foster, Nathan Srebro, and Blake Woodworth. Lower bounds for non-convex stochastic optimization. *Mathematical Programming*, 199(1): 165–214, 2023. Dimitri Bertsekas. *Nonlinear Programming*, volume 4. Athena Scientific, 2016. Pascal Bianchi, Walid Hachem, and Adil Salim. A fully stochastic primal-dual algorithm. *Optimiza- tion Letters*, 15(2):701–710, 2021. Tsung-Hui Chang, Mingyi Hong, Hoi-To Wai, Xinwei Zhang, and Songtao Lu. Distributed learning in the nonconvex world: From batch data to streaming and beyond. *IEEE Signal Processing Magazine*, 37(3):26–38, 2020. Ashok Cutkosky and Francesco Orabona. Momentum-based variance reduction in non-convex sgd. *Advances in neural information processing systems*, 32, 2019. Luyao Guo, Sulaiman A Alghunaim, Kun Yuan, Laurent Condat, and Jinde Cao. Revisiting decen- tralized proxskip: Achieving linear speedup. *arXiv preprint arXiv:2310.07983*, 2023. Davood Hajinezhad and Mingyi Hong. Perturbed proximal primal–dual algorithm for nonconvex nonsmooth optimization. *Mathematical Programming*, 176(1):207–245, 2019. Mingyi Hong, Davood Hajinezhad, and Ming-Min Zhao. Prox-pda: The proximal primal-dual algorithm for fast distributed nonconvex optimization and learning over networks. In *International Conference on Machine Learning*, pages 1529–1538. PMLR, 2017. Peter Kairouz, H Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, et al. Ad- vances and open problems in federated learning. *Foundations and trends® in machine learning*, 14(1–2):1–210, 2021. Hamed Karimi, Julie Nutini, and Mark Schmidt. Linear convergence of gradient and proximal- gradient methods under the polyak-łojasiewicz condition. In *Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2016, Riva del Garda, Italy, September 19-23, 2016, Proceedings, Part I 16*, pages 795–811. Springer, 2016. Anastasia Koloskova, Tao Lin, Sebastian U Stich, and Martin Jaggi. Decentralized deep learning with arbitrary communication compression. In *International Conference on Learning Representations*, 2019a. Anastasia Koloskova, Sebastian Stich, and Martin Jaggi. Decentralized stochastic optimization and gossip algorithms with compressed communication. In *International Conference on Machine Learning*, pages 3478–3487. PMLR, 2019b. Anastasia Koloskova, Nicolas Loizou, Sadra Boreiri, Martin Jaggi, and Sebastian Stich. A unified theory of decentralized sgd with changing topology and local updates. In *International Conference on Machine Learning*, pages 5381–5393. PMLR, 2020. Anastasiia Koloskova, Tao Lin, and Sebastian U Stich. An improved analysis of gradient tracking for decentralized machine learning. *Advances in Neural Information Processing Systems*, 34: 11422–11435, 2021. Dmitry Kovalev, Elnur Gasanov, Alexander Gasnikov, and Peter Richtarik. Lower bounds and optimal algorithms for smooth and strongly convex decentralized optimization over time-varying networks. *Advances in Neural Information Processing Systems*, 34:22325–22335, 2021. Dmitry Kovalev, Ekaterina Borodich, Alexander Gasnikov, and Dmitrii Feoktistov. Lower bounds and optimal algorithms for non-smooth convex decentralized optimization over time-varying networks. *arXiv preprint arXiv:2405.18031*, 2024.

[2] Jinlong Lei, Han-Fu Chen, and Hai-Tao Fang. Asymptotic properties of primal-dual algorithm for distributed stochastic optimization over random networks with imperfect communications. *SIAM Journal on Control and Optimization*, 56(3):2159–2188, 2018. Huan Li and Zhouchen Lin. Accelerated gradient tracking over time-varying graphs for decentralized optimization. *Journal of Machine Learning Research*, 25(274):1–52, 2024. Xiangru Lian, Ce Zhang, Huan Zhang, Cho-Jui Hsieh, Wei Zhang, and Ji Liu. Can decentralized algorithms outperform centralized algorithms? a case study for decentralized parallel stochastic gradient descent. *Advances in neural information processing systems*, 30, 2017. Yue Liu, Tao Lin, Anastasia Koloskova, and Sebastian U Stich. Decentralized gradient tracking with local steps. *Optimization Methods and Software*, pages 1–28, 2024. Ilan Lobel and Asuman Ozdaglar. Distributed subgradient methods for convex optimization over random networks. *IEEE Transactions on Automatic Control*, 56(6):1291–1306, 2010. Paolo Di Lorenzo and Gesualdo Scutari. Next: In-network nonconvex optimization. *IEEE Transac- tions on Signal and Information Processing over Networks*, 2(2):120–136, 2016. Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. *arXiv preprint arXiv:1608.03983*, 2016. Songtao Lu, Xinwei Zhang, Haoran Sun, and Mingyi Hong. Gnsd: A gradient-tracking based nonconvex stochastic algorithm for decentralized optimization. In *2019 IEEE Data Science Workshop (DSW)*, pages 315–321. IEEE, 2019. Konstantin Mishchenko, Grigory Malinovsky, Sebastian Stich, and Peter Richtárik. Proxskip: Yes! local gradient steps provably lead to communication acceleration! finally! In *International Conference on Machine Learning*, pages 15750–15769. PMLR, 2022. Giorgi Nadiradze, Amirmojtaba Sabour, Peter Davies, Shigang Li, and Dan Alistarh. Asynchronous decentralized sgd with quantized and local updates. *Advances in Neural Information Processing Systems*, 34:6829–6842, 2021. Angelia Nedic and Asuman Ozdaglar. Distributed subgradient methods for multi-agent optimization. *IEEE Transactions on Automatic Control*, 54(1):48–61, 2009. Angelia Nedic, Alex Olshevsky, and Wei Shi. Achieving geometric convergence for distributed optimization over time-varying graphs. *SIAM Journal on Optimization*, 27(4):2597–2633, 2017. Shi Pu, Alex Olshevsky, and Ioannis Ch Paschalidis. A sharp estimate on the transient time of distributed stochastic gradient descent. *IEEE Transactions on Automatic Control*, 67(11):5900– 5915, 2021. Tiancheng Qin, S Rasoul Etesami, and César A Uribe. Communication-efficient decentralized local sgd over undirected networks. In *2021 60th IEEE Conference on Decision and Control (CDC)*, pages 3361–3366. IEEE, 2021. Guannan Qu and Na Li. Harnessing smoothness to accelerate distributed optimization. *IEEE Transactions on Control of Network Systems*, 5(3):1245–1260, 2017. S Sundhar Ram, Angelia Nedic, and Venugopal V Veeravalli. Distributed stochastic subgradient ´ projection algorithms for convex optimization. *Journal of optimization theory and applications*, 147:516–545, 2010. Wei Shi, Qing Ling, Gang Wu, and Wotao Yin. Extra: An exact first-order algorithm for decentralized consensus optimization. *SIAM Journal on Optimization*, 25(2):944–966, 2015. Jinming Xu, Shanying Zhu, Yeng Chai Soh, and Lihua Xie. Convergence of asynchronous distributed gradient methods over stochastic networks. *IEEE Transactions on Automatic Control*, 63(2): 434–448, 2017.

[3] Chung-Yiu Yau and Hoi-To Wai. Fully stochastic distributed convex optimization on time-varying graph with compression. In *2023 62nd IEEE Conference on Decision and Control (CDC)*, pages 145–150. IEEE, 2023. Xinlei Yi, Lisha Yao, Tao Yang, Jemin George, and Karl H Johansson. Distributed optimization for second-order multi-agent systems with dynamic event-triggered communication. In *2018 IEEE Conference on Decision and Control (CDC)*, pages 3397–3402. IEEE, 2018. Xinlei Yi, Shengjun Zhang, Tao Yang, Tianyou Chai, and Karl H Johansson. Linear convergence of first-and zeroth-order primal–dual algorithms for distributed nonconvex optimization. *IEEE Transactions on Automatic Control*, 67(8):4194–4201, 2021.
# NeurIPS Paper Checklist

#### 1. Claims

 Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: [NA]

Guidelines:

 • The answer NA means that the abstract and introduction do not include the claims made in the paper. • The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers. • The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings. • It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: [NA]

#### Guidelines:

 • The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper. • The authors are encouraged to create a separate "Limitations" section in their paper. • The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be. • The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated. • The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon. • The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size. • If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness. • While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an impor- tant role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

#### 3. Theory assumptions and proofs

 Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Justification: [NA]

Guidelines:

 • The answer NA means that the paper does not include theoretical results. • All the theorems, formulas, and proofs in the paper should be numbered and cross- referenced. • All assumptions should be clearly stated or referenced in the statement of any theorems. • The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition. • Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.

#### 4. Experimental result reproducibility

 Question: Does the paper fully disclose all the information needed to reproduce the main ex- perimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: [NA]

Guidelines:

 • The answer NA means that the paper does not include experiments. • If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not. • If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable. • Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed. • While NeurIPS does not require releasing code, the conference does require all submis- sions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

 Question: Does the paper provide open access to the data and code, with sufficient instruc- tions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: [NA]

Guidelines:

 • The answer NA means that paper does not include experiments requiring code. • Please see the NeurIPS code and data submission guidelines ([https://nips.cc/](https://nips.cc/public/guides/CodeSubmissionPolicy) [public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details. • While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark). • The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ([https:](https://nips.cc/public/guides/CodeSubmissionPolicy) [//nips.cc/public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details. • The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc. • The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why. • At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable). • Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

#### 6. Experimental setting/details

 Question: Does the paper specify all the training and test details (e.g., data splits, hyper- parameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: [NA]

Guidelines:

 • The answer NA means that the paper does not include experiments. • The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them. • The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

 Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

 Justification: Due to limited computing resources and time constraints, we are unable to perform multiple runs of our algorithms and report the error bars. We will produce the error bar statistics if time permits.

Guidelines:

 • The answer NA means that the paper does not include experiments. • The authors should answer "Yes" if the results are accompanied by error bars, confi- dence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper. • The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions). • The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors).

 • It should be clear whether the error bar is the standard deviation or the standard error of the mean. • It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified. • For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates). • If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

#### 8. Experiments compute resources

 Question: For each experiment, does the paper provide sufficient information on the com- puter resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: [NA]

Guidelines:

 • The answer NA means that the paper does not include experiments. • The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage. • The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute. • The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

 Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?

Answer: [Yes]

Justification: [NA]

Guidelines:

 • The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. • If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics. • The authors should make sure to preserve anonymity (e.g., if there is a special consid-eration due to laws or regulations in their jurisdiction).

#### 10. Broader impacts

 Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: [NA]

Guidelines:

 • The answer NA means that there is no societal impact of the work performed. • If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact. • Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

 • The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster. • The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology. • If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

#### 11. Safeguards

 Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: [NA]

Guidelines:

 • The answer NA means that the paper poses no such risks. • Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters. • Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images. • We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

#### 12. Licenses for existing assets

 Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: [NA]

Guidelines:

 • The answer NA means that the paper does not use existing assets. • The authors should cite the original paper that produced the code package or dataset. • The authors should state which version of the asset is used and, if possible, include a URL. • The name of the license (e.g., CC-BY 4.0) should be included for each asset. • For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided. • If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, <paperswithcode.com/datasets> has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset. • For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

 • If this information is not available online, the authors are encouraged to reach out to the asset's creators.

#### 13. New assets

 Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: [NA]

Guidelines:

 • The answer NA means that the paper does not release new assets. • Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc. • The paper should discuss whether and how consent was obtained from people whose asset is used. • At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

#### 14. Crowdsourcing and research with human subjects

 Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: [NA]

Guidelines:

 • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribu- tion of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

#### 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

 Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: [NA]

Guidelines:

 • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

 Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: [NA]

Guidelines:

 • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (<https://neurips.cc/Conferences/2025/LLM>) for what should or should not be described.