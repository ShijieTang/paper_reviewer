# Online Convex Optimization with Heavy Tails: Old Algorithms, New Regrets, and Applications

Anonymous Author(s) Affiliation Address email

## Abstract

 In Online Convex Optimization (OCO), when the stochastic gradient has a finite variance, many algorithms provably work and guarantee a sublinear regret. How- ever, limited results are known if the gradient estimate has a heavy tail, i.e., the stochastic gradient only admits a finite p-th central moment for some p ∈ (1, 2]. Motivated by it, this work examines different old algorithms for OCO (e.g., Online Gradient Descent) in the more challenging heavy-tailed setting. Under the standard bounded domain assumption, we establish new regrets for these classical methods without any algorithmic modification. Remarkably, these regret bounds are fully optimal in all parameters (can be achieved even without knowing p), suggesting that OCO with heavy tails can be solved effectively without any extra operation (e.g., gradient clipping). Our new results have several applications. A particularly interesting one is the first provable convergence result for nonsmooth nonconvex optimization under heavy-tailed noise without gradient clipping.

## 1 Introduction

 This paper studies the online learning problem with convex losses, also known as Online Convex Optimization (OCO), a widely applicable framework that learns under streaming data [\[4,](#page-9-0) [10,](#page-9-1) [27,](#page-10-0) [35\]](#page-11-0). OCO has tons of implications for both designing and analyzing algorithms in different areas, for example, stochastic optimization [\[8,](#page-9-2) [23,](#page-10-1) [14\]](#page-10-2), PAC learning [\[3\]](#page-9-3), control theory [\[1,](#page-9-4) [11\]](#page-9-5), etc.

 In an OCO problem, a learning algorithm A would interact with the environment in T rounds, where T ∈ N can be either known or unknown. Formally, in each round round t, the learner A first decides an output x<sup>t</sup> ∈ X from a convex feasible set X ⊆ <sup>R</sup> d , then the environment reveals a convex loss function ℓ<sup>t</sup> : X → <sup>R</sup>, and A incurs a loss of ℓt(xt). After T many rounds, the quantity measuring the algorithm's performance is called regret, defined relative to any fixed competitor x ∈ X as follows:

$$\mathbb{R}_T^A(\mathbf{x}) \triangleq \sum_{t=1}^T \ell_t(\mathbf{x}_t) - \ell_t(\mathbf{x}).$$

 In the classical setting, instead of observing full information about ℓt, the learner A is only guaranteed to receive a subgradient ∇ℓt(xt) ∈ ∂ℓt(xt) at its decision, where ∂ℓt(xt) denotes the subdifferential set of ℓ<sup>t</sup> at x<sup>t</sup> [\[33\]](#page-11-1). This turns out to be enough for our purpose of minimizing the regret, since any OCO problem can be reduced to an Online Linear Optimization (OLO) instance via the inequality ℓt(xt) − ℓt(x) ≤ ⟨∇ℓt(xt), x<sup>t</sup> − x⟩, which holds due to convexity. Under the standard bounded domain assumption, i.e., X has a finite diameter D, many classical algorithms, e.g., Online Gradient Descent (OGD) [\[50\]](#page-12-0), guarantee an optimal sublinear regret GD√ T for G-Lipschitz ℓt. Even better, in the case that computing an exact subgradient is intractable, and one could only query a stochastic estimate g<sup>t</sup> satisfying <sup>E</sup> [g<sup>t</sup> | xt] ∈ ∂ℓt(xt), the OGD algorithm can still solve OCO effectively with

a provable (G + σ)D √ <sup>33</sup> T regret bound in expectation if the stochastic noise g<sup>t</sup> − ∇ℓt(xt) has a bounded second moment σ 2 <sup>34</sup> for some σ ≥ 0, which is called the finite variance condition.

 However, many works have pointed out that even for the easier stochastic optimization (i.e., ℓ<sup>t</sup> = F for a common F), the typical finite variance assumption is too optimistic and can be violated in different tasks [\[12,](#page-9-6) [37,](#page-11-2) [45\]](#page-12-1), and their observations suggest that the stochastic gradient only admits a finite p-th central moment upper bounded by σ p for some p ∈ (1, 2], which is named heavy-tailed noise. This new assumption generalizes the classical finite variance condition (p = 2) and becomes challenging when p < 2. A particular evidence is that the famous Stochastic Gradient Descent (SGD) algorithm [\[32\]](#page-11-3) (which is exactly OGD for stochastic optimization) provably diverges [\[45\]](#page-12-1).

 Though heavy-tailed stochastic optimization has been extensively studied [\[18,](#page-10-3) [26,](#page-10-4) [34\]](#page-11-4), limited results are known for OCO with heavy tails. The only work under this topic that we are aware of is [\[47\]](#page-12-2), which established a parameter-free regret bound in high probability (more discussions provided later). However, their algorithm includes many nontrivial modifications like gradient clipping and significantly deviates from the existing simple OCO algorithms used in practice. Especially, consider OGD as an example. Though the heavy-tailed issue is known, OGD (or just think of it as SGD) still works (sometimes very well) in practice even without gradient clipping and is arguably one of the most popular optimizers, which seemingly contradicts the theory of unconvergence mentioned before. This indicates that, for classical OCO algorithms under heavy-tailed noise, a huge gap exists between the empirical convergence (or even the effective practical performance) and theoretical guarantees. Therefore, we are naturally led to the following question:

<sup>53</sup> *In what context can old OCO algorithms work under heavy tails, in what sense, and to what extent?*

## <sup>54</sup> 1.1 Contributions

<sup>55</sup> Motivated by the above question, we examine three classical algorithms for OCO: Online Gradient <sup>56</sup> Descent (OGD) [\[50\]](#page-12-0), Dual Averaging (DA) [\[25,](#page-10-5) [43\]](#page-12-3), and AdaGrad [\[9,](#page-9-7) [22\]](#page-10-6), and answer it as follows:

*Under the standard bounded domain assumption, the in-expectation regret* E R A T (x) <sup>57</sup> *is finite and* <sup>58</sup> *optimal for any* A ∈ {OGD, DA, AdaGrad}*, without any algorithmic modification.*

<sup>59</sup> In detail, our new results for heavy-tailed OCO are summarized here:

- We prove the only and the first optimal regret bound E
- -R A T
- (x) <sup>≲</sup> GD√ T +σDT<sup>1</sup>/<sup>p</sup> <sup>60</sup> , ∀x ∈ X for <sup>61</sup> any A ∈ {OGD, DA, AdaGrad}. Remarkably, AdaGrad can achieve this result without knowing <sup>62</sup> any of the Lipschitz parameter G, noise level σ, and tail index p. <sup>63</sup> • We extend the analysis of OGD to Online Strongly Convex Optimization with heavy tails and establish the first provable result E
  - -R OGD T
- (x) ≲ G<sup>2</sup> log T <sup>µ</sup> + σ <sup>p</sup>G2−<sup>p</sup> µ T 2−p <sup>64</sup> , ∀x ∈ X , where µ > 0 is the modulus of strong convexity and T 0 <sup>65</sup> should be read as log T.

<sup>66</sup> Based on the new regret bounds for OCO with heavy tails, we provide the following applications:

<sup>67</sup> • For nonsmooth convex optimization with heavy tails, we show the first optimal in-expectation rate

GD/√

T + σD/T<sup>1</sup>−1/<sup>p</sup>

<sup>68</sup> achieved without gradient clipping, which applies to both the average <sup>69</sup> iterate and last iterate, demonstrating that SGD does converge once the domain is bounded. <sup>70</sup> • For nonsmooth nonconvex optimization with heavy tails, we show the first provable sample

complexity of G<sup>2</sup>

δ −1 ϵ <sup>−</sup><sup>3</sup> + σ

p <sup>p</sup>−<sup>1</sup> δ −1 ϵ − 2p−1 <sup>p</sup>−<sup>1</sup> <sup>71</sup> for finding a (δ, ϵ)-stationary point without gradient <sup>72</sup> clipping. Moreover, we give the first convergence result when the problem-dependent parameters

<sup>73</sup> (like G, σ, and p) are unknown in advance.

## <sup>74</sup> 1.2 Discussion on [\[47\]](#page-12-2)

 As noted, [\[47\]](#page-12-2) is the only work for OCO with heavy tails, as far as we know. There are two major discrepancies between them and us. First, they consider the case where the feasible set X is unbounded and aim to establish a parameter-free regret bound, i.e., the regret bound has a linear dependency on ∥x∥ (up to an extra polylog ∥x∥) for any competitor x ∈ X . Second, they focus on high-probability rather than in-expectation analysis. As such, their regret is in the form of

R A T (x) ≲ (G + σ) ∥x∥ T 1/p <sup>80</sup> , ∀x ∈ X (up to extra polylogarithmic factors) with high probability. <sup>81</sup> Without a doubt, their setting is harder than ours implying their bound is stronger as it can convert to an in-expectation regret E -R A T (x) ≲ (G + σ)DT<sup>1</sup>/<sup>p</sup> <sup>82</sup> for any bounded domain X with a diameter D. <sup>83</sup> We emphasize that the motivation behind [\[47\]](#page-12-2) differs heavily from ours. They aim to solve heavy-<sup>84</sup> tailed OCO with a new proposed method that contains many nontrivial technical tricks, including <sup>85</sup> gradient clipping, artificially added regularization, and solving the additional fixed-point equation. <sup>86</sup> However, their result cannot reflect why the existing simple OCO algorithms like OGD work in <sup>87</sup> practice under heavy-tailed noise. In contrast, our goal is to examine whether, when, and how the <sup>88</sup> classical OCO algorithms work under heavy tails, thereby filling the missing piece in the literature.

Moreover, we would like to mention two drawbacks of [\[47\]](#page-12-2). First, though the T 1/p <sup>89</sup> regret seems <sup>90</sup> tight as it matches the lower bound [\[24,](#page-10-7) [30,](#page-11-5) [41\]](#page-11-6), this may not be the best, since an optimal bound should recover the standard √ <sup>91</sup> T regret in the deterministic case (i.e., σ = 0), as one can imagine. <sup>92</sup> This suggests that their bound is not entirely optimal. Second, we remark that they require knowing <sup>93</sup> both problem-dependent parameters G, σ, p and time horizon T in the algorithm, which may be hard to satisfy in the online setting. In comparison, our regret bound GD√ T + σDT<sup>1</sup>/<sup>p</sup> <sup>94</sup> is fully optimal <sup>95</sup> in all parameters. Importantly, AdaGrad can achieve it while oblivious to the problem information.

## <sup>96</sup> 2 Preliminary

<sup>97</sup> Notation. <sup>N</sup> denotes the set of natural numbers (excluding 0). [T] ≜ {1, . . . , T} , ∀T ∈ <sup>N</sup>. a ∧ b ≜ <sup>98</sup> min {a, b} and a ∨ b ≜ max {a, b}. We write a ≲ b if a ≤ Cb for a universal constant C > 0. <sup>99</sup> ⌊·⌋ and ⌈·⌉ respectively represent the floor and ceiling functions. ⟨·, ·⟩ denotes the Euclidean inner product and ∥·∥ ≜ p ⟨·, ·⟩ is the standard 2-norm. Given x ∈ <sup>R</sup> d and D > 0, B d <sup>100</sup> (x, D) is the Euclidean ball in R d centered at x with a radius D. In the case x = 0, we use the shorthand B d <sup>101</sup> (D). Given a nonempty closed convex set A ⊆ R d <sup>102</sup> , Π<sup>A</sup> is the Euclidean projection operator onto A. For a <sup>103</sup> convex function f, ∂f(x) denotes its subgradient set at x.

<sup>104</sup> *Remark* 1*.* We choose the Euclidean norm only for simplicity. Extending the results in this work to <sup>105</sup> any general norm is straightforward.

<sup>106</sup> This work studies OCO in the context of Assumption [1.](#page-2-0)

<sup>107</sup> Assumption 1. *We consider the following series of assumptions:*

• X ⊂ R d

- <sup>108</sup> *is a nonempty closed convex set bounded by* D*, i.e.,* supx,y∈X ∥x − y∥ ≤ D*.* <sup>109</sup> • ℓ<sup>t</sup> : X → <sup>R</sup> *is convex for all* t ∈ [T]*.* <sup>110</sup> • ℓ<sup>t</sup> *is* G*-Lipschitz on* X *, i.e.,* ∥∇ℓt(x)∥ ≤ G, ∀x ∈ X , ∇ℓt(x) ∈ ∂ℓt(x)*, for all* t ∈ [T]*.*
- *Given a point* x<sup>t</sup> ∈ X *at the* t*-th iteration, one can query* g<sup>t</sup> ∈ <sup>R</sup> d <sup>111</sup> *satisfying* ∇ℓt(xt) ≜ <sup>E</sup> [g<sup>t</sup> | Ft−1] ∈ ∂ℓt(xt) *and* <sup>E</sup> ∥ϵt∥ p ≤ σ p <sup>112</sup> *for some* p ∈ (1, 2] *and* σ ≥ 0*, where* F<sup>t</sup> ≜ σ(g<sup>1</sup> , . . . , g<sup>t</sup> <sup>113</sup> ) *denotes the natural filtration and* ϵ<sup>t</sup> <sup>≜</sup> g<sup>t</sup> − ∇ℓt(xt) *is the stochastic noise.* <sup>114</sup> *Remark* 2*.* D is recognized as known, like ubiquitously assumed in the OCO literature. Moreover, <sup>115</sup> x<sup>t</sup> denotes the decision/output of the online learning algorithm by default. <sup>116</sup> In Assumption [1,](#page-2-0) the first three points are standard, and the fourth is the heavy-tailed noise assumption.

<sup>117</sup> In particular, p = 2 recovers the standard finite variance condition.

## <sup>118</sup> 3 Old Algorithms under Heavy Tails

<sup>119</sup> In this section, we revisit three classical algorithms for OCO: OGD, DA, and AdaGrad, whose regret <sup>120</sup> bounds are well-studied in the finite variance case but remain unknown under heavy-tailed noise.

 The basic idea of proving these algorithms work under heavy tails is to leverage the boundness property of X . We will describe it in more detail using OGD as an illustrated example. The analysis of DA follows a similar way at a high level, but differs in some details. However, though AdaGrad can be viewed as OGD with an adaptive stepsize, the way to utilize the boundness property is entirely different. All formal proofs are deferred to the appendix due to space limitations.

## <sup>126</sup> 3.1 New Regret for Online Gradient Descent

Algorithm 1 Online Gradient Descent (OGD) [\[50\]](#page-12-0)

Input: initial point x<sup>1</sup> ∈ X , stepsize η<sup>t</sup> > 0

for t = 1 to T do

xt+1 = Π<sup>X</sup> (x<sup>t</sup> − ηtg<sup>t</sup>

)

end for

<sup>127</sup> We begin from arguably the most basic algorithm for OCO, Online Gradient Descent (OGD). <sup>128</sup> A well known analysis. The regret bound of OGD has been extensively studied [\[10,](#page-9-1) [27,](#page-10-0) [35\]](#page-11-0). The <sup>129</sup> most well known analysis is perhaps the following one: for any x ∈ X , there is

$$\|\mathbf{x}_{t+1} - \mathbf{x}\|^2 = \|\Pi_{\mathcal{X}}(\mathbf{x}_t - \eta_t \mathbf{g}_t) - \Pi_{\mathcal{X}}(\mathbf{x})\|^2 \leq \|\mathbf{x}_t - \eta_t \mathbf{g}_t - \mathbf{x}\|^2,$$

<sup>130</sup> where the inequality holds by the nonexpansive property of Π<sup>X</sup> . Expanding both sides and rearranging <sup>131</sup> terms yield that

$$\langle g_t, x_t - x \rangle \leq \frac{\|x_t - x\|^2 - \|x_{t+1} - x\|^2}{2\eta_t} + \frac{\eta_t \|g_t\|^2}{2}. \quad (1)$$

If g<sup>t</sup> <sup>132</sup> admits a finite variance, i.e., p = 2 in Assumption [1,](#page-2-0) taking expectations on both sides, then following a standard analysis for η<sup>t</sup> = D (G+σ) √ t (or η<sup>t</sup> = D (G+σ) √ T <sup>133</sup> if T is known) gives the regret

$$\mathbb{E} [\mathbf{R}_T^{\text{OGD}}(\mathbf{x})] \lesssim (G + \sigma) D \sqrt{T}, \forall \mathbf{x} \in \mathcal{X}.$$

<sup>134</sup> However, the step of taking expectations on the R.H.S. of [\(1\)](#page-3-0) crucially relies on the finite variance condition of g<sup>t</sup> <sup>135</sup> . Therefore, one may naturally think OGD would not guarantee a finite regret if p < 2.

A less well known analysis[<sup>1</sup>](#page-3-1) . As discussed, the failure of the above proof under heavy-tailed noise is due to [\(1\)](#page-3-0). Therefore, if a tighter inequality than [\(1\)](#page-3-0) exists, then it might be possible to show that OGD still works for p < 2. However, does it exist?

<sup>139</sup> Actually, there is another less well known analysis to produce a better inequality than [\(1\)](#page-3-0). That is, <sup>140</sup> first showing for any x ∈ X , by the optimality condition of the update rule,

$$\langle g_t, x_{t+1} - x \rangle \leq \frac{\langle x_t - x_{t+1}, x_{t+1} - x \rangle}{\eta_t} = \frac{\|x_t - x\|^2 - \|x_{t+1} - x\|^2 - \|x_t - x_{t+1}\|^2}{2\eta_t},$$

<sup>141</sup> and then obtaining

$$\langle \mathbf{g}_t, \mathbf{x}_t - \mathbf{x} \rangle \leq \frac{\|\mathbf{x}_t - \mathbf{x}\|^2 - \|\mathbf{x}_{t+1} - \mathbf{x}\|^2}{2\eta_t} + \langle \mathbf{g}_t, \mathbf{x}_t - \mathbf{x}_{t+1} \rangle - \frac{\|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2}{2\eta_t}. \quad (2)$$

Note that [\(2\)](#page-3-2) is tighter than [\(1\)](#page-3-0) as ⟨g<sup>t</sup> , <sup>x</sup><sup>t</sup> <sup>−</sup> <sup>x</sup>t+1⟩ ≤ ∥gt∥ ∥x<sup>t</sup> <sup>−</sup> <sup>x</sup>t+1∥ ≤ <sup>η</sup>t∥gt<sup>∥</sup> <sup>2</sup> + ∥xt−xt+1∥ 2η<sup>t</sup> <sup>142</sup> , <sup>143</sup> where the first step is due to Cauchy-Schwarz inequality and the second one is by AM-GM inequality.

 Handle p < 2 in a simple way. Though we have tightened [\(1\)](#page-3-0) into [\(2\)](#page-3-2), can inequality [\(2\)](#page-3-2) help to overcome heavy tails? The answer is surprisingly positive, and our solution is fairly simple. Instead of directly applying AM-GM inequality in the second step, we recall g<sup>t</sup> = ∇ℓt(xt) + ϵ<sup>t</sup> and use triangle inequality to obtain

$$\langle g_t, x_t - x_{t+1} \rangle \leq \|g_t\| \|x_t - x_{t+1}\| \leq (\|\nabla \ell_t(x_t)\| + \|\epsilon_t\|) \|x_t - x_{t+1}\|. \quad (3)$$

<sup>148</sup> On the one hand, by ∥∇ℓt(xt)∥ ≤ G and AM-GM inequality, there is

$$\|\nabla \ell_t(\mathbf{x}_t)\| \|\mathbf{x}_t - \mathbf{x}_{t+1}\| \leq G \|\mathbf{x}_t - \mathbf{x}_{t+1}\| \leq \eta_t G^2 + \frac{\|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2}{4\eta_t}. \quad (4)$$

<sup>1</sup>To clarify, the phrase "less well known" is compared to the first one. This analysis itself is also well known.

On the other hand, let p<sup>⋆</sup> ≜ p p−1 and C(p) ≜ (4p−4)<sup>p</sup>−<sup>1</sup> p <sup>p</sup> <sup>149</sup> , we have

$$\begin{aligned} \|\epsilon_t\| \|\mathbf{x}_t - \mathbf{x}_{t+1}\| &= \left( \frac{4\eta_t}{\mathbf{p}_*} \right)^{\frac{1}{\mathbf{p}_*}} \|\epsilon_t\| \|\mathbf{x}_t - \mathbf{x}_{t+1}\|^{1-\frac{2}{\mathbf{p}_*}} \cdot \left( \frac{\mathbf{p}_* \|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2}{4\eta_t} \right)^{\frac{1}{\mathbf{p}_*}} \\ &\stackrel{(a)}{\leq} \frac{\left( \frac{4\eta_t}{\mathbf{p}_*} \right)^{\frac{\mathbf{p}}{\mathbf{p}_*}} \|\epsilon_t\|^{\mathbf{p}} \|\mathbf{x}_t - \mathbf{x}_{t+1}\|^{\mathbf{p}-\frac{2\mathbf{p}}{\mathbf{p}_*}}}{\mathbf{p}} + \frac{\|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2}{4\eta_t} \\ &\stackrel{(b)}{\leq} C(\mathbf{p})\eta_t^{\mathbf{p}-1} \|\epsilon_t\|^{\mathbf{p}} D^{2-\mathbf{p}} + \frac{\|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2}{4\eta_t}, \end{aligned} \quad (5)$$

where (a) is by Young's inequality and (b) is due to ∥x<sup>t</sup> − xt+1∥ ≤ D, p<sup>⋆</sup> = p p−1 <sup>150</sup> , and C(p) = (4p−4)<sup>p</sup>−<sup>1</sup> p <sup>p</sup> <sup>151</sup> . Next, we plug [\(4\)](#page-3-3) and [\(5\)](#page-4-0) back into [\(3\)](#page-3-4), then combine with [\(2\)](#page-3-2) to know

$$\langle g_t, x_t - x \rangle \leq \frac{\|x_t - x\|^2 - \|x_{t+1} - x\|^2}{2\eta_t} + \eta_t G^2 + C(p)\eta_t^{p-1} \|\epsilon_t\|^p D^{2-p}. \quad (6)$$

Notably, the term ∥ϵt∥ p <sup>152</sup> has a correct exponent p. Thus, we can safely take expectations on both sides. <sup>153</sup> Finally, a standard analysis yields the following Theorem [1](#page-4-1) (see Appendix [A](#page-13-0) for a formal proof).

Theorem 1. *Under Assumption [1,](#page-2-0) taking* η<sup>t</sup> = D G √ t ∧ D σt1/<sup>p</sup> <sup>154</sup> *in* OGD *(Algorithm [1\)](#page-3-5), we have*

$$\mathbb{E} [R_T^{\text{OGD}}(\mathbf{x})] \lesssim GD\sqrt{T} + \sigma DT^{1/p}, \forall \mathbf{x} \in \mathcal{X}.$$

 As far as we know, Theorem [1](#page-4-1) is the first and the only provable result for OGD under heavy tails. Remarkably, it is not only tight in T [\[24,](#page-10-7) [30,](#page-11-5) [41\]](#page-11-6) but also fully optimal in all parameters, in contrast to the bound (G + σ)DT<sup>1</sup>/<sup>p</sup> of [\[47\]](#page-12-2). This reveals that OCO with heavy tails can be optimally solved as effectively as the finite variance case once the domain is bounded, a classical condition adapted in many existing works.

 Strongly convex functions. We highlight that the above idea can also be applied to Online Strongly Convex Optimization and leads to a sublinear regret T −<sup>p</sup> better than T 1/p . This extension can be found in Appendix [A.](#page-13-0)

## <sup>163</sup> 3.2 New Regret for Dual Averaging

Algorithm 2 Dual Averaging (DA) [\[25,](#page-10-5) [43\]](#page-12-3) Input: initial point x<sup>1</sup> ∈ X , stepsize η<sup>t</sup> > 0

**for** 
$$t$$
 **to**  $T$  **do**  $x_{t+1} = \Pi_{\mathcal{X}}(x_1 - \eta_t \sum_{s=1}^t g_s)$   
**end for**

<sup>164</sup> *Remark* 3*.* It is known that DA is a special realization of the more general Follow-the-Regularized-<sup>165</sup> Leader (FTRL) framework [\[21\]](#page-10-8). To keep the work concise, we only focus on DA. The key idea to <sup>166</sup> prove Theorem [2](#page-4-2) can directly extend to show new regret for FTRL under heavy-tailed noise.

 We turn our attention to the second candidate, the Dual Averaging (DA) algorithm, which is given in Algorithm [2.](#page-4-3) Though DA coincides with OGD when X = R d and η<sup>t</sup> = η, these two methods in general are not equivalent and can have significant performance differences in practice. Therefore, it is also important to understand DA under heavy tails.

 Despite the proof strategies for OGD and DA are in different flavors (even for p = 2), the basic idea presented before for OGD still works here, i.e., apply the boundness property of X to make the term ∥ϵt∥ have a correct exponent. Armed with this thought, we can prove the following new regret bound for DA under heavy-tailed noise. We refer the reader to Appendix [B](#page-14-0) for its proof.

Theorem 2. *Under Assumption [1,](#page-2-0) taking* η<sup>t</sup> = D G √ t ∧ D σt1/<sup>p</sup> <sup>175</sup> *in* DA *(Algorithm [2\)](#page-4-3), we have*

$$\mathbb{E} [R_T^{\text{DA}}(\mathbf{x})] \lesssim GD\sqrt{T} + \sigma DT^{1/p}, \forall \mathbf{x} \in \mathcal{X}.$$

<sup>176</sup> As far as we know, Theorem [2](#page-4-2) is the first provable and optimal regret for DA under heavy tails. It <sup>177</sup> guarantees the same tight bound as in Theorem [1](#page-4-1) up to different constants.

## <sup>178</sup> 3.3 New Regret for AdaGrad

Algorithm 3 AdaGrad [\[9,](#page-9-7) [22\]](#page-10-6)

Input: initial point x<sup>1</sup> ∈ X , stepsize η > 0

for t = 1 to T do <sup>η</sup><sup>t</sup> <sup>=</sup> ηV <sup>−</sup>1/<sup>2</sup>

<sup>t</sup> where V<sup>t</sup> =

P<sup>t</sup>

<sup>s</sup>=1 ∥gs∥

2

xt+1 = Π<sup>X</sup> (x<sup>t</sup> − ηtg<sup>t</sup>

)

end for

<sup>179</sup> *Remark* 4*.* Algorithm [3](#page-5-0) is also named AdaGrad-Norm (e.g., [\[42\]](#page-12-4)). We simply call it AdaGrad. It is <sup>180</sup> straightforward to generalize Theorem [3](#page-5-1) below to the per-coordinate update version. <sup>181</sup> Although Theorems [1](#page-4-1) and [2](#page-4-2) are optimal, they both suffer from an undesired point. That is, the stepsize η<sup>t</sup> = D G √ t ∧ D σt1/<sup>p</sup> <sup>182</sup> requires knowing all problem-dependent parameters. However, it may not <sup>183</sup> be easy to obtain them in an online setting. Especially, it heavily depends on the prior information <sup>184</sup> about the tail index p, which is hard to know (even approximately) in advance. In other words, they <sup>185</sup> both lack the adaptive property to an unknown environment. <sup>187</sup> AdaGrad is just OGD with an adaptive stepsize. However, it is this adaptive stepsize that can help us <sup>188</sup> to overcome the above undesired point. Theorem 3. *Under Assumption [1,](#page-2-0) taking* <sup>η</sup> <sup>=</sup> D/√ <sup>189</sup> 2 *in* AdaGrad *(Algorithm [3\)](#page-5-0), we have*

<sup>186</sup> To handle this issue, we consider AdaGrad, a classical adaptive algorithm for OCO. As can be seen,

$$\mathbb{E} \left[ \mathbf{R}_T^{\text{AdaGrad}}(\mathbf{x}) \right] \lesssim GD\sqrt{T} + \sigma DT^{1/p}, \forall \mathbf{x} \in \mathcal{X}.$$

<sup>190</sup> *Remark* 5*.* We also establish a similar result for DA with an adaptive stepsize. See Theorem [7](#page-15-0) in <sup>191</sup> Appendix [B](#page-14-0) for details.

 Theorem [3](#page-5-1) provides the first regret bound for AdaGrad under heavy tails. Impressively, it is optimal even without knowing any of G, σ, and p. This surprising result once again demonstrates the power of the adaptive method, indicating it is robust to an unknown environment and even heavy-tailed noise, which may partially explain the favorable performance of many adaptive optimizers designed based on AdaGrad like RMSProp [\[40\]](#page-11-7) and Adam [\[14\]](#page-10-2).

 We point out that the key to establishing Theorem [3](#page-5-1) differs from the idea used before for OGD and DA. Actually, Theorem [3](#page-5-1) can be obtained in an embarrassingly simple way. It is known that AdaGrad with <sup>η</sup> <sup>=</sup> D/√ 2 on a bounded domain guarantees the following path-wise regret

$$\sum_{t=1}^T \langle \mathbf{g}_t, \mathbf{x}_t - \mathbf{x} \rangle \lesssim D \sqrt{\sum_{t=1}^T \|\mathbf{g}_t\|^2}. \quad (7)$$

Observe that qP<sup>T</sup> <sup>t</sup>=1 ∥gt∥ <sup>2</sup> ≲ qP<sup>T</sup> <sup>t</sup>=1 ∥∇ℓt(xt)∥ <sup>2</sup> + qP<sup>T</sup> <sup>t</sup>=1 ∥ϵt∥ <sup>2</sup> ≤ G √ T + P<sup>T</sup> <sup>t</sup>=1 ∥ϵt∥ p 1 p <sup>200</sup> , where the last step is due to ∥·∥<sup>2</sup> ≤ ∥·∥<sup>p</sup> <sup>201</sup> for any p ∈ [1, 2]. After taking expectations on both sides of [\(7\)](#page-5-2) and applying Hölder's inequality to obtain E P<sup>T</sup> <sup>t</sup>=1 ∥ϵt∥ p 1 p ≤ P<sup>T</sup> <sup>t</sup>=1 <sup>E</sup> -∥ϵt∥ p 1 p ≤ σT 1 <sup>p</sup> <sup>202</sup> , <sup>203</sup> we conclude Theorem [3.](#page-5-1) To make the work self-consistent, we produce the formal proof of Theorem <sup>204</sup> [3](#page-5-1) in Appendix [C.](#page-16-0)

## <sup>205</sup> 4 Applications

<sup>206</sup> We provide some applications based on the new regret bounds established in Section [3.](#page-2-1) The basic <sup>207</sup> problem we study is optimizing a single objective F, which could be either convex or nonconvex.

## <sup>208</sup> 4.1 Nonsmooth Convex Optimization

<sup>210</sup> Convergence of the average iterate. First, we focus on convergence in average. By the classical <sup>211</sup> online-to-batch conversion [\[3\]](#page-9-3), the following corollary immediately holds.

Corollary 1. *Under Assumption [1](#page-2-0) for* ℓt(x) = ⟨∇F(xt), x⟩ *and let* x¯<sup>T</sup> ≜ 1 T P<sup>T</sup> <sup>212</sup> <sup>t</sup>=1 xt*, for any* <sup>213</sup> A ∈ {OGD, DA, AdaGrad}*, we have*

$$\mathbb{E}[F(\bar{x}_T) - F(\mathbf{x})] \leq \frac{\mathbb{E}[R_T^A(\mathbf{x})]}{T} \lesssim \frac{GD}{\sqrt{T}} + \frac{\sigma^p D}{T^{1-\frac{1}{p}}}, \forall \mathbf{x} \in \mathcal{X}.$$

*Proof.* By convexity, F(x¯<sup>T</sup> ) − F(x) ≤ P<sup>T</sup> <sup>t</sup>=1 F (xt)−F (x) <sup>T</sup> ≤ R A <sup>T</sup> (x) T <sup>214</sup> is valid for any OCO algorithm <sup>215</sup> A. We conclude from invoking Theorems [1,](#page-4-1) [2](#page-4-2) and [3.](#page-5-1)

 To the best of our knowledge, Corollary [1](#page-6-0) gives the first and optimal convergence rate for these three algorithms in stochastic optimization with heavy tails. Especially, it implies that once the domain is bounded, the widely implemented SGD algorithm provably converges under heavy-tailed noise without any algorithmic change considered in many prior works, e.g., gradient clipping [\[18,](#page-10-3) [26\]](#page-10-4).

<sup>220</sup> We are only aware of two works [\[19,](#page-10-9) [41\]](#page-11-6) based on Stochastic Mirror Descent (SMD) [\[24\]](#page-10-7) that gave <sup>221</sup> convergence results without clipping. However, they share a common drawback, i.e., their bounds are both in the form of (G + σ)D/T<sup>1</sup>−1/<sup>p</sup> , which cannot recover the optimal rate GD/√ <sup>222</sup> T when σ = 0.

<sup>223</sup> Lastly, we highlight that for A = AdaGrad, Corollary [1](#page-6-0) is not only optimal but also adaptive to the <sup>224</sup> tail index p. As far as we know, no result has achieved this property before. This once again evidences <sup>225</sup> the benefit of adaptive gradient methods.

 Convergence of the last iterate. Next, we consider the more challenging last-iterate convergence, which has a long history in stochastic optimization and fruitful results in the case of p = 2 (see, e.g., [\[28,](#page-10-10) [36,](#page-11-8) [49\]](#page-12-5)). However, less is known about heavy-tailed problems. So far, only two works [\[19,](#page-10-9) [29\]](#page-11-9) have established the last-iterate convergence. The former is based on SMD, and the latter employs gradient clipping in SGD. Unfortunately, their rates are both in the suboptimal order (G + σ)D/T<sup>1</sup>−1/<sup>p</sup> <sup>231</sup> .

<sup>232</sup> We will provide an optimal last-iterate rate based on the following lemma, which reduces the <sup>233</sup> last-iterate convergence to an online learning problem.

Lemma 1 (Theorem 1 of [\[7\]](#page-9-8)). *Suppose* x1, . . . , x<sup>T</sup> *and* y<sup>1</sup> <sup>234</sup> , . . . , y<sup>T</sup> *are two sequences of vectors* <sup>235</sup> *satisfying* x<sup>t</sup> ∈ X , x<sup>1</sup> = y<sup>1</sup> *and*

$$\mathbf{y}_{t+1} = \mathbf{y}_t + \frac{T-t}{T} (\mathbf{x}_{t+1} - \mathbf{x}_t). \quad (8)$$

*Given a convex function* F(x)*, let* ℓt(x) = ⟨∇F(y<sup>t</sup> <sup>236</sup> ), x⟩*. Then for any online learner* A*, we have*

$$F(\mathbf{y}_T) - F(\mathbf{x}) \leq \frac{\mathbf{R}_T^A(\mathbf{x})}{T}, \forall \mathbf{x} \in \mathcal{X}.$$

We emphasize that the stochastic gradient g<sup>t</sup> received by A is an estimate of ∇F(y<sup>t</sup> <sup>237</sup> ) instead of <sup>238</sup> ∇F(xt). This flexibility is due to the generality of the OCO framework. Moreover, for OGD, suppose there is no projection step, then [\(8\)](#page-6-1) is equivalent to yt+1 = y<sup>t</sup> − T −t T ηtg<sup>t</sup> <sup>239</sup> , which can be viewed as SGD with a stepsize <sup>T</sup> <sup>−</sup><sup>t</sup> T <sup>240</sup> ηt. For proof of Lemma [1,](#page-6-2) we refer the interested reader to [\[7\]](#page-9-8).

Corollary 2. *Under Assumption [1](#page-2-0) for* ℓt(x) = ⟨∇F(y<sup>t</sup> ), x⟩*, where* y<sup>t</sup> <sup>241</sup> *satisfies [\(8\)](#page-6-1), for any* A ∈ <sup>242</sup> {OGD, DA, AdaGrad}*, we have*

$$\mathbb{E}[F(\mathbf{y}_T) - F(\mathbf{x})] \leq \frac{\mathbb{E}[R_T^A(\mathbf{x})]}{T} \lesssim \frac{GD}{\sqrt{T}} + \frac{\sigma^p D}{T^{1-\frac{1}{p}}}, \forall \mathbf{x} \in \mathcal{X}.$$

<sup>243</sup> *Proof.* Combine Lemma [1](#page-6-2) and Theorems [1,](#page-4-1) [2](#page-4-2) and [3](#page-5-1) to conclude.

<sup>244</sup> As far as we know, Corollary [2](#page-6-3) is the first optimal last-iterate convergence rate for stochastic convex <sup>245</sup> optimization with heavy tails, closing the gap in existing works.

One may notice that y<sup>t</sup> <sup>246</sup> itself is not the decision made by the online learner and naturally may ask <sup>247</sup> whether x<sup>t</sup> ensures the last-iterate convergence if we simply pick ℓ<sup>t</sup> = F. The answer turns out to

<sup>248</sup> be positive at least for OGD (which is equivalent to SGD now). However, to prove this result, we <sup>249</sup> rely on a technique specialized to stochastic optimization recently developed by [\[19,](#page-10-9) [44\]](#page-12-6). To not <sup>250</sup> diverge from the topic of OCO, we defer the last-iterate convergence of OGD to Appendix [D,](#page-17-0) in <sup>251</sup> which Theorem [8](#page-17-1) gives a general result for any stepsize η<sup>t</sup> and Corollary [4](#page-19-0) shows the last-iterate rate under the same stepsize η<sup>t</sup> = D G √ t ∧ D σt <sup>252</sup> <sup>1</sup>/<sup>p</sup> as in Theorem [1](#page-4-1) before.

## <sup>253</sup> 4.2 Nonsmooth Nonconvex Optimization

<sup>254</sup> This section contains another application, nonsmooth nonconvex optimization with heavy tails. Due <sup>255</sup> to limited space, we will provide only the necessary background. For more details, we refer the reader <sup>256</sup> to [\[6,](#page-9-9) [13,](#page-9-10) [15,](#page-10-11) [16,](#page-10-12) [38,](#page-11-10) [39\]](#page-11-11) for recent progress. We start with a new set of conditions.

<sup>257</sup> Assumption 2. *We consider the following series of assumptions:*

<sup>258</sup> • *The objective* F *is lower bounded by* F<sup>⋆</sup> ≜ infx∈R<sup>d</sup> F(x) ∈ <sup>R</sup>*.*

• <sup>F</sup> *is differentiable and well-behaved, i.e.,* <sup>F</sup>(x) <sup>−</sup> <sup>F</sup>(y) = R <sup>1</sup>

0

<sup>259</sup> ⟨∇F(y + t(x − y)), x − y⟩ dt*.*

• F *is* G*-Lipschitz on* R

d

*, i.e.,* ∥∇F(x)∥ ≤ G, ∀x ∈ <sup>R</sup>

d

<sup>260</sup> *.*

• *Given* z<sup>t</sup> ∈ <sup>R</sup> <sup>d</sup> *at the* t*-th iteration, one can query* g<sup>t</sup> ∈ <sup>R</sup> d *satisfying* <sup>E</sup> [g<sup>t</sup> <sup>261</sup> | Ft−1] = ∇F(zt) *and* E -∥ϵt∥ p ≤ σ p <sup>262</sup> *for some* p ∈ (1, 2] *and* σ ≥ 0*, where* F<sup>t</sup> *denotes the natural filtration and* <sup>263</sup> ϵ<sup>t</sup> <sup>≜</sup> g<sup>t</sup> − ∇F(zt) *is the stochastic noise.*

<sup>264</sup> *Remark* 6*.* The second point is a mild regularity condition introduced by [\[5\]](#page-9-11) and becomes standard <sup>265</sup> in the literature [\[2,](#page-9-12) [17,](#page-10-13) [48\]](#page-12-7). See Definition 1 and Proposition 2 of [\[5\]](#page-9-11) for more details. In the fourth <sup>266</sup> point, we use the same notation z<sup>t</sup> as in the algorithm being studied later. In fact, it can be arbitrary.

<sup>267</sup> In nonsmooth nonconvex optimization, we aim to find a (δ, ϵ)-stationary point [\[46\]](#page-12-8) (see the formal Definition [2](#page-20-0) in Appendix [E\)](#page-20-1). This goal can be reduced to finding a point x ∈ R d <sup>268</sup> such that ∥∇F(x)∥<sup>δ</sup> ≤ ϵ, where ∥∇F(x)∥<sup>δ</sup> <sup>269</sup> is a quantity introduced by [\[5\]](#page-9-11) as follows.

Definition 1 (Definition 5 of [\[5\]](#page-9-11)). Given a point x ∈ R d <sup>270</sup> , a number δ > 0 and an almost-everywhere differentiable function <sup>F</sup>, define ∥∇F(x)∥<sup>δ</sup> <sup>≜</sup> infS⊂B(x,δ), |S| P <sup>y</sup>∈<sup>S</sup> y=x |S| P <sup>y</sup>∈<sup>S</sup> ∇F(y) <sup>271</sup> .

The only existing sample complexity under Assumption [2](#page-7-0) is (G+σ) p <sup>p</sup>−<sup>1</sup> δ −1 ϵ − 2p−1 <sup>p</sup>−<sup>1</sup> <sup>272</sup> in high probability <sup>273</sup> [\[17\]](#page-10-13), where we only report the dominant term and hide the dependency on the failure probability.

However, on the theoretical side, their result cannot recover the optimal bound G<sup>2</sup> δ −1 ϵ −3 <sup>274</sup> [\[5\]](#page-9-11) in the <sup>275</sup> deterministic case. On the practical side, their method also employs the gradient clipping step, which <sup>276</sup> introduces a new clipping parameter to tune. In fact, as stated in their Section 5, they observed in <sup>277</sup> experiments that their algorithm without the clipping operation (exactly the algorithm we study next) <sup>278</sup> still works under heavy tails. In addition, in their Section 6, they also explicitly ask whether the <sup>279</sup> requirement to know G and A can be removed.

<sup>280</sup> As will be seen later, we can address these points with the new regret bounds presented before.

## <sup>281</sup> 4.2.1 Online-to-Nonconvex Conversion under Heavy Tails

Algorithm 4 Online-to-Nonconvex Conversion (O2NC) [\[5\]](#page-9-11)

Input: initial point y<sup>0</sup> ∈ <sup>R</sup> d , K ∈ N, T ∈ N, online learning algorithm A.

for n = 1 to KT do Receive x<sup>n</sup> from A y<sup>n</sup> = yn−<sup>1</sup> + x<sup>n</sup>

z<sup>n</sup> = yn−<sup>1</sup> + snx<sup>n</sup> where s<sup>n</sup> ∼ Uniform [0, 1] i.i.d.

Query a stochastic gradient g<sup>n</sup> at z<sup>n</sup> Send g<sup>n</sup> to A

end for

<sup>282</sup> *Remark* 7*.* Note that O2NC is a randomized algorithm. Therefore, the definition of the natural filtration is adjusted to F<sup>n</sup> <sup>≜</sup> σ(s1, g<sup>1</sup> <sup>283</sup> , . . . , sn, gn, sn+1) accordingly.

 We provide the Online-to-Nonconvex Conversion (O2NC) framework in Algorithm [4,](#page-7-1) which serves as a meta algorithm. Roughly speaking, Algorithm [4](#page-7-1) reduces a nonconvex optimization problem to an OCO (in fact, OLO) problem, for which the K-shifting regret (see [\(9\)](#page-8-0)) of the online learner A crucially affects the final convergence rate. However, the existing Theorem 8 of [\[5\]](#page-9-11), a general convergence result for the above reduction, cannot directly apply to heavy-tailed noise, since its proof relies on the finite variance condition on g<sup>n</sup> (see Appendix [E](#page-20-1) for more details).

Theorem 4. *Under Assumption [2](#page-7-0) and let* v<sup>k</sup> ≜ −D PkT <sup>n</sup>=(k−1)<sup>T</sup> +1 ∇F (zn) PkT <sup>n</sup>=(k−1)<sup>T</sup> +1 <sup>∇</sup><sup>F</sup> (zn)∥ <sup>290</sup> , ∀k ∈ [K] *for arbitrary* <sup>291</sup> D > 0*, then for any online learning algorithm* A *in* O2NC *(Algorithm [4\)](#page-7-1), we have*

$$\mathbb{E} \left[ \sum_{k=1}^K \frac{1}{K} \left\| \frac{1}{T} \sum_{n=(k-1)T+1}^{kT} \nabla F(\mathbf{z}_n) \right\| \right] \lesssim \frac{F(\mathbf{y}_0) - F_*}{DKT} + \frac{\mathbb{E}[\mathbf{R}_T^A(\mathbf{v}_1, \dots, \mathbf{v}_K)]}{DKT} + \frac{\sigma}{T^{1-\frac{1}{p}}}.$$

R A T <sup>292</sup> (v1, · · · , vK) in Theorem [4](#page-8-1) is called K*-shifting regret* [\[5\]](#page-9-11), defined as follows:

$$\mathbf{R}_T^A(\mathbf{v}_1, \dots, \mathbf{v}_K) \triangleq \sum_{k=1}^K \sum_{n=(k-1)T+1}^T \ell_n(\mathbf{x}_n) - \ell_n(\mathbf{v}_k) \quad \text{where} \quad \ell_n(\mathbf{x}) \triangleq \langle \mathbf{g}_n, \mathbf{x} \rangle. \quad (9)$$

<sup>293</sup> Theorem [4](#page-8-1) here provides a new and the first theoretical guarantee for O2NC under heavy tails. <sup>294</sup> Especially, it recovers Theorem 8 of [\[5\]](#page-9-11) when p = 2. A remarkable point is that the O2NC algorithm <sup>295</sup> itself does not need any information about p. The proof of Theorem [4](#page-8-1) can be found in Appendix [E.](#page-20-1)

#### <sup>296</sup> 4.2.2 Convergence Rates

Theorem [4](#page-8-1) enables us to apply the results presented in Section [3.](#page-2-1) Concretely, for X = B d <sup>297</sup> (D) and <sup>298</sup> any A ∈ {OGD, DA, AdaGrad}, if we reset the stepsize in A after every T iterations, there will be E -R A T (v1, · · · , vK) <sup>≲</sup> GDK√ T + σDKT<sup>1</sup>/<sup>p</sup> <sup>299</sup> by our new regret bounds, since v<sup>k</sup> ∈ X . With a <sup>300</sup> carefully picked D, we obtain the following Theorem [5.](#page-8-2) Its proof is deferred to Appendix [E.](#page-20-1)

Theorem 5. *Under Assumption [2](#page-7-0) and let* ∆ <sup>≜</sup> F(y<sup>0</sup> )−F<sup>⋆</sup> *and* z¯<sup>k</sup> ≜ 1 T PkT <sup>n</sup>=(k−1)<sup>T</sup> +1 <sup>301</sup> zn, ∀k ∈ [K]*, setting any* A ∈ {OGD, DA, AdaGrad} *in* O2NC *(Algorithm [4\)](#page-7-1) with a domain* X = B d <sup>302</sup> (D) *for* <sup>303</sup> D = δ/T *and resetting the stepsize in* A *after every* T *iterations, we have*

$$\mathbb{E} \left[ \frac{1}{K} \sum_{k=1}^K \|\nabla F(\bar{z}_k)\|_{\delta} \right] \lesssim \frac{\Delta}{\delta K} + \frac{G}{\sqrt{T}} + \frac{\sigma}{T^{1-\frac{1}{p}}}.$$

<sup>304</sup> Notably, this is the first time confirming that gradient clipping is indeed unnecessary for the O2NC <sup>305</sup> framework, matching the experimental observation of [\[17\]](#page-10-13).

<sup>306</sup> Corollary 3. *Under the same setting of Theorem [5,](#page-8-2) suppose we have* N ≥ 2 *stochastic gradient budgets, taking* <sup>K</sup> <sup>=</sup> ⌊N/T⌋ *and* <sup>T</sup> <sup>=</sup> ⌈N/2⌉ ∧ l(δGN/∆) <sup>2</sup> 3 m ∨ l (δσN/∆) p 2p−1 m <sup>307</sup> *, we have*

$$\mathbb{E} \left[ \frac{1}{K} \sum_{k=1}^K \|\nabla F(\bar{\mathbf{z}}_k)\|_{\delta} \right] \lesssim \frac{G}{\sqrt{N}} + \frac{\sigma}{N^{1-\frac{1}{p}}} + \frac{\Delta}{\delta N} + \frac{G^{\frac{2}{3}} \Delta^{\frac{1}{3}}}{(\delta N)^{\frac{1}{3}}} + \frac{\sigma^{\frac{p}{2p-1}} \Delta^{\frac{p-1}{2p-1}}}{(\delta N)^{\frac{p-1}{2p-1}}}.$$

<sup>308</sup> Corollary [3](#page-8-3) is obtained by optimizing K and T in Theorem [5.](#page-8-2) It implies a sample complexity of G<sup>2</sup> δ −1 ϵ <sup>−</sup><sup>3</sup> + σ <sup>p</sup>−<sup>1</sup> δ −1 ϵ − 2p−1 <sup>p</sup>−<sup>1</sup> <sup>309</sup> for finding a (δ, ϵ)-stationary point, improved over the previous bound (G + σ) p <sup>p</sup>−<sup>1</sup> δ −1 ϵ − 2p−1 <sup>p</sup>−<sup>1</sup> <sup>310</sup> [\[17\]](#page-10-13). Furthermore, leveraging the adaptive feature of AdaGrad, Corollary [5](#page-23-0) <sup>311</sup> in Appendix [E](#page-20-1) shows how to set K and T without G, σ, and p, resulting in the first provably rate for <sup>312</sup> O2NC when no problem information is known in advance, which solves the problem asked by [\[17\]](#page-10-13).

## <sup>313</sup> 5 Conclusion and Limitation

 This paper shows that three classical OCO algorithms, OGD, DA, and AdaGrad, can achieve the optimal in-expectation regret under heavy tails without any algorithmic modification if the feasible set is bounded, and provides some applications in stochastic optimization. The main limitation of our work is that all the proof crucially relies on the bounded domain assumption, which may not always be suitable in practice. Finding a weaker sufficient condition, under which the classical OCO algorithms work with heavy tails provably, is a direction worth studying in the future.

## References


[1] Naman Agarwal, Brian Bullins, Elad Hazan, Sham Kakade, and Karan Singh. Online control with adversarial disturbances. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, *Proceedings of the 36th International Conference on Machine Learning*, volume 97 of *Pro- ceedings of Machine Learning Research*, pages 111–119. PMLR, 09–15 Jun 2019. URL <https://proceedings.mlr.press/v97/agarwal19c.html>. [2] Kwangjun Ahn and Ashok Cutkosky. Adam with model exponential moving aver- age is effective for nonconvex optimization. In A. Globerson, L. Mackey, D. Bel- grave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, *Advances in Neural Information Processing Systems*, volume 37, pages 94909–94933. Curran Associates, Inc., 2024. URL [https://proceedings.neurips.cc/paper\\_files/paper/2024/file/](https://proceedings.neurips.cc/paper_files/paper/2024/file/ac8ec9b4d94c03f0af8c4fe3d5fad4fd-Paper-Conference.pdf) [ac8ec9b4d94c03f0af8c4fe3d5fad4fd-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/ac8ec9b4d94c03f0af8c4fe3d5fad4fd-Paper-Conference.pdf). [3] N. Cesa-Bianchi, A. Conconi, and C. Gentile. On the generalization ability of on-line learning algorithms. *IEEE Transactions on Information Theory*, 50(9):2050–2057, 2004. doi: 10.1109/ TIT.2004.833339. [4] Nicolo Cesa-Bianchi and Gabor Lugosi. *Prediction, Learning, and Games*. Cambridge University Press, 2006. [5] Ashok Cutkosky, Harsh Mehta, and Francesco Orabona. Optimal stochastic non-smooth non- convex optimization through online-to-non-convex conversion. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Pro- ceedings of Machine Learning Research*, pages 6643–6670. PMLR, 23–29 Jul 2023. URL <https://proceedings.mlr.press/v202/cutkosky23a.html>. [6] Damek Davis, Dmitriy Drusvyatskiy, Yin Tat Lee, Swati Padmanabhan, and Guanghao Ye. A gradient sampling method with complexity guarantees for lipschitz functions in high and low dimensions. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, *Advances in Neural Information Processing Systems*, volume 35, pages 6692–6703. Curran Associates, Inc., 2022. URL [https://proceedings.neurips.cc/paper\\_files/paper/](https://proceedings.neurips.cc/paper_files/paper/2022/file/2c8d9636f74d0207ff4f65956010f450-Paper-Conference.pdf) [2022/file/2c8d9636f74d0207ff4f65956010f450-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2022/file/2c8d9636f74d0207ff4f65956010f450-Paper-Conference.pdf). [7] Aaron Defazio, Ashok Cutkosky, Harsh Mehta, and Konstantin Mishchenko. Optimal linear decay learning rate schedules and further refinements. *arXiv preprint arXiv:2310.07831*, 2023. [8] John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. *Journal of Machine Learning Research*, 12(61):2121–2159, 2011. URL <http://jmlr.org/papers/v12/duchi11a.html>. [9] John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. *Journal of machine learning research*, 12(7), 2011. [10] Elad Hazan. Introduction to online convex optimization. *Foundations and Trends® in Optimization*, 2(3-4):157–325, 2016. ISSN 2167-3888. doi: 10.1561/2400000013. URL <http://dx.doi.org/10.1561/2400000013>. [\[](https://arxiv.org/abs/2211.09619)11] Elad Hazan and Karan Singh. Introduction to online control, 2025. URL [https://arxiv.](https://arxiv.org/abs/2211.09619) [org/abs/2211.09619](https://arxiv.org/abs/2211.09619). [12] Liam Hodgkinson and Michael Mahoney. Multiplicative noise and heavy tails in stochastic optimization. In *International Conference on Machine Learning*, pages 4262–4274. PMLR, 2021. [13] Michael Jordan, Guy Kornowski, Tianyi Lin, Ohad Shamir, and Manolis Zampetakis. De- terministic nonsmooth nonconvex optimization. In Gergely Neu and Lorenzo Rosasco, editors, *Proceedings of Thirty Sixth Conference on Learning Theory*, volume 195 of *Pro- ceedings of Machine Learning Research*, pages 4570–4597. PMLR, 12–15 Jul 2023. URL <https://proceedings.mlr.press/v195/jordan23a.html>.

[14] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*, 2014. [15] Guy Kornowski and Ohad Shamir. Oracle complexity in nonsmooth nonconvex optimiza- tion. *Journal of Machine Learning Research*, 23(314):1–44, 2022. URL [http://jmlr.org/](http://jmlr.org/papers/v23/21-1507.html) [papers/v23/21-1507.html](http://jmlr.org/papers/v23/21-1507.html). [16] Guy Kornowski and Ohad Shamir. On the complexity of finding small subgradients in non- smooth optimization. In *OPT 2022: Optimization for Machine Learning (NeurIPS 2022 Workshop)*, 2022. URL <https://openreview.net/forum?id=SaRQ4oTqWbP>. [17] Langqi Liu, Yibo Wang, and Lijun Zhang. High-probability bound for non-smooth non-convex stochastic optimization with heavy tails. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, *Proceedings of the 41st International Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning Research*, pages 32122–32138. PMLR, 21–27 Jul 2024. URL [https:](https://proceedings.mlr.press/v235/liu24bo.html) [//proceedings.mlr.press/v235/liu24bo.html](https://proceedings.mlr.press/v235/liu24bo.html). [18] Zijian Liu and Zhengyuan Zhou. Stochastic nonsmooth convex optimization with heavy-tailed noises: High-probability bound, in-expectation rate and initial distance adaptation. *arXiv preprint arXiv:2303.12277*, 2023. [19] Zijian Liu and Zhengyuan Zhou. Revisiting the last-iterate convergence of stochastic gradient methods. In *The Twelfth International Conference on Learning Representations*, 2024. URL <https://openreview.net/forum?id=xxaEhwC1I4>. [20] Zijian Liu and Zhengyuan Zhou. Nonconvex stochastic optimization under heavy-tailed noises: Optimal convergence without gradient clipping. In *The Thirteenth International Conference on Learning Representations*, 2025. URL <https://openreview.net/forum?id=NKotdPUc3L>. [21] Brendan McMahan. Follow-the-regularized-leader and mirror descent: Equivalence theo- rems and l1 regularization. In Geoffrey Gordon, David Dunson, and Miroslav Dudík, edi- tors, *Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics*, volume 15 of *Proceedings of Machine Learning Research*, pages 525–533, Fort Lauderdale, FL, USA, 11–13 Apr 2011. PMLR. URL [https://proceedings.mlr.press/](https://proceedings.mlr.press/v15/mcmahan11b.html) [v15/mcmahan11b.html](https://proceedings.mlr.press/v15/mcmahan11b.html). [22] H Brendan McMahan and Matthew Streeter. Adaptive bound optimization for online convex optimization. *arXiv preprint arXiv:1002.4908*, 2010. [23] H. Brendan McMahan and Matthew J. Streeter. Adaptive bound optimization for online convex optimization. In *Conference on Learning Theory (COLT)*, pages 244–256. Omnipress, 2010. [24] Arkadi Nemirovski and David Yudin. Problem complexity and method efficiency in optimization. *Wiley-Interscience*, 1983. [25] Yurii Nesterov. Primal-dual subgradient methods for convex problems. *Mathematical program- ming*, 120(1):221–259, 2009. [26] Ta Duy Nguyen, Thien H Nguyen, Alina Ene, and Huy Nguyen. Improved convergence in high probability of clipped gradient methods with heavy tailed noise. In A. Oh, T. Nau- mann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, *Advances in Neu- ral Information Processing Systems*, volume 36, pages 24191–24222. Curran Associates, Inc., 2023. URL [https://proceedings.neurips.cc/paper\\_files/paper/2023/file/](https://proceedings.neurips.cc/paper_files/paper/2023/file/4c454d34f3a4c8d6b4ca85a918e5d7ba-Paper-Conference.pdf) [4c454d34f3a4c8d6b4ca85a918e5d7ba-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/4c454d34f3a4c8d6b4ca85a918e5d7ba-Paper-Conference.pdf). [27] Francesco Orabona. A modern introduction to online learning. *arXiv preprint arXiv:1912.13213*, 2019. [28] Francesco Orabona. Last iterate of sgd converges (even in unbounded domains). 2020. URL [https://parameterfree.com/2020/08/07/](https://parameterfree.com/2020/08/07/last-iterate-of-sgd-converges-even-in-unbounded-domains/) [last-iterate-of-sgd-converges-even-in-unbounded-domains/](https://parameterfree.com/2020/08/07/last-iterate-of-sgd-converges-even-in-unbounded-domains/).

[29] Daniela Angela Parletta, Andrea Paudice, and Saverio Salzo. An improved analysis of the clipped stochastic subgradient method under heavy-tailed noise, 2025. URL [https://arxiv.](https://arxiv.org/abs/2410.00573) [org/abs/2410.00573](https://arxiv.org/abs/2410.00573). [30] Maxim Raginsky and Alexander Rakhlin. Information complexity of black-box convex op- timization: A new look via feedback information theory. In *2009 47th Annual Allerton Conference on Communication, Control, and Computing (Allerton)*, pages 803–510, 2009. doi: 10.1109/ALLERTON.2009.5394945. [31] Alexander Rakhlin, Ohad Shamir, and Karthik Sridharan. Making gradient descent optimal for strongly convex stochastic optimization. *arXiv preprint arXiv:1109.5647*, 2011. [32] Herbert Robbins and Sutton Monro. A Stochastic Approximation Method. *The Annals of Mathematical Statistics*, 22(3):400 – 407, 1951. doi: 10.1214/aoms/1177729586. URL [https:](https://doi.org/10.1214/aoms/1177729586) [//doi.org/10.1214/aoms/1177729586](https://doi.org/10.1214/aoms/1177729586). [33] R Tyrrell Rockafellar. *Convex analysis*, volume 28. Princeton university press, 1997. [34] Abdurakhmon Sadiev, Marina Danilova, Eduard Gorbunov, Samuel Horváth, Gauthier Gidel, Pavel Dvurechensky, Alexander Gasnikov, and Peter Richtárik. High-probability bounds for stochastic optimization and variational inequalities: the case of unbounded variance. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, *Proceedings of the 40th International Conference on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*, pages 29563–29648. PMLR, 23–29 Jul 2023. URL <https://proceedings.mlr.press/v202/sadiev23a.html>. [35] Shai Shalev-Shwartz. Online learning and online convex optimization. *Foundations and Trends® in Machine Learning*, 4(2):107–194, 2012. ISSN 1935-8237. doi: 10.1561/2200000018. URL <http://dx.doi.org/10.1561/2200000018>. [36] Ohad Shamir and Tong Zhang. Stochastic gradient descent for non-smooth optimization: Convergence results and optimal averaging schemes. In Sanjoy Dasgupta and David McAllester, editors, *Proceedings of the 30th International Conference on Machine Learning*, volume 28 of *Proceedings of Machine Learning Research*, pages 71–79, Atlanta, Georgia, USA, 17–19 Jun 2013. PMLR. URL <https://proceedings.mlr.press/v28/shamir13.html>. [37] Umut Simsekli, Levent Sagun, and Mert Gurbuzbalaban. A tail-index analysis of stochastic gradient noise in deep neural networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, *Proceedings of the 36th International Conference on Machine Learning*, volume 97 of *Proceedings of Machine Learning Research*, pages 5827–5837. PMLR, 09–15 Jun 2019. URL <https://proceedings.mlr.press/v97/simsekli19a.html>. [38] Lai Tian and Anthony Man-Cho So. No dimension-free deterministic algorithm computes approximate stationarities of lipschitzians. *Mathematical Programming*, 208(1):51–74, 2024. [39] Lai Tian, Kaiwen Zhou, and Anthony Man-Cho So. On the finite-time complexity and prac- tical computation of approximate stationarity concepts of Lipschitz functions. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, *Proceedings of the 39th International Conference on Machine Learning*, volume 162 of *Pro- ceedings of Machine Learning Research*, pages 21360–21379. PMLR, 17–23 Jul 2022. URL <https://proceedings.mlr.press/v162/tian22a.html>. [40] Tijmen Tieleman, Geoffrey Hinton, et al. Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. *COURSERA: Neural networks for machine learning*, 4(2): 26–31, 2012. [41] Nuri Mert Vural, Lu Yu, Krishna Balasubramanian, Stanislav Volgushev, and Murat A Erdogdu. Mirror descent strikes again: Optimal stochastic convex optimization under infinite noise variance. In Po-Ling Loh and Maxim Raginsky, editors, *Proceedings of Thirty Fifth Conference on Learning Theory*, volume 178 of *Proceedings of Machine Learning Research*, pages 65–102. PMLR, 02–05 Jul 2022. URL <https://proceedings.mlr.press/v178/vural22a.html>.

[42] Rachel Ward, Xiaoxia Wu, and Leon Bottou. AdaGrad stepsizes: Sharp convergence over nonconvex landscapes. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, *Proceedings of the 36th International Conference on Machine Learning*, volume 97 of *Proceedings of Machine Learning Research*, pages 6677–6686. PMLR, 09–15 Jun 2019. URL [https://](https://proceedings.mlr.press/v97/ward19a.html) [proceedings.mlr.press/v97/ward19a.html](https://proceedings.mlr.press/v97/ward19a.html). [43] Lin Xiao. Dual averaging method for regularized stochastic learning and online opti- mization. In Y. Bengio, D. Schuurmans, J. Lafferty, C. Williams, and A. Culotta, edi- tors, *Advances in Neural Information Processing Systems*, volume 22. Curran Associates, Inc., 2009. URL [https://proceedings.neurips.cc/paper\\_files/paper/2009/file/](https://proceedings.neurips.cc/paper_files/paper/2009/file/7cce53cf90577442771720a370c3c723-Paper.pdf) [7cce53cf90577442771720a370c3c723-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2009/file/7cce53cf90577442771720a370c3c723-Paper.pdf). [44] Moslem Zamani and François Glineur. Exact convergence rate of the last iterate in subgradient methods. *arXiv preprint arXiv:2307.11134*, 2023. [45] Jingzhao Zhang, Sai Praneeth Karimireddy, Andreas Veit, Seungyeon Kim, Sashank Reddi, Sanjiv Kumar, and Suvrit Sra. Why are adaptive methods good for attention models? In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, *Advances in Neu- ral Information Processing Systems*, volume 33, pages 15383–15393. Curran Associates, Inc., 2020. URL [https://proceedings.neurips.cc/paper\\_files/paper/2020/file/](https://proceedings.neurips.cc/paper_files/paper/2020/file/b05b57f6add810d3b7490866d74c0053-Paper.pdf) [b05b57f6add810d3b7490866d74c0053-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2020/file/b05b57f6add810d3b7490866d74c0053-Paper.pdf). [46] Jingzhao Zhang, Hongzhou Lin, Stefanie Jegelka, Suvrit Sra, and Ali Jadbabaie. Complexity of finding stationary points of nonconvex nonsmooth functions. In Hal Daumé III and Aarti Singh, editors, *Proceedings of the 37th International Conference on Machine Learning*, volume 119 of *Proceedings of Machine Learning Research*, pages 11173–11182. PMLR, 13–18 Jul 2020. URL <https://proceedings.mlr.press/v119/zhang20p.html>. [47] Jiujia Zhang and Ashok Cutkosky. Parameter-free regret in high probability with heavy tails. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, *Advances in Neural Information Processing Systems*, volume 35, pages 8000–8012. Curran Associates, Inc., 2022. URL [https://proceedings.neurips.cc/paper\\_files/paper/2022/file/](https://proceedings.neurips.cc/paper_files/paper/2022/file/349956dee974cfdcbbb2d06afad5dd4a-Paper-Conference.pdf) [349956dee974cfdcbbb2d06afad5dd4a-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2022/file/349956dee974cfdcbbb2d06afad5dd4a-Paper-Conference.pdf). [48] Qinzi Zhang and Ashok Cutkosky. Random scaling and momentum for non-smooth non-convex optimization. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, *Proceedings of the 41st International Conference on Machine Learning*, volume 235 of *Proceedings of Machine Learning Research*, pages 58780–58799. PMLR, 21–27 Jul 2024. URL [https://proceedings.mlr.press/](https://proceedings.mlr.press/v235/zhang24k.html) [v235/zhang24k.html](https://proceedings.mlr.press/v235/zhang24k.html). [49] Tong Zhang. Solving large scale linear prediction problems using stochastic gradient descent algorithms. In *Proceedings of the twenty-first international conference on Machine learning*, page 116, 2004. [50] Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In *Proceedings of the 20th international conference on machine learning (icml-03)*, pages 928–936, 2003.
## <sup>506</sup> A Missing Proofs for Online Gradient Descent

<sup>507</sup> This section provides missing proofs for regret bounds of OGD. Before showing the formal proof, <sup>508</sup> we recall the following core inequality that holds for any x ∈ X given in [\(6\)](#page-4-4):

$$\langle g_t, \mathbf{x}_t - \mathbf{x} \rangle \leq \frac{\|\mathbf{x}_t - \mathbf{x}\|^2 - \|\mathbf{x}_{t+1} - \mathbf{x}\|^2}{2\eta_t} + \eta_t G^2 + C(\mathbf{p})\eta_t^{p-1} \|\epsilon_t\|^p D^{2-p}. \quad (10)$$

<sup>509</sup> The key to establishing the above result is showing

$$\langle g_t, x_t - x_{t+1} \rangle - \frac{\|x_t - x_{t+1}\|^2}{2\eta_t} \leq \eta_t G^2 + C(p)\eta_t^{p-1} \|\epsilon_t\|^p D^{2-p}, \quad (11)$$

<sup>510</sup> the proof of which is by combining [\(3\)](#page-3-4), [\(4\)](#page-3-3), and [\(5\)](#page-4-0) established in the main text.

## <sup>511</sup> A.1 Proof of Theorem [1](#page-4-1)

*Proof.* For any x ∈ X , sum up [\(10\)](#page-13-1) from t = 1 to T and drop the term − ∥x<sup>T</sup> +1−x∥ 2η<sup>T</sup> <sup>512</sup> to obtain

$$\begin{aligned} & \sum_{t=1}^T \langle \mathbf{g}_t, \mathbf{x}_t - \mathbf{x} \rangle \\ & \leq \frac{\|\mathbf{x}_1 - \mathbf{x}\|^2}{2\eta_1} + \sum_{t=1}^{T-1} \left( \frac{1}{\eta_{t+1}} - \frac{1}{\eta_t} \right) \frac{\|\mathbf{x}_{t+1} - \mathbf{x}\|^2}{2} + \sum_{t=1}^T \eta_t G^2 + C(p)\eta_t^{p-1} \|\epsilon_t\|^p D^{2-p} \quad (12) \end{aligned}$$

$$\leq \frac{D^2}{\eta_T} + \sum_{t=1}^T \eta_t G^2 + C(\mathfrak{p}) \eta_t^{\mathfrak{p}-1} \|\boldsymbol{\epsilon}_t\|^{\mathfrak{p}} D^{2-\mathfrak{p}}, \quad (13)$$

<sup>513</sup> where the last step is due to ∥x<sup>t</sup> − x∥ ≤ D, ∀t ∈ [T] and ηt+1 ≤ ηt, ∀t ∈ [T − 1].

<sup>514</sup> Taking expectations on both sides of [\(13\)](#page-13-2) yields that

$$\mathbb{E} [\mathbf{R}_T^{\text{OGD}}(\mathbf{x})] \leq \frac{D^2}{\eta_T} + \sum_{t=1}^T \eta_t G^2 + \mathbf{C}(\mathbf{p}) \eta_t^{p-1} \sigma^p D^{2-p}, \quad (14)$$

where for the L.H.S., we use <sup>E</sup> [⟨g<sup>t</sup> , x<sup>t</sup> − x⟩] = <sup>E</sup> [<sup>E</sup> [⟨g<sup>t</sup> <sup>515</sup> , x<sup>t</sup> − x⟩ | Ft−1]] and

$$\mathbb{E} [\langle \mathbf{g}_t, \mathbf{x}_t - \mathbf{x} \rangle \mid \mathcal{F}_{t-1}] = \langle \mathbb{E} [\mathbf{g}_t \mid \mathcal{F}_{t-1}], \mathbf{x}_t - \mathbf{x} \rangle = \langle \nabla \ell_t(\mathbf{x}_t), \mathbf{x}_t - \mathbf{x} \rangle \geq \ell_t(\mathbf{x}_t) - \ell_t(\mathbf{x}), \quad (15)$$

for the R.H.S., we use E -∥ϵt∥ p ≤ σ p <sup>516</sup> .

Finally, we plug η<sup>t</sup> = D G √ t ∧ D σt1/<sup>p</sup> , ∀t ∈ [T] into [\(14\)](#page-13-3), then use P<sup>T</sup> <sup>t</sup>=1 √ 1 t ≲ √ T and P<sup>T</sup> t=1 1 t <sup>517</sup> <sup>1</sup>−1/<sup>p</sup> ≲ T 1/p <sup>518</sup> to conclude <sup>≲</sup> GD√

E -R OGD T

(x)  T + σDT<sup>1</sup>/<sup>p</sup>

.

519

#### <sup>520</sup> A.2 Extension to Online Strongly Convex Optimization

<sup>521</sup> Next, we extend Theorem [1](#page-4-1) to the strongly convex case, i.e., ∃µ > 0 such that for all t ∈ [T],

$$\frac{\mu}{2} \|\mathbf{x} - \mathbf{y}\|^2 + \langle \nabla \ell_t(\mathbf{y}), \mathbf{x} - \mathbf{y} \rangle + \ell_t(\mathbf{y}) \leq \ell_t(\mathbf{x}), \forall \mathbf{x}, \mathbf{y} \in \mathcal{X}, \nabla \ell_t(\mathbf{y}) \in \partial \ell_t(\mathbf{y}). \quad (16)$$

<sup>522</sup> In this setting, it is well known that OGD achieves a logarithmic regret bound when p = 2 [\[10,](#page-9-1) [27\]](#page-10-0). <sup>523</sup> Theorem [6](#page-13-4) below provides the first provable result for p < 2.

Theorem 6. *Under Assumption [1](#page-2-0) and additionally assuming [\(16\)](#page-13-5), taking* η<sup>t</sup> = 1 µt <sup>524</sup> *in* OGD *(Algorithm* <sup>525</sup> *[1\)](#page-3-5), we have*

$$\mathbb{E} [\mathbf{R}_T^{\text{OGD}}(\mathbf{x})] \lesssim \frac{G^2(1 + \log T)}{\mu} + \frac{\sigma^p G^{2-p}}{\mu} \times \begin{cases} T^{2-p} & \mathbf{p} \in (1, 2), \\ 1 + \log T & \mathbf{p} = 2 \end{cases}, \forall \mathbf{x} \in \mathcal{X}.$$

<sup>526</sup> Theorem [6](#page-13-4) shows that under strongly convexity, OGD for p ∈ (1, 2) achieves a better sublinear regret T 2−p than T 1/p <sup>527</sup> in Theorem [1](#page-4-1) as 2 − p ≤ 1/p, ∀p > 0. One point we highlight here is that the stepsize η<sup>t</sup> = 1 µt <sup>528</sup> is commonly used in the OCO literature and is independent of the tail index p.

<sup>529</sup> However, in contrast to Theorem [1,](#page-4-1) we suspect Theorem [6](#page-13-4) is not tight in T for p ∈ (1, 2). The reason <sup>530</sup> is that for nonsmooth strongly convex optimization with heavy tails (i.e., ℓ<sup>t</sup> = F, ∀t ∈ [T] where F is strongly convex), Theorem [6](#page-13-4) can convert to a convergence rate only in the order of 1/T<sup>p</sup>−<sup>1</sup> <sup>531</sup> , which is worse than the lower bound 1/T<sup>2</sup>−2/<sup>p</sup> <sup>532</sup> [\[45\]](#page-12-1). Therefore, we conjecture that a way to obtain a better regret bound than T 2−p <sup>533</sup> exists, which we leave as future work.

<sup>534</sup> *Proof of Theorem [6.](#page-13-4)* For any x ∈ X , we take expectations on both sides of [\(12\)](#page-13-6) to have

$$\mathbb{E} [\mathbf{R}_T^{\text{OGD}}(\mathbf{x})] \leq \left( \frac{1}{\eta_1} - \mu \right) \frac{\|\mathbf{x}_1 - \mathbf{x}\|^2}{2} + \sum_{t=1}^{T-1} \left( \frac{1}{\eta_{t+1}} - \frac{1}{\eta_t} - \mu \right) \frac{\mathbb{E} [\|\mathbf{x}_{t+1} - \mathbf{x}\|^2]}{2} \\ + \sum_{t=1}^T \eta_t G^2 + \mathbf{C}(\mathbf{p}) \eta_t^{\mathbf{p}-1} \sigma^\mathbf{p} D^{2-\mathbf{p}}, \quad (17)$$

<sup>535</sup> where for the L.H.S., we follow a similar step of reasoning out [\(15\)](#page-13-7) but instead using

$$\langle \nabla \ell_t(\mathbf{x}_t), \mathbf{x}_t - \mathbf{x} \rangle \geq \ell_t(\mathbf{x}_t) - \ell_t(\mathbf{x}) + \frac{\mu}{2} \|\mathbf{x}_t - \mathbf{x}\|^2,$$

for the R.H.S., we use E -∥ϵt∥ p ≤ σ p <sup>536</sup> .

Next, we plug η<sup>t</sup> = µt <sup>537</sup> , ∀t ∈ [T] into [\(17\)](#page-14-1) to obtain

$$\begin{aligned}\mathbb{E} [\mathbf{R}_T^{\text{OGD}}(\mathbf{x})] &\lesssim \sum_{t=1}^T \frac{G^2}{\mu t} + \frac{\sigma^{\mathbf{p}} D^{2-\mathbf{p}}}{\mu^{\mathbf{p}-1} t^{\mathbf{p}-1}} \\ &\lesssim \frac{G^2 (1 + \log T)}{\mu} + \frac{\sigma^{\mathbf{p}} D^{2-\mathbf{p}}}{\mu^{\mathbf{p}-1}} \times \begin{cases} T^{2-\mathbf{p}} & \mathbf{p} \in (1, 2) \\ 1 + \log T & \mathbf{p} = 2 \end{cases}\end{aligned}$$

<sup>538</sup> Lastly, it is known that if ℓ<sup>t</sup> is G-Lipschitz and µ-strongly convex on a domain X with a diameter D, then it satisfies D ≲ G µ <sup>539</sup> (e.g., see Lemma 2 of [\[31\]](#page-11-12)). Therefore, when p ∈ (1, 2),

$$\mathbb{E} [R_T^{\text{OGD}}(\mathbf{x})] \lesssim \frac{G^2 (1 + \log T)}{\mu} + \frac{\sigma^p G^{2-p}}{\mu} T^{2-p}.$$

540

## <sup>541</sup> B Missing Proofs for Dual Averaging

<sup>542</sup> This section provides missing proofs for regret bounds of DA.

## <sup>543</sup> B.1 Proof of Theorem [2](#page-4-2)

*Proof.* Let Lt(x) ≜ ∥x−x1∥ 2ηt−<sup>1</sup> + P<sup>t</sup>−<sup>1</sup> <sup>s</sup>=1 ⟨g<sup>s</sup> <sup>544</sup> , x⟩, ∀t ∈ [T + 1], where η<sup>0</sup> ≜ η1. Then DA can be <sup>545</sup> equivalently written as

$$\mathbf{x}_t = \operatorname{argmin}_{\mathbf{x} \in \mathcal{X}} L_t(\mathbf{x}), \forall t \in [T+1].$$

<sup>546</sup> By Lemma 7.1 of [\[27\]](#page-10-0), for any x ∈ X ,

$$\begin{aligned} \sum_{t=1}^T \langle \mathbf{g}_t, \mathbf{x}_t - \mathbf{x} \rangle &= \frac{\|\mathbf{x} - \mathbf{x}_1\|^2}{2\eta_T} + L_{T+1}(\mathbf{x}_{T+1}) - L_{T+1}(\mathbf{x}) + \sum_{t=1}^T L_t(\mathbf{x}_t) + \langle \mathbf{g}_t, \mathbf{x}_t \rangle - L_{t+1}(\mathbf{x}_{t+1}) \\ &\leq \frac{\|\mathbf{x} - \mathbf{x}_1\|^2}{2\eta_T} + \sum_{t=1}^T L_t(\mathbf{x}_t) - L_{t+1}(\mathbf{x}_{t+1}) + \langle \mathbf{g}_t, \mathbf{x}_t \rangle, \end{aligned}$$

<sup>547</sup> where the inequality holds by L<sup>T</sup> +1(x<sup>T</sup> +1) ≤ L<sup>T</sup> +1(x), ∀x ∈ X due to x<sup>T</sup> +1 = <sup>548</sup> argminx∈X L<sup>T</sup> +1(x). Note that for any t ∈ [T],

$$\begin{aligned} & L_t(\mathbf{x}_t) - L_{t+1}(\mathbf{x}_{t+1}) + \langle \mathbf{g}_t, \mathbf{x}_t \rangle \\ &= L_t(\mathbf{x}_t) - L_t(\mathbf{x}_{t+1}) + \langle \mathbf{g}_t, \mathbf{x}_t - \mathbf{x}_{t+1} \rangle + \frac{\|\mathbf{x}_{t+1} - \mathbf{x}_1\|^2}{2\eta_{t-1}} - \frac{\|\mathbf{x}_{t+1} - \mathbf{x}_1\|^2}{2\eta_t} \\ &\stackrel{(a)}{\leq} L_t(\mathbf{x}_t) - L_t(\mathbf{x}_{t+1}) + \langle \mathbf{g}_t, \mathbf{x}_t - \mathbf{x}_{t+1} \rangle \\ &\stackrel{(b)}{\leq} \langle \mathbf{g}_t, \mathbf{x}_t - \mathbf{x}_{t+1} \rangle - \frac{\|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2}{2\eta_{t-1}}, \end{aligned}$$

where (a) is by η<sup>t</sup> ≤ ηt−1, ∀t ∈ [T] and (b) is holds because L<sup>t</sup> is <sup>1</sup> ηt−<sup>1</sup> <sup>549</sup> -strongly convex and <sup>550</sup> x<sup>t</sup> = argminx∈X Lt(x), which together imply

$$L_t(\mathbf{x}_t) - L_t(\mathbf{x}_{t+1}) \leq \langle \nabla L_t(\mathbf{x}_t), \mathbf{x}_t - \mathbf{x}_{t+1} \rangle - \frac{\|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2}{2\eta_{t-1}} \leq -\frac{\|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2}{2\eta_{t-1}}.$$

<sup>551</sup> Therefore, we have

$$\sum_{t=1}^T \langle g_t, x_t - x \rangle \leq \frac{\|x - x_1\|^2}{2\eta_T} + \sum_{t=1}^T \langle g_t, x_t - x_{t+1} \rangle - \frac{\|x_t - x_{t+1}\|^2}{2\eta_{t-1}}. \quad (18)$$

<sup>552</sup> By the same argument as proving [\(11\)](#page-13-8) but replacing η<sup>t</sup> with ηt−1, there is

$$\langle g_t, x_t - x_{t+1} \rangle - \frac{\|x_t - x_{t+1}\|^2}{2\eta_{t-1}} \leq \eta_{t-1} G^2 + C(p) \eta_{t-1}^{p-1} \|\epsilon_t\|^p D^{2-p}.$$

<sup>553</sup> As such, we know

$$\sum_{t=1}^T \langle g_t, \mathbf{x}_t - \mathbf{x} \rangle \leq \frac{\|\mathbf{x} - \mathbf{x}_1\|^2}{2\eta_T} + \sum_{t=1}^T \eta_{t-1} G^2 + C(p) \eta_{t-1}^{p-1} \|\epsilon_t\|^p D^{2-p}.$$

<sup>554</sup> Finally, following similar steps in proving Theorem [1](#page-4-1) in Appendix [A,](#page-13-0) we conclude

$$\mathbb{E} \left[ \mathbb{R}_T^{\text{DA}}(\mathbf{x}) \right] \lesssim GD\sqrt{T} + \sigma DT^{1/p}.$$

#### <sup>556</sup> B.2 Dual Averaging with an Adaptive Stepsize

We show that DA with an adaptive stepsize can also achieve the optimal regret GD√ T + σDT<sup>1</sup>/<sup>p</sup> <sup>557</sup> .

Theorem 7. *Under Assumption [1,](#page-2-0) taking* <sup>η</sup><sup>t</sup> = 2DV <sup>−</sup>1/<sup>2</sup> <sup>t</sup> *and* V<sup>t</sup> = P<sup>t</sup> <sup>s</sup>=1 ∥gs∥ 2 <sup>558</sup> *in* DA *(Algorithm* <sup>559</sup> *[2\)](#page-4-3), we have*

$$\mathbb{E} [\mathbf{R}_T^{\text{DA}}(\mathbf{x})] \lesssim GD\sqrt{T} + \sigma DT^{1/p}, \forall \mathbf{x} \in \mathcal{X}.$$

<sup>560</sup> *Proof.* For any x ∈ X , we have

$$\sum_{t=1}^T \langle \mathbf{g}_t, \mathbf{x}_t - \mathbf{x} \rangle \stackrel{(18)}{\leq} \frac{\|\mathbf{x} - \mathbf{x}_1\|^2}{2\eta_T} + \sum_{t=1}^T \langle \mathbf{g}_t, \mathbf{x}_t - \mathbf{x}_{t+1} \rangle - \frac{\|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2}{2\eta_{t-1}}, \quad (19)$$

<sup>561</sup> where η<sup>0</sup> ≜ η1. On the one hand, we can use AM-GM inequality to bound

$$\langle g_t, x_t - x_{t+1} \rangle - \frac{\|x_t - x_{t+1}\|^2}{2\eta_{t-1}} \leq \frac{\eta_{t-1} \|g_t\|^2}{2}.$$

<sup>562</sup> On the other hand, we know

$$\langle \mathbf{g}_t, \mathbf{x}_t - \mathbf{x}_{t+1} \rangle - \frac{\|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2}{2\eta_{t-1}} \leq \langle \mathbf{g}_t, \mathbf{x}_t - \mathbf{x}_{t+1} \rangle \leq \|\mathbf{g}_t\| \|\mathbf{x}_t - \mathbf{x}_{t+1}\| \leq \|\mathbf{g}_t\| D, \quad (20)$$

<sup>563</sup> where the second step is by Cauchy-Schwarz inequality. Therefore, for any t ≥ 2,

$$\begin{aligned} \langle \mathbf{g}_t, \mathbf{x}_t - \mathbf{x}_{t+1} \rangle - \frac{\|\mathbf{x}_t - \mathbf{x}_{t+1}\|^2}{2\eta_{t-1}} &\leq \frac{\eta_{t-1} \|\mathbf{g}_t\|^2}{2} \wedge \|\mathbf{g}_t\| D \stackrel{(a)}{\leq} \frac{2}{\frac{2}{\eta_{t-1} \|\mathbf{g}_t\|^2} + \frac{1}{\|\mathbf{g}_t\| D}} \\ &\stackrel{(b)}{=} \frac{2D \|\mathbf{g}_t\|^2}{\sqrt{\sum_{s=1}^{t-1} \|\mathbf{g}_s\|^2 + \|\mathbf{g}_t\|}} \stackrel{(c)}{\leq} \frac{2D \|\mathbf{g}_t\|^2}{\sqrt{\sum_{s=1}^t \|\mathbf{g}_s\|^2}}, \end{aligned} \quad (21)$$

where (a) is due to x ∧ y ≤ 2 <sup>x</sup>−1+y−<sup>1</sup> , <sup>∀</sup>x, y > <sup>0</sup>, (b) is by <sup>η</sup>t−<sup>1</sup> <sup>=</sup> √ 2D P<sup>t</sup>−<sup>1</sup> <sup>s</sup>=1∥gs∥ 2 <sup>564</sup> , and (c) holds because of qP<sup>t</sup> <sup>s</sup>=1 ∥gs∥ <sup>2</sup> ≤ qPt−<sup>1</sup> <sup>s</sup>=1 ∥gs∥ 2 <sup>565</sup> + ∥gt∥. Note that [\(21\)](#page-16-1) is also true for t = 1 by [\(20\)](#page-15-2).

<sup>566</sup> Combine [\(19\)](#page-15-3) and [\(21\)](#page-16-1) and use ∥x − x1∥ ≤ D to obtain

$$\sum_{t=1}^T \langle \mathbf{g}_t, \mathbf{x}_t - \mathbf{x} \rangle \leq \frac{D^2}{2\eta_T} + \sum_{t=1}^T \frac{2D \|\mathbf{g}_t\|^2}{\sqrt{\sum_{s=1}^t \|\mathbf{g}_s\|^2}} = \frac{D^2}{2\eta_T} + \sum_{t=1}^T \eta_t \|\mathbf{g}_t\|^2,$$

<sup>567</sup> which only differs from [\(22\)](#page-16-2) by a constant. Hence, by a similar proof for [\(24\)](#page-17-2), there is

$$\sum_{t=1}^T \langle \mathbf{g}_t, \mathbf{x}_t - \mathbf{x} \rangle \lesssim D \left[ \sqrt{\sum_{t=1}^T \|\nabla \ell_t(\mathbf{x}_t)\|^2} + \left( \sum_{t=1}^T \|\epsilon_t\|^p \right)^{\frac{1}{p}} \right],$$

<sup>568</sup> implying

568 implying 
$$\mathbb{E} [\mathbf{R}_T^{\text{DA}}(\mathbf{x})] \lesssim GD\sqrt{T} + \sigma DT^{1/p}$$
.

569

## <sup>570</sup> C Missing Proofs for AdaGrad

<sup>571</sup> This section provides missing proofs for regret bounds of AdaGrad.

## <sup>572</sup> C.1 Proof of Theorem [3](#page-5-1)

*Proof.* As mentioned, AdaGrad can be viewed as OGD with a stepsize η<sup>t</sup> = √ η V<sup>t</sup> <sup>=</sup> √ η P<sup>t</sup> <sup>s</sup>=1∥gs∥ 2 <sup>573</sup> . <sup>574</sup> Therefore, we can use [\(1\)](#page-3-0) for AdaGrad to know for any x ∈ X ,

$$\langle \mathbf{g}_t, \mathbf{x}_t - \mathbf{x} \rangle \leq \frac{\|\mathbf{x}_t - \mathbf{x}\|^2 - \|\mathbf{x}_{t+1} - \mathbf{x}\|^2}{2\eta_t} + \frac{\eta_t \|\mathbf{g}_t\|^2}{2}.$$

Sum up the above inequality from t = 1 to T and drop the term − ∥x<sup>T</sup> +1−x∥ 2η<sup>T</sup> <sup>575</sup> to have

$$\begin{aligned} \sum_{t=1}^T \langle \mathbf{g}_t, \mathbf{x}_t - \mathbf{x} \rangle &\leq \frac{\|\mathbf{x}_1 - \mathbf{x}\|^2}{2\eta_1} + \sum_{t=1}^T \left( \frac{1}{\eta_{t+1}} - \frac{1}{\eta_t} \right) \frac{\|\mathbf{x}_{t+1} - \mathbf{x}\|^2}{2} + \sum_{t=1}^T \frac{\eta_t \|\mathbf{g}_t\|^2}{2} \\ &\leq \frac{D^2}{2\eta_T} + \sum_{t=1}^T \frac{\eta_t \|\mathbf{g}_t\|^2}{2}, \end{aligned} \quad (22)$$

<sup>576</sup> where the last step is by ∥x<sup>t</sup> − x∥ ≤ D, ∀t ∈ [T] and ηt+1 ≤ ηt, ∀t ∈ [T − 1].

<sup>577</sup> Next, observe that for any t ∈ [T],

$$\|\mathbf{g}_t\|^2 = \frac{\eta^2}{\eta_t^2} - \frac{\eta^2}{\eta_{t-1}^2} = \eta^2 \left( \frac{1}{\eta_t} - \frac{1}{\eta_{t-1}} \right) \left( \frac{1}{\eta_t} + \frac{1}{\eta_{t-1}} \right) \leq \frac{2\eta^2}{\eta_t} \left( \frac{1}{\eta_t} - \frac{1}{\eta_{t-1}} \right),$$

<sup>578</sup> where 1/η<sup>0</sup> should be read as 0. The above inequality implies

$$\sum_{t=1}^T \frac{\eta_t \|\mathbf{g}_t\|^2}{2} \leq \eta^2 \sum_{t=1}^T \frac{1}{\eta_t} - \frac{1}{\eta_{t-1}} = \frac{\eta^2}{\eta_T}. \quad (23)$$

<sup>579</sup> Combine [\(22\)](#page-16-2) and [\(23\)](#page-16-3) to have

$$\sum_{t=1}^T \langle g_t, x_t - x \rangle \leq \frac{D^2}{2\eta_T} + \frac{\eta^2}{\eta_T} = \left( \frac{D^2}{2\eta} + \eta \right) \sqrt{\sum_{t=1}^T \|g_t\|^2}.$$

<sup>580</sup> Note that there is

$$\begin{aligned} \sqrt{\sum_{t=1}^T \|\mathbf{g}_t\|^2} &\leq \sqrt{\sum_{t=1}^T 2 \|\nabla \ell_t(\mathbf{x}_t)\|^2 + 2 \|\boldsymbol{\epsilon}_t\|^2} \leq \sqrt{2 \sum_{t=1}^T \|\nabla \ell_t(\mathbf{x}_t)\|^2} + \sqrt{2 \sum_{t=1}^T \|\boldsymbol{\epsilon}_t\|^2} \\ &\leq \sqrt{2 \sum_{t=1}^T \|\nabla \ell_t(\mathbf{x}_t)\|^2 + \sqrt{2} \left( \sum_{t=1}^T \|\boldsymbol{\epsilon}_t\|^{\mathbf{p}} \right)^{\frac{1}{\mathbf{p}}}}, \end{aligned}$$

where the last step is due to ∥·∥<sup>2</sup> ≤ ∥·∥<sup>p</sup> <sup>581</sup> for any p ∈ [1, 2]. Hence, we obtain

$$\sum_{t=1}^T \langle g_t, x_t - x \rangle \leq \sqrt{2} \left( \frac{D^2}{2\eta} + \eta \right) \left[ \sqrt{\sum_{t=1}^T \|\nabla \ell_t(x_t)\|^2} + \left( \sum_{t=1}^T \|\epsilon_t\|^p \right)^{\frac{1}{p}} \right]. \quad (24)$$

<sup>582</sup> We take expectations on both sides of [\(24\)](#page-17-2), then apply Hölder's inequality to have

$$\mathbb{E} \left[ \left( \sum_{t=1}^T \|\epsilon_t\|^{\text{p}} \right)^{\frac{1}{\text{p}}} \right] \leq \left( \sum_{t=1}^T \mathbb{E} [\|\epsilon_t\|^{\text{p}}] \right)^{\frac{1}{\text{p}}} \leq \sigma T^{\frac{1}{\text{p}}},$$

and finally plug in <sup>η</sup> <sup>=</sup> D/√ <sup>583</sup> 2 to conclude

$$\mathbb{E} \left[ \mathbf{R}_T^{\text{AdaGrad}}(\mathbf{x}) \right] \lesssim GD\sqrt{T} + \sigma DT^{1/p}.$$

584

## <sup>585</sup> D Missing Proofs for Applications: Nonsmooth Convex Optimization

<sup>586</sup> We prove the following last-iterate convergence result for SGD (i.e., OGD for stochastic optimization) <sup>587</sup> under heavy-tailed noise. The proof of Theorem [8](#page-17-1) is inspired by [\[19,](#page-10-9) [44\]](#page-12-6).

<sup>588</sup> Theorem 8. *Under Assumption [1](#page-2-0) [f](#page-2-0)or* ℓt(x) = F(x)*, for any stepsize* η<sup>t</sup> > 0 *in* OGD *(Algorithm [1\)](#page-3-5),* <sup>589</sup> *we have*

$$\mathbb{E} [F(\mathbf{x}_T) - F(\mathbf{x})] \lesssim \frac{D^2}{\sum_{t=1}^T \eta_t} + G^2 \sum_{t=1}^T \frac{\eta_t^2}{\sum_{s=(t+1) \wedge T}^T \eta_s} + \sigma^p D^{2-p} \sum_{t=1}^T \frac{\eta_t^p}{\sum_{s=(t+1) \wedge T}^T \eta_s}.$$

<sup>590</sup> *Proof.* Given x ∈ X , we recursively define

$$\mathbf{y}_0 \triangleq \mathbf{x} \quad \text{and} \quad \mathbf{y}_t \triangleq \left(1 - \frac{w_{t-1}}{w_t}\right) \mathbf{x}_t + \frac{w_{t-1}}{w_t} \mathbf{y}_{t-1}, \forall t \in [T], \quad (25)$$

<sup>591</sup> in which

$$w_t \triangleq \frac{\eta_T}{\sum_{s=t+1}^T \eta_s}, \forall t \in \{0\} \cup [T-1] \quad \text{and} \quad w_T \triangleq w_{T-1} = 1. \quad (26)$$

Equivalently, y<sup>t</sup> <sup>592</sup> can be written into a convex combination of x, x1, . . . , x<sup>t</sup> as

$$\mathbf{y}_t = \frac{w_0}{w_t} \mathbf{x} + \sum_{s=1}^t \frac{w_s - w_{s-1}}{w_t} \mathbf{x}_s, \forall t \{0\} \cup [T]. \quad (27)$$

We invoke [\(10\)](#page-13-1) for y<sup>t</sup> <sup>594</sup> to obtain

$$\langle g_t, x_t - y_t \rangle \leq \frac{\|x_t - y_t\|^2 - \|x_{t+1} - y_t\|^2}{2\eta_t} + \eta_t G^2 + C(p)\eta_t^{p-1} \|\epsilon_t\|^p D^{2-p}. \quad (28)$$

<sup>595</sup> Since xt, y<sup>t</sup> ∈ Ft−1, there is

$$\mathbb{E}[\langle g_t, x_t - y_t \rangle] = \mathbb{E}[\langle \mathbb{E}[g_t \mid \mathcal{F}_{t-1}], x_t - y_t \rangle] = \mathbb{E}[\langle \nabla F(x_t), x_t - y_t \rangle] \geq \mathbb{E}[F(x_t) - F(y_t)],$$

<sup>596</sup> where the last step is due to the convexity of F. As such, we can take expectations on both sides of <sup>597</sup> [\(28\)](#page-18-0) to have

$$\begin{aligned}\mathbb{E}[F(\mathbf{x}_t) - F(\mathbf{y}_t)] &\leq \frac{\mathbb{E}\left[\|\mathbf{x}_t - \mathbf{y}_t\|^2\right] - \mathbb{E}\left[\|\mathbf{x}_{t+1} - \mathbf{y}_t\|^2\right]}{2\eta_t} + \eta_t G^2 + C(\mathbf{p})\eta_t^{\mathbf{p}-1}\sigma^\mathbf{p}D^{2-\mathbf{p}} \\ &\leq \frac{\mathbb{E}\left[\frac{w_{t-1}}{w_t} \|\mathbf{x}_t - \mathbf{y}_{t-1}\|^2\right] - \mathbb{E}\left[\|\mathbf{x}_{t+1} - \mathbf{y}_t\|^2\right]}{2\eta_t} + \eta_t G^2 + C(\mathbf{p})\eta_t^{\mathbf{p}-1}\sigma^\mathbf{p}D^{2-\mathbf{p}},\end{aligned}\tag{29}$$

where the second step is due to ∥x<sup>t</sup> − yt∥ <sup>2</sup> ≤ 1 − wt−<sup>1</sup> w<sup>t</sup> ∥x<sup>t</sup> − xt∥ <sup>2</sup> + wt−<sup>1</sup> w<sup>t</sup> x<sup>t</sup> − yt−<sup>1</sup> 2 <sup>598</sup> = wt−<sup>1</sup> w<sup>t</sup> x<sup>t</sup> − yt−<sup>1</sup> 2 by [\(25\)](#page-17-3) and the convexity of ∥x<sup>t</sup> − ·∥<sup>2</sup> <sup>599</sup> . Mutiply both sides of [\(29\)](#page-18-1) by wtη<sup>t</sup> and <sup>600</sup> sum up from t = 1 to T to obtain

$$\begin{aligned} & \mathbb{E} \left[ \sum_{t=1}^T w_t \eta_t (F(\mathbf{x}_t) - F(\mathbf{y}_t)) \right] \\ & \leq \frac{w_0 \|\mathbf{x}_1 - \mathbf{y}_0\|^2 - \mathbb{E} [w_T \|\mathbf{x}_{T+1} - \mathbf{y}_T\|^2]}{2} + \sum_{t=1}^T w_t \eta_t^2 G^2 + C(\mathbf{p}) w_t \eta_t^{\mathbf{p}} \sigma^{\mathbf{p}} D^{2-\mathbf{p}} \\ & \leq \frac{w_0 D^2}{2} + \sum_{t=1}^T w_t \eta_t^2 G^2 + C(\mathbf{p}) w_t \eta_t^{\mathbf{p}} \sigma^{\mathbf{p}} D^{2-\mathbf{p}}. \end{aligned} \quad (30)$$

<sup>601</sup> Now observe that

$$F(\mathbf{y}_t) - F(\mathbf{x}) \stackrel{(27)}{\leq} \frac{w_0}{w_t} (F(\mathbf{x}) - F(\mathbf{x})) + \sum_{s=1}^t \frac{w_s - w_{s-1}}{w_t} (F(\mathbf{x}_s) - F(\mathbf{x})),$$

$$= \sum_{s=1}^t \frac{w_s - w_{s-1}}{w_t} (F(\mathbf{x}_s) - F(\mathbf{x})),$$

<sup>602</sup> which implies

$$\begin{aligned} \sum_{t=1}^T w_t \eta_t (F(\mathbf{y}_t) - F(\mathbf{x})) &\leq \sum_{t=1}^T \sum_{s=1}^t (w_s - w_{s-1}) \eta_t (F(\mathbf{x}_s) - F(\mathbf{x})) \\ &= \sum_{t=1}^T (w_t - w_{t-1}) \left( \sum_{s=t}^T \eta_s \right) (F(\mathbf{x}_t) - F(\mathbf{x})). \end{aligned}$$

<sup>603</sup> Thus, we can lower bound the L.H.S. of [\(30\)](#page-18-2) by

$$\begin{aligned} \sum_{t=1}^T w_t \eta_t (F(\mathbf{x}_t) - F(\mathbf{y}_t)) &= \sum_{t=1}^T w_t \eta_t (F(\mathbf{x}_t) - F(\mathbf{x})) - w_t \eta_t (F(\mathbf{y}_t) - F(\mathbf{x})) \\ &\geq \sum_{t=1}^T \left[ w_t \eta_t - (w_t - w_{t-1}) \left( \sum_{s=t}^T \eta_s \right) \right] (F(\mathbf{x}_t) - F(\mathbf{x})) \\ &= w_T \eta_T (F(\mathbf{x}_T) - F(\mathbf{x})), \end{aligned} \quad (31)$$

<sup>604</sup> where the last step is due to, for t ∈ [T − 1],

$$\begin{aligned} w_t \eta_t - (w_t - w_{t-1}) \left( \sum_{s=t}^T \eta_s \right) &\stackrel{(26)}{=} \frac{\eta_T}{\sum_{s=t+1}^T \eta_s} \cdot \eta_t - \left( \frac{\eta_T}{\sum_{s=t+1}^T \eta_s} - \frac{\eta_T}{\sum_{s=t}^T \eta_s} \right) \left( \sum_{s=t}^T \eta_s \right) \\ &= \frac{\eta_T}{\sum_{s=t+1}^T \eta_s} \cdot \eta_t - \frac{\eta_T}{\sum_{s=t+1}^T \eta_s} \cdot \eta_t = 0, \end{aligned}$$

and w<sup>T</sup> [\(26\)](#page-17-5) <sup>605</sup> = w<sup>T</sup> <sup>−</sup><sup>1</sup> = 1.

<sup>606</sup> We plug [\(31\)](#page-18-3) back into [\(30\)](#page-18-2) and divide both sides by w<sup>T</sup> η<sup>T</sup> to obtain

$$\begin{aligned}\mathbb{E} [F(\mathbf{x}_T) - F(\mathbf{x})] &\leq \frac{w_0 D^2}{2w_T \eta_T} + \sum_{t=1}^T \frac{w_t \eta_t^2}{w_T \eta_T} G^2 + C(\mathbf{p}) \frac{w_t \eta_t^p}{w_T \eta_T} \sigma^{\mathbf{p}} D^{2-\mathbf{p}} \\ &\stackrel{(26)}{\lesssim} \frac{D^2}{\sum_{t=1}^T \eta_t} + G^2 \sum_{t=1}^T \frac{\eta_t^2}{\sum_{s=(t+1) \wedge T}^T \eta_s} + \sigma^{\mathbf{p}} D^{2-\mathbf{p}} \sum_{t=1}^T \frac{\eta_t^p}{\sum_{s=(t+1) \wedge T}^T \eta_s}.\end{aligned}$$

607

 Equipped with Theorem [8,](#page-17-1) we show the following anytime last-iterate convergence rate for SGD/OGD. As far as we know, this is the first and the only provable result demonstrating that the last iterate of SGD can converge in heavy-tailed stochastic optimization without gradient clipping. Compared to Corollary [2,](#page-6-3) the difference is up to an extra logarithmic factor. Therefore, it is nearly optimal.

Corollary 4. *Under Assumption [1](#page-2-0) for* ℓt(x) = F(x)*, taking* η<sup>t</sup> = D G √ t ∧ D σt1/<sup>p</sup> <sup>612</sup> *in* OGD *(Algorithm* <sup>613</sup> *[1\)](#page-3-5), we have*

$$\mathbb{E} [F(\mathbf{x}_T) - F(\mathbf{x})] \lesssim \frac{GD(1 + \log T)}{\sqrt{T}} + \frac{\sigma D(1 + \log T)}{T^{1-\frac{1}{p}}}.$$

<sup>614</sup> *Proof.* By Theorem [8,](#page-17-1) we have

$$\begin{aligned} & \mathbb{E} [F(\mathbf{x}_T) - F(\mathbf{x})] \\ & \lesssim \frac{D^2}{\sum_{t=1}^T \eta_t} + G^2 \sum_{t=1}^T \frac{\eta_t^2}{\sum_{s=(t+1) \wedge T}^T \eta_s} + \sigma^p D^{2-p} \sum_{t=1}^T \frac{\eta_t^p}{\sum_{s=(t+1) \wedge T}^T \eta_s} \\ & = \frac{D^2}{\sum_{t=1}^T \eta_t} + G^2 \left( \eta_T + \sum_{t=1}^{T-1} \frac{\eta_t^2}{\sum_{s=t+1}^T \eta_s} \right) + \sigma^p D^{2-p} \left( \eta_T^{p-1} + \sum_{t=1}^{T-1} \frac{\eta_t^p}{\sum_{s=t+1}^T \eta_s} \right). \end{aligned}$$

<sup>615</sup> For any t ∈ {0} ∪ [T − 1], observe that by Cauchy-Schwarz inequality

$$(T - t)^2 \leq \left( \sum_{s=t+1}^T \frac{1}{\eta_s} \right) \left( \sum_{s=t+1}^T \eta_s \right) \Rightarrow \frac{1}{\sum_{s=t+1}^T \eta_s} \leq \frac{\sum_{s=t+1}^T \frac{1}{\eta_s}}{(T - t)^2}.$$

<sup>616</sup> Thus, there is

$$\begin{aligned} \mathbb{E}[F(\mathbf{x}_T) - F(\mathbf{x})] &\lesssim \frac{D^2}{T^2} \sum_{t=1}^T \frac{1}{\eta_t} + G^2 \left( \eta_T + \sum_{t=1}^{T-1} \frac{\eta_t^2 \sum_{s=t+1}^T \frac{1}{\eta_s}}{(T-t)^2} \right) \\ &\quad + \sigma^p D^{2-p} \left( \eta_T^{p-1} + \sum_{t=1}^{T-1} \frac{\eta_t^p \sum_{s=t+1}^T \frac{1}{\eta_s}}{(T-t)^2} \right). \end{aligned} \quad (32)$$

<sup>617</sup> We first bound

$$\sum_{t=1}^T \frac{1}{\eta_t} = \sum_{t=1}^T \frac{G\sqrt{t}}{D} \vee \frac{\sigma t^{1/p}}{D} \leq \sum_{t=1}^T \frac{G\sqrt{t}}{D} + \frac{\sigma t^{1/p}}{D} \lesssim \frac{G}{D} T^{3/2} + \frac{\sigma}{D} T^{1+1/p},$$

<sup>618</sup> which implies

$$\frac{D^2}{T^2} \sum_{t=1}^T \frac{1}{\eta_t} \lesssim \frac{GD}{\sqrt{T}} + \frac{\sigma D}{T^{1-\frac{1}{p}}}. \quad (33)$$

<sup>619</sup> Next, we know

$$\eta_T + \sum_{t=1}^{T-1} \frac{\eta_t^2 \sum_{s=t+1}^T \frac{1}{\eta_s}}{(T-t)^2} \stackrel{(a)}{\leq} \frac{D}{G\sqrt{T}} + \sum_{t=1}^{T-1} \left[ \frac{D}{G} \cdot \frac{\sum_{s=t+1}^T \sqrt{s}}{t(T-t)^2} + \frac{\sigma D}{G^2} \cdot \frac{\sum_{s=t+1}^T s^{1/p}}{t(T-t)^2} \right] \\ \lesssim \frac{\text{Fact 1}}{G\sqrt{T}} + \frac{D}{G\sqrt{T}} + \frac{D(1 + \log T)}{G\sqrt{T}} + \frac{\sigma D(1 + \log T)}{G^2 T^{1-\frac{1}{p}}},$$

where (a) is by η<sup>t</sup> ≤ D G √ t and <sup>1</sup> η<sup>s</sup> ≤ G √ s D ∨ σs1/<sup>p</sup> D <sup>620</sup> . Hence, there is

$$G^2 \left( \eta_T + \sum_{t=1}^{T-1} \frac{\eta_t^2 \sum_{s=t+1}^T \frac{1}{\eta_s}}{(T-t)^2} \right) \lesssim \frac{GD(1 + \log T)}{\sqrt{T}} + \frac{\sigma D(1 + \log T)}{T^{1-\frac{1}{p}}}. \quad (34)$$

<sup>621</sup> Similarly, we can bound

$$\sigma^p D^{2-p} \left( \eta_T^{p-1} + \sum_{t=1}^{T-1} \frac{\eta_t^p \sum_{s=t+1}^T \frac{1}{\eta_s^2}}{(T-t)^2} \right) \lesssim \frac{GD (1 + \log T)}{\sqrt{T}} + \frac{\sigma D (1 + \log T)}{T^{1-\frac{1}{p}}}. \quad (35)$$

<sup>622</sup> Finally, we plug [\(33\)](#page-20-2), [\(34\)](#page-20-3) and [\(35\)](#page-20-4) back into [\(32\)](#page-19-1) to conclude.

## <sup>623</sup> E Missing Proofs for Applications: Nonsmooth Nonconvex Optimization

## <sup>624</sup> E.1 (δ, ϵ)-Stationary Points

Definition 2 (Definition 4 of [\[5\]](#page-9-11)). A point x ∈ R d <sup>625</sup> is a (δ, ϵ)-stationary point of an almost-everywhere differentiable function F if there is a finite subset S ⊂ B<sup>d</sup> <sup>626</sup> (x, δ) such that for y selected uniformly at <sup>627</sup> random from S, <sup>E</sup> [y] = x and ∥<sup>E</sup> [∇F(y)]∥ ≤ ϵ.

<sup>628</sup> The concept of the (δ, ϵ)-stationary point presented here is due to [\[5\]](#page-9-11), which is mildly more stringent <sup>629</sup> than the notion of [\[46\]](#page-12-8), since the latter does not require <sup>E</sup> [y] = x. For more discussions, see Section <sup>630</sup> 2.1 of [\[5\]](#page-9-11).

## <sup>631</sup> E.2 Proof of Theorem [4](#page-8-1)

 In this section, our ultimate goal is to prove Theorem [4](#page-8-1) for the O2NC algorithm, extending Theorem 8 of [\[5\]](#page-9-11) from p = 2 to any p ∈ (1, 2]. Notably, our new result does not require any modification to the O2NC method, but is obtained only from a more careful analysis, indicating that O2NC is a robust and powerful algorithmic framework.

<sup>636</sup> We begin with Lemma [2,](#page-20-5) which lies as the cornerstone for establishing the convergence of O2NC.

<sup>637</sup> Lemma 2 (Theorem 7 of [\[5\]](#page-9-11)). *Under Assumption [2](#page-7-0) (only need the second point and the unbiased part in the fourth point), for any sequence of vectors* u1, . . . ,uKT ∈ <sup>R</sup> d <sup>638</sup> *,* O2NC *(Algorithm [4\)](#page-7-1) guarantees*

$$\mathbb{E}[F(\mathbf{y}_{KT})] = F(\mathbf{y}_0) + \mathbb{E} \left[ \sum_{n=1}^{KT} \langle \mathbf{g}_n, \mathbf{x}_n - \mathbf{u}_n \rangle \right] + \mathbb{E} \left[ \sum_{n=1}^{KT} \langle \mathbf{g}_n, \mathbf{u}_n \rangle \right]. \quad (36)$$

<sup>639</sup> To relate Lemma [2](#page-20-5) to the concept of K-shifting regret introduced before (see [\(9\)](#page-8-0)), suppose now a <sup>640</sup> sequence of vectors v1, . . . , v<sup>K</sup> is given, if we set u<sup>n</sup> = v<sup>k</sup> for all n ∈ {(k − 1)T + 1, . . . , kT} and k ∈ [K], then the second term on the R.H.S. of [\(36\)](#page-20-6) can be written as <sup>E</sup> -R A T (v1, . . . , vK) <sup>641</sup> , and the third term can be simplified into P<sup>K</sup> <sup>k</sup>=1 <sup>E</sup> hDPkT <sup>n</sup>=(k−1)<sup>T</sup> +1 gn, v<sup>k</sup> Ei <sup>642</sup> .

Same as [\[5\]](#page-9-11), we pick v<sup>k</sup> ≜ −D PkT <sup>n</sup>=(k−1)<sup>T</sup> +1 ∇F (zn) PkT <sup>n</sup>=(k−1)<sup>T</sup> +1 <sup>∇</sup><sup>F</sup> (zn)∥ <sup>643</sup> for some constant D > 0, which gives us

$$\begin{aligned}\mathbb{E} \left[ \left\langle \sum_{n=(k-1)T+1}^{kT} \mathbf{g}_n, \mathbf{v}_k \right\rangle \right] &= \mathbb{E} \left[ \left\langle \sum_{n=(k-1)T+1}^{kT} \mathbf{e}_n, \mathbf{v}_k \right\rangle \right] - D\mathbb{E} \left[ \left\| \sum_{n=(k-1)T+1}^{kT} \nabla F(\mathbf{z}_n) \right\| \right] \\ &\leq D\mathbb{E} \left[ \left\| \sum_{n=(k-1)T+1}^{kT} \mathbf{e}_n \right\| \right] - D\mathbb{E} \left[ \left\| \sum_{n=(k-1)T+1}^{kT} \nabla F(\mathbf{z}_n) \right\| \right].\end{aligned}$$

<sup>644</sup> If ϵ<sup>n</sup> has a finite variance (i.e., p = 2), then like [\[5\]](#page-9-11), one can invoke Hölder's inequality and use the <sup>645</sup> fact <sup>E</sup> [⟨ϵm, ϵn⟩] = 0, ∀m ̸= n ∈ [KT] to obtain for any k ∈ [K],

$$\mathbb{E} \left[ \left\| \sum_{n=(k-1)T+1}^{kT} \boldsymbol{\epsilon}_n \right\| \right] \leq \sqrt{\mathbb{E} \left[ \left\| \sum_{n=(k-1)T+1}^{kT} \boldsymbol{\epsilon}_n \right\|^2 \right]} = \sqrt{\sum_{n=(k-1)T+1}^{kT} \mathbb{E} \left[ \|\boldsymbol{\epsilon}_n\|^2 \right]} \leq \sigma \sqrt{T}.$$

However, this argument immediately fails when p < 2 as E h ∥ϵn∥ 2 i <sup>646</sup> can be +∞. To handle this <sup>647</sup> potential issue, we require the following Lemma [3.](#page-21-0)

<sup>648</sup> Lemma 3 (Lemma 4.3 of [\[20\]](#page-10-14)). *Given a vector-valued martingale difference sequence* w1, . . . , w<sup>T</sup> *,* <sup>649</sup> *there is*

$$\mathbb{E} \left[ \left\| \sum_{t=1}^T \mathbf{w}_t \right\| \right] \leq 2\sqrt{2} \mathbb{E} \left[ \left( \sum_{t=1}^T \|\mathbf{w}_t\|^p \right)^{\frac{1}{p}} \right], \forall \mathbf{p} \in [1, 2].$$

<sup>650</sup> Equipped with Lemmas [2](#page-20-5) and [3,](#page-21-0) we are ready to formally prove Theorem [4,](#page-8-1) demonstrating that the <sup>651</sup> O2NC framework provably works under heavy-tailed noise.

<sup>652</sup> *Proof of Theorem [4.](#page-8-1)* We invoke Lemma [2](#page-20-5) with u<sup>n</sup> = v⌈n/T⌉, ∀n ∈ [KT] (equivalently, u<sup>n</sup> = v<sup>k</sup> if <sup>653</sup> n ∈ {(k − 1)T + 1, . . . , kT}) and use the definition of K-shifting regret (see [\(9\)](#page-8-0)) to obtain

$$\mathbb{E}[F(\mathbf{y}_{KT})] = F(\mathbf{y}_0) + \mathbb{E}[\mathbf{R}_T^A(\mathbf{v}_1, \dots, \mathbf{v}_K)] + \sum_{k=1}^K \mathbb{E}\left[\left\langle \sum_{n=(k-1)T+1}^{kT} \mathbf{g}_n, \mathbf{v}_k \right\rangle\right]. \quad (37)$$

<sup>654</sup> Recall that g<sup>n</sup> = ∇F(zn) + ϵn, which implies for any k ∈ [K],

$$\begin{aligned}\mathbb{E} \left[ \left\langle \sum_{n=(k-1)T+1}^{kT} \mathbf{g}_n, \mathbf{v}_k \right\rangle \right] &= \mathbb{E} \left[ \left\langle \sum_{n=(k-1)T+1}^{kT} \epsilon_n, \mathbf{v}_k \right\rangle \right] + \mathbb{E} \left[ \left\langle \sum_{n=(k-1)T+1}^{kT} \nabla F(\mathbf{z}_n), \mathbf{v}_k \right\rangle \right] \\ &\leq \mathbb{E} \left[ \left\| \sum_{n=(k-1)T+1}^{kT} \epsilon_n \right\| \left\| \mathbf{v}_k \right\| \right] + \mathbb{E} \left[ \left\langle \sum_{n=(k-1)T+1}^{kT} \nabla F(\mathbf{z}_n), \mathbf{v}_k \right\rangle \right] \\ &= D\mathbb{E} \left[ \left\| \sum_{n=(k-1)T+1}^{kT} \epsilon_n \right\| \right] - D\mathbb{E} \left[ \left\| \sum_{n=(k-1)T+1}^{kT} \nabla F(\mathbf{z}_n) \right\| \right], \quad (38)\end{aligned}$$

<sup>655</sup> where the second step is by Cauchy-Schwarz inequality and the last equation holds due to

$$v_k = -D \frac{\sum_{n=(k-1)T+1}^{kT} \nabla F(\mathbf{z}_n)}{\left\| \sum_{n=(k-1)T+1}^{kT} \nabla F(\mathbf{z}_n) \right\|}, \forall k \in [K]. \quad (39)$$

Combine [\(37\)](#page-21-1) and [\(38\)](#page-21-2), apply F(yKT <sup>656</sup> ) ≥ F⋆, and rearrange terms to have

$$\begin{aligned} & \mathbb{E} \left[ \sum_{k=1}^K \frac{1}{K} \left\| \frac{1}{T} \sum_{n=(k-1)T+1}^{kT} \nabla F(\mathbf{z}_n) \right\| \right] \\ & \leq \frac{F(\mathbf{y}_0) - F_*}{DKT} + \frac{\mathbb{E} [R_T^A(\mathbf{v}_1, \dots, \mathbf{v}_K)]}{DKT} + \frac{\sum_{k=1}^K \mathbb{E} \left[ \left\| \sum_{n=(k-1)T+1}^{kT} \epsilon_n \right\| \right]}{KT}. \end{aligned} \quad (40)$$

For any fixed k ∈ [K], we apply Lemma [3](#page-21-0) with w<sup>t</sup> = ϵ(k−1)<sup>T</sup> <sup>+</sup><sup>t</sup> <sup>657</sup> , ∀t ∈ [T] to know

$$\begin{aligned}\mathbb{E} \left[ \left\| \sum_{n=(k-1)T+1}^{kT} \boldsymbol{\epsilon}_n \right\| \right] &\leq 2\sqrt{2}\mathbb{E} \left[ \left( \sum_{n=(k-1)T+1}^{kT} \|\boldsymbol{\epsilon}_n\|^p \right)^{\frac{1}{p}} \right] \\ &\leq 2\sqrt{2} \left( \sum_{n=(k-1)T+1}^{kT} \mathbb{E} [\|\boldsymbol{\epsilon}_n\|^p] \right)^{\frac{1}{p}} \leq 2\sqrt{2}\sigma T^{\frac{1}{p}}, \quad (41)\end{aligned}$$

<sup>658</sup> where the second step is by Hölder's inequality (note that p > 1). Finally, we conclude the proof <sup>659</sup> after plugging [\(41\)](#page-22-0) back into [\(40\)](#page-21-3).

## <sup>660</sup> E.3 Proof of Theorem [5](#page-8-2)

<sup>661</sup> *Proof.* By Theorem [4,](#page-8-1) there is

$$\mathbb{E} \left[ \sum_{k=1}^K \frac{1}{K} \left\| \frac{1}{T} \sum_{n=(k-1)T+1}^{kT} \nabla F(\mathbf{z}_n) \right\| \right] \lesssim \frac{F(\mathbf{y}_0) - F_\star}{DKT} + \frac{\mathbb{E} [\mathbf{R}_T^A(\mathbf{v}_1, \dots, \mathbf{v}_K)]}{DKT} + \frac{\sigma}{T^{1-\frac{1}{\rho}}}. \quad (42)$$

Note that A has the domain X = B d <sup>662</sup> (D) and s<sup>n</sup> ∼ Uniform [0, 1]. Thus, for any n ∈ [KT],

$$\|\mathbf{x}_n\| \leq D \quad \text{and} \quad s_n \in [0, 1]. \quad (43)$$

<sup>663</sup> We first lower bound the L.H.S. of [\(42\)](#page-22-1). Given k ∈ [K], for any m < n ∈ {(k − 1)T + 1, . . . , kT}, <sup>664</sup> observe that

$$\begin{aligned} \|\mathbf{z}_n - \mathbf{z}_m\| &= \left\| \mathbf{y}_{n-1} + s_n \mathbf{x}_n - \mathbf{y}_{m-1} - s_m \mathbf{x}_m \right\| = \left\| s_n \mathbf{x}_n - s_m \mathbf{x}_m + \sum_{i=m}^{n-1} \mathbf{x}_i \right\| \\ &\leq s_n \|\mathbf{x}_n\| + (1 - s_m) \|\mathbf{x}_m\| + \sum_{i=m+1}^{n-1} \|\mathbf{x}_i\| \stackrel{(43)}{\leq} (n - m + 1) D \leq DT. \end{aligned}$$

Recall that z¯<sup>k</sup> = 1 T PkT <sup>n</sup>=(k−1)<sup>T</sup> +1 <sup>665</sup> z<sup>n</sup> and D = δ/T now, then the above inequality implies

$$\|\mathbf{z}_n - \bar{\mathbf{z}}_k\| \leq DT = \delta, \forall n \in \{(k-1)T + 1, \dots, kT\}, \quad (44)$$

<sup>666</sup> which means

$$\mathbf{z}_n \in \mathcal{B}^d(\bar{\mathbf{z}}_k, \delta), \forall n \in \{(k-1)T + 1, \dots, kT\}.$$

By the definition of ∥∇F(z¯k)∥<sup>δ</sup> <sup>667</sup> (see Definition [1\)](#page-7-2), there is

$$\|\nabla F(\bar{\mathbf{z}}_k)\|_{\delta} \leq \left\| \frac{1}{T} \sum_{n=(k-1)T+1}^{kT} \nabla F(\mathbf{z}_n) \right\|. \quad (45)$$

<sup>668</sup> Next, we upper bound the R.H.S. of [\(42\)](#page-22-1). By the definition of K-shifting regret (see [\(9\)](#page-8-0)), there is

$$\mathbb{E} \left[ \mathbf{R}_T^A(\mathbf{v}_1, \dots, \mathbf{v}_K) \right] = \sum_{k=1}^K \mathbb{E} \left[ \sum_{n=(k-1)T+1}^{kT} \langle \mathbf{g}_n, \mathbf{x}_n - \mathbf{v}_k \rangle \right].$$

Note that we reset the stepsize in A after every T iterations and v<sup>k</sup> ∈ B<sup>d</sup> <sup>669</sup> (D) by its definition (see [\(39\)](#page-21-4)). Then for any A ∈ {OGD, DA, AdaGrad}, we can invoke its regret bound[<sup>2</sup>](#page-22-3) <sup>670</sup> (i.e., Theorems [1,](#page-4-1) [2](#page-4-2) <sup>671</sup> and [3\)](#page-5-1) to obtain

$$\mathbb{E} \left[ \sum_{n=(k-1)T+1}^{kT} \langle \mathbf{g}_n, \mathbf{x}_n - \mathbf{v}_k \rangle \right] \lesssim GD\sqrt{T} + \sigma DT^{1/p}, \forall k \in [K],$$

<sup>2</sup>A minor point here is that the current function ℓn(x) = ⟨gn, x⟩ does not entirely fit Assumption [1.](#page-2-0) We clarify that one does not need to worry about it, since all results proved in Section [3](#page-2-1) hold under this change. For example, in the proof of Theorem [1,](#page-4-1) we can safely replace the L.H.S. of [\(14\)](#page-13-3) with E hP<sup>T</sup> <sup>t</sup>=1 ⟨g<sup>t</sup> , x<sup>t</sup> − x⟩ i .

<sup>672</sup> which implies

$$\mathbb{E} [\mathbf{R}_T^A(\mathbf{v}_1, \dots, \mathbf{v}_K)] \lesssim GDK\sqrt{T} + \sigma DKT^{1/p}. \quad (46)$$

Finally, we plug [\(45\)](#page-22-4) and [\(46\)](#page-23-1) back into [\(42\)](#page-22-1), then use D = δ/T and ∆ = F(y<sup>0</sup> <sup>673</sup> ) − F<sup>⋆</sup> to have

$$\mathbb{E} \left[ \frac{1}{K} \sum_{k=1}^K \|\nabla F(\bar{\mathbf{z}}_k)\|_{\delta} \right] \lesssim \frac{\Delta}{\delta K} + \frac{G}{\sqrt{T}} + \frac{\sigma}{T^{1-\frac{1}{p}}}.$$

674

## <sup>675</sup> E.4 Proof of Corollary [3](#page-8-3)

<sup>676</sup> *Proof.* Recall that we pick

$$K = \left\lfloor \frac{N}{T} \right\rfloor \quad \text{and} \quad T = \left\lfloor \frac{N}{2} \right\rfloor \wedge \left( \left\lceil \left( \frac{\delta GN}{\Delta} \right)^{\frac{2}{3}} \right\rceil \right) \vee \left\lceil \left( \frac{\delta \sigma N}{\Delta} \right)^{\frac{p}{2p-1}} \right\rceil \right),$$

where ∆ = F(y<sup>0</sup> <sup>677</sup> ) − F⋆. We invoke Theorem [5](#page-8-2) and use KT ≥ N/4 (see Fact [2\)](#page-24-1) to obtain

$$\mathbb{E} \left[ \frac{1}{K} \sum_{k=1}^K \|\nabla F(\bar{\mathbf{z}}_k)\|_{\delta} \right] \lesssim \frac{\Delta T}{\delta N} + \frac{G}{\sqrt{T}} + \frac{\sigma}{T^{1-\frac{1}{p}}}.$$

<sup>678</sup> By the definition of T, we know

$$\frac{\Delta T}{\delta N} \lesssim \frac{\Delta}{\delta N} \left[ 1 + \left( \frac{\delta G N}{\Delta} \right)^{\frac{2}{3}} + \left( \frac{\delta \sigma N}{\Delta} \right)^{\frac{p}{2p-1}} \right] = \frac{\Delta}{\delta N} + \frac{G^{\frac{2}{3}} \Delta^{\frac{1}{3}}}{(\delta N)^{\frac{1}{3}}} + \frac{\sigma^{\frac{p}{2p-1}} \Delta^{\frac{p-1}{2p-1}}}{(\delta N)^{\frac{p-1}{2p-1}}},$$

<sup>679</sup> and

$$\frac{G}{\sqrt{T}} \lesssim \frac{G}{\sqrt{N}} + \frac{G\frac{2}{3}\Delta\frac{1}{3}}{(\delta N)^{\frac{1}{3}}}, \quad \frac{\sigma}{T^{1-\frac{1}{p}}} \lesssim \frac{\sigma}{N^{1-\frac{1}{p}}} + \frac{\sigma\frac{p-1}{2p-1}\Delta\frac{p-1}{2p-1}}{(\delta N)^{\frac{p-1}{2p-1}}}.$$

<sup>680</sup> Therefore, there is

$$\mathbb{E} \left[ \frac{1}{K} \sum_{k=1}^K \|\nabla F(\bar{\mathbf{z}}_k)\|_{\delta} \right] \lesssim \frac{G}{\sqrt{N}} + \frac{\sigma}{N^{1-\frac{1}{p}}} + \frac{\Delta}{\delta N} + \frac{G^{\frac{2}{3}} \Delta^{\frac{1}{3}}}{(\delta N)^{\frac{1}{3}}} + \frac{\sigma^{\frac{p}{2p-1}} \Delta^{\frac{p-1}{2p-1}}}{(\delta N)^{\frac{p-1}{2p-1}}}.$$

681

## <sup>682</sup> E.5 Extension to the Case of Unknown Problem-Dependent Parameters

 In Corollary [5,](#page-23-0) we show how to set K and T when all problem-dependent parameters are unknown. It is particularly meaningful for AdaGrad. As in that case, the rate is achieved without knowing any problem-dependent parameter. This kind of result is the first to appear for nonsmooth nonconvex optimization with heavy tails. However, the rate is not as good as Corollary [3.](#page-8-3) It is currently unclear whether the same bound 1/(δN) p−1 <sup>2</sup>p−<sup>1</sup> as in Corollary [3](#page-8-3) can be obtained when no information about the problem is known.

<sup>689</sup> Corollary 5. *Under the same setting of Theorem [5,](#page-8-2) suppose we have* N ≥ 2 *stochastic gradient budgets, taking* <sup>K</sup> <sup>=</sup> ⌊N/T⌋ *and* <sup>T</sup> <sup>=</sup> ⌈N/2⌉ ∧ l (δN) 2 m <sup>690</sup> *, we have*

$$\mathbb{E} \left[ \frac{1}{K} \sum_{k=1}^K \|\nabla F(\bar{\mathbf{z}}_k)\|_{\delta} \right] \lesssim \frac{\Delta}{(\delta N) \wedge (\delta N)^{\frac{1}{3}}} + \frac{G}{\sqrt{N} \wedge (\delta N)^{\frac{1}{3}}} + \frac{\sigma}{N^{1-\frac{1}{p}} \wedge (\delta N)^{\frac{2(p-1)}{3p}}}.$$

<sup>691</sup> *Proof.* We invoke Theorem [5](#page-8-2) and use KT ≥ N/4 (see Fact [2\)](#page-24-1) to obtain

$$\mathbb{E} \left[ \frac{1}{K} \sum_{k=1}^K \|\nabla F(\bar{\mathbf{z}}_k)\|_{\delta} \right] \lesssim \frac{\Delta T}{\delta N} + \frac{G}{\sqrt{T}} + \frac{\sigma}{T^{1-\frac{1}{p}}}.$$

<sup>692</sup> By the definition of T, we know

$$\frac{\Delta T}{\delta N} \lesssim \frac{\Delta}{\delta N} \left[ 1 + (\delta N)^{\frac{2}{3}} \right] \lesssim \frac{\Delta}{(\delta N) \wedge (\delta N)^{\frac{1}{3}}}.$$

<sup>693</sup> and

$$\frac{G}{\sqrt{T}} \lesssim \frac{G}{\sqrt{N}} + \frac{G}{(\delta N)^{\frac{1}{3}}} \lesssim \frac{G}{\sqrt{N} \wedge (\delta N)^{\frac{1}{3}}},$$

$$\frac{\sigma}{T^{1-\frac{1}{p}}} \lesssim \frac{\sigma}{N^{1-\frac{1}{p}}} + \frac{\sigma}{(\delta N)^{\frac{2(p-1)}{3p}}} \lesssim \frac{\sigma}{N^{1-\frac{1}{p}} \wedge (\delta N)^{\frac{2(p-1)}{3p}}}.$$

<sup>694</sup> Therefore, there is

$$\mathbb{E} \left[ \frac{1}{K} \sum_{k=1}^K \|\nabla F(\bar{z}_k)\|_{\delta} \right] \lesssim \frac{\Delta}{(\delta N) \wedge (\delta N)^{\frac{1}{3}}} + \frac{G}{\sqrt{N} \wedge (\delta N)^{\frac{1}{3}}} + \frac{\sigma}{N^{1-\frac{1}{p}} \wedge (\delta N)^{\frac{2(p-1)}{3p}}}.$$

695

## <sup>696</sup> F Algebraic Facts

<sup>697</sup> We give two useful algebraic facts in this section.

<sup>698</sup> Fact 1. *For any* T ∈ N *and* a ∈ (0, 1)*, there is*

$$\sum_{t=1}^{T-1} \frac{\sum_{s=t+1}^T s^a}{t(T-t)^2} \lesssim \frac{1 + \log T}{T^{1-a}}.$$

*Proof.* Note that P<sup>T</sup> <sup>s</sup>=t+1 s <sup>a</sup> ≤ (T − t)T a <sup>699</sup> , which implies

$$\sum_{t=1}^{T-1} \frac{\sum_{s=t+1}^T s^a}{t(T-t)^2} \leq \sum_{t=1}^{T-1} \frac{T^a}{t(T-t)} = \frac{1}{T^{1-a}} \sum_{t=1}^{T-1} \frac{1}{t} + \frac{1}{T-t} = \frac{2}{T^{1-a}} \sum_{t=1}^{T-1} \frac{1}{t} \lesssim \frac{1 + \log T}{T^{1-a}}.$$

700

<sup>701</sup> Fact 2. *Given* 2 ≤ N ∈ N*,* K = ⌊N/T⌋ *and* T ∈ N *satisfying* T ≤ ⌈N/2⌉*, there is* KT ≥ N/4*.*

<sup>702</sup> *Proof.* Note that KT = ⌊N/T⌋ T ≥ N − T ≥ (N − 1)/2 ≥ N/4.

## NeurIPS Paper Checklist

#### 1. Claims

 Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

 Justification: The main claims made in the abstract and introduction accurately reflect the paper's contributions and scope.

Guidelines:

 • The answer NA means that the abstract and introduction do not include the claims made in the paper. • The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers. • The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings. • It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

#### 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We discuss the limitation in Section [5.](#page-8-4)

Guidelines:

 • The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper. • The authors are encouraged to create a separate "Limitations" section in their paper. • The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be. • The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated. • The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon. • The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size. • If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness. • While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an impor- tant role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

 Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

 Justification: For each theoretical result, the paper provides the full set of assumptions and a complete (and correct) proof.

Guidelines:

 • The answer NA means that the paper does not include theoretical results. • All the theorems, formulas, and proofs in the paper should be numbered and cross- referenced. • All assumptions should be clearly stated or referenced in the statement of any theorems. • The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition. • Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

 Question: Does the paper fully disclose all the information needed to reproduce the main ex- perimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [NA]

Justification: The paper does not include experiments.

Guidelines:

 • The answer NA means that the paper does not include experiments. • If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not. • If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable. • Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed. • While NeurIPS does not require releasing code, the conference does require all submis- sions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

 Question: Does the paper provide open access to the data and code, with sufficient instruc- tions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [NA]

Justification: The paper does not include experiments requiring code.

Guidelines:

 • The answer NA means that paper does not include experiments requiring code. • Please see the NeurIPS code and data submission guidelines ([https://nips.cc/](https://nips.cc/public/guides/CodeSubmissionPolicy) [public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details. • While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark). • The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ([https:](https://nips.cc/public/guides/CodeSubmissionPolicy) [//nips.cc/public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details. • The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc. • The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why. • At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable). • Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

 Question: Does the paper specify all the training and test details (e.g., data splits, hyper- parameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [NA]

Justification: The paper does not include experiments.

Guidelines:

 • The answer NA means that the paper does not include experiments. • The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them. • The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

 Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [NA]

Justification: The paper does not include experiments.

Guidelines:

 • The answer NA means that the paper does not include experiments. • The authors should answer "Yes" if the results are accompanied by error bars, confi- dence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper. • The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions). • The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors). • It should be clear whether the error bar is the standard deviation or the standard error of the mean.

 • It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified. • For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates). • If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

 Question: For each experiment, does the paper provide sufficient information on the com- puter resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [NA]

Justification: The paper does not include experiments.

Guidelines:

 • The answer NA means that the paper does not include experiments. • The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage. • The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute. • The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

 Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?

Answer: [Yes]

 Justification: The research conducted in the paper conforms, in every respect, with the NeurIPS Code of Ethics.

Guidelines:

 • The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. • If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics. • The authors should make sure to preserve anonymity (e.g., if there is a special consid-eration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

 Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

 Justification: There is no societal impact of the work performed because this paper is purely theoretical.

Guidelines:

 • The answer NA means that there is no societal impact of the work performed. • If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact. • Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

 • The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster. • The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology. • If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

#### 11. Safeguards

 Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: The paper poses no such risks.

Guidelines:

 • The answer NA means that the paper poses no such risks. • Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters. • Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images. • We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

#### 12. Licenses for existing assets

 Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: The paper does not use existing assets.

Guidelines:

 • The answer NA means that the paper does not use existing assets. • The authors should cite the original paper that produced the code package or dataset. • The authors should state which version of the asset is used and, if possible, include a URL. • The name of the license (e.g., CC-BY 4.0) should be included for each asset. • For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided. • If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, <paperswithcode.com/datasets> has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset. • For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

 • If this information is not available online, the authors are encouraged to reach out to the asset's creators.

#### 13. New assets

 Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: The paper does not release new assets.

Guidelines:

 • The answer NA means that the paper does not release new assets. • Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc. • The paper should discuss whether and how consent was obtained from people whose asset is used. • At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

#### 14. Crowdsourcing and research with human subjects

 Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects.

Guidelines:

 • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribu- tion of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

#### 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

 Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

 Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

 • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

 Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

 Justification: The core method development in this research does not involve LLMs as any important, original, or non-standard components.

Guidelines:

 • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (<https://neurips.cc/Conferences/2025/LLM>) for what should or should not be described.