# Multi-Marginal Schrodinger Bridge Matching ¨

Anonymous Author(s) Affiliation Address email

# Abstract

 Understanding the continuous evolution of populations from discrete temporal snapshots is a critical research challenge, particularly in fields like developmental biology and systems medicine where longitudinal tracking of individual entities is often impossible. Such trajectory inference is vital for unraveling the mechanisms of dynamic processes. While Schrodinger Bridge (SB) offer a potent framework, ¨ their traditional application to pairwise time points can be insufficient for systems defined by multiple intermediate snapshots. This paper introduces Multi-Marginal Schrodinger Bridge Matching (MSBM), a novel algorithm specifically designed ¨ for the multi-marginal SB problem. MSBM extends iterative Markovian fitting (IMF) to effectively handle multiple marginal constraints. This technique ensures robust enforcement of all intermediate marginals while preserving the continuity of the learned global dynamics across the entire trajectory. Empirical validations on synthetic data and real-world single-cell RNA sequencing datasets demonstrate the competitive or superior performance of MSBM in capturing complex trajectories and respecting intermediate distributions, all with notable computational efficiency.

## 1 Introduction

 Understanding the continuous evolution of populations from discrete temporal snapshots represents a significant challenge in various scientific disciplines, particularly in fields like developmental biology [\[7,](#page-9-0) [42\]](#page-11-0) and systems medicine [\[29\]](#page-10-0) where tracking individual entities longitudinally is often unfeasible. The ability to infer trajectories from such snapshot data is crucial for elucidating the underlying mechanisms of dynamic processes. The Schrodinger Bridge (SB) problem, originally ¨ rooted in statistical mechanics [\[43\]](#page-11-1), has garnered substantial interest in machine learning as an entropy-regularized, continuous-time formulation of optimal transport [\[20,](#page-10-1) [30\]](#page-10-2). It seeks to identify the most probable evolutionary path between prescribed initial and terminal distributions, and has been successfully employed in generative modeling [\[3,](#page-9-1) [4,](#page-9-2) [9,](#page-9-3) [26,](#page-10-3) [27,](#page-10-4) [37,](#page-11-2) [38,](#page-11-3) [45,](#page-11-4) [49\]](#page-11-5).

 However, many real-world scenarios present observations or constraints at multiple time points, not just at the beginning and end of a process. For instance, in single-cell RNA sequencing (scRNA-seq) experiments, which are pivotal for studying complex biological processes like cell differentiation, cells are typically destroyed upon measurement [\[6,](#page-9-4) [17,](#page-10-5) [28\]](#page-10-6). This destructive nature makes it impossible to track individual cells over time, thus necessitating the inference of developmental trajectories from population-level snapshots collected at several intermediate stages. Similarly, meteorological systems may have partial observations across various times [\[11,](#page-9-5) [32\]](#page-10-7). Such situations necessitate a multi-marginal generalization of the SB problem (mSBP), where the path measure must align with prescribed marginal distributions at multiple intermediate time points. While the traditional SB framework offers a powerful approach, its standard application to pairwise time points can prove insufficient for systems characterized by multiple intermediate snapshots. Although more specialized methods for mSBP have recently been developed [\[8,](#page-9-6) [18,](#page-10-8) [44\]](#page-11-6), the direct application of some multi-marginal approaches can lead to error accumulation if not carefully managed, particularly  when learned controls are even slightly inaccurate. These challenges highlight the need for robust and scalable solutions for the mSBP that can effectively integrate information across all observed time points.

 This paper introduces Multi-Marginal Schrodinger Bridge Matching (MSBM), a novel algorithm ¨ specifically developed to address the multi-marginal SB problem by building upon and extending the Iterative Markovian Fitting (IMF) algoritmhs [\[36,](#page-11-7) [45\]](#page-11-4). MSBM is designed to effectively manage mul- tiple marginal constraints by constructing local SBs on each interval and seamlessly integrating them. This local construction strategy, underpinned by a shared global parametrization of control functions, ensures the robust enforcement of all intermediate marginal distributions while crucially preserving the continuity of the learned global dynamics across the entire trajectory. Empirical validations conducted on synthetic datasets as well as real-world single-cell RNA sequencing data demonstrate that MSBM achieves competitive or superior performance in capturing complex trajectories and accurately respecting intermediate distributions, all while exhibiting notable computational efficiency. Our work aims to provide a robust and scalable computational method for these multi-marginal settings, addressing the critical need for consistent and tractable dynamic inference when data is available as snapshots at multiple time points.

We summarize our contributions as follows:

 • We extend the theoretical and algorithmic foundations of SBs, including the IMF iteration and optimal control perspectives, to the challenging multi-marginal setting. • We introduce an efficient modeling approach for trajectory inference, that constructs and smoothly integrates local SBs across sub-intervals, inherently allows for parallelized train- ing, leading to significant speed-ups. • Through comprehensive experiments on both synthetic and real-world single-cell RNA sequenc- ing data, we demonstrate that MSBM accurately models complex population dynamics and outperforms state-of-the-art methods in both trajectory fidelity and computational speed.

Notation. Let P[0,T] denote the space of continuous functions taking values in <sup>R</sup> d on the interval [0, T]. We use an uppercase letter <sup>P</sup> ∈ P[0,T] to represent a path measure. For a path measure <sup>P</sup> ∈ P[0,T] , the marginal distribution at discrete time points T = {t0, . . . , tk}, where 0 = t<sup>0</sup> < t<sup>1</sup> < · · · < t<sup>k</sup> = T is denoted by <sup>P</sup><sup>T</sup> ∈ P<sup>T</sup> , where we define P<sup>T</sup> as the set of measures <sup>P</sup> over <sup>R</sup> <sup>d</sup>×|T | . Additionally, the conditional distribution of <sup>P</sup>, given T , is denoted by <sup>P</sup>|T ∈ P[0,T] . Moreover, a path measure <sup>P</sup> can be defined as mixture. For any Borel measurable set A ∈ B(Ω), <sup>P</sup> can be defined by <sup>P</sup>(A) = R <sup>R</sup>d×|T | <sup>P</sup>|T (A|x<sup>T</sup> )dP<sup>T</sup> (x<sup>T</sup> ), where <sup>P</sup> ∈ P0,T and <sup>P</sup> ∈ P<sup>T</sup> , and we use the shorthand x<sup>T</sup> := (x1, · · · , xk) and [0 : k] := {0, 1, · · · , k}. The Kullback-Leibler (KL) divergence between two probability measures µ and ν on space X is defined as DKL(µ|ν) = R X log dµ dν (X)dµ(X) when µ is absolutely continuous with respect to ν (µ ≪ ν), and DKL(µ|ν) = +∞ otherwise. We will often refer to probability measures on R d and their Lebesgue densities interchangeably, under the standard assumption of absolute continuity. Finally, for a function V : [0, T] × <sup>R</sup> <sup>d</sup> → <sup>R</sup>, we define the gradient and laplcaian operators with respect to x ∈ R d as ∇V and ∆V, respectively, and its partial derivative with respect to time t ∈ [0, T] as ∂tV.

# 2 Preliminaries

# 2.1 Schrodinger Bridge Matching (SBM) ¨

 The Schrodinger Bridge problem (SBP) [ ¨ [16,](#page-10-9) [43\]](#page-11-1) is a stochastic optimal transport problem [\[30\]](#page-10-2) that seeks the optimal transport plan for endpoint marginals ρ<sup>0</sup> and ρ<sup>T</sup> . In this paper, we focus on the dynamical representation, where a reference distribution Q ∈ P[0,T] is induced by the SDEs:

$$d\mathbf{X}_t = f_t(\mathbf{X}_t) dt + \sigma d\mathbf{W}_t, \quad \mathbf{X}_0 \sim \rho_0, \quad (1)$$

where f<sup>t</sup> : <sup>R</sup> <sup>d</sup> → <sup>R</sup> d is a drift, σ ∈ <sup>R</sup> is a diffusion, and W<sup>t</sup> ∈ <sup>R</sup> d is a standard Wiener process. With the base reference path measure Q, the dynamic representation of the SB [\[20,](#page-10-1) [35,](#page-11-8) [39\]](#page-11-9) is:

$$\min_{\mathbb{P} \in \mathcal{P}_{[0,T]}} D_{\text{KL}}(\mathbb{P}|\mathbb{Q}), \quad \text{subject to} \quad \mathbb{P}_0 \sim \rho_0, \quad \mathbb{P}_T \sim \rho_T. \quad (\text{SBP})$$

 Recent advancements in dynamical optimal transport [\[37,](#page-11-2) [45\]](#page-11-4) have introduced a novel numerical methodology for solving [SBP](#page-1-0). This approach reframes [SBP](#page-1-0) by decomposing its dynamical constraints into the time-evolving marginal distributions <sup>P</sup><sup>t</sup> for all t ∈ [0, T] and the joint coupling <sup>P</sup>0,T . This optimization relies on IMF [\[45\]](#page-11-4), a technique that iteratively refines the path measure <sup>P</sup> ∈ P[0,T] <sup>88</sup> . IMF alternates between two projection called Markovian and Reciprocal projections to preserve the correct endpoint marginals (ρ0, ρ<sup>T</sup> ) throughout the optimization.

<sup>91</sup> Reciprocal Projection R. For a given reference measure Q from [\(1\)](#page-1-1), and a path measure P with <sup>92</sup> marginals specified at end points T = {0, T} the reciprocal projection is defined as:

$$\mathcal{R}(\mathbb{P}, \mathcal{T}) := \mathbb{P}_{\mathcal{T}} \mathbb{Q}_{|\mathcal{T}} = \mathbb{P}_{0, T} \mathbb{Q}_{|0, T}. \quad (2)$$

 This projection constructs a new path measure by taking the endpoint coupling P0,T from P and forming a mixture of bridge process using Q conditioned on these end points. Sampling from Π := R(P, T ) involves drawing end points samples (X0, X<sup>T</sup> ) ∼ <sup>P</sup>0,T and then generating a path X<sup>T</sup> <sup>t</sup> between them using conditional reference measure Q|0,T which induced by following SDEs, for any (x0, x<sup>T</sup> ):

$$d\mathbf{X}_t^T = [f_t(\mathbf{X}_t^T) + \sigma^2 \nabla \log \mathbb{Q}_{T|t}(\mathbf{x}_T|\mathbf{X}_t^T)] dt + \sigma d\mathbf{W}_t, \quad \mathbf{X}_0^T = \mathbf{x}_0, \quad (3)$$

<sup>98</sup> If Q|0,T has tractable bridge formulation, for example, when Q is chosen as a Brownian motion <sup>99</sup> *i*.*e*., dX<sup>t</sup> = σdWt, sampling the path at time t given the endpoints can be performed as:

$$\mathbf{X}_t^T \sim \mathcal{N}\left((1 - \frac{t}{T})\mathbf{X}_0 + \frac{t}{T}\mathbf{X}_T, t(1 - \frac{t}{T})\sigma^2\right), \quad \text{where } (\mathbf{X}_0, \mathbf{X}_T) \sim \mathbb{P}_{0,T}. \quad (4)$$

 Markov Projection M. Although the reciprocal projection R in [\(2\)](#page-2-0) preserves end point marginals (ρ0, ρ<sup>T</sup> ), its sampling process in [\(4\)](#page-2-1) requires both (X0, X<sup>T</sup> ), making it non-Markovian and thus ill-suited for generative modeling aimed at sampling from ρ<sup>T</sup> without knowing X<sup>T</sup> . The Markov projection M resolves this by projecting Π := R(P, T ) into a family of Markov process while ensuring <sup>P</sup> <sup>104</sup> <sup>⋆</sup> = Π<sup>t</sup> for all t ∈ [0, T]. Again, when Q is chosen as a Brownian motion *i*.*e*., dX<sup>t</sup> = σdWt, the Markov projection of Π, <sup>P</sup> <sup>105</sup> <sup>⋆</sup> = M(Π, T ), is induced by following SDEs:

$$d\mathbf{X}_t^* = \sigma v^*(t, \mathbf{X}_t^*) dt + \sigma d\mathbf{W}_t, \quad \mathbf{X}_0^* \sim \Pi_0, \quad (5)$$

where 
$$\mathbf{v}^*(t, \mathbf{x}) = \frac{1}{T-t} (\mathbb{E}_{\mathbb{Q}_T|t} [\mathbf{X}_T | \mathbf{X}_t = \mathbf{x}] - \mathbf{x})$$
. (6)

Intuitively, the term <sup>E</sup>Q<sup>T</sup> <sup>|</sup><sup>t</sup> [X<sup>T</sup> |X<sup>t</sup> = x] can be understood as a prediction of the target state X<sup>⋆</sup> t <sup>106</sup> . Flow matching [\[23\]](#page-10-10) of Bridge matching [\[37\]](#page-11-2) tackles the approximation X<sup>⋆</sup> <sup>T</sup> ≈ <sup>E</sup>Q<sup>T</sup> <sup>|</sup><sup>t</sup> <sup>107</sup> [X<sup>T</sup> |X<sup>t</sup> = x] by learning a drift function. This learned drift guides the evolution of X<sup>⋆</sup> t <sup>108</sup> such that its terminal <sup>109</sup> state aligns with the target, often by regressing the drift agains a target drift derived from samples of <sup>110</sup> (X0, X<sup>T</sup> ) under the reference conditional bridge measure Q|0,T .

<sup>111</sup> Building upon the projections R and M, Schrodinger Bridge Matching (SBM) methods [ ¨ [37,](#page-11-2) [45\]](#page-11-4) <sup>112</sup> refines the path measure through an alternating iteraive procedure:

$$\mathbb{P}^{(2n+1)} := \mathcal{M}(\mathbb{P}^{(2n)}, \mathcal{T}), \quad \mathbb{P}^{(2n+2)} := \mathcal{R}(\mathbb{P}^{(2n+1)}, \mathcal{T}). \quad (7)$$

Initialized with P (0) = <sup>P</sup> (0) <sup>T</sup> Q|0,T , utilizing <sup>P</sup> (0) T <sup>113</sup> is independent coupling of ρ<sup>0</sup> and ρ<sup>T</sup> along with the <sup>114</sup> reference conditional bridge measure Q|T . Please refer to [\[37,](#page-11-2) [45\]](#page-11-4) for more details.

# <sup>115</sup> 3 Multi-Marginal Iterative Markovian Fitting

 Dynamic SB methods, as discussed in Section [2,](#page-1-2) have traditionally focused on problems defined by two endpoint marginal distributions, (ρ0, ρ<sup>T</sup> ). However, in real-world applications, particularly in fields like developmental biology (e.g., scRNA-seq studies of cellular differentiation), systems are often observed through snapshots at multiple intermediate time points, not just at the beginning and end of a process. This prevalence of multi-stage data highlights a critical limitation of standard SB approaches. While the theoretical extension of SB methods to handle multiple marginals has been explored [\[1,](#page-9-7) [31\]](#page-10-11), the development of robust and scalable computational methods for these multi-marginal settings has lagged. Recently, methods with IPF-type objectives have been derived for multi-marginal cases [\[8,](#page-9-6) [44\]](#page-11-6). However, challenges persist in ensuring global dynamic consistency across all intervals, maintaining computational tractability as the number of marginals increases.

 In this section, we extends the SBM framework−conventionally applied to problems with two endpoint marginals (ρ0, ρ<sup>T</sup> ) and foundational to IMF methods−to handle cases involving k + 1 multiple snapshots (ρ0, ρ<sup>t</sup><sup>1</sup> , · · · , ρ<sup>T</sup> ) on discrete time stamps T = {t0, t1, · · · , tk} where 0 = t<sup>0</sup> < t<sup>1</sup> < · · · < t<sup>k</sup> = T . Similar to [SBP](#page-1-0), the dynamic multi-marginal Schrodinger Bridge problem can ¨ be formally defined as [\[10\]](#page-9-8) the entropy minimization problem:

$$\min_{\mathbb{P} \in \mathcal{P}_{[0,T]}} D_{\text{KL}}(\mathbb{P}|\mathbb{Q}), \quad \text{subject to} \quad \mathbb{P}_t \sim \rho_t, \quad \forall t \in \mathcal{T}. \quad (\text{mSBP})$$

To find a most probable path P mSBP <sup>131</sup> , the solution of [mSBP](#page-3-1) under multiple constraints, we will generalize <sup>132</sup> the principles of SBM in Section [2.1](#page-1-3) to the multi-marginal cases in Section [3.1.](#page-3-2) The extension of <sup>133</sup> dynamic SB optimality [\[20,](#page-10-1) [35\]](#page-11-8) and the associated stochastic optimal control problem [\[39\]](#page-11-9) to multi-<sup>134</sup> marginal settings is presented in Appendix A.

#### <sup>135</sup> 3.1 Multi-Marginal Projection operators

 To develop multi-marginal extension of SBM, we investigate how the IMF framework can be adapted to scenarios with multiple snapshots (*i*.*e*., where the set of time points T has cardinality |T | > 2). This adaptation necessitates extending the fundamental building blocks of SBM—specifically, the reciprocal projection R and the Markov projection M—to handle multiple marginal constraints.

Multi-Marginal Reciprocal Projection Rmm <sup>140</sup> . First, we state and prove a proposition that character-<sup>141</sup> izes the reciprocal structure of conditional path measures. In particular, we focus on a mixture of <sup>142</sup> bridges Π = Π<sup>T</sup> Q|T ∈ <sup>P</sup>[0,T] constrained by the marginals at multiple timestamps in T .

Proposition 1 (Reciprocal Property). *For any* x<sup>T</sup> := (x0, x<sup>t</sup><sup>1</sup> , · · · , x<sup>T</sup> ) ∈ <sup>R</sup> <sup>d</sup>×(k+1) <sup>143</sup> *and* t ∈ <sup>144</sup> [ti−1, ti)*, the marginal distribution of* Q|T (·|x<sup>T</sup> ) *at* t *satisfies:*

$$\mathbb{Q}|\mathcal{T}(\mathbf{x}_t|\mathbf{x}_{\mathcal{T}}) = \mathbb{Q}|_{t_{i-1}, t_i}(\mathbf{x}_t|\mathbf{x}_{t_i}, \mathbf{x}_{t_{i-1}}). \quad (8)$$

*Therefore, for any* <sup>P</sup> ∈ P[0,T] *the reciprocal projection* Rmm <sup>145</sup> (P, T ) *admits the following factorization:*

$$\mathcal{R}^{mn}(\mathbb{P}, \mathcal{T}) = \mathbb{P}_{\mathcal{T}} \mathbb{Q}|_{\mathcal{T}} = \mathbb{P}_{t_0, \dots, t_k} \mathbb{Q}|_{t_0, \dots, t_k} = \mathbb{P}_{t_0, \dots, t_k} \prod_{i=1}^k \mathbb{Q}|_{t_{i-1}, t_i}, \quad \mathbb{P}\text{-}a.e. \quad (9)$$

 A key implication of the reciprocal property, detailed in Proposition [1,](#page-3-3) is that a mixture of diffusion bridges constrained on T factorizes into independent segments over successive time intervals. This factorization simplifies the analysis and simulation of the overall path measure. Since each segment can then be treated as a standard conditional bridge process as in [\(3\)](#page-2-2), closed-form sampling, such as in [\(4\)](#page-2-1), can be applied independently in parallel to each subinterval {ti−1, ti}i∈[1:k] . This tractability is essential for developing an efficient multi-marginal SBM algorithm.

Multi-Marginal Markov Projection Mmm <sup>152</sup> . With the reciprocal property and factorization in [\(9\)](#page-3-4), <sup>153</sup> we show that the Markov projection on multi-marginal case can be constructed by similar fashion.

<sup>154</sup> Proposition 2 (Multi-Marginal Markovian Projection). *Let* Π ∈ P[0,T] *admit factorzation in* [\(9\)](#page-3-4)*. The multi-marginal Markov projection of* Π*,* P ⋆ := Mmm(Π, T ) ∈ P[0,T] <sup>155</sup> *, is associated with the SDE:*

$$d\mathbf{X}_t^* = [f_t(\mathbf{X}_t^*) + \sigma v^*(t, \mathbf{X}_t^*)] dt + \sigma d\mathbf{W}_t, \quad \mathbf{X}_0^* \sim \Pi_0, \quad (10)$$

$$\text{where } v^*(t, \mathbf{x}) = \sum_{i=1}^k \mathbf{1}_{[t_{i-1}, t_i]} \mathbb{E}_{\Pi_{t_i|t_i}} [\nabla \log \mathbb{Q}_{t_i|t}(\mathbf{X}_{t_i}|\mathbf{X}_t)|\mathbf{X}_t = \mathbf{x}]. \quad (11)$$

*Moreover,* v ⋆ <sup>156</sup> *satisfies the Fokker-Planck equation (FPE) [\[40\]](#page-11-10):*

$$\partial_t \rho_t = -\nabla \cdot (v_t^*(\mathbf{x}) \rho_t(\mathbf{x})) + \frac{\sigma^2}{2} \Delta \rho_t(\mathbf{x}) = 0, \quad \rho_t = \Pi_t, \quad \forall t \in \mathcal{T}, \quad (12)$$

*where* p<sup>t</sup> *is marginal density of* Πt*. In other words,* <sup>P</sup> ⋆ <sup>157</sup> <sup>t</sup> = Π<sup>t</sup> *for all* t ∈ [0, T]*. d*

<sup>158</sup> As established in Proposition [2,](#page-3-5) constructing a global diffusion process via [\(10\)](#page-3-6) with the optimal control v ⋆ [\(11\)](#page-3-7)) yields a multi-marginal Markov projection X<sup>⋆</sup> [0,T] <sup>159</sup> that is continuous over the entire time interval [0, T]. The continuity arises because the local Markov projections, X<sup>⋆</sup> [ti−1,ti] <sup>160</sup> , on each sub-interval are derived from factorized conditional bridge Q|ti−1,t<sup>i</sup> <sup>161</sup> in [\(9\)](#page-3-4). These bridges are

<sup>1</sup>Our framework accommodates arbitrary time intervals between successive time stamps.

anchored by identical marginal distributions at there shared boundaries; for instance, both X<sup>⋆</sup> [ti−1,ti] 162 and X<sup>⋆</sup> [ti,ti+1] is guaranteed to match the marginal distribution ρ<sup>t</sup><sup>i</sup> at time t<sup>i</sup> <sup>163</sup> . Consequently, these local <sup>164</sup> diffusion processes connect seamlessly at adjacent timestamps, resulting in a smooth and well-defined path for X<sup>⋆</sup> [0,T] . The well-defined nature of the global path, in conjunction with the projections Rmm <sup>165</sup> and Mmm <sup>166</sup> , is fundamental to successfully applying the SBM framework to the [mSBP](#page-3-1). Finally, the <sup>167</sup> uniquness condition for standard SB [\[45,](#page-11-4) Proposition 5] can also be extended to multi-marginal case. ⋆

Proposition 3 (Uniqueness). *Let* P <sup>168</sup> *be a Markov measure which is reciprocal class of* Q *satisfying* P ⋆ <sup>t</sup> = ρ<sup>t</sup> *for all* t ∈ T *. Then,* <sup>P</sup> ⋆ *is unique solution* P mSBP <sup>169</sup> *of the* [mSBP](#page-3-1)*.*

Building on the projection operators Rmm <sup>170</sup> ,Mmm with the uniquness result of Proposition [3,](#page-4-0) we can <sup>171</sup> apply the iterative algorithm used in SBM algorithm [\[45,](#page-11-4) Algorithm 1] to the multi-marginal setting:

$$\mathbb{P}^{(2n+1)} := \mathcal{M}^{\text{mm}}(\mathbb{P}^{(2n)}, \mathcal{T}), \quad \mathbb{P}^{(2n+2)} := \mathcal{R}^{\text{mm}}(\mathbb{P}^{(2n+1)}, \mathcal{T}), \quad |\mathcal{T}| > 2. \quad (13)$$

<sup>172</sup> The convergence guarantees proved for the iteration apply equally well to the multi-marginal case.

Proposition 4 (Convergence). P (n) = <sup>P</sup> mSBP <sup>173</sup> *of* [mSBP](#page-3-1) *as* n ↑ ∞ *with iterative procedure in* [\(13\)](#page-4-1)*.*

#### <sup>174</sup> 3.2 Practical Implementation.

In practice, at each iteration n of [\(13\)](#page-4-1) we approximate the optimal control v ⋆ <sup>175</sup> from [\(11\)](#page-3-7) by a neural <sup>176</sup> network vθ. By Girsanov theorem, θ are chosen to minimize the following training objective function:

$$\mathcal{L}(\theta, \mathcal{T}, \Pi_{\mathcal{T}}) = \int_0^T \mathbb{E}_{\Pi_{\mathcal{T}}, \tau} [||\sigma \nabla \log \mathbb{Q}_{\beta_{\mathcal{T}}(t)}|_t (\mathbf{X}_{\beta_{\mathcal{T}}(t)} | \mathbf{X}_t) - v_\theta(t, \mathbf{X}_t)||^2 dt], \quad (14)$$

<sup>177</sup> where β<sup>T</sup> (t) = minu{u > t|t ∈ T } ∈ [0, T] is the most recent time point in T after time t. With <sup>178</sup> this notation, the SBM can be generalized to the case of multi-marginal constraints. For example, <sup>179</sup> when T = {0, T} then [\(14\)](#page-4-2) reduces to the objective function described in [\[45\]](#page-11-4).

The learned Markov control v<sup>θ</sup> <sup>⋆</sup> (t, xt) then ensures <sup>P</sup> θ ⋆ <sup>180</sup> <sup>t</sup> = Π<sup>t</sup> for all t ∈ [0, T]. Moreover, prior <sup>181</sup> SBM algorithms interleave forward and backward-time Markov projections to re-anchor the terminal distribution and prevent bias between P (n) T <sup>182</sup> and Π<sup>T</sup> accumulate for each n ∈ <sup>N</sup>. In the multi-marginal <sup>183</sup> setting, we again build the backward-time Markov projection as in Proposition [2](#page-3-5) by *gluing* the local bridge reversals, so that P ⋆ <sup>184</sup> is governed by both SDEs [\(10\)](#page-3-6) and the corresponding backward dynamics:

$$d\mathbf{Y}_t^* = [-f_{T-t}(\mathbf{Y}_t^*) + \sigma u^*(t, \mathbf{Y}_t^*)] dt + \sigma d\mathbf{W}_t, \quad \mathbf{Y}_0^* \sim \Pi_T, \quad (15)$$

where 
$$u^*(t, \mathbf{y}) = \sum_{i=1}^k \mathbf{1}_{(t_{i-1}, t_i]}(t) \mathbb{E}_{\Pi_{t|t_{i-1}}} [\nabla \log \mathbb{Q}_{t|t_{i-1}}(\mathbf{Y}_t | \mathbf{Y}_{t_{i-1}}) | \mathbf{Y}_t = \mathbf{y}]$$
, (16)

where the backward optimal control u ⋆ <sup>185</sup> in [\(16\)](#page-4-3) can be approximated with neural network u<sup>ϕ</sup> where ϕ <sup>186</sup> is chosen to minimize the following training objective function with γ<sup>T</sup> (t) = maxu{u < t|t ∈ T }:

$$\mathcal{L}(\phi, \mathcal{T}, \Pi_{\mathcal{T}}) = \int_0^T \mathbb{E}_{\Pi_t, \mathcal{T}} [||\sigma \nabla \log \mathbb{Q}_{t|\gamma_{\mathcal{T}}(t)}(\mathbf{Y}_t | \mathbf{Y}_{\gamma_{\mathcal{T}}(t)}) - u_{\phi}(t, \mathbf{Y}_t)||^2 dt]. \quad (17)$$

# <sup>187</sup> 4 Multi-Marginal Schrodinger Bridge Matching ¨

A na¨ıve extension of the standard SBM using, multi-marginal projections Rmm and Mmm <sup>188</sup> in Sec [3,](#page-2-3) <sup>189</sup> encounters significant limitations not present in the traditional two-endpoint setting. In such an <sup>190</sup> extension, each iteration typically enforces marginal constraints only at the global endpoints (ρ0, ρ<sup>T</sup> ). The multi-marginal coupling Π (n) T <sup>191</sup> at each iteration n of [\(13\)](#page-4-1) is then derived by propagating the <sup>192</sup> projected dynamics in [\(10\)](#page-3-6) or [\(15\)](#page-4-4) solely from these end points ρ<sup>0</sup> or ρ<sup>T</sup> , respectively.

<sup>193</sup> This approach leads to critical issues specific to the multi-marginal context. Firstly, if the learned controls, such as v ⋆ (forward) or u ⋆ <sup>194</sup> (backward), are even slightly inaccurate, significant biases can arise between the inferred intermediate marginals (Π(n) t1 , · · · Π (n) tk−<sup>1</sup> <sup>195</sup> ) and the target marginals (ρ<sup>t</sup><sup>1</sup> , · · · , ρ<sup>t</sup>k−<sup>1</sup> <sup>196</sup> ). Secondly, these discrepancies tend to accumulate iteratively. This accumulation is exacerbated because, beyond an initialization Π(0) = <sup>P</sup> (0) <sup>T</sup> Q|T with <sup>P</sup> (0) T <sup>197</sup> , independent joint coupling <sup>198</sup> of {ρt}t∈T , where the joint distribution might be informed by all prescribed data distributions, <sup>199</sup> the subsequent self-refinement process for the dynamics often does not directly incorporate the

Algorithm 1 Training of MSBM

- 1: Input: Snapshots {ρt}t∈T , bridge Q|T , N ∈ <sup>N</sup> 2: Let {<sup>P</sup>
- (0) T<sup>i</sup> }i∈[1:k] joint coupling of {ρt∈T<sup>i</sup> }i∈[1:k] . 3: for n ∈ {0, . . . , N − 1} do 4: for i ∈ {1, . . . , k − 1} do in parallel 5: Let Π (2n) T<sup>i</sup> = P (2n) T<sup>i</sup> 6: Estimate L(ϕ, Ti, Π (2n) T<sup>i</sup> , Q|T<sup>i</sup> ) 7: Estimate L˜(ϕ) = P<sup>k</sup> <sup>i</sup>=1 L(ϕ, Ti, Π (2n) T<sup>i</sup> , Q|T<sup>i</sup> ) 8: uϕ<sup>⋆</sup> = arg min<sup>ϕ</sup> P<sup>k</sup> <sup>i</sup>=1L˜(ϕ) 9: Simulate local backward SBs {<sup>P</sup> i,(2n+1)}i∈[1:k] 10: for i ∈ {1, . . . , k − 1} do in parallel 11: Let Π (2n+1) T<sup>i</sup> = P (2n+1) T<sup>i</sup> 12: Estimate L(θ, Ti, Π (2n+1) T<sup>i</sup> , Q|T<sup>i</sup> ) 13: Estimate L˜(θ) = P<sup>k</sup> <sup>i</sup>=1 L(θ, Ti, Π (2n+1) T<sup>i</sup> , Q|T<sup>i</sup> ) 14: vθ<sup>⋆</sup> = arg min<sup>θ</sup> P<sup>k</sup> <sup>i</sup>=1L(θ, Ti, Π (2n+1) T<sup>i</sup> ) 15: Simulate local forward SBs {<sup>P</sup> i,(2n+2) [ti−1,ti] } 16: end for 17: Output: v ⋆ θ , u<sup>⋆</sup> ϕ

Algorithm 2 Simulation of MSBM (forward)

Input: Initial ρ0, learned control vθ<sup>⋆</sup> Sample X<sup>0</sup> ∼ ρ<sup>0</sup> Simulate forward SDE over [0, T] dX<sup>⋆</sup> <sup>t</sup> = [f<sup>t</sup> + σvθ<sup>⋆</sup> (t, X<sup>⋆</sup> <sup>t</sup> )] dt + σdWt, Output: Trajectory X<sup>⋆</sup> [0,T ]

![](_page_5_Figure_4.jpeg)

Figure 1: (Left) The na¨ıve extension fails to model intermediate states due to the accumulation of errors. (Right) In contrast, MSBM successfully models the ground truth data.

intermediate data distributions (ρ<sup>t</sup><sup>1</sup> , · · · , ρ<sup>t</sup>k−<sup>1</sup> ) into its training objective except ρ<sup>0</sup> and ρ<sup>T</sup> . Without explicit targets for the intermediate marginals guiding each iteration, the inferred paths between ρ<sup>0</sup> and ρ<sup>T</sup> can "collapse" or drift away from the desired states. Consequently, precisely satisfying all intermediate constraints becomes increasingly challenging as iterations proceed.

 To address this issue of error accumulation and ensure all marginal constraints {ρt}t∈T are satisfied, we propose a method that involves constructing local SBs on each interval [ti−1, t<sup>i</sup> <sup>205</sup> ] and then seamlessly *gluing* them together. Instead of propagating dynamics from the global endpoints ρ<sup>0</sup> and ρ<sup>T</sup> alone, our approach first establishes local SBs for each segment. The resulting local couplings are then systematically integrated to satisfy all specified marginal distributions {ρt}t∈T across the entire time interval [0, T]. This local construction strategy helps prevent the compounding of errors at intermediate time points while still aiming to achieve the overall multi-marginal SB solution, P mSBP <sup>210</sup> . The theoretical basis is provided by the following result.

Corollary 5 (Multi-Marginal Schrodinger Bridge) ¨ . *Assume a sequence of controls* {v i , u<sup>i</sup>}i∈[1:k] <sup>212</sup> *, where each* v i , u<sup>i</sup> *induced local SBs* P <sup>i</sup> *of* [SBP](#page-1-0) *over local interval* [ti−1, t<sup>i</sup> <sup>213</sup> ] *with distributions* (ρ<sup>t</sup>i−<sup>1</sup> , ρ<sup>t</sup><sup>i</sup> ) *in a forward and backward direction, respectively. If* limt↑t<sup>i</sup> v i (t, x) = v <sup>i</sup>+1 <sup>214</sup> (t, x) *and* limt↓ti−<sup>1</sup> u i (t, x) = u i−1 (t, x) *for all* i ∈ [1 : k]*, then* <sup>P</sup> mSBP <sup>215</sup> *of* [mSBP](#page-3-1) *induced by following SDEs:*

$$d\mathbf{X}_t^* = [f_t(\mathbf{X}_t^*) + \sigma v^*(t, \mathbf{X}_t^*)] dt + \sigma d\mathbf{W}_t, \quad \mathbf{X}_0^* \sim \rho_0. \quad (18a)$$

$$d\mathbf{Y}_t^* = [-f_{T-t}(\mathbf{Y}_t^*) + \sigma u^*(t, \mathbf{Y}_t^*)] dt + \sigma d\mathbf{W}_t, \quad \mathbf{Y}_0^* \sim \rho_T, \quad (18b)$$

$$\text{where } v^*(t, \mathbf{x}) = \sum_{i=1}^k \mathbf{1}_{[t_{i-1}, t_i)}(t) v^i(t, \mathbf{x}), \quad u^*(t, \mathbf{x}) = \sum_{i=1}^k \mathbf{1}_{(t_{i-1}, t_i]}(t) u^i(t, \mathbf{x}). \quad (18\text{c})$$

<sup>216</sup> Building upon Corollary [5,](#page-5-0) we introduce our Multi-Marginal Schrodinger Bridge Matching (MSBM) ¨ <sup>217</sup> method to solve the [mSBP](#page-3-1). A cornerstone of MSBM is divide the global [mSBP](#page-3-1) into local [SBP](#page-1-0)s while maintaining the continuity of the composite drift functions v ⋆ and u ⋆ <sup>218</sup> in [\(18c\)](#page-5-1) across adjacent intervals, which guarantees a globally continuous diffusion process inducing P mSBP <sup>219</sup> . Furthermore, by explicitly constraining each local SBs, P i , on its corresponding marginals (ρ<sup>t</sup>i−<sup>1</sup> , ρ<sup>t</sup><sup>i</sup> <sup>220</sup> ), MSBM is designed to <sup>221</sup> mitigate the accumulation of bias at intermediate marginals, as shown in Figure [1.](#page-5-2)

<sup>222</sup> A key challenge of the MSBM is rigorously satisfying the continuity conditions at the boundaries of local controls: limt↑t<sup>i</sup> v i (t, x) = v <sup>i</sup>+1(t, x) and limt↓ti−<sup>1</sup> u i (t, x) = u i−1 <sup>223</sup> (t, x) for all i ∈ [1 : k]. If <sup>224</sup> these conditions are not met, discontinuities or "kinks" can arise at the intermediate time steps. Such kinks would imply that the overall path measure P ⋆ ̸= Mmm(<sup>P</sup> ⋆ <sup>225</sup> , T ). This would, in turn, hinder the <sup>226</sup> optimlaity for [mSBP](#page-3-1), because, following Proposition [3,](#page-4-0) the desired continuous Markov process is a fixed point of both Rmm and Markov projections Mmm <sup>227</sup> under multiple time points T :

$$\mathbb{P}^* = \mathcal{R}^{\text{mn}}(\mathbb{P}^*, \mathcal{T}) = \mathcal{M}^{\text{mn}}(\mathbb{P}^*, \mathcal{T}). \quad (19)$$

 To construct local SBs such that the continuity requirements for forming a valid global solution are met, thereby preventing the aforementioned kinks and ensuring [\(19\)](#page-5-3), our MSBM introduces a shared global parametrization vθ, u<sup>ϕ</sup> for its respective local controls {v i , u<sup>i</sup>}i∈[1:k] for each sub-interval, where each local controls are parallel updated with following aggregate objective function:

$$\tilde{\mathcal{L}}(\theta) = \sum_{i=1}^k \mathcal{L}(\theta, \mathcal{T}_i, \Pi_{\mathcal{T}_i}), \quad \tilde{\mathcal{L}}(\phi) = \sum_{i=1}^k \mathcal{L}(\phi, \mathcal{T}_i, \Pi_{\mathcal{T}_i}), \quad (20a)$$

where T<sup>i</sup> = {ti−1, ti} define sub-intervals with local coupling ΠT<sup>i</sup> for end-points marginals in interval [ti−1, t<sup>i</sup> ] and L is defined in [\(14\)](#page-4-2) and [\(17\)](#page-4-5) for forward and backward direction, respectively.

 The MSBM training procedure, summarized in Algorithm [1,](#page-5-2) adapts the standard IMF algorithm presented in [\[45,](#page-11-4) Algorithm 1]. A key distinction in our MSBM approach is the parallel application of the IMF procedure to each local time interval, utilizing globally shared forward v<sup>θ</sup> and backward u<sup>ϕ</sup> across all local intervals. This parallel processing across sub-intervals contributes to a significant reduction in overall training time.

# 5 Related Work

 The solution of [SBP](#page-1-0) often utilize Iterative Proportional Fitting (IPF) [\[19\]](#page-10-12), with modern adaptations learning SDE drifts for two-marginal settings [\[4,](#page-9-2) [9,](#page-9-3) [13,](#page-9-9) [49\]](#page-11-5). A distinct iterative approach, IMF, as featured in [\[37,](#page-11-2) [45\]](#page-11-4), offers improved stability by alternating projections onto different classes of path measures. Moreover, emerging research also explores non-iterative algorithm [\[12,](#page-9-10) [38\]](#page-11-3). These methodologies primarily concentrate on the SB problem itself, iteratively refining path measures or directly computing the bridge measure. Moreover, the SB algorithm is studied under the assumption that the optimal coupling is given [\[27,](#page-10-4) [46\]](#page-11-11). While recent studies have extended foundational SB ideas to the multi-marginal setting of [mSBP](#page-3-1), research in this area remains relatively limited.

 In multi-marginal setting, [\[8\]](#page-9-6) extends the problem to phase space to encourage smoother trajectories and introduces a novel training methodology inspired by the Bregman iteration [\[5\]](#page-9-11) to handle multiple marginal constraints. Relatedly, [\[44\]](#page-11-6) presented an approach that, similar to our work, segments the problem across intervals; they learn piecewise SBs and use likelihood-based training to iteratively refine a global reference dynamic. While these methods are often IPF-based or focus on specific reference refinement strategies, our MSBM extends the previous IMF-type algorithm into multi- marginal setting and effectively handles multiple constraints. We demonstrate that our MSBM framework offers substantial gains in training efficiency. This enhanced efficiency is primarily attributed to its direct multi-marginal formulation that adeptly manages multiple constraints, thereby circumventing the computationally intensive iterative refinements common in IPF-based methods

 Paralleling these SB-centric developments, other significant lines of work model dynamic trajectories by directly learning potential functions or velocity fields, often drawing from optimal transport or continuous normalizing flows. For instance, [\[18,](#page-10-8) [24](#page-10-13)[–26\]](#page-10-3) extend SBs to incorporate potentials or mean-field interactions, connecting to stochastic optimal control and earlier mean-field game frameworks [\[22,](#page-10-14) [41\]](#page-11-12). The broader field of trajectory inference from snapshot data, crucial for applications like scRNA-seq, has seen methods like [\[48\]](#page-11-13) using CNFs with dynamic OT, and [\[15\]](#page-9-12) employing Neural ODEs on learned data manifolds. More recently, [\[33,](#page-11-14) [34\]](#page-11-15) offer variational objectives to learn dynamics from marginal samples.

# 6 Experiments

 In this section, we empirically demonstrate the effectiveness of our MSBM. Specifically, our goal is to infer a dynamic model from datasets composed of samples from marginal distributions ρ<sup>t</sup> observed at discrete time points. We evaluate MSBM on both synthetic datasets and real-world single- cell RNA sequencing datasets, including human embryonic stem cells (hESC) [\[11\]](#page-9-5) and embryoid body (EB) [\[32\]](#page-10-7). To ensure consistency and fair comparison, our experiments follow the respective experimental setups established by baseline methods. In particular, for the petal dataset, we adopt the experimental setup from DMSB [\[8\]](#page-9-6), and for the hESC dataset, we follow SBIRR [\[44\]](#page-11-6). For the EB dataset, we perform evaluations on both 5-dim and 100-dim PCA-reduced data; here, we follow the 100-dim experimental setup of DMSB and the 5-dim setup from NLSB [\[18\]](#page-10-8). Accordingly,

![](_page_7_Figure_0.jpeg)

![](_page_7_Diagram_1.jpeg)

Figure 3: Comparison of generated population dynamics using MIOFlow, DMSB and MSBM on a 2-dim petal dataset. All trajectories are generated by simulating the dynamics from ρ<sup>t</sup><sup>0</sup> .

 we utilize evaluation metrics consistent with previous studies, including the Sliced-Wasserstein Distance (SWD)[\[2\]](#page-9-13), Maximum Mean Discrepancy (MMD)[\[14\]](#page-9-14), as well as the 1-Wasserstein (W1) and 2-Wasserstein (W2) distances. All experimental results reported are averaged mean value over three independent runs with different random seeds. We highlight the best-performing results in bold and the second-best results in blue. Further experimental details are provided in Appendix C.

#### <sup>281</sup> 6.1 Synthetic Data

t0 t1 t2 t3 t4 Time 0.1 0.2 0.3 0.4 DMSB MIOFlow MSBM t0 t1 t2 t3 t4 Time 0.00 0.02 0.04 0.06 0.08 0.10 Figure 2: Evaluation results of W<sup>2</sup> and MMD. <sup>282</sup> Petal The petal dataset [\[15\]](#page-9-12) serves as a sim-<sup>283</sup> ple yet complex challenge because it mimics <sup>284</sup> the natural dynamics seen in processes such as <sup>285</sup> cellular differentiation, which include phenom-<sup>286</sup> ena like bifurcations and merges. We compare <sup>287</sup> our MSBM with MIOFlow [\[15\]](#page-9-12) and DMSB [\[8\]](#page-9-6) <sup>288</sup> in Figure [2.](#page-7-0) As shown in Figure [3,](#page-7-1) we ob-<sup>289</sup> serve that MSBM exhibits the most accurate and <sup>290</sup> clearly defined trajectory, closely resembling the <sup>291</sup> ground truth. Furthermore, Figure [2](#page-7-0) demonstrates the evaluation results for the trajectories through <sup>292</sup> W<sup>2</sup> and MMD distances, highlighting that MSBM consistently outperforms MIOFlow and DMSB.

![](_page_7_Figure_6.jpeg)

#### <sup>293</sup> 6.2 Single-cell Sequencing Data

 We evaluated our MSBM on real-world single-cell RNA sequencing data from two sources: 1) human embryonic stem cells (hESCs) [\[11\]](#page-9-5) undergoing differentiation into definitive endoderm over a 4-day period, measured at 6 distinct time points (t0:0 hours, t1:12 hours, t2:24 hours, t3:36 hours, t4:72 hours, and t5:96 hours); 2) embryoid body (EB) cells [\[32\]](#page-10-7) differentiating into mesoderm, endoderm, neuroectoderm, and neural crest over 27 days, with samples collected at 5 time windows (t0:0-3 days, t1:6-9 days, t2:12-15 days, t3:18-21 days, and t4:24-27 days). Following the experimental setup of baselines, we preprocessed these datasets using the pipeline outlined in [\[48\]](#page-11-13), and the collected cells were projected into a lower-dimensional space using principal component analysis (PCA).

> Table 1: Performance on the 5 dim PCA of hESC dataset. W<sup>2</sup> is compute between test ρ<sup>t</sup><sup>i</sup> and generated ρˆ<sup>t</sup><sup>i</sup> by simulating the dynamics from test ρ<sup>t</sup><sup>0</sup> .

| Methods TrajectoryNet † | W t 1 | 2 ↓ t 3 | Runtime hours |
|-------------------------|-------|---------|---------------|
|                         | 1.30  | 1.93    | 10.19         |
| DMSB †                  |       |         |               |
|                         | 1.10  | 1.51    | 15.54         |
| SBIRR †                 |       |         |               |
|                         | 1.08  | 1.33    | 0.36 (0.38) ∗ |
| MSBM (Ours)             | 1.09  | 1.30    | 0.09          |

MSBM (Ours) 1.09 1.30 0.09 † result from [\[44\]](#page-11-6). <sup>311</sup> Embryoid Body We validate our MSBM on both 5-dim and <sup>312</sup> 100-dim PCA spaces. First, for the 5-dim experiment, we adopt the <sup>313</sup> experimental setup from NLSB. Given 5 observation time points T = {t0, t1, t2, t3, t4}, we divide the data using train/test splits ρ tr T /ρte T <sup>314</sup> , with the goal of predicting population-level dynamics from ρ tr t0 . Similar to NLSB, we train the dynamics based on ρ tr T <sup>315</sup> and

 hESC To follow the experimental setup from SBIRR [\[44\]](#page-11-6), we reduced the data to the first five principal components and excluded the final time point t<sup>6</sup> from our dataset, resulting in three train- ing time points T = {t0, t2, t4} and two intermediate test points Ttest = {t1, t3}. Our objective was to train the dynamics based on the available marginals at the training points in T and interpolate the intermediate test marginals at Ttest, which were not observed during training. Table [1](#page-7-2) demonstrates that our proposed MSBM method performs competitively, achieving lower W<sup>2</sup> distances.

Table 3: Performance on the 100-dim PCA of EB dataset. MMD and SWD are computed between test ρ te ti and generated ρˆ<sup>t</sup><sup>i</sup> by simulating the dynamics from test ρ te t0 . Figure 4: Comparison of generated population dynamics using DMSB and MSBM on a 100-dim PCA of EB dataset. The plot displays the first two principal components as the x and y axes, respectively.

| NLSB †    |   | Full |   | t 1 | MMD | ↓ t 2 |   | t 3 |   | Full |   | t 1 | SWD | ↓ t 2 |   | t 3 |
|-----------|---|------|---|-----|-----|-------|---|-----|---|------|---|-----|-----|-------|---|-----|
| [18]      | 0 | 66   | 0 | 38  | 0   | 37    | 0 | 37  | 0 | 54   | 0 | 55  | 0   | 54    | 0 | 55  |
| MIOFlow † |   |      |   |     |     |       |   |     |   |      |   |     |     |       |   |     |
| [15]      | 0 | 23   | 0 | 23  | 0   | 90    | 0 | 23  | 0 | 35   | 0 | 49  | 0   | 72    | 0 | 50  |
| DMSB †    |   |      |   |     |     |       |   |     |   |      |   |     |     |       |   |     |
| [8]       | 0 | 03   | 0 | 04  | 0   | 04    | 0 | 04  | 0 | 16   | 0 | 20  | 0   | 19    | 0 | 18  |
| MSBM      | 0 | 02   | 0 | 04  | 0   | 04    | 0 | 05  | 0 | 11   | 0 | 18  | 0   | 17    | 0 | 19  |

† result from [\[8\]](#page-9-6). evaluate the W<sup>1</sup> distance between ρ te ti and the generated ρˆ<sup>t</sup><sup>i</sup> from previous test snapshot ρ te ti−<sup>1</sup> <sup>316</sup> . <sup>317</sup> In Table [2,](#page-8-0) we find that MSBM outperforms several SB methods.

DMSB MSBM Groud Truth

t0 t1 t2 t3 t4 traj

Table 2: Performance on the 5-dim PCA of EB dataset. W<sup>1</sup> is computed between test ρ te ti and generated ρˆ<sup>t</sup><sup>i</sup> by simulating the dynamics from previous test ρ te ti−<sup>1</sup> .

| Methods Neural SDE † |   | t 1 |   | t 2 | W | 1 t 3 | ↓ | t 4 |   | Mean |
|----------------------|---|-----|---|-----|---|-------|---|-----|---|------|
| [21]                 | 0 | 69  | 0 | 91  | 0 | 85    | 0 | 81  | 0 | 82   |
| TrajectoryNet †      |   |     |   |     |   |       |   |     |   |      |
| [48]                 | 0 | 73  | 1 | 06  | 0 | 90    | 1 | 01  | 0 | 93   |
| IPF (GP) †           |   |     |   |     |   |       |   |     |   |      |
| [49]                 | 0 | 70  | 1 | 04  | 0 | 94    | 0 | 98  | 0 | 92   |
| IPF (NN) †           |   |     |   |     |   |       |   |     |   |      |
| [4]                  | 0 | 73  | 0 | 89  | 0 | 84    | 0 | 83  | 0 | 82   |
| SB-FBSDE †           |   |     |   |     |   |       |   |     |   |      |
| [9]                  | 0 | 56  | 0 | 80  | 1 | 00    | 1 | 00  | 0 | 84   |
| NLSB †               |   |     |   |     |   |       |   |     |   |      |
| [18]                 | 0 | 68  | 0 | 84  | 0 | 81    | 0 | 79  | 0 | 78   |
| OT-CFM †             |   |     |   |     |   |       |   |     |   |      |
| [47]                 | 0 | 78  | 0 | 76  | 0 | 77    | 0 | 75  | 0 | 77   |
| WLF-SB ‡             |   |     |   |     |   |       |   |     |   |      |
| [34]                 | 0 | 63  | 0 | 79  | 0 | 77    | 0 | 75  | 0 | 73   |
| MSBM (Ours)          | 0 | 64  | 0 | 73  | 0 | 72    | 0 | 73  | 0 | 71   |

† result from [\[18\]](#page-10-8), ‡ result from [\[34\]](#page-11-15). Computational Efficiency For an fair comparison of training efficiency against recent multi-marginal SB al- gorithms, we benchmarked DMSB and SBIRR on the identical hardware configuration employed for MSBM (denoted by <sup>∗</sup> in Table [1\)](#page-7-2). On the hESC dataset, MSBM achieved a runtime over 4× faster than SBIRR. Furthermore, on the petal and 100-dim PCA of EB dataset, MSBM significantly outperformed DSMB in training speed, with detailed results presented in Figure [5.](#page-8-2)

 For the 100-dim experiment, we borrow the experimental setup from DMSB, where the goal is predict population dynamics given that observations are available for all time points T (denoted as Full in Table [3\)](#page-8-1), or when one of the snapshot is left out (denoted as t<sup>i</sup> <sup>322</sup> in Table [3,](#page-8-1) where snapshot ρ tr ti at t<sup>i</sup> <sup>323</sup> is excluded during training). The high performance in this task represent the robustness of the model to accurately predict population dynamics. In Ta- ble [3,](#page-8-1) MSBM consistently yields performance improve- ments. Moreover, as shown in Figure [4,](#page-8-1) the trajectories and generated marginal distributions ρˆ<sup>T</sup> in PCA space fur- ther justifies the numerical result and highlights the variety and quality of the samples produced by MSBM.

Figure 5: Training time

 This enhanced computational efficiency primarily originates from core algorithmic differences. SBIRR, for example, utilizes maxi- mum likelihood training, which requires extensive gradient compu- tations and the storage of all intermediate paths. DMSB employs an IPF-type objective with Bregman Iteration [\[5\]](#page-9-11). In contrast, MSBM directly optimizes controls using an IMF-type objective, which not only eliminates the need to store intermediate states but also fa- cilitates parallel computation across sub-intervals. This approach substantially promotes faster convergence of the algorithm.

![](_page_8_Figure_8.jpeg)

# <sup>346</sup> 7 Conclusion and Limitation

 This paper revisits previously established frameworks for the [SBP](#page-1-0), extending them to the [mSBP](#page-3-1). Specifically, we introduce a computationally efficient framework for [mSBP](#page-3-1), termed MSBM, which builds upon existing SBM methods [\[37,](#page-11-2) [45\]](#page-11-4). MSBM is tailored for various trajectory inference problems where snapshots of data are available at multi-marginal time steps. Through the successful adaptation of the IMF algorithm to this multi-marginal setting, our approach significantly accelerates training processes while ensuring accurate dynamic modeling when compared to existing methods.

 Despite these advantages, the performance degradation of MSBM is more pronounced than that of DMSB when a time point is omitted in Table [3.](#page-8-1) This may occur because the including velocity term could better accommodate unknown trajectory. Furthermore, the current MSBM framework is restricted to the case involving snapshot data samples, highlighting a need for enhancements to address problems with continuous potentials, such mean-field games [\[18,](#page-10-8) [24](#page-10-13)[–26\]](#page-10-3).

# References


[1] Aymeric Baradat and Christian Leonard. Minimizing relative entropy of path measures under ´ marginal constraints. *arXiv preprint arXiv:2001.10920*, 2020. [2] Nicolas Bonneel, Julien Rabin, Gabriel Peyre, and Hanspeter Pfister. Sliced and Radon Wasser- ´ stein barycenters of measures. *Journal of Mathematical Imaging and Vision*, 51:22–45, 2015. [3] Valentin De Bortoli, Iryna Korshunova, Andriy Mnih, and Arnaud Doucet. Schrodinger bridge flow for unpaired data translation. In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024. [4] Valentin De Bortoli, James Thornton, Jeremy Heng, and Arnaud Doucet. Diffusion Schrodinger ¨ bridge with applications to score-based generative modeling. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan, editors, *Advances in Neural Information Processing Systems*, 2021. [5] L.M. Bregman. The relaxation method of finding the common point of convex sets and its application to the solution of problems in convex programming. *USSR Computational Mathematics and Mathematical Physics*, 1967. [6] Jason D Buenrostro, Beijing Wu, Ulrike M Litzenburger, Dave Ruff, Michael L Gonzales, Michael P Snyder, Howard Y Chang, and William J Greenleaf. Single-cell chromatin accessi- bility reveals principles of regulatory variation. *Nature*, 523(7561):486–490, 2015. [7] Charlotte Bunne, Stefan G Stark, Gabriele Gut, Jacobo Sarabia Del Castillo, Mitch Levesque, Kjong-Van Lehmann, Lucas Pelkmans, Andreas Krause, and Gunnar Ratsch. Learning single- ¨ cell perturbation responses using neural optimal transport. *Nature methods*, 20(11):1759–1768, 2023. [8] Tianrong Chen, Guan-Horng Liu, Molei Tao, and Evangelos Theodorou. Deep momentum multi-marginal schrodinger bridge. ¨ *Advances in Neural Information Processing Systems*, 36:57058–57086, 2023. [9] Tianrong Chen, Guan-Horng Liu, and Evangelos Theodorou. Likelihood training of schrodinger ¨ bridge using forward-backward SDEs theory. In *International Conference on Learning Repre- sentations*, 2022. [10] Yongxin Chen, Giovanni Conforti, Tryphon T Georgiou, and Luigia Ripani. Multi-marginal schrodinger bridges. In ¨ *International Conference on Geometric Science of Information*, pages 725–732. Springer, 2019. [11] Li-Fang Chu, Ning Leng, Jue Zhang, Zhonggang Hou, Daniel Mamott, David T Vereide, Jeea Choi, Christina Kendziorski, Ron Stewart, and James A Thomson. Single-cell rna-seq reveals novel regulators of human embryonic stem cell differentiation to definitive endoderm. *Genome biology*, 17:1–20, 2016. [12] Valentin De Bortoli, Iryna Korshunova, Andriy Mnih, and Arnaud Doucet. Schrodinger bridge flow for unpaired data translation. *Advances in Neural Information Processing Systems*, 37:103384–103441, 2024. [13] Wei Deng, Weijian Luo, Yixin Tan, Marin Bilos, Yu Chen, Yuriy Nevmyvaka, and Ricky T. Q. ˇ Chen. Variational schrodinger diffusion models. In ¨ *Forty-first International Conference on Machine Learning*, 2024. [14] Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Scholkopf, and Alexander ¨ Smola. A kernel two-sample test. *The Journal of Machine Learning Research*, 13(1):723–773, 2012. [15] Guillaume Huguet, Daniel Sumner Magruder, Alexander Tong, Oluwadamilola Fasina, Manik Kuchroo, Guy Wolf, and Smita Krishnaswamy. Manifold interpolating optimal-transport flows for trajectory inference. *Advances in neural information processing systems*, 35:29705–29718, 2022.

[16] Benton Jamison. The Markov processes of Schrodinger. ¨ *Zeitschrift fur Wahrscheinlichkeitsthe- ¨ orie und verwandte Gebiete*, 32(4):323–331, 1975. [17] Allon M Klein, Linas Mazutis, Ilke Akartuna, Naren Tallapragada, Adrian Veres, Victor Li, Leonid Peshkin, David A Weitz, and Marc W Kirschner. Droplet barcoding for single-cell transcriptomics applied to embryonic stem cells. *Cell*, 161(5):1187–1201, 2015. [18] Takeshi Koshizuka and Issei Sato. Neural Lagrangian Schrodinger bridge: Diffusion modeling ¨ for population dynamics. *arXiv preprint arXiv:2204.04853*, 2022. [19] Solomon Kullback. Probability densities with given marginals. *The Annals of Mathematical Statistics*, 39(4):1236–1243, 1968. [20] Christian Leonard. A survey of the Schr ´ odinger problem and some of its connections with ¨ optimal transport. *arXiv preprint arXiv:1308.0215*, 2013. [21] Xuechen Li, Ting-Kam Leonard Wong, Ricky TQ Chen, and David Duvenaud. Scalable gradi- ents for stochastic differential equations. In *International Conference on Artificial Intelligence and Statistics*, pages 3870–3882. PMLR, 2020. [22] Alex Tong Lin, Samy Wu Fung, Wuchen Li, Levon Nurbekyan, and Stanley J. Osher. Alternating the population and control neural networks to solve high-dimensional stochastic mean-field games. *Proceedings of the National Academy of Sciences*, 2021. [23] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In *The Eleventh International Conference on Learning Representations*, 2023. [24] Guan-Horng Liu, Tianrong Chen, Oswin So, and Evangelos Theodorou. Deep generalized schrodinger bridge. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, ¨ editors, *Advances in Neural Information Processing Systems*, 2022. [25] Guan-Horng Liu, Tianrong Chen, and Evangelos A Theodorou. Deep generalized schr\" odinger bridges: From image generation to solving mean-field games. *arXiv preprint arXiv:2412.20279*, 2024. [26] Guan-Horng Liu, Yaron Lipman, Maximilian Nickel, Brian Karrer, Evangelos Theodorou, and Ricky T. Q. Chen. Generalized schrodinger bridge matching. In ¨ *The Twelfth International Conference on Learning Representations*, 2024. [27] Guan-Horng Liu, Arash Vahdat, De-An Huang, Evangelos A Theodorou, Weili Nie, and Anima Anandkumar. I<sup>2</sup> SB: Image-to-image Schrodinger bridge. ¨ *arXiv preprint arXiv:2302.05872*, 2023. [28] Evan Z Macosko, Anindita Basu, Rahul Satija, James Nemesh, Karthik Shekhar, Melissa Goldman, Itay Tirosh, Allison R Bialas, Nolan Kamitaki, Emily M Martersteck, et al. Highly parallel genome-wide expression profiling of individual cells using nanoliter droplets. *Cell*, 161(5):1202–1214, 2015. [29] Kenneth G Manton, XiLiang Gu, and Gene R Lowrimore. Cohort changes in active life expectancy in the us elderly population: Experience from the 1982–2004 national long-term care survey. *The Journals of Gerontology Series B: Psychological Sciences and Social Sciences*, 63(5):S269–S281, 2008. [30] Toshio Mikami. *Stochastic optimal transportation: stochastic control with fixed marginals*. Springer Nature, 2021. [31] Abdulwahab Mohamed, Alberto Chiarini, and Oliver Tse. Schrodinger bridges with multi- ¨ marginal constraints. 2021. [32] Kevin R Moon, David Van Dijk, Zheng Wang, Scott Gigante, Daniel B Burkhardt, William S Chen, Kristina Yim, Antonia van den Elzen, Matthew J Hirn, Ronald R Coifman, et al. Vi- sualizing structure and transitions in high-dimensional biological data. *Nature biotechnology*, 37(12):1482–1492, 2019.

[33] Kirill Neklyudov, Rob Brekelmans, Daniel Severo, and Alireza Makhzani. Action matching: Learning stochastic dynamics from samples. In *Proceedings of the 40th International Confer- ence on Machine Learning*, volume 202 of *Proceedings of Machine Learning Research*. PMLR, 23–29 Jul 2023. [34] Kirill Neklyudov, Rob Brekelmans, Alexander Tong, Lazar Atanackovic, Qiang Liu, and Alireza Makhzani. A computational framework for solving Wasserstein Lagrangian flows. *arXiv preprint arXiv:2310.10649*, 2023. [35] Michele Pavon and Anton Wakolbinger. On free energy, stochastic control, and Schrodinger ¨ processes. In *Modeling, Estimation and Control of Systems with Uncertainty: Proceedings of a Conference held in Sopron, Hungary, September 1990*, pages 334–348. Springer, 1991. [36] Stefano Peluchetti. Non-denoising forward-time diffusions, 2022. [37] Stefano Peluchetti. Diffusion bridge mixture transports, schrodinger bridge problems and ¨ generative modeling. *Journal of Machine Learning Research*, 24(374):1–51, 2023. [38] Stefano Peluchetti. BM\$ˆ2\$: Coupled schrodinger bridge matching. ¨ *Transactions on Machine Learning Research*, 2025. [39] Paolo Dai Pra. A stochastic control approach to reciprocal diffusion processes. *Applied Mathematics and Optimization*, 23:313–329, 1991. [40] Hannes Risken and Hannes Risken. *Fokker-planck equation*. Springer, 1996. [41] Lars Ruthotto, Stanley J. Osher, Wuchen Li, Levon Nurbekyan, and Samy Wu Fung. A machine learning framework for solving high-dimensional mean field game and mean field control problems. *Proceedings of the National Academy of Sciences*, 2020. [42] Geoffrey Schiebinger, Jian Shu, Marcin Tabaka, Brian Cleary, Vidya Subramanian, Aryeh Solomon, Joshua Gould, Siyan Liu, Stacie Lin, Peter Berube, et al. Optimal-transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming. *Cell*, 176(4):928–943, 2019. [43] Erwin Schrodinger. ¨ *Uber die umkehrung der naturgesetze ¨* . Verlag der Akademie der Wis- senschaften in Kommission bei Walter De Gruyter u . . . , 1931. [44] Yunyi Shen, Renato Berlinghieri, and Tamara Broderick. Multi-marginal Schrodinger bridges ¨ with iterative reference refinement. *arXiv preprint arXiv:2408.06277*, 2024. [45] Yuyang Shi, Valentin De Bortoli, Andrew Campbell, and Arnaud Doucet. Diffusion schrodinger ¨ bridge matching. *Advances in Neural Information Processing Systems*, 36, 2024. [46] Vignesh Ram Somnath, Matteo Pariset, Ya-Ping Hsieh, Maria Rodriguez Martinez, Andreas Krause, and Charlotte Bunne. Aligned diffusion schr\" odinger bridges. *arXiv preprint arXiv:2302.11419*, 2023. [47] Alexander Tong, Kilian FATRAS, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based genera- tive models with minibatch optimal transport. *Transactions on Machine Learning Research*, 2024. Expert Certification. [48] Alexander Tong, Jessie Huang, Guy Wolf, David Van Dijk, and Smita Krishnaswamy. Trajecto- rynet: A dynamic optimal transport network for modeling cellular dynamics. In *International conference on machine learning*, pages 9526–9536. PMLR, 2020. [49] Francisco Vargas, Pierre Thodoroff, Austen Lamacraft, and Neil Lawrence. Solving Schrodinger ¨ bridges via maximum likelihood. *Entropy*, 23(9):1134, 2021.
# NeurIPS Paper Checklist

#### 1. Claims

 Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

 Justification: The key claims stated in the abstract and introduction correspond appropriately to the scope of the paper.

Guidelines:

 • The answer NA means that the abstract and introduction do not include the claims made in the paper. • The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers. • The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings. • It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

#### 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The conclusion section provides a discussion on the limitations.

Guidelines:

 • The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper. • The authors are encouraged to create a separate "Limitations" section in their paper. • The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be. • The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated. • The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon. • The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size. • If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness. • While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an impor- tant role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

### 3. Theory assumptions and proofs

 Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

 Justification: Yes, we are confident that our proof and assumptions are both valid and adequate.

Guidelines:

 • The answer NA means that the paper does not include theoretical results. • All the theorems, formulas, and proofs in the paper should be numbered and cross- referenced. • All assumptions should be clearly stated or referenced in the statement of any theorems. • The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition. • Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.

#### 4. Experimental result reproducibility

 Question: Does the paper fully disclose all the information needed to reproduce the main ex- perimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

 Justification: Yes, all the necessary data to reproduce the results can be found in the Appendix C.

Guidelines:

 • The answer NA means that the paper does not include experiments. • If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not. • If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable. • Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed. • While NeurIPS does not require releasing code, the conference does require all submis- sions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

 Question: Does the paper provide open access to the data and code, with sufficient instruc- tions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: We provided our code.

Guidelines:

 • The answer NA means that paper does not include experiments requiring code. • Please see the NeurIPS code and data submission guidelines ([https://nips.cc/](https://nips.cc/public/guides/CodeSubmissionPolicy) [public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details. • While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark). • The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ([https:](https://nips.cc/public/guides/CodeSubmissionPolicy) [//nips.cc/public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details. • The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc. • The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why. • At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable). • Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

#### 6. Experimental setting/details

 Question: Does the paper specify all the training and test details (e.g., data splits, hyper- parameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: We have included the details of the experiments.

Guidelines:

 • The answer NA means that the paper does not include experiments. • The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them. • The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

 Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

 Justification: Yes, we ran our code three times and reported the mean and standard deviations in the appendix. Due to space limitations, only the mean values are presented in the main text. The complete results can be found in Appendix C.

Guidelines:

 • The answer NA means that the paper does not include experiments. • The authors should answer "Yes" if the results are accompanied by error bars, confi- dence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper. • The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

 • The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors). • It should be clear whether the error bar is the standard deviation or the standard error of the mean. • It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified. • For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates). • If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

#### 8. Experiments compute resources

 Question: For each experiment, does the paper provide sufficient information on the com- puter resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

 Justification: Yes, the necessary resources are included in the experimental details section. Guidelines:

 • The answer NA means that the paper does not include experiments. • The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage. • The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute. • The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

# 9. Code of ethics

 Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?

Answer: [Yes]

Justification: We support the NeurIPS Code of Ethics.

Guidelines:

 • The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. • If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics. • The authors should make sure to preserve anonymity (e.g., if there is a special consid-eration due to laws or regulations in their jurisdiction).

### 10. Broader impacts

 Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

 Justification: This paper presents work aimed at advancing the field of machine learning. Our research may have various societal consequences. However, we do not believe any of these require specific emphasis here.

Guidelines:

 • The answer NA means that there is no societal impact of the work performed. • If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

 • Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations. • The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster. • The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology. • If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

#### 11. Safeguards

 Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: We believe our paper poses no such risks.

Guidelines:

 • The answer NA means that the paper poses no such risks. • Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters. • Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images. • We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

#### 12. Licenses for existing assets

 Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: Yes, the license and terms of use are noted.

Guidelines:

 • The answer NA means that the paper does not use existing assets. • The authors should cite the original paper that produced the code package or dataset. • The authors should state which version of the asset is used and, if possible, include a URL. • The name of the license (e.g., CC-BY 4.0) should be included for each asset. • For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided. • If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, <paperswithcode.com/datasets> has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

 • For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided. • If this information is not available online, the authors are encouraged to reach out to the asset's creators.

#### 13. New assets

 Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: The paper does not release new assets.

Guidelines:

 • The answer NA means that the paper does not release new assets. • Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc. • The paper should discuss whether and how consent was obtained from people whose asset is used. • At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

#### 14. Crowdsourcing and research with human subjects

 Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: We do not involve crowdsourcing or research with human subjects.

Guidelines:

 • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribu- tion of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

#### 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

 Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: We do not involve crowdsourcing or research with human subjects

Guidelines:

 • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

#### 16. Declaration of LLM usage

 Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

 Justification: We do not use LLM for core methodology, scientific rigorousness, or originality of the research.

Guidelines:

 • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (<https://neurips.cc/Conferences/2025/LLM>) for what should or should not be described.