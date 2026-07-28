# On Flow-based Generative Models for Probabilistic Forecasting

Anonymous Author(s) Affiliation Address email

# Abstract

 Flow-based generative models (FBGM) have emerged as a dominant approach to generative modeling in many domains for their scalability and controllability, but have notably not made the same impact on autoregressive probabilistic forecasting. Although the methodology behind these models can be applied directly to the time series setting, and in theory offers the potential to apply the advances in generative modeling to time series, this direct approach is difficult to use in practice. In this work, we investigate this methodological gap by generalizing the key elements of flow-based generative modeling to the time series setting to devise a more practical related algorithm. We show that FBGMs based on linear stochastic differential equations are instances of a more general mean-field variational inference algorithm for conditional exponential family distributions that constructs Bayes estimators of natural parameters. This insight yields a family of mean-squared error based latent probabilistic forecasters that contains a discrete time counterpart of FBGMs for time series. We demonstrate that the models we develop inherit the convenient theoretical properties of FBGMs while being easy to work with in practice.

# 1 Introduction

 Flow-based generative models (FBGM), including denoising diffusion, score based diffusion, and flow matching models, have become the dominant approach to generative modeling. These models represent a stochastic differential equation (SDE) that transforms samples from a known prior distribution into samples from an unknown target distribution, and often use a different recipe for solving the generative modeling problem compared to traditional approaches. This alternative approach is highly scalable [\[Ramesh et al., 2022,](#page-9-0) [Podell et al., 2023,](#page-9-1) [Saharia et al., 2022\]](#page-9-2), can leverage conditioning information in flexible ways [\[Dhariwal and Nichol, 2021,](#page-9-3) [Ho and Salimans, 2022\]](#page-9-4), and [c](#page-9-6)an be controlled in order to incorporate user defined dynamics [\[Liu et al., 2024,](#page-9-5) [Domingo-Enrich](#page-9-6) [et al., 2024,](#page-9-6) [Havens et al., 2025\]](#page-9-7). Furthermore, FBGMs are capable of learning from paired data. If x<sup>0</sup> and x<sup>1</sup> are samples from an unknown joint distribution p(x0, x1), then one can use the same approach to construct an SDE whose transition distribution from t = 0 to t = 1 is p(x1|x0) [\[De Bortoli et al.,](#page-9-8) [2023\]](#page-9-8). Given this capability, it directly follows that this approach could, in principle, be used to construct an SDE to model time series data. If p(x1:<sup>N</sup> ) = p(x1) Q<sup>N</sup>−<sup>1</sup> <sup>k</sup>=1 p(xk+1|x1:k) represents the unknown distribution of time series data, then each of the transition terms, p(xk+1|x1:k), can be interpreted as a target distribution for a FBGM in the paired data setting where the data pairs are consecutive elements of the time series, (xk+1, xk), and the previous elements x1:k−<sup>1</sup> can be thought of as extra conditioning information. In theory, learning this kind of model for time series would inherit the scalability and controllability that FBGMs possess, allowing practitioners to port over the recent advances in generative modeling to time series applications. However, this approach has surprisingly only recently been explored [\[Chen et al., 2024a,](#page-9-9) [Tamir et al., 2024,](#page-9-10) [Park et al., 2024,](#page-9-11) [Chen et al., 2024b\]](#page-9-12) even though diffusion based time series models have been studied for several

 years [\[Yang et al., 2024,](#page-9-13) [Meijer and Chen, 2024\]](#page-9-14). We attribute this gap to the practical numerical difficulties associated with training and sampling from these models as one must first learn, and then simulate, a stochastic differential equation, with potentially non-smooth dynamics, over a long time domain compared to the short time domain encountered in standard generative modeling. To address this problem, we develop a discrete time version of Neural SDEs derived from FBGMs that are founded on the same theoretical principles, while being substantially easier to work with in practice. We do this by generalizing two key elements needed to construct FBGMs, stochastic interpolation and the Markovian projection, to the time series setting, where they become Gaussian condition random fields and a form of mean-field variational inference respectively. We construct a family of latent probabilistic time series models that are closely related to existing time series models, including MSE based non-probabilistic forecasters and conditional Gaussian autoregressive models, and compare their performance on various latent probabilistic forecasting problems.

# 2 Background

 We will first review how flow-based generative models are constructed and then build intuition for how to go about generalizing this construction to the time series setting. Suppose that p(y0, y1) is a joint distribution over a source and target random variable. The (paired) generative modeling problem is to find a parametric approximation of p(y1|y0) . Flow-based generative models solve this problem by constructing, and then learning, a *latent* SDE whose transition distribution from times t = 0 to t = 1 is p(y1|y0). There are three steps involved in constructing and learning this SDE - stochastic interpolation, the Markovian projection, and matching.

 Stochastic interpolation [\[Albergo and Vanden-Eijnden, 2023\]](#page-9-15) is used to interpolate between proba- bility distributions by defining interpolations between their samples. For example, consider the joint distribution p(x0, xt, x1), where x<sup>t</sup> = (1 − t)x<sup>0</sup> + tx<sup>1</sup> and (x0, x1) ∼ p(x0, x1). By the definition of xt, it is true that p(xt=1) = p(x1), and also that p(xt=1|x0) = p(x1|x0), so we verify that the marginal distribution of x<sup>t</sup> interpolates between p(x0) and p(x1). In practice, one assumes that at times t = 0 and t = 1, x<sup>0</sup> := y<sup>0</sup> and x<sup>1</sup> := y<sup>1</sup> so that p(xt) is an interpolation between p(y0) and p(y1).

 A popular method for constructing stochastic interpolants, which we use in this paper, is conditioning a user-defined base SDE, whose diffusion coefficient does not depend on the current state, to start at x<sup>0</sup> and end at x1. This SDE takes the form dx<sup>t</sup> = bt(xt)dt + LtdW<sup>t</sup> where bt(xt) is the drift of this base SDE and L<sup>t</sup> is the diffusion coefficient. This SDE is used to construct a joint distribution of the form p(x0, xt, x1) = p(xt|x0, x1)p(x0, x1) where p(xt|x0, x1) is the probability of x<sup>t</sup> when the base SDE has been conditioned to start at x<sup>0</sup> and end at x1. In order to solve the generative modeling problem of p(x1|x0), FBGMs are constructed as an SDE whose marginal distribution is p(xt|x0). This is accomplished using the Markovian projection.

Proposition 1 (Markovian projection SDE [\[Shi et al., 2024\]](#page-9-16)). *Let* p(x1|x0) *be a conditional distribu- tion over target variables given source variables and let* p(xt|x0, x1) *denote the distribution of the base SDE* dx<sup>t</sup> = bt(xt)dt + LtdW<sup>t</sup> *when conditioned to start at* x<sup>0</sup> *and end at* x1*. The "Markovian projection SDE" is an SDE whose marginal distribution, denoted by* q ∗ (xt|x0) *is equal to* p(xt|x0)*. It is given by:*

$$dx_t = (b_t(x_t) + L_t T_t^T \mathbb{E}_{p(x_1|x_0, x_t)} [\nabla \log p(x_1|x_0, x_t)]) dt + L_t dW_t \quad (1)$$

 See Prop 3. of [\[De Bortoli et al., 2023\]](#page-9-8) for a proof. Proposition [1](#page-1-1) is a solution to the paired generative modeling problem because q ∗ (xt=1|x0) = p(x1|x0) := p(y1|y0). Given a sample from the source distribution, x<sup>0</sup> ∼ p(x0), we can simulate the SDE from t = 0 to t = 1 to generate a sample from the target distribution. However, this SDE contains an intractable drift term that depends on the posterior distribution of x<sup>1</sup> given x<sup>0</sup> and xt. This is addressed using a matching learning objective. For example, in score matching, [\[Vincent, 2011,](#page-9-17) [Song et al., 2021\]](#page-10-0), one writes the drift in the following variational form:

$$\nabla \log q^*(x_t|x_0) = \underset{s_t(x_t, x_0)}{\operatorname{argmin}} \mathbb{E}_{p(x_0, x_1, x_t)} \left[ \|L_t L_t^T \nabla \log p(x_1|x_0, x_t) - s_t(x_t, x_0)\|^2 \right] \quad (2)$$

The unpaired setting is when we do not condition on y0.

 If s(xt, x0; θ) is parameterized by a neural network, then one can minimize this expectation using the standard machine learning toolkit to find the Markovian projection SDE. However, obtaining a Monte Carlo estimate of the expectation for stochastic gradient descent requires being able to sample from p(x0, x1, xt), which requires simulation of the base SDE. As such, the base SDE is chosen so that this distribution is tractable. After training is complete, then the flow-based generative model is given by the SDE dx<sup>t</sup> = (bt(xt) + LtL T t st(xt, x0))dt + LtdWt. In general, matching algorithms, such as score matching, drift matching and bridge matching, are algorithms for learning the Bayes estimator of a random variable because of the well known relationship between posterior expectations and mean squared error [\[Jaynes, 2003\]](#page-10-1):

Proposition 2 (Bayes estimate of parameter). *Let* p(z, θ) *be a joint distribution and let* θ ∗ (z) *be the Bayes estimate of* θ *based on* z *under the squared error risk. Then the Bayes estimate takes the following two forms:*

$$\theta^*(z) = \mathbb{E}_{p(\theta|z)}[\theta] = \underset{f(z)}{\operatorname{argmin}} \mathbb{E}_{p(z,\theta)}[\|f(z) - \theta\|^2] \quad (3)$$

 See Appendix [C.3](#page-14-0) for a derivation. In score matching, one would have z = (x0, xt) and θ = [∇](#page-9-15) log p(x1|x0, xt), while other matching approaches, such as flow matching [\[Albergo and Vanden-](#page-9-15)[Eijnden, 2023,](#page-9-15) [Lipman et al., 2023,](#page-10-2) [Liu et al., 2023\]](#page-10-3) and bridge matching [\[Shi et al., 2024\]](#page-9-16).

 Given the strong theoretical, interpretability, and empirical results of FBGMs, one might expect that a direct application to time series would inherit the same benefits. However, this approach has surprisingly only recently been explored [\[Chen et al., 2024a,](#page-9-9)[b,](#page-9-12) [Tamir et al., 2024,](#page-9-10) [Park et al., 2024\]](#page-9-11) even though diffusion based time series models have been studied in a different manner for several years [\[Yang et al., 2024,](#page-9-13) [Meijer and Chen, 2024\]](#page-9-14). We attribute this gap to the challenges that the time series setting presents to flow-based methods compared to settings such as image generation. In the standard image generation setting, there is no coupling between the prior and data distributions, and [s](#page-10-3)o one can learn SDEs that can be easily simulated with a few number of function evaluations [\[Liu](#page-10-3) [et al., 2023,](#page-10-3) [Pooladian et al., 2023\]](#page-10-4). However, SDEs that are constructed to model time series data present a challenge during inference due to compounding numerical errors that are attributed to either a mismatch between the learned model and data, or due to the numerical solver itself, get accumulated during generation which can lead to poor performance in practice. Discrete time autoregressive models, on the other hand, do not suffer from these issues to the extent that Neural SDEs do and are much more widely used in practice. With this in mind, we aim to understand find a discrete time version of FBGMs for time series that will work better in practice.

# 3 Method

We present a generalization of the FBGM construction for the time series setting.

### 3.1 Generalized linear stochastic interpolation

 Recall that stochastic interpolation constructs a distribution over a latent stochastic process, which we denote by x, that is sampled from a base SDE that is conditioned to start at x<sup>0</sup> := y<sup>0</sup> and end at x<sup>1</sup> := y1. Our generalization of stochastic interpolation is founded on the observation that many of the base SDEs used in practice are linear SDEs, and that the FBGM recipe is unchanged if we introduce Gaussian potential functions to relax the endpoint conditions. Since linear SDEs have Gaussian transition distributions, they can naturally be combined with these Gaussian potentials to construct a Gaussian conditional random field. This conditional random field will serve as our tool for stochastic interpolation, which we call "generalized linear stochastic interpolation".

Let y<sup>τ</sup>1:<sup>T</sup> denote time series data that is generated by an unknown distribution p(y<sup>τ</sup>1:<sup>T</sup> ). For brevity, we assume that τ1:<sup>T</sup> is the same for all time series, but note that our theory accommodates datasets with series sampled at different times. We will construct, and perform inference, in the distribution p(x|y<sup>τ</sup>1:<sup>T</sup> ), which we will obtain by conditioning a linear SDE on user defined Gaussian potential functions. The potential function at time t<sup>k</sup> ∈ R will be denoted by ϕ(x<sup>t</sup><sup>k</sup> |θ<sup>t</sup><sup>k</sup> (y<sup>τ</sup>1:<sup>T</sup> )), where θ<sup>t</sup><sup>k</sup> the the natural parameter of the Gaussian that arbitrarily depends on y<sup>τ</sup>1:<sup>T</sup> . See Appendix [C](#page-13-0) for a review of exponential family distributions. We also use the notation ϕk+1|k(xk+1|xk) = N(xk+1|Ax<sup>k</sup> + u, Σ) to denote a Gaussian transition distribution from x<sup>k</sup> to xk+1 with state transition matrix A, bias vector u and covariance matrix Σ.

![](_page_3_Diagram_0.jpeg)

Figure 1: Generalized stochastic interpolation incorporates Gaussian potential functions to relax the endpoint conditions of stochastic interpolation and is applied to time series data.

#### <sup>135</sup> 3.1.1 Gaussian conditional random fields

<sup>136</sup> Chain structured Gaussian CRFs are a tractable class of probabilistic models that are widely used in <sup>137</sup> time series modeling (CITE):

Definition 1 (Conditional Random Field [\[Lafferty et al., 2001,](#page-10-5) [Sutton et al., 2012\]](#page-10-6)). *Let* x1:<sup>N</sup> *be a sequence of random variables,* ϕk+1|k(xk+1|xk) *be a set of Gaussian transition distributions between consecutive variables, and* ϕ(xk|θk) *a set of Gaussian potential functions with natural parameters* θ<sup>k</sup> ∈ θ*. A conditional random field (CRF) is a probability distribution given by:*

$$p(x_{1:N}|\theta) \propto \prod_{k=1}^{N-1} \phi_{k+1|k}(x_{k+1}|x_k) \prod_{k=1}^N \phi(x_k|\theta_k) \quad (4)$$

<sup>142</sup> Due to the chain-structure of p(x1:<sup>N</sup> |θ) and the fact it is jointly Gaussian, inference can be performed <sup>143</sup> efficiently using message passing. The backward messages, defined below, will play a significant role <sup>144</sup> in our theory:

<sup>145</sup> Proposition 3 (Backward messages). *The* k*'th backward message associated with the CRF in* <sup>146</sup> *Definition [1](#page-3-0) is defined with the following recurrence relation:*

$$\phi(x_{k-1}|\beta_{k-1}) = \int \phi_{k|k-1}(x_k|x_{k-1})\phi(x_k|\theta_k + \beta_k)dx_k, \quad \beta_N = 0 \quad (5)$$

<sup>147</sup> *where* θk+1 +βk+1 *denotes the direct sum of* θk+1 *and* βk+1*. This recurrence also uniquely identifies* <sup>148</sup> *a function, denoted by* Φk,k+1 *that performs the parameter updates as:*

$$\beta_k = \Phi_{k,k+1}(\theta_{k+1} + \beta_{k+1}) \quad (6)$$

 Note that each β<sup>k</sup> is a function of θk+1:<sup>N</sup> . See Appendix [D](#page-15-0) for a full derivation of sequential and parallel message passing, and Appendix [H](#page-30-0) for pseudo code and implementation considerations. Although we do not focus on the forward messages, they are defined with analogous recurrence relations to the backward messages and can be used to extend our methodology to flow-matching models for time series forecasting (see Corollary [5\)](#page-22-0). CRFs offer an efficient way to model the latent variables at a fixed set of times, but are not immediately suited for continuous time.

#### <sup>155</sup> 3.1.2 Linear time-invariant stochastic differential equations

 We will use linear-time invariant SDEs to construct the transition distributions of continuous time CRFs. Linear time-invariant SDEs (LTI-SDEs) are SDEs of the form dx<sup>t</sup> = F xtdt + LdWt, where the drift matrix F and diffusion coefficient matrix L are constant with respect to t and xt. LTI-SDEs [h](#page-10-7)ave the convenient property that their transition distribution is available in closed form [\[Särkkä and](#page-10-7) [Solin, 2019,](#page-10-7) [Singhal et al., 2023\]](#page-10-8). The transition distribution from x<sup>t</sup> to xt+s, where s > 0 is an increment of time, is given by

$$\phi_{t+s|t}(x_{t+s}|x_t) = N(x_{t+s}|A_s x_t, \Sigma_s), \quad \text{where} \quad \begin{bmatrix} A_s & \Sigma_s A_s^{-T} \\ 0 & A_s^{-T} \end{bmatrix} := \exp \left\{ \begin{bmatrix} F & LL^T \\ 0 & -F^T \end{bmatrix} s \right\} \quad (7)$$

 We use LTI-SDEs for their tractability, but note that our theory is completely compatible with more general linear SDEs. One can directly plug in this transition distribution into a CRF in Definition [1](#page-3-0) to obtain a conditional random field over a continuous time domain. However, we can be more general. In the next proposition, we highlight a relationship between conditioned linear SDEs and CRFs ([\[Särkkä et al., 2006,](#page-10-9) [Särkkä and Solin, 2019\]](#page-10-7)):

<sup>167</sup> Proposition 4 (Conditioned LTI-SDE). *Let* ϕt+s|t(xt+s|xt) *be the transition distribution of the LTI-SDE* dx<sup>t</sup> = F xtdt + LdW<sup>t</sup> *and let* {ϕ(x<sup>t</sup><sup>k</sup> |θ<sup>t</sup><sup>k</sup> <sup>168</sup> )}<sup>t</sup>k∈R *be potential functions at times in the set* <sup>169</sup> R*. Then the piecewise-linear SDE,*

$$dx_t = (Fx_t + LL^T \nabla \log \phi(x_t|\beta_t))dt + LdW_t, \quad x_{t+1} \sim \phi(x_{t+1}|\beta_{t+1} + \theta_1) \quad (8)$$

<sup>170</sup> *where* t ∈ (tk, tk+1) *and* tk, tk+1 ∈ R*, has a joint distribution at the times* t1:<sup>N</sup> = T ⊇ R *that is* <sup>171</sup> *given by a CRF:*

$$p(x_{t_{1:N}}|\theta) \propto \prod_{t_k \in \mathcal{T}} \phi_{t_{k+1}|t_k}(x_{t_{k+1}}|x_{t_k}) \prod_{t_k \in \mathcal{R}} \phi(x_{t_k}|\theta_{t_k}) \quad (9)$$

*where* β<sup>t</sup> = Φt,tk+1 (θ<sup>t</sup>k+1 + β<sup>t</sup>k+1 <sup>172</sup> )*.*

 See appendix Appendix [E.1](#page-20-0) for the full proof and Corollary [5](#page-22-0) for a nice expression for the associated probability flow ODE in terms of both the forward and backward messages. Proposition [4](#page-4-0) suggests that a practical way to work with conditioned linear SDEs in practice is convert them into CRFs on a discretization of the time domain so that inference can be performed via message passing. This results in the ability to sample and perform inference in linear SDEs O(log |T |) time on parallel compute [\[Hassan et al., 2021,](#page-10-10) [Corenflos et al., 2021,](#page-10-11) [Smith et al., 2023\]](#page-10-12). The conditioned SDE Proposition [4](#page-4-0) is our main tool for stochastic interpolation as it gives us the ability to sample from p(x|θ(y<sup>τ</sup>1:<sup>T</sup> <sup>179</sup> )) at an arbitrary discretization of the time domain.

#### <sup>181</sup> 3.2 Target probabilistic model for FBGM

<sup>182</sup> Recall that in the FBGM recipe, we used the stochastic interpolation to construct a joint distribution <sup>183</sup> over the interpolant and the data, p(y0, xt, y1), before performing the Markovian projection. We can take the same step here to construct a joint distribution over y<sup>τ</sup>1:<sup>T</sup> <sup>184</sup> and x using the data distribution, p(y<sup>τ</sup>1:<sup>T</sup> ) and the distribution of the interpolant, p(x|y<sup>τ</sup>1:<sup>T</sup> ) := p(x|θ(y<sup>τ</sup>1:<sup>T</sup> <sup>185</sup> )).

Definition 2 (Target joint distribution). *Let* p(y<sup>τ</sup>1:<sup>T</sup> <sup>186</sup> ) *be the distribution of observed time series data and let* p(x|y<sup>τ</sup>1:<sup>T</sup> <sup>187</sup> ) *be the distribution of the generalized linear stochastic interpolant, which is the distribution of a linear SDE conditioned on the user defined potential functions* {θ<sup>t</sup><sup>k</sup> (y<sup>τ</sup>1:<sup>T</sup> <sup>188</sup> )}<sup>t</sup>k∈R *at* <sup>189</sup> *the times* R*, as in Proposition [4.](#page-4-0) Then the induced joint distribution over* x *at the times* t1:<sup>N</sup> = T ⊃ R *and* y<sup>τ</sup>1:<sup>T</sup> <sup>190</sup> *is given by:*

$$p(x_{t_{1:N}}, y_{\tau_{1:T}}) = p(y_{\tau_{1:T}}) \left( \frac{1}{Z(y_{\tau_{1:T}})} \prod_{t_k \in \mathcal{T}} \phi_{t_{k+1}|t_k}(x_{t_{k+1}}|x_{t_k}) \prod_{t_k \in \mathcal{R}} \phi(x_{t_k}|\theta_{t_k}(y_{\tau_{1:T}})) \right) \quad (10)$$

*where* Z(y<sup>τ</sup>1:<sup>T</sup> ) *is the partition function of* p(x<sup>t</sup>1:<sup>N</sup> |y<sup>τ</sup>1:<sup>T</sup> <sup>191</sup> )*.*

 Before continuing, it is crucial that we understand this joint distribution and the role it plays in the FBGM recipe. Unlike the standard approach to generative modeling where one defines a joint distribution by defining a prior over the latent variable and a likelihood distribution over the data, the FBGM uses an alternate construction to build p(x, y<sup>τ</sup>1:<sup>T</sup> ) using the data distribution directly. Furthermore, the tools FBGMs employ are fundamentally designed for probabilistic inference in x instead of y<sup>τ</sup>1:<sup>T</sup> . Since x is completely user designed through the choice of base LTI-SDE and potential functions, we are able to solve a wide range time series problems.

 Suppose we split each sequence of data into observed and unobserved portions, y<sup>τ</sup>1:<sup>T</sup> = (yO, y<sup>U</sup> ), where y<sup>O</sup> is a subsequence that we observe at both train and test time while y<sup>U</sup> is only observed at training time, as is the case in time series forecasting.[<sup>2</sup>](#page-4-1) The ability to perform inference in p(x|yO) would solve a general latent probabilistic forecasting problem that reduces to the stan-dard forecasting problem if the Gaussian potential functions are chosen as dirac delta functions -

<sup>2</sup>This also covers the imputation setting, but we do not explore this in the interest of keeping a narrow scope.

![](_page_5_Diagram_0.jpeg)

Figure 2: The CMFVI approximation of p(x|z) is q ∗ (x|z). Choosing (x, z, θ) = (x<sup>t</sup>1:<sup>N</sup> , yO, θ(y<sup>τ</sup>1:<sup>T</sup> )) recovers q MSE , (x, z, θ) = (x<sup>t</sup><sup>k</sup> ,(x<sup>t</sup>1:k−<sup>1</sup> , yO), θ(y<sup>τ</sup>1:<sup>T</sup> )) recovers q MSE-AR and (x, z, θ) = lims→0(xt+s,(xt, x<sup>t</sup>1:k−<sup>1</sup> , yO), θ(y<sup>τ</sup>1:<sup>T</sup> )) for t ∈ (tk, tk+1) recovers q Neural-SDE .

ϕ(x<sup>t</sup><sup>k</sup> |θ<sup>t</sup><sup>k</sup> (y<sup>τ</sup>1:<sup>T</sup> )) := δ(x<sup>t</sup><sup>k</sup> − y<sup>t</sup><sup>k</sup> <sup>204</sup> ). For example, if one chooses the LTI-SDE to be the Wiener ve-<sup>205</sup> locity model [\[Särkkä and Solin, 2019,](#page-10-7) [Särkkä et al., 2006\]](#page-10-9) and potential functions of the form ϕ(x<sup>t</sup><sup>k</sup> |θ(y<sup>τ</sup>1:<sup>T</sup> )) ∝ N(x<sup>t</sup><sup>k</sup> |y<sup>t</sup><sup>k</sup> , σ<sup>2</sup> <sup>206</sup> I), then inference in p(x|yO) corresponds to forecasting the smoothed position and velocity of the particle whose positions were observed at y<sup>τ</sup>1:<sup>T</sup> <sup>207</sup> . However, p(x|yO) is intractable because p(y<sup>τ</sup>1:<sup>T</sup> <sup>208</sup> ) is arbitrary. To this end, we develop variational inference <sup>209</sup> algorithms for this task.

#### <sup>210</sup> 3.3 Neural latent SDE for latent probabilistic forecasting

<sup>211</sup> The first inference algorithm we develop is a direct extension of flow-based generative models to the <sup>212</sup> latent probabilistic forecasting setting. For a fixed discretization of the time domain, we can treat consecutive latent variables (x<sup>t</sup><sup>k</sup> , x<sup>t</sup>k+1 <sup>213</sup> ) as elements of a paired dataset with the previous elements x<sup>t</sup>1:k−<sup>1</sup> <sup>214</sup> and observations y<sup>O</sup> as extra conditioning information. This lets us directly apply the existing <sup>215</sup> FBGM recipe to construct a conditional, piecewise SDE to solve the latent probabilistic forecasting <sup>216</sup> problem.

Proposition 5 (Neural latent SDE). *Let* p(x<sup>t</sup>1:<sup>N</sup> , y<sup>τ</sup>1:<sup>T</sup> ) *be the joint distribution defined in Definition [2](#page-4-2) and suppose that* y<sup>τ</sup>1:<sup>T</sup> = (yO, y<sup>U</sup> )*, where* O *and* U *are the times at which sequences are observed and unobserved at test time, respectively. Then the neural latent SDE is the following piecewise SDE:*

$$dx_t = (F_t x_t + L_t L_t^T \nabla \log \phi(x_t | \beta_t^*(x_t, x_{1:k}, y_{\mathcal{O}}))) dt + L_t dW_t, \quad (11)$$

where 
$$\beta_t^*(x_t, x_{t_{1:k}}, y_{\mathcal{O}}) = \mathbb{E}_p(y_{\mathcal{U}}|x_t, x_{t_{1:k}}, y_{\mathcal{O}}) [\beta_t(y_{\tau_{1:T}})]$$
, and  $t \in (t_k, t_{k+1})$  (12)

*Furthermore, the transition distribution of this SDE from time* t<sup>k</sup> *to* tk+1 *is* p(x<sup>t</sup>k+1 |x<sup>t</sup>1:<sup>k</sup> <sup>220</sup> , yO)*. We will use* q *Neural-SDE* <sup>221</sup> *to denote the path measure associated to this SDE.*

 See Appendix [G.2](#page-29-0) for a proof and Appendix [G](#page-27-0) for the general constructions of the score function, Markovian projection SDE and probability flow ODE. By construction, Proposition [5](#page-5-0) can be used to solve the latent probabilistic forecasting problem because it has the correct joint distribution over the latent space. Furthermore, its form is almost identical to that of its base LTI-SDE in Proposition [4,](#page-4-0) except that its parameter, β ∗ , is the Bayes estimator of a backward message. We will show next that models of this form can be derived by solving a constrained mean-field variational inference problem.

#### <sup>228</sup> 3.4 Constrained mean-field variational inference

 Next we introduce our main contribution which is the variational inference algorithm underlying FBGMs, which we call "constrained mean-field variational inference". Given a conditional expo- nential family distribution p(x|z, θ), CMFVI constructs a variational approximation of p(x|z) that is given by p(x|z, θ<sup>∗</sup> (z)) where θ ∗ (z) is the Bayes estimator of θ given z. We first introduce CMFVI in an abstract way and then show how it can be used to do variational inference on the latent probabilistic forecasting distribution, p(x<sup>t</sup>1:<sup>N</sup> |yO).

 Suppose that z is a random variable, θ ∼ p(θ|z) is the natural parameter of an exponential family distribution, and x ∼ p(x|z, θ) is a random variable drawn from a conditional exponential family of the form p(x|z, θ) = exp{⟨tz(x), θ⟩−A(z, θ)}. For intuition, assume that x represents the future of a stochastic process, z represents its past , and θ represents the parameters of this process. Furthermore,  suppose that the parameters are only available at training time so that at test time, sampling x given z requires the ability to sample from p(x|z). Our goal is to predict the future of the process given its past, which requires the ability to sample from p(x|z), however this distribution is intractable because p(θ|z) is arbitrary. To this end, we introduce a variational approximation of p(x|z) using an algorithm closely resembling mean field variational inference, which we call "constrained mean field variational inference" (CMFVI):

<sup>245</sup> Theorem 1 (Constrained mean field VI solution). *Let* p(x|z, θ) ∝ exp{⟨tz(x), θ⟩ − A(z, θ)} *be* <sup>246</sup> *a conditional exponential family distribution with* θ ∼ p(θ|z)*. The constrained mean field VI approximation of* p(x|z)*, denoted by* q ∗ <sup>247</sup> (x|z)*, is defined as follows:*

$$q^*(x|z) = \operatorname{argmin}_{q(x|z)} \text{KL} [q(x|z)p(\theta|z) \| p(x, \theta|z)] \quad (13)$$

$$= p(x|z, \theta^*(z)), \quad \text{where } \theta^*(z) = \mathbb{E}_{p(\theta|z)} [\theta] \quad (14)$$

 See Appendix [F.1](#page-23-0) for a proof, Lemma [4](#page-23-1) for equivalent expressions for the objective involving KL[q ∗ (x|z)∥p(x|z)] and a term resembling the mutual information between x and θ given z. The parameter θ ∗ (z) is the Bayes estimator of θ given z and by Proposition [2](#page-2-0) can be learned using mean squared error minimization, provided that it is possible to sample from p(z, θ). While this variational approximation is tractable, it seems restrictive because it is a conditional random field and only exact when θ and x are conditionally independent given z. However, this may not be a terrible assumption in the time series setting. If the process is deterministic, then we should be able to compute x directly from z without needing to know θ, and so this independence assumption will hold because one will be able to compute the future values of the process directly from its past. In fact, in Corollary [8,](#page-26-0) we show that a direct application of CMFVI to p(x<sup>t</sup>1:<sup>N</sup> |yO), by selecting x = x<sup>t</sup>1:<sup>N</sup> , z = y<sup>O</sup> and θ = θ(y<sup>τ</sup>1:<sup>T</sup> ), exactly recovers MSE based non-probabilistic forecasters, which are clearly capable of learning deterministic processes (see Corollary [8\)](#page-26-0). We denote the model in Corollary [8](#page-26-0) by q MSE <sup>259</sup> . In general, provided that the process is not too stochastic, we might expect that given a long enough history and a short enough prediction horizon that CMFVI could yield a reasonable approximation of p(x|z), and perhaps with an infinitely short prediction horizon we may recover something exactly. This intuition motivates the use of CMFVI for learning the autoregressive factors of p(x<sup>t</sup>1:<sup>N</sup> |yO) in order to construct an autoregressive model to solve the probabilistic forecasting problem.

Suppose that p(x<sup>t</sup><sup>k</sup> |x<sup>t</sup>1:k−<sup>1</sup> <sup>265</sup> , yO) is one of the autoregressive factors of the latent forecasting distribution p(x<sup>t</sup>1:<sup>N</sup> |yO). We can use CMFVI to approximate each of the k factors by setting x = x<sup>t</sup><sup>k</sup> <sup>266</sup> , z = (x<sup>t</sup>1:k−<sup>1</sup> , yO) and θ = θ(y<sup>τ</sup>1:<sup>T</sup> <sup>267</sup> ):

<sup>268</sup> Proposition 6 (CMFVI transition approximation). *Let* p(x<sup>t</sup>1:<sup>N</sup> |yO) *be the target distribution and consider its* k*'th autoregressive factor* p(x<sup>t</sup><sup>k</sup> |x<sup>t</sup>1:k−<sup>1</sup> <sup>269</sup> , yO)*. Then the CMFVI transition approximation* <sup>270</sup> *is given by:*

$$q^{transition}(x_{t_k} | x_{t_{1:k-1}}, y_{\mathcal{O}}) \propto \phi_{t_k | t_{k-1}}(x_{t_k} | x_{t_{k-1}}) \phi(x_{t_k} | \beta_{t_k}^*(x_{t_{1:k-1}}, y_{\mathcal{O}})) \quad (15)$$

*where* β ∗ tk (x<sup>t</sup>1:k−<sup>1</sup> , yO) = <sup>E</sup>p(y<sup>U</sup> <sup>|</sup>xt1:k−<sup>1</sup> ,yO) [β<sup>t</sup><sup>k</sup> (y<sup>τ</sup>1:<sup>T</sup> )] *is the Bayes estimate of* β<sup>t</sup><sup>k</sup> (y<sup>τ</sup>1:<sup>T</sup> <sup>271</sup> )*, which is defined using the message passing update operator* Φ<sup>t</sup>k,tk+1 <sup>272</sup> *from Definition [7](#page-18-0) as:*

$$\beta_{t_k} = \begin{cases} \Phi_{t_k, t_{k+1}}(\beta_{t_{k+1}}(y_{\tau_{1:T}}) + \theta_{t_{k+1}}(y_{\tau_{1:T}})) & \text{if } t_{k+1} \in \mathcal{R} \\ \Phi_{t_k, t_{k+1}}(\beta_{t_{k+1}}(y_{\tau_{1:T}})) & \text{otherwise} \end{cases} \quad (16)$$

<sup>273</sup> See Proposition [6](#page-6-0) for a proof. The form of Proposition [6](#page-6-0) almost exactly matches the transition distribution of p(x<sup>t</sup>1:<sup>N</sup> |y<sup>τ</sup>1:<sup>T</sup> <sup>274</sup> ) in Proposition [12,](#page-19-0) except that the backward messages are replaced with their Bayes estimators. We will use q transition <sup>275</sup> to construct an autoregressive approximation model that <sup>276</sup> will be a discrete time version of the Markovian projection SDE.

 To use CMFVI to construct a discrete time version of FBGMs for time series, we will need to make the assumption that the covariances of the potential functions are independent of the values of y<sup>τ</sup>1:<sup>T</sup> . This assumption holds in both the data space forecasting setting where we use dirac delta potential functions, and also in the case where the CRF is constructed as a linear dynamical system with constant observation noise. In this setting, it is also possible to rewrite q Neural SDE <sup>281</sup> in a more interpretable form where the only unknown value is the mean of the next backward message:

<sup>283</sup> Corollary 1 (Neural latent SDE using potentials with fixed covariances). *If the covariance matrices associated with* q *Neural SDE are constant with respect to* y*, then the SDE associated with* q *Neural SDE* <sup>284</sup> *is:*

$$dx_t = (F_t x_t + L_t L_t^T \nabla \log N(x_t | \mu_t^{\beta^*}(x_t, x_{t:1:k-1}, y_{\mathcal{O}}), \Sigma_t^{\beta})) dt + L_t dW_t \quad (17)$$

*where* t ∈ (tk−1, tk)*,* Σ β t *is the covariance of* ϕ(xt|βt(y<sup>τ</sup>1:<sup>T</sup> )) *and* µ ∗ t (xt, x<sup>t</sup>1:k−<sup>1</sup> <sup>285</sup> , yO) *is the Bayes* <sup>286</sup> *estimator for it's mean.*

The result follows directly from converting β<sup>t</sup><sup>k</sup> <sup>287</sup> from natural parameters to standard parameters <sup>288</sup> of a Gaussian and the linear equivariance of the Bayes estimator Appendix [F.2.](#page-25-0) Note that by our assumption that the parameters of the potential functions do not depend on y<sup>τ</sup>1:<sup>T</sup> , Σ β t <sup>289</sup> can be computed by performing message passing on p(x<sup>t</sup>1:<sup>N</sup> |<sup>∅</sup>τ1:<sup>T</sup> ), where <sup>∅</sup>τ1:<sup>T</sup> <sup>290</sup> is an empty (or random) sequence sampled at the same times as y<sup>τ</sup>1:<sup>T</sup> <sup>291</sup> .

#### <sup>292</sup> 3.5 Discrete time Markovian projection

<sup>293</sup> We propose an conditional Gaussian autoregressive model whose transition distributions are given by q transition, which we denote by q MSE-AR <sup>294</sup> . We will directly relate it to Markovian projection SDE q Neural-SDE by associating q MSE-AR with a piecewise linear SDE that closely resembles q Neural-SDE <sup>295</sup> .

<sup>296</sup> Proposition 7 (Autoregressive CMFVI solution). *Let* p(x<sup>t</sup>1:<sup>N</sup> |yO) *be the target distribution, as-*<sup>297</sup> *sume that the covariance matrices of its potential functions are constant with respect to* y*. The autoregressive model whose transitions are CMFVI solution, denoted by* q *MSE-AR* <sup>298</sup> *is given by:*

$$q^{MSE-AR}(x_{t_{1:N}}|y_{\mathcal{O}}) \propto p(x_{t_1}|y_{\mathcal{O}}) \prod_{t_k \in \mathcal{T}} \phi_{t_k|t_{k-1}}(x_{t_k}|x_{t_{k-1}})^N(x_{t_k}|\mu_{t_k}^{\beta^*}(x_{t_{1:k-1}}, y_{\mathcal{O}}), \Sigma_{t_k}^{\beta}) \quad (18)$$

*where* Σ β tk *and* µ β tk ∗ (x<sup>t</sup>1:k−<sup>1</sup> , yO) *are the same as in Corollary [1.](#page-6-1) Furthermore,* q *MSE-AR* <sup>299</sup> *has the same* <sup>300</sup> *joint distribution over* x<sup>t</sup>1:<sup>N</sup> *as the following piecewise linear SDE:*

$$dx_t = (F_t x_t + L_t L_t^T \nabla \log N(x_t | \mu_t^\beta(x_{t_{1:k-1}}, y_{\mathcal{O}}), \Sigma_t^\beta)) dt + L_t dW_t, \quad x_{t_1} \sim p(x_{t_1} | y_{\mathcal{O}}) \quad (19)$$

*where* µ ∗ t (x<sup>t</sup>1:k−<sup>1</sup> , yO) *is the Bayes estimator for the mean of* βt(y<sup>τ</sup>1:<sup>T</sup> ) = Φt,t<sup>k</sup> (β<sup>t</sup>k+1 (y<sup>τ</sup>1:<sup>T</sup> ))*,* Σ β t <sup>301</sup> *is* <sup>302</sup> *its covariance matrix and* t ∈ (tk−1, tk) *for* k = 2, . . . , T*.*

<sup>303</sup> See Appendix [F.3](#page-27-1) and Definition [9](#page-27-2) for a proof. A comparison of the piecewise linear SDE associated with q MSE-AR with the piecewise SDE associated to q Neural-SDE reveals why we interpret q MSE-AR <sup>304</sup> as the <sup>305</sup> discrete time version of the Markovian projection SDE. We can see that the only difference between the two SDEs are their Bayes estimators for µ β t (y<sup>τ</sup>1:<sup>T</sup> <sup>306</sup> ):

$$q^{\text{MSE-AR}} : \mu_t^{\beta^*}(x_{t_{1:k}}, y_{\mathcal{O}}) = \mathbb{E}_{p(y_{\mu}|x_{t_{1:k}}, y_{\mathcal{O}})} [\mu_t^{\beta}(y_{\tau_{1:T}})]$$

$$q^{\text{Neural-SDE}} : \mu_t^{\beta^*}(x_t, x_{t_{1:k}}, y_{\mathcal{O}}) = \mathbb{E}_{p(y_{\mu}|x_t, x_{t_{1:k}}, y_{\mathcal{O}})} [\mu_t^{\beta}(y_{\tau_{1:T}})]$$

<sup>307</sup> The only difference between the two Bayes estimators is their dependence on the current state xt. If x<sup>t</sup> does not carry more information about y<sup>U</sup> compared to what is already available from x<sup>t</sup>1:<sup>k</sup> 308 and yO, then we can expect that q MSE-AR and q Neural-SDE <sup>309</sup> will model nearly the same distribution. As <sup>310</sup> we will show in our experiments, this is something that one can expect in the time series setting because data is usually sampled frequently enough where the extra capacity that q Neural-SDE <sup>311</sup> has over q MSE-AR may not make enough of an impact in practice to warrant using q Neural-SDE <sup>312</sup> in practice. We introduced three different CMFVI based time series models - q MSE [8,](#page-26-0) q MSE-AR [7](#page-7-0) and q Neural-SDE <sup>313</sup> [1](#page-6-1) <sup>314</sup> which use CMFVI to joint distribution, transition distributions, and infinitesimal transitions of the <sup>315</sup> target distribution respecitvely. All of these models are Gaussian, and are therefore closely related to <sup>316</sup> existing time series models.

#### <sup>317</sup> 3.6 Connection to traditional time series models

<sup>318</sup> The CMFI-based time series models that we have developed all have an autoregressive Gaussian <sup>319</sup> structure which makes them related to existing time series models. First, when one chooses potential functions to align with the data times R = τ1:<sup>T</sup> , then q MSE <sup>320</sup> is identical to MSE based non-probabilistic <sup>321</sup> forecasters, which are are trained to predict the future of a time series, y<sup>U</sup> given an observed history, yO. Next, q MSE-AR <sup>322</sup> is a conditional Gaussian autoregressive model that is trained to minimize a <sup>323</sup> mean-squared error based objective. This model is in the same family as conditional Gaussian models that are trained for maximum likelihood, but differ in that q MSE-AR <sup>324</sup> can be though of parameterizing <sup>325</sup> the mean of each transition distribution whereas maximum likelihood models parameterize both the <sup>326</sup> mean and covariance. Overall, the models that we have developed can be seen as mean-squared <sup>327</sup> error based time series models for probabilistic forecasting where the uncertainty in the models only <sup>328</sup> depend on the time in between observations and not the observations themselves.

|        |          |      | Brusselator | Double | Pendulum |       | FitzHugh |       | Lorenz |      | Lotka  | Van   | der Pol |
|--------|----------|------|-------------|--------|----------|-------|----------|-------|--------|------|--------|-------|---------|
| MSE    |          | 3.04 | ± 0.69      | 9.03   | ± 0.34   | 27.75 | ± 4.50   | 5.91  | ± 0.60 | 2.16 | ± 1.18 | -0.77 | ± 0.01  |
| AR-MSE |          | 0.49 | ± 0.18      | 0.61   | ± 0.02   | 15.08 | ± 1.18   | 8.82  | ± 0.29 | 0.12 | ± 0.25 | -0.59 | ± 0.01  |
| AR-MLE | (Latent) | 3.39 | ± 1.91      | 0.43   | ± 0.01   | 13.10 | ± 2.48   | 8.49  | ± 1.05 | 0.23 | ± 0.27 | -0.70 | ± 0.00  |
| AR-MLE | (Obs.)   | 3.79 | ± 2.05      | 0.42   | ± 0.01   | 13.35 | ± 2.47   | 7.77  | ± 0.76 | 0.11 | ± 0.32 | -0.70 | ± 0.00  |
| FBGM   | (Latent) | 2.06 | ± 1.12      | 0.56   | ± 0.03   | 6.15  | ± 0.75   | 12.11 | ± 0.80 | 0.17 | ± 0.42 | -0.69 | ± 0.00  |
| FBGM   | (Obs.)   | 0.93 | ± 0.29      | 0.51   | ± 0.01   | 11.67 | ± 1.80   | 5.28  | ± 0.50 | 0.47 | ± 0.67 | -0.71 | ± 0.00  |

(a) Negative log likelihood (lower is better)

|        |          |      | Brusselator | Double | Pendulum |      | FitzHugh |      | Lorenz |      | Lotka  | Van  | der Pol |
|--------|----------|------|-------------|--------|----------|------|----------|------|--------|------|--------|------|---------|
| MSE    |          | 0.56 | ± 0.02      | 0.99   | ± 0.00   | 2.15 | ± 0.16   | 1.09 | ± 0.01 | 0.50 | ± 0.02 | 0.48 | ± 0.00  |
| AR-MSE |          | 0.59 | ± 0.01      | 1.16   | ± 0.01   | 3.58 | ± 0.27   | 1.25 | ± 0.01 | 0.55 | ± 0.03 | 0.52 | ± 0.00  |
| AR-MLE | (Latent) | 0.65 | ± 0.04      | 1.27   | ± 0.01   | 2.32 | ± 0.17   | 1.26 | ± 0.03 | 0.59 | ± 0.03 | 0.52 | ± 0.01  |
| AR-MLE | (Obs.)   | 0.66 | ± 0.05      | 1.27   | ± 0.01   | 2.37 | ± 0.13   | 1.26 | ± 0.04 | 0.58 | ± 0.03 | 0.52 | ± 0.01  |
| FBGM   | (Latent) | 0.62 | ± 0.05      | 1.20   | ± 0.01   | 2.34 | ± 0.17   | 1.09 | ± 0.03 | 0.55 | ± 0.03 | 0.49 | ± 0.01  |
| FBGM   | (Obs.)   | 0.64 | ± 0.02      | 1.17   | ± 0.01   | 2.29 | ± 0.15   | 1.08 | ± 0.02 | 0.55 | ± 0.03 | 0.51 | ± 0.00  |

(b) Normalized root mean squared error (lower is better)

Table 1: Evaluation metrics for our models (MSE and AR-MSE) for probabilistic forecasting compared to baseline models trained in both the latent and data spaces.

# <sup>329</sup> 4 Experiments

<sup>330</sup> We compare the performance of our models versus other approaches to time series modeling in latent <sup>331</sup> probabilistic forecasting on dynamical system datasets. We created 6 synthetic datasets representing <sup>332</sup> noisy observations of dynamical systems. Our models used a Wiener velocity model as our base SDE and emission potentials of the form ϕ(x<sup>t</sup><sup>k</sup> |θ<sup>t</sup><sup>k</sup> (y<sup>τ</sup>1:<sup>N</sup> )) ∝ N(y<sup>t</sup><sup>k</sup> |x<sup>t</sup><sup>k</sup> , σ<sup>2</sup> I). Our models, q MSE <sup>333</sup> and q MSE-AR <sup>334</sup> , and the baseline models were trained to approximate the probabilistic forecasting distribution p(x<sup>t</sup>k+1:<sup>N</sup> |x<sup>t</sup>1:<sup>k</sup> <sup>335</sup> , yO). See Appendix [I](#page-33-0) for details about the datasets, parameters used for stochastic interpolation and other implementation details. Our models, q MSE and q MSE-AR <sup>336</sup> , were each trained <sup>337</sup> using mean squared error to learn their respective Bayes estimators. We used a non-autoregressive <sup>338</sup> FBGM trained with flow-matching and a conditional Gaussian chain trained for maximum likelihood as our baselines. We trained each of these baselines in two ways to learn p(x<sup>t</sup>k+1:<sup>N</sup> |x<sup>t</sup>1:<sup>k</sup> <sup>339</sup> , yO). First, <sup>340</sup> we trained these baseline models to learn the latent distribution directly by learning directly from <sup>341</sup> samples from p(x<sup>t</sup>1:<sup>N</sup> |y<sup>τ</sup>1:<sup>N</sup> ). Second, we trained these models in the observation space to learn <sup>342</sup> p(y<sup>U</sup> |yO) directly, and at test time, produced latent samples x<sup>t</sup>k+1:<sup>N</sup> by first sampling y<sup>U</sup> using yO, <sup>343</sup> and then sampling from the stochastic interpolator using the full sequence (yO, y<sup>U</sup> ). For all of the autoregressive models, instead of learning the distribution of the first point p(x<sup>t</sup>k+1 <sup>344</sup> |yO), we produced <sup>345</sup> a heuristic sample by sampling from the stochastic interpolant that is only conditioned on yO. We <sup>346</sup> always chose tk+1 to be a time contained in O in order for this heuristic to give reasonable samples. <sup>347</sup> For each model, we trained using 5 different seeds and report the (empirical) negative log likelihood <sup>348</sup> and normalized root mean squared error of samples from the true distribution, p(x<sup>t</sup>k+1:<sup>N</sup> |y<sup>U</sup> ), using <sup>349</sup> 32 sampled trajectories from each model, averaged over each dimension and time step. In all of our <sup>350</sup> models, we used a one layer recurrent neural network with a GRU cell as we found that this model <sup>351</sup> had sufficient model capacity to represent our data. Our results are displayed in Table [1.](#page-8-0) We can see <sup>352</sup> that the AR

# <sup>353</sup> 5 Conclusion

 We showed how to generalize the elements that comprise flow-based generative models to the time series setting and uncovered a discrete time version of these models that shares convenient properties that FBGMs possess, including a closed form solution and Bayes estimator parameters. Our framework also encapsulates other existing time series models, including MSE based non- probabilistic forecasters and conditional Gaussian autoregressive models. This unified perspective sheds light into the role that FBGMs can play in time series.

# References


[1] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text- conditional image generation with clip latents. *arXiv preprint arXiv:2204.06125*, 1(2):3, 2022. Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. *arXiv preprint arXiv:2307.01952*, 2023. Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. *Advances in neural information processing systems*, 35:36479–36494, 2022. Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. *Advances in neural information processing systems*, 34:8780–8794, 2021. Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. *arXiv preprint arXiv:2207.12598*, 2022. Guan-Horng Liu, Yaron Lipman, Maximilian Nickel, Brian Karrer, Evangelos Theodorou, and Ricky T. Q. Chen. Generalized schrödinger bridge matching. In *The Twelfth International Conference on Learning Representations*, 2024. URL <https://openreview.net/forum?id=SoismgeX7z>. Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, and Ricky TQ Chen. Adjoint matching: Fine-tuning flow and diffusion generative models with memoryless stochastic optimal control. *arXiv preprint arXiv:2409.08861*, 2024. Aaron Havens, Benjamin Kurt Miller, Bing Yan, Carles Domingo-Enrich, Anuroop Sriram, Brandon Wood, Daniel Levine, Bin Hu, Brandon Amos, Brian Karrer, et al. Adjoint sampling: Highly scalable diffusion samplers via adjoint matching. *arXiv preprint arXiv:2504.11713*, 2025. Valentin De Bortoli, Guan-Horng Liu, Tianrong Chen, Evangelos A Theodorou, and Weilie Nie. Augmented bridge matching. *arXiv preprint arXiv:2311.06978*, 2023. Yifan Chen, Mark Goldstein, Mengjian Hua, Michael S. Albergo, Nicholas M. Boffi, and Eric Vanden-Eijnden. Probabilistic forecasting with stochastic interpolants and föllmer processes, 2024a. Ella Tamir, Najwa Laabid, Markus Heinonen, Vikas Garg, and Arno Solin. Conditional flow matching for time series modelling. In *ICML 2024 Workshop on Structured Probabilistic Inference* {\*&*} *Generative Modeling*, 2024. Byoungwoo Park, Hyungi Lee, and Juho Lee. Efficient modeling of irregular time-series with stochas- tic optimal control. In *NeurIPS 2024 Workshop on Bayesian Decision-making and Uncertainty*, 2024. URL <https://openreview.net/forum?id=KRtuDGFJzu>. Yu Chen, Marin Biloš, Sarthak Mittal, Wei Deng, Kashif Rasul, and Anderson Schneider. Recurrent interpolants for probabilistic time series prediction. *arXiv preprint arXiv:2409.11684*, 2024b. Yiyuan Yang, Ming Jin, Haomin Wen, Chaoli Zhang, Yuxuan Liang, Lintao Ma, Yi Wang, Chenghao Liu, Bin Yang, Zenglin Xu, et al. A survey on diffusion models for time series and spatio-temporal data. *arXiv preprint arXiv:2404.18886*, 2024. Caspar Meijer and Lydia Y. Chen. The rise of diffusion models in time-series forecasting, 2024. Michael Samuel Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In *The Eleventh International Conference on Learning Representations*, 2023. URL <https://arxiv.org/abs/2209.15571>. Yuyang Shi, Valentin De Bortoli, Andrew Campbell, and Arnaud Doucet. Diffusion schrödinger bridge matching. *Advances in Neural Information Processing Systems*, 36, 2024. Pascal Vincent. A connection between score matching and denoising autoencoders. *Neural computa-tion*, 23(7):1661–1674, 2011.

[2] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In *International Conference on Learning Representations*, 2021. URL [https://openreview.net/forum?id=](https://openreview.net/forum?id=PxTIG12RRHS) [PxTIG12RRHS](https://openreview.net/forum?id=PxTIG12RRHS). Edwin T Jaynes. *Probability theory: The logic of science*. Cambridge university press, 2003. Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In *The Eleventh International Conference on Learning Repre- sentations*, 2023. URL <https://openreview.net/forum?id=PqvMRDCJT9t>. Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In *The Eleventh International Conference on Learning Representations*, 2023. URL <https://openreview.net/forum?id=XVjTT1nw5z>. Aram-Alexandre Pooladian, Heli Ben-Hamu, Carles Domingo-Enrich, Brandon Amos, Yaron Lipman, and Ricky T. Q. Chen. Multisample flow matching: Straightening flows with mini- batch couplings. In *International Conference on Machine Learning*, 2023. URL [https:](https://api.semanticscholar.org/CorpusID:258418096) [//api.semanticscholar.org/CorpusID:258418096](https://api.semanticscholar.org/CorpusID:258418096). John Lafferty, Andrew McCallum, Fernando Pereira, et al. Conditional random fields: Probabilistic models for segmenting and labeling sequence data. In *Icml*, volume 1, page 3. Williamstown, MA, 2001. Charles Sutton, Andrew McCallum, et al. An introduction to conditional random fields. *Foundations and Trends® in Machine Learning*, 4(4):267–373, 2012. Simo Särkkä and Arno Solin. *Applied stochastic differential equations*, volume 10. Cambridge University Press, 2019. Raghav Singhal, Mark Goldstein, and Rajesh Ranganath. Where to diffuse, how to diffuse, and how to get back: Automated learning for multivariate diffusions. In *The Eleventh International Conference on Learning Representations*, 2023. URL <https://openreview.net/forum?id=osei3IzUia>. Simo Särkkä et al. *Recursive Bayesian inference on stochastic differential equations*. Helsinki University of Technology, 2006. Syeda Sakira Hassan, Simo Särkkä, and Ángel F García-Fernández. Temporal parallelization of inference in hidden markov models. *IEEE Transactions on Signal Processing*, 69:4875–4887, 2021. Adrien Corenflos, Zheng Zhao, and Simo Särkkä. Gaussian process regression in logarithmic time. *arXiv preprint arXiv*, 2102, 2021. Jimmy T.H. Smith, Andrew Warrington, and Scott Linderman. Simplified state space layers for sequence modeling. In *The Eleventh International Conference on Learning Representations*, 2023. URL <https://openreview.net/forum?id=Ai8Hw3AXqks>. Calvin Luo. Understanding diffusion models: A unified perspective. *arXiv preprint arXiv:2208.11970*, 2022. [S](https://sander.ai/2023/07/20/perspectives.html)ander Dieleman. Perspectives on diffusion, 2023. URL [https://sander.ai/2023/07/20/](https://sander.ai/2023/07/20/perspectives.html) [perspectives.html](https://sander.ai/2023/07/20/perspectives.html). Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In *International conference on machine learning*, pages 2256–2265. PMLR, 2015. Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. *Advances in neural information processing systems*, 33:6840–6851, 2020. Tim Dockhorn, Arash Vahdat, and Karsten Kreis. Score-based generative modeling with critically- damped langevin diffusion. In *International Conference on Learning Representations*, 2022. URL <https://openreview.net/forum?id=CzceR82CYc>.

[3] Tianrong Chen, Jiatao Gu, Laurent Dinh, Evangelos Theodorou, Joshua M. Susskind, and Shuangfei Zhai. Generative modeling with phase stochastic bridge. In *The Twelfth International Conference on Learning Representations*, 2024c. URL <https://openreview.net/forum?id=tUtGjQEDd4>. Yaakov Bar-Shalom, X. Rong Li, and Thiagalingam Kirubarajan. *Estimation with Applications to Tracking and Navigation*. John Wiley & Sons, New York, 2001. ISBN 9780471221272. doi: 10.1002/0471221279. URL [https://onlinelibrary.wiley.com/doi/book/10.1002/](https://onlinelibrary.wiley.com/doi/book/10.1002/0471221279) [0471221279](https://onlinelibrary.wiley.com/doi/book/10.1002/0471221279). Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. *Advances in neural information processing systems*, 34:21696–21707, 2021. Marcel Kollovieh, Abdul Fatir Ansari, Michael Bohlke-Schneider, Jasper Zschiegner, Hao Wang, and Yuyang Bernie Wang. Predict, refine, synthesize: Self-guiding diffusion models for probabilistic time series forecasting. *Advances in Neural Information Processing Systems*, 36:28341–28364, 2023. Xinyu Yuan and Yan Qiao. Diffusion-TS: Interpretable diffusion for general time series generation. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https://](https://openreview.net/forum?id=4h1apFjO99) [openreview.net/forum?id=4h1apFjO99](https://openreview.net/forum?id=4h1apFjO99). Marcel Kollovieh, Marten Lienen, David Lüdke, Leo Schwinn, and Stephan Günnemann. Flow matching with gaussian process priors for probabilistic time series forecasting. In *The Thirteenth International Conference on Learning Representations*, 2025. URL [https://openreview.net/](https://openreview.net/forum?id=uxVBbSlKQ4) [forum?id=uxVBbSlKQ4](https://openreview.net/forum?id=uxVBbSlKQ4). Yang Hu, Xiao Wang, Lirong Wu, Huatian Zhang, Stan Z Li, Sheng Wang, and Tianlong Chen. Fm-ts: Flow matching for time series generation. *arXiv preprint arXiv:2411.07506*, 2024. Kashif Rasul, Calvin Seward, Ingmar Schuster, and Roland Vollgraf. Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting. In *International Conference on Machine Learning*, pages 8857–8868. PMLR, 2021. Macheng Shen and Chen Cheng. Neural sdes as a unified approach to continuous-domain sequence modeling. *arXiv preprint arXiv:2501.18871*, 2025. Ahmed El-Gazzar and Marcel van Gerven. Probabilistic forecasting via autoregressive flow matching. *arXiv preprint arXiv:2503.10375*, 2025. Matthew James Beal. *Variational algorithms for approximate Bayesian inference*. University of London, University College London (United Kingdom), 2003. Matthew James Johnson et al. *Bayesian time series models and scalable inference*. PhD thesis, Massachusetts Institute of Technology, 2014. Simo Särkkä and Ángel F García-Fernández. Temporal parallelization of bayesian smoothers. *IEEE Transactions on Automatic Control*, 66(1):299–306, 2020. Daphane Koller. *Probabilistic Graphical Models: Principles and Techniques*. The MIT Press, 2009. Bernt Øksendal and Bernt Øksendal. *Stochastic differential equations*. Springer, 2003. Rudolph Emil Kalman. A new approach to linear filtering and prediction problems. *Transactions of the ASME–Journal of Basic Engineering*, 82(Series D):35–45, 1960. H. E. Rauch, F. Tung, and C. T. Striebel. Maximum likelihood estimates of linear dynamic systems. *AIAA Journal*, 3(8):1445–1450, 1965. Emily Beth Fox. *Bayesian nonparametric learning of complex dynamical phenomena*. PhD thesis, Massachusetts Institute of Technology, 2009. Matthew Johnson and Scott Linderman. pylds: Bayesian inference for linear dynamical systems. <https://github.com/mattjj/pylds>, 2015. Accessed: 2025-05-07.
### A Appendix

 The appendix contains proofs and implementation details for the main paper. It is organized as follows:

 1. Related work Appendix [B](#page-12-0) 2. Background Appendix [C](#page-13-0) • Exponential family distributions Appendix [C.1](#page-13-1) • Mean field variational inference Appendix [C.2](#page-14-1) • Bayes estimation Appendix [C.3](#page-14-2) 3. Message passing [\(D\)](#page-15-0) • Sequential message passing [\(D.1\)](#page-15-1) • Parallel message passing [\(D.2\)](#page-16-0) • Basic probabilistic queries [\(D.4\)](#page-19-1) 4. Conditioned linear SDEs [\(E\)](#page-20-1) • Conditioned linear SDEs [\(E.1\)](#page-20-2) • Basic probabilistic queries [\(E.2\)](#page-21-0) • Corresponding probability flow ODE [\(E.3\)](#page-22-1) 5. Constrained mean field VI [\(F\)](#page-23-2) • Derivation [\(F.1\)](#page-23-3) • Bayes estimator equivariance [\(F.2\)](#page-25-0) • CMFVI time series models [\(F.3\)](#page-26-1) 6. Flow-based generative models [\(G\)](#page-27-0) • Score function of FBGMs [\(G.1\)](#page-27-3) • General form of Markovian projection SDE [\(G.2\)](#page-28-0) • General form of Markovian projection ODE [\(G.3\)](#page-29-1) 7. Message passing implementation details [\(H\)](#page-30-0) • Numerical stability considerations [\(H.1\)](#page-30-1) • Message passing pseudocode [\(H.2\)](#page-31-0) 8. Dataset details [\(I\)](#page-33-0) 9. Model implementation details [\(J\)](#page-33-1)

# B Related Work

 There are numerous perspectives on flow-based generative models [\[Luo, 2022,](#page-10-13) [Dieleman, 2023\]](#page-10-14) and even more variants of these models. At their core, these models start by constructing a stochastic process that starts at a prior distribution and ends at the data distribution. Diffusion models use progressive noising of data to build this map [\[Sohl-Dickstein et al., 2015,](#page-10-15) [Ho et al., 2020,](#page-10-16) [Song et al.,](#page-10-0) [2021\]](#page-10-0) via a simple SDE whose stationary distribution is Gaussian. On the other hand, flow-matching models [\[Liu et al., 2023,](#page-10-3) [Albergo and Vanden-Eijnden, 2023,](#page-9-15) [Lipman et al., 2023\]](#page-10-2) use a stochastic bridge to build this map by conditioning a simple SDE to start at a point in the prior distribution and end at the data distribution. The choice of simple SDE used in all of these models is a user-defined choice that typically is a linear SDE, such as variance preserving SDE [\[Song et al., 2021\]](#page-10-0), Brownian motion, Ornstein-Uhlenbeck process, and others, due to their tractability as Gaussian processes [\[Särkkä and Solin, 2019\]](#page-10-7), and is even used to construct more exotic latent SDEs such as critically damped langevin dynamics [\[Dockhorn et al., 2022,](#page-10-17) [Chen et al., 2024c\]](#page-11-0) or the Weiner velocity model [\[Bar-Shalom et al., 2001,](#page-11-1) [Särkkä et al., 2006\]](#page-10-9). In our paper, we abstract away these choices and generally consider using linear SDEs to construct the initial map between distributions. There are a few different ways to go from this initial stochastic process to a FBGM. A common way to construct a FBGM from this is construct and optimize and ELBO for the likelihood of data under this initial process [\[Kingma et al., 2021\]](#page-11-2). Alternatively, one can directly solve for the SDE whose marignal distribution is that of this initial process [\[Song et al., 2021,](#page-10-0) [Lipman et al., 2023\]](#page-10-2) or define it as the

[S](#page-9-8)DE whose path measure is as close as possible to the initial process [\[Shi et al., 2024,](#page-9-16) [De Bortoli](#page-9-8) [et al., 2023\]](#page-9-8) in terms of KL divergence, called the Markovian projection. We adopt the latter view over the ELBO view because it explicitly constructs a solution to the generative modeling problem and is available in closed form while this is hidden in the ELBO formulation and show that the solution to a mean field variational inference problem can be seen as an approximate discrete time counterpart.

 Flow-based generative models have been successfully applied to time series problems in a *non- [a](#page-11-6)utoregressive* fashion [\[Kollovieh et al., 2023,](#page-11-3) [Yuan and Qiao, 2024,](#page-11-4) [Kollovieh et al., 2025,](#page-11-5) [Hu](#page-11-6) [et al., 2024,](#page-11-6) [Yang et al., 2024,](#page-9-13) [Meijer and Chen, 2024\]](#page-9-14). These models transform the time series generative modeling problem into the standard generative modeling problem used in image generation by treating each time series as a single vector by concatenating all times together, and then learning a map from a Gaussian vector of the same size to the data vector. These approaches can be conditioned [u](#page-11-3)sing guidance [\[Rasul et al., 2021,](#page-11-7) [Dhariwal and Nichol, 2021,](#page-9-3) [Ho and Salimans, 2022,](#page-9-4) [Kollovieh](#page-11-3) [et al., 2023\]](#page-11-3) which allows them to perform tasks such as forecasting and imputation. Our approach differs from these in that we construct autoregressive models.

 The class of models most relevant to our paper are autoregressive neural SDEs that are trained using principles from flow-based generative models. [\[Chen et al., 2024a\]](#page-9-9) uses a FÃ˝ullmer process to model the transition distributions of the distribution of time series data, which is the same approach that we adopt in our Neural SDE model. [\[Park et al., 2024\]](#page-9-11) also learns a similar latent Neural SDE model that uses a similar form of soft conditioning as us (through the use of emission potentials), and is trained to maximize the likelihood of data. [\[Tamir et al., 2024\]](#page-9-10) is also similar where they perform stochastic interpolation using Gaussian processes and perform inference with Kalman smoothing as well, which is a form of message passing. Finally, [\[Shen and Cheng, 2025\]](#page-11-8) learns a more general SDE to learn the distribution of time series data where the diffusion coefficient is not independent of the current state and also maximize the likelihood of data. These related papers are all related to the Neural SDE that we describe in our paper. Our main contributions are centered around investigating how to apply the approach used to construct these continuous time models for creating similar discrete time models. [\[El-Gazzar and van Gerven, 2025\]](#page-11-9) used flow matching to learn the next state distribution of time series data, but did not learn a FÃ˝ullmer process for this task and instead learned to transform a Gaussian into the next state distribution.

# C Background

# C.1 Exponential family distributions

 Our findings can be most easily written using exponential family distributions. Although we restrict our attention to Gaussian distributions, the form of our results are most readable in natural parameter space.

 Definition 3 (Exponential family distribution). *An probability distribution is in the exponential family if its density function can be written in the following form:*

$$p(x|\theta) = \exp\{\langle t(x), \theta \rangle - A(\theta)\} \quad (20)$$

*where* t(x) *is called the sufficient statistic,* θ *the natural parameter and* A(θ) *the partition function.*

 The member of this family that we will use is the multivariate Gaussian distribution. A multivariate Gaussian with mean µ and covariance matrix Σ has the sufficient statistic t(x) = (x, xx<sup>T</sup> ) and natural parameters θ = (− Σ −1 , Σ <sup>−</sup><sup>1</sup>µ). In practice, it is more convenient to drop the − scaling term and work with the parameters (J, h) = (−Σ −1 , Σ −1 µ), where J is the precision matrix of the distribution. While these are not exactly the natural parameters, we will refer to them as so. Throughout this paper, we will work with unnormalized Gaussian distributions, which we call "Gaussian potentials". We use the notation ϕ(x|θ) to denote a Gaussian potential function over x with natural parameters θ. A convenient property of the natural parameter form is that the score function takes a simple form.

$$\nabla \log \phi(x|\theta) = Jx - h \quad (21)$$

 Another Gaussian distribution that we will use extensively is the Gaussian transition distribution. We write ϕk+1|k(xk+1|xk) = N(xk+1|Ax<sup>k</sup> + u, Σ) to denote the Gaussian transition distribution from x<sup>k</sup> to xk+1 with state transition matrix A, bias vector u and covariance matrix Σ.

### <sup>596</sup> C.2 Mean field variational inference

 Mean field variational inference is an approximate inference algorithm for probabilistic models. It's main feature is that it's solution is available in a simple closed form expression. Let p(x, θ) be a joint distribution over x and θ. The mean field variational problem is to find distributions, qx(x) and qθ(θ) that minimize the KL divergence between qx(x)qθ(θ) and p(x, θ).

<sup>601</sup> Proposition 8 (Mean field variational inference for CRFs). *Let* p(θ) *be a distribution over* θ*,* p(x|θ) <sup>602</sup> *be the CRF in Definition [1](#page-3-0) and* p(x, θ) = p(θ)p(x|θ) *be the joint distribution over* x *and* θ*. Then the* <sup>603</sup> *solutions to*

$$\text{argmin}_{q_x(x), q_\theta(\theta)} \text{KL} [q_x(x)q_\theta(\theta)|p(x, \theta)] \quad (22)$$

<sup>604</sup> *will satisfy:*

$$q_x(x) \propto \exp\{\mathbb{E}_{q_\theta}(\theta) [\log p(x|\theta)]\} \quad (23)$$

$$q_\theta(\theta) \propto \exp\{\mathbb{E}_{q_x(x)} [\log p(\theta|x)]\} \quad (24)$$

 See [\[Beal, 2003\]](#page-11-10) for a proof. Typical use cases of mean field VI use tractable classes of distributions for p(θ) and p(x|θ) so that one can perform EM style, alternating updates to obtain the optimal q distributions [\[Beal, 2003,](#page-11-10) [Johnson et al., 2014\]](#page-11-11). However, in our setting, we will use mean field VI differently. We will assume nothing about the form of p(θ), but will constrain the variational problem by fixing qθ(θ) = p(θ).

### <sup>610</sup> C.3 Bayes estimation

Lemma 1 (Bayes estimate of parameter). *Let* p(z, θ) *be a joint distribution and let* θ ∗ <sup>611</sup> (z) *be the* <sup>612</sup> *Bayes estimate of* θ *based on* z *under the squared error risk. Then the Bayes estimate takes the* <sup>613</sup> *following two forms:*

$$\theta^*(z) = \mathbb{E}_{p(\theta|z)}[\theta] = \operatorname{argmin}_{f(z)} \mathbb{E}_{p(z,\theta)} [\|f(z) - \theta\|^2] \quad (25)$$

<sup>614</sup> *Proof.* Let L[f] be the loss function defined as follows:

$$\mathcal{L}[f] = \mathbb{E}_{p(z)} [\|f(z) - \theta^*(z)\|^2]$$

Clearly, the minimizer of L[f] is θ ∗ <sup>615</sup> (z). With a bit of rearranging and using Bayes rule, we can <sup>616</sup> rewrite L[f] as follows:

$$\begin{aligned}\mathcal{L}[f] &= \mathbb{E}_{p(z)} [\|f(z) - \theta^*(z)\|^2] \\ &= \mathbb{E}_{p(z)} [\|f(z)\|^2] - 2\mathbb{E}_{p(z)} [\langle f(z), \theta^*(z) \rangle] + \underbrace{\mathbb{E}_{p(z)} [\|\theta^*(z)\|^2]}_{\text{const. w.r.t. } f} \\ &= \mathbb{E}_{p(z,\theta)} [\|f(z)\|^2] - 2\mathbb{E}_{p(z)} [\langle f(z), \mathbb{E}_{p(\theta|z)}[\theta] \rangle] + \text{const.} \\ &= \mathbb{E}_{p(z,\theta)} [\|f(z)\|^2] - 2\mathbb{E}_{p(z,\theta)} [\langle f(z), \theta \rangle] + \text{const.} \\ &\quad (\text{complete the square}) \\ &= \mathbb{E}_{p(z,\theta)} [\|f(z) - \theta\|^2] - \underbrace{\mathbb{E}_{p(z,\theta)} [\|\theta\|^2]}_{\text{const. w.r.t. } f} + \text{const.}\end{aligned}$$

The minimizer of L[f] is unaffected by the constant terms, and so we have that θ ∗ (z) = <sup>E</sup>p(θ|z) <sup>617</sup> [θ] is <sup>618</sup> the solution to

$$\operatorname{argmin}_{f(z)} \mathbb{E}_{p(z,\theta)} [\|\theta - f(z)\|^2]$$

# D Message passing

 In this section we will review message passing and identify the key operations that are needed to perform message passing updates. We defer the discussion of numerically stable implementations of these operations to Appendix [H.](#page-30-0) First we'll identify the key operations that are needed to perform message passing updates for the backward messages and then show how these operations can be used to perform message passing updates for the forward messages.

 At a high level, the sequential and parallel message passing algorithms are variable elimination algorithms that eliminate different variables of the chain structured graph. The sequential algorithms operates on individual nodes and begins at one of the ends of the chain and sequentially eliminate variable at the end of the chain, whereas the parallel algorithm operates on pairs of nodes and eliminates the middle variable of the pair. For example, a rough sketch of the sequential elimination process looks like (0), 1, 2, 3, 4 → (1), 2, 3, 4 → (2), 3, 4 → (3), 4 → (4), where the parentheses indicate the current node that is being processed. On the other hand, the parallel algorithm looks like (0, 1), 2, 3, 4 → (0, 2), 3, 4 → (0, 3), 4 → (0, 4).

# D.1 Sequential message passing

 The sequential message passing updates for the backward messages can be written using the following recurrence relation:

$$\phi(x_{k-1}|\beta_{k-1}) = \int \phi_{k|k-1}(x_k|x_{k-1})\phi(x_k|\theta_k)\phi(x_k|\beta_k)dx_k, \quad \beta_N = 0 \quad (26)$$

 See Appendix [H.3](#page-32-0) for pseudocode. There are two operations on Gaussians that are needed to perform these updates. The first is a "multiply" operation that takes two potential functions and returns a new potential function, and the second is an "update" operation that absorbs a potential function into a transition function.

 Definition 4 (Multiply). *Let* ϕ1(x) *and* ϕ2(x) *be potential functions over the same variable. Then the "multiply" operation is defined as*

$$\phi_1(x)\phi_2(x) \mapsto \hat{\phi}(x) \quad (27)$$

 When ϕ1(x) and ϕ2(x) are parameterized using natural parameters, then the multiply operation simply adds the natural parameters, i.e. if θ<sup>1</sup> and θ<sup>2</sup> are the natural parameters of ϕ1(x) and ϕ2(x), then ϕ1(x|θ1)ϕ2(x|θ2) 7→ ϕ1(x|θ<sup>1</sup> + θ2). We used this property to write the sequential message passing updates for the backward messages ??. We do note that when one uses a different parameterization, the multiply operation may look different. We will examples of this in Appendix [H.](#page-30-0)

 The second operation is the "update" operation, which absorbs a potential function into a transition function. This operation is what handles the integral in the recurrence relation.

Definition 5 (Update). *Let* ϕ(y|x) *be a transition function and* ϕ(y) *be a potential function over the first variable. Then the "update" operation is defined as*

$$\phi(y)\phi_{y|x}(y|x) \mapsto \hat{\phi}_{y|x}(y|x)\hat{\phi}(x) \quad (28)$$

*where* ϕˆ <sup>y</sup>|x(y|x) *and* <sup>ϕ</sup><sup>ˆ</sup> (x) *are a new transition function and potential function, respectively.*

 Essentially, the update operation performs a change of variables of the coupling of x and y on the LHS. Furthermore, when the terms of the LHS are Gaussian, then the terms of the RHS are also Gaussian. This allows us to perform the update operation in closed form (see Appendix [H\)](#page-30-0).

 The multiply and update operations are sufficient to perform the sequential message passing updates for the backward messages. For example, the backward message passing updates can be written as:

$$\int \phi_{k|k-1}(x_k|x_{k-1}) \underbrace{\phi(x_k|\theta_k)\phi(x_k|\beta_k)}_{\text{multiply } \rightarrow \phi(x_k|\theta_k+\beta_k)} dx_k \quad (29)$$

$$= \int \underbrace{\phi(x_k | \theta_k + \beta_k) \phi_{k|k-1}(x_k | x_{k-1})}_{\text{update} \rightarrow \hat{\phi}_{k|k-1}(x_k | x_{k-1}) \phi(x_{k-1} | \beta_{k-1})} dx_k \quad (30)$$

$$= \underbrace{\int \hat{\phi}_k|_{k-1}(x_k|x_{k-1})dx_k \phi(x_{k-1}|\beta_{k-1})}_{\text{transition integrates to 1}} \quad (31)$$

$$= \phi(x_{k-1}|\beta_{k-1}) \quad (32)$$

<sup>658</sup> The forward messages can be computed in a similar manner. The forward messages are given by:

$$\phi(x_{k+1}|\alpha_{k+1}) = \int \phi_{k+1|k}(x_{k+1}|x_k)\phi(x_k|\theta_k)\phi(x_k|\alpha_k)dx_k, \quad \alpha_1 = 0 \quad (33)$$

<sup>659</sup> To find the forward messages, we can exploit the fact that our transition functions are Gaussian and <sup>660</sup> can therefore be reversed. This means that given a transition ϕ(y|x), we can find a reversed transition ϕ T <sup>661</sup> (x|y) that evaluates to the same value as ϕ(y|x) for all x, y

<sup>662</sup> Definition 6 (Reversed transition). *Let* ϕ(y|x) *be a transition function. Then the reversed transition* <sup>663</sup> *is defined as*

$$\phi^T(x|y) = \phi(y|x) \quad (34)$$

*so that* ϕ T (x|y) = ϕ(y|x) *for all* x, y *and* R ϕ T (x|y)dx = R <sup>664</sup> ϕ(y|x)dx = 1*.*

<sup>665</sup> Using this reverse operation, we can simply reverse the transition distributions and then find the <sup>666</sup> forward messages by using the same recurrence relation as for the backward messages:

$$\int \underbrace{\phi_{k+1|k}(x_{k+1}|x_k)}_{\text{reverse}} \underbrace{\phi(x_k|\theta_k)\phi(x_k|\alpha_k)}_{\text{multiply } \rightarrow \phi(x_k|\theta_k + \alpha_k)} dx_k \quad (35)$$

$$= \int \underbrace{\phi^T(x_k|x_{k+1})\phi(x_k|\theta_k + \alpha_k)}_{\text{update} \rightarrow \hat{\phi}^T(x_k|x_{k+1})\phi(x_{k+1}|\alpha_{k+1})} dx_k \quad (36)$$

$$= \underbrace{\int \hat{\phi}^T(x_k|x_{k+1}) dx_k \phi(x_{k+1}|\alpha_{k+1})}_{\text{transition integrates to 1}} \quad (37)$$

$$= \phi(x_{k+1} | \alpha_{k+1}) \quad (38)$$

 These message passing updates can be computed in O(N) time using the the multiply, update and reverse operations. However, there is a more efficient way to compute the forward messages using the parallel scan algorithm [\[Särkkä and García-Fernández, 2020\]](#page-11-12) that reduces the complexity to O(log N) on parallel compute. We will describe this algorithm in Appendix [D.2.](#page-16-0)

### <sup>671</sup> D.2 Parallel message passing

 In this section we will use slightly different notation to describe the parallel message passing algorithm. We will avoid writing out the parameters of our potential functions and call them by their parameter name. For example, instead of writing ϕ(xk|θk), we will write ϕk(xk) and instead of writing ϕ(xk|βk), we will write β(xk).

 The building block of the parallel message passing algorithm [Särkkä and García-Fernández](#page-11-12) [\[2020\]](#page-11-12) is an unnormalized potential function over two variables, which we denote by Ψ(y, x). We assume that Ψ(y, x) can be decomposed into a (normalized) transition distribution and an unnormalized potential function:

$$\Psi(y, x) = \Psi(y|x)\Psi(x) \quad (39)$$

 Whenever we write Ψ(y|x), we are referring to a valid conditional probability distribution ( R Ψ(y|x)dy = 1). Since Ψ(y, x) is jointly Gaussian over x and y, we are able to integrate out variables in x and y and can also combine neighboring potentials into a new Gaussian potential. These properties allow us to construct a chain operation over potentials that combines neighboring potentials and then integrates out the common variable. We denote this chain operation by ⊗:

$$\Psi(y, x) := \int \Psi(y, z) \Psi(z, x) dz =: \Psi(y, z) \otimes \Psi(z, x) \quad (40)$$

<sup>685</sup> An important property of the chain operation is that it is associative due to the fact that we can swap <sup>686</sup> the order or integration (we will prove this in Appendix [D.3\)](#page-17-0).

<sup>687</sup> A useful perspective of this chain operation is that it amounts to performing variable elimination on <sup>688</sup> the graph defined by the potentials, i.e. performs some sort of message passing [\[Koller, 2009\]](#page-11-13). With <sup>689</sup> this in mind, we can perform message passing by constructing the appropriate joint potentials:

<sup>690</sup> Proposition 9 (Parallel messages). *Let* ϕk+1|<sup>k</sup> *and* ϕ<sup>k</sup> *be the potential functions for the CRF in* <sup>691</sup> *Definition [1](#page-3-0) and* α *and* β *be the messages defined in Eqs.* [\(26\)](#page-15-2) *and* [\(33\)](#page-16-1)*. Then*

$$\alpha_k(x_k) = \int \Psi_{1:k}^{\text{fwd}}(x_k, x_1) dx_1 \quad \text{and} \quad \beta_k(x_k) = \int \Psi_{k:N}^{\text{bwd}}(x_N | x_k) dx_N \quad (41)$$

<sup>692</sup> *where*

$$\Psi_{1:k}^{fwd}(x_k, x_1) = \bigotimes_{i=1}^{k-1} \phi_{i+1|i}(x_{i+1}|x_i)\phi_i(x_i) \quad (42)$$

$$\text{and } \Psi_{k:N}^{bwd}(x_N|x_k) = \bigotimes_{i=N-1}^k \phi_{i+1|i}(x_{i+1}|x_i)\phi_{i+1}(x_{i+1}) \quad (43)$$

 See appendix Appendix [D.3](#page-17-1) for a proof and ?? for pseudocode. Since ⊗ is associative, we can evaluate Eq. [\(42\)](#page-17-2) in O(log N) time using the parallel scan algorithm [\[Särkkä and García-Fernández,](#page-11-12) [2020\]](#page-11-12). The rough idea is that on parallel compute, one can, in parallel, chain together consecutive pairs of potentials and then recurse on these new chained potentials in order to eventually chain the entire sequence. We provide pseudocode for this a special case of this algorithm in Appendix [H.3.](#page-33-2) Ψfwd 1:k (xk, x1) and Ψbwd <sup>698</sup> <sup>k</sup>:<sup>N</sup> (x<sup>N</sup> |xk) can be thought of as the result of marginalization over the variables between x<sup>1</sup> and x<sup>k</sup> and x<sup>k</sup> and x<sup>N</sup> , respectively.

### <sup>700</sup> D.3 Chain operation

<sup>701</sup> Recall that the chain operation is defined in Eq. [\(40\)](#page-16-2) as

$$\Psi(y, x) := \int \Psi(y, z) \Psi(z, x) dz =: \Psi(y, z) \otimes \Psi(z, x) \quad (44)$$

<sup>702</sup> To see that it is associative, we need to check that Ψ(y, z) ⊗ (Ψ(z, x) ⊗ Ψ(x, w)) = <sup>703</sup> (Ψ(y, z) ⊗ Ψ(z, x)) ⊗ Ψ(x, w)

$$\Psi(y, z) \otimes (\Psi(z, x) \otimes \Psi(x, w)) = \int \Psi(y, z) \left( \int \Psi(z, x) \Psi(x, w) dx \right) dz \quad (45)$$

$$= \int \int \Psi(y, z) \Psi(z, x) \Psi(x, w) dx dz \quad (46)$$

$$= \int \left( \int \Psi(y, z) \Psi(z, x) dz \right) \Psi(x, w) dw \quad (47)$$

$$= (\Psi(y, z) \otimes \Psi(z, x)) \otimes \Psi(x, w) \quad (48)$$

<sup>704</sup> Proposition 10 (Parallel messages). *Let* ϕk+1|<sup>k</sup> *and* ϕ<sup>k</sup> *be the potential functions for the CRF in* <sup>705</sup> *Definition [1](#page-3-0) and* α *and* β *be the messages defined in Eqs.* [\(26\)](#page-15-2) *and* [\(33\)](#page-16-1)*. Then*

$$\alpha_k(x_k) = \int \Psi_{1:k}^{\text{fwd}}(x_k, x_1) dx_1 \quad \text{and} \quad \beta_k(x_k) = \int \Psi_{k:N}^{\text{bwd}}(x_N | x_k) dx_N \quad (49)$$

<sup>706</sup> *where*

$$\Psi_{1:k}^{\text{fwd}}(x_k, x_1) = \bigotimes_{i=1}^{k-1} \phi_{i+1|i}(x_{i+1}|x_i)\phi_i(x_i) \quad (50)$$

$$\text{and } \Psi_{k:N}^{bwd}(x_N|x_k) = \bigotimes_{i=N-1}^k \phi_{i+1|i}(x_{i+1}|x_i) \phi_{i+1}(x_{i+1}) \quad (51)$$

<sup>707</sup> *Proof.* First for notational clarity, define

$$\Psi_{i+1,i}^{\text{bwd}}(x_{i+1}|x_i) = \phi_{i+1|i}(x_{i+1}|x_i)\phi_{i+1}(x_{i+1}) \quad \text{and} \quad \Psi_{i+1,i}^{\text{fwd}}(x_{i+1}, x_i) = \phi_{i+1|i}(x_{i+1}|x_i)\phi_i(x_i) \quad (52)$$

<sup>708</sup> We can compute the cumulative potentials as follows:

$$\begin{aligned}
\Psi_{k:N}^{\text{bwd}}(x_N|x_k) &= \bigotimes_{i=N-1}^k \Psi_{i+1,i}^{\text{bwd}}(x_{i+1}|x_i) \\
&= \Psi_{N:N-1}^{\text{bwd}}(x_N|x_{N-1}) \otimes \Psi_{N-1:N-2}^{\text{bwd}}(x_{N-1}|x_{N-2}) \otimes \cdots \otimes \Psi_{k+1:k}^{\text{bwd}}(x_{k+1}|x_k) \\
&= \int \Psi_{N:N-1}^{\text{bwd}}(x_N|x_{N-1}) \int \Psi_{N-1:N-2}^{\text{bwd}}(x_{N-1}|x_{N-2}) dx_{N-1} \int \Psi_{N-2:N-3}^{\text{bwd}}(x_{N-2}|x_{N-3}) dx_{N-2} \cdots dx_{k+1} \\
&= \int \cdots \int \prod_{i=k}^{N-1} \Psi_{i:i+1}^{\text{bwd}}(x_{i+1}|x_i) dx_{N-1} \cdots dx_{k+1} \tag{55}
\end{aligned}$$

<sup>709</sup> And similarly for the forward potentials:

$$\Psi_{1:k}^{\text{fwd}}(x_k, x_1) = \bigotimes_{i=1}^{k-1} \Psi_{i+1,i}^{\text{fwd}}(x_{i+1}, x_i) \quad (57)$$

$$= \int \cdots \int \prod_{i=1}^{k-1} \Psi_{i+1,i}^{\text{fwd}}(x_{i+1}, x_i) dx_2 \cdots dx_{k-1} \quad (58)$$

<sup>710</sup> Next, we can rewrite the joint distribution of the CRF in a similar form:

$$p(x_{1:N}) = \prod_{k=1}^{N-1} \phi_{k+1|k}(x_{k+1}|x_k) \prod_{k=1}^N \phi_k(x_k) \quad (59)$$

$$= \phi_k(x_k) \prod_{i=k}^{N-1} \Psi_{i+1,i}^{\text{bwd}}(x_{i+1} | x_i) \prod_{i=1}^{k-1} \Psi_{i+1,i}^{\text{fwd}}(x_{i+1}, x_i), \quad \forall k \in \{1, \dots, N\} \quad (60)$$

Then, integrating over the variables dx1, . . . , ˆdxk, . . . , dx<sup>N</sup> , where ˆ <sup>711</sup> dx<sup>k</sup> denotes that we are not <sup>712</sup> integrating over xk, completes the proof:

$$p(x_k) = \int \cdots \int p(x_{1:N}) dx_1 \dots d\hat{x}_k \dots dx_N \quad (61)$$

$$\propto \int \cdots \int \prod_{k=1}^{N-1} \phi_{k+1|k}(x_{k+1}|x_k) \prod_{k=1}^N \phi_k(x_k) dx_1 \dots d\hat{x}_k \dots dx_N \quad (62)$$

$$= \phi_k(x_k) \int \cdots \int \prod_{i=k}^{N-1} \Psi_{i+1,i}^{\text{bwd}}(x_{i+1} | x_i) \prod_{i=1}^k \Psi_{i+1,i}^{\text{fwd}}(x_{i+1}, x_i) dx_1 \dots \hat{d}x_k \dots dx_N \quad (63)$$

$$= \phi_k(x_k) \underbrace{\int \Psi_{k:N}^{\text{bwd}}(x_N | x_k) dx_N}_{\beta_k(x_k)} \underbrace{\int \Psi_{1:k}^{\text{fwd}}(x_k, x_1) dx_1}_{\alpha_k(x_k)} \quad (64)$$

<sup>713</sup> We can recognize the terms in the last equation as the forward and backward messages, which <sup>714</sup> completes the proof.

<sup>717</sup> Definition 7 (Message passing update operator). *Let* ϕk+1|k(xk+1, xk) *be a Gaussian transition* <sup>718</sup> *function and let* ϕ(xk+1|ηk+1) *be a Gaussian node potential with natural parameters* ηk+1*. Next* <sup>719</sup> *consider the message passing update:*

$$\phi(x_k | \eta_k) = \int \phi_{k+1 | k}(x_{k+1} | x_k) \phi(x_{k+1} | \eta_{k+1}) dx_{k+1} \quad (65)$$

<sup>720</sup> *The message passing update operator is denoted by* Φk,k+1(ηk+1) *and is defined to satisfy:*

$$\eta_k = \Phi_{k,k+1}(\eta_{k+1}) \quad (66)$$

<sup>721</sup> *In particular, the update rule for the backward messages is given by:*

$$\beta_k = \Phi_{k,k+1}(\beta_{k+1} + \theta_{k+1}) \quad (67)$$

Corollary 2 (Mixed parameterization update rule). *Let* ϕk+1|k(xk+1|xk) := N(xk+1|Axk+u, Σ) *be a Gaussian transition function and let* ϕ(xk+1|ηk+1) := N(xk+1|µk+1, J−<sup>1</sup> <sup>k</sup>+1 <sup>723</sup> ) *be a Gaussian node potential where* Jk+1 *is the precision matrix. If* η<sup>k</sup> *and* ηk+1 *represent the mean and precision matrix of a Gaussian distribution, then the update and marginalize operator is denoted by* Φk,k+1(ηk+1) *and is given by:*

$$\Phi_{k,k+1}(\mu_{k+1}, J_{k+1}) = \left( A^{-1}(\mu_{k+1} - u), \Phi_{k,k+1}^{(J)}(J_{k+1}) \right) \quad (68)$$

*where* Φ (J) k,k+1 <sup>727</sup> (Jk+1) *is a nonlinear function of* Jk+1*.*

<sup>728</sup> *Proof.* The result follows from Appendix [H.3.](#page-31-1)

# <sup>729</sup> D.4 Probabilistic queries

<sup>730</sup> The forward and backward messages can be used to compute the majority of the probabilistic queries <sup>731</sup> of interest on a CRF. Recall our definition of a CRF:

$$p(x_{1:N}|\theta) \propto \prod_{k=1}^{N-1} \phi_{k+1|k}(x_{k+1}|x_k) \prod_{k=1}^N \phi(x_k|\theta_k) \quad (69)$$

<sup>732</sup> Next we will describe two probabilistic queries of interest: the marginal distribution and the transition <sup>733</sup> distribution.

**Proposition 11** (Marginal distribution). 
$$p(x_k|\theta) = \phi(x_k|\theta_k + \alpha_k + \beta_k) \quad (70)$$

<sup>734</sup> *Proof.* The derivation is given in Eq. [\(61\)](#page-18-1). For completness, we will change notation:

$$p(x_k) = \phi_k(x_k)\beta_k(x_k)\alpha_k(x_k) \text{ (notation in previous section)} \quad (71)$$

$$:= \phi(x_k|\theta_k)\phi(x_k|\alpha_k)\phi(x_k|\beta_k) \text{ (notation in this section and in main text)} \quad (72)$$

$$= \phi(x_k|\theta_k + \alpha_k + \beta_k) \quad (73)$$

$$= \phi(x_k|\theta_k + \alpha_k + \beta_k) \quad (73)$$

735

Proposition 12 (Transition distribution).

$$p(x_{k+1}|x_k, \theta) \propto \phi_{k+1|k}(x_{k+1}|x_k) \phi(x_{k+1}|\theta_{k+1} + \beta_{k+1}) \quad (74)$$

<sup>736</sup> *Proof.* We can start by computing the joint distribution p(xk+1, xk|θ). By using variable elimination, <sup>737</sup> we can show that

$$p(x_{k+1}, x_k|\theta) = \phi(x_k|\alpha_k)\phi_{k+1|k}(x_{k+1}|x_k)\phi(x_{k+1}|\theta_{k+1})\phi(x_{k+1}|\beta_{k+1}) \quad (75)$$

<sup>738</sup> Dividing by the marginal distribution p(xk|θ) and using the definition of the transition distribution, <sup>739</sup> we get

$$p(x_{k+1}|x_k, \theta) = \phi_{k+1|k}(x_{k+1}|x_k) \frac{\phi(x_{k+1}|\beta_{k+1} + \theta_{k+1})}{\phi(x_k|\beta_k + \theta_k)} \quad (76)$$

<sup>742</sup> Corollary 3 (Autoregressive factorization). *The autoregressive factorization of* p(x1:<sup>N</sup> |θ) *takes the* <sup>743</sup> *following form:*

$$p(x_{1:N}|\theta) \propto \phi(x_1|\theta_1 + \beta_1) \prod_{k=1}^{N-1} \phi_{k+1|k}(x_{k+1}|x_k) \phi(x_{k+1}|\theta_{k+1} + \beta_{k+1}) \quad (77)$$

<sup>744</sup> *Proof.* This follows directly from applying Proposition [11](#page-19-2) and Proposition [12](#page-19-0) to p(x1:<sup>N</sup> |θ) = p(x1|θ) Q<sup>N</sup>−<sup>1</sup> <sup>k</sup>=1 <sup>745</sup> p(xk+1|xk, θ).

# <sup>746</sup> E Conditioned SDEs

<sup>747</sup> In this section we derive the form of conditioned linear SDEs as well as the corresponding probability <sup>748</sup> flow ODEs.

# <sup>749</sup> E.1 Conditioned linear SDE

<sup>750</sup> Proposition 13 (Conditioned Linear SDE). *Let* ϕt+s|t(xt+s|xt) *be the transition distribution of the linear SDE* dx<sup>t</sup> = Ftxtdt + LtdW<sup>t</sup> *and let* {ϕ(x<sup>t</sup><sup>k</sup> |θ<sup>t</sup><sup>k</sup> <sup>751</sup> )}<sup>t</sup>k∈R *be potential functions at times in the* <sup>752</sup> *set* R*. Then the piecewise-linear SDE,*

$$dx_t = (F_t x_t + L_t L_t^T \nabla \log \phi(x_t | \beta_t)) dt + L_t dW_t, \quad x_{t_1} \sim \phi(x_{t_1} | \beta_1 + \theta_1) \quad (78)$$

<sup>753</sup> *where* t ∈ (tk, tk+1) *and* tk, tk+1 ∈ R*, has a joint distribution over any superset of times* t1:<sup>N</sup> = <sup>754</sup> T ⊇ R *that is given by a CRF:*

$$p(x_{t_{1:N}}|\theta) \propto \prod_{t_k \in \mathcal{T}} \phi_{t_{k+1}|t_k}(x_{t_{k+1}}|x_{t_k}) \prod_{t_k \in \mathcal{R}} \phi(x_{t_k}|\theta_{t_k}) \quad (79)$$

<sup>755</sup> *where* β<sup>t</sup> *is the extension of the backward message defined in* ?? *to time* t*:*

$$\phi(x_t | \beta_t) = \int \phi_{t_{k+1} | t}(x_{t_{k+1}} | x_t) \phi(x_{t_{k+1}} | \theta_{t_{k+1}} + \beta_{t_{k+1}}) dx_{t_{k+1}} \quad (80)$$

<sup>756</sup> *Proof.* We will first construct the transition distribution of the conditioned SDE and then use Doob's <sup>757</sup> h-transform to identify the form of the SDE. Recall that Doob's h-transform ([\[Särkkä and Solin,](#page-10-7) <sup>758</sup> [2019\]](#page-10-7) section 7.5) is used to find the SDE associated with a transition distribution of the form p(xt+s|xt) = ϕt+s|t(xt+s|xt) ht+s(xt+s) <sup>759</sup> <sup>h</sup>t(xt) where <sup>ϕ</sup>t+s|t(xt+s|xt) is the transition distribution of <sup>760</sup> a base SDE with the form dx<sup>t</sup> = utdt + LtdW<sup>t</sup> and h<sup>t</sup> is a function that satisfies ht(xt) = R <sup>t</sup>+<sup>s</sup> t <sup>761</sup> ϕt+s|t(xt+s|xt)ht+s(xt+s)dxt+s. Then the SDE whose transition distribution is p(xt+s|xt) is <sup>762</sup> given by

$$dx_t = (u_t + L_t L_t^T \nabla \log h_t(x_t)) dt + L_t dW_t \quad (81)$$

<sup>763</sup> We will show that the backward messages of the CRF are of the form ht(xt) and then use Doob's <sup>764</sup> h-transform to identify the form of the conditioned SDE.

<sup>765</sup> Suppose t ∈ (tk, tk+1) and s > 0 is small enough so that t + s ∈ (tk, tk+1). Then we can construct <sup>766</sup> the joint distribution over (tt+s, tk+1, . . . , t<sup>N</sup> ) given x<sup>t</sup> as

$$p(x_{t+s} | x_t) = \int \cdots \int p(x_{t_{k+1}:N}, x_{t+s} | x_t) dx_{t_{k+1}} \cdots dx_{t_N} \quad (82)$$

$$\propto \int \cdots \int \phi(x_{t_{k+1}} | \theta_{t_{k+1}}) \underbrace{\left( \prod_{i=k+1}^{N-1} \phi_{t_{i+1} | t_i}(x_{t_{i+1}} | x_{t_i}) \phi(x_{t_{i+1}} | \theta_{t_{i+1}}) \right)}_{\text{integrate to get parallel bwd message (Proposition 9)}} \phi_{t_{k+1} | t+s}(x_{t_{k+1}} | x_{t+s}) dx_{t_{k+1}} \cdots dx_{t_N} \phi_{t+s | t}(x_{t+s} | x_t)$$

(83)

$$= \int \int \phi(x_{t_{k+1}} | \theta_{t_{k+1}}) \Psi_{k+1:N}^{\text{bwd}}(x_{t_N} | x_{t_{k+1}}) \phi_{t_{k+1} | t+s}(x_{t_{k+1}} | x_{t+s}) dx_{t_N} dx_{t_{k+1}} \phi_{t+s | t}(x_{t+s} | x_t) \quad (84)$$

$$= \underbrace{\int \phi(x_{t_{k+1}} | \theta_{t_{k+1}}) \phi(x_{t_{k+1}} | \beta_{t_{k+1}}) \phi_{t_{k+1} | t+s}(x_{t_{k+1}} | x_{t+s}) dx_{t_{k+1}} \phi_{t+s | t}(x_{t+s} | x_t)}_{=: \phi(x_{t+s} | \beta_{t+s})} \quad (85)$$

$$= \phi(x_{t+s} | \beta_{t+s}) \phi_{t+s|t}(x_{t+s} | x_t) \quad (86)$$

<sup>767</sup> We can find the normalizing constant by integrating over xt+s:

$$\int \phi(x_{t+s}|\beta_{t+s})\phi_{t+s|t}(x_{t+s}|x_t)dx_{t+s} \quad (87)$$

$$= \int \int \phi(x_{t_{k+1}} | \theta_{t_{k+1}}) \phi(x_{t_{k+1}} | \beta_{t_{k+1}}) \phi_{t_{k+1}|t+s}(x_{t_{k+1}} | x_{t+s}) dx_{t_{k+1}} \phi_{t+s|t}(x_{t+s} | x_t) dx_{t+s} \quad (88)$$

$$= \int \phi(x_{t_{k+1}} | \theta_{t_{k+1}}) \phi(x_{t_{k+1}} | \beta_{t_{k+1}}) \underbrace{\int \phi_{t_{k+1} | t+s}(x_{t_{k+1}} | x_{t+s}) \phi_{t+s | t}(x_{t+s} | x_t) dx_{t+s}}_{\phi_{t_{k+1} | t}(x_{t_{k+1}} | x_t)} dx_{t_{k+1}}} \quad (89)$$

$$\begin{aligned} &= \int \phi(x_{t_{k+1}} | \theta_{t_{k+1}}) \phi(x_{t_{k+1}} | \beta_{t_{k+1}}) \phi_{t_{k+1} | t}(x_{t_{k+1}} | x_t) dx_{t_{k+1}} \\ &= \phi(x_t | \beta_t) \end{aligned} \tag{90}$$
(91)

<sup>768</sup> Therefore, the transition distribution is

$$p(x_{t+s}|x_t) = \phi_{t+s|t}(x_{t+s}|x_t) \frac{\phi(x_{t+s}|\beta_{t+s})}{\phi(x_t|\beta_t)} \quad (92)$$

<sup>769</sup> Note that Eq. [\(87\)](#page-21-1) also verifies that ϕ(xt|βt) satisfies the normalization condition for ht(xt) in Doob's <sup>770</sup> h-transform. Directly applying Doob's h-transform to the transition distribution in Eq. [\(82\)](#page-20-3) identifies <sup>771</sup> the form of the conditioned SDE:

$$dx_t = (F_t x_t + L_t L_t^T \nabla \log \phi(x_t | \beta_t)) dt + L_t dW_t \quad (93)$$

This piecewise-linear SDE has the correct conditional distribution, p(xt|x<sup>t</sup>k<sup>1</sup> <sup>772</sup> ), but requires an initial distribution. One can verify that the initial distribution p(x<sup>t</sup><sup>1</sup> ) ∝ ϕ(x<sup>t</sup><sup>1</sup> |θ<sup>t</sup><sup>1</sup> + β<sup>t</sup><sup>1</sup> <sup>773</sup> ) is the first marginal <sup>774</sup> distribution of the CRF in Definition [1.](#page-3-0)

# <sup>775</sup> E.2 Probabilistic queries for conditioned linear SDEs

<sup>776</sup> Lemma 2 (Marignal distribution of conditioned SDE). *Suppose* t ∈ (tk, tk+1) *is a time in between* <sup>777</sup> *the inducing points* t<sup>k</sup> *and* tk+1 *of the conditioned linear SDE in Proposition [4.](#page-4-0) Then the marginal* <sup>778</sup> *distribution of the SDE at time* t *is given by*

$$p(x_t) = \phi(x_t | \alpha_t + \beta_t) \quad (94)$$

<sup>779</sup> *where* α<sup>t</sup> *and* β<sup>t</sup> *are extensions of the forward and backward messages defined in Eq.* [\(33\)](#page-16-1) *and* <sup>780</sup> *Eq.* [\(26\)](#page-15-2) *to time* t*:*

$$\phi(x_t|\alpha_t) = \int \phi_{t|t_{k-1}}(x_t|x_{t_{k-1}})\phi(x_{t_{k-1}}|\theta_{t_{k-1}} + \alpha_{t_{k-1}})dx_{t_{k-1}} \quad (95)$$

<sup>781</sup> *and*

$$\phi(x_t | \beta_t) = \int \phi_t|_{t_{k+1}}(x_t | x_{t_{k+1}}) \phi(x_{t_{k+1}} | \theta_{t_{k+1}} + \beta_{t_{k+1}}) dx_{t_{k+1}} \quad (96)$$

<sup>782</sup> *Proof.* We can simply incorporate t into the set discretization times, t1:<sup>N</sup> , used in Proposition [4](#page-4-0) to get the desired result. Suppose t ∈ (t<sup>i</sup> <sup>783</sup> , ti+1) for some i. Then we can write the joint distribution as

$$p(x_t, x_{t+1:N} | \theta) \propto \phi_{t+1:t}(x_{t+1:t} | x_{t+1}) \phi_{t+1:t}(x_t | x_{t+1}) \prod_{t_k \in \mathcal{T}} \phi_{t_{k+1}:t_k}(x_{t_{k+1}:t_k} | x_{t_k}) \prod_{t_k \in \mathcal{R}} \phi(x_{t_k} | \theta_{t_k}) \quad (97)$$

<sup>784</sup> Then we can run variable elimination on the ends of the chain until we are left with the marginal <sup>785</sup> distribution of xt:

$$\begin{aligned}
p(x_t) &= \int p(x_t, x_{t_{1:N}} | \theta) dx_{t_{1:N}} \\
&= \int \int \phi(x_{t_i} | \alpha_{t_i} + \theta_{t_i}) \phi_{t|t_i}(x_t | x_{t_i}) \phi_{t_{i+1}|t}(x_{t_{i+1}} | x_t) \phi(x_{t_{i+1}} | \beta_{t_{i+1}} + \theta_{t_{i+1}}) dx_{t_{i+1}} dx_{t_i} \\
& \tag{99}
\end{aligned}$$

$$= \underbrace{\int \phi(x_{t_i} | \alpha_{t_i} + \theta_{t_i}) \phi_{t_i|t_i}(x_t | x_{t_i}) dx_{t_i}}_{\phi(x_t | \alpha_t)} \underbrace{\int \phi_{t_{i+1}|t}(x_{t_{i+1}} | x_t) \phi(x_{t_{i+1}} | \beta_{t_{i+1}} + \theta_{t_{i+1}}) dx_{t_{i+1}}}_{\phi(x_t | \beta_t)} \quad (100)$$

$$= \phi(x_t|\alpha_t + \beta_t) \tag{101}$$

786

 Lemma 3 (Transition distribution of conditioned linear SDE). *Suppose* t ∈ (tk, tk+1) *is a time in between the inducing points* t<sup>k</sup> *and* tk+1 *of the conditioned linear SDE in Proposition [4,](#page-4-0) and suppose that* s > 0 *is small enough so that* t + s ∈ (tk, tk+1)*. Then the transition distribution of the SDE at time* t *is given by*

$$\phi_{t+s|t}(x_{t+s}|x_t) \propto \phi_{t+s|t}(x_{t+s}|x_t)\phi(x_{t+s}|\beta_{t+s}) \quad (102)$$

<sup>791</sup> *Proof.* The proof is embedded in the derivation of the conditioned linear SDE at Eq. [\(92\)](#page-21-2).

<sup>792</sup> Corollary 4 (Autoregressive factorization). *The autoregressive factorization of* p(x<sup>t</sup>1:<sup>N</sup> |θ) *is given* <sup>793</sup> *by*

$$p(x_{t_1:N}|\theta) = p(x_{t_1}|\theta) \prod_{t_k \in \mathcal{T}} \phi_{t_k|t_{k-1}}(x_{t_k}|x_{t_{k-1}}) \phi(x_{t_k}|\beta_{t_k}) \quad (103)$$

$$\text{where } \beta_{t_k} = \begin{cases} \Phi_{t_k, t_{k+1}}(\beta_{t_{k+1}} + \theta_{t_{k+1}}) & \text{if } t_k \in \mathcal{R} \\ \Phi_{t_k, t_{k+1}}(\beta_{t_{k+1}}) & \text{otherwise} \end{cases} \quad (104)$$

*where* Φ<sup>t</sup>k,tk+1 <sup>794</sup> *is the message passing update operator defined in Definition [7.](#page-18-0)*

<sup>795</sup> *Proof.* Recall that

$$p(x_{t_{1:N}}|\theta) \propto \prod_{t_k \in \mathcal{T}} \phi_{t_{k+1}|t_k}(x_{t_{k+1}}|x_{t_k}) \prod_{t_k \in \mathcal{R}} \phi(x_{t_k}|\theta_{t_k}) \quad (105)$$

<sup>796</sup> Suppose that for each t<sup>k</sup> ∈ R/ , we introduce a new potential function whose natural parameters are 0, which we will denote by ϕ(x<sup>t</sup><sup>k</sup> |∅<sup>t</sup><sup>k</sup> <sup>797</sup> ). These new potentials have no effect on the joint distribution, <sup>798</sup> but allow us to rewrite the joint distribution in the same form as in Corollary [3,](#page-19-3) which yields the <sup>799</sup> result.

# <sup>800</sup> E.3 Probability flow ODE for conditioned linear SDEs

<sup>801</sup> Corollary 5 (Probability flow ODE). *The probability flow ODE of the SDE in Proposition [4](#page-4-0) is given* <sup>802</sup> *by*

$$\frac{dx_t}{dt} = F_t x_t + \frac{1}{2} L_t L_t^T (\nabla \log \phi(x_t | \beta_t) - \nabla \log \phi(x_t | \alpha_t)) \quad (106)$$

<sup>803</sup> β<sup>t</sup> *is the same as in Proposition [4](#page-4-0) and* α<sup>t</sup> *is the extension of the forward message defined in Eq.* [\(33\)](#page-16-1) <sup>804</sup> *to time* t*:*

$$\phi(x_t | \alpha_t) = \int \phi_{t | t_k}(x_t | x_{t_k}) \phi(x_{t_k} | \theta_{t_k} + \alpha_{t_k}) dx_{t_k} \quad (107)$$

<sup>805</sup> *Proof.* Let dx<sup>t</sup> = utdt + LtdW<sup>t</sup> be an SDE. Then the probability flow ODE is defined [Song et al.](#page-10-0) <sup>806</sup> [\[2021\]](#page-10-0) as

$$\frac{dx_t}{dt} = u_t - \frac{1}{2}L_tL_t^T \nabla \log p_t(x_t) \quad (108)$$

<sup>807</sup> where pt(xt) is defined as the marginal distribution of the SDE, which is given by Lemma [2.](#page-21-3) We can <sup>808</sup> apply this directly to our SDE in Proposition [4](#page-4-0) to get the result:

$$\frac{dx_t}{dt} = (F_t x_t + L_t L_t^T \nabla \log \phi(x_t | \beta_t)) - \frac{1}{2} L_t L_t^T \nabla \log p_t(x_t) \quad (109)$$

$$= (F_t x_t + L_t L_t^T \nabla \log \phi(x_t | \beta_t)) - \frac{1}{2} L_t L_t^T (\nabla \log \phi(x_t | \alpha_t) + \nabla \log \phi(x_t | \beta_t)) \quad (110)$$

$$= F_t x_t + \frac{1}{2} L_t L_t^T (\nabla \log \phi(x_t | \beta_t) - \nabla \log \phi(x_t | \alpha_t)) \quad (111)$$

# <sup>810</sup> F CMFVI proofs

#### <sup>811</sup> F.1 Constrained mean field VI

 Let θ ∼ p(θ) be an unknown prior distribution on the parameters of the conditional exponential family distribution, p(x|z, θ) ∝ exp{⟨tz(x), θ⟩ − A(z, θ)}, where tz(x) is the sufficient statistic of the exponential family distribution and A(z, θ) is the log partition function. In our setting, we interpret x and z as unobserved and observed variables and θ as a a parameter that they both depend on. We are interested in performing inference in the predictive distribution p(x|z), where we must integrate out θ. This distribution can be written as:

$$p(x|z) = \int p(x|z, \theta) p(\theta|z) d\theta \quad (112)$$

$$= \mathbb{E}_{p(\theta|z)} [\exp\{\langle t_z(x), \theta \rangle - A(z, \theta)\}] \quad (113)$$

<sup>818</sup> where tz(x) is the sufficient statistic of the conditional exponential family distribution. Since this <sup>819</sup> distribution is intractable, we use a variational approximation to approximate it. Our variational <sup>820</sup> approximation is called the constrained mean field VI approximation and is given by:

$$q^*(x|z) = \underset{q(x|z)}{\operatorname{argmin}} \text{KL}[q(x|z)p(\theta|z) \| p(x, \theta|z)] \quad (114)$$

In this appendix section we will derive facts about q ∗ <sup>821</sup> (x|z).

<sup>822</sup> Lemma 4 (Alternate constrained mean field VI objectives). *The constrained mean field VI objective,*

$$\text{KL} [q(x|z)p(\theta|z) \| p(x, \theta|z)] \quad (115)$$

<sup>823</sup> *is equal to the following expressions:*

*1.*

$$\mathbb{E}_{q(x|z) \ p(\theta|z)} \left[ \log \frac{p(\theta|z)}{p(\theta|x, z)} \right] + \text{KL} [q(x|z) \| p(x|z)] \quad (116)$$

*2.*

$$\mathbb{E}_{q(x|z) p(\theta|z)} \left[ \log \frac{p(x|z)}{p(x|z, \theta)} \right] + \text{KL} [q(x|z) \| p(x|z)] \quad (117)$$

*3.*

$$\mathbb{E}_{q(x|z)} [\log q(x|z) - \mathbb{E}_{p(\theta|z)} [\log p(x|z, \theta)]] \quad (118)$$

<sup>824</sup> *Proof.* The proof is a straightforward rearrangement of terms:

$$\text{KL}[q(x|z)p(\theta|z)\|p(x, \theta|z)] = \int \int q(x|z)p(\theta|z) \log \frac{q(x|z)p(\theta|z)}{p(x, \theta|z)} dx dy \quad (119)$$

$$= \int \int q(x|z)p(\theta|z) \log \frac{p(\theta|z)}{p(\theta|x,z)} \frac{q(x|z)}{p(x|z)} dx dy \quad (\text{equals 1}) \quad (120)$$

$$= \int \int q(x|z)p(\theta|z) \log \frac{p(x|z)}{p(x|z, \theta)} \frac{q(x|z)}{p(x|z)} dx dy \quad (\text{equals 2}) \quad (21)$$

$$= \int \int q(x|z)p(\theta|z) \log \frac{q(x|z)}{p(x|z, \theta)} dx dy \quad (122)$$

$$= \mathbb{E}_{q(x|z)} [\log q(x|z) - \mathbb{E}_{p(\theta|z)} [\log p(x|z, \theta)]] \quad (123)$$

825

<sup>826</sup> Theorem 2 (Constrained mean field VI solution). *Let* p(x|z, θ) ∝ exp{⟨tz(x), θ⟩ − A(z, θ)} *be an* <sup>827</sup> *exponential family distribution and that* θ ∼ p(θ|z)*. The constrained mean field VI approximation of* p(x|z)*, denoted by* q ∗ <sup>828</sup> (x|z)*, is defined as follows:*

$$q^*(x|z) = \operatorname{argmin}_{q(x|z)} \text{KL} [q(x|z)p(\theta|z) \| p(x, \theta|z)] \quad (124)$$

$$= p(x|z, \theta^*(z)), \quad \text{where } \theta^*(z) = \mathbb{E}_{p(\theta|z)} [\theta] \quad (125)$$

<sup>829</sup> *Proof.* The proof can follow quickly from the standard mean field VI solutions [Beal](#page-11-10) [\[2003\]](#page-11-10), but for <sup>830</sup> completeness we will derive it from scratch. Starting from the result of Lemma [4,](#page-23-1) we have that

$$q^*(x|z) = \underset{q(x|z)}{\operatorname{argmin}} \quad \mathbb{E}_{q(x|z)} [\log q(x|z) - \mathbb{E}_{p(\theta|z)} [\log p(x|z, \theta)]] \quad (126)$$

<sup>831</sup> We can introduce a Lagrange multiplier to enforce the constraint that the distribution is normalized. <sup>832</sup> Let qϵ(x|z) = q(x|z) + ϵη(x|z) where η is the variation function and ϵ is a scalar. Then we can take <sup>833</sup> a variation by differentiating with respect to ϵ:

$$\frac{\partial}{\partial \epsilon} \left( \mathbb{E}_{q_\epsilon(x|z)} [\log q_\epsilon(x|z) - \mathbb{E}_{p(\theta|z)} [\log p(x|z, \theta)]] + \lambda \left( \int q_\epsilon(x|z) dx - 1 \right) \right) = 0 \quad (127)$$

$$\Rightarrow \frac{\partial}{\partial \epsilon} \int q_\epsilon(x|z) \log q_\epsilon(x|z) dx + \int \eta(x|z) (\mathbb{E}_{p(\theta|z)} [\log p(x|z, \theta)] + \lambda) dx = 0 \quad (128)$$

<sup>834</sup> The negative entropy term simplies as follows:

$$\frac{\partial}{\partial \epsilon} \int q_\epsilon(x|z) \log q_\epsilon(x|z) dx = \int \frac{\partial}{\partial \epsilon} q_\epsilon(x|z) \log q_\epsilon(x|z) dx + \int q_\epsilon(x|z) \frac{\partial}{\partial \epsilon} \log q_\epsilon(x|z) dx \quad (129)$$

$$= \int \frac{\partial q_\epsilon(x|z)}{\partial \epsilon} \log q_\epsilon(x|z) dx + \int q_\epsilon(x|z) \frac{\partial \log q_\epsilon(x|z)}{\partial \epsilon} dx \quad (130)$$

$$= \int \eta(x|z) \log q_\epsilon(x|z) dx - \int q_\epsilon(x|z) \frac{1}{q_\epsilon(x|z)} \frac{\partial q_\epsilon(x|z)}{\partial \epsilon} dx \quad (131)$$

$$= \int \eta(x|z) (\log q_\epsilon(x|z) - 1) dx \quad (132)$$

<sup>835</sup> Plugging this back into the original equation and setting it equal to zero implies that the integrand <sup>836</sup> must be zero:

$$\mathbb{E}_{p(\theta|z)} [\log p(x|z, \theta)] + \lambda + \log q_\epsilon(x|z) - 1 = 0 \quad (133)$$

<sup>837</sup> Solving for log qϵ(x|z) (and setting ϵ = 0) yields:

$$\log q(x|z) = \mathbb{E}_{p(\theta|z)} [\log p(x|z, \theta)] + \lambda - 1 \quad (134)$$

<sup>838</sup> The lagrange multiplier λ ensures that the distribution is normalized, and so we have that

$$q^*(x|z) = \exp \left\{ \mathbb{E}_{p(\theta|z)} [\log p(x|z, \theta)] + \lambda - 1 \right\} \quad (135)$$

$$\propto \exp \left\{ \mathbb{E}_{p(\theta|z)} [\log p(x|z, \theta)] \right\} \quad (136)$$

$$\propto \exp \left\{ \langle t_z(x), \mathbb{E}_{p(\theta|z)} [\theta] \rangle \right\} \quad (137)$$

And so we can recognize that q ∗ <sup>839</sup> (x|z) is in the same exponential family as p(x|z, θ) but with natural parameter <sup>E</sup>p(θ|z) <sup>840</sup> [θ]. This completes the proof.

<sup>841</sup> Next, we emphasize another form of the CMFVI solution that is convenient when deriving CMFVI <sup>842</sup> solutions of other models.

<sup>843</sup> Lemma 5 (Mean field form of CMFVI solution). *The CMFVI approximation of* p(x|z) *has the* <sup>844</sup> *following form:*

$$q^*(x|z) \propto \exp \{ \mathbb{E}_{p(\theta|z)} [\log p(x|z, \theta)] \} \quad (138)$$

<sup>845</sup> *Proof.* See Eq. [\(136\)](#page-24-0)

<sup>846</sup> Corollary 6 (Value of CMFVI objective at optimum). *The value of the CMFVI objective at the* <sup>847</sup> *optimum is given by:*

$$\text{KL} [q^*(x|z)p(\theta|z)||p(x, \theta|z)] = \mathbb{E}_{p(\theta|z)} [A(z, \theta)] - A(z, \theta^*(z)) \quad (139)$$

*Proof.* Let θ ∗ (z) = <sup>E</sup>p(θ|z) [θ]. Recall that p(x|z, θ) = exp {⟨tz(x), θ⟩ − A(z, θ)}, q ∗ <sup>849</sup> (x|z) = p(x|z, θ<sup>∗</sup> <sup>850</sup> (z)) and that the CMFVI objective can be written using an identity from Lemma [4:](#page-23-1)

$$\text{KL} [q(x|z)p(\theta|z)\|p(x,\theta|z)] = \mathbb{E}_{q(x|z)} [\log q(x|z) - \mathbb{E}_{p(\theta|z)} [\log p(x|z,\theta)]] \quad (140)$$

We can plug q ∗ <sup>851</sup> (x|z) and p(x|z, θ) into the identity to get:

$$\text{KL}[q^*(x|z)p(\theta|z)\|p(x,\theta|z)] \quad (141)$$

$$\begin{aligned} & \text{KL} [q^*(x|z)p(\theta|z)\|p(x, \theta|z)] \\ &= \mathbb{E}_{q^*(x|z)} [\log q^*(x|z) - \mathbb{E}_{p(\theta|z)} [\log p(x|z, \theta)]] \end{aligned} \quad (141)$$

$$\begin{aligned} &= \mathbb{E}_{Q^*(x|z)} [\log Q(x|z) - \mathbb{E}_{P(\theta|z)} [\log P(x|z, \theta)]] \\ &= \mathbb{E}_{Q^*(x|z)} \left[ \left( \langle t_z(x), \theta^*(z) \rangle - A(z, \theta^*(z)) \right) - \left( \underbrace{\langle t_z(x), \mathbb{E}_{P(\theta|z)} [\theta] \rangle}_{\theta^*(z)} - \mathbb{E}_{P(\theta|z)} [A(z, \theta)] \right) \right] \\ &= \mathbb{E}_{P(\theta|z)} [A(z, \theta)] - A(z, \theta^*(z)) \end{aligned} \quad (143)$$

$$= \mathbb{E}_{P(\theta|z)} [A(z, \theta)] - A(z, \theta^*(z)) \quad (144)$$

Proposition 14 (Forward KL divergence). *The forward KL divergence between* p(x|z) *and* q ∗ <sup>853</sup> (x|z) <sup>854</sup> *is given by:*

$$\text{KL}[p(x|z) \| q^*(x|z)] = -H_p[x|z] - \langle t^*(z), \theta^*(z) \rangle + A(z, \theta^*(z)) \quad (145)$$

*where* Hp[x|z] *is the differential entropy of* p(x|z)*,* t ∗ (z) = <sup>E</sup>p(x|z) [tz(x)]*,* θ ∗ (z) = <sup>E</sup>p(θ|z) <sup>855</sup> [θ] *and* <sup>856</sup> A(z, θ) *is the partition function of* p(x|z, θ)*.*

<sup>857</sup> *Proof.* This follows from a direct computation:

$$\text{KL}[p(x|z)||q^*(x|z)] = -H_p[x|z] - \int p(x|z) \log q^*(x|z) dx \quad (146)$$

$$= -H_p[x|z] - \int p(x|z) (\langle t_z(x), \theta^*(z) \rangle - A(z, \theta^*(z))) dx \quad (147)$$

$$= -H_p[x|z] - \left\langle \int p(x|z)t_z(x)dx, \theta^*(z) \right\rangle + A(z, \theta^*(z)) \quad (148)$$

$$= -H_p[x|z] - \langle t^*(z), \theta^*(z) \rangle + A(z, \theta^*(z)) \quad (149)$$

858

### <sup>859</sup> F.2 Bayes estimator equivariance

 We will use the equivariance of the Bayes estimator to linear transformations to show that it is also equivariant to message passing updates when the Gaussian potential functions of the corresponding CRF have covariances that only depend on the node index. This result will allow us to reparameterize the Bayes estimator of the backward messages in terms of the previously computed backward messages, and also in terms of the potential function means themselves. This will be useful for relating the CMFVI time series models we construct back traditional time series models, and also for proving that the autoregressive CMFVI model we construct is an approximation of flow-based generative models for time series.

 Corollary 7 (Commutativity of Bayes estimator with update and marginalize opera- tor). *Let* ϕk+1|k(xk+1|xk) *be a Gaussian transition function and let* ϕ(xk+1|ηk+1) := N(xk+1|µk+1(y), J−<sup>1</sup> <sup>k</sup>+1 <sup>870</sup> ) *be a Gaussian node potential where* y ∼ p(y) *is an auxilary variable set of variables that only the mean of the potential depends on. Then the Bayes estimator of* η<sup>k</sup> *commutes with the update and marginalize operator. That is,*

$$\mathbb{E}_{p(y)}[\eta_k(y)] = \mathbb{E}_{p(y)}[\Phi_{k,k+1}(\eta_{k+1}(y))] = \Phi_{k,k+1}(\mathbb{E}_{p(y)}[\eta_{k+1}(y)]) \quad (150)$$

<sup>873</sup> *Proof.* We can examine the form of Φk,k+1 from Corollary [2](#page-19-4) to see that Φk,k+1 is linear with respect <sup>874</sup> to µk+1(y). Then the result follows from linearity equivariance of the Bayes estimator.

#### <sup>875</sup> F.3 CMFVI time series models

<sup>876</sup> Proposition 15 (Naive CMFVI solution). *Let* p(x<sup>t</sup>1:<sup>N</sup> |yO) *be the target distribution. Then the naive CMFVI solution, denoted by* q *CRF* <sup>877</sup> (x<sup>t</sup>1:<sup>N</sup> ) *is the CMFVI approximation of* p(x<sup>t</sup>1:<sup>N</sup> |yO) *and is given* <sup>878</sup> *by:*

$$q^{CRF}(x_{t_{1:N}}) \propto \prod_{t_k \in \mathcal{T}} \phi_{t_{k+1}|t_k}(x_{t_{k+1}}|x_{t_k}) \prod_{t_k \in \mathcal{R}} \phi(x_{t_k}|\theta_{t_k}^*(y_{\mathcal{O}})) \quad (151)$$

*where* θ ∗ tk (yO) = <sup>E</sup>p(y<sup>U</sup> <sup>|</sup>yO) [θ<sup>t</sup><sup>k</sup> (y<sup>τ</sup>1:<sup>T</sup> )] *is the Bayes estimator of* θ<sup>t</sup><sup>k</sup> <sup>879</sup> *.*

*Proof.* By expanding q ∗ <sup>880</sup> using Lemma [5,](#page-24-1) one finds that the terms of the log likelihood is linear with respect to θ<sup>t</sup><sup>k</sup> (y<sup>τ</sup>1:<sup>T</sup> <sup>881</sup> ). Then the result follows from the equivariance of the Bayes estimator to linear <sup>882</sup> transformations.

<sup>883</sup> Proposition 16 (CMFVI transition approximation). *Let* p(x<sup>t</sup>1:<sup>N</sup> |yO) *be the target distribution and consider its* k*'th autoregressive factor* p(x<sup>t</sup><sup>k</sup> |x<sup>t</sup>1:k−<sup>1</sup> <sup>884</sup> , yO)*. Then the CMFVI transition approximation* <sup>885</sup> *is given by:*

$$q^{transition}(x_{t_k} | x_{t_{1:k-1}}, y_{\mathcal{O}}) \propto \phi_{t_k | t_{k-1}}(x_{t_k} | x_{t_{k-1}}) \phi(x_{t_k} | \beta_{t_k}^*(x_{t_{1:k-1}}, y_{\mathcal{O}})) \quad (152)$$

*where* β ∗ tk (x<sup>t</sup>1:k−<sup>1</sup> , yO) = <sup>E</sup>p(y<sup>U</sup> <sup>|</sup>xt1:k−<sup>1</sup> ,yO) [β<sup>t</sup><sup>k</sup> (y<sup>τ</sup>1:<sup>T</sup> )] *is the Bayes estimate of* β<sup>t</sup><sup>k</sup> (y<sup>τ</sup>1:<sup>T</sup> <sup>886</sup> )*, which is defined using the message passing update operator* Φ<sup>t</sup>k,tk+1 <sup>887</sup> *from Definition [7](#page-18-0) as:*

$$\beta_{t_k} = \begin{cases} \Phi_{t_k, t_{k+1}}(\beta_{t_{k+1}}(y_{\tau_{1:T}}) + \theta_{t_{k+1}}(y_{\tau_{1:T}})) & \text{if } t_{k+1} \in \mathcal{R} \\ \Phi_{t_k, t_{k+1}}(\beta_{t_{k+1}}(y_{\tau_{1:T}})) & \text{otherwise} \end{cases} \quad (153)$$

<sup>888</sup> *Proof.* The transition distribution in the fully observed setting is given by:

$$p(x_{t_k} | x_{t_{1:k-1}}, y_{\tau_{1:T}}) = p(x_{t_k} | x_{t_{k-1}}, y_{\tau_{1:T}}) \quad (154)$$

$$\propto \phi_{t_k|t_{k-1}}(x_{t_k}|x_{t_{k-1}})\phi(x_{t_k}|\beta_{t_k}(y_{\tau_{1:T}})) \quad (155)$$

If we expand the log likelihood of p(x<sup>t</sup><sup>k</sup> |x<sup>t</sup>1:k−<sup>1</sup> , y<sup>τ</sup>1:<sup>T</sup> <sup>889</sup> ), we would find that the log likelihood is linear with respect to β<sup>t</sup><sup>k</sup> (y<sup>τ</sup>1:<sup>T</sup> <sup>890</sup> ), and so writing the CMFVI solution using Eq. [\(136\)](#page-24-0) yields the result.

We denote this model by q MSE <sup>891</sup> (x<sup>t</sup>1:<sup>N</sup> |yO).

<sup>892</sup> Corollary 8 (MSE Forecaster). *Let* p(x<sup>t</sup>1:<sup>N</sup> |yO) *be the target distribution and suppose the co-*<sup>893</sup> *variances of its potentials are constant with respect to* y*. Then the MSE-CMFVI solution, denoted by* q *MSE* <sup>894</sup> (x<sup>t</sup>1:<sup>N</sup> ) *is the CMFVI approximation of* p(x<sup>t</sup>1:<sup>N</sup> |yO) *obtained by choosing* (x, z, θ) = (x<sup>t</sup>1:<sup>N</sup> , yO, θ(y<sup>τ</sup>1:<sup>T</sup> <sup>895</sup> ))*:*

$$q^{MSE}(x_{t_{1:N}}|y_{\mathcal{O}}) \propto \prod_{t_k \in \mathcal{T}} \phi_{t_{k+1}|t_k}(x_{t_{k+1}}|x_{t_k}) \prod_{t_k \in \mathcal{R}} N(x_{t_k}|\mu_{t_k}^*(y_{\mathcal{O}}), \Sigma_{t_k}) \quad (156)$$

*where* µ ∗ tk (yO) = <sup>E</sup>p(y<sup>U</sup> <sup>|</sup>yO) [µ<sup>t</sup><sup>k</sup> (y<sup>τ</sup>1:<sup>T</sup> )] *is the Bayes estimate of* µ<sup>t</sup><sup>k</sup> *, and* ϕ(x<sup>t</sup><sup>k</sup> |θ<sup>t</sup><sup>k</sup> (y<sup>τ</sup>1:<sup>T</sup> <sup>896</sup> )) = N(x<sup>t</sup><sup>k</sup> |µ ∗ tk (y<sup>τ</sup>1:<sup>T</sup> ), Σ<sup>t</sup><sup>k</sup> <sup>897</sup> )*.*

#### <sup>898</sup> See Appendix [F.3](#page-26-2) for a proof.

<sup>899</sup> Definition 8 (Autoregressive CMFVI solution). *Let* p(x<sup>t</sup>1:<sup>N</sup> |yO) *be the target distribution. Then the autoregressive CMFVI solution, denoted by* q *AR* <sup>900</sup> (x<sup>t</sup>1:<sup>N</sup> ) *is the CMFVI approximation of* p(x<sup>t</sup>1:<sup>N</sup> |yO) <sup>901</sup> *and is given by:*

$$q^{\text{AR}}(x_{t_{1:N}}) \propto p(x_{t_1} | y_{\mathcal{O}}) \prod_{t_k \in \mathcal{T}} q^{\text{transition}}(x_{t_k} | x_{t_{1:k-1}}, y_{\mathcal{O}}) \quad (157)$$

*where* q *transition*(x<sup>t</sup><sup>k</sup> |x<sup>t</sup>1:k−<sup>1</sup> <sup>902</sup> , yO) *is the CMFVI transition approximation given by Proposition [6.](#page-6-0)*

<sup>903</sup> Corollary 9 (MSE Forecaster). *Let* p(x<sup>t</sup>1:<sup>N</sup> |yO) *be the target distribution and suppose the covari-*<sup>904</sup> *ances of its potentials are constant with respect to* y*. Then the MSE-CMFVI solution, denoted by* q *MSE* <sup>905</sup> (x<sup>t</sup>1:<sup>N</sup> ) *is the CMFVI approximation of* p(x<sup>t</sup>1:<sup>N</sup> |yO) *and is given by:*

$$q^{MSE}(x_{t_{1:N}}) \propto \prod_{t_k \in \mathcal{T}} \phi_{t_{k+1}|t_k}(x_{t_{k+1}}|x_{t_k}) \prod_{t_k \in \mathcal{R}} N(x_{t_k}|\mu_{t_k}^*(y_{\mathcal{O}}), \Sigma_{t_k}) \quad (158)$$

<sup>907</sup> *Proof.* This follows from the fact that the potentials are constant with respect to y and the linear <sup>908</sup> equivariance of the Bayes estimator.

<sup>909</sup> Corollary 10 (Autoregressive MSE Forecaster). *Let* p(x<sup>t</sup>1:<sup>N</sup> |yO) *be the target distribution and* <sup>910</sup> *suppose the covariances of its potentials are constant with respect to* y*. Then the autoregressive MSE-CMFVI solution, denoted by* q *AR-MSE* <sup>911</sup> (x<sup>t</sup>1:<sup>N</sup> ) *is the CMFVI approximation of* p(x<sup>t</sup>1:<sup>N</sup> |yO) *and is* <sup>912</sup> *given by:*

$$q^{AR-MSE}(x_{t_{1:N}}) \propto p(x_{t_1}|y_{\mathcal{O}}) \prod_{t_k \in \mathcal{T}} \phi_{t_k|t_{k-1}}(x_{t_k}|x_{t_{k-1}}) \prod_{t_k \in \mathcal{R}} N(x_{t_k} | (\mu_{t_k}^{\beta})^*(x_{t_{1:k}}, y_{\mathcal{O}}), \Sigma_{t_k}^{\beta}) \quad (159)$$

*where* Ä µ β tk ä∗ (x<sup>t</sup>1:<sup>k</sup> , yO) = <sup>E</sup>p(y<sup>U</sup> <sup>|</sup>xt1:<sup>k</sup> ,yO) î µ β tk (y<sup>τ</sup>1:<sup>T</sup> ) ó *is the Bayes estimate of* µ β tk *and* Σ β tk <sup>913</sup> *is the covariance of the backward message of* p(x<sup>t</sup>1:<sup>N</sup> |y<sup>τ</sup>1:<sup>T</sup> <sup>914</sup> )*.*

<sup>915</sup> *Proof.* This follows from the fact that the potentials are constant with respect to y and the linear <sup>916</sup> equivariance of the Bayes estimator.

Definition 9 (Continuous extension of AR-MSE model). *Let* q *AR* <sup>917</sup> *be the autoregressive CMFVI solution and consider the setting where the potential functions of* p(x<sup>t</sup>1:<sup>N</sup> |y<sup>τ</sup>1:<sup>T</sup> <sup>918</sup> ) *have covariances that do not depend on* y*. Then the continuous extension of* q *AR* <sup>919</sup> *is given by the following piecewise* <sup>920</sup> *linear SDE:*

$$dx_t = (F_t x_t + L_t T_t^T \nabla \log \phi(x_t | \beta_t^*(x_{t+1:k}, y_{\mathcal{O}}))) dt + L_t dW_t, \quad (160)$$

$$\text{where } \beta_t^*(x_{t_{1:k}}, y_{\mathcal{O}}) = \mathbb{E}_{p(y_{\mathcal{O}}|x_{t_{1:k}}, y_{\mathcal{O}})} [\beta_t(y_{\tau_{1:T}})], \text{ and } t \in (t_k, t_{k+1}) \quad (161)$$

*where* β ∗ t (x<sup>t</sup>1:<sup>k</sup> , yO) *is the Bayes estimator of* βt(y<sup>τ</sup>1:<sup>T</sup> ) = Φt,tk+1 (β<sup>t</sup>k+1 (y<sup>τ</sup>1:<sup>T</sup> <sup>921</sup> ))*.*

*Proof.* We just need to verify that this piecewise linear SDE has the same joint distribution as q AR <sup>922</sup> <sup>923</sup> on t1:<sup>N</sup> . To do this, we can just check that each of the linear SDEs that are defined on the intervals (tk, tk+1) have the same joint distribution as q transition(x<sup>t</sup><sup>k</sup> |x<sup>t</sup>1:k−<sup>1</sup> <sup>924</sup> , yO) from Proposition [6.](#page-6-0) This is <sup>925</sup> true by construction TODO: add proof.

# <sup>926</sup> G Flow-based generative models proofs

 In this section we provide basic results about Bayes estimation for generalized linear stochastic interpolants. Let dx<sup>t</sup> = Ftxtdt + LtdW<sup>t</sup> be the base linear SDE and let the distribution of random draws, at times t1:<sup>N</sup> , be denoted by p(x<sup>t</sup>1:<sup>N</sup> |c). Let p(x<sup>t</sup>1:<sup>N</sup> |θ, c) be its conditional distribution given parameters θ that are only available during training time and some extra conditioning information c that is avilable at both training and test time, and suppose that p(θ|c) is the (unknown) distribution of θ given c. The goal of the techniques in this section (and FBGMs in general), is to construct, and learn, the distribution of p(x<sup>t</sup>1:<sup>N</sup> |c), which is the distribution needed to generate samples of x<sup>t</sup>1:<sup>N</sup> when we do not have access to the parameters θ. At a high level, FBGMs offer different inference algroithms for this task. In this section, we will derive three of these inference algorithms.

#### <sup>936</sup> G.1 Score function for FBGMs

Proposition 17 (Score function for FBGMs). *Suppose that* p(θ|c) *is a probability distribution over* θ *given some extra conditioning information* c *and* p(xt|θ, c) *is the marignal distribution of a generalized linear stochastic interpolant whose base linear SDE is given by* dx<sup>t</sup> = Ftxtdt + LtdWt*. Then the score function of* p(xt|c) *is given by:*

$$\nabla \log p(x_t|c) = \nabla \log \phi(x_t|\alpha_t^*(x_t, \theta, c) + \beta_t^*(x_t, \theta, c)) \quad (162)$$

*where* α ∗ t (xt, θ, c) = <sup>E</sup>p(θ|xt,c) [αt(θ, c)] *and* β ∗ t (xt, θ, c) = <sup>E</sup>p(θ|xt,c) <sup>941</sup> [βt(θ, c)] *are Bayes estimators* <sup>942</sup> *of the forward and backward messages to time* t *using* x<sup>t</sup> *respectively.*

<sup>943</sup> *Proof.* A straightforward calculation will lead to the desired result.

$$\nabla \log p(x_t|c) = \frac{1}{p(x_t|c)} \nabla p(x_t|c) \quad (163)$$

$$= \frac{1}{p(x_t|c)} \nabla \int p(\theta|c) p(x_t|\theta, c) d\theta \quad (164)$$

$$= \frac{1}{p(x_t|c)} \int p(\theta|c) \nabla p(x_t|\theta, c) d\theta \quad (165)$$

$$= \int \frac{p(\theta|c)p(x_t|\theta, c)}{p(x_t|c)} \nabla \log p(x_t|\theta, c) d\theta \quad (16.6)$$

$$= \mathbb{E}_{p(\theta|x_t, c)} [\nabla \log p(x_t|\theta, c)] \quad (167)$$

$$= \mathbb{E}_{p(\theta|x_{t,c})} [\nabla \log \phi(x_t|\alpha_t(\theta, c) + \beta_t(\theta, c))] \quad \because \text{Lemma 2} \quad (168)$$

$$= \nabla \log \phi(x_t | \alpha_t^*(x_t, \theta, c) + \beta_t^*(x_t, \theta, c)) \quad \because Eq. (21) \quad (169)$$

944

# <sup>945</sup> G.2 General form of Markovian projection SDE

Lemma 6 (General form of Markovian projection SDE). *Suppose that* p(θ|c) *is a probability distribution over* θ *given some extra conditioning information* c *and* p(xt|θ, c) *is the marignal distribution of a generalized linear stochastic interpolant whose base linear SDE is given by* dx<sup>t</sup> = Ftxtdt + LtdWt*. Then the Markovian projection SDE is given by:*

$$dx_t = (F_t x_t + L_t L_t^T \nabla \log \phi(x_t | \beta_t^*(x_t, \theta, c))) dt + L_t dW_t \quad (170)$$

*where* β ∗ t (xt, θ, c) = <sup>E</sup>p(θ|xt,c) <sup>950</sup> [βt(θ, c)] *is the Bayes estimate of the backward message to time* t <sup>951</sup> *using* xt*.*

*Proof.* The Markovian projection SDE is the SDE whose marginal distribution evolves in time in the same way that p(xt|c) evolves in time, and so our proof strategy will follow the same strategy as [\[Lipman et al., 2023,](#page-10-2) Theorem 1] where we take the time derivative of p(xt|c) and recognize the form of the SDE.

 First, recall that the Fokker-Planck equation [\[Särkkä and Solin, 2019,](#page-10-7) [Øksendal and Øksendal, 2003\]](#page-11-14) relates an SDE to the time derivative of its marginal distribution. Let p(xt|θ, c) be the marginal distribution of the generalized linear stochastic interpolant and recall that its corresponding SDE is given by dx<sup>t</sup> = (Ftx<sup>t</sup> + LtL T <sup>t</sup> ∇ log ϕ(xt|βt(θ, c)))dt + LtdW<sup>t</sup> (see Proposition [4\)](#page-4-0). Then the Fokker-Planck equation for this SDE is given by:

$$\frac{\partial p(x_t|\theta, c)}{\partial t} = -\text{Div}(p(x_t|\theta, c)(F_t x_t + L_t L_t^T \nabla \log \phi(x_t|\beta_t(\theta, c)))) + \frac{1}{2} L_t L_t^T \text{Div}(\nabla p(x_t|\theta, c)) \quad (171)$$

LtL T t <sup>961</sup> appears outside the divergence operator because it does not depend on xt. Next, we can directly <sup>962</sup> take the time derivative of p(xt|c) and recognize the form of the corresponding SDE.

$$\frac{\partial p(x_t|c)}{\partial t} = \mathbb{E}_{p(\theta|c)} \left[ \frac{\partial p(x_t|\theta, c)}{\partial t} \right] \quad (172)$$

$$= \mathbb{E}_{p(\theta|c)} \left[ -\text{Div}(p(x_t|\theta, c)(F_t x_t + L_t L_t^T \nabla \log \phi(x_t|\beta_t(\theta, c)))) + \frac{1}{2} L_t L_t^T \text{Div}(\nabla p(x_t|\theta, c)) \right]$$

(173)

$$= \mathbb{E}_{p(\theta|c)} [-\text{Div}(p(x_t|\theta, c)F_tx_t)] \quad (\text{A})$$

$$+ \mathbb{E}_{p(\theta|c)} \left[ -\text{Div}(p(x_t|\theta, c) L_t T \nabla \log \phi(x_t|\beta_t(\theta, c))) \right] \quad (\text{B}) \quad (175)$$

$$+ \mathbb{E}_{p(\theta|c)} \left[ \frac{1}{2} L_t L_t^T \text{Div}(\nabla p(x_t|\theta, c)) \right] \quad (\text{C}) \quad (176)$$

<sup>963</sup> Since all of the divergence and gradient operators depend only on xt, we can pass the expectation <sup>964</sup> through these terms. We can simplify each terms as follows:

$$(\mathbf{A}) \quad \mathbb{E}_{p(\theta|c)} [-\text{Div}(p(x_t|\theta, c)F_t x_t)] = -\text{Div}(p(x_t|c)F_t x_t) \quad (177)$$

(B)

$$\begin{aligned}
 \mathbf{(b)} \quad \mathbb{E}_{p(\theta|c)} \left[ -\text{Div}(p(x_t|\theta, c) L_t L_t^T \nabla \log \phi(x_t|\beta_t(\theta, c))) \right] &= -\text{Div} \left( \int p(\theta|c) p(x_t|\theta, c) L_t L_t^T \nabla \log \phi(x_t|\beta_t(\theta, c)) d\theta \right) \\
 &\quad (178) \\
 &= -\text{Div} \left( \int p(\theta|x_t, c) p(x_t|c) L_t L_t^T \nabla \log \phi(x_t|\beta_t(\theta, c)) d\theta \right) \\
 &\quad (179) \\
 &= -\text{Div}(p(x_t|c) L_t L_t^T \mathbb{E}_{p(\theta|x_t, c)} [\nabla \log \phi(x_t|\beta_t(\theta, c))]) \\
 &\quad (180)
 \end{aligned}$$

(C)

$$\mathbb{E}_{p(\theta|c)} \left[ \frac{1}{2} L_t L_t^T \text{Div}(\nabla p(x_t|\theta, c)) \right] = \frac{1}{2} L_t L_t^T \text{Div}(\nabla \mathbb{E}_{p(\theta|c)} [p(x_t|\theta, c)]) \quad (181)$$

$$= \frac{1}{2} L_t L_t^T \text{Div}(\nabla p(x_t|c)) \quad (182)$$

<sup>965</sup> Putting these terms back together, we get:

$$\frac{\partial p(x_t|c)}{\partial t} = -\text{Div}(p(x_t|c)) \underbrace{(F_t x_t + L_t L_t^T \mathbb{E}_{p(\theta|x_t, c)} [\nabla \log \phi(x_t|\beta_t(\theta, c))])}_{\text{recognize as drift term in Fokker-Planck equation}} + \frac{1}{2} L_t L_t^T \text{Div}(\nabla p(x_t|c)) \quad (183)$$

<sup>966</sup> We can see that the form of the Markovian projection SDE is given by:

$$dx_t = (F_t x_t + L_t L_t^T \mathbb{E}_{p(\theta|x_{t,c})} [\nabla \log \phi(x_t|\beta_t(\theta, c))]) dt + L_t dW_t \quad (184)$$

<sup>967</sup> Lastly because ϕ(xt|βt(θ, c)) is a Gaussian distribution with natural parameters βt(θ, c), its pdf is <sup>968</sup> given by:

$$\phi(x_t|\beta_t(\theta, c)) = \exp\{\langle t_c(x_t), \beta_t(\theta, c) \rangle - A(c, \theta)\} \quad (185)$$

(186)

<sup>969</sup> where tc(xt) is the sufficient statistic of the Gaussian distribution and A(c, θ) is the log partition <sup>970</sup> function. From this form, we can immediately see that the expectation around the score function <sup>971</sup> passes through to the natural parameters:

$$\mathbb{E}_{p(\theta|x_{t,c})} [\nabla \log \phi(x_t|\beta_t(\theta, c))] = \langle \nabla t_c(x_t), \mathbb{E}_{p(\theta|x_{t,c})} [\beta_t(\theta, c)] \rangle \quad (187)$$

If we let β ∗ t (xt, θ, c) = <sup>E</sup>p(θ|xt,c) [βt(θ, c)] and stop the gradient with respect to x<sup>t</sup> through β ∗ t <sup>972</sup> , then <sup>973</sup> we recover the desired result.

 Proposition 18 (Neural latent SDE). *Let* p(x<sup>t</sup>1:<sup>N</sup> , y1:<sup>T</sup> ) *be the joint distribution defined in Definition [2](#page-4-2) and suppose that* y = (yO, y<sup>U</sup> )*, where* O *and* U *are the times at which sequences are observed and unobserved, respectively. Then the neural latent SDE is the following piecewise SDE defined on the intervals* (tk, tk+1) *for* k = 1, . . . , N*:*

$$dx_t = (F_t x_t + L_t L_t^T \nabla \log \phi(x_t | \beta_t^*(x_t, x_{t_{1:k}}, y_{\mathcal{O}}))) dt + L_t dW_t, \quad (188)$$

$$\text{where } \beta_t^*(x_t, x_{t_{1:k}}, y_{\mathcal{O}}) = \mathbb{E}_{p(y_{\mathcal{O}}|x_t, x_{t_{1:k}}, y_{\mathcal{O}})} [\beta_t(y_{1:T})], \text{ and } t \in (t_k, t_{k+1}) \quad (189)$$

β ∗ t (xt, x<sup>t</sup>1:<sup>k</sup> <sup>978</sup> , yO) *is the Bayes estimator of* β<sup>t</sup> *using the current state* xt*.*

*Proof.* The result follows directly from Lemma [6](#page-28-1) by choosing θ = y<sup>U</sup> and c = x<sup>t</sup>1:<sup>k</sup> <sup>979</sup> .

# <sup>980</sup> G.3 General form of Markovian projection ODE

 Lemma 7 (General form of Markovian projection ODE). *Suppose that* p(θ|c) *is a probability distribution over* θ *given some extra conditioning information* c *and* p(xt|θ, c) *is the marignal distribution of a generalized linear stochastic interpolant whose base linear SDE is given by* dx<sup>t</sup> = Ftxtdt + LtdWt*. Then the Markovian projection ODE is defined as the probability flow ODE of the Markovian projection SDE and is given by:*

$$\frac{dx_t}{dt} = F_t x_t + \frac{1}{2} L_t L_t^T (\nabla \log \phi(x_t | \beta_t^*(x_t, \theta, c)) - \nabla \log \phi(x_t | \alpha_t^*(x_t, \theta, c))) \quad (190)$$

*where* β ∗ t (xt, θ, c) = <sup>E</sup>p(θ|xt,c) [βt(θ, c)] *and* α ∗ t (xt, θ, c) = <sup>E</sup>p(θ|xt,c) <sup>986</sup> [αt(θ, c)] *are Bayes estimators* <sup>987</sup> *of the forward and backward messages to time* t *using* x<sup>t</sup> *respectively.*

<sup>988</sup> *Proof.* Recall that the definition of the probability flow ODE of an SDE of the form dx<sup>t</sup> = ut(xt)dt+ <sup>989</sup> LtdW<sup>t</sup> is given by [\[Song et al., 2021\]](#page-10-0):

$$\frac{dx_t}{dt} = u_t(x_t) - \frac{1}{2}L_t L_t^T \nabla \log p(x_t|c) \quad (191)$$

<sup>990</sup> Plugging in drift of the Markovian projection SDE in Lemma [6,](#page-28-1) and the score function of p(xt|c) in <sup>991</sup> Proposition [17,](#page-27-4) we get the desired result.

# <sup>992</sup> H Message Passing Implementation Details

 We devise a careful implementation of message passing to ensure numerical stability. There are many different ways to implement message passing. For example, [\[Särkkä et al., 2006\]](#page-10-9) parameterizes the potentials in the standard form of Gaussians and uses Kalman filtering [\[Kalman, 1960\]](#page-11-15) to obtain the forward messages and does not directly compute the backward messages, but instead uses the Rauch-Tung-Striebel smoother [\[Rauch et al., 1965\]](#page-11-16) to blend the forward and backward message computations to obtain the smoothed potentials. Alternatively, [\[Fox, 2009,](#page-11-17) [Johnson and Linderman,](#page-11-18) [2015\]](#page-11-18) utilize a natural parameterization of the potentials in order to have simple message passing updates. Our implementation requires that we can express both total uncertainty, and total certainty, in a variable in order to be able to work with incomplete, or missing data, and to condition exactly on variables. To do this, we adopt a mixed parametrization that contains the mean of the Gaussian and precision matrix so that we can express total uncertainty using a precision matrix of 0 and total certainty in the mean value by using a symbolic infinity. We also use symbolic zeros to mitigate accumulation of errors when perform message passing on long chains of latent variables without any evidence.

### <sup>1007</sup> H.1 Numerical stability considerations

<sup>1008</sup> Before we look at the implementation details, we will look at what considerations we need to make <sup>1009</sup> for the implementation of these operations in a numerically stable way. Recall that the transition <sup>1010</sup> distribution of an LTI-SDE is given by

$$\phi(x_{t+s}|x_t) = N(x_{t+s}|A_s x_t, \Sigma_s) \quad (192)$$

<sup>1011</sup> where

$$\begin{bmatrix} A_s & \Sigma_s A_s^{-T} \\ 0 & A_s^{-T} \end{bmatrix} := \exp\{\begin{bmatrix} F & LL^T \\ 0 & -F^T \end{bmatrix} s\} \quad (193)$$

<sup>1012</sup> and that potential functions can be written in natural or standard form as:

$$\phi(x) = \exp\{-\frac{1}{2}x^T Jx + x^T h - \log Z\} \quad (194)$$

$$= \exp\{-\frac{1}{2}x^T \Sigma^{-1} x + x^T \Sigma^{-1} \mu - \log Z\} \quad (195)$$

where Σ = J −1 and µ = J −1 <sup>1013</sup> h. We assume that the time intervals between consecutive variables are bounded and nonzero so that Σs, As, and A−<sup>T</sup> s <sup>1014</sup> are numerically stable. We also assume that the <sup>1015</sup> covariance matrices that the user specifies for the node potentials, e.g. Σ or J, are well conditioned. We do not assume that Σ −1 s , Σ <sup>−</sup><sup>1</sup> nor J −1 <sup>1016</sup> are well conditioned. These assumptions are made to <sup>1017</sup> accomodate operations that a user might perform in practice. For example, a user may choose to <sup>1018</sup> express 0 certainty in a variable by setting Σ → ∞ or J = 0 and can choose to express 0 uncertainty <sup>1019</sup> by setting Σ = 0 or J → ∞. Furthermore, if a user chooses to discretize an SDE at points where s is small, or even exactly 0, then Σ<sup>s</sup> is close to 0 and so Σ −1 s <sup>1020</sup> can be very large. To account <sup>1021</sup> for these considerations, we use symbolic computation to represent matrices that are 0 or ∞ as <sup>1022</sup> needed. Furthermore, we use three different parameterizations of the Gaussian to ensure that we can handle all cases. We use the standard parameterization, (µ, Σ), natural parameterization [<sup>3</sup>](#page-30-2) <sup>1023</sup> , (J = Σ−<sup>1</sup> , h = Σ−<sup>1</sup>µ), and mixed parameterization (J = Σ−<sup>1</sup> <sup>1024</sup> , µ). For brevity, we will not include <sup>1025</sup> the updates for the normalizing constant log Z in our pseudocode.

<sup>3</sup>The true natural parameters are scaled by − 2

# <sup>1026</sup> H.2 Message passing pseudocode

<sup>1027</sup> In Appendix [D](#page-15-0) we identified the key operations that are needed to perform variable elimination in the <sup>1028</sup> sequential and parallel settings (see Appendices [D.1](#page-15-1) and [D.2\)](#page-16-0). These operations are:

 1. An "add" operation adds the parameters of two potential functions together (code in Ap- pendix [H.3\)](#page-31-2). 2. An "update" operation that absorbs a potential function into a transition function (defined in Definition [5](#page-15-3) and code in Appendix [H.3\)](#page-31-3). 3. A "marginalize" operation that marginalizes out a variable from a Gaussian joint distribution. In practice, we fuse this with the "update" operation (code in Appendix [H.3\)](#page-31-1). 4. A "reverse" operation that reverses the direction of a transition (code in Appendix [H.3\)](#page-31-4). 5. A "chain" operation that chains two transition functions (defined in Eq. [\(40\)](#page-16-2) and code in Appendix [H.3\)](#page-32-1).

<sup>1038</sup> In Appendix [H.3,](#page-32-0) Appendix [H.3,](#page-32-2) Appendix [H.3,](#page-32-3) and Appendix [H.3](#page-33-2) we provide pseudocode for <sup>1039</sup> message passing that involves these operations.

# <sup>1040</sup> H.3 Update rules

Now we provide pseudocode for the update rules.

Algorithm 1 Add

- 1. Require: potential functions ϕ<sup>1</sup> and ϕ<sup>2</sup>
- 2. (J1, h1) = to\_natural(ϕ1)
- 3. (J2, h2) = to\_natural(ϕ2)
- 4. Return from\_natural((J<sup>1</sup> + J2, h<sup>1</sup> + h2))

1041

Algorithm 2 Update

- 1. Require: potential function ϕ and transition ϕk+1|<sup>k</sup>
- 2. (J, µ) = to\_mixed(ϕ)
- 3. (A, u, Σ) = ϕk+1|<sup>k</sup>
- 4. R = J(I + ΣJ) −1
- 5. S = ΣR
- 6. T = I − S
- 7. ϕ¯ <sup>k</sup>+1|<sup>k</sup> = (T A, T u + Sµ, TΣ)
- 8. ϕ¯ = from\_mixed((A<sup>T</sup> R<sup>T</sup> A, A−<sup>1</sup> (µ − u)))
- 9. Ψk+1,k = (ϕ¯ <sup>k</sup>+1|k, <sup>ϕ</sup>¯)
- 10. Return Ψk+1,k

Algorithm 3 Update and marginalize

- 1. Require: potential function ϕ and transition ϕk+1|<sup>k</sup>
- 2. (\_, <sup>ϕ</sup>¯) = Update(ϕ, ϕk+1|k)
- 3. Return ϕ¯

Algorithm 4 Reverse

- 1. Require: transition ϕk+1|<sup>k</sup>
- 2. (A, u, Σ) = ϕk+1|<sup>k</sup>
- 3. A¯ = A−<sup>1</sup>
- 4. u¯ = −A−<sup>1</sup>u
- 5. Σ =¯ A−<sup>1</sup>ΣA−<sup>T</sup>
- 6. Return (A, ¯ u, ¯ Σ) ¯

Algorithm 5 Chain

- 1. Require: transition functions ϕk|k−<sup>1</sup> and ϕk+1|<sup>k</sup>
- 2. Ak, uk, Σ<sup>k</sup> = ϕk+1|<sup>k</sup>
- 3. Ak−1, uk−1, Σk−<sup>1</sup> = ϕk|k−<sup>1</sup>
- 4. A = AkAk−<sup>1</sup>
- 5. u = Akuk−<sup>1</sup> + u<sup>k</sup>
- 6. Σ = Σ<sup>k</sup> + AkΣk−1A<sup>T</sup> k
- 7. Return (A, u, Σ)

Algorithm 6 BackwardMessagePassing

- 1. Require (ϕ2|1, . . . , ϕN|N−1) and (ϕ1, . . . , ϕ<sup>N</sup> )
- 2. Initialize β<sup>N</sup> = 0
- 3. For k = N, . . . , 2:
  - (a) Ψk,k−<sup>1</sup> = Update(ϕk|k−1, ϕ<sup>k</sup> + βk)
  - (b) βk−<sup>1</sup> = Marginalize(Ψk,k−1)
- 4. Return (β1, . . . , β<sup>N</sup> )

Algorithm 7 ParallelBackwardMessagePassing

- 1. Require (ϕ2|1, . . . , ϕN|N−1) and (ϕ1, . . . , ϕ<sup>N</sup> )
- 2. In parallel, for k = N, . . . , 2:
  - (a) Ψk,k−<sup>1</sup> = Update(ϕk|k−1, ϕk)
- 3. (Ψ1:<sup>N</sup> , . . . , ΨN−1:<sup>N</sup> ) = AssociativeScan(Chain, Ψ2,1, . . . , ΨN,N−1)
- 4. In parallel, for k = N − 1, . . . , 1:
  - (a) β<sup>k</sup> = Marginalize(Ψk:<sup>N</sup> )
- 5. β<sup>N</sup> = 0
- 6. Return (β1, . . . , β<sup>N</sup> )

Algorithm 8 ForwardMessagePassing

- 1. Require (ϕ2|1, . . . , ϕN|N−1), (ϕ1, . . . , ϕ<sup>N</sup> ) and use\_parallel
- 2. For k = 1, . . . , N − 1:
  - (a) ϕk|k+1 = Reverse(ϕk+1|k)
- 3. If use\_parallel:
  - (a) MessagePassing = ParallelBackwardMessagePassing
- 4. Else:
  - (a) MessagePassing = BackwardMessagePassing
- 5. (α<sup>N</sup> , . . . , α1) = MessagePassing((ϕN−1|<sup>N</sup> , . . . , ϕ1|2),(ϕ<sup>N</sup> , . . . , ϕ1))
- 6. Return (α1, . . . , α<sup>N</sup> )

Algorithm 9 AssociativeScan (Even number of elements only)

- 1. Require: operator ⊕, elements (t1, t2, . . . , tn) where n is a power of 2
- 2. If n == 1:
  - (a) Return t<sup>1</sup>
- 3. In parallel, for k = 1, . . . , n/2:
  - (a) p<sup>k</sup> = t2k−<sup>1</sup> ⊕ t2<sup>k</sup>
- 4. (r2, r4, . . . , rn) = AssociativeScan(⊕,(p1, p2, . . . , pn/2))
- 5. In parallel, for k = 1, . . . , n/2 − 1:
  - (a) r2k+1 = r2<sup>k</sup> ⊕ t2k+1
- 6. r<sup>1</sup> = t<sup>1</sup>
- 7. Return (r1, r2, . . . , rn)

# <sup>1042</sup> I Dataset details

 We used two synthetic datasets and five real-world datasets for our experiments - a synthetic noisy double pendulum and synthetic sine wave datasets, and real world datasets for modeling stocks, energy, etth, mujoco, and fmri datasets. For all of our experiments, we use an 80/10/10 split for the training, validation, and test sets. We adopted two different approaches to generate these splits, one for then the dataset only containd a single time series, and one for when the dataset containd multiple time series. For datasets that only contain a single time series, such as the noisy double pendulum, stocks, etth and fmri datasets, we split our data into training, validation, and test sets by splitting the series into three contiguous segments for the training, validation, and test sets respectively, using the 80/10/10 split, and then construct windowed batches of a fixed length for each of the training, validation, and test sets.

# <sup>1053</sup> J Model implementation details

# <sup>1054</sup> J.1 Neural network architecture and training details

 To ensure a fair comparison, we use nearly the exact same neural network architectures and training procedures for all of the models. The architecture that we use is an encoder-decoder transformer architecture where each transformer has 10 layers, 32 heads and a hidden dimension of 128. In between each transformer layer we use a Wavenet convolution block that has 256 channels and uses a kernel size of 4. The observed sequence of variables is passed through the encoder and then used to condition the decoder as it processes the currently generated sequence. We did not do extensive architecture tuning and chose this model early on because it performed well enough for our experiments. We incorporated information about the times in each series by constructing

 a feature vector for each scalar time and concatenating it with the observed sequence of variables before passing the contatenation to the transformer. For the models that needed to be autoregressive, we used causal convolutions and causal attention masks to ensure that the Jacobian matrix of the model was lower triangular. See our code for full details.

Each of our models were trained on a single 2080ti GPU using a learning rate of 10−<sup>4</sup> using the adamw optimizer, linear warmup of 1000 steps, and an effictive batch size of 256 (we used a batch size of 64 and 4 gradient accumulation steps). For each experiment, we used 5 random seeds to initialize the model parameters and to split the data into training, validation, and test sets using an 80/10/10 split. We evaluated the objective function on the entire validation set every 1000 gradient updates and stopped training when the value of the objective function over the entire validation set stopped improving for 5 evaluations. We normalized the elements of each series by subtracting the mean and dividing by the standard deviation of the first, observed variable in the series to ensure that the elements of each series were on a similar scale.

#### J.2 Model details

 We implemented 8 different models, of which 6 are latent space forecasters and 2 are observation space forecasters. The baseline, observation space models, were trained to model p(yk+1:<sup>N</sup> |y1:k) while the latent space models were trained to model p(x1:<sup>N</sup> |y1:k). Of the latent space forecasters, 4 are CMFVI based models and while the last 2 are the same baseline models that we used for the observation space models, just trained on the latent process instead of the observed process.

 1. Baselines probabilistic forecasters (Trained to approximate p(yk+1:<sup>N</sup> |y1:k)): (a) Conditional Gaussian autoregressive model (b) Diffusion model 2. Latent probabilistic forecasters (Trained to approximate p(x1:<sup>N</sup> |y1:k)): (a) CMFVI models: i. MSE forecaster ii. Autoregressive MSE forecaster iii. Neural ODE iv. Neural SDE (b) Conditional gaussian autoregressive (c) Diffusion model

 The encoder networks in each model accept as input y1:<sup>k</sup> and output a context embedding that is used to condition the decoder. The decoder accepts as input a sequence of variables that are currently being generated and outputs a sequence of different quantities whose interpretation depends on the model. Next, we will describe each of the models that we implemented, what their decoder outputs are, what their training objective is, and how they generate samples.

 Conditional Gaussian autoregressive model The Gaussian conditional chains parameterize the distribution of the next variable in the sequence as a Gaussian distribution. The decoder transformer network outputs the mean and covariance of the next distribution for the entire sequence of generated variables at once. Since the decoder is autoregressive, the mean and covariance of the next distribution is found at the same position as the most recently generated variable. For the latent space model, the first variable is sampled from a CRF, of the same kind used to construct the latent process, that is conditioned on the observed variables. The model is trained to maximize the log likelihood of the unobserved sequence given the observed sequence.

 Diffusion model The diffusion model is trained using flow-matching [\[Lipman et al., 2023\]](#page-10-2) using a brownian bridge between a Gaussian random variable and the sequence of unobserved variables. This model is effectively the same as standard diffusion models for images, but applied to a flattened time series vector. The decoder transformer network outputs the vector field of the probability flow ODE that is used to simulate the process. Samples are generated by passing a sequence of Gaussian random variables of the same size as yk+1:<sup>N</sup> to an ODE solver that uses the vector field output by the decoder to simulate the process.

 MSE forecaster The MSE forecaster predicts the mean of the potential functions of the CRF used to construct the latent process. This model is trained to minimize the mean squared error between the predicted mean of each potential function, and the mean of the potential function of the target process. To generate samples from this model, we use the input y1:<sup>k</sup> to generate the means of the CRF potentials for the entire sequence of generated variables. We then sample from the CRF defined by these potentials to get a sample from this model.

 Autoregressive MSE forecaster This model is also a conditional Gaussian autoregressive model, except that the model only parameterizes the mean of each transition distribution, and not the covariance, because, as mentioned in (REF), when the covariance matrices of the potential functions do not depend on the values of y, then the covariance matrices are known analytically using Kalman smoothing. To train this model, we minimize the mean squared error between the means of the true transition distributions (using the entire observed sequence), p(xi+1|x<sup>i</sup> , y1:<sup>N</sup> ), and the mean predicted by our model for q(xi+1|x<sup>i</sup> , y1:k). We generate samples from this model using the same procedure as the one for the conditional Gaussian autoregressive model defined above.

 Neural ODE/SDE We designed a novel parameterization of neural process models based on flow- based generative models in order to be able to use the same autoregressive transformer architecture as the other models, and also to make these scalable during training. Recall that a single step of training a flow-based generative model requires constructing a stochastic bridge between samples from a source and target distribution, sampling a random time in between the source and target time, sampling from the stochastic bridge at this time and then computing the probability flow ODE vector (or drift) of the bridge at this time. To extend this to time series, we must be able to perform this procedure for every pair of consecutive time points in a time series. To this end, we construct our transformer decoder to take as input the latent sequence that we are generating at the fixed set of times T := {t1, . . . , t<sup>N</sup> } and also elements of the latent sequence at (uniformly) random times inbetween these times, compute both the predicted and true control (either probability flow ODE vector or drift vector) at both the original and new times, and then return the mean squared error between the two.

 More formally, at training time suppose that we uniformly sample times in between the times in T as τ<sup>i</sup> ∼ U(t<sup>i</sup> , ti+1) for i = 1, . . . , N − 1. Then we can sample from the stochas- tic bridge at these times to get a sample from the model, x<sup>T</sup> <sup>+</sup><sup>τ</sup> ∼ p(x<sup>T</sup> <sup>+</sup><sup>τ</sup> |y1:<sup>N</sup> ), where x<sup>T</sup> <sup>+</sup><sup>τ</sup> := (x<sup>t</sup><sup>1</sup> , x<sup>τ</sup><sup>1</sup> , x<sup>t</sup><sup>2</sup> , x<sup>τ</sup><sup>2</sup> , . . . , x<sup>τ</sup>N−<sup>1</sup> , x<sup>t</sup><sup>N</sup> ). Our decoder transformer network takes as input x<sup>T</sup> <sup>+</sup><sup>τ</sup> and the embedding of y1:<sup>k</sup> from the encoder and outputs the probability flow ODE vector (if we are training a neural ODE) or the drift vector (if we are training a neural SDE) at the times T + τ . Our conditioned linear SDE library allows us to efficiently sample from p(x<sup>T</sup> <sup>+</sup><sup>τ</sup> |y1:<sup>N</sup> ), as well as compute the target control vector for the samples. We then compute the mean squared error between the predicted control vector and the target control vector to get our loss function. Since we ensure that our decoder network is autoregressive, we are able to compute the loss for the drift for the entire sequence at once, rather than having to compute for a single time step as is the case in existing implementations of these kinds of models (CITE).

 Our sample generation procedure simulates and ODE/SDE where the control vector at time t is given by the k'th element of the decoder output, where t ∈ (tk, tk+1). To begin, we first sample an initial point from pCRF(x<sup>t</sup><sup>0</sup> |y1:k). Note that this distribution is not equal to the target p(x<sup>t</sup><sup>0</sup> |y1:k), but is a reasonable approximation if k is reasonably large. Then we sample a set of times, τ , in between the times in T , like we do during training, to hold the intermediate variables that we store in order to feed the neural network an input that looks similar to the one used during training. The sampling procedure can be broken down into a sequence of k steps, where at step k ∈ [0, N), we simulate the variable x<sup>t</sup><sup>k</sup> forward in time from time t = tk, tk+1 to predict the next element of the sequence, x<sup>t</sup>k+1 . At the first step, we initialize the buffer of 2N − 1 elements (x<sup>t</sup><sup>0</sup> , 0, . . . , 0). Then for each step k ∈ [0, N), we simulate the variable x<sup>t</sup><sup>k</sup> forward in time from time t = tk−1, t<sup>k</sup> to predict the next element of the sequence, x<sup>t</sup><sup>k</sup> . The control of this simulation process is computed by passing the current buffer of variables to the decoder network. During simulation, we record the value of the process at the time, τk, so that at the end of step k, we update the buffer to include both x<sup>τ</sup><sup>k</sup> and x<sup>t</sup>k+1 . We then repeat this process for each step k ∈ [0, N) to get a sample from the model. See ?? for a discussion on the performance of this sampling procedure.

# NeurIPS Paper Checklist

 The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

 Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

 • You should answer [Yes] , [No] , or [NA] . • [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available. • Please provide a short (1-2 sentence) justification right after your answer (even for NA).

 The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

 The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

#### 1. Claims

 Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

 Justification: We introduced a generalization of the key elements of flow-based generative models that are relevant to the time series setting and showed how this can be used to construct related discrete time models.

Guidelines:

 • The answer NA means that the abstract and introduction do not include the claims made in the paper. • The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers. • The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings. • It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

# 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

 Justification: In section 3.4 and 3.6 we explained how the class of models we introduced are ultimately just mean squared error based conditional Gaussian models and therefore may not work as well in practice as their maximum likelihood counterparts on more stochastic data.

 • The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper. • The authors are encouraged to create a separate "Limitations" section in their paper. • The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be. • The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated. • The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon. • The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size. • If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness. • While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an impor- tant role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

# 3. Theory assumptions and proofs

 Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: We provide all of our proofs in the appendix.

Guidelines:

 • The answer NA means that the paper does not include theoretical results. • All the theorems, formulas, and proofs in the paper should be numbered and cross- referenced. • All assumptions should be clearly stated or referenced in the statement of any theorems. • The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition. • Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.

# 4. Experimental result reproducibility

 Question: Does the paper fully disclose all the information needed to reproduce the main ex- perimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

 Justification: We provide all of our implementation details in the appendix and provide our code as supplementary material.

Guidelines:

 • The answer NA means that the paper does not include experiments. • If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

 • If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable. • Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed. • While NeurIPS does not require releasing code, the conference does require all submis- sions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

### 5. Open access to data and code

 Question: Does the paper provide open access to the data and code, with sufficient instruc- tions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: We include our code as supplementary material.

#### Guidelines:

 • The answer NA means that paper does not include experiments requiring code. • Please see the NeurIPS code and data submission guidelines ([https://nips.cc/](https://nips.cc/public/guides/CodeSubmissionPolicy) [public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details. • While we encourage the release of code and data, we understand that this might not be possible, so âAIJNoâ ˘ A˘ ˙ I is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark). • The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ([https:](https://nips.cc/public/guides/CodeSubmissionPolicy) [//nips.cc/public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details. • The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc. • The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why. • At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable). • Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

 Question: Does the paper specify all the training and test details (e.g., data splits, hyper- parameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: We explain our experimental setting in the experiments section

Guidelines:

 • The answer NA means that the paper does not include experiments. • The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them. • The full details can be provided either with the code, in appendix, or as supplemental material.

### 7. Experiment statistical significance

 Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

 Justification: We provide the mean and standard error for the models trained in our experi-ments.

Guidelines:

 • The answer NA means that the paper does not include experiments. • The authors should answer "Yes" if the results are accompanied by error bars, confi- dence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper. • The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions). • The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors). • It should be clear whether the error bar is the standard deviation or the standard error of the mean. • It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified. • For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates). • If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

# 8. Experiments compute resources

 Question: For each experiment, does the paper provide sufficient information on the com- puter resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: We provide these details in the appendix.

Guidelines:

 • The answer NA means that the paper does not include experiments. • The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage. • The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

 • The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

# 9. Code of ethics

 Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?

Answer: [Yes]

Justification: We read the code of ethics.

Guidelines:

 • The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. • If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics. • The authors should make sure to preserve anonymity (e.g., if there is a special consid-eration due to laws or regulations in their jurisdiction).

### 10. Broader impacts

 Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: Our paper is mostly theoretical with limited societal impacts at this stage.

Guidelines:

 • The answer NA means that there is no societal impact of the work performed. • If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact. • Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations. • The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster. • The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology. • If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

# 11. Safeguards

 Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: Our method does not require safeguards.

 • Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters. • Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images. • We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

#### 12. Licenses for existing assets

 Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [NA]

Justification: We wrote the code for our models and datasets from scratch.

Guidelines:

 • The answer NA means that the paper does not use existing assets. • The authors should cite the original paper that produced the code package or dataset. • The authors should state which version of the asset is used and, if possible, include a URL. • The name of the license (e.g., CC-BY 4.0) should be included for each asset. • For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided. • If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, <paperswithcode.com/datasets> has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset. • For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided. • If this information is not available online, the authors are encouraged to reach out to the asset's creators.

### 13. New assets

 Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

Justification: N/A

Guidelines:

 • The answer NA means that the paper does not release new assets. • Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc. • The paper should discuss whether and how consent was obtained from people whose asset is used. • At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

# 14. Crowdsourcing and research with human subjects

 Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Guidelines:

 • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribu- tion of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

#### 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

 Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: N/A

Guidelines:

 • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

# 16. Declaration of LLM usage

 Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: We do not use LLMs in this work.

Guidelines:

 • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (<https://neurips.cc/Conferences/2025/LLM>) for what should or should not be described.