# On Flow-Based Generative Models For Probabilistic Forecasting

Anonymous Author(s)
Affiliation Address email

## Abstract

1 Flow-based generative models (FBGM) have emerged as a dominant approach to 2 generative modeling in many domains for their scalability and controllability, but 3 have notably not made the same impact on autoregressive probabilistic forecasting.

4 Although the methodology behind these models can be applied directly to the time 5 series setting, and in theory offers the potential to apply the advances in generative 6 modeling to time series, this direct approach is difficult to use in practice. In this 7 work, we investigate this methodological gap by generalizing the key elements of 8 flow-based generative modeling to the time series setting to devise a more practical 9 related algorithm. We show that FBGMs based on linear stochastic differential 10 equations are instances of a more general mean-field variational inference algorithm 11 for conditional exponential family distributions that constructs Bayes estimators 12 of natural parameters. This insight yields a family of mean-squared error based 13 latent probabilistic forecasters that contains a discrete time counterpart of FBGMs 14 for time series. We demonstrate that the models we develop inherit the convenient 15 theoretical properties of FBGMs while being easy to work with in practice.

17 Flow-based generative models (FBGM), including denoising diffusion, score based diffusion, and 18 flow matching models, have become the dominant approach to generative modeling. These models 19 represent a stochastic differential equation (SDE) that transforms samples from a known prior 20 distribution into samples from an unknown target distribution, and often use a different recipe 21 for solving the generative modeling problem compared to traditional approaches. This alternative 22 approach is highly scalable [Ramesh et al., 2022, Podell et al., 2023, Saharia et al., 2022], can leverage 23 conditioning information in flexible ways [Dhariwal and Nichol, 2021, Ho and Salimans, 2022], and 24 can be controlled in order to incorporate user defined dynamics [Liu et al., 2024, Domingo-Enrich 25 et al., 2024, Havens et al., 2025]. Furthermore, FBGMs are capable of learning from paired data. If x0 26 and x1 are samples from an unknown joint distribution p(x0, x1), then one can use the same approach 27 to construct an SDE whose transition distribution from t = 0 to t = 1 is p(x1|x0) [De Bortoli et al.,
28 2023]. Given this capability, it directly follows that this approach could, in principle, be used to construct an SDE to model time series data. If p(x1:N ) = p(x1)QN−1 k=1 29 p(xk+1|x1:k) represents the 30 unknown distribution of time series data, then each of the transition terms, p(xk+1|x1:k), can be 31 interpreted as a target distribution for a FBGM in the paired data setting where the data pairs are 32 consecutive elements of the time series, (xk+1, xk), and the previous elements x1:k−1 can be thought 33 of as extra conditioning information. In theory, learning this kind of model for time series would 34 inherit the scalability and controllability that FBGMs possess, allowing practitioners to port over 35 the recent advances in generative modeling to time series applications. However, this approach has 36 surprisingly only recently been explored [Chen et al., 2024a, Tamir et al., 2024, Park et al., 2024, 37 Chen et al., 2024b] even though diffusion based time series models have been studied for several Submitted to 39th Conference on Neural Information Processing Systems (NeurIPS 2025). Do not distribute.

## 16 **1 Introduction**

38 years [Yang et al., 2024, Meijer and Chen, 2024]. We attribute this gap to the practical numerical 39 difficulties associated with training and sampling from these models as one must first learn, and 40 then simulate, a stochastic differential equation, with potentially non-smooth dynamics, over a long 41 time domain compared to the short time domain encountered in standard generative modeling. To 42 address this problem, we develop a discrete time version of Neural SDEs derived from FBGMs 43 that are founded on the same theoretical principles, while being substantially easier to work with 44 in practice. We do this by generalizing two key elements needed to construct FBGMs, stochastic 45 interpolation and the Markovian projection, to the time series setting, where they become Gaussian 46 condition random fields and a form of mean-field variational inference respectively. We construct a 47 family of latent probabilistic time series models that are closely related to existing time series models, 48 including MSE based non-probabilistic forecasters and conditional Gaussian autoregressive models, 49 and compare their performance on various latent probabilistic forecasting problems.

## 50 **2 Background**

51 We will first review how flow-based generative models are constructed and then build intuition for 52 how to go about generalizing this construction to the time series setting. Suppose that p(y0, y1) is a 53 joint distribution over a source and target random variable. The (paired) generative modeling problem is to find a parametric approximation of p(y1|y0)
1 54 . Flow-based generative models solve this problem 55 by constructing, and then learning, a *latent* SDE whose transition distribution from times t = 0 to 56 t = 1 is p(y1|y0). There are three steps involved in constructing and learning this SDE - **stochastic** 57 **interpolation**, the **Markovian projection**, and **matching**.

58 **Stochastic interpolation** [Albergo and Vanden-Eijnden, 2023] is used to interpolate between proba59 bility distributions by defining interpolations between their samples. For example, consider the joint 60 distribution p(x0, xt, x1), where xt = (1 − t)x0 + tx1 and (x0, x1) ∼ p(x0, x1). By the definition 61 of xt, it is true that p(xt=1) = p(x1), and also that p(xt=1|x0) = p(x1|x0), so we verify that the 62 marginal distribution of xt interpolates between p(x0) and p(x1). In practice, one assumes that at 63 times t = 0 and t = 1, x0 := y0 and x1 := y1 so that p(xt) is an interpolation between p(y0) and 64 p(y1).

65 A popular method for constructing stochastic interpolants, which we use in this paper, is conditioning 66 a user-defined base SDE, whose diffusion coefficient does not depend on the current state, to start at 67 x0 and end at x1. This SDE takes the form dxt = bt(xt)dt + LtdWt where bt(xt) is the drift of this 68 base SDE and Lt is the diffusion coefficient. This SDE is used to construct a joint distribution of 69 the form p(x0, xt, x1) = p(xt|x0, x1)p(x0, x1) where p(xt|x0, x1) is the probability of xt when the 70 base SDE has been conditioned to start at x0 and end at x1. In order to solve the generative modeling 71 problem of p(x1|x0), FBGMs are constructed as an SDE whose marginal distribution is p(xt|x0).

72 This is accomplished using the **Markovian projection**. 73 **Proposition 1** (Markovian projection SDE [Shi et al., 2024]). Let p(x1|x0) be a conditional distribu74 tion over target variables given source variables and let p(xt|x0, x1) *denote the distribution of the* 75 base SDE dxt = bt(xt)dt + LtdWt when conditioned to start at x0 and end at x1. The "Markovian projection SDE" is an SDE whose marginal distribution, denoted by q
∗
76 (xt|x0) *is equal to* p(xt|x0).

77 *It is given by:*

$\mathbb{M}$ c. 
dxt = (bt(xt) + LtL
T
t Ep(x1|x0,xt)[∇ log p(x1|x0, xt)])dt + LtdWt (1)
78 See Prop 3. of [De Bortoli et al., 2023] for a proof. Proposition 1 is a solution to the paired generative modeling problem because q
∗
79 (xt=1|x0) = p(x1|x0) := p(y1|y0). Given a sample from the source 80 distribution, x0 ∼ p(x0), we can simulate the SDE from t = 0 to t = 1 to generate a sample from the 81 target distribution. However, this SDE contains an intractable drift term that depends on the posterior 82 distribution of x1 given x0 and xt. This is addressed using a **matching** learning objective. For 83 example, in score matching, [Vincent, 2011, Song et al., 2021], one writes the drift in the following 84 variational form:

$$\nabla\log q^{*}(x_{t}|x_{0})=\operatorname*{argmin}_{s_{t}(x_{t},x_{0})}\mathbb{E}_{p(x_{0},x_{1},x_{t})}\left[\left\|L_{t}L_{t}^{T}\nabla\log p(x_{1}|x_{0},x_{t})-s_{t}(x_{t},x_{0})\right\|^{2}\right].$$
st(xt,x0)
2ó(2)
$$(1)$$
$$(2)^{\frac{1}{2}}$$

85 If s(xt, x0; θ) is parameterized by a neural network, then one can minimize this expectation using
86 the standard machine learning toolkit to find the Markovian projection SDE. However, obtaining a 87 Monte Carlo estimate of the expectation for stochastic gradient descent requires being able to sample
88 from p(x0, x1, xt), which requires simulation of the base SDE. As such, the base SDE is chosen so
89 that this distribution is tractable. After training is complete, then the flow-based generative model is
given by the SDE dxt = (bt(xt) + LtL
T
t
90 st(xt, x0))dt + LtdWt. In general, matching algorithms,
91 such as score matching, drift matching and bridge matching, are algorithms for learning the Bayes 92 estimator of a random variable because of the well known relationship between posterior expectations 93 and mean squared error [Jaynes, 2003]:
Proposition 2 (Bayes estimate of parameter). Let p(z, θ) *be a joint distribution and let* θ
∗
94 (z) be 95 the Bayes estimate of θ based on z *under the squared error risk. Then the Bayes estimate takes the* 96 *following two forms:*
$$\theta^{*}(z)=\mathbb{E}_{p(\theta|z)}[\theta]=\operatorname*{argmin}_{f(z)}\ \mathbb{E}_{p(z,\theta)}\left[\|f(z)-\theta\|^{2}\right]$$
2(3)
$$({\mathfrak{I}})$$
97 See Appendix C.3 for a derivation. In score matching, one would have z = (x0, xt) and θ = 98 ∇ log p(x1|x0, xt), while other matching approaches, such as flow matching [Albergo and Vanden99 Eijnden, 2023, Lipman et al., 2023, Liu et al., 2023] and bridge matching [Shi et al., 2024].

100 Given the strong theoretical, interpretability, and empirical results of FBGMs, one might expect 101 that a direct application to time series would inherit the same benefits. However, this approach has 102 surprisingly only recently been explored [Chen et al., 2024a,b, Tamir et al., 2024, Park et al., 2024] 103 even though diffusion based time series models have been studied in a different manner for several 104 years [Yang et al., 2024, Meijer and Chen, 2024]. We attribute this gap to the challenges that the time 105 series setting presents to flow-based methods compared to settings such as image generation. In the 106 standard image generation setting, there is no coupling between the prior and data distributions, and 107 so one can learn SDEs that can be easily simulated with a few number of function evaluations [Liu 108 et al., 2023, Pooladian et al., 2023]. However, SDEs that are constructed to model time series data 109 present a challenge during inference due to compounding numerical errors that are attributed to either 110 a mismatch between the learned model and data, or due to the numerical solver itself, get accumulated 111 during generation which can lead to poor performance in practice. Discrete time autoregressive 112 models, on the other hand, do not suffer from these issues to the extent that Neural SDEs do and are 113 much more widely used in practice. With this in mind, we aim to understand find a discrete time 114 version of FBGMs for time series that will work better in practice.

## 115 **3 Method**

116 We present a generalization of the FBGM construction for the time series setting.

## 117 **3.1 Generalized Linear Stochastic Interpolation**

118 Recall that stochastic interpolation constructs a distribution over a latent stochastic process, which 119 we denote by x, that is sampled from a base SDE that is conditioned to start at x0 := y0 and end at 120 x1 := y1. Our generalization of stochastic interpolation is founded on the observation that many 121 of the base SDEs used in practice are linear SDEs, and that the FBGM recipe is unchanged if we 122 introduce Gaussian potential functions to relax the endpoint conditions. Since linear SDEs have 123 Gaussian transition distributions, they can naturally be combined with these Gaussian potentials to 124 construct a Gaussian conditional random field. This conditional random field will serve as our tool 125 for stochastic interpolation, which we call "generalized linear stochastic interpolation".

Let yτ1:Tdenote time series data that is generated by an unknown distribution p(yτ1:T
126 ). For brevity, 127 we assume that τ1:T is the same for all time series, but note that our theory accommodates datasets 128 with series sampled at different times. We will construct, and perform inference, in the distribution p(x|yτ1:T
129 ), which we will obtain by conditioning a linear SDE on user defined Gaussian potential functions. The potential function at time tk ∈ R will be denoted by ϕ(xtk |θtk(yτ1:T)), where θtk 130 the the natural parameter of the Gaussian that arbitrarily depends on yτ1:T
131 . See Appendix C for a review of 132 exponential family distributions. We also use the notation ϕk+1|k(xk+1|xk) = N(xk+1|Axk + u, Σ)
133 to denote a Gaussian transition distribution from xk to xk+1 with state transition matrix A, bias 134 vector u and covariance matrix Σ.

y0 y1 xp(x|y0, y1)
y 1 p(x| (y 1 : 4 ))
(xtk| tk (y 1 : 4 ))
y x 3 y 0 y 2
Figure 1: Generalized stochastic interpolation incorporates Gaussian potential functions to relax the endpoint conditions of stochastic interpolation and is applied to time series data.

## 135 **3.1.1 Gaussian Conditional Random Fields**

136 Chain structured Gaussian CRFs are a tractable class of probabilistic models that are widely used in 137 time series modeling (CITE):
138 **Definition 1** (Conditional Random Field [Lafferty et al., 2001, Sutton et al., 2012]). Let x1:N *be a* 139 sequence of random variables, ϕk+1|k(xk+1|xk) *be a set of Gaussian transition distributions between* 140 consecutive variables, and ϕ(xk|θk) *a set of Gaussian potential functions with natural parameters* 141 θk ∈ θ*. A conditional random field (CRF) is a probability distribution given by:*

$$p(x_{1:N}|\theta)\propto\prod_{k=1}^{N-1}\phi_{k+1|k}(x_{k+1}|x_{k})\prod_{k=1}^{N}\phi(x_{k}|\theta_{k})$$
$$\quad(4)$$
$$(S)$$
$$(6)$$
.$+\,\beta_{k+1})$  . 
142 Due to the chain-structure of p(x1:N |θ) and the fact it is jointly Gaussian, inference can be performed 143 efficiently using message passing. The backward messages, defined below, will play a significant role 144 in our theory:
145 **Proposition 3** (Backward messages). The k*'th backward message associated with the CRF in* 146 *Definition 1 is defined with the following recurrence relation:*

$$\phi(x_{k-1}|\beta_{k-1})=\int\phi_{k|k-1}(x_{k}|x_{k-1})\phi(x_{k}|\theta_{k}+\beta_{k})d x_{k},\quad\beta_{N}=0$$
$$\beta_{k}=\Phi_{k,k}$$

147 where θk+1 +βk+1 denotes the direct sum of θk+1 and βk+1*. This recurrence also uniquely identifies* 148 a function, denoted by Φk,k+1 *that performs the parameter updates as:*
βk = Φk,k+1(θk+1 + βk+1) (6)

## 155 **3.1.2 Linear Time-Invariant Stochastic Differential Equations**

156 We will use linear-time invariant SDEs to construct the transition distributions of continuous time 157 CRFs. Linear time-invariant SDEs (LTI-SDEs) are SDEs of the form dxt = F xtdt + LdWt, where 158 the drift matrix F and diffusion coefficient matrix L are constant with respect to t and xt. LTI-SDEs 159 have the convenient property that their transition distribution is available in closed form [Särkkä and 160 Solin, 2019, Singhal et al., 2023]. The transition distribution from xt to xt+s, where s > 0 is an 161 increment of time, is given by

$$\phi_{t+s|t}(x_{t+s}|x_{t})=N(x_{t+s}|A_{s}x_{t},\Sigma_{s}),\quad\text{where}\begin{bmatrix}A_{s}&\Sigma_{s}A_{s}^{-T}\\ 0&A_{s}^{-T}\end{bmatrix}:=\exp\left\{\begin{bmatrix}F&LL^{T}\\ 0&-F^{T}\end{bmatrix}s\right\}\tag{7}$$

4 149 Note that each βk is a function of θk+1:N . See Appendix D for a full derivation of sequential and 150 parallel message passing, and Appendix H for pseudo code and implementation considerations. 151 Although we do not focus on the forward messages, they are defined with analogous recurrence 152 relations to the backward messages and can be used to extend our methodology to flow-matching 153 models for time series forecasting (see Corollary 5). CRFs offer an efficient way to model the latent 154 variables at a fixed set of times, but are not immediately suited for continuous time. 162 We use LTI-SDEs for their tractability, but note that our theory is completely compatible with more 163 general linear SDEs. One can directly plug in this transition distribution into a CRF in Definition 1 to 164 obtain a conditional random field over a continuous time domain. However, we can be more general. 165 In the next proposition, we highlight a relationship between conditioned linear SDEs and CRFs 166 ([Särkkä et al., 2006, Särkkä and Solin, 2019]):
167 **Proposition 4** (Conditioned LTI-SDE). Let ϕt+s|t(xt+s|xt) *be the transition distribution of the* LTI-SDE dxt = F xtdt + LdWt *and let* {ϕ(xtk|θtk 168 )}tk∈R *be potential functions at times in the set* 169 R*. Then the piecewise-linear SDE,*

$$d x_{t}=(F x_{t}+L L^{T}\nabla\log\phi(x_{t}|\beta_{t}))d t+L d W_{t},\quad x_{t_{1}}\sim\phi(x_{t_{1}}|\beta_{1}+\theta_{1})$$

170 where t ∈ (tk, tk+1) and tk, tk+1 ∈ R, has a joint distribution at the times t1:N = T ⊇ R *that is* 171 *given by a CRF:*

$$(8)$$
$$p(x_{t_{1:N}}|\theta)\propto\prod_{t_{k}\in\mathcal{T}}\phi_{t_{k+1}|t_{k}}(x_{t_{k+1}}|x_{t_{k}})\prod_{t_{k}\in\mathcal{R}}\phi(x_{t_{k}}|\theta_{t_{k}})$$
$$(9)$$

ϕ(xtk|θtk) (9)
where βt = Φt,tk+1 (θtk+1 + βtk+1 172 ).

173 See appendix Appendix E.1 for the full proof and Corollary 5 for a nice expression for the associated 174 probability flow ODE in terms of both the forward and backward messages. Proposition 4 suggests 175 that a practical way to work with conditioned linear SDEs in practice is convert them into CRFs on a 176 discretization of the time domain so that inference can be performed via message passing. This results 177 in the ability to sample and perform inference in linear SDEs O(log |T |) time on parallel compute 178 [Hassan et al., 2021, Corenflos et al., 2021, Smith et al., 2023]. The conditioned SDE Proposition 4 is our main tool for stochastic interpolation as it gives us the ability to sample from p(x|θ(yτ1:T
179 )) at 180 an arbitrary discretization of the time domain.

## 181 **3.2 Target Probabilistic Model For Fbgm**

182 Recall that in the FBGM recipe, we used the stochastic interpolation to construct a joint distribution 183 over the interpolant and the data, p(y0, xt, y1), before performing the Markovian projection. We can take the same step here to construct a joint distribution over yτ1:T
184 and x using the data distribution, p(yτ1:T) and the distribution of the interpolant, p(x|yτ1:T) := p(x|θ(yτ1:T
185 )).

Definition 2 (Target joint distribution). Let p(yτ1:T
186 ) be the distribution of observed time series data and let p(x|yτ1:T
187 ) be the distribution of the generalized linear stochastic interpolant, which is the distribution of a linear SDE conditioned on the user defined potential functions {θtk
(yτ1:T
188 )}tk∈R at 189 the times R, as in Proposition 4. Then the induced joint distribution over x at the times t1:N = *T ⊃ R*
and yτ1:T
190 *is given by:*

$$p(x_{t_{1:N}},y_{r_{1:T}})=p(y_{r_{1:T}})\left({\frac{1}{Z(y_{r_{1:T}})}}\prod_{t_{k}\in T}\phi_{t_{k+1}|t_{k}}(x_{t_{k+1}}|x_{t_{k}})\prod_{t_{k}\in\mathcal{R}}\phi(x_{t_{k}}|\theta_{t_{k}}(y_{r_{1:T}}))\right).$$
|θtk(yτ1:T))!(10)
where Z(yτ1:T
) *is the partition function of* p(xt1:N |yτ1:T
191 ). 192 Before continuing, it is crucial that we understand this joint distribution and the role it plays in 193 the FBGM recipe. Unlike the standard approach to generative modeling where one defines a joint 194 distribution by defining a prior over the latent variable and a likelihood distribution over the data, the FBGM uses an alternate construction to build p(x, yτ1:T
195 ) using the data distribution directly. 196 Furthermore, the tools FBGMs employ are fundamentally designed for probabilistic inference in x instead of yτ1:T
197 . Since x is completely user designed through the choice of base LTI-SDE and 198 potential functions, we are able to solve a wide range time series problems.

199 Suppose we split each sequence of data into observed and unobserved portions, yτ1:T = (yO, yU ),
200 where yO is a subsequence that we observe at both train and test time while yU is only observed at training time, as is the case in time series forecasting.2 201 The ability to perform inference in 202 p(x|yO) would solve a general latent probabilistic forecasting problem that reduces to the stan203 dard forecasting problem if the Gaussian potential functions are chosen as dirac delta functions -
2This also covers the imputation setting, but we do not explore this in the interest of keeping a narrow scope.

$$(10)$$

z x
*(z)
z x

## 210 **3.3 Neural Latent Sde For Latent Probabilistic Forecasting**

ϕ(xtk |θtk
(yτ1:T
)) := δ(xtk − ytk 204 ). For example, if one chooses the LTI-SDE to be the Wiener ve205 locity model [Särkkä and Solin, 2019, Särkkä et al., 2006] and potential functions of the form ϕ(xtk |θ(yτ1:T
)) ∝ N(xtk |ytk
, σ2 206 I), then inference in p(x|yO) corresponds to forecasting the smoothed position and velocity of the particle whose positions were observed at yτ1:T
207 . However, p(x|yO) is intractable because p(yτ1:T
208 ) is arbitrary. To this end, we develop variational inference 209 algorithms for this task. 211 The first inference algorithm we develop is a direct extension of flow-based generative models to the 212 latent probabilistic forecasting setting. For a fixed discretization of the time domain, we can treat consecutive latent variables (xtk, xtk+1 213 ) as elements of a paired dataset with the previous elements xt1:k−1 214 and observations yO as extra conditioning information. This lets us directly apply the existing 215 FBGM recipe to construct a conditional, piecewise SDE to solve the latent probabilistic forecasting 216 problem.

Proposition 5 (Neural latent SDE). Let p(xt1:N , yτ1:T
217 ) *be the joint distribution defined in Definition 2* 218 *and suppose that* yτ1:T = (yO, yU ), where O and U *are the times at which sequences are observed* 219 *and unobserved at test time, respectively. Then the neural latent SDE is the following piecewise SDE:*
dxt = (Ftxt + LtL
T
t ∇ log ϕ(xt|β
∗
t(xt, xt1:k, yO)))dt + LtdWt, (11)
where β
∗
t(xt, xt1:k, yO) = Ep(yU |xt,xt1:k
,yO)[βt(yτ1:T)] , and t ∈ (tk, tk+1) (12)

## 228 **3.4 Constrained Mean-Field Variational Inference**

229 Next we introduce our main contribution which is the variational inference algorithm underlying 230 FBGMs, which we call "constrained mean-field variational inference". Given a conditional expo231 nential family distribution p(x|*z, θ*), CMFVI constructs a variational approximation of p(x|z) that is given by p(x|*z, θ*∗(z)) where θ
∗
232 (z) is the Bayes estimator of θ given z. We first introduce CMFVI in 233 an abstract way and then show how it can be used to do variational inference on the latent probabilistic 234 forecasting distribution, p(xt1:N |yO). 235 Suppose that z is a random variable, θ ∼ p(θ|z) is the natural parameter of an exponential family 236 distribution, and x ∼ p(x|*z, θ*) is a random variable drawn from a conditional exponential family of 237 the form p(x|*z, θ*) = exp{⟨tz(x), θ⟩−A(*z, θ*)}. For intuition, assume that x represents the future of a 238 stochastic process, z represents its past , and θ represents the parameters of this process. Furthermore, 222 See Appendix G.2 for a proof and Appendix G for the general constructions of the score function, 223 Markovian projection SDE and probability flow ODE. By construction, Proposition 5 can be used to 224 solve the latent probabilistic forecasting problem because it has the correct joint distribution over the 225 latent space. Furthermore, its form is almost identical to that of its base LTI-SDE in Proposition 4, except that its parameter, β
∗
226 , is the Bayes estimator of a backward message. We will show next that 227 models of this form can be derived by solving a constrained mean-field variational inference problem.

Furthermore, the transition distribution of this SDE from time tk to tk+1 is p(xtk+1 |xt1:k 220 , yO). We will use q Neural-SDE 221 *to denote the path measure associated to this SDE.*

239 suppose that the parameters are only available at training time so that at test time, sampling x given 240 z requires the ability to sample from p(x|z). Our goal is to predict the future of the process given 241 its past, which requires the ability to sample from p(x|z), however this distribution is intractable 242 because p(θ|z) is arbitrary. To this end, we introduce a variational approximation of p(x|z) using an 243 algorithm closely resembling mean field variational inference, which we call "constrained mean field 244 variational inference" (CMFVI):
245 **Theorem 1** (Constrained mean field VI solution). Let p(x|z, θ) ∝ exp{⟨tz(x), θ⟩ − A(*z, θ*)} be
246 a conditional exponential family distribution with θ ∼ p(θ|z). The constrained mean field VI
approximation of p(x|z)*, denoted by* q
∗
247 (x|z)*, is defined as follows:*
$q^{*}(x|z)=\underset{q(x|z)}{\operatorname{argmin}}\operatorname{KL}\left[q(x|z)p(\theta|z)\right][p(x,\theta|z)]$  $$=p(x|z,\theta^{*}(z)),\quad\text{where}\theta^{*}(z)=\mathbb{E}_{p(\theta|z)}\left[\theta\right]$$
KL [q(x|z)p(θ|z)∥p(*x, θ*|z)] (13)
$$(13)$$

$$(14)$$
∗(z) = Ep(θ|z)[θ] (14)
248 See Appendix F.1 for a proof, Lemma 4 for equivalent expressions for the objective involving KL[q
∗
249 (x|z)∥p(x|z)] and a term resembling the mutual information between x and θ given z. The parameter θ
∗
250 (z) is the Bayes estimator of θ given z and by Proposition 2 can be learned using mean 251 squared error minimization, provided that it is possible to sample from p(*z, θ*). While this variational 252 approximation is tractable, it seems restrictive because it is a conditional random field and only exact 253 when θ and x are conditionally independent given z. However, this may not be a terrible assumption 254 in the time series setting. If the process is deterministic, then we should be able to compute x directly 255 from z without needing to know θ, and so this independence assumption will hold because one will 256 be able to compute the future values of the process directly from its past. In fact, in Corollary 8, 257 we show that a direct application of CMFVI to p(xt1:N |yO), by selecting x = xt1:N , z = yO and θ = θ(yτ1:T
258 ), exactly recovers MSE based non-probabilistic forecasters, which are clearly capable of learning deterministic processes (see Corollary 8). We denote the model in Corollary 8 by q MSE 259 . In 260 general, provided that the process is not too stochastic, we might expect that given a long enough 261 history and a short enough prediction horizon that CMFVI could yield a reasonable approximation of 262 p(x|z), and perhaps with an infinitely short prediction horizon we may recover something exactly.

263 This intuition motivates the use of CMFVI for learning the autoregressive factors of p(xt1:N |yO) in 264 order to construct an autoregressive model to solve the probabilistic forecasting problem.

Suppose that p(xtk |xt1:k−1 265 , yO) is one of the autoregressive factors of the latent forecasting distribution p(xt1:N |yO). We can use CMFVI to approximate each of the k factors by setting x = xtk 266 ,
z = (xt1:k−1, yO) and θ = θ(yτ1:T
267 ):
268 **Proposition 6** (CMFVI transition approximation). Let p(xt1:N |yO) be the target distribution and consider its k*'th autoregressive factor* p(xtk|xt1:k−1 269 , yO)*. Then the CMFVI transition approximation* 270 *is given by:*
q transition(xtk |xt1:k−1
, yO) ∝ ϕtk|tk−1
(xtk |xtk−1
)ϕ(xtk |β
∗
tk
(xt1:k−1
, yO)) (15)
where β
∗
tk
(xt1:k−1
, yO) = Ep(yU |xt1:k−1
,yO)[βtk
(yτ1:T
)] *is the Bayes estimate of* βtk
(yτ1:T
271 )*, which is* defined using the message passing update operator Φtk,tk+1 272 *from Definition 7 as:*

$$\beta_{t_{k}}={\begin{cases}\Phi_{t_{k},t_{k+1}}(\beta_{t_{k+1}}(y_{\tau_{1:T}})+\theta_{t_{k+1}}(y_{\tau_{1:T}}))&{{\mathrm{if~}}t_{k+1}\in{\mathcal{R}}}\\ \Phi_{t_{k},t_{k+1}}(\beta_{t_{k+1}}(y_{\tau_{1:T}}))&{{\mathrm{otherwise}}}\end{cases}}$$

273 See Proposition 6 for a proof. The form of Proposition 6 almost exactly matches the transition distribution of p(xt1:N |yτ1:T
274 ) in Proposition 12, except that the backward messages are replaced with their Bayes estimators. We will use q transition 275 to construct an autoregressive approximation model that 276 will be a discrete time version of the Markovian projection SDE. 277 To use CMFVI to construct a discrete time version of FBGMs for time series, we will need to 278 make the assumption that the covariances of the potential functions are independent of the values of yτ1:T
279 . This assumption holds in both the data space forecasting setting where we use dirac delta 280 potential functions, and also in the case where the CRF is constructed as a linear dynamical system with constant observation noise. In this setting, it is also possible to rewrite q Neural SDE 281 in a more 282 interpretable form where the only unknown value is the mean of the next backward message: 283 **Corollary 1** (Neural latent SDE using potentials with fixed covariances). If the covariance matrices associated with q Neural SDE are constant with respect to y*, then the SDE associated with* q Neural SDE 284 is:
dxt = (Ftxt + LtL
T
t ∇ log N(xt|µ β t
∗(xt, xt1:k−1
, yO), Σ
β t))dt + LtdWt (17)

$$(16)$$
$$(17)^{\frac{1}{2}}$$
$\square$
where t ∈ (tk−1, tk), Σ
β tis the covariance of ϕ(xt|βt(yτ1:T)) and µ
∗
t(xt, xt1:k−1 285 , yO) *is the Bayes* 286 *estimator for it's mean.*
The result follows directly from converting βtk 287 from natural parameters to standard parameters 288 of a Gaussian and the linear equivariance of the Bayes estimator Appendix F.2. Note that by our assumption that the parameters of the potential functions do not depend on yτ1:T, Σ
β t 289 can be computed by performing message passing on p(xt1:N |∅τ1:T
), where ∅τ1:T
290 is an empty (or random) sequence sampled at the same times as yτ1:T
291 .

## 292 **3.5 Discrete Time Markovian Projection**

293 We propose an conditional Gaussian autoregressive model whose transition distributions are given by q transition, which we denote by q MSE-AR 294 . We will directly relate it to Markovian projection SDE
q Neural-SDE by associating q MSE-AR with a piecewise linear SDE that closely resembles q Neural-SDE 295 .

296 **Proposition 7** (Autoregressive CMFVI solution). Let p(xt1:N |yO) be the target distribution, as297 sume that the covariance matrices of its potential functions are constant with respect to y. The autoregressive model whose transitions are CMFVI solution, denoted by q MSE-AR 298 *is given by:*
q MSE-AR(xt1:N |yO) ∝ p(xt1 |yO)Y
tk∈T
ϕtk|tk−1
(xtk |xtk−1
)N(xtk |µ β tk
∗(xt1:k−1
, yO), Σ
β tk
) (18)
where Σ
β tk and µ β tk
∗(xt1:k−1, yO) *are the same as in Corollary 1. Furthermore,* q MSE-AR 299 *has the same* 300 joint distribution over xt1:N *as the following piecewise linear SDE:*
dxt = (Ftxt + LtL
T
t ∇ log N(xt|µ β t
∗(xt1:k−1
, yO), Σ
β t
))dt + LtdWt, xt1 ∼ p(xt1 |yO) (19)
where µ
∗ t
(xt1:k−1
, yO) *is the Bayes estimator for the mean of* βt(yτ1:T
) = Φt,tk
(βtk+1 (yτ1:T
)), Σ
β t 301 is 302 its covariance matrix and t ∈ (tk−1, tk) for k = 2*, . . . , T*.

303 See Appendix F.3 and Definition 9 for a proof. A comparison of the piecewise linear SDE associated with q MSE-AR with the piecewise SDE associated to q Neural-SDE reveals why we interpret q MSE-AR 304 as the 305 discrete time version of the Markovian projection SDE. We can see that the only difference between the two SDEs are their Bayes estimators for µ β t(yτ1:T
306 ):
q MSE-AR :µ β t
∗(xt1:k
, yO) = Ep(yU |xt1:k
,yO)
îµ β t
(yτ1:T
)ó q Neural-SDE :µ β t
∗(xt, xt1:k, yO) = Ep(yU |xt,xt1:k
,yO)
îµ β t(yτ1:T)
ó 307 The only difference between the two Bayes estimators is their dependence on the current state xt.

If xt does not carry more information about yU compared to what is already available from xt1:k 308 and yO, then we can expect that q MSE-AR and q Neural-SDE 309 will model nearly the same distribution. As 310 we will show in our experiments, this is something that one can expect in the time series setting because data is usually sampled frequently enough where the extra capacity that q Neural-SDE 311 has over q MSE-AR may not make enough of an impact in practice to warrant using q Neural-SDE 312 in practice. We introduced three different CMFVI based time series models - q MSE 8, q MSE-AR 7 and q Neural-SDE 313 1 314 which use CMFVI to joint distribution, transition distributions, and infinitesimal transitions of the 315 target distribution respecitvely. All of these models are Gaussian, and are therefore closely related to 316 existing time series models.

## 317 **3.6 Connection To Traditional Time Series Models**

318 The CMFI-based time series models that we have developed all have an autoregressive Gaussian 319 structure which makes them related to existing time series models. First, when one chooses potential functions to align with the data times R = τ1:T , then q MSE 320 is identical to MSE based non-probabilistic 321 forecasters, which are are trained to predict the future of a time series, yU given an observed history, yO. Next, q MSE-AR 322 is a conditional Gaussian autoregressive model that is trained to minimize a 323 mean-squared error based objective. This model is in the same family as conditional Gaussian models that are trained for maximum likelihood, but differ in that q MSE-AR 324 can be though of parameterizing 325 the mean of each transition distribution whereas maximum likelihood models parameterize both the 326 mean and covariance. Overall, the models that we have developed can be seen as mean-squared 327 error based time series models for probabilistic forecasting where the uncertainty in the models only 328 depend on the time in between observations and not the observations themselves.

Brusselator Double Pendulum FitzHugh Lorenz Lotka Van der Pol

MSE 3.04 ± 0.69 9.03 ± 0.34 27.75 ± 4.50 5.91 ± 0.60 2.16 ± 1.18 -0.77 ± 0.01 AR-MSE 0.49 ± 0.18 0.61 ± 0.02 15.08 ± 1.18 8.82 ± 0.29 0.12 ± 0.25 -0.59 ± 0.01 AR-MLE (Latent) 3.39 ± 1.91 0.43 ± 0.01 13.10 ± 2.48 8.49 ± 1.05 0.23 ± 0.27 -0.70 ± 0.00 AR-MLE (Obs.) 3.79 ± 2.05 0.42 ± 0.01 13.35 ± 2.47 7.77 ± 0.76 0.11 ± 0.32 -0.70 ± 0.00 FBGM (Latent) 2.06 ± 1.12 0.56 ± 0.03 6.15 ± 0.75 12.11 ± 0.80 0.17 ± 0.42 -0.69 ± 0.00 FBGM (Obs.) 0.93 ± 0.29 0.51 ± 0.01 11.67 ± 1.80 5.28 ± 0.50 0.47 ± 0.67 -0.71 ± 0.00

(a) Negative log likelihood (lower is better)

Brusselator Double Pendulum FitzHugh Lorenz Lotka Van der Pol

MSE 0.56 ± 0.02 0.99 ± 0.00 2.15 ± 0.16 1.09 ± 0.01 0.50 ± 0.02 0.48 ± 0.00 AR-MSE 0.59 ± 0.01 1.16 ± 0.01 3.58 ± 0.27 1.25 ± 0.01 0.55 ± 0.03 0.52 ± 0.00 AR-MLE (Latent) 0.65 ± 0.04 1.27 ± 0.01 2.32 ± 0.17 1.26 ± 0.03 0.59 ± 0.03 0.52 ± 0.01 AR-MLE (Obs.) 0.66 ± 0.05 1.27 ± 0.01 2.37 ± 0.13 1.26 ± 0.04 0.58 ± 0.03 0.52 ± 0.01 FBGM (Latent) 0.62 ± 0.05 1.20 ± 0.01 2.34 ± 0.17 1.09 ± 0.03 0.55 ± 0.03 0.49 ± 0.01 FBGM (Obs.) 0.64 ± 0.02 1.17 ± 0.01 2.29 ± 0.15 1.08 ± 0.02 0.55 ± 0.03 0.51 ± 0.00

(b) Normalized root mean squared error (lower is better)

Table 1: Evaluation metrics for our models (MSE and AR-MSE) for probabilistic forecasting compared to baseline models trained in both the latent and data spaces.

## 329 **4 Experiments**

330 We compare the performance of our models versus other approaches to time series modeling in latent 331 probabilistic forecasting on dynamical system datasets. We created 6 synthetic datasets representing 332 noisy observations of dynamical systems. Our models used a Wiener velocity model as our base SDE
and emission potentials of the form ϕ(xtk |θtk
(yτ1:N )) ∝ N(ytk |xtk
, σ2I). Our models, q MSE 333 and q MSE-AR 334 , and the baseline models were trained to approximate the probabilistic forecasting distribution p(xtk+1:N |xt1:k 335 , yO). See Appendix I for details about the datasets, parameters used for stochastic interpolation and other implementation details. Our models, q MSE and q MSE-AR 336 , were each trained 337 using mean squared error to learn their respective Bayes estimators. We used a non-autoregressive 338 FBGM trained with flow-matching and a conditional Gaussian chain trained for maximum likelihood as our baselines. We trained each of these baselines in two ways to learn p(xtk+1:N |xt1:k 339 , yO). First, 340 we trained these baseline models to learn the latent distribution directly by learning directly from 341 samples from p(xt1:N |yτ1:N ). Second, we trained these models in the observation space to learn 342 p(yU |yO) directly, and at test time, produced latent samples xtk+1:N by first sampling yU using yO,
343 and then sampling from the stochastic interpolator using the full sequence (yO, yU ). For all of the autoregressive models, instead of learning the distribution of the first point p(xtk+1 344 |yO), we produced 345 a heuristic sample by sampling from the stochastic interpolant that is only conditioned on yO. We 346 always chose tk+1 to be a time contained in O in order for this heuristic to give reasonable samples.

347 For each model, we trained using 5 different seeds and report the (empirical) negative log likelihood 348 and normalized root mean squared error of samples from the true distribution, p(xtk+1:N |yU ), using 349 32 sampled trajectories from each model, averaged over each dimension and time step. In all of our 350 models, we used a one layer recurrent neural network with a GRU cell as we found that this model 351 had sufficient model capacity to represent our data. Our results are displayed in Table 1. We can see 352 that the AR

## 353 **5 Conclusion**

354 We showed how to generalize the elements that comprise flow-based generative models to the 355 time series setting and uncovered a discrete time version of these models that shares convenient 356 properties that FBGMs possess, including a closed form solution and Bayes estimator parameters.

357 Our framework also encapsulates other existing time series models, including MSE based non358 probabilistic forecasters and conditional Gaussian autoregressive models. This unified perspective 359 sheds light into the role that FBGMs can play in time series.

## 360 **References**

361 Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text362 conditional image generation with clip latents. *arXiv preprint arXiv:2204.06125*, 1(2):3, 2022. 363 Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe 364 Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image 365 synthesis. *arXiv preprint arXiv:2307.01952*, 2023. 366 Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar 367 Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic 368 text-to-image diffusion models with deep language understanding. *Advances in neural information* 369 *processing systems*, 35:36479–36494, 2022. 370 Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. *Advances* 371 *in neural information processing systems*, 34:8780–8794, 2021. 372 Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. *arXiv preprint arXiv:2207.12598*, 373 2022. 374 Guan-Horng Liu, Yaron Lipman, Maximilian Nickel, Brian Karrer, Evangelos Theodorou, and Ricky 375 T. Q. Chen. Generalized schrödinger bridge matching. In *The Twelfth International Conference on* 376 *Learning Representations*, 2024. URL https://openreview.net/forum?id=SoismgeX7z.

377 Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, and Ricky TQ Chen. Adjoint matching: 378 Fine-tuning flow and diffusion generative models with memoryless stochastic optimal control. 379 *arXiv preprint arXiv:2409.08861*, 2024. 380 Aaron Havens, Benjamin Kurt Miller, Bing Yan, Carles Domingo-Enrich, Anuroop Sriram, Brandon 381 Wood, Daniel Levine, Bin Hu, Brandon Amos, Brian Karrer, et al. Adjoint sampling: Highly 382 scalable diffusion samplers via adjoint matching. *arXiv preprint arXiv:2504.11713*, 2025. 383 Valentin De Bortoli, Guan-Horng Liu, Tianrong Chen, Evangelos A Theodorou, and Weilie Nie. 384 Augmented bridge matching. *arXiv preprint arXiv:2311.06978*, 2023. 385 Yifan Chen, Mark Goldstein, Mengjian Hua, Michael S. Albergo, Nicholas M. Boffi, and Eric 386 Vanden-Eijnden. Probabilistic forecasting with stochastic interpolants and föllmer processes, 387 2024a. 388 Ella Tamir, Najwa Laabid, Markus Heinonen, Vikas Garg, and Arno Solin. Conditional flow matching 389 for time series modelling. In *ICML 2024 Workshop on Structured Probabilistic Inference* {\&}
390 *Generative Modeling*, 2024. 391 Byoungwoo Park, Hyungi Lee, and Juho Lee. Efficient modeling of irregular time-series with stochas392 tic optimal control. In *NeurIPS 2024 Workshop on Bayesian Decision-making and Uncertainty*, 393 2024. URL https://openreview.net/forum?id=KRtuDGFJzu. 394 Yu Chen, Marin Biloš, Sarthak Mittal, Wei Deng, Kashif Rasul, and Anderson Schneider. Recurrent 395 interpolants for probabilistic time series prediction. *arXiv preprint arXiv:2409.11684*, 2024b. 396 Yiyuan Yang, Ming Jin, Haomin Wen, Chaoli Zhang, Yuxuan Liang, Lintao Ma, Yi Wang, Chenghao 397 Liu, Bin Yang, Zenglin Xu, et al. A survey on diffusion models for time series and spatio-temporal 398 data. *arXiv preprint arXiv:2404.18886*, 2024.

399 Caspar Meijer and Lydia Y. Chen. The rise of diffusion models in time-series forecasting, 2024. 400 Michael Samuel Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic 401 interpolants. In *The Eleventh International Conference on Learning Representations*, 2023. URL
402 https://arxiv.org/abs/2209.15571. 403 Yuyang Shi, Valentin De Bortoli, Andrew Campbell, and Arnaud Doucet. Diffusion schrödinger 404 bridge matching. *Advances in Neural Information Processing Systems*, 36, 2024. 405 Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computa406 *tion*, 23(7):1661–1674, 2011. 407 Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben 408 Poole. Score-based generative modeling through stochastic differential equations. In *International* 409 *Conference on Learning Representations*, 2021. URL https://openreview.net/forum?id= 410 PxTIG12RRHS. 411 Edwin T Jaynes. *Probability theory: The logic of science*. Cambridge university press, 2003. 412 Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow 413 matching for generative modeling. In The Eleventh International Conference on Learning Repre414 *sentations*, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t. 415 Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate 416 and transfer data with rectified flow. In *The Eleventh International Conference on Learning* 417 *Representations*, 2023. URL https://openreview.net/forum?id=XVjTT1nw5z.

418 Aram-Alexandre Pooladian, Heli Ben-Hamu, Carles Domingo-Enrich, Brandon Amos, Yaron 419 Lipman, and Ricky T. Q. Chen. Multisample flow matching: Straightening flows with mini420 batch couplings. In *International Conference on Machine Learning*, 2023. URL https: 421 //api.semanticscholar.org/CorpusID:258418096. 422 John Lafferty, Andrew McCallum, Fernando Pereira, et al. Conditional random fields: Probabilistic 423 models for segmenting and labeling sequence data. In *Icml*, volume 1, page 3. Williamstown, MA, 424 2001. 425 Charles Sutton, Andrew McCallum, et al. An introduction to conditional random fields. *Foundations* 426 *and Trends® in Machine Learning*, 4(4):267–373, 2012. 427 Simo Särkkä and Arno Solin. *Applied stochastic differential equations*, volume 10. Cambridge 428 University Press, 2019. 429 Raghav Singhal, Mark Goldstein, and Rajesh Ranganath. Where to diffuse, how to diffuse, and how to 430 get back: Automated learning for multivariate diffusions. In *The Eleventh International Conference* 431 *on Learning Representations*, 2023. URL https://openreview.net/forum?id=osei3IzUia.

432 Simo Särkkä et al. *Recursive Bayesian inference on stochastic differential equations*. Helsinki 433 University of Technology, 2006.

434 Syeda Sakira Hassan, Simo Särkkä, and Ángel F García-Fernández. Temporal parallelization of 435 inference in hidden markov models. *IEEE Transactions on Signal Processing*, 69:4875–4887, 436 2021. 437 Adrien Corenflos, Zheng Zhao, and Simo Särkkä. Gaussian process regression in logarithmic time. 438 *arXiv preprint arXiv*, 2102, 2021. 439 Jimmy T.H. Smith, Andrew Warrington, and Scott Linderman. Simplified state space layers for 440 sequence modeling. In *The Eleventh International Conference on Learning Representations*, 2023. 441 URL https://openreview.net/forum?id=Ai8Hw3AXqks.

442 Calvin Luo. Understanding diffusion models: A unified perspective. *arXiv preprint arXiv:2208.11970*,
443 2022.

444 Sander Dieleman. Perspectives on diffusion, 2023. URL https://sander.ai/2023/07/20/ 445 perspectives.html.

446 Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised 447 learning using nonequilibrium thermodynamics. In *International conference on machine learning*, 448 pages 2256–2265. PMLR, 2015. 449 Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. *Advances in* 450 *neural information processing systems*, 33:6840–6851, 2020. 451 Tim Dockhorn, Arash Vahdat, and Karsten Kreis. Score-based generative modeling with critically452 damped langevin diffusion. In *International Conference on Learning Representations*, 2022. URL
453 https://openreview.net/forum?id=CzceR82CYc.

454 Tianrong Chen, Jiatao Gu, Laurent Dinh, Evangelos Theodorou, Joshua M. Susskind, and Shuangfei 455 Zhai. Generative modeling with phase stochastic bridge. In *The Twelfth International Conference on* 456 *Learning Representations*, 2024c. URL https://openreview.net/forum?id=tUtGjQEDd4. 457 Yaakov Bar-Shalom, X. Rong Li, and Thiagalingam Kirubarajan. *Estimation with Applications* 458 *to Tracking and Navigation*. John Wiley & Sons, New York, 2001. ISBN 9780471221272. 459 doi: 10.1002/0471221279. URL https://onlinelibrary.wiley.com/doi/book/10.1002/ 460 0471221279. 461 Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. *Advances* 462 *in neural information processing systems*, 34:21696–21707, 2021.

463 Marcel Kollovieh, Abdul Fatir Ansari, Michael Bohlke-Schneider, Jasper Zschiegner, Hao Wang, and 464 Yuyang Bernie Wang. Predict, refine, synthesize: Self-guiding diffusion models for probabilistic 465 time series forecasting. *Advances in Neural Information Processing Systems*, 36:28341–28364, 466 2023. 467 Xinyu Yuan and Yan Qiao. Diffusion-TS: Interpretable diffusion for general time series generation. 468 In *The Twelfth International Conference on Learning Representations*, 2024. URL https:// 469 openreview.net/forum?id=4h1apFjO99. 470 Marcel Kollovieh, Marten Lienen, David Lüdke, Leo Schwinn, and Stephan Günnemann. Flow 471 matching with gaussian process priors for probabilistic time series forecasting. In *The Thirteenth* 472 *International Conference on Learning Representations*, 2025. URL https://openreview.net/
473 forum?id=uxVBbSlKQ4. 474 Yang Hu, Xiao Wang, Lirong Wu, Huatian Zhang, Stan Z Li, Sheng Wang, and Tianlong Chen. Fm-ts: 475 Flow matching for time series generation. *arXiv preprint arXiv:2411.07506*, 2024. 476 Kashif Rasul, Calvin Seward, Ingmar Schuster, and Roland Vollgraf. Autoregressive denoising 477 diffusion models for multivariate probabilistic time series forecasting. In *International Conference* 478 *on Machine Learning*, pages 8857–8868. PMLR, 2021. 479 Macheng Shen and Chen Cheng. Neural sdes as a unified approach to continuous-domain sequence 480 modeling. *arXiv preprint arXiv:2501.18871*, 2025. 481 Ahmed El-Gazzar and Marcel van Gerven. Probabilistic forecasting via autoregressive flow matching.

482 *arXiv preprint arXiv:2503.10375*, 2025.

483 Matthew James Beal. *Variational algorithms for approximate Bayesian inference*. University of 484 London, University College London (United Kingdom), 2003. 485 Matthew James Johnson et al. *Bayesian time series models and scalable inference*. PhD thesis, 486 Massachusetts Institute of Technology, 2014. 487 Simo Särkkä and Ángel F García-Fernández. Temporal parallelization of bayesian smoothers. *IEEE* 488 *Transactions on Automatic Control*, 66(1):299–306, 2020. 489 Daphane Koller. *Probabilistic Graphical Models: Principles and Techniques*. The MIT Press, 2009. 490 Bernt Øksendal and Bernt Øksendal. *Stochastic differential equations*. Springer, 2003. 491 Rudolph Emil Kalman. A new approach to linear filtering and prediction problems. *Transactions of* 492 *the ASME–Journal of Basic Engineering*, 82(Series D):35–45, 1960. 493 H. E. Rauch, F. Tung, and C. T. Striebel. Maximum likelihood estimates of linear dynamic systems. 494 *AIAA Journal*, 3(8):1445–1450, 1965. 497 Matthew Johnson and Scott Linderman. pylds: Bayesian inference for linear dynamical systems. 498 https://github.com/mattjj/pylds, 2015. Accessed: 2025-05-07. 495 Emily Beth Fox. *Bayesian nonparametric learning of complex dynamical phenomena*. PhD thesis, 496 Massachusetts Institute of Technology, 2009.

## 499 **A Appendix**

500 The appendix contains proofs and implementation details for the main paper. It is organized as 501 follows: 508 - Sequential message passing (D.1) 509 - Parallel message passing (D.2) 510 - Basic probabilistic queries (D.4) 511 4. Conditioned linear SDEs (E) 512 - Conditioned linear SDEs (E.1) 513 - Basic probabilistic queries (E.2) 514 - Corresponding probability flow ODE (E.3) 515 5. Constrained mean field VI (F) 516 - Derivation (F.1) 517 - Bayes estimator equivariance (F.2) 518 - CMFVI time series models (F.3) 519 6. Flow-based generative models (G) 520 - Score function of FBGMs (G.1) 521 - General form of Markovian projection SDE (G.2) 522 - General form of Markovian projection ODE (G.3) 523 7. Message passing implementation details (H) 524 - Numerical stability considerations (H.1) 525 - Message passing pseudocode (H.2) 526 8. Dataset details (I) 527 9. Model implementation details (J)

## 528 **B Related Work**

529 There are numerous perspectives on flow-based generative models [Luo, 2022, Dieleman, 2023] and 530 even more variants of these models. At their core, these models start by constructing a stochastic 531 process that starts at a prior distribution and ends at the data distribution. Diffusion models use 532 progressive noising of data to build this map [Sohl-Dickstein et al., 2015, Ho et al., 2020, Song et al., 533 2021] via a simple SDE whose stationary distribution is Gaussian. On the other hand, flow-matching 534 models [Liu et al., 2023, Albergo and Vanden-Eijnden, 2023, Lipman et al., 2023] use a stochastic 535 bridge to build this map by conditioning a simple SDE to start at a point in the prior distribution and 536 end at the data distribution. The choice of simple SDE used in all of these models is a user-defined 537 choice that typically is a linear SDE, such as variance preserving SDE [Song et al., 2021], Brownian 538 motion, Ornstein-Uhlenbeck process, and others, due to their tractability as Gaussian processes 539 [Särkkä and Solin, 2019], and is even used to construct more exotic latent SDEs such as critically 540 damped langevin dynamics [Dockhorn et al., 2022, Chen et al., 2024c] or the Weiner velocity model 541 [Bar-Shalom et al., 2001, Särkkä et al., 2006]. In our paper, we abstract away these choices and 542 generally consider using linear SDEs to construct the initial map between distributions. There are a 543 few different ways to go from this initial stochastic process to a FBGM. A common way to construct 544 a FBGM from this is construct and optimize and ELBO for the likelihood of data under this initial 545 process [Kingma et al., 2021]. Alternatively, one can directly solve for the SDE whose marignal 546 distribution is that of this initial process [Song et al., 2021, Lipman et al., 2023] or define it as the 502 1. Related work Appendix B 503 2. Background Appendix C 504 - Exponential family distributions Appendix C.1 505 - Mean field variational inference Appendix C.2 506 - Bayes estimation Appendix C.3 507 3. Message passing (D) 547 SDE whose path measure is as close as possible to the initial process [Shi et al., 2024, De Bortoli 548 et al., 2023] in terms of KL divergence, called the Markovian projection. We adopt the latter view 549 over the ELBO view because it explicitly constructs a solution to the generative modeling problem 550 and is available in closed form while this is hidden in the ELBO formulation and show that the 551 solution to a mean field variational inference problem can be seen as an approximate discrete time 552 counterpart. 553 Flow-based generative models have been successfully applied to time series problems in a non554 *autoregressive* fashion [Kollovieh et al., 2023, Yuan and Qiao, 2024, Kollovieh et al., 2025, Hu 555 et al., 2024, Yang et al., 2024, Meijer and Chen, 2024]. These models transform the time series 556 generative modeling problem into the standard generative modeling problem used in image generation 557 by treating each time series as a single vector by concatenating all times together, and then learning a 558 map from a Gaussian vector of the same size to the data vector. These approaches can be conditioned 559 using guidance [Rasul et al., 2021, Dhariwal and Nichol, 2021, Ho and Salimans, 2022, Kollovieh 560 et al., 2023] which allows them to perform tasks such as forecasting and imputation. Our approach 561 differs from these in that we construct autoregressive models. 562 The class of models most relevant to our paper are autoregressive neural SDEs that are trained using 563 principles from flow-based generative models. [Chen et al., 2024a] uses a FÃ˝ullmer process to model 564 the transition distributions of the distribution of time series data, which is the same approach that we 565 adopt in our Neural SDE model. [Park et al., 2024] also learns a similar latent Neural SDE model that 566 uses a similar form of soft conditioning as us (through the use of emission potentials), and is trained 567 to maximize the likelihood of data. [Tamir et al., 2024] is also similar where they perform stochastic 568 interpolation using Gaussian processes and perform inference with Kalman smoothing as well, which 569 is a form of message passing. Finally, [Shen and Cheng, 2025] learns a more general SDE to learn 570 the distribution of time series data where the diffusion coefficient is not independent of the current 571 state and also maximize the likelihood of data. These related papers are all related to the Neural 572 SDE that we describe in our paper. Our main contributions are centered around investigating how to 573 apply the approach used to construct these continuous time models for creating similar discrete time 574 models. [El-Gazzar and van Gerven, 2025] used flow matching to learn the next state distribution of 575 time series data, but did not learn a FÃ˝ullmer process for this task and instead learned to transform a 576 Gaussian into the next state distribution.

## 577 **C Background** 578 **C.1 Exponential Family Distributions**

579 Our findings can be most easily written using exponential family distributions. Although we restrict 580 our attention to Gaussian distributions, the form of our results are most readable in natural parameter 581 space. 582 **Definition 3** (Exponential family distribution). *An probability distribution is in the exponential family* 583 *if its density function can be written in the following form:*
p(x|θ) = exp{⟨t(x), θ⟩ − A(θ)} (20)
584 where t(x) is called the sufficient statistic, θ the natural parameter and A(θ) *the partition function.* 585 The member of this family that we will use is the multivariate Gaussian distribution. A multivariate Gaussian with mean µ and covariance matrix Σ has the sufficient statistic t(x) = (*x, xx*T
586 ) and natural parameters θ = (−
1 2Σ
−1, Σ
−1µ). In practice, it is more convenient to drop the −
1 2 587 scaling term and work with the parameters (*J, h*) = (−Σ
−1, Σ
−1 588 µ), where J is the precision matrix of the distribution. 589 While these are not exactly the natural parameters, we will refer to them as so. Throughout this paper, 590 we will work with unnormalized Gaussian distributions, which we call "Gaussian potentials". We 591 use the notation ϕ(x|θ) to denote a Gaussian potential function over x with natural parameters θ. A
592 convenient property of the natural parameter form is that the score function takes a simple form.

∇ log ϕ(x|θ) = Jx − h (21)
593 Another Gaussian distribution that we will use extensively is the Gaussian transition distribution. We 594 write ϕk+1|k(xk+1|xk) = N(xk+1|Axk + u, Σ) to denote the Gaussian transition distribution from 595 xk to xk+1 with state transition matrix A, bias vector u and covariance matrix Σ.

$$\nabla\log\phi(x|\theta)=J x-h$$

## 596 **C.2 Mean Field Variational Inference**

597 Mean field variational inference is an approximate inference algorithm for probabilistic models. It's 598 main feature is that it's solution is available in a simple closed form expression. Let p(*x, θ*) be a joint 599 distribution over x and θ. The mean field variational problem is to find distributions, qx(x) and qθ(θ) 600 that minimize the KL divergence between qx(x)qθ(θ) and p(*x, θ*).

601 **Proposition 8** (Mean field variational inference for CRFs). Let p(θ) *be a distribution over* θ, p(x|θ) 602 be the CRF in Definition 1 and p(*x, θ*) = p(θ)p(x|θ) be the joint distribution over x and θ*. Then the* 603 *solutions to*

$$\operatorname*{argmin}_{q_{x}(x),q_{\theta}(\theta)}\operatorname{KL}\left[q_{x}(x)q_{\theta}(\theta)|p(x,\theta)\right]$$
$$(22)$$

KL [qx(x)qθ(θ)|p(*x, θ*)] (22)
604 *will satisfy:*

$$\begin{array}{l}{{q_{x}(x)\propto\exp\{\mathbb{E}_{q_{\theta}(\theta)}\left[\log p(x|\theta)\right]\}}}\\ {{q_{\theta}(\theta)\propto\exp\{\mathbb{E}_{q_{x}(x)}\left[\log p(\theta|x)\right]\}}}\end{array}$$
$$(23)$$
$$(24)$$
qx(x) ∝ exp{Eqθ(θ)[log p(x|θ)]} (23)
qθ(θ) ∝ exp{Eqx(x)[log p(θ|x)]} (24)
605 See [Beal, 2003] for a proof. Typical use cases of mean field VI use tractable classes of distributions 606 for p(θ) and p(x|θ) so that one can perform EM style, alternating updates to obtain the optimal q 607 distributions [Beal, 2003, Johnson et al., 2014]. However, in our setting, we will use mean field VI
608 differently. We will assume nothing about the form of p(θ), but will constrain the variational problem 609 by fixing qθ(θ) = p(θ).

## 610 **C.3 Bayes Estimation**

Lemma 1 (Bayes estimate of parameter). Let p(z, θ) *be a joint distribution and let* θ
∗
611 (z) *be the* 612 Bayes estimate of θ based on z *under the squared error risk. Then the Bayes estimate takes the* 613 *following two forms:*

$$\theta^{*}(z)=\mathbb{E}_{p(\theta|z)}[\theta]={\underset{f(z)}{\operatorname{argmin}}}\ \ \mathbb{E}_{p(z,\theta)}\left[\|f(z)-\theta\|^{2}\right]$$

614 *Proof.* Let L[f] be the loss function defined as follows:

$${\mathcal{L}}[f]=\mathbb{E}_{p(z)}\left[\|f(z)-\theta^{*}(z)\|^{2}\right]$$

Clearly, the minimizer of L[f] is θ
∗
615 (z). With a bit of rearranging and using Bayes rule, we can 616 rewrite L[f] as follows:

s follows:  $$\mathcal{L}[f]=\mathbb{E}_{p(z)}\left[\|f(z)-\theta^{*}(z)\|^{2}\right]$$ $$=\mathbb{E}_{p(z)}\left[\|f(z)\|^{2}\right]-2\mathbb{E}_{p(z)}\left[\langle f(z),\theta^{*}(z)\rangle\right]+\underbrace{\mathbb{E}_{p(z)}\left[\|\theta^{*}(z)\|^{2}\right]}_{\text{const.w.r.t.}f}$$
$$(25)$$
$$=\mathbb{E}_{p(z,\theta)}\left[\left|\left|f(z)\right|\right|^{2}\right]-2\mathbb{E}_{p(z)}\left[\left\langle f(z),\mathbb{E}_{p(\theta|z)}\left[\theta\right]\right\rangle\right]+\text{const.}$$ $$=\mathbb{E}_{p(z,\theta)}\left[\left|\left|f(z)\right|\right|^{2}\right]-2\mathbb{E}_{p(z,\theta)}\left[\left\langle f(z),\theta\right\rangle\right]+\text{const.}$$ $$\text{(complete the square)}$$ $$=\mathbb{E}_{p(z,\theta)}\left[\left|\left|f(z)-\theta\right|\right|^{2}\right]-\underbrace{\mathbb{E}_{p(z,\theta)}\left[\left|\left|\theta\right|\right|^{2}\right]}_{\text{const.w.r.t.}f}+\text{const.}$$
The minimizer of L[f] is unaffected by the constant terms, and so we have that θ
∗(z) = Ep(θ|z)
617 [θ] is 618 the solution to

$$\operatorname{argmin}_{f(z)}\ \mathbb{E}_{p(z,\theta)}\left[\|\theta-f(z)\|^{2}\right]$$

## 620 **D Message Passing**

621 In this section we will review message passing and identify the key operations that are needed to 622 perform message passing updates. We defer the discussion of numerically stable implementations of 623 these operations to Appendix H. First we'll identify the key operations that are needed to perform 624 message passing updates for the backward messages and then show how these operations can be used 625 to perform message passing updates for the forward messages. 626 At a high level, the sequential and parallel message passing algorithms are variable elimination 627 algorithms that eliminate different variables of the chain structured graph. The sequential algorithms 628 operates on individual nodes and begins at one of the ends of the chain and sequentially eliminate 629 variable at the end of the chain, whereas the parallel algorithm operates on pairs of nodes and 630 eliminates the middle variable of the pair. For example, a rough sketch of the sequential elimination 631 process looks like (0), 1, 2, 3, 4 → (1), 2, 3, 4 → (2), 3, 4 → (3), 4 → (4), where the parentheses 632 indicate the current node that is being processed. On the other hand, the parallel algorithm looks like 633 (0, 1), 2, 3, 4 → (0, 2), 3, 4 → (0, 3), 4 → (0, 4).

## 634 **D.1 Sequential Message Passing**

635 The sequential message passing updates for the backward messages can be written using the following 636 recurrence relation:

$$\phi(x_{k-1}|\beta_{k-1})=\int\phi_{k|k-1}(x_{k}|x_{k-1})\phi(x_{k}|\theta_{k})\phi(x_{k}|\beta_{k})d x_{k},\quad\beta_{N}=0$$

637 See Appendix H.3 for pseudocode. There are two operations on Gaussians that are needed to perform 638 these updates. The first is a "multiply" operation that takes two potential functions and returns a new 639 potential function, and the second is an "update" operation that absorbs a potential function into a 640 transition function.

641 **Definition 4** (Multiply). Let ϕ1(x) and ϕ2(x) *be potential functions over the same variable. Then* 642 *the "multiply" operation is defined as*

$$(26)$$
$$\phi_{1}(x)\phi_{2}(x)\mapsto{\hat{\phi}}(x)$$
$$(27)^{\frac{1}{2}}$$
ϕ1(x)ϕ2(x) 7→ ϕˆ(x) (27)
643 When ϕ1(x) and ϕ2(x) are parameterized using natural parameters, then the multiply operation simply 644 adds the natural parameters, i.e. if θ1 and θ2 are the natural parameters of ϕ1(x) and ϕ2(x), then 645 ϕ1(x|θ1)ϕ2(x|θ2) 7→ ϕ1(x|θ1 + θ2). We used this property to write the sequential message passing 646 updates for the backward messages ??. We do note that when one uses a different parameterization, 647 the multiply operation may look different. We will examples of this in Appendix H. 648 The second operation is the "update" operation, which absorbs a potential function into a transition 649 function. This operation is what handles the integral in the recurrence relation. 650 **Definition 5** (Update). Let ϕ(y|x) be a transition function and ϕ(y) *be a potential function over the* 651 *first variable. Then the "update" operation is defined as*

$$\phi(y)\phi_{y|x}(y|x)\mapsto{\hat{\phi}}_{y|x}(y|x){\hat{\phi}}(x)$$
$$(28)^{\frac{1}{2}}$$
ϕ(y)ϕy|x(y|x) 7→ ϕˆy|x(y|x)ϕˆ(x) (28)
where ϕˆy|x(y|x) and ϕˆ 652 (x) *are a new transition function and potential function, respectively.*
653 Essentially, the update operation performs a change of variables of the coupling of x and y on the 654 LHS. Furthermore, when the terms of the LHS are Gaussian, then the terms of the RHS are also 655 Gaussian. This allows us to perform the update operation in closed form (see Appendix H).

656 The multiply and update operations are sufficient to perform the sequential message passing updates 657 for the backward messages. For example, the backward message passing updates can be written as:
$$\int\phi_{k|k-1}(x_{k}|x_{k-1})\underbrace{\phi(x_{k}|\theta_{k})\phi(x_{k}|\beta_{k})}_{\mathrm{multiply}\to\phi(x_{k}|\theta_{k}+\beta_{k})}\,d x_{k}$$
dxk (29)
=
$$\underbrace{\phi(x_{k}|\theta_{k}+\beta_{k})\phi_{k|k-1}\big(x_{k}|x_{k-1}\big)}\ dx_{k}$$ update $\rightarrow\hat{\phi}_{k|k-1}(x_{k}|x_{k-1})\phi(x_{k-1}|\beta_{k-1})$
dxk (30)
$$(29)$$
$$(30)$$
$$=\underbrace{\int\hat{\phi}_{k|k-1}(x_{k}|x_{k-1})dx_{k}}_{\text{transition integrates to1}}\phi(x_{k-1}|\beta_{k-1})$$ $$=\phi(x_{k-1}|\beta_{k-1})$$
$$(31)$$
$$(32)$$
$$(33)$$
$$\phi(x_{k+1}|\alpha_{k+1})=\int\phi_{k+1|k}(x_{k+1}|x_{k})\phi(x_{k}|\theta_{k})\phi(x_{k}|\alpha_{k})d x_{k},\quad\alpha_{1}=0$$
665 Using this reverse operation, we can simply reverse the transition distributions and then find the
666 forward messages by using the same recurrence relation as for the backward messages:
 $ \int\underbrace{\phi_{k+1|k}(x_{k+1}|x_k)}_{\text{reverse}}\underbrace{\phi(x_k|\theta_k)\phi(x_k|\alpha_k)}_{\text{multiply}\to\phi(x_k|\theta_k+\alpha_k)}\ dx_k$  $ =\int\underbrace{\phi^T(x_k|x_{k+1})\phi(x_k|\theta_k+\alpha_k)}_{\text{update}\to\hat{\phi}^T(x_k|x_{k+1})\phi(x_{k+1}|\alpha_{k+1})}\ dx_k$  $ =\underbrace{\int\hat{\phi}^T(x_k|x_{k+1})dx_k\ \phi(x_{k+1}|\alpha_{k+1})}_{\text{transition}\text{integrates to1}}$  $ =\phi(x_{k+1}|\alpha_{k+1})$
$$(34)$$
$$(35)$$
$$(36)$$
$$(37)$$
$$(38)$$
$$\Psi(y,x)=\Psi(y|x)\Psi(x)$$
$$\Psi(y,x):=\int\Psi(y,z)\Psi(z,x)d z=:\Psi(y,z)\otimes\Psi(z,x)$$
ZΨ(*y, z*)Ψ(z, x)dz =: Ψ(y, z) ⊗ Ψ(*z, x*) (40)
$\eqref{eq:walpha}$. 

## 671 **D.2 Parallel Message Passing**

658 The forward messages can be computed in a similar manner. The forward messages are given by:

659 To find the forward messages, we can exploit the fact that our transition functions are Gaussian and
660 can therefore be reversed. This means that given a transition ϕ(y|x), we can find a reversed transition
ϕ
T
661 (x|y) that evaluates to the same value as ϕ(y|x) for all *x, y*
662 **Definition 6** (Reversed transition). Let ϕ(y|x) *be a transition function. Then the reversed transition* 663 *is defined as*
$$\phi^{T}(x|y)=\phi(y|x)$$
T(x|y) = ϕ(y|x) (34)
 $\phi^{2}\left(x|y\right)=\phi(y|x)$  $and\int\phi^{T}(x|y)dx=\int\phi(y|x)dx=1$. 
so that ϕ T(x|y) = ϕ(y|x) for all x, y and Rϕ 664 ϕ(y|x)dx = 1. 667 These message passing updates can be computed in O(N) time using the the multiply, update and 668 reverse operations. However, there is a more efficient way to compute the forward messages using 669 the parallel scan algorithm [Särkkä and García-Fernández, 2020] that reduces the complexity to 670 O*(log* N) on parallel compute. We will describe this algorithm in Appendix D.2. 672 In this section we will use slightly different notation to describe the parallel message passing 673 algorithm. We will avoid writing out the parameters of our potential functions and call them by their 674 parameter name. For example, instead of writing ϕ(xk|θk), we will write ϕk(xk) and instead of 675 writing ϕ(xk|βk), we will write β(xk).

676 The building block of the parallel message passing algorithm Särkkä and García-Fernández [2020] is 677 an unnormalized potential function over two variables, which we denote by Ψ(y, x). We assume that 678 Ψ(y, x) can be decomposed into a (normalized) transition distribution and an unnormalized potential 679 function:
Ψ(*y, x*) = Ψ(y|x)Ψ(x) (39)
680 Whenever we write Ψ(y|x), we are referring to a valid conditional probability distribution
(R
681 Ψ(y|x)dy = 1). Since Ψ(y, x) is jointly Gaussian over x and y, we are able to integrate out 682 variables in x and y and can also combine neighboring potentials into a new Gaussian potential. 683 These properties allow us to construct a chain operation over potentials that combines neighboring 684 potentials and then integrates out the common variable. We denote this chain operation by ⊗:
685 An important property of the chain operation is that it is associative due to the fact that we can swap 686 the order or integration (we will prove this in Appendix D.3). 687 A useful perspective of this chain operation is that it amounts to performing variable elimination on 688 the graph defined by the potentials, i.e. performs some sort of message passing [Koller, 2009]. With 689 this in mind, we can perform message passing by constructing the appropriate joint potentials:
690 **Proposition 9** (Parallel messages). Let ϕk+1|k and ϕk *be the potential functions for the CRF in* 691 Definition 1 and α and β *be the messages defined in Eqs.* (26) and (33)*. Then*

$$\alpha_{k}(x_{k})=\int\Psi_{1:k}^{b e d}(x_{k},x_{1})d x_{1}\quad a n d\quad\beta_{k}(x_{k})=\int\Psi_{k:N}^{b e d}(x_{N}|x_{k})d x_{N}$$

692 *where*

$$\Psi_{1:k}^{\text{fnd}}(x_{k},x_{1})=\bigotimes_{i=1}^{k-1}\phi_{i+1|i}(x_{i+1}|x_{i})\phi_{i}(x_{i})$$  _and $\Psi_{k:N}^{\text{fnd}}(x_{N}|x_{k})=\bigotimes_{i=N-1}^{k}\phi_{i+1|i}(x_{i+1}|x_{i})\phi_{i+1}(x_{i+1})$_
$$(41)$$
(42)  $\binom{43}{4}$  (43)  . 
693 See appendix Appendix D.3 for a proof and ?? for pseudocode. Since ⊗ is associative, we can 694 evaluate Eq. (42) in O(log N) time using the parallel scan algorithm [Särkkä and García-Fernández, 695 2020]. The rough idea is that on parallel compute, one can, in parallel, chain together consecutive 696 pairs of potentials and then recurse on these new chained potentials in order to eventually chain the 697 entire sequence. We provide pseudocode for this a special case of this algorithm in Appendix H.3.

Ψfwd 1:k
(xk, x1) and Ψbwd 698 k:N (xN |xk) can be thought of as the result of marginalization over the variables 699 between x1 and xk and xk and xN , respectively.

## 700 **D.3 Chain Operation**

701 Recall that the chain operation is defined in Eq. (40) as

$$\Psi(y,x):=\int\Psi(y,z)\Psi(z,x)d z=:\Psi(y,z)\otimes\Psi(z,x)$$

702 To see that it is associative, we need to check that Ψ(*y, z*) ⊗ (Ψ(z, x) ⊗ Ψ(*x, w*)) =
703 (Ψ(y, z) ⊗ Ψ(z, x)) ⊗ Ψ(*x, w*)

$$\Psi(y,z)\otimes(\Psi(z,x)\otimes\Psi(x,w))=\int\Psi(y,z)\left(\int\Psi(z,x)\Psi(x,w)dx\right)dz$$ $$=\int\int\Psi(y,z)\Psi(z,x)\Psi(x,w)dxdz$$ $$=\int\left(\int\Psi(y,z)\Psi(z,x)dz\right)\Psi(x,w)dx$$ $$=(\Psi(y,z)\otimes\Psi(z,x))\otimes\Psi(x,w)$$
$$(44)$$
(45)  $\binom{46}{45}$  (46)  . 
 $\left(47\right)$  (48)  ... 
704 **Proposition 10** (Parallel messages). Let ϕk+1|k and ϕk *be the potential functions for the CRF in* 705 Definition 1 and α and β *be the messages defined in Eqs.* (26) and (33)*. Then*

$$\alpha_{k}(x_{k})=\int\Psi_{1:k}^{b e d}(x_{k},x_{1})d x_{1}\quad a n d\quad\beta_{k}(x_{k})=\int\Psi_{k:N}^{b e d}(x_{N}|x_{k})d x_{N}$$

706 *where*

$$\Psi_{1:k}^{\text{fwd}}(x_{k},x_{1})=\bigotimes_{i=1}^{k-1}\phi_{i+1|i}(x_{i+1}|x_{i})\phi_{i}(x_{i})$$  _and $\Psi_{k:N}^{\text{bwd}}(x_{N}|x_{k})=\bigotimes_{i=N-1}^{k}\phi_{i+1|i}(x_{i+1}|x_{i})\phi_{i+1}(x_{i+1})$_
$$(49)$$
$$\begin{array}{c}{{({\bf50})}}\\ {{}}\end{array}$$  $$\begin{array}{c}{{({\bf51})}}\\ {{}}\end{array}$$

707 *Proof.* First for notational clarity, define

Ψ
bwd
i+1,i(xi+1|xi) = ϕi+1|i(xi+1|xi)ϕi+1(xi+1) and Ψ
$$\Psi_{i+1,i}^{\rm fwd}(x_{i+1},x_{i})=\phi_{i+1|i}(x_{i+1}|x_{i})\phi_{i}(x_{i})\tag{52}$$
708 We can compute the cumulative potentials as follows:

$$\Psi^{\rm bad}_{k:N}(x_{N}|x_{k})=\bigotimes_{i=N-1}^{k}\Psi^{\rm bad}_{i+1,i}(x_{i+1}|x_{i})$$ $$=\Psi^{\rm bad}_{N:N-1}(x_{N}|x_{N-1})\otimes\Psi^{\rm bad}_{N-1:N-2}(x_{N-1}|x_{N-2})\otimes\cdots\otimes\Psi^{\rm bad}_{k+1:k}(x_{k+1}|x_{k})$$ $$=\int\Psi^{\rm bad}_{N:N-1}(x_{N}|x_{N-1})\int\Psi^{\rm bad}_{N-1:N-2}(x_{N-1}|x_{N-2})dx_{N-1}\int\Psi^{\rm quad}_{N-2:N-3}(x_{N})$$ $$=\int\cdots\int\prod_{i=k}^{N-1}\Psi^{\rm bad}_{i:i+1}(x_{i+1}|x_{i})dx_{N-1}\cdots dx_{k+1}$$
$$(54)$$
$$\Psi_{N-2}^{\rm{bwd}}\tag{55}$$

709 And similarly for the forward potentials:

$$\Psi^{\rm fwd}_{1:k}(x_{k},x_{1})=\bigotimes_{i=1}^{k-1}\Psi^{\rm fwd}_{i+1,i}(x_{i+1},x_{i})$$ $$=\int\cdots\int\prod_{i=1}^{k-1}\Psi^{\rm fwd}_{i+1,i}(x_{i+1},x_{i})dx_{2}\cdots dx_{k-1}$$
$$(\mathbb{S}3)$$

710 Next, we can rewrite the joint distribution of the CRF in a similar form:

$$p(x_{1:N})=\prod_{k=1}^{N-1}\phi_{k+1|k}(x_{k+1}|x_{k})\prod_{k=1}^{N}\phi_{k}(x_{k})$$ $$=\phi_{k}(x_{k})\prod_{i=k}^{N-1}\Psi_{i+1,i}^{\text{bnd}}(x_{i+1}|x_{i})\prod_{i=1}^{k-1}\Psi_{i+1,i}^{\text{bnd}}(x_{i+1},x_{i}),\quad\forall k\in\{1,\ldots,N\}$$
$$(57)$$
$$(58)$$
(59)  $\binom{60}{5}$  . 
Then, integrating over the variables dx1*, . . . ,*
ˆdxk*, . . . , dx*N , where ˆ 711 dxk denotes that we are not 712 integrating over xk, completes the proof:

(64) 713 We can recognize the terms in the last equation as the forward and backward messages, which p(xk) = Z· · · Zp(x1:N )dx1 . . .ˆdxk . . . dxN (61) ∝ Z· · · Z NY−1 k=1 ϕk+1|k(xk+1|xk)Y N k=1 ϕk(xk)dx1 . . . ˆdxk . . . dxN (62) = ϕk(xk) Z· · · Z NY−1 i=k Ψ bwd i+1,i(xi+1|xi)Y k i=1 Ψ fwd i+1,i(xi+1, xi)dx1 . . .ˆdxk . . . dxN (63) = ϕk(xk) ZΨ bwd k:N (xN |xk)dxN ZΨ fwd 1:k(xk, x1)dx1 | {z } βk(xk) | {z } αk(xk)
$$(61)$$
(62)  $$\begin{array}{l}\mathbf{(63)^{}}\end{array}$$ = (64)  $$\begin{array}{l}\mathbf{(64)^{}}\end{array}$$ . 
714 completes the proof.

715 It will be convenient later to define an operator that actually transforms the parameters of the backward 716 messages.

717 **Definition 7** (Message passing update operator). Let ϕk+1|k(xk+1, xk) *be a Gaussian transition* 718 function and let ϕ(xk+1|ηk+1) be a Gaussian node potential with natural parameters ηk+1*. Next* 719 *consider the message passing update:*

$$\phi(x_{k}|\eta_{k})=\int\phi_{k+1|k}(x_{k+1}|x_{k})\phi(x_{k+1}|\eta_{k+1})d x_{k+1}$$

720 The message passing update operator is denoted by Φk,k+1(ηk+1) *and is defined to satisfy:*

$$(65)$$
_ed by $\Phi_{k,k+1}(\eta_{k+1})$ and is defined._
$$(67)^{\frac{1}{2}}$$

ηk = Φk,k+1(ηk+1) (66)

$$\beta_{k}=\Phi_{k,k+1}(\beta_{k+1}+\theta_{k+1})$$
βk = Φk,k+1(βk+1 + θk+1) (67)
$$\mid p a r t c u l a r;$$

721 *In particular, the update rule for the backward messages is given by:*
722 **Corollary 2** (Mixed parameterization update rule). Let ϕk+1|k(xk+1|xk) := N(xk+1|Axk+u, Σ) be a Gaussian transition function and let ϕ(xk+1|ηk+1) := N(xk+1|µk+1, J−1 k+1 723 ) *be a Gaussian node* 724 potential where Jk+1 is the precision matrix. If ηk and ηk+1 *represent the mean and precision matrix* 725 *of a Gaussian distribution, then the update and marginalize operator is denoted by* Φk,k+1(ηk+1)
726 *and is given by:*

$$\Phi_{k,k+1}\left(\mu_{k+1},J_{k+1}\right)=\left(A^{-1}(\mu_{k+1}-u),\Phi_{k,k+1}^{(J)}(J_{k+1})\right)$$

where Φ
(J)
k,k+1 727 (Jk+1) *is a nonlinear function of* Jk+1.

728 *Proof.* The result follows from Appendix H.3.

## 729 **D.4 Probabilistic Queries**

730 The forward and backward messages can be used to compute the majority of the probabilistic queries 731 of interest on a CRF. Recall our definition of a CRF: 732 Next we will describe two probabilistic queries of interest: the marginal distribution and the transition 733 distribution.

$$\mathrm{{\bf~tion~11~}}(\mathrm{{\bfM}}\mathrm{{\bfa}}$$
Proposition 11 (Marginal distribution).
p(xk|θ) = ϕ(xk|θk + αk + βk) (70)
734 *Proof.* The derivation is given in Eq. (61). For completness, we will change notation:

the derivation is given in Eq. (10). For completeness, we will change notation:  $$p(x_{k})=\phi_{k}(x_{k})\beta_{k}(x_{k})\alpha_{k}(x_{k})\text{(notation in previous section)}$$ $$:=\phi(x_{k}|\theta_{k})\phi(x_{k}|\alpha_{k})\phi(x_{k}|\beta_{k})\text{(notation in this section and in main text)}$$ $$=\phi(x_{k}|\theta_{k}+\alpha_{k}+\beta_{k})$$
735
Proposition 12 (Transition distribution).
$$\begin{array}{l}{{\mathrm{In}(\mathrm{In},\mathrm{~distribution}),}}\\ {{p(x_{k+1}|x_{k},\theta)\propto\phi_{k+1|k}(x_{k+1}|x_{k})\phi(x_{k+1}|\theta_{k+1}+\beta_{k+1})}}\end{array}$$
736 *Proof.* We can start by computing the joint distribution p(xk+1, xk|θ). By using variable elimination, 737 we can show that p(xk+1, xk|θ) = ϕ(xk|αk)ϕk+1|k(xk+1|xk)ϕ(xk+1|θk+1)ϕ(xk+1|βk+1) (75)
738 Dividing by the marginal distribution p(xk|θ) and using the definition of the transition distribution, 739 we get

$$p(x_{k+1}|x_{k},\theta)=\phi_{k+1|k}(x_{k+1}|x_{k}){\frac{\phi(x_{k+1}|\beta_{k+1}+\theta_{k+1})}{\phi(x_{k}|\beta_{k}+\theta_{k})}}$$
$$(76)$$

740 which, after absorbing the denominator into the normalization constant, is equivalent to the desired 741 result.

 ${p(x_{1:N}|\theta)\propto\prod_{k=1}^{N-1}\phi_{k+1|k}(x_{k+1}|x_k)\prod_{k=1}^N\phi(x_k|\theta_k)}$  In probabilistic versions of interest the marginal distribution and the true. 

$$(68)$$
$\square$
$$(69)$$

$$(70)$$
$$(71)$$
$$(74)$$