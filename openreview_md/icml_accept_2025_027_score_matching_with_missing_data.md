# 

Josh Givens 1 Song Liu 1 **Henry W J Reeve** 2

## Abstract

Score matching is a vital tool for learning the distribution of data with applications across many areas including diffusion processes, energy based modelling, and graphical model estimation. Despite all these applications, little work explores its use when data is incomplete. We address this by adapting score matching (and its major extensions) to work with missing data in a flexible setting where data can be partially missing over any subset of the coordinates. We provide two separate score matching variations for general use, an importance weighting (IW) approach, and a variational approach. We provide finite sample bounds for our IW approach in finite domain settings and show it to have especially strong performance in small sample lower dimensional cases. Complementing this, we show our variational approach to be strongest in more complex highdimensional settings which we demonstrate on graphical model estimation tasks on both real and simulated data.

## 1. Introduction

Over the last decade, score matching has established itself as a powerful tool with downstream use in many areas of machine learning. Examples include: energy based modelling (Swersky et al., 2011; Bao et al., 2020; Li et al., 2019b), mode-seeking clustering (Sasaki et al., 2014), and perhaps most prominently of all Diffusion processes (Song & Ermon, 2019; Song et al., 2021b; Tashiro et al., 2021; Song et al., 2021a; Huang et al., 2021). Score matching aims to learn the score of a distribution which is the gradient of the log of the probability density function (PDF) (s(x) = ∇x log p(x)). In contrast to modelling the density directly, the score does not need to integrate to one meaning there is no need to calculate a normalising constant. This allows it to be much more more flexibly modelled than the 1School of Mathematics, University of Bristol, Bristol, UK
2School of Artificial Intelligence, Nanjing University, China. Correspondence to: Josh Givens <josh.givens@bristol.ac.uk>.

density itself. Furthermore, the validity of the score matching objective itself requires only very mild assumptions of the family of proposed scores further ensuring this flexibility. Alongside the classical method (Hyvärinen, 2005), various adaptations of score matching have arisen to improve performance, decrease computational cost, and extend the approach to a wider range of settings (Hyvärinen, 2007; Vincent, 2011; Song et al., 2020; Liu et al., 2022).

In this work, we extend the score matching framework to handle missing data at training time. Specifically, we learn the full score function from partially missing multidimensional input data, a paradigm we term missing score matching. Crucially, our approach is compatible with any parameterised score model, enabling its application to both explicit score formulations and more general approaches such as neural networks (NNs). We propose two methods to adapt the original score matching method as well as its popular adaptations, truncated, sliced, and denoising score matching (Hyvärinen, 2007; Vincent, 2011; Liu & Wang, 2017; Song et al., 2020). These two distinct but closely related methods complement each other allowing for a wide range of problems to be tackled. The first method is a simpler importance weighting (IW) approach which we refer to as marginal IW score matching. For this method we obtain finite sample bounds in the bounded domain setting under certain conditions. We also provide experimental results demonstrating its efficacy in lower dimensional settings and where less data is available. Our second approach is a more computationally sophisticated variational approach which we refer to as marginal variational score matching. We demonstrate the efficacy of this approach in more complex, high dimensional settings by applying it to the problem of graphical model estimation with both real and synthetic datasets. In section 2 we discuss relevant works for score matching and related fields. In section 3 we will introduce our problem more formally including score matching and any notation used. Section 4 will be used to introduce our methods. Section 5 will present results on some real and simulated datasets. In Section 6 we give our conclusion.

## 2. Related Works

While there has been some work which utilises score matching with missing data, these approaches mostly do so 1 exclusively through the lens of diffusion models. Specifically works such as MissDiff (Ouyang et al., 2023) and Ambient Diffusion (Daras et al., 2023) require the score function itself to take the form of a neural network (NN) which learns the scores of the fully-observed and corrupted scores simultaneously. This prohibits their use in situations where our model for the score is some explicit parameterisation whose parameters we want to learn as is the case in settings such as energy based modelling Li et al. (2023); Bao et al. (2020); Salimans & Ho (2021) and Gaussian graphical models (Lin et al., 2016; Yu et al., 2018). Ambient Diffusion also requires the data to be further artificially corrupted in order to create a pseudo-supervised learning paradigm making both Ambient Diffusion and MissDiff subject to various levels of out of sample learning without specific adjustments for this phenomenon. Looking more generally at distribution estimation with missing data, multiple works in the field of generative modelling have looked to tackle the problem of providing a generative model for a distribution given corrupted samples from it. Prominent among these are MisGAN (Li et al., 2019a), which presents a marginalised GAN framework and MCFlow (Richardson et al., 2020) , which presents a EM like normalising flow framework. Neither of these approaches allow for flexible specification of a parametric density estimate however with MCFlow requiring the density to be a normalising flow and MisGAN having no model for the density whatsoever. To our knowledge, the only approach which seems to adapt score matching to missing data in a parameter preserving manner is presented in (Uehara et al., 2020) using an iterative EM-like procedure. However they themselves admit that there is little intuitive understanding of when this approach will converge. Additionally, due to the nature of the score matching objective, the expectation step cannot be directly approximated using Monte Carlo estimation and instead requires fractional importance weighting, a method which employs nested Monte Carlo estimates introducing bias into the training objective. Parallel to this, some papers have looked to extend score matching to the latent variable setting, an area with much commonality to missing data (Vértes & Sahani, 2016; Bao et al., 2020; 2021). Latent variable modelling differs in two crucial aspects from missing score matching. Firstly the components which are unobserved (the latent variables) remain constant between samples, and secondly there is not necessarily a notion of a ground truth for the unobserved components in when data is corrupted. Additionally each of these works has limitations; Vértes & Sahani (2016) only applies to exponential families, Bao et al. (2020) requires a gradient unrolling step in its optimisation which is computationally expensive and can lead to errors in the optimisation procedure (as acknowledged in their follow on work), and Bao et al. (2021) is only given for denoising score matching, not for classical or sliced score matching.

## 3. Setting

3.1. Notation For n ∈ N let [n] := {1*, . . . , n*}. For a random variable Z
we use supp(Z) for the support of Z. For f : R
d → R
we write ∂jf(x) :=∂f
∂xj where x = (x1*, . . . , x*d)
⊤and
∇xf(x) := (∂jf(x)*, . . . , ∂*df(x))⊤, the gradient of f. For f : R
d → R
dtake f(x)j as the j th component of f(x) and write ∇x · f(x) := ∂1f(x)1 + *· · ·* + ∂df(x)d. Finally for a, b ∈ R
d, take a ◦ b to be the Hadamard product.

We now introduce some indexing notation which we will be using for RVs and functions throughout. This will prove useful when identifying the missing non-missing components of our data. Let Z be a random variable taking values in R
d. We use Zj to refer to the j th component Z and for λ ⊆ [d] take Zλ = {Zj}j∈λ. We use negation in indexing to mean the complementing coordinates. More precisely we let −j denote [d] \ {j} and let −λ denote [d] \ λ. We typically use Z
(i)to denote an independent copy of Z. For a function f : *X → Y* and xλ ∈ Xλ, x
′
−λ ∈ X−λ, we take f(xλ, x
′
−λ) to be f(z) where

$$z_{j}:={\begin{cases}x_{j}&{\mathrm{if~}}j\in\lambda\\ x_{j}^{\prime}&{\mathrm{if~}}j\in-\lambda\end{cases}}$$
.
We will take X to be a RV taking values in X ⊆ R
drepresenting our original dataset and X′to be a RV representing some generative/variational/importance weighting distribution. i.e., the "artificial distributions" we will utilise in our method. Similarly, we take E, E
′to be expectations with respect to (w.r.t.) *X, X*′respectively.

Throughout we take p to be the pdf of the RV, X, and pθ to be a model therein. We let q represent an unnormalised density (i.e. N −1· q = p for some normalising constant N > 0.) We will write marginalisations/conditionings for both true and model densities implicitly with p(xλ) := 
RX
p(x)dx−λ and p(xλ|x−λ) being the conditional density of Xλ|X−λ = x−λ for example.

Now that we have introduced our notation we can move onto the key area of focus for our work, score matching.

## 3.2. Score Matching

First proposed by (Hyvärinen, 2005), score matching aims to learn the gradient of the log-density (score). The advantage of this framework over full density approaches such as maximum likelihood estimation (MLE) is that we are not restricted to parametric models which integrate to 1.

This allows us to be much more flexible in how we parameterise in turn making high dimensional distribution modelling more feasible. We now introduce the approach.

Let X be a RV over R
d with PDF p. We say that q is the unnormalised density of X if N −1· q(x) = p(x) where p is the PDF of X and N is the normalising constant of q. Define the score, of X to be

$$\mathbf{s}(\mathbf{x}):=\nabla_{\mathbf{x}}\log p(\mathbf{x})=\nabla_{\mathbf{x}}\log q(x).$$

The aim of score matching is to learn s from a collection of IID copies of X which we denote D := {X(i)}
n i=1. Following Hyvärinen (2005), we introduce a generic parameterised proposal score sθ for θ ∈ Θ ⊆ R
pand aim to minimise the *Fisher Divergence* between the true distribution and our proposal distribution which is given by

$$F(\theta):=\mathbb{E}[\|\mathbf{s}(X)-\mathbf{s}_{\theta}(X)\|^{2}].$$

The key result from Hyvärinen (2005) which enables us to practically implement score matching is that under certain (fairly minimal) regularity conditions, which we provide in Appendix D.1, we have

  **Remark 2.1**, we have  $$L(\theta):=\mathbb{E}\left[2\nabla_{X}\cdot\boldsymbol{s}_{\theta}(X)+\|\boldsymbol{s}_{\theta}(X)\|^{2}\right]=F(\theta)-C\tag{1}$$
where here and throughout, we take C to represent any constant which does not depend upon θ. Crucially, L(θ) is now an expectation of observable random variables. Hence we can now approximate this with our data and take ˆθ as

$${\hat{\theta}}:=\operatorname*{argmin}_{\theta}{\frac{1}{n}}\sum_{i=1}^{n}\left[2\nabla_{X^{(i)}}\cdot\mathbf{s}_{\theta}(X^{(i)})+\|\mathbf{s}_{\theta}(X^{(i)})\|^{2}\right].$$

TRUNCATED SCORE MATCHING
A limitation of standard score matching is that it requires limxi→∞ p(x) = 0 for all xi ∈ R. Thus it cannot be used for many distributions with compact support if the density does not converge to zero at the (topological) boundary. Initial work to adapt score matching to truncated distributions was presented in (Hyvärinen, 2007) for distributions on [0, ∞) then further expanded in (Liu et al., 2022; Yu et al., 2022) to general compact spaces X . For our compact space X ⊆ R
d we use ∂X to denote the (topological)
boundary. We now minimise some weighted version of the Fisher divergence whose weights go to zero at the boundary. Specifically let g : X → R be a function satisfying limx→x′ g(x)j = 0 for any x
′ ∈ ∂X , j ∈ [d]. Our objective is then

$$F_{\mathrm{T}}(\theta):=\mathbb{E}\left[\left\|\mathbf{g}^{\frac{1}{2}}(X)\circ(\mathbf{s}_{\theta}(X)-\mathbf{s}(X))\right\|^{2}\right].$$

Just as in classical score matching we obtain an equivalence
(though this time via Green's theorem rather than simple integration by parts) giving us that under certain regularity conditions on g, s, and X ,

$$L_{\mathrm{T}}(\theta):=\mathbb{E}\left[\sum_{j\in d}\mathbf{g}(X)_{j}\left(2\partial_{j}\mathbf{s}_{\theta}(X)_{j}+\mathbf{s}_{\theta}(X)_{j}^{2}\right)\right]$$ $$+\ \mathbb{E}\left[\sum_{j\in d}\partial_{j}\mathbf{g}(X)_{j}\mathbf{s}_{\theta}(X)_{j}\right]=F_{\mathrm{T}}(\theta)-C.$$

This can again be approximated via data using standard Monte Carlo approximation. Full details on the conditions required for this approach alongside the proof can be found in (Liu et al., 2022). Two other key extensions of score matching are sliced score matching (Song et al., 2020) and denoising score matching (Vincent, 2011). We introduce these extensions in Appendix D with our corresponding adaptations to missing data given in Appendix A.1. Now, we give our missing data scenario.

## 3.3. Missing Data Scenario

Instead of observing samples from X we assume that we observe samples from the corrupted version of the RV
given by X˜. To define X˜ we introduce a mask RV M over
{0, 1}
dand then define X˜ by

$${\bar{X}}_{j}=\begin{cases}X_{j}&\text{if}M_{j}=1\\ \varnothing&\text{if}M_{j}=0\end{cases}$$

where X˜j = ∅ represents that coordinate being missing. We will be focussing on the missing completely at random scenario where M ⊥ X. However, we do provide an extension to missing not at random data in Appendix A.1.4. We introduce the RV Λ on P([d]) defined by Λ := {i ∈ [d]|Mi = 1} so that Λ gives the non-corrupted coordinates of X˜ and take λ to be a sample of Λ. Crucially given samples from X˜, we also have samples from XΛ.

Our aim is to adapt the score matching objective to estimate the full score s by a parameterised score sθ using samples from the corrupted data D˜ := {X˜(i)}
n i=1 ≡ {X
(i)
Λi
}
n i=1.

## 4. Marginal Score Matching

To motivate our approach we look at how we might use MLE in the case where the normalising constant and conditional normalising constants were calculable. For pθ our parametric model of the density, we would choose ˆθ to be

$$\hat{\theta}:=\operatorname*{argmax}_{\theta}\sum_{i=1}^{n}\log\tilde{p}_{\theta}(\tilde{X}^{(i)})$$

where p˜θ is the associated corrupted data density when X ∼ pθ. As our data is missing completely at random this is actually equivalent to maximising

$$\sum_{i=1}^{n}\log p_{\theta;\Lambda_{i}}(X_{\Lambda_{i}}^{(i)}),\mathrm{where}\ p_{\theta;\lambda}(\mathbf{x}_{\lambda}):=\int_{\mathcal{X}_{-\lambda}}p_{\theta}(\mathbf{x}_{\lambda})$$
pθ(x)dx−λ.
For notational simplicity we will thus reframe our problem as working with marginal samples {X
(i) Λi
}
n i=1.

## 4.1. Marginal Score Matching

Our approach is to directly alter the score matching objective similarly. Just as densities have associated marginal densities so do scores have associated *marginal scores*. Definition 4.1 (Marginal Score function). Let s be a score function with s(x) = ∇x log q(x) for q an unnormalised PDF. Then the associated *marginal score* function is

$${\mathbf{s}}_{\lambda}({\mathbf{x}}_{\lambda}):=\nabla_{{\mathbf{x}}_{\lambda}}\log\int_{\mathbb{R}^{d-|\Lambda|}}q({\mathbf{x}})\mathrm{d}{\mathbf{x}}_{-\lambda}.$$

This definition of marginal scores restricts s to a genuine score function. For this reason we will also want sθ to always be a genuine score function or at least to have an anti-derivative. The simplest way to achieve this is to work with qθ : X → (0, ∞) as our baseline and define sθ(x) := ∇x log qθ(x). We will also take pθ(x) :=
RX
qθ(x)dx−1qθ(x) which we assume to be unknown.

With this notion of a marginal score we can define our marginal Fisher divergence to be

$$F_{\mathrm{M}}(\theta):=\mathbb{E}[\|s_{\Lambda}(X_{\Lambda})-s_{\Lambda;\theta}(X_{\Lambda})\|^{2}]$$
2] (3)
where sλ;θ is defined analogously to sλ. As with normal score matching can relate this objective to one involving no terms of sλ. We first need the following assumptions.

Assumption 4.2. For any θ > 0, λ ∈ supp(Λ):
(a) pθ is well defined, i.e. RX
qθ(x)dx < ∞;
(b) E[∥sλ(Xλ)∥
2], E[∥sλ;θ(Xλ)∥
2] < ∞;
(c) pλ(x) is differentiable and qλ;θ is twice differentiable;
(d) pλ(xλ)sλ;θ(xλ)−→0 as ∥x*∥−→∞*;
(e) pλ;θ(Xλ) = pλ(Xλ) almost surely (a.s.) for all λ ∈
supp(Λ), implies that pθ(X) = p(X) a.s..

Assumption (a) ensures that our proposal unnormalised density is always a genuine unnormalised density. Assumptions (b)-(d) are similar to the standard assumptions given for standard score matching. Assumption (e) is an identifiability assumption which is required to be feasibly able to learn the true data distribution from our corrupted data. Proposition 4.3. *Given Assumptions 4.2(a)-(d) hold*

$$L_{\rm M}(\theta):=\mathbb{E}[2\nabla_{X_{\Lambda}}\cdot s_{\Lambda;\theta}(X_{\Lambda})+\|s_{\Lambda;\theta}(X_{\Lambda})\|^{2}]\tag{4}$$ $$=F_{\rm M}(\theta)-C.$$

If (e) also holds and there exists some θ
∗*such that* sθ∗ (X) = s(X) a.s.. Then if ˜θ *is a minimiser of* LM(θ) we have that qθ˜(X) = N p(X) a.s. for some constant N,
i.e. the minimiser is the true unnormalised density. Through this result we have shown, much like with standard score matching, that under certain regularity conditions our objective is uniquely minimised by the true unnormalised density. We then approximate this objective by

$$\hat{L}_{\mathrm{M};n}(\theta):=\frac{1}{n}\sum_{i=1}^{n}\nabla_{X_{\Lambda_{i}}^{(i)}}\cdot\mathbf{s}_{\Lambda_{i},\theta}(X_{\Lambda_{i}}^{(i)})+\|\mathbf{s}_{\Lambda_{i};\theta}(X_{\Lambda_{i}}^{(i)})\|^{2}$$
$$\left(2\right)$$

and choose ˆθ = argminθ LˆM;n(θ).

Unfortunately this approach in its current state is practically infeasible as the integrals involved in deriving the marginal scores for any non-trivial problem will be intractable. Hence, we must devise a way to estimate the marginal scores without having to compute the integrals. We tackle this issue in Section 4.2, but first we provide a similar result for the case of truncated score matching.

## 4.1.1. Truncated Score Matching

Truncated score matching can be adapted similarly to standard score matching by simply having marginal weighting functions gλ : Xλ → [0, ∞) for each subset λ ∈ supp(Λ)
and taking the marginal truncated Fisher divergence to be

$$F_{\mathrm{TM}}(\theta){:=}\operatorname{\mathbb{E}}\left[\left\|g_{\Lambda}(X_{\Lambda})^{\frac{1}{2}}\circ(s_{\Lambda}(X_{\Lambda})-s_{\Lambda;\theta}(X_{\Lambda}))\right\|^{2}\right].$$
$$({\mathfrak{I}})$$

using integration by parts gives the following equivalence

$$L_{\rm TM}(\theta):=\mathbb{E}\!\left[\sum_{j\in\Lambda}\mathbf{g}_{\Lambda}(X_{\Lambda})_{j}\big{(}\mathbf{s}_{\Lambda;\theta}(X_{\Lambda})_{j}^{2}+2\partial_{j}\mathbf{s}_{\Lambda;\theta}(X_{\Lambda})_{j}\big{)}\right]$$ $$+\ \mathbb{E}\!\left[\sum_{j\in\Lambda}2\partial_{j}\mathbf{g}_{\Lambda}(X_{\Lambda})_{j}\mathbf{s}_{\Lambda;\theta}(X_{\Lambda})_{j}\right]\tag{5}$$ $$=\!F_{\rm TM}(\theta)-C.$$

Proof given in Appendix C.1.1. We then take LˆTM;n as the Monte-Carlo estimate of LTM. We also construct similar objectives from sliced and denoising score matching as well as a similar result for missing not at random data in Appendix A.1. We now move to the task of estimating the marginal scores in these objectives.

## 4.2. Importance Weighting

Our first proposal is an importance weighting approach. Let p
′ be a density over Rd−|λ| which we can both evaluate and sample from then

$$\int_{\mathbb{R}^{d-|\lambda|}}q_{\theta}(\mathbf{x})\mathrm{d}\mathbf{x}_{-\lambda}=\mathbb{E}_{X^{\prime}_{\lambda}\sim p^{\prime}}\left[\frac{q_{\theta}(\mathbf{x}_{\lambda},X^{\prime}_{-\lambda})}{p^{\prime}(X^{\prime}_{-\lambda})}\right].\tag{6}$$

4 Algorithm 1 Marginal IW Score Matching Input: {X
(i)
Λi
}i∈[n], qθ, p
′, θ0, r ∈ N.

Set θ = θ0.

repeat for i=1 to n do Sample {X′(i,k)}k∈r from p
′(.|X
(i)
Λi
).

Use X
(i)
Λi
, {X
′(i,k)
−Λi
}k∈[r]to get Monte-Carlo estimates, sˆΛi,r;θ(X
(i) Λi
), of the marginal scores by (7).

end for Use sˆΛi,r;θ(X
(i)
Λi
) to obtain LˆM/TM;n,r(θ) by (4).

Compute ∇θLˆM/TM;n,r(θ) and update the value of θ.

until Maximum iteration reached.

This allows us to define our *marginal score estimate*. Definition 4.4 (Marginal Score Estimate). For a given λ ∈
supp(Λ) , xλ ∈ Xλ, score model, sθ, and r ∈ N we take our estimate of sθ;λ,r(xλ) to be

$$\hat{\mathbf{s}}_{\lambda,r;\theta}:=\nabla_{\mathbf{x}_{\lambda}}\log\left({\frac{1}{r}}\sum_{k=1}^{r}{\frac{q_{\theta}(\mathbf{x}_{\lambda},X_{-\lambda}^{\prime(k)})}{p^{\prime}(X_{-\lambda}^{\prime(k)})}}\right)$$
$$\mathbf{\Phi}(T)$$

where X
$X^{\prime}(1)$  $-\lambda$  $X^{\prime}(r)$  are IID copies of $X^{\prime}_{-\lambda}\sim p^{\prime}$
′.

4.2.1. IW SAMPLE OBJECTIVE
We can now plug these marginal score estimates into our sample objective for either normal or truncated score matching. We use M/TM to denote analogous definitions and results for both marginal and truncated marginal score matching. Let {X
(i)
Λi
}
n i=1 be our samples from XΛ. We then take our IW sample objective to be as LˆM/TM;n(θ)
but with sˆΛi,r;θ(X
(i)
Λi
) replacing sΛi;θ(X
(i)
Λi
). The full objective is given in Appendix E.1.1 We refer to this sample objective as LˆM/TM;n,r(θ) and take our estimate to be

$${\hat{\theta}}:=\operatorname*{argmin}_{\theta}{\hat{L}}_{\mathrm{M/TM;}n,r}(\theta).$$

Algorithm 1 gives our high level estimation algorithm. Remark 4.5. Algorithm 1 can directly be applied to both sliced and denoised score matching by replacing equation (4) by equations (13) and (15) respectively.

## 4.2.2. Finite Sample Bounds

A benefit of truncated score matching is that it allows us to work on distributions with densities bounded below which enables us to give finite sample bounds for the error of our estimated score w.r.t. our marginal objective. We briefly present these now with more detail given in Appendix A.2. Theorem 4.6. Suppose assumption 4.2 alongside assumptions A.1, A.11, A.13 from the Appendix hold and let θn,r ∈ Θ *be the minimiser of* LˆTM;n,r(θ). If Θ ⊆ R
p with diam(Θ) = A then for sufficiently large *n, r*

$$\mathbb{P}\left(F_{\mathrm{TM}}(\theta_{n,r})\geq\beta_{1}{\sqrt{\frac{p\log(d n r A/\delta)}{\operatorname*{min}\{r,n\}}}}\right)<\delta.$$

Note that r is the number of importance weighting samples for each data sample and therefore is something we can choose ourself. This means that with this approach we can achieve approximately 
√n convergence rates. A downside however is that to achieve this we need r to be of order at least n which would lead to an O(n 2) computational cost.

In practice we find relatively strong performance choosing r small. Setting it at r = 10 in our experiments. Remark 4.7. The error presented is measured with respect to our Marginal Fisher Divergence, rather than the full Fisher Divergence (which would be the preferred accuracy metric). Relating these two quantities requires connecting the fully observed distribution to its marginals, a task that depends on the specific form of the distribution. Investigating the assumptions and conditions under which this connection can be made offers an interesting and valuable direction for future research.

## 4.3. Gradient First Approach

A key limitation with an IW approach is that it will struggle in higher dimensional scenarios. Additionally the importance weighting is embedded inside other functions which leads to the same nested expectation issue as the EM approach of Uehara et al. (2020), causing bias in our estimator. As an alternative to this we build upon a variational approach initially discussed in the context of latent variable models in Vértes & Sahani (2016); Bao et al. (2020; 2021).

The core idea is to start with LM as before and then take gradients w.r.t. our parameters before then writing our objective in terms of expectations over X−λ|λ;θ. As we don't then need to take gradients of these expectations w.r.t. θ, we can estimate them with any black-box method we desire, opening the door for variational approximation to be used. This approach has been explored for exponential family distributions (Vértes & Sahani, 2016) and for denoising score matching (Bao et al., 2021) however we provide the most general version of this result which can be applied to any of the score matching methods and any model class. We first introduce the following key Lemma.

Lemma 4.8. Fix λ ⊆ [d], xλ ∈ Xλ. We have that for any function hθ : X → R.

$$\mathbf{s}_{\theta;\lambda}(\mathbf{x}_{\lambda})=\mathbb{E}^{\prime}[\mathbf{s}_{\theta}(\mathbf{x}_{\lambda},X^{\prime}_{-\lambda})_{\lambda}]\tag{8}$$ $$\nabla\mathbb{E}^{\prime}[h_{\theta}(\mathbf{x}_{\lambda},X^{\prime}_{-\lambda})]=\mathbb{E}^{\prime}[\nabla h_{\theta}(\mathbf{x}_{\lambda},X^{\prime}_{-\lambda})]$$ (9) $$+\text{Cov}^{\prime}(\mathbf{s}_{\theta}(\mathbf{x}_{\lambda},X^{\prime}_{-\lambda}),h_{\theta}(\mathbf{x}_{\lambda},X^{\prime}_{-\lambda}))$$

where ∇ represents the gradient w.r.t. either xλ or θ and here E
′, Cov′*are w.r.t.* X′−λ |Xλ = xλ ∼ pθ(.|xλ).

This results allows us to obtain our alternative objective.

Corollary 4.9. Let LM *be defined as in* (4)*. We have that*

$$\nabla_{\theta}L_{\rm M}(\theta)=\mathbb{E}\left[2\sum_{j\in\Lambda}\left(\Psi_{\Lambda}(\mathbf{s}_{\theta}(.)_{j}^{2}+\partial_{j}\mathbf{s}_{\theta}(.)_{j}\right)\right.\tag{10}$$  $$\left.-\mathbb{E}^{\prime}[\mathbf{s}_{\theta}(X_{\Lambda},X_{-\Lambda}^{\prime})_{j}]\Psi_{\Lambda}(\mathbf{s}_{\theta}(.)_{j})\right)\right]$$

where for any function hθ : R
d → R, λ ⊆ [d],

$$\begin{array}{c}{{\Psi_{\Lambda}(h_{\theta})=\mathbb{E}^{\prime}[\nabla_{\theta}h_{\theta}(X_{\Lambda},X_{-\Lambda}^{\prime})]}}\\ {{\qquad\qquad+\mathrm{Cov}^{\prime}\left(\nabla_{\theta}\log q_{\theta}(X_{\Lambda},X_{-\Lambda}^{\prime}),h_{\theta}(X_{\Lambda},X_{-\Lambda}^{\prime})\right)}}\end{array}$$

and E
′, Cov′*are w.r.t.* X′−Λ
|XΛ ∼ pθ(.|XΛ) with E being w.r.t. XΛ ∼ p.

Proofs for both results are given in Appendix C.3.

Crucially E
′, Cov′can be estimated freely. This allows us to use variational inference to approximate pθ(x−λ|xλ) and in turn the expectations and covariances in (10). Remark 4.10. We provide additional implementation details for computing this gradient estimate in Appendix A.5. We also discuss equivalences between this objective and our marginal IW objective in A.3.

We explore estimation of E
′, Cov′in Section 4.3.2 but first we provide a similar result for truncated score matching.

4.3.1. TRUNCATED SCORE MATCHING
We define a similar objective for truncated score matching.

Corollary 4.11. With LTM *defined as in* (5) *we have that*

$$\nabla_{\theta}L_{\rm TM}(\theta)=\mathbb{E}\left[2\sum_{j\in\Lambda}\left(\mathbf{g}_{\Lambda}(X_{\Lambda})_{j}\bigg{\{}\Psi_{\Lambda}(\mathbf{s}_{\theta}(.)_{j}^{2}+\partial_{j}\mathbf{s}_{\theta}(.)_{j})\right.\right.$$ $$\left.\left.-\,\mathbb{E}^{\prime}[\mathbf{s}_{\theta}(X_{\Lambda},X_{-\Lambda}^{\prime})_{j}]\Psi_{\Lambda}(\mathbf{s}_{\theta}(.)_{j})\right\}\right.$$ $$\left.+\ \left.\partial_{j}\mathbf{g}_{\Lambda}(X_{\Lambda})_{j}\Psi_{\Lambda}(\mathbf{s}_{\theta}(.)_{j})\right)\right]\tag{11}$$

with ΨΛ and E
′ *defined as in Corollary 4.9.*
Proof given in Appendix C.1.1. Similar results for sliced and denoising score matching are given in Appendix A.1.

## 4.3.2. Variational Approximation

We can now use variational approximation to estimate the expectations and covariances in Corollaries 4.9 & 4.11.

Specifically, let p
′
ϕ
(x−λ|xλ) be some generative conditional distribution dependent upon parameter ϕ. We want to train p
′ϕ to approximate pθ. We may write ϕ(θ) to highlight the dependence on our current parameter estimate however we will omit this for brevities sake. The following proposition from Bao et al. (2020) shows us how to train ϕ. Proposition 4.12 (Bao et al. (2020)). *For distributions* p
′, p let F(p
′|p) and KL(p
′|p) be the Fisher and KL divergences between p
′ and p. We have that for any λ ⊆ [d], xλ ∈ Xλ

$$\begin{split}\text{KL}(p^{\prime}_{\phi}(.|\mathbf{x}_{\lambda})|p_{\theta}(.|\mathbf{x}_{\lambda}))&=\mathbb{E}^{\prime}\Bigg{[}\log\left(\frac{p^{\prime}_{\phi}(X^{\prime}_{-\lambda}|\mathbf{x}_{\lambda})}{q_{\theta}(\mathbf{x}_{\lambda},X^{\prime}_{-\lambda})}\right)\Bigg{]}+B\\ F(p^{\prime}_{\phi}(.|\mathbf{x}_{\lambda})|p_{\theta}(.|\mathbf{x}_{\lambda}))&=\mathbb{E}^{\prime}\Bigg{[}\left\|\nabla_{X^{\prime}_{-\lambda}}\log\left(p^{\prime}_{\phi}(X^{\prime}_{-\lambda}|\mathbf{x}_{\lambda})\right)\right.\\ &\left.-\left.\mathbf{s}_{\theta}(\mathbf{x}_{\lambda},X^{\prime}_{-\lambda})_{-\lambda}\right\|^{2}\right]\end{split}$$

where expectations are w.r.t. X′−Λ ∼ p
′ ϕ
(.|xλ) and B
is a constant not depending upon ϕ *(but will depend on* θ*.) In other words we can fit to the conditional density* pθ(.|xλ) given only the unconditional unnormalised density qθ(xλ, .) *or full score* sθ(xλ, .). This allows us to train p
′ ϕ
(.|xλ) to approximate the conditional density, pθ(.|xλ). In our case we won't be learning this variational model for a fixed xλ or even fixed observed coordinates λ. Hence we take our objective to be one of

$$J_{K L}(\phi,\theta)\coloneqq\mathbb{E}\left[\log\left(\frac{p_{\phi}^{\prime}(X_{-\Lambda}^{\prime}|X_{\Lambda})}{q_{\theta}(X_{\Lambda},X_{-\Lambda}^{\prime})}\right)\right]$$ $$J_{F}(\phi,\theta)\coloneqq\mathbb{E}\left[\left\|\nabla_{X_{-\Lambda}^{\prime}}\log\left(\frac{p_{\phi}^{\prime}(X_{-\Lambda}^{\prime}|X_{\Lambda})}{q_{\theta}(X_{\Lambda},X_{-\Lambda}^{\prime})}\right)\right\|^{2}\right]$$

with (XΛ, X′−Λ
) ∼ p
′ϕ
(X′−Λ
|XΛ)p(XΛ). We then take and JˆF , JˆF to be the Monte-Carlo approximations with samples (XΛ, X′−Λ) from the same distribution.

Remark 4.13. JF has the advantage of not needing to know the normalising constant of q
′ϕ = Nϕ · p
′ϕ either.

Remark 4.14. As ϕ depends upon θ, we need to update it each time we update θ. In practice we find taking 10 gradient steps of ϕ for each gradient step of θ to work well.

With this, we define ∇dθLM/TM(θ) to be the Monte-Carlo estimate of (10)/(11) with samples {(X
(i) Λi
, X′(i,k)
−Λi
)}(i,k)∈[n]×[r] where X
(i)
Λare our original corrupted data samples from p and X
′(i,k)
−Λare our variational samples from p
′ϕ
(.|X
(i)
Λi
). We can now state our full variational approach which is given in Algorithm 2. Remark 4.15. Algorithm 2 can directly be applied to both sliced and denoised score matching by replacing equation (10) by equations (14) and (16) respectively.

Algorithm 2 Marginal Variational Score Matching Input: {X
(i)
Λi
}i∈[n], qθ, p
′
ϕ, θ0, ϕ0, L ∈ N, r ∈ N.

Set θ = θ0, ϕ = ϕ0.

repeat for l = 1 to L do For i ∈ [n] sample X′(i)from p
′
ϕ(.|X
(i)
Λi
).

Use {(X
(i)
Λi
, X′(i)
−Λi
)}i∈[n]to get Monte-Carlo approximates of JKL/F (*ϕ, θ*) given by J ˆ *KL/F* (ϕ, θ).

Compute ∇ϕJ ˆ 
KL/F (*ϕ, θ*) and update ϕ.

end for For i ∈ [n] sample {X′(i,k)}k∈r from p
′ ϕ
(.|X
(i)
Λi
).

Use {(X
(i)
Λi
, X′(i,k)
−Λi
)}(i,k)∈[n]×[r]to get our Monte-
Carlo estimate, ∇dθLM/TM(θ) using equation
(10)/(11). Use this gradient estimate to update θ.

until Maximum iterations reached.

## 5. Results

Here we go through simulated results comparing our IW
approach (Marg-IW) in Algorithm 1 and our variational approach (Marg-Var) in Algorithm 2 to the EM approach of Uehara et al. (2020). We also compare to a naive marginalisation approach involving zeroing out the missing dimensions and only taking the observed output dimensions of the score, which we call Zeroed Score Matching. This approach is the natural adaptation of MissDiff from Ouyang et al. (2023) away from NN to explicitly parameterised models. We describe Zeroed Score Matching and its relation to MissDiff in Appendix D.2. In our experiments, we highlight a unique strength of our methods by applying them to explicitly parameterised score models. We could however, equally apply them to more complex, noninterpretable models such as NNs. More implementation details can be found in Appendix E.3. 1

## 5.1. Parameter Estimation 5.1.1. Truncated Gaussian Model

In this experiment a 10-dim normal distribution is set up with fixed mean and random covariance before being truncated on the first 3 dimensions. 1000 samples are taken and corrupted independently on each coordinate with probability 0.2. For each of our methods a Gaussian score is fit and the Fisher divergence between this score and the truth computed. This is repeated 200 times with the mean Fisher divergence alongside 95% C.I.s then presented in figure 1. More details in Appendix E.3.1. Marg-IW and EM perform best with Marg-Var approaching asymptotically. We see the effect of Zeroed's naive marginalisation as it does 1All code and data for the experiments presented can also be found at https://github.com/joshgivens/ ScoreMatchingwithMissingData

Fi s h e r D
iv e r g e n c e Marg-IW (Ours)
Marg-Var (Ours) Zeroed EM
200 400 600 800 1000 Sample Size 0 1 2 3
not converge, a phenomenon we discuss more in Appendix D.2. In Appendix B.1.1 we present the average mean and precision estimation error for this experiment. In Appendix B.1.2 we present the untruncated results and illustrate how the naive marginalisation poorly models strong relationship between dimensions 1 and 10.

5.1.2. NON-GAUSSIAN MODEL
For this experiment we tested our parameter estimation for a an ICA inspired unnormalisable model of the form

$$p(\mathbf{x})\propto\exp\sum_{i,j}\theta_{i,j}^{*}x_{i}^{2}x_{j}^{2}.$$

Here we parameterise our model identically with the aim of estimating θ
∗. We vary the dimension of X and plot the estimation error with a sample size of 1,000 and each coordinate missing independently with probability 0.5. The results are presented in Figure 2.

10 20 30 40 50 Dimension 0 1 2 3 4 T
h e t a
 
E
rr o r Marg-IW (Ours) Marg-Var (Ours) Zeroed EM
Our variational method (Marg-Var) consistently yields the lowest error. Moreover, as the dimensionality increases, the performance gap between Marg-Var and the other methods widens. This supports the notion that our approach is more accurately able to capture complex marginalisations than the competing approaches which fail as the dimension grows. We note that all other methods perform comparably with the performance of EM and Marg-IW being indistinguishable, a pattern we observe throughout our experiments. This similarity is unsurprising both approaches use self normalised importance weighting to approximate conditional expectations with respect to our current score estimate while being broadly motivated by fitting to the marginal scores. Nevertheless, the precise mechanism for this similarity remains unclear and warrants further exploration. Additional experiments exploring the effect of sample size and missingness probability on estimation accuracy are given in appendix B.1.3.

## 5.2. Gaussian Graphical Model Estimation

Gaussian graphical models (GGM) are a popular way of modelling dependence between dimensions of data. Let us assume that the underlying data follows a Gaussian distribution with mean µ ∈ R
dand precision P ∈ R
d×d. In this setting, a Bayesian network (BN) can represent the dependencies between the dimensions of X with the (undirected) edges of the BN exactly being the non-zero off-diagonal entries of the precision, P. Hence estimating the precision matrix P gives the BN. Score matching has been shown to be an effective way of achieving this with L1-regularisation on the off-diagonal of P to push terms to 0 (Lin et al., 2016; Yu et al., 2018). Decreasing the level of L1-regularisation then gives a range of classifiers with increasing True and False positive rates (TPR/FPR) as the level of regularisation decreases. Score matching can also be applied to truncated GGMs where we aim to learn the original BN but only observe the samples inside some truncated region. We apply our methods to learn GGMs and truncated GGMs with missing data as well. We use varying levels of L1 regularisation on our objective via proximal stochastic gradient descent in our optimisation (Beck, 2017).

5.2.1. STAR SHAPED TRUNCATED GRAPHICAL MODEL
Here we create a star shaped GGM in which one node has a high probability of being connected with each other node independently and all other connections have probability 0. We truncate the data along a random hyperplane such that 20% of the distribution lies outside of the truncation boundary. Each coordinate is then MCAR independently with the same probability. We run multiple experiments with this probability ranging from 0.2 to 0.9 and present the results in figure 3. As we can see here Marg-Var performs best with all other approaches performing comparably. For illustrative purposes, we plot individual ROC curves from this experiment in Appendix B.2.3.

0.6 0.7 0.8 0.9 1.0 A
U
C

Marg-IW (Ours) Marg-Var (Ours)
Zeroed EM
0.2 0.4 0.6 0.8 Missingness Probability
5.2.2. UNSTRUCTURED DENSE GRAPHICAL MODEL
Here we create a GGM by making each edge occur independently with probability 0.5. The rest of the experiment was constructed as before. Results are given in Figure 4. Again we can see that our variational approach performs

0.2 0.4 0.6 0.8 Missingness Probability 0.60 0.65 0.70 0.75 0.80 0.85 A
U
C

Marg-IW (Ours) Marg-Var (Ours)
Zeroed EM
best though not as clearly as in the previous example. We believe this to be because for more unstructured problems, naive marginalisation performs moderately well.

## 5.2.3. Increasing Number Of Stars

To explore this further, we construct and experiment where we vary the number of star centres (high degree nodes) while keeping the edge density constant. We present the results in Figure 5. As we increase the number of star centres, Marg-Var no longer noticeably outperforms the other approaches. This is because as the number of stars increases, (i.e. the structure of the graph decreases) naive marginalisation is a better approximation. This is illustrated on the marginal precisions themselves in Appendix B.2.1.

## 5.2.4. S&P 100

Here we took closing price data over 5 years for the 100 stocks in the S&P 100 with each stock being a dimension

Marg-IW (Ours) Marg-Var (Ours)
Zeroed EM
1 2 3 4 5 Number of Star Centres 0.70 0.75 0.80 0.85 A
U
C

and each day being a sample. Gaussian graphical models with various levels of connectivity were then constructed using standard score matching on the fully observed data. The data was then artificially corrupted and each missing score matching approach applied. The AUC was then calculated for each method taking the GGM from fully observed score matching as the ground truth. More details given in appendix E.3.3. The results are shown in figure 6.

0.2 0.4 0.6 0.8 Missing Probability 0.6 0.7 0.8 0.9 A
U
C

Marg-IW (Ours)
Marg-Var (Ours)
Zeroed EM
As we can see Marg-Var clearly out performs all the other approaches which appear to perform equivalently.

## 5.2.5. Yeast Data

Here data first introduced in Brem & Kruglyak (2005) is used consisting of readings of expression for 7086 genes/ORFs across 262 yeast segregants. Each gene represents a dimension with each segregant representing a sample. We subset the data to take the 106 genes present in at least 95% of the samples with the aim of learning the relationship between them. The same approach as the previous section is applied with the results shown in figure 7. Again Marg-Var clearly outperforms the other approaches which all perform comparably.

0.6 0.7 0.8 0.9 A
U
C

Marg-IW (Ours) Marg-Var (Ours) Zeroed EM
0.2 0.4 0.6 0.8 Missing Probability

## 6. Conclusion

To conclude, score matching is a versatile method whose applications at the heart of modern machine learning problems. In this work we have tackled the problem of adapting score matching to partially missing data. We have presented two separate but related approaches to this method, one using importance weighting and another using variational approximation. We have also provided extensions of these methods to truncated score matching, sliced and denoising score matching. For truncated score matching with our IW approach we have provided finite sample bounds on the accuracy of the estimated score in terms of the marginal truncated Fisher divergence. We have provided several simulated and real world experiments demonstrating our methods' efficacy for both parameter estimation and downstream GGM edge detection. We have shown the benefits and drawbacks of each approach with IW performing best in lower dimensional settings with less data and the variational approach performing best in more complicated higher dimensional settings. There is, however still much work to be done in this area. From a theoretical perspective, while we have finite sample bound on the error of our loss, marginal nature of the loss makes it unclear exactly how this translates to parameter or general score model accuracy, leaving room for further theoretical exploration. From an implementation perspective, variational inference in the presence of missing data requires accounting for the randomness of "latent" and "observed" variables. The standard variational inference technique can be further refined to accommodate this setting. Finally, since our method is compatible with denoised score matching, it can naturally be extended to diffusionbased model. This paves the way for future work on applying our approach to generative modelling with diffusion processes in the presence of missing data.

## References

Daras, G., Shah, K., Dagan, Y., Gollakota, A., Dimakis, A., and Klivans, A. Ambient diffusion: Learning clean distributions from corrupted data. In Thirty-seventh conference on neural information processing systems, 2023. URL https://openreview.net/forum? id=wBJBLy9kBY.

## Acknowledgements

Josh Givens was supported by a PhD studentship from the EPSRC Centre for Doctoral Training in Computational Statistics and Data Science (COMPASS).

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

Fattahi, S. and Sojoudi, S. Graphical lasso and thresholding: Equivalence and closed-form solutions. Journal of Machine Learning Research, 20(10):1–44, 2019. URL http://jmlr.org/papers/v20/ 17-501.html.

Huang, C.-W., Lim, J. H., and Courville, A. C. A variational perspective on diffusion-based generative models and score matching. Advances in Neural Information Processing Systems, 34:22863–22876, 2021.

Bao, F., LI, C., Xu, K., Su, H., Zhu, J., and Zhang, B. Bi-level score matching for learning energy-based latent variable models. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), Advances in neural information processing systems, volume 33, pp. 18110–18122. Curran Associates, Inc., 2020. URL https://proceedings.neurips. cc/paper_files/paper/2020/file/ d25a34b9c2a87db380ecd7f7115882ec-Paper. pdf.

Hyvärinen, A. Estimation of non-normalized statistical models by score matching. Journal of Machine Learning Research, 6(24):695–709, 2005.

Hyvärinen, A. Some extensions of score matching. Computational Statistics & Data Analysis, 51(5):2499–2512, 2007. ISSN 0167-9473.

doi: https://doi.org/10.1016/j.csda.2006.09.003. URL https://www.sciencedirect.com/ science/article/pii/S0167947306003264.

Bao, F., Xu, K., Li, C., Hong, L., Zhu, J., and Zhang, B. Variational (gradient) estimate of the score function in energy-based latent variable models. In Meila, M.

and Zhang, T. (eds.), Proceedings of the 38th international conference on machine learning, volume 139 of Proceedings of machine learning research, pp. 651–661. PMLR, July 2021. URL https://proceedings. mlr.press/v139/bao21b.html.

Li, S. C.-X., Jiang, B., and Marlin, B. MisGAN: Learning from Incomplete Data with Generative Adversarial Networks. In International Conference on Learning Representations, 2019a. URL https://openreview. net/forum?id=S1lDV3RcKm.

Beck, A. The proximal gradient method. In Firstorder methods in optimization, pp. 269–329. Society for Industrial and Applied Mathematics, 2017. doi: 10.1137/1.9781611974997.ch10. URL https://epubs.siam.org/doi/abs/10. 1137/1.9781611974997.ch10.

Li, Z., Chen, Y., and Sommer, F. T. Learning energy-based models in high-dimensional spaces with multi-scale denoising score matching, 2019b. arXiv: 1910.07762 [stat.ML].

Li, Z., Chen, Y., and Sommer, F. T. Learning energybased models in high-dimensional spaces with multiscale denoising-score matching. Entropy. An International and Interdisciplinary Journal of Entropy and Information Studies, 25(10), 2023. ISSN 1099-4300. doi:
10.3390/e25101367. URL https://www.mdpi. com/1099-4300/25/10/1367. Number: 1367 tex.pubmedid: 37895489.

Brem, R. B. and Kruglyak, L. The landscape of genetic complexity across 5,700 gene expression traits in yeast. Proceedings of the National Academy of Sciences of the United States of America, 102(5):1572–1577, February 2005. ISSN 0027-8424 1091-6490. doi: 10.1073/pnas. 0408709102. Place: United States.

Burda, Y., Grosse, R. B., and Salakhutdinov, R. Importance weighted autoencoders. In Bengio, Y. and LeCun, Y. (eds.), 4th international conference on learning representations, ICLR 2016, san juan, puerto rico, may 2-4, 2016, conference track proceedings, 2016. URL http: //arxiv.org/abs/1509.00519. tex.bibsource:
dblp computer science bibliography, https://dblp.org tex.timestamp: Thu, 25 Jul 2019 14:25:37 +0200.

Lin, L., Drton, M., and Shojaie, A. Estimation of highdimensional graphical models using regularized score matching. *Electronic Journal of Statistics*, 10(1):806 - 854, 2016. doi: 10.1214/16-EJS1126. URL https:// doi.org/10.1214/16-EJS1126. Publisher: Institute of Mathematical Statistics and Bernoulli Society.

Liu, Q. and Wang, D. Learning Deep Energy Models: Contrastive Divergence vs. Amortized MLE, July 2017. URL http://arxiv.org/abs/1707. 00797. arXiv:1707.00797 [cs, stat].

Liu, S., Kanamori, T., and Williams, D. J. Estimating density models with truncation boundaries using score matching. Journal of Machine Learning Research, 23(186):1–38, 2022. URL http://jmlr. org/papers/v23/21-0218.html.

Ouyang, Y., Xie, L., Li, C., and Cheng, G. MissDiff: Training diffusion models on tabular data with missing values. In ICML 2023 workshop on structured probabilistic inference & generative modeling, 2023. URL https: //openreview.net/forum?id=S435pkeAdT.

Richardson, T. W., Wu, W., Lin, L., Xu, B., and Bernal, E. A. MCFlow: Monte Carlo Flow Models for Data Imputation. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14193– 14202, June 2020. doi: 10.1109/CVPR42600.2020. 01421. ISSN: 2575-7075.

Salimans, T. and Ho, J. Should EBMs model the energy or the score? In Energy based models workshop - ICLR 2021, 2021. URL https://openreview.

net/forum?id=9AS-TF2jRNb.

Sasaki, H., Hyvärinen, A., and Sugiyama, M. Clustering via mode seeking by direct estimation of the gradient of a log-density. In Calders, T., Esposito, F., Hüllermeier, E., and Meo, R. (eds.), Machine learning and knowledge discovery in databases, pp. 19–34, Berlin, Heidelberg, 2014. Springer Berlin Heidelberg. ISBN 978-3662-44845-8.

Song, Y. and Ermon, S. Generative Modeling by Estimating Gradients of the Data Distribution. In Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

Song, Y., Garg, S., Shi, J., and Ermon, S. Sliced score matching: A scalable approach to density and score estimation. In Adams, R. P. and Gogate, V. (eds.), Proceedings of the 35th Uncertainty in Artificial Intelligence Conference, volume 115 of Proceedings of Machine Learning Research, pp. 574–584. PMLR, jul 2020.

Song, Y., Durkan, C., Murray, I., and Ermon, S. Maximum likelihood training of score-based diffusion models. Advances in Neural Information Processing Systems, 34:
1415–1428, 2021a.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A.,
Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021b.

Swersky, K., Ranzato, M., Buchman, D., Freitas, N. D.,
and Marlin, B. M. On autoencoders and score matching for energy based models. In Proceedings of the 28th international conference on machine learning (ICML-11), pp. 1201–1208, 2011.

Tashiro, Y., Song, J., Song, Y., and Ermon, S. CSDI: Conditional score-based diffusion models for probabilistic time series imputation. In Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W. (eds.), *Advances in Neural* Information Processing Systems, 2021.

Uehara, M., Matsuda, T., and Kim, J. K. Imputation estimators for unnormalized models with missing data. In Chiappa, S. and Calandra, R. (eds.), *Proceedings* of the twenty third international conference on artificial intelligence and statistics, volume 108 of Proceedings of machine learning research, pp. 831–841. PMLR,
August 2020. URL https://proceedings.mlr. press/v108/uehara20b.html.

Vincent, P. A connection between score matching and denoising autoencoders. *Neural Computation*, 23(7):1661– 1674, 2011. doi: 10.1162/NECO_a_00142.

Vértes, E. and Sahani, M. Learning doubly intractable latent variable models via score matching. *InSymposium* on advances in approximate Bayesian inference (AABI), 2016.

Yang, E. and Lozano, A. C. Robust gaussian graphical modeling with the trimmed graphical lasso. In Cortes, C., Lawrence, N., Lee, D., Sugiyama, M., and Garnett, R. (eds.), *Advances in neural information* processing systems, volume 28. Curran Associates, Inc.,
2015. URL https://proceedings.neurips.

cc/paper_files/paper/2015/file/ 3fb451ca2e89b3a13095b059d8705b15-Paper.

pdf.

Yoon, J., Jordon, J., and Schaar, M. Gain: Missing data imputation using generative adversarial nets. In International conference on machine learning, pp. 5689–5698, 2018. tex.organization: PMLR.

Yu, S., Drton, M., and Shojaie, A. Graphical models for non-negative data using generalized score matching.

In Storkey, A. and Perez-Cruz, F. (eds.), Proceedings of the twenty-first international conference on artificial intelligence and statistics, volume 84 of Proceedings of machine learning research, pp. 1781–1790. PMLR, April 2018. URL https://proceedings.mlr. press/v84/yu18b.html.

Yu, S., Drton, M., and Shojaie, A. Generalized score matching for general domains. Information and Inference: A Journal of the IMA, 11(2):739–780, 2022.

## A. Additional Theoretical Results

Here we present some interesting results which we feel help further build up the landscape of our method but were unable to fit within the main body of the paper.

## A.1. Additional Methods

Firstly some additional adaptations of score matching. Most of these are relatively immediate adaptations following our framework for missing score matching although there are some important aspects and caveats which make them worth officially documenting. Missing and sliced score matching are introduced in detail in Appendix D.

A.1.1. TRUNCATED SCORE MATCHING
We have already presented truncated score matching in the paper however we present it in more details alongside its assumptions here. Assumption A.1. For any λ ∈ supp(Λ), θ ∈ Θ:
- Xλ is connected, open and Lipschitz;
- pλ, gλ, qλ;θ ∈ H1(Xλ);
- pλ, gλ are continuously differentiable and qθ;λ is twice continuously differentiable;
- for any x
′
λ ∈ ∂Xλ, and j ∈ λ we have

$$\operatorname*{lim}_{\mathbf{x}_{\lambda}\longrightarrow\mathbf{x}^{\prime}}\mathbf{s}_{\lambda;\theta}(\mathbf{x}_{\lambda})_{j}p_{\lambda}(\mathbf{x}_{\lambda})g_{\lambda}(\mathbf{x}_{\lambda})v_{j}(\mathbf{x}_{\lambda}^{\prime})=0.$$
$${}^{2}\,\}=L_{\mathrm{{TM}}}(\theta)-C$$
$$(12)$$
$$\square$$
xλ−→x′λ
where v(x
′
λ) is the normal vector to the boundary δXλ.

This now leads us to our proposition on the validity of our population objective. Proposition A.2. *Suppose that assumptions 4.2 & A.1 hold. Then we have* JTM(θ) := EgΛ(X)∥sΛ;θ(XΛ) − sΛ(XΛ)∥
2	= LTM(θ) − C (12)
where C is does not depend upon θ. As a direct result for ˜θ a minimiser of LTM(θ) *we have that* sθ˜(X) = s(X) a.s..

Proof. Proof given in Appendix C.1.1 A.1.2. MISSING SLICED SCORE MATCHING
For readers who are unfamiliar with sliced score matching we provide a brief introduction in Appendix D.3. For sliced score matching the only adaptations we need to make is to use our marginal scores and now alter our projection vectors to be over the appropriate subspace. Thus our objective becomes

$L_{\rm SM}(\theta):=\mathbb{E}[2\left\{\nabla_{X_{\Lambda}}(V_{\Lambda}^{\top}\,\mathbf{s}_{\Lambda;\theta}(X_{\Lambda}))\right\}^{\top}V_{\Lambda}+V_{\Lambda}^{\top}\,\mathbf{s}_{\Lambda;\theta}(X_{\Lambda})]$  $=F_{\rm SM}(\theta)-C$
where for any λ ∈ supp(Λ), Vλ is a RV on R
|λ|satisfying E[VλC
⊤
λ] positive definite and E[∥Vλ∥
2] < ∞.

To write this and it's gradient in terms of the full score, sθ we can again use Lemma 4.8.

This gives the following results Proposition A.3.

LSM(θ) =2E hE ′h∇XΛ V ⊤ Λ sθ(XΛ, X′−Λ ⊤VΛ) i+ E ′[(V ⊤sθ(XΛ, X′−Λ))2] − E ′[(V ⊤sθ(XΛ, X′−Λ))]2i(13) ∇θLSM(θ) =2E ΨΛ ∇XΛ V ⊤ Λ sθ(.)⊤VΛ) + ΨΛV ⊤sθ(.))2− E ′[V ⊤sθ(XΛ, X′−Λ)]ΨΛV

⊤sθ(.)(14)
$$(13)$$  $$(14)$$
where for any function hθ : R
d → R,

$$\Psi_{\Lambda}(h_{\theta})=\mathbb{D}$$
$$_{\Lambda},X_{-\Lambda}^{\prime})]+\mathrm{Cov}^{\prime}\big(\nabla_{\theta}\log q_{\theta}(X_{\Lambda},X_{-\Lambda}^{\prime}),h_{\theta}(X_{\Lambda},X_{-\Lambda}^{\prime})\big)$$

and E
′, Cov′*are w.r.t.* X′−Λ
|XΛ ∼ pθ(.|XΛ) with E *being w.r.t.* XΛ ∼ p.

Proof. We first have that

LSM(θ) = E
h2∇XΛ
(V
⊤
Λ E
′[sθ(XΛ, X′−Λ)Λ])	⊤VΛ + (V
⊤
Λ E
′[sθ(XΛ, X′−Λ)Λ)
2]
i
= E

2X
j∈Λ
Vj∇XΛ E
′[sθ(XΛ, X′−Λ)j ]
⊤VΛ

 + (V
⊤
Λ E
′[sθ(XΛ, X′−Λ)Λ])2

= E

2X
j∈Λ
VjE
′[∇XΛ sθ(XΛ, X′−Λ)j ] + Cov(sθ(XΛ, X′−Λ), sθ(XΛ, X′−Λ)j )⊤VΛ

+ (V
⊤
Λ E
′[sθ(XΛ, X′−Λ)Λ])2

= E
 2E
′h∇XΛ
V
⊤
Λ sθ(XΛ, X′−Λ
⊤VΛ)
i+ E
′[(V
⊤sθ(XΛ, X′−Λ))2] − E
′[(V
⊤sθ(XΛ, X′−Λ))]2
+ E
′[V
⊤
Λ sθ(XΛ, X′−Λ)Λ]
2
= 2E
hE

′h∇XΛ
V
⊤
Λ sθ(XΛ, X′−Λ
⊤VΛ)
i+ E
′[(V
⊤sθ(XΛ, X′−Λ))2] − E
′[(V
⊤sθ(XΛ, X′−Λ))]2i
The second results directly from applying Lemma 4.8 again alongside the chain rule.

A.1.3. MISSING DENOISED SCORE MATCHING
As with sliced score matching the adaptation is relatively immediate however we do first need to make some further restrictions on our noising process. Specifically we require that for any t ∈ [0, 1], and *j, j*′ ∈ [d] we have X(t)j ⊥
X(t)j
′ |X(0)j .

In most practical implementations each coordinate is independently noised therefore satisfying this condition. We require this to allow us to easily write the marginal transition kernel for any λ ∈ supp(Λ) given by pλ(xλ(t)|xλ(0)). We then make our population objective

$${\cal L}_{\mathrm{DM}}(\theta):=\mathbb{E}\left[\nu(t)\left\{\|s_{\Lambda;\theta}(X_{\Lambda}(t),t)\|^{2}+\nabla_{X_{\Lambda}(t)}\log p_{\Lambda}(X_{\Lambda}(t)|X_{\Lambda}(0))\right\}\right].$$

We can again write this in terms of sθ as we do in the following proposition Proposition A.4.

LDM(θ) = E  ν(t)   X j∈Λ E ′-sθ(XΛ, X′−Λ)j 2+ ∇XΛ(t)log pΛ(XΛ(t)|XΛ(0))     (15) ∇θLDM(θ) = E " ν(t) X j∈Λ E ′-sθ(XΛ(t), X′−Λ, t)jE ′[∂jsθ(XΛ(t), X′−Λ)j ] (16) + Cov′(sθ(XΛ(t), X′−Λ, sθ(XΛ(t), X′−Λ, t)j ) 

$$+\ \nabla_{X_{\Lambda}(t)}\log p_{\Lambda}(X_{\Lambda}(t)|X_{\Lambda}(0))\Biggr\}$$
$$g\,q_{\theta}(X_{\Lambda}(t),X_{-\Lambda}^{\prime}),h_{\theta}(X_{\Lambda}(t),X_{-\Lambda}^{\prime}))$$
where for any function hθ : R
d → R,
ΨΛ(hθ) =E
′[∇θhθ(XΛ(t), X′−Λ)] + Cov′∇θ log qθ(XΛ(t), X′−Λ), hθ(XΛ(t), X′−Λ)
and E
′, Cov′are w.r.t. X′−Λ|XΛ(t) ∼ pθ(.|XΛ) with E *being w.r.t.* XΛ(t) ∼ pt.

Proof. Using Lemma 4.8, we have that

$$\Psi_{\Lambda}(I)$$
Proof.: Using Lemma 4.8, we have that  $$L_{\mathrm{DM}}(\theta)=\mathbb{E}\left[\nu(t)\left\{\sum_{j\in\Lambda}\boldsymbol{s}_{\Lambda,\theta}(X(t)_{\Lambda},t)_{j}^{2}\ +\ \nabla_{X_{\Lambda}(t)}\log p_{\Lambda}(X_{\Lambda}(t)|X_{\Lambda}(0))\right\}\right]$$ $$=\mathbb{E}\left[\nu(t)\left\{\sum_{j\in\Lambda}\mathbb{E}^{\prime}\left[\boldsymbol{s}_{\theta}(X_{\Lambda}(t),X^{\prime}_{-\Lambda},t)_{j}\right]^{2}\ +\ \nabla_{X_{\Lambda}(t)}\log p_{\Lambda}(X_{\Lambda}(t)|X_{\Lambda}(0))\right\}\right].$$  A second application of the lemma then gives,

A second application of the lemma then gives, ∇θLDM(θ) = E " ν(t) X j∈Λ E ′-sθ(XΛ(t), X′−Λ, t)jE ′[∂jsθ(XΛ(t), X′−Λ)j ] + Cov′(sθ(XΛ(t), X′−Λ, t), sθ(XΛ(t), X′−Λ, t)j )} 

$$+\ \nabla_{X_{\Lambda}(t)}\log p_{\Lambda}(X_{\Lambda}(t)|X_{\Lambda}(0))\Biggr\}\Biggr].$$

A.1.4. MISSING NOT AT RANDOM DATA
So far we have assumed that our data is missing completely at random so that XΛ|Λ = λ ∼ Xλ. In other words, we could treat corrupted samples as though they were simply marginal samples and still perform valid inference. Often however, such an assumption is unrealistic and the probability of parts of a sample begin missing depends upon the sample itself. Generally this is split into two cases. Missing at Random (MAR) and Missing not at Random (MNAR). MAR data occurs when the probability of a coordinate being missing depends only upon other coordinates of the sample. This means that Mj ⊥ Xj |X−j . In MNAR data we allow Mj to depend upon Xj as well meaning that an observations value determines its own probability of being missing. Here we will focus in the MNAR scenario and treat the MAR scenario as a special case of this. The core idea of this approach will be to work with a "joint" score rather than a marginal score. Before we do this we need to set-up our MNAR case. Specifically for λ ∈ supp(Λ) define the event

$$E_{\lambda}:=\{X_{\lambda}^{\prime}\neq\varnothing,X_{-\lambda}^{\prime}=\varnothing\}$$

and define φλ(X) := P(Eλ|X). Throughout we will assume each φλ to be *known*. This is often an unrealistic assumption however this allows us the flexibility of having a method which is independent of how the φλ are learned.

To work with this MNAR data we need to define some adaptations of densities and score functions. Definition A.5. X with PDF p and event E define p(x; E) to be the "joint" density satisfying

$$\int_{B}p(\mathbf{x};E)\mathrm{d}x=\mathbb{P}(\{X\in B\}\cup E)$$
B
for all B ∈ BX .

From this and with our particular events we can redefine the missing score as,

sλ(xΛ) = ∇xλlog pλ(xλ; Eλ) (17)
$$\begin{split}\boldsymbol{s}_{\lambda}(\boldsymbol{x}_{\Lambda})&=\nabla_{\boldsymbol{x}_{\lambda}}\log p_{\lambda}(\boldsymbol{x}_{\lambda};E_{\lambda})\\ &=\nabla_{\boldsymbol{x}_{\lambda}}\log\left(\int p(\boldsymbol{x};E)\mathrm{d}\boldsymbol{x}_{-\lambda}\right)\\ &=\nabla_{\boldsymbol{x}_{\lambda}}\log\left(\int p(\boldsymbol{x})\varphi_{\lambda}(\boldsymbol{x})\mathrm{d}\boldsymbol{x}_{-\lambda}\right)\end{split}$$
(18)
$$(17)$$
$$(18)$$
14 Remark A.6. this missing score is not the same as the marginal score. We slightly abuse notation here using the same notation as we did for the marginal score. This is however reasonable as for the MCAR case the marginal score and the missing score are identical.

With this newly defined score, we can proceed similarly to the MCAR case and use the objective LˆM(θ) defined in (4) or
(5) but with our new defined score. We now show a provide a similar justification for this approach as in the MCAR case but first need to introduce an additional assumption.

Assumption A.7. For each λ ∈ supp(Λ), P(Eλ|Xλ) > 0 a.s..

Remark A.8. We do not require every missingness pattern to have positive probability just that if a missingness pattern does have positive probability, it has positive probability for every possible underlying sample. This then leads us to our desired result. Proposition A.9. Suppose with are in our MNAR set-up and assume that assumptions 4.2 & A.7 hold and that there exists θ
∗ *with* sθ
∗ (X) = sθ(X) a.s.. Then if ˜θ is a minimiser of LM(θ) *where the missing scores are defined by (17 then* sθ˜(X) = s(X) *a.s..*
The proof for this is similar to the MCAR case and is given in Appendix C.1.2.

Now we have our objective we need to see how we can derive sλ(xλ). Again we can do this similarly to the MCAR case. Let qθ be our estimate of the unnormalised density then

sλ;θ(xλ) = ∇xλlog pλ;θ(xλ; Eλ) = ∇xλlog ZX−λ pθ(x; Eλ)dx−Λ = ∇xλlog Z X−λ pθ(x)φλ(x)dx−Λ = ∇xλlog ZX−λ qθ(x)φλ(x)dx−Λ = ∇xλlog Ep′ qθ(xλ, X′−λ)φλ(xλ, X′−λ) p ′(X′−λ )  k=1  "qθ(xλ, X ′(k) −λ )φλ(xλ, X ′(k) −λ ) p ′(X ′(k) −λ ) ≈ ∇xλ log 1r X r #
As a result we can approximate our objective analogously to our approach for MCAR data. A.2. Finite Sample Bounds for Truncated Importance Weighted Score Matching To be able to prove finite sample bound results we first need to present some key definitions.

Definition A.10 (Approximate Truncated Marginal Score Matching Objective). For n, r ∈ N, θ ∈ Θ take our sample objective to be

LˆTM;n,r(θ) := 1 n X n i=1 gΛi(X (i) Λi )∥sˆΛi,r;θ(X (i) Λi )∥ 2+2gΛi(X (i) Λi )∇X (i) Λi · sˆΛi,r;θ(X (i) Λi ) + 2∇X (i) Λi gΛi(X (i) Λi ) ⊤sˆΛi,r;θ(X (i) Λi )
with sˆλ,r;θ(xλ) being our estimated marginal score from Definition 4.4.

Additionally we define

$$f_{0,\lambda}(\mathbf{x},\theta):=\frac{g_{\theta}(\mathbf{x})}{p^{\prime}(\mathbf{x}_{-\lambda})}\qquad\qquad f_{1,\lambda}(\mathbf{x},\theta):=\frac{\nabla_{\mathbf{x}}g_{\theta}(\mathbf{x})}{p^{\prime}(\mathbf{x}_{-\lambda})}\qquad\qquad f_{2,\lambda}(\mathbf{x},\theta):=\frac{\nabla_{\mathbf{x}}cdot(\nabla_{\mathbf{x}}g_{\theta}(\mathbf{x}))}{p^{\prime}(\mathbf{x}_{-\lambda})}.$$

We now set-up the following assumptions Assumption A.11. There exists a > 0 s.t. for all x ∈ X , λ ∈ supp(Λ), k ∈ {0, 1, 2}
- ∥fk,λ(x, θ)∥, gλ(xλ), ∥∇xλ gλ(xλ)∥< a,
•1a < f0,λ(x, θ)
Remark A.12. It is this assumptions which restrict us from obtaining a similar result in the non-truncated case as it is unrealistic to have both 1a < f0,λ(x) and p(xλ) → 0 as ∥xλ*∥→ ∞*.

Assumption A.13. For each λ ∈ supp(Λ), l ∈ {0, 1, 2} we have that for any *θ, θ*′ ∈ Θ:

$$\|f_{l,\lambda}(\mathbf{x},\theta)-f_{0,\lambda}(\mathbf{x},\theta^{\prime})|\leq M_{k}(\mathbf{x})\rho(\theta,\theta^{\prime}),$$

where Mk(Xλ, x−λ), Mk(xλ, X′−λ
) are sub-Gaussian with parameters σl,λ, σ′l,−λrespectively for all x−λ ∈ X−λ.

Remark A.14. This assumption is immediately satisfied if Θ is compact and fl,λ(x, θ) are pointwise Lipschitz w.r.t. θ.

Hence this assumption is slightly weaker than a uniformly lipschitz assumption We can now state our theorem Theorem A.15. Assume that assumptions 4.2, A.1, A.11, A.13 hold and let θn,r ∈ Θ ⊆ R
p *be the minimisers of* LˆTM;n,r(θ)*. If* Θ ⊆ R
pthen for sufficiently large *n, r*

$$\mathbb{P}\left(F_{\text{TM}}(\theta_{n,r})\geq\beta_{1}\sqrt{\frac{p\log(d n r\,\text{diam}(\Theta)/\delta)}{r}}+\beta_{2}\sqrt{\frac{p\log(n\,\text{diam}(\Theta)/\delta)}{n}}+\beta_{3}\left(\frac{n+r}{n r}\right)\left(C+\sqrt{\frac{\log(n/\delta)}{n}}\right)\right)<\delta.$$

where β1, β2 depend upon a, β3 depends upon a, {σλ,l, σ′−λ,l}(l,λ)∈{0,1,2}×supp(Λ) and C *depends upon* a,
{E[Mk(Xλ, X′−λ
)]}(l,λ)∈{0,1,2}×supp(Λ).

Proof. The proof for this alongside multiple intermediary results can be found in C.2 Here we have shown convergence of our sample/approximate objective to the population objective. This combined with proposition A.2 which states that our population objective is minimised by the true score suggests that our approach does give valid inference for learning the score. A key limitation of our result is that to obtain convergence, we require r−→∞.

Furthermore, to obtain log(n)/n rate convergence we need r to be of the same order as n. As the computational complexity of our algorithm in O(nr), this suggests that to obtain our desired convergence to the population objective will have O(n 2)
computational complexity. Remark *A.16*. Our dependency on our Lipschitz constants only enters into the C term with the associate sub-Gaussian parameters entering only into the σ. Remark A.17. Dependence upon g simply requires g and ∇g bounded. This is achieved on a compact X by g(x) =
minx′∈∂X d(x, x
′) and on a non-compact space by g(x) = minx′∈∂X d(x, x
′)V1.

## A.3. Relationship Between Iw And Variational Objectives

Despite being derived quite differently from the marginal score matching objective. We show below that the two objectives are actually identical in some cases. Specifically, when the IW density p
′ doesn't depend upon the observed data xΛ, we can treat the importance weighted approach as an importance weighting approximation of the gradient estimate in (10). We demonstrate this through the two results below Lemma A.18. *For some density* p
′ *which generates IID samples* {X
′(k)
−λ}k∈r let

wk := qθ(xλ, X′(k) −λ) p ′(X ′(k) −λ ) w¯k := wk  Xr k′=1 wk′ !−1 sˆθ,λ(xλ) := ∇xλ log 1r X r k=1 wk Eˆiw[gθ(X)] := 1 r X r k=1 w¯kgθ(xλ, X′(k) −λ ) Covˆiw(f(X), gθ(X)) := 1 r X r k=1 w¯kgθ(xλ, X′(k) −λ )f(xλ, X′(k) −λ ) −  1 r X r k=1 w¯kgθ(xλ, X′(k) −λ ) ! 1 r X r k=1 w¯kf(xλ, X′(k) −λ ) ! .
$$\square$$
$$\square$$
Then

$$\hat{\mathbf{s}}_{\theta,\lambda}(\mathbf{x}_{\lambda})=\hat{\mathbb{E}}_{iw}[\mathbf{s}_{\theta}(\mathbf{x}_{\lambda},X^{\prime}_{-\lambda})]\tag{19}$$ $$\nabla\hat{\mathbb{E}}_{iw}[g_{\theta}(\mathbf{x}_{\lambda},X^{\prime}_{-\lambda})]=\hat{\mathbb{E}}_{iw}[\nabla g_{\theta}(\mathbf{x}_{\lambda},X^{\prime}_{-\lambda})]+\hat{\mathbb{C}}\text{ov}_{iw}(\mathbf{s}_{\theta}(\mathbf{x}_{\lambda},X^{\prime}_{-\lambda}),g_{\theta}(\mathbf{x}_{\lambda},X^{\prime}_{-\lambda})).\tag{20}$$

where ∇ represents the gradient w.r.t. xλ or θ*. In other words, we can take importance weights first then gradients (LHS)*
or gradients and then importance weights (RHS).

Proof. Proof given in Section C.4.

Corollary A.19. *We have that*

∇θLˆ(θ; xλ, X′−Λ) = − 2Eˆiw[sθ(xλ, X′−Λ)i] nEˆiw[∇θsθ(xλ, X′−Λ)i] + Covˆiw ∇θ log qθ(xλ, X′−Λ), sθ(xλ, X′−Λ)io + 2(Eˆiw[∇θsθ(xλ, X′−Λ) 2 i] + Covˆiw ∇θ log qθ(xλ, X′−Λ), sθ(xλ, X′−Λ) 2 i ) (21) + 2(Eˆiw[∇θ∂isθ(xλ, X′−Λ)i] + Covˆiw ∇θ log qθ(xλ, X′−Λ), ∂isθ(xλ, X′−Λ)i)
Proof. Proof given in Section C.4. A.4. Exploring the Marginal Fisher Divergence for Normal Distributions While intuitively the Fisher divergences of the marginal distributions should act as effective proxies for the Fisher divergence for the fully observed distributions, we would like to be able to examine the relationship between the two more explicitly. We do know that marginal Fisher divergences will be zero when then fully observed distributions equivalent however here we give a more detailed examination in the case of normal distributions.

Suppose that X ∼ N(*µ, P* −1)

$$p(x)=\exp\{-\frac{1}{2}(x-\mu)^{\top}P(x-\mu)\}+C$$

with C a constant w.r.t. x. We then have that s(x) = −P(x − µ)
If we suppose that our unnormalised density/score model is of the form

$$q_{\theta}(\mathbf{x}):=\exp\left\{-{\frac{1}{2}}(\mathbf{x}-\mathbf{\mu}_{\theta})^{\top}P_{\theta}(\mathbf{x}-\mathbf{\mu}_{\theta})\right\}$$
$$\Rightarrow\mathbf{s}_{\theta}=P_{\theta}(\mathbf{x}-\mu_{\theta})$$. 
Then with the marginal Fisher taken to be

$$F_{\mathrm{M}}(\theta)=\mathbb{E}_{\Lambda,X_{\Lambda}}\left[\|\mathbf{s}_{\Lambda}(X_{\Lambda})\mathbf{s}_{\theta;\Lambda}\|^{2}\right]$$

where here for each λ ∈ supp(Λ), sλ, sθ;λ are the true marginal scores. Using properties of the normal distribution and the Schur complement we know that the precision of Xλ is given by

$$\left\{\left(P^{-1}\right)_{\lambda,\lambda}\right\}^{-1}=P_{\lambda,\lambda}-P_{\lambda,-\lambda}P_{-\lambda,-\lambda}^{-1}P_{-\lambda,\lambda}.$$

Plugging this in we get

$$F_{\rm M}(\theta)=\mathbb{E}\Bigg{[}\Bigg{]}\Big{[}(P_{\Lambda,\Lambda}-P_{\theta;\Lambda,\Lambda})X_{\Lambda}+(P_{\Lambda,-\Lambda}P_{-\Lambda,-\Lambda}^{-1}P_{-\Lambda,\Lambda}-P_{\theta;\Lambda,-\Lambda}P_{\theta;-\Lambda,-\Lambda}^{-1}P_{\theta;-\Lambda,-\Lambda})X_{\Lambda}$$ $$\qquad-((P_{\Lambda,\Lambda}-P_{\Lambda,-\Lambda}P_{-\Lambda,-\Lambda}^{-1}P_{-\Lambda,\Lambda})\mu_{\Lambda}-(P_{\theta;\Lambda,\Lambda}-P_{\theta;\Lambda,-\Lambda}P_{\theta;-\Lambda,-\Lambda}^{-1}P_{\theta;-\Lambda,\Lambda})\mu_{\theta;\Lambda})\Bigg{]}^{2}\Bigg{]}.$$
.
This shows why naive marginalisation by zeroing out missing coordinates of our score would not work as in this case the Fisher divergence would be given by

FM(θ) =E " ((PΛ,Λ − PΛ,−ΛP −1 −Λ,−ΛP−Λ,Λ) − Pθ;Λ,Λ)XΛ − ((PΛ,Λ − PΛ,−ΛP −1 −Λ,−ΛP−Λ,Λ)µΛ − Pθ;Λ,Λµθ;Λ)  2#
which encourages Pθ;λ,λ to be close to Pλ,λ − Pλ,−λP
−1
−λ,−λP−λ,λ for all λ ∈ supp(Λ) meaning it will not give us the true density. A.5. Variational Pseudo-loss When using (10) it is helpful to be able to view it as the gradient of some pseudo-loss allowing it to plug into a more standard ML framework where we calculate the loss, take the gradient w.r.t. our parameter using auto-differentiation, and update our parameter estimate based on this. The below result show how we can do this by creating a loss with certain instances of our parameter detached from the computational graph. Proposition A.20. Let

J(θ, θ′, xλ, X′−λ) := − 2E ′[sθ ′ (xλ, X′−Λ)i]E ′[sθ(xλ, X′−Λ)i] + Cov′log qθ(xλ, X′−Λ), sθ ′ (xλ, X′−Λ)i	 + 2(E ′[sθ(xλ, X′−Λ) 2 i] + Cov′log qθ(xλ, X′−Λ), sθ′ (xλ, X′−Λ) 2 i ) + 2(E ′[∂isθ(xλ, X′−Λ)i] + Cov′log qθ(xλ, X′−Λ), ∂isθ ′ (xλ, X′−Λ)i )
where E
′, Cov′*are w.r.t.* X′−λ|Xλ = xλ; θ
′ *Then*

$$\nabla_{\theta^{\prime}}L(\theta^{\prime},{\mathbf{x}}_{\lambda})=\left.{\frac{\partial}{\partial\theta}}J(\theta,\theta^{\prime},{\mathbf{x}}_{\lambda})\right|_{\theta=\theta^{\prime}}$$

Proof. This just follows directly from the exchangeability of expectations and gradients (when the gradient is w.r.t. something independent of the expectation distribution.) Hence we can use this loss (by replacing all instances of θ
′ with θ and then detaching them from the computation graph)
to treat our problem as a standard gradient descent problem. Note that while we can treat this like a loss for our optimisation, our intent is not actually to minimise it. The estimated form of the loss is given in the proof of Corollary 4.9 which is given in C.3 but we state it again explicitly here for convenience.

$$L_{\rm M}(\theta)=\mathbb{E}\left[\sum_{j\in\Lambda}-\mathbb{E}^{\prime}[s_{\theta}(X_{\Lambda},X^{\prime}_{-\lambda})_{j}]^{2}+2\mathbb{E}^{\prime}[s_{\theta}(X_{\Lambda},X^{\prime}_{-\Lambda})_{j}^{2}]+2\mathbb{E}^{\prime}[\partial_{i}s_{\theta}(X_{\Lambda},X^{\prime}_{-\Lambda})_{j}]\right].$$

Pre ci si o n Di st a n c e 
(F
ro b e ni u s)
Marg-IW (Ours)
Marg-Var (Ours)
Zeroed EM
200 400 600 800 1000 Sample Size 0.0 0.5 1.0 1.5 2.0 M
e a n Di sta nc e 
(L
2)Marg-IW (Ours)
Marg-Var (Ours)
Zeroed EM
200 400 600 800 1000 Sample Size 0.0 0.5 1.0 1.5 2.0
Figure 8: Average parameter estimation error for truncated Gaussian score estimates alongside 95% confidence intervals under various methods.

## B. Additional Experimental Results

Here we present some additional experimental results not in the main body of the paper.

## B.1. Parameter Estimation

B.1.1. TRUNCATED GAUSSIAN MODEL
Here we present the accompanying mean and precision error results for Gaussian model estimation experiment presented in Section 5.1.1. These results are presented in Figure 8.

B.1.2. UNTRUNCATED GAUSSIAN MODEL
Here we present the untruncated version of the experiment presented in the main paper. Details of the distribution are the same as presented in Appendix E.3 but without the truncation.

200 400 600 800 1000 Sample Size 0.0 0.5 1.0 1.5 2.0 2.5 Fi s h e r Di v er g e nc e Marg-IW (Ours)
Marg-Var (Ours) Zeroed EM
As we can see we obtains similar results here as in the truncated case. We also illustrate what the true covariance and precision matrix look like for this example alongside the naive marginalisation in order to highlight where Zeroed Score Matching goes wrong. In Figure 10 we can see the covariance and precision of a sample distribution where we can clearly see the strong dependence of dimensions 1 and 10 relative to the others.

In Figure 11 we can see the naive and true marginal precisions when dimension 1 is missing. For this plot the values have

1 1 9 10 −3
−2 −1 0 1 2 3 1.0 2 2 3 3 0.5 4 4 5 5 0.0 6 6 7 7 9 10 −1.0
−0.5 8 8 1 2 3 4 5 6 7 8 9 10 1 2 3 4 5 6 7 8 9 10
(a) Covariance
(b) Precision
been cube-rooted in order to emphasize the difference between zero and non-zero entries. Here we can see that the naive marginalisation wouldn't capture the dependence between dimension 10 and the other dimensions that gets introduced when dimension 1 is removed. This means that a naive marginalisation would assume that dimension 10 must have a direct dependence on dimensions 2-9 even when that is not true. Interestingly, the rest of the marginalisation seems very similar suggesting that in some potentially less structured cases, naive marginalisation can provide a semi-reasonable approximation. This supports the results we see in our GGM estimation where highly structured graphs like star graphs are much more affected naive marginalisation than unstructured graphs.

## B.1.3. Non-Gaussian Estimation

Here we present further experiments exploring the non-Gaussian model presented in Section 5.1.2. Here we fix the dimension as 10. In Figure 12a we fix the missing probability as 0.5 and vary the sample size. In Figure 12b we fix the sample size as 1000 and vary the missing probability. From Figure 12a we see that both EM and Marg-IW have the smallest estimation error for larger sample sizes. Zeroed Score Matching has the largest estimation error due to its inability to appropriately marginalise the distribution. In Figure 12b, we observe that Marg-Var has the smallest estimation error with its performance convergence to that of Marg-IW and EM as the missing probability increases.

## B.2. Ggm Estimation

B.2.1. VARYING NUMBER OF STAR CENTRES
Here we present illustrations of the marginalisations for our star-shaped graphs with 1 node and then 5 nodes both with the same edge density. In Figure 13 we show the covariance, precision, marginal covariance, and marginal precision for a star graph with 1 centre where the marginal terms are with dimension 1 removed. As we can see clearly the only meaningful structure left in the graph after marginalisation are in the negative precision terms which the model naive marginalisation fails to capture. In Figure 14 we show the same thing for the case of a star graph with 5 centres. As we can see in the 5 centre case, the naive marginalisation picks up most of the structure as there are fewer negative terms which it ignores and also lots of additional positive terms which it does successfully pick up.

## B.2.2. Varying Number Of Dimensions

Here we use our same star-shaped GGM as in the main paper but with a varying number of dimensions. Throughout 1,000 samples are used and each coordinate is missing independently with probability 0.7. Results are presented in Figure 15.

As we can see, for higher dimensions the variational approach clearly performs best however at lower dimensions the other