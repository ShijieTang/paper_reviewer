# Mixed Likelihood Variational Gaussian Processes

## Anonymous Authors1 Abstract

Gaussian processes (GPs) are powerful models for human-in-the-loop experiments due to their flexibility and well-calibrated uncertainty. However, GPs modeling human responses typically ignore auxiliary information, including a priori domain expertise and non-task performance information like user confidence ratings. We propose mixed likelihood variational GPs to leverage auxiliary information, which combine multiple likelihoods in a single evidence lower bound to model multiple types of data. We demonstrate the benefits of mixing likelihoods in three realworld experiments with human participants. First, we use mixed likelihood training to impose prior knowledge constraints in GP classifiers, which accelerates active learning in a visual perception task where users are asked to identify geometric errors resulting from camera position errors in virtual reality. Second, we show that leveraging Likert scale confidence ratings by mixed likelihood training improves model fitting for haptic perception of surface roughness. Lastly, we show that Likert scale confidence ratings improve human preference learning in robot gait optimization. The modeling performance improvements found using our framework across this diverse set of applications illustrates the benefits of incorporating auxiliary information into active learning and preference learning by using mixed likelihoods to jointly model multiple inputs.

## 1. Introduction

Gaussian process (GPs) are indispensable models for many machine learning and AI applications (Williams & Rasmussen, 2006). As a Bayesian nonparametric model, it is favored for its well-calibrated uncertainty estimates and flexibility. When using a GP for regression with Gaussian 1Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>.

1 000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054 errors, i.e., a Gaussian likelihood, the posterior distribution is a multivariate normal whose mean and covariance can be computed analytically via the Kriging equations (Gramacy, 2020). This analytic tractability does not hold for other types of observations, such as classification or preference data, but GP modeling in those settings can be done with variational approximation (e.g., Hensman et al., 2013; 2015) or other approximate inference schemes (Kuss & Rasmussen, 2005).

Human feedback, which is generally non-Gaussian, has recently become an important setting for GP modeling with applications including health screening (Gardner et al.,
2015a;b), AR/VR development (Guan et al., 2022; 2023; Kwak et al., 2024), and robot locomotion learning (Tucker et al., 2020). In particular, preference learning has attracted a great deal of attention in recent years for its usefulness in large language model (LLM) training and reinforcement learning with human feedback (RLHF) (Stiennon et al., 2020; Ouyang et al., 2022). GPs are a natural fit for many preference learning problems (Chu & Ghahramani, 2005; Houlsby et al., 2012), including for RLHF (Kupcsik et al., 2018). Due to their well calibrated uncertainty, GPs are especially useful in human-in-the-loop experiments where the human's time is valuable, as GPs can be used with active learning to increase trial efficiency (Owen et al., 2021). In many non-Gaussian observation settings, multiple data of different types can be observed simultaneously. For example, in preference learning, we can solicit both preferences (binary comparison data) and strengths of preference (e.g., Likert scale survey data). Studies of human perception can measure whether or not a stimulus was perceived (binary classification data) while simultaneously recording response time (continuous but non-Gaussian data). Presumably, combining these different types of data into a single GP would help improve modeling performance. In addition, we may also have domain knowledge about the responses for some special inputs. For instance, in studies of human perception, a stimulus with no intensity cannot be perceived at all. This domain knowledge constraint, as we will show, can be also be considered as an additional observation type that we wish to include in the GP. Here, we present a framework for joint GP modeling of multiple types of data and expand GP modeling to a rich new set of multi-data-type problems via the following contributions:

## 2. Background

055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109

A likelihood is a distribution that models the relation between the observed training labels y and the latent function values f. Thus, different types of data require different likelihoods. For regression, it is common to use a Gaussian
likelihood
$$p(\mathbf{y}\mid\mathbf{f})={\mathcal{N}}(\mathbf{y};\mathbf{f},\sigma^{2}\mathbf{I}),$$
where σ is the standard deviation of the label noise. For classification, it is common to use a Bernoulli likelihood
- We develop a novel evidence lower bound (ELBO)
formulation that includes multiple likelihoods in the same variational approximation.

- We show that our mixed likelihood training can be used to encode domain knowledge in non-Gaussian settings, and thereby accelerate active learning.

- We develop both synthetic and real-world examples of mixed likelihood variational GPs improving model performance by incorporating auxiliary survey data into preference learning and human perception studies, also developing a new Likert likelihood.

A Gaussian process (GP) f ∼ GP(0, k) defined by a kernel function k : R
d × R
d → R is a stochastic process whose function values f =f(X) on any training data X ∈ R
n×d follow a joint Gaussian distribution f ∼ N (0, Kf,f), where Kf,f = k(f(X), f(X)) is the covariance matrix.

$$p(\mathbf{y}\mid\mathbf{f})=\operatorname{Bernoulli}(\Phi(\mathbf{f})).$$

On the test data X∗, the GP prediction is the posterior conditioned on the training labels p(f
∗| y). For a Gaussian likelihood, the posterior distribution is also Gaussian with a closed-form expressions for its mean and covariance:

$$\begin{array}{l}{{\mathbb{E}[\mathbf{f}^{*}\mid\mathbf{y}]=\mathbf{K}_{*,\mathbf{f}}(\mathbf{K}_{\mathbf{f},\mathbf{f}}+\sigma^{2}\mathbf{I})^{-1}\mathbf{y},}}\\ {{\mathbb{D}[\mathbf{f}^{*}\mid\mathbf{y}]=\mathbf{K}_{*,*}-\mathbf{K}_{*,\mathbf{f}}(\mathbf{K}_{\mathbf{f},\mathbf{f}}+\sigma^{2}\mathbf{I})^{-1}\mathbf{K}_{\mathbf{f},*}.}}\end{array}$$

However, for non-Gaussian likelihoods, the exact posterior is almost always intractable, and thus needs approximation. Variational GPs (e.g., Titsias, 2009; Hensman et al., 2013; 2015) approximate the exact posterior by inducing point approximation and variational inference (Blei et al., 2017).

A set of inducing variables u is introduced, and the joint distribution factors as follows:

$$p(\mathbf{f},\mathbf{f}^{*},\mathbf{u})=p(\mathbf{f}\mid\mathbf{u})p(\mathbf{f}^{*}\mid\mathbf{u})p(\mathbf{u}),$$

where each component admits the form p(u) = N (u; 0, Ku,u),
p(f | u) = N (f; Kf,uK−1 u,uu, Kf,f − Kf,uK−1 u,uKu,f),
p(f
∗| u) = N (f
∗; K∗,uK−1 u,uu, K∗,∗ − K∗,uK−1 u,uKu,∗).

2 Given the inducing values u, the latent function values on the training data and the test data are conditionally independent. As a result, the prediction on the test data is completely controlled by the inducing variables. Inference in variational GPs is performed by maximizing the evidence lower bound (ELBO):

$$\operatorname*{maximize}_{q(\mathbf{u})}\mathbb{E}_{q(\mathbf{f})}\log p(\mathbf{y}\mid\mathbf{f})-\mathsf{D}_{\mathrm{KL}}(q(\mathbf{u}),p(\mathbf{u})),$$

where the variational distribution q(u) is usually restricted to a Gaussian family and q(f) = Rp(f | u)q(u) du is the marginalized variational distribution over the latent function values. The optimal variational distribution q(u) is then used to construct the approximate posterior

$$p(\mathbf{f}^{*}\mid\mathbf{y})\approx\int p(\mathbf{f}^{*}\mid\mathbf{u})q(\mathbf{u})\,\mathrm{d}\mathbf{u}.$$

## 3. Mixed Likelihood Variational Inference

Suppose we have T different types of data available X(t), y
(t), t = 1, 2, · · · *, T,*
where X(t)'s are training data locations and y
(t)'s are labels of different types. For example, y
(1) could be regression labels while y
(2) are classification labels.

We assume that all labels are generated from the same latent function and that the labels y
(t)are conditionally independent given the latent function values f
(t) = fX(t), across data types t. We then jointly model the different data types by training a single variational GP on all data using a combined ELBO. As before, we use a variational distribution q(u) to approximate the GP posterior. Let y = {y
(t)}
T t=1 now represent the complete collection of training labels across data types, and f = {f
(t)}
T
t=1 their corresponding latent function values. Because of the conditional independence of the various observations, we have that

$$\log p(\mathbf{y}\mid\mathbf{f})=\sum_{t=1}^{T}\log p_{t}{\big(}\mathbf{y}^{(t)}\mid\mathbf{f}^{(t)}{\big)}.$$

Each type of data uses a different likelihood. For instance, p1(*· | ·*) is a Gaussian likelihood if y
(1) are regression labels, and p2(*· | ·*) is a Bernoulli likelihood if y
(2) are classification labels. The evidence term in the ELBO thus decomposes, and we can write a valid evidence lower-bound for mixed likelihoods as:

$$\sum_{t=1}^{T}\mathbb{E}_{q\left(\mathbf{f}^{(t)}\right)}\log p_{t}{\big(}\mathbf{y}^{(t)}\mid\mathbf{f}^{(t)}{\big)}-\mathsf{D}_{\mathrm{KL}}(q(\mathbf{u}),p(\mathbf{u})).\quad(1)$$

For the special case of T = 1, the ELBO in (1) reduces to the usual variational GP ELBO. The interesting behavior happens when T > 1, where multiple types of data are incorporated to learn the same latent function f. By virtue of being a valid evidence lower bound, maximizing (1) yields a variational distribution approximating the GP posterior jointly conditioned on multiple types of data. In the following sections §4 and §5, we demonstrate that this simple idea can solve many problems arising from experimental design and preference learning.

## 4. Encoding Domain Knowledge Constraints In Active Learning

The mixed likelihood training scheme can be used to encode domain knowledge constraints into active learning problems with non-Gaussian data. We demonstrate this in level set estimation with Bernoulli observations, a problem setting with important applications in perception science.

## 4.1. Visual Psychophysics

110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 Understanding human vision and characterizing visual perception is challenging because human self-report is unreliable and individual decision-making criteria are highly variable. Vision scientists use forced-choice experimental paradigms to address these challeneges (Palmer, 1999; Wolfe et al., 2006). Figure 1 describes a psychophysical study design to determine how much render-camera offset is detectable to a person in a stereoscopic (i.e. 3D) display. Rather than asking participants if a particular camera offset looks acceptable, they are given a zero-offset option (the reference) and an option with some offset (the comparison), and asked identify which option has offset. Camera offset is varied over hundreds or thousands of trials with the aim of identifying the set of render-camera offsets that Figure 1: Illustrative depictions of perceived stereoscopic 3D distortions when render cameras are offset from the viewer's eyes. **Left:** Stereoscopic images rendered with cameras at the viewer's eyes have no 3D distortion. **Center:** Small camera offsets result in minimal perceived distortions, and participants cannot reliably identify any errors. **Right:** Large camera offsets result in obvious distortions and are easily recognized. **Bernoulli Level Set Estimation:** If participants are presented the left and middle options in randomized order and asked to select the distorted option, the probability of selecting the correct option will be close to 50%, i.e., at chance. On the other hand, when the left and right options are presented, the distorted option will be selected close to 100% of the time. We aim to identify the space of camera placement configurations such that the distortion detection probability of a participant is below 75%, a threshold that is often considered a just-detectable difference from zero error.

cannot reliably be differentiated from the zero-offset reference. This is often taken as the offsets for which the probability of correctly selecting the comparison stimulus is below 75% (McKee et al., 1985; Ulrich & Miller, 2004), and this problem can be formulated as Bernoulli level set estimation (Letham et al., 2022).

## 4.2. Bernoulli Level Set Estimation

Given a black-box function f : R
d → R, we are concerned with learning the sublevel set {x ∈ R
d: f(x) ≤ γ} for some constant γ ∈ R. The black-box function f cannot be evaluated directly, but can be "probed" by Bernoulli observations. For any x ∈ R
d, we may observe a random variable y(x) ∈ {0, 1} where y(x) ∼ Bernoulli(Φ(f(x))).

We iteratively query the latent function via the Bernoulli observations with the goal of learning the sublevel set. Active learning can be done using a variational GP classification model for f, and one of several acquisition functions for proposing new queries (Letham et al., 2022). The visual psychophysics experiment paradigm described in §4.1 can be cast as Bernoulli level set estimation by taking x as a visual stimulus and f(x) as the perceptual intensity. We measure, via Bernoulli observations y(x), how well the human participant can differentiate between x and the reference stimulus xref, the perceptual intensity of which is zero. The sublevel set of f is the set of imperceptible stimuli which we wish to identify.

## 4.3. Encoding Prior Knowledge With Soft Constraints

Learning level sets with Bernoulli queries is challenging as Bernoulli observations are inherently noisy, especially for 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 detection probabilities close to 50%. We generally require many repeated trials in order to accurately estimate the latent response probabilities. Moreover, in the initial stage of active learning, the model is unable to distinguish between stimulus pairs that are obviously different (100% correct) and pairs that are almost identical (50% correct), but guessed correctly. Therefore, we wish to encode *a priori* knowledge about the experimental paradigm.

In the particular case of the visual perception task of §4.1, we know that the detection probability should be exactly 50% when camera offset is 0, and should be close to 100% for maximum offset values. We expect this extra information will improve efficiency of human-in-the-loop experiments with active learning. Moreover, the domain knowledge constraints may add resiliency to prevent active learning from exploring easily-detectable areas away from the target level set in cases when outlier responses occur during the early stage of data collection, e.g., accidental misclicks when
the camera offset is large, or several consecutive correct
detections by chance even though the camera offset is small.
We encode this type of domain knowledge as constraints
on the latent function by directly regressing against the target constraint values. In addition to the Bernoulli observations (X(1), y
(1)), we produce a set of regression labels
(X(2), y
(2)) provided by domain experts that encode the
known latent function values at special locations—for instance, the latent function value should be zero where the detection probability is known to be 50%. We enforce the soft constraints
f(x
$$\stackrel{(2)}{i})\approx y_{i}^{(2)}$$
by mixing Bernoulli with Gaussian likelihoods:
$d\geq2$
$$p{\Big(}y_{i}^{(2)}\mid f{\big(}\mathbf{x}_{i}^{(2)}{\big)}{\Big)}={\mathcal{N}}{\Big(}y_{i}^{(2)};f{\big(}\mathbf{x}_{i}^{(2)}{\big)},\sigma_{i}^{2}{\Big)},$$
where σi's are fixed noise standard deviations that control the softness or hardness of the constraints. Intuitively, we use the Bernoulli likelihood to fit binary responses and the Gaussian likelihood to enforce the constraints. Figure 2 shows an example of enforcing constraints on an one dimensional objective by mixing Bernoulli and Gaussian likelihoods (σ 2 = 0.001). Mixed likelihood modeling leads to a significant reduction in posterior uncertainty, as regression labels at the points with known value provide stronger learning signals than the Bernoulli observations.

## 4.4. Synthetic Experiments

We show that enforcing constraints effectively encodes domain knowledge and improves active learning for Bernoulli level set estimation. We benchmark on three synthetic latent functions. The first is a synthetic two-dimensional psychometric discrimination

Bernoulli Bernoulli + Gaussian 0.0 0.5 1.0 1.5 2.0
−1 0 1 2 0.0 0.5 1.0 1.5 2.0 true latent f prediction ˆf positive data negative data 95% interval constraints
objective from Letham et al. (2022). The others are scaled norm functions 2∥x∥ in 2D and 4D respectively. All of these synthetic functions have locations where response probabilities equal exactly 50%, and locations where the response probability is close to 100% probabilities. We use mixed likelihood training to set constraints at a subset of these locations—see §A.1 for more details on the functions and the constraint locations.

We set the Gaussian likelihood noise, which determines strength of the constraint, according to the target value yi as: σi = 0.2 · yi + 0.1. Intuitively, this allows a 20% relative violation plus 0.1 absolute violation. For a constraint with yi = 0 (i.e. 50% response probability), this implies an *a priori* 95% credible interval on the response probability of [0.422, 0.578]. For yi = 2 (i.e. a 98% response probability), the credible interval for the response probability is [0.846, 0.999]. This policy was not extensively tuned, but produces a desirable behavior of maintaining soft constraints across the range of response probabilities. Our preliminary experiments indicate that enforcing constraints with strict tolerances, e.g., σ 2 i = 10−4, does not necessarily improve active learning performance as the variational GP tends to be rigid and adapts to new Bernoulli observations slowly. The GP spends most of its prediction capacity fitting the constraints, while spending less weight on Bernoulli observations. This is especially detrimental to look-ahead acquisition functions that depend on the change in posterior conditioning on virtual data. A natural alternative approach that we use as a baseline is to add Bernoulli pseudo data to the standard single-likelihood GP, to push predictions at those locations towards the target constraint values. Pseudo data are added before active learning, and are added at the same locations used for constraints in the mixed likelihood model. Locations with a response

psych. discrimination norm ball (2d)
norm ball (4d)
pseudo+GlobalMI pseudo+EAVC mixed +GlobalMI mixed +EAVC
0 250 500 750 1000 num. of queries 0.4 0.5 0.6 0.7 0.8 0.9 visual sensitivity 0 200 400 num. of queries 0.5 0.6 0.7 0.8 0.9 pseudo+GlobalMI pseudo+EAVC mixed +GlobalMI mixed +EAVC
0 250 500 750 1000 num. of queries 0.3 0.4 0.5 0.6 0.7 0.6 0.7 0.8 f1 score pseudo+GlobalMI pseudo+EAVC mixed +GlobalMI mixed +EAVC
0 200 400 num. of queries pseudo+GlobalMI pseudo+EAVC mixed +GlobalMI mixed +EAVC
220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 probability of 50% were given 5 positive and 5 negative Bernoulli data, while those with response probability close to 100% were given a single positive data. We run active learning for Bernoulli level set estimation with two competitive acquisition functions developed by Letham et al. (2022): global mutual information (GlobalMI) and expected absolute volume change (EAVC). The models were seeded with 10 Sobol-sampled trials before active learning. We evaluate performance by evaluating F1 score of how well the model identifies the sublevel set (with 75%
detection probability) at each iteration. F1 score was chosen rather than accuracy because the ground-truth sublevel set only covers a small fraction of the domain, leading to label imbalance typical of real visual psychophysics experiments. Figure 3 shows active learning performance with four different combinations of models (pseudo-data vs. mixed likelihood) and acquisition functions (GlobalMI vs. EAVC). For both acquisition functions, imposing domain knowledge via the mixed likelihood framework is better for active learning than the heuristic pseudo data approach.

## 4.5. Mixed-Reality Video Passthrough

We now evaluate the efficacy of the mixed likelihood training for imposing domain knowledge on non-Gaussian observations in a real-world experiment. The real experiment was to measure visual sensitivity to video passthrough camera displacements in a head-mounted display (HMD). Video passthrough is a feature that uses cameras on the outside of an HMD to enable interacting with the world while wearing the device. The passthrough cameras are physically displaced from the user's actual eye position, resulting in inaccurate 3D perception and erroneous motion of the world when the user moves (Biocca & Rolland, 1998). There are three parameters of interest in this experiment: (a)
differences between camera separation and user interpupillary distance; (b) camera z-axis offsets from the user's eyes due to headset thickness; and (c) passthrough latency, which results in delays between when an image is captured by the cameras and when it is actually seen by a user. A vision scientist helped us set up a psychophysical experiment to identify the combinations of these three parameters that cannot be reliably differentiated from rendering at the user's actual eye position. We used virtual content and render cameras in order to adaptively explore camera placement relative to viewer eye position, as opposed to real passthrough cameras which would require making changes to physical hardware. They consented to data collection, and collected 900 Bernoulli observations which we analyze in this section. We fit a zero-centered parametric ellipsoid1 with a sigmoid link function on the collected data:

## Y(X) ∼ Bernoullis(X ⊤Wx), X ∈ R 3,

where s(·) is a sigmoid function and W ∈ S
3++ a symmetric positive definite matrix. This fitted parametric ellipsoid was treated as the ground truth for our model evaluation here. We ran active learning to identify the sublevel set of the 75%
detection probability: {x ∈ R
3: x
⊤Wx ≤ s
−1(0.75)}.

Note that this is a slightly misspecified problem—the data generation process is not exactly the same as the model assumption, as the parametric ellipsiod used a sigmoid link function, not the normal CDF link function used by the Bernoulli likelihood. A total of 21 constraints are imposed by mixed likelihood training: one constraint at the origin x = 0 where the output probability is 50%, and 20 constraints sampled from the domain boundary where the output probability is close to 100%. We again add Bernoulli pseudo data for the standard 1A quarter ellipsoid to be precise, since headset thickness and latency have to be nonnegative.

variational GP to serve as a baseline. More details of this problem are deferred to §A.1. In Figure 3 the last panel "visual sensitivity," we plot F1 scores across active learning queries. Mixed likelihood constraints led to higher F1 scores than pseudo data, especially when using the GlobalMI acquisition function, and in early iterations for EAVC.

## 5. **Combining User Confidence And Preference**

Many vision studies investigate detection thresholds and can be modeled with Bernoulli level set estimation because there is a clearly-defined correct/incorrect user response. However, in many other domains of research related to human perception researchers often evaluate subjective user preferences with psychopysics where there may not be an objectively correct response (Maloney & Yang, 2003; Bartoshuk, 1978; Jones & Tan, 2012). These problems are ideal candidates for preference learning techniques and we next explore how Likert scale survey responses can be used to improve studies in this domain.

## 5.1. Preference Learning And The Likert Likelihood

Given two stimuli x1, x2 ∈ R
d, the preference likelihood models the probability that x1 is preferred to x2 as

$$\operatorname*{Pr}(\mathbf{x}_{1}\succeq\mathbf{x}_{2})=\Phi(f(\mathbf{x}_{1})-f(\mathbf{x}_{2})),$$

where Φ(·) is the normal CDF. The difference of GPs, f(x1) − f(x2), is itself a GP, hence it is common to directly learn the difference as a GP with a special preference kernel (Houlsby et al., 2011). Then, fitting GPs on preference data is reduced to GP classification.

Suppose that in addition to asking whether x1 is preferred to x2 we also ask for a rating of the strength of preference. For

ground truth w/o Likert scale w/ Likert scale positive data negative data Likert scale 0 Likert scale 1 Likert scale 2 0.0 0.2 0.4 0.6 0.8 1.0 1 2 3 4 5 subject ids 0.08 0.10 0.12 0.14 0.16 br ier sc or e w/o ls w/ ls w/o ls w/ ls 1 2 3 4 5 subject ids 0.80 0.85 0.90 f1 sc or e

example, participants can be asked to rate their confidence on a scale of 1 to 10 after indicating which of the two stimuli is the preferred choice. This differentiates between situations where x1 is strongly preferred to x2 or only marginally better. With mixed likelihood training, we can model Likert scale survey responses alongside preference observations to more effectively learn a user's underlying latent function. We propose a novel likelihood for Likert scale survey responses. Let y ∈ {1, 2, · · · , l} be the the Likert scale response, with l ∈ N the number of options. We call the absolute value of the difference |f(x1) − f(x2)| the preference strength. Intuitively, high strength preference is correlated with larger Likert scale response. We define cut points,

$$0=c_{1}\leq c_{2}\leq,\cdots,\leq c_{l}<c_{l+1}=\infty,$$

that divide all nonnegative numbers into l intervals,

$$I_{i}=[c_{i},c_{i+1}),\quad i=1,2,\cdots,l.$$

Each interval Ii corresponds to a response option. We construct the likelihood so that the probability of observing y =
i is highest when the preference strength |f(x1) − f(x2)| falls into the corresponding interval. We also wish for the likelihood of y = i to be negatively correlated with the distance from the preference strength to the corresponding interval. Hence, we propose the following Likert scale likelihood:

$$\Pr(y=i\mid f(\mathbf{x}_{1}),f(\mathbf{x}_{2}))={\frac{\exp(-\operatorname{dist}_{i})}{\sum_{j=1}^{l}\exp(-\operatorname{dist}_{j})}}.\quad(2)$$

Here the distance is taken as the minimum possible distance to any point in the interval:

$$\operatorname{dist}_{i}=\operatorname*{min}_{a\in I_{i}}\Bigl\vert\vert f(\mathbf{x}_{1})-f(\mathbf{x}_{2})\vert-a\Bigr\vert,\quad i=1,2,\cdots,l.$$

The cut points ciin the likelihood are learned automatically by maximizing the ELBO in (1) along with the variational 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 parameters on the training data. We add constraints on the cut points to avoid overfitting and a lapse rate parameter to damp the probability (2) with a uniform distribution for enhanced robustness (see §B). Below we apply this Likert scale likelihood on a synthetic and real-world experiments. One of the challenges of mixing different likelihoods is to ensure they are compatible with each other, in the sense that they can operate on the same latent function. For example, ordinal likelihoods are common for ordinal data, including the Likert scale confidence ratings. However, they cannot be mixed directly with preference observations because they treat the latent function as an "ordinal" strength ranging over entire real numbers. On the other hand, it is the absolute value of the latent difference f(x1) − f(x2) that represents the preference strength. This necessitates our development of a Likert scale likelihood here.

## 5.2. Synthetic Experiments

In this section, we present experiments on a synthetic latent function. The ground truth function is a univariate identity function f(x) = x, and the preference observations are Bernoulli observations with

$$\operatorname*{Pr}(x_{1}\succeq x_{2})=\Phi(x_{1}-x_{2}).$$

The Likert scale responses on a scale of 0 to 2 are generated deterministically based on which interval the preference strength |f(x1) − f(x2)| falls into: [0, 0.5], [0.5, 1], [1, ∞).

Figure 4 shows GP fits on synthetic data generated from the ground truth latent function. We observe that the GP trained by mixing Bernoulli and Likert scale likelihoods learns the ground truth latent function more accurately, particularly at the corners where the standard GP confidence is too low.

## 5.3. Learning Haptic Preferences

Haptics is the study of how humans perceive the world through the sensation of touch (Kappers & Bergmann Tiest, 2013). We obtained haptic experimental data from Driller (2024, Chapter 5) by contacting the authors. In this study

50 100 150 200 number of data points 0.78 0.80 0.82 0.84 0.86 w/o likert scale w/ likert scale 50 100 150 200 number of data points 0.09 0.10 0.11 0.12 0.13 0.14 brie r score f1 score w/o likert scale w/ likert scale
five participants were presented 50 pairs of 3D printed surfaces with different degrees of microscale surface roughness and material elasticity. For each pair, the user touched both surfaces, reported which felt rougher, and rated the confidence of their judgement on a scale from one to nine. We split each participant's dataset into 20 training trials and 30 test trials (see §C for additional ablation studies). We train preference GPs with and without confidence ratings on the training set and evaluate them on the test set. Because class imbalance is not a concern with preference data, in addition to the F1 score we also measure the Brier score, which is the mean squared error between the predicted and actual probabilities (Brier, 1950). We repeat the process 100 times with different training/test splits. The average Brier scores and F1 scores with one standard errors are shown in Figure 5. Mixed likelihood GPs trained with the Likert scale likelihood achieved lower Brier scores and higher F1 scores for all subjects except for subject \#2. We observe that subject \#2's Likert responses were predominately confident ratings: 48% of their confidence ratings were 9 (the highest rating), 84% of their confidence ratings them were equal to or above 7, and no confidence ratings below 3 were reported. As a result, the Likert scale likelihood struggled to learn the cut points for subject \#2. In contrast, other subjects' confidence ratings were spread more evenly (see Figure 9). This suggests that a calibration stage could be important for using survey data in preference learning.

## 5.4. Robot Gait Optimization

Preference learning can also be applied to robotics to align robot behavior with desired outcomes (e.g., Tucker et al., 2020). We consider the robot gait optimization task from Shvartsman et al. (2024), who use human preferences to determine optimal motion control parameters that yield naturalistic robot gait. In their experiment, two videos of quadruped robots are played side by side, and study subjects choose the video with their preferred gait (see Figure 6).

385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 We repeated their study protocol, but additionally collected confidence ratings on a scale of 1 to 3 on each trial. One of authors of this paper participated in this experiment and collected 472 preference responses and confidence ratings. In Figure 7, we report Brier and F1 scores on test sets averaged over 100 different train/test splits. GPs that mixed the Likert scale likelihood with the preference likelihood consistently improved both the Brier score and the F1 score.

## 6. Related Work

From the view of probabilistic graphical models, mixed likelihood training conditions the latent variables (the latent GP function values and inducing values) on different types of observations using variational inference. In fact, variational GPs have been *implicitly* trained with mixed likelihoods in several applications throughout the years. For example, the heteroscedastic Gaussian likelihood, which assigns different noise levels to different data points, is technically a form of mixed likelihood training (Kersting et al., 2007; Lazaro-Gredilla & Titsias ´ , 2011; Binois et al., 2018). Another example is the OR-channel likelihood for modeling multi-tone response in audiometry (Gardner et al., 2015b). Different parameters (number of tones) of the OR-channel likelihood correspond essentially to different likelihoods, and thus it is also an example mixing likelihoods, though using a Laplace approximation and not variational inference. Importantly, these past examples are all mixtures of likelihoods from the same family. We step further in this paper and introduce a framework for mixing vastly different likelihoods. Recently, Shvartsman et al. (2024) have developed response time GPs that are jointly trained on human choices and response time, but their approach is a single likelihood modeling the joint distribution of both human choices and response time. Their likelihood is based on an approximation of the diffusion decision model designed by domain experts, which is not easily generalizable to other likelihoods or data types. Murray & Kjellstrom¨ (2018) mixed likelihoods specifically for unsupervised representation learning in GP latent value models. Our work provides a general approach that, as we show, solves many problems arising from experimental designs and preference learning. Mixed likelihood variational training is closely related to multitask GPs, where the goal is to learn *multiple* correlated latent functions (e.g., Bonilla et al., 2007a;b). Inter-task correlations can be encoded either via Kronecker kernel matrices or, more commonly for variational GPs, using the linear model of coregionalization (LMC) (Alvarez et al., 2012), in which the prediction for each task is a linear combination of multiple variational GPs. Prior work has constructed multi-task models with different likelihoods, each of which is associated with several latent functions, based on both Kronecker and LMC models (Pourmohamad & Lee, 2016; Moreno-Munoz et al. ˜ , 2018). Our mixed likelihood approach learns a single, shared latent function from multiple data types. Merging all information into a single GP is necessary for the real-world applications we demonstrate here, such as enforcing domain knowledge constraints. In §4 we developed domain knowledge constraints as a use case of mixed likelihood modeling. There are other ways to enforce constraints in variational GPs. Recently, Cosier et al. (2024) proposed enforcing constraints with a set of fixed inducing points with fixed inducing values. Compared to this approach, mixed likelihood training has two advantages. First, mixed likelihood training supports soft constraints by tuning the Gaussian likelihood noise, whereas fixing inducing values generally enforces hard constraints. Second, mixed likelihood training is easier to implement, as it is compatible with all off-the-shelf GP variational inference implementations. The only change needed is the training objective. In contrast, fixing inducing values requires a custom implementation of GP variational inference, more specifically a custom whitening strategy.

## 7. Discussion

We have shown that variational GPs can be trained with mixed likelihoods to incorporate multiple types of data in human-in-the-loop experiments. We demonstrated two main applications of mixed likelihood training in this paper: (a) imposing soft constraints on the latent function into GPs by mixing Gaussian likelihoods with Bernoulli likelihoods, and using the constrained variational GPs to accelerate active learning for Bernoulli level set estimation; and (b) leveraging Likert scale confidence ratings by mixing with a Likert scale likelihood to improve preference learning. A few extensions to our framework are possible. Response time in human-in-the-loop experiments is typically correlated with preference strength, i.e., longer the response time often implies more uncertainty. With an appropriate likelihood for response times, which naturally shares the same latent function with preference observations, mixed likelihood training could significantly simplify the expert-designed likelihood of Shvartsman et al. (2024) based on the diffusion decision model.

Confidence ratings could also be used in active learning with mixed likelihood training, though this comes with challenges. As discussed in §5.3, some participants produce low-quality ratings, which may require calibration (on the fly) before feeding them into the model. Furthermore, asking for confidence ratings increases the cognitive load on participants and may increase the experiment time per trial. It may be best to collect confidence ratings only in the early stage of active learning, when model uncertainty is highest.

## Impact Statement

This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

## References

Alvarez, M. A., Rosasco, L., Lawrence, N. D., et al. Kernels for vector-valued functions: A review. Foundations and Trends® in Machine Learning, 4(3):195–266, 2012. 6 Bartoshuk, L. M. The psychophysics of taste. *The American* Journal of Clinical Nutrition, 31(6):1068–1077, 1978. 5 Binois, M., Gramacy, R. B., and Ludkovski, M. Practical heteroscedastic Gaussian process modeling for large simulation experiments. *Journal of Computational and* Graphical Statistics, 27(4):808–821, 2018. 6 Biocca, F. A. and Rolland, J. P. Virtual eyes can rearrange your body: Adaptation to visual displacement in seethrough, head-mounted displays. *Presence*, 7(3):262–277, 1998. 4.5 Blei, D. M., Kucukelbir, A., and McAuliffe, J. D. Variational inference: A review for statisticians. Journal of the American statistical Association, 112(518):859–877, 2017. 2 Bonilla, E. V., Agakov, F. V., and Williams, C. K. I. Kernel multi-task learning using task-specific features. In Meila, M. and Shen, X. (eds.), Proceedings of the Eleventh International Conference on Artificial Intelligence and Statistics, volume 2 of Proceedings of Machine Learning Research, pp. 43–50, San Juan, Puerto Rico, 21–24 Mar 2007a. PMLR. 6 Bonilla, E. V., Chai, K., and Williams, C. Multi-task Gaussian process prediction. In Platt, J., Koller, D., Singer, Y.,
and Roweis, S. (eds.), Advances in Neural Information Processing Systems, volume 20. Curran Associates, Inc., 2007b. 6 Brier, G. W. Verification of forecasts expressed in terms of probability. *Monthly weather review*, 78(1):1–3, 1950. 5.3 Chu, W. and Ghahramani, Z. Preference learning with Gaussian processes. In Proceedings of the 22nd International Conference on Machine Learning, ICML, pp. 137–144, 2005. 1 Cosier, L. C., Iordan, R., Zwane, S. N. T., Franzese, G.,
Wilson, J. T., Deisenroth, M., Terenin, A., and Bekiroglu, Y. A unifying variational framework for Gaussian process motion planning. In Dasgupta, S., Mandt, S., and Li, Y.

(eds.), *Proceedings of The 27th International Conference* on Artificial Intelligence and Statistics, volume 238 of Proceedings of Machine Learning Research, pp. 1315– 1323. PMLR, 02–04 May 2024. 6 Driller, K. K. From Cue to Construct: Cues, Mechanisms, and Stability in Haptic Perception. PhD thesis, Delft University of Technology, 2024. 5.3 Gardner, J. R., Malkomes, G., Garnett, R., Weinberger, K. Q., Barbour, D., and Cunningham, J. P. Bayesian active model selection with an application to automated audiometry. In Advances in Neural Information Processing Systems 28, pp. 2386–2394, 2015a. 1 Gardner, J. R., Song, X., Weinberger, K. Q., Barbour, D. L.,
and Cunningham, J. P. Psychophysical detection testing with Bayesian active learning. In Proceedings of the 31st Conference on Uncertainty in Artificial Intelligence, UAI, pp. 286–295, 2015b. 1, 6 Gramacy, R. B. *Surrogates: Gaussian Process Modeling,*
Design and Optimization for the Applied Sciences. Chapman Hall/CRC, Boca Raton, Florida, 2020. 1 Guan, P., Mercier, O., Shvartsman, M., and Lanman, D.

Perceptual requirements for eye-tracked distortion correction in VR. In ACM SIGGRAPH 2022 Conference Proceedings, SIGGRAPH, 2022. 1 Guan, P., Penner, E., Hegland, J., Letham, B., and Lanman, D. Perceptual requirements for world-locked rendering in AR and VR. In SIGGRAPH Asia 2023 Conference Papers, SA, 2023. 1 Hensman, J., Fusi, N., and Lawrence, N. D. Gaussian processes for big data. In Proceedings of the Twenty- Ninth Conference on Uncertainty in Artificial Intelligence, 2013. 1, 2 Hensman, J., Matthews, A. G. d. G., and Ghahramani, Z.

Scalable variational Gaussian process classification. In Proceedings of The 18th International Conference on Artificial Intelligence and Statistics, AISTATS, pp. 351– 360, 2015. 1, 2 Houlsby, N., Huszar, F., Ghahramani, Z., and Lengyel, M. ´
Bayesian active learning for classification and preference learning. *arXiv preprint arXiv:1112.5745*, 2011. 5.1 Houlsby, N., Huszar, F., Ghahramani, Z., and Hernandez- ´
lobato, J. Collaborative Gaussian processes for preference learning. In *Advances in Neural Information Processing* Systems 25, 2012. 1 Jones, L. A. and Tan, H. Z. Application of psychophysical techniques to haptic research. IEEE transactions on haptics, 6(3):268–284, 2012. 5 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 Kappers, A. M. and Bergmann Tiest, W. M. Haptic perception. *Wiley Interdisciplinary Reviews: Cognitive Science*, 4(4):357–374, 2013. 5.3 Kersting, K., Plagemann, C., Pfaff, P., and Burgard, W.

Most likely heteroscedastic Gaussian process regression. In Proceedings of the 24th International Conference on Machine Learning, ICML, pp. 393–400, 2007. 6 Kupcsik, A., Hsu, D., and Lee, W. S. *Learning Dynamic* Robot-to-Human Object Handover from Human Feedback, pp. 161–176. Springer International Publishing, Cham, 2018. 1 Kuss, M. and Rasmussen, C. Assessing approximations for Gaussian process classification. In Advances in Neural Information Processing Systems 18, pp. 699–706, 2005. 1 Kwak, Y., Penner, E., Wang, X., Saeedpour-Parizi, M. R.,
Mercier, O., Wu, X., Murdison, S., and Guan, P. Saccadecontingent rendering. In ACM SIGGRAPH 2024 Conference Papers, pp. 1–9, 2024. 1 Lazaro-Gredilla, M. and Titsias, M. K. Variational het- ´
eroscedastic Gaussian process regression. In *Proceedings* of the 28th International Conference on Machine Learning, ICML, pp. 841–848, 2011. 6 Letham, B., Guan, P., Tymms, C., Bakshy, E., and Shvartsman, M. Look-ahead acquisition functions for Bernoulli level set estimation. In Proceedings of The 25th International Conference on Artificial Intelligence and Statistics, AISTATS, pp. 8493–8513, 2022. 4.1, 4.2, 4.4, A
Maloney, L. T. and Yang, J. N. Maximum likelihood difference scaling. *Journal of Vision*, 3(8):5–5, 2003. 5 McKee, S. P., Klein, S. A., and Teller, D. Y. Statistical properties of forced-choice psychometric functions: Implications of probit analysis. *Perception & psychophysics*, 37(4):286–298, 1985. 4.1 Moreno-Munoz, P., Art ˜ es, A., and ´ Alvarez, M. Hetero- ´
geneous multi-output Gaussian process prediction. In Advances in Neural Information Processing Systems 31, 2018. 6 Murray, S. and Kjellstrom, H. Mixed likelihood Gaus- ¨
sian process latent variable model. arXiv preprint arXiv:1811.07627, 2018. 6 Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C.,
Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M.,
Askell, A., Welinder, P., Christiano, P. F., Leike, J., and Lowe, R. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems 35, volume 35, pp. 27730–27744, 2022. 1 Owen, L., Browder, J., Letham, B., Stocek, G., Tymms, C., and Shvartsman, M. Adaptive nonparametric psychophysics, 2021. URL https://arxiv.org/abs/ 2104.09549. 1 Palmer, S. E. *Vision science: Photons to phenomenology*.

MIT press, 1999. 4.1 Pourmohamad, T. and Lee, H. K. H. Multivariate stochastic process models for correlated responses of mixed type. Bayesian Analysis, 11(3):797–820, 2016. 6 Shvartsman, M., Letham, B., Bakshy, E., and Keeley, S.

Response time improves Gaussian process models for perception and preferences. In Proceedings of the 40th Conference on Uncertainty in Artificial Intelligence, UAI, pp. 3211–3226, 2024. 5.4, 6, 7 Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R.,
Voss, C., Radford, A., Amodei, D., and Christiano, P. F. Learning to summarize with human feedback. In Advances in Neural Information Processing Systems 33, pp. 3008–3021, 2020. 1 Titsias, M. Variational learning of inducing variables in sparse Gaussian processes. In van Dyk, D. and Welling, M. (eds.), Proceedings of the Twelfth International Conference on Artificial Intelligence and Statistics, volume 5 of *Proceedings of Machine Learning Research*, pp. 567– 574, Hilton Clearwater Beach Resort, Clearwater Beach, Florida USA, 16–18 Apr 2009. PMLR. 2 Tucker, M., Novoseller, E., Kann, C., Sui, Y., Yue, Y., Burdick, J. W., and Ames, A. D. Preference-based learning for exoskeleton gait optimization. In Proceedings of the IEEE International Conference on Robotics and Automation, ICRA, pp. 2351–2357, 2020. 1, 5.4 Ulrich, R. and Miller, J. Threshold estimation in twoalternative forced-choice (2afc) tasks: The spearmankarber method. ¨ *Perception & Psychophysics*, 66(3):517–
533, 2004. 4.1 Williams, C. K. and Rasmussen, C. E. Gaussian processes for machine learning, volume 2. MIT press Cambridge, MA, 2006. 1 Wolfe, J. M., Kluender, K. R., Levi, D. M., Bartoshuk, L. M.,
Herz, R. S., Klatzky, R. L., Lederman, S. J., and Merfeld, D. M. *Sensation & perception*. Sinauer Sunderland, MA, 2006. 4.1

## A. Experimental Details Of Bernoulli Level Set Estimation

Inudcing Points. All inducing points are fixed, not learned in hyperparameter optimization, because otherwise the GP models overfit easily on the problems we consider. Two types of inducing points are used in the GP models: (a) 100 Sobol samples from the domain; and (b) an inducing point at every constraint location. Both the standard variational GP and the mixed likelihood-trained GP use the same set of inducing points. Crucially, we found the additional inducing points at constraint locations are especially important for mixed likelihood-trained GP. Without these inducing points, the mixed likelihood-trained GP tends to be inflexible.

Evaluation Metric. Different from the prior work Letham et al. (2022), which primarily use the Brier score as the evaluation metric, we use the F1 score to evaluate active learning performance for Bernoulli level set estimation.

The Brier score for level set estimation used by Letham et al. (2022) is defined as

$${\frac{1}{n}}\sum_{i=1}^{n}(p_{i}-o_{i})^{2},$$

where pi ∈ [0, 1] is the model's probability prediction on a set Sobol samples in the domain and oi ∈ {0, 1} is the ground truth indicating whether each xi belongs to the sublevel set {x ∈ R
d: f(x) ≤ γ} for the threshold γ ∈ R. Here, the model's probability prediction for sublevel set is

$$p_{i}=\operatorname*{Pr}(f(\mathbf{x}_{i})\leq\gamma)=\Phi{\bigg(}{\frac{\gamma-\mu_{\mathcal{D}}(\mathbf{x}_{i})}{\sigma_{\mathcal{D}}(\mathbf{x}_{i})}}{\bigg)},$$

where µD is the posterior mean and σD is the posterior standard deviation. It is often the case, especially in high dimensions, that only a small portion of the domain will have values below the target level set, i.e., the ground truth sublevel set is a tiny fraction of the entire domain. As a result, the vast majority of the ground truths oi are 0's, which results in label imbalance. For some high dimensional problems, we observe that the Brier score is not reliable due the label imbalance issue, i.e., predicting constant zero might even achieve lower Brier scores than active learning in some cases. Thus, we opt for the F1 scores evaluated on a set of Sobol samples in the domain for Bernoulli level set estimation. Let V ⊆ R
d be the ground truth sublevel set and Vb ⊆ R
d be the estimate. In the context of level set estimation, the precision and recall have clear geometric interpretations:

$${\mathrm{precision}}={\frac{|V\cap{\hat{V}}|}{|{\hat{V}}|}},{\mathrm{~recall}}={\frac{|V\cap{\hat{V}}|}{|V|}}.$$

We use 106 Sobol samples in the domain to estimate the F1 scores.

550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 Level Set Estimation Threshold. Every Bernoulli level set estimation problem in this paper aims for a target sublevel set with a threshold of 75% in the probability space. Equivalently, this is the same as estimating the Φ
−1(0.75) sublevel in the latent function value space

$$\{\mathbf{x}\in\mathbb{R}^{d}:f(\mathbf{x})\leq\Phi^{-1}(0.75)\}.$$

The only exception is the parametric ellipsoid in the visual sensitivity task (see §4), where the target sublevel set in the latent function space is

$$\{\mathbf{x}\in\mathbb{R}^{d}:f(\mathbf{x})\leq s^{-1}(0.75)\},\quad s(z)={\frac{1}{1+\exp(-z)}}$$
.
This difference is because the parametric ellipsiod uses a sigmoid link function, not the normal CDF.

Additional Details. Global look-ahead acquisition functions like GlobalMI and EAVC proposed by Letham et al. (2022) require a set of global reference points. Those reference points are Sobol samples in the domain for estimating global changes in mutual information and sublevel set volumes. We use 104 Sobol samples as reference points. Active learning starts with 10 initial Sobol samples.

## A.1. Objectives

Psychometric Discrimination. This function is defined as

$$f(x_{1},x_{2})={\frac{1+x_{2}}{0.05+0.4x_{1}^{2}(0.2)}}$$
$ \overline{\frac{2x_1-1)^2}{}}$
(0.2x1 − 1)2
on a domain (x1, x2) ∈ [−1, 1]2. It is clear that the Bernoulli probabilities Φ(f(x)) are exactly 50% on the line

$=\;-1$  . 
{(x1, x2) : −1 ≤ x1 ≤ 1, x2 = −1},
where and the domain is [−30, 50] × [0, 60] × [0, 75] with each axis being IPD offsets, camera z-axis errors, and passthrough latency. Note that W is symmetric and positive definite. The weight matrix W is estimated by maximum likelihood on the collected human data with convex optimization. Note that the link function for this objective is a sigmoid function s(·) = 1/(1 + exp(− ·)), not a normal CDF. We impose a constraint at the origin x = 0 and sample 5 Sobol samples as constraint locations from each of the following faces 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659

## B. Additional Details Of The Likert Scale Likelihood

A constraint ci+1 − ci ≤ 2 is enforced for each pair of adjacent cut points to avoid overfitting. Note that Φ(2) ≈ 0.98. This constraint enforces that the preference probability within the same Likert scale response is no greater than 98%, a natural A total of 20 constraints are added: 10 points on the line x2 = −1 and another 10 points on the line x2 = +1. The constraint target values are set to the ground truth latent function values.

Norm Ball. This function is defined as
$f(\mathbf{x})=2\|\mathbf{x}\|_2$
on a domain x ∈ [−1, 1]d. In the main paper, we have used d = 2 and d = 4. Note that there is a multiplication coefficient
2. The factor 2 makes sure that the function grows fast enough so that the Bernoulli probability Φ(f(x)) is close to 100% on the domain boundary. We impose a constraint at the origin x = 0 and additionally sample 5 Sobol samples as constraint
locations from every hypercube face. The constraint target values are set to the ground truth latent function values.
Parametric Ellipsoid. This is a 3D function defined as
$$f(\mathbf{x})=\mathbf{x}^{\top}\mathbf{W}\mathbf{x},$$

$$\begin{array}{r l r l}{{(+0.00345447}&{{}-0.00344695}&{{}-0.00144475)}\\ {-0.00344695}&{{}+0.00556409}&{{}+0.00252343}\\ {-0.00144475}&{{}+0.00252343}&{{}+0.00466492}\end{array}$$

$$\mathbf{W}=$$

$F_{0}=\{(x_{1},x_{2},x_{3}):x_{1}=-30,\ 0\leq x_{2}\leq60,\ 0\leq x_{3}\leq75\},$
F1 = {(x1, x2, x3) : x1 = 50, 0 ≤ x2 ≤ 60, 0 ≤ x3 ≤ 75}, F2 = {(x1, x2, x3) : −30 ≤ x1 ≤ 50, x2 = 60, 0 ≤ x3 ≤ 75}, F3 = {(x1, x2, x3) : −30 ≤ x1 ≤ 50, 0 ≤ x2 ≤ 60, x3 = 75},
which are four faces that do not contain the origin. The constraint target value at a location x is set to Φ
−1(min{s(x), 0.999}).

Namely, we first evaluate the ground truth probability s(x), then truncate it by 99.9%, and then covert it into the corresponding latent function value as if the link function was the normal CDF Φ(·). The ground truth latent function values cannot be directly used in mixed likelihood training because of the link functions mismatch with each other. Thus, the conversion is necessary. The truncation is also necessary, because otherwise Φ
−1(s(x)) might be infinity due to floating point overflow.

Note that this is an example that the "believed" latent function value, not the ground truth latent function value, is used in constraints.

$$\{(x_{1},x_{2}):-1\leq x_{1}\leq1,\ x_{2}=+1\}.$$

and the Bernoulli probabilities are close to 100% on the line assumption on the Likert scale response. With limited number of data, the cut points and the latent function may not be learned accurately. Thus, it is important to add constraints on the cut points to avoid overfitting.

In addition, we introduce a lapse rate parameter to damp the Likert scale likelihood. Let p1, p2, · · · , pl be the probabilities produced by the Likert scale likelihood (2). Then, we damp the probabilities with a mixutre of uniform distribution

$$p_{i}^{\mathrm{\tiny~damp}}=(1-\lambda)p_{i}+\lambda\cdot{\frac{1}{l}},$$

where λ ≥ 0 is the lapse rate parameter. The damped probability is used to train variational GPs in the experiments, and we use a lapse rate of λ = 0.1 throughout. Intuitively, the damped Likert scale likelihood is more robust because it prevents extremely small probabilities, particularly when the number of data is limited.

## C. Learning Haptic Preferences

The raw Likert scale confidence ratings on a scale of 1 to 9 are mapped to a scale of 0 to 2:

$$1,2,3\mapsto0,\quad4,5,6\mapsto1$$
1, 2, 3 7→ 0, 4, 5, 6 7→ 1, 7, 8, 9 7→ 2,
which is primarily due to the easy of programming. The Brier score shown in Figure 5 is defined as

$${\frac{1}{n}}\sum_{i=1}^{n}(p_{i}-o_{i})^{2},$$
$\Rightarrow$ 2. 
$$({\mathfrak{I}})$$
2, (3)
where oi ∈ {0, 1} indicating which stimulus (xi1 or xi2) is preferred and piis the GP probability prediction

$$\Phi\!\left({\frac{\mu_{\mathcal{D}}(\mathbf{x}_{i1},\mathbf{x}_{i2})}{\sqrt{1+\sigma_{\mathcal{D}}^{2}(\mathbf{x}_{i1},\mathbf{x}_{i2})}}}\right),$$

660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714 where µD(xi1, xi2) is the posterior mean of the latent difference f(xi1) − f(xi2) conditioned on the training data D, and σ 2D(xi1, xi2) is the posterior variance of the latent difference f(xi1) − f(xi2) conditioned on the training data.

We use 100 Sobol samples as inducing points for both the standard variational GPs and mixed likelihood-trained GPs. In Figure 8, we present the Brier scores and F1 scores of GPs trained with varying number of data points. Since we only have 50 data points per human subject, we only experiment with training sizes of 10, 20, and 30. Likert scale confidence ratings again improve both Brier scores and F1 scores except for subject \#2. As discussed in the main paper, it is most likely because subject \#2 Likert scale ratings are predominantly confident. In Figure 9, we plot all subjects' confidence rating histograms. There is a clear difference between the histogram of subject \#2 and the remaining subjects: subject \#2's ratings tend to be more confident then remain subjects.

## D. Robot Gait Optimization

The Brier score presented in Figure 7 is computed similarly as discussed in §C. We use 100 Sobol samples as inducing points for both the standard variational GPs and mixed likelihood-trained GPs. In Figure 10, we plot the distribution of confidence ratings in the data collected from the robot gait optimization task: 218 of them are 1; 176 of them are 2; and 78 of them are 3. The confidence ratings are generally well-balanced.

715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769 10 training data points 20 training data points 30 training data points w/o ls w/ ls w/o ls w/ ls w/o ls w/ ls 0.075 0.100 0.125 0.150 0.175 0.200 b ri er sc or e w/o ls w/ ls w/o ls w/ ls w/o ls w/ ls 0.75 0.80 0.85 0.90 0.95 f1 score 1 2 3 4 5 subject ids 1 2 3 4 5 subject ids 1 2 3 4 5 subject ids 1 2 3 4 5 6 7 8 9 confidence ratings 0 5 10 15 20 25subject 1 subject 2 subject 3 subject 4 subject 5 1 2 3 4 5 6 7 8 9 confidence ratings 1 2 3 4 5 6 7 8 9 confidence ratings 1 2 3 4 5 6 7 8 9 confidence ratings 1 2 3 4 5 6 7 8 9 confidence ratings
770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 810 811 812 813 814 815 816 817 818 819 820 821 822 823 824

robot gait confidence rating histogram 200 150 100 50 1 2 3 confidence ratings 0