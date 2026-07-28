011

014 015 016

018

024

026

034

036

038

# Mixed Likelihood Variational Gaussian Processes

Anonymous Authors<sup>1</sup>

# Abstract

Gaussian processes (GPs) are powerful models for human-in-the-loop experiments due to their flexibility and well-calibrated uncertainty. However, GPs modeling human responses typically ignore auxiliary information, including a priori domain expertise and non-task performance information like user confidence ratings. We propose mixed likelihood variational GPs to leverage auxiliary information, which combine multiple likelihoods in a single evidence lower bound to model multiple types of data. We demonstrate the benefits of mixing likelihoods in three realworld experiments with human participants. First, we use mixed likelihood training to impose prior knowledge constraints in GP classifiers, which accelerates active learning in a visual perception task where users are asked to identify geometric errors resulting from camera position errors in virtual reality. Second, we show that leveraging Likert scale confidence ratings by mixed likelihood training improves model fitting for haptic perception of surface roughness. Lastly, we show that Likert scale confidence ratings improve human preference learning in robot gait optimization. The modeling performance improvements found using our framework across this diverse set of applications illustrates the benefits of incorporating auxiliary information into active learning and preference learning by using mixed likelihoods to jointly model multiple inputs.

# 1. Introduction

Gaussian process (GPs) are indispensable models for many machine learning and AI applications [\(Williams & Ras](#page-9-0)[mussen,](#page-9-0) [2006\)](#page-9-0). As a Bayesian nonparametric model, it is favored for its well-calibrated uncertainty estimates and flexibility. When using a GP for regression with Gaussian

errors, i.e., a Gaussian likelihood, the posterior distribution is a multivariate normal whose mean and covariance can be computed analytically via the Kriging equations [\(Gramacy,](#page-8-0) [2020\)](#page-8-0). This analytic tractability does not hold for other types of observations, such as classification or preference data, but GP modeling in those settings can be done with variational approximation (e.g., [Hensman et al.,](#page-8-1) [2013;](#page-8-1) [2015\)](#page-8-2) or other approximate inference schemes [\(Kuss & Rasmussen,](#page-9-1) [2005\)](#page-9-1).

Human feedback, which is generally non-Gaussian, has recently become an important setting for GP modeling with applications including health screening [\(Gardner et al.,](#page-8-3) [2015a](#page-8-3)[;b\)](#page-8-4), AR/VR development [\(Guan et al.,](#page-8-5) [2022;](#page-8-5) [2023;](#page-8-6) [Kwak et al.,](#page-9-2) [2024\)](#page-9-2), and robot locomotion learning [\(Tucker](#page-9-3) [et al.,](#page-9-3) [2020\)](#page-9-3). In particular, preference learning has attracted a great deal of attention in recent years for its usefulness in large language model (LLM) training and reinforcement learning with human feedback (RLHF) [\(Stiennon et al.,](#page-9-4) [2020;](#page-9-4) [Ouyang et al.,](#page-9-5) [2022\)](#page-9-5). GPs are a natural fit for many preference learning problems [\(Chu & Ghahramani,](#page-8-7) [2005;](#page-8-7) [Houlsby et al.,](#page-8-8) [2012\)](#page-8-8), including for RLHF [\(Kupcsik et al.,](#page-9-6) [2018\)](#page-9-6). Due to their well calibrated uncertainty, GPs are especially useful in human-in-the-loop experiments where the human's time is valuable, as GPs can be used with active learning to increase trial efficiency [\(Owen et al.,](#page-9-7) [2021\)](#page-9-7).

In many non-Gaussian observation settings, multiple data of different types can be observed simultaneously. For example, in preference learning, we can solicit both preferences (binary comparison data) and strengths of preference (e.g., Likert scale survey data). Studies of human perception can measure whether or not a stimulus was perceived (binary classification data) while simultaneously recording response time (continuous but non-Gaussian data). Presumably, combining these different types of data into a single GP would help improve modeling performance. In addition, we may also have domain knowledge about the responses for some special inputs. For instance, in studies of human perception, a stimulus with no intensity cannot be perceived at all. This domain knowledge constraint, as we will show, can be also be considered as an additional observation type that we wish to include in the GP.

Here, we present a framework for joint GP modeling of multiple types of data and expand GP modeling to a rich new set of multi-data-type problems via the following contributions:

<sup>1</sup>Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>.

058

071

074

076

078

087 088

090 091

093 094

096

098

100

104

106

- We develop a novel evidence lower bound (ELBO) formulation that includes multiple likelihoods in the same variational approximation.
- We show that our mixed likelihood training can be used to encode domain knowledge in non-Gaussian settings, and thereby accelerate active learning.
- We develop both synthetic and real-world examples of mixed likelihood variational GPs improving model performance by incorporating auxiliary survey data into preference learning and human perception studies, also developing a new Likert likelihood.

## 2. Background

A Gaussian process (GP) <sup>f</sup> ∼ GP(0, k) defined by a kernel function k : R <sup>d</sup> × <sup>R</sup> <sup>d</sup> → <sup>R</sup> is a stochastic process whose function values <sup>f</sup> <sup>=</sup>f(X) on any training data <sup>X</sup> ∈ <sup>R</sup> n×d follow a joint Gaussian distribution <sup>f</sup> ∼ N (0, <sup>K</sup>f,<sup>f</sup>), where Kf,<sup>f</sup> = k(f(X), f(X)) is the covariance matrix.

A likelihood is a distribution that models the relation between the observed training labels y and the latent function values f. Thus, different types of data require different likelihoods. For regression, it is common to use a Gaussian likelihood

$$p(\mathbf{y} \mid \mathbf{f}) = \mathcal{N}(\mathbf{y}; \mathbf{f}, \sigma^2 \mathbf{I}),$$

where σ is the standard deviation of the label noise. For classification, it is common to use a Bernoulli likelihood

$$p(\mathbf{y} \mid \mathbf{f}) = \text{Bernoulli}(\Phi(\mathbf{f})).$$

On the test data X<sup>∗</sup> , the GP prediction is the posterior conditioned on the training labels p(f ∗ | <sup>y</sup>). For a Gaussian likelihood, the posterior distribution is also Gaussian with a closed-form expressions for its mean and covariance:

$$\mathbb{E}[\mathbf{f}^* \mid \mathbf{y}] = \mathbf{K}_{*,\mathbf{f}}(\mathbf{K}_{f,\mathbf{f}} + \sigma^2\mathbf{I})^{-1}\mathbf{y},$$

$$\mathbb{D}[\mathbf{f}^* \mid \mathbf{y}] = \mathbf{K}_{*,\mathbf{f}} - \mathbf{K}_{*,\mathbf{f}}(\mathbf{K}_{f,\mathbf{f}} + \sigma^2\mathbf{I})^{-1}\mathbf{K}_{f,\mathbf{f}}.$$

However, for non-Gaussian likelihoods, the exact posterior is almost always intractable, and thus needs approximation.

Variational GPs (e.g., [Titsias,](#page-9-8) [2009;](#page-9-8) [Hensman et al.,](#page-8-1) [2013;](#page-8-1) [2015\)](#page-8-2) approximate the exact posterior by inducing point approximation and variational inference [\(Blei et al.,](#page-8-9) [2017\)](#page-8-9). A set of inducing variables u is introduced, and the joint distribution factors as follows:

$$p(\mathbf{f}, \mathbf{f}^*, \mathbf{u}) = p(\mathbf{f} \mid \mathbf{u})p(\mathbf{f}^* \mid \mathbf{u})p(\mathbf{u}),$$

where each component admits the form

$$\begin{aligned}
p(\mathbf{u}) &= \mathcal{N}(\mathbf{u}; \mathbf{0}, \mathbf{K}_{\mathbf{u},\mathbf{u}}), \\
p(\mathbf{f} \mid \mathbf{u}) &= \mathcal{N}(\mathbf{f}; \mathbf{K}_{\mathbf{f},\mathbf{u}}\mathbf{K}_{\mathbf{u},\mathbf{u}}^{-1}\mathbf{u}, \mathbf{K}_{\mathbf{f},\mathbf{f}} - \mathbf{K}_{\mathbf{f},\mathbf{u}}\mathbf{K}_{\mathbf{u},\mathbf{u}}^{-1}\mathbf{K}_{\mathbf{u},\mathbf{f}}), \\
p(\mathbf{f}^* \mid \mathbf{u}) &= \mathcal{N}(\mathbf{f}^*; \mathbf{K}_{\mathbf{u},\mathbf{u}}\mathbf{K}_{\mathbf{u},\mathbf{u}}^{-1}\mathbf{u}, \mathbf{K}_{\mathbf{u},*} - \mathbf{K}_{\mathbf{u},\mathbf{u}}\mathbf{K}_{\mathbf{u},\mathbf{u}}^{-1}\mathbf{K}_{\mathbf{u},\mathbf{u}}).
\end{aligned}$$

Given the inducing values u, the latent function values on the training data and the test data are conditionally independent. As a result, the prediction on the test data is completely controlled by the inducing variables.

Inference in variational GPs is performed by maximizing the evidence lower bound (ELBO):

$$\max_{q(\mathbf{u})} \mathbb{E}_{q(\mathbf{f})} \log p(\mathbf{y} \mid \mathbf{f}) - D_{\text{KL}}(q(\mathbf{u}), p(\mathbf{u})),$$

where the variational distribution q(u) is usually restricted to a Gaussian family and q(f) = R <sup>p</sup>(<sup>f</sup> | <sup>u</sup>)q(u) d<sup>u</sup> is the marginalized variational distribution over the latent function values. The optimal variational distribution q(u) is then used to construct the approximate posterior

$$p(\mathbf{f}^* \mid \mathbf{y}) \approx \int p(\mathbf{f}^* \mid \mathbf{u}) q(\mathbf{u}) d\mathbf{u}.$$

## 3. Mixed Likelihood Variational Inference

Suppose we have T different types of data available

$$(\mathbf{X}^{(t)}, \mathbf{y}^{(t)}), \quad t = 1, 2, \dots, T,$$

where X(t) 's are training data locations and y (t) 's are labels of different types. For example, y (1) could be regression labels while y (2) are classification labels.

We assume that all labels are generated from the same latent function and that the labels y (t) are conditionally independent given the latent function values f (t) = f X(t) , across data types t. We then jointly model the different data types by training a single variational GP on all data using a combined ELBO. As before, we use a variational distribution <sup>q</sup>(u) to approximate the GP posterior. Let <sup>y</sup> <sup>=</sup> {<sup>y</sup> (t)} T t=1 now represent the complete collection of training labels across data types, and <sup>f</sup> <sup>=</sup> {<sup>f</sup> (t)} T <sup>t</sup>=1 their corresponding latent function values. Because of the conditional independence of the various observations, we have that

$$\log p(\mathbf{y} \mid \mathbf{f}) = \sum_{t=1}^T \log p_t(\mathbf{y}^{(t)} \mid \mathbf{f}^{(t)}).$$

Each type of data uses a different likelihood. For instance, <sup>p</sup>1(· | ·) is a Gaussian likelihood if <sup>y</sup> (1) are regression labels, and <sup>p</sup>2(· | ·) is a Bernoulli likelihood if <sup>y</sup> (2) are classification labels. The evidence term in the ELBO thus decomposes, and we can write a valid evidence lower-bound for mixed likelihoods as:

$$\sum_{t=1}^T \mathbb{E}_{q(\mathbf{f}^{(t)})} \log p_t(\mathbf{y}^{(t)} \mid \mathbf{f}^{(t)}) - D_{\text{KL}}(q(\mathbf{u}), p(\mathbf{u})). \quad (1)$$

![](_page_2_Picture_0.jpeg)

124

126

128

131

134

136

138

151

154

158

160

Figure 1: Illustrative depictions of perceived stereoscopic 3D distortions when render cameras are offset from the viewer's eyes. Left: Stereoscopic images rendered with cameras at the viewer's eyes have no 3D distortion. Center: Small camera offsets result in minimal perceived distortions, and participants cannot reliably identify any errors. Right: Large camera offsets result in obvious distortions and are easily recognized. Bernoulli Level Set Estimation: If participants are presented the left and middle options in randomized order and asked to select the distorted option, the probability of selecting the correct option will be close to 50%, i.e., at chance. On the other hand, when the left and right options are presented, the distorted option will be selected close to 100% of the time. We aim to identify the space of camera placement configurations such that the distortion detection probability of a participant is below 75%, a threshold that is often considered a just-detectable difference from zero error.

happens when T > 1, where multiple types of data are incorporated to learn the same latent function f. By virtue of being a valid evidence lower bound, maximizing [\(1\)](#page-1-0) yields a variational distribution approximating the GP posterior jointly conditioned on multiple types of data.

In the following sections [§4](#page-2-0) and [§5,](#page-5-0) we demonstrate that this simple idea can solve many problems arising from experimental design and preference learning.

## 4. Encoding Domain Knowledge Constraints in Active Learning

The mixed likelihood training scheme can be used to encode domain knowledge constraints into active learning problems with non-Gaussian data. We demonstrate this in level set estimation with Bernoulli observations, a problem setting with important applications in perception science.

#### 4.1. Visual Psychophysics

Understanding human vision and characterizing visual perception is challenging because human self-report is unreliable and individual decision-making criteria are highly variable. Vision scientists use forced-choice experimental paradigms to address these challeneges [\(Palmer,](#page-9-9) [1999;](#page-9-9) [Wolfe et al.,](#page-9-10) [2006\)](#page-9-10). Figure [1](#page-2-1) describes a psychophysical study design to determine how much render-camera offset is detectable to a person in a stereoscopic (i.e. 3D) display. Rather than asking participants if a particular camera offset looks acceptable, they are given a zero-offset option (the reference) and an option with some offset (the comparison), and asked identify which option has offset. Camera offset is varied over hundreds or thousands of trials with the aim of identifying the set of render-camera offsets that

cannot reliably be differentiated from the zero-offset reference. This is often taken as the offsets for which the probability of correctly selecting the comparison stimulus is below 75% [\(McKee et al.,](#page-9-11) [1985;](#page-9-11) [Ulrich & Miller,](#page-9-12) [2004\)](#page-9-12), and this problem can be formulated as Bernoulli level set estimation [\(Letham et al.,](#page-9-13) [2022\)](#page-9-13).

#### 4.2. Bernoulli Level Set Estimation

Given a black-box function f : R <sup>d</sup> → <sup>R</sup>, we are concerned with learning the sublevel set {<sup>x</sup> ∈ <sup>R</sup> d : <sup>f</sup>(x) ≤ <sup>γ</sup>} for some constant <sup>γ</sup> ∈ <sup>R</sup>. The black-box function <sup>f</sup> cannot be evaluated directly, but can be "probed" by Bernoulli observations. For any <sup>x</sup> ∈ <sup>R</sup> d , we may observe a random variable <sup>y</sup>(x) ∈ {0, <sup>1</sup>} where

$$y(\mathbf{x}) \sim \text{Bernoulli}(\Phi(f(\mathbf{x}))).$$

We iteratively query the latent function via the Bernoulli observations with the goal of learning the sublevel set. Active learning can be done using a variational GP classification model for f, and one of several acquisition functions for proposing new queries [\(Letham et al.,](#page-9-13) [2022\)](#page-9-13).

The visual psychophysics experiment paradigm described in [§4.1](#page-2-2) can be cast as Bernoulli level set estimation by taking x as a visual stimulus and f(x) as the perceptual intensity. We measure, via Bernoulli observations y(x), how well the human participant can differentiate between x and the reference stimulus xref, the perceptual intensity of which is zero. The sublevel set of f is the set of imperceptible stimuli which we wish to identify.

### 4.3. Encoding Prior Knowledge with Soft Constraints

Learning level sets with Bernoulli queries is challenging as Bernoulli observations are inherently noisy, especially for

168

171

174

176

178

194

196 197 198

200

204

206

208

211

214 215 216

218

detection probabilities close to 50%. We generally require many repeated trials in order to accurately estimate the latent response probabilities. Moreover, in the initial stage of active learning, the model is unable to distinguish between stimulus pairs that are obviously different (100% correct) and pairs that are almost identical (50% correct), but guessed correctly. Therefore, we wish to encode *a priori* knowledge about the experimental paradigm.

In the particular case of the visual perception task of [§4.1,](#page-2-2) we know that the detection probability should be exactly 50% when camera offset is 0, and should be close to 100% for maximum offset values. We expect this extra information will improve efficiency of human-in-the-loop experiments with active learning. Moreover, the domain knowledge constraints may add resiliency to prevent active learning from exploring easily-detectable areas away from the target level set in cases when outlier responses occur during the early stage of data collection, e.g., accidental misclicks when the camera offset is large, or several consecutive correct detections by chance even though the camera offset is small.

We encode this type of domain knowledge as constraints on the latent function by directly regressing against the target constraint values. In addition to the Bernoulli observations (X(1) , y (1)), we produce a set of regression labels (X(2) , y (2)) provided by domain experts that encode the known latent function values at special locations—for instance, the latent function value should be zero where the detection probability is known to be 50%. We enforce the soft constraints

$$f(\mathbf{x}_i^{(2)}) \approx y_i^{(2)}$$

by mixing Bernoulli with Gaussian likelihoods:

$$p\left(y_i^{(2)} \mid f(\mathbf{x}_i^{(2)})\right) = \mathcal{N}\left(y_i^{(2)}; f(\mathbf{x}_i^{(2)}), \sigma_i^2\right),$$

where σi's are fixed noise standard deviations that control the softness or hardness of the constraints. Intuitively, we use the Bernoulli likelihood to fit binary responses and the Gaussian likelihood to enforce the constraints.

Figure [2](#page-3-0) shows an example of enforcing constraints on an one dimensional objective by mixing Bernoulli and Gaussian likelihoods (σ <sup>2</sup> = 0.001). Mixed likelihood modeling leads to a significant reduction in posterior uncertainty, as regression labels at the points with known value provide stronger learning signals than the Bernoulli observations.

#### 4.4. Synthetic Experiments

We show that enforcing constraints effectively encodes domain knowledge and improves active learning for Bernoulli level set estimation.

We benchmark on three synthetic latent functions. The first is a synthetic two-dimensional psychometric discrimination

![](_page_3_Figure_0.jpeg)

Figure 2: Left: A standard variational GP fit to Bernoulli observations. Right: A mixed likelihood GP trained on the same data with two constraints f(0) = 0 and f(2) = 2. The mixed likelihood-trained GP has near-zero uncertainty at the constraint locations. The true latent function is <sup>1</sup>/<sup>2</sup> · <sup>x</sup> 2 .

objective from [Letham et al.](#page-9-13) [\(2022\)](#page-9-13). The others are scaled norm functions <sup>2</sup>∥x∥ in 2D and 4D respectively. All of these synthetic functions have locations where response probabilities equal exactly 50%, and locations where the response probability is close to 100% probabilities. We use mixed likelihood training to set constraints at a subset of these locations—see [§A.1](#page-11-0) for more details on the functions and the constraint locations.

We set the Gaussian likelihood noise, which determines strength of the constraint, according to the target value y<sup>i</sup> as: <sup>σ</sup><sup>i</sup> = 0.<sup>2</sup> · <sup>y</sup><sup>i</sup> + 0.1. Intuitively, this allows a 20% relative violation plus 0.1 absolute violation. For a constraint with y<sup>i</sup> = 0 (i.e. 50% response probability), this implies an *a priori* 95% credible interval on the response probability of [0.422, 0.578]. For y<sup>i</sup> = 2 (i.e. a 98% response probability), the credible interval for the response probability is [0.846, 0.999]. This policy was not extensively tuned, but produces a desirable behavior of maintaining soft constraints across the range of response probabilities. Our preliminary experiments indicate that enforcing constraints with strict tolerances, e.g., σ 2 <sup>i</sup> = 10−<sup>4</sup> , does not necessarily improve active learning performance as the variational GP tends to be rigid and adapts to new Bernoulli observations slowly. The GP spends most of its prediction capacity fitting the constraints, while spending less weight on Bernoulli observations. This is especially detrimental to look-ahead acquisition functions that depend on the change in posterior conditioning on virtual data.

A natural alternative approach that we use as a baseline is to add Bernoulli pseudo data to the standard single-likelihood GP, to push predictions at those locations towards the target constraint values. Pseudo data are added before active learning, and are added at the same locations used for constraints in the mixed likelihood model. Locations with a response

![](_page_4_Figure_0.jpeg)

234

236

238

254

256

258

260

264

266

268

271

Figure 3: F1 scores (higher is better) of active learning for sublevel set estimation using two different acquisition functions (GlobalMI and EAVC). Domain knowledge for each problem is added either by mixing Bernoulli and Gaussian likelihoods (solid lines) or by adding Bernoulli pseudo data (dotted lines). For both acquisition functions, incorporating domain knowledge with mixed likelihoods led to better F1 scores than the pseudo data approach. Shaded areas show one standard errors over 100 different random seeds.

probability of 50% were given 5 positive and 5 negative Bernoulli data, while those with response probability close to 100% were given a single positive data.

We run active learning for Bernoulli level set estimation with two competitive acquisition functions developed by [Letham](#page-9-13) [et al.](#page-9-13) [\(2022\)](#page-9-13): global mutual information (GlobalMI) and expected absolute volume change (EAVC). The models were seeded with 10 Sobol-sampled trials before active learning. We evaluate performance by evaluating F1 score of how well the model identifies the sublevel set (with 75% detection probability) at each iteration. F1 score was chosen rather than accuracy because the ground-truth sublevel set only covers a small fraction of the domain, leading to label imbalance typical of real visual psychophysics experiments.

Figure [3](#page-4-0) shows active learning performance with four different combinations of models (pseudo-data *vs*. mixed likelihood) and acquisition functions (GlobalMI *vs*. EAVC). For both acquisition functions, imposing domain knowledge via the mixed likelihood framework is better for active learning than the heuristic pseudo data approach.

#### 4.5. Mixed-Reality Video Passthrough

We now evaluate the efficacy of the mixed likelihood training for imposing domain knowledge on non-Gaussian observations in a real-world experiment. The real experiment was to measure visual sensitivity to video passthrough camera displacements in a head-mounted display (HMD). Video passthrough is a feature that uses cameras on the outside of an HMD to enable interacting with the world while wearing the device. The passthrough cameras are physically displaced from the user's actual eye position, resulting in inaccurate 3D perception and erroneous motion of the world when the user moves [\(Biocca & Rolland,](#page-8-10) [1998\)](#page-8-10).

differences between camera separation and user interpupillary distance; (b) camera z-axis offsets from the user's eyes due to headset thickness; and (c) passthrough latency, which results in delays between when an image is captured by the cameras and when it is actually seen by a user.

A vision scientist helped us set up a psychophysical experiment to identify the combinations of these three parameters that cannot be reliably differentiated from rendering at the user's actual eye position. We used virtual content and render cameras in order to adaptively explore camera placement relative to viewer eye position, as opposed to real passthrough cameras which would require making changes to physical hardware. They consented to data collection, and collected 900 Bernoulli observations which we analyze in this section. We fit a zero-centered parametric ellipsoid[<sup>1</sup>](#page-4-1) with a sigmoid link function on the collected data:

$$y(\mathbf{x}) \sim \text{Bernoulli}(s(\mathbf{x}^\top \mathbf{W} \mathbf{x})), \quad \mathbf{x} \in \mathbb{R}^3,$$

where <sup>s</sup>(·) is a sigmoid function and <sup>W</sup> ∈ <sup>S</sup> 3 ++ a symmetric positive definite matrix. This fitted parametric ellipsoid was treated as the ground truth for our model evaluation here. We ran active learning to identify the sublevel set of the 75% detection probability: {<sup>x</sup> ∈ <sup>R</sup> 3 : x <sup>⊤</sup>Wx ≤ <sup>s</sup> −1 (0.75)}. Note that this is a slightly misspecified problem—the data generation process is not exactly the same as the model assumption, as the parametric ellipsiod used a sigmoid link function, not the normal CDF link function used by the Bernoulli likelihood.

A total of 21 constraints are imposed by mixed likelihood training: one constraint at the origin x = 0 where the output probability is 50%, and 20 constraints sampled from the domain boundary where the output probability is close to 100%. We again add Bernoulli pseudo data for the standard

<sup>1</sup>A quarter ellipsoid to be precise, since headset thickness and latency have to be nonnegative.

![](_page_5_Figure_1.jpeg)

289 290

294

296

298

300

304

306

308 309

311

314 315 316

318

324

326

Figure 4: Preference probabilities Pr(x<sup>1</sup> ⪰ <sup>x</sup>2) predicted by GPs trained on synthetic data. Left: The ground truth probabilities Φ(x<sup>1</sup> − <sup>x</sup>2). Mid: A standard variational GP trained on preference observations only, which tends to be under confident at top left and bottom right corners. Right: A mixed likelihood GP trained on the same data but with additional synthetic Likert scale ratings on a scale of 0 to 2.

variational GP to serve as a baseline. More details of this problem are deferred to [§A.1.](#page-11-0)

In Figure [3](#page-4-0) the last panel "visual sensitivity," we plot F1 scores across active learning queries. Mixed likelihood constraints led to higher F1 scores than pseudo data, especially when using the GlobalMI acquisition function, and in early iterations for EAVC.

## 5. Combining User Confidence and Preference

Many vision studies investigate detection thresholds and can be modeled with Bernoulli level set estimation because there is a clearly-defined correct/incorrect user response. However, in many other domains of research related to human perception researchers often evaluate subjective user preferences with psychopysics where there may not be an objectively correct response [\(Maloney & Yang,](#page-9-14) [2003;](#page-9-14) [Bar](#page-8-11)[toshuk,](#page-8-11) [1978;](#page-8-11) [Jones & Tan,](#page-8-12) [2012\)](#page-8-12). These problems are ideal candidates for preference learning techniques and we next explore how Likert scale survey responses can be used to improve studies in this domain.

#### 5.1. Preference Learning and the Likert Likelihood

Given two stimuli <sup>x</sup>1, <sup>x</sup><sup>2</sup> ∈ <sup>R</sup> d , the preference likelihood models the probability that x<sup>1</sup> is preferred to x<sup>2</sup> as

$$\Pr(\mathbf{x}_1 \succeq \mathbf{x}_2) = \Phi(f(\mathbf{x}_1) - f(\mathbf{x}_2)),$$

where Φ(·) is the normal CDF. The difference of GPs, <sup>f</sup>(x1) − <sup>f</sup>(x2), is itself a GP, hence it is common to directly learn the difference as a GP with a special preference kernel [\(Houlsby et al.,](#page-8-13) [2011\)](#page-8-13). Then, fitting GPs on preference data is reduced to GP classification.

![](_page_5_Figure_0.jpeg)

Figure 5: The Brier scores (↓) and F1 scores (↑) of GPs trained on haptic data collected from five participants. The error bars show one standard error. Mixed likelihood GPs that include Likert scale confidence ratings generally achieve lower Brier scores and higher F1 scores.

example, participants can be asked to rate their confidence on a scale of 1 to 10 after indicating which of the two stimuli is the preferred choice. This differentiates between situations where x<sup>1</sup> is strongly preferred to x<sup>2</sup> or only marginally better. With mixed likelihood training, we can model Likert scale survey responses alongside preference observations to more effectively learn a user's underlying latent function.

We propose a novel likelihood for Likert scale survey responses. Let <sup>y</sup> ∈ {1, <sup>2</sup>, · · · , l} be the the Likert scale response, with <sup>l</sup> ∈ <sup>N</sup> the number of options. We call the absolute value of the difference |f(x1) − <sup>f</sup>(x2)| the preference strength. Intuitively, high strength preference is correlated with larger Likert scale response. We define cut points,

$$0 = c_1 \leq c_2 \leq \dots, \leq c_l < c_{l+1} = \infty,$$

that divide all nonnegative numbers into l intervals,

$$I_i = [c_i, c_{i+1}), \quad i = 1, 2, \dots, l.$$

Each interval I<sup>i</sup> corresponds to a response option. We construct the likelihood so that the probability of observing y = <sup>i</sup> is highest when the preference strength |f(x1) − <sup>f</sup>(x2)| falls into the corresponding interval. We also wish for the likelihood of y = i to be negatively correlated with the distance from the preference strength to the corresponding interval. Hence, we propose the following Likert scale likelihood:

$$\Pr(y = i \mid f(\mathbf{x}_1), f(\mathbf{x}_2)) = \frac{\exp(-\text{dist}_i)}{\sum_{j=1}^l \exp(-\text{dist}_j)}. \quad (2)$$

Here the distance is taken as the minimum possible distance to any point in the interval:

$$\text{dist}_i = \min_{a \in I_i} \left| |f(\mathbf{x}_1) - f(\mathbf{x}_2)| - a \right|, \quad i = 1, 2, \dots, l.$$

![](_page_6_Figure_0.jpeg)

334

336

338

351

354

356

358

360 361

364

366

368

371

374

378

Figure 6: The robot gait data collection GUI. After watching two videos side by side, the human subject reports which robot walks more naturally, and their confidence.

parameters on the training data. We add constraints on the cut points to avoid overfitting and a lapse rate parameter to damp the probability [\(2\)](#page-5-1) with a uniform distribution for enhanced robustness (see [§B\)](#page-11-1). Below we apply this Likert scale likelihood on a synthetic and real-world experiments.

One of the challenges of mixing different likelihoods is to ensure they are compatible with each other, in the sense that they can operate on the same latent function. For example, ordinal likelihoods are common for ordinal data, including the Likert scale confidence ratings. However, they cannot be mixed directly with preference observations because they treat the latent function as an "ordinal" strength ranging over entire real numbers. On the other hand, it is the absolute value of the latent difference <sup>f</sup>(x1) − <sup>f</sup>(x2) that represents the preference strength. This necessitates our development of a Likert scale likelihood here.

#### 5.2. Synthetic Experiments

In this section, we present experiments on a synthetic latent function. The ground truth function is a univariate identity function f(x) = x, and the preference observations are Bernoulli observations with

$$\Pr(x_1 \succeq x_2) = \Phi(x_1 - x_2).$$

The Likert scale responses on a scale of 0 to 2 are generated deterministically based on which interval the preference strength |f(x1) − <sup>f</sup>(x2)| falls into: [0, <sup>0</sup>.5], [0.5, 1], [1, ∞).

Figure [4](#page-5-2) shows GP fits on synthetic data generated from the ground truth latent function. We observe that the GP trained by mixing Bernoulli and Likert scale likelihoods learns the ground truth latent function more accurately, particularly at the corners where the standard GP confidence is too low.

### 5.3. Learning Haptic Preferences

Haptics is the study of how humans perceive the world through the sensation of touch [\(Kappers & Bergmann Tiest,](#page-9-15) [2013\)](#page-9-15). We obtained haptic experimental data from [Driller](#page-8-14) [\(2024,](#page-8-14) Chapter 5) by contacting the authors. In this study

![](_page_6_Figure_1.jpeg)

Figure 7: The Brier scores (↓) and F1 scores (↑) of GPs trained on robot gait human evaluations. Error bars show one standard errors. GPs trained with likert scale responses consistently achieve lower Brier scores and higher F1 scores.

five participants were presented 50 pairs of 3D printed surfaces with different degrees of microscale surface roughness and material elasticity. For each pair, the user touched both surfaces, reported which felt rougher, and rated the confidence of their judgement on a scale from one to nine.

We split each participant's dataset into 20 training trials and 30 test trials (see [§C](#page-12-0) for additional ablation studies). We train preference GPs with and without confidence ratings on the training set and evaluate them on the test set. Because class imbalance is not a concern with preference data, in addition to the F1 score we also measure the Brier score, which is the mean squared error between the predicted and actual probabilities [\(Brier,](#page-8-15) [1950\)](#page-8-15). We repeat the process 100 times with different training/test splits. The average Brier scores and F1 scores with one standard errors are shown in Figure [5.](#page-5-3) Mixed likelihood GPs trained with the Likert scale likelihood achieved lower Brier scores and higher F1 scores for all subjects except for subject #2.

We observe that subject #2's Likert responses were predominately confident ratings: 48% of their confidence ratings were 9 (the highest rating), 84% of their confidence ratings them were equal to or above 7, and no confidence ratings below 3 were reported. As a result, the Likert scale likelihood struggled to learn the cut points for subject #2. In contrast, other subjects' confidence ratings were spread more evenly (see Figure [9\)](#page-13-0). This suggests that a calibration stage could be important for using survey data in preference learning.

#### 5.4. Robot Gait Optimization

Preference learning can also be applied to robotics to align robot behavior with desired outcomes (e.g., [Tucker et al.,](#page-9-3) [2020\)](#page-9-3). We consider the robot gait optimization task from [Shvartsman et al.](#page-9-16) [\(2024\)](#page-9-16), who use human preferences to determine optimal motion control parameters that yield naturalistic robot gait. In their experiment, two videos of quadruped robots are played side by side, and study subjects choose the video with their preferred gait (see Figure [6\)](#page-6-0).

394

396

We repeated their study protocol, but additionally collected confidence ratings on a scale of 1 to 3 on each trial. One of authors of this paper participated in this experiment and collected 472 preference responses and confidence ratings. In Figure [7,](#page-6-1) we report Brier and F1 scores on test sets averaged over 100 different train/test splits. GPs that mixed the Likert scale likelihood with the preference likelihood consistently improved both the Brier score and the F1 score.

## 6. Related Work

From the view of probabilistic graphical models, mixed likelihood training conditions the latent variables (the latent GP function values and inducing values) on different types of observations using variational inference. In fact, variational GPs have been *implicitly* trained with mixed likelihoods in several applications throughout the years. For example, the heteroscedastic Gaussian likelihood, which assigns different noise levels to different data points, is technically a form of mixed likelihood training [\(Kersting et al.,](#page-9-17) [2007;](#page-9-17) [Lazaro-Gredilla & Titsias](#page-9-18) ´ , [2011;](#page-9-18) [Binois et al.,](#page-8-16) [2018\)](#page-8-16). Another example is the OR-channel likelihood for modeling multi-tone response in audiometry [\(Gardner et al.,](#page-8-4) [2015b\)](#page-8-4). Different parameters (number of tones) of the OR-channel likelihood correspond essentially to different likelihoods, and thus it is also an example mixing likelihoods, though using a Laplace approximation and not variational inference.

Importantly, these past examples are all mixtures of likelihoods from the same family. We step further in this paper and introduce a framework for mixing vastly different likelihoods. Recently, [Shvartsman et al.](#page-9-16) [\(2024\)](#page-9-16) have developed response time GPs that are jointly trained on human choices and response time, but their approach is a single likelihood modeling the joint distribution of both human choices and response time. Their likelihood is based on an approximation of the diffusion decision model designed by domain experts, which is not easily generalizable to other likelihoods or data types. [Murray & Kjellstrom¨](#page-9-19) [\(2018\)](#page-9-19) mixed likelihoods specifically for unsupervised representation learning in GP latent value models. Our work provides a general approach that, as we show, solves many problems arising from experimental designs and preference learning.

Mixed likelihood variational training is closely related to multitask GPs, where the goal is to learn *multiple* correlated latent functions (e.g., [Bonilla et al.,](#page-8-17) [2007a](#page-8-17)[;b\)](#page-8-18). Inter-task correlations can be encoded either via Kronecker kernel matrices or, more commonly for variational GPs, using the linear model of coregionalization (LMC) [\(Alvarez et al.,](#page-8-19) [2012\)](#page-8-19), in which the prediction for each task is a linear combination of multiple variational GPs. Prior work has constructed multi-task models with different likelihoods, each of which is associated with several latent functions, based on both Kronecker and LMC models [\(Pourmohamad & Lee,](#page-9-20)

[2016;](#page-9-20) [Moreno-Munoz et al.](#page-9-21) ˜ , [2018\)](#page-9-21). Our mixed likelihood approach learns a single, shared latent function from multiple data types. Merging all information into a single GP is necessary for the real-world applications we demonstrate here, such as enforcing domain knowledge constraints.

In [§4](#page-2-0) we developed domain knowledge constraints as a use case of mixed likelihood modeling. There are other ways to enforce constraints in variational GPs. Recently, [Cosier et al.](#page-8-20) [\(2024\)](#page-8-20) proposed enforcing constraints with a set of fixed inducing points with fixed inducing values. Compared to this approach, mixed likelihood training has two advantages. First, mixed likelihood training supports soft constraints by tuning the Gaussian likelihood noise, whereas fixing inducing values generally enforces hard constraints. Second, mixed likelihood training is easier to implement, as it is compatible with all off-the-shelf GP variational inference implementations. The only change needed is the training objective. In contrast, fixing inducing values requires a custom implementation of GP variational inference, more specifically a custom whitening strategy.

## 7. Discussion

We have shown that variational GPs can be trained with mixed likelihoods to incorporate multiple types of data in human-in-the-loop experiments. We demonstrated two main applications of mixed likelihood training in this paper: (a) imposing soft constraints on the latent function into GPs by mixing Gaussian likelihoods with Bernoulli likelihoods, and using the constrained variational GPs to accelerate active learning for Bernoulli level set estimation; and (b) leveraging Likert scale confidence ratings by mixing with a Likert scale likelihood to improve preference learning.

A few extensions to our framework are possible. Response time in human-in-the-loop experiments is typically correlated with preference strength, i.e., longer the response time often implies more uncertainty. With an appropriate likelihood for response times, which naturally shares the same latent function with preference observations, mixed likelihood training could significantly simplify the expert-designed likelihood of [Shvartsman et al.](#page-9-16) [\(2024\)](#page-9-16) based on the diffusion decision model.

Confidence ratings could also be used in active learning with mixed likelihood training, though this comes with challenges. As discussed in [§5.3,](#page-6-2) some participants produce low-quality ratings, which may require calibration (on the fly) before feeding them into the model. Furthermore, asking for confidence ratings increases the cognitive load on participants and may increase the experiment time per trial. It may be best to collect confidence ratings only in the early stage of active learning, when model uncertainty is highest.

- 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 Impact Statement This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here. References Alvarez, M. A., Rosasco, L., Lawrence, N. D., et al. Kernels for vector-valued functions: A review. *Foundations and Trends® in Machine Learning*, 4(3):195–266, 2012. [6](#page-7-0) Bartoshuk, L. M. The psychophysics of taste. *The American Journal of Clinical Nutrition*, 31(6):1068–1077, 1978. [5](#page-5-2) Binois, M., Gramacy, R. B., and Ludkovski, M. Practical heteroscedastic Gaussian process modeling for large simulation experiments. *Journal of Computational and Graphical Statistics*, 27(4):808–821, 2018. [6](#page-7-0) Biocca, F. A. and Rolland, J. P. Virtual eyes can rearrange your body: Adaptation to visual displacement in seethrough, head-mounted displays. *Presence*, 7(3):262–277, 1998. [4.5](#page-4-2) Blei, D. M., Kucukelbir, A., and McAuliffe, J. D. Variational inference: A review for statisticians. *Journal of the American statistical Association*, 112(518):859–877, 2017. [2](#page-1-1) Bonilla, E. V., Agakov, F. V., and Williams, C. K. I. Kernel multi-task learning using task-specific features. In Meila,
  - M. and Shen, X. (eds.), *Proceedings of the Eleventh International Conference on Artificial Intelligence and Statistics*, volume 2 of *Proceedings of Machine Learning Research*, pp. 43–50, San Juan, Puerto Rico, 21–24 Mar 2007a. PMLR. [6](#page-7-0) Bonilla, E. V., Chai, K., and Williams, C. Multi-task Gaussian process prediction. In Platt, J., Koller, D., Singer, Y., and Roweis, S. (eds.), *Advances in Neural Information Processing Systems*, volume 20. Curran Associates, Inc., 2007b. [6](#page-7-0) Brier, G. W. Verification of forecasts expressed in terms of probability. *Monthly weather review*, 78(1):1–3, 1950. [5.3](#page-6-2) Chu, W. and Ghahramani, Z. Preference learning with Gaussian processes. In *Proceedings of the 22nd International Conference on Machine Learning*, ICML, pp. 137–144, 2005. [1](#page-0-0) Cosier, L. C., Iordan, R., Zwane, S. N. T., Franzese, G., Wilson, J. T., Deisenroth, M., Terenin, A., and Bekiroglu,
- Y. A unifying variational framework for Gaussian process motion planning. In Dasgupta, S., Mandt, S., and Li, Y. (eds.), *Proceedings of The 27th International Conference on Artificial Intelligence and Statistics*, volume 238 of *Proceedings of Machine Learning Research*, pp. 1315– 1323. PMLR, 02–04 May 2024. [6](#page-7-0) Driller, K. K. *From Cue to Construct: Cues, Mechanisms, and Stability in Haptic Perception*. PhD thesis, Delft University of Technology, 2024. [5.3](#page-6-2) Gardner, J. R., Malkomes, G., Garnett, R., Weinberger,
  - K. Q., Barbour, D., and Cunningham, J. P. Bayesian active model selection with an application to automated audiometry. In *Advances in Neural Information Processing Systems 28*, pp. 2386–2394, 2015a. [1](#page-0-0) Gardner, J. R., Song, X., Weinberger, K. Q., Barbour, D. L., and Cunningham, J. P. Psychophysical detection testing with Bayesian active learning. In *Proceedings of the 31st Conference on Uncertainty in Artificial Intelligence*, UAI, pp. 286–295, 2015b. [1,](#page-0-0) [6](#page-7-0) Gramacy, R. B. *Surrogates: Gaussian Process Modeling, Design and Optimization for the Applied Sciences*. Chapman Hall/CRC, Boca Raton, Florida, 2020. [1](#page-0-0) Guan, P., Mercier, O., Shvartsman, M., and Lanman, D. Perceptual requirements for eye-tracked distortion correction in VR. In *ACM SIGGRAPH 2022 Conference Proceedings*, SIGGRAPH, 2022. [1](#page-0-0) Guan, P., Penner, E., Hegland, J., Letham, B., and Lanman,
  - D. Perceptual requirements for world-locked rendering in AR and VR. In *SIGGRAPH Asia 2023 Conference Papers*, SA, 2023. [1](#page-0-0) Hensman, J., Fusi, N., and Lawrence, N. D. Gaussian processes for big data. In *Proceedings of the Twenty-Ninth Conference on Uncertainty in Artificial Intelligence*, 2013. [1,](#page-0-0) [2](#page-1-1) Hensman, J., Matthews, A. G. d. G., and Ghahramani, Z. Scalable variational Gaussian process classification. In *Proceedings of The 18th International Conference on Artificial Intelligence and Statistics*, AISTATS, pp. 351– 360, 2015. [1,](#page-0-0) [2](#page-1-1) Houlsby, N., Huszar, F., Ghahramani, Z., and Lengyel, M. ´ Bayesian active learning for classification and preference learning. *arXiv preprint arXiv:1112.5745*, 2011. [5.1](#page-5-4) Houlsby, N., Huszar, F., Ghahramani, Z., and Hernandez- ´ lobato, J. Collaborative Gaussian processes for preference learning. In *Advances in Neural Information Processing Systems 25*, 2012. [1](#page-0-0) Jones, L. A. and Tan, H. Z. Application of psychophysical techniques to haptic research. *IEEE transactions on haptics*, 6(3):268–284, 2012. [5](#page-5-2)

504

506

508 509

511

514 515 516

518

524

526

528

531

534

536

538

- Kappers, A. M. and Bergmann Tiest, W. M. Haptic perception. *Wiley Interdisciplinary Reviews: Cognitive Science*, 4(4):357–374, 2013. [5.3](#page-6-2) Kersting, K., Plagemann, C., Pfaff, P., and Burgard, W. Most likely heteroscedastic Gaussian process regression. In *Proceedings of the 24th International Conference on Machine Learning*, ICML, pp. 393–400, 2007. [6](#page-7-0) Kupcsik, A., Hsu, D., and Lee, W. S. *Learning Dynamic Robot-to-Human Object Handover from Human Feedback*, pp. 161–176. Springer International Publishing, Cham, 2018. [1](#page-0-0) Kuss, M. and Rasmussen, C. Assessing approximations for Gaussian process classification. In *Advances in Neural Information Processing Systems 18*, pp. 699–706, 2005. [1](#page-0-0) Kwak, Y., Penner, E., Wang, X., Saeedpour-Parizi, M. R., Mercier, O., Wu, X., Murdison, S., and Guan, P. Saccadecontingent rendering. In *ACM SIGGRAPH 2024 Conference Papers*, pp. 1–9, 2024. [1](#page-0-0) Lazaro-Gredilla, M. and Titsias, M. K. Variational het- ´ eroscedastic Gaussian process regression. In *Proceedings of the 28th International Conference on Machine Learning*, ICML, pp. 841–848, 2011. [6](#page-7-0) Letham, B., Guan, P., Tymms, C., Bakshy, E., and Shvartsman, M. Look-ahead acquisition functions for Bernoulli level set estimation. In *Proceedings of The 25th International Conference on Artificial Intelligence and Statistics*, AISTATS, pp. 8493–8513, 2022. [4.1,](#page-2-2) [4.2,](#page-2-3) [4.4,](#page-3-1) [A](#page-10-0) Maloney, L. T. and Yang, J. N. Maximum likelihood difference scaling. *Journal of Vision*, 3(8):5–5, 2003. [5](#page-5-2) McKee, S. P., Klein, S. A., and Teller, D. Y. Statistical properties of forced-choice psychometric functions: Implications of probit analysis. *Perception & psychophysics*, 37(4):286–298, 1985. [4.1](#page-2-2) Moreno-Munoz, P., Art ˜ es, A., and ´ Alvarez, M. Hetero- ´ geneous multi-output Gaussian process prediction. In *Advances in Neural Information Processing Systems 31*, 2018. [6](#page-7-0) Murray, S. and Kjellstrom, H. Mixed likelihood Gaus- ¨ sian process latent variable model. *arXiv preprint arXiv:1811.07627*, 2018. [6](#page-7-0) Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P. F., Leike, J., and Lowe, R. Training language models to follow instructions with human feedback. In *Advances in Neural Information Processing Systems 35*, volume 35, pp. 27730–27744, 2022. [1](#page-0-0) Owen, L., Browder, J., Letham, B., Stocek, G., Tymms, C., and Shvartsman, M. Adaptive nonparametric psychophysics, 2021. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2104.09549) [2104.09549](https://arxiv.org/abs/2104.09549). [1](#page-0-0) Palmer, S. E. *Vision science: Photons to phenomenology*. MIT press, 1999. [4.1](#page-2-2) Pourmohamad, T. and Lee, H. K. H. Multivariate stochastic process models for correlated responses of mixed type. *Bayesian Analysis*, 11(3):797–820, 2016. [6](#page-7-0) Shvartsman, M., Letham, B., Bakshy, E., and Keeley, S. Response time improves Gaussian process models for perception and preferences. In *Proceedings of the 40th Conference on Uncertainty in Artificial Intelligence*, UAI, pp. 3211–3226, 2024. [5.4,](#page-6-3) [6,](#page-7-0) [7](#page-7-1) Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R., Voss, C., Radford, A., Amodei, D., and Christiano, P. F. Learning to summarize with human feedback. In *Advances in Neural Information Processing Systems 33*, pp. 3008–3021, 2020. [1](#page-0-0) Titsias, M. Variational learning of inducing variables in sparse Gaussian processes. In van Dyk, D. and Welling,
  - M. (eds.), *Proceedings of the Twelfth International Conference on Artificial Intelligence and Statistics*, volume 5 of *Proceedings of Machine Learning Research*, pp. 567– 574, Hilton Clearwater Beach Resort, Clearwater Beach, Florida USA, 16–18 Apr 2009. PMLR. [2](#page-1-1) Tucker, M., Novoseller, E., Kann, C., Sui, Y., Yue, Y., Burdick, J. W., and Ames, A. D. Preference-based learning for exoskeleton gait optimization. In *Proceedings of the IEEE International Conference on Robotics and Automation*, ICRA, pp. 2351–2357, 2020. [1,](#page-0-0) [5.4](#page-6-3) Ulrich, R. and Miller, J. Threshold estimation in twoalternative forced-choice (2afc) tasks: The spearmankarber method. ¨ *Perception & Psychophysics*, 66(3):517– 533, 2004. [4.1](#page-2-2) Williams, C. K. and Rasmussen, C. E. *Gaussian processes for machine learning*, volume 2. MIT press Cambridge, MA, 2006. [1](#page-0-0) Wolfe, J. M., Kluender, K. R., Levi, D. M., Bartoshuk, L. M., Herz, R. S., Klatzky, R. L., Lederman, S. J., and Merfeld,
    - D. M. *Sensation & perception*. Sinauer Sunderland, MA, 2006. [4.1](#page-2-2)

554

556

558

560

564

566

568

571

574

576

578

594

596

598

### A. Experimental Details of Bernoulli Level Set Estimation

Inudcing Points. All inducing points are fixed, not learned in hyperparameter optimization, because otherwise the GP models overfit easily on the problems we consider. Two types of inducing points are used in the GP models: (a) 100 Sobol samples from the domain; and (b) an inducing point at every constraint location. Both the standard variational GP and the mixed likelihood-trained GP use the same set of inducing points. Crucially, we found the additional inducing points at constraint locations are especially important for mixed likelihood-trained GP. Without these inducing points, the mixed likelihood-trained GP tends to be inflexible.

Evaluation Metric. Different from the prior work [Letham et al.](#page-9-13) [\(2022\)](#page-9-13), which primarily use the Brier score as the evaluation metric, we use the F1 score to evaluate active learning performance for Bernoulli level set estimation.

The Brier score for level set estimation used by [Letham et al.](#page-9-13) [\(2022\)](#page-9-13) is defined as

$$\frac{1}{n} \sum_{i=1}^n (p_i - o_i)^2,$$

where <sup>p</sup><sup>i</sup> ∈ [0, 1] is the model's probability prediction on a set Sobol samples in the domain and <sup>o</sup><sup>i</sup> ∈ {0, <sup>1</sup>} is the ground truth indicating whether each <sup>x</sup><sup>i</sup> belongs to the sublevel set {<sup>x</sup> ∈ <sup>R</sup> d : <sup>f</sup>(x) ≤ <sup>γ</sup>} for the threshold <sup>γ</sup> ∈ <sup>R</sup>. Here, the model's probability prediction for sublevel set is

$$p_i = \Pr(f(\mathbf{x}_i) \leq \gamma) = \Phi\left(\frac{\gamma - \mu_{\mathcal{D}}(\mathbf{x}_i)}{\sigma_{\mathcal{D}}(\mathbf{x}_i)}\right),$$

where µ<sup>D</sup> is the posterior mean and σ<sup>D</sup> is the posterior standard deviation.

It is often the case, especially in high dimensions, that only a small portion of the domain will have values below the target level set, i.e., the ground truth sublevel set is a tiny fraction of the entire domain. As a result, the vast majority of the ground truths o<sup>i</sup> are 0's, which results in label imbalance. For some high dimensional problems, we observe that the Brier score is not reliable due the label imbalance issue, i.e., predicting constant zero might even achieve lower Brier scores than active learning in some cases.

Thus, we opt for the F1 scores evaluated on a set of Sobol samples in the domain for Bernoulli level set estimation. Let <sup>V</sup> ⊆ <sup>R</sup> <sup>d</sup> be the ground truth sublevel set and <sup>V</sup><sup>b</sup> ⊆ <sup>R</sup> <sup>d</sup> be the estimate. In the context of level set estimation, the precision and recall have clear geometric interpretations:

$$\text{precision} = \frac{|V \cap \hat{V}|}{|\hat{V}|}, \text{ recall} = \frac{|V \cap \hat{V}|}{|V|}.$$

We use 10<sup>6</sup> Sobol samples in the domain to estimate the F1 scores.

Level Set Estimation Threshold. Every Bernoulli level set estimation problem in this paper aims for a target sublevel set with a threshold of 75% in the probability space. Equivalently, this is the same as estimating the Φ −1 (0.75) sublevel in the latent function value space

$$\{\mathbf{x} \in \mathbb{R}^d : f(\mathbf{x}) \leq \Phi^{-1}(0.75)\}.$$

The only exception is the parametric ellipsoid in the visual sensitivity task (see [§4\)](#page-2-0), where the target sublevel set in the latent function space is

$$\{\mathbf{x} \in \mathbb{R}^d : f(\mathbf{x}) \leq s^{-1}(0.75)\}, \quad s(z) = \frac{1}{1 + \exp(-z)}.$$

This difference is because the parametric ellipsiod uses a sigmoid link function, not the normal CDF.

Additional Details. Global look-ahead acquisition functions like GlobalMI and EAVC proposed by [Letham et al.](#page-9-13) [\(2022\)](#page-9-13) require a set of global reference points. Those reference points are Sobol samples in the domain for estimating global changes in mutual information and sublevel set volumes. We use 10<sup>4</sup> Sobol samples as reference points. Active learning starts with 10 initial Sobol samples.

#### A.1. Objectives

Psychometric Discrimination. This function is defined as

$$f(x_1, x_2) = \frac{1 + x_2}{0.05 + 0.4x_1^2(0.2x_1 - 1)^2}$$

on a domain (x1, x2) ∈ [−1, 1]<sup>2</sup> . It is clear that the Bernoulli probabilities Φ(f(x)) are exactly 50% on the line

$$\{(x_1, x_2) : -1 \leq x_1 \leq 1, x_2 = -1\},$$

and the Bernoulli probabilities are close to 100% on the line

$$\{(x_1, x_2) : -1 \leq x_1 \leq 1, x_2 = +1\}.$$

A total of 20 constraints are added: 10 points on the line <sup>x</sup><sup>2</sup> <sup>=</sup> −<sup>1</sup> and another 10 points on the line <sup>x</sup><sup>2</sup> = +1. The constraint target values are set to the ground truth latent function values.

Norm Ball. This function is defined as

$$f(\mathbf{x}) = 2\|\mathbf{x}\|$$

on a domain <sup>x</sup> ∈ [−1, 1]<sup>d</sup> . In the main paper, we have used d = 2 and d = 4. Note that there is a multiplication coefficient 2. The factor 2 makes sure that the function grows fast enough so that the Bernoulli probability Φ(f(x)) is close to 100% on the domain boundary. We impose a constraint at the origin x = 0 and additionally sample 5 Sobol samples as constraint locations from every hypercube face. The constraint target values are set to the ground truth latent function values.

Parametric Ellipsoid. This is a 3D function defined as

$$f(\mathbf{x}) = \mathbf{x}^\top \mathbf{W} \mathbf{x},$$

where

$$\mathbf{W} = \begin{pmatrix} +0.00345447 & -0.00344695 & -0.00144475 \\ -0.00344695 & 0.00556409 & 0.00252343 \\ -0.00144475 & 0.00252343 & 0.00466492 \end{pmatrix}$$

and the domain is [−30, 50] × [0, 60] × [0, 75] with each axis being IPD offsets, camera z-axis errors, and passthrough latency. Note that W is symmetric and positive definite. The weight matrix W is estimated by maximum likelihood on the collected human data with convex optimization. Note that the link function for this objective is a sigmoid function <sup>s</sup>(·) = 1/(1 + exp(− ·)), not a normal CDF. We impose a constraint at the origin <sup>x</sup> <sup>=</sup> <sup>0</sup> and sample 5 Sobol samples as constraint locations from each of the following faces

$$\begin{aligned}
F_0 &= \{(x_1, x_2, x_3) : x_1 = -30, 0 \leq x_2 \leq 60, 0 \leq x_3 \leq 75\}, \\
F_1 &= \{(x_1, x_2, x_3) : x_1 = 50, 0 \leq x_2 \leq 60, 0 \leq x_3 \leq 75\}, \\
F_2 &= \{(x_1, x_2, x_3) : -30 \leq x_1 \leq 50, x_2 = 60, 0 \leq x_3 \leq 75\}, \\
F_3 &= \{(x_1, x_2, x_3) : -30 \leq x_1 \leq 50, 0 \leq x_2 \leq 60, x_3 = 75\},
\end{aligned}$$

which are four faces that do not contain the origin. The constraint target value at a location x is set to Φ −1 (min{s(x), <sup>0</sup>.999}). Namely, we first evaluate the ground truth probability s(x), then truncate it by 99.9%, and then covert it into the corresponding latent function value as if the link function was the normal CDF Φ(·). The ground truth latent function values cannot be directly used in mixed likelihood training because of the link functions mismatch with each other. Thus, the conversion is necessary. The truncation is also necessary, because otherwise Φ −1 (s(x)) might be infinity due to floating point overflow. Note that this is an example that the "believed" latent function value, not the ground truth latent function value, is used in constraints.

# B. Additional Details of the Likert Scale Likelihood

A constraint <sup>c</sup>i+1 − <sup>c</sup><sup>i</sup> ≤ <sup>2</sup> is enforced for each pair of adjacent cut points to avoid overfitting. Note that Φ(2) ≈ <sup>0</sup>.98. This constraint enforces that the preference probability within the same Likert scale response is no greater than 98%, a natural

689 690

694

696

698

700

704

706

708 709

711

assumption on the Likert scale response. With limited number of data, the cut points and the latent function may not be learned accurately. Thus, it is important to add constraints on the cut points to avoid overfitting.

In addition, we introduce a lapse rate parameter to damp the Likert scale likelihood. Let <sup>p</sup>1, p2, · · · , p<sup>l</sup> be the probabilities produced by the Likert scale likelihood [\(2\)](#page-5-1). Then, we damp the probabilities with a mixutre of uniform distribution

$$p_i^{\text{damp}} = (1 - \lambda)p_i + \lambda \cdot \frac{1}{l},$$

where <sup>λ</sup> ≥ <sup>0</sup> is the lapse rate parameter. The damped probability is used to train variational GPs in the experiments, and we use a lapse rate of λ = 0.1 throughout. Intuitively, the damped Likert scale likelihood is more robust because it prevents extremely small probabilities, particularly when the number of data is limited.

# C. Learning Haptic Preferences

The raw Likert scale confidence ratings on a scale of 1 to 9 are mapped to a scale of 0 to 2:

1, 2, 3 
$$\mapsto$$
 0,      4, 5, 6  $\mapsto$  1,      7, 8, 9  $\mapsto$  2,

which is primarily due to the easy of programming.

The Brier score shown in Figure [5](#page-5-3) is defined as

$$\frac{1}{n} \sum_{i=1}^n (p_i - o_i)^2, \quad (3)$$

where <sup>o</sup><sup>i</sup> ∈ {0, <sup>1</sup>} indicating which stimulus (xi<sup>1</sup> or <sup>x</sup>i2) is preferred and <sup>p</sup><sup>i</sup> is the GP probability prediction

$$\Phi \left( \frac{\mu_{\mathcal{D}}(\mathbf{x}_{i1}, \mathbf{x}_{i2})}{\sqrt{1 + \sigma_{\mathcal{D}}^2(\mathbf{x}_{i1}, \mathbf{x}_{i2})}} \right),$$

where <sup>µ</sup>D(xi1, <sup>x</sup>i2) is the posterior mean of the latent difference <sup>f</sup>(xi1) − <sup>f</sup>(xi2) conditioned on the training data D, and σ <sup>D</sup>(xi1, <sup>x</sup>i2) is the posterior variance of the latent difference <sup>f</sup>(xi1) − <sup>f</sup>(xi2) conditioned on the training data.

We use 100 Sobol samples as inducing points for both the standard variational GPs and mixed likelihood-trained GPs. In Figure [8,](#page-13-1) we present the Brier scores and F1 scores of GPs trained with varying number of data points. Since we only have 50 data points per human subject, we only experiment with training sizes of 10, 20, and 30. Likert scale confidence ratings again improve both Brier scores and F1 scores except for subject #2. As discussed in the main paper, it is most likely because subject #2 Likert scale ratings are predominantly confident. In Figure [9,](#page-13-0) we plot all subjects' confidence rating histograms. There is a clear difference between the histogram of subject #2 and the remaining subjects: subject #2's ratings tend to be more confident then remain subjects.

# D. Robot Gait Optimization

The Brier score presented in Figure [7](#page-6-1) is computed similarly as discussed in [§C.](#page-12-0) We use 100 Sobol samples as inducing points for both the standard variational GPs and mixed likelihood-trained GPs. In Figure [10,](#page-14-0) we plot the distribution of confidence ratings in the data collected from the robot gait optimization task: 218 of them are 1; 176 of them are 2; and 78 of them are 3. The confidence ratings are generally well-balanced.

![](_page_13_Figure_0.jpeg)

![](_page_13_Figure_2.jpeg)

Figure 8: Brier scores and F1 scores of GPs with varying number of training data points on the haptic perception dataset.

Figure 9: Human subjects' confidence rating histograms in the haptic dataset.

![](_page_14_Figure_1.jpeg)

Figure 10: The confidence rating histogram of the data collected from the robot gait optimization task.