# Rb-Modulation: Training-Free Stylization Using Reference-Based Modulation

Litu Rout1,2∗ Yujia Chen1 **Nataniel Ruiz**1 Abhishek Kumar3 Constantine Caramanis2 Sanjay Shakkottai2 **Wen-Sheng Chu**1 1 Google 2 UT Austin 3 Google DeepMind
{litu.rout,constantine,sanjay.shakkottai}@utexas.edu
{liturout,yujiachen,natanielruiz,abhishk,wschu}@google.com

## Abstract

We propose Reference-Based Modulation (RB-Modulation), a new plug-and-play solution for training-free personalization of diffusion models. Existing trainingfree approaches exhibit difficulties in (a) style extraction from reference images in the absence of additional style or content text descriptions, (b) unwanted content leakage from reference style images, and (c) effective composition of style and content. RB-Modulation is built on a novel stochastic optimal controller where a style descriptor encodes the desired attributes through a terminal cost. The resulting drift not only overcomes the difficulties above, but also ensures high fidelity to the reference style and adheres to the given text prompt. We also introduce a cross-attention-based feature aggregation scheme that allows RB-Modulation to decouple content and style from the reference image. With theoretical justification and empirical evidence, our test-time optimization framework demonstrates precise extraction and control of *content* and *style* in a training-free manner. Further, our method allows a seamless composition of content and style, which marks a departure from the dependency on external adapters or ControlNets. See project page https://rb-modulation.github.io/ for code and further details.

## 1 Introduction

Text-to-image (T2I) generative models (Ramesh et al., 2021; Rombach et al., 2022; Saharia et al., 2022) have excelled in crafting visually appealing images from text prompts. These T2I models are increasingly employed in creative endeavors such as visual arts (Xu et al., 2024), gaming (Pearce et al., 2023), personalized image synthesis (Ruiz et al., 2023; Huang et al., 2024a; Hu et al., 2021; Shah et al., 2023), stylized rendering (Sohn et al., 2023; Hertz et al., 2023; Wang et al., 2024a; Jeong et al., 2024), and image inversion or editing (Ulyanov et al., 2018; Delbracio & Milanfar, 2023; Rout et al., 2023b; 2024; Mokady et al., 2023). Content creators often need precise control over both the content and the *style* of generated images to match their vision. While the content of an image can be conveyed through text, articulating an artist's unique style - characterized by distinct brushstrokes, color palette, material, and texture - is substantially more nuanced. This has led to research on personalization through visual prompting (Sohn et al., 2023; Hertz et al., 2023; Wang et al., 2024a). Recent studies have focused on finetuning pre-trained T2I models to learn style from a set of reference images (Gal et al., 2022; Ruiz et al., 2023; Sohn et al., 2023; Hu et al., 2021). This involves optimizing the model's text embeddings, model weights, or both, using the denoising diffusion loss. However, these methods demand substantial computational resources for training or finetuning large-scale foundation models, thus making them expensive to adapt to new, unseen styles. Furthermore, these methods often depend on human-curated images of the same style, which is less practical and can compromise quality when only a single reference image is available. In training-free **stylization**, recent methods (Hertz et al., 2023; Wang et al., 2024a; Jeong et al., 2024) manipulate keys and values within the attention layers using just one reference style image. These methods face challenges in both extracting the style from the reference style image and accurately transferring the style to a target content image. For instance, during the DDIM inversion step (Song et al., 2021a) utilized by StyleAligned (Hertz et al., 2023), fine-grained details tend to be compromised. To mitigate this issue, InstantStyle (Wang et al., 2024a) incorporates features from
∗This work was done during an internship at Google.

1

Reference content Reference style A guitar A piano A butterfly A skyscraper A lighthouse A kangaroo A dwarf A dragon An elf
the reference style image into specific layers of a previously trained IP-Adapter (Ye et al., 2023).

However, identifying the exact layer for feature injection in a model is complex and not universally applicable across models. Also, feature injection can cause content leakage from the style image into the generated content. Moving on to content-style **composition**, InstantStyle (Wang et al., 2024a) employs a ControlNet (Zhang et al., 2023) (an additionally trained network) to preserve image layout, which inadvertently limits its diversity. We introduce Reference-Based Modulation (RB-Modulation), a novel approach for stylization and composition that eliminates the need for training or finetuning diffusion models (e.g. Control- Net (Zhang et al., 2023) or adapters (Ye et al., 2023; Hu et al., 2021)). Our work reveals that the reverse dynamics in diffusion models can be formulated as stochastic optimal control problem. By incorporating style features into the controller's terminal cost, we modulate the drift field in diffusion model's reverse dynamics, enabling training-free personalization. Unlike conventional attention processors that often leak content from the reference style image, we propose to enhance the image fidelity via an Attention Feature Aggregation (AFA) module that decouples content from reference style image. We demonstrate the effectiveness of our method in stylization (Hertz et al., 2023; Wang et al., 2024a; Jeong et al., 2024) and style+content composition, as illustrated in Figure 1(a) and (b), respectively. Our experiments show that RB-Modulation outperforms current SoTA methods (Hertz et al., 2023; Wang et al., 2024a) in terms of human preference and prompt-alignment metrics.

## Our Contributions Are Summarized As Follows:

- We present reference-based modulation (RB-Modulation), a novel stochastic optimal control based test-time optimization framework that enables training-free, personalized style and content control, with a new Attention Feature Aggregation (AFA) module to maintain high fidelity to the reference image while adhering to the given prompt (§4).

- We provide theoretical justifications connecting optimal control and reverse diffusion dynamics. We leverage this connection to incorporate desired attributes (e.g., style) in our controller's terminal cost and personalize T2I models in a training-free manner (§5).

- We perform extensive experiments covering stylization and content-style composition, demonstrating superior performance over SoTA methods in human preference metrics (§6).

## 2 Related Work

Personalization of T2I models: T2I generative models (Rombach et al., 2022; Podell et al., 2023; Pernias et al., 2024) can now generate high quality images from text prompts. Their text-following ability has unlocked new avenues in personalized content creation, including text-guided image editing (Mokady et al., 2023; Rout et al., 2024), solving inverse problems (Rout et al., 2023b; 2024), concept-driven generation (Ruiz et al., 2023; Tewel et al., 2023; Kumari et al., 2023; Chen et al., 2024), personalized outpainting (Tang et al., 2023), identity-preservation (Ruiz et al., 2024; Huang et al., 2024a; Wang et al., 2024b), and stylized synthesis (Sohn et al., 2023; Wang et al., 2024a; Hertz et al., 2023; Shah et al., 2023). To tailor T2I models for a specific style (e.g., painting) or content (e.g., object), existing methods follow one of two recipes: (1) full finetuning (FT) or parameter efficient finetuning (PEFT) and (2) training-free, which we discuss below. Finetuning T2I models for personalization: FT (Ruiz et al., 2023; Everaert et al., 2023) and PEFT (Kumari et al., 2023; Hu et al., 2021; Sohn et al., 2023; Shah et al., 2023) methods excel at capturing style or object details when the underlying T2I model can be finetuned on a few (typically 4) reference images for few thousand iterations. PARASOL (Tarres et al. ´ , 2024) requires supervised data via a cross-modal search to train both the denoising U-Net and a projector network. Diff-NST (Ruta et al., 2023) trains the attention processor by targeting the 'V' values within the denoising U-Net. The curation of supervised data and resource-intensive finetuning for every style or content makes these methods challenging for practical usage.

Training-free methods for personalization: Training-free personalization methods are preferable to finetuning methods given the vastly faster time of execution. In **StyleAligned** (Hertz et al., 2023), a reference style image and a text prompt describing the style are used to extract style features via DDIM inversion (Song et al., 2021a). Target queries and keys are then normalized using adaptive instance normalization (Huang & Belongie, 2017) based on reference counterparts. Finally, reference image keys and values are merged with DDIM-inverted latents in self-attention layers, which tends to leak content information from the reference style image (Figure 2). Moreover, the need for textual description in the DDIM inversion step can degrade its performance. DiffusionDisentanglement (Wu et al., 2023) aims to reduce the approximation error in DDIM inversion by jointly minimizing a perceptual loss and a directional CLIP loss, which is prone to content leakage (Wang et al., 2024a). **Swapping Self-Attention (SSA)** (Jeong et al., 2024) addresses these limitations by replacing the target keys and values in self-attention layers with those from a reference style image. It still relies on DDIM inversion to cache keys and values of the reference style, which tends to compromise fine-grained details (Wang et al., 2024a). Both StyleAligned (Hertz et al., 2023) and SSA (Jeong et al., 2024) require two reverse processes to share their attention layer features and thus demand significant memory. **InstantStyle** (Wang et al., 2024a) injects reference style features into specific cross-attention layers of IP-Adapter (Ye et al., 2023), addressing two key limitations: DDIM inversion and memory-intensive reverse processes. However, pinpointing the exact layers for feature injection is complex, and may not generalize to other models. In addition, when composing style and content, InstantStyle (Wang et al., 2024a) relies on ControlNet (Zhang et al., 2023), which can limit the diversity of generated images to fixed layouts and deviate from the prompt. Optimal Control: Stochastic optimal control finds wide applications in diverse fields such as molecular dynamics (Holdijk et al., 2024), economics (Fleming & Rishel, 2012), non-convex optimization (Chaudhari et al., 2018), robotics (Theodorou et al., 2011), and mean-field games (Carmona et al., 2018) Despite its extensive use, and recent works on its connections to diffusion based generative models (Berner et al., 2024; Tzen & Raginsky, 2019; Chen et al., 2023), it has been less explored in training-free personalization. In this paper, we introduce a novel test-time optimization framework leveraging the main concepts from optimal control to achieve training-free personalization. A key aspect of optimal control is designing a controller to guide a stochastic process towards a desired terminal condition (Fleming & Rishel, 2012). This aligns with our goal of training-free personalization, as we target a specific style or content at the end of the reverse diffusion process, which can be incorporated in the controller's terminal condition. RB-Modulation overcomes several challenges encountered by SoTA methods (Hertz et al., 2023; Jeong et al., 2024; Wang et al., 2024a). Since RB-Modulation does not require DDIM inversion, it retains fine-grained details unlike StyleAligned (Hertz et al., 2023). Using a stochastic controller to refine the trajectory of a single reverse process, it overcomes the limitation of coupled reverse processes (Hertz et al., 2023). By incorporating a style descriptor in our controller's terminal cost, it eliminates the dependency on Adapters (Ye et al., 2023; Hu et al., 2021) or ControlNets (Zhang et al., 2023) by InstantStyle (Wang et al., 2024a).

3 PRELIMINARIES
Diffusion models consist of two stochastic processes: (a) *noising process*, modeled by a Stochastic Differential Equation (SDE) known as forward-SDE: dXt = f(Xt, t) dt + g(Xt, t) dWt, X0 ∼ p0, and (b) *denoising process*, modeled by the time-reversal of forward-SDE under mild regularity conditions (Anderson, 1982), also known as reverse-SDE:

$$\mathrm{)}\,\mathrm{d}W_{t},\qquad X_{1}\sim{\mathcal{N}}\left(0,\mathrm{I}_{d}\right).\quad\mathrm{(1)}$$

dXt =-f(Xt, t) − g 2(Xt, t)∇ log p(Xt, t)dt + g(Xt, t) dWt, X1 ∼ N (0,Id). (1)
Here, W = (Wt)t≥0 is standard Brownian motion in a filtered probability space, (Ω, F,(Ft)t≥0,P), p(·, t) denotes the marginal density of p at time t, and ∇ log pt(·) the corresponding score function. f(Xt, t) and g(Xt, t) are called drift and volatility, respectively. A popular choice of f(Xt, t) = −Xt and g(Xt, t) = 
√2 corresponds to the well-known forward Ornstein-
Uhlenbeck (OU) process.

For T2I generation, the reverse-SDE (1) is simulated using a neural network s (xt, t; θ) (Hyvarinen ¨ & Dayan, 2005; Vincent, 2011) to approximate ∇x log p(xt, t). Importantly, to accelerate the sampling process in practice (Song et al., 2021a; Karras et al., 2022; Zhang & Chen, 2022), -
the reverse-SDE (1) shares the same path measure with a probability flow ODE: dXt =
f(Xt, t) −
1 2 g 2(Xt, t)∇ log p(Xt, t)dt, where X1 ∼ N (0,Id).

Personalized diffusion models either fully finetune θ of s (xt, t; θ) (Ruiz et al., 2023; Everaert et al., 2023), or train a parameter-efficient adapter ∆θ for s (xt, t; θ + ∆θ) on reference style images (Hu et al., 2021; Sohn et al., 2023; Shah et al., 2023). Our method does not finetune θ or train ∆θ. Instead, we derive a new drift field through a stochastic control that *modulates* the reverse-SDE (1).

## 4 Method

Personalization using optimal control: Normalize time t by the total number of diffusion steps T such that 0 ≤ t ≤ 1. Let us denote by u : R
d × [0, 1] → R
da controller from the admissible set of controls U ⊆ R
d, Xu t ∈ R
da state variable, ` : R
d × R
d × [0, 1] → R the transient cost, and h : R
d → R the terminal cost of the reverse process (Xu t)
0 t=1. We show in §5 that training-free personalization can be formulated as a control problem where the drift of the standard reverse-SDE (1) is modified via RB-modulation:

$$\min_{u\in\mathcal{U}}\mathbb{E}[\int_{1}^{0}\ell\left(X_{t}^{u},u(X_{t}^{u},t),t\right)\mathrm{d}t+\gamma h(X_{0}^{u})],\quad\text{where}\tag{2}$$ $$\mathrm{d}X_{t}^{u}=\left[f(X_{t}^{u},t)-g^{2}(X_{t}^{u},t)\nabla\log p(X_{t}^{u},t)+u(X_{t}^{u},t)\right]\mathrm{d}t+g(X_{t}^{u},t)\mathrm{d}W_{t},X_{1}^{u}\sim\mathcal{N}\left(0,\mathrm{I}_{d}\right).$$

Importantly, the terminal cost h(·), weighted by γ, captures the discrepancy in feature space between the styles of the reference image and the generated image. The resulting controller u(·, t) modulates the drift over time to satisfy this terminal cost. We derive the solution to this optimal control problem through the Hamilton-Jacobi-Bellman (HJB) equation (Fleming & Rishel, 2012); refer to Appendix A for details. Our proposed RB-Modulation **Algorithm** 1 has two key components: (a) stochastic optimal controller and (b) attention feature aggregation. Below, we discuss each in turn. (a) Stochastic Optimal Controller (SOC): We show that the reverse dynamics in diffusion models can be framed as a stochastic optimal control problem with a quadratic terminal cost (theoretical analysis in §5). For personalization using a reference style image X
f 0 = z0, we use a Contrastive Style Descriptor (CSD) (Somepalli et al., 2024) to extract style features Ψ(X
f 0). Since the score functions s (xt, t; θ)≈∇ log p (Xt, t) are available from pre-trained diffusion models (Podell et al.,
2023; Pernias et al., 2024), our goal is to add a correction term u(·, t) to modulate the reverse-
SDE and minimize the overall cost (2). We approximate Xu 0 with its conditional expectation using Tweedie's formula (Efron, 2011; Rout et al., 2023b; 2024). Finally, we incorporate the style features into our controller's terminal cost as: h (Xu 0) = kΨ(X
f 0) − Ψ(E [Xu 0 |Xu t])k 22.

Our theoretical results (§5) suggest that the optimal controller can be obtained by solving the HJB equation and letting γ → ∞. In practice, this translates to dropping the transient cost ` (Xu t
, u(Xu t
, t), t) and solving (2) with only the terminal constraint, i.e.,

$$\operatorname*{min}_{u\in{\mathcal{U}}}\|\Psi(X_{0}^{f})-\Psi(\mathbb{E}\left[X_{0}^{u}|X_{t}^{u}\right])\|_{2}^{2}.$$
2. (3)
$$\left(2\right)$$

Thus, we solve (3) to find the optimal control u and use this controller in the reverse dynamics (2) to update the current state from Xu tto Xu t−∆t
(recall that time flows backwards in the reverse-SDE (1)).

Our implementation of (3) is given in **Algorithm** 1, which follows from our theoretical insights. Implementation challenge: For smaller models (Rombach et al., 2022), we can directly solve our control problem (3). However, for larger models (Podell et al., 2023; Pernias et al., 2024), the control objective (3) requires back propagation through the score network with tentatively billions of parameters. This significantly increases time and memory complexity (Rout et al., 2023b; 2024). We propose a test-time proximal gradient descent approach to address this challenge. The key ingredient of our **Algorithm** 1 is to find the previous state Xt−∆t by modulating the current state Xt based on an optimal controller u
∗. The optimal controller u
∗is obtained by minimizing the discrepancy in style between X¯ u 0
:= E[Xu 0 |Xu t = xt], obtained using our controlled reverse-SDE (3), and the reference style image z0. Motivated by this interpretation, an alternate **Algorithm** 2 avoids back propagation through s(xt, t; θ) by introducing a dummy variable x0, which serves as a proxy for X¯ u 0in the terminal cost. Instead of forcing x0 to be decided by the dynamics of the reverse-SDE as in **Algorithm** 1, we allow it to be only approximately faithful to the dynamics. This is implemented by adding a proximal penalty, i.e. x
∗0 = arg minx0∈Rd kΨ(X
f 0
) − Ψ(x0)k 22 + λkx0 − E [Xu 0|Xu t]k 22, where the hyper-parameter λ controls the faithfulness of the reverse dynamics. This penalty assumes that with a small step-size in (3), x
∗0and E[Xu 0|Xu t = xt] will be close. Thus, **Algorithm** 2 enables personalization of large-scale foundation models, matching the speed of training-free methods and obtaining 5-20X speedup over training-based methods; see Table 4 in Appendix B.2 for details. While prior works (Chung et al., 2023; Zhu et al., 2023; He et al., 2024) have used a proximal sampler in related settings, their underlying generative model is not personalized. We believe that this is an important reason why our method results in a significant speedup while satisfying the terminal constraints. Our paper takes the first step in personalizing the underlying generative model via a novel attention processor as discussed below. (b) Attention Feature Aggregation (AFA): Let d denote the dimension of the latent variable Xt, nq the embedding dimension for query Q, and nh the output dimension of the hidden layer.

Transformer-based diffusion models (Rombach et al., 2022; Podell et al., 2023; Pernias et al., 2024)
consist of self-attention and cross-attention layers operating on latent embedding xt ∈ R
d×nh .

Within the attention module Attention(*Q, K, V* ), xt is projected into queries Q ∈ R
d×nq, keys K ∈ R
d×nq, and values V ∈ R
d×nh using linear projections. Through Q, K, and V , attention layers capture global context and improve long-range dependencies within xt. To incorporate a reference image (e.g., style or content) while retaining alignment with the prompt, we introduce the Attention Feature Aggregation (AFA) module. Given a prompt p, a reference style image Is, and a reference content image Ic, we first extract the embeddings using CLIP text encoder (Radford et al., 2021) and CSD image encoder (Somepalli et al., 2024). These embeddings are projected into keys and values using linear projection. We denote by Kp and Vp the keys and values from p, Ks and Vs from Is, Kc and Vc from Ic (used only in content-style composition). The query Q, derived from a linear projection of xt, remains consistent in the AFA module. To maintain consistency between text and style, we compose the keys and values of both text and style in our attention mechanism. The final output of the AFA module is given by AF A = Avg (Atext, Astyle, Atext+style), A*text* = Attention(Q, [K; Kp], [V ; Vp]),
A*style* = Attention(Q, [K; Ks], [V ; Vs]), Atext+*style* = Attention(Q, [K; Kp; Ks], [V ; Vp; Vs]),
where [K; Kp] ∈ R
2d×nqindicates concatenation of K with Kp along the number of tokens dimension. For style-content composition, we process the content image Ic in the same way as the reference style image Is, and obtain another set of attention outputs:
AF A = Avg (Atext, Astyle, Acontent, Acontent+*style*), A*content* = Attention(Q, [K; Kc], [V ; Vc]), Acontent+*style* = Attention(Q, [K; Ks; Kc], [V ; Vs; Vc]). Importantly, the AFA module is computationally tractable as it only requires the computation of a multi-head attention, which is widely used in practice (Podell et al., 2023). Disentangling content and style. In stylization (content described by text; style illustrated by a reference style image), prior works (Hertz et al., 2023; Wang et al., 2024a) inject the entire reference style image Is that does not disentangle content and style. However, our AFA module injects

Algorithm 1: RB-Modulation (Exact)
Input: Diffusion steps T, reference prompt p, reference style image z0, style descriptor Ψ(·),
score network s(·, ·, ·; θ)
Tunable parameter: Stepsize η, optimization steps M
Output: Personalized latent Xu 0 1 Initialize xT ← N (0, Id) 2 for t = T to 1 do 3 Initialize controller u = 0 4 for m = 1 to M do 5 xˆt = xt + u . controlled state 6 X¯ u 0 
= √
xˆt α¯t
+
(1−α¯t)
√α¯t s (xˆt*, t,* p; θ)
7 h(X¯ u 0
) = kΨ(z0) − Ψ(X¯ u 0
)k 2 2 using Eq. (3)
8 u = u − η∇uh(X¯ u 0
) . update controller 9 end 10 x
∗ t 
= xt + u . optimally controlled state 11 X¯ u 0 
=
x
∗
√ tα¯t
+
(1−α¯t)
√α¯t s (x
∗ t
, t, p; θ) . terminal state 12 xt−1 ← DDIM(X¯ u 0
, x
∗ t
) . one denoising update 13 end 14 **return** Xu 0 Algorithm 2: RB-Modulation (Proximal)
Input: Diffusion time steps T, reference prompt p, reference style image z0, style descriptor Ψ(·),
score network s(·, ·, ·; θ)
Tunable parameters: Stepsize η, optimization steps M, proximal strength λ Output: Personalized latent Xu 0 1 Initialize xT ← N (0, Id)
2 for t = T to 1 do 3 Compute posterior mean E[Xu 0 |Xu t 
= xt] = √
xt α¯t
+
(1−α¯t)
√α¯ts (xt*, t,* p; θ)
4 Initialize opt. variable x0 = E[Xu 0|Xu t = xt]
5 for m = 1 to M do 6 Compute controller's cost L(x0) := kΨ(z0) −
Ψ(x0)k 2 2 
+ λkx0 − E [Xu 0 |Xu t 
= xt]k 2 2 7 Update optimization variable x0 = x0 − η∇x0L(x0)
8 end 9 xt−1 ← DDIM(x0, xt) . one denoising step 10 end 11 **return** Xu 0
only the style features from Is using the style attention head of the Vision Transformer (ViT) in CSD (Somepalli et al., 2024). The AFA module achieves content-style disentanglement by computing separate attention maps for content from text and style from image. In this case, SOC does not handle content and focuses solely on style aspects by using the style attention head as Ψ(·). In content-style composition (content described by both text and a reference content image; style described by a reference style image), the AFA module injects content (extracted from the reference content) and style features (from the reference style image) separately using their respective attention heads in the ViT (Somepalli et al., 2024). The SOC module controls *content* by minimizing the discrepancy between content features from the generated image and the *reference content* image, and *style* by minimizing the discrepancy between style features extracted from the generated and reference style image. This distinction from prior works enables our method to prevent leakage.

## 5 Theoretical Justifications

Problem setup: We outline an approach to derive the optimal controller for a special case of our control problem (2). We substitute t ← 1−t to account for the time reversal in the reverse-SDE (1).

Here, Xu 0 
∼ N (0,Id) and Xu 1 
∼ p*data*. We consider the dynamic without the Brownian motion:
dXu t 
= v(Xu t
, u, t)d*t, X*u t0 
= x0, where 0 ≤ t0 ≤ t ≤ tN ≤ 1 and v : R
d × R
d × [t0, tN ] → R
d denotes the drift field. The optimal controller u
∗can be derived by solving the Hamilton-Jacobi-
Bellman (HJB) equation (Fleming & Rishel, 2012; Basar et al., 2020), see Appendix A for details.

Incorporating optimal control in diffusion: Following recent works (Kappen, 2008; Chen et al.,
2023), we consider a dynamical system whose drift field minimizes a transient trajectory cost and a terminal cost (weighted by γ) to ensure "closeness" to reference content x1 (Appendix A.1). Proposition A.2 (Chen et al., 2023) outlines the optimal control in the limiting setting where γ → ∞.

Furthermore, suppose we replace x1 with its conditional expectation (discussed in Remark A.3), the resulting dynamic is the standard reverse-SDE for the Orstein-Uhlenbeck (OU) diffusion process for a particular noise schedule. This connection between classic linear quadratic control and the standard reverse-SDE allows us to study other diffusion problems (e.g., personalization) through the lens of stochastic optimal control. For instance, we derive the optimal controller given reference style features y1 at the terminal time.

Proposition 5.1. *Suppose* A ∈ R
k×d *be a linear style extractor that operates on the terminal state* Xu 1 
∈ R
d. Given reference style features y1*, consider the control problem:*

$$\min_{u\in\mathcal{U}}\int_{t_{0}}^{1}\frac{1}{2}\left\|u(X_{t}^{u},t)\right\|^{2}dt+\frac{\gamma}{2}\left\|A X_{1}^{u}-y_{1}\right\|_{2}^{2},\text{where}\mathrm{d}X_{t}^{u}=u(X_{t}^{u},t)\,\mathrm{d}t,\ X_{t_{0}}^{u}=x_{0}.$$

Then, in the limit when γ → ∞*, the optimal controller* u
∗ =
$$\left(A^{T}A\right)^{-1}A^{T}\left(y_{1}-A\mathbf{x}_{t}\right)$$
1−t, which yields the
following controlled dynamic: dXu
t =
(A
T A)
−1A
T (y1−Axt)
1−tdt.

Implication. The optimal controller depends on the reference *style features* y1 at the terminal time, instead of the image content encoded in x1. To simulate the controlled dynamic in practice, we use CSD (Somepalli et al., 2024) as a style feature extractor and replace y1 with the style features extracted from the expected terminal state E[Xu 1 |Xu t
], as discussed in Appendix A.2.

Drift modulation through optimal controller: We then study a control problem where the velocity field is a linear combination of the state and the control variable. This problem is interesting to study because the reverse-SDE dynamic of the standard OU process has a drift field of the form:
v (Xt, t) = −Xt − 2∇ log p(Xt, t). For a Gaussian prior X0 ∼ N (0,I), the law of the OU process satisfies ∇ log p (Xt, t) = −Xt, and the corresponding drift field becomes v (Xt, t) = Xt. Our goal is to modulate this drift field using a controller u (Xu t, t). The result below provides the structure of the optimal control (again in the setting where the terminal objective is known; see Appendix A1).

Proposition 5.2. *Suppose* A ∈ R
k×d *be a linear style extractor that operates on the terminal state* Xu 1 ∈ R
d. Let pt *denote* ∇xV
∗(x, t) *in HJB equation (A.1). Given reference style features* y1, consider the control problem:

min u∈U  Z 1 t0 1 2 ku(Xu t, t)k 2dt + γ 2 kAXu 1 − y1k 2 2 , where dXu t = [Xu t + u(Xu t, t)] dt, Xu t0 = x0, Then, the optimal controller becomes u ∗(t) = −pt, where the instantaneous state Xu t = xt and pt satisfy the following coupled transitions: xt pt = x0e t − γ 2  AT(Ax1 − y1) e 1+t + γ 2  AT(Ax1 − y1) e 1−t γAT(Ax1 − y1) e 1−t .
Summary. We build on the connection between optimal control and reverse diffusion (see Appendices A.1-A.3 for details). The general strategy is to derive the optimal controller with known terminal state, and then replace the terminal state in the controller with its estimate using Tweedie's formula. For stylized models and Gaussian prior, the controllers have an explicit form. However in practice, the data distribution may not be Gaussian, and thus, we do not aim for a closed-form expression to modulate the drift. This line of analysis, however, points to our method RB-Modulation.

As discussed in §4, we incorporate a style descriptor in our controller's terminal cost and evaluate the resulting drift at each reverse time step either through back propagating through the score network (**Algorithm** 1), or an approximation based on proximal gradient updates (**Algorithm** 2).

## 6 Experiments

Metrics: Evaluating stylized synthesis is challenging due to the subjective nature of style, making simple metrics inadequate. We follow a two step approach: first using metrics from prior

| Human                     | Ours vs. InstantStyle   | Ours vs. StyleAligned   | Ours vs. IP-Adapter   |      |      |      |      |      |      |
|---------------------------|-------------------------|-------------------------|-----------------------|------|------|------|------|------|------|
| Preference (%)            | OQ ↑                    | SA ↑                    | PA ↑                  | OQ ↑ | SA ↑ | PA ↑ | OQ ↑ | SA ↑ | PA ↑ |
| Alternative               | 39.8                    | 38.5                    | 39.5                  | 24.4 | 27.8 | 29.4 | 8.1  | 20.1 | 8.3  |
| Tie                       | 9.3                     | 6.4                     | 7.3                   | 8.8  | 7.1  | 5.8  | 6.9  | 4.8  | 4.5  |
| RB-Modulation (ours) 51.0 | 55.1                    | 53.3                    | 66.9                  | 65.1 | 64.9 | 85.0 | 75.1 | 87.2 |      |

works and then conducting human evaluation. To evaluate prompt-image alignment, we use CLIP-T score (Hertz et al., 2023; Sohn et al., 2023; Wang et al., 2024a) and ImageReward (Xu et al., 2024), which also consider human aesthetics, distortions, and object completeness. When a style description is provided, CLIP-T and ImageReward also capture style alignment. We assess style similarity using DINO (Caron et al., 2021) and content similarity using CLIP-I (Radford et al., 2021) as in prior work (Hertz et al., 2023; Ruiz et al., 2023; Sohn et al., 2023), and highlight their limitations in disentangling style and content performance in evaluation. Given the importance of human evaluation in T2I personalization (Hertz et al., 2023; Sohn et al., 2023; Ruiz et al., 2023; Shah et al., 2023; Jeong et al., 2024), we also conduct a user study though Amazon Mechanical Turk to measure both style and text alignment. Datasets and baselines: We use style images from StyleAligned benchmark (Hertz et al., 2023) for stylization and content images from DreamBooth (Ruiz et al., 2023) for content-style composition. We base RB-Modulation on the recently released StableCascade (Pernias et al., 2024). We compare with three training-free methods: InstantStyle (Wang et al., 2024a) (state-of-the-art), IP-Adapter
(Ye et al., 2023), and StyleAligned (Hertz et al., 2023). For completeness, we also compare with training-based methods StyleDrop (Sohn et al., 2023) and ZipLoRA (Shah et al., 2023). Implementation details: All experiments run on a single A100 NVIDIA GPU. We use the same hyper-parameters for our method across tasks, and default settings for alternative methods as per their original papers. Details are provided in Appendix B.1.

## 6.1 Image Stylization

Qualitative analysis: This section describes image stylization experiments using a text prompt and a reference style image. Figure 2 compares our method with SoTA **training-free** InstantStyle (Wang et al., 2024a) and StyleAligned (Hertz et al., 2023), and **training-based** StyleDrop (Sohn et al., 2023). Except for StyleDrop, which requires ∼5 minutes of training per style, all methods, including ours, are training-free and complete inference in <1 minute. While all methods produce reasonable outputs, alternative methods encounter issues with information leakage. For instance, in the third row of Figure 2, StyleAligned and StyleDrop generate a wine bottle and book resembling the smartphone in the reference style image. In the last row, StyleAligned leaks the house and the background of the reference image; InstantStyle exhibits color leakage from the house, resulting in similar-colored images. Our method accurately adheres to the prompt in the desired style. As illustrated in the second and the third row, our method generates only one glass of wine and a highfidelity rubber duck, compared to baselines where extra items appear (wine bottles styled like the left smartphone) or incorrect styles (cartoon-style rubber duck). User study: Given the subjective nature of this field, we conduct a user study on Amazon Mechanical Turk with 155 participants using 100 styles from the StyleAligned dataset (Hertz et al., 2023), collecting a total of 7,200 answers (8 responses for each question). Each user answers 3 questions comparing our method with an alternative method regarding (1) overall quality, (2) style alignment, and (3) prompt alignment (details in the Appendix B.8). Table 1 summarizes the percentage of human preferences for our method, the alternative method, or a tie. Our method consistently outperforms the alternatives, including the current SoTA method InstantStyle (Wang et al., 2024a). The preference rates over all three metrics highlight the effectiveness of our method RB-Modulation.

Quantitative analysis: Table 2 evaluates 300 prompts and 100 styles on the StyleAligned dataset (Hertz et al., 2023) using three metrics, with and without style descriptions in the prompts. Our method outperforms others notably in the ImageReward metric, closely matching human aesthetics assessment from the user study in Table 1. In addition, the CLIP-T score indicates our effective alignment between generated images and text prompts. While IP-Adapter and StyleAligned

Reference style StableCascade DirectConcat AFA only SOC only Content prompt **AFA + SOC**
ìA catî ìA pianoî

Table 2: **Quantitative results for stylization:** We compare alternative methods on three metrics: ImageReward (Xu et al., 2024) and CLIP-T (Radford et al., 2021) for prompt alignment, DINO (Caron et al., 2021) for style alignment. Note that DINO score does not capture information leakage, so higher scores are not necessarily better (§B.5).

ImageReward ↑ CLIP-T score ↑ **DINO score**

With style description? No Yes No Yes No Yes IP-Adapter (Ye et al., 2023) -1.99 -1.51 0.21 0.26 0.89 0.89 StyleAligned (Hertz et al., 2023) -0.68 0.01 0.26 0.31 0.80 0.85 InstantStyle (Wang et al., 2024a) 0.09 0.72 0.29 0.33 0.68 0.72 RB-Modulation (ours) 0.91 1.18 0.30 0.34 0.68 0.73

have higher DINO scores, their lower rating in ImageReward, CLIP-T and user preference expose information leakage from the reference style images. Nevertheless, our DINO score remains competitive with the leading method InstantStyle. Notably, all metrics show improvement with style descriptions, particularly in ImageReward, where leveraging style descriptions enhances prompt alignment. Our method achieves high ImageReward and CLIP-T score even without style descriptions, suggesting robustness in prompt alignment without explicit style information in the prompt. Ablation Study: Figure 3 shows an ablation study of the AFA and SOC modules. We include a baseline, "DirectConcat", which concatenates reference style embeddings with text embeddings in the cross-attention modules. DirectConcat mixes both embeddings, making it less effective in disentangling style from prompts (e.g., cat vs. lighthouse). While AFA or SOC alone mitigates this by modulating the reverse drift and attention modules (§4), each has drawbacks. AFA alone fails to capture the cat's style accurately, and SOC alone misplaces elements, like "a lighthouse hat on the cat" and "a railroad trunk on a piano". We observe consistent improvements with each module, with the best results when combined.

## 6.2 Content-Style Composition

Since this paper primarily focuses on style-based personalization, we perform extensive experiments on stylization. To further demonstrate the versatility of our framework, we also explore content-style composition as an additional capability. Qualitative analysis: Content-style composition aims to preserve the essence of both content and style depicted in the reference images, while ensuring the resulting image aligns with a given text prompt. Figure 4 compares our method against **training-free** InstantStyle (Wang et al., 2024a), IP- Adapter (Ye et al., 2023), and **training-based** ZipLoRA (Shah et al., 2023). Notably, the trainingfree InstantStyle and IP-Adapter rely on ControlNet (Zhang et al., 2023), which often constrains their ability to accurately follow prompts for changing the pose of the generated content, such as illustrating "dancing" in Figure 4(b), or "walking" in (c). In contrast, our method avoids the need for ControlNet or adapters, and can effectively capture the distinctive attributes of both style and content images while adhering to the prompt to generate diverse images. In Figure 4(a), our method accurately captures elements like "table" and "river" that are overlooked in InstantStyle and IP- Adapter. In addition, our method mitigates information leakage, as evidenced in Figure 4(b), where the trunk of the tree behind the sloth is erroneously captured by InstantStyle and IP-Adapter but not

Ref. content Reference styles Reference styles Reference styles IP-
Ada pt er Instan tStyle Zi pL
oRA
Ou rs
(a) ìA dog dancing on a table near the riverî **(b)** ìA sloth walking on the streetî **(c)** ìA cat walkingî

Table 3: **Quantitative results for composition:** In addition to stylization metrics, we use CLIP-T score (Radford et al., 2021) to evaluate content alignment with the reference image. Similar to DINO, CLIP-I could inflate test score (Sohn et al., 2023; Shah et al., 2023) due to content leakage, but does not correlate to user preference; higher scores do not indicate better human preference.

ImageReward ↑ CLIP-T score ↑ **DINO score CLIP-I score**

IP-Adapter -0.78 0.22 0.73 0.68 InstantStyle -0.54 0.21 0.71 0.71 RB-Modulation (ours) 0.74 0.26 0.74 0.71

by ours. Compared to ZipLoRA (Shah et al., 2023) that requires training of 12 LoRAs (Hu et al.,
2021) and additional merge layers for each composition, our method requires no training at all while yielding competitive or better results. For instance, our method effectively captures the 2D cartoon and 3D rendering styles as illustrated in Figures 4(a) and (b). Quantitative analysis: Table 3 shows quantitative evaluation using 50 styles from StyleAligned dataset (Hertz et al., 2023) and 5 contents from DreamBooth dataset (Ruiz et al., 2023). Unlike prior works (Hertz et al., 2023; Sohn et al., 2023; Shah et al., 2023; Ruiz et al., 2023; Jeong et al., 2024) reporting either DINO and CLIP-I scores, we present both metrics and demonstrate comparable performance across them. Additionally, we obtain notably higher ImageReward score, which aligns closely with human aesthetics assessment as evidenced in §6.1 and (Xu et al., 2024). Consequently, we omitted a user study in this section. For more details, please refer to Appendix B.1.

## 7 Conclusion

We introduced Reference-Based modulation (RB-Modulation), a test-time optimization method for personalizing transformer-based diffusion models. RB-Modulation builds on concepts from stochastic optimal control to modulate the drift field of reverse diffusion dynamics, incorporating desired attributes (e.g., style or content) via a terminal cost. Our Attention Feature Aggregation (AFA) module decouples content and style in the cross-attention layers and enables precise control over both.

In addition, we derived theoretical connections between linear quadratic control and the denoising diffusion process, which led to the creation of RB-Modulation. Empirically, our method outperformed current state-of-the-art methods in stylization and content-style composition. To our best knowledge, this is the first training-free personalization framework using stochastic optimal control, which marks the departure from external adapters or ControlNets.

## 8 Broader Impact Statement

Social impact: Image stylization and content-style composition based on diffusion models potentially have both positive and negative social impact. This technology provides an easy-to-use tool to the general public for image generation which can help visualize their artistic ideas. On the other hand, our work on stylization and content-style composition poses a risk of generating arts that closely mimic or infringe upon existing copyrighted material, leading to legal and ethical issues. More broadly, our method inherits the risks from T2I models which are capable of generating fake contents that can be misused by malicious users. Safeguards: We build on StableCascade (Pernias et al., 2024), which has a mechanism to filter offensive image generations. Our framework RB-Modulation inherits these safeguards. In addition, to mitigate misuse, we believe it is crucial to ensure the underlying model's safety, which may involve (i) watermarking AI-generated artworks and (ii) implementing an NSFW filter to remove inappropriate contents. Reproducibility: The pseudocode and hyper-parameter details have been provided in the paper. The source code is available on the project page: https://rb-modulation.github.io/.

## Acknowledgments

This research has been supported by NSF Grant 2019844, a Google research collaboration award, and the UT Austin Machine Learning Lab. Litu Rout has been supported by Ju-Nam and Pearl Chew Presidential Fellowship and George J. Heuer Graduate Fellowship from UT Austin.

## References

Brian D.O. Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3):313–326, 1982.

Karl J. Astrom. *Introduction to Stochastic Control Theory*. Elsevier Science, 1971. Tamer Basar, Sean Meyn, and William R Perkins. Lecture notes on control system theory and design. *arXiv preprint arXiv:2007.01367*, 2020.

Julius Berner, Lorenz Richter, and Karen Ullrich. An optimal control perspective on diffusion-based generative modeling. *Transactions on Machine Learning Research*, 2024. ISSN 2835-8856. URL
https://openreview.net/forum?id=oYIjw37pTP.

Rene Carmona, Franc¸ois Delarue, et al. ´ *Probabilistic theory of mean field games with applications* I-II. Springer, 2018.

Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J ´ egou, Julien Mairal, Piotr Bojanowski, and ´
Armand Joulin. Emerging properties in self-supervised vision transformers. In *Proceedings of* the IEEE/CVF international conference on computer vision, pp. 9650–9660, 2021.

Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan ´
Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. In Proceedings of the 40th International Conference on Machine Learning, pp. 4055–4075, 2023.

Pratik Chaudhari, Adam Oberman, Stanley Osher, Stefano Soatto, and Guillaume Carlier. Deep relaxation: partial differential equations for optimizing deep neural networks. Research in the Mathematical Sciences, 5:1–30, 2018.

Tianrong Chen, Jiatao Gu, Laurent Dinh, Evangelos Theodorou, Joshua M Susskind, and Shuangfei Zhai. Generative modeling with phase stochastic bridge. In The Twelfth International Conference on Learning Representations, 2023.

Wenhu Chen, Hexiang Hu, Yandong Li, Nataniel Ruiz, Xuhui Jia, Ming-Wei Chang, and William W
Cohen. Subject-driven text-to-image generation via apprenticeship learning. *Advances in Neural* Information Processing Systems, 36, 2024.

Hyungjin Chung, Jong Chul Ye, Peyman Milanfar, and Mauricio Delbracio. Prompt-tuning latent diffusion models for inverse problems. *arXiv preprint arXiv:2310.01110*, 2023.

Mauricio Delbracio and Peyman Milanfar. Inversion by direct iteration: An alternative to denoising diffusion for image restoration. *Transactions on Machine Learning Research*, 2023.

Bradley Efron. Tweedie's formula and selection bias. Journal of the American Statistical Association, 106(496):1602–1614, 2011.

Martin Nicolas Everaert, Marco Bocchio, Sami Arpa, Sabine Susstrunk, and Radhakrishna Achanta. ¨
Diffusion in style. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 2251–2261, 2023.

Wendell H Fleming and Raymond W Rishel. *Deterministic and stochastic optimal control*, volume 1. Springer Science & Business Media, 2012.

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. *arXiv preprint arXiv:2208.01618*, 2022.

Zinan Guo, Yanze Wu, Zhuowei Chen, Lang Chen, and Qian He. Pulid: Pure and lightning id customization via contrastive alignment. *arXiv preprint arXiv:2404.16022*, 2024.

Yutong He, Naoki Murata, Chieh-Hsin Lai, Yuhta Takida, Toshimitsu Uesaka, Dongjun Kim, Wei-
Hsiang Liao, Yuki Mitsufuji, J Zico Kolter, Ruslan Salakhutdinov, and Stefano Ermon. Manifold preserving guided diffusion. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=o3BxOLoxm1.

Amir Hertz, Andrey Voynov, Shlomi Fruchter, and Daniel Cohen-Or. Style aligned image generation via shared attention. *arXiv preprint arXiv:2312.02133*, 2023.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. *Advances in* Neural Information Processing Systems, 33:6840–6851, 2020.

Lars Holdijk, Yuanqi Du, Ferry Hooft, Priyank Jaini, Berend Ensing, and Max Welling. Stochastic optimal control for collective variable free sampling of molecular transition paths. Advances in Neural Information Processing Systems, 36, 2024.

Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2021.

Jiehui Huang, Xiao Dong, Wenhui Song, Hanhui Li, Jun Zhou, Yuhao Cheng, Shutao Liao, Long Chen, Yiqiang Yan, Shengcai Liao, et al. Consistentid: Portrait generation with multimodal finegrained identity preserving. *arXiv preprint arXiv:2404.16771*, 2024a.

Jiehui Huang, Xiao Dong, Wenhui Song, Hanhui Li, Jun Zhou, Yuhao Cheng, Shutao Liao, Long Chen, Yiqiang Yan, Shengcai Liao, et al. Consistentid: Portrait generation with multimodal finegrained identity preserving. *arXiv preprint arXiv:2404.16771*, 2024b.

Xun Huang and Serge Belongie. Arbitrary style transfer in real-time with adaptive instance normalization. In *Proceedings of the IEEE international conference on computer vision*, pp. 1501–1510, 2017.

Aapo Hyvarinen and Peter Dayan. Estimation of non-normalized statistical models by score match- ¨
ing. *Journal of Machine Learning Research*, 6(4), 2005.

Jaeseok Jeong, Junho Kim, Yunjey Choi, Gayoung Lee, and Youngjung Uh. Visual style prompting with swapping self-attention. *arXiv preprint arXiv:2402.12974*, 2024.

HJ Kappen. Stochastic optimal control theory. ICML, Helsinki, Radbound University, Nijmegen, Netherlands, 2008.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. *Advances in Neural Information Processing Systems*, 35:26565–26577, 2022.

Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1931–1941, 2023.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. *arXiv preprint arXiv:2210.02747*, 2022.

Xingchao Liu, Chengyue Gong, et al. Flow straight and fast: Learning to generate and transfer data with rectified flow. In *The Eleventh International Conference on Learning Representations*, 2022.

Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In *Proceedings of the IEEE/CVF Conference* on Computer Vision and Pattern Recognition, pp. 6038–6047, 2023.

Tim Pearce, Tabish Rashid, Anssi Kanervisto, Dave Bignell, Mingfei Sun, Raluca Georgescu, Sergio Valcarcel Macua, Shan Zheng Tan, Ida Momennejad, Katja Hofmann, and Sam Devlin. Imitating human behaviour with diffusion models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id= Pv1GPQzRrC8.

Pablo Pernias, Dominic Rampas, Mats Leon Richter, Christopher Pal, and Marc Aubreville.

Wurstchen: An efficient architecture for large-scale text-to-image diffusion models. In ¨ The Twelfth International Conference on Learning Representations, 2024. URL https:// openreview.net/forum?id=gU58d5QeGv.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Muller, Joe ¨
Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In *The Twelfth International Conference on Learning Representations*, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *International conference on machine learning*, pp. 8748–8763. PMLR, 2021.

Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In *International Conference on Machine* Learning, pp. 8821–8831. PMLR, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High- ¨
resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10684–10695, 2022.

Litu Rout, Advait Parulekar, Constantine Caramanis, and Sanjay Shakkottai. A theoretical justification for image inpainting using denoising diffusion probabilistic models. *arXiv preprint* arXiv:2302.01217, 2023a.

Litu Rout, Negin Raoof, Giannis Daras, Constantine Caramanis, Alexandros G Dimakis, and Sanjay Shakkottai. Solving inverse problems provably via posterior sampling with latent diffusion models. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023b. URL https://openreview.net/forum?id=XKBFdYwfRo.

Litu Rout, Yujia Chen, Abhishek Kumar, Constantine Caramanis, Sanjay Shakkottai, and Wen-
Sheng Chu. Beyond first-order tweedie: Solving inverse problems using latent diffusion. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman.

Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22500– 22510, 2023.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models. In *Proceedings of the IEEE/CVF Conference on Computer Vision* and Pattern Recognition, pp. 6527–6536, 2024.

Dan Ruta, Gemma Canet Tarres, Andrew Gilbert, Eli Shechtman, Nicholas Kolkin, and John Col- ´
lomosse. Diff-nst: Diffusion interleaving for deformable neural style transfer. arXiv preprint arXiv:2307.04157, 2023.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models with deep language understanding. arXiv preprint arXiv:2205.11487, 2022.

Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, and Varun Jampani. ZipLoRA: Any subject in any style by effectively merging loras. arXiv preprint arXiv:2311.13600, 2023.

Kihyuk Sohn, Nataniel Ruiz, Kimin Lee, Daniel Castro Chin, Irina Blok, Huiwen Chang, Jarred Barber, Lu Jiang, Glenn Entis, Yuanzhen Li, et al. Styledrop: Text-to-image generation in any style. In *37th Conference on Neural Information Processing Systems (NeurIPS)*. Neural Information Processing Systems Foundation, 2023.

Gowthami Somepalli, Anubhav Gupta, Kamal Gupta, Shramay Palta, Micah Goldblum, Jonas Geiping, Abhinav Shrivastava, and Tom Goldstein. Measuring style similarity in diffusion models. arXiv preprint arXiv:2404.01292, 2024.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021a. URL https://openreview.net/ forum?id=St1giarCHLP.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021b. URL https://openreview.net/ forum?id=PxTIG12RRHS.

Luming Tang, Nataniel Ruiz, Qinghao Chu, Yuanzhen Li, Aleksander Holynski, David E Jacobs, Bharath Hariharan, Yael Pritch, Neal Wadhwa, Kfir Aberman, et al. Realfill: Reference-driven generation for authentic image completion. *arXiv preprint arXiv:2309.16668*, 2023.

Gemma Canet Tarres, Dan Ruta, Tu Bui, and John Collomosse. Parasol: Parametric style control ´
for diffusion image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2432–2442, 2024.

Yoad Tewel, Rinon Gal, Gal Chechik, and Yuval Atzmon. Key-locked rank one editing for text-toimage personalization. In *ACM SIGGRAPH 2023 Conference Proceedings*, pp. 1–11, 2023.

Evangelos Theodorou, Freek Stulp, Jonas Buchli, and Stefan Schaal. An iterative path integral stochastic optimal control approach for learning robotic tasks. *IFAC Proceedings Volumes*, 44(1): 11594–11601, 2011.

Belinda Tzen and Maxim Raginsky. Theoretical guarantees for sampling and inference in generative models with latent diffusions. In *Conference on Learning Theory*, pp. 3084–3114. PMLR, 2019.

Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. Deep image prior. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018.

Pascal Vincent. A connection between score matching and denoising autoencoders. Neural computation, 23(7):1661–1674, 2011.

Haofan Wang, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. Instantstyle: Free lunch towards style-preserving in text-to-image generation. *arXiv preprint arXiv:2404.02733*, 2024a.

Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, and Anthony Chen. Instantid: Zero-shot identitypreserving generation in seconds. *arXiv preprint arXiv:2401.07519*, 2024b.

Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, and Anthony Chen. Instantid: Zero-shot identitypreserving generation in seconds. *arXiv preprint arXiv:2401.07519*, 2024c.

Qiucheng Wu, Yujian Liu, Handong Zhao, Ajinkya Kale, Trung Bui, Tong Yu, Zhe Lin, Yang Zhang, and Shiyu Chang. Uncovering the disentanglement capability in text-to-image diffusion models. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 1900–1910, 2023.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36, 2024.

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. *arXiv preprint arXiv:2308.06721*, 2023.

Jiwen Yu, Yinhuai Wang, Chen Zhao, Bernard Ghanem, and Jian Zhang. Freedom: Training-free energy-guided conditional diffusion model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 23174–23184, 2023.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 3836–3847, 2023.

Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator.

In *The Eleventh International Conference on Learning Representations*, 2022.

Yuanzhi Zhu, Kai Zhang, Jingyun Liang, Jiezhang Cao, Bihan Wen, Radu Timofte, and Luc Van Gool. Denoising diffusion models for plug-and-play image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1219–1229, 2023.

## A Additional Theoretical Results

In this section, we restate the propositions more precisely and provide their technical proofs. First, we recall standard terminologies from optimal control literature (Fleming & Rishel, 2012). For 0 ≤ t0 ≤ t ≤ tN ≤ 1, the cost function associated with the controller u(·) is defined by the integral:

$$V(u;{\bf x}_{0},t_{0})=\int_{t_{0}}^{t_{N}}\ell\left(X_{t}^{u},u,t\right)dt+h\left(X_{t_{N}}^{u}\right),\ \ X_{t_{0}}^{u}={\bf x}_{0},\tag{4}$$

where `(*· · ·*) denotes a scalar valued function of the state Xu t, controller u(·), and instantaneous time t. The value function V
∗(x0, t0) is defined as the minimum value of V (u; x0, t0) over the set

of admissible controllers $\mathcal{U}$, i.e.,  $$V^{*}=V^{*}(\mathbf{x}_{0},t_{0})=\min_{u\in\mathcal{U}}V(u;\mathbf{x}_{0},t_{0})=\min_{u\in\mathcal{U}}\int_{t_{0}}^{t_{N}}\ell\left(X_{t}^{u},u,t\right)dt+h\left(X_{t_{N}}^{u}\right),\ \ X_{t_{0}}^{u}=\mathbf{x}_{0},\tag{5}$$  which satisfies a Partial Differential Equation (PDE) given below in **Theorem**$\mathbb{A}_{\cdot}$.  
Theorem A.1 (HJB Equation, (Fleming & Rishel, 2012; Basar et al., 2020)). If V
∗ *has continuous* partial derivatives, then it must satisfy the following PDE, also known as Hamilton-Jacobi-Bellman (HJB) equation:

_(HD) equation:_  $$-\frac{\partial V^{*}}{\partial t}\left(\mathbf{x},t\right)=\min_{u\in\mathcal{U}}\left[H\left(\mathbf{x},\nabla_{\mathbf{x}}V^{*}\left(\mathbf{x},t\right),u,t\right)\coloneqq\ell\left(\mathbf{x},u,t\right)+\left(\nabla_{\mathbf{x}}V^{*}\left(\mathbf{x},t\right)\right)^{T}v\left(\mathbf{x},u,t\right)\right].$$  _Also, the Hamiltonian $H\left(\mathbf{x},\nabla_{\mathbf{x}}V^{*}\left(\mathbf{x},t\right),u,t\right)$, optimal controller $u^{*}(t)$ and the state trajectory $\mathbf{x}^{*}(t)$ must satisfy_
$$\operatorname*{min}_{u\in\mathcal{U}}H\left(\mathbf{x}^{*}(t),\nabla_{\mathbf{x}}V^{*}\left(\mathbf{x}^{*}(t),t\right),u,t\right)=H\left(\mathbf{x}^{*}(t),\nabla_{\mathbf{x}}V^{*}\left(\mathbf{x}^{*}(t),t\right),u^{*}(t),t\right).$$
A.1 INTERPRETING REVERSE-SDE AS A SOLUTION TO OPTIMAL CONTROL For clarity, we restate the problem setup here and describe the main ideas from §4 in more details. Problem setup: We discuss a standard approach to derive the optimal controller in a special case of our control problem (2). We substitute t ← 1 − t to account for the time reversal in the reverse-
SDE (1). In this setup, Xu 0 ∼ N (0,Id) and Xu 1 ∼ p*data*. We consider the following dynamic without the Brownian motion:

$$\mathrm{d}X_{t}^{u}=v(X_{t}^{u},u,t)\mathrm{d}t,\quad X_{t_{0}}^{u}=\mathbf{x}_{0},$$
$$(6)$$
= x0, (6)
where 0 ≤ t0 ≤ t ≤ tN ≤ 1 and v : R
d × R
d × [t0, tN ] → R
d denotes the drift field. The optimal controller u
∗can be derived by solving the Hamilton-Jacobi-Bellman (HJB) equation (Fleming &
Rishel, 2012; Basar et al., 2020), see Appendix A for details. By certainty equivalence (when the drift and diffusion coefficients are linear time-varying (Astrom, 1971), which occurs when p*data* is Gaussian; see also discussion in Section A.3), the same u
∗applies to a more general case with the Brownian motion (Chen et al., 2023), where

$$\mathrm{d}X_{t}^{u}=v(X_{t}^{u},u,t)\mathrm{d}t+\mathrm{d}W_{t},\quad X_{t_{0}}^{u}=\mathbf{x}_{0}.$$

t0 = x0. (7)
Therefore, we analyze the reverse dynamic in the absence of the Brownian motion, and employ the same controller in more general cases with the Brownian motion. Below, we consider a dynamical system whose drift field is chosen to minimize a transient trajectory cost and a terminal cost (weighted by γ) that enforces "closeness" to reference content x1. Proposition A.2 provides the structure of the optimal control in the limiting setting where γ → ∞. Furthermore, suppose we replace x1 with its conditional expectation (discussed in Remark A.3), the resulting dynamic, interestingly, is the standard reverse-SDE for the Orstein-Uhlenbeck (OU) diffusion process. This connection between optimal control (more precisely, classic Linear Quadratic Control) and the standard reverse-SDE provides us a path to study other diffusion problems (e.g. personalization (Ruiz et al., 2023; Hertz et al., 2023; Sohn et al., 2023; Wang et al., 2024a), image editing or inversion (Mokady et al., 2023; Delbracio & Milanfar, 2023; Rout et al., 2023b; 2024; 2023a)) through the lens of stochastic optimal control.

$$(T)$$

Proposition A.2 (Linear optimal control with quadratic cost (Chen et al., 2023)). Consider the control problem:

$$\operatorname*{min}_{u\in\mathcal{U}}\int_{t_{0}}^{1}{\frac{1}{2}}\left\|u(X_{t}^{u},t)\right\|^{2}d t+{\frac{\gamma}{2}}\left\|X_{1}^{u}-x_{1}\right\|_{2}^{2},$$  _where $\operatorname{d}\!X_{t}^{u}=u(X_{t}^{u},t)\operatorname{d}\!t,\quad X_{t_{0}}^{u}=x_{0}$_
Then, in the limit when γ → ∞*, the optimal controller is given by* u
∗ =
x1−Xu t 1−t*, which yields* dXu t =
x1−Xu t 1−tdt *for the deterministic case and* dXu t =
x1−Xu t 1−tdt + dWt *for the stochastic case.*
The optimal controller for the problem presented in Proposition A.2 can be derived using established techniques from control theory (Fleming & Rishel, 2012; Basar et al., 2020; Kappen, 2008); the specific form of the above result follows from (Chen et al., 2023) (but without their momentum term). The key steps in this derivation include: (1) computing the Hamiltonian, (2) applying the minimum principle theorem to derive a set of differential equations, and (3) taking the limit as γ → ∞. These three steps are fundamental in deriving a closed-form solution. The final step is critical for satisfying hard terminal constraint and is essential for the practical implementation of Algorithm 1 and **Algorithm** 2, as detailed in §4. For generative modeling, the controlled dynamics described in Proposition A.2 cannot be directly applied. This limitation arises because the optimal control u
∗ depends on the terminal state x1, making it non-causal or reliant on future information. Inspired by recent advancements in flow-based generative models (Lipman et al., 2022; Liu et al., 2022), we make the optimal controller causal by replacing the terminal state with its conditional expectation given the current state, i.e., , i.e. x1 ←
E[Xu 1 |Xu t = xt]. This modification results in a controlled dynamic that can be simulated to produce a generative model incorporating principles from optimal control, as elaborated in Remark A.3. Remark A.3 (Connections between diffusion-based generative modeling and stochastic optimal control). Following conditional diffusion models and optimal transport paths (Lipman et al., *2022;* Liu et al., *2022), where* X
f t = tXf0 + (1 − t)*, the state variable* Xu t*is equal in distribution to* X
f 1−t = (1 − t)X
f 0 + t,  ∼ N (0,Id) after time reversal. Now, we use Tweedie's formula (Efron, 2011) to compute the posterior mean:

$$\mathbb{E}[X_{1}^{u}|X_{t}^{u}]=\frac{X_{t}^{u}}{1-t}+\frac{t^{2}}{1-t}\nabla\log p\left(X_{t}^{u},1-t\right).\tag{8}$$

Substituting the posterior mean in the controlled reverse dynamic of Proposition A.2*, we arrive at*

$$\mathrm{d}X_{t}^{u}=\frac{(\mathbb{E}[X_{1}^{u}|X_{t}^{u}]-X_{t}^{u})}{(1-t)}\mathrm{d}t+\mathrm{d}W_{t}$$ $$=\Big{[}\frac{t}{(1-t)^{2}}X_{t}^{u}+\frac{t^{2}}{(1-t)^{2}}\nabla\log p(X_{t}^{u},1-t)\Big{]}\mathrm{d}t+\mathrm{d}W_{t}.$$

We observe that the above equation is structurally the same as reverse-SDE associated with a forward Orstein-Uhlenbeck (OU) diffusion process. This relation between diffusion-based generative models and optimal control is further explored in the Appendices below. Indeed, diffusion models (Ho et al., 2020; Song et al., 2021b; Rombach et al., 2022; Podell et al., 2023; Pernias et al., 2024) provide an effective approximation to the terminal state of a denoising process. This approximation has been used for a variety of generative modeling tasks. Also, the terminal state can be approximated using Tweedie's formula (Efron, 2011) with a learned score function (Ho et al., 2020)
1. By utilizing these pre-trained diffusion models, we can employ the connection to optimal control as discussed above to develop practically implementable generative models that incorporates terminal objectives such as style and personalization. Consequently, the subsequent sections are dedicated to deriving the optimal controller assuming a known terminal state; we will approximate this in practice using Tweedie's formula as above.

## A.2 Incorporating Personalized Style Constraints Through A Terminal Cost

In this section, we derive the optimal controller when we have access to the reference *style features* y1 at the terminal time (instead of the content of the image encoded through x1).

Proposition A.4. *Suppose* A ∈ R
k×d *be a linear style extractor that operates on the terminal state* Xu 1 ∈ R
d. Given reference style features y1*, consider the control problem:*

$$\min_{u\in\mathcal{U}}\int_{t_{0}}^{1}\frac{1}{2}\left\|u(X_{t}^{u},t)\right\|^{2}dt+\frac{\gamma}{2}\left\|AX_{1}^{u}-y_{1}\right\|_{2}^{2},\tag{9}$$  _where $\mathrm{d}X_{t}^{u}=u(X_{t}^{u},t)\,\mathrm{d}t,\quad\ X_{t_{0}}^{u}=x_{0},$_  $$\left(\begin{array}{cc}\mathcal{T}&\mathcal{T}\\ \end{array}\right)^{-1}\mathcal{T}.$$

Then, in the limit when γ → ∞*, the optimal controller* u
∗ =
$\underline{\left({A}^{T}A\right)}^{-1}{A}^{T}\left(y_{1}-A X_{t}^{u}\right)$, $\underline{W}$. 
1−t*, which yields*
the following controlled dynamic:
$$\mathrm{d}X_{t}^{u}=\frac{\left(A^{T}A\right)^{-1}A^{T}\left(y_{1}-AX_{t}^{u}\right)}{1-t}\mathrm{d}t.\tag{11}$$
Proof. We derive the closed-form solution of the optimal controller given a fixed terminal state condition. This is similar to (Chen et al., 2023), where the reverse process is accelerated using momentum (see also (Kappen, 2008; Basar et al., 2020) for further details on this approach). The distinction, however, lies in the treatment of the terminal constraint. For completeness, we provide full details of the proof below.

To derive the closed-form solution2, recall from equation (5) that `(xt, ut, t) = 12 kutk 2and the terminal cost h(x1) = γ2 kAx1 − y1k 2. Let pt represent ∇xV
∗(x, t) in Theorem A.1. Then, the Hamiltonian of the control problem (9) is given by

$$H(\mathbf{x}_{t},\mathbf{p}_{t},\mathbf{u}_{t},t)=\ell(\mathbf{x}_{t},\mathbf{u}_{t},t)+\mathbf{p}_{t}^{T}\mathbf{u}_{t}$$ $$={\frac{1}{2}}\left\|\mathbf{u}_{t}\right\|^{2}+\mathbf{p}_{t}^{T}\mathbf{u}_{t}.$$
$$(12)$$
$$(13)$$
$$(14)$$
$$(15)$$
$$(16)$$

Since the minimizer of the Hamiltonian is u
∗
t = −pt, the value function becomes

$$V^{*}=\operatorname*{min}_{\mathbf{u}_{t}}H(\mathbf{u}_{t},\mathbf{p}_{t},\mathbf{u}_{t},t)=H(\mathbf{u}_{t},\mathbf{p}_{t},\mathbf{u}_{t}^{*},t)=-{\frac{1}{2}}\left\|\mathbf{p}_{t}\right\|^{2}.$$

Integrating both sides of (13), we have

$$\int_{t_{0}}^{1}\mathrm{d}\mathbf{x}_{t}=-\int_{t_{0}}^{1}\mathbf{p}_{t}\mathrm{d}t=-\mathbf{p}\left(1-t_{0}\right),$$

where the last equality is due to (14), which states that pt is a constant independent of time t. This implies x1 = xt0 − p(1 − t0). From (16), we know for tN = 1 that

$$\mathbf{p}_{1}=\gamma A^{T}\left(A\mathbf{x}_{1}-y_{1}\right)$$ $$=\gamma\left(A^{T}A\left(x_{0}-\mathbf{p}(1-t_{0})\right)-A^{T}y_{1}\right)$$ $$=\gamma A^{T}A x_{0}-\gamma A^{T}A\mathbf{p}_{1}(1-t_{0})-\gamma A^{T}y_{1}$$

2With slight abuse of notation, we use xt to denote X
u t and ut to denote u(X
u t, t) in the deterministic case.

Now, we use minimum principle theorem (Basar et al., 2020) to obtain the following set of differential equations:
$$\begin{array}{l}{{\frac{\mathrm{d}\mathbf{x}_{t}}{\mathrm{d}t}=\nabla_{\mathbf{p}}H\left(\mathbf{x}_{t},\mathbf{p}_{t},\mathbf{u}_{t}^{*},t\right)=-\mathbf{p}_{t};}}\\ {{\frac{\mathrm{d}\mathbf{p}_{t}}{\mathrm{d}t}=-\nabla_{\mathbf{x}}H\left(\mathbf{x}_{t},\mathbf{p}_{t},\mathbf{u}_{t}^{*},t\right)=0;}}\\ {{\mathbf{x}_{t_{0}}=x_{0};}}\\ {{\mathbf{p}_{t_{N}}=\nabla_{\mathbf{x}}h\left(\mathbf{x}_{t_{N}},t_{N}\right)=\gamma A^{T}\left(A\mathbf{x}_{t_{N}}-y_{1}\right).}}\end{array}$$ (1) we have 
$$(17)$$
$$(18)$$
Rearranging (18) and solving for p1, we get p1 = γI + γAT A (1 − t0)−1 A T Ax0 − A Ty1  = I γ + A T A (1 − t0) −1A T Ax0 − A Ty1 = p (19) Passing (19) through the limit γ → ∞, we get lim γ→∞ p = AT A−1 AT Ax0 − AT y1 1 − t0. (20) Therefore, the optimal control becomes u ∗ t = −p = − (A T A) −1(A T Axt−A T y1) 1−t, and the resulting
dynamical system is given by
$$\mathrm{d}\mathbf{x}_{t}={\frac{\left(A^{T}A\right)^{-1}A^{T}\left(y_{1}-A\mathbf{x}_{t}\right)}{1-t}}\mathrm{d}t,$$
for the deterministic process and
$$\mathrm{d}\mathbf{x}_{t}={\frac{\left(A^{T}A\right)^{-1}A^{T}\left(y_{1}-A\mathbf{x}_{t}\right)}{1-t}}\mathrm{d}t+\mathrm{d}W_{t},$$
for the stochastic process with the Brownian motion. This completes the statement of the proof.
Implications: The optimal controller depends on the reference *style features* y1 at the terminal time (instead of the image content x1 as in Appendix A.1). The reverse dynamic can be simulated in practice by using CSD (Somepalli et al., 2024) as a style feature extractor and replacing y1 with the extracted style features from the expected terminal state E[Xu 1 |Xu t
], as discussed in Remark A.3.

This makes the controller drift causal and non-anticipating future information

## A.3 Incorporating Style Through Modulation And A Terminal Cost

In this section, we study a control problem where the velocity field is a linear combination of the state and the control variable. This problem is interesting to study because of the following reason. The reverse-SDE dynamic of the standard OU process has a drift field of the form:
v (Xt, t) = −Xt − 2∇ log p(Xt, t).

For a Gaussian prior X0 ∼ N (0,I), the law of the OU process satisfies ∇ log p (Xt, t) = −Xt, and the corresponding drift field becomes v (Xt, t) = Xt. Our goal is to modulate this drift field using a controller u (Xu t, t). The result below provides the structure of the optimal control (again in the setting where the terminal objective is known; see Appendix A1).

Proposition A.5. *Suppose* A ∈ R
k×d *be a linear style extractor that operates on the terminal state* Xu 1 ∈ R
d. Let pt *denote* ∇xV
∗(x, t) *in HJB equation (A.1). Given reference style features* y1, consider the control problem:

$$\min_{u\in\mathcal{U}}\int_{t_{0}}^{1}\frac{1}{2}\left\|u(X_{t}^{u},t)\right\|^{2}dt+\frac{\gamma}{2}\left\|AX_{1}^{u}-y_{1}\right\|_{2}^{2},\tag{21}$$  _where $\mathrm{d}X_{t}^{u}=[X_{t}^{u}+u(X_{t}^{u},t)]\,\mathrm{d}t,\quad X_{t_{0}}^{u}=x_{0},$_ (22)  _iller becomes $u^{*}(t)=-\mathbf{p}_{t}$, where the instantaneous state $X_{t}^{u}=\mathbf{x}_{t}$ and $\mathbf{p}_{t}$._
Then, the optimal controller becomes u
satisfy the following:
_wing_,  $$\begin{bmatrix}\mathbf{x_{t}}\\ \mathbf{p_{t}}\end{bmatrix}=\begin{bmatrix}x_{0}e^{t}-\frac{\gamma}{2}A^{T}\left(A\mathbf{x_{1}}-y_{1}\right)e^{1+t}+\frac{\gamma}{2}A^{T}\left(A\mathbf{x_{1}}-y_{1}\right)e^{1-t}\\ \gamma A^{T}\left(A\mathbf{x_{1}}-y_{1}\right)e^{1-t}\end{bmatrix}.$$

Proof. The proof of Proposition A.5 is similar to Proposition A.4. One key distinction is the set of differential equations obtained using minimum principle theorem (Basar et al., 2020). We begin with the Hamiltonian:

$$\begin{array}{c}{{H(\mathbf{x}_{t},\mathbf{p}_{t},\mathbf{u}_{t},t)=\ell(\mathbf{x}_{t},\mathbf{u}_{t},t)+\mathbf{p}_{t}^{T}\left(\mathbf{u}_{t}+\mathbf{x}_{t}\right)}}\\ {{=\frac{1}{2}\left\|\mathbf{u}_{t}\right\|^{2}+\mathbf{p}_{t}^{T}\mathbf{u}_{t}+\mathbf{p}_{t}^{T}\mathbf{x}_{t},}}\end{array}$$

which gives us the minimizer of the Hamiltonian u
∗
t = −pt and its value function becomes V
∗ = minut H(ut, pt, ut, t) = H(ut, pt, u
∗
t, t) = −
1 2 kptk 2 + p T
t xt. By the minimum principle theorem (Basar et al., 2020),

x˙ t:= dxt dt= ∇pH (xt, pt, u ∗ t, t) = −pt + xt; (23) p˙ t:= dpt dt= −∇xH (xt, pt, u ∗ t, t) = −pt; (24) xt0 = x0; (25) ptN = ∇xh (xtN , tN ) = γAT(AxtN − y1). (26) This leads to a coupled system of differential equations with boundary conditions as given below:
$$(23)$$
(24)  $\text{(25)}$  . 
This leads to a coupled system of differential equations with boundary conditions as given below: x˙ t p˙ t = 1 −1 0 −1  xt pt ; xt0 = x0; p1 = γAT(Ax1 − y1). This can be solved numerically using ODE solvers, see (Fleming & Rishel, 2012; Basar et al., 2020) for details. Denote q˙t = x˙ t p˙ t and M = 1 −1 0 −1 . We seek a solution of the form q(t) = qe λt. If q(t) is a solution of the above problem, then it must satisfy the following eigen value problem: qe λtλ = Mqe λt. (27) Writing the characteristic polynomial of (27), we get det (M − λI) = 0, which gives the eigen
$$(26)$$
values λ = {1, −1}. Substituting these eigen values, we have
$$\begin{bmatrix}0&-1\\ 0&-2\end{bmatrix}\begin{bmatrix}q_{1}\\ q_{2}\end{bmatrix}=\mathbf{0},\quad\begin{bmatrix}2&-1\\ 0&0\end{bmatrix}\begin{bmatrix}q_{1}\\ q_{2}\end{bmatrix}=\mathbf{0},$$  which gives two fundamental solutions. By combining these two, we obtain the final solution  $$\begin{bmatrix}\mathbf{x}_{t}\\ \mathbf{p}_{t}\end{bmatrix}=\omega\begin{bmatrix}1\\ 0\end{bmatrix}e^{t}+\xi\begin{bmatrix}1\\ 2\end{bmatrix}e^{-t},$$  where $\omega$ and $\xi$ can be found using the boundary conditions. Since $\mathbf{x}_{0}=x_{0}$ and $\mathbf{p}_{t}$
γAT(Ax1 − y1), we get ω = x0 −
γ
2 AT(Ax1 − y1) e and ξ =
γ
2 AT(Ax1 − y1) e. Substituting the values of $\omega$ and $\xi$, we arrive at  $$\begin{bmatrix}\mathbf{x}_{t}\\ \mathbf{p}_{t}\end{bmatrix}=\begin{bmatrix}x_{0}e^{t}-\frac{\gamma}{2}A^{T}\left(A\mathbf{x}_{1}-y_{1}\right)e^{1+t}+\frac{\gamma}{2}A^{T}\left(A\mathbf{x}_{1}-y_{1}\right)e^{1-t}\\ \gamma A^{T}\left(A\mathbf{x}_{1}-y_{1}\right)e^{1-t}\end{bmatrix}.$$
This completes the proof of the proposition. Summary: Though Appendices A.1-A.3, we have seen the connection between optimal control and diffusion based generation with a personalized terminal constraint. The general strategy has been to derive the optimal controller with known terminal state, and then replace the terminal state in the controller with its estimate using Tweedie's formula. While the controllers so far have an explicit form, in practice, the data distribution is not Gaussian, and thus, we do not have a closed-form expression for the drift of the controller. This line of analysis, however, points to our method RB-Modulation. As discussed in §4, we incorporate a contrastive style descriptor in our controller's terminal cost and numerically evaluate the drift of the controller at each reverse time step either through back propagation through the score network, or an approximation based on proximal gradient updates.

## B Additional Experiments

In this section, we provide implementation details and additional experimental evaluation which have been omitted from the main draft due to limited space.