# RB-MODULATION: TRAINING-FREE STYLIZATION USING REFERENCE-BASED MODULATION

Litu Rout<sup>1</sup>,2<sup>∗</sup> Yujia Chen<sup>1</sup> Nataniel Ruiz<sup>1</sup>

Abhishek Kumar<sup>3</sup> Constantine Caramanis<sup>2</sup> Sanjay Shakkottai<sup>2</sup> Wen-Sheng Chu<sup>1</sup> <sup>1</sup> Google <sup>2</sup> UT Austin <sup>3</sup> Google DeepMind

{litu.rout,constantine,sanjay.shakkottai}@utexas.edu

{liturout,yujiachen,natanielruiz,abhishk,wschu}@google.com

## ABSTRACT

We propose Reference-Based Modulation (RB-Modulation), a new plug-and-play solution for training-free personalization of diffusion models. Existing trainingfree approaches exhibit difficulties in (a) style extraction from reference images in the absence of additional style or content text descriptions, (b) unwanted content leakage from reference style images, and (c) effective composition of style and content. RB-Modulation is built on a novel stochastic optimal controller where a style descriptor encodes the desired attributes through a terminal cost. The resulting drift not only overcomes the difficulties above, but also ensures high fidelity to the reference style and adheres to the given text prompt. We also introduce a cross-attention-based feature aggregation scheme that allows RB-Modulation to decouple content and style from the reference image. With theoretical justification and empirical evidence, our test-time optimization framework demonstrates precise extraction and control of *content* and *style* in a training-free manner. Further, our method allows a seamless composition of content and style, which marks a departure from the dependency on external adapters or ControlNets. See project page <https://rb-modulation.github.io/> for code and further details.

# 1 INTRODUCTION

Text-to-image (T2I) generative models [\(Ramesh et al.,](#page-12-0) [2021;](#page-12-0) [Rombach et al.,](#page-12-1) [2022;](#page-12-1) [Saharia et al.,](#page-13-0) [2022\)](#page-13-0) have excelled in crafting visually appealing images from text prompts. These T2I models are increasingly employed in creative endeavors such as visual arts [\(Xu et al.,](#page-14-0) [2024\)](#page-14-0), gaming [\(Pearce](#page-12-2) [et al.,](#page-12-2) [2023\)](#page-12-2), personalized image synthesis [\(Ruiz et al.,](#page-12-3) [2023;](#page-12-3) [Huang et al.,](#page-11-0) [2024a;](#page-11-0) [Hu et al.,](#page-11-1) [2021;](#page-11-1) [Shah et al.,](#page-13-1) [2023\)](#page-13-1), stylized rendering [\(Sohn et al.,](#page-13-2) [2023;](#page-13-2) [Hertz et al.,](#page-11-2) [2023;](#page-11-2) [Wang et al.,](#page-13-3) [2024a;](#page-13-3) [Jeong](#page-11-3) [et al.,](#page-11-3) [2024\)](#page-11-3), and image inversion or editing [\(Ulyanov et al.,](#page-13-4) [2018;](#page-13-4) [Delbracio & Milanfar,](#page-11-4) [2023;](#page-11-4) [Rout](#page-12-4) [et al.,](#page-12-4) [2023b;](#page-12-4) [2024;](#page-12-5) [Mokady et al.,](#page-12-6) [2023\)](#page-12-6). Content creators often need precise control over both the *content* and the *style* of generated images to match their vision. While the content of an image can be conveyed through text, articulating an artist's unique style – characterized by distinct brushstrokes, color palette, material, and texture – is substantially more nuanced. This has led to research on personalization through visual prompting [\(Sohn et al.,](#page-13-2) [2023;](#page-13-2) [Hertz et al.,](#page-11-2) [2023;](#page-11-2) [Wang et al.,](#page-13-3) [2024a\)](#page-13-3).

Recent studies have focused on finetuning pre-trained T2I models to learn style from a set of reference images [\(Gal et al.,](#page-11-5) [2022;](#page-11-5) [Ruiz et al.,](#page-12-3) [2023;](#page-12-3) [Sohn et al.,](#page-13-2) [2023;](#page-13-2) [Hu et al.,](#page-11-1) [2021\)](#page-11-1). This involves optimizing the model's text embeddings, model weights, or both, using the denoising diffusion loss. However, these methods demand substantial computational resources for training or finetuning large-scale foundation models, thus making them expensive to adapt to new, unseen styles. Furthermore, these methods often depend on human-curated images of the same style, which is less practical and can compromise quality when only a single reference image is available.

In training-free stylization, recent methods [\(Hertz et al.,](#page-11-2) [2023;](#page-11-2) [Wang et al.,](#page-13-3) [2024a;](#page-13-3) [Jeong et al.,](#page-11-3) [2024\)](#page-11-3) manipulate keys and values within the attention layers using just one reference style image. These methods face challenges in both extracting the style from the reference style image and accurately transferring the style to a target content image. For instance, during the DDIM inversion step [\(Song et al.,](#page-13-5) [2021a\)](#page-13-5) utilized by StyleAligned [\(Hertz et al.,](#page-11-2) [2023\)](#page-11-2), fine-grained details tend to be compromised. To mitigate this issue, InstantStyle [\(Wang et al.,](#page-13-3) [2024a\)](#page-13-3) incorporates features from

<sup>∗</sup>This work was done during an internship at Google.

![](_page_1_Picture_1.jpeg)

Figure 1: Given a single reference image (rounded rectangle), our method RB-Modulation offers a plug-and-play solution for (a) stylization, and (b) content-style composition with various prompts while maintaining sample diversity and prompt alignment. For instance, given a reference style image (e.g., "melting golden 3d rendering style") and content image (e.g., "a dog"), our method adheres to the desired prompts without leaking contents (e.g., flower) from the reference style image and without being restricted to the fixed pose or layout of the reference dog image.

the reference style image into specific layers of a previously trained IP-Adapter [\(Ye et al.,](#page-14-1) [2023\)](#page-14-1). However, identifying the exact layer for feature injection in a model is complex and not universally applicable across models. Also, feature injection can cause content leakage from the style image into the generated content. Moving on to content-style composition, InstantStyle [\(Wang et al.,](#page-13-3) [2024a\)](#page-13-3) employs a ControlNet [\(Zhang et al.,](#page-14-2) [2023\)](#page-14-2) (an additionally trained network) to preserve image layout, which inadvertently limits its diversity.

We introduce Reference-Based Modulation (RB-Modulation), a novel approach for stylization and composition that eliminates the need for training or finetuning diffusion models (*e.g*. Control-Net [\(Zhang et al.,](#page-14-2) [2023\)](#page-14-2) or adapters [\(Ye et al.,](#page-14-1) [2023;](#page-14-1) [Hu et al.,](#page-11-1) [2021\)](#page-11-1)). Our work reveals that the reverse dynamics in diffusion models can be formulated as stochastic optimal control problem. By incorporating style features into the controller's terminal cost, we modulate the drift field in diffusion model's reverse dynamics, enabling training-free personalization. Unlike conventional attention processors that often leak content from the reference style image, we propose to enhance the image fidelity via an Attention Feature Aggregation (AFA) module that decouples content from reference style image. We demonstrate the effectiveness of our method in stylization [\(Hertz et al.,](#page-11-2) [2023;](#page-11-2) [Wang](#page-13-3) [et al.,](#page-13-3) [2024a;](#page-13-3) [Jeong et al.,](#page-11-3) [2024\)](#page-11-3) and style+content composition, as illustrated in Figure [1\(](#page-1-0)a) and (b), respectively. Our experiments show that RB-Modulation outperforms current SoTA methods [\(Hertz](#page-11-2) [et al.,](#page-11-2) [2023;](#page-11-2) [Wang et al.,](#page-13-3) [2024a\)](#page-13-3) in terms of human preference and prompt-alignment metrics.

#### Our contributions are summarized as follows:

- We present reference-based modulation (RB-Modulation), a novel stochastic optimal control based test-time optimization framework that enables training-free, personalized style and content control, with a new Attention Feature Aggregation (AFA) module to maintain high fidelity to the reference image while adhering to the given prompt (§[4\)](#page-3-0).
- We provide theoretical justifications connecting optimal control and reverse diffusion dynamics. We leverage this connection to incorporate desired attributes (*e.g*., style) in our controller's terminal cost and personalize T2I models in a training-free manner (§[5\)](#page-5-0).
- We perform extensive experiments covering stylization and content-style composition, demonstrating superior performance over SoTA methods in human preference metrics (§[6\)](#page-6-0).

Personalization of T2I models: T2I generative models [\(Rombach et al.,](#page-12-1) [2022;](#page-12-1) [Podell et al.,](#page-12-7) [2023;](#page-12-7) [Pernias et al.,](#page-12-8) [2024\)](#page-12-8) can now generate high quality images from text prompts. Their text-following ability has unlocked new avenues in personalized content creation, including text-guided image editing [\(Mokady et al.,](#page-12-6) [2023;](#page-12-6) [Rout et al.,](#page-12-5) [2024\)](#page-12-5), solving inverse problems [\(Rout et al.,](#page-12-4) [2023b;](#page-12-4) [2024\)](#page-12-5), concept-driven generation [\(Ruiz et al.,](#page-12-3) [2023;](#page-12-3) [Tewel et al.,](#page-13-6) [2023;](#page-13-6) [Kumari et al.,](#page-12-9) [2023;](#page-12-9) [Chen et al.,](#page-10-0) [2024\)](#page-10-0), personalized outpainting [\(Tang et al.,](#page-13-7) [2023\)](#page-13-7), identity-preservation [\(Ruiz et al.,](#page-13-8) [2024;](#page-13-8) [Huang](#page-11-0) [et al.,](#page-11-0) [2024a;](#page-11-0) [Wang et al.,](#page-14-3) [2024b\)](#page-14-3), and stylized synthesis [\(Sohn et al.,](#page-13-2) [2023;](#page-13-2) [Wang et al.,](#page-13-3) [2024a;](#page-13-3) [Hertz](#page-11-2) [et al.,](#page-11-2) [2023;](#page-11-2) [Shah et al.,](#page-13-1) [2023\)](#page-13-1). To tailor T2I models for a specific style (*e.g*., painting) or content (*e.g*., object), existing methods follow one of two recipes: (1) full finetuning (FT) or parameter efficient finetuning (PEFT) and (2) training-free, which we discuss below.

Finetuning T2I models for personalization: FT [\(Ruiz et al.,](#page-12-3) [2023;](#page-12-3) [Everaert et al.,](#page-11-6) [2023\)](#page-11-6) and PEFT [\(Kumari et al.,](#page-12-9) [2023;](#page-12-9) [Hu et al.,](#page-11-1) [2021;](#page-11-1) [Sohn et al.,](#page-13-2) [2023;](#page-13-2) [Shah et al.,](#page-13-1) [2023\)](#page-13-1) methods excel at capturing style or object details when the underlying T2I model can be finetuned on a few (typically 4) reference images for few thousand iterations. PARASOL [\(Tarres et al.](#page-13-9) ´ , [2024\)](#page-13-9) requires supervised data via a cross-modal search to train both the denoising U-Net and a projector network. Diff-NST [\(Ruta et al.,](#page-13-10) [2023\)](#page-13-10) trains the attention processor by targeting the 'V' values within the denoising U-Net. The curation of supervised data and resource-intensive finetuning for every style or content makes these methods challenging for practical usage.

Training-free methods for personalization: Training-free personalization methods are preferable to finetuning methods given the vastly faster time of execution. In StyleAligned [\(Hertz et al.,](#page-11-2) [2023\)](#page-11-2), a reference style image and a text prompt describing the style are used to extract style features via DDIM inversion [\(Song et al.,](#page-13-5) [2021a\)](#page-13-5). Target queries and keys are then normalized using adaptive instance normalization [\(Huang & Belongie,](#page-11-7) [2017\)](#page-11-7) based on reference counterparts. Finally, reference image keys and values are merged with DDIM-inverted latents in self-attention layers, which tends to leak content information from the reference style image (Figure [2\)](#page-6-1). Moreover, the need for textual description in the DDIM inversion step can degrade its performance. DiffusionDisentanglement [\(Wu et al.,](#page-14-4) [2023\)](#page-14-4) aims to reduce the approximation error in DDIM inversion by jointly minimizing a perceptual loss and a directional CLIP loss, which is prone to content leakage [\(Wang](#page-13-3) [et al.,](#page-13-3) [2024a\)](#page-13-3). Swapping Self-Attention (SSA) [\(Jeong et al.,](#page-11-3) [2024\)](#page-11-3) addresses these limitations by replacing the target keys and values in self-attention layers with those from a reference style image. It still relies on DDIM inversion to cache keys and values of the reference style, which tends to compromise fine-grained details [\(Wang et al.,](#page-13-3) [2024a\)](#page-13-3). Both StyleAligned [\(Hertz et al.,](#page-11-2) [2023\)](#page-11-2) and SSA [\(Jeong et al.,](#page-11-3) [2024\)](#page-11-3) require two reverse processes to share their attention layer features and thus demand significant memory. InstantStyle [\(Wang et al.,](#page-13-3) [2024a\)](#page-13-3) injects reference style features into specific cross-attention layers of IP-Adapter [\(Ye et al.,](#page-14-1) [2023\)](#page-14-1), addressing two key limitations: DDIM inversion and memory-intensive reverse processes. However, pinpointing the exact layers for feature injection is complex, and may not generalize to other models. In addition, when composing style and content, InstantStyle [\(Wang et al.,](#page-13-3) [2024a\)](#page-13-3) relies on ControlNet [\(Zhang et al.,](#page-14-2) [2023\)](#page-14-2), which can limit the diversity of generated images to fixed layouts and deviate from the prompt.

Optimal Control: Stochastic optimal control finds wide applications in diverse fields such as molecular dynamics [\(Holdijk et al.,](#page-11-8) [2024\)](#page-11-8), economics [\(Fleming & Rishel,](#page-11-9) [2012\)](#page-11-9), non-convex optimization [\(Chaudhari et al.,](#page-10-1) [2018\)](#page-10-1), robotics [\(Theodorou et al.,](#page-13-11) [2011\)](#page-13-11), and mean-field games [\(Carmona](#page-10-2) [et al.,](#page-10-2) [2018\)](#page-10-2) Despite its extensive use, and recent works on its connections to diffusion based generative models [\(Berner et al.,](#page-10-3) [2024;](#page-10-3) [Tzen & Raginsky,](#page-13-12) [2019;](#page-13-12) [Chen et al.,](#page-10-4) [2023\)](#page-10-4), it has been less explored in training-free personalization. In this paper, we introduce a novel test-time optimization framework leveraging the main concepts from optimal control to achieve training-free personalization. A key aspect of optimal control is designing a controller to guide a stochastic process towards a desired terminal condition [\(Fleming & Rishel,](#page-11-9) [2012\)](#page-11-9). This aligns with our goal of training-free personalization, as we target a specific style or content at the end of the reverse diffusion process, which can be incorporated in the controller's terminal condition.

RB-Modulation overcomes several challenges encountered by SoTA methods [\(Hertz et al.,](#page-11-2) [2023;](#page-11-2) [Jeong et al.,](#page-11-3) [2024;](#page-11-3) [Wang et al.,](#page-13-3) [2024a\)](#page-13-3). Since RB-Modulation does not require DDIM inversion, it retains fine-grained details unlike StyleAligned [\(Hertz et al.,](#page-11-2) [2023\)](#page-11-2). Using a stochastic controller to refine the trajectory of a single reverse process, it overcomes the limitation of coupled reverse processes [\(Hertz et al.,](#page-11-2) [2023\)](#page-11-2). By incorporating a style descriptor in our controller's terminal cost, it eliminates the dependency on Adapters [\(Ye et al.,](#page-14-1) [2023;](#page-14-1) [Hu et al.,](#page-11-1) [2021\)](#page-11-1) or ControlNets [\(Zhang](#page-14-2) [et al.,](#page-14-2) [2023\)](#page-14-2) by InstantStyle [\(Wang et al.,](#page-13-3) [2024a\)](#page-13-3).

#### 3 PRELIMINARIES

Diffusion models consist of two stochastic processes: (a) *noising process*, modeled by a Stochastic Differential Equation (SDE) known as forward-SDE: dX<sup>t</sup> = f(Xt, t) dt + g(Xt, t) dWt, X<sup>0</sup> ∼ p0, and (b) *denoising process*, modeled by the time-reversal of forward-SDE under mild regularity conditions [\(Anderson,](#page-10-5) [1982\)](#page-10-5), also known as reverse-SDE:

$$dX_t = [f(X_t, t) - g^2(X_t, t) \nabla \log p(X_t, t)] dt + g(X_t, t) dW_t, \quad X_1 \sim \mathcal{N}(0, I_d). \quad (1)$$

Here, W = (Wt)t≥<sup>0</sup> is standard Brownian motion in a filtered probability space, (Ω, F,(Ft)t≥0,P), p(·, t) denotes the marginal density of p at time t, and ∇ log pt(·) the corresponding score function. f(Xt, t) and g(Xt, t) are called drift and volatility, respectively. A popular choice of <sup>f</sup>(Xt, t) = <sup>−</sup>X<sup>t</sup> and <sup>g</sup>(Xt, t) = √ 2 corresponds to the well-known forward Ornstein-Uhlenbeck (OU) process.

For T2I generation, the reverse-SDE [\(1\)](#page-3-1) is simulated using a neural network s (xt, t; θ) [\(Hyvarinen](#page-11-10) ¨ [& Dayan,](#page-11-10) [2005;](#page-11-10) [Vincent,](#page-13-13) [2011\)](#page-13-13) to approximate ∇<sup>x</sup> log p(xt, t). Importantly, to accelerate the sampling process in practice [\(Song et al.,](#page-13-5) [2021a;](#page-13-5) [Karras et al.,](#page-12-10) [2022;](#page-12-10) [Zhang & Chen,](#page-14-5) [2022\)](#page-14-5), the reverse-SDE ( -[1\)](#page-3-1) shares the same path measure with a probability flow ODE: dX<sup>t</sup> = f(Xt, t) − 2 g 2 (Xt, t)∇ log p(Xt, t) dt, where X<sup>1</sup> ∼ N (0,Id).

Personalized diffusion models either fully finetune θ of s (xt, t; θ) [\(Ruiz et al.,](#page-12-3) [2023;](#page-12-3) [Everaert et al.,](#page-11-6) [2023\)](#page-11-6), or train a parameter-efficient adapter ∆θ for s (xt, t; θ + ∆θ) on reference style images [\(Hu](#page-11-1) [et al.,](#page-11-1) [2021;](#page-11-1) [Sohn et al.,](#page-13-2) [2023;](#page-13-2) [Shah et al.,](#page-13-1) [2023\)](#page-13-1). Our method does not finetune θ or train ∆θ. Instead, we derive a new drift field through a stochastic control that *modulates* the reverse-SDE [\(1\)](#page-3-1).

#### 4 METHOD

Personalization using optimal control: Normalize time t by the total number of diffusion steps T such that 0 ≤ t ≤ 1. Let us denote by u : R <sup>d</sup> × [0, 1] → <sup>R</sup> d a controller from the admissible set of controls U ⊆ R d , X<sup>u</sup> <sup>t</sup> ∈ <sup>R</sup> d a state variable, ` : R <sup>d</sup> × <sup>R</sup> <sup>d</sup> × [0, 1] → <sup>R</sup> the transient cost, and h : R <sup>d</sup> → <sup>R</sup> the terminal cost of the reverse process (X<sup>u</sup> t ) 0 <sup>t</sup>=1. We show in §[5](#page-5-0) that training-free personalization can be formulated as a control problem where the drift of the standard reverse-SDE [\(1\)](#page-3-1) is modified via RB-modulation:

$$\min_{u \in U} \mathbb{E} \left[ \int_1^0 \ell(X_t^u, u(X_t^u, t), t) dt + \gamma h(X_0^u) \right], \quad \text{where} \quad (2)$$

$$dX_t^u = [f(X_t^u, t) - g^2(X_t^u, t) \nabla \log p(X_t^u, t) + u(X_t^u, t)] dt + g(X_t^u, t) dW_t, \quad X_1^u \sim \mathcal{N}(0, \text{I}_d).$$

Importantly, the terminal cost h(·), weighted by γ, captures the discrepancy in feature space between the styles of the reference image and the generated image. The resulting controller u(·, t) modulates the drift over time to satisfy this terminal cost. We derive the solution to this optimal control problem through the Hamilton-Jacobi-Bellman (HJB) equation [\(Fleming & Rishel,](#page-11-9) [2012\)](#page-11-9); refer to Appendix [A](#page-15-0) for details. Our proposed RB-Modulation Algorithm [1](#page-5-1) has two key components: (a) stochastic optimal controller and (b) attention feature aggregation. Below, we discuss each in turn.

(a) Stochastic Optimal Controller (SOC): We show that the reverse dynamics in diffusion models can be framed as a stochastic optimal control problem with a quadratic terminal cost (theoretical analysis in §[5\)](#page-5-0). For personalization using a reference style image X f <sup>0</sup> = z0, we use a Contrastive Style Descriptor (CSD) [\(Somepalli et al.,](#page-13-14) [2024\)](#page-13-14) to extract style features Ψ(X f 0 ). Since the score functions s (xt, t; θ)≈∇ log p (Xt, t) are available from pre-trained diffusion models [\(Podell et al.,](#page-12-7) [2023;](#page-12-7) [Pernias et al.,](#page-12-8) [2024\)](#page-12-8), our goal is to add a correction term u(·, t) to modulate the reverse-SDE and minimize the overall cost [\(2\)](#page-3-2). We approximate X<sup>u</sup> <sup>0</sup> with its conditional expectation using Tweedie's formula [\(Efron,](#page-11-11) [2011;](#page-11-11) [Rout et al.,](#page-12-4) [2023b;](#page-12-4) [2024\)](#page-12-5). Finally, we incorporate the style features into our controller's terminal cost as: h (X<sup>u</sup> 0 ) = kΨ(X f ) − Ψ(<sup>E</sup> [X<sup>u</sup> 0 |X<sup>u</sup> t ])k 2 2 .

Our theoretical results (§[5\)](#page-5-0) suggest that the optimal controller can be obtained by solving the HJB equation and letting γ → ∞. In practice, this translates to dropping the transient cost ` (X<sup>u</sup> t , u(X<sup>u</sup> t , t), t) and solving [\(2\)](#page-3-2) with only the terminal constraint, *i.e*.,

$$\min_{u \in \mathcal{U}} \|\Psi(X_0^f) - \Psi(\mathbb{E}[X_0^u | X_t^u])\|_2^2. \quad (3)$$

Thus, we solve [\(3\)](#page-3-3) to find the optimal control u and use this controller in the reverse dynamics [\(2\)](#page-3-2) to update the current state from X<sup>u</sup> t to X<sup>u</sup> t−∆t (recall that time flows backwards in the reverse-SDE [\(1\)](#page-3-1)). Our implementation of [\(3\)](#page-3-3) is given in Algorithm [1](#page-5-1), which follows from our theoretical insights.

Implementation challenge: For smaller models [\(Rombach et al.,](#page-12-1) [2022\)](#page-12-1), we can directly solve our control problem [\(3\)](#page-3-3). However, for larger models [\(Podell et al.,](#page-12-7) [2023;](#page-12-7) [Pernias et al.,](#page-12-8) [2024\)](#page-12-8), the control objective [\(3\)](#page-3-3) requires back propagation through the score network with tentatively billions of parameters. This significantly increases time and memory complexity [\(Rout et al.,](#page-12-4) [2023b;](#page-12-4) [2024\)](#page-12-5).

We propose a test-time proximal gradient descent approach to address this challenge. The key ingredient of our Algorithm [1](#page-5-1) is to find the previous state Xt−∆<sup>t</sup> by modulating the current state X<sup>t</sup> based on an optimal controller u ∗ . The optimal controller u ∗ is obtained by minimizing the discrepancy in style between X¯ <sup>u</sup> 0 := <sup>E</sup>[X<sup>u</sup> 0 |X<sup>u</sup> <sup>t</sup> = xt], obtained using our controlled reverse-SDE [\(3\)](#page-3-3), and the reference style image z0. Motivated by this interpretation, an alternate Algorithm [2](#page-5-2) avoids back propagation through s(xt, t; θ) by introducing a dummy variable x0, which serves as a proxy for X¯ <sup>u</sup> in the terminal cost. Instead of forcing x<sup>0</sup> to be decided by the dynamics of the reverse-SDE as in Algorithm [1](#page-5-1), we allow it to be only approximately faithful to the dynamics. This is implemented by adding a proximal penalty, *i.e*. x ∗ <sup>0</sup> = arg min<sup>x</sup>0∈R<sup>d</sup> kΨ(X f 0 ) − Ψ(x0)k 2 <sup>2</sup> + λkx<sup>0</sup> − <sup>E</sup> [X<sup>u</sup> 0 |X<sup>u</sup> t 2 2 , where the hyper-parameter λ controls the faithfulness of the reverse dynamics. This penalty assumes that with a small step-size in [\(3\)](#page-3-3), x ∗ 0 and <sup>E</sup>[X<sup>u</sup> 0 |X<sup>u</sup> <sup>t</sup> = xt] will be close. Thus, Algorithm [2](#page-5-2) enables personalization of large-scale foundation models, *matching the speed of training-free methods and obtaining 5-20X speedup over training-based methods*; see Table [4](#page-21-0) in Appendix [B.2](#page-21-1) for details.

While prior works [\(Chung et al.,](#page-11-12) [2023;](#page-11-12) [Zhu et al.,](#page-14-6) [2023;](#page-14-6) [He et al.,](#page-11-13) [2024\)](#page-11-13) have used a proximal sampler in related settings, their underlying generative model is not personalized. We believe that this is an important reason why our method results in a significant speedup while satisfying the terminal constraints. Our paper takes the first step in personalizing the underlying generative model via a novel attention processor as discussed below.

(b) Attention Feature Aggregation (AFA): Let d denote the dimension of the latent variable Xt, n<sup>q</sup> the embedding dimension for query Q, and n<sup>h</sup> the output dimension of the hidden layer. Transformer-based diffusion models [\(Rombach et al.,](#page-12-1) [2022;](#page-12-1) [Podell et al.,](#page-12-7) [2023;](#page-12-7) [Pernias et al.,](#page-12-8) [2024\)](#page-12-8) consist of self-attention and cross-attention layers operating on latent embedding x<sup>t</sup> ∈ <sup>R</sup> <sup>d</sup>×n<sup>h</sup> . Within the attention module Attention(Q, K, V ), x<sup>t</sup> is projected into queries Q ∈ <sup>R</sup> d×n<sup>q</sup> , keys K ∈ R d×n<sup>q</sup> , and values V ∈ R <sup>d</sup>×n<sup>h</sup> using linear projections. Through Q, K, and V , attention layers capture global context and improve long-range dependencies within xt.

To incorporate a reference image (*e.g*., style or content) while retaining alignment with the prompt, we introduce the Attention Feature Aggregation (AFA) module. Given a prompt p, a reference style image Is, and a reference content image Ic, we first extract the embeddings using CLIP text encoder [\(Radford et al.,](#page-12-11) [2021\)](#page-12-11) and CSD image encoder [\(Somepalli et al.,](#page-13-14) [2024\)](#page-13-14). These embeddings are projected into keys and values using linear projection. We denote by K<sup>p</sup> and V<sup>p</sup> the keys and values from p, K<sup>s</sup> and V<sup>s</sup> from Is, K<sup>c</sup> and V<sup>c</sup> from I<sup>c</sup> (used only in content-style composition). The query Q, derived from a linear projection of xt, remains consistent in the AFA module. To maintain consistency between text and style, we compose the keys and values of both text and style in our attention mechanism. The final output of the AFA module is given by

$$\begin{aligned} AFA &= \text{Avg} (A_{text}, A_{style}, A_{text+style}), A_{text} = \text{Attention}(Q, [K; K_p], [V; V_p]), \\ A_{style} &= \text{Attention}(Q, [K; K_s], [V; V_s]), A_{text+style} = \text{Attention}(Q, [K; K_p, K_s], [V; V_p, V_s]), \end{aligned}$$

where [K; Kp] ∈ <sup>R</sup> 2d×n<sup>q</sup> indicates concatenation of K with K<sup>p</sup> along the number of tokens dimension. For style-content composition, we process the content image I<sup>c</sup> in the same way as the reference style image Is, and obtain another set of attention outputs:

$$\begin{aligned} AFA &= \text{Avg}(A_{text}, A_{style}, A_{content}, A_{content+style}), \\ A_{content} &= \text{Attention}(Q, [K; K_c], [V; V_c]), A_{content+style} = \text{Attention}(Q, [K; K_s; K_c], [V; V_s; V_c]). \end{aligned}$$

Importantly, the AFA module is computationally tractable as it only requires the computation of a multi-head attention, which is widely used in practice [\(Podell et al.,](#page-12-7) [2023\)](#page-12-7).

Disentangling content and style. In stylization (content described by text; style illustrated by a reference style image), prior works [\(Hertz et al.,](#page-11-2) [2023;](#page-11-2) [Wang et al.,](#page-13-3) [2024a\)](#page-13-3) inject the entire reference style image I<sup>s</sup> that does not disentangle content and style. However, our AFA module injects

|    | Algorithm 1: RB-Modulation (Exact)                              |
|----|-----------------------------------------------------------------|
|    | Input: Diffusion steps T , reference prompt p , reference style |
|    | image z 0 , style descriptor Ψ( ) ,                             |
|    | score network s ( , , ; θ )                                     |
|    | Tunable parameter: Stepsize η , optimization steps M            |
|    | Output: Personalized latent X u                                 |
| 1  | Initialize x T ← N (0 , I d )                                   |
| 2  | for t = T to 1 do                                               |
| 3  | Initialize controller u = 0                                     |
| 4  | for m = 1 to M do                                               |
| 5  | x ˆ t = x t + u controlled state                                |
| 6  | X ¯ u                                                           |
|    | 0 = √                                                           |
|    | x ˆ t                                                           |
|    | α ¯ t                                                           |
|    | (1 − α ¯ t )                                                    |
|    | √ α ¯ t                                                         |
|    | s ( x ˆ t , t, p ; θ )                                          |
| 7  | h ( X ¯ u                                                       |
|    | ) = k Ψ( z 0 ) − Ψ( X ¯ u                                       |
|    | ) k                                                             |
|    | using Eq. (3)                                                   |
| 8  | u = u − η ∇ u h ( X ¯ u                                         |
|    | ) update controller                                             |
| 9  | end                                                             |
| 10 | x                                                               |
|    | t = x t + u optimally controlled state                          |
| 11 | X ¯ u                                                           |
|    | 0 =                                                             |
|    | √ t                                                             |
|    | α ¯ t                                                           |
|    | (1 − α ¯ t )                                                    |
|    | √ α ¯ t                                                         |
|    | s ( x                                                           |
|    | , t, p ; θ ) terminal state                                     |
| 12 | x t − 1 ← DDIM ( X ¯ u                                          |
|    | , x                                                             |
|    | ) one denoising update                                          |
| 13 | end                                                             |
| 14 | return X u                                                      |

Algorithm 2: RB-Modulation (Proximal)

| Algorithm    |              | 2: RB-Modulation | (Proximal)                               |
|--------------|--------------|------------------|------------------------------------------|
| Input:       | Diffusion    | time             | steps T , reference prompt p , reference |
|              | style image  | z 0 ,            | style descriptor Ψ( ) ,                  |
|              | score        | network s        | ( , , ; θ )                              |
| Tunable      | parameters:  |                  | Stepsize η , optimization steps M ,      |
| proximal     | strength     | λ                |                                          |
| Output:      | Personalized |                  | latent X u                               |
| 1 Initialize | x T          | ← N (0           | , I d )                                  |
| 2 for t      | = T to       | 1 do             |                                          |
| 3            | Compute      | posterior        | mean                                     |
|              | E [ X u      |                  |                                          |
|              |              | X u              |                                          |
|              |              | t =              | x t ] = √                                |
|              |              |                  | x t                                      |
|              |              |                  | α ¯ t                                    |
|              |              |                  | (1 − α ¯ t )                             |
|              |              |                  | √ α ¯ t                                  |
|              |              |                  | s ( x t , t, p ; θ )                     |
| 4            | Initialize   | opt. variable    | x 0 = E [ X u                            |
|              |              |                  | X u                                      |
|              |              |                  | t = x t ]                                |
| 5            | for m =      | 1 to M           | do                                       |
| 6            | Compute      | controller’s     | cost L ( x 0 ) := k Ψ( z 0 ) −           |
|              | Ψ(           | x 0 ) k          |                                          |
|              |              | 2                | + λ k x 0 − E [ X u                      |
|              |              |                  | X u                                      |
|              |              |                  | t = x t ] k                              |
| 7            | Update       | optimization     | variable                                 |
|              | x            | 0 = x 0          | − η ∇ x 0 L ( x 0 )                      |
| 8            | end          |                  |                                          |
| 9            | x t − 1 ←    | DDIM (           | x 0 , x t ) one denoising step           |
| 10 end       |              |                  |                                          |
| 11 return    | X u          |                  |                                          |

*only the style features* from I<sup>s</sup> using the style attention head of the Vision Transformer (ViT) in CSD [\(Somepalli et al.,](#page-13-14) [2024\)](#page-13-14). The AFA module achieves content-style disentanglement by computing separate attention maps for content from text and style from image. In this case, SOC does not handle content and focuses solely on style aspects by using the style attention head as Ψ(·).

In content-style composition (content described by both text and a reference content image; style described by a reference style image), the AFA module injects content (extracted from the reference content) and style features (from the reference style image) separately using their respective attention heads in the ViT [\(Somepalli et al.,](#page-13-14) [2024\)](#page-13-14). The SOC module controls *content* by minimizing the discrepancy between content features from the generated image and the *reference content* image, and *style* by minimizing the discrepancy between style features extracted from the generated and *reference style* image. This distinction from prior works enables our method to prevent leakage.

## 5 THEORETICAL JUSTIFICATIONS

Problem setup: We outline an approach to derive the optimal controller for a special case of our control problem [\(2\)](#page-3-2). We substitute t ← 1−t to account for the time reversal in the reverse-SDE [\(1\)](#page-3-1). Here, X<sup>u</sup> <sup>0</sup> ∼ N (0,Id) and X<sup>u</sup> <sup>1</sup> ∼ pdata. We consider the dynamic without the Brownian motion: dX<sup>u</sup> <sup>t</sup> = v(X<sup>u</sup> t , u, t)dt, X<sup>u</sup> <sup>t</sup><sup>0</sup> = x0, where 0 ≤ t<sup>0</sup> ≤ t ≤ t<sup>N</sup> ≤ 1 and v : <sup>R</sup> <sup>d</sup> × <sup>R</sup> <sup>d</sup> × [t0, t<sup>N</sup> ] → <sup>R</sup> d denotes the drift field. The optimal controller u ∗ can be derived by solving the Hamilton-Jacobi-Bellman (HJB) equation [\(Fleming & Rishel,](#page-11-9) [2012;](#page-11-9) [Basar et al.,](#page-10-6) [2020\)](#page-10-6), see Appendix [A](#page-15-0) for details.

Incorporating optimal control in diffusion: Following recent works [\(Kappen,](#page-11-14) [2008;](#page-11-14) [Chen et al.,](#page-10-4) [2023\)](#page-10-4), we consider a dynamical system whose drift field minimizes a transient trajectory cost and a terminal cost (weighted by γ) to ensure "closeness" to reference content x<sup>1</sup> (Appendix [A.1\)](#page-15-1). Proposition [A.2](#page-16-0) [\(Chen et al.,](#page-10-4) [2023\)](#page-10-4) outlines the optimal control in the limiting setting where γ → ∞. Furthermore, suppose we replace x<sup>1</sup> with its conditional expectation (discussed in Remark [A.3\)](#page-16-1), *the resulting dynamic is the standard reverse-SDE for the Orstein-Uhlenbeck (OU) diffusion process for a particular noise schedule.* This connection between classic linear quadratic control and the standard reverse-SDE allows us to study other diffusion problems (*e.g*., personalization) through the lens of stochastic optimal control. For instance, we derive the optimal controller given reference *style features* y<sup>1</sup> at the terminal time.

Proposition 5.1. *Suppose* A ∈ R <sup>k</sup>×<sup>d</sup> *be a linear style extractor that operates on the terminal state* X<sup>u</sup> <sup>1</sup> ∈ <sup>R</sup> d *. Given reference style features* y1*, consider the control problem:*

$$\min_{u \in \mathcal{U}} \int_{t_0}^1 \frac{1}{2} \|u(X_t^u, t)\|^2 dt + \frac{\gamma}{2} \|AX_1^u - y_1\|_2^2, \text{ where } dX_t^u = u(X_t^u, t) dt, \quad X_{t_0}^u = x_0.$$

*Then, in the limit when* γ → ∞*, the optimal controller* u <sup>∗</sup> = (A <sup>T</sup> <sup>A</sup>) <sup>−</sup><sup>1</sup><sup>A</sup> <sup>T</sup> (y1−Axt) 1−t *, which yields the following controlled dynamic:* dX<sup>u</sup> <sup>t</sup> = (A <sup>T</sup> <sup>A</sup>) <sup>−</sup><sup>1</sup><sup>A</sup> <sup>T</sup> (y1−Axt) 1−t dt.

![](_page_6_Picture_1.jpeg)

![](_page_6_Figure_2.jpeg)

Figure 2: Qualitative results for stylization: A comparison with state-of-the-art methods (InstantStyle [\(Wang et al.,](#page-13-3) [2024a\)](#page-13-3), StyleAligned [\(Hertz et al.,](#page-11-2) [2023\)](#page-11-2), StyleDrop [\(Sohn et al.,](#page-13-2) [2023\)](#page-13-2)) highlights our advantages in preventing information leakage from the reference style and adhering more closely to desired prompts.

Implication. The optimal controller depends on the reference *style features* y<sup>1</sup> at the terminal time, instead of the image content encoded in x1. To simulate the controlled dynamic in practice, we use CSD [\(Somepalli et al.,](#page-13-14) [2024\)](#page-13-14) as a style feature extractor and replace y<sup>1</sup> with the style features extracted from the expected terminal state <sup>E</sup>[X<sup>u</sup> 1 |X<sup>u</sup> t ], as discussed in Appendix [A.2](#page-17-0).

Drift modulation through optimal controller: We then study a control problem where the velocity field is a linear combination of the state and the control variable. This problem is interesting to study because the reverse-SDE dynamic of the standard OU process has a drift field of the form: v (Xt, t) = −X<sup>t</sup> − 2∇ log p(Xt, t). For a Gaussian prior X<sup>0</sup> ∼ N (0,I), the law of the OU process satisfies ∇ log p (Xt, t) = −Xt, and the corresponding drift field becomes v (Xt, t) = Xt. Our goal is to modulate this drift field using a controller u (X<sup>u</sup> t , t). The result below provides the structure of the optimal control (again in the setting where the terminal objective is known; see Appendix A1).

Proposition 5.2. *Suppose* A ∈ R <sup>k</sup>×<sup>d</sup> *be a linear style extractor that operates on the terminal state* X<sup>u</sup> <sup>1</sup> ∈ <sup>R</sup> d *. Let* p<sup>t</sup> *denote* ∇xV ∗ (x, t) *in HJB equation [\(A.1\)](#page-15-2). Given reference style features* y1*, consider the control problem:*

$$\min_{u \in \mathcal{U}} \int_{t_0}^1 \frac{1}{2} \|u(X_t^u, t)\|^2 dt + \frac{\gamma}{2} \|AX_1^u - y_1\|_2^2, \text{ where } dX_t^u = [X_t^u + u(X_t^u, t)] dt, \quad X_{t_0}^u = x_0,$$

*Then, the optimal controller becomes* u ∗ (t) = −pt*, where the instantaneous state* X<sup>u</sup> <sup>t</sup> = x<sup>t</sup> *and* p<sup>t</sup> *satisfy the following coupled transitions:*

$$\begin{bmatrix} \mathbf{x}_t \\ \mathbf{p}_t \end{bmatrix} = \begin{bmatrix} x_0 e^t - \frac{\gamma}{2} A^T (A \mathbf{x}_1 - y_1) e^{1+t} + \frac{\gamma}{2} A^T (A \mathbf{x}_1 - y_1) e^{1-t} \\ \gamma A^T (A \mathbf{x}_1 - y_1) e^{1-t} \end{bmatrix}.$$

Summary. We build on the connection between optimal control and reverse diffusion (see Appendices [A.1-](#page-15-1)[A.3](#page-18-0) for details). The general strategy is to derive the optimal controller with known terminal state, and then replace the terminal state in the controller with its estimate using Tweedie's formula. For stylized models and Gaussian prior, the controllers have an explicit form. However in practice, the data distribution may not be Gaussian, and thus, we do not aim for a closed-form expression to modulate the drift. This line of analysis, however, points to our method RB-Modulation. As discussed in §[4,](#page-3-0) we incorporate a style descriptor in our controller's terminal cost and evaluate the resulting drift at each reverse time step either through back propagating through the score network (Algorithm [1](#page-5-1)), or an approximation based on proximal gradient updates (Algorithm [2](#page-5-2)).

Metrics: Evaluating stylized synthesis is challenging due to the subjective nature of style, making simple metrics inadequate. We follow a two step approach: first using metrics from prior

Table 1: User study: We report the % of human preference on ours *vs*. alternatives for overall quality (OQ), style alignment (SA), and prompt alignment (PA), including ties where users couldn't decide. Our method consistently outperforms alternatives, achieving higher scores in all metrics.

| Human Preference (%) | Ours OQ ↑ | vs SA ↑ | InstantStyle PA ↑ | Ours OQ ↑ | vs StyleAligned SA ↑ | PA ↑ | Ours OQ ↑ | vs IP-Adapter SA ↑ | PA ↑ |
|----------------------|-----------|---------|-------------------|-----------|----------------------|------|-----------|--------------------|------|
| Alternative          | 39.8      | 38.5    | 39.5              | 24.4      | 27.8                 | 29.4 | 8.1       | 20.1               | 8.3  |
| Tie                  | 9.3       | 6.4     | 7.3               | 8.8       | 7.1                  | 5.8  | 6.9       | 4.8                | 4.5  |
| RB-Modulation (ours) | 51.0      | 55.1    | 53.3              | 66.9      | 65.1                 | 64.9 | 85.0      | 75.1               | 87.2 |

works and then conducting human evaluation. To evaluate prompt-image alignment, we use CLIP-T score [\(Hertz et al.,](#page-11-2) [2023;](#page-11-2) [Sohn et al.,](#page-13-2) [2023;](#page-13-2) [Wang et al.,](#page-13-3) [2024a\)](#page-13-3) and ImageReward [\(Xu et al.,](#page-14-0) [2024\)](#page-14-0), which also consider human aesthetics, distortions, and object completeness. When a style description is provided, CLIP-T and ImageReward also capture style alignment. We assess style similarity using DINO [\(Caron et al.,](#page-10-7) [2021\)](#page-10-7) and content similarity using CLIP-I [\(Radford et al.,](#page-12-11) [2021\)](#page-12-11) as in prior work [\(Hertz et al.,](#page-11-2) [2023;](#page-11-2) [Ruiz et al.,](#page-12-3) [2023;](#page-12-3) [Sohn et al.,](#page-13-2) [2023\)](#page-13-2), and highlight their limitations in disentangling style and content performance in evaluation. Given the importance of human evaluation in T2I personalization [\(Hertz et al.,](#page-11-2) [2023;](#page-11-2) [Sohn et al.,](#page-13-2) [2023;](#page-13-2) [Ruiz et al.,](#page-12-3) [2023;](#page-12-3) [Shah et al.,](#page-13-1) [2023;](#page-13-1) [Jeong et al.,](#page-11-3) [2024\)](#page-11-3), we also conduct a user study though Amazon Mechanical Turk to measure both style and text alignment.

Datasets and baselines: We use style images from StyleAligned benchmark [\(Hertz et al.,](#page-11-2) [2023\)](#page-11-2) for stylization and content images from DreamBooth [\(Ruiz et al.,](#page-12-3) [2023\)](#page-12-3) for content-style composition. We base RB-Modulation on the recently released StableCascade [\(Pernias et al.,](#page-12-8) [2024\)](#page-12-8). We compare with three training-free methods: InstantStyle [\(Wang et al.,](#page-13-3) [2024a\)](#page-13-3) (state-of-the-art), IP-Adapter [\(Ye et al.,](#page-14-1) [2023\)](#page-14-1), and StyleAligned [\(Hertz et al.,](#page-11-2) [2023\)](#page-11-2). For completeness, we also compare with training-based methods StyleDrop [\(Sohn et al.,](#page-13-2) [2023\)](#page-13-2) and ZipLoRA [\(Shah et al.,](#page-13-1) [2023\)](#page-13-1).

Implementation details: All experiments run on a single A100 NVIDIA GPU. We use the same hyper-parameters for our method across tasks, and default settings for alternative methods as per their original papers. Details are provided in Appendix [B.1.](#page-20-0)

#### 6.1 IMAGE STYLIZATION

Qualitative analysis: This section describes image stylization experiments using a text prompt and a reference style image. Figure [2](#page-6-1) compares our method with SoTA training-free InstantStyle [\(Wang et al.,](#page-13-3) [2024a\)](#page-13-3) and StyleAligned [\(Hertz et al.,](#page-11-2) [2023\)](#page-11-2), and training-based StyleDrop [\(Sohn](#page-13-2) [et al.,](#page-13-2) [2023\)](#page-13-2). Except for StyleDrop, which requires ∼5 minutes of training per style, all methods, including ours, are training-free and complete inference in <1 minute. While all methods produce reasonable outputs, alternative methods encounter issues with information leakage. For instance, in the third row of Figure [2,](#page-6-1) StyleAligned and StyleDrop generate a wine bottle and book resembling the smartphone in the reference style image. In the last row, StyleAligned leaks the house and the background of the reference image; InstantStyle exhibits color leakage from the house, resulting in similar-colored images. Our method accurately adheres to the prompt in the desired style. As illustrated in the second and the third row, our method generates only one glass of wine and a highfidelity rubber duck, compared to baselines where extra items appear (wine bottles styled like the left smartphone) or incorrect styles (cartoon-style rubber duck).

User study: Given the subjective nature of this field, we conduct a user study on Amazon Mechanical Turk with 155 participants using 100 styles from the StyleAligned dataset [\(Hertz et al.,](#page-11-2) [2023\)](#page-11-2), collecting a total of 7,200 answers (8 responses for each question). Each user answers 3 questions comparing our method with an alternative method regarding (1) overall quality, (2) style alignment, and (3) prompt alignment (details in the Appendix [B.8\)](#page-25-0). Table [1](#page-7-0) summarizes the percentage of human preferences for our method, the alternative method, or a tie. Our method consistently outperforms the alternatives, including the current SoTA method InstantStyle [\(Wang et al.,](#page-13-3) [2024a\)](#page-13-3). The preference rates over all three metrics highlight the effectiveness of our method RB-Modulation.

Quantitative analysis: Table [2](#page-8-0) evaluates 300 prompts and 100 styles on the StyleAligned dataset [\(Hertz et al.,](#page-11-2) [2023\)](#page-11-2) using three metrics, with and without style descriptions in the prompts. Our method outperforms others notably in the ImageReward metric, closely matching human aesthetics assessment from the user study in Table [1.](#page-7-0) In addition, the CLIP-T score indicates our effective alignment between generated images and text prompts. While IP-Adapter and StyleAligned

![](_page_8_Picture_1.jpeg)

Figure 3: Ablation study: We show the effectiveness of our different proposed components by sequentially adding them to vanila StableCascade [\(Pernias et al.,](#page-12-8) [2024\)](#page-12-8). DirectConcat involves concatenating reference image embeddings with prompt embeddings.

Table 2: Quantitative results for stylization: We compare alternative methods on three metrics: ImageReward [\(Xu et al.,](#page-14-0) [2024\)](#page-14-0) and CLIP-T [\(Radford et al.,](#page-12-11) [2021\)](#page-12-11) for prompt alignment, DINO [\(Caron et al.,](#page-10-7) [2021\)](#page-10-7) for style alignment. Note that DINO score does not capture information leakage, so higher scores are not necessarily better (§[B.5\)](#page-23-0).

| With          | style description? |        |       |        | No    | ImageReward ↑ Yes | CLIP-T No | score ↑ Yes | DINO No | score Yes |
|---------------|--------------------|--------|-------|--------|-------|-------------------|-----------|-------------|---------|-----------|
| IP-Adapter    | (Ye et             | al.,   | 2023) |        | -1.99 | -1.51             | 0.21      | 0.26        | 0.89    | 0.89      |
| StyleAligned  | (Hertz             | et     | al.,  | 2023)  | -0.68 | 0.01              | 0.26      | 0.31        | 0.80    | 0.85      |
| InstantStyle  | (Wang              | et     | al.,  | 2024a) | 0.09  | 0.72              | 0.29      | 0.33        | 0.68    | 0.72      |
| RB-Modulation |                    | (ours) |       |        | 0.91  | 1.18              | 0.30      | 0.34        | 0.68    | 0.73      |

have higher DINO scores, their lower rating in ImageReward, CLIP-T and user preference expose information leakage from the reference style images. Nevertheless, our DINO score remains competitive with the leading method InstantStyle. Notably, all metrics show improvement with style descriptions, particularly in ImageReward, where leveraging style descriptions enhances prompt alignment. Our method achieves high ImageReward and CLIP-T score even without style descriptions, suggesting robustness in prompt alignment without explicit style information in the prompt.

Ablation Study: Figure [3](#page-8-1) shows an ablation study of the AFA and SOC modules. We include a baseline, "DirectConcat", which concatenates reference style embeddings with text embeddings in the cross-attention modules. DirectConcat mixes both embeddings, making it less effective in disentangling style from prompts (*e.g*., cat *vs*. lighthouse). While AFA or SOC alone mitigates this by modulating the reverse drift and attention modules (§[4\)](#page-3-0), each has drawbacks. AFA alone fails to capture the cat's style accurately, and SOC alone misplaces elements, like "a lighthouse hat on the cat" and "a railroad trunk on a piano". We observe consistent improvements with each module, with the best results when combined.

#### 6.2 CONTENT-STYLE COMPOSITION

Since this paper primarily focuses on style-based personalization, we perform extensive experiments on stylization. To further demonstrate the versatility of our framework, we also explore content-style composition as an additional capability.

Qualitative analysis: Content-style composition aims to preserve the essence of both content and style depicted in the reference images, while ensuring the resulting image aligns with a given text prompt. Figure [4](#page-9-0) compares our method against training-free InstantStyle [\(Wang et al.,](#page-13-3) [2024a\)](#page-13-3), IP-Adapter [\(Ye et al.,](#page-14-1) [2023\)](#page-14-1), and training-based ZipLoRA [\(Shah et al.,](#page-13-1) [2023\)](#page-13-1). Notably, the trainingfree InstantStyle and IP-Adapter rely on ControlNet [\(Zhang et al.,](#page-14-2) [2023\)](#page-14-2), which often constrains their ability to accurately follow prompts for changing the pose of the generated content, such as illustrating "dancing" in Figure [4\(](#page-9-0)b), or "walking" in (c). In contrast, our method avoids the need for ControlNet or adapters, and can effectively capture the distinctive attributes of both style and content images while adhering to the prompt to generate diverse images. In Figure [4\(](#page-9-0)a), our method accurately captures elements like "table" and "river" that are overlooked in InstantStyle and IP-Adapter. In addition, our method mitigates information leakage, as evidenced in Figure [4\(](#page-9-0)b), where the trunk of the tree behind the sloth is erroneously captured by InstantStyle and IP-Adapter but not

![](_page_9_Figure_1.jpeg)

![](_page_9_Picture_2.jpeg)

Figure 4: Qualitative results for content-style composition: Our method shows better prompt alignment and greater diversity than training-free methods IP-Adapter [\(Ye et al.,](#page-14-1) [2023\)](#page-14-1) and InstantStyle [\(Wang et al.,](#page-13-3) [2024a\)](#page-13-3), and have competitive performance with training-based ZipLoRA [\(Shah et al.,](#page-13-1) [2023\)](#page-13-1) .

Table 3: Quantitative results for composition: In addition to stylization metrics, we use CLIP-T score [\(Radford et al.,](#page-12-11) [2021\)](#page-12-11) to evaluate content alignment with the reference image. Similar to DINO, CLIP-I could inflate test score [\(Sohn et al.,](#page-13-2) [2023;](#page-13-2) [Shah et al.,](#page-13-1) [2023\)](#page-13-1) due to content leakage, but does not correlate to user preference; higher scores do not indicate better human preference.

|                      | ImageReward ↑ | CLIP-T score ↑ | DINO score | CLIP-I score |
|----------------------|---------------|----------------|------------|--------------|
| IP-Adapter           | -0.78         | 0.22           | 0.73       | 0.68         |
| InstantStyle         | -0.54         | 0.21           | 0.71       | 0.71         |
| RB-Modulation (ours) | 0.74          | 0.26           | 0.74       | 0.71         |

by ours. Compared to ZipLoRA [\(Shah et al.,](#page-13-1) [2023\)](#page-13-1) that requires training of 12 LoRAs [\(Hu et al.,](#page-11-1) [2021\)](#page-11-1) and additional merge layers for each composition, our method requires no training at all while yielding competitive or better results. For instance, our method effectively captures the 2D cartoon and 3D rendering styles as illustrated in Figures [4\(](#page-9-0)a) and (b).

Quantitative analysis: Table [3](#page-9-1) shows quantitative evaluation using 50 styles from StyleAligned dataset [\(Hertz et al.,](#page-11-2) [2023\)](#page-11-2) and 5 contents from DreamBooth dataset [\(Ruiz et al.,](#page-12-3) [2023\)](#page-12-3). Unlike prior works [\(Hertz et al.,](#page-11-2) [2023;](#page-11-2) [Sohn et al.,](#page-13-2) [2023;](#page-13-2) [Shah et al.,](#page-13-1) [2023;](#page-13-1) [Ruiz et al.,](#page-12-3) [2023;](#page-12-3) [Jeong et al.,](#page-11-3) [2024\)](#page-11-3) reporting either DINO and CLIP-I scores, we present both metrics and demonstrate comparable performance across them. Additionally, we obtain notably higher ImageReward score, which aligns closely with human aesthetics assessment as evidenced in §[6.1](#page-7-1) and [\(Xu et al.,](#page-14-0) [2024\)](#page-14-0). Consequently, we omitted a user study in this section. For more details, please refer to Appendix [B.1.](#page-20-0)

## 7 CONCLUSION

We introduced Reference-Based modulation (RB-Modulation), a test-time optimization method for personalizing transformer-based diffusion models. RB-Modulation builds on concepts from stochastic optimal control to modulate the drift field of reverse diffusion dynamics, incorporating desired attributes (*e.g*., style or content) via a terminal cost. Our Attention Feature Aggregation (AFA) module decouples content and style in the cross-attention layers and enables precise control over both. In addition, we derived theoretical connections between linear quadratic control and the denoising diffusion process, which led to the creation of RB-Modulation. Empirically, our method outperformed current state-of-the-art methods in stylization and content-style composition. To our best knowledge, this is the first training-free personalization framework using stochastic optimal control, which marks the departure from external adapters or ControlNets.

# 8 BROADER IMPACT STATEMENT

Social impact: Image stylization and content-style composition based on diffusion models potentially have both positive and negative social impact. This technology provides an easy-to-use tool to the general public for image generation which can help visualize their artistic ideas. On the other hand, our work on stylization and content-style composition poses a risk of generating arts that closely mimic or infringe upon existing copyrighted material, leading to legal and ethical issues. More broadly, our method inherits the risks from T2I models which are capable of generating fake contents that can be misused by malicious users.

Safeguards: We build on StableCascade [\(Pernias et al.,](#page-12-8) [2024\)](#page-12-8), which has a mechanism to filter offensive image generations. Our framework RB-Modulation inherits these safeguards. In addition, to mitigate misuse, we believe it is crucial to ensure the underlying model's safety, which may involve (i) watermarking AI-generated artworks and (ii) implementing an NSFW filter to remove inappropriate contents.

Reproducibility: The pseudocode and hyper-parameter details have been provided in the paper. The source code is available on the project page: <https://rb-modulation.github.io/>.

## ACKNOWLEDGMENTS

This research has been supported by NSF Grant 2019844, a Google research collaboration award, and the UT Austin Machine Learning Lab. Litu Rout has been supported by Ju-Nam and Pearl Chew Presidential Fellowship and George J. Heuer Graduate Fellowship from UT Austin.

## REFERENCES


[1] Brian D.O. Anderson. Reverse-time diffusion equation models. *Stochastic Processes and their Applications*, 12(3):313–326, 1982. Karl J. Astrom. *Introduction to Stochastic Control Theory*. Elsevier Science, 1971. Tamer Basar, Sean Meyn, and William R Perkins. Lecture notes on control system theory and design. *arXiv preprint arXiv:2007.01367*, 2020. Julius Berner, Lorenz Richter, and Karen Ullrich. An optimal control perspective on diffusion-based generative modeling. *Transactions on Machine Learning Research*, 2024. ISSN 2835-8856. URL <https://openreview.net/forum?id=oYIjw37pTP>. Rene Carmona, Franc¸ois Delarue, et al. ´ *Probabilistic theory of mean field games with applications I-II*. Springer, 2018. Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J ´ egou, Julien Mairal, Piotr Bojanowski, and ´ Armand Joulin. Emerging properties in self-supervised vision transformers. In *Proceedings of the IEEE/CVF international conference on computer vision*, pp. 9650–9660, 2021. Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan ´ Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. In *Proceedings of the 40th International Conference on Machine Learning*, pp. 4055–4075, 2023. Pratik Chaudhari, Adam Oberman, Stanley Osher, Stefano Soatto, and Guillaume Carlier. Deep relaxation: partial differential equations for optimizing deep neural networks. *Research in the Mathematical Sciences*, 5:1–30, 2018. Tianrong Chen, Jiatao Gu, Laurent Dinh, Evangelos Theodorou, Joshua M Susskind, and Shuangfei Zhai. Generative modeling with phase stochastic bridge. In *The Twelfth International Conference on Learning Representations*, 2023. Wenhu Chen, Hexiang Hu, Yandong Li, Nataniel Ruiz, Xuhui Jia, Ming-Wei Chang, and William W Cohen. Subject-driven text-to-image generation via apprenticeship learning. *Advances in Neural Information Processing Systems*, 36, 2024.

[2] Hyungjin Chung, Jong Chul Ye, Peyman Milanfar, and Mauricio Delbracio. Prompt-tuning latent diffusion models for inverse problems. *arXiv preprint arXiv:2310.01110*, 2023. Mauricio Delbracio and Peyman Milanfar. Inversion by direct iteration: An alternative to denoising diffusion for image restoration. *Transactions on Machine Learning Research*, 2023. Bradley Efron. Tweedie's formula and selection bias. *Journal of the American Statistical Association*, 106(496):1602–1614, 2011. Martin Nicolas Everaert, Marco Bocchio, Sami Arpa, Sabine Susstrunk, and Radhakrishna Achanta. ¨ Diffusion in style. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 2251–2261, 2023. Wendell H Fleming and Raymond W Rishel. *Deterministic and stochastic optimal control*, volume 1. Springer Science & Business Media, 2012. Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. *arXiv preprint arXiv:2208.01618*, 2022. Zinan Guo, Yanze Wu, Zhuowei Chen, Lang Chen, and Qian He. Pulid: Pure and lightning id customization via contrastive alignment. *arXiv preprint arXiv:2404.16022*, 2024. Yutong He, Naoki Murata, Chieh-Hsin Lai, Yuhta Takida, Toshimitsu Uesaka, Dongjun Kim, Wei-Hsiang Liao, Yuki Mitsufuji, J Zico Kolter, Ruslan Salakhutdinov, and Stefano Ermon. Manifold preserving guided diffusion. In *The Twelfth International Conference on Learning Representations*, 2024. URL <https://openreview.net/forum?id=o3BxOLoxm1>. Amir Hertz, Andrey Voynov, Shlomi Fruchter, and Daniel Cohen-Or. Style aligned image generation via shared attention. *arXiv preprint arXiv:2312.02133*, 2023. Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems*, 33:6840–6851, 2020. Lars Holdijk, Yuanqi Du, Ferry Hooft, Priyank Jaini, Berend Ensing, and Max Welling. Stochastic optimal control for collective variable free sampling of molecular transition paths. *Advances in Neural Information Processing Systems*, 36, 2024. Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. In *International Conference on Learning Representations*, 2021. Jiehui Huang, Xiao Dong, Wenhui Song, Hanhui Li, Jun Zhou, Yuhao Cheng, Shutao Liao, Long Chen, Yiqiang Yan, Shengcai Liao, et al. Consistentid: Portrait generation with multimodal finegrained identity preserving. *arXiv preprint arXiv:2404.16771*, 2024a. Jiehui Huang, Xiao Dong, Wenhui Song, Hanhui Li, Jun Zhou, Yuhao Cheng, Shutao Liao, Long Chen, Yiqiang Yan, Shengcai Liao, et al. Consistentid: Portrait generation with multimodal finegrained identity preserving. *arXiv preprint arXiv:2404.16771*, 2024b. Xun Huang and Serge Belongie. Arbitrary style transfer in real-time with adaptive instance normalization. In *Proceedings of the IEEE international conference on computer vision*, pp. 1501–1510, 2017. Aapo Hyvarinen and Peter Dayan. Estimation of non-normalized statistical models by score match- ¨ ing. *Journal of Machine Learning Research*, 6(4), 2005. Jaeseok Jeong, Junho Kim, Yunjey Choi, Gayoung Lee, and Youngjung Uh. Visual style prompting with swapping self-attention. *arXiv preprint arXiv:2402.12974*, 2024. HJ Kappen. Stochastic optimal control theory. *ICML, Helsinki, Radbound University, Nijmegen, Netherlands*, 2008.

[3] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. *Advances in Neural Information Processing Systems*, 35:26565–26577, 2022. Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 1931–1941, 2023. Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. *arXiv preprint arXiv:2210.02747*, 2022. Xingchao Liu, Chengyue Gong, et al. Flow straight and fast: Learning to generate and transfer data with rectified flow. In *The Eleventh International Conference on Learning Representations*, 2022. Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 6038–6047, 2023. Tim Pearce, Tabish Rashid, Anssi Kanervisto, Dave Bignell, Mingfei Sun, Raluca Georgescu, Sergio Valcarcel Macua, Shan Zheng Tan, Ida Momennejad, Katja Hofmann, and Sam Devlin. Imitating human behaviour with diffusion models. In *The Eleventh International Conference on Learning Representations*, 2023. URL [https://openreview.net/forum?id=](https://openreview.net/forum?id=Pv1GPQzRrC8) [Pv1GPQzRrC8](https://openreview.net/forum?id=Pv1GPQzRrC8). Pablo Pernias, Dominic Rampas, Mats Leon Richter, Christopher Pal, and Marc Aubreville. Wurstchen: An efficient architecture for large-scale text-to-image diffusion models. In ¨ *The Twelfth International Conference on Learning Representations*, 2024. URL [https://](https://openreview.net/forum?id=gU58d5QeGv) [openreview.net/forum?id=gU58d5QeGv](https://openreview.net/forum?id=gU58d5QeGv). Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Muller, Joe ¨ Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In *The Twelfth International Conference on Learning Representations*, 2023. Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *International conference on machine learning*, pp. 8748–8763. PMLR, 2021. Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In *International Conference on Machine Learning*, pp. 8821–8831. PMLR, 2021. Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High- ¨ resolution image synthesis with latent diffusion models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 10684–10695, 2022. Litu Rout, Advait Parulekar, Constantine Caramanis, and Sanjay Shakkottai. A theoretical justification for image inpainting using denoising diffusion probabilistic models. *arXiv preprint arXiv:2302.01217*, 2023a. Litu Rout, Negin Raoof, Giannis Daras, Constantine Caramanis, Alexandros G Dimakis, and Sanjay Shakkottai. Solving inverse problems provably via posterior sampling with latent diffusion models. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023b. URL <https://openreview.net/forum?id=XKBFdYwfRo>. Litu Rout, Yujia Chen, Abhishek Kumar, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Beyond first-order tweedie: Solving inverse problems using latent diffusion. In *2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024. Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 22500– 22510, 2023.

[4] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa, Michael Rubinstein, and Kfir Aberman. Hyperdreambooth: Hypernetworks for fast personalization of text-to-image models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 6527–6536, 2024. Dan Ruta, Gemma Canet Tarres, Andrew Gilbert, Eli Shechtman, Nicholas Kolkin, and John Col- ´ lomosse. Diff-nst: Diffusion interleaving for deformable neural style transfer. *arXiv preprint arXiv:2307.04157*, 2023. Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models with deep language understanding. *arXiv preprint arXiv:2205.11487*, 2022. Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, and Varun Jampani. ZipLoRA: Any subject in any style by effectively merging loras. *arXiv preprint arXiv:2311.13600*, 2023. Kihyuk Sohn, Nataniel Ruiz, Kimin Lee, Daniel Castro Chin, Irina Blok, Huiwen Chang, Jarred Barber, Lu Jiang, Glenn Entis, Yuanzhen Li, et al. Styledrop: Text-to-image generation in any style. In *37th Conference on Neural Information Processing Systems (NeurIPS)*. Neural Information Processing Systems Foundation, 2023. Gowthami Somepalli, Anubhav Gupta, Kamal Gupta, Shramay Palta, Micah Goldblum, Jonas Geiping, Abhinav Shrivastava, and Tom Goldstein. Measuring style similarity in diffusion models. *arXiv preprint arXiv:2404.01292*, 2024. Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In *International Conference on Learning Representations*, 2021a. URL [https://openreview.net/](https://openreview.net/forum?id=St1giarCHLP) [forum?id=St1giarCHLP](https://openreview.net/forum?id=St1giarCHLP). Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In *International Conference on Learning Representations*, 2021b. URL [https://openreview.net/](https://openreview.net/forum?id=PxTIG12RRHS) [forum?id=PxTIG12RRHS](https://openreview.net/forum?id=PxTIG12RRHS). Luming Tang, Nataniel Ruiz, Qinghao Chu, Yuanzhen Li, Aleksander Holynski, David E Jacobs, Bharath Hariharan, Yael Pritch, Neal Wadhwa, Kfir Aberman, et al. Realfill: Reference-driven generation for authentic image completion. *arXiv preprint arXiv:2309.16668*, 2023. Gemma Canet Tarres, Dan Ruta, Tu Bui, and John Collomosse. Parasol: Parametric style control ´ for diffusion image synthesis. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 2432–2442, 2024. Yoad Tewel, Rinon Gal, Gal Chechik, and Yuval Atzmon. Key-locked rank one editing for text-toimage personalization. In *ACM SIGGRAPH 2023 Conference Proceedings*, pp. 1–11, 2023. Evangelos Theodorou, Freek Stulp, Jonas Buchli, and Stefan Schaal. An iterative path integral stochastic optimal control approach for learning robotic tasks. *IFAC Proceedings Volumes*, 44(1): 11594–11601, 2011. Belinda Tzen and Maxim Raginsky. Theoretical guarantees for sampling and inference in generative models with latent diffusions. In *Conference on Learning Theory*, pp. 3084–3114. PMLR, 2019. Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. Deep image prior. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, June 2018. Pascal Vincent. A connection between score matching and denoising autoencoders. *Neural computation*, 23(7):1661–1674, 2011. Haofan Wang, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. Instantstyle: Free lunch towards style-preserving in text-to-image generation. *arXiv preprint arXiv:2404.02733*, 2024a.

[5] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, and Anthony Chen. Instantid: Zero-shot identitypreserving generation in seconds. *arXiv preprint arXiv:2401.07519*, 2024b. Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, and Anthony Chen. Instantid: Zero-shot identitypreserving generation in seconds. *arXiv preprint arXiv:2401.07519*, 2024c. Qiucheng Wu, Yujian Liu, Handong Zhao, Ajinkya Kale, Trung Bui, Tong Yu, Zhe Lin, Yang Zhang, and Shiyu Chang. Uncovering the disentanglement capability in text-to-image diffusion models. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 1900–1910, 2023. Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. *Advances in Neural Information Processing Systems*, 36, 2024. Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. *arXiv preprint arXiv:2308.06721*, 2023. Jiwen Yu, Yinhuai Wang, Chen Zhao, Bernard Ghanem, and Jian Zhang. Freedom: Training-free energy-guided conditional diffusion model. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 23174–23184, 2023. Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pp. 3836–3847, 2023. Qinsheng Zhang and Yongxin Chen. Fast sampling of diffusion models with exponential integrator. In *The Eleventh International Conference on Learning Representations*, 2022. Yuanzhi Zhu, Kai Zhang, Jingyun Liang, Jiezhang Cao, Bihan Wen, Radu Timofte, and Luc Van Gool. Denoising diffusion models for plug-and-play image restoration. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 1219–1229, 2023.
#### A ADDITIONAL THEORETICAL RESULTS

In this section, we restate the propositions more precisely and provide their technical proofs. First, we recall standard terminologies from optimal control literature [\(Fleming & Rishel,](#page-11-9) [2012\)](#page-11-9). For 0 ≤ t<sup>0</sup> ≤ t ≤ t<sup>N</sup> ≤ 1, the cost function associated with the controller u(·) is defined by the integral:

$$V(u; \mathbf{x}_0, t_0) = \int_{t_0}^{t_N} \ell(X_t^u, u, t) dt + h(X_{t_N}^u, X_{t_0}^u), \quad (4)$$

where `(· · ·) denotes a scalar valued function of the state X<sup>u</sup> t , controller u(·), and instantaneous time t. The value function V ∗ (x0, t0) is defined as the minimum value of V (u; x0, t0) over the set of admissible controllers U, i.e.,

$$V^* = V^*(\mathbf{x}_0, t_0) = \min_{u \in \mathcal{U}} V(u; \mathbf{x}_0, t_0) = \min_{u \in \mathcal{U}} \int_{t_0}^{t_N} \ell(X_t^u, u, t) dt + h(X_{t_N}^u), \quad X_{t_0}^u = \mathbf{x}_0, \quad (5)$$

which satisfies a Partial Differential Equation (PDE) given below in Theorem [A.1](#page-15-2).

Theorem A.1 (HJB Equation, [\(Fleming & Rishel,](#page-11-9) [2012;](#page-11-9) [Basar et al.,](#page-10-6) [2020\)](#page-10-6)). *If* V <sup>∗</sup> *has continuous partial derivatives, then it must satisfy the following PDE, also known as Hamilton-Jacobi-Bellman (HJB) equation:*

$$-\frac{\partial V^*}{\partial t}(\mathbf{x}, t) = \min_{u \in \mathcal{U}} \left[ H(\mathbf{x}, \nabla_{\mathbf{x}} V^*(\mathbf{x}, t), u, t) := \ell(\mathbf{x}, u, t) + (\nabla_{\mathbf{x}} V^*(\mathbf{x}, t))^T v(\mathbf{x}, u, t) \right].$$

*Also, the Hamiltonian* H (x, ∇xV ∗ (x, t), u, t)*, optimal controller* u ∗ (t) *and the state trajectory* x ∗ (t) *must satisfy*

$$\min_{u \in \mathcal{U}} H(\mathbf{x}^*(t), \nabla_{\mathbf{x}} V^*(\mathbf{x}^*(t), t), u, t) = H(\mathbf{x}^*(t), \nabla_{\mathbf{x}} V^*(\mathbf{x}^*(t), t), u^*(t), t).$$

#### A.1 INTERPRETING REVERSE-SDE AS A SOLUTION TO OPTIMAL CONTROL

For clarity, we restate the problem setup here and describe the main ideas from §[4](#page-3-0) in more details. Problem setup: We discuss a standard approach to derive the optimal controller in a special case of our control problem [\(2\)](#page-3-2). We substitute t ← 1 − t to account for the time reversal in the reverse-SDE [\(1\)](#page-3-1). In this setup, X<sup>u</sup> <sup>0</sup> ∼ N (0,Id) and X<sup>u</sup> <sup>1</sup> ∼ pdata. We consider the following dynamic without the Brownian motion:

$$dX_t^u = v(X_t^u, u, t)dt, \quad X_{t_0}^u = \mathbf{x}_0, \quad (6)$$

where 0 ≤ t<sup>0</sup> ≤ t ≤ t<sup>N</sup> ≤ 1 and v : <sup>R</sup> <sup>d</sup> × <sup>R</sup> <sup>d</sup> × [t0, t<sup>N</sup> ] → <sup>R</sup> <sup>d</sup> denotes the drift field. The optimal controller u ∗ can be derived by solving the Hamilton-Jacobi-Bellman (HJB) equation [\(Fleming &](#page-11-9) [Rishel,](#page-11-9) [2012;](#page-11-9) [Basar et al.,](#page-10-6) [2020\)](#page-10-6), see Appendix [A](#page-15-0) for details. By certainty equivalence (when the drift and diffusion coefficients are linear time-varying [\(Astrom,](#page-10-8) [1971\)](#page-10-8), which occurs when pdata is Gaussian; see also discussion in Section [A.3\)](#page-18-0), the same u ∗ applies to a more general case with the Brownian motion [\(Chen et al.,](#page-10-4) [2023\)](#page-10-4), where

$$dX_t^u = v(X_t^u, u, t)dt + dW_t, \quad X_t^u = \mathbf{x}_0. \quad (7)$$

Therefore, we analyze the reverse dynamic in the absence of the Brownian motion, and employ the same controller in more general cases with the Brownian motion.

Below, we consider a dynamical system whose drift field is chosen to minimize a transient trajectory cost and a terminal cost (weighted by γ) that enforces "closeness" to reference content x1. Proposition [A.2](#page-16-0) provides the structure of the optimal control in the limiting setting where γ → ∞. Furthermore, suppose we replace x<sup>1</sup> with its conditional expectation (discussed in Remark [A.3\)](#page-16-1), the resulting dynamic, interestingly, is the standard reverse-SDE for the Orstein-Uhlenbeck (OU) diffusion process. This connection between optimal control (more precisely, classic Linear Quadratic Control) and the standard reverse-SDE provides us a path to study other diffusion problems (*e.g*. personalization [\(Ruiz et al.,](#page-12-3) [2023;](#page-12-3) [Hertz et al.,](#page-11-2) [2023;](#page-11-2) [Sohn et al.,](#page-13-2) [2023;](#page-13-2) [Wang et al.,](#page-13-3) [2024a\)](#page-13-3), image editing or inversion [\(Mokady et al.,](#page-12-6) [2023;](#page-12-6) [Delbracio & Milanfar,](#page-11-4) [2023;](#page-11-4) [Rout et al.,](#page-12-4) [2023b;](#page-12-4) [2024;](#page-12-5) [2023a\)](#page-12-12)) through the lens of stochastic optimal control.

Proposition A.2 (Linear optimal control with quadratic cost [\(Chen et al.,](#page-10-4) [2023\)](#page-10-4)). *Consider the control problem:*

$$\min_{u \in \mathcal{U}} \int_{t_0}^1 \frac{1}{2} \|u(X_t^u, t)\|^2 dt + \frac{\gamma}{2} \|X_1^u - x_1\|_2^2,$$
where  $dX_t^u = u(X_t^u, t) dt$ ,  $X_{t_0}^u = x_0$ 

*Then, in the limit when* γ → ∞*, the optimal controller is given by* u <sup>∗</sup> = x1−X<sup>u</sup> t 1−t *, which yields* dX<sup>u</sup> <sup>t</sup> = x1−X<sup>u</sup> t 1−t dt *for the deterministic case and* dX<sup>u</sup> <sup>t</sup> = x1−X<sup>u</sup> t 1−t dt + dW<sup>t</sup> *for the stochastic case.*

The optimal controller for the problem presented in Proposition [A.2](#page-16-0) can be derived using established techniques from control theory [\(Fleming & Rishel,](#page-11-9) [2012;](#page-11-9) [Basar et al.,](#page-10-6) [2020;](#page-10-6) [Kappen,](#page-11-14) [2008\)](#page-11-14); the specific form of the above result follows from [\(Chen et al.,](#page-10-4) [2023\)](#page-10-4) (but without their momentum term). The key steps in this derivation include: (1) computing the Hamiltonian, (2) applying the minimum principle theorem to derive a set of differential equations, and (3) taking the limit as γ → ∞. These three steps are fundamental in deriving a closed-form solution. The final step is critical for satisfying hard terminal constraint and is essential for the practical implementation of Algorithm [1](#page-5-1) and Algorithm [2](#page-5-2), as detailed in §[4.](#page-3-0)

For generative modeling, the controlled dynamics described in Proposition [A.2](#page-16-0) cannot be directly applied. This limitation arises because the optimal control u <sup>∗</sup> depends on the terminal state x1, making it non-causal or reliant on future information. Inspired by recent advancements in flow-based generative models [\(Lipman et al.,](#page-12-13) [2022;](#page-12-13) [Liu et al.,](#page-12-14) [2022\)](#page-12-14), we make the optimal controller causal by replacing the terminal state with its conditional expectation given the current state, i.e., , *i.e*. x<sup>1</sup> ← <sup>E</sup>[X<sup>u</sup> 1 |X<sup>u</sup> <sup>t</sup> = xt]. This modification results in a controlled dynamic that can be simulated to produce a generative model incorporating principles from optimal control, as elaborated in Remark [A.3](#page-16-1).

Remark A.3 (Connections between diffusion-based generative modeling and stochastic optimal control). *Following conditional diffusion models and optimal transport paths [\(Lipman et al.,](#page-12-13) [2022;](#page-12-13) [Liu et al.,](#page-12-14) [2022\)](#page-12-14), where* X f <sup>t</sup> <sup>=</sup> tX<sup>f</sup> <sup>0</sup> + (1 − t)*, the state variable* X<sup>u</sup> t *is equal in distribution to* X f <sup>1</sup>−<sup>t</sup> = (1 − t)X f <sup>0</sup> + t, ∼ N (0,Id) *after time reversal. Now, we use Tweedie's formula [\(Efron,](#page-11-11) [2011\)](#page-11-11) to compute the posterior mean:*

$$\mathbb{E}[X_1^u | X_t^u] = \frac{X_t^u}{1-t} + \frac{t^2}{1-t} \nabla \log p(X_t^u, 1-t). \quad (8)$$

*Substituting the posterior mean in the controlled reverse dynamic of Proposition [A.2](#page-16-0), we arrive at*

$$\begin{aligned} dX_t^u &= \frac{[\mathbb{E}[X_1^u | X_t^u] - X_t^u]}{(1-t)} dt + dW_t \\ &= \left[ \frac{t}{(1-t)^2} X_t^u + \frac{t^2}{(1-t)^2} \nabla \log p(X_t^u, 1-t) \right] dt + dW_t. \end{aligned}$$

We observe that the above equation is structurally the same as reverse-SDE associated with a forward Orstein-Uhlenbeck (OU) diffusion process. This relation between diffusion-based generative models and optimal control is further explored in the Appendices below.

Indeed, diffusion models [\(Ho et al.,](#page-11-15) [2020;](#page-11-15) [Song et al.,](#page-13-15) [2021b;](#page-13-15) [Rombach et al.,](#page-12-1) [2022;](#page-12-1) [Podell et al.,](#page-12-7) [2023;](#page-12-7) [Pernias et al.,](#page-12-8) [2024\)](#page-12-8) provide an effective approximation to the terminal state of a denoising process. This approximation has been used for a variety of generative modeling tasks. Also, the terminal state can be approximated using Tweedie's formula [\(Efron,](#page-11-11) [2011\)](#page-11-11) with a learned score function [\(Ho et al.,](#page-11-15) [2020\)](#page-11-15) . By utilizing these pre-trained diffusion models, we can employ the connection to optimal control as discussed above to develop practically implementable generative models that incorporates terminal objectives such as style and personalization. Consequently, the subsequent sections are dedicated to deriving the optimal controller assuming a known terminal state; we will approximate this in practice using Tweedie's formula as above.

<sup>1</sup>Alternatively, when the reverse process is described by a probability flow ODE, a trained neural network can directly predict the terminal state [\(Song et al.,](#page-13-5) [2021a\)](#page-13-5).

A.2 INCORPORATING PERSONALIZED STYLE CONSTRAINTS THROUGH A TERMINAL COST

In this section, we derive the optimal controller when we have access to the reference *style features* y<sup>1</sup> at the terminal time (instead of the content of the image encoded through x1).

Proposition A.4. *Suppose* A ∈ R <sup>k</sup>×<sup>d</sup> *be a linear style extractor that operates on the terminal state* X<sup>u</sup> <sup>1</sup> ∈ <sup>R</sup> d *. Given reference style features* y1*, consider the control problem:*

$$\min_{u \in \mathcal{U}} \int_{t_0}^1 \frac{1}{2} \|u(X_t^u, t)\|^2 dt + \frac{\gamma}{2} \|AX_1^u - y_1\|_2^2, \quad (9)$$

$$\text{where } dX_t^u = u(X_t^u, t) dt, \quad X_{t_0}^u = x_0, \quad (10)$$

*Then, in the limit when* γ → ∞*, the optimal controller* u <sup>∗</sup> = (A <sup>T</sup> <sup>A</sup>) <sup>−</sup><sup>1</sup><sup>A</sup> <sup>T</sup> (y1−AX<sup>u</sup> t ) 1−t *, which yields the following controlled dynamic:*

$$dX_t^u = \frac{(A^T A)^{-1} A^T (y_1 - AX_t^u)}{1 - t} dt. \quad (11)$$

*Proof.* We derive the closed-form solution of the optimal controller given a fixed terminal state condition. This is similar to [\(Chen et al.,](#page-10-4) [2023\)](#page-10-4), where the reverse process is accelerated using momentum (see also [\(Kappen,](#page-11-14) [2008;](#page-11-14) [Basar et al.,](#page-10-6) [2020\)](#page-10-6) for further details on this approach). The distinction, however, lies in the treatment of the terminal constraint. For completeness, we provide full details of the proof below.

To derive the closed-form solution[<sup>2</sup>](#page-17-1) , recall from equation [\(5\)](#page-15-3) that `(xt, ut, t) = <sup>1</sup> 2 kutk 2 and the terminal cost h(x1) = <sup>γ</sup> 2 kAx<sup>1</sup> − y1k 2 . Let p<sup>t</sup> represent ∇xV ∗ (x, t) in Theorem [A.1](#page-15-2). Then, the Hamiltonian of the control problem [\(9\)](#page-17-2) is given by

$$\begin{aligned} H(\mathbf{x}_t, \mathbf{p}_t, \mathbf{u}_t, t) &= \ell(\mathbf{x}_t, \mathbf{u}_t, t) + \mathbf{p}_t^T \mathbf{u}_t \\ &= \frac{1}{2} \|\mathbf{u}_t\|^2 + \mathbf{p}_t^T \mathbf{u}_t. \end{aligned}$$

Since the minimizer of the Hamiltonian is u ∗ <sup>t</sup> = −pt, the value function becomes

$$V^* = \min_{\mathbf{u}_t} H(\mathbf{u}_t, \mathbf{p}_t, \mathbf{u}_t, t) = H(\mathbf{u}_t, \mathbf{p}_t, \mathbf{u}_t^*, t) = -\frac{1}{2} \|\mathbf{p}_t\|^2. \quad (12)$$

Now, we use minimum principle theorem [\(Basar et al.,](#page-10-6) [2020\)](#page-10-6) to obtain the following set of differential equations:

$$\frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t} = \nabla_{\mathbf{p}} H(\mathbf{x}_t, \mathbf{p}_t, \mathbf{u}_t^*, t) = -\mathbf{p}_t; \quad (13)$$

$$\frac{d\mathbf{p}_t}{dt} = -\nabla_{\mathbf{x}} H(\mathbf{x}_t, \mathbf{p}_t, \mathbf{u}_t^*, t) = 0; \quad (14)$$

$$\mathbf{x}_{t_0} = x_0; \quad (15)$$

$$\mathbf{p}_{t_N} = \nabla_{\mathbf{x}} h(\mathbf{x}_{t_N}, t_N) = \gamma A^T (A\mathbf{x}_{t_N} - y_1). \quad (16)$$

Integrating both sides of [\(13\)](#page-17-3), we have

$$\int_{t_0}^1 d\mathbf{x}_t = - \int_{t_0}^1 \mathbf{p}_t dt = -\mathbf{p}(1-t_0), \quad (17)$$

where the last equality is due to [\(14\)](#page-17-4), which states that p<sup>t</sup> is a constant independent of time t. This implies x<sup>1</sup> = x<sup>t</sup><sup>0</sup> − p(1 − t0). From [\(16\)](#page-17-5), we know for t<sup>N</sup> = 1 that

$$\begin{aligned} \mathbf{p}_1 &= \gamma A^T (A\mathbf{x}_1 - y_1) \\ &= \gamma (A^T A (x_0 - \mathbf{p}(1 - t_0)) - A^T y_1) \\ &= \gamma A^T A x_0 - \gamma A^T A \mathbf{p}_1 (1 - t_0) - \gamma A^T y_1 \end{aligned} \tag{18}$$

<sup>2</sup>With slight abuse of notation, we use x<sup>t</sup> to denote X u <sup>t</sup> and u<sup>t</sup> to denote u(X u t , t) in the deterministic case.

Rearranging [\(18\)](#page-17-6) and solving for p1, we get

$$\begin{aligned} \mathbf{p}_1 &= \gamma (I + \gamma A^T A (1 - t_0))^{-1} (A^T A x_0 - A^T y_1) \\ &= \begin{pmatrix} I \\ \frac{I}{\gamma} + A^T A (1 - t_0) \end{pmatrix}^{-1} (A^T A x_0 - A^T y_1) = \mathbf{p} \end{aligned} \quad (19)$$

Passing [\(19\)](#page-18-1) through the limit γ → ∞, we get

$$\lim_{\gamma \rightarrow \infty} \mathbf{p} = \frac{(A^T A)^{-1} (A^T A x_0 - A^T y_1)}{1 - t_0}. \quad (20)$$

Therefore, the optimal control becomes u ∗ <sup>t</sup> = −p = − (A <sup>T</sup> <sup>A</sup>) −1 (A <sup>T</sup> Axt−A <sup>T</sup> <sup>y</sup>1) 1−t , and the resulting dynamical system is given by

$$d\mathbf{x}_t = \frac{(A^T A)^{-1} A^T (y_1 - A\mathbf{x}_t)}{1 - t} dt,$$

for the deterministic process and

$$d\mathbf{x}_t = \frac{(A^T A)^{-1} A^T (y_1 - A\mathbf{x}_t)}{1 - t} dt + dW_t,$$

for the stochastic process with the Brownian motion. This completes the statement of the proof.

Implications: The optimal controller depends on the reference *style features* y<sup>1</sup> at the terminal time (instead of the image content x<sup>1</sup> as in Appendix [A.1\)](#page-15-1). The reverse dynamic can be simulated in practice by using CSD [\(Somepalli et al.,](#page-13-14) [2024\)](#page-13-14) as a style feature extractor and replacing y<sup>1</sup> with the extracted style features from the expected terminal state <sup>E</sup>[X<sup>u</sup> 1 |X<sup>u</sup> t ], as discussed in Remark [A.3](#page-16-1). This makes the controller drift causal and non-anticipating future information

## A.3 INCORPORATING STYLE THROUGH MODULATION AND A TERMINAL COST

In this section, we study a control problem where the velocity field is a linear combination of the state and the control variable. This problem is interesting to study because of the following reason. The reverse-SDE dynamic of the standard OU process has a drift field of the form:

$$v(X_t, t) = -X_t - 2\nabla \log p(X_t, t).$$

For a Gaussian prior X<sup>0</sup> ∼ N (0,I), the law of the OU process satisfies ∇ log p (Xt, t) = −Xt, and the corresponding drift field becomes v (Xt, t) = Xt. Our goal is to modulate this drift field using a controller u (X<sup>u</sup> t , t). The result below provides the structure of the optimal control (again in the setting where the terminal objective is known; see Appendix A1).

Proposition A.5. *Suppose* A ∈ R <sup>k</sup>×<sup>d</sup> *be a linear style extractor that operates on the terminal state* X<sup>u</sup> <sup>1</sup> ∈ <sup>R</sup> d *. Let* p<sup>t</sup> *denote* ∇xV ∗ (x, t) *in HJB equation [\(A.1\)](#page-15-2). Given reference style features* y1*, consider the control problem:*

$$\min_{u \in \mathcal{U}} \int_{t_0}^1 \frac{1}{2} \|u(X_t^u, t)\|^2 dt + \frac{\gamma}{2} \|AX_1^u - y_1\|_2^2, \quad (21)$$

$$\text{where } dX_t^u = [X_t^u + u(X_t^u, t)] dt, \quad X_{t_0}^u = x_0, \quad (22)$$

*Then, the optimal controller becomes* u ∗ (t) = −pt*, where the instantaneous state* X<sup>u</sup> <sup>t</sup> = x<sup>t</sup> *and* p<sup>t</sup> *satisfy the following:*

$$\begin{bmatrix} \mathbf{x}_t \\ \mathbf{p}_t \end{bmatrix} = \begin{bmatrix} x_0 e^t - \frac{\gamma}{2} A^T (A \mathbf{x}_1 - y_1) e^{1+t} + \frac{\gamma}{2} A^T (A \mathbf{x}_1 - y_1) e^{1-t} \\ \gamma A^T (A \mathbf{x}_1 - y_1) e^{1-t} \end{bmatrix}.$$

*Proof.* The proof of Proposition [A.5](#page-18-2) is similar to Proposition [A.4](#page-17-7). One key distinction is the set of differential equations obtained using minimum principle theorem [\(Basar et al.,](#page-10-6) [2020\)](#page-10-6). We begin with the Hamiltonian:

$$\begin{aligned} H(\mathbf{x}_t, \mathbf{p}_t, \mathbf{u}_t, t) &= \ell(\mathbf{x}_t, \mathbf{u}_t, t) + \mathbf{p}_t^T (\mathbf{u}_t + \mathbf{x}_t) \\ &= \frac{1}{2} \|\mathbf{u}_t\|^2 + \mathbf{p}_t^T \mathbf{u}_t + \mathbf{p}_t^T \mathbf{x}_t, \end{aligned}$$

which gives us the minimizer of the Hamiltonian u ∗ <sup>t</sup> = −p<sup>t</sup> and its value function becomes V <sup>∗</sup> = min<sup>u</sup><sup>t</sup> H(ut, pt, ut, t) = H(ut, pt, u ∗ t , t) = − 1 2 kptk <sup>2</sup> + p T <sup>t</sup> xt. By the minimum principle theorem [\(Basar et al.,](#page-10-6) [2020\)](#page-10-6),

$$\dot{\mathbf{x}}_t := \frac{d\mathbf{x}_t}{dt} = \nabla_{\mathbf{p}} H(\mathbf{x}_t, \mathbf{p}_t, \mathbf{u}_t^*, t) = -\mathbf{p}_t + \mathbf{x}_t; \quad (23)$$

$$\dot{\mathbf{p}}_t := \frac{d\mathbf{p}_t}{dt} = -\nabla_{\mathbf{x}} H(\mathbf{x}_t, \mathbf{p}_t, \mathbf{u}_t^*, t) = -\mathbf{p}_t; \quad (24)$$

$$\mathbf{x}_{t_0} = x_0; \quad (25)$$

$$\mathbf{p}_{t_N} = \nabla_{\mathbf{x}} h(\mathbf{x}_{t_N}, t_N) = \gamma A^T (A\mathbf{x}_{t_N} - y_1). \quad (26)$$

This leads to a coupled system of differential equations with boundary conditions as given below:

$$\begin{bmatrix} \dot{\mathbf{x}}_t \\ \dot{\mathbf{p}}_t \end{bmatrix} = \begin{bmatrix} 1 & -1 \\ 0 & -1 \end{bmatrix} \begin{bmatrix} \mathbf{x}_t \\ \mathbf{p}_t \end{bmatrix};$$

$$\mathbf{x}_{t_0} = \mathbf{x}_0;$$

$$\mathbf{p}_1 = \gamma A^T (A\mathbf{x}_1 - y_1).$$

This can be solved numerically using ODE solvers, see [\(Fleming & Rishel,](#page-11-9) [2012;](#page-11-9) [Basar et al.,](#page-10-6) [2020\)](#page-10-6) for details. Denote q˙<sup>t</sup> = x˙ t p˙ t and M = 1 −1 0 −1 . We seek a solution of the form q(t) = qe λt. If q(t) is a solution of the above problem, then it must satisfy the following eigen value problem:

$$\mathbf{q}e^{\lambda t} \lambda = \mathbf{M}\mathbf{q}e^{\lambda t}. \quad (27)$$

Writing the characteristic polynomial of [\(27\)](#page-19-0), we get det (M − λI) = 0, which gives the eigen values λ = {1, −1}. Substituting these eigen values, we have

$$\begin{bmatrix} 0 & -1 \\ 0 & -2 \end{bmatrix} \begin{bmatrix} q_1 \\ q_2 \end{bmatrix} = \mathbf{0}, \quad \begin{bmatrix} 2 & -1 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} q_1 \\ q_2 \end{bmatrix} = \mathbf{0},$$

which gives two fundamental solutions. By combining these two, we obtain the final solution

$$\begin{bmatrix} \mathbf{x}_t \\ \mathbf{p}_t \end{bmatrix} = \omega \begin{bmatrix} 1 \\ 0 \end{bmatrix} e^t + \xi \begin{bmatrix} 1 \\ 2 \end{bmatrix} e^{-t},$$

where ω and ξ can be found using the boundary conditions. Since x<sup>0</sup> = x<sup>0</sup> and p<sup>1</sup> = γA<sup>T</sup> (Ax<sup>1</sup> − y1), we get ω = x<sup>0</sup> − γ <sup>2</sup> <sup>A</sup><sup>T</sup> (Ax<sup>1</sup> − y1) e and ξ = γ <sup>2</sup> <sup>A</sup><sup>T</sup> (Ax<sup>1</sup> − y1) e. Substituting the values of ω and ξ, we arrive at

$$\begin{bmatrix} \mathbf{x}_t \\ \mathbf{p}_t \end{bmatrix} = \begin{bmatrix} x_0 e^t - \frac{\gamma}{2} A^T (A \mathbf{x}_1 - y_1) e^{1+t} + \frac{\gamma}{2} A^T (A \mathbf{x}_1 - y_1) e^{1-t} \\ \gamma A^T (A \mathbf{x}_1 - y_1) e^{1-t} \end{bmatrix}.$$

This completes the proof of the proposition.

Summary: Though Appendices [A.1](#page-15-1)[-A.3,](#page-18-0) we have seen the connection between optimal control and diffusion based generation with a personalized terminal constraint. The general strategy has been to derive the optimal controller with known terminal state, and then replace the terminal state in the controller with its estimate using Tweedie's formula. While the controllers so far have an explicit form, in practice, the data distribution is not Gaussian, and thus, we do not have a closed-form expression for the drift of the controller.

This line of analysis, however, points to our method RB-Modulation. As discussed in §[4,](#page-3-0) we incorporate a contrastive style descriptor in our controller's terminal cost and numerically evaluate the drift of the controller at each reverse time step either through back propagation through the score network, or an approximation based on proximal gradient updates.

# B ADDITIONAL EXPERIMENTS

In this section, we provide implementation details and additional experimental evaluation which have been omitted from the main draft due to limited space.

## B.1 IMPLEMENTATION DETAILS

Baselines: We demonstrate the applicability of our method RB-Modulation with StableCascade [\(Pernias et al.,](#page-12-8) [2024\)](#page-12-8) (released before April 2024). To our best knowledge, RB-Modulation is the first framework that introduces new capabilities to StableCascade by incorporating SOC and AFA modules. Since there are no existing training-free personalization baselines designed for StableCascade, we seek alternatives built on other comparable state-of-the-art models such as SDXL [\(Podell et al.,](#page-12-7) [2023\)](#page-12-7) and Muse [\(Chang et al.,](#page-10-9) [2023\)](#page-10-9) .

Among alternate training-free baselines, InstantStyle [\(Wang et al.,](#page-13-3) [2024a\)](#page-13-3) does not directly apply to StableCascade because it requires feature injection into specific layers of an IP-Adapter, which is not available for StableCascade. Similarly, StyleAligned [\(Hertz et al.,](#page-11-2) [2023\)](#page-11-2) relies on DDIM inversion, which is currently applicable only to single-stage diffusion models. In contrast, StableCascade utilizes a two-stage diffusion process, making the application of standard DDIM inversion [\(Song](#page-13-5) [et al.,](#page-13-5) [2021a\)](#page-13-5) infeasible. We run the official source code for InstantStyle[<sup>4</sup>](#page-20-2) and StyleAligned[<sup>5</sup>](#page-20-3) . In the absence of a style description, we use "image in style" for DDIM inversion in StyleAligned. Following InstantStyle [\(Wang et al.,](#page-13-3) [2024a\)](#page-13-3), we also compare with IP-Adapter. We include the quantitative comparison in Table [2,](#page-8-0) and only compare qualitatively with stronger baselines in Figure [2.](#page-6-1)

For completeness, we also compare with training-based baselines: StyleDrop [\(Sohn et al.,](#page-13-2) [2023\)](#page-13-2) and ZipLoRA [\(Shah et al.,](#page-13-1) [2023\)](#page-13-1). Since the official codebase for StyleDrop[<sup>6</sup>](#page-20-4) and ZipLoRA[<sup>7</sup>](#page-20-5) are not publicly available, we use the third-party implementation and follow the training details in the corresponding papers. It takes 5 minutes for training StyleDrop for 1000 steps and 20 minutes for training each LoRA for ZipLoRA. We train each LoRA with only one reference image for both content and styles to make a fair comparison with other methods. Similarly, we train StyleDrop with only one reference image. When a style description is not provided, we follow the original paper [\(Sohn et al.,](#page-13-2) [2023\)](#page-13-2) and use "in a [v\*] style" instead.

Tunable parameters. Our method introduces only two hyper-parameters: stepsize η and optimization steps M in Algorithm [1](#page-5-1). We use DDIM sampling with η = 0.1 and M = 3 for all the experiments. Figure [5](#page-20-6) illustrates an overall pipeline of RB-Modulation.

![](_page_20_Diagram_6.jpeg)

Figure 5: Overall pipeline of RB-Modulation. AFA module replaces the cross-attention processor in the denoising UNet, disentangling the content and style of the reference image using CSD [43].

Content-style composition. The prompt-guided content-style composition task introduces a new layer of complexity beyond stylization. This task necessitates the disentanglement of the text prompt, reference style image, and reference content image through additional conditioning [\(Shah](#page-13-1) [et al.,](#page-13-1) [2023;](#page-13-1) [Wang et al.,](#page-14-7) [2024c;](#page-14-7) [Huang et al.,](#page-11-16) [2024b;](#page-11-16) [Guo et al.,](#page-11-17) [2024\)](#page-11-17). Such complexity poses significant challenges for DDIM inversion [\(Song et al.,](#page-13-5) [2021a\)](#page-13-5) and attention caching mechanisms [\(Hertz et al.,](#page-11-2) [2023\)](#page-11-2) due to the inherent dependencies on multiple reverse paths.

<sup>3</sup>Note that StableCascade and SDXL have comparable performance in prompt alignment whereas Stable-Cascade is more efficient due to a highly compressed semantic latent space [\(Pernias et al.,](#page-12-8) [2024\)](#page-12-8).

<sup>4</sup><https://github.com/InstantStyle/InstantStyle>

<sup>5</sup><https://github.com/google/style-aligned>

<sup>6</sup><https://github.com/aim-uofa/StyleDrop-PyTorch>

<sup>7</sup><https://github.com/mkshing/ziplora-pytorch>

Our AFA module effectively addresses these challenges. It manipulates transformer layers to easily incorporate these additional conditions. The content information is integrated in a manner similar to the style information. Specifically, we use a pre-trained ViT-L/14 model to extract content features in the SOC framework and update the latent embeddings concurrently via the AFA module, using an additional set of keys and values illustrated in Figure [6.](#page-21-2)

![](_page_21_Diagram_2.jpeg)

Figure 6: Attention Feature Aggregation (AFA): Within the cross-attention layers, the keys and values from the previous layers (K,V ), text embedding (Kp,Vp), reference style image (Ks,Vs) and reference content image (Kc,Vc) are concatenated and processed separately to disentangle the information, which is followed by an averaging layer for the output. Kc,V<sup>c</sup> and only used for content-style composition.

Furthermore, to better preserve the identity of the foreground content, we extract the desired content using LangSAM[<sup>8</sup>](#page-21-3) based on the content prompt. This step is optional but offers more user control when multiple subjects are present in the reference image.

#### B.2 IMPLEMENTATION USING LARGE-SCALE DIFFUSION MODELS

The exact implementation of our control problem [\(3\)](#page-3-3) is given in Algorithm [1](#page-5-1), which follows from our theoretical insights. In practice, our controller encounters a challenge when the generative model contains billions of parameters as in StableCascade [\(Pernias et al.,](#page-12-8) [2024\)](#page-12-8) due to back propagation through the score network, as discussed in §[4.](#page-3-0) Our strategy to overcome this practical challenge involves a proximal gradient update, given in Line 7-8 of Algorithm [2](#page-5-2). To accelerate the sampling process, we run a few steps (M = 3) of gradient descent after initializing x<sup>0</sup> = <sup>E</sup> [X<sup>u</sup> 0 |X<sup>u</sup> <sup>t</sup> = xt], resulting in only two hyperparameters to tune: stepsize η and the number of optimization steps M. Further, since the CSD model expects a clean image to extract style features, we apply the previewer model available in StableCascade on the terminal state before extracting style features. After obtaining the final personalized latent using our Algorithm [1](#page-5-1) and Algorithm [2](#page-5-2), we follow the decoding process as per the inference pipeline of the adopted generative model. In Table [4,](#page-21-0) we show the computational overhead of our method in comparison with competing methods.

Table 4: RB-Modulation matches the speed of training-free methods and offers 5-20X speedup over training-based methods like StyleDrop [11] and ZipLoRA [10]. For instance, StyleDrop and ZipLoRA require 300 seconds (s) and 1200s, respectively, for training specific components, in addition to their standard inference times of 30s and 40s. RB-Modulation does not use DDIM inversion or additional parameters in the UNet, thus further reducing the computational overhead.

| Method        |        | Runtime (s) | Training-Free | DDIM Inv. | Params in UNet         |
|---------------|--------|-------------|---------------|-----------|------------------------|
| IP-Adapter    | [21]   | 21          | Yes           | Yes       | Adapters               |
| StyleAligned  | [12]   | 39          | Yes           | Yes       | No                     |
| InstantStyle  | [13]   | 22          | Yes           | Yes       | Adapters, ControlNets  |
| StyleDrop     | [11]   | 300+30      | No            | No        | Adapters               |
| ZipLoRA       | [10]   | 1200+40     | No            | No        | 2 LoRAs, 1 Merge layer |
| RB-Modulation | (ours) | 44          | Yes           | No        | No                     |

<sup>8</sup><https://github.com/luca-medeiros/lang-segment-anything>

![](_page_22_Figure_1.jpeg)

Figure 7: Impact of style descriptions in the prompt: (a) When style descriptions are provided, all methods yield better results. (b) Without style descriptions (*e.g*., hard for users to describe in text), alternative methods could struggle to capture the intended style in the reference image. Our method offers consistent stylization even without explicit style descriptions.

#### B.3 IMPACT OF HYPERPARAMETERS ON CONTROLLING STYLE AND CONTENT FEATURES

As detailed in §[4](#page-3-0) and the ablation study in §[6.1,](#page-7-1) SOC helps disentangle the style and the prompt information by updating the drift field in the standard reverse-SDE. We study the impact of the two hyperparameters present in Algorithm [1](#page-5-1) and Algorithm [2](#page-5-2) that enables this disentanglement, as shown in Figure [8.](#page-22-0) We found better disentanglement when the step size η = 0.1 and the number of optimization steps M = 3. However, increasing the step size further results in style image information leaking into the output (top row). Additionally, adding more optimization steps increases computational overhead without yielding much performance gain (bottom row).

![](_page_22_Figure_5.jpeg)

Figure 8: Qualitative results of different tunable hyperparameters: Improved style-prompt disentanglement are shown when increasing to our best configurations optimization step size η = 0.1 and optimization steps M = 3.

#### B.4 STYLE DESCRIPTION IN TEXT PROMPTS FOR BETTER ASSIMILATION OF UNIQUE STYLES

In addition to the quantitative analysis in §[6.1,](#page-7-1) Figure [7](#page-22-1) demonstrates that our method generates consistent stylized results with and without the style description. In contrast, the alternatives fail to accurately follow the prompt when the style description is absent. Although all results show noticeable improvement when the style description is provided, it is often challenging for users to describe styles in many real-world scenarios. We believe our early results by RB-Modulation will pave the way for interesting future research along this direction.

We present additional qualitative results on stylization with (Figure [11\)](#page-27-0) and without (Figure [12\)](#page-28-0) style descriptions using StyleAligned dataset [\(Hertz et al.,](#page-11-2) [2023\)](#page-11-2). Our results consistently align with the reference style and the prompt, while other methods encounter several issues: (1) difficulty in following prompt guidance, (2) information leakage from the style reference image, and (3) failure

![](_page_23_Picture_1.jpeg)

Figure 9: A gallery of additional qualitative results on stylization using RB-Modulation.

to achieve reasonable prompt/style alignment in the absence of style descriptions. Figure [9](#page-23-1) presents a gallery of text-driven stylization results using RB-Modulation.

#### B.5 EVALUATION CHALLENGES IN MEASURING STYLE AND CONTENT LEAKAGE

In §[6,](#page-6-0) we discussed the limitations of metrics used in previous works [\(Sohn et al.,](#page-13-2) [2023;](#page-13-2) [Hertz et al.,](#page-11-2) [2023;](#page-11-2) [Shah et al.,](#page-13-1) [2023\)](#page-13-1), such as DINO [\(Caron et al.,](#page-10-7) [2021\)](#page-10-7) and CLIP-I score [\(Radford et al.,](#page-12-11) [2021\)](#page-12-11). To quantify these limitations, we use results from our ablation study shown in Figure [3.](#page-8-1) As illustrated in Figure [10,](#page-24-0) DINO and CLIP-I scores are not well-suited for measuring style similarity in the presence of content leakage. This is because images with high semantic correlations to the reference style image consistently receive higher scores. For instance, in the top row, although the last two columns visually align more closely with the isometric illustration styles of the reference image, the DirectConcat output featuring a lighthouse receives higher scores. The margin is particularly pronounced for CLIP-I score.

A similar observation can be made in the bottom row, where images containing train-related objects receive higher scores regardless of their stylistic similarity. Conversely, images with less content leakage (as seen in the last column) are assigned lower scores. This indicates that DINO and CLIP-I scores prioritize semantic content over stylistic fidelity, thus failing to accurately measure style similarity in scenarios where content leakage prevails.

On the other hand, our final method (last column), which combines AFA and SOC, demonstrates high scores for both prompt alignment metrics: ImageReward [\(Xu et al.,](#page-14-0) [2024\)](#page-14-0) and CLIP-T [\(Radford](#page-12-11) [et al.,](#page-12-11) [2021\)](#page-12-11). This method also shows higher user preference, as evidenced in Table [1.](#page-7-0) In contrast, the DirectConcat results suffer from information leakage and poor alignment with the prompt, resulting in significantly lower or even negative reward scores.

In the ablation study, our primary focus is on the disentanglement of prompts and reference styles. The conventional metrics fail to accurately reflect true performance due to information leakage. Consequently, we emphasize qualitative demonstrations and place greater importance on user study results, as shown in Table [1,](#page-7-0) similar to previous approaches [\(Hertz et al.,](#page-11-2) [2023;](#page-11-2) [Sohn et al.,](#page-13-2) [2023\)](#page-13-2).

| Reference style StableCascade | DirectConcat | AFA only | SOC only | AFA + SOC |
|-------------------------------|--------------|----------|----------|-----------|
| 0.55                          | 0.80         | 0.68     | 0.70     | 0.65      |
| 0.28                          | 0.23         | 0.29     | 0.28     | 0.29      |
| 1.49                          | 0.06         | 1.39     | 1.11     | 1.43      |
| 0.57                          | 0.82         | 0.66     | 0.61     | 0.61      |
| 0.48                          | 0.74         | 0.66     | 0.75     | 0.72      |
| 0.27                          | 0.23         | 0.27     | 0.22     | 0.27      |
| 0.99                          | -1.63        | 0.17     | -1.20    | 0.40      |
| 0.50                          | 0.88         | 0.70     | 0.73     | 0.72      |

Figure 10: Comparison of different evaluation metrics: The StableCascade output is provided for reference because it doesn't use the reference style image. The highest score for each metric is marked bold with underscore. We compare four metrics: ImageReward and CLIP-T score for prompt alignment, DINO and CLIP-I score for style alignment. The prompt for the top row is "A cat" and for the bottom row is "A piano".

#### B.6 MORE QUALITATIVE RESULTS ON STYLIZATION AND CONTENT-STYLE COMPOSITION

We also showcase results on consistent style generation using user defined prompts in Figure [13.](#page-29-0) Our results with different prompts consistently align with the styles while introducing various scenarios following the prompts. The other methods face challenges like information leakage (*e.g*. hiking boots and the monocular) and monotonous scenes (*e.g*. InstantStyle). Note that the original StyleDrop paper [\(Sohn et al.,](#page-13-2) [2023\)](#page-13-2) has mentioned its difficulty when training with one image without description. We keep the results for completeness even though they are less satisfying. In Figure [15,](#page-30-0) we provide additional comparison with training-based and training-free personalization approaches. Figure [14](#page-30-1) shows stylization given hand drawn reference style images: plastic crayon[<sup>9</sup>](#page-24-1) , pencil sketch[<sup>10</sup>](#page-24-2), and commercial paint[<sup>11</sup>](#page-24-3). In Figure [16,](#page-31-0) we show qualitative results obtained by integrating the AFA and SOC modules in SDXL [\(Podell et al.,](#page-12-7) [2023\)](#page-12-7) pipeline, justifying the plugand-play nature of RB-Modulation.

Compatibility with ControlNet. Our method readily adapts to layout guidance via Control-Net [\(Zhang et al.,](#page-14-2) [2023\)](#page-14-2), as shown in Figure [17.](#page-31-1) Since ControlNet enhances the denoising network, the proposed method effectively minimizes the terminal cost associated with the expected terminal state, ensuring that SOC remains practical and effective. Furthermore, the AFA module integrates seamlessly by replacing the default attention processor in the denoising network, maintaining its functionality even when ControlNet is employed.

Controllability of AFA Module. Figure [18](#page-32-0) demonstrates the precise control provided by the AFA module. The pair (Kp, Vp) is computed using the given prompt (e.g., "a cat") without using text description of the reference style image, and (Ks, Vs) using the style attention head of the CSD feature extractor applied to the reference style image. By gradually increasing the strength of the

<sup>9</sup><https://ar.pinterest.com/pin/742953269772065667/>

<sup>10</sup><https://www.pinterest.com/pin/509891989063791950/>

<sup>11</sup><https://www.pinterest.com/pin/ms-paint-drawing-art--690106342901777263/>

style image embedding, our method progressively incorporates features from the reference style image, enabling fine-grained control over stylization.

Figure [19](#page-32-1) demonstrates the ability of our method RB-Modulation to generate novel and unseen styles by continuously interpolating between CSD style embedding of two reference style images.

In Figure [21,](#page-34-0) we demonstrate more qualitative results for content-style composition Figure [22](#page-35-0) shows the impact of content image in content-style composition. Figure [23](#page-35-1) highlights the robustness of RB-Modulation in capturing content-specific features independently of color.

#### B.7 ADDITIONAL RELATED WORK

In this section, we discuss missing related works from the main paper. DiffusionDisentanglement [\(Wu et al.,](#page-14-4) [2023\)](#page-14-4) relies on VGG 16 for perceptual loss and ViT/B-32 for directional CLIP loss, which is prone to content leakage [\(Wang et al.,](#page-13-3) [2024a\)](#page-13-3). In contrast, our method injects features exclusively from the style attention head of the fine-tuned CSD-CLIP model, ensuring better content-style disentanglement in the AFA module. Additionally, our approach introduces an optimal controller framework to minimize a terminal cost, offering a richer design space and superior controllability compared to [\(Wu et al.,](#page-14-4) [2023\)](#page-14-4). Lastly, our method reduces sampling bias by optimizing the controller u in Algorithm [1,](#page-5-1) unlike [\(Wu et al.,](#page-14-4) [2023\)](#page-14-4), which can provably fail to sample from the correct posterior.

In FreeDoM [\(Yu et al.,](#page-14-8) [2023\)](#page-14-8), the conditional guidance term ∇<sup>x</sup><sup>t</sup> log p(·|xt) is approximated by the gradient of an energy function, ∇<sup>x</sup><sup>t</sup> E(·, xt). Our Algorithm [1](#page-5-1) differs by replacing ∇<sup>x</sup><sup>t</sup> log p(·|xt) with a controller u, optimized to minimize this approximation error. Algorithm 2 in FreeDom introduces a time-travel resampling strategy to mitigate poor guidance problem in their Algorithm 1 by iteratively noising and denoising the intermediate latents. While effective, this process is computationally expensive. In contrast, our approach (Algorithm [2](#page-5-2)) is grounded in optimal control, where we optimize the expected terminal state to satisfy constraints, such as aligning the style of the generated image with the input. Thus, our Algorithm [2](#page-5-2) avoids the need for gradient computation through the denoising score network, which is particularly expensive for large-scale models like SDXL or StableCascade. Additionally, we propose a novel attention processor, namely AFA module to disentangle content and style, whereas FreeDoM uses the standard attention processor, known to suffer from content leakage [\(Hertz et al.,](#page-11-2) [2023;](#page-11-2) [Wang et al.,](#page-13-3) [2024a\)](#page-13-3).

PARASOL [\(Tarres et al.](#page-13-9) ´ , [2024\)](#page-13-9) and Diff-NST [\(Ruta et al.,](#page-13-10) [2023\)](#page-13-10) are training-based methods, while our approach is entirely training-free. PARASOL requires supervised data via a cross-modal search (Section 3.1 in [\(Tarres et al.](#page-13-9) ´ , [2024\)](#page-13-9)) to train both the denoising U-Net and a projector network. Diff-NST [\(Ruta et al.,](#page-13-10) [2023\)](#page-13-10) trains the attention processor by targeting the 'V' values within the denoising U-Net architecture. In contrast, our method uses two training-free modules: the AFA module replaces the default attention processor in the denoising U-Net to disentangle content and style, and the SOC module minimizes a terminal cost to enhance stylization and content-style composition.

#### B.8 HUMAN EVALUATION TO DISCERN HIGHLY SUBJECTIVE NATURE OF STYLE

We conduct a user study with 155 participants via Amazon Mechanical Turk using 100 styles from the StyleAligned dataset [\(Hertz et al.,](#page-11-2) [2023\)](#page-11-2). The study requires no personally identifiable information of the participants. There is no risk incurred and no vulnerable population. The standard guidelines have been followed while conducting the user study.

We first provide participants with instructions to familiarize them with the relevant terminologies. For each style, we randomly sample three outputs using three different prompts. Participants see two rows of model outputs in random order (3 images per row) and answer 3 questions, as illustrated in Figure [20.](#page-33-0)

- 1. In which row below, the images align better with the reference style image?
- 2. In which row below, the images align better with the reference text prompt above each image?
- 3. In which row below, the images overall align better with the reference style image AND the text prompt above each image AND with high quality?

For each question, participants choose one of three options. We collect 8 responses for each question, with each question comparing our method against one of the alternatives. In total, we gathered 7,200 responses.

#### B.9 FAILURE CASES OF TRAINING-FREE STYLIZATION USING RB-MODULATION

In Figure [24,](#page-35-2) we illustrate stylization of different letters using a single reference style image. Although our method captures the intended style and generates prompted letters, we notice that there is an inherent tendency to generate upper-case letters (Figure [24](#page-35-2) (a)), even though it is prompted to generate lower-case letters. Upon further investigation, we observed that this issue stems from the underlying generative model StableCascade, as shown in Figure [24](#page-35-2) (b). This highlights a crucial limitation of our method. As a training-free method, RB-Modulation shares a concern with other training-free methods [\(Wang et al.,](#page-13-3) [2024a;](#page-13-3) [Hertz et al.,](#page-11-2) [2023;](#page-11-2) [Jeong et al.,](#page-11-3) [2024\)](#page-11-3) that the performance is influenced by the original generative prior.

## B.10 LIMITATIONS

In this paper, we proposed a framework and demonstrated its efficacy by incorporating a style descriptor [\(Somepalli et al.,](#page-13-14) [2024\)](#page-13-14) in a pre-trained diffusion model [\(Pernias et al.,](#page-12-8) [2024\)](#page-12-8). The inherent limitations of the style descriptor or diffusion model might propagate into our framework. We believe these limitations can be addressed by an appropriate descriptor or a generative prior.

![](_page_27_Figure_1.jpeg)

![](_page_27_Picture_2.jpeg)

Figure 11: Additional qualitative results for stylization with style description: While the alternative methods face challenges like following the prompts (*e.g*., multiple airplanes instead of an airplane) and information leakage (*e.g*., the clouds on the cornflake bowl and the guitar in the milkshake image), our method demonstrates strong performance on both prompt and style alignment. Style description is in blue. 28

![](_page_28_Picture_1.jpeg)

![](_page_28_Figure_2.jpeg)

Figure 12: Additional qualitative results for stylization without style description: StyleAligned and StyleDrop show severe performance drop after removing the style descriptions (*e.g*., see fireman and cat images). InstantStyle results show more information leakage (*e.g*., the pink ladybug and leopard), whereas no obvious performance drop is observed in our results.

![](_page_29_Figure_1.jpeg)

![](_page_29_Picture_2.jpeg)

Figure 13: Additional qualitative results for consistent stylization for user defined prompts: With no style description, our results demonstrate more diversity while following the styles and prompts. InstantStyle results show monotonous scenes and StyleAligned results suffer from severe information leakage. We report StyleDrop results for completeness and it is known to perform worse with no style description and single training image [\(Sohn et al.,](#page-13-2) [2023\)](#page-13-2).

![](_page_30_Picture_1.jpeg)

![](_page_30_Figure_2.jpeg)

Figure 14: Qualitative results for hand-drawn reference style images. The proposed method is agnostic to real or generated reference images. Given hand drawn reference style images (e.g., "paint" from a commercial service provider) and desired text prompts (e.g., "a tiger"+style description), RB-Modulation captures the reference style in the generated content image. Please see §[B.6](#page-24-4) for the reference style image credits.

![](_page_30_Figure_4.jpeg)

Figure 15: Qualitative comparison with classical personalization methods. The proposed method significantly outperforms other training-free methods while remaining comparable to or better than classical training-based personalization approaches. Prompt:"a baby penguin in 3d rendering style."

![](_page_31_Picture_1.jpeg)

Figure 16: Qualitative results using SDXL [\(Podell et al.,](#page-12-7) [2023\)](#page-12-7) as base model. This verifies the plug-and-play nature of RB-Modulation for training-free personalization.

![](_page_31_Picture_3.jpeg)

![](_page_31_Figure_4.jpeg)

Figure 17: Qualitative results demonstrating compatibility with ControlNet [\(Zhang et al.,](#page-14-2) [2023\)](#page-14-2). Given the Canny edge map of a reference content and an image of a reference style, the proposed method effectively controls the pose of the generated samples while accurately capturing the desired style.

![](_page_32_Figure_1.jpeg)

Figure 18: Qualitative results showing controllability of our method for stylization. By progressively increasing the strength of the style image embedding derived from the CSD style descriptor, our method gradually integrates features from the reference style image, providing fine-grained control over stylization.

![](_page_32_Figure_3.jpeg)

![](_page_32_Picture_4.jpeg)

Figure 19: Qualitative results showing interpolation of two different reference style images. The interpolation strength parameter provides additional control for blending features from multiple reference styles (e.g., "a lighthouse in mosaic art style" → "a lighthouse in cyberpunk art style"). This highlights RB-Modulation's capability to generate novel and previously unseen styles.

![A screenshot of a statement testing set for a style similarity prompt. The set includes a 'Instructions' section with a 'Review of the definition:' and a 'Reference Image' section with a 'Reference Image' icon, a 'Reference Image' icon with a temple icon, a 'Reference Image' icon with a dog icon, and a 'Reference Image' icon with a lion icon. The 'Select an option' section includes 'Top row' and 'Bottom row' checkboxes. A 'Submit' button is also visible.]()Figure 20: User study interface: Three randomly sampled outputs are shown for each method given a style reference image, forming two rows of images. The users are asked to answer three questions on (1) style alignment (2) prompt alignment and (3) overall alignment and quality.

![](_page_34_Picture_1.jpeg)

Figure 21: Additional qualitative results for content-style composition: Our results show better prompt and style alignment while preserving reference content without leaking contents from the reference style images (*e.g*. background of the first column and fruits in the last column,). Unlike compared baselines, our method is not restricted to a fixed pose of the reference content image, illustrating sample diversity.

![](_page_35_Picture_1.jpeg)

Figure 22: Qualitative results on content-style composition to illustrate the impact of content image. Excluding the content reference image (i.e., removing K<sup>c</sup> and V<sup>c</sup> from the AFA module) results in a loss of content details, such as the dog breed and car type, as highlighted in the red box.

![](_page_35_Picture_3.jpeg)

Figure 23: Qualitative comparisons for content-style composition by graying out the reference content image. Notably, the content (e.g., dog) is effectively transferred even after the grayscale transformation, demonstrating the robustness of our method in capturing and transferring content-specific features independently of color.

![](_page_35_Figure_5.jpeg)

Figure 24: Failure cases for stylization: The top row shows the results of our method, RB-Modulation, while the bottom row displays the results of the backbone, StableCascade. Notably, the stylized images do not adhere to the prompt,"lower-case letter". This highlights the limitations imposed by the pre-trained generative priors on the capabilities of training-free personalization models (top row).