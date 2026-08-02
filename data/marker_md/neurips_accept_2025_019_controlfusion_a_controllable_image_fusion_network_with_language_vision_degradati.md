# ControlFusion: A Controllable Image Fusion Network with Language-Vision Degradation Prompts

Linfeng Tang1,†, Yeda Wang1,†, Zhanchuan Cai<sup>2</sup> , Junjun Jiang<sup>3</sup> , Jiayi Ma1,\* <sup>1</sup>Electronic Information School, Wuhan University

<sup>2</sup>School of Computer Science and Engineering, Macau University of Science and Technology <sup>3</sup>Faculty of Computing, Harbin Institute of Technology

linfeng0419@gmail.com, wangyeda@whu.edu.cn, zccai@must.edu.mo, jiangjunjun@hit.edu.cn, jyma2010@gmail.com,

# Abstract

Current image fusion methods struggle with real-world composite degradations and lack the flexibility to accommodate user-specific needs. To address this, we propose ControlFusion, a controllable fusion network guided by language-vision prompts that adaptively mitigates composite degradations. On the one hand, we construct a degraded imaging model based on physical mechanisms, such as the Retinex theory and atmospheric scattering principle, to simulate composite degradations and provide a data foundation for addressing realistic degradations. On the other hand, we devise a prompt-modulated restoration and fusion network that dynamically enhances features according to degradation prompts, enabling adaptability to varying degradation levels. To support user-specific preferences in visual quality, a text encoder is incorporated to embed user-defined degradation types and levels as degradation prompts. Moreover, a spatial-frequency collaborative visual adapter is designed to autonomously perceive degradations from source images, thereby reducing complete reliance on user instructions. Extensive experiments demonstrate that Control-Fusion outperforms SOTA fusion methods in fusion quality and degradation handling, particularly under real-world and compound degradations. The source code is publicly available at <https://github.com/Linfeng-Tang/ControlFusion>.

# 1 Introduction

Image fusion is a crucial technique in image processing. By effectively leveraging complementary information from multiple sources, the limitations associated with information collection by singlemodal sensors can be significantly mitigated [\[41\]](#page-12-0). Among the research areas in this field, infraredvisible image fusion (IVIF) has garnered considerable attention. IVIF integrates essential thermal information from infrared (IR) images with the intricate texture details from visible (VI) images, offering a richer and more complete depiction of the scene [\[46\]](#page-12-1). By harmonizing diverse information and producing visually striking results, IVIF has found extensive applications in areas such as security surveillance [\[47\]](#page-12-2), military detection [\[22\]](#page-11-0), scene understanding [\[44\]](#page-12-3), and assisted driving [\[1\]](#page-10-0), *etc*.

Recently, IVIF has emerged as a focal point of research, leading to rapid advancements in related methods. Based on the adopted network architectures, these methods can be categorized into convolutional neural network-based [\[29,](#page-11-1) [51\]](#page-12-4), autoencoder-based [\[10,](#page-10-1) [11\]](#page-10-2), generative adversarial network-based [\[19,](#page-11-2) [15\]](#page-10-3), Transformer-based [\[20,](#page-11-3) [45\]](#page-12-5), and diffusion model-based [\[50,](#page-12-6) [32\]](#page-11-4) methods. In addition, from a functional perspective, they can be grouped into visual-oriented [\[29,](#page-11-1) [51\]](#page-12-4), joint registration-fusion [\[27,](#page-11-5) [37\]](#page-12-7), semantic-driven [\[28,](#page-11-6) [16\]](#page-10-4), and degradation-robust [\[31,](#page-11-7) [42,](#page-12-8) [43\]](#page-12-9) schemes.

<sup>†</sup> Equal contribution; \* Corresponding author.

![](_page_1_Picture_0.jpeg)

Figure 1: Comparison across real-world, composite degradation, and varying degradation levels.

Notably, although degradation-robust schemes can mitigate degradation to some extent, certain limitations exist. First, existing restoration-fusion methods often adopt simplistic strategies for training data construction, overlooking the domain gap between simulated data and realistic images, which hampers their generalizability in practical scenarios, as illustrated in Fig. [1](#page-1-0) (I). Second, they are tailored for specific or single types of degradation, making them ineffective in handling more complex composite degradations, as illustrated in Fig. [1](#page-1-0) (II). Finally, as shown in Fig. [1](#page-1-0) (III), existing methods lack degradation level modeling, causing a sharp decline in performance as degradation intensifies. Moreover, they lack flexibility to adapt fusion results to diverse user preferences.

To overcome these limitations, we propose a versatile and controllable fusion model based on language-vision prompts, termed ControlFusion. On the one hand, we introduced a physics-driven degradation imaging model that differs from existing approaches by simultaneously simulating degradation processes for both visible and infrared modalities with high precision. This model not only effectively narrows the gap between synthetic data and real-world images but also provides crucial data support for addressing complex multi-modal composite degradation challenges. On the other hand, we develop a prompt-modulated image restoration and fusion network to generate high-quality fusion results. The prompt-modulated module enables our network to dynamically adjust feature distribution based on degradation characteristics (which can be specified by users), achieving robust feature enhancement. This also allows our method to respond to diverse user requirements. Furthermore, we devise a spatial-frequency visual adapter that combines frequencydomain degradation priors to directly extract degradation cues from degraded inputs, eliminating the heavy reliance on user instructions, as customizing prompts for each scene is time-consuming and labor-intensive. As shown in Fig. [1,](#page-1-0) benefiting from the aforementioned designs, our method excels in handling both real-world and composite degradation scenarios and can flexibly respond to user needs. In summary, our main contributions are as follows:

- We propose ControlFusion, a versatile image restoration and fusion framework that uniformly models diverse degradation types and degrees, using textual and visual prompts as a medium. Specifically, its controllability enables it to respond to user-specific customization needs.
- A spatial-frequency visual adapter is devised to integrate frequency characteristics and directly extract text-aligned degradation prompts from visual images, enabling automated deployment.
- A physics-driven imaging model is developed, integrating physical mechanisms such as the Retinex theory and atmospheric scattering principle to bridge the gap between synthetic data and real-world images, while taking into account the degradation simulation of infrared-visible dual modalities.

# 2 Related Work

Image Fusion. Earlier visual-oriented fusion approaches primarily concentrated on merging complementary information from multiple modalities and improving visual fidelity. The mainstream network architectures primarily include CNN-based [\[14,](#page-10-5) [48\]](#page-12-10), AE-based [\[11,](#page-10-2) [49\]](#page-12-11), GAN-based [\[19,](#page-11-2) [15\]](#page-10-3), Transformers [\[20\]](#page-11-3) and diffusion models [\[50\]](#page-12-6). Furthermore, several schemes including joint registration and fusion [\[33,](#page-11-8) [37\]](#page-12-7), semantic-driven [\[28,](#page-11-6) [15\]](#page-10-3), and degradation-robust [\[39,](#page-12-12) [42,](#page-12-8) [38\]](#page-12-13) methods, are proposed to broaden the practical applications of image fusion. For instance, Tang et al. [\[30\]](#page-11-9) and Liu et al. [\[17\]](#page-10-6) developed corresponding solutions for illumination distortions and noise interference, respectively. In addition, [\[39\]](#page-12-12) proposed a image restoration and fusion network to address various degradations. However, it fails to handle mixed degradations and struggles to generalize to real-world scenarios. In particular, it relies on tedious manual efforts to customize text prompts for each scene, hindering large-scale automated deployment.

Image Restoration with Vision-Language Models. With the advancement of deep learning toward multimodal integration, the image restoration field has progressively evolved into a new paradigm of text-driven schemes. CLIP [\[25\]](#page-11-10) establishes visual-textual semantic representation alignment through dual-stream Transformer encoders pre-trained on 400 million image-text pairs. This framework lays the foundation for the widespread adoption of *Prompt Engineering* in computer vision. Recent studies integrate CLIP encoders with textual user instructions, proposing generalized restoration frameworks including PromptIR [\[23\]](#page-11-11), AutoDIR [\[8\]](#page-10-7), and InstructIR [\[2\]](#page-10-8), which effectively handle diverse degradation types. Moreover, SPIRE [\[24\]](#page-11-12) further introduces fine-grained textual restoration cues by quantifying degradation severity levels and supplementing semantic information for precision restoration. To eliminate reliance on manual guidance, Luo et al. [\[18\]](#page-11-13) developed DA-CLIP by fine-tuning CLIP on mixed-degradation datasets, enabling degradation-aware embeddings to facilitate distortion handling. However, existing unified restoration methods are primarily designed for natural images, which exhibit notable limitations when processing multimodal images (*e.g.*, infrared-visible).

# 3 Physics-driven Degraded Imaging Model

Due to the differences in the imaging mechanisms of infrared and visible, the types of degradation they face also vary. To tackle the complex and variable degradation challenges, we propose a physics-driven imaging model for infrared and visible images aimed at reducing the domain gap between simulated data and real-world imagery. Infrared (IR) images usually suffer from sensorrelated interference, such as stripe noise and low contrast. While Visible (VI) images are typically degraded by illumination conditions (low light, over-exposure), weather (rain, haze), and sensorrelated issues (noise, blur). Given a clear image Im(m ∈ {ir, vi}), the proposed imaging model can be mathematically represented as:

$$D_m = \mathcal{P}_s \left( \mathcal{P}_w \left( \mathcal{P}_i \left( I_m \right) \right) \right), \quad (1)$$

where D<sup>m</sup> is the corresponding degraded image, and P<sup>i</sup> , Pw, and P<sup>s</sup> represent illumination-, weather-, and sensor-related distortions, respectively.

Sensor-related Distortions. Sensor-related distortions encompass various types of noise, motion blur, and contrast degradation. Contrast degradation and stripe noise of infrared images can be modeled as:

$$D_{ir}^s = \mathcal{P}_s(I_{ir}) = \alpha \cdot I_{ir} + \mathbf{1}_H \mathbf{n}^\top, \quad (2)$$

where α is a constant less than 1 for contrast reduction, 1<sup>H</sup> ∈ <sup>R</sup> <sup>H</sup> is an all-ones column vector and n ∈ <sup>R</sup><sup>W</sup> represents the column-wise Gaussian noise vector sampled from N (0, ϵ<sup>2</sup> ) with ϵ ∈ [1, 15]. Moreover, Gaussian noise and motion blur in source images can be modeled as:

$$D_m^s = \mathcal{P}_s(I_m) = I_m * K(N, \theta) + \mathcal{N}(0, \sigma^2), \quad (3)$$

where N (0, σ<sup>2</sup> ) represents Gaussian noise with σ ∈ [5, 20]. ∗ denotes convolution with a blur kernel K(N, θ) = <sup>1</sup> <sup>N</sup> Rθ(δ<sup>c</sup> ⊗ 1<sup>N</sup> ) constructed by rotating an impulse δ<sup>c</sup> ⊗ 1<sup>N</sup> with random angle R<sup>θ</sup> (θ ∈ [10, 80]), using <sup>1</sup> N to normalize the kernel energy. Here, N ∈ [3, 12] controls the blur level.

Illumination-related Distortions. Following the theoretical framework of Retinex, the formation of illumination-degraded images D<sup>i</sup> vi is mathematically modeled as:

$$D_{vi}^i = \mathcal{P}_i(I_{vi}) = \frac{I_{vi}}{L} \cdot L^\gamma, \quad (4)$$

where L is the illumination map estimated by LIME [\[5\]](#page-10-9). γ ∈ [0.5, 3] controls the illumination level.

Weather-related Distortions. According to the methodologies in [\[21,](#page-11-14) [12\]](#page-10-10), we employ the following formula to simulate weather-related degradations (*i.e.* rain and haze):

$$D_{vi}^w = \mathcal{P}_w(I_{vi}) = I_{vi} \cdot t + A(1-t) + R, \quad (5)$$

where t and A denote the transmission map and atmospheric light, respectively. And t is defined by the exponential decay of light, expressed as t = e −βd(x) , where the haze density coefficient β ∈ [0.5, 2.0]. d(x) refers to the scene depth, estimated by DepthAnything-V2 [\[40\]](#page-12-14). Besides, A is constrained within [0.3, 0.9] for realistic simulation.

![](_page_3_Diagram_10.jpeg)

Figure 2: The overall framework of our controllable image fusion network.

Based on Eq. (1), we construct a multi-modal composite **Degradation Dataset** with four **Levels** (DDL-12), comprising 12 distinct degradations. We selecte 2, 050 high-quality clear images from RoadScene [36], LLVIP [7], and MSRS [29] datasets. Subsequently, the degraded imaging model is employed to synthesize degraded images with 12 types and 4 levels of degradation. Finally, the DDL-12 dataset contains approximately 48, 000 training image pairs and 4, 800 test image pairs.

## 4 Methodology

### 4.1 Problem Formulation

Given two source images  $I_{ir} \in \mathbb{R}^{H \times W \times 1}$  and  $I_{vi} \in \mathbb{R}^{H \times W \times 3}$ , the typical fusion paradigm employs a network  $\mathcal{N}_f$  to synthesize the fused image  $I_f$ , expressed as:  $I_f = \mathcal{N}_f(I_{ir}, I_{vi})$ . However, in complex scenarios, source images usually suffer from degradation interference, thus the advanced fusion paradigm must take both image restoration and fusion into account. While the concatenation approach is straightforward, it does not model recovery and fusion as an end-to-end optimization process, yielding suboptimal outcomes. As illustrated in Fig. 2, we propose a controllable paradigm that couples restoration and fusion with degradation prompts, termed ControlFusion, to overcome this limitation. On the one hand, the coupled approach enhances task synergy. On the other hand, leveraging degradation prompts to modulate the restoration and fusion process enables the network to adapt to diverse degradation distributions while meeting user-specific customization requirements. The proposed restoration and fusion paradigm can be defined as:  $I_f = \mathcal{N}_{rf}(I_{ir}, I_{vi}, p \mid \Omega)$ , where  $\mathcal{N}_{rf}$  is a restoration and fusion network,  $P$  denotes degradation prompts, and  $\Omega$  indicates the degradation set. Our ControlFusion employs a two-stage optimization protocol. Stage I aligns textual prompts with visual embedding, while Stage II optimizes the overall framework.

### 4.2 Stage I: Textual-Visual Prompts Alignment

**Spatial-Frequency Collaborative Visual Adapter.** To achieve unified multimodal degradation representation, we first employ text semantics for explicit degradation modeling. Using a pre-trained CLIP [25] text encoder  $\mathcal{E}_t$ , the user instruction  $T_{instr}$  is converted into text embedding:  $p_{text} = \mathcal{E}_t(T_{instr})$ . However, text-dependent models limit deployment flexibility. We therefore develop a spatial-frequency collaborative visual adapter (SFVA) to extract visual embeddings directly from images while maintaining semantic alignment with  $p_{text}$ .

As shown in Fig. 3, degraded images exhibit distinct spectral characteristics, indicating frequency features contain rich degradation priors. Our SFVA contains dual branches: In the frequency branch,

![](_page_4_Figure_0.jpeg)

Figure 3: The visualization of various types of degradation in the spatial and frequency domains.

Fast Fourier Transform (FFT) and CNN extract frequency features:

$$F_{fre}^m = \sum_{x=0}^{W-1} \sum_{y=0}^{H-1} D_m(x, y) e^{-j2\pi \left(\frac{yx}{W} + \frac{yy}{H}\right)}, F_{fre} = \text{Linear} \left( \text{Conv} \left( [F_{fre}^{ir}, F_{fre}^{vi}] \right) \right), \quad (6)$$

where [·, ·] denotes channel-wise concatenation. The spatial branch employs cropping/downsampling augmentation and CNN to extract spatial features Fspa.

The fused visual embedding pvis is obtained through concatenation and linear projection of Ff re and Fspa. To ensure semantic consistency between visual/text embeddings, we apply MSE loss Lmse and cosine similarity loss Lcos:

$$\mathcal{L}_I = \lambda_1 \underbrace{\|p_{vis} - p_{text}\|^2}_{\mathcal{L}_{\text{mse}}} + \lambda_2 \underbrace{(1 - \frac{p_{vis} \cdot p_{text}}{\|p_{vis}\| \|p_{text}\|})}_{\mathcal{L}_{\text{cos}}}, \quad (7)$$

where λ<sup>1</sup> and λ<sup>2</sup> balance loss components. This design enables automatic degradation-aware adaptation while preserving text-aligned semantics.

# 4.3 Stage II: Prompt-modulated Restoration and Fusion

# 4.3.1 Network Architectures

Feature Encoding and Fusion Layer. As shown Fig. [2,](#page-3-0) we devise the hierarchical transformer-based encoders to extract multi-scale feature representations from degraded infrared (Dir) and visible (Dvi) images separately, which is formulated as:{Fir, Fvi} = Eir(Dir), Evi(Dvi), where Eir and Evi are infrared and visible image encoders.

Furthermore, to achieve comprehensive cross-modal feature integration, we design the intra-domain fusion unit, where the cross-attention (CR-ATT) mechanism is employed to facilitate interaction between features across modalities. Specifically, the linear transformation functions F qkv ir and F qkv vi project Fir and Fvi into their corresponding Q, K, and V , expressed as:

$$\{Q_{ir}, K_{ir}, V_{ir}\} = \mathcal{F}_{ir}^{qkv}(F_{ir}), \{Q_{vi}, K_{vi}, V_{vi}\} = \mathcal{F}_{vi}^{qkv}(F_{vi}). \quad (8)$$

Subsequently, we swap the queries Q of complementary modalities to promote spatial interaction:

$$F_f^{ir} = \text{softmax}\left(\frac{Q_{vi}K_{ir}}{\sqrt{d_k}}\right)V_{ir}, F_f^{vi} = \text{softmax}\left(\frac{Q_{ir}K_{vi}}{\sqrt{d_k}}\right)V_{vi}, \quad (9)$$

where d<sup>k</sup> is scaling factor. We concatenate F ir f and F vi f to get fusion features: F<sup>f</sup> = [F ir f , Fvi f ].

Prompt-modulated Module and Image Decoder. To dynamically adapt fusion feature distributions to degradation patterns and degrees, we propose a Prompt-Modulated Module (PMM) for robust feature enhancement. First, sequential MLPs (Φp) derive distributional optimization parameters:[γp, βp] = Φp(p), where p ∈ {pvis, ptext}. Then, γ<sup>p</sup> and β<sup>p</sup> are applied for feature scaling and bias shifting in a residual manner:Fˆ <sup>f</sup> = (1 + γp) ⊙ F<sup>f</sup> + βp, where ⊙ denotes the Hadamard product, and Fˆ <sup>f</sup> indicates enhanced features incorporating degradation prompts. We further deploy a series of Transformer-based decoder D<sup>f</sup> with the self-attention to progressively reconstruct the fused image I<sup>f</sup> = D<sup>f</sup> (Fˆ <sup>f</sup> ). In particular, PMM and D<sup>f</sup> are tightly coupled in a multi-stage process, effectively incorporating fusion features and degradation prompts to synthesize the desired image.

### 4.3.2 Loss Functions

Following the typical fusion paradigm [\[20\]](#page-11-3), we introduce the intensity loss, structural similarity (SSIM) loss, maximum gradient loss, and color consistency loss to constrain the training of Stage II.

The intensity loss Lint maximizes the target prominence of fusion results, defined as:

$$\mathcal{L}_{int} = \frac{1}{HW} \|I_f - \max(I_{ir}^{hq}, I_{vi}^{hq})\|_1, \quad (10)$$

where I hq ir and I hq vi are the high-quality source images.

The structural similarity loss Lssim ensures the fused image maintains structural consistency with the high-quality source images, preserving essential structural information, and is formulated as:

$$\mathcal{L}_{ssim} = 2 - (\text{SSIM}(I_f, I_{ir}^{hq}) + \text{SSIM}(I_f, I_{vi}^{hq})). \quad (11)$$

The maximum gradient loss Lgrad maximizes the retention of key edge information from both source images, generating fusion results with clearer textures, formulated as:

$$\mathcal{L}_{grad} = \frac{1}{HW} \|\nabla I_f - \max(\nabla I_{ir}^h, \nabla I_{vi}^h)\|_1, \quad (12)$$

where ∇ denotes the Sobel operator. Moreover, the color consistency loss Lcolor ensures that the fusion results maintain color consistency with the visible image. We convert the image to YCbCr space and minimize the distance between the Cb and Cr channels, expressed as:

$$\mathcal{L}_{color} = \frac{1}{HW} \|\mathcal{F}_{CbCr}(I_f) - \mathcal{F}_{CbCr}(I_{vi}^{hq})\|_1, \quad (13)$$

where FCbCr denotes the transfer function of RGB to CbCr.

Finally, the total loss LII for Stage II is the weighted sum of the aforementioned losses:

$$\mathcal{L}_{II} = \alpha_{int} \cdot \mathcal{L}_{int} + \alpha_{ssim} \cdot \mathcal{L}_{ssim} + \alpha_{grad} \cdot \mathcal{L}_{grad} + \alpha_{color} \cdot \mathcal{L}_{color}, \quad (14)$$

where αint, αssim, αgrad, and αcolor are hyper-parameters.

# 5 Experiments

#### 5.1 Implementation and Experimental Configurations

Our image restoration and fusion network is built on a four-stage encoder–decoder architecture, with channel dimensions increasing from 48 to 384, specifically configured as [48, 96, 192, 384]. The model is trained on the proposed DDL-12 dataset. During training, 224 × 224 patches are randomly cropped as inputs, with a batch size of 12 over 100 epochs. Optimization is performed using AdamW, starting with a learning rate of 1 × 10−<sup>3</sup> and decayed to 1 × 10−<sup>5</sup> via a cosine annealing schedule. For the loss configuration, λ<sup>1</sup> and λ<sup>2</sup> are set with a weight ratio of 1 : 3, and αint, αssim, αgrad, and αcolor are assigned values of 8 : 1 : 10 : 12, respectively.

We introduce text prompts to enable the unified network to effectively handle diverse and complex degradations. Each prompt specifies the affected *modality*, the *degradation type*, and its *severity*, enabling user-controllable flexibility. For example, a typical prompt for a single degradation is: *We are performing infrared and visible image fusion, where the modality suffers from a gradeseverity degradation type.* For composite degradations, we extend this template to specify multiple modality–degradation pairs, e.g., *We are performing infrared and visible image fusion. Please handle a grade-severity-A degradation type-A in the modality-A, and a grade-severity-B degradation type-B in the modality-B.* Details of the prompt construction paradigm are provided in the Appendix.

We compare our method with seven SOTA fusion methods, *i.e.*, DDFM [\[50\]](#page-12-6), DRMF [\[31\]](#page-11-7), EMMA [\[51\]](#page-12-4), LRRNet [\[11\]](#page-10-2), SegMiF [\[16\]](#page-10-4), Text-IF [\[39\]](#page-12-12), and Text-DiFuse [\[42\]](#page-12-8). Firstly, we evaluate the fusion performance on four widely used datasets, *i.e.*, MSRS [\[29\]](#page-11-1), LLVIP [\[7\]](#page-10-11), RoadScene [\[35\]](#page-11-16), and FMB [\[16\]](#page-10-4). The test set sizes for MSRS, LLVIP, RoadScene, and FMB are 361, 50, 50, and 50, respectively. Four metrics, *i.e.*, EN, SD, VIF, and Qabf , are used to quantify fusion performance. Additionally, we evaluate the restoration and fusion performance across various degradations, including low-contrast, random noise, and stripe noise in infrared images (IR), as well as blur, rain, over-exposure, and low light in visible images (VI). Additionally, we evaluate its effectiveness in handling composite degradation scenarios. Each degradation scenario includes 100 image pairs from DDL-12 dataset. We use CLIP-IQA [\[34\]](#page-11-17), MUSIQ [\[9\]](#page-10-12), TReS [\[3\]](#page-10-13), EN, and SD to quantify both restoration and fusion performance. All experiments are conducted on NVIDIA RTX 4090 GPUs with an Intel(R) Xeon(R) Platinum 8180 CPU (2.50 GHz) using the PyTorch framework.

Table 1: Quantitative comparison results on typical fusion datasets. The best and second-best results are highlighted in Red and Purple.

| Methods       | MSRS EN | SD     | VIF   | Qabf  | LLVIP EN | SD     | VIF   | Qabf  | EN    | RoadScene SD | VIF   | Qabf  | FMB EN | SD     | VIF   | Qabf  |
|---------------|---------|--------|-------|-------|----------|--------|-------|-------|-------|--------------|-------|-------|--------|--------|-------|-------|
| DDFM          | 6.431   | 47.815 | 0.844 | 0.643 | 6.914    | 48.556 | 0.693 | 0.517 | 6.994 | 47.094       | 0.775 | 0.595 | 6.426  | 40.597 | 0.495 | 0.442 |
| DRMF          | 6.268   | 45.117 | 0.669 | 0.550 | 6.901    | 50.736 | 0.786 | 0.626 | 6.231 | 44.221       | 0.728 | 0.527 | 6.842  | 41.816 | 0.578 | 0.372 |
| EMMA          | 6.747   | 52.753 | 0.886 | 0.605 | 6.366    | 47.065 | 0.743 | 0.547 | 6.959 | 46.749       | 0.698 | 0.664 | 6.788  | 38.174 | 0.542 | 0.436 |
| LRRNet        | 6.761   | 49.574 | 0.713 | 0.667 | 6.191    | 48.336 | 0.864 | 0.575 | 7.185 | 46.400       | 0.756 | 0.658 | 6.432  | 48.154 | 0.501 | 0.368 |
| SegMiF        | 7.006   | 57.073 | 0.764 | 0.586 | 7.260    | 45.892 | 0.539 | 0.459 | 6.736 | 48.975       | 0.629 | 0.584 | 6.363  | 47.398 | 0.539 | 0.482 |
| Text-IF       | 6.619   | 55.881 | 0.753 | 0.656 | 6.364    | 49.868 | 0.859 | 0.566 | 6.836 | 47.596       | 0.634 | 0.609 | 7.397  | 47.726 | 0.568 | 0.528 |
| Text-DiFuse   | 6.990   | 56.698 | 0.850 | 0.603 | 7.546    | 55.725 | 0.883 | 0.659 | 6.826 | 50.230       | 0.683 | 0.662 | 6.888  | 49.558 | 0.793 | 0.653 |
| ControlFusion | 7.340   | 60.360 | 0.927 | 0.718 | 7.354    | 56.631 | 0.968 | 0.738 | 7.421 | 51.759       | 0.817 | 0.711 | 7.036  | 50.905 | 0.872 | 0.730 |

Table 2: Quantitative comparison results under different degradation scenarios with pre-enhancement.

| Methods       | VI (Blur) CLIP-IQA | MUSIQ     | TReS   | SD     | VI (Rain) CLIP-IQA | MUSIQ          | TReS       | SD     | VI (Low CLIP-IQA | light, LL) MUSIQ | TReS   | SD     | VI CLIP-IQA | (Over-exposure, MUSIQ | OE) TReS | SD     |
|---------------|--------------------|-----------|--------|--------|--------------------|----------------|------------|--------|------------------|------------------|--------|--------|-------------|-----------------------|----------|--------|
| DDFM          | 0.141              | 39.421    | 39.047 | 35.411 | 0.191              | 38.836         | 46.285     | 36.376 | 0.156            | 39.495           | 41.782 | 31.759 | 0.143       | 43.167                | 43.440   | 32.099 |
| DRMF          | 0.128              | 40.739    | 40.968 | 40.722 | 0.174              | 48.164         | 48.565     | 41.174 | 0.143            | 41.428           | 37.947 | 38.287 | 0.190       | 48.334                | 42.582   | 44.256 |
| EMMA          | 0.131              | 43.472    | 41.744 | 42.553 | 0.138              | 45.824         | 44.916     | 43.378 | 0.158            | 39.674           | 44.827 | 40.857 | 0.180       | 46.731                | 47.616   | 40.242 |
| LRRNet        | 0.163              | 42.981    | 37.268 | 45.389 | 0.185              | 43.291         | 41.891     | 46.285 | 0.164            | 40.486           | 34.836 | 41.639 | 0.160       | 42.548                | 48.414   | 42.190 |
| SegMiF        | 0.152              | 43.005    | 43.516 | 44.000 | 0.195              | 40.528         | 49.094     | 44.274 | 0.177            | 44.073           | 48.376 | 44.829 | 0.166       | 49.132                | 38.019   | 38.484 |
| Text-IF       | 0.164              | 44.801    | 46.542 | 48.401 | 0.164              | 41.287         | 47.380     | 49.298 | 0.163            | 41.096           | 49.174 | 47.257 | 0.172       | 40.298                | 45.599   | 47.330 |
| Text-DiFuse   | 0.172              | 44.958    | 47.699 | 46.376 | 0.173              | 39.243         | 50.017     | 47.297 | 0.192            | 46.734           | 50.126 | 49.883 | 0.183       | 39.095                | 49.596   | 50.279 |
| ControlFusion | 0.184              | 47.849    | 50.240 | 50.287 | 0.196              | 52.319         | 52.465     | 50.901 | 0.183            | 48.420           | 51.072 | 53.787 | 0.191       | 50.301                | 52.961   | 54.218 |
|               | VI (Rain           | and Haze, | RH)    |        | IR                 | (Low-contrast, | LC)        |        | IR (Random       | noise,           | RN)    |        | IR (Stripe  | noise,                | SN)      |        |
| Methods       | CLIP-IQA           | MUSIQ     | TReS   | EN     | CLIP-IQA           | MUSIQ          | TReS       | SD     | CLIP-IQA         | MUSIQ            | TReS   | EN     | CLIP-IQA    | MUSIQ                 | TReS     | EN     |
| DDFM          | 0.158              | 43.119    | 44.095 | 6.897  | 0.172              | 44.490         | 44.545     | 37.452 | 0.186            | 34.059           | 32.807 | 6.029  | 0.209       | 48.479                | 32.377   | 6.399  |
| DRMF          | 0.171              | 45.524    | 43.693 | 6.230  | 0.208              | 43.982         | 45.561     | 40.315 | 0.192            | 44.617           | 44.085 | 5.744  | 0.187       | 44.714                | 43.650   | 6.354  |
| EMMA          | 0.169              | 39.092    | 46.046 | 6.517  | 0.154              | 48.724         | 43.113     | 53.861 | 0.172            | 39.666           | 44.466 | 6.184  | 0.153       | 42.802                | 44.239   | 6.382  |
| LRRNet        | 0.170              | 48.571    | 48.973 | 7.363  | 0.151              | 48.718         | 45.266     | 51.605 | 0.158            | 48.274           | 37.090 | 7.295  | 0.137       | 46.511                | 36.610   | 7.702  |
| SegMiF        | 0.147              | 46.139    | 45.019 | 7.281  | 0.160              | 44.659         | 51.427     | 44.448 | 0.180            | 42.664           | 39.496 | 6.669  | 0.169       | 49.887                | 39.247   | 6.855  |
| Text-IF       | 0.178              | 50.568    | 50.271 | 6.956  | 0.187              | 49.299         | 49.266     | 47.032 | 0.169            | 46.647           | 48.491 | 6.256  | 0.161       | 49.019                | 47.755   | 6.085  |
| Text-DiFuse   | 0.175              | 52.788    | 53.073 | 7.470  | 0.165              | 50.092         | 51.715     | 39.429 | 0.203            | 49.278           | 49.479 | 6.982  | 0.194       | 51.762                | 49.288   | 7.089  |
| ControlFusion | 0.189              | 54.287    | 54.465 | 7.891  | 0.196              | 51.986         | 52.846     | 57.827 | 0.189            | 50.711           | 51.668 | 7.724  | 0.200       | 50.097                | 50.264   | 7.619  |
|               | VI (OE)            | and IR    | (LC)   |        | VI(Low             | light and      | Noise, LN) |        | VI (RH)          | and IR           | (RN)   |        | VI (LL)     | and IR                | (SN)     |        |
| Methods       | CLIP-IQA           | MUSIQ     | TReS   | SD     | CLIP-IQA           | MUSIQ          | TReS       | EN     | CLIP-IQA         | MUSIQ            | TReS   | SD     | CLIP-IQA    | MUSIQ                 | TReS     | EN     |
| DDFM          | 0.168              | 43.814    | 41.894 | 36.095 | 0.172              | 48.293         | 31.791     | 6.298  | 0.151            | 33.440           | 32.134 | 37.342 | 0.189       | 36.433                | 42.630   | 5.776  |
| DRMF          | 0.184              | 42.399    | 39.374 | 40.847 | 0.201              | 44.363         | 43.063     | 5.875  | 0.174            | 43.663           | 43.858 | 37.997 | 0.142       | 38.241                | 41.049   | 5.280  |
| EMMA          | 0.130              | 39.892    | 42.076 | 43.362 | 0.174              | 42.201         | 43.382     | 5.838  | 0.165            | 39.146           | 44.458 | 51.205 | 0.130       | 37.367                | 43.888   | 6.318  |
| LRRNet        | 0.136              | 47.209    | 42.636 | 46.684 | 0.144              | 46.386         | 35.779     | 7.306  | 0.128            | 47.954           | 36.831 | 49.917 | 0.154       | 38.426                | 35.970   | 7.007  |
| SegMiF        | 0.114              | 44.021    | 42.256 | 33.647 | 0.136              | 49.178         | 38.570     | 5.819  | 0.147            | 42.354           | 39.156 | 31.717 | 0.151       | 41.287                | 37.079   | 6.767  |
| Text-IF       | 0.174              | 48.808    | 47.998 | 48.848 | 0.217              | 48.100         | 47.510     | 5.204  | 0.158            | 45.821           | 47.626 | 46.543 | 0.140       | 41.429                | 46.220   | 5.525  |
| Text-DiFuse   | 0.131              | 49.021    | 50.980 | 47.640 | 0.185              | 50.775         | 48.610     | 6.440  | 0.181            | 48.645           | 48.937 | 38.808 | 0.161       | 47.734                | 48.448   | 6.738  |
| ControlFusion | 0.187              | 50.479    | 50.298 | 50.955 | 0.225              | 49.333         | 49.513     | 7.111  | 0.179            | 50.107           | 51.091 | 55.417 | 0.167       | 50.632                | 48.971   | 7.055  |

# 5.2 Fusion Performance Comparison

Comparison without Pre-enhancement. Table [1](#page-6-0) summarizes the quantitative assessment across benchmark datasets. Notably, our method attains SOTA performance in SD, VIF, and Qabf metrics, showcasing its exceptional ability to maintain structural integrity and edge details. The EN metric remains competitive, indicating that our fusion results encapsulate abundant information.

Comparison with Pre-enhancement. For a fair comparison, we exclusively use the visual prompts to characterize degradation in the quantitative comparison. Meanwhile, we employ several SOTA image restoration algorithms as preprocessing steps to address specific degradations. Specifically, OneRestore [\[6\]](#page-10-14) is used for weather-related degradations, Instruct-IR [\[2\]](#page-10-8) for illumination-related degradations, and DA-CLIP [\[18\]](#page-11-13) and WDNN [\[4\]](#page-10-15) for sensor-related degradations. For composite degradations, we select the best-performing method from these algorithms.

The quantitative results in Tab. [2](#page-6-1) show that ControlFusion achieves superior CLIP-IQA, MUSIQ, and TReS scores across most degradation scenarios, demonstrating a strong capability in degradation mitigation and complementary context aggregation. Additionally, no-reference fusion metrics (SD, EN) show that ControlFusion performs on par with SOTA image fusion methods. Qualitative comparisons in Fig. [4](#page-7-0) indicate that ControlFusion not only excels in addressing single-modal degradation but also effectively tackles more challenging single- and multi-modal composite degradations. In particular, when rain and haze coexist in the visible image, ControlFusion successfully perceives these degradations and eliminates their effects. In multi-modal composite degradation scenarios, it leverages degradation prompts to adjust the distribution of fusion features, achieving high-quality restoration and fusion. Extensive experiments demonstrate that ControlFusion exhibits significant advantages in complex degradation scenarios.

![](_page_7_Figure_0.jpeg)

![](_page_7_Picture_1.jpeg)

Figure 4: Visualization of fusion results under different degradation scenarios.

![](_page_7_Picture_3.jpeg)

![](_page_7_Figure_4.jpeg)

Figure 5: Generalization results under real-world degradation scenarios.

# 5.3 Extended Experiments

Table 3: Quantitative comparison on MSRS nighttime scenes

and AWMM-100k with real-world weather degradations. Methods CLIP-IQA MUSIQ TReS SD CLIP-IQA MUSIQ TReS SD DDFM 0.104 25.034 19.404 33.935 0.227 41.919 53.858 36.010 DRMF 0.102 27.518 16.052 46.718 0.232 50.054 56.671 39.735 EMMA 0.087 26.713 18.511 42.659 0.253 47.161 49.722 40.398 LRRNet 0.098 27.042 16.909 42.692 0.192 46.751 50.707 37.190 SegMiF 0.093 27.405 21.079 44.315 0.257 50.712 54.310 33.311 Text-IF 0.101 28.155 20.640 49.752 0.213 45.633 51.747 32.086 Text-DiFuse 0.116 28.620 18.967 43.728 0.233 46.901 57.708 41.530 ControlFusion 0.141 28.457 21.607 52.933 0.280 57.513 60.302 44.244 Real-world Generalization. As shown in Tab. [3,](#page-7-1) we report comparative results on nighttime scenes from MSRS and the real weather-degraded benchmark AWMM-100k [\[13\]](#page-10-16). Our method achieves the best performance on CLIP-IQA, TRes, and SD, while slightly trailing Text-DiFuse on MUSIQ . On AWMM-100k, our approach consistently outperforms competing methods across all evaluation metrics under real-world weather degradation conditions.

| Methods       | CLIP-IQA | MSRS MUSIQ | (Nighttime) TReS | SD     | AWMM-100k CLIP-IQA | MUSIQ  | (Weather TReS | Degradations) SD |
|---------------|----------|------------|------------------|--------|--------------------|--------|---------------|------------------|
| DDFM          | 0.104    | 25.034     | 19.404           | 33.935 | 0.227              | 41.919 | 53.858        | 36.010           |
| DRMF          | 0.102    | 27.518     | 16.052           | 46.718 | 0.232              | 50.054 | 56.671        | 39.735           |
| EMMA          | 0.087    | 26.713     | 18.511           | 42.659 | 0.253              | 47.161 | 49.722        | 40.398           |
| LRRNet        | 0.098    | 27.042     | 16.909           | 42.692 | 0.192              | 46.751 | 50.707        | 37.190           |
| SegMiF        | 0.093    | 27.405     | 21.079           | 44.315 | 0.257              | 50.712 | 54.310        | 33.311           |
| Text-IF       | 0.101    | 28.155     | 20.640           | 49.752 | 0.213              | 45.633 | 51.747        | 32.086           |
| Text-DiFuse   | 0.116    | 28.620     | 18.967           | 43.728 | 0.233              | 46.901 | 57.708        | 41.530           |
| ControlFusion | 0.141    | 28.457     | 21.607           | 52.933 | 0.280              | 57.513 | 60.302        | 44.244           |

In addition to quantitative comparisons, Fig. [5](#page-7-2) presents visual examples demonstrating our method's strong generalization ability in removing composite degradations in real scenes. The top image shows typical data collected by our multimodal sensors under challenging low-light and noisy conditions,

![](_page_8_Figure_0.jpeg)

![](_page_8_Picture_1.jpeg)

Figure 6: Fusion results with various level prompts.

Table 4: Computational efficiency of image restoration and fusion algorithms on 640×480 resolution.

| Methods    | Image Parm.(M) | Restoration Flops(G) | Time(s) | Methods  | Parm.(M) | Flops(G) | Image Time(s) | Fusion Methods | Parm.(M) | Flops(G) | Time(s) |
|------------|----------------|----------------------|---------|----------|----------|----------|---------------|----------------|----------|----------|---------|
| InstructIR | 15.94          | 151.54               | 0.038   | DDFM ∗   | 552.70   | 5220.50  | 34.511        | SegMiF ∗       | 45.64    | 707.39   | 0.371   |
| DA-CLIP    | 295.19         | 1214.55              | 13.578  | DRMF     | 5.05     | 121.10   | 0.842         | Text-IF        | 152.44   | 1518.90  | 0.157   |
| WDNN       | 0.01           | 1.11                 | 0.004   | EMMA ∗   | 1.52     | 41.54    | 0.048         | Text-DiFuse    | 157.35   | 4872.00  | 12.903  |
| OneRestore | 18.12          | 114.56               | 0.024   | LRRNet ∗ | 0.05     | 14.17    | 0.096         | ControlFusion  | 182.54   | 1622.00  | 0.183   |

while the bottom image illustrates the robustness of our method in handling extremely low-light scenarios. Additional qualitative results are provided in the Appendix.

Flexible Degree Control. We employ numerical values to quantify the severity of degradation in design, where higher numbers indicate more severe degradation, enabling precise continuous control. It should be noted that while only four anchor points (1, 4, 7, and 10) were used during training, the model demonstrates remarkable generalization capability to intermediate degrees during testing, owing to the inherent encoding characteristics of CLIP. As shown in Fig. [6,](#page-8-0) with degradation level prompts, our ControlFusion can adapt to varying degrees of degradation and deliver satisfactory fusion results. Moreover, fusion results generated using adjacent degradation-level prompts exhibit subtle yet perceptible differences, allowing users to obtain customized outcomes through their specific prompts. More visualization results are presented in the Appendix.

Computational Efficiency. Table [4](#page-8-1) details the computational costs for various methods. While standalone fusion models such as EMMA, LRRNet, and SegMiF, which are marked with <sup>∗</sup> , appear more lightweight, they necessitate a separate, computationally intensive restoration stage to form a practical pipeline. When the overhead of this two-stage process is factored in, the efficiency advantages of our unified model becomes evident. Moreover, our model's efficiency remains highly competitive with other joint restoration-fusion approaches like DRMF, Text-DiFuse, and Text-IF.

| Table Methods | 5: Prec. | Quantitative Recall | comparison AP@0.50 | of object AP@0.75 | detection. mAP@0.5:0.95 |
|---------------|----------|---------------------|--------------------|-------------------|-------------------------|
| DDFM          | 0.947    | 0.848               | 0.911              | 0.655             | 0.592                   |
| DRMF          | 0.958    | 0.851               | 0.937              | 0.672             | 0.607                   |
| EMMA          | 0.942    | 0.872               | 0.927              | 0.647             | 0.598                   |
| LRRNet        | 0.939    | 0.878               | 0.933              | 0.672             | 0.608                   |
| SegMiF        | 0.965    | 0.896               | 0.931              | 0.690             | 0.603                   |
| Text-IF       | 0.959    | 0.892               | 0.939              | 0.655             | 0.601                   |
| Text-DiFuse   | 0.961    | 0.885               | 0.941              | 0.656             | 0.606                   |
| ControlFusion | 0.971    | 0.889               | 0.949              | 0.685             | 0.609                   |

Object Detection. We evaluate object detection performance on LLVIP to assess the fusion quality, using the retrained YOLOv8 [\[26\]](#page-11-18). Quantitative results are presented in Tab. [5.](#page-8-2) Our results enable the detector to identify all pedestrians with higher confidence, achieving the highest mAP@0.5-0.95. Visualizations are in the Appendix.

![](_page_9_Picture_0.jpeg)

![](_page_9_Figure_1.jpeg)

Figure 7: Visual results of ablation studies under degradation scenarios.

Table 6: Quantitative results of the ablation studies.

| Configs | VI(LL CLIP-IQA | & Noise) MUSIQ | TReS   | EN    | VI(OE) CLIP-IQA | and IR(LC) MUSIQ | TReS   | SD     | VI (RH) CLIP-IQA | and MUSIQ | IR(Noise) TReS | SD     |
|---------|----------------|----------------|--------|-------|-----------------|------------------|--------|--------|------------------|-----------|----------------|--------|
| I       | 0.132          | 42.424         | 42.839 | 4.788 | 0.166           | 41.983           | 42.841 | 39.917 | 0.147            | 43.619    | 47.932         | 48.935 |
| II      | 0.152          | 45.582         | 45.358 | 5.855 | 0.151           | 43.646           | 42.002 | 39.208 | 0.167            | 47.862    | 45.007         | 44.347 |
| III     | 0.154          | 46.571         | 44.495 | 5.013 | 0.155           | 44.286           | 44.561 | 42.068 | 0.156            | 43.007    | 43.816         | 41.544 |
| IV      | 0.129          | 38.960         | 41.310 | 5.414 | 0.172           | 41.743           | 39.125 | 38.748 | 0.118            | 48.882    | 46.245         | 45.910 |
| V       | 0.173          | 45.281         | 46.291 | 6.279 | 0.181           | 45.386           | 47.519 | 46.860 | 0.149            | 46.714    | 48.094         | 46.950 |
| Ours    | 0.225          | 49.333         | 49.513 | 7.111 | 0.187           | 50.479           | 50.298 | 50.955 | 0.179            | 50.107    | 51.091         | 55.417 |

Ablation Studies. To verify the effectiveness of components, we conduct detailed ablation studies, involves: (I) w/o Lcolor, (II) w/o Lgrad, (III) w/o Lint, (IV) w/o PMM, and (V) w/o frequency branch. As shown in Fig. [7,](#page-9-0) removing any loss or module significantly impacts the fusion quality. Specifically, without Lcolor, color distortion occurs, while removing Lgrad leads to the loss of essential texture information. Excluding Lint fails to highlight thermal targets, and removing PMM diminishes the ability to address composite degradation. Additionally, removing the frequency branch from SFVA causes visual prompt confusion. The t-SNE results further validate the importance of frequency priors. The quantitative results in Tab. [6](#page-9-1) also confirm that each design element is crucial for enhancing fusion performance.

Discussions and Limitations. To mitigate the gap between simulated and real-world data, our method constructs training samples using the degradation imaging model. An alternative strategy is test-time adaptation, in which part of the model is fine-tuned on the test set to better accommodate new data distributions. It should be noted that the degradation imaging model is specifically designed for infrared and visible image fusion and is difficult to generalize to other fusion tasks, such as medical image fusion and multi-focus image fusion.

# 6 Conclusion

This work proposes a controllable framework for image restoration and fusion leveraging languagevisual prompts. Initially, we develop a physics-driven degraded imaging model to bridge the domain gap between synthetic data and real-world images, providing a solid foundation for addressing composite degradations. Moreover, we devise a prompt-modulated network that adaptively adjusts the fusion feature distribution, enabling robust feature enhancement based on degradation prompts. Prompts can either come from text instructions to support user-defined control or be extracted from source images using a spatial-frequency visual adapter embedded with frequency priors, facilitating automated deployment. Extensive experiments demonstrate that our method excels in handling real-world and composite degradations, showing strong robustness across various degradation levels.

# 7 Acknowledgement

# References


[1] Fanglin Bao, Xueji Wang, Shree Hari Sureshbabu, Gautam Sreekumar, Liping Yang, Vaneet Aggarwal, Vishnu N Boddeti, and Zubin Jacob. Heat-assisted detection and ranging. *Nature*, 619(7971):743–748, 2023. [2] Marcos V Conde, Gregor Geigle, and Radu Timofte. Instructir: High-quality image restoration following human instructions. In *Proceedings of the European Conference on Computer Vision*, pages 1–21, 2024. [3] S Alireza Golestaneh, Saba Dadsetan, and Kris M Kitani. No-reference image quality assessment via transformers, relative ranking, and self-consistency. In *Proceedings of the IEEE Winter Conference on Applications of Computer Vision*, pages 1220–1230, 2022. [4] Juntao Guan, Rui Lai, and Ai Xiong. Wavelet deep neural network for stripe noise removal. *IEEE Access*, 7:44544–44554, 2019. [5] Xiaojie Guo, Yu Li, and Haibin Ling. Lime: Low-light image enhancement via illumination map estimation. *IEEE Transactions on Image Processing*, 26(2):982–993, 2016. [6] Yu Guo, Yuan Gao, Yuxu Lu, Huilin Zhu, Ryan Wen Liu, and Shengfeng He. Onerestore: A universal restoration framework for composite degradation. In *Proceedings of the European Conference on Computer Vision*, pages 255–272, 2024. [7] Xinyu Jia, Chuang Zhu, Minzhen Li, Wenqi Tang, and Wenli Zhou. Llvip: A visible-infrared paired dataset for low-light vision. In *Proceedings of the IEEE International Conference on Computer Vision*, pages 3496–3504, 2021. [8] Yitong Jiang, Zhaoyang Zhang, Tianfan Xue, and Jinwei Gu. Autodir: Automatic all-in-one image restoration with latent diffusion. In *Proceedings of the European Conference on Computer Vision*, pages 340–359, 2024. [9] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In *Proceedings of the IEEE International Conference on Computer Vision*, pages 5148–5157, 2021. [10] Hui Li and Xiao-Jun Wu. Densefuse: A fusion approach to infrared and visible images. *IEEE Transactions on Image Processing*, 28(5):2614–2623, 2019. [11] Hui Li, Tianyang Xu, Xiao-Jun Wu, Jiwen Lu, and Josef Kittler. Lrrnet: A novel representation learning guided fusion network for infrared and visible images. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 45(9):11040–11052, 2023. [12] Ruoteng Li, Loong-Fah Cheong, and Robby T Tan. Heavy rain image restoration: Integrating physics model and conditional adversarial learning. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 1633–1642, 2019. [13] Xilai Li, Wuyang Liu, Xiaosong Li, Fuqiang Zhou, Huafeng Li, and Feiping Nie. All-weather multimodality image fusion: Unified framework and 100k benchmark. *arXiv preprint arXiv:2402.02090*, 2024. [14] Pengwei Liang, Junjun Jiang, Xianming Liu, and Jiayi Ma. Fusion from decomposition: A self-supervised decomposition approach for image fusion. In *Proceedings of the European Conference on Computer Vision*, pages 719–735, 2022. [15] Jinyuan Liu, Xin Fan, Zhanbo Huang, Guanyao Wu, Risheng Liu, Wei Zhong, and Zhongxuan Luo. Target-aware dual adversarial learning and a multi-scenario multi-modality benchmark to fuse infrared and visible for object detection. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 5802–5811, 2022. [16] Jinyuan Liu, Zhu Liu, Guanyao Wu, Long Ma, Risheng Liu, Wei Zhong, Zhongxuan Luo, and Xin Fan. Multi-interactive feature learning and a full-time multi-modality benchmark for image fusion and segmentation. In *Proceedings of the IEEE International Conference on Computer Vision*, pages 8115–8124, 2023. [17] Zhu Liu, Jinyuan Liu, Benzhuang Zhang, Long Ma, Xin Fan, and Risheng Liu. Paif: Perception-aware infrared-visible image fusion for attack-tolerant semantic segmentation. In *Proceedings of the ACM International Conference on Multimedia*, pages 3706–3714, 2023.

[18] Ziwei Luo, Fredrik K. Gustafsson, Zheng Zhao, Jens Sjölund, and Thomas B. Schön. Controlling visionlanguage models for multi-task image restoration. In *International Conference on Learning Representations*, 2024. [19] Jiayi Ma, Wei Yu, Pengwei Liang, Chang Li, and Junjun Jiang. Fusiongan: A generative adversarial network for infrared and visible image fusion. *Information Fusion*, 48:11–26, 2019. [20] Jiayi Ma, Linfeng Tang, Fan Fan, Jun Huang, Xiaoguang Mei, and Yong Ma. Swinfusion: Cross-domain long-range learning for general image fusion via swin transformer. *IEEE/CAA Journal of Automatica Sinica*, 9(7):1200–1217, 2022. [21] Earl J McCartney. Optics of the atmosphere: scattering by molecules and particles. *New York*, 1976. [22] Amanda C Muller and Sundaram Narayanan. Cognitively-engineered multisensor image fusion for military applications. *Information Fusion*, 10(2):137–149, 2009. [23] Vaishnav Potlapalli, Syed Waqas Zamir, Salman H Khan, and Fahad Shahbaz Khan. Promptir: Prompting for all-in-one image restoration. *Advances in Neural Information Processing Systems*, 36:71275–71293, 2023. [24] Chenyang Qi, Zhengzhong Tu, Keren Ye, Mauricio Delbracio, Peyman Milanfar, Qifeng Chen, and Hossein Talebi. Spire: Semantic prompt-driven image restoration. In *Proceedings of the European Conference on Computer Vision*, pages 446–464, 2024. [25] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *Proceedings of the International Conference on Machine Learning*, pages 8748–8763, 2021. [26] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 779–788, 2016. [27] Linfeng Tang, Yuxin Deng, Yong Ma, Jun Huang, and Jiayi Ma. Superfusion: A versatile image registration and fusion network with semantic awareness. *IEEE/CAA Journal of Automatica Sinica*, 9(12):2121–2137, 2022. [28] Linfeng Tang, Jiteng Yuan, and Jiayi Ma. Image fusion in the loop of high-level vision tasks: A semanticaware real-time infrared and visible image fusion network. *Information Fusion*, 82:28–42, 2022. [29] Linfeng Tang, Jiteng Yuan, Hao Zhang, Xingyu Jiang, and Jiayi Ma. Piafusion: A progressive infrared and visible image fusion network based on illumination aware. *Information Fusion*, 83:79–92, 2022. [30] Linfeng Tang, Xinyu Xiang, Hao Zhang, Meiqi Gong, and Jiayi Ma. Divfusion: Darkness-free infrared and visible image fusion. *Information Fusion*, 91:477–493, 2023. [31] Linfeng Tang, Yuxin Deng, Xunpeng Yi, Qinglong Yan, Yixuan Yuan, and Jiayi Ma. Drmf: Degradationrobust multi-modal image fusion via composable diffusion prior. In *Proceedings of the ACM International Conference on Multimedia*, pages 8546–8555, 2024. [32] Linfeng Tang, Chunyu Li, and Jiayi Ma. Mask-difuser: A masked diffusion model for unified unsupervised image fusion. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2025. [33] Di Wang, Jinyuan Liu, Xin Fan, and Risheng Liu. Unsupervised misaligned infrared and visible image fusion via cross-modality image generation and registration. In *Proceedings of the International Joint Conference on Artificial Intelligence*, pages 3508–3515, 7 2022. [34] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring clip for assessing the look and feel of images. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 37, pages 2555–2563, 2023. [35] Han Xu, Jiayi Ma, Zhuliang Le, Junjun Jiang, and Xiaojie Guo. Fusiondn: A unified densely connected network for image fusion. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 34, pages 12484–12491, 2020. [36] Han Xu, Jiayi Ma, Junjun Jiang, Xiaojie Guo, and Haibin Ling. U2fusion: A unified unsupervised image fusion network. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(1):502–518, 2022.

[37] Han Xu, Jiteng Yuan, and Jiayi Ma. Murf: Mutually reinforcing multi-modal image registration and fusion. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 45(10):12148–12166, 2023. [38] Han Xu, Xunpeng Yi, Chen Lu, Guangcan Liu, and Jiayi Ma. Urfusion: Unsupervised unified degradationrobust image fusion network. *IEEE Transactions on Image Processing*, 34:5803–5818, 2025. [39] Xunpeng Yi, Han Xu, Hao Zhang, Linfeng Tang, and Jiayi Ma. Text-if: Leveraging semantic text guidance for degradation-aware and interactive image fusion. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 27026–27035, 2024. [40] Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Restormer: Efficient transformer for high-resolution image restoration. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 5728–5739, 2022. [41] Hao Zhang, Han Xu, Xin Tian, Junjun Jiang, and Jiayi Ma. Image fusion meets deep learning: A survey and perspective. *Information Fusion*, 76:323–336, 2021. [42] Hao Zhang, Lei Cao, and Jiayi Ma. Text-difuse: An interactive multi-modal image fusion framework based on text-modulated diffusion model. *Advances in Neural Information Processing Systems*, 37:39552–39572, 2024. [43] Hao Zhang, Lei Cao, Xuhui Zuo, Zhenfeng Shao, and Jiayi Ma. Omnifuse: Composite degradation-robust image fusion with language-driven semantics. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 47(9):7577–7595, 2025. [44] Jiaming Zhang, Huayao Liu, Kailun Yang, Xinxin Hu, Ruiping Liu, and Rainer Stiefelhagen. Cmx: Cross-modal fusion for rgb-x semantic segmentation with transformers. *IEEE Transactions on Intelligent Transportation Systems*, 24(12):14679–14694, 2023. [45] Jing Zhang, Aiping Liu, Dan Wang, Yu Liu, Z Jane Wang, and Xun Chen. Transformer-based end-to-end anatomical and functional image fusion. *IEEE Transactions on Instrumentation and Measurement*, 71: 5019711, 2022. [46] Xingchen Zhang and Yiannis Demiris. Visible and infrared image fusion using deep learning. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 45(8):10535–10554, 2023. [47] Yue Zhang, Bin Song, Xiaojiang Du, and Mohsen Guizani. Vehicle tracking using surveillance with multimodal data fusion. *IEEE Transactions on Intelligent Transportation Systems*, 19(7):2353–2361, 2018. [48] Wenda Zhao, Shigeng Xie, Fan Zhao, You He, and Huchuan Lu. Metafusion: Infrared and visible image fusion via meta-feature embedding from object detection. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 13955–13965, 2023. [49] Zixiang Zhao, Haowen Bai, Jiangshe Zhang, Yulun Zhang, Shuang Xu, Zudi Lin, Radu Timofte, and Luc Van Gool. Cddfuse: Correlation-driven dual-branch feature decomposition for multi-modality image fusion. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 5906–5916, 2023. [50] Zixiang Zhao, Haowen Bai, Yuanzhi Zhu, Jiangshe Zhang, Shuang Xu, Yulun Zhang, Kai Zhang, Deyu Meng, Radu Timofte, and Luc Van Gool. Ddfm: Denoising diffusion model for multi-modality image fusion. In *Proceedings of the IEEE International Conference on Computer Vision*, pages 8082–8093, 2023. [51] Zixiang Zhao, Haowen Bai, Jiangshe Zhang, Yulun Zhang, Kai Zhang, Shuang Xu, Dongdong Chen, Radu Timofte, and Luc Van Gool. Equivariant multi-modality image fusion. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pages 25912–25921, 2024.
# 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction of the paper clearly state the claims made, including the contributions made in the paper and important assumptions and limitations.

Guidelines:

- The answer NA means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

# 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The paper discusses the limitations of the work performed by the authors, providing a balanced view of the study's scope and potential areas for improvement.

Guidelines:

- The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate "Limitations" section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

# 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Justification: All the theorems, formulas, and proofs in the paper have been numbered and cross-referenced, and all assumptions have been clearly stated or referenced in the statement of any theorems.

### Guidelines:

- The answer NA means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

# 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The paper has fully disclosed all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and conclusions of the paper.

- The answer NA means that the paper does not include experiments.
- If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
  - (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
  - (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
  - (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
  - (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

# 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: We will open access to the code after the paper is accepted.

Guidelines:

- The answer NA means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ([https://nips.cc/](https://nips.cc/public/guides/CodeSubmissionPolicy) [public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ([https:](https://nips.cc/public/guides/CodeSubmissionPolicy) [//nips.cc/public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

# 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

Justification: We have specified all the training and test details necessary to understand the results.

Guidelines:

- The answer NA means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

# 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [No]

Justification: This paper does not include error bars or formal statistical significance tests.

- The answer NA means that the paper does not include experiments.
- The authors should answer "Yes" if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).
- It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates).
- If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

# 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: The paper provides sufficient information on the computer resource for experiments.

# Guidelines:

- The answer NA means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

# 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?

Answer: [Yes]

Justification: The research conducted in the paper conforms in every respect with the NeurIPS Code of Ethics, ensuring ethical standards are upheld throughout the study.

# Guidelines:

- The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

# 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: The paper discusses both potential positive societal impacts and negative societal impacts of the work performed, providing a comprehensive evaluation of the study's broader implications.

- The answer NA means that there is no societal impact of the work performed.

- If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

# 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [No]

Justification: We do not foresee any high risk for misuse of this work.

Guidelines:

- The answer NA means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

### 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The creators or original owners of assets used in the paper are properly credited, and the license and terms of use are explicitly mentioned and properly respected.

- The answer NA means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.
- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, <paperswithcode.com/datasets> has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

### 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer:[NA]

Justification: The paper does not release new assets.

Guidelines:

- The answer NA means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

### 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

#### 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve crowdsourcing nor research with human subjects. Guidelines:

- The answer NA means that the paper does not involve crowdsourcing nor research with human subjects.
- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

#### 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: The core method development in the paper does not involve LLMs as any important, original, or non-standard components.

- The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy (<https://neurips.cc/Conferences/2025/LLM>) for what should or should not be described.

# A Appendix

# A.1 Prompt Design Details

We provide additional details of the prompt construction paradigm to complement the main text.

Prompt components. Each prompt specifies three key components: (i) the affected *modality* (infrared or visible), (ii) the *degradation type* (*e.g.*, rain, haze, low-light, overexposure, random noise, stripe noise, low-contrast, etc.), and (iii) the *severity* level (an integer from 1 to 10).

Severity levels. To characterize degradation severity, we define a scale from 1 to 10, with representative anchor points as follows:

- Level 1: barely perceptible degradation,
- Level 4: degradation begins to interfere with human scene understanding,
- Level 7: degradation severely hinders human perception of the scene,
- Level 10: most useful information is completely obscured.

Intermediate levels can be interpolated between these anchor points.

Prompt templates. A typical template for a single degradation is structured as: We are performing infrared and visible image fusion, where the *modality* suffers from a grade-*severity degradation type*.

For composite degradations, two extensions are designed: 1) We are performing infrared and visible image fusion. Please handle a grade-*severity-A degradation type-A* in the *modality-A*, and a grade*severity-B degradation type-B* in the *modality-B*; 2) We are performing infrared and visible image fusion. Please address level-*severity degradation type-A* and *degradation type-B* in the *modality*.

These templates flexibly represent both intra- and inter-modal combinations. For instance, an instantiation could be: We are performing infrared and visible image fusion. Please handle a grade-6 low-light in the visible modality, and a grade-8 stripe noise in the infrared modality.

Linguistic diversity. To enhance robustness, we curated over 100 unique textual formulations for each task and its associated parameters. This prompt rephrasing strategy augments linguistic variability and improves the model's generalization to diverse prompt styles during inference.

Additionally, when users are uncertain about severity, the proposed SFVA module can automatically perceive both degradation type and severity from the input. This design balances user-controllable flexibility with automated practicality.

# A.2 Visual Embedding versus Textual Embedding

From Fig. [8,](#page-20-0) we can find that both visual and textual prompts assist Nrf in restoring the scene illumination. Besides, the residual plots between the two are not obvious, which indicates that our SFVA successfully captures the type and extent of degradation.This capability is crucial for automatic deployment, as it allows the model to adaptively adjust its restoration strategy with fewer manual lintervention or hyperparameters tuning for different degradation scenarios.

![](_page_20_Picture_15.jpeg)

**Textual prompt: This is infrared and visible image fusion task with the visible image suffers from grad #4 low-light issue**

Figure 8: Comparison of visual and textual embedding.

# A.3 Flexible Degree Control

In order to verify the effectiveness of the proposed flexible degree control, we perform different degrees of prompts on a pair of low-light images. The results are shown in Fig. [9](#page-21-0) (b). We splice all the results to obtain images with continuously changing brightness in different regions, and count the

![](_page_21_Figure_0.jpeg)

Figure 9: Fusion results with various level prompts.

brightness of the same region of these images. The results are shown in Fig. [9](#page-21-0) (c). This shows that our degree control achieves fine-grained regulation.

### A.4 Real-world Generalization

We conduct comprehensive experiments on the FMB dataset under various composite degradation scenarios. Notably, these test data are collected from real-world environments and represent degradation patterns not encountered in the training set. During evaluation, we rely exclusively on visual prompts automatically extracted from the input images, demonstrating our method's ability to handle unseen degradation conditions without manual intervention, as shown in Fig. [10.](#page-22-0)

### A.5 Effect of Severity Level Granularity

To ensure our model learns to discern fine-grained degradation severity, we train it on a dense set of four anchor degradation levels (1&4&7&10) from the DDL-12 dataset. This design is supported by a comparative analysis against models trained on sparser subsets of only two levels, specifically the moderate pair (4&7) or the extreme pair (1&10). As shown in Tab. [7,](#page-21-1) these sparser training regimes lead to a notable drop in generalization performance on real-world scenarios, particularly when only the extreme levels (1&10) are used. These results indicate the importance of dense sampling of training levels for robust generalization.

Table 7: Comparison of models trained with varying degradation levels on real-world datasets.

| Level        | numbers | CLIP-IQA | MUSIQ  | LLVIP TReS | SD     | CLIP-IQA | MUSIQ  | FMB TReS | SD     |
|--------------|---------|----------|--------|------------|--------|----------|--------|----------|--------|
| 2            | (1&10)  | 0.320    | 53.498 | 61.889     | 54.665 | 0.192    | 52.839 | 63.115   | 50.122 |
| 2            | (4&7)   | 0.332    | 55.364 | 64.382     | 55.742 | 0.207    | 52.771 | 63.328   | 50.429 |
| 4 (1&4&7&10) |         | 0.347    | 55.890 | 65.014     | 56.657 | 0.208    | 53.034 | 63.704   | 51.707 |

### A.6 Object Detection

As illustrated in Fig. [11,](#page-22-1) our fusion method achieves superior object detection performance with consistently high confidence scores. Notably, the detector successfully identifies all pedestrians in challenging scenarios, including heavily occluded cases where other methods fail. These results

![](_page_22_Picture_0.jpeg)

![](_page_22_Figure_1.jpeg)

Figure 10: Real-world composite degradation scenario testing.

![](_page_22_Picture_3.jpeg)

Figure 11: Visualization results of target detection.

quantitatively validate the effectiveness of our approach in preserving and enhancing critical visual information through the fusion process, particularly for small and obscured targets. The improved detection performance further confirms that our method maintains both thermal signatures and texture details essential for downstream vision tasks.

# A.7 Model Complexity Analysis

As demonstrated in Fig. [7](#page-9-0) and Tab. [6,](#page-9-1) both CLIP and SFVA (*i.e.*, visual–text prompt alignment) yield substantial improvements in restoration and fusion performance. We analyze the computational cost of these components in Tab. [8.](#page-23-0) Although CLIP accounts for a significant number of parameters (102M, 55.9% of the total), they are frozen during training and thus introduce no training overhead. In terms of inference cost, CLIP and SFVA are responsible for a modest 0.81% and 7.5% of the total

Table 8: Overhead of individual components. Percentages indicate the relative share in the full model.

| Components     | Parm. (M)      | FLOPs (G)      | Time (s)       |
|----------------|----------------|----------------|----------------|
| CLIP-L14@30xpc | 100.0 (55.88%) | 10.088 (0.81%) | 0.0102 (6.55%) |
| Stark          | 18.2 (8.37%)   | 12.007 (0.74%) | 0.0067 (3.66%) |
| Full Model     | 182.54         | 1622.00        | 0.1833         |

FLOPs, respectively, and their combined runtime constitutes a small fraction of the total. Overall, both components offer a compelling trade-off, delivering notable performance gains for a moderate computational expense.