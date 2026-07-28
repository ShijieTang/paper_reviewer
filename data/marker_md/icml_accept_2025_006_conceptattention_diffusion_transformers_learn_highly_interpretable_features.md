## ConceptAttention: Diffusion Transformers Learn Highly Interpretable Features

Alec Helbling <sup>1</sup> Tuna Han Salih Meral <sup>2</sup> Benjamin Hoover 1 3 Pinar Yanardag <sup>2</sup> Duen Horng (Polo) Chau <sup>1</sup>

![](_page_0_Figure_2.jpeg)

![](_page_0_Diagram_3.jpeg)

## Abstract

Do the rich representations of multi-modal diffusion transformers (DiTs) exhibit unique properties that enhance their interpretability? We introduce CONCEPTATTENTION, a novel method that leverages the expressive power of DiT attention layers to generate high-quality saliency maps that precisely locate textual concepts within images. Without requiring additional training, CONCEPTATTENTION repurposes the parameters of DiT attention layers to produce highly contextualized *concept embeddings*, contributing the major discovery that performing linear projections in the output space of DiT attention layers yields significantly sharper saliency maps compared to commonly used cross-attention maps. CONCEPTATTENTION even achieves state-of-theart performance on zero-shot image segmentation benchmarks, outperforming 15 other zeroshot interpretability methods on the ImageNet-Segmentation dataset. CONCEPTATTENTION works for popular image models and even seamlessly generalizes to video generation. Our work contributes the first evidence that the representations of multi-modal DiTs are highly transferable to vision tasks like segmentation.

## 1. Introduction

Diffusion models have recently gained widespread popularity, emerging as the state-of-the-art approach for a variety of generative tasks, particularly text-to-image synthesis [\(Rom](#page-11-0)[bach et al.,](#page-11-0) [2022\)](#page-11-0). These models transform random noise into photorealistic images guided by textual descriptions, achieving unprecedented fidelity and detail. Despite the impressive generative capabilities of diffusion models, our understanding of their internal mechanisms remains limited. Diffusion models operate as black boxes, where the

Figure 1. CONCEPTATTENTION produces saliency maps that precisely localize the presence of textual concepts in images. We compare Flux raw cross attention, DAAM [\(Tang et al.,](#page-11-1) [2022\)](#page-11-1) with SDXL, and TextSpan [\(Gandelsman et al.,](#page-10-0) [2024\)](#page-10-0) for CLIP.

relationships between input prompts and generated outputs are visible, but the decision-making processes that connect them are hidden from human understanding.

Existing work on interpreting T2I models has predominantly focused on UNet-based architectures [\(Podell et al.,](#page-11-2) [2023;](#page-11-2) [Rombach et al.,](#page-11-0) [2022\)](#page-11-0), which utilize shallow cross-attention mechanisms between prompt embeddings and image patch representations. UNet *cross attention maps* can produce high-fidelity saliency maps that predict the location of textual concepts [\(Tang et al.,](#page-11-1) [2022\)](#page-11-1) and have found numerous applications in tasks like image editing [\(Hertz et al.,](#page-10-1) [2022;](#page-10-1) [Chefer et al.,](#page-9-0) [2023\)](#page-9-0). However, the interpretability

<sup>1</sup>Georgia Tech <sup>2</sup>Virginia Tech <sup>3</sup> IBM Research. Correspondence to: Alec Helbling <alechelbling@gatech.edu>.

Code: <alechelbling.com/ConceptAttention/>

![](_page_1_Diagram_2.jpeg)

Figure 2. CONCEPTATTENTION augments multi-modal DiTs with a sequence of concept embeddings that can be used to produce saliency maps. (Left) An unmodified multi-modal attention (MMATTN) layer processes both prompt and image tokens. (Right) CONCEPTATTENTION augments these layers without impacting the image appearance to create a set of contextualized concept tokens.

of more recent multi-modal diffusion transformers (DiTs) remains underexplored. DiT-based models have recently replaced UNets [\(Ronneberger et al.,](#page-11-3) [2015\)](#page-11-3) as the state-ofthe-art architecture for image generation, with models such as Flux [\(Labs,](#page-10-2) [2023\)](#page-10-2) and SD3 [\(Esser et al.,](#page-9-1) [2024\)](#page-9-1) achieving breakthroughs in text-to-image generation. The rapid advancement and enhanced capabilities of DiT-based models highlight the critical importance of methods that improve their interpretability, transparency, and safety.

In this work, we propose CONCEPTATTENTION, a novel method that leverages the representations of multi-modal DiTs to produce high-fidelity saliency maps that localize textual concepts within images. Our method provides insight into the rich semantics of DiT representations. CONCEPTAT-TENTION is lightweight and requires no additional training, instead it repurposes the existing parameters of DiT attention layers. CONCEPTATTENTION works by producing a set of rich contextualized text embeddings each corresponding to visual concepts (e.g. "dragon", "sun"). By linearly projecting these *concept embeddings* and the image we can produce rich saliency maps that are even higher quality than commonly used cross attention maps.

We evaluate the efficacy of CONCEPTATTENTION in a zeroshot semantic segmentation task on real world images. We compare our interpretative maps against annotated segmentations to measure the accuracy and relevance of the attributions generated by our method. Our experiments and extensive comparisons demonstrate that CONCEPTATTEN-TION provides valuable insights into the inner workings of these otherwise complex black-box models. By explaining the meaning of the representations of generative models our method paves the way for advancements in interpretability, controllability, and trust in generative AI systems.

In summary, we contribute:

- CONCEPTATTENTION, a method for interpreting

text-to-image diffusion transformers. Our method requires no additional training, by leveraging the representations of multi-modal DiTs to generate highly interpretable saliency maps that depict the presence of arbitrary textual concepts (e.g. "dragon", "sky", etc.) in images (as shown in Figure [1\)](#page-0-0).

- The novel discovery that the output vectors of attention operations produce higher-quality saliency maps than cross attentions. CONCEPTATTENTION repurposes the parameters of DiT attention layers to produce rich textual embeddings corresponding to different concepts, something that is uniquely enabled by multi-modal DiT architectures. By performing linear projections between these *concept embeddings* and image patch representations in the attention output space we can produce high quality saliency maps.
- CONCEPTATTENTION achieves state-of-the-art performance in zero-shot segmentation on benchmarks like ImageNet Segmentation and Pascal VOC across multiple DiT architectures. We achieve superior performance to a diverse set of zero-shot interpretability methods based on various foundation models like CLIP, DINO, and UNet-based diffusion models; this highlights the potential for the representations of DiTs to transfer to important downstream vision tasks like segmentation. We replicate our results quantitatively on both Flux and Stable Diffusion 3.5 Turbo.
- CONCEPTATTENTION works with a video generation DiT model. Additionally, we demonstrate qualitatively that CONCEPTATTENTION seamlessly generalizes to the CogVideoX [\(Yang et al.,](#page-11-4) [2025\)](#page-11-4) video generation MMDiT model, producing higher-quality saliency maps than native cross attention maps.

![](_page_2_Picture_2.jpeg)

Figure 3. CONCEPTATTENTION can generate high-quality saliency maps for multiple concepts simultaneously. Additionally, our approach is not restricted to concepts in the prompt vocabulary.

## 2. Related Work

Diffusion Model Interpretability A fair amount of existing work attempts to interpret diffusion models. Some works investigate diffusion models from an analytic lens [\(Kadkhodaie et al.,](#page-10-3) [2024;](#page-10-3) [Wang et al.,](#page-11-5) [2024\)](#page-11-5), attempting to understand how diffusion models geometrically model the manifold of data. Other works attempt to understand how models memorize images [\(Carlini et al.,](#page-9-2) [2023\)](#page-9-2). An increasing body of work attempts to repurpose the representations of diffusion models for various tasks like classification [\(Li](#page-10-4) [et al.,](#page-10-4) [2023a\)](#page-10-4), segmentation [\(Karazija et al.,](#page-10-5) [2024\)](#page-10-5), and even robotic control [\(Gupta et al.,](#page-10-6) [2024\)](#page-10-6). However, most relevant to our work is the substantial body of methods investigating how the representations of the neural network architectures underpinning diffusion can be used to garner insight into how these models work, steer their behavior, and improve their safety.

Numerous papers have observed that the cross attention mechanisms of UNet-based diffusion models like Stable Diffusion [\(Rombach et al.,](#page-11-0) [2022\)](#page-11-0) and SDXL [\(Podell et al.,](#page-11-2) [2023\)](#page-11-2) can produce interpretable saliency maps of textual concepts [\(Tang et al.,](#page-11-1) [2022\)](#page-11-1). Cross attention maps are used in a variety of image editing tasks like producing masks that localize objects of interest to edit [\(Dalva et al.,](#page-9-3) [2024\)](#page-9-3), controlling the layout of images [\(Chen et al.,](#page-9-4) [2023;](#page-9-4) [Epstein et al.,](#page-9-5) [2023\)](#page-9-5), altering the appearance of an image but retaining its layout [\(Hertz et al.,](#page-10-1) [2022\)](#page-10-1), and even generating synthetic data to train instruction based editing models [\(Brooks et al.,](#page-9-6) [2023\)](#page-9-6). Other works observe that performing interventions on cross attention maps can improve the faithfulness of images to prompts by ensuring attributes are assigned to the correct objects [\(Meral et al.,](#page-10-7) [2024;](#page-10-7) [Chefer et al.,](#page-9-0) [2023\)](#page-9-0). Additionally, it has been observed that self-attention layers of diffusion models encode useful information about the layout of images [\(Liu et al.,](#page-10-8) [2024\)](#page-10-8).

Zero-shot Image Segmentation In this work, we evaluate CONCEPTATTENTION on the task of zero-shot image segmentation, which is a natural way to assess the accuracy

of our saliency maps and the transferability of the representations of multi-modal DiT architectures to downstream vision tasks. This task also provides a good setting to compare to a variety of other interpretability methods for various foundation model architectures like CLIP [\(Radford et al.,](#page-11-6) [2021\)](#page-11-6), DINO [\(Caron et al.,](#page-9-7) [2021\)](#page-9-7), and diffusion models.

A variety of works train models from scratch for the task of image segmentation [\(Amit et al.,](#page-9-8) [2022;](#page-9-8) [Karazija et al.,](#page-10-5) [2024\)](#page-10-5) or attempt to fine-tune pretrained models [\(Baranchuk et al.,](#page-9-9) [2022\)](#page-9-9). Another line of work leverages diffusion models to generate synthetic data that can be used to train segmentation models that transfer zero-shot to new classes [\(Li et al.,](#page-10-9) [2023b\)](#page-10-9). While effective, these methods are training-based and thus do not provide as much insight into the representations of existing text-to-image generation models, which is the key motivation behind CONCEPTATTENTION.

A significant body of work attempts to improve the interpretability of CLIP vision transformers (ViTs) [\(Dosovitskiy](#page-9-10) [et al.,](#page-9-10) [2021\)](#page-9-10). The authors of [\(Chefer et al.,](#page-9-11) [2021\)](#page-9-11) develop a method for generating saliency maps for ViT models, and they introduce an evaluation protocol for assessing the effectiveness of these saliency maps. This evaluation protocol centers around the ImageNet-Segmentation dataset [\(Guil](#page-10-10)[laumin et al.,](#page-10-10) [2014\)](#page-10-10), and we extend this evaluation to the PascalVOC dataset [\(Everingham et al.,](#page-9-12) [2015\)](#page-9-12). They compare to a variety of zero-shot interpretability methods like GradCAM [\(Selvaraju et al.,](#page-11-7) [2019\)](#page-11-7), Layerwise-Relevance Propagation [\(Binder et al.,](#page-9-13) [2016\)](#page-9-13), raw attentions, and the Rollout method [\(Abnar & Zuidema,](#page-9-14) [2020\)](#page-9-14). The authors of [\(Gandelsman et al.,](#page-10-0) [2024\)](#page-10-0) demonstrate an approach to expressing image patches in terms of textual concepts. We also compare our approach to zero-shot diffusion based methods [\(Tang et al.,](#page-11-1) [2022;](#page-11-1) [Wang et al.,](#page-11-5) [2024\)](#page-11-5) and the self-attention maps of DINO ViT models [\(Caron et al.,](#page-9-7) [2021\)](#page-9-7).

Another line of work performs unsupervised segmentation without any class or text conditioning by performing clustering of the embeddings of models [\(Cho et al.,](#page-9-15) [2021;](#page-9-15) [Hamilton](#page-10-11) [et al.,](#page-10-11) [2022;](#page-10-11) [Tian et al.,](#page-11-8) [2024\)](#page-11-8). Despite not producing class predictions, these models are often evaluated on semantic

CONCEPT ATTENTION Cross Attn (Flux)

DAAM (SDXL) TransInterp (CLIP)

LRP (CLIP) Last Layer LRP (CLIP)

Raw Attn (CLIP)

GradCAM (CLIP)

Rollout (CLIP)

TextSpan (CLIP)

Raw Scores

Preds

Ground Truth

Figure 4. CONCEPTATTENTION produces higher fidelity raw scores and saliency maps than alternative methods, sometimes surpassing in quality even the ground truth saliency map provided by the ImageNet-Segmentation task. Top row shows the soft predictions of each method and the bottom shows the binarized predictions.

segmentation datasets by using approaches like Hungarian matching [\(Kuhn,](#page-10-12) [1955\)](#page-10-12) to pair unlabeled segmentation predictions with the best matching ones in a multi-class semantic segmentation dataset. In contrast, CONCEPTAT-TENTION enables text conditioning so we do not compare to this family of methods. We also don't compare to models like SAM [\(Kirillov et al.,](#page-10-13) [2023;](#page-10-13) [Ravi et al.,](#page-11-9) [2024\)](#page-11-9) as it is trained on a large scale dataset.

### 3. Preliminaries

#### 3.1. Rectified-Flow Models for Image Generation

Flux and Stable Diffusion 3 leverage multi-modal DiTs that are trained to parameterize rectified flow models. Throughout this paper we may refer to rectified flow models as diffusion models for convenience. These models attempt to generate realistic images from noise that correspond to given text prompts. Flow based models [\(Lipman et al.,](#page-10-14) [2023\)](#page-10-14) attempt to map a sample x<sup>1</sup> from a noise distribution p1, typically p<sup>1</sup> ∼ N (0, I), to a sample x<sup>0</sup> in the data distribution. Rectified flows [\(Liu et al.,](#page-10-15) [2022\)](#page-10-15) attempt to learn ODEs that follow straight paths between the p<sup>0</sup> and p1, i.e.

$$z_t = (1 - t)x_0 + t\epsilon, \epsilon \sim \mathcal{N}(0, 1). \quad (1)$$

Flux and SD3 are trained using a conditional flow matching objective which can be expressed conveniently as

$$-\frac{1}{2}\mathbb{E}_{t \sim \mathcal{U}(t), \epsilon \sim \mathcal{N}(0, I)}[w_t \lambda'_t \| \epsilon_{\Theta}(z_t, t) - \epsilon \|^2] \quad (2)$$

where λ ′ t corresponds to a signal-to-noise ratio and w<sup>t</sup> is a time dependent-weighting factor. Above ϵΘ(zt, t) is parameterized by a multi-modal diffusion transformer network. The architecture of this model and it's properties is of primary interest in this work.

#### 3.2. The Anatomy of a Multi-modal DiT Layer

Multi-modal DiTs like Flux and Stable Diffusion 3 leverage *multi-modal attention layers (*MMATTN*)* that process a combination of textual tokens and image patches. There

are two key classes of layers: one that keeps separate residual streams for each modality and one that uses a single stream. In this work, we take advantage of the properties of these dual stream layers, which we refer to as multi-modal attention layers (MMATTNs).

The input to a given layer is a sequence of image patch representations x ∈ R h×w×d and prompt token embeddings p ∈ R l×d . The initial prompt embeddings at the beginning of the network are formed by taking the T5 [\(Raffel et al.,](#page-11-10) [2023\)](#page-11-10) embeddings of the prompt tokens.

Following [\(Peebles & Xie,](#page-11-11) [2023\)](#page-11-11), each MMATTN layer leverages a set of adaptive layer norm *modulation layers* [\(Xu et al.,](#page-11-12) [2019\)](#page-11-12), conditioned on the time-step and global CLIP vector. An adaptive layernorm operation is applied to the input image and text embeddings. The final modulated outputs are then residually added back to the original input. Notably, the image and text modalities are kept in separate residual streams. The exact details of this operation are omitted for brevity.

The key workhorse in MMATTN layers is the familiar multihead self attention operation. The prompt and image embeddings have separate learned key, value, and query projection matrices which we refer to as Kx, Qx, V<sup>x</sup> for images and Kp, Qp, V<sup>p</sup> for text. The keys, queries, and values for both modalities are collectively denoted qxp, kxp, and vxp, where for example kxp = [Kxx1, . . . , Kpp<sup>1</sup> . . . ]. A self attention operation is then performed

$$o_x, o_p = \text{softmax}(q_{xp}k_{xp}^T)v_{xp} \quad (3)$$

Here we refer to o<sup>x</sup> and o<sup>p</sup> as the *attention output* vectors. Another linear layer is then applied to these outputs and added to a separate residual streams weighted according to the output of the modulation layer. This gives us updated embeddings x <sup>L</sup>+1 and p <sup>L</sup>+1 which are given as input to the next layer.

## 4. Methods

We introduce CONCEPTATTENTION, a method for generating high quality saliency maps depicting the location of

![](_page_4_Diagram_1.jpeg)

Figure 5. (a) MMATTN combines cross and self attention operations between the prompt and image tokens. (b) Our CONCEPTAT-TENTION allows the concept tokens to incorporate information from other concept tokens and the image tokens, but not the other way around.

textual concepts in images. CONCEPTATTENTION works by creating a set of contextualized *concept embeddings* for simple textual concepts (e.g. "cat", "sky", etc.). These concept embeddings are sequentially updated alongside the text and image embeddings, so they match the structure that each MMATTN layer expects. However, unlike the text prompt, concept embeddings do not impact the appearance of the image. We can produce high-fidelity saliency maps by projecting image patch representations onto the concept embeddings. CONCEPTATTENTION requires no additional training and has minimal impact on model latency and memory footprint. A high level depiction of our methodology is shown in Figure [2.](#page-1-0)

#### 4.1. Using CONCEPTATTENTION

The user specifies a set of r single token concepts, like "cat", "sky", etc. which are passed through a T5 encoder to produce an initial embedding c 0 for each concept. For each MMATTN layer (indexed by L) we layer-normalize the input concept embeddings c <sup>L</sup> and repurpose the text prompt's projection matrices (i.e. Kp, Qp, Vp), to produce a set of keys, values, and queries

$$k_c = [K_p c_1, \dots], q_c = [Q_p c_1, \dots], v_c = [V_p c_1, \dots] \in \mathbb{R}^{r \times d}. \quad (4)$$

One-directional Attention Operation We would like to update our concept embeddings so they are compatible with subsequent layers, but also prevent them from impacting the image tokens. Let k<sup>x</sup> and v<sup>x</sup> be the keys and values of the image patches x respectively. We can concatenate the image and concept keys to get

$$k_{xc} = [K_x x_1 \dots, K_x x_n, K_p c_1 \dots, K_p c_r] \quad (5)$$

and the image and concept values to get

$$v_{xc} = [V_x x_1 \dots, V_x x_n, V_p c_1 \dots, V_p c_r] \quad (6)$$

We can then perform the following attention operation

$$o_c = \text{softmax}(q_c k_{xc}^T) v_{xc} \quad (7)$$

which produces a set of *concept output embeddings*.

Notice, that instead of just performing a cross attention (i.e. softmax(qck T x )vx) our approach leverages both cross attention from the image patches to the concepts and self attention among the concepts. We found that performing both operations improves performance on downstream evaluation tasks like segmentation (See Table [4\)](#page-6-0). We hypothesize this is because it allows the concept embeddings to repel from each other, avoiding redundancy between concepts.

Meanwhile, the image patch and prompt tokens ignore the concept tokens and attend only to each other as in

$$o_x, o_p = \text{softmax}(q_{xp} k_{xp}^T) v_{xp}. \quad (8)$$

A diagram of these operations is shown in Figure [5\(](#page-4-0)b).

A Concept Residual Stream The above operations create a residual stream of concept embeddings distinct from the image and patch embeddings. Following the pretrained transformer's design, after the MMATTN we apply another projection matrix P and MLP, adding the result residually to c <sup>L</sup>. We apply an adaptive layernorm at the end of the attention operation which outputs several values: a scale γ, shift β, and some gating values α<sup>1</sup> and α2. The residual stream is then updated as

$$c^{L+1} \leftarrow c^L + \alpha_1(Po_c) \quad (9)$$

$$c^{L+1} \leftarrow c^{L+1} + \alpha_2 \text{MLP} \left( (1 + \gamma) \text{lnorm}(c^{L+1}) + \beta \right) \quad (10)$$

where ← denotes assignment. The parameters from each of our modulation, projection, and MLP layers are the same as those used to process the text prompt.

Saliency Maps in the Attention Output Space These concept embeddings can be combined with the image patch embeddings to produce saliency maps for each layer L. Specifically, we found that taking a simple dot-product similarity between the image output vectors o<sup>x</sup> and concept output vectors o<sup>c</sup> produces high-quality saliency maps

| Method LRP (Binder et al., 2016) Partial-LRP (Binder et al., 2016) Rollout (Abnar & Zuidema, 2020) ViT Attention (Dosovitskiy et al., GradCAM (Selvaraju et al., 2020) TextSpan (Gandelsman et al., TransInterp (Chefer et al., 2021) CLIPasRNN (Sun et al., 2024) OVAM (Marcos-Manchon et al. ´ , DINO SA (Caron et al., 2021) DINOv2 SA (Oquab et al., 2024) DINOv2 Reg SA (Darcet et al., iBOT SA (Zhou et al., 2022) DAAM (Tang et al., 2022) DAAM (Tang et al., 2022) | Architecture CLIP ViT CLIP ViT CLIP ViT 2021) CLIP ViT CLIP ViT 2024) CLIP ViT CLIP ViT CLIP ViT 2024) SDXL UNet DINO ViT DINOv2 ViT 2024) DINOv2 Reg iBOT ViT SDXL UNet SD2 UNet | Acc ↑ 51.09 76.31 73.54 67.84 64.44 75.21 79.70 74.05 79.41 81.97 77.39 72.04 76.34 78.47 64.52 | ImageNet-Segmentation mIoU ↑ 32.89 57.94 55.42 46.37 40.82 54.50 61.95 58.80 65.02 69.44 63.12 56.31 61.73 64.56 47.62 | mAP ↑ 55.68 84.67 84.76 80.24 71.60 81.61 86.03 84.80 88.12 86.12 84.19 80.83 82.04 88.79 78.01 | PascalVOC Acc ↑ 48.77 71.52 69.81 68.51 70.44 75.00 76.90 61.76 73.50 80.71 79.65 77.16 74.96 72.76 64.28 | (Single mIoU ↑ 31.44 51.39 51.26 44.81 44.90 56.24 57.08 41.48 58.12 64.33 57.61 56.60 55.80 55.95 45.01 | Class) mAP ↑ 52.89 84.86 85.34 83.63 76.80 84.79 86.74 76.57 87.91 88.90 87.26 86.35 85.26 88.34 83.04 |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| Cross Attention                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Flux DiT                                                                                                                                                                          | 74.92                                                                                           | 59.90                                                                                                                  | 87.23                                                                                           | 80.37                                                                                                     | 54.77                                                                                                    | 89.08                                                                                                  |
| Cross Attention                                                                                                                                                                                                                                                                                                                                                                                                                                                            | SD3.5 DiT                                                                                                                                                                         | 77.80                                                                                           | 63.67                                                                                                                  | 83.50                                                                                           | 80.22                                                                                                     | 61.46                                                                                                    | 86.97                                                                                                  |
| C ONCEPT A TTENTION                                                                                                                                                                                                                                                                                                                                                                                                                                                        | SD3.5 DiT                                                                                                                                                                         | 81.92                                                                                           | 67.47                                                                                                                  | 90.79                                                                                           | 83.90                                                                                                     | 69.93                                                                                                    | 90.02                                                                                                  |
| C ONCEPT A TTENTION                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Flux DiT                                                                                                                                                                          | 83.07                                                                                           | 71.04                                                                                                                  | 90.45                                                                                           | 87.85                                                                                                     | 76.45                                                                                                    | 90.19                                                                                                  |

Table 1. CONCEPTATTENTION outperforms a variety of Diffusion, DINO, and CLIP ViT interpretability methods on ImageNet-Segmentation and PascalVOC (Single Class).

$$\phi(o_x, o_c) = \text{softmax}(o_x o_c^T). \quad (11)$$

This is in contrast to cross attention maps which are between the image keys k<sup>x</sup> and prompt queries qp.

We can aggregate the information from multiple layers by averaging them <sup>1</sup> |L| P<sup>|</sup>L<sup>|</sup> <sup>L</sup>=1 ϕ(o L x , o<sup>L</sup> c ) where |L| denotes the number of MMATTN layers (Flux has |L| = 18). These attention output space maps are unique to MM-DiT models as they leverage *concept embeddings* corresponding to textual concepts which fundamentally can not be produced by UNet-based models.

### 4.2. Limitations of Raw Cross Attention Maps

For multi-modal DiT architectures, we could additionally consider using the raw cross attention maps

$$\phi(k_x, q_p) = \text{softmax}(q_p k_x^T) \quad (12)$$

to produce saliency maps. However, these have a key limitation in that their vocabulary is limited to the tokens in the user's prompt. Unlike UNet-based models, multi-modal DiTs sequentially update a set of prompt embeddings with each MMATTN layer. This makes it difficult to produce cross attention maps for an open-set of concepts, as you would need to add the concept to the prompt sequence which would then change the appearance of the image. CONCEP-TATTENTION overcomes this key limitation, and makes the additional discovery that the output space of attention mechanisms produces high-fidelity saliency maps.

| Method               | Acc ↑ | mIoU ↑ |
|----------------------|-------|--------|
| TextSpan             | 73.84 | 38.10  |
| DAAM                 | 62.89 | 10.97  |
| Flux Cross Attention | 79.52 | 27.04  |
| C ONCEPT A TTENTION  | 86.99 | 51.39  |

Table 2. CONCEPTATTENTION outperforms alternative methods on images with multiple classes from PascalVOC. Notably, the margin between CONCEPTATTENTION and other methods is even higher for this task than when a single class is in each image.

### 5. Experiments

### 5.1. Implementation Details

Flux DiT For most of our experiments we use the Flux DiT architecture implemented in PyTorch [\(Paszke et al.,](#page-11-16) [2019\)](#page-11-16). In particular, we use the distilled Flux-Schnell model. We encode real images with the DiT by first mapping them to the VAE latent space and then adding varying degrees of Gaussian noise before passing them through the Flux DiT. We then cache all of the concept output o<sup>c</sup> and image output vectors o<sup>x</sup> from each MMATTN layer. At the end of generation we then construct our concept saliency maps for each layer and average them over all layers of interest. In our experiments we leverage the activations from the last 10 of the 18 MMATTN layers.

Stable Diffusion 3.5 Turbo We found that our approach replicated on the Stable Diffusion 3.5 Turbo [\(Esser et al.,](#page-9-1) [2024\)](#page-9-1) DiT architecture (Figure [6\)](#page-6-1).

![](_page_6_Figure_1.jpeg)

Figure 6. CONCEPTATTENTION is capable of generating high quality saliency maps with Stable Diffusion 3.5 Turbo. Furthermore, the top example highlights a potential failure case of CONCEPTATTENTION. The concepts "sky", "mountain", and "sun" all semantically overlap, resulting in unclear object boundaries.

| Space  | Softmax Acc ↑ | mIoU ↑ | mAP ↑ |
|--------|---------------|--------|-------|
| CA     | 66.59         | 49.91  | 73.17 |
| CA     | ✓ 74.92       | 59.90  | 87.23 |
| Value  | 45.93         | 29.81  | 65.79 |
| Value  | ✓ 45.78       | 29.68  | 39.61 |
| Output | 78.75         | 64.95  | 88.39 |
| Output | ✓ 83.07       | 71.04  | 90.45 |

Table 3. The output space of DiT attention layers produces more transferable representations than cross attentions. We explore the transferability of several representation spaces of a DiT: the cross attentions (CA), the value space, and the output space. We performed linear projections of the image patches and concept vectors in each of these spaces and evaluated their performance on ImageNet-Segmentation.

CogVideo X CONCEPTATTENTION generalizes to the CogVideoX [\(Yang et al.,](#page-11-4) [2025\)](#page-11-4) multi-modal DiT video generation model. The only change we make is additionally averaging over the added frame dimension.

#### 5.2. Zero-shot Image Segmentation

We are interested in investigating (1) the efficacy of CON-CEPTATTENTION to generate highly localized and semantically meaningful saliency maps, and (2) understand the transferability of multi-modal DiT representations to important downstream vision tasks. Zero-shot image segmentation is a natural choice for achieving these goals.

Datasets We leverage two key datasets zero-shot image segmentation datasets. First, we use a commonly used [\(Gan](#page-10-0)[delsman et al.,](#page-10-0) [2024;](#page-10-0) [Chefer et al.,](#page-9-11) [2021\)](#page-9-11) zero-shot segmentation benchmark called ImageNet-Segmentation [\(Guillaumin](#page-10-10) [et al.,](#page-10-10) [2014\)](#page-10-10). It is composed of 4,276 images from 445 categories. Each image primarily depicts a single central object or concept, which makes it a good method for comparing CONCEPTATTENTION to a variety of methods which

| CA SA | Acc ↑ | mIoU ↑ | mAP ↑ |
|-------|-------|--------|-------|
|       | 52.63 | 35.72  | 70.21 |
| ✓     | 51.68 | 34.85  | 69.36 |
| ✓     | 76.51 | 61.96  | 86.73 |
| ✓ ✓   | 83.07 | 71.04  | 90.45 |

Table 4. CONCEPTATTENTION performs best when we utilize both cross and self attention. We tested the effectiveness of performing just a cross attention operation between the concepts and image tokens, just a self attention among the concepts, both cross and self attention, and neither. We found that doing both operations leads to the best results. Metrics are computed on the ImageNet Segmentation benchmark.

generate a single saliency map that are unable to generate class-specific segmentation maps. For the second dataset we leverage PascalVOC 2012 benchmark [\(Everingham et al.,](#page-9-12) [2015\)](#page-9-12). We investigate both a single class (930 images) and multi-class split (1,449 images) of this dataset. Many methods (e.g. DINO) do not condition their saliency map on class, so for these methods we restrict our evaluation to examples only containing a single class and the background. For methods that can accept text as conditioning we evaluate on the full dataset.

Key Baseline Methods We compare our approach to a variety of zero-shot interpretability methods which leverage several different multi-modal foundation models. We compare to numerous interpretability methods compatible with CLIP: Layerwise Relevance Propagation (LRP) [\(Binder et al.,](#page-9-13) [2016\)](#page-9-13), LRP on just the final-layer of a ViT (Partial-LRP), Attention Rollout (Rollout) [\(Abnar &](#page-9-14) [Zuidema,](#page-9-14) [2020\)](#page-9-14), Raw Vision Transformer Attention (ViT Attention) [\(Dosovitskiy et al.,](#page-9-10) [2021\)](#page-9-10), GradCAM [\(Selvaraju](#page-11-7) [et al.,](#page-11-7) [2019\)](#page-11-7), TextSpan [\(Gandelsman et al.,](#page-10-0) [2024\)](#page-10-0), CLIP as RNN [\(Sun et al.,](#page-11-14) [2024\)](#page-11-14), and the Transformer Attribution method from [\(Chefer et al.,](#page-9-11) [2021\)](#page-9-11) (TransInterp). We also

![](_page_7_Figure_1.jpeg)

Figure 7. (Left) Later MMATTN layers encode richer features for zero-shot segmentation. We investigated the impact of using features from various MMATTN layers and found that deeper layers lead to better performance on segmentation metrics like pixelwise accuracy, mIoU, and mAP. We also found that combining the information from all layers further improves performance. (Right) Optimal segmentation performance requires some noise to be present in the image. We evaluated the performance of CONCEPTATTENTION by encoding samples from a variety of timesteps (determines the amount of noise). Interestingly, we found that the optimal amount of noise was not zero, but in the middle to later stages of the noise schedule.

compare to UNet-based interpretability methods that aggregates information from UNet cross attention layers called DAAM [\(Tang et al.,](#page-11-1) [2022\)](#page-11-1) on both SDXL [\(Podell et al.,](#page-11-2) [2023\)](#page-11-2) and SD2, and OVAM [\(Li et al.,](#page-10-9) [2023b\)](#page-10-9) with SDXL. We compare to the self-attention maps of various DINO models: DINOv1 [\(Caron et al.,](#page-9-7) [2021\)](#page-9-7), DINOv2 [\(Oquab](#page-10-17) [et al.,](#page-10-17) [2024\)](#page-10-17), and DINOv2 with registers [\(Darcet et al.,](#page-9-16) [2024\)](#page-9-16). Finally, we compare to the raw cross attention maps produced by Flux and Stable Diffusion 3.5 Turbo.

Single Object Image Segmentation For our first task we closely follow the established evaluation framework from [\(Gandelsman et al.,](#page-10-0) [2024\)](#page-10-0) and [\(Chefer et al.,](#page-9-11) [2021\)](#page-9-11). We perform this evaluation setup on both ImageNet-Segmentation and a subset of 930 PascalVOC images containing only a single class. For each method we assume the class present in the image is known and use simplified descriptions of each ImageNet class (e.g. "Maltese dog" → "dog) this allows the concepts to be captured by a single token. We construct a concept vocabulary for each image composed of this target class and a set of fixed background concepts for all examples (e.g. "background", "grass", "sky").

Quantitative Evaluation Each method produces a set of scalar *raw scores* for each image patch which we then threshold based on the mean value to produce a binary segmentation prediction. We compare each method using standard segmentation evaluation metrics, namely: mean Intersection over Union (mIoU), patch/pixelwise accuracy (Acc), and mean Average Precision (mAP). Accuracy alone is an insufficient metric as our dataset is highly imbalanced, mIoU is significantly better, and mAP captures threshold agnostic segmentation capability. We found that CONCEPTATTEN-TION significantly out performs all of the baselines we tested

across all three metrics (Table [1\)](#page-5-0). This is true for diffusion, CLIP, and DINO based interpretability methods.

Qualitative Evaluation We show qualitative results comparing the segmentation performance to each baseline in Figure [4](#page-3-0) and more qualitative results in Appendix [B.](#page-13-0) It is worth noting that the qualitative segmentation results highlight (a) the ambiguity of zero-shot image segmentation, and (b) the limitations of human data annotation. For example, Figure [4](#page-3-0) shows our method does not segment the part of the dog between the ears and it's body, while the ground truth does.

Multi Object Image Segmentation We also wanted to evaluate the capabilities of our method at differentiating between multiple classes in an image. However, only a subset of methods produce distinct saliency maps for open ended classes. For this experiment we compare to DAAM using a SDXL backbone, TextSpan using a CLIP backbone, and the raw cross attentions of Flux. Instead of binarizing the image to produce segmentations, for each patch we predict the concept with the highest score. We used pixelwise accuracy and mIoU as our evaluation metrics and found that our method significantly outperformed the baselines (Table [2\)](#page-5-1). We also show qualitative results of our approach differentiating between multiple concepts in a single image in Figures [1,](#page-0-0) [3](#page-2-0) and we show more results in Appendix [B.](#page-13-0)

### 5.3. Ablation Studies

We perform several ablation studies to investigate the impact of various architectural choices and hyperparameters on the performance of CONCEPTATTENTION.

![](_page_8_Figure_1.jpeg)

Figure 8. CONCEPTATTENTION generalizes seamlessly to video generation MMDiT models like CogVideoX. We apply CONCEP-TATTENTION to a CogVideoX [\(Yang et al.,](#page-11-4) [2025\)](#page-11-4) video generation model. We take several frames from the video and compare the saliency maps generated by CONCEPTATTENTION to the model's internal cross attention maps.

Impact of Layer Depth on Segmentation We hypothesized that deeper MMATTN layers in the DiT would have more refined representations that better transfer to segmentation. This was confirmed by our evaluation (Figure [7\)](#page-7-0). We pull features from each diffusion layer and evaluated the segmentation performance on ImageNet-Segmentation. We also compare the performance of combining all layers simultaneously, which we found performs better than any individual layer.

Impact of Diffusion Timestep on Segmentation We add Gaussian noise to encoded images before passing them to the DiTs, this conforms with the expectations of the models. Intuitively one might expect the later timesteps (less noise) to have much higher segmentation performance as less information about the original image is corrupted. However, we found that the middle diffusion timesteps best (Figure [7\)](#page-7-0). Throughout the rest of our experiments we use timestep 500 out of 1000 following this result.

Concept Attention Operation Ablations We compared the performance on the ImageNet Segmentation benchmark of performing (a) just cross attention from the image patches to the concept vectors, (b) just self attention, (c) no attention operations, and (d) both cross and self attention. Our results seen in Table [4](#page-6-0) indicate that using a combination of both cross and self attention achieves the best performance. We also investigated the impact of applying a pixelwise softmax operation over our predicted segmentation coefficients. We found that it slightly improves segmentation performance in the attention output space and significantly improves performance for the cross attention maps (Table [3\)](#page-6-2)

### 5.4. Video Model Results

We include qualitative results demonstrating the efficacy of CONCEPTATTENTION on the CogVideoX video generation multi-modal DiT model (Figure [8\)](#page-8-0). Also see Figures [16](#page-16-0) and [17](#page-17-0) in the Appendix.

### 5.5. Limitations

The primary limitation of CONCEPTATTENTION is that it struggles to differentiate between very similar textual concepts. For example, for a photo with a sky with the sun in it, the model does not necessarily know where the boundary of the sun resides, instead capturing a halo around the sun (Figure [6\)](#page-6-1). Additionally, when no relevant concept is present, CONCEPTATTENTION will select the most similar one even if it is incorrect (Figure [15](#page-15-0) in the Appendix).

### 6. Conclusion

We introduce CONCEPTATTENTION, a method for interpreting the rich features of multi-modal DiTs. Our approach allows a user to produce high quality saliency maps of an open-set of textual concepts that shed light on how a diffusion model "sees" an image. We perform an extensive evaluation of the saliency maps on zero-shot segmentation and find that they significantly outperform a variety of other zero-shot interpretability methods. Our results suggest the potential for DiT models to act as powerful and interpretable image encoders with representations that are transferable zero-shot to tasks like image segmentation.

- Impact Statement Generative models for images have numerous ethical concerns: they have the potential to spread misinformation through realistic fake images (i.e. deepfakes), they may disrupt different creative industries, and have the potential to reinforce existing social biases present in their training data. Our work directly serves to improve the transparency of these models, and we believe our work could be used to understand the biases present in models. Acknowledgments This paper is supported by the National Science Foundation Graduate Research Fellowship. This work was also supported in part by Cisco, NSF #2403297, gifts from Google, Amazon, Meta, NVIDIA, Avast, Fiddler Labs, Bosch. References Abnar, S. and Zuidema, W. Quantifying Attention Flow in Transformers, May 2020. URL [http://arxiv.org/](http://arxiv.org/abs/2005.00928) [abs/2005.00928](http://arxiv.org/abs/2005.00928). arXiv:2005.00928 [cs]. Amit, T., Shaharbany, T., Nachmani, E., and Wolf, L. SegDiff: Image Segmentation with Diffusion Probabilistic Models, September 2022. URL [http://arxiv.](http://arxiv.org/abs/2112.00390) [org/abs/2112.00390](http://arxiv.org/abs/2112.00390). arXiv:2112.00390 [cs]. Baranchuk, D., Rubachev, I., Voynov, A., Khrulkov, V., and Babenko, A. Label-Efficient Semantic Segmentation with Diffusion Models, March 2022. URL [http://arxiv.](http://arxiv.org/abs/2112.03126) [org/abs/2112.03126](http://arxiv.org/abs/2112.03126). arXiv:2112.03126 [cs]. Binder, A., Montavon, G., Bach, S., Muller, K.-R., and ¨ Samek, W. Layer-wise Relevance Propagation for Neural Networks with Local Renormalization Layers, April 2016. URL [http://arxiv.org/abs/1604.](http://arxiv.org/abs/1604.00825) [00825](http://arxiv.org/abs/1604.00825). arXiv:1604.00825 [cs]. Brooks, T., Holynski, A., and Efros, A. A. InstructPix2Pix: Learning to Follow Image Editing Instructions, January 2023. URL [http://arxiv.org/abs/2211.](http://arxiv.org/abs/2211.09800) [09800](http://arxiv.org/abs/2211.09800). arXiv:2211.09800 [cs]. Carlini, N., Hayes, J., Nasr, M., Jagielski, M., Sehwag, V., Tramer, F., Balle, B., Ippolito, D., and Wallace, E. ` Extracting Training Data from Diffusion Models, January 2023. URL [http://arxiv.org/abs/2301.](http://arxiv.org/abs/2301.13188) [13188](http://arxiv.org/abs/2301.13188). arXiv:2301.13188 [cs]. Caron, M., Touvron, H., Misra, I., Jegou, H., Mairal, ´ J., Bojanowski, P., and Joulin, A. Emerging Properties in Self-Supervised Vision Transformers, May 2021. URL [http://arxiv.org/abs/2104.](http://arxiv.org/abs/2104.14294) [14294](http://arxiv.org/abs/2104.14294). arXiv:2104.14294 [cs]. Chefer, H., Gur, S., and Wolf, L. Transformer Interpretability Beyond Attention Visualization, April 2021. URL [http://arxiv.org/abs/2012.](http://arxiv.org/abs/2012.09838) [09838](http://arxiv.org/abs/2012.09838). arXiv:2012.09838 [cs]. Chefer, H., Alaluf, Y., Vinker, Y., Wolf, L., and Cohen-Or, D. Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models. *ACM Transactions on Graphics*, 42(4):148:1–148:10, July 2023. ISSN 0730- 0301. doi: 10.1145/3592116. URL [https://dl.acm.](https://dl.acm.org/doi/10.1145/3592116) [org/doi/10.1145/3592116](https://dl.acm.org/doi/10.1145/3592116). Chen, M., Laina, I., and Vedaldi, A. Training-Free Layout Control with Cross-Attention Guidance, November 2023. URL [http://arxiv.org/abs/2304.](http://arxiv.org/abs/2304.03373) [03373](http://arxiv.org/abs/2304.03373). arXiv:2304.03373 [cs]. Cho, J. H., Mall, U., Bala, K., and Hariharan, B. PiCIE: Unsupervised Semantic Segmentation using Invariance and Equivariance in Clustering, March 2021. URL [http://](http://arxiv.org/abs/2103.17070) [arxiv.org/abs/2103.17070](http://arxiv.org/abs/2103.17070). arXiv:2103.17070 [cs]. Dalva, Y., Venkatesh, K., and Yanardag, P. FluxSpace: Disentangled Semantic Editing in Rectified Flow Transformers, December 2024. URL [http://arxiv.org/](http://arxiv.org/abs/2412.09611) [abs/2412.09611](http://arxiv.org/abs/2412.09611). arXiv:2412.09611 [cs]. Darcet, T., Oquab, M., Mairal, J., and Bojanowski,
  - P. Vision Transformers Need Registers, April 2024. URL [http://arxiv.org/abs/2309.](http://arxiv.org/abs/2309.16588) [16588](http://arxiv.org/abs/2309.16588). arXiv:2309.16588 [cs]. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby,
  - N. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, June 2021. URL [http://](http://arxiv.org/abs/2010.11929) [arxiv.org/abs/2010.11929](http://arxiv.org/abs/2010.11929). arXiv:2010.11929 [cs]. Epstein, D., Jabri, A., Poole, B., Efros, A. A., and Holynski,
  - A. Diffusion Self-Guidance for Controllable Image Generation, June 2023. URL [http://arxiv.org/abs/](http://arxiv.org/abs/2306.00986) [2306.00986](http://arxiv.org/abs/2306.00986). arXiv:2306.00986 [cs]. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Muller, ¨ J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., Podell, D., Dockhorn, T., English, Z., Lacey, K., Goodwin, A., Marek, Y., and Rombach, R. Scaling Rectified Flow Transformers for High-Resolution Image Synthesis, March 2024. URL [http://arxiv.org/abs/](http://arxiv.org/abs/2403.03206) [2403.03206](http://arxiv.org/abs/2403.03206). arXiv:2403.03206. Everingham, M., Eslami, S. M. A., Van Gool, L., Williams,
    - C. K. I., Winn, J., and Zisserman, A. The Pascal Visual Object Classes Challenge: A Retrospective.

- *International Journal of Computer Vision*, 111(1):98– 136, January 2015. ISSN 1573-1405. doi: 10.1007/ s11263-014-0733-5. URL [https://doi.org/10.](https://doi.org/10.1007/s11263-014-0733-5) [1007/s11263-014-0733-5](https://doi.org/10.1007/s11263-014-0733-5). Gandelsman, Y., Efros, A. A., and Steinhardt, J. Interpreting CLIP's Image Representation via Text-Based Decomposition, March 2024. URL [http://arxiv.org/abs/](http://arxiv.org/abs/2310.05916) [2310.05916](http://arxiv.org/abs/2310.05916). arXiv:2310.05916 [cs]. Guillaumin, M., Kuttel, D., and Ferrari, V. ImageNet ¨ Auto-Annotation with Segmentation Propagation. *International Journal of Computer Vision*, 110(3):328– 348, December 2014. ISSN 1573-1405. doi: 10.1007/ s11263-014-0713-9. URL [https://doi.org/10.](https://doi.org/10.1007/s11263-014-0713-9) [1007/s11263-014-0713-9](https://doi.org/10.1007/s11263-014-0713-9). Gupta, G., Yadav, K., Gal, Y., Batra, D., Kira, Z., Lu, C., and Rudner, T. G. J. Pre-trained Text-to-Image Diffusion Models Are Versatile Representation Learners for Control, May 2024. URL [http://arxiv.org/abs/](http://arxiv.org/abs/2405.05852) [2405.05852](http://arxiv.org/abs/2405.05852). arXiv:2405.05852 [cs]. Hamilton, M., Zhang, Z., Hariharan, B., Snavely, N., and Freeman, W. T. Unsupervised Semantic Segmentation by Distilling Feature Correspondences, March 2022. URL [http://arxiv.org/abs/](http://arxiv.org/abs/2203.08414) [2203.08414](http://arxiv.org/abs/2203.08414). arXiv:2203.08414 [cs]. Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., and Cohen-Or, D. Prompt-to-Prompt Image Editing with Cross Attention Control, August 2022. URL [http://arxiv.org/abs/2208.](http://arxiv.org/abs/2208.01626) [01626](http://arxiv.org/abs/2208.01626). arXiv:2208.01626 [cs]. Kadkhodaie, Z., Guth, F., Simoncelli, E. P., and Mallat, S. Generalization in diffusion models arises from geometry-adaptive harmonic representations, April 2024. URL [http://arxiv.org/abs/2310.](http://arxiv.org/abs/2310.02557) [02557](http://arxiv.org/abs/2310.02557). arXiv:2310.02557 [cs]. Karazija, L., Laina, I., Vedaldi, A., and Rupprecht, C. Diffusion Models for Open-Vocabulary Segmentation, September 2024. URL [http://arxiv.org/abs/2306.](http://arxiv.org/abs/2306.09316) [09316](http://arxiv.org/abs/2306.09316). arXiv:2306.09316 [cs]. Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollar, P., and Girshick, R. Segment Anything, ´ April 2023. URL [http://arxiv.org/abs/2304.](http://arxiv.org/abs/2304.02643) [02643](http://arxiv.org/abs/2304.02643). arXiv:2304.02643 [cs]. Kuhn, H. W. The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, 2(1-2):83–97, 1955. ISSN 1931- 9193. doi: 10.1002/nav.3800020109. URL [https://onlinelibrary.wiley.com/doi/](https://onlinelibrary.wiley.com/doi/abs/10.1002/nav.3800020109) [abs/10.1002/nav.3800020109](https://onlinelibrary.wiley.com/doi/abs/10.1002/nav.3800020109). eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1002/nav.3800020109. Labs, B. F. FLUX, 2023. URL [https://github.com/](https://github.com/black-forest-labs/flux) [black-forest-labs/flux](https://github.com/black-forest-labs/flux). Li, A. C., Prabhudesai, M., Duggal, S., Brown, E., and Pathak, D. Your Diffusion Model is Secretly a Zero-Shot Classifier, September 2023a. URL [http://arxiv.](http://arxiv.org/abs/2303.16203) [org/abs/2303.16203](http://arxiv.org/abs/2303.16203). arXiv:2303.16203 [cs]. Li, Z., Zhou, Q., Zhang, X., Zhang, Y., Wang, Y., and Xie,
  - W. Open-vocabulary Object Segmentation with Diffusion Models, August 2023b. URL [http://arxiv.org/](http://arxiv.org/abs/2301.05221) [abs/2301.05221](http://arxiv.org/abs/2301.05221). arXiv:2301.05221 [cs]. Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., and Le, M. Flow Matching for Generative Modeling, February 2023. URL [http://arxiv.org/abs/2210.](http://arxiv.org/abs/2210.02747) [02747](http://arxiv.org/abs/2210.02747). arXiv:2210.02747 [cs]. Liu, B., Wang, C., Cao, T., Jia, K., and Huang,
  - J. Towards Understanding Cross and Self-Attention in Stable Diffusion for Text-Guided Image Editing, March 2024. URL [http://arxiv.org/abs/](http://arxiv.org/abs/2403.03431) [2403.03431](http://arxiv.org/abs/2403.03431). arXiv:2403.03431 [cs]. Liu, X., Gong, C., and Liu, Q. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow, September 2022. URL [http://arxiv.org/](http://arxiv.org/abs/2209.03003) [abs/2209.03003](http://arxiv.org/abs/2209.03003). arXiv:2209.03003 [cs]. Marcos-Manchon, P., Alcover-Couso, R., SanMiguel, J. C., ´ and Mart´ınez, J. M. Open-Vocabulary Attention Maps with Token Optimization for Semantic Segmentation in Diffusion Models, March 2024. URL [http://arxiv.](http://arxiv.org/abs/2403.14291) [org/abs/2403.14291](http://arxiv.org/abs/2403.14291). arXiv:2403.14291 [cs]. Meral, T. H. S., Simsar, E., Tombari, F., and Yanardag, P. CONFORM: Contrast is All You Need for High-Fidelity Text-to-Image Diffusion Models. pp. 9005–9014, 2024. URL [https://openaccess.](https://openaccess.thecvf.com/content/CVPR2024/html/Meral_CONFORM_Contrast_is_All_You_Need_for_High-Fidelity_Text-to-Image_Diffusion_CVPR_2024_paper.html) [thecvf.com/content/CVPR2024/html/](https://openaccess.thecvf.com/content/CVPR2024/html/Meral_CONFORM_Contrast_is_All_You_Need_for_High-Fidelity_Text-to-Image_Diffusion_CVPR_2024_paper.html) [Meral\\_CONFORM\\_Contrast\\_is\\_All\\_You\\_](https://openaccess.thecvf.com/content/CVPR2024/html/Meral_CONFORM_Contrast_is_All_You_Need_for_High-Fidelity_Text-to-Image_Diffusion_CVPR_2024_paper.html) [Need\\_for\\_High-Fidelity\\_Text-to-Image\\_](https://openaccess.thecvf.com/content/CVPR2024/html/Meral_CONFORM_Contrast_is_All_You_Need_for_High-Fidelity_Text-to-Image_Diffusion_CVPR_2024_paper.html) [Diffusion\\_CVPR\\_2024\\_paper.html](https://openaccess.thecvf.com/content/CVPR2024/html/Meral_CONFORM_Contrast_is_All_You_Need_for_High-Fidelity_Text-to-Image_Diffusion_CVPR_2024_paper.html). Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V., Synnaeve, G., Xu, H., Jegou, H., Mairal, J., Labatut, P., Joulin, A., and Bojanowski, P. DINOv2: Learning Robust Visual Features without Supervision, February 2024. URL [http://arxiv.org/abs/2304.](http://arxiv.org/abs/2304.07193) [07193](http://arxiv.org/abs/2304.07193). arXiv:2304.07193 [cs].

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Rai- ¨ son, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., and Chintala, S. PyTorch: An Imperative Style, High-Performance Deep Learning Library, December 2019. URL [http://arxiv.org/abs/1912.](http://arxiv.org/abs/1912.01703) [01703](http://arxiv.org/abs/1912.01703). arXiv:1912.01703 [cs]. Peebles, W. and Xie, S. Scalable Diffusion Models with Transformers, March 2023. URL [http://arxiv.](http://arxiv.org/abs/2212.09748) [org/abs/2212.09748](http://arxiv.org/abs/2212.09748). arXiv:2212.09748. Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Muller, J., Penna, J., and Rombach, R. SDXL: ¨ Improving Latent Diffusion Models for High-Resolution Image Synthesis, July 2023. URL [http://arxiv.](http://arxiv.org/abs/2307.01952) [org/abs/2307.01952](http://arxiv.org/abs/2307.01952). arXiv:2307.01952 [cs]. Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning Transferable Visual Models From Natural Language Supervision, February 2021. URL [http://arxiv.org/abs/2103.](http://arxiv.org/abs/2103.00020) [00020](http://arxiv.org/abs/2103.00020). arXiv:2103.00020 [cs]. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer, September 2023. URL [http://arxiv.](http://arxiv.org/abs/1910.10683) [org/abs/1910.10683](http://arxiv.org/abs/1910.10683). arXiv:1910.10683 [cs]. Ravi, N., Gabeur, V., Hu, Y.-T., Hu, R., Ryali, C., Ma, T., Khedr, H., Radle, R., Rolland, C., Gustafson, L., ¨ Mintun, E., Pan, J., Alwala, K. V., Carion, N., Wu, C.-Y., Girshick, R., Dollar, P., and Feichtenhofer, C. ´ SAM 2: Segment Anything in Images and Videos, October 2024. URL [http://arxiv.org/abs/2408.](http://arxiv.org/abs/2408.00714) [00714](http://arxiv.org/abs/2408.00714). arXiv:2408.00714 [cs]. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-Resolution Image Synthesis with Latent Diffusion Models, April 2022. URL [http://arxiv.](http://arxiv.org/abs/2112.10752) [org/abs/2112.10752](http://arxiv.org/abs/2112.10752). arXiv:2112.10752 [cs]. Ronneberger, O., Fischer, P., and Brox, T. U-Net: Convolutional Networks for Biomedical Image Segmentation, May 2015. URL [http://arxiv.org/abs/1505.](http://arxiv.org/abs/1505.04597) [04597](http://arxiv.org/abs/1505.04597). arXiv:1505.04597. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., and Batra, D. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, December 2019. URL [http://arxiv.org/](http://arxiv.org/abs/1610.02391) [abs/1610.02391](http://arxiv.org/abs/1610.02391). arXiv:1610.02391. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., and Batra, D. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *International Journal of Computer Vision*, 128(2): 336–359, February 2020. ISSN 0920-5691, 1573-1405. doi: 10.1007/s11263-019-01228-7. URL [http://](http://arxiv.org/abs/1610.02391) [arxiv.org/abs/1610.02391](http://arxiv.org/abs/1610.02391). arXiv:1610.02391 [cs]. Sun, S., Li, R., Torr, P., Gu, X., and Li, S. CLIP as RNN: Segment Countless Visual Concepts without Training Endeavor, May 2024. URL [http://arxiv.org/](http://arxiv.org/abs/2312.07661) [abs/2312.07661](http://arxiv.org/abs/2312.07661). arXiv:2312.07661 [cs]. Tang, R., Liu, L., Pandey, A., Jiang, Z., Yang, G., Kumar, K., Stenetorp, P., Lin, J., and Ture, F. What the DAAM: Interpreting Stable Diffusion Using Cross Attention, December 2022. URL [http://arxiv.org/](http://arxiv.org/abs/2210.04885) [abs/2210.04885](http://arxiv.org/abs/2210.04885). arXiv:2210.04885 [cs]. Tian, J., Aggarwal, L., Colaco, A., Kira, Z., and Gonzalez-Franco, M. Diffuse, Attend, and Segment: Unsupervised Zero-Shot Segmentation using Stable Diffusion, April 2024. URL [http://arxiv.org/abs/2308.](http://arxiv.org/abs/2308.12469) [12469](http://arxiv.org/abs/2308.12469). arXiv:2308.12469 [cs]. Wang, P., Zhang, H., Zhang, Z., Chen, S., Ma, Y., and Qu, Q. Diffusion Models Learn Low-Dimensional Distributions via Subspace Clustering, December 2024. URL [http://arxiv.org/abs/](http://arxiv.org/abs/2409.02426) [2409.02426](http://arxiv.org/abs/2409.02426). arXiv:2409.02426 [cs]. Xu, J., Sun, X., Zhang, Z., Zhao, G., and Lin, J. Understanding and Improving Layer Normalization, November 2019. URL [http://arxiv.org/abs/1911.](http://arxiv.org/abs/1911.07013) [07013](http://arxiv.org/abs/1911.07013). arXiv:1911.07013 [cs]. Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., Yin, D., Zhang, Y., Wang, W., Cheng, Y., Xu, B., Gu, X., Dong, Y., and Tang, J. CogVideoX: Textto-Video Diffusion Models with An Expert Transformer, March 2025. URL [http://arxiv.org/](http://arxiv.org/abs/2408.06072) [abs/2408.06072](http://arxiv.org/abs/2408.06072). arXiv:2408.06072 [cs]. Zhou, J., Wei, C., Wang, H., Shen, W., Xie, C., Yuille, A., and Kong, T. iBOT: Image BERT Pre-Training with Online Tokenizer, January 2022. URL [http://arxiv.](http://arxiv.org/abs/2111.07832) [org/abs/2111.07832](http://arxiv.org/abs/2111.07832). arXiv:2111.07832 [cs].

## A. More In-depth Explanation of Concept Attention

We show pseudo-code depicting the difference between a vanilla multi-modal attention mechanism and a multi-modal attention mechanism with concept attention added to it.

```

(a) Multi-Modal Attention
def multi_modal_attn(img, txt):
    # Compute the keys, queries, and values
    img_k, img_q, img_v = img_projection(img)
    txt_k, txt_q, txt_v = txt_projection(txt)    # Concat the image and text keys, queries, and vals
    img_txt_k = concat([img_k, txt_k])
    img_txt_q = concat([img_q, txt_q])
    img_txt_v = concat([img_v, txt_v])    # Perform self attention on combined sequence
    attn_out = self_attention(img_txt_k, img_txt_q, img_txt_v)
    # Unpack the attention outputs
    img = attn_out[:img.shape[0]], attn_out[img.shape[0]:]    return img, txt(b) Multi-modal Attention with Concept Attention
+ def multi_modal_attn_with_concept_attn(img, txt, concepts):
    # Compute the keys, queries, and values
    img_k, img_q, img_v = img_projection(img)
    txt_k, txt_q, txt_v = txt_projection(txt)
    + concept_k, concept_q, concept_v = txt_projection(concepts)
    # Concat the image and text keys, queries, and vals
    img_txt_k = concat([img_k, txt_k])
    img_txt_q = concat([img_q, txt_q])
    img_txt_v = concat([img_v, txt_v])
    # Perform self attention on combined sequence
    attn_out = self_attention(img_txt_k, img_txt_q, img_txt_v)
    # Unpack the attention outputs
    img, txt = attn_out[:img.shape[0]], attn_out[img.shape[0]:]    + # Concatenate the image and concept keys and values
    + img_concept_k = concat([img_k, concept_k])
    + img_concept_v = concat([img_v, concept_v])
    + # Perform the concept attention
    + concept_attn_map = matmul(concept_q, img_concept_k.T)
    + concept_attn_map = softmax(concept_attn_map, axis=-1) * scale
    + concepts = matmul(concept_attn_map, img_concept_v)
    + return img, txt, concepts

```

**Figure 9. Pseudo-code depicting the (a) multi-modal attention operation used by Flux DiTs and (b) our CONCEPTATTENTION operation.** We leverage the parameters of a multi-modal attention layer to construct a set of contextualized concept embeddings. The concepts query the image tokens (cross-attention) and other concept tokens (self-attention) in an attention operation. The updated concept embeddings are returned in addition to the image and text embeddings.

## B. More Qualitative Results

Here we show a variety of qualitative results for our method that we could not fit into the original paper.

![](_page_13_Figure_3.jpeg)

Figure 10. A qualitative comparison between our method and several others.

![](_page_13_Picture_5.jpeg)

Figure 11. A qualitative comparison between our method and several others.

## C. Concept Attention on Video Generation Models

![](_page_14_Picture_1.jpeg)

![](_page_14_Figure_2.jpeg)

Figure 12. A qualitative comparison between our method and several others.

![](_page_14_Picture_4.jpeg)

![](_page_14_Figure_5.jpeg)

Figure 13. A qualitative comparison between numerous baselines on ImageNet Segmentation Images. The top row shows the soft predictions of each method and the bottom shows the binarized segmentation predictions.

![](_page_14_Figure_7.jpeg)

![](_page_14_Picture_8.jpeg)

Figure 14. A qualitative comparison between numerous baselines on ImageNet Segmentation Images. The top row shows the soft predictions of each method and the bottom shows the binarized segmentation predictions.

![](_page_15_Figure_2.jpeg)

# Correct concept "bike" chosen over similar concept "car" when both are given

Figure 15. The behavior of CONCEPTATTENTION when multiple relevant concepts are present and when no relevant one is. When multiple similar concepts are given, like "car" and "bike", the most similar one will be chosen. However, when no relevant concept is presented, CONCEPTATTENTION will fall back on the most relevant one, in this case "car" for the bike patches.

![](_page_16_Figure_2.jpeg)

Figure 16. CONCEPTATTENTION generalizes seamlessly to video generation MMDiT models like CogVideoX. We apply CON-CEPTATTENTION to a CogVideoX [\(Yang et al.,](#page-11-4) [2025\)](#page-11-4) video generation model. We take several frames from the video and compare the saliency maps generated by CONCEPTATTENTION to the model's internal cross attention maps.

![](_page_17_Figure_2.jpeg)

Figure 17. CONCEPTATTENTION generalizes seamlessly to video generation MMDiT models like CogVideoX. We apply CON-CEPTATTENTION to a CogVideoX [\(Yang et al.,](#page-11-4) [2025\)](#page-11-4) video generation model. We take several frames from the video and compare the saliency maps generated by CONCEPTATTENTION to the model's internal cross attention maps.