# Dsh-Bench: A Difficulty- And Scenario-Aware Benchmark With Hierarchical Subject Taxonomy For Subject-Driven Text-To-Image Generation

Anonymous Author(s)
Affiliation Address email

Subject Difficulty Level Classification Prompt Scenario Classification Variation in subject viewpoint or size A wide-angle shot of a cat basking in the sun, captured with a highangle perspective, surrounded by scattered autumn *leaves.*
Interaction with other entities Background change Please generate an image based on the reference image and the given prompt.

A cat lounging peacefully on a grassy meadow, surrounded by wildflowers under a blue sky.

A cat playing with a curious puppy in a garden, their movements creating dynamic, playful action.

Hard Medium Style change Attribute change Imagination A watercolor painting of a cat sleeping amidst soft pastel tones and diffuse edges that blend seamlessly.

A cat floating weightlessly in space, wearing a tiny astronaut helmet and pawing at sparkling stars nearby.

A cat with sleek black fur. The background is basically the same as the original picture.

Image Quality Please change the image style to oil painting.

31.2/100

(Good)
HPSv2 21.2/100 (Poor)

HPSv2 Evaluation Dimension Prompt Following The cat is on the table with a sofa and a pot of flowers in the *room.*
33.2/100 (Good)
CLIP-T **Score**
15.7/100 (Poor)

CLIP-T Score Subject Preservation The perfume is outdoors. It is dusk now and there is a beautiful sunset glow.

4/5
(Good)
SICS **Score**
1/5
(Poor)
SICS **Score**
Figure 1: **Overview of DSH-Bench**. We curate a diverse dataset of subject images and categorize them into three difficulty levels—easy, **medium**, and **hard**—based on the complexity of preserving subject details. Leveraging GPT-4o's capabilities, we systematically generate contextually appropriate prompts for various scenarios. The generated images are then rigorously evaluated across three key dimensions: Subject Preservation, **Prompt Following**, and **Image Quality**.

## Abstract

1 Significant progress has been achieved in subject-driven text-to-image (T2I) gen2 eration, which aims to synthesize new images depicting target subjects according 3 to user instructions. However, evaluating these models remains a significant chal4 lenge. Existing benchmarks exhibit critical limitations: 1) insufficient diversity 5 and comprehensiveness in subject images, and 2) inadequate granularity in as6 sessing model performance across different subject difficulty levels and prompt 7 scenarios. To address these limitations, we propose DSH-Bench, a comprehensive 8 benchmark that enables systematic multi-perspective analysis of subject-driven T2I 9 models through three principal innovations: 1) a hierarchical taxonomy sampling 10 mechanism ensuring comprehensive subject representation across 58 fine-grained 11 categories, 2) an innovative classification scheme categorizing both subject diffi12 culty level and prompt scenario for granular model capability assessment, and 3) a 13 novel Subject Identity Consistency Score (SICS) metric demonstrating 9.4% higher 14 correlation with human evaluation compared to existing measures in quantifying Submitted to 39th Conference on Neural Information Processing Systems (NeurIPS 2025). Do not distribute.

15 subject preservation. Through empirical evaluation of 15 subject-driven T2I mod16 els, DSH-Bench uncovers previously obscured limitations in current approaches 17 while establishing concrete directions for future research.

## 18 **1 Introduction**

19 Subject-driven text-to-image (T2I) generation aims to generate images conditioned on both textual 20 prompts and specific reference images. It has become feasible due to significant advancements in 21 large-scale T2I generative models [9, 13, 51, 47, 3, 5, 25, 10]. In subject-driven T2I generation, 22 aside from image quality considerations, two other fundamental criteria must be satisfied: Subject 23 Preservation and Prompt Following. Subject Preservation requires that the generated image accurately 24 maintain the details of the reference subject. Prompt Following demands that the generated image 25 consistently reflects the content in the prompt. For example, a user might request an image of "his dog 26 traveling around the world" [50]. In this scenario, the generated image must depict a dog identical to 27 the reference image while illustrating the act of traveling as described in the prompt. 28 Significant progress has been made in subject-driven T2I generation in recent years [50, 14, 28, 58, 30, 29 70, 16, 62, 21, 45]. One approach involves fine-tuning general T2I models to create specialized models 30 that reproduce specific subjects present in the training datasets. Alternatively, encoder-based methods 31 achieve subject preservation by adapting features to incorporate reference subject into a general T2I 32 model. Despite these advancements, challenges remain in comprehensively and effectively evaluating 33 the actual performance of these models. An effective evaluation method should not only provide a 34 comprehensive and unbiased assessment, but also align with human perception to ensure reliable 35 measurement. Furthermore, the evaluation method is expected to provide valuable insights for future 36 research. However, current benchmarks [50, 28, 6, 59, 41] are limited by insufficient diversity and 37 comprehensiveness in subject image collection, which restricts the thoroughness of model evaluation. 38 In addition, they do not facilitate a detailed understanding of subject difficulty and prompt scenarios, 39 thus constraining the depth of insights obtainable from the evaluation. As shown in Figure 2, our 40 analysis of numerous model-generated instances reveals that different subject images and prompts 41 place varying demands on a model's ability. For example, although subject-driven T2I models are 42 capable of effectively preserving the details of relatively simple objects (e.g., a tennis ball), they often 43 struggle to accurately reproduce objects with more intricate features (e.g., a camera). This observation 44 highlights the importance of categorizing the subject difficulty and prompt scenario to better assess 45 model performance. To address the aforementioned requirements, we introduce DSH-Bench, a novel 46 benchmark offers three notable advantages: 47 1. *The diversity of subject images in DSH-Bench is substantially greater* To mitigate evaluation 48 bias caused by low diversity of subject images, we employ a hierarchical taxonomy in image 49 collection. We referenced COCO [32], ImageNet [8], and category lists from Wikipedia [63] in the 50 hierarchical taxonomy construction. As shown in Figure 3(a), the widely used DreamBench [50] 51 includes only 6 categories and 30 subjects. In contrast, our benchmark expands the dataset to 48 52 categories and 459 subjects—representing an increase of 8× and 15×, respectively. Although 53 DreamBench++ [41] offers 150 subjects, its diversity is constrained by its image collection. Notably, 54 33% of our categories are not represented in DreamBench++. Therefore, benefiting from DSH- 55 Bench's greater subject diversity, we enable more comprehensive evaluation of models. 56 2. *An innovative classification scheme for subject difficulty level and prompt scenario* Figure 2 57 shows the model's performance varies significantly with different samples, highlighting the necessity 58 for a classification of both subject image and prompt. Although DreamBench++ [41] categorizes 59 prompts based on their perceived difficulty, the criteria underlying this classification are not clearly 60 defined. Additionally, DreamBench++ [41] does not analyze the difficulty levels associated with 61 different subjects. To address these limitations, we propose an innovative classification scheme. We 62 categorize subjects into three difficulty levels (easy, medium, and hard) according to the difficulty of 63 preserving visual appearance and classify prompts into six scenarios (background change, variation in 64 subject viewpoint or size, interaction with other entities, attribute change, style change, imagination). 65 As a result, our approach enables a more comprehensive and granular analysis of the challenges 66 faced by current models. 67 3. *A human-aligned and more efficient metric for subject preservation* DreamBench++ replaces 68 CLIP [46] and DINO [4] with GPT-4o [37] for evaluation, resulting in improved alignment with

Difficulty Distribution of Subject DreamBench CustomConcept101 DreamBench++ DSH-Bench Easy Medium **Hard**
[**Prompt**] A {} on a table indoors with a sofa and a bouquet of flowers in the background Variation in subject viewpoint or size 0 50 100 150 200 250 Easy Medium Hard Scenario Distribution of Prompt DreamBench CustomConcept101 DreamBench++ DSH-Bench Interaction with other entities Background change Reference subjectA cat lounging peacefully on a grassy meadow, surrounded by wildflowers under a clear blue sky. 

A cat playing with a curious puppy in a garden, their movements creating dynamic, playful action.

A cat with sleek black fur. The background is basically the same as the original picture.

A wide-angle shot of a cat basking in the sun, captured with a highangle perspective, surrounded by scattered autumn leaves.

0 200 400 600 800 1000 Attribute change Imagination Style change A watercolor painting of a cat sleeping amidst soft pastel tones and diffuse edges that blend seamlessly.

A cat floating weightlessly in space, wearing a tiny astronaut helmet and pawing at sparkling stars nearby. 

Background change Variation in subject viewpoint or size Interaction with other entities Attribute change Style change Imagination Subject Category Distribution DreamBench CustomConcept101 DreamBench++ DSH-Bench
(b) t-SNE Visualization of DreamBench++ vs. DSH-Bench (a) Distribution of Subject Images Across Different Categories (Under Photorealistic Category)
0 5 10 15 20 25 30 35 40 Vehicle Musical Instrument Public Facility Food and Beverage Medical Supply Book Furniture Home Appliance Amphibian Building Digital Product Insect Stationery Daily Necessity Plant Jewelry Beauty and Skincare Artwork Clothing Sports Equipment Shoe, Bag and Accessory Toy Mammal Reptile Bird Fish Half-body or Full-body Photo Facial Close-up Artistic Image and Celebrity

69 human evaluation. However, our benchmark reveals that per-model evaluation under this paradigm 70 requires approximately 20,000 API calls to GPT-4o, incurring prohibitive computational costs 71 exceeding $400 for each evaluation. To address the limitation, we introduce **Subject Identity** 72 **Consistency Score** (SICS). Firstly, five annotators label a training dataset containing 5,000 image73 text pairs, focusing on subject preservation evaluation. We then fine-tune Qwen2.5-VL-7B [2] on 74 this dataset. Finally, we use Kendall's τ value to quantify the alignment between model outputs and 75 human evaluation. Experimental results demonstrate that SICS achieves a statistically significant 76 improvement, outperforming GPT-4o by 9.4% in human evaluation correlation metrics. 77 **Takeaways** We present some insightful findings from evaluating fifteen methods: i) Our evaluation 78 reveals that no single method demonstrates consistently robust performance across all categories. 79 Therefore, implementing hierarchical taxonomy sampling of subject images is critical for mitigating 80 potential evaluation biases. ii) All methods exhibit degraded performance on hard subject images. It is 81 crucial to enhance models' ability to encode and reconstruct complex subject details more effectively 82 in future research. iii) The subject-driven T2I model's capability for different prompt scenarios is not 83 robust. Future research on subject-driven T2I generation should focus on optimizing for adaptation to 84 a variety of prompt scenarios. 85 In summary, our contributions are as follows: 1) We employ a hierarchical taxonomy in image 86 collection to ensure both the diversity and comprehensiveness of subject images. 2) We propose an 87 innovative classification scheme to categorize subject difficulty levels and prompt scenarios. This 88 scheme enables us to obtain valuable insights. 3) We propose a human-aligned metric to evaluate 89 subject preservation, which offers greater efficiency compared to GPT-4o-based approaches. We are 90 open-sourcing DSH-Bench, including all subject images, prompts, generated images, related code, 91 and the SICS model.

## 92 **2 Related Work** 93 **2.1 Subject-Driven Text-To-Image Generation**

94 In recent years, subject-driven T2I generation has attracted significant research attention [50, 14, 95 28, 58, 30, 70, 16, 15, 62, 21, 45]. Within the context of diffusion models, optimization-based 96 model [14, 50, 28, 57, 34, 22, 18] enables subject-driven generation by introducing lightweight 97 parameters and performs parameter-efficient fine-tuning for each subject. In contrast, the encoder98 based methods [62, 70, 52, 35, 7, 31, 29, 49, 71, 20, 23, 67, 38, 64, 24, 19] leverage additional 99 image encoders and network layers to encode the reference image of the subject. ELITE [62] uses a 100 learning-based encoder for subject customization, which consists of a global mapping network to 101 encode reference subjects into pseudo words and a local mapping network to maintain subject details. 102 IP-Adapter [70] introduces cross-attention through an additional image encoder to incorporate control 103 signals. Furthermore, SSR-Encoder [73] enhances identity preservation. This strategy facilitates 104 subject-driven generation without necessitating further fine-tuning when introducing new concepts. 105 The Diffusion Transformers (DiT) [40] uses transformer as a denoising network to iteratively refine 106 noisy image tokens, applied in T2I models widely [43, 48]. Based on these foundation models, 107 approaches like OminiControl [55] and UNO [64] explore the inherent image reference capabilities 108 of transformers, suggesting that DiT itself can serve as an image encoder for subject reference.

## 109 **2.2 Subject-Driven T2I Generation Benchmark**

110 Evaluation for subject-driven T2I generation involves a variety of metrics focusing on different 111 aspects. For image quality, several notable studies [68, 27, 65, 1, 69, 60] have conducted Dream112 Sim [12], CLIP-I [46], and DINO Score [4] are commonly adopted to measure perceptual similarity. 113 In terms of semantic consistency, the CLIP score [46] is frequently used. However, in subject-driven 114 image generation tasks, existing perceptual similarity metrics often diverge from human perception. 115 To address this limitation, researchers have proposed new metrics [41] that better align with human 116 judgments. DreamBench [50] is limited in the diversity of subjects and prompt scenarios. Dream117 Bench++ [41] increases to 150 subject images. Moreover, current benchmarks can not provide a 118 systematic categorization of subjects and prompts, making it difficult to derive meaningful insights 119 from the evaluation results.

## 120 **2.3 Subject Preservation Evaluation**

121 Subject preservation evaluation plays a crucial role in the evaluation of subject-driven T2I generation. 122 Learning-based metrics [11, 72, 44] compute the distances between image features extracted by deep 123 neural networks. However, these approaches fall short in capturing the full range of nuances present 124 in human perception. To address this limitation, image embeddings from large vision models like 125 CLIP [46] and DINO [4] have been utilized. The image-retrieval score [33] has been used to assess 126 the visual similarity. To better align with human perceptual judgments, DreamSim [12] has been 127 introduced to assess image similarity with a focus on foreground objects and semantic content.

## 128 **3 Dsh-Bench**

129 This section provides an overview of the primary components of DSH-Bench. Section 3.1 outlines 130 the data construction process. In Section 3.2, we present a concise introduction to the definitions 131 and evaluation methods for three evaluation dimensions. *A detailed explanation is available in the* 132 *supplementary materials.*

## 133 **3.1 Benchmark Dataset Construction** 134 **3.1.1 Subject Image Collection**

135 **Hierarchical Taxonomy Establishment** As shown in Figure 4, we establish a hierarchical taxon136 omy. For the first- and second-level categories, we primarily refer to existing benchmarks from prior 137 studies [50, 28, 41], resulting in two first-level categories and six second-level categories. For the 138 third-level categories, we first reference COCO [32], ImageNet [8] and Wikipedia to compile a list of

Step1. Subject Image Collection **Step2. Subject Image Processing**
Step3. Prompt Generation Candidate category labels mining COCO ImageNet **Wikipedia**
[Task **Description]** In the given image, the primary subject is a cat.

Please generate prompts for this subject, addressing various dimensions as specified by the following requirements. Two prompts are generated for each dimension. (1) **Background change:** Only change the background environment while keeping the subject's default attributes. (2) Variation in subject viewpoint or **size:** The subject's location, perspective, lighting conditions could be adjusted properly. The image maybe shot in wide-angle, telephoto, bird's-eye view, lowangle shot, high-angle shot, eye-level, close-up, medium shot, long shot and so on.

(3) Interaction with **other entities:** Create interactions between the subject and other entities, or add effects like obstruction, reflection.

(4) **Attribute change:** Alter the attributes of the subject, such as color, shape, material, or appearance, and so on.

(5) **Style change:** Modify the style of the scene, including different art movements or artistic forms.

(6) **Imagination:** Imagine some imaginative and unrealistic scenario for this subject.

Multiple subjectsNoisy Background Low image qualityInappropriate proportions Good for generation car cat table book printer printer oven cabinet vase bowl guitar sofa pillow scissors ……
Please integrate common sense and merge the category labels. Obtaining a hierarchical category system.

Filter Images Establish Hierarchical Taxonomy GPT-4o GenerationHuman Inspection Good for generation Human assessmentAutomatic assessment Center the subject by cropping.

……

PhotorealisticRootNon-photorealistic Animal Object Person Animal Object Person Toys Clothing Furniture Vehicle GPT-4o associate based on categories.

Human propose based on categories.

(1) Background **change:** A cat lounging peacefully on a grassy meadow, surrounded by wildflowers under a blue sky.

(2) Variation in subject viewpoint or **size:** A wide-angle shot of a cat basking in the sun, with a high-angle perspective, surrounded by autumn leaves.

ukulele double bed scissors sofa cat …… dinning table cap pen Collect Keywords
……

Classify Subject DifficultyPlease assign appropriate labels to the images based on the complexity of the details contained within the subject. Human Annotators GPT-4o

(3) Interaction with other **entities**:
A cat playing with a curious puppy in a garden, their movements creating dynamic, playful action.

(4) Attribute **change:** A cat with sleek black fur. The background is basically the same as the original picture
… …
Search keywords Collect Images Unsplash Pinterest
(5) Style **change:** A watercolor painting of a cat sleeping amidst soft pastel tones and diffuse edges that blend seamlessly.

(6) **Imagination** : A cat floating weightlessly in space, wearing a tiny astronaut helmet and pawing at sparkling stars nearby.

Easy Medium Hard
…… …… ……

## 152 **3.1.2 Subject Image Processing**

153 **Image Filtering** To filter unsuitable images, human annotators remove images with multiple 154 subjects and noisy backgrounds. We use aesthetic score [69] and SAM [26] to filter images with low 155 image quality and inappropriate proportions of subject regions. The curated images are subsequently 156 cropped to centralize the reference subject. 157 **Subject Difficulty Level Classification** As illustrated in Figure 2, the model's performance varies 158 considerably across different samples. To derive meaningful insights, we classify the subject images 159 according to the difficulty level that the model experiences in preserving details of the reference 160 subject. We define three subject difficulty levels, including (1) **Easy:** Subjects characterized by 161 minimal surface complexity and homogeneous textural properties, exemplified by smooth-surfaced 162 objects such as a ceramic mug with uniform coloration. These instances present negligible challenges 163 for detail preservation due to their structural regularity. (2) **Medium:** Subjects containing discernible 164 high-frequency features while maintaining global structural coherence, such as cylindrical containers 165 with legible typographic elements. These cases require intermediate detail preservation capabilities. 166 (3) **Hard:** Subjects exhibiting non-uniform texture distributions and multi-scale geometric details, 167 typified by objects like book covers containing fine-grained calligraphic elements. Such instances 168 expose model limitations in maintaining structural fidelity and textural granularity under complex 169 topological constraints. We utilize GPT-4o to classify the subject images according to the aforemen170 tioned criteria. Subsequently, all images are reviewed and corrected by human annotators to ensure 171 accuracy and consistency.

139 candidate category labels, then utilize GPT-4o to consolidate them into 58 refined categories. The 140 final hierarchical taxonomy is confirmed and refined through co-authors' discussion. The detailed 141 process and the category contents are provided in Appendix A. 142 **Keyword Collection & Internet Image Collection** In DreamBench++ [41], keywords collection 143 relies on GPT-4o and human input. The approach does not adequately ensure the diversity of the 144 obtained keywords, potentially introducing bias during the image collection process. In contrast, 145 DSH-Bench derives keywords from a hierarchical taxonomy. For each third-level category, we use 146 GPT-4o to generate associated keywords, which are further supplemented by humans. All keywords 147 are then consolidated and deduplicated, resulting in a final set of 400 unique keywords—significantly 148 surpassing DreamBench++'s 300. The specific keywords are provided in the Appendix B. Given a set 149 of selected keywords, we retrieve images from Unsplash [56] and Pinterest [42]. Keywords without 150 suitable images are discarded. We also add some excellent images from previous work. *Each image's* 151 *copyright status has been verified for academic suitability*.

Reference Image BLIPDiffusion UNO RealCustom++ MS-Diffusion Emu2 OminiControl  −Eclipse IP-Adapter SSR-Encoder OmniGen NeTI HiPer Custom Diffusion DreamBooth **Textual** 
Inversion The green velvet sofa situated outdoors in a garden setting surrounded by blooming flowers and lush greenery, maintaining the same sofa attributes Easy A plain beige bowl placed on a wooden table with a scenic countryside view in the background.

A pillow resting against a wooden cabin wall, surrounded by warm, earthy tones.

Medium A book titled 'A BOOK

FULL OF HOPE' laying on a sandy beach with soft waves visible in the background.

A yellow alarm clock photographed from a bird's-eye view, placed on a messy work desk filled with scattered papers, pens, and a coffee cup.

Hard An eye-level shot of the kitten sitting at the edge of a pond, surrounded by autumn leaves and gently rippling water reflecting the vibrant oranges and yellows of the trees.

## 172 **3.1.3 Prompt Generation**

173 Although DreamBench++ categorizes prompts based on their perceived difficulty, it does not provide 174 empirical evidence to substantiate the criterion. To address this limitation, we organize the prompts 175 according to specific application scenarios, dividing them into six categories, including (1) Back176 **ground change (BC):** scenarios involving changes in background elements. (2) **Variation in subject**
177 **viewpoint or size (VS):** scenarios that entail changes in camera angle, which may include variations 178 in subject size, lighting, or shadows. (3) **Interaction with other entities (IE):** scenarios requiring 179 complex interactions with additional entities, potentially resulting in occlusion and necessitating 180 adherence to physical plausibility. (4) **Attribute change (AC):** scenarios involving modifications to 181 certain attributes of the subject, such as color or shape. (5) **Style change (SC):** scenarios involving 182 alterations in the artistic or visual style of the subject. (6) **Imagination (IM):** scenarios where the 183 target image depicts an imagined or fictional scene. We generate two prompts for each scenario. 184 The specific instructions employed for prompt generation are depicted in Figure 4. All prompts are 185 reviewed by two human annotators to ensure they are ethical and free from defects. 186 Finally, we obtain a total of 459 high-quality images and **5,508** prompts. Figure 2 shows the 187 distribution of subject image difficulty levels and prompt scenarios. We visualize the t-SNE of 188 images from our benchmark and DreamBench++ in Figure 3(b). The results clearly indicate that our 189 benchmark achieves superior diversity.

## 190 **3.2 Evaluation Dimension**

191 Previous notable works [50, 14, 28, 58] evaluate the performance of subject-driven T2I models 192 from two perspectives: Subject Preservation and Prompt Following. Mao et al. [36] also uses 193 ImageReward [68] to evaluate image quality. Therefore, DSH-Bench evaluates from the three 194 aforementioned dimensions. 195 **Subject Preservation** DreamBench++ [41] utilizes GPT-4o for evaluation to improve alignment 196 with human assessments. However, the GPT-4o-based method is prohibitively expensive. To 197 address this limitation, we propose a novel metric—**Subject Identity Consistency Score** (SICS). 198 Firstly, we establish a scoring criterion for assessing subject preservation, the details are provided 199 in Appendix E.2. Five annotators label the collected image pairs according to the criterion. During 200 the annotation process, each image pair is not only assigned a score but also accompanied by an 201 explanation. Previous work [61] has indicated that labeled data with explanatory reasoning can help 202 models better understand the underlying logic and reasoning behind the labels. We then perform 203 meticulous fine-tuning of the model using this annotated dataset. Although GPT-4o demonstrates 204 outstanding performance across a wide range of tasks, it has not been specifically optimized for 205 subject preservation evaluation. More details of the SICS metric can be found in Appendix E.2.

206 **Prompt Following** Prompt following primarily evaluates whether a model can generate images 207 that accurately correspond to textual prompts. DreamBench++ has demonstrated that the CLIP-T 208 score [46] is highly consistent with human annotations. Therefore, we also adopt CLIP-T score as 209 the evaluation metric for prompt following. 210 **Image Quality** HPSv2 [65] utilizes professionally annotated data to more accurately reflect human 211 aesthetic preferences for generated images. Previous studies [54] demonstrate that models opti212 mized with HPSv2 achieve superior performance in image quality assessment compared to existing 213 approaches. Therefore, we adopt HPSv2 for image quality evaluation in this work.

## 214 **4 Experiment** 215 **4.1 Experiment Setup**

216 **Implementation Details** We conduct experiments on two mainstream approaches: i) Finetuning217 *based:* 1) Textual Inversion(TI) [15], 2) DreamBooth [50], 3) Custom Diffusion [28], 4) Hiper [17], 218 5) NeTI [1]. *ii) Encoder-based:* 1) BLIP-Diffusion [30], 2) IP-Adapter [70], 3) MS-Diffusion [59], 219 4) Emu2 [53], 5) OminiControl [55], 6) SSR-Encoder [73], 7) RealCustom++ [36], 8) OmniGen [66], 220 9) λ-Eclipse [39], 10) UNO [64]. Our experiments are conducted using the official implementations 221 to guarantee reliability and fairness. More details can be found in Appendix E. 222 **Human Annotation** Five human annotators label the training datasets for SICS. To assess the 223 alignment between various evaluation metrics and human evaluation, the same group of annotators 224 is tasked with labeling the ground truth for images generated by each method on the DSH-Bench 225 dataset. We provide human annotators with sufficient training to ensure they fully understand the 226 subject-driven T2I generation task and can provide unbiased and discriminating scores.

Table 1: The human alignment degree among different evaluation metrics, measured by **Kendall's**
τ **value** and **Spearman correlation coefficient value**. H: Human, G: GPT-4o, D: DINO, Dv2:
DINOv2, CB: CLIP-B, CL: CLIP-L, S: SICS.

Method Kendall↑ **Spearman**↑

H-CB H-CL H-D H-Dv2 H-G H-S H-CB H-CL H-D H-Dv2 H-G H-S

BLIP-Diffusion 0.228 0.176 0.285 0.167 0.354 **0.531** 0.285 0.215 0.350 0.206 0.383 **0.554** IP-Adapter 0.294 0.296 0.258 0.290 0.419 **0.622** 0.364 0.371 0.325 0.364 0.459 **0.657** MS-Diffusion 0.158 0.090 0.116 0.122 0.119 0.178 **0.194** 0.109 0.144 0.156 0.131 0.189 OminiControl 0.375 0.371 0.337 0.348 0.650 **0.713** 0.490 0.486 0.441 0.453 0.729 **0.764** SSR-Encoder 0.264 0.338 0.295 0.348 0.504 **0.664** 0.328 0.421 0.368 0.434 0.549 **0.697** UNO 0.249 0.218 0.299 0.240 0.236 **0.385** 0.340 0.297 0.390 0.312 0.268 **0.426** RealCustom++ 0.181 0.128 0.206 0.241 0.291 **0.464** 0.229 0.162 0.266 0.303 0.325 **0.511**

OmniGen 0.465 0.396 0.344 0.349 0.617 **0.621** 0.579 0.497 0.440 0.456 **0.697** 0.667

λ-Eclipse 0.143 0.233 0.084 0.103 0.325 **0.375** 0.176 0.287 0.103 0.127 0.352 **0.393** Custom Diffusion 0.316 0.336 0.382 0.425 0.487 **0.642** 0.388 0.409 0.470 0.519 0.512 **0.654** DreamBooth 0.639 0.591 0.537 0.429 0.647 **0.692** 0.733 0.721 0.661 0.537 0.705 **0.740** Textual Inversion 0.482 0.459 0.447 0.438 0.541 **0.568** 0.587 0.559 0.545 0.534 0.582 **0.590** HiPer 0.338 0.387 0.351 0.404 0.584 **0.625** 0.417 0.469 0.430 0.496 0.629 **0.655** NeTI 0.469 0.456 0.431 0.417 0.617 **0.728** 0.573 0.561 0.529 0.512 0.682 **0.778** ALL 0.416 0.411 0.350 0.376 0.619 **0.677** 0.529 0.522 0.451 0.483 0.697 **0.734**

## 227 **4.2 Main Results**

228 **SICS Results** Table 1 presents a rigorous study of human alignment using Kendall's τ *value* (KDV) 229 and *Spearman correlation coefficient value* (SCV) (metric selection rationale in Appendix E.2). Our 230 experimental results demonstrate that **SICS achieves superior alignment with human evaluations** 231 **compared to existing methods**, showing consistently higher agreement across both correlation 232 metrics in most experimental settings. Although SICS attains second-highest correlation scores in 233 MS-Diffusion and OmniGen (Bold font: the maximum value in a row. An underline: the second 234 highest value in a row), it significantly outperforms GPT-4o [41] by **9.37%** (KDV) and **5.31%** (SCV). 235 This performance gap strongly suggests SICS's enhanced capability in modeling human evaluation. 236 Notably, GPT-4o demonstrates greater consistency with human evaluation than CLIP and DINO, 237 aligning with DreamBench++ findings. Importantly, our proposed SICS metric surpasses all existing 238 metrics in human judgment consistency. 239 **Quantitative & Qualitative Results** Table 2 shows overall evaluation results. The results show 240 that: **i) DSH-Bench poses more significant challenges than existing benchmarks.** For subject 241 preservation and image quality, the majority of methods consistently yield lower scores on DSH- 242 Bench. The result can be attributed to the hierarchical taxonomy sampling method employed, which 243 allows our dataset to more accurately represent the true data distribution. Moreover, it highlights 244 that benchmarks derived from true distributions present greater challenges. ii) For prompt following, 245 DreamBench yields slightly lower scores than DSH-Bench for certain methods. In DreamBench,

Method Subject Preservation Prompt Following **Image Quality**

DB DB++ HB DB DB++ HB DB DB++ HB

BLIP-Diffusion 0.229 0.216 **0.204** 0.291 0.278 **0.277** 0.267 0.254 **0.223** IP-Adapter 0.230 0.244 **0.229** 0.321 0.318 **0.315** 0.291 0.296 **0.266** MS-Diffusion **0.316** 0.346 0.352 **0.332** 0.339 0.338 0.311 0.314 **0.294** OminiControl 0.279 0.268 0.258 **0.325** 0.337 0.334 0.312 0.308 **0.290** SSR-Encoder 0.231 **0.202 0.202** 0.290 **0.287** 0.295 0.273 0.270 **0.247** UNO **0.409** 0.410 0.409 **0.317** 0.322 0.323 0.304 0.297 **0.278** Emu2 0.360 0.343 0.341 **0.291** 0.309 0.304 0.272 0.278 **0.260** RealCustom++ 0.377 0.380 0.375 **0.325** 0.329 0.332 0.316 0.314 **0.298**

Table 3: **DSH-Bench leaderboard.** The models are ranked by the final score Sh. We only present the top models; the complete ranking can be found in the Appendix D.2.

246 prompts requiring attribute change constitute 22.7%, which is higher than the 16.7% observed in 247 DSH-Bench. Figure 6(b) indicates that all methods exhibit relatively poor average performance on 248 prompts involving attribute change. **iii)** Table 3 shows that there exists a trade-off between subject 249 preservation and prompt following. We plot the Pareto frontier (see in Appendix D.1) using the data 250 presented in Table 3. The primary objective is to identify a Pareto optimal solution that effectively 251 balances the two objectives. *Additional results and discussions can be found in Appendix D.2*. 252 **Leaderboard** In order to assess a model's overall capability, we define the final score as:

$${\mathcal{S}}_{\mathrm{h}}={\frac{3}{{\frac{\lambda}{\mathrm{SP}}}+{\frac{\gamma}{\mathrm{PF}}}+{\frac{\mu}{\mathrm{IQ}}}}}$$

$$(1)$$

| Method            | T2I Model      | Subject   | Prompt   | Image   | Sh↑   |
|-------------------|----------------|-----------|----------|---------|-------|
| Preservation      | Following      | Quality   |          |         |       |
| UNO               | FLUX.1-dev     | 0.409     | 0.323    | 0.278   | 0.252 |
| RealCustom++      | SDXL           | 0.375     | 0.332    | 0.294   | 0.251 |
| MS-Diffusion      | SDXL           | 0.352     | 0.338    | 0.294   | 0.248 |
| Emu2              | SDXL           | 0.341     | 0.304    | 0.260   | 0.228 |
| OminiControl      | FLUX.1-schnell | 0.258     | 0.334    | 0.290   | 0.218 |
| IP-Adapter        | SDXL           | 0.256     | 0.292    | 0.266   | 0.199 |
| λ-Eclipse         | SDXL           | 0.229     | 0.315    | 0.242   | 0.198 |
| OmniGen           | SD v1.5        | 0.202     | 0.295    | 0.265   | 0.183 |
| SSR-Encoder       | SDXL           | 0.188     | 0.322    | 0.247   | 0.181 |
| NeTI              | SD v1.4        | 0.192     | 0.301    | 0.234   | 0.176 |
| BLIP-Diffusion    | SD v1.5        | 0.204     | 0.277    | 0.223   | 0.174 |
| DreamBooth        | SD v1.5        | 0.158     | 0.321    | 0.245   | 0.164 |
| HiPer             | SD v1.4        | 0.135     | 0.318    | 0.247   | 0.151 |
| Textual Inversion | SD v1.5        | 0.109     | 0.299    | 0.225   | 0.129 |
| Custom Diffusion  | SD v1.4        | 0.062     | 0.323    | 0.240   | 0.091 |

253 SP, PF, and IQ represent the scores for Subject Preservation, Prompt Following, and Image Quality, 254 respectively. *λ, γ, µ* are the weights assigned to the importance of each corresponding dimension. 255 In this study, we set λ = 1.5, γ = 1.5, µ = 1, as subject preservation and prompt following are of 256 paramount importance in subject-driven T2I generation. The harmonic mean ensures that a model 257 must perform well across all evaluation dimensions to achieve a high overall assessment. We rank 258 all models based on Sh scores. Table 3 shows the leaderboard. UNO demonstrates relatively strong 259 overall performance. We attribute this improvement to the novel architectural design of UNO and the 260 minimal yet effective modifications implemented in DiT.

## 261 **5 Analysis**

262 In this section, we conduct a detailed analysis of the performance of all methods based on the 263 hierarchical category system, the subject difficulty level classification, and the prompt scenario 264 classification. The results are as follows:
265 **A scientific and comprehensive subject image sampling method is necessary** Figure 6(c) and 266 Figure 6(d) present the performance of various methods in the third-level categories. The results 267 reveal that model robustness varies considerably among categories. For example, performance in

0.12 0.16 0.2 0.24 0.28 0.32 0.36 Vehicle Musical Instrument Public Facility Food and Beverage Medical Supply Book Furniture Home Appliance Amphibian Building Digital Product Insect Stationery Daily Necessity Jewelry Plant Beauty and Skincare Artwork Clothing Sports Equipment Shoe, Bag and Accessory Toy Mammal Reptile Bird Fish Half-body or Full-body Photo Facial Close-up Artistic Image and Celebrity 0 0.1 0.2 0.3 0.4 0.5 Easy Meidum Hard Subject Preservation 0.22 0.26 0.3 0.34 0.38 Easy Meidum Hard Prompt Following 0 0.1 0.2 0.3 0.4 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 Background change Variation in subject viewpoint or size Interaction with other entities Attribute change Style change Imagination 0.3 0.305 0.31 0.315 0.32 0.325 Background change Variation in subject viewpoint or size Interaction with other entities Attribute change Style change Imagination 0.246 0.248 0.25 0.252 0.254 0.256 0.258 0.26 0.262 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 Background change Variation in subject viewpoint or size Interaction with other entities Attribute change Style change Imagination Subject Preservation Prompt Following Image Quality Easy Meidum Hard Image Quality
(a) Subject Difficulty Level 0 0.05 0.1 0.15 0.2 0.25 0.3 0.35
(b) Prompt Scenario BLIP-Diffusion IP-Adapter MS_Diffusion OminiControl SSR-Encoder UNO Emu2 Realcustom Omnigen Custom Diffusion Dreambooth Textual Inversion λ-Eclipse HiPer NeTI
0.18 0.23 0.28 0.33 0.38 Vehicle Musical Instrument Public Facility Food and Beverage Medical Supply Book Furniture Home Appliance Amphibian Building Digital Product Insect Stationery Daily Necessity Jewelry Plant Beauty and Skincare Artwork Clothing Sports Equipment Shoe, Bag and Accessory Toy Mammal Reptile Bird Fish Half-body or Full-body Photo Facial Close-up Artistic Image and Celebrity 0.16 0.2 0.24 0.28 0.32 0.36 Vehicle Musical Instrument Public Facility Food and Beverage Medical Supply Book Furniture Home Appliance Amphibian Building Digital Product Insect Stationery Daily Necessity Jewelry Plant **Beauty and Skincare**
Artwork Clothing Sports Equipment Shoe, Bag and Accessory Toy Mammal Reptile Bird Fish Half-body or Full-body Photo Facial Close-up Artistic Image and Celebrity

-0.35 -0.15 0.05 0.25 0.45 0.65 Vehicle Musical Instrument Public Facility Food and Beverage Medical Supply Book Furniture Home Appliance Amphibian Building Digital Product Insect Stationery Daily Necessity Jewelry Plant Beauty and Skincare Artwork Clothing Sports Equipment Shoe, Bag and Accessory Toy Mammal Reptile Bird Fish Half-body or Full-body Photo Facial Close-up Artistic Image and Celebrity 0.14 0.18 0.22 0.26 0.3 0.34 Vehicle Musical Instrument Public Facility Food and Beverage Medical Supply Book Furniture Home Appliance Amphibian Building Digital Product Insect Stationery Daily Necessity Jewelry Plant Beauty and Skincare Artwork Clothing Sports Equipment Shoe, Bag and Accessory Toy Mammal Reptile Bird Fish Half-body or Full-body Photo Facial Close-up Artistic Image and Celebrity

-0.25 -0.15

-0.05 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 Vehicle Musical Instrument Public Facility Food and Beverage Medical Supply Book Furniture Home Appliance Amphibian Building Digital Product Insect Stationery Daily Necessity Jewelry Plant Beauty and Skincare Artwork Clothing Sports Equipment Shoe, Bag and Accessory Toy Mammal Reptile Bird Fish Half-body or Full-body Photo Facial Close-up Artistic Image and Celebrity Subject Preservation Prompt Following Image Quality Subject Preservation Prompt Following Image Quality
(c) The Third Category (Photorealistic)
(d) The Third Category (Non-photorealistic)
BLIP-Diffusion IP-Adapter MS-Diffusion OminiControl SSR-Encoder UNO Emu2 RealCustom++ OmniGen Custom Diffusion DreamBooth Textual Inversion λ-Eclipse HiPer NeTI

## 296 **6 Conclusion**

268 categories "*artwork*" (both photorealistic and non-photorealistic) is substantially lower. This disparity 269 suggests that the absence of subject images from specific categories can lead to biased evaluation 270 results, highlighting the importance of data diversity. Furthermore, Figure 6 also demonstrates that 271 none of the current models perform well across all categories. We hypothesize that this may be 272 related to the varying complexity of the subjects within different categories. A more detailed analysis 273 of model performance in different categories can be found in Appendix D.1. 274 **Current subject-driven T2I models exhibit performance degradation on hard level subjects** 275 As illustrated in Figure 6(a), the model exhibits substantial variation in performance across different 276 difficulty levels: 1) For subject preservation, there is a pronounced decline in performance as the 277 difficulty of the subject images increases. The model achieves significantly better results on images 278 classified as simple compared to those categorized as hard. This observation supports the validity 279 of our image difficulty classification scheme. 2) For prompt following, Figure 6(a) shows that 280 the capability of the models is minimally influenced by the subject difficulty level. This could be 281 explained by the fact that CLIP-T primarily emphasizes overall semantic information. Consequently, 282 as long as the generated image correctly represents the general category and overall shape, the 283 evaluation score is unlikely to be substantially reduced, even if finer details are not perfectly captured. 284 *Given these findings, it is crucial to enhance models' ability to encode and reconstruct complex* 285 *subject details more effectively in future research endeavors*.

286 **The subject-driven T2I capability for different prompt scenarios is not robust** Figure 6(b)
287 shows the average performance of all models across six prompt scenarios. The results show that: 1) 288 In BC, VS, and IE scenarios, the model's performance consistently declines across all evaluation 289 dimensions. This trend suggests that the difficulty of the scenarios increases progressively from 290 BC to IE. Notably, the finding that the IE scenario is more challenging than the BC scenario aligns 291 with intuitive expectations. 2) For subject preservation, the model's average performance across 292 the AC, SC, and IM prompt scenarios remains relatively low. This could be because the generated 293 subjects undergo partial modifications relative to the original subjects in these three scenarios. *Given* 294 *these findings, more emphasis should be placed on enhancing methods for IE prompt scenario. For* 295 *instance, increasing the volume of training data tailored to these specific contexts.* 297 This paper introduces a novel benchmark called DSH-Bench, designed specifically for subject-driven 298 T2I generation. DSH-Bench presents unique challenges for subject-driven T2I generation models. 299 Key features include: 1) a hierarchical category system in image collection to ensure both the diversity 300 and comprehensiveness of subject images; 2) an innovative classification scheme for categorizing 301 subject difficulty levels and prompt scenarios to obtain valuable insights; and 3) a human-aligned and 302 more efficient metric for subject preservation. The benchmark will be publicly available to support 303 the advancement in the subject-driven T2I generation era.

## 304 **References**

305 [1] Yuval Alaluf, Elad Richardson, Gal Metzer, and Daniel Cohen-Or. A neural space-time 306 representation for text-to-image personalization. ACM Transactions on Graphics (TOG), 42(6): 307 1–10, 2023. 308 [2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, 309 Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang 310 Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen 311 Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 312 2025. URL https://arxiv.org/abs/2502.13923.

313 [3] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, 314 Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, et al. ediff-i: Text-to-image diffusion 315 models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022. 316 [4] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, 317 and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings 318 of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 319 [5] Huiwen Chang, Han Zhang, Jarred Barber, Aaron Maschinot, Jose Lezama, Lu Jiang, Ming320 Hsuan Yang, Kevin Patrick Murphy, William T Freeman, Michael Rubinstein, et al. Muse: 321 Text-to-image generation via masked generative transformers. In International Conference on 322 Machine Learning, pages 4055–4075. PMLR, 2023. 323 [6] Wenhu Chen, Hexiang Hu, Yandong Li, Nataniel Ruiz, Xuhui Jia, Ming-Wei Chang, and 324 William W Cohen. Subject-driven text-to-image generation via apprenticeship learning. 325 Advances in Neural Information Processing Systems, 36:30286–30305, 2023. 326 [7] Zhuowei Chen, Shancheng Fang, Wei Liu, Qian He, Mengqi Huang, Yongdong Zhang, and 327 Zhendong Mao. Dreamidentity: Improved editability for efficient face-identity preserved image 328 generation, 2023. URL https://arxiv.org/abs/2307.00300.

329 [8] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large330 scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern 331 recognition, pages 248–255. Ieee, 2009. 332 [9] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, 333 Xu Zou, Zhou Shao, Hongxia Yang, et al. Cogview: Mastering text-to-image generation via 334 transformers. Advances in neural information processing systems, 34:19822–19835, 2021. 335 [10] Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jian336 jian Sun, Hongyu Zhou, Haoran Wei, et al. Dreamllm: Synergistic multimodal comprehension 337 and creation. In ICLR, 2024. 338 [11] Alexey Dosovitskiy and Thomas Brox. Generating images with perceptual similarity met339 rics based on deep networks. In D. Lee, M. Sugiyama, U. Luxburg, I. Guyon, and R. Gar340 nett, editors, Advances in Neural Information Processing Systems, volume 29. Curran As341 sociates, Inc., 2016. URL https://proceedings.neurips.cc/paper_files/paper/2016/
342 file/371bce7dc83817b7893bcdeed13799b5-Paper.pdf. 343 [12] Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and 344 Phillip Isola. Dreamsim: Learning new dimensions of human visual similarity using synthetic 345 data. In Advances in Neural Information Processing Systems, volume 36, pages 50742–50768, 346 2023.

347 [13] Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taig348 man. Make-a-scene: Scene-based text-to-image generation with human priors. In European 349 Conference on Computer Vision, pages 89–106. Springer, 2022.

350 [14] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, and 351 Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using 352 textual inversion, 2022. URL https://arxiv.org/abs/2208.01618. 353 [15] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, 354 and Daniel Cohen-or. An image is worth one word: Personalizing text-to-image generation 355 using textual inversion. In The Eleventh International Conference on Learning Representations, 356 2023. URL https://openreview.net/forum?id=NAQvF08TcyG. 357 [16] Rinon Gal, Moab Arar, Yuval Atzmon, Amit H Bermano, Gal Chechik, and Daniel Cohen358 Or. Encoder-based domain tuning for fast personalization of text-to-image models. ACM
359 Transactions on Graphics (TOG), 42(4):1–13, 2023. 360 [17] Inhwa Han, Serin Yang, Taesung Kwon, and Jong Chul Ye. Highly personalized text embedding 361 for image manipulation by stable diffusion. arXiv preprint arXiv:2303.08767, 2023. 362 [18] Shaozhe Hao, Kai Han, Shihao Zhao, and Kwan-Yee K Wong. Vico: Plug-and-play visual 363 condition for personalized text-to-image generation. arXiv preprint arXiv:2306.00971, 2023. 364 [19] Junjie He, Yuxiang Tuo, Binghui Chen, Chongyang Zhong, Yifeng Geng, and Liefeng Bo. Anys365 tory: Towards unified single and multiple subject personalization in text-to-image generation, 366 2025. URL https://arxiv.org/abs/2501.09503.

367 [20] Hexiang Hu, Kelvin C. K. Chan, Yu-Chuan Su, Wenhu Chen, Yandong Li, Kihyuk Sohn, Yang 368 Zhao, Xue Ben, Boqing Gong, William Cohen, Ming-Wei Chang, and Xuhui Jia. Instruct369 imagen: Image generation with multi-modal instruction, 2024. URL https://arxiv.org/ 370 abs/2401.01952.

371 [21] Hexiang Hu, Kelvin CK Chan, Yu-Chuan Su, Wenhu Chen, Yandong Li, Kihyuk Sohn, Yang 372 Zhao, Xue Ben, Boqing Gong, William Cohen, et al. Instruct-imagen: Image generation with 373 multi-modal instruction. In Proceedings of the IEEE/CVF conference on computer vision and 374 pattern recognition, pages 4754–4763, 2024. 375 [22] Miao Hua, Jiawei Liu, Fei Ding, Wei Liu, Jie Wu, and Qian He. Dreamtuner: Single image is 376 enough for subject-driven generation, 2023. URL https://arxiv.org/abs/2312.13691.

377 [23] Linyan Huang, Haonan Lin, Yanning Zhou, and Kaiwen Xiao. Flexip: Dynamic control of 378 preservation and personality for customized image generation, 2025. URL https://arxiv. 379 org/abs/2504.07405. 380 [24] Zhipeng Huang, Shaobin Zhuang, Canmiao Fu, Binxin Yang, Ying Zhang, Chong Sun, Zhizheng 381 Zhang, Yali Wang, Chen Li, and Zheng-Jun Zha. Wegen: A unified model for interactive 382 multimodal generation as we chat, 2025. URL https://arxiv.org/abs/2503.01115. 383 [25] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and 384 Taesung Park. Scaling up gans for text-to-image synthesis. In Proceedings of the IEEE/CVF 385 conference on computer vision and pattern recognition, pages 10124–10134, 2023. 386 [26] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, 387 Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In 388 Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 389 2023.

390 [27] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy.

391 Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in 392 Neural Information Processing Systems, 36:36652–36663, 2023. 393 [28] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi394 concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF conference 395 on computer vision and pattern recognition, pages 1931–1941, 2023. 396 [29] Duong H. Le, Tuan Pham, Sangho Lee, Christopher Clark, Aniruddha Kembhavi, Stephan 397 Mandt, Ranjay Krishna, and Jiasen Lu. One diffusion to generate them all, 2024. URL 398 https://arxiv.org/abs/2411.16318. 399 [30] Dongxu Li, Junnan Li, and Steven Hoi. Blip-diffusion: Pre-trained subject representation for 400 controllable text-to-image generation and editing. Advances in Neural Information Processing 401 Systems, 36:30146–30166, 2023. 402 [31] Zhen Li, Mingdeng Cao, Xintao Wang, Zhongang Qi, Ming-Ming Cheng, and Ying Shan. 403 Photomaker: Customizing realistic human photos via stacked id embedding, 2023. URL 404 https://arxiv.org/abs/2312.04461. 405 [32] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr 406 Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer 407 vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, 408 proceedings, part v 13, pages 740–755. Springer, 2014. 409 [33] Zheyuan Liu, Cristian Rodriguez-Opazo, Damien Teney, and Stephen Gould. Image retrieval on 410 real-life images with pre-trained vision-and-language models. In Proceedings of the IEEE/CVF 411 International Conference on Computer Vision, pages 2125–2134, 2021. 412 [34] Zhiheng Liu, Yifei Zhang, Yujun Shen, Kecheng Zheng, Kai Zhu, Ruili Feng, Yu Liu, Deli 413 Zhao, Jingren Zhou, and Yang Cao. Cones 2: Customizable image synthesis with multiple 414 subjects, 2023. URL https://arxiv.org/abs/2305.19327.

415 [35] Jian Ma, Junhao Liang, Chen Chen, and Haonan Lu. Subject-diffusion:open domain personal416 ized text-to-image generation without test-time fine-tuning, 2024. URL https://arxiv.org/ 417 abs/2307.11410. 418 [36] Zhendong Mao, Mengqi Huang, Fei Ding, Mingcong Liu, Qian He, and Yongdong Zhang. 419 Realcustom++: Representing images as real-word for real-time customization. arXiv preprint 420 arXiv:2408.09744, 2024. 421 [37] OpenAI. Introducing gpt-4o and more tools to chatgpt free users, 2024. URL https://openai. 422 com/index/gpt-4o-and-more-tools-to-chatgpt-free/. Accessed: 2024-06-15. 423 [38] Or Patashnik, Rinon Gal, Daniil Ostashev, Sergey Tulyakov, Kfir Aberman, and Daniel Cohen424 Or. Nested attention: Semantic-aware attention values for concept personalization, 2025. URL 425 https://arxiv.org/abs/2501.01407. 426 [39] Maitreya Patel, Sangmin Jung, Chitta Baral, and Yezhou Yang. λ-eclipse: Multi-concept 427 personalized text-to-image diffusion models by leveraging clip latent space. arXiv preprint 428 arXiv:2402.05195, 2024. 429 [40] William Peebles and Saining Xie. Scalable diffusion models with transformers, 2023. URL
430 https://arxiv.org/abs/2212.09748. 431 [41] Yuang Peng, Yuxin Cui, Haomiao Tang, Zekun Qi, Runpei Dong, Jing Bai, Chunrui Han, 432 Zheng Ge, Xiangyu Zhang, and Shu-Tao Xia. Dreambench++: A human-aligned benchmark 433 for personalized image generation. In The Thirteenth International Conference on Learning 434 Representations, 2025. URL https://openreview.net/forum?id=4GSOESJrk6. 435 [42] pin. https://www.pinterest.com/. https://www.pinterest.com/. 436 [43] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe 437 Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image 438 synthesis, 2023. URL https://arxiv.org/abs/2307.01952. 439 [44] Ekta Prashnani, Hong Cai, Yasamin Mostofi, and Pradeep Sen. Pieapp: Perceptual image-error 440 assessment through pairwise preference. In 2018 IEEE/CVF Conference on Computer Vision 441 and Pattern Recognition, pages 1808–1817, 2018. doi: 10.1109/CVPR.2018.00194. 442 [45] Zeju Qiu, Weiyang Liu, Haiwen Feng, Yuxuan Xue, Yao Feng, Zhen Liu, Dan Zhang, Adrian 443 Weller, and Bernhard Schölkopf. Controlling text-to-image diffusion by orthogonal finetuning. 444 Advances in Neural Information Processing Systems, 36:79320–79362, 2023. 445 [46] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, 446 Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual 447 models from natural language supervision. In International conference on machine learning, 448 pages 8748–8763. PmLR, 2021. 449 [47] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High450 resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF 451 conference on computer vision and pattern recognition, pages 10684–10695, 2022. 452 [48] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High453 resolution image synthesis with latent diffusion models, 2022. URL https://arxiv.org/abs/ 454 2112.10752. 455 [49] Ciara Rowles, Shimon Vainer, Dante De Nigris, Slava Elizarov, Konstantin Kutsy, and Simon 456 Donné. Ipadapter-instruct: Resolving ambiguity in image-based conditioning using instruct 457 prompts, 2024. URL https://arxiv.org/abs/2408.03209. 458 [50] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. 459 Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In 460 Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 461 22500–22510, 2023.

462 [51] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, 463 Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. 464 Photorealistic text-to-image diffusion models with deep language understanding. Advances in 465 neural information processing systems, 35:36479–36494, 2022. 466 [52] Jing Shi, Wei Xiong, Zhe Lin, and Hyun Joon Jung. Instantbooth: Personalized text-to-image 467 generation without test-time finetuning, 2023. URL https://arxiv.org/abs/2304.03411. 468 [53] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming 469 Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in470 context learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern 471 Recognition, pages 14398–14409, 2024. 472 [54] Shangkun Sun, Bowen Qu, Xiaoyu Liang, Songlin Fan, and Wei Gao. Ie-bench: Advancing 473 the measurement of text-driven image editing for human perception alignment. arXiv preprint 474 arXiv:2501.09927, 2025. 475 [55] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol:
476 Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098, 477 2024.

478 [56] uns. https://unsplash.com/. https://unsplash.com/.

479 [57] Andrey Voynov, Qinghao Chu, Daniel Cohen-Or, and Kfir Aberman. P+: Extended textual 480 conditioning in text-to-image generation, 2023. URL https://arxiv.org/abs/2303.09522. 481 [58] Haofan Wang, Matteo Spinelli, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. In482 stantstyle: Free lunch towards style-preserving in text-to-image generation. arXiv preprint 483 arXiv:2404.02733, 2024. 484 [59] Xierui Wang, Siming Fu, Qihan Huang, Wanggui He, and Hao Jiang. Ms-diffusion: Multi485 subject zero-shot image personalization with layout guidance. arXiv preprint arXiv:2406.07209, 486 2024. 487 [60] Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for 488 multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025. 489 [61] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, 490 Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language 491 models. In Advances in Neural Information Processing Systems (NeurIPS), 2022. 492 [62] Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, and Wangmeng Zuo. Elite:
493 Encoding visual concepts into textual embeddings for customized text-to-image generation.

494 In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15943–
495 15953, 2023. 496 [63] Wikipedia. https://en.wikipedia.org/. https://en.wikipedia.org/. [Online; accessed 11497 May-2025]. 498 [64] Shaojin Wu, Mengqi Huang, Wenxu Wu, Yufeng Cheng, Fei Ding, and Qian He. Less-to499 more generalization: Unlocking more controllability by in-context generation. arXiv preprint 500 arXiv:2504.02160, 2025. 510 [68] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao 511 Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. 512 Advances in Neural Information Processing Systems, 36:15903–15935, 2023. 513 [69] Jiazheng Xu, Yu Huang, Jiale Cheng, Yuanming Yang, Jiajun Xu, Yuan Wang, Wenbo Duan, 514 Shen Yang, Qunlin Jin, Shurun Li, Jiayan Teng, Zhuoyi Yang, Wendi Zheng, Xiao Liu, Ming 515 Ding, Xiaohan Zhang, Xiaotao Gu, Shiyu Huang, Minlie Huang, Jie Tang, and Yuxiao Dong. 516 Visionreward: Fine-grained multi-dimensional human preference learning for image and video 517 generation, 2024. URL https://arxiv.org/abs/2412.21059. 518 [70] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image 519 prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023. 520 [71] Yu Zeng, Vishal M. Patel, Haochen Wang, Xun Huang, Ting-Chun Wang, Ming-Yu Liu, and 521 Yogesh Balaji. Jedi: Joint-image diffusion models for finetuning-free personalized text-to-image 522 generation, 2024. URL https://arxiv.org/abs/2407.06187.

523 [72] Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The 524 unreasonable effectiveness of deep features as a perceptual metric. In 2018 IEEE/CVF 525 Conference on Computer Vision and Pattern Recognition, pages 586–595, 2018. doi:
526 10.1109/CVPR.2018.00068. 527 [73] Yuxuan Zhang, Yiren Song, Jiaming Liu, Rui Wang, Jinpeng Yu, Hao Tang, Huaxia Li, Xu Tang, 528 Yao Hu, Han Pan, et al. Ssr-encoder: Encoding selective subject representation for subject529 driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and 530 Pattern Recognition, pages 8069–8078, 2024.

504 [66] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan 505 Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. arXiv 506 preprint arXiv:2409.11340, 2024. 507 [67] Zhexiao Xiong, Wei Xiong, Jing Shi, He Zhang, Yizhi Song, and Nathan Jacobs. Grounding508 booth: Grounding text-to-image customization, 2025. URL https://arxiv.org/abs/2409. 509 08520. 501 [65] Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score: 502 Better aligning text-to-image models with human preference. In Proceedings of the IEEE/CVF 503 International Conference on Computer Vision, pages 2096–2105, 2023.

## 531 **A Details Of Hierarchical Category Establishing**

532 **The First-level Category** We observed the composition of existing benchmark data. From a more 533 abstract and higher-level perspective perspective, images in these datasets could be categorized 534 into two types: photorealistic and non-photorealistic. Theoretically, the specific image categories 535 represented within these two types can be identical. To maintain consistency with previous work and 536 to ensure comprehensive data sampling, we designated photorealistic and non-photorealistic as the 537 first-level categories. Furthermore, we ensure that the specific subcategories under both photorealistic 538 and non-photorealistic types are fully aligned. 539 **The Second-level Category** We examined both the DreamBench and DreamBench++ datasets. In 540 DreamBench, the dataset is divided into two categories: living subjects and objects. DreamBench++ 541 further refines this categorization by introducing three categories: living subjects, objects, and style. 542 We construct our secondary subcategories based on them. We define our secondary categories as 543 objects, humans, and animals. Specifically, we subdivide the "living subjects" category into "humans" 544 and "animals," as humans exhibit significantly different visual characteristics compared to animals. 545 For the human category, we place particular emphasis on the accuracy of facial feature reconstruction, 546 acknowledging the existence of dedicated research domains focused on facial preservation. In 547 contrast, animals generally display greater variability in appearance than human faces. In comparison 548 to DreamBench++, we exclude the "style" category. This decision is motivated by the focus of our 549 task on subject-driven T2I generation, where "style" does not constitute a tangible entity. Moreover, 550 including the style category would complicate the calculation of subject consistency, whereas our 551 work is primarily concerned with the customization of entities. 552 **The Third-level Category** For the third-level categories, our objective was to strike a balance 553 between granularity and generality. Categories that are too broad may result in insufficient keyword 554 retrieval, potentially introducing bias into the final image sampling. Conversely, overly fine-grained 555 categories may hinder subsequent experimental analysis by diluting meaningful insights. To address 556 this, we consulted existing large-scale datasets such as COCO and ImageNet, as well as Wikipedia, 557 to compile a list of candidate category labels. The specific labels are listed in Table 4. This 558 comprehensive set of labels ensured broad coverage. However, many of these labels were excessively 559 detailed, so we employ GPT-4o to merge them, followed by manual review to ensure the rationality 560 and coherence of the final categories. The correspondence between the third-level categories and the 561 candidate category labels is presented in Table 4. For the "human" category, we introduced a specific 562 distinction by dividing it into "celebrities & artistic figures," "facial close-ups," and "half-body or 563 full-body photo". We observed that models tend to perform significantly better on celebrities, which 564 we hypothesize is due to the inclusion of celebrity data in the training sets of text-to-image foundation 565 models. Table 14 provides empirical support for our hypothesis to some extent. The rationale for 566 distinguishing between facial close-ups and non-facial close-ups is that the former focuses exclusively 567 on the facial details of the individual in the reference image, whereas the latter also requires attention 568 to the body details. 569 Through the aforementioned steps, we constructed a hierarchical category system. The resulting 570 category hierarchy is presented in Figure 7.

ROOT
Photorealistic Photorealistic Animal Object Human Human Object Animal MA RE BI FI HA FA AR

MS BO FUR HA
AM BU DP IN ST DN PL JE

VE MI PUF FB
HA MA RE BI FI FA AR
MS BO FUR HA
AM BU DP IN ST DN PL JE
VE MI PUF FB
BS AR CL SE SBA TO
BS AR CL SE SBA TO
VE: Vehicle MI: Musical Instrument PUF: Public Facility FB: Food and Beverage SBA: Shoe, Bag, and Accessory TO: Toy MA: Mammal RE: Reptile AR: Artistic and Celebrity MS: Medical Supply BO: Book FUR: Furniture HA: Home Appliance AM: Amphibian BU: Building DP: Digital Product IN: Insect ST: Stationery DN: Daily Necessity PL: Plant JE: Jewelry BS: Beauty and Skincare AR: Artwork CL: Clothing SE: Sports Equipment BI: Bird FI: Fish HA: Half or Full Body FA: Facial Close-up

| Table 4: The correspondence between the third-level categories and the candidate category labels Candidate Category Labels The Third-level Category reptile lizard dinosaur turtle crocodile chameleon gecko Reptile fly firefly ant butterfly ladybug locust dragonfly Insect amphibian frog bullfrog toad salamander Amphibian fish goldfish seahorse shark tilapia Fish bird chicken duck owl swan goose rooster Bird hen turkey swallow crow pigeon mammal cat dog horse sheep cow elephant Mammal bear squirrel giraffe lion monkey tiger bunny goat pig kangaroo rhinoceros deer hippo platypus whale aardvark rabbit zebra mouse street fountain fire hydrant traffic light sign parking meter goal net Public Facility field goal post soccer net basketball court bus stop sign furniture dining table sofa chair couch bed desk table coffee table side table bench cabinet mirror carpet Furniture window door chandelier table lamp gate flower potted plant tree sunflower cactus lavender Plant cookie milk pancake pasta grape cereal bean pineapple carrot broccoli banana orange strawberry apple bread sandwich cake pizza soup meat pumpkin Food and Beverage cheese cupcake donut hot dog bacon egg tomato dryer fridge refrigerator microwave oven toaster washer Home Appliance blender hair drier fan (ceil/floor) printer fax machine copier necklace bracelet ring pendant brooch anklet Jewelry wheelchair gauze crutch stethoscope syringe Medical Supply pants jacket long sleeve shirt short sleeve shirt pajamas underpants shirt Clothing shorts scarf tie super hero costume sock book magazine textbook dictionary biography Book bat skis snowboard tennis racket basketball hoop baseball glove soccer ball Sports Equipment sports ball basketball football tennis net hoop flip flop handbag glove shoe backpack Shoe, Bag, and Accessory pen pencil fax machine stapler Stationery vehicle car van truck bus train boat sailboat raft airplane helicopter hot air balloon rocket bicycle Vehicle unicycle motorcycle motorbike skateboard house building roof bridge church Building picture frame movie (disc) playing cards table cloth Artwork musical instrument guitar drum flute violin Musical Instrument telephone laptop computer tablet ipad iphone cell phone remote mouse keyboard printer desktop copier radio Digital Product kite toy cars toy legos robot doll hair brush toner blush serum emulsion sunscreen Beauty and Skincare bottle plate cup bowl teapot fork knife Daily Necessity spoon clock toothbrush vase towel candle balloon box chopping board ladder basket pillow power outlet light switch person Person   |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## 571 **B Details Of Keywords Collection**

572 The keywords utilized during the image collection process are presented in Table 5. During the 573 keyword collection process, we utilized the following prompt for GPT-4o:
574 575 *"You are a researcher with extensive knowledge of various real-world entity classifications.* 576 *Given a specific category, please generate detailed, non-redundant instances relevant to this category.* 577 *The category is {}.* 578 *The corresponding instances are as follows:"*

## 579 **C Details Of Prompt Generation**

580 The specific instructions used in prompt generation are detailed in Figure 4. During the actual 581 generation process, some of the prompts produced by GPT-4o did not meet the required criteria.

| The Third-level         | Keywords              |                     |                   |                  |                        |                 |                    |
|-------------------------|-----------------------|---------------------|-------------------|------------------|------------------------|-----------------|--------------------|
| Category Vehicle        | van                   | steam locomotive    | car               | airplane         | UFO                    | hot air balloon | oil tanker         |
| pickup truck            | bicycle               | boat                | taxi              | motorcycle       | subway                 |                 |                    |
| Musical Instrument      | guitar pick           | electronic drum     | digital piano     | guitar           | snare drum             | flute           | african drum       |
| suona                   | saxophone             | harmonica           | cello             | violin           | pipa                   | erhu            |                    |
| Public Facility         | fire extinguisher     | traffic sign        | street lamp       | street           | station                |                 |                    |
| edible oil              | instant noodles       | water               | pastries          | coffee           | biscuits               | edible salt     |                    |
| pineapple               | milk                  | orange              | avocado           | can              | juice                  | milk powder     |                    |
| Food and Beverage       | apple                 | donut               | durian            | sports drink     | canned health products | egg             | rice               |
| vegetable               | chicken               | noodles             | hamburger         | salad            | chocolate              | yogurt          |                    |
| Medical Supply          | band-aid              | medicine            | wheelchair        | disinfectant     | first aid kit          | medication      | medicine bottle    |
| blood glucose meter     | crutch                | stethoscope         | syringe           |                  |                        |                 |                    |
| Book                    | yearbook              | almanac             | workbook          | comic            | encyclopedia           | atlas           | pamphlet           |
| book                    | notebook              | magazine            | dictionary        |                  |                        |                 |                    |
| Furniture               | shelf                 | makeup mirror       | stool             | bathroom cabinet | cabinet                | bean bag chair  | children's chair   |
| barber chair            | office chair          | bathroom mirror     | chair             | sofa             | dining table           | bed             |                    |
| ottoman                 | bookcase              | wardrobe            | nightstand        | dresser          |                        |                 |                    |
| Home Appliance          | beauty device         | kettle              | speaker           | massage chair    | vacuum cleaner         | rice cooker     | robot vacuum       |
| microphone              | refrigerator          | hair dryer          | humidifier        | washing machine  | microwave oven         | curling iron    |                    |
| television              | oven                  | juicer              | dishwasher        |                  |                        |                 |                    |
| Amphibian               | newt                  | olm                 | bullfrog          | wood frog        | Surinam toad           | alpine newt     | glass frog         |
| frog                    | toad                  | caecilian           | salamander        |                  |                        |                 |                    |
| Building                | house                 | apartment building  | duplex house      | church           | temple of heaven       | castle          | golden gate bridge |
| hut                     | leaning tower of pisa | pyramid             | statue of liberty | eiffel tower     |                        |                 |                    |
| Digital Product         | smart robot           | headphones          | e-book reader     | desktop computer | roll of film           | router          | tablet             |
| printer                 | camcorder             | camera              | smart camera      | laptop           | mobile phone           | walkie-talkie   |                    |
| smartwatch              | vintage camera        | monitor             | drone             | projector        | fitness tracker        |                 |                    |
| Insect                  | shrimp                | crab                | ant               | grasshopper      | butterfly              |                 |                    |
| Stationery              | glue stick            | globe               | calculator        | floppy disk      | tape measure           | scissors        | compass            |
| stapler                 | crayon                | ballpoint pen       | eraser            |                  |                        |                 |                    |
| hammer                  | candle                | mug                 | teapot            | berry bowl       | curtain                | pillow          |                    |
| birdcage                | alarm clock           | spoon               | bowl              | toothbrush       | shower gel             | clock           |                    |
| Daily Necessity         | glass jar             | vase                | hanger            | soap dish        | frying pan             | baby bottle     | kitchen knife      |
| electric saw            | mop                   | broom               | comb              |                  |                        |                 |                    |
| Plant                   | cactus                | coconut tree        | tree              | potted plant     | peony                  | willow tree     | maple leaf         |
| mint                    | rose                  | sunflower           | tulip             | cactus           | lavender               |                 |                    |
| Jewelry                 | earrings              | ring                | crystal           | bracelet         | watch                  | hair accessory  | beaded bracelet    |
| tiara                   | crown                 | stud                | chain             | gemstone         | choker                 | hairpin         |                    |
| gold bar                | necklace              | pendant             | brooch            | anklet           | locket                 |                 |                    |
| Beauty and Skincare     | perfume               | makeup brush        | lotion            | sunscreen spray  | face cream             | nail polish     | toner              |
| blush                   | eye shadow            | facial serum        | emulsion          | serum            | mascara                | lipstick        |                    |
| Artwork                 | bouquet of flowers    | clay sculpture      | wood carving      | classical bust   | stone carving          | catstatue       | mugskulls          |
| sculpture               | ceramic craft         | mural               | relief            |                  |                        |                 |                    |
| Clothing                | dress                 | baby clothes        | clothing          | jeans            | sweatshirt             | T-shirt         | socks              |
| pants                   | shirt                 | down jacket         | coat              | skirt            | shorts                 | vest            |                    |
| Sports Equipment        | tennis                | ball                | tent              | trekking poles   | yoga mat               | billiard        | badminton          |
| adjustable bench        | knee pad              | backpack            | soccer            | sleeping bag     | baseball               | flamingo float  |                    |
| treadmill               | skateboard            | barbell             | dumbbell          |                  |                        |                 |                    |
| Shoe, Bag and Accessory | suitcase              | slippers            | sunglasses        | canvas shoes     | high-top shoes         | sports shoes    | scarf              |
| glasses                 | sandals               | shoes               | luggage purse     | fancy boot       | belt                   | sneaker         |                    |
| hat                     | backpack              | cap                 | tie               | handbag          | sandals                |                 |                    |
| Toy                     | actionfigure          | monster toy         | car               | egg              | duck toy               | teddy bear      | balloon            |
| robot                   | motorbike toy         | magic cube          | poop emoji        | sloth plushie    | bear plushie           | red cartoon     |                    |
| minion                  | smart robot           | robot toy           | toy               | wolf plushie     | doll                   | Eevee figurine  |                    |
| Mammal                  | rabbit                | fox                 | wolf              | Siamese cat      | polar bear             | cat             | deer               |
| panda                   | elephant              | llama               | tiger             | dog              | raccoon                | lion            |                    |
| alpaca                  | puppy                 | monkey              | kitten            | dolphin          | French bulldog         |                 |                    |
| Reptile                 | cobra                 | gecko               | rattlesnake       | crocodile        | chameleon              | alligator       | iguana             |
| turtle                  | sea turtle            | soft-shelled turtle | snake             | lizard           |                        |                 |                    |
| Bird                    | heron                 | pigeon              | toucan            | parrot           | stork                  | flamingo        | penguin            |
| woodpecker              | nightingale           | duck                | turkey            | chicken          | crow                   | eagle           |                    |
| peacock                 | swallow               | owl                 | kingfisher        | hawk             | dove                   | anchovy         |                    |
| bird                    | canary                | sparrow             | rooster           |                  |                        |                 |                    |
| Fish                    | shark                 | tropical fish       | jellyfish         | goldfish         | perch                  | eel             | monkfish           |
| skate                   | swordfish             | herring             | sardine           | carp             | salmon                 | tuna            |                    |
| Person                  | person                |                     |                   |                  |                        |                 |                    |

582 Therefore, we instructed GPT-4o to generate multiple prompts for each image, and then manually 583 selected those that best matched the intended scenarios. Figure 13 presents the results generated by 584 different methods in this study, along with their corresponding prompts.

## 585 **D Additional Discussions And Details Of Model Performance** 586 **D.1 Additional Discussions**

587 **Analysis of The First-Level Category** The primary categories are divided into photorealistic and 588 non-photorealistic. Table 6 and Figure 8 present the performance of different methods on these 589 two categories across three evaluation dimensions. The results show that: *(1) Subject Preservation:* 590 Almost all methods perform better on photorealistic categories than on non-photorealistic ones. 591 We speculate that this is because, when referencing subjects from non-photorealistic categories, 592 these methods tend to generate photorealistic images based on the prompt, which results in lower 593 subject consistency. *(2) Prompt Following:* The performance gap between photorealistic and non594 photorealistic categories is relatively small. This can be attributed to the fact that CLIP-T focuses 595 primarily on the semantic information of the image. As long as the generated subject matches the 596 category and general appearance described in the prompt, the CLIP-T score will not be significantly 597 reduced. *(3) Image quality:* There is little difference in performance between photorealistic and 598 non-photorealistic categories. This indicates that the distinction between these two categories does 599 not affect the quality of image generation, and the HPSv2 metric does not show a preference for 600 either category. 601 **Analysis of The Second-Level Category** The secondary categories under both the realistic and 602 non-realistic primary categories are further subdivided into objects, humans, and animals. Table 7 603 and Figure 9 present the performance of various methods across these three dimensions for both 604 realistic and non-realistic categories. The results demonstrate that, irrespective of whether the primary 605 category is realistic or non-realistic, the scores for the subject preservation dimension are consistently 606 lower for the human category across nearly all models. As detailed in Table 8, this phenomenon can 607 be attributed to the distribution of difficulty levels within the human category, where the proportions 608 of simple, medium, and hard cases are 1.96%, 50.98%, and 47.06%, respectively. In contrast, the 609 object and animal categories exhibit a higher proportion of subjects at the simple difficulty level and a 610 lower proportion at the hard difficulty level, which likely contributes to their relatively higher subject 611 preservation scores. 612 **Implications for Technical Approaches** (1) Figure 10 shows that, as base models and model 613 architectures are updated, the performance boundary of these models consistently expands outward. 614 Table 9 presents all the base models used by each method. It can be observed that the top-performing 615 methods consistently employ relatively recent text-to-image base models. For instance, UNO utilizes 616 FLUX as its foundational model. This observation suggests that the adoption of advanced text-to617 image base models is a critical factor in enhancing performance on subject-driven T2I tasks. (2) 618 Historically, fine-tuning methods have generally outperformed encoder-based approaches in terms 619 of subject preservation. This advantage is attributed to their ability to better retain the original 620 text-image conditional distribution by fine-tuning on images of the specified subject. In contrast, 621 encoder-based methods often encounter interference during feature injection, which can hinder precise 622 prompt alignment. However, with the development of more advanced encoding techniques, the 623 adoption of larger and more powerful base models, and the availability of extensive training datasets, 624 encoder-based methods have demonstrated significantly improved performance. From an application 625 standpoint, fine-tuning methods require substantial computational resources for optimization and often 626 exhibit limited generalization capabilities. In contrast, encoder-based methods are less constrained 627 by these limitations, making them more practical for future applications. Nevertheless, our analysis 628 indicates that current encoder-based methods still face challenges in accurately reconstructing subjects 629 with high-frequency details in images. This limitation may stem from the characteristics of commonly 630 used image encoders, such as CLIP, which tend to prioritize semantic information over fine-grained 631 details. Consequently, future research should focus on enhancing the restoration of challenging 632 subject details.

## 633 **D.2 Details Of Model Performance**

634 In this section, we present the detailed evaluation results for each metric across all models. To 635 comprehensively evaluate the effectiveness of different metrics for assessing subject consistency, we 636 calculated multiple metrics for each method. The detailed results are presented in Table 10, 11, 12.

637 In section 5, we present the performance of all methods across images with different difficulty

0.000 0.050 0.100 0.150 0.200 0.250 0.300 0.350 0.400 0.450 0.500Model Performance on The First-Level Categories Photorealistic Non-photorealistic Subject preservatio n 0.000 0.050 0.100 0.150 0.200 0.250 0.300 0.350 0.400 0.450 Prompt Followin g 0.000 0.050 0.100 0.150 0.200 0.250 0.300 0.350 Image Quality BLIP-Diffusion IP-Adapter MS-Diffusion OminiControl SSR-Encoder UNO Emu2 Realcustom Omnigen Custom Diffusion Dreambooth Textual Inversion lambda-Eclipse HiPer NeTI
-0.2

-0.1 0 0.1 0.2 0.3 0.4 0.5 0.6 PH_O
0.200 0.220 0.240 0.260 0.280 0.300 0.320 0.340 0.360 PH_O
0.200 0.220 0.240 0.260 0.280 0.300 0.320 0.340 PH_O
N-PH_A
PH_H
N-PH_A
PH_H
N-PH_A
PH_H
N-PH_O
N-PH_H
PH_A
N-PH_O
N-PH_H
PH_A
N-PH_O
N-PH_H
PH_A
Image Quality Subject Preservation Prompt Following

Method Subject Preservation↑ Prompt Following↑ **Image Quality**↑

PH N-PH PH N-PH PH N-PH

BLIP-Diffusion 0.209 0.190 0.276 0.279 0.225 0.220 IP-Adapter 0.232 0.220 0.315 0.318 0.266 0.266 MS-Diffusion 0.356 0.341 **0.338 0.336** 0.295 0.291

OminiControl 0.258 0.259 0.334 0.333 0.289 0.292

SSR-Encoder 0.209 0.185 0.295 0.296 0.248 0.245 UNO **0.414 0.394** 0.324 0.320 0.279 0.275 Emu2 0.359 0.294 0.305 0.301 0.261 0.257 RealCustom++ 0.371 0.383 0.332 0.331 **0.297 0.298** OmniGen 0.183 0.201 0.323 0.321 0.266 0.264 Custom Diffusion 0.066 0.052 0.323 0.322 0.239 0.240 DreamBooth 0.165 0.138 0.320 0.323 0.243 0.250 Textual Inversion 0.117 0.088 0.301 0.293 0.226 0.222

λ-Eclipse 0.263 0.236 0.292 0.293 0.243 0.240

HiPer 0.144 0.112 0.317 0.323 0.247 0.247

NeTI 0.201 0.169 0.304 0.292 0.237 0.228

Aver. 0.236 0.217 0.313 0.312 0.257 0.256

Method **Subject Preservation**

PH_O PH_H PH_A **N-PH_O N-PH_H N-PH_A**

BLIP-Diffusion 0.202 0.201 0.24 0.186 0.189 0.206 IP-Adapter 0.232 0.193 0.267 0.226 0.188 0.237 MS-Diffusion 0.362 0.315 0.371 0.358 0.296 0.333 OminiControl 0.293 0.114 0.249 0.291 0.17 0.247 SSR-Encoder 0.199 0.186 0.26 0.193 0.162 0.185 UNO **0.453** 0.312 0.361 **0.428 0.315** 0.365 Emu2 0.358 **0.326** 0.387 0.305 0.266 0.285 RealCustom++ 0.383 0.291 **0.396** 0.415 0.26 **0.412** OmniGen 0.183 0.194 0.176 0.19 0.196 0.249 Custom Diffusion 0.067 0.014 0.103 0.059 0.035 0.043 DreamBooth 0.188 0.044 0.184 0.164 0.07 0.124 Textual Inversion 0.104 0.091 0.184 0.078 0.101 0.105 λ-Eclipse 0.252 0.266 0.3 0.236 0.221 0.256 HiPer 0.143 0.083 0.195 0.126 0.079 0.098 NeTI 0.195 0.159 0.259 0.164 0.156 0.201 Aver. 0.241 0.186 0.262 0.228 0.180 0.223

Method **Prompt Following**

PH_O PH_H PH_A **N-PH_O N-PH_H N-PH_A**

BLIP-Diffusion 0.281 0.237 0.293 0.285 0.26 0.282 IP-Adapter 0.317 0.294 0.322 0.317 0.319 0.317 MS-Diffusion **0.340** 0.319 0.347 **0.338** 0.332 0.337 OminiControl 0.335 0.319 0.344 0.334 0.33 **0.338** SSR-Encoder 0.302 0.261 0.3 0.297 0.287 0.301 UNO 0.327 0.297 0.337 0.321 0.311 0.325 Emu2 0.307 0.282 0.317 0.306 0.283 0.303 RealCustom++ 0.333 0.312 0.342 0.331 **0.333** 0.333 OmniGen 0.320 **0.320** 0.334 0.318 0.328 0.324 Custom Diffusion 0.324 0.313 0.33 0.322 0.319 0.324 DreamBooth 0.321 0.319 0.319 0.322 0.323 0.327 Textual Inversion 0.301 0.282 0.315 0.292 0.291 0.298 λ-Eclipse 0.295 0.268 0.3 0.294 0.283 0.303 HiPer 0.318 0.307 0.32 0.323 0.319 0.328 NeTI 0.306 0.279 0.315 0.294 0.285 0.297 Aver. 0.315 0.294 0.322 0.313 0.307 0.316

Method **Image Quality**

PH_O PH_H PH_A **N-PH_O N-PH_H N-PH_A**

BLIP-Diffusion 0.213 0.233 0.262 0.21 0.228 0.244 IP-Adapter 0.251 0.294 0.298 0.25 0.293 0.289

MS-Diffusion 0.287 0.307 0.315 0.284 0.301 0.306

OminiControl 0.283 0.295 0.307 0.284 0.302 0.308 SSR-Encoder 0.236 0.259 0.281 0.232 0.262 0.271 UNO 0.270 0.285 0.305 0.265 0.282 0.3 Emu2 0.249 0.284 0.287 0.249 0.265 0.278

RealCustom++ 0.289 0.312 0.317 **0.288 0.316 0.314**

OmniGen 0.256 0.294 0.278 0.254 0.284 0.277 Custom Diffusion 0.236 0.237 0.255 0.236 0.241 0.249 DreamBooth 0.238 0.255 0.255 0.245 0.254 0.267 Textual Inversion 0.218 0.231 0.248 0.214 0.234 0.235 λ-Eclipse 0.234 0.257 0.263 0.23 0.254 0.262

HiPer 0.237 0.256 0.273 0.238 0.255 0.271

NeTI 0.228 0.244 0.261 0.218 0.24 0.249 Aver. 0.248 0.270 0.280 0.246 0.267 0.275