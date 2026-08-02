**006**

**009**

**019**

**024**

**029 030**

**032**

**034**

**036**

# MULAN: MULTIMODAL-LLM AGENT FOR PROGRES-SIVE AND INTERACTIVE MULTI-OBJECT DIFFUSION

Anonymous authors Paper under double-blind review

#### ABSTRACT

Existing text-to-image models still struggle to generate images of multiple objects, especially in handling their spatial positions, relative sizes, overlapping, and attribute bindings. To efficiently address these challenges, we develop a trainingfree Multimodal-LLM agent (MuLan), as a human painter, that can progressively generate multi-object with intricate planning and feedback control. MuLan harnesses a large language model (LLM) to decompose a prompt to a sequence of sub-tasks, each generating only one object by stable diffusion, conditioned on previously generated objects. Unlike existing LLM-grounded methods, MuLan only produces a high-level plan at the beginning while the exact size and location of each object are determined upon each sub-task by an LLM and attention guidance. Moreover, MuLan adopts a vision-language model (VLM) to provide feedback to the image generated in each sub-task and control the diffusion model to re-generate the image if it violates the original prompt. Hence, each model in every step of MuLan only needs to address an easy sub-task it is specialized for. The multi-step process also allows human users to monitor the generation process and make preferred changes at any intermediate step via text prompts, thereby improving the human-AI collaboration experience. We collect 200 prompts containing multi-objects with spatial relationships and attribute bindings from different benchmarks to evaluate MuLan. The results demonstrate the superiority of MuLan in generating multiple objects over baselines and its creativity when collaborating with human users.

# 1 INTRODUCTION

Diffusion models [\(Sohl-Dickstein et al., 2015;](#page-10-0) [Ho et al., 2020;](#page-10-1) [Song et al., 2020\)](#page-10-2) have shown growing potential in generative AI tasks, especially in creating diverse and high-quality images with text prompts [\(Saharia et al., 2022;](#page-10-3) [Rombach et al., 2022\)](#page-10-4). However, current state-of-the-art text-to-image (T2I) models such as Stable Diffusion [\(Rombach et al., 2022\)](#page-10-4) and DALL-E 3 [\(Betker](#page-9-0) [et al., 2023\)](#page-9-0) still struggle to deal with complicated prompts involving multiple objects and lack precise control of their spatial relations, potential occlusions, relative sizes, etc. As shown in Figure [2,](#page-2-0) to generate a sketch of "The orange pumpkin is on the right side of the black door", even the SOTA open-source T2I model, Stable Diffusion XL [\(Podell et al., 2023\)](#page-10-5), still generates wrong attribute-binding as well as incorrect spatial positions of several objects.

Among works that aim to improve the controllability of T2I models on complicated prompts, a recent promising line of research seeks to utilize large language models (LLMs), e.g., ChatGPT, GPT-4 [\(Achiam et al., 2023\)](#page-9-1), to guide the generation process [\(Lian et al., 2023;](#page-10-6) [Feng et al., 2023\)](#page-9-2). Specifically, an LLM is prompted to generate a layout for the given prompt, i.e., a bounding box for each object in the image, given detailed instructions or demonstrations if necessary. However, due to the limited spatial reasoning capability of LLMs as well as their lack of alignment with the diffusion models, it is still challenging for LLMs to directly generate a complete and precise layout for multiple objects. Without a feedback loop interacting with the generative process, the layout's possible mistakes cannot be effectively detected and corrected. Moreover, the layout is often applied as an extra condition in addition to the original prompt (e.g., bounding boxes combined with GLIGEN [\(Li et al., 2023\)](#page-10-7)), so the diffusion models may still generate an incorrect image due to its misunderstanding of the complicated prompt.

**059**

**061**

**064**

**067**

**069 070 071**

**074**

**079**

**089 090 091**

**094**

**104 105 106**

![](_page_1_Diagram_1.jpeg)

Figure 1: The proposed training-free Multimodal-LLM Agent (MuLan) for Progressive Multi-Object Diffusion. MuLan consists of three main components: (1) LLM planning; (2) Single-object diffusion with attention guidance; and (3) VLM-feedback control. MuLan first decomposes a complicated prompt into a sequence of sub-prompts each for one object, and then generates one object per step conditioned on a sub-prompt and previously generated objects, where LLM plans the rough layout of the object and attention guidance provides an accurate mask for it. The VLM-feedback control allows MuLan to correct mistakes in each step by adjusting hyperparameters in (2).

To address the limitations and challenges of previous methods, we develop a training-free and controllable T2I generation paradigm that does not require demonstrations but mainly focuses on improving the tool usage of existing models. Our paradigm is built upon a progressive multi-object generation by a Multimodal-LLM agent (MuLan), which generates only one object per stage, conditioned on generated objects in the image and attention masks of the most plausible positions to place the new object. Unlike previous methods that add conditions to each model and make the task even more challenging, MuLan uses an LLM as a planner decomposing the original T2I task into a sequence of easier subtasks. Each subtask generates one single object, which can be easily handled by diffusion models. To be noted, the LLM applied at the beginning of MuLan only focuses on high-level planning rather than a precise layout of bounding boxes, while the exact size and position of each object are determined later in each stage by LLM and attention guidance based on the generated objects in the image. Hence, we can avoid mistakes in the planning stage and find a better placement for each object adaptive to the generated content and adhering to the original prompt. In addition, MuLan builds a feedback loop monitoring the generation process, which assesses the generated image per stage using a vision-language model (VLM). When the generated image violates the prompt, the VLM will adjust the diffusion model to re-generate the image so any mistake can be corrected before moving to the next stage. Furthermore, we develop a strategy applied in each stage to handle the overlapping between objects, which is commonly ignored by previous work [\(Lian](#page-10-6) [et al., 2023\)](#page-10-6).

Therefore, MuLan obtains better controllability of the multi-object composition. An illustration of the progressive generation process is shown in Figure [1.](#page-1-0) Note that there is a concurrent work called RPG [\(Yang et al., 2024\)](#page-11-0) sharing a similar high-level idea (i.e., decomposing the prompt into sub-tasks) with MuLan. However, there still exist substantial differences between ours and RPG. MuLan generates each object conditioned on previously generated objects while RPG generates all objects independently. MuLan does not require any manually designed demonstrations for in-context learning. In addition, as shown in Section [4.1,](#page-7-0) MuLan can be directly applied to human-agent interaction during generation, which greatly boosts the flexibility and effectiveness of the generation. To evaluate MuLan, we curate a dataset of intricate and challenging prompts from

**114 115**

**117**

**119**

**127**

**129 130**

**134**

**136**

different benchmarks. To compare MuLan with existing approaches, we prompt GPT-4V [\(OpenAI,](#page-10-8) [2023\)](#page-10-8) several questions based on the input texts to comprehensively evaluate the alignment of the generated images with the prompts from three aspects. We further conduct human evaluations of the generated images. Extensive experimental results show that MuLan can achieve better controllability over the generation process and generate high-quality images aligning better with the prompts than the baselines. Example images generated by different methods are shown in Figure [2.](#page-2-0) Our main contributions are summarized as follows:

Figure 2: Examples of MuLan-generated images, compared to the original SD-v1.4 [\(Rombach et al.,](#page-10-4) [2022\)](#page-10-4), the original SDXL [\(Podell et al., 2023\)](#page-10-5), Structure diffusion [\(Feng et al., 2022\)](#page-9-3), Promptist [\(Hao et al., 2022\)](#page-10-9), and PixArt-α [\(Chen et al., 2023\)](#page-9-4).

- We propose a novel training-free paradigm for text-to-image generation and a Multimodal-LLM agent. It achieves better control in generating images for complicated prompts consisting of multiple objects with specified spatial relationships and attribute bindings.
- We propose an effective strategy to handle multi-object occlusion in T2I generation, which improves the image quality and makes them more realistic.
- We curate a dataset of prompts to evaluate multi-object composition with spatial relationships and attribute bindings in T2I tasks. The quantitative results and human evaluation results show that our method can achieve better results compared to different controllable generation methods and general T2I generation methods.
- We show that the proposed framework can be applied to human-agent interaction during generation. This enables users to effectively monitor and change/adjust the generation process during generation instead of waiting until all the generation is finished.

# 2 RELATED WORK

Diffusion models As a new family of generative models, diffusion models have attracting more and more attention due to its powerful creative capability. Text-to-image generation, which aims to generate the high-quality image aligning with given text prompts, is one of the most popular applications [\(Nichol et al., 2021;](#page-10-10) [Saharia et al., 2022;](#page-10-3) [Rombach et al., 2022;](#page-10-4) [Betker et al., 2023\)](#page-9-0). Among different powerful diffusion models, the latent diffusion model [\(Rombach et al., 2022\)](#page-10-4) has shown amazing capability and has been widely used in practice due to the efficiency and superior performance, which is also the backbone of the current SOTA stable diffusion models. Different from the typical diffusion models which directly perform the diffusion and denoising process in the pixel space, the latent diffusion model perform the whole process in the encoded latent space [\(Rombach](#page-10-4) [et al., 2022\)](#page-10-4), which can greatly reduce the training and inference time. Recently, empowered by a significantly expanded model capacity, Stable Diffusion XL has demonstrated performance levels approaching commercial application standards [\(Podell et al., 2023\)](#page-10-5). Detailed background on the procedure of diffusion models is provided in Appendix [G.](#page-15-0)

Composed generation in diffusion models Although Stable Diffusion model has shown unprecedented performance on the T2I generation task, it still struggles with text prompts with multi-object, especially when there are several spatial relationships and attribute bindings in the prompts. To achieve more controllable and accurate image compositions, many compositional generation methods have been proposed. StructureDiffusion [\(Feng et al., 2022\)](#page-9-3) proposed a training-free method to parse the input prompt and combine it with the cross-attention to achieve better control over attribute

**166 167**

**169**

**171**

**204**

**206**

bindings and compositional generation. On the other hand, Promptist [\(Hao et al., 2022\)](#page-10-9) aimed to train a language model with the objective of optimizing input prompts, rendering them more comprehensible and facilitative for diffusion models. Recently, Ranni [\(Feng et al., 2024\)](#page-10-11) finetunes an LLM to generate bounding boxes and colors. Then they use these as conditions to finetune textto-image models for image generation. In addition, AnyDoor [\(Chen et al., 2024b\)](#page-9-5) also requires finetuning of diffusion models for better generation. Several works utilize the large language model to directly generate the whole layout for the input prompt with in-context learning, and then generate the image conditioned on the layout [\(Lian et al., 2023;](#page-10-6) [Feng et al., 2023;](#page-9-2) [Wu et al., 2024\)](#page-11-1). While all the previous take the whole input prompt, we propose to turn the original complicated task into several easier sub-tasks. A training-free multimodal-LLM agent is utilized to progressively generate objects with feedback control so that the whole generation process would be better controlled. Very recently, a concurrent work RPG [\(Yang et al., 2024\)](#page-11-0) also proposed to utilize LLM agent to decompose the prompt into different subtasks. However, MuLan generates each object step by step and correct mistakes after each step rather than treating all subtasks independently and does not need a well-designed in-context learning demonstrations. We defer a more thorough discussion with RPG [\(Yang et al., 2024\)](#page-11-0) in Appendix [B.](#page-12-0)

### 3 MULTIMODAL-LLM AGENT (MULAN)

Existing diffusion models often struggle with complicated prompts but can handle simpler ones. Recent approaches train a model or apply in-context learning given similar examples to produce a detailed layout for the prompt in advance and the diffusion model can generate each part of the layout with a simpler prompt separately. Rather than generating all objects at once or in parallel, MuLan is inspired by many human painters, who start by making a high-level plan, painting objects one after another as planned, and correcting mistakes after each step if needed. Thereby, the constraints between objects can be naturally taken into account.

#### 3.1 OVERVIEW

MuLan begins by strategically planning and decomposing an intricate input prompt into a manageable sequence of sub-prompts, each focusing on an easier sub-task generating one single object. MuLan then adopts a progressive strategy that generates one object in each stage conditioned on previously generated objects using a diffusion model. Simultaneously, a VLM offers insightful feedback and adaptively adjusts the generation process to guarantee precision in accomplishing each subtask. Compared to previous methods, MuLan is entirely training-free and does not require any in-context examples. As illustrated in Fig. [1,](#page-1-0) MuLan is composed of three components:

- Prompt decomposition by LLM planning, which produces a sequence of sub-prompts, each focusing on generating one object in the prompt.
- Conditional single-object diffusion with LLM planning and attention guidance, which generates a new object conditioned on the previous step's image using a stable diffusion model. While a sub-prompt from LLM planning provides text guidance, the object's size and position are controlled by an attention mask, which guides the object to be correctly positioned and generated.
- Feedback control by interacting with VLM, which inspects the image generated per stage and adjusts hyperparameters and attention guidance to re-generate the image if it violates the original prompt.

#### 3.2 PROMPT DECOMPOSITION BY LLM PLANNING

Given a complex prompt p, MuLan first uses an LLM to automatically decompose p into N objectwise sub-prompts p1:<sup>N</sup> . During decompostion, MuLan specifically asks the LLM to produce a sequence of objects that will be created in the default order from left to right and bottom to top in the image. The LLM can easily finish this task by leveraging its prior knowledge to fill all objects of p to an empty list of the pre-defined order without in-context learning which requires manually designed examples. Let objs = {obj<sup>1</sup> , · · · , objn, · · · , obj<sup>N</sup> } be the LLM-planned N objects extracted from p. For the first object, the sub-prompt is simply p<sup>1</sup> ="{obj1}". For object-n with n > 1, the subtask is to generate object-n conditioned on previous objects and the textual subprompt is defined as p<sup>n</sup> ="{objn} and {objn−1}". MuLan conducts the above global planning

**224**

**236 237**

**254**

**256**

**259**

by an LLM at the very beginning before generating any image. The detailed prompts and template for LLM planning can be found in Appendix [I.](#page-16-0)

When generating each object in Section [3.3,](#page-4-0) we will use the LLM again as a local planner of the object's position and size, i.e., by generating a mask in the image and coordinating its overlap with previous objects. Then a diffusion model is used to generate the object under the attention guidance of the mask. These will be further elaborated in Section [3.3.](#page-4-0)

#### 3.3 CONDITIONAL SINGLE-OBJECT DIFFUSION WITH LLM PLANNING AND ATTENTION GUIDANCE

At stage-n, the diffusion model only focuses on generating obj<sup>n</sup> according to the sub-prompt pn, ensuring that obj<sup>n</sup> can be correctly positioned and generated. To this end, MuLan utilizes the LLM to plan the relative position and size of objn, allocating a rough mask (i.e., a bounding box) M<sup>n</sup> for objn. Then, cross-attention guidance is applied during the generation of obj<sup>n</sup> to ensure obj<sup>n</sup> is appropriately positioned within Mn. The pipeline is given in Figure [3](#page-4-1) with the complete procedure listed in Algorithm [1](#page-16-1) in Appendix [H.](#page-15-1) We will introduce it step by step in the following.

**Sub-prompt**  ���� Rough mask ��" **GuidanceAttention Single-ObjectTask** ������" ������" **Stage 1 to N** ������"#\$ ������" ������% Objects {������&}&'\$ % ��. �� \$ .% Precise masks {��.&}&'\$ % **Stage 1 to N** ��."#\$ ��." ������\$ Figure 3: Single object diffusion with LLM planning and attention guidance for obj<sup>n</sup> (detailed procedure in Algorithm [1](#page-16-1) in Appendix [H\)](#page-15-1). At stage-n, MuLan first allocates a rough mask as a bounding box M<sup>n</sup> ≜ (xn, yn, wn, hn) (x/y coordinates of the top-left corner, width, and height) to guide the generation of obj<sup>n</sup> in the image. As shown in Figure [3,](#page-4-1) M<sup>n</sup> can be derived from objn's relative position opt<sup>n</sup> ∈ Opts={left,right,top,bottom}, the total number of objects Num<sup>n</sup> in the same position/region as objn, and current available space in the image. Num<sup>n</sup> and current available space, combined together, determines the size of objn. MuLan utilizes the LLM planner to reason opt<sup>n</sup> and Num<sup>n</sup> given the sub-prompt p<sup>n</sup> [1](#page-4-2) , while the current available space can be determined by the precise mask M˜ <sup>n</sup>−<sup>1</sup> which describes the exact position of previously generated objn−<sup>1</sup> and can be easily extracted from the cross-attention maps. It is worth noting that since there is no previously generated objects for the first object, the available space for obj<sup>1</sup> is the whole image. For detailed computation of Mn, please refer to Appendix [K.](#page-17-0)

![](_page_4_Diagram_7.jpeg)

### LLM Planning of a Rough Mask for **obj**n.

Once M<sup>n</sup> is determined, the cross-attention guidance is utilized during generation of obj<sup>n</sup> to ensure obj<sup>n</sup> is correctly generated within Mn, as elaborated in the following.

Single-Object Generation with Attention Guidance. Given the rough mask M<sup>n</sup> of objn, the next is to ensure the generated obj<sup>n</sup> will be correctly located within Mn. A natural and intuitive way to achieve this in diffusion models is to guide the generation of the cross-attention map of objn, which builds the relevance between the text prompt and the location of generated object.

To this end, MuLan manipulates the cross-attention map of obj<sup>n</sup> under the guidance of Mn, using the backward guidance method [\(Chen et al., 2024a\)](#page-9-6), to maximize the relevance inside Mn. Specifically, let A be the cross-attention map, Am,k represents the relevance between the spatial location m and token-k that describes obj<sup>n</sup> in the prompt. Larger value in Am,k indicates that obj<sup>n</sup> is more likely located at the spatial location of m. The goal is to maximize the relevance Am,k inside the mask M<sup>n</sup> while minimizing the relevance outside the mask Mn. Hence the following energy function is utilized:

$$E(\mathbf{A}, \mathbf{M}_n, k) = \left(1 - \frac{\sum_{m \in \mathbf{M}_n} \mathbf{A}_{m,k}}{\sum_m \mathbf{A}_{m,k}}\right)^2, \quad (1)$$

<sup>1</sup>The detailed prompt template can be found in Appendix [J.](#page-16-2)

**289 290 291**

**294**

**301**

**304**

**306**

**309**

**314 315**

**318 319**

**321**

where P m∈M<sup>n</sup> denotes the summation over the spatial locations included in Mn, and P <sup>m</sup> denotes the summation over all the spatial locations in the attention map. In every step-t of the earlier generation process, MuLan applies gradient descent to minimize the energy by updating the input latent zn,t for object objn. In this way, the cross-attention map corresponding to obj<sup>n</sup> will achieve the largest relevance inside Mn, meaning obj<sup>n</sup> can be correctly positioned inside the rough mask.

On the other hand, to take the previous objects and their constraints into account when generating objn, we further combine the latent of obj<sup>n</sup> and objn−<sup>1</sup> . Specifically, after step-t of reverse process (t varies from T to 0), we update the latent zn,(t−1) by

$$z_{n,(t-1)} = M'_n \odot z_{n,(t-1)} + (1 - M'_n) \odot z_{(n-1),(t-1)}, \quad (2)$$

where ⊙ computes element-wise product and [M′ n ]uv = <sup>1</sup>u∈[xn,xn+wn],v∈[yn,yn+hn] is the 0-1 indicator of whether coordinates (u, v) is included in the bounding box of Mn.

MuLan applies the above single-object diffusion to each object one after another from obj<sup>1</sup> to obj<sup>N</sup> , as planned by the LLM at the very beginning. The procedure of generating obj<sup>n</sup> is detailed in Algorithm [1.](#page-16-1)

Objects Overlapping. Overlapping between objects is a key challenge in text-to-image diffusion models. However, it lacks attention in previous methods [\(Lian et al., 2023;](#page-10-6) [Feng et al., 2023\)](#page-9-2). Instead, we propose an effective strategy that can be merged into the procedure above. Specifically, at the generation of object objn, we prompt the LLM to judge if there is overlapping between obj<sup>n</sup> and objn−<sup>1</sup> . If there is overlapping, we first compute three candidates for the rough mask {Mn,i} 3 <sup>i</sup>=1, associated with three overlapping ratios {ri} 3 <sup>i</sup>=1 = {10%, 30%, 50%} between objn−<sup>1</sup> and objn.

Given the three masks Mn,i, MuLan generates three candidate images using Algorithm [1.](#page-16-1) Then the CLIP scores [\(Hessel et al., 2021\)](#page-10-12) between the generated images and the input prompt p<sup>n</sup> are computed and the image with the maximal CLIP score is selected as the generated image for objn. An illustration is given in Figure [11](#page-18-0) with more details of candidate masks in Appendix [L.](#page-18-1)

#### 3.4 INTERACTION WITH VLM AND HUMAN USERS DURING GENERATION

To correct the possible mistakes made in the sequential generation process, MuLan builds an adaptive feedback-loop control by interacting with a vision-language model (VLM). After each generation stage, MuLan queries the VLM to inspect the generated object(s) and its consistency with the input prompt. If they do not align well, MuLan will adjust the backward guidance of the current stage to re-generate the object. More specifically, MuLan will modify the hyperparameters of backward guidance to control the strength of the guidance. We empirically found that the errors are typically the size or the position of the generated object. For example, the object may be too large and outside the rough mask. Hence the guidance strength needs to be larger to make the object smaller. In the whole generation process, if MuLan needs to regenerate an object, it will try different guidance strength, i.e., the weight of the gradient of the energy function (Eq. [1\)](#page-4-3), and the loss threshold that is used for stopping criteria of guidance. In cases with incorrect positions, it will also re-plan the spatial location and regenerate the object. Such a close-loop control involves LLM, diffusion, and VLM and significantly automates the T2I generation for complicated prompts, leading to a more accurate generation in practice.

In addition, the multi-step process naturally allows human-agent interaction/collaboration during generation in practice. Users can timely monitor the generation process. In this way, the interaction enables users to make preferred changes and adjustments to the generated images easily and effectively by providing adjusting prompts to MuLan at any intermediate step, such as attribute adjustment, object adjustment, and spatial relationship adjustment. With the adjusting prompts, MuLan will utilize the LLM to modify the original prompt accordingly and change the generation process to the preferred one. An illustration for different changes or adjustments during generation is shown in Figure [4,](#page-6-0) which indicates MuLan can achieve both simple and composed complex adjustments with interaction. In contrast, for other existing generation and editing methods, users have to wait until the whole generation process is finished. Therefore, the proposed framework is more user-friendly and flexible in terms of human-agent interaction and collaboration.

**329**

**334**

**354 355 356**

**358 359**

**361**

**364**

**369**

![](_page_6_Diagram_1.jpeg)

Figure 4: An illustration tree for difference cases of human-agent interaction during generation. The middle branch (connected by blue arrows) shows the original generation process without human-agent interaction. The top and bottom branches show different complex composed humanagent interaction during generation for various adjustments, involving object adjustments, attribute adjustments, and spatial relationship adjustments, which demonstrate the flexibility and effectiveness of MuLan for human-agent interaction during generation.

### 4 EXPERIMENTS

Dataset To evaluate our framework, we construct a prompt dataset from different benchmarks. Specifically, since our focus is to achieve better generation for complex prompts containing multiobjects with both spatial relationships and attribute bindings, we first collect all complex spatial prompts from T2I-CompBench [\(Huang et al., 2023\)](#page-10-13). To make the experiments more comprehensive, we let ChatGPT generate about 400 prompts with different objects, spatial relationships, and attribute bindings so that the prompt sets consists of about 600 prompts. To further evaluate the capability of our framework on extremely complex and hard prompts, we manually add prompts that SDXL fails to generate, leading to a hard prompt dataset containing 200 prompts. Similar to the complex spatial prompts in T2I-CompBench [\(Huang et al., 2023\)](#page-10-13), each prompt in our curated dataset typically contains two objects with various spatial relationships, with each object containing attribute bindings randomly selected from {color,shape,texture}.

Models & Baseline As a training-free framework, MuLan can be incorporated into any existing diffusion models. We evaluate two stable diffusion models with our framework, Stable Diffusion v1.4 [\(Rombach et al., 2022\)](#page-10-4) and the SOTA Stable Diffusion XL [\(Podell et al., 2023\)](#page-10-5). To verify the superiority of MuLan, we compare it with previous controllable generation methods and general T2I generation methods. Specifically, we evaluate Structure Diffusion [\(Feng et al., 2022\)](#page-9-3), Promptist [\(Hao et al., 2022\)](#page-10-9), the original Stable Diffusion v1.4, the original SDXL, and the recent SOTA diffusion model PixArt-α [\(Chen et al., 2023\)](#page-9-4).

Implementation Details MuLan use GPT-4 [\(Achiam et al., 2023\)](#page-9-1) as the LLM planner, and LLaVA-1.5 [\(Liu et al., 2023\)](#page-10-14) as the VLM checker to provide the feedback. We also conducted an ablation study to show the importance of the feedback control provided by the VLM and the effect of different VLMs. Moreover, we found the attention blocks utilized during the attention guidance are vital, which can be classified as near-input blocks, near-middle blocks, and near-output blocks. We utilize the near-middle blocks in our main experiments and also show the ablation results of different block. Our codes (including the prompt dataset) are available in the supplementary material. All the experiments are conducted on a single NVIDIA RTX A6000 GPU.

Evaluation Since the prompt dataset contains texts with complex compositions, we design a questionnaire to comprehensively investigate the alignment between the generated image and the corresponding input text. The questionnaire is composed of three aspects - object completeness, correctness of attribute bindings, and correctness of spatial relationships. We only set two options for each

**381**

**384**

**386**

question (Yes or No), without any ambiguity. For detailed questions and examples of the evaluation, please refer to Appendix [M.](#page-18-2) For each aspect of the evaluation, we compute the percentage of answers with "Yes". Given the generated image, we assess the image's quality using a questionnaire asking both the state-of-the-art multi-modal large language model (GPT-4V [\(OpenAI, 2023\)](#page-10-8)) and the human evaluator.

#### 4.1 MAIN RESULTS AND ANALYSIS

Results on GPT Evaluation Given the generated image, we prompt GPT-4V to answer the questions about the image in the questionnaire, where each only focuses on one of the three aspects. The results for different methods and different base models are shown in Table [1.](#page-7-1) The results show that our framework can achieve the best performance compared to different controllable generation methods and T2I generation methods. In particular, in the two 'harder' aspects - attribute bindings and spatial relationships, MuLan can surpass other methods by a large margin. More results can be found in Figure [5](#page-7-2) and Appendix [O.](#page-21-0)

Table 1: GPT-4V evaluation human evaluation of images generated by different methods for complicated prompts.

| Method                                  | Object completeness | Attribute bindings | Spatial relationships | Overall  |
|-----------------------------------------|---------------------|--------------------|-----------------------|----------|
| Structure Diffusion (Feng et al., 2022) | 88.97%             |                    |                       |          |
|                                         | 87.37%              | 54.62%            |                       |          |
|                                         |                     | 62.63%             | 34.36%               |          |
|                                         |                     |                    | 24.24%                | 64.31%  |
| Promptist-SD v1.4 (Hao et al., 2022)    | 80.36%             |                    |                       |          |
|                                         | 70.71%              | 49.23%            |                       |          |
|                                         |                     | 52.02%             | 24.49%               |          |
|                                         |                     |                    | 13.13%                | 56.73%  |
| Promptist-SDXL (Hao et al., 2022)       | 94.36%             |                    |                       |          |
|                                         | 93.94%              | 70.00%            |                       |          |
|                                         |                     | 78.28%             | 35.89%               |          |
|                                         |                     |                    | 33.33%                | 72.92%  |
| SD v1.4 (Rombach et al., 2022)          | 90.31%             |                    |                       |          |
|                                         | 74.49%              | 57.14%            |                       |          |
|                                         |                     | 51.02%             | 37.24%               |          |
|                                         |                     |                    | 32.65%                | 66.43%  |
| SDXL (Podell et al., 2023)              | 94.64%             |                    |                       |          |
|                                         | 78.57%              | 66.07%            |                       |          |
|                                         |                     | 53.06%             | 41.14%               |          |
|                                         |                     |                    | 24.49%                | 72.34%  |
| PixArt- α (Chen et al., 2023)           | 92.09%             |                    |                       |          |
|                                         | 76.53%              | 66.58%            |                       |          |
|                                         |                     | 61.22%             | 34.69%               |          |
|                                         |                     |                    | 32.65%                | 70.41%  |
| MuLan-SD v1.4 (Ours)                    | 93.11%             |                    |                       |          |
|                                         | 86.36%              | 74.23%            |                       |          |
|                                         |                     | 74.24%             | 51.53%               |          |
|                                         |                     |                    | 54.54%                | 77.24%  |
| MuLan-SDXL (Ours)                       | 96.17%             |                    |                       |          |
|                                         | 90.40%              | 75.00%            |                       |          |
|                                         |                     | 79.29%             | 39.29%               |          |
|                                         |                     |                    | 49.49%                | 76.33%  |

![](_page_7_Picture_6.jpeg)

Figure 5: More qualitative examples of images generated by different methods on intricate prompts.

Results on Human Evaluation To further accurately evaluate the generated images about the alignments with human preferences, we further conduct a human evaluation by randomly sampling 100 prompts from the prompt dataset. Similarly, we ask human evaluators to finish the questionnaire used in GPT evaluation. The results are shown in Table [1,](#page-7-1) which indicates that our method can still achieve the best performance and is consistent with the GPT-4V evaluation results.

Results on Human-Agent Interaction To show MuLan is still very effective if users want to modify the input prompt or edit the generated images during the generation, i.e., the human-agent

interaction, we use ChatGPT to mimic the user to generate various adjusting prompts for the interaction with MuLan on randomly sampled 50 prompts. SD v1.4 [\(Rombach et al., 2022\)](#page-10-4) is utilized as the base model. The generated adjusting prompts focus on several aspect, i.e., attribute adjustment, object adjustment, and spatial relationship adjustment. We use GPT-4V [\(OpenAI, 2023\)](#page-10-8) to quantitatively evaluate the performance of MuLan given the final generated images and final text prompts, as shown in Table [2.](#page-8-0) The results indicate that MuLan can still achieve high accuracy even with various adjustments/changes during generation.

Table 2: GPT-4V evaluation of final generated images and final prompts after adjustments/changes. The results show that MuLan is still very effective with various adjustment of prompts during generation.

|               | Objects | Attributes | Spatial | Overall |
|---------------|---------|------------|---------|---------|
| MuLan-SD v1.4 | 95.92%  | 72.45%     | 28.57%  | 73.06%  |

#### 4.2 ABLATION STUDY

In this section, we show ablation results on the effect of the attention blocks during diffusion generation and the importance of the VLM feedback control in the proposed framework. 50 prompts are randomly sampled from the prompt dataset for all experiments in the ablation study.

Ablation on the attention blocks As we mentioned at the beginning of Section [4,](#page-6-1) there are three options for the attention blocks used for backward guidance, i.e., near-input blocks, near-middle blocks, and near-output blocks. We empirically found the near-middle blocks can achieve the best control and performance for the generation, which generally contains the richest semantics. Hence here we show the ablation results on different choices of the attention blocks. We utilize SD-v1.4 as the base model, and evaluate the performance of different attention blocks under our framework by GPT-4V. The results are shown in Table [3,](#page-8-1) which indicates the diffusion generation with near-middle blocks can achieve much better results compared to the other two options.

Table 3: Ablation study on attention blocks with SD-v1.4 as the base model. "Objects", "Attributes", and "Spatial" denote Object completeness, Attribute bindings, and Spatial relationships. The results (evaluated by GPT-4V [\(OpenAI, 2023\)](#page-10-8)) show that near-middle attention blocks perform the best for attention guidance.

| Guidance    | Objects | Attributes | Spatial | Overall |
|-------------|---------|------------|---------|---------|
| near-input  | 83.67%  | 55.10%     | 14.29%  | 58.37%  |
| near-middle | 97.96%  | 80.61%     | 30.61%  | 77.55%  |
| near-output | 72.45%  | 45.92%     | 22.45%  | 51.84%  |

Ablation on the VLM feedback control The VLM feedback control is a key componenet in MuLan to provide feedback and adjust the generation process to ensure the every stage's correct generation. Here, we show the importance of the feedback by removing feedback control from the whole framework. As shown in Table [4,](#page-8-2) after removing the VLM, the results would be much worse. It is because there is no guarantee or adaptive adjustment for each generation stage, which verifies that the feedback control provided by the VLM is essential to handle complex prompts. Moreover, we also test MuLan's compatibility with different VLMs. As shown in Table [5,](#page-9-7) we compare the Mulan's performance using different VLMs including LLaVA-1.5 [\(Liu et al., 2023\)](#page-10-14), GPT-4V [\(OpenAI, 2023\)](#page-10-8), and Gemini-Pro [\(Team et al., 2023\)](#page-10-15). The results show that MuLan could still maintain a good performance with different choices of the VLM and achieve good compatibility.

Table 4: Ablation study comparing MuLan with vs. without VLM feedback control, using SDv1.4 as the diffusion model and GPT-4 as the judge in evaluations. It indicates that feedback control can significantly improve the performance.

| <b>Mulah</b> | <b>Objects</b> | <b>Admitteds</b> | <b>Spatial</b> | <b>Overall</b> |
|--------------|----------------|------------------|----------------|----------------|
| w/ Feedback  | <b>95.96%</b>  | <b>80.61%</b>    | <b>30.61%</b>  | <b>77.55%</b>  |
| w/o Feedback | 81.63%         | 59.18%           | 18.37%         | 60.00%         |

**504**

**506**

**509**

**514 515 516**

**518 519**

**524**

**529**

**539**

Table 5: Ablation study of the VLM used in MuLan, using SD-v1.4 as the diffusion model and GPT-4 as the judge in evaluations. The results show that the choice of the VLM would not affect the overall performance too much.

| VLM in Mulan                   | Objects | Attributes | Spatials | Overall |
|--------------------------------|---------|------------|----------|---------|
| LLaVA-1.5 (Liu et al., 2023)   | 95.96%  | 80.61%     | 30.61%   | 77.55%  |
| GPT-4V (OpenAI, 2023)          | 95.92%  | 80.61%     | 28.57%   | 76.33%  |
| Gemini-Pro (Team et al., 2023) | 95.92%  | 83.67%     | 38.78%   | 79.59%  |

# 5 CONCLUSIONS AND LIMITATIONS

In this paper, we propose a training-free multimodal-LLM agent (MuLan) to progressively generate objects contained in the complicated input prompt with closed-loop feedback control, achieving better and more precise control on the whole generation process. By first decomposing the complicated prompt into easier sub-tasks, our method takes turns to deal with each object, conditioned on the previous one. The VLM checker further provides a guarantee with feedback control and adaptive adjustment for correct generation at each stage. Moreover, the application to the human-agent interaction further enhances the significance of MuLan, making the generation more flexible and effective to align with the preferences of users. Extensive experiments demonstrate the superiority of MuLan over previous methods, showing the potential of MuLan as a new paradigm of controllable diffusion generation. However, there are still limitations to be further addressed in the future work. Since the whole generation contains multiple stages, depending on the number of objects, it will take a longer time than a one-stage generation approach. On the other hand, MuLan may also fail to generate correct objects in some non-common corner cases of image composition. We defer more detailed discussion and illustrations of the limitations to Appendix [N.](#page-19-0)

### REFERENCES


[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical

[2] report. *arXiv preprint arXiv:2303.08774*, 2023. James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. *Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf*, 2(3), 2023. Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. *arXiv preprint arXiv:2310.00426*, 2023. Minghao Chen, Iro Laina, and Andrea Vedaldi. Training-free layout control with cross-attention guidance. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, pp. 5343–5353, 2024a. Xi Chen, Lianghua Huang, Yu Liu, Yujun Shen, Deli Zhao, and Hengshuang Zhao. Anydoor: Zeroshot object-level image customization. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 6593–6602, 2024b. Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam ¨ Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In *Forty-first International Conference on Machine Learning*, 2024. Weixi Feng, Xuehai He, Tsu-Jui Fu, Varun Jampani, Arjun Akula, Pradyumna Narayana, Sugato Basu, Xin Eric Wang, and William Yang Wang. Training-free structured diffusion guidance for compositional text-to-image synthesis. *arXiv preprint arXiv:2212.05032*, 2022. Weixi Feng, Wanrong Zhu, Tsu-jui Fu, Varun Jampani, Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. Layoutgpt: Compositional visual planning and genera-

[3] tion with large language models. *arXiv preprint arXiv:2305.15393*, 2023.

[4] **554 555 556**

[5] **559**

[6] **561**

[7] **564**

[8] **569**

[9] **579**

[10] **584**

[11] Yutong Feng, Biao Gong, Di Chen, Yujun Shen, Yu Liu, and Jingren Zhou. Ranni: Taming text-toimage diffusion for accurate instruction following. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 4744–4753, 2024. Yaru Hao, Zewen Chi, Li Dong, and Furu Wei. Optimizing prompts for text-to-image generation. *arXiv preprint arXiv:2212.09611*, 2022. Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. *arXiv preprint arXiv:2104.08718*, 2021. Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. *Advances in neural information processing systems*, 33:6840–6851, 2020. Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. *arXiv preprint arXiv:2307.06350*, 2023. Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 22511–22521, 2023. Long Lian, Boyi Li, Adam Yala, and Trevor Darrell. Llm-grounded diffusion: Enhancing prompt understanding of text-to-image diffusion models with large language models. *arXiv preprint arXiv:2305.13655*, 2023. Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. *arXiv preprint arXiv:2310.03744*, 2023. Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B Tenenbaum. Compositional visual generation with composable diffusion models. In *European Conference on Computer Vision*, pp. 423–439. Springer, 2022. Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. *arXiv preprint arXiv:2112.10741*, 2021. OpenAI. Gpt-4v(ision) system card. 2023. Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Muller, Joe ¨ Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. *arXiv preprint arXiv:2307.01952*, 2023. Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High- ¨ resolution image synthesis with latent diffusion models. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 10684–10695, 2022. Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. *Advances in Neural Information Processing Systems*, 35:36479–36494, 2022. Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In *International conference on machine learning*, pp. 2256–2265. PMLR, 2015. Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. *arXiv preprint arXiv:2011.13456*, 2020. Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. *arXiv preprint arXiv:2312.11805*, 2023.

[12] Tsung-Han Wu, Long Lian, Joseph E Gonzalez, Boyi Li, and Trevor Darrell. Self-correcting llmcontrolled diffusion models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 6327–6336, 2024.

[13] Ling Yang, Zhaochen Yu, Chenlin Meng, Minkai Xu, Stefano Ermon, and Bin Cui. Mastering textto-image diffusion: Recaptioning, planning, and generating with multimodal llms. *arXiv preprint arXiv:2401.11708*, 2024.

[14] **654**

[15] **656**

[16] **659**

[17] **661**

[18] **664 665**

[19] **669**

[20] **674**

[21] **684**

[22] **686**

[23] **689 690 691**
#### A BROADER IMPACT

Our work will bring significant advantages to both the research community focused on diffusion models and the practical application of T2I generation.

In terms of the research community, we present a new and novel controllable image generation paradigm that demonstrates exceptional controllability and produces remarkable results even when tackling challenging tasks. This pioneering approach can offer valuable insights for future investigations into diffusion models.

Regarding industrial applications, our method can be readily employed by T2I generation service providers to enhance the performance of their models. Moreover, the diffusion models operating within our framework are less likely to generate harmful content due to the meticulous control exerted at each generation stage.

# B DIFFERENCES BETWEEN MULAN AND THE CONCURRENT WORK RPG

As stated in Introduction and Related work, although we acknowledge that our proposed framework shares a similar high-level idea with RPG, we would like to emphasize that there are still substantial differences between ours and RPG.

Firstly, our proposed MuLan aims to progressively generate each object given each subprompt. At the same time, the objects are generated conditioned on previously generated objects. In RPG, on the other hand, all objects are generated independently. In addition, different from RPG which requires manually designed in-context examples for the CoT reasoning, ours does not have such requirement. We directly utilize LLMs for the planning during generation, which is an easier task and can be done by LLMs without in-context learning. What's more, MuLan can adaptively control and correct the generation results using feedback by the VLMs while RPG does not have the feedback for the generation. Also, for the common overlapping problem between objects, we propose a strategy to generate several candidates to deal with it. In contrast, in RPG, the overlapping parts are treated as a whole for generation.

More importantly, as we show in Section [4.1,](#page-7-0) our proposed framework can be directly applied to human-agent interaction during generation to facilitate flexible and effective changes/adjustments of the process while RPG cannot achieve the interaction. To summarize, the main differences between MuLan and RPG are as follows:

- Our proposed MuLan generates each object conditioned on previously generated objects while RPG generates all objects in parallel independently.
- MuLan does not require any in-context learning during the whole generation; in RPG, specifically designed in-context examples are needed for Chain-of-Thought reasoning.
- MuLan utilizes the VLM-based feedback control to ensure each object can be generated correctly while RPG does NOT have such a feedback mechanism.
- We propose a strategy to deal with overlapping/interaction between objects whereas RPG directly treats overlapping objects as a whole part to generate.
- MuLan can be directly applied to human-agent interaction during generation for flexible and various adjustments of the generation process while RPG cannot achieve it.

### C MORE COMPARISON RESULTS WITH CONTROLLABLE IMAGE GENERATION METHODS

Here we present more quantitative results between MuLan and other state-of-the-art controllable image generation methods, Ranni [\(Feng et al., 2024\)](#page-10-11) and Composable Diffusion [\(Liu et al., 2022\)](#page-10-16). We randomly sample 50 prompts from the prompt dataset and use GPT-4V to evaluate the alignment between generated images and prompts.

Table 6: GPT-4V evaluation of MuLan and more controllable generation methods. The results show that MuLan with SD-v1.4 performs better, even surpassing training-based methods.

|                                         | Objects | Attributes | Spatial | Overall |
|-----------------------------------------|---------|------------|---------|---------|
| MuLan-SD v1.4                           | 97.96%  | 80.61%     | 30.61%  | 77.55%  |
| Ranni (Feng et al., 2024)               | 70.41%  | 38.78%     | 20.41%  | 47.76%  |
| Composable Diffusion (Liu et al., 2022) | 90.82%  | 63.27%     | 22.45%  | 66.12%  |

#### D COMPARISON WITH STABLE DIFFUSION 3

To further evaluate the effectiveness of the proposed training-free framework MuLan, we also qualitatively compare MuLan with the latest state-of-the-art text-to-image generation model, Stable Diffusion 3 [\(Esser et al., 2024\)](#page-9-8). As shown in Figure [6,](#page-13-1) even Stable Diffusion 3 cannot deal with prompts with simple spatial relationships steadily, while MuLan with SD-v1.4 can achieve controllable generation and generate correct images that align with prompts, indicating the effectiveness of the proposed framework.

![](_page_13_Picture_5.jpeg)

Figure 6: Qualitative comparison between MuLan and Stable Diffusion 3.

# E VISUAL QUALITY AND REALISM OF MULAN-GENERATED IMAGES

Please note that since MuLan is training-free, the visual quality and realism of generated images highly depend on the utilized base models, e.g., SD v-1.4, SDXL, etc. MuLan does not degrade the visual quality of generated images. To further show this, we present more visualization results of MuLan and the base models. As shown in Figure [7,](#page-14-0) MuLan with SDXL and the original SDXL have very similar performance in terms of visual quality and realism.

Figure 7: Visual quality and realism comparison between MuLan and the original base model.

**814 815**

**817**

**819**

**829**

**834**

**836**

**854**

**856**

# F MORE RESULTS ON COMPLEX OVERLAPPING PROMPTS

To further verify the effectiveness of the proposed overlapping processing module, we show more visualization results on complex overlapping prompts, including interaction between animals and humans. As shown in Figure [11,](#page-18-0) MuLan can deal with complex overlapping prompts better and show effectiveness for different overlapping cases.

![](_page_15_Picture_3.jpeg)

Figure 8: Visualization results on complex overlapping prompts.

### G BACKGROUND ON (LATENT) DIFFUSION MODELS

Consisting of the diffusion process and the reverse process, diffusion models have shown impressive capability for high-quality image generation by iteratively adding noise and denoising [\(Ho et al.,](#page-10-1) [2020\)](#page-10-1). Let x<sup>0</sup> ∼ q(x0) be the true data distribution. Starting from x0, the diffusion process adds different levels of noise pre-defined by the schedule {βt} T 1 , producing x1, · · · , x<sup>T</sup> . As T → ∞, x<sup>T</sup> will become the standard Gaussian distribution N (0, I). Accordingly, the reverse process aims to reverse the above process and reconstruct the true data distribution from p(x<sup>T</sup> ) = N (0, I) by a parameterized noise model ϵθ(·). With ϵ ∼ N (0, I), the training loss of the model can be simplified as

$$L(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, t)\|^2. \quad (3)$$

Latent diffusion models [\(Rombach et al., 2022\)](#page-10-4) have recently attracted growing attention due to their efficiency and superior performance. Instead of performing diffusion and its reverse process in the pixel space, they add noise and denoise in a latent space of z encoded by a pre-trained encoder E. Thereby, the diffusion process starts from z<sup>0</sup> = E(x0) and subsequently produces latent states z1, · · · , zt, · · · , z<sup>T</sup> . Accordingly, the training loss becomes

$$L_{LDM} = \mathbb{E}_{\mathbf{z}_0, \boldsymbol{\epsilon}, t} \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t)\|^2. \quad (4)$$

# H ALGORITHM PROCEDURE OF SINGLE-OBJECT DIFFUSION IN MULAN

**869**

**874**

**884**

**886**

**889 890 891**

**904**

**906**

**909**

Algorithm 1 Single Object Diffusion in MuLan

1: Input: Object number <sup>n</sup>, sub-prompt <sup>p</sup>n, LLM planner Planner, precise mask <sup>M</sup>˜ <sup>n</sup>−<sup>1</sup> (only for n >

1), latents {z(n−1),(t−1)}

T

<sup>t</sup>=1 (only for n > 1), attention guidance timestep threshold T

′

, combination

timestep threshold T

∗

(only for n > 1), learning rate η, diffusion model D.

2: Output: Image with obj<sup>n</sup> and its precise mask <sup>M</sup>˜ <sup>n</sup>.

3: if n = 1 then 4: opt<sup>1</sup>

, Num<sup>1</sup> = Planner(p<sup>1</sup>

)

5: Apply Eq. equation [5](#page-17-1) to compute M<sup>1</sup>

6: for t = T, · · · , 1 do 7: if t > T′

then 8: z1,t = z1,t − η · ∇<sup>z</sup>1,tE(A,M1, k)

9: end if

10: z1,(t−1) = D(z1,t, t, p<sup>1</sup>

) {Single denoising step}

11: end for 12: else

13: optn, Num<sup>n</sup> = Planner(pn, {obji}

n−1 <sup>i</sup>=1 )

14: Apply Eq. equation [6](#page-18-3) to compute M<sup>n</sup>

15: for t = T, · · · , 1 do 16: if t > T′

then 17: zn,t = zn,t − η · ∇<sup>z</sup>n,tE(A,Mn, k)

18: end if

19: zn,(t−1) = D(zn,t, t, pn)

20: if t > T <sup>∗</sup>

then

21: Apply Eq. equation [2](#page-5-0) to combine latent of obj<sup>n</sup> and objn−<sup>1</sup>

22: end if 23: end for 24: end if 25: obj<sup>n</sup> = zn,<sup>0</sup>

26: M˜ <sup>n</sup> = (˜xn, y˜n, w˜n, h˜n), a bounding box based on thresholding of <sup>1</sup>

|B| P <sup>j</sup>∈<sup>B</sup> A (j) (:,k)

{Token-k corre-

sponds to objn}

### I DETAILED PROMPT TEMPLATE OF THE GLOBAL PLANNING BY THE LLM

As stated in Section [3.2,](#page-3-0) MuLan first conduct the global planning to decompose the input prompts into N objects before the whole generation process. To this end, given the input prompt p, we prompt the LLM using the following template:

You are an excellent painter. I will give you some descriptions. Your task is to turn the description into a painting. You only need to list the objects in the description by painting order, from left to right, from down to top. Do not list additional information other than the objects mentioned in the description. Description: {p}.

In this way, the LLM will decompose the input prompt p following the pre-defined order.

# J DETAILED PROMPT TEMPLATE OF THE LOCAL PLANNING BY THE LLM

As stated in Section [3.3,](#page-4-0) the LLM is also utilized during the generation stage for local planning of the object's rough position and the object counting.

For the rough position opt<sup>1</sup> planning of the first object, we utilize the following template:

You are an excellent painter. I will give you some descriptions. Your task is to turn the description into a painting. Now given the description: {p}. If I want to paint the {obj1} in the painting firstly, where to put the {obj1}? Choose from left, right, top, and bottom. You can make reasonable guesses. Give one answer.

**924**

**929**

**954**

**956**

**959**

**961**

You are an excellent painter. I will give you some descriptions. Your task is to turn the description into a painting. Now given the description: {p}. How many non-overlapping objects are there in the horizontal direction? ONLY give the final number.

If opt<sup>1</sup> = bottom, the prompt template would be:

You are an excellent painter. I will give you some descriptions. Your task is to turn the description into a painting. Now given the description: {p}. How many non-overlapping objects are there in the vertical direction? ONLY give the final number.

For the rough position optn(n ≥ 2), we utilize the following template:

You are an excellent painter. I will give you some descriptions. Your task is to turn the description into a painting. Now given the description: {p}. If I already have a painting that contains {{obji} n−1 <sup>i</sup>=1 }, what is the position of the {objn} relative to the {objn−1}? Choose from left, right, above, bottom, and none of above. You can make reasonable guesses. Give one answer.

Then we prompt the LLM to figure out the object number by:

You are an excellent painter. I will give you some descriptions. Your task is to turn the description into a painting. Now given the description: {p}. If I already have a painting that contains {{obji} n−1 <sup>i</sup>=1 }, how many objects are there in/on the {optn} of {objn−1}? Only give the final number.

### K DETAILS FOR THE COMPUTATION OF ROUGH MASKS

When n = 1, since there is no object generated yet, both the position opt<sup>1</sup> and Num<sup>1</sup> are unrestricted and the LLM can be prompted to determine opt<sup>1</sup> and Num<sup>1</sup> given sub-prompt p<sup>1</sup> . Since the object order starts from left to right and bottom to top, there will be only two position options opt<sup>1</sup> ∈ {left, bottom} for obj<sup>1</sup> . Once opt<sup>1</sup> determined, MuLan evenly splits the whole image's width/height (W/H) to Num<sup>1</sup> parts and assigns the very left (bottom) part to obj<sup>1</sup> , which leads to the following bounding box (an illustration for the computation is shown in Figure [9\)](#page-17-2):

$$M_1 = \begin{cases} (0, 0, \frac{W}{\text{Num}_1}, H), & \text{if opt}_1 = \text{left}, \\ (\frac{(\text{Num}_1-1) \cdot H}{\text{Num}_1}, 0, W, \frac{H}{\text{Num}_1}), & \text{if opt}_1 = \text{bottom}. \end{cases} \quad (5)$$

![](_page_17_Figure_11.jpeg)

Figure 9: Illustration of the rough mask M<sup>1</sup> of obj<sup>1</sup> . There are only two options left, bottom for the mask since the LLM is prompted to plan the object order from left to right, bottom to top.

When n > 1, the position opt<sup>n</sup> denotes {obj}n's relational position to the previous object {obj}n−1. Since MuLan generates objects from left to right and from bottom to top, opt<sup>n</sup> ∈ {right,top}. Given sub-prompt pn, an LLM is prompted to select opt<sup>n</sup> and determine Numn. Meanwhile, the precise mask M˜ <sup>n</sup>−<sup>1</sup> = (˜xn−1, y˜n−1, w˜n−1, h˜ <sup>n</sup>−1) of optn−<sup>1</sup> can be extracted from the image with {obj}n−<sup>1</sup> generated (e.g., by text-image cross-attention maps in the diffusion model), which is utilized as the condition for the computation of bounding box boundary of the rough mask Mn. Hence, the rough mask M<sup>n</sup> for obj<sup>n</sup> can be derived from optn, Numn, and

**979**

**989 990 991**

**994**

**1011**

**1014 1015**

**1017**

M˜ <sup>n</sup>−<sup>1</sup> as followings.

$$M_n = \begin{cases} (\tilde{x}_{n-1} + \tilde{w}_{n-1}, 0, \frac{W - \tilde{x}_{n-1} + \tilde{w}_{n-1}}{\text{Num}_n}, H), & \text{if opt}_n = \text{right}, \\ (0, \frac{\tilde{y}_{n-1} \cdot (\text{Num}_n - 1)}{\text{Num}_n}, W, \frac{\tilde{y}_{n-1}}{\text{Num}_n}), & \text{if opt}_n = \text{top}. \end{cases} \quad (6)$$

Figure [10](#page-18-4) illustrates how the rough mask can be computed based on the precise mask of previous objects.

![](_page_18_Figure_4.jpeg)

Figure 10: The rough mask <sup>M</sup><sup>n</sup> of objn(n > 1) is derived from the precise mask <sup>M</sup>˜ <sup>n</sup>−<sup>1</sup> of the previously generated object objn−<sup>1</sup> .

#### L MORE DETAILS ON THE OVERLAPPING PROCESSING

Given opt<sup>n</sup> and <sup>M</sup>˜ <sup>n</sup>−1, the rough mask <sup>M</sup>n,i can be computed as

$$M_{n,i} = \begin{cases} \left( \tilde{x}_{n-1} \cdot r_i + (\tilde{x}_{n-1} + \tilde{w}_{n-1}) \cdot (1 - r_i), \tilde{y}_{n-1}, \tilde{w}_{n-1} \cdot r_i + \frac{W - \tilde{x}_{n-1} - \tilde{w}_{n-1}}{\text{Num}_n}, \tilde{h}_{n-1} \right), \\ \text{if opt}_n = \text{right}, \\ \left( \tilde{x}_{n-1}, \frac{(\text{Num}_n - 1) \cdot \tilde{y}_{n-1}}{\text{Num}_n}, \tilde{w}_{n-1}, \tilde{h}_{n-1} \cdot r_i + \frac{\tilde{y}_{n-1}}{\text{Num}_n} \right), \\ \text{if opt}_n = \text{top}. \end{cases} \quad (7)$$

The illustration for different overlapping ratios is shown in Figure [11.](#page-18-0)

![](_page_18_Figure_10.jpeg)

Figure 11: Three candidate masks Mn,i of obj<sup>n</sup> at position opt<sup>n</sup> = top. They correspond to obj<sup>n</sup> overlapping with 10%, 30%, and 50% of objn−<sup>1</sup> .

### M MORE DETAILS ON THE EVALUATION QUESTIONNAIRE

As shown in Section [4,](#page-6-1) we design a questionnaire to comprehensively evaluate the alignment between the generated image and the text by GPT-4V [\(OpenAI, 2023\)](#page-10-8) and human, from three aspects

- object completeness, correctness of attribute bindings, and correctness of spatial relationships. Specifically, given an image and a text prompt, for object completeness, we will evaluate if the image contains each single object in the prompt. If the object appears in the image, we will then judge if the attribute bindings of the object in the image align with the corresponding attribute bindings in the text prompt, to evaluate the correctness of attribute bindings. We will also ask GPT-4V or human to judge if the spatial relationships are correct and match the text, as the evaluation of the spatial relationships.

Examples of the questionnaire for different images and text prompts are shown in Figure [12.](#page-19-1)

![](_page_19_Picture_3.jpeg)

Figure 12: Illustration of the questionnaire for the evaluation of generated images

# N LIMITATIONS

Inference time of MuLan Since MuLan generates objects in a progressive manner, it will take longer time than one-stage methods. However, there is a tradeoff between accuracy and efficiency. Most existing one-stage methods generally fail on the complex prompts we focus on. We aim to

**1099**

**1104**

**1106**

**1109**

**1119**

accurately and precisely control the generation process by the proposed progressive pipeline. To show the tradeoff more clearly, we conducted experimental comparisons on how the image-prompt alignment and inference time would vary with the increasing number of objects. As shown in the visualization results of Figure [13,](#page-20-0) although the inference time of MuLan increases with more objects, the image-prompt alignment can be maintained. In one stage methods (e.g., SDXL [\(Podell et al.,](#page-10-5) [2023\)](#page-10-5), PixArt-α [\(Chen et al., 2023\)](#page-9-4)), however, the alignment with prompt becomes worse and worse with more objects.

![](_page_20_Figure_2.jpeg)

Figure 13: The inference time of MuLan and one-stage methods. The prompts are 'a cute kitten', 'the orange pumpkin is on the right of the black door', 'A blue refrigerator on the left, a green chair on the middle, and a yellow table on the right', and 'From left to right, an indoor room with a cute kitten sitting on top of a blue fridge, a black dog sitting on top of a green chair, and a cute kid', respectively. For one object, MuLan reduces to the utilized base diffusion model(e.g., SDXL [\(Podell et al., 2023\)](#page-10-5)). For two or more objects, although MuLan requires more inference time, the image-prompt alignment can be maintained and controlled. This is a tradeoff between accuracy and efficiency. One-stage methods, however, generate worse and worse results with increasing objects.

Also, the inference time of MuLan is not linearly increasing with the number of objects.

![](_page_20_Picture_6.jpeg)

Figure 14: Possible failure case. In some noncommon corner cases of image composition, like 'in a bathroom, a dinosaur is sitting in a sink', base diffusion models may fail to figure out relative size and practical scenes of objects, making generated images unnatural, as shown in the figure.

If the base model used in MuLan is powerful enough, several objects can be generated simultaneously in one stage, further reducing the inference time.

Possible failure cases Note that since Mu-Lan is totally training-free, the generation capability highly depends on the off-the-shelf base model such as stable diffusion in MuLan. We discuss two more cases here. First, for those non-common single object that base model itself cannot generate, it is hard for base models to generate even a single object. In this case, MuLan also cannot generate correct objects. Secondly, for those non-common corner cases of image composition, such as the prompt 'in a bathroom, a huge dinosaur is sitting in a sink', MuLan may also fail to correctly generate them, as shown in Figure [14.](#page-20-1) The reason may be that for these cases, diffusion models cannot figure out reasonable relative size and practical scenes for them.

# O MORE QUALITATIVE RESULTS

We show more examples of different methods in Figure [15.](#page-21-1)

Figure 15: More qualitative examples of images generated by different methods on intricate prompts.