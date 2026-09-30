# Get RICH or Die Scaling: Profitably Trading Inference Compute for Robustness

Anonymous Author(s) Affiliation Address email

## Abstract

 Recent work shows that increasing inference-time compute through generation of long reasoning traces improves not just capability scores, but robustness to various text jailbreaks designed to control models or lower their guardrails. However, multimodal reasoning offers comparatively little defense against vision jailbreaks, which typically succeed by creating noise-like perturbations. When attacking a robust model, vision attacks are also capable of and often must resort to producing human-interpretable perturbations. Rather than operating in a model's blind-spot or out of its training distribution, such interpretable attacks construct familiar concepts connected to the attacker's goal. Inspired by the ability of robust models to force attacks into this space that appears more in-distribution for reasoning tasks, we posit the Robustness from Inference Compute Hypothesis (RICH): defending against attacks with inference compute (like reasoning) profits as those attacks become more in-distribution. To test this, we adversarially attack models of varying robustness with black-box-transfer and white-box attacks. RICH predicts a rich- get-richer dynamic: models that start with higher initial robustness gain more robustness benefits from increases in inference-time compute. Consistent with RICH, we find that robust models benefit more from increased compute, whereas non-robust models show little to no improvement. Our work suggests that inference- time compute can be an effective defense against adversarial attacks, provided the base model has some degree of robustness. In particular, layering disparate train-time and test-time defenses aids robustness not additively, but synergistically.

![](_page_0_Figure_4.jpeg)

Figure 1: As model robustness increases, the benefits of inference-time compute on robustness also increase. A red dot indicates the step at which the model first generates the output targeted by the PGD attack. Robustness increases from LLaVA-v1.5 to FARE-LLaVA-v1.5 to Delta2LLaVA-v1.5.

## 1 Introduction

 Foundation models have grown increasingly capable with the scaling of their pretraining and post- training [\[Kaplan et al., 2020,](#page-9-0) [Hoffmann et al., 2022,](#page-9-1) [Sardana and Frankle, 2023\]](#page-9-2). More recently, inference-time compute scaling to produce long reasoning trajectories has proved capable of generat- ing human-expert-level performances on various benchmarks [\[OpenAI et al., 2024,](#page-9-3) [OpenAI, 2025,](#page-9-4) [Guo et al., 2025,](#page-10-0) [DeepMind, 2025,](#page-10-1) [Anthropic, 2025\]](#page-10-2). However, despite these advances, adversarial robustness remains an open challenge, particularly in safety-critical applications such as autonomous driving. Neural networks are known to be vulnerable to carefully designed inputs that can subvert their intended behavior, bypass guardrails, or generate harmful output [\[Szegedy et al., 2013\]](#page-10-3). Solving this challenge is the key to a successful deployment of AI in real-world applications.

 Recent work by [Zaremba et al.](#page-10-4) [\[2025\]](#page-10-4) represents an exciting direction, showing that inference-time compute offers an intriguing dual benefit: not only does it improve task performance, but it also enhances robustness to text jailbreaks. However, we found that this benefit does not extend cleanly to the vision domain (see Figure [2\)](#page-2-0). Multimodal reasoning [\[Liu et al., 2023,](#page-10-5) [Zaremba et al., 2025\]](#page-10-4), while effective at tasks like visual question answering, offers comparatively little defense against vision jailbreaks, which typically succeed by introducing noise-like perturbations that remain uninterpretable to both humans and models. These perturbations frequently occur in underexplored or off-distribution regions of the input space, where noise-like distortions mislead the model or confuse its semantic understanding of the image, making additional computation at test time only marginally effective.

 A separate line of work on adversarially trained image classification models and vision-language mod- els has shown that increasing robustness of the model alters the nature of adversarial attacks: rather than remaining imperceptible or noise-like, attacks become visually interpretable and often resemble semantically meaningful concepts (e.g., textures, patterns, or objects aligned with the attacker's objective) [\[Gaziv et al., 2023,](#page-10-6) [Bartoldson et al., 2024,](#page-10-7) [Wang et al., 2025,](#page-10-8) [Fort and Lakshminarayanan,](#page-10-9) [2024\]](#page-10-9). Appearing as everyday objects, these interpretable perturbations may be closer to the model's training distribution, and we suspect they may thus be more amenable to reasoning-based defenses – particularly those implemented by increased inference-time compute.

 Inspired by this observation, we introduce the Robustness from Inference Compute Hypothesis (RICH): inference-time compute (e.g., long reasoning traces) is most effective as a defense when attacks are forced into in-distribution regimes understandable by the model. In other words, inference- compute-based defenses work best when the model is already somewhat robust, and thus able to push attackers into a domain where test-time reasoning is effective.

 RICH predicts a "rich-get-richer" dynamic: models that begin with higher baseline robustness gain disproportionately more robustness benefits from additional inference-time compute. In contrast, non-robust models, which remain vulnerable to out-of-distribution (OOD) perturbations, see compute scaling provide little to no defense against attacks that easily generate data that is OOD for the model.

 To test this hypothesis, we conduct adversarial evaluations of VLMs with varying degrees of robust- ness using both white-box and black-box-transfer attacks. We systematically vary inference-time compute and analyze how its defense benefits scale as a function of base robustness. As shown in Figure [1,](#page-0-0) more robust models exhibit increased resistance as compute scales, with attacks requiring more steps or exhibiting reduced success rates. Conversely, non-robust models are comparatively brittle regardless of the amount of reasoning at test time.

 These findings demonstrate that inference-time compute and train-time defenses interact not additively but synergistically: together they provide greater robustness than either alone. The contributions of this work are as follows:

 1. We propose a hypothesis that explains prior failures of inference-time compute to signifi- cantly boost robustness to vision attacks, and which suggests that these failures could be addressed by using more robust base models. 2. We test our hypothesis using attacks from prior work and novel attacks. Our novel white-box vision attack is the first white-box attack used to test the multimodal robustness benefits of scaling inference-time compute, to the best of our knowledge. 3. Consistent with our hypothesis, we demonstrate that inference-time compute provides larger benefits when the base model is more robust. This result clarifies how to improve robustness in exchange for inference-time compute, with a better rate of return.

![](_page_2_Figure_0.jpeg)

Figure 2: Frontier models with inference-time compute defenses are less robust than adversarially trained VLMs on vision attacks. Using Attack-Bard data [\[Dong et al., 2023\]](#page-10-10), we show model accuracy on clean (left) and adversarial (right) data, evaluating under low and high inference-time compute settings. Moreover, for LLaVA-v1.5, a non-robust model, increased inference-time compute does not necessarily provide benefits, consistent with the fact that reasoning on top of a corrupted image understanding is not beneficial. See Figure [3](#page-3-0) for Attack-Bard image descriptions from the VLMs we study, and see Section [3.2](#page-5-0) for experiment details.

## 2 Background and Exploratory Findings

[Zaremba et al.](#page-10-4) [\[2025\]](#page-10-4) found that scaling inference-time compute defends against adversarial attacks, driving attack success rates towards zero for many settings. However, this inference-time scaling seems to fall short for vision attacks defended against via multimodal reasoning. Indeed, as shown in Figure [2,](#page-2-0) the accuracy of o1-v on clean images – i.e., data that isn't attacked – in a low-compute setting (left panel) is not able to be reached on attacked images (right panel), even when using the highest level of inference-time compute.

 Given the high economic cost of raising inference-time compute to such levels, this o1-v result of Figure [2](#page-2-0) suggests that inference-time compute may be a prohibitively expensive defense strategy. Indeed, these images are affected only by static black-box attacks optimized for a separate model [\[Dong et al., 2023\]](#page-10-10) – white-box vision attacks on o1-v itself would be much more difficult to defend against and could pose an insurmountable financial burden if addressed via reasoning. Moreover, [Zaremba et al.](#page-10-4) [\[2025\]](#page-10-4) leaves unclear whether reasoning can even defend against white-box vision attacks (studying only black-box vision attacks) and notes that enhancing robustness to vision adversarial attacks remains an important area for future research.

 In this paper, we aim to clarify whether inference-time compute scaling can be a cost-effective defense to attacks of various strengths, and (further) how such scaling might be improved. Our experiments focus on vision attacks and thus multimodal reasoning. Our initial testing in Figure [2](#page-2-0) shows that Delta2LLaVA-v1.5 [Wang et al.](#page-10-8) [\[2025\]](#page-10-8) – a highly adversarially robust model (RM) – does not require any inference-time compute scaling to outperform the robustness of o1-v at its highest inference-time compute level (see Section [3.2](#page-5-0) for experiment details). This further calls into question whether inference-compute scaling as a defense is worth its price.

 Interestingly, we also see that non-robust models (LLaVA-v1.5 in Figure [2\)](#page-2-0) may fail to benefit from scaling of inference-compute on attacked data, even when they receive benefits on clean data. This negative result may not be surprising, as attacks may leverage a model's inability to operate correctly on data outside its training distribution, and it is not clear that adding more reasoning would be able to facilitate the removal of such an attack's effects. In other words, while the noise-like pattern such attacks have may be negligible to a human – see Figure [3](#page-3-0) for an example of an adversarial Attack-Bard image that humans can understand – it could nonetheless represent a significant shift away from the training distribution of the model. Corroborating this, Figure [3](#page-3-0) shows that LLaVA-v1.5 produces an image description that is completely unrelated to the target ("American Coot"), which will prevent reasoning from providing a benefit regardless of the inference-time compute level.

 Finally, it is necessary to consider the potential ability of inference-time compute scaling as a defense on a class of vision adversarial attacks that recent literature has highlighted for its effectiveness against models with state-of-the-art robustness. These attacks do not appear as noisy versions of their base images; i.e., they depart from the pattern in Figure [3.](#page-3-0) Rather than producing data that appears outside

| Model           | Description                                                                              | Prediction (Low) | Prediction (High) | Clean                                                                        |
|-----------------|------------------------------------------------------------------------------------------|------------------|-------------------|------------------------------------------------------------------------------|
| LLaVA-1.5       | The image is a colorful abstract representation of a person swimming in the ocean...     | seashore         | seashore          | <img alt="A person swimming in the ocean." data-bbox="750 100 900 250"/>     |
| FARE-LLaVA -1.5 | The image features a black duck swimming in a body of water possibly a lake or a pond... | drake            | drake             | Attacked                                                                     |
| Delta2LLaVA-1.5 | The image features a black bird possibly a duck swimming in a body of water...           | redshank         | American Coot     | <img alt="A bird swimming in a body of water." data-bbox="750 250 900 400"/> |

Figure 3: Example model behavior under black-box attack. We show models' image descriptions and associated predictions for an attacked image of an "American Coot" from the Attack-Bard dataset [\[Dong et al., 2023\]](#page-10-10).

 the training distribution, these attacks produce semantically interpretable features in the attacked images [\[Gaziv et al., 2023,](#page-10-6) [Bartoldson et al., 2024,](#page-10-7) [Wang et al., 2025,](#page-10-8) [Fort and Lakshminarayanan,](#page-10-9) [2024\]](#page-10-9). Prior work shows that such adversarial images are produced when attacking sufficiently robust networks: intuitively, if a robustified model cannot be attacked through subtle perturbations, then visually instantiating the attacker's target can become a less-difficult path towards attack success. We reproduce this finding in Figure [4,](#page-4-0) constructing a version of these attacks that is novel to the best of our knowledge.

 Specifically, Figure [4](#page-4-0) shows for the first time that, not only can this type of attack alter a shape from spherical to cuboid, but the extent of the alteration needed for attack success is increased by the addition of scaled inference-time compute. This attack setup is analogous to real world settings, which [i](#page-10-4)nclude following safety specifications regardless of adversarially persuasive user inputs [\[Zaremba](#page-10-4) [et al., 2025\]](#page-10-4), and self-driving of vehicles using multimodal systems that must understand the need to prioritize specific modalities (e.g., a human driver's command to stop should override a model's potentially hijacked understanding of the visual scene). See Figure [9](#page-17-0) for an illustration of an attack we produce in a setting directly relevant to self-driving vehicles.

 Critical to our work, the aforementioned interpretable-attack strategy suggests that, when adversar- ially attacking robust models, the attacked data may not go outside (or as far outside) the training distribution as it does when attacking less-robust models. Instead, attacks on robust models may rely on their ability to produce responses based on an accurate understanding of the attacked data. If this is the case, then reasoning may become more relevant as a defense. In other words, robust models may ease the problem of applying safety specifications to attacked inputs [\[Zaremba et al., 2025\]](#page-10-4) by making reasoning about the relationship of those inputs to the specifications easier. In sum, robust models may reduce the problem of providing a robust response to one that is solvable by reasoning of the type that the model was trained to do, whereas it's unclear that reasoning on out-of-distribution representations from non-robust models has a chance to succeed.

 These exploratory findings and subsequent analyses motivate the following hypothesis, which we validate via rigorous testing in the remainder of this work.

The Robustness from Inference Compute Hypothesis. *Inference-time compute is most effective as a defense when attacks are forced into in-distribution regimes understandable by the model.*

![](_page_4_Picture_0.jpeg)

![](_page_4_Figure_1.jpeg)

Figure 4: Attacking highly robust models, especially when they have scaled inference-time compute, causes visual instantiation of an instance of the attacker's target text in the attacked image. The image is modified by the attacker until the target text "Cube" is output by the model. We show the attacked images and model attention maps. When K >= 1, the prompt text in brown is included, and the portion in braces is repeated K times to naively increase inference-time compute.

## 3 Methodology

 We study how inference-time compute provides robustness to adversarial visual jailbreaks by testing VLMs with varying robustness levels (low, medium, and high): LLaVA-v1.5 [\[Liu et al., 2024\]](#page-10-11), FARE- [L](#page-10-4)LaVA-v1.5 [\[Schlarmann et al., 2024\]](#page-10-12), and Delta2LLaVA-v1.5 [\[Wang et al., 2025\]](#page-10-8). While [Zaremba](#page-10-4) [et al.](#page-10-4) [\[2025\]](#page-10-4) consider a non-robust reasoning model, our approach makes explicit the potential effect of robust vision representations, or the lack thereof, on measuring the benefits of reasoning defenses.

 We adopt LLaVA-v1.5 as our baseline VLM. While this model operates with a strong connection between the visual and text domains, due to its visual-instruction tuning, it is not robust to adver- sarial image attacks as neither its image encoder nor its language model experienced adversarial training. Contrast this with FARE-LLaVA-v1.5 which replaces the frozen CLIP image encoder with a robust version achieved through unsupervised adversarial finetuning on ImageNet. Finally, Delta2LLaVA-v1.5 adds two levels of defense: full, web-scale adversarial contrastive CLIP pretrain- ing and adversarial visual instruction tuning. Increased adversarial training yields strong benefits to performance. For example[,Wang et al.](#page-10-8) [\[2025\]](#page-10-8) report that when comparing LLaVAs on a task requiring visual reasoning like VQAv2 [\[Goyal et al., 2017\]](#page-10-13), Delta2LLaVA-v1.5 achieves 59.5% accuracy under a ℓ<sup>∞</sup> ε = 4/255 attack while FARE-LLaVA-v1.5 reaches 31% and non-robust LLaVA-v1.5 obtains 0%. For our FARE-LLaVA-v1.5 experiments, we use the FARE-CLIP encoder finetuned with ε = 2/255 under the ℓ<sup>∞</sup> norm.

## 3.1 White-box PGD Attack

 We evaluate VLM robustness under a novel white-box adversarial attack. Our attack creates a conflict between modalities by providing correct information in the text input (e.g., mentioning that a soccer ball is "round" as shown in Figure [4\)](#page-4-0) while simultaneously applying a PGD attack on the image input that targets an incorrect model output (e.g., "Cube").

 Inference-time Compute Scaling To investigate how additional inference-time compute affects robustness, we use textual repetition to raise computational effort. Specifically, we repeat the correct text description K times in the instruction prompt, and we explicitly instruct the model to defer to the text modality when the text and vision inputs conflict. Higher K represents increased levels of inference-time compute. Notably, this is not the same inference-time compute scaling performed by reasoning models like o1, but it allows us to investigate how naively scaling inference-time compute affects robustness. While our black-box experiments provide closer proxies for prior work with closed models on scaling reasoning for robustness [\[Zaremba et al., 2025\]](#page-10-4) – see Section [4.4](#page-8-0) – we expect our novel white-box methodology to provide a strong test of inference-time compute's robustness benefits. In particular, scaling K may make the model more inclined to defer to the answer given in the text input; i.e., the probability of the model calling a ball "red" is expected to increase with the number of in-context statements describing the ball with this color, consistent with patterns found in the model's training data. This increased evidence for choosing a particular value through scaling K can be seen as proxying for the ability of state-of-the-art reasoning systems to produce increasing amounts of evidence for choosing a particular value through a reasoning trace.

 Attack Details For each attack instance, we run a PGD attack with step size 0.1 for 100 iterations, using a perturbation budget ε ∈ {16/255, 64/255}. At each step, we track both the cross-entropy loss of the target tokens and whether the model generates the target response. We record the minimum number of PGD steps required for successful attack (lower values indicate lower robustness). The attack is considered failed if the model does not generate the target response after all 100 steps. Experiments were conducted using a single NVIDIA 80GB H100 GPU.

## 3.2 Black-Box Transfer Attacks

 We also test RICH on a dataset of transferred, black-box adversarial examples using an image classification task. Attack-Bard consists of 200 images generated from a white-box adversarial attack on an ensemble of surrogate models [\[Dong et al., 2023\]](#page-10-10). These images were optimized for transfer to Bard and GPT-4V with ε = 16/255 under the ℓ<sup>∞</sup> norm. The clean counterparts to these 200 images are used to measure the baseline strength of each model's visual perception and the benefits of adaptive inference-time compute on classifying natural images.

 Attack-Bard with Augmented Reasoning We evaluate each VLM for its classification accuracy on Attack-Bard, under low and high inference-time compute settings. We apply each model to predict the class label of an input image using its multimodal context —the image pixels and the instruction prompt. As the VLMs surveyed have moderate instruction-following capabilities and struggle on their own to classify an image when prompted with the full label set, we augment each VLM with adaptive inference-time compute and predict the label in two stages. First, we prompt the VLM to provide a description for each image. Then using this description, we apply Claude 3.7 Sonnet to judge which label best matches the generated description [\[Anthropic, 2025\]](#page-10-2). Using the "extended thinking" feature of the judge, we create low and high inference-time compute settings. Both the low and high inference-time compute settings use a temperature of 1 and set the max number of tokens generated to 20,000. The high inference-time compute setting uses a budget of 16,000 thinking tokens. Details on the Claude prompts used can be found in Appendix [B.1.](#page-13-0)

 Attack-Bard with Chain of Thought Additionally, we leverage Attack-Bard to examine black-box attack success when the VLM's intrinsic reasoning capabilities are invoked through chain of Thought (CoT) prompting techniques [\[Wei et al., 2022,](#page-10-14) [Kojima et al., 2022,](#page-10-15) [Wang et al., 2022\]](#page-10-16). This setup does not use an external judge and instead asks the model to classify the image with varying degrees of intermediate reasoning. For each image, we construct a multiple choice question including the true label and 29 other answers chosen from the label set at random. We devise a low inference-time compute, no CoT, setting where the model is prompted to select the correct label from the provided choices. In the high inference compute regime, we apply CoT reasoning to elicit classification from step-by-step thinking. Image labels were generated from the VLM using greedy sampling with 0 temperature generating a maximum of 5 and 500 tokens for the low and high respective settings. Details on the CoT prompts can be found in [B.1.](#page-13-0) Experiments were conducted using a single 80GB Nvidia H100 GPU.

![](_page_6_Figure_0.jpeg)

Figure 5: When ε is sufficiently high at 64/255, only the most robust model benefits significantly from inference-time compute. Robustness increases from LLaVA-v1.5 to FARE-LLaVA-v1.5 to Delta2LLaVA-v1.5.

## 4 Experiments

## 4.1 Does Inference-Compute Scaling Help All Models Equally?

 As [Zaremba et al.](#page-10-4) [\[2025\]](#page-10-4) only studied one model, it's unclear if scaling inference-compute provides the same benefits regardless of the base model (and its robustness). E.g., a constant benefit might be expected if reasoning aids defense by making attack optimization more complex. Alternatively, RICH suggests that reasoning's robustness benefits depend on the base model's robustness.

 To test this, we use white-box PGD attacks on models with increasing levels of adversarial robustness. If RICH is correct, we would expect to see robust models are harder to attack at a given inference- time compute level, relative to less robust models. Alternatively, if the benefits of scaling inference compute are unrelated to the model, we would expect that there's no relationship between a base model's robustness and the benefits it obtains from scaling inference compute.

 Figure [5](#page-6-0) shows the PGD attack loss curves for VLMs with increasing inference-compute levels when ε = 64/255. It is found that the loss for the most robust model (Delta2LLaVA-v1.5) has a substantial rise when the compute level rises, leading to substantially increased numbers of PGD steps to break the model. In contrast, models with lower robustness do not exhibit such changes. This observation is consistent with RICH. Specifically, the benefits of scaling inference compute depend on the robustness of the model.

Does Inference-Compute Scaling Help All Models Equally? No, we find that inferencecompute scaling benefits robustness more when the model is initially more robust.

#### 4.2 Can Inference-Compute Scaling Only Benefit Robustness in Select Models?

 We have seen that the benefits of scaling inference-time compute depend on the model. However, it remains unclear why this is the case. One possibility is that only Delta2-LLaVA-1.5 benefits notably because it was visually instruction tuned while under adversarial attacks [\[Wang et al., 2025\]](#page-10-8). Indeed, FARE had comparatively light adversarial training that only fine-tuned the vision embedding model [\[Schlarmann et al., 2024\]](#page-10-12). Thus, we may expect that only Delta2-LLaVA-1.5 can significantly benefit from inference-time compute scaling in our setup because it was the only model trained to perform multimodal reasoning when under attack.

 Alternatively, reasoning may be able to support robustness as long as the data being reasoned about is close enough to being in-distribution. We might expect this to be the case if, for example, inference-time compute scaling boosts defenses by enhancing the model's ability to perform correct classification given an accurate representation of the image, and if less robust models are capable of providing accurate representations of attacked images as long as the perturbation is small enough.

 To test this, we used a smaller perturbation budget ε = 16/255, bringing the attacked images closer to the distribution the model was trained on. If reasoning relies on in-distribution data to provide benefits, we would expect to see scaling providing benefits to less adversarially trained models as ε

![](_page_7_Figure_0.jpeg)

Figure 6: Robust models benefits from inference-compute scaling when attacked image is indistribution. PGD steps required for successful attacks with increasing inference-time compute levels and variations in perturbation strength. Failed attacks are marked by black circles.

Table 1: PGD steps required for a successful attack across models, perturbation budget ε, and inference-compute levels K. Mean (standard error) computed on three attack variations of an image.

| ε K 0 16/255 1 3 | 4.3 5.3 6.3 | LLaVA-v1.5 (0.7) (1.5) (1.2) | 8.0 20.0 23.3 | FARE-LLaVA-v1.5 (2.2) (6.1) (8.5) | Attack Attack Attack | Delta2LLaVA-v1.5 Failed Failed Failed |
|------------------|-------------|------------------------------|---------------|-----------------------------------|----------------------|---------------------------------------|
| 0                | 5.7         | (1.4)                        | 7.7           | (3.0)                             | 13.7                 | (5.9)                                 |
| 1                | 6.0         | (1.6)                        | 7.3           | (1.5)                             | 60.3                 | (18.4)                                |
| 3                | 6.7         | (1.0)                        | 9.7           | (2.6)                             | 67.0                 | (13.8)                                |

 decreases. Alternatively, if adversarial visual instruction tuning [\[Wang et al., 2025\]](#page-10-8) is critical, we would expect no benefits from reasoning when ε is reduced.

 In Figure [6,](#page-7-0) we observe that inference-compute scaling benefits robustness in our setup, even if these models were not explicitly trained to perform multimodal reasoning when under attack. This supports the hypothesis that inference-time compute benefits defenses when the attacks are in-distribution.

Can Inference-Compute Scaling Only Benefit Robustness in Select Models? No. Our experiments suggest that, provided the attacked data is sufficiently close to the model's training distribution, inference-compute scaling can benefit robustness.

#### 4.3 Is the Robustness from Inference Compute Hypothesis Supported Across General Attack Targets and Images?

 To verify our findings, we explored a series of attacks that target different image aspects and base image. We designed variations of our white-box attack setup for color, shape, and material attacks, for the example image and others that include traffic/driving imagery as an example of high safety risk situations. Table [1](#page-7-1) shows averaged PGD steps required for successful attacks across these experiments. We observe that increasing compute level K consistently increase the required PGD steps across all models, with the effect most pronounced in robust models and when attacks are in-distribution at lower ε = 16/255. Delta2LLaVA-v1.5 demonstrates strong improvements under high ε = 64/255 attacks when more compute is added: mean PGD attack steps increase from 13.7 at k = 0 to 67.0 at k = 3, nearly a 5× improvement. Results for each attack can be found in Appendix [C.](#page-16-0) These results provide strong evidence that inference-time compute acts as an effective defense multiplier, especially when models are robust and attacks remain within the training distribution.

Is the RICH Supported Across General Attack Targets and Images? Yes, we corroborate our central hypothesis in experiments across general images and attack targets.

#### 4.4 Does Chain-of-Thought Provide Improved Defenses in Robust Models?

 Prior experiments left two things unclear: (1) is the RICH supported by black-box attacks? It's important to know this because frontier models often do not provide white-box access. (2) What happens when using more traditional reasoning approaches? In particular, earlier experiments do not match traditional inference-time compute scaling approaches with reasoning, using a novel context scaling approach (e.g., see Figure [1\)](#page-0-0) or a separate model for reasoning (i.e., see Figure [2\)](#page-2-0).

Evaluation Model No CoT CoT Clean LLaVA-v1.5 68.5 68.5 Delta2LLaVA-v1.5 57.5 60.5 Adv. LLaVA-v1.5 36.5 37.0 Delta2LLaVA-v1.5 55.0 59.5 Table 2: Classification accuracy on Attack-Bard blackbox transfer attacks for multiple-choice questions and CoT inference-compute scaling Here, we test the dependence of our re- sults on all of the above factors by using our black-box CoT experiment setup. If our white-box attacks are critical to our findings, we would not expect to see sup- port for the Robustness from Inference Compute Hypothesis here. Similarly, if the reasoning must be done by a frontier model or the k-scaling experiment setup is important to our result, we would not expect to find support for RICH. Alter- natively, if the Robustness from Inference Compute Hypothesis is applicable to various inference- compute scaling approaches and adversarial attack settings, we would expect to see that switching from short answers to CoT-based answers provides a benefit primarily to robustified models. Table [2](#page-8-1) shows that our results are consistent with the Robustness from Inference Compute Hypothesis. In particular, when shifting to a setting that more closely proxies for the original inference-compute-scaling-for-robustness setup of [Zaremba et al.](#page-10-4) [\[2025\]](#page-10-4), we still find that the robustness benefits of

| Evaluation Clean | Model LLaVA-v1.5 | No CoT 68.5 | CoT 68.5 |
|------------------|------------------|-------------|----------|
|                  | Delta2LLaVA-v1.5 | 57.5        | 60.5     |
| Adv.             | LLaVA-v1.5       | 36.5        | 37.0     |
|                  | Delta2LLaVA-v1.5 | 55.0        | 59.5     |

inference-time compute scaling improve with base model robustness.

Does Chain-of-Thought Provide Improved Defenses in Robust Models? Yes, the RICH is broadly observed regardless of how inference-time compute is scaled.

## 5 Discussion

 Scaling inference-time compute has been shown to provide many benefits that even extend to increased robustness. Enhancing robustness and other model safety/security capabilities is key to obtaining the trust needed for widespread use and benefits of frontier AI. Prior work found that this robustness benefit of increasing inference-time compute was limited when adversaries used vision attacks. We proposed a hypothesis to explain this limitation as well as how to ensure robustness benefits from inference-time compute scaling in cost-effective manner. Our hypothesis, the Robustness from Inference Compute Hypothesis, was validated through a variety of experiments that include novel white-box and previously explored black-box attacks.

 In Appendix [A,](#page-12-0) we discuss additional related work on out of distribution (OOD) robustness, adversar-ial attacks, and adversarial training.

 Limitations We explored a phenomenon first uncovered in a large-scale reasoning model (o1) using experiments at a comparatively much smaller scale. While our model scale facilitates tests of the most adversarially robust VLMs that we know of [\[Wang et al., 2025\]](#page-10-8), it is necessary to validate our findings at larger scales, which see widespread deployment of models and which pose the largest potential harm when attacks are successful. Towards this, future work could adversarially train larger (possibly frontier-scale) models to test our core hypothesis more broadly.

 Broader Impact As LLM capabilities improve, studying defenses against adversarial attacks that lower their safety guardrails can potentially enhance trust in and benefits of LLM deployment in vari- ous settings. However, automated defenses like scaling inference compute should be complemented by attentive and responsible evaluation/monitoring.

## References


[1] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361*, 2020.

[2] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. *arXiv preprint arXiv:2203.15556*, 2022.

[3] Nikhil Sardana and Jonathan Frankle. Beyond chinchilla-optimal: Accounting for inference in language model scaling laws. *ArXiv*, abs/2401.00448, 2023.

[4] OpenAI, Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally Bennett, Ananya Kumar, Andre Saraiva, Andrea Vallone, Andrew Duberstein, Andrew Kondrich, Andrey Mishchenko, Andy Applebaum, Angela Jiang, Ashvin Nair, Barret Zoph, Behrooz Ghor- bani, Ben Rossen, Benjamin Sokolowsky, Boaz Barak, Bob McGrew, Borys Minaiev, Botao Hao, Bowen Baker, Brandon Houghton, Brandon McKinzie, Brydon Eastman, Camillo Lugaresi, Cary Bassin, Cary Hudson, Chak Ming Li, Charles de Bourcy, Chelsea Voss, Chen Shen, Chong Zhang, Chris Koch, Chris Orsinger, Christopher Hesse, Claudia Fischer, Clive Chan, Dan Roberts, Daniel Kappler, Daniel Levy, Daniel Selsam, David Dohan, David Farhi, David Mely, David Robinson, Dimitris Tsipras, Doug Li, Dragos Oprica, Eben Freeman, Eddie Zhang, Edmund Wong, Elizabeth Proehl, Enoch Cheung, Eric Mitchell, Eric Wallace, Erik Ritter, Evan Mays, Fan Wang, Felipe Petroski Such, Filippo Raso, Florencia Leoni, Foivos Tsimpourlas, Francis Song, Fred von Lohmann, Freddie Sulit, Geoff Salmon, Giambattista Parascandolo, Gildas Chabot, Grace Zhao, Greg Brockman, Guillaume Leclerc, Hadi Salman, Haiming Bao, Hao Sheng, Hart Andrin, Hessam Bagherinezhad, Hongyu Ren, Hunter Lightman, Hyung Won Chung, Ian Kivlichan, Ian O'Connell, Ian Osband, Ignasi Clavera Gilaberte, Ilge Akkaya, Ilya Kostrikov, Ilya Sutskever, Irina Kofman, Jakub Pachocki, James Lennon, Jason Wei, Jean Harb, Jerry Twore, Jiacheng Feng, Jiahui Yu, Jiayi Weng, Jie Tang, Jieqi Yu, Joaquin Quiñonero Candela, Joe Palermo, Joel Parish, Johannes Heidecke, John Hallman, John Rizzo, Jonathan Gordon, Jonathan Uesato, Jonathan Ward, Joost Huizinga, Julie Wang, Kai Chen, Kai Xiao, Karan Singhal, Karina Nguyen, Karl Cobbe, Katy Shi, Kayla Wood, Kendra Rimbach, Keren Gu-Lemberg, Kevin Liu, Kevin Lu, Kevin Stone, Kevin Yu, Lama Ahmad, Lauren Yang, Leo Liu, Leon Maksin, Leyton Ho, Liam Fedus, Lilian Weng, Linden Li, Lindsay McCallum, Lindsey Held, Lorenz Kuhn, Lukas Kondraciuk, Lukasz Kaiser, Luke Metz, Madelaine Boyd, Maja Trebacz, Manas Joglekar, Mark Chen, Marko Tintor, Mason Meyer, Matt Jones, Matt Kaufer, Max Schwarzer, Meghan Shah, Mehmet Yatbaz, Melody Y. Guan, Mengyuan Xu, Mengyuan Yan, Mia Glaese, Mianna Chen, Michael Lampe, Michael Malek, Michele Wang, Michelle Fradin, Mike McClay, Mikhail Pavlov, Miles Wang, Mingxuan Wang, Mira Murati, Mo Bavarian, Mostafa Rohaninejad, Nat McAleese, Neil Chowd- hury, Neil Chowdhury, Nick Ryder, Nikolas Tezak, Noam Brown, Ofir Nachum, Oleg Boiko, Oleg Murk, Olivia Watkins, Patrick Chao, Paul Ashbourne, Pavel Izmailov, Peter Zhokhov, Rachel Dias, Rahul Arora, Randall Lin, Rapha Gontijo Lopes, Raz Gaon, Reah Miyara, Reimar Leike, Renny Hwang, Rhythm Garg, Robin Brown, Roshan James, Rui Shu, Ryan Cheu, Ryan Greene, Saachi Jain, Sam Altman, Sam Toizer, Sam Toyer, Samuel Miserendino, Sandhini Agarwal, Santiago Hernandez, Sasha Baker, Scott McKinney, Scottie Yan, Shengjia Zhao, Shengli Hu, Shibani Santurkar, Shraman Ray Chaudhuri, Shuyuan Zhang, Siyuan Fu, Spencer Papay, Steph Lin, Suchir Balaji, Suvansh Sanjeev, Szymon Sidor, Tal Broda, Aidan Clark, Tao Wang, Taylor Gordon, Ted Sanders, Tejal Patwardhan, Thibault Sottiaux, Thomas Degry, Thomas Dimson, Tianhao Zheng, Timur Garipov, Tom Stasi, Trapit Bansal, Trevor Creech, Troy Peterson, Tyna Eloundou, Valerie Qi, Vineet Kosaraju, Vinnie Monaco, Vitchyr Pong, Vlad Fomenko, Weiyi Zheng, Wenda Zhou, Wes McCabe, Wojciech Zaremba, Yann Dubois, Yinghai Lu, Yining Chen, Young Cha, Yu Bai, Yuchen He, Yuchen Zhang, Yunyun Wang, Zheng Shao, and Zhuohan Li. Openai o1 system card, 2024. URL <https://arxiv.org/abs/2412.16720>.

[5] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025. [D](https://deepmind.google/technologies/gemini/flash-thinking/)eepMind. Gemini 2.0 flash thinking. *Google DeepMind website*, 2025. URL [https://deepmind.](https://deepmind.google/technologies/gemini/flash-thinking/) [google/technologies/gemini/flash-thinking/](https://deepmind.google/technologies/gemini/flash-thinking/). Anthropic. Claude 3.7 sonnet extended thinking. *Anthropic System Card*, 2025. URL [https://assets.anthropic.com/m/785e231869ea8b3b/original/](https://assets.anthropic.com/m/785e231869ea8b3b/original/claude-3-7-sonnet-system-card.pdf) [claude-3-7-sonnet-system-card.pdf](https://assets.anthropic.com/m/785e231869ea8b3b/original/claude-3-7-sonnet-system-card.pdf). Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. *arXiv preprint arXiv:1312.6199*, 2013. Wojciech Zaremba, Evgenia Nitishinskaya, Boaz Barak, Stephanie Lin, Sam Toyer, Yaodong Yu, Rachel Dias, Eric Wallace, Kai Xiao, Johannes Heidecke, et al. Trading inference-time compute for adversarial robustness. *arXiv preprint arXiv:2501.18841*, 2025. Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. *Advances in neural information processing systems*, 36:34892–34916, 2023. Guy Gaziv, Michael J Lee, and James J DiCarlo. Robustified anns reveal wormholes between human category percepts. *arXiv preprint arXiv:2308.06887*, 2023. Brian R Bartoldson, James Diffenderfer, Konstantinos Parasyris, and Bhavya Kailkhura. Adversarial robustness limits via scaling-law and human-alignment studies. *arXiv preprint arXiv:2404.09349*, 2024. Zeyu Wang, Cihang Xie, Brian Bartoldson, and Bhavya Kailkhura. Double visual defense: Adver- sarial pre-training and instruction tuning for improving vision-language model robustness. *arXiv preprint arXiv:2501.09446*, 2025. Stanislav Fort and Balaji Lakshminarayanan. Ensemble everything everywhere: Multi-scale aggrega- tion for adversarial robustness, 2024. URL <https://arxiv.org/abs/2408.05446>. Yinpeng Dong, Huanran Chen, Jiawei Chen, Zhengwei Fang, Xiao Yang, Yichi Zhang, Yu Tian, Hang Su, and Jun Zhu. How robust is google's bard to adversarial image attacks? *arXiv preprint arXiv:2309.11751*, 2023. Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 26296–26306, 2024. Christian Schlarmann, Naman Deep Singh, Francesco Croce, and Matthias Hein. Robust clip: Unsupervised adversarial fine-tuning of vision embeddings for robust large vision-language models. *ICML*, 2024. Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 6904–6913, 2017. Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed H. Chi, F. Xia, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. *ArXiv*, abs/2201.11903, 2022. Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. *Advances in neural information processing systems*, 35: 22199–22213, 2022. Xuezhi Wang, Jason Wei, D. Schuurmans, Quoc Le, Ed H. Chi, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. *ArXiv*, abs/2203.11171, 2022. Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. *arXiv preprint arXiv:1412.6572*, 2014.

[6] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. *arXiv preprint arXiv:1706.06083*, 2017. Francesco Croce, Maksym Andriushchenko, Vikash Sehwag, Edoardo Debenedetti, Nicolas Flam- marion, Mung Chiang, Prateek Mittal, and Matthias Hein. Robustbench: a standardized adversarial robustness benchmark. *arXiv preprint arXiv:2010.09670*, 2020. Stanislav Fort. A note on implementation errors in recent adaptive attacks against multi-resolution self-ensembles, 2025. URL <https://arxiv.org/abs/2501.14496>. Luke Bailey, Euan Ong, Stuart Russell, and Scott Emmons. Image hijacks: Adversarial images can control generative models at runtime. *arXiv preprint arXiv:2309.00236*, 2023. Stanislav Fort, Jie Ren, and Balaji Lakshminarayanan. Exploring the limits of out-of-distribution detection. In *Proceedings of the 35th International Conference on Neural Information Processing Systems*, NIPS '21, Red Hook, NY, USA, 2021. Curran Associates Inc. ISBN 9781713845393. Stanislav Fort. Adversarial vulnerability of powerful near out-of-distribution detection, 2022. URL <https://arxiv.org/abs/2201.07012>. Nikolaus Howe, Ian McKenzie, Oskar Hollinsworth, Michał Zajac, Tom Tseng, Aaron Tucker, Pierre- Luc Bacon, and Adam Gleave. Effects of scale on language model robustness. *arXiv preprint arXiv:2407.18213*, 2024.
## A Additional Related Work

 Model performance degrades when data is adversarially perturbed by attacks that humans are robust to [\[Szegedy et al., 2013\]](#page-10-3). Adversarial training [\[Goodfellow et al., 2014,](#page-10-17) [Madry et al., 2017\]](#page-11-0) can help improve model robustness. However, according to RobustBench [\[Croce et al., 2020\]](#page-11-1), the robustness problem is still unsolved on toy datasets like CIFAR-10. Recent work predicts that an alternative paradigm is needed, as scaling adversarial training may be a computationally infeasible solution [\[Bartoldson et al., 2024\]](#page-10-7).

 Additionally, [Bartoldson et al.](#page-10-7) [\[2024\]](#page-10-7) showed that attacking a robustified network by altering an image of a horse would lead to the labeling of the image as a dog by the robustified network *and by humans*; i.e., the image was altered to resemble a dog rather than affected by noise-like patterns that humans can ignore. Additional corroboration in [Fort](#page-11-2) [\[2025\]](#page-11-2) is shown for a model that was not directly adversarially trained, suggesting that robustness itself rather than a means of achieving it might be directly linked to interpretability.

 We show that this phenomenon plays a major role in how inference-time attacks and defenses play out for multimodal reasoning models. Specifically, we find that PGD attacks [\[Madry et al., 2017\]](#page-11-0) on VLMs like LLaVA 1.5 [\[Liu et al., 2024\]](#page-10-11) proceed by generating noise-like patterns and quickly achieve their targeted output [\[Bailey et al., 2023\]](#page-11-3). However, attacks on robustified VLMs like Delta2LLaVA-v1.5 [\[Wang et al., 2025\]](#page-10-8) produce semantically interpretable features and struggle to successfully achieve the targeted model output. An example of this is shown in Figure [4.](#page-4-0)

 This suggests that, while adversarial attacks are typically seen as producing out-of-distribution (OOD) data, this is not necessarily the case (especially in robust models). Our core hypothesis (RICH) suggests that reasoning can provide larger robustness benefits when model attacks are not OOD. In this work, we explored the concept of adversarially robust models' ability to keep adversarial attacks in distribution, but our hypothesis may relate to other ways in which OOD can arise. However, while OOD robustness can be enhanced by scaling [Fort et al.](#page-11-4) [\[2021\]](#page-11-4), even the most robust near-OOD detectors are still brittle to targeted adversarial attacks [\[Fort, 2022\]](#page-11-5), accenting that robustness against distribution shifts and robustness against active attacks might be different kinds of phenomena.

 Importantly, [Howe et al.](#page-11-6) [\[2024\]](#page-11-6) found that larger LLMs tend to be more robust, despite not being adversarially trained. Thus, it's possible that the relatively strong defensive benefit of scaling inference-time compute on text input attacks is consistent with the Robustness from Inference Compute Hypothesis. Specifically, inference-time compute may have benefitted LLM robustness in [Zaremba et al.](#page-10-4) [\[2025\]](#page-10-4) because the frontier-scale model being tested was already somewhat robust to the text attacks considered, even though that model had no explicit training to encourage robustness to attacks.

## <sup>468</sup> B Experiment Details

#### <sup>469</sup> B.1 Vision Language Model Instructions

<sup>470</sup> All image classification requests to Claude 3.7 Sonnet use the image description generated by the <sup>471</sup> queried VLM and take the following form:

Claude 3.7 Sonnet Image Classification

The following is an image description: {Image description from VLM} please tell me the category that best applies to the image description. You must pick from the following categories, and return to me just one category from this list (e.g., just reply "yurt"). I want you to respond with only the category so i can paste your response into a CSV column to check to see if it matches a ground truth. categories: african crocodile, airliner, alp, american alligator, american coot, analog clock, ant, bagel, bakery, bald eagle, ballplayer, bannister, barbell, barn, basenji, basketball, beach wagon, bearskin, bee, beer glass, bell cote, bobsled, bow tie, brass, bubble, buckeye, buckle, burrito, cab, candle, cannon, canoe, car mirror, car wheel, carbonara, carousel, carton, cash machine, castle, category, centipede, cheeseburger, church, cinema, cliff, container ship, convertible, coral reef, cornet, crane, crash helmet, crock pot, dishrag, dome, dough, drake, dung beetle, dutch oven, espresso, fire engine, fly, football helmet, freight car, garter snake, gasmask, gazelle, geyser, giant panda, gondola, gorilla, grand piano, granny smith, grasshopper, greenhouse, grille, grocery store, groom, hog, hummingbird, indian elephant, ipod, jackolantern, jay, jeep, jellyfish, kelpie, lampshade, library, loggerhead, longhorned beetle, lorikeet, lycaenid, mailbox, manhole cover, mantis, marmot, matchstick, megalith, menu, military uniform, minivan, monarch, monastery, mountain tent, organ, ostrich, otter, palace, parachute, park bench, payphone, pedestal, pier, pizza, plate, pole, pot, prison, racket, rapeseed, redbacked sandpiper, redshank, reflex camera, refrigerator, restaurant, rugby ball, running shoe, sarong, scabbard, seashore, seat belt, slug, snail, soccer ball, soup bowl, speedboat, spider web, stage, steel arch bridge, stone wall, street sign, suspension bridge, tank, thatch, theater curtain, throne, tile roof, toaster, toyshop, trench coat, triumphal arch, trombone, turnstile, umbrella, upright, vulture, wallet, washer, water buffalo, weevil, wool, worm fence, yurt"

<sup>473</sup> Example low inference-time compute prompt with multiple-choice question for the giant panda <sup>474</sup> image. The model is instructed to make a classification directly without additional reasoning.

| Low  | Inference-Time Compute (No CoT)                                            |
|------|----------------------------------------------------------------------------|
| The  | image is described by one of the following labels:                         |
| (1)  | giant panda                                                                |
| (2)  | basenji                                                                    |
| (3)  | mantis                                                                     |
| (4)  | dome                                                                       |
| (5)  | organ                                                                      |
| (6)  | car wheel                                                                  |
| (7)  | carbonara                                                                  |
| (8)  | upright                                                                    |
| (9)  | buckle                                                                     |
| (10) | container ship                                                             |
| (11) | barbell                                                                    |
| (12) | thatch                                                                     |
| (13) | football helmet                                                            |
| (14) | snail                                                                      |
| (15) | cornet                                                                     |
| (16) | freight car                                                                |
| (17) | hog                                                                        |
| (18) | Dutch oven                                                                 |
| (19) | bubble                                                                     |
| (20) | bald eagle                                                                 |
| (21) | restaurant                                                                 |
| (22) | bannister                                                                  |
| (23) | Crock Pot                                                                  |
| (24) | spider web                                                                 |
| (25) | mailbox                                                                    |
| (26) | turnstile                                                                  |
| (27) | toyshop                                                                    |
| (28) | scabbard                                                                   |
| (29) | lampshade                                                                  |
| (30) | tank                                                                       |
|      | Please respond with the number of the label that best describes the image. |
| Your | response must be a single number and nothing else.                         |

<sup>476</sup> Example high inference-time compute prompt with multiple-choice question for the giant panda <sup>477</sup> image. The model is instructed to make a classification using careful, step-by-step reasoning.

| High | Inference-Time Compute (With CoT)                                          |
|------|----------------------------------------------------------------------------|
| The  | image is described by one of the following labels:                         |
| (1)  | giant panda                                                                |
| (2)  | basenji                                                                    |
| (3)  | mantis                                                                     |
| (4)  | dome                                                                       |
| (5)  | organ                                                                      |
| (6)  | car wheel                                                                  |
| (7)  | carbonara                                                                  |
| (8)  | upright                                                                    |
| (9)  | buckle                                                                     |
| (10) | container ship                                                             |
| (11) | barbell                                                                    |
| (12) | thatch                                                                     |
| (13) | football helmet                                                            |
| (14) | snail                                                                      |
| (15) | cornet                                                                     |
| (16) | freight car                                                                |
| (17) | hog                                                                        |
| (18) | Dutch oven                                                                 |
| (19) | bubble                                                                     |
| (20) | bald eagle                                                                 |
| (21) | restaurant                                                                 |
| (22) | bannister                                                                  |
| (23) | Crock Pot                                                                  |
| (24) | spider web                                                                 |
| (25) | mailbox                                                                    |
| (26) | turnstile                                                                  |
| (27) | toyshop                                                                    |
| (28) | scabbard                                                                   |
| (29) | lampshade                                                                  |
| (30) | tank                                                                       |
|      | Please reflect on the image contents, then provide the number of the label |
| that | you think best describes the image.                                        |

## <sup>479</sup> C Additional Results

![](_page_16_Figure_1.jpeg)

Figure 7: PGD attack on color of the red soccer ball. Target: Gold.

![](_page_16_Figure_3.jpeg)

Figure 8: PGD attack on material of the soccer ball. Target: Wood.

![](_page_17_Figure_0.jpeg)

Figure 9: PGD attack on color of the yellow traffic light. Target: Green.

![](_page_17_Figure_2.jpeg)

Figure 10: PGD attack on color of the red traffic light. Target: Green.

## NeurIPS Paper Checklist

#### 1. Claims

 Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

 Justification: The claims made in the abstract and introduction are consistent with the contributions and scope of the paper. We clearly state our main contributions, and these are supported with experimental results.

Guidelines:

 • The answer NA means that the abstract and introduction do not include the claims made in the paper. • The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers. • The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings. • It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The limitations are discussed in Section [5](#page-8-2) Limitations.

Guidelines:

 • The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper. • The authors are encouraged to create a separate "Limitations" section in their paper. • The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be. • The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated. • The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon. • The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size. • If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness. • While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an impor- tant role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Answer: [NA]

Justification: The paper does not propose any theoretical results.

Guidelines:

 • The answer NA means that the paper does not include theoretical results. • All the theorems, formulas, and proofs in the paper should be numbered and cross- referenced. • All assumptions should be clearly stated or referenced in the statement of any theorems. • The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition. • Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.

#### 4. Experimental result reproducibility

 Question: Does the paper fully disclose all the information needed to reproduce the main ex- perimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

 Justification: We provide detailed descriptions of the methodology and experimental setup in Section [3](#page-4-1) and in Appendix [B.1.](#page-13-0) This will ensure our experiments are reproducible.

Guidelines:

 • The answer NA means that the paper does not include experiments. • If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not. • If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable. • Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed. • While NeurIPS does not require releasing code, the conference does require all submis- sions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

 Question: Does the paper provide open access to the data and code, with sufficient instruc- tions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: We will provide the code and data used to open source it.

Guidelines:

 • The answer NA means that paper does not include experiments requiring code. • Please see the NeurIPS code and data submission guidelines ([https://nips.cc/](https://nips.cc/public/guides/CodeSubmissionPolicy) [public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details. • While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark). • The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ([https:](https://nips.cc/public/guides/CodeSubmissionPolicy) [//nips.cc/public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details. • The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc. • The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why. • At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable). • Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

#### 6. Experimental setting/details

 Question: Does the paper specify all the training and test details (e.g., data splits, hyper- parameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

 Justification: We provide detailed descriptions of the approach, experimental setup and hyperparameters in Section [3](#page-4-1) and in Appendix [B.1.](#page-13-0)

Guidelines:

 • The answer NA means that the paper does not include experiments. • The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them. • The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

 Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

 Justification: We provide error bar for results in Figure [2.](#page-2-0) We report mean and standard error for results presented (Table [1\)](#page-7-1).

Guidelines:

 • The answer NA means that the paper does not include experiments. • The authors should answer "Yes" if the results are accompanied by error bars, confi- dence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper. • The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

 • The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors). • It should be clear whether the error bar is the standard deviation or the standard error of the mean. • It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified. • For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates). • If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

#### 8. Experiments compute resources

 Question: For each experiment, does the paper provide sufficient information on the com- puter resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: The paper clarifies the specific GPU type and quantity used in Section [3.](#page-4-1)

Guidelines:

 • The answer NA means that the paper does not include experiments. • The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage. • The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute. • The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

### 9. Code of ethics

 Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?

Answer: [Yes]

 Justification: The research conducted in this paper conforms with the NeurIPS Code of Ethics.

Guidelines:

 • The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. • If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics. • The authors should make sure to preserve anonymity (e.g., if there is a special consid-eration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

 Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: We discuss the broader impact of the work in Section [5.](#page-8-2)

Guidelines:

 • The answer NA means that there is no societal impact of the work performed. • If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact.

 • Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations. • The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster. • The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology. • If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

#### 11. Safeguards

 Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [Yes]

 Justification: The experiments in the paper use open-source models and data. We do not release any trained models or new datasets.

Guidelines:

 • The answer NA means that the paper poses no such risks. • Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters. • Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images. • We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

#### 12. Licenses for existing assets

 Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The models used are properly credited in Section [3.](#page-4-1)

Guidelines:

 • The answer NA means that the paper does not use existing assets. • The authors should cite the original paper that produced the code package or dataset. • The authors should state which version of the asset is used and, if possible, include a URL. • The name of the license (e.g., CC-BY 4.0) should be included for each asset. • For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

 • If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, <paperswithcode.com/datasets> has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset. • For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided. • If this information is not available online, the authors are encouraged to reach out to the asset's creators.

#### 13. New assets

 Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

 Justification: Our paper uses publicly available datasets and models. It does not currently introduce new code, data, or models.

Guidelines:

 • The answer NA means that the paper does not release new assets. • Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc. • The paper should discuss whether and how consent was obtained from people whose asset is used. • At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

#### 14. Crowdsourcing and research with human subjects

 Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: The paper does not involve crowd-sourcing experiments.

Guidelines:

 • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contribu- tion of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

#### 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

 Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: The paper does not involve a study with human subjects.

Guidelines:

 • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

 • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

#### 16. Declaration of LLM usage

 Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [Yes]

 Justification: Our paper presents findings in robustness of vision language models. The models used are described in Section [3.](#page-4-1)

Guidelines:

 • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (<https://neurips.cc/Conferences/2025/LLM>) for what should or should not be described.