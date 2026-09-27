# A Is For Absorption: Studying Feature Splitting And Absorption In Sparse Autoencoders

David Chanin*,1,2 James Wilken-Smith*,1 Tomáš Dulka*,1 Hardik Bhatnagar***,1,3**
Satvik Golechha4 Joseph Bloom1,5 1LASR Labs 2University College London 3 Tübingen AI Center, University of Tübingen 4MATS 5Decode Research

## Abstract

Sparse Autoencoders (SAEs) aim to decompose the activation space of large language models (LLMs) into human-interpretable latent directions or features. As we increase the number of features in the SAE, hierarchical features tend to split into finer features ("math" may split into "algebra", "geometry", etc.), a phenomenon referred to as feature splitting. However, we show that sparse decomposition and splitting of hierarchical features is not robust. Specifically, we show that seemingly monosemantic features fail to fire where they should, and instead get "absorbed" into their children features. We coin this phenomenon feature absorption, and show that it is caused by optimizing for sparsity in SAEs whenever the underlying features form a hierarchy. We introduce a metric to detect absorption in SAEs, and validate our findings empirically on hundreds of LLM SAEs. Our investigation suggests that varying SAE sizes or sparsity is insufficient to solve this issue. We discuss the implications of feature absorption in SAEs and some potential approaches to solve the fundamental theoretical issues before SAEs can be used for interpreting LLMs robustly and at scale.

## 1 Introduction

Large Language Models (LLMs) have achieved remarkable performance across a wide range of tasks, yet our understanding of their internal mechanisms lags behind their capabilities. This gap between performance and interpretability raises concerns about the "black box" nature of these models [30]. The field of mechanistic interpretability aims to address this issue by reverse-engineering the internal algorithms of neural networks and performing causal analysis on them [24]. Recent work theorizes that models represent concepts as linear directions in a high-dimensional space, known as the Linear Representation Hypothesis (LRH) [26, 8]. The model is able to represent far more concepts than it has neurons in its hidden space by allowing these concept directions to overlap slighly, known as superposition [8]. Superposition makes it challenging to directly interpret neurons in an LLM, and requires different techniques to extract interpretable feature directions. As long as the active features in a given LLM activation are sparse and the underlying features follow the LRH, Sparse Autoencoders (SAEs) should be able to recover the true LLM features despite supersition using sparse dictionary learning [27]. Indeed, SAEs have shown potential in decomposing the dense, polysemantic activations of LLMs into more "interpretable" latent features [5, 2]. However, we show that even if all underlying features are linear and sparsely activating, an SAE will still fail to recover the true underlying features if the features form a hierarchy. Instead, an SAE will

| Ideal interpretable solution                          | Uninterpretable absorption solution   |                      |             |                       |                 |
|-------------------------------------------------------|---------------------------------------|----------------------|-------------|-----------------------|-----------------|
| SAE Encoder                                           | SAE Decoder                           | SAE Encoder          | SAE Decoder |                       |                 |
| Latent 1                                              | "starts with S"                       | "starts with S"      |             |                       |                 |
| Latent 2                                              | "short"                               | "short"              | Latent 1    | ¬"short" ∧ "starts S" | "starts with S" |
| Latent 2                                              | "short"                               | "short" + "starts S" |             |                       |                 |
| Absorption only requires firing 1 latent to represent |                                       |                      |             |                       |                 |
| The ideal interpretable solution requires             |                                       |                      |             |                       |                 |

The ideal interpretable solution requires firing 2 latents to represent "short".

Absorption only requires firing 1 latent to represent
"short", and is unfortunately what the SAE learns.

Figure 1: In feature absorption, seemingly monosemantic latents fail to fire in cases where they apparently should. Here, we see an SAE can represent the word "short" and the concept "starts with S" more sparsely by absorbing the "starts with S" direction into the "short" latent, and then not firing the "starts with S" latent on the word "short", despite "short" starting with "S". Logical notation is used to describe the SAE encoder to emphasize its role as a classifier. learn gerrymandered latents that fail to fire on seemingly arbitrary cases where the latent should fire according to the mainline interpretation of the latent. We refer to this failure as feature absorption. Feature absorption is demonstrated in Figure 1, where the feature "short" always fires alongside a feature representing "starts with S". Instead of learning an interpretable latent representing "starts with S", the SAE can increase sparsity by instead disabling the "starts with S" latent when "short" is active while still getting perfect reconstruction. We present the following contributions: (1) we identify a problematic variant of feature-splitting we call "feature absorption", where an SAE latent appears to track a human-interpretable concept, but fails to activate on seemingly arbitrary tokens. Instead, more specific latents activate and contribute a component of feature direction, "absorbing" the feature. (2) We demonstrate that feature absorption is caused by hierarchical features. (3) We develop a metric to detect feature absorption in LLM SAEs. And (4) we validate that feature absorption occurs in every LLM SAE we tested, including hundreds of open-source SAEs. Feature absorption poses an obstacle to the practical application of SAEs since it suggests SAE latents may be inherently unreliable classifiers. This is particularly important for applications where we need confidence that latents are fully tracking behaviors, such as bias or deceptive behavior. Furthermore, techniques which seek to describe circuits in terms of a sparse combination of latents will also be more difficult in the presence of feature absorption [21].

An online explorer for our results can be found at https://feature-absorption.streamlit.

app. Code is available at https://github.com/lasr-spelling/sae-spelling.

## 2 Background

Hierarchical features. We say features f1 and f2 form a hierarchy with f1 as the parent and f2 as the child if f2 =⇒ f1, meaning every time f2 fires f1 must also fire.

Linear probing. A linear probe is a simple linear classifier trained on the hidden activations of a neural network, typically using logistic regression (LR) [1]. K-sparse probing. A k-sparse probe [12] is a linear probe trained on a sparse subset of k neurons or SAE latents. Training a k-sparse probe first requires selecting the k best neurons or SAE latents that in-aggregate act as a good classifier, and then training a standard linear probe on just those k neurons or latents. Gurnee et al. [12] proposed several methods of estimating the best k neurons or latents to pick, one of which involves first training a LR probe with a L1 loss term, and selecting the k largest elements by probe weight. When we refer to k-sparse probing in this work, we use this method of selecting k latents.

Sparse autoencoders. An SAE consists of an encoder, Wenc, a decoder, Wdec, and corresponding biases benc and bdec. The SAE has a nonlinearity, σ, typically a ReLU (or variant such as JumpReLU
[29]). Given input activation, a, the SAE computes a hidden representation, f, and reconstruction, aˆ:

$$f=\sigma(W_{e n c}a+b_{e n c})$$
$f+b_{deg}$
f =σ(Wenca + benc) (1)
aˆ =Wdecf + bdec (2)
SAEs attempt to reconstruct input activations by projecting into an overcomplete basis using a sparsity-inducing loss term (typically L1 loss), or a certain number of non-zero latents (L0) on the hidden activations. SAE feature ablation. In an ablation study we examine the downstream causal effect of an SAE
latent by computing how patching its activation to 0 changes a downstream metric (e.g. logit difference). A negative ablation effect means intervening on the SAE latent would lower the metric. We follow the work of Marks et al. [21] and provide the algorithm in Appendix A.4.

## 3 Toy Models Of Feature Absorption

Absorption is caused by the SAE sparsity penalty in the presence of hierarchical features. When two features form a hierarchy, for instance "starts with S" and "short", the SAE can merge the "starts with S" feature direction into a latent tracking "short" and then not fire the main "starts with S" latent. This means firing one latent instead of two, increasing sparsity while retaining perfect reconstruction. We demonstrate that hierarchical feature co-occurrence causes absorption in a simple toy setting. Our initial setup consists of 4 true features, each randomly initialized into orthogonal directions with a 50 dimensional representation vector and unit norm. Each feature fires with magnitude 1.0. Feature f0 fires with probability 0.25, and features f1, f2, and f3 fire with probability 0.05. Each SAE training input is created by sampling from these true features and summing the directions of each firing feature. We train a SAE with 4 latents to match the 4 true features using SAELens [15]. The SAE uses L1 loss with L1 coefficient 3e-5, and learning rate 3e-4. We train on 100M activations. Independently firing features When the true features fire independently, we find that the SAE is able to perfectly recover these features as shown in Figure 2a. The SAE learns one latent per true feature. The decoder representations perfectly match the true feature representations, and the encoder learns to perfectly segment out each feature from the other features.

Cos sim with true features (Independent features)
SAE encoder SAE decoder
−1.0 −0.5 0.0 0.5 1.0 0 0 3 S
AE L
atent 3 S
AE L
atent co s sim 1 1 2 2 0 1 2 3 True feature 0 1 2 3 True feature Cos sim with true features (feat 1 co-occurs w/feat 0)
SAE encoder SAE decoder
−1.0 −0.5 0.0 0.5 1.0 0 0 3 S
AE L
atent 3 S
AE L
atent co s sim 1 1 2 2 0 1 2 3 True feature 0 1 2 3 True feature
Hierarchical features cause absorption Next, we modify the firing pattern of feature 1 so it fires only if feature 0 also fires. We keep the overall firing rate of feature 1 the same as before, firing in 5% of activations. Features 2 and 3 remain independent. Figure 2b shows the encoder and decoder cosine similarities with the true features in the hierarchical co-occurrence setup. Here, we see a clear example of feature absorption. Latent 0 Figure 3: Interpretation of learned SAE latents with co-occurrence between feature 0 and feature 1 (feature 1 only fires if feature 0 fires).

| SAE ENCODER   | SAE DECODER       |                 |
|---------------|-------------------|-----------------|
| LATENT 0      | ¬ feat 1 ∧ feat 0 | feat 0          |
| LATENT 1      | feat 1            | feat 0 + feat 1 |
| LATENT 2      | feat 3            | feat 3          |
| LATENT 3      | feat 2            | feat 2          |

3 has learned a perfect representation of feature 0, but the encoder has a hole in its recall. Latent 0 fires if feature 0 is active but not feature 1. This is exactly the sort of gerrymandered feature firing pattern we will see later in real SAEs in Section 5.2 - the encoder has learned to stop the latent firing on specific cases where it looks like it should be firing. In addition, we see that latent 1, which tracks feature 1, has absorbed the feature 0 direction. This results in latent 1 representing a combination of feature 0 and feature 1. We see that the independently firing features 2 and 3 are untouched - the SAE still learns perfect representations of these features. These results are summarized in Table 3. We explore absorption in more toy settings in Appendix A.3. Proof: hierarchical features cause absorption We further provide an analytical proof that in the hierarchical setup described above, feature absorption decreases SAE loss in Appendix A.2.

## 4 Experimental Setup

Our experiments on LLM SAEs focus on predicting the first-letter of a single token containing characters from the English alphabet (a-z, A-Z) and an optional leading space. We use in-context learning (ICL) prompts to elicit knowledge from the model, using templates of the form:
{token} has the first letter: {capitalized_first_letter}
An example of an ICL prompt consisting of 2 in-context examples is shown below. The model should output the _D token:
tartan has the first letter: T mirth has the first letter: M dog has the first letter:
In the above prompt, we extract residual stream activations at the _dog token index. These activations are used both for LR probe training and for applying SAEs. We use a train/test split of 80% / 20%, and evaluate only on the test set of the probes, including when running experiments on SAEs. When applying SAEs, we include the SAE error term [21] to avoid changing model output. To determine the causal effect of SAE latents on the first-letter identification task we conduct ablation studies. We use a metric consisting of the logit of the correct letter minus the mean logit of all incorrect letters. This measures the propensity of the model to choose the correct starting letter as opposed to other letters. Formally, our metric m is defined below, where g refers to the final token logits, L is the set of uppercase letters, and y is the uppercase letter that is the correct starting letter:

$$m=g[y]-{\frac{1}{|L|-1}}\sum_{l\in\{L\setminus y\}}g[l]$$

We discuss this metric and alternative formulations further in Appendix A.10. To determine how well multiple latents perform as a classifier when used together, we use k-sparse probing, increasing the value of k from 1 to 15. We train a LR probe using a L1 loss term with coefficient 0.01, and select the top k latents by magnitude.

We use the base Gemma-2-2B model for most of our studies, along with the full set of Gemma Scope residual stream SAEs of width 16k and 65k released by Deepmind [19]. We also evaluate absorption on our own SAEs trained on Qwen2 0.5B [32] and Llama 3.2 1B [6].

## 5 Results

Our results are divided into four sections. First, we compare the performance of linear probes with SAE latents on recovering first-character information from model activations, showing that despite appearing to track first letter features, a wide variety of precision / recall is achieved. Second, we motivate our definition of feature absorption with a case-study, emphasizing how an absorbing latent can unexpectedly causally mediate first letter information whilst the first-letter latent (unexpectedly) fails to fire. Next, we attempt to quantify feature splitting and feature absorption, showing that tuning of hyper-parameters may partially assist but not fully alleviate feature absorption.

## 5.1 Do Saes Learn Latents That Track First Letter Information?

We compare the performance of LR probes with the performance of the SAE latent whose encoder direction has highest cosine similarity with the probe, resulting in 26 "first-letter" latents. We observed that for each probe, there was clearly one or at most a couple of outlier SAE latents with high probe cosine similarity. Full plots of cosine similarity vs letter are shown in Appendix A.8. We also tried using k=1 sparse probing [12] to identify SAE latents, and found this gives similar results. Further comparison of using k=1 sparse probing vs encoder cosine similarity to identify latents is explored in Appendix A.7.

First-letter SAE f1 vs Layer SAE width 16k 65k Probe 0 2 4 6 8 10 12 14 16 18 20 22 24 Layer 0.2 0.4 0.6 0.8 Mean f 1 First-letter SAE f1 vs L0 (layers 0-12)
First-letter SAE f1 vs L0 (layers 13-25)
SAE width 16k 65k SAE width 16k 65k 0 100 200 300 400 L0 0.2 0.3 0.4 0.5 0.6 0.7 0 100 200 300 400 L0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 Mean f 1 Mean f 1 First-letter SAE precision vs L0 SAE width 16k 65k First-letter SAE recall vs L0 0 100 200 300 400 L0 0.2 0.4 0.6 0.8 1.0 0 100 200 300 400 L0 0.2 0.4 0.6 0.8 1.0 Mean precis ion Mean reca ll SAE width 16k 65k First-letter SAE precision vs recall 0.2 0.4 0.6 0.8 1.0 Mean precision 0.2 0.4 0.6 0.8 1.0 0 100 200 300 400 Mean recall L0
We observe wide variance in the performance of Gemma Scope SAEs at the first-letter identification task, but no SAE matches LR probe performance. We show the mean F1 score by layer as well as the F1 score of the LR probe in Figure 4a. We further investigate the F1 score of these SAE encoder latents as a function of L0 and SAE width in Figures 4b and 4c. Whether or not an SAE learns a clear "first-letter" latent for each letter is highly dependent on L0, with low L0 SAEs tending to learn high-precision low-recall latents, and high L0 SAEs learning low-precision high-recall latents (Figure 5). We caution drawing conclusions about an "optimal" L0 from these plots, as we find further variance when broken-down by letter, shown in Appendix A.8.

## 5.2 Why Do Sae Latents Underperform?

The Gemma Scope layer 3, 16k width, 59 L0 SAE has a latent, 6510, which appears to act as a classifier for "starts with S", achieving an F1 of 0.81. However, this latent fails to activate on some tokens the probe can classify, and which the model can spell, such as the token _short.

Figure 6a shows a sample prompt containing a series of tokens that start with "S", and the activations of top SAE latents by ablation score for these tokens. The main "starts with S" latent, 6510, activates on all these tokens except _short. This SAE also has a token-aligned latent, 1085, which activates on variants of the word "short" (" short", "SHORT", etc...). The Neuronpedia dashboard [20] for latent 1085 is shown in Appendix A.15. For the token _short, the main "starts with S" latent does not activate but the "short" latent activates instead.

'S' activations by token, layer 3, 16k width, 59 L0 snakesoggysteamshortsoccersax Token 0 10 20 30 40 50 Latent ID
1085 6510 A
ctivat io n Cos in e simila rity Main latent Absorbing latent id 6510, cos 0.52 id 1085, cos 0.12 Cosine similary between SAE decoder and 'S' probe 0.0 0.2 0.4 0 2500 5000 7500 10000 12500 15000 SAE latent ID
Ab lati on e ffe ct First-letter ablation effects for " short" token, layer 3, L0=59 10852854557141261496426732791913437755489 SAE latent ID
−6 −4 −2 0 Ab lati on e ffe ct Ablation effects for " short" token projecting out probe 285455714126267327919134377554891076512163 SAE latent ID
−6 −4 −2 0
Latent 1085 has a cosine similarity with the "starts with S" probe of 0.12, indicating it contains a component of the "starts with S" direction, although much smaller than the main "starts with S" latent. Cosine similarity of the SAE decoder with the "starts with S" LR probe is shown in Figure 6b. Interestingly, despite latent 1085 having only about 1/5 the cosine similarity with the probe as the main latent 6510, we see it activates with about 5 times the magnitude of latent 6510 on the _short token, thus contributing a similar amount of the "starts with S" probe direction to the residual stream.

We study the ablation effect of each SAE latent on the _short token, shown in Figure 7a, and see that latent 1085 has a dramatically larger ablation effect compared with all other SAE latents. This suggests latent 1085 is causally responsible for the model knowing that _short starts with S. Is it possible that the probe projection is not the causally important component of latent 1085? We conduct another ablation effect experiment, except now we remove the probe direction from latent 1085 via projection before ablation. The results of this experiment are shown in Figure 7b. After removing the probe component from latent 1085, it no longer has a significant ablation effect. Thus we know the probe projection of latent 1085 is responsible for model behavior. These experiments show the "starts with S" feature has been "absorbed" by the token-aligned latent 1085, likely along with other semantic concepts related to the word "short". After observing that the main "starts with S" latent 6510 activates on most tokens that begin with "S", it may be tempting to conclude this latent tracks the interpretable feature of beginning with the letter "S". However, this latent quietly fails to activate on the _short token, leading us to a false sense of understanding. Here we clearly see feature absorption. The seemingly interpretable SAE latent 6510 fails to activate on arbitrary positive examples, and instead the feature is "absorbed" into more specific latents. Feature absorption is likely a logical consequence of SAE sparsity loss. If a dense and sparse feature co-occur, absorbing the dense feature into a latent tracking the sparse feature will increase sparsity. Table 1: Sample max activating examples for latents 7112 and 7657 for Gemma Scope 16k, layer 0, 105 L0 from Neuronpedia. The token where the SAE latent activates is highlighted in yellow. Latent 7112 appears to be a lowercase "L" starting-letter latent, and latent 7657 appears to be a corresponding uppercase "L" latent.

| LATENT 7112                | LATENT 7657                    |
|----------------------------|--------------------------------|
| žda se naplacuje naknada ´ | LC, an aluminum boat           |
| . E. Søli, 20              | as LIFT and LF-Net. Once       |
| a></code></li></ul         | latter's sister Louise, who in |

## 5.3 Measuring Feature Splitting And Feature Absorption

Feature splitting A key phenomenon identified from previous studies of SAEs is feature-splitting [2], where a feature represented in a single latent in a smaller SAE can split into two or more latents in a larger SAE. During our experiments, we found strong evidence of feature-splitting in the Gemma Scope SAEs. For instance, in the layer 0, 16k width, 105 L0 SAE, we find two encoder latents (id:7112 and id:7657 1) which align with the "L" starting letter probe. Inspecting max activating examples, we see latent 7112 activates on tokens starting with lowercase "l", while 7657 activates on tokens starting with uppercase "L". Some activating examples for these latents are shown in Table 1. Feature splitting like this is not necessarily problematic for interpretability efforts since the split features are still easily identifiable, and depending on the context it may be more useful to have either a single "starts with L" latent or a pair of "starts with uppercase / lowercase L" latents. We measure feature splitting using k-sparse probing [12] on SAE activations. If increasing the k-sparse probe from k to k + 1 causes a significant increase in probe F1 score, then the additional SAE latent provides a meaningful signal, and the combination of these k + 1 latents is likely a feature split. In the example of the uppercase "L" and lowercase "l" split, a k-sparse probe with k = 2 trained on both these latents should predict "starts with letter L" much better than either latent on its own. Figure 8a shows F1 vs K for letters "L" and "N". The "L" k-sparse probe shows a significant jump in F1 score moving from k=1 to k=2 corresponding to feature splitting, while the F1 score for the "N" k-sparse probe is relatively constant.

Starts with "N"
K-sparse probing results Starts with "L"
1 2 3 4 5 K
0.0 0.2 0.4 0.6 0.8 F1 1 2 3 4 5 K
Mean feature splits per first-letter vs L0 SAE width 16k 65k 0 50 100 150 200 250 300 L0 0.0 0.5 1.0 1.5 2.0 2.5 Num feature s pli ts
We detect feature splitting by measuring whether increasing k by one causes a jump in F1 score by more than a threshold, τ . We use τ = 0.03 as a reasonable choice after inspecting situations like in Figure 8a, where feature splitting corresponds to an F1 score jump between 0.05 - 0.1. Figure 8b shows feature splitting vs L0 for all 16k and 65k width Gemma Scope SAEs.

1https://www.neuronpedia.org/list/cm0h1n2mt00019jdk274owq9e The single latent or a set of traditional feature split latents that seem to act as a classifier for a human-interpretable feature like "starts with S" fail to fire in a seemingly arbitrary number of cases. What fires instead are often approximately token-aligned latents with small but positive alignment with the LR probe. We say these latents are absorbing the feature.

Mean absorption rate for first-letter task vs Layer Mean absorption rate for first-letter task vs L0 Mean absorption rate for first-letter task vs layer 0 1 2 3 4 5 6 7 8 Layer 0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 Layer 0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 SAE width 16k 65k 0 100 200 300 400 L0 0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 SAE width 16k 65k Model / SAE

Llama 3.2 1B / L1 Llama 3.2 1B / TopK

Qwen 2 0.5B / L1 Mean abs orpti on rate Mean abs orpti on rate Mean ab so rption rate
We quantify the extent to which feature absorption occurs with the metric **feature absorption**
rate. We first find k feature splits for a first-letter feature using a k-sparse probe. We then find false-negative tokens that all k feature-split SAE latents fail to activate on, but which the LR probe correctly classifies, and run an integrated-gradients ablation experiment on those tokens. The ablation effect finds the most causally important SAE latents for the spelling of that token. If the SAE latent receiving the largest negative magnitude ablation effect has a cosine similarity with the LR probe above 0.025, and is at least 1.0 larger than the latent with the second highest ablation effect, we say that feature absorption has occurred. These thresholds were chosen from manual inspection of the data to best distinguish the absorption phenomenon. We then calculate feature absorption rate as:

$${\mathsf{a b s o r p t i o n_{\mathsf{T a t e}}}}={\frac{\mathsf{n u m_{\mathsf{a b s o r p t i o n s}}}}{1\mathsf{r_{\mathsf{p b o e_{\mathsf{t r u e_{\mathsf{p o s i v e}}}}}}}}}$$

If there are more than 200 false negatives per letter, we randomly pick 200 samples to estimate the number of absorptions. We see absorption rate increases with higher sparsity and higher SAE width. Lower L0 likely pushes the SAE to absorb dense features like spelling information, increasing feature sparsity. Feature absorption rate vs L0 for Gemma Scope SAEs layers 0-17 is shown in Figure 9b. Absorption rate by letter is shown in Appendix A.14. We also train our own set of standard L1 loss SAEs on the first 8 layers of Qwen2 0.5B [32] and Llama 3.2 1B [6], and TopK SAEs [10] on Llama 3.2 1B. In Figure 9c we show that absorption occurs in these SAEs as well. Our metric cannot capture absorption past layer 17 in Gemma 2 2B since we rely on ablation experiments to be certain the absorbed feature causally mediates model behavior. Past layer 17, attention has already moved the starting letter information from the source token into the final token position, so any ablations on the source token past layer 17 have little effect. This is a limitation of our absorption metric - we rely on ablation to be certain of the causal impact of absorbed features on model behavior, but this limits the layer depth our metric can be applied. We discuss this further in Appendix A.12 and discuss alternative formulations of the metric in Appendix A.13. Our absorption metric is not perfect, and is likely an under-estimate of the true level of feature absorption. We only consider absorption to have occurred if a single SAE latent has a much larger ablation effect than all other latents, and if the main SAE latents for a feature do not activate at all. Our metric will not capture multiple absorbing latents activating together, or the main latents activating weakly. Regardless, we feel our metric is a reasonable conservative baseline.

## 6 Related Work

Applications of Probes and SAEs for Model Interpretability Probing methods can extract interpretable information from language models, though this does not guarantee the model uses these representations in its computation, and requires labeled data [7]. Prior work has shown that many human-interpretable concepts in LLM activations are represented as linear directions in activation space, known as the linear representation hypothesis [8, 28]. Li et al. [18] used non-linear probes to recover board representations from a transformer trained on Othello scripts ("OthelloGPT"). Nanda [23] later showed that linear representations were not only recoverable but also editable. Karvonen et al. [16] developed objective metrics for SAE evaluation using Chess and Othello board states, but does not apply these to SAEs trained on LLM activations. Work by Olah et al. [26], Kissane et al. [17], Templeton et al. [31] noted poor precision/recall of SAE latents compared to known proxies. We extend this by showing how sparsity mediates precision/recall across many Gemma Scope SAEs and offer a possible explanation of low recall due to feature absorption. Engels et al. [9] investigated SAE errors, finding that not all SAE error is linearly decomposable. Studying precision and recall of SAE Latents Most existing work on SAE interpretability mainly studies max activating examples [5], which may be misleading. There are more rigorous works which only measure precision [2, 31, 17]. Recent work has briefly explored recall and found it to be worse than expected naively, but this remains poorly understood [26]. We build on this work by evaluating precision / recall on a large number of SAEs, and offer a partial explanation for lower-than-expected recall of SAE latents in the form of feature absorption. Decomposing SAE Latents Feature splitting was first described in Bricken et al. [2], which noted that different SAE widths and sparsities induce latents of different granularity, with wider SAEs often learning more specific variants of features. Bussmann et al. [4] find that by training an SAE on the decoder of another SAE, a technique called Meta-SAEs, it is possible to break down a single SAE latent like "Einstein" into subcomponents like "German" and "Physicist" and "starts with E".

## 7 Discussion

Limitations Our Absoption metric uses ablation effect to ensure that the absorbed features causally mediate model behavior, and thus might not be easily transferable to the final model layers. Alternate metric formulations mitigating this are discussed in Appendix A.13. Due to compute constraints, we only train and evaluate a small number of non-JumpReLU SAEs in Figure 9. As our goal was only to show absorption occurs in all SAE architectures, we did not feel this is a significant drawback. Future Work The primary goal of future work is to find solutions to feature absorption. We are particularly hopeful that work extending Meta-SAEs [4] may solve or mitigate feature absorption. Another possible solution may be attribution dictionary learning [25]. Finally, structured sparsity techniques such as group lasso [13] or hierarchical sparse coding [14] may also be a promising direction of future work. Other possible directions include allowing absorption to occur and using it as a way to recover hierarchies between features in a LLM. Our toy model results suggest that absorption leads to an asymmetric pattern in the encoder and decoder of the SAE, so it may be possible to use this insight to detect absorption (although there may be other reasons for an asymmetry in the SAE encoder and decoder beyond absorption). Conclusion We identify a form of feature splitting we call "feature absorption", where more specific latents "steal credit" from more general ones. Absorption creates an interpretability illusion, where a seemingly interpretable latent has arbitrary false negatives in its mainline interpretation. Lower recall poses problems for using SAEs for high-stakes classification or finding sparse circuits [21], as the number of latents needed to characterize model behavior may be much larger than expected. We show that absorption is a consequence of hierarchical co-occurrence between sparse and dense features. If a dense feature like "starts with letter D" always co-occurs with a more sparse feature like
"dogs", the SAE can increase sparsity by absorbing the "starts with D" feature into a "dogs" latent.

We hope that our work highlights the fundamental limitations of sparse feature extraction and prompts future research on SAEs such as identifying cases where a feature "should have activated" but does not due to absorption, and exploring theoretical solutions to absorption. The ease of demonstrating absorption in toy models makes it easier to validate potential solutions.

## Acknowledgments And Disclosure Of Funding

This project was produced as part of the LASR Labs research program. DC was supported thanks to EPSRC EP/S021566/1.

## References

[1] Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes, 2017. URL https://openreview.net/forum?id=ryF7rTqgl.

[2] Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, et al. Towards monosemanticity: Decomposing language models with dictionary learning. *Transformer Circuits Thread*, 2, 2023.

[3] Bart Bussmann, Patrick Leask, and Neel Nanda. Batchtopk sparse autoencoders, 2024. URL
https://arxiv.org/abs/2412.06410.

[4] Bart Bussmann, Michael Pearce, Patrick Leask, Joseph Bloom, Lee Sharkey, and Neel Nanda.

Showing sae latents are not atomic using meta-saes, 2024. URL https://www.lesswrong.

com/posts/TMAmHh4DdMr4nCSr5.

[5] Hoagy Cunningham, Logan Riggs Smith, Aidan Ewart, Robert Huben, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?

id=F76bwRSLeK.

[6] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaoqing Ellen Tan, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aaron Grattafiori, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alex Vaughan, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Franco, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, Danny Wyatt, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Firat Ozgenel, Francesco Caggioni, Francisco Guzmán, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Govind Thattai, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Karthik Prasad, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kun Huang, Kunal Chawla, Kushal Lakhotia, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Maria Tsimpoukelli, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikolay Pavlovich Laptev, Ning Dong, Ning Zhang, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Rohan Maheswari, Russ Howes, Ruty Rinott, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Kohler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vítor Albiero, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaofang Wang, Xiaojian Wu, Xiaolan Wang, Xide Xia, Xilun Wu, Xinbo Gao, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yuchen Hao, Yundi Qian, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, and Zhiwei Zhao. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.

[7] Yanai Elazar, Shauli Ravfogel, Alon Jacovi, and Yoav Goldberg. Amnesic probing: Behavioral explanation with amnesic counterfactuals. Transactions of the Association for Computational Linguistics, 9:160–175, 2021.

[8] Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, et al. Toy models of superposition. *arXiv preprint arXiv:2209.10652*, 2022.

[9] Joshua Engels, Eric J Michaud, Isaac Liao, Wes Gurnee, and Max Tegmark. Not all language model features are one-dimensionally linear. In *The Thirteenth International Conference on* Learning Representations, 2025. URL https://openreview.net/forum?id=d63a4AM4hb.

[10] Leo Gao, Tom Dupré la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, and Jeffrey Wu. Scaling and evaluating sparse autoencoders. arXiv preprint arXiv:2406.04093, 2024.

[11] Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. Dissecting recall of factual associations in auto-regressive language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12216–12235, 2023.

[12] Wes Gurnee, Neel Nanda, Matthew Pauly, Katherine Harvey, Dmitrii Troitskii, and Dimitris Bertsimas. Finding neurons in a haystack: Case studies with sparse probing. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/ forum?id=JYs1R9IMJr.

[13] Laurent Jacob, Guillaume Obozinski, and Jean-Philippe Vert. Group lasso with overlap and graph lasso. In *Proceedings of the 26th annual international conference on machine learning*, pages 433–440, 2009.

[14] Rodolphe Jenatton, Julien Mairal, Guillaume Obozinski, and Francis Bach. Proximal methods for hierarchical sparse coding. *The Journal of Machine Learning Research*, 12:2297–2334, 2011.

[15] Curt Tigges Joseph Bloom and David Chanin. Saelens. https://github.com/jbloomAus/
SAELens, 2024.

[16] Adam Karvonen, Benjamin Wright, Can Rager, Rico Angell, Jannik Brinkmann, Logan Riggs Smith, Claudio Mayrink Verdun, David Bau, and Samuel Marks. Measuring progress in dictionary learning for language model interpretability with board game models. In ICML 2024 Workshop on Mechanistic Interpretability, 2024.

[17] Connor Kissane, Robert Krzyzanowski, Joseph Isaac Bloom, Arthur Conmy, and Neel Nanda.

Interpreting attention layer outputs with sparse autoencoders, 2024. URL https://arxiv. org/abs/2406.17759.

[18] Kenneth Li, Aspen K Hopkins, David Bau, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. Emergent world representations: Exploring a sequence model trained on a synthetic task. *ICLR*, 2023.

[19] Tom Lieberum, Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Nicolas Sonnerat, Vikrant Varma, János Kramár, Anca Dragan, Rohin Shah, and Neel Nanda. Gemma Scope:
Open Sparse Autoencoders Everywhere All At Once on Gemma 2, August 2024.

[20] Johnny Lin and Joseph Bloom. Analyzing neural networks with dictionary learning, 2023. URL
https://www.neuronpedia.org. Software available from neuronpedia.org.

[21] Samuel Marks, Can Rager, Eric J. Michaud, Yonatan Belinkov, David Bau, and Aaron Mueller.

Sparse feature circuits: Discovering and editing interpretable causal graphs in language models.

Computing Research Repository, arXiv:2403.19647, 2024. URL https://arxiv.org/abs/
2403.19647.

[22] Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. *Advances in Neural Information Processing Systems*, 36, 2022. arXiv:2202.05262.

[23] Neel Nanda. Actually, othello-gpt has a linear emergent world model, mar 2023. URL<
https://neelnanda.io/mechanistic-interpretability/othello, 2023.

[24] Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, and Shan Carter.

Zoom in: An introduction to circuits. *Distill*, 5(3):e00024–001, 2020.

[25] Chris Olah, Adly Templeton, Trenton Bricken, and Adam Jermyn. April update. https:
//transformer-circuits.pub/2024/april-update/index.html, 2024. URL https: //transformer-circuits.pub/2024/april-update/index.html.

[26] Chris Olah, Nicholas Turner, Adam Jermyn, and Joshua Batson. July update. https://
transformer-circuits.pub/2024/july-update/index.html, 2024. URL https:// transformer-circuits.pub/2024/july-update/index.html.

[27] Bruno A Olshausen and David J Field. Sparse coding with an overcomplete basis set: A strategy employed by v1? *Vision research*, 37(23):3311–3325, 1997.

[28] Kiho Park, Yo Joong Choe, and Victor Veitch. The linear representation hypothesis and the geometry of large language models. In *Forty-first International Conference on Machine* Learning, 2024. URL https://openreview.net/forum?id=UGpGkLzwpP.

[29] Senthooran Rajamanoharan, Tom Lieberum, Nicolas Sonnerat, Arthur Conmy, Vikrant Varma, János Kramár, and Neel Nanda. Jumping ahead: Improving reconstruction fidelity with jumprelu sparse autoencoders. *arXiv preprint arXiv:2407.14435*, 2024.

[30] Cynthia Rudin. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, pages 206–215, 2019.

[31] Adly Templeton, Tom Conerly, Jonathan Marcus, Jack Lindsey, Trenton Bricken, Brian Chen, Adam Pearce, Craig Citro, Emmanuel Ameisen, Andy Jones, Hoagy Cunningham, Nicholas L Turner, Callum McDougall, Monte MacDiarmid, Alex Tamkin, Esin Durmus, Tristan Hume, Francesco Mosconi, C. Daniel Freeman, Theodore R. Sumers, Edward Rees, Joshua Batson, Adam Jermyn, Shan Carter, Chris Olah, and Tom Henighan. Scaling monosemanticity: Extracting interpretable features from claude 3 sonnet. https://transformer-circuits.pub/ 2024/scaling-monosemanticity/, May 2024. Accessed on May 21, 2024.

[32] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. Qwen2 technical report. *arXiv preprint arXiv:2407.10671*, 2024.

## A Appendix A.1 Glossary Of Terms

Sparse Autoencoders (SAEs): Neural networks trained to reconstruct their input while enforcing sparsity in their hidden layer. In the context of this paper, SAEs are used to decompose the dense activations of language models into more interpretable features. SAE error term: When inserting a SAE into the computation path of the model, errors in SAE reconstruction will propagate to later parts of the model and can change the model output. We refer to the error as the SAE error term, and corresponds to the difference between the SAE output and the original SAE input activation. Marks et al. [21] introduced the idea of adding this error term back to the SAE output to ensure that the SAE does not change model output. Latent: We refer to neurons in the hidden layer of a SAE as latents to avoid overloading the term "feature". This is in contrast to earlier work which used the term "feature" to refer to both human-interpretable concepts and SAE hidden layer neurons. Feature: We use the term "feature" to refer to an idealized human-interpretable concept that the model represent in its activations and which a SAE latent may or may not represent. Monosemantic: Referring to a feature or representation that corresponds to a single, clear semantic concept. In the context of SAEs, a monosemantic feature would ideally capture one interpretable aspect of the input.

Interpretable: A latent being interpretable is not well defined in the field, making it difficult to ensure that different authors mean the same thing when referring to SAE interpretability. When we refer to a SAE latent as being interpretable in this work, we mean that it should behave in line with how it appears to behave after inspecting its activation patterns. If an SAE latent appears to track a feature X by a reasonable inspection of its activations but has subtle deviations from this behavior in reality, we say this is not interpretable. We thus measure interpretability via classification performance when a latent appears to be a classifier over some feature. Feature dashboard: A dashboard showing activation patterns and max-activating examples for a SAE latent. Feature dashboards are commonly used to interpret the behavior of an SAE latent. Neuronpedia: A platform, https://neuronpedia.org, which hosts feature dashboards for popular SAEs [20].

Token-aligned latent: A latent which seems to roughly fire on variants of the same token. For instance, a "Snake" token-aligned latent may fire on the tokens "Snake", "SNAKE", "_snakes", etc... Feature splitting: A phenomenon in SAEs introduced by Bricken et al. [2], where a SAE latent tracks a general feature in a narrow SAE, but splits into multiple more specific SAE latents in a wider SAE. For instance, a latent tracking "starts with L" in a narrow SAE may split into a latent tracking
"starts with capital L" and a latent tracking "starts with lowercase L" in a wider SAE.

Feature absorption: A problematic form of feature splitting where a SAE latent appears to track an interpretable feature, but that latent has seemingly arbitrary exception cases where it fails to fire. Instead, a more specific latent "absorbs" the feature direction and fires in place of the main latent. Circuit: In the context of neural network interpretability, a circuit refers to a subgraph of neurons or latents within a neural network that work together to perform a specific function or computation.

The study of circuits aims to understand how different components of a neural network interact to process information and produce outputs. Linear probe: A simple linear classifier (typically logistic regression) trained on the hidden activations of a neural network to predict some property or task. Used to assess what information is linearly decodable from the network's representations. K-sparse probing: A variant of linear probing where only the k most important latents (as determined by some selection method) are used to train the probe. This helps identify which specific neurons or latents are most relevant for a given task. Ablation study: An experimental method where a component of a system (in this case, a neuron or latent in a neural network) is removed or altered to observe its effect on the system's performance. This helps determine the causal importance of the component. Integrated gradients (IG): An attribution method that assigns importance scores to input latents by accumulating gradients along a path from a baseline input to the actual input. In this paper, it's used as an approximation technique for ablation studies. In-context learning (ICL): A paradigm where a language model is given examples of a task within its input prompt, allowing it to adapt to new tasks without fine-tuning. Often used with few-shot learning techniques. Residual stream: In the context of transformer architectures, the residual stream refers to the main information flow that bypasses the self-attention and feed-forward layers through residual connections. Logits: The raw, unnormalized outputs of a neural network's final layer, before any activation function (like softmax) is applied. In language models, logits typically represent the model's scores for each token in the vocabulary.

Activation patching: An interpretability technique where activations at specific locations in a neural network are replaced or modified to observe the effect on the network's output. This helps in understanding the causal role of different parts of the network in producing its final output.

## A.2 Proof: Absorption Decreases Sae Loss For Hierarchical Features

We analyze the effect of a specific form of feature absorption, termed δ-absorption, within a Sparse Autoencoder (SAE) framework. We consider two hierarchically related features, f1 and f2 (where f2 ⊂ f1), and demonstrate that for a defined family of encoder and decoder weights parameterized by δ ∈ [0, 1] (δ = 0 corresponds to no absorption, and δ = 1 corresponds to full absorption):
1. Perfect reconstruction of inputs composed of these features is maintained across all values of δ.

2. The sparsity loss component attributable to these features is a decreasing function of δ, provided the child feature f2 has a non-zero probability of appearing.

3. Consequently, optimizing for sparsity encourages higher values of δ, i.e., greater absorption.

## 1. Preliminaries And Assumptions

H1. Dataset and Features Let D be a dataset. We consider a set of features F = {f1, f2*, . . . , f*d}.

Each feature fi ∈ R
kis a vector with unit norm, ∥fi∥2 = 1. Features are mutually orthogonal:
fi· fj = δij (Kronecker delta), where δij = 1 if i = j and 0 if i ̸= j. An activation h ∈ R
kin the model's residual stream is a linear combination of active features: h =Pj∈ActiveFeatures fj .

H2. Feature Hierarchy and Probabilities We focus on two features f1, f2 ∈ F with a hierarchy f2 ⊂ f1. This implies that if f2 is present in a datapoint, f1 must also be present. The probabilities of observing combinations of f1 and f2 are:
- p(f1, f2) = p11: Probability of f1 and f2 co-occurring (e.g., input is f1 + f2). - p(f1, ¬f2) = p10: Probability of f1 occurring without f2 (e.g., input is f1). - p(¬f1, f2) = p01: Probability of f2 occurring without f1. By the hierarchy assumption, p01 = 0.

- p(¬f1, ¬f2) = p00: Probability of neither f1 nor f2 occurring (e.g., input is 0 or some fk, k ̸= 1, 2).

We assume p11 + p10 + p00 = 1.

H3. Sparse Autoencoder (SAE) Model The SAE reconstructs an input h as hˆ = fϕ(h). The reconstruction is hˆ = Wdz, where z = ReLU(Weh). No bias terms are used. We,i is the i-th row of We (encoder vector for latent i), and Wd,i is the i-th column of Wd (decoder vector for latent i). We analyze two specific latents, z1 and z2, intended to capture f1 and f2. Other latents zj for j > 2 are assumed to perfectly reconstruct other features fj (e.g. We,j = fj , Wd,j = fj ) and do not interact with f1, f2 due to orthogonality.

H4. SAE Loss Function The total loss is L = Lrec+λLsp, where λ > 0. Lrec = Eh∼D h∥h − hˆ∥
22 i.

Lsp = Eh∼D Pi |zi|. Our analysis will focus on the contributions of z1, z2 to Lrec and Lsp.

H5. Definition of δ**-Absorption** We define a specific parameterization for the encoder and decoder weights associated with f1 and f2 by a parameter δ ∈ [0, 1]:
- We,1 = f1 − δf2 - We,2 = f2 - Wd,1 = f1 - Wd,2 = f2 + δf1 δ = 0 represents no absorption, while δ = 1 represents full absorption.

2. Proposition 1: Perfect Reconstruction under δ**-Absorption**
For any δ ∈ [0, 1], and for inputs h *consisting only of* f1, f2 or 0*, the reconstruction* hˆ = Wd,1z1 +
Wd,2z2 perfectly reconstructs h*, i.e., the reconstruction loss component* L
(1,2)
rec due to these features is 0.

Proof We consider the possible input types based on f1, f2: Case 1: h = f1 (only parent feature f1 **is present).** The latent activations are:
z1 = ReLU(We,1 · h) = ReLU((f1 − δf2) · f1)
= ReLU(f1 · f1 − δf2 · f1)
= ReLU(1 − δ · 0) = 1 (by H1)
z2 = ReLU(We,2 · h) = ReLU(f2 · f1) = ReLU(0) = 0 (by H1)
The reconstruction is:
hˆ = z1Wd,1 + z2Wd,2 = 1 · f1 + 0 · (f2 + δf1) = f1 Thus, hˆ = h.

Case 2: h = f1 + f2 (both parent f1 and child f2 **are present).** The latent activations are:
z1 = ReLU(We,1 · h) = ReLU((f1 − δf2) · (f1 + f2))
= ReLU(f1 · f1 + f1 · f2 − δf2 · f1 − δf2 · f2)
= ReLU(1 + 0 − δ · 0 − δ · 1) = ReLU(1 − δ) (by H1)
Since δ ∈ [0, 1], 1 − δ ≥ 0, so z1 = 1 − δ.

z2 = ReLU(We,2 · h) = ReLU(f2 · (f1 + f2))
= ReLU(f2 · f1 + f2 · f2)
= ReLU(0 + 1) = 1 (by H1)
The reconstruction is:
hˆ = z1Wd,1 + z2Wd,2 = (1 − δ)f1 + 1 · (f2 + δf1)
= (1 − δ)f1 + f2 + δf1 = f1 − δf1 + f2 + δf1 = f1 + f2 Thus, hˆ = h.

Case 3: h = 0 (neither f1 nor f2 **is present).**
z1 = ReLU((f1 − δf2) · 0) = 0 z2 = ReLU(f2 · 0) = 0 The reconstruction is:
hˆ = 0 · f1 + 0 · (f2 + δf1) = 0 Thus, hˆ = h.

$\iota\cdot\rho=\iota\cdot\rho$. 
Case 4: h = f2 (only child feature f2 **is present).** This case is disallowed by assumption H2 (p01 = 0), as f2 ⊂ f1 implies f1 must be present if f2 is.

In all permissible cases, h − hˆ = 0, so ∥h − hˆ∥
22 = 0. Therefore, the reconstruction loss component due to f1, f2, denoted L
(1,2)
rec , is 0 for any δ ∈ [0, 1].

3. Proposition 2: Sparsity Loss under δ**-Absorption**
The expected sparsity loss contribution from latents z1 and z2*, denoted* L
(1,2)
sp = Eh∼D[|z1| + |z2|],
is given by:

$${\mathcal{L}}_{s p}^{(1,2)}=p_{11}(2-\delta)$$

Furthermore, its derivative with respect to δ is:

$$\frac{d{\mathcal{L}}_{s p}^{(1,2)}}{d\delta}=-p_{11}$$

Proof We calculate the sum of absolute latent activations |z1| + |z2| for each case from Proposition 1 and weight them by their probabilities (H2):

 - If $h=f_1+f_2$ (probability $p_{11}$): $z_1=1-\delta$, $z_2=1.$ Since $\delta\in[0,1]$, $1-\delta\geq0$, so: $|z_1|=1-\delta$, $|z_2|=1.$ Thus, $|z_1|+|z_2|=(1-\delta)+1=2-\delta$. 
- If h = f1 (probability p10): z1 = 1, z2 = 0. Thus, |z1| + |z2| = 1 + 0 = 1. - If h = 0 (probability p00, neither f1 nor f2 present): z1 = 0, z2 = 0. Thus, |z1| + |z2| =
0 + 0 = 0.

The case corresponding to p01 does not occur.

The expected sparsity loss from z1, z2 is:

$$\begin{array}{c}{{{\mathcal{L}}_{\mathrm{sp}}^{(1,2)}=p_{11}\cdot(2-\delta)+p_{10}\cdot1+p_{00}\cdot0}}\\ {{=p_{11}(2-\delta)+p_{10}}}\end{array}$$

Taking the derivative with respect to δ:

$$\frac{d{\mathcal{L}}_{\mathrm{sp}}^{(1,2)}}{d\delta}=\frac{d}{d\delta}(2p_{11}-\delta p_{11}+p_{10})$$

Since p11 and p10 are constants with respect to δ:

$$\frac{d{\mathcal{L}}_{\mathrm{sp}}^{(1,2)}}{d\delta}=-p_{11}$$

$\square$

## 4. Corollary: Increasing Absorption Decreases Sparsity Loss

If p11 > 0 (i.e., the child feature f2 co-occurs with f1 with non-zero probability), then increasing δ strictly decreases L
(1,2)
sp .

Proof From Proposition 2, dL
(1,2) sp dδ = −p11. If p11 > 0, then −p11 < 0. A negative derivative implies that L
(1,2)
sp is a decreasing function of δ for δ ∈ [0, 1]. The minimum value of L
(1,2)
sp over this interval occurs at δ = 1 (full absorption), yielding L
(1,2)
sp (δ = 1) = p11(2 − 1) + p10 = p11 + p10.

The maximum value occurs at δ = 0 (no absorption), yielding L
(1,2)
sp (δ = 0) = p11(2 − 0) + p10 =
2p11 + p10. □

## 5. Conclusion

Given the specified δ-absorption mechanism for an SAE handling two hierarchical features f1, f2 (where f2 ⊂ f1):
1. Perfect reconstruction of inputs composed of f1 and f2 is maintained irrespective of the degree of absorption δ. Thus, L
(1,2)
rec is unaffected by δ.

2. The sparsity loss component L
(1,2)
sp associated with these features is p11(2 − δ) + p10.

3. If p11 > 0, the total loss L (focusing on the components related to f1, f2) decreases as δ increases because L
(1,2)
rec is constant (zero) and L
(1,2)
sp decreases.

4. Therefore, an optimization process like gradient descent, when minimizing the total loss L = Lrec + λLsp (where λ > 0), will favor increasing δ towards 1, thereby promoting feature absorption for these hierarchically related features, assuming the SAE learns or is constrained to these forms of We and Wd.

This formalizes the argument that, under the given conditions and definitions, absorption is a mechanism that can reduce SAE loss by improving sparsity without harming reconstruction for hierarchical features.

## A.3 Extended Toy Model Experiments

In this section we explore further variants on absorption in toy models. We use the same setting as our main toy model experiment, with four mutually-orthogonal true features, and train an SAE with four latents. Each true feature f ∈ R
50. Unless otherwise stated, every time feature 1 fires feature 0 must also fire, but feature 0 is allowed to fire on its own as well. This is to simulate hierarchal features such as our example "starts with S" and "short" features, where every time the "short" feature fires we expect "starts with S" must also fire since "short" starts with "S", but "starts with S" can fire on its own as well. Feature 2 and 3 are fully independent. All features fire with magnitude 1.0 and variance 0.0 unless otherwise stated.

Magnitude variance causes partial absorption In our main toy model experiment, each true feature fires with magnitude exactly 1.0. This is not very realistic, though - likely there will be some variance in feature firing magnitudes in real LLMs. We simulate this by adding variance of 0.1 to the firing magnitude of feature 0, so the relative magnitudes of feature 0 and 1 are no longer fixed. We show the plots of cosine similarity between SAE encoder and decoder in Figure 10. Here, we still see the same absorption pattern in the SAE encoder and decoder with the latent 3 encoder containing a negative component of feature 1, and the latent 0 decoder merging features 0 and 1. We show some sample true feature firings and corresponding SAE latent activations in Table 2.

Cos sim with true features (feat 1 co-occurs w/feat 0, feat 0 magnitude varies)
SAE encoder SAE decoder
−1.0 −0.5 0.0 0.5 1.0 0 0 3 SAE L
atent 3 SAE L
atent cos sim 1 1 2 2 0 1 2 3 True feature 0 1 2 3 True feature
We see that now the SAE latent tracking feature 0 still fires when the true values of features 0 and 1 are both 1.0, but very weakly. However, if the magnitude of feature 0 drops down to 0.75, then the feature 0 latent fully turns off.

| TRUE FEATURES   | SAE LATENT ACTS   |      |      |      |      |      |      |
|-----------------|-------------------|------|------|------|------|------|------|
| 1.00            | 0.00              | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| 1.00            | 1.00              | 0.00 | 0.00 | 1.27 | 0.00 | 0.00 | 0.22 |
| 0.90            | 1.00              | 0.00 | 0.00 | 1.27 | 0.00 | 0.00 | 0.12 |
| 0.75            | 1.00              | 0.00 | 0.00 | 1.26 | 0.00 | 0.00 | 0.00 |

We call this phenonemon **partial absorption**. In partial absorption, there's co-occurrence between a dense and sparse feature, and the sparse feature absorbs the direction of the dense feature. However, the SAE latent tracking the dense feature still fires when both the dense and sparse feature are active, only very weakly. If the magnitude of the dense feature drops below some threshold, it stops firing entirely. Feature absorption is an optimal strategy for minimizing the L1 loss and maximizing sparsity. However, when a SAE absorbs one latent into another, the absorbing latent loses the ability to modulate the magnitudes of the underlying features relative to each other. The SAE can address this by firing the latent tracking the dense feature as a "correction" to add back some of the dense feature direction into the reconstruction. Since the dense feature latent is firing weakly, it still has lower L1 loss than if the SAE fully separated out the features into their own latents. Imperfect co-occurrence can still lead to partial absorption Next, we test what will happen if feature 1 is more likely to fire if feature 0 is active, but can still fire without feature 0. We set up feature 1 to fire with feature 0 95% of the time, but 5% of the time it can fire on its own. For this experiment, all features fire with magnitude 1.0 and 0 variance. We show the cosine similarities of the SAE encoder and decoder with true features in Figure 11. Some sample feature firings and corresponding SAE activations are shown in Table 3.

Cos sim with true features (feat 1 partially co-occurs w/feat 0)
SAE encoder SAE decoder
−1.0 −0.5 0.0 0.5 1.0 0 0 3 S
AE L
atent 3 S
AE L
atent cos sim 1 1 2 2 0 1 2 3 True feature 0 1 2 3 True feature
We see signs of partial absorption here as well. We see the same absorption pattern in the SAE encoder and decoder as we saw in our other absorption examples, although less severe than the previous examples. We also see in the sample firing patterns that when both feature 0 and 1 fire together, the latent tracking feature 0 fires with noticeably lower magnitude than when feature 0 fires on its own. Here, even though the co-occurrence between features 0 and 1 is not perfect, we still see partial absorption. Absorption also affects TopK SAEs So far, we have only shown feature absorption occurring with standard L1 SAEs. Next, we examine how other absorption affects other architectures using

| TRUE FEATURES   | SAE LATENT ACTS   |      |      |      |      |      |      |
|-----------------|-------------------|------|------|------|------|------|------|
| 1.00            | 0.00              | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| 1.00            | 1.00              | 0.00 | 0.00 | 0.00 | 1.05 | 0.00 | 0.67 |
| 0.00            | 1.00              | 0.00 | 0.00 | 0.00 | 0.95 | 0.00 | 0.00 |

Table 3: Sample feature values and corresponding SAE activations. Feature 1 can only fire if feature 0 is active 95% of the time, but 5% of the time feature 1 can fire on its own. We see signs of partial absorption, where the latent tracking feature 0 fires noticeably more weakly if feature 1 is active. a batch topk SAE [3]. Batch topk SAEs are an improved version of topk SAEs [10] where the top k ∗ B latents are used to reconstruct the SAE input, where B is the batch size. As the topk function enforces sparsity, there is no additional L1 loss term. Topk SAEs are harder to use for very small toy models like our 4-feature toy model above, since if the k is too large relative to the size of the SAE the SAE will not learn correct features. To address this, we use a slightly larger toy model with 12 mutually orthogonal true features. All features fire independently with probability 0.15, except for the first 2 features. Feature 0 is the parent feature in our hierarchy, and fires with probability 0.4. Feature 1 is the child feature, and fires with probability 0.6 only if feature 1 fires, but never fires if feature 1 does not fire. All features fire with magnitude 1.0. We train a batch topk SAE with k = 2. We show the cosine similarities of the SAE encoder and decoder with true features in Figure 12.

Cos sim with true features (feat 1 co-occurs w/feat 0, batch topk SAE)
SAE encoder SAE decoder
−1.00
−0.75
−0.50 −0.25 0.00 0.25 0.50 0.75 1.00 0 0 1 1 2 2 3 3 11 SAE
 Late nt 11 SAE
 Late nt 4 4 cos sim 5 5 6 6 7 7 8 8 9 9 10 10 0 1 2 3 4 5 6 7 8 9 10 11 True feature 0 1 2 3 4 5 6 7 8 9 10 11 True feature
We still see a clear absorption pattern between the latents tracking features 0 and 1 despite the lack of L1 loss. Absorption increases sparsity, which allows the topk SAE to have better reconstruction loss at a given k, and is thus what the SAE learns.

## A.4 Ablation Algorithm A.5 How Good Is Gemma-2 On Character Identification Tasks?

We evaluate how well can Gemma-2-2B identify the first letter or all the letters in a token (spelling the full token). We evaluate the accuracy of the model on all tokens in the LR probe validation set with a prompt containing 10 in-context examples selected at random from the full vocabulary. Our results are shown in Figure 13. We see that performance on the first-letter identification task is high throughout token length, while the full-word spelling performance decreases as the length of the token increases.