# In Search Of Adam'S Secret Sauce

Antonio Orvieto 
∗
ELLIS Institute Tübingen, MPI-IS
Tübingen AI Center, Germany Robert M. Gower CCM, Flatiron Institute, Simons Foundation New York, US

## Abstract

Understanding the remarkable efficacy of Adam when training transformer-based language models has become a central research topic within the optimization community. To gain deeper insights, several simplifications of Adam have been proposed, such as the signed gradient and signed momentum methods. In this work, we conduct an extensive empirical study - training over 1,500 language models across different data configurations and scales - comparing Adam to several known simplified variants. We find that signed momentum methods are faster than SGD, but consistently underperform relative to Adam, even after careful tuning of momentum, clipping setting and learning rates. However, our analysis reveals a compelling option that preserves near-optimal performance while allowing for new insightful reformulations: constraining the Adam momentum parameters to be equal, β1 = β2. Beyond robust performance, this choice affords new theoretical insights, highlights the "*secret sauce*" on top of signed momentum, and grants a precise statistical interpretation: we show that Adam in this setting implements a natural online algorithm for estimating the mean and variance of gradients—one that arises from a mean-field Gaussian variational inference perspective.

## 1 Introduction

Despite a decade of research into efficient and performant adaptive optimizers for deep learning, the *de facto* choice for largescale training today remains Adam [Kingma and Ba, 2014], especially for training language models (LMs) [Grattafiori et al., 2024, Liu et al., 2024]. At the root of this choice is the peculiar geometry of optimization landscapes induced by the transformer architecture [Noci et al., 2022, Zhang et al., 2024a], as well as the noisy/unbalanced nature of tokenized text data [Zhang et al., 2020a, Kunstner et al., 2024]. In recent years, the surge of extremely large-scale and expensive-to-pretrain language models has further pushed the community to better understand Adam's performance and to propose faster, efficient, and robust alternatives. Towards achieving this goal, contemporary studies [Kunstner et al., 2023, Bernstein and Newhouse, 2024] have brought up a close similarity between the performance of Adam and SignSGD [Bernstein et al., 2018] with momentum. While such results are extremely valuable to forward our understanding, they are not precise enough : already at a scale of 160M parameters we found that extensive tuning of Signum (SignSGD with momentum), while closing 96% of the perplexity gap between SGD and Adam, results in a 25% effective slowdown (Figure 1).

∗antonio@tue.ellis.eu.

()%*	$$%+!'*+,&+'*+	
	 	 	 
	
*&+'#&* 

" &,%-$)/+'
	+'# %$)/+'
	+'#
%$)/+'	+'#
)
"& "&
 
)($

."+
/

25% slower
Table 1: **(Signum closes 96% of the perplexity gap between Adam and SGD)** Validation perplexity comparison of widely used optimizers that interpolate between SGD and Adam, evaluated on a language modeling task (160M parameters, 3.2B SlimPajama tokens, sequence length 2048, batch size 256 - Chinchilla optimal). We report the mean and 2-sigma interval of validation perplexity (on 100M held-out tokens) across 3 initialization seeds.

Weight decay is always decoupled [Loshchilov and Hutter, *2019] and set to* 0.1 [Biderman et al., 2023, Liu et al., 2024] except for SGD where we further tune (§B). RMSprop does not use momentum, and Gclip is global norm clipping to 1 (before applying momentum), Cclip is coordinate-wise clipping (after applying momentum). Other hyperparameters, for all other methods, are carefully tuned, see e.g. Figure 2 and §3.

To optimally tune hyperparameters (e.g. Figure *2), we performed a total of 582 full training runs.*

| Adam     | Signum      | RMSprop     | SGD+Cclip   | SignSGD     | SGD+Gclip   | SGD         |             |
|----------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| Val ppl. | 21.86± 0.21 | 23.23± 0.16 | 27.04± 0.34 | 33.40± 0.39 | 36.78± 0.57 | 37.76± 0.61 | 53.62± 5.14 |

While for large-scale training, the slowdown in Figure 1 is not acceptable, it may seem unnecessary or anachronistic to further explain it, in light of recent algorithms claiming to have further improved the performance of Adam, e.g. Muon [Jordan et al., 2024, Liu et al., 2025, Shah et al., 2025], Scion [Pethick et al., 2025], and Shampoo-based [Gupta et al., 2018] methods such as SOAP [Vyas et al., 2025]. However, a close inspection of such optimizers reveals that, while gains over vanilla Adam are solid, most of these methods still use Adam *on a specific subset of parameters*: For instance, in recent scaled-up versions of Muon [Liu et al., 2025, Shah et al., 2025], Adam is used to update embedding, LM heads and normalization parameters 2, and on the other parameters the Muon update is normalized to have a similar RMS value similar to the Adam update. Further, SOAP's improvements stem from the application of Adam in the preconditioner's eigenbasis.

The discussion above and the results in Figure 1 inspires us to further dissect - once again [Balles and Hennig, 2018] - the mechanisms of Adam compared to those of simpler methods in language modeling with transformers.

Towards improving our understanding of Adam, we make the following contributions:
- We perform a large-scale evaluation (∼ 10 thousand NVIDIA A100-SXM4-80GB GPU hours)
of the performance of established algorithms which claim a theoretical or empirical similarity/dissimilarity with Adam on 160M parameters LMs with usual configurations [Biderman et al., 2023, Black et al., 2022], at a compute-optimal budget on different datasets, at different batch-sizes and sequence lengths (up to 2048 tokens). Crucially, we sweep over all momentum parameters for each method, for each learning rate in our grid - for each of our settings. We find that, while clipping and sign descent methods can close most of the gap with SGD, their performance is not satisfactory in comparison to Adam (Figure 2). We make all of our data, e.g. loss dynamics for all our settings, publicly available at https://github.com/aorvieto/SecretSauce.

- Through our extensive tuning of Adam (e.g., Figure 2, comprising 200 distinct hyperparameter settings), we identify one simplification that does perform well: that of setting β1 = β2 (emerging practical choice in contemporary literature [Zhao et al., 2025, Shah et al., 2025, Cattaneo and Shigida, 2025, Zhang et al., 2025]). We validate this finding (§3.2) at different batchsizes, data source, token budget, sequence length and larger scale (410M): β1 = β2 performs at near-optimality across the majority of our experiments, see Figure 3. Given the breadth of our evaluation and the robustness of this finding, we recommend adopting β1 = β2 as the default setting for Adam for training language models at similar data and parameter scales. More broadly, this perspective suggests that Adam can be effectively treated as a one-parameter optimizer (as Signum).

- We show in §4, that reducing β1 = β2 = β to a single parameter, leads to a surprising new interpretation of Adam: it is built on top of a nontrivial yet principled online method for estimating mean and variance of the gradients. Indeed, we can view the two momentum buffers as the result of an online Gaussian Variational inference method for tracking the mean and variance of the gradients as they change across iterations. This viewpoint directly adds to the discussion by Balles and Hennig [2018], yet affords more precision induced by our empirically-informed simplification.

- We offer a toy quadratic example illustrative of our findings, building on top of recent works on the peculiar landscape of transformer-based language modeling problems [Noci et al., 2022, Zhang et al., 2024a]. This example replicates the gaps between tuned SGD, Signum, and Adam with equal betas in a 9-dimensional setting, helpful for future research and to gain intuition.

#$   

#$   

#$   

#$   

#$   

% %
  " 
 # !

"
!"

% %
  "
% %
  "
% %
  "
% %
  " 
  







 	  



 	  



 	  

$$(1)$$
$$(\operatorname{Adam})$$

 	  

## 2 Preliminaries And Related Works

For a signal (sk)k∈N and β ∈ [0, 1), we define the β-normalized exponential moving average:
EMAβ[sk] = βEMAβ[sk−1] + (1 − β)sk, EMAβ[s0] := s0 (or zero). (1)
The Adam optimizer [Kingma and Ba, 2014] without bias correction 3takes the following form:

$$w_{k+1}=w_{k}-\eta_{k}\left(\sqrt{\texttt{EMA}_{\beta_{2}}[g_{k}^{2}]}+\epsilon\right)^{-1}\texttt{EMA}_{\beta_{1}}[g_{k}]$$

where all division and multiplications are element-wise, wk, gk ∈ R
dare model parameters and gradients at iteration k, ηk is the scheduled learning rate, and ϵ > 0 is a small constant. RMSprop [Tieleman and Hinton, 2012] is an earlier method that sets β1 = 0. One special case, and simplification, of Adam is to consider β1 = β2 = ϵ = 0 which gives SignSGD:

wk+1 = wk − ηksign[gk]. (SignSGD)
A practical variant of SignSGD, which has shown strong performance in language modeling [Kunstner et al., 2023], first computes an exponential moving average (EMA) - or momentum - of the gradients before applying the sign operator [Bernstein et al., 2018]:
wk+1 = wk − ηksign[EMAβ[gk]]. (Signum)

$$\mathbf{\Pi}^{-}$$ (SignsGD)
$$w_{k+1}=w_{k}-\eta_{k}\mathrm{{\bf{sign}}}[\mathrm{{\tt{EMA}}}_{\beta}[g_{k}]].$$
$$(\operatorname{Signum})$$

setting and to make gradients more robust to the stochasticity of language [Zhang et al., 2020b]. Global norm clipping (that we abbreviate Gclip), processes gradients fresh out of the backward pass:

$$\mathtt{G c l i p}[g_{k}]=\operatorname*{min}\left\{1,{\frac{1}{\|g_{k}\|_{2}}}\right\}g_{k}.$$

In our experiments, we start from vanilla SGD with momentum: wk+1 = wk −ηkEMAβ[gk] and ablate on the positive effect of Gclip before applying momentum. Regarding coordinate clipping (Cclip), a softer version of sign, we consider applying it to EMAβ[gk] - in connection with Signum. Research on Adam, a short summary. Despite initial concerns on generalization [Wilson et al., 2017] and convergence [Reddi et al., 2018], after the introduction of decoupled weight decay (i.e., AdamW [Loshchilov and Hutter, 2019]) Adam rapidly became the de-facto standard optimizer in deep learning, with works highlighting its landscape adaptation properties [Orvieto et al., 2022] and its debated connections to empirical Fisher preconditioning [Kunstner et al., 2019]. With the advent of Transformers [Vaswani et al., 2017], early works noticed an intriguing gap with SGD performance in language modeling [Xiong et al., 2020] (much larger than what can be observed, e.g., in CNNs on image data), that was initially attributed to heavy-tail noise in text data [Simsekli et al., 2019, Zhang et al., 2020a] - suggesting Adam performance to be correlated with its adaptive coordinate clipping mechanism [Zhang et al., 2020a]. As models became larger and more hardware-demanding, interest spiked in the community to reduce the memory footprint of Adam [Li et al., 2023, Zhang et al., 2024b] and to search for more efficient options [Chen et al., 2023, Liu et al., 2023]. Current trends, draw an intriguing connection between Adam and SignSGD [Bernstein and Newhouse, 2024], and in particular with its momentum variant: Signum [Bernstein et al., 2018]. This connection was first suggested in early attempts to understand Adam's empirical performance [Balles and Hennig, 2018], and has recently gained renewed attention in light of transformer architectures and their heterogeneous optimization landscapes [Noci et al., 2022, Zhang et al., 2024a, Tomihari and Sato, 2025, Kunstner et al., 2024, Zhao et al., 2025]. These landscape-based arguments are now more compelling, as recent evidence shows that Adam and signed momentum methods outperform SGD even in deterministic settings [Kunstner et al., 2023].

Our approach. Although recent literature highlights many connections between Adam and simpler methods such as Signum—which involve fewer hyperparameters, the computational demands of thoroughly studying Adam on small- to medium-scale language models remain prohibitive for most academic optimization researchers. This challenge is amplified by the combinatorial explosion of hyperparameter configurations required for rigorous comparisons. In §3, we aim to provide a comprehensive empirical reference for optimizer performance across a range of language modeling settings. Our key findings are distilled into two main takeaways (Figure 2), which are further supported by theoretical insights in §4.

## 3 Experiments

In our experiments, we systematically explore Transformer-based language models using a nanoGPT [Karpathy, 2022] implementation4enhanced by recent advancements such as Rotational Positional Embeddings [Su et al., 2024], RMSNorm normalization [Zhang and Sennrich, 2019], and SwiGLU activation functions [Shazeer, 2020]. We adopt a robust training protocol inspired by successful practices established in large language models like LLaMa [Touvron et al., 2023], GPT-neox [Black et al., 2022], GPT-J [Wang and Komatsuzaki, 2022] and Pythia [Biderman et al., 2023], leveraging techniques including bfloat16 precision, linear warm-up followed by a cosine annealing schedule [Loshchilov and Hutter, 2016], and global gradient norm clipping (unless specified). Our model configurations follow [Biderman et al., 2023] and are presented, alongside a detailed description of all tuning settings and resources, in §A.

## 3.1 Extensive Benchmarking At 160M Parameters

We conduct 475 compute-optimal pretraining runs on the SlimPajama-627B dataset [Soboleva et al., 2023], using a sequence length of 2048, a batch size of 256, and a decoupled weight decay of 0.1 [Loshchilov and Hutter, 2019] (except for SGD). We always report validation perplexity on a 4https://github.com/Niccolo-Ajroldi/plainLM/tree/main held-out subset of 100M tokens. Results from these tuning sweeps are summarized in Table 1, Figure 2, and Appendix B.1. The runs span the following configurations:
- SGD (131 runs): Tuned parameters include weight decay (too large causes instability), global norm clipping (Gclip). We also consider clipping coordinates after applying momentum (Cclip). For all these options, momentum and learning rates are independently tuned.

- RMSprop (48 runs): Tuned parameters include momentum on the preconditioner and learning rate. - Signum (70 runs): Tuned parameters include global norm clipping, momentum, and learning rate. - Momentum on SignSGD (35 runs): This variant inverts the order of the sign and EMA operations
(and performs worse than Signum). Clipping has no effect here due to the sign operation.

- AdamW (200 runs): Tuned parameters include both momentum terms and the learning rate.

Two additional seeds are provided for the best performing hyperparameter settings, see Table. 1.

Choice for betas grid. While we vary the learning rate by powers of two, our choice of moving average parameters is guided by recent insights into Adam scaling behavior [Malladi et al., 2022, Compagnoni et al., 2025]: we choose β = 1−κ(1−βbase) where βbase = 0.9 and κ ∈ {2
−5, 2
−4*, . . . ,* 2 2}.

This makes it such that the accumulation factor 1/(1 − β) = 1/(κ(1 − βbase)).

Takeaway 1. As shown in Figure 2 and Table 1, optimally tuning Signum with weight decay leads to significant improvements over standard SGD, in line with recent findings [Kunstner et al., 2023, Zhao et al., 2025]. Nonetheless, Adam consistently outperforms the alternatives across most settings, suggesting that it retains a key advantage—a "secret sauce"—that continues to set it apart from better-understood methods in large-scale optimization tasks. This gap is not limited to this specific setup. In §3.2 we discuss results on another dataset (Fineweb), with disabled weight decay, and shorter sequence lengths. Further, we ablate on other potential confounders (initialization of moving averages, bias corrections, Adam ϵ value) in §3.3. Takeaway 2 (a). In Figure 2, we clearly see that β1 = β2 yields near-optimal performance in Adam, for the five β1 values we considered. In § 3.2 we show similar results at different batch sizes, different sequence lengths, and with disabled weight decay on a different dataset. We also extend this observation to 410M parameters models (Figure 5). This empirical finding serves as a basis for our theory in §4.

Takeaway 2 (b). As a corollary to Takeaway 2, Figure 3 shows that the best performance is not only achieved when β1 = β2, but also improves as the two values become closer. Among 500 runs on 160M-parameter models, we observe a clear correlation: lower loss is associated with smaller differences between β1 and β2. This suggests that gradient smoothing (β1) and preconditioner smoothing (β2) should not be treated as independent operations—optimal performance often arises when they act in concert. To put to the test our second takeaway in **different training settings**, we consider shorter sequence lengths (512, Figure 14), higher/lower batch sizes (Figure 16 & Figure 17), different data (Fineweb) and absence of weight decay (Fig, 18). See discussion in §3.2. Standard choice for betas. While in standard deep learning (also Pytorch default) β2 > β1 (0.999, 0.9), in language modeling the choice β1 = 0.9, β2 = 0.95 is much more common. A lower value for β2 was shown to help mitigate loss spikes [Cattaneo and Shigida, 2025, Compagnoni et al., 2025], and several recent studies have started to adopt β1 = β2 = 0.95 as a default [Zhao et al., 2025, Shah et al., 2025, Zhang et al., 2025]. All our findings confirm this choice for tuning (see e.g. Figure 2), of which we evaluate validity extensively for several values of β1.

Val ppl. 

gap to best 0

	

	

batch size 512 

	
0.17 0.34 0.08 Val ppl. 

gap to best batch size 256 


	

	


	

0 0.17 0.34 0.08 Val ppl. 

gap to best batch size 128 0 0.17 0.34 0.08 

	


	

AdamW, β1 = 0.9 AdamW, β1 = 0.95 AdamW, β1 = 0.975 10
−3 10
−2 learning rate 15.7 15.8 15.9 16.0 16.1 16.2 16.3 16.4 16.5 β2 = 0.8 β2 = 0.9 β2 = 0.95 β2 = 0.975 β2 = 0.9875 β2 = 0.99375 fin al t es t p pl best with equal betas best with equal betas 10
−3 10
−2 learning rate best with equal betas 10
−3 10
−2 learning rate
Theoretical relations between betas. We note that a correlation between β parameters was also noted first by Reddi et al. [2018], Alacaoglu et al. [2020] for AMSgrad, and later by Zhang et al. [2022] for Adam, where it is shown that if β2 is large enough and β1 <
√β2, it converges to the neighborhood of critical points. Further, Xie and Li [2024] showed that weight decay in AdamW leads to convergence to a constrained minimizer only if β2 > β1.

## 3.2 Ablations

More Tokens. We find our **Takeaway 2** to also hold at a higher token budget. In §B.2, we show a trend very similar to Fig. 2 for models trained for 2× the Chinchilla-optimal budget. Different batch size. We find our **Takeaway 2** to be robust to batch size. In the same setting as Figure 2 yet at a slightly lower compute budget due to hardware limitations (2.5B parameters), we find that, even at batch size 128 and 512 the choice β1 = β2 yields near-optimal performance. This step involves training 500 models, see §B.4 for visualizations similar to Figure 2 and a discussion. Different sequence length. In §B.3, we find our **Takeaway 2** to also hold at shorter sequence length of 512 (Figure 14). We note that here performance of Signum is closer to that of Adam compared to Figure 2 - yet, Adam is still superior by a substantial margin ( 0.7 validation perplexity), **Takeaway**
1. This pattern agrees well with the results in [Zhao et al., 2025], who found other methods to be competitive with Adam at short context lengths. Our experiments in Figure 14 and Figure 2 suggest that Adam performance particularly shines at higher sequence lengths. Different data and weight decay. In Figure 18 we test both **Takeaway 1** and **Takeaway 2** on Fineweb [Penedo et al., 2024]. We take this opportunity to also deactivate weight decay (λ = 0), as the optimal Signum learning rates in Figure 2 suggest decoupled weight decay w = w − ληw acts differently for the two methods, likely needing different tuning. When deactivated, we still see a substantial gap between Signum and Adam, as well as strong performance with equal betas. Larger Models. We restrict our attention to the SlimPajama dataset and to validation of Takeaway 2. Results are presented in Figure 5, comprising 44 full compute-optimal training runs of 410M parameter models, which confirm yet again strong and near-optimal performance at β1 = β2.

## 3.3 Checking For Confounders

 $%'

	  ( 
	)  ( 
	) ( 
	)  ( 
	) )
""$

	
1e-03
& & 
 
	 

$#$
!!
When comparing Signum with Adam, here are a few confounders that might affect results: The value of ϵ in Adam was shown to be important for numerical stability, and might affect performance [Yuan and Gao, 2020]. We show in Table 2 that one can choose an extremely small ϵ value in our setting. We cross-check the impact of including an ϵ factor in Signum: we found that little can be gained from this strategy (Figure 4). In short, we found that ϵ is not a crucial parameter in our setup. This is also liked to our findings on adaptive mollifiers, cf. §4.

Initialization of moving average parameters. In Figure 4 we also ablate on initialization of the moving average in Signum and found no substantial differences. We perform this same ablation for Adam and report comprehensive results with seeds in §B.6. Bias correction. While bias correction in Adam is helpful in early training, final validation performance is almost unchanged, see the full training curve and results with seeds in §B.6.

| Table 2: Effect of ϵ in AdamW– other parameters optimally tuned for ϵ = 10−8 (setting: Figure 2). All values between ϵ ∈ [10−6 , 10−15] result in a similar validation perplexity. ϵ = 1e − 3 ϵ = 1e − 6 ϵ = 1e − 8 ϵ = 1e − 10 ϵ = 1e − 12 ϵ = 1e − 15 Val ppl 23.34± 0.31 21.56± 0.19 21.86± 0.21 21.87± 0.04 21.89± 0.2 21.91± 0.18   |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## 4 New Viewpoints Of Adam

We now show that restricting to the case β1 = β2 = β yields a useful interpretation of Adam. Since the Adam update is coordinate-wise, it suffices to analyze a single scalar gradient gk ∈ R. Moreover, ablations (Table 2, Table 3) indicate that neither the ϵ-term nor the bias correction significantly affect performance. Thus, for clarity, we set ϵ = 0 and study the simplified Adam update:

$$d_{k}=\frac{\texttt{EMA}_{\beta}[g_{k}]}{\sqrt{\texttt{EMA}_{\beta}[g_{k}^{2}]}}.\tag{1}$$
$$(2)$$
$$({\mathfrak{I}})$$

We next rewrite (proof in the Appendix) the update to explicitly highlight the role of variance.

Proposition 1. Let mk = EMAβ[gk]*. Then the update* (2) *admits the equivalent representation:*

$$d_{k}\ =\ \frac{m_{k}}{\sqrt{m_{k}^{2}+\beta\,E\!M\!\!A\beta\big{[}(m_{k-1}-g_{k})^{2}\big{]}}}.\tag{1}$$

This shows that the denominator depends on the exponential moving average of the squared deviation between the momentum mk−1 and the incoming gradients gk, with an **interesting multiplier** β. As we demonstrate in the next section, this quantity is in fact an online estimator of the gradient variance.

## 4.1 Adam **Estimates Mean And Variance Using Variational Inference**

We show that Adam admits a natural interpretation as an online variational inference method, where

$$m_{k}:=\mathsf{E M A}_{\beta}[g_{k}]\quad\mathrm{and}$$
$$\mathrm{and}\quad\sigma_{k}^{2}:=\beta\,\mathsf{EMA}_{\beta}[(m_{k-1}-g_{k})^{2}]$$

correspond to online estimates of the mean and variance of the stochastic gradients. We reintroduce Adam through this lens.

Suppose we are given a sequence of stochastic gradients {g1*, . . . , g*k}, where each gk is sampled from an unknown Gaussian distribution whose mean and variance may vary with k. Rather than taking steps directly along these noisy gradients, we aim to estimate their mean and variance online and use these estimates to define a more informed search direction.

At iteration k, let (mk, σ2k
) denote our current estimates of the gradient mean and variance, respectively. Upon receiving a new gradient sample gk+1 ∼ N (*m, σ*2) with unknown (*m, σ*2), we wish to update our estimates to (mk+1, σ2k+1) so that it becomes more *likely* that gk+1 was drawn from N (mk+1, σ2k+1). Since we also expect the underlying distribution to vary slowly over time, we prefer that N (mk+1, σ2k+1) remain close to the previous estimate N (mk, σ2k). These two goals—fitting the new observation and ensuring smooth updates—can be traded off via a regularized maximum likelihood problem:

$$\operatorname*{min}_{m,\sigma^{2}\geq0}\;-\log p(g_{k+1}\mid m,\sigma^{2})+\;\frac{1}{\lambda}\mathrm{KL}\left({\cal N}(m_{k},\sigma_{k}^{2})\,\|\,{\cal N}(m,\sigma^{2})\right),$$

$$(4)$$

where p(gk+1 | *m, σ*2) is the Gaussian likelihood, λ ≥ 0 is a regularization parameter, and KL
denotes the Kullback–Leibler divergence:

$$-\log p(g_{k+1}\mid m,\sigma^{2})=\frac{1}{2}\log\sigma^{2}+\frac{1}{2\sigma^{2}}(g_{k+1}-m)^{2},\tag{5}$$ $$\text{KL}\left(\mathcal{N}(m_{k},\sigma_{k}^{2})\parallel\mathcal{N}(m,\sigma^{2})\right)=\frac{1}{2}\left[\frac{\sigma_{k}^{2}}{\sigma^{2}}+\frac{(m_{k}-m)^{2}}{\sigma^{2}}-1-\log\left(\frac{\sigma_{k}^{2}}{\sigma^{2}}\right)\right].\tag{6}$$

The following result, proved in the appendix, characterizes the solution of (4), showing that the moving averages used in Adam correspond exactly to an instance of online variational inference:

  **Theorem 4.1**.: Let $\beta=\frac{1}{1+\lambda}$. Then the solution to the optimization problem (4) is given by
$m_{k+1}=\beta m_{k}+(1-\beta)g_{k+1}=\texttt{EMA}_{\beta}[g_{k+1}]$,  $\sigma_{k+1}^{2}=\beta\sigma_{k}^{2}+\beta(1-\beta)(m_{k}-g_{k+1})^{2}=\beta\texttt{EMA}_{\beta}\left[\left(m_{k}-g_{k+1}\right)^{2}\right]$.  
$$\mathbf{\Omega}(7)$$
$$({\mathfrak{s}})$$
$$(9)$$
2. (8)
As a consequence, the Adam update direction in (3) can be rewritten as

$$d_{k}\ =\ \frac{m_{k}}{\sqrt{m_{k}^{2}+\beta\mathbb{E}M_{\beta}[(m_{k-1}-g_{k})^{2}]}}\ =\ \frac{m_{k}}{\sqrt{m_{k}^{2}+\sigma_{k}^{2}}}=\frac{\text{sign}(m_{k})}{\sqrt{1+\sigma_{k}^{2}/m_{k}^{2}}}.\tag{9}$$  This shows that Adam may be interpreted as an _adaptive mollified_ variant of Sigum, where the 
mollification depends on the local noise-to-signal ratio. This mollified viewpoint aligns well with one of the first papers on understanding Adam [Balles and Hennig, 2018], as discussed after Proposition 1. Using these insights, we can better formalize the *noise-to-signal* interpretation of Adam [Balles and Hennig, 2018] (see also §4.2). Let mk/σk denote the signal-to-noise ratio (SNR). We show that Adam can be viewed as a steepest descent method whose trust region is modulated by the SNR.

To build this connection, consider first the Signum update. It corresponds to the steepest descent direction under an ℓ∞-norm constraint [Balles and Hennig, 2018], solving

$$-{\rm sign}(m_{k})\ =\ {\rm argmin}\ -m_{k}\cdot\theta\quad{\rm subject\ to}\ |\theta|\leq1.$$  That is, Signum selects the direction most aligned with $-m_{k}$ within a unit trust region.  
In contrast, Adam can be interpreted as a steepest descent method with a variable trust region, defined by the (inverse) signal-to-noise ratio:

$$(10)$$
$$-\frac{\text{sign}(m_{k})}{\sqrt{1+\sigma_{k}^{2}/m_{k}^{2}}}\ =\ \underset{\theta\in\mathbb{R}}{\text{argmin}}\ -m_{k}\cdot\theta\quad\text{subject to}|\theta|\leq\frac{1}{\sqrt{1+\sigma_{k}^{2}/m_{k}^{2}}}\,.\tag{11}$$  Here, the effective step size shrinks when the noise dominates the signal, and expands toward 1 as 
uncertainty decreases. In this sense, Adam adapts its update magnitude according to a confidenceweighted trust region.

## 4.2 Comparison With Balles And Hennig [2018]

Balles and Hennig [2018] first drew a connection between Adam, Signum and Signal-to-noise Ratio
regularization. Their observation was as follows. Let mk = EMAβ1[gk], and vk = EMAβ2[g
2k]. We can
trivially re-write the Adam direction as
 ### Don as  ${d_k=\frac{m_k}{\sqrt{v_k}}=\frac{m_k}{\sqrt{m_k^2+v_k-m_k^2}}}$. 
.

If we now *assume* that σ 2k
:= vk − m2k is a measure of variance, then dividing the Adam direction through by |mk|, as done in (9), we arrive at a Signal-to-noise Ratio regularized variant of the Signum method. In particular, as the noise goes to zero (σ 2k → 0), we arrive at the Signum method.

The missing piece in their insight was to show when and if the term vk − m2kis a measure of variance.

We show that β1 = β2, a choice that was not common5at the time of Balles and Hennig [2018],
allows for more precise claims: Proposition 1 shows that when β1 = β2 = β the term vk − m2k is 5Default parameters have for long been β1 = 0.9, β2 = 0.999, see https://docs.pytorch.org/docs/
stable/generated/torch.optim.Adam.html.

precisely equal to βEMAβ[(mk−1 −gk)
2], which in turn is a online estimate of variance (Theorem 4.1).

We further show that vk − m2k only has a precise variance interpretation for the case β1 = β2: indeed, we prove in §C.2 that Adam can be represented as

$$d_{k}=\frac{m_{k}}{\sqrt{m_{k}^{2}+\gamma\,\mathsf{EMA}_{\tau}\left[(a m_{k-1}-b g_{k})^{2}\right]}}$$

$$(12)$$

for some *a, b, γ* ∈ R and τ ∈ (0, 1) *if and only if* β1 = β2. In other words, connecting vk − m2k to variance estimation, and in turn Adam to an SNR-controlled trust region method (11), can only be done precisely for the case of equal betas. Ablating hyperparameters in our reformulation. While (12) reduces to Adam with equal betas if and only if *a, b* = 1 and β = γ = τ , we found it interesting to consider (12), with a = b = 1, as a new method with no precise connection to simultaneous variance and mean estimation, with hyperparameters *β, γ, τ* . In §C.4, we train 150 additional language models ablating on such parameters, and found no advantage in setting β ̸= τ or τ ̸= γ. We believe such evidence further strengthens our claims: best performance is aligned to the theoretical choice τ = γ = β.

block 1 block 2 block 3 Eigenvalues (both Hessians)
Figure 6: **Top row:** Training performance (median and 25%/75% quantiles over 10 seeds) of SGD, Signum*, and* Adam on two 9-dimensional convex quadratic problems (§D) inspired by Zhang et al. *[2024a]. All optimizers* use moving average parameters set to 0.95, with a 10% warmup followed by cosine decay to zero. Both problems share the same Hessian eigenspectrum and have a 3 × 3 *block structure. The landscape on the* left is homogeneous*, with each block containing both large and small eigenvalues. The landscape on the* right is heterogeneous, with each block having eigenvalues of different magnitudes. In this setting, *Adam* clearly outperforms SGD, with Signum closing part of the gap. **Bottom row:** Dynamics of the variance term in Proposition 1. The value of this term varies both across iterations and across blocks, adapting to the local curvature structure. This adaptive behavior improves performance over Signum *in the heterogeneous setting.*
While our theoretical analysis in §4 offers a new perspective on Adam, it is not tied to any specific architecture. To enhance intuition and provide a controlled setting for future work, we validate our findings on a simplified model of transformer loss landscapes introduced by Zhang et al. [2024a], building on signal propagation theory [Noci et al., 2022].

As noted in Zhang et al. [2024a], Kunstner et al. [2024], Zhao et al. [2025], the landscape of autoregressive language models is highly heterogeneous: Hessian blocks associated with semantically distinct parameter groups (e.g., normalization layers, embeddings, or softmax-related parameters) exhibit markedly different eigenspectra and thus demand different learning rates. In contrast to homogeneous models (e.g., CNNs), this heterogeneity is where Adam significantly outperforms SGD [cf. Zucchet and Orvieto, 2024].

Figure 6 illustrates this point. On a toy heterogeneous quadratic landscape, tuned Adam with equal β values substantially outperforms tuned SGD with momentum, echoing results from Zhang et al. [2024a]. We also observe that Signum closes much of the gap but still falls short of Adam. This is consistent with our findings in Table 1 for language models. In Proposition 1, we showed that the key difference between Signum and Adam lies in the variance correction term βEMAβ[(mk−1 − gk)
2] in the denominator. Understanding how this term evolves is essential: it cannot be approximated by a constant. In the second row of Figure 6, we observe that the variance estimate not only varies over time, but also differs in scale across the three blocks—mimicking the parameter groupings in transformer models. This block-wise variation reinforces the idea that the variance term dynamically adapts to the local curvature and cannot be substituted by a fixed value. In Figure 7 and 4, we show a similar effect in heterogeneous quadratic and language models, respectively: replacing βEMAβ[(mk−1 − gk)
2] with a fixed constant ϵ cannot provide the same adaptive effect.

Fixed mollifier on Signum 0 100 200 300 400 500 iteration 10 4 10 2 10 0 10 2 10 4 10 6
= 10.0 = 1.0 = 0.1

= 0.01 lo ss

## 6 Conclusion

We have presented an extensive numerical study of Adam, comparing it against several proposed simplifications. Our main finding is that, on generative language modeling tasks, Adam significantly outperforms these simplified variants. Notably, we observe that setting β1 = β2 is often optimal or near-optimal. Based on this observation, we recommend Adam with β1 = β2 as a simplified model, and we provide a new variational inference interpretation for this setting. Our findings come with some limitations. First, our numerical experiments fix a grid over the hyperparameters; the results are therefore sensitive to the choice of grid, and different grids may lead to different conclusions. However, for all our hyperparameters, we show explicitly all tuning curves demonstrating that we are always at optimality inside the grid (and not at the edge). Second, while β1 = β2 often performs well, we note that at small batch sizes, Figure 3 suggests a slight shift.

Finally, although Theorem 4.1 shows that Adam's two momentum buffers can be interpreted as online estimates of the gradient's mean and variance, it does not explain why these estimates should be arranged into the quotient used in Adam (9). Lemma 1 in [Balles and Hennig, 2018] can provide a starting point to further dissect this interesting choice and explore alternatives.

## Acknowledgements

We would like to thank Niccolo Ajroldi, Sam Liang, Weronika Ormaniec, and Enea Monzio Compagnoni for their comments. We additionally thank the NeurIPS 2025 and ICML 2025 HiLD workshop reviewers for their valuable feedback and references. Antonio Orvieto acknowledges the financial support of the Hector Foundation, and is thankful for the compute made available by MPI-IS and the Tübingen AI ecosystem.

## References

Ahmet Alacaoglu, Yura Malitsky, Panayotis Mertikopoulos, and Volkan Cevher. A new regret analysis for adam-type algorithms. In *International conference on machine learning*, pages 202–210. PMLR, 2020.

Lukas Balles and Philipp Hennig. Dissecting Adam: The Sign, Magnitude and Variance of Stochastic Gradients. In *ICML*, 2018.

Jeremy Bernstein and Laker Newhouse. Old optimizer, new norm: An anthology. arXiv preprint arXiv:2409.20325, 2024.

Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, and Animashree Anandkumar. signsgd:
Compressed optimisation for non-convex problems. In *ICML*, 2018.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In *ICML*, 2023.

Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, et al. Gpt-neox-20b: An open-source autoregressive language model. *arXiv preprint arXiv:2204.06745*, 2022.

Matias D. Cattaneo and Boris Shigida. Tuning adam(w): Default β2 may be too large, 2025. URL
https://mdcattaneo.github.io/papers/Cattaneo-Shigida_2025_TuningAdam.pdf.

Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, et al. Symbolic discovery of optimization algorithms.

Advances in neural information processing systems, 36:49205–49233, 2023.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. *Journal of Machine Learning Research*, 24(240):1–113, 2023.

Enea Monzio Compagnoni, Tianlin Liu, Rustem Islamov, Frank Norbert Proske, Antonio Orvieto, and Aurelien Lucchi. Adaptive methods through the lens of SDEs: Theoretical insights on the role of noise. In ICLR, 2025.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memoryefficient exact attention with io-awareness. *Advances in neural information processing systems*, 35, 2022.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024.

Vineet Gupta, Tomer Koren, and Yoram Singer. Shampoo: Preconditioned stochastic tensor optimization, 2018.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. *arXiv preprint arXiv:2203.15556*, 2022.

Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. URL https:
//kellerjordan.github.io/posts/muon/.

Andrej Karpathy. Nanogpt, 2022.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. *arXiv preprint* arXiv:1412.6980, 2014.

Frederik Kunstner, Philipp Hennig, and Lukas Balles. Limitations of the empirical fisher approximation for natural gradient descent. In *Advances in Neural Information Processing Systems*, 2019.

Frederik Kunstner, Jacques Chen, Jonathan Wilder Lavington, and Mark Schmidt. Noise is not the main factor behind the gap between sgd and adam on transformers, but sign descent might be. In ICLR, 2023.

Frederik Kunstner, Alan Milligan, Robin Yadav, Mark Schmidt, and Alberto Bietti. Heavy-tailed class imbalance and why adam outperforms gradient descent on language models. Advances in Neural Information Processing Systems, 2024.

Bingrui Li, Jianfei Chen, and Jun Zhu. Memory efficient optimizers with 4-bit states. Advances in Neural Information Processing Systems, 36:15136–15171, 2023.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. *arXiv preprint* arXiv:2412.19437, 2024.

Hong Liu, Zhiyuan Li, David Hall, Percy Liang, and Tengyu Ma. Sophia: A scalable stochastic second-order optimizer for language model pre-training. *arXiv preprint arXiv:2305.14342*, 2023.

Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, et al. Muon is scalable for LLM training. arXiv preprint arXiv:2502.16982, 2025.

Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983, 2016.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In *ICLR*, 2019. Sadhika Malladi, Kaifeng Lyu, Abhishek Panigrahi, and Sanjeev Arora. On the sdes and scaling rules for adaptive gradient algorithms. *Advances in Neural Information Processing Systems*, 2022.

Toan Q Nguyen and Julian Salazar. Transformers without tears: Improving the normalization of self-attention. *arXiv preprint arXiv:1910.05895*, 2019.

Lorenzo Noci, Sotiris Anagnostidis, Luca Biggio, Antonio Orvieto, Sidak Pal Singh, and Aurelien Lucchi. Signal propagation in transformers: Theoretical perspectives and the role of rank collapse.

Advances in Neural Information Processing Systems, 2022.

Antonio Orvieto, Jonas Kohler, Dario Pavllo, Thomas Hofmann, and Aurélien Lucchi. Vanishing curvature in randomly initialized deep relu networks. In *AISTATS*, pages 7942–7975, 2022.

Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. On the difficulty of training recurrent neural networks. In *ICML*, 2013.

Guilherme Penedo, Hynek Kydlícek, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin ˇ
Raffel, Leandro Von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at scale. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. URL https://openreview.net/forum?id= n6SCkn2QaG.

Thomas Pethick, Wanyun Xie, Kimon Antonakopoulos, Zhenyu Zhu, Antonio Silveti-Falls, and Volkan Cevher. Training deep learning models with norm-constrained lmos. arXiv preprint arXiv:2502.07529, 2025.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. *OpenAI blog*, 1(8):9, 2019.

Sashank J Reddi, Satyen Kale, and Sanjiv Kumar. On the convergence of adam and beyond. In International Conference on Learning Representations, 2018.

Ishaan Shah, Anthony M Polloreno, Karl Stratos, Philip Monk, Adarsh Chaluvaraju, Andrew Hojel, Andrew Ma, Anil Thomas, Ashish Tanwer, Darsh J Shah, et al. Practical efficiency of muon for pretraining. *arXiv preprint arXiv:2505.02222*, 2025.

Noam Shazeer. Glu variants improve transformer. *arXiv preprint arXiv:2002.05202*, 2020. Umut Simsekli, Levent Sagun, and Mert Gurbuzbalaban. A tail-index analysis of stochastic gradient noise in deep neural networks. In ICML, 2019.

Daria Soboleva, Faisal Al-Khateeb, Robert Myers, Jacob R Steeves, Joel Hestness, and Nolan Dey. SlimPajama: A 627B token cleaned and deduplicated version of RedPajama, 2023. URL https://huggingface.co/datasets/cerebras/SlimPajama-627B.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. *Neurocomputing*, 568:127063, 2024.

Tijmen Tieleman and Geoffrey Hinton. Lecture 6.5-rmsprop, coursera: Neural networks for machine learning. *University of Toronto, Technical Report*, 6, 2012.

Akiyoshi Tomihari and Issei Sato. Understanding why adam outperforms sgd: Gradient heterogeneity in transformers. *arXiv preprint arXiv:2502.00213*, 2025.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, and Sham M. Kakade. SOAP: Improving and stabilizing shampoo using adam for language modeling. In *ICLR*, 2025.

Ben Wang and Aran Komatsuzaki. Gpt-j-6b: A 6 billion parameter autoregressive language model.

2021. *URL https://github. com/kingoflolz/mesh-transformer-jax*, page 8, 2022.

Ashia C Wilson, Rebecca Roelofs, Mitchell Stern, Nati Srebro, and Benjamin Recht. The marginal value of adaptive gradient methods in machine learning. *Advances in neural information processing* systems, 30, 2017.

Shuo Xie and Zhiyuan Li. Implicit bias of adamw: ℓ∞-norm constrained optimization. In *ICML*,
2024.

Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tieyan Liu. On layer normalization in the transformer architecture.

In *International conference on machine learning*, pages 10524–10533. PMLR, 2020.

Wei Yuan and Kai-Xin Gao. Eadam optimizer: How ϵ impact adam. *arXiv preprint arXiv:2011.02150*,
140, 2020.

Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32, 2019.

Hanlin Zhang, Depen Morwani, Nikhil Vyas, Jingfeng Wu, Difan Zou, Udaya Ghai, Dean Foster, and Sham M. Kakade. How does critical batch size scale in pre-training? In *ICLR*, 2025.

Jingzhao Zhang, Tianxing He, Suvrit Sra, and Ali Jadbabaie. Why gradient clipping accelerates training: A theoretical justification for adaptivity. In *ICLR*, 2020a.

Jingzhao Zhang, Sai Praneeth Karimireddy, Andreas Veit, Seungyeon Kim, Sashank Reddi, Sanjiv Kumar, and Suvrit Sra. Why are adaptive methods good for attention models? Advances in Neural Information Processing Systems, 33:15383–15393, 2020b.

Yushun Zhang, Congliang Chen, Naichen Shi, Ruoyu Sun, and Zhi-Quan Luo. Adam can converge without any modification on update rules. *Advances in Neural Information Processing Systems*, 2022.

Yushun Zhang, Congliang Chen, Tian Ding, Ziniu Li, Ruoyu Sun, and Zhi-Quan Luo. Why transformers need adam: A hessian perspective. In *Neural Information Processing Systems*, 2024a.

Yushun Zhang, Congliang Chen, Ziniu Li, Tian Ding, Chenwei Wu, Diederik P Kingma, Yinyu Ye, Zhi-Quan Luo, and Ruoyu Sun. Adam-mini: Use fewer learning rates to gain more, 2024b.

Rosie Zhao, Depen Morwani, David Brandfonbrener, Nikhil Vyas, and Sham M Kakade. Deconstructing what makes a good optimizer for autoregressive language models. In *ICLR*, 2025.

Nicolas Zucchet and Antonio Orvieto. Recurrent neural networks: vanishing and exploding gradients are not the end of the story. *Advances in Neural Information Processing Systems*, 2024.

## Contents

1 Introduction 1 2 Preliminaries and Related Works 3 3 Experiments 4 3.1 Extensive benchmarking at 160M parameters . . . . . . . . . . . . . . . . . . . . 4 3.2 Ablations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6 3.3 Checking for confounders . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6 4 New Viewpoints of Adam 7 4.1 Adam Estimates Mean and Variance using Variational Inference . . . . . . . . . . 7 4.2 Comparison with Balles and Hennig [2018] . . . . . . . . . . . . . . . . . . . . . 8 5 Why an adaptive trust region? Insights from heterogeneous quadratics 9 6 Conclusion 10 A Experimental details 16 A.1 Experiments on SlimPajama - 160M parameters model . . . . . . . . . . . . . . . 17 A.1.1 Sequence Length 2048, Batch size 256, 3.2 B Tokens (6200 gradient steps) 17 A.1.2 Sequence Length 2048, Batch size 256, 6.4 B Tokens (12400 gradient steps) 18 A.1.3 Sequence Length 512, Batch size 256, 3.2 B Tokens (24800 gradient steps) 18 A.1.4 Sequence Length 2048, Variable batch size, 2.5 B Tokens . . . . . . . . . . 19 A.2 Experiments on SlimPajama - 410M parameters model, 8.2 B tokens . . . . . . . . 19 A.3 Experiments on Fineweb - 160M parameters model, 3.2B tokens - no weight decay 20 B Complementary Experimental Results 20 B.1 Tuning for Table 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20 B.2 Effect of More Training Tokens in Figure 2 . . . . . . . . . . . . . . . . . . . . . 22 B.3 Effect of Shorter Sequence Length in Figure 2 . . . . . . . . . . . . . . . . . . . . 23 B.4 Batch size ablation for Figure 2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24 B.5 Figure 2 on Fineweb (no weight decay) . . . . . . . . . . . . . . . . . . . . . . . 25 B.6 Effect of Bias Correction and Zero Initialization on Adam . . . . . . . . . . . . . . 25 C Missing proofs and derivations 26 C.1 Proof of Proposition 1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26 C.2 Generalization of Proposition 1 - Necessity of equal betas for variance interpretation 26 C.3 Proof of Theorem 4.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28 C.4 Performance of generalized Adam reformulation . . . . . . . . . . . . . . . . . . 29 D Toy Quadratic Example 30

## A Experimental Details

For pre-training Transformers on Causal Language Modeling, we build upon the nanoGPT [Karpathy, 2022] implementation, augmenting it with Rotational Positional Embedding [Su et al., 2024], RM- SNorm [Zhang and Sennrich, 2019], and SwiGLU [Shazeer, 2020]. All our models have a vocabulary size of 50280 and make use of GPT-Neox tokenizer [Black et al., 2022]. We adopt an enhanced training recipe, made popular by large language models such as LLaMa [Touvron et al., 2023]. These modifications include: training in bfloat16; employing a linear learning rate warm-up for 10% of the training steps, followed by cosine annealing to 1e − 5. Global norm clipping is used (unless specified or ablated upon) for gradients with norms above 1 (on the raw gradient, as a first step). We have no weight tying between the embedding and the last linear layer. We always report validation perplexity on a separate subset consisting of 100M tokens. Seeds, when provided, are relative to distinct network initialization.

Computational Resources. All our experiments at a 160M parameter scale are performed on a single NVIDIA A100-SXM4-80GB. At compute optimality (most of our experiments) each run takes approximately 5.83 hours. Our runs at a 410M parameter scale are performed on 8 NVIDIA A100-SXM4-80GB GPUs, and each run here takes approximately 4.83 hours. For all our runs, we fill up memory and optimize to minimize the gradient accumulation steps (usually, around 8). Code. All our runs use the repository https://github.com/Niccolo-Ajroldi/plainLM
Model settings (160M). We use the same configuration as [Biderman et al., 2023]: https:// github.com/EleutherAI/pythia/blob/main/models/160M/pythia-160m.yml
- *Layers:* 12 Transformer [Vaswani et al., 2017] layers - *Attention heads:* 12 - *Hidden size:* 768 - *Attention implementation:* Flashattention [Dao et al., 2022]. - *MLP type:* SwiGLU [Shazeer, 2020] with expansion factor 8/3.

- *Backbone:* PreLN transformer [Xiong et al., 2020] with skip connections.

- *Normalization:* RMSnorm [Zhang and Sennrich, 2019] for both Attention and MLP. - *Position embeddings:* Rotary embeddings (RoPE) to 25% of dimensions ([Su et al., 2024])
- *Initialization:* the MLP and Attention output weights are initialized with variance 0.02/p2\#layers (scaling also similar to [Radford et al., 2019]). All other weights (comprising embeddings) are initialized with a standard deviation of 0.02 (Nguyen and Salazar [2019], Wang and Komatsuzaki [2022], Sec. 2.2). Biases are always initialized at zero.

- *Precision:* Mixed precision FP16 enabled. - *Dropout:* Disabled for both hidden and attention layers (see also Chowdhery et al. [2023]).

Model settings (410 M). We use the same setting as [Biderman et al., 2023], configuration can be found here: https://github.com/EleutherAI/pythia/blob/main/models/410M/
pythia-410m-deduped.yml
- *Layers:* 24 Transformer layers - *Attention heads:* 16 - *Hidden size:* 1024 - Other settings as 160M parameters.

## A.1 Experiments On Slimpajama - 160M **Parameters Model**

On the Cerebras SlimPajama-627B [Soboleva et al., 2023] dataset: https://huggingface.co/ datasets/cerebras/SlimPajama-627B at a 160M scale we present three experimental sections:
- Section A.1.1 - core setting, ablating on **all optimizers**. - Section A.1.3 - ablating on a **smaller sequence length**. - Section A.1.4 - ablating at **different batch sizes**.

A.1.1 Sequence Length 2048, Batch size 256, 3.2 B Tokens (6200 gradient steps)
This setup comprises a total of 747 full training runs. We always use warm-up (10%) and cosine anneal until a learning rate of 1e − 5. This setting is Chinchilla-optimal (20 tokens/parameter).

λ here denotes the weight decay, always decoupled [Loshchilov and Hutter, 2019]. Core experiments: These are the core experimental results for this setting.

- SGD with momentum β and **global norm clipping** to 1 (Gclip, dampening to 1 − β)
- *84 full runs* (Figure 8, top).

(*η, β, λ*) ∈ [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
× [0.*9875*, 0.975, 0.95, 0.9] × [0, 1e − 3, 1e − 4].

- SGD with momentum β with (1) **global norm clipping** of raw gradient to 1 (Gclip) and (2)
coordinate clipping (Cclip) to 1 after momentum is applied. Dampening is set to 1 − β, λ (weight decay) is set to 0, as the previous point revealed decreasing performance on SGD
- *24 full runs* (Figure 8, bottom).

(*η, β, λ*) ∈ [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]
× [0.9875, 0.975, 0.95, 0.9]
- SGD with momentum β (vanilla, dampening to 1 − β, **no clipping**). λ = 0 (weight decay).

- *28 full runs* (Figure 9)
(η, β) ∈ [0.25, 0.125, 0.*0625*, 0.03125, 0.015625, 0.0078125, 0.00390625]
× [0.*9875*, 0.975, 0.95, 0.9].

- **Adam** with global norm clipping to 1 and with λ = 0.1 (weight decay) and ϵ = 1e − 8
(usual Pytorch setup, see also Biderman et al. [2023]).

- *200 full runs* (Figure 2)
(η, β1, β2) ∈ [0.016, 0.008, 0.004, 0.002, 0.001]
× [0.9875, 0.975, 0.95, 0.9, 0.8] × [0.996875, 0.99375, 0.9875, 0.975, 0.95, 0.9, 0.8, 0.6]
- **Adam without global norm clipping** and with λ = 0.1 (weight decay) and ϵ = 1e − 8
(usual Pytorch setup, see also Biderman et al. [2023]).

- *165 full runs* (Figure 12)
(η, β1, β2) ∈ [0.032, 0.016, 0.008, 0.004, 0.002, 0.001]
× [0.975, 0.95, 0.9, 0.8] × [0.9875, 0.975, 0.95, 0.9, 0.8, 0.6]
- **RMSprop** implemented using the AdamW Pytorch class using β1 = 0. We again use λ = 0.1 (weight decay) and ϵ = 1e − 8.

- *48 full runs* (Figure 10).

(*η, β*2) ∈ [0.004, 0.002, 0.001, 0.0005, 0.*00025*, 0.000125]
× [0.9875, 0.975, 0.95, 0.9, 0.8, 0.6, 0.4, 0.0]
- **Signum** with weight decay λ = 0.1 as also suggested by [Zhao et al., 2025] (their Figure 5, top left panel). We **ablate on presence of global norm gradient clipping** (to norm 1).

- *70 full runs* (Figure 2).

(*η, β,* clip) ∈ [0.004, 0.002, 0.001, 0.*0005*, 0.00025, 0.000125, 0.0000625]
× [0.*9875*, 0.975, 0.95, 0.9, 0.8] × [True, False]
Note that Signum with and without gradient clipping are two different methods: here, clipped gradients are first averaged and only then the sign is taken. Instead, clipping on the EMA of signed gradients (next method) should have no effect (apart from non-determinism).

- **EMASign** with weight decay λ = 0.1. We ablate on the presence of global norm gradient clipping (to norm 1) *out of mistake*: the two methods are equal!

- *70 runs (35 duplicate runs)* (Figure 11)
(*η, β,* clip) ∈ [0.001, 0.0005, 0.00025, 0.000125, 0.0000625, 0.00003125, 0.000015625]
× [0.9875, 0.975, 0.95, 0.9, 0.8] × [True, False]
Ablations: These ablations were performed to test side-claims in the paper.

- **Adam** with global norm clipping to 1 and λ = 0.1, β1 = β2 = 0.95, η = 0.008 (best setup from Figure 2). We report performance for 3 seeds using different ϵ values.

- *18 full runs* (Table 2).

ϵ ∈ [1e − 3, 1e − 6, 1e − 8, 1e − 10, 1e − 12, 1e − 15],
and influence of initializing exponential moving averages to zero (default, ZI) or to the stochastic quantity of interest (gradient initialization, GI). At the same time, we try to remove bias correction. These experiments are presented with 3 random initialization seeds:
- *9 full runs* (Table 3).

- **Signum** with global norm clipping to 1 and λ = 0.1, β = 0.95 (best setting from Figure 2):
we ablate on fixed mollifiers for zero-initialized (ZI) or gradient-initialized (GI) momentum.

The mollified we study is mk/(
√mk + ϵ):
(η, ϵ) ∈ [0.001, 0.0005, 0.00025, 0.000125] × [1e − 3, 1e − 6, 1e − 9]
- *12 full runs* (Table 2).

We additionally test the influence of ZI vs. GI with three random seeds at ϵ = 0.

- *5 full runs* (Table 3).

Other: for the best-performing variants of core experiments, we initialize the model with two other random seeds. This accounts for
- *14 additional full runs* (Table 1).

## A.1.2 Sequence Length 2048, Batch Size 256, 6.4 B Tokens (12400 Gradient Steps)

The setup here is exactly as in §A.1.1, but we train for 2× the token budget. We test our core claim (β1 = β2 works well), and hence we run:
- **Adam** with global norm clipping to 1 and with λ = 0.1 (weight decay) and ϵ = 1e − 8.

- *168 full runs* (Figure 13)
(*η, β*1, β2) ∈ [0.032, 0.016, 0.008, 0.004, 0.002, 0.001, 0.0005]
× [0.9875, 0.975, 0.95, 0.9] × [0.99375, 0.9875, 0.975, 0.95, 0.9, 0.8]

## A.1.3 Sequence Length 512, Batch Size 256, 3.2 B Tokens (24800 Gradient Steps)

This setup comprises a total of 55 full training runs. We test our core claims (Signum underperforms Adam, β1 = β2 works well) at a smaller sequence length. Setting is exactly the same as §A.1.1 for all methods, unless stated otherwise.

- **Adam**, we limit this ablation to β1 = 0.95,
(*η, β*2) ∈ [0.001, 0.002, 0.004, 0.008, 0.016] × [0.*99375*, 0.9875, 0.975, 0.95, 0.9, 0.8]
- *25 full runs* (Figure 14).

- **Signum**, we do a full ablation using global norm gradient clipping to 1.

(η, β) ∈ [0.0000625, 0.000125, 0.00025, 0.0005, 0.001, 0.002]×[0.9875, 0.975, 0.95, 0.9, 0.8]
- *30 full runs* (Figure 14).

## A.1.4 Sequence Length 2048, Variable Batch Size, 2.5 B Tokens

We use here a slightly reduced token budget (2.5B, 20 tokens for every non-embedding parameter) and run the same Adam tuning experiment presented in Figure 2 for batch size 256. We actually run this experiment again at a batch size of 256, and test batch sizes of 128 and 512 reducing or doubling the number of steps accordingly (same token budget). The sequence length is still 2048, and the dataset SlimPajama. Due to the reduced number of tokens, each run takes approximately 4.7 hours on our hardware. We implement variation of batch size using gradient accumulation (4, 8, 16) at a micro-batch size of 32 sequences. This setup comprises a total of 500 full training runs. Adam with λ = 0.1 (weight decay) and ϵ = 1e − 8 (usual setup, see Biderman et al. [2023]), we clip gradients to global norm 1. - For batch size 256:
(η, β1, β2) ∈ [0.016, 0.008, 0.004, 0.002, 0.001]
× [0.9875, 0.975, 0.95, 0.9, 0.8] × [0.996875, 0.99375, 0.9875, 0.975, 0.95, 0.9, 0.8, 0.6]
- For batch size 128 and 512:
(η, β1, β2) ∈ [0.0005, 0.001, 0.*0014*, 0.002, 0.0028, 0.004, 0.*0056*, 0.008, 0.0112, 0.016]
× [0.975, 0.95, 0.9] × 1 − [4, 2, 1, 0.5, 0.25] · (1 − β1) (i.e. 3 higher and 2 lower values in grid)
Note that here we overturned the learning rate, the reason for this is the square root scaling law in Malladi et al. [2022], Compagnoni et al. [2025]: if batch size scales by 2, learning rate should scale as 
√2. We see in §B.4 that this indeed seems to hold true, yet noise prevents us from making precise verification claims.

- *500 full runs* (§B.4).

## A.2 Experiments On Slimpajama - 410M **Parameters Model, 8.2 B Tokens**

All our experiments here use the Cerebras SlimPajama-627B [Soboleva et al., 2023] dataset: https:
//huggingface.co/datasets/cerebras/SlimPajama-627B. We focus on evaluating whether β1 = β2 yields good performance in this settings. We scale up the batch size by a factor 2 compared to Section A.1, as suggested by [Zhang et al., 2025]. We perform our experiments at compute optimality (8.2B tokens, 20 tokens per parameter). Adam with λ = 0.1 (weight decay) and ϵ = 1e − 8 (usual setup, see Biderman et al. [2023]), we clip gradients to global norm 1:
- β1 = 0.9
(*η, β*2) ∈ [0.016, 0.008, 0.004, 0.002] × [0.95, 0.9, 0.8]
- β1 = 0.95
(*η, β*2) ∈ [0.016, 0.008, 0.004, 0.002] × [0.9875, 0.975, 0.95, 0.9]
- β1 = 0.975
(*η, β*2) ∈ [0.016, 0.008, 0.004, 0.002] × [0.99375, 0.*9875*, 0.975, 0.95]
- *44 full runs* (Figure 5).

## A.3 Experiments On Fineweb - 160M **Parameters Model, 3.2B Tokens - No Weight Decay**

While testing our claims on a different dataset, we also crucially *remove weight decay* here. Our setting is otherwise identical to that of §A.1.1: Sequence length is 2048, batch size is 256, model has 160 parameters and we train on 3.2B tokens from Fineweb [Penedo et al., 2024] https:// huggingface.co/datasets/HuggingFaceFW/fineweb.

- **Adam** with λ = 0 (no weight decay!) and ϵ = 1e − 8 (usual setup, see Biderman et al.

[2023]). We clip gradients to global norm 1.

(η, β1, β2) ∈ [0.032, 0.016, 0.008, 0.004, 0.002, 0.001]
× [0.975, 0.95, 0.9] × [0.9875, 0.975, 0.95, 0.9, 0.8]
- *90 full runs* (Figure 18)
- **Signum** with λ = 0 (no weight decay) as also suggested by [Zhao et al., 2025] (Figure 5, top left panel). We clip gradients to global norm 1.

(*η, β*) ∈ [0.004, 0.002, 0.001, 0.0005, 0.0000625, 0.*00025*, 0.000125]
× [0.975, 0.95, 0.9]
- *24 full runs* (Figure 18).

## B Complementary Experimental Results

The results in this section complement the discussion in §3. We organize them in 5 subsections, and report all technical details in §A.

- §B.1 outlines all hyperparameter tuning curves for the setting in Table 1 for SGD (with/without clipping and with/without weight decay) - Figure 8 and 9, RMSprop without momentum - Figure 10, and momentum on top of SignSGD - Figure 11.

- §B.3 validates that β1 = β2 is a strong-performing option for Adam at a shorter sequence length. Here, we also show that Signum performance is still suboptimal (cf. Figure 2).

- §B.4 validates that β1 = β2 is a strong-performing option for Adam across different batchsizes. This data, comprising training 500 models, is summarized in Figure 3.

- §B.5 reproduces the Signum-Adam gap on Fineweb [Penedo et al., 2024]. Compared to Figure 2 and the settings above, here we compare at zero weight decay to eliminate this additional confounder.

- §B.6 confirms on the validity of our findings when ablating on nuances of Signum and Adam such as initialization and bias correction. These findings complement §3.3.

## B.1 Tuning For Table 1

Setup Summary. 160 M parameters LM on SlimPajama, trained for 3.2 B tokens at a batchsize of 256 × 2048 sequence length. Comment. Our objective here is to tune to best, despite the combinatorially exploding number of options, our methods in Table 1. Details regarding our hyperparameters grid and model configurations are reported in §A. We remind that tuning for Signum and Adam is presented directly in the main paper as Figure 2. **All figures below show optimal tuning jointly in learning rate and momentum** space. While tuning for RMSprop and momentum on SignSGD is straightforward, SGD requires more attention: we found that removing weight decay was always beneficial when global norm clipping the raw gradient, hence we adopt this option also for the non-clipped variant, and for the variant that includes an additional coordinate clipping step after applying momentum. We believe this is due to the decoupled nature of weight decay, combined with the high learning rates required for good performance in SGD.

Finalizing Table 1. After careful tuning, we select for each method the best configuration (given by figures below) and run two additional seeds to report final results with 2-sigma confidence bars.