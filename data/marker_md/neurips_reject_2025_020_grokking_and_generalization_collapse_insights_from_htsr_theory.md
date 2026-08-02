# Grokking and generalization Collapse: Insights from **HTSR** theory

Anonymous Author(s)

Affiliation

Address

email

### Abstract

 Grokking is a surprising phenomenon in neural network training where test ac- curacy remains low for an extended period despite near-perfect training accuracy, only to suddenly leap to strong generalization. In this work, we study grokking using a depth-3, width-200 ReLU MLP trained on a subset of MNIST. We inves- tigate it's long-term dynamics under both weight-decay and, critically, no-decay regimes—the latter often characterized by increasing l weight norms. Our pri- mary tool is the theory of Heavy-Tailed Self-Regularization (HTSR), where we track the heavy-tailed exponent α. We find that α reliably predicts both the initial grokking transition and subsequent anti-grokking. We benchmark these insights against four prior approaches: progress measures—Activation Sparsity, Absolute Weight Entropy, and Approximate Local Circuit Complexity —and weight norm (l ) analysis. Our experiments show that while comparative approaches register significant changes, in this regime of increasing l norm, the heavy-tailed expo- nent α demonstrates a unique correlation with the ensuing large, long-term dip in test accuracy, a signal not reliably captured by most other measures.

 Extending our zero weight decay experiment significantly beyond typical timescales (10<sup>5</sup> to approximately 10<sup>7</sup> optimization steps), we reveal a late-stage catastrophic generalization collapse ("anti-grokking"), characterized by a dramatic drop in test accuracy (over 25 percentage points) while training accuracy remains perfect; notably, the heavy-tail metric α uniquely provides an early warning of this impending collapse. Our results underscore the utility of Heavy-Tailed Self-Regularization theory for tracking generalization dynamics, even in the challenging regimes without explicit weight decay regularization.

### 1 Introduction

 Grokking is an intriguing phenomenon where a neural network achieves near-perfect training accu- racy quickly, yet the test accuracy lags significantly, often near chance level, before abruptly surging towards high generalization [\[15\]](#page-9-0). Figure [1](#page-1-0) illustrates this for a depth-3, width-200 ReLU MLP trained on a subset of MNIST.

 To dissect this phenomenon and uncover deeper dynamics, our primary analytical lens is the recently developed theory of Heavy-Tailed Self-Regularization (HTSR), following Martin et.al.[\[10\]](#page-9-1). The HTSR theory examines the empirical spectral density (ESD) of individual layer weight matrices (W), quantified by the heavy-tailed power law (PL) exponent α. We find α provides a sensitive measure of correlation structure within layers, tracking the transition into the grokking phase, and crucially, predicting a subsequent decrease in generalization.

- 1. Weight Norm Analysis: Motivated by studies like Liu et al. [\[6\]](#page-9-2), we examine the l norm of the weights. We observe that grokking occurs even without weight decay (leading to an increasing norm), suggesting weight norm alone is not a complete explanation, confirming the weight-norm related findings by Golechha [\[2\]](#page-9-3) . 2. Progress Measures: We utilize metrics proposed by Golechha [\[2\]](#page-9-3).—Activation Sparsity, Absolute Weight Entropy, and Approximate Local Circuit Complexity—which capture broader structural and functional changes in the network during training.

![](_page_1_Figure_1.jpeg)

Figure 1: The three phases of grokking. Training curves for a depth-3, width-200 MLP on MNIST. The initial pre-grokking phase (grey): training accuracy (red line) surges at 10<sup>2</sup> steps, saturating between 10<sup>4</sup>−10<sup>5</sup> steps, while test accuracy (purple line) remains low; the grokking phase (yellow): with test accuracy rapidly increasing after ∼ 10<sup>5</sup> steps, and reaching a maximum at 10<sup>6</sup> steps; and the newly revealed late-stage anti-grokking phase (green): test accuracy collapses (to 0.5).

 Our Contributions: Our work makes several related contributions that helps explain the underlying mechanisms associated with the grokking phenomena:

- 1. By extending training significantly (up to 10<sup>7</sup> steps) under zero weight decay (W D = 0), we identify and characterize late-stage generalization collapse: a substantial drop in test accuracy long after initial grokking, despite perfect training accuracy and a continually increasing l weight norm. We call this anti-grokking. 2. We show that the HTSR layer quality metric α (the heavy-tailed power-law (PL) exponent), effectively tracks the grokking transition under both the traditional setting of weight decay (W D > 0) and zero weight decay W D = 0), outperforming the l weight norm and the other progress metrics. Only the HTSR α can distinguish between all 3 phases of grokking. 3. We identify the mechanism of the pre-grokking phase, where the training accuracy is per- fect but the model does not generalize. This phase occurs because only a subset of the model layers are well trained (i.e. α ≤ 4), whereas at least one layer is underfit (i.e. α ≥ 5). Moreover, the layers can show great variability between training runs, indicating their instabili. Importantly, the layer α's here are distinct from those in the anti-grokking phases, despite both phases having perfect training accuracy and low test accuracy. . 4. We demonstrate that when the HTSR PL exponent α < 2, this identifies the collapse. Also, in this phase, we observe the presence of anomalous rank-one (or greater) perturbations in one or more underlying layer weight matrices W. We call these correlation traps and identify them by randomizing W elementwise, forming Wrand , and looking for unusually large eigenvalues, λtrap ≫ λ <sup>+</sup> (where λ <sup>+</sup> is the right-most edge of the associated Marchenko-Pastur (MP) distribution [\[9\]](#page-9-4)).

# 2 Related Work

 Grokking [\[15\]](#page-9-0), the delayed emergence of generalization well after training accuracy saturation, has prompted significant research into its underlying mechanisms. Initial studies often explored grokking in algorithmic tasks [\[12,](#page-9-5) [13,](#page-9-6) [16\]](#page-9-7), frequently linking the phenomenon to the presence of weight decay (WD) which favors simpler, lower-norm solutions [\[6\]](#page-9-2). Other approaches include <sup>70</sup> mechanistic interpretability [\[13\]](#page-9-6) and analyses identifying competing memorization and generaliza-<sup>71</sup> tion circuits [\[16,](#page-9-7) [12\]](#page-9-5).

 Varma et al. [\[16\]](#page-9-7) defined 'ungrokking' as generalization loss when retraining a grokked network on a *smaller* dataset (D < Dcrit), attributing it to shifting circuit efficiencies under WD. In contrast, we observe late-stage generalization collapse ('anti-grokking') occurring on the *original* dataset after prolonged training (˜10<sup>7</sup> steps) *without* WD (WD=0). This distinct phenomenon is not predicted by [\[16\]](#page-9-7) as it falls outside of the crucial weight decay assumption on which it relies. Grokking studies now include real-world tasks [\[2,](#page-9-3) [4\]](#page-9-8). Golechha et al. [\[2\]](#page-9-3) introduced progress mea- sures (e.g., Activation Sparsity, Absolute Weight Entropy) and notably observed grokking without WD, resulting in increasing l 2 norms, similar to our setup. We use their metrics for comparison but extend training drastically (up to 10<sup>7</sup> steps), revealing the subsequent 'anti-grokking' collapse—a phenomenon not reported in their work despite the similar WD=0 regime. We employ the theory of Heavy-Tailed Self-Regularization (HTSR) [\[10,](#page-9-1) [11\]](#page-9-9), tracking the spectral exponent α. We find α predicts both the initial grokking and, uniquely, the subsequent dip and even- tual 'anti-grokking' collapse under WD=0. Our contribution lies in identifying and characterizing this anti-grokking phenomenon using α for long-term generalization stability, extending prior work

<sup>86</sup> that either required WD or did not explore sufficiently long training horizons.

## <sup>87</sup> 3 Measures and Metrics

#### <sup>88</sup> 3.1 Heavy–Tailed Self-Regularization (**HTSR**)

From weights to spectra. For each layer weight matrix W ∈ <sup>R</sup> <sup>89</sup> <sup>N</sup>×<sup>M</sup>, we build the un-centred <sup>90</sup> *correlation* (Gram) matrix

$$\mathbf{X} = \frac{1}{N} \mathbf{W}^T \mathbf{W} \in \mathbb{R}^{M \times M}. \quad (1)$$

Let {λi}<sup>M</sup> <sup>i</sup>=1 <sup>91</sup> be the eigenvalues of X. Their empirical spectral density (ESD) is the discrete measure

$$\rho_{emp}(\lambda) = \frac{1}{M} \sum_{i=1}^M \delta(\lambda - \lambda_i). \quad (2)$$

Gaussian baseline. If the entries of W are i.i.d. N (0, σ<sup>2</sup> <sup>92</sup> ), then, in the limit N →∞,M →∞ with <sup>93</sup> aspect ratio Q = N/M ≥ 1 fixed, ρemp(λ) converges to the Marchenko–Pastur (MP) density [\[7\]](#page-9-10)

$$\rho_{\text{MP}}(\lambda) = \begin{cases} \frac{Q}{2\pi\sigma^2} \frac{\sqrt{(\lambda^+ - \lambda)(\lambda - \lambda^-)}}{\lambda}, & \lambda \in [\lambda^-, \lambda^+], & \lambda^\pm = \sigma^2(1 \pm Q^{-1/2})^2 \\ 0, & \text{otherwise,} & \end{cases} \quad (3)$$

 This provides a principled "null model" against which real, trained weights can be compared. In a well-trained model, the eigenvalues of any layer W will rarely conform closely to an MP distribu- tion and will almost always have a significant number of large eigenvalues extending beyond any recognizable bulk MP region (λ ≫ λ <sup>+</sup> <sup>97</sup> ) if not being fully heavy-tailed power-law. If, however, we randomize W elementwise,

$$\mathbf{W} \rightarrow \mathbf{W}^{rand} \quad (4)$$

then the elements of Wrand will be i.i.d. by construction, and we expect that the ESD of Wrand <sup>99</sup> <sup>100</sup> can be very well fit to an MP distribution. This is shown below, on in Figure [2](#page-3-0) (Right).

 Heavy–Tailed Self-Regularization (**HTSR**) Theory Prior work[\[10,](#page-9-1) [11,](#page-9-9) [9\]](#page-9-4) shows that the ESD of real-world DNN layers with learned correlations almost never sits entirely within the Marchenko–Pastur bulk predicted for i.i.d. Gaussian weights; instead, the right edge flares into a power law (PL) tail . Formally,

$$\rho_{emp}(\lambda) \sim \lambda^{-\alpha}, \quad \lambda_{\min} < \lambda < \lambda_{\max}, \quad (5)$$

<sup>105</sup> with the exponent α quantifying the strength of the correlations. According to the HTSR framework <sup>106</sup> [\[11\]](#page-9-9), different ranges of α correspond to the different phases of training and different levels of <sup>107</sup> convergence for each layer:

 • α ≳ 5 − 6: Random-like or Bulk-plus-Spikes — the spectrum is close to the Gaussian baseline; little task structure is present. • 2 ≲ α ≲ 5 − 6: Weak (WHT) to Moderate Heavy (Fat) Tailed (MHT) — correlations build up; layers are well-conditioned and typically generalise better. • α = 2 Ideal value: Corresponds to fully optimized layers in models. Associated with layers in models that generalize best. • α < 2: Very-Heavy-Tailed (VHT) — extremely heavy tails indicate potentially over- fitting to the training data and often precede and/or are associated with decreases in the generalization / test accuracy.

 Note that the lower bound of α = 2 on the Fat-Tailed phase is a hard cutoff, whereas the upper bound α ≳ 5 − 6 is somewhat looser because it can depend on the aspect ratio Q. See Martin et. al. [\[10,](#page-9-1) [9\]](#page-9-4) for more details.

 Estimating α. Following [\[11\]](#page-9-9), we fit the tail of ρemp to a PL [5](#page-2-0) through the maximum likeli- hood estimator (MLE) [\[1\]](#page-9-11). The start of the Pl tail, λmin, is chosen automatically to minimize the Kolmogorov-Smirnov distance between the empirical and fitted distributions. All calculations are performed with WeightWatcher v0.7.5.5 [\[8\]](#page-9-12), which automates

- SVD extraction of singular values σ<sup>i</sup> (λ<sup>i</sup> = σ i ), • PL fits and goodness-of-fit KS tests (including selection of λmin and λmax) • Detection of correlation traps (optional)

 Figure [2](#page-3-0) (Left) shows an example of a PL fit on a log-log scale for a representative layer after training. The plot displays the ESD for a typical NN layer (a histogram or kernel density estimate of eigenvalues), the automatically chosen λmin (xmin, vertical line, red), the λmax (xmax, vertical line, orange), and the best fit for the PL tail (dashed line, red).

![](_page_3_Figure_5.jpeg)

Figure 2: Left: Example of the ESD derived from a well-correlated W (blue) and the Power-Law fit to the tail (red), on a Log-Log plot. Right: Example of the ESD of Wrand (light purple) and the MP fit (red), on a Log-Linear plot.

 Note that the PL fit is very sensitive to the choice of λmin, and a poor choice will result in a poorly estimated α. If λmin is too large (bad xmin, vertical line, purple), then the PL tail is too small and results in a larger α. The selection of λmin is very important in the calculation of the tail alpha (α) and is fully automated using the open-source WeightWatcher tool.[\[8\]](#page-9-12)

 Significance for Grokking/Anti-Grokking. Across all experiments, the trajectory α(t) proves to be a highly sensitive indicator of the network's generalization state: large drops toward α ≈ 2 coincide with the onset of grokking, while a further fall below α < 2 foreshadows (and then characterizes) the eventual "anti-grokking" collapse .

### 3.2 Correlation Traps

 To better understand the origin of anti-grokking (generalization collapse), it is instructive to look for evidence of potential overfitting in the layer weight matrices W, which appear as what we call Correlation Traps [\[9\]](#page-9-4). Recall that for a well-trained model, we expect the ESDs of Wrand <sup>142</sup> <sup>143</sup> to be well-fit by an MP distribution; here we argue that deviations from this are significant and <sup>144</sup> informative. To identify these deviations, we compare the randomized layer ESDs against the MP <sup>145</sup> distribution at the different stages of training to assess deviations from randomness. We identify these deviations as anomalously large eigenvalues in the underlying Wrand <sup>146</sup> . We call such large eigenvalues correlation traps, λtrap, when they are significantly larger than the bulk edge λ + rand <sup>147</sup> of <sup>148</sup> the best fit MP distribution.

$$\lambda_{trap} \gg \lambda_{rand}^+ \quad (6)$$

<sup>149</sup> See the Appendix [D](#page-12-0) for additional statistical validation of the presence of such traps, as well as the

<sup>150</sup> Supplementary Information. Also, see [\[9\]](#page-9-4) for more details. <sup>151</sup> The WeightWatcher tool [\[8\]](#page-9-12) detects correlation traps automatically; it randomizes W, then performs automated MP fits by estimating the variance σ 2 MP <sup>152</sup> of the underlying randomized matrix Wrand, finding the fit that best describes the bulk of its ESD of Wrand <sup>153</sup> . It then finds all eigenval-<sup>154</sup> ues λtrap that are significantly larger (i.e. beyond the Tracy-Widom fluctuations) of the MP bulk edge λ + rand of the ESD of <sup>W</sup>rand <sup>155</sup> . Figure [3](#page-4-0) depicts two layers from the models studied here with <sup>156</sup> correlation traps.

![](_page_4_Figure_5.jpeg)

Figure 3: Examples of Correlation Traps. ESDs of (Wrand) (light purple) of Layer 2 for the randomized weight matrix Wrand for different models, compared to an MP fit (red). Correlation traps λtrap are depicted a small spikes to the right of the MP fit. (x-axis is log scale) Left: Right Before Collapse (i.e. at more than ∼ 10<sup>6</sup> steps) (σmp ≈ 0.9879). The KS test (P-value ≈ 4×10−<sup>13</sup>) indicates a strong deviation from the MP model. A single, prominent correlation trap appears at λtrap ≈ 10<sup>6</sup>.<sup>5</sup> . Right: Final Generalization Collapse. The KS test (P-value ≈ 1.877 × 10−<sup>5</sup> ) indicates a strong deviation from the MP model. Multiple correlation traps are observed, λtrap ∈ [10<sup>2</sup>.x , 10<sup>6</sup>.<sup>5</sup> ].

<sup>157</sup> For additional statistical validation, here, we also use the Kolmogorov-Smirnov (KS) test to quantify the dissimilarity between the ESD of Wrand <sup>158</sup> and its best MP fit. A large difference, combined with <sup>159</sup> a visual inspection of the data, indicates the presence of one or more correlation traps (λtrap).

#### <sup>160</sup> 3.3 Other Benchmarked Metrics

We benchmarked our HTSR-based findings against l 2 weight norm analysis [\[6\]](#page-9-2) and several progress measures proposed by Golechha [\[2\]](#page-9-3), these include Activation Sparsity (As), Absolute Weight En- tropy (Habs(W)), and Approximate Local Circuit Complexity (ΛLC ). Detailed definitions of these measures are provided in Appendix [B.](#page-10-0)

![](_page_5_Figure_0.jpeg)

Figure 4: HTSR results vs. optimization steps. Top: Average α across layers. Middle: α for the first fully connected layer (FC1). Bottom: α for the second fully connected layer (FC2). Note the significant dip below the critical threshold α = 2, especially in FC2, coinciding with the "antigrokking" performance drop seen in Fig. [1](#page-1-0) after 1M steps.

### <sup>165</sup> 4 Results and Analysis

#### <sup>166</sup> 4.1 Layer Metrics for Tracking Grokking

<sup>167</sup> HTSR layer quality metric α: Our primary metric, the HTSR layer quality metric α, reveals <sup>168</sup> critical dynamics missed by other measures. Figure [4](#page-5-0) shows the evolution of α averaged across <sup>169</sup> layers (top) and for individual fully connected layers (middle, bottom).

Table 1: Layer-wise and average **HTSR** α exponents. At the right edge of each grokking phase: Pre-grokking ∼ 10<sup>5</sup> steps, Grokking 10<sup>6</sup> steps, and Anti-grokking 10<sup>7</sup> steps, For the zero-weightdecay (W D = 0) experiment; values are taken from Fig. [4.](#page-5-0) Various seeds are used and variability in initialization, optimizer trajectory may occur.

|         | Layer, Metric |   |   |     |    | Grokking |   | (Max | Test Acc.) |   |   |     | (Collapse) |
|---------|---------------|---|---|-----|----|----------|---|------|------------|---|---|-----|------------|
| FC1     | α             | 4 | 0 | ± 1 | 3  | 3        | 2 | ± 0  | 6          | 1 | 0 | ± 0 | 40         |
| FC2     | α             | 4 | 6 | ± 0 | 5  | 2        | 4 | ± 0  | 1          | 1 | 4 | ± 0 | 24         |
| average | α             | 4 | 3 | ± 0 | 70 | 2        | 8 | ± 0  | 30         | 1 | 2 | ± 0 | 23         |

 Initially, α is high, reflecting random-like weights. As training progresses and the network begins to fit the training data, α decreases. The sharp drop towards the optimal (fat-tailed) regime (2 ≲ α ≲ 5−6) coincides with the rapid improvement in test accuracy characteristic of grokking (around 10<sup>4</sup> -10<sup>5</sup> steps in Figure [1\)](#page-1-0). Crucially, as training continues into the millions of steps, α consistently dips below 2, entering the Very Heavy-Tailed (VHT) regime. This occurs notably in the second fully connected layer (FC2, bottom panel). This drop below α = 2, indicating potential layer non- optimality and overly strong correlations, directly precedes and coincides with the significant drop in test accuracy—the "anti-grokking" phase—observed after 10<sup>6</sup> steps in Figure [1.](#page-1-0)

 Together, these observations highlight the unique sensitivity of the HTSR α metric. This metric not only identifies the grokking transition but also provides an early warning for the subsequent insta- bility and the novel "anti-grokking" phenomenon, highlighting potentially pathological correlation structures forming deep into training. The layer-wise analysis (Figure [4\)](#page-5-0) further suggests that this instability might originate in specific layers (i.e. FC2 here) becoming over correlated (α < 2).

 Comparative metrics: In contrast, the comparative metrics capture the initial training and grokking phases but fail to predict the late-stage generalization collapse. Figure [5](#page-6-0) displays the Acti- vation Sparsity, Absolute Weight Entropy, and Approximate Local Circuit Complexity. While these metrics show clear trends during the initial learning and grokking phases (e.g., changes in sparsity and complexity), their trajectories become relatively stable or lack distinct features corresponding to the dramatic performance drop seen during "anti-grokking". For example, circuit complexity remains relatively flat in the late stages up until some noise at the end, offering no warning of the impending collapse. Though Activation Sparsity shows an inflection around peak test accuracy and does detect grokking, it generally continues its upward trend through the late-stage collapse.

![](_page_6_Figure_2.jpeg)

Figure 5: Alternative progress measures (Golechha [\[2\]](#page-9-3)) vs. optimization steps. Top: Activation Sparsity. Middle: Absolute Weight Entropy. Bottom: Approximate Local Circuit Complexity. While these metrics show changes during the initial training and grokking phases (Activation Sparsity for example), they do not exhibit clear signals predicting the magnitude of the late-stage "antigrokking" performance dip observed after 10<sup>6</sup> steps.

 In our primary WD=0 experiments, A<sup>s</sup> generally increases throughout training (Figure [5\)](#page-6-0), seemingly tracking the pre-grokking and grokking phases, however, it fails the negative control in the anti- grokking phase because it continues to increase in the same way as in pre-grokking. Prior studies have linked activation sparsity to generalization [\[5,](#page-9-13) [12,](#page-9-5) [14\]](#page-9-14) and reported specific dynamics such as plateauing before grokking [\[2\]](#page-9-3) or an increase preceding a rise in test loss [\[3\]](#page-9-15). Specifically, we observe a subtle inflection or dip in A<sup>s</sup> coinciding with the point of maximum test accuracy before a slight increase. While this feature appears to mark a shift around peak test accuracy, its specific predictive utility for subsequent generalization dynamics is questionable. In other words, without knowing the proper sparsity cutoff, it is impossible to determine if increasing A<sup>s</sup> corresponds to pre-grokking or anti-grokking. In contrast, because the HTSR α = 2 is a theoretically established universal cutoff, one can distinguish between the two phases correctly.

 Additionally, in our WD=0.01 control experiment, as detailed in Appendix [C,](#page-11-0) a similar inflection in A<sup>s</sup> occurs where test accuracy, after a slight initial decrease from its peak, subsequently plateaus rather than undergoing a catastrophic collapse as seen in the WD=0 case. Therefore, observing this dip in A<sup>s</sup> alone does not allow one to distinguish whether test accuracy will catastrophically decline or stabilize, suggesting it primarily indicates that some form of transitional change has oc- curred around the point of maximum generalization, rather than predicting the specific nature of the subsequent trajectory. Our findings indicate limitations in the other two comparitive metrics for tracking the anti-grokking phase. Absolute Weight Entropy (Habs(W)), despite its suggested link to generalization [\[2\]](#page-9-3), also decreases sharply during the collapse, thus not reliably distinguishing this anti-grokking phase. Similarly, ΛLC [\[2\]](#page-9-3) remains low throughout the collapse, failing to reflect the performance degradation. We also confirm, consistent with [\[2\]](#page-9-3), that grokking occurs robustly even with increasing weight norms and no weight decay.

#### 4.2 Correlation Traps and Anti-Grokking

 To better understand the origin of anti-grokking (generalization collapse), it is instructive to look for evidence of potential overfitting in the layer weight matrices W, in the form correlation traps. As described in Section [3.1,](#page-2-1) we analyze the eigenvalues {λi} of the randomized weight matrices Wrand derived from each layer's weight matrix W for layers FC1 and FC2.

Table 2: Average number of detected correlation traps in layers FC1 and FC2 at the right edge of of the three grokking phases: Pre-grokking ∼ 10<sup>5</sup> steps, Grokking 10<sup>6</sup> steps, and Anti-grokking 10<sup>7</sup> steps. Results shown for both experiments, with (W D > 0) and without W D = 0 weight decay.

|     | Model, | Layer | Pre-grokking | Grokking (Max Test Acc.) | Anti-grokking |    |     | (Collapse) |
|-----|--------|-------|--------------|--------------------------|---------------|----|-----|------------|
| W D | = 0    | , FC1 | 0            | 0                        | 6             | 33 | ± 5 | 44         |
| W D | = 0    | , FC2 | 0            | 0                        | 1             | 00 | ± 0 | 00         |
| W D | > 0    | , FC1 | 0            | 0                        | 2             | 00 | ± 0 | 00         |
| W D | > 0    | , FC2 | 0            | 0                        | 1             | 00 | ± 0 | 00         |

 As show in Table [2,](#page-7-0) for both layers, FC1 and FC2, and for both experiments, with and without weight decay, neither layer shows evidence of correlation traps until the anti-grokking phase. The presence of such traps corresponds to HTSR α < 2 for these layers, as predicted by previous work[\[9\]](#page-9-4). Further statistical analysis for the FC2 layers is provided in Appendix [D.](#page-12-0) The presence of correlation traps, combined with α < 2, is a definitive signal indicating the model is in the anti-grokking phase.

### 5 Conclusion

 This study investigated the well-known grokking phenomena in neural networks (NN) under the lens of the recently developed theory of Heavy-Tailed Self Regularization (HTSR) [\[10\]](#page-9-1). Previous work has attempted to explain grokking (using the l norm), but only succeeds in the presence of weight decay (WD), and has been unable to explain grokking without weight decay[\[6,](#page-9-2) [2\]](#page-9-3). For this reason, we have studied the long-term generalization dynamics of the grokking phenomena both with weight decay (W D > 0) and without (W D = 0). We compare the application of the HTSR theory to the l norm and several previous proposed metrics.[\[6\]](#page-9-2) Our primary finding is that the HTSR layer quality metric α can effectively track grokking both with and without weight decay. In particular, the HTSR α tracks the initial grokking transition and subsequent performance dips in both case (W D = 0 , W D > 0) and, in doing so, offers new insights into the grokking phenomena.

 Moreover, and critically, in the W D = 0 setting, the HTSR α also provides an early indication of a novel late-stage generalization collapse, called anti-grokking. This collapse is characterized by a significant drop in test accuracy despite sustained perfect training accuracy (and a large l norm), and is observed after extensive training (up to 10<sup>7</sup> steps).

We also examined several other grokking progress measures, in addition to the l norm [\[6\]](#page-9-2), includ- ing Activation Sparsity As, Absolute Weight Entropy Habs(W), and Approximate Local Circuit Complexity ΛLC [\[2\]](#page-9-3). Although A<sup>s</sup> and ΛLC captured initial training and grokking phases, and do  change at the anti-grokking transition, they failed to unambiguously predict the appearance and/or presence of anti-grokking.

 In examining the HTSR results on all 3 phases of grokking, we propose a new explanation of the grokking phenomena. During the first phase, pre-grokking, where only training accuracy saturates, only a subset of the individual layers will converge, and only far enough (i.e, α ≈ 4) to describe the training data, while other layers will appear almost random (i.e, α ≈ 5). Importantly, some layers will be more important for generalization than others, and these will not have converged very well at all. During the grokking phase, when the test accuracy is maximal, all important layers will converge extremely well, with α metrics approach the optimal value with α ≈ 2.0–exactly as predicted by the HTSR theory. In the third anti-grokking phase, where the test accuracy drops substantially, one or more layer will overfit the data in some yet undetermined way). They will have α < 2, and may exhibit correlation traps (and/or even rank collapse). (Note these results are also supported by recent theoretical developments in HTSR (and SETOL) theory[\[9\]](#page-9-4).)

 In particular, we consider the implications of observing numerous correlation traps in the anti- grokking phase. The 'traps' are anomalous rank-one (or greater) perturbations in the weight matrix W, causing a large mean-shift in underlying distribution of elements: <sup>E</sup>[Wij ] → *large* and, 'push- ing' the ESD into the VHT phase where α < 2. The large shift in <sup>E</sup>[Wij ] → *large* indicates that the distribution of weights is *atypical*. That is, different random samples of the weights could have very different means. And as with any statistical estimator, an atypical distribution will not generalize well. (Similar results have been seen in training a similar model with very large learning rates[\[9\]](#page-9-4).) Consequently, it is hypothesized that layers with large numbers of correlation traps are overfit to the training data (in some unspecified way), and hurt the overall model test accuracy.

 These results underscore the utility of HTSR for monitoring and understanding long-term gener- alization stability across different regularization schemes, with a particular strength in identifying potential catastrophic collapse. The observed layer-specific changes in α during the W D = 0 col- lapse suggest that potential over-fitting may develop deep into training. While our current findings are based on a specific MLP architecture and MNIST subset, further research should validate these observations across diverse datasets, architectures, hyperparameter configurations, and optimizers. Promising future work includes developing α-guided adaptive training strategies. Additionally, de- signing differentiable regularizers or loss terms based on α could potentially enable faster and more stable generalization, for instance, by encouraging convergence towards α ≈ 2.

#### 6 Limitations

 Our study, while providing insights into generalization dynamics via Heavy-Tailed Self- Regularization (HTSR), has limitations that define important avenues for future research. The empir- ical findings are primarily derived from a specific three-layer MLP architecture trained on an MNIST subset. Consequently, the generalizability of the observed α trajectories and their specific predictive power for phenomena like grokking and late-stage generalization collapse warrants further valida- tion across a wider range of model architectures (e.g., CNNs, Transformers), datasets, tasks, and diverse training configurations, including different optimizers and hyperparameter settings.

 Furthermore, HTSR is an empirically-grounded, phenomenological framework, supported theoret- ically with a novel application of Random Matrix Theory (RMT). While its correlations between the heavy-tailed PL exponent α and network generalization states are compelling, the interpretation requires careful consideration of context. For instance, while well-generalized models often exhibit α values within the range (e.g., 2 ≤ α ≤ 6), and α ≈ 2 is frequently associated with optimal per- formance or critical transitions, this is not a strictly bidirectional implication. It is conceivable that layers or models might exhibit α values near or even below 2 (typically indicating over-correlation) yet display suboptimal generalization. Other very-well trained models may have layers fairly large alphas. This is not yet fully understood. This highlights that while α provides strong correlational insights into learning phases and stability, the precise mapping of specific α values to absolute per- formance levels can be context-dependent and is an area for ongoing refinement of the theory (see [\[9\]](#page-9-4)). Our work contributes observations within specific phenomena, acknowledging that the broader applicability and predictive nuances of the HTSR theory will benefit from continued exploration.

 These limitations underscore the importance of ongoing empirical and theoretical work to further refine, validate, and extend the understanding of HTSR theory in deep learning.

# References


[1] Aaron Clauset, Cosma Rohilla Shalizi, and Mark E.J. Newman. Power-law distributions in empirical data. *SIAM Review*, 51(4):661–703, 2009. [2] Satvik Golechha. Progress measures for grokking on real-world tasks, 2024. [3] Karim Huesmann, Luis Garcia Rodriguez, Lars Linsen, and Benjamin Risse. The impact of ac- tivation sparsity on overfitting in convolutional neural networks. In *Pattern Recognition. ICPR International Workshops and Challenges: Virtual Event, January 10–15, 2021, Proceedings, Part III*, volume 12663 of *Lecture Notes in Computer Science*, pages 130–145. Springer Inter- national Publishing, 2021. [4] Ahmed Imtiaz Humayun, Randall Balestriero, and Richard Baraniuk. Deep networks always grok and here is why, 2024. [5] Zonglin Li, Chong You, Srinadh Bhojanapalli, Daliang Li, Ankit Singh Rawat, Sashank J. Reddi, Ke Ye, Felix Chern, Felix Yu, Ruiqi Guo, and Sanjiv Kumar. The lazy neuron phe- nomenon: On emergence of activation sparsity in transformers. In *The Eleventh International Conference on Learning Representations (ICLR)*, 2023. arXiv:2210.06313. [6] Ziming Liu, Ouail Kitouni, Niklas S. Nolte, Eric J. Michaud, Max Tegmark, and Mike Williams. Towards understanding grokking: An effective theory of representation learning. In Surbhi Koyejo, Sham Kakade (formerly Mohamed), Aarti Agarwal, Danielle Belgrave, Kyunghyun Cho, and Alice Oh, editors, *Advances in Neural Information Processing Systems*, volume 35, pages 34651–34663. Curran Associates, Inc., 2022. [7] Vladimir A. Marchenko and Leonid Andreevich Pastur. Distribution of eigenvalues for some sets of random matrices. *Matematicheskii Sbornik*, 72(114)(4):507–536, 1967. [8] Charles H. Martin. WeightWatcher: Analyze Deep Learning Models without Training or Data. <https://github.com/CalculatedContent/WeightWatcher>, 2018-2024. Ver- sion 0.7.5.5 used in this study. Accessed May 12, 2025. [9] Charles H. Martin, Christopher Hinrichs, and Michael W. Mahoney. SETOL: A Semi- Empirical Theory of (Deep) Learning. [https://github.com/CalculatedContent/](https://github.com/CalculatedContent/setol_paper/blob/main/setol_draft.pdf) [setol\\_paper/blob/main/setol\\_draft.pdf](https://github.com/CalculatedContent/setol_paper/blob/main/setol_draft.pdf), 2025. Preprint. [10] Charles H. Martin and Michael W. Mahoney. Implicit self-regularization in deep neural net- works: Evidence from heavy-tailed spectral analysis, 2021. [11] Charles H. Martin and Michael W. Mahoney. Predicting trends in the quality of state-of-the-art deep learning models, 2021. [12] William Merrill, Nikolaos Tsilivis, and Aman Shukla. A tale of two circuits: grokking as competition of sparse and dense subnetworks, 2023. [13] Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability, 2023. [14] Ze Peng, Lei Qi, Yinghuan Shi, and Yang Gao. Theoretical explanation of activation sparsity through flat minima and adversarial robustness, 2023. [15] Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra. Grokking: Generalization beyond overfitting on small algorithmic datasets, 2022. [16] Vikrant Varma, Rohin Shah, Zachary Kenton, Janos Kram ´ ar, and Ramana Kumar. Explaining ´ grokking through circuit efficiency, 2023.
# <sup>339</sup> Appendices

# <sup>340</sup> A Experimental Setup

 We train a Multi-Layer Perceptron (MLP) on a subset of the MNIST dataset using the hyperparame- ters detailed in Table [3.](#page-10-1) The training subset is constructed by randomly selecting 100 samples from each of the 10 MNIST classes, ensuring a balanced dataset of 1,000 unique training points. This was run on an Nvidia Quadro P2000 and took approximately 11 hours. A considerable part of the time is due to the speed of saving the measures.

Table 3: Experimental hyperparameters used in the study (details in Appendix [A\)](#page-10-2).

| Parameter           | Value                                                                                  |
|---------------------|----------------------------------------------------------------------------------------|
| Network             | Architecture Fully Connected MLP                                                       |
| Depth               | 3 Linear layers (Input → Hidden1 → Hidden2 → Output)                                   |
| Width               | 200 hidden units per hidden layer                                                      |
| Activation          | Function ReLU (Rectified Linear Unit)                                                  |
| Input Layer         | Size 784 (Flattened MNIST image 28 × 28 )                                              |
| Output Layer        | Size 10 (MNIST digits 0-9)                                                             |
| Weight              | Initialization Default PyTorch (Kaiming Uniform for weights), parameters scaled by 8.0 |
| Bias Initialization | Default PyTorch (Uniform), then scaled by 8.0                                          |
| Dataset             | MNIST                                                                                  |
| Training Points     | 1,000 (100 per class, stratified random sampling)                                      |
| Test Points         | Standard MNIST test set (10,000 samples)                                               |
| Batch Size          | 200                                                                                    |
| Loss Function       | Mean Squared Error (MSE) with one-hot encoded targets                                  |
| Optimizer           | AdamW                                                                                  |
| Learning Rate       | (LR) 5 × 10 − 4                                                                        |
| Weight Decay        | (WD) 0.0 (for main results), 0.01 (for Appendix C comparison)                          |
| AdamW β 1           | 0.9 (PyTorch default)                                                                  |
| AdamW β 2           | 0.999 (PyTorch default)                                                                |
| AdamW ϵ             | 10 − 8                                                                                 |
|                     | (PyTorch default)                                                                      |
| Optimization        | Steps 10 7                                                                             |
| Data Type           | (PyTorch) ‘torch.float64‘                                                              |
| Random Seed         | 0 (for all libraries)                                                                  |
| Software            | Framework PyTorch                                                                      |
| HTSR Tool           | WeightWatcher v0.7.5.5 [8]                                                             |

 Note on Weight Decay: The primary results presented in this paper, particularly those demonstrat- ing grokking followed by late-stage generalization collapse (Figure [1\)](#page-1-0), were obtained with weight decay explicitly set to 0. This allows observation of the learning dynamics driven purely by the op- timizer and the loss landscape while exhibiting both phenomena, whereas the other proposed mea- sures fail to detect the grokking transition of increasing test accuracy. Runs with non-zero weight decay (e.g., WD=0.01, see Appendix [C\)](#page-11-0) were also performed for comparison, showing different dynamics but confirming the general utility of HTSR.

# <sup>353</sup> B Comparative Grokking Progress Metrics and Measures

<sup>354</sup> Weight Norm Analysis Following observations that weight decay can influence grokking [\[6\]](#page-9-2), we monitor the l 2 <sup>355</sup> norm of the network's weights,

$$\|\mathbf{w}\|_2 = \sqrt{\sum_l \|\mathbf{w}_l\|_F^2}, \quad (7)$$

<sup>356</sup> throughout training. We specifically run experiments with weight decay disabled (WD=0) to isolate <sup>357</sup> the effect of the optimization dynamics on the norm itself.

<sup>358</sup> Activation Sparsity. For a given layer with activations bi,j (representing the activation of neuron <sup>359</sup> j for input example i), the activation sparsity A<sup>s</sup> is defined as:

$$A_s = \frac{1}{T} \sum_{i=1}^T \frac{1}{n} \sum_{j=1}^n \mathbf{1}(b_{i,j} < \tau), \quad (8)$$

 where T is the number of training examples, n is the number of neurons in the layer, τ is a chosen threshold, and 1(·) is the indicator function. This metric measures neuron inactivity. Prior studies have linked activation sparsity to generalization [\[5,](#page-9-13) [12,](#page-9-5) [14\]](#page-9-14) and reported specific dynamics such as plateauing before grokking [\[2\]](#page-9-3) or an increase preceding a rise in test loss [\[3\]](#page-9-15).

Absolute Weight Entropy. For a weight matrix W ∈ R <sup>m</sup>×<sup>n</sup> <sup>364</sup> , the absolute weight entropy <sup>365</sup> Habs(W) is given by:

$$H_{abs}(W) = - \sum_{i=1}^m \sum_{j=1}^n |w_{i,j}| \log |w_{i,j}|. \quad (9)$$

<sup>366</sup> This entropy quantifies the spread of absolute weight magnitudes. Golechha et al. [\[2\]](#page-9-3) suggested its <sup>367</sup> sharp decrease signals generalization.

Approximate Local Circuit Complexity. Let L (W) <sup>368</sup> (x) denote the output logits for input x using weights W, and let L (W′ <sup>369</sup> (x) denote the logits when 10% of the weights are set to zero (forming W′ <sup>370</sup> ). The approximate local circuit complexity, denoted ΛLC , is the summed KL divergence:

$$\Lambda_{LC} = \sum_{k=1}^{N_{data}} \sum_{j \in \mathcal{C}} \Pr(j | L^{(W)}(x_k)) \log \frac{\Pr(j | L^{(W)}(x_k))}{\Pr(j | L^{(W')}(x_k))}. \quad (10)$$

 Here, Ndata is the number of training examples xk, C is the set of classes, and Pr(j|L(x)) is the probability of class j derived from the logits L(x) (e.g., via softmax). This measure captures out- put sensitivity to minor weight perturbations. Lower ΛLC has been linked to stable, generalizable representations [\[2\]](#page-9-3).

# <sup>375</sup> C Experiment with Weight Decay

 To further understand the influence of weight decay on the observed generalization dynamics and the behavior of our tracked metrics, we conducted an experiment identical to our main study (WD=0) but with a small amount of weight decay (WD=0.01) applied. The training curves and metric evolutions for this WD=0.01 experiment are presented in Figures [6,](#page-12-1) and [7.](#page-12-2)

A key characteristic of training with weight decay is the tendency for the l 2 <sup>380</sup> norm of the weights to <sup>381</sup> decrease over time, or stabilize at a lower value, which is observed in this experiment (Figure [7\)](#page-12-2). This contrasts with the continuously increasing l 2 <sup>382</sup> weight norm seen in our primary WD=0 experi-<sup>383</sup> ments.

 In this WD=0.01 regime, the network still achieves a high level of test accuracy. Notably, after the initial grokking phase, the test accuracy slightly decreases and then enters a prolonged plateau, maintaining near peak performance for a significant number of optimization steps (Figure [6\)](#page-12-1). Cor- respondingly, the average heavy-tail exponent, α, also exhibits the decrease and a distinct plateau around the critical value of α ≈ 2 during this period (Figure [6,](#page-12-1) top left panel).

 The other progress measures considered—Activation Sparsity and Approximate Local Circuit Com- plexity—also tend to plateau or stabilize during this phase of peak test performance in the WD=0.01 setting (Figure [7\)](#page-12-2). This contrasts with the WD=0 scenario where, despite eventual grokking, the system does not find such a stable long-term plateau and instead proceeds towards a late-stage gen- eralization collapse. The observation that α (and other metrics) plateau in conjunction with peak, stable test accuracy under traditional weight decay settings aligns with some existing understanding of well-regularized training.

<sup>396</sup> While HTSR and the α exponent provide valuable insights in both regimes, its unique capability <sup>397</sup> to signal impending collapse in the absence of weight decay underscores its importance for under-<sup>398</sup> standing layer dynamics under various scenarios.

![](_page_12_Figure_0.jpeg)

Figure 6: HTSR α exponent evolution for the MLP trained with WD=0.01.

![](_page_12_Figure_2.jpeg)

Figure 7: Progress measures (Activation Sparsity, Weight Entropy, Circuit Complexity) and l Weight Norm for the MLP trained with WD=0.01.

# D Statistical Analysis and Validation of Correlation Traps

 Here, to further validate the presence of correlation traps for the zero weight decay W D = 0 experiment , we report the results of statistical tests designed to determine if the randomized ESD of the Wrand fits an MP distribution or not. Briefly we fit the ESD to a MP distribution and report the fitted variance σmp, the Kolmogorov-Smirnov (KS) statistic of the fit, and the p-value for the MP fit as the null model. We also report the number of correlation traps, as determined using the open-source WeightWatcher tool[\[8\]](#page-9-12). Results for layer FC1 are presented in Table [4.](#page-13-0) Results for FC2 are similar (not shown). Additional details are provided in the supplementary material.

Table 4: Statistical validation of correlation traps. Selected results for layer FC1 at different training stages for zero weight decay (W D = 0) experiment. MP Variance (σMP ) Kolmogorov-Smirnov (KS) test statistic, p-value for MP fit, and number of detected correlation traps. Pregrokking ∼ 10<sup>5</sup> steps, Grokking 10<sup>6</sup> steps, and Anti-grokking 10<sup>7</sup> steps,

| Model State                  | MP variance ( σ mp ) | KS Statistic | p-value        | # Traps |
|------------------------------|----------------------|--------------|----------------|---------|
| Pre-Grokking                 | ≈ 1 002              | 0.0120       | ≈ 1 0          | 0       |
| Grokking (Max Test Accuracy) | ≈ 0 999              | 0.0212       | ≈ 1 0          | 0       |
| Anti-Grokking (Collapse)     | ≈ 0 949              | 0.3044       | 1 877 × 10 − 5 | 9       |

 Initial Layer State (Pre-Grokking WD=0): Immediately after initialization, the network weights are expected to be largely random, and their ESD should conform well to the MP distribution. Figure [2](#page-3-0) (Right) shows an MP fit to an ESD from a representative layer Wrand of the newly initialized model. A KS test comparing this empirical ESD to the fitted MP distribution (using σmp ≈ 1.0024 as estimated by WeightWatcher) yielded a KS statistic of 0.0120 and a p-value ≈ 1.0. This high p-value indicates this ESD is statistically consistent with the MP distribution, as expected.

 Best Layer State (Grokking phase WD=0): As the network learns and reaches its maximum test accuracy, significant structure develops in the elements of the weight matrices Wi,j . This can be seen by randomizing the layer weight matrix elementwise, W → Wrand , and plotting ESD, and looking for deviations from the theoretical MP distribution. The ESD now typically exhibits a pronounced heavy tail, with eigenvalues extending beyond the bulk region that might be approximated by an MP fit. For our model at peak test accuracy, the KS test against a fitted MP model (σmp ≈ 0.999) resulted in a KS statistic of 0.0212 and a p-value ≈ 1. Again, this is an MP distribution.

 Final Layer State (Anti-Grokking phase WD=0): In the late-stage of training, as the model undergoes generalization collapse and enters an over-correlated state (characterized by α < 2), the ESD of Wrand structure continues to reflect a non-random configuration. The KS test for the final model against an MP fit (with an estimated σmp ≈ 2) yielded a KS statistic of 0.3044 and a p-value of 1.877 × 10−<sup>5</sup> Figure [3](#page-4-0) (Right) . This result further confirms that the network's structure remains significantly different from a random matrix baseline, consistent with the highly correlated or near rank-collapsed state indicated by our HTSR analysis.

 These quantitative comparisons demonstrate a transition from an initially random-like state (consis- tent with MPD) to progressively more structured, non-random states as learning occurs and eventu- ally leads to over-correlation. The inability of the MP distribution to describe these learned features, especially the heavy tails, necessitates the use of tools like the HTSR theory, the PL exponent α, and the open-source WeightWatcher tool, to properly characterize these complex correlation structures and their relationship to generalization performance.

# NeurIPS Paper Checklist

 The checklist is designed to encourage best practices for responsible machine learning research, addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not re- move the checklist: The papers not including the checklist will be desk rejected. The checklist should follow the references and follow the (optional) supplemental material. The checklist does NOT count towards the page limit.

 Please read the checklist guidelines carefully for information on how to answer these questions. For each question in the checklist:

 • You should answer [Yes] , [No] , or [NA] . • [NA] means either that the question is Not Applicable for that particular paper or the relevant information is Not Available. • Please provide a short (1–2 sentence) justification right after your answer (even for NA).

 The checklist answers are an integral part of your paper submission. They are visible to the reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it (after eventual revisions) with the final version of your paper, and its final version will be published with the paper.

 The reviewers of your paper will be asked to use the checklist as one of the factors in their evalu- ation. While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a proper justification is given (e.g., "error bars are not reported because it would be too computationally expensive" or "we were unable to find the license for the dataset we used"). In general, answering "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we acknowledge that the true answer is often more nuanced, so please just use your best judgment and write a justification to elaborate. All supporting evidence can appear either in the main paper or the supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification please point to the section(s) where related material for the question can be found.

IMPORTANT, please:

 • Delete this instruction block, but keep the section heading "NeurIPS Paper Check- list", • Keep the checklist subsection headings, questions/answers and guidelines below. • Do not modify the questions and only use the provided macros for your answers.

#### 1. Claims

 Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

 Justification: The abstract claims that the heavy-tailed exponent alpha from HTSR theory reliably predicts grokking, anti-grokking (a late-stage generalization collapse), and pro- vides an early warning for this collapse, especially in no-decay regimes where other mea- sures may not. It also mentions the identification of "correlation traps." The introduction reiterates these points. Section 1 "Our Contributions" and Section 4 "Results and Anal- ysis" (particularly subsections 4.1 and 4.2, and Table 2) provide experimental results and discussion supporting these claims, such as alpha dropping below 2 before collapse and the appearance of correlation traps.

Guidelines:

 • The answer NA means that the abstract and introduction do not include the claims made in the paper. • The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers. • The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

 • It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

#### 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

#### Answer: [Yes]

 Justification: Section 7, titled "Limitations," explicitly discusses the limitations. These include the specificity of the MLP architecture and MNIST dataset used, calling for valida- tion across diverse models and data. It also mentions that the interpretation of alpha can be context-dependent, and is not a bidirectional relationship

Guidelines:

 • The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper. • The authors are encouraged to create a separate "Limitations" section in their paper. • The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The au- thors should reflect on how these assumptions might be violated in practice and what the implications would be. • The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated. • The authors should reflect on the factors that influence the performance of the ap- proach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon. • The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size. • If applicable, the authors should discuss possible limitations of their approach to ad- dress problems of privacy and fairness. • While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an impor- tant role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

#### 3. Theory assumptions and proofs

 Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

#### Answer: [Yes]

 Justification: Any necessary proofs along with assumptions will be provided in the suppli-mental material.

Guidelines:

 • The answer NA means that the paper does not include theoretical results. • All the theorems, formulas, and proofs in the paper should be numbered and cross- referenced. • All assumptions should be clearly stated or referenced in the statement of any theo- rems. • The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition. • Inversely, any informal proof provided in the core of the paper should be comple-mented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

#### 4. Experimental result reproducibility

 Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclu-sions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

 Justification: Appendix A ("Experimental Setup") and Table A.1 provide a comprehensive list of experimental settings: network architecture, depth, width, activation, input/output sizes, weight/bias initialization (including scaling), dataset (MNIST), training points (1000, (100 per class), stratified random sampling), test points (standard MNIST test set), batch size, loss function (MSE), optimizer (AdamW), learning rate, weight decay, AdamW betas and epsilon, optimization steps, data type, random seed, and software framework. Also mentioned is the Weightwatcher version which is an open source package.

Guidelines:

 • The answer NA means that the paper does not include experiments. • If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not. • If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable. • Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed. • While NeurIPS does not require releasing code, the conference does require all sub- missions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to re- produce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case au- thors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

#### 5. Open access to data and code

 Question: Does the paper provide open access to the data and code, with sufficient instruc- tions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: Provided with supplementary in accordance with guidlines.

 • Please see the NeurIPS code and data submission guidelines ([https://nips.cc/](https://nips.cc/public/guides/CodeSubmissionPolicy) [public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details. • While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark). • The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ([https:](https://nips.cc/public/guides/CodeSubmissionPolicy) [//nips.cc/public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details. • The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc. • The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why. • At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable). • Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

#### 6. Experimental setting/details

 Question: Does the paper specify all the training and test details (e.g., data splits, hyper- parameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

 Justification: Appendix A ("Experimental Setup") and Table A.1 provide a comprehensive list of experimental settings: network architecture, depth, width, activation, input/output sizes, weight/bias initialization (including scaling), dataset (MNIST), training points (1000, 100 per class, stratified random sampling), test points (standard MNIST test set), batch size, loss function (MSE), optimizer (AdamW), learning rate, weight decay, AdamW betas and epsilon, optimization steps, data type, random seed, and software framework.

Guidelines:

 • The answer NA means that the paper does not include experiments. • The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them. • The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

 Question: Does the paper report error bars suitably and correctly defined or other appropri-ate information about the statistical significance of the experiments?

#### Answer: [Yes]

 Justification: Table ("Layer-wise and average HTSR alpha exponents") and Table ("Aver- age number of detected correlation traps") report values with mean and standard deviation, likely over runs/seeds, though the exact source of this variability (e.g., multiple runs vs. variability across layers/checkpoints) is explicitly detailed for these tables. KS test p-values are reported in Section 3.2, Appendix B (Table A.2), and Figure 2 caption when discussing MP fits and correlation traps, which is a measure of statistical significance for those spe- cific tests. The experiments take considerably long time to run (each experiment takes 11 hours) so n is limited

Guidelines:

 • The answer NA means that the paper does not include experiments. • The authors should answer "Yes" if the results are accompanied by error bars, confi- dence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

 • The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions). • The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors). • It should be clear whether the error bar is the standard deviation or the standard error of the mean. • It is OK to report 1-sigma error bars, but one should state it. The authors should prefer- ably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified. • For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates). • If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

#### 8. Experiments compute resources

 Question: For each experiment, does the paper provide sufficient information on the com- puter resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

 Justification: Appendix A ("Experimental Setup") states: "This was run on an Nvidia Quadro P2000 and took approximately 11 hours." This provides the type of GPU and the approximate execution time for the main experiment (10<sup>7</sup> steps). The GPU used has 5 GB memory.

Guidelines:

 • The answer NA means that the paper does not include experiments. • The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage. • The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute. • The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

#### 9. Code of ethics

 Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?

Answer: [Yes]

 Justification: The research uses standard open datasets and open methodologies, with no ethical red flags based on our reasearch.

Guidelines:

 • The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. • If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics. • The authors should make sure to preserve anonymity (e.g., if there is a special consid-eration due to laws or regulations in their jurisdiction).

#### 10. Broader impacts

 Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Guidelines:

 • The answer NA means that there is no societal impact of the work performed. • If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact. • Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact spe- cific groups), privacy considerations, and security considerations. • The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster. • The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology. • If there are negative societal impacts, the authors could also discuss possible mitiga- tion strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

#### 11. Safeguards

 Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: Same as above not applicable

Guidelines:

 • The answer NA means that the paper poses no such risks. • Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by re- quiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters. • Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images. • We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

#### 12. Licenses for existing assets

 Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

 Justification: Weightwatcher and the experimental code is Apache 2.0 , MNIST is Creative Commons Attribution-Share Alike 3.0. Pytorch is BSD 3-Clause License.

Guidelines:

 • The answer NA means that the paper does not use existing assets. • The authors should cite the original paper that produced the code package or dataset. • The authors should state which version of the asset is used and, if possible, include a URL. • The name of the license (e.g., CC-BY 4.0) should be included for each asset.

 • For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided. • If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, [paperswithcode.com/](paperswithcode.com/datasets) [datasets](paperswithcode.com/datasets) has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset. • For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided. • If this information is not available online, the authors are encouraged to reach out to the asset's creators.

#### 13. New assets

 Question: Are new assets introduced in the paper well documented and is the documenta-tion provided alongside the assets?

Answer: [Yes]

Justification: Code is provided with comments

Guidelines:

 • The answer NA means that the paper does not release new assets. • Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc. • The paper should discuss whether and how consent was obtained from people whose asset is used. • At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

#### 14. Crowdsourcing and research with human subjects

 Question: For crowdsourcing experiments and research with human subjects, does the pa- per include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: Not applicable.

Guidelines:

 • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contri- bution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, cura- tion, or other labor should be paid at least the minimum wage in the country of the data collector.

#### 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

 Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification:Not Applicable.

Guidelines:

 • Depending on the country in which research is conducted, IRB approval (or equiva- lent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

#### 16. Declaration of LLM usage

 Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

Justification: Not Applicable

Guidelines:

 • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy ([https://neurips.cc/Conferences/](https://neurips.cc/Conferences/2025/LLM) [2025/LLM](https://neurips.cc/Conferences/2025/LLM)) for what should or should not be described.