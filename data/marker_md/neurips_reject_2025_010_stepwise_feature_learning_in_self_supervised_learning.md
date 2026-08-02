# Stepwise Feature Learning in Self-Supervised Learning

Anonymous Author(s) Affiliation Address email

# Abstract

 Recent advances in self-supervised learning (SSL) have shown remarkable progress in representation learning. However, SSL models often exhibit shortcut learning phenomenon, where they exploit dataset-specific biases rather than learning gen- eralizable features, sometimes leading to severe over-optimization on particular datasets. We present a theoretical framework that analyzes this shortcut learning phenomenon through the lens of *extent bias* and *amplitude bias*. By investigating the relations among extent bias, amplitude bias, and learning priorities in SSL, we demonstrate that learning dynamics is fundamentally governed by the dimen- sional properties and amplitude of features rather than their semantic importance. Our analysis reveals how the eigenvalues of the feature cross-correlation matrix influence which features are learned earlier, providing insights into why models preferentially learn shortcut features over more generalizable features.

# 1 Introduction

 While deep neural networks have shown remarkable success in various learning tasks, recent studies have revealed a concerning trend: models often exploit unexpected learning behavior, particularly shortcut learning, which tends to take easier but potentially less reliable paths to solve general tasks [13]. For example, in image classification tasks, models tend to learn earlier larger background features than smaller foreground objects [17], potentially leading them to classify cows based on whether they appear on grass rather than learning actual cow features, or identify camels primarily by detecting desert backgrounds [5]. This phenomenon is prevalent even in SSL [11, 22, 29, 10].

 While previous research has shown that neural networks are vulnerable to spurious correlations in data [1], several other contributing factors to shortcut learning have been identified. Hermann et al. [17] find shortcuts emerging from color, size, and background. Rahaman et al. [25], Tancik et al. [27] find spectral bias that low-frequency features are learned faster than high-frequency features. While significant progress has been achieved, current theoretical frameworks provide insufficient explanations for why models consistently induce shortcuts.

 Recent studies have demonstrated that SSL models with small weight initialization exhibit stepwise learning dynamics, where features are learned sequentially based on the corresponding eigenvalues of the feature cross-correlation matrix [26]. Building on this insight, we analyze the eigenvalue and eigenvector structure of the feature cross-correlation matrix. This approach provides a novel theoretical framework for understanding why certain features, regardless of their semantic importance, are consistently learned earlier in the training process. Our investigation focuses particularly on how dimensional properties influence learning priority, potentially explaining some observed shortcut learning phenomena beyond traditional spurious correlations.

 • We establish theoretical connections between shortcut learning phenomenon, stepwise learning, and eigenvalue-eigenvector of feature cross-correlation matrix on SSL. • We extend theoretical research on shortcut learning from supervised learning to SSL. • We characterize *extent bias*, a tendency to prioritize features based on their dimensional extent or spatial coverage rather than their semantic importance. • We analyze how amplitude and frequency determine which features are learned earlier in SSL, and characterize *amplitude bias*, a tendency to prioritize features based on their amplitude rather than their semantic importance.

## 2 Related Works

 Self-supervised learning SimCLR [7] established a foundational contrastive learning framework but required large batch sizes to generate sufficient negative pairs for preventing representational collapse. This limitation prompted research into non-contrastive approaches, leading to innovations like SimSiam [8] and BYOL [14]. Further research introduced methods focusing on different training objectives: VICReg [4] introduced variance-invariance-covariance regularization, while Barlow Twins [31] employed cross-correlation matrix to prevent collapse. DINO [6] advanced the field by introducing self-distillation with no labels. The success of DINO v2 [23] sparked interest in Joint Embedding Predictive Architectures (JEPA) [2], with recent work by Littwin et al. [20] revealing JEPA's tendency to prioritize learning "related" features over "frequently" occurring ones.

 Learning dynamics Following the introduction of Neural Tangent Kernel (NTK) [18], researchers have discovered important connections between eigenvalue dynamics and learning behavior, including spectral bias phenomena [27, 15]. This theoretical framework has enabled deeper analysis of loss function trajectories and saddle point behaviors [19, 24]. Notably, Simon et al. [26] demonstrated that these saddle-to-saddle dynamics appear not only in supervised learning but also extend to SSL settings.

 Shortcut learning Shortcut learning was first identified in Geirhos et al. [13], describing how neural networks take easier but incorrect paths to solve tasks. This phenomenon appears in various ways: Geirhos et al. [12], Baker et al. [3], Hermann and Lampinen [16] showed that CNNs rely on object texture rather than object shape, Wu et al. [30] demonstrated that even a single pixel can mislead model's decisions, and Hermann et al. [17] revealed that CNNs preferentially learn salient but potentially irrelevant features like scale and background elements. These shortcuts can arise from dataset properties, particularly through spurious correlations [1] and implicit biases. Our work specifically examines how dataset correlations contribute to shortcut learning.

# 3 Background (Stepwise Nature of SSL [26])

 In this section, following Simon et al. [26], we analyze the stepwise learning dynamics of SSL systems through the lens of toy Barlow Twins models [31]. We first introduce the loss function and gradient flow dynamics, then derive the connection between cross-correlation matrix and feature learning. Finally, we examine how the eigendecomposition of feature cross-correlation matrix connects to the theoretical foundation for our analysis of extent bias, amplitude bias.

Given training data {x (i) ∈ <sup>R</sup> <sup>m</sup> : i = 1, 2, · · · , n}, the training loss of toy Barlow twins is defined as L = ||C −Id||<sup>2</sup> F , C ≡ n P<sup>n</sup> i=1(W x(i) )(W x′(i) ) <sup>⊤</sup> + (W x′(i) )(W x(i) ) <sup>⊤</sup> , where ||·||<sup>F</sup> is Frobenius norm, W ∈ R <sup>d</sup>×<sup>m</sup> is learnable parameters, and C ∈ <sup>R</sup> d×d is cross-correlation matrix of W x and W x′ for another view x ′ from x. Using the feature cross-correlation matrix

$$\Gamma \equiv \frac{1}{2n} \sum_{i=1}^n (x^{(i)} x'^{(i)\top} + x'^{(i)} x^{(i)\top}) \in \mathbb{R}^{m \times m}, \quad (1)$$

we have L = ||WΓW<sup>⊤</sup> − Id||<sup>2</sup> F and C = WΓW<sup>⊤</sup> . The eigendecomposition of the feature cross-correlation matrix is Γ = VΓΛΓV <sup>⊤</sup> <sup>Γ</sup> with Λ<sup>Γ</sup> <sup>=</sup> diag(γ1, · · · , γm) and <sup>V</sup><sup>Γ</sup> = [v<sup>1</sup> · · · <sup>v</sup>m] ∈ R <sup>m</sup>×<sup>m</sup> ,where γ<sup>1</sup> ≥ γ<sup>2</sup> ≥ · · · ≥ γ<sup>m</sup> are eigenvalues of Γ and vi's are the corresponding eigenvectors for γi's.

<sup>82</sup> Using (3), we can express the gradient flow as follows:

$$\frac{dW}{dt} = -\nabla_W \mathcal{L} = -4(W\Gamma W^\top - I_d)W\Gamma. \quad (2)$$

<sup>83</sup> To analyze eigenvector dynamics of weights, we assume weight initialization is aligned.

<sup>84</sup> Assumption 3.1 (Aligned Initialization Simon et al. [26]). At the initialization, we assume that the <sup>85</sup> right-singular vectors of W(0) are aligned with the top d eigenvectors of Γ, i.e., the singular value decomposition is W(0) = US0V (≤d)⊤ Γ for a orthogonal matrix U ∈ R d×d <sup>86</sup> , the top-d eigenvector matrix V (≤d) <sup>Γ</sup> = [v<sup>1</sup> · · · vd] ∈ <sup>R</sup> m×d <sup>87</sup> , and a diagonal matrix S<sup>0</sup> = diag(s1(0), · · · , sd(0)) with a <sup>88</sup> small initialization s<sup>j</sup> (0) > 0.

<sup>89</sup> Under Assumption 3.1, the solution W(t) for the gradient flow (2) can be expressed as follows [26, Proposition 4.1]: W(t) = US(t)V (≤d)⊤ Γ <sup>90</sup> for S(t) = diag(s1(t), · · · , sd(t)), where the singular <sup>91</sup> values of W(t) evolve as

$$s_j(t) = \frac{e^{4\gamma_j t}}{\sqrt{s_j^{-2}(0) + (e^{8\gamma_j t} - 1)\gamma_j}}$$

which has a limit of γ −1/2 j <sup>92</sup> as t → ∞ and nearly sigmoidal

$$s_j^2(t) \approx \frac{1}{\gamma_j + s_j^{-2}(0)e^{-8\gamma_j t}} =: \tilde{s}_j^2(t). \quad (3)$$

Solving s˜ 2 j (t) = <sup>1</sup> 2 s 2 j <sup>93</sup> (∞) at its critical time t = τ<sup>j</sup> , we have

$$\tau_j = -\frac{\log(s_j^2(0)\gamma_j)}{8\gamma_j} \quad (4)$$

around which s<sup>j</sup> (t) (or s˜<sup>j</sup> (t)) passes <sup>1</sup> 2 γ −1/2 j <sup>94</sup> and rapidly increases from near zero to near the saturation γ −1/2 j <sup>95</sup> .

<sup>96</sup> In this paper, we focus on the property that the eigenvector feature v<sup>j</sup> corresponding to a larger γ<sup>j</sup> <sup>97</sup> leads to an earlier critical point τ<sup>j</sup> from (4).

# <sup>98</sup> 4 Extent bias

 In computer vision tasks, backgrounds typically span larger regions while foreground objects occupy more concentrated areas. Recent work by Hermann et al. [17] reveals that CNNs preferentially learn these background features over object-specific details, creating a specific form of spurious correlation between backgrounds and class labels. For example, cows are often classified based on grass backgrounds rather than their distinctive features, and camels are identified through desert scenes [5]. This phenomenon points to a underlying learning mechanism we term *extent bias*, a fundamental tendency of neural networks to prioritize features based on their dimensional extent or spatial coverage rather than their semantic importance. The connection between extent bias and learning dynamics implies the need for understanding a more fundamental mechanism beyond traditional spurious correlations. While spurious correlations emerge from dataset-specific relationships, the bias toward learning background features is inherent in the learning dynamics of neural networks themselves. Through our analysis of SSL systems, we demonstrate that this bias for background features emerges naturally from how models learn earlier features with higher extent bias, independent of their semantic relevance or predictive power.

<sup>113</sup> In this section, we investigate how different feature properties influence learning priorities in SSL. <sup>114</sup> Through extent bias analysis, we demonstrate how features with larger dimensional coverage are <sup>115</sup> learned before those with smaller coverage, regardless of their semantic importance.

 We construct a theoretical framework that identifies dimensional effects in feature learning. By analyzing how SSL models process features of varying extent bias, we can directly observe how extent bias influences learning priority and connects to the background-foreground learning dynamics observed in practice.

![](_page_3_Figure_0.jpeg)

Figure 1: Effects of extent bias on learning dynamics in SSL. (Left) Stepwise learning curves of Barlow Twins. There are two (d = 2) learning steps shown with two black dashed vertical lines (also shown in the other two panels) which indicate the time steps t<sup>1</sup> and t<sup>2</sup> with t<sup>1</sup> : t<sup>2</sup> ≈ 1 γl : γ<sup>s</sup> = 1 m<sup>l</sup> : m<sup>s</sup> . The predicted loss (dashed green) of L = P<sup>d</sup> j=1(λ˜ <sup>j</sup> (t) − 1)<sup>2</sup> = P<sup>d</sup> <sup>j</sup>=1(˜s 2 j (t)γ<sup>j</sup> − 1)<sup>2</sup> using (3) match the empirical result (solid green). (Center) Evolution of eigenvalues λ<sup>j</sup> 's of C during training. At the beginning, the first eigenvalue λ<sup>1</sup> (blue) increases to 1 and then later the second λ<sup>2</sup> (red) follows. We also compare them with the predicted evolution λ˜ <sup>j</sup> (t) (dashed lines). (Right) Evolution of the feature alignment ||W e||<sup>2</sup> for e = e<sup>l</sup> (blue) and e = e<sup>s</sup> (red). It shows very similar behaviors with the eigenvalues λ˜ 1/2 j (dashed lines). See Theorem 4.5. We use m<sup>l</sup> = 9, m<sup>s</sup> = 1. See Appendix A.1 for more detailed settings.

#### <sup>120</sup> 4.1 Settings

We first consider the following base input xbase = [bl1 ⊤ m<sup>l</sup> , bs1 ⊤ m<sup>s</sup> <sup>⊤</sup> ∈ <sup>R</sup> <sup>m</sup>, where b<sup>l</sup> , b<sup>s</sup> i.i.d. <sup>121</sup> ∼ B(p = <sup>122</sup> 0.5) follow the Bernoulli distribution and take the value ±1 with the equal probability, m<sup>l</sup> and m<sup>s</sup> <sup>123</sup> indicate the size of larger part and smaller part, respectively, i.e., m<sup>l</sup> > m<sup>s</sup> and m<sup>l</sup> + m<sup>s</sup> = m, and <sup>124</sup> 1<sup>k</sup> is the k-dimensional all-one vector. From now on, we will use the subscript l and s for the indices <sup>125</sup> with respect to the *larger*-part and *smaller*-part features, respectively.

Then, to obtain the positive pair (x, x′ <sup>126</sup> ), we introduce the following data augmentation x = xbase + ε and x ′ = xbase + ε ′ , with the noise ε, ε′ i.i.d. ∼ N (0m, a<sup>2</sup> <sup>127</sup> Im) for some a > 0.

## <sup>128</sup> 4.2 Learning Dynamics on extent bias

<sup>129</sup> In this subsection, we discuss the relationship between γ<sup>j</sup> and L, focusing on which features are <sup>130</sup> learned earlier. From Section 4.1, we can simplify the feature cross-correlation matrix Γ by analyzing <sup>131</sup> the expected values of the augmented features. Based on the definition in (1), we have:

$$\Gamma = \frac{1}{2n} \sum_{i=1}^n (x^{(i)} x'^{(i)\top} + x'^{(i)} x^{(i)\top}) = \mathbb{E}[x_{\text{base}} x_{\text{base}}^\top]. \quad (5)$$

<sup>132</sup> To identify which features drive the loss as stepwise phenomena, we consider basis vectors that <sup>133</sup> disentangle individual features. Specifically, we define basis vectors e<sup>l</sup> and e<sup>s</sup> where each vector has <sup>134</sup> ones only in the dimensions corresponding to its respective feature:

$$e_l = [\mathbf{1}_{m_l}^\top, \mathbf{0}_{m_s}^\top]^\top, e_s = [\mathbf{0}_{m_l}^\top, \mathbf{1}_{m_s}^\top]^\top \in \mathbb{R}^m.$$

$$\text{FA}(e) = \|We\|_2 \text{ for } e = e_l, e_s. \quad (6)$$

<sup>135</sup> By measuring the feature alignment between these basis vectors and the weight matrix through <sup>136</sup> FA(e) = ∥W e∥2, we can identify which features are being learned at each stage of the training <sup>137</sup> process.

<sup>138</sup> The eigendecomposition of Γ is given by the following proposition:

<sup>139</sup> Theorem 4.1. *For the correlation matrix in (5), we have the eigenvalue matrix* Λ<sup>Γ</sup> *and eigenvector* <sup>140</sup> *matrix* VΓ*:*

$$\Lambda_\Gamma = \text{diag} \left( [m_l, m_s, \mathbf{0}_{m-2}] \right), V_\Gamma^{(\leq 2)} = [e_l/\sqrt{m_l} \ e_s/\sqrt{m_s}] .$$

<sup>141</sup> We defer the proof to Appendix B.1.

<sup>142</sup> We hypothesize that features with larger dimensions are learned faster, regardless of their predictive <sup>143</sup> power or potential to cause shortcuts. This is particularly relevant in vision tasks where such features  might correspond to larger pixel regions. We experiment using a simple toy model to validate our theoretical analysis of dimensional influence on feature learning. In our experimental setup, we used two distinct features with different dimensional coverage (m<sup>l</sup> = 9 and m<sup>s</sup> = 1), allowing us to clearly observe the learning dynamics. As shown in Figure 1, the results demonstrate three key phenomena: Figure 1 (Left) shows loss trajectory (green line) exhibits two distinct stepwise phenomena, marked by black vertical lines. These stepwise decreases precisely align with the abrupt increase in the eigenvalue observed in Figure 1 (Center), confirming our theoretical prediction that eigenvalue dynamics drives the learning process. Figure 1 (Center) shows a clear stepwise pattern in which two distinct eigenvalues of Γ increase sequentially. This sequential increase directly corresponds to the learning priority of feature, with the higher-dimensional feature (m<sup>l</sup> = 9) being learned first. Figure 1 (Right) shows that, feature alignment measurements ||W e||<sup>2</sup> from (6) provide direct evidence of the learning order: the alignment with e<sup>1</sup> (blue line, corresponding to the larger feature dimension) increases during the first loss decrease, while e<sup>2</sup> alignment (red line) follows during the second phase. This learning pattern strongly supports our hypothesis that dimensional coverage determines how early the features learned. This result suggests that the spatial extent of features, rather than their semantic content, plays a crucial role in determining learning priority.

## 4.3 Cross-Correlation eigenvalue λ and Loss Relationship

 In this subsection, we analyze the relationship between the eigenvalues λ<sup>j</sup> of cross-correlation matrix C.

 Theorem 4.2. *Under Assumption 3.1, the eigenvalues* λ<sup>j</sup> *of feature cross-correlation matrix* C = WΓW⊤*, using the approximation* s<sup>j</sup> ≈ s˜<sup>j</sup> *in (3), are approximated as* λ<sup>j</sup> = s j γ<sup>j</sup> ≈ s˜ j γ<sup>j</sup> =: λ˜ <sup>j</sup>

*which have*

$$\tilde{\lambda}_j(\tau_j) = \frac{1}{2} \text{ and } \tilde{\lambda}'_i(\tau_j) \begin{cases} = 2\gamma_j & \text{if } i = j, \\ \approx 0 & \text{if } i \neq j \end{cases} \quad (7)$$

*at* τ<sup>j</sup> = −log(s j (0)γ<sup>j</sup> )/8γ<sup>j</sup> *in (4). For the Barlow Twins loss* L = ∥C − Id∥ F *, we have* L = P<sup>d</sup> <sup>j</sup>=1(λ<sup>j</sup> − 1)<sup>2</sup> *and* − dL dt (τ<sup>j</sup> ) <sup>≈</sup> <sup>λ</sup>˜′ j (τ<sup>j</sup> ) = 2γ<sup>j</sup> .

We defer the proof to Appendix B.3.

 Figure 6 in Appendix C shows the relationship between cross-correlation eigenvalue λ differentiated with respect to t and loss derivatives <sup>d</sup><sup>L</sup> dt . The close alignment between the loss derivative and λ derivative curves demonstrates that the decrease in loss is directly driven by λ, with larger m<sup>l</sup> features learned, and smaller m<sup>s</sup> features learned later. The curves' relative magnitudes show an approximate m<sup>l</sup> : m<sup>s</sup> ratio, which matches our theoretical predictions.

# 4.4 Weight Singular Value Evolution

To verify the dynamics of weight singular values s<sup>j</sup> , we propose the following theorem:

Theorem 4.3. *Using the approximation (3), the singular values of the weight matrix* W *satisfy*

$$\tilde{s}_j(\tau_j) = 1/\sqrt{2\gamma_j} \text{ and } \tilde{s}'_j(\tau_j) = \sqrt{2\gamma_j}$$

*at the critical point* t = τ<sup>j</sup> *.*

We defer the proof to Appendix B.4.

 Figure 7 in Appendix C shows two key aspects of singular value dynamics during training. First, the singular values s<sup>j</sup> evolve to their theoretical limits 1/ √<sup>γ</sup><sup>j</sup> and <sup>1</sup>/ √ <sup>γ</sup>s, as predicted by our analysis. Second, the derivatives of these singular values exhibit peaks at their respective critical points, with magnitudes that follow the predicted √ γ<sup>l</sup> : √ 2γ<sup>s</sup> ratio. These results provide strong empirical validation of our theoretical framework, demonstrating that both the convergence values and learning priority on different features are governed by their corresponding eigenvalues in the feature cross-correlation matrix Γ.

## 4.5 Aligned Initialization and Subspace Alignment

 To justify our alignment initialization assumption in Assumption 3.1, we first define the following subspace alignment metric:

 Definition 4.4 (Subspace Alignment). We define subspace alignment of two subspaces Im(A) and Im(B):

$$\text{SA}(A, B) = \|A^\top B\|_F^2/d,$$

where Im(A) = {Av ∈ <sup>R</sup> <sup>m</sup> : v ∈ <sup>R</sup> <sup>d</sup>}, A = [a<sup>1</sup> · · · ad], B = [b<sup>1</sup> · · · bd] ∈ <sup>R</sup> m×d and a<sup>i</sup> , b<sup>i</sup> ∈ <sup>R</sup> <sup>m</sup> are unit vectors.

 Note that 0 ≤ SA(A, B) ≤ 1 and it attains SA(A, B) = 0 when Im(A) ⊥ Im(B), and SA(A, B) = 1 when Im(A) = Im(B). Figure 10 (Top) in Appendix D empirically validates Assumption 3.1 using the subspace alignment metric. The model becomes aligned rapidly in the early stages of training, satisfying the assumption.

# 4.6 Orthogonal Feature Learning

 Our analysis shows that features are learned as orthogonal to each other, where each feature is acquired independently without interference from others. This orthogonal learning pattern is particularly evident in the evolution of the model's weight matrix singular vectors. To formalize this observation, we analyze how the left singular vectors of the weight matrix align with the feature vectors during training.

Theorem 4.5. *Under Assumption 3.1, the left singular vectors* u *of* W(t) *learn features orthogonally:*

$$Proj_{U(\leq 2)}(We_l) := (u_l^\top We_l, u_s^\top We_l) = (\sqrt{\lambda_l}, 0),$$

$$Proj_{U(\leq 2)}(We_s) := (u_l^\top We_s, u_s^\top We_s) = (0, \sqrt{\lambda_s}),$$

*where* u<sup>l</sup> , u<sup>s</sup> *are the corresponding left singular vectors for the singular values* s<sup>l</sup> , ss*.*

 Figure 11 shows orthogonal learning pattern that features are learned independently and sequentially, supporting our theoretical analysis of stepwise learning dynamics.

# 4.7 Non-linear multi layer network

 Nonlinearity exhibits distinct learning dynamics compared to linearity. Therefore, we aim to investi- gate whether extent biass also exists in multilayer perceptrons (MLPs). We experiment with a 3-layer network, using leakyReLU as the activation function, for understanding non-linear feature learning dynamics. Our non-linear network experiments demonstrate that extent bias persists beyond linear models. As shown in Figure 14 in Appendix G, the non-linear network exhibits remarkably similar stepwise learning patterns to those observed in linear models Figure 1. Key similarities include: simi- lar eigenvalue evolution patterns, consistent stepwise loss reduction phases. These results suggest that extent bias is a fundamental learning phenomenon that transcends network architecture complexity, rather than being merely an artifact of linear models.

# 4.8 Practical Study on Colored-MNIST Dataset

 We conducted experiments using a Colored-MNIST dataset, where we adjusted the ratio of digits pixels relative to the total image pixels. We tested three different ratios: 0.05, 0.10, and 0.15. In this dataset, we set the correlation between background and label to 70% for both training and test sets, making it difficult for a model that predicts solely based on background to achieve accuracy higher than 70%. According to our hypothesis, since backgrounds have larger extent bias than objects, the test set accuracy would rapidly increase from an initial 10% (random choosing) to 70% (as the model learns background features), then plateau for a period, before slowly rising to 100% (as it learns object features). We also hypothesized that this plateau period would decrease as the ratio of label pixels increases in the images, with shorter plateaus observed in the 0.15 ratio condition compared to 0.05.

 Figure 2 supports our hypothesis. Across all pixel ratio conditions (0.05, 0.10, 0.15), test accuracy exhibited a consistent pattern: a rapid increase from initial 10% to 70%, followed by a plateau period,

![](_page_6_Figure_0.jpeg)

Figure 2: Extent bias effects on spurious datasets. ResNet18 on the Colored MNIST dataset. (Left) Loss decreases even though the error rate doesn't decrease. (Right) The error rate has a plateau at 70%, which corresponds to the correlation between background and object. The lengths of the plateaus become shorter as the object's pixel ratio increases. See Appendix A.2 for more detailed settings.

 and then a gradual ascent to 100%. Notably, as the object pixel ratio increased, the duration of the plateau phase decreased. The loss function continued to decrease even when accuracy remained stagnant at 70%. This suggests a extent bias where larger objects are prioritized during the learning process. The pattern reflects how the model initially achieves 70% accuracy by relying on background features, which statistically occupy larger regions, before progressively learning object features. Furthermore, this indicates that larger extents occupy greater eigenvalues, implying a reduction in the critical point τ<sup>j</sup> .

# <sup>240</sup> 5 Amplitude Bias

 In regression tasks, the phenomenon of spectral bias has been observed, wherein low-frequency components are learned more rapidly than high-frequency components during the training process. Conversely, in classification tasks, a phenomenon known as frequency shortcut [28] has been observed, wherein the model preferentially learns the distinctive Fourier components of the input during the training process. While these studies have primarily focused on supervised learning, we extend this investigation to the SSL, seeking to understand whether similar learning dynamics persist within SSL frameworks.

## <sup>248</sup> 5.1 Settings

<sup>249</sup> To analyze how frequency and amplitude bias affect learning dynamics, we consider input data xbase ∈ <sup>R</sup> <sup>m</sup> <sup>250</sup> composed of two sinusoidal components with different frequencies:

$$x_{\text{base}}[t] = c_h b_h \sin(f_h t) + c_l b_l \sin(f_l t), \quad (8)$$

where f<sup>h</sup> = 2π <sup>m</sup> k and f<sup>l</sup> = 2π <sup>m</sup> k ′ represent different frequencies for some integers k and k ′ , bh, b<sup>l</sup> i.i.d. <sup>251</sup> ∼ B(p = 0.5) follow the Bernoulli distribution and take the value ±1. Suppose f<sup>h</sup> < f<sup>l</sup> <sup>252</sup> to examine <sup>253</sup> the learning dynamics between low and high frequency components. The coefficients c<sup>h</sup> and c<sup>l</sup> <sup>254</sup> control the amplitude of each sinusoidal component, allowing us to investigate how magnitudes affect learning earlier. The Bernoulli variables b<sup>h</sup> and b<sup>l</sup> <sup>255</sup> introduce phase reversal in the signal. The time <sup>256</sup> vector t spans the input dimension m. We use the same augmentation with (4.1) to generate positive pairs (x, x′ <sup>257</sup> ) by adding Gaussian noise.

# <sup>258</sup> 5.2 Learning Dynamics on Amplitude Bias

Similar to Section 4.2, we consider basis vectors e<sup>h</sup> and e<sup>l</sup> <sup>259</sup> that isolate individual features: e<sup>h</sup> = <sup>260</sup> c<sup>h</sup> sin(fht) and e<sup>l</sup> = c<sup>l</sup> sin(flt), where 0 ≤ t ≤ m. Note that these two are orthogonal since f<sup>h</sup> = 2π <sup>m</sup> k and f<sup>l</sup> = 2π <sup>m</sup> k ′ with k ̸= k ′ <sup>261</sup> . Similar to Theorem 4.1, the cross-correlation matrix Γ for the <sup>262</sup> data generated from (8) can be expressed as follows:

<sup>263</sup> Theorem 5.1. *Under (8), the correlation matrix* Γ *has*

$$\Lambda_\Gamma = \text{diag} \left( [c_h^2 m/2, c_l^2 m/2, \mathbf{0}_{m-2}] \right), V_\Gamma^{(\leq 2)} = [e_h \ e_l].$$

![](_page_7_Figure_0.jpeg)

Figure 3: Amplitude bias effects on learning dynamics in SSL. See the caption of Figure 1. Note that the time steps t<sup>1</sup> and t<sup>2</sup> with t<sup>1</sup> : t<sup>2</sup> ≈ γ<sup>h</sup> : γl = 1 c h : c . We use c<sup>h</sup> = 1, c<sup>l</sup> = 1/2. See Appendix A.3 for more detailed settings.

<sup>264</sup> We defer the proof to Appendix B.2.

From (9), we observe that eigenvalues are proportional to the squares of the coefficients c 2 h and c 2 <sup>265</sup> . This implies that the learning dynamics are more strongly influenced by the amplitude rather than the underlying frequency. To validate our theoretical analysis of amplitude bias effect on learning dynamics, we conduct experiments using input data defined in (8). Especially, we set c<sup>h</sup> > c<sup>l</sup> <sup>269</sup> . This configuration shown in Figure 4 in Appendix A, allows us to examine how high-amplitude c<sup>h</sup> sin(fht) and low-amplitude c<sup>l</sup> sin(flt) affects feature amplitude bias. More details about the experiment are in Appendix A.3. Our analysis reveals two dominant eigenvalues. The large eigenvalue corresponds to the high- amplitude feature, and small eigenvalue corresponds to the low-amplitude component. The eigen- vectors of Γ are shown in Figure 5 , Appendix A. The first eigenvector, which corresponds to the largest eigenvalue, captures the dominant high-amplitude oscillation. The second eigenvector, which matches next-largest eigenvalue, captures the low-amplitude oscillation. Other eigenvectors are noise, corresponding to eigenvalues that are almost 0.

# <sup>278</sup> 5.3 Cross-Correlation eigenvalue λ and Loss Relationship

<sup>279</sup> We analyze how the eigenvalues λ relate to the loss dynamics. The relationship follows similar

<sup>280</sup> patterns to those observed in Section 4.3, but with coefficients c<sup>h</sup> and c<sup>l</sup> rather than m<sup>l</sup> and ms. <sup>281</sup> Figure 8 in Appendix C shows the close relationship between the derivatives of cross-correlation eigenvalues dλ<sup>h</sup> dt , dλ<sup>l</sup> dt and <sup>d</sup><sup>L</sup> dt <sup>282</sup> . The peaks in these derivatives occur at the critical points with magnitudes proportional to the corresponding coefficients γ<sup>h</sup> : γ<sup>l</sup> = c 2 h : c 2 l <sup>283</sup> (see (9)). This shows our <sup>284</sup> theoretical predictions Theorem 4.2 matches empirical result.

# <sup>285</sup> 5.4 Weight Singular Value Evolution

<sup>286</sup> We now analyze how the singular values of the weight matrix evolve during training. Similarly to the <sup>287</sup> extent bias case, we expect the singular values s<sup>j</sup> to converge to theoretical limits determined by the

<sup>288</sup> feature coefficients. <sup>289</sup> Figure 9 in Appendix C shows the evolution of singular values s<sup>h</sup> and s<sup>l</sup> of weight matrix W (Left) and their derivatives (Right). The singular values converge to their theoretical limits 1/ √ <sup>290</sup> <sup>γ</sup><sup>j</sup> predicted by Theorem 4.3, where γ<sup>j</sup> = c 2 j m 2 <sup>291</sup> . At the critical points τ<sup>j</sup> , the derivatives achieve their maximum values of p <sup>292</sup> 2γ<sup>j</sup> , showing that rates of feature learning are proportional to the coefficients. These <sup>293</sup> results confirm that the feature coefficients, rather than their frequencies, govern both the convergence <sup>294</sup> values and rates of feature learning.

# <sup>295</sup> 5.5 Aligned Initialization and Subspace Alignment

 To validate Assumption 3.1 about alignment between the weight matrix singular vectors and eigen- vectors of Γ, we measure the subspace alignment metric as defined in the extent case Definition 4.4. Figure 10 (Bottom) in Appendix D empirically validates our assumption through subspace alignment measurements. As discussed in Section 4.5, the model achieves alignment rapidly in the early stages of training, even with small random initializations.

# 5.6 Orthogonal Feature Learning

 Similar to the extent case, we investigate how the weight matrix learns different frequency components orthogonally as shown in Theorem 4.5. The orthogonal learning pattern reveals how frequency features are acquired independently despite their different spectral characteristics.

 Figure 12 in Appendix E shows the trajectories of weight matrix in terms of their alignments with frequency components e<sup>h</sup> and e<sup>l</sup> . The blue trajectory shows the first learning phase where u<sup>1</sup> aligns with the high-amplitude feature (c<sup>h</sup> sin(fht)), followed by the red trajectory showing u<sup>2</sup> aligning with the low-amplitude feature (c<sup>l</sup> sin(flt)). This sequential, orthogonal learning pattern demonstrates that feature learning is primarily determined by coefficient magnitudes rather than frequency characteristics, supporting our analysis in Theorem 4.5.

# 5.7 Non-linear multi layer network

Same as Section 4.7 in Appendix G, we conduct experiments with a 3-layer network using leakyReLU

 activations to analyze how amplitude coefficients affect learning dynamics in non-linear settings. Figure 15 in Appendix G demonstrates amplitude bias effects in non-linear networks is similar to linear networks on Figure 3. These results confirm that amplitude bias persists in non-linear architectures, suggesting amplitude magnitude remains a primary determinant of feature learning priority regardless of network complexity.

# 5.8 Discussion

 Figure 13 in Appendix F shows that a learning process is driven primarily by feature coefficient magnitude rather than frequency characteristics. The key observation is that the first learned features are those with large coefficients, independent of their spectral properties. This finding parallels frequency shortcut [28] in classification tasks, but reveals a different underlying mechanism. While frequency shortcut suggests models preferentially learn distinctive Fourier components, our results demonstrate that amplitude magnitude—not frequency characteristics—primarily determines feature learning priority.

# 6 Conclusion

 In this work, we establish a theoretical connection between eigendecomposition of the feature cross- correlation matrix, shortcut learning, and stepwise learning behavior in SSL. We provide insights into how dimensional feature properties influence the learning process in SSL frameworks. This work not only explains observed shortcut learning phenomena but also offers a theoretical lens for understanding and potentially mitigating such learning biases. This theoretical framework lays the groundwork for developing more robust SSL algorithms. Future work should focus on leveraging these insights to design mechanisms that encourage learning of generalizable features despite their potentially lower extent bias or amplitude bias.

# References


[1] M. Arjovsky, L. Bottou, I. Gulrajani, and D. Lopez-Paz. Invariant risk minimization. *arXiv preprint arXiv:1907.02893*, 2019. [2] M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, and N. Bal- las. Self-supervised learning from images with a joint-embedding predictive architecture. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 15619–15629, 2023. [3] N. Baker, H. Lu, G. Erlikhman, and P. J. Kellman. Deep convolutional networks do not classify based on global object shape. *PLoS computational biology*, 14(12):e1006613, 2018. [4] A. Bardes, J. Ponce, and Y. LeCun. Vicreg: Variance-invariance-covariance regularization for self-supervised learning. *arXiv preprint arXiv:2105.04906*, 2021.

[5] S. Beery, G. Van Horn, and P. Perona. Recognition in terra incognita. In *Proceedings of the European Conference on Computer Vision (ECCV)*, September 2018. [6] M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, and A. Joulin. Emerging properties in self-supervised vision transformers. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 9650–9660, 2021. [7] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton. A simple framework for contrastive learning of visual representations. In *International conference on machine learning*, pages 1597–1607. PMLR, 2020. [8] X. Chen and K. He. Exploring simple siamese representation learning. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 15750–15758, 2021. [9] L. Deng. The mnist database of handwritten digit images for machine learning research [best of the web]. *IEEE signal processing magazine*, 29(6):141–142, 2012. [10] C. Doersch, A. Gupta, and A. A. Efros. Unsupervised visual representation learning by context prediction. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, December 2015. [11] C. Doersch, A. Gupta, and A. A. Efros. Unsupervised visual representation learning by context prediction. In *Proceedings of the IEEE international conference on computer vision*, pages 1422–1430, 2015. [12] R. Geirhos, P. Rubisch, C. Michaelis, M. Bethge, F. A. Wichmann, and W. Brendel. Imagenet- trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness. *arXiv preprint arXiv:1811.12231*, 2018. [13] R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, and F. A. Wichmann. Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2(11):665–673, 2020. [14] J.-B. Grill, F. Strub, F. Altché, C. Tallec, P. Richemond, E. Buchatskaya, C. Doersch, B. Avila Pires, Z. Guo, M. Gheshlaghi Azar, et al. Bootstrap your own latent-a new ap- proach to self-supervised learning. *Advances in neural information processing systems*, 33: 21271–21284, 2020. [15] M. S. Halvagal, A. Laborieux, and F. Zenke. Implicit variance regularization in non-contrastive ssl. *arXiv preprint arXiv:2212.04858*, 2022. [16] K. Hermann and A. Lampinen. What shapes feature representations? exploring datasets, architectures, and training. *Advances in Neural Information Processing Systems*, 33:9995– 10006, 2020. [17] K. L. Hermann, H. Mobahi, T. Fel, and M. C. Mozer. On the foundations of shortcut learning. *arXiv preprint arXiv:2310.16228*, 2023. [18] A. Jacot, F. Gabriel, and C. Hongler. Neural tangent kernel: Convergence and generalization in neural networks. *Advances in neural information processing systems*, 31, 2018. [19] A. Jacot, F. Ged, B. ¸Sim¸sek, C. Hongler, and F. Gabriel. Saddle-to-saddle dynamics in deep linear networks: Small initialization training, symmetry, and sparsity. *arXiv preprint arXiv:2106.15933*, 2021. [20] E. Littwin, O. Saremi, M. Advani, V. Thilak, P. Nakkiran, C. Huang, and J. Susskind. How jepa avoids noisy features: The implicit bias of deep linear self distillation networks. *arXiv preprint arXiv:2407.03475*, 2024. [21] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. *arXiv preprint arXiv:1711.05101*, 2017. [22] M. Noroozi, H. Pirsiavash, and P. Favaro. Representation learning by learning to count. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, Oct 2017.

[23] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al. Dinov2: Learning robust visual features without supervision. *arXiv preprint arXiv:2304.07193*, 2023. [24] S. Pesme and N. Flammarion. Saddle-to-saddle dynamics in diagonal linear networks. *Advances in Neural Information Processing Systems*, 36:7475–7505, 2023. [25] N. Rahaman, A. Baratin, D. Arpit, F. Draxler, M. Lin, F. Hamprecht, Y. Bengio, and A. Courville. On the spectral bias of neural networks. In *International conference on machine learning*, pages 5301–5310. PMLR, 2019. [26] J. B. Simon, M. Knutins, L. Ziyin, D. Geisz, A. J. Fetterman, and J. Albrecht. On the stepwise nature of self-supervised learning. In *International Conference on Machine Learning*, pages 31852–31876. PMLR, 2023. [27] M. Tancik, P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal, R. Ra- mamoorthi, J. Barron, and R. Ng. Fourier features let networks learn high frequency functions in low dimensional domains. *Advances in neural information processing systems*, 33:7537–7547, 2020. [28] S. Wang, R. Veldhuis, C. Brune, and N. Strisciuglio. What do neural networks learn in image classification? a frequency shortcut perspective. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 1433–1442, 2023. [29] D. Wei, J. J. Lim, A. Zisserman, and W. T. Freeman. Learning and using the arrow of time. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, June 2018. [30] S. Wu, S. Chen, C. Xie, and X. Huang. One-pixel shortcut: on the learning preference of deep neural networks. *arXiv preprint arXiv:2205.12141*, 2022. [31] J. Zbontar, L. Jing, I. Misra, Y. LeCun, and S. Deny. Barlow twins: Self-supervised learning via redundancy reduction. In *International conference on machine learning*, pages 12310–12320. PMLR, 2021.
# <sup>418</sup> A Experimental Details

#### <sup>419</sup> A.1 Extent bias Experiment

<sup>420</sup> For the extent bias experiment shown in Section 4.1, we train the model using 400 epochs. The <sup>421</sup> augmentation noise parameter a was set to 0.01. We use a dataset size of n = 1000 samples with feature dimension m = 10. We also use learning rate η = 6 · 10−<sup>4</sup> and scaling factor 5 · 10−<sup>1</sup> <sup>422</sup> .

#### <sup>423</sup> A.2 Colored MNIST Experiment

 For the Colored MNIST shown in Section 4.8, we train the model using default augmentation (RandomResizedCrop, RandomHorizontalFlip, RandomColorJitter, RandomGrayscale, Random- GaussianBlur, RandomSolarization) with augmentated image size 42×42. We use background colors as [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [0, 123, 123], [123, 0, 123], [123, 123, 0], [123, 0, 0]][digit]. We trained ResNet18 with 60 epochs, AdamW [21] with learning rate η = 4 × 10−<sup>6</sup> <sup>429</sup> .

#### <sup>430</sup> A.3 Amplitude Experiment

<sup>431</sup> For the amplitude experiment shown in Section 5.1, we train the model using 500 epochs. The <sup>432</sup> augmentation noise parameter a is set to 0.1. We use a dataset size of n = 1000 samples with feature frequency f<sup>h</sup> = 2 <sup>2</sup><sup>π</sup> <sup>24</sup> , f<sup>l</sup> = 32 <sup>2</sup><sup>π</sup> <sup>24</sup> . We also use learning rate <sup>η</sup> = 5 · <sup>10</sup>−<sup>5</sup> , scaling factor 3 · 10−<sup>3</sup> <sup>433</sup> and m = 96.

![](_page_11_Figure_7.jpeg)

Figure 4: Input data x = xbase + ϵ. xbase[t] = bhc<sup>h</sup> sin(fht) + blc<sup>l</sup> sin(flt), where c<sup>h</sup> = 1, c<sup>l</sup> = 0.5, f<sup>h</sup> = 2π <sup>m</sup> 32, f<sup>l</sup> = 2π <sup>m</sup> 8, m = 96.

![](_page_11_Figure_9.jpeg)

Figure 5: The eigenvectors vi's of Γ for i = 1, 2, 3 (from Left to Right). (Left) The first eigenvector that correspondent to the largest eigenvalue indicates the (high frequency) feature with a high amplitude c<sup>h</sup> sin (fht), (Center) the second the (low frequency) feature with a low amplitude feature c<sup>l</sup> sin (flt), (Right) the third (and beyond) noise, where c<sup>l</sup> < ch.

434

# <sup>435</sup> B Proofs

# <sup>436</sup> B.1 Proof of Theorem 4.1

<sup>437</sup> Through matrix analysis, we can express:

$$\Gamma = \mathbb{E}[x_{\text{base}} x_{\text{base}}^\top] = \begin{bmatrix} \mathbf{1}_{m_l \times m_l} & \mathbf{0}_{m_s \times m_s} \\ \mathbf{0}_{m_l \times m_s} & \mathbf{1}_{m_s \times m_s} \end{bmatrix},$$

<sup>438</sup> which has two eigenvectors el/∥el∥ and es/∥es∥ correspond to nonzero eigenvalues. We get the <sup>439</sup> eigenvalues m<sup>l</sup> and m<sup>s</sup> from the following equation:

$$\det(\Gamma - \lambda I) = \det(\mathbf{1}_{m_l \times m_l} - \lambda I_{m_l \times m_l}) \det(\mathbf{1}_{m_s \times m_s} - \lambda I_{m_s \times m_s}) = 0.$$

<sup>440</sup> Finally, we can get the eigendecomposition Γ = VΓΛΓV<sup>Γ</sup> where

$$\Lambda_\Gamma = \text{diag}([m_l, m_s, \mathbf{0}_{m-2}]),$$

$$V_\Gamma^{(\leq d)} = \left[ \frac{1}{\sqrt{m_l}} e_l \frac{1}{\sqrt{m_s}} e_s \right].$$

## <sup>441</sup> B.2 Proof of Theorem 5.1

<sup>442</sup> The cross-correlation matrix Γ for this input can be expressed using (5):

$$\begin{aligned}\Gamma &= \mathbb{E}[x_{\text{base}} x_{\text{base}}^\top] \\ &= \mathbb{E}[c_h^2 b_h^2 \sin(f_h t) \sin(f_h t)^\top + c_l^2 b_l^2 \sin(f_l t) \sin(f_l t)^\top + c_h c_l b_h b_l \sin(f_h t) \sin(f_l t)^\top + c_h c_l b_h b_l \sin(f_l t) \sin(f_h t)^\top] \\ &= c_h^2 \sin(f_h t) \sin(f_h t)^\top + c_l^2 \sin(f_l t) \sin(f_l t)^\top.\end{aligned}$$

<sup>443</sup> Using the orthogonality between sin(fht) and sin(flt) (f<sup>h</sup> ̸= fl), where t ∈ <sup>N</sup>,

$$\begin{aligned}\Gamma &= c_h^2 \sin(f_h t) \sin(f_h t)^\top + c_l^2 \sin(f_l t) \sin(f_l t)^\top, \\\Gamma \sin(f_h t) &= c_h^2 \|\sin(f_h t)\|^2 \sin(f_h t), \\\Gamma \sin(f_l t) &= c_l^2 \|\sin(f_l t)\|^2 \sin(f_l t).\end{aligned}$$

<sup>444</sup> We find eigenvector and eigenvalue as:

$$\begin{aligned} \Lambda_\Gamma &= \text{diag} \left( [c_h^2 \|\sin(f_h t)\|^2, c_i^2 \|\sin(f_i t)\|^2, \mathbf{0}_{m-2}] \right), \\ V_\Gamma^{(\leq 2)} &= [e_h \ e_i]^\top. \end{aligned}$$

With f = 2π <sup>445</sup> <sup>m</sup> k for some integer k, we have

$$\begin{aligned} \|\sin(fx)\|^2 &= \int_0^m \sin^2(fx) dx = \int_0^m \frac{1 - \cos(2fx)}{2} dx \\ &= \frac{1}{2} \left[ x - \frac{\sin(2fx)}{2} \right]_0^m = \frac{m}{2} - \frac{\sin(2fm)}{4} = \frac{m}{2}. \end{aligned}$$

<sup>446</sup> Finally, we have

$$\Lambda_\Gamma = \text{diag} \left( \left[ c_h^2 \frac{m}{2}, c_l^2 \frac{m}{2}, \mathbf{0}_{m-2} \right] \right),$$

$$V_\Gamma^{(\leq 2)} = [e_h \ e_l] .$$

<sup>447</sup> B.3 Proof of Theorem 4.2

<sup>448</sup> We have

$$\tilde{\lambda}_j(t) = \tilde{s}_j^2(t)\gamma_j = (1 + \lambda_j(0)^{-1}e^{-8\gamma_j t})^{-1},$$

and thus if we plug in τ<sup>j</sup> = − log(λ<sup>j</sup> (0))/8γ<sup>j</sup> , i.e., exp(−8γ<sup>j</sup> τ<sup>j</sup> ) = λ<sup>j</sup> (0), then we have λ˜ <sup>449</sup> <sup>j</sup> (τ<sup>j</sup> ) = (1 + 1)<sup>−</sup><sup>1</sup> = 2 . The derivative λ˜′ j <sup>450</sup> (t) at t = τ<sup>j</sup> is given as follows:

$$\begin{aligned}\tilde{\lambda}'_j(t) &= -(1 + \lambda_j(0)^{-1}e^{-8\gamma_j t})^{-2}(-8\gamma_j\lambda_j(0)^{-1}e^{-8\gamma_j t}) \\ &= -\tilde{\lambda}_j^2(t)(-8\gamma_j\lambda_j(0)^{-1}e^{-8\gamma_j t}) \\ \tilde{\lambda}'_j(\tau_j) &= -\tilde{\lambda}_j^2(\tau_j)(-8\gamma_j\lambda_j^{-1}(0)\lambda_j(0)) \\ &= 2\gamma_j.\end{aligned}$$

<sup>451</sup> Using the equations

$$C = \sum_{j=1}^d \lambda_j u_j u_j^\top \text{ and } C^2 = \sum_{j=1}^d \lambda_j^2 u_j u_j^\top,$$

<sup>452</sup> we get the loss

$$\begin{aligned}\mathcal{L} &= \|C - I\|_F^2 = \text{Tr}((C - I)(C - I)) = \text{Tr}(C^2) - 2 \text{Tr}(C) + d \\ &= \sum_{j=1}^d \lambda_j^2 - 2 \sum_{j=1}^d \lambda_j + d = \sum_{j=1}^d (\lambda_j - 1)^2.\end{aligned}$$

<sup>453</sup> Thus, we get the following equation:

$$\begin{aligned} \frac{d\mathcal{L}}{dt}(\tau_j) &= \sum_{i=1}^d 2(\lambda_i(\tau_j) - 1)\lambda'_i(\tau_j) \\ &\approx \sum_{i=1}^d 2(\tilde{\lambda}_i(\tau_j) - 1)\tilde{\lambda}'_i(\tau_j) \\ &\approx 2(\tilde{\lambda}_j(\tau_j) - 1)\tilde{\lambda}'_j(\tau_j) \\ &= -\tilde{\lambda}'_j(\tau_j) = -2\gamma_j. \end{aligned}$$

## <sup>454</sup> B.4 Proof of Theorem 4.3

<sup>455</sup> First, we have

$$\begin{aligned}\tilde{s}_j(t) &= (\gamma_j + s_j^{-2}(0) \exp(-8\gamma_j t))^{-1/2}, \\ \tilde{s}_j(\tau_j) &= (\gamma_j + s_j^{-2}(0)\lambda_j(0))^{-1/2} \\ &= (2\gamma_j)^{-1/2}.\end{aligned}$$

<sup>456</sup> and its derivative is given as follows:

$$\begin{aligned} \tilde{s}'_j(t) &= -\frac{1}{2}(\gamma_j + s_j^{-2}(0) \exp(-8\gamma_j t))^{-3/2}(-8\gamma_j s_j^{-2}(0) \exp(-8\gamma_j t)), \\ \tilde{s}'_j(\tau_j) &= -\frac{1}{2}(\gamma_j + s_j^{-2}(0) \lambda_j(0))^{-3/2}(-8\gamma_j s_j^{-2}(0) \lambda_j(0)) \\ &= -\frac{1}{2}(2\gamma_j)^{-3/2}(-8\gamma_j^2) \\ &= (2\gamma_j)^{1/2}. \end{aligned}$$

# <sup>457</sup> C Derivatives

![](_page_14_Figure_1.jpeg)

Figure 6: Derivatives dλ<sup>l</sup> dt (blue), dλ<sup>s</sup> dt (red), and − dL dt (black dashed). The derivative dλ<sup>l</sup> dt (τl) (solid blue), dλ<sup>s</sup> dt (τs) (solid red) are approximately equal to 2γ<sup>l</sup> = 2m<sup>l</sup> (dashed blue), 2γ<sup>s</sup> = 2m<sup>s</sup> (dashed red).

![](_page_14_Figure_3.jpeg)

Figure 7: Evolution of s<sup>j</sup> (t) and s ′ j (t). (Left) Evolution of singular values s<sup>l</sup> (solid blue) and s<sup>s</sup> (solid red) of W during training. They converge near to 1/ √<sup>γ</sup><sup>l</sup> = 1/<sup>3</sup> (dashed horizontal blue) and 1/ √<sup>γ</sup><sup>s</sup> = 1 (dashed horizontal red), respectively. The predicted singular values (dashed blue, dashed red) match the empirical result. (Right) Evolution of the derivatives ds<sup>l</sup> dt (solid blue) and ds<sup>s</sup> dt (solid red). The derivatives ds<sup>l</sup> dt (τl), ds<sup>s</sup> dt (τs) are approximately equal to √ 2γ<sup>l</sup> (dashed horizontal blue), √ 2γ<sup>s</sup> (dashed horizontal red). The predicted derivatives of singular values (dashed blue, dashed red) also match the empirical result. We use m<sup>l</sup> = 9 and m<sup>s</sup> = 1.

![](_page_15_Figure_0.jpeg)

Figure 8: Derivatives dλ<sup>h</sup> dt (blue), dλ<sup>l</sup> dt (red), and − dL dt (black dashed). The derivative dλ<sup>h</sup> dt (τh) (solid blue), dλ<sup>l</sup> dt (τl) (solid red) are approximately equal to 2γ<sup>h</sup> = 2c h (dashed blue), 2γ<sup>l</sup> = 2c 2 l (dashed red). See Figure 6 together.

![](_page_15_Figure_2.jpeg)

Figure 9: Evolution of s<sup>j</sup> (t) and s ′ j (t). See the caption of Figure 7. (Left) They converge near to 1/ √<sup>γ</sup><sup>h</sup> = 1/ q c 2 h m and 1/ √<sup>γ</sup><sup>l</sup> = 1/ q c 2 l m . (Right) The derivatives ds<sup>h</sup> dt (τh), ds<sup>l</sup> dt (τl) are approximately equal to √ 2γh, √ 2γ<sup>l</sup> . We use c<sup>h</sup> = 1 and c<sup>l</sup> = 1/2.

# <sup>458</sup> D Subspace Alignment

![](_page_16_Figure_1.jpeg)

Figure 10: Evolution of subspace alignment SA(V (≤d) , V (≤d) Γ ) (d = 2) between the top-d right singular vectors of W and eigenvectors of Γ. We use the data (Top) from Section 4.1 and (Bottom) from Section 5.1. See Appendix A.

# <sup>459</sup> E Orthogonal Feature Learning

![](_page_17_Figure_1.jpeg)

Figure 11: Visualization of the trajectory of W e<sup>l</sup> and W e<sup>s</sup> on the subspace spanned by u1, u<sup>2</sup> during training. The high-dimensional feature W e<sup>h</sup> (blue solid line) aligns with u<sup>1</sup> and the lowdimensional feature W e<sup>l</sup> (red solid line) aligns with u2. Dashed lines are predicted trajectory (see Theorem 4.5).

![](_page_17_Figure_3.jpeg)

Figure 12: Visualization of the trajectory of W e<sup>h</sup> and W e<sup>l</sup> on the subspace spanned by u1, u<sup>2</sup> during training. See the caption of Figure 11.

# <sup>460</sup> F Right Singular Vectors of W

![](_page_18_Figure_1.jpeg)

Figure 13: The first two right singular vectors (Top/Bottom) of W during training (from Left to Right). (Left) At t = 0, the two singular vectors are just noise. (Center) A little after t = τ1, the first singular value reaches the plateau as shown in Figure 3 and only the (high frequency) feature with a high amplitude is learned. (Right) At the convergence, the model learns the two features.

# <sup>461</sup> G Non-linear Experiments

![](_page_19_Figure_1.jpeg)

Figure 14: Effects of extent bias on learning dynamics in non-linear network. (Left) Stepwise learning curves of Barlow Twins. There are two (d = 2) learning steps shown with two black dashed vertical lines (also shown in the other two panels) on empirical result (solid green). (Right) Evolution of eigenvalues λ<sup>j</sup> 's of C during training. At the beginning, the first eigenvalue λ<sup>1</sup> (blue) increases to 1 and then later the second λ<sup>2</sup> (red) follows. We use same inputs in Figure 1.

![](_page_19_Figure_3.jpeg)

Figure 15: Amplitude bias effects on learning dynamics in non-linear network. (Left) Stepwise learning curves of Barlow Twins showing two distinct learning phases with vertical dashed lines marking critical transition points during training. The green line shows empirical loss decreasing in two clear stages. (Right) Evolution of eigenvalues λ<sup>j</sup> of correlation matrix C during training. The eigenvalue λ<sup>l</sup> (blue) increases first, followed by the eigenvalue λ<sup>s</sup> (red), demonstrating amplitudebased learning prioritization. We use same inputs in Figure 3.

# <sup>462</sup> H Limitations

<sup>463</sup> Our study has several limitations due to its simplified assumptions. While our theoretical analysis <sup>464</sup> provides valuable insights into the relationship between extent bias and shortcut learning, several <sup>465</sup> limitations should be acknowledged:

 • Linear Network Assumption: We focus on one-layer linear networks, which may not capture the complexities of multi-layer non-linear neural networks. • Feature Independence: Our assumption of independent features may not reflect the complex interdependencies in practical scenarios. • Augmentation Limitations: Augmentation Limitations: Our basic augmentation approach may not fully represent the sophisticated strategies used in modern SSL methods.

<sup>472</sup> Future work could address these limitations by extending the theoretical framework to non-linear <sup>473</sup> networks, incorporating feature interactions, and analyzing the impact of more complex augmentation <sup>474</sup> strategies.

# <sup>475</sup> I Supplementary Studies

## <sup>476</sup> I.1 Non-linear Feature Learned Measurement

<sup>477</sup> Nonlinearity exhibits distinct learning dynamics compared to linearity. Therefore, we aim to investi-<sup>478</sup> gate whether extent biases also exists in multilayer perceptrons (MLPs). We define a measurement of <sup>479</sup> feature learning as:

 Definition I.1. (Feature Learning Distance). When a model f(·, θ) has sufficiently learned a specific latent feature vector e<sup>f</sup> , f(X, θ) contains information about e<sup>f</sup> for input X = p(e<sup>f</sup> ) ∈ R<sup>m</sup> where p represents some non-linear transformation function. Consequently, if a simple linear probing function g can extract e<sup>f</sup> from f(X, θ), we can define that the model f has meaningfully learned e<sup>f</sup> . Furthermore, to quantify the degree of learning, assuming an optimally trained probe g, we define a feature learning metric

$$\text{FLD}(k) = \min_g \mathbb{E}_{e_f \in \mathcal{P}_k} \left[ \frac{\text{MSE}(g(f(X, \theta)), e_f)}{\|e_f\|_2^2} \right], \quad (9)$$

<sup>486</sup> where P<sup>k</sup> is distribution of feature k.

# <sup>487</sup> I.2 Non-linear on extent bias

<sup>488</sup> We experiment on Section 4.7, for understanding non-linear feature learning dynamics. Figure 14 <sup>489</sup> shows this results.

![](_page_20_Figure_12.jpeg)

Figure 16: Effects of extent bias on learning dynamics in non-linear network. (Left) Stepwise learning curves of Barlow Twins. There are two (d = 2) learning steps shown with two black dashed vertical lines (also shown in the other two panels) on empirical result (solid green). (Center) Evolution of eigenvalues λ<sup>j</sup> 's of C during training. At the beginning, the first eigenvalue λ<sup>1</sup> (blue) increases to 1 and then later the second λ<sup>2</sup> (red) follows. (Right) Evolution of the feature learning distance FLD(e) for e<sup>l</sup> (blue) and e<sup>s</sup> (red). See Definition I.1. We use m<sup>l</sup> = 9, m<sup>s</sup> = 1. See Appendix A.1 for more detailed settings.

<sup>490</sup> From Figure 16, we observe FLD(el) drop earlier than FLD(es). Therefore, the phenomenon of e<sup>l</sup> <sup>491</sup> being learned before e<sup>s</sup> is consistent with the linear case.

![](_page_21_Figure_0.jpeg)

Figure 17: Amplitude bias effects on learning dynamics in non-linear network. (Left) Stepwise learning curves of Barlow Twins showing two distinct learning phases with vertical dashed lines marking critical transition points during training. The green line shows empirical loss decreasing in two clear stages. (Center) Evolution of eigenvalues λ<sup>j</sup> of correlation matrix C during training. The eigenvalue λ<sup>h</sup> (blue) increases first, followed by the eigenvalue λ<sup>l</sup> (red), demonstrating amplitude-based learning prioritization. (Right) Evolution of feature learning distance FLD(e) for high-amplitude feature el (blue) and low-amplitude feature es (red), confirming that features with higher amplitude coefficients (ch) are learned before those with lower amplitude (cl), even in non-linear architectures. Note that FLD decreases as the network learns to represent the corresponding feature. We use c<sup>h</sup> = 1, c<sup>l</sup> = 0.5 and a 3-layer network with leakyReLU activations. See the caption of Figure 16. See Appendix A for additional experimental details.

## <sup>492</sup> I.3 Non-linear on amplitude bias

 Using Definition I.1, we experiment on Section 5.7. Figure 17 demonstrates amplitude bias effects in non-linear networks. The results show that features with higher amplitude (ch) are learned before those with lower amplitude (cl), consistent with our linear model findings. Specifically, F LD(eh) decreases earlier than F LD(el), mirroring the eigenvalue increase patterns observed in the left and center panels. These results confirm that amplitude bias persists in non-linear architectures, suggesting that amplitude magnitude remains a primary determinant of feature learning priority regardless of network complexity. This provides additional evidence that deep learning models respond more sensitively to amplitude characteristics than frequency properties, even when non-linearities are introduced.

# <sup>502</sup> I.4 Eigenvalues on Shift Augmentation

$$x_{base} = c_a \sin(f_a t + \epsilon_a) + c_b \sin(f_b t + \epsilon_b)$$

$$\epsilon_a, \epsilon_b \stackrel{\text{i.i.d.}}{\sim} U(-\pi, \pi)$$

503

$$\begin{aligned}\Gamma &= \mathbb{E}[x_{base}x_{base}^\top] \\ \Gamma_{ij} &= \mathbb{E}[c_a^2 \sin(f_a i + \epsilon_a) \sin(f_a j + \epsilon_a) + c_a c_b \sin(f_a i + \epsilon_a) \sin(f_b j + \epsilon_b) \\ &\quad + c_a c_b \sin(f_b i + \epsilon_b) \sin(f_a j + \epsilon_a) + c_b^2 \sin(f_b i + \epsilon_b) \sin(f_b j + \epsilon_b)]\end{aligned}$$

504

$$\begin{aligned}\mathbb{E}_{\epsilon_a, \epsilon_b}[\sin(\theta_a + \epsilon_a) \sin(\theta_b + \epsilon_b)] &= \mathbb{E}_{\epsilon_a, \epsilon_b}[\text{Im}(\exp(i(\theta_a + \epsilon_a))) \text{Im}(\exp(i(\theta_b + \epsilon_b)))] \\ &= \mathbb{E}_{\epsilon_a}[\text{Im}(\exp(i(\theta_a + \epsilon_a))) \mathbb{E}_{\epsilon_b}[\text{Im}(\exp(i(\theta_b + \epsilon_b)))] \\ &= \text{Im}(\mathbb{E}_{\epsilon_a}[\exp(i(\theta_a + \epsilon_a))]) \text{Im}(\mathbb{E}_{\epsilon_b}[\exp(i(\theta_b + \epsilon_b))]) \\ &= \text{Im}(\mathbb{E}_{\epsilon_a}[\exp(i\epsilon_a) \exp(i\theta_a)]) \text{Im}(\mathbb{E}_{\epsilon_b}[\exp(i\epsilon_b) \exp(i\theta_b)]) \\ &= \text{Im}(\varphi(1) \exp(i\theta_a)) \text{Im}(\varphi(1) \exp(i\theta_b))\end{aligned}$$

<sup>505</sup> We can define u, d as u = µ + α, d = µ − α, α = 2π.

$$\varphi(1) = \frac{\exp(iu) - \exp(id)}{i(u - d)} = \frac{\exp(i\mu)}{\alpha i} \frac{\exp(i\alpha) - \exp(-i\alpha)}{2i} = \frac{\exp(i\mu)}{\alpha i} \sin(\alpha) = 0$$

$$\mathbb{E}_{\epsilon_a, \epsilon_b} [\sin(\theta_a + \epsilon_a) \sin(\theta_b + \epsilon_b)] = 0$$

<sup>507</sup> Similar,

$$\begin{aligned}\mathbb{E}[\sin(\theta_a + \epsilon_a) \sin(\theta_b + \epsilon_a)] &= -\frac{1}{2}\mathbb{E}[\cos(\theta_a + \theta_b + 2\epsilon_a) - \cos(\theta_a - \theta_b)] \\ &= -\frac{1}{2}\mathbb{E}[\cos(\theta_a + \theta_b + 2\epsilon_a)] + \frac{1}{2}\cos(\theta_a - \theta_b) \\ &= -\frac{1}{2} \int_a^b \left[ \frac{1}{b-a} \cos(\theta_a + \theta_b + 2x) dx \right] + \frac{1}{2} \cos(\theta_a - \theta_b) \\ &= -\frac{1}{4} \frac{1}{b-a} [\sin(\theta_a + \theta_b + 2b) - \sin(\theta_a + \theta_b + 2a)] + \frac{1}{2} \cos(\theta_a - \theta_b) \\ &= -\frac{1}{4} \frac{1}{b-a} [2 \cos(\theta_a + \theta_b + a + b) \sin(b-a)] + \frac{1}{2} \cos(\theta_a - \theta_b)\end{aligned}$$

<sup>508</sup> we assumed b − a = 2π,

$$\mathbb{E}[\sin(\theta_a + \epsilon_a) \sin(\theta_b + \epsilon_a)] = \frac{1}{2} \cos(\theta_a - \theta_b)$$

<sup>509</sup> finally, we get

$$\Gamma_{ij} = \frac{c_a^2}{2} \cos(f_a(i - j)) + \frac{c_b^2}{2} \cos(f_b(i - j))$$

is symmetric circulant matrix when f<sup>a</sup> = a 2π N , f<sup>b</sup> = b 2π N <sup>510</sup> ,

$$\begin{aligned} c_j &= \frac{c_a^2}{2} \cos(f_a j) + \frac{c_b^2}{2} \cos(f_b j) \\ \Lambda_{\Gamma,k} &= \sum_{j=0}^{N-1} c_j \omega^{-kj} \\ V_{\Gamma,k} &= \frac{1}{\sqrt{N}} \left[ 1, \omega^k, \omega^{2k}, \dots, \omega^{(N-1)k} \right]^T \\ \omega &= \exp\left(\frac{2\pi i}{n}\right) = \cos\left(\frac{2\pi}{n}\right) + i \sin\left(\frac{2\pi}{n}\right) \end{aligned}$$

<sup>511</sup> This is symmetric, so eigenvalues are real. The eigenvectors can be expressed either in complex form <sup>512</sup> or as pairs of real vectors. Using properties of Discrete Fourier Transform (DFT) matrix on ΛΓ,k,

$$\Lambda_{\Gamma, k} = \begin{cases} 0 & (k \neq l_a, N - l_a, l_b, N - l_b) \\ \frac{c_a^2}{2} & (k = l_a \text{ or } k = N - l_a) \\ \frac{c_b^2}{2} & (k = l_b \text{ or } k = N - l_b) \end{cases}$$

<sup>513</sup> Finally, we can derive as:

$$\Lambda_\Gamma = \text{diag} \left( \left[ \frac{c_a^2}{2}, \frac{c_b^2}{2}, \frac{c_b^2}{2}, \frac{c_b^2}{2}, \mathbf{0}_{m-2} \right] \right),$$

$$V_\Gamma^{(\leq 4)} = \left[ \frac{1}{\sqrt{N}} e_{h, \text{cos}} \frac{1}{\sqrt{N}} e_{h, \text{sin}} \frac{1}{\sqrt{N}} e_{l, \text{cos}} \frac{1}{\sqrt{N}} e_{l, \text{sin}} \right].$$

<sup>514</sup> where

$$\begin{aligned} e_{h,\cos} &= c_a \cos(f_a t), \\ e_{h,\sin} &= c_a \sin(f_a t), \\ e_{l,\cos} &= c_b \cos(f_b t), \\ e_{l,\sin} &= c_b \sin(f_b t). \end{aligned}$$

# NeurIPS Paper Checklist

#### 1. Claims

 Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

 Justification: The abstract and introduction clearly state our main contributions: (1) establish- ing theoretical connections between shortcut learning, stepwise learning, and dataset's cross correlation's eigendecomposition in SSL, (2) extending theoretical research on shortcut learning to SSL, and (3) characterizing extent bias and amplitude bias in learning dynamics. These claims accurately reflect the scope of our work as demonstrated in Section 4, and Section 5 where we provide both theoretical foundations and empirical validation.

Guidelines:

 • The answer NA means that the abstract and introduction do not include the claims made in the paper. • The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers. • The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings. • It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

 Justification: We acknowledge the limitations of our work in Appendix H. Our analysis primarily focuses on linear networks, which may not fully capture the complexities of deep non-linear architectures used in practice. We also assume feature independence which simplifies analysis but may not reflect real-world feature interdependencies. Additionally, our augmentation approach is more basic than sophisticated strategies used in modern SSL systems. We suggest future research directions to address these limitations.

Guidelines:

 • The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper. • The authors are encouraged to create a separate "Limitations" section in their paper. • The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be. • The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated. • The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon. • The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size. • If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

 • While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an impor- tant role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

# 3. Theory assumptions and proofs

 Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

 Justification: All theoretical results in our paper are presented with complete assumptions and rigorous proofs. Each theorem explicitly states its assumptions and corresponding proofs are provided in Appendix B with detailed derivations. We use a consistent numbering system for cross-referencing and provide proof sketches in the main paper to build intuition before directing readers to the complete proofs.

Guidelines:

 • The answer NA means that the paper does not include theoretical results. • All the theorems, formulas, and proofs in the paper should be numbered and cross- referenced. • All assumptions should be clearly stated or referenced in the statement of any theorems. • The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition. • Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.

# 4. Experimental result reproducibility

 Question: Does the paper fully disclose all the information needed to reproduce the main ex- perimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

 Justification: We provide comprehensive details to reproduce our experimental results in Section 4 and Section 5, with additional specifics in Appendix A.

Guidelines:

 • The answer NA means that the paper does not include experiments. • If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not. • If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable. • Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed. • While NeurIPS does not require releasing code, the conference does require all submis- sions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

 (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

# 5. Open access to data and code

 Question: Does the paper provide open access to the data and code, with sufficient instruc- tions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: Yes

Guidelines:

 • The answer NA means that paper does not include experiments requiring code. • Please see the NeurIPS code and data submission guidelines (https://nips.cc/ public/guides/CodeSubmissionPolicy) for more details. • While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark). • The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //nips.cc/public/guides/CodeSubmissionPolicy) for more details. • The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc. • The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why. • At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable). • Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

# 6. Experimental setting/details

 Question: Does the paper specify all the training and test details (e.g., data splits, hyper- parameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [Yes]

 Justification: Section 4.1 and Section 5.1 detail our experimental setup, while Appendix A provides comprehensive information about hyperparameters, training procedures, and imple- mentation details. Extent bias experiments, we specify relevant parameters including dataset size, feature dimensions, learning rates. All essential information needed to understand and reproduce our results is included.

Guidelines:

 • The answer NA means that the paper does not include experiments. • The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them. • The full details can be provided either with the code, in appendix, or as supplemental material.

# 7. Experiment statistical significance

 Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Yes

Guidelines:

 • The answer NA means that the paper does not include experiments. • The authors should answer "Yes" if the results are accompanied by error bars, confi- dence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper. • The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions). • The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.) • The assumptions made should be given (e.g., Normally distributed errors). • It should be clear whether the error bar is the standard deviation or the standard error of the mean. • It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified. • For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates). • If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

# 8. Experiments compute resources

 Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [No]

 Justification: Our experiments do not require a lot of resources. We used a single L40s GPU for training Resnet18, and used L4 GPU for linear model.

Guidelines:

 • The answer NA means that the paper does not include experiments. • The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage. • The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute. • The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

# 9. Code of ethics

 Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: Our research fully complies with the NeurIPS Code of Ethics.

Guidelines:

 • The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. • If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics.

 • The authors should make sure to preserve anonymity (e.g., if there is a special consid-eration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

 Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

 Justification: We discuss broader impacts in Section 6. Positively, our work could lead to more robust machine learning models that are less susceptible to shortcut learning, potentially improving fairness and reliability in real-world applications. Understanding extent bias may help address issues where models learn background correlations rather than meaningful object features.

## Guidelines:

 • The answer NA means that there is no societal impact of the work performed. • If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact. • Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations. • The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster. • The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology. • If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

# 11. Safeguards

 Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

 Justification: Our work is primarily theoretical with controlled toy experiments that do not produce models or datasets with potential for misuse. We do not release pre-trained models, generative systems, or scraped datasets that would require safeguards against harmful applications.

# Guidelines:

 • The answer NA means that the paper poses no such risks. • Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters. • Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images. • We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

# 12. Licenses for existing assets

 Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

 Justification: We properly cite all relevant prior work including Simon et al. [26] and Zbontar et al. [31] whose theoretical frameworks we build upon. For the Colored-MNIST dataset adaptation in Section 4.8, we acknowledge the original MNIST dataset Deng [9] which is in the public domain. No proprietary or restrictively licensed code or data was used in our research.

Guidelines:

 • The answer NA means that the paper does not use existing assets. • The authors should cite the original paper that produced the code package or dataset. • The authors should state which version of the asset is used and, if possible, include a URL. • The name of the license (e.g., CC-BY 4.0) should be included for each asset. • For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided. • If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset. • For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided. • If this information is not available online, the authors are encouraged to reach out to the asset's creators.

# 13. New assets

 Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [NA]

 Justification: Our paper does not introduce new datasets or code libraries intended for community use beyond the experimental validation of our theoretical claims

Guidelines:

 • The answer NA means that the paper does not release new assets. • Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc. • The paper should discuss whether and how consent was obtained from people whose asset is used. • At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

# 14. Crowdsourcing and research with human subjects

 Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

 Justification: Our research is purely theoretical and computational, involving no human subjects, crowdsourced data collection, or human evaluation.

Guidelines:

 • Including this information in the supplemental material is fine, but if the main contribu- tion of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

#### 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

 Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

 Justification: No human subjects were involved in our research, so IRB approval was not required or sought.

Guidelines:

 • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

# 16. Declaration of LLM usage

 Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [NA]

 Justification: No large language models were used in the development of our research methodology, theoretical analysis, or experimental design.

Guidelines:

 • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what should or should not be described.