# From Steering Vectors To Conceptors: Compositional Affine Activation Steering For Llms

Anonymous Author(s)
Affiliation Address email

## Abstract 14 **1 Introduction**

1 Controlling and understanding the internal representations of large language models 2 (LLMs) remain central challenges. We combine conceptor theory with activation 3 steering to develop a principled framework for provably optimal affine steering 4 of LLM activations. Conceptors compress sets of activation vectors and act as 5 soft projection matrices, enabling precise and interpretable control over internal 6 states. Our framework derives optimal steering functions from first principles and 7 consistently outperforms additive steering across in-context learning tasks and 8 alignment-relevant behavior. We further demonstrate how Boolean operations over 9 conceptors allow for compositional steering toward multiple objectives, yielding 10 better performance than traditional vector combination methods. Together, these 11 results establish conceptor-based steering as a powerful tool for both controlling 12 LLM behavior and gaining insight into their internal mechanisms. We will release 13 our code and data as part of a flexible open-source library for activation steering. 15 Large Language Models (LLMs) have rapidly advanced AI capabilities (Xu & Poo, 2023), but their 16 potential for misinformation (Pan et al., 2023), reinforcing biases (Gallegos et al., 2024), and harmful 17 behaviors (Shevlane et al., 2023) necessitates methods to understand their internals and control their 18 outputs. While approaches like Reinforcement Learning from Human Feedback (RLHF) (Ouyang 19 et al., 2024), supervised fine-tuning (Devlin et al., 2019), and prompt engineering (Liu et al., 2023) 20 aim to control LLMs, they are often computationally expensive, struggle with generalization (Bottou 21 et al., 2018; Amodei et al., 2016), or yield inconsistent results (Chen et al., 2023).

22 Activation steering (AS) has emerged as a promising alternative, in which one modifies the model's 23 activations at inference without needing costly parameter updates. Early work into AS demonstrated 24 the potential of modifying internal activations in LLMs at inference. Subramani et al. (2022) 25 introduced "steering vectors" added to hidden states to guide generation, though their sample-specific 26 optimization limited scalability. Turner et al. (2023) proposed a contrastive approach in which 27 steering vectors are computed from the activation differences of contrastive prompt pairs, effectively 28 controlling sentiment, topics, and styles. This more efficient method was then further refined by 29 (Rimsky et al., 2024b) where larger datasets of contrastive pairs were used to generate more precise 30 steering vectors. These foundational methods, while pioneering, primarily relied on simple vector 31 arithmetic and laid the groundwork for numerous applications, from exposing vulnerabilities (Wang 32 & Shu, 2024; Ghandeharioun et al., 2024) to mitigating biases and unwanted behaviors (Price et al.,
33 2024; Lu & Rimsky, 2024). Despite prior success, most activation addition work has been primarily 34 empirical without strong justification behind the usage of these techniques. More theoretically 35 grounded approaches are now emerging. Todd et al. (2024) introduced "function vectors" as specific 36 input-output mappings in activation space, crucial for in-context learning. Park et al. (2024) explored

## 51 **2 A Theoretical Framework For Activation Steering** 52 **2.1 Preliminaries**

$$(1)$$
$\neg$ J 7. 
$\downarrow$ . 
$\Gamma=\lambda$. 
$$\mathbf{H}(s)=\mathbf{enc}(s):\Sigma^{*}\to\mathbb{R}^{D},$$
H(s) = enc(s) : !→ → RD, (2)
63 which is distributed according to:

$$\mathbb{P}(\mathbf{H}=\mathbf{h}\mid C=c)=\mathbb{P}(\mathbf{H}^{-1}(\mathbf{h})\mid C=c)=\sum_{s\in\Sigma^{*}}m_{c}(s)\mathbf{1}\{\mathbf{h}=\mathsf{enc}(s)\}$$
mc(s)1{h = enc(s)} (3)
64 We assume that H is of finite first and second moment, and denote the concept-conditional mean of H with respect to c as µc, the concept-conditional second moment as !˜ 65 c, and the concept-conditional 41 Our work introduces a more general and theoretically grounded framework for activation steering. 42 We derive optimal linear and affine steering functions from first principles in Section 2, connecting 43 our results to conceptor theory (Jaeger, 2014b), to move beyond the limitations of arithmetically 44 combined activation vectors. Our approach employs (soft) projections via steering matrices and 45 optional bias vector translations, further enhanced by a Boolean algebra for principled composition 46 of these steering functions. Our theory is not restricted to binary concepts, and does not require an 47 explicit concept encoding function, as in the work by Singh et al. (2024). We demonstrate that our 48 mechanisms achieve superior performance on function vector tasks (Todd et al., 2024) (Section 3.2) 49 and their Boolean combinations (Section 3.3). Crucially, we also establish improved efficacy over 50 additive vector baselines in complex AI safety-related tasks (Rimsky et al., 2024a) (Section 3.4). Let ! be an alphabet, *i.e.,* a finite and non-empty set. A language model p is a distribution over !→ 53 ,
the set of all strings over the alphabet !. Let ω be a concept-encoding function ω : ! 54 → → C, which 55 maps any given string s to its corresponding concept c = ω(s). Let C be the set of concepts that may be active in the current text sequence s ↑ !→ 56 . These concepts may correspond to functions (Todd 57 et al., 2024), binary concepts (Singh et al., 2024), or other behaviors exhibited by language models.

58 Given a language model m, we define the following conditional distribution:
mc(s) := m(s | C = c) ↓ m(s)1{ω(s) = c}, (1)
which expresses the probability of sampling a string s with concept c present. Let enc : !→ → RD 59 60 be a language encoder, a deterministic function from the set of strings to real-valued vectors. This 61 need not be a specialized module - we use it to denote the hidden activations of an LLM. With a fixed 62 encoder function, we define the following random variable:

$$(2)$$
$$({\mathfrak{I}})$$

37 the Linear Representation Hypothesis, positing that meaningful information is encoded in linear 38 subspaces, providing a theoretical basis for AS. Singh et al. (2024) derived optimal affine steering 39 functions, showing that under "guardedness" constraints, simple additive steering can be optimal, 40 thus justifying existing methods. A more detailed review of related work is given in Appendix B.

66 covariance matrix as !c:

 In $\mu_c$,  $\mu_c=\mathbb{E}[\mathbf{H}_c],\quad\bar{\Sigma}_c=\mathbb{E}[\mathbf{H}_c\mathbf{H}_c^\top],\quad\Sigma_c=\mathbb{E}[\mathbf{H}_c\mathbf{H}_c^\top]-\mu_c\mu_c^\top$ . 
We are interested in *intervention functions* f : RD → RD 67 that map representation-valued random 68 variables to other representation-valued random variables (Singh et al., 2024). We are specifically 69 interested in *steering functions* fc, which are intervention functions that steer a given representation 70 towards some concept c ↑ C.

71 **Definition 1** (ω-assisted steering function). We define a steering function fc to be ω-assisted, and call it fωc 72 *, if it is of the form:*

$$(4)$$
$$f_{c}^{\phi}(\mathbf{H}(s))={\begin{cases}f_{c}(\mathbf{H}(s))&{{\mathrm{if~}}\phi(s)\neq c^{\prime}}\\ \mathbf{H}(s)&{{\mathrm{if~}}\phi(s)=c,}\end{cases}}$$
$$(S)$$
H(s) if ω(s) = c,(5)
where fc : RD → RD is a steering function and ω : ! 73 → → C *is a concept encoding function.*
74 Singh et al. (2024) investigate such ω-assisted steering functions. In the present paper, we instead 75 consider *unassisted steering functions* which do not explicitly make use of a concept encoding 76 function ω when steering the model, following prior work on activation steering (Turner et al., 2023; 77 Li et al., 2023; Subramani et al., 2022). This approach is more computationally efficient since the 78 concept encoding function can be expensive to obtain and evaluate—-for instance, Singh et al. (2024) 79 train a small MLP for this task. Additionally, unassisted steering functions maintain their linear 80 structure throughout the entire input space, rather than becoming piecewise linear with nonlinear 81 decision boundaries (as determined by the concept encoding function). This linearity is particularly 82 valuable for the interpretability of these models, as it allows for clearer analysis of how the steering 83 mechanism affects model behavior.

## 84 **2.2 Additive Steering Functions**

85 Additive steering functions have been the dominant approach to steering model behavior (Turner 86 et al., 2023; Rimsky et al., 2024b; van der Weij et al., 2024).

87 **Definition 2** (additive steering function). We define a function fc *to be an additive steering function* 88 *if it is of the form:*
fc(H(s)) = bc + H(s) (6)
where bc ↑ RD 89 *is the steering vector that corresponds to concept* c.

90 Typically, this additive steering vector is chosen to be bc = µc (see Eq. 4) (Turner et al., 2023). In contrastive activation addition, the steering vector is chosen to be bc = µc ↔ µc 91 ↑ where c is the target concept and c↗ 92 is a contrastive concept that is opposite to c. Singh et al. (2024) have shown that, when 93 "guardedness" is required (see Appendix B), the optimal affine steering method for binary concepts 94 simplifies to contrastive additive steering. We relax this requirement in our theory.

## 95 **2.3 Linear Steering Functions**

96 Let's now consider the class of linear steering functions in which conceptors are found. Linear 97 steering functions map the activations of the model onto their steered counterpart through a linear 98 transformation. This approach is fundamentally different from additive steering, as the change in 99 activation is not restricted to a single direction. Instead, linear transformations can modify activations 100 along multiple directions simultaneously, allowing for more nuanced and context-sensitive steering.

101 A geometric intuition for this distinction is illustrated in Figure 1.

102 **Definition 3** (linear steering function). We define a function fc *to be a linear steering function if it is* 103 *of the form:*
fc(H(s)) = CH(s) (7)
where C ↑ RD↘D 104 *is the steering matrix that corresponds to concept* c.

As such, a linear steering function contains D2 105 parameters and can therefore represent more complex 106 steering functions than an additive steering function, which contains only D parameters.

107 We now wish to define a linear steering function that is "optimal" for steering a representation towards 108 a concept c, in the sense that it should minimize the change to the representation for representations 109 that already exhibit the concept c while still effectively steering all the other representations toward 110 the concept c. We formalize this in the following definition. 111 **Definition 4** (optimal linear steering function). *We define the optimal linear steering function to be* 112 the function fc(H(s)) = CH(s) where C *solves the following optimization problem:*

$$C(\alpha)=\operatorname*{arg\,min}_{C}\mathbb{E}_{c}\left[\|\mathbf{H}_{c}-C\mathbf{H}_{c}\|_{2}^{2}\right]+\alpha^{-2}\|C\|_{F}^{2}$$
$$({\boldsymbol{\delta}})$$
$+ ε↑2↘C↘2F (8)
113 where ↘ · ↘F is the Frobenius norm, and ε *is a regularization parameter, referred to as "aperture".*
114 This optimization problem has been studied by Jaeger (2014b) and has a unique, closed-form solution. 115 The aperture parameter ε balances the trade-off between accurately representing concept-positive 116 activation patterns and maintaining a generalized representation. When ε is large, the eigenvalues 117 µi approach 1 and C approaches the identity matrix, causing the conceptor to allow for more signal 118 components to pass through the conceptor. When ε is small, the eigenvalues µi approach 0, causing 119 the conceptor to allow for less variability and approaching the zero mapping.

Proposition 1. Let !˜ 120 c be the concept-conditional second moment of the random variable H(s) and ε ↑ (0, ≃)*. Then, the conceptor* C(!˜ 121 c, ε) *is uniquely defined and can be directly computed as:*

$$C(\tilde{\Sigma}_{c},\alpha)=\tilde{\Sigma}_{c}\left(\tilde{\Sigma}_{c}+\alpha^{-2}I\right)^{-1}$$
$$(9)$$
&↑1(9)
The matrix C(!˜ 122 c, ε) *is positive semi-definite with eigenvalues in the range* [0, 1).

123 *Proof. See Appendix A.1 and Jaeger (2014b).*

## 131 **2.3.1 Combining Linear Steering Functions With Boolean Operations**

$C_{1}\lor C_{2}=\left(\Sigma_{c_{1}}+\Sigma_{c_{2}}\right)\left(\Sigma_{c_{1}}+\Sigma_{c_{2}}+\alpha^{-2}I\right)^{-1}$  _an be rewritten as:_
(↑1 (10)
141 *Using Equation 9, this can be rewritten as:*

$$C_{1}\lor C_{2}=\left(I+\left(C_{1}(I-C_{1})^{-1}+C_{2}(I-C_{2})^{-1}\right)^{-1}\right)^{-1}$$
$$\neg C=\Sigma_{c}^{-1}(\Sigma_{c}^{-1}+\alpha^{-2}I)^{-1}$$
↑1 (12)
147 *Using Equation 9, this can be rewritten as:*

$$(10)$$
$$\neg C=I-C$$
¬C = I ↔ C (13)
132 We can combine multiple steering matrices using Boolean operations on conceptors, as defined by 133 Jaeger (2014b). These operations allow us to merge conceptors computed on different data samples to 134 construct more complex steering targets. We begin by defining the OR operation on two conceptors, 135 which is computed by summing the covariance matrices on which they are based. This operation can 136 be understood as merging the data from which each conceptor was derived. The resulting conceptor 137 is then computed based on the sum of these covariance matrices.

138 **Definition 5** (OR Operation on Conceptors). Let C1 and C2 be two conceptors computed from covariance matrices !c1 and !c2 139 , respectively. The OR operation, C1 ⇐ C2*, combines these conceptors* 140 *by adding their covariance matrices and is given by:* 142 Next, we define the NOT operation. This operation inverts the covariance matrix, producing a 143 conceptor that captures data that co-varies inversely to the original conceptor. 144 **Definition 6** (NOT Operation on Conceptors). Let C *be a conceptor derived from covariance matrix* 145 !c. The NOT operation on a conceptor, denoted by ¬C*, is computed by inverting the covariance* 146 *matrix. The NOT operation is defined as:* 124 The unique, closed-form solution is known as the conceptor C(ε) - a positive semi-definite matrix 125 with eigenvalues between zero and one. We refer to the application of the conceptor as a "soft 126 projection" of the representation towards the concept c. Where the context is apparent, we drop 127 the function notation and denote the conceptor matrix simply by C. The conceptor matrix C
128 captures the principal directions and variances of a set of neural activation vectors. This structure 129 can be visualized as a high-dimensional ellipsoid that describes the overall shape and spread of the 130 activations' "underlying pattern" or state space region, see Figure 6.

$$(11)$$
$$(12)$$

$$(13)^{\frac{1}{2}}$$

148 From these operations, we can use de Morgan's law to define the AND operation which captures the 149 intersection between two conceptors. The formal definition is given in Appendix C.1. 150 These Boolean operations can be used to combine multiple conceptor steering matrices into *composite* 151 *steering functions*. Similar operations have been proposed for additive steering methods. Todd et al. 152 (2024) propose a task arithmetic on function vectors and demonstrate it on a some toy tasks, while 153 Subramani et al. (2022) use a vector arithmetic on steering vectors. The negation of additive steering 154 vectors has been used widely in contrastive steering as introduced by Rimsky et al. (2024b). We note 155 that the AND and OR operations on conceptor steering matrices do not clearly correspond to the 156 addition operation on steering vectors. In Section 3.3, we compare combinations of steering vectors 157 against combinations of conceptor-based steering matrices.

## 158 **2.4 Affine Steering Functions**

159 We now turn to the class of affine steering functions, in order to generalize the results on conceptors 160 (Jaeger, 2014b), additive steering functions (Turner et al., 2023), and affine steering functions (Singh 161 et al., 2024) into a more general framework of affine activation steering.

162 **Definition 7** (affine steering function). We define a function fc *to be an affine steering function if it is* 163 *of the form:*
fc(H(s)) = CH(s) + b (14)
where C ↑ RD↘D *is the steering matrix, and* b ↑ RD 164 is the steering vector, both of which corre165 *sponding to concept* c. 166 We define the *optimal affine steering function* in an analogous way to how we defined the optimal 167 linear steering function, as the solution to an optimization problem. 168 **Definition 8** (optimal affine steering function). *We define the optimal affine steering function to be* 169 the function fc(H(s*)) =* CH(s) + b *which solves the following optimization problem:*

$\min\limits_{C\in\mathbb{R}^{D\times D},b\in\mathbb{R}^{D}}\mathbb{E}\left[||\mathbf{H}_{c}-(C\mathbf{H}_{c}+b)||_{2}^{2}\right]+\alpha^{-2}||C||_{F}^{2}$
$$(15)$$

$$(16)$$
$$\begin{array}{r}{f_{c}(\mathbf{H}(s))=C x+b=C x+\mu_{c}-C\mu_{c}}\\ {=C(x-\mu_{c})+\mu_{c}}\end{array}$$
$+ ε↑2↘C↘2F (15)
170 In the following proposition, we derive the unique solution for the optimal affine steering function.

171 **Proposition 2.** Let !c be the concept-conditional covariance matrix of H(s), µc its concept172 conditional mean, and ε ↑ (0, ≃). Then, the optimal affine steering function fc*, as defined* 173 *above, can be directly computed as:*

$$\begin{array}{l}{{C(\Sigma_{c},\alpha)=\Sigma_{c}(\Sigma_{c}+2\alpha^{-2}I)^{-1}}}\\ {{b(\Sigma_{c},\alpha)=\mu_{c}-C(\Sigma_{c},\alpha)\mu_{c}}}\end{array}$$
174 Let C := C(!c, ε) and b := b(!c, ε)*, then the final steering function is of the form:*
fc(H(s)) = Cx + b = Cx + µc ↔ Cµc (18)
= C(x ↔ µc) + µc (19)
175 *Proof. See Appendix A.2.*

## 176 **2.5 Residual Steering Functions**

177 In standard conceptor steering, the mapping fc(x) = C x attenuates or preserves each principal 178 component of x by a factor µi ↑ [0, 1]. When we instead apply the conceptor residually, *i.e.,*:
fc(x) = Cx + x = (C + I)x (20)
179 the effective steering matrix becomes C + I and all "steering modes" are shifted to singular values 180 ϑi + 1 ↑ [1, 2]. We argue that this shift has two benefits in LLMs. Firstly, as argued by Elhage et al. (2021), transformers propagate information via additive updates1 181 x ⇒→ x + "(x) and by adding 182 the steered representation, we conform exactly to that inductive bias–injecting the concept signal 183 as an additive perturbation rather than a standalone linear gating. Secondly, original conceptors 184 can only scale down directions (ϑi ⇑ 1), potentially erasing subtle features. In contrast, (I + C)
1This is the case for recurrent and hybrid models, including the ones used in this paper.

$$(18)$$

$\left(19\right)$ . 

$$f_{c}(x)=C x+x=(C+I)x$$
$\mathbf{a}$

## 190 **3 Experiments**

191 We demonstrate the effectiveness of our steering methods on a set of tasks across several models.

## 192 **3.1 Implementing Conceptor Steering**

Given a finite sample Hc ↑ RD↘n 193 of n representations with concept c ↑ C from Hc, we approximate the concept-conditional mean with µˆc = 1 194 nHc1n and the concept-conditional second moment with
!

ˆ˜ c = 1nHcH↔c . From µˆc, and !

ˆ˜ 195 c, we compute linear (Eq. 9), affine (Eq. 19), and compositional (Eq.

196 51) conceptor steering functions.

197 **Steering location** The input of an LLM is a sequence of tokens ti (where i is the token index) which are transformed into embeddings x0i ↑ RD using a learned embedding matrix E ↑ RD↘V 198 where V
is the vocabulary size. At each layer 1 ⇑ ϖ ⇑ L, the input vector sequence xε↑1 t 199 is transformed by the token mixing operation ϱ as x ε,1 t = xε↑1 t + ϱ (xε↑1 t 200 ) and a subsequent channel mixing operation ς as xεt = x ε,1 t + ς(x ε,1 t 201 ). The transformation of a full layer is thus given by

$$x_{t}^{\ell}=x_{t}^{\ell-1}+\tau(x_{t}^{\ell-1})+\zeta(x_{t}^{\ell-1}+\tau(x_{t}^{\ell-1}))$$
$$(21)$$
xεt = xε↑1 t + ϱ (xε↑1 t ) + ς(xε↑1 t + ϱ (xε↑1 t )) (21)

## 213 **3.2 Function Steering**

202 The channel mixing operation ς is typically implemented as a multi-layer perceptron (MLP) or a 203 mixture-of-expert (MoE), and the token mixing operation ϱ is typically implemented as a multi-head 204 attention (MHA) operation or a recurrent neural network (RNN). Both operations typically contain a pre- or post-normalization operation. Following Elhage et al. (2021), we refer to xεt and x ε,1 t 205 as 206 samples from the residual stream. Unless otherwise specified, we steer the activations of the residual stream before the token mixing operation, *i.e.*, we intervene on the variable xεt 207 for 0 ⇑ ϖ < L.

208 **Hyperparameters** We already introduced ε as a hyperparameter for conceptor-based steering. 209 Following prior work, we introduce φ as a hyperparameter for the *steering strength*. For additive steering, this is applied by using an effective bias vector beff 210 c = φbc. For conceptor-based steering, this is applied by using an effective conceptor C 211 eff = φC. For all experiments, we find optimal 212 hyperparameters for each steering method at every layer, see Appendix D.

214 We compare conceptor-based and additive steering mechanisms on their ability to steer a given model 215 toward correctly executing a set of in-context-learning tasks ("functions"). We test both methods on 216 GPT-J with 6B parameters and GPT-NeoX with 20B parameters. For each function, the experiment 217 was repeated five times with random seeds, and all reported results were averaged across these runs.

218 The examples of the input-output functions come from the dataset by Todd et al. (2024). We use 219 the following subset of five functions: antonyms (e.g. good→bad), present-past (e.g. go→went), 220 English-French (e.g. hello→bonjour), singular-plural (e.g. mouse→mice), country-capital (e.g.

221 Netherlands→Amsterdam), and capitalize (e.g. word→Word). To ensure comparability of our results, 222 we follow the work by Todd et al. (2024) as closely as possible. For more details, see Appendix D.1. 223 The results in Figure 2 show that conceptor-based steering outperforms the additive steering baseline 224 (Todd et al., 2024) for every task on both tested models. Results show the best-performing model 225 across a range of hyperparameters. Conceptor steering is strictly more performant than additive 226 steering across all tasks for most layers. Results for the complete hyperparameter sweep are presented 227 in Appendix D.5. In line with previous findings (Todd et al., 2024; Jorgensen et al., 2023a), steering 228 is most effective across layers 9-16 for GPT-J and layers 10-30 for GPT-NeoX.

2As in activation addition, the norm of the vectors is normalized by the succeeding layernorm.

185 preserves every component (smallest gain ⇓ 1) and gently amplifies concept-relevant modes (largest gain ⇑ 2), strengthening signals without discarding baseline information2 186 . Taken together, residual 187 conceptor application both respects the architectural biases of LLMs and leverages mild, controlled 188 amplification of concept-specific subspaces—likely explaining the empirical improvements observed 189 when steering via C + I rather than C alone. 229 As illustrated in Figure 1, additive and conceptor steering correspond to 230 different interventions onto the model activations. To compare conceptor 231 steering to another linear steering function that would have equivalent 232 expressivity, we also train full rank LoRA adapters at the same position 233 as the steering interventions. For each task, we select the best layer 234 for conceptor steering and train until convergence. The performance 235 averaged across all tasks is shown in Figure 3. Despite the adapters using 236 at least 10⇔ more compute than the conceptor, they do not outperform 237 their competitor. For more details, see Appendix D.2. 238 We also present results for affine conceptors in Table 1, as derived in 239 Section 2.4. We compare affine conceptors against linear conceptors, and 240 also relate these results against a similar operation on additive steering 241 called "mean-centering" (Jorgensen et al., 2023b). Mean-centering im242 proves the performance of additive steering by as much as 2⇔ on the 243 country-capital task. Analogously, affine conceptors improved steering 244 accuracy on some of the tasks, but the relative improvement was limited to no more than 5% in 245 accuracy. For more details, see Appendix D.3.

Figure 3: Performance of custom LoRA adapters compared against steering functions.

| the best performance across all hyperparameters and across all layers. antonyms capitalize country-capital english-french   | present-past   |        |        |        |        |
|-----------------------------------------------------------------------------------------------------------------------------|----------------|--------|--------|--------|--------|
| Addition                                                                                                                    | 20.54%         | 93.16% | 32.04% | 18.88% | 69.66% |
| Addition (MC)                                                                                                               | 31.20%         | 95.00% | 63.90% | 34.32% | 83.32% |
| Linear conceptor                                                                                                            | 52.14%         | 96.68% | 81.62% | 59.02% | 91.56% |
| Affine conceptor                                                                                                            | 52.82%         | 96.26% | 85.32% | 61.32% | 91.88% |

## 246 **3.3 Steering Composite Functions**

247 To further investigate whether the boolean operators of conceptors can be leveraged for steering 248 composite functions, we created three novel compound input-output functions: English-French &
249 atonyms (e.g. good→mauvais), English-French & capitalize (e.g. good→Bon), singular-plural & 250 capitalize (e.g. mouse→Mice). This additinal dataset was generated using GPT-4o and will be made 251 available for the camera-ready paper, for additional details on the experiment see Appendix D.4.

To establish a baseline, we show performance of the conceptor C1,2 and the steering vector ¯h1,2 ε 252 253 computed directly from the example activations of the compound function. We then combine the conceptors computed on the individual functions C1 and C2 254 using the AND operation as C1 ↖ C2, and we combine the steering vectors ¯h1ε and ¯h2ε using their arithmetic mean 12 ( ¯h1ε + ¯h2ε 255 ).

256 Figure 4 shows the performance of all methods across all layers of the GPT-J model. In line with 257 results from Section 3.2, the conceptor baseline outperformed the additive baseline on all tasks. 258 The AND-combined con259 ceptor outperforms both 260 the mean-combined steer261 ing vectors and the addi262 tive baseline, in all tasks, 263 suggesting that the compo264 sitional operators of con265 ceptors align more naturally 266 with language composition267 ality than simple vector ad268 dition.

## 269 **3.4 Steering Complex Behaviors**

270 To further evaluate our steering frameworks, we investigate their performance on a complex, safety271 relevant behavioral task: the "Coordinate with other AIs" task from Perez et al. (2022). In this task, 272 the model decides whether to coordinate with another AI, potentially diverging from human interests. 273 For this specific evaluation, positive examples are instances where the model's activations correspond 274 to outputs agreeing to coordinate, while negative examples represent refusals. 275 The steering mechanisms were computed as follows: The standard Conceptor was derived using 276 activations solely from these positive examples, following the formulation in Proposition 1. The 277 Contrastive Conceptor leveraged the Boolean algebra for conceptors detailed earlier (Section 2), for 278 instance by combining a conceptor representing positive examples with the negation of a conceptor 279 representing negative examples. The additive steering baseline, Contrastive Vector, was calculated 280 as the mean difference between activations from the positive and negative example sets following 281 previous work (Rimsky et al., 2024b).

282 We selected two distinct model architectures for this evaluation. The Qwen 2.5-1.5B Instruct 283 model (Qwen et al., 2025), a transformer-based LLM, was chosen for its wide adoption and strong 284 performance. The Mamba 2.8B model Gu & Dao (2024), a recurrent state space model (SSM), was 285 included to investigate the steering performance on LLMs that are not based on the transformer 286 architecture.

287 Figure 5a suggests that conceptor-based methods can outperform the contrastive vector method in 288 controlling complex behavior on the multiple-choice "Coordinate with other AIs" task. More results 289 and details for closed-ended datasets, including the one shown here, can be found in D.6. Furthermore, 290 although we anticipate that this enhanced control will coincide with enhanced qualitative display of 291 the target behavior as measured by an LLM judge, open-ended steering proves more challenging and 292 underperforms vector steering for the specific layer chosen (Figure 5b). We attribute the discrepancy 293 between the MCQ and open-ended results to the more sensitive search space for open-ended steering, 294 which we'll explore more exhaustively in the camera-ready version of the paper, as our current 295 hyperparameter search was coarse and limited to a <50% subset of the model's layers. Should 296 conceptor-steered open generation match the performance of A/B question answering, our conceptor297 based framework would advance the Pareto frontier of activation steering, offering more focused and 298 potent behavioral modulation while preserving core model competencies. More relevant results and 299 details can be found in D.6, and for more details on the analysis of conceptors, see section E
300 The anticipated efficacy of these methods is informed by recent work. Braun et al. (2025) highlight 301 that the reliability of steering vectors is strongly conditional on the geometric separability of the 302 target concept's positive and negative examples in activation space. This implies that if a concept is 303 not clearly distinguishable, steering attempts may be ineffective or unpredictable. This aligns with the 304 theoretical underpinnings of conceptors, which, by capturing richer geometric information, may offer 305 more robust steering, particularly for concepts not perfectly represented by simple linear directions.

## 306 **4 Conclusion**

307 The integration of conceptor theory with AS provides a new lens for understanding and manipulating 308 LLMs. By deriving optimal steering functions from first principles, we establish a rigorous theoretical 309 foundation for conceptor steering. Where additive steering applies a uniform translation on all neural 310 activations, conceptors enable linear transformation over activations while maintaining a reasonable 311 computational cost compared to its LoRA counterpart. In addition, the design of conceptors enables 312 them to capture the covariance structure of neural activations, allowing them to encode richer hidden 313 state representations, beyond average activation patterns. Notably, conceptor-steering, is inherently 314 adaptive without requiring an additional mechanism as the one proposed by Wang et al. (2024). 315 This adaptivity occurs naturally because activations already residing within the conceptor's region 316 experience minimal change, whereas activations outside this region undergo more substantial shifts. 317 Additionally, the compositional nature of conceptor operations, implemented through Boolean algebra, 318 offers a powerful mechanism for multi-task steering. By combining conceptors using operations 319 like AND and OR, we are able to create composite steering objectives that outperform traditional 320 methods of combining steering vectors. This demonstrates the versatility of our approach, allowing 321 for more sophisticated control of LLMs, especially in multi-task scenarios where steering objectives 322 may conflict or overlap. 323 While our theoretical and empirical results establish conceptor-based steering as a powerful and 324 versatile AS technique, the scope of our claims is confined to the model families (transformers 325 and recurrent SSMs) and tasks evaluated; extension to larger architectures, long-range dialogue, or 326 multilingual settings may reveal additional challenges. While introducing additional complexity 327 (requiring covariance matrix computation and more hyperparameter tuning) compared to simpler 328 additive methods, conceptor steering's trade-offs are justified by gains in precision, especially 329 where additive steering is insufficient. As highlighted by Krasheninnikov & Krueger (2024), it is 330 important to consider that more highly parameterized steering methods—such as conceptors with D2 331 parameters—may require more data to perform optimally compared to simpler additive vector 332 approaches with only D parameters. Importantly, conceptor steering does not by itself guarantee 333 fairness: latent biases present in training corpora can persist or even be accentuated within projected 334 subspaces, so rigorous fairness audits across demographic and linguistic groups are essential. From 335 a safety and ethics standpoint, the ability to suppress or amplify behaviours via conceptors offers 336 both promise (e.g., reducing toxic or misleading outputs) and risk (e.g., covertly enabling adversarial 337 manipulation). Thorough evaluation under adversarial conditions, alongside quantitative safety 338 benchmarks, will be critical to assess dual-use implications before real-world deployment. 339 Our work unites conceptor theory and AS, offering a robust framework for both controlling and 340 understanding LLMs. By deriving a provably optimal affine steering mechanism and introducing 341 composable Boolean operations, we provide a method that not only surpasses traditional steering 342 approaches but also lays the groundwork for more advanced activation engineering techniques. While 343 challenges remain, the combination of theoretical rigor and empirical success positions conceptor344 based steering as a powerful tool for the future of LLM control and interpretability.

## 345 **References**

346 Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané. 347 Concrete problems in AI safety. *arXiv*, abs/1606.06565, 2016. URL https://arxiv.org/abs/ 348 1606.06565. 349 Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff, and Stella 350 Biderman. LEACE: Perfect linear concept erasure in closed form. November 2023. URL
351 https://openreview.net/forum?id=awIpKpwTwF&noteId=Ju4XcafMir.

352 Léon Bottou, Frank E. Curtis, and Jorge Nocedal. Optimization methods for large-scale machine 353 learning. *arXiv*, abs/1606.04838, 2018. URL https://arxiv.org/abs/1606.04838. 354 Joschka Braun, Carsten Eickhoff, David Krueger, Seyed Ali Bahrainian, and Dmitrii Krasheninnikov. 355 Understanding (un)reliability of steering vectors in language models. In *ICLR 2025 Workshop on* 356 *Building Trust in Language Models and Applications*, 2025. URL https://openreview.net/ 357 forum?id=JZiKuvIK1t. 358 Paul Bricman. Nested state clouds: Distilling knowledge graphs from contextual embeddings.

359 Bachelor's Project Thesis, University of Groningen, Supervisors: Prof. Dr. Herbert Jaeger, Dr.

360 Jacolien van Rij-Tange, July 2022. URL https://fse.studenttheses.ub.rug.nl/27840/.

361 Yuanpu Cao, Tianrong Zhang, Bochuan Cao, Ziyi Yin, Lu Lin, Fenglong Ma, and Jinghui Chen. Per362 sonalized Steering of Large Language Models: Versatile Steering Vectors Through Bi-directional 363 Preference Optimization. *CoRR*, January 2024. URL https://openreview.net/forum?id= 364 MJgVF5HCRr. 365 Banghao Chen, Zhaofeng Zhang, Nicolas Langren'e, and Shengxin Zhu. Unleashing the potential of 366 prompt engineering in large language models: a comprehensive review. *ArXiv*, abs/2310.14735, 367 2023. doi: 10.48550/arXiv.2310.14735. 368 Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of 369 deep bidirectional transformers for language understanding. In Jill Burstein, Christy Doran, and 370 Thamar Solorio (eds.), *Proceedings of the 2019 Conference of the North American Chapter of the* 371 *Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and* 372 *Short Papers)*, pp. 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational 373 Linguistics. doi: 10.18653/v1/N19-1423. URL https://aclanthology.org/N19-1423.

374 Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda 375 Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, 376 Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal 377 Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris 378 Olah. A mathematical framework for transformer circuits. *Transformer Circuits Thread*, 2021.

379 https://transformer-circuits.pub/2021/framework/index.html. 380 Isabel O. Gallegos, Ryan A. Rossi, Joe Barrow, Md Mehrab Tanjim, Sungchul Kim, Franck Dernon381 court, Tong Yu, Ruiyi Zhang, and Nesreen K. Ahmed. Bias and fairness in large language models: 382 A survey. *arXiv*, abs/2309.00770, 2024. URL https://arxiv.org/abs/2309.00770.

383 Asma Ghandeharioun, Ann Yuan, Marius Guerard, Emily Reif, Michael A. Lepori, and Lucas Dixon.

384 Who's asking? User personas and the mechanics of latent misalignment, August 2024. URL
385 http://arxiv.org/abs/2406.12094. arXiv:2406.12094 [cs]. 386 Aaron Gokaslan, Vanya Cohen, Ellie Pavlick, and Stefanie Tellex. Openwebtext corpus. http:
387 //Skylion007.github.io/OpenWebTextCorpus, 2019. 388 Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces, 2024.

389 URL https://arxiv.org/abs/2312.00752.

390 Owen He. *Continual lifelong learning in neural systems: overcoming catastrophic forgetting and* 391 *transferring knowledge for future learning*. PhD thesis, University of Groningen, 2023.

392 Herbert Jaeger. Conceptors: an easy introduction. *arXiv*, abs/1406.2671, 2014a. URL https: 393 //arxiv.org/abs/1406.2671.

394 Herbert Jaeger. Controlling Recurrent Neural Networks by Conceptors. March 2014b. _eprint: 395 1403.3369. 396 Herbert Jaeger. Controlling recurrent neural networks by conceptors. *arXiv*, abs/1403.3369, 2017. 397 URL https://arxiv.org/abs/1403.3369.

398 Ole Jorgensen, Dylan Cope, Nandi Schoots, and Murray Shanahan. Improving activation steering 399 in language models with mean-centring. *arXiv*, abs/2312.03813, 2023a. URL https://arxiv. 400 org/abs/2312.03813.

401 Ole Jorgensen, Dylan Cope, Nandi Schoots, and Murray Shanahan. Improving Activation Steering in 402 Language Models with Mean-Centring, December 2023b. URL http://arxiv.org/abs/2312. 403 03813. arXiv:2312.03813 [cs]. 404 Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, 405 Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models, 406 2020. URL https://arxiv.org/abs/2001.08361.

407 Dmitrii Krasheninnikov and David Krueger. Steering clear: A systematic study of activation steering 408 in a toy setup. In *MINT: Foundation Model Interventions*, 2024. URL https://openreview. 409 net/forum?id=ygvbAGTgzA.

410 Jesper Kuiper. Using conceptors to extract abstraction hierarchies from corpora of natural text: 411 Combatting word polysemy using word sense disambiguation techniques. Master's thesis / essay, 412 University of Groningen, Groningen, Netherlands, January 2024. 413 Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. Inference414 Time Intervention: Eliciting Truthful Answers from a Language Model. November 2023. URL
415 https://openreview.net/forum?id=aLLuYpn83y.

416 Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig.

417 Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language 418 processing. *ACM Comput. Surv.*, 55(9), jan 2023. ISSN 0360-0300. doi: 10.1145/3560815. URL
419 https://doi.org/10.1145/3560815.

420 Dawn Lu and Nina Rimsky. Investigating Bias Representations in Llama 2 Chat via Activation 421 Steering, February 2024. URL http://arxiv.org/abs/2402.00402. arXiv:2402.00402 [cs].

422 Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong 423 Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, 424 Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and 425 Ryan Lowe. Training language models to follow instructions with human feedback. In *Proceedings* 426 *of the 36th International Conference on Neural Information Processing Systems*, NIPS '22, Red 427 Hook, NY, USA, 2024. Curran Associates Inc. ISBN 9781713871088. 428 Yikang Pan, Liangming Pan, Wenhu Chen, Preslav Nakov, Min-Yen Kan, and William Yang Wang. 429 On the risk of misinformation pollution with large language models. In *The 2023 Conference on* 430 *Empirical Methods in Natural Language Processing*, 2023. URL https://openreview.net/ 431 forum?id=voBhcwDyPt. 432 Kiho Park, Yo Joong Choe, and Victor Veitch. The Linear Representation Hypoth433 esis and the Geometry of Large Language Models. June 2024. URL https:
434 //openreview.net/forum?id=UGpGkLzwpP&referrer=%5Bthe%20profile%20of%20Yo%
435 20Joong%20Choe%5D(%2Fprofile%3Fid%3D~Yo_Joong_Choe1).

436 Ethan Perez, Sam Ringer, Kamile Luko ˙ !iut¯ e, Karina Nguyen, Edwin Chen, Scott Heiner, Craig ˙ 437 Pettit, Catherine Olsson, Sandipan Kundu, Saurav Kadavath, Andy Jones, Anna Chen, Ben Mann, 438 Brian Israel, Bryan Seethor, Cameron McKinnon, Christopher Olah, Da Yan, Daniela Amodei, 439 Dario Amodei, Dawn Drain, Dustin Li, Eli Tran-Johnson, Guro Khundadze, Jackson Kernion, 440 James Landis, Jamie Kerr, Jared Mueller, Jeeyoon Hyun, Joshua Landau, Kamal Ndousse, Landon 441 Goldberg, Liane Lovitt, Martin Lucas, Michael Sellitto, Miranda Zhang, Neerav Kingsland, Nelson 442 Elhage, Nicholas Joseph, Noemí Mercado, Nova DasSarma, Oliver Rausch, Robin Larson, Sam 443 McCandlish, Scott Johnston, Shauna Kravec, Sheer El Showk, Tamera Lanham, Timothy Telleen444 Lawton, Tom Brown, Tom Henighan, Tristan Hume, Yuntao Bai, Zac Hatfield-Dodds, Jack Clark, 445 Samuel R. Bowman, Amanda Askell, Roger Grosse, Danny Hernandez, Deep Ganguli, Evan 446 Hubinger, Nicholas Schiefer, and Jared Kaplan. Discovering language model behaviors with 447 model-written evaluations, 2022. URL https://arxiv.org/abs/2212.09251.

448 Sara Price, Arjun Panickssery, Sam Bowman, and Asa Cooper Stickland. Future Events as Backdoor 449 Triggers: Investigating Temporal Vulnerabilities in LLMs, July 2024. URL http://arxiv.org/ 450 abs/2407.04108. arXiv:2407.04108 [cs].

451 Chen Qian, Jie Zhang, Wei Yao, Dongrui Liu, Zhenfei Yin, Yu Qiao, Yong Liu, and Jing Shao. 452 Towards Tracing Trustworthiness Dynamics: Revisiting Pre-training Period of Large Language 453 Models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), *Findings of the Association* 454 *for Computational Linguistics ACL 2024*, pp. 4864–4888, Bangkok, Thailand and virtual meeting, 455 August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.290. 456 URL https://aclanthology.org/2024.findings-acl.290. 457 Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan 458 Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, 459 Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin 460 Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi 461 Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, 462 Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL
463 https://arxiv.org/abs/2412.15115.

464 Nate Rahn, Pierluca D'Oro, and Marc G. Bellemare. Controlling Large Language Model Agents 465 with Entropic Activation Steering. June 2024. URL https://openreview.net/forum?id= 466 3eBdq2n848.

467 Shauli Ravfogel, Michael Twiton, Yoav Goldberg, and Ryan D. Cotterell. Linear Adversarial 468 Concept Erasure. In Proceedings of the 39th International Conference on Machine Learn469 ing, pp. 18400–18421. PMLR, June 2022. URL https://proceedings.mlr.press/v162/ 470 ravfogel22a.html. ISSN: 2640-3498.

471 Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. Steer472 ing llama 2 via contrastive activation addition. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar 473 (eds.), *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics* 474 *(Volume 1: Long Papers)*, pp. 15504–15522, Bangkok, Thailand, August 2024a. Association for 475 Computational Linguistics. URL https://aclanthology.org/2024.acl-long.828.

476 Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. 477 Steering Llama 2 via Contrastive Activation Addition. In Lun-Wei Ku, Andre Martins, and Vivek 478 Srikumar (eds.), *Proceedings of the 62nd Annual Meeting of the Association for Computational* 479 *Linguistics (Volume 1: Long Papers)*, pp. 15504–15522, Bangkok, Thailand, August 2024b. 480 Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.828. URL https: 481 //aclanthology.org/2024.acl-long.828. 482 Toby Shevlane, Sebastian Farquhar, Ben Garfinkel, Mary Phuong, Jess Whittlestone, Jade Leung, 483 Daniel Kokotajlo, Nahema Marchal, Markus Anderljung, Noam Kolt, Lewis Ho, Divya Siddarth, 484 Shahar Avin, Will Hawkins, Been Kim, Iason Gabriel, Vijay Bolina, Jack Clark, Yoshua Bengio, 485 Paul Christiano, and Allan Dafoe. Model evaluation for extreme risks, 2023. URL https: 486 //arxiv.org/abs/2305.15324.

487 Shashwat Singh, Shauli Ravfogel, Jonathan Herzig, Roee Aharoni, Ryan Cotterell, and Ponnurangam 488 Kumaraguru. Representation Surgery: Theory and Practice of Affine Steering. In *Proceedings* 489 *of the 41st International Conference on Machine Learning*, pp. 45663–45680. PMLR, July 2024.

490 URL https://proceedings.mlr.press/v235/singh24d.html. ISSN: 2640-3498. 491 Asa Cooper Stickland, Alexander Lyzhov, Jacob Pfau, Salsabila Mahdi, and Samuel R. Bowman.

492 Steering Without Side Effects: Improving Post-Deployment Control of Language Models, June 493 2024. URL http://arxiv.org/abs/2406.15518. arXiv:2406.15518 [cs].

494 Nishant Subramani, Nivedita Suresh, and Matthew Peters. Extracting Latent Steering Vectors from 495 Pretrained Language Models. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), 496 *Findings of the Association for Computational Linguistics: ACL 2022*, pp. 566–581, Dublin, Ireland, 497 May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-acl.48.

498 URL https://aclanthology.org/2022.findings-acl.48.

499 Daniel Chee Hian Tan, David Chanin, Aengus Lynch, Adrià Garriga-Alonso, Dimitrios Kanoulas, 500 Brooks Paige, and Robert Kirk. Analyzing the Generalization and Reliability of Steering Vectors. 501 June 2024. URL https://openreview.net/forum?id=akCsMk4dDL.

502 Eric Todd, Millicent Li, Arnab Sen Sharma, Aaron Mueller, Byron C Wallace, and David Bau.

503 Function vectors in large language models. In *The Twelfth International Conference on Learning* 504 *Representations*, 2024. URL https://openreview.net/forum?id=AwyxtyMwaG.

505 Alexander Matt Turner, Lisa Thiergart, David Udell, Gavin Leech, Ulisse Mini, and Monte MacDi506 armid. Activation Addition: Steering Language Models Without Optimization. August 2023. doi: 507 10.48550/ARXIV.2308.10248. Publisher: arXiv _eprint: 2308.10248. 508 Teun van der Weij, Massimo Poesio, and Nandi Schoots. Extending Activation Steering to Broad 509 Skills and Multiple Behaviours, March 2024. URL http://arxiv.org/abs/2403.05767. 510 arXiv:2403.05767 [cs]. 511 Haoran Wang and Kai Shu. Trojan Activation Attack: Red-Teaming Large Language Models using 512 Activation Steering for Safety-Alignment, August 2024. URL http://arxiv.org/abs/2311. 513 09433. arXiv:2311.09433 [cs].

514 Tianlong Wang, Xianfeng Jiao, Yifan He, Zhongzhi Chen, Yinghao Zhu, Xu Chu, Junyi Gao, 515 Yasha Wang, and Liantao Ma. Adaptive Activation Steering: A Tuning-Free LLM Truthfulness 516 Improvement Method for Diverse Hallucinations Categories. *CoRR*, January 2024. URL https: 517 //openreview.net/forum?id=OAPmI3Y1Al.

518 Bo Xu and M. Poo. Large language models and brain-inspired general intelligence. *National Science* 519 *Review*, 10, 2023. doi: 10.1093/nsr/nwad267. 520 Li S. Yifei, Lyle Ungar, and João Sedoc. Conceptor-aided debiasing of large language models.

521 In *The 2023 Conference on Empirical Methods in Natural Language Processing*, 2023. URL
522 https://openreview.net/forum?id=M6BJfQ9oup.

## 870 **Neurips Paper Checklist**

871 1. **Claims**

872 Question: Do the main claims made in the abstract and introduction accurately reflect the 873 paper's contributions and scope? 874 Answer: [Yes] 875 Justification: The paper's claims in the introduction accurately reflect the contributions, 876 namely: introducing a general framework for activation steering, proposing conceptor877 based steering for LLMs, showing its superior performance on function vector tasks, and 878 demonstrating how Boolean operations on conceptors can combine functions, and good 879 performance on other alignment-relevant benchmarks.

880 2. **Limitations**

881 Question: Does the paper discuss the limitations of the work performed by the authors? 882 Answer: [Yes] 883 Justification: a detailed discussion of the limitations is provided in the discussion section of 884 the paper with our assumptions, scope of the claims, computational efficiency, and fairness. 885 3. **Theory assumptions and proofs** 886 Question: For each theoretical result, does the paper provide the full set of assumptions and 887 a complete (and correct) proof? 888 Answer: [Yes] 889 Justification: The paper provides theoretical results with clear assumptions and complete 890 proofs. For instance, the optimal linear and affine steering functions are formally defined 891 with their optimization objectives, and Proposition 1 for the conceptor matrix and Proposition 892 2 for the optimal affine steering function are stated with reference to proofs (in the appendix).

893 4. **Experimental result reproducibility**

894 Question: Does the paper fully disclose all the information needed to reproduce the main ex895 perimental results of the paper to the extent that it affects the main claims and/or conclusions 896 of the paper (regardless of whether the code and data are provided or not)? 897 Answer: [Yes] 898 Justification: The paper provides details of the experimental setup, including model specifi899 cations (GPT-J 6B, GPT-NeoX 20B, Mamba 2.8B, Qwen 3B), datasets used, hyperparameter 900 search procedures, and specific implementation details for the steering methods. The authors 901 reference previous works they follow and mention that additional details are in the appendix.

902 5. **Open access to data and code**

903 Question: Does the paper provide open access to the data and code, with sufficient instruc904 tions to faithfully reproduce the main experimental results, as described in supplemental 905 material? 906 Answer: [Yes] 907 Justification: all code and data will be made available on GitHub for the camera-ready 908 version of the paper. A core contribution of the paper is a flexible and minimalistic Python 909 package for steering LLMs, which will be made available for the camera-ready submission.

910 6. **Experimental setting/details**

911 Question: Does the paper specify all the training and test details (e.g., data splits, hyper912 parameters, how they were chosen, type of optimizer, etc.) necessary to understand the 913 results? 914 Answer: [Yes] 915 Justification: The paper specifies the models used (GPT-J 6B, GPT-NeoX 20B, GPT-2 916 Small), the tasks tested, and mentions that optimal hyperparameters were found for each

917 steering method at every layer with details of the grid search in the appendix. The paper

918 also describes the implementation of conceptor-based steering in Equations 8-9. Moreover, 919 the code (including all scripts for the experiments) will be made available on GitHub for the

920 camera-ready submission.

922 Question: Does the paper report error bars suitably and correctly defined or other appropriate 923 information about the statistical significance of the experiments? 924 Answer: [Yes] 925 Justification: The paper states that each experiment was repeated N times with different 926 random seeds (where N is specified in the appendix, typically N = 3 or N = 5), and the 927 reported results are averaged across these runs. Experiments in Section 3.4 were not repeated 928 multiple times but proper error bars will be included in extended runs in the camera-ready 929 version of the paper.

930 8. **Experiments compute resources** 931 Question: For each experiment, does the paper provide sufficient information on the com932 puter resources (type of compute workers, memory, time of execution) needed to reproduce 933 the experiments? 934 Answer: [Yes] 935 Justification: The paper includes information about the computational resources used for 936 running experiments with different models, including hardware specifications, memory 937 requirements, and approximate execution times.

938 9. **Code of ethics** 939 Question: Does the research conducted in the paper conform, in every respect, with the 940 NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

941 Answer: [Yes] 942 Justification: The research focuses on improving methods for controlling language model 943 behavior, which aligns with the NeurIPS Code of Ethics' emphasis on reliable and con944 trollable AI systems. The paper works with pre-trained open-source models and publicly 945 available datasets, with no apparent ethical concerns. 946 10. **Broader impacts** 947 Question: Does the paper discuss both potential positive societal impacts and negative 948 societal impacts of the work performed? 949 Answer: [Yes] 950 Justification: our paper includes a discussion of broader impacts and how steering methods 951 could help with reducing harmful behavior in LLMs, while also potentially being misused to 952 manipulate model outputs in harmful ways. However, the proposed steering mechanism is 953 open and transparent, allowing for auditability and oversight, and we believe that this trans954 parency fosters collaborative oversight, making covert misuse more difficult and enabling 955 the community to detect and correct issues early. 956 11. **Safeguards** 957 Question: Does the paper describe safeguards that have been put in place for responsible 958 release of data or models that have a high risk for misuse (e.g., pretrained language models, 959 image generators, or scraped datasets)? 960 Answer: [NA]
961 Justification: The paper does not release any data or models. It proposes a method for 962 steering existing models, working with publicly available models and datasets.

963 12. **Licenses for existing assets** 964 Question: Are the creators or original owners of assets (e.g., code, data, models), used in 965 the paper, properly credited and are the license and terms of use explicitly mentioned and 966 properly respected? 967 Answer: [Yes] 968 Justification: the original owners of all assets are properly credited and the license are 969 properly respected.

970 13. **New assets**

## 921 7. **Experiment Statistical Significance**

| 971   | Question: Are new assets introduced in the paper well documented and is the documentation         |
|-------|---------------------------------------------------------------------------------------------------|
| 972   | provided alongside the assets?                                                                    |
| 974   | Justification: No assets are introduced in the paper. All artefacts are pre-existing or generated |
| 975   | using pre-trained models and easy to reproduce (see reproducibility section).                     |
| 976   | 14. Crowdsourcing and research with human subjects                                                |
| 977   | Question: For crowdsourcing experiments and research with human subjects, does the paper          |
| 978   | include the full text of instructions given to participants and screenshots, if applicable, as    |
| 979   | well as details about compensation (if any)?                                                      |
| 981   | Justification: The paper does not involve crowdsourcing or research with human subjects.          |
| 982   | All experiments are conducted with language models and pre-existing or programmatically           |
| 983   | generated datasets.                                                                               |
| 984   | 15. Institutional review board (IRB) approvals or equivalent for research with human              |
| 986   | Question: Does the paper describe potential risks incurred by study participants, whether         |
| 987   | such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)           |
| 988   | approvals (or an equivalent approval/review based on the requirements of your country or          |
| 989   | institution) were obtained?                                                                       |
| 991   | Justification: The paper does not involve research with human subjects, so IRB approval           |
| 992   | was not required.                                                                                 |
| 993   | 16. Declaration of LLM usage                                                                      |
| 994   | Question: Does the paper describe the usage of LLMs if it is an important, original, or           |
| 995   | non-standard component of the core methods in this research? Note that if the LLM is used         |
| 996   | only for writing, editing, or formatting purposes and does not impact the core methodology,       |
| 997   | scientific rigorousness, or originality of the research, declaration is not required.             |