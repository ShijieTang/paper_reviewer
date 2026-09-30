# Shift Is Good: Mismatched Data Mixing Improves Test Performance

Anonymous Author(s)
Affiliation Address email

## Abstract 7 **1 Introduction**

15 The answer depends on the specific learning curve for improvement in test performance within a 16 topic as a function of the number of training examples from that topic. But at least for a generic 1/n 17 scaling (as obtained from e.g., both learning VC classes and in parametric regression), the answer, 18 as we will see in Section 3, is that you would benefit from a distribution shift, and should study 19 75% European History and 25% Chinese history—this would reduce your test error by 20% over the 20 90/10 non-shifted training. 21 We just saw an example of what we term **Positive Distribution Shift**: Even if we have unlimited data 22 from the target test distribution Dtest, training on a shifted distribution Dtrain →= Dtest can actually 23 *improve* test performance. This contrasts the typical study of *distribution shift*, i.e. training on one 24 distribution but then applying the predictor, or testing, on another. Typically, it is implicitly assumed 25 that the ideal case would be to train on the test distribution, that training on a different distribution 26 is a compromise, either because we don't know or have access to the true Dtest, or it's expensive 27 to sample from it, or we have only a limited number of samples and want to supplement them with 28 additional data from related distributions. Distribution shift is usually studied as "how much worse 29 do things get if we train on Dtrain →= Dtest", with answers of the form "if Dtrain is close or related 30 enough to Dtest, then it's not much worse". In this paper, we investigate one of several ways in which 31 distribution shift can be *positive*. 32 Specifically, we systematically study the benefit of such distribution shift when training with mis33 matched mixing proportions relative to the test distribution. We model the test distribution as a mixture of K components, with known mixing proportions {pk}Kk=1 34 , and consider training distributions which are mixtures over the same components but with different mixing proportions {qk}Kk=1 35 .

1 We consider training and testing on mixture distributions with different training 2 and test proportions. We show that in many settings, and in some sense generi3 cally, distribution shift can be beneficial, and test performance can improve due 4 to mismatched training proportions. In a variety of scenarios, we identify the 5 optimal training proportions and the extent to which such distribution shift can be 6 beneficial. 8 Imagine that you are taking a high-stakes exam next week. The exam will be 90% on European 9 history and 10% on Chinese history. Both topics are equally familiar to you and equally difficult, and 10 additional study will help you with each topic similarly. You have unlimited access to study material 11 and practice questions for both. How should you spend your limited studying budget? Should your 12 training match your test distribution, studying 90% European and 10% Chinese? Or would you 13 benefit from a distribution shift? Studying more Chinese history? Less? Only European history? We 14 *encourage the reader to pause and make an intuitive guess.*

## 56 **2 Setup**

57 **Learning Setup and Loss** For concreteness, let ε(h, z) be the loss function that describes how 58 well a model h performs on and instance z ↔ Z. For example, in supervised learning, z can be 59 an input-output pair (x, y), and ε(h, z) can be the prediction error of h(x) vs y. Or, in next-word 60 prediction, z can be a document and ε(h, z) can be the average cross-entropy loss when using h to 61 predict each of the next tokens in the document. In any case, for a test distribution Dtest over z, we evaluate the model through the *test loss* LDtest (h) := Ez↑Dtest 62 [ε(h, z)].

63 **Test Distribution.** We consider test distributions consisting of a mixture of K components D1*,...,* DK. A mixture Dp = !k 64 pkDk is then specified by mixing proportions p =
36 We can either think of this as providing guidance when we can actively control mixing between 37 different known components, or as helping us understand how and why a mismatched training 38 distribution can actually be beneficial. In Section 5 we discuss how the analysis is also applicable to a 39 setting where we are not testing on a mixture, but rather on compositional tasks, requiring composing 40 multiple skills, and the skills appear with differing frequencies—this compositional setting served as 41 a major motivation for our study. 42 We consider different per-component learning curves, capturing different error decays, differing 43 hardness among the components, and the possibility of transfer between components. In Section 3 we 44 consider power law error decay, both the 1/n decay mentioned earlier and more general power laws, 45 including with differing component hardnesses or error decays. In Section 4 we consider learning 46 curves corresponding to "fact memorization" scenarios (discussed in Section 4), including those 47 applicable to the skill composition setting, and which correspond to coupon-collector type learning 48 curves. In Section 6 we consider the possibility of transfer between components. In all of these, 49 we show that a mismatched training distribution can be beneficial, characterize the optimal training 50 mixture, and the extent to which mismatch can improve test performance and reduce the training 51 complexity. 52 Beyond all the specific scenarios, we then argue, in Section 7, that benefiting from mismatch is 53 not the exception but rather the rule. We show that only in rare situations (either measure zero or 54 satisfying a conservation property that does not generally hold) is the optimal training distribution 55 equal to the test distribution, while in "most" cases shift is good.

65 (p1*,...,p*K) ↔ !K on the probability simplex !K. We let p be the mixing proportions in the test distribution, i.e. Dtest = Dp, and so the test loss is LDp 66 (h) = Lp(h), where here and elsewhere 67 we use the subscript p to denote the mixture Dp. 68 **Learning Algorithm.** We consider abstract "learning algorithm" A, which, given training data (or sequence of training examples) S ↔ ZN 69 of size N, outputs a model A(S) with test loss Dp(A(S)).

Training Distribution. We consider training on i.i.d. samples S ↗ DNq 70 from mixtures Dq of the 71 same K components, but with potentially different mixing proportions q ↔ !K. For training mixing proportions q, we denote LN (p, q) = ES↑DNp 72 [Lp(A(S))] the expected test error on Dtest = Dp 73 when training with Dtrain = Dq (we frequently drop the subscript N if its clear from context).

The "non-shifted" expected test loss is then denoted Lsame 74 N (p) = LN (p, p). In contrast, we denote L→N (p) = minq↓!K LN (p, q) the test error with the best mixing ratios, and q→ 75 the minimizing ratios.

When L→ < Lsame and so q→ 76 →= p, this means we can benefit from mismatched training. **Our main**
analysis objective is to charactarize q→, L→ **and the improvement over** Lsame 77 .

78 We can measure the mismatch benefit through the improvement in test error for a fixed training budget Lratio N = L→N /Lsame 79 N . Or, we can consider the training complexity Nω(p, q) = min N s.t. LN (p, q) ↘
ϑ and the improvement Nratio ω := N→ε (p)
Nsame ε (p) 80 .

Specifying the Learning Model The expected test loss LN (p, q), and so q→ 81 and the benefit of 82 mismatch, depend on the data distributions and learning behaviour of the algorithm. We capture 83 these by modeling the *subpoluation error function* ek(n), i.e. the error on each component Dk when training with ni examples from each component Di 84 . That is, for a vector of sample sizes n = (n1*,...,n*K) ↔ ZK↔0, denote Dn = (D1)n1 ≃*···≃*(DK) 85 nK the distributions over samples with ni examples from each component Di. Then ek(n) = ES↑Dn [LDk 86 (A(S))]. When ek(n) = gk(nk)
87 depends only on the amount of within-component data, we say the components are *orthogonal*, 88 meaning there is no transfer between them (as in our Chinese and European history example). The 89 scalar function gk(nk) then captures the *learning curve* for each component. But more generally, 90 there might also be transfer, with data from one component helping learning on another.

In any case, the learnability function e : ZK↔0 ⇐ RK 91 , captures our "learning model". In each Section, we consider different forms of learning models and characterize q→ and L→ 92 for these models. 93 **Data Sets and Training Sequences** In our analysis, we refer to the training budget N and our 94 learning model specifying learning based on nk examples per component k. We can think of N and 95 n as specifying the number of training examples, in which case the training complexity is a sample 96 complexity. Or, we can think of N as indicating the number of training steps, and nk as indicating 97 the number of steps in which an example from component k is used. In this case, training complexity 98 is a measure of training time. Either interpretation is valid. But we should emphasize that we only 99 study a dependence on *how many* examples are used from each component, not on the *order* (as in 100 curriculum learning).

101 **Learnabilities and Mixing Ratios.** We model learning as a function of the *number* of examples 102 from each component, but for our analysis, it will useful to introduce the function e¯N,k(q) =
103 ES↑(Dq)n [Lk(A(S))], which captures the expected error on component k with mixing proportions 104 q. We will refer to e¯k(q) as the subpopulation error function in terms of the mixture q. Since the per-component counts n are multinomial, we have e¯N (q) = En↑Mult(q,N)[e(n)] ↔ RK 105 and 106 LN (p, q) = ⇒p, e¯N (q)⇑. Frequently for large sample size N, e¯N (q) will concentrate around e(qN), 107 and we will sometimes exploit this in the analysis, or analyze for e¯(q) ⇓ e(qN).

## 108 **3 Orthogonal Power Law**

109 Many machine learning tasks can be captured with power law error functions. Some classic examples include linear regression or learning VC classes, both of which have error rate ↑ 1n 110 , where n is the 111 number of data samples. More recently, there have been many papers studying the loss curves for 112 large language models for various tasks as a function of the compute budget in various scaling laws, 113 such as the Chinchilla Scaling Law [Hoffmann et al., 2022]. 114 To model these situations, we will first consider a setup where each of the K tasks is orthogonal and 115 their subpopulation error functions in terms of the number of samples follow a simple power law.

116 **Model 3.1** (Orthogonal Power Law Error Tasks). There are K orthogonal tasks, each of which takes data from one of the K subpopulations Di 117 that appear in the test distribution with probability pi and whose subpopulation error function ek(n) follows a power law, i.e. ek(n) = Ak nωk k +Bk 118 for some Ak > 0, Bk ⇔ 0, and 0 < ωk ↘ 1.

1 119 120 In Proposition 3.2, we characterize the test error improvement from the positive distribution shift 121 from optimal data mixing ratios in Model 3.1 when the size of the training data n is large.

122 **Proposition 3.2** (Optimal Data Mixing Ratios For General Power Law). In Model *3.1, if for the* 123 exponents it holds that ω1 = ω2 = ··· = ωS < ωS+1 ↘ ωS+2 ↘ ··· ↘ ωK *for some* S
124 then there exist ϖ1, ϖ2 ⇔ 0 that depend on ωi such that for any test data mixing ratio p *and any* n>n0(Ai, Bi, ωi 125 , pi) *we have that the following holds*

1 ωi+1 + o *1 N  ωi↑ω1 ωi+1   (ωipiAi) %!Si=1(ωipiAi) 1 ω1+1 &ε1+1  N  ωi↑ω1 ωi+1  + q→i =  1  Lsame(p) = 1Nε1 , S i=1 p 1↗ε1 i Ai + o - 1 Nε1+ϑ1 .. (2) i=1 (ωipiAi) 1 ωi+1  +ε1 , S  L→(p) = 1Nε1 *, S i=1 (piAi) 1 ωi+1  + o - 1 Nε1+ϑ2 ω ωi ωi+1 i The o(·) notation hides dependence on Ai, Bi, pi, K and ωi 127 .
$$\left(2\right)$$
(1)  $\frac{1}{2}$ ................................. (1)  ... 
$$\quad(3)$$
.. (3)
126 128 Proposition 3.2 shows that in the power law Model 3.1, positive distribution shift from optimal data 129 mixing ratios improves the prefactor of the test error dependence on the number of data samples N
130 but does not change the decay rate in terms of N. For the proof of Proposition 3.2 and a more precise 131 statement, see Appendix A.1. 132 To show that this can have significant implications for making training more data efficient, we show 133 the improvement from this positive distribution shift on the sample complexity in the case where we 134 have one majority population and K ↓ 1 minority populations that all have the same power exponent 135 ω. This will also include the test-taking example from Section 1.

136 **Corollary 3.3** (Sample Complexity Improvement From Optimal Data Mixing For General Power 137 Law). Consider Model 3.1 with S = K, i.e. ω1 = ··· = ωK = ω and A1 = ··· = AK = A *with* p = (p, 1↗p K↗1 ,..., 1↗p K↗1 138 )*. We have that for any* ϑ > 0

$$N_{\epsilon}^{\mathrm{ratio}}(\mathbf{p})\leq(1-p)+2\frac{\alpha+1}{\alpha}\left(\frac{p}{1-p}\right)^{\frac{1}{\alpha+1}}K^{-\frac{\alpha}{\alpha+1}}.$$

Furthermore, the optimal mixing ratios are given by q→1 ↑ p 
1
ω+1 and q→i ↑
x $\left(\frac{1-p}{K-1}\right)^{\frac{1}{\alpha+1}}for\:i\geq2$. 
139 for i ⇔ 2.

140 Corollary 3.3 demonstrates an example case, that if we have one majority population and a number 141 of minority populations, the positive distribution shift from optimal data mixing ratio significantly improves sample complexity. For fixed p*, if* K is large enough, Nratio(p) will be close to Nratio 142 (p) ⇓
143 1 ↓ p < 1, i.e. we get sample complexity improvement of up to p. For example, for p = 0.7, ω = 0.28, and K = 100, for any ϑ > 0, Nratio ω 144 (p) ⇓ 0.75, i.e. we achieve the same error with ⇓ 25%
145 less samples. We illustrate this in Figure 2. For the proof of Corollary 3.3, see Appendix A.1. 146 Furthermore, the test taking example considered in the introduction Section 1 follows from Corol147 lary 3.3, by taking K = 2, ω = 1, and p = (0.9, 0.1). In particular, this shows that the optimal studying budget allocation is q→ = (0.75, 0.25) and the improvement is Nratio 148 (p)=0.8. This means that if you study for the exam with the right mixing ratio q→ 149 , you would need to study 20% less time to achieve the same score as compared to using the test mixing ratio p. Further, taking ω = 1 150 2 we get the second example on Figure 2. This shows that we indeed get q 151 → = (0.812 *...,* 0.188 ...) and Nratio 152 (p)=0.944.

1We will also use the convention that if Bk = 0 then ek(n) = min{Ck, Ak nωk k} for some Ck > 0. This will prevent L(p, q) from blowing up to infinity.

## 153 **4 Orthogonal Memorization Tasks**

154 We consider a task of memorizing a number of unique elements from a dataset of fixed size, where 155 the test distribution is a mixture of the tasks we are trying to memorize.

156 **Model 4.1** (Orthogonal Memorization Tasks). Suppose there are K tasks, each of which is a 157 memorization of a unique element. The test distribution is a mixture of these K tasks, where the k-th 158 task appears with probability pk. In this case the subpopulation error functions in terms of n is given 159 by ek(n) = 1{nk=0}.

160 The following theorem characterizes the test error improvement from the positive distribution shift 161 from optimal data mixing ratios in the Orthogonal Memorization Task Model 4.1.

162 **Theorem 4.2** (Optimal Data Mixing Test Error Improvement For Orthogonal Memorization Task).

In Model *4.1, for all* p ↔ !K↗1 163 with p1 ⇔ p2 ⇔ ··· ⇔ pK*, the expected loss when training on* n 164 *samples is given by*

$$L^{\rm same}(\mathbf{p})=\sum_{k=1}^{K}p_{k}(1-p_{k})^{N}$$  $$L^{*}(\mathbf{p})=(K_{N}(\mathbf{p})-1)\delta_{N}(\mathbf{p})+\sum_{k=K_{N}(\mathbf{p})+1}^{K}p_{k},$$
$$(4)$$
$$(S)$$
$$(\mathbf{6})$$

165 *where* ϱN (p) ↔ [pKN (p)+1, pKN (p)) and KN (p) *is defined as follows:*

$$K_{N}(\mathbf{p}):=\operatorname*{max}\left\{s\leq K:\sum_{k=1}^{s-1}(1-(p_{s}/p_{k})^{1/(K-1)})<1\right\}.$$

166 To understand the magnitute of the test error improvement in Theorem 4.2, we will assume that the test proportions p follow a power law pk = "(k↗ε 167 ) for some ω > 1 and that the number of tasks to 168 memorize K is larger than the size of the training set N. In this case, we show that the improvement 169 from positive distribution shift Theorem 4.2 improves even the test error scaling in terms of N. For 170 the proof of Theorem 4.2, see Appendix A.2. 171 **Corollary 4.3** (Test Error Improvement For Orthogonal Memorization Taks with Power Law Test Mixing Ratios). If pk = "(k↗ε 172 ) for some ω > 1 and K = \#(N)*, then*

$$L^{\mathrm{same}}(\mathbf{p})=\Theta(N^{-1+{\frac{1}{\alpha}}}),\qquad L^{*}(\mathbf{p})=\Theta(N^{-\alpha+1}).$$

For example, when ω = 1.5, we have Lsame(p) = "(N ↗1/3) and L→(p) = "(N ↗1/2 173 ). For the 174 proof of Corollary 4.3, see Appendix A.2.

## 175 **5 Connection To Skill Composition** 203 **6 Non-Orthogonal Tasks And Transfer Learning**

176 All the above analyses focus on the case where tasks are orthogonal. However, if we already know 177 that the test distribution can be decomposed into K tasks, then maybe we should deal with these K 178 tasks independently. So why do we have test mixing ratios in the first place? 179 We note here that in some cases, we may need to compose these K tasks later at inference time, and 180 the test mixing ratios can come from the proportions in the composition. Imagine that we are training 181 a language model to do mathematical reasoning. Each problem may involve several math skills, and 182 a language model can acquire a math skill only if it sees the skill enough times during training. This 183 can be conceptually modeled as the orthogonal memorization task discussed above, but at inference 184 time, the language model has to sequentially apply the math skills in its chain of thought (CoT). The 185 natural distribution of math skills then determines the test mixing ratios we care about. 186 We demonstrate this in a concrete synthetic task on skill composition. There are K skills, where the i-th skill is a function gi 187 that maps a number from {0,..., 9} to {0*,...,* 9}. Each skill has a unique En188 glish name. Assume that all these skills are randomly sampled: the names are uniformly random from a name set, and each gi 189 is uniformly random among all possible functions that map from {0*,...,* 9} to {0*,...,* 9}. At inference time, a set of k skills gi1 *,...,g*ik 190 are sampled IID following a power law with 191 exponent ω = 1.5. The language model is prompted with the names of these skills and a number x ↔
192 {0*,...,* 9}: "[x] -> [skill name 1] -> [skill name 2] -> ··· -> [skill name k]".

The model is expected to output the result after function composition: y = gik (gik↑1 (··· gi1 193 (x)···)).

194 Let Dtest be the distribution of the above prompt and a CoT calculating the correct answer, with M = 105 195 , k sampled uniformly from 10 to 50. Is the best strategy just training on the same 196 distribution (Dtrain = Dtest)? Inspired by our calculation for the orthogonal memorization task 197 above, properly adjusting the occurrence probability for each skill may lead to better test accruacy.

198 To demonstrate this, we construct another distribution Duniform consisting of strings in the form of 199 "[x] [skill name] = [expected output]", where the skill and input number are uniformly 200 sampled. In Figure 3, we conduct experiments with a model with GPT-2 architecture and ↗50M 201 parameters. We show that training with Dtrain = 30% · Duniform + 70% · Dtest significantly 202 outperform training with Dtest directly. We defer the experiment details to Appendix C.

204 Many transfer learning setups, such as multi-task learning of linear classifiers over linear representa205 tion with feature learning Baxter [2011], Maurer [2009], Pontil and Maurer [2013], Aliakbarpour 206 et al. [2024] and multi-task learning with shared sparsityWang et al. [2016, 2017], the subpopulation error functions ek(n) can be written in the form ek(n) = A0,k
(n1+···+nk)ωk + 
A1,k nωk k 207 . For example, 208 in multi-task learning of shared sparsity Wang et al. [2017], the error bound takes this form with 209 ω1 = ··· = ωK = 1.

210 To model all of these cases, we consider the following model of transfer learning. 211 **Model 6.1** (Standard Transfer Learning Model). There are K subpopulations, each of which appears 212 in the test distribution with proportion pk. The subpopulation error functions depend on the number of samples n as ek(n) = A0,k
(n1+···+nk)ωk + 
A1,k nωk k 213 , for some A0,k, A1,k > 0 and 0 < ωk ↘ 1.

214 Interestingly, the Standard Transfer Learning Model 6.1 is equivalent to the setup of Orthogonal Power Law Tasks Model 3.1 in the sense that we can understand optimal data mixing ratio q→ 215 and the error 216 improvement of the Standard Transfer Learning model from a specific instance of the Orthogonal 217 Power Law model. Namely, the transfer term in each of the subpopulation loss functions can be decomposed into a transfer error term and a specific task error term ek(n) = etransfer k (n) + e spec k 218 (n),
where etransfer k (n) = A0,k
(n1+···+nk)ωk 219 is independent of the distribution of samples across different tasks, and e spec k (n) = A1,k nωk k only depends on nk. Therefore, the transfer error term etransfer k 220 (n) in each of the subpoluation error functions will only offset the final expected loss L(p, q) by !Ki=1 pi A0,k Nωk 221 , which 222 only depends on the total number of samples N. On the other hand, the specific task error terms e spec k 223 (n) can be thought of as orthogonal tasks and will behave tha same as in Model 3.1. So, for the Standard Transfer Learning Model 6.1, the optimal data mixing ratio q→ 224 and the expected test losses L→(p) and Lsame 225 (p) are given by Equation (1) and Equation (2) respectively in Proposition 3.2 with 226 Ak being replaced by A1,k.

## 227 **6.1 Data Mixing Transfer Learning.**

228 Ye et al. [2025] consider the problem of estimating the outcome performance of a large langue model 229 trained on a mixture of domains. In particular, they find that an exponential function over the linear 230 combinations of mixing proportions leads to good prediction. Namely, they fix the training budget N 231 and only vary the mixing ratio q and show that the validation loss on i-th domain can be predicted well by a function of the form ci + bi exp %↓!Kj=1 tij qj
&, where ci, bi 232 , tij are parameters to fit.

233 Following their work, we propose the following model for the Data Mixing Transfer Learning. 234 **Model 6.2** (Data Mixing Transfer Learning). There are K subpopulations, each of which appears 235 with probability pk in the test distribution. Each of the subpopulation error functions in terms of the mixing ratio q are e¯k(q) = ck + bk exp 
%↓!Kj=1 tij qj
&
236 for some constants ck and bk > 0, tij . 237 We note that even though Model 6.2 is indeed not defined by the subpopulation error functions 238 ek(n), it is precisely the setup that Ye et al. [2025] consider. This slightly deviates from our 239 main setup, which focuses on specifying models by their error functions. However, when the 240 number of samples N is large, it is reasonable to make the approximation that ek(n) ⇓ ek(qN),
241 and Model 6.2 can be interpreted as being defined by the subpopulation error functions of the form ek(n) = ck(|n|) + bk(|n|) exp 
%↓!Kj=1 tij (|n|)nj
&
242 , where ck, bk, and tij are functions that 243 depend only on the total compute budget N = |n|.

244 The following proposition characterizes the test error improvement from the positive distribution 245 shift coming from the optimal data mixing ratio in the data mixing transfer model.

246 **Proposition 6.3** (Optimal Train Data Mixing Ratio for Data Mixing Transfer Learning Model). In Model 6.2, if the coefficients tij are such that T *is invertible and and* (T T )↗1 247 1 > 0*, and* pi →= 0 for 248 all i*, the following hold*

$$\begin{array}{c}{{\mathbf{q}^{*}=(\mathbf{T})^{-1}\left(\frac{1+\mathbf{I}^{\top}\mathbf{T}^{-1}\tau}{\mathbf{I}\mathbf{T}^{-1}\mathbf{I}}\mathbf{I}-\tau\right)}}\\ {{L^{\mathrm{same}}(\mathbf{p})=\sum_{i=1}^{K}c_{i}p_{i}+\sum_{i=1}^{K}p_{i}b_{i}\exp\left(-\sum_{j=1}^{K}t_{i j}p_{j}\right)}}\\ {{L^{*}(\mathbf{p})=\sum_{i=1}^{K}c_{i}p_{i}+\exp\left(\frac{-1-\mathbf{I}^{\top}\mathbf{T}^{-1}\tau}{\mathbf{I}^{T}\mathbf{T}^{-1}\mathbf{I}}\right)\mathbf{I}^{T}(\mathbf{T}^{\top})^{-1}\mathbf{I},}}\end{array}$$

where ς *is a vector with entreis* ςl = log %[(T ↓)↑11]l
$$i t h\;e n t r e i s\;\tau_{l}=\log\bigg(\frac{[(\mathbf{T}^{\top})^{-1}\mathbf{I}]_{l}}{p_{l}\,b_{l}}\bigg).$$
249 .

250 Proposition 6.3 shows the positive distribution from the optimal data mixing for Model 6.2. Note that 251 the additional conditions on T , pi are technical conditions used in order to simplify presentation. For 252 the complete statement and the proof of Proposition 6.3, see Appendix A.3. 253 To demonstrate how large the gap can be, we consider the problem of data mixing transfer learning 254 Model 6.2 with K = 2 tasks and a one-directional transfer from the second to the first task. 255 **Corollary 6.4** (Optimal Data Mixing Ratio Can Have Significant Improvement in the Transfer Learning Model). Let K = 2*, let* p = ( 12 , 12 ), and let b1 = b2 = b > 0*. If* T =
-1 ω 0 1
.

256 *then we* 257 *have that*

$$L^{\mathrm{same}}-L^{*}=2b e^{-\frac{1}{2}}\left(1-\frac{1}{4}\alpha+O(a^{2})\right)$$

Furthermore, if we let C = c1+c2 2 and B = be↗ 
1 258 2 *then we have that*

$$L^{\,ratio}=\frac{L_{N}}{L^{*}}=\frac{C-B}{C+B}+\frac{BC}{2(B+C)^{2}}\alpha+O(\alpha^{2})\,.$$

259 Corollary 6.4 shows that for two tasks with a small of transfer between the second to the first we 260 can have error improvement from the positive distribution shift by mismatching training and test distribution, that is Lratio ⇓ C↗B 261 C+B < 1 for small ω. For the proof of Corollary 6.4, see Appendix A.3.

## 262 **7 It'S Almost Always Better To Mismatch**

263 So far, we have shown the existence of and quantified the positive distribution shift coming from 264 mistmatched test and train data mixing ratios for the cases of orthogonal power law tasks in Section 3, 265 orthogonal memorization tasks in Section 4, and standard transfer learning and data mixing transfer 266 learning in Section 6. that positive distribution shift from mismatching test and train mixing ratios 267 exists. In this section, we will provide further mathematical justification that a positive distribution 268 shift coming from the data mixing ratio almost always exists. That is, we show that it's almost always better to mismatch the training and test distributions: q→ →= p and L→(p, q→) < Lsame 269 (p).

270 More precisely, we will show that either the test data mixing ratio is on a measure zero set of 271 the simplex or the subpopulation error functions ek(n) have to be very specific functions, which 272 are meaningless. For example, in the case of orthogonal tasks, either the test mixing ratio is on a 273 measure zero subset or the subpopulation error functions ek(n) are all constants, which we show in 274 Corollary 7.4.

We define the probability simplex !K↗1 := p ↔ RK : p ⇔ 0, |p| = 1, and its interior !K↗1 275 + := p ↔ RK : p > 0, |p| = 1, where |p| := !Kk=1 276 pk. We will define fk(p) by extending the domain of each e¯k(p) to the set of non-zero, non-negative vectors RK↔0 \ {0} by defining fk(p) := ¯ek( p |p| 277 ). We further define Lsame(p) := !Kk=1 pkfk(p), which extends the definition of Lsame 278 to the set of non-zero, non-negative vectors RK↔0 279 \ {0}. Condition 7.1 (Conservation Condition). (f1(p)*,...,f*K(p)) = ↖Lsame(p) *for all* p ↔ RK↔0 280 \ {0}.

281 **Theorem 7.2** (Positive Distribution Shift Almost Always Exists For Data Mixing). *For any set of* 282 subpopulations D1,..., DK and any learning algorithm A, either Condition 7.1 holds, or there exists a zero-measure set U on !K↗1 such that for all p ↔ !K↗1 \ U, L→N (p) < Lsame 283 (p).

Theorem 7.2 shows that either p is on a measure zero set U on !K↗1 284 or the Conservation Condi285 tion 7.1 must hold. We will show that Conservation Condition 7.1 happens only for very specific 286 cases of subpopulation error functions. 287 **Conservation Condition Rarely Holds.** First, we will show that if the subtasks are orthogonal, the 288 conservation condition Condition 7.1 is only satisfied if all of the subpopulation error functions are 289 constants.

Lemma 7.3 (Orthogonal Tasks). If K ⇔ 3*, and if for all* k ↔ [K], fk(p) = gk( pk |p| 290 ) *for some function* 291 gk, then Condition 7.1 holds if and only if gk*'s are all constant functions.*
292 Theorem 7.2 and Lemma 7.3 together show that in the case of orthogonal tasks, positive distirbution 293 shift always exists by changing the training data mixing ratio away from the test mixing ratio, unless 294 all the subpopulation error functions are constant.

295 **Corollary 7.4** (Positive Distribution Shift Always Exists for Orthogonal Tasks). *For any set of* 296 K ⇔ 3 subpopulations D1,..., DK and any learning algorithm A*, if there exists subpopulation* 297 k ↔ [K] such that its error function ek is not a constant functions over [N] where N *is the number* of total samples then there exists a measure zero set U on !K↗1 *such that for all* p ↔ !K↗1 298 \ U
positive distribution shift from data mixing exists in the sense that there is q→ 299 →= p *for which* LN (p, q) = L→(p) < Lsame 300 (p).

301 Further, we show that if the Conservation Condition 7.1 is satisfied, then one function fi determines 302 the rest up to a constant.

Lemma 7.5. If both (f1,...,fK, Lsame) and ( ˆf1*,...,* ˆfK,Lˆsame 303 ) satisfy Condition *7.1, and if* fi = ˆfi for some i ↔ [m]*, then for all* k →= i, fk(p) = ˆ 304 fk(p) + Ck *for some constant* Ck.

305 The above Lemma 7.5 implies that for every k and corresponding error function ek(n), there exists at most one tuple of error functions {ej}Kj=1,j≃=k 306 (up to a individual constant offset for each error 307 function ej ) that positive distribution shift does not happen for p of positive measure. This further 308 implies the following corollary. 309 **Corollary 7.6** (Positive Distribution Shift *Almost* Always Exists for General Tasks). *For any set* of K ⇔ 3 subpopulations D1,..., DK and any learning algorithm A*, for all* p ↔ !K↗1 310 + *, the* 311 configuration of [ek(n)]k↓[K],n *that positive distribution shift does not happen is zero-measure.*
312 Corollary 7.6 shows that either the test mixing ratio p is on a set of measure zero on the simplex or 313 the configuration of subpopulation error functions ek(n) is on a set of measure zero. This implies 314 that positive distribution shift exists *almost* always.

## 315 **8 Related Works**

316 **Distribution Shift That is Not Harmful.** The benefits of mismathcing the training and test distri317 bution has already been in studied in some settings. González and Abu-Mostafa [2015] demonstrate 318 in many linear regression problems that mismatched training and test distributions can outperform 319 matched ones. Unlike in our paper, they do not restrict to changing the train distribution only through 320 data mixing, so their results do not fit our framework. On the other hand, we explicitly characterize 321 the positive distribution shift, while González and Abu-Mostafa [2015] only show its existence for 322 linear regression problems and are only able to characterize the distribution explicitly in very special 323 cases. Canatar et al. [2021] show how in high-dimensional kernel regression problems to numerically 324 optimize the training distribution for better test performance. However, they do not characterize 325 the positive distribution shift, but rather only show how to numerically find it for kernel regression.

326 Similarly, they do not restrict the test distribution to one coming from a data mixture, so their results 327 do not fit our framework. 328 **Data Mixing.** There a number of recent empiricaly works that consider the same setting of data 329 mixing as we do. Ye et al. [2025] introduce data mixing laws, quantitative empirical predictions 330 of large language model performance based on the data mixture proportions. Furthermore, they 331 show experimental results demonstrating that their approach significantly decreases the number of 332 steps needed to reach certain performance. This paper informed our data mixing transfer model and 333 fits in our framework. Goyal et al. [2024] show that data curation for VLMs cannot be compute 334 agnostic. They introduce neural scaling laws that allow for estimating performance on multiple 335 data pools without jointly training on them. Their work fits our framework. Similarly, we also find 336 that optimal mixing ratios are not compute agnostic, specifically in the orthogonal power law tasks, 337 orthogonal memorization task, and standard transfer learning task. Jiang et al. [2025] introduce an 338 algorithm for online optimization of data distributions, that adjusts mixture based on the estimated 339 per-domain learning potential, achieving comparable or better performance than previous methods 340 while maintaing compuatational efficiency. While all of these works consider the same phenomena 341 of changing the training mixing ratio to improve test performacne, the main difference between our 342 work and theirs is that we consider positive distribution shift from data mixing ratio in a broader 343 context and from the theoretical standpoint as well.

## 344 **References**

345 Maryam Aliakbarpour, Konstantina Bairaktari, Gavin Brown, Adam Smith, Nathan Srebro, and 346 Jonathan Ullman. Metalearning with very few samples per task. In Shipra Agrawal and Aaron 347 Roth, editors, *Proceedings of Thirty Seventh Conference on Learning Theory*, volume 247 of 348 *Proceedings of Machine Learning Research*, pages 46–93. PMLR, 30 Jun–03 Jul 2024. URL
349 https://proceedings.mlr.press/v247/aliakbarpour24a.html.

350 Jonathan Baxter. A model of inductive bias learning. *CoRR*, abs/1106.0245, 2011. URL http: 351 //arxiv.org/abs/1106.0245.

352 Abdulkadir Canatar, Blake Bordelon, and Cengiz Pehlevan. Out-of-distribution generalization in ker353 nel regression. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan, 354 editors, *Advances in Neural Information Processing Systems*, volume 34, pages 12600–12612. Cur355 ran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper_files/paper/ 356 2021/file/691dcb1d65f31967a874d18383b9da75-Paper.pdf. 357 Carlos R. González and Yaser S. Abu-Mostafa. Mismatched training and test distributions can 358 outperform matched ones. *Neural Computation*, 27(2):365–387, 2015. doi: 10.1162/NECO_a_ 359 00697. 360 Sachin Goyal, Pratyush Maini, Zachary C. Lipton, Aditi Raghunathan, and J. Zico Kolter. Scaling 361 laws for data filtering—data curation cannot be compute agnostic. In *2024 IEEE/CVF Conference* 362 *on Computer Vision and Pattern Recognition (CVPR)*, pages 22702–22711, 2024. doi: 10.1109/
363 CVPR52733.2024.02142. 364 Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Ruther365 ford, Diego de las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, 366 Eric Noland, Katherine Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon 367 Osindero, Karen Simonyan, Erich Elsen, Oriol Vinyals, Jack William Rae, and Laurent Sifre. An 368 empirical analysis of compute-optimal large language model training. In Alice H. Oh, Alekh Agar369 wal, Danielle Belgrave, and Kyunghyun Cho, editors, *Advances in Neural Information Processing* 370 *Systems*, 2022. URL https://openreview.net/forum?id=iBBcRUlOAPR.

371 Yiding Jiang, Allan Zhou, Zhili Feng, Sadhika Malladi, and J Zico Kolter. Adaptive data optimization:
372 Dynamic sample selection with scaling laws. In *The Thirteenth International Conference on* 373 *Learning Representations*, 2025. URL https://openreview.net/forum?id=aqok1UX7Z1.

374 Andreas Maurer. Transfer bounds for linear feature learning. *Machine Learning*, 75:327–350, 2009. 375 URL https://api.semanticscholar.org/CorpusID:14682470.

376 Massimiliano Pontil and Andreas Maurer. Excess risk bounds for multitask learning with trace 377 norm regularization. In Shai Shalev-Shwartz and Ingo Steinwart, editors, *Proceedings of the 26th* 378 *Annual Conference on Learning Theory*, volume 30 of *Proceedings of Machine Learning Research*,
379 pages 55–76, Princeton, NJ, USA, 12–14 Jun 2013. PMLR. URL https://proceedings.mlr. 380 press/v30/Pontil13.html.

381 Jialei Wang, Mladen Kolar, and Nathan Srerbo. Distributed multi-task learning. In Arthur Gretton 382 and Christian C. Robert, editors, *Proceedings of the 19th International Conference on Artificial* 383 *Intelligence and Statistics*, volume 51 of *Proceedings of Machine Learning Research*, pages 751–
384 760, Cadiz, Spain, 09–11 May 2016. PMLR. URL https://proceedings.mlr.press/v51/
385 wang16d.html. 386 Jialei Wang, Mladen Kolar, Nathan Srebro, and Tong Zhang. Efficient distributed learning with 387 sparsity. In Doina Precup and Yee Whye Teh, editors, *Proceedings of the 34th International* 388 *Conference on Machine Learning*, volume 70 of *Proceedings of Machine Learning Research*,
389 pages 3636–3645. PMLR, 06–11 Aug 2017. URL https://proceedings.mlr.press/v70/ 390 wang17f.html. 391 Jiasheng Ye, Peiju Liu, Tianxiang Sun, Jun Zhan, Yunhua Zhou, and Xipeng Qiu. Data mixing 392 laws: Optimizing data mixtures by predicting language modeling performance. In *The Thirteenth* 393 *International Conference on Learning Representations*, 2025. URL https://openreview.net/
394 forum?id=jjCB27TMK3.

## 395 **Neurips Paper Checklist**

396 The checklist is designed to encourage best practices for responsible machine learning research, 397 addressing issues of reproducibility, transparency, research ethics, and societal impact. Do not remove 398 the checklist: **The papers not including the checklist will be desk rejected.** The checklist should 399 follow the references and follow the (optional) supplemental material. The checklist does NOT count 400 towards the page limit. 401 Please read the checklist guidelines carefully for information on how to answer these questions. For 402 each question in the checklist:
403 - You should answer [Yes] , [No] , or [NA] .

404 - [NA] means either that the question is Not Applicable for that particular paper or the relevant 405 information is Not Available. 406 - Please provide a short (1–2 sentence) justification right after your answer (even for NA). 407 **The checklist answers are an integral part of your paper submission.** They are visible to the 408 reviewers, area chairs, senior area chairs, and ethics reviewers. You will be asked to also include it 409 (after eventual revisions) with the final version of your paper, and its final version will be published 410 with the paper. 411 The reviewers of your paper will be asked to use the checklist as one of the factors in their evaluation. 412 While "[Yes] " is generally preferable to "[No] ", it is perfectly acceptable to answer "[No] " provided a 413 proper justification is given (e.g., "error bars are not reported because it would be too computationally 414 expensive" or "we were unable to find the license for the dataset we used"). In general, answering 415 "[No] " or "[NA] " is not grounds for rejection. While the questions are phrased in a binary way, we 416 acknowledge that the true answer is often more nuanced, so please just use your best judgment and 417 write a justification to elaborate. All supporting evidence can appear either in the main paper or the 418 supplemental material, provided in appendix. If you answer [Yes] to a question, in the justification 419 please point to the section(s) where related material for the question can be found. 420 IMPORTANT, please: 421 - **Delete this instruction block, but keep the section heading "NeurIPS Paper Checklist"**, 422 - **Keep the checklist subsection headings, questions/answers and guidelines below.** 423 - **Do not modify the questions and only use the provided macros for your answers**.

424 1. **Claims**
425 Question: Do the main claims made in the abstract and introduction accurately reflect the paper's 426 contributions and scope? 427 Answer: [Yes] 428 Justification: Yes, the main claim accuretly reflects the paper's contribution and scope. 429 Guidelines: 430 - The answer NA means that the abstract and introduction do not include the claims made in 431 the paper.

432 - The abstract and/or introduction should clearly state the claims made, including the contribu433 tions made in the paper and important assumptions and limitations. A No or NA answer to 434 this question will not be perceived well by the reviewers. 435 - The claims made should match theoretical and experimental results, and reflect how much 436 the results can be expected to generalize to other settings. 437 - It is fine to include aspirational goals as motivation as long as it is clear that these goals are 438 not attained by the paper. 439 2. **Limitations**
440 Question: Does the paper discuss the limitations of the work performed by the authors?

441 Answer: [Yes] 442 Justification: Yes, we discuss the limitations of our work and clearly define the scope of each of 443 our claims.

## 444 Guidelines:

445 - The answer NA means that the paper has no limitation while the answer No means that the 446 paper has limitations, but those are not discussed in the paper. 447 - The authors are encouraged to create a separate "Limitations" section in their paper. 448 - The paper should point out any strong assumptions and how robust the results are to vi449 olations of these assumptions (e.g., independence assumptions, noiseless settings, model 450 well-specification, asymptotic approximations only holding locally). The authors should 451 reflect on how these assumptions might be violated in practice and what the implications 452 would be. 453 - The authors should reflect on the scope of the claims made, e.g., if the approach was only 454 tested on a few datasets or with a few runs. In general, empirical results often depend on 455 implicit assumptions, which should be articulated.

456 - The authors should reflect on the factors that influence the performance of the approach. For 457 example, a facial recognition algorithm may perform poorly when image resolution is low or 458 images are taken in low lighting. Or a speech-to-text system might not be used reliably to 459 provide closed captions for online lectures because it fails to handle technical jargon. 460 - The authors should discuss the computational efficiency of the proposed algorithms and how 461 they scale with dataset size. 462 - If applicable, the authors should discuss possible limitations of their approach to address 463 problems of privacy and fairness. 464 - While the authors might fear that complete honesty about limitations might be used by review465 ers as grounds for rejection, a worse outcome might be that reviewers discover limitations that 466 aren't acknowledged in the paper. The authors should use their best judgment and recognize 467 that individual actions in favor of transparency play an important role in developing norms 468 that preserve the integrity of the community. Reviewers will be specifically instructed to not 469 penalize honesty concerning limitations.

## 470 3. **Theory Assumptions And Proofs**

471 Question: For each theoretical result, does the paper provide the full set of assumptions and a 472 complete (and correct) proof? 473 Answer: [Yes] 474 Justification: We provide full set of assumptions and complete and corrected proofs in the 475 appendix. For some of the claims, we only state an informal or a limited scope version in the 476 main body for the ease of presentation. 477 Guidelines: 478 - The answer NA means that the paper does not include theoretical results. 479 - All the theorems, formulas, and proofs in the paper should be numbered and cross-referenced. 480 - All assumptions should be clearly stated or referenced in the statement of any theorems. 481 - The proofs can either appear in the main paper or the supplemental material, but if they appear 482 in the supplemental material, the authors are encouraged to provide a short proof sketch to 483 provide intuition. 484 - Inversely, any informal proof provided in the core of the paper should be complemented by 485 formal proofs provided in appendix or supplemental material.

486 - Theorems and Lemmas that the proof relies upon should be properly referenced.

## 487 4. **Experimental Result Reproducibility**

488 Question: Does the paper fully disclose all the information needed to reproduce the main 489 experimental results of the paper to the extent that it affects the main claims and/or conclusions 490 of the paper (regardless of whether the code and data are provided or not)? 491 Answer: [Yes] 492 Justification: Yes, we disclose the information needed to reproduce the experiments. 493 Guidelines: 494 - The answer NA means that the paper does not include experiments. 495 - If the paper includes experiments, a No answer to this question will not be perceived well by 496 the reviewers: Making the paper reproducible is important, regardless of whether the code 497 and data are provided or not. 498 - If the contribution is a dataset and/or model, the authors should describe the steps taken to 499 make their results reproducible or verifiable. 500 - Depending on the contribution, reproducibility can be accomplished in various ways. For 501 example, if the contribution is a novel architecture, describing the architecture fully might 502 suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary 503 to either make it possible for others to replicate the model with the same dataset, or provide 504 access to the model. In general. releasing code and data is often one good way to accomplish 505 this, but reproducibility can also be provided via detailed instructions for how to replicate the 506 results, access to a hosted model (e.g., in the case of a large language model), releasing of a 507 model checkpoint, or other means that are appropriate to the research performed. 508 - While NeurIPS does not require releasing code, the conference does require all submissions 509 to provide some reasonable avenue for reproducibility, which may depend on the nature of 510 the contribution. For example 511 (a) If the contribution is primarily a new algorithm, the paper should make it clear how to 512 reproduce that algorithm. 513 (b) If the contribution is primarily a new model architecture, the paper should describe the 514 architecture clearly and fully. 515 (c) If the contribution is a new model (e.g., a large language model), then there should either 516 be a way to access this model for reproducing the results or a way to reproduce the model 517 (e.g., with an open-source dataset or instructions for how to construct the dataset). 518 (d) We recognize that reproducibility may be tricky in some cases, in which case authors are 519 welcome to describe the particular way they provide for reproducibility. In the case of 520 closed-source models, it may be that access to the model is limited in some way (e.g., 521 to registered users), but it should be possible for other researchers to have some path to 522 reproducing or verifying the results.

## 523 5. **Open Access To Data And Code**

524 Question: Does the paper provide open access to the data and code, with sufficient instructions 525 to faithfully reproduce the main experimental results, as described in supplemental material? 526 Answer: [Yes] 527 Justification: Yes, we provide the access in to the code and data in the appendix. 528 Guidelines: 529 - The answer NA means that paper does not include experiments requiring code.

530 - Please see the NeurIPS code and data submission guidelines (https://nips.cc/public/ 531 guides/CodeSubmissionPolicy) for more details. 532 - While we encourage the release of code and data, we understand that this might not be 533 possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including 534 code, unless this is central to the contribution (e.g., for a new open-source benchmark). 535 - The instructions should contain the exact command and environment needed to run to 536 reproduce the results. See the NeurIPS code and data submission guidelines (https://nips. 537 cc/public/guides/CodeSubmissionPolicy) for more details. 538 - The authors should provide instructions on data access and preparation, including how to 539 access the raw data, preprocessed data, intermediate data, and generated data, etc. 540 - The authors should provide scripts to reproduce all experimental results for the new proposed 541 method and baselines. If only a subset of experiments are reproducible, they should state 542 which ones are omitted from the script and why.

543 - At submission time, to preserve anonymity, the authors should release anonymized versions 544 (if applicable).

545 - Providing as much information as possible in supplemental material (appended to the paper)
546 is recommended, but including URLs to data and code is permitted. 547 6. **Experimental setting/details**
548 Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, 549 how they were chosen, type of optimizer, etc.) necessary to understand the results? 550 Answer: [Yes] 551 Justification: Yes, we specify all the details of the experiment necessary to understand and 552 reproduce the experiments. 553 Guidelines: 554 - The answer NA means that the paper does not include experiments. 555 - The experimental setting should be presented in the core of the paper to a level of detail that 556 is necessary to appreciate the results and make sense of them. 557 - The full details can be provided either with the code, in appendix, or as supplemental material.

## 558 7. **Experiment Statistical Significance**

559 Question: Does the paper report error bars suitably and correctly defined or other appropriate 560 information about the statistical significance of the experiments? 561 Answer: [Yes] 562 Justification: Yes, we provide information about statistical significance of results where appropri563 ate. 564 Guidelines: 565 - The answer NA means that the paper does not include experiments. 566 - The authors should answer "Yes" if the results are accompanied by error bars, confidence 567 intervals, or statistical significance tests, at least for the experiments that support the main 568 claims of the paper. 569 - The factors of variability that the error bars are capturing should be clearly stated (for example, 570 train/test split, initialization, random drawing of some parameter, or overall run with given 571 experimental conditions). 572 - The method for calculating the error bars should be explained (closed form formula, call to a 573 library function, bootstrap, etc.)
574 - The assumptions made should be given (e.g., Normally distributed errors).

575 - It should be clear whether the error bar is the standard deviation or the standard error of the 576 mean. 577 - It is OK to report 1-sigma error bars, but one should state it. The authors should preferably 578 report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality 579 of errors is not verified. 580 - For asymmetric distributions, the authors should be careful not to show in tables or figures 581 symmetric error bars that would yield results that are out of range (e.g. negative error rates). 582 - If error bars are reported in tables or plots, The authors should explain in the text how they 583 were calculated and reference the corresponding figures or tables in the text.

## 584 8. **Experiments Compute Resources**

585 Question: For each experiment, does the paper provide sufficient information on the computer 586 resources (type of compute workers, memory, time of execution) needed to reproduce the 587 experiments? 588 Answer: [Yes] 589 Justification: Yes, we provide sufficient information on the computer resources needed to 590 reproduce the experiments in the appendix. 591 Guidelines: 592 - The answer NA means that the paper does not include experiments. 593 - The paper should indicate the type of compute workers CPU or GPU, internal cluster, or 594 cloud provider, including relevant memory and storage. 595 - The paper should provide the amount of compute required for each of the individual experi596 mental runs as well as estimate the total compute.

597 - The paper should disclose whether the full research project required more compute than the 598 experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it 599 into the paper).

600 9. **Code of ethics**
601 Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS
602 Code of Ethics https://neurips.cc/public/EthicsGuidelines?

603 Answer: [Yes] 604 Justification: Yes, our research conforms in every aspect to the NeurIPS Code of Ethics. 605 Guidelines: 606 - The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. 607 - If the authors answer No, they should explain the special circumstances that require a deviation 608 from the Code of Ethics.

609 - The authors should make sure to preserve anonymity (e.g., if there is a special consideration 610 due to laws or regulations in their jurisdiction).

## 611 10. **Broader Impacts**

612 Question: Does the paper discuss both potential positive societal impacts and negative societal 613 impacts of the work performed? 614 Answer: [NA] 615 Justification: As this is mainly a theoretical paper, there is no immediate societal impact of the 616 owrk. 617 Guidelines: 618 - The answer NA means that there is no societal impact of the work performed.

619 - If the authors answer NA or No, they should explain why their work has no societal impact or 620 why the paper does not address societal impact. 621 - Examples of negative societal impacts include potential malicious or unintended uses (e.g., 622 disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deploy623 ment of technologies that could make decisions that unfairly impact specific groups), privacy 624 considerations, and security considerations. 625 - The conference expects that many papers will be foundational research and not tied to 626 particular applications, let alone deployments. However, if there is a direct path to any 627 negative applications, the authors should point it out. For example, it is legitimate to point out 628 that an improvement in the quality of generative models could be used to generate deepfakes 629 for disinformation. On the other hand, it is not needed to point out that a generic algorithm 630 for optimizing neural networks could enable people to train models that generate Deepfakes 631 faster. 632 - The authors should consider possible harms that could arise when the technology is being 633 used as intended and functioning correctly, harms that could arise when the technology is 634 being used as intended but gives incorrect results, and harms following from (intentional or 635 unintentional) misuse of the technology. 636 - If there are negative societal impacts, the authors could also discuss possible mitigation 637 strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms 638 for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, 639 improving the efficiency and accessibility of ML). 640 11. **Safeguards**
641 Question: Does the paper describe safeguards that have been put in place for responsible release 642 of data or models that have a high risk for misuse (e.g., pretrained language models, image 643 generators, or scraped datasets)? 644 Answer: [NA] 645 Justification: The paper poses no such risks. 646 Guidelines: 647 - The answer NA means that the paper poses no such risks. 648 - Released models that have a high risk for misuse or dual-use should be released with necessary 649 safeguards to allow for controlled use of the model, for example by requiring that users adhere 650 to usage guidelines or restrictions to access the model or implementing safety filters. 651 - Datasets that have been scraped from the Internet could pose safety risks. The authors should 652 describe how they avoided releasing unsafe images. 653 - We recognize that providing effective safeguards is challenging, and many papers do not 654 require this, but we encourage authors to take this into account and make a best faith effort.

## 655 12. **Licenses For Existing Assets**

656 Question: Are the creators or original owners of assets (e.g., code, data, models), used in the 657 paper, properly credited and are the license and terms of use explicitly mentioned and properly 658 respected? 659 Answer: [Yes] 660 Justification: Yes, we properly credit all the original owners of assets where due. 661 Guidelines: 662 - The answer NA means that the paper does not use existing assets. 663 - The authors should cite the original paper that produced the code package or dataset.

664 - The authors should state which version of the asset is used and, if possible, include a URL.

665 - The name of the license (e.g., CC-BY 4.0) should be included for each asset. 666 - For scraped data from a particular source (e.g., website), the copyright and terms of service 667 of that source should be provided.

668 - If assets are released, the license, copyright information, and terms of use in the package 669 should be provided. For popular datasets, paperswithcode.com/datasets has curated 670 licenses for some datasets. Their licensing guide can help determine the license of a dataset.

671 - For existing datasets that are re-packaged, both the original license and the license of the 672 derived asset (if it has changed) should be provided. 673 - If this information is not available online, the authors are encouraged to reach out to the 674 asset's creators. 675 13. **New assets** 676 Question: Are new assets introduced in the paper well documented and is the documentation 677 provided alongside the assets? 678 Answer: [NA] 679 Justification: We do not realease new assets. 680 Guidelines: 681 - The answer NA means that the paper does not release new assets. 682 - Researchers should communicate the details of the dataset/code/model as part of their sub683 missions via structured templates. This includes details about training, license, limitations, 684 etc. 685 - The paper should discuss whether and how consent was obtained from people whose asset is 686 used. 687 - At submission time, remember to anonymize your assets (if applicable). You can either create 688 an anonymized URL or include an anonymized zip file.

## 689 14. **Crowdsourcing And Research With Human Subjects**

690 Question: For crowdsourcing experiments and research with human subjects, does the paper 691 include the full text of instructions given to participants and screenshots, if applicable, as well as 692 details about compensation (if any)? 693 Answer: [NA] 694 Justification: The paper does not involve crowdourcing nor research with human subjects. 695 Guidelines:
696 - The answer NA means that the paper does not involve crowdsourcing nor research with 697 human subjects.

698 - Including this information in the supplemental material is fine, but if the main contribution of 699 the paper involves human subjects, then as much detail as possible should be included in the 700 main paper. 701 - According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or 702 other labor should be paid at least the minimum wage in the country of the data collector.

## 703 15. **Institutional Review Board (Irb) Approvals Or Equivalent For Research With Human Subjects**

704 Question: Does the paper describe potential risks incurred by study participants, whether such 705 risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or 706 an equivalent approval/review based on the requirements of your country or institution) were 707 obtained? 708 Answer: [NA] 709 Justification: See previous point. 710 Guidelines:
711 - The answer NA means that the paper does not involve crowdsourcing nor research with 712 human subjects. 713 - Depending on the country in which research is conducted, IRB approval (or equivalent) may 714 be required for any human subjects research. If you obtained IRB approval, you should 715 clearly state this in the paper. 716 - We recognize that the procedures for this may vary significantly between institutions and 717 locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines 718 for their institution. 719 - For initial submissions, do not include any information that would break anonymity (if 720 applicable), such as the institution conducting the review.

## 721 16. **Declaration Of Llm Usage**

722 Question: Does the paper describe the usage of LLMs if it is an important, original, or non723 standard component of the core methods in this research? Note that if the LLM is used only for 724 writing, editing, or formatting purposes and does not impact the core methodology, scientific 725 rigorousness, or originality of the research, declaration is not required. 726 Answer: [NA] 727 Justification: The core methods developed in this research do not involve LLMs as any important, 728 original, or non-standard components. 729 Guidelines: 730 - The answer NA means that the core method development in this research does not involve 731 LLMs as any important, original, or non-standard components.

732 - Please refer to our LLM policy (https://neurips.cc/Conferences/2025/LLM) for what 733 should or should not be described.