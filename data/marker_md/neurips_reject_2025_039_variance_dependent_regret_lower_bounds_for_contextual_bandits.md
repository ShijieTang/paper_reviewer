# Variance-Dependent Regret Lower Bounds for Contextual Bandits

Anonymous Author(s)

Affiliation Address email

## Abstract

<sup>1</sup> Variance-dependent regret bounds for linear contextual bandits, which improve upon the classical <sup>O</sup>e(<sup>d</sup> √ <sup>K</sup>) regret bound to <sup>O</sup>e(<sup>d</sup> qP<sup>K</sup> <sup>k</sup>=1 σ 2 k <sup>2</sup> ), where d is the context dimension, K is the number of rounds, and σ 2 k <sup>3</sup> is the noise variance in round <sup>4</sup> k, has been widely studied in recent years. However, most existing works focus <sup>5</sup> on the regret upper bounds instead of lower bounds. To our knowledge, the only <sup>6</sup> lower bound is from [Jia et al.](#page-9-0) [\(2024\)](#page-9-0), which proved that for any eluder dimension <sup>d</sup>elu and total variance budget Λ, there exists an instance with P<sup>K</sup> <sup>k</sup>=1 σ 2 <sup>7</sup> <sup>k</sup> ≤ Λ for which any algorithm incurs a variance-dependent lower bound of Ω(√ <sup>8</sup> deluΛ). However, this lower bound has a √ <sup>9</sup> d gap with existing upper bounds. More-<sup>10</sup> over, it only considers a fixed total variance budget Λ and does not apply to a general variance sequence {σ 1 , . . . , σ<sup>2</sup> <sup>11</sup> <sup>K</sup>}. In this paper, to overcome the limita-<sup>12</sup> tions of [Jia et al.](#page-9-0) [\(2024\)](#page-9-0), we consider the general variance sequence under two <sup>13</sup> settings. For a prefixed sequence, where the entire variance sequence is revealed <sup>14</sup> to the learner at the beginning of the learning process, we establish a variancedependent lower bound of Ω(d qP<sup>K</sup> <sup>k</sup>=1 σ 2 k <sup>15</sup> / log K) for linear contextual bandits. For an adaptive sequence, where an adversary can generate the variance σ 2 k <sup>16</sup> in <sup>17</sup> each round k based on historical observations, we show that when the adversary must generate σ 2 k <sup>18</sup> before observing the decision set Dk, a similar lower bound of Ω(d qP<sup>K</sup> <sup>k</sup>=1 σ 2 k / log<sup>6</sup> <sup>19</sup> (dK)) holds. In both settings, our results match the up-<sup>20</sup> per bounds of the SAVE algorithm [\(Zhao et al., 2023\)](#page-9-1) up to logarithmic factors. <sup>21</sup> Furthermore, if the adversary can generate the variance σ<sup>k</sup> after observing the <sup>22</sup> decision set Dk, we construct a counter-example showing that it is impossible <sup>23</sup> to construct a variance-dependent lower bound if the adversary properly selects <sup>24</sup> variances in collaboration with the learner. Our lower bound proofs use a novel <sup>25</sup> peeling technique that groups rounds by variance magnitude. For each group, <sup>26</sup> we construct separate instances and assign the learner distinct decision sets. We <sup>27</sup> believe this proof technique may be of independent interest.

## <sup>28</sup> 1 Introduction

 We consider the linear contextual bandit problem, where each arm is represented by a feature vector and the expected reward is a linear function of this feature vector with an unknown parameter vector. Numerous studies have developed algorithms achieving optimal regret bounds for linear bandits [\(Chu et al., 2011;](#page-9-2) [Abbasi-Yadkori et al., 2011a\)](#page-9-3). However, while these works establish minimax- optimal regret bounds in the worst-case, they do not exploit additional problem-dependent structures. Our work focuses on incorporating reward variance information into the analysis, building upon a line of research studying variance-dependent regret bounds for linear bandits [\(Zhou et al., 2021;](#page-9-4) [Zhang et al., 2021;](#page-9-5) [Zhou and Gu, 2022;](#page-9-6) [Zhao et al., 2022;](#page-9-7) [Kim et al., 2022;](#page-9-8) [Zhao et al., 2023\)](#page-9-1) and general function approximation [\(Jia et al., 2024\)](#page-9-0), which includes linear bandits as a special Submitted to 39th Conference on Neural Information Processing Systems (NeurIPS 2025). Do not distribute.

<sup>38</sup> case. Notably, [Zhao et al.](#page-9-1) [\(2023\)](#page-9-1) established a near-optimal regret guarantee without requiring prior <sup>39</sup> knowledge of the variances:

<sup>40</sup> Theorem 1.1 (Theorem 2.3, [Zhao et al. 2023\)](#page-9-1). For any linear contextual bandit problem, the regret <sup>41</sup> of the SAVE algorithm in the first K rounds is upper bounded by:

$$\text{Regret}(K) \leq \tilde{O}\left(d\sqrt{\sum_{k=1}^K \sigma_k^2} + d\right),$$

where d is the dimension and σ 2 k <sup>42</sup> is the noise variance of the selected action in round k.

 However, most of these works have focused on developing algorithms with regret upper bound guarantees, while variance-dependent lower bounds remain understudied. The only exception is [Jia et al.](#page-9-0) [\(2024\)](#page-9-0), which focuses on general function classes with finite eluder dimension delu and provides the following variance-dependent lower bound:

<sup>47</sup> Theorem 1.2 (Theorem 5.1, [Jia et al. 2024\)](#page-9-0). For any dimension d ≥ 2, action space size A, number <sup>48</sup> of rounds K ≥ 2, and total variance budget Λ ∈ [0, K], there exists a contextual bandit problem with <sup>49</sup> eluder dimension delu = d, action space size A, and an adversarial sequence of variances satisfying P<sup>K</sup> <sup>k</sup>=1 σ 2 <sup>50</sup> <sup>k</sup> ≤ Λ such that for any algorithm, the regret is lower bounded by:

$$\text{Regret}(K) \geq \Omega(\min(\sqrt{d\Lambda} + d, \sqrt{AK})).$$

When restricted to the linear bandit case, where d ≥ √ <sup>A</sup>, the above lower bound reduces to √ <sup>51</sup> dΛ, [w](#page-9-0)hich has a gap of √ <sup>52</sup> d factor compared with the upper bound in [Zhao et al.](#page-9-1) [\(2023\)](#page-9-1). Moreover, [Jia](#page-9-0) <sup>53</sup> [et al.](#page-9-0) [\(2024\)](#page-9-0) only considers instances with a fixed budget Λ and relies on carefully designed variance sequences {σ 2 , σ<sup>2</sup> , . . . , σ<sup>2</sup> <sup>54</sup> <sup>K</sup>}, failing to provide lower bounds for general variance sequences. <sup>55</sup> Therefore, an open question arises:

<sup>56</sup> *Can we prove variance-dependent regret lower bounds for general variance sequences?*

#### <sup>57</sup> 1.1 Our Contributions

<sup>58</sup> In this paper, we answer this question affirmatively by constructing hard-to-learn instances in several different settings. For any prefixed sequence {σ 2 1 , . . . , σ<sup>2</sup> <sup>K</sup>}, we achieve a Ω( e <sup>d</sup> qP<sup>K</sup> <sup>k</sup>=1 σ 2 k <sup>59</sup> ) <sup>60</sup> variance-dependent expected lower bound, which matches the upper bound in [Zhao et al.](#page-9-1) [\(2023\)](#page-9-1) <sup>61</sup> up to logarithmic factors and demonstrates its optimality. For general adaptive variance sequences where a weak adversary (potentially collaborating with the learner) can generate variance σ 2 k <sup>62</sup> in each <sup>63</sup> round k based on historical observations, our instance provides a high-probability lower bound of Ω( e <sup>d</sup> qP<sup>K</sup> <sup>k</sup>=1 σ 2 k <sup>64</sup> ), which also matches the upper bound in [Zhao et al.](#page-9-1) [\(2023\)](#page-9-1) up to logarithmic fac-<sup>65</sup> tors. To the best of our knowledge, this is the first high-probability lower bound for linear contextual <sup>66</sup> bandit.

<sup>67</sup> Our construction and analysis rely on the following new techniques:

 • A peeling technique for prefixed variance sequences that divides rounds into groups based on variance magnitude. Through orthogonal decision set construction, each group only interacts with its corresponding parameters, allowing us to establish separate lower bounds for different variance scales and combine them effectively. • A multi-instance framework that handles unknown group sizes in the adaptive setting. For each variance group, we maintain multiple instances designed for different possible intervals of round numbers and assign the learner to these instances in a cyclic manner, ensuring uniform visits across instances and guaranteeing the visiting times of one instance matches its designed interval. • A high-probability lower bound that handles adaptive group sizes through a union bound. We first convert expected regret bounds to constant-probability bounds through careful variance con- trol and auxiliary algorithms, then boost these to high-probability bounds by creating multiple independent instances.

 Furthermore, we also study the setting with a strong adversary that can generate the variance σ<sup>k</sup> after observing the decision set Dk. Under this scenario, we proposed a counter algorithm that can collaborate with the adversary by properly selecting variance, achieving an O(d) regret even the total variance P<sup>K</sup> <sup>k</sup>=1 σ <sup>k</sup> = Ω(K). This implies that it is impossible to derive a variance-dependent lower bound for general variance sequence with strong adversary. As a direct extension of this result,  we also show that it is impossible to derive a variance-dependent lower bound for stochastic linear bandits, where the decision set is fixed even for a general prefixed variance sequence.

 Notation We use lower case letters to denote scalars, and use lower and upper case bold face letters to denote vectors and matrices respectively. We denote by [n] the set {1, . . . , n}. For a vector x ∈ R d and a positive semi-definite matrix Σ ∈ R d×d , we denote by ∥x∥<sup>2</sup> the vector's ℓ<sup>2</sup> norm and by ∥x∥<sup>Σ</sup> = √ x⊤Σx the Mahalanobis norm. For two positive sequences {an} and {bn} with n = 1, 2, . . . , we write a<sup>n</sup> = O(bn) if there exists an absolute constant C > 0 such that a<sup>n</sup> ≤ Cb<sup>n</sup> holds for all n ≥ 1 and write a<sup>n</sup> = Ω(bn) if there exists an absolute constant C > 0 such that <sup>a</sup><sup>n</sup> ≥ Cb<sup>n</sup> holds for all <sup>n</sup> ≥ <sup>1</sup>. We use <sup>O</sup>e(·) to further hide the polylogarithmic factors. We use <sup>1</sup>{·} to denote the indicator function.

## 2 Related Work

 Heteroscedastic Linear Bandits. For linear bandit problems, the worst-case regret has been widely studied [\(Auer, 2002;](#page-9-9) [Dani et al., 2008;](#page-9-10) [Li et al., 2010;](#page-9-11) [Chu et al., 2011;](#page-9-2) [Abbasi-Yadkori et al., 2011b;](#page-9-12) [Li et al., 2019\)](#page-9-13), achieving <sup>O</sup>e( √ K) bounds in the first K rounds. Recently, a series of works has [c](#page-9-14)onsidered heteroscedastic variants where noise distributions vary across rounds. [Kirschner and](#page-9-14) [Krause](#page-9-14) [\(2018\)](#page-9-14) first formally proposed a linear bandit model with heteroscedastic noise, assuming [σ](#page-9-15)k-sub-Gaussian noise in round k ∈ [K]. Subsequently, [\(Zhou et al., 2021;](#page-9-4) [Zhang et al., 2021;](#page-9-5) [Kim](#page-9-15) [et al., 2021;](#page-9-15) [Zhou and Gu, 2022;](#page-9-6) [Dai et al., 2022;](#page-9-16) [Zhao et al., 2023;](#page-9-1) [Jia et al., 2024\)](#page-9-0) relaxed this to variance-based constraints where round k has variance σ k . Among these works, [Zhou et al.](#page-9-4) [\(2021\)](#page-9-4) and [Zhou and Gu](#page-9-6) [\(2022\)](#page-9-6) obtained near-optimal regret guarantees of <sup>O</sup>e(<sup>d</sup> qP<sup>K</sup> <sup>k</sup>=1 σ k ), but required knowledge of σ<sup>k</sup> after observing the reward in round k. In contrast, [Zhang et al.](#page-9-5) [\(2021\)](#page-9-5); [Kim et al.](#page-9-15) [\(2021\)](#page-9-15) handled unknown variances with computationally inefficient algorithms, achieving a weaker <sup>O</sup>e(poly(d) qP<sup>K</sup> <sup>k</sup>=1 σ k ) bound. Recently, [Zhao et al.](#page-9-1) [\(2023\)](#page-9-1) improved upon these results with an efficient algorithm (SAVE) achieving the near-optimal <sup>O</sup>e(<sup>d</sup> qP<sup>K</sup> <sup>k</sup>=1 σ k ) bound without requiring variance knowledge. Beyond standard linear bandits, two directions have been explored. [Dai et al.](#page-9-16) [\(2022\)](#page-9-16) studied heteroscedastic sparse linear bandits, providing a framework to convert standard algorithms to the sparse setting. In a different direction, [Jia et al.](#page-9-0) [\(2024\)](#page-9-0) extended the analysis to contextual bandits with general function classes having finite eluder dimension, which includes linear bandits as a special case, and achieved a variance-dependent regret upper bounds.

 Lower Bounds for Linear Contextual Bandits. For linear contextual bandit problems, several works [\(Dani et al., 2008;](#page-9-10) [Chu et al., 2011;](#page-9-2) [Li et al., 2019\)](#page-9-13) have established theoretical lower bounds to illustrate the fundamental difficulty in learning process. For linear bandits with finite action sets, [Chu et al.](#page-9-2) [\(2011\)](#page-9-2) established an Ω( e √ dK) lower bound, matching the upper bound up to logarithmic [f](#page-9-10)actors in the action set size and number of rounds K. For general stochastic linear bandits, [Dani](#page-9-10) [et al.](#page-9-10) [\(2008\)](#page-9-10) constructed an instance with 2 Ω(d) actions and obtained an Ω(d √ K) lower bound. Later, [Li et al.](#page-9-13) [\(2019\)](#page-9-13) focused on linear contextual bandits, where the decision set can vary across rounds, and provided an Ω(d √ K log K) lower bound. However, all these works only focus on worst-case regret bounds and do not consider the heteroscedastic variance information. The only exception is [Jia et al.](#page-9-0) [\(2024\)](#page-9-0), which provided an Ω(√ dΛ) variance-dependent lower bound for a fixed total variance budget Λ. Nevertheless, this work cannot handle general variance sequences and leaves open the question of variance-dependent lower bounds in the general setting.

## 3 Preliminaries

[I](#page-9-5)n this work, we consider the heteroscedastic linear contextual bandit [\(Zhou et al., 2021;](#page-9-4) [Zhang](#page-9-5) [et al., 2021\)](#page-9-5), where the noise variance varies across rounds. Let K be the total number of rounds. In each round k ∈ [K], the interaction between the learner and the environment proceeds as follows:

- 1. The environment generates an arbitrary decision set D<sup>k</sup> ⊆ <sup>R</sup> d , where each element repre- sents a feasible action that can be selected by the learner; 2. The learner observes D<sup>k</sup> and selects x<sup>k</sup> ∈ Dk; 3. The environment generates the stochastic noise ϵ<sup>k</sup> and reveals the stochastic reward r<sup>k</sup> = ⟨µ, xk⟩ + ϵ<sup>k</sup> to the learner, where µ ∈ <sup>R</sup> d is the unknown weight vector for the underlying linear reward function.

<sup>136</sup> Without loss of generality, we assume the random noise ϵ<sup>k</sup> in each round k satisfies:

$$\mathbb{P}(|\epsilon_k| \leq R) = 1, \quad \mathbb{E}[\epsilon_k|\mathbf{x}_{1:k}, \epsilon_{1:k-1}] = 0, \quad \mathbb{E}[\epsilon_k^2|\mathbf{x}_{1:k}, \epsilon_{1:k-1}] = \sigma_k^2 \leq 1, \forall k \in [K] \quad (3.1)$$

<sup>137</sup> For any algorithm Alg and linear bandit instance M, the cumulative regret is defined as follows:

$$\text{Regret}_{\text{Alg}}(K, \mathcal{M}) = \sum_{k \in [K]} \langle \mathbf{x}_k, \boldsymbol{\mu} \rangle - \langle \mathbf{x}_k, \boldsymbol{\mu} \rangle, \quad \text{where } \mathbf{x}_k = \underset{\mathbf{x} \in \mathcal{D}_k}{\text{argmax}} \langle \mathbf{x}, \boldsymbol{\mu} \rangle. \quad (3.2)$$

 For simplicity, we may omit the subscripts Alg and/or M when there is no ambiguity. Additionally, with a slight abuse of notation, we may use σ<sup>k</sup> to represent the variance σ 2 k (which is originally the standard deviation) when there is no ambiguity. In this work, we focus on providing variance- dependent lower bounds for the regret based on the variances sequence {σ1, ..., σK}. We consider two settings for the variance sequence {σ1, . . . , σK}:

 • Prefixed Sequence: The variance sequence is revealed to the learner at the beginning of the learning process. • Adaptive Sequence: An adversary (potentially collaborating with the learner) can generate the variance σ<sup>k</sup> in each round k based on historical observations, with the learner receiving each variance at the beginning of the corresponding round. This setting can be further divided into two categories based on the power of the adversary: – Weak Adversary: The adversary must generate the variance σ<sup>k</sup> before observing the decision set Dk. – Strong Adversary: The adversary can generate the variance σ<sup>k</sup> after observing the decision set Dk.

 Remark 3.1. Unlike the typical adversarial setting focused on maximizing regret for a specific algorithm, our work uses the idea of an "adversary" to represent the environment's inherent ability to select the variance sequence. This "adversary" might even strategically choose variance levels (σk) based on the past decision sets D<sup>k</sup> observed so far, potentially leading to variance levels that could temporarily improve the learner's performance or make the learning process appear easier. This seeming "cooperation," however, is ultimately aimed at exploring the fundamental lower bounds on regret that must hold for any learner in any environment. The key is that the variance is chosen without direct knowledge of the true underlying patterns µ. When this "adversary" (our "strong adversary") can adjust the variance based on the learner's actions (Dk), this strategic "cooperation," informed by past observations but blind to µ, becomes more effective in probing the true limits of learnability and challenging our lower bound results.

## <sup>164</sup> 4 Variance-Dependent Lower Bound with Prefixed Variance Sequence

<sup>165</sup> In this section, we consider the setting where the variance sequence {σ1, . . . , σK} is prefixed and <sup>166</sup> fully revealed to the learner at the beginning of the learning process.

#### <sup>167</sup> 4.1 Main Results

<sup>168</sup> We establish the following theorem for the variance-dependent lower bound.

<sup>169</sup> Theorem 4.1. For any dimension d > 1, prefixed sequence of variance {σ1, ..., σK} satisfying P<sup>K</sup> <sup>k</sup>=1 σ 2 <sup>k</sup> ≥ 1 + 384d 2 <sup>170</sup> and algorithm Alg, there exists a hard linear contextual bandit instance such <sup>171</sup> that each action a ∈ D<sup>k</sup> in round k has variance bounded by σk. For this instance, the expected <sup>172</sup> regret of algorithm Alg over K rounds is lower bounded by:

$$\mathbb{E}[\text{Regret}(K)] \geq \Omega\left(d\sqrt{\sum_{i=1}^K \sigma_k^2/(\log K)}\right).$$

<sup>173</sup> Remark 4.2. For a prefixed sequence {σ1, ..., σK}, Theorem [4.1](#page-3-0) shows that any algorithm incurs a regret lower bounded of Ω( e <sup>d</sup> qP<sup>K</sup> <sup>k</sup>=1 σ 2 k <sup>174</sup> ), which matches the upper bound in [Zhao et al.](#page-9-1) [\(2023\)](#page-9-1) up <sup>175</sup> to logarithmic factors. Compared to the lower bound in [Jia et al.](#page-9-0) [\(2024\)](#page-9-0), Theorem [4.1](#page-3-0) focuses on the linear contextual bandit setting and achieves a √ <sup>176</sup> d improvement over the standard linear bandit <sup>177</sup> setting. It is also worth noting that the lower bound in [Jia et al.](#page-9-0) [\(2024\)](#page-9-0) only considers instances with a fixed total variance P<sup>K</sup> <sup>k</sup>=1 σ 2 k <sup>178</sup> , constructed by using constant variance in the early rounds and zero <sup>179</sup> variance in later rounds. In comparison, Theorem [4.1](#page-3-0) applies to any fixed variance sequence and is <sup>180</sup> more flexible.

In Theorem [4.1,](#page-3-0) we require that the total variance is no less than Ω(d 2 ), which reduces to K ≥ Ω(d 2 <sup>181</sup> ) <sup>182</sup> when all variances σ<sup>k</sup> = 1. A similar requirement exists in standard linear bandits, since a trivial lower bound of Ω(K) always holds for any algorithm, and the lower bound of Ω(d √ <sup>183</sup> K) can only be achieved when K ≥ Ω(d 2 <sup>184</sup> ). Furthermore, for general sequences of variances with total variance smaller than O(d <sup>185</sup> ), a large number of rounds K alone is not sufficient to establish the desired <sup>186</sup> lower bound. The presence of early rounds with zero variance would increase the total number of <sup>187</sup> rounds without affecting the fundamental complexity of the problem. This observation suggests that requiring total variance no less than Ω(d 2 <sup>188</sup> ) (or other equivalent conditions) may be necessary for <sup>189</sup> establishing the lower bound.

#### <sup>190</sup> 4.2 Proof of Theorem [4.1](#page-3-0)

<sup>191</sup> In this subsection, we prove the variance-dependent lower bound in Theorem [4.1.](#page-3-0) We first start <sup>192</sup> with a fixed variance threshold σ, and construct a class of hard-to-learn instances where actions are chosen from a hypercube action set A = {−1, 1} d <sup>193</sup> , and for any action a ∈ A, the reward follows a scaled Bernoulli distribution σ ·B(1/3 +⟨µ, a⟩), where ∆ = 1/ √ 96K and µ ∈ {−∆, ∆} d <sup>194</sup> . In this setting, the variance for each action is upper bounded by σ 2 <sup>195</sup> , and these instances can be represented as a linear bandit problem with feature (σ, σ · a) and weight vector µ <sup>196</sup> ′ = (1/3, µ). Based on these <sup>197</sup> hard-to-learn instances, we have the following variance-dependent lower bound for the regret:

<sup>198</sup> Lemma 4.3. For a fixed variance threshold σ and any bandit algorithm Alg, if the weight vector µ ∈ {−∆, ∆} d is uniformly random selected from {−∆, ∆} d <sup>199</sup> , the variance in each round is bounded by σ 2 , and the expected regret over K ≥ 1.5 · d 2 <sup>200</sup> rounds is lower bounded by:

$$\mathbb{E}_{\mu}[\text{Regret}(K)] \geq d\sqrt{K\sigma^2}/8\sqrt{6}.$$

 Remark 4.4. Lemma [4.3](#page-4-0) establishes a variance-dependent lower bound for the regret with a fixed variance threshold σ. When all variances are equal (σ<sup>1</sup> = ... = σ<sup>K</sup> = σ), this bound matches the upper bound in [Zhao et al.](#page-9-1) [\(2023\)](#page-9-1) up to logarithmic factors. In addition, under this fixed-variance setting, this lemma provides a tighter logarithmic dependency on the number of rounds K compared to Theorem [4.1,](#page-3-0) though it does not extend to dynamic variances.

<sup>206</sup> Now, for any prefixed variance sequence {σ1, ..., σK}, we divide the rounds into L = ⌈log<sup>2</sup> K⌉ + 1 <sup>207</sup> different groups based on the range of their variance as follows:

$$\mathcal{K}_0 = \{k : \sigma_k \leq 1/K\}, \quad \mathcal{K}_i = \{k : 2^{i-1}/K < \sigma_k \leq 2^i/K\}, \quad \text{for } i = 1, \dots, L-1.$$

<sup>208</sup> For each group K<sup>i</sup> with i ∈ [L − 1], we construct a bandit instance M<sup>i</sup> with weight vector µ<sup>i</sup> <sup>209</sup> following Lemma [4.3,](#page-4-0) where:

- the variance threshold is set to be σ(i) = 2<sup>i</sup>−<sup>1</sup> <sup>210</sup> /K;
- the number of rounds is K<sup>i</sup> = |K<sup>i</sup> <sup>211</sup> |; <sup>212</sup> • the dimension is d<sup>i</sup> = d/L.

 For group K0, we construct a different type of instance M0: a d/L-armed bandit, where one ran- domly chosen arm gives constant reward 1 while all other arms give reward 0. Note that this instance in M<sup>0</sup> can be equivalently represented as a d<sup>0</sup> = d/L-dimensional linear bandit where actions are one-hot vectors e<sup>i</sup> <sup>216</sup> .

<sup>217</sup> Based on these sub-instances, we create a combined linear bandit instance with dimension <sup>218</sup> d<sup>0</sup> + d<sup>1</sup> + ... + dL−<sup>1</sup> = d with weight vector µ = (µ0, ..., µL−1): At the beginning of each round <sup>k</sup>, if round <sup>k</sup> belongs to group <sup>K</sup><sup>i</sup> <sup>219</sup> , then the learner receives the decision set D<sup>k</sup> = (0<sup>d</sup><sup>0</sup> , ..., 0<sup>d</sup>i−<sup>1</sup> , x, 0<sup>d</sup>i+1 , ..., 0<sup>d</sup>L−<sup>1</sup> ) : x ∈ A<sup>i</sup> , where 0<sup>d</sup><sup>j</sup> <sup>220</sup> corresponds to a zero vector with dimension d<sup>j</sup> and A<sup>i</sup> is the action set in the bandit instance M<sup>i</sup> <sup>221</sup> . Under this construction, for any round k ∈ K<sup>i</sup> , the reward in the combined instance coincides with that of sub-instance M<sup>i</sup> <sup>222</sup> . Specifically, <sup>223</sup> after the learner selects action x, they receive a reward drawn from a scaled Bernoulli distribution with variance upper bounded by σ 2 (i) = 2 <sup>i</sup>−<sup>1</sup>/K<sup>2</sup> <sup>224</sup> for i ̸= 0, and variance 0 for i = 0. Note that in all groups, the variance is bounded by σ 2 k <sup>225</sup> . With this construction in hand, we now proceed to <sup>226</sup> prove the lower bound in Theorem [4.1.](#page-3-0)

<sup>227</sup> Remark 4.5 (Linear Contextual Bandits vs. Stochastic Linear Bandits). In the proof of Theorem <sup>228</sup> [4.1,](#page-3-0) we heavily rely on assigning different decision sets to rounds in the contextual bandit envi-<sup>229</sup> ronment. This approach, however, does not extend to stochastic linear bandit problems, where all <sup>230</sup> rounds share the same decision set. To see this limitation, consider any prefixed variance sequence <sup>231</sup> with σ<sup>1</sup> = · · · = σ<sup>d</sup> = 0. In this case, the learner can select canonical basis of the decision set in <sup>232</sup> the first d rounds. Since these rounds have zero variance, the learner learns the exact rewards for <sup>233</sup> all actions in the decision set and incurs no regret in subsequent rounds, regardless of the values of <sup>σ</sup>d+1, . . . , σK. Consequently, it is impossible to establish a lower bound of Ω( e <sup>d</sup> qP<sup>K</sup> <sup>k</sup>=1 σ 2 k <sup>234</sup> ) in this <sup>235</sup> setting.

*Proof of Theorem [4.1.](#page-3-0)* Due to the orthogonal construction of decision sets across different groups K<sup>i</sup> <sup>237</sup> , actions in group K<sup>i</sup> provide no information about the weight vector µ<sup>j</sup> for j ̸= i. Consequently, the total regret can be decomposed into the sum of regrets from each sub-instance. For each sub-instance M<sup>i</sup> with i ̸= 0, the regret is lower bounded by:

$$\begin{aligned}\mathbb{E}_{\boldsymbol{\mu}_i} \left[ \sum_{k \in \mathcal{K}_i} \max_{\mathbf{x} \in \mathcal{D}_k} \langle \boldsymbol{\mu}_i, \mathbf{x} \rangle - \langle \boldsymbol{\mu}_i, \mathbf{x}_k \rangle \right] &\geq \mathbb{I}(K_i \geq 1.5d_i^2) \cdot \frac{d_i \sqrt{K_i \sigma^2(i)}}{8\sqrt{6}} \\ &\geq \frac{d_i \sqrt{K_i \sigma^2(i)}}{8\sqrt{6}} - \frac{d_i \sqrt{1.5d_i^2 \cdot \sigma^2(i)}}{8\sqrt{6}} \\ &\geq \frac{d_i \sqrt{\sum_{k \in \mathcal{K}_i} \sigma_k^2}}{16\sqrt{6}} - \frac{d_i^2 \cdot \sigma(i)}{16},\end{aligned}\tag{4.1}$$

<sup>240</sup> where the first inequality follows from Lemma [4.3,](#page-4-0) the second inequality holds due to <sup>I</sup>(x ≥ y) √ x ≥ √ x − √<sup>y</sup>, and the last inequality follows from the definition of group <sup>K</sup><sup>i</sup> <sup>241</sup> . <sup>242</sup> Taking a summation of [\(4.1\)](#page-5-0) over all groups, the total regret can be lower bounded as follows:

$$\begin{aligned}\mathbb{E}_{\boldsymbol{\mu}}[\text{Regret}(K)] &= \sum_{i=0}^{L-1} \mathbb{E}_{\boldsymbol{\mu}_i} \left[ \sum_{k \in \mathcal{K}_i} \max_{\mathbf{x} \in \mathcal{D}_k} \langle \boldsymbol{\mu}_i, \mathbf{x} \rangle - \langle \boldsymbol{\mu}_i, \mathbf{x}_k \rangle \right] \\ &\geq \sum_{i=1}^{L-1} \frac{d_i \sqrt{\sum_{k \in \mathcal{K}_i} \sigma_k^2}}{16\sqrt{6}} - \frac{d_i^2 \cdot \sigma(i)}{16} \\ &\geq \sum_{i=1}^{L-1} \frac{d \sqrt{\sum_{k \in \mathcal{K}_i} \sigma_k^2}}{16\sqrt{6}L} - \frac{d^2}{4L^2} \\ &\geq \frac{d \sqrt{\sum_{i=1}^{L-1} \sum_{k \in \mathcal{K}_i} \sigma_k^2}}{16\sqrt{6}L} - \frac{d^2}{4L^2},\end{aligned}\tag{4.2}$$

<sup>243</sup> where the first inequality follows from [\(4.1\)](#page-5-0), the second inequality follows from the definition of variance threshold σ(i) and dimension d<sup>i</sup> = d/L, and the last inequality holds due to P i √ 244 p x<sup>i</sup> ≥ P i x<sup>i</sup> <sup>245</sup> . In addition, for the group K0, we have

$$\sum_{k \in \mathcal{K}_0} \sigma_k^2 \leq \sum_{k \in \mathcal{K}_0} 1/K \leq 1, \quad (4.3)$$

<sup>246</sup> where the first inequality follows from the definition of group K<sup>0</sup> and the second inequality follows <sup>247</sup> from |K0| ≤ K. Therefore, we have

$$\begin{aligned}\mathbb{E}_{\boldsymbol{\mu}}[\text{Regret}(K)] &\geq \frac{d\sqrt{\sum_{i=1}^{L-1} \sum_{k \in \mathcal{K}_i} \sigma_k^2}}{16\sqrt{6L}} - \frac{d^2}{4L^2} \\ &\geq \frac{d\sqrt{\sum_{k=1}^K \sigma_k^2 - 1}}{16\sqrt{6L}} - \frac{d^2}{4L^2} \\ &\geq \frac{d\sqrt{\sum_{k=1}^K \sigma_k^2 - 1}}{32\sqrt{6L}},\end{aligned}$$

<sup>248</sup> where the first inequality follows from [\(4.2\)](#page-5-1), the second inequality follows from [\(4.3\)](#page-5-2), and the last inequality follows from the fact that P<sup>K</sup> <sup>k</sup>=1 σ 2 <sup>k</sup> ≥ 1 + 384d <sup>249</sup> . Thus, we complete the proof of <sup>250</sup> Theorem [4.1.](#page-3-0)

## 5 Variance-Dependent Lower Bounds with Adaptive Variance Sequence

 In the previous section, we focused on the setting where the variance sequence is prefixed and revealed to the learner at the beginning of the learning process. In this section, we extend our analysis to the setting where the variance sequence can be adaptive based on historical observations, with the learner receiving the adaptive variance at the beginning of each round.

#### 5.1 Main Results

#### 5.1.1 Weak Adversary

 We first describe the learning process and the mechanism of variance adaptation. In detail, the adaptive variance process proceeds as follows:

 1. At the beginning of each round k, a (weak) adversary selects the variance level σ<sup>k</sup> based on the historical observations, including actions {a1, . . . , ak−1}, rewards {r1, . . . , rk−1}, and decision sets {D1, D2, . . . , Dk−1}. The adversary has access to all historical information but not to the underlying reward model parameters; 2. Given the selected variance level σk, we construct and assign a decision set D<sup>k</sup> to the learner, where the variance of the reward for each action a ∈ D<sup>k</sup> is bounded by σ k ; 3. The learner observes the decision set D<sup>k</sup> and variance level σk, then determines an action a<sup>k</sup> from D<sup>k</sup> based on its historical observations and current information. After selecting the action, the learner receives a reward r<sup>k</sup> with variance bounded by σ k .

 Remark 5.1. It is worth noting that our concept of adversary differs from the weak/strong adversary in [Jia et al.](#page-9-0) [\(2024\)](#page-9-0). Specifically, [Jia et al.](#page-9-0) [\(2024\)](#page-9-0) considers an adversary that attempts to hinder the learner's learning by allocating a fixed total variance budget P<sup>K</sup> <sup>k</sup>=1 σ <sup>k</sup> ≤ Λ across rounds to max- imize regret. In contrast, our work considers an adversary that attempts to break the lower bounds themselves by collaborating with the learner. To prevent such exploitation, we must restrict the ad- versary from knowing the weight vector of the underlying reward model. Without this restriction, the adversary could encode each entry µ<sup>i</sup> of the weight vector µ through the corresponding variance σ<sup>i</sup> = µ<sup>i</sup> , allowing the learner to learn the weight vector after d rounds.

Under this setting, we establish the following theorem for the variance-dependent lower bound.

Theorem 5.2 (Weak Adversary). For any dimension d > 1, adaptive sequence of variances {σ1, . . . , σK} and algorithm Alg, there exists a hard instance such that each action a ∈ D<sup>k</sup> in round k has variance bounded by σ k . For this instance, if P<sup>K</sup> <sup>k</sup>=1 σ <sup>k</sup> ≥ Ω(d ), then with probability at least 1 − 1/K, the regret of algorithm Alg over K rounds is lower bounded by:

$$\text{Regret}(K) \geq \Omega\left(d\sqrt{\sum_{k=1}^K \sigma_k^2 / \log^6(dK)}\right).$$

Remark 5.3. Theorem [5.2](#page-6-0) provides a high-probability lower bound of <sup>Ω</sup>e d qP<sup>K</sup> <sup>k</sup>=1 σ k , which matches the upper bound in [Zhao et al.](#page-9-1) [\(2023\)](#page-9-1) up to logarithmic factors, albeit with looser logarith- mic dependencies than Theorem [4.1](#page-3-0) due to the adaptive nature of the variance sequence. Unlike the expected lower bound in Theorem [4.1,](#page-3-0) for adaptive variance sequences, the cumulative variance P<sup>K</sup> <sup>k</sup>=1 σ k depends on the random process and observations. This dependence makes it challenging to establish an expected variance-dependent regret bound - a fundamental difficulty that does not arise for standard d √ K-type lower bounds in linear contextual bandits. To the best of our knowledge, our result provides the first high-probability lower bound for linear contextual bandit.

#### 5.1.2 Strong Adversary

 In Theorem [5.2,](#page-6-0) we require that for each round k ∈ [K], all actions x ∈ D<sup>k</sup> share the same adaptive variance σk. This is more restrictive than the setting in [Zhao et al.](#page-9-1) [\(2023\)](#page-9-1), where the variance can differ across actions x ∈ Dk. However, extending our lower bound to action-dependent variances is not possible in the adaptive setting. This limitation arises because we construct the decision set D<sup>k</sup> after the adversary chooses the variance σk, which prevents assigning specific variances to individual actions x ∈ Dk. Moreover, we now consider a strong adversary that can choose σ<sup>k</sup> after observing the decision set Dk. The interaction between the learner and this strong adversary proceeds as follows:

 1. At the beginning of each round k, we construct and assign a decision set D<sup>k</sup> based on historical observations, including actions {a1, . . . , ak−1} and rewards {r1, . . . , rk−1};

 2. Given the decision set D<sup>k</sup> in round k, the strong adversary selects the variance level σ<sup>k</sup> for round k. The adversary has access to all historical information but not to the underlying reward model parameters; 3. The learner observes the decision set D<sup>k</sup> and variance level σk, then determines an action a<sup>k</sup> from D<sup>k</sup> based on its historical observations and current information. After selecting the action, the learner receives a reward r<sup>k</sup> with variance bounded by σ k . The following theorem shows that under this setting, the adversary could cooperate with the learner to break the lower bound. Theorem 5.4 (Strong Adversary). For any linear contextual bandit problem and number of rounds K ≥ 2d, if we first provide the decision set D<sup>k</sup> and then allow an adversary to choose the variance σ<sup>k</sup> based on the decision set Dk, there exists one such type of adversary such that, there exists an algorithm whose regret in the first K rounds is upper bounded by Regret(K) ≤ d, where the total variance P<sup>K</sup> <sup>k</sup>=1 σ <sup>k</sup> ≥ K/2. Remark 5.5. Theorem [5.4](#page-7-0) highlights why Theorem [5.2](#page-6-0) requires a weak adversary that set the vari- ance sequence before seeing the learner's choices. If the adversary could see the decision set first, it could potentially choose variances that would invalidate our lower bound. This finding underscores that our construction is precise and pinpoints the exact condition under which the derived lower bound holds. Remark 5.6. It is worth noting that [Jia et al.](#page-9-0) [\(2024\)](#page-9-0) also considered the case where the adver- sary assigns variances to actions after observing the decision set and action choice, and provided a variance-dependent lower bound. However, their analysis focuses on an adversary that allocates variance across rounds to maximize the regret. In contrast, our work considers an adversary that attempts to break these bounds, making it more challenging to establish lower bounds for general variance sequences. It is also worth noting that if the adversary's goal is to increase regret, choosing a prefixed sequence is a viable strategy. This case is already covered by our Theorem [4.1](#page-3-0) for prefixed sequences, which provides a tighter lower bound than Theorem [5.2.](#page-6-0) Theorem [5.4](#page-7-0) suggests that it is impossible to derive a variance-dependent lower bound if the ad- versary can determine the variance σ<sup>k</sup> after observing the decision set Dk, which further precludes establishing a lower bound when the adversary has the ability to assign action-dependent variances for each action x ∈ D<sup>k</sup> after observing the decision set Dk. This result naturally extends to stochas- tic linear bandit problems, where the decision set D remains fixed across all rounds. In this case, since the adversary knows the decision set D<sup>k</sup> = D in advance, Theorem [5.4](#page-7-0) directly implies: Corollary 5.7. For any stochastic linear bandit problem with fixed decision set D and number of rounds K ≥ 2d, there exists a prefixed sequence {σ1, . . . , σK} such that there exists an algorithm whose regret in the first K rounds is upper bounded by: RegretAlg (K) ≤ d, where the total variance P<sup>K</sup> <sup>k</sup>=1 σ <sup>k</sup> ≥ K/2. 5.2 Proof Sketch of Theorem [5.2](#page-6-0) In this section, we provide the proof sketch of Theorem [5.2.](#page-6-0) Overall, the proof follows a similar structure as Theorem [4.1,](#page-3-0) where we divide the rounds into several groups based on their variance magnitude and create hard instances for each group. The key idea is to calculate individual regret bounds for each group and combine them for the final lower bound. However, there exist several challenges when dealing with adaptive variance sequences that require careful handling. Varying Size of Groups K<sup>i</sup> As discussed in Section [4.2,](#page-4-1) for each group K<sup>i</sup> , we create individual instance M<sup>i</sup> with fixed variance threshold σ(i) = 2<sup>i</sup>−<sup>1</sup> /K and establish a lower bound of Ω( e <sup>d</sup><sup>i</sup> p σ (i)|K<sup>i</sup> |) on the expected regret. However, the construction of such instances relies on prior knowledge of the number of rounds |K<sup>i</sup> |, which can be calculated at the beginning for a pre- fixed variance sequence {σ1, . . . , σK}. In contrast, for general adaptive variance sequences, the number of rounds |K<sup>i</sup> | is not known a priori and can even be a random variable, which creates a barrier in constructing these instances. To address the unknown number of rounds |K<sup>i</sup> |, instead of constructing a single instance M<sup>i</sup> for each group, we create L instances Mi,j , where L = ⌈log<sup>2</sup> K⌉ + 1. Each instance Mi,j is designed for a specific range of round numbers, specifically Mi,j for 2 <sup>j</sup>−<sup>1</sup> ≤ |K<sup>i</sup> | < 2 j . For each round k in group K<sup>i</sup> , the learner receives a decision set D<sup>i</sup> from one of the instances in {Mi,1, . . . ,Mi,L} in a cyclic manner. Through this sequential assignment, the number of visits to

each instance Mi,j is |K<sup>i</sup> |/L. Consequently, we expect that the instance Mi,j corresponding to the

true range 2 <sup>j</sup>−<sup>1</sup> ≤ |K<sup>i</sup> | < 2 <sup>j</sup> provides a lower bound of Ω( e <sup>d</sup><sup>i</sup> p σ <sup>2</sup>(i)|K<sup>i</sup> |) = Ω( e <sup>d</sup><sup>i</sup> p σ <sup>2</sup>(i) · 2 j <sup>356</sup> ), which leads to the final lower bound of Ω( e <sup>d</sup> qP<sup>K</sup> <sup>k</sup>=1 σ k <sup>357</sup> ).

<sup>358</sup> Converting Expected Lower Bound to High-Probability Lower Bound. Another challenge is establishing the lower bound for the triggered instance Mi,j corresponding to the true range 2 <sup>359</sup> <sup>j</sup>−<sup>1</sup> ≤ |K<sup>i</sup> | < 2 j <sup>360</sup> . Traditional analysis of lower bounds in linear contextual bandits has focused on the <sup>361</sup> expected regret. However, when dealing with adaptive variance sequences, this approach becomes <sup>362</sup> insufficient as the adversary can dynamically adjust the variance sequence to break these bounds.

For instance, an adversary might continuously set <sup>σ</sup><sup>k</sup> = 1 until the lower bound of Ω( e <sup>d</sup> qP<sup>k</sup> <sup>i</sup>=1 σ i <sup>363</sup> ) <sup>364</sup> is violated at some round k, then switch to σ<sup>k</sup> = 0 for all future rounds, causing the total variance sum P<sup>K</sup> <sup>k</sup>=1 σ 2 k <sup>365</sup> to remain unchanged. In our construction, this means all rounds could fall into group <sup>366</sup> KL, allowing the adversary to adaptively change the number of rounds between different intervals 2 <sup>j</sup>−<sup>1</sup> ≤ |KL| < 2 j <sup>367</sup> . Since the failure of the lower bound in any single instance ML,j leads to failure <sup>368</sup> of the whole construction, an expected lower bound on regret cannot guarantee robust performance <sup>369</sup> against adaptive sequences. This necessitates a stronger high-probability lower bound that holds <sup>370</sup> uniformly for all instances.

Unfortunately, an expectation of Ω( e <sup>d</sup><sup>i</sup> p σ <sup>2</sup>(i)2<sup>j</sup> <sup>371</sup> ) in instance Mi,j only implies a low-probability regret Regret ≥ Ω( e <sup>d</sup><sup>i</sup> p σ <sup>2</sup>(i)2<sup>j</sup> ) ≥ d<sup>i</sup> ·2 −j/2 <sup>372</sup> , since the cumulative regret in K<sup>i</sup> can be up to σ(i)· |K<sup>i</sup> <sup>373</sup> | in our instance. To solve this problem, we introduce an auxiliary algorithm that automatically <sup>374</sup> detects the cumulative regret and switches to the standard OFUL algorithm [\(Abbasi-Yadkori et al.,](#page-9-3) [2011a\)](#page-9-3) if the cumulative regret is larger than Ω(d<sup>i</sup> p σ <sup>2</sup>(i)2<sup>j</sup> ). [1](#page-0-0) <sup>375</sup> For this auxiliary algorithm, we can guarantee that the upper bound is at most Ω( e <sup>d</sup><sup>i</sup> p σ <sup>2</sup>(i)2<sup>j</sup> <sup>376</sup> ) while maintaining the same probability of high regret as the original algorithm. Therefore, an expectation of Ω( e <sup>d</sup><sup>i</sup> p σ <sup>2</sup>(i)2<sup>j</sup> <sup>377</sup> ) in instance Mi,j implies a constant-probability regret P Regret ≥ Ω( e <sup>d</sup><sup>i</sup> p σ <sup>2</sup>(i)2<sup>j</sup> ) <sup>378</sup> = Ω(1).

<sup>379</sup> After constructing an instance with constant-probability lower bound, we boost this probability by creating Ω log<sup>2</sup> (dK) <sup>380</sup> independent instances. When the learner encounters instance Mi,j , it is <sup>381</sup> assigned to one of these instances in a cyclic manner. Through this construction, with probability at least <sup>1</sup> − <sup>1</sup>/poly(K), the final regret is lower bounded by Regret ≥ Ω( e <sup>d</sup><sup>i</sup> p σ <sup>2</sup>(i)2<sup>j</sup> <sup>382</sup> ).

 Remark 5.8. Unlike previous lower bounds for linear bandit problems which focus on expected regret, to the best of our knowledge, our result provides the first high-probability lower bound for linear contextual bandits. It is worth noting that our construction requires separate decision sets across different rounds in the random assignment process. For stochastic linear bandits with a fixed decision set, we can only derive a constant-probability lower bound. Moreover, for a fixed decision set in stochastic linear bandit problem with covering number log N ≤ <sup>O</sup>e(d), an algorithm can randomly select one action from the covering set and perform this action in all rounds. In this case, there exists a probability of 1/N = 1/ exp(d) to achieve zero regret, which precludes the possibility of establishing high-probability lower bounds for large round numbers K. More details about the high-probability lower bound can be found in Section [5.2.](#page-6-0)

## <sup>393</sup> 6 Conclusion and Future Work

 In this paper, we study variance-dependent lower bounds for linear contextual bandits in different settings. For both prefixed and adaptive variance sequences with weak adversary, we establish tight lower bounds matching the upper bounds in [Zhao et al.](#page-9-1) [\(2023\)](#page-9-1) up to logarithmic factors. We further demonstrate a fundamental limitation: when a strong adversary can select variances after observ- ing decision sets, it becomes impossible to establish meaningful variance-dependent lower bounds. However, our work has focused exclusively on linear bandit settings, while [Jia et al.](#page-9-0) [\(2024\)](#page-9-0) has established variance-dependent lower bounds for general function approximation with a fixed total variance budget Λ. Therefore, we leave for future work the generalization of our analysis of general variance sequence to contextual bandits with general function approximation.

<sup>1</sup> In general settings, detecting cumulative regret is impossible as the learner lacks prior knowledge of the optimal reward and variance. However, in our lower bound construction, all instances are randomly selected from instance classes sharing the same optimal reward and variance, which are known to the learner. This knowledge enables the construction of the auxiliary algorithm.

## References


[1] ABBASI-YADKORI, Y., PAL´ , D. and SZEPESVARI ´ , C. (2011a). Improved algorithms for linear stochastic bandits. In *Advances in Neural Information Processing Systems*. ABBASI-YADKORI, Y., PAL´ , D. and SZEPESVARI ´ , C. (2011b). Improved algorithms for linear stochastic bandits. In *NIPS*, vol. 11. AUER, P. (2002). Using confidence bounds for exploitation-exploration trade-offs. *Journal of Machine Learning Research* 3 397–422. CESA-BIANCHI, N. and LUGOSI, G. (2006). *Prediction, learning, and games*. Cambridge univer- sity press. CHU, W., LI, L., REYZIN, L. and SCHAPIRE, R. (2011). Contextual bandits with linear payoff functions. In *Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics*. JMLR Workshop and Conference Proceedings. DAI, Y., WANG, R. and DU, S. S. (2022). Variance-aware sparse linear bandits. *arXiv preprint arXiv:2205.13450* . DANI, V., HAYES, T. P. and KAKADE, S. M. (2008). Stochastic linear optimization under bandit feedback . JIA, Z., QIAN, J., RAKHLIN, A. and WEI, C.-Y. (2024). How does variance shape the regret in contextual bandits? *arXiv preprint arXiv:2410.12713* . KIM, Y., YANG, I. and JUN, K.-S. (2021). Improved regret analysis for variance-adaptive linear bandits and horizon-free linear mixture mdps. *arXiv preprint arXiv:2111.03289* . KIM, Y., YANG, I. and JUN, K.-S. (2022). Improved regret analysis for variance-adaptive lin- ear bandits and horizon-free linear mixture mdps. *Advances in Neural Information Processing Systems* 35 1060–1072. KIRSCHNER, J. and KRAUSE, A. (2018). Information directed sampling and bandits with het- eroscedastic noise. In *Conference On Learning Theory*. PMLR. LI, L., CHU, W., LANGFORD, J. and SCHAPIRE, R. E. (2010). A contextual-bandit approach to personalized news article recommendation. In *Proceedings of the 19th international conference on World wide web*. LI, Y., WANG, Y. and ZHOU, Y. (2019). Nearly minimax-optimal regret for linearly parameterized bandits. In *Conference on Learning Theory*. PMLR. ZHANG, Z., YANG, J., JI, X. and DU, S. S. (2021). Improved variance-aware confidence sets for linear bandits and linear mixture mdp. *Advances in Neural Information Processing Systems* 34 4342–4355. ZHAO, H., HE, J., ZHOU, D., ZHANG, T. and GU, Q. (2023). Variance-dependent regret bounds for linear bandits and reinforcement learning: Adaptivity and computational efficiency. In *The Thirty Sixth Annual Conference on Learning Theory*. PMLR. ZHAO, H., ZHOU, D., HE, J. and GU, Q. (2022). Bandit learning with general function classes: Heteroscedastic noise and variance-dependent regret bounds. *arXiv preprint arXiv:2202.13603* . ZHOU, D. and GU, Q. (2022). Computationally efficient horizon-free reinforcement learning for linear mixture mdps. *Advances in neural information processing systems* 35 36337–36349. ZHOU, D., GU, Q. and SZEPESVARI, C. (2021). Nearly minimax optimal reinforcement learning for linear mixture markov decision processes. In *Conference on Learning Theory*. PMLR.
## <sup>445</sup> A Proof of Theorem [5.2](#page-6-0)

<sup>446</sup> In this section, we prove the variance-dependent lower bound for adaptive variance sequences es-<sup>447</sup> tablished in Theorem [5.2.](#page-6-0) We begin with the instance construction from Lemma [4.3](#page-4-0) and establish <sup>448</sup> the following constant-probability lower bound for the regret:

Lemma A.1. For a fixed variance threshold σ, number of rounds K ≥ 1.5d 2 <sup>449</sup> , and any bandit algorithm Alg, for the instance constructed in Lemma [4.3,](#page-4-0) with probability at least Ω 1/ log(dK) <sup>450</sup> , <sup>451</sup> the regret is lower bounded by

$$\text{Regret}(K) \geq \frac{d\sqrt{K\sigma^2}}{16\sqrt{6}}.$$

<sup>452</sup> Based on the constant-probability lower bound, we boost this probability by creating L = Ω log<sup>2</sup> (dK) independent instances with dimension d <sup>453</sup> ′ = d/L and number of rounds K′ = K/L, <sup>454</sup> where each instance follows the structure in Lemma [4.3](#page-4-0) with i.i.d. sampled weight vectors. Un-<sup>455</sup> der this construction, the total dimension of all instances is d, which can be represented as a d-<sup>456</sup> dimensional linear contextual bandit through orthogonal embedding, similar to our previous con-<sup>457</sup> struction: for instance i, we augment its actions by padding zeros in dimensions reserved for other <sup>458</sup> instances, ensuring actions from different instances only interact with their corresponding param-<sup>459</sup> eters. Here, we consider the case where the learner visits the instances in a cyclic manner and <sup>460</sup> establish the following high-probability regret lower bound for the constructed instance:

Lemma A.2. For a fixed variance threshold σ, number of rounds K ≥ 1.5d 2 <sup>461</sup> , and any bandit algorithm Alg, with probability at least Ω 1/ log(dK) <sup>462</sup> , the regret is lower bounded by

$$\text{Regret}(K) \geq \Omega(d\sqrt{K\sigma^2}/\log^3(dK)).$$

 With the help of this high-probability lower regret bound from Lemma [A.2,](#page-10-0) we begin the proof of Theorem [5.2.](#page-6-0) Following a similar framework to the fixed-variance case, we first divide the rounds into groups based on their variance magnitude. Specifically, for any variance sequence {σ1, . . . , σK}, we partition the rounds into L = ⌈log<sup>2</sup> K⌉ + 1 groups as follows:

$$\begin{aligned}\mathcal{K}_0 &= \{k : \sigma_k \leq 1/K\}, \\ \mathcal{K}_i &= \{k : 2^{i-1}/K < \sigma_k \leq 2^i/K\}, \quad \text{for } i = 1, \dots, L-1.\end{aligned}$$

To address the unknown number of rounds K<sup>i</sup> = |K<sup>i</sup> <sup>467</sup> |, instead of constructing a single instance <sup>468</sup> M<sup>i</sup> for each group, we create L instances Mi,j , where L = ⌈log<sup>2</sup> K⌉ + 1. Each instance Mi,j is constructed according to Lemma [A.2](#page-10-0) with dimension d ′ = d/L<sup>2</sup> , variance σ(i) = 2<sup>i</sup>−<sup>1</sup> <sup>469</sup> /K and number of rounds K′ = 2j−<sup>1</sup> . For each round k in group K<sup>i</sup> <sup>470</sup> , the learner receives a decision set D<sup>i</sup> <sup>471</sup> from one of the instances in {Mi,1, . . . ,Mi,L} in a cyclic manner.

<sup>472</sup> *Proof of Theorem [5.2.](#page-6-0)* According to Lemma [A.2,](#page-10-0) for each instance Mi,j , with probability at least 1 − 1/K<sup>3</sup> , the regret in the first 2 j−1 <sup>473</sup> visits is lower bounded by

$$\text{Regret}(2^{j-1}, \mathcal{M}_{i,j}) \geq \mathbb{I}(2^{j-1} \geq 1.5d'^2) \cdot \Omega(d' \sqrt{2^{j-1} \sigma^2(i)} / \log^3(d' K')), \quad (\text{A.1})$$

where the indicator reflects the requirement that K′ = 2j−<sup>1</sup> ≥ 1.5d ′2 <sup>474</sup> . For simplicity, we define E <sup>475</sup> as the event that [\(A.1\)](#page-10-1) holds for all instances Mi,j . By union bound, we have <sup>P</sup>(E) ≥ 1 − 1/K. Conditioned on event E, for an adaptive sequence and each corresponding group K<sup>i</sup> <sup>476</sup> , due to the cyclic visiting pattern, each instance Mi,j is visited |K<sup>i</sup> <sup>477</sup> |/L times. There exists an instance Mi,j with matching interval for the round number, i.e., 2 <sup>j</sup>−<sup>1</sup> ≤ |K<sup>i</sup> |/L ≤ 2 j <sup>478</sup> . Therefore, we have

$$\begin{aligned} & \sum_{k \in \mathcal{K}_i} \max_{\mathbf{x} \in \mathcal{D}_k} \langle \boldsymbol{\mu}_i, \mathbf{x} \rangle - \langle \boldsymbol{\mu}_i, \mathbf{x}_k \rangle \\ & \geq \text{Regret}(2^{j-1}, \mathcal{M}_{i,j}) \\ & \geq \mathbb{I}(2^{j-1} \geq 1.5d'^2) \cdot \Omega(d\sqrt{2^{j-1}\sigma^2(i)}/\log^3(d'K')) \\ & \geq \mathbb{I}(K_i \geq 3d'^2 L) \cdot \Omega(d\sqrt{K_i\sigma^2(i)}/\log^4(dK)) \\ & \geq \Omega\left(d'\sqrt{K_i\sigma^2(i)}/\log^3(dK) - d'\sqrt{3d'^2L\sigma^2(i)}/\log^4(dK)\right) \end{aligned}$$

$$\geq \Omega\left(d' \sqrt{\sum_{k \in \mathcal{K}_i} \sigma_k^2 / \log^4(dK)} - \sqrt{3L}d'^2 \cdot \sigma(i) / \log^4(dK)\right), \quad (\text{A.2})$$

where the first inequality follows from 2 <sup>j</sup>−<sup>1</sup> ≤ |K<sup>i</sup> |/L ≤ 2 j <sup>479</sup> , the second inequality holds by the definition of event E, the third inequality follows from 2 <sup>j</sup>−<sup>1</sup> ≤ |K<sup>i</sup> |/L ≤ 2 j <sup>480</sup> , the fourth inequality holds due to <sup>I</sup>(x ≥ y) √ x ≥ √ x − √ <sup>481</sup> <sup>y</sup>, and the last inequality follows from the definition of group K<sup>i</sup> <sup>482</sup> .

<sup>483</sup> Taking a summation of [\(A.2\)](#page-11-0) over all groups, the total regret can be lower bounded as follows:

$$\begin{aligned} & \text{Regret}(K) \\ &= \sum_{i=0}^{L-1} \sum_{k \in \mathcal{K}_i} \max_{\mathbf{x} \in \mathcal{D}_k} \langle \boldsymbol{\mu}_i, \mathbf{x} \rangle - \langle \boldsymbol{\mu}_i, \mathbf{x}_k \rangle \\ &\geq \sum_{i=1}^{L-1} \Omega \left( d' \sqrt{\sum_{k \in \mathcal{K}_i} \sigma_k^2 / \log^4(dK)} - \sqrt{3L} d'^2 \cdot \sigma(i) / \log^4(dK) \right) \\ &\geq \Omega \left( \sum_{i=1}^{L-1} d/L^2 \cdot \sqrt{\sum_{k \in \mathcal{K}_i} \sigma_k^2 / \log^4(dK)} - 2\sqrt{3L} d^2 / (L^4 \log^4(dK)) \right) \\ &\geq \Omega \left( d/L^2 \cdot \sqrt{\sum_{i=1}^{L-1} \sum_{k \in \mathcal{K}_i} \sigma_k^2 / \log^4(dK)} - 2\sqrt{3L} d^2 / (L^4 \log^4(dK)) \right), \end{aligned} \tag{A.3}$$

<sup>484</sup> where the first inequality follows from [\(A.2\)](#page-11-0), the second inequality follows from the definition of variance threshold σ(i) and dimension d ′ = d/L<sup>2</sup> , and the last inequality holds due to P i √ 485 p x<sup>i</sup> ≥ P i x<sup>i</sup> <sup>486</sup> . In addition, for the group K0, we have

$$\sum_{k \in \mathcal{K}_0} \sigma_k^2 \leq \sum_{k \in \mathcal{K}_0} 1/K \leq 1, \quad (\text{A.4})$$

<sup>487</sup> where the first inequality follows from the definition of group K<sup>0</sup> and the second inequality follows <sup>488</sup> from |K0| ≤ K. Therefore, we have

$$\begin{aligned} & \text{Regret}(K) \\ & \geq \Omega\left(d/L^2 \cdot \sqrt{\sum_{i=1}^{L-1} \sum_{k \in \mathcal{K}_i} \sigma_k^2 / \log^4(dK) - 2\sqrt{3L}d^2 / (L^4 \log^4(dK))}\right) \\ & \geq \Omega\left(d/L^2 \cdot \sqrt{\sum_{i=1}^{L-1} \sum_{k \in \mathcal{K}_i} \sigma_k^2 - 1 / \log^4(dK) - 2\sqrt{3L}d^2 / (L^4 \log^4(dK))}\right) \\ & \geq \Omega\left(d \cdot \sqrt{\sum_{i=1}^{L-1} \sum_{k \in \mathcal{K}_i} \sigma_k^2 / \log^6(dK)}\right), \end{aligned}$$

<sup>489</sup> where the first inequality follows from [\(A.3\)](#page-11-1), the second inequality follows from [\(A.4\)](#page-11-2), and the last inequality follows from the fact that P<sup>K</sup> <sup>k</sup>=1 σ 2 <sup>k</sup> ≥ Ω(d 2 <sup>490</sup> ). Thus, we complete the proof of Theorem <sup>491</sup> [5.2.](#page-6-0)

## <sup>492</sup> B Proof of Theorem [5.4](#page-7-0)

<sup>493</sup> In this subsection, we provide the proof of Theorem [5.4.](#page-7-0) We begin by describing a simple algorithm:

 1. The learner maintains an explored action set A, which is initialized as empty. 2. For each decision set D<sup>k</sup> in round k, if there exists an action x<sup>k</sup> not in the spanning space of the explored action set A, the learner: • Selects an action x<sup>k</sup> and receives reward rk; • Updates the explored set: A = A ∪ {(xk, rk)}.

 3. Otherwise, when all actions lie in the spanning space of A, the learner: • Estimates the reward for each action through linear combinations of (x, r) ∈ A; • Selects the action with maximum estimated reward.

 It is worth noting that this algorithm assumes the received rewards r<sup>k</sup> have no noise to provide accurate estimates in step 3. While this assumption does not hold in general, when an adversary can choose the variance σ<sup>k</sup> based on the decision set Dk, they can cooperate with the learner by setting:

 • σ<sup>k</sup> = 0 when step 2 is triggered (exploration); • σ<sup>k</sup> = 1 when step 3 is triggered (exploitation).

 For a d-dimensional linear bandit problem, the explored action set satisfies |A| ≤ d. This implies the learner performs at most d exploration steps with zero variance, while all remaining steps have variance one. Under this construction, the regret in the first K rounds is upper bounded by:

$$\text{Regret}_{\text{Alg}}(K) \leq d$$
,

where the total variance P<sup>K</sup> <sup>k</sup>=1 σ <sup>k</sup> = K −d ≥ K/2 (since K ≥ 2d). Thus, through this cooperation between the adversary and learner, the Ω( e <sup>d</sup> qP<sup>K</sup> <sup>k</sup>=1 σ k ) lower bound is broken, completing the proof of Theorem [5.4.](#page-7-0)

## C Proof of Key Lemmas

## C.1 Proof of Lemma [4.3](#page-4-0)

 In this subsection, we provide the proof of Lemma [4.3.](#page-4-0) When the variance threshold σ = 1, our construction reduces to the standard lower bound instances for linear contextual bandits [\(Zhou et al.,](#page-9-4) [2021\)](#page-9-4). Specifically, when the number of rounds K satisfying K ≥ 1.5 · d , [Zhou et al.](#page-9-4) [\(2021\)](#page-9-4) provided the following variance-independent lower bound for these hard instances:

Lemma C.1 (Lemma C.8, [Zhou et al. 2021\)](#page-9-4). For any bandit algorithm Alg, if the weight vector µ ∈ {−∆, ∆} d is drawn uniformly at random from {−∆, ∆} d , then the expected regret over K rounds is lower bounded by:

$$\mathbb{E}_{\mu}[\text{Regret}(K)] \geq \frac{d\sqrt{K}}{8\sqrt{6}}.$$

With the help of Lemma [C.1,](#page-12-0) we start the proof of Lemma [4.3.](#page-4-0)

*Proof of Lemma [4.3.](#page-4-0)* For any algorithm Alg for linear contextual bandit with fixed variance thresh- old σ, we construct an auxiliary algorithm Alg1 to solve the standard linear contextual bandit prob-lem:

 • At the beginning of each round k ∈ K, Alg1 observes the decision set D<sup>k</sup> and sends it to Alg; • Alg selects action a<sup>k</sup> ∈ D<sup>k</sup> based on the historical observations and delivers it to Alg1; • Alg1 performs the action ak, receives the reward r<sup>k</sup> and sends the normalized reward σ · r<sup>k</sup> to Alg.

 Now, we consider the performance of auxiliary algorithm Alg1 for the standard linear contextual bandit problem. It is worth noticing that the reward/noise in bandit instances for algorithm Alg1 and algorithm Alg only differ by a scalar factor σ, therefore for each instance, we have

$$\mathbb{E}[\text{Regret}_{\text{Alg}}(K)] = \sigma \cdot \mathbb{E}[\text{Regret}_{\text{Alg}_1}(K)]. \quad (\text{C.1})$$

If we randomly select a weight parameter vector µ ∈ {−∆, ∆} d , then according to Lemma [C.1,](#page-12-0) the regret for Alg is lower bounded by

$$\mathbb{E}_{\boldsymbol{\mu}}[\text{Regret}_{\text{Alg}}(K)] = \sigma \cdot \mathbb{E}_{\boldsymbol{\mu}}[\text{Regret}_{\text{Alg}}(K)] \geq \sigma \cdot \frac{d\sqrt{K}}{8\sqrt{6}} = \frac{d\sqrt{K\sigma^2}}{8\sqrt{6}},$$

#### <sup>538</sup> C.2 Proof of Lemma [A.1](#page-10-2)

<sup>539</sup> In this subsection, we provide the proof of Lemma [A.1.](#page-10-2) We begin by recalling the OFUL algorithm <sup>540</sup> in [Abbasi-Yadkori et al.](#page-9-3) [\(2011a\)](#page-9-3) and its corresponding upper bound for the regret:

<sup>541</sup> Lemma C.2 (Theorem 3 in [Abbasi-Yadkori et al. 2011a\)](#page-9-3). For any linear contextual bandit problem, <sup>542</sup> with probability at least 1−δ, the regret for OFUL algorithm in the first K rounds is upper bounded by Regret(K) ≤ <sup>O</sup>e d p K log(dK/δ) <sup>543</sup> .

<sup>544</sup> It is worth noting that the reward/noise in the instance construction from Lemma [4.3](#page-4-0) only differs by <sup>545</sup> a scalar factor σ from the standard bandit. Therefore, as discussed in Section [C.1,](#page-12-2) the regret in these <sup>546</sup> two cases also only differs by a scalar factor σ. This leads to the following corollary:

<sup>547</sup> Corollary C.3. For the instance construction from Lemma [4.3,](#page-4-0) there exists a constant C such that <sup>548</sup> with probability at least 1−δ, the regret for OFUL algorithm in the first K rounds is upper bounded by Regret(K) ≤ Cdp Kσ<sup>2</sup> <sup>549</sup> log(dK/δ).

<sup>550</sup> With the help of Corollary [C.3,](#page-13-0) we can begin the proof of Lemma [A.1.](#page-10-2)

<sup>551</sup> *Proof of Lemma [A.1.](#page-10-2)* For any algorithm Alg, we construct an auxiliary algorithm Alg1 as follows:

 • At the beginning of each round k ∈ [K], Alg1 observes the decision set D<sup>k</sup> and sends it to <sup>553</sup> Alg; • Alg selects action a<sup>k</sup> ∈ D<sup>k</sup> based on the historical observations and delivers it to Alg1; • Alg1 performs the action a<sup>k</sup> and receives the reward rk; • Alg1 calculates the pseudo regret as:

$$\text{Regret}'(k) = \sum_{i=1}^k \frac{1}{3} + \frac{d}{\sqrt{96K}} - r_k.$$

If the pseudo regret is larger than d √ Kσ2/(8√ 6) + σ p <sup>557</sup> 2K log(2K/δ), Alg1 removes all <sup>558</sup> previous information and performs the OFUL algorithm in all future rounds.

 Based on the construction of the instances, whatever the weight vector µ is, the optimal action is to select an action in the same direction as the weight vector, obtaining an expected reward of <sup>1</sup>/3 + d/√ 96K. Under this scenario, with probability at least 1 − δ, for any round k ∈ [K], the difference between pseudo regret Regret′ (k) and true regret Regret(k) can be upper bounded by

$$|\text{Regret}(k) - \text{Regret}'(k)| = \left| \sum_{i=1}^k \epsilon_i \right| \leq \sigma \sqrt{2K \log(2K/\delta)}, \quad (\text{C.2})$$

<sup>563</sup> where the inequality holds due to Lemma [D.1](#page-14-0) with the fact that the noise satisfies <sup>564</sup> <sup>E</sup>[ϵk|a1:k, r1:k−1] = 0 and |ϵk| ≤ σ. Thus, according to the criterion of auxiliary algorithm <sup>565</sup> Alg1, with probability at least 1 − δ, the regret of Alg1 before transitioning to OFUL is up to d √ Kσ2/(8√ 6) + 2σ p <sup>566</sup> 2K log(2K/δ). On the other hand, for the stage after transitioning to <sup>567</sup> OFUL, Corollary [C.3](#page-13-0) suggests that with probability at least 1 − δ, the regret is no more than Cdp Kσ<sup>2</sup> <sup>568</sup> log(dK/δ). Therefore, with a selection of δ = 1/K, we have

$$\mathbb{P}[\text{Regret}_{\text{Alg}_1}(K) \geq Cd\sqrt{K\sigma^2 \log(dK^2)} + d\sqrt{K\sigma^2}/(8\sqrt{6}) + 2\sigma\sqrt{2K \log(2K^2)}] \leq 2/K. \quad (\text{C.5})$$

For simplicity, let R = Cdp

Kσ<sup>2</sup> log(dK<sup>2</sup>) + d

√

Kσ2/(8√

6) + 2σ

p

2K log(2K<sup>2</sup> <sup>569</sup> ) and we have

$$\begin{aligned}
& \mathbb{E}_{\boldsymbol{\mu}}[\text{Regret}_{\text{Alg}_1}(K)] \\
& \leq \mathbb{P}[\text{Regret}_{\text{Alg}_1}(K) \geq R] \cdot K\sigma + \mathbb{P}[\text{Regret}_{\text{Alg}_1}(K) \geq d\sqrt{K\sigma^2}/(16\sqrt{6})] \cdot R \\
& \quad + \mathbb{P}[\text{Regret}_{\text{Alg}_1}(K) \geq 0] \cdot d\sqrt{K\sigma^2}/(16\sqrt{6}) \\
& \leq 2\sigma + \mathbb{P}[\text{Regret}_{\text{Alg}_1}(K) \geq d\sqrt{K\sigma^2}/(16\sqrt{6})] \cdot \tilde{O}(d\sqrt{K\sigma^2 \log(dK)}) + d\sqrt{K\sigma^2}/(16\sqrt{6}),
\end{aligned}$$

<sup>570</sup> where the first inequality holds due to <sup>E</sup>[X] ≤ <sup>P</sup>(X ≥ x1) · R + <sup>P</sup>(X ≥ x2) · x<sup>1</sup> + <sup>P</sup>(X ≥ 0) · x<sup>2</sup> <sup>571</sup> for 0 ≤ X ≤ R and x<sup>1</sup> > x<sup>2</sup> > 0, and the second inequality holds due to [\(C.3\)](#page-13-1). Combining this <sup>572</sup> result with the lower bound of expected regret in Lemma [4.1,](#page-3-0) we have

$$d\sqrt{K\sigma^2}/(8\sqrt{6}) \geq 2\sigma + \mathbb{P}[\text{Regret}_{\text{Alg}_1}(K) \geq d\sqrt{K\sigma^2}/(16\sqrt{6})] \cdot \tilde{O}(d\sqrt{K\sigma^2} \log(dK))$$

$$+ d\sqrt{K\sigma^2}/(16\sqrt{6}),$$

<sup>573</sup> which implies that

$$\mathbb{P}[\text{Regret}_{\text{Alg}_1}(K) \geq d\sqrt{K\sigma^2}/(16\sqrt{6})] \geq \Omega(1/\log(dK)). \quad (\text{C.4})$$

<sup>574</sup> In addition, according to the criterion of auxiliary algorithm Alg1 with [\(C.2\)](#page-13-2), with probability at <sup>575</sup> least 1 − δ = 1 − 1/K, Alg1 will not switch to the OFUL algorithm until the cumulative regret is larger than d √ Kσ2/(8√ <sup>576</sup> 6), which implies that

$$\begin{aligned}\mathbb{P}[\text{Regret}_{\text{Alg}}(K) \geq d\sqrt{K\sigma^2}/(16\sqrt{6})] &\geq \mathbb{P}[\text{Regret}_{\text{Alg}_1}(K) \geq d\sqrt{K\sigma^2}/(16\sqrt{6})] - 1/K \\ &= \Omega(1/\log(dK)).\end{aligned}$$

<sup>577</sup> Thus, we complete the proof of Lemma [A.1.](#page-10-2)

## <sup>578</sup> C.3 Proof of Lemma [A.2](#page-10-0)

<sup>579</sup> In this subsection, we provide the proof of Lemma [A.2.](#page-10-0)

<sup>580</sup> *Proof of Lemma [A.2.](#page-10-0)* Since the learner visits the instances in a cyclic manner, over all K rounds, <sup>581</sup> each instance M<sup>i</sup> (i = 1, 2, . . . , L) is visited K′ = K/L times. As actions from different instances only interact with their corresponding parameters, according to Lemma [A.1,](#page-10-2) for each instance M<sup>i</sup> <sup>582</sup> , with probability at least Ω 1/ log(dK) <sup>583</sup> , the regret is lower bounded by

$$\text{Regret}(K', \mathcal{M}_i) \geq \frac{d' \sqrt{K' \sigma^2}}{16\sqrt{6}} = \frac{d \sqrt{K \sigma^2}}{16\sqrt{6} \cdot L^{1.5}}.$$

<sup>584</sup> Note that the weight vectors for each instance are independently sampled, hence the probability that at least one instance has regret no less than d √ Kσ2/16√ 6 · L 1.5 <sup>585</sup> is at least

$$1 - \left(1 - \Omega(1/\log(dK))\right)^L \geq 1 - 1/K^3 \text{ Qingyue: } ???$$

<sup>586</sup> Under this condition, the total regret can be lower bounded as:

$$\text{Regret}(K) = \sum_{i=1}^L \text{Regret}(K', \mathcal{M}_i) \geq \frac{d\sqrt{K\sigma^2}}{16\sqrt{6} \cdot L^{0.5}}. \quad (\text{C.5})$$

<sup>587</sup> Thus, we obtain a high-probability lower bound and complete the proof of Lemma [A.2.](#page-10-0)

## <sup>588</sup> D Auxiliary Lemmas

Lemma D.1 (Azuma–Hoeffding inequality, [Cesa-Bianchi and Lugosi 2006\)](#page-9-17). Let {ηk} K <sup>k</sup>=1 <sup>589</sup> be a mar-<sup>590</sup> tingale difference sequence with respect to a filtration {Gk} satisfying |ηk| ≤ R for some constant R, η<sup>k</sup> is Gk+1-measurable, <sup>E</sup> ηk|G<sup>k</sup> <sup>591</sup> = 0. Then for any 0 < δ < 1, with high probability at least <sup>592</sup> 1 − δ, we have

$$\sum_{k=1}^K \eta_k \leq R\sqrt{2K \log(1/\delta)}.$$

## NeurIPS Paper Checklist

### 1. Claims

 Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

 Justification: In both abstract and introduction, we highlight the contribution in our pa- per. The proposed algorithm and the corresponding theoretical results are discussed in the followed sections

Guidelines:

 • The answer NA means that the abstract and introduction do not include the claims made in the paper. • The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A No or NA answer to this question will not be perceived well by the reviewers. • The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings. • It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We explicitly list all the necessary assumptions for our theoretical analysis.

Guidelines:

 • The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper. • The authors are encouraged to create a separate "Limitations" section in their paper. • The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The au- thors should reflect on how these assumptions might be violated in practice and what the implications would be. • The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated. • The authors should reflect on the factors that influence the performance of the ap- proach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon. • The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size. • If applicable, the authors should discuss possible limitations of their approach to ad- dress problems of privacy and fairness. • While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an impor- tant role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Answer: [Yes]

 Justification: The complete set of assumptions for our analysis is presented in Section 3, with the detailed proofs of all our claims provided in a later section.

Guidelines:

 • The answer NA means that the paper does not include theoretical results. • All the theorems, formulas, and proofs in the paper should be numbered and cross- referenced. • All assumptions should be clearly stated or referenced in the statement of any theo- rems. • The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition. • Inversely, any informal proof provided in the core of the paper should be comple- mented by formal proofs provided in appendix or supplemental material. • Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

 Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclu-sions of the paper (regardless of whether the code and data are provided or not)?

Answer: [NA]

Justification: The paper does not include experiments.

Guidelines:

 • The answer NA means that the paper does not include experiments. • If the paper includes experiments, a No answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not. • If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable. • Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed. • While NeurIPS does not require releasing code, the conference does require all sub- missions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm. (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully. (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to re- produce the model (e.g., with an open-source dataset or instructions for how to construct the dataset). (d) We recognize that reproducibility may be tricky in some cases, in which case au- thors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

 Question: Does the paper provide open access to the data and code, with sufficient instruc- tions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [NA]

Justification: The paper does not include experiments.

Guidelines:

 • The answer NA means that paper does not include experiments requiring code. • Please see the NeurIPS code and data submission guidelines ([https://nips.cc/](https://nips.cc/public/guides/CodeSubmissionPolicy) [public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details. • While we encourage the release of code and data, we understand that this might not be possible, so "No" is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark). • The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ([https:](https://nips.cc/public/guides/CodeSubmissionPolicy) [//nips.cc/public/guides/CodeSubmissionPolicy](https://nips.cc/public/guides/CodeSubmissionPolicy)) for more details. • The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc. • The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why. • At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable). • Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

#### 6. Experimental setting/details

 Question: Does the paper specify all the training and test details (e.g., data splits, hyper- parameters, how they were chosen, type of optimizer, etc.) necessary to understand the results?

Answer: [NA]

Justification: The paper does not include experiments.

Guidelines:

 • The answer NA means that the paper does not include experiments. • The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them. • The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

 Question: Does the paper report error bars suitably and correctly defined or other appropri-ate information about the statistical significance of the experiments?

Answer: [NA]

Justification: The paper does not include experiments.

Guidelines:

 • The answer NA means that the paper does not include experiments. • The authors should answer "Yes" if the results are accompanied by error bars, confi- dence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper. • The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions). • The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

 • The assumptions made should be given (e.g., Normally distributed errors). • It should be clear whether the error bar is the standard deviation or the standard error of the mean. • It is OK to report 1-sigma error bars, but one should state it. The authors should prefer- ably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified. • For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g. negative error rates). • If error bars are reported in tables or plots, The authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

#### 8. Experiments compute resources

 Question: For each experiment, does the paper provide sufficient information on the com- puter resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [NA]

Justification: The paper does not include experiments.

Guidelines:

 • The answer NA means that the paper does not include experiments. • The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage. • The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute. • The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

#### 9. Code of ethics

 Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics <https://neurips.cc/public/EthicsGuidelines>?

Answer: [Yes]

 Justification: The research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics.

Guidelines:

 • The answer NA means that the authors have not reviewed the NeurIPS Code of Ethics. • If the authors answer No, they should explain the special circumstances that require a deviation from the Code of Ethics. • The authors should make sure to preserve anonymity (e.g., if there is a special consid-eration due to laws or regulations in their jurisdiction).

#### 10. Broader impacts

 Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [NA]

Justification: The paper is a theoretical work with no societal impact.

Guidelines:

 • The answer NA means that there is no societal impact of the work performed. • If the authors answer NA or No, they should explain why their work has no societal impact or why the paper does not address societal impact. • Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact spe-cific groups), privacy considerations, and security considerations.

 • The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster. • The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology. • If there are negative societal impacts, the authors could also discuss possible mitiga- tion strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

#### 11. Safeguards

 Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pretrained language models, image generators, or scraped datasets)?

Answer: [NA]

Justification: The paper is a theoretical work and poses no such risks

Guidelines:

 • The answer NA means that the paper poses no such risks. • Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by re- quiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters. • Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images. • We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

#### 12. Licenses for existing assets

 Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

#### Answer: [Yes]

 Justification: We have described the related works, especially those work which our work is based on with proper citations in corresponding sections.

Guidelines:

 • The answer NA means that the paper does not use existing assets. • The authors should cite the original paper that produced the code package or dataset. • The authors should state which version of the asset is used and, if possible, include a URL. • The name of the license (e.g., CC-BY 4.0) should be included for each asset. • For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided. • If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, [paperswithcode.com/](paperswithcode.com/datasets) [datasets](paperswithcode.com/datasets) has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset. • For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

 • If this information is not available online, the authors are encouraged to reach out to the asset's creators.

#### 13. New assets

 Question: Are new assets introduced in the paper well documented and is the documenta-tion provided alongside the assets?

Answer: [No]

Justification: This is a theoretical paper without experiments.

Guidelines:

 • The answer NA means that the paper does not release new assets. • Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc. • The paper should discuss whether and how consent was obtained from people whose asset is used. • At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

#### 14. Crowdsourcing and research with human subjects

 Question: For crowdsourcing experiments and research with human subjects, does the pa- per include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [NA]

Justification: This paper does not include crowdsourcing or human subjects.

Guidelines:

 • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Including this information in the supplemental material is fine, but if the main contri- bution of the paper involves human subjects, then as much detail as possible should be included in the main paper. • According to the NeurIPS Code of Ethics, workers involved in data collection, cura- tion, or other labor should be paid at least the minimum wage in the country of the data collector.

#### 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

 Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [NA]

Justification: This paper does not include crowdsourcing or human subjects.

Guidelines:

 • The answer NA means that the paper does not involve crowdsourcing nor research with human subjects. • Depending on the country in which research is conducted, IRB approval (or equiva- lent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper. • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution. • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

 Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigorousness, or originality of the research, declaration is not required.

Answer: [No]

 Justification: We only used an LLM to rephrase the writing, which did not affect the core methodology, scientific rigor, or originality of the research.

Guidelines:

 • The answer NA means that the core method development in this research does not involve LLMs as any important, original, or non-standard components. • Please refer to our LLM policy ([https://neurips.cc/Conferences/](https://neurips.cc/Conferences/2025/LLM) [2025/LLM](https://neurips.cc/Conferences/2025/LLM)) for what should or should not be described.