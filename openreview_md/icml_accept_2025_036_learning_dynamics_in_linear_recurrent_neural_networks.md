# 

Alexandra M. Proca 1 **Clementine C.J. Domin** ´ e´
2 Murray Shanahan 1 **Pedro A.M. Mediano** 1 3

## Abstract

Recurrent neural networks (RNNs) are powerful models used widely in both machine learning and neuroscience to learn tasks with temporal dependencies and to model neural dynamics.

However, despite significant advancements in the theory of RNNs, there is still limited understanding of their learning process and the impact of the temporal structure of data. Here, we bridge this gap by analyzing the learning dynamics of linear RNNs (LRNNs) analytically, enabled by a novel framework that accounts for task dynamics. Our mathematical result reveals four key properties of LRNNs: (1) Learning of data singular values is ordered by both scale and temporal precedence, such that singular values that are larger and occur later are learned faster. (2) Task dynamics impact solution stability and extrapolation ability. (3) The loss function contains an effective regularization term that incentivizes small weights and mediates a tradeoff between recurrent and feedforward computation. (4) Recurrence encourages feature learning, as shown through a novel derivation of the neural tangent kernel for finitewidth LRNNs. As a final proof-of-concept, we apply our theoretical framework to explain the behavior of LRNNs performing sensory integration tasks. Our work provides a first analytical treatment of the relationship between the temporal dependencies in tasks and learning dynamics in LRNNs, building a foundation for understanding how complex dynamic behavior emerges in cognitive models.

## 1. Introduction

Recurrent neural networks (RNNs) are important tools in both machine learning and neuroscience for learning tasks with temporal dependencies. Recently, (linear) recurrent architectures (state space models), have had a resurgence of popularity in long-range sequence modeling (Gu et al., 2020; 2022; Orvieto et al., 2023; Gu & Dao, 2024). In tandem with the success of dynamical systems theory in describing neural activity related to motor control, working memory, and decision-making (Remington et al., 2018a; Vyas et al.,
2020; Khona & Fiete, 2021), RNNs have also become a popular choice for cognitive models of neural dynamics
(Barak, 2017), as they not only replicate recurrent dynamics recorded in animals but are also capable of performing abstractions of the same cognitive tasks used in experiments (Mante et al., 2013; Engel et al., 2015; Chaisangmongkon et al., 2017; Wang et al., 2017; Masse et al., 2018; Remington et al., 2018b; Orhan & Ma, 2018; Masse et al., 2020; Beiran et al. ´ , 2023). More generally, RNNs present an interesting model of study due to the complex computational capabilities given by their hidden layer that evolves with time and which is a universal approximator of any open dynamical system (Doya, 1993; Schafer & Zimmermann ¨ , 2007). Accompanying the popularity of RNNs, there have been significant efforts dedicated to their theoretical understanding, both from deep learning theoreticians (Cohen-Karlik et al., 2023; Orvieto et al., 2024; Zucchet & Orvieto, 2024) and neuroscientists relating these findings to observations about the brain (Sussillo & Barak, 2013; Mastrogiuseppe & Ostojic, 2018; Yang et al., 2019; Schuessler et al., 2020a;b; Turner et al., 2021; Dubreuil et al., 2022; Farrell et al., 2022; Turner & Barak, 2023; Driscoll et al., 2024; Liu et al., 2024). However, most theoretical studies of RNNs are done at the end of training - analyzing properties of the solutions they find, but ignoring the learning process itself
(Saxe et al., 2020). Of the work that does study learning, the focus is often related to practical considerations about training, such as learning long-range dependencies. Overall, despite the widespread use and known complex computational abilities of RNNs, it is still unknown how their underlying functional structures emerge as a result of training on temporally-structured tasks.

1 One related line of previous work has focused on using deep linear networks to analyze learning dynamics (Saxe et al., 2014; 2018; Braun et al., 2022; Domine et al. ´ , 2025). Although unable to solve nonlinear problems (note however that there has been progress to overcome this limitation (Saxe et al., 2022; Sandbrink et al., 2024)), these networks exhibit complex nonlinear learning dynamics and are analytically tractable, providing a useful framework for theoretical investigation. Applied to cognitive neuroscience, the study of learning dynamics has been used to propose a theory of semantic development (Saxe et al., 2018), cognitive flexibility (Sandbrink et al., 2024), and localization in receptive fields (Lufkin et al., 2024), among other work. Despite its successes, the analytical treatment of learning dynamics in linear networks has primarily remained in the domain of feedforward networks. In order to more broadly characterize learning, however, theory needs to account for the impact of dynamic task settings and the rich structure endowed by recurrent networks, especially since it is such a critical component of neural computation. Of the few prior studies of learning dynamics in linear RNNs, Schuessler et al. (2020b)
showed that networks make low-rank changes to their connectivity during learning and Smekal et al. ´ (2024) showed how overparameterization accelerates convergence time by studying the frequency domain. However, the influence of temporally structured data on learning has not been studied analytically to our knowledge. In this work, we study the learning dynamics of linear RNNs (LRNN) to better understand the influence of temporal data on learning in recurrent cognitive systems, unifying the areas of RNN theory and learning dynamics. Our theoretical results contribute to explanations of many phenomena spanning both topics, including low-rank connectivity (Mastrogiuseppe & Ostojic, 2018), rich and lazy learning (Farrell et al., 2023), extrapolation capabilities (Cohen-Karlik et al., 2023), and network stability (Sompolinsky et al., 1988). Taken together, this represents a substantial step towards the theoretical understanding of learning in recurrent deep learning models, building a foundation for new theories and hypotheses of learning in neural networks and the brain. Our contributions are as follows:
- We provide, for the first time, a closed-form analytical expression for the energy function of LRNNs decoupled along singular/eigen-value dimensions. We use this result, together with a novel framework to describe task dynamics, to accurately predict solutions found by LRNNs.

- We identify how both the magnitude and temporal ordering of data singular values affect learning speed.

- We describe how task dynamics impact network stability and extrapolation ability, even in cases where the network achieves 0 loss.

- We identify an effective regularization term in the energy function that incentivizes small weights, and a phase transition in the connectivity modes that leads to low-rank solutions.

- We derive the neural tangent kernel (Jacot et al., 2018)
for finite-width LRNNs and show that recurrence facilitates feature learning.

- We demonstrate the generalizability of our results by applying our theoretical framework to describe the behavior of LRNNs trained on sensory integration tasks, relaxing our prior assumptions.

## 2. Mathematical Setup 2.1. Model

We study a LRNN (Figure 1) parameterized by matrices Wx ∈ R
Nh×Nx , Wh ∈ R
Nh×Nh , Wy ∈ R
Ny×Nh with a hidden state ht ∈ R
Nh that receives an input xt ∈ R
Nx at each timestep t and updates its hidden state. For simplicity, in the main text we study the *single-output* case, where the network only produces an output yˆT ∈ R
Ny at the last timestep T. In Appendix M, we generalize our approach to networks trained to produce outputs yˆt at every timestep t (the autoregressive *T-output* case). The network is characterized by the equations

$$\begin{array}{c}{{h_{t+1}=W_{h}h_{t}+W_{x}x_{t}~,}}\\ {{\hat{y}_{T}=W_{y}h_{T+1}~.}}\end{array}$$

We initialize the hidden state h1 as a vector of zeros, yielding

$$h_{t+1}=\sum_{i=1}^{t}W_{h}^{t-i}W_{x}x_{i}\;.$$
$$\begin{array}{l}{(1)}\\ {(2)}\end{array}$$
$$({\mathfrak{I}})$$

We analyze learning in the LRNN when trained using backpropagation through time on the squared error over P trajectories {xp,1, xp,2, . . . , xp,T , yp,T }
P p=1

$${\mathcal{L}}={\frac{1}{2}}\sum_{p=1}^{P}\|\mathbf{y}_{p,T}-W_{y}(\sum_{i=1}^{T}W_{h}^{T-i}W_{x}\mathbf{x}_{p,i})\|^{2}\tag{4}$$

## 2.2. Temporal Singular Values

With the model and loss function fixed, our next step is to specify a task for the model to learn. In this linear setting, the task is fully specified by the sequence of matrices Σ
Y Xt =PP
p=1 yp,T x
⊤
p,t, the input-output correlation matrix between the input xp,t at timestep t and the final output

... ...
Figure 1. **Linear RNN model captures task dynamics through**
temporally-dependent singular values. The data correlation matrices Σ
Y Xt have constant left and right singular vectors, varying only in their singular values St across time. target yp,T . Extending the approach by Saxe et al. (2014; 2018), we can represent these matrices through either their singular value decomposition (SVD; case 1 in the assumptions below), or their eigendecomposition (case 2), which enables us to account for task dynamics. In the main text, we follow the precedence from prior work and base our analysis around a derivation using SVD, which, although more restrictive with task dynamics, simplifies the setting and allows for non-square networks. Then, in Appendices N
and O, we use an eigendecomposition to derive a similar but more general form (accounting for complex eigenvalues thus allowing for rotational dynamics) and show that our framework and results extend naturally to this case. To simplify our derivations, we make the following assumptions: Assumption 1 (whitened input): Inputs are uncorrelated and whitened across all timesteps and dimensions, such that Σ
XtXt = I; ΣXtXt′ = 0, t ̸= t
′.

Assumption 2 (constant singular vectors or **eigenvectors):**
All input-output correlation matrices have either (1) constant left and right singular matrices Uy, Vx and only vary in their singular values St over a trajectory, such that Σ
Y Xt = UyStV
⊤
x
, ∀t; or (2) constant eigenvectors P and only vary in their eigenvalues D over a trajectory, such that Σ
Y Xt = P DtP
†.

Assumption 3 (aligned model): The model is aligned to either (1) the data singular vectors at initialization such that U
⊤
y Wy(0)Ry, R⊤
y WhRx, R⊤
x Wx(0)Vx yield diagonal matrices Wy(0), Wh(0), Wx(0) for some orthogonal matrices Ry, Rx, or (2) the data eigenvectors P such that P
†Wy(0)P,P
†Wh(0)P,P
†Wx(0)P yield diagonal matrices Wy(0), Wh(0), Wx(0).

Although seemingly restrictive, we argue these assumptions still capture meaningful learning scenarios. For example: (1) data can be whitened using the innovations form of a Kalman filter (Durbin & Koopman, 2012); (2)
singular/eigen-vectors are constant if data is generated by a diagonalizable LRNN teacher (Appendices B and N), and in fact, because we don't restrict the dynamics of the singular/eigen-values, our form captures more general settings than the standard teacher-student setup (which constrains task dynamics to the form δλT −t); (3) prior work has shown that model alignment occurs early in training for networks initialized with small random weights (Atanasov et al., 2022) and there are theoretical and practical justifications for diagonalizable state spaces (Hazan et al., 2018; Gupta & Berant, 2022).

## 3. Results 3.1. Lrnn Energy Function

With the aforementioned assumptions, we can diagonalize the network, eliminating cross-terms. Let aα, bα, cα be the α th diagonal entry of Wx, Wh, Wy, respectively, and sα,t be the α th singular value (SV) of St. Assuming a small learning rate 1/τ (i.e., the *gradient flow* regime), we can write the gradients of the network parameters as a set of differential equations in terms of these variables, or connectivity modes, the dynamics of which decouple across SV
dimensions α (Appendix B). We refer to aα as the input, bα the recurrent, and cα the output connectivity mode. Their dynamics are given by

$$\begin{array}{l}{{\tau\frac{d}{d t_{\theta}}a_{\alpha}=\sum_{i=1}^{T}c_{\alpha}b_{\alpha}^{T-i}(s_{\alpha,i}-c_{\alpha}b_{\alpha}^{T-i}a_{\alpha})}}\\ {{\tau\frac{d}{d t_{\theta}}b_{\alpha}=\sum_{i=1}^{T-1}(T-i)c_{\alpha}b_{\alpha}^{T-i-1}a_{\alpha}(s_{\alpha,i}-c_{\alpha}b_{\alpha}^{T-i}a_{\alpha})}}\end{array}$$
$$\mathbf{5})$$
$$(6)$$
$$\left(T\right)$$

$$\tau\frac{d}{d t_{\theta}}c_{\alpha}=\sum_{i=1}^{T}b_{\alpha}^{T-i}a_{\alpha}(s_{\alpha,i}-c_{\alpha}b_{\alpha}^{T-i}a_{\alpha})\;,$$
α aα) , (7)
where tθ refers to timesteps of gradient-based learning as opposed to the trajectory timesteps t. Our first result shows that these dynamics arise from gradient descent on an energy function.

Lemma 3.1. Given Assumptions 1-3, the energy function of the LRNN is given by

$$E(a_{\alpha},b_{\alpha},c_{\alpha})=\frac{1}{2\tau}\sum_{i=1}^{T}(s_{\alpha,i}-c_{\alpha}b_{\alpha}^{T-i}a_{\alpha})^{2}\;.\tag{8}$$

To ease notation, we omit specifying α when referring to connectivity modes in the remainder of the paper, although note that all terms (st*, a, b, c*) still refer to a particular SV dimension α. We also generally refer to the input-output modes (ac) together and treat them as a single term since there isn't any meaningful distinction between them.

in p uto utp ut m o d e 
(a c)
learning input-output, fixed recurrent re c urr e nt m o d e 
(b)
learning recurrent, fixed input-output 2 0.7 1.0 1.2 2 1 0.5 theory 1 f( ,t)
1 0.7 T t 0.7 t theory 0 10000 20000 30000 40000 training steps (t )
0 0 10000 20000 30000 40000 training steps (t )
0.0
To provide some intuition about the reduced form, the magnitude of the data SVs (sα,t) correspond to the strength of correlation between the input (xt) at trajectory timestep t and the output target (yT ) in different SV dimensions α. In this work, we're interested in understanding how recurrence and the task dynamics (given by s1:T ) impacts the LRNN's learning dynamics.

## 3.2. Solutions To Lrnn Learning Dynamics

While perhaps trivial, we can think of the LRNN as performing two functions: the input-output mode (ac) performs a constant scaling and the recurrent mode (b) learns a timedependent function. Thus, by decomposing the data SVs (s1:T ) into a constant and temporal component, we might better understand the solutions LRNNs converge to. We decompose each data SV as st = δf(λ, t), where δ is constant across all data SVs, and f(*λ, t*) is some function parameterized by λ that is dependent on trajectory timestep t. We derive a full solution for the learning dynamics of the input-output modes when recurrent modes are frozen (Appendix C), as well as a local approximation to the recurrent modes when input-output modes are frozen (Appendix D). Intuitively, the network should use recurrent modes to learn the dynamic component of the data since it varies in its contribution to the output through time, whereas input-output modes do not vary with time and thus can only contribute some form of scaling. By studying the learning dynamics of the recurrent and input-output modes separately, we confirm that they indeed learn these different components (Figure 2). The distinction between dynamic and scaling components of data SVs also highlights an important difference in learning dynamics between (deep) feedforward and recurrent linear networks. Feedforward linear networks learn the largest SVs first. In recurrent networks, however, the loss is computed over T SVs in each dimension (as opposed to one) and the network must optimize for SVs across time. A consequence is that SVs at different timesteps are weighted differently in the gradient. SV trajectories that are larger and have SVs occurring *later* in the trajectory are learned faster (assuming recurrent connectivity modes are initialized b < 1). We can see this effect by looking at the gradients of the connectivity modes (Equations (5) to (7)): gradients from early trajectory timesteps are weighted by the recurrent mode b exponentially with trajectory length. Since we initialize connectivity modes to be less than 1, this has the effect of downscaling the gradient contribution from earlier trajectory timesteps compared to later timesteps. Thus, we see a more complex portrait of the effect of both SV magnitude and SV dynamics (time) playing into the ordering of learning in recurrent networks. This effect can be seen in Figure 2 (right) - the blue curve converges to a smaller solution than the orange curve but is initially learned faster, which differs from the behavior of feedforward networks (e.g., the left plot), and is driven by the fact that the singular value at the last timestep is larger for the blue curve than the orange curve (from δ).

We study this further in Appendix E.

## 3.3. Task Dynamics Determine Solution Stability And Extrapolation Ability

RNNs suffer from problems related to stability during training and inference. By stability we refer to the state of the RNN parameters which may lead to exploding gradients or diverging hidden layer activity. Because of the exponential effect of the recurrent layer, (nonlinear) RNNs with eigenvalues larger than 1 exhibit chaotic behavior (Sompolinsky et al., 1988). By looking at the energy function (Equation (8)), we can further see the well-known effect of vanishing (exploding) gradients, given by |b| < 1 (|b| > 1) as T → ∞ (Bengio et al., 1994; Hochreiter et al., 2001; Pascanu et al., 2012), which makes training on tasks with long-range dependencies challenging and to which there have been numerous methods introduced to alleviate these difficulties (Hochreiter & Schmidhuber, 1997; Le et al.,
2015; Orvieto et al., 2023; Zucchet et al., 2023). Another open problem in RNNs is their ability to extrapolate (or interpolate) to sequence lengths that differ from those trained on, which is not well understood (Cohen-Karlik et al., 2023; Beiran et al. ´ , 2023). Here, we study how network stability and extrapolation ability are impacted by an additional factor: the underlying task dynamics a RNN is trained on. To do this, we study task dynamics that are perfectly learnable (provably the only task dynamics with 0-loss solutions; Appendix F), but differ in their hidden layer stability or ability to extrapolate. Recall that we can decompose data SVs as st = δf(*λ, t*). We distinguish between three cases with known analytical 0-loss solutions (Appendix F.3), which offer natural settings to study how perfectly-learnable data

extrapolate to other trajectory lengths T
dynamics with early-importance are *unstable*
Figure 3. **Task dynamics determine solution stability and extrapolation ability.** For RNNs trained on (*left*) constant dynamics (*f(λ, t*) = 1), solution stability is dependent on the scaling term δ, where δ → 0 is stable and δ ≫ 0 is less stable. (*Middle*) Inverseexponential (f(*λ, t*) = λ T −t) and (*right*) exponential (f(*λ, t) =* λ t) dynamics produce unstable solutions when λ > 1 and λ < 1, respectively, which correspond to *early-importance* dynamics (st > st+1). Further, the solution to the input-output modes (ac = δλT)
for exponential dynamics depends on trajectory length, so solutions learned for one length will not extrapolate to other trajectory lengths.

impacts solution stability and extrapolation ability. We consider cases where the data SVs are *constant* (f(λ, t) = 1), change inverse-exponentially (f(λ, t) = λ T −t), or change exponentially (f(λ, t) = λ t). By varying δ, λ, we can parameterize the task dynamics differently and elicit particular network behavior. We note that technically all of these dynamics can be reparameterized as inverse-exponential dynamics when the trajectory length is fixed (Appendix F.1), but for simplicity, we will keep these separate. For constant task dynamics, the global solution exists at b = 1*, ac* = δ; this can be understood as 'equally weighting' the input at each timestep, while the input-output connectivity modes learn an appropriate scaling δ. For inverse-exponential task dynamics, the minimum is found at b = *λ, ac* = δ, as the dynamics of the singular values (st = δf(λ, t), f(λ, t) = λ T −t) correspond exactly to the dynamics of the LRNN (cbT −ta). Finally, for exponential task dynamics, the minimum exists at b = 1*/λ, ac* = δλT.

By studying these solutions, we can first observe that inverse-exponential and exponential task dynamics yield unstable solutions (where the recurrent mode b > 1) for λ > 1 and λ < 1, respectively (green dashed lines in Figure 3). In both cases, the data SVs decrease across the trajectory (st > st+1); we hence refer to this as *early-importance*.

Due to the instability of exploding gradients as the recurrent mode b increases over 1, early-importance dynamics are more challenging, if not impossible, to learn as trajectory length increases (T → ∞). Constant dynamics are essentially an intermediary between exponential and inverse-exponential dynamics (i.e., because f(λ = 1, t) = 1t = 1T −t = 1). Constant dynamics are common, as they correspond to basic integration of input. They also present a way to study the influence of the scaling term (δ) on solution stability. Constant dynamics have stable solutions when the scaling term is small (δ → 0) because it keeps SVs and input-output modes small as the recurrent mode (b) approaches 1; however, when the scaling term is large (δ ≫ 0), optimization is more challenging as solutions approach unstable solutions near b = 1 (left in Figure 3).

This observation about δ also generalizes to other dynamics when the solution for the recurrent mode is not close to 0 (b ≫ 0). Although each case of task dynamics we consider here has a 0-loss solution, not all of these solutions extrapolate perfectly to other trajectory lengths T. In particular, the global solution for exponential task dynamics is dependent on the trajectory length T (ac = δλT). As such, this solution will not perfectly extrapolate to trajectory lengths that differ to the one trained on, and in fact the error will grow as the difference in trajectory length increases (see Appendix F.1). In conclusion, we can see that even for data that is perfectly learnable, properties of the task dynamics crucially impact the stability of solutions and the ability to extrapolate to other trajectory lengths. In particular, we find that (1) data with correlations that decrease over trajectory time produce unstable solutions, (2) task dynamics with a large scaling term (δ) are less stable, and (3) task dynamics with solutions that depend on trajectory length (such as exponential dynamics) do not extrapolate with 0-loss to other trajectory lengths. We show in Section 3.6 and Appendix L that these findings hold for RNNs without our theoretical assumptions.

## 3.4. Connectivity Modes Exhibit Phase Transitions Between Recurrent And Feedforward Computations

In the previous section we studied task dynamics with perfect solutions. However, most real-world tasks will, naturally, not exhibit inverse-exponential dynamics and may not have perfect 0-loss solutions (although the loss may be low in practice). A separate observation is that RNNs initialized with small random weights seem to learn low-rank solutions along effective 'task dimensions' (Schuessler et al., 2020b;

re c urre n t m o d e 
(b)
constant task dynamics re c urre n t m o d e 
(b)
exponential task dynamics re c urre n t m o d e 
(b)
Dirac delta task dynamics 0 1 recurrent computation ( )
0.0 0.5 1.0 1.5 2.0 0 1 1.2 recurrent computation ( )
0.0 0.2 0.4 0.6 0.8 1.0 0 1 2.4 recurrent computation ( )
0 1 2 3 4 0.0 0.2 0.4 0.6 0.8 1.0 1.2 f e e d f orw ard c o m p u t a tio n 
( )

0-loss solution in p u t-o u t p u t m o d e 
(a c)
in p u t-o u t p u t m o d e 
(a c)
in p u t-o u t p u t m o d e 
(a c)
0 1 2.4 recurrent computation ( )
0.00 0.25 0.50 0.75 1.00 1.25 0 1 1.2 recurrent computation ( )
0.00 0.25 0.50 0.75 1.00 1.25 0 1 recurrent computation ( )
0.00 0.25 0.50 0.75 1.00 1.25
E =1
$${\frac{1}{2\tau}}\left(\sum_{i=1}^{T}s_{i}^{2}-2s_{i}c b^{T-i}a\right)+\underbrace{{\frac{1}{2\tau}}c^{2}a^{2}{\frac{1-b^{2T}}{1-b^{2}}}}_{\mathrm{effective\regruization\term}}$$
.

(9)
where, as T → ∞*, the second term goes to infinity for* cba ≫ 0.

For task dynamics that are learnable with low loss, both terms cancel out. However, when the LRNN cannot fit the data, the second term acts as an effective regularizer that incentivizes connectivity modes to remain close to 0 (Appendix G). This suggests that LRNNs might have an implicit bias towards effectively low-rank solutions, both when tasks span only a few dimensions (i.e., sα,t ≈ 0, ∀t for most α) and when tasks are not perfectly learnable. To further investigate how data impacts what a LRNN learns, we modify the task dynamics we studied earlier (constant, exponential, inverse-exponential) to have a SV at the last timestep, sT , that *does not* follow the task dynamics as the rest of the trajectory. More specifically,

$$s_{t}=\begin{cases}\delta f(\lambda,t)&\text{if}t<T\\ \kappa&\text{if}t=T\end{cases}\tag{10}$$

where κ ̸= δf(*λ, T*). Here, there is no way for the network to perfectly fit the task dynamics (which would only be the case if κ = δf(*λ, T*)). We can think of the two cases as a recurrent contribution to the task dynamics (*t < T*) and a feedforward contribution (t = T). Using this setup, we experiment with varying each of the different parameters to show that each one can affect the underlying task dynamics and consequently influence the network's behavior. For example, when studying constant task dynamics (where λ = 1 by default), we change the value of the scaling term (δ). Instead, when studying exponential task dynamics, we change the dynamic term (λ). Finally, in all settings, we vary the feedforward computation (κ) independently of the other (recurrent) parameters, to study how the network deals with the tradeoff between learning solutions for *t < T* and t = T, which cannot be learned simultaneously. In particular, this construction forces the network to either approximate the recurrent dynamics (constant: ac → *δ, b* → λ; exponential: ac → δλT, b → 1/λ)
or the feedforward computation (ac → *κ, b* → 0), each of which will incur a non-zero error from its counterpart. We run simulations varying across these different task dynamics and plot the final solutions the connectivity modes converge to (Figure 4). We find that the aforementioned tradeoff between recurrent and feedforward computation manifests as a rapid phase transition of connectivity mode values across different task dynamics and becomes sharper as trajectory length T increases. We show that this phase transition can be induced by varying either the scaling term (δ; left in Figure 4) or the dynamic term (λ; middle in Figure 4). When the error term is dominated by the feedforward computation (κ is large, δf(*λ, t*) is small), the network effectively prunes the recurrent mode (b → 0) rather

aligned weights unaligned weights 1 1 7 8 traj e cto ry l e n gth 
(T)
7 8 traj e cto ry l e n gth 
(T)
2 3 2 3 0.0 0.2 0.4 0.6 0.8 ker nel d ista nce 4 5 4 5 6 6 0.01 0.06 0.120.17 0.230.28 0.340.39 0.450.5 initialization scale 0.01 0.060.12 0.17 0.230.28 0.340.39 0.450.5 initialization scale

than approximating the recurrent computation (s1:T −1) and the input-output mode learns the feedforward computation (ac ≈ κ). As δf(*λ, t*) increases and the recurrent computation becomes more important (has a greater contribution to the loss), the network rapidly transitions to a regime where it approximates the task dynamics and approaches the 0loss solution for the case where κ = δf(*λ, T*), following the dynamic trajectory to the last timestep (dashed line in Figure 4) and ignoring the feedforward computation. To further illustrate the tradeoff between feedforward and recurrent computation, we simulate networks trained on task dynamics produced by two Dirac delta functions, which have no 0-loss solution:

$$s_{t}={\begin{cases}\beta&{\mathrm{if}}\;t=1\\ \kappa&{\mathrm{if}}\;t=T\\ 0&{\mathrm{otherwise}}\end{cases}}$$
$$\quad(11)$$

As before, we vary the recurrent and feedforward computations separately by independently changing β and κ. As shown in Figure 4 (right), we again see a sharp transition as the recurrent computation (β) increases, where the recurrent mode becomes non-zero, while simultaneously, the input-output mode decreases in magnitude. In Appendix H, we show that the phase transition depends only on the ratio of recurrent to feedforward computation (β/κ), such that the recurrent computation is pruned when this ratio is small. Using Landau theory, we show analytically that this corresponds to a first-order phrase transition for T > 3. Taken altogether, these results suggest an implicit bias towards small weights and low-rank connectivity in RNNs, mediated by an effective regularization term. They further illustrate a tradeoff between feedforward and recurrent computations, and show cases where the network prunes connectivity modes to deal with task dynamics that are not perfectly learnable, leading to low-rank connectivity. If the recurrent part of the computation is small, and/or there is a strong correlation with the input at the final timestep and the output, the network will prune that dimension, leading to a low-rank RNN. While this behavior might seem an artifact of the setting, we emphasize that the cases we study here are likely not the only task dynamics with representational tradeoffs when there are T singular values to fit to in a single dimension. It's unclear how RNNs might prioritize learning certain computations over others in various scenarios. These results, together with those on learning speed (Section 3.2), suggest a recency bias, although the cumulative effect of the recurrent computation can outweigh this.

## 3.5. Recurrence Facilitates Rich Learning

Prior work has identified two distinct learning regimes in neural networks: feature learning (rich learning), where networks learn structured task-relevant representations, and non-feature learning (*lazy* learning), where networks perform high-dimensional projections of the input (Heij et al., 2007; Yang, 2020; Farrell et al., 2023); rich learning typically occurs in networks initialized with small random weights and lazy in networks with large weights. Significant progress has been made in the theoretical understanding of these regimes, particularly in feedforward architectures
(Arora et al., 2019; Azulay et al., 2021; Braun et al., 2022; Saxe et al., 2022; Kunin et al., 2024; Domine et al. ´ , 2025). However, research into how non-feedforward architectures affect these learning regimes remain limited. Notably, Liu et al. (2024) examined the role of weight connectivity in shaping learning regimes in RNNs and Schuessler et al. (2024) showed that RNNs have different learning regimes characterized by either aligned or oblique recurrent dynamics. Building on this line of inquiry, we explore how recurrence impacts feature learning dynamics. Specifically, we investigate whether recurrent architectures impose additional constraints on the learning problem, thereby biasing the network towards the rich learning regime. The rich and lazy learning regimes are typically evaluated using the *neural tangent kernel* (NTK) (Jacot et al., 2018), which is constant during lazy learning and non-constant during rich learning. We derive the NTK for finite-width LRNNs (Appendix I), which we then use to study what learning regimes emerge in LRNNs with different initializations and trajectory lengths. Importantly, our derivation does not place any assumptions on the alignment of LRNN
weights as in the prior sections.

To quantify feature learning, we measure the kernel distance between the NTK at initialization and the end of training for LRNNs trained on constant task dynamics as a function of trajectory length and weight initialization scale in both the aligned and unaligned case (Figure 5). As expected, we see that the kernel moves further in networks with smaller initializations relative to the target (= 1), but surprisingly, the NTK still moves substantially even across larger initializations (Appendix J). We also find that the kernel distance increases as the network transitions from a feedforward net-

yT = mean(x1 : T)
yT = sum(x1 : T)
0 500 1000 1500 2000 2500 0.0 0.2 0.4 0.6 0.8 1.0 0 100 200 300 400 500 0.0 0.2 0.4 0.6 0.8 1.0 1.2 sin gul ar va lues sin gul ar va lues recurrent input-output global solution T = 4 extrapolation to T = 50 extrapolation to T = 75 extrapolation to T = 100 0 100 200 300 400 500 training steps (tθ)
0 200 400 600 800 1000 1200 0 500 1000 1500 2000 2500 training steps (tθ)
0 20 40 60 lo s s 
()
lo s s 
()
work (i.e., T = 1) to a recurrent network, indicating greater feature learning. In Appendix J, we further study the effects of rotational tasks, larger initializations and widths, and independently-initialized modes.

## 3.6. Sensory Integration Task

Although in this work we study a simplified setting that cannot fully capture all of the rich neural dynamics exhibited in animals, LRNNs can still learn and perform some basic computations studied in neuroscience, such as temporal integration of input and rotational dynamics (Khona & Fiete, 2021). Here, as a proof of concept, we study a sensory integration task where we remove our prior assumptions about whitened data and aligned weights, and show that the insights developed in our theoretical model generalize to predict behavior in this setting. We consider two versions of a sensory integration task where the network is given noisy input in several dimensions and tasked with producing either the mean input activity in each dimension, or the sum of the input in each dimension. We train LRNNs with small random weights, making no other architectural assumptions. In such a task, the output is equally correlated across inputs at all trajectory timesteps, thus exhibiting *constant* dynamics. Our theory predicts that networks trained on tasks with constant dynamics produce recurrent modes equal to one, and that the input-output modes learn to scale these dynamics. In the case where the output is a sum of inputs, no additional scaling is necessary so input-output modes should become one, while in the case where the output is the mean, the input-output modes should become 1/T to appropriately scale.

Our theory also predicts that task dynamics that produce solutions dependent on trajectory length will be unable to extrapolate to other trajectory lengths. Thus, since inputoutput modes should learn 1/T for the mean-integration task, we do not expect it to extrapolate to other trajectory lengths, while we would expect the sum-integration task to extrapolate perfectly. By simulating networks on the sensory integration tasks and plotting the network SVs, we see that our theory indeed predicts the solutions found by networks for both mean-integration and sum-integration (top row in Figure 6). As expected, we also find that the networks trained on sum-integration tasks are able to extrapolate to other trajectory lengths perfectly, while networks trained on meanintegration accumulate error as a function of the difference in trajectory length from that trained on (bottom row in Figure 6). In Appendix L, we further extend this setting to show that our predictions about stability (early-importance versus late-importance dynamics) are validated in networks without our assumptions. In summary, these results illustrate the application of our theoretical framework for understanding the behavior and capabilities of LRNNs more generally.

## 4. Discussion & Related Work

Summary of results. In this work, we extend the growing literature on learning dynamics to a new architecture, linear RNNs. We derive an analytical solution to the energy function and learning dynamics of LRNNs under certain conditions, using a novel approach that accounts for task dynamics. Unlike feedforward networks, LRNNs learn data singular values ordered by both their scale and temporal precedence, with larger and later singular values being learned first. We identify how task dynamics impact solution stability and extrapolation ability, an often understudied aspect of RNN dynamics. We further reveal a tradeoff between recurrent and feedforward computation that leads to low-rank solutions, mediated by an effective regularization term in the energy function. We extend existing work on rich and lazy learning in RNNs beyond the effect of initial connectivity by deriving the NTK for finite-width LRNNs and showing that recurrence encourages feature learning. Finally, we demonstrate an application of our results in a sensory integration task where we relax our prior assumptions and find that our theory explains the behavior of LRNNs. Learning dynamics in linear networks. Differing from prior work on learning dynamics in linear networks (Saxe et al., 2014; 2018; 2022; Braun et al., 2022; Sandbrink et al.,
2024; Domine et al. ´ , 2025), we study a recurrent network, allowing us to analyze how other architectures constrain optimization in ways that differ from feedforward ones. Notably, Schuessler et al. (2020b) previously studied learning dynamics in LRNNs to study how networks make low-rank changes to their connectivity, but used a task with constant input in the limit of infinite trajectory length, and Smekal ´ et al. (2024) studied learning in the frequency domain but focused on the effects of overparameterization on convergence time. Instead, our work accounts for the effect of task dynamics, which are critically important for modeling and understanding dynamic cognitive behavior.

Stability and extrapolation. The problem of stability in training RNNs is a well-studied problem (Bengio et al.,
1994; Hochreiter et al., 2001; Pascanu et al., 2012; Zucchet & Orvieto, 2024) with numerous proposed solutions (Hochreiter & Schmidhuber, 1997; Le et al., 2015; Orvieto et al., 2023; Zucchet et al., 2023). Here, we highlight an additional, understudied factor - the impact of task dynamics. We show how certain task dynamics (those with early-importance) can lead to unstable training regimes as a result of the solutions they drive the network to. This suggests that practical approaches to such problems should take task dynamics into account when designing new solutions. Although less theoretically understood (Emami et al., 2021; Cohen-Karlik et al., 2022; Beiran et al. ´ , 2023), our framework sheds light on how task dynamics impact LRNN's extrapolation to sequence lengths different to those in the training set and how this is driven by a mismatch between architecture and the latent structure of the data. Low-rank connectivity. Networks with low-rank connectivity have been used as more interpretable models from which to study dynamics related to cognition (Mastrogiuseppe & Ostojic, 2018; Schuessler et al., 2020a;b; Dubreuil et al., 2022), motivated by the fact that neural population activity is often low-dimensional, and it's been shown that RNNs learn low-rank solutions along task dimensions (Schuessler et al., 2020b). Complementing this work, we identify an effective regularization term in the energy function that incentivizes small-weight solutions and demonstrate specific cases of task dynamics where RNNs prune connectivity modes resulting in low-rank connectivity. Rich and lazy learning. Neural networks can lie in two different learning regimes (so-called rich or lazy) depending on their weight initialization and width, and there is increasing evidence that these regimes are related to the representational geometry of different brain regions (Rigotti et al., 2013; Bernardi et al., 2020; Flesch et al., 2022; Farrell et al., 2023; Payeur et al., 2023). Most theoretical studies of rich and lazy learning have been done in feedforward networks, with the exception of Liu et al. (2024), which showed that connectivity rank impacts features learning, and Schuessler et al. (2024), which showed that the scale of the readout acts as a control parameter between aligned and oblique dynamics in RNNs. Here we reveal an additional factor impacting feature learning: the effect of recurrence. Although lazy learning is still possible in recurrent networks (e.g., with larger weight initializations or widths), we find that recurrence induces substantial NTK movement.

Limitations and future directions. In this work, we perform our theoretical analysis on LRNNs with *data-aligned* weights, trained on tasks with input-output correlations that have constant singular/eigen-vectors. While our form based on SVD severely restricts the expressivity of the network, our derivation based on an eigendecomposition (Appendix N) relieves many of these limitations, whereby our framework and results naturally extend to this case. Interestingly, other work has shown that there are some practical justifications for using diagonal state spaces (Gupta & Berant, 2022; Orvieto et al., 2023), and it's also a common choice in theoretical work (Hazan et al., 2018; Zucchet & Orvieto, 2024). In this paper, we make several choices when constructing the setting we study, including our focus on the single-output case (rather than an autoregressive one), our initialization of the hidden layer at 0, and our use of small square networks. While we do extend our main derivations to the autoregressive (T-output) case, fully characterizing the behavior of RNNs will require charting these different settings. Finally, although linear networks are more tractable, many computations of interest can only be implemented in RNNs with nonlinear dynamics. Thus, an important future direction of theory will be to find new ways to study learning in networks with nonlinearities, potentially through gating (Saxe et al., 2022; Sandbrink et al., 2024; Jarvis et al., 2025). It's an open question to what extent the findings in this work will generalize to other settings, but we believe the framework we have constructed is flexible and will support new research inquiries in this direction.

## 5. Conclusion

This work presents a theoretical study of learning dynamics in linear RNNs and the effect of temporally-structured data. It presents one of the few studies of learning dynamics in recurrent networks, and, to our best knowledge, the first to account for the effect of task dynamics and to more explicitly connect recurrence to feature learning by studying the transition from feedforward to recurrent networks using trajectory length. This study generates new insights into the learning process of RNNs and encourages further theoretical developments to consider the learning process and the impact of temporal data when studying RNNs. We hope future work can characterize how complex dynamics, such as those in the brain, are developed during learning and ultimately, help us better understand cognition from a dynamic perspective.

## Acknowledgements Impact Statement References

Braun, L., Domine, C., Fitzgerald, J., and Saxe, A. Exact ´
learning dynamics of deep linear networks with prior knowledge. Advances in Neural Information Processing Systems, 35:6615–6629, 2022.

AP is funded by the Imperial College London President's PhD Scholarship. CD was supported by the Gatsby Charitable Foundation (GAT3755). This research was funded in part by the Wellcome Trust [216386/Z/19/Z].

Chaisangmongkon, W., Swaminathan, S. K., Freedman, D. J., and Wang, X.-J. Computing by robust transience: How the fronto-parietal network performs sequential, category-based decisions. *Neuron*, 93:1504–1517.e4, 2017.

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

Cohen-Karlik, E., David, A. B., Cohen, N., and Globerson, A. On the implicit bias of gradient descent for temporal extrapolation. International Conference on Artificial Intelligence and Statistics, 151, 2022.

Cohen-Karlik, E., Menuhin-Gruman, I., Giryes, R., Cohen, N., and Globerson, A. Learning low dimensional state spaces with overparameterized recurrent neural nets.

International Conference on Learning Representations, 2023.

Alemohammad, S., Wang, Z., Balestriero, R., and Baraniuk, R. The recurrent neural tangent kernel. International Conference on Learning Representations, 2021.

Arora, S., Cohen, N., Golowich, N., and Hu, W. A convergence analysis of gradient descent for deep linear neural networks. International Conference on Learning Representations, 2019.

Domine, C. C., Anguita, N., Proca, A. M., Braun, L., Kunin, ´
D., Mediano, P. A., and Saxe, A. M. From lazy to rich: Exact learning dynamics in deep linear networks. International Conference on Learning Representations, 2025.

Atanasov, A., Bordelon, B., and Pehlevan, C. Neural networks as kernel learners: The silent alignment effect. International Conference on Learning Representations, 2022.

Doya, K. Universality of fully-connected recurrent neural networks. *IEEE Transactions on Neural Networks*, 1993.

Driscoll, L., Shenoy, K., and Sussillo, D. Flexible multitask computation in recurrent networks utilizes shared dynamical motifs. *Nature Neuroscience*, 2024.

Azulay, S., Moroshko, E., Nacson, M. S., Woodworth, B. E.,
Srebro, N., Globerson, A., and Soudry, D. On the implicit bias of initialization shape: Beyond infinitesimal mirror descent. *International Conference on Machine Learning*, pp. 468–477, 2021.

Dubreuil, A. M., Valente, A., Beiran, M., Mastrogiuseppe, ´
F., and Ostojic, S. The role of population structure in computations through neural dynamics. *Nature Neuroscience*, 25:783 - 794, 2022.

Barak, O. Recurrent neural networks as versatile tools of neuroscience research. *Current Opinion in Neurobiology*, 46:1–6, 2017.

Durbin, J. and Koopman, S. J. Time Series Analysis by State Space Methods. OUP Oxford, 2012.

Beiran, M., Meirhaeghe, N., Sohn, H., Jazayeri, M., and ´
Ostojic, S. Parametric control of flexible timing through low-dimensional neural manifolds. *Neuron*, 111:739– 753.e8, 2023.

Emami, M. M., Sahraee-Ardakan, M., Pandit, P., Rangan, S., and Fletcher, A. K. Implicit bias of linear rnns. International Conference on Machine Learning, 2021.

Engel, T. A., Chaisangmongkon, W., Freedman, D. J., and Wang, X.-J. Choice-correlated activity fluctuations underlie learning of neuronal category representation. Nature Communications, 6, 2015.

Bengio, Y., Simard, P. Y., and Frasconi, P. Learning longterm dependencies with gradient descent is difficult. *IEEE* Transactions on Neural Networks, 5 2:157–66, 1994.

Bernardi, S., Benna, M. K., Rigotti, M., and Salzman, C. D.

The geometry of abstraction in the hippocampus and prefrontal cortex. *Cell*, 183:954–967.e21, 2020.

Farrell, M., Recanatesi, S., and Shea-Brown, E. From lazy to rich to exclusive task representations in neural networks and neural codes. *Current opinion in neurobiology*, 83: 102780, 2023.

Bordelon, B., Cotler, J., Pehlevan, C., and Zavatone-Veth, J. A. Dynamically learning to integrate in recurrent neural networks. 2025.

Farrell, M. T., Recanatesi, S., Moore, T., Lajoie, G., and Shea-Brown, E. Gradient-based learning drives robust representations in recurrent neural networks by balancing compression and expansion. *Nature Machine Intelligence*, 4:564 - 573, 2022.

Flesch, T., Juechems, K., Dumbalska, T., Saxe, A., and Summerfield, C. Orthogonal representations for robust context-dependent task performance in brains and neural networks. *Neuron*, 110(7):1258–1270, 2022.

Fort, S., Dziugaite, G. K., Paul, M., Kharaghani, S., Roy, D. M., and Ganguli, S. Deep learning versus kernel learning: an empirical study of loss landscape geometry and the time evolution of the neural tangent kernel.

Advances in Neural Information Processing Systems, 33:
5850–5861, 2020.

Gu, A. and Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. Conference on Language Modelling, 2024.

Gu, A., Dao, T., Ermon, S., Rudra, A., and Re, C. Hippo: Re- ´
current memory with optimal polynomial projections. Advances in Neural Information Processing Systems, 2020.

Gu, A., Goel, K., and Re, C. Efficiently modeling long ´
sequences with structured state spaces. International Conference on Learning Representations, 2022.

Gupta, A. and Berant, J. Diagonal state spaces are as effective as structured state spaces. Advances in Neural Information Processing Systems, abs/2203.14343, 2022.

Hazan, E., Lee, H., Singh, K., Zhang, C., and Zhang, Y. Spectral filtering for general linear dynamical systems. Advances in Neural Information Processing Systems, 2018.

Heij, C., Ran, A. C., and Van Schagen, F. Introduction to Mathematical Systems Theory. Springer, 2007.

Hochreiter, S. and Schmidhuber, J. Long short-term memory.

Neural Computation, 9(8):1735–1780, 1997.

Hochreiter, S., Bengio, Y., Frasconi, P., and Schmidhuber, J. Gradient Flow in Recurrent Nets: the Difficulty of Learning Long-Term Dependencies. IEEE, 2001.

Jacot, A., Gabriel, F., and Hongler, C. Neural tangent kernel:
Convergence and generalization in neural networks. In Advances in Neural Information Processing Systems, pp. 8571–8580, 2018.

Jarvis, D., Klein, R., Rosman, B., and Saxe, A. M. Make haste slowly: A theory of emergent structured mixed selectivity in feature learning relu networks. International Conference on Learning Representations, 2025.

Khona, M. and Fiete, I. R. Attractor and integrator networks in the brain. *Nature Reviews Neuroscience*, 23:744 - 766, 2021.

Kunin, D., Raventos, A., Domin ´ e, C., Chen, F., Klindt, ´
D., Saxe, A., and Ganguli, S. Get rich quick: exact solutions reveal how unbalanced initializations promote rapid feature learning. Advances in Neural Information Processing Systems, 2024.

Le, Q. V., Jaitly, N., and Hinton, G. E. A simple way to initialize recurrent networks of rectified linear units. ArXiv, 2015.

Liu, Y. H., Baratin, A., Cornford, J., Mihalas, S., Shea-
Brown, E., and Lajoie, G. How connectivity structure shapes rich and lazy learning in neural circuits. International Conference on Learning Representations, 2024.

Lufkin, L., Saxe, A. M., and Grant, E. Nonlinear dynamics of localization in neural receptive fields. In Advances in Neural Information Processing Systems, 2024.

Mante, V., Sussillo, D., Shenoy, K. V., and Newsome, W. T.

Context-dependent computation by recurrent dynamics in prefrontal cortex. *Nature*, 503:78 - 84, 2013.

Masse, N. Y., Yang, G. R., Song, H. F., Wang, X.-J., and Freedman, D. J. Circuit mechanisms for the maintenance and manipulation of information in working memory.

Nature Neuroscience, 22:1159 - 1167, 2018.

Masse, N. Y., Rosen, M. C., and Freedman, D. J. Reevaluating the role of persistent neural activity in short-term memory. *Trends in Cognitive Sciences*, 24:242–258, 2020.

Mastrogiuseppe, F. and Ostojic, S. Linking connectivity, dynamics, and computations in low-rank recurrent neural networks. *Neuron*, 99(3):609–623, 2018.

Molano-Mazon, M., Barbosa, J., Pastor-Ciurana, J., Fradera, ´
M., Zhang, R.-Y., Forest, J., del Pozo, J., Ji-An, L., Cueva, C., de la Rocha, J., Narain, D., and Yang, G. R. NeuroGym: An open resource for developing and sharing neuroscience tasks. 2022.

Orhan, A. E. and Ma, W. J. A diverse range of factors affect the nature of neural representations underlying short-term memory. *Nature Neuroscience*, 22:275 - 283, 2018.

Orvieto, A., Smith, S. L., Gu, A., Fernando, A., Gulcehre, C., Pascanu, R., and De, S. Resurrecting recurrent neural networks for long sequences. *International Conference* on Machine Learning, 2023.

Orvieto, A., De, S., Gulcehre, C., Pascanu, R., and Smith, S. L. Universality of linear recurrences followed by nonlinear projections: Finite-width guarantees and benefits of complex eigenvalues. 2024.

Pascanu, R., Mikolov, T., and Bengio, Y. On the difficulty of training recurrent neural networks. In International Conference on Machine Learning, 2012.

Payeur, A., Orsborn, A. L., and Lajoie, G. Neural manifolds and learning regimes in neural-interface tasks. *bioRxiv*, 2023.

Remington, E. D., Egger, S. W., Narain, D., Wang, J., and Jazayeri, M. A dynamical systems perspective on flexible motor timing. *Trends in Cognitive Sciences*, 22(10):938– 952, 2018a.

Remington, E. D., Narain, D., Hosseini, E. A., and Jazayeri, M. Flexible sensorimotor computations through rapid reconfiguration of cortical dynamics. *Neuron*, 98:1005– 1019.e5, 2018b.

Rigotti, M., Barak, O., Warden, M. R., Wang, X.-J., Daw, N. D., Miller, E. K., and Fusi, S. The importance of mixed selectivity in complex cognitive tasks. *Nature*, 497:585–590, 2013.

Sandbrink, K. J., Bauer, J. P., Proca, A. M., Saxe, A. M.,
Summerfield, C., and Hummos, A. Flexible task abstractions emerge in linear networks with fast and bounded units. 2024.

Saxe, A., Sodhani, S., and Lewallen, S. J. The neural race reduction: Dynamics of abstraction in gated networks.

In *International Conference on Machine Learning*, pp. 19287–19309. PMLR, 2022.

Saxe, A. M., McClelland, J. L., and Ganguli, S. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. International Conference on Learning Representations, 2014.

Saxe, A. M., McClelland, J. L., and Ganguli, S. A mathematical theory of semantic development in deep neural networks. Proceedings of the National Academy of Sciences, 116:11537 - 11546, 2018.

Saxe, A. M., Nelli, S., and Summerfield, C. If deep learning is the answer, what is the question? Nature Reviews Neuroscience, 22:55 - 67, 2020.

Schuessler, F., Dubreuil, A., Mastrogiuseppe, F., Ostojic, S.,
and Barak, O. Dynamics of random recurrent networks with correlated low-rank structure. Physical Review Research, 2(1):013111, 2020a.

Schuessler, F., Mastrogiuseppe, F., Dubreuil, A., Ostojic, S.,
and Barak, O. The interplay between randomness and structure during learning in rnns. In Advances in Neural Information Processing Systems, 2020b.

Schuessler, F., Mastrogiuseppe, F., Ostojic, S., and Barak, O. Aligned and oblique dynamics in recurrent neural networks. *eLife*, 2024.

Schafer, A. M. and Zimmermann, H. G. Recurrent neu- ¨
ral networks are universal approximators. *International* Journal of Neural Systems, 17(4):253–63, 2007.

Smekal, J., Smith, J., Kleinman, M., Biderman, D., and ´
Linderman, S. Towards a theory of learning dynamics in deep state space models. *ICML 2024 Next Generation of* Sequence Modeling Architectures Workshop, 2024.

Sompolinsky, H., Crisanti, A., and Sommers, H.-J. Chaos in random neural networks. *Physical review letters*, 61 (3):259, 1988.

Sussillo, D. and Barak, O. Opening the black box: Lowdimensional dynamics in high-dimensional recurrent neural networks. *Neural Computation*, 25:626–649, 2013.

Turner, E. and Barak, O. The simplicity bias in multi-task rnns: Shared attractors, reuse of dynamics, and geometric representation. In *Advances in Neural Information* Processing Systems, 2023.

Turner, E., Dabholkar, K., and Barak, O. Charting and navigating the space of solutions for recurrent neural networks. In Advances in Neural Information Processing Systems, 2021.

Vyas, S., Golub, M. D., Sussillo, D., and Shenoy, K. V.

Computation through neural population dynamics. Annual Review of Neuroscience, 43:249–275, 2020.

Wang, J., Narain, D., Hosseini, E. A., and Jazayeri, M.

Flexible timing by temporal scaling of cortical responses. Nature Neuroscience, 21:102 - 110, 2017.

Yang, G. Tensor programs ii: Neural tangent kernel for any architecture. *ArXiv*, 2020.

Yang, G. R., Joglekar, M. R., Song, H. F., Newsome, W. T.,
and Wang, X.-J. Task representations in neural networks trained to perform many cognitive tasks. Nature Neuroscience, 22:297 - 306, 2019.

Zucchet, N. and Orvieto, A. Recurrent neural networks:
vanishing and exploding gradients are not the end of the story. *Advances in Neural Information Processing* Systems, 2024.

Zucchet, N., Meier, R., Schug, S., Mujika, A., and Sacramento, J. Online learning of long-range dependencies. Advances in Neural Information Processing Systems, 2023.

## Appendix

| Table of Contents A Notation                                             | 14                                                                                                       |    |
|--------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|----|
| B                                                                        | Derivation of gradient flow equations and energy function                                                | 15 |
| C                                                                        | Exact solution of input-output connectivity modes                                                        | 17 |
| D                                                                        | Local approximation of recurrent connectivity modes                                                      | 17 |
| D.1                                                                      | Analytical approximation using Faa di Bruno formula and Bell polynomials `                               | 18 |
| E                                                                        | Effect of task dynamics on the ordering of learning                                                      | 19 |
| F                                                                        | Zero-loss solutions only exist for inverse-exponential task dynamics                                     | 20 |
| F.1                                                                      | Discussion on exponential task dynamics as reparameterization of inverse-exponential task dynamics . . . | 20 |
| F.2                                                                      | Proof                                                                                                    | 20 |
| F.3                                                                      | Global solutions of task dynamics                                                                        | 21 |
| G                                                                        | Effective regularization term incentivizes small-weights                                                 | 23 |
| H                                                                        | Connectivity modes exhibit phase transition as a function of task dynamics                               | 23 |
| H.1                                                                      | T = 3 case                                                                                               | 25 |
| H.2                                                                      | T > 3 case                                                                                               | 25 |
| I                                                                        | Finite-width neural tangent kernel of LRNN                                                               | 26 |
| J                                                                        | Analyzing the impact of recurrence on feature learning                                                   | 28 |
| K                                                                        | Impact of connectivity modes on the energy function                                                      | 31 |
| L                                                                        | Early-importance task dynamics lead to unstable solutions                                                | 32 |
| M Extending to the (autoregressive) T-output case                        | 33                                                                                                       |    |
| M.1 Exact solution of input-output connectivity modes                    |                                                                                                          | 34 |
| M.2 Local approximation of recurrent connectivity modes                  | 35                                                                                                       |    |
| M.3 Zero-loss solutions only exist for inverse-exponential task dynamics | 35                                                                                                       |    |
| M.4 Existence of effective regularization term                           | 35                                                                                                       |    |
| M.5 Neural tangent kernel                                                |                                                                                                          | 35 |
| N                                                                        | Generalizing gradient flow equations to the eigenspace to capture rotations                              | 35 |
| O                                                                        | Learning dynamics of rotations in the complex plane                                                      | 37 |
| P                                                                        | Simulations                                                                                              | 41 |
| P.1                                                                      | LRNN initialization                                                                                      | 41 |
| P.2                                                                      | Training                                                                                                 | 41 |
| P.3                                                                      | Recovering connectivity modes                                                                            | 42 |
| P.4                                                                      | Tasks                                                                                                    | 42 |

A. Notation

| Table A1. Notation             |                                                                                                                          |                                                                                                      |
|--------------------------------|--------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| Symbol                         | Description                                                                                                              |                                                                                                      |
| t                              | trajectory timestep                                                                                                      |                                                                                                      |
| T                              | trajectory length (final timestep)                                                                                       |                                                                                                      |
| tθ                             | learning timestep                                                                                                        |                                                                                                      |
| τ                              | learning timescale (inverse learning rate)                                                                               |                                                                                                      |
| η                              | learning rate                                                                                                            |                                                                                                      |
| P                              | dataset size                                                                                                             |                                                                                                      |
| p                              | data sample index                                                                                                        |                                                                                                      |
| Nx                             | input size                                                                                                               |                                                                                                      |
| Nh                             | hidden size                                                                                                              |                                                                                                      |
| Ny                             | output size                                                                                                              |                                                                                                      |
| xp,t ∈ R Nx                    | input sample p at timestep t                                                                                             |                                                                                                      |
| ht+1 = Whht + Wxxt ∈ R Nh      | hidden state at timestep t + 1                                                                                           |                                                                                                      |
| yˆt = Wyht+1 ∈ R Ny            | model output at timestep t                                                                                               |                                                                                                      |
| yp,t ∈ R Ny                    | output target at timestep t                                                                                              |                                                                                                      |
| Wx ∈ R Nh×Nx                   | input weight matrix                                                                                                      |                                                                                                      |
| Wh ∈ R Nh×Nh                   | recurrent weight matrix                                                                                                  |                                                                                                      |
| Wy ∈ R Ny×Nh                   | output weight matrix                                                                                                     |                                                                                                      |
| Y Xt = PP Σ                    | yp,T x ⊤ p,t                                                                                                             | input-output correlation matrix between input xt at trajectory timestep t and final output target yT |
| p=1                            |                                                                                                                          |                                                                                                      |
| Σ XtXt ′ = PP p=1 xp,tx ⊤ p,t′ | input-input correlation matrix between input xt at trajectory timestep t and xt ′ at t ′                                 |                                                                                                      |
| UyStV ⊤ x = ΣY Xt              | singular value decomposition of input-output correlation matrix for input at timestep t                                  |                                                                                                      |
| P DtP † = ΣY Xt                | eigendecomposition of input-output correlation matrix for input at timestep t                                            |                                                                                                      |
| Wx                             | diagonalized input matrix                                                                                                |                                                                                                      |
| Wh                             | diagonalized recurrent matrix                                                                                            |                                                                                                      |
| Wy                             | diagonalized output matrix                                                                                               |                                                                                                      |
| α                              | singular/eigen-value dimension                                                                                           |                                                                                                      |
| aα                             | input connectivity mode at dimension α                                                                                   |                                                                                                      |
| bα                             | recurrent connectivity mode at dimension α                                                                               |                                                                                                      |
| cα                             | output connectivity mode at dimension α                                                                                  |                                                                                                      |
| sα,t                           | singular value of St at dimension α                                                                                      |                                                                                                      |
| dα,t                           | eigenvalue of Dt at dimension α                                                                                          |                                                                                                      |
| E                              | energy function of connectivity modes decoupled along singular/eigen-value dimensions                                    |                                                                                                      |
| δ                              | constant component/parameter of data singular/eigen-values (when st, dt = δf(λ, t))                                      |                                                                                                      |
| λ                              | dynamic component/parameter of data singular/eigen-values (when st, dt = δf(λ, t))                                       |                                                                                                      |
| f(λ, t) = 1                    | constant task dynamics                                                                                                   |                                                                                                      |
| f(λ, t) = λ T −t               | inverse-exponential task dynamics                                                                                        |                                                                                                      |
| f(λ, t) = λ t                  | exponential task dynamics                                                                                                |                                                                                                      |
| κ                              | 'feedforward computation' (= sT ) in phase transition experiments                                                        |                                                                                                      |
| β                              | 'recurrent computation' (= s1) in Dirac delta task dynamics                                                              |                                                                                                      |
| ˜                              | indicating teacher parameters                                                                                            |                                                                                                      |
| u = ac                         | single variable for input-output modes                                                                                   |                                                                                                      |
| ⋆                              | indicating global solution                                                                                               |                                                                                                      |
| Σ YtXt ′ = PP p=1 yp,tx ⊤ p,t′ | input-output correlation matrix between input at timestep t ′ and output at timestep t (t ′ ≤ t) for autoregressive case |                                                                                                      |
| †                              | conjugate transpose of a matrix                                                                                          |                                                                                                      |
| ∗                              | complex conjugate                                                                                                        |                                                                                                      |
| Rδ                             | radial component of δ (= Rδe ϕδi )                                                                                       |                                                                                                      |
| ϕδ                             | angle component of δ                                                                                                     |                                                                                                      |
| Rλ                             | radial component of λ (= Rλe ϕλi )                                                                                       |                                                                                                      |
| ϕλ                             | angle component of λ                                                                                                     |                                                                                                      |

## 14 B. Derivation Of Gradient Flow Equations And Energy Function

Recall our model definition as

$$h_{t+1}=W_{h}h_{t}+W_{x}x_{t}$$
$\mathbf{h}_{t+1}=W_{h}\mathbf{h}_{t}+W_{x}\mathbf{x}_{t}$  $$=\sum_{i=1}^{t}W_{h}^{t-i}W_{x}\mathbf{x}_{i}$$ $$\hat{\mathbf{y}}_{t}=W_{y}\mathbf{h}_{t+1}$$

with a loss of

$${\cal L}=\frac{1}{2}\sum_{p=1}^{P}\|\mathbf{y}_{p,T}-W_{y}(\sum_{i=1}^{T}W_{h}^{T-i}W_{x}\mathbf{x}_{p,i})\|^{2}\tag{1}$$

By taking the derivative of the loss with respect to each set of parameters Wx, Wh, Wy, we get the following equations

p=1  X T i=1  W (T −i)⊤ h W⊤ y  yT ,p −X T j=1 WyW T −j h Wxxj,p)   x ⊤ i,p     (16) ∂L ∂Wx = −X P p=1   W (r)⊤ h W⊤ y  j=1 WyW T −j h Wxxj,p     ∂L ∂Wh = −X P  T X−1 i=1 T X−i−1 r=0 yT ,p −X T  x ⊤ i,pW⊤ x W (T −i−1−r)⊤ h p=1  i=1    j=1 WyW T −j h Wxxj,p   x ⊤ i,pW⊤ x W (T −i)⊤ h    ∂L ∂Wy = −X P X T yT ,p −X T  (18)
$$(12)$$
(13)  $\binom{14}{2}$  (14)  ... 
$$(15)$$
(16)  $$\begin{array}{l}\mathbf{(17)}\end{array}$$ = (18)  $$\begin{array}{l}\mathbf{(18)}\end{array}$$ . 
 (17)
We define the input-output correlation matrices between an input at trajectory timestep t and the final output as

$$\Sigma^{Y X_{t}}=\sum_{p=1}^{P}\mathbf{y}_{p,T}\mathbf{x}_{p,t}^{\top}\tag{1}$$
$$(19)$$
$$(20)$$

and the input-input correlation matrices between two inputs at trajectory timesteps *t, t*′as

$$\Sigma^{X_{t}X_{t^{\prime}}}=\sum_{p=1}^{P}\mathbf{x}_{p,t}\mathbf{x}_{p,t^{\prime}}^{\top}\tag{1}$$
$$(21)$$
$$(22)$$
$$(23)$$
(24)  $$\begin{array}{l}\small\mathbf{(25)^{}}\end{array}$$ . 
Under the assumption of whitened input with 0 mean, the input-input correlation matrices become Σ
XtXt
′ = 0, ∀t ̸= t
′and Σ
XtXt = I. Substituting the correlation matrices and assuming the gradient flow regime where the learning rate (η = 1/τ )
is small, we can rewrite the gradient equations above as a set of differential equations over training time tθ

$$\tau\frac{d}{dt_{\theta}}W_{x}=\sum_{i=1}^{T}W_{h}^{(T-i)\top}W_{y}^{\top}(\Sigma^{YX_{i}}-W_{y}W_{h}^{T-i}W_{x})$$ $$\tau\frac{d}{dt_{\theta}}W_{h}=\sum_{i=1}^{T-1}\sum_{r=0}^{T-i-1}W_{h}^{(r)\top}W_{y}^{\top}(\Sigma^{YX_{i}}-W_{y}W_{h}^{T-i}W_{x})W_{x}^{\top}W_{h}^{(T-i-1-r)\top}$$ $$\tau\frac{d}{dt_{\theta}}W_{y}=\sum_{i=1}^{T}(\Sigma^{YX_{i}}-W_{y}W_{h}^{T-i}W_{x})W_{x}^{\top}W_{h}^{(T-i)\top}$$

We assume that the input-output correlation matrices have constant left and right singular vectors across trajectory timesteps, such that only their singular values vary through time. Although this may seem like a restrictive assumption, note that this assumption holds for any data generated by a teacher linear RNN with weights that can be diagonalized.

Proof. Data generated by a linear RNN teacher parameterized by W˜x, W˜h, W˜y that can be diagonalized with SVD has constant left and right singular vectors across trajectory timesteps.

$$\Sigma^{YX_{t}}=\sum_{p=1}^{P}\mathbf{y}_{p,T}\mathbf{x}_{p,t}^{\top}$$ $$=\sum_{p=1}^{P}\tilde{W}_{y}\tilde{W}_{h}^{T-t}\tilde{W}_{x}\mathbf{x}_{p,t}\mathbf{x}_{p,t}^{\top}$$ $$=\tilde{W}_{y}\tilde{W}_{h}^{T-t}\tilde{W}_{x}$$
$$(26)$$
(27)  $\binom{28}{2}$  . 
$$(29)$$

We assume W˜h is diagonalized by orthogonal matrices R˜y, R˜x such that W˜y = UySyR˜⊤
y, W˜x = R˜xSxV
⊤
x, W˜h = R˜yShR˜⊤
x.

Then,

$$\bar{W}_{y}\bar{W}_{h}^{T-t}\bar{W}_{x}=U_{y}S_{t}V_{x}^{\top}\tag{1}$$
$$\lceil\!\!\!\perp\!\!\!\perp$$
x(29)
We place no additional assumptions on the temporal dynamics of the singular values through time St, such that they could be generated by any dynamic process. Substituting the singular value decomposition (SVD) of the data-correlation matrix into the gradient flow equations yields

$$\tau\frac{d}{dt_{\theta}}W_{x}=\sum_{i=1}^{T}W_{h}^{(T-i)\top}W_{y}^{\top}(U_{y}S_{i}V_{x}^{\top}-W_{y}W_{h}^{T-i}W_{x})\tag{30}$$ $$\tau\frac{d}{dt_{\theta}}W_{h}=\sum_{i=1}^{T-1}\sum_{r=0}^{T-i}W_{h}^{(\tau)\top}W_{y}^{\top}(U_{y}S_{i}V_{x}^{\top}-W_{y}W_{h}^{T-i}W_{x})W_{x}^{\top}W_{h}^{(T-i-1-r)\top}$$ (31) $$\tau\frac{d}{dt_{\theta}}W_{y}=\sum_{i=1}^{T}(U_{y}S_{i}V_{x}^{\top}-W_{y}W_{h}^{T-i}W_{x})W_{x}^{\top}W_{h}^{(T-i)\top}\tag{32}$$

Similarly to Saxe et al. (2014; 2018), we assume the LRNN is *data-aligned* at initialization such that for some orthogonal matrices Ry, Rx, R⊤
y Wh(0)Rx = Wh(0), R⊤
x Wx(0)Vx = Wx(0), U
⊤
y Wy(0)Ry = Wy(0), where Wx, Wh, Wy are diagonal matrices. Atanasov et al. (2022) showed that this alignment happens early in training for networks initialized with small random weights. Performing a change of variables in the gradient flow equations and simplifying yields,

τd dtθ Wx =X T i=1 W (T −i)⊤ h W ⊤ y (Si − WyW T −i h Wx) (33) τd dtθ Wh = T X −1 i=1 T X −i−1 r=0 W (r)⊤ h W ⊤ y (Si − WyW T −i h Wx)W ⊤ x W (T −i−1−r)⊤ h(34) τ d dtθ Wy =X T i=1 (Si − WyW T −i h Wx)W ⊤ x W (T −i)⊤ h(35)
Let aα, bα, cα be the α th diagonal entry of Wx, Wh, Wy, respectively, and sα,t be the α th singular value of St. We can then rewrite the above equations in terms of these variables, or *connectivity modes* that decouple along singular value dimensions

(33)  $$\begin{array}{l}\small\mathbf{(34)^{}}\end{array}$$ = $$\begin{array}{l}\small\mathbf{(35)^{}}\end{array}$$ . 
α,

τ d dtθ aα =X T i=1 b T −i α cα(sα,i − cαb T −i α aα) (36) τ d dtθ bα = T X−1 i=1 T X−i−1 r=0 b (r) α cα(sα,i − cαb T −i α aα)aαb (T −i−1−r) α (37) = T X−1 i=1 (T − i)cα(sα,i − cαb T −i α aα)aαb (T −i−1) α (38) τ d dtθ cα =X T i=1 (sα,i − cαb T −i α aα)aαb T −i α (39)
These dynamics arise from gradient descent on the energy function

$$E=\frac{1}{2\tau}\sum_{\alpha}\sum_{i=1}^{T}(s_{\alpha,i}-c_{\alpha}b_{\alpha}^{T-i}a_{\alpha})^{2}$$
$$(36)$$
(37)  $$\begin{array}{l}\small\mathbf{(38)^{}}\end{array}$$ . 
$$(39)$$
$$(40)$$

To ease notation, we omit specifying α when referring to connectivity modes, although note that all terms (st*, a, b, c*) still refer to a particular singular value dimension α.

## C. Exact Solution Of Input-Output Connectivity Modes

We solve for the learning dynamics of the input-output connectivity modes when the recurrent connectivity mode is frozen. If we assume balanced weights such that a = c, we can solve for both modes u = ac together. This equation can be integrated to yield

$$\tau\frac{d}{dt_{\theta}}u=c(\tau\frac{d}{dt_{\theta}})+a(\tau\frac{d}{dt_{\theta}}c)$$ $$=c(\sum_{i=1}^{T}b^{T-i}c(s_{i}-cb^{T-i}a))+a(\sum_{i=1}^{T}ab^{T-i}(s-cb^{T-i}a))$$ $$=2u(\sum_{i=1}^{T}b^{T-i}(s_{i}-b^{T-i}u))$$
$$(41)$$
$$(42)$$
$$(43)$$
$$(45)$$
tθ = τ Z u(tθ) du PT i=1 2ubT −i(si − ubT −i) u(0) = τ 2 log(u) − log(PT i=1 b T −isi − ub2(T −i) PT i=1 b T −isi u(tθ) u(0) (45) =τ 2PT i=1 b T −isi log u(tθ)(PT i=1 b T −isi − u(0)b 2(T −i)) u(0)(PT i=1 b T −isi − u(tθ)b 2(T −i)) u(tθ) = e 2tθ(PT i=1 b T−isi)/τ (PT i=1 b T −isi) (PT i=1 b T −isi)/u(0) − (PT i=1 b 2(T −i)) + e 2tθ(PT i=1 bT−isi)/τ (PT i=1 b 2(T −i))
$$(444)$$
$$(46)$$
$$(47)$$

## D. Local Approximation Of Recurrent Connectivity Modes

Due to the exponential term, the learning dynamics of the recurrent mode b are difficult to solve for. Instead, we take an approach similar to Schuessler et al. (2020b), by performing a Taylor expansion on the learning dynamics of b through training time (with input-output modes held constant),

$$b(t_{\theta}/\tau)=\sum_{n=0}^{\infty}\frac{d^{n}b(0)}{dt_{\theta}^{n}}\frac{(t_{\theta}/\tau)^{n}}{n!}\tag{48}$$

First we solve for the nth partial derivative of the energy function E with respect to the recurrent mode b which has an explicit closed-form solution given by

$$\frac{d^{n}E}{d\omega^{n}}=\sum_{i=1}^{T-n}\left[\left(T-i\right)\left(\prod_{j=1}^{n-1}\left(T-i-j\right)\right)s_{i}\omega b^{T-i-n}\right]-\sum_{i=1}^{\left(2T-n\right)/2}\left[\left(T-i\right)\left(\prod_{j=1}^{n-1}\left(2T-2i-j\right)\right)c^{2}a^{2}b^{2T-2i-n}\right]\tag{49}$$
$$(S{\mathfrak{J}})$$
$$(54)$$

Recall that in the gradient flow regime, the recurrent mode b changes continuously according to db dtθ
=
dE
db . Thus, to compute higher-order derivatives of b,

$$\frac{d^{n}b}{dt_{\theta}^{n}}=\frac{d}{dt_{\theta}}(\frac{d^{n-1}b}{dt_{\theta}^{n-1}})$$ $$=\frac{d}{db}(\frac{d^{n-1}b}{dt_{\theta}^{n-1}})\frac{db}{dt_{\theta}}$$ $$=\frac{d}{db}(\frac{d^{n-1}b}{dt_{\theta}^{n-1}})\frac{dE}{db}.$$
(50)  $$\begin{array}{l}\small\mathbf{(51)}\end{array}$$ = (52)  . 
Note that this is a recursive operation and does not give a simple closed-form expression. Applying this to higher-orders and using chain rule, we compute the time-derivatives of b up to 5th order,

db dtθ = dE db (53) d 2b dt2θ = d 2E db2 dE db (54) d 3b dt3θ = ( d 2E db2 ) 2 + d 3E db3 dE db  dE db (55) d 4b dt4θ = 4 d 2E db2 d 3E db3 ( dE db  ) 1 + (d 2E db2 ) 3 + d 4E db4 ( dE db  ) 2dE db (56) d 5b dt5 = ( d 2E db2 ) 4 + 11(d 2E db2 ) 2d 3E db3 ( dE db  ) + 4(d 3E db3 ) 2( dE db  ) 2 + 7 d 4E db4 d 2E db2 ( dE db  ) 2 + d 5E db5 ( dE db  ) 3 dE
db (57)
We then approximate the learning dynamics of the recurrent mode b and substitute the formula above for the nth partial derivative of the energy function

b(tθ/τ ) ≈ b(0) +  dE db  (tθ/τ ) +  d 2E db2 dE db (tθ/τ ) 2 2! + ( d 2E db2 ) 2 + d 3E db3 dE db  dE db (tθ/τ ) 3 3! (58) + 4 d 2E db2 d 3E db3 ( dE db  ) 1 + (d 2E db2 ) 3 + d 4E db4 ( dE db  ) 2 dE db (tθ/τ ) 4 4! (59) + ( d 2E db2 ) 4 + 11(d 2E db2 ) 2d 3E db3 ( dE db  ) + 4(d 3E db3 ) 2( dE db  ) 2 + 7 d 4E db4 d 2E db2 ( dE db  ) 2 + d 5E db5 ( dE db  ) 3dE db (tθ/τ ) 5 + O(6) (61)
5! (60)
In practice, when simulating the learning dynamics using this approximation, we apply the solution locally across a window of size ∆ and iterate over each window b(tθ : tθ + ∆). The window-size is dependent on the smoothness of the connectivity mode dynamics (i.e., how sharp the gradient is).

## D.1. Analytical Approximation Using Faa Di Bruno Formula And Bell Polynomials `

Here we use the Faa di Bruno formula/Bell polynomials to write out a combinatorial solution for the ` nth derivative of b, which can be used to expand the learning dynamics of b to higher orders without repeated recursive chain rule. Using this

 (58)  $\text{}$  (59)  $\text{}$  $\therefore\text{}$ (60)  (61)  $\text{}$
approach, the nth derivative of b is

$$\begin{array}{l}{{\frac{d^{n}b}{d t_{\theta}^{n}}=\sum_{k=1}^{n-1}\frac{d^{k+1}E}{d b^{k+1}}B_{n-1,k}[\frac{d b}{d t_{\theta}},\frac{d^{2}b}{d t_{\theta}^{2}},\ldots,\frac{d^{n-k}b}{d t_{\theta}^{n-k}}]}}\\ {{=\sum_{k=1}^{n-1}\frac{d^{k+1}E}{d b^{k+1}}\sum_{\{m_{1},m_{2},\ldots,m_{n-k}\}}\frac{(n-1)!}{m_{1}!m_{2}!\ldots m_{n-k}!}\prod_{j=1}^{n-k}\frac{1}{j!m_{j}}(\frac{d^{j}b}{d t_{\theta}^{j}})^{m_{j}}}}\end{array}$$
(62)  $\binom{63}{2}$  . 
$$\begin{array}{l}{(64)}\\ {(65)}\end{array}$$

where the summation over {m1, m2*, . . . , m*n−k} indicates a summation over all n − k partitions of nonnegative integers satisfying

$m_{1}+m_{2}+\cdots+m_{n-k}=k$  $1m_{1}+2m_{2}+\cdots+(n-k)m_{n-k}=n_{1}$
Although useful, we note that this approach still requires substitution of other lower-order terms of b (because of the d jb dtjθ term). The equation can be substituted back into the Taylor expansion of b through training time,

$$b(t_{\theta}/\tau)=\sum_{n=0}^{\infty}\frac{d^{n}b(0)}{d\theta_{0}^{n}}\frac{(t_{\theta}/\tau)^{n}}{n!}\tag{66}$$ $$=b(0)+\frac{dE}{db}(t_{\theta}/\tau)+\sum_{n=2}^{\infty}\left(\sum_{k=1}^{n-1}\frac{d^{k+1}E}{db^{k+1}}\sum_{(m_{1},m_{2},...,m_{n-k})}\frac{(n-1)!}{m_{1}!m_{2}!\ldots m_{n-k}!}\prod_{j=1}^{n-k}\frac{1}{j!m_{j}!}\frac{(d^{j}b}{d\theta_{j}^{j}})^{m_{j}}\right)\frac{(t_{\theta}/\tau)^{n}}{n!}\tag{67}$$

## E. Effect Of Task Dynamics On The Ordering Of Learning

To illustrate the effect of the ordering of singular values on learning speed, we compare task dynamics where the network connectivity modes learn solutions of the same magnitude (for different modes such that input-output and recurrent modes
"swap" solutions), but the ordering of singular values is either ascending or descending. More specifically, we consider the case of inverse-exponential task dynamics given by st = δf(λ, t); f(*λ, t*) = λ T −t. The solution for inverse-exponential dynamics are ac = *δ, b* = λ. Thus, we switch the values for *δ, λ* (orange: δ = 1.1, λ = 0.5, blue: δ = 0.5, λ = 1.1) for two simulations so that the network connectivity modes learn solutions of the same magnitude (so that we somewhat control for the effect that larger singular values has on accelerating learning speed), but the task dynamics either have ascending or descending singular values over the trajectory length, and, more importantly, the magnitude of the SVs at the end of the trajectory differ. We indeed see that modes trained on task dynamics with larger singular values occurring later in the trajectory learn faster. As we discuss in the main text, this is due to the fact that the recurrent connectivity mode b scales the gradient contribution for early trajectory timesteps exponentially. Since b is initialized to be less than 1, this has the effect of downscaling the gradient contribution of earlier timesteps compared to later ones. This manifests in singular values occurring later in the trajectory to "contribute more" to learning (when b < 1), such that modes trained on task dynamics with larger and later singular values learn faster (because they have larger gradient updates).

st < st + 1 st > st + 1 1 7 trajectory timesteps (t)
0.0 0.2 0.4 0.6 0.8 1.0 0 100 200 300 400 500 training timesteps t 0.0 0.2 0.4 0.6 0.8 1.0 co nn ec tiv it y m o des ta sk d y n a mi cs 
(st
)

recurrent (b)
input-output (ac)
This relates to the well-studied problems of learning of long-range dependencies and vanishing gradients in RNNs, as data from earlier timesteps are harder to learn as trajectory length increases because of the exponential downscaling effect of the recurrent mode. Here, we build on these ideas to understand learning speed and its dependency on task dynamics, including both the ordering of singular values and their scale. This also relates to our study on the tradeoff between feedforward versus recurrent computation in LRNNs in Section 3.4. Because later singular values have a greater effect on the gradient of the loss, the feedforward computation is effectively favored in learning, although the cumulative effect of the recurrent computation can of course outweigh this.

## F. Zero-Loss Solutions Only Exist For Inverse-Exponential Task Dynamics F.1. Discussion On Exponential Task Dynamics As Reparameterization Of Inverse-Exponential Task Dynamics

In the main text, we refer to inverse-exponential and exponential task dynamics separately. Here, we make the distinction that although we refer to these separately, they can both be rewritten as reparameterizations of each other (i.e., exponential task dynamics can be rewritten in the inverse-exponential form and vice versa) specifically when the trajectory length is held constant. We refer to inverse-exponential and exponential dynamics separately in the main text primarily to distinguish between cases that extrapolate (or don't) and illustrate how RNNs can learn perfect solutions that do not match the ground-truth data generating process.

In particular, for fixed T, exponential task dynamics given by st = δf(*λ, t*) where f(*λ, t*) = λ tcan equivalently be written as st = δλTg(*λ, t*) for g(*λ, t*) = ( 1λ
)
T −t.

This illustrates how and why LRNNs can still learn a perfect solution to exponential task dynamics for a fixed trajectory length (by overfitting), but because their architecture does not match the latent structure of the ground-truth data generating process, the network will not extrapolate to other trajectory lengths. More generally, we can see how mismatches between latent task dynamics and the network's recurrent dynamics can lead to non-extrapolating solutions. Of course, this does not occur when data is generated by a teacher network with a matched architecture because the latent form of the task dynamics and recurrent dynamics are the same.

## F.2. Proof

Here we prove that zero-loss solutions only exist for inverse-exponential task dynamics. Recall the energy function is given by

$$E={\frac{1}{2\tau}}\sum_{i=1}^{T}(s_{i}-c b^{T-i}a)^{2}$$
2(68)
$$(68)$$