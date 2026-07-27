# Constrained Optimization From A Control Perspective Via Feedback Linearization

## Anonymous Authors1 Abstract

Constrained optimization is fundamental to numerous applications. While first-order iterative algorithms are widely used for solving these problems, understanding their continuous-time counterparts—formulated as differential equations—can provide valuable theoretical insights into stability and convergence. Among various approaches, Feedback Linearization (FL), a well-established nonlinear control technique, has demonstrated potential for addressing nonconvex equality-constrained optimization problems, yet remains relatively underexplored. This paper aims to develop rigorous theoretical foundations for applying feedback linearization to solve constrained optimization. For equalityconstrained optimization, we establish global convergence rates to first-order Karush-Kuhn- Tucker (KKT) points and uncover the close connection between the FL method and the Sequential Quadratic Programming (SQP) algorithm. Building on this relationship, we extend the FL approach to handle inequality-constrained problems. Furthermore, we introduce a momentumaccelerated FL algorithm that achieves faster convergence, and provide a rigorous convergence guarantee.

000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054

## 1. Introduction

Constrained optimization, also known as nonlinear programming, has found vast applications in several domains including robotics (Alonso-Mora et al., 2017), supply chains (Garcia and You, 2015), and safe operations of power systems (Dommel and Tinney, 1968). First-order iterative algorithms are widely used to solve such problems, particularly in optimization and machine learning settings with large-scale datasets. These algorithms can be interpreted as discrete-time (DT) dynamical systems, while their continuous-time (CT) counterparts, derived by con-
Preliminary work. Under review by the International Conference on Machine Learning (ICML). Do not distribute.

1 sidering infinitesimal step sizes, take the form of differential equations. Analyzing these continuous-time systems can provide valuable theoretical insights, such as stability properties and convergence rates. This perspective is welldeveloped for unconstrained optimization, exemplified by the gradient flow x˙ = −∇f(x) (Elkabetz and Cohen, 2021; Arora et al., 2019; Saxe et al., 2013; Garg and Panagou, 2021; Andrei), the continuous-time counterpart of gradient descent, as well as its accelerated variants (Su et al., 2016; Wilson et al., 2018; Muehlebach and Jordan, 2019). However, for constrained optimization, this approach remains less thoroughly explored. Recent studies (Cerone et al., 2024; Gunjal et al., 2024) have explored the dynamical properties of CT constrained optimization algorithms. These works leverage a feedback control perspective to design and analyze the performance of optimization methods. Specifically, they propose frameworks that model constrained optimization problems as control problems, where the iterations of the optimization algorithm are represented by a dynamical system, and the Lagrange multipliers act as control inputs. The objective in this framework is to drive the system to a feasible steady state that satisfies the constraints. Within this framework, various control strategies can be employed to design the update of Lagrange multipliers, resulting in different control-based first-order methods. In this work, we adopt the same control perspective as above, and specifically focus on using Feedback Linearization (FL) approach, a standard approach in nonlinear control (cf. (Isidori, 1985; Henson and Seborg, 1997)), to design the Lagrange multiplier. One key advantage of this method is its natural suitability for handling nonconvex constrained optimization problems. Although this approach has been explored in the literature (Cerone et al., 2024; Schropp and Singer, 2000), its theoretical properties are not yet fully understood. Several important questions remain open. The first question concerns global convergence and convergence rates. While existing works established local stability (Cerone et al., 2024), global convergence and convergence rate have not been rigorously analyzed. The second question concerns the relationship between the FL approach and existing optimization algorithms, specifically whether the discretization of FL dynamics aligns with any known optimization method. Additionally, since most existing studies (Cerone et al., 2024; Schropp and Singer, 2000) focus exclusively on equality constraints, this raises the third question: how can the FL approach be extended to effectively handle inequality constraints? Lastly, it remains an open question whether insights from acceleration techniques in optimization (e.g., momentum acceleration) can be leveraged to develop faster FL-based algorithms. Our contributions. Motivated by the open questions discussed above, we aim to deepen the theoretical understanding of the feedback linearization (FL) approach for constrained optimization by addressing these questions. Specifically, our contributions are as follows:
1. We establish a global convergence rate to a first-order Karush-Kuhn-Tucker (KKT) point for the feedback linearization method for equality-constrained optimization (Section 3.1).

2. We demonstrate that the FL method is closely related to the Sequential Quadratic Programming (SQP) algorithm, providing a new perspective on its connection to established optimization techniques (Section 3.2).

3. Building on this insight, we extend the method to handle inequality constraints, broadening its applicability (Section 4).

4. Finally, leveraging these findings, we propose a momentum-accelerated FL algorithm, which empirically achieves accelerated convergence. Additionally, we provide a rigorous convergence guarantee for the continuous-time momentum-accelerated FL method (Section 5). To the best of our knowledge, both the proposed algorithm and its analysis are novel contributions to the field.

Due to space limits, a comprehensive review of related literature is deferred to Appendix A.

055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109 Notations: We use the notation [n], n ∈ N to denote the set {1, 2, 3*, . . . , n*}. We use ∇f(x) to denote the gradient of a scalar function f : R
n → R evaluated at the point x ∈ R
n and use ∇2f(x) to denote its corresponding Hessian matrix. We use Jh(x) to denote the Jacobian matrix of a function h : R
n → R
m evaluated at x ∈ R
n, i.e.

[Jh(x)]i,j =
∂hi(x)
∂xj, i ∈ [m], j ∈ [n]. Unless specified otherwise, we use *∥ · ∥* to denote the L2 norm of matrices and vectors and use *∥ · ∥*∞ to denote the L∞ norm. For a positive definite matrix A, we use ∥X∥A := ∥A− 
1 2 X∥ to denote the A-norm of X. For a set A, we use Acto denote its complement.

## 2. Feedback Linearization For Solving Equality Constrained Optimization

In this section, we briefly review related works that adopt a control perspective, particularly focusing on the use of feedback linearization to address equality-constrained optimization problems. Control perspective on equality-constrained optimization (Cerone et al., 2024) Consider the constrained optimization problem with equality constraints

$$\operatorname*{min}_{x}f(x)\qquad s.t.\ h(x)=0,$$
$$(1)$$

$$\mathrm{(2)}$$
xf(x) s.t. h(x*) = 0*, (1)
where x ∈ R
n, f : R
n → R, h : R
n → R
m. The firstorder KKT conditions are given by

$$-\nabla f(x)-J_{h}(x)^{\top}\lambda=0,\quad h(x)=0$$
(3) $\frac{1}{2}$
The key idea is to view finding the KKT point as a control problem (Figure 1)

$$\dot{x}=-T(x)\left(\nabla f(x)+J_{h}(x)^{\top}\lambda\right),\quad y=h(x)$$

Figure 1: Control Perspective for Constrained Optimization where x represents the system state, y = h(x) is system constraint variable and λ is the control input. T(x) here is a positive definite matrix and throughout the paper we assume that there exists λmin, λmax such that for all x,

$$\lambda_{\operatorname*{min}}I\preceq H(x)\preceq\lambda_{\operatorname*{max}}I$$

Note that at an equilibrium point x
⋆ of the system in Fig. 1 it must satisfy:

$$\dot{x}=0\implies\nabla f(x^{\star})+J_{h}^{\top}(x^{\star})=0.$$

Further, if x
⋆is feasible, i.e. h(x
⋆) = 0, then we get that x
⋆
satisfies the first order KKT conditions (2). Thus, the key idea is to manipulate the evolution of x so that we stabilize the system to equilibrium and feasibility. (For a more detailed overview about optimization from a control perspective, see Appendix A.) To achieve the goal of reaching a feasible equilibrium, we next introduce the feedback linearization (FL) approach, which is the main focus of this paper.

110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 Feedback linearization for equality-constrained optimization (Cerone et al., 2024) Feedback linearization (FL) (Isidori, 1985; Henson and Seborg, 1997) is a classical control method for controlling nonlinear dynamics which generally takes the following form:

$${\dot{x}}=F(x)+G(x)\lambda$$

As directly designing a stabilizing controller for the nonlinear system can be a challenging task, the FL approach circumvents the difficulty by transforming a nonlinear control system into an equivalent linear control system, which is much easier to analyze, through a change of variables and a suitable control input. In particular, one seeks a change of coordinates y = Φ(x) and the control input λ = a(x) + b(x)u such that the system becomes a linear system: y˙ = Ay + Bu.

In the equality constrained optimization problem, we have that F(x) = −∇f(x), G(x) = −Jh(x). The difference is that we don't seek a bijective change of coordinates y = Φ(x), but only focus on the observations y = h(x) instead. If λ = a(x) + b(x)u and we write out the time derivative for y we have that

$$\dot{y}=J_{h}(x)\dot{x}=-J_{h}(x)T(x)\nabla f(x)-J_{h}(x)T(x)J_{h}(x)^{\top}\lambda$$ $$=-J_{h}(x)T(x)\nabla f(x)-J_{h}(x)T(x)J_{h}(x)^{\top}(a(x)+b(x)u),$$

Thus, in the scenario where Jh(x) has full row rank, by setting we have that y˙ = u, then we can simply set u = −Ky where K is a Hurwitz matrix to guarantee that y asymptotically converge to zero. Thus the feedback linearization (FL) dynamics is given as follows:
FL for Equality-Constrained Optimization (Cerone et al., 2024)

$$\dot{x}=-T(x)\left(\nabla f(x)+J_{h}(x)^{\top}\lambda\right)$$
⊤λ(4)
$$:)T(x)J_{h}(x)^{\top})^{-1}(J_{l})$$

λ=−Jh(x)T(x)Jh(x)
⊤−1(Jh(x)T(x)∇f(x)−Kh(x))
The FL approach is particularly advantageous for handling nonlinear dynamics, making it well-suited for nonconvex constrained optimization. This is supported by numerical results in (Cerone et al., 2024; Schropp and Singer, 2000), which demonstrate its strong performance in such settings. However, its theoretical properties remain less well understood. Existing analyses primarily focus on local stability (Cerone et al., 2024), while global convergence and convergence rates are largely unexplored. Furthermore, the relationship between the FL algorithm and existing optimization methods remains unclear.

The following Section 3 will focus on addressing the above open problems in the FL method that remain unsolved in the existing literature, including convergence rate (Section 3.1) and relationship to existing optimization algorithms (Section 3.2)

## 3. Fl Control Method: Convergence And Relationship To Sqp 3.1. Convergence Results

Section 2 introduces the FL method for equalityconstrained optimization. The analyses in existing works mainly focus on the local stability, and little is known in terms of the global convergence property. In this section, we establish a global convergence rate to a first order KKT point (Contribution 1).

The result relies on the following assumption:
Assumption 1. *We make the following assumptions on the* function f and h:

_1.1 There exists a constant $D$ such that $(J_{h}(x)J_{h}(x)^{\top})^{-1}\prec D^{2}I$ for all $x$;_
1.2 There exists a constant M such that ∥∇f(x)∥ < M,
∥Jh(x)∥ < M *for all* x; 1.3 The function f(x) *is lower-bounded, i.e.* f(x) ≥ fmin for all x.

Note that Assumption 1.1 is similar to the assumption made in (Cerone et al., 2024) which assumes that rank(Jh(x)) = m for all x, which is equivalent to Jh(x)Jh(x)
⊤ is invertible. This assumption is also known as the linear independence constraint qualification (LICQ, cf. (Peterson, 1973; Nocedal and Wright, 2006), see more discussion in Appendix A) in optimization literature. Assumption 1.2 implies that the functions f and g are Lipschitz. We would also like to acknowledge that this Assumption is quite restrictive and is solely for analysis purpose. In our numerical simulations we found that the algorithm is suitable for non-uniformly-Lipschitz functions. We define the KKT-gap of (*x, λ*) as follows:

$$\|h(x)\|_{\infty}\}$$
$$|*||D$$

KKT-gap(*x, λ*):=max{∥∇f(x)+Jh(x)
⊤λ∥, ∥h(x)∥∞} (5)
We now state our result in terms of the convergence rate

$$\begin{array}{l}{{a(x)=-\left(J_{h}(x)T(x)J_{h}(x)^{\top}\right)^{-1}J_{h}(x)T(x)\nabla f(x),}}\\ {{b(x)=-\left(J_{h}(x)T(x)J_{h}(x)^{\top}\right)^{-1},}}\end{array}$$

diag{ki}
m i=1, where ki > 0, we have that the dynamic of feedback linearization method (4) satisfies:
1. For the set Ei:= {x : hi(x) ≥ 0}*, if* x(0) ∈ Ei*, then* x(t) ∈ Eifor all t ≥ 0*. Similarly, if* x(0) ∈ Ec i*, then* x(t) ∈ Ec ifor all t ≥ 0*, further*

$$h_{i}(x(t))=e^{-k_{i}t}h_{i}(x(0)),$$

i.e., h(x(t)) → 0 with an exponential rate as t → +∞.

2. *Define* ℓ(x) := f(x)+ λmax λmin
(MD)
2 Pm i=1 |hi(x)|*, then* ℓ(x(t)) *is non-increasing with respect to* t.

3. _Let_ $\overline{\lambda}(t):=-\left(J_{h}(x)T(x)J_{h}(x)^{\top}\right)^{-1}J_{h}(x)T(x)\nabla f(x(t))$_, then we have that_
t=0
$$\int_{0}^{T}\left|\nabla f(x(t))+J_{h}(x(t))^{\top}\overline{{{\lambda}}}(t)\right|^{2}d t\leq{\frac{1}{\lambda_{\mathrm{min}}}}(\ell(x(0))-\ell(x(T)))$$
_and that $\lim_{t\to+\infty}\left(\lambda(t)-\lambda(t)\right)=0$._
4. *(Asymptotic convergence and convergence rate) The* above statements imply that, inf 0≤t≤T
KKT-gap(x(t), λ(t)) ≤
further, we have that

$$\operatorname*{lim}_{t\rightarrow+\infty}\mathrm{KKT-gap}(x(t),\overline{{{\lambda}}}(t))=0,$$ $$\operatorname*{lim}_{t\rightarrow+\infty}\mathrm{KKT-gap}(x(t),\lambda(t))=0.$$

165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 Statement 4 in Theorem 1 implies that the algorithm can find an ϵ-first-order-KKT-point within time 1 ϵ 2 . We note that ensuring last-iterate convergence in nonconvex optimization is generally challenging. Hence, our analysis focuses on the best iterate, a widely adopted criterion in nonconvex optimization. Due to space limitations, we defer the detailed proof in Appendix B. The key step of the proof involves constructing the Lyapunov function ℓ(x) in Statement 2. We would also like to note that ℓ(x) also serves as the exact penalty function in constrained optimization literature (cf. (Eremin, 1967; Zangwill, 1967)).

## 3.2. Relationship With Sqp

The FL dynamics (4) provides a concise and elegant formulation, prompting the question of whether certain optimization algorithms can be derived through its discretization. In this section, we establish a fundamental connection between the continuous-time FL dynamics and the Sequential Quadratic Programming (SQP) algorithm (Contribution 2). Specifically, we demonstrate that the forward-Euler discretization (cf. (Atkinson, 1991; Ascher and Petzold, 1998)) of (4) is equivalent to the SQP algorithm. The state space continuous time dynamic for (4) is

$${\dot{x}}\!=\!-T(x){\Big(}\nabla f(x)\!-\!J_{h}(x)^{\top}\left(J_{h}(x)T(x)J_{h}(x)^{\top}\right)^{-1}{\Big)}$$
$$\cdot\left(J_{h}(x)T(x)\nabla f(x)-K h(x)\right).$$
$$x_{t})-J_{h}(x_{t})^{\top}$$
$$(6)$$

Its forward-Euler discretization scheme is xt+1 = xt − ηT(xt)
∇f(xt) − Jh(xt)
⊤ (6)

$$\cdot\big(J_{h}(x_{t})T(x_{t})J_{h}(x_{t})^{\top}\big)^{-1}\big(J_{h}(x_{t})T(x_{t})\nabla f(x_{t})-K h(x_{t})\big)\big).$$

We now consider the following SQP method, which is widely discussed in literature (cf. (Nocedal and Wright, 2006; Bonnans et al., 2006; Oztoprak et al., 2021)):

$$x_{t+1}=\arg\min_{x}\nabla f(x_{t})^{\top}(x-x_{t})+\frac{1}{2\eta}(x-x_{t})^{\top}T(x_{t})^{-1}(x-x_{t})$$ $$s.t.h(x_{t})+J_{h}(x_{t})(x-x_{t})=0\tag{7}$$

We are now ready to state the main result of this section, which demonstrates the equivalence of (6) and (7) Theorem 2. *When* K =
1 η I, the discretization of feedback linearization (6) is equivalent to the SQP algorithm (7). The proof of Theorem 2 leverages the fact that (7) satisfies the relaxed Slater condition, and thus the KKT conditions are necessary and sufficient for global optimality. Then Theorem 2 can be obtained by studying the KKT conditions of 7. The detailed proof is deferred to Appendix C
Remark 1 (Choice of T(x)). Theorem 2 provides insights into the selection of T(x) for the FL approach. Different choices of T(x) will correspond to different types of SQP algorithms. Here we mainly discuss two specific types of T(x). Firstly, when T(x) is chosen as the inverse of the Hessian matrix, i.e., T(x) = ∇2f(x)−1, then (7) *corresponds to the Newton-type algorithm where* the quadratic term in the objective function is given by
(x − xt)
⊤∇2f(x)(x − xt), which is widely considered in literature (cf. (Nocedal and Wright, 2006; Bonnans et al., 2006)). For this specific type of T(x), we name its corresponding FL dynamics (4) *as the* **FL-Newton** method. However, in the setting where the Hessian information is not available, another choice of T(x) is simply setting it as the identity matrix T(x) = I, which is considered in recent works such as (Oztoprak et al., 2021). In this case, the objective function resembles a proximal operator (cf. (Boyd, 2004; Parikh et al., 2014)), hence we name this as FL-proximal *method. Due to space limit, we defer a more* comprehensive overview of SQP in to Appendix A.

$$\max\Biggl{\{}\sqrt{\frac{2}{T}\Biggl{(}\frac{f(x(0))-f_{\min}}{\lambda_{\min}}+\frac{\lambda_{\max}M^{2}D^{2}}{\lambda_{\min}{}^{2}}\sum_{i}|h_{i}(x(0))|\Biggr{)},$$ $$\max_{1\leq i\leq m}\left\{h_{i}(x(0))e^{-\frac{k_{i}T}{2}}\right\}\Biggr{\}}\sim O\left(\frac{1}{\sqrt{T}}\right)$$
$$),{\overline{{\lambda}}}(t))\leq$$
,
$\downarrow$ . 
Z T

## 4. Extension To Inequality Constraints

$$\operatorname*{min}_{x}f(x)\qquad s.t.\ h(x)\leq0,$$

The KKT conditions for the above problem are given by

$$-\nabla f(x)-J_{h}(x)^{\top}\lambda=0,\quad h(x)\leq0\tag{9}$$ $$\lambda\geq0,\quad\lambda^{\top}h(x)=0$$
Thus, we can still view the problem as a control problem
whose corresponding dynamics can be written as:
$$\dot{x}=-T(x)\left(\nabla f(x)+J_{h}(x)^{\top}\lambda\right)\tag{10}$$ $$y=h(x),\quad\lambda\geq0.$$

However, the problem becomes more complicated because we require the non-negativity constraints λ ≥ 0 and complementary slackness λ
⊤h(x) = 0. It is at first glance unclear how to guarantee theses conditions through the control process. However, inspired by the relationship with SQP algorithms, we carefully design a more intricate FL controller as follows:
FL for Inequality-Constrained Optimization

$$\dot{x}=-T(x)\left(\nabla f(x)+J_{h}(x)^{\top}\lambda\right)$$ $$\lambda=\operatorname*{arg\,min}_{\lambda\geq0}\left(\frac{1}{2}\lambda^{\top}J_{h}(x)T(x)J_{h}(x)^{\top}\lambda\right.\tag{11}$$ $$\left.+\lambda^{\top}\left(J_{h}(x)T(x)\nabla f(x)-Kh(x)\right)\right)$$

Here we assume that the optimization problem 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274

$$\lambda=\operatorname*{arg\,min}_{\lambda\geq0}\biggl(\frac{1}{2}\lambda^{\top}J_{h}(x)T(x)J_{h}(x)^{\top}\lambda$$ $$+\lambda^{\top}\left(J_{h}(x)T(x)\nabla f(x)-Kh(x)\right)\biggr).$$

admits a unique solution. We would also like to point out that λ in (11) takes the form of the solution of an optimization problem, resulting in a non-smooth trajectory. A similar formulation of non-smooth ordinary differential equations (ODEs) has been explored in the context of differential variational inequalities (cf. (Dupuis and Nagurney, 1993; Pang and Stewart, 2008; Camlibel et al., 2007)). At first glance, it may not be immediately clear why the algorithm is structured as in (11). The derivation of (11) was inspired by the connection between the FL method and SQP in the equality-constrained setting. Hence, for the inequality-constrained case, we first analyzed SQP
and then reverse-engineered its principles to derive its continuous-time counterpart, leading to the formulation of the FL method in (11). To ensure a coherent and intuitive presentation, we begin by establishing its relationship with the SQP algorithm.

$$({\boldsymbol{\delta}})$$

Relationship with the SQP algorithm The corresponding forward Euler discretization of (11) is given by

$$\begin{array}{l}{{x_{t+1}=x_{t}-\eta T(x)\left(\nabla f(x_{t})+J_{h}(x_{t})^{\top}\lambda_{t}\right)}}\\ {{\lambda_{t}{\mathrm{=arg\,min}}_{\lambda\geq0}\left({\frac{1}{2}}\lambda^{\top}J_{h}(x_{t})T(x_{t})J_{h}(x_{t})^{\top}\lambda_{t}\right)}}\\ {{\mathrm{=}}}\end{array}$$

$$\begin{array}{r}{{}}&{{h\geq0\left(2\right.}}\\ {{}}&{{}}\\ {{}}&{{+\left.\lambda^{\top}\left(J_{h}(x_{t})T(x_{t})\nabla f(x_{t})-K h(x_{t})\right)\right)}}\end{array}$$
$$(12)^{\frac{1}{2}}$$

We now consider the following SQP type of optimization method

$$x_{t+1}=\arg\min_{x}\nabla f(x_{t})^{\top}(x-x_{t})+\frac{1}{2\eta}(x-x_{t})^{\top}T(x_{t})^{-1}(x-x_{t})$$ $$s.t.\quad h(x_{t})+J_{h}(x_{t})(x-x_{t})\leq0\tag{13}$$

The following theorem states the equivalence between (12) and (13). Theorem 3. *When* K =
1 η I*, if* (13) is feasible, then the discretization of feedback linearization (12) is equivalent to the SQP algorithm (13).

Similar to the proof of Theorem 2, the proof of Theorem 3 also leverages strong duality and KKT conditions. THe detailed proof is deferred to Appendix C. Convergence Result Theorem 3 demonstrates the relationship between the FL algorithm (11) and (13). Since SQP algorithms are known to be capable of converging to a KKT point (Nocedal and Wright, 2006), intuitively similar convergence can be established for our FL algorithm (11), which is the main focus of the following part.

We define the index set I(x) := {i : hi(x) > 0}. We also use I(x)
cto denote the complimentary set of I(x). Our results rely on the following assumptions:
Assumption 2. *We make the following assumptions on the* function f and h 2.1 *Given the initial state* x(0) at t = 0, the optimization problem in (11)

$$\lambda=\arg\operatorname*{min}_{\lambda\geq0}\Bigl({\frac{1}{2}}\lambda^{\top}J_{h}(x)T(x)J_{h}(x)^{\top}\lambda\Bigr)$$
$$\left(\begin{array}{l}{{+\lambda^{\top}\left(J_{h}(x)T(x)\nabla f(x)-K h(x)\right)}}\end{array}\right)$$

admits bounded a solution ∥λ∥∞ ≤ L *for all* x ∈ E, where E is defined by E := {x|0 < hi(x) ≤ hi(x(0)), ∀i ∈ I(x(0))}.

2.2 There exists a constant M such that ∥∇f(x)∥ < M,
∥Jh(x)∥ < M *for all* x.

The above sections primarily focus on the constrained optimization setting with equality constraints (1). This section aims to address the question of whether we can extend to setting with inequality constraints (Contribution 3), i.e.,
2.3 The function f(x) *is lower-bounded, i.e.* f(x) ≥ fmin for all x.

Although Assumption 2.1 is quite complicated and relatively hard to verify, there are some simplified versions that serve as a sufficient condition of Assumption 2.1. For example, if we start with a feasible x(0), then E = ∅ and hence Assumption 2.1 is automatically satisfied. Additionally, note that Assumption 1.1 is another sufficient condition of Assumption 2.1 (see Lemma 3 in Appendix F) We define the KKT-gap of the state variable x and nonnegative control variable λ ≥ 0 as follows:
KKT-gap(x,λ):=maxn∥∇f(x)+Jh(x)
⊤λ∥,
λ
⊤h(x)
,max i[hi(x)]+
o, where [hi(x)]+ = max{hi(x), 0}.

Theorem 4. *Under Assumption 2, for a diagonal matrix* K = diag{ki}i=1, where ki > 0*, the learning dynamics* (11) satisfies the following properties 1. dhi(x(t))
dt ≤ −kihi(x(t))*, for* i = 1, 2, . . . , m, and hence the dynamic will asymptotically converge to the feasible set.

2. *Define* ℓ(x) := f(x(t)) + LPi
[hi(x)]+*, then* ℓ(x(t)) is non-increasing w.r.t t*. Here* [hi(x)]+ = max{hi(x), 0}.

3. *The following inequality holds*

$$\sim O\left({\frac{1}{\sqrt{T}}}\right)$$
$\left(\begin{array}{c}\includegraphics[height=36.135pt]{./figures/.eps}\end{array}\right)$  $\left(\begin{array}{c}\includegraphics[height=36.135pt]{./figures/.eps}\end{array}\right)$
275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329

$$\begin{split}\int_{0}^{T}&\left(\|\nabla f(x(t))+J_{h}(x(t))\lambda(t)\|_{T(x(t))}^{2}-\sum_{i\in\mathcal{I}(x)^{c}}k_{i}\lambda_{i}(t)h_{i}(x(t))\right)dt\\ &\leq\ell(x(0))-\ell(x(T))\end{split}$$
Z T
4. (Asymptotic convergence and convergence rate) The
above statements imply that
inf
$\mathrm{p}{\leq}\mathrm{t}{\leq}2$
KKT-gap(*x(t), λ(t*))
$$\leq\max\left\{\sqrt{\frac{2}{\lambda_{\min}T}\left(f(x(0))-f_{\min}+L\sum_{i\in\mathcal{I}(x(0))}h_{i}(x(0))\right)}\right)$$

,

$\downarrow$ ]. 

Further we have that KKT-gap asymptotically converges to zero, i.e.

$$\operatorname*{lim}_{t\rightarrow+\infty}\mathrm{KKT-gap}(x(t),\lambda(t))=0$$

Statement 4 in Theorem 4 implies that the algorithm can find an ϵ-firs-order KKT-point within time 1 ϵ 2 . Similar to Theorem 1, the key step of the proof is to construct the Lyapunov function in Statement 2 (detailed proof deferred to Appendix D).

## 5. Momentum Acceleration For Constrained Optimization

In Remark 1, we introduced the FL-proximal and FL- Newton algorithms. Generally, FL-Newton achieves faster convergence than FL-proximal due to its use of secondorder information. However, in scenarios where Hessian information is unavailable, FL-proximal must be used instead, raising the question of whether its convergence can be accelerated. Given that momentum acceleration has been shown to improve convergence rates in unconstrained optimization, a natural question arises: can a momentum-accelerated version of the FL-proximal algorithm, along with its corresponding discrete-time SQP formulation, achieve faster convergence? This section aims to address this question as part of Contribution 4. Momentum acceleration is a technique commonly used in optimization to enhance convergence rates (cf. (Polyak, 1964; Nesterov, 1983; d'Aspremont et al., 2021), see Appendix A for more detailed introduction about momentum acceleration). For unconstrained optimization, the discretetime momentum acceleration for gradient descent generally takes the form of

$$\begin{array}{c}{{w_{t}=x_{t}+\beta(x_{t}-x_{t-1})}}\\ {{x_{t+1}=w_{t}-\eta\nabla f(w_{t})}}\end{array}$$
$$(14)^{\frac{1}{2}}$$
x˙ = z
z˙ = −αz −∇f(x) + Jh(x)
⊤λ
Its corresponding continuous-time analogue can be written as a second-order ODE (Polyak, 1964; Su et al., 2016):

$${\dot{x}}=z$$ $${\dot{z}}=-\alpha z-\nabla f(x)$$
$$(16)$$
$$(15)$$

Inspired by the form of (14) and (15), for equality constrained optimization, we propose the following heuristic momentum-accelerated discrete time SQP scheme

$$w_{t}=x_{t}+\beta(x_{t}-x_{t-1})$$
$w_{t}=x_{t}+\beta(x_{t}-x_{t-1})$  $\lambda_{t}=-\big{(}J_{h}(w_{t})J_{h}(w_{t})^{\top}\big{)}^{-1}(J_{h}(w_{t})\nabla f(w_{t})-\frac{1}{\eta}h(w_{t}))$
$$x_{t+1}=w_{t}+\eta\nabla f(w_{t})+J_{h}(w_{t})^{\top}\lambda_{t}$$
⊤λt (16)
and continuous time FL scheme, which we name as FL- momentum:
FL-momentum for Equality-Constrained Optimization λ = −(Jh(x)Jh(x)
⊤)
−1(Jh(x)∇f(x)−Kh(x))
(17)
Note that the only difference in (16) is that we add a momentum step wt = xt+β(xt−xt−1). Similarly we can propose the FL-momentum scheme for inequality constraint case as follows:
FL-momentum for Inquality-Constrained Optimization 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384

x˙ = z
$$\dot{z}=-\alpha z-\left(\nabla f(x)+J_{h}(x)^{\top}\lambda\right)\tag{18}$$ $$\lambda=\operatorname*{arg\,min}_{\lambda\geq0}\frac{1}{2}\lambda^{\top}J_{h}(x)J_{h}(x)^{\top}\lambda+\lambda^{\top}(J_{h}(x)\nabla f(x)-Kh(x)$$
The numerical simulation in Section 6 (Figure 2 and 3) demonstrates the comparison between the standard and momentum accelerated methods, which suggests that momentum methods indeed accelerate the convergence rate. We would also like to note that as far as we know, the acceleration of SQP methods are generally achieved via Newton or quasi-Newton methods, there's little work on exploring acceleration via momentum approaches, which makes our proposed momentum algorithm a novel contribution.

## 5.1. Analysis

In this section, we provide some convergence guarantees for the proposed algorithm. In particular, we primarily focus on the convergence analysis for the continuous-time algorithm for equality constrained optimization (17). It remains future work to establish the convergence for the discrete-time algorithm (16) or the inequality-constrained algorithm (18).

We first define the following notation

$$\overline{\lambda}(x):=-(J_{h}(x)J_{h}(x)^{\top})^{-1}(J_{h}(x)\nabla f(x))\tag{19}$$

Apart from Assumption 1, we also make the following assumptions on f and h. Assumption 3. Both f(x), h(x) are three-time differentiable and the derivatives are bounded, thus, we know that there exists some constant Lf , L1, L2 *such that*

$$\left\|\nabla^{2}f(x)\right\|\leq L_{f},\ \ \left\|\frac{\partial\bar{\lambda}(x)}{\partial x}\right\|\leq L_{2},$$ $$\left\|\frac{\partial\left(J_{h}(x)^{\top}\lambda(x)+\left(\frac{\partial\bar{\lambda}(x)}{\partial x}^{\top}h(x)\right)\right)}{\partial x}\right\|\leq L_{2}\,$$

Assumption 4. We also assume that that there exists a constant H¯ *such that*

$${\mathit{w h e r e\bar{H}}}(x):=[h(x)^{\top}\nabla^{2}h_{i}(x)]_{i=1}^{n}.$$

We are now ready to state our main result Theorem 5. Assume that Assumption 1, 3 and 4 hold. Let two positive constant a1, a2 *be such that*

$$a_{2}\geq\left(4{\frac{\lambda_{\mathrm{max}}(K)}{\lambda_{\mathrm{min}}(K)}}L_{2}D+{\frac{L_{1}^{2}}{\lambda_{\mathrm{min}}(K)}}\right)\times a_{1}\geq0.$$

We define the following Lyapunov function:

ℓ(*x, z*) = a1αf(x)+ 
a2α
2
$$a_{1}\alpha f(x)+{\frac{a_{2}\alpha}{2}}\|h(x)\|^{2}+a_{1}\alpha\bar{\lambda}(x)^{\top}h(x)+\|z\|^{2}$$
$$+\left(a_{1}\nabla f(x)+a_{2}J_{h}(x)^{\top}h(x)+a_{1}J_{h}(x)^{\top}\bar{\lambda}(x)+a_{1}\frac{\partial\bar{\lambda}(x)^{\top}}{\partial x}h(x)\right)^{\top}$$
then for

$$\alpha\geq\left(a_{1}(L_{f}+L_{2})+a_{2}(M^{2}+\bar{H})+\frac{1}{a_{1}}+\frac{2(\lambda_{\mathrm{max}}(K)D^{2})}{a_{2}}\right)+1$$

we have that 1. ℓ(x(t), z(t)) *is non-increasing with respect to* t. 2. *the following inequality holds*

$$\int_{t=0}^{T_{\underline{a}2}\lambda_{\min}(K)}\frac{1}{8}\|h(x(t))\|^{2}+\frac{a_{1}}{4}\|\nabla f(x(t))+J_{h}(x(t))^{\top}\overline{\lambda}(x(t))\|^{2}dt$$ $$\leq\ell(x(0),z(0))-\min_{x,z}\ell(x,z)$$
3. *We can bound the KKT-gap by*
$$\operatorname*{inf}_{0\leq t\leq T}\operatorname{KKT-gap}(x(t),{\overline{{\lambda}}}(x(t)))$$
$$\leq{\sqrt{\frac{\ell(x(0),z(0))-\ell_{\operatorname*{min}}}{\operatorname*{min}\left\{{\frac{a_{2}\lambda_{\operatorname*{min}}(K)}{8}},{\frac{a_{1}}{4}}\right\}T}}}\sim O({\frac{1}{\sqrt{T}}})$$
and
$\lim_{t\to+\infty}$ KKT$-$gap($x(t),\lambda(x(t)))=0$, $\lim_{t\to+\infty}$ KKT$-$gap($x(t),\lambda(t))=0$.  
The detailed proof is provided in Appendix E. One limitation of Theorem 5 is that it establishes convergence but not acceleration over FL-proximal. However, when the constraint function h(x) is affine, the algorithm is equivalent to the momentum-accelerated projected gradient method (see Appendix E.1), offering insight into its potential for accelerating optimization.

## 6. Numerical Simulation 6.1. Heterogeneous Logistic Regression

In this section, we consider a logistic regression problem involving heterogeneous clients (Shen et al., 2022; Hounie et al., 2024). Many scenarios, such as federated learning and fair machine learning, require training a common model in a distributed manner by utilizing data samples from diverse clients or distributions. In practice, heterogeneity in local data distributions often results in uneven model performance across clients (Li et al., 2020; Wang et al., 2020). Since this outcome may be undesirable, a reasonable objective in such settings is to add constraints to ensure that the model's loss is comparable across all clients. We formulate the above problem as a constrained optimization problem as follows: consider solving the logistic regression for C clients. For each client c ∈ {1, 2, . . . , C},
7

$$\|{\bar{H}}(x)\|\leq{\bar{H}},\ \ \forall x,$$

it is associated with its own dataset Dc = {(xi, yi)}
|Dc| i=1 ,
where the label is yi *∈ {−*1, 1} and data feature is xi ∈ Rd.

For each client c, its own logistic regression loss Rc(θ) is defined as:

$$R_{c}(\theta):={\frac{1}{|D_{c}|}}\sum_{i\in D_{c}}\log(1+\exp(-y_{i}\cdot\theta^{\top}x_{i})),$$

where θ is the parameter of the regression model. We further define the averaged regression loss R¯(θ) as

$$\bar{R}(\theta):=\frac{1}{C}\sum_{c=1}^{C}f_{c}(\theta)$$

As suggested in (Shen et al., 2022; Hounie et al., 2024), heterogeneity challenges can be addressed by introducing a proximity constraint that links the performance of each individual client, Rc, to the average loss across all clients, R¯. This approach naturally formulates a constrained learning problem:

$$\min_{\theta}\bar{R}(\theta),\quad s.t.\ R_{c}(\theta)-\bar{R}(\theta)-\epsilon\leq0,\ \forall c\in\{1,2,\ldots,C\}\tag{20}$$

385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439

$$\operatorname*{min}_{x}f(x),\;\;s.t.\;h_{\mathrm{eq}}(x)=0,\;h_{\mathrm{ineq}}(x)\leq0,$$

Figure 2: Result for Heterogeneous Logistic Regression
We solve the constrained optimization problem (20) by running the FL-proximal, FL-Newton, and FL-momentum algorithm . Here we set the number of clients to C = 5 and |Dc| = 200, the data yiis randomly generated from a Bernoulli distribution and xiis generated from a Gaussian distribution whose mean differs among different agents. The results of the numerical simulation are presented in Figure 2. Notably, all algorithms successfully converge to a first order KKT point, with FL-Newton exhibiting the fastest convergence, followed by FL-momentum, which outperforms FL-proximal in terms of convergence speed.

## 6.2. Optimal Power Flow

The Alternating Current Optimal Power Flow (AC OPF) problem is a fundamental optimization task in power systems. Its goal is to determine the most efficient operating conditions while satisfying system constraints. This involves optimizing the generation and distribution of electrical power to minimize costs, losses, or other objectives while ensuring that physical laws (such as power flow equations) and operational limits are respected, thus it can be summarized as the following constrained optimization problem:
where the objective function f(x) represents the power generation cost and the equality constraints heq(x) generally represents the physical law of the power system, i.e.,
the power flow equations and hineq includes operational limits in terms of voltage, power generation, transmission capacities etc. The optimization variable x generally consists of voltage angles and magnitudes at each bus, and the real and reactive power injections at each generator (see (Low, 2014) for a detailed introduction on AC OPF). We solve the AC OPF problem (21) by running the FL-proximal algorithm, FL-Newton algorithm, and FL- momentum algorithm. Figure 3 presents the numerical results for solving AC OPF on the IEEE-39 and IEEE- 118 bus systems, respectively. In both cases, FL-Newton demonstrates the fastest convergence, which is expected given that it leverages second-order information (i.e., the Hessian). Comparing FL-proximal and FL-momentum, both of which rely solely on first-order information, Figure 3 indicates that FL-momentum accelerates the learning process and achieves faster convergence than FL-proximal for the IEEE-39 bus system. However, in the IEEE-118 bus system, FL-proximal and FL-momentum exhibit similar convergence speeds, with their learning curves nearly overlapping. We hypothesize that this problem is illconditioned, limiting the effectiveness of momentum in accelerating the algorithm.

## 7. Conclusion

In this paper, we study the theoretical foundations for solving constrained optimization problems from a control perspective via feedback linearization (FL). We established global convergence rates for equality-constrained optimization, highlighted the relationship between FL and Sequential Quadratic Programming (SQP), and extended FL methods to handle inequality constraints. Furthermore, we introduced a momentum-accelerated FL algorithm, which empirically demonstrated faster convergence and provided rigorous convergence guarantees for its continuous-time dynamics. Future directions include exploring the potential extension to zeroth-order optimization settings and relaxing assumptions in the theoretical analysis.

where ϵ > 0 is a small, fixed positive scalar.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning, in particular in the domain of control and optimization. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

## References

440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 Stephen Boyd. Convex optimization. *Cambridge UP*,
2004.

M. Kanat Camlibel, Jong-Shi Pang, and Jinglai Shen.

Lyapunov Stability of Complementarity and Extended Systems. *SIAM Journal on Optimization*, 17(4):1056– 1101, January 2007. ISSN 1052-6234. doi: 10.1137/ 050629185. URL https://epubs.siam.org/ doi/10.1137/050629185. Publisher: Society for Industrial and Applied Mathematics.

V. Cerone, S. M. Fosson, S. Pirrera, and D. Regruto. A new framework for constrained optimization via feedback control of Lagrange multipliers, March 2024. URL http://arxiv.org/abs/
2403.12738. arXiv:2403.12738 [cs, eess, math].

Frank E. Curtis, Hao Jiang, and Daniel P. Robinson.

An adaptive augmented Lagrangian method for largescale constrained optimization. Mathematical Programming, 152(1-2):201–245, August 2015. ISSN 00255610, 1436-4646. doi: 10.1007/s10107-014-0784-y.

URL http://link.springer.com/10.1007/ s10107-014-0784-y.

Frank E. Curtis, Tim Mitchell, and Michael L. Overton.

A BFGS-SQP method for nonsmooth, nonconvex, constrained optimization and its evaluation using relative minimization profiles. Optimization Methods and Software, 32(1):148–181, January 2017. ISSN 1055-6788, 1029-4937. doi: 10.1080/10556788.2016.1208749.

URL https://www.tandfonline.com/doi/ full/10.1080/10556788.2016.1208749.

Alexandre d'Aspremont, Damien Scieur, and Adrien Taylor. Acceleration Methods. *Foundations and Trends®* in Optimization, 5(1-2):1–245, 2021. ISSN 2167-3888, 2167-3918. doi: 10.1561/2400000036. URL http:// arxiv.org/abs/2101.09545. arXiv:2101.09545 [cs, math].

II Dikin. Iterative solution of problems of linear and quadratic programming. In *Doklady Akademii Nauk*, volume 174, pages 747–748. Russian Academy of Sciences, 1967.

Dongsheng Ding and Mihailo R Jovanovic. Global expo- ´
nential stability of primal-dual gradient flow dynamics based on the proximal augmented lagrangian. In 2019 American Control Conference (ACC), pages 3414–3419. IEEE, 2019.

Hermann W Dommel and William F Tinney. Optimal power flow solutions. IEEE Transactions on power apparatus and systems, (10):1866–1876, 1968.

Paul Dupuis and Anna Nagurney. Dynamical systems and variational inequalities. *Annals of Operations Research*,
44:7–42, 1993.

Javier Alonso-Mora, Stuart Baker, and Daniela Rus. Multirobot formation control and object transport in dynamic environments via constrained optimization. The International Journal of Robotics Research, 36(9):1000–1021, 2017.

Wangpeng An, Haoqian Wang, Qingyun Sun, Jun Xu, Qionghai Dai, and Lei Zhang. A PID Controller Approach for Stochastic Optimization of Deep Networks. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8522–8531, Salt Lake City, UT, June 2018. IEEE. ISBN 978-1-5386-64209. doi: 10.1109/CVPR.2018.00889. URL https:// ieeexplore.ieee.org/document/8578987/.

Neculai Andrei. Gradient Flow Algorithm for Unconstrained Optimization.

Sanjeev Arora, Noah Golowich, Nadav Cohen, and Wei Hu. A CONVERGENCE ANALYSIS OF GRADI- ENT DESCENT FOR DEEP LINEAR NEURAL NET- WORKS. 2019.

Uri M Ascher and Linda R Petzold. Computer methods for ordinary differential equations and differentialalgebraic equations. SIAM, 1998.

Kendall Atkinson. *An introduction to numerical analysis*.

John wiley & sons, 1991.

Amir Beck and Marc Teboulle. A fast iterative shrinkagethresholding algorithm for linear inverse problems. SIAM journal on imaging sciences, 2(1):183–202, 2009.

Ernesto G Birgin and Jose Mario Mart ´ ´ınez. Practical augmented Lagrangian methods for constrained optimization. SIAM, 2014.

Paul Boggs and Jon Tolle. Sequential Quadratic Programming. *Acta Numerica*, 4:1–51, January 1995. doi: 10.1017/S0962492900002518.

Joseph-Fred´ eric Bonnans, Jean Charles Gilbert, Claude ´
Lemarechal, and Claudia A Sagastiz ´ abal. ´ Numerical optimization: theoretical and practical aspects. Springer Science & Business Media, 2006.

495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 Florian Dorfler, Zhiyu He, Giuseppe Belgioioso, Saverio ¨
Bolognani, John Lygeros, and Michael Muehlebach. Toward a Systems Theory of Algorithms. IEEE Control Systems Letters, 8:1198–1210, 2024. ISSN 2475-1456. doi: 10.1109/LCSYS.2024.3406943. URL https://ieeexplore.ieee.org/document/ 10540567/?arnumber=10540567. Conference Name: IEEE Control Systems Letters.

Omer Elkabetz and Nadav Cohen. Continuous vs. Discrete Optimization of Deep Neural Networks, December 2021. URL http://arxiv.org/abs/2107.

06608. arXiv:2107.06608 [cs].

II Eremin. The penalty method in convex programming.

Cybernetics, 3(4):53–56, 1967.

Mahyar Fazlyab, Manfred Morari, and George J. Pappas. Safety Verification and Robustness Analysis of Neural Networks via Quadratic Constraints and Semidefinite Programming. IEEE Transactions on Automatic Control, 67(1):1–15, January 2022. ISSN 1558-2523. doi: 10.1109/TAC.2020.3046193. URL https://ieeexplore.ieee.org/document/
9301422/?arnumber=9301422. Conference Name: IEEE Transactions on Automatic Control.

Daniel J Garcia and Fengqi You. Supply chain design and optimization: Challenges and opportunities. Computers & Chemical Engineering, 81:153–170, 2015.

Kunal Garg and Dimitra Panagou. Fixed-Time Stable Gradient Flows: Applications to Continuous-Time Optimization. IEEE Transactions on Automatic Control, 66(5):2002–2015, May 2021. ISSN 0018-9286, 1558-2523, 2334-3303. doi: 10.1109/TAC.2020. 3001436. URL http://arxiv.org/abs/1808. 10474. arXiv:1808.10474 [math].

Philip E. Gill and Daniel P. Robinson. A Globally Convergent Stabilized SQP Method. SIAM Journal on Optimization, 23(4):1983–2010, January 2013. ISSN 1052-6234, 1095-7189. doi: 10.1137/
120882913. URL http://epubs.siam.org/ doi/10.1137/120882913.

Philip E. Gill, Vyacheslav Kungurtsev, and Daniel P.

Robinson. A stabilized SQP method: global convergence. *IMA Journal of Numerical Analysis*, 37(1):407–443, January 2017. ISSN 0272-4979, 1464-3642. doi: 10.1093/imanum/drw004. URL https://academic.oup.com/imajna/ article-lookup/doi/10.1093/imanum/
drw004.

Revati Gunjal, Syed Shadab Nayyer, Sushama Wagh, and Navdeep Singh. Unified Control Framework:
A Novel Perspective on Constrained Optimization, Optimization-based Control, and Parameter Estimation, July 2024. URL http://arxiv.org/abs/2407. 00780. arXiv:2407.00780 [math].

William W. Hager. Stabilized Sequential Quadratic Programming. In Jong-Shi Pang, editor, Computational Optimization, pages 253–273. Springer US,
Boston, MA, 1999. ISBN 978-1-4613-7367-4 9781-4615-5197-3. doi: 10.1007/978-1-4615-5197-3 13. URL http://link.springer.com/10.1007/ 978-1-4615-5197-3_13.

Shih-Ping Han. A globally convergent method for nonlinear programming. *Journal of optimization theory and* applications, 22(3):297–309, 1977.

Adrian Hauswirth, Saverio Bolognani, Gabriela Hug, and Florian Dorfler. Timescale Separation in Autonomous ¨ Optimization. *IEEE Transactions on Automatic* Control, 66(2):611–624, February 2021. ISSN 1558-2523. doi: 10.1109/TAC.2020.2989274. URL https://ieeexplore.ieee.org/document/
9075378/?arnumber=9075378. Conference Name: IEEE Transactions on Automatic Control.

Adrian Hauswirth, Zhiyu He, Saverio Bolognani, Gabriela Hug, and Florian Dorfler. Optimization Algo- ¨ rithms as Robust Feedback Controllers, January 2024. URL http://arxiv.org/abs/2103. 11329. arXiv:2103.11329.

Zhiyu He, Saverio Bolognani, Jianping He, Florian Dorfler, and Xinping Guan. Model-Free Non- ¨ linear Feedback Optimization. IEEE Transactions on Automatic Control, 69(7):4554–4569, July 2024. ISSN 1558-2523. doi: 10.1109/TAC.2023. 3341752. URL https://ieeexplore.ieee. org/document/10354356. Conference Name: IEEE Transactions on Automatic Control.

Michael A Henson and Dale E Seborg. Feedback linearizing control. In *Nonlinear process control*, volume 4, pages 149–231. Prentice-Hall Upper Saddle River, NJ,
USA, 1997.

Ignacio Hounie, Alejandro Ribeiro, and Luiz FO Chamon.

Resilient constrained learning. Advances in Neural Information Processing Systems, 36, 2024.

Bin Hu and Laurent Lessard. Control Interpretations for First-Order Optimization Methods, March 2017. URL http://arxiv.org/abs/1703. 01670. arXiv:1703.01670 [cs].

Bin Hu, Peter Seiler, and Anders Rantzer. A Unified Analysis of Stochastic Optimization Methods Using Jump 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 System Theory and Quadratic Constraints. In Proceedings of the 2017 Conference on Learning Theory, pages 1157–1189. PMLR, June 2017. URL https:// proceedings.mlr.press/v65/hu17b.html.

Alberto Isidori. Nonlinear control systems: an introduction. Springer, 1985.

T Kose. Solutions of saddle value problems by differential equations. *Econometrica, Journal of the Econometric* Society, pages 59–70, 1956.

Laurent Lessard, Benjamin Recht, and Andrew Packard.

Analysis and Design of Optimization Algorithms via Integral Quadratic Constraints. SIAM Journal on Optimization, 26(1):57–95, January 2016. ISSN 1052-6234, 1095-7189. doi: 10.1137/15M1009597. URL http: //arxiv.org/abs/1408.3595. arXiv:1408.3595
[math].

Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. Federated optimization in heterogeneous networks. *Proceedings of* Machine learning and systems, 2:429–450, 2020.

Steven H Low. Convex relaxation of optimal power flow—part i: Formulations and equivalence. IEEE
Transactions on Control of Network Systems, 1(1):15– 27, 2014.

Michael Muehlebach and Michael Jordan. A dynamical systems perspective on nesterov acceleration. In International Conference on Machine Learning, pages 4656– 4662. PMLR, 2019.

Y. Nesterov. A method for solving the convex programming problem with convergence rate o(1/k2), 1983. URL https://cir.nii.ac.jp/crid/ 1370862715914709505.

Jorge Nocedal and Stephen J. Wright. Numerical optimization. Springer series in operation research and financial engineering. Springer, New York, NY, second edition edition, 2006. ISBN 978-0-387-30303-1 978-1-49393711-0.

Figen Oztoprak, Richard Byrd, and Jorge Nocedal. Constrained Optimization in the Presence of Noise, October 2021. URL http://arxiv.org/abs/2110. 04355. arXiv:2110.04355 [math].

Jong-Shi Pang and David E. Stewart. Differential variational inequalities. *Mathematical Programming*, 113(2):
345–424, June 2008. ISSN 1436-4646. doi: 10.1007/
s10107-006-0052-x. URL https://doi.org/10.

1007/s10107-006-0052-x.

Neal Parikh, Stephen Boyd, et al. Proximal algorithms.

Foundations and trends® in Optimization, 1(3):127– 239, 2014.

David W Peterson. A review of constraint qualifications in finite-dimensional spaces. *Siam Review*, 15(3):639–654, 1973.

Boris Polyak and Pavel Shcherbakov. Lyapunov Functions: An Optimization Theory Perspective. IFAC-PapersOnLine, 50(1):7456–7461, July 2017. ISSN 24058963. doi: 10.1016/j.ifacol.2017.08. 1513. URL https://linkinghub.elsevier. com/retrieve/pii/S2405896317320955.

B.T. Polyak. Some methods of speeding up the convergence of iteration methods. USSR Computational Mathematics and Mathematical Physics, 4(5):1–17, January 1964. ISSN 00415553. doi: 10.1016/0041-5553(64)90137-5. URL https://linkinghub.elsevier.com/ retrieve/pii/0041555364901375.

Florian A Potra and Stephen J Wright. Interior-point methods. *Journal of computational and applied mathematics*, 124(1-2):281–302, 2000.

Michael JD Powell. Algorithms for nonlinear constraints that use lagrangian functions. Mathematical programming, 14:224–248, 1978.

Guannan Qu and Na Li. On the exponential stability of primal-dual gradient dynamics. IEEE Control Systems Letters, 3(1):43–48, 2018.

Maxim Raginsky, Alexander Rakhlin, and Matus Telgarsky. Non-convex learning via Stochastic Gradient Langevin Dynamics: a nonasymptotic analysis. In Proceedings of the 2017 Conference on Learning Theory, pages 1674–1703. PMLR, June 2017. URL https://proceedings.mlr.press/v65/ raginsky17a.html. ISSN: 2640-3498.

Ivan Dario Jimenez Rodriguez, Aaron Ames, and Yisong Yue. LyaNet: A Lyapunov Framework for Training Neural ODEs. In Proceedings of the 39th International Conference on Machine Learning, pages 18687–18703.

PMLR, June 2022. URL https://proceedings. mlr.press/v162/rodriguez22a.html. ISSN:
2640-3498.

Halsey Royden and Patrick Michael Fitzpatrick. Real analysis. China Machine Press, 2010.

Andrew M Saxe, James L McClelland, and Surya Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. arXiv preprint arXiv:1312.6120, 2013.

605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659 Katya Scheinberg, Donald Goldfarb, and Xi Bai. Fast First-Order Methods for Composite Convex Optimization with Backtracking. Foundations of Computational Mathematics, 14(3):389–417, June 2014. ISSN 16153375, 1615-3383. doi: 10.1007/s10208-014-9189-9. URL http://link.springer.com/10.1007/ s10208-014-9189-9.

J. Schropp and I. Singer. A dynamical systems approach to constrained minimization. Numerical Functional Analysis and Optimization, 21(3-4): 537–551, January 2000. ISSN 0163-0563, 15322467. doi: 10.1080/01630560008816971. URL http://www.tandfonline.com/doi/abs/ 10.1080/01630560008816971.

Zebang Shen, Juan Cervino, Hamed Hassani, and Alejandro Ribeiro. AN AGNOSTIC APPROACH TO FED- ERATED LEARNING WITH CLASS IMBALANCE. 2022.

Elias M Stein. Singular integrals and differentiability properties of functions. Princeton university press, 1970.

Weijie Su, Stephen Boyd, and Emmanuel J Candes. A
differential equation for modeling nesterov's accelerated gradient method: Theory and insights. Journal of Machine Learning Research, 17(153):1–43, 2016.

Hao Wang, Zakhary Kaplan, Di Niu, and Baochun Li. Optimizing federated learning on non-iid data with reinforcement learning. In IEEE INFOCOM 2020-IEEE conference on computer communications, pages 1698–1707. IEEE, 2020.

S. Wang, X.Q. Yang, and K.L. Teo. A Unified Gradient Flow Approach to Constrained Nonlinear Optimization Problems. Computational Optimization and Applications, 25(1):251–268, April 2003. ISSN 15732894. doi: 10.1023/A:1022973608903. URL https: //doi.org/10.1023/A:1022973608903.

Ashia C. Wilson, Benjamin Recht, and Michael I. Jordan.

A Lyapunov Analysis of Momentum Methods in Optimization, March 2018. URL http://arxiv.org/ abs/1611.02635. arXiv:1611.02635 [cs, math].

Robert B Wilson. A simplicial algorithm for concave programming. *Ph. D. Dissertation, Graduate School of* Bussiness Administration, 1963.

Stephen J Wright. Superlinear Convergence of a Stabilized SQP Method to a Degenerate Solution. 1998.

Zeke Xie, Issei Sato, and Masashi Sugiyama. A Diffusion Theory For Deep Learning Dynamics: Stochastic Gradient Descent Exponentially Favors Flat Minima, January 2021. URL http://arxiv.org/abs/2002.

03495. arXiv:2002.03495 [cs].

Hiroshi Yamashita. A differential equation approach to nonlinear programming. *Mathematical Programming*, 18(1):155–168, December 1980. ISSN 0025-5610, 1436-4646. doi: 10.1007/BF01588311. URL http:// link.springer.com/10.1007/BF01588311.

Willard I Zangwill. Non-linear programming via penalty functions. *Management science*, 13(5):344–358, 1967.

Xianlin Zeng, Jinlong Lei, and Jie Chen. Dynamical Primal-Dual Nesterov Accelerated Method and Its Application to Network Optimization. *IEEE Transactions* on Automatic Control, 68(3):1760–1767, March 2023. ISSN 1558-2523. doi: 10.1109/TAC.2022.3152720.

URL https://ieeexplore.ieee.org/ document/9718149/?arnumber=9718149.

Conference Name: IEEE Transactions on Automatic Control.

Dongdong Zhang and Zhongwen Chen. Superlinear convergence of a stabilized SQP-type method for nonlinear semidefinite programming. Journal of Applied Mathematics and Computing, October 2024. ISSN 1598-5865, 1865-2085. doi: 10.1007/s12190-024-02277-z. URL https://link.springer.com/10.1007/ s12190-024-02277-z.

Changhong Zhao, Ufuk Topcu, Na Li, and Steven Low.

Power system dynamics as primal-dual algorithm for optimal load control. *arXiv preprint arXiv:1305.0585*, 2013.

Limei Zhou, Yue Wu, Liwei Zhang, and Guang Zhang. Convergence analysis of a differential equation approach for solving nonlinear programming problems. Applied Mathematics and Computation, 184(2):789–797, January 2007. ISSN 0096-3003. doi: 10.1016/j.amc.2006.05.190.

URL https://www.sciencedirect.com/ science/article/pii/S0096300306007582.