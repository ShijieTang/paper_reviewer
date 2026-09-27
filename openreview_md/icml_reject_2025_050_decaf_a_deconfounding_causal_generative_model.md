# Ò **Decaf: A Deconfounding Causal Generative Model**

Anonymous Authors1

## Abstract

Causal generative models (CGMs) have recently emerged as capable approaches to simulate the causal mechanisms generating our observations, enabling causal inference. Unfortunately, existing approaches either are *overly restrictive*, assuming the absence of hidden confounders, or lack generality, being tailored to a particular query and graph. In this work, we introduce Decaf, a CGM that accounts for hidden confounders in a single amortized training process using only observational data and the causal graph. Importantly, Decaf can provably identify all causal queries with a valid adjustment set or sufficiently informative proxy variables. Remarkably, for the first time to our knowledge, we show that a confounded counterfactual query is identifiable, and thus solvable by Decaf, as long as its interventional counterpart is as well. Our empirical results on diverse settings—including the Ecoli70 dataset, with 3 independent hidden confounders, tens of observed variables and hundreds of causal queries—show that Decaf outperforms existing approaches, while demonstrating its out-of-the-box flexibility.

## 1 **Introduction**

Causal queries, or *what if* questions, seek to determine how changes in one variable affect another, which is crucial to evaluate the effects of interventions in fields such as healthcare (Feuerriegel et al., 2024), marketing policies (Varian, 2016) or education (Zhao & Heffernan, 2017). Importantly, when empirical trials are infeasible due to ethical, financial, or practical constraints, answering causal queries from observational data becomes essential. To adress this challenge, causal generative models (CGMs) (Javaloy et al., 2023; Chao et al., 2023; Khemakhem et al., 2021) have recently emerged as powerful and flexible tools for modelling structural causal models (SCMs), allowing 1Anonymous Institution, Anonymous City, Anonymous Region, Anonymous Country. Correspondence to: Anonymous Author <anon.email@domain.com>.

1

b1191 eutG
fixC
sucA
yceP
traA ygcE
atpG
cchB
ibpB yfaD
asnA
cspG
atpD
lacA
cspA yeoC
pspB
yedE
b1963 icdA
aceB yhel lacY
yfiA lpdA
lacZ
nuoM
hupB
pspA
ycgX
folK dnaK
yaeM
mopB ftsJ nmpC
b1583 dnaG
Figure 1: Decaf can be effortlessly applied to highly complex causal graphs, such as that of the Ecoli70 dataset
(Schafer & Strimmer ¨ , 2005), with multiple independent hidden confounders and dozens of variables. We dash hidden confounders, and highlight direct *confounded* effects that are now identifiable, or still unidentifiable, with Decaf. for efficiently sampling interventional and counterfactuals distributions, and enabling the estimation of any causal query of interest. However, all existing CGMs also assume causal sufficiency, i.e., that all confounders are observed.

However, *causal sufficiency* is rarely satisfied in practice, making *hidden confounding* a major challenge in causality, as it generally renders causal queries *unidentifiable*, i.e., that they cannot be uniquely expressed as a function of the observations. While recent advances have shown that some confounded causal queries are identifiable if there exist sufficiently informative proxies of the hidden confounders (Miao et al., 2018; 2023; Wang & Blei, 2021), these approaches are still limited to specific intervention-outcome pairs and do not allow for counterfactual estimation. Our objective is to bridge the gap between these two lines of work. To this end, we introduce the deconfounding causal normalizing *flow* (Decaf Ò), to the best of our knowledge, the first CGM that allows the estimation of any *identifiable* causal query—including *counterfactuals*—in the presence of *hidden confounders*, with only observational data, the causal graph, and a single amortized training process. More in detail, Decaf resembles variational autoencoders (Kingma
& Welling, 2014) as it is trained with an ELBO and comprises: i) a causal normalizing flow (CNF) (Javaloy et al.,
000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109 2023) as "decoder", adapted to be conditioned on the (potentially many) hidden confounders; and ii) a conditional normalizing flow (Winkler et al., 2019) as "encoder", computing the posterior distribution of the hidden confounders. Furthermore, we theoretically demonstrate that Decaf accurately estimates all identifiable causal queries (interventional and *counterfactual*) for which we can find a valid adjustment set or sufficiently informative proxy variables, significantly extending existing results from prior works (Miao et al., 2018; Wang & Blei, 2021; Javaloy et al., 2023).

All of the above is well illustrated in the Ecoli70 dataset
(Schafer & Strimmer ¨ , 2005), whose causal graph is depicted in Fig. 1. Specifically, by training Decaf *once* on this dataset, we can efficiently model all 43 observed variables and 3 independent hidden confounders and, most importantly, compute any causal query on demand during deployment. Out of all the direct causal effects (i.e., edges) in Fig. 1, Decaf can accurately estimate all unconfounded effects, as well as 8 out of the 11 confounded ones. In stark contrast with previous works, Decaf also estimates counterfactual queries, increasing the previous count to 16 identifiable queries. In order to assist practitioners, we provide algorithms to easily check whether a particular query of interest is identifiable in our framework, and we will make our code publicly avaiable upon acceptance. Moreover, we empirically validate our claims on semi-synthetic and real-world experiments, demonstrating that Decaf outperforms existing alternatives while being widely applicable. Therefore, *Decaf offers a* practical and efficient solution for causal inference in the presence of hidden confounding, bridging the gap between general CGMs and specialized solutions.

## 2 **Related Works**

We discuss the most relevant works to put Decaf into context, and provide a more detailed literature review in App. D. Generative causal models. In order to faithfully learn a SCM, one common approach consists modeling each variable as a function of its causal parents with an independent model, starting from the root nodes. As of the choice for modeling these functions, prior works range from simple yet well-established additive noise models (ANMs) (Hoyer et al., 2008), to more complex but powerful diffusion-based causal models (DCMs) (Chao et al., 2023), among others (Kocaoglu et al., 2018; Yang et al., 2020; Pawlowski et al., 2020; Parafita & Vitria`, 2022). Due to its nature, this approach typically is parameter-intensive, and can easily overfit and propagate errors to descendant variables.

Alternatively, recent works have explored using a single (structurally-constrained) network to model the SCM at once, e.g., using autoregressive flows (Khemakhem et al.,
2021; Javaloy et al., 2023), or graph neural networks (GNNs) (Zecevi ˇ c et al. ´ , 2021; Sanchez-Mart ´ ´ın et al., 2022).

Among these, the causal normalizing flow (CNF) deserves special attention, given its flexibility and theoretical guarantees, which we discuss later in §3.2. Most importantly, all the approaches above assume *causal sufficiency*, i.e. the absence of hidden confounders, limiting their applicability in settings with hidden confounding.

Causal inference with latent confounders. Another line of work relies on structural assumptions for correctly answering causal queries. However, these approaches typically deal only with interventional queries (i.e., not counterfactual ones) and are tailored to a specific causal graph and a single treatment-outcome pair, requiring us to train one model per query. In particular, existing works exploit instrumental variables (IVs) (Angrist & Pischke, 2009) or mediators (Pearl, 2009) to achieve this goal and, more recently, a body of works exploit proxy variables to account for latent confounding (Allman et al., 2009; Kuroki & Pearl, 2014; Kallus et al., 2018; Louizos et al., 2017; Miao et al., 2023; 2018). Of particular interest is the Deconfounder by Wang & Blei (2021), a probabilistic model that interprets multiple treatments as null proxies to find a substitute of the hidden confounders and estimate causal queries.

## 3 **Background** 3.1 **Confounded Structural Causal Models**

Next, we introduce some ideas from the causality literature used throughout this work to model the causal structure of the data and answer causal queries of interest. Definition 1. A (confounded) Structural Causal Model
(SCM) is a triplet M := (f, Pu, Pz) describing a datagenerating process over a set of D observed (endogenous)
variables x := (x1, x2*, . . . ,* xD) as

$$\begin{array}{r l}{{\mathrm{\boldmath~r~}}}&{{i=1,2,\ldots,D\,,}}\\ {{}}&{{}}\\ {{\mathrm{\boldmath~1~}}_{D})\sim P_{\mathbf{u}}\,,\ \mathbf{z}\sim P_{\mathbf{z}}\,,}\end{array}\tag{1}$$

xi:= fi(pa(i), ui, z) for i = 1, 2*, . . . , D ,* (1) with u := (u1, u2*, . . . ,* uD) ∼ Pu , z ∼ Pz ,
and where fi represents the structural equation to compute the i-th endogenous variable, xi, from its observed *causal* parents, pa(i), the i-th exogenous variable, ui, and the vector of *hidden confounders*, z.

1 Note that, while we make the dependence on the hidden confounders explicit for all observed variables in Eq. 1, we assume w.l.o.g. that a subset of them may not be directly affected by the hidden confounders. Furthermore, given a SCM M, we denote by G the *faithful* causal graph that it induces, representing *only* the direct causal relationships between pairs of endogenous and hidden variables and, when necessary, also exogenous variables. One key element in causality is the do operator (Pearl, 2012), denoted by do(t), which conceptualizes the action of ex1Bold denotes random vectors.

ternally intervening on a treatment variable t, i.e., to set t to a fixed value independently of its parents. In turn, the do operator enables the computation of interventional and counterfactual queries in SCMs (Peters et al., 2017), i.e., of population and instance-wise *what if* questions.

Definition 2. A *causal query* Q(M) := p(y | do(t), c) is a distribution over y ∈ x (the *outcome* variable), as a result of intervening upon the variable t ∈ x (the *treatment* variable). Additionally, Q(M) denotes an *interventional* or counterfactual query if the variable c is, respectively, the empty set or the vector of factual observed values, x f.

However, in the presence of *hidden confounders*, one cannot simply apply the do-operator to evaluate causal queries, as the computations involve the causal parents and the unaccounted confounders would bias the results. Instead, one needs to find alternative ways to compute these quantities if possible, as we discuss in §2.

## 3.2 **Causal Normalizing Flows**

Causal normalizing flows (CNFs) (Javaloy et al., 2023) play an important role in this work, as they form the basic building blocks of Decaf, given their identifiability guarantees despite a mild set of assumptions. Similar to Eq. 1, a CNF is defined as a pair (Tθ, Pu) forming a data-generating process that yields a set of D endogenous variables as x := T
−1 θ(u), where u ∼ Pu and Tθ : R
D → R
D is a normalizing flow (Papamakarios et al.,
2021). In particular, Tθ is a normalizing flow with additional structural constraints, ensuring that it induces the same causal graph as the underlying SCM. Javaloy et al. (2023) demonstrated that CNFs form a general class of identifiable SCMs, and that they can approximate the underlying SCM as closely as required simply by maximizing the observed joint evidence, i.e., maxθ log pθ(x).

Moreover, CNFs also allow for efficient sampling of any interventional and counterfactual distribution, enabling their use for complex causal-inference task. Unfortunately, as discussed in §1, CNFs need to assume causal sufficiency to provide the above guarantees, thus limiting their applicability. In this work, we attempt to address this limitation and account for the presence of hidden confounders without losing theoretical guarantees.

## 4 **Problem Statement**

In this work, we assume the existence of an unobserved confounded SCM, M, as in Def. 1, of which we have access to N i.i.d. observations and its induced causal graph, G.

Our objective is therefore to design a CGM that can faithfully answer as many causal queries from the original SCM as possible, despite the presence of unobserved hidden confounders. In other words, to find a substitute model of M that we can use to accurately perform causal inference.

x T
−1 ϕz T
−1 θ ε u x G
Figure 2: **Sketch of Decaf architecture**. Tϕ and Tθ are conditional normalizing flows, with the top input as condition; G is the causal graph, and ε is a non-causal random variable needed by the normalizing flow to sample z. Assumptions. Regarding the underlying SCM M, we simply assume that it i) has C
1-diffeomorphic structural equations,2and ii) induces an acyclic graph. We denote the family of SCMs meeting these assumptions by M.

## 5 **Deconfounding Causal Normalizing Flows**

To help bridge the gap between CGMs and tailored hiddenconfounding solutions, we now present the deconfounding causal normalizing flow (or Decaf Ò). Intuitively, Decaf takes a well-grounded CGM such as the causal normalizing flow (Javaloy et al., 2023), which can provably approximate unconfounded SCMs and perform causal inference, and expand it such that it accounts for hidden confounding by building data-driven substitutes of these confounders, an idea that has been successfully explored in the past (Wang & Blei, 2019; 2021; Bica et al., 2020).

Decaf achieves the above by following a similar structure as that of a variational autoencoder (Kingma & Welling, 2014). That is, Decaf comprises two main components. First, an inference network which approximates the intractable posterior distribution of the hidden confounders, given their observed children. Second, a generative network that exploits structural constraints to accurately model the underlying SCM, given a substitute for hidden confounders. Each of these parts comes with their own challenges, however, which we now explain in detail: Generative network. As mentioned in §3.2, we use CNFs (Javaloy et al., 2023) as our starting point. However, since our generative model needs to take in hidden confounders as conditional inputs, we adapt CNFs to use conditional normalizing flows (Winkler et al., 2019), instead of unconditional ones. The resulting model, Tθ, is thus an invertible transformation describing a data-generating process, conditioned on z, which can map a set of exogenous variables u to our observations and vice versa, i.e.,
Tθ(x, z) = u ∼ Pu and x = T
−1 θ(u, z), (2)
110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 where we further exploit the given causal graph to ensure that the generative process is faithful, i.e., such that

$$p_{\boldsymbol{\theta}}(\mathbf{x}\mid\mathbf{z})=\prod_{i=1}^{D}p_{\boldsymbol{\theta}}(\mathbf{x}_{i}\mid\mathrm{pa}(i),\mathbf{z})\,,$$

165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 defining now a process similar to that given in Def. 1. Just as in Def. 1, only the children of z will actually condition on z in Eq. 3. Furthermore, Tθ allows us to write down the exact likelihood of the data given z,

$$\log p_{\theta}(\mathbf{x}\mid\mathbf{z})=p_{\mathbf{u}}(T_{\theta}(\mathbf{x},\mathbf{z}))|\mathrm{det}(\nabla_{\mathbf{x}}T_{\theta}(\mathbf{x},\mathbf{z}))|\,.$$

Deconfounding network. To model the posterior distribution of the hidden confounders given our observations, i.e., the abduction step needed to compute counterfactuals (Pearl, 2009), we use another conditional normalizing flow (Winkler et al., 2019), as it can approximate the true posterior distribution arbitrarily well. Once again, we exploit prior knowledge about the causal graph and mask the resulting network, Tϕ, such that it models each *independent* hidden confounder zk using only its observed children, i.e.,

$$q_{\phi}({\bf z}\mid{\bf x})=\prod_{k=1}^{D_{\bf z}}q_{\phi}({\bf z}_{k}\mid{\rm ch}({\bf z}_{k}))\,,\tag{5}$$

where Dz is the number of independent hidden confounders.

Training process. We jointly train both networks defined above as it would be typically done in deep latent-variable models, i.e., during training we *maximize* the evidence lower bound (ELBO) (Kingma & Welling, 2014):

$$\mathcal{L}(\mathbf{\theta},\mathbf{\phi})=\mathbb{E}_{q_{\phi}}\left[\log p_{\mathbf{\theta}}(\mathbf{x}\mid\mathbf{z})\right]-\mathrm{KL}[q_{\mathbf{\phi}}(\mathbf{z}\mid\mathbf{x})\|\,p(\mathbf{z})]\tag{6}$$ $$=\mathbb{E}_{q_{\phi}}\left[\log p_{\mathbf{\theta}}(\mathbf{x},\mathbf{z})\right]+\mathrm{H}(q_{\mathbf{\phi}}(\mathbf{z}\mid\mathbf{x}))\,,\tag{7}$$
$$\max_{\phi,\theta}\ {\cal L}(\phi,\theta)=\min_{\phi,\theta}\ {\rm KL}[p_{\rm data}({\bf x})\|\,p_{\theta}({\bf x})]$$ $$+\ {\rm KL}[q_{\phi}({\bf z}\mid{\bf x})\|\,p_{\theta}({\bf z}\mid{\bf x})]\,.\tag{8}$$

Causal inference. Since the tuple (Tθ, Pu, Pz) defines a confounded SCM as defined in Def. 1, we can use Decaf

$$(3)$$

to efficiently sample from observational and interventional distributions by: i) sampling z from p(z); and ii) sampling x from either pθ(x | z) or pθ(x | z, do(t)), as proposed by Javaloy et al. (2023). For counterfactual inference, we can use the deconfounding network to perform the induction step, as the second KL term in Eq. 8 shows that it approximates the posterior induced by Tθ (i.e., its z-inverse given x).

Therefore, to generate counterfactual samples we simply need to: i) sample from qϕ(z | x f); and ii) sample again from pθ(x | z, do(t)). We provide more details about these steps and the do-operator in App. C.

$$(4)$$

## 6 **Theoretical Results**

We take advantage that our work is at intersection of CGMs and hidden-confounding solutions to leverage and expand the theory of both research fields. While we present here an intuitive summary of our main theoretical results, formal statements and derivations can be found in App. A. Note that, throughout this section, we assume that Decaf matches the true data evidence, i.e., pdata(x) = pθ(x).

Given that CNFs (and hence Decaf) are universal density approximators (Papamakarios et al., 2021), we should be able to always meet this assumption, provided enough resources.

## 6.1 **Causal Query Identifiability**

First, we study which queries are identifiable with Decaf. We call a query identifiable if we are guaranteed to produce the same query distribution as the original SCM by matching the data evidence. More formally, we adopt the following definition (Pearl, 2009, Def. 3.2.4):

$$\begin{array}{l}{(6)}\\ {(7)}\end{array}$$

Definition 3. Let Q(M) be a causal query of a model M.

We call Q *identifiable* if, for any two models M1,M2 ∈ M,
Q(M1) = Q(M2) whenever pM1(x) = pM2(x) > 0 .

Another relevant concept for this section is that of a *valid* adjustment set (Peters et al., 2017, Def. 6.38). In plain terms, if we were to compute a causal query, say p(y | do(t)), a valid adjustment set b is a subset of variables such that: i) it blocks all backdoor paths between y and t, and ii) it is independent of the variable t after severing all incoming edges in t in the associated causal graph. As a consequence, we can use b to apply the adjustment formula,

$$p({\bf y}\mid{\rm do(t)})=\int p({\bf y}\mid{\rm t,b})\;\;p({\bf b})\;{\rm d}{\bf b}\,.\tag{9}$$

Additionally, we refer to b as *invalid* if only i) holds.

6.1.1 INTERVENTIONAL QUERIES
We first look at the identifiability of interventional queries, i.e., queries of the form Q(M) = pM(y | do(t)), where y, t ∈ x are any two endogenous variables. We summarize our findings in the following proposition, which we properly formalize in App. A.2:
where p(z) is the prior distribution of z, KL the Kullback- Leibler divergence (Kullback & Leibler, 1951), and H the differential entropy (Kolmogorov, 1956). The motivation for this choice is three-fold. First, we want the generative network to explain the observations given samples from qϕ (first term of Eq. 6). Second, as we do not know the optimal size for z, we need to prevent the deconfounding network from allocating information exclusive of x in z (entropy term in Eq. 7). Finally, all the theory in §6 relies on Decaf matching the data evidence, pdata(x), which we encourage Decaf to do since 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274

n z w L /
R
t y b
Figure 3: **Generic causal graph** where we are interested in the interventional query Q(M) = p(y | do(t)). Blue and red edges play a crucial role, as their presence or absence induce different types of identifiability conditions. Proposition 6.1 (Informal). Decaf is able to identify a given interventional causal query if one of the following exists:
i) a valid adjustment set b not containing z, ii) an invalid one where p(b | do(t)) is identifiable, or iii) *sufficiently informative proxy and null proxy variables.*
To help us go through the requirements in Prop. 6.1, let us break them down with the example depicted in Fig. 3 where, depending on the presence or absence of edges L and R, we face qualitatively different identifiability scenarios: 1. Unconfounded case, LR. If neither treatment nor outcome are directly influenced by z, then we can always find a valid adjustment set that does not include z. We extend the results of Javaloy et al. (2023) to show that Decaf can identify any interventional causal query of this type. 2. Confounded-treatment case, LR. If only the treatment is directly affected by z, we run into two possible scenarios.

First, if we are able to find a valid adjustment set e.g., b and w in Fig. 3, then Decaf can always identify the interventional query. Otherwise, Decaf could still identify the query if we find an *invalid* adjustment set where p(b | do(t)) is still identifiable by Decaf. 3. Confounded-outcome case, LR. When only the outcome variable directly depends on z, Decaf can identify any interventional query, as it necessarily exists a valid adjustment set not containing z. In our running example, variables n and b would block all backdoor paths in Fig. 3, and Decaf would properly estimate the interventional query. 4. Fully-confounded case, LR. When both variables directly depend on z, identifiability is more challenging, as any adjustment set necessarily involves the hidden confounder. In this case, we extend in Prop. A.2 the results from Miao et al. (2018) and Wang & Blei (2021) to allow for general causal graphs and additional covariates. In short, we find that an interventional query is identifiable if we can find: i) a proxy w, independent of t, to distinguish z from the exogenous variables u; and ii) a null proxy n, independent of y given t and z, to discern the correct structural equation. Additionally, as in prior works (Miao et al., 2018; Wang & Blei, 2021), z should be *complete* given the proxies (refer to

t cf y cf ôôô n cf wcf un ut z uy uw n f wf t f y f
Def. 5 for a formal definition). That is, both proxies should be sufficiently informative to accurately approximate z.

## 6.1.2 Counterfactual Queries

We focus next on the identifiability of counterfactual queries, i.e., queries of the form Q(M) = pM(y cf | do(t cf), x f),
where x fis the observed factual, and where we are interested in the distribution the outcome would have had, had we intervened on the treatment variable. We demonstrate, for the first time to our knowledge, that counterfactual query identifiability holds for as many queries as for the interventional case. More specifically, we show that: Proposition 6.2 (Informal). *When an interventional query* p(y | do(t)) is identifiable by Decaf, then it equally identifies the counterfactual query p(y cf | do(t cf), x f).

The formal result can be found in Prop. A.7. In short, our result means that, if we can identify an interventional query, then we can identify its counterfactual counterpart as well. Our result exploits the notion of twin SCM (Balke & Pearl, 1994), which duplicates the structural equations for the factual and counterfactual worlds while sharing the exogenous variables, and the fact that Prop. A.2 allows for queries with additional covariates as long as they do not form colliders, which is always the case with x fin pM(y cf | do(t cf), x f), as we show in the example twin network from Fig. 4.

## 6.2 **Identifying Exogenous Distributions**

Besides causal query identifiability, another question of interest is whether Decaf recovers the true exogenous variables, up to component-wise transformations, disentangling the sources of variability of each endogenous variable. In App. A.1, we expand the results of Javaloy et al. (2023) to prove that Decaf identifies3the underlying SCM for those variables not directly affected by z, i.e.: Corollary 6.3 (Informal). Decaf identifies the underlying SCM, restricted to every variable other than ch(z)*, up to an* 3In the sense of Xi & Bloem-Reddy (2023).

element-wise transformation of the exogenous distribution. Moreover, we conjecture that Decaf should in most cases properly disentangle the rest of exogenous variables and z, Although we do not formally prove it, we refer to the use case of §7.3 to illustrate that the exogenous variables and the latent variables extracted by Decaf. Our intuition is that, if some children of z are conditionally independent, the information common to them can only be explained via z. In addition, the entropy term in Eq. 7 discourages Decaf from using the components of z that are not necessary for explaining the observations. Recent works proved similar results under slightly stronger assumptions (von Kugelgen ¨
et al., 2021; Zheng et al., 2022; Brady et al., 2023).

## 6.3 **Practical Guidelines & Implications**

In this section, we outline the different aspects to consider for the successful application of Decaf to solve causal queries in real-world scenarios. Training. One key advantage of Decaf is that it needs to train only once per dataset. However, maximizing the ELBO
makes it also susceptible to posterior collapse (Wang et al., 2021), i.e., to the KL term in Eq. 6 vanishing, and hence the posterior equating the prior distribution. Fortunately, we can leverage existing solutions, e.g., implement regularization terms as the one proposed by Vahdat & Kautz (2020). Recall also that, following §6, model selection should use an observational goodness-of-fit metric as selection criterion. Solving causal queries. Whilst Decaf can compute any causal query, unidentifiable causal queries may still lead to incorrect estimates. To ensure reliability, we must verify the identifiability of each specific query of interest, for which we provide algorithms that check identifiability in the causal graph in App. E. Namely, Alg. 5 checks if a query that involves a specific treatment-outcome pair, which includes average treatment effects and counterfactuals, is identifiable. If we were interested in a query on all variables, e.g., as samples from an interventional distribution, we should evaluate the identifiability of the causal effects between the treatment and all its descendants, as proposed in Alg. 6. Limitations. Decaf relaxes the assumption of causal sufficiency, but it still relies on completeness for the proxies, a common condition for nonparametric identification in causal inference (D'Haultfoeuille, 2011; Chen et al., 2014). This condition is untestable with observational data alone, though collecting additional proxies can help satisfy completeness (Andrews, 2011). Moreover, we assume that the true SCM is C
1-*diffeomorphic* with respect to the exogenous variables, which precludes theoretical guarantees for modeling discrete variables, although Javaloy et al. (2023);
de Vassimon Manela et al. (2024) show that, in practice, CNFs effectively approximate discrete distributions.

275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329

Linear Nonlinear S
0 1 2 3 5 10 Oracle 0.5 1.0 CF
 error 0 2 4 6 8 10 Dz 0.15 0.20 0 2 4 6 8 10 Dz
Figure 5: **Ablation**. Counterfactual error as we change the number of proxy variables and the latent dimensionality. We show means and 95 % confidence intervals over 5 realizations, intervening on the 25th, 50th, and 75th percentile of t.

## 7 **Empirical Evaluation**

In this section, we assess the performance of Decaf comparatively to existing methods. Namely, we show that Decaf accurately estimate interventional and counterfactual queries when the requirements of Prop. 6.1 are met, and that it effectively estimates the exogenous information. We provide all experimental details in App. B. Common evaluation. For all experiments, we estimate the performance on the interventional and counterfactual regimes via the mean absolute error (MAE) of, respectively, the average treatment effect (ATE) and the counterfactual samples, with respect to the ground-truth values. Moreover, we use as reference a CNF that *does observe* the hidden confounders, which we refer to as *oracle*. We also account for differences across observed variables by computing all errors over the standardized variables.

## 7.1 **Ablation Study**

First, we conduct a simple ablation to understand how misspecifying the dimensionality of z may affect Decaf, as well as its sensitivity to the number of available proxies. For additional details and results, refer to App. B.1.

. . .

Experimental setup. We consider two synthetic SCMs, linear and non-linear, that follow the causal graph depicted in the inset figure, comprising two independent hidden confounders affecting every variable, and S null proxies. Then, we evaluate how well Decaf estimates the direct effect of t on y while changing the number of proxy variables, S, and the specified latent dimensionality, Dz.

n1 nS z1 z2 t y Results. Fig. 5 shows the counterfactual error for all cases, where we clearly observe that increasing the number of proxies reduces them, and with a drastic change as we add the second proxy, corroborating Prop. 6.1.

Similarly, we observe that underestimating Dz increases the error (especially if we assume causal sufficiency, Dz = 0)
while overestimating it does not. This indicates that, indeed, the entropy term in Eq. 6 prevents non-shared information from being modeled through z, as discussed in §5.

## 7.2 **Semi-Synthetic Experiments**

Next, we evaluate how Decaf performs relatively to existing approaches and, to this end, we consider semi-synthetic datasets for which we have access to the ground-truth SCMs.

Additional details can be found in Apps. B.2 and B.3.

Baselines. We compare Decaf with three CGMs which assume causal sufficiency and are thus *unaware* of the hidden confounders: CNFs (Javaloy et al., 2023); ANMs (Hoyer et al., 2008); and DCMs (Chao et al., 2023); and with the Deconfounder (Wang & Blei, 2019), which uses proxies to provide unbiased ATE estimates under hidden confounding, yet it requires a model per treatment-outcome pair. We use the oracle as reference model to lower bound the error.

## 7.2.1 Protein-Signalling Networks

We first conduct a similar semi-synthetic experiment as that of Chao et al. (2023), based on a protein-signalling network 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384

PKC PKA
Raf Jnk Mek Erk Akt P38 Plcg PIP3 PIP2
Figure 7: **Sachs' causal** graph. Green denotes identifiable confounded effects.

dataset (Sachs et al., 2005).

Specifically, we randomly generate a non-linear SCM
that induces the same causal graph as the original dataset, depicted in Fig. 7, except for the root nodes, for which we use the original data. As a result, we have a hidden confounder with two dimensions, PKC and PKA, and three treatment variables to intervene upon, Raf, Mek, and Erk. We consider additive and non-additive structural equations, measure the effect of interventions on the downstream nodes and, more importantly, ensure that the randomized effect of the hidden confounder is perceptible.

Results. We present a summary of the results in Fig. 6a, where we can observe that Decaf outperforms every approach in all cases, for both ATE and counterfactual errors, remaining fairly close to the oracle model. Moreover, we appreciate a great difference in performance between Decaf and CNFs, which corroborates the importance of the proposed encoder and variational training employed by Decaf, since a CNF is equivalent to Decaf with Dz = 0 .

7.2.2 GENE NETWORKS
Next, we repeat a similar experiment as in the previous section, considering this time the causal graph of the Ecoli70 dataset (Schafer & Strimmer ¨ , 2005) as reference, shown in Fig. 1, representing a gene network from E. coli data. This time, we replace root nodes with Gaussian variables. Results. Similar to the previous case, the results presented in Fig. 6b demonstrate that Decaf is indeed able to closely match the performance of the oracle model, outperforming existing approaches. However, the non-additive case also shows significant long-tailed error distributions for all models, showing that Decaf can suffer the same problems as any data-centric approach, and that it is still needed to put attention on its effective training.

0.0 0.2 0.4 0.6 0.8
ATE err or It is also worth-pointing out that the striking performance of the Deconfounder is a result of evaluating causal queries that cannot be identified by the model. As we discuss in App. B, the Deconfounder offers guarantees regarding ATE estimation and with more restrictive assumptions. If we plot instead the ATE error evaluated on only those paths that meet the assumptions placed by the Deconfounder, as shown in the inset figure, we see that it now achieves significantly lower errors that the *unaware* approaches. Remarkably, this experiment highlights every strength of the proposed approach, since Decaf: i) models several hidden confounders affecting different sets of variables; ii) identifies all causal queries for which we have some proxy information; and **iii)** achieves the above in an agnostic manner, i.e., training out-of-the-box and *one single time*, despite the graph having 43 observed variables.

## 7.3 **Fairness Real-World Use Case**

Taking inspiration from the experiments by Kusner et al. (2017) and Javaloy et al. (2023), we aim to show how model-

Oracle Decaf Deconfounder CNF ANM DCM
Oracle Decaf Deconfounder CNF ANM DCM
ATE error CF error Additive 0.0 0.5 1.0 1.5 ATE error CF error Additive 0.0 0.5 1.0 ATE error CF error Nonadditive ATE error CF error Nonadditive
(a) **Sachs' dataset**.

(b) **Ecoli70 dataset**.
Figure 6: Error boxenplots for different CGMs, averaged over all identifiable direct effects of the Sachs (Fig. 7) (a) and Ecoli70 (Fig. 1) (b) datasets, after intervening in their 25th, 50th, and 75th percentiles in 5 random initializations.

| Unfair   | Unaware   | Decaf Ò Fair K   | Fair Add   | Mean   |       |      |
|----------|-----------|------------------|------------|--------|-------|------|
| RMSE     | 1.477     | 1.479            | 1.652      | 2.818  | 2.817 | 2.83 |
| MMD      | 0.110     | 0.102            | 0.0018     | 10−6   | 10−8  | 0    |

385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439

Sex GPA
Race LSAT Know Decile3 Fam FYA
ling confounded SCMs with Decaf can be leveraged beyond causal query estimation and, in particular, for counterfactual fairness prediction. See App. B.4 for further details.

Dataset and objective. Our aim is to train a predictor, using the law school dataset (Wightman, 1998) which comprises information of 21 790 law students, that remains accurate while being fair—using *demographic parity* as fairness criterion (Feldman et al., 2015)—toward the sensitive attributes of the students. In particular, we are interested in predicting the decile of a student in its 3rd year of university, given their undergraduate and 1st year grades, family income, race, and sex. Experimental setup. First, we train Decaf assuming a causal graph such as the one in Fig. 8, excluding Decile3, where all grades are affected by a common "knowledge" hidden confounder. Then, we train a simple predictor using as input the hidden confounder and non-sensitive exogenous variables estimated by Decaf. If, as discussed in §6.2, Decaf successfully recovers the exogenous variables, we expect the predictor to be fair yet slightly less accurate, since Decile3 is directly affected by the sensitive attributes. Results. Tab 1 shows the prediction error (RMSE) and the difference between groups (MMD) for the proposed predictor using Decaf, comparing with an *unfair* predictor that uses sensitive attributes; an *unaware* predictor that excludes sensitive attributes, and two fair predictors—*Fair K* and Fair Add—proposed by Kusner et al. (2017). As shown in Fig. 9, Decaf provides a much fairer predictor than the *unfair* and the *unaware* predictors at the cost of slightly higher RMSE. We can also appreciate that the other two fair approaches are so by predicting a constant value for every individual, which can be also observed comparing the RMSE obtained by these predictors with a naive predictor that predicts the mean of the distribution in Tab 1.

U
nfair Sex Race U
naw ar e Dec af Fair K
5 0 5 10 15 3rd year decile Fair AddFemale Male White Non-white 5 0 5 10 15 3rd year decile

## 8 **Concluding Remarks**

In this work, we have bridged the current gap between CGMs, which fail to account for hidden confounders, and hidden-confounding solutions, which are tailored to a specific causal query and thus need to train once per query.

To this end, we have introduced Decaf, and theoretically shown that it can accurately estimate causal queries in the presence of hidden confounders, if there exists a valid adjustment set or sufficiently informative proxies, extending prior results (Miao et al., 2018) to also consider counterfactuals. We have empirically shown that Decaf outperforms all considered baselines, better estimating confounded causal queries shown to be identifiable, and properly identifying exogenous distributions to train fair classifiers. Finally, we have provided algorithms to check the identifiability of causal queries which, along Decaf, provides practitioners with a powerful pipeline to perform causal inference in the presence of hidden confounders. Future work. Our work opens many intriguing venues, e.g., integrating alternative identification strategies, such as instrumental variables (Hartford et al., 2017), to expand the range of identifiable queries that Decaf can estimate. We also find it interesting to apply Decaf to settings with timevarying treatments, where multiple interventions have to be performed. In real-world scenarios, it would be exciting to include interventional data during training, and seeing Decaf applied to real-world problems such as decision support systems (Sanchez et al., 2022), educational analysis
(Murnane, 2010), or policy making (Fougere & Jacquemet ` , 2021), yet always validating them with interventional data.

## Impact Statement

440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 486 487 488 489 490 491 492 493 494 This research contributes to advance causal inference in machine learning, particularly enhancing the ability to estimate causal effects despite unobserved variables. Thus, this work supports more informed decision-making in scenarios where controlled experimentation is impractical or unethical, such as healthcare or education. As with all advances in causal inference, practitioners should be aware of the limitations and assumptions of causal models. Particularly, in sensitive applications, where decisions are based on accurate causal conclusions, validation with interventional data should be prioritized whenever possible to ensure reliability.

Overall, this work aligns with the broader goal of improving machine learning and does not introduce significant ethical risk beyond those traditionally associated with the field.

## Bibliography

Allman, E. S., Matias, C., and Rhodes, J. A. Identifiability of parameters in latent structure models with many observed variables. *The Annals of Statistics*, 37(6A):3099–3132, 2009. ISSN 00905364, 21688966. URL http://www. jstor.org/stable/25662188. (Cited in pages 2, 29, and 30.)
Andrews, D. W. Examples of L2-complete and boundedlycomplete distributions. 2011. (Cited in pages 6 and 15.)
Angrist, J. D. and Pischke, J.-S. Mostly harmless econometrics: An empiricist's companion. Princeton university press, 2009. (Cited in pages 2 and 29.)
Balke, A. and Pearl, J. Probabilistic Evaluation of Counterfactual Queries. *Probabilistic and Causal Inference*,
1994. URL https://api.semanticscholar. org/CorpusID:18845266. (Cited in page 5.)
Bica, I., Alaa, A. M., and van der Schaar, M. Time Series Deconfounder: Estimating Treatment Effects over Time in the presence of hidden confounders. In *Proceedings of* the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of *Proceedings of Machine Learning Research*, pp. 884– 895. PMLR, 2020. URL http://proceedings.ml r.press/v119/bica20a.html. (Cited in page 3.)
Bingham, E., Chen, J. P., Jankowiak, M., Obermeyer, F., Pradhan, N., Karaletsos, T., Singh, R., Szerlip, P., Horsfall, P., and Goodman, N. D. Pyro: Deep universal probabilistic programming. *Journal of machine learning research*, 20(28):1–6, 2019. (Cited in page 27.)
Brady, J., Zimmermann, R. S., Sharma, Y., Scholkopf, B., ¨
von Kugelgen, J., and Brendel, W. Provably Learning ¨
Object-Centric representations. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J.

(eds.), *International Conference on Machine Learning,* ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pp. 3038–3062. PMLR, 2023. URL https: //proceedings.mlr.press/v202/brady23a. html. (Cited in page 6.)
Carrasco, M., Florens, J.-P., and Renault, E. Linear inverse problems in structural econometrics estimation based on spectral decomposition and regularization. *Handbook of* econometrics, 6:5633–5751, 2007. (Cited in page 16.)
Chao, P., Blobaum, P., and Kasiviswanathan, S. P. In- ¨
terventional and counterfactual inference with diffusion models. *ArXiv preprint*, abs/2302.00860, 2023. URL https://arxiv.org/abs/2302.00860. (Cited in pages 1, 2, 7, and 22.)
Chen, X., Chernozhukov, V., Lee, S., and Newey, W. K.

Local identification of nonparametric and semiparametric models. *Econometrica*, 82(2):785–809, 2014. (Cited in page 6.)
de Vassimon Manela, D., Battaglia, L., and Evans, R. J. Marginal Causal Flows for Validation and Inference. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. (Cited in page 6.)
D'Haultfoeuille, X. On the completeness condition in nonparametric instrumental problems. *Econometric Theory*, 27(3):460–471, 2011. (Cited in page 6.)
Feldman, M., Friedler, S. A., Moeller, J., Scheidegger, C.,
and Venkatasubramanian, S. Certifying and Removing Disparate Impact. In Cao, L., Zhang, C., Joachims, T., Webb, G. I., Margineantu, D. D., and Williams, G. (eds.), Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, Sydney, NSW, Australia, August 10-13, 2015, pp. 259–
268. ACM, 2015. doi: 10.1145/2783258.2783311. URL https://doi.org/10.1145/2783258.2783 311. (Cited in page 8.)
Feuerriegel, S., Frauen, D., Melnychuk, V., Schweisthal, J., Hess, K., Curth, A., Bauer, S., Kilbertus, N., Kohane, I. S., and van der Schaar, M. Causal machine learning for predicting treatment outcomes. *Nature Medicine*, 30(4):
958–968, 2024. (Cited in page 1.)
Fougere, D. and Jacquemet, N. Policy evaluation using `
causal inference methods. In Handbook of Research Methods and Applications in Empirical Microeconomics, pp. 294–324. Edward Elgar Publishing, 2021. (Cited in page 8.)
Hartford, J. S., Lewis, G., Leyton-Brown, K., and Taddy, M.

Deep IV: A Flexible Approach for Counterfactual prediction. In Precup, D. and Teh, Y. W. (eds.), Proceedings of 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, volume 70 of *Proceedings of Machine Learning Research*, pp. 1414–1423. PMLR, 2017. URL http://procee dings.mlr.press/v70/hartford17a.html. (Cited in pages 8 and 29.)
Hoyer, P. O., Janzing, D., Mooij, J. M., Peters, J., and Scholkopf, B. Nonlinear causal discovery with additive ¨ noise models. In Koller, D., Schuurmans, D., Bengio, Y., and Bottou, L. (eds.), Advances in Neural Information Processing Systems 21, Proceedings of the Twenty- Second Annual Conference on Neural Information Processing Systems, Vancouver, British Columbia, Canada, December 8-11, 2008, pp. 689–696. Curran Associates, Inc., 2008. URL https://proceedings.neurip s.cc/paper/2008/hash/f7664060cc52bc6 f3d620bcedc94a4b6-Abstract.html. (Cited in pages 2 and 7.)
Javaloy, A., Sanchez-Mart ´ ´ın, P., and Valera, I. Causal normalizing flows: from theory to practice. In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S. (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/p aper/2023/hash/b8402301e7f06bdc97a31 bfaa653dc32-Abstract-Conference.html. (Cited in pages 1, 2, 3, 4, 5, 6, 7, 15, 18, 19, 25, 27, and 28.)
Kallus, N., Mao, X., and Udell, M. Causal Inference with Noisy and Missing Covariates via Matrix factorization. In Bengio, S., Wallach, H. M., Larochelle, H., Grauman, K., Cesa-Bianchi, N., and Garnett, R. (eds.), Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montreal, ´
Canada, pp. 6921–6932, 2018. URL https://proc eedings.neurips.cc/paper/2018/hash/8 6a1793f65aeef4aeef4b479fc9b2bca-Abstr act.html. (Cited in pages 2, 29, and 30.)
Kallus, N., Mao, X., and Zhou, A. Interval Estimation of Individual-Level Causal Effects Under Unobserved confounding. In Chaudhuri, K. and Sugiyama, M. (eds.), The 22nd International Conference on Artificial Intelligence and Statistics, AISTATS 2019, 16-18 April 2019, Naha, Okinawa, Japan, volume 89 of Proceedings of Machine Learning Research, pp. 2281–2290. PMLR, 2019. URL http://proceedings.mlr.press/v89/kall us19a.html. (Cited in pages 29 and 30.)
Kaltenpoth, D. and Vreeken, J. Nonlinear Causal Discovery with Latent confounders. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of *Proceedings of Machine Learning Research*, pp. 15639–15654. PMLR, 2023. URL https://procee dings.mlr.press/v202/kaltenpoth23a.h tml. (Cited in page 22.)
Khemakhem, I., Monti, R. P., Leech, R., and Hyvarinen, A. ¨
Causal Autoregressive flows. In Banerjee, A. and Fukumizu, K. (eds.), *The 24th International Conference on* Artificial Intelligence and Statistics, AISTATS 2021, April 13-15, 2021, Virtual Event, volume 130 of Proceedings of Machine Learning Research, pp. 3520–3528. PMLR, 2021. URL http://proceedings.mlr.press/ v130/khemakhem21a.html. (Cited in pages 1 and 2.)
Kingma, D. P. and Welling, M. Auto-Encoding Variational Bayes. In Bengio, Y. and LeCun, Y. (eds.),
2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings, 2014. URL http: //arxiv.org/abs/1312.6114. (Cited in pages 1, 3, and 4.)
Kocaoglu, M., Snyder, C., Dimakis, A. G., and Vishwanath, S. Causalgan: Learning Causal Implicit Generative Models with Adversarial training. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net, 2018. URL https: //openreview.net/forum?id=BJE-4xW0W. (Cited in page 2.)
Kolmogorov, A. On the Shannon theory of information transmission in the case of continuous signals. IRE Transactions on Information Theory, 2(4):102–108, 1956. (Cited in page 4.)
Kullback, S. and Leibler, R. A. On information and sufficiency. *The annals of mathematical statistics*, 22(1): 79–86, 1951. (Cited in page 4.)
Kuroki, M. and Pearl, J. Measurement bias and effect restoration in causal inference. *Biometrika*, 101(2):423–437, 2014. (Cited in pages 2, 29, and 30.)
Kusner, M. J., Loftus, J. R., Russell, C., and Silva, R. Counterfactual fairness. In Guyon, I., von Luxburg, U., Bengio, S., Wallach, H. M., Fergus, R., Vishwanathan, S. V. N., and Garnett, R. (eds.), Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pp. 4066–4076, 2017. URL https://proceedings.neurips.cc/paper
/2017/hash/a486cd07e4ac3d270571622f4 f316ec5-Abstract.html. (Cited in pages 7, 8, 25, 26, and 27.)
Long, C. P. and Antoniewicz, M. R. Metabolic flux analysis of Escherichia coli knockouts: lessons from the Keio collection and future outlook. Current opinion in biotechnology, 28:127–133, 2014. (Cited in page 23.)
Louizos, C., Shalit, U., Mooij, J. M., Sontag, D. A., Zemel, R. S., and Welling, M. Causal Effect Inference with Deep Latent-Variable models. In Guyon, I., von Luxburg, U., Bengio, S., Wallach, H. M., Fergus, R., Vishwanathan, S. V. N., and Garnett, R. (eds.), Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, pp. 6446–6456, 2017.

URL https://proceedings.neurips.cc/p aper/2017/hash/94b5bde6de888ddf9cde6 748ad2523d1-Abstract.html. (Cited in pages 2, 29, and 30.)
Luo, R. and Zhao, H. Bayesian hierarchical modeling for signaling pathway inference from single cell interventional data. *The annals of applied statistics*, 5:725–745, 2011. doi: 10.1214/10-AOAS425. (Cited in page 22.)
Miao, W., Geng, Z., and Tchetgen Tchetgen, E. J. Identifying causal effects with proxy variables of an unmeasured confounder. *Biometrika*, 105(4):987–993, 2018. (Cited in pages 1, 2, 5, 8, 16, 17, 18, 29, and 30.)
Miao, W., Hu, W., Ogburn, E. L., and Zhou, X.-H. Identifying effects of multiple treatments in the presence of unmeasured confounding. Journal of the American Statistical Association, 118(543):1953–1967, 2023. (Cited in pages 1, 2, 15, 29, and 30.)
Murnane, R. *Methods matter: Improving causal inference* in educational and social science research. Oxford University Press, 2010. (Cited in page 8.)
Nasr-Esfahany, A., Alizadeh, M., and Shah, D. Counterfactual Identifiability of Bijective Causal models. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pp. 25733–25754. PMLR, 2023. URL https://proceedings.mlr.press/v202/n asr-esfahany23a.html. (Cited in page 15.)
Papamakarios, G., Nalisnick, E. T., Rezende, D. J., Mohamed, S., and Lakshminarayanan, B. Normalizing Flows for Probabilistic Modeling and inference. J. Mach. Learn.

Res., 22:57:1–57:64, 2021. URL http://jmlr.org /papers/v22/19-1028.html. (Cited in pages 3 and 4.)
550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 Parafita, A. and Vitri ´ a, J. Estimand-agnostic causal query `
estimation with deep causal graphs. *IEEE Access*, 10: 71370–71386, 2022. (Cited in page 2.)
Pawlowski, N., de Castro, D. C., and Glocker, B. Deep Structural Causal Models for Tractable Counterfactual inference. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, Neur- IPS 2020, December 6-12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/paper /2020/hash/0987b8b338d6c90bbedd8631b c499221-Abstract.html. (Cited in page 2.)
Pearl, J. *Causality*. Cambridge university press, 2009.

(Cited in pages 2, 4, 20, and 29.)
Pearl, J. The Do-Calculus Revisited. In de Freitas, N.

and Murphy, K. P. (eds.), Proceedings of the Twenty- Eighth Conference on Uncertainty in Artificial Intelligence, Catalina Island, CA, USA, August 14-18, 2012, pp. 3–11. AUAI Press, 2012. URL https://dslpitt. org/uai/displayArticleDetails.jsp?mm nu=1&smnu=2&article_id=2330&proceedi ng_id=28. (Cited in page 2.)
Pearl, J., Glymour, M., and Jewell, N. P. *Causal inference* in statistics: A primer. John Wiley & Sons, 2016. (Cited in page 28.)
Peters, J., Janzing, D., and Scholkopf, B. ¨ *Elements of causal* inference: foundations and learning algorithms. The MIT Press, 2017. (Cited in pages 3, 4, 16, 17, 19, and 20.)
Ranganath, R. and Perotte, A. Multiple causal inference with latent confounding. *ArXiv preprint*, abs/1805.08273, 2018. URL https://arxiv.org/abs/1805.0 8273. (Cited in page 30.)
Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., and Nolan, G. P. Causal Protein-Signaling Networks Derived from Multiparameter Single-Cell Data. *Science*, 308
(5721):523–529, 2005. doi: 10.1126/science.1105809.

URL https://www.science.org/doi/abs/ 10.1126/science.1105809. (Cited in pages 7 and 22.)
Sanchez, P., Voisey, J. P., Xia, T., Watson, H. I., O'Neil, A. Q., and Tsaftaris, S. A. Causal machine learning for healthcare and precision medicine. Royal Society Open Science, 9(8):220638, 2022. (Cited in page 8.)
Sanchez-Mart ´ ´ın, P., Rateike, M., and Valera, I. VACA:
Designing Variational Graph Autoencoders for Causal queries. In Thirty-Sixth AAAI Conference on Artificial Intelligence, AAAI 2022, Thirty-Fourth Conference on Innovative Applications of Artificial Intelligence, IAAI 2022, The Twelveth Symposium on Educational Advances in Artificial Intelligence, EAAI 2022 Virtual Event, February 22 - March 1, 2022, pp. 8159–8168. AAAI Press, 2022. URL https://ojs.aaai.org/index.php/A AAI/article/view/20789. (Cited in page 2.)
Schafer, J. and Strimmer, K. A shrinkage approach to large- ¨
scale covariance matrix estimation and implications for functional genomics. *Statistical applications in genetics* and molecular biology, 4(1), 2005. (Cited in pages 1, 2, 7, and 23.)
Scutari, M. Learning Bayesian Networks with the bnlearn R Package. *Journal of Statistical Software*, 35(3):1–22, 2010. doi: 10.18637/jss.v035.i03. (Cited in pages 22 and 23.)
Spirtes, P., Glymour, C., and Scheines, R. Causation, prediction, and search. MIT press, 2001. (Cited in page 17.)
Tchetgen, E. J. T., Ying, A., Cui, Y., Shi, X., and Miao, W. An introduction to proximal causal learning. *ArXiv* preprint, abs/2009.10982, 2020. URL https://arxi v.org/abs/2009.10982. (Cited in page 29.)
Vahdat, A. and Kautz, J. NVAE: A Deep Hierarchical Variational autoencoder. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), *Advances* in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/p aper/2020/hash/e3b21256183cf7c2c7a66 be163579d37-Abstract.html. (Cited in page 6.)
Varian, H. R. Causal inference in economics and marketing.

Proceedings of the National Academy of Sciences, 113 (27):7310–7315, 2016. (Cited in page 1.)
von Kugelgen, J., Sharma, Y., Gresele, L., Brendel, W., ¨
Scholkopf, B., Besserve, M., and Locatello, F. Self- ¨ Supervised Learning with Data Augmentations Provably Isolates content from style. In Ranzato, M., Beygelzimer, A., Dauphin, Y. N., Liang, P., and Vaughan, J. W. (eds.), Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pp. 16451–16467, 2021. URL
https://proceedings.neurips.cc/paper /2021/hash/8929c70f8d710e412d38da624 b21c3c8-Abstract.html. (Cited in page 6.)
Wang, Y. and Blei, D. M. The blessings of multiple causes.

Journal of the American Statistical Association, 114(528): 1574–1596, 2019. (Cited in pages 3, 7, 23, 24, and 30.)
605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659 Wang, Y. and Blei, D. M. A Proxy Variable View of Shared confounding. In Meila, M. and Zhang, T. (eds.), Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pp. 10697–10707. PMLR, 2021. URL http://proceedings.mlr.press/v139/wan g21c.html. (Cited in pages 1, 2, 3, 5, 15, 16, 18, 23, and 30.)
Wang, Y., Blei, D. M., and Cunningham, J. P. Posterior Collapse and Latent Variable Non-identifiability. In Ranzato, M., Beygelzimer, A., Dauphin, Y. N., Liang, P., and Vaughan, J. W. (eds.), Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pp. 5443–5455, 2021. URL https://proceedings.neurips.cc/p aper/2021/hash/2b6921f2c64dee16ba21e bf17f3c2c92-Abstract.html. (Cited in page 6.)
Wightman, L. F. Lsac National Longitudinal Bar Passage Study. LSAC Research Report Series. 1998. (Cited in page 8.)
Winkler, C., Worrall, D., Hoogeboom, E., and Welling, M. Learning Likelihoods with Conditional Normalizing flows. *ArXiv preprint*, abs/1912.00042, 2019. URL ht tps://arxiv.org/abs/1912.00042. (Cited in pages 2, 3, and 4.)
Xi, Q. and Bloem-Reddy, B. Indeterminacy in Generative Models: Characterization and Strong identifiability. In Ruiz, F. J. R., Dy, J. G., and van de Meent, J. (eds.), International Conference on Artificial Intelligence and Statistics, 25-27 April 2023, Palau de Congressos, Valencia, Spain, volume 206 of Proceedings of Machine Learning Research, pp. 6912–6939. PMLR, 2023. URL https://proceedings.mlr.press/v206/x i23a.html. (Cited in page 5.)
Yang, M., Liu, F., Chen, Z., Shen, X., Hao, J., and Wang, J. Causalvae: Structured causal disentanglement in variational autoencoder. *ArXiv preprint*, abs/2004.08697, 2020. URL https://arxiv.org/abs/2004.0 8697. (Cited in page 2.)
Zecevi ˇ c, M., Dhami, D. S., Velivckovi ´ c, P., and Kersting, ´
K. Relating graph neural networks to structural causal models. *ArXiv preprint*, abs/2109.04173, 2021. URL https://arxiv.org/abs/2109.04173. (Cited in page 2.)
Zhao, S. and Heffernan, N. Estimating Individual Treatment Effect from Educational Studies with Residual Counterfactual Networks. *International Educational Data Mining* Society, 2017. (Cited in page 1.)
660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 689 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714

| Zheng, Y., Ng, I., and Zhang, K.                                                                                                                                                                                                                                                                                                                                                                                                                                 | On the Identifiability   |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|
| of Nonlinear ICA: Sparsity and beyond. In Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., and Oh, A. (eds.), Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper_files/p aper/2022/hash/6801fa3fd290229efc490 ee0cf1c5687-Abstract-Conference.html. (Cited in page 6.) |                          |

# Appendix

## Table Of Contents

715 716 717 718 719 720 721 722 723 724 725 726 727 728 729 730 731 732 733 734 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 752 753 754 755 756 757 758 759 760 761 762 763 764 765 766 767 768 769

| Table of Contents A Causal identifiability                                       | 15                                                     |    |    |
|----------------------------------------------------------------------------------|--------------------------------------------------------|----|----|
| A.1                                                                              | Model identifiability                                  | 15 |    |
| A.2                                                                              | Query identifiability                                  | 15 |    |
| A.3                                                                              | Counterfactual query identifiability                   | 20 |    |
| B                                                                                | Experimental details and additional results            | 21 |    |
| B.1                                                                              | Ablation study                                         | 21 |    |
| B.2                                                                              | Semi-synthetic Sachs' dataset                          | 22 |    |
| B.3                                                                              | Semi-synthetic Ecoli70 dataset                         |    | 23 |
| B.4                                                                              | Law school fairness use-case                           |    | 25 |
| C                                                                                | Do-operator                                            | 27 |    |
| C.1                                                                              | Do-operator in causal normalizing flows                |    | 27 |
| C.2                                                                              | Do-operator in interventional distributions with Decaf |    | 28 |
| C.3                                                                              | Do-operator in counterfactuals with Decaf              |    | 28 |
| D Additional details on related work of causal inference with hidden confounders | 29                                                     |    |    |
| E                                                                                | Algorithms for causal query identification             | 30 |    |
| E.1                                                                              | Pipeline for using Decaf                               | 31 |    |

## A **Causal Identifiability** A.1 **Model Identifiability**

In this section, we briefly discuss the identifiability of those variables that are indirectly confounded by z or not confounded at all, i.e., of those variables that are not children of any hidden confounder. As we discuss now, we can reduce our SCM to a conditional SCM that only models these variables, recovering the identifiability guarantees from Javaloy et al. (2023). To prove model identifiability, we resort to what we call the induced conditional SCM, which intuitively represents the original SCM where we restrict our view to these variables, and assume the rest of the variables are given.

Definition 4 (Induced conditional SCM). Given a SCM M = (f, Pu, Pz), and a subset of observed variables x
′ ⊂ x, we define the induced conditional SCM of M *given* x
′, denoted by M|x′ , to the SCM result of having observed x
′, and where causal generators and exogenous variables are restricted to only those components associated with the rest of variables.

•
x1
•
x2 x1 z x2 x3 x5 x3 x5 x6 •
x7 x8 x6 x7 x8
(a) Confounded SCM.
Figure 10: Example of: (a) a confounded SCM M; and (b) its induced conditional counterpart, M|x′ , when the children of the hidden confounder are observed and fixed. Note that M|x′ has no hidden confounding.

We provide a visual depiction of this idea in Fig. 10. Using this definition, we can observe that, if we were to condition of the children of the hidden confounder, we would be left with a (conditional) unconfounded SCM, as the influence of the hidden confounder has been completely blocked by conditioning on its children. Now, if we have two models that perfectly match their marginal distributions, this means that they perfectly match their induced conditional SCM, no matter which value we observed for ch(z), and we can thus leverage existing results from Javaloy et al. (2023) for unconfounded SCMs.

Corollary A.1. *Assume that we have two SCMs* M := (f, Pu, Pz) and M˜ := (˜f, Pu˜ , Pz˜) *that are compatible, i.e., they* induce the same causal graph, and which coincide in their marginal distributions, p(x) = ˜p(x)*. Then, both SCMs, restricted* to every variable other than ch(z), are equal up to an element-wise transformation of the exogenous distributions. Proof. The proof follows almost directly from (Javaloy et al., 2023, Theorem 1). First, note that the two induced conditional SCMs are no longer influenced by z once that we have observed a specific realization of ch(z), so that we can drop z from their structure, i.e., we can denote them by M|ch(z) = (f|ch(z), Pu|ch(z)) and M˜|ch(z) = (˜f|ch(z), Pu˜|ch(z)). To ease notation, let us call x
∁
:= x \ ch(z) the variables that are not children of z.

Next, note that for almost every realization of ch(z), we have that p(x
∁
| ch(z)) = ˜p(x
∁
| ch(z)) since p(x) = ˜p(x) by assumption and p(x) = p(x
∁
| ch(z))p(ch(z)). As a result, for each realization of ch(z) we can apply Theorem 1 of Javaloy et al. (2023), which yields that the two induced conditional SCMs are equal up to an element-wise transformation of the exogenous distribution.

Finally, since the causal generators and exogenous distributions of the induced SCMs are, for almost every ch(z), identical to their counterparts in the original SCMs (as we have just discarded those components associated with ch(z)), we get that the elements in the two SCMs associated with every variable except those in ch(z) are identical up to said (possibly ch(z)-dependent) transformation.

## A.2 **Query Identifiability**

We now prove the identifiability of the causal queries considered in the main text. To this end, one key property that we will use in the following is that of completeness (see, e.g., the work of Wang & Blei
(2021)). Intuitively, we say that a random variable z is complete given another random variable n if "any infinitesimal change in z is accompanied by variability in n" (Miao et al., 2023), yielding enough information to recover the posterior distribution of z. This concept is similar in spirit to that of variability in the case of discrete random variables (Nasr-Esfahany et al., 2023). In practice, completeness is more likely to be achieved the more proxies we measure (Andrews, 2011).

770 771 772 773 774 775 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 794 795 796 797 798 799 800 801 802 803 804 805 806 807 808 809 810 811 812 813 814 815 816 817 818 819 820 821 822 823 824 Definition 5 (Completeness). We say that a random variable z is complete given n for all c if, for any square-integrable function g(·) and almost all c,Rg(z, c)p(z | c, n) dz = 0 for almost all n, if and only if g(z, c) = 0 for almost all z.

825 826 827 828 829 830 831 832 833 834 835 836 837 838 839 840 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874 875 876 877 878 879 The following proposition is a generalization of the results previously presented by Miao et al. (2018) and Wang & Blei (2021), where we include an additional covariate c to the causal query, and make no implicit assumptions on the causal graph allowing, e.g., for the treatment and outcome variables to share some observed parents. However, note that c cannot be a collider (e.g., forming a subgraph of the form n → c ← y) as, otherwise, conditioning on it would make independent variables dependent (in the example, y and n), and the causal effect of t on y would not be identifiable (Peters et al., 2017).

Proposition A.2 (Query identifiability). *Assume that we have two SCMs* M := (f, Pu, Pz) and M˜ := (˜f, Pu˜ , Pz˜) *that are* compatible, i.e., they induce the same causal graph, and which coincide in their marginal distributions, p(x) = ˜p(x)*. Then,*
they compute the same causal query, p(y | do(t), c) = ˜p(y | do(t), c), where y, t, c ⊂ x*, if there exists two proxies* w, n ⊂ x and a variable b ⊂ x*, none of them overlapping nor containing variables from the previous subsets, such that:*
i) w is conditionally independent of (t, n) given b, z and c*. That is,* w ⊥⊥ (t, n) | b, z, c .

ii) n is conditionally independent of y *given* t, b, z and c*. That is,* y ⊥⊥ n | t, b, z, c .

iii) (b, z) forms a valid adjustment set for the query p(y | do(t), c). That is, given c, they are independent of t after severing any incoming edges to it, do(t) ⊥⊥ (b, z) | c , and they block every backdoor path from t to y.

iv) z is complete given n for all t, b, and c, v) z˜ is complete given w for all b and c, and the following regularity conditions also hold:
vi) RR p˜(z˜ | w, b, c)˜p(w | z˜, b, c) dz˜ dw < ∞ for all b, c, and vii) Rp˜(y | t, b, z˜, c)
2p˜(z˜ | b, c) dz˜ < ∞ for all t, b, and c.

$$\tilde{p}({\bf y}\mid{\bf t},{\bf b},\tilde{\bf z},{\bf c})=\int\tilde{h}({\bf y},{\bf t},{\bf b},{\bf w},{\bf c})\tilde{p}({\bf w}\mid{\bf b},\tilde{\bf z},{\bf c})\,{\mathrm{d}}{\bf w}\,,$$

$$(10)^{\frac{1}{2}}$$
$$[e q u a l\;m a r g i n a l s]$$
Zh˜(y, t, b, w, c)˜p(w | b, z˜, c) dw , (10)
$$p(\mathbf{y}\mid\mathbf{t},\mathbf{b},\mathbf{n},\mathbf{c})={\bar{p}}(\mathbf{y}\mid\mathbf{t},\mathbf{b},\mathbf{n},\mathbf{c})$$
p(y | t, b, n, c) = ˜p(y | t, b, n, c) [*equal marginals*] (11)
$$\int{\bar{p}}(\mathbf{y}\mid\mathbf{t},\mathbf{b},\mathbf{n},{\bar{\mathbf{z}}},\mathbf{c}){\bar{p}}({\bar{\mathbf{z}}}\mid\mathbf{t},\mathbf{b},\mathbf{n},\mathbf{c})\,\mathrm{d}{\bar{\mathbf{z}}}$$
$\overline{\phantom{\rule{0.000pt}{0ex}}}$
Zp˜(y | t, b, n, z˜, c)˜p(z˜ | t, b, n, c) dz˜ [*augment with* z˜] (12)
$$=\int\bar{h}(\mathbf{y},\mathbf{t},\mathbf{b},\mathbf{w},\mathbf{c})p(\mathbf{w}\mid\mathbf{t},\mathbf{b},\mathbf{n},\mathbf{c})\,\mathrm{d}\mathbf{w}\,.$$
Zh˜(y, t, b, w, c)p(w | t, b, n, c) dw . [*equal marginals*] (16)
= =
Z Z h˜(y, t, b, w, c)˜p(w | b, z˜, c)˜p(z˜ | t, b, n, c) dz˜ dw [plug *Eq. 10*] (14)
Z Z h˜(y, t, b, w, c)˜p(w | b, z˜, t, n, c)˜p(z˜ | t, b, n, c) dz˜ dw [*assumption* i)] (15)
$\downarrow$
$$=\int{\bar{p}}(\mathbf{y}\mid\mathbf{t},\mathbf{b},{\bar{\mathbf{z}}},\mathbf{c}){\bar{p}}({\bar{\mathbf{z}}}\mid\mathbf{t},\mathbf{b},\mathbf{n},\mathbf{c})\,\mathrm{d}{\bar{\mathbf{z}}}$$
Zp˜(y | t, b, z˜, c)˜p(z˜ | t, b, n, c) dz˜ [assumption ii)] (13)
Similarly, we can relate the expression for the interventional distribution of both models:

$${\tilde{p}}(\mathbf{y}\mid\mathbf{do}(\mathbf{t}),\mathbf{c})=\int{\tilde{p}}(\mathbf{y}\mid\mathbf{do}(\mathbf{t}),\mathbf{b},{\tilde{\mathbf{z}}},\mathbf{c}){\tilde{p}}(\mathbf{b},{\tilde{\mathbf{z}}}\mid\mathbf{c})\,\mathbf{db}\,\mathbf{d}{\tilde{\mathbf{z}}}$$
Zp˜(y | do(t), b, z˜, c)˜p(b, z˜ | c) db dz˜ [augment and assumption *iii)*] (17)
$$=\int\tilde{p}(\mathbf{y}\mid\mathbf{t},\mathbf{b},\tilde{\mathbf{z}},\mathbf{c})\tilde{p}(\mathbf{b},\tilde{\mathbf{z}}\mid\mathbf{c})\,\mathrm{d}\mathbf{b}\,\mathrm{d}\tilde{\mathbf{z}}$$
Zp˜(y | t, b, z˜, c)˜p(b, z˜ | c) db dz˜ [*backdoor criterion*] (18)
$$(111)$$
$$[a u g m e n t\ w i t h\ \bar{\bf z}]$$
$$(12)^{\frac{1}{2}}$$
$$[a s s u m p t i o n\,i i)]$$
$$(13)^{\frac{1}{2}}$$
$$[p l u g\;E q.\;I O]$$
$$(14)$$

$$[a s s u m p t i o n\;i)]$$
$$[e q u a l\,m a r g i n a l s]$$
$$[a u g m e n t\ a n d\ a s s u m p i o n\ i u i o n\ i u i o n\ i u i o n]$$
$$(17)$$

$$[b a c k d o o r\;c r i t e r i o n]$$
$$(18)^{\frac{1}{2}}$$

since assumption vi) ensures that the conditional expectation operator is compact (Carrasco et al., 2007), assumption v) that all square-integrable functions are in the image of the operator (i.e., the operator is surjective), and assumption **vii)** that p˜(y | t, b, z˜, c) is indeed part of the image.

We can show that h˜ also solves a similar integral equation, this time over the other SCM, M, as follows:
Proof. First, note that the first three independence assumptions hold for both models, M and M˜ , as they induce the same causal graph. Following the same arguments as Miao et al. (2018, Proposition 1), we have that assumptions v), vi), and **vii)**
guarantee the existence of a function h˜ such that it solves the integral equation over M˜ ,

$$[p l u g\;E q.\;I O]$$
$$[e q u a l\,m a r g i n a l s]$$
$$(20)$$
$$[E q.\ I\theta]$$
$$(22)$$
$$(23)$$

where the last equality is a consequence of Eq. 16 as we will show now. More specifically, we have that

$$p(\mathbf{y}\mid\mathbf{t},\mathbf{b},\mathbf{n},\mathbf{c})=\int{\bar{h}}(\mathbf{y},\mathbf{t},\mathbf{b},\mathbf{w},\mathbf{c})p(\mathbf{w}\mid\mathbf{t},\mathbf{b},\mathbf{n},\mathbf{c})\,\mathrm{d}\mathbf{w}$$
Zh˜(y, t, b, w, c)p(w | t, b, n, c) dw [*Eq. 16*] (22)
Similarly, we have that which justifies the last equality in Eq. 21.

$$=\int\tilde{h}(\mathbf{y},\mathbf{t},\mathbf{b},\mathbf{w},\mathbf{c})p(\mathbf{b},\mathbf{w}\mid\mathbf{c})\,\mathrm{d}\mathbf{b}\,\mathrm{d}\mathbf{w}\,,$$
Zh˜(y, t, b, w, c)p(b, w | c) db dw , [*equal marginals*] (32)
=
Z Z h˜(y, t, b, w, c)p(w | b, z, c)p(b, z | c) db dz dw [*Eq. 28*] (31)
$$(26)$$
$$(27)^{\frac{1}{2}}$$
$$(28)^{\frac{1}{2}}$$
$$[a u g m e n t\ a n d\ a s s u m p t i o n\ i u i)]$$
$$(29)$$
$$[b a c k d o o r\,c r i t e r i o n]$$
$$(30)^{\frac{1}{2}}$$
$$[E q.\ 28]$$
$$(31)$$
$$[e q u a l\;m a r g i n a l s]$$

n z w t y b
Figure 11: Example for which Prop. A.2 applies, and where b is not the empty set.

$$p(\mathbf{y}\mid\mathsf{t},\mathbf{b},\mathbf{n},\mathbf{c})=\int p(\mathbf{y}\mid\mathsf{t},\mathbf{b},\mathbf{n},\mathbf{z},\mathbf{c})p(\mathbf{z}\mid\mathsf{t},\mathbf{b},\mathbf{n},\mathbf{c})\,\mathrm{d}\mathbf{z}$$ [_augment with_ $$\mathbf{z}$$ ] $$=\int p(\mathbf{y}\mid\mathsf{t},\mathbf{b},\mathbf{z},\mathbf{c})p(\mathbf{z}\mid\mathsf{t},\mathbf{b},\mathbf{n},\mathbf{c})\,\mathrm{d}\mathbf{z}\,.$$ [_assumption_ $$\mathsf{ii}$$ ]
$$[a u g m e n t\;w i t h\;\mathbf{z}]$$
$$(24)$$
$$(25)$$
$$0=\iint\left\{p(\mathbf{y}\mid\mathbf{t},\mathbf{b},\mathbf{z},\mathbf{c})-\int\tilde{h}(\mathbf{y},\mathbf{t},\mathbf{b},\mathbf{w},\mathbf{c})p(\mathbf{w}\mid\mathbf{b},\mathbf{z},\mathbf{c})\,\mathrm{d}\mathbf{w}\right\}p(\mathbf{z}\mid\mathbf{t},\mathbf{b},\mathbf{n},\mathbf{c})\,\mathrm{d}\mathbf{z}\,,$$

which, due to assumption iv), implies that

$$p(\mathbf{y}\mid\mathbf{t},\mathbf{b},\mathbf{z},\mathbf{c})\ {\stackrel{\mathrm{a.e.}}{=}}\ \int{\bar{h}}(\mathbf{y},\mathbf{t},\mathbf{b},\mathbf{w},\mathbf{c})p(\mathbf{w}\mid\mathbf{b},\mathbf{z},\mathbf{c})\,\mathrm{d}\mathbf{w}\ .$$
Zh˜(y, t, b, w, c)p(w | b, z, c) dw . (28)
Finally, putting all together we see that we can write the interventional distribution of the original model using h˜,

lily, putting all together we see that we can write the interventional distribution  $$p(\mathbf{y}\mid\mathbf{do}(\mathbf{t}),\mathbf{c})=\iint p(\mathbf{y}\mid\mathbf{do}(\mathbf{t}),\mathbf{b},\mathbf{z},\mathbf{c})p(\mathbf{b},\mathbf{z}\mid\mathbf{c})\,\mathrm{d}\mathbf{b}\,\mathrm{d}\mathbf{z}$$ $$=\iint p(\mathbf{y}\mid\mathbf{t},\mathbf{b},\mathbf{z},\mathbf{c})p(\mathbf{b},\mathbf{z}\mid\mathbf{c})\,\mathrm{d}\mathbf{b}\,\mathrm{d}\mathbf{z}$$
Z Z p(y | do(t), b, z, c)p(b, z | c) db dz [augment and assumption *iii)*] (29)
Z Z p(y | t, b, z, c)p(b, z | c) db dz [*backdoor criterion*] (30)
Using a causal graph similar to the one presented by Miao et al. (2018), we now provide some intuition on the semantics of each random variable in Prop. A.2. More specifically, consider the causal graph that we depict in Fig. 11, and say that we want to identify the causal query p(y | do(t)) (that is, the same query as in Prop. A.2 but with c = ∅). As it is common in the causal inference literature (Peters et al., 2017; Spirtes et al., 2001), t and y represent the treatment and outcome random variables. More specific to Prop. A.2 are n and w. The variable w is a proxy variable whose role is that of distinguishing the information from z and other variables, to reconstruct the information of z and block the backdoor path that z would usually block. Similarly, the variable n is another proxy variable which, in this case, serves the purpose of verifying that the substitute formed 880 881 882 883 884 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934

= =
Z Z h˜(y, t, b, w, c)˜p(w | b, z˜, c)˜p(b, z˜ | c) db dw dz˜ [plug *Eq. 10*] (19)
Zh˜(y, t, b, w, c)p(b, w | c) db dw [*equal marginals*] (20)
= p(y | do(t), c), (21)
= =
Z Z h˜(y, t, b, w, c)p(w | b, z, t, n, c)p(z | t, b, n, c) dw dz , [*augment with* z] (23)
Z Z h˜(y, t, b, w, c)p(w | b, z, c)p(z | t, b, n, c) dw dz . [*assumption* i)] (24)
Now, equating both expressions we have that with w is indeed a good substitute. Finally, the variable b serves the purpose of blocking all the remaining backdoor paths that z may not block, so that we can apply the backdoor criterion.

Moreover, note that for all interventional queries we will let c be the empty set, similar to the results proved by Miao et al. (2018) and Wang & Blei (2021). We will consider cases when c is not empty later in App. A.3 to prove counterfactual identifiability. Note also that Prop. A.2 reduces to the existing results when we have that c = b = ∅ . Using this general proposition, we can now reason about causal identifiability in a wide range of scenarios, where t and y may or may not be directly caused by the hidden confounder, as we show in the following subsections.

A.2.1 UNCONFOUNDED CASE
First, we consider the case where neither t nor y are directly affected by the hidden confounder, i.e., z ∈/ ch(z). In this case, the proof can be simplified and drop the requirement of finding valid proxy variables.

Corollary A.3 (Unconfounded case). *Assume that we have two SCMs* M := (f, Pu, Pz) and M˜ := (˜f, Pu˜ , Pz˜) that are compatible, i.e., they induce the same causal graph, and which coincide in their marginal distributions, p(x) = ˜p(x). Assume that y, t ∈/ ch(z)*. Then,* p(y | do(t), c) = ˜p(y | do(t), c), where y, t, c ⊂ x . Proof. The proof follows directly by applying Prop. A.2 with the minimal subset b ⊂ pa(t) \ {c} that blocks all the backdoor paths, and by noticing that in this case there is no need to use the variables z and z˜. That is, we can go from Eq. 17 to Eq. 21 directly by using only b and the equal-marginals assumption:

$${\tilde{p}}(\mathbf{y}\mid\mathbf{do}(\mathbf{t}),\mathbf{c})=\int{\tilde{p}}(\mathbf{y}\mid\mathbf{do}(\mathbf{t}),\mathbf{b},\mathbf{c}){\tilde{p}}(\mathbf{b}\mid\mathbf{c})\,\mathrm{d}\mathbf{b}$$
Zp˜(y | do(t), b, c)˜p(b | c) db (33)
935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 967 968 969 970 971 972 973 974 975 976 977 978 979 980 981 982 983 984 985 986 987 988 989 Proof. The proof is identical to that of Cor. A.3.

z Front-door example. While the proof above is trivial given the previous results, it is worth stressing that for them to hold it is necessary to model the hidden confounder as we do in this work with the proposed Decaf, and that other approaches may not work for all cases. As an example, consider the SCM depicted in Fig. 12, where we have that the outcome is directly confounded by z, while t is not. In this case, a Decaf should be able to identify the true causal query p(y | do(t)), using z˜ to model the influence of b

b t y
Figure 12: Textbook example of a front-door in a SCM.

Even though we can leverage and simplify Prop. A.2 as shown above, it is worth remarking that, for this particular case, the model identifiability results described in App. A.1 are stronger, as it provides results on the identifiability of the causal generators and exogenous distributions, and therefore of any causal query derived from them.

A.2.2 FULLY CONFOUNDED CASE
In the case where both variables are directly confounded by z, we cannot do much but to see whether we can apply Prop. A.2 with c = ∅ and a valid b. If we manage to find two proxies w and n that hold the independence conditions from Prop. A.2 and that change the posterior of z enough, then we can use the proposition to ensure the identifiability of the query. Otherwise, the query is not identifiable and the model might or might not estimate the query correctly.

A.2.3 CONFOUNDED OUTCOME CASE
For the case where only the outcome random variable is directly affected by the hidden variable, we can apply a similar reasoning as we did in the case with no direct confounding, although this time we cannot leverage the model identifiability results from Javaloy et al. (2023). More specifically:
Corollary A.4 (Confounded-outcome case). *Assume that we have two SCMs* M := (f, Pu, Pz) and M˜ := (˜f, Pu˜ , Pz˜) *that* are compatible, i.e., they induce the same causal graph, and which coincide in their marginal distributions, p(x) = ˜p(x). Assume that t ∈/ ch(z)*. Then,* p(y | do(t), c) = ˜p(y | do(t), c), where y, t, c ⊂ x .

$J$  $=\int p(\mathbf{y}\mid\mathbf{t},\mathbf{b},\mathbf{c})p(\mathbf{b}\mid\mathbf{c})\,\mathrm{d}\mathbf{b}$  $=p(\mathbf{y}\mid\mathrm{d}\mathbf{o}(\mathbf{t}),\mathbf{c})$.  
$\eqref{eq:walpha}$. 
Zp(y | t, b, c)p(b | c) db (35)
= p(y | do(t), c). (36)
$$(33)^{\frac{1}{2}}$$
$$=\int{\tilde{p}}(\mathbf{y}\mid\mathbf{t},\mathbf{b},\mathbf{c}){\tilde{p}}(\mathbf{b}\mid\mathbf{c})\,\mathrm{d}\mathbf{b}$$
$\eqref{eq:walpha}$. 
Zp˜(y | t, b, c)˜p(b | c) db (34)
onto y that is not explained through t. Other models that do not model z (e.g., an unaware causal normalizing flow (Javaloy et al., 2023)), would not be able to match the observed marginal likelihood as they assume that y ⊥⊥ b | t yet we know that y ̸⊥⊥ b | t in the true model. Even more, with those models we would have that p(y | do(t)) = p(y | t) which is clearly false by just looking at Fig. 12. To be even more explicit, in this case we would have a factorization of the form

$$(37)$$
$$\tilde{p}({\bf b},{\bf t},{\bf y},\tilde{\bf z})=\tilde{p}(\tilde{\bf z})\tilde{p}({\bf b}\mid\tilde{\bf z})\tilde{p}({\bf t}\mid{\bf b})\tilde{p}({\bf y}\mid{\bf t},\tilde{\bf z})\,.$$
$$[{\bf b}\,f o r m s\,a\,\,v a l i d\,\,a d j u s t m e n t\,\,s e t]$$
[_latent factorization and equal marginals_]  [_causal graph factorization in Eq. 37_]
$$[m a r g i n a l i z e{\mathrm{~b}}]$$
p˜(b, t, y, z˜) = ˜p(z˜)˜p(b | z˜)˜p(t | b)˜p(y | t, z˜). (37)
Then, the estimated interventional distribution that a Decaf estimates as Rp˜(y | t, z˜) dz˜ equals the true one:

$$p(\mathbf{y}\mid\mathbf{do(t)})=\int p(\mathbf{y}\mid\mathbf{t},\mathbf{b})p(\mathbf{b})\,\mathrm{d}\mathbf{b}$$
$|(1)\cap$. 
$|(1)\uparrow$. 
Zp(y | t, b)p(b) db [b *forms a valid adjustment set*] (38)
Remarkably, the identification of p(y | do(t)) allows us to solve also the query p(y | do(b)) leveraging the frontdoor criterion (Peters et al., 2017).

A.2.4 CONFOUNDED TREATMENT CASE

= Z Zp˜(y | t, b, z˜)˜p(z˜ | t, b) dz˜ = =
p˜(b) db [*latent factorization and equal marginals*] (39)
Z Z p˜(y | t, z˜)˜p(z˜ | b)˜p(b) db dz˜ [causal graph factorization in *Eq. 37*] (40)
Zp˜(y | t, z˜)˜p(z˜) dz˜ [*marginalize* b] (41)
= ˜p(y | do(t)). (42)
$$p(\mathbf{y}\mid\mathbf{do}(\mathbf{b}))=\int p(\mathbf{t}\mid\mathbf{b})p(\mathbf{y}\mid\mathbf{do}(\mathbf{t}))\,\mathrm{d}\mathbf{t}$$
$$=\int p(t\mid\mathbf{b})p(\mathbf{y}\mid\mathrm{do}(t))\,\mathrm{d}t$$ [_frontdoor criterion_] $$=\int\tilde{p}(t\mid\mathbf{b})\int\tilde{p}(\mathbf{y}\mid t,\tilde{\mathbf{z}})\tilde{p}(\tilde{\mathbf{z}})\,\mathrm{d}\tilde{\mathbf{z}}\,\mathrm{d}t$$ [_plug in Eq. 41 and equal marginals_] $$=\tilde{p}(\mathbf{y}\mid\mathrm{do}(\mathbf{b}))$$
990 991 992 993 994 995 996 997 998 999 1000
$$|\left(0\right)|$$
1002
1003
1004 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024
1025
1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 1036 1037 1038 1039 1040 1041 1042 1043 1044 When only the treatment variable t is directly confounded, we can find two different scenarios: if we are able to find a valid adjustment set b blocking all confounded paths, in which case we can reason just as in the other partially confounded case, and otherwise, where we rely on the identifiability with respect to this invalid adjustment set. For example, if it happens to be a parent of y which is directly caused by the treatment variable t and the hidden confounder z as in Fig. 13, we cannot find a valid adjustment set for the causal query, but an invalid one may still serve us if we can identify the same query with the adjustment set as outcome. Corollary A.5 (Confounded-treatment case). *Assume that we have two compatibleSCMs* M := (f, Pu, Pz) and M˜ := (˜f, Pu˜ , Pz˜)*, i.e., they induce the same causal graph, and which* coincide in their marginal distributions, p(x) = ˜p(x)*. Assume also that* y ∈/ ch(z)*. Then,* p(y | do(t), c) = ˜p(y | do(t), c), where y, t, c ⊂ x if there exists a subset b ⊂ x not containing variables from the previous subsets, such that one of the following two conditions are true:
i) b *forms a valid adjustment set for the query* p(y | do(t), c).

ii) b forms an invalid adjustment set for the query p(y | do(t), c) but the query p(b | do(t), c) *is identifiable. That is,* b blocks all the backdoor paths, and p(b | do(t), c) = ˜p(b | do(t), c).

$$(38)^{\frac{1}{2}}$$
$$(41)^{\frac{1}{2}}$$
$$(42)^{\frac{1}{2}}$$
$$(43)$$
$$(44)^{\frac{1}{2}}$$
$$(45)^{\frac{1}{2}}$$

z xi t y
$\eqref{eq:walpha}$. 
Figure 13: Case where no valid adjustment set can be found.

$${\tilde{p}}(\mathbf{y}\mid\mathbf{do}(\mathbf{t}),\mathbf{c})=\int{\tilde{p}}(\mathbf{y}\mid\mathbf{do}(\mathbf{t}),\mathbf{b},\mathbf{c}){\tilde{p}}(\mathbf{b}\mid\mathbf{do}(\mathbf{t}),\mathbf{c})\,\mathrm{d}\mathbf{b}$$
Zp˜(y | do(t), b, c)˜p(b | do(t), c) db (46)
19 Proof. If condition i) holds, then we have a valid adjustment set, and the proof is identical to that of Cor. A.3. Otherwise, if condition ii) holds, we have that the interventional query on y equals the observational query when conditioned on b, but that now b is not independent of do(t), i.e.,

n z w t y
(a)
un ut z uy uw n f wf t f y f
(b)
t cf y cf ôôô n cf wcf un ut z uy uw n f wf t f y f
$$(47)^{\frac{1}{2}}$$
$\left(48\right)^2$
$\eqref{eq:walpha}$. 
(c)
Figure 14: Example of the transition from (a) the regular depiction of a (confounded) SCM, to (b) an explicit SCM where the exogenous variables are drawn, and (c) a counterfactual twin SCM where the data-generating process is replicated in the
"factual and counterfactual worlds". Besides, figure (c) also depicts which nodes are observed and which edges are severed, in order to compute a counterfactual query of the type p(y cf | do(t cf), x f).

## A.3 **Counterfactual Query Identifiability**

where we needed to use that the query p(b | do(t), c) is identifiable in the third equality.

In this section, we show that counterfactual query identifiability is a direct result of the interventional query identifiability from the previous section. In order to formally define counterfactuals, in this section we introduce the concept of counterfactual SCMs in a somewhat novel way. Namely, we combine the concepts of twin networks from Pearl (2009) (which replicates the data-generating process) and that of counterfactual SCMs from Peters et al. (2017) (which defines a counterfactual *prior* to the intervention). Definition 6 (Counterfactual twin SCM). Given a SCM M = (f, Pu, Pz), we define its counterfactual twin SCM as a SCM Mcf where all structural equations are duplicated, and the exogenous noise is shared across replications, and where additionally one of the halves is observed ("the factual world"), and the other half is unobserved ("the counterfactual world").

We provide in Fig. 14 a more intuitive depiction on the construction of these counterfactual twin networks. From this definition, one can recover the counterfactual SCM defined by Peters et al. (2017) by just focusing on the replicated part of the counterfactual twin network, and conditioning the exogenous noise and hidden confounder on the observed half, i.e., (f, Pu | xf , Pz | xf). Similarly, one can compute the usual counterfactual query by performing an intervention on the counterfactual twin network, i.e., by replacing the intervened equations by the constant intervened value, and computing the query conditioned on the factual variables, p(y cf | do(t cf), x f). This is visually represented in Fig. 14c.

In order to prove query identifiability in the counterfactual setting, we need to use the following technical result regarding the completeness of a random variable: Lemma A.6. If a random variable z is complete given n for all b, as given by Def. 5, then it is complete given n for all b and c, where c *is another continuous random variable.* Proof. We prove this result by contradiction. Assume that the result does not hold, then there must exist a non-zero measure subset of the space of b×cfor which there exists a square-integrable function g(·) such that Rg(z, b, c)p(z | b, c, n) dz = 0 for almost all n, but g(z, b, c) ̸= 0 for almost all z.

Since this subset has positive measure, there must contain an ε-ball within. If we now focus on the b-projection of this ball where we fix c to its value on the centre, we have that it is a subset of non-zero measure in the space of b (as otherwise it would be zero-measure in the Cartesian-product measure), where the function g(·, c) breaks our initial assumption of the completeness of z. Thus, we reach a contradiction.

1045 1046 1047 1048 1049 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079 1080 1081 1082 1083 1084 1085 1086 1087 1088 1089 1090 1091 1092 1093 1094 1095 1096 1097 1098 1099

$$\int{\bar{p}}(\mathbf{y}\mid\mathbf{t},\mathbf{b},\mathbf{c}){\bar{p}}(\mathbf{b}\mid\operatorname{do}(\mathbf{t}),\mathbf{c})\operatorname{d}\mathbf{b}$$
$\overline{\phantom{\rule{0.000pt}{0ex}}}$
Zp˜(y | t, b, c)˜p(b | do(t), c) db (47)
$$\begin{array}{c}{{\int\,p(\mathbf{y}\mid\mathbf{t},\mathbf{b},\mathbf{c})p(\mathbf{b}\mid\mathrm{do}(\mathbf{t}),\mathbf{c})\,\mathrm{d}\mathbf{b}}}\\ {{=p(\mathbf{y}\mid\mathrm{do}(\mathbf{t}),\mathbf{c})\,,}}\end{array}$$
Zp(y | t, b, c)p(b | do(t), c) db (48)
= p(y | do(t), c), (49)