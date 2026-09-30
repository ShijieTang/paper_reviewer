# 

Ermis Soumalias * 1 2 **Jakob Heiss** * 2 3 4 **Jakob Weissteiner** 1 2 5 **Sven Seuken** 1 2

## Abstract

We study the design of *iterative combinatorial* auctions (ICAs). The main challenge in this domain is that the bundle space grows exponentially in the number of items. To address this, recent work has proposed machine learning (ML)-based preference elicitation algorithms that aim to elicit only the most critical information from bidders to maximize efficiency. However, while the SOTA
ML-based algorithms elicit bidders' preferences via *value queries*, ICAs that are used in practice elicit information via *demand queries*. In this paper, we introduce a novel ML algorithm that provably makes use of the full information from both value and demand queries, and we show via experiments that combining both query types results in significantly better learning performance in practice. Building on these insights, we present MLHCA, a new ML-powered auction that uses value and demand queries. MLHCA significantly outperforms the previous SOTA, reducing efficiency loss by up to a factor 10, with up to 58% fewer queries. Thus, MLHCA achieves large efficiency improvements while also reducing bidders' cognitive load, establishing a new benchmark for both practicability and efficiency. Our code is available at https://github.com/ marketdesignresearch/MLHCA.

## 1. Introduction

pression of value. CAs have enjoyed widespread adoption in practice, with their applications ranging from allocating spectrum licenses (Cramton, 2013) to TV ad slots (Goetzendorff et al., 2015) and airport landing/take-off slots (Rassenti et al., 1982). The key challenge in CAs is that the bundle space grows exponentially in the number of items, making it impossible for bidders to report their full value function in all but the smallest domains. Moreover, Nisan & Segal (2006) showed that, for arbitrary value functions, CAs require an exponential number of bids to guarantee full efficiency. Thus, practical CA mechanisms cannot provide efficiency guarantees in real world settings with more than a modest number of items. Instead, the focus has shifted towards iterative combinatorial auctions (ICAs), where bidders interact with the auctioneer over a series of rounds, providing only a limited
(i.e., practically feasible) amount of information, with the aim to maximize the efficiency of the final allocation. The most established ICA following this interaction paradigm is the *combinatorial clock auction (CCA)* (Ausubel et al., 2006). Extensively used for allocating spectrum licenses, the CCA generated over USD 20 *billion* in revenue between 2012 and 2014 alone (Ausubel & Baranov, 2017). However, a key challenge for any ICA, including the CCA, is balancing *speed of convergence* with efficiency. Each bidding round involves significant computational costs and complex business modeling for participants (Kwasnica et al., 2005; Milgrom & Segal, 2017; Bichler et al., 2017), making faster convergence highly desirable. Large spectrum auctions conducted under the CCA format can require over 100 bidding rounds, prompting practitioners to adopt aggressive price update rules to reduce the number of rounds. For example, prices may be increased by up to 10% per round, but such approaches come at the expense of efficiency (Ausubel & Baranov, 2017). This trade-off highlights the ongoing challenge of designing ICAs that achieve both high efficiency and rapid convergence. Given the value of resources allocated in these auctions, even a one-percentage-point improvement in efficiency translates to welfare gains of hundreds of millions of dollars.

1

## 1.1. Ml-Powered Iterative Combinatorial Auctions

To tackle this challenge, researchers have explored using machine learning (ML) to enhance the efficiency of ICAs. The foundational works of Blum et al. (2004) and Lahaie & Parkes (2004) were the first to frame preference elicitation in CAs as a learning problem. More recently, Brero et al. (2018; 2021) and Weissteiner & Seuken (2020); Weissteiner et al. (2022b;a; 2023) introduced ML-powered ICAs. Central to these approaches is an ML-based preference elicitation algorithm that trains an ML model on each bidder's value function to generate informative *value queries (VQs)*
(e.g., "What is your value for the bundle {*A, B*}?"), which iteratively refine the ML model of each bidder's values.1 Soumalias et al. (2024c) took a different approach. To increase the likelihood of their approach being adopted in practice, they introduced ML-CCA, an ML-powered auction that follows the established interaction paradigm of the CCA using demand queries. Building on earlier works by Brero & Lahaie (2018); Brero et al. (2019), their design iteratively trains individual ML models for each bidder using their previously answered *demand queries (DQs)* and then selects the next DQ with the highest clearing potential. Although ML-CCA marked a major step towards a practical ML-powered ICA and outperformed the baseline CCA used in real-world applications, it still faced two key shortcomings. First, it fell short of achieving the SOTA efficiency of the VQ-based ML-powered ICAs. Second, like the CCA, it relied on a very large number of supplementary round bids to enhance its efficiency, requiring bidders to decide on additional *value bids*—a cognitively demanding task.

We address these shortcomings by introducing the Machine Learning-powered Hybrid Combinatorial Auction
(MLHCA). Leveraging sophisticated DQ and VQ generation algorithms, MLHCA maintains the established interaction paradigm of the CCA while achieving unprecedented efficiency gains. MLHCA outperforms the previous SOTA
across all tested domains, reducing efficiency loss by up to a factor of ten. Based on the value of goods traded (Ausubel & Baranov, 2017), these efficiency improvements correspond to welfare gains of hundreds of millions of USD. At the same time, MLHCA significantly reduces the cognitive load on bidders: compared to BOCA, the previous SOTA, ML- HCA requires at least 42% fewer queries to achieve the same efficiency, and compared to ML-CCA, the SOTA auction following CCA's interaction paradigm, MLHCA requires at least 26% fewer queries. Moreover, unlike the CCA and ML-CCA, in MLHCA bidders do not need to decide which bundles to bid for in its VQ rounds, as the auction automatically suggests these bundles. Thus, MLHCA achieves 1From an optimization perspective, this can be viewed as a combinatorial Bayesian optimization problem.

unprecedented efficiency gains while significantly reducing bidders' cognitive load, establishing a new benchmark for both practicability and efficiency.

## 1.2. Our Contributions

We introduce the Machine Learning-powered Hybrid Combinatorial Auction (MLHCA), a practical ICA that achieves unprecedented efficiency and convergence speed. First, we establish a theoretical foundation and provide illustrative examples to demonstrate the advantages and limitations of DQs and VQs from an auction design perspective (Section 3). Then we develop a learning algorithm that effectively leverages both query types (Section 4). We provide strong experimental evidence of the learning benefits of combining both query types, as well as the advantages of starting an auction with DQs instead of VQs. We then integrate these auction and ML insights to design MLHCA, the first ICA to incorporate sophisticated DQ and VQ generation algorithms (Section 5). Simulations in realistic domains (Section 6) show that MLHCA significantly outperforms the previous SOTA, achieving unprecedented efficiency while also using fewer queries, thus setting a new benchmark for both efficiency and practicality.

## 1.3. Further Related Work

In the field of *automated mechanism design*, Dutting et al. ¨ (2015; 2019), Golowich et al. (2018) and Narasimhan et al. (2016) used ML to learn new mechanisms from data, while Cole & Roughgarden (2014); Morgenstern & Roughgarden (2015) and Balcan et al. (2023) bounded the sample complexity of learning approximately optimal mechanisms. In contrast to this line of prior work, in our design, the ML algorithm is part of the mechanism itself. Lahaie & Lubin (2019) suggest an adaptive price update rule that increases price expressivity as the rounds progress in order to improve efficiency and speed of convergence. Unlike that work, we aim to improve efficiency without increasing price expressivity, as that is not a popular interaction paradigm in practice, and can cause added cognitive load on the bidders.

Preference elicitation is also a key challenge in combinatorial allocation without money. Soumalias et al. (2024b)
introduce an ML-powered mechanism for course allocation that improves preference elicitation by asking comparison queries. See Appendix A.1 for further related work.

## 1.4. Practical Considerations And Incentives

MLHCA can be seen as a sophisticated modification of the CCA. In practice, many other considerations (beyond preference elicitation complexity and efficiency) are important. For example, Ausubel & Baranov (2017) discussed the vital role of well-designed *activity rules* to induce truthful bidding in the clock phase of the CCA. In Appendix B.3, we provide a detailed discussion of the most common activity rules used in the CCA, and we detail how MLHCA can also leverage these rules for the same goal. Additionally, in Appendix B.4, we prove that MLHCA can immediately detect if a bidder's reports are inconsistent. The payment rule used in the supplementary round of the CCA is also important for incentives. Cramton (2013) argued that the use of the VCG-nearest payment rule, while not strategyproof, induces good incentives in practice. Similar to the supplementary round of the CCA, the VQ-based phase of MLHCA is not strategyproof. However, in Appendix B.5, we argue that the VQ-based phase of MLHCA offers strong incentives in practice, and we show that, under two additional assumptions, truthful bidding is an ex-post Nash equilibrium (following the arguments from Brero et al. (2021) for the MLCA).

## 2. Preliminaries 2.1. Formal Model For Icas

We consider *multiset* CA domains with a set N = {1*, . . . , n*} of bidders and a set M = {1*, . . . , m*} of distinct items with corresponding *capacities*, i.e., number of available copies, c = (c1*, . . . , c*m) ∈ N
m. We denote by x ∈ X = {0, . . . , c1} × . . . × {0*, . . . , c*m} a bundle of items represented as a positive integer vector, where xj = k iff item j ∈ M is contained k-times in x. The bidders' true preferences over bundles are represented by their (private) value functions vi: X → R≥0, i ∈ N, i.e.,
vi(x) represents bidder i's value for bundle x ∈ X . We assume that viis nondecreasing and satisfies vi(0) = 0. We collect the value functions viin the vector v = (vi)i∈N .

By a = (a1, . . . , an) ∈ X n we denote an *allocation* of bundles to bidders, where aiis the bundle bidder i obtains. We denote the set of *feasible* allocations by F =a ∈ X n :Pi∈N aij ≤ cj , ∀j ∈ M	. We assume that bidders have *quasilinear utility functions* of the form ui(ai) = vi(ai) − πi where vi can be highly non-linear and πi ∈ R≥0 denotes the bidder's payment. This implies that the (true) *social welfare* V (a) of an allocation a is equal to the sum of all bidders' values Pi∈N vi(ai).

2 We let a
∗ ∈ arg maxa∈F V (a) denote a social-welfare maximizing, i.e., *efficient*, allocation. The *efficiency* of any allocation a ∈ F is determined as V (a)/V (a
∗).

An ICA *mechanism* defines how the bidders interact with the auctioneer and how the allocation and payments are determined. We consider ICAs that iteratively ask bidders both *demand queries* (DQs) and *value queries* (VQs). Definition 2.1 (Demand Query). In a (linear) demand query,

${}^{2}$Note that $V(a)=\sum_{i\in N}u_{i}(a_{i})+u_{\rm anticonver}(a)=\sum_{i\in N}\left(v_{i}(a_{i})-\pi_{i}\right)+\sum_{i\in N}\pi_{i}=\sum_{i\in N}v_{i}(a_{i})$.  
the auctioneer presents a vector of item prices p ∈ R
m ≥0 and each bidder i responds with her utility-maximizing bundle,

$$x_{i}^{*}(p)\in\operatorname*{arg\,max}_{x\in\mathcal{X}}\left\{v_{i}(x)-\langle p,x\rangle\right\}\,i\in N,\tag{1}$$

where ⟨·, ·⟩ denotes the Euclidean scalar product in R
m.

Definition 2.2 (Value Query). In a value query, the auctioneer presents to bidder i a bundle of items x and bidder i responds with her value at those prices, i.e., vi(x) ∈ R≥0. For bidder i ∈ N, we denote her K ∈ N elicited DQs as R
DQ
i = {(x
∗ i
(p r), pr)}
K
r=1 and her L ∈ N elicited VQs as R
VQ
i =x li
, vi(x li
)	L
l=1. Bidder i's reports are denoted as Ri = (R
DQ
i, RVQ
i). We collect the elicited reports of all bidders in the tuple R = (R1*, . . . , R*n).

In auctions using DQs, a key concept is the bidder's inferred value. This represents the maximum lower bound on a bidder's value for a bundle that the auctioneer can deduce from the bidder's reports, without assuming monotonicity. The inferred value is always weakly lower than the bidder's true value, with equality achieved if the bidder has answered the corresponding VQ for that bundle. Formally: Definition 2.3 (Inferred Value). Bidder i's inferred value for bundle x ∈ X given her reports Riis

  **for** **c** **and** $\varepsilon\in N$** given** **for** **l** **is**  $$\widetilde{v}_{i}(x;R_{i})=\left\{\begin{array}{ll}v_{i}(x),&\mbox{if}(x,v_{i}(x))\in R_{i}^{\rm VO},\\ \max\left\{\left\{\left\langle x,p^{r}\right\rangle:(x,p^{r})\in R_{i}^{\rm RO}\right\}\cup\left\{0\right\}\right\},\mbox{else}.\end{array}\right.\tag{2}$$
The ICA's final allocation a
∗(R) ∈ F and payments πi:= πi(R) ∈ R
n
≥0are computed based *only* on the *elicited* reports R. Concretely, a
∗(R) ∈ F is determined by solving the *Winner Determination Problem (WDP):*

$$a^{*}(R)\in\arg\max_{a\in{\cal F}}\sum_{i\in N}\tilde{v}_{i}(a_{i};R_{i}),\tag{3}$$

where Pi∈N vei(ai; Ri) is the allocation's inferred social welfare, a lower bound on its *social welfare* Pi∈N vi(ai).

## 2.2. Benchmark Icas

In this section, we briefly introduce the three main benchmark mechanisms considered in this paper. CCA The most established ICA is the Combinatorial Clock Auction (CCA) (Ausubel et al., 2006). The CCA consists of two phases. The initial *clock phase* proceeds in rounds. In each round r, the auctioneer sets anonymous (i.e., same prices for all bidders) item prices p r ∈ R
m
≥0, prompting each bidder to respond to a DQ, declaring her utility-maximizing bundle at p r. In the next round, the prices of over-demanded items are increased by a fixed percentage, until over-demand is eliminated. The second phase of the CCA, known as the *supplementary round*, allows bidders to report their *values* for additional bundles of their choice. The *clock bids raised heuristic* suggests that bidders report their values for all bundles they requested during the clock phase. The final allocation is determined by solving the WDP based on all reports as in Equation (3). ML-CCA The most efficient DQ-based ICA is the Machine Learning-powered Combinatorial Clock Auction (ML- CCA) (Soumalias et al., 2024c). ML-CCA has the same interaction paradigm as the CCA, but with a substantially more refined DQ-generation algorithm in its clock phase. In each round, an ML model is trained to estimate each bidder's value function based on previously submitted DQ responses. Then, the prices are not increased by a percentage like in the CCA, but instead a convex optimization problem determines the prices with the highest clearing potential. BOCA The SOTA ICA in terms of efficiency is the VQ- based Bayesian optimization-based combinatorial auction (BOCA) (Weissteiner et al., 2023). The main idea of BOCA is that in each round, the auctioneer creates an estimate of the upper confidence bound of the value function of each agent based on her past responses. Then, the auctioneer solves an ML-based WDP to find the feasible allocation with the highest upper bound on its estimated social welfare, and queries each agent her value for her bundle in that allocation. This allows the mechanism to balance between exploring and exploiting the bundle space.

## 2.3. Ml Framework

The ML models used by ML-CCA, and as basis for the construction of the confidence bound estimates in BOCA are monotone-value neural networks (MVNNs) Mθ: X → R
(Weissteiner et al., 2022a). MVNNs are a class of NNs specifically designed to represent *monotone combinatorial* valuations. MVNNs have also had success in combinatorial allocation domains without money, e.g., for course allocation (Soumalias et al., 2024b). Soumalias et al. (2024c) introduced *multiset MVNNs (mMVNNs)*, an extension of MVNNs that also incorporates at a structural level the information that some items in the auction are identical copies of each other. In this work, we instantiate our ML models using mMVNNs, and denote agent i's model as Mθ i: X → R.

Within this work, we will refer to all mMVNNs simply as MVNNs. We provide more details in Appendix C.

## 3. A Theoretical Framework For Effectively Combining Dqs And Vqs

This section develops a theoretical framework for effectively combining DQs and VQs. Proofs are deferred to Appendix D.

VQ-Based Approaches Rely on Cognitively Complex Random VQs. At the start of an ICA, no specific information about bidders' preferences is available, making it challenging to identify bundles relevant to them. Thus, most VQ-based auctions, including the SOTA approach (Weissteiner et al., 2023), begin by querying bidders about randomly selected bundles. However, in practice, answering VQs for such random bundles that do not align with the bidders' interests is cognitively demanding. In contrast, the most widely used ICAs in practice (e.g., the CCA) employ DQs, which are easier for bidders to answer effectively (Cramton, 2013). This key advantage highlights why relying exclusively on VQs is often impractical in real-world auctions. DQs Offer Superior Efficiency Gains in Initial Rounds. Even if bidders could easily answer random VQs, in Appendix D.1 we detail the significant advantages of DQs in the initial rounds of an ML-ICA, where DQs are providing more actionable and efficient information. This superior efficiency is formalized in the following proposition: Proposition 3.1. *The expected social welfare of an auction* that uses a single random demand query can be arbitrarily larger than that of an auction that uses any constant number (k ≪ 2 m*) of random value queries.*
Additionally, DQs can establish a proof of optimality, allowing the auction to terminate early (Proposition D.3). These theoretical insights are validated by our experimental results in Section 6. An auction initialized with DQs has up to 20% points higher efficiency after its initial queries compared to an auction initialized with VQs. VQs Offer Superior Efficiency Gains in Later Rounds. This raises the question: is it sufficient to rely exclusively on DQs? Theorem 3.2 proves that the answer is negative:
Theorem 3.2. For every ϵ > 0*, there exist infinitely many* instances of auctions for which no combination of DQs can achieve an efficiency above 50%+ϵ. This remains true even for infinite combinations of DQs and even if the bidders additionally report their true values for all bundles they requested in those DQs. Notably, Theorem 3.2 shows that this limitation persists even when supplementing the auction with the clock-bids raised heuristic. Without this heuristic, adding more DQs can even *decrease* efficiency. In Proposition 3.3 we prove that a single DQ can reduce the auction's efficiency arbitrarily close to 100%, whereas VQ-based auctions do not face this issue.

Proposition 3.3. *In a DQ-based ICA, adding DQs can* actually reduce efficiency. A single DQ can cause an efficiency drop arbitrarily close to 100*%. By comparison, in a* VQ-based ICA, adding additional queries can never reduce efficiency (assuming truthful bidding). These issues are also very prevalent in the real world. In Section 6, we demonstrate that in realistic domains, the gap in *average* efficiency between the SOTA DQ-based and VQ-based auctions can reach up to 8% points. Furthermore, the *average* efficiency of the CCA, the prominent DQ-based auction, declines by 8% points during the auction. VQ-based auctions avoid these pitfalls entirely. First, they can always achieve 100% efficiency after sufficiently many VQs (Lemma D.14). Second, asking additional VQs never reduces the efficiency of a VQ-based auction. In Appendix D.2, we provide further theoretical and intuitive arguments on why relying solely on DQs is insufficient. Optimally Combining DQs and VQs. Building on the discussion in this section, it is natural to leverage the strengths of both query types by starting with DQs and then transitioning to VQs. Example 1 in Appendix D.3 illustrates why this approach is effective: even after infinitely many DQs could not achieve more than 55% efficiency, a single VQ can achieve 100% efficiency. However, caution is required, as introducing even a single VQ can reduce the auction's efficiency by nearly 100% (Lemma D.7). To address this, we introduce the *bridge bid*, a specialized VQ designed to seamlessly connect the DQ and VQ phases of a hybrid auction. The bridge bid asks each bidder her value for the bundle she would have received according to the WDP (Equation (3)) after the final DQ round. Incorporating the bridge bid guarantees that the auction's final efficiency will be no less than its DQ-only efficiency (Lemma D.9). We demonstrate the significance of this bid in practice in Section 6.3 and Appendix G.8. Appendix D.3 provides further insights into combining DQs with VQs.

## 4. Mixed Query Learning

Combining DQs and VQs not only improves the final efficiency of auctions but also enables the global learning of bidders' value functions. In this section, we introduce a mixed training algorithm that leverages both query types. Specifically, we demonstrate the learning benefits of initializing auctions with DQs over VQs and show how integrating both query types leads to superior learning performance. Further details are presented in Appendix E.

## 4.1. Mixed Training Algorithm

To leverage the advantages of both DQs and VQs, we propose a two-stage training algorithm compatible with modern NN architectures, including mMVNNs. In each epoch, the ML model is first trained on all DQ responses using the loss function of Soumalias et al. (2024c). The key idea is to predict the bidder's utility-maximizing bundle at the given

| OPTIMIZATION TRAIN POINTS   | R2   | KT                          | MAE SCALED   | R2 c                                 |    |    |    |    |    |    |
|-----------------------------|------|-----------------------------|--------------|--------------------------------------|----|----|----|----|----|----|
| METRIC                      | VQS  | DQS                         | Tr           | Tp                                   | Tr | Tp | Tr | Tp | Tr | Tp |
| R 2 ON Vr                   | 20   | 40                          | 0.84         | 0.42 0.79 0.80 0.037 0.044 0.84 0.80 |    |    |    |    |    |    |
| 60                          | 0    | 0.73 −10.07 0.68 0.64 0.052 | 0.236        | 0.74 0.20                            |    |    |    |    |    |    |
| 0                           | 60   | 0.24 −3.07 0.77 0.77 0.103  | 0.128        | 0.83 0.76                            |    |    |    |    |    |    |
| R 2 ON Vp                   | 20   | 40                          | 0.82         | 0.01 0.79 0.80 0.041 0.062 0.84 0.83 |    |    |    |    |    |    |
| 60                          | 0    | 0.76 −3.40 0.72 0.62 0.049  | 0.141        | 0.77 0.05                            |    |    |    |    |    |    |
| 0                           | 60   | −0.05 −6.24 0.78 0.72 0.103 | 0.154        | 0.84 0.69                            |    |    |    |    |    |    |

R

2 ON Vr 20 40 0.84 0.42 0.79 0.80 0.037 0.044 0.84 0.80

60 0 0.73 −10.07 0*.68 0.*64 0.052 0.236 0.74 0.20

0 60 0.24 −3.07 0.77 0*.77 0.*103 0.128 0.83 0.76

R

2 ON Vp 20 40 0.82 0.01 0.79 0.80 0.041 0.062 0.84 0.83

60 0 0.76 −3.40 0.72 0.62 0*.049 0.141 0.77 0.*05

0 60 −0.05 −6.24 0.78 0.72 0.103 0.154 0.84 0.69

Table 1: Learning comparison of training only on DQs, only on VQs, or on both. Shown are averages over ten instances. Winners marked in gray.

prices by treating her ML model as her true value function. If the predicted reply deviates from the bidder's true reply, the loss equals the difference in predicted utility between the two bundles. This loss function provably captures all information provided by the DQ responses. Additionally, the model is trained on the VQ responses using a standard regression loss. For details, please refer to Appendix E.1.

## 4.2. Experimental Analysis

We demonstrate the learning benefits of initializing auctions with DQs rather than VQs and highlight how combining both query types leads to superior learning performance.

We conduct the following experiment: We perform hyperparameter optimization (HPO) to train an MVNN for the most critical bidder in the most realistic simulation domain
(see Appendix G.1 for details on the simulation and Appendix E.3 for results for other domains). For this bidder, we generate three training sets: (1) 40 DQs simulating 40 CCA clock rounds and 20 random VQs, (2) 60 DQs simulating 60 clock rounds with no VQs, and (3) 60 random VQs with no DQs. The models are evaluated on two validation sets: a *random bundle set* (Vr) with 50,000 uniformly sampled bundles, and a *price-driven set* (Vp) containing bundles requested under 200 random price vectors. Vr tests generalization across all bundles, while Vp focuses on utilitymaximizing bundles. We select the configuration with the best coefficient of determination (R2), averaged over 10 bidder instances. The selected configurations are then tested on 10 new bidders, generating hold-out tests sets Tr and Tp in the same way as Vp and Vr. We report R2, Kendall Tau (KT), *scaled* Mean Absolute Error (scaled MAE), and R2c. An R2 c value of 1 indicates perfect learning up to a constant shift, with differences between R2cand R2reflecting shift magnitude.

HPO procedures were consistent across all training sets, with identical test instances, seeds, search spaces, and computation time. Additional details are in Appendix G.3. Table 1 shows that training on a mix of DQs and VQs consistently outperforms training on either query type, particularly for utility-maximizing bundles in Tp, where mixed training achieves nearly three times lower MAE. The mixed model also closely matches the mean value for both test sets, as indicated by the small gap between R2 cand R2. In contrast, DQ-only models lack absolute value information, leading to relative but not unique value function learning, as evidenced by the large difference between R2and R2 c. This limitation goes even beyond constant shifts: Example 1 shows that even with all possible DQs, unique identification up to a constant shift is impossible. Meanwhile, VQ-only models suffer from distributional shifts between test sets, reflected in the significant discrepancy in R2and MAE across the two test sets. These shifts prevent VQ-trained models from capturing critical, high-value bundles due to the absence of utility-maximizing bundles in their training data.3 DQ-trained models generalize better to Tp than VQ-trained models, as Tp emphasizes high-value bundles critical for efficient allocations. This motivates initially training with DQs, as they provide global information about the allocation space and focus on high-value regions from the outset. These learning advantages are so pronounced that, as shown in Section 6, MLHCA only needs to follow up its 40 DQs with at most 18 VQs to outperform the previous SOTA,
which requires 100 VQs.

## 5. The Mechanism

In this section, we describe our ML-powered Hybrid Combinatorial Auction (MLHCA), which combines the auction and ML insights from Sections 3 and 4. We present a simplified version of MLHCA in Algorithm 1.

In Lines 2 to 5, we generate the first QCCA ∈ N DQs using the price update rule of the CCA. Similar to ML-CCA, we use larger price increments to arrive to similar prices as the ML-CCA in fewer rounds. In each of the next QDQ ∈ N ML-
powered rounds, we first train, for each bidder, an mMVNN on her demand responses (Line 8), and call NEXTPRICE (Soumalias et al., 2024c) (see Appendix A.3) to generate the next DQ based on the agents' trained mMVNNs (Line 9). If MLHCA has found market-clearing prices, then the corresponding allocation is efficient and is returned, along with payments π(R) according to the deployed payment rule (Line 15). MLHCA is plug-and-play compatible with many payment rules, such as VCG and VCG-nearest. If, by the end of the ML-powered DQs, the market has not cleared we switch to VQ rounds. In the first VQ round (Line 17) we ask each bidder for her *bridge bid* (see Definition D.8). This single VQ bid ensures that the MLHCA's efficiency is lower bounded by the efficiency after just the DQ rounds (Lemma D.9). For a detailed experimental evaluation of the bridge bid see Appendix G.8. In the final QVQ − 1 VQ
rounds, for each bidder, we query her value for the bundle 3At the start of a VQ-based auction, models are not accurate enough to target value-maximizing bundles.

Algorithm 1: MLHCA(Q
CCA, QDQ, QVQ, π)
Parameters :Q
CCA, QDQ, Q
VQ and π 1 R
VQ, RDQ ← ({})
N
i=1,({})
N i=1 2 for r = 1, ..., QCCA do ▷ Draw QCCA initial prices 3 p r ← *CCA(R*
DQ)
4 **foreach** i ∈ N do ▷ Initial DQs 5 R
DQ
i ← R
DQ
i ∪ {(x
∗
i (p r), pr)}
6 for r = Q
CCA + 1, ..., QCCA + Q
DQ do ▷ ML-powered DQs 7 **foreach** i ∈ N do 8 Mθ i ← MIXEDTRAINING(R
DQ
i, RVQ
i)
▷ Algorithm 4 9 p r ← NEXTPRICE(Mθ in i=1) ▷ Appendix A.3 10 **foreach** i ∈ N do 11 R
DQ
i ← R
DQ
i ∪ {(x
∗
i (p r), pr)}
12 if Pn i=1
(x
∗
i (p k))j = cj ∀j ∈ M **then**
▷ Market-clearing prices found 13 a
∗(R
DQ, RVQ) ← (x
∗
i (p r))n i=1 14 π(R
DQ, RVQ) ← (πi(R
DQ, RVQ))n i=1 15 **return** a
∗(R
DQ, RVQ) and π(R
DQ, RVQ)
16 **foreach** i ∈ N do ▷ Bridge bid 17 R
VQ
i ←
R
VQ
i ∪ {(a
∗
i (R
DQ, RVQ*), v*i(a
∗
i (R
DQ, RVQ)))}
18 for r = Q
CCA + Q
DQ + 2, ..., QCCA + Q
DQ + Q
VQ do
▷ ML-powered VQs 19 **foreach** i ∈ N do 20 Mθ i ← MIXEDTRAINING(R
DQ
i, RVQ
i)
▷ Algorithm 4 21 a ← NEXTALLOCATION Mθ i n i=1), RDQ, RVQ
▷ Appendix F
22 **foreach** i ∈ N do 23 R
VQ
i ← R
VQ
i ∪ {(ai, vi(ai))} ▷ Value query responses 24 Calculate final allocation a
∗(R
DQ, RVQ) as in Equation (3)
25 Calculate payments π(R
DQ, RVQ) ▷ E.g., VCG
(Appendix B)
26 **return** a
∗(R
DQ, RVQ) and π(R
DQ, RVQ)
she is allocated in the predicted optimal allocation (based on all ML models), under the constraint that she has not answered a VQ for that bundle in the past (Lines 21 to 23).4 The final allocation and payments are then determined based on all reports (Lines 24 to 25). For details, please see Appendix F.

## 6. Experiments

In this section, we experimentally evaluate MLHCA. We compare its efficiency against BOCA (Weissteiner et al., 2023) and ML-CCA (Soumalias et al., 2024b) the SOTA
VQ-based and DQ-based ICAs, respectively.

4This VQ algorithm was introduced in Brero et al. (2021) and used in most follow-up work following the MLCA framework.

## 6.1. Experiment Setup

To generate synthetic CA instances, we use the *spectrum* auction test suite (SATS) (Weiss et al., 2017), which includes various value models (domains) designed to simulate different auction environments. Following standard practice in this line of research (e.g., Soumalias et al. (2024c); Weissteiner et al. (2023)), we conduct experiments on the GSVM, LSVM, SRVM, and MRVM domains (see Appendix G.1 for details). SATS provides access to the true optimal allocation a
∗ ∈ F, allowing us to measure the efficiency loss, defined as 1 − V (a
∗(R))/V (a
∗), where R represents elicited reports. We focus on efficiency rather than revenue, as do all mechanisms we compare against. This is consistent with the primary application of ICAs in spectrum allocation, a government-run operation with a welfare-maximization mandate (Cramton, 2013). For results on revenue, see Appendix G.7. To ensure a fair comparison with prior work, we limit all auction mechanisms to 100 total queries. These consist of 100 VQs for BOCA, 100 DQs for ML-CCA, and 40 DQs and 60 VQs for MLHCA. For BOCA and ML-CCA, we use the best mechanism configurations and hyperparameters reported in their respective papers. For MLHCA's VQ rounds, we performed HPO separately for each bidder type in each domain, as detailed in Appendix E.2. For the DQ rounds, we adopted the HPO parameters reported by Soumalias et al. (2024c), since our learning algorithm, when restricted to DQs, is equivalent to theirs. For further experimental details and analysis of MLHCA's low computational costs, please refer to Appendices G.3 and G.4 respectively.

## 6.2. Efficiency Results

In Table 2, we show the average efficiency loss of each mechanism after 100 queries. For ML-CCA, we also report results if it were supplemented with the clock bids raised heuristic (see Section 2.2), which would involve up to an additional 100 VQs per bidder.5 Finally, we report the number of queries that MLHCA requires to outperform the final efficiency of each other mechanism, i.e., in GSVM, with 42 queries (40 DQs and 2 VQs) MLHCA statistically outperforms ML-CCA, even if ML-CCA were supplemented with 100 VQs from the clock bids raised heuristic. In Table 2, we observe that MLHCA significantly outperforms all other mechanisms across all domains. Notably, MLHCA is the *only* mechanism capable of achieving a perfect 100% efficiency in SRVM. Remarkably, it accomplishes this with fewer than 60 queries, while the other mechanisms fail even with 100 queries. In the LSVM domain, MLHCA achieves a 10-fold reduction in efficiency loss compared to BOCA, the previous SOTA. The most realistic domain, 5In the clock bids raised heuristic, the bidders only need to report their value for each *unique* bundle they bid on during the auction, which, for 100 DQs, can be up to 100 bundles.

MRVM further highlights MLHCA's superiority. Here, ML- HCA exceeds the efficiency of all other mechanisms by over 2% points, making MLHCA the first mechanism to substantially outperform CCA. MRVM simulates the 2014 Canadian spectrum auction (Weiss et al., 2017) with a revenue of USD 5.27 billion (Ausubel & Baranov, 2017), where 2% points correspond to over USD 100 million. Speed of convergence is another critical factor in these auctions. In all domains, MLHCA requires at most 74 queries (40 DQs and 34 VQs) to statistically outperform the final efficiency of both BOCA and ML-CCA, which use 100 VQs and 100 DQs, respectively. Furthermore, in three out of four domains, MLHCA surpasses the 100 DQ efficiency of ML-CCA with only 40 DQs and 2 VQs. These results align with our theoretical analysis in Appendix D.3, where we show that, once DQs have sufficiently informed the bidders' value functions, a single VQ can lead to 100% efficiency.

Figure 1 illustrates the efficiency loss path for all domains, highlighting MLHCA's consistent superiority. Up to query 40, MLHCA and ML-CCA perform identically since both mechanisms employ the same DQs and network configurations during these rounds. However, after query 40, ML- HCA's integration of VQs leads to a marked reduction in efficiency loss compared to ML-CCA, aligning with our insights on the efficiency of VQs and on the learning advantages of combining DQs and VQs (Sections 3 and 4). Across all domains, MLHCA also consistently outperforms BOCA, leveraging the early-stage advantages of DQs when ML models are still being quite uninformed and the later-stage learning advantages of combining DQs and VQs. In summary, MLHCA outperforms both DQ-based and VQ- based SOTA mechanisms in terms of both efficiency and speed of convergence, achieving high efficiency with fewer queries. This makes MLHCA a powerful and practical choice for real-world auction scenarios where high efficiency and rapid convergence are crucial. These empirical findings not only highlight the efficiency and convergence speed of MLHCA but also closely align with our theoretical insights. In the next section, we analyze how these results validate the predictions and theoretical guarantees established in this paper.

## 6.3. Alignment With Theoretical Insights

Figure 1 further validates our theoretical findings. The nonmonotonicity of DQ-based mechanisms, as suggested in Proposition 3.3, is evident in the efficiency loss path of both the CCA and the ML-CCA. Notably, in the LSVM domain, the CCA achieves higher *average* efficiency after just 5 DQs compared to 100. Additionally, the comparison between BOCA and ML-CCA underscores the inefficiency of random VQs in the early stages (Proposition 3.1), particularly in the MRVM domain, where BOCA's efficiency loss is

| EFFICIENCY LOSS IN %                                                                                  | QUERIES TO REJECT NULL HYPOTHESIS   |                          |             |                                                       |    |    |    |
|-------------------------------------------------------------------------------------------------------|-------------------------------------|--------------------------|-------------|-------------------------------------------------------|----|----|----|
| DOMAIN MLHCA                                                                                          | BOCA                                | ML-CCACLOCK ML-CCARAISED | CCA         | BOCA ≥ MLHCA ML-CCACLOCK ≥ MLHCA ML-CCARAISED ≥ MLHCA |    |    |    |
| GSVM 0.00 ± 0.00                                                                                      | -                                   | 1.77 ± 0.68              | 1.07 ± 0.37 | 9.60 ± 1.49                                           | -  | 42 | 42 |
| LSVM                                                                                                  | 0.04 ± 0.07 0.39 ± 0.31             | 8.36 ± 1.70              | 3.61 ± 0.77 | 17.44 ± 1.60                                          | 58 | 42 | 43 |
| SRVM                                                                                                  | 0.00 ± 0.00 0.06 ± 0.02             | 0.41 ± 0.11              | 0.07 ± 0.02 | 0.37 ± 0.11                                           | 42 | 42 | 42 |
| MRVM 4.81 ± 0.57 7.77 ± 0.35                                                                          | 6.94 ± 0.24                         | 6.68 ± 0.22              | 7.53 ± 0.48 | 54                                                    | 74 | 79 |    |
| Table 2: MLHCA (40DQs + 60VQs) vs BOCA (100VQs), ML-CCA (ML-CCAclock) (100DQs) and ML-CCA with raised |                                     |                          |             |                                                       |    |    |    |

GSVM 0.00 ± 0.00 - 1.77 ± 0.68 1.07 ± 0.37 9.60 ± 1.49 - 42 42 LSVM 0.04 ± 0.07 0.39 ± 0.31 8.36 ± 1.70 3.61 ± 0*.77 17.*44 ± 1.60 58 42 43 SRVM 0.00 ± 0.00 0.06 ± 0.02 0.41 ± 0.11 0.07 ± 0.02 0.37 ± 0.11 42 42 42 MRVM 4.81 ± 0.57 7.77 ± 0.35 6.94 ± 0.24 6.68 ± 0.22 7.53 ± 0.48 54 74 79

Table 2: MLHCA (40DQs + 60VQs) vs BOCA (100VQs), ML-CCA (ML-CCAclock) (100DQs) and ML-CCA with raised clock bids (ML-CCAraised) (100DQs and up to 100VQs). Shown are averages and a 95% CI. Winners based on a t-test with

significance level of 5% are marked in grey.

GSVM
0 20 40 60 80 100 Number of Elicited Bids 10 5 10 4 10 3 10 2 10 1 10 0 LSVM
0 20 40 60 80 100 Number of Elicited Bids 10 6 10 5 10 4 10 3 10 2 10 1 10 0 SRVM
10 0 MRVM
0 20 40 60 80 100 Number of Elicited Bids 10 4 10 3 10 2 10 1 10 0 Eff ici e nc y Los s (
Lo g S
ca le)
10 1 0 20 40 60 80 100 Number of Elicited Bids MLHCA ML-CCA CCA BOCA Start of ML DQ Rounds Start of ML VQ Rounds
orders of magnitude worse than that of mechanisms employing ML-powered DQs. Finally, MLHCA's performance after query 40 demonstrates the potential efficiency gains of supplementing DQs with VQs. The switch to ML-powered VQs results in a dramatic reduction in efficiency loss—by several orders of magnitude in the GSVM and SRVM domains—while the DQ-based ML-CCA, which was identical to MLHCA up to that point, stagnates. This aligns with Theorem 3.2, which proves that once ML models effectively capture bidder preferences, VQs can dramatically enhance efficiency. In contrast, ML-CCA's reliance on DQs prevents further improvements, even with well-trained models.

0 20 40 60 80 100 Number of Elicited Bids 40 50 60 70 80 90 100 Effi cie nc y (%
)

MLHCA Efficiency (Bridge Bid) MLHCA Efficiency (No Bridge Bid) Start of ML DQ Rounds Start of ML VQ Rounds No Bridge reaches DQ-only efficiency at 60 bids No Bridge matches Bridge Bid efficiency at 82 bids
ure 2, we plot MLHCA's efficiency in MRVM–the most realistic domain–against the number of bids, comparing performance with and without the bridge bid. Without the bridge bid, MLHCA's efficiency drops by 7.3% points when it transitions to its VQ rounds. Notably, MLHCA requires 20 of our powerful ML-powered VQs just to recover the efficiency lost by the introduction of the first VQ. This is consistent with Lemma D.7, where we showed that efficiency can arbitrarily decrease when a VQ is introduced in a DQ-based auction. In contrast, the bridge bid completely mitigates this efficiency drop, as proven in Lemma D.9. In Appendix G.8, we provide a detailed analysis and explaining the bridge bid's efficacy relative to market competition.

Finally, in Appendix G.9, we experimentally evaluate the Inverse variant of MLHCA, which uses the inverse query order: it begins with VQs and then transitions to DQs. Across all tested domains, reversing the query order results in substantial efficiency losses, reaching up to 5 percentage points. In the inverse auction, ML-powered DQs fail to improve upon the efficiency achieved by the preceding VQs. Moreover, the early use of VQs alone cannot match the efficiency attained by the later-stage VQs in MLHCA, due to significantly weaker learning performance when the bidders' models have not been trained on both query types. These findings further reinforce our theoretical results on the critical role of query ordering in hybrid auctions.

To demonstrate the effectiveness of the bridge bid, in Fig-

## 7. Conclusion

We have introduced MLHCA, the first ICA to effectively combine both demand and value queries. By employing tailored query generation algorithms, incorporating the full information from both query types, and leveraging the theoretical insights developed in this work, MLHCA significantly outperforms current SOTA mechanisms across all tested domains and with significantly fewer queries. Notably, prior to MLHCA, the best-performing mechanism varied by domain, but MLHCA unifies the SOTA, delivering the best performance across all domains. At first glance, it might seem obvious that combining DQs and VQs improves performance. However, one of the key insights of our work is that the ordering of queries matters. DQs provide broad but imprecise information across the entire space, while VQs offer targeted, precise information. As a result, DQs are more effective at the beginning of an auction, while VQs become advantageous once the auction's ML model has already been trained for a while. A second insight is that combining both query types requires careful handling. The efficiency of an auction using both DQs and VQs is non-monotone with respect to answered queries, as DQ responses establish lower bounds on bidders' valuations for queried bundles. Naively combining the two can lead to sharp efficiency drops, particularly in low-competition scenarios. However, by introducing a single, carefully-designed VQ, we can mitigate this effect and guarantee that the auction's efficiency does not fall below its DQ-only value. A promising direction for future work is incorporating epistemic uncertainty into MLHCA to enhance efficiency. Another is developing an algorithm to dynamically determine the optimal switch to VQs, reducing cognitive load.

## Acknowledgments

We are grateful to Greg d'Eon, Bin Yu, Josef Teichmann, and Denise Kunzli for helpful discussions and their sup- ¨ port. This work was supported by the Swiss National Science Foundation (SNSF) Postdoc.Mobility fellowship [grant number P500PT 225356] and ETH Zurich. ¨

## Impact Statement

This paper advances the field of iterative combinatorial auctions (ICAs) by introducing MLHCA, a novel machine learning-powered auction mechanism that achieves unprecedented efficiency while reducing bidders' cognitive load. The primary goal of this work is to enhance the practicality and efficiency of real-world auctions, such as those used in spectrum allocation, with large potential benefits for social welfare. By enabling more accessible and efficient auctions, MLHCA has the potential to positively impact market design, increasing participation and improving resource allocation across various domains. The methods proposed rely on standard ML and optimization techniques, and we do not foresee immediate ethical concerns arising from their application. However, as in any setting involving self-interested agents, potential conflicts of interest between participants may arise. While such concerns are important in practice (as they are for any auction mechanism), addressing them lies outside the scope of this paper.

## References

Almahdi, M., Mohammed, Y. A., and Attia, T. A. Simulating spectrum auctions: A reinforcement learning approach. In 2025 Emerging Technologies for Intelligent Systems (ETIS), pp. 1–6, February 2025. doi: 10.1109/ETIS64005.2025.10961724. URL https:// ieeexplore.ieee.org/document/10961724.

Ausubel, L. M. and Baranov, O. A practical guide to the combinatorial clock auction. *Economic Journal*, 127(605):F334–F350, 2017.

Ausubel, L. M. and Baranov, O. Iterative vickrey pricing in dynamic auctions, 2019.

Ausubel, L. M. and Baranov, O. Revealed preference and activity rules in dynamic auctions. International Economic Review, 61(2):471–502, 2020. doi: https://doi.org/10. 1111/iere.12431. URL https://onlinelibrary. wiley.com/doi/abs/10.1111/iere.12431.

Ausubel, L. M. and Baranov, O. V. Market design and the evolution of the combinatorial clock auction.

The American Economic Review, 104(5):446–451, 2014.

ISSN 00028282. URL http://www.jstor.org/ stable/42920978.

Ausubel, L. M., Cramton, P., and Milgrom, P. The clockproxy auction: A practical combinatorial auction design.

In Cramton, P., Shoham, Y., and Steinberg, R. (eds.), Combinatorial Auctions, pp. 115–138. MIT Press, 2006.

Balcan, M.-F., Sandholm, T., and Vitercik, E. Generalization guarantees for multi-item profit maximization: Pricing, auctions, and randomized mechanisms, 2023.

Bichler, M., Hao, Z., and Adomavicius, G. *Coalition-based* pricing in ascending combinatorial auctions, pp. 493– 528. Cambridge University Press, October 2017. ISBN 9781107135345. doi: 10.1017/9781316471609.025.

Bikhchandani, S. and Ostroy, J. M. The package assignment model. *Journal of Economic theory*, 107(2):377–406, 2002.

Blum, A., Jackson, J., Sandholm, T., and Zinkevich, M.

Preference elicitation and query learning. *Journal of* Machine Learning Research, 5:649–667, 2004.

Brero, G. and Lahaie, S. A bayesian clearing mechanism for combinatorial auctions. In Proceedings of the 32nd AAAI Conference on Artificial Intelligence, 2018.

Brero, G., Lubin, B., and Seuken, S. Combinatorial auctions via machine learning-based preference elicitation. In Proceedings of the 27th International Joint Conference on Artificial Intelligence, 2018.

Brero, G., Lahaie, S., and Seuken, S. Fast iterative combinatorial auctions via bayesian learning. In *Proceedings of* the 33rd AAAI Conference of Artificial Intelligence, 2019.

Brero, G., Lubin, B., and Seuken, S. Machine learningpowered iterative combinatorial auctions. arXiv preprint arXiv:1911.08042, Jan 2021.

Cole, R. and Roughgarden, T. The sample complexity of revenue maximization. In Proceedings of the Forty-Sixth Annual ACM Symposium on Theory of Computing, STOC
'14, pp. 243–252, New York, NY, USA, 2014. Association for Computing Machinery. ISBN 9781450327107. doi: 10.1145/2591796.2591867. URL https://doi. org/10.1145/2591796.2591867.

Cramton, P. Spectrum auction design. Review of Industrial Organization, 42(2):161–190, 2013.

d'Eon, G., Newman, N., and Leyton-Brown, K. Understanding iterative combinatorial auction designs via multiagent reinforcement learning. In Proceedings of the 25th ACM Conference on Economics and Computation, EC
'24, pp. 1102–1130, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400707049.

doi: 10.1145/3670865.3673644. URL https://doi. org/10.1145/3670865.3673644.

Dutting, P., Fischer, F., Jirapinyo, P., Lai, J. K., Lubin, B., ¨
and Parkes, D. C. Payment rules through discriminantbased classifiers. ACM Transactions on Economics and Computation, 3(1):5, 2015.

Dutting, P., Feng, Z., Narasimhan, H., Parkes, D. C., and ¨
Ravindranath, S. S. Optimal auctions through deep learning. In Proceedings of the 36th International Conference on Machine Learning, 2019.

Dutting, P., Mirrokni, V., Paes Leme, R., Xu, H., and Zuo, ¨
S. Mechanism design for large language models. In Proceedings of the ACM Web Conference 2024, WWW
'24, pp. 144–155, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400701719.

doi: 10.1145/3589334.3645511. URL https://doi. org/10.1145/3589334.3645511.

Estermann, B., Kramer, S., Wattenhofer, R., and Wang, Y.

Deep learning-powered iterative combinatorial auctions with active learning. In Proceedings of the 2023 International Conference on Autonomous Agents and Multiagent Systems, pp. 2919–2921, 2023.

Goeree, J. K. and Holt, C. A. Hierarchical package bidding:
A paper & pencil combinatorial auction. Games and Economic Behavior, 70(1):146–169, 2010.

Goetzendorff, A., Bichler, M., Shabalin, P., and Day, R. W.

Compact bid languages and core pricing in large multiitem auctions. *Management Science*, 61(7):1684–1703, 2015. doi: 10.1287/mnsc.2014.2076. URL https: //doi.org/10.1287/mnsc.2014.2076.

Golowich, N., Narasimhan, H., and Parkes, D. C. Deep learning for multi-facility location mechanism design. In Proceedings of the Twenty-seventh International Joint Conference on Artificial Intelligence and the Twenty-third European Conference on Artificial Intelligence, pp. 261– 267, 2018.

Heiss, J. Inductive Bias of Neural Networks and Selected Applications. Doctoral thesis, ETH Zurich, Zurich, 2024.

URL https://www.research-collection. ethz.ch/handle/20.500.11850/699241.

Heiss, J., Teichmann, J., and Wutte, H. How implicit regularization of Neural Networks affects the learned function - Part I, November 2019. URL https://arxiv.org/ abs/1911.02903.

Heiss, J., Teichmann, J., and Wutte, H. How infinitely wide neural networks can benefit from multitask learning - an exact macroscopic characterization. arXiv preprint arXiv:2112.15577, 2021. doi: 10.3929/
ETHZ-B-000550890. URL https://arxiv.org/ abs/2112.15577.

Heiss, J., Teichmann, J., and Wutte, H. How (implicit)
regularization of relu neural networks characterizes the learned function - part ii: the multi-d case of two layers with random first layer, 2023. URL https://arxiv. org/abs/2303.11454.

Huang, D., Marmolejo-Coss´ıo, F., Lock, E., and Parkes, D.

Accelerated preference elicitation with llm-based proxies, 2025. URL https://arxiv.org/abs/2501. 14625.

Innovation, Science and Economic Development Canada. 3800 mhz auction - provisional results, 2023. URL https://ised-isde.canada.ca/site/ spectrum-management-telecommunications/ en/spectrum-allocation/ 3800-mhz-auction-provisional-results\#
t1. Accessed: 2024-10-08.

Kwasnica, A. M., Ledyard, J. O., Porter, D., and De-
Martini, C. A new and improved design for multiobject iterative auctions. *Management Science*, 51(3): 419–434, 2005. ISSN 00251909, 15265501. URL http://www.jstor.org/stable/20110340.

Lahaie, S. and Lubin, B. Adaptive-price combinatorial auctions. In Proceedings of the 2019 ACM Conference on Economics and Computation, EC '19, pp. 749–750, New York, NY, USA, 2019. Association for Computing Machinery. ISBN 9781450367929. doi: 10. 1145/3328526.3329615. URL https://doi.org/ 10.1145/3328526.3329615.

Lahaie, S. M. and Parkes, D. C. Applying learning algorithms to preference elicitation. In Proceedings of the 5th ACM Conference on Electronic Commerce, 2004.

Lubin, B., Seuken, S., Beyeler, M., and Brero, G. imlca: Machine learning-powered iterative combinatorial auctions with interval bidding, 2021. URL https://arxiv. org/abs/2009.13605.

Maruo, R. and Kashima, H. Efficient preference elicitation in iterative combinatorial auctions with many participants, 2024. URL https://arxiv.org/abs/ 2403.19075.

Milgrom, P. and Segal, I. Designing the us incentive auction. *Handbook of spectrum auction design*, pp. 803–812, 2017.

Morgenstern, J. and Roughgarden, T. The pseudo-dimension of near-optimal auctions. In Proceedings of the 28th International Conference on Neural Information Processing Systems - Volume 1, NIPS'15, pp. 136–144, Cambridge, MA, USA, 2015. MIT Press.

Narasimhan, H., Agarwal, S. B., and Parkes, D. C. Automated mechanism design without money via machine learning. In Proceedings of the 25th International Joint Conference on Artificial Intelligence, 2016.

Nisan, N. and Segal, I. The communication requirements of efficient allocations and supporting prices. Journal of Economic Theory, 129(1):192–224, 2006.

Ongie, G., Willett, R., Soudry, D., and Srebro, N. A
function space view of bounded norm infinite width relu nets: The multivariate case. arXiv preprint arXiv:1910.01635, 2019. URL https://arxiv. org/pdf/1910.01635.pdf.

Parhi, R. and Nowak, R. D. What kinds of functions do deep neural networks learn? insights from variational spline theory. *SIAM Journal on Mathematics of Data Science*, 4 (2):464–489, 2022.

Rassenti, S. J., Smith, V. L., and Bulfin, R. L. A combinatorial auction mechanism for airport time slot allocation. The Bell Journal of Economics, pp. 402–417, 1982.

Savarese, P., Evron, I., Soudry, D., and Srebro, N. How do infinite width bounded norm networks look in function space? *arXiv preprint arXiv:1902.05040*, 2019. URL https://arxiv.org/abs/1902.05040.

Scheffel, T., Ziegler, G., and Bichler, M. On the impact of package selection in combinatorial auctions: an experimental study in the context of spectrum auction design. Experimental Economics, 15:667–692, 2012a.

Scheffel, T., Ziegler, G., and Bichler, M. On the impact of package selection in combinatorial auctions: an experimental study in the context of spectrum auction design. Experimental Economics, 15(4):667–692, 2012b.

Soumalias, E., Curry, M. J., and Seuken, S. Truthful aggregation of llms with an application to online advertising, 2024a. URL https://arxiv.org/abs/ 2405.05905.

Soumalias, E., Zamanlooy, B., Weissteiner, J., and Seuken, S. Machine learning-powered course allocation. In Proceedings of the 25th ACM Conference on Economics and Computation, EC '24, pp. 1099, New York, NY, USA, 2024b. Association for Computing Machinery. ISBN 9798400707049. doi: 10.1145/ 3670865.3673573. URL https://doi.org/10. 1145/3670865.3673573.

Soumalias, E. N., Weissteiner, J., Heiss, J., and Seuken, S.

Machine learning-powered combinatorial clock auction. Proceedings of the AAAI Conference on Artificial Intelligence, 38(9):9891–9900, Mar. 2024c. doi: 10.1609/ aaai.v38i9.28850. URL https://ojs.aaai.org/
index.php/AAAI/article/view/28850.

Weiss, M., Lubin, B., and Seuken, S. Sats: A universal spectrum auction test suite. In *Proceedings of the 16th* Conference on Autonomous Agents and MultiAgent Systems, pp. 51–59, 2017.

Weissteiner, J. Integrating advanced machine learning methods into market mechanisms. PhD thesis, University of Zurich, 2023.

Weissteiner, J. and Seuken, S. Deep learning—powered iterative combinatorial auctions. *Proceedings of the* AAAI Conference on Artificial Intelligence, 34(02): 2284–2293, Apr. 2020. doi: 10.1609/aaai.v34i02.

5606. URL https://ojs.aaai.org/index. php/AAAI/article/view/5606.

Weissteiner, J., Heiss, J., Siems, J., and Seuken, S.

Monotone-value neural networks: Exploiting preference monotonicity in combinatorial assignment. In Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI-22, pp. 541–
548. International Joint Conferences on Artificial Intelligence Organization, 7 2022a. doi: 10.24963/ijcai.2022/
77. URL https://doi.org/10.24963/ijcai. 2022/77. Main Track.

Weissteiner, J., Wendler, C., Seuken, S., Lubin, B., and Puschel, M. Fourier analysis-based iterative combina- ¨ torial auctions. In Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI-22, pp. 549–556. International Joint Conferences on Artificial Intelligence Organization, 7 2022b. doi: 10.24963/ijcai.2022/78. URL https://doi.org/ 10.24963/ijcai.2022/78. Main Track.

Weissteiner, J., Heiss, J., Siems, J., and Seuken, S. Bayesian optimization-based combinatorial assignment. Proceedings of the AAAI Conference on Artificial Intelligence, 37, 2023.

Williams, F., Trager, M., Panozzo, D., Silva, C., Zorin, D.,
and Bruna, J. Gradient dynamics of shallow univariate relu networks. In Advances in Neural Information Processing Systems, pp. 8378–8387, 2019.

## A. Extended Preliminaries And Literature Review A.1. Extended Literature Review

In addition to the related work mentioned in Section 1, we also want to mention some further recent work an ML-based ICAs. Estermann et al. (2023) use more diverse VQs for the initial VQs. They show that this diversity leads to higher efficiency than just asking initial VQs for i.i.d. uniformly random bundles. However, this does not solve the problem of it being cognitively very hard for bidders to answer these VQs that are not aligned with their preferences. Moreover, their efficiency results are outperformed by our MLHCA. Maruo & Kashima (2024) uses multi-task learning to transfer to improve the generalization of the MVNNs by leveraging similarities among the value functions across bidders. This technique should also be compatible with our MLHCA. Thus, it would be an interesting direction for future work to incorporate multi-task learning into MLHCA and to evaluate how much this would improve efficiency. From a game theoretical perspective, one should think very carefully if multi-task learning could change the incentives of bidders. From a game-theoretical perspective, one would achieve incentive-alignment with social welfare, if each bidder i cannot change the marginal efficiency of the economy N \ {i} (see Appendix B.5). For MLCA, 3 out of 4 VQs actually query these marginal economies, such that Mθ ihas no direct influence on these queries, which provides quite a strong game theoretical argument. Via multi-task learning, bidders have a more direct way to influence other bidders' models. While multi-task learning is a very promising direction to explore, one should be aware of potential game-theoretical risks imposed by multi-task learning. Lubin et al. (2021) allow bidders to answer VQs with an interval over prices instead of an exact price. It would be interesting to combine this approach with MLHCA in future work. Weissteiner (2023) and Heiss (2024, Section 4.4) provide a broader picture on ML-based ICAs.

Huang et al. (2025) explore how LLMs can be leveraged to create a new interaction paradigm for auctions, where the bidders interact with the mechanism by providing only natural language input. Another related line of research is *mechanism design for LLMs*, where participants bid to effect the output of an ML model, specifically an LLM, e.g. Dutting et al. ¨ (2024); Soumalias et al. (2024a). d'Eon et al. (2024); Almahdi et al. (2025) apply reinforcement learning algorithms to combinatorial auctions to better understand bidder strategies. Extending this line of work to mechanisms such as MLHCA would be an interesting direction for future research.

## A.2. A Machine Learning-Powered Ica

In this section, we present in detail the *machine learning-powered combinatorial auction (MLCA)* by Brero et al. (2021).

At the core of MLCA is a *query module* (Algorithm 2), which, for each bidder i ∈ I ⊆ N, determines a new value query qi.

First, in the *estimation step* (Line 1), an ML algorithm Aiis used to learn bidder i's valuation from reports Ri. Next, in the *optimization step* (Line 2), an *ML-based WDP* is solved to find a candidate q of value queries. In principle, any ML
algorithm Aithat allows for solving the corresponding ML-based WDP in a fast way could be used. Finally, if qi has already been queried before (Line 4), another, more restricted ML-based WDP (Line 6) is solved and qiis updated correspondingly.

This ensures that all final queries q are new. In Algorithm 3, we present MLCA. In the following, let R−i = (R1*, . . . , R*i−1, Ri+1*, . . . , R*n). MLCA proceeds in rounds until a maximum number of queries per bidder Qmax is reached. In each round, it calls Algorithm 2 (Qround − 1)n + 1 times: for each bidder i ∈ N, Qround − 1 times excluding a different bidder j ̸= i (Lines 5–10, sampled *marginal economies*) and once including all bidders (Line 11, *main economy*). In total each bidder is queried Qround bundles per round in MLCA.

At the end of each round, the mechanism receives reports Rnew from all bidders for the newly generated queries q new and updates the overall elicited reports R (Lines 12–14). In Lines 16–17, MLCA computes an allocation a
∗R that maximizes the *reported* social welfare (see Equation (3)) and determines VCG payments p(R) based on the reported values R (see Appendix Definition B.1).

Algorithm 2: NEXTQUERIES(*I, R*) (Brero et al. 2021)
Inputs : Index set of bidders I and reported values R
1 **foreach** i ∈ I do Fit Ai on Ri: Ai[Ri] ▷ Estimation step 2 Solve q ∈ arg max a∈F
P
i∈I
Ai[Ri](ai) ▷ Optimization step 3 **foreach** i ∈ I do 4 if (qi, vi(qi)) ∈ Ri **then** ▷ Bundle already queried 5 Define F
′ = {a ∈ F : ai ̸= x, ∀*(x, v*i(x)) ∈ Ri}
6 Re-solve q
′ ∈ arg maxa∈F′Pl∈I Al[Rl](al)
7 Update qi = q
′ i 8 **return** Profile of new queries q = (q1*, . . . , q*n)
Algorithm 3: MLCA(Q
init, Qmax, Qround) (Brero et al. 2021)
Params :Q
init, Qmax, Qround initial, max and \#queries/round 1 **foreach** i ∈ N do 2 Receive reports Ri for Q
init randomly drawn bundles 3 for k = 1*, ...,* ⌊(Q
max − Q
init)/Q*round*⌋ do ▷Round iterator 4 **foreach** i ∈ N do ▷ Marginal economy queries 5 Draw uniformly without replacement (Q
round−1) bidders from N \ {i} and store them in N˜
6 **foreach** j ∈ N˜ do 7 q new = q new∪ NEXTQUERIES(N \ {j}, R−j )
8 q new = q new∪ NEXTQUERIES(*N, R*) ▷ Main economy queries 9 **foreach** i ∈ N do 10 Receive reports R
new ifor q new i, set Ri = Ri ∪ R
new i 11 Given elicited reports R compute a
∗R as in Equation (3)
12 Given elicited reports R compute VCG-payments p(R) 13 **return** *Final allocation* a
∗R *and payments* p(R)

## A.3. Ml-Powered Demand Query Generation

In this section, we reprint the ML-powered demand query generation algorithm from Soumalias et al. (2024c). The critical notions behind the idea are those of indirect utility and revenue and clearing prices.

Definition A.1 (Indirect Utility and Revenue). For linear prices p ∈ R
m
≥0, a bidder's indirect utility U and the seller's indirect revenue R are defined as

$$U(p,v_{i}):=\operatorname*{max}_{x\in{\mathcal{X}}}\left\{v_{i}(x)-\langle p,x\rangle\right\}{\mathrm{~and~}}$$
{vi(x) − ⟨*p, x*⟩} and (4)
i.e., at prices p, Equations (4) and (5) are the maximum utility a bidder can achieve for all x ∈ X and the maximum revenue the seller can achieve among all feasible allocations.

Definition A.2 (Clearing Prices). Prices p ∈ R
m ≥0 are *clearing prices* if there exists an allocation a(p) ∈ F such that 1. for each bidder i, the bundle ai(p) maximizes her utility, i.e., vi(ai(p)) − ⟨*p, a*i(p)⟩ = U(p, vi), ∀i ∈ N, and 2. the allocation a(p) ∈ F maximizes the sellers revenue, i.e., Pi∈N ⟨*p, a*i(p)⟩ = R(p).

6 Theorem A.3 extends Bikhchandani & Ostroy (2002, Theorem 3.1), establishing a connection between the aforementioned definitions:
Theorem A.3 (Soumalias et al. (2024c)). Consider the notation from Definitions A.1 and A.2 *and the objective function* 6For linear prices, this maximum is achieved by selling every item, i.e., ∀j ∈ M :Pi∈N (ai)j = cj .

$$R(p):=\operatorname*{max}_{a\in{\mathcal{F}}}\left\{\sum_{i\in N}\left\langle p,a_{i}\right\rangle\right\}=\sum_{j\in M}c_{j}p_{j},$$
$$(4)$$
$$({\boldsymbol{5}})$$
cjpj , (5)
$$J^{\mathrm{{}}}$$

W(p, v) := R(p) + Pi∈N U(p, vi)*. Then it holds that, if a linear clearing price vector exists, every price vector*

$$p^{\prime}\in{\underset{{\hat{p}}\in\mathbb{R}_{\geq0}^{m}}{\operatorname{arg\,min}}}$$
W(˜p, v) (6a)
$$W({\bar{p}},v)$$
$$(6\mathrm{a})$$

such that (x
$$(x_{i}^{*}({\tilde{p}}))_{i\in N}\in{\mathcal{F}}$$
$$s u c h\;t h a t$$
$$(6\mathbf{b})$$
i(˜p))i∈N ∈ F (6b)
is a clearing price vector and the corresponding allocation a(p
′) ∈ F is efficient.

7 Theorem A.3 does not claim the existence of *linear clearing prices (LCPs)* p ∈ R
m
≥0. For general value functions v, LCPs may not exist (Bikhchandani & Ostroy, 2002). However, in the case that LCPs do exist, Theorem A.3 shows that all minimizers of equation 6 are LCPs and their corresponding allocation is efficient. This is at the core of their ML-powered demand query generation algorithm.

Their key idea to generate ML-powered demand queries is the following: As an approximation for the true value function vi, they use for each bidder a distinct mMVNN Mθ i
: X → R≥0 that has been trained on the bidder's elicited DQ data Ri.

Motivated by Theorem A.3, they then try to find the DQ p ∈ R
m
≥0 minimizing W(p, Mθ i n i=1) subject to the feasibility constraint equation 6b. This way, we find demand queries p ∈ R
m
≥0 which, given the already observed demand responses R,
have high clearing potential. Note that equation 6 is a hard, bi-level optimization problem. Instead, Theorem A.4 allows them to minimize the problem via gradient descent:
Theorem A.4 ((Soumalias et al., 2024c)). Let Mθ i n i=1 *be a tuple of trained mMVNNs and let* xˆ
∗
i(p) ∈
arg maxx∈X Mθ i
(x) − ⟨p, x⟩	*denote each bidder's predicted utility maximizing bundle w.r.t.* Mθ i
. Then it holds that p 7→ W(p, Mθ i n i=1) is convex, Lipschitz-continuous and a.e. differentiable*. Moreover,*

$$c-\sum_{i\in N}\hat{x}_{i}^{*}(p)\in\nabla_{p}^{\rm sub}W(p,\left({\cal M}_{i}^{\theta}\right)_{i=1}^{n})\tag{1}$$
$$\left(7\right)$$

$$(8\mathrm{a})$$

is always a sub-gradient and a.e. a classical gradient.

With Theorem A.4, we obtain the following update rule of classical GD p new j a.e. = pj − γ(cj −Pi∈N (ˆx
∗ i
(p))j ), ∀j ∈ M.

Interestingly, this equation has an intuitive economic interpretation. If the j th item is over/under-demanded based on the predicted utility-maximizing bundles xˆ
∗
i(p), then its new price p new jis increased/decreased by the learning rate times its over/under-demand. To enforce constraint equation 6b in GD, they asymmetrically increase the prices 1 + µ ∈ R≥0 times more in case of over-demand than they decrease them in case of under-demand. This leads to the final update rule:

$$p_{j}^{\text{new}}\stackrel{{a.e.}}{{=}}p_{j}-\tilde{\gamma}_{j}(c_{j}-\sum_{i\in N}(\hat{x}_{i}^{*}(p))_{j}),\,\forall j\in M,$$  $$\tilde{\gamma}_{j}:=\left\{\begin{array}{ll}\gamma\cdot(1+\mu)&,c_{j}<\sum_{i\in N}(\hat{x}_{i}^{*}(p))_{j}\\ \gamma&,\text{else}\end{array}\right.$$
$$({\mathfrak{s b}})$$

## B. Payment And Activity Rules

In this section, we reprint the VCG and VCG-nearest payment rules, as well as give an overview of activity rules for the CCA, and argue why the most prominent choices are also applicable to our MLHCA. Finally, we show how MLHCA can immediately detect if a bidder's reports are inconsistent with any valuation function.

## B.1. Vcg Payments

Definition B.1. (VCG PAYMENTS FROM DEMAND AND VALUE QUERY DATA) Let R = (R1*, . . . , R*n) denote an elicited set of both demand and value query data from each bidder and let R−i:= (R1, . . . , Ri−1, Ri+1*, . . . , R*n). We then calculate 7More precisely, constraint equation 6b should be reformulated as

$\mathbb{E}(x_{i}^{*}(\bar{p}))_{i\in N}\in\mathbb{X}_{i}^{*}(\bar{p}):(x_{i}^{*}(\bar{p}))_{i\in N}\in\mathbb{F}$,
where X
∗
i (˜p) := arg maxx∈X {vi(x) − ⟨*p, x* ˜ ⟩}, since in theory, x
∗
i (˜p) does not always have to be unique.

the VCG payments $\pi^{\text{VCG}}$(. 
as follows:
$$(R)=(\pi_{1}^{\mathrm{VCG}}(R)\ldots,\pi_{n}^{\mathrm{VCG}}(R))\in\mathbb{R}_{\geq0}^{n}{\mathrm{~as~for~}}R$$
$$\pi_{i}^{\mathrm{VCG}}(R):=\sum_{j\in N\setminus\{i\}}\widetilde{v}_{j}\left(a^{*}(R_{-i})_{j};R_{j}\right)-\sum_{j\in N\setminus\{i\}}\widetilde{v}_{j}\left(a^{*}(R)_{j};R_{j}\right).$$
∗(R)j ; Rj ). (9)
$${\mathrm{obllows}}1$$
$$(9)$$
excluding bidder $i$, i.e., . 
$$z{\mathrm{~inferred~so~}}$$
$$a^{*}(R_{-i})\in\arg\operatorname*{max}_{a\in{\mathcal{F}}}\sum_{j\in N\backslash\{i\}}{\tilde{v}}_{j}(a_{j};R_{j}),$$
$$(10)$$
vej (aj ; Rj ), (10)
$u_{i}=v_{i}(a^{*}(R)_{i})-\pi_{i}^{\rm VCG}(R)$  $=v_{i}(a^{*}(R)_{i})+\sum_{j\in N\backslash\{i\}}\widetilde{v}_{j}\;(a^{*}(R)_{j};R_{j})$  $=\sum_{j\in N\backslash\{i\}}\widetilde{v}_{j}\;(a^{*}(R_{-i})_{j};R_{j})$.  

$$(11)$$

## B.2. Vcg-Nearest Payments B.3. On The Importance Of Activity Rules To Align Incentives

and a
∗(R) is the inferred social welfare-maximizing allocation (see Equation (3)).

Thus, when using VCG payments, bidder i's utility is: To define the VCG-nearest payments, we must first introduce the core:
Definition B.2. (THE CORE) An outcome (a, π) *∈ F ×* R
n≥0
(i.e., a tuple of a feasible allocation a and payments π) is in the core if it satisfies the following two properties:
1. The outcome is *individual rational*, i.e, ui = vi(ai) − πi ≥ 0 for all i ∈ N

 $\text{2.The core constraints}$  . 
vi(ai) (11)
$$\forall\;L\subseteq N\;\sum_{i\in N\setminus L}\pi_{i}(R)\geq\operatorname*{max}_{a^{\prime}\in{\mathcal{F}}}\sum_{i\in L}v_{i}(a_{i}^{\prime})-\sum_{i\in L}v_{i}(a_{i})$$
where $v_i(a_i)$ is bidder 
where vi(ai) is bidder i's value for bundle ai and F is the set of feasible allocations.

In words, a payment vector π (together with a feasible allocation a) is in the core if no coalition of bidders L ⊂ N is willing to pay more for the items than the mechanism is charging the winners. Note that by replacing the true values vi(ai) with the bidders' (possibly untruthful) *inferred values* based on their reports vei(ai; Ri) in Definition B.2 one can equivalently define the *revealed core*. Now, we can define Definition B.3. (MINIMUM REVENUE CORE) Among all payment vectors in the (revealed) core, the (revealed) minimum revenue core is the set of payment vectors with smallest L1-norm, i.e., which minimize the sum of the payments of all bidders. We can now define VCG-nearest payments:
Definition B.4. (VCG-NEAREST PAYMENTS) Given an allocation aR for bidder reports R, the VCG-nearest payments π VCG-nearest(R) are defined as the vector of payments in the (revealed) minimum revenue core that minimizes the L2-norm to the VCG payment vector π VCG(R).

In the CCA, activity rules serve multiple purposes. First, they help accelerate the auction process. Second, they reduce
"bid-sniping" opportunities—bidders concealing their true intentions until the very last rounds of the auction.8 Third, they limit surprise bids in the supplementary round of the CCA, significantly reducing a bidder's ability to drive up opponents' payments by overbidding on bundles they cannot win (Ausubel & Baranov, 2017). There are two types of activity rules that are implemented in a CCA:
8The notion of "bid-sniping" originated in eBay auctions with predetermined ending times, where high-value bidders could reduce their payments by submitting bids at the very last moment.

where $ a^*(R_{-i})$ is th
∗(R−i) is the allocation that maximizes the inferred social welfare when excluding bidder i, i.e.,
1. *Clock phase activity rules*, which limit the bundles that an agent can bid on during the clock phase, based on their bids in previous clock rounds.

2. *Supplementary round activity rules*, which restrict the amounts that an agent can bid on specific sets of items during the supplementary round.

Traditionally, most clock phase activity rules in the CCA have relied on either revealed-preference principles or a points-based system, where points are assigned to each item before the auction, and bidders are only allowed to submit monotonically non-increasing bids in terms of points. In other words, as prices rise across rounds, bidders cannot submit bids for larger sets of items. Both of these approaches, as well as hybrid combinations thereof, were shown to actually further interfere with truthful bidding in some cases (Ausubel & Baranov, 2014; 2020). However, Ausubel & Baranov (2019) showed that basing clock phase activity rules entirely on the *generalized axiom of* revealed preference (GARP) can dynamically approximate VCG payoffs, thus improving the bidding incentives of the CCA. GARP imposes revealed-preference constraints (see Definition B.5) on the bidder's demand responses. The GARP activity rule requires that the bidder demonstrates rational behavior in her demand choices, without necessitating a monotonic price trajectory. As a result, it can also be applied during the ML-powered DQ phase of MLHCA, allowing our mechanism to enjoy similar improvements in bidding incentives. For the supplementary round, the CCA's most prominent activity rules are again based on a combination of points-based systems and revealed-preference ideas, which we outline below: Definition B.5. (REVEALED-PREFERENCE CONSTRAINT) The revealed-preference constraint for bundle x ∈ X with respect to clock round r is bi(x) ≤ bi(x r) + ⟨p r, x − x r⟩, (12)
where bi(x) ∈ R≥0 is bidder i's bid for bundle x ∈ X in the supplementary round, x r ∈ X is the bundle demanded by the agent at clock round r, bi(x r) ∈ R≥0 is the final bid for bundle x r ∈ X and p r ∈ R
m
≥0is the linear price vector of clock round r. Intuitively, the revealed-preference constraint ensures that a bidder cannot claim a higher value for bundle x relative to bundle x r, given that they expressed a preference for bundle x rat the given prices p r(see Equation (1)). The difference between the three most prominent supplementary round activity rules is with respect to *which clock rounds* the revealed-preference constraint should be satisfied. Specifically:
1. *Final Cap:* A bid for bundle x ∈ X should satisfy the revealed-preference constraint (Definition *B.5)* with respect to the *final* clock round's price p QCCA∈ R≥0 and bundle x QCCA ∈ X .

2. *Relative Cap:* A bid for bundle x ∈ X should satisfy the revealed-preference constraint (Definition *B.5)* with respect to the last clock round for which the bidder was eligible for that bundle x ∈ X , based on the points-based system.

3. *Intermediate Cap:* A bid for bundle x ∈ X should satisfy the revealed-preference constraint (Definition *B.5)* with respect to all eligibility-reducing rounds, starting from the last clock round for which the bidder was eligible for x ∈ X based on the point system.

Ausubel & Baranov (2017) showed that combining the *Final Cap* and *Relative Cap* activity rules leads to the largest amount of reduction in bid-sniping opportunities for the UK 4G auction, as measured by the theoretical bid amount that each bidder would need to increase her bid by in the supplementary round in order to protect her final clock round bundle. Finally, note that the Final- and *Intermediate Cap* activity rules can also be applied to the ML-powered DQ phase of our MLHCA.9 To conclude, both the DQ and VQ phases of MLHCA are compatible with the most prominent activity rules of the CCA, and MLHCA also remains compatible with the commonly used VCG-nearest pricing rule (Definition B.4). Combined with MLHCA's similar interaction paradigm to the CCA, these aspects provide strong evidence that our mechanism can leverage activity rules to effectively mitigate bidder misreporting opportunities, much like the classical CCA.

9Soumalias et al. (2024c) argued that with the modification for the *Relative Cap* rule that the revealed-preference constraint should hold for the Q
CCA rounds that follow the same price update rule as the CCA, and then the ML-powered clock rounds should be treated as corresponding to the same amount of points, since the prices in these rounds on aggregate stay very close to the prices of the last Q
init round.

## B.4. Mlhca Can Detect Inconsistent Misreports

In the following lemma, we formally prove that if a bidder's reports are inconsistent with any valuation function, then the training loss for that bidder's network will be strictly positive, thus MLHCA can detect such misreports.

Lemma B.6 (Strictly Positive Loss from an Inconsistent Datapoint). Let R = (RDQ, RVQ) *be a set of elicited reports by* a bidder that is rationalizable by some monotone valuation function v0 : X → R≥0*. Suppose, that during the MLHCA*
auction (Algorithm *1), the bidder responds to the next query, either a DQ* (xe
∗(p re), pr˜) or a VQ (˜x, v˜(˜x)) *and assume that* no monotone valuation v can simultaneously rationalize all of her responses R′. Then, when using Algorithm 4 (with any regression loss F for the VQ responses that satisfies F ≥ 0 and y = ˜y ⇐⇒ F(y, y˜) = 0) to fit an MVNN Mθto R′, we have minθ L(θ) > 0.

Proof. We prove the claim in cases. Case 1: Suppose that the bidder misreports in a way that is non-rationalizable by any valuation function during the DQ phase of the auction. In that phase, the bidder's set of reports consists only of demand queries. For each datapoint (x
∗(p r), pr) in RDQ, Algorithm 4 attempts to make

$$\hat{x}^{*}(p^{r})\;\;\in\;\arg\operatorname*{max}_{x\in{\mathcal{X}}}\Bigl[{\mathcal{M}}^{\theta}(x)-\langle p^{r},\;x\rangle\Bigr]$$
$$x^{*}(p^{r})\rangle]$$
$$\mathbf{\hat{\Sigma}}$$

match the reported x
∗(p r). If it does not match, the loss is incremented by a nonnegative amount:

∆Lr(θ) = -Mθ(ˆx
$$\left[{\mathcal M}^{\theta}(\hat{x}^{*}(p^{r}))-\langle p^{r},\hat{x}^{*}(p^{r})\rangle\right]\;-\;\left[{\mathcal M}^{\theta}(x^{*}(p^{r}))-\langle p^{r}\rangle\right]$$
r, x∗(p
r)⟩≥ 0.

Hence the total loss L(θ) is always weakly positive. Suppose, for contradiction, that there exists θ with L(θ) = 0. If L(θ) = 0, it means the predicted best response matches the reported one, i.e., xˆ
∗(p r) = x
∗(p r) for all r, including r = ˜r.

However, for any θ ∈ Θ, the (m)MVNNMθis by construction a valid valuation function satisfying free disposal (Weissteiner et al., 2022a; Soumalias et al., 2024c). The condition xˆ
∗(p r) = x
∗(p r) for all r means precisely that Mθ*rationalizes all* data in D′. Thus, there exists a valuation function rationalizing all data points, including xe
∗(p r), a contradiction.

Case 2: Suppose that the bidder misreports in a way that is non-rationalizable by any valuation function during the VQ phase of the auction. Similarly, given that the loss function in each datapoint (both DQs and VQs) is weakly positive, the only way the loss can be zero is if it is zero on every point. But then, the MVNN once again has rationalized the agent's reports. Thus, a value function exists that rationalizes all of the agent's reports, a contradiction. Note that Lemma B.6 can also be applied to the case where we observe 0 VQs. Thus, Lemma B.6 can also be applied to detect inconsistent misreporting for DQ-only auctions such as ML-CCA. Further note that Lemma B.6 can always detect inconsistent misreporting, while other forms of misreporting cannot be detected this way.

## B.5. On The Importance Of Marginal Economies To Align Incentives

In this section, we review the key arguments from Brero et al. (2021) on why MLCA provides strong incentives for truthful reporting in practice. These arguments extend to any ML-powered ICA that employs the same VQ-generation algorithm, including MLHCA. Bidder i's utility in MLCA (and MLHCA) under VCG payments (see Definition B.1) can be expressed as:

$$u_{i}=v_{i}(a^{*}(R)_{i})-\pi_{i}^{\mathrm{VCG}}(R)$$ $$=v_{i}(a^{*}(R)_{i})+\sum_{j\in N\setminus\{i\}}\widetilde{v}_{j}\left(a^{*}(R)_{j};R_{j}\right)-\underbrace{\sum_{j\in N\setminus\{i\}}\widetilde{v}_{j}\left(a^{*}(R_{-i})_{j};R_{j}\right)}_{\text{(b)Inferred SW of marginal economy}}\.$$
$\square$
Any beneficial misreport by bidder i must increase the difference (a) − (b).

MLCA has two features that mitigate manipulations. First, MLCA explicitly queries each bidder's marginal economy (Algorithm 3, Line 5), which implies that (b) is practically independent of bidder i's reports. Experimental evidence supporting this claim is provided in Section 7.3 of Brero et al. (2021). Second, MLCA (and also MLHCA) enables bidders to "push" information to the auction which they deem useful. This mitigates certain manipulations that target (a), as it allows bidders to increase (a) with truthful information. Brero et al. (2021) argue that any remaining manipulation would be implausible as it would require almost complete information. Under further assumptions, we can also derive two theoretical incentive guarantees:
- Assumption 1 requires that, for all bidders i ∈ N, if all other bidders report truthfully, then the reported social welfare of bidder i's marginal economy (i.e., term (b)) is *independent* of her value reports.

- Assumption 2 requires that, if all bidders i ∈ N bid truthfully, then MLCA *finds an efficient allocation*.

Result 1: Social Welfare Alignment Under Assumption 1, and given that all other bidders are truthful, MLCA is social welfare aligned. This means that the only way for a bidder to increase her true utility is by increasing the reported social welfare of a
∗(R) in the main economy (i.e., term (a)), which, in this case, equals the true social welfare of a
∗(R) (Brero et al., 2021, Proposition 3). The same is true for the VQ phase of MLHCA, as it employs the same allocation and payment rules. Result 2: Ex-Post Nash Equilibrium If both Assumption 1 and Assumption 2 hold, then bidding truthfully constitutes an ex-post Nash equilibrium in MLCA (Brero et al., 2021, Proposition 4). The same is true for the VQ phase of MLHCA, as it employs the same allocation and payment rules. Remark B.7 (Experimental Evaluation of Assumption 2). The results shown in Tables 2 and 9 suggest that Assumption 2 is more realistic for MLHCA than for any other mechanism. For GSVM, Assumption 2 is absolutely realistic for MLHCA and was already realistic for other VQ-based mechanisms such as the ones proposed by (Weissteiner & Seuken, 2020; Weissteiner et al., 2022a; 2023). Also for SRVM, Assumption 2 is very realistic for MLHCA. In fact, MLHCA is the first method from Table 2 that always found an efficient allocation (only methods from Table 9 that use significantly more than 200 can keep up with this). Theoretically achieving 100% efficiency in all 50 random instances of an auction does not suffice as mathematical proof that the auction will always achieve 100% efficiency. However, for GSVM and SRVM, the fact that MLHCA found an efficient allocation within the first 60 out of 100 queries for all 50 instances, strongly suggests that 100 queries allow MLHCA to find an efficient allocation with almost 100% probability. For LSVM, MLHCA found an efficient allocation in 49 out of 50 auction instances, which from a practical point of view also almost satisfies Assumption 2, and with a few queries more fully satisfying Assumption 2 might be in reach. At least for every domain, MLHCA is closer to satisfying Assumption 2 than its competitors.

To conclude, MLHCA's compatibility with both *activity rules* during its DQ rounds and *marginal economies* during its VQ rounds, as well as its compatibility with VCG and VCG-nearest payments, provides strong evidence that MLHCA can effectively mitigate opportunities for bidder misreporting.

## C. Mvnn

The original definition (Weissteiner et al., 2022a) is a special case of the more general definition (Soumalias et al., 2024c) that we state here.

Definition C.1 (MVNN). An MVNN Mθ i: X → R≥0 for bidder i ∈ N is defined as

$\mathcal{M}_{i}^{\theta}(x):=W^{i,K_{i}}\varphi_{0,t^{i},K_{i}-1}\left(\ldots\varphi_{0,t^{i},1}\left(W^{i,1}\left(Dx\right)+b^{i,1}\right)\ldots\right)$
i,1)*. . .*(13)
- Ki + 2 ∈ N is the number of layers (Ki hidden layers),
- {φ0,ti,k }
Ki−1 k=1 are the MVNN-specific activation functions with cutoff t i,k > 0, called *bounded ReLU (bReLU)*:

$$(13)$$
$$(14)$$

φ0,ti,k (·) := min(t i,k, max(0, ·)) (14)
- Wi:= (Wi,k)
Ki k=1 with Wi,k ≥ 0 and b i:= (b i,k)
Ki−1 k=1 with b i,k ≤ 0 are the *non-negative* weights and *non-positive* biases of dimensions d i,k × d i,k−1and d i,k, whose parameters are stored in θ = (Wi, bi).

- D := diag (1/c1*, . . . ,*
1/cm) is the linear normalization layer that ensures Dx ∈ [0, 1] and is not trainable.

Remark C.2. The index i of the MVNN Mθ i(x) emphasizes that we train an individual MVNN for every bidder i to approximate vi. In the following, we sometimes omit the index i if we just want to make general arguments about the MVNN architecture without. Remark C.3 (Linear Skip Connection). Sometimes we also use linear skip connections as introduced in Weissteiner et al. (2023, Definition F.1) Remark C.4 (Initiaization). We always use the initialization scheme from Weissteiner et al. (2023, Section 3.2 and Appendix E), which offers crucial advantages over standard initialization schemes as discussed in Weissteiner et al. (2023, Section 3.2 and Appendix E).

## C.1. On The Inductive Bias Of Mvnns

Weissteiner et al. (2022a); Soumalias et al. (2024c) have shown that MVNNs can represent any monotonic normalized function on X . However, for finitely many data points, multiple different monotonic functions can fit the data equally well, but the training algorithm will choose only one of these functions. We want to understand according to which preferences the algorithm makes this choice, i.e., we want to understand its inductive bias.

For certain ReLU-NNs it has been shown that L2-regularization (also known as "weight decay") of the parameters θ corresponds to regularizing a Lp-norm of the second derivative of the function (Heiss et al., 2019; 2023; 2021; Heiss, 2024; Savarese et al., 2019; Ongie et al., 2019; Williams et al., 2019; Parhi & Nowak, 2022). Since the second derivative of linear functions is zero, these NNs prefer linear functions. However, MVNNs use a different activation function (Weissteiner et al., 2022a). For MVNNs, no theoretical result about their second derivative has been proven so far. It is quite clear that the L2-regularization of the parameters of a MVNN does not exactly correspond to any Lp-norm of the second derivative. Weissteiner et al. (2023) modified the MVNN architecture by adding so-called linear skip connections (Weissteiner et al., 2023, Definition F.1) to obtain an inductive bias towards linear functions. If one uses unregularized linear skip connections but regularizes all other parameters, it is quite obvious that the optimal parameters will only have non-zero weights in the linear skip connections if a monotonic linear function can perfectly explain the data.10 In the setting of Example 1 (which is based on the example in the proof of Theorem 3.2) one can also prove that MVNNs with arbitrarily small L2-regularization, would always choose a function that is linear on X given any possible truthful DQ responses from bidder 2, even without linear skip connections.

Proposition C.5. As in Example *1, let* n = 2, m = 1, c1 = 10 and v2 such that whenever bidder 2 is queried a DQ she answers in the following way: