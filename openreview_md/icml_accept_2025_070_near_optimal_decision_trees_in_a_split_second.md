# 

Varun Babbar * 1 **Hayden McTavish** * 1 Cynthia Rudin 1 **Margo Seltzer** 2

## Abstract

Decision tree optimization is fundamental to interpretable machine learning. The most popular approach is to greedily search for the best feature at every decision point, which is fast but provably suboptimal. Recent approaches find the global optimum using branch and bound with dynamic programming, showing substantial improvements in accuracy and sparsity at great cost to scalability. An ideal solution would have the accuracy of an optimal method and the scalability of a greedy method. We introduce a family of algorithms called SPLIT (SParse Lookahead for Interpretable Trees) that moves us significantly forward in achieving this ideal balance. We demonstrate that not all sub-problems need to be solved to optimality to find high quality trees; greediness suffices near the leaves. Since each depth adds an exponential number of possible trees, this change makes our algorithms orders of magnitude faster than existing optimal methods, with negligible loss in performance. We extend this algorithm to allow scalable computation of sets of near-optimal trees (i.e., the Rashomon set).

## 1. Introduction

Decision tree optimization is core to interpretable machine learning (Rudin et al., 2022). Simple decision trees present the entire model reasoning process transparently, directly allowing faithful interpretations of the model (Arrieta et al., 2020). This helps users choose whether to trust the model and to critically examine any perceived flaws.

*Equal contribution 1Department of Computer Science, Duke University, Durham, USA 2Department of Computer Science, University of British Columbia, Vancouver, Canada. Correspondence to: Varun <varun.babbar@duke.edu>, Hayden <hayden.mctavish@duke.edu>.

1 Optimizing the performance of decision trees while preserving their simplicity presents a significant challenge. Traditional greedy methods scale linearly with both dataset size and the number of features (Breiman, 1984; Quinlan, 2014). However, these methods tend to yield suboptimal results, lacking general guarantees on either sparsity or accuracy. Recent advances in decision tree algorithms use dynamic programming techniques combined with branch-and-bound strategies, offering solutions that are faster than brute-force approaches and provably optimal (Lin et al., 2020; Aglin et al., 2020; Demirovic et al. ´ , 2022; McTavish et al., 2022). In fact, Demirovic et al. ´ (2022) and van der Linden et al. (2024) reveal an average gap of 1-2 percentage points between greedy and optimal trees, with Demirovic et al. ´ (2022) showing that some datasets can exhibit gaps as large as 10 percentage points. These algorithms struggle to scale to datasets with hundreds or thousands of features or to deeper trees. It seems that we should return to greedy methods for larger-scale problems, but this would come at a loss of performance. Ideally, we should leverage greed only when it does not significantly deviate from optimality and use dynamic programming otherwise. Dynamic programming approaches build trees recursively, downward from the root. Problems farther from the root contain fewer samples and produce fewer splits. As we show, *greedy splits near the* root sacrifice performance, while greedy splits near the leaves produce performance close to the optimal. This suggests that we can tolerate less precision on problems close to leaves than on problems closer to the root - and that full optimization on those problems closer to the leaves yields only marginal returns relative to greedy, since we only have a few splits remaining. This has enormous implications, since the number of candidate trees increases exponentially with increases in depth; using greedy splitting closer to the leaves of the tree massively reduces the search space. We leverage this observation to construct SPLIT (SParse Lookahead for Interpretable Trees), a family of decision tree algorithms that are over 100× **faster than state of the art**
optimal decision tree algorithms, with negligible sacrifice in performance. They can also be tuned to a user-defined level of sparsity. Instead of searching through the entire space of decision trees up to a given depth, our algorithm performs dynamic programming with branch and bound up to only a shallow "lookahead" depth, conditioned on all splits henceforth being chosen greedily. Our contributions are as follows.

- We develop a family of decision tree algorithms that scale with the dataset size and number of features comparably to standard greedy algorithms but produce trees that are as accurate and sparse as optimal ones (e.g., Lin et al., 2020).

- We extend our decision tree algorithms to allow scalable, accurate approximations of the Rashomon set of decision trees (Breiman, 2001; Xin et al., 2022).

- We theoretically prove that our algorithms scale exponentially faster in the number of features than optimal decision tree methods and are capable of performing arbitrarily better than a purely greedy approach.

## 2. Related Work

We are interested in accurate, interpretable decision tree classifiers that we can find efficiently. We discuss these three goals as they pertain to existing work. Consistent with recommendations from Rudin et al. (2022); Costa & Pedreira (2023), we emphasize sparsity, expressed in terms of the number of leaves, as the primary mechanism for tree interpretability. Sparsity has a strong correlation with user comprehension (Piltaver et al., 2016). Zhou et al. (2018) fit a regression model to user-reported interpretability for decision trees, also finding that trees with fewer leaves were more interpretable. They also found that deep, sparse trees were more interpretable than shallow trees with the same sparsity. Izza et al. (2022) provides a way to use a sparse decision tree to provide succinct individual explanations. However, finding deep, sparse trees with existing methods can be computationally infeasible. We bridge this gap - our algorithms are capable of finding sparse trees without constraining them to be shallow. Greedy Decision Trees A long line of work explores greedy algorithms such as CART (Breiman, 1984) and C4.5 (Quinlan, 2014). These methods first define a heuristic feature quality metric such as the Gini impurity score (Breiman, 1984) or the information gain (Quinlan, 2014) rather than choosing a global objective function. At every decision node, the feature with the highest quality is chosen as the splitting feature. This process is repeated until a termination criteria is reached. One such criteria often used is the minimum support of each leaf. Trees can then be postprocessed with pruning methods. Branch and Bound Optimization Among the many methods for globally optimizing trees, Branch-and-bound approaches with dynamic programming are state of the art for scalability, because they exploit the structure of decision trees (Costa & Pedreira, 2023; Lin et al., 2020; Demirovic´ et al., 2022; McTavish et al., 2022; Aglin et al., 2020). While many other methods exist for optimizing trees, such as MIP solvers (Bertsimas & Dunn, 2017; Verwer & Zhang, 2019), we focus our discussion and comparison of globally optimal decision tree methods on the currently fastest types of approaches - dynamic programming with branch and bound (DPBnB). These approaches search through the space of decision trees while tracking lower and upper bounds of the overall objective at each split to reduce the search space.

They can find optimal trees on medium-sized datasets with tens of features and shallow maximum tree depths (Sullivan et al., 2024; Aglin et al., 2020; Lin et al., 2020; Demirovic´ et al., 2022). Aglin et al. (2020) uses a DPBnB method with advanced caching techniques to find optimal decision trees, though it does not explicitly optimize for sparsity. In contrast, Lin et al. (2020); Hu et al. (2019) use a DPBnB approach to find a tree that optimizes a weighted combination of empirical risk and sparsity, defined by the number of leaves in the tree. McTavish et al. (2022) further enhances this approach by incorporating smart guessing strategies to construct tighter lower bounds for DPBnB, resulting in computational speedups. Demirovic et al. ´ (2022) extends the work of Aglin et al. (2020) by focusing on finding the optimal tree with a hard constraint on the number of permissible nodes, using advanced caching techniques and an optimized depth-2 decision tree solver. Mazumder et al. (2022) addresses continuous features by defining lower and upper bounds based on quantiles of feature distributions.

However, their method is applicable only to shallow optimal trees with depth ≤ 3, limiting its utility in scenarios with higher-order feature interactions.

Lookahead Trees Some older approaches to greedy decision tree optimization consider multiple levels of splits before selecting the best split at a given iteration (Norton, 1989). That is, unlike the other greedy approaches, these approaches do not pick the split that optimizes a heuristic immediately. Instead, they pick a split that sets up the best possible heuristic value on the following split. These approaches still focus on locally optimizing a heuristic measure that is not necessarily aligned with a global objective. By contrast, our method selects splits to directly optimize the sparse misclassification rate of the final tree. We globally optimize the search up to the specified lookahead depth, switching to heuristics only when deciding splits past our lookahead depth. In so doing, our method largely avoids the pathology noted in Murthy & Salzberg (1995), who note cases where their own lookahead approach results in a substantially worse tree than one constructed with a standard greedy approach. For our method, it is provably impossible for a fully greedy entropy-based method with the same constraints as our approach to achieve a better training set objective than our approach. (See Theorem A.1) Other Hybrid Methods Several other approaches are compatible with branch and bound techniques. Blanc et al. (2024) seek to bridge the gap between greedy and optimal decision trees by selecting a fixed subset of the top k feature splits for each sub-problem. However, this framework does not explicitly account for sparsity. Further, the method is limited by using a *global* setting for search precision: the approach considers the same number of candidate splits at each subproblem. As we show in our experiments, there is merit to tailoring the level of search precision to parts of the search space where it is most needed. The Blossom algorithm (Demirovic et al. ´ , 2023) traverses a branch and bound dependency graph structure while using greedy heuristics to guide the search order. Relative to our approach, this algorithm optimizes from the bottom up, starting with greedy splits at each level, then optimizing the splits furthest from the root first. This choice guarantees eventual optimality while giving anytime behavior, but misses out on leveraging the property motivating this work - that greedy splits are most detrimental near the top of the tree. Like the approach of Blanc et al. (2024), Blossom also does not account for sparsity. There are a few methods that use probabilistic search techniques to optimize trees. Sullivan et al. (2024) take a Bayesian approach, finding the maximum-a-posteriori tree by optimizing over an AND/OR graph, akin to the graph used in earlier branch-and-bound methods like that of Lin et al. (2020). Although their method demonstrates strong performance, their experimental results reveal that it is not responsive to sparsity-inducing hyperparameters - accordingly, we found in our experiments that the method struggles to optimize for sparsity. Recent work by Chaouki et al. (2024) devises a Monte Carlo Tree Search algorithm using Thompson sampling to enable online, adaptive learning of sparse decision trees. We show that our method achieves superior performance and sparsity on all datasets tested.

## 3. Preliminaries

We consider a typical supervised machine learning setup, with a dataset D = {(xi, yi)}
N
i=1 sampled from a distribution D, where xi ∈ {0, 1}
K is a binary feature vector and yi ∈ {0, 1} is a binary label.1 Let F be the set of features.

Define D(f) as the subset of D consisting of all samples where feature f ∈ F is 1 (and D(¯f) as the subset where feature f is 0). Let D+ and D− denote the set of examples with positive and negative labels, respectively.

Node specific notation Let Dt be the support set of node t in a tree (i.e., the set of training examples assigned to this node); we call each Dt a *subproblem*. Let ft ∈ F be the feature we split on at t. Let Dt(ft) and Dt(¯ft) be the support sets of the children of t. Unless stated otherwise, a greedy split at node t chooses the feature f that maximizes the information gain, which is equivalent to solving:

$$f_{t}=\operatorname*{min}_{f\in\mathcal{F}}{\frac{|D_{t}(f)|}{|D_{t}|}}H\!\left({\frac{|D_{t}^{+}(f)|}{|D_{t}(f)|}}\right)+{\frac{|D_{t}({\bar{f}})|}{|D_{t}|}}H\!\left({\frac{|D_{t}^{+}({\bar{f}})|}{|D_{t}({\bar{f}})|}}\right)$$
$$\operatorname{g}p-(1-p)\log$$
$ H(p)=-$
$\mathbb{I}-p$)
with entropy H(p) = −p log p − (1 − p) log(1 − p). Tree specific notation We now briefly discuss sparse greedy and optimal trees. We define Tg(*D, d, λ*) to be a decision tree of depth at most d trained greedily on D with sparsity penalty λ. Intuitively, this sparse greedy algorithm will make a split at a node only when the gain in overall accuracy is greater than λ. Algorithm 4 in the Appendix illustrates this procedure. Modern methods such as Lin et al. (2020); McTavish et al. (2022), on the other hand, find a tree T in the space of decision trees T that solves the following optimization problem:

$$\mathcal{L}^{*}(D,d,\lambda)=\min_{T\in\mathcal{T}}L(T,D,\lambda)\text{s.t.depth}(T)\leq d\tag{1}$$ $$=\min_{T\in\mathcal{T}}\sum_{i=1}^{|D|}\frac{1}{N}\Big{(}l\big{(}T(\mathbf{x}_{i}),y_{i}\big{)}+\lambda S(T)\Big{)}\text{s.t.depth}(T)\leq d$$

where L(*T, D, λ*) is the regularized loss of tree T on dataset (or data subset) D, S(T) is the number of leaves in T, ℓ(T(x), y) is the loss incurred by T in its prediction on x 1The discussions and methods in this paper can trivially be extended to multiclass problems; we focus our discussion and evaluation of the methodology on binary labels.

(for this paper, we set ℓ to be the 0-1 loss), and N is the global dataset size. As discussed in Section 2, the fastest contemporary methods solve this problem using a branchand-bound approach (Costa & Pedreira, 2023; Lin et al., 2020; Demirovic et al. ´ , 2022; McTavish et al., 2022). Rashomon Sets Our work is motivated by the properties of near-optimal decision trees and allows for scalable approximation of that set. Xin et al. (2022) define the Rashomon set, denoted by R(*D, λ, ϵ, d*), as the collection of all trees whose objective is within ϵ of the minimum value in Equation 1. Formally:

$$\mathcal{R}(D,\lambda,\epsilon,d)=\{T\in\mathcal{T}:L(T,D,\lambda)\leq\mathcal{L}^{*}(D,d,\lambda)+\epsilon$$ $$\wedge\text{depth}(T)\leq d\}.\tag{2}$$

In Section 4, we use Rashomon sets to investigate properties of near-optimal trees. Rashomon sets can be used for a range of downstream tasks (Rudin et al., 2024); one crucial task is the measurement of variable importance over a set of near-optimal models instead of only for a single model (Donnelly et al., 2023; Fisher et al., 2019). Reliable variable importance measures in this setting rely on minimal feature selection prior to computing the Rashomon set and minimal constraints on the tree's depth to allow high-order interactions. Our approach can be used to accelerate the computation of a Rashomon set, supporting the feasibility of these approaches. Branch and Bound Given a depth budget d, branch and bound with a sparsity penalty (Lin et al., 2020; McTavish et al., 2022) finds the optimal loss L
∗(*D, d, λ*) that minimizes Equation 1. The key insight behind branch and bound is that the optimal solution for dataset D at depth d
′ has a dependency on the optimal solution for datasets D(f) and D(
¯f) at depth d
′ − 1, for each f ∈ F. Starting from the root, branch and bound algorithms consider different candidate features, f, on which to split in the process of determining the objective. As candidates are considered, we identify the subproblems we encounter by the subset of data they relate to and their remaining depth. We track current upper and lower bounds of subproblems in order to prune parts of the search space as we explore it. In particular, if our lower bounds on L
∗(Dt(f1), d′ − 1, λ) and L
∗(Dt(¯f1), d′ − 1, λ)
sum to a larger value than the sum of upper bounds on L
∗(Dt(f2), d′ − 1, λ) and L
∗(Dt(¯f2), d′ − 1, λ), for example, then we have proven that f1 is not the minimizing split for dataset D.

L(Dt, d′, λ) can always start with an upper bound of ub =
λ + min |D−
t | |Dt|
,
|D+
t | |Dt|
. A universal lower bound is λ. To get a tighter lower bound, if d
′ > 0, the lower bound can start at min(ub, 2λ), since either L(Dt, d′, λ) = ub, or the objective will be the sum of two other L calls, both of which must necessarily have cost at least λ. These upper and lower bounds are then updated as we explore a graph structure containing these subproblems. Once these bounds have converged, and we know the value of L(D, d′, λ) for the whole dataset D, we can extract the optimal tree by simply tracking the feature f that leads to the optimal score for D and then successively track the splits for the optimal value with respect to D(f) and D(
¯f), and so on.

Discretization Our algorithm will assume feature vectors to be binary, i.e., xi ∈ {0, 1}
K. Real-world datasets often have features that require discretization to fit our setting. While some methods preserve optimality (e.g., splitting at the mean between unique values in the training set), others such as bucketization (described and proven to be suboptimal in Lin et al., 2020), binning into quantiles, and feature engineering reduce the search space at the cost of optimality. In our experiments, we use threshold guessing (McTavish et al., 2022), which sacrifices optimality with respect to a real-valued dataset but maintains theoretical and empirical guarantees relative to a reference decision tree ensemble.

## 4. Algorithm Motivation

A key motivating property of SPLIT is that we can find high quality trees even when splitting greedily far from the root of the tree. To support this intuition, we empirically investigate how frequently near optimal trees behave greedily far from the root. To do so, we first generate the Rashomon set of decision trees for various values of sparsity penalty λ and Rashomon bound ϵ. Let T ∈ R(*D, λ, ϵ, d*) be a tree in the Rashomon set, and let n ∈ T be any node in T. Then, we compute the fraction of all nodes at a given level ℓ ≤ d (where level 0 corresponds to the root) that were greedy (by which we mean that the split at this node in the tree is optimal with respect to information gain). This corresponds to the following proportion:

$$\sum_{T\in\mathcal{R}(D,\lambda,\epsilon,d)}\sum_{n\in T}\mathbb{1}\left[n\text{is greedy}\wedge\text{level}(n)=\ell\right]\tag{3}$$

Figure 2 shows the results of this investigation for 6 different datasets for different values of ϵ and λ. We note that there is a general increase in percentage of greedy splits as one goes deeper in the tree.

cov erty pe Level 0 Level 1 Level 2 Level 3 2 4 1e 3 0.01 0.02 0.03 2 4 1e 3 2 4 1e 3 2 4 1e 3 neth erla nds 0.5 1.0 1e 2 0.01 0.02 0.03 0.750 0.5 1.0 1e 2 0.5 1.0 1e 2 0.5 1.0 1e 2 Pro por tion of Gre edy S
plit s 0.5 1.0 1e 2 0.01 0.02 0.03 com pas 0.5 1.0 1e 2 0.5 1.0 1e 2 0.5 1.0 1e 2 2.5 5.0 7.5 1e 3 0.01 0.02 0.03 bike 0.375 2.5 5.0 7.5 1e 3 2.5 5.0 7.5 1e 3 2.5 5.0 7.5 1e 3 2 4 1e 3 0.01 0.02 0.03 hel oc 2 4 1e 3 2 4 1e 3 2 4 1e 3 2 4 6 1e 3 2 4 6 1e 3 2 4 6 1e 3 0.000 2 4 6 1e 3 0.01 0.02 0.03 adu lt
Additional motivating empirical results for using greedy splits far from the root of the tree are provided in Appendix A.2.

## 5. Algorithm Details 5.1. Sparse Lookahead For Interpretable Trees (Split)

We now formalize our main algorithm, SPLIT, which takes as input a *lookahead depth* parameter. This is the depth up to which a search algorithm optimizes over all combinations of feature splits, conditioned on splits beyond this depth behaving greedily. Our algorithm exploits the fact that subproblems closer to the leaves exhibit smaller optimality gaps than those at the root, providing a mechanism to trade off among runtime, accuracy, and sparsity.

Formulating the optimization problem Concretely, for a given depth budget d, lookahead depth dl < d, and feature set F, we **first** solve the following recursive equation:

$$0,d^{\prime},\lambda)=$$
L(D, d′, λ) =
$$\mathcal{L}(D,a^{\prime},\lambda)=$$ $$\left\{\min\left\{\lambda+\frac{|D^{-}|}{N},\lambda+\frac{|D^{+}|}{N},\right.\right.$$ $$\left.\min_{f\in\mathcal{F}}\left\{L\Big{(}T_{g}\big{(}D(f),d^{\prime},\lambda\big{)}\Big{)}+L\Big{(}T_{g}\big{(}D(\bar{f}),d^{\prime},\lambda\big{)}\Big{)}\right\}\right\}$$ $$\left.\min\left\{\lambda+\frac{|D^{-}|}{N},\lambda+\frac{|D^{+}|}{N},\right.\right.$$ $$\left.\min_{f\in\mathcal{F}}\left\{\mathcal{L}\Big{(}D(f),d^{\prime}-1,\lambda\Big{)}+\mathcal{L}\Big{(}D(\bar{f}),d^{\prime}-1,\lambda\Big{)}\right\}\right\}$$ $$\left.\text{if}d^{\prime}>d-d_{l}.\right.\tag{4}$$
$\eqref{eq:walpha}$. 
$$\mathrm{nds}(D,\,d_{l},\,d,\,c)$$

Where N is the size of the dataset at the root. We can con-

$${\mathrm{{get}}}_{-}{\mathrm{Don}}$$

Algorithm 1 get bounds(D, dl, d, d
′, N) → lb, ub Require: D, dl, d, d
′, N {support, lookahead depth, current search depth, maximum search depth, size of full dataset in GOSDT call}
1: if d
′ = dl **then**
2: Tg = Greedy(D, d − dl, λ) {Find greedy tree rooted at D (Alg 4 in the Appendix)}
3: S(Tg) = \# Leaves in Tg 4: α ← 1N
P(x,y)∈D 1[y ̸= Tg(x)] + λS(Tg)
5: lb ← α 6: ub ← α {subproblem solved because ub = lb}
7: **else** {use basic initial bounds} 8: lb ← 2λ 9: ub ← λ + min n|D−| N,
|D+| N
o 10: **end if** 11: **return** lb,ub {Return Lower and Upper Bounds}
strain the search space to include only greedy trees past the lookahead depth by modifying the lower and upper bounds used in branch and bound (see Algorithm 1). In particular, sub-problem nodes initialized at depths up to the lookahead depth are assigned initial lower and upper bounds equivalent to that in GOSDT (Lin et al., 2020) (see Section 2). At the lookahead depth, however, the lower and upper bounds for a subproblem are fixed to be the loss of a greedy subtree trained on that subproblem. After these bound assignments, our algorithm uses the GOSDT algorithm with these new bounds to solve Equation 4 - this is summarized by Lines 1-2 in Algorithm 2. We defer more details of the GOSDT algorithm to Section A.12 in the Appendix.

Postprocessing with Optimal Subtrees Once we have solved Equation 4, we do not need to use greedy sub-trees past the lookahead depth. We can improve our approach by Algorithm 2 SPLIT(ℓ, D, λ, dl, d, p) Require: ℓ, D, λ, dl, d , p {loss function, samples, regularizer, lookahead depth, depth budget, postprocess flag}
1: ModifiedGOSDT = GOSDT reconfigured to use get **bounds** (Algorithm 1) whenever it encounters a new subproblem 1: t*lookahead* = ModifiedGOSDT(*ℓ, D, λ, d*l) {Call ModifiedGOSDT with depth budget dl}
2: if p **then** {Fill in the leaves of this prefix} 3: for leaf u ∈ t*lookahead* do 4: du = depth of leaf 5: D(u) = subproblem associated with u 6: λu = λ|D| |D(u)|
{Renormalize λ for the subproblem in question}
7: tu = GOSDT(D(u), d−du, λu) {Find the optimal subtree for D(u)}
8: if tu is not a leaf **then** 9: Replace leaf u with sub-tree tu 10: **end if** 11: **end for** 12: **end if**
13: **return** t*lookahead* replacing these subtrees with fully optimal decision trees. Lines 3-9 in Algorithm 2 illustrate this. Thus, the performance of the lookahead tree with the aforementioned greedy subtrees is just an upper bound on the objective of the tree our method ultimately finds. Note that the renormalization in line 6 of Algorithm 2 ensures that the λ penalty stays proportional to the penalty for each misclassified point. Our objective (Equation 1) assigns a 1 N
penalty for each misclassification, where N is the size of the full dataset with which GOSDT was called. If the original dataset is D, then when we call GOSDT on any descendent subproblem D(u), our penalty per misclassification goes up by a factor of |D| |D(u)|
. We need to scale λ appropriately to stay proportional to the original dataset D.

## 5.2. Licketysplit: Polynomial-Time Split

We present a polynomial-time variant of SPLIT, called LicketySPLIT, in Algorithm 3. This method works by recursively applying SPLIT with lookahead depth 1. That is, we first find the optimal initial split for the dataset, given that we are fully greedy henceforth. Then, during postprocessing, instead of doing what SPLIT would do —running a fully optimal decision tree algorithm on the root's left and right subproblems —we run LicketySPLIT recursively on these two subproblems. We stop considering further calls to LicketySPLIT for a subproblem if SPLIT returns a leaf instead of making splits (either due to the depth limit or λ).

Algorithm 3 LicketySPLIT(ℓ, D, λ, d) Require: ℓ, D, λ, d {loss function, samples, regularizer, full depth}
0: t*lookahead* = SPLIT(ℓ, D, λ, 1*, d,* 0) {Call SPLIT with lookahead depth 1 and no post-processing}
1: if tlookahead is not a leaf **then** 2: for child u ∈ tlookahead do 3: D(u) = subproblem associated with u 4: λu = λ|D| |D(u)|
{Renormalize λ for the subproblem in question}
5: tu = LicketySPLIT(ℓ, D(u), λu, d − 1) 6: Replace u with subtree tu 7: **end for** 8: **end if**
9: **return** tlookahead

## 5.3. Resplit: Rashomon Set Estimation With Split

At the cutting edge of compute requirements for decision tree optimization is the computation of Rashomon sets of decision trees. Xin et al. (2022) compute a Rashomon set of all near-optimal trees, based on the GOSDT algorithm (Lin et al., 2020). This task generates an extraordinary number of trees and has high memory and runtime costs. To make this tractable, Xin et al. (2022) leverage depth constraints and feature selection from prior work to reduce the depth and set of features considered (McTavish et al., 2022). While necessary for scalability, this can prevent exploration of near-optimal models across all features or at greater decision tree depths. Both factors are relevant for work on variable importance based on Rashomon sets (Fisher et al., 2019; Dong & Rudin, 2020; Donnelly et al., 2023). We leverage SPLIT as a way to dramatically improve scalability of Rashomon set computation, reliably approximating the full Rashomon set and allowing feasible exploration while relaxing or removing depth and feature constraints. Our algorithm, RESPLIT, is described in Appendix A.10; it first leverages SPLIT as a subroutine to obtain a set of prefix trees such that completing them greedily up to the depth budget would result in an ϵ approximation of the optimal solution to Equation 4. At each leaf of each prefix tree, it calls TreeFARMS (Xin et al., 2022) to find a large set of shallow subtrees that are at least as good as being greedy, yielding an approximate Rashomon set computed much faster than state of the art. We also show a novel indexing mechanism to query RESPLIT trees in Appendix A.11.

## 6. Theoretical Analysis Of Runtime And Optimality

We present theoretical results establishing the performance and scalability of our algorithms. All proofs, including additional lemmas not described below, are in Appendix Section A.8. Even without the speedups discussed in Section 5.2, Algorithm 2 is quite scalable. Theorem 6.1 shows the asymptotic analysis of the algorithm, with and without caching. Note that the default behaviour of Algorithm 2 is to cache repeated sub-problems. Theorem 6.1 (Runtime Complexity of SPLIT). *For a* dataset D with k features and n *samples, depth constraint* d such that d ≪ k, and lookahead depth 0 ≤ dl < d, Algorithm 2 *has runtime* On(d − dl)k dl+1 + nkd−dl. If we cache repeated subproblems, the runtime reduces to O
n(d−dl)k dl+1 dl! +
nkd−dl
(d−dl)!.

This algorithm is linear in sample size and, because dl < d and d − dl < d, is exponentially faster than a globally optimal approach, which searches through O((2k)
d) subproblems in the worst case. Corollaries 6.2 and 6.3 show that, compared to globally optimal approaches, we see substantial improvements in runtime when lookahead depth is around half the global search depth. Corollary 6.2 (Optimal Lookahead Depth for Minimal Runtime). *The optimal lookahead depth that minimizes the* asymptotic runtime of Algorithm 2 is dl =
(d−1)
2*for large* k*, regardless of whether subproblems are cached.*
Corollary 6.3 (Runtime Savings of SPLIT Relative to Globally Optimal Approaches). Asymptotically, under the same conditions as Theorem 6.1 and with caching repeated subproblems, Algorithm 2 *saves a factor of* O
k d−1 2 d 2
!

in runtime relative to globally optimal approaches (e.g., GOSDT). Theorem 6.4 describes the runtime complexity of our LicketySPLIT method from Section 5.2, showing that it can be even faster than Algorithm 2 (indeed, achieving low-order polynomial runtime). Theorem 6.4 (Runtime Complexity of LicketySPLIT). For a dataset D with k features and n *samples, and for depth* constraint d, Algorithm 3 *has runtime* O(nk2d 2).

We can thus use Algorithm 3 to leverage a recursive search while remaining comfortably polynomial. This is a dramatic improvement to asymptotic scalability relative to globally optimal decision tree construction methods, which solve an NP-hard problem. Theorem 6.5 (SPLIT Can be Arbitrarily Better than Greedy). For every ϵ > 0 and depth budget d, there exists a data distribution D and sample size n for which, with high probability over a random sample S ∼ Dn*, Algorithm* 2 *with* dl =
d−1 2achieves accuracy at least 1 − ϵ *but a pure* greedy approach achieves accuracy at most 12 + ϵ.

Theorem 6.5 shows that Algorithm 2 can arbitrarily outperform greedy methods in accuracy, even when we choose its minimum runtime configuration of dl =
d−1 2. We prove a similar claims for LicketySPLIT and RESPLIT in the appendix (see Theorems A.7 and A.6).

## 7. Experiments

Our experiments provide an evaluation of decision trees, considering aspects of performance, interpretability, and training budget. To this end, our evaluation addresses the following questions:
1. How fast are SPLIT and LicketySPLIT compared to unmodified GOSDT?

2. Are SPLIT and LicketySPLIT able to produce trees that lie on the frontier of sparsity, test loss performance, and training time?

3. How good is the Rashomon set approximation produced by RESPLIT?

For all experiments below we set the depth budget of our algorithms to 5. The lookahead depth for Algorithm 2 is set to 2 since, from Corollary 6.2, this produces the lowest runtime for the chosen depth budget. We defer more details of our experimental setup and datasets to Appendix A.7.4. Appendix A.3 has additional evaluations of our methods.

## 7.1. How Do Our Algorithms Compare To Gosdt?

Our first experiments support the claim that our method is significantly faster than GOSDT whilst achieving similar regularized test losses. This is shown in Figure 3. Here, we vary the sparsity penalty, λ, which is a common input to all algorithms in this figure, and compute the regularized test objective from Equation 1 for each value of λ. We set a timeout limit of 1200 seconds for GOSDT, after which it gives the best solution found so far. We note two regimes:
- When all methods have lower regularized objective values (left side of each plot), **our methods are orders of** magnitude faster than GOSDT. For instance, on the Bike dataset, SPLIT has training times of ∼10 seconds, while GOSDT runs for ∼103seconds. LicketySPLIT
takes merely a second in most cases. This is the regime most relevant to our algorithms.

- When the optimal objective is high and the tree is supersparse (right side of each plot), SPLIT and Lickety- SPLIT have small overhead costs and can be slower, because we need to train a greedy tree for each subproblem encountered at the lookahead depth in order to initialize bounds via Algorithm 1. However, in this regime, all methods already have runtimes of ∼1 second, so the extra overhead cost is insignificant. This is especially seen in the COMPAS and Netherlands datasets.

## 7.2. Characterising The Frontier Of Test Loss, Sparsity, And Runtime

Figure 4 characterises the frontier of training time, sparsity, and test loss for several algorithms. Here, we vary hyperparameters associated with each algorithm to produce trees of varying sparsity levels (where sparsity is the number of leaves). We see that there exists a frontier between test loss and sparsity, and different methods lie on different parts of the frontier. To maximize interpretability and accuracy, we want a tree to lie in the bottom left corner of the frontier, within the highlighted red rectangle. Out of all algorithms tested, ours consistently lie on the frontier and in the red rectangle. Alongside state of the art performance, our algorithms are often over 100× **faster** than their contemporaries.

For more datasets, see Figure 5 in the Appendix.

## 7.3. Rashomon Set Approximation

We now show that RESPLIT enables fast, accurate approximation of the Rashomon set of near-optimal trees, while scaling much more favorably than state-of-the-art method TreeFARMS (Xin et al., 2022). We demonstrate that variable importance conclusions using RID (Donnelly et al., 2023) remain almost identical under RESPLIT, relative to the full Rashomon set. That is, RESPLIT allows accurate summary statistics of the full Rashomon set to be computed at greater depths and over more binary features while enhancing scalability. Table 1 shows computation of RID with and without RESPLIT. RESPLIT enables 10 − 20× faster variable importance computation. Furthermore, the correlation between variable importances is very close to 1, suggesting that RESPLIT trees serve as good proxies for estimating importances derived from the complete Rashomon set. Table 2 also shows that most of the trees output by RESPLIT lie in the true Rashomon set or very close to it. Table 1. Table summarizing the advantages of RESPLIT. The first 2 columns show the time taken to compute all bootstrapped Rashomon sets for the Rashomon Importance Distribution (RID)
(Donnelly et al., 2023) with and without RESPLIT. \# of bootstrapped datasets = 10, λ = 0.02, ϵ = 0.01, depth budget 5, lookahead depth 3. The last column shows the Pearson correlation between variable importances computed by RID and RID + RE- SPLIT. There is nearly perfect correlation seen in every case.

| Dataset     | Full (s)   | RESPLIT (s)   | τ     |
|-------------|------------|---------------|-------|
| COMPAS      | 152        | 18            | 1.0   |
| Spambase    | 2659       | 154           | 0.930 |
| Netherlands | 4255       | 216           | 0.932 |
| HELOC       | 5564       | 337           | 0.979 |
| HIV         | 9273       | 388           | 0.959 |
| Bike        | 14330      | 194           | 0.999 |

Test Loss vs # Leaves for Different Decision Tree **Algorithms**
bike SPLIT (ours)
LicketySPLIT (ours)
greedy topk (dl8.5)
tsdt gosdt maptree murtree netherlands adult 0 10 20 30 Number of Leaves 0.30 0.35 0.40 Te st
 
Lo s s 0.4 0 10 20 30 Number of Leaves 0.175 0.200 0.225 0.2 0 10 20 30 Number of Leaves 10 1 10 1 10 3 Training Time (s)
0.17 0.18 0.19 Test Loss vs Training Time for Models in the Red Box 0.32 Te st Lo s s 0.20 0.30 0.15 10 1 10 1 10 3 Training Time (s)
10 1 10 1 10 3 Training Time (s)

| Dataset     | Precision        | Precision (Slack .01)   |
|-------------|------------------|-------------------------|
| Bike        | 0.974 (370/380)  | 1.000                   |
| COMPAS      | 1.000 (27/27)    | 1.000                   |
| HELOC       | 0.974 (528/542)  | 1.000                   |
| HIV         | 0.528 (243/460)  | 0.984                   |
| Netherlands | 0.911 (102/112)  | 1.000                   |
| Spambase    | 0.597 (850/1422) | 0.933                   |

Table 2. Proportion of RESPLIT Trees in the true Rashomon set (precision) and within at most .01 loss of being in the set. Most of the trees output by RESPLIT end up being in the Rashomon set. Trees which are not in the Rashomon set are almost always very close to being in it. We employ the same parameters as Table 1.

## 8. Conclusion

We introduced SPLIT, LicketySPLIT, and RESPLIT, a novel family of decision tree optimization algorithms. At their core, these algorithms perform branch and bound search up to a lookahead depth, beyond which they switch to greedy splitting. Our experimental results show dramatic improvements in runtime compared to state of the art algorithms, with negligible loss in accuracy or sparsity. RESPLIT also scalably finds a set of near-optimal trees without adversely impacting downstream variable importance tasks. Future work could explore conditions under which subproblems exhibit large optimality gaps, offering new insights for efficient decision tree and Rashomon set optimization.

## Acknowledgements

We acknowledge funding from the National Institutes of Health under 5R01-DA054994, the National Science Foundation under award NSF 2147061, and through the Department of Energy under grant DE-SC0023194. We thank Srikar Katta, Jon Donnelly, Zachery Boner, Yixiao Wang, and Zakk Heile for helpful discussions and feedback throughout this project.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

Aglin, G., Nijssen, S., and Schaus, P. Learning optimal decision trees using caching branch-and-bound search. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pp. 3146–3153, 2020.

Arrieta, A. B., D´ıaz-Rodr´ıguez, N., Del Ser, J., Bennetot, A.,
Tabik, S., Barbado, A., Garc´ıa, S., Gil-Lopez, S., Molina, ´ D., Benjamins, R., et al. Explainable artificial intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI. *Information Fusion*, 58:82–115, 2020.

Balcan, M.-F. and Sharma, D. Learning accurate and interpretable decision trees. In Proceedings of the Fortieth Conference on Uncertainty in Artificial Intelligence, UAI
'24. JMLR.org, 2024.

Bertsimas, D. and Dunn, J. Optimal classification trees.

Machine Learning, 106:1039–1082, 2017.

Blanc, G., Lange, J., and Tan, L.-Y. Top-down induction of decision trees: rigorous guarantees and inherent limitations. *arXiv preprint arXiv:1911.07375*, 2019.

Blanc, G., Lange, J., Pabbaraju, C., Sullivan, C., Tan, L.-
Y., and Tiwari, M. Harnessing the power of choices in decision tree learning. Advances in Neural Information Processing Systems, 36, 2024.

Breiman, L. *Classification and regression trees*. Routledge, 1984.

Breiman, L. Statistical modeling: The two cultures (with comments and a rejoinder by the author). Statistical Science, 16(3):199–231, 2001.

Chaouki, A., Read, J., and Bifet, A. Online learning of decision trees with Thompson sampling. In Dasgupta, S., Mandt, S., and Li, Y. (eds.), *Proceedings of The 27th* International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings of Machine Learning Research, pp. 2944–2952. PMLR, 02–04 May 2024.

Chatzigeorgiou, I. Bounds on the lambert function and their application to the outage analysis of user cooperation. IEEE Communications Letters, 17(8):1505–1508, August 2013.

Costa, V. G. and Pedreira, C. E. Recent advances in decision trees: An updated survey. *Artificial Intelligence Review*, 56(5):4765–4800, 2023.

Demirovic, E., Lukina, A., Hebrard, E., Chan, J., Bailey, ´
J., Leckie, C., Ramamohanarao, K., and Stuckey, P. J. Murtree: Optimal decision trees via dynamic programming and search. *Journal of Machine Learning Research*,
23(26):1–47, 2022.

Demirovic, E., Hebrard, E., and Jean, L. Blossom: an ´
anytime algorithm for computing optimal decision trees.

In *International Conference on Machine Learning*, pp.

7533–7562. PMLR, 2023.

Dong, J. and Rudin, C. Exploring the cloud of variable importance for the set of all good models. *Nature Machine* Intelligence, 2(12):810–824, 2020.

Donnelly, J., Katta, S., Rudin, C., and Browne, E. P. The rashomon importance distribution: Getting RID of unstable, single model-based variable importance. In Advances in Neural Information Processing Systems, 2023.

Fanaee-T, H. and Gama, J. Event labeling combining ensemble detectors and background knowledge. Progress in Artificial Intelligence, pp. 1–15, 2013. doi: 10.1007/
s13748-013-0040-3.

FICO. Home equity line of credit (heloc)
dataset. https://community.fico.com/s/ explainable-machine-learning-challenge, 2018. FICO Explainable Machine Learning Challenge.

Fisher, A., Rudin, C., and Dominici, F. All models are wrong, but many are useful: Learning a variable's importance by studying an entire class of prediction models simultaneously. *Journal of Machine Learning Research*, 20(177):1–81, 2019.

Hu, X., Rudin, C., and Seltzer, M. Optimal sparse decision trees. In Advances in Neural Information Processing Systems, volume 32, pp. 7265–7273, 2019.

Izza, Y., Ignatiev, A., and Marques-Silva, J. On tackling explanation redundancy in decision trees. Journal of Artificial Intelligence Research, 75:261–321, 2022.

Lin, J., Zhong, C., Hu, D., Rudin, C., and Seltzer, M. Generalized and scalable optimal sparse decision trees. In International Conference on Machine Learning, pp. 6150–
6160. PMLR, 2020.

Loczi, L. Explicit and recursive estimates of the Lambert ´
W function. *arXiv 2008.06122*, 2021.

Mazumder, R., Meng, X., and Wang, H. Quant-BnB: A scalable branch-and-bound method for optimal decision trees with continuous features. In International Conference on Machine Learning, volume 162, pp. 15255–15277. PMLR, 17–23 Jul 2022.

McTavish, H., Zhong, C., Achermann, R., Karimalis, I.,
Chen, J., Rudin, C., and Seltzer, M. Fast sparse decision tree optimization via reference ensembles. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pp. 9604–9613, 2022.

Murthy, S. and Salzberg, S. Lookahead and pathology in decision tree induction. In *International Joint Conference* on Artificial Intelligence, pp. 1025–1033, 1995.

Norton, S. W. Generating better decision trees. In International Joint Conference on Artificial Intelligence, 1989.

Piltaver, R., Lustrek, M., Gams, M., and Martin ˇ ciˇ c-Ip ´ siˇ c, S. ´
What makes classification trees comprehensible? Expert Systems with Applications, 62:333–346, 2016.

Quinlan, J. R. *C4.5: programs for machine learning*. Elsevier, 2014.

Rudin, C., Chen, C., Chen, Z., Huang, H., Semenova, L.,
and Zhong, C. Interpretable machine learning: Fundamental principles and 10 grand challenges. *Statistics* Surveys, 16:1–85, 2022.

Rudin, C., Zhong, C., Semenova, L., Seltzer, M., Parr, R.,
Liu, J., Katta, S., Donnelly, J., Chen, H., and Boner, Z. Amazing things come from having many good models. In Proceedings of the International Conference on Machine Learning, 2024.

Sullivan, C., Tiwari, M., and Thrun, S. MAPTree: Beating "optimal" decision trees with Bayesian decision trees. Proceedings of the AAAI Conference on Artificial Intelligence, 38(8):9019–9026, March 2024.

van der Linden, J. G., Vos, D., de Weerdt, M. M., Verwer, S., and Demirovic, E. Optimal or greedy decision trees? ´ revisiting their objectives, tuning, and performance. arXiv preprint arXiv:2409.12788, 2024.

Verwer, S. and Zhang, Y. Learning optimal classification trees using a binary linear program formulation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 1625–1632, 2019.

Xin, R., Zhong, C., Chen, Z., Takagi, T., Seltzer, M., and Rudin, C. Exploring the whole rashomon set of sparse decision trees. Advances in Neural Information Processing Systems, 35:14071–14084, 2022.

Zhou, Q., Liao, F., Mou, C., and Wang, P. Measuring interpretability for different types of machine learning models. In *Trends and Applications in Knowledge Discovery and* Data Mining: PAKDD 2018 Workshops, BDASC, BDM,
ML4Cyber, PAISI, DaMEMO, Melbourne, VIC, Australia, June 3, 2018, Revised Selected Papers 22, pp. 295–308.

Springer, 2018.

## A. Appendix

A.1. Further Comparisons With Other Methods A.1.1. MORE DATASETS WITH DEPTH 5 TREES
In Section 7 of the paper, we showed results for three datasets. Here, we evaluate SPLIT, LicketySPLIT, and its contemporaries on 6 additional datasets. All datasets were evaluated on three random 80-20 train-test splits of the data, with the average and standard error reported. Results are in Figure 5. Note that Covertype has smaller error bars because the dataset size is much larger - it has ∼ 5 × 106examples, while COMPAS and HELOC have only ∼ 104examples.

compas SPLIT (ours) LicketySPLIT (ours)
greedy topk (dl8.5)
tsdt gosdt maptree murtree covertype heloc 0 10 20 30 Number of Leaves 0.30 0.35 0.40 0.45 0.35 Test Loss 0.4 0.30 0.25 0 10 20 30 Number of Leaves 0.3 0 10 20 30 Number of Leaves Test Loss vs Training Time for Models in the Red Box 0.32 0.34 0.36 Test Loss 0.24 0.25 0.26 10 1 10 1 10 3 Training Time (s)
0.29 0.30 0.31 10 1 10 1 10 3 Training Time (s)
10 1 10 1 10 3 Training Time (s)
bank 0 10 20 30 Number of Leaves 0.000 0.025 0.050 0.075 hypothyroid 0.1 0.2 0.3 0.4 spambase Test Loss 0.3 0.2 0 10 20 30 Number of Leaves 0.1 0 10 20 30 Number of Leaves Test Loss vs Training Time for Models in the Red Box 0.10 0.11 0.12 0.13 10 1 10 1 10 3 Training Time (s)
0.00 0.01 0.02 0.03 0.10 0.12 0.14 Test Loss 10 1 10 1 10 3 Training Time (s)
10 1 10 1 10 3 Training Time (s)
A.1.2. WHAT ABOUT DEPTH 4 TREES?

In this section, we perform the same evaluation as above, but with depth 4 trees. We set the lookahead depth as 2.

Test Loss vs # Leaves for Different Decision Tree **Algorithms**
bike SPLIT (ours) LicketySPLIT (ours)
greedy topk (dl8.5)
tsdt gosdt maptree murtree netherlands adult 0.40 Test Loss 0.175 0.200 0.225 0.4 0.35 0.2 0.30 0 10 20 30 Number of Leaves 0 10 20 30 Number of Leaves 0 10 20 30 Number of Leaves 0.175 0.180 0.185 0.190 Test Loss vs Training Time for Models in the Red Box Test Loss 0.32 0.20 0.30 0.15 10 1 10 1 10 3 Training Time (s)
10 1 10 1 10 3 Training Time (s)
10 1 10 1 10 3 Training Time (s)
Test Loss vs # Leaves for Different Decision Tree Algorithms covertype compas heloc 0.35 0.35 0.40 0.45 Test Loss 0.4 0.30 0.25 0 10 20 30 Number of Leaves 0.3 0 10 20 30 Number of Leaves 0 10 20 30 Number of Leaves Test Loss vs Training Time for Models in the Red Box 10 1 10 1 10 3 Training Time (s)
0.32 0.34 0.36 Test Loss 0.29 0.30 0.31 10 1 10 1 10 3 Training Time (s)
0.24 0.25 0.26 10 1 10 1 10 3 Training Time (s)
Test Loss vs # Leaves for Different Decision Tree **Algorithms**
bank 0 10 20 30 Number of Leaves 0.000 0.025 0.050 0.075 hypothyroid 0 10 20 30 Number of Leaves 0.1 0.2 0.3 0.4 spambase 0.6 Test Loss 0.4 0.2 0 10 20 30 Number of Leaves Test Loss vs Training Time for Models in the Red Box 10 1 10 1 10 3 Training Time (s)
0.10 0.11 0.12 0.13 Test Loss 0.02 0.15 0.01 10 1 10 1 10 3 Training Time (s)
0.10 10 1 10 1 10 3 Training Time (s)
0.00

## A.1.3. What About Depth 6 Trees?

In this section, we perform the same evaluation as above, but with depth 6 trees. We set the lookahead depth as 2. Note that Murtree and GOSDT are not included in the comparison as they take much longer to run for deeper trees.

Test Loss vs # Leaves for Different Decision Tree **Algorithms**
bike SPLIT (ours)
LicketySPLIT (ours)
topk (dl8.5)
tsdt greedy maptree netherlands adult 0.40 Test Loss 0.175 0.200 0.225 0.4 0.35 0.2 0.30 0 102030 Number of Leaves 0 10 20 30 Number of Leaves 0 102030 Number of Leaves Test Loss vs Training Time for Models in the Red Box 10 1 10 1 10 3 Training Time (s)
0.29 0.30 0.31 0.32 0.19 Test Loss 0.20 0.18 0.15 10 1 10 1 10 3 Training Time (s)
10 1 10 1 10 3 Training Time (s)
covertype compas heloc 0 10 20 30 Number of Leaves 0.30 0.35 0.40 0.45 0.35 Test Loss 0.4 0.30 0.25 0 102030 Number of Leaves 0.3 0 102030 Number of Leaves Test Loss vs Training Time for Models in the Red Box 0.32 0.34 0.36 Test Loss 0.24 0.25 0.26 10 1 10 1 10 3 Training Time (s)
0.29 0.30 0.31 10 1 10 1 10 3 Training Time (s)
10 1 10 1 10 3 Training Time (s)
Test Loss vs \# Leaves for Different Decision Tree **Algorithms**

bank 0 10 20 30 Number of Leaves 0.000 0.025 0.050 0.075 hypothyroid 0.1 0.2 0.3 0.4 spambase 0.6 Te st Lo ss 0.4 0.2 0 102030 Number of Leaves 0 10 20 30 Number of Leaves Test Loss vs Training Time for Models in the Red Box 10 1 10 1 10 3 Training Time (s)
0.10 0.11 0.12 0.13 10 1 10 1 10 3 Training Time (s)
0.00 0.01 0.02 0.03 0.10 0.12 0.14 Te st Lo ss

$${\mathfrak{H}}$$
$$(6)$$

10 1 10 1 10 3 Training Time (s)

## A.2. Many Near-Optimal Trees Exhibit Monotonically Decreasing Optimality Gaps Closer To Leaves

Consider an ϵ-optimal tree T ∈ R(*D, λ, ϵ, d*). For a subtree t of T, define λt as the value of λ that results in the greedy tree, Tg, having the same number of leaves as t. We now define the *optimality gap* δ(Dt, t) as the difference between the loss of t and the loss of an equally sparse greedy tree on the sub-problem associated with t. This enables a fair performance comparison between greedy and optimal trees, as the training loss of any given tree will otherwise monotonically decrease with the number of leaves.

δ(Dt, t) = L(t, Dt, λ) − L(Tg(Dt, depth(t), λt), Dt, λt). (5)
For a tree T ∈ R, we then compute the average optimality gap associated with subtrees at each level. That is, given a level ℓ, we compute:

$$\delta(D_{t},t)=L(t,D_{t},\lambda)-L(T)$$
$$\beta(T,D,\ell)={\frac{\sum\limits_{t\in T}\delta(D_{t},t)\mathbb{1}\left[t{\mathrm{~is~rooted~at~level~}}\ell\right]}{\sum\limits_{t\in T}\mathbb{1}\left[t{\mathrm{~is~rooted~at~level~}}\ell\right]}}.$$
$$\mathbf{h}(t),\lambda_{t}),D_{t},\lambda_{t}).$$
1[t is rooted at level ℓ]. (6)
We want to determine if β(*T, D, l*) is monotonically decreasing with ℓ for a given tree T - if this is true, then being greedy closer to the leaf does not incur much loss in performance. Our intuition is as follows: if there are many such near optimal trees, then a semi-greedy search strategy could potentially uncover at least one of them. The following statistic computes the proportion of all trees in the Rashomon set that have monotonically decreasing optimality gaps as ℓ increases (i.e., moves from root towards leaves):

$$m(D,\lambda,\epsilon,d)={\frac{\sum_{T\in\mathbb{R}(D,\lambda,\epsilon,d)}\mathbb{1}\left[\beta(T,D,\ell){\mathrm{~is~monotonically~decreasing~with~}}\ell\right]}{|\mathcal{R}(D,\lambda,\epsilon,d)|}}.$$

Figure 8 shows this statistic for Rashomon sets with varying values of the sparsity penalty λ. We fix ϵ = 0.025. The sparser a near-optimal tree, the more likely that it will be greedy, however, for all datasets, there exist near-optimal trees with monotonically decreasing optimality gaps even for low sparsity penalties. This has important algorithmic implications for developing interpretable models, because it means that a search strategy that is increasingly greedy near the leaves can produce a near-optimal tree.

$$(7)$$

## A.3. Miscellaneous Properties Of Split A.3.1. Which Lookahead Depth Should I Use?

In this section, we explore the effect of the lookahead depth on the runtime and regularised test and train losses. We use the aggressively binarized versions of the datasets, as elaborated in Section A.7.

compas 10 1 heloc 10 1 covertype 10 1 Training time (s)
10 0 10 0 10 0 10 1 10 1 10 1 10 2 10 2 10 2 netherlands 10 1 adult 10 1 bike 10 1 Training time (s)
10 1 10 0 10 1 10 0 10 1 10 0 0 1 2 3 4 Lookahead Depth 10 2 0 1 2 3 4 Lookahead Depth 10 2 0 1 2 3 4 Lookahead Depth 10 2 compas 0.247 0.248 0.248 0.248 0.248 heloc 0.290 0.290 0.290 0.290 0.291 covertype Regularised Loss 0.319 0.320 0.320 0.321 netherlands 0 1 2 3 4 Lookahead Depth 0.180 0.180 0.180 0.180 0.181 adult 0.148 bike Regularised Loss 0.300 0.146 0.290 0.144 0.280 0 1 2 3 4 Lookahead Depth 0 1 2 3 4 Lookahead Depth compas heloc 0.300 covertype 0.3075 0.3100 0.3125 0.3150 0.3175 Regularised Loss 0.24700 0.24725 0.24750 0.24775 0.295 netherlands 0 1 2 3 4 Lookahead Depth 0.1866 0.1868 0.1870 0.1872 0.1874 adult bike Regularised Loss 0.31 0 1 2 3 4 Lookahead Depth 0.142 0.144 0.146 0.148 0.30 0.29 0 1 2 3 4 Lookahead Depth
From the figures, we see that there indeed exists an optimal lookahead depth that minimizes the runtime of SPLIT. At this depth, however, there is only a small increase in regularised training loss. Surprisingly, the test loss can also be lower at the runtime minimizing depth.

## A.3.2. Are Split Trees In The Rashomon Set?

This evaluation characterises the near optimal behaviour of trees produced by our algorithms. In particular, we're interested in understanding how often trees produced by our algorithms lie in the Rashomon set. To do this, we sweep over values of λ. For each λ, we first generate SPLIT and LicketySPLIT trees and compute the minimum value of ϵ needed such that they are in the corresponding Rashomon set of decision trees with depth budget 5 - this is denoted by the respective frontiers of both algorithms.

adult 0.000 0.002 0.004 0.006 0.008 heloc SPLIT LicketySPLIT
bike SPLIT LicketySPLIT
SPLIT LicketySPLIT
0.00000 0.00025 0.00050 0.00075 0.00100 0.004 E

p sil o n

(

)

0.002 0.000 compas 10 3 10 2 10 1 Lambda ( )
0.000 0.002 0.004 0.006 0.008 netherlands SPLIT LicketySPLIT
covertype 10 3 10 2 10 1 Lambda ( )
0.000 0.002 0.004 0.006 0.008 SPLIT LicketySPLIT
SPLIT LicketySPLIT
10 3 10 2 10 1 Lambda ( )
0.000 0.001 0.002 0.003 E

p silo n

(

)

Figure 12 shows that this minimum ϵ is small regardless of the value of λ. While SPLIT has a smaller minimum ϵ, implying a lower optimality gap, particularly noteworthy is the performance of LicketySPLIT. Despite admitting a polynomial runtime, it manages to lie in the Rashomon set even for ϵ as small as 10−3.

## A.3.3. Split With Optimality Preserving Discretization

In this section, we briefly consider how SPLIT performs under full binarization of the dataset. For a given dataset, we perform full binarization by collecting every possible threshold (i.e. split point) present in every feature. We then compare the resulting regularised test loss and runtimes to that of threshold guessing.

- For this experiment, we first randomly choose 2000 examples from the Netherlands, Covertype, HELOC, and Bike datasets. Larger dataset sizes would produce around 105features for the fully binarized dataset, which would make optimization extremely expensive computationally.

- We then produce two versions of the dataset - a fully binarized version (which contains around 3000-5000 features for each dataset), and a threshold-guessed version (McTavish et al., 2022) with num estimators = 200. The latter ensures that the number of features in the resulting datasets is between 40-60.

For a given dataset, let D∗and Dtg its the fully binarized and threshold guessed version. We then run SPLIT and