# Spectral Perturbation Bounds For Low-Rank Approximation With Applications To Privacy

| Phuc Tran     | Nisheeth K. Vishnoi∗   | Van H. Vu       |
|---------------|------------------------|-----------------|
| VinUniversity | Yale University        | Yale University |

## Abstract

A central challenge in machine learning is to understand how noise or measurement errors affect low-rank approximations—particularly in the spectral norm. This question is especially important in differentially private low-rank approximation, where one aims to preserve the top-p structure of a data-derived matrix while ensuring privacy. Prior work often analyzes Frobenius norm error or changes in reconstruction quality, but these metrics can over- or under-estimate true subspace distortion. The spectral norm, by contrast, captures worst-case directional error and provides the strongest utility guarantees. We establish new high-probability spectral-norm perturbation bounds for symmetric matrices that refine the classical Eckart–Young–Mirsky theorem and explicitly capture interactions between a matrix A ∈ R
n×n and an arbitrary symmetric perturbation E. Under mild eigengap and norm conditions, our bounds yield sharp estimates for ∥(A + E)p − Ap∥,
where Ap is the best rank-p approximation of A, with improvements of up to a factor of 
√n. As an application, we derive improved utility guarantees for differentially private PCA, resolving an open problem in the literature. Our analysis relies on a novel contour bootstrapping method from complex analysis and extends it to a broad class of spectral functionals, including polynomials and matrix exponentials. Empirical results on real-world datasets confirm that our bounds closely track the actual spectral error under diverse perturbation regimes.

## 1 Introduction

Low-rank approximation is a foundational technique in machine learning, data science, and numerical linear algebra, with applications ranging from dimensionality reduction and clustering to recommendation systems and privacy-preserving data analysis [1, 4, 5, 14, 21, 23, 24, 42, 45].

A common setting involves a real symmetric matrix A ∈ R
n×n, such as a sample covariance matrix derived from high-dimensional data. Let λ1 *≥ · · · ≥* λn denote the eigenvalues of A, with corresponding orthonormal eigenvectors u1*, . . . , u*n. The best rank-p approximation of A is denoted by Ap := Pp i=1 λiuiu
⊤ i
. This approximation solves the optimization problem Ap = arg minrank(B)≤p ∥A − B∥, where the norm can be any *unitarily invariant norm* [7, 10]. In particular, Ap minimizes both the spectral norm *∥· ∥*, measuring worst-case error, and the Frobenius norm *∥ · ∥*F , measuring average deviation.

In many applications, the matrix A is not directly available—it may be corrupted by noise, compressed for efficiency, or randomized to preserve privacy. A standard model introduces a symmetric perturbation E, yielding the observed matrix A˜ := A + E. The approximation A˜p, computed from A˜, is often used in downstream learning and inference. This leads to a central question: How does the perturbation E affect the top-p *approximation* Ap? Understanding the deviation ∥A˜p − Ap∥ is critical for ensuring the reliability and robustness of low-rank methods under noise.

∗Alphabetical order. Correspondence to nisheeth.vishnoi@gmail.com.

Motivating application: differential privacy. The stability under perturbations is especially important when the matrix A encodes *sensitive information*, such as user behavior or medical data. In such settings, even low-rank approximations of A can inadvertently leak private information [6]. To address this risk, differential privacy (DP) [14] has become the standard framework for designing privacy-preserving algorithms. Several mechanisms have been developed to release private low-rank approximations while satisfying DP guarantees [8, 9, 15, 25, 29, 31, 34, 39]. A canonical method, introduced in [15], adds a symmetric noise matrix E with i.i.d. Gaussian entries to the input matrix A, yielding the perturbed matrix A˜ = A + E. The algorithm then releases A˜p as the privatized output. The *utility* of such mechanisms is typically assessed by comparing A˜p to the ideal (non-private) approximation Ap. Two standard metrics are: (1) the *Frobenius norm error* ∥A˜p − Ap∥F , and (2) the *change in reconstruction error* |∥A−Ap∥⋆−∥A−A˜p∥⋆|, which measures how much the quality of low-rank approximation degrades due to noise, for a norm *∥ · ∥*⋆ [3, 11, 15, 29]. These metrics offer insight into the effect of noise on overall variance or total reconstruction error. However, as we explain next, they may fail to capture *worst-case directional misalignment*, which is often critical for downstream tasks and algorithmic guarantees. Limitations of existing utility metrics. The Frobenius norm error and reconstruction error may not be appropriate in applications that rely on the geometry of the top-p eigenspace. In particular, the Frobenius norm may *overestimate* the impact of noise by up to a factor of 
√p when the perturbation E lies largely in directions orthogonal to the top-p subspace. The reconstruction error metric can *underestimate* subspace deviation—sometimes dramatically. In some cases, it remains small (or even zero) despite substantial rotation in the top-p eigenspace. (See Sections B for concrete illustrations.) These limitations motivate the use of the *spectral norm* ∥A˜p − Ap∥, which captures the *worst-case* directional deviation between the two low-rank approximations. The spectral norm also governs algorithmic robustness in many downstream applications, such as PCA-based learning, private clustering, and subspace tracking. A classical spectral norm bound, derived from the Eckart–Young–Mirsky theorem [7, 16], states that ∥A˜p − Ap∥ ≤ 2(λp+1 + ∥E∥), which holds for arbitrary matrices and noise. However, such bounds are often pessimistic and fail to exploit the structure of A and E. More refined bounds exist in the Frobenius norm setting. For example, recent work [29, 30] shows that when A is positive semidefinite and has a nontrivial eigengap δp := λp − λp+1 ≥ 4∥E∥, and when E is drawn from a complex Gaussian ensemble, one obtains: E∥A˜p − Ap∥F = O˜(
√p · ∥E∥ · λp δp
), which improves on the earlier reconstruction-error-based bounds of [15] by a factor of 
√p. However, these bounds have important limitations: They hold only *in expectation* and do not yield high-probability guarantees; They often assume Gaussian noise distributions; They are not spectral norm bounds and therefore do not directly quantify the worst-case impact on the eigenspace. These limitations prompt the following open question, raised in [29, Remark 5.3]: *Can one obtain high-probability spectral norm* bounds for ∥A˜p − Ap∥ under natural structural assumptions on A *and realistic noise models?*
Our contributions. We resolve the open question posed in [29, Remark 5.3], proving new highprobability spectral norm bounds for low-rank approximation under symmetric perturbations. Our results rely on natural structural assumptions on A and E and yield the first such guarantees for differentially private PCA (DP-PCA).

- **Two high-probability spectral norm bounds.** Under the same eigengap condition as [29], δp :=
λp−λp+1 ≥ 4∥E∥, we prove ∥A˜p−Ap∥ = O
∥E∥ · λp δp and ∥A˜p−Ap∥ = O˜∥E∥ + r 2x ·
λp δp
,
where r is the *halving distance* (a measure of spectral decay) and x := maxi,j≤r |u
⊤
i Euj | quantifies noise–eigenspace alignment (Theorems 2.1–2.2). In addition, our contour-based framework extends to a broader class of spectral functionals f(A) (beyond f(A) = A), encompassing matrix powers, exponentials, and trigonometric transforms; see Theorem 2.3.

- **Spectral utility bounds for DP-PCA.** Our first bound yields a high-probability spectral norm utility guarantee for differentially private PCA under sub-Gaussian noise, improving existing Frobenius-norm bounds by up to a factor of 
√p (Corollary 2.4). While prior work has achieved spectral norm guarantees in iterative or multi-pass settings [17, 18], our contribution concerns the *direct noise-addition* model, where this appears to be the first such result. For matrices with low stable rank and weak eigenspace–noise interaction, our second bound further improves by up to 
√n.

- **Novel analytical technique: contour bootstrapping.** Our proof relies on a contour bootstrapping argument (Lemma 3.1), which provides a new way to analyze the contour representation of perturbations [19, 26, 35], enabling analysis of a broader class of spectral functionals (Theorem 2.3). The bootstrapping argument here is a generalization of the argument used to handle eigenspaces perturbation introduced in [37].

- **Empirical validation.** We benchmark our bounds on real covariance matrices under both Gaussian and Rademacher noise. Across datasets and noise regimes, the predicted error closely matches empirical behavior and consistently surpasses classical baselines, confirming the sharpness and robustness of our theoretical results (Section 4).

## 2 Main Results

Main spectral norm bound. For clarity, we state our main bounds assuming A ∈ R
n×n is positive semi-definite (PSD); extensions to symmetric matrices appear in Section D. Let λ1 *≥ · · · ≥* λn ≥ 0 be the eigenvalues of A, with corresponding orthonormal eigenvectors u1*, . . . , u*n, and define the eigengap δk := λk − λk+1. Given a real symmetric perturbation matrix E, we let A˜ := A + E, and define Ap and A˜p as the best rank-p approximations of A and A˜, respectively. Our goal is to bound the spectral error ∥A˜p − Ap∥.

Theorem 2.1 (**Main spectral bound - PSD).** If 4∥E∥ ≤ δp, then: ∥A˜p − Ap∥ ≤ O(∥E∥ · λp δp
).

The O(·) notation here hides a small universal constant (less than 7), which we have not optimized; see Section D.1 for the proof of the generalization to the symmetric setting, of which this theorem is a special case. For Wigner noise—i.e., a symmetric matrix E with i.i.d. sub-Gaussian entries of mean 0 and variance 1—we have ∥E∥ = (2 + o(1))√n with high probability [41, 43],
so Theorem 2.1 reduces to ∥A˜p − Ap∥ = O
√n λp δp
. The right-hand side is explicitly noisedependent, addressing a key limitation of the classical Eckart–Young–Mirsky bound. Moreover, in many widely studied structured models (e.g., spiked covariance, stochastic block, and graph Laplacian models), one typically has λp = O(δp), yielding the clean bound O(∥E∥). This rate is theoretically tight: for instance, when A is a PSD diagonal matrix and E = µIn for some µ > 0, we have
∥A˜p − Ap∥ = µ = ∥E∥.

Gap condition. Our assumption 4∥E∥ < δp aligns with standard conditions in prior work, including [29, 30], and is satisfied in many well-studied matrix models—such as spiked covariance (Wishart) models, deformed Wigner ensembles, stochastic block models, and kernel matrices for clustering. It also arises naturally in classical perturbation theory [12, 26, 28]. Empirical analyses [29, Section B] further show that this condition holds for real-world datasets commonly used in private matrix approximation (e.g., the 1990 U.S. Census and the UCI Adult dataset [3, 11]). Hence, Theorem 2.1 operates under a mild and broadly applicable assumption, satisfied across both theoretical models and practical benchmarks.

Comparison to the Eckart–Young–Mirsky bound. Using λp = δp + λp+1, Theorem 2.1 rewrites as ∥A˜p − Ap∥ = O(∥E∥ + λp+1 ·
∥E∥
δp
). This improves on the E-Y-M bound O(∥E∥ + λp+1)
when λp+1 ≫ ∥E∥, by a factor of min{
λp+1
∥E∥
,
δp
∥E∥
}. For example, consider a matrix with spectrum
{10n, 9n, . . . , n, n/2, 1*, . . . ,* 1} and p = 10. For Gaussian noise with ∥E∥ = O(
√n), E-Y-M yields O(n) error, while our bound gives O(
√n), a √n-factor gain.

Comparison to Mangoubi-Vishnoi bounds [29, **30].** Our bound also improves upon the Frobenius norm bounds of [29, 30], which under the same gap assumption yield: E∥A˜p − Ap∥F =
O˜(
√p∥E∥ · λp δp
). We eliminate the 
√p factor, upgrade from expectation to high probability, and support real-valued, non-Gaussian noise models. A more detailed comparison appears later in this section (Corollary 2.4), where we analyze implications for differentially private PCA. Proof technique: contour bootstrapping. Unlike prior analyses [29, 30], which rely on Dyson Brownian motion and tools from random matrix theory (see Section A, our proof of Theorem 2.1 uses a contour-integral representation of the rank-p projector. This approach, which we call *contour* bootstrapping, isolates the top-p eigenspace via complex-analytic techniques and avoids powerseries or Davis–Kahan-type expansions. It enables tighter, structure-aware spectral bounds and extends naturally to refined perturbation results (Theorem 2.2) and general spectral functionals (Theorem 2.3). Full details appear in Section 3. Refined bound via eigenspace interaction. To sharpen our analysis, we incorporate fine-grained structure of the eigenspace and its interaction with the noise. Inspired by the recent works [33, 38], we start with the observation that the rank-p perturbation is primarily influenced by the cluster of eigenvalues near λp, and the interaction between E and the corresponding eigenvectors. To control these factors, we define the *halving distance* r (w.r.t the index p) as the smallest integer such that λr+1 ≤ λp/2, and *interaction term* x := max1≤i,j≤r |u
⊤
i Euj |, measuring the alignment between the noise E and the top-r eigenvectors of A. This yields a refined spectral norm bound:
Theorem 2.2 (**Interaction-aware bound).** If 4∥E∥ ≤ δp, then ∥A˜p − Ap∥ ≤ O˜(∥E∥ + r 2x ·
λp δp
).

See Section D.2 for the proof and its generalization to the symmetric setting. This bound improves upon the basic eigengap bound O
∥E∥ · λp δp when the interaction term r 2x is small. This occurs, for instance, when (i) A has low stable rank or clustered eigenvalues (e.g., spiked models, multi-cluster Laplacians), (ii) the noise E is random and approximately orthogonal to the leading eigenspace, or (iii) λp/δp is large but x = O˜(1) and r = O˜(1). In such regimes, the bound simplifies to O˜∥E∥ +
λp δp
, yielding up to a 
√n-factor improvement over Theorem 2.1. This highlights the benefit of explicitly incorporating spectral decay and noise–eigenspace alignment when analyzing noise-robust low-rank approximations. In practice, many public DP datasets (e.g., Census, Adult, KDD) have small dimensions and modest eigenspace decay, the simple bound is more effective. However, the refined bound becomes especially informative in large-scale or synthetically structured settings. Thus, the two bounds are best viewed as *complementary*: the first is robust and broadly applicable, while the second highlights structural regimes where stronger stability is provable. Extension to spectral functionals. Beyond approximating A itself, many applications involve lowrank approximations of spectral functions f(A), such as Ak, exp(A), or cos(A); see [7, 44]. Our contour-based analysis extends naturally to this broader setting. Let fp(A) := Pp i=1 f(λi)uiu
⊤ i denote the best rank-p approximation of f(A). We obtain the following general perturbation bound.

Theorem 2.3 (**Perturbation bounds for general functions).** If 4∥E∥ ≤ δp*, then*

$$\|f_{p}({\bar{A}})-f_{p}(A)\|\leq O\left(\operatorname*{max}_{z\in\Gamma_{1}}\|f(z)\|\right)$$
δp
,
where Γ1 is the rectangle with vertices (x0, T),(x1, T),(x1, −T),(x0, −T) *with*

$\sim\frac{11-\sqrt{3}}{\sqrt{3}}$
$\vdash\bot$
x0 := λp −
δp 2
, x1 := 2λ1, T := 2λ1.

The O(·) notation hides a small universal constant (less than 4), which we have not attempted to optimize; see Section F for details. For example, let f(z) = z 3, so that fp(A˜) and fp(A) correspond to the best rank-p approximations of A˜3and A3, respectively. Since maxz∈Γ1 ∥f(z)∥ ≤ 64∥A∥
3, Theorem 2.3 yields ∥A˜3p − A3p∥ = O∥A∥
3· ∥E∥/δp
. This result applies to many important classes of functions—e.g., polynomials, exponentials, and trigonometric functions—and hence we expect it to be broadly useful. However, Theorem 2.3 does not apply to non-entire functions such as f(z) = z cfor non-integer c, where singularities obstruct the contour representation (1). In particular, when c < 0, the expression fp(A) is no longer the best rank-p approximation to f(A), so the conclusion of Theorem 2.3 is not meaningful in that setting. We note that in a related work [36], the first two authors present an extension of the setting f(z) = z
−1.

Application: differentially private low-rank approximation. We now apply our spectral norm bound to analyze a standard differentially private (DP) mechanism for releasing a low-rank approximation of a sensitive matrix A, commonly assumed to be a sample covariance matrix and hence PSD. Under (*ε, δ*)-DP [14], the Gaussian mechanism releases A˜ := A + E, where E is a symmetric matrix with i.i.d. Gaussian entries scaled to sensitivity ∆ = O(plog(1/δ)/ε). A common postprocessing step is to compute A˜p, the best rank-p approximation of A˜. Prior analyses [3, 15, 30]
focused primarily on Frobenius norm or reconstruction error. For instance, [30] showed that under complex Wigner noise and a moderate eigengap, E∥A˜p − Ap∥F ≤
√*pn λ*p δpup to lower-order terms.

Since ∥A˜p − Ap*∥ ≤ ∥*A˜p − Ap∥F , the above inequality implies an expected spectral norm error of O˜√pn λp δp
. In contrast, our bound yields the following high-probability spectral norm guarantee:
Corollary 2.4 (**Application to differential privacy).** Let A be PSD and E be a real or complex Wigner matrix. If δp ≥ 8.01√n*, then with probability* 1 − o(1), ∥A˜p − Ap∥ ≤ O(
√n ·
λp δp
).

This follows directly from Theorem 2.1, using the fact that ∥E∥ = O(
√n) with high probability for Wigner matrices [40, 43]. Compared to [30], this result provides a spectral norm (rather than Frobenius) guarantee, holds with high probability instead of in expectation, applies to both real and complex Wigner noise, removes the loglog log n n factor, and eliminates restrictive assumptions such as λ1 ≤ n 50. It also improves the dependence on p by a factor of 
√p, thereby resolving the open question posed in [30, Remark 5.3]. The spectral norm better captures subspace distortion, which is critical in applications like private PCA. Unlike Frobenius or reconstruction error—both of which may remain small even when A˜p deviates significantly from the true top-p eigenspace—the spectral norm reflects worst-case directional error and is thus a more reliable utility metric. This distinction is empirically validated in Figure 3. Moreover, Corollary 2.4 further yields high-probability Frobenius norm and reconstruction error bounds on the perturbation of low-rank approximations:

$\|\tilde{A}_{p}-A_{p}\|_{F}\leq O\big{(}\sqrt{pn}\cdot\frac{\lambda_{p}}{\delta_{p}}\big{)},\ \mbox{and}\ \|\|\tilde{A}_{p}-A\|-\|A_{p}-A\|\|\leq O\big{(}\sqrt{n}\cdot\frac{\lambda_{p}}{\delta_{p}}\big{)}.$
Finally, while Corollary 2.4 is stated for sub-Gaussian noise, Theorem 2.1 extends to any symmetric perturbation satisfying the norm and gap conditions, including subsampled or quantized Gaussians and Laplace noise. We leave the detailed analysis of these settings to future work.

| Table 1: Summary table of perturbation bounds on A˜ p − Ap for noise E. Bound type Norm Noise model Assumption Extra factor vs ∥E∥  1 + λp+1                 |                  |           |                  |                          |    |     |    |
|----------------|------------------|-----------|------------------|--------------------------|----|-----|----|
| EYM bound      | High-probability | Spectral  | Real and Complex | None                     | O  | ∥E∥ |    |
| M-V bound [29] | Expectation      | Frobenius | GOE (real)       | δi > 4∥E∥ ∀ 1 ≤ i ≤ p    | O  √pλp δp     |     |    |
| M-V bound [30] | Expectation      | Frobenius | GUE (complex)    | δp > 2∥E∥, λ1 < n50      | O˜  √pλp δp     |     |    |
| Thm. 2.1       | High-probability | Spectral  | Real and Complex | δp > 4∥E∥                | O  λp  δp    |     |    |
| Thm. 2.2       | High-probability | Spectral  | sub-Gaussian     | δp > 4∥E∥, rankA = O˜(1) | O˜  1 +    | λp  |     |
| δp∥E∥          |                  |           |                  |                          |    |     |    |

"EYM" and "M–V" denote the Eckart–Young–Mirsky and [29, 30] bounds, respectively.

Alternative methods for approximating Ap. Hardt and Price [17, 18] proposed a random iterative method which, under the condition δp ≫
√n log n, produces a rank-k approximation A′ of Ap with k = p + O(1), satisfying the trade-off bound ∥A′ − Ap∥ = O˜√n λ1 δp max1≤i≤n ∥ui∥∞
, where ui denotes the eigenvectors of A.

If *at least one* eigenvector uiis localized (i.e., max1≤i≤n ∥ui∥∞ = 1/O˜(1)), this simplifies to O˜√n λ1 δp
. In this regime, Theorem 2.1 achieves a smaller bound by a factor of O˜(λ1/λp)—up to 
√n when λ1 = Θ(n) and λp = Θ(√n). Furthermore, Theorem 2.2 provides an additional improvement by a factor of O
minn √n r 2 ,
λ1 δp o, which can reach 
√n when r = O˜(1) and δp =
Θ(√n)—a common regime in high-dimensional data.

If all eigenvectors ui are delocalized (i.e., max1≤i≤n ∥ui∥∞ = O˜(1)/
√n), the Hardt–Price bound reduces to O˜(λ1/δp). Theorem 2.1 achieves a comparable rate when σ1 = Θ(n) and λp = c δp =
Θ(√n), while Theorem 2.2 yields an improvement by a factor of λ1/λp whenever r = O˜(1), i.e.,
when A is approximately low-rank.

## 3 Proof Outline

In the preceding section, we stated our main results—Theorems 2.1, 2.2, and 2.3. Here, we first sketch the key ideas behind the proof of Theorem 2.1, then adapt the same framework, with minor refinements, to derive Theorems 2.2 and 2.3. The proof of Theorem 2.1 proceeds in three main steps. First, using the contour method, we obtain the contour-based bound of our perturbation ∥A˜p − Ap∥ ≤ F(z) := 1 2πi
∥RΓ
z[(zI − A˜)
−1 − (zI −
A)
−1]∥dz. Here Γ is a contour on the complex plane, isolating the p-leading eigenvalues of A and A˜. This contour step captures the A–E interaction that the Eckart–Young–Mirsky bound omits (see Appendix A). Secondly, we develop the *contour bootstrapping technique* (Lemma 3.1), which under the gap assumption 4∥E∥ ≤ δp, yields F(z) ≤ 2F1(z) with F1(z) := RΓ
∥z(zI − A)
−1E(zI −
A)
−1∥|dz|. This technique (valid for any entire function f) replaces the traditional series expansions and the heavy analysis of the matrix-derivative operator (the limitation of the Mangoubi-Vishnoi approach [29, 30], Appendix A) with a computable quantity. Third, we construct a bespoke contour Γ— one specifically tailored so that the top-p eigenvalues of A and A˜ lie at prescribed distances from its sides. This precise alignment makes the integral defining F1(z) both tractable and essentially optimal, yielding a tight perturbation bound.

Step 1: Representing ∥fp(A˜) − fp(A)∥ **via the classical contour method.** Let λ1 *≥ · · · ≥*
λn be the eigenvalues of A with the corresponding eigenvectors {ui}
n i=1. We now present the contour method to bound matrix perturbations in the spectral norm. Let Γ be a contour in C that encloses λ1, λ2*, . . . , λ*p and excludes λp+1, λp+2*, . . . , λ*n. Let f be any entire function and recall fp(A) = Pp i=1 f(λp)uiu
⊤
i. Since f is analytic on the whole plane C, the well-known contour integral representation [19, 26, 35] gives us:
1 2πi RΓ
f(z)(zI − A)
−1dz =Pp i=1 f(λi)uiu
⊤
i = fp(A).

Let λ˜1 *≥ · · · ≥* λ˜n denote the eigenvalue of A˜ with the corresponding eigenvectors u˜1, u˜2*, . . . ,* u˜n.

The construction of Γ (presented later) and the gap assumption 4∥E∥ < δp ensure that the eigenvalues λ˜i for 1 ≤ i ≤ p lie inside Γ, while all λ˜j for *j > p* remain outside. Then, similarly, we have 1 2πi RΓ
f(z)(zI − A˜)
−1dz =Pp i=1 f(λ˜i)˜uiu˜
⊤
i:= fp(A˜). Thus, we obtain the following contour identity for the perturbation:

$$f_{p}(\tilde{A})-f_{p}(A)=\frac{1}{2\pi\mathrm{i}}\int_{\Gamma}f(z)[(z I-\tilde{A})^{-1}-(z I-A)^{-1}]\,|d z|.$$
$$(1)$$
−1] |dz|. (1)
Now we bound the perturbation by the corresponding integral
∥fp(A˜) − fp(A)∥ ≤ 1 2π RΓ
∥f(z)[(zI − A˜)
−1 − (zI − A)
−1]∥dz =: F(f). (2)
This inequality makes the interaction of A and E explicit and is widely used in functional perturbation analysis, e.g., [19, 26, 28, 32, 33, 37]. However, obtaining a sharp bound on its right-hand side remains a formidable analytical challenge.

Step 2: Bounding F ≤ 2F1 **via the contour bootstrapping method.** Attempts to control F(f),
the right-hand side of (2), often use series expansion and analytical tools. By repeatedly applying the resolvent formula, one can expand f(z)[(zI − A˜)
−1 − (zI − A)
−1] into P∞
s=1 f(z)(zI −
A)
−1[E(zI − A)
−1]
s. This yields the bound:
F(f) ≤P∞
s=1 Fs(f), where Fs(f) = 
1 2π RΓ
f(z)(zI − A)
−1[E(zI − A)
−1]
s |dz|.

One needs to estimate Fs(f) for each s. For example, when f(z) = 1, [26, Part 2] bounds Fs(1) by O
∥E∥
sRΓ|dz| mini∈[n]|z−λi| s+1 
= O [(||E||/δp)
s], where Γ is a union of vertical lines isolating
{λi, i ∈ p}, yielding the Davis-Kahan bound O (∥E∥/δp). However, for f(z) = z (relevant for low-rank perturbations), this approach fails as |z*| → ∞*. These estimates are highly nontrivial and rely on deep analytical techniques, making generalization to arbitrary f challenging.

Moreover, for f(z) = 1, under certain conditions, the dominant term is F1(f), i.e., F(f) = O(F1(f)); see, e.g., [22, 27, 32, 33, 37]. In particular, using contour-bootstrapping technique, the authors in [37] proved F(f(z) = 1) ≤ 2F1(f(z) = 1). Inspired by this technique, we prove that F(f) ≤ 2F1(f) for any entire function f. Lemma 3.1 (**Contour bootstrapping for entire function** f). If δp ≥ 4∥E∥*, then* F(f) ≤ 2F1(f), *where* F1(f) := 1 2π RΓ
f(z)(zI − A)
−1E(zI − A)
−1 |dz|.

Our *contour bootstrapping argument* is designed to prove Lemma 3.1. Our argument is concise and novel, avoiding the need for series expansion and convergence analysis. In the context of standard low-rank approximations, where f(z) ≡ z and fp(A) = Ap, we write F(z) and F1(z) instead of F(f) and F1(f) respectively.

Step 3: Construction of Γ, F1(z)-estimation, and proof completion of Thm. **2.1.** Given Lemma 3.1, we now need to carefully choose the contour Γ and estimate F1(f). Constructing Γ (so that the perturbation analysis via contour integration provides a sharp bound) is delicate; for example, the classical pick of two vertically parallel lines and any Γ placed too near any λi can blow up F1(z) to infinity. Indeed, we tailor Γ w.r.t F1(z) as follows. First, we choose Γ to be rectangular as this simplifies integration. To control the factor (zI − A)
−1in the expression of F1(f), we need to ensure that the distance |z − λi| for any z ∈ Γ and i ∈ [n] are relatively large. Since Γ separates λp and λp+1, this minimal distance minz∈Γ,i∈[n]|z − λi| cannot exceed Θ(δp). Thus, we simply construct Γ through the midpoint x0 =
λp+λp+1 2. Finally, by setting the contour sufficiently high in the complex plane (while avoiding excessive height to prevent |f(z)| from diverging), we ensure that the primary contribution to the integral is from the vertical segments of Γ. This is because the distance |z − λi| is minimized on these segments. Note that, under the assumption 4∥E∥ < δp, this construction ensures that the p-leading eigenvalues of A and A˜ are well aligned inside the contour.

Now, in particular, to prove Theorem 2.1, we will estimate

$$2\pi F_{1}(z)=\int_{\Gamma}\|z(z I-A)^{-1}E(z I-A)^{-1}\|\,|d z|,$$

in which the contour Γ is set to be a rectangle with vertices
(x0, T),(x1, T),(x1, −T),(x0, −T), where x0 := λp − δp/2, x1 := 2λ1, T := 2λ1. Then, we split Γ into four segments: Γ1 := {(x0, t)| − T ≤ t ≤ T}; Γ2 := {(*x, T*)|x0 ≤ x ≤ x1}; Γ3 := {(x1, t)|T ≥ t ≥ −T}; Γ4 := {(x, −T)|x1 ≥ x ≥ x0}.

$$\overline{{\lambda_{p+1}}}$$

λp+1 λp λ1 Γ1 Γ3 Γ2

$$\mathbb{T}(\mathbb{Z}^{T}-\mathbb{Z})$$

Γ4
Given the construction of Γ, we have 2πF1 =P4k=1 Mk, where

$$M_{k}:=\int_{\Gamma;k}$$

z(zI − A)
−1E(zI − A)
−1 |dz|.

Intuitively, we set *T, x*1 large (= 2∥A∥) so that the main term is the integral along Γ1, i.e., M1. Indeed, factoring our E and using the fact that |z − λi*| ≥ |*z − λp| =
qδ 2 p + t 2 for all 1 ≤ i ≤ n and z ∈ Γ1 := {(x0, t)| − T ≤ t ≤ T}, we have M1 ≤RΓ1
∥E∥ · |z| mini∈[n]|z−λi|2 |dz| ≤ ∥E∥ ·
R T−T
√x 2 0+t 2
(δp/2)2+t 2 dt. Directly compute the integral R T−T
√x 2 0+t 2
(δp/2)2+t 2 dt (see Section E.3), we obtain:
M1 ≤ ∥E∥ · O (x0/δp) = O (∥E∥λp/δp).

By a similar manner, replace Γ1 by Γ3 := {(x1, t)| − T ≤ t ≤ T}, we have

$$M_{3}\leq\|E\|\cdot\int_{\Gamma_{3}}{\frac{|z|}{\operatorname*{min}_{i\in[n]}|z-\lambda_{i}|^{2}}}|d z|\leq\|E\|\cdot\int_{\Gamma_{3}}{\frac{|z|}{\operatorname*{min}_{i\in[n]}|z-\lambda_{i}|^{2}}}|d z|$$
√x
2
1+t
2
λ
2
1+t
2 dt,
where the last inequality follows the fact that mini∈[n]|z − λi| = |z − λ1| =p(x1 − λ1)
2 + t 2 =
pλ 2 1 + t 2. Directly compute the integral R T−T
√x 2 1+t2 λ 21+t 2 dt (see Section E.3), we obtain:
M3 ≤ ∥E∥ · O (x1/λ1) = O (∥E∥).

Similarly, M2, M4 = O(∥E∥) ( Section E.2). These estimates on M1, M2, M3, M4 imply F1(z) =
O
∥E∥ · λp δp
, which together with Lemma 3.1 proves Theorem 2.1.

 $\mathbf{i}$) = 0
Proving the contour bootstrapping lemma (Lemma **3.1).** The first observation is that using the Sherman-Morrison-Woodbury formula M−1 −(M +N)
−1 = (M +N)
−1NM−1[20] and the fact that A˜ = A + E, we obtain

$$(z I-A)^{-1}-(z I-\bar{A})^{-1}=(z I-A)^{-1}E(z I-\bar{A})^{-1}$$

Using this, we can rewrite

$$F(f)=\frac{1}{2\pi}\int_{\Gamma}\|f(z)(z I-A)^{-1}E(z I-\tilde{A})^{-1}\|\,|d z|\ \mathrm{as}\ .$$

$\mathcal{O}$
$=\;\pm\phi$

$\frac{E\left(2I-A\right)\quad||\,|a2|}{2}+1$
$\sqrt{(\pm7)(\pm7\pm\sqrt{17})}$
$$\cdot\,A)^{-1}E\|\cdot F(f).$$
$$({\mathfrak{I}})$$

1 2π RΓ
∥f(z)(zI − A)
−1E(zI − A)
−1 − f(z)(zI − A)
−1E[(zI − A)
−1 − (zI − A˜)
−1]∥ |dz|.

Using triangle inequality, we first see that F(f) is at most

$\underline{\phantom{\rule{0.000pt}{0.000pt}}}$
 $\blacksquare$
RΓ
∥f(z)(zI−A)
−1E(zI−A)
−1∥|dz|
2π +
RΓ
∥f(z)(zI−A)
−1E[(zI−A)
−1−(zI−A˜)
−1]∥|dz|
$\frac{2\pi}{\pi}$
| {z }
.

Next is the key observation that the second term in the equation above can be rearranged and upperbounded as follows so that the original perturbation appears again:

$$\frac{\operatorname*{max}_{z\in\Gamma}\left\|(z I-A)^{-1}E\right\|}{2\pi}\int_{\Gamma}\left\|f(z)[(z I-A)^{-1}-(z I-\bar{A})^{-1}]\right\|\,|d z|.$$
$$F(f)\leq F_{1}^{n}$$

Thus, we have F(f) ≤ F1(f) + maxz∈Γ
(zI − A)
−1E · F(f). (3)
Now we need our gap assumption that 4∥E∥ < δp and the construction of Γ, which imply minz∈Γ,i∈[n]|z − λi| ≥ δp/2 ≥ 2∥E∥. Therefore, we have

maxz∈Γ(zI − A)
−1E ≤ maxz∈Γ ∥(zI − A)
−1*∥ · ∥*E∥ =∥E∥
minz∈Γ,i∈[n]|z−λi| ≤
∥E∥
2∥E∥ =
1 2
.

Together with (3), it follows that F(f) ≤ F1(f) + 12 F(f). Therefore, 12 F(f, S) ≤ F1(*f, S*). This proves Lemma 3.1.

2 Remark 3.2. *Using a similar strategy, one can prove that* F1(f) ≤ maxz∈Γ ∥f(z)∥ · 1 2π RΓ
∥(zI − A)
−1E(zI − A)
−1∥|dz| ≤ maxz∈Γ ∥f(z)∥ · 2∥E∥
δp; see Appendix F. Together, this estimate and Lemma 3.1 prove Theorem *2.3.*
Second upper bound of M1 and proof of Theorem **2.2.** The key idea of the second bound is to replace (zI − A)
−1 by its spectral expansion Pn i=1 uiu
⊤ i z−λi
. Hence, M1 is rewritten as RΓ1
∥P1≤i,j≤nz
(z−λi)(z−λj )
uiu
⊤
i Euju
⊤ j
∥dz.

There are n 2terms in the expression, and the direct use of the triangle inequality cannot provide a good estimate. The next key trick is grouping up the r-top eigenvectors {ui}
ri=1. Formally, M1 is at most RΓ1
∥P1≤i,j≤rz
(z−λi)(z−λj )
uiu
⊤
i Euju
⊤
j∥|dz| +RΓ1
∥Pn≥i,j>rz
(z−λi)(z−λj )
uiu
⊤
i Euju
⊤
j∥|dz|
+RΓ1
∥Pi≤r<j i>r≥j z
(z−λi)(z−λj )
uiu
⊤
i Euju
⊤
j
∥|dz|.

To estimate the first term, we apply the triangle inequality. For each term, we factor out components independent of z and carefully evaluate the integral. Specifically, by the triangle inequality, the first term is at most

$\sum_{1\leq i,j\leq r}\int_{\Gamma_{1}}\|\frac{z}{(z-\lambda_{i})(z-\lambda_{j})}u_{i}u_{i}^{\top}Eu_{j}u_{j}^{\top}\|\|dz|=\sum_{1\leq i,j\leq r}\int_{\Gamma_{1}}\frac{|u_{i}^{\top}Eu_{j}|\cdot\|u_{i}u_{j}^{\top}\|\cdot\|z|}{|(z-\lambda_{i})(z-\lambda_{j})|}$.  
 -$ ||u_i u_j||\cdot||z|\over||(z-\lambda_j)||$ $ |dz$
Since max1≤i,j≤r |u
⊤
i Euj | ≤ x, ∥uiu
⊤ j
∥ = 1, and Γ1 := {z | z = x0 + it, −T ≤ t ≤ T}, the r.h.s.

is at most

$$\sum_{i,j\leq r}x\int_{-T}^{T}{\frac{\sqrt{x_{0}^{2}+t^{2}}}{\sqrt{((x_{0}-\lambda_{i})^{2}+t^{2})((x_{0}-\lambda_{j})^{2}+t^{2})}}}d t\leq\sum_{i,j\leq r}x\int_{-T}^{T}{\frac{|x_{0}|+|t|}{\sqrt{((x_{0}-\lambda_{i})^{2}+t^{2})((x_{0}-\lambda_{j})^{2}+t^{2})}}}d t.$$

By the construction of Γ1, we have |x0 − λi| ≥ 
δp 2 for all i ∈ [n]. Thus, the r.h.s. is bounded by r 2xR T−T|x0|+|t| t 2+(δp/2)2 dt, which by direct computation (see Appendix E.1 for full details) is less than or equals

$$r^{2}x\left({\frac{2\pi x_{0}}{\delta_{p}}}+2\log\left({\frac{3T}{\delta_{p}}}\right)\right)=\tilde{O}\left(r^{2}x{\frac{\lambda_{p}}{\delta_{p}}}\right).$$

To estimate the second term, we apply matrix-norm inequalities to factor out E from the integral: RΓ1
∥Pn i,j=rz
(z−λi)(z−λj )
uiu
⊤
i Euju
⊤ j
∥|dz| ≤ RΓ1 |z| · ∥Pn≥i>r uiu
⊤ i z−λi
∥ ·
∥E∥ · ∥Pn≥i>r uiu
⊤ i z−λi
∥|dz|, which is at most ∥E∥RΓ1|z| minn≥i>r |z−λi| 2 |dz| =
√x 2 0+t 2 minn≥i>r[(x0−λi)
2+t 2]
dt. Moreover, by the construction of Γ1 and the definition of r, |x0 − λi| = |(λp + λp+1)/2 − λi| ≥ |(λp + λp+1)/2 − λr+1| ≥ λp−λr+1 2 ≥
λp 4
, where the first inequality follows the fact *i > r*. Thus, the second term is at most
∥E∥R T
−T

$$\|E\|\int_{-T}^{T}\frac{\sqrt{x_{0}^{2}+t^{2}}}{t^{2}+(\lambda_{p}/4)^{2}}d t\leq\tilde{O}(\|E\|);$$

see Section E.1 for the detailed estimation.

Similar to estimating the second term, the last term is also O˜(∥E∥). Combining the estimates on three parts of M1, we obtain M1 ≤ O˜r 2x λp δp
+ ∥E∥
. Consequently, by Lemma 3.1, we finally

have
$$F(z)\leq2F_{1}(z)=O(M_{1})=\tilde{O}\left(\|E\|+r^{2}x{\frac{\lambda_{p}}{\delta_{p}}}\right)\ \ \mathrm{as~desired.}$$

## 4 Empirical Results

In this section, we empirically evaluate the sharpness of our spectral-gap bound (Theorem 2.1) in real-world settings central to privacy-preserving low-rank approximation. We compare: (1)
the actual spectral error ∥A˜p − Ap∥, (2) our theoretical bound3 7∥E∥ · λp δp
, (3) and the classical Eckart–Young–Mirsky (EYM) bound 2(∥E∥ + λp+1). Each quantity is computed over 100 trials and 20 noise levels. Because prior bounds [15, 29, 30] apply only to Gaussian noise and involve unspecified constants, we exclude them from this evaluation.

Setting. We study three covariance matrices A from the UCI Machine Learning Repository [13]: the 1990 US Census (n = 69), the 1998 KDD-Cup network-intrusion data (n = 416), and the Adult dataset (n = 6). These matrices—henceforth Census, KDD, and Adult—are standard benchmarks in DP PCA [3, 11, 29]. The low-rank parameter p is chosen so that the Frobenius norm of Ap contains > 99% of the Frobenius norm of A, giving p = 10 for A = Census, p = 2 for A = KDD, and p = 4 for A = Adult [29, Section B]. Each matrix is perturbed with either GOE noise E1 or Rademacher noise E2, scaled by twenty evenly spaced factors ranging from 0 to 1. Note that with high probability [41, 43], ∥E1∥ = ∥E2∥ =
(2 + o(1))√n, so the gap condition 4∥Ek∥ < δp simplifies to 8
√*n < δ*p . For Census (n =
69, p = 10), we have δp ≈ 1433.99 > 8
√69 ≈ 66.45. For KDD (n = 416, p = 2), we get δp ≈ 351.3 > 8
√416 ≈ 163.2. For Adult (n = 6, p = 4), we find δp ≈ 37.02 > 8
√6 ≈ 19.6.

Hence 4∥Ek∥ < δp holds in all tested configurations. Evaluation. Each data matrix is preprocessed as follows: non-numeric entries are replaced with 0; rows shorter than the maximum length are padded with zeros; each row is scaled to unit Euclidean norm; and each column is centered to have zero mean. We compute the covariance matrix A :=
M⊤M, where M is the processed data matrix. For each configuration (A, Ek, p), we run 100 independent trials. In each trial, we perturb A with noise Ek ∈ {E1, E2} to form A˜ = A + Ek, compute its best rank-p approximation A˜p, and measure the spectral error ∥A˜p − Ap∥. We compare this with our bound 7∥Ek∥ · λp/δp and the classical EYM bound 2(∥Ek∥ + λp+1). Following standard practice, all reported values are averaged over 100 trials, with error bars shown for *Actual* Error and *Our Bound* (cap width = 3pt). Result and conclusion. Across all experiments—the 69 × 69 US Census, the 416 × 416 KDD- Cup, and the 6 × 6 Adult matrix—our bound closely matches the empirical error for both Gaussian and Rademacher noise (Figs. 1–2), consistently outperforming the classical EYM estimate. (Note: the error bars for Census and KDD are too small to see.) Over all three benchmark datasets, two distinct noise models, and twenty escalation levels per model, our spectral-gap estimate never deviates from the observed error by more than a single order of magnitude. This uniform tightness, achieved without any dataset-specific tuning, demonstrates that the bound of Theorem 2.1 is not merely sufficient but practically sharp across matrix sizes spanning two orders of magnitude and privacy-motivated perturbations spanning the entire operational range. Consequently, the bound can serve as a reliable, application-agnostic error certificate for low-rank covariance approximation in both differential-privacy pipelines and more general noisy-matrix workflows.

## 5 Conclusion And Future Work

We established new spectral norm perturbation bounds for low-rank approximations that explicitly account for the interaction between a matrix A and its perturbation E. Our results extend the Eckart—Young–Mirsky theorem, improving upon prior Frobenius-norm-based analyses. A key contribution is a novel application of the *contour bootstrapping* technique, which simplifies spectral perturbation arguments and enables refined estimates. Our bounds provide sharper guarantees for differentially private low-rank approximations with high probability spectral norm bounds that improve upon prior results. We also extended our approach to general spectral functionals, broadening its applicability. Several limitations and open questions remain. While spectral norm error bounds are standard and widely used in both theoretical and applied settings, can we extend our analysis to other structured metrics such as Schatten-p norm, the Ky Fan norm, or subspace affinity norm? Can our bounds be further refined for matrices with specific spectral structures, such as polynomial or exponential decay? What can be the threshold for the gap assumption so that one still obtains a meaningful bound beyond the Eckart–Young–Mirsky theorem?4 Additionally, real-world noise often exhibits structured dependencies—can our techniques be adapted to handle sparse or correlated perturbations?

4For an empirical comparison between our new bound and the Eckart–Young–Mirsky bound beyond the gap condition 4∥E∥ < δp, see Section C.

## Acknowledgments

This work was funded in part by NSF Award CCF-2112665, Simons Foundation Award SFI-MPS- SFM-00006506, and NSF Grant AWD 0010308.

## References

[1] Dimitris Achlioptas and Frank McSherry. Fast computation of low-rank matrix approximations. *Journal of the ACM (JACM)*, 54(2):9–es, 2007.

[2] U. Alon, N. Barkai, D. A. Notterman, K. Gish, S. Ybarra, D. Mack, and A. J. Levine. Broad patterns of gene expression revealed by clustering analysis of tumor and normal colon tissues probed by oligonucleotide arrays. Proceedings of the National Academy of Sciences of the United States of America, 96(12):6745–6750, 1999.

[3] Kareem Amin, Travis Dick, Alex Kulesza, Andres Munoz, and Sergei Vassilvitskii. Differentially private covariance estimation. *Advances in Neural Information Processing Systems*, 32, 2019.

[4] Y. Azar, A. Flat, A. Karlin, F. McSherry, and J. Saia. Spectral analysis of data. In Proceedings of the thirty-third annual ACM symposium on Theory of computing, pages 619–626, 2001.

[5] Zhidong Bai and Jack William Silverstein. Spectral analysis of large dimensional random matrices. Springer, 2009.

[6] James Bennett and Stan Lanning. The Netflix Prize. In *Proceedings of KDD cup and workshop*,
volume 2007, page 35. New York, NY, USA., 2007.

[7] Rajendra Bhatia. *Matrix analysis*, volume 169. Springer Science & Business Media, 2013. [8] Jeremiah Blocki, Avrim Blum, Anupam Datta, and Or Sheffet. The Johnson-Lindenstrauss transform itself preserves differential privacy. In 2012 IEEE 53rd Annual Symposium on Foundations of Computer Science, pages 410–419. IEEE, 2012.

[9] Avrim Blum, Cynthia Dwork, Frank McSherry, and Kobbi Nissim. Practical privacy: the sulq framework. In Proceedings of the twenty-fourth ACM SIGMOD-SIGACT-SIGART symposium on Principles of database systems, pages 128–138, 2005.

[10] Avrim Blum, John Hopcroft, and Ravindran Kannan. *Foundations of data science*. Cambridge University Press, 2020.

[11] Kamalika Chaudhuri, Anand Sarwate, and Kaushik Sinha. Near-optimal differentially private principal components. *Advances in neural information processing systems*, 25:989–997, 2012.

[12] C. Davis and W. M. Kahan. The rotation of eigenvectors by a perturbation. SIAM Journal on Numerical Analysis, 7:1–46, 1970.

[13] Dheeru Dua and Casey Graff. UCI machine learning repository. https://archive.ics.

uci.edu/ml, 2017.

[14] Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In *Theory of cryptography conference*, pages 265–284. Springer, 2006.

[15] Cynthia Dwork, Kunal Talwar, Abhradeep Thakurta, and Li Zhang. Analyze Gauss: Optimal bounds for privacy-preserving principal component analysis. In *Proceedings of the forty-sixth* annual ACM symposium on Theory of computing, pages 11–20, 2014.

[16] G. Eckart and G. Young. The approximation of one matrix by another of lower rank. Psychometrika 1, pages 211–218, 1936.

[17] Moritz Hardt. Robust subspace iteration and privacy-preserving spectral analysis. In *2013 51st* Annual Allerton Conference on Communication, Control, and Computing (Allerton), pages 1624–1626. IEEE, 2013.

[18] Moritz Hardt and Eric Price. The noisy power method: A meta algorithm with applications.

Advances in neural information processing systems, 27, 2014.

[19] Nicholas J. Higham. *Functions of Matrices: Theory and Computation*. SIAM, 2008. See §1.3 and §7.6 for the Cauchy–Dunford integral.

[20] R. A. Horn and C. R. Johnson. *Matrix Analysis*. Cambridge University Press, 2012. [21] M. Ivanovs, R. Kadikis, and K. Ozols. Perturbation-based methods for explaining deep neural networks: A survey. *Pattern Recognition Letters*, 150:228–234, 2021.

[22] M. Jirak and M. Wahl. Perturbation bounds for eigenspaces under a relative gap condition.

Proceedings of the American Mathematical Society, 148(2):479–494, 2020.

[23] R. Kannan, H. Salmasian, and S. Vempala. The spectral method for general mixture models.

SIAM Journal on Computing, 38(3):1141–1156, 2008.

[24] R. Kannan and S. Vempala. Spectral algorithms. Foundations and Trends in Theoretical Computer Science, 4(3-4):157–288, 2009.

[25] Michael Kapralov and Kunal Talwar. On differentially private low rank approximation. In Proceedings of the twenty-fourth annual ACM-SIAM symposium on Discrete algorithms, pages 1395–1414. SIAM, 2013.

[26] Tosio Kato. *Perturbation Theory for Linear Operators*. Classics in Mathematics. Springer, New York, NY, 1980.

[27] V. Koltchinskii and K. Lounici. Concentration inequalities and moment bounds for sample covariance operators. *Bernoulli*, 23:110–133, 2017.

[28] Vladimir Koltchinskii and Dong Xia. Perturbation of linear forms of singular vectors under Gaussian noise. In *High Dimensional Probability VII: The Cargese Volume* ` , pages 397–423. Springer, 2016.

[29] Oren Mangoubi and Nisheeth Vishnoi. Re-analyze Gauss: Bounds for private matrix approximation via Dyson Brownian motion. In *Advances in Neural Information Processing Systems*, volume 35, pages 38585–38599, 2022.

[30] Oren Mangoubi and Nisheeth K. Vishnoi. Private low-rank approximation for covariance matrices, Dyson Brownian Motion, and eigenvalue-gap bounds for Gaussian perturbations. J. ACM, 72(2), March 2025.

[31] Oren Mangoubi, Yikai Wu, Satyen Kale, Abhradeep Thakurta, and Nisheeth K Vishnoi. Private matrix approximation and geometry of unitary orbits. In *Conference on Learning Theory*, pages 3547–3588. PMLR, 2022.

[32] Sean O'Rourke, Van Vu, and Ke Wang. Random perturbation of low rank matrices: Improving classical bounds. *Linear Algebra and its Applications*, 540:26–59, 2018.

[33] Sean O'Rourke, Van Vu, and Ke Wang. Matrices with Gaussian noise: Optimal estimates for singular subspace perturbation. *IEEE Transactions on Information Theory*, 2023.

[34] Or Sheffet. Old techniques in differentially private linear regression. In Algorithmic Learning Theory, pages 789–827. PMLR, 2019.

[35] G. W. Stewart and Ji Guang Sun. *Matrix Perturbation Theory*. Academic Press, 1990. See Chap. III, §3.

[36] Phuc Tran and Nisheeth K. Vishnoi. Perturbation bounds for low-rank inverse approximations under noise. In Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS 2025), 2025.

[37] Phuc Tran and Van Vu. Davis–Kahan theorem under a moderate gap condition. *Communications in Contemporary Mathematics*, 2025. World Scientific, doi: 10.1142/S021919972550035X.

[38] Phuc Tran and Van Vu. New matrix perturbation bounds with relative norm: Perturbation of eigenspaces. *ArXiv preprint: 2409.20207*, 2026.

[39] Jalaj Upadhyay. The price of privacy for low-rank factorization. *Advances in Neural Information Processing Systems*, 31, 2018.

[40] Ramon Van Handel. On the spectral norm of Gaussian random matrices. Transactions of the American Mathematical Society, 369(11):8161–8178, 2017.

[41] Sabine Van Huffel and Joos Vandewalle. On the accuracy of total least squares and least squares techniques in the presence of errors on all data. *Automatica*, 25(5):765–769, 1989.

[42] Roman Vershynin. High-dimensional probability: An introduction with applications in data science, volume 47. Cambridge university press, 2018.

[43] Van Vu. Spectral norm of random matrices. *Combinatorica*, 27(6):721–736, 2007. [44] Martin J Wainwright. *High-dimensional statistics: A non-asymptotic viewpoint*, volume 48.

Cambridge university press, 2019.

[45] M.J. Wainwright. *High-Dimensional Statistics: A Non-Asymptotic view point*. Cambridge Series in Statistical and Probabilistic Mathematics, 2019.

[46] Hermann Weyl. Das asymptotische verteilungsgesetz der eigenwerte linearer partieller differentialgleichungen. *Mathematische Annalen*, 71(4):441–479, 1912.

Contents

| 1   | Introduction                                                                | 1   |    |
|-----|-----------------------------------------------------------------------------|-----|----|
| 2   | Main results                                                                | 3   |    |
| 3   | Proof outline                                                               | 6   |    |
| 4   | Empirical results                                                           | 9   |    |
| 5   | Conclusion and future work                                                  | 10  |    |
| A   | Limitations of prior approaches                                             | 15  |    |
| B   | Comparison of error metrics                                                 | 16  |    |
| C   | Empirical evaluation beyond gap assumption                                  | 17  |    |
| D   | Extensions of Theorem 2.1 and Theorem 2.2 to the symmetric matrices         | 18  |    |
| D.1 | Extension of Theorem 2.1 to the symmetric matrices                          |     | 18 |
| D.2 | Extension of Theorem 2.2 to the symmetric matrices                          |     | 20 |
| E   | Estimating integrals over segments                                          | 21  |    |
| E.1 | Estimating integrals over vertical segments for interaction-dependent bound | 21  |    |
| E.2 | Estimating integrals over horizontal segments                               |     | 24 |
| E.3 | Estimating integrals over vertical segments for non-interaction bound       |     | 24 |
| F   | Perturbation of matrix functionals - Theorem 2.3                            | 24  |    |
| G   | Some classical perturbation bounds                                          | 26  |    |
| H   | Notation                                                                    | 26  |    |

## A Limitations Of Prior Approaches

This section explains why existing perturbation methods fail to yield spectral norm bounds of the form ∥A˜p − Ap∥ that incorporate interaction between A and the perturbation E.

Eckart–Young–Mirsky: lack of interaction sensitivity. Let σ1 ≥ σ2 *≥ · · · ≥* σn ≥ 0 denote the singular values of A. The Eckart–Young–Mirsky theorem gives ∥A − Ap∥ = σp+1, and by the triangle inequality:
∥A˜p − Ap∥ ≤ ∥A − Ap∥ + ∥A˜ − A∥ + ∥A˜ − A˜p∥ ≤ σp+1 + ∥E∥ + ˜σp+1 ≤ 2(σp+1 + ∥E∥),
where the final step uses Weyl's inequality [46]. While this bound is assumption-free, it is uninformative in regimes where σp+1 ≫ ∥E∥, which are common in practice. The key limitation is that the triangle inequality treats A and E independently, failing to capture how structure or spectral gaps in A might mitigate the effect of E.

Mangoubi–Vishnoi: Frobenius only, spectral norm intractable. The strategy of [29, 30] models noise as a continuous-time matrix-valued Brownian motion:

$A(t):=A$. 
$\square$
A(t) := A + tE = A + B(t),
with eigen-decomposition

$A(t)=U(t)\,$Diag[$\lambda_{1}(t),\ldots,\lambda_{n}(t)$] U(t)${}^{\top}$
$$\geq\cdot\cdot\geq\lambda$$
$${\mathrm{re}}\;U(t)\;,$$
$$\operatorname{wh}_{\mathrm{nc}}$$
where U(t) = [ui(t)] and λ1(t) *≥ · · · ≥* λn(t). The rank-p approximation at time t is

$$A_{p}(t)=U(t)\operatorname{Diag}$$

Ap(t) = U(t) Diag[λ1(t), . . . , λp(t), 0*, . . . ,* 0]U(t)
⊤.

The total perturbation is then expressed as an integral:

* [16] A. A. K.  
$$\bar{A}_{p}-A_{p}=\int_{0}^{1}d A_{p}(t).$$
Using properties of Dyson Brownian motion and Ito calculus, they derive a Frobenius-norm identity: ˆ

$$\mathbb{E}\left\|\int_{0}^{1}dA_{p}(t)\right\|_{F}^{2}=\sum_{i=1}^{n}\int_{0}^{1}\left(\mathbb{E}\left[\sum_{j\neq i}\frac{(\lambda_{i}-\lambda_{j})^{2}}{(\lambda_{i}(t)-\lambda_{j}(t))^{2}}\right]+\left(\sum_{j\neq i}\frac{\lambda_{i}-\lambda_{j}}{(\lambda_{i}(t)-\lambda_{j}(t))^{2}}\right)^{2}\right).$$

$$\left|\begin{array}{l}{d t.}\\ {}\end{array}\right.$$
dt.

Bounding these expressions depends on repulsion properties of the eigenvalues; for GOE matrices, Weyl's inequality suffices, while for GUE matrices, stronger gap estimates are used. Although this method captures the spectral structure of A and interaction with E, it only yields Frobenius-norm bounds. Extending it to the spectral norm would require controlling

$$\|\bar{A}_{p}-A_{p}\|=\left\|\int_{0}^{1}d A_{p}(t)\right\|,$$

which entails bounding the operator norm of the full stochastic process. This requires detailed control over the dynamics of U(t) and λ(t), including their correlations—none of which are tractable with current techniques.

Moreover, for generalized functionals such as ∥fp(A˜) − fp(A)∥, the problem becomes even harder: one must analyze R 1 0 dfp(A(t)), which involves matrix-valued analytic functions under random perturbation, a setting far beyond existing random matrix tools.

In contrast, our approach bypasses these limitations by using a complex-analytic representation of spectral projectors that directly captures interaction between A and E, yielding sharp spectral norm bounds under broad assumptions.

## B Comparison Of Error Metrics

This section studies three common metrics for low-rank approximation under perturbation—namely:
- the spectral-norm error ∥A˜p − Ap∥, - the Frobenius-norm error ∥A˜p − Ap∥F , and - the "changein-error"
∥A − Ap*∥ − ∥*A − A˜p∥
.

We compare these metrics both empirically (through Monte Carlo simulations) and theoretically. Empirically, we examine how the metrics behave under Gaussian noise applied to both synthetic and real-world matrices (Figure 3). Theoretically, we analyze their interpretability and limitations, highlighting that while Frobenius norms capture aggregate error and change-in-error quantifies residual shifts, only the spectral norm controls worst-case subspace distortion. A simple 2 × 2 example (Example B.1) further illustrates how residual-based measures can completely mask subspace drift, underscoring the robustness and interpretability of the spectral norm for tasks such as private low-rank approximation. Empirical comparison of utility metrics. We perform three Monte Carlo experiments under additive Gaussian perturbations. The first uses a synthetic PSD matrix A ∈ R
50×50 with exponentially decaying eigenvalues λi = 0.8 i, and sets p = 5. The second and third use real-world covariance matrices derived from: - the 1990 US Census dataset (n = 69), - the 1998 KDD-Cup dataset (n = 416). All datasets are drawn from the UCI Machine Learning Repository [13] and have been widely used in private matrix approximation and PCA [30, 29, 11].

In each setting, we compute the best rank-p approximation Ap, perturb A with symmetric Gaussian noise of varying standard deviation σ, and measure:
1. Spectral norm deviation: ∥A˜p − Ap∥, 2. Frobenius norm deviation: ∥A˜p − Ap∥F ,
3. Change-in-error:
∥A − Ap*∥ − ∥*A − A˜p∥
.

As shown in Figure 3, the Frobenius norm error grows fastest, reflecting total energy deviation. The change-in-error metric remains much smaller and, in the real-world cases, nearly flat, suggesting it may fail to capture meaningful distortion. Notably, in the synthetic case (left), the spectral norm error closely tracks the change-in-error—despite their differing intent—which may result from nearalignment of the top subspaces. However, such behavior is not guaranteed in general.

Theoretical distinction between utility metrics. Frobenius norm bounds of the form ∥A˜p −
Ap∥F ≤ εF aggregate squared deviations across all directions, but may hide large errors in individual components. Spectral norm bounds ∥A˜p − Ap∥ ≤ ε directly constrain the worst-case deviation and are thus more reliable in sensitive applications such as differentially private PCA.

In contrast, residual-error metrics such as ∥A − Ap*∥ − ∥*A − A˜p∥ are commonly used for their
analytical convenience. However, they reflect only changes in residual energy and are insensitive to subspace movement. In particular, this metric can be nearly zero even when the top-p eigenspaces have shifted significantly. Given the spectral decompositions
the spectral decompositions  $A_{p}=U_{p}\operatorname{Diag}(\lambda_{1},\ldots,\lambda_{p},0,\ldots,0)\,U_{p}^{\top},\quad\tilde{A}_{p}=\tilde{U}_{p}\operatorname{Diag}(\tilde{\lambda}_{1},\ldots,\tilde{\lambda}_{p},0,\ldots,0)\,\tilde{U}_{p}^{\top},$  ange-in-error vanishes whenever $U_{p}U_{p}^{\top}\approx\tilde{U}_{p}\tilde{U}_{p}^{\top}$ and $\lambda_{p+1}$ is large. Such condition 
typical when noise E is small and p ≤ sr(A) := Pn
i=1 λi/λ1. Moreover, standard perturbation
typical when noise $E$ is small and $p\leq\operatorname{sr}(A):=\sum_{i=1}\lambda_{i}/\lambda_{1}$ results imply  $$\|U_{p}U_{p}^{\top}-\tilde{U}_{p}\tilde{U}_{p}^{\top}\|=\tilde{O}\left(\frac{\|E\|}{\lambda_{p}}+\frac{1}{\delta_{p}}\right).$$  **Example B.1 (Rank-1 rotation in $\mathbb{R}^{2}$).**_Let_  $$A=\begin{pmatrix}1&0\\ 0&0\end{pmatrix},\quad p=1,$$
$$[{\bf33},{\bf38}].$$

$$A=\begin{pmatrix}1&0\\ 0&0\end{pmatrix},\quad p=1,$$  I am not sure how to do. 
so that $A_{p}=A$. Define the rotated matrix_
_The rotated matrix_  $$\tilde{A}=R_{\theta}AR_{\theta}^{\top},\quad\text{where}\quad R_{\theta}=\begin{pmatrix}\cos\theta&-\sin\theta\\ \sin\theta&\cos\theta\end{pmatrix}.$$  _though the top eigenspace has rotated by $\theta$, the change 
$\bar{A}$ . 
Then A˜p = A˜, and although the top eigenspace has rotated by θ*, the change-in-error is zero:*
$||A-A_{p}||=||A-A_{p}||=0$.  _e in:_  $|\sin\theta|$, $||A_{p}-A_{p}||_{F}=\sqrt{2}|\sin\theta|$.  
Yet the true subspace drift is visible in: This example highlights the limitations of residual-based utility metrics and illustrates why spectral norm deviation provides a more reliable and interpretable signal of approximation quality under perturbation.

In summary, both our analysis and experiments support the use of the spectral norm as the most informative and robust error metric for evaluating private low-rank approximations. Unlike Frobenius and residual metrics, it captures the worst-case directional distortion and provides a tighter connection to subspace stability.

## C Empirical Evaluation Beyond Gap Assumption

In this section, we empirically compare (1) the actual spectral error ∥A˜p − Ap∥, (2) our theoretical bound 7∥E∥ · λp δp
, (3) and the classical Eckart–Young–Mirsky (EYM) bound 2(∥E∥ + λp+1) in the setting beyond the gap assumption that 4∥E∥ < δp.

Setup. We conducted a simulation on a covariance matrix A with n = 2000, derived from the Alon colon-cancer microarray dataset [2]. The low-rank parameter p is chosen so that the Frobenius norm of Ap contains > 95% of the Frobenius norm of A, giving p = 9 with λp ≈ 46.29. We first computed δp. Gaussian noise was then added in the form E = α · N (0, In), with α chosen over 11 evenly spaced values such that

$${\frac{\|E\|}{\delta_{p}}}\in\{0.05,0.10,\ldots,0.50\}.$$

For each α, we computed the following quantities:

- the true error: ∥A˜p − Ap∥,
- the classical EYM bound: 2(∥E∥ + σp+1),
- our bound: 7∥E∥ · λp
δp
,
- the ratios our bound true error and our bound classical bound .

$\overline{c}\|+\sigma_{p+1}$). 
Results. Table 2 summarizes the results. The ratio our bound true error remains remarkably stable even beyond the regime 4∥E∥ < δp (i.e., 
∥E∥
δp< .25), and our bound outperforms the classical bound precisely when 4∥E∥ < δp (i.e., ∥E∥
δp< .25).

| Table 2: Comparison of bounds under increasing noise levels.   |       |       |       |       |       |       |       |       |       |       |
|----------------------------------------------------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| ∥E∥/δp                                                         | 0.05  | 0.10  | 0.15  | 0.20  | 0.25  | 0.30  | 0.35  | 0.40  | 0.45  | 0.50  |
| our bound true error                                           | 90.17 | 88.27 | 87.02 | 89.83 | 89.44 | 87.81 | 88.39 | 89.29 | 87.08 | 87.26 |
| classical bound                                                | 0.20  | 0.40  | 0.60  | 0.79  | 0.98  | 1.17  | 1.36  | 1.53  | 1.70  | 1.88  |
| our bound                                                      |       |       |       |       |       |       |       |       |       |       |

## D Extensions Of Theorem 2.1 And Theorem 2.2 **To The Symmetric Matrices**

In this section, we extend Theorem 2.1 and Theorem 2.2 to the setting where A is a symmetric matrix. These extensions are naturally important since the data in real-world applications is often arbitrary, making it natural for the eigenvalues of A to span both signs. While singular value decomposition (SVD) could be used to apply Theorem 2.1 or Theorem 2.2, singular value gaps are typically small. By working directly with eigenvalues, we exploit the fact that the eigenvalue gap δk = λk − λk+1 is significantly large when λk · λk+1 < 0.

## D.1 Extension Of Theorem 2.1 **To The Symmetric Matrices**

To simplify the presentation, we assume that the eigenvalues (singular values) are different, so the eigenvectors (singular vectors) are well-defined (up to signs). However, our results hold for matrices with multiple eigenvalues. Let *A, E* be n × n real symmetric matrices, and let 1 ≤ p ≤ n denote the rank of approximation. Let λk be the kth largest eigenvalue of A and uk be the corresponding orthonormal eigenvector. Let A˜ := A + E. Let Ap, A˜p denote the best rank-p approximations of A and A˜ respectively. Define 1 ≤ k ≤ p such that the set of the top p singular values corresponds to {λπ(1), . . . , λπ(p)} = {λ1*, . . . , λ*k > 0 ≥ λn−(p−k)+1*, . . . , λ*n}. In other words, the pth singular value of A is either λk or |λn−(p−k)+1|. Let δi:= λi − λi+1, for i ∈ [n − 1]. Theorem 2.1 is extended to the following result. Theorem D.1 (Extension of Theorem 2.1 **to the symmetric matrices).** If 4∥E∥ ≤
min{δk, δn−(p−k)}, and 2∥E∥ < σp − σp+1*, then*

$$\left\|{\tilde{A}}_{p}-A_{p}\right\|\leq6\|E\|\left(\log\left({\frac{6\sigma_{1}}{\delta_{k}}}\right)+{\frac{\lambda_{k}}{\delta_{k}}}+\log\left|{\frac{6\sigma_{1}}{\delta_{n-(p-k)}}}\right|+{\frac{\left|\lambda_{n-(p-k)+1}\right|}{\delta_{n-(p-k)}}}\right)\right).$$

Note that when A is not PSD, {|λ˜1|*, . . . ,* |λ˜k|, |λ˜n−(p−k)+1|*, . . . ,* |λ˜n|} may not correspond to the p leading singular values of A˜. This issue is resolved by enforcing the singular-value gap condition σp − σp+1 > 2∥E∥. Indeed, by Weyl's inequality, given σp − σp+1 > 2∥E∥, we have

$$\begin{array}{l}{{\tilde{\lambda}_{k}\geq\lambda_{k}-\|E\|\geq\sigma_{p}-\|E\|=\sigma_{p+1}+\delta-\|E\|}}\\ {{\geq|\lambda_{n-(p-k)}|+\delta-\|E\|\geq|\tilde{\lambda}_{n-(p-k)}|+\delta-2\|E\|>|\tilde{\lambda}_{n-(p-k)}|,}}\end{array}$$

here δ = σp − σp+1. By a similar argument, we also have |λ˜n−(p−k)+1| > λ˜k+1. Therefore,

$\{\tilde{\lambda}_{\pi(1)},\tilde{\lambda}_{\pi(2)},\ldots,\tilde{\lambda}_{\pi(p)}\}=\{\tilde{\lambda}_{1}\geq\tilde{\lambda}_{2}\geq\ldots\geq\tilde{\lambda}_{k}>0\geq\tilde{\lambda}_{n-(p-k)+1}\geq\tilde{\lambda}_{n-(p-k)+2}\geq\tilde{\lambda}_{n-(p-k)+1}$
as we want. Note that the gap condition of eigenvalues cannot guarantee this fact. For example, consider the following matrices

$$A=\begin{pmatrix}30\sqrt{n}&0\\ 0&-28\sqrt{n}\end{pmatrix},E=\begin{pmatrix}-2\sqrt{n}&0\\ 0&-2\sqrt{n}\end{pmatrix},\;\;\mathrm{then}\;\tilde{A}=\begin{pmatrix}28\sqrt{n}&0\\ 0&-30\sqrt{n}\end{pmatrix}.$$

Here, clearly, S = {1}, S˜ = {1} and |λ1| is the largest singular value of A, but |λ˜1| is not the largest singular value of A˜ (λ˜1 is still the largest eigenvalue).

Proof of Theorem D.1 Let 1 ≤ k ≤ p be a natural number such that
{λπ(1), λπ(2), . . . , λπ(p)} = {λ1, λ2*, . . . , λ*k > 0 ≥ λn−(p−k)+1, λn−(p−k)+2*, . . . , λ*n}.

Thus, we can split Ap as Ak + Bp−k, in which

$$B_{p-k}=\sum_{n\geq i\geq n-(p-k)+1}\lambda_{i}u_{i}u_{i}^{\top}.$$
Similarly, A˜p = A˜k + B˜p−k. Therefore, A˜p − Ap  = A˜k + B˜p−k − Ak − Bp−k  ≤ A˜k − Ak  + B˜p−k − Bp−k  . Applying the contour bootstrapping argument onA˜k − Ak with contour Γ [1] and on B˜p−k − Bp−k with another contour Γ [2] (we define these contours later), we obtain ∥A˜k−Ak∥ 2 ≤ F [1] 1:= 1 2π RΓ[1]z(zI − A) −1E(zI − A) −1 |dz|, ∥B˜p−k−Bp−k∥ 2 ≤ F [2] 1:= 1 2π RΓ[2] z(zI − A) −1E(zI − A) −1 |dz|, and hence, A˜p − Ap  ≤ 2 F [1] 1 + F [2] 1 . (4)
We set Γ
[1] and Γ
[2] to be rectangles, whose vertices are Γ
[1] : (a0, T),(a1, T),(a1, −T),(a0, −T) with a0 := λk − δk/2, a1 := 2σ1, T := 2σ1; and Γ
[2] : (b0, T),(b1, T),(b1, −T),(b0, −T) with b0 := λn−(p−k)+1+δn−(p−k)/2, b1 := −2σ1, T := 2σ1.

Now, we are going to bound F
[1]
1. First, we split Γ
[1] into four segments:
- Γ1 := {(a0, t)| − T ≤ t ≤ T}. - Γ2 := {(*x, T*)|a0 ≤ x ≤ a1}. - Γ3 := {(a1, t)|T ≥ t ≥ −T}. - Γ4 := {(x, −T)|a1 ≥ x ≥ a0}.

Γ2 λk+1 λk λ1 Γ1 Γ3 Re (z) = 0 Γ4
Therefore,
$$F_{1}^{[1]}=\sum_{l=1}^{4}\int_{\Gamma_{l}}\left\|z(zI-A)^{-1}E(zI-A)^{-1}\right\|\,|dz|.$$  $$\left\|z(zI-A)^{-1}E(zI-A)^{-1}\right\|\leq\left\|E\right\|\frac{|z|}{\min_{i\in[n]}|z-\lambda_{i}|^{2}},$$  $$2\pi E^{[1]}\leq\left\|E\right\|\left(\sum_{i=1}^{4}N_{i}\right).$$
Notice that
we further obtain
$$2\pi F_{1}^{[1]}\leq\|E\|\left(\sum_{l=1}^{4}N_{l}\right),$$
in which Nl:= RΓl|z| mini |z−λi| 2 |dz| for l = 1, 2, 3, 4.

We use the following lemmas, whose proofs are delayed to the next section.

$\int_{\Gamma}\;\frac{|z|}{|z|+|z|^2}|dz|$ . 
Lemma D.2. Under the assumption of Theorem *D.1,*

$$N_{1}\leq{\frac{2\pi a_{0}}{\delta_{k}}}+4\log{\frac{\pi}{2}}$$


3T
δk
 .

Lemma D.3. Under the assumption of Theorem *D.1,*

$$N_{3}\leq\frac{\pi a_{1}}{|a_{1}-\lambda_{1}|}+4\log\left|\frac{3T}{a_{1}-\lambda_{1}}\right|.$$
Lemma D.4. Under the assumption of Theorem *D.1,*
$$N_{2},N_{4}\leq{\frac{\sqrt{2}(a_{1}-a_{0})}{T}},$$

Since *p < n*, then k + 1 > n − (p − k) + 1 and hence k + 1 ∈ { / π(1)*, . . . , π*(p)}. It means |λk+1| ≤ λk. Thus 0 ≤ a0 ≤ λk, and hence

$$N_{1}\leq\frac{2\pi\lambda_{k}}{\delta_{k}}+4\log\left|\frac{6\sigma_{1}}{\delta_{k}}\right|.$$
By the setting that a1 = T = 2σ1,
 $N_2,N_4\leq\frac{\sqrt{2}a_1}{T}=\sqrt{2},$  $N_3\leq\frac{2\pi\sigma_1}{2\sigma_1-\lambda_1}+4\log\left|\frac{3T}{a_1-\lambda_1}\right|\leq\frac{2\pi\sigma_1}{\sigma_1}+4\log\left|\frac{6\sigma_1}{\sigma_1}\right|=2\pi+4\log6.$  I am not sure what is. 
Thus, using above estimates, we obtain
$$F_{1}^{[1]}\leq\frac{|E|}{2\pi}\left(2\pi+4\log6+2\sqrt{2}+\frac{2\pi\lambda_{k}}{\sigma_{k}}+4\log\left|\frac{6\sigma_{k}}{\sigma_{k}}\right|\right)\tag{5}$$ $$\leq\frac{|E|}{2\pi}\left(15\log\left|\frac{6\sigma_{k}}{\sigma_{k}}\right|+\frac{2\pi\lambda_{k}}{\sigma_{k}}\right)$$ $$\leq3|E|\left(\log\left|\frac{6\sigma_{k}}{\sigma_{k}}\right|+\frac{\lambda_{k}}{\sigma_{k}}\right)\,.$$  Applying a similar argument on contour $\Gamma^{[2]}$, we obtain 
$$F_{1}^{[2]}\leq3\|E\|\left(\log\left|{\frac{6\sigma_{1}}{\delta_{n-(p-k)}}}\right|+{\frac{\left|\lambda_{n-(p-k)+1}\right|}{\delta_{n-(p-k)}}}\right).$$
(6)  $$\begin{array}{l}\small\mathbf{(6)^{n}}\end{array}$$ . 
Combining (4), (5) and (6), we complete our proof.

## D.2 Extension Of Theorem 2.2 **To The Symmetric Matrices**

Let A be a symmetric matrix with eigenvalues λ1 ≥ λ2 *≥ · · · ≥* λn, in which λn is not necessarily positive. Recall the setting from the previous section that 1 ≤ k ≤ p is the positive integer such that the set of the top p singular values is {λπ(1), . . . , λπ(p)} = {λ1*, . . . , λ*k > 0 ≥ λn−(p−k)+1*, . . . , λ*n}. To extend Theorem 2.2, we first generalize the definition of the halving distance r and *interaction term* x as follows. Let r1, r2 respectively be the smallest positive integer satisfying λk 2 ≤ λk − λr1+1, and |λn−(p−k)+1| 2 ≤ λn−r2+1 − λn−(p−k)+1. Define the "halving distance" r := max{r1, r2}. Next, let x1 := max1≤i,j≤r1|u
⊤
i Euj | and x2 := max1≤i,j≤r2 |u
⊤
n−i+1Eun−j+1|. Define the interaction parameter x¯ := max{x1, x2}.

Theorem D.5 (Extension of Theorem 2.2 **to the symmetric matrices).** Assume that 4∥E∥ ≤
min{δk, δn−(p−k)} and 2∥E∥ < σp − σp+1*, then*

$$\left\|{\bar{A}}_{p}-A_{p}\right\|\leq12\left(\|E\|+r^{2}{\bar{x}}\right)\left(\log\left({\frac{6\sigma_{s}}{\delta_{k}}}\right)+\log\left({\frac{6\sigma_{s}}{\delta_{n-(p-s)}}}\right)\right)+30r^{2}{\bar{x}}\left({\frac{\lambda_{s}}{\delta_{k}}}+{\frac{|\lambda_{n-(p-k)+1}|}{\delta_{n-(p-k)}}}\right).$$

Proof of Theorem D.5 First, we still split (A˜p, Ap) into (Ak, Bp−k, A˜k, B˜p−k) and apply the contour bootstrapping argument on A˜k − Ak
 ,
B˜p−k − Bp−k
. We also obtain

$$\left\|{\bar{A}}_{p}-A_{p}\right\|\leq2\left(F_{1}^{[1]}+F_{1}^{[2]}\right).$$