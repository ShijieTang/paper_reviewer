# The Dark Side Of The Forces: Assessing Non-Conservative Force Models For Atomistic Machine Learning

Filippo Bigi 1 Marcel F. Langer 1 **Michele Ceriotti** 1

## Abstract 1. Introduction

The use of machine learning to estimate the energy of a group of atoms, and the forces that drive them to more stable configurations, has revolutionized the fields of computational chemistry and materials discovery. In this domain, rigorous enforcement of symmetry and conservation laws has traditionally been considered essential. For this reason, interatomic forces are usually computed as the derivatives of the potential energy, ensuring energy conservation. Several recent works have questioned this physically constrained approach, suggesting that directly predicting the forces yields a better trade-off between accuracy and computational efficiency, and that energy conservation can be learned during training. This work investigates the applicability of such nonconservative models in microscopic simulations. We identify and demonstrate several fundamental issues, from ill-defined convergence of geometry optimization to instability in various types of molecular dynamics. Given the difficulty in monitoring and correcting the lack of energy conservation, direct forces should be used with great care. We show that the best approach to exploit the acceleration they afford is to use them in conjunction with conservative forces. A model can be pre-trained efficiently on direct forces, then fine-tuned using backpropagation. At evaluation time, both force types can be used together to avoid unphysical effects while still benefitting almost entirely from the computational efficiency of direct forces.

Interatomic potentials model the microscopic interactions between atoms, and determine - directly or by modulating the response to thermal excitations - the stability and reactivity of molecules and materials. Over many decades, interatomic potentials have been used in Monte Carlo (Metropolis et al., 1953) and molecular dynamics (MD) (Andersen, 1980) simulations, geometry optimization, and other techniques, allowing a mechanistic study of the atomic-scale behavior and properties of molecules, materials, and biological systems (Allen & Tildesley, 2017; Tuckerman, 2008).

While traditional interatomic potentials are based on simple physically inspired functional forms, the last decade has seen machine-learned interatomic potentials (MLIPs) obtain remarkable accuracies at a high level of computational efficiency, most often by learning reference energies and forces from much slower first-principle quantum mechanical calculations (Behler, 2021; Unke et al., 2021). At first, such machine-learned interatomic potentials were trained on quantum mechanical data specific to the chemical system of interest. Over the last years, more diverse datasets have emerged, consisting of up to billions of targets, and spanning much of the periodic table (Chanussot et al., 2021; Tran et al., 2023; Barroso-Luque et al., 2024; Schmidt et al., 2024). This has not only led to an increase in the complexity of the models, going from linear or kernel regression to deep graph neural networks, but it has also pushed many recent models to abandon some of the underlying physical symmetries of interatomic potentials in favor of simpler - and potentially more scalable and efficient - architectures (Pozdnyakov & Ceriotti, 2023; Neumann et al., 2024; Qu & Krishnapriyan, 2024). In such models, symmetries are learned from data, aided by data augmentation.

Some recent models for interatomic potentials also disregard the property of energy conservation (Gasteiger et al., 2021; Neumann et al., 2024; Liao et al., 2023). While conservative models calculate interatomic forces as the derivatives of a total energy with respect to the positions of the atoms, non-conservative models predict them directly, therefore breaking this constraint. Although this practice can lead to more computationally efficient neural networks, the use of such models in practical atomistic simulations has not yet 1 been studied in detail. This work compares conservative and non-conservative machine-learned interatomic potentials. Following a brief review of the most common types of potentials and their applications, we discuss the theoretical implications of using non-conservative forces to drive atomistic simulations, highlighting potential shortcomings. We then demonstrate the impact of these conceptual problems in several case studies, highlighting the pitfalls of using non-conservative interatomic forces in production, and demonstrating how, contrary to the case of geometric symmetry breaking, it is difficult to monitor and correct the impact of non-conservative models. Instead, it might be preferable to supplement a conservative model with direct force predictions, using them to accelerate simulations that have well-defined, energyconserving forces as the ground truth.

## 2. Background And Related Work

2.1. Interatomic potentials and atomistic simulations An interatomic potential V describes the potential energy between the N atoms of a molecule or material as a function of their positions {ri}
N
i=1 and chemical nature {ai}
N
i=1:

$$V(\{\mathbf{r}_{i},a_{i}\}_{i=1}^{N})\,.$$
i=1). (1)
While some applications, such as Monte Carlo simulations or energy difference calculations (for transition state or defect energies, material stability, etc.) need only the values of V , the most popular uses of interatomic potentials require the evaluation of interatomic forces, defined as the negative derivatives of V with respect to the atomic positions:

$$\mathbf{f}_{j}=-\partial V(\{\mathbf{r}_{i},a_{i}\}_{i=1}^{N})\,/\,\partial\mathbf{r}_{j}\,.$$
i=1)∂rj . (2)
The most notable applications which make use of interatomic forces are: Geometry optimization. This technique consists of finding one or more minima of the potential energy surface V to identify the preferred structure of a microscopic system. Both local and global optimization algorithms are employed, and the vast majority use the forces fj both during optimization and as a stopping criterion. Similar methods are used to identify saddle points of V , which are associated with the energy barriers that determine the rate of chemical reactions (Henkelman et al., 2000). Molecular dynamics. MD aims at simulating the behavior of a microscopic system by solving its classical equations of motion numerically. Using a time step ∆t, the simplest forms of molecular dynamics propagate a discretized version of Hamilton's equations (Verlet, 1967). Over more than 50 years, many variants of molecular dynamics have emerged, with various goals (sampling different thermodynamic ensembles (Andersen, 1980), improving sampling efficiency, observing rare events (Laio & Parrinello, 2002), or accounting for nuclear quantum effects (Chandler & Wolynes, 1981), etc.). In this work, we focus on some of the simplest variants: MD in the NVE ensemble. The number of atoms N, volume V and total energy E (kinetic plus potential) are kept constant, describing the behavior of an isolated system. Usually, this is accomplished by using the velocity Verlet integrator (Verlet, 1967) with a short time step ∆t. MD in the NVT ensemble. Here, the goal is to simulate a system at constant temperature T. This is achieved by modifying the dynamics with so-called thermostats. *Local* thermostats apply velocity corrections on each atom independently (Bussi & Parrinello, 2007). While they correctly sample the desired constant-temperature ensemble and can be used to evaluate static, equilibrium properties, this comes at the cost of disrupting the dynamics of the system. *Global* thermostats (Bussi et al., 2007) address this shortcoming by modifying the atomic velocities in a concerted manner, making it possible to investigate time-dependent properties (Bussi & Parrinello, 2008).

$$(\mathbf{l})$$

Many other applications of interatomic potentials (such as lattice dynamics, phonon calculations, etc.) require the calculation of forces and their higher-order derivatives. However, for simplicity, this work focuses on the most common use cases, discussed above.

## 2.2. Physical Symmetries

$$(2)^{\frac{1}{2}}$$

Interatomic potentials obey a number of physical symmetries and constraints.

E(3)-invariance. Interatomic potentials are invariant under the transformations of the Euclidean group in three dimensions E(3), including translations, rotations, and reflections. Mathematically, given a group element g ∈ E(3) that acts on all positions r,

$$V(\{g\cdot\mathbf{r}_{i},a_{i}\}_{i=1}^{N})=V(\{\mathbf{r}_{i},a_{i}\}_{i=1}^{N})\,.$$
(3) $$\binom{3}{4}$$. 
i=1). (3)
Permutation invariance. Interatomic potentials are invariant with respect to permutations of atom indices, i.e.,

$$V(..,\mathbf{r}_{i},a_{i},..,\mathbf{r}_{j},a_{j},..)=V(..,\mathbf{r}_{j},a_{j},..,\mathbf{r}_{i},a_{i},..).\tag{4}$$

Energy conservation. Interatomic forces are conservative, i.e., their mechanical work W over a closed loop is zero:

$$W=\oint\sum_{j=1}^{N}\mathbf{f}_{j}\cdot\mathrm{d}\mathbf{r}_{j}=0\,.$$
$$(S)$$

This is a direct consequence of (2). The term conservative alludes to the fact that an isolated physical system in which only conservative forces act obeys the principle of conservation of mechanical energy.

## 2.3. Interatomic Potentials Via Graph Neural Networks

For many years, most machine learning interatomic potentials made use of feature-based models (Behler & Parrinello, 2007; Bartok et al. ´ , 2010) trained on energies and forces from quantum mechanical calculations. This combination of physically inspired descriptors (Musil et al., 2021; Langer et al., 2022) and classical machine learning methods (linear and kernel models, as well as shallow neural networks) delivers good accuracy and computational efficiency on small datasets, usually created for a specific application in chemistry and/or materials science. However, recent efforts to build much larger and more comprehensive datasets (Deng et al., 2023; Schmidt et al., 2024; Eastman et al., 2023; Tran et al., 2023; Chanussot et al., 2021), have made it apparent that graph neural networks exhibit superior accuracy and scalability compared to earlier models (Batzner et al., 2022), as they respect the permutation symmetry of interatomic potentials while delivering rich and flexible representations of the atomic structures. For a review of graph neural networks, we redirect the interested reader to Zhou et al. (2020), while their applications to interatomic potentials is detailed in Duval et al. (2023).

## 2.4. Breaking Physical Constraints

Several recent architectures do not enforce all physical constraints listed in 2.2, aiming to increase expressivity, ability to scale to large datasets, and computational efficiency. Efficient inference is crucial to enable the length and time scales required for practical molecular dynamics simulations, as forces must be evaluated at every time step. Rotationally unconstrained models. In the case of rotational symmetry this approach, combined with training-time rotational augmentation, has been shown to yield accurate and efficient models (Pozdnyakov & Ceriotti, 2023; Neumann et al., 2024; Qu & Krishnapriyan, 2024) with negligible, or easily controllable, impact on physical observables (Langer et al., 2024). Non-conservative models. The conservative property for forces in Eq. (5) is satisfied if and only if there exists a function (usually named the potential energy) of which the forces are the spatial derivatives, see Eq. (2). Therefore, a machine learning model that calculates interatomic forces as derivatives of the potential energy according to Eq. (2) will be trivially conservative. The required derivatives, the gradient of the scalar V , can be obtained efficiently (i.e.,
with the same asymptotic computational cost as the energy prediction) with automatic differentiation (Griewank & Walther, 2008). Nevertheless, differentiation incurs a computational overhead, typically 2−3× for inference and 3× for training; the exact theoretical factors are discussed further in Appendix B. This overhead can be avoided by directly predicting forces during the forward pass. However, by removing the relationship between forces and energy defined in Eq. (2), such models do not enforce energy conservation. The possibility of performing a direct, nonconservative, evaluation of the forces was realized early (Li et al., 2015), but used only sparsely until recently, when it was applied to equivariant graph neural networks such as GemNet (Gasteiger et al., 2021) and several more recent universal machine learning interatomic potentials including ORB (Neumann et al., 2024) and Equiformer (Liao et al., 2023), which are trained on large datasets and promise broad applicability for chemical modeling. Some of these architectures also perform direct evaluation of stresses, which leads to the violation of conservation of enthalpy (see App. G.3).

The impact of lack of energy conservation for practical simulations and property prediction has recently been investigated in related works: Eissler et al. (2025) probe the limits of unconstrained architectures and find that lack of energy conservation becomes more pronounced for larger target systems. Loew et al. (2024) find that direct-force models perform poorly at the prediction of phonon properties, which require second derivatives of the potential energy surface. Fu et al. (2025) observe a better correlation between test set error and downstream predictive performance, for instance for phonons, for models that can perform stable MD simulations, i.e., that are able to conserve energy.

## 2.5. Use Of Forces For Training

Machine learning models of V are typically trained jointly on energy and force labels. The relative weight of these labels in training, or in the extreme case, whether to train exclusively on energy or on forces, has been discussed extensively, both for interatomic potentials and diffusion models for atomistic systems (Chmiela et al., 2018; Christensen & von Lilienfeld, 2020; Wang et al., 2024; Ren et al., 2024).

One important consideration is that focusing on energies or forces emphasizes different components of a potential energy surface, with consequences that are not directly visible when inspecting training and validation accuracies. To a first approximation, the distortions observed in NVT MD simulations - a common strategy to build training sets - can be interpreted as a collection of quasi-harmonic oscillators of different frequency ω, with high-frequency modes associated with short-range covalent bonding, and low-frequency ones associated with collective motions, which are often the most relevant for applications. The statistical mechanics of a harmonic oscillator imply that ⟨V ⟩ ∝ 1, while ⟨f 2⟩ ∝ ω 2
(see Appendix A). In other terms, the largest contributions to the forces from a thermally sampled dataset come from highfrequency modes that are hard to integrate and may lead to instabilities in the dynamics, while the contribution from the slow modes, which are hard to sample, but usually the most relevant, are under-emphasized. The potential energy provides a more balanced representation of the different molecular time and length scales. The question of relative energy and force weights during training is related to, but distinct from, the question of learning energy and forces consistently, i.e., enforcing that the predicted forces are exactly the gradient of the predicted energy. Models targeting mean-field energies, as in coarsegraining (Wang et al., 2019) and centroid MD (Musil et al., 2022), usually train exclusively on estimates of the mean forces, but use a conservative formulation as these forces are defined as derivatives of a thermodynamic potential that is difficult to evaluate explicitly. Disregarding the connection with reference energies may also be beneficial when the two targets are subtly inconsistent, because of convergence issues, the use of heterogeneous calculation settings, or numerical techniques like Fermi smearing (Marzari et al., 1999), which can be required for metallic systems. This may contribute to the empirical observation that non-consistent training is preferable in datasets like OC20 (Chanussot et al., 2021) and OC22 (Tran et al., 2023) that contain a high fraction of metallic configurations.

## 3. Theory

Some of the consequences of using non-conservative models descend directly from their mathematical formulation and can be used to foresee their impact on typical applications.

## 3.1. Measuring Non-Conservative Behavior

If a vector field f is the derivative of a smooth scalar function V , then its Jacobian J (the Hessian of V ) contains the mixed second derivatives of V , and must therefore be symmetric:

$$J_{i\alpha,j\beta}={\frac{\partial\mathbf{f}_{i\alpha}}{\partial\mathbf{r}_{j\beta}}}={\frac{\partial\mathbf{f}_{j\beta}}{\partial\mathbf{r}_{i\alpha}}}=J_{j\beta,i\alpha}\,.$$

In order to quantitatively capture the amount of nonconservation in a specific force prediction from a trained model, it is then possible to compare the Frobenius norm (or any other matrix norm) of the antisymmetric component of the Jacobian to that of the Jacobian itself:

$$\lambda={\frac{\|\mathbf{J}_{\mathrm{anti}}\|_{\mathrm{F}}}{\|\mathbf{J}\|_{\mathrm{F}}}}\,,$$
, (7)
where Janti = (J − J
⊤)/2. λ then defines a metric going from 0 for conservative forces to 1 for forces that have no conservative component. λ can also be computed only for entries associated with an atom i, or a pair of atoms ij, providing a finer-grained assessment of the violation of Eq. 6, as seen in Figure 1. This local symmetry breaking can also be measured by integrating the work done by the forces over a closed loop, that is bound to be zero (within integration error) for a conservative force field (Eq. 5). An explicit test of non-conservative behavior can also be implemented by monitoring the total energy in an NVE MD simulation, or equivalently a conserved quantity that keeps track of the heat flux associated with the thermostat (Bussi & Parrinello, 2007) in an NVT simulation. To find the power P (energy per unit time) injected by the non-conservative forces during a molecular dynamics trajectory, it is sufficient to calculate the average rate of change in the conserved quantity C during a section of the trajectory, P = ∆C/∆t.

## 3.2. Side-Effects Of Direct Force Prediction

Predicting forces as the derivatives of a translationally invariant potential ensures that the total net force acting on all atoms is zero, and that for a potential that is rotationally invariant the torque on an isolated molecule is zero. A direct force model - irrespective of whether it is E(3) invariant - does not have the same guarantees. These spurious effects are easy to remedy, by subtracting the total force and torque from each prediction. This technique is used by ORB (Neumann et al., 2024), and we also adopt it here. There is another non-trivial consequence of using a directforce prediction architecture. Whereas the potential energy is usually estimated as the sum of atomic contributions but is a global property, forces are atom-centered. When predicting them as derivatives, many atomic environments contribute to the force fi on each atom. When predicting directly, only the i-atom centered environment contributes to fi. Hence, direct force models can be expected to be affected more directly by the geometric degeneracies of lowbody-order atom-centered descriptors (Pozdnyakov et al.,
2020) and do not benefit from the same extended effective interaction range as conservative forces (Artrith et al., 2011). We discuss these effects in Appendix C, presenting some empirical evidence that direct force models require a larger range to match the force accuracy of a comparable conservative model.

$$(6)$$

## 3.3. Effects On Geometry Optimization

$$\left(7\right)$$

In order to assess the stability of materials or molecules at low temperature, a common approximation is to search for minimum-energy configurations. This can be achieved by minimizing the potential energy V as a function of the atomic positions - with most of the widely used algorithms relying (in some cases exclusively) on the value of the gradient. The lack of a consistent potential energy is problematic for most optimization schemes: those which require an explicit evaluation of the objective function (e.g., to perform line searches) cannot be used; those that just "follow the forces", relying on the vanishing magnitude of the force as a stopping criterion, can fail because non-conservative forces can keep driving indefinitely in the same direction, e.g., following closed loops with negative total work.

## 3.4. Effects On Molecular Dynamics

It is not uncommon to observe violations of energy conservation in MD simulations, since finite-timestep integrators violate the exact correspondence between the change in kinetic and potential energy along a trajectory. This is usually accepted, because (1) in well-designed simulations this leads to small fluctuations, and not to a run-away effect; (2)
the notion of a *shadow Hamiltonian* (Hairer et al., 2006) ensures that simulations reach a steady state that is "statistically close" to that generated by an exact integrator; (3) thermostatting techniques can relatively easily control small integration errors, so that structural and dynamical observables are not affected significantly (Morrone et al., 2011). The fact that no underlying Hamiltonian can be defined for the dynamics generated by a non-conservative force field suggests that, in this case, artifacts might be more pronounced and harder to correct. For example, the symplectic nature of Hamiltonian dynamics is no longer valid (this is easy to see, for example, from Eq. 5.2, Chapter 6 of Hairer et al. (2006)), and the theorem of equipartition of energy (whose proof is also based on the existence of a Hamiltonian (Pathria, 2017)) does not apply. As we shall see, this is what we observe empirically.

## 3.5. Learning Conservative Behavior

The standard approach to making symmetries more easily learnable from data is to employ data augmentation at training time: A random element of the underlying symmetry group, for instance rotations, is selected and applied for every sample or mini-batch. This approach has been successfully employed in the domain of MLIPs, as well as in a range of other applications including computer vision (Quiroga et al., 2020). However, it is only applicable to explicit geometric symmetries, and therefore not to energy conservation, which is not a symmetry with respect to inputs, but rather a symmetry of derivatives, as discussed in Section 3.1. Nevertheless, we briefly consider different schemes to promote energy conservation during training. One approach is to include the measure of Jacobian symmetry λ, Equation (7), as a term in the loss function. This faces severe practical issues: With automatic differentiation, computing J requires multiple (3N in the absence of sparsity or stochastic approximations) evaluations of the potential.

An alternative approach is to train both a conservative and non-conservative predictor and adding a force-matching loss term, or simply training both on the same forces labels. We will discuss this approach in further detail in Section 4.8. It is important to note that, in both strategies, energy conservation can be trained both on labeled and unlabeled data.

## 4. Results

Having introduced the subject of non-conservative force fields and discussed the potential pitfalls that might be incurred when using them in practice, we will now examine their effect on a range of applications, using liquid water as the main, paradigmatic example.

## 4.1. The Models

In order to substantiate our empirical observations, we perform our experiments on multiple models. Our main examples rely on the rotationally unconstrained PET architecture (Pozdnyakov & Ceriotti, 2023), trained from scratch on the bulk water dataset of Ref. 22 using both a conservative (PET) and non-conservative (PET-NC) architecture.

Additionally, we train "PET-M" to predict both conservative and non-conservative forces. To assess the implications of a direct prediction of forces in the most favorable possible context, we primarily use the best-performing, customtrained models. We also show some results for the nonconservative ORB-v2 model (Neumann et al., 2024), which is currently state-of-the-art for several materials prediction benchmarks. Even though ORB is not trained on this specific dataset, and is therefore at a clear disadvantage, it provides an indication of the relevance of the issues we discuss. In the appendices, we also discuss several other architectures, including a "legacy" SOAP-BPNN architecture, as well as pre-trained foundation models, MACE-MP- 0 (Batatia et al., 2023), SevenNet (Park et al., 2024), and EquiformerV2 (Liao et al., 2023), which we apply to a few diverse materials in Appendix G. A table describing all employed models, along with full details on the different architectures, can be found in Appendix D.

## 4.2. Accuracy

In terms of sheer accuracy, see Table 1, our tests confirm that forces provide very useful information to train an interatomic potential, in particular for a dataset containing relatively large configurations. Using forces in the training of a conservative model dramatically improves the accuracy of energy predictions with just a minor degradation in the accuracy for f with respect to a model trained only on forces. A non-conservative force model exhibits about 30% higher force error than a conservative architecture, and including a separate energy head leads to lower error than a model trained just on V - indicating that the sharing of weights within the architecture is beneficial. We also show results for the PET-M hybrid architecture, which makes both conservative and non-conservative force predictions. Its accuracy is less than 10% worse than the best models for either architectures. As we will discuss in Section 4.8, this is an excellent way to exploit non-conservative forces in simulations. Before doing so, however, we will assess the behavior of purely non-conservative models.

## 4.3. Non-Conservative Behavior

The asymmetry of the Jacobian is the most direct, pointwise measure of non-conservative behavior. Different nonconservative models show widely different values of λ - 0.015 for ORB, 0.017 for Equiformer, 0.032 for SOAP- BPNN-NC and 0.004 for PET-NC, computed on a few water structures from the test set of Cheng et al. (2019). As we shall see, the magnitude of λ correlates qualitatively with the stability of the models in simulations. The symmetry of J applies separately to each pair of atoms, and so it is possible to extract further insights by computing the norm of the antisymmetric part of each block Jij resolved for different atomic pairs and plotted as a function of the interatomic distance (Figure 1). The asymmetry is also present for "on-site" blocks, i.e., swapping only the Cartesian coordinates used in the derivatives for a given atom; the relative magnitude of Jasym is small compared to the magnitude of the Jacobian; the asymmetric component between atoms i and j decays with the interatomic distance more slowly than the absolute magnitude of J (which has been used as a measure of the interactions between pairs of atoms (Herbold & Behler, 2022)), and in the intermediate regime around 6A it becomes comparable in size - so ˚
that the pair-resolved Jacobian asymmetry λij approaches 1 for large interatomic distances. This latter observation has important implications when applying these models in simulations, as the impact of non-conservative behavior on different atomic-scale processes is not uniform, and it tends to be larger - in a relative sense - for collective processes involving long-range correlations.

| ARCH.   | TYPE   | TRAINING   | MAE(V )   | MAE(f)   |
|---------|--------|------------|-----------|----------|
| PET     | -      | V          | 4.7       | -        |
| PET     | C      | f          | -         | 18.6     |
| PET     | C      | V,f        | 0.55      | 19.4     |
| PET     | NC     | f          | -         | 24.3     |
| PET     | NC     | V,f        | 1.42      | 24.8     |
| PET-M   | C      | V,f        | 0.59      | 20.2     |
| NC      | 26.7   |            |           |          |

0 2 4 6 8 10 rij / Å
10 4 10 3 10 2 10 1 10 0 10 1 10 2 |Jij| |Jij Jji|/2 ij 10 4 10 3 10 2 10 1 10 0 10 1 10 2 J 
/ 
e V/Å
2

Non-conservative behavior can also be demonstrated by computing the work along a closed path (see Appendix E). Given that the choice of the path is arbitrary, we think it is more relevant to quantify the practical implications of this issue in terms of an energy drift in molecular simulations.

## 4.4. Constant-Energy Molecular Dynamics

Let us now consider the use of non-conservative MLIPs in the context of MD simulations. Also in this case, we focus our analysis on the best-performing models for liquid water, PET and PET-NC. A more thorough comparison, including several foundation models and a few homogeneous and heterogeneous material structures, is discussed in Appendix G, and consistently corroborates the observations we make here. Constant-pressure simulations, which, if performed with direct-stress models, break conservation of *enthalpy* and therefore lead to drift in the volume of the simulation, are shown in App. G.3 using the general-purpose PET-MAD potential (Mazitov et al., 2025). Given that non-conservative models lack a well-defined conserved quantity by construction, we rely on indirect measurements of the sampled ensemble. We consider in particular the kinetic temperature T = 2K/(3N kB), where N is the number of particles considered. This is just a rescaling of the kinetic energy K; its ensemble average should correspond to the target temperature for NVT trajectories (300 K in these tests), and to a value in its vicinity for NVE trajectories initialized from a thermally equilibrated configuration. As a sensitive indicator of the dynamical behavior of the system, we compute the Fourier transform of the velocityvelocity correlation function, cˆvv(ω). Its peaks are closely related to the density of vibrational modes and to infrared and Raman spectra, and its ω → 0 limit is proportional to the diffusion coefficient.

300 400 500 600 700 800 ORB, NVE PET, NVE
PET-NC, NVE PET-M, NVE MTS-8 T 
/ 
K

0 10000 20000 30000 40000 step
The failure of the non-conservative model in NVE dynamics is apparent in Figure 2. Whereas for a conservative potential the kinetic temperature fluctuates around the initial value, the spurious work associated with non-conservative forces leads to a large drift of T: To put it on a human scale, this unphysical drift corresponds to a rate of heating of about 7'000 billion degrees per second for the custom-trained PET-NC model, and another 10 times larger for the general-purpose ORB model. This spurious energy flow is a clear signature of non-conservative behavior, and makes direct-force models entirely useless for constant-energy simulations. This runaway increase of the kinetic energy can be mitigated - whenever an auxiliary model is available to evaluate the potential energy - by adjusting the velocities to enforce energy conservation artificially (see Appendix G.4). Similar to what we will discuss for constant-temperature simulations, the trajectories are still affected by large artefacts.

## 4.5. Equilibrium Properties In The Nvt Ensemble

The very use of a finite time step in the integration of MD trajectories causes energy fluctuations, and it is not uncommon to use advanced simulation schemes that violate energy conservation (Kuhne et al. ¨ , 2007; Mazzola & Sorella, 2017; Morrone et al., 2011; Laio & Parrinello, 2002), for instance because they use approximations that yield forces contaminated by a stochastic noise. In these cases, using a thermostat can counterbalance the energy error, and obtain stable trajectories that yield configurations and equilibrium average properties close the correct NVT ensemble despite the drift of the conserved quantity that generalizes the total energy for constant-temperature simulations.

Judging by the average temperature ⟨T⟩ (Table 2), it is relatively easy to control the non-conservative behavior using a white-noise (WN) Langevin thermostat. However, strong couplings τ (the time scale over which the thermostat inter-

THRM. TYPE *τ /fs* ⟨T⟩/K ⟨TH⟩/K ⟨TO⟩/K

WN NC 1000 51.0(6) 60.4(5) 33(1) WN NC 100 4.2(2) 5.9(3) 0.9(3) WN NC 10 0.4(1) 0.6(1) 0.1(1)

SVR NC 10 1.0(1) 36.2(8) -70(2)

PET

WN C 100 0.1(2) 0.0(2) 0.3(3) WN NC 1000 12.8(5) 11.2(7) 16.2(5) WN NC 100 1.4(2) 1.3(2) 1.6(3) WN NC 10 0.1(1) 0.0(1) 0.2(1)

SVR C 10 0.1(1) -0.4(3) 1.0(7) SVR NC 10 0.3(1) -4.4(3) 9.9(6) SVR M-8 10 0.0(1) -0.1(4) 0.1(9)

feres with atomic motion) are needed, as even at 100 fs the average temperature is significantly off the target value. We discuss how these deviations in the equilibrium temperature affect structural properties of water in Appendix G.5. The upshot is that for accurate models and strong thermostatting, the effects are small but noticeable. Furthermore, the strong Langevin thermostatting reduces the sampling efficiency, and so even with a respectable 1 ns trajectory, many simple structural averages are not fully converged.

## 4.6. Sampling Efficiency And Time-Dependent Properties

| ORB PET   |
|-----------|

Aggressive Langevin dynamics is bound to dramatically change time-dependent properties, and in particular to reduce the diffusion coefficient - so that longer trajectories are needed to collect statistically independent atomic configurations. This slow-down is apparent when looking at the velocity correlation spectra (Figure 3). In the weak coupling regime (WN, τ = 1000 fs) there is a (small) increase in diffusion coefficient relative to the reference, because of the unphysically higher temperature, while the high-frequency peaks corresponding to stretching and bending are only weakly perturbed. Stronger couplings alter the dynamics dramatically, and reduce the diffusion coefficient (and hence the efficiency in sampling slow, collective motion) by a factor of about 1.5 (τ = 100 fs) and 5 (τ = 10 fs), negating the inference speed-up of a non-conservative model - while making it impossible to accurately evaluate any timedependent property. A potential solution - applied often in similar cases, including to control the artifacts of non-invariant predictions of V (Langer et al., 2024) - is to resort to a *global* thermostat

10 0 cv v( 
) 
/ 
arb
. 

u ni ts 10 1 10 2 C, SVR, = 10 fs NC, WN, = 1 ps NC, WN, = 100 fs NC, WN, = 10 fs NC, SVR, = 10 fs
/ cm 1 10 3 10 1 10 2 10 3 50 100 200 400 Va l. fo rc e M
A
E / m e V/
Å
PET-C
PET-C, fine-tuned PET-NC
20 5 10 20 50 100 200 GPU hours
that only acts on the total kinetic energy rather than on individual particle momenta, achieving efficient temperature control without disrupting dynamics. We use the stochastic velocity rescaling (SVR) method (Bussi et al., 2007), which indeed brings the average temperature to within 1% of the target, without dramatically altering cˆvv(ω) even with a strong τ = 10 fs coupling. However, a global thermostat cannot help when non-conservative terms act differently on the various degrees of freedom. This is evident in how the temperature of O and H atoms, computed separately, deviates by up to 10% from the target, which is reflected in loss of structure as measured by g(r), and in an overestimation of the diffusion coefficient.

Some of the more sophisticated thermostats used to stabilize other types of MD approximations - such as carefully tuned generalized Langevin equations (Ceriotti et al., 2009; Morrone et al., 2011) - can also be used to enforce more aggressive temperature control with reduced dynamical disruption in conservative molecular dynamics, but they still modify the natural dynamical properties significantly, and they can fail catastrophically when used with non-conservative forces, as shown in Appendix G.6). These experiments show clearly that while it is possible to mitigate the runaway temperature increase associated with the lack of energy conservation, doing so in a way that does not disrupt structural and/or dynamical observables is highly nontrivial or even impossible.

## 4.7. Geometry Optimization

MD performed at very low temperature can be regarded as a form of geometry optimization. Our observations from MD suggest that sufficiently accurate non-conservative models should be able to reach reasonable, low-energy structures.

We restrict ourselves to optimization algorithms based only on gradients, to avoid the complication of using inconsistent energies, as discussed in Section 3.3, comparing the FIRE (Bitzek et al., 2006) algorithm - that is similar in spirit to zero-temperature MD - and LBFGS (Liu & Nocedal, 1989) - a quasi-Newton algorithm that uses an approximation of the Hessian to accelerate convergence. Comparing different models on the task of optimizing a water snapshot from a MD configuration (Figure 7) shows that inaccurate non-conservative models, such as SOAP-BPNN-NC, fail catastrophically at geometry optimization, while more accurate models, such as PET-NC and ORB, can reach a locally stable configuration, especially using FIRE. We note, however, that non-conservative models are less stable when used with a Hessian-based method, with large fluctuations in the residual force that make it hard to define a stopping criterion.

On a practical level, non-conservative forces are bound to make geometry optimization more fragile, and to require careful choice of the minimization algorithm and its convergence parameters, as we observe in Appendix F when comparing different general-purpose models.

## 4.8. Non-Conservative Forces As Accelerators

While we have observed that conservative MLIPs are better suited for practical simulations, we suggest that hybrid models, which additionally support direct, non-conservative, force predictions, can be used for faster inference and training. Such models can be obtained by training both force heads jointly, as demonstrated in the PET-M model, or, more efficiently, by first training a non-conservative model and then fine-tuning its energy head to yield conservative forces. As shown in Figure 4 and further discussed in Appendix H, conservative fine tuning leads to the accuracy and physical correctness of conservative models at highly reduced training time. In simulations, one can then use the conservative forces of such a hybrid model for validation, error monitoring and correction, and the direct forces for faster inference. A good example is to use multiple time-stepping (MTS) techniques (Tuckerman et al., 1992) for molecular dynamics, where the non-conservative forces are used to integrate the equations of motion, and the conservative forces are applied every M steps as a correction. This reduces the theoretical overhead of a conservative trajectory from a factor of F ≈ 2 to one of 1+ (F −1)/M. The results using this technique in Table 2, Figure 2 and in Appendix I are essentially indistinguishable from fully conservative ones, using M = 8, which leads to a small, approximately 20% slowdown compared to a direct-force, non-conservative trajectory. Appendix I contains further explanation of the MTS technique, as well as more detailed results for MTS simulations using models trained on the water and OC20 datasets. The technique can be also successfully used for constant-pressure simulations, as shown in Appendix G.3.

## 5. Discussion

Chemical and materials modeling is at the forefront of development in the applications of machine learning to science. The field has long been advocating for the use of physically informed architectural constraints, but there are indications that its bitter-lesson moment is coming, with the realization that deploying physics-agnostic models at scale provides better outcomes than exploiting physical priors. It appears that this is the case for some of the geometric symmetry constraints (Pozdnyakov & Ceriotti, 2023; Neumann et al., 2024; Qu & Krishnapriyan, 2024), and a growing number of architectures disregard the physical connection between the interatomic potential and the corresponding forces (Gasteiger et al., 2021; Neumann et al., 2024; Liao et al., 2023). With respect to this latter constraint, our study paints a nuanced picture. Atomistic simulations rely on the assumption that forces are the exact derivatives of the potential, and small deviations from this constraints lead to instabilities. Non-conservative behavior also results in molecular dynamics trajectories exhibiting a spontaneous drift away from the desired thermodynamic conditions. Controlling this effect with thermostats requires careful tuning, and disrupts both time-dependent properties and the sampling efficiency of the trajectory, negating the computational advantage of a direct-force architecture.

Contrary to the case of rotational symmetry that is easy to monitor and correct at inference time (Langer et al.,
2024), and learn through data augmentation, assessing nonconservative behavior requires the explicit evaluation of the Jacobian both as diagnostics and as additional loss term. Furthermore, energy and forces are complementary targets, and disregarding the former may lead to potentials that appear stable, resilient to dataset inconsistencies, and with good validation set accuracy, but are less reliable in describing the slow, collective structural rearrangements that are often the key drivers of the most relevant microscopic processes. Given that the target forces are conservative, accurate models usually exhibit less pronounced non-conservative behavior. As a consequence, one can expect that, as the field moves to larger training datasets and more expressive models, some of the pathological effects we observe will become less severe. Our findings, however, suggest that the best way to exploit the speed-up afforded by direct prediction of the forces is not to replace conservative models, but to augment them with a non-conservative head. This can be used to accelerate training by first training a non-conservative model and then fine-tuning its energy head to yield accurate conservative forces through differentiation. The resulting
"multi-force" models can also be used to speed up many different types of simulations, by alternating conservative and non-conservative evaluations, avoiding the narrower applicability and inherent instability associated with relying exclusively on direct force predictions. This insight enables the efficient training of the next generation of universal machine-learning interatomic potentials while retaining the physical correctness required for practical simulations.

## Software And Data

Code and data required to reproduce the results in this work are available on Zenodo at https://zenodo.org/records/14778891. An example of how to perform multiple-time-step dynamics with conservative and direct forces can be found at https://atomistic-cookbook.org/ examples/pet-mad-nc/pet-mad-nc.html, and an example of conservative fine-tuning at https://atomistic-cookbook.org/
examples/pet-finetuning/pet-ft-nc.html.

More details on the software used are available in Appendix J.

## Acknowledgements

The authors would like to thank Federico Grasselli and Niklas Schmitz for stimulating discussion, and Rafael Gomez-Bombarelli for a coffee-break conversation which inspired us to look into this problem. ML and MC acknowledge funding from the European Research Council (ERC)
under the European Union's Horizon 2020 research and innovation programme Grant No. 101001890-FIAMMA. FB and MC acknowledge support from the NCCR MAR-
VEL, funded by the Swiss National Science Foundation (SNSF, grant number 182892) and from the Swiss Platform for Advanced Scientific Computing (PASC).

## Impact Statement References

Kozinsky, B. E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. Nature Communications, 13(1):2453, May 2022. ISSN 20411723. doi: 10.1038/s41467-022-29939-5.

This paper presents work whose goal is to advance the application of machine learning to simulations of molecules and materials, which can be used for many purposes. As potential impacts are hard to predict and wide-ranging, we do not believe it is necessary to discuss them here in detail.

Behler, J. Four Generations of High-Dimensional Neural Network Potentials. *Chemical Reviews*, 121(16):10037– 10072, August 2021. ISSN 0009-2665, 1520-6890. doi: 10.1021/acs.chemrev.0c00868.

Behler, J. and Parrinello, M. Generalized Neural-Network Representation of High-Dimensional Potential-Energy Surfaces. *Physical Review Letters*, 98(14):146401, April 2007. ISSN 0031-9007. doi: 10.1103/PhysRevLett.98. 146401.

Allen, M. P. and Tildesley, D. J. Computer Simulation of Liquids, volume 1. Oxford University Press, November 2017. ISBN 978-0-19-880319-5. doi: 10.1093/oso/ 9780198803195.001.0001.

Andersen, H. C. Molecular dynamics simulations at constant pressure and/or temperature. The Journal of Chemical Physics, 72(4):2384–2393, 1980. ISSN 00219606. doi: 10.1063/1.439486.

Bigi, F., Chong, S., Kristiadi, A., and Ceriotti, M. Flashmd:
long-stride, universal prediction of molecular dynamics.

arXiv preprint arXiv:2505.19350, 2025.

Bitzek, E., Koskinen, P., Gahler, F., Moseler, M., and Gumb- ¨
sch, P. Structural Relaxation Made Simple. Physical Review Letters, 97(17):170201, October 2006. ISSN 00319007, 1079-7114. doi: 10.1103/PhysRevLett.97.170201.

Artrith, N., Morawietz, T., and Behler, J. High-dimensional neural-network potentials for multicomponent systems: Applications to zinc oxide. *Physical Review B*, 83(15): 153101, April 2011. ISSN 1098-0121. doi: 10.1103/ PhysRevB.83.153101.

Bussi, G. and Parrinello, M. Accurate sampling using Langevin dynamics. *Physical Review E*, 75(5):56707, 2007. doi: 10.1103/PhysRevE.75.056707.

Barroso-Luque, L., Shuaibi, M., Fu, X., Wood, B. M.,
Dzamba, M., Gao, M., Rizvi, A., Zitnick, C. L., and Ulissi, Z. W. Open materials 2024 (omat24) inorganic materials dataset and models. *arXiv preprint* arXiv:2410.12771, 2024.

Bussi, G. and Parrinello, M. Stochastic thermostats: Comparison of local and global schemes. Computer Physics Communications, 179(1-3):26–29, July 2008. ISSN 00104655. doi: 10.1016/j.cpc.2008.01.006.

Bartok, A. P., Payne, M. C., Kondor, R., and Cs ´ anyi, G. ´
Gaussian Approximation Potentials: The Accuracy of Quantum Mechanics, without the Electrons. *Physical* Review Letters, 104(13):136403, April 2010. ISSN 00319007. doi: 10.1103/PhysRevLett.104.136403.

Bussi, G., Donadio, D., and Parrinello, M. Canonical sampling through velocity rescaling. Journal of Chemical Physics, 126(1):14101, 2007.

Bussi, G., Zykova-Timan, T., and Parrinello, M. Isothermalisobaric molecular dynamics using stochastic velocity rescaling. *The Journal of Chemical Physics*, 130(7): 074101, February 2009. ISSN 0021-9606. doi: 10.1063/ 1.3073889.

Bartok, A. P., Kondor, R., and Cs ´ anyi, G. On repre- ´
senting chemical environments. *Physical Review B*, 87(18):184115, May 2013. ISSN 1098-0121. doi: 10.1103/PhysRevB.87.184115.

Batatia, I., Kovacs, D. P., Simm, G. N. C., Ortner, C., and Csanyi, G. MACE: Higher order equivariant message passing neural networks for fast and accurate force fields. In Oh, A. H., Agarwal, A., Belgrave, D., and Cho, K. (eds.), Advances in Neural Information Processing Systems, 2022.

Ceriotti, M., Bussi, G., and Parrinello, M. Langevin equation with colored noise for constant-temperature molecular dynamics simulations. *Physical Review Letters*, 102(2):020601, January 2009. ISSN 00319007. doi: 10.1103/PhysRevLett.102.020601.

Ceriotti, M., Bussi, G., and Parrinello, M. Colored-noise thermostats a la Carte. ` *Journal of Chemical Theory* and Computation, 6(4):1170–1180, April 2010. ISSN
15499626. doi: 10.1021/ct900563s.

Batatia, I., Benner, P., Chiang, Y., Elena, A. M., Kovacs, ´
D. P., Riebesell, J., Advincula, X. R., Asta, M., Avaylon, M., Baldwin, W. J., et al. A foundation model for atomistic materials chemistry. *arXiv preprint* arXiv:2401.00096, 2023.

Chandler, D. and Wolynes, P. G. Exploiting the isomorphism between quantum theory and classical statistical mechanics of polyatomic fluids. Journal of Chemical Physics, 74(7):4078–4095, 1981.

Batzner, S., Musaelian, A., Sun, L., Geiger, M., Mailoa, J. P., Kornbluth, M., Molinari, N., Smidt, T. E., and Chanussot, L., Das, A., Goyal, S., Lavril, T., Shuaibi, M.,
Riviere, M., Tran, K., Heras-Domingo, J., Ho, C., Hu, W., et al. Open catalyst 2020 (oc20) dataset and community challenges. *Acs Catalysis*, 11(10):6059–6072, 2021.

Cheng, B., Engel, E. A., Behler, J., Dellago, C., and Ceriotti, M. Ab initio thermodynamics of liquid and solid water. Proceedings of the National Academy of Sciences of the United States of America, 116(4):1110–1115, January 2019. ISSN 10916490. doi: 10.1073/pnas.1815117116.

Chmiela, S., Sauceda, H. E., Muller, K.-R., and Tkatchenko, ¨
A. Towards exact molecular dynamics simulations with machine-learned force fields. *Nature Communications*, 9:
3887, 2018. doi: 10.1038/s41467-018-06169-2.

Christensen, A. S. and von Lilienfeld, O. A. On the role of gradients for machine learning of molecular energies and forces. *Machine Learning: Science and Technology*, 1(4): 045018, 2020. doi: 10.1088/2632-2153/abba6f.

Deng, B., Zhong, P., Jun, K., Riebesell, J., Han, K., Bartel, C. J., and Ceder, G. Chgnet as a pretrained universal neural network potential for charge-informed atomistic modelling. *Nature Machine Intelligence*, 5(9):1031–1041, 2023.

Duval, A., Mathis, S. V., Joshi, C. K., Schmidt, V., Miret, S., Malliaros, F. D., Cohen, T., Lio, P., Bengio, Y., and Bronstein, M. A hitchhiker's guide to geometric gnns for 3d atomic systems. *arXiv preprint arXiv:2312.07511*, 2023.

Eastman, P., Behara, P. K., Dotson, D. L., Galvelis, R., Herr, J. E., Horton, J. T., Mao, Y., Chodera, J. D., Pritchard, B. P., Wang, Y., De Fabritiis, G., and Markland, T. E. SPICE, A Dataset of Drug-like Molecules and Peptides for Training Machine Learning Potentials. *Scientific Data*, 10(1):11, January 2023. ISSN 2052-4463. doi: 10.1038/ s41597-022-01882-6.

Eissler, M., Korjakow, T., Ganscha, S., Unke, O. T., Muller, ¨
K.-R., and Gugler, S. How simple can you go? an off-theshelf transformer approach to molecular dynamics, 2025. URL https://arxiv.org/abs/2503.01431.

Fu, X., Wood, B. M., Barroso-Luque, L., Levine, D. S., Gao, M., Dzamba, M., and Zitnick, C. L. Learning smooth and expressive interatomic potentials for physical property prediction, 2025. URL https://arxiv.org/abs/ 2502.12147.

Gasteiger, J., Becker, F., and Gunnemann, S. Gemnet: Uni- ¨
versal directional graph neural networks for molecules. Advances in Neural Information Processing Systems, 34:
6790–6802, 2021.

Griewank, A. and Walther, A. *Evaluating Derivatives*.

Society for Industrial and Applied Mathematics, January 2008. ISBN 9780898716597, 9780898717761. doi: 10.1137/1.9780898717761. URL https://doi. org/10.1137/1.9780898717761.

Hairer, E., Hochbruck, M., Iserles, A., and Lubich, C. Geometric numerical integration. *Oberwolfach Reports*, 3(1): 805–882, 2006.

Henkelman, G., Uberuaga, B. P., and Jonsson, H. A climb- ´
ing image nudged elastic band method for finding saddle points and minimum energy paths. *Journal of Chemical* Physics, 113(22):9901, 2000.

Herbold, M. and Behler, J. A Hessian-based assessment of atomic forces for training machine learning interatomic potentials. *The Journal of Chemical Physics*, 156(11): 114106, March 2022. ISSN 0021-9606, 1089-7690. doi: 10.1063/5.0082952.

Hjorth Larsen, A., Jørgen Mortensen, J., Blomqvist, J.,
Castelli, I. E., Christensen, R., Dułak, M., Friis, J., Groves, M. N., Hammer, B., Hargus, C., Hermes, E. D., Jennings, P. C., Bjerre Jensen, P., Kermode, J., Kitchin, J. R., Leonhard Kolsbjerg, E., Kubal, J., Kaasbjerg, K., Lysgaard, S., Bergmann Maronsson, J., Maxson, T., Olsen, T., Pastewka, L., Peterson, A., Rostgaard, C.,
Schiøtz, J., Schutt, O., Strange, M., Thygesen, K. S., ¨ Vegge, T., Vilhelmsen, L., Walter, M., Zeng, Z., and Jacobsen, K. W. The atomic simulation environmenta Python library for working with atoms. *Journal of* Physics: Condensed Matter, 29(27):273002, July 2017. ISSN 0953-8984, 1361-648X. doi: 10.1088/1361-648X/ aa680e.

Kapil, V., VandeVondele, J., and Ceriotti, M. Accurate molecular dynamics and nuclear quantum effects at low cost by multiple steps in real and imaginary time: Using density functional theory to accelerate wavefunction methods. *Journal of Chemical Physics*, 144 (5):054111, February 2016. ISSN 00219606. doi: 10.1063/1.4941091.

Kuhne, T. D., Krack, M., Mohamed, F. R., and Parrinello, ¨
M. Efficient and Accurate Car-Parrinello-like Approach to Born-Oppenheimer Molecular Dynamics. Physical Review Letters, 98(6):66401, February 2007. ISSN 00319007. doi: 10.1103/PhysRevLett.98.066401.

Laio, A. and Parrinello, M. Escaping free-energy minima.

Proceedings of the National Academy of Sciences, 99(20):
12562–12566, 2002.

Langer, M. F., Goeßmann, A., and Rupp, M. Representations of molecules and materials for interpolation of quantum-mechanical simulations via machine learning.

npj Computational Materials, 8(1):41, December 2022. ISSN 2057-3960. doi: 10.1038/s41524-022-00721-x.

Langer, M. F., Pozdnyakov, S. N., and Ceriotti, M. Probing the effects of broken symmetries in machine learning. *Machine Learning: Science and Technology*, 5 (4):04LT01, December 2024. ISSN 2632-2153. doi: 10.1088/2632-2153/ad86a0.

Li, Z., Kermode, J. R., and De Vita, A. Molecular dynamics with on-the-fly machine learning of quantum-mechanical forces. *Physical Review Letters*, 114(9):096405, March 2015. ISSN 10797114. doi: 10.1103/PhysRevLett.114. 096405.

Liao, Y.-L., Wood, B., Das, A., and Smidt, T. Equiformerv2:
Improved equivariant transformer for scaling to higherdegree representations. *arXiv preprint arXiv:2306.12059*, 2023.

Litman, Y., Kapil, V., Feldman, Y. M. Y., Tisi, D., Begusiˇ c,´
T., Fidanyan, K., Fraux, G., Higer, J., Kellner, M., Li, T. E., Pos, E. S., Stocco, E., Trenins, G., Hirshberg, B., ´ Rossi, M., and Ceriotti, M. I-PI 3.0: A flexible and efficient framework for advanced atomistic simulations. The Journal of Chemical Physics, 161(6):062504, August 2024. ISSN 0021-9606, 1089-7690. doi: 10.1063/5. 0215869.

Liu, D. C. and Nocedal, J. On the limited memory BFGS
method for large scale optimization. Mathematical Programming, 45(1-3):503–528, August 1989. ISSN 00255610, 1436-4646. doi: 10.1007/BF01589116.

Loew, A., Sun, D., Wang, H.-C., Botti, S., and Marques, M.

A. L. Universal machine learning interatomic potentials are ready for phonons, 2024. URL https://arxiv. org/abs/2412.16551.

Marzari, N., Vanderbilt, D., De Vita, A., and Payne, M. C. Thermal Contraction and Disordering of the Al(110) Surface. *Physical Review Letters*, 82(16):3296– 3299, April 1999. ISSN 0031-9007, 1079-7114. doi: 10.1103/PhysRevLett.82.3296.

Mazitov, A., Bigi, F., Kellner, M., Pegolo, P., Tisi, D., Fraux, G., Pozdnyakov, S., Loche, P., and Ceriotti, M. PET- MAD, a universal interatomic potential for advanced materials modeling. *arXiv preprint arXiv:2503.14118*, 2025.

Mazzola, G. and Sorella, S. Accelerating *ab initio* Molecular Dynamics and Probing the Weak Dispersive Forces in Dense Liquid Hydrogen. *Physical Review Letters*, 118
(1):015703, January 2017. ISSN 0031-9007, 1079-7114.

doi: 10.1103/PhysRevLett.118.015703.

Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N.,
Teller, A. H., and Teller, E. Equation of State Calculations by Fast Computing Machines. Journal of Chemical Physics, 21(6):1087–1092, 1953.

Morrone, J. A., Markland, T. E., Ceriotti, M., and Berne, B. J. Efficient multiple time scale molecular dynamics: Using colored noise thermostats to stabilize resonances. *Journal of Chemical Physics*, 134(1):14103, January 2011. ISSN 00219606. doi: 10.1063/1.3518369.

Musil, F., Grisafi, A., Bartok, A. P., Ortner, C., Cs ´ anyi, G., ´
and Ceriotti, M. Physics-Inspired Structural Representations for Molecules and Materials. *Chemical Reviews*, 121 (16):9759–9815, August 2021. ISSN 0009-2665, 15206890. doi: 10.1021/acs.chemrev.1c00021.

Musil, F., Zaporozhets, I., Noe, F., Clementi, C., and ´
Kapil, V. Quantum dynamics using path integral coarsegraining. *The Journal of Chemical Physics*, 157(18): 181102, November 2022. ISSN 0021-9606, 1089-7690. doi: 10.1063/5.0120386.

Neumann, M., Gin, J., Rhodes, B., Bennett, S., Li, Z.,
Choubisa, H., Hussey, A., and Godwin, J. Orb: A
fast, scalable neural network potential. arXiv preprint arXiv:2410.22570, 2024.

Nigam, J., Pozdnyakov, S., Fraux, G., and Ceriotti, M. Unified theory of atom-centered representations and messagepassing machine-learning schemes. The Journal of Chemical Physics, 156(20):204115, May 2022. ISSN 00219606, 1089-7690. doi: 10.1063/5.0087042.

Park, Y., Kim, J., Hwang, S., and Han, S. Scalable parallel algorithm for graph neural network interatomic potentials in molecular dynamics simulations. *Journal of Chemical* Theory and Computation, 2024.

Pathria, R. K. Statistical Mechanics: International Series of Monographs in Natural Philosophy, volume 45. Elsevier, 2017.

Pozdnyakov, S. and Ceriotti, M. Smooth, exact rotational symmetrization for deep learning on point clouds. In Advances in Neural Information Processing Systems, volume 36, pp. 79469–79501. Curran Associates, Inc., 2023.

Pozdnyakov, S. N., Willatt, M. J., Bartok, A. P., Ortner, C., ´
Csanyi, G., and Ceriotti, M. Incompleteness of Atomic ´ Structure Representations. *Physical Review Letters*, 125: 166001, 2020. doi: 10.1103/PhysRevLett.125.166001.

Qu, E. and Krishnapriyan, A. S. The importance of being scalable: Improving the speed and accuracy of neural network interatomic potentials across chemical domains. arXiv preprint arXiv:2410.24169, 2024.

Quiroga, F., Ronchetti, F., Lanzarini, L., and Bariviera, A. F.

Revisiting data augmentation for rotational invariance in convolutional neural networks. In Modelling and Simulation in Management Sciences: Proceedings of the International Conference on Modelling and Simulation in Management Sciences (MS-18), pp. 127–141. Springer, 2020.

Ramakrishnan, R., Dral, P. O., Rupp, M., and Von Lilienfeld, O. A. Quantum chemistry structures and properties of 134 kilo molecules. *Scientific data*, 1(1):1–7, 2014.

Ren, Y., Zheng, D., Liu, C., Jin, P., Shi, Y., Huang, L.,
He, J., Luo, S., Qin, T., and Liu, T.-Y. Physical consistency bridges heterogeneous data in molecular multitask learning. In Advances in Neural Information Processing Systems 38 (NeurIPS 2024), Vancouver, Canada, Dec 10–Dec 15, pp. not available. not available, 2024. doi:
notavailable. URL https://openreview.net/ forum?id=GnF9tavqgc.

Schmidt, J., Cerqueira, T. F., Romero, A. H., Loew, A.,
Jager, F., Wang, H.-C., Botti, S., and Marques, M. A. ¨ Improving machine-learning models in materials science through large datasets. *Materials Today Physics*, 48: 101560, 2024.

Tran, R., Lan, J., Shuaibi, M., Wood, B. M., Goyal, S., Das, A., Heras-Domingo, J., Kolluru, A., Rizvi, A., Shoghi, N., et al. The open catalyst 2022 (oc22) dataset and challenges for oxide electrocatalysts. *ACS Catalysis*, 13 (5):3066–3084, 2023.

Tuckerman, M. Statistical Mechanics and Molecular Simulations. Oxford University Press, 2008.

Tuckerman, M., Berne, B. J., and Martyna, G. J. Reversible multiple time scale molecular dynamics. The Journal of Chemical Physics, 97(3):1990, 1992. ISSN 00219606.

doi: 10.1063/1.463137.

Unke, O. T., Chmiela, S., Sauceda, H. E., Gastegger, M.,
Poltavsky, I., Schutt, K. T., Tkatchenko, A., and M ¨ uller, ¨ K.-R. Machine learning force fields. *Chemical Reviews*, 121(16):10142–10186, 2021. doi: 10.1021/acs.chemrev. 0c01111.

Verlet, L. Computer "Experiments" on Classical Fluids. I. Thermodynamical Properties of Lennard-Jones Molecules. *Physical Review*, 159(1):98–103, July 1967. ISSN 0031-899X. doi: 10.1103/PhysRev.159.98.

Villar, S., Hogg, D. W., Storey-Fisher, K., Yao, W., and Blum-Smith, B. Scalars are universal: Equivariant machine learning, structured like classical physics. Advances in Neural Information Processing Systems, 34:28848–
28863, 2021. doi: 10.48550/arXiv.2106.06610.

Wang, J., Olsson, S., Wehmeyer, C., Perez, A., Charron, ´
N. E., De Fabritiis, G., Noe, F., and Clementi, C. Machine ´ Learning of Coarse-Grained Molecular Dynamics Force Fields. *ACS Central Science*, 5(5):755–767, May 2019. ISSN 2374-7943, 2374-7951. doi: 10.1021/acscentsci. 8b00913.

Wang, Y., Wang, L., Shen, Y., Wang, Y., Yuan, H.,
Wu, Y., and Gu, Q. Protein conformation generation via force-guided SE(3) diffusion models. In Proceedings of the 41st International Conference on Machine Learning, pp. 56835–56859. PMLR, 2024. doi:
notavailable. URL https://proceedings.mlr. press/v235/wang24cv.html.

Zhou, J., Cui, G., Hu, S., Zhang, Z., Yang, C., Liu, Z., Wang, L., Li, C., and Sun, M. Graph neural networks: A review of methods and applications. *AI open*, 1:57–81, 2020.

## A. Forces And Potentials As Training Targets

Including both forces and the potential energy V as targets when training a ML potential (either separately, or jointly for a conservative model) requires choosing a weighting factor to combine the errors into a single loss function. Independently from the relative weight, it is interesting to consider how these two targets affect the accuracy of a model for different types of molecular displacements. To investigate this aspect, we can approximate a material in the vicinity of a stable structure, i.e., a local minimum of the potential energy V , as a quadratic form, which can be written in the basis of the eigenvectors of the (mass-scaled) Hessian to give a superimposition of harmonic modes,

$$V({\bf q})=\sum_{k}V_{k}(q_{k})=\frac{1}{2}\sum_{k}m\omega_{k}^{2}q_{k}^{2},$$
$$(8)$$

$$(9)$$

k, (8)
where m is the atomic mass (the expression generalizes to the case of multiple atomic types), ωk are normal mode frequencies and qk the displacements along the eigenvectors.

If we now consider how configurations are sampled at a constant temperature T (running short MD trajectories is a common strategy to generate training sets for MLIPs), one sees that each harmonic mode is distributed as p(qk) ∝
exp(−Vk(qk)/kBT) = exp(−mω2k q 2 k
/2kBT). As a consequence, one can easily compute the expectation values

$$\langle q_{k}^{2}\rangle\propto\frac{k_{\mathrm{B}}T}{\omega_{k}^{2}},\quad\langle V_{k}(q_{k})\rangle\propto k_{\mathrm{B}}T,\quad\langle f_{k}^{2}\rangle\propto k_{\mathrm{B}}T\omega_{k}^{2}$$
⟩ ∝ kBT ω2k(9)
These textbook results highlight the following facts: (1) low-frequency normal modes are those associated with the largest structural deformations - and therefore, often, with phase transitions and important molecular rearrangements; (2) thermal excitations affect all normal modes equally in terms of potential energy contributions; (3) the largest force contributions come from high-frequency (low-displacement) vibrations.

Thus, when training on forces using an L
2loss, molecular modes associated with high-frequency vibrations are overemphasized. For example, if one considers the water dataset we use in this work (Cheng et al., 2019), the total force acting on each water molecule has a root mean square of 0.97 eV/A, while the residual "intra-molecular" forces (that are ˚ predominantly short-ranged and associated with high-frequency molecular vibrations) are four times larger, 3.93 eV/A. This ˚
very crude analysis highlights the non-trivial implications of using forces as (direct or indirect) training targets.

## B. Theoretical Computational Cost

Conservative forces are generally computed from potential energies by backward propagation of gradients. The vast majority of operations in neural networks (and nearly all those that take up significant computational time) are binary operations which can be expressed as the computation of f(*x, y*) starting from x and y. During the backward step corresponding to such operation, *∂V /∂x* and *∂V /∂y* must be found from *∂V /∂f*. In the case of matrix multiplication, the forward calculation of f(*x, y*), the backward calculation of *∂V /∂x* and the backward calculation of *∂V /∂y* each consist of a matrix multiplication with the same computational cost. Since matrix multiplications can be assumed to be the most costly components of neural networks, this generally implies that backward gradient computations are around twice as expensive as the corresponding forward function evaluation. However, in the case of backward force evaluation, operations where x is an internal representation and y is a weight can save the *∂V /∂y* calculation. In a simple multi-layer perceptron, where all linear layers correspond to this type of computation, this would yield a backward propagation that is roughly as expensive as the forward pass. This is not the case in transformers, as the attention mechanism involves comparatively expensive operations where both x and y are internal representations, and one can expect the backward propagation of gradients to be somewhere between 1× and 2× the cost of the forward evaluation.

## C. Range Of Back-Propagated And Direct Force Models

The use of a cutoff to restrict the range of interactions is ubiquitous in the construction of physics-based potentials, and is also an integral part of descriptor-based ML potentials (Musil et al., 2021). It is often argued that models that incorporate correlations between at least two neighbors of each central atom achieve an effective interaction range of twice the cutoff distance (Artrith et al., 2011) (see Figure 5a). A similar effect also applies to message-passing architectures (Nigam et al.,
2022).

One can see clearly that this extension of the interaction range beyond the cutoff (or the receptive radius of the GNN) does not apply to the case in which forces are predicted directly. Considering for simplicity the case of a three-body model in which the potential contribution from each atomic environment i can be written as a sum of a function of its distances with two neighbors j and k, and the distance between the neighbors rjk

$$V_{i}=\sum_{j,k}v(r_{ij},r_{ik},r_{jk}),\tag{10}$$

one sees that the dependency on interatomic distances greater than the cutoff is due to the relative position of atoms other than the central atom. Thus, a direct model that predicts a similar 3-body force

$$\mathbf{f}_{i}=\sum_{j,k}\mathbf{f}(r_{i j},r_{i k},r_{j k}),$$
$$(11)$$

f(rij , rik, rjk), (11)
or any other functional form limited to the neighbors of the i-th atom, only contains information on atoms within the cutoff.

The dependency of fi on the coordinates of far-away atoms is a consequence of the fact that the total energy is built as a sum over multiple centers. It is only through the terms of the form ∂Vj/∂rithat occur naturally when evaluating forces through back-propagation, than the force depends on the position of the neighbor's neighbors. More generally, in a message-passing implementation, backpropagation ensures that force predictions benefit from an effective receptive radius that is twice that of atom-centered energy predictions. A further concern for direct force models is that - at least for low-body-order models - atom-centered descriptors can be shown to have low resolving power, with pairs of distinct atomic environments having precisely the same representation (Pozdnyakov et al., 2020). For all known degeneracies, descriptors centered on other atoms allows distinguishing the structure *as a whole*, and as a consequence the total energy and interatomic forces can be still differentiated. This would not be the case for direct force predictions, which would fail completely to differentiate degenerate pairs when using atom-centered low-body-order models, and would be more sensitive to numerical instabilities for higher-order models. In practice, we find consistent evidence of the practical impact of these considerations. As an example, Table 3 reports the accuracy of PET-C and PET-NC models, using 2 and 3 message passing layers. It can be seen that the accuracy of the direct force model benefits much more from the increase in the receptive radius.

| Model   | 2 message-passing layers   | 3 message-passing layers   |
|---------|----------------------------|----------------------------|
| PET-C   | 20.8                       | 18.6                       |
| PET-NC  | 32.8                       | 24.3                       |

## D. Models And Architectures

Even though the main text is focused on custom-trained models, and emphasize the most accurate non-conservative model we could obtain, we also want to provide an overview of the behavior of a less-performant custom-trained model, and of several publicly available general-purpose models. A comprehensive list of the models we consider is reported in Table 4.

| Model        | Description                                                                                      |
|--------------|--------------------------------------------------------------------------------------------------|
| ORB-v2       | Non-equivariant, non-conservative model, trained on the Alexandria and MPtrj datasets.           |
| Equiformer   | Equivariant, non-conservative model, trained on the Alexandria and MPtrj datasets.               |
| MACE-MP-0    | Equivariant, conservative model, trained on the MPtrj dataset.                                   |
| SevenNet     | Equivariant, conservative model, trained on the MPtrj dataset.                                   |
| PET-C        | A re-implementation of the PET architecture, trained on the bulk water dataset.                  |
| PET-NC       | A modified PET architecture, trained on the bulk water dataset.                                  |
| SOAP-BPNN-C  | A SOAP-BPNN architecture, trained on the bulk water dataset.                                     |
| SOAP-BPNN-NC | A modified SOAP-BPNN architecture, trained on the bulk water dataset to directly predict forces. |

Table 4. Models used in the present work.

It should be noted that:
- The ORB model (orb-v2) is more accurate than MACE-MP-0 and SevenNet, as the former is pre-trained on the Alexandria (Schmidt et al., 2024) dataset and then fine-tuned on MPtrj, while the latter two are only trained on MPtrj. Despite its higher accuracy, ORB yields problematic physical behavior as discussed in this work.

- The PET-NC and SOAP-BPNN-NC models are simply obtained from the respective conservative models by changing the output head to predict atomic forces directly. In the case of SOAP-BPNN-NC, an equivariant vector representation is generated internally thanks to the formalism in Villar et al. (2021). In both cases, due to the marginal increase in number of parameters in the force head, the non-conservative models have slightly more parameters than their conservative counterparts. Within the calculators for these two non-conservative models, we implemented the net force removal suggested in Neumann et al. (2024).

The architectures of these models are further described here:
- ORB: A rotationally unconstrained and non-conservative architecture, presented in Neumann et al. (2024). - MACE: A rotationally invariant and conservative architecture, presented in Batatia et al. (2022). - SevenNet: The SevenNet model (Park et al., 2024) makes use of the NequIP (Batzner et al., 2022) architecture, which is rotationally invariant and conservative.

- PET-C and PET-NC: A re-implementation of the architecture in Pozdnyakov & Ceriotti (2023), which is rotationally unconstrained and conservative. The non-conservative version changes the final head to predict forces instead of energies.

- SOAP-BPNN-C and SOAP-BPNN-NC: A Behler-Parrinello neural network architecture (Behler & Parrinello, 2007),
using SOAP (Bartok et al. ´ , 2013) descriptors. This architecture is rotationally invariant and conservative. The nonconservative version makes use of the formalism in Villar et al. (2021) to predict forces (a vector) from a scalar internal representation.

## D.1. Timings Of General-Purpose Models

Table 5 shows the timings for the four general-purpose models tested in this work, compared with the PET models we train here. The large version of MACE-MP-0 was used in this table and throughout this work. The small version of EquiformerV2 was used in this table and throughout this work, except to calculate work loops, where the large version was used.

| MODEL           | TIMING PER STEP   | TIMING PER STEP PER ATOM   |
|-----------------|-------------------|----------------------------|
| MACE (C)        | 26.9              | 0.140                      |
| SEVENNET (C)    | 52.8              | 0.275                      |
| ORB (NC)        | 11.9              | 0.062                      |
| EQUIFORMER (NC) | 1580              | 8.230                      |
| PET (C)         | 19.4              | 0.101                      |
| PET (NC)        | 8.58              | 0.047                      |

| ARCHITECTURE   | TYPE   | TRAINING   | MAE(V )   | MAE(f)   | TIMING (TR.)   | TIMING (EV.)   |
|----------------|--------|------------|-----------|----------|----------------|----------------|
| PET            | -      | V          | 4.7       | 1025.6*  | 5.48           | 0.0264         |
| PET            | C      | f          | 1.26**    | 18.6     | 15.30          | 0.0713         |
| PET            | C      | V,f        | 0.55      | 19.4     | 15.31          | 0.0716         |
| PET            | NC     | f          | -         | 24.3     | 5.55           | 0.0224         |
| PET            | NC     | V,f        | 1.42      | 24.8     | 5.63           | 0.0269         |
| PET-M          | C      | V,f        | 0.59      | 20.2     | 15.43          | 0.0715         |
| NC             | 26.7   | 0.0265     |           |          |                |                |
| PET-M-FT***    | C      | V,f        | 0.50      | 20.0     | 56.42***       | 0.0714         |
| SOAP-BPNN      | -      | V          | 2.16      | 177.0*   | 3.57           | 0.1065         |
| SOAP-BPNN      | C      | f          | 1.89**    | 40.6     | 36.10          | 0.6394         |
| SOAP-BPNN      | C      | V,f        | 1.38      | 41.4     | 36.20          | 0.6367         |
| SOAP-BPNN      | NC     | f          | -         | 112.2    | 5.81           | 0.1515         |
| SOAP-BPNN      | NC     | V,f        | 3.20      | 111.9    | 6.34           | 0.1674         |

Table 6. Test errors (energies in meV per atom, forces in meV/A), training and evaluation timings for models trained on the bulk water ˚
dataset. Training timings correspond to the time to compute a single epoch (in seconds) on 4 H100 GPUs with a total batch size of 64. Evaluation timings correspond to the average time per atom (in ms) for energy and/or force evaluations for single structures across the test set. *The force errors of energy-only models are computed by evaluating forces as derivatives of the energies, despite the fact that no explicit force training took place. **A linear fit was executed to minimize training errors on energies for the force-only models, in order to calculate the best constant shift for the fictitious energy of which the forces are the derivatives. ***Trained on a single GPU as opposed to four, and for a single day as opposed to two.

## D.2. Accuracy And Timings Of Water Models

Similarly to Table 1, we evaluate the accuracy of the SOAP-BPNN architecture on the same bulk water dataset under different training conditions. The results in Table 6 show, once again, that the lack of energy conservation without a corresponding data augmentation strategy seems to hurt the accuracy of the models. In general, although the non-conservative models trained in this work on the bulk water dataset (with little more than 250000 targets) seem to show worse accuracy, training on larger datasets has shown that non-conservative models can be competitive in accuracy with conservative models. This is not only because energy conservation can then be effectively learned, but also because non-conservative models, by virtue of being faster, can train for a larger number of epochs at a given computational budget. Training duration can be the limiting factor to accuracy on large datasets. The timings of the PET models are fully consistent with the theoretical cost analysis in Appendix B. In contrast, the SOAP-BPNN models rely on the SOAP atomic descriptors as implemented in https://github.com/metatensor/featomic. Within this library for atomistic descriptors, three implementation details account for the SOAP-BPNN timings: 1) although the models are trained and evaluated on GPU, the feature calculation is executed on CPU; 2) feature calculation is parallelized across different structures, but not different chemical environments within the same structure (effectively meaning that no evaluation-time parallelization is present); 3) the equivariant calculation of forces makes it necessary to evaluate additional features with an angular momentum quantum number of 1.

## E. Non-Conservative Work

Since non-conservative models do not obey equation (5), we evaluate indicative magnitudes of the work over a closed loop for the non-conservative models considered in this study. These are shown in Figure 6. The closed path corresponds to the rotation and deformation of a single water molecule within a liquid water structure, while keeping all the other atoms fixed. Figure 6 shows the cumulative work of the models considered in this study. Even though the cumulative work curves are very similar, the conservative models results in zero overall work on the closed path; meanwhile, the non-conservative models leads to an overall non-zero work along the path.

Figure 6. Cumulative work along a closed path for all models considered in the study. While the total work for the conservative models is zero up to machine precision, the non-conservative models exhibit a non-zero total work (ORB: 15 meV, Equiformer: -241 meV, PET-NC: 132 meV, SOAP-BPNN-NC: -410 meV). The first figure from the left shows the overall path, while the other two zoom in on the last part.

0.10 40 30 20 10 0 0.4 0.2 0.0 0.2 0.4 PET-NC
SOAP-NC ORB
Equiformer PET-C SOAP-C SevenNet MACE-MP0 C
u m ul at iv e w ork / eV
0.05 0.00 0.05 990 995 1000 Structure number 0.10 0 500 1000 Structure number 980 990 1000 Structure number

## F. Geometry Optimization F.1. Optimization Trajectories

We compare the behavior of different models when quenching a liquid-water snapshot, optimizing it towards the nearest potential energy minimum, (Figure 7) showing both the convergence in terms of the force modulus, and the trajectory of configurations as a latent-space projection built on geometric descriptors. The latent-space plots are obtained by computing SOAP descriptors for all configurations in all the trajectories, averaged over atomic centers in each structures, and are projected on the axes of highest variability using simple Principal Component Analysis (PCA). The qualitative features of the latent space plots are insensitive to the details of the SOAP descriptors used.

10 1 10 2 10 3 step 10 4 10 3 10 2 101 10 0 10 1 SOAP-BPNN-NC
SOAP-BPNN
ORB
PET-NC
PET
|f| 
/ 
e V/Å
PCA[1]
PCA
[2
]

We discuss first the results for the FIRE algorithm (left panel in Figure 7). Despite the very different level of accuracy, the conservative SOAP-BPNN model and PET converge to a similar configuration (with PET forces saturating at about 10−3eV/A, due to numerical precision). The non-conservative PET-NC model also converges to a similar structure, ˚
indicating that a sufficiently accurate non-conservative model can indeed be used with a gradient-based structural optimization algorithm. The non-conservative SOAP-BPNN model, however, displays a catastrophic mode of failure, with the force never decreasing below 0.1 eV/A, and the trajectory drifting off in a different direction without ever reaching a stable state. ORB ˚
converges to a very different structure than the other models, which might be due to the different reference DFT functional used for training. The BFGS optimization trajectories (right panel in Figure 7) are very similar to those obtained with FIRE, except for SOAP-BPNN-NC whose catastrophic failure is apparen in the trajectory going in a completely different direction, and in the force modulus never converging below 0.1 eV/A. Thanks to the second-order nature of LBFGS, convergence is much faster, ˚
until PET-based models reach their precision limit. ORB shows an interesting behavior, as it reaches a similar configuration as with FIRE, but the force modulus exhibits large fluctuations, and it would be hard to determine a clear threshold to establish convergence. On a practical level, this experiment shows that non-conservative forces make geometry optimization more fragile - with first-order methods being somewhat more stable although slower - and require careful choice of the minimization algorithm and its convergence parameters.

## F.2. Failure Rates

In Table 7, geometry optimization is attempted with a range of conservative and non-conservative models. For each model, three cases are considered: 1) geometry optimization of gas-phase water molecules, starting from the experimental geometry, randomly displacing the coordinates with a standard deviation of 0.5 A, and relaxing the geometry; 2) geometry optimization ˚
of bulk water structures from the test set of (Cheng et al., 2019), 3) geometry optimization of molecules chosen at random from the QM9 dataset (Ramakrishnan et al., 2014), randomly displacing the coordinates with a standard deviation of 0.5 A before relaxing the structures. Geometry optimization is performed with the L-BFGS ( ˚ Liu & Nocedal, 1989) algorithm as implemented in ASE (Hjorth Larsen et al., 2017). Optimization runs that do not converge within 1000 optimization steps are considered as failed.

| MODEL                   | H2O(g)   | H2O(l)   | QM9   | MPTRJ   |
|-------------------------|----------|----------|-------|---------|
| ORB-LOW-PRECISION* (NC) | 3        | 0        | 0     | 10      |
| ORB (NC)                | 69       | 9        | 1     | 76      |
| SEVENNET (C)            | 81       | 88       | 92    | 97      |
| MACE (C)                | 94       | 83       | 94    | 99      |
| PET-NC (NC)             | 75       | 52       | -     | -       |
| PET-C (C)               | 83       | 58       | -     | -       |
| SOAP-BPNN-NC (NC)       | 79       | 0        | -     | -       |
| SOAP-BPNN-C (C)         | 91       | 59       | -     | -       |

*This is an ORB model used with its default settings, which lower the precision of matrix multiplications. Given the results here, we deactivated this setting for all ORB results shown in the rest of this work.

Non-conservative models consistently show lower rates of success in geometry optimization. It should be noted that strict convergence criteria were used (fmax=1e-5˚A for molecules and fmax=1e-4˚A for bulk systems).